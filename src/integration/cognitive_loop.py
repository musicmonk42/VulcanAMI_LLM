from __future__ import annotations

import asyncio
import logging
import math
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)

Token = Union[int, str]
Tokens = List[Token]
SpeculativeFunction = Callable[
    [Any, Any, Tokens, float, int], Tuple[Tokens, Optional[Token], Dict[str, Any]]
]

# =========================== CONFIG STRUCTS =========================== #


@dataclass
class LoopSamplingConfig:
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 128
    min_tokens: int = 1
    stop_tokens: Tuple[Token, ...] = field(default_factory=lambda: ("</s>",))
    stop_strings: Tuple[str, ...] = field(default_factory=lambda: ("\n\n",))
    allow_repetition: bool = False
    repetition_window: int = 50
    repetition_penalty: float = 1.1
    adaptive_temperature: bool = True
    temperature_floor: float = 0.3
    temperature_decay: float = 0.98
    dynamic_top_k: bool = True
    min_top_k: int = 10
    use_beam_search: bool = False
    beam_width: int = 4
    beam_length_penalty: float = 0.7
    beam_max_expansions: int = 8
    speculative_enabled: bool = False
    speculative_draft_steps: int = 4
    speculative_max_rejects: int = 10


@dataclass
class LoopRuntimeConfig:
    enable_stream: bool = True
    enable_audit: bool = True
    enable_observability: bool = True
    safety_per_token: bool = True
    safety_per_sequence: bool = True
    max_errors: int = 5
    error_backoff_seconds: float = 0.05
    trace_reasoning: bool = True
    attach_logits: bool = False
    attach_probabilities: bool = False
    attach_world_model_hooks: bool = True
    time_budget_seconds: Optional[float] = None
    require_consensus_per_token: bool = False
    consensus_timeout_seconds: float = 2.0
    parallel_score_candidates: bool = False
    max_parallel_workers: int = 4
    enable_rerank: bool = False
    top_n_rerank: int = 10
    enable_strategy_bandit_feedback: bool = True
    early_stop_score_delta: Optional[float] = None
    early_stop_entropy_delta: Optional[float] = None
    score_window: int = 10
    attach_token_rationale: bool = True


@dataclass
class CognitiveLoopResult:
    tokens: Tokens
    text: str
    reasoning_trace: List[Dict[str, Any]]
    safety_events: List[Dict[str, Any]]
    audit_records: List[Dict[str, Any]]
    beam_metadata: Optional[Dict[str, Any]]
    speculative_stats: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    completed: bool
    stopped_reason: str
    duration_seconds: float


# =========================== UTILS =========================== #


def softmax(xs: Sequence[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s == 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def apply_top_k(logits: List[float], k: int) -> List[float]:
    if k <= 0 or k >= len(logits):
        return logits
    indexed_logits = [(logits[i], i) for i in range(len(logits))]
    kth_value = sorted(indexed_logits, key=lambda x: x[0], reverse=True)[k - 1][0]
    return [l if l >= kth_value else float("-inf") for l in logits]


def apply_top_p(logits: List[float], p: float) -> List[float]:
    if p >= 1.0:
        return logits
    probs = softmax(logits)
    sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cumulative, keep = 0.0, []
    for idx in sorted_idx:
        cumulative += probs[idx]
        keep.append(idx)
        if cumulative >= p:
            break
    out = [float("-inf")] * len(logits)
    for idx in keep:
        out[idx] = logits[idx]
    return out


def penalize_repetition(
    logits: List[float], generated: Tokens, penalty: float, window: int
) -> List[float]:
    if penalty <= 1.0 or not generated:
        return logits
    rec = generated[-window:] if window > 0 else generated
    if not all(isinstance(t, int) for t in rec):
        return logits
    counts = {}
    for t in rec:
        counts[t] = counts.get(t, 0) + 1
    out = logits[:]
    for idx in range(len(out)):
        if counts.get(idx, 0) > 0:
            if out[idx] > 0:
                out[idx] = out[idx] / penalty
            else:
                out[idx] = out[idx] * penalty
    return out


def choose_token(logits: List[float], temperature: float) -> int:
    if not logits:
        return 0
    if temperature <= 0:
        return max(range(len(logits)), key=lambda i: logits[i])
    scaled = [l / max(temperature, 1e-9) for l in logits]
    probs = softmax(scaled)
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if cum >= r:
            return i
    return max(range(len(probs)), key=lambda i: probs[i])


def _sequence_entropy(probs: List[float]) -> float:
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy += -p * math.log(p, 2)
    return entropy


# =========================== MAIN CLASS =========================== #


class CognitiveLoop:
    def __init__(
        self,
        bridge: Any,
        transformer: Any,
        safety: Any,
        sampling_config: Optional[LoopSamplingConfig] = None,
        runtime_config: Optional[LoopRuntimeConfig] = None,
        observability_manager: Optional[Any] = None,
        audit_log: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        vocab: Optional[Any] = None,
        consensus_engine: Optional[Any] = None,
        draft_transformer: Optional[Any] = None,
        speculative_function: Optional[SpeculativeFunction] = None,
        reranker: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
        candidate_scorer: Optional[Callable[[Any, Any, Any], float]] = None,
    ) -> None:
        self.bridge = bridge
        self.transformer = transformer
        self.safety = safety
        self.sampling = sampling_config or LoopSamplingConfig()
        self.runtime = runtime_config or LoopRuntimeConfig()
        self.observability = observability_manager
        self.audit_log = audit_log
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.consensus = consensus_engine
        self.draft_transformer = draft_transformer
        self.speculative_function = speculative_function
        self.reranker = reranker
        self.candidate_scorer = candidate_scorer
        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._beam_state: List[Dict[str, Any]] = []
        self._perplex_nll: List[float] = []
        self._entropy_scores: List[float] = []
        self._speculative_stats: Dict[str, Any] = {}
        self.strategy_weights: Dict[str, float] = {"language": 1.0}

    def set_speculative_function(self, fn: SpeculativeFunction) -> None:
        self._speculative_fn = fn

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.runtime.max_parallel_workers
            )
        return self._executor

    # ---------------------- PUBLIC ENTRY ---------------------- #

    async def generate(
        self,
        prompt: Union[str, Tokens],
        max_tokens: Optional[int] = None,
        stream_callback: Optional[Callable[[Token, str, Dict[str, Any]], None]] = None,
        stop_tokens: Optional[Tuple[Token, ...]] = None,
        stop_strings: Optional[Tuple[str, ...]] = None,
    ) -> Union[AsyncGenerator[Dict[str, Any], None], CognitiveLoopResult]:
        start = time.time()
        max_steps = max_tokens or self.sampling.max_tokens
        stop_tok_set = set(stop_tokens or ()) | set(self.sampling.stop_tokens)
        stop_str_patterns = tuple(stop_strings or ()) + self.sampling.stop_strings

        if isinstance(prompt, str):
            init_tokens = await self._tokenize(prompt)
        else:
            init_tokens = prompt[:]

        generated: Tokens = []
        reasoning_trace: List[Dict[str, Any]] = []
        safety_events: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []
        safety_interventions = 0
        beam_metadata: Optional[Dict[str, Any]] = None
        speculative_stats: Optional[Dict[str, Any]] = None
        errors = 0
        perplex_nll: List[float] = []
        entropy_scores: List[float] = []

        self._perplex_nll = []
        self._entropy_scores = []
        self._speculative_stats = {}

        temperature = self.sampling.temperature
        top_k = self.sampling.top_k

        if self.sampling.use_beam_search:
            self._beam_state = [
                {
                    "tokens": init_tokens,
                    "log_prob": 0.0,
                    "score": 0.0,
                    "is_finished": False,
                }
            ]

        async def _generator() -> AsyncGenerator[Dict[str, Any], None]:
            # ADD generated TO nonlocal (critical fix)
            nonlocal temperature, top_k, errors, beam_metadata, speculative_stats, safety_interventions, generated

            last_reasoning_meta: Dict[str, Any] = {}

            for step in range(max_steps):
                if self.sampling.use_beam_search and all(
                    b["is_finished"] for b in self._beam_state
                ):
                    break

                if (
                    self.runtime.time_budget_seconds
                    and (time.time() - start) >= self.runtime.time_budget_seconds
                ):
                    yield await self._finalize(
                        prompt,
                        init_tokens,
                        generated,
                        reasoning_trace,
                        safety_events,
                        audit_records,
                        beam_metadata,
                        speculative_stats,
                        start,
                        True,
                        "time_budget_exceeded",
                        perplex_nll,
                        entropy_scores,
                        safety_interventions,
                    )
                    return

                try:
                    step_result = await self._step(
                        prompt_tokens=init_tokens + generated,
                        temperature=temperature,
                        top_k=top_k,
                        stop_tokens=stop_tok_set,
                        stop_strings=stop_str_patterns,
                        step=step,
                    )
                    errors = 0
                except Exception as e:
                    print(f"Error during cognitive step: {e}\n{traceback.format_exc()}")
                    errors += 1
                    if errors > self.runtime.max_errors:
                        yield await self._finalize(
                            prompt,
                            init_tokens,
                            generated,
                            reasoning_trace,
                            safety_events,
                            audit_records,
                            beam_metadata,
                            speculative_stats,
                            start,
                            False,
                            f"error_exceeded:{e.__class__.__name__}",
                            perplex_nll,
                            entropy_scores,
                            safety_interventions,
                        )
                        return
                    await asyncio.sleep(self.runtime.error_backoff_seconds)
                    continue

                token = step_result["token"]
                token_info = step_result["info"]

                reasoning_trace.append(token_info.get("reasoning", {}))
                last_reasoning_meta = token_info.get("reasoning", {})

                if token_info.get("safety_event"):
                    safety_events.append(token_info["safety_event"])
                    if token_info["safety_event"].get("reason") in [
                        "token_validation_replacement",
                        "world_model_intervention",
                        "sequence_replaced",
                        "consensus_rejection",
                    ]:
                        safety_interventions += 1
                if token_info.get("audit_record"):
                    audit_records.append(token_info["audit_record"])
                if token_info.get("beam"):
                    beam_metadata = token_info["beam"]
                    generated = token_info["beam"].get(
                        "best_sequence_tokens", generated
                    )
                if token_info.get("speculative"):
                    speculative_stats = token_info["speculative"]
                    generated.extend(
                        token_info["speculative"].get("accepted_tokens", [])
                    )
                    token = token_info["speculative"].get("next_token", None)

                if (
                    token_info.get("logits")
                    and token_info.get("chosen_index") is not None
                ):
                    logits = token_info["logits"]
                    chosen_idx = token_info["chosen_index"]
                    probs = softmax(logits)
                    nll = 0.0
                    if 0 <= chosen_idx < len(probs):
                        nll = -math.log(max(probs[chosen_idx], 1e-12))
                    entropy = _sequence_entropy(probs)
                    perplex_nll.append(nll)
                    entropy_scores.append(entropy)
                    self._perplex_nll.append(nll)
                    self._entropy_scores.append(entropy)

                if len(self._perplex_nll) >= self.runtime.score_window:
                    if self.runtime.early_stop_score_delta is not None:
                        ppl_improvement = self._improvement(
                            self._perplex_nll[-self.runtime.score_window :]
                        )
                        if ppl_improvement < self.runtime.early_stop_score_delta:
                            yield await self._finalize(
                                prompt,
                                init_tokens,
                                generated,
                                reasoning_trace,
                                safety_events,
                                audit_records,
                                beam_metadata,
                                speculative_stats,
                                start,
                                True,
                                "early_stop_ppl",
                                perplex_nll,
                                entropy_scores,
                                safety_interventions,
                            )
                            return
                    if self.runtime.early_stop_entropy_delta is not None:
                        entropy_improvement = self._improvement(
                            self._entropy_scores[-self.runtime.score_window :]
                        )
                        if entropy_improvement < self.runtime.early_stop_entropy_delta:
                            yield await self._finalize(
                                prompt,
                                init_tokens,
                                generated,
                                reasoning_trace,
                                safety_events,
                                audit_records,
                                beam_metadata,
                                speculative_stats,
                                start,
                                True,
                                "early_stop_entropy",
                                perplex_nll,
                                entropy_scores,
                                safety_interventions,
                            )
                            return

                if token is not None:
                    if (
                        not self.sampling.use_beam_search
                        and not self.sampling.speculative_enabled
                    ):
                        generated.append(token)

                    current_text = await self._decode(generated)

                    stream_chunk = {
                        "token": token,
                        "text": current_text,
                        "token_info": token_info,
                    }
                    if self.runtime.enable_stream:
                        yield stream_chunk

                    if stream_callback:
                        stream_callback(token, current_text, token_info)

                    if token in stop_tok_set:
                        yield await self._finalize(
                            prompt,
                            init_tokens,
                            generated,
                            reasoning_trace,
                            safety_events,
                            audit_records,
                            beam_metadata,
                            speculative_stats,
                            start,
                            True,
                            f"stop_token:{token}",
                            perplex_nll,
                            entropy_scores,
                            safety_interventions,
                        )
                        return

                    for pattern in stop_str_patterns:
                        if pattern and pattern in current_text:
                            try:
                                full_text = await self._decode(init_tokens + generated)
                                split_text = full_text.split(pattern, 1)
                                final_text_segment = split_text[0] + pattern
                                final_tokens = await self._tokenize(final_text_segment)
                                generated = final_tokens[len(init_tokens) :]
                            except Exception as e:
                                # Log tokenization errors but continue
                                logger.warning(f"Failed to tokenize stop pattern: {e}")
                            yield await self._finalize(
                                prompt,
                                init_tokens,
                                generated,
                                reasoning_trace,
                                safety_events,
                                audit_records,
                                beam_metadata,
                                speculative_stats,
                                start,
                                True,
                                f"stop_string:{pattern}",
                                perplex_nll,
                                entropy_scores,
                                safety_interventions,
                            )
                            return

                if self.sampling.adaptive_temperature:
                    temperature = max(
                        self.sampling.temperature_floor,
                        temperature * self.sampling.temperature_decay,
                    )
                if self.sampling.dynamic_top_k:
                    top_k = max(self.sampling.min_top_k, int(top_k * 0.99))

                if len(generated) >= max_steps:
                    break

            # Removed redundant bandit feedback block (was referencing undefined reasoning_meta)
            yield await self._finalize(
                prompt,
                init_tokens,
                generated,
                reasoning_trace,
                safety_events,
                audit_records,
                beam_metadata,
                speculative_stats,
                start,
                True,
                "max_tokens_reached",
                perplex_nll,
                entropy_scores,
                safety_interventions,
            )
            return

        if self.runtime.enable_stream:
            return _generator()
        else:
            final_result = None
            async for result in _generator():
                final_result = result
            return final_result

    async def _step(
        self,
        prompt_tokens: Tokens,
        temperature: float,
        top_k: int,
        stop_tokens: set,
        stop_strings: Tuple[str, ...],
        step: int,
    ) -> Dict[str, Any]:
        t0 = time.time()
        sub_times = {}  # Timing dictionary
        token_info: Dict[str, Any] = {}
        reasoning_meta: Dict[str, Any] = {}
        token = None
        chosen_index = None
        logits = None
        candidates: List[Any] = []
        retrieved_context: Dict[str, Any] = {}
        hidden_state: Any = None

        if self.sampling.use_beam_search:
            t_beam = time.time()
            (
                token,
                beam_meta,
                prompt_tokens,
            ) = await self._multi_step_beam_search_expansion(
                prompt_tokens, temperature, top_k, stop_tokens
            )
            sub_times["beam_search_ms"] = (time.time() - t_beam) * 1000
            token_info["beam"] = beam_meta
            chosen_index = beam_meta.get("chosen_index")
            reasoning_meta["strategy"] = "beam_search"
        else:
            t_ctx = time.time()
            retrieved_context = await self._async_safe(
                self.bridge.before_execution, {"prompt_tokens": prompt_tokens}, {}
            )
            sub_times["context_retrieval_ms"] = (time.time() - t_ctx) * 1000
            token_info["retrieved_context"] = retrieved_context

            if self.runtime.attach_world_model_hooks and hasattr(
                self.bridge, "world_model"
            ):
                try:
                    t_wm_up = time.time()
                    await self._async_safe(
                        self.bridge.world_model.update, {"tokens": prompt_tokens}, None
                    )
                    sub_times["wm_update_ms"] = (time.time() - t_wm_up) * 1000
                except Exception as e:
                    # World model update is optional; log errors but continue
                    logger.debug(f"World model update failed: {e}")

            t_enc = time.time()
            hidden_state = await self._async_safe(
                self.transformer.encode, prompt_tokens, None
            )
            sub_times["encode_ms"] = (time.time() - t_enc) * 1000
            reasoning_meta["hidden_state_shape"] = getattr(hidden_state, "shape", None)

            available_strategies = list(self.strategy_weights.keys())
            strategy_weights = [self.strategy_weights[s] for s in available_strategies]
            reasoning_meta["strategy"] = random.choices(
                available_strategies, weights=strategy_weights
            )[0]

            t_cand = time.time()
            candidates = await self._select_candidates(hidden_state, retrieved_context)
            sub_times["select_candidates_ms"] = (time.time() - t_cand) * 1000
            reasoning_meta["candidate_count"] = len(candidates)

            candidate_scores = None
            if self.runtime.parallel_score_candidates and self.candidate_scorer:
                t_score = time.time()
                candidate_scores = await self._parallel_score(
                    hidden_state, candidates, retrieved_context
                )
                sub_times["parallel_score_ms"] = (time.time() - t_score) * 1000
                reasoning_meta["candidate_scores"] = candidate_scores

            if self.runtime.enable_rerank and candidate_scores and self.reranker:
                bundle = [
                    {"candidate": c, "score": s, "index": i}
                    for i, (c, s) in enumerate(zip(candidates, candidate_scores))
                ]
                t_rerank = time.time()
                reranked = await self._async_safe(self.reranker, bundle, bundle)
                sub_times["rerank_ms"] = (time.time() - t_rerank) * 1000
                candidates = [b["candidate"] for b in reranked]
                reasoning_meta["reranked"] = True

            t_logits = time.time()
            logits = await self._obtain_logits(hidden_state, prompt_tokens, candidates)
            sub_times["get_logits_ms"] = (time.time() - t_logits) * 1000
            token_info["logits"] = logits if self.runtime.attach_logits else None

            if self.sampling.speculative_enabled and self.draft_transformer:
                t_spec = time.time()
                accepted_tokens, token, spec_meta = await self._speculative_decoding(
                    prompt_tokens, temperature, top_k
                )
                sub_times["speculative_ms"] = (time.time() - t_spec) * 1000
                token_info["speculative"] = spec_meta
                prompt_tokens.extend(accepted_tokens)
            else:
                t_sample = time.time()
                chosen_index, adjusted_logits = self._sample(
                    logits=logits,
                    generated_tokens=prompt_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=self.sampling.top_p,
                )
                token = await self._index_to_token(chosen_index)
                sub_times["sample_ms"] = (time.time() - t_sample) * 1000

            token_info["chosen_index"] = chosen_index

        reasoning_meta["selected_token"] = token

        safety_event = None
        if (
            self.runtime.require_consensus_per_token
            and token is not None
            and self.consensus
        ):
            t_cons = time.time()
            approved = await self._per_token_consensus(
                token,
                prompt_tokens,
                chosen_index,
                extra_context={
                    "strategy": reasoning_meta.get("strategy"),
                    "candidate_count": reasoning_meta.get("candidate_count"),
                },
            )
            sub_times["consensus_ms"] = (time.time() - t_cons) * 1000
            if not approved:
                safety_event = {
                    "blocked_token": token,
                    "reason": "consensus_rejection",
                    "timestamp": time.time(),
                }
                token = None

        if token is not None:
            if self.runtime.safety_per_token:
                t_val_tok = time.time()
                safe_token, intervention_notes = await self._validate_token(
                    token, retrieved_context, hidden_state
                )
                sub_times["validate_token_ms"] = (time.time() - t_val_tok) * 1000
                if safe_token != token:
                    safety_event = safety_event or {
                        "original": token,
                        "replacement": safe_token,
                        "reason": "token_validation_replacement",
                        "timestamp": time.time(),
                    }
                    token = safe_token
                    reasoning_meta["intervention"] = intervention_notes

            if self.runtime.safety_per_sequence:
                hypot_seq = prompt_tokens + [token]
                t_val_seq = time.time()
                seq_ok = await self._async_safe(
                    getattr(self.safety, "validate_sequence", None),
                    (
                        hypot_seq,
                        retrieved_context,
                        getattr(self.bridge, "world_model", None),
                    ),
                    True,
                )
                sub_times["validate_sequence_ms"] = (time.time() - t_val_seq) * 1000
                if isinstance(seq_ok, list) and seq_ok[-1] != token:
                    safety_event = safety_event or {
                        "reason": "sequence_replaced",
                        "timestamp": time.time(),
                        "original_tail": hypot_seq[-5:],
                    }
                    token = seq_ok[-1]
                elif seq_ok is False:
                    safety_event = safety_event or {
                        "reason": "sequence_block",
                        "blocked_token": token,
                        "timestamp": time.time(),
                    }
                    token = None

        memory_record = {
            "prompt_tokens": prompt_tokens,
            "emitted_token": token,
            "time": time.time(),
            "strategy": reasoning_meta.get("strategy"),
            "reasoning_meta": reasoning_meta,
        }
        t_after_exec = time.time()
        await self._async_safe(self.bridge.after_execution, memory_record, None)
        sub_times["after_execution_ms"] = (time.time() - t_after_exec) * 1000

        audit_record = None
        if self.runtime.enable_audit:
            audit_record = {
                "timestamp": time.time(),
                "token": token,
                "duration_ms": (time.time() - t0) * 1000,
                "candidate_count": reasoning_meta.get("candidate_count"),
                "strategy": reasoning_meta.get("strategy"),
                "consensus_required": self.runtime.require_consensus_per_token,
                "safety_applied": bool(safety_event),
            }
            if self.audit_log:
                await self._async_safe(self._log_audit_record, audit_record, None)

        if self.runtime.enable_observability:
            self._obs_sync(
                "cognitive_step",
                {
                    "token": token,
                    "duration_ms": (time.time() - t0) * 1000,
                    "strategy": reasoning_meta.get("strategy"),
                    "candidates": reasoning_meta.get("candidate_count"),
                    "beam": bool(self.sampling.use_beam_search),
                    "speculative": bool(self.sampling.speculative_enabled),
                },
            )

        if (
            self.runtime.enable_strategy_bandit_feedback
            and step > self.runtime.score_window
            and reasoning_meta.get("strategy")
        ):
            window = self._perplex_nll[-self.runtime.score_window :]
            if window:
                ppl_improvement = self._improvement(window)
                strategy = reasoning_meta["strategy"]
                reward = min(1.0, max(-1.0, -ppl_improvement))
                self.strategy_weights[strategy] = (
                    self.strategy_weights.get(strategy, 1.0) + reward
                )
                self._obs_sync(
                    "bandit.strategy_feedback",
                    {
                        "strategy": strategy,
                        "reward": reward,
                        "new_weight": self.strategy_weights[strategy],
                    },
                )

        if self.runtime.attach_token_rationale and token is not None:
            t_rationale = time.time()
            token_rationale = await self._build_token_rationale(
                token, hidden_state, candidates, retrieved_context
            )
            sub_times["rationale_ms"] = (time.time() - t_rationale) * 1000
            reasoning_meta["rationale"] = token_rationale

        if self.runtime.trace_reasoning:
            token_info["reasoning"] = {
                "phase": "SELECT->APPLY",
                "strategy": reasoning_meta.get("strategy"),
                "hidden_state_shape": reasoning_meta.get("hidden_state_shape"),
                "sampling": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": self.sampling.top_p,
                    "beam": self.sampling.use_beam_search,
                    "speculative": self.sampling.speculative_enabled,
                },
                "candidate_preview": candidates[:10],
                "rationale": reasoning_meta.get("rationale"),
                "intervention": reasoning_meta.get("intervention"),
                "timings": sub_times,  # Add timing info
            }

        if safety_event:
            token_info["safety_event"] = safety_event
        if audit_record:
            token_info["audit_record"] = audit_record

        token_info["token"] = token
        return {"token": token, "info": token_info}

    async def _select_candidates(self, hidden_state: Any, context: Any) -> List[Any]:
        if hidden_state is None:
            return []
        reasoning = getattr(self.bridge, "reasoning", None)
        if reasoning and hasattr(reasoning, "select_next_token"):
            cands = await self._async_safe(
                reasoning.select_next_token, (hidden_state, context), [hidden_state]
            )
            if isinstance(cands, (list, tuple)):
                return list(cands)
            return [cands]
        return [hidden_state]

    async def _obtain_logits(
        self, hidden_state: Any, prompt_tokens: Tokens, candidates: List[Any]
    ) -> List[float]:
        if hasattr(self.transformer, "get_logits"):
            try:
                logits = await self._async_safe(
                    self.transformer.get_logits, (hidden_state, prompt_tokens), None
                )
                if isinstance(logits, list):
                    return logits
                if hasattr(logits, "tolist"):
                    return logits.tolist()
            except Exception as e:
                # Log logit extraction failure and fall through to alternative methods
                logger.warning(f"Failed to get logits from transformer: {e}")
        vocab_size = getattr(self.transformer, "vocab_size", None)
        if vocab_size is None and self.vocab and hasattr(self.vocab, "size"):
            vocab_size = await self._async_safe(self.vocab.size, (), 200)
        if vocab_size is None:
            vocab_size = 200
        return [0.0] * vocab_size

    def _sample(
        self,
        logits: List[float],
        generated_tokens: Tokens,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[int, List[float]]:
        if not logits:
            return 0, logits
        filtered = apply_top_k(logits, top_k)
        filtered = apply_top_p(filtered, top_p)
        if not self.sampling.allow_repetition:
            filtered = penalize_repetition(
                filtered,
                generated_tokens,
                self.sampling.repetition_penalty,
                self.sampling.repetition_window,
            )
        chosen = choose_token(filtered, temperature)
        return chosen, filtered

    async def _index_to_token(self, idx: int) -> Token:
        if self.vocab and hasattr(self.vocab, "id_to_token"):
            try:
                return await asyncio.to_thread(self.vocab.id_to_token, idx)
            except Exception as e:
                # Log vocab lookup failure, fallback to returning index
                logger.debug(f"Failed to convert index to token: {e}")
        return idx

    async def _multi_step_beam_search_expansion(
        self, init_tokens: Tokens, temperature: float, top_k: int, stop_tokens: set
    ) -> Tuple[Token, Dict[str, Any], Tokens]:
        if not self._beam_state:
            self._beam_state = [
                {
                    "tokens": init_tokens,
                    "log_prob": 0.0,
                    "score": 0.0,
                    "is_finished": False,
                }
            ]
        new_beams: List[Dict[str, Any]] = []
        for exp in range(self.sampling.beam_max_expansions):
            beams_to_expand = [b for b in self._beam_state if not b["is_finished"]]
            if not beams_to_expand and exp > 0:
                break
            temp_new_beams: List[Dict[str, Any]] = []
            for beam in beams_to_expand:
                current_tokens = beam["tokens"]
                hidden_state = await self._async_safe(
                    self.transformer.encode, current_tokens, None
                )
                logits = await self._obtain_logits(hidden_state, current_tokens, [])
                filtered_logits = apply_top_k(logits, self.sampling.beam_width)
                probs = softmax(filtered_logits)
                top_indices_with_probs = sorted(
                    [(p, idx) for idx, p in enumerate(probs) if p > 0],
                    key=lambda x: x[0],
                    reverse=True,
                )[: self.sampling.beam_width]
                for p, idx in top_indices_with_probs:
                    log_prob = math.log(p)
                    new_log_prob = beam["log_prob"] + log_prob
                    token_id = await self._index_to_token(idx)
                    new_tokens = current_tokens + [token_id]
                    length_norm = len(new_tokens) ** self.sampling.beam_length_penalty
                    score = new_log_prob / length_norm
                    is_finished = token_id in stop_tokens
                    new_beam = {
                        "tokens": new_tokens,
                        "log_prob": new_log_prob,
                        "score": score,
                        "is_finished": is_finished,
                    }
                    temp_new_beams.append(new_beam)
            self._beam_state = [
                b for b in self._beam_state if b["is_finished"]
            ] + temp_new_beams
            self._beam_state = sorted(
                self._beam_state, key=lambda b: b["score"], reverse=True
            )[: self.sampling.beam_width]
        best_overall_beam = max(self._beam_state, key=lambda b: b["score"])
        final_token = best_overall_beam["tokens"][-1]
        meta = {
            "beam_candidates": self._beam_state,
            "chosen_score": best_overall_beam["score"],
            "best_sequence_tokens": best_overall_beam["tokens"][len(init_tokens) :],
            "beam_width": self.sampling.beam_width,
        }
        return final_token, meta, best_overall_beam["tokens"]

    async def _speculative_decoding(
        self, prompt_tokens: Tokens, temperature: float, top_k: int
    ) -> Tuple[Tokens, Optional[Token], Dict[str, Any]]:
        if not self._speculative_fn or not self.draft_transformer:
            raise RuntimeError(
                "Speculative decoding requested but no speculative function / draft_transformer attached"
            )
        main_model = getattr(self.transformer, "backend", self.transformer)
        accepted_tokens, token, spec_meta = await self._async_safe(
            self._speculative_fn,
            (main_model, self.draft_transformer, prompt_tokens, temperature, top_k),
            ([], None, {}),
        )
        if token is None:
            return [], None, spec_meta or {}
        self._speculative_stats = spec_meta or {}
        return accepted_tokens, token, spec_meta or {}

    async def _parallel_score(
        self, hidden_state: Any, candidates: List[Any], context: Any
    ) -> List[float]:
        if not candidates or not self.candidate_scorer:
            return [0.0] * len(candidates)
        loop = asyncio.get_event_loop()
        executor = self._get_executor()
        scorer = self.candidate_scorer
        futures = [
            loop.run_in_executor(executor, scorer, hidden_state, cand, context)
            for cand in candidates
        ]
        default_score = -0.5
        scores = []
        timeout = self.runtime.consensus_timeout_seconds
        for future in futures:
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
                score = result[0] if isinstance(result, tuple) else result
                scores.append(float(score))
            except asyncio.TimeoutError:
                self._obs_sync("scoring.timeout", {"timeout_seconds": timeout})
                scores.append(default_score)
            except Exception as e:
                self._obs_sync("scoring.error", {"error": str(e)})
                scores.append(default_score)
        return scores

    async def _per_token_consensus(
        self,
        token: Token,
        prompt_tokens: Tokens,
        chosen_index: int,
        extra_context: Dict[str, Any],
    ) -> bool:
        if not self.consensus or not hasattr(self.consensus, "approve"):
            return True
        proposal = {
            "type": "token_emission",
            "token": token,
            "position": len(prompt_tokens),
            "chosen_index": chosen_index,
            "timestamp": time.time(),
            **extra_context,
        }
        try:
            approve_func = self.consensus.approve
            approved = await self._async_safe(
                approve_func,
                proposal,
                default=True,
            )
            return bool(approved)
        except (asyncio.TimeoutError, Exception):
            return True

    async def _validate_token(
        self, token: Token, context: Any, hidden_state: Any
    ) -> Tuple[Token, Optional[Dict[str, Any]]]:
        if not self.safety or not hasattr(self.safety, "validate_generation"):
            if hasattr(self.bridge, "world_model") and hasattr(
                self.bridge.world_model, "intervene_before_emit"
            ):
                intervention = await self._async_safe(
                    self.bridge.world_model.intervene_before_emit,
                    {"token": token, "context": context, "hidden_state": hidden_state},
                    {},
                )
                if intervention and intervention.get("modified_token") is not None:
                    return intervention["modified_token"], intervention
                return token, None
            return token, None
        try:
            final_token, notes = await self.bridge.validate_token(
                token, context, hidden_state
            )
            return final_token, notes
        except Exception:
            return token, None

    async def _build_token_rationale(
        self, token: Token, hidden_state: Any, candidates: List[Any], context: Any
    ) -> Dict[str, Any]:
        if not self.runtime.attach_token_rationale:
            return {}
        rationale = {
            "token": token,
            "candidate_set_size": len(candidates),
        }
        reasoning = getattr(self.bridge, "reasoning", None)
        if reasoning and hasattr(self.bridge, "explain_choice"):
            try:
                explanation = await self.bridge.explain_choice(
                    token, hidden_state, context
                )
                rationale["explanation"] = explanation
            except Exception as e:
                # Rationale explanation is supplementary; log but continue
                logger.debug(f"Failed to get choice explanation: {e}")
        return rationale

    async def _finalize(
        self,
        prompt: Union[str, Tokens],
        init_tokens: Tokens,
        generated: Tokens,
        reasoning_trace: List[Dict[str, Any]],
        safety_events: List[Dict[str, Any]],
        audit_records: List[Dict[str, Any]],
        beam_metadata: Optional[Dict[str, Any]],
        speculative_stats: Optional[Dict[str, Any]],
        start_time: float,
        completed: bool,
        stopped_reason: str,
        perplex_nll: List[float],
        entropy_scores: List[float],
        safety_interventions_count: int,
    ) -> CognitiveLoopResult:
        duration = time.time() - start_time
        text = await self._decode(generated)
        metrics = {
            "num_generated": len(generated),
            "duration_seconds": duration,
            "tokens_per_second": (len(generated) / duration) if duration > 0 else 0.0,
            "stopped_reason": stopped_reason,
            "safety_interventions_count": safety_interventions_count,
            "total_safety_events": len(safety_events),
        }
        if perplex_nll:
            metrics["avg_token_nll"] = sum(perplex_nll) / len(perplex_nll)
            metrics["perplexity"] = math.exp(metrics["avg_token_nll"])
        if entropy_scores:
            metrics["avg_entropy"] = sum(entropy_scores) / len(entropy_scores)
        return CognitiveLoopResult(
            tokens=generated,
            text=text,
            reasoning_trace=reasoning_trace,
            safety_events=safety_events,
            audit_records=audit_records,
            beam_metadata=beam_metadata,
            speculative_stats=speculative_stats or self._speculative_stats,
            metrics=metrics,
            completed=completed,
            stopped_reason=stopped_reason,
            duration_seconds=duration,
        )

    async def _tokenize(self, text: str) -> Tokens:
        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            try:
                return await asyncio.to_thread(self.tokenizer.encode, text)
            except Exception as e:
                # Log tokenization failure, fallback to word splitting
                logger.warning(f"Tokenization failed, using word split: {e}")
        return list(text.split())

    async def _decode(self, tokens: Tokens) -> str:
        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            try:
                return await asyncio.to_thread(self.tokenizer.decode, tokens)
            except Exception as e:
                # Log decoding failure, fallback to sync method
                logger.warning(f"Async decoding failed, using fallback: {e}")
        return self._decode_sync(tokens)

    def _decode_sync(self, tokens: Tokens) -> str:
        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(tokens)
            except Exception as e:
                # Log sync decoding failure
                logger.warning(f"Sync decoding failed: {e}")
        if (
            tokens
            and isinstance(tokens[0], int)
            and self.vocab
            and hasattr(self.vocab, "id_to_token")
        ):
            try:
                return "".join(str(self.vocab.id_to_token(t)) for t in tokens)
            except Exception as e:
                # Log vocab-based decoding failure
                logger.warning(f"Vocab-based decoding failed: {e}")
        return " ".join(str(t) for t in tokens)

    def _improvement(self, window: List[float]) -> float:
        if len(window) < 2:
            return 0.0
        diffs = [window[i] - window[i - 1] for i in range(1, len(window))]
        avg_delta = sum(diffs) / len(diffs)
        return -avg_delta

    async def _async_safe(self, fn: Optional[Callable], args: Any, default: Any) -> Any:
        if fn is None:
            return default
        args_tuple = args if isinstance(args, tuple) else (args,)
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args_tuple)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, fn, *args_tuple)
        except Exception as e:
            # Log async operation failures
            logger.warning(f"Async safe operation failed, returning default: {e}")
            return default

    def _log_audit_record(self, record: Dict[str, Any]) -> None:
        if self.audit_log:
            try:
                if hasattr(self.audit_log, "append"):
                    self.audit_log.append(record)
                elif hasattr(self.audit_log, "record"):
                    self.audit_log.record("cognitive_step", record)
            except Exception as e:
                # Audit logging failures should be visible
                logger.warning(f"Failed to log audit record: {e}")

    def _obs_sync(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.observability:
            return
        try:
            if hasattr(self.observability, "record"):
                self.observability.record(event_type, payload)
            elif hasattr(self.observability, "log"):
                self.observability.log(event_type, payload)
        except Exception as e:
            # Observability failures should not break operations
            logger.debug(f"Observability recording failed: {e}")
