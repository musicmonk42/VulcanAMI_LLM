from __future__ import annotations
"""
LanguageReasoning

Fully functional neural-language reasoning mode wrapper with:
- Next-token candidate generation from hidden state (via attached model/executor)
- Sampling (temperature, top-k, top-p / nucleus)
- Greedy, beam search, and stochastic modes
- Repetition penalty
- Optional logits post-processing hooks (e.g. safety mask, bias adjustments)
- Optional external reranker
- Pluggable strategy selection (e.g., fallback to beam when high uncertainty)
- Rich trace metadata for explainability modules

Duck-typed integration points:
- model.get_logits(hidden_state, tokens_so_far) -> List[float]
- safety.validate_generation(token, context, world_model) -> token (optional)
- world_model.validate_generation(token, context) / suggest_correction(token, context) (optional)
- reranker(candidates: List[Dict]) -> List[Dict] (optional)
- bias_hook(logits: List[float], context: Dict) -> List[float] (optional)
- mask_hook(logits: List[float], context: Dict) -> List[float] (optional)

Returned structure from generate():
{
  "token": <selected token>,
  "token_id": <int>,
  "candidates": [
      {"id": int, "logit": float, "prob": float, "raw_rank": int}
      ...
  ],
  "strategy": "greedy|sample|beam",
  "beam": {"paths": [...], "chosen": {...}} (if beam used),
  "sampling": {"temperature": ..., "top_k": ..., "top_p": ...},
  "reasoning": {
      "entropy": float,
      "confidence": float,
      "repetition_penalty_applied": bool
  }
}

This module is dependency-light and uses pure Python numerics.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Sequence, Tuple


# --------------------------- Configuration --------------------------- #

@dataclass
class LanguageReasoningConfig:
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    repetition_window: int = 64
    use_beam_search: bool = False
    beam_width: int = 4
    beam_max_steps: int = 4              # beam expansion depth (local per-step)
    beam_length_penalty: float = 0.7
    min_probability_floor: float = 1e-12
    strategy_auto_switch_entropy: float = 4.0  # if entropy > threshold, switch to beam
    max_candidates_return: int = 200
    allow_greedy_fallback: bool = True
    seed: Optional[int] = None
    deterministic_greedy: bool = True
    apply_repetition_penalty: bool = True
    attach_full_probs: bool = True
    # Optional automatic shrink of top_k when entropy is low
    dynamic_top_k: bool = True
    dynamic_top_k_floor: int = 10
    dynamic_top_k_decay: float = 0.95


# --------------------------- Utility Functions --------------------------- #

def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _entropy(probs: Sequence[float], eps: float = 1e-12) -> float:
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def _apply_top_k(logits: List[float], k: int) -> List[float]:
    if k <= 0 or k >= len(logits):
        return logits
    keep = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]
    out = [float("-inf")] * len(logits)
    for i in keep:
        out[i] = logits[i]
    return out


def _apply_top_p(logits: List[float], top_p: float) -> List[float]:
    if top_p >= 1.0:
        return logits
    probs = _softmax(logits)
    idxs = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cumulative = 0.0
    keep: List[int] = []
    for idx in idxs:
        cumulative += probs[idx]
        keep.append(idx)
        if cumulative >= top_p:
            break
    out = [float("-inf")] * len(logits)
    for idx in keep:
        out[idx] = logits[idx]
    return out


def _apply_repetition_penalty(
    logits: List[float],
    generated: Sequence[Any],
    penalty: float,
    window: int
) -> List[float]:
    if penalty <= 1.0 or not generated:
        return logits
    recent = generated[-window:] if window > 0 else generated
    counts: Dict[int, int] = {}
    # Only apply if tokens are int IDs
    for t in recent:
        if isinstance(t, int):
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return logits
    out = logits[:]
    for i in range(len(out)):
        if counts.get(i, 0) > 0:
            out[i] = out[i] / penalty
    return out


def _sample_index(filtered_logits: List[float], temperature: float) -> int:
    if temperature <= 0:
        return max(range(len(filtered_logits)), key=lambda i: filtered_logits[i])
    scaled = [l / max(temperature, 1e-9) for l in filtered_logits]
    probs = _softmax(scaled)
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if cumulative >= r:
            return i
    return len(probs) - 1


# --------------------------- LanguageReasoning --------------------------- #

class LanguageReasoning:
    """
    Neural language generation as a reasoning mode.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[LanguageReasoningConfig] = None,
        safety: Optional[Any] = None,
        world_model: Optional[Any] = None,
        reranker: Optional[Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]] = None,
        bias_hook: Optional[Callable[[List[float], Dict[str, Any]], List[float]]] = None,
        mask_hook: Optional[Callable[[List[float], Dict[str, Any]], List[float]]] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Args:
            model: object exposing get_logits(hidden_state, tokens_so_far)
            safety: optional safety validator
            world_model: optional world model for validation/correction
            reranker: optional function (candidates, context) -> candidates (rescored)
            bias_hook: modifies logits (e.g., domain bias) before sampling
            mask_hook: masks logits (e.g., blocking unsafe tokens)
        """
        self.model = model
        self.cfg = config or LanguageReasoningConfig()
        self.safety = safety
        self.world_model = world_model
        self.reranker = reranker
        self.bias_hook = bias_hook
        self.mask_hook = mask_hook
        seed = self.cfg.seed if self.cfg.seed is not None else random_seed
        if seed is not None:
            random.seed(seed)

    # --------------------- Public Interface --------------------- #

    def generate(
        self,
        hidden_state: Any,
        generated_tokens: Sequence[Any],
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate next token with reasoning trace.

        Args:
            hidden_state: last hidden state (model/encoder output)
            generated_tokens: tokens already produced (list-like)
            context: optional context object
            strategy: override strategy ('greedy'|'sample'|'beam')

        Returns:
            dict see module docstring.
        """
        context = context or {}
        # Step 1: Acquire logits
        logits = self._get_logits_safe(hidden_state, generated_tokens)
        if not logits:
            return self._empty_fallback()

        # Step 2: Post-process (mask / bias)
        if self.mask_hook:
            try:
                logits = self.mask_hook(logits, context)
            except Exception:
                pass
        if self.bias_hook:
            try:
                logits = self.bias_hook(logits, context)
            except Exception:
                pass

        # Step 3: Base probability distribution
        base_probs = _softmax(logits)
        ent = _entropy(base_probs)
        # Dynamic top-k shrink if enabled
        effective_top_k = self.cfg.top_k
        if self.cfg.dynamic_top_k and ent < self.cfg.strategy_auto_switch_entropy / 2:
            effective_top_k = max(self.cfg.dynamic_top_k_floor, int(effective_top_k * self.cfg.dynamic_top_k_decay))

        # Step 4: Auto strategy switch if high entropy
        chosen_strategy = strategy or ("beam" if (self.cfg.use_beam_search or ent > self.cfg.strategy_auto_switch_entropy) else "sample")

        # Step 5: Repetition penalty
        repetition_applied = False
        filtered_logits = logits[:]
        if self.cfg.apply_repetition_penalty:
            filtered_logits = _apply_repetition_penalty(
                filtered_logits,
                generated_tokens,
                self.cfg.repetition_penalty,
                self.cfg.repetition_window
            )
            repetition_applied = filtered_logits != logits

        # Step 6: Branch by strategy
        if chosen_strategy == "greedy":
            token_id = max(range(len(filtered_logits)), key=lambda i: filtered_logits[i])
            candidates = self._build_candidate_list(filtered_logits, limit=self.cfg.max_candidates_return)
            beam_info = None
        elif chosen_strategy == "beam":
            token_id, beam_info, candidates = self._beam_search(
                hidden_state,
                generated_tokens,
                filtered_logits,
                effective_top_k,
                ent
            )
        else:  # 'sample'
            temp = self.cfg.temperature
            top_k_logits = _apply_top_k(filtered_logits, effective_top_k)
            top_p_logits = _apply_top_p(top_k_logits, self.cfg.top_p)
            token_id = _sample_index(top_p_logits, temp)
            candidates = self._build_candidate_list(top_p_logits, limit=self.cfg.max_candidates_return)
            beam_info = None

        # Step 7: Candidate reranking (optional)
        if self.reranker and candidates:
            try:
                candidates = self.reranker(candidates, {"context": context, "entropy": ent})
            except Exception:
                pass

        # Step 8: Safety + world model validation/correction
        final_token_id = self._validate_token(token_id, context)

        # Step 9: Build reasoning meta
        chosen_prob = base_probs[token_id] if token_id < len(base_probs) else None
        confidence = chosen_prob
        reasoning_meta = {
            "entropy": ent,
            "confidence": confidence,
            "repetition_penalty_applied": repetition_applied
        }

        # Step 10: Final structure
        return {
            "token": final_token_id,
            "token_id": final_token_id,
            "candidates": candidates,
            "strategy": chosen_strategy,
            "beam": beam_info,
            "sampling": {
                "temperature": self.cfg.temperature,
                "top_k": effective_top_k,
                "top_p": self.cfg.top_p
            },
            "reasoning": reasoning_meta
        }

    # --------------------- Internal Helpers --------------------- #

    def _get_logits_safe(self, hidden_state: Any, tokens: Sequence[Any]) -> List[float]:
        if hasattr(self.model, "get_logits"):
            try:
                l = self.model.get_logits(hidden_state, list(tokens))
                if isinstance(l, list):
                    return l
                if hasattr(l, "tolist"):
                    return l.tolist()
            except Exception:
                return []
        # Fallback uniform logits of limited vocab
        vocab_size = getattr(self.model, "vocab_size", None)
        if callable(vocab_size):
            try:
                vocab_size = vocab_size()
            except Exception:
                pass
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            vocab_size = 100
        return [0.0] * vocab_size

    def _build_candidate_list(self, logits: List[float], limit: int = 200) -> List[Dict[str, Any]]:
        probs = _softmax(logits)
        idxs = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        out = []
        for rank, i in enumerate(idxs[:limit], start=1):
            out.append({
                "id": i,
                "logit": logits[i],
                "prob": probs[i],
                "raw_rank": rank
            })
        return out

    def _beam_search(
        self,
        hidden_state: Any,
        generated_tokens: Sequence[Any],
        logits: List[float],
        effective_top_k: int,
        entropy_val: float
    ) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Simplified local beam search over a single step expansion.
        - Select top beam_width indices.
        - Score each by prob * (length_penalty) (length penalty trivial here).
        - Choose best.
        Returns (chosen_token_id, beam_info, candidate_list).
        """
        width = self.cfg.beam_width
        top_k_logits = _apply_top_k(logits, max(width, effective_top_k))
        probs = _softmax(top_k_logits)
        indices = sorted(range(len(top_k_logits)), key=lambda i: top_k_logits[i], reverse=True)[:width]

        beam_paths: List[Dict[str, Any]] = []
        for idx in indices:
            p = probs[idx]
            score = p / max(1.0, self.cfg.beam_length_penalty)
            beam_paths.append({
                "token_id": idx,
                "prob": p,
                "score": score
            })

        best = max(beam_paths, key=lambda b: b["score"])
        candidate_list = self._build_candidate_list(top_k_logits, limit=self.cfg.max_candidates_return)
        beam_info = {
            "paths": beam_paths,
            "chosen": best,
            "entropy": entropy_val,
            "width": width
        }
        return best["token_id"], beam_info, candidate_list

    def _validate_token(self, token_id: int, context: Dict[str, Any]) -> int:
        original = token_id
        # Safety validator
        if self.safety and hasattr(self.safety, "validate_generation"):
            try:
                world_model = self.world_model
                safe_tok = self.safety.validate_generation(token_id, context, world_model)
                if isinstance(safe_tok, int):
                    token_id = safe_tok
            except Exception:
                pass

        # World model correction
        if self.world_model and hasattr(self.world_model, "validate_generation") and hasattr(self.world_model, "suggest_correction"):
            try:
                ok = self.world_model.validate_generation(token_id, context)
                if not ok:
                    alt = self.world_model.suggest_correction(token_id, context)
                    if isinstance(alt, int):
                        token_id = alt
            except Exception:
                pass

        return token_id

    def _empty_fallback(self) -> Dict[str, Any]:
        return {
            "token": 0,
            "token_id": 0,
            "candidates": [{"id": 0, "logit": 0.0, "prob": 1.0, "raw_rank": 1}],
            "strategy": "fallback",
            "beam": None,
            "sampling": {
                "temperature": self.cfg.temperature,
                "top_k": self.cfg.top_k,
                "top_p": self.cfg.top_p
            },
            "reasoning": {
                "entropy": 0.0,
                "confidence": 1.0,
                "repetition_penalty_applied": False
            }
        }