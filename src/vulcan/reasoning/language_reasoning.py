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

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Initialize logger for this module
logger = logging.getLogger(__name__)

# ----------------------------- Constants ----------------------------- #

# Default EOS token ID - configurable via LanguageReasoningConfig.eos_token_id
DEFAULT_EOS_TOKEN_ID = 0

# Maximum tokens to keep in history to prevent memory leaks
DEFAULT_MAX_TOKEN_HISTORY = 4096

# Minimum confidence threshold for early stopping
MIN_CONFIDENCE_THRESHOLD = 0.1

# Default validation confidence threshold
DEFAULT_VALIDATION_CONFIDENCE = 0.3

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
    beam_max_steps: int = 4  # beam expansion depth (local per-step)
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
    # FIX: Configurable EOS token ID - many tokenizers use different values (0, 2, 50256, etc.)
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    # FIX: Configurable set of EOS token IDs for models with multiple end tokens
    eos_token_ids: Set[int] = field(default_factory=lambda: {DEFAULT_EOS_TOKEN_ID})
    # FIX: Maximum token history to prevent memory leaks in long-running generation
    max_token_history: int = DEFAULT_MAX_TOKEN_HISTORY


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
    logits: List[float], generated: Sequence[Any], penalty: float, window: int
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


def _sample_index(
    filtered_logits: List[float], 
    temperature: float, 
    rng: Optional[random.Random] = None
) -> int:
    """
    Sample an index from filtered logits using temperature scaling.
    
    Args:
        filtered_logits: List of logit values
        temperature: Sampling temperature (0 = greedy, higher = more random)
        rng: Optional thread-safe Random instance. Uses global random if None.
        
    Returns:
        Selected index
    """
    if temperature <= 0:
        return max(range(len(filtered_logits)), key=lambda i: filtered_logits[i])
    scaled = [l / max(temperature, 1e-9) for l in filtered_logits]
    probs = _softmax(scaled)
    # FIX: Use provided RNG instance for thread safety
    r = rng.random() if rng else random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if cumulative >= r:
            return i
    return len(probs) - 1


# --------------------------- LanguageReasoning ---------------------------


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
        reranker: Optional[
            Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]
        ] = None,
        bias_hook: Optional[
            Callable[[List[float], Dict[str, Any]], List[float]]
        ] = None,
        mask_hook: Optional[
            Callable[[List[float], Dict[str, Any]], List[float]]
        ] = None,
        random_seed: Optional[int] = None,
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
        # FIX: Use thread-safe random.Random() instance instead of global random.seed()
        # This prevents race conditions in multi-threaded environments
        seed = self.cfg.seed if self.cfg.seed is not None else random_seed
        self._rng = random.Random(seed)

    # --------------------- Public Interface --------------------- #

    def generate(
        self,
        hidden_state: Any,
        generated_tokens: Sequence[Any],
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
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
            
        Raises:
            ValueError: If hidden_state is None or generated_tokens has invalid format
        """
        # FIX: Input validation to catch invalid inputs early
        if hidden_state is None:
            logger.warning("generate() called with None hidden_state")
            return self._empty_fallback()
        
        if generated_tokens is not None and not isinstance(generated_tokens, (list, tuple, Sequence)):
            logger.warning(f"generated_tokens has unexpected type: {type(generated_tokens)}")
            generated_tokens = []
        
        context = context or {}
        # Step 1: Acquire logits
        logits = self._get_logits_safe(hidden_state, generated_tokens)
        if not logits:
            return self._empty_fallback()
        
        # FIX: Validate logits size is reasonable
        if len(logits) == 0:
            logger.warning("Received empty logits from model")
            return self._empty_fallback()

        # Step 2: Post-process (mask / bias)
        # FIX: More specific exception handling for hooks
        if self.mask_hook:
            try:
                logits = self.mask_hook(logits, context)
            except (TypeError, ValueError) as e:
                # Expected errors from hook - log and continue
                logger.warning(f"Mask hook failed with expected error: {e}")
            except Exception as e:
                # Unexpected error - log as error with stack trace
                logger.error(f"Mask hook failed with unexpected error: {e}", exc_info=True)
        if self.bias_hook:
            try:
                logits = self.bias_hook(logits, context)
            except (TypeError, ValueError) as e:
                # FIX: More specific exception handling
                logger.warning(f"Bias hook failed with expected error: {e}")
            except Exception as e:
                # Unexpected error - log as error with stack trace
                logger.error(f"Bias hook failed with unexpected error: {e}", exc_info=True)

        # Step 3: Base probability distribution
        base_probs = _softmax(logits)
        ent = _entropy(base_probs)
        # Dynamic top-k shrink if enabled
        effective_top_k = self.cfg.top_k
        if self.cfg.dynamic_top_k and ent < self.cfg.strategy_auto_switch_entropy / 2:
            effective_top_k = max(
                self.cfg.dynamic_top_k_floor,
                int(effective_top_k * self.cfg.dynamic_top_k_decay),
            )

        # Step 4: Auto strategy switch if high entropy
        chosen_strategy = strategy or (
            "beam"
            if (self.cfg.use_beam_search or ent > self.cfg.strategy_auto_switch_entropy)
            else "sample"
        )

        # Step 5: Repetition penalty
        repetition_applied = False
        filtered_logits = logits[:]
        if self.cfg.apply_repetition_penalty:
            filtered_logits = _apply_repetition_penalty(
                filtered_logits,
                generated_tokens,
                self.cfg.repetition_penalty,
                self.cfg.repetition_window,
            )
            repetition_applied = filtered_logits != logits

        # Step 6: Branch by strategy
        if chosen_strategy == "greedy":
            token_id = max(
                range(len(filtered_logits)), key=lambda i: filtered_logits[i]
            )
            candidates = self._build_candidate_list(
                filtered_logits, limit=self.cfg.max_candidates_return
            )
            beam_info = None
        elif chosen_strategy == "beam":
            token_id, beam_info, candidates = self._beam_search(
                hidden_state, generated_tokens, filtered_logits, effective_top_k, ent
            )
        else:  # 'sample'
            temp = self.cfg.temperature
            top_k_logits = _apply_top_k(filtered_logits, effective_top_k)
            top_p_logits = _apply_top_p(top_k_logits, self.cfg.top_p)
            # FIX: Pass thread-safe RNG instance to _sample_index
            token_id = _sample_index(top_p_logits, temp, self._rng)
            candidates = self._build_candidate_list(
                top_p_logits, limit=self.cfg.max_candidates_return
            )
            beam_info = None

        # Step 7: Candidate reranking (optional)
        # FIX: More specific exception handling
        if self.reranker and candidates:
            try:
                candidates = self.reranker(
                    candidates, {"context": context, "entropy": ent}
                )
            except (TypeError, ValueError) as e:
                logger.warning(f"Candidate reranking failed with expected error: {e}")
            except Exception as e:
                logger.error(f"Candidate reranking failed with unexpected error: {e}", exc_info=True)

        # Step 8: Safety + world model validation/correction
        final_token_id = self._validate_token(token_id, context)

        # Step 9: Build reasoning meta
        chosen_prob = base_probs[token_id] if token_id < len(base_probs) else None
        confidence = chosen_prob
        reasoning_meta = {
            "entropy": ent,
            "confidence": confidence,
            "repetition_penalty_applied": repetition_applied,
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
                "top_p": self.cfg.top_p,
            },
            "reasoning": reasoning_meta,
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
            except Exception as e:
                # Logits extraction failed - log and return empty list
                logger.warning(f"Failed to extract logits: {e}")
                return []
        # Fallback uniform logits of limited vocab
        vocab_size = getattr(self.model, "vocab_size", None)
        if callable(vocab_size):
            try:
                vocab_size = vocab_size()
            except Exception as e:
                # Vocab size callable failed - log and use default
                logger.debug(f"Failed to get vocab size: {e}")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            vocab_size = 100
        return [0.0] * vocab_size

    def _build_candidate_list(
        self, logits: List[float], limit: int = 200
    ) -> List[Dict[str, Any]]:
        probs = _softmax(logits)
        idxs = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        out = []
        for rank, i in enumerate(idxs[:limit], start=1):
            out.append(
                {"id": i, "logit": logits[i], "prob": probs[i], "raw_rank": rank}
            )
        return out

    def _beam_search(
        self,
        hidden_state: Any,
        generated_tokens: Sequence[Any],
        logits: List[float],
        effective_top_k: int,
        entropy_val: float,
    ) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Simplified single-step beam search for token selection.
        
        NOTE: This is NOT full beam search that maintains multiple hypotheses
        across multiple generation steps. This is a simplified version that:
        - Selects top beam_width token candidates from current logits
        - Scores each by probability / length_penalty
        - Returns the highest-scoring token
        
        For proper multi-step beam search, use a dedicated beam search decoder
        that maintains beam hypotheses across the full generation sequence.
        
        Args:
            hidden_state: Current model hidden state (unused in this simplified version)
            generated_tokens: Previously generated tokens (unused in this simplified version)
            logits: Current logit distribution
            effective_top_k: Top-k filtering parameter
            entropy_val: Current entropy for metadata
            
        Returns:
            Tuple of (chosen_token_id, beam_info_dict, candidate_list)
        """
        width = self.cfg.beam_width
        top_k_logits = _apply_top_k(logits, max(width, effective_top_k))
        probs = _softmax(top_k_logits)
        indices = sorted(
            range(len(top_k_logits)), key=lambda i: top_k_logits[i], reverse=True
        )[:width]

        beam_paths: List[Dict[str, Any]] = []
        for idx in indices:
            p = probs[idx]
            score = p / max(1.0, self.cfg.beam_length_penalty)
            beam_paths.append({"token_id": idx, "prob": p, "score": score})

        best = max(beam_paths, key=lambda b: b["score"])
        candidate_list = self._build_candidate_list(
            top_k_logits, limit=self.cfg.max_candidates_return
        )
        beam_info = {
            "paths": beam_paths,
            "chosen": best,
            "entropy": entropy_val,
            "width": width,
            "note": "Simplified single-step beam selection, not full beam search",
        }
        return best["token_id"], beam_info, candidate_list

    def _validate_token(self, token_id: int, context: Dict[str, Any]) -> int:
        # Safety validator
        if self.safety and hasattr(self.safety, "validate_generation"):
            try:
                world_model = self.world_model
                safe_tok = self.safety.validate_generation(
                    token_id, context, world_model
                )
                if isinstance(safe_tok, int):
                    token_id = safe_tok
            except Exception as e:
                # Safety validation failure - log error but continue
                logger.error(f"Safety validation failed: {e}", exc_info=True)

        # World model correction
        if (
            self.world_model
            and hasattr(self.world_model, "validate_generation")
            and hasattr(self.world_model, "suggest_correction")
        ):
            try:
                ok = self.world_model.validate_generation(token_id, context)
                if not ok:
                    alt = self.world_model.suggest_correction(token_id, context)
                    if isinstance(alt, int):
                        token_id = alt
            except Exception as e:
                # World model validation/correction failure - log but continue
                logger.warning(f"World model validation/correction failed: {e}")

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
                "top_p": self.cfg.top_p,
            },
            "reasoning": {
                "entropy": 0.0,
                "confidence": 1.0,
                "repetition_penalty_applied": False,
            },
        }


# --------------------------- Reasoning Interface Wrapper --------------------------- #


class SimpleLanguageModel:
    """
    Simple heuristic language model for generating logits when no external model provided.
    Uses hash-based features from input to generate consistent but varied distributions.
    
    FIX: Uses thread-safe random.Random() instance instead of global random.seed()
    to prevent race conditions in multi-threaded environments.
    """

    def __init__(self, vocab_size: int = 1000, seed: Optional[int] = None):
        self.vocab_size = vocab_size
        self.seed = seed
        # FIX: Use dedicated Random instance for thread safety
        self._rng = random.Random(seed)

    def get_logits(self, hidden_state: Any, tokens_so_far: List[Any]) -> List[float]:
        """
        Generate logits based on hidden state and context.
        Uses deterministic hashing for consistency while maintaining thread safety.
        """
        # Create base distribution from hidden state hash
        state_hash = hash(str(hidden_state)) % 10000
        # FIX: Create temporary RNG seeded from state_hash for deterministic behavior
        # without modifying global random state
        temp_rng = random.Random(state_hash)

        # Generate base logits with some structure
        logits = []
        for i in range(self.vocab_size):
            # Mix hash-based and position-based components
            base = temp_rng.gauss(0, 1.5)
            position_bias = -0.1 * (
                i / self.vocab_size
            )  # Slight preference for earlier tokens
            logits.append(base + position_bias)

        # Apply context from tokens so far
        if tokens_so_far:
            # Boost logits for tokens that continue patterns
            context_hash = hash(tuple(tokens_so_far[-10:])) % self.vocab_size
            logits[context_hash % self.vocab_size] += 2.0
            logits[(context_hash + 1) % self.vocab_size] += 1.5
            logits[(context_hash - 1) % self.vocab_size] += 1.5

        return logits

class LanguageReasoner:
    """
    Reasoning interface wrapper for LanguageReasoning engine.
    Implements the standard reasoner interface expected by UnifiedReasoner.
    
    FIX: 
    - Uses ReasoningType.LANGUAGE instead of SYMBOLIC
    - Uses configurable EOS token IDs instead of hardcoded 0
    - Implements max_token_history to prevent memory leaks
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[LanguageReasoningConfig] = None,
        vocab_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize LanguageReasoner.

        Args:
            model: Optional external model with get_logits() method
            config: LanguageReasoningConfig for generation parameters
            vocab_size: Vocabulary size for default model
            **kwargs: Additional arguments passed to LanguageReasoning
        """
        # Use provided model or create simple default
        if model is None:
            model = SimpleLanguageModel(vocab_size=vocab_size)

        self.model = model
        self.config = config or LanguageReasoningConfig()
        self.engine = LanguageReasoning(model, config=self.config, **kwargs)
        self.generated_tokens: List[int] = []

    def _trim_token_history(self) -> None:
        """
        FIX: Trim token history to prevent unbounded memory growth.
        Keeps the most recent max_token_history tokens.
        """
        max_history = self.config.max_token_history
        if len(self.generated_tokens) > max_history:
            # Keep only the most recent tokens
            self.generated_tokens = self.generated_tokens[-max_history:]

    def _is_eos_token(self, token_id: int) -> bool:
        """
        FIX: Check if token is an end-of-sequence token.
        Uses configurable EOS token IDs instead of hardcoded assumption.
        
        Args:
            token_id: Token ID to check
            
        Returns:
            True if token is EOS, False otherwise
        """
        # Check against configured EOS token set
        if token_id in self.config.eos_token_ids:
            return True
        # Also check the single configured EOS token
        if token_id == self.config.eos_token_id:
            return True
        return False

    def reason(self, input_data: Any, query: Optional[Dict[str, Any]] = None) -> Any:
        """
        Main reasoning interface compatible with UnifiedReasoner.

        Args:
            input_data: Input to process (text, dict, etc.)
            query: Query dictionary with generation parameters

        Returns:
            ReasoningResult with generated text and metadata
        """
        # Import here to avoid circular dependency
        from .reasoning_types import ReasoningResult, ReasoningType

        query = query or {}

        # Extract generation parameters from query
        strategy = query.get("strategy", "sample")
        max_tokens = query.get("max_tokens", 50)
        temperature = query.get("temperature", self.config.temperature)

        # Update config if temperature specified
        if temperature != self.config.temperature:
            self.config.temperature = temperature

        # Create context from input_data and query
        context = {
            "input_data": input_data,
            "query": query.get("query", ""),
            "task": query.get("task", "generation"),
        }

        # Generate tokens iteratively
        generated_text_tokens = []
        total_confidence = 0.0
        generation_steps = []

        for step in range(max_tokens):
            # FIX: Trim history before generation to prevent memory leak
            self._trim_token_history()
            
            # Generate next token
            result = self.engine.generate(
                hidden_state=context,
                generated_tokens=self.generated_tokens,
                context=context,
                strategy=strategy,
            )

            token_id = result["token_id"]
            confidence = result["reasoning"].get("confidence", 0.5)

            # Track generated tokens
            self.generated_tokens.append(token_id)
            generated_text_tokens.append(token_id)
            total_confidence += confidence if confidence else 0.5

            # Store step metadata
            generation_steps.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "confidence": confidence,
                    "strategy": result["strategy"],
                    "entropy": result["reasoning"].get("entropy", 0.0),
                }
            )

            # FIX: Early stopping with configurable conditions
            # Check for low confidence (including 0.0 which is falsy but should trigger)
            if confidence is not None and confidence < MIN_CONFIDENCE_THRESHOLD:
                logger.debug(f"Early stopping: low confidence ({confidence:.3f})")
                break
            # FIX: Use configurable EOS token check instead of hardcoded 0
            if self._is_eos_token(token_id):
                logger.debug(f"Early stopping: EOS token ({token_id})")
                break

        # Calculate average confidence
        avg_confidence = (
            total_confidence / len(generated_text_tokens)
            if generated_text_tokens
            else 0.0
        )

        # Create human-readable conclusion
        if isinstance(input_data, str):
            conclusion = f"Generated response to: '{input_data[:100]}...'"
        elif isinstance(query.get("query"), str):
            conclusion = f"Generated response to query: '{query['query'][:100]}...'"
        else:
            conclusion = "Generated language response"

        # Add token sequence information
        conclusion += f" (generated {len(generated_text_tokens)} tokens)"

        # Build explanation
        explanation = (
            f"Neural language generation using {strategy} strategy. "
            f"Generated {len(generated_text_tokens)} tokens with average confidence {avg_confidence:.3f}. "
            f"Temperature: {temperature}, Top-k: {self.config.top_k}, Top-p: {self.config.top_p}"
        )

        # Create metadata
        metadata = {
            "generated_tokens": generated_text_tokens,
            "generation_steps": generation_steps,
            "strategy": strategy,
            "avg_entropy": (
                sum(s["entropy"] for s in generation_steps) / len(generation_steps)
                if generation_steps
                else 0.0
            ),
            "total_tokens": len(generated_text_tokens),
            "sampling_params": {
                "temperature": temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            },
        }

        # FIX: Use ReasoningType.LANGUAGE instead of SYMBOLIC
        return ReasoningResult(
            conclusion=conclusion,
            confidence=float(avg_confidence),
            reasoning_type=ReasoningType.LANGUAGE,  # FIX: Language generation is its own type
            explanation=explanation,
            metadata=metadata,
        )

    def reset(self):
        """Reset generation state (clear token history)"""
        self.generated_tokens.clear()
