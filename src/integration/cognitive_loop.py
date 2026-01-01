from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
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

# Optional numpy import for vectorized operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Initialize logger for this module
logger = logging.getLogger(__name__)


# =========================== CHECKPOINT HELPER =========================== #
# Helper class for consistent checkpoint logging during token generation.
# Logs elapsed time since generation started to help diagnose hang locations.


class _Checkpoint:
    """Checkpoint helper for consistent timing and logging during token generation.
    
    Usage:
        _cp = _Checkpoint()
        _cp.reset()  # Call at start of generate()
        _cp.log("Starting tokenization...")
        tokens = await self._tokenize(prompt)
        _cp.log(f"Tokenization done: {len(tokens)} tokens")
    
    All logs are at INFO level so they appear in production logs.
    """
    
    def __init__(self):
        self.start: float = 0.0  # Will be set by reset()
    
    def reset(self):
        """Reset the checkpoint timer. Call at start of generate()."""
        self.start = time.time()
    
    def log(self, msg: str):
        """Log a checkpoint message with elapsed time.
        
        If reset() wasn't called, logs a warning and uses current time as reference.
        """
        if self.start == 0.0:
            logger.warning("[CHECKPOINT] reset() not called, using current time as reference")
            self.start = time.time()
        elapsed = time.time() - self.start
        logger.info(f"[CHECKPOINT {elapsed:.3f}s] {msg}")


# Global checkpoint instance for token generation
_cp = _Checkpoint()


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
    max_tokens: int = 2000  # Increased for diagnostic purposes (was 128)
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
    # PERFORMANCE: New options to control logging and caching behavior
    enable_verbose_logging: bool = False  # Enable detailed [DIAG] logging (default: False for performance)
    log_interval: int = 10  # Only log diagnostic messages every N steps
    context_cache_interval: int = 10  # Refresh context cache every N steps
    aggressive_cache_threshold: int = 10  # Step after which to use aggressive caching
    aggressive_context_cache_interval: int = 20  # Context cache interval after warmup
    world_model_update_interval: int = 5  # Update world model every N steps
    async_operation_timeout_seconds: float = 10.0  # Timeout for individual async operations in _async_safe()
    # OUTPUT FORMATTING MODE: When True, skip reasoning hooks for faster output generation
    # This is used when VULCAN's reasoning is already complete and we just need prose output.
    # Key architecture insight: The LLM shouldn't reason - VULCAN's mind (orchestrator, 
    # agents, symbolic systems) already did that work. The LLM should just format output.
    output_formatting_mode: bool = False  # Default False for backwards compatibility


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
    """Compute softmax probabilities. Uses numpy if available for better performance."""
    if not xs:
        return []
    if HAS_NUMPY:
        arr = np.array(xs, dtype=np.float64)
        arr = arr - np.max(arr)  # Numerical stability
        exp_arr = np.exp(arr)
        sum_exp = np.sum(exp_arr)
        if sum_exp == 0:
            return [1.0 / len(xs)] * len(xs)
        return (exp_arr / sum_exp).tolist()
    # Fallback to pure Python
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s == 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def apply_top_k(logits: List[float], k: int) -> List[float]:
    """Apply top-k filtering. Uses numpy if available for O(n) performance."""
    if k <= 0 or k >= len(logits):
        return logits
    if HAS_NUMPY:
        arr = np.array(logits, dtype=np.float64)
        # Use argpartition for O(n) instead of O(n log n) sort
        top_k_indices = np.argpartition(arr, -k)[-k:]
        kth_value = np.min(arr[top_k_indices])
        result = np.where(arr >= kth_value, arr, float("-inf"))
        return result.tolist()
    # Fallback to pure Python
    indexed_logits = [(logits[i], i) for i in range(len(logits))]
    kth_value = sorted(indexed_logits, key=lambda x: x[0], reverse=True)[k - 1][0]
    return [l if l >= kth_value else float("-inf") for l in logits]


def apply_top_p(logits: List[float], p: float) -> List[float]:
    """Apply nucleus (top-p) sampling. Uses numpy for vectorized operations.
    
    Note: This function computes probabilities to determine which tokens to keep
    (those that together have cumulative probability mass >= p), but returns
    the original logits for the kept tokens (not probabilities). This is the
    correct behavior for nucleus sampling.
    """
    if p >= 1.0:
        return logits
    if HAS_NUMPY:
        arr = np.array(logits, dtype=np.float64)
        probs = np.exp(arr - np.max(arr))
        probs = probs / np.sum(probs)
        sorted_indices = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[sorted_indices])
        # Find cutoff
        cutoff_idx = np.searchsorted(cumulative, p) + 1
        keep_indices = set(sorted_indices[:cutoff_idx].tolist())
        # Return original logits for kept indices, -inf for filtered ones
        result = np.where(
            np.isin(np.arange(len(arr)), list(keep_indices)),
            arr,  # Original logits for kept tokens
            float("-inf")
        )
        return result.tolist()
    # Fallback to pure Python
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
        out[idx] = logits[idx]  # Original logits
    return out


def penalize_repetition(
    logits: List[float], generated: Tokens, penalty: float, window: int
) -> List[float]:
    """Apply repetition penalty. Vectorized with numpy when possible."""
    if penalty <= 1.0 or not generated:
        return logits
    rec = generated[-window:] if window > 0 else generated
    if not all(isinstance(t, int) for t in rec):
        return logits
    
    if HAS_NUMPY:
        arr = np.array(logits, dtype=np.float64)
        counts = np.zeros(len(arr), dtype=np.int32)
        for t in rec:
            if 0 <= t < len(counts):
                counts[t] += 1
        # Apply penalty where counts > 0
        mask = counts > 0
        # For positive logits, divide by penalty; for negative, multiply
        arr = np.where(
            mask & (arr > 0),
            arr / penalty,
            np.where(mask, arr * penalty, arr)
        )
        return arr.tolist()
    
    # Fallback to pure Python
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
    """Sample a token from logits. Uses numpy for efficient sampling."""
    if not logits:
        return 0
    if temperature <= 0:
        if HAS_NUMPY:
            return int(np.argmax(logits))
        return max(range(len(logits)), key=lambda i: logits[i])
    
    if HAS_NUMPY:
        arr = np.array(logits, dtype=np.float64) / max(temperature, 1e-9)
        arr = arr - np.max(arr)  # Numerical stability
        probs = np.exp(arr)
        probs = probs / np.sum(probs)
        # Use numpy's random choice for efficient sampling
        return int(np.random.choice(len(probs), p=probs))
    
    # Fallback to pure Python
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
    """Compute entropy of probability distribution."""
    if HAS_NUMPY:
        arr = np.array(probs, dtype=np.float64)
        arr = arr[arr > 0]  # Filter out zeros
        return float(-np.sum(arr * np.log2(arr)))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy += -p * math.log(p, 2)
    return entropy


# =========================== PERFORMANCE CACHES =========================== #


class EncodingCache:
    """LRU cache for transformer encoding results with TTL support.
    
    This cache stores encoded hidden states to avoid redundant transformer calls
    for similar token sequences. It uses a hash of the token sequence as the key.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, tokens: Tokens) -> tuple:
        """Compute a cache key for the token sequence.
        
        Uses tuple directly for hashing (faster than MD5 string conversion).
        Only uses first/last tokens plus length for large sequences.
        """
        # Use first and last 50 tokens plus length for efficient hashing
        if len(tokens) <= 100:
            # Use tuple directly - hashable and efficient
            return tuple(tokens)
        else:
            # For long sequences, use fingerprint tuple
            key_tokens = tokens[:50] + tokens[-50:]
            return (tuple(key_tokens), len(tokens))
    
    def get(self, tokens: Tokens) -> Optional[Any]:
        """Get cached encoding result if available and not expired."""
        key = self._compute_key(tokens)
        with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._timestamps[key] < self.ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]
            self._misses += 1
            return None
    
    def put(self, tokens: Tokens, value: Any) -> None:
        """Store encoding result in cache."""
        key = self._compute_key(tokens)
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                if oldest_key in self._timestamps:
                    del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class LogitsCache:
    """Cache for logits computations on similar patterns.
    
    This cache stores logits results for repeated token patterns,
    reducing redundant computation for common sequences.
    """
    
    def __init__(self, max_size: int = 500, ttl_seconds: float = 30.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[tuple, float] = {}
        self._lock = threading.RLock()
    
    def _compute_key(self, tokens: Tokens, context_hash: Optional[str] = None) -> tuple:
        """Compute cache key from last N tokens and optional context.
        
        Uses tuple for efficient hashing instead of MD5.
        """
        # Use last 20 tokens for key (sufficient for pattern matching)
        key_tokens = tokens[-20:] if len(tokens) > 20 else tokens
        if context_hash:
            return (tuple(key_tokens), context_hash)
        return tuple(key_tokens)
    
    def get(self, tokens: Tokens, context_hash: Optional[str] = None) -> Optional[List[float]]:
        """Get cached logits if available."""
        key = self._compute_key(tokens, context_hash)
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self.ttl_seconds:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
            return None
    
    def put(self, tokens: Tokens, logits: List[float], context_hash: Optional[str] = None) -> None:
        """Store logits in cache."""
        key = self._compute_key(tokens, context_hash)
        with self._lock:
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                if oldest_key in self._timestamps:
                    del self._timestamps[oldest_key]
            self._cache[key] = logits
            self._timestamps[key] = time.time()


class SamplingTableCache:
    """Pre-computed sampling tables for common vocabulary distributions.
    
    This cache stores pre-computed cumulative distribution functions (CDFs)
    for efficient token sampling without repeated softmax computation.
    """
    
    def __init__(self, vocab_size: int = 50257, common_patterns: int = 100):
        self.vocab_size = vocab_size
        self.common_patterns = common_patterns
        self._tables: Dict[str, Tuple[List[int], List[float]]] = {}
        self._lock = threading.RLock()
    
    def get_or_compute(
        self, logits: List[float], temperature: float, top_k: int
    ) -> Tuple[List[int], List[float]]:
        """Get pre-computed sampling table or compute and cache it.
        
        Returns:
            Tuple of (sorted_indices, cumulative_probs) for efficient sampling.
        """
        # Create a comprehensive fingerprint for the logits distribution
        # Use statistical moments across full array for better coverage
        if HAS_NUMPY:
            arr = np.array(logits, dtype=np.float32)
            # PERFORMANCE FIX: Handle NaN/Inf values to prevent numerical instability
            # This addresses RuntimeWarning: invalid value encountered in subtract
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            # Compute moments across full array
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            # Add min/max for distribution shape
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            # Add skewness approximation (for distribution asymmetry)
            median_val = float(np.median(arr))
            fingerprint = (
                f"{mean_val:.3f}_{std_val:.3f}_{min_val:.3f}_{max_val:.3f}_"
                f"{median_val:.3f}_{temperature:.2f}_{top_k}_{len(logits)}"
            )
        else:
            # Pure Python fallback
            if logits:
                mean_val = sum(logits) / len(logits)
                sorted_logits = sorted(logits)
                min_val = sorted_logits[0]
                max_val = sorted_logits[-1]
                median_val = sorted_logits[len(sorted_logits) // 2]
                fingerprint = f"{mean_val:.3f}_{min_val:.3f}_{max_val:.3f}_{median_val:.3f}_{temperature:.2f}_{top_k}_{len(logits)}"
            else:
                fingerprint = f"empty_{temperature:.2f}_{top_k}"
        
        with self._lock:
            if fingerprint in self._tables:
                return self._tables[fingerprint]
        
        # Compute new table
        filtered = apply_top_k(logits, top_k)
        scaled = [l / max(temperature, 1e-9) if l > float("-inf") else l for l in filtered]
        probs = softmax(scaled)
        
        # Create sorted (index, cumprob) for efficient sampling
        indexed = [(i, p) for i, p in enumerate(probs) if p > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        
        sorted_indices = [x[0] for x in indexed]
        cumulative = []
        cum = 0.0
        for _, p in indexed:
            cum += p
            cumulative.append(cum)
        
        result = (sorted_indices, cumulative)
        
        # Cache if we have room
        with self._lock:
            if len(self._tables) < self.common_patterns:
                self._tables[fingerprint] = result
        
        return result


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
        
        # ========== PERFORMANCE OPTIMIZATIONS ==========
        # Context caching - context and world model don't change much per token
        self._cached_context: Optional[Dict[str, Any]] = None
        self._context_cache_step: int = -1
        # PERF: Use config values for cache intervals
        self._world_model_update_interval: int = self.runtime.world_model_update_interval
        
        # Encoding cache for transformer outputs (reduces redundant encode calls)
        self._encoding_cache = EncodingCache(max_size=500, ttl_seconds=60.0)
        
        # Logits cache for repeated patterns (reduces redundant logits computation)
        self._logits_cache = LogitsCache(max_size=300, ttl_seconds=30.0)
        
        # Pre-computed sampling tables for efficient token selection
        vocab_size = getattr(transformer, 'vocab_size', 50257) if transformer else 50257
        self._sampling_table_cache = SamplingTableCache(vocab_size=vocab_size, common_patterns=100)
        
        # PERF: Use config values for aggressive caching thresholds
        self._aggressive_cache_threshold: int = self.runtime.aggressive_cache_threshold
        self._context_cache_interval: int = self.runtime.context_cache_interval
        self._aggressive_context_cache_interval: int = self.runtime.aggressive_context_cache_interval
        
        # Token index cache for fast lookups (OrderedDict for LRU eviction)
        self._token_index_cache: OrderedDict = OrderedDict()
        self._token_index_cache_lock = threading.RLock()
        self._token_index_cache_max_size: int = 10000  # Configurable
        
        # Performance metrics
        self._perf_metrics = {
            "total_encode_time_ms": 0.0,
            "total_sample_time_ms": 0.0,
            "total_context_time_ms": 0.0,
            "encoding_cache_hits": 0,
            "logits_cache_hits": 0,
            "tokens_generated": 0,
        }

    def set_speculative_function(self, fn: SpeculativeFunction) -> None:
        self._speculative_fn = fn

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.runtime.max_parallel_workers
            )
        return self._executor

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring and optimization.
        
        Returns:
            Dictionary containing cache hit rates, timing metrics, etc.
        """
        return {
            "encoding_cache": self._encoding_cache.get_stats(),
            "perf_metrics": self._perf_metrics.copy(),
            "context_cache_step": self._context_cache_step,
            "world_model_update_interval": self._world_model_update_interval,
        }

    def clear_caches(self) -> None:
        """Clear all performance caches. Useful when model state changes."""
        self._encoding_cache.clear()
        self._logits_cache._cache.clear()
        self._logits_cache._timestamps.clear()
        with self._token_index_cache_lock:
            self._token_index_cache.clear()
        self._cached_context = None
        self._context_cache_step = -1

    # ---------------------- PUBLIC ENTRY ---------------------- #

    async def generate(
        self,
        prompt: Union[str, Tokens],
        max_tokens: Optional[int] = None,
        stream_callback: Optional[Callable[[Token, str, Dict[str, Any]], None]] = None,
        stop_tokens: Optional[Tuple[Token, ...]] = None,
        stop_strings: Optional[Tuple[str, ...]] = None,
    ) -> Union[AsyncGenerator[Dict[str, Any], None], CognitiveLoopResult]:
        # Reset checkpoint timer at the start of each generation
        _cp.reset()
        prompt_len = len(prompt) if isinstance(prompt, str) else len(prompt)
        _cp.log(f"generate() START: prompt_len={prompt_len}, max_tokens={max_tokens or self.sampling.max_tokens}")
        
        start = time.time()
        max_steps = max_tokens or self.sampling.max_tokens
        stop_tok_set = set(stop_tokens or ()) | set(self.sampling.stop_tokens)
        stop_str_patterns = tuple(stop_strings or ()) + self.sampling.stop_strings

        # CRITICAL FIX: Add timeout to tokenization to prevent hang during generate() setup
        # Tokenization should complete in a few seconds at most
        TOKENIZE_TIMEOUT = 10.0  # 10 seconds max for tokenization
        
        if isinstance(prompt, str):
            try:
                _cp.log("Calling _tokenize()...")
                init_tokens = await asyncio.wait_for(
                    self._tokenize(prompt),
                    timeout=TOKENIZE_TIMEOUT
                )
                _cp.log(f"_tokenize() done: {len(init_tokens)} tokens")
            except asyncio.TimeoutError:
                _cp.log(f"_tokenize() TIMEOUT after {TOKENIZE_TIMEOUT}s! Using word split fallback.")
                logger.error(
                    f"[CognitiveLoop] Tokenization timed out after {TOKENIZE_TIMEOUT}s! "
                    f"Falling back to simple word split. Prompt length: {len(prompt)}"
                )
                # Fallback to simple word splitting to avoid complete failure
                # Note: Token type is Union[int, str], so string tokens from word split are valid
                # This matches the existing fallback behavior in _tokenize() method
                init_tokens = list(prompt.split())
        else:
            init_tokens = prompt[:]
            _cp.log(f"Using pre-tokenized input: {len(init_tokens)} tokens")

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
        # Reset caches for new generation
        self._cached_context = None
        self._context_cache_step = -1

        temperature = self.sampling.temperature
        top_k = self.sampling.top_k
        
        _cp.log(f"Creating _generator() async generator: max_steps={max_steps}")

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
            # DIAGNOSTIC FIX: Log generator start at INFO level so we can see it in production logs
            logger.info(
                f"[DIAG] Generator starting: max_steps={max_steps}, enable_stream={self.runtime.enable_stream}"
            )

            for step in range(max_steps):
                # DIAGNOSTIC FIX: Log step 0 at INFO level (critical for diagnosing first token hang)
                # Other steps are logged at DEBUG level to reduce noise
                if step == 0:
                    logger.info(
                        f"[DIAG] Generator loop step={step} (FIRST TOKEN), generated_len={len(generated)}"
                    )
                elif step % 10 == 0:
                    logger.debug(
                        f"[DIAG] Generator loop step={step}, generated_len={len(generated)}"
                    )

                if self.sampling.use_beam_search and all(
                    b["is_finished"] for b in self._beam_state
                ):
                    logger.debug("[DIAG] Generator: All beams finished, breaking")
                    break

                if (
                    self.runtime.time_budget_seconds
                    and (time.time() - start) >= self.runtime.time_budget_seconds
                ):
                    logger.debug("[DIAG] Generator: Time budget exceeded, finalizing")
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
                    # DIAGNOSTIC FIX: Log step 0 call at INFO level for diagnosing first token hang
                    if step == 0:
                        logger.info(f"[DIAG] Generator: Calling _step({step}) for FIRST TOKEN...")
                    elif step % 10 == 0:
                        logger.debug(f"[DIAG] Generator: Calling _step({step})...")
                    step_result = await self._step(
                        prompt_tokens=init_tokens + generated,
                        temperature=temperature,
                        top_k=top_k,
                        stop_tokens=stop_tok_set,
                        stop_strings=stop_str_patterns,
                        step=step,
                    )
                    # DIAGNOSTIC FIX: Log step 0 completion at INFO level
                    if step == 0:
                        logger.info(f"[DIAG] Generator: _step({step}) completed for FIRST TOKEN")
                    errors = 0
                except Exception as e:
                    print(f"Error during cognitive step: {e}\n{traceback.format_exc()}")
                    logger.error(f"[DIAG] Generator: _step({step}) EXCEPTION: {e}")
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
                    # PERF: Reduced per-token logging to debug level
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
                    # PERF: Removed else branch logging

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
                else:
                    # PERF: Changed to debug level
                    logger.debug(
                        f"[DIAG] Generator: Token is None! Step will continue without yielding"
                    )

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

    async def generate_fast(
        self,
        prompt: Union[str, Tokens],
        max_tokens: Optional[int] = None,
        stream_callback: Optional[Callable[[Token, str, Dict[str, Any]], None]] = None,
        stop_tokens: Optional[Tuple[Token, ...]] = None,
        stop_strings: Optional[Tuple[str, ...]] = None,
    ) -> Union[AsyncGenerator[Dict[str, Any], None], CognitiveLoopResult]:
        """
        Fast generation mode that bypasses reasoning hooks.
        
        This method is specifically designed for OUTPUT FORMATTING when VULCAN's
        reasoning has already completed. It provides significantly faster token
        generation by skipping:
        - bridge.before_execution() (reasoning hook)
        - world_model.update() (reasoning hook)
        
        ARCHITECTURE INSIGHT:
        Neither GraphixVulcanLLM nor OpenAI is "the mind." They're output formatters.
        VULCAN's mind (orchestrator, agents, symbolic systems) already did its reasoning
        work before this LLM is invoked. This method acknowledges that architecture.
        
        PERFORMANCE:
        - Before: ~2400ms for first token (transformer + reasoning + world model)
        - After: ~500ms for first token (transformer only)
        - 30-token response: ~15s instead of TIMEOUT
        
        Args:
            prompt: Input text or tokens to generate from
            max_tokens: Maximum tokens to generate
            stream_callback: Optional callback for streaming tokens
            stop_tokens: Tokens that stop generation
            stop_strings: Strings that stop generation
            
        Returns:
            AsyncGenerator in streaming mode, or CognitiveLoopResult when complete
        """
        # Enable output formatting mode (skips reasoning hooks)
        original_mode = self.runtime.output_formatting_mode
        self.runtime.output_formatting_mode = True
        
        try:
            logger.info("[CognitiveLoop] generate_fast() - OUTPUT FORMATTING MODE enabled")
            return await self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                stream_callback=stream_callback,
                stop_tokens=stop_tokens,
                stop_strings=stop_strings,
            )
        finally:
            # Restore original mode
            self.runtime.output_formatting_mode = original_mode

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
        
        # DIAG: Log entry into _step for debugging hangs
        # DIAGNOSTIC FIX: Use INFO level for step 0 so it always appears in production logs
        is_first_token = (step == 0)
        should_log = is_first_token or (step % self.runtime.log_interval == 0) or self.runtime.enable_verbose_logging
        if is_first_token:
            _cp.log(f"_step({step}) START (FIRST TOKEN), prompt_tokens_len={len(prompt_tokens)}")
            logger.info(f"[DIAG] _step({step}) STARTED (FIRST TOKEN), prompt_tokens_len={len(prompt_tokens)}")
        elif should_log:
            logger.debug(f"[DIAG] _step({step}) started, prompt_tokens_len={len(prompt_tokens)}")

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
            # OUTPUT FORMATTING MODE: Skip reasoning hooks for faster generation
            # When output_formatting_mode is True, VULCAN's reasoning is already complete.
            # The LLM should just format output, not reason. This fixes the timeout issue
            # where reasoning was being called on EVERY TOKEN when it already happened.
            if self.runtime.output_formatting_mode:
                # Fast path: Skip bridge.before_execution() and world_model.update()
                # These are reasoning hooks that add ~5ms+ overhead per token
                retrieved_context = self._cached_context or {}
                sub_times["context_retrieval_ms"] = 0.0
                sub_times["wm_update_ms"] = 0.0
                if is_first_token:
                    logger.info("[DIAG] Step 0: OUTPUT_FORMATTING_MODE - skipping reasoning hooks (fast path)")
            else:
                # OPTIMIZATION: Aggressive context caching strategy
                # - Before warmup (step < _aggressive_cache_threshold): cache every _context_cache_interval steps
                # - After warmup: cache even less frequently (_aggressive_context_cache_interval steps)
                cache_interval = (
                    self._aggressive_context_cache_interval 
                    if step >= self._aggressive_cache_threshold 
                    else self._context_cache_interval
                )
                
                should_refresh_context = (
                    self._cached_context is None
                    or step == 0
                    or (step - self._context_cache_step) >= cache_interval
                )
                
                if should_refresh_context:
                    t_ctx = time.time()
                    if is_first_token:
                        _cp.log("Calling bridge.before_execution...")
                        logger.info("[DIAG] Step 0: Calling bridge.before_execution...")
                    retrieved_context = await self._async_safe(
                        self.bridge.before_execution, {"prompt_tokens": prompt_tokens}, {}
                    )
                    if is_first_token:
                        _cp.log(f"bridge.before_execution() done in {(time.time() - t_ctx)*1000:.1f}ms")
                        logger.info(f"[DIAG] Step 0: bridge.before_execution completed in {(time.time() - t_ctx)*1000:.1f}ms")
                    ctx_time = (time.time() - t_ctx) * 1000
                    sub_times["context_retrieval_ms"] = ctx_time
                    self._perf_metrics["total_context_time_ms"] += ctx_time
                    # Cache the context for reuse
                    self._cached_context = retrieved_context
                    self._context_cache_step = step
                    # PERF: Only log context retrieval at debug level
                    logger.debug(
                        f"[PERF] Context retrieved in {ctx_time:.1f}ms (step {step})"
                    )
                else:
                    # Reuse cached context
                    retrieved_context = self._cached_context
                    sub_times["context_retrieval_ms"] = 0.0

                # OPTIMIZATION: Adaptive world model update frequency
                # Increase interval after warmup for better performance
                wm_update_interval = (
                    self._world_model_update_interval * 2 
                    if step >= self._aggressive_cache_threshold 
                    else self._world_model_update_interval
                )
                
                if self.runtime.attach_world_model_hooks and hasattr(
                    self.bridge, "world_model"
                ):
                    if step == 0 or step % wm_update_interval == 0:
                        try:
                            t_wm_up = time.time()
                            if is_first_token:
                                logger.info("[DIAG] Step 0: Calling world_model.update...")
                            await self._async_safe(
                                self.bridge.world_model.update,
                                {"tokens": prompt_tokens},
                                None,
                            )
                            sub_times["wm_update_ms"] = (time.time() - t_wm_up) * 1000
                            if is_first_token:
                                logger.info(f"[DIAG] Step 0: world_model.update completed in {sub_times['wm_update_ms']:.1f}ms")
                            else:
                                logger.debug(
                                    f"[PERF] World model updated in {sub_times['wm_update_ms']:.1f}ms"
                                )
                        except Exception as e:
                            # World model update is optional; log errors but continue
                            logger.debug(f"World model update failed: {e}")
                    else:
                        sub_times["wm_update_ms"] = 0.0
            token_info["retrieved_context"] = retrieved_context

            t_enc = time.time()
            
            # OPTIMIZATION: Check encoding cache first
            cached_hidden = self._encoding_cache.get(prompt_tokens)
            if cached_hidden is not None:
                hidden_state = cached_hidden
                sub_times["encode_ms"] = 0.1  # Cache hit
                sub_times["encode_cache_hit"] = True
                self._perf_metrics["encoding_cache_hits"] += 1
                if is_first_token:
                    _cp.log("transformer.encode() cache HIT")
                    logger.info("[DIAG] Step 0: Encoding cache HIT")
            else:
                if is_first_token:
                    _cp.log("Calling transformer.encode()...")
                    logger.info("[DIAG] Step 0: Calling transformer.encode...")
                hidden_state = await self._async_safe(
                    self.transformer.encode, prompt_tokens, None
                )
                # Store in cache for future use
                if hidden_state is not None:
                    self._encoding_cache.put(prompt_tokens, hidden_state)
                sub_times["encode_ms"] = (time.time() - t_enc) * 1000
                sub_times["encode_cache_hit"] = False
                if is_first_token:
                    _cp.log(f"transformer.encode() done in {sub_times['encode_ms']:.1f}ms")
                    logger.info(f"[DIAG] Step 0: transformer.encode completed in {sub_times['encode_ms']:.1f}ms")
                
            self._perf_metrics["total_encode_time_ms"] += sub_times["encode_ms"]
            # PERF: Only log slow encodes (>50ms)
            if sub_times["encode_ms"] > 50.0:
                logger.debug(
                    f"[PERF] Slow encode: {sub_times['encode_ms']:.1f}ms (step {step})"
                )
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
            
            # OPTIMIZATION: Check logits cache first
            cached_logits = self._logits_cache.get(prompt_tokens)
            if cached_logits is not None:
                logits = cached_logits
                sub_times["get_logits_ms"] = 0.1  # Cache hit
                sub_times["logits_cache_hit"] = True
                self._perf_metrics["logits_cache_hits"] += 1
                if is_first_token:
                    _cp.log("_obtain_logits() cache HIT")
                    logger.info("[DIAG] Step 0: Logits cache HIT")
            else:
                if is_first_token:
                    _cp.log("Calling _obtain_logits()...")
                    logger.info("[DIAG] Step 0: Calling _obtain_logits...")
                logits = await self._obtain_logits(hidden_state, prompt_tokens, candidates)
                # Store in cache for future use
                if logits:
                    self._logits_cache.put(prompt_tokens, logits)
                sub_times["get_logits_ms"] = (time.time() - t_logits) * 1000
                sub_times["logits_cache_hit"] = False
                if is_first_token:
                    logits_len = len(logits) if logits else 0
                    _cp.log(f"_obtain_logits() done in {sub_times['get_logits_ms']:.1f}ms (len={logits_len})")
                    logger.info(f"[DIAG] Step 0: _obtain_logits completed in {sub_times['get_logits_ms']:.1f}ms (len={logits_len})")
            
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
                if is_first_token:
                    _cp.log("Sampling token...")
                    logger.info("[DIAG] Step 0: Sampling token...")
                chosen_index, adjusted_logits = self._sample_optimized(
                    logits=logits,
                    generated_tokens=prompt_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=self.sampling.top_p,
                )
                token = await self._index_to_token_cached(chosen_index)
                sub_times["sample_ms"] = (time.time() - t_sample) * 1000
                self._perf_metrics["total_sample_time_ms"] += sub_times["sample_ms"]
                self._perf_metrics["tokens_generated"] += 1
                if is_first_token:
                    _cp.log(f"Sampling done in {sub_times['sample_ms']:.1f}ms, token={token}")
                    logger.info(f"[DIAG] Step 0: Sampling completed in {sub_times['sample_ms']:.1f}ms, token={token}")

            token_info["chosen_index"] = chosen_index

        reasoning_meta["selected_token"] = token
        # PERF: Only log slow steps (>100ms) at debug level
        total_step_ms = (time.time() - t0) * 1000
        if total_step_ms > 100.0:
            logger.debug(
                f"[PERF] Slow step: {total_step_ms:.1f}ms (step {step})"
            )

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
        
        # DIAG: Log completion of step 0 at INFO level for production visibility
        if is_first_token:
            total_step_time = (time.time() - t0) * 1000
            logger.info(f"[DIAG] Step 0 COMPLETED: token={token}, total_time={total_step_time:.1f}ms")
        
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
        # Get vocab_size - handle both property/attribute and method cases
        vocab_size = getattr(self.transformer, "vocab_size", None)
        if callable(vocab_size):
            # It's a method, call it
            try:
                vocab_size = vocab_size()
            except Exception as e:
                logger.warning(f"Failed to call vocab_size method: {e}")
                vocab_size = None
        # Ensure vocab_size is an integer - handle float values that represent whole numbers
        if vocab_size is not None and not isinstance(vocab_size, int):
            try:
                if isinstance(vocab_size, float) and vocab_size.is_integer():
                    # Float that represents a whole number (e.g., 50257.0)
                    vocab_size = int(vocab_size)
                else:
                    vocab_size = int(vocab_size)
            except (TypeError, ValueError, AttributeError):
                logger.warning(f"Could not convert vocab_size to int: {type(vocab_size).__name__}")
                vocab_size = None
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

    def _sample_optimized(
        self,
        logits: List[float],
        generated_tokens: Tokens,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[int, List[float]]:
        """Optimized sampling with pre-computed tables and vectorized operations.
        
        This method uses numpy vectorization and pre-computed sampling tables
        for faster token selection. Falls back to standard _sample if optimization
        is not beneficial.
        """
        if not logits:
            return 0, logits
        
        # Use vectorized operations for filtering
        filtered = apply_top_k(logits, top_k)
        filtered = apply_top_p(filtered, top_p)
        
        if not self.sampling.allow_repetition:
            filtered = penalize_repetition(
                filtered,
                generated_tokens,
                self.sampling.repetition_penalty,
                self.sampling.repetition_window,
            )
        
        # Try to use pre-computed sampling table for fast selection
        try:
            sorted_indices, cumulative = self._sampling_table_cache.get_or_compute(
                filtered, temperature, top_k
            )
            if sorted_indices and cumulative:
                # Fast sampling using pre-computed CDF
                r = random.random()
                for i, cum in enumerate(cumulative):
                    if r <= cum:
                        return sorted_indices[i], filtered
                # Fallback to last index if random exceeded all cumulative probs
                return sorted_indices[-1] if sorted_indices else 0, filtered
        except Exception as e:
            logger.debug(f"Sampling table cache failed, using standard sampling: {e}")
        
        # Fallback to standard choose_token
        chosen = choose_token(filtered, temperature)
        return chosen, filtered

    async def _index_to_token_cached(self, idx: int) -> Token:
        """Convert index to token with caching for fast repeated lookups.
        
        Maintains a cache of index->token mappings to avoid repeated
        vocab/tokenizer lookups for common tokens. Uses OrderedDict for LRU eviction.
        """
        # Check cache first
        with self._token_index_cache_lock:
            if idx in self._token_index_cache:
                # Move to end (most recently used) for LRU
                self._token_index_cache.move_to_end(idx)
                return self._token_index_cache[idx]
        
        # Use standard resolution
        token = await self._index_to_token(idx)
        
        # Cache the result with batch LRU eviction
        with self._token_index_cache_lock:
            # Batch evict 10% of cache when full to avoid tight loop eviction
            if len(self._token_index_cache) >= self._token_index_cache_max_size:
                # Remove 10% of oldest entries
                evict_count = max(1, self._token_index_cache_max_size // 10)
                for _ in range(evict_count):
                    if self._token_index_cache:
                        self._token_index_cache.popitem(last=False)
            self._token_index_cache[idx] = token
        
        return token

    async def _index_to_token(self, idx: int) -> Token:
        # First try using the provided vocab object
        if self.vocab and hasattr(self.vocab, "id_to_token"):
            try:
                return await asyncio.to_thread(self.vocab.id_to_token, idx)
            except Exception as e:
                # Log vocab lookup failure, try tokenizer
                logger.debug(f"Failed to convert index to token via vocab: {e}")

        # Try using the tokenizer's id_to_word dictionary
        if self.tokenizer and hasattr(self.tokenizer, "id_to_word"):
            try:
                id_to_word = self.tokenizer.id_to_word
                if idx in id_to_word:
                    return id_to_word[idx]
                # Modulo the index to fit within known vocabulary for fallback
                known_ids = list(id_to_word.keys())
                if known_ids:
                    fallback_idx = known_ids[idx % len(known_ids)]
                    return id_to_word[fallback_idx]
            except Exception as e:
                logger.debug(f"Failed to convert index to token via tokenizer: {e}")

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

    async def _async_safe(self, fn: Optional[Callable], args: Any, default: Any, timeout: Optional[float] = None) -> Any:
        """Execute a function asynchronously with timeout protection.
        
        Args:
            fn: The function to execute (can be sync or async)
            args: Arguments to pass to the function
            default: Default value to return on failure or timeout
            timeout: Optional timeout override. If None, uses runtime.async_operation_timeout_seconds
            
        Returns:
            The function result, or default on failure/timeout
        """
        if fn is None:
            return default
        args_tuple = args if isinstance(args, tuple) else (args,)
        # Use provided timeout or fall back to config
        op_timeout = timeout if timeout is not None else self.runtime.async_operation_timeout_seconds
        try:
            if asyncio.iscoroutinefunction(fn):
                if op_timeout is not None and op_timeout > 0:
                    return await asyncio.wait_for(fn(*args_tuple), timeout=op_timeout)
                else:
                    return await fn(*args_tuple)
            else:
                loop = asyncio.get_event_loop()
                coro = loop.run_in_executor(None, fn, *args_tuple)
                if op_timeout is not None and op_timeout > 0:
                    return await asyncio.wait_for(coro, timeout=op_timeout)
                else:
                    return await coro
        except asyncio.TimeoutError:
            fn_name = getattr(fn, '__name__', str(fn))
            logger.warning(f"Async operation '{fn_name}' timed out after {op_timeout}s, returning default")
            return default
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
