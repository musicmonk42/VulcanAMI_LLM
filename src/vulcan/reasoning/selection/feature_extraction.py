"""
Multi-Tier Feature Extraction for Tool Selection.

Extracts features at different levels of complexity and cost,
from fast character n-gram features (Tier 1) to deep semantic
embeddings via SentenceTransformer (Tier 3).

Extracted from tool_selector.py to reduce module size.

PERFORMANCE NOTES:
- Uses singleton pattern for embedding model to prevent reloading
- Uses LRU cache for embedding results to avoid recomputation
- Supports circuit breaker pattern for latency protection
"""

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Optional dependency availability flags ---
try:
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. MultiTierFeatureExtractor will have limited semantic capabilities."
    )

try:
    from .embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
        get_embedding_circuit_breaker,
        get_circuit_breaker_stats,
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    EmbeddingCircuitBreaker = None
    get_embedding_circuit_breaker = None
    get_circuit_breaker_stats = None

# ==============================================================================
# Constants
# ==============================================================================

# Memory cleanup thresholds for embedding cache
# These values are tuned based on production observations of memory degradation
CLEANUP_CACHE_CAPACITY_THRESHOLD = 0.9  # Trigger cleanup at 90% cache capacity
CLEANUP_MISS_INTERVAL = 100  # Trigger cleanup every N cache misses

# Multimodal tool configuration
# CPU OPTIMIZATION: Increased from 1.5 to 3.0 to allow multimodal operations
# sufficient time headroom under CPU-only execution
MULTIMODAL_TIME_BUDGET_MULTIPLIER = 3.0  # Allow multimodal more time headroom

# Quality penalty for meta tools when domain-specific tools are available
# This ensures symbolic/math/probabilistic/causal engines preferred over meta-reasoning
META_TOOL_QUALITY_PENALTY = 0.85  # 15% reduction in quality estimate

# Embedding timeout configuration
# PERFORMANCE FIX: Reduced from 30s to 5s to prevent query routing cascade delays
# CONFIGURABLE: Set VULCAN_EMBEDDING_TIMEOUT environment variable to override
try:
    EMBEDDING_TIMEOUT = float(os.environ.get("VULCAN_EMBEDDING_TIMEOUT", "5.0"))
except (ValueError, TypeError):
    logger.warning("Invalid VULCAN_EMBEDDING_TIMEOUT, using default 5.0")
    EMBEDDING_TIMEOUT = 5.0


class MultiTierFeatureExtractor:
    """
    Extracts features at different levels of complexity and cost.

    PERFORMANCE FIX: Uses singleton pattern for embedding model to prevent
    reloading the SentenceTransformer model on every instantiation.

    PERFORMANCE FIX (2): Uses LRU cache for embedding results to prevent
    recomputing embeddings for repeated queries. Based on production logs,
    embedding batch times varied from 0.15s to 16.63s due to CPU contention.
    Caching reduces this variance by returning cached results instantly.
    """

    # PERFORMANCE FIX: Class-level singleton for embedding model
    _shared_embedding_model = None
    _shared_model_lock = threading.Lock()  # Initialize at class definition time for thread safety
    _model_load_attempted = False

    # PERFORMANCE FIX: Class-level LRU cache for embedding results
    # Prevents recomputing embeddings for the same text (0.15s-16s per call)
    _embedding_cache: OrderedDict = OrderedDict()
    _embedding_cache_lock = threading.Lock()
    _embedding_cache_maxsize = 5000  # Increased from 2000 for better hit rate
    _embedding_cache_hits = 0
    _embedding_cache_misses = 0

    # PERFORMANCE FIX: Dedicated executor for embedding operations with timeout
    _embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

    @classmethod
    def _get_shared_model(cls):
        """Get or create the shared embedding model (singleton pattern).

        PERFORMANCE FIX: Uses global model registry to ensure SentenceTransformer
        is loaded exactly ONCE per process and shared across all components.
        """
        if cls._shared_embedding_model is None and not cls._model_load_attempted:
            with cls._shared_model_lock:
                # Double-checked locking
                if cls._shared_embedding_model is None and not cls._model_load_attempted:
                    cls._model_load_attempted = True
                    # Use global model registry for process-wide singleton
                    try:
                        from vulcan.models.model_registry import get_sentence_transformer
                        cls._shared_embedding_model = get_sentence_transformer("all-MiniLM-L6-v2")
                        if cls._shared_embedding_model is not None:
                            logger.info("[TIMING] SentenceTransformer obtained from model registry (tool selector)")
                    except ImportError as e:
                        logger.debug(f"[TIMING] Model registry not available ({e}), using fallback")
                        # Fallback to direct load if registry not available
                        if TRANSFORMERS_AVAILABLE:
                            try:
                                logger.info("[TIMING] Loading SentenceTransformer for tool selector (fallback)...")
                                start = time.perf_counter()
                                cls._shared_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                                elapsed = time.perf_counter() - start
                                logger.info(f"[TIMING] SentenceTransformer loaded in {elapsed:.2f}s (tool selector fallback)")
                            except Exception as e:
                                logger.error(f"Failed to load SentenceTransformer model: {e}")

        return cls._shared_embedding_model

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize text for consistent cache key generation.

        Ensures cache hits by normalizing query text consistently.
        Without normalization, "hello world" and "Hello World " would generate
        different cache keys despite being semantically identical queries.

        Normalization steps:
        1. Strip leading/trailing whitespace
        2. Collapse multiple whitespaces to single space
        3. Convert to lowercase for case-insensitive matching

        Note: This is ONLY used for cache key generation. The original text
        is still passed to the embedding model to preserve semantic nuances.
        """
        # Strip whitespace and convert to lowercase
        normalized = text.strip().lower()
        # Collapse multiple whitespaces to single space
        normalized = ' '.join(normalized.split())
        return normalized

    @classmethod
    def _compute_cache_key(cls, text: str) -> str:
        """Compute cache key for text using SHA-256 truncated to 32 chars.

        Uses SHA-256 with 32 chars (128-bit space) to reduce collision risk in high-throughput.
        This is a shared helper to ensure consistent key computation across cache operations.
        Normalizes text before hashing to ensure consistent cache hits.
        """
        # Normalize text before computing hash to ensure consistent cache keys
        normalized_text = cls._normalize_text(text)
        return hashlib.sha256(normalized_text.encode(), usedforsecurity=False).hexdigest()[:32]

    @classmethod
    def _get_cached_embedding(cls, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available (LRU eviction)."""
        cache_key = cls._compute_cache_key(text)

        with cls._embedding_cache_lock:
            if cache_key in cls._embedding_cache:
                # Move to end (most recently used)
                cls._embedding_cache.move_to_end(cache_key)
                cls._embedding_cache_hits += 1
                # Return copy to prevent mutation (embeddings are small ~384-512 floats)
                return cls._embedding_cache[cache_key].copy()
            cls._embedding_cache_misses += 1
            return None

    @classmethod
    def _cache_embedding(cls, text: str, embedding: np.ndarray) -> None:
        """Cache embedding with batch LRU eviction for efficiency."""
        cache_key = cls._compute_cache_key(text)

        with cls._embedding_cache_lock:
            # Batch eviction: remove 10% when at capacity to reduce lock contention
            if len(cls._embedding_cache) >= cls._embedding_cache_maxsize:
                evict_count = max(1, cls._embedding_cache_maxsize // 10)  # Remove 10% (min 1)
                for _ in range(evict_count):
                    if cls._embedding_cache:
                        cls._embedding_cache.popitem(last=False)

            cls._embedding_cache[cache_key] = embedding.copy()  # Store copy

    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get embedding cache statistics for monitoring."""
        with cls._embedding_cache_lock:
            total = cls._embedding_cache_hits + cls._embedding_cache_misses
            hit_rate = cls._embedding_cache_hits / total if total > 0 else 0.0
            return {
                "size": len(cls._embedding_cache),
                "maxsize": cls._embedding_cache_maxsize,
                "hits": cls._embedding_cache_hits,
                "misses": cls._embedding_cache_misses,
                "hit_rate": hit_rate,
            }

    @classmethod
    def clear_embedding_cache(cls) -> None:
        """
        Clear the embedding cache and trigger garbage collection.

        This can be called periodically to prevent memory accumulation
        from the embedding model and cached embeddings.
        """
        import gc

        with cls._embedding_cache_lock:
            cleared_count = len(cls._embedding_cache)
            cls._embedding_cache.clear()
            logger.info(f"[EmbeddingCache] Cleared {cleared_count} cached embeddings")

        # Trigger garbage collection to free memory
        gc.collect()

    @classmethod
    def cleanup_if_needed(cls, force: bool = False) -> bool:
        """
        Perform cleanup if cache is above threshold or if forced.

        This method implements periodic cleanup to prevent the progressive
        degradation seen in production logs (embedding batch times going
        from 0.6s to 20s over 15 queries).

        Args:
            force: If True, always perform cleanup regardless of cache size.

        Returns:
            True if cleanup was performed, False otherwise.
        """
        import gc

        with cls._embedding_cache_lock:
            total_ops = cls._embedding_cache_hits + cls._embedding_cache_misses
            cache_size = len(cls._embedding_cache)

        # Cleanup conditions:
        # 1. Force requested
        # 2. Cache is at threshold capacity (prevents memory bloat)
        # 3. Every N cache misses (prevents progressive degradation)
        should_cleanup = (
            force or
            cache_size >= cls._embedding_cache_maxsize * CLEANUP_CACHE_CAPACITY_THRESHOLD or
            (cls._embedding_cache_misses > 0 and cls._embedding_cache_misses % CLEANUP_MISS_INTERVAL == 0)
        )

        if should_cleanup:
            gc.collect()
            logger.debug(
                f"[EmbeddingCache] Cleanup performed: cache_size={cache_size}, "
                f"total_ops={total_ops}"
            )
            return True

        return False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.dim = config.get("feature_dim", 128)
        # PERFORMANCE FIX: Use shared singleton model instead of loading per-instance
        self.semantic_model = MultiTierFeatureExtractor._get_shared_model()

    def extract_tier1(self, problem: Any) -> np.ndarray:
        """Fast, low-cost surface features."""
        problem_str = str(problem)[:2000]
        # Bag-of-words-like features based on character n-grams
        features = np.zeros(self.dim)
        for i in range(len(problem_str) - 2):
            trigram = problem_str[i : i + 3]
            # Simple hash-based feature vector
            index = hash(trigram) % self.dim
            features[index] += 1

        norm = np.linalg.norm(features)
        return features / (norm + 1e-10)

    def extract_tier2(self, features: np.ndarray) -> np.ndarray:
        """Structural features (placeholder logic)."""
        # In a real system, this would analyze syntax, structure of dicts/lists etc.
        # For now, we add polynomial features as a simple structural transformation.
        poly_features = np.hstack([features, features**2, np.sqrt(np.abs(features))])
        # Use hashing to project back to original dimension
        projected = np.zeros(self.dim)
        for i, val in enumerate(poly_features):
            index = (i * 31 + hash(val)) % self.dim
            projected[index] += val

        norm = np.linalg.norm(projected)
        return projected / (norm + 1e-10)

    def extract_tier3(self, problem: Any) -> np.ndarray:
        """Deep semantic features using a transformer model.

        PERFORMANCE FIX: Uses LRU cache to avoid recomputing embeddings for
        repeated queries. Cache reduces 0.15s-16.63s embedding time to instant.

        MEMORY FIX: Triggers periodic garbage collection to prevent progressive
        degradation (embedding times going from 0.6s to 20s over 15 queries).
        """
        if not self.semantic_model:
            logger.warning(
                "Semantic model not available, falling back to Tier 1 features."
            )
            return self.extract_tier1(problem)

        problem_str = str(problem)

        # Compute cache key for logging using shared helper method
        cache_key = MultiTierFeatureExtractor._compute_cache_key(problem_str)

        # PERFORMANCE FIX: Check cache first
        start_time = time.perf_counter()
        cached_embedding = MultiTierFeatureExtractor._get_cached_embedding(problem_str)
        if cached_embedding is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            # FIX 4: Cache Configuration - Log cache stats with hits for monitoring
            stats = MultiTierFeatureExtractor.get_cache_stats()
            hit_rate = stats.get('hit_rate', 0.0)
            logger.info(
                f"Embedding cache: hit=True, key={cache_key[:16]}, time={elapsed_ms:.2f}ms, "
                f"hit_rate={hit_rate:.1%}"
            )

            # Resize cached embedding if necessary
            if cached_embedding.shape[0] != self.dim:
                if cached_embedding.shape[0] > self.dim:
                    cached_embedding = cached_embedding[: self.dim]
                else:
                    padded = np.zeros(self.dim)
                    padded[: cached_embedding.shape[0]] = cached_embedding
                    cached_embedding = padded
            return cached_embedding

        try:
            # Get sentence embedding (expensive - 0.15s to 16s under load)
            embedding = self.semantic_model.encode(problem_str, show_progress_bar=False)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Cache the raw embedding before resizing
            MultiTierFeatureExtractor._cache_embedding(problem_str, embedding)

            # FIX 4: Cache Configuration - Log cache stats with each miss for diagnosis
            stats = MultiTierFeatureExtractor.get_cache_stats()
            cache_size = stats.get('size', 0)
            maxsize = stats.get('maxsize', 0)
            hit_rate = stats.get('hit_rate', 0.0)
            logger.info(
                f"Embedding cache: hit=False, key={cache_key[:16]}, time={elapsed_ms:.2f}ms, "
                f"cache_size={cache_size}/{maxsize}, hit_rate={hit_rate:.1%}"
            )

            # MEMORY FIX: Periodic cleanup to prevent progressive degradation
            MultiTierFeatureExtractor.cleanup_if_needed()

            # Resize to the required dimension if necessary
            if embedding.shape[0] != self.dim:
                if embedding.shape[0] > self.dim:
                    embedding = embedding[: self.dim]
                else:
                    padded = np.zeros(self.dim)
                    padded[: embedding.shape[0]] = embedding
                    embedding = padded
            return embedding
        except Exception as e:
            logger.error(f"Tier 3 (semantic) extraction failed: {e}")
            return self.extract_tier1(problem)

    def extract_tier4(self, problem: Any) -> np.ndarray:
        """Multimodal features (placeholder)."""
        # A real implementation would use a model like CLIP.
        # This placeholder checks for multimodal hints and combines Tier 3 features.
        if isinstance(problem, dict) and any(
            k in problem for k in ["image", "audio", "video"]
        ):
            text_part = str(problem.get("text", ""))
            # Simulate a fused embedding
            text_embedding = self.extract_tier3(text_part)
            modal_hint = np.zeros(self.dim)
            modal_hint[0] = 1.0  # Mark as multimodal
            return (text_embedding + modal_hint) / 2.0
        return self.extract_tier3(problem)

    def extract_tier3_with_timeout(self, problem: Any, timeout: float = EMBEDDING_TIMEOUT) -> np.ndarray:
        """Extract semantic features with timeout protection and circuit breaker.

        This prevents indefinite blocking when embedding operations take too long
        (observed 12-24s under CPU contention). The circuit breaker pattern
        automatically skips embeddings when latency degrades consistently.

        Circuit Breaker Logic:
        1. CLOSED: Normal operation, embeddings allowed
        2. OPEN: Skip embeddings entirely (latency too high), use keyword fallback
        3. HALF_OPEN: Test if embeddings have recovered

        Args:
            problem: Problem to extract features from
            timeout: Maximum time in seconds to wait for embedding

        Returns:
            Feature vector (falls back to tier1 on timeout or circuit open)
        """
        # PERFORMANCE FIX: Check circuit breaker first
        if CIRCUIT_BREAKER_AVAILABLE and get_embedding_circuit_breaker is not None:
            circuit_breaker = get_embedding_circuit_breaker()
            if circuit_breaker.should_skip_embedding():
                logger.warning(
                    f"[Embedding] Circuit breaker OPEN - skipping embedding, "
                    f"using keyword fallback"
                )
                return self.extract_tier1(problem)

        start_time = time.perf_counter()
        try:
            future = MultiTierFeatureExtractor._embedding_executor.submit(
                self.extract_tier3, problem
            )
            result = future.result(timeout=timeout)

            # Record successful latency to circuit breaker
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if CIRCUIT_BREAKER_AVAILABLE and get_embedding_circuit_breaker is not None:
                circuit_breaker = get_embedding_circuit_breaker()
                circuit_breaker.record_latency(elapsed_ms)

            return result
        except FuturesTimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"[Embedding] Timeout after {timeout}s ({elapsed_ms:.0f}ms) - "
                f"semantic matching skipped, falling back to keywords"
            )

            # Record timeout as a failure/slow operation
            if CIRCUIT_BREAKER_AVAILABLE and get_embedding_circuit_breaker is not None:
                circuit_breaker = get_embedding_circuit_breaker()
                # Record the timeout as latency (it's at least timeout * 1000 ms)
                circuit_breaker.record_latency(timeout * 1000)

            return self.extract_tier1(problem)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"[Embedding] Failed after {elapsed_ms:.0f}ms: {e}, falling back to tier1")

            # Record failure to circuit breaker
            if CIRCUIT_BREAKER_AVAILABLE and get_embedding_circuit_breaker is not None:
                circuit_breaker = get_embedding_circuit_breaker()
                circuit_breaker.record_failure()

            return self.extract_tier1(problem)

    def extract_adaptive(self, problem: Any, time_budget: float) -> np.ndarray:
        """Adaptively choose feature tier based on time budget.

        PERFORMANCE FIX: Uses timeout wrapper to prevent embedding operations
        from blocking indefinitely under CPU contention.
        """
        if time_budget < 100 and not isinstance(
            problem, dict
        ):  # Fast path for simple problems
            return self.extract_tier1(problem)
        elif time_budget < 1000:
            # Medium budget - use timeout protection with reduced timeout
            return self.extract_tier3_with_timeout(problem, timeout=min(5.0, EMBEDDING_TIMEOUT))
        else:
            # Higher budget - allow full timeout
            return self.extract_tier3_with_timeout(problem, timeout=EMBEDDING_TIMEOUT)
