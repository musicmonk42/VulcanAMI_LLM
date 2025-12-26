"""
Embedding cache to eliminate redundant embedding computations.

Part of the VULCAN-AGI system.

This module provides high-performance caching for text embeddings used in
query routing and analysis. Reduces query routing latency from 65s to <100ms
for cached queries.

Key Features:
- Thread-safe LRU cache with configurable size
- SHA-256 based cache keys for collision resistance
- Batch embedding support with partial cache hits
- Fast-path detection for simple queries
- Comprehensive statistics for monitoring

Performance Characteristics:
- Cache lookup: O(1) average case
- Batch embedding: Only computes uncached texts
- Memory: ~4KB per cached embedding (384-dim float32)

Usage:
    # Single embedding with caching
    from vulcan.routing.embedding_cache import get_embedding_cached
    embedding = get_embedding_cached(text, model)

    # Batch embeddings with partial caching
    from vulcan.routing.embedding_cache import get_embeddings_batch_cached
    embeddings = get_embeddings_batch_cached(texts, model)

    # Check for simple queries (fast-path)
    from vulcan.routing.embedding_cache import is_simple_query
    if is_simple_query(query):
        # Skip heavy embedding computation
        pass
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Thread-safe embedding cache with LRU eviction.

    Stores computed embeddings keyed by text hash to avoid redundant
    computation. Uses LRU eviction when cache reaches capacity.

    Attributes:
        max_size: Maximum number of embeddings to cache
        _cache: OrderedDict storing hash -> embedding mappings
        _hits: Counter for cache hits
        _misses: Counter for cache misses
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache (default: 10000)
        """
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._compute_time_saved_ms = 0.0

        logger.info(f"[EmbeddingCache] Initialized with max_size={max_size}")

    def _make_key(self, text: str) -> str:
        """
        Create cache key from text using SHA-256.

        Uses full SHA-256 hash (64 chars) for collision resistance.
        SHA-256 provides ~128 bits of collision resistance.

        Args:
            text: Text to hash

        Returns:
            64-character hexadecimal hash string
        """
        return hashlib.sha256(text.encode(), usedforsecurity=False).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            text: Text to look up

        Returns:
            Cached embedding list or None if not found
        """
        key = self._make_key(text)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                logger.debug(f"[EmbeddingCache] HIT: {key[:8]}...")
                return self._cache[key].copy()  # Return copy to prevent mutation

            self._misses += 1
            return None

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Performs batch eviction (10% of capacity) when full to reduce
        lock contention from frequent single-item evictions.

        Args:
            text: Text that was embedded
            embedding: Embedding vector to cache
        """
        key = self._make_key(text)

        with self._lock:
            # Batch eviction: remove 10% when at capacity
            if len(self._cache) >= self._max_size:
                evict_count = max(1, self._max_size // 10)
                for _ in range(evict_count):
                    if self._cache:
                        self._cache.popitem(last=False)

            # Store copy to prevent external mutation
            self._cache[key] = embedding.copy() if hasattr(embedding, 'copy') else list(embedding)
            logger.debug(
                f"[EmbeddingCache] STORED: {key[:8]}..., "
                f"cache_size={len(self._cache)}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with size, hits, misses, hit_rate, and time_saved_ms
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "compute_time_saved_ms": self._compute_time_saved_ms,
            }

    def record_time_saved(self, time_ms: float) -> None:
        """
        Record time saved by cache hit.

        Args:
            time_ms: Estimated compute time saved in milliseconds
        """
        with self._lock:
            self._compute_time_saved_ms += time_ms

    def clear(self) -> None:
        """Clear the cache and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._compute_time_saved_ms = 0.0
            logger.info("[EmbeddingCache] Cleared")


# Global cache instance (singleton)
_embedding_cache: Optional[EmbeddingCache] = None
_cache_lock = threading.Lock()


def _get_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _embedding_cache

    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache()

    return _embedding_cache


def get_embedding_cached(
    text: str,
    model: Any,
    estimated_compute_ms: float = 100.0,
) -> List[float]:
    """
    Get embedding with caching.

    Checks cache first, computes embedding only if not cached.
    Records time saved for monitoring.

    Args:
        text: Text to embed
        model: Embedding model with .encode() method
        estimated_compute_ms: Estimated time for uncached computation

    Returns:
        Embedding vector as list of floats
    """
    cache = _get_cache()

    # Check cache first
    cached = cache.get(text)
    if cached is not None:
        cache.record_time_saved(estimated_compute_ms)
        return cached

    # Compute embedding
    start = time.perf_counter()

    try:
        embedding = model.encode(text)
    except AttributeError:
        # Handle models that use different method names
        if hasattr(model, 'embed'):
            embedding = model.embed(text)
        else:
            raise ValueError(f"Model {type(model)} has no encode() or embed() method")

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Handle numpy arrays
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    # Store in cache
    cache.put(text, embedding)

    logger.info(
        f"[EmbeddingCache] Computed in {elapsed_ms:.0f}ms, "
        f"cache_size={cache.get_stats()['size']}"
    )

    return embedding


def get_embeddings_batch_cached(
    texts: List[str],
    model: Any,
    estimated_compute_ms_per_text: float = 100.0,
) -> List[List[float]]:
    """
    Get embeddings for multiple texts with caching.

    Only computes embeddings for uncached texts, maximizing cache utilization.

    Args:
        texts: List of texts to embed
        model: Embedding model with .encode() method (supporting batches)
        estimated_compute_ms_per_text: Estimated time per text for stats

    Returns:
        List of embedding vectors (same order as input texts)
    """
    cache = _get_cache()

    results: List[Optional[List[float]]] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []
    cached_count = 0

    # Check cache for each text
    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
            cached_count += 1
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Record time saved for cached items
    if cached_count > 0:
        cache.record_time_saved(cached_count * estimated_compute_ms_per_text)

    # Compute uncached embeddings in batch
    if uncached_texts:
        start = time.perf_counter()

        try:
            new_embeddings = model.encode(uncached_texts)
        except AttributeError:
            if hasattr(model, 'embed'):
                new_embeddings = model.embed(uncached_texts)
            else:
                raise ValueError(f"Model {type(model)} has no encode() or embed() method")

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Store results
        for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            # Handle numpy arrays
            if hasattr(emb, "tolist"):
                emb = emb.tolist()

            cache.put(text, emb)
            results[idx] = emb

        logger.info(
            f"[EmbeddingCache] Batch: {len(texts)} texts, "
            f"{cached_count} cached, "
            f"{len(uncached_texts)} computed in {elapsed_ms:.0f}ms"
        )
    else:
        logger.info(f"[EmbeddingCache] Batch: {len(texts)} texts, ALL CACHED")

    # Type assertion - all results should be filled
    return [r if r is not None else [] for r in results]


# Simple query fast-path patterns
SIMPLE_GREETING_PATTERNS: Tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "ok",
    "okay",
    "yes",
    "no",
    "sure",
    "yep",
    "nope",
)

SIMPLE_QUESTION_STARTERS: Tuple[str, ...] = (
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "can you",
    "could you",
    "please",
    "help",
)

# Configuration constants for simple query detection
SIMPLE_QUERY_MAX_LENGTH = 15  # Maximum characters for very short query fast-path
SIMPLE_QUERY_MAX_WORDS = 2  # Maximum words for word count fast-path
SHORT_QUESTION_MAX_WORDS = 5  # Maximum words for short question fast-path


def is_simple_query(query: str) -> bool:
    """
    Check if query is simple enough to skip heavy embedding computation.

    Simple queries can use fast-path routing without computing full
    semantic embeddings, significantly reducing latency.

    Criteria:
        - Very short queries (<SIMPLE_QUERY_MAX_LENGTH chars)
        - Matches simple greeting patterns
        - Single word queries
        - Common question starters with short follow-up

    Args:
        query: Query text to check

    Returns:
        True if query should use fast-path
    """
    query_lower = query.lower().strip()

    # Very short queries
    if len(query_lower) < SIMPLE_QUERY_MAX_LENGTH:
        return True

    # Matches simple greeting patterns
    for pattern in SIMPLE_GREETING_PATTERNS:
        if query_lower.startswith(pattern):
            # Check if it's just the pattern plus punctuation
            remainder = query_lower[len(pattern):].strip()
            if not remainder or all(c in "!.?,;: " for c in remainder):
                return True

    # Single or two word queries
    words = query_lower.split()
    if len(words) <= SIMPLE_QUERY_MAX_WORDS:
        return True

    # Short questions (<=5 words starting with question word)
    if len(words) <= SHORT_QUESTION_MAX_WORDS:
        for starter in SIMPLE_QUESTION_STARTERS:
            if query_lower.startswith(starter):
                return True

    return False


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring.

    Returns:
        Dictionary with cache size, hit rate, and time saved
    """
    return _get_cache().get_stats()


def clear_cache() -> None:
    """Clear the embedding cache."""
    _get_cache().clear()


def estimate_complexity_from_length(query: str) -> float:
    """
    Quick complexity estimate based on query length.

    Provides a fast heuristic for query complexity without computing
    embeddings. Used as part of fast-path routing.

    Args:
        query: Query text

    Returns:
        Complexity score between 0.0 and 1.0
    """
    # Length-based complexity (0 to 0.4)
    length = len(query)
    length_score = min(0.4, length / 500)

    # Word count complexity (0 to 0.3)
    word_count = len(query.split())
    word_score = min(0.3, word_count / 50)

    # Sentence complexity (0 to 0.2)
    sentence_count = query.count(".") + query.count("?") + query.count("!")
    sentence_score = min(0.2, sentence_count / 5)

    # Question complexity (0 to 0.1)
    question_score = 0.1 if "?" in query else 0

    return min(1.0, length_score + word_score + sentence_score + question_score)
