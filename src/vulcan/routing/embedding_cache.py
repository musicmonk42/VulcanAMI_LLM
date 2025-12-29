"""
High-Performance Embedding Cache for Query Routing.

Part of the VULCAN-AGI system.

This module provides high-performance caching for text embeddings used in
query routing and semantic analysis. The cache eliminates redundant embedding
computations, reducing query routing latency from seconds to milliseconds
for cached queries.

Key Features:
    - Thread-safe LRU cache with configurable size limits
    - SHA-256 based cache keys for collision resistance
    - Batch embedding support with partial cache hits
    - Fast-path detection for simple queries (skip embedding entirely)
    - Comprehensive statistics tracking for monitoring
    - Automatic batch eviction to reduce lock contention
    - Memory-efficient storage with copy-on-read semantics

Performance Characteristics:
    - Cache lookup: O(1) average case using OrderedDict
    - Cache insertion: O(1) amortized with batch eviction
    - Batch embedding: Only computes embeddings for uncached texts
    - Memory footprint: ~4KB per cached embedding (384-dim float32)
    - Thread safety: RLock-based with minimal critical sections

Cache Key Generation:
    Uses SHA-256 hash of text content (64 hex characters) for:
    - Collision resistance (~128 bits of security)
    - Consistent key length regardless of input size
    - Fast computation for typical query lengths

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
        # Skip heavy embedding computation - use direct routing
        pass  # Application-specific fast-path logic here

    # Monitor cache performance
    from vulcan.routing.embedding_cache import get_cache_stats

    stats = get_cache_stats()
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Time saved: {stats['compute_time_saved_ms']:.0f}ms")

Thread Safety:
    All public functions are thread-safe. The module uses a singleton
    pattern with RLock for safe concurrent access. Copy-on-read
    semantics prevent mutation of cached values.

Memory Management:
    The cache uses batch eviction (10% of capacity) when full to reduce
    lock contention. LRU ordering ensures frequently accessed embeddings
    remain cached.
"""

import atexit
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Logging prefix for consistent output
LOG_PREFIX = "[EmbeddingCache]"

# Default cache size (number of embeddings)
DEFAULT_MAX_CACHE_SIZE = 10000

# Batch eviction percentage when cache is full
EVICTION_BATCH_PERCENTAGE = 0.10  # 10%

# Default estimated compute time per embedding (ms)
DEFAULT_ESTIMATED_COMPUTE_MS = 100.0

# Simple query detection thresholds
SIMPLE_QUERY_MAX_LENGTH = 15  # Maximum chars for very short query fast-path
SIMPLE_QUERY_MAX_WORDS = 2  # Maximum words for word count fast-path
SHORT_QUESTION_MAX_WORDS = 5  # Maximum words for short question fast-path

# Complexity estimation parameters
COMPLEXITY_LENGTH_MAX = 500  # Characters for max length score
COMPLEXITY_WORD_MAX = 50  # Words for max word count score
COMPLEXITY_SENTENCE_MAX = 5  # Sentences for max sentence score


# =============================================================================
# Simple Query Patterns
# =============================================================================

# Common greeting patterns that don't need semantic analysis
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
    "good",
    "great",
    "fine",
    "cool",
)

# Question starters for short question detection
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
    "is",
    "are",
    "do",
    "does",
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheStatistics:
    """
    Statistics for monitoring cache performance.

    Tracks hits, misses, and compute time saved for performance analysis
    and capacity planning.

    Attributes:
        size: Current number of cached embeddings
        max_size: Maximum cache capacity
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Cache hit rate (0.0 to 1.0)
        compute_time_saved_ms: Estimated total compute time saved
        evictions: Number of eviction operations performed
    """

    size: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float
    compute_time_saved_ms: float
    evictions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary for serialization.

        Returns:
            Dictionary representation of cache statistics.
        """
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "compute_time_saved_ms": self.compute_time_saved_ms,
            "evictions": self.evictions,
        }


# =============================================================================
# Main Cache Class
# =============================================================================


class EmbeddingCache:
    """
    Thread-safe embedding cache with LRU eviction.

    Stores computed embeddings keyed by text hash to avoid redundant
    computation. Uses LRU (Least Recently Used) eviction when cache
    reaches capacity, with batch eviction to reduce lock contention.

    Thread Safety:
        All public methods are thread-safe using RLock. The cache uses
        copy-on-read semantics to prevent external mutation of cached
        values.

    Memory Management:
        When the cache reaches capacity, it evicts 10% of entries in a
        single batch operation to reduce the frequency of eviction
        operations and associated lock contention.

    Attributes:
        max_size: Maximum number of embeddings to cache
        _cache: OrderedDict storing hash -> embedding mappings
        _hits: Counter for cache hits
        _misses: Counter for cache misses
        _evictions: Counter for eviction operations

    Example:
        >>> cache = EmbeddingCache(max_size=5000)
        >>>
        >>> # Check cache
        >>> embedding = cache.get("Hello world")
        >>> if embedding is None:
        ...     embedding = compute_embedding("Hello world")
        ...     cache.put("Hello world", embedding)
        >>>
        >>> # Get statistics
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    """

    def __init__(self, max_size: int = DEFAULT_MAX_CACHE_SIZE) -> None:
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache.
                Default is 10000, which uses approximately 40MB for
                384-dimensional embeddings.

        Raises:
            ValueError: If max_size is less than 1.
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")

        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()

        # Statistics tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._compute_time_saved_ms = 0.0

        logger.info(f"{LOG_PREFIX} Initialized with max_size={max_size}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for consistent cache key generation.

        CRITICAL FIX: This was the root cause of 0% cache hit rate.
        Without normalization, "hello world" and "Hello World " would generate
        different cache keys despite being semantically identical queries.

        Normalization steps:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase for case-insensitive matching
        3. Collapse multiple whitespaces to single space

        Note: This is ONLY used for cache key generation. The original text
        is still passed to the embedding model to preserve semantic nuances.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text string.
        """
        # Strip whitespace and convert to lowercase
        normalized = text.strip().lower()
        # Collapse multiple whitespaces to single space
        normalized = " ".join(normalized.split())
        return normalized

    @staticmethod
    def _make_key(text: str) -> str:
        """
        Create cache key from text using SHA-256.

        Uses full SHA-256 hash (64 hex characters) for collision resistance.
        SHA-256 provides ~128 bits of collision resistance, making
        collisions astronomically unlikely for any practical use case.

        CRITICAL FIX: Now normalizes text before hashing to ensure cache hits.
        This fixes the 0% hit rate issue where equivalent queries like
        "Hello World" and "hello world" generated different keys.

        Args:
            text: Text to hash.

        Returns:
            64-character hexadecimal hash string.

        Note:
            The usedforsecurity=False flag indicates this hash is not
            used for security purposes (just cache keying), which allows
            the hash to work in FIPS-restricted environments.
        """
        # Normalize text before hashing to ensure consistent cache keys
        normalized = EmbeddingCache._normalize_text(text)
        return hashlib.sha256(
            normalized.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.

        If found, moves the entry to the end of the OrderedDict to
        implement LRU behavior (most recently used).

        Args:
            text: Text to look up.

        Returns:
            Copy of cached embedding list, or None if not found.
            Returns a copy to prevent external mutation of cached values.

        Example:
            >>> embedding = cache.get("Hello world")
            >>> if embedding is not None:
            ...     print(f"Cache hit! Embedding dim: {len(embedding)}")
        """
        # FIX #1: Comprehensive debug logging for cache operations
        normalized = self._normalize_text(text)
        key = self._make_key(text)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)

                # Compute hit rate for logging
                total = self._hits + self._misses
                hit_rate = self._hits / total if total > 0 else 0.0

                # FIX #1: Enhanced debug logging with original and normalized text
                logger.debug(
                    f"{LOG_PREFIX} HIT: key={key[:16]}... | "
                    f"original='{text[:50]}{'...' if len(text) > 50 else ''}' | "
                    f"normalized='{normalized[:50]}{'...' if len(normalized) > 50 else ''}' | "
                    f"stats=(hits={self._hits}, misses={self._misses}, rate={hit_rate:.1%})"
                )

                # Return copy to prevent external mutation
                return self._cache[key].copy()

            self._misses += 1
            # Log cache miss with debug details
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            # FIX #1: Enhanced debug logging for cache misses
            logger.debug(
                f"{LOG_PREFIX} MISS: key={key[:16]}... | "
                f"original='{text[:50]}{'...' if len(text) > 50 else ''}' | "
                f"normalized='{normalized[:50]}{'...' if len(normalized) > 50 else ''}' | "
                f"stats=(hits={self._hits}, misses={self._misses}, rate={hit_rate:.1%}, "
                f"cache_size={len(self._cache)}/{self._max_size})"
            )
            return None

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Performs batch eviction (10% of capacity) when full to reduce
        lock contention from frequent single-item evictions.

        Args:
            text: Text that was embedded.
            embedding: Embedding vector to cache. A copy is stored
                to prevent external mutation.

        Example:
            >>> embedding = model.encode("Hello world")
            >>> cache.put("Hello world", embedding)
        """
        # FIX #1: Comprehensive debug logging for cache operations
        normalized = self._normalize_text(text)
        key = self._make_key(text)

        with self._lock:
            # Batch eviction when at capacity
            if len(self._cache) >= self._max_size:
                evict_count = max(1, int(self._max_size * EVICTION_BATCH_PERCENTAGE))
                for _ in range(evict_count):
                    if self._cache:
                        self._cache.popitem(last=False)  # Remove oldest
                self._evictions += 1

                logger.debug(
                    f"{LOG_PREFIX} Evicted {evict_count} entries "
                    f"(eviction #{self._evictions})"
                )

            # Store copy to prevent external mutation
            if hasattr(embedding, "copy"):
                self._cache[key] = embedding.copy()
            elif hasattr(embedding, "tolist"):
                # Handle numpy arrays
                self._cache[key] = embedding.tolist()
            else:
                self._cache[key] = list(embedding)

            # FIX #1: Enhanced debug logging with original and normalized text
            logger.debug(
                f"{LOG_PREFIX} STORED: key={key[:16]}... | "
                f"original='{text[:50]}{'...' if len(text) > 50 else ''}' | "
                f"normalized='{normalized[:50]}{'...' if len(normalized) > 50 else ''}' | "
                f"cache_size={len(self._cache)}/{self._max_size}"
            )

    def record_time_saved(self, time_ms: float) -> None:
        """
        Record time saved by cache hit.

        Used for monitoring and reporting the effectiveness of the cache.

        Args:
            time_ms: Estimated compute time saved in milliseconds.

        Example:
            >>> if embedding := cache.get(text):
            ...     cache.record_time_saved(100.0)  # 100ms saved
        """
        with self._lock:
            self._compute_time_saved_ms += time_ms

    def get_stats(self) -> CacheStatistics:
        """
        Get cache statistics for monitoring.

        Returns:
            CacheStatistics dataclass with current metrics.

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Size: {stats.size}/{stats.max_size}")
            >>> print(f"Hit rate: {stats.hit_rate:.1%}")
            >>> print(f"Time saved: {stats.compute_time_saved_ms:.0f}ms")
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return CacheStatistics(
                size=len(self._cache),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                hit_rate=hit_rate,
                compute_time_saved_ms=self._compute_time_saved_ms,
                evictions=self._evictions,
            )

    def clear(self) -> None:
        """
        Clear the cache and reset statistics.

        Useful for testing or when a fresh cache is needed.

        Example:
            >>> cache.clear()
            >>> stats = cache.get_stats()
            >>> assert stats.size == 0
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._compute_time_saved_ms = 0.0

        logger.info(f"{LOG_PREFIX} Cache cleared")

    def clear_and_rebuild(self) -> None:
        """
        Clear the cache and reset all internal state for corruption recovery.

        This method is useful when the cache may be corrupted due to:
        - Serialization/deserialization issues
        - Concurrent access race conditions
        - Memory corruption
        - Application crashes during cache operations

        Unlike clear(), this method:
        - Logs the previous state before clearing for audit purposes
        - Emits a warning-level log message to indicate corruption recovery
        - Performs all operations under a single lock to prevent race conditions

        Note: This clears the cache entirely. Actual cache entries are rebuilt
        lazily through normal cache operations (get/put) after calling this method.

        Example:
            >>> cache.clear_and_rebuild()  # Cache may be corrupted
            >>> stats = cache.get_stats()
            >>> assert stats.size == 0
        """
        with self._lock:
            # Capture state for logging before clearing
            old_size = len(self._cache)
            old_hits = self._hits
            old_misses = self._misses

            # Clear cache and reset statistics (inline to ensure single lock scope)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._compute_time_saved_ms = 0.0

        # Log with warning level for audit purposes (corruption recovery scenario)
        logger.warning(
            f"{LOG_PREFIX} Cache cleared and rebuilt (corruption recovery). "
            f"Previous state: size={old_size}, hits={old_hits}, misses={old_misses}"
        )

    def __len__(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of embeddings currently cached.
        """
        with self._lock:
            return len(self._cache)

    def __contains__(self, text: str) -> bool:
        """
        Check if text is in cache without affecting LRU order.

        Args:
            text: Text to check.

        Returns:
            True if text is cached, False otherwise.
        """
        key = self._make_key(text)
        with self._lock:
            return key in self._cache

    def dump_statistics(self) -> str:
        """
        Dump comprehensive cache statistics for debugging.

        Returns a formatted string with:
        - Cache size and capacity
        - Hit rate and performance metrics
        - Sample of cached keys (first 10)
        - Eviction and time saved information

        Returns:
            Formatted statistics string for logging/debugging.

        Example:
            >>> print(cache.dump_statistics())
            === EmbeddingCache Statistics ===
            Size: 50/10000
            Hit rate: 75.0% (150 hits, 50 misses)
            ...
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            # Get first 10 keys for sample
            keys = list(self._cache.keys())[:10]
            key_sample = [f"  {k[:32]}..." for k in keys]

            lines = [
                f"=== {LOG_PREFIX} Statistics ===",
                f"Size: {len(self._cache)}/{self._max_size}",
                f"Hit rate: {hit_rate:.1%} ({self._hits} hits, {self._misses} misses)",
                f"Evictions: {self._evictions}",
                f"Time saved: {self._compute_time_saved_ms:.0f}ms",
                f"Sample keys ({min(10, len(keys))} of {len(keys)}):",
            ]
            lines.extend(key_sample)

            return "\n".join(lines)

    def debug_cache_state(self) -> Dict[str, Any]:
        """
        FIX #1: Get detailed debug information about cache state.

        This method provides comprehensive diagnostic data for debugging
        cache hit rate issues. Returns all internal metrics and sample entries.

        Returns:
            Dictionary with debug information including:
            - size: Current cache size
            - max_size: Maximum cache capacity
            - hits: Total cache hits
            - misses: Total cache misses
            - hit_rate: Calculated hit rate
            - evictions: Total eviction operations
            - time_saved_ms: Estimated time saved by cache
            - sample_keys: First 5 cache keys (truncated)
            - fill_percentage: Cache fill percentage

        Example:
            >>> debug_info = cache.debug_cache_state()
            >>> if debug_info['hit_rate'] < 0.1:
            ...     logger.warning("Low cache hit rate detected!")
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            fill_pct = (
                (len(self._cache) / self._max_size * 100) if self._max_size > 0 else 0.0
            )

            # Get sample keys for debugging
            sample_keys = [k[:32] + "..." for k in list(self._cache.keys())[:5]]

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "hit_rate_pct": f"{hit_rate:.1%}",
                "evictions": self._evictions,
                "time_saved_ms": self._compute_time_saved_ms,
                "sample_keys": sample_keys,
                "fill_percentage": f"{fill_pct:.1f}%",
                "total_operations": total,
            }

    def verify_key_generation(self, text: str) -> Dict[str, str]:
        """
        FIX #1: Verify cache key generation for debugging.

        This method shows exactly how a text string is normalized and hashed,
        helping diagnose why cache hits might be failing.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary with normalization steps:
            - original: The original input text
            - normalized: The normalized text
            - key: The generated cache key
            - is_cached: Whether this text is currently cached

        Example:
            >>> result = cache.verify_key_generation("  Hello World  ")
            >>> print(f"Key: {result['key']}, Cached: {result['is_cached']}")
        """
        normalized = self._normalize_text(text)
        key = self._make_key(text)

        with self._lock:
            is_cached = key in self._cache

        return {
            "original": text,
            "original_repr": repr(text),
            "normalized": normalized,
            "normalized_repr": repr(normalized),
            "key": key,
            "key_short": key[:16] + "...",
            "is_cached": is_cached,
        }


# =============================================================================
# Global Singleton Management
# =============================================================================

# Global cache instance (singleton)
_embedding_cache: Optional[EmbeddingCache] = None
_cache_lock = threading.Lock()


def _get_cache(max_size: int = DEFAULT_MAX_CACHE_SIZE) -> EmbeddingCache:
    """
    Get or create the global embedding cache singleton.

    Uses double-checked locking for thread-safe lazy initialization.

    Args:
        max_size: Maximum cache size (only used on first call).

    Returns:
        Global EmbeddingCache instance.
    """
    global _embedding_cache

    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache(max_size=max_size)

    return _embedding_cache


def _shutdown_cache() -> None:
    """Atexit handler to log cache statistics on shutdown."""
    global _embedding_cache

    if _embedding_cache is not None:
        stats = _embedding_cache.get_stats()
        logger.info(
            f"{LOG_PREFIX} Final stats: "
            f"size={stats.size}, hits={stats.hits}, misses={stats.misses}, "
            f"hit_rate={stats.hit_rate:.1%}, time_saved={stats.compute_time_saved_ms:.0f}ms"
        )


# Register atexit handler
atexit.register(_shutdown_cache)


# =============================================================================
# Public API Functions
# =============================================================================


def get_embedding_cached(
    text: str,
    model: Any,
    estimated_compute_ms: float = DEFAULT_ESTIMATED_COMPUTE_MS,
) -> List[float]:
    """
    Get embedding with caching.

    Checks cache first, computes embedding only if not cached.
    Records time saved for monitoring when cache hits occur.

    Args:
        text: Text to embed.
        model: Embedding model with .encode() or .embed() method.
            The model should accept a string and return an embedding
            vector (list or numpy array).
        estimated_compute_ms: Estimated time for uncached computation.
            Used for statistics tracking. Default is 100ms.

    Returns:
        Embedding vector as list of floats.

    Raises:
        ValueError: If model has no encode() or embed() method.

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>>
        >>> # First call computes embedding
        >>> emb1 = get_embedding_cached("Hello world", model)
        >>>
        >>> # Second call returns cached embedding
        >>> emb2 = get_embedding_cached("Hello world", model)
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
        if hasattr(model, "encode"):
            embedding = model.encode(text)
        elif hasattr(model, "embed"):
            embedding = model.embed(text)
        else:
            raise ValueError(
                f"Model {type(model).__name__} has no encode() or embed() method"
            )
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Embedding computation failed: {e}")
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Convert numpy arrays to list
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    # Store in cache
    cache.put(text, embedding)

    stats = cache.get_stats()
    logger.info(
        f"{LOG_PREFIX} Computed in {elapsed_ms:.0f}ms, "
        f"cache_size={stats.size}/{stats.max_size}, "
        f"hit_rate={stats.hit_rate:.1%}"
    )

    return embedding


def get_embeddings_batch_cached(
    texts: List[str],
    model: Any,
    estimated_compute_ms_per_text: float = DEFAULT_ESTIMATED_COMPUTE_MS,
) -> List[List[float]]:
    """
    Get embeddings for multiple texts with caching.

    Only computes embeddings for uncached texts, maximizing cache
    utilization. This is more efficient than calling get_embedding_cached
    in a loop because it batches the uncached computations.

    Args:
        texts: List of texts to embed.
        model: Embedding model with .encode() or .embed() method
            that supports batch input (list of strings).
        estimated_compute_ms_per_text: Estimated time per text for
            statistics tracking. Default is 100ms.

    Returns:
        List of embedding vectors (same order as input texts).
        Each embedding is a list of floats.

    Raises:
        ValueError: If model has no encode() or embed() method.

    Example:
        >>> texts = ["Hello", "World", "Hello"]  # Note: "Hello" appears twice
        >>> embeddings = get_embeddings_batch_cached(texts, model)
        >>> # Only 2 embeddings computed (Hello cached for second occurrence)
        >>> assert len(embeddings) == 3
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
            if hasattr(model, "encode"):
                new_embeddings = model.encode(uncached_texts)
            elif hasattr(model, "embed"):
                new_embeddings = model.embed(uncached_texts)
            else:
                raise ValueError(
                    f"Model {type(model).__name__} has no encode() or embed() method"
                )
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Batch embedding computation failed: {e}")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Store results and update cache
        for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            # Convert numpy arrays to list
            if hasattr(emb, "tolist"):
                emb = emb.tolist()

            cache.put(text, emb)
            results[idx] = emb

        logger.info(
            f"{LOG_PREFIX} Batch: {len(texts)} texts, "
            f"{cached_count} cached, "
            f"{len(uncached_texts)} computed in {elapsed_ms:.0f}ms"
        )
    else:
        logger.info(f"{LOG_PREFIX} Batch: {len(texts)} texts, ALL CACHED")

    # Ensure all results are filled (should never have None values)
    return [r if r is not None else [] for r in results]


def is_simple_query(query: str) -> bool:
    """
    Check if query is simple enough to skip heavy embedding computation.

    Simple queries can use fast-path routing without computing full
    semantic embeddings, significantly reducing latency for common
    interactions.

    Criteria for simple queries:
        - Very short queries (< SIMPLE_QUERY_MAX_LENGTH characters)
        - Matches simple greeting patterns (hello, thanks, bye, etc.)
        - Single or two word queries
        - Common question starters with short follow-up (<=5 words)

    Args:
        query: Query text to check.

    Returns:
        True if query should use fast-path routing.

    Example:
        >>> is_simple_query("Hello")
        True
        >>> is_simple_query("What is the weather?")
        True
        >>> is_simple_query("Explain the quantum mechanical model of the atom")
        False
    """
    query_lower = query.lower().strip()

    # Very short queries
    if len(query_lower) < SIMPLE_QUERY_MAX_LENGTH:
        return True

    # Matches simple greeting patterns
    for pattern in SIMPLE_GREETING_PATTERNS:
        if query_lower.startswith(pattern):
            # Check if it's just the pattern plus punctuation/whitespace
            remainder = query_lower[len(pattern) :].strip()
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


def estimate_complexity_from_length(query: str) -> float:
    """
    Quick complexity estimate based on query length.

    Provides a fast heuristic for query complexity without computing
    embeddings. Used as part of fast-path routing decisions.

    Complexity factors:
        - Length score: 0 to 0.4 based on character count
        - Word score: 0 to 0.3 based on word count
        - Sentence score: 0 to 0.2 based on sentence count
        - Question score: 0.1 if query contains a question mark

    Args:
        query: Query text.

    Returns:
        Complexity score between 0.0 and 1.0.

    Example:
        >>> estimate_complexity_from_length("Hello")
        0.03
        >>> estimate_complexity_from_length("What is the meaning of life?")
        0.26
    """
    # Length-based complexity (0 to 0.4)
    length = len(query)
    length_score = min(0.4, length / COMPLEXITY_LENGTH_MAX)

    # Word count complexity (0 to 0.3)
    word_count = len(query.split())
    word_score = min(0.3, word_count / COMPLEXITY_WORD_MAX)

    # Sentence complexity (0 to 0.2)
    sentence_count = query.count(".") + query.count("?") + query.count("!")
    sentence_score = min(0.2, sentence_count / COMPLEXITY_SENTENCE_MAX)

    # Question complexity (0 to 0.1)
    question_score = 0.1 if "?" in query else 0.0

    return min(1.0, length_score + word_score + sentence_score + question_score)


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring.

    Returns:
        Dictionary with cache size, hit rate, and time saved.

    Example:
        >>> stats = get_cache_stats()
        >>> print(f"Size: {stats['size']}/{stats['max_size']}")
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Time saved: {stats['compute_time_saved_ms']:.0f}ms")
    """
    return _get_cache().get_stats().to_dict()


def clear_cache() -> None:
    """
    Clear the embedding cache and reset statistics.

    Useful for testing or when cache invalidation is needed.

    Example:
        >>> clear_cache()
        >>> stats = get_cache_stats()
        >>> assert stats['size'] == 0
    """
    _get_cache().clear()


def clear_and_rebuild_cache() -> None:
    """
    Clear the embedding cache and reset internal state for corruption recovery.

    This function should be called when the cache may be corrupted.
    It clears all cached entries and resets statistics to ensure clean state.
    Cache entries are rebuilt lazily through normal operations after calling this.

    Use cases:
        - Cache corruption suspected (inconsistent results)
        - After application crashes
        - When cache behavior is erratic
        - Memory corruption recovery

    Example:
        >>> clear_and_rebuild_cache()  # Cache may be corrupted
        >>> stats = get_cache_stats()
        >>> assert stats['size'] == 0
    """
    _get_cache().clear_and_rebuild()


def configure_cache(max_size: int = DEFAULT_MAX_CACHE_SIZE) -> None:
    """
    Configure or reconfigure the global cache.

    WARNING: This clears the existing cache if already initialized.
    Should typically only be called during application startup.

    Args:
        max_size: Maximum number of embeddings to cache.

    Example:
        >>> configure_cache(max_size=20000)  # Increase cache size
    """
    global _embedding_cache

    with _cache_lock:
        if _embedding_cache is not None:
            logger.warning(
                f"{LOG_PREFIX} Reconfiguring cache (existing cache will be cleared)"
            )
        _embedding_cache = EmbeddingCache(max_size=max_size)

    logger.info(f"{LOG_PREFIX} Cache configured with max_size={max_size}")


def dump_cache_statistics() -> str:
    """
    Dump comprehensive cache statistics for debugging.

    Returns a formatted string with cache size, hit rate, sample keys,
    and other diagnostic information.

    Example:
        >>> print(dump_cache_statistics())
        === [EmbeddingCache] Statistics ===
        Size: 50/10000
        Hit rate: 75.0% (150 hits, 50 misses)
        ...
    """
    return _get_cache().dump_statistics()


def debug_cache_state() -> Dict[str, Any]:
    """
    FIX #1: Get detailed debug information about cache state.

    This function provides comprehensive diagnostic data for debugging
    cache hit rate issues. Use this to diagnose 0% hit rate problems.

    Returns:
        Dictionary with debug information including:
        - size: Current cache size
        - max_size: Maximum cache capacity
        - hits: Total cache hits
        - misses: Total cache misses
        - hit_rate: Calculated hit rate (0.0 to 1.0)
        - hit_rate_pct: Hit rate as percentage string
        - evictions: Total eviction operations
        - time_saved_ms: Estimated time saved by cache
        - sample_keys: First 5 cache keys (truncated)
        - fill_percentage: Cache fill percentage
        - total_operations: Total get operations

    Example:
        >>> debug_info = debug_cache_state()
        >>> print(f"Hit rate: {debug_info['hit_rate_pct']}")
        >>> if debug_info['hit_rate'] < 0.1:
        ...     print("WARNING: Low cache hit rate!")
    """
    return _get_cache().debug_cache_state()


def verify_cache_key(text: str) -> Dict[str, str]:
    """
    FIX #1: Verify how a text string is normalized and hashed.

    This function shows exactly how cache keys are generated, helping
    diagnose why cache hits might be failing (0% hit rate issue).

    Args:
        text: The text to analyze.

    Returns:
        Dictionary with normalization details:
        - original: The original input text
        - original_repr: repr() of original text (shows hidden chars)
        - normalized: The normalized text
        - normalized_repr: repr() of normalized text
        - key: The full cache key (SHA-256 hash)
        - key_short: Truncated cache key for display
        - is_cached: Whether this text is currently cached

    Example:
        >>> result = verify_cache_key("  Hello World  ")
        >>> print(f"Normalized: {result['normalized']}")
        >>> print(f"Is cached: {result['is_cached']}")
    """
    return _get_cache().verify_key_generation(text)
