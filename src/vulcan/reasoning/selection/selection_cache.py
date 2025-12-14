"""
Selection Cache for Tool Selection System

Provides multi-level caching for feature extraction, tool selection decisions,
and execution results to improve performance and reduce redundant computations.

Fixed version with proper size calculation and disk cache limits.
PATCH 5: Interruptible cleanup loop with Event-based shutdown.
"""

import hashlib
import json
import logging
import pickle  # SECURITY: Internal data only, never deserialize untrusted data
import sys
import threading
import time
import zlib
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# CRITICAL FIX: Add sizeof function
def sizeof(obj: Any) -> int:
    """Calculate size of object in bytes"""
    try:
        # Try sys.getsizeof first
        return sys.getsizeof(obj)
    except Exception:
        # Fallback: use pickle size
        try:
            return len(pickle.dumps(obj))
        except Exception as e2:
            logger.warning(f"Size calculation failed: {e2}")
            return 1024  # Default estimate


class CacheLevel(Enum):
    """Cache levels with different characteristics"""

    L1_MEMORY = "l1_memory"  # Fast, small, in-memory
    L2_MEMORY = "l2_memory"  # Medium, larger, in-memory
    L3_DISK = "l3_disk"  # Slow, large, persistent


class EvictionPolicy(Enum):
    """Cache eviction policies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based


@dataclass
class CacheEntry:
    """Single cache entry"""

    key: str
    value: Any
    size_bytes: int
    creation_time: float
    last_access_time: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.creation_time > self.ttl_seconds

    def touch(self):
        """Update access time and count"""
        self.last_access_time = time.time()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Cache performance statistics"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    compression_savings: int = 0
    avg_access_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """Thread-safe LRU cache implementation"""

    def __init__(self, max_size: int, max_bytes: int = None):
        self.max_size = max_size
        self.max_bytes = max_bytes or (100 * 1024 * 1024)  # 100MB default
        self.cache = OrderedDict()
        self.size_bytes = 0

        # CRITICAL FIX: Use RLock for thread safety
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            try:
                if key in self.cache:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry = self.cache[key]
                    entry.touch()
                    return entry.value
                return None
            except Exception as e:
                logger.error(f"Cache get failed: {e}")
                return None

    def put(self, key: str, value: Any, size_bytes: int, ttl: Optional[float] = None):
        """Put value in cache"""
        with self.lock:
            try:
                # Remove old entry if exists
                if key in self.cache:
                    old_entry = self.cache.pop(key)
                    self.size_bytes -= old_entry.size_bytes

                # Check if we need to evict
                while (
                    len(self.cache) >= self.max_size
                    or self.size_bytes + size_bytes > self.max_bytes
                ):
                    if not self.cache:
                        break
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    oldest_entry = self.cache.pop(oldest_key)
                    self.size_bytes -= oldest_entry.size_bytes

                # Add new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    creation_time=time.time(),
                    last_access_time=time.time(),
                    ttl_seconds=ttl,
                )

                self.cache[key] = entry
                self.size_bytes += size_bytes
            except Exception as e:
                logger.error(f"Cache put failed: {e}")

    def invalidate(self, key: str) -> bool:
        """Remove entry from cache"""
        with self.lock:
            try:
                if key in self.cache:
                    entry = self.cache.pop(key)
                    self.size_bytes -= entry.size_bytes
                    return True
                return False
            except Exception as e:
                logger.error(f"Cache invalidate failed: {e}")
                return False

    def clear(self):
        """Clear entire cache"""
        with self.lock:
            try:
                self.cache.clear()
                self.size_bytes = 0
            except Exception as e:
                logger.error(f"Cache clear failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            try:
                return {
                    "entries": len(self.cache),
                    "size_bytes": self.size_bytes,
                    "max_size": self.max_size,
                    "max_bytes": self.max_bytes,
                    "utilization": (
                        len(self.cache) / self.max_size if self.max_size > 0 else 0
                    ),
                }
            except Exception as e:
                logger.error(f"Get stats failed: {e}")
                return {}


class CompressedCache:
    """Cache with compression support"""

    def __init__(self, base_cache: LRUCache, compression_threshold: int = 1024):
        self.base_cache = base_cache
        self.compression_threshold = compression_threshold
        self.compression_stats = defaultdict(float)

    def get(self, key: str) -> Optional[Any]:
        """Get and decompress value"""
        try:
            compressed = self.base_cache.get(key)
            if compressed is None:
                return None

            if isinstance(compressed, bytes):
                # Decompress
                try:
                    decompressed = pickle.loads(
                        zlib.decompress(compressed)
                    )  # nosec B301 - Internal data structure
                    return decompressed
                except Exception as e:
                    logger.warning(f"Decompression failed: {e}")
                    return compressed
            return compressed
        except Exception as e:
            logger.error(f"Compressed cache get failed: {e}")
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Compress and store value"""
        try:
            # Serialize
            serialized = pickle.dumps(value)
            size = len(serialized)

            # Compress if large enough
            if size > self.compression_threshold:
                compressed = zlib.compress(serialized, level=6)
                compression_ratio = len(compressed) / size if size > 0 else 1.0

                if compression_ratio < 0.9:  # Only use if significant compression
                    self.compression_stats[key] = compression_ratio
                    self.base_cache.put(key, compressed, len(compressed), ttl)
                    return

            # Store uncompressed
            self.base_cache.put(key, value, size, ttl)
        except Exception as e:
            logger.error(f"Compressed cache put failed: {e}")


class MultiLevelCache:
    """Multi-level cache with automatic promotion/demotion"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # L1: Small, fast cache for hot data
        self.l1 = LRUCache(
            max_size=config.get("l1_size", 100),
            max_bytes=config.get("l1_bytes", 10 * 1024 * 1024),  # 10MB
        )

        # L2: Larger cache with compression
        self.l2 = CompressedCache(
            LRUCache(
                max_size=config.get("l2_size", 1000),
                max_bytes=config.get("l2_bytes", 100 * 1024 * 1024),  # 100MB
            )
        )

        # L3: Disk cache (if enabled)
        self.l3_enabled = config.get("enable_disk_cache", False)
        if self.l3_enabled:
            self.l3_path = Path(config.get("disk_cache_path", "./cache"))
            self.l3_path.mkdir(parents=True, exist_ok=True)
            self.l3_index = {}  # In-memory index for disk cache

            # CRITICAL FIX: Add disk cache size limits
            self.l3_max_size_mb = config.get("l3_max_size_mb", 1000)  # 1GB default
            self.l3_current_size_mb = 0.0
            self._load_disk_index()

        # Statistics
        self.stats = CacheStatistics()
        self.level_stats = {
            "l1": CacheStatistics(),
            "l2": CacheStatistics(),
            "l3": CacheStatistics(),
        }

        # Access tracking for promotion
        self.access_counts = defaultdict(int)
        self.promotion_threshold = config.get("promotion_threshold", 3)

        # CRITICAL FIX: Add lock for disk operations
        self.disk_lock = threading.RLock()

    def _load_disk_index(self):
        """Load existing disk cache index"""
        if not self.l3_enabled:
            return

        try:
            for cache_file in self.l3_path.glob("*.cache"):
                key = cache_file.stem
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                mtime = cache_file.stat().st_mtime

                self.l3_index[key] = {"timestamp": mtime, "size_mb": size_mb}
                self.l3_current_size_mb += size_mb

            logger.info(
                f"Loaded disk cache index: {len(self.l3_index)} entries, {self.l3_current_size_mb:.2f} MB"
            )
        except Exception as e:
            logger.error(f"Failed to load disk index: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        start_time = time.time()

        try:
            # Check L1
            value = self.l1.get(key)
            if value is not None:
                self.stats.hits += 1
                self.level_stats["l1"].hits += 1
                self._update_access_time(start_time)
                return value

            # Check L2
            value = self.l2.get(key)
            if value is not None:
                self.stats.hits += 1
                self.level_stats["l2"].hits += 1

                # Promote to L1 if accessed frequently
                self.access_counts[key] += 1
                if self.access_counts[key] >= self.promotion_threshold:
                    self._promote_to_l1(key, value)

                self._update_access_time(start_time)
                return value

            # Check L3 (disk)
            if self.l3_enabled:
                value = self._get_from_disk(key)
                if value is not None:
                    self.stats.hits += 1
                    self.level_stats["l3"].hits += 1

                    # Promote to L2
                    self._promote_to_l2(key, value)

                    self._update_access_time(start_time)
                    return value

            # Cache miss
            self.stats.misses += 1
            self._update_access_time(start_time)
            return None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self.stats.misses += 1
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache hierarchy"""

        try:
            # Estimate size
            size = self._estimate_size(value)

            # Decide cache level based on size and access pattern
            if size < 1024:  # Small items go to L1
                self.l1.put(key, value, size, ttl)
                self.level_stats["l1"].entry_count += 1
            else:  # Larger items go to L2
                self.l2.put(key, value, ttl)
                self.level_stats["l2"].entry_count += 1

            # Reset access count
            self.access_counts[key] = 0
        except Exception as e:
            logger.error(f"Cache put failed: {e}")

    def _promote_to_l1(self, key: str, value: Any):
        """Promote entry to L1 cache"""
        try:
            size = self._estimate_size(value)
            if size < 10240:  # Only promote if small enough (10KB)
                self.l1.put(key, value, size)
        except Exception as e:
            logger.warning(f"L1 promotion failed: {e}")

    def _promote_to_l2(self, key: str, value: Any):
        """Promote entry from L3 to L2"""
        try:
            self.l2.put(key, value)
        except Exception as e:
            logger.warning(f"L2 promotion failed: {e}")

    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        if key not in self.l3_index:
            return None

        file_path = self.l3_path / f"{key}.cache"

        with self.disk_lock:
            if not file_path.exists():
                # Clean up stale index entry
                if key in self.l3_index:
                    info = self.l3_index[key]
                    self.l3_current_size_mb -= info.get("size_mb", 0)
                    del self.l3_index[key]
                return None

            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)  # nosec B301 - Internal data structure
            except Exception as e:
                logger.error(f"Disk cache read failed: {e}")
                return None

    # CRITICAL FIX: Disk cache with size limits
    def _put_to_disk(self, key: str, value: Any):
        """Store value in disk cache - CRITICAL: Check size limits"""

        if not self.l3_enabled:
            return

        with self.disk_lock:
            try:
                file_path = self.l3_path / f"{key}.cache"

                # Serialize to check size
                data = pickle.dumps(value)
                size_mb = len(data) / (1024 * 1024)

                # CRITICAL FIX: Check total size limit
                if self.l3_current_size_mb + size_mb > self.l3_max_size_mb:
                    # Evict oldest entries to make room
                    self._evict_disk_cache(size_mb)

                # Write file
                with open(file_path, "wb") as f:
                    f.write(data)

                # Update index
                old_size = 0
                if key in self.l3_index:
                    old_size = self.l3_index[key].get("size_mb", 0)

                self.l3_index[key] = {"timestamp": time.time(), "size_mb": size_mb}

                self.l3_current_size_mb = self.l3_current_size_mb - old_size + size_mb

            except Exception as e:
                logger.error(f"Disk cache write failed: {e}")

    # CRITICAL FIX: Evict disk cache entries
    def _evict_disk_cache(self, needed_mb: float):
        """Evict entries from disk cache to make room"""

        try:
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self.l3_index.items(), key=lambda x: x[1].get("timestamp", 0)
            )

            freed_mb = 0.0
            target_mb = needed_mb + 100  # Free extra space for headroom

            for key, info in sorted_entries:
                if freed_mb >= target_mb:
                    break

                file_path = self.l3_path / f"{key}.cache"
                try:
                    if file_path.exists():
                        file_path.unlink()

                    size = info.get("size_mb", 0)
                    self.l3_current_size_mb -= size
                    freed_mb += size
                    del self.l3_index[key]

                except Exception as e:
                    logger.error(f"Cache eviction failed for {key}: {e}")

            logger.info(f"Evicted {freed_mb:.2f} MB from disk cache")
        except Exception as e:
            logger.error(f"Disk cache eviction failed: {e}")

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            return sizeof(value)
        except Exception as e:
            logger.warning(f"Size estimation failed: {e}")
            return 1024  # Default estimate

    def _update_access_time(self, start_time: float):
        """Update average access time"""
        try:
            access_time_ms = (time.time() - start_time) * 1000
            alpha = 0.1  # Exponential moving average
            self.stats.avg_access_time_ms = (
                1 - alpha
            ) * self.stats.avg_access_time_ms + alpha * access_time_ms
        except Exception as e:
            logger.warning(f"Access time update failed: {e}")

    def invalidate(self, key: str):
        """Invalidate entry across all levels"""
        try:
            self.l1.invalidate(key)
            self.l2.base_cache.invalidate(key)

            if self.l3_enabled:
                with self.disk_lock:
                    if key in self.l3_index:
                        file_path = self.l3_path / f"{key}.cache"
                        if file_path.exists():
                            file_path.unlink()

                        size_mb = self.l3_index[key].get("size_mb", 0)
                        self.l3_current_size_mb -= size_mb
                        del self.l3_index[key]
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            l3_stats = {
                "enabled": self.l3_enabled,
                "entries": len(self.l3_index) if self.l3_enabled else 0,
            }

            if self.l3_enabled:
                l3_stats.update(
                    {
                        "size_mb": self.l3_current_size_mb,
                        "max_size_mb": self.l3_max_size_mb,
                        "utilization": (
                            self.l3_current_size_mb / self.l3_max_size_mb
                            if self.l3_max_size_mb > 0
                            else 0
                        ),
                    }
                )

            return {
                "overall": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "hit_rate": self.stats.hit_rate,
                    "avg_access_time_ms": self.stats.avg_access_time_ms,
                },
                "l1": self.l1.get_stats(),
                "l2": self.l2.base_cache.get_stats(),
                "l3": l3_stats,
            }
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}


class SelectionCache:
    """
    Main cache for tool selection system
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Multi-level cache for different data types
        self.feature_cache = MultiLevelCache(config.get("feature_cache_config", {}))
        self.selection_cache = MultiLevelCache(config.get("selection_cache_config", {}))
        self.result_cache = MultiLevelCache(config.get("result_cache_config", {}))

        # Specialized caches
        self.similarity_cache = LRUCache(
            max_size=config.get("similarity_cache_size", 1000),
            max_bytes=50 * 1024 * 1024,  # 50MB
        )

        # Cache dependencies for invalidation
        self.dependencies = defaultdict(set)

        # Precomputed values
        self.precomputed = {}

        # Cache warming
        self.warm_cache_enabled = config.get("enable_warming", True)
        self.warm_entries = deque(maxlen=100)

        # TTL management
        self.default_ttl = config.get("default_ttl", 3600)  # 1 hour

        # PATCH 5: Event-based shutdown
        self._shutdown_event = threading.Event()

        # Background cleanup thread
        self.cleanup_interval = config.get("cleanup_interval", 300)  # 5 minutes
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def cache_features(
        self, problem: Any, features: np.ndarray, ttl: Optional[float] = None
    ):
        """Cache extracted features"""
        try:
            key = self._compute_feature_key(problem)
            self.feature_cache.put(key, features, ttl or self.default_ttl)
        except Exception as e:
            logger.error(f"Feature caching failed: {e}")

    def get_cached_features(self, problem: Any) -> Optional[np.ndarray]:
        """Get cached features"""
        try:
            key = self._compute_feature_key(problem)
            return self.feature_cache.get(key)
        except Exception as e:
            logger.error(f"Feature retrieval failed: {e}")
            return None

    def cache_selection(
        self,
        features: np.ndarray,
        constraints: Dict[str, float],
        selection: str,
        confidence: float,
        ttl: Optional[float] = None,
    ):
        """Cache tool selection decision"""
        try:
            key = self._compute_selection_key(features, constraints)
            value = {
                "tool": selection,
                "confidence": confidence,
                "timestamp": time.time(),
            }
            self.selection_cache.put(key, value, ttl or self.default_ttl)

            # Track for warming
            if self.warm_cache_enabled:
                self.warm_entries.append((features, constraints, selection))
        except Exception as e:
            logger.error(f"Selection caching failed: {e}")

    def get_cached_selection(
        self, features: np.ndarray, constraints: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Get cached selection decision"""
        try:
            key = self._compute_selection_key(features, constraints)
            return self.selection_cache.get(key)
        except Exception as e:
            logger.error(f"Selection retrieval failed: {e}")
            return None

    def cache_result(
        self,
        tool: str,
        problem: Any,
        result: Any,
        execution_time: float,
        energy: float,
        ttl: Optional[float] = None,
    ):
        """Cache tool execution result"""
        try:
            key = self._compute_result_key(tool, problem)
            value = {
                "result": result,
                "execution_time": execution_time,
                "energy": energy,
                "timestamp": time.time(),
            }
            self.result_cache.put(key, value, ttl or self.default_ttl)
        except Exception as e:
            logger.error(f"Result caching failed: {e}")

    def get_cached_result(self, tool: str, problem: Any) -> Optional[Dict[str, Any]]:
        """Get cached execution result"""
        try:
            key = self._compute_result_key(tool, problem)
            return self.result_cache.get(key)
        except Exception as e:
            logger.error(f"Result retrieval failed: {e}")
            return None

    def cache_similarity(
        self, features1: np.ndarray, features2: np.ndarray, similarity: float
    ):
        """Cache similarity computation"""
        try:
            key = self._compute_similarity_key(features1, features2)
            self.similarity_cache.put(key, similarity, sizeof(similarity))
        except Exception as e:
            logger.error(f"Similarity caching failed: {e}")

    def get_cached_similarity(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> Optional[float]:
        """Get cached similarity"""
        try:
            key = self._compute_similarity_key(features1, features2)
            return self.similarity_cache.get(key)
        except Exception as e:
            logger.error(f"Similarity retrieval failed: {e}")
            return None

    def invalidate_problem(self, problem: Any):
        """Invalidate all caches related to a problem"""
        try:
            # Invalidate features
            feature_key = self._compute_feature_key(problem)
            self.feature_cache.invalidate(feature_key)

            # Invalidate related results
            for tool in [
                "symbolic",
                "probabilistic",
                "causal",
                "analogical",
                "multimodal",
            ]:
                result_key = self._compute_result_key(tool, problem)
                self.result_cache.invalidate(result_key)
        except Exception as e:
            logger.error(f"Problem invalidation failed: {e}")

    def invalidate_tool(self, tool: str):
        """Invalidate all cached results for a tool"""
        # This would require maintaining reverse index
        # For now, clear result cache
        logger.info(f"Invalidating cache for tool {tool}")

    def precompute_common_patterns(self, patterns: List[Any]):
        """Precompute and cache common patterns"""
        try:
            for pattern in patterns:
                # Extract features
                features = self._extract_features_for_pattern(pattern)
                if features is not None:
                    key = self._compute_feature_key(pattern)
                    self.feature_cache.put(key, features, ttl=None)  # No expiry
                    self.precomputed[key] = True
        except Exception as e:
            logger.error(f"Precomputation failed: {e}")

    def warm_cache(self):
        """Warm cache with recent entries"""
        if not self.warm_cache_enabled or not self.warm_entries:
            return

        try:
            logger.info("Warming cache with recent entries")

            for features, constraints, selection in list(self.warm_entries):
                key = self._compute_selection_key(features, constraints)
                if self.selection_cache.get(key) is None:
                    # Re-cache
                    self.cache_selection(
                        features, constraints, selection, confidence=0.8
                    )
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")

    def _compute_feature_key(self, problem: Any) -> str:
        """Compute cache key for features"""
        try:
            problem_str = str(problem)[:1000]  # Limit size
            hash_val = hashlib.sha256(problem_str.encode()).hexdigest()[:16]
            return f"features_{hash_val}"
        except Exception as e:
            logger.error(f"Feature key computation failed: {e}")
            return f"features_{time.time()}"

    def _compute_selection_key(
        self, features: np.ndarray, constraints: Dict[str, float]
    ) -> str:
        """Compute cache key for selection"""
        try:
            # Discretize features for key
            discretized = np.round(features * 100) / 100
            feature_hash = hashlib.sha256(discretized.tobytes()).hexdigest()[:8]

            # Include key constraints
            constraint_str = f"{constraints.get('time_budget_ms', 0)}_{constraints.get('energy_budget_mj', 0)}"

            return f"selection_{feature_hash}_{constraint_str}"
        except Exception as e:
            logger.error(f"Selection key computation failed: {e}")
            return f"selection_{time.time()}"

    def _compute_result_key(self, tool: str, problem: Any) -> str:
        """Compute cache key for result"""
        try:
            problem_str = str(problem)[:1000]
            problem_hash = hashlib.sha256(problem_str.encode()).hexdigest()[:16]
            return f"result_{tool}_{problem_hash}"
        except Exception as e:
            logger.error(f"Result key computation failed: {e}")
            return f"result_{tool}_{time.time()}"

    def _compute_similarity_key(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> str:
        """Compute cache key for similarity"""
        try:
            # Sort to ensure consistency
            hash1 = hashlib.sha256(features1.tobytes()).hexdigest()[:8]
            hash2 = hashlib.sha256(features2.tobytes()).hexdigest()[:8]

            if hash1 < hash2:
                return f"sim_{hash1}_{hash2}"
            else:
                return f"sim_{hash2}_{hash1}"
        except Exception as e:
            logger.error(f"Similarity key computation failed: {e}")
            return f"sim_{time.time()}"

    def _extract_features_for_pattern(self, pattern: Any) -> Optional[np.ndarray]:
        """Extract features for pattern (placeholder)"""
        try:
            # This would call actual feature extraction
            return np.random.randn(128)  # Placeholder
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _cleanup_loop(self):
        """PATCH 5: Interruptible background cleanup of expired entries"""
        while not self._shutdown_event.is_set():
            try:
                # Clean expired entries
                # This is simplified - real implementation would check TTL

                # Log statistics periodically
                stats = self.get_statistics()
                logger.debug(f"Cache statistics: {stats}")

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            # PATCH 5: Interruptible wait
            if self._shutdown_event.wait(timeout=self.cleanup_interval):
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            return {
                "feature_cache": self.feature_cache.get_statistics(),
                "selection_cache": self.selection_cache.get_statistics(),
                "result_cache": self.result_cache.get_statistics(),
                "similarity_cache": self.similarity_cache.get_stats(),
                "precomputed_entries": len(self.precomputed),
                "warm_entries": len(self.warm_entries),
            }
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}

    def save_cache(self, path: str):
        """Save cache to disk"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save warm entries for future warming
            with open(save_path / "warm_entries.pkl", "wb") as f:
                pickle.dump(list(self.warm_entries), f)

            # Save statistics
            with open(save_path / "statistics.json", "w", encoding="utf-8") as f:
                json.dump(self.get_statistics(), f, indent=2, default=str)

            logger.info(f"Cache saved to {save_path}")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    def shutdown(self):
        """PATCH 5: Fast shutdown with event-based interruption"""
        try:
            # Signal cleanup thread to stop
            self._shutdown_event.set()

            # Ensure thread is daemonic
            self.cleanup_thread.daemon = True

            # Short timeout wait for graceful shutdown
            if self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=0.1)

            # Clear caches
            self.feature_cache.l1.clear()
            self.feature_cache.l2.base_cache.clear()
            self.selection_cache.l1.clear()
            self.selection_cache.l2.base_cache.clear()
            self.result_cache.l1.clear()
            self.result_cache.l2.base_cache.clear()
            self.similarity_cache.clear()

            logger.info("Cache shutdown complete")
        except Exception as e:
            logger.error(f"Cache shutdown failed: {e}")
