"""
Local File System Cache

This module provides comprehensive local caching with support for eviction policies,
size limits, TTL, statistics, and concurrent access.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """
    Metadata for a cached item.

    Attributes:
        key: Cache key
        path: File path
        size: Size in bytes
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        access_count: Number of accesses
        ttl: Time-to-live in seconds
    """

    key: str
    path: Path
    size: int
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_size": self.total_size,
            "entry_count": self.entry_count,
            "hit_rate": self.hit_rate,
        }


class LocalCache:
    """
    Local file system cache with eviction policies and size limits.

    Features:
    - Multiple eviction policies (LRU, LFU, FIFO, TTL)
    - Size-based limits
    - TTL support
    - Thread-safe operations
    - Comprehensive statistics
    - Atomic writes

    Example:
        cache = LocalCache(root=Path("/tmp/cache"), max_size=1_000_000_000)

        # Store data
        cache.put("user/profile", b"data...")

        # Retrieve data
        data = cache.get("user/profile")

        # Check stats
        stats = cache.get_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
    """

    def __init__(
        self,
        root: Path,
        max_size: int = 10 * 1024 * 1024 * 1024,  # 10 GB
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: Optional[float] = None,
        enable_compression: bool = False,
    ):
        """
        Initialize local cache.

        Args:
            root: Root directory for cache
            max_size: Maximum cache size in bytes
            eviction_policy: Eviction policy to use
            default_ttl: Default TTL for entries in seconds
            enable_compression: Whether to compress cached data
        """
        self.root = Path(root)
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression

        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)

        # Entry tracking
        self.entries: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()

        # Statistics
        self.stats = CacheStats()

        # Load existing entries
        self._load_existing_entries()

        logger.info(
            f"Initialized LocalCache: root={root}, max_size={max_size}, "
            f"policy={eviction_policy.value}"
        )

    def _load_existing_entries(self) -> None:
        """Load metadata for existing cached files"""
        try:
            for file_path in self.root.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.root)
                    key = str(rel_path)

                    stat = file_path.stat()
                    entry = CacheEntry(
                        key=key,
                        path=file_path,
                        size=stat.st_size,
                        created_at=stat.st_ctime,
                        accessed_at=stat.st_atime,
                    )

                    self.entries[key] = entry
                    self.stats.total_size += entry.size
                    self.stats.entry_count += 1

            logger.info(
                f"Loaded {len(self.entries)} existing entries "
                f"({self.stats.total_size / 1024 / 1024:.2f} MB)"
            )
        except Exception as e:
            logger.error(f"Failed to load existing entries: {e}")

    def _normalize_key(self, key: str) -> str:
        """Normalize cache key"""
        return key.strip("/")

    def _get_path(self, key: str) -> Path:
        """Get file path for key"""
        return self.root / self._normalize_key(key)

    def get(self, rel: str, touch: bool = True) -> Optional[bytes]:
        """
        Get cached data.

        Args:
            rel: Relative cache key
            touch: Whether to update access metadata

        Returns:
            Cached bytes, or None if not found
        """
        with self.lock:
            key = self._normalize_key(rel)
            entry = self.entries.get(key)

            if entry is None:
                self.stats.misses += 1
                logger.debug(f"Cache miss: {key}")
                return None

            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                logger.debug(f"Cache miss (expired): {key}")
                return None

            # Check if file exists
            if not entry.path.exists():
                self._remove_entry(key)
                self.stats.misses += 1
                logger.debug(f"Cache miss (file missing): {key}")
                return None

            # Read data
            try:
                data = entry.path.read_bytes()

                # Update metadata
                if touch:
                    entry.touch()

                self.stats.hits += 1
                logger.debug(f"Cache hit: {key} ({len(data)} bytes)")

                return data

            except Exception as e:
                logger.error(f"Failed to read cache entry {key}: {e}")
                self._remove_entry(key)
                self.stats.misses += 1
                return None

    def put(self, rel: str, data: bytes, ttl: Optional[float] = None) -> bool:
        """
        Put data in cache.

        Args:
            rel: Relative cache key
            data: Data to cache
            ttl: Optional TTL override

        Returns:
            True if successfully cached
        """
        with self.lock:
            key = self._normalize_key(rel)
            path = self._get_path(key)

            # Check if we need to evict
            data_size = len(data)
            self._ensure_space(data_size)

            # Atomic write
            try:
                # Create parent directories
                path.parent.mkdir(parents=True, exist_ok=True)

                # Write to temporary file first
                temp_path = path.with_suffix(path.suffix + ".tmp")
                temp_path.write_bytes(data)

                # Atomic rename
                temp_path.replace(path)

                # Update or create entry
                if key in self.entries:
                    old_entry = self.entries[key]
                    self.stats.total_size -= old_entry.size

                entry = CacheEntry(
                    key=key, path=path, size=data_size, ttl=ttl or self.default_ttl
                )

                self.entries[key] = entry
                self.stats.total_size += data_size
                self.stats.entry_count = len(self.entries)

                logger.debug(f"Cached: {key} ({data_size} bytes)")
                return True

            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
                return False

    def _ensure_space(self, required_size: int) -> None:
        """Ensure sufficient space by evicting entries"""
        while self.stats.total_size + required_size > self.max_size:
            if not self.entries:
                break

            # Select entry to evict based on policy
            victim_key = self._select_eviction_victim()
            if victim_key:
                self._remove_entry(victim_key)
                self.stats.evictions += 1
            else:
                break

    def _select_eviction_victim(self) -> Optional[str]:
        """Select entry to evict based on policy"""
        if not self.entries:
            return None

        if self.eviction_policy == EvictionPolicy.LRU:
            # Least recently accessed
            return min(self.entries.items(), key=lambda x: x[1].accessed_at)[0]

        elif self.eviction_policy == EvictionPolicy.LFU:
            # Least frequently accessed
            return min(self.entries.items(), key=lambda x: x[1].access_count)[0]

        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Oldest entry
            return min(self.entries.items(), key=lambda x: x[1].created_at)[0]

        elif self.eviction_policy == EvictionPolicy.TTL:
            # Expired or oldest
            expired = [k for k, e in self.entries.items() if e.is_expired()]
            if expired:
                return expired[0]
            return min(self.entries.items(), key=lambda x: x[1].created_at)[0]

        return None

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        entry = self.entries.get(key)
        if not entry:
            return

        try:
            if entry.path.exists():
                entry.path.unlink()

            self.stats.total_size -= entry.size
            del self.entries[key]
            self.stats.entry_count = len(self.entries)

            logger.debug(f"Removed cache entry: {key}")
        except Exception as e:
            logger.error(f"Failed to remove cache entry {key}: {e}")

    def delete(self, rel: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            key = self._normalize_key(rel)
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False

    def has(self, rel: str) -> bool:
        """Check if key exists in cache"""
        with self.lock:
            key = self._normalize_key(rel)
            entry = self.entries.get(key)

            if entry is None:
                return False

            if entry.is_expired():
                self._remove_entry(key)
                return False

            return entry.path.exists()

    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            for key in list(self.entries.keys()):
                self._remove_entry(key)

            # Remove empty directories
            try:
                for item in self.root.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
            except Exception as e:
                logger.error(f"Failed to clear cache directories: {e}")

            logger.info("Cleared cache")

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self.lock:
            expired = [k for k, e in self.entries.items() if e.is_expired()]

            for key in expired:
                self._remove_entry(key)

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired entries")

            return len(expired)

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size=self.stats.total_size,
                entry_count=len(self.entries),
            )

    def list_keys(self) -> List[str]:
        """List all cached keys"""
        with self.lock:
            return list(self.entries.keys())

    def get_size(self) -> int:
        """Get total cache size in bytes"""
        with self.lock:
            return self.stats.total_size

    def __len__(self) -> int:
        """Return number of cached entries"""
        with self.lock:
            return len(self.entries)

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.has(key)


def create_cache(root: Path, **kwargs) -> LocalCache:
    """
    Convenience function to create a cache.

    Args:
        root: Cache root directory
        **kwargs: Additional arguments for LocalCache

    Returns:
        LocalCache instance
    """
    return LocalCache(root=root, **kwargs)
