"""
cache_manager.py - Unified cache management for semantic bridge
Part of the VULCAN-AGI system

Provides centralized cache management with memory limits, priority-based eviction,
and performance tracking across all caches in the system.
"""

import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified cache manager with memory limits and priority-based eviction

    Manages multiple caches across the system with:
    - Memory limit enforcement
    - Priority-based eviction (low priority caches cleared first)
    - Hit/miss tracking for performance monitoring
    - Callback support for cache clearing events
    """

    def __init__(self, max_memory_mb: int = 1000):
        """
        Initialize cache manager

        Args:
            max_memory_mb: Maximum total memory for all caches in megabytes
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.caches = {}  # name -> cache dict
        self.priorities = {}  # name -> priority (higher = keep longer)
        self.metadata = {}  # name -> metadata dict
        self._lock = threading.RLock()

        # Global statistics
        self.total_evictions = 0
        self.last_memory_check = time.time()
        self.memory_check_interval = 60.0  # Check every minute

        logger.info("CacheManager initialized with %d MB limit", max_memory_mb)

    def register_cache(
        self,
        name: str,
        cache: Dict,
        priority: int = 5,
        clear_callback: Optional[Callable] = None,
    ):
        """
        Register a cache for management

        Args:
            name: Cache identifier (must be unique)
            cache: The cache dictionary to manage
            priority: Priority level (1-10, higher = kept longer during eviction)
            clear_callback: Optional callback function called when cache is cleared
        """
        with self._lock:
            if name in self.caches:
                logger.warning("Cache '%s' already registered, updating", name)

            # Validate priority
            if not (1 <= priority <= 10):
                logger.warning("Priority %d out of range [1,10], clamping", priority)
                priority = max(1, min(10, priority))

            self.caches[name] = cache
            self.priorities[name] = priority
            self.metadata[name] = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "last_clear": time.time(),
                "clear_callback": clear_callback,
                "created": time.time(),
            }

            logger.debug("Registered cache '%s' with priority %d", name, priority)

    def unregister_cache(self, name: str):
        """
        Unregister a cache from management

        Args:
            name: Cache identifier
        """
        with self._lock:
            if name in self.caches:
                del self.caches[name]
                del self.priorities[name]
                del self.metadata[name]
                logger.debug("Unregistered cache '%s'", name)
            else:
                logger.warning("Cache '%s' not found for unregistration", name)

    def check_memory(self) -> Dict[str, Any]:
        """
        Check memory usage and evict if needed

        Returns:
            Dictionary with memory statistics and eviction info
        """
        with self._lock:
            self.last_memory_check = time.time()

            # Estimate total size
            total_size = 0
            cache_sizes = {}

            for name, cache in self.caches.items():
                try:
                    # Base size of cache dict
                    size = sys.getsizeof(cache)

                    # Sample entries to estimate content size
                    if len(cache) > 0:
                        sample_size = min(10, len(cache))
                        sample_items = list(cache.items())[:sample_size]

                        content_size = 0
                        for k, v in sample_items:
                            content_size += sys.getsizeof(k) + sys.getsizeof(v)

                        # Extrapolate to full cache
                        if len(cache) > sample_size:
                            avg_item_size = content_size / sample_size
                            size += avg_item_size * len(cache)
                        else:
                            size += content_size

                    cache_sizes[name] = size
                    total_size += size

                except Exception as e:
                    logger.debug("Error estimating size for cache '%s': %s", name, e)
                    cache_sizes[name] = 0

            result = {
                "total_mb": total_size / (1024 * 1024),
                "limit_mb": self.max_memory / (1024 * 1024),
                "usage_percent": (total_size / self.max_memory * 100)
                if self.max_memory > 0
                else 0,
                "cache_sizes": {k: v / (1024 * 1024) for k, v in cache_sizes.items()},
                "over_limit": total_size > self.max_memory,
                "evicted": [],
            }

            # Evict if over limit
            if total_size > self.max_memory:
                logger.info(
                    "Memory limit exceeded (%.1f / %.1f MB), evicting caches",
                    result["total_mb"],
                    result["limit_mb"],
                )
                evicted = self._evict_low_priority(cache_sizes, total_size)
                result["evicted"] = evicted
                self.total_evictions += len(evicted)

            return result

    def _evict_low_priority(
        self, cache_sizes: Dict[str, int], current_total: int
    ) -> List[Dict[str, Any]]:
        """
        Evict caches with lowest priority until under memory limit

        Args:
            cache_sizes: Dictionary of cache sizes in bytes
            current_total: Current total memory usage in bytes

        Returns:
            List of eviction records
        """
        evicted = []
        freed_memory = 0

        # Sort by priority (lower first), then by size (larger first)
        sorted_caches = sorted(
            self.priorities.items(), key=lambda x: (x[1], -cache_sizes.get(x[0], 0))
        )

        for name, priority in sorted_caches:
            cache_size = cache_sizes.get(name, 0)

            if cache_size > 0:
                cache = self.caches[name]
                original_size = len(cache)

                # Clear cache
                cache.clear()

                # Call callback if provided
                callback = self.metadata[name].get("clear_callback")
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(
                            "Cache clear callback failed for '%s': %s", name, e
                        )

                # Update metadata
                self.metadata[name]["evictions"] += 1
                self.metadata[name]["last_clear"] = time.time()

                freed_memory += cache_size

                evicted.append(
                    {
                        "cache": name,
                        "priority": priority,
                        "entries_cleared": original_size,
                        "bytes_freed": cache_size,
                        "mb_freed": cache_size / (1024 * 1024),
                    }
                )

                logger.info(
                    "Evicted cache '%s' (priority %d, %d entries, %.1f MB freed)",
                    name,
                    priority,
                    original_size,
                    cache_size / (1024 * 1024),
                )

                # Check if we've freed enough memory
                if (current_total - freed_memory) < self.max_memory:
                    logger.info(
                        "Memory now under limit after evicting %d caches", len(evicted)
                    )
                    break

        return evicted

    def _check_memory_ok(self) -> bool:
        """
        Quick check if memory is under limit (without full calculation)

        Returns:
            True if under limit, False otherwise
        """
        try:
            total = 0
            for cache in self.caches.values():
                total += sys.getsizeof(cache)
                if total > self.max_memory:
                    return False
            return True
        except Exception as e:
            logger.debug("Memory check failed: %s", e)
            return True  # Assume OK if check fails

    def record_hit(self, cache_name: str):
        """
        Record cache hit for statistics

        Args:
            cache_name: Name of cache that had a hit
        """
        with self._lock:
            if cache_name in self.metadata:
                self.metadata[cache_name]["hits"] += 1
            else:
                logger.debug(
                    "Cannot record hit for unregistered cache '%s'", cache_name
                )

    def record_miss(self, cache_name: str):
        """
        Record cache miss for statistics

        Args:
            cache_name: Name of cache that had a miss
        """
        with self._lock:
            if cache_name in self.metadata:
                self.metadata[cache_name]["misses"] += 1
            else:
                logger.debug(
                    "Cannot record miss for unregistered cache '%s'", cache_name
                )

    def clear_cache(self, cache_name: str, force: bool = False):
        """
        Clear a specific cache

        Args:
            cache_name: Name of cache to clear
            force: If True, clear even if high priority
        """
        with self._lock:
            if cache_name not in self.caches:
                logger.warning("Cache '%s' not found", cache_name)
                return

            priority = self.priorities[cache_name]

            # Protect high-priority caches unless forced
            if not force and priority >= 9:
                logger.warning(
                    "Refusing to clear high-priority cache '%s' without force=True",
                    cache_name,
                )
                return

            cache = self.caches[cache_name]
            original_size = len(cache)
            cache.clear()

            # Call callback if provided
            callback = self.metadata[cache_name].get("clear_callback")
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error(
                        "Cache clear callback failed for '%s': %s", cache_name, e
                    )

            # Update metadata
            self.metadata[cache_name]["evictions"] += 1
            self.metadata[cache_name]["last_clear"] = time.time()

            logger.info("Cleared cache '%s' (%d entries)", cache_name, original_size)

    def clear_all(self, force: bool = False):
        """
        Clear all registered caches

        Args:
            force: If True, clear even high-priority caches
        """
        with self._lock:
            cleared_count = 0

            for cache_name in list(self.caches.keys()):
                try:
                    self.clear_cache(cache_name, force=force)
                    cleared_count += 1
                except Exception as e:
                    logger.error("Failed to clear cache '%s': %s", cache_name, e)

            logger.info("Cleared %d caches (force=%s)", cleared_count, force)

    def get_cache_info(self, cache_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific cache

        Args:
            cache_name: Name of cache

        Returns:
            Cache information dictionary or None if not found
        """
        with self._lock:
            if cache_name not in self.caches:
                return None

            cache = self.caches[cache_name]
            meta = self.metadata[cache_name]
            total_requests = meta["hits"] + meta["misses"]

            return {
                "name": cache_name,
                "size": len(cache),
                "priority": self.priorities[cache_name],
                "hits": meta["hits"],
                "misses": meta["misses"],
                "total_requests": total_requests,
                "hit_rate": meta["hits"] / total_requests
                if total_requests > 0
                else 0.0,
                "evictions": meta["evictions"],
                "last_clear": meta["last_clear"],
                "age_seconds": time.time() - meta["created"],
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics for all registered caches

        Returns:
            Dictionary with statistics for each cache and overall stats
        """
        with self._lock:
            cache_stats = {}
            total_entries = 0
            total_hits = 0
            total_misses = 0

            for name in self.caches:
                cache = self.caches[name]
                meta = self.metadata[name]
                total_requests = meta["hits"] + meta["misses"]

                cache_stats[name] = {
                    "size": len(cache),
                    "priority": self.priorities[name],
                    "hits": meta["hits"],
                    "misses": meta["misses"],
                    "hit_rate": meta["hits"] / total_requests
                    if total_requests > 0
                    else 0.0,
                    "evictions": meta["evictions"],
                    "last_clear": meta["last_clear"],
                    "age_seconds": time.time() - meta["created"],
                }

                total_entries += len(cache)
                total_hits += meta["hits"]
                total_misses += meta["misses"]

            overall_requests = total_hits + total_misses

            return {
                "caches": cache_stats,
                "summary": {
                    "total_caches": len(self.caches),
                    "total_entries": total_entries,
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "overall_hit_rate": total_hits / overall_requests
                    if overall_requests > 0
                    else 0.0,
                    "total_evictions": self.total_evictions,
                    "last_memory_check": self.last_memory_check,
                    "memory_limit_mb": self.max_memory / (1024 * 1024),
                },
            }

    def update_priority(self, cache_name: str, new_priority: int):
        """
        Update priority of a cache

        Args:
            cache_name: Name of cache
            new_priority: New priority value (1-10)
        """
        with self._lock:
            if cache_name not in self.priorities:
                logger.warning("Cache '%s' not found for priority update", cache_name)
                return

            # Validate priority
            if not (1 <= new_priority <= 10):
                logger.warning(
                    "Priority %d out of range [1,10], clamping", new_priority
                )
                new_priority = max(1, min(10, new_priority))

            old_priority = self.priorities[cache_name]
            self.priorities[cache_name] = new_priority

            logger.info(
                "Updated priority for cache '%s': %d -> %d",
                cache_name,
                old_priority,
                new_priority,
            )

    def should_check_memory(self) -> bool:
        """
        Check if it's time for a periodic memory check

        Returns:
            True if memory should be checked
        """
        return (time.time() - self.last_memory_check) > self.memory_check_interval

    def auto_check_memory(self):
        """
        Automatically check memory if interval has passed

        Returns:
            Memory check result if checked, None otherwise
        """
        if self.should_check_memory():
            return self.check_memory()
        return None

    def get_cache_by_priority(self) -> List[Tuple[str, int, int]]:
        """
        Get list of caches sorted by priority

        Returns:
            List of tuples (cache_name, priority, size)
        """
        with self._lock:
            result = []
            for name in self.caches:
                result.append((name, self.priorities[name], len(self.caches[name])))

            # Sort by priority (descending), then by size (descending)
            result.sort(key=lambda x: (-x[1], -x[2]))
            return result

    def optimize_priorities(self):
        """
        Optimize cache priorities based on hit rates

        Increases priority of high-hit-rate caches and decreases priority
        of low-hit-rate caches.
        """
        with self._lock:
            for name in self.caches:
                meta = self.metadata[name]
                total_requests = meta["hits"] + meta["misses"]

                if total_requests < 100:
                    continue  # Not enough data

                hit_rate = meta["hits"] / total_requests
                current_priority = self.priorities[name]

                # Adjust priority based on hit rate
                if hit_rate > 0.8 and current_priority < 10:
                    # High hit rate - increase priority
                    self.priorities[name] = min(10, current_priority + 1)
                    logger.debug(
                        "Increased priority for '%s' to %d (hit rate: %.2f)",
                        name,
                        self.priorities[name],
                        hit_rate,
                    )
                elif hit_rate < 0.3 and current_priority > 1:
                    # Low hit rate - decrease priority
                    self.priorities[name] = max(1, current_priority - 1)
                    logger.debug(
                        "Decreased priority for '%s' to %d (hit rate: %.2f)",
                        name,
                        self.priorities[name],
                        hit_rate,
                    )

    def __repr__(self) -> str:
        """String representation"""
        return f"CacheManager(caches={len(self.caches)}, limit={self.max_memory / (1024 * 1024):.1f}MB)"
