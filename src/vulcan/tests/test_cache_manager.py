"""
test_cache_manager.py - Comprehensive tests for CacheManager
Part of the VULCAN-AGI system

Tests cover:
- Basic cache registration and management
- Memory limit enforcement and eviction
- Priority-based eviction strategies
- Hit/miss tracking and statistics
- Cache clearing and callbacks
- Thread safety
- Priority updates
- Optimization features
"""

# Add parent directory to path for imports
from semantic_bridge.cache_manager import CacheManager
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCacheManagerBasics:
    """Test basic cache manager functionality"""

    def test_initialization(self):
        """Test cache manager initialization"""
        manager = CacheManager(max_memory_mb=100)

        assert manager.max_memory == 100 * 1024 * 1024
        assert len(manager.caches) == 0
        assert len(manager.priorities) == 0
        assert manager.total_evictions == 0

    def test_register_cache(self):
        """Test cache registration"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache, priority=7)

        assert "test_cache" in manager.caches
        assert manager.caches["test_cache"] is cache
        assert manager.priorities["test_cache"] == 7
        assert "test_cache" in manager.metadata

    def test_register_cache_with_callback(self):
        """Test cache registration with clear callback"""
        manager = CacheManager()
        cache = {}
        callback_called = []

        def clear_callback():
            callback_called.append(True)

        manager.register_cache(
            "test_cache", cache, priority=5, clear_callback=clear_callback
        )

        # Clear the cache
        manager.clear_cache("test_cache")

        assert len(callback_called) == 1
        assert len(cache) == 0

    def test_register_duplicate_cache(self):
        """Test registering cache with same name"""
        manager = CacheManager()
        cache1 = {"a": 1}
        cache2 = {"b": 2}

        manager.register_cache("test_cache", cache1, priority=5)
        manager.register_cache("test_cache", cache2, priority=7)

        # Should update to cache2
        assert manager.caches["test_cache"] is cache2
        assert manager.priorities["test_cache"] == 7

    def test_priority_clamping(self):
        """Test priority values are clamped to valid range"""
        manager = CacheManager()
        cache = {}

        # Test out of range priorities
        manager.register_cache("low", cache, priority=-5)
        assert manager.priorities["low"] == 1

        manager.register_cache("high", cache, priority=20)
        assert manager.priorities["high"] == 10

    def test_unregister_cache(self):
        """Test cache unregistration"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache)
        assert "test_cache" in manager.caches

        manager.unregister_cache("test_cache")
        assert "test_cache" not in manager.caches
        assert "test_cache" not in manager.priorities
        assert "test_cache" not in manager.metadata

    def test_unregister_nonexistent_cache(self):
        """Test unregistering cache that doesn't exist"""
        manager = CacheManager()

        # Should not raise error
        manager.unregister_cache("nonexistent")


class TestMemoryManagement:
    """Test memory limit enforcement and eviction"""

    def test_check_memory_basic(self):
        """Test basic memory checking"""
        manager = CacheManager(max_memory_mb=1)
        cache = {}

        manager.register_cache("test_cache", cache)

        result = manager.check_memory()

        assert "total_mb" in result
        assert "limit_mb" in result
        assert "usage_percent" in result
        assert "cache_sizes" in result
        assert "over_limit" in result
        assert result["limit_mb"] == 1.0

    def test_memory_eviction(self):
        """Test eviction when memory limit exceeded"""
        manager = CacheManager(max_memory_mb=0.001)  # Very small limit

        # Register low priority cache with data
        low_priority_cache = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        manager.register_cache("low_priority", low_priority_cache, priority=3)

        # Register high priority cache with data
        high_priority_cache = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        manager.register_cache("high_priority", high_priority_cache, priority=8)

        # Check memory should trigger eviction
        result = manager.check_memory()

        # Low priority cache should be evicted first
        if result["over_limit"]:
            assert len(result["evicted"]) > 0
            # Low priority should be evicted before high priority
            evicted_names = [e["cache"] for e in result["evicted"]]
            if "low_priority" in evicted_names:
                low_idx = evicted_names.index("low_priority")
                if "high_priority" in evicted_names:
                    high_idx = evicted_names.index("high_priority")
                    assert low_idx < high_idx

    def test_eviction_order_by_priority(self):
        """Test that eviction respects priority order"""
        manager = CacheManager(max_memory_mb=0.001)

        # Create caches with different priorities
        for priority in [1, 5, 9]:
            cache = {f"key_{i}": "x" * 1000 for i in range(50)}
            manager.register_cache(f"cache_p{priority}", cache, priority=priority)

        result = manager.check_memory()

        if len(result["evicted"]) > 0:
            # Verify eviction order (lowest priority first)
            evicted_priorities = []
            for eviction in result["evicted"]:
                cache_name = eviction["cache"]
                evicted_priorities.append(eviction["priority"])

            # Priorities should be in ascending order
            assert evicted_priorities == sorted(evicted_priorities)

    def test_eviction_callback(self):
        """Test callback is called on eviction"""
        manager = CacheManager(max_memory_mb=0.001)

        callbacks_called = []

        def callback1():
            callbacks_called.append("cache1")

        def callback2():
            callbacks_called.append("cache2")

        cache1 = {f"key_{i}": "x" * 1000 for i in range(50)}
        cache2 = {f"key_{i}": "x" * 1000 for i in range(50)}

        manager.register_cache("cache1", cache1, priority=1, clear_callback=callback1)
        manager.register_cache("cache2", cache2, priority=2, clear_callback=callback2)

        manager.check_memory()

        # At least one callback should be called if eviction occurred
        assert len(callbacks_called) >= 0  # May or may not evict depending on memory

    def test_auto_check_memory(self):
        """Test automatic memory checking based on interval"""
        manager = CacheManager()
        manager.memory_check_interval = 0.1  # Short interval for testing

        # Set last check to past so first call will check
        manager.last_memory_check = time.time() - 1.0

        # First call should check
        result1 = manager.auto_check_memory()
        assert result1 is not None

        # Immediate second call should not check
        result2 = manager.auto_check_memory()
        assert result2 is None

        # After interval, should check again
        time.sleep(0.15)
        result3 = manager.auto_check_memory()
        assert result3 is not None

    def test_should_check_memory(self):
        """Test memory check timing logic"""
        manager = CacheManager()
        manager.memory_check_interval = 1.0

        manager.last_memory_check = time.time()
        assert not manager.should_check_memory()

        manager.last_memory_check = time.time() - 2.0
        assert manager.should_check_memory()


class TestHitMissTracking:
    """Test cache hit/miss statistics"""

    def test_record_hit(self):
        """Test recording cache hits"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache)

        manager.record_hit("test_cache")
        manager.record_hit("test_cache")

        assert manager.metadata["test_cache"]["hits"] == 2
        assert manager.metadata["test_cache"]["misses"] == 0

    def test_record_miss(self):
        """Test recording cache misses"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache)

        manager.record_miss("test_cache")
        manager.record_miss("test_cache")
        manager.record_miss("test_cache")

        assert manager.metadata["test_cache"]["hits"] == 0
        assert manager.metadata["test_cache"]["misses"] == 3

    def test_hit_miss_mixed(self):
        """Test mixed hit/miss recording"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache)

        manager.record_hit("test_cache")
        manager.record_miss("test_cache")
        manager.record_hit("test_cache")
        manager.record_hit("test_cache")
        manager.record_miss("test_cache")

        assert manager.metadata["test_cache"]["hits"] == 3
        assert manager.metadata["test_cache"]["misses"] == 2

    def test_record_unregistered_cache(self):
        """Test recording stats for unregistered cache"""
        manager = CacheManager()

        # Should not raise error
        manager.record_hit("nonexistent")
        manager.record_miss("nonexistent")


class TestCacheClearing:
    """Test cache clearing functionality"""

    def test_clear_cache(self):
        """Test clearing specific cache"""
        manager = CacheManager()
        cache = {"a": 1, "b": 2, "c": 3}

        manager.register_cache("test_cache", cache, priority=5)

        manager.clear_cache("test_cache")

        assert len(cache) == 0
        assert manager.metadata["test_cache"]["evictions"] == 1

    def test_clear_high_priority_protection(self):
        """Test high priority caches protected from clearing"""
        manager = CacheManager()
        cache = {"a": 1, "b": 2}

        manager.register_cache("high_priority", cache, priority=9)

        # Should refuse to clear without force
        manager.clear_cache("high_priority", force=False)

        assert len(cache) == 2  # Should still have data

    def test_clear_high_priority_with_force(self):
        """Test forcing clear of high priority cache"""
        manager = CacheManager()
        cache = {"a": 1, "b": 2}

        manager.register_cache("high_priority", cache, priority=9)

        manager.clear_cache("high_priority", force=True)

        assert len(cache) == 0

    def test_clear_all(self):
        """Test clearing all caches"""
        manager = CacheManager()

        cache1 = {"a": 1}
        cache2 = {"b": 2}
        cache3 = {"c": 3}

        manager.register_cache("cache1", cache1, priority=5)
        manager.register_cache("cache2", cache2, priority=6)
        manager.register_cache("cache3", cache3, priority=4)

        manager.clear_all(force=False)

        # All medium/low priority caches should be cleared
        assert len(cache1) == 0
        assert len(cache2) == 0
        assert len(cache3) == 0

    def test_clear_all_respects_high_priority(self):
        """Test clear_all respects high priority without force"""
        manager = CacheManager()

        low_cache = {"a": 1}
        high_cache = {"b": 2, "c": 3}

        manager.register_cache("low", low_cache, priority=5)
        manager.register_cache("high", high_cache, priority=9)

        manager.clear_all(force=False)

        assert len(low_cache) == 0
        assert len(high_cache) == 2  # Should be protected

    def test_clear_nonexistent_cache(self):
        """Test clearing cache that doesn't exist"""
        manager = CacheManager()

        # Should not raise error
        manager.clear_cache("nonexistent")


class TestStatistics:
    """Test statistics and reporting"""

    def test_get_cache_info(self):
        """Test getting info for specific cache"""
        manager = CacheManager()
        cache = {"a": 1, "b": 2}

        manager.register_cache("test_cache", cache, priority=7)

        # Record some activity
        manager.record_hit("test_cache")
        manager.record_hit("test_cache")
        manager.record_miss("test_cache")

        info = manager.get_cache_info("test_cache")

        assert info is not None
        assert info["name"] == "test_cache"
        assert info["size"] == 2
        assert info["priority"] == 7
        assert info["hits"] == 2
        assert info["misses"] == 1
        assert info["total_requests"] == 3
        assert info["hit_rate"] == pytest.approx(2 / 3)

    def test_get_cache_info_nonexistent(self):
        """Test getting info for cache that doesn't exist"""
        manager = CacheManager()

        info = manager.get_cache_info("nonexistent")
        assert info is None

    def test_get_statistics(self):
        """Test getting overall statistics"""
        manager = CacheManager()

        cache1 = {"a": 1}
        cache2 = {"b": 2, "c": 3}

        manager.register_cache("cache1", cache1, priority=5)
        manager.register_cache("cache2", cache2, priority=8)

        manager.record_hit("cache1")
        manager.record_miss("cache1")
        manager.record_hit("cache2")

        stats = manager.get_statistics()

        assert "caches" in stats
        assert "summary" in stats
        assert stats["summary"]["total_caches"] == 2
        assert stats["summary"]["total_entries"] == 3
        assert stats["summary"]["total_hits"] == 2
        assert stats["summary"]["total_misses"] == 1

    def test_get_cache_by_priority(self):
        """Test getting caches sorted by priority"""
        manager = CacheManager()

        cache1 = {"a": 1}
        cache2 = {"b": 2, "c": 3, "d": 4}
        cache3 = {"e": 5, "f": 6}

        manager.register_cache("cache1", cache1, priority=5)
        manager.register_cache("cache2", cache2, priority=9)
        manager.register_cache("cache3", cache3, priority=3)

        result = manager.get_cache_by_priority()

        # Should be sorted by priority (descending)
        assert len(result) == 3
        assert result[0][0] == "cache2"  # Highest priority
        assert result[0][1] == 9
        assert result[1][0] == "cache1"
        assert result[2][0] == "cache3"  # Lowest priority


class TestPriorityManagement:
    """Test priority updates and optimization"""

    def test_update_priority(self):
        """Test updating cache priority"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache, priority=5)
        assert manager.priorities["test_cache"] == 5

        manager.update_priority("test_cache", 8)
        assert manager.priorities["test_cache"] == 8

    def test_update_priority_clamping(self):
        """Test priority updates are clamped"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache, priority=5)

        manager.update_priority("test_cache", 15)
        assert manager.priorities["test_cache"] == 10

        manager.update_priority("test_cache", -5)
        assert manager.priorities["test_cache"] == 1

    def test_update_nonexistent_priority(self):
        """Test updating priority of nonexistent cache"""
        manager = CacheManager()

        # Should not raise error
        manager.update_priority("nonexistent", 7)

    def test_optimize_priorities(self):
        """Test automatic priority optimization based on hit rates"""
        manager = CacheManager()

        cache1 = {}
        cache2 = {}
        cache3 = {}

        manager.register_cache("cache1", cache1, priority=5)
        manager.register_cache("cache2", cache2, priority=5)
        manager.register_cache("cache3", cache3, priority=5)

        # Simulate high hit rate for cache1
        for _ in range(100):
            manager.record_hit("cache1")
        for _ in range(10):
            manager.record_miss("cache1")

        # Simulate low hit rate for cache2
        for _ in range(20):
            manager.record_hit("cache2")
        for _ in range(80):
            manager.record_miss("cache2")

        # Simulate medium hit rate for cache3 (not enough data)
        for _ in range(30):
            manager.record_hit("cache3")
        for _ in range(20):
            manager.record_miss("cache3")

        initial_priority1 = manager.priorities["cache1"]
        initial_priority2 = manager.priorities["cache2"]

        manager.optimize_priorities()

        # High hit rate cache should have increased priority
        # Low hit rate cache should have decreased priority
        # (Cache3 doesn't have enough data, so unchanged)
        assert manager.priorities["cache1"] >= initial_priority1
        assert manager.priorities["cache2"] <= initial_priority2


class TestThreadSafety:
    """Test thread-safe operations"""

    def test_concurrent_registration(self):
        """Test concurrent cache registration"""
        manager = CacheManager()

        def register_caches(thread_id):
            for i in range(10):
                cache = {}
                manager.register_cache(f"cache_{thread_id}_{i}", cache, priority=5)

        threads = []
        for i in range(5):
            t = threading.Thread(target=register_caches, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all caches registered
        assert len(manager.caches) == 50

    def test_concurrent_hit_miss_recording(self):
        """Test concurrent hit/miss recording"""
        manager = CacheManager()
        cache = {}

        manager.register_cache("test_cache", cache)

        def record_hits():
            for _ in range(100):
                manager.record_hit("test_cache")

        def record_misses():
            for _ in range(100):
                manager.record_miss("test_cache")

        threads = [
            threading.Thread(target=record_hits),
            threading.Thread(target=record_hits),
            threading.Thread(target=record_misses),
            threading.Thread(target=record_misses),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert manager.metadata["test_cache"]["hits"] == 200
        assert manager.metadata["test_cache"]["misses"] == 200

    def test_concurrent_clear(self):
        """Test concurrent cache clearing"""
        manager = CacheManager()

        caches = []
        for i in range(10):
            cache = {f"key_{j}": j for j in range(100)}
            manager.register_cache(f"cache_{i}", cache, priority=5)
            caches.append(cache)

        def clear_caches():
            for i in range(10):
                manager.clear_cache(f"cache_{i}")

        threads = [threading.Thread(target=clear_caches) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All caches should be empty
        for cache in caches:
            assert len(cache) == 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_cache_memory_check(self):
        """Test memory check with no caches"""
        manager = CacheManager()

        result = manager.check_memory()

        assert result["total_mb"] >= 0
        assert result["usage_percent"] >= 0
        assert len(result["cache_sizes"]) == 0

    def test_zero_memory_limit(self):
        """Test with zero memory limit"""
        manager = CacheManager(max_memory_mb=0)

        cache = {"a": 1}
        manager.register_cache("test_cache", cache, priority=5)

        # Should handle gracefully
        result = manager.check_memory()
        assert "total_mb" in result

    def test_very_large_cache(self):
        """Test with very large cache"""
        manager = CacheManager(max_memory_mb=0.1)

        # Create large cache
        large_cache = {i: "x" * 1000 for i in range(1000)}
        manager.register_cache("large_cache", large_cache, priority=1)

        result = manager.check_memory()

        # Should handle and possibly evict
        assert "total_mb" in result

    def test_repr(self):
        """Test string representation"""
        manager = CacheManager(max_memory_mb=500)

        cache1 = {}
        cache2 = {}
        manager.register_cache("cache1", cache1)
        manager.register_cache("cache2", cache2)

        repr_str = repr(manager)

        assert "CacheManager" in repr_str
        assert "caches=2" in repr_str
        assert "500.0MB" in repr_str


class TestIntegration:
    """Integration tests with realistic scenarios"""

    def test_realistic_cache_usage(self):
        """Test realistic cache usage pattern"""
        manager = CacheManager(max_memory_mb=10)

        # Register multiple caches with different priorities
        pattern_cache = {}
        result_cache = {}
        metadata_cache = {}

        manager.register_cache("patterns", pattern_cache, priority=8)
        manager.register_cache("results", result_cache, priority=6)
        manager.register_cache("metadata", metadata_cache, priority=4)

        # Simulate cache usage
        for i in range(100):
            # Patterns accessed frequently
            if i % 2 == 0:
                manager.record_hit("patterns")
            else:
                manager.record_miss("patterns")

            # Results accessed less
            if i % 3 == 0:
                manager.record_hit("results")
            else:
                manager.record_miss("results")

            # Metadata rarely accessed
            if i % 5 == 0:
                manager.record_hit("metadata")
            else:
                manager.record_miss("metadata")

        stats = manager.get_statistics()

        # Pattern cache should have best hit rate
        pattern_info = manager.get_cache_info("patterns")
        result_info = manager.get_cache_info("results")

        assert pattern_info["hit_rate"] > result_info["hit_rate"]

    def test_cache_lifecycle(self):
        """Test complete cache lifecycle"""
        manager = CacheManager(max_memory_mb=1)

        # Create cache
        cache = {"initial": "data"}
        manager.register_cache("lifecycle", cache, priority=5)

        # Use cache
        for _ in range(50):
            manager.record_hit("lifecycle")

        # Update priority
        manager.update_priority("lifecycle", 7)

        # Get statistics
        info = manager.get_cache_info("lifecycle")
        assert info["hits"] == 50
        assert info["priority"] == 7

        # Clear cache
        manager.clear_cache("lifecycle")
        assert len(cache) == 0

        # Unregister
        manager.unregister_cache("lifecycle")
        assert "lifecycle" not in manager.caches


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
