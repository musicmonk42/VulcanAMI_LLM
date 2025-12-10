"""
Comprehensive tests for selection_cache.py

Tests all cache levels, eviction policies, compression, thread safety,
and the complete SelectionCache API.
"""

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the cache module
from vulcan.reasoning.selection.selection_cache import (CacheEntry, CacheLevel,
                                                        CacheStatistics,
                                                        CompressedCache,
                                                        EvictionPolicy,
                                                        LRUCache,
                                                        MultiLevelCache,
                                                        SelectionCache, sizeof)


class TestSizeOf:
    """Test the sizeof utility function"""

    def test_sizeof_basic_types(self):
        """Test sizeof with basic Python types"""
        assert sizeof(42) > 0
        assert sizeof("hello") > 0
        assert sizeof([1, 2, 3]) > 0
        assert sizeof({"key": "value"}) > 0

    def test_sizeof_numpy_array(self):
        """Test sizeof with numpy arrays"""
        arr = np.random.randn(100)
        size = sizeof(arr)
        assert size > 0
        assert size > 100  # Should be at least as large as array

    def test_sizeof_fallback(self):
        """Test sizeof fallback for complex objects"""

        class CustomClass:
            def __init__(self):
                self.data = "test"

        obj = CustomClass()
        size = sizeof(obj)
        assert size > 0


class TestCacheEntry:
    """Test CacheEntry dataclass"""

    def test_cache_entry_creation(self):
        """Test creating cache entry"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=100,
            creation_time=time.time(),
            last_access_time=time.time(),
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 100
        assert entry.access_count == 0

    def test_cache_entry_ttl_not_expired(self):
        """Test entry TTL when not expired"""
        entry = CacheEntry(
            key="test",
            value="value",
            size_bytes=100,
            creation_time=time.time(),
            last_access_time=time.time(),
            ttl_seconds=3600,
        )

        assert not entry.is_expired()

    def test_cache_entry_ttl_expired(self):
        """Test entry TTL when expired"""
        entry = CacheEntry(
            key="test",
            value="value",
            size_bytes=100,
            creation_time=time.time() - 7200,  # 2 hours ago
            last_access_time=time.time() - 7200,
            ttl_seconds=3600,  # 1 hour TTL
        )

        assert entry.is_expired()

    def test_cache_entry_touch(self):
        """Test entry touch updates"""
        entry = CacheEntry(
            key="test",
            value="value",
            size_bytes=100,
            creation_time=time.time(),
            last_access_time=time.time(),
        )

        initial_count = entry.access_count
        initial_time = entry.last_access_time

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_access_time > initial_time


class TestCacheStatistics:
    """Test CacheStatistics dataclass"""

    def test_statistics_defaults(self):
        """Test default statistics values"""
        stats = CacheStatistics()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = CacheStatistics(hits=75, misses=25)

        assert stats.hit_rate == 0.75

    def test_hit_rate_no_accesses(self):
        """Test hit rate with no accesses"""
        stats = CacheStatistics(hits=0, misses=0)

        assert stats.hit_rate == 0.0


class TestLRUCache:
    """Test LRU cache implementation"""

    def test_lru_cache_creation(self):
        """Test creating LRU cache"""
        cache = LRUCache(max_size=10)

        assert cache.max_size == 10
        assert cache.max_bytes == 100 * 1024 * 1024
        assert cache.size_bytes == 0

    def test_lru_put_and_get(self):
        """Test basic put and get operations"""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", 100)
        result = cache.get("key1")

        assert result == "value1"
        assert cache.size_bytes == 100

    def test_lru_get_missing(self):
        """Test getting non-existent key"""
        cache = LRUCache(max_size=10)

        result = cache.get("missing_key")

        assert result is None

    def test_lru_eviction_by_size(self):
        """Test LRU eviction when max size reached"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 100)
        cache.put("key3", "value3", 100)
        cache.put("key4", "value4", 100)  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
        assert len(cache.cache) == 3

    def test_lru_eviction_by_bytes(self):
        """Test LRU eviction when max bytes reached"""
        cache = LRUCache(max_size=100, max_bytes=250)

        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 100)
        cache.put("key3", "value3", 100)  # Should evict key1

        assert cache.get("key1") is None
        assert cache.size_bytes <= 250

    def test_lru_order_preservation(self):
        """Test LRU order is maintained"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 100)
        cache.put("key3", "value3", 100)

        # Access key1 to move it to end
        cache.get("key1")

        # Add new key, should evict key2
        cache.put("key4", "value4", 100)

        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"

    def test_lru_update_existing(self):
        """Test updating existing key"""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", 100)
        cache.put("key1", "value2", 150)

        assert cache.get("key1") == "value2"
        assert cache.size_bytes == 150

    def test_lru_invalidate(self):
        """Test invalidating entry"""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", 100)
        result = cache.invalidate("key1")

        assert result is True
        assert cache.get("key1") is None
        assert cache.size_bytes == 0

    def test_lru_invalidate_missing(self):
        """Test invalidating non-existent key"""
        cache = LRUCache(max_size=10)

        result = cache.invalidate("missing")

        assert result is False

    def test_lru_clear(self):
        """Test clearing cache"""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 100)
        cache.clear()

        assert len(cache.cache) == 0
        assert cache.size_bytes == 0

    def test_lru_get_stats(self):
        """Test getting cache statistics"""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)

        stats = cache.get_stats()

        assert stats["entries"] == 2
        assert stats["size_bytes"] == 300
        assert stats["max_size"] == 10

    def test_lru_thread_safety(self):
        """Test thread safety of LRU cache"""
        cache = LRUCache(max_size=100)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    cache.put(f"key_{worker_id}_{i}", f"value_{i}", 100)
                    cache.get(f"key_{worker_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCompressedCache:
    """Test compressed cache implementation"""

    def test_compressed_cache_creation(self):
        """Test creating compressed cache"""
        base = LRUCache(max_size=10)
        cache = CompressedCache(base, compression_threshold=1024)

        assert cache.base_cache == base
        assert cache.compression_threshold == 1024

    def test_compressed_put_small_value(self):
        """Test putting small value (no compression)"""
        base = LRUCache(max_size=10)
        cache = CompressedCache(base, compression_threshold=1024)

        cache.put("key1", "small value")
        result = cache.get("key1")

        assert result == "small value"

    def test_compressed_put_large_value(self):
        """Test putting large value (with compression)"""
        base = LRUCache(max_size=10, max_bytes=10 * 1024 * 1024)
        cache = CompressedCache(base, compression_threshold=100)

        large_value = "x" * 10000  # Large compressible string
        cache.put("key1", large_value)
        result = cache.get("key1")

        assert result == large_value

    def test_compressed_numpy_array(self):
        """Test compressing numpy arrays"""
        base = LRUCache(max_size=10, max_bytes=10 * 1024 * 1024)
        cache = CompressedCache(base, compression_threshold=100)

        arr = np.random.randn(1000)
        cache.put("array", arr)
        result = cache.get("array")

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, arr)


class TestMultiLevelCache:
    """Test multi-level cache implementation"""

    def test_multilevel_cache_creation(self):
        """Test creating multi-level cache"""
        cache = MultiLevelCache()

        assert cache.l1 is not None
        assert cache.l2 is not None
        assert not cache.l3_enabled

    def test_multilevel_with_disk_cache(self):
        """Test multi-level cache with disk enabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"enable_disk_cache": True, "disk_cache_path": tmpdir}
            cache = MultiLevelCache(config)

            assert cache.l3_enabled
            assert cache.l3_path.exists()

    def test_multilevel_l1_hit(self):
        """Test L1 cache hit"""
        cache = MultiLevelCache()

        # Put small value (goes to L1)
        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"
        assert cache.stats.hits == 1
        assert cache.level_stats["l1"].hits == 1

    def test_multilevel_l2_hit(self):
        """Test L2 cache hit"""
        cache = MultiLevelCache()

        # Put large value (goes to L2)
        large_value = "x" * 2000
        cache.put("key1", large_value)

        # Clear L1 to force L2 hit
        cache.l1.clear()

        result = cache.get("key1")

        assert result == large_value
        assert cache.level_stats["l2"].hits >= 1

    def test_multilevel_promotion_to_l1(self):
        """Test promotion from L2 to L1"""
        cache = MultiLevelCache({"promotion_threshold": 2})

        # Put value in L2
        cache.l2.put("key1", "value1")

        # Access multiple times to trigger promotion
        for _ in range(3):
            cache.get("key1")

        # Should now be in L1
        assert cache.l1.get("key1") == "value1"

    def test_multilevel_cache_miss(self):
        """Test cache miss across all levels"""
        cache = MultiLevelCache()

        result = cache.get("missing_key")

        assert result is None
        assert cache.stats.misses == 1

    def test_multilevel_invalidate(self):
        """Test invalidation across all levels"""
        cache = MultiLevelCache()

        cache.put("key1", "value1")
        cache.invalidate("key1")

        assert cache.get("key1") is None

    def test_multilevel_disk_cache_operations(self):
        """Test disk cache operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "enable_disk_cache": True,
                "disk_cache_path": tmpdir,
                "l3_max_size_mb": 10,
            }
            cache = MultiLevelCache(config)

            # Put to disk
            cache._put_to_disk("disk_key", "disk_value")

            # Clear memory caches
            cache.l1.clear()
            cache.l2.base_cache.clear()

            # Should retrieve from disk
            result = cache.get("disk_key")

            assert result == "disk_value"

    def test_multilevel_disk_eviction(self):
        """Test disk cache eviction when size limit reached"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "enable_disk_cache": True,
                "disk_cache_path": tmpdir,
                "l3_max_size_mb": 1,  # Small limit
            }
            cache = MultiLevelCache(config)

            # Put multiple large items
            for i in range(10):
                large_value = "x" * 200000  # ~200KB each
                cache._put_to_disk(f"key_{i}", large_value)
                time.sleep(0.01)  # Ensure different timestamps

            # Check that eviction occurred
            assert cache.l3_current_size_mb <= cache.l3_max_size_mb

    def test_multilevel_get_statistics(self):
        """Test getting comprehensive statistics"""
        cache = MultiLevelCache()

        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("missing")

        stats = cache.get_statistics()

        assert "overall" in stats
        assert "l1" in stats
        assert "l2" in stats
        assert "l3" in stats
        assert stats["overall"]["hits"] > 0
        assert stats["overall"]["misses"] > 0


class TestSelectionCache:
    """Test main SelectionCache class"""

    def test_selection_cache_creation(self):
        """Test creating selection cache"""
        cache = SelectionCache()

        assert cache.feature_cache is not None
        assert cache.selection_cache is not None
        assert cache.result_cache is not None
        assert cache.similarity_cache is not None

    def test_cache_and_get_features(self):
        """Test caching and retrieving features"""
        cache = SelectionCache()

        problem = "test problem"
        features = np.random.randn(128)

        cache.cache_features(problem, features)
        result = cache.get_cached_features(problem)

        assert result is not None
        assert np.allclose(result, features)

    def test_cache_and_get_selection(self):
        """Test caching and retrieving selection decision"""
        cache = SelectionCache()

        features = np.random.randn(128)
        constraints = {"time_budget_ms": 100, "energy_budget_mj": 50}

        cache.cache_selection(features, constraints, "symbolic", 0.95)
        result = cache.get_cached_selection(features, constraints)

        assert result is not None
        assert result["tool"] == "symbolic"
        assert result["confidence"] == 0.95

    def test_cache_and_get_result(self):
        """Test caching and retrieving execution result"""
        cache = SelectionCache()

        problem = "test problem"
        result_data = {"answer": 42}

        cache.cache_result(
            "symbolic", problem, result_data, execution_time=10.5, energy=25.3
        )
        result = cache.get_cached_result("symbolic", problem)

        assert result is not None
        assert result["result"] == result_data
        assert result["execution_time"] == 10.5
        assert result["energy"] == 25.3

    def test_cache_and_get_similarity(self):
        """Test caching and retrieving similarity"""
        cache = SelectionCache()

        features1 = np.random.randn(128)
        features2 = np.random.randn(128)

        cache.cache_similarity(features1, features2, 0.85)
        result = cache.get_cached_similarity(features1, features2)

        assert result == 0.85

    def test_similarity_key_symmetry(self):
        """Test similarity cache key is symmetric"""
        cache = SelectionCache()

        features1 = np.random.randn(128)
        features2 = np.random.randn(128)

        cache.cache_similarity(features1, features2, 0.85)

        # Should work in reverse order
        result = cache.get_cached_similarity(features2, features1)
        assert result == 0.85

    def test_invalidate_problem(self):
        """Test invalidating problem-related caches"""
        cache = SelectionCache()

        problem = "test problem"
        features = np.random.randn(128)
        result_data = {"answer": 42}

        cache.cache_features(problem, features)
        cache.cache_result(
            "symbolic", problem, result_data, execution_time=10.5, energy=25.3
        )

        cache.invalidate_problem(problem)

        assert cache.get_cached_features(problem) is None
        assert cache.get_cached_result("symbolic", problem) is None

    def test_precompute_common_patterns(self):
        """Test precomputing common patterns"""
        cache = SelectionCache()

        patterns = ["pattern1", "pattern2", "pattern3"]

        with patch.object(
            cache, "_extract_features_for_pattern", return_value=np.random.randn(128)
        ):
            cache.precompute_common_patterns(patterns)

            assert len(cache.precomputed) > 0

    def test_warm_cache(self):
        """Test cache warming"""
        cache = SelectionCache({"enable_warming": True})

        # Add entries
        for i in range(5):
            features = np.random.randn(128)
            constraints = {"time_budget_ms": 100 * i}
            cache.cache_selection(features, constraints, "symbolic", 0.9)

        # Clear selection cache
        cache.selection_cache.l1.clear()
        cache.selection_cache.l2.base_cache.clear()

        # Warm it back up
        cache.warm_cache()

        # Entries should be back (or at least attempted)
        assert len(cache.warm_entries) > 0

    def test_get_statistics(self):
        """Test getting comprehensive statistics"""
        cache = SelectionCache()

        # Generate some activity
        problem = "test"
        features = np.random.randn(128)
        cache.cache_features(problem, features)
        cache.get_cached_features(problem)

        stats = cache.get_statistics()

        assert "feature_cache" in stats
        assert "selection_cache" in stats
        assert "result_cache" in stats
        assert "similarity_cache" in stats

    def test_save_cache(self):
        """Test saving cache to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SelectionCache()

            # Add some data
            features = np.random.randn(128)
            constraints = {"time_budget_ms": 100}
            cache.cache_selection(features, constraints, "symbolic", 0.9)

            # Save
            cache.save_cache(tmpdir)

            # Check files exist
            save_path = Path(tmpdir)
            assert (save_path / "warm_entries.pkl").exists()
            assert (save_path / "statistics.json").exists()

    def test_shutdown(self):
        """Test cache shutdown"""
        cache = SelectionCache()

        # Add some data
        cache.cache_features("test", np.random.randn(128))

        # Shutdown
        cache.shutdown()

        # Caches should be cleared
        assert len(cache.feature_cache.l1.cache) == 0

    def test_thread_safety(self):
        """Test thread safety of SelectionCache"""
        cache = SelectionCache()
        errors = []

        def worker(worker_id):
            try:
                for i in range(50):
                    problem = f"problem_{worker_id}_{i}"
                    features = np.random.randn(128)
                    constraints = {"time_budget_ms": 100}

                    cache.cache_features(problem, features)
                    cache.get_cached_features(problem)

                    cache.cache_selection(features, constraints, "symbolic", 0.9)
                    cache.get_cached_selection(features, constraints)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCacheIntegration:
    """Integration tests for complete cache system"""

    def test_full_workflow(self):
        """Test complete caching workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "enable_disk_cache": True,
                "disk_cache_path": tmpdir,
                "enable_warming": True,
            }
            cache = SelectionCache(config)

            # Step 1: Cache features
            problem = "complex reasoning problem"
            features = np.random.randn(128)
            cache.cache_features(problem, features)

            # Step 2: Cache selection
            constraints = {"time_budget_ms": 1000, "energy_budget_mj": 100}
            cache.cache_selection(features, constraints, "symbolic", 0.95)

            # Step 3: Cache result
            result = {"solution": "42", "steps": ["step1", "step2"]}
            cache.cache_result(
                "symbolic", problem, result, execution_time=50.5, energy=45.2
            )

            # Step 4: Retrieve all
            cached_features = cache.get_cached_features(problem)
            cached_selection = cache.get_cached_selection(features, constraints)
            cached_result = cache.get_cached_result("symbolic", problem)

            assert np.allclose(cached_features, features)
            assert cached_selection["tool"] == "symbolic"
            assert cached_result["result"] == result

            # Step 5: Get statistics
            stats = cache.get_statistics()
            assert stats["feature_cache"]["overall"]["hits"] > 0

            # Step 6: Save and shutdown
            cache.save_cache(tmpdir)
            cache.shutdown()

    def test_cache_performance_under_load(self):
        """Test cache performance under load"""
        cache = SelectionCache()

        start_time = time.time()

        # Generate load
        for i in range(1000):
            problem = f"problem_{i}"
            features = np.random.randn(128)
            constraints = {"time_budget_ms": 100 + i}

            cache.cache_features(problem, features)
            cache.cache_selection(features, constraints, "symbolic", 0.9)

            # Some retrievals
            if i % 2 == 0:
                cache.get_cached_features(problem)
                cache.get_cached_selection(features, constraints)

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

        # Check hit rate
        stats = cache.get_statistics()
        hit_rate = stats["feature_cache"]["overall"]["hit_rate"]
        assert hit_rate > 0  # Should have some hits

    def test_memory_management(self):
        """Test memory management and eviction"""
        # FIX: The config must be nested correctly for SelectionCache to parse it.
        config = {
            "feature_cache_config": {
                "l1_size": 10,
                "l1_bytes": 1024 * 100,  # 100KB
                "l2_size": 50,
                "l2_bytes": 1024 * 500,  # 500KB
            }
        }
        cache = SelectionCache(config)

        # Fill cache beyond limits
        for i in range(100):
            problem = f"problem_{i}"
            features = np.random.randn(1000)  # Large features
            cache.cache_features(problem, features)

        # Check that caches respect limits
        l1_stats = cache.feature_cache.l1.get_stats()
        assert l1_stats["entries"] <= 10

        l2_stats = cache.feature_cache.l2.base_cache.get_stats()
        assert l2_stats["entries"] <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
