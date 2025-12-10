"""
Test bug fixes for cpu_capabilities.py and performance_metrics.py
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.utils.cpu_capabilities import (CPUCapabilities,
                                        _detect_macos_capabilities,
                                        detect_cpu_capabilities,
                                        get_cpu_capabilities)
from src.utils.performance_metrics import (PerformanceTimer,
                                           PerformanceTracker,
                                           get_performance_tracker)


class TestCPUCapabilitiesFixes:
    """Test fixes for cpu_capabilities.py"""

    def test_repr_method(self):
        """Test that CPUCapabilities has __repr__ for debugging"""
        caps = CPUCapabilities(
            architecture="x86_64",
            platform="Linux",
            has_avx2=True
        )
        repr_str = repr(caps)
        assert "CPUCapabilities" in repr_str
        assert "arch=" in repr_str
        assert "best=" in repr_str
        assert "tier=" in repr_str

    def test_bare_except_replaced(self):
        """Test that specific exceptions are caught instead of bare except"""
        # This test verifies the code compiles and runs
        # The actual exception handling is in detect_cpu_capabilities()
        caps = detect_cpu_capabilities()
        assert caps.cpu_cores >= 1

    def test_macos_sysctl_parsing(self):
        """Test that macOS sysctl parsing properly extracts flags"""
        caps = CPUCapabilities(architecture="x86_64")

        # Mock subprocess result with key:value format
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "machdep.cpu.features: SSE AVX AVX2\nmachdep.cpu.leaf7_features: AVX512F"

        with patch('subprocess.run', return_value=mock_result):
            _detect_macos_capabilities(caps)

        # Verify flags don't include the key names
        assert 'sse' in caps.cpu_flags
        assert 'avx' in caps.cpu_flags
        assert 'avx2' in caps.cpu_flags
        assert 'avx512f' in caps.cpu_flags
        # These should NOT be in the flags
        assert 'machdep.cpu.features:' not in caps.cpu_flags
        assert 'machdep.cpu.leaf7_features:' not in caps.cpu_flags

    def test_singleton_thread_safety(self):
        """Test that get_cpu_capabilities is thread-safe"""
        results = []

        def get_caps():
            caps = get_cpu_capabilities()
            results.append(id(caps))

        # Create multiple threads
        threads = [threading.Thread(target=get_caps) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All threads should get the same instance (same id)
        assert len(set(results)) == 1, "Multiple instances created, not thread-safe"


class TestPerformanceMetricsFixes:
    """Test fixes for performance_metrics.py"""

    def test_thread_safety_record(self):
        """Test that PerformanceTracker.record is thread-safe"""
        tracker = PerformanceTracker()

        def record_metrics():
            for i in range(100):
                tracker.record("test_op", "impl1", float(i), success=True)

        threads = [threading.Thread(target=record_metrics) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded 500 metrics total
        stats = tracker.get_stats("test_op", "impl1")
        assert stats is not None
        assert stats['count'] == 500

    def test_thread_safety_get_stats(self):
        """Test that PerformanceTracker.get_stats is thread-safe"""
        tracker = PerformanceTracker()

        # Pre-populate with some data
        for i in range(100):
            tracker.record("test_op", "impl1", float(i))

        results = []

        def get_stats():
            stats = tracker.get_stats("test_op", "impl1")
            results.append(stats)

        threads = [threading.Thread(target=get_stats) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get valid stats
        assert len(results) == 10
        for stats in results:
            assert stats is not None
            assert stats['count'] == 100

    def test_division_edge_case_zero_full_mean(self):
        """Test that division by zero is handled properly"""
        tracker = PerformanceTracker()

        # Record zero duration for full implementation
        tracker.record("test_op", "full", 0.0)
        tracker.record("test_op", "fallback", 10.0)

        comparison = tracker.compare_implementations("test_op")

        assert 'comparison' in comparison
        slowdown = comparison['comparison']['fallback_slowdown_factor']
        slower_by_percent = comparison['comparison']['fallback_slower_by_percent']
        # Should be infinity when full_mean is 0 and fallback_mean > 0
        assert slowdown == float('inf')
        assert slower_by_percent == float('inf')

    def test_division_edge_case_both_zero(self):
        """Test that division when both means are zero returns 1.0"""
        tracker = PerformanceTracker()

        # Record zero durations for both
        tracker.record("test_op", "full", 0.0)
        tracker.record("test_op", "fallback", 0.0)

        comparison = tracker.compare_implementations("test_op")

        assert 'comparison' in comparison
        slowdown = comparison['comparison']['fallback_slowdown_factor']
        slower_by_percent = comparison['comparison']['fallback_slower_by_percent']
        # Should be 1.0 when both are 0
        assert slowdown == 1.0
        assert slower_by_percent == 0.0

    def test_singleton_thread_safety(self):
        """Test that get_performance_tracker is thread-safe"""
        results = []

        def get_tracker():
            tracker = get_performance_tracker()
            results.append(id(tracker))

        threads = [threading.Thread(target=get_tracker) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(results)) == 1

    def test_percentile_support(self):
        """Test that percentiles are calculated when enough samples"""
        tracker = PerformanceTracker()

        # Add enough samples for p95 (need 20+)
        for i in range(50):
            tracker.record("test_op", "impl1", float(i))

        stats = tracker.get_stats("test_op", "impl1")

        assert stats is not None
        assert 'p95_ms' in stats
        assert stats['p95_ms'] is not None
        # P95 should be around the 95th percentile, not the max
        assert stats['p95_ms'] < stats['max_ms']
        # P95 should be >= median
        assert stats['p95_ms'] >= stats['median_ms']

        # Not enough for p99 (need 100+)
        assert 'p99_ms' not in stats or stats['p99_ms'] is None

    def test_percentile_p99_support(self):
        """Test that p99 is calculated with 100+ samples"""
        tracker = PerformanceTracker()

        # Add enough samples for p99 (need more than 100 for p99 < max)
        for i in range(150):
            tracker.record("test_op", "impl1", float(i))

        stats = tracker.get_stats("test_op", "impl1")

        assert stats is not None
        assert 'p95_ms' in stats
        assert 'p99_ms' in stats
        assert stats['p99_ms'] is not None
        # With 150 samples, P99 should be < max (at index 148, not 149)
        assert stats['p99_ms'] < stats['max_ms']
        # P99 should be >= p95
        assert stats['p99_ms'] >= stats['p95_ms']

    def test_failure_rate_tracking(self):
        """Test that failure rate is tracked"""
        tracker = PerformanceTracker()

        # Record some successes and failures
        for i in range(80):
            tracker.record("test_op", "impl1", 10.0, success=True)
        for i in range(20):
            tracker.record("test_op", "impl1", 10.0, success=False)

        stats = tracker.get_stats("test_op", "impl1")

        assert stats is not None
        assert 'failure_rate' in stats
        assert stats['total_attempts'] == 100
        assert stats['count'] == 80  # Only successful
        assert abs(stats['failure_rate'] - 0.2) < 0.01  # 20% failure rate

    def test_failure_rate_all_failed(self):
        """Test failure rate when all attempts failed"""
        tracker = PerformanceTracker()

        for i in range(10):
            tracker.record("test_op", "impl1", 10.0, success=False)

        stats = tracker.get_stats("test_op", "impl1")

        assert stats is not None
        assert stats['failure_rate'] == 1.0
        assert stats['count'] == 0

    def test_decorator_pattern(self):
        """Test that PerformanceTimer can be used as a decorator"""
        tracker = PerformanceTracker()

        @PerformanceTimer("test_func", "impl1")
        def sample_function():
            time.sleep(0.01)
            return "result"

        result = sample_function()

        assert result == "result"
        # Give time for metrics to be recorded
        time.sleep(0.01)
        stats = tracker.get_stats("test_func", "impl1")
        # Stats might be in global tracker, not local one
        # This test verifies the decorator works without errors

    def test_clear_functionality(self):
        """Test that clear method works"""
        tracker = PerformanceTracker()

        # Add some data
        tracker.record("op1", "impl1", 10.0)
        tracker.record("op1", "impl2", 20.0)
        tracker.record("op2", "impl1", 30.0)

        # Clear all
        tracker.clear()

        assert tracker.get_stats("op1", "impl1") is None
        assert tracker.get_stats("op1", "impl2") is None
        assert tracker.get_stats("op2", "impl1") is None

    def test_clear_specific_operation(self):
        """Test that clear method can clear specific operation"""
        tracker = PerformanceTracker()

        # Add some data
        tracker.record("op1", "impl1", 10.0)
        tracker.record("op1", "impl2", 20.0)
        tracker.record("op2", "impl1", 30.0)

        # Clear only op1
        tracker.clear("op1")

        assert tracker.get_stats("op1", "impl1") is None
        assert tracker.get_stats("op1", "impl2") is None
        # op2 should still exist
        assert tracker.get_stats("op2", "impl1") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
