"""
Tests for performance instrumentation utilities.
"""

import asyncio
import time
import pytest
from unittest.mock import patch

from src.utils.performance_instrumentation import (
    PerformanceMetrics,
    PerformanceTracker,
    TimingContext,
    get_performance_tracker,
    timed,
    timed_async,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        metrics = PerformanceMetrics()
        assert metrics.total_encode_time_ms == 0.0
        assert metrics.tokens_generated == 0
        assert metrics.encoding_cache_hits == 0
    
    def test_record_encode_time(self):
        """Test recording encode times."""
        metrics = PerformanceMetrics()
        metrics.record_encode_time(100.0)
        metrics.record_encode_time(50.0)
        
        assert metrics.total_encode_time_ms == 150.0
        assert len(metrics.encode_times) == 2
    
    def test_record_logits_time(self):
        """Test recording logits times."""
        metrics = PerformanceMetrics()
        metrics.record_logits_time(25.0)
        
        assert metrics.total_logits_time_ms == 25.0
        assert len(metrics.logits_times) == 1
    
    def test_record_sample_time(self):
        """Test recording sample times."""
        metrics = PerformanceMetrics()
        metrics.record_sample_time(10.0)
        
        assert metrics.total_sample_time_ms == 10.0
        assert len(metrics.sample_times) == 1
    
    def test_get_percentile(self):
        """Test percentile calculation."""
        metrics = PerformanceMetrics()
        
        # Add sample data
        for i in range(100):
            metrics.encode_times.append(float(i))
        
        # Check percentiles
        p50 = metrics.get_percentile(metrics.encode_times, 50)
        assert 45 <= p50 <= 55
        
        p95 = metrics.get_percentile(metrics.encode_times, 95)
        assert 90 <= p95 <= 99
    
    def test_get_percentile_empty(self):
        """Test percentile with empty data."""
        metrics = PerformanceMetrics()
        assert metrics.get_percentile(metrics.encode_times, 50) == 0.0
    
    def test_get_summary(self):
        """Test summary generation."""
        metrics = PerformanceMetrics()
        metrics.tokens_generated = 10
        metrics.total_encode_time_ms = 100.0
        metrics.encoding_cache_hits = 5
        metrics.encoding_cache_misses = 5
        
        summary = metrics.get_summary()
        
        assert summary["tokens_generated"] == 10
        assert summary["timing"]["total_encode_ms"] == 100.0
        assert summary["timing"]["avg_encode_ms"] == 10.0
        assert summary["cache"]["encoding_hit_rate"] == 0.5
    
    def test_reset(self):
        """Test reset functionality."""
        metrics = PerformanceMetrics()
        metrics.tokens_generated = 10
        metrics.total_encode_time_ms = 100.0
        metrics.record_encode_time(50.0)
        
        metrics.reset()
        
        assert metrics.tokens_generated == 0
        assert metrics.total_encode_time_ms == 0.0
        assert len(metrics.encode_times) == 0


class TestTimingContext:
    """Tests for TimingContext context manager."""
    
    def test_timing_basic(self):
        """Test basic timing functionality."""
        with TimingContext("test_operation") as timer:
            time.sleep(0.01)  # 10ms
        
        assert timer.elapsed_ms >= 10.0
        assert timer.elapsed_ms < 100.0  # Sanity check
    
    def test_timing_name(self):
        """Test that name is stored correctly."""
        with TimingContext("my_operation") as timer:
            pass
        
        assert timer.name == "my_operation"
    
    def test_timing_with_exception(self):
        """Test timing when exception occurs."""
        with pytest.raises(ValueError):
            with TimingContext("failing_operation") as timer:
                raise ValueError("test error")
        
        # Timer should still have recorded time
        assert timer.elapsed_ms >= 0


class TestTimedDecorator:
    """Tests for @timed decorator."""
    
    def test_timed_sync_function(self):
        """Test timed decorator on sync function."""
        @timed("test_sync")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"
    
    def test_timed_preserves_return_value(self):
        """Test that return value is preserved."""
        @timed("test_return")
        def compute():
            return 42
        
        assert compute() == 42
    
    def test_timed_preserves_exception(self):
        """Test that exceptions are re-raised."""
        @timed("test_exception")
        def failing():
            raise RuntimeError("expected")
        
        with pytest.raises(RuntimeError, match="expected"):
            failing()


class TestTimedAsyncDecorator:
    """Tests for @timed_async decorator."""
    
    @pytest.mark.asyncio
    async def test_timed_async_function(self):
        """Test timed_async decorator on async function."""
        @timed_async("test_async")
        async def slow_async():
            await asyncio.sleep(0.01)
            return "async_done"
        
        result = await slow_async()
        assert result == "async_done"
    
    @pytest.mark.asyncio
    async def test_timed_async_preserves_return_value(self):
        """Test that return value is preserved for async."""
        @timed_async("test_async_return")
        async def compute_async():
            return 42
        
        assert await compute_async() == 42
    
    @pytest.mark.asyncio
    async def test_timed_async_preserves_exception(self):
        """Test that exceptions are re-raised for async."""
        @timed_async("test_async_exception")
        async def failing_async():
            raise RuntimeError("expected async")
        
        with pytest.raises(RuntimeError, match="expected async"):
            await failing_async()


class TestPerformanceTracker:
    """Tests for PerformanceTracker singleton."""
    
    def test_singleton_pattern(self):
        """Test that only one instance exists."""
        tracker1 = get_performance_tracker()
        tracker2 = get_performance_tracker()
        
        assert tracker1 is tracker2
    
    def test_record_and_get_stats(self):
        """Test recording and retrieving stats."""
        tracker = get_performance_tracker()
        tracker.reset()  # Start fresh
        
        tracker.record("test_op", 10.0)
        tracker.record("test_op", 20.0)
        tracker.record("test_op", 30.0)
        
        stats = tracker.get_stats("test_op")
        
        assert stats["count"] == 3
        assert stats["avg_ms"] == 20.0
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 30.0
    
    def test_get_stats_unknown_operation(self):
        """Test getting stats for unknown operation."""
        tracker = get_performance_tracker()
        stats = tracker.get_stats("unknown_op_12345")
        
        assert stats["count"] == 0
        assert stats["avg_ms"] == 0.0
    
    def test_enable_disable(self):
        """Test enable/disable functionality."""
        tracker = get_performance_tracker()
        tracker.reset()
        
        tracker.disable()
        tracker.record("disabled_op", 100.0)
        
        stats = tracker.get_stats("disabled_op")
        assert stats["count"] == 0  # Should not be recorded
        
        tracker.enable()
        tracker.record("enabled_op", 100.0)
        
        stats = tracker.get_stats("enabled_op")
        assert stats["count"] == 1  # Should be recorded
    
    def test_reset(self):
        """Test reset functionality."""
        tracker = get_performance_tracker()
        tracker.record("reset_test", 100.0)
        
        tracker.reset()
        
        stats = tracker.get_stats("reset_test")
        assert stats["count"] == 0
    
    def test_get_all_stats(self):
        """Test getting all operation stats."""
        tracker = get_performance_tracker()
        tracker.reset()
        
        tracker.record("op1", 10.0)
        tracker.record("op2", 20.0)
        
        all_stats = tracker.get_all_stats()
        
        assert "op1" in all_stats
        assert "op2" in all_stats


# Run with: pytest tests/test_performance_instrumentation.py -v
