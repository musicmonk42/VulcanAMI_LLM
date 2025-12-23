"""
Tests for performance instrumentation utilities.

This module contains comprehensive tests for the performance instrumentation
utilities including metrics collection, timing decorators, and the global
performance tracker.
"""

import asyncio
import time

import pytest

from src.utils.performance_instrumentation import (
    GenerationPerformanceMetrics,
    TimingContext,
    get_generation_performance_tracker,
    timed,
    timed_async,
)


class TestGenerationPerformanceMetrics:
    """Tests for GenerationPerformanceMetrics class."""

    def test_init_defaults(self):
        """Test default initialization values."""
        metrics = GenerationPerformanceMetrics()
        assert metrics.total_encode_time_ms == 0.0
        assert metrics.tokens_generated == 0
        assert metrics.encoding_cache_hits == 0

    def test_record_encode_time(self):
        """Test recording encode times accumulates correctly."""
        metrics = GenerationPerformanceMetrics()
        metrics.record_encode_time(100.0)
        metrics.record_encode_time(50.0)

        assert metrics.total_encode_time_ms == 150.0
        assert len(metrics.encode_times) == 2

    def test_record_encode_time_invalid(self):
        """Test that invalid encode times are rejected."""
        metrics = GenerationPerformanceMetrics()
        metrics.record_encode_time(-10.0)  # Negative
        metrics.record_encode_time("invalid")  # Wrong type

        assert metrics.total_encode_time_ms == 0.0
        assert len(metrics.encode_times) == 0

    def test_record_logits_time(self):
        """Test recording logits computation times."""
        metrics = GenerationPerformanceMetrics()
        metrics.record_logits_time(25.0)

        assert metrics.total_logits_time_ms == 25.0
        assert len(metrics.logits_times) == 1

    def test_record_sample_time(self):
        """Test recording sampling times."""
        metrics = GenerationPerformanceMetrics()
        metrics.record_sample_time(10.0)

        assert metrics.total_sample_time_ms == 10.0
        assert len(metrics.sample_times) == 1

    def test_get_percentile(self):
        """Test percentile calculation with sample data."""
        metrics = GenerationPerformanceMetrics()

        # Add sample data (0-99)
        for i in range(100):
            metrics.encode_times.append(float(i))

        # Check p50 (median)
        p50 = metrics.get_percentile(metrics.encode_times, 50)
        assert 45 <= p50 <= 55

        # Check p95
        p95 = metrics.get_percentile(metrics.encode_times, 95)
        assert 90 <= p95 <= 99

    def test_get_percentile_empty(self):
        """Test percentile with empty data returns zero."""
        metrics = GenerationPerformanceMetrics()
        assert metrics.get_percentile(metrics.encode_times, 50) == 0.0

    def test_get_percentile_invalid(self):
        """Test percentile with invalid values gets clamped."""
        metrics = GenerationPerformanceMetrics()
        metrics.encode_times.append(50.0)

        # Should handle out-of-range percentiles gracefully
        result = metrics.get_percentile(metrics.encode_times, 150)
        assert result >= 0.0

    def test_get_summary(self):
        """Test summary generation with populated metrics."""
        metrics = GenerationPerformanceMetrics()
        metrics.tokens_generated = 10
        metrics.total_encode_time_ms = 100.0
        metrics.encoding_cache_hits = 5
        metrics.encoding_cache_misses = 5

        summary = metrics.get_summary()

        assert summary["tokens_generated"] == 10
        assert summary["timing"]["total_encode_ms"] == 100.0
        assert summary["timing"]["avg_encode_ms"] == 10.0
        assert summary["cache"]["encoding_hit_rate"] == 0.5

    def test_get_summary_no_tokens(self):
        """Test summary with zero tokens generated doesn't divide by zero."""
        metrics = GenerationPerformanceMetrics()
        summary = metrics.get_summary()

        # Should use max(1, 0) = 1 for division
        assert summary["timing"]["avg_encode_ms"] == 0.0

    def test_reset(self):
        """Test reset clears all metrics."""
        metrics = GenerationPerformanceMetrics()
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

    def test_timing_threshold(self):
        """Test custom threshold is stored."""
        ctx = TimingContext("test", threshold_ms=50.0, log_always=True)
        assert ctx.threshold_ms == 50.0
        assert ctx.log_always is True

    def test_timing_with_exception(self):
        """Test timing records time even when exception occurs."""
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
        """Test that decorated function return value is preserved."""
        @timed("test_return")
        def compute():
            return 42

        assert compute() == 42

    def test_timed_preserves_exception(self):
        """Test that exceptions are re-raised from decorated function."""
        @timed("test_exception")
        def failing():
            raise RuntimeError("expected")

        with pytest.raises(RuntimeError, match="expected"):
            failing()

    def test_timed_custom_threshold(self):
        """Test custom threshold parameter."""
        @timed("test_threshold", threshold_ms=1.0)
        def fast_function():
            return "fast"

        result = fast_function()
        assert result == "fast"


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
        """Test that async return value is preserved."""
        @timed_async("test_async_return")
        async def compute_async():
            return 42

        assert await compute_async() == 42

    @pytest.mark.asyncio
    async def test_timed_async_preserves_exception(self):
        """Test that async exceptions are re-raised."""
        @timed_async("test_async_exception")
        async def failing_async():
            raise RuntimeError("expected async")

        with pytest.raises(RuntimeError, match="expected async"):
            await failing_async()


class TestGenerationPerformanceTracker:
    """Tests for GenerationPerformanceTracker singleton."""

    def test_singleton_pattern(self):
        """Test that only one instance exists."""
        tracker1 = get_generation_performance_tracker()
        tracker2 = get_generation_performance_tracker()

        assert tracker1 is tracker2

    def test_record_and_get_stats(self):
        """Test recording operations and retrieving statistics."""
        tracker = get_generation_performance_tracker()
        tracker.reset()  # Start fresh

        tracker.record("test_op", 10.0)
        tracker.record("test_op", 20.0)
        tracker.record("test_op", 30.0)

        stats = tracker.get_stats("test_op")

        assert stats["count"] == 3
        assert stats["avg_ms"] == 20.0
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 30.0

    def test_record_invalid_time(self):
        """Test that invalid time values are rejected."""
        tracker = get_generation_performance_tracker()
        tracker.reset()

        tracker.record("invalid_test", -10.0)  # Negative
        tracker.record("invalid_test", "bad")  # Wrong type

        stats = tracker.get_stats("invalid_test")
        assert stats["count"] == 0

    def test_get_stats_unknown_operation(self):
        """Test getting stats for unknown operation returns zeros."""
        tracker = get_generation_performance_tracker()
        stats = tracker.get_stats("unknown_op_12345")

        assert stats["count"] == 0
        assert stats["avg_ms"] == 0.0
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 0.0

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        tracker = get_generation_performance_tracker()
        tracker.reset()

        tracker.disable()
        tracker.record("disabled_op", 100.0)

        stats = tracker.get_stats("disabled_op")
        assert stats["count"] == 0  # Should not be recorded

        tracker.enable()
        tracker.record("enabled_op", 100.0)

        stats = tracker.get_stats("enabled_op")
        assert stats["count"] == 1  # Should be recorded

    def test_is_enabled(self):
        """Test is_enabled returns correct state."""
        tracker = get_generation_performance_tracker()

        tracker.enable()
        assert tracker.is_enabled() is True

        tracker.disable()
        assert tracker.is_enabled() is False

        # Re-enable for other tests
        tracker.enable()

    def test_reset(self):
        """Test reset clears all tracked data."""
        tracker = get_generation_performance_tracker()
        tracker.record("reset_test", 100.0)

        tracker.reset()

        stats = tracker.get_stats("reset_test")
        assert stats["count"] == 0

    def test_get_all_stats(self):
        """Test getting all operation statistics."""
        tracker = get_generation_performance_tracker()
        tracker.reset()

        tracker.record("op1", 10.0)
        tracker.record("op2", 20.0)

        all_stats = tracker.get_all_stats()

        assert "op1" in all_stats
        assert "op2" in all_stats


class TestThreadSafety:
    """Tests for thread safety of metrics and tracker."""

    def test_metrics_thread_safety(self):
        """Test that metrics can be accessed from multiple threads."""
        import threading

        metrics = GenerationPerformanceMetrics()
        errors = []

        def record_times():
            try:
                for i in range(100):
                    metrics.record_encode_time(float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_times) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 500 entries (5 threads * 100 each), limited by maxlen
        assert len(metrics.encode_times) <= 1000

    def test_tracker_thread_safety(self):
        """Test that tracker can be accessed from multiple threads."""
        import threading

        tracker = get_generation_performance_tracker()
        tracker.reset()
        errors = []

        def record_ops():
            try:
                for i in range(100):
                    tracker.record("thread_test", float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_ops) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = tracker.get_stats("thread_test")
        assert stats["count"] > 0


class TestThrottledLogger:
    """Tests for ThrottledLogger class."""

    def test_throttled_logger_initialization(self):
        """Test ThrottledLogger initializes with correct defaults."""
        from src.utils.performance_instrumentation import ThrottledLogger

        logger = ThrottledLogger("test")
        assert logger.name == "test"
        assert logger.log_interval == 10
        assert logger.time_interval_ms == 1000.0
        assert logger._call_count == 0

    def test_throttled_logger_custom_interval(self):
        """Test ThrottledLogger with custom intervals."""
        from src.utils.performance_instrumentation import ThrottledLogger

        logger = ThrottledLogger("custom", log_interval=5, time_interval_ms=500.0)
        assert logger.log_interval == 5
        assert logger.time_interval_ms == 500.0

    def test_throttled_logger_logs_at_interval(self):
        """Test that logger respects interval count."""
        from src.utils.performance_instrumentation import ThrottledLogger

        # Use a very high time interval so only count-based logging triggers
        logger = ThrottledLogger("interval_test", log_interval=5, time_interval_ms=100000.0)

        # Call _should_log multiple times
        logged_count = 0
        for _ in range(20):
            if logger._should_log():
                logged_count += 1

        # Should log on 1st (time check triggers for first call with _last_log_time=0),
        # 5th, 10th, 15th, 20th call = 5 times
        assert logged_count == 5

    def test_throttled_logger_reset(self):
        """Test ThrottledLogger reset functionality."""
        from src.utils.performance_instrumentation import ThrottledLogger

        logger = ThrottledLogger("reset_test", log_interval=5)

        # Make some calls
        for _ in range(7):
            logger._should_log()

        assert logger._call_count == 7

        # Reset
        logger.reset()
        assert logger._call_count == 0
        assert logger._last_log_time == 0.0

    def test_get_step_logger(self):
        """Test get_step_logger returns global instance."""
        from src.utils.performance_instrumentation import get_step_logger

        logger1 = get_step_logger()
        logger2 = get_step_logger()

        assert logger1 is logger2
        assert logger1.name == "STEP"

    def test_get_token_logger(self):
        """Test get_token_logger returns global instance."""
        from src.utils.performance_instrumentation import get_token_logger

        logger1 = get_token_logger()
        logger2 = get_token_logger()

        assert logger1 is logger2
        assert logger1.name == "TOKEN"
