"""
test_adaptive_thresholds.py - Comprehensive tests for adaptive threshold management
Part of the VULCAN-AGI system test suite

Tests cover:
- AdaptiveThresholds initialization and configuration
- Manual threshold adjustments
- Auto-calibration based on performance
- PerformanceTracker recording and analysis
- StrategyProfiler profiling and recommendations
- Thread safety and edge cases
"""

import pytest
import numpy as np
import time
import threading
from collections import Counter

# Import modules to test
from vulcan.problem_decomposer.adaptive_thresholds import (
    AdaptiveThresholds,
    PerformanceTracker,
    StrategyProfiler,
    ThresholdType,
    ThresholdConfig,
    PerformanceRecord,
    StrategyProfile,
    StrategyStatus,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def adaptive_thresholds():
    """Create AdaptiveThresholds instance"""
    return AdaptiveThresholds()


@pytest.fixture
def custom_thresholds():
    """Create AdaptiveThresholds with custom values"""
    return AdaptiveThresholds({"confidence": 0.8, "complexity": 5.0, "timeout": 120.0})


@pytest.fixture
def performance_tracker():
    """Create PerformanceTracker instance"""
    return PerformanceTracker(window_size=100)


@pytest.fixture
def strategy_profiler():
    """Create StrategyProfiler instance"""
    return StrategyProfiler()


# ============================================================
# ADAPTIVE THRESHOLDS TESTS
# ============================================================


class TestAdaptiveThresholdsInitialization:
    """Test AdaptiveThresholds initialization"""

    def test_default_initialization(self, adaptive_thresholds):
        """Test default threshold values"""
        assert adaptive_thresholds.get_current("confidence") == 0.7
        assert adaptive_thresholds.get_current("complexity") == 3.0
        assert adaptive_thresholds.get_current("performance") == 0.6
        assert adaptive_thresholds.get_current("timeout") == 60.0
        assert adaptive_thresholds.get_current("resource") == 0.8

    def test_custom_initialization(self, custom_thresholds):
        """Test custom threshold values"""
        assert custom_thresholds.get_current("confidence") == 0.8
        assert custom_thresholds.get_current("complexity") == 5.0
        assert custom_thresholds.get_current("timeout") == 120.0

    def test_get_all_thresholds(self, adaptive_thresholds):
        """Test getting all thresholds at once"""
        all_thresholds = adaptive_thresholds.get_current()

        assert isinstance(all_thresholds, dict)
        assert len(all_thresholds) == 5
        assert "confidence" in all_thresholds
        assert "complexity" in all_thresholds
        assert "timeout" in all_thresholds

    def test_invalid_threshold_type(self, adaptive_thresholds):
        """Test requesting invalid threshold type"""
        result = adaptive_thresholds.get_current("invalid_type")
        assert result is None


class TestThresholdAdjustments:
    """Test manual threshold adjustments"""

    def test_adjust_up_single_threshold(self, adaptive_thresholds):
        """Test adjusting single threshold upward"""
        initial = adaptive_thresholds.get_current("confidence")
        adaptive_thresholds.adjust_up(0.1, "confidence")
        new_value = adaptive_thresholds.get_current("confidence")

        assert new_value > initial
        assert new_value == pytest.approx(initial * 1.1, rel=1e-5)

    def test_adjust_down_single_threshold(self, adaptive_thresholds):
        """Test adjusting single threshold downward"""
        initial = adaptive_thresholds.get_current("confidence")
        adaptive_thresholds.adjust_down(0.1, "confidence")
        new_value = adaptive_thresholds.get_current("confidence")

        assert new_value < initial
        assert new_value == pytest.approx(initial * 0.9, rel=1e-5)

    def test_adjust_up_all_thresholds(self, adaptive_thresholds):
        """Test adjusting all thresholds upward"""
        initial_values = adaptive_thresholds.get_current()
        adaptive_thresholds.adjust_up(0.05)
        new_values = adaptive_thresholds.get_current()

        for key in initial_values:
            assert new_values[key] > initial_values[key]

    def test_adjust_down_all_thresholds(self, adaptive_thresholds):
        """Test adjusting all thresholds downward"""
        initial_values = adaptive_thresholds.get_current()
        adaptive_thresholds.adjust_down(0.05)
        new_values = adaptive_thresholds.get_current()

        for key in initial_values:
            assert new_values[key] < initial_values[key]

    def test_threshold_min_bound(self, adaptive_thresholds):
        """Test threshold respects minimum bound"""
        # Adjust down multiple times to hit minimum
        for _ in range(20):
            adaptive_thresholds.adjust_down(0.5, "confidence")

        final_value = adaptive_thresholds.get_current("confidence")
        assert final_value >= 0.1  # Min value for confidence

    def test_threshold_max_bound(self, adaptive_thresholds):
        """Test threshold respects maximum bound"""
        # Adjust up multiple times to hit maximum
        for _ in range(20):
            adaptive_thresholds.adjust_up(0.5, "confidence")

        final_value = adaptive_thresholds.get_current("confidence")
        assert final_value <= 0.99  # Max value for confidence

    def test_adjustment_history_tracking(self, adaptive_thresholds):
        """Test that adjustments are tracked in history"""
        initial_count = len(adaptive_thresholds.adjustment_history)

        adaptive_thresholds.adjust_up(0.1, "confidence")
        adaptive_thresholds.adjust_down(0.1, "timeout")

        assert len(adaptive_thresholds.adjustment_history) == initial_count + 2

        # Check history contains required fields
        last_adjustment = adaptive_thresholds.adjustment_history[-1]
        assert "type" in last_adjustment
        assert "threshold" in last_adjustment
        assert "old_value" in last_adjustment
        assert "new_value" in last_adjustment


class TestAutoCalibration:
    """Test automatic threshold calibration"""

    def test_auto_calibrate_low_success_rate(self, adaptive_thresholds):
        """Test calibration with low success rate"""
        initial_confidence = adaptive_thresholds.get_current("confidence")

        # Simulate low success rate (30%)
        performance = [
            {"success": i % 10 < 3, "execution_time": 10.0, "complexity": 2.0}
            for i in range(50)
        ]

        adaptive_thresholds.auto_calibrate(performance)

        new_confidence = adaptive_thresholds.get_current("confidence")
        assert new_confidence < initial_confidence  # Should decrease

    def test_auto_calibrate_high_success_rate(self, adaptive_thresholds):
        """Test calibration with high success rate"""
        initial_confidence = adaptive_thresholds.get_current("confidence")

        # Simulate high success rate (95%)
        performance = [
            {"success": i % 20 < 19, "execution_time": 10.0, "complexity": 2.0}
            for i in range(50)
        ]

        adaptive_thresholds.auto_calibrate(performance)

        new_confidence = adaptive_thresholds.get_current("confidence")
        assert new_confidence > initial_confidence  # Should increase

    def test_auto_calibrate_timeout_adjustment(self, adaptive_thresholds):
        """Test timeout adjustment based on execution times"""
        initial_timeout = adaptive_thresholds.get_current("timeout")

        # Simulate execution times close to timeout
        performance = [
            {"success": True, "execution_time": 55.0, "complexity": 2.0}
            for _ in range(50)
        ]

        adaptive_thresholds.auto_calibrate(performance)

        new_timeout = adaptive_thresholds.get_current("timeout")
        assert new_timeout > initial_timeout  # Should increase

    def test_auto_calibrate_complexity_adjustment(self, adaptive_thresholds):
        """Test complexity adjustment based on problem complexity"""
        initial_complexity = adaptive_thresholds.get_current("complexity")

        # Simulate high complexity problems
        performance = [
            {"success": True, "execution_time": 10.0, "complexity": 5.0}
            for _ in range(50)
        ]

        adaptive_thresholds.auto_calibrate(performance)

        new_complexity = adaptive_thresholds.get_current("complexity")
        assert new_complexity > initial_complexity  # Should increase

    def test_auto_calibrate_empty_performance(self, adaptive_thresholds):
        """Test auto-calibration with empty performance list"""
        initial_values = adaptive_thresholds.get_current()

        adaptive_thresholds.auto_calibrate([])

        new_values = adaptive_thresholds.get_current()
        assert initial_values == new_values  # Should not change

    def test_auto_calibration_counter(self, adaptive_thresholds):
        """Test that auto-calibrations are counted"""
        initial_count = adaptive_thresholds.auto_calibrations

        performance = [
            {"success": True, "execution_time": 10.0, "complexity": 2.0}
            for _ in range(30)
        ]

        adaptive_thresholds.auto_calibrate(performance)

        assert adaptive_thresholds.auto_calibrations == initial_count + 1


class TestUpdateFromOutcome:
    """Test updating thresholds from execution outcomes"""

    def test_update_adds_to_performance_window(self, adaptive_thresholds):
        """Test that updates are added to performance window"""
        initial_size = len(adaptive_thresholds.performance_window)

        adaptive_thresholds.update_from_outcome(2.5, True, 15.0)

        assert len(adaptive_thresholds.performance_window) == initial_size + 1

    def test_update_triggers_auto_calibration(self, adaptive_thresholds):
        """Test that sufficient updates trigger auto-calibration"""
        initial_calibrations = adaptive_thresholds.auto_calibrations

        # Add 20 updates to trigger calibration
        for i in range(20):
            adaptive_thresholds.update_from_outcome(2.0, i % 2 == 0, 10.0)

        assert adaptive_thresholds.auto_calibrations > initial_calibrations

    def test_performance_window_size_limit(self, adaptive_thresholds):
        """Test performance window respects size limit"""
        # Add more than max size (50)
        for i in range(100):
            adaptive_thresholds.update_from_outcome(2.0, True, 10.0)

        assert len(adaptive_thresholds.performance_window) <= 50


class TestThresholdStatistics:
    """Test threshold statistics"""

    def test_get_statistics(self, adaptive_thresholds):
        """Test getting threshold statistics"""
        # Make some adjustments
        adaptive_thresholds.adjust_up(0.1, "confidence")
        adaptive_thresholds.adjust_down(0.1, "timeout")

        stats = adaptive_thresholds.get_statistics()

        assert "current_thresholds" in stats
        assert "total_adjustments" in stats
        assert "auto_calibrations" in stats
        assert "recent_adjustments" in stats

        assert stats["total_adjustments"] >= 2

    def test_get_confidence_threshold(self, adaptive_thresholds):
        """Test convenience method for confidence threshold"""
        confidence = adaptive_thresholds.get_confidence_threshold()

        assert isinstance(confidence, float)
        assert 0.0 < confidence < 1.0


# ============================================================
# PERFORMANCE TRACKER TESTS
# ============================================================


class TestPerformanceTrackerInitialization:
    """Test PerformanceTracker initialization"""

    def test_default_initialization(self, performance_tracker):
        """Test default initialization"""
        assert performance_tracker.window_size == 100
        assert len(performance_tracker.records) == 0
        assert performance_tracker.total_attempts == 0
        assert performance_tracker.total_successes == 0
        assert performance_tracker.total_failures == 0

    def test_custom_window_size(self):
        """Test custom window size"""
        tracker = PerformanceTracker(window_size=50)
        assert tracker.window_size == 50


class TestRecordingAttempts:
    """Test recording decomposition attempts"""

    def test_record_attempt(self, performance_tracker):
        """Test recording attempt"""
        initial_count = performance_tracker.total_attempts

        attempt_id = performance_tracker.record_attempt("problem_123")

        assert performance_tracker.total_attempts == initial_count + 1
        assert "problem_123" in attempt_id
        assert isinstance(attempt_id, str)

    def test_record_success(self, performance_tracker):
        """Test recording successful decomposition"""
        initial_successes = performance_tracker.total_successes

        performance_tracker.record_success(
            "problem_123", "hierarchical_decomposition", execution_time=1.5
        )

        assert performance_tracker.total_successes == initial_successes + 1
        assert len(performance_tracker.records) == 1

        record = performance_tracker.records[0]
        assert record.success is True
        assert record.strategy_used == "hierarchical_decomposition"

    def test_record_failure(self, performance_tracker):
        """Test recording failed decomposition"""
        initial_failures = performance_tracker.total_failures

        performance_tracker.record_failure(
            "problem_123", "timeout", strategy_used="brute_force_search"
        )

        assert performance_tracker.total_failures == initial_failures + 1
        assert len(performance_tracker.records) == 1

        record = performance_tracker.records[0]
        assert record.success is False
        assert record.failure_reason == "timeout"

    def test_strategy_tracking(self, performance_tracker):
        """Test strategy success/failure tracking"""
        performance_tracker.record_success("prob1", "strategy_A", 1.0)
        performance_tracker.record_success("prob2", "strategy_A", 1.5)
        performance_tracker.record_failure("prob3", "failed", "strategy_A")

        assert performance_tracker.strategy_successes["strategy_A"] == 2
        assert performance_tracker.strategy_failures["strategy_A"] == 1

    def test_failure_reason_tracking(self, performance_tracker):
        """Test failure reason counting"""
        performance_tracker.record_failure("prob1", "timeout")
        performance_tracker.record_failure("prob2", "timeout")
        performance_tracker.record_failure("prob3", "invalid_input")

        assert performance_tracker.failure_reasons["timeout"] == 2
        assert performance_tracker.failure_reasons["invalid_input"] == 1

    def test_records_window_limit(self, performance_tracker):
        """Test that records respect window size"""
        # Add more than window size
        for i in range(150):
            performance_tracker.record_success(f"prob_{i}", "strategy", 1.0)

        assert len(performance_tracker.records) == 100  # Window size


class TestSuccessRateCalculation:
    """Test success rate calculations"""

    def test_success_rate_all_success(self, performance_tracker):
        """Test success rate with all successes"""
        for i in range(10):
            performance_tracker.record_success(f"prob_{i}", "strategy", 1.0)

        success_rate = performance_tracker.get_success_rate()
        assert success_rate == 1.0

    def test_success_rate_all_failures(self, performance_tracker):
        """Test success rate with all failures"""
        for i in range(10):
            performance_tracker.record_failure(f"prob_{i}", "failed")

        success_rate = performance_tracker.get_success_rate()
        assert success_rate == 0.0

    def test_success_rate_mixed(self, performance_tracker):
        """Test success rate with mixed results"""
        for i in range(10):
            if i % 2 == 0:
                performance_tracker.record_success(f"prob_{i}", "strategy", 1.0)
            else:
                performance_tracker.record_failure(f"prob_{i}", "failed")

        success_rate = performance_tracker.get_success_rate()
        assert success_rate == 0.5

    def test_success_rate_custom_window(self, performance_tracker):
        """Test success rate with custom window"""
        # Add 100 successes, then 10 failures
        for i in range(100):
            performance_tracker.record_success(f"prob_{i}", "strategy", 1.0)

        for i in range(10):
            performance_tracker.record_failure(f"prob_fail_{i}", "failed")

        # Check last 10 (should all be failures)
        recent_rate = performance_tracker.get_success_rate(window=10)
        assert recent_rate == 0.0

        # Check last 50 (should be 40 successes, 10 failures)
        window_rate = performance_tracker.get_success_rate(window=50)
        assert window_rate == 0.8

    def test_success_rate_empty_records(self, performance_tracker):
        """Test success rate with no records"""
        success_rate = performance_tracker.get_success_rate()
        assert success_rate == 0.5  # Default


class TestStrategyPerformance:
    """Test strategy-specific performance tracking"""

    def test_get_strategy_performance(self, performance_tracker):
        """Test getting performance for specific strategy"""
        performance_tracker.record_success("prob1", "strategy_A", 1.0)
        performance_tracker.record_success("prob2", "strategy_A", 1.5)
        performance_tracker.record_failure("prob3", "failed", "strategy_A")

        perf = performance_tracker.get_strategy_performance("strategy_A")

        assert perf["success_rate"] == pytest.approx(2 / 3, rel=1e-5)
        assert perf["total_attempts"] == 3
        assert perf["successes"] == 2
        assert perf["failures"] == 1

    def test_get_strategy_performance_no_data(self, performance_tracker):
        """Test getting performance for strategy with no data"""
        perf = performance_tracker.get_strategy_performance("unknown_strategy")

        assert perf["success_rate"] == 0.5  # Default
        assert perf["total_attempts"] == 0


class TestProblemHistory:
    """Test problem-specific history tracking"""

    def test_get_problem_history(self, performance_tracker):
        """Test getting history for specific problem"""
        performance_tracker.record_success("prob1", "strategy_A", 1.0)
        performance_tracker.record_success("prob1", "strategy_B", 1.5)
        performance_tracker.record_failure("prob1", "failed", "strategy_C")

        history = performance_tracker.get_problem_history("prob1")

        assert len(history) == 3
        assert history[0].success is True
        assert history[1].success is True
        assert history[2].success is False

    def test_problem_history_size_limit(self, performance_tracker):
        """Test problem history respects size limit"""
        # Add more than limit (50)
        for i in range(100):
            performance_tracker.record_success("prob1", "strategy", 1.0)

        history = performance_tracker.get_problem_history("prob1")
        assert len(history) <= 50


class TestFailureAnalysis:
    """Test failure analysis"""

    def test_get_failure_analysis(self, performance_tracker):
        """Test getting failure analysis"""
        performance_tracker.record_failure("prob1", "timeout")
        performance_tracker.record_failure("prob2", "timeout")
        performance_tracker.record_failure("prob3", "invalid_input")
        performance_tracker.record_failure("prob4", "timeout")

        analysis = performance_tracker.get_failure_analysis()

        assert analysis["total_failures"] == 4
        assert analysis["reasons"]["timeout"] == 3
        assert analysis["reasons"]["invalid_input"] == 1
        assert analysis["percentages"]["timeout"] == 0.75
        assert analysis["top_reason"] == "timeout"

    def test_failure_analysis_empty(self, performance_tracker):
        """Test failure analysis with no failures"""
        analysis = performance_tracker.get_failure_analysis()

        assert analysis["total_failures"] == 0
        assert analysis["reasons"] == {}


class TestPerformanceStatistics:
    """Test overall performance statistics"""

    def test_get_statistics(self, performance_tracker):
        """Test getting overall statistics"""
        # Add some data
        for i in range(10):
            performance_tracker.record_success(f"prob_{i}", "strategy_A", 1.0)

        for i in range(5):
            performance_tracker.record_failure(f"prob_fail_{i}", "failed")

        stats = performance_tracker.get_statistics()

        assert stats["total_attempts"] == 15
        assert stats["total_successes"] == 10
        assert stats["total_failures"] == 5
        assert stats["overall_success_rate"] == pytest.approx(10 / 15, rel=1e-5)
        assert "unique_problems" in stats
        assert "strategies_used" in stats


# ============================================================
# STRATEGY PROFILER TESTS
# ============================================================


class TestStrategyProfilerInitialization:
    """Test StrategyProfiler initialization"""

    def test_default_initialization(self, strategy_profiler):
        """Test default initialization"""
        assert len(strategy_profiler.profiles) == 0
        assert strategy_profiler.total_updates == 0


class TestStrategyProfiling:
    """Test strategy profiling"""

    def test_update_creates_profile(self, strategy_profiler):
        """Test that update creates profile for new strategy"""
        strategy_profiler.update("strategy_A", latency=1.5, success=True)

        assert "strategy_A" in strategy_profiler.profiles

        profile = strategy_profiler.profiles["strategy_A"]
        assert profile.total_attempts == 1
        assert profile.successful_attempts == 1
        assert profile.avg_latency == 1.5

    def test_update_existing_profile(self, strategy_profiler):
        """Test updating existing profile"""
        strategy_profiler.update("strategy_A", latency=1.0, success=True)
        strategy_profiler.update("strategy_A", latency=2.0, success=True)
        strategy_profiler.update("strategy_A", latency=1.5, success=False)

        profile = strategy_profiler.profiles["strategy_A"]

        assert profile.total_attempts == 3
        assert profile.successful_attempts == 2
        assert profile.failed_attempts == 1
        assert profile.success_rate == pytest.approx(2 / 3, rel=1e-5)
        assert profile.avg_latency == pytest.approx(1.5, rel=1e-5)

    def test_class_specific_costs(self, strategy_profiler):
        """Test class-specific cost tracking"""
        strategy_profiler.update(
            "strategy_A", latency=1.0, success=True, problem_class="optimization"
        )
        strategy_profiler.update(
            "strategy_A", latency=2.0, success=True, problem_class="optimization"
        )

        cost = strategy_profiler.get_cost("strategy_A", "optimization")

        # Should be exponential moving average
        assert cost > 0


class TestCostEstimation:
    """Test cost estimation"""

    def test_get_cost_with_profile(self, strategy_profiler):
        """Test getting cost for profiled strategy"""
        strategy_profiler.update("strategy_A", latency=2.5, success=True)

        cost = strategy_profiler.get_cost("strategy_A")
        assert cost == 2.5

    def test_get_cost_with_class(self, strategy_profiler):
        """Test getting class-specific cost"""
        strategy_profiler.update(
            "strategy_A", latency=1.0, success=True, problem_class="class1"
        )
        strategy_profiler.update(
            "strategy_A", latency=3.0, success=True, problem_class="class1"
        )

        cost = strategy_profiler.get_cost("strategy_A", "class1")
        assert cost > 0  # EMA of 1.0 and 3.0

    def test_get_cost_unknown_strategy(self, strategy_profiler):
        """Test getting cost for unknown strategy"""
        cost = strategy_profiler.get_cost("unknown_strategy")
        assert cost == 10.0  # Default


class TestOptimalOrdering:
    """Test optimal strategy ordering"""

    def test_get_optimal_ordering(self, strategy_profiler):
        """Test getting optimal strategy ordering"""
        # Add strategies with different cost-effectiveness
        strategy_profiler.update("fast_good", latency=1.0, success=True)
        strategy_profiler.update("fast_good", latency=1.0, success=True)

        strategy_profiler.update("slow_good", latency=10.0, success=True)
        strategy_profiler.update("slow_good", latency=10.0, success=True)

        strategy_profiler.update("fast_bad", latency=1.0, success=False)
        strategy_profiler.update("fast_bad", latency=1.0, success=False)

        ordering = strategy_profiler.get_optimal_ordering()

        # fast_good should be first (high success, low cost)
        assert ordering[0] == "fast_good"
        # fast_bad should be last (low success)
        assert ordering[-1] == "fast_bad"

    def test_ordering_caching(self, strategy_profiler):
        """Test that ordering is cached"""
        strategy_profiler.update("strategy_A", latency=1.0, success=True)

        # Get ordering twice
        ordering1 = strategy_profiler.get_optimal_ordering()
        ordering2 = strategy_profiler.get_optimal_ordering()

        assert ordering1 == ordering2
        assert len(strategy_profiler.ordering_cache) > 0

    def test_ordering_cache_limit(self, strategy_profiler):
        """Test ordering cache size limit"""
        # Add many different problem classes to exceed cache
        for i in range(150):
            strategy_profiler.get_optimal_ordering(f"class_{i}")

        # Cache should have been cleared
        assert len(strategy_profiler.ordering_cache) <= 100


class TestStrategyRecommendation:
    """Test strategy recommendation"""

    def test_recommend_strategy(self, strategy_profiler):
        """Test recommending best strategy"""
        strategy_profiler.update("strategy_A", latency=1.0, success=True)
        strategy_profiler.update("strategy_A", latency=1.0, success=True)

        strategy_profiler.update("strategy_B", latency=2.0, success=True)

        recommendation = strategy_profiler.recommend_strategy()

        # Should recommend strategy_A (better cost-effectiveness)
        assert recommendation == "strategy_A"

    def test_recommend_with_latency_constraint(self, strategy_profiler):
        """Test recommendation with latency constraint"""
        strategy_profiler.update("fast", latency=1.0, success=True)
        strategy_profiler.update("slow", latency=10.0, success=True)

        # With tight constraint
        recommendation = strategy_profiler.recommend_strategy(max_latency=5.0)
        assert recommendation == "fast"

    def test_recommend_with_success_rate_threshold(self, strategy_profiler):
        """Test that low success strategies are excluded"""
        # Low success strategy
        strategy_profiler.update("bad_strategy", latency=1.0, success=False)
        strategy_profiler.update("bad_strategy", latency=1.0, success=False)
        strategy_profiler.update("bad_strategy", latency=1.0, success=True)

        # Good strategy
        strategy_profiler.update("good_strategy", latency=2.0, success=True)
        strategy_profiler.update("good_strategy", latency=2.0, success=True)

        recommendation = strategy_profiler.recommend_strategy()

        # Should recommend good_strategy (success_rate >= 0.3)
        assert recommendation == "good_strategy"


class TestProfilerStatistics:
    """Test profiler statistics"""

    def test_get_statistics(self, strategy_profiler):
        """Test getting profiler statistics"""
        strategy_profiler.update("strategy_A", latency=1.0, success=True)
        strategy_profiler.update("strategy_B", latency=2.0, success=False)

        stats = strategy_profiler.get_statistics()

        assert stats["total_strategies"] == 2
        assert stats["total_updates"] == 2
        assert "strategy_summaries" in stats
        assert "strategy_A" in stats["strategy_summaries"]
        assert "strategy_B" in stats["strategy_summaries"]

    def test_get_strategy_profile(self, strategy_profiler):
        """Test getting individual strategy profile"""
        strategy_profiler.update("strategy_A", latency=1.5, success=True)

        profile = strategy_profiler.get_strategy_profile("strategy_A")

        assert profile is not None
        assert profile.name == "strategy_A"
        assert profile.total_attempts == 1

    def test_get_all_profiles(self, strategy_profiler):
        """Test getting all profiles"""
        strategy_profiler.update("strategy_A", latency=1.0, success=True)
        strategy_profiler.update("strategy_B", latency=2.0, success=True)

        all_profiles = strategy_profiler.get_all_profiles()

        assert len(all_profiles) == 2
        assert "strategy_A" in all_profiles
        assert "strategy_B" in all_profiles


# ============================================================
# THREAD SAFETY TESTS
# ============================================================


class TestThreadSafety:
    """Test thread safety of components"""

    def test_adaptive_thresholds_concurrent_access(self, adaptive_thresholds):
        """Test concurrent threshold adjustments"""

        def adjust_thresholds():
            for _ in range(100):
                adaptive_thresholds.adjust_up(0.01, "confidence")
                adaptive_thresholds.adjust_down(0.01, "timeout")

        threads = [threading.Thread(target=adjust_thresholds) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        stats = adaptive_thresholds.get_statistics()
        assert stats["total_adjustments"] > 0

    def test_performance_tracker_concurrent_recording(self, performance_tracker):
        """Test concurrent performance recording"""

        def record_performances():
            for i in range(50):
                performance_tracker.record_success(f"prob_{i}", "strategy", 1.0)
                performance_tracker.record_failure(f"prob_fail_{i}", "failed")

        threads = [threading.Thread(target=record_performances) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        stats = performance_tracker.get_statistics()
        assert stats["total_attempts"] > 0

    def test_strategy_profiler_concurrent_updates(self, strategy_profiler):
        """Test concurrent strategy updates"""

        def update_strategies():
            for i in range(50):
                strategy_profiler.update(
                    f"strategy_{i % 5}", latency=float(i % 10), success=i % 2 == 0
                )

        threads = [threading.Thread(target=update_strategies) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        stats = strategy_profiler.get_statistics()
        assert stats["total_strategies"] > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Test integration between components"""

    def test_thresholds_with_performance_tracker(
        self, adaptive_thresholds, performance_tracker
    ):
        """Test using performance tracker data for threshold calibration"""
        # Record some performances
        for i in range(30):
            performance_tracker.record_success(
                f"prob_{i}", "strategy", 1.0, metadata={"complexity": 2.0}
            )

        for i in range(10):
            performance_tracker.record_failure(f"prob_fail_{i}", "timeout")

        # Get recent performance
        recent_records = list(performance_tracker.records)[-20:]

        # Convert to format for calibration
        performance_data = [
            {
                "success": r.success,
                "execution_time": r.execution_time,
                "complexity": r.metadata.get("complexity", 0),
            }
            for r in recent_records
        ]

        # Calibrate thresholds
        initial_values = adaptive_thresholds.get_current()
        adaptive_thresholds.auto_calibrate(performance_data)
        new_values = adaptive_thresholds.get_current()

        # Thresholds should have been adjusted
        assert initial_values != new_values or len(performance_data) == 0

    def test_profiler_with_performance_tracker(
        self, strategy_profiler, performance_tracker
    ):
        """Test using performance tracker for strategy profiling"""
        # Record performances
        for i in range(20):
            strategy = f"strategy_{i % 3}"
            latency = float(i % 5)
            success = i % 2 == 0

            if success:
                performance_tracker.record_success(f"prob_{i}", strategy, latency)
            else:
                performance_tracker.record_failure(f"prob_{i}", "failed", strategy)

            strategy_profiler.update(strategy, latency, success)

        # Get recommendation
        recommendation = strategy_profiler.recommend_strategy()

        # Should recommend one of the strategies
        assert recommendation in ["strategy_0", "strategy_1", "strategy_2"]

        # Check consistency between tracker and profiler
        for strategy_name in ["strategy_0", "strategy_1", "strategy_2"]:
            tracker_perf = performance_tracker.get_strategy_performance(strategy_name)
            profiler_profile = strategy_profiler.get_strategy_profile(strategy_name)

            if profiler_profile:
                assert tracker_perf["total_attempts"] == profiler_profile.total_attempts


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_threshold_adjustment_with_zero_factor(self, adaptive_thresholds):
        """Test adjustment with zero factor"""
        initial = adaptive_thresholds.get_current("confidence")
        adaptive_thresholds.adjust_up(0.0, "confidence")
        final = adaptive_thresholds.get_current("confidence")

        assert initial == final  # No change

    def test_threshold_adjustment_with_large_factor(self, adaptive_thresholds):
        """Test adjustment with very large factor"""
        adaptive_thresholds.adjust_up(100.0, "confidence")
        final = adaptive_thresholds.get_current("confidence")

        assert final <= 0.99  # Should hit max bound

    def test_empty_performance_window_calibration(self, adaptive_thresholds):
        """Test calibration with empty performance window"""
        # Should not crash
        adaptive_thresholds.auto_calibrate([])

    def test_performance_record_with_none_values(self, performance_tracker):
        """Test recording with None values"""
        # Should handle gracefully
        performance_tracker.record_success(
            "prob", "strategy", execution_time=0.0, metadata=None
        )

        assert len(performance_tracker.records) == 1


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
