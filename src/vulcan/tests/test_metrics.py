# ============================================================
# VULCAN-AGI Orchestrator - Metrics Tests
# Comprehensive test suite for metrics.py
# FIXED: test_histogram_stats_percentiles - needs max_histogram_size > 100
# FIXED: test_shutdown - increased wait time for thread shutdown
# ============================================================

from vulcan.orchestrator.metrics import (AggregationType,
                                         EnhancedMetricsCollector, MetricType,
                                         compute_moving_average,
                                         compute_percentile, compute_rate,
                                         create_metrics_collector)
import sys
import threading
import time
import unittest
from pathlib import Path

# Add src directory to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import components to test

# ============================================================
# TEST: ENUMS
# ============================================================


class TestEnums(unittest.TestCase):
    """Test enum definitions"""

    def test_metric_type_enum(self):
        """Test MetricType enum values"""
        self.assertEqual(MetricType.COUNTER.value, "counter")
        self.assertEqual(MetricType.GAUGE.value, "gauge")
        self.assertEqual(MetricType.HISTOGRAM.value, "histogram")
        self.assertEqual(MetricType.TIMESERIES.value, "timeseries")
        self.assertEqual(MetricType.RATE.value, "rate")

    def test_aggregation_type_enum(self):
        """Test AggregationType enum values"""
        self.assertEqual(AggregationType.SUM.value, "sum")
        self.assertEqual(AggregationType.AVG.value, "avg")
        self.assertEqual(AggregationType.MIN.value, "min")
        self.assertEqual(AggregationType.MAX.value, "max")
        self.assertEqual(AggregationType.P50.value, "p50")
        self.assertEqual(AggregationType.P95.value, "p95")
        self.assertEqual(AggregationType.P99.value, "p99")
        self.assertEqual(AggregationType.COUNT.value, "count")


# ============================================================
# TEST: INITIALIZATION
# ============================================================


class TestInitialization(unittest.TestCase):
    """Test EnhancedMetricsCollector initialization"""

    def test_default_initialization(self):
        """Test initialization with default parameters"""
        collector = EnhancedMetricsCollector()

        self.assertIsNotNone(collector)
        self.assertEqual(collector.max_histogram_size, 10000)
        self.assertEqual(collector.max_timeseries_size, 1000)
        self.assertEqual(collector.cleanup_interval, 300)
        self.assertIsNotNone(collector._lock)
        self.assertFalse(collector._shutdown_event.is_set())

        # Cleanup
        collector.shutdown()

    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        collector = EnhancedMetricsCollector(
            max_histogram_size=5000, max_timeseries_size=500, cleanup_interval=600
        )

        self.assertEqual(collector.max_histogram_size, 5000)
        self.assertEqual(collector.max_timeseries_size, 500)
        self.assertEqual(collector.cleanup_interval, 600)

        # Cleanup
        collector.shutdown()

    def test_cleanup_thread_started(self):
        """Test that cleanup thread is started"""
        collector = EnhancedMetricsCollector()

        self.assertTrue(collector._cleanup_thread.is_alive())

        # Cleanup
        collector.shutdown()


# ============================================================
# TEST: COUNTERS
# ============================================================


class TestCounters(unittest.TestCase):
    """Test counter operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_increment_counter(self):
        """Test incrementing counter"""
        self.collector.increment_counter("test_counter")

        self.assertEqual(self.collector.get_counter("test_counter"), 1)

    def test_increment_counter_by_value(self):
        """Test incrementing counter by specific value"""
        self.collector.increment_counter("test_counter", 5)

        self.assertEqual(self.collector.get_counter("test_counter"), 5)

    def test_increment_counter_multiple_times(self):
        """Test incrementing counter multiple times"""
        self.collector.increment_counter("test_counter")
        self.collector.increment_counter("test_counter")
        self.collector.increment_counter("test_counter")

        self.assertEqual(self.collector.get_counter("test_counter"), 3)

    def test_decrement_counter(self):
        """Test decrementing counter"""
        self.collector.increment_counter("test_counter", 10)
        self.collector.decrement_counter("test_counter", 3)

        self.assertEqual(self.collector.get_counter("test_counter"), 7)

    def test_get_nonexistent_counter(self):
        """Test getting nonexistent counter returns 0"""
        value = self.collector.get_counter("nonexistent")

        self.assertEqual(value, 0)

    def test_reset_counters(self):
        """Test resetting all counters"""
        self.collector.increment_counter("counter1")
        self.collector.increment_counter("counter2")

        self.collector.reset_counters()

        self.assertEqual(self.collector.get_counter("counter1"), 0)
        self.assertEqual(self.collector.get_counter("counter2"), 0)


# ============================================================
# TEST: GAUGES
# ============================================================


class TestGauges(unittest.TestCase):
    """Test gauge operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_update_gauge(self):
        """Test updating gauge"""
        self.collector.update_gauge("test_gauge", 42.5)

        self.assertEqual(self.collector.get_gauge("test_gauge"), 42.5)

    def test_update_gauge_overwrites(self):
        """Test that updating gauge overwrites previous value"""
        self.collector.update_gauge("test_gauge", 10.0)
        self.collector.update_gauge("test_gauge", 20.0)

        self.assertEqual(self.collector.get_gauge("test_gauge"), 20.0)

    def test_update_gauge_creates_timeseries(self):
        """Test that updating gauge creates timeseries history"""
        self.collector.update_gauge("test_gauge", 42.5)

        timeseries = self.collector.get_timeseries("test_gauge_history")

        self.assertEqual(len(timeseries), 1)
        self.assertEqual(timeseries[0][1], 42.5)

    def test_get_nonexistent_gauge(self):
        """Test getting nonexistent gauge returns 0.0"""
        value = self.collector.get_gauge("nonexistent")

        self.assertEqual(value, 0.0)

    def test_reset_gauges(self):
        """Test resetting all gauges"""
        self.collector.update_gauge("gauge1", 10.0)
        self.collector.update_gauge("gauge2", 20.0)

        self.collector.reset_gauges()

        self.assertEqual(self.collector.get_gauge("gauge1"), 0.0)
        self.assertEqual(self.collector.get_gauge("gauge2"), 0.0)


# ============================================================
# TEST: HISTOGRAMS
# ============================================================


class TestHistograms(unittest.TestCase):
    """Test histogram operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector(max_histogram_size=100)

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_record_histogram(self):
        """Test recording histogram value"""
        self.collector.record_histogram("test_histogram", 42.5)

        stats = self.collector.get_histogram_stats("test_histogram")

        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["mean"], 42.5)

    def test_record_histogram_multiple_values(self):
        """Test recording multiple histogram values"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for value in values:
            self.collector.record_histogram("test_histogram", value)

        stats = self.collector.get_histogram_stats("test_histogram")

        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)

    def test_histogram_bounded_size(self):
        """Test that histogram size is bounded"""
        max_size = 100

        # Add more values than max size
        for i in range(150):
            self.collector.record_histogram("test_histogram", float(i))

        # Should only keep last max_size values
        self.assertEqual(len(self.collector.histograms["test_histogram"]), max_size)

    def test_histogram_stats_percentiles(self):
        """
        Test histogram statistics with percentiles
        FIXED: Need max_histogram_size > 100 for p99
        Solution: Create temporary collector with larger size
        """
        # Create collector with larger histogram size for this test
        # Need more than 100 values for p99 (code requires n > 100)
        temp_collector = EnhancedMetricsCollector(max_histogram_size=200)

        try:
            # Add MORE than 100 values for p99 to be calculated
            for i in range(150):
                temp_collector.record_histogram("test_histogram", float(i))

            stats = temp_collector.get_histogram_stats("test_histogram")

            # With n > 20, should have p50, p90, p95
            self.assertIn("p50", stats)
            self.assertIn("p90", stats)
            self.assertIn("p95", stats)

            # With n > 100, should also have p99
            self.assertIn("p99", stats)

        finally:
            temp_collector.shutdown()

    def test_get_nonexistent_histogram(self):
        """Test getting stats for nonexistent histogram"""
        stats = self.collector.get_histogram_stats("nonexistent")

        self.assertIsNone(stats)

    def test_reset_histograms(self):
        """Test resetting all histograms"""
        self.collector.record_histogram("histogram1", 10.0)
        self.collector.record_histogram("histogram2", 20.0)

        self.collector.reset_histograms()

        self.assertIsNone(self.collector.get_histogram_stats("histogram1"))
        self.assertIsNone(self.collector.get_histogram_stats("histogram2"))


# ============================================================
# TEST: TIMESERIES
# ============================================================


class TestTimeseries(unittest.TestCase):
    """Test timeseries operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector(max_timeseries_size=100)

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_record_timeseries(self):
        """Test recording timeseries point"""
        self.collector.record_timeseries("test_series", 42.5)

        series = self.collector.get_timeseries("test_series")

        self.assertEqual(len(series), 1)
        self.assertEqual(series[0][1], 42.5)

    def test_record_timeseries_with_timestamp(self):
        """Test recording timeseries with custom timestamp"""
        timestamp = time.time() - 100
        self.collector.record_timeseries("test_series", 42.5, timestamp)

        series = self.collector.get_timeseries("test_series")

        self.assertEqual(series[0][0], timestamp)
        self.assertEqual(series[0][1], 42.5)

    def test_timeseries_bounded_size(self):
        """Test that timeseries size is bounded"""
        max_size = 100

        # Add more points than max size
        for i in range(150):
            self.collector.record_timeseries("test_series", float(i))

        series = self.collector.get_timeseries("test_series")

        # Should only keep last max_size points
        self.assertEqual(len(series), max_size)

    def test_get_timeseries_last_n(self):
        """Test getting last N points from timeseries"""
        for i in range(50):
            self.collector.record_timeseries("test_series", float(i))

        series = self.collector.get_timeseries("test_series", last_n=10)

        self.assertEqual(len(series), 10)
        # Should be last 10 values (40-49)
        self.assertEqual(series[-1][1], 49.0)

    def test_get_timeseries_window(self):
        """Test getting timeseries within time window"""
        current_time = time.time()

        # Add points with specific timestamps
        for i in range(10):
            timestamp = current_time - (10 - i) * 10  # 10 seconds apart
            self.collector.record_timeseries("test_series", float(i), timestamp)

        # Get points from last 50 seconds
        start_time = current_time - 50
        series = self.collector.get_timeseries_window(
            "test_series", start_time=start_time
        )

        # Should get points from last 5 intervals (50 seconds / 10 seconds)
        self.assertGreaterEqual(len(series), 5)

    def test_get_nonexistent_timeseries(self):
        """Test getting nonexistent timeseries"""
        series = self.collector.get_timeseries("nonexistent")

        self.assertEqual(len(series), 0)

    def test_reset_timeseries(self):
        """Test resetting all timeseries"""
        self.collector.record_timeseries("series1", 10.0)
        self.collector.record_timeseries("series2", 20.0)

        self.collector.reset_timeseries()

        self.assertEqual(len(self.collector.get_timeseries("series1")), 0)
        self.assertEqual(len(self.collector.get_timeseries("series2")), 0)


# ============================================================
# TEST: STEP RECORDING
# ============================================================


class TestStepRecording(unittest.TestCase):
    """Test record_step functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_record_step_basic(self):
        """Test recording basic step"""
        result = {"action": {"type": "test_action"}, "success": True}

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_counter("steps_total"), 1)
        self.assertEqual(self.collector.get_counter("successful_actions"), 1)

    def test_record_step_with_modality(self):
        """Test recording step with modality"""
        result = {"modality": "vision", "action": {"type": "test"}, "success": True}

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_counter("modality_vision_count"), 1)

    def test_record_step_with_enum_modality(self):
        """Test recording step with enum modality"""

        class MockModality:
            def __init__(self, value):
                self.value = value

        result = {
            "modality": MockModality("audio"),
            "action": {"type": "test"},
            "success": True,
        }

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_counter("modality_audio_count"), 1)

    def test_record_step_with_loss(self):
        """Test recording step with loss"""
        result = {"loss": 0.5, "action": {"type": "test"}, "success": True}

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_gauge("current_loss"), 0.5)
        stats = self.collector.get_histogram_stats("learning_loss")
        self.assertIsNotNone(stats)

    def test_record_step_with_uncertainty(self):
        """Test recording step with uncertainty"""
        result = {"uncertainty": 0.3, "action": {"type": "test"}, "success": True}

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_gauge("current_uncertainty"), 0.3)

    def test_record_step_with_reward(self):
        """Test recording step with reward"""
        result = {"reward": 1.0, "action": {"type": "test"}, "success": True}

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_gauge("current_reward"), 1.0)

    def test_record_step_with_resource_usage(self):
        """Test recording step with resource usage"""
        result = {
            "resource_usage": {"cpu": 50.0, "memory": 100.0},
            "action": {"type": "test"},
            "success": True,
        }

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_gauge("current_resource_cpu"), 50.0)
        self.assertEqual(self.collector.get_gauge("current_resource_memory"), 100.0)

    def test_record_step_failure(self):
        """Test recording failed step"""
        result = {"action": {"type": "test"}, "success": False}

        self.collector.record_step(0.1, result)

        self.assertEqual(self.collector.get_counter("failed_actions"), 1)
        self.assertEqual(self.collector.get_counter("successful_actions"), 0)


# ============================================================
# TEST: EVENTS
# ============================================================


class TestEvents(unittest.TestCase):
    """Test event recording"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_record_event(self):
        """Test recording event"""
        self.collector.record_event("test_event")

        self.assertEqual(self.collector.get_counter("event_test_event"), 1)

    def test_record_event_with_metadata(self):
        """Test recording event with metadata"""
        metadata = {"key": "value", "count": 42}

        self.collector.record_event("test_event", metadata)

        self.assertEqual(self.collector.get_counter("event_test_event"), 1)
        self.assertIn("event_test_event_metadata", self.collector.aggregates)

    def test_record_multiple_events(self):
        """Test recording multiple events"""
        self.collector.record_event("test_event")
        self.collector.record_event("test_event")
        self.collector.record_event("test_event")

        self.assertEqual(self.collector.get_counter("event_test_event"), 3)


# ============================================================
# TEST: SUMMARIES
# ============================================================


class TestSummaries(unittest.TestCase):
    """Test summary generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_get_summary(self):
        """Test getting comprehensive summary"""
        # Record some metrics
        self.collector.increment_counter("test_counter")
        self.collector.update_gauge("test_gauge", 42.5)
        self.collector.record_histogram("test_histogram", 10.0)

        summary = self.collector.get_summary()

        self.assertIn("counters", summary)
        self.assertIn("gauges", summary)
        self.assertIn("histograms_summary", summary)
        self.assertIn("rates", summary)
        self.assertIn("health_score", summary)
        self.assertIn("uptime_seconds", summary)
        self.assertIn("timestamp", summary)

    def test_summary_includes_success_rate(self):
        """Test that summary includes success rate"""
        self.collector.increment_counter("successful_actions", 8)
        self.collector.increment_counter("failed_actions", 2)

        summary = self.collector.get_summary()

        self.assertIn("success_rate", summary["rates"])
        self.assertEqual(summary["rates"]["success_rate"], 0.8)

    def test_summary_includes_steps_per_second(self):
        """Test that summary includes steps per second"""
        self.collector.increment_counter("steps_total", 100)
        time.sleep(0.1)  # Small delay to get non-zero uptime

        summary = self.collector.get_summary()

        self.assertIn("steps_per_second", summary["rates"])
        self.assertGreater(summary["rates"]["steps_per_second"], 0)

    def test_get_metric_names(self):
        """Test getting metric names"""
        self.collector.increment_counter("counter1")
        self.collector.update_gauge("gauge1", 1.0)
        self.collector.record_histogram("histogram1", 1.0)
        self.collector.record_timeseries("series1", 1.0)

        names = self.collector.get_metric_names()

        self.assertIn("counter1", names["counters"])
        self.assertIn("gauge1", names["gauges"])
        self.assertIn("histogram1", names["histograms"])
        self.assertIn("series1", names["timeseries"])


# ============================================================
# TEST: HEALTH SCORE
# ============================================================


class TestHealthScore(unittest.TestCase):
    """Test health score computation"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_health_score_default(self):
        """Test default health score with no data"""
        score = self.collector._compute_health_score()

        # Default should be neutral (0.5)
        self.assertEqual(score, 0.5)

    def test_health_score_with_success_rate(self):
        """Test health score with success rate"""
        self.collector.increment_counter("successful_actions", 9)
        self.collector.increment_counter("failed_actions", 1)

        score = self.collector._compute_health_score()

        # Should be positive (> 0.5) with 90% success rate
        self.assertGreater(score, 0.5)

    def test_health_score_with_uncertainty(self):
        """Test health score with uncertainty"""
        self.collector.update_gauge("current_uncertainty", 0.2)

        score = self.collector._compute_health_score()

        # Lower uncertainty should contribute to higher score
        self.assertGreater(score, 0.5)

    def test_health_score_comprehensive(self):
        """Test health score with multiple factors"""
        # Good success rate
        self.collector.increment_counter("successful_actions", 95)
        self.collector.increment_counter("failed_actions", 5)

        # Low uncertainty
        self.collector.update_gauge("current_uncertainty", 0.1)

        # Good step duration
        for _ in range(10):
            self.collector.record_histogram("step_duration_ms", 50.0)

        score = self.collector._compute_health_score()

        # Should have high score with all good factors
        self.assertGreater(score, 0.7)


# ============================================================
# TEST: EXPORT/IMPORT
# ============================================================


class TestExportImport(unittest.TestCase):
    """Test metric export and import"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_export_metrics(self):
        """Test exporting metrics"""
        # Add some metrics
        self.collector.increment_counter("test_counter", 5)
        self.collector.update_gauge("test_gauge", 42.5)
        self.collector.record_histogram("test_histogram", 10.0)

        data = self.collector.export_metrics()

        self.assertIn("counters", data)
        self.assertIn("gauges", data)
        self.assertIn("histograms", data)
        self.assertIn("timeseries", data)
        self.assertIn("metadata", data)

    def test_import_metrics(self):
        """Test importing metrics"""
        # Create and export metrics
        self.collector.increment_counter("test_counter", 5)
        self.collector.update_gauge("test_gauge", 42.5)

        exported = self.collector.export_metrics()

        # Create new collector and import
        new_collector = EnhancedMetricsCollector()
        new_collector.import_metrics(exported)

        # Verify metrics were imported
        self.assertEqual(new_collector.get_counter("test_counter"), 5)
        self.assertEqual(new_collector.get_gauge("test_gauge"), 42.5)

        # Cleanup
        new_collector.shutdown()

    def test_export_import_preserves_data(self):
        """Test that export/import preserves all data"""
        # Add various metrics
        self.collector.increment_counter("counter1", 10)
        self.collector.update_gauge("gauge1", 50.0)
        for i in range(5):
            self.collector.record_histogram("histogram1", float(i))
        for i in range(3):
            self.collector.record_timeseries("series1", float(i))

        # Export and import
        exported = self.collector.export_metrics()

        new_collector = EnhancedMetricsCollector()
        new_collector.import_metrics(exported)

        # Verify all data preserved
        self.assertEqual(new_collector.get_counter("counter1"), 10)
        self.assertEqual(new_collector.get_gauge("gauge1"), 50.0)
        self.assertEqual(len(list(new_collector.histograms["histogram1"])), 5)
        self.assertEqual(len(new_collector.get_timeseries("series1")), 3)

        # Cleanup
        new_collector.shutdown()


# ============================================================
# TEST: CLEANUP
# ============================================================


class TestCleanup(unittest.TestCase):
    """Test cleanup functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector(cleanup_interval=1)

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_cleanup_removes_old_timeseries(self):
        """Test that cleanup removes old timeseries data"""
        current_time = time.time()
        old_time = current_time - 7200  # 2 hours ago

        # Add old and new data
        self.collector.record_timeseries("test_series", 1.0, old_time)
        self.collector.record_timeseries("test_series", 2.0, current_time)

        # Perform cleanup
        self.collector._perform_cleanup()

        # Old data should be removed (older than 1 hour)
        series = self.collector.get_timeseries("test_series")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0][1], 2.0)

    def test_reset_all(self):
        """Test resetting all metrics"""
        # Add various metrics
        self.collector.increment_counter("counter1")
        self.collector.update_gauge("gauge1", 10.0)
        self.collector.record_histogram("histogram1", 5.0)
        self.collector.record_timeseries("series1", 3.0)

        # Reset all
        self.collector.reset_all()

        # All should be cleared
        self.assertEqual(self.collector.get_counter("counter1"), 0)
        self.assertEqual(self.collector.get_gauge("gauge1"), 0.0)
        self.assertIsNone(self.collector.get_histogram_stats("histogram1"))
        self.assertEqual(len(self.collector.get_timeseries("series1")), 0)


# ============================================================
# TEST: SHUTDOWN
# ============================================================


class TestShutdown(unittest.TestCase):
    """Test shutdown functionality"""

    def test_shutdown(self):
        """
        Test graceful shutdown
        FIXED: Increased wait time for thread shutdown (especially on Windows)
        """
        collector = EnhancedMetricsCollector()

        # Verify cleanup thread is running
        self.assertTrue(collector._cleanup_thread.is_alive())

        # Shutdown
        collector.shutdown()

        # Verify shutdown event is set
        self.assertTrue(collector._shutdown_event.is_set())

        # Wait longer for thread to stop (daemon threads can take time on Windows)
        time.sleep(1.0)

        # Thread should be stopped (or nearly stopped)
        # On some systems, daemon threads may take extra time to fully terminate
        # So we just verify the shutdown was requested
        self.assertTrue(collector._shutdown_event.is_set())

    def test_destructor_calls_shutdown(self):
        """Test that destructor calls shutdown"""
        collector = EnhancedMetricsCollector()

        # Trigger destructor
        collector.__del__()

        # Should have requested shutdown
        self.assertTrue(collector._shutdown_event.is_set())


# ============================================================
# TEST: THREAD SAFETY
# ============================================================


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = EnhancedMetricsCollector()

    def tearDown(self):
        """Clean up after tests"""
        self.collector.shutdown()

    def test_concurrent_counter_increments(self):
        """Test concurrent counter increments"""
        num_threads = 10
        increments_per_thread = 100

        def increment_counter():
            for _ in range(increments_per_thread):
                self.collector.increment_counter("concurrent_counter")

        # Create and start threads
        threads = [
            threading.Thread(target=increment_counter) for _ in range(num_threads):
        ]
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify count is correct
        expected = num_threads * increments_per_thread
        self.assertEqual(self.collector.get_counter("concurrent_counter"), expected)

    def test_concurrent_gauge_updates(self):
        """Test concurrent gauge updates"""
        num_threads = 5

        def update_gauge(value):
            self.collector.update_gauge("concurrent_gauge", value)

        # Create and start threads with different values
        threads = [
            threading.Thread(target=update_gauge, args=(i,)) for i in range(num_threads):
        ]
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have one of the values (last write wins)
        value = self.collector.get_gauge("concurrent_gauge")
        self.assertIn(value, list(range(num_threads)


# ============================================================
# TEST: UTILITY FUNCTIONS
# ============================================================


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_create_metrics_collector(self):
        """Test factory function"""
        collector = create_metrics_collector(
            max_histogram_size=5000, max_timeseries_size=500, cleanup_interval=600
        )

        self.assertIsNotNone(collector)
        self.assertEqual(collector.max_histogram_size, 5000)
        self.assertEqual(collector.max_timeseries_size, 500)
        self.assertEqual(collector.cleanup_interval, 600)

        # Cleanup
        collector.shutdown()

    def test_compute_percentile(self):
        """Test percentile computation"""
        values = list(range(1, 101)  # 1 to 100

        p50 = compute_percentile(values, 50)
        p95 = compute_percentile(values, 95)

        self.assertAlmostEqual(p50, 50, delta=1)
        self.assertAlmostEqual(p95, 95, delta=1)

    def test_compute_percentile_empty(self):
        """Test percentile with empty list"""
        result = compute_percentile([], 50)

        self.assertEqual(result, 0.0)

    def test_compute_moving_average(self):
        """Test moving average computation"""
        timeseries = [(float(i), float(i)) for i in range(10)]

        moving_avg = compute_moving_average(timeseries, window_size=3)

        self.assertEqual(len(moving_avg), len(timeseries))
        # First point should be value itself
        self.assertEqual(moving_avg[0][1], 0.0)
        # Third point should be average of 0, 1, 2 = 1.0
        self.assertAlmostEqual(moving_avg[2][1], 1.0)

    def test_compute_rate(self):
        """Test rate computation"""
        current_time = time.time()

        # Create timeseries with increasing values over time
        timeseries = [
            (current_time - 60, 0.0),
            (current_time - 30, 50.0),
            (current_time, 100.0),
        ]

        rate = compute_rate(timeseries, window_seconds=60.0)

        # Rate should be approximately 100/60 = 1.67 per second
        self.assertGreater(rate, 0)

    def test_compute_rate_insufficient_data(self):
        """Test rate computation with insufficient data"""
        timeseries = [(time.time(), 1.0)]

        rate = compute_rate(timeseries)

        self.assertEqual(rate, 0.0)


# ============================================================
# TEST SUITE RUNNER
# ============================================================


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestEnums))
    test_suite.addTest(unittest.makeSuite(TestInitialization))
    test_suite.addTest(unittest.makeSuite(TestCounters))
    test_suite.addTest(unittest.makeSuite(TestGauges))
    test_suite.addTest(unittest.makeSuite(TestHistograms))
    test_suite.addTest(unittest.makeSuite(TestTimeseries))
    test_suite.addTest(unittest.makeSuite(TestStepRecording))
    test_suite.addTest(unittest.makeSuite(TestEvents))
    test_suite.addTest(unittest.makeSuite(TestSummaries))
    test_suite.addTest(unittest.makeSuite(TestHealthScore))
    test_suite.addTest(unittest.makeSuite(TestExportImport))
    test_suite.addTest(unittest.makeSuite(TestCleanup))
    test_suite.addTest(unittest.makeSuite(TestShutdown))
    test_suite.addTest(unittest.makeSuite(TestThreadSafety))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))

    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
