"""
Comprehensive test suite for tool_monitor.py
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import time
import threading

from tool_monitor import (
    MetricType,
    AlertSeverity,
    HealthStatus,
    ToolMetrics,
    SystemMetrics,
    Alert,
    TimeSeriesBuffer,
    AnomalyDetector,
    ToolMonitor,
)


@pytest.fixture
def tool_metrics():
    """Create ToolMetrics instance."""
    return ToolMetrics(tool_name="test_tool")


@pytest.fixture
def system_metrics():
    """Create SystemMetrics instance."""
    return SystemMetrics()


@pytest.fixture
def time_series_buffer():
    """Create TimeSeriesBuffer instance."""
    return TimeSeriesBuffer(window_size=100)


@pytest.fixture
def anomaly_detector():
    """Create AnomalyDetector instance."""
    return AnomalyDetector()


@pytest.fixture
def tool_monitor():
    """Create ToolMonitor instance."""
    monitor = ToolMonitor(config={'monitoring_interval': 10.0})  # Long interval for tests
    yield monitor
    monitor.shutdown()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEnums:
    """Test enum classes."""
    
    def test_metric_type_enum(self):
        """Test MetricType enum."""
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.ERROR_RATE.value == "error_rate"
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum."""
        assert AlertSeverity.INFO.value == 0
        assert AlertSeverity.CRITICAL.value == 3
    
    def test_health_status_enum(self):
        """Test HealthStatus enum."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.CRITICAL.value == "critical"


class TestToolMetrics:
    """Test ToolMetrics dataclass."""
    
    def test_initialization(self, tool_metrics):
        """Test metrics initialization."""
        assert tool_metrics.tool_name == "test_tool"
        assert tool_metrics.total_executions == 0
        assert tool_metrics.health_score == 1.0
    
    def test_update_latency_percentiles(self, tool_metrics):
        """Test updating latency percentiles."""
        latencies = [100, 200, 300, 400, 500]
        
        tool_metrics.update_latency_percentiles(latencies)
        
        assert tool_metrics.p50_latency_ms == 300
        assert tool_metrics.p95_latency_ms > 400
    
    def test_update_latency_percentiles_empty(self, tool_metrics):
        """Test updating with empty latencies."""
        tool_metrics.update_latency_percentiles([])
        
        # Should not crash
        assert tool_metrics.p50_latency_ms == 0.0
    
    def test_to_dict(self, tool_metrics):
        """Test conversion to dictionary."""
        result = tool_metrics.to_dict()
        
        assert isinstance(result, dict)
        assert 'tool_name' in result
        assert 'total_executions' in result
        assert 'health_score' in result


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""
    
    def test_initialization(self, system_metrics):
        """Test system metrics initialization."""
        assert system_metrics.total_requests == 0
        assert system_metrics.cpu_usage_percent == 0.0
    
    def test_to_dict(self, system_metrics):
        """Test conversion to dictionary."""
        result = system_metrics.to_dict()
        
        assert isinstance(result, dict)
        assert 'total_requests' in result
        assert 'cpu_usage_percent' in result


class TestAlert:
    """Test Alert dataclass."""
    
    def test_creation(self):
        """Test creating alert."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            metric_type=MetricType.LATENCY,
            message="Test alert",
            tool_name="test_tool",
            value=500.0,
            threshold=300.0
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            metric_type=MetricType.ERROR_RATE,
            message="Error occurred"
        )
        
        result = alert.to_dict()
        
        assert isinstance(result, dict)
        assert result['severity'] == 'ERROR'
        assert result['metric_type'] == 'error_rate'


class TestTimeSeriesBuffer:
    """Test TimeSeriesBuffer."""
    
    def test_initialization(self, time_series_buffer):
        """Test buffer initialization."""
        assert time_series_buffer.window_size == 100
        assert len(time_series_buffer.data) == 0
    
    def test_add_value(self, time_series_buffer):
        """Test adding value."""
        time_series_buffer.add(10.0)
        
        assert len(time_series_buffer.data) == 1
        assert time_series_buffer.data[0] == 10.0
    
    def test_add_multiple_values(self, time_series_buffer):
        """Test adding multiple values."""
        for i in range(10):
            time_series_buffer.add(float(i))
        
        assert len(time_series_buffer.data) == 10
    
    def test_window_size_limit(self):
        """Test that window size is enforced."""
        buffer = TimeSeriesBuffer(window_size=10)
        
        for i in range(20):
            buffer.add(float(i))
        
        assert len(buffer.data) == 10
    
    def test_get_stats_empty(self):
        """Test getting stats from empty buffer."""
        buffer = TimeSeriesBuffer()
        
        stats = buffer.get_stats()
        
        assert stats['mean'] == 0.0
        assert stats['count'] == 0
    
    def test_get_stats_with_data(self, time_series_buffer):
        """Test getting stats with data."""
        for i in range(10):
            time_series_buffer.add(float(i))
        
        stats = time_series_buffer.get_stats()
        
        assert stats['mean'] == 4.5
        assert stats['count'] == 10
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
    
    def test_get_trend_insufficient_data(self, time_series_buffer):
        """Test trend with insufficient data."""
        for i in range(5):
            time_series_buffer.add(float(i))
        
        trend = time_series_buffer.get_trend()
        
        assert trend == "insufficient_data"
    
    def test_get_trend_increasing(self, time_series_buffer):
        """Test detecting increasing trend."""
        for i in range(20):
            time_series_buffer.add(float(i))
        
        trend = time_series_buffer.get_trend()
        
        assert trend == "increasing"
    
    def test_get_trend_decreasing(self, time_series_buffer):
        """Test detecting decreasing trend."""
        for i in range(20, 0, -1):
            time_series_buffer.add(float(i))
        
        trend = time_series_buffer.get_trend()
        
        assert trend == "decreasing"
    
    def test_get_trend_stable(self, time_series_buffer):
        """Test detecting stable trend."""
        for i in range(20):
            time_series_buffer.add(5.0)
        
        trend = time_series_buffer.get_trend()
        
        assert trend == "stable"


class TestAnomalyDetector:
    """Test AnomalyDetector."""
    
    def test_initialization(self, anomaly_detector):
        """Test detector initialization."""
        assert anomaly_detector.z_threshold == 3.0
    
    def test_update_no_anomaly(self, anomaly_detector):
        """Test update with normal values."""
        for i in range(30):
            anomaly = anomaly_detector.update("metric1", 100.0)
            assert anomaly is False
    
    def test_update_with_anomaly(self, anomaly_detector):
        """Test update with anomalous value."""
        # Add baseline data
        for i in range(30):
            anomaly_detector.update("metric1", 100.0)
        
        # Add anomaly
        anomaly = anomaly_detector.update("metric1", 500.0)
        
        assert anomaly is True
    
    def test_get_anomaly_score(self, anomaly_detector):
        """Test getting anomaly score."""
        # Build baseline
        for i in range(30):
            anomaly_detector.update("metric1", 100.0)
        
        # Get score for normal value
        score1 = anomaly_detector.get_anomaly_score("metric1", 100.0)
        assert score1 < 0.5
        
        # Get score for anomalous value
        score2 = anomaly_detector.get_anomaly_score("metric1", 500.0)
        assert score2 > 0.5


class TestToolMonitor:
    """Test ToolMonitor main class."""
    
    def test_initialization(self, tool_monitor):
        """Test monitor initialization."""
        assert tool_monitor is not None
        assert tool_monitor.monitoring is True
    
    def test_record_execution_success(self, tool_monitor):
        """Test recording successful execution."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.9
        )
        
        metrics = tool_monitor.tool_metrics["test_tool"]
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.consecutive_failures == 0
    
    def test_record_execution_failure(self, tool_monitor):
        """Test recording failed execution."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=False,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.5
        )
        
        metrics = tool_monitor.tool_metrics["test_tool"]
        
        assert metrics.failed_executions == 1
        assert metrics.consecutive_failures == 1
    
    def test_record_multiple_executions(self, tool_monitor):
        """Test recording multiple executions."""
        for i in range(10):
            tool_monitor.record_execution(
                tool_name="test_tool",
                success=True,
                latency_ms=100.0 + i,
                energy_mj=10.0,
                confidence=0.8
            )
        
        metrics = tool_monitor.tool_metrics["test_tool"]
        
        assert metrics.total_executions == 10
        assert metrics.successful_executions == 10
    
    def test_calculate_health_score_perfect(self, tool_monitor):
        """Test health score calculation for perfect metrics."""
        metrics = ToolMetrics(
            tool_name="test",
            total_executions=100,
            successful_executions=100,
            failed_executions=0,
            error_rate=0.0,
            consecutive_failures=0,
            avg_confidence=0.9,
            p95_latency_ms=50.0
        )
        
        score = tool_monitor._calculate_health_score(metrics)
        
        assert score > 0.8
    
    def test_calculate_health_score_poor(self, tool_monitor):
        """Test health score calculation for poor metrics."""
        metrics = ToolMetrics(
            tool_name="test",
            total_executions=100,
            successful_executions=50,
            failed_executions=50,
            error_rate=0.5,
            consecutive_failures=10,
            avg_confidence=0.3,
            p95_latency_ms=2000.0
        )
        
        score = tool_monitor._calculate_health_score(metrics)
        
        assert score < 0.5
    
    def test_get_health_status_healthy(self, tool_monitor):
        """Test getting healthy status."""
        # Record successful executions
        for i in range(10):
            tool_monitor.record_execution(
                tool_name="test_tool",
                success=True,
                latency_ms=50.0,
                energy_mj=5.0,
                confidence=0.9
            )
        
        status = tool_monitor.get_health_status()
        
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def test_get_tool_rankings(self, tool_monitor):
        """Test getting tool rankings."""
        # Add metrics for multiple tools
        for tool_name in ["tool1", "tool2", "tool3"]:
            for i in range(10):
                tool_monitor.record_execution(
                    tool_name=tool_name,
                    success=True,
                    latency_ms=100.0 + ord(tool_name[-1]),
                    energy_mj=10.0,
                    confidence=0.8
                )
        
        rankings = tool_monitor.get_tool_rankings()
        
        assert len(rankings) == 3
        assert all(isinstance(r, tuple) for r in rankings)
        assert all(len(r) == 2 for r in rankings)
    
    def test_get_metrics_summary(self, tool_monitor):
        """Test getting metrics summary."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.8
        )
        
        summary = tool_monitor.get_metrics_summary()
        
        assert 'uptime_seconds' in summary
        assert 'system' in summary
        assert 'tools' in summary
        assert 'health_status' in summary
    
    def test_get_time_series(self, tool_monitor):
        """Test getting time series data."""
        # Record some data
        for i in range(10):
            tool_monitor.record_execution(
                tool_name="test_tool",
                success=True,
                latency_ms=100.0 + i,
                energy_mj=10.0,
                confidence=0.8
            )
        
        time_series = tool_monitor.get_time_series("test_tool_latency")
        
        assert 'data' in time_series
        assert 'timestamps' in time_series
        assert 'stats' in time_series
    
    def test_diagnose_tool(self, tool_monitor):
        """Test diagnosing tool performance."""
        # Record executions
        for i in range(20):
            tool_monitor.record_execution(
                tool_name="test_tool",
                success=i % 5 != 0,  # Fail every 5th
                latency_ms=100.0 + i * 5,
                energy_mj=10.0,
                confidence=0.7
            )
        
        diagnosis = tool_monitor.diagnose_tool("test_tool")
        
        assert 'tool' in diagnosis
        assert 'metrics' in diagnosis
        assert 'health_score' in diagnosis
        assert 'issues' in diagnosis
        assert 'recommendations' in diagnosis
    
    def test_diagnose_nonexistent_tool(self, tool_monitor):
        """Test diagnosing nonexistent tool."""
        diagnosis = tool_monitor.diagnose_tool("nonexistent")
        
        assert 'error' in diagnosis
    
    def test_export_metrics_json(self, tool_monitor, temp_dir):
        """Test exporting metrics as JSON."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.8
        )
        
        export_path = Path(temp_dir) / "metrics.json"
        tool_monitor.export_metrics(str(export_path), format='json')
        
        assert export_path.exists()
    
    def test_export_metrics_csv(self, tool_monitor, temp_dir):
        """Test exporting metrics as CSV."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.8
        )
        
        export_path = Path(temp_dir) / "metrics.csv"
        tool_monitor.export_metrics(str(export_path), format='csv')
        
        assert export_path.exists()
    
    def test_reset_metrics_specific_tool(self, tool_monitor):
        """Test resetting metrics for specific tool."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.8
        )
        
        tool_monitor.reset_metrics("test_tool")
        
        metrics = tool_monitor.tool_metrics["test_tool"]
        assert metrics.total_executions == 0
    
    def test_reset_all_metrics(self, tool_monitor):
        """Test resetting all metrics."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            energy_mj=10.0,
            confidence=0.8
        )
        
        tool_monitor.reset_metrics()
        
        assert len(tool_monitor.tool_metrics) == 0
    
    def test_get_statistics(self, tool_monitor):
        """Test getting statistics."""
        stats = tool_monitor.get_statistics()
        
        assert 'monitoring_uptime' in stats
        assert 'total_tools_monitored' in stats
        assert 'total_executions' in stats


class TestAlertManagement:
    """Test alert creation and management."""
    
    def test_alert_creation(self, tool_monitor):
        """Test creating alerts."""
        initial_count = len(tool_monitor.alerts)
        
        tool_monitor._create_alert(
            AlertSeverity.WARNING,
            MetricType.LATENCY,
            "Test alert",
            tool_name="test_tool",
            value=500.0
        )
        
        assert len(tool_monitor.alerts) > initial_count
    
    def test_alert_cooldown(self, tool_monitor):
        """Test alert cooldown period."""
        # Create first alert
        tool_monitor._create_alert(
            AlertSeverity.WARNING,
            MetricType.LATENCY,
            "Test alert",
            tool_name="test_tool"
        )
        
        initial_count = len(tool_monitor.alerts)
        
        # Try to create same alert immediately
        tool_monitor._create_alert(
            AlertSeverity.WARNING,
            MetricType.LATENCY,
            "Test alert",
            tool_name="test_tool"
        )
        
        # Should be blocked by cooldown
        assert len(tool_monitor.alerts) == initial_count
    
    def test_register_alert_callback(self, tool_monitor):
        """Test registering alert callback."""
        callback_called = []
        
        def callback(alert):
            callback_called.append(alert)
        
        tool_monitor.register_alert_callback(callback)
        
        tool_monitor._create_alert(
            AlertSeverity.ERROR,
            MetricType.ERROR_RATE,
            "Test alert"
        )
        
        # Callback should be called
        assert len(callback_called) > 0


class TestThreadSafety:
    """Test thread safety."""
    
    def test_concurrent_recordings(self, tool_monitor):
        """Test concurrent execution recordings."""
        def record_executions(thread_id):
            for i in range(10):
                tool_monitor.record_execution(
                    tool_name=f"tool_{thread_id}",
                    success=True,
                    latency_ms=100.0,
                    energy_mj=10.0,
                    confidence=0.8
                )
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_executions, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have recorded all executions
        total = sum(m.total_executions for m in tool_monitor.tool_metrics.values())
        assert total == 50


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_executions(self, tool_monitor):
        """Test with zero executions."""
        summary = tool_monitor.get_metrics_summary()
        
        assert summary['system']['total_requests'] == 0
    
    def test_very_high_latency(self, tool_monitor):
        """Test with very high latency."""
        tool_monitor.record_execution(
            tool_name="test_tool",
            success=True,
            latency_ms=1000000.0,
            energy_mj=10.0,
            confidence=0.8
        )
        
        metrics = tool_monitor.tool_metrics["test_tool"]
        assert metrics.max_latency_ms == 1000000.0
    
    def test_negative_values(self, tool_monitor):
        """Test handling negative values."""
        # Should handle gracefully
        try:
            tool_monitor.record_execution(
                tool_name="test_tool",
                success=True,
                latency_ms=-100.0,
                energy_mj=-10.0,
                confidence=-0.5
            )
        except:
            pass  # Acceptable to fail


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])