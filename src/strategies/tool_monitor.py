"""
Tool Monitor for Tool Selection System

Comprehensive monitoring, alerting, and diagnostics for tool selection performance,
health, and resource usage with real-time tracking and trend analysis.
"""

import json
import logging
import pickle
import queue
import threading
import time
import traceback
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    RESOURCE_USAGE = "resource_usage"
    QUALITY = "quality"
    CONFIDENCE = "confidence"
    DRIFT = "drift"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class HealthStatus(Enum):
    """System health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ToolMetrics:
    """Metrics for a single tool"""

    tool_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_time_ms: float = 0.0
    total_energy_mj: float = 0.0
    avg_confidence: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    health_score: float = 1.0
    last_execution: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0

    def update_latency_percentiles(self, latencies: List[float]):
        """Update latency percentiles"""
        if latencies:
            self.p50_latency_ms = np.percentile(latencies, 50)
            self.p95_latency_ms = np.percentile(latencies, 95)
            self.p99_latency_ms = np.percentile(latencies, 99)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_name": self.tool_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "avg_time_ms": self.total_time_ms / max(1, self.total_executions),
            "avg_energy_mj": self.total_energy_mj / max(1, self.total_executions),
            "avg_confidence": self.avg_confidence,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "error_rate": self.error_rate,
            "success_rate": self.success_rate,
            "health_score": self.health_score,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class SystemMetrics:
    """System-wide metrics"""

    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_time_ms: float = 0.0
    total_energy_mj: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    active_threads: int = 0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    drift_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "avg_time_ms": self.total_time_ms / max(1, self.total_requests),
            "avg_energy_mj": self.total_energy_mj / max(1, self.total_requests),
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "disk_io_mb": self.disk_io_mb,
            "network_io_mb": self.network_io_mb,
            "active_threads": self.active_threads,
            "queue_depth": self.queue_depth,
            "cache_hit_rate": self.cache_hit_rate,
            "drift_detected": self.drift_detected,
        }


@dataclass
class Alert:
    """System alert"""

    severity: AlertSeverity
    metric_type: MetricType
    message: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "severity": self.severity.name,
            "metric_type": self.metric_type.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


class TimeSeriesBuffer:
    """Buffer for time series data"""

    def __init__(self, window_size: int = 1000, time_window: Optional[float] = None):
        """
        Initialize buffer

        Args:
            window_size: Maximum number of points
            time_window: Optional time window in seconds
        """
        self.window_size = window_size
        self.time_window = time_window
        self.data = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def add(self, value: float, timestamp: Optional[float] = None):
        """Add value to buffer"""
        if timestamp is None:
            timestamp = time.time()

        self.data.append(value)
        self.timestamps.append(timestamp)

        # Remove old data if time window specified
        if self.time_window:
            cutoff = timestamp - self.time_window
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
                self.data.popleft()

    def get_stats(self) -> Dict[str, float]:
        """Get statistics from buffer"""
        if not self.data:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        data_array = np.array(self.data)

        return {
            "mean": np.mean(data_array),
            "std": np.std(data_array),
            "min": np.min(data_array),
            "max": np.max(data_array),
            "count": len(data_array),
            "rate": self._calculate_rate(),
        }

    def _calculate_rate(self) -> float:
        """Calculate rate per second"""
        if len(self.timestamps) < 2:
            return 0.0

        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return len(self.data) / time_span

        return 0.0

    def get_trend(self) -> str:
        """Get trend direction"""
        if len(self.data) < 10:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(self.data))
        y = np.array(self.data)

        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"


class AnomalyDetector:
    """Detect anomalies in metrics"""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.baseline = defaultdict(lambda: TimeSeriesBuffer(window_size))

    def update(self, metric_name: str, value: float) -> bool:
        """
        Update and check for anomaly

        Returns:
            True if anomaly detected
        """
        buffer = self.baseline[metric_name]

        # Check for anomaly before adding
        if len(buffer.data) >= 20:
            stats = buffer.get_stats()
            z_score = abs((value - stats["mean"]) / max(stats["std"], 0.01))

            if z_score > self.z_threshold:
                return True

        # Add to baseline
        buffer.add(value)

        return False

    def get_anomaly_score(self, metric_name: str, value: float) -> float:
        """Get anomaly score for value"""
        buffer = self.baseline[metric_name]

        if len(buffer.data) < 20:
            return 0.0

        stats = buffer.get_stats()
        z_score = abs((value - stats["mean"]) / max(stats["std"], 0.01))

        return min(1.0, z_score / self.z_threshold)


class ToolMonitor:
    """
    Main monitoring system for tool selection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Configuration
        self.monitoring_interval = config.get("monitoring_interval", 1.0)
        self.alert_cooldown = config.get("alert_cooldown", 300)  # 5 minutes
        self.history_size = config.get("history_size", 10000)

        # Tool metrics
        self.tool_metrics = defaultdict(lambda: ToolMetrics(tool_name="unknown"))
        self.tool_latencies = defaultdict(lambda: deque(maxlen=1000))

        # System metrics
        self.system_metrics = SystemMetrics()

        # Time series buffers
        self.time_series = defaultdict(lambda: TimeSeriesBuffer(1000))

        # Anomaly detection
        self.anomaly_detector = AnomalyDetector()

        # Alert management
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        self.last_alert_time = defaultdict(float)

        # Execution history
        self.execution_history = deque(maxlen=self.history_size)

        # Resource monitoring
        self.process = psutil.Process()

        # Monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        # Alert thresholds
        self.thresholds = {
            "error_rate": config.get("error_rate_threshold", 0.1),
            "latency_p95": config.get("latency_p95_threshold", 1000),
            "cpu_usage": config.get("cpu_usage_threshold", 80),
            "memory_usage": config.get("memory_usage_threshold", 80),
            "consecutive_failures": config.get("consecutive_failures_threshold", 5),
        }

        # Statistics
        self.start_time = time.time()

        logger.info("Tool Monitor initialized")

    def record_execution(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        energy_mj: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record tool execution"""

        timestamp = time.time()

        # Update tool metrics
        metrics = self.tool_metrics[tool_name]
        metrics.tool_name = tool_name
        metrics.total_executions += 1

        if success:
            metrics.successful_executions += 1
            metrics.consecutive_failures = 0
        else:
            metrics.failed_executions += 1
            metrics.consecutive_failures += 1
            metrics.last_failure = timestamp

        metrics.total_time_ms += latency_ms
        metrics.total_energy_mj += energy_mj
        metrics.last_execution = timestamp

        # Update confidence with exponential moving average
        alpha = 0.1
        metrics.avg_confidence = (
            1 - alpha
        ) * metrics.avg_confidence + alpha * confidence

        # Update latency stats
        metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
        metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
        self.tool_latencies[tool_name].append(latency_ms)

        # Update percentiles
        metrics.update_latency_percentiles(list(self.tool_latencies[tool_name]))

        # Update rates
        metrics.error_rate = metrics.failed_executions / metrics.total_executions
        metrics.success_rate = metrics.successful_executions / metrics.total_executions

        # Update health score
        metrics.health_score = self._calculate_health_score(metrics)

        # Update system metrics
        self.system_metrics.total_requests += 1
        if success:
            self.system_metrics.total_successes += 1
        else:
            self.system_metrics.total_failures += 1

        self.system_metrics.total_time_ms += latency_ms
        self.system_metrics.total_energy_mj += energy_mj

        # Add to time series
        self.time_series[f"{tool_name}_latency"].add(latency_ms, timestamp)
        self.time_series[f"{tool_name}_confidence"].add(confidence, timestamp)
        self.time_series["total_latency"].add(latency_ms, timestamp)

        # Check for anomalies
        if self.anomaly_detector.update(f"{tool_name}_latency", latency_ms):
            self._create_alert(
                AlertSeverity.WARNING,
                MetricType.LATENCY,
                f"Anomalous latency detected for {tool_name}: {latency_ms:.1f}ms",
                tool_name=tool_name,
                value=latency_ms,
            )

        # Add to history
        self.execution_history.append(
            {
                "timestamp": timestamp,
                "tool": tool_name,
                "success": success,
                "latency_ms": latency_ms,
                "energy_mj": energy_mj,
                "confidence": confidence,
                "metadata": metadata,
            }
        )

        # Check thresholds
        self._check_thresholds(tool_name, metrics)

    def _calculate_health_score(self, metrics: ToolMetrics) -> float:
        """Calculate health score for tool"""

        score = 1.0

        # Penalize high error rate
        score -= metrics.error_rate * 0.4

        # Penalize consecutive failures
        score -= min(0.3, metrics.consecutive_failures * 0.06)

        # Penalize high latency
        if metrics.p95_latency_ms > self.thresholds["latency_p95"]:
            score -= 0.1

        # Penalize low confidence
        if metrics.avg_confidence < 0.5:
            score -= 0.2

        # Penalize staleness
        if metrics.last_execution:
            time_since_execution = time.time() - metrics.last_execution
            if time_since_execution > 600:  # 10 minutes
                score -= min(0.1, time_since_execution / 3600)

        return max(0.0, min(1.0, score))

    def _check_thresholds(self, tool_name: str, metrics: ToolMetrics):
        """Check metrics against thresholds"""

        # Error rate threshold
        if metrics.error_rate > self.thresholds["error_rate"]:
            self._create_alert(
                AlertSeverity.ERROR,
                MetricType.ERROR_RATE,
                f"High error rate for {tool_name}: {metrics.error_rate:.2%}",
                tool_name=tool_name,
                value=metrics.error_rate,
                threshold=self.thresholds["error_rate"],
            )

        # Latency threshold
        if metrics.p95_latency_ms > self.thresholds["latency_p95"]:
            self._create_alert(
                AlertSeverity.WARNING,
                MetricType.LATENCY,
                f"High P95 latency for {tool_name}: {metrics.p95_latency_ms:.1f}ms",
                tool_name=tool_name,
                value=metrics.p95_latency_ms,
                threshold=self.thresholds["latency_p95"],
            )

        # Consecutive failures
        if metrics.consecutive_failures >= self.thresholds["consecutive_failures"]:
            self._create_alert(
                AlertSeverity.CRITICAL,
                MetricType.ERROR_RATE,
                f"Tool {tool_name} has failed {metrics.consecutive_failures} times consecutively",
                tool_name=tool_name,
                value=metrics.consecutive_failures,
                threshold=self.thresholds["consecutive_failures"],
            )

    def _create_alert(
        self,
        severity: AlertSeverity,
        metric_type: MetricType,
        message: str,
        tool_name: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ):
        """Create and handle alert"""

        # Check cooldown
        alert_key = f"{tool_name}_{metric_type.value}"
        if time.time() - self.last_alert_time[alert_key] < self.alert_cooldown:
            return

        alert = Alert(
            severity=severity,
            metric_type=metric_type,
            message=message,
            tool_name=tool_name,
            value=value,
            threshold=threshold,
        )

        self.alerts.append(alert)
        self.last_alert_time[alert_key] = time.time()

        # Log alert
        if severity == AlertSeverity.CRITICAL:
            logger.critical(message)
        elif severity == AlertSeverity.ERROR:
            logger.error(message)
        elif severity == AlertSeverity.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _monitoring_loop(self):
        """Background monitoring loop"""

        while self.monitoring:
            try:
                # Update resource metrics
                self._update_resource_metrics()

                # Check system health
                health = self.get_health_status()
                if health == HealthStatus.CRITICAL:
                    self._create_alert(
                        AlertSeverity.CRITICAL,
                        MetricType.RESOURCE_USAGE,
                        "System health is critical",
                    )
                elif health == HealthStatus.UNHEALTHY:
                    self._create_alert(
                        AlertSeverity.ERROR,
                        MetricType.RESOURCE_USAGE,
                        "System health is unhealthy",
                    )

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _update_resource_metrics(self):
        """Update system resource metrics"""

        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.system_metrics.cpu_usage_percent = cpu_percent
            self.time_series["cpu_usage"].add(cpu_percent)

            # Memory usage
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            self.system_metrics.memory_usage_percent = memory_percent
            self.time_series["memory_usage"].add(memory_percent)

            # Thread count
            self.system_metrics.active_threads = threading.active_count()

            # Check resource thresholds
            if cpu_percent > self.thresholds["cpu_usage"]:
                self._create_alert(
                    AlertSeverity.WARNING,
                    MetricType.RESOURCE_USAGE,
                    f"High CPU usage: {cpu_percent:.1f}%",
                    value=cpu_percent,
                    threshold=self.thresholds["cpu_usage"],
                )

            if memory_percent > self.thresholds["memory_usage"]:
                self._create_alert(
                    AlertSeverity.WARNING,
                    MetricType.RESOURCE_USAGE,
                    f"High memory usage: {memory_percent:.1f}%",
                    value=memory_percent,
                    threshold=self.thresholds["memory_usage"],
                )

        except Exception as e:
            logger.debug(f"Resource monitoring error: {e}")

    def get_health_status(self) -> HealthStatus:
        """Get overall system health status"""

        # Calculate overall health metrics
        if not self.tool_metrics:
            return HealthStatus.HEALTHY

        # Average tool health
        avg_health = np.mean([m.health_score for m in self.tool_metrics.values()])

        # System error rate
        system_error_rate = self.system_metrics.total_failures / max(
            1, self.system_metrics.total_requests
        )

        # Resource pressure
        resource_pressure = max(
            self.system_metrics.cpu_usage_percent / 100,
            self.system_metrics.memory_usage_percent / 100,
        )

        # Determine status
        if avg_health < 0.3 or system_error_rate > 0.3 or resource_pressure > 0.9:
            return HealthStatus.CRITICAL
        elif avg_health < 0.5 or system_error_rate > 0.2 or resource_pressure > 0.8:
            return HealthStatus.UNHEALTHY
        elif avg_health < 0.7 or system_error_rate > 0.1 or resource_pressure > 0.7:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_tool_rankings(self) -> List[Tuple[str, float]]:
        """Get tools ranked by performance"""

        rankings = []

        for tool_name, metrics in self.tool_metrics.items():
            # Composite score based on multiple factors
            score = (
                metrics.success_rate * 0.3
                + (1 - min(1, metrics.p50_latency_ms / 1000)) * 0.2
                + metrics.avg_confidence * 0.2
                + metrics.health_score * 0.3
            )

            rankings.append((tool_name, score))

        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""

        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": uptime,
            "system": self.system_metrics.to_dict(),
            "tools": {
                name: metrics.to_dict() for name, metrics in self.tool_metrics.items()
            },
            "health_status": self.get_health_status().value,
            "tool_rankings": self.get_tool_rankings(),
            "active_alerts": len(
                [a for a in self.alerts if time.time() - a.timestamp < 3600]
            ),
            "recent_alerts": [a.to_dict() for a in list(self.alerts)[-10:]],
        }

    def get_time_series(
        self, metric_name: str, time_range: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get time series data for metric"""

        if metric_name not in self.time_series:
            return {"error": "Metric not found"}

        buffer = self.time_series[metric_name]

        if time_range:
            # Filter by time range
            cutoff = time.time() - time_range
            data = []
            timestamps = []

            for value, timestamp in zip(buffer.data, buffer.timestamps):
                if timestamp >= cutoff:
                    data.append(value)
                    timestamps.append(timestamp)
        else:
            data = list(buffer.data)
            timestamps = list(buffer.timestamps)

        return {
            "metric": metric_name,
            "data": data,
            "timestamps": timestamps,
            "stats": buffer.get_stats(),
            "trend": buffer.get_trend(),
        }

    def diagnose_tool(self, tool_name: str) -> Dict[str, Any]:
        """Diagnose specific tool performance"""

        if tool_name not in self.tool_metrics:
            return {"error": "Tool not found"}

        metrics = self.tool_metrics[tool_name]

        # Recent executions
        recent_executions = [
            ex for ex in self.execution_history if ex["tool"] == tool_name
        ][-100:]

        # Calculate trends
        if recent_executions:
            recent_latencies = [ex["latency_ms"] for ex in recent_executions]
            recent_successes = [ex["success"] for ex in recent_executions]

            latency_trend = self._calculate_trend(recent_latencies)
            success_trend = self._calculate_trend(
                [1 if s else 0 for s in recent_successes]
            )
        else:
            latency_trend = "no_data"
            success_trend = "no_data"

        # Anomaly scores
        latency_anomaly = self.anomaly_detector.get_anomaly_score(
            f"{tool_name}_latency", metrics.p50_latency_ms
        )

        # Diagnosis
        issues = []
        recommendations = []

        if metrics.error_rate > 0.1:
            issues.append(f"High error rate: {metrics.error_rate:.2%}")
            recommendations.append(
                "Investigate error causes and improve error handling"
            )

        if metrics.consecutive_failures > 3:
            issues.append(f"Consecutive failures: {metrics.consecutive_failures}")
            recommendations.append("Check tool health and consider restart")

        if metrics.p95_latency_ms > 1000:
            issues.append(f"High P95 latency: {metrics.p95_latency_ms:.1f}ms")
            recommendations.append("Optimize performance or increase resources")

        if latency_trend == "increasing":
            issues.append("Latency is increasing over time")
            recommendations.append("Monitor for resource exhaustion or memory leaks")

        if metrics.health_score < 0.5:
            issues.append(f"Low health score: {metrics.health_score:.2f}")
            recommendations.append("Consider replacing or repairing tool")

        return {
            "tool": tool_name,
            "metrics": metrics.to_dict(),
            "health_score": metrics.health_score,
            "latency_trend": latency_trend,
            "success_trend": success_trend,
            "anomaly_score": latency_anomaly,
            "issues": issues,
            "recommendations": recommendations,
            "recent_performance": {
                "executions": len(recent_executions),
                "avg_latency": np.mean(recent_latencies) if recent_latencies else 0,
                "success_rate": np.mean(recent_successes) if recent_successes else 0,
            },
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""

        if len(values) < 10:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Normalize by mean to get relative change
        mean_val = np.mean(values)
        if mean_val > 0:
            relative_slope = slope / mean_val

            if abs(relative_slope) < 0.01:
                return "stable"
            elif relative_slope > 0:
                return "increasing"
            else:
                return "decreasing"

        return "stable"

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)

    def export_metrics(self, path: str, format: str = "json"):
        """Export metrics to file"""

        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.get_metrics_summary()

        if format == "json":
            with open(export_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
        elif format == "csv":
            # Export tool metrics as CSV
            import csv

            with open(export_path, "w", newline="") as f:
                if self.tool_metrics:
                    fieldnames = list(
                        next(iter(self.tool_metrics.values())).to_dict().keys()
                    )
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for metrics in self.tool_metrics.values():
                        writer.writerow(metrics.to_dict())

        logger.info(f"Metrics exported to {export_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""

        return {
            "monitoring_uptime": time.time() - self.start_time,
            "total_tools_monitored": len(self.tool_metrics),
            "total_executions": self.system_metrics.total_requests,
            "total_alerts": len(self.alerts),
            "health_status": self.get_health_status().value,
            "buffer_sizes": {
                name: len(buffer.data) for name, buffer in self.time_series.items()
            },
        }

    def reset_metrics(self, tool_name: Optional[str] = None):
        """Reset metrics for tool or all tools"""

        if tool_name:
            if tool_name in self.tool_metrics:
                self.tool_metrics[tool_name] = ToolMetrics(tool_name=tool_name)
                self.tool_latencies[tool_name].clear()
                logger.info(f"Reset metrics for {tool_name}")
        else:
            # Reset all
            self.tool_metrics.clear()
            self.tool_latencies.clear()
            self.system_metrics = SystemMetrics()
            self.execution_history.clear()
            logger.info("Reset all metrics")

    def shutdown(self):
        """Shutdown monitoring"""

        logger.info("Shutting down Tool Monitor")

        self.monitoring = False

        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

        # Export final metrics
        self.export_metrics("./final_metrics.json")

        logger.info("Tool Monitor shutdown complete")
