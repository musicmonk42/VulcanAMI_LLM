# ============================================================
# VULCAN-AGI Orchestrator - Metrics Module
# Comprehensive metrics collection and monitoring
# FULLY FIXED VERSION - Enhanced with proper bounds, cleanup, and no circular dependencies
# FIXED: Converted long time.sleep(self.cleanup_interval) to interruptible wait.
# ============================================================

import logging
import threading
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# METRIC TYPES AND ENUMS
# ============================================================


class MetricType(Enum):
    """Types of metrics that can be collected"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMESERIES = "timeseries"
    RATE = "rate"


class AggregationType(Enum):
    """Types of aggregations for metrics"""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    COUNT = "count"


# ============================================================
# ENHANCED METRICS COLLECTOR
# ============================================================


class EnhancedMetricsCollector:
    """
    Comprehensive metrics collection and monitoring with bounded memory usage.

    Features:
    - Thread-safe operations
    - Bounded memory for all data structures
    - No circular dependencies
    - Comprehensive metric types (counters, gauges, histograms, timeseries)
    - Health score computation
    - Automatic cleanup of old data
    """

    def __init__(
        self,
        max_histogram_size: int = 10000,
        max_timeseries_size: int = 1000,
        cleanup_interval: int = 300,
    ):
        """
        Initialize metrics collector

        Args:
            max_histogram_size: Maximum number of values to keep in histograms
            max_timeseries_size: Maximum number of points to keep in timeseries
            cleanup_interval: Interval in seconds between cleanup operations
        """
        # Counters: monotonically increasing values
        self.counters = defaultdict(int)

        # Gauges: point-in-time values
        self.gauges = defaultdict(float)

        # Histograms: bounded lists of values for statistical analysis
        self.histograms = defaultdict(lambda: deque(maxlen=max_histogram_size))

        # Timeseries: bounded time-value pairs
        self.timeseries = defaultdict(lambda: deque(maxlen=max_timeseries_size))

        # Aggregates: computed values and metadata
        self.aggregates = defaultdict(dict)

        # Thread safety
        self._lock = threading.RLock()

        # Cleanup configuration
        self.max_histogram_size = max_histogram_size
        self.max_timeseries_size = max_timeseries_size
        self.cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

        # Start time for uptime tracking
        self._start_time = time.time()

        # Cleanup thread
        self._shutdown_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="MetricsCleanup"
        )
        self._cleanup_thread.start()

        logger.info("EnhancedMetricsCollector initialized")

    def record_step(self, duration: float, result: Dict[str, Any]):
        """
        Record step metrics from orchestrator execution

        Args:
            duration: Step duration in seconds
            result: Result dictionary from step execution
        """
        with self._lock:
            # Record basic step metrics
            self.counters["steps_total"] += 1
            self.histograms["step_duration_ms"].append(duration * 1000)
            self.timeseries["step_duration_ms_time"].append(
                (time.time(), duration * 1000)
            )

            # FIXED: No circular import - handle modality without importing config
            modality = result.get("modality")
            if modality is not None:
                # Handle both enum and string modalities
                if hasattr(modality, "value"):
                    modality_str = modality.value
                else:
                    modality_str = str(modality)

                self.counters[f"modality_{modality_str}_count"] += 1

            # Record action type
            action = result.get("action", {})
            if isinstance(action, dict):
                action_type = action.get("type", "unknown")
            else:
                action_type = str(action)

            self.counters[f"action_{action_type}_count"] += 1

            # Record learning metrics
            if "loss" in result:
                loss = result["loss"]
                self.histograms["learning_loss"].append(loss)
                self.timeseries["loss_over_time"].append((time.time(), loss))
                self.gauges["current_loss"] = loss

            # Record uncertainty
            if "uncertainty" in result:
                uncertainty = result["uncertainty"]
                self.histograms["uncertainty"].append(uncertainty)
                self.timeseries["uncertainty_over_time"].append(
                    (time.time(), uncertainty)
                )
                self.gauges["current_uncertainty"] = uncertainty

            # Record resource usage
            if "resource_usage" in result:
                for resource, value in result["resource_usage"].items():
                    metric_name = f"resource_{resource}"
                    self.histograms[metric_name].append(value)
                    self.timeseries[f"{metric_name}_time"].append((time.time(), value))
                    self.gauges[f"current_{metric_name}"] = value

            # Record success/failure
            is_success = (
                result.get("success", False) or result.get("status") == "completed"
            )
            if is_success:
                self.counters["successful_actions"] += 1
            else:
                self.counters["failed_actions"] += 1

            # Record reward if present
            if "reward" in result:
                reward = result["reward"]
                self.histograms["reward"].append(reward)
                self.timeseries["reward_over_time"].append((time.time(), reward))
                self.gauges["current_reward"] = reward

            # Periodic cleanup check
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self._perform_cleanup()

    def update_gauge(self, name: str, value: float):
        """
        Update gauge metric

        Args:
            name: Gauge name
            value: Current value
        """
        with self._lock:
            self.gauges[name] = value
            self.timeseries[f"{name}_history"].append((time.time(), value))

    def increment_counter(self, name: str, value: int = 1):
        """
        Increment counter metric

        Args:
            name: Counter name
            value: Amount to increment by
        """
        with self._lock:
            self.counters[name] += value

    def decrement_counter(self, name: str, value: int = 1):
        """
        Decrement counter metric

        Args:
            name: Counter name
            value: Amount to decrement by
        """
        with self._lock:
            self.counters[name] -= value

    def record_histogram(self, name: str, value: float):
        """
        Record value in histogram

        Args:
            name: Histogram name
            value: Value to record
        """
        with self._lock:
            self.histograms[name].append(value)

    def record_timeseries(
        self, name: str, value: float, timestamp: Optional[float] = None
    ):
        """
        Record time series point

        Args:
            name: Timeseries name
            value: Value to record
            timestamp: Timestamp (defaults to current time)
        """
        timestamp = timestamp or time.time()
        with self._lock:
            self.timeseries[name].append((timestamp, value))

    def record_event(self, event_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record discrete event

        Args:
            event_type: Type of event
            metadata: Optional event metadata
        """
        with self._lock:
            self.counters[f"event_{event_type}"] += 1

            if metadata:
                # Store only the most recent metadata for each event type
                self.aggregates[f"event_{event_type}_metadata"] = {
                    "timestamp": time.time(),
                    "data": metadata,
                }

    def get_counter(self, name: str) -> int:
        """
        Get counter value

        Args:
            name: Counter name

        Returns:
            Counter value
        """
        with self._lock:
            return self.counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """
        Get gauge value

        Args:
            name: Gauge name

        Returns:
            Gauge value
        """
        with self._lock:
            return self.gauges.get(name, 0.0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary

        Returns:
            Dictionary with all metrics summaries
        """
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms_summary": {},
                "rates": {},
                "health_score": self._compute_health_score(),
                "uptime_seconds": time.time() - self._start_time,
                "timestamp": time.time(),
            }

            # Compute histogram summaries
            for name, values in self.histograms.items():
                if values:
                    summary["histograms_summary"][name] = self._compute_histogram_stats(
                        list(values)
                    )

            # Compute rates
            total_actions = self.counters.get(
                "successful_actions", 0
            ) + self.counters.get("failed_actions", 0)
            if total_actions > 0:
                summary["rates"]["success_rate"] = (
                    self.counters["successful_actions"] / total_actions
                )
                summary["rates"]["failure_rate"] = (
                    self.counters["failed_actions"] / total_actions
                )
            else:
                summary["rates"]["success_rate"] = 0.0
                summary["rates"]["failure_rate"] = 0.0

            # Compute steps per second
            uptime = time.time() - self._start_time
            if uptime > 0:
                summary["rates"]["steps_per_second"] = (
                    self.counters.get("steps_total", 0) / uptime
                )
            else:
                summary["rates"]["steps_per_second"] = 0.0

            return summary

    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific histogram

        Args:
            name: Histogram name

        Returns:
            Dictionary with histogram statistics or None if not found
        """
        with self._lock:
            if name not in self.histograms or not self.histograms[name]:
                return None

            return self._compute_histogram_stats(list(self.histograms[name]))

    def _compute_histogram_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Compute statistics for a list of values

        Args:
            values: List of numeric values

        Returns:
            Dictionary with statistical measures
        """
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        stats = {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": float(np.mean(values)),
            "median": sorted_values[n // 2],
            "std": float(np.std(values)),
        }

        # Add percentiles if enough data
        if n > 20:
            stats["p50"] = sorted_values[int(n * 0.50)]
            stats["p90"] = sorted_values[int(n * 0.90)]
            stats["p95"] = sorted_values[int(n * 0.95)]

        if n > 100:
            stats["p99"] = sorted_values[int(n * 0.99)]

        return stats

    def get_timeseries(
        self, metric_name: str, last_n: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        Get time series data for metric

        Args:
            metric_name: Name of the timeseries metric
            last_n: Number of most recent points to return (None = all)

        Returns:
            List of (timestamp, value) tuples
        """
        with self._lock:
            if metric_name not in self.timeseries:
                return []

            series = list(self.timeseries[metric_name])

            if last_n is not None:
                return series[-last_n:]

            return series

    def get_timeseries_window(
        self,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        Get time series data within a time window

        Args:
            metric_name: Name of the timeseries metric
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            List of (timestamp, value) tuples within the window
        """
        with self._lock:
            if metric_name not in self.timeseries:
                return []

            series = list(self.timeseries[metric_name])

            if start_time is None and end_time is None:
                return series

            filtered = []
            for timestamp, value in series:
                if start_time is not None and timestamp < start_time:
                    continue
                if end_time is not None and timestamp > end_time:
                    continue
                filtered.append((timestamp, value))

            return filtered

    def _compute_health_score(self) -> float:
        """
        Compute overall system health score (0.0 to 1.0)

        Returns:
            Health score where 1.0 is perfect health
        """
        factors = []
        weights = []

        # Success rate factor (weight: 0.4)
        total_actions = self.counters.get("successful_actions", 0) + self.counters.get(
            "failed_actions", 0
        )
        if total_actions > 0:
            success_rate = self.counters["successful_actions"] / total_actions
            factors.append(success_rate)
            weights.append(0.4)

        # Uncertainty factor (weight: 0.3)
        if "current_uncertainty" in self.gauges:
            uncertainty = self.gauges["current_uncertainty"]
            # Lower uncertainty is better
            uncertainty_score = max(0.0, 1.0 - uncertainty)
            factors.append(uncertainty_score)
            weights.append(0.3)

        # Resource efficiency factor (weight: 0.3)
        if (
            "step_duration_ms" in self.histograms
            and len(self.histograms["step_duration_ms"]) > 0
        ):
            recent_durations = list(self.histograms["step_duration_ms"])[-100:]
            avg_time = np.mean(recent_durations)
            # Assume target is 100ms, score decreases as we deviate
            if avg_time > 0:
                efficiency = min(1.0, 100.0 / avg_time)
            else:
                efficiency = 0.0
            factors.append(efficiency)
            weights.append(0.3)

        # Compute weighted average
        if factors:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(f * w for f, w in zip(factors, weights))
                return weighted_sum / total_weight

        # Default to neutral score if no data
        return 0.5

    def _perform_cleanup(self):
        """Perform cleanup of old data to maintain memory bounds"""
        with self._lock:
            current_time = time.time()

            # Clean up old timeseries data (older than 1 hour)
            cutoff_time = current_time - 3600

            for name, series in list(self.timeseries.items()):
                # Remove old entries
                while series and series[0][0] < cutoff_time:
                    series.popleft()

                # If series is empty, remove it
                if not series:
                    del self.timeseries[name]

            # Clean up old aggregate metadata (older than 1 hour)
            for key in list(self.aggregates.keys()):
                if isinstance(self.aggregates[key], dict):
                    timestamp = self.aggregates[key].get("timestamp", 0)
                    if timestamp < cutoff_time:
                        del self.aggregates[key]

            self._last_cleanup = current_time

            logger.debug("Metrics cleanup completed")

    def _cleanup_loop(self):
        """
        Background cleanup loop

        FIXED: Converted blocking time.sleep(self.cleanup_interval) to
        interruptible self._shutdown_event.wait(self.cleanup_interval).
        """
        # FIXED: Use interruptible wait
        while not self._shutdown_event.is_set():
            try:
                # If shutdown is signaled, break immediately
                if self._shutdown_event.wait(self.cleanup_interval):
                    break

                self._perform_cleanup()
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}", exc_info=True)

        logger.info("Metrics cleanup thread stopped")

    def reset_counters(self):
        """Reset all counters to zero"""
        with self._lock:
            self.counters.clear()
            logger.info("Counters reset")

    def reset_gauges(self):
        """Reset all gauges to zero"""
        with self._lock:
            self.gauges.clear()
            logger.info("Gauges reset")

    def reset_histograms(self):
        """Clear all histogram data"""
        with self._lock:
            self.histograms.clear()
            logger.info("Histograms reset")

    def reset_timeseries(self):
        """Clear all timeseries data"""
        with self._lock:
            self.timeseries.clear()
            logger.info("Timeseries reset")

    def reset_all(self):
        """Reset all metrics"""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timeseries.clear()
            self.aggregates.clear()
            logger.info("All metrics reset")

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics in a structured format

        Returns:
            Dictionary with all metrics data
        """
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: list(values) for name, values in self.histograms.items()
                },
                "timeseries": {
                    name: list(series) for name, series in self.timeseries.items()
                },
                "aggregates": dict(self.aggregates),
                "metadata": {
                    "start_time": self._start_time,
                    "current_time": time.time(),
                    "uptime_seconds": time.time() - self._start_time,
                    "max_histogram_size": self.max_histogram_size,
                    "max_timeseries_size": self.max_timeseries_size,
                },
            }

    def import_metrics(self, data: Dict[str, Any]):
        """
        Import metrics from exported data

        Args:
            data: Dictionary with metrics data (from export_metrics)
        """
        with self._lock:
            # Import counters
            if "counters" in data:
                self.counters = defaultdict(int, data["counters"])

            # Import gauges
            if "gauges" in data:
                self.gauges = defaultdict(float, data["gauges"])

            # Import histograms
            if "histograms" in data:
                for name, values in data["histograms"].items():
                    self.histograms[name] = deque(
                        values, maxlen=self.max_histogram_size
                    )

            # Import timeseries
            if "timeseries" in data:
                for name, series in data["timeseries"].items():
                    self.timeseries[name] = deque(
                        series, maxlen=self.max_timeseries_size
                    )

            # Import aggregates
            if "aggregates" in data:
                self.aggregates = defaultdict(dict, data["aggregates"])

            logger.info("Metrics imported successfully")

    def get_metric_names(self) -> Dict[str, List[str]]:
        """
        Get names of all metrics

        Returns:
            Dictionary mapping metric type to list of metric names
        """
        with self._lock:
            return {
                "counters": list(self.counters.keys()),
                "gauges": list(self.gauges.keys()),
                "histograms": list(self.histograms.keys()),
                "timeseries": list(self.timeseries.keys()),
            }

    def shutdown(self):
        """Gracefully shutdown metrics collector"""
        logger.info("Shutting down metrics collector")

        # Stop cleanup thread
        self._shutdown_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        logger.info("Metrics collector shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logger.debug(f"Error in destructor: {e}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def create_metrics_collector(
    max_histogram_size: int = 10000,
    max_timeseries_size: int = 1000,
    cleanup_interval: int = 300,
) -> EnhancedMetricsCollector:
    """
    Factory function to create metrics collector

    Args:
        max_histogram_size: Maximum number of values in histograms
        max_timeseries_size: Maximum number of points in timeseries
        cleanup_interval: Cleanup interval in seconds

    Returns:
        EnhancedMetricsCollector instance
    """
    return EnhancedMetricsCollector(
        max_histogram_size=max_histogram_size,
        max_timeseries_size=max_timeseries_size,
        cleanup_interval=cleanup_interval,
    )


def compute_percentile(values: List[float], percentile: float) -> float:
    """
    Compute percentile of values

    Args:
        values: List of numeric values
        percentile: Percentile to compute (0-100)

    Returns:
        Percentile value
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(len(sorted_values) * (percentile / 100.0))
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def compute_moving_average(
    timeseries: List[Tuple[float, float]], window_size: int
) -> List[Tuple[float, float]]:
    """
    Compute moving average of timeseries

    Args:
        timeseries: List of (timestamp, value) tuples
        window_size: Size of moving average window

    Returns:
        List of (timestamp, moving_avg) tuples
    """
    if len(timeseries) < window_size:
        return timeseries

    result = []
    values = [v for _, v in timeseries]

    for i in range(len(timeseries)):
        if i < window_size - 1:
            # Not enough data yet
            avg = np.mean(values[: i + 1])
        else:
            # Compute moving average
            avg = np.mean(values[i - window_size + 1 : i + 1])

        result.append((timeseries[i][0], avg))

    return result


def compute_rate(
    timeseries: List[Tuple[float, float]], window_seconds: float = 60.0
) -> float:
    """
    Compute rate of change over time window

    Args:
        timeseries: List of (timestamp, value) tuples
        window_seconds: Time window in seconds

    Returns:
        Rate of change per second
    """
    if len(timeseries) < 2:
        return 0.0

    current_time = time.time()
    cutoff_time = current_time - window_seconds

    # Filter to window
    windowed = [(t, v) for t, v in timeseries if t >= cutoff_time]

    if len(windowed) < 2:
        return 0.0

    # Compute rate
    time_diff = windowed[-1][0] - windowed[0][0]
    value_diff = windowed[-1][1] - windowed[0][1]

    if time_diff > 0:
        return value_diff / time_diff

    return 0.0


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "EnhancedMetricsCollector",
    "MetricType",
    "AggregationType",
    "create_metrics_collector",
    "compute_percentile",
    "compute_moving_average",
    "compute_rate",
]
