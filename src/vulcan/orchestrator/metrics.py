# ============================================================
# VULCAN-AGI Orchestrator - Metrics Module
# Comprehensive metrics collection and monitoring
# FULLY FIXED VERSION - Enhanced with proper bounds, cleanup, and no circular dependencies
# FIXED: Converted long time.sleep(self.cleanup_interval) to interruptible wait.
# ============================================================

import logging
import threading
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
    - Thread-safe operations with sharded locks for reduced contention
    - Bounded memory for all data structures
    - No circular dependencies
    - Comprehensive metric types (counters, gauges, histograms, timeseries)
    - Health score computation with queue depth factor
    - Automatic cleanup of old data
    
    Performance Optimization:
    - Uses separate locks for different metric types (sharded locking)
    - Allows concurrent updates to different metric types
    - Reduces convoy effect in high-concurrency scenarios
    """

    def __init__(
        self,
        max_histogram_size: int = 10000,
        max_timeseries_size: int = 1000,
        cleanup_interval: int = 300,
        max_healthy_queue_depth: int = 100,
    ):
        """
        Initialize metrics collector

        Args:
            max_histogram_size: Maximum number of values to keep in histograms
            max_timeseries_size: Maximum number of points to keep in timeseries
            cleanup_interval: Interval in seconds between cleanup operations
            max_healthy_queue_depth: Maximum queue depth considered healthy for health score
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

        # Thread safety - SHARDED LOCKS for reduced contention
        # Each metric type has its own lock to allow concurrent updates
        self._counter_lock = threading.RLock()
        self._gauge_lock = threading.RLock()
        self._histogram_lock = threading.RLock()
        self._timeseries_lock = threading.RLock()
        self._aggregate_lock = threading.RLock()

        # Cleanup configuration
        self.max_histogram_size = max_histogram_size
        self.max_timeseries_size = max_timeseries_size
        self.cleanup_interval = cleanup_interval
        self.max_healthy_queue_depth = max_healthy_queue_depth
        self._last_cleanup = time.time()
        self._cleanup_lock = threading.RLock()

        # Start time for uptime tracking
        self._start_time = time.time()

        # Cleanup thread
        self._shutdown_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="MetricsCleanup"
        )
        self._cleanup_thread.start()

        logger.info("EnhancedMetricsCollector initialized with sharded locks")

    def record_step(self, duration: float, result: Dict[str, Any]):
        """
        Record step metrics from orchestrator execution
        
        Uses sharded locks for performance - different metric types can be updated concurrently.

        Args:
            duration: Step duration in seconds
            result: Result dictionary from step execution
        """
        current_time = time.time()
        duration_ms = duration * 1000
        
        # Record counters (batch to minimize lock acquisition)
        with self._counter_lock:
            self.counters["steps_total"] += 1
            
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
            
            # Record success/failure
            is_success = (
                result.get("success", False) or result.get("status") == "completed"
            )
            if is_success:
                self.counters["successful_actions"] += 1
            else:
                self.counters["failed_actions"] += 1

        # Record histograms
        with self._histogram_lock:
            self.histograms["step_duration_ms"].append(duration_ms)
            
            # Record learning metrics
            if "loss" in result:
                self.histograms["learning_loss"].append(result["loss"])
            
            # Record uncertainty
            if "uncertainty" in result:
                self.histograms["uncertainty"].append(result["uncertainty"])
            
            # Record resource usage
            if "resource_usage" in result:
                for resource, value in result["resource_usage"].items():
                    metric_name = f"resource_{resource}"
                    self.histograms[metric_name].append(value)
            
            # Record reward
            if "reward" in result:
                self.histograms["reward"].append(result["reward"])
        
        # Record timeseries
        with self._timeseries_lock:
            self.timeseries["step_duration_ms_time"].append((current_time, duration_ms))
            
            if "loss" in result:
                self.timeseries["loss_over_time"].append((current_time, result["loss"]))
            
            if "uncertainty" in result:
                self.timeseries["uncertainty_over_time"].append((current_time, result["uncertainty"]))
            
            if "resource_usage" in result:
                for resource, value in result["resource_usage"].items():
                    metric_name = f"resource_{resource}"
                    self.timeseries[f"{metric_name}_time"].append((current_time, value))
            
            if "reward" in result:
                self.timeseries["reward_over_time"].append((current_time, result["reward"]))
        
        # Record gauges
        with self._gauge_lock:
            if "loss" in result:
                self.gauges["current_loss"] = result["loss"]
            
            if "uncertainty" in result:
                self.gauges["current_uncertainty"] = result["uncertainty"]
            
            if "resource_usage" in result:
                for resource, value in result["resource_usage"].items():
                    self.gauges[f"current_resource_{resource}"] = value
            
            if "reward" in result:
                self.gauges["current_reward"] = result["reward"]

        # Periodic cleanup check (using cleanup lock)
        with self._cleanup_lock:
            if current_time - self._last_cleanup > self.cleanup_interval:
                self._perform_cleanup()

    def update_gauge(self, name: str, value: float):
        """
        Update gauge metric

        Args:
            name: Gauge name
            value: Current value
        """
        current_time = time.time()
        with self._gauge_lock:
            self.gauges[name] = value
        
        with self._timeseries_lock:
            self.timeseries[f"{name}_history"].append((current_time, value))

    def increment_counter(self, name: str, value: int = 1):
        """
        Increment counter metric

        Args:
            name: Counter name
            value: Amount to increment by
        """
        with self._counter_lock:
            self.counters[name] += value

    def decrement_counter(self, name: str, value: int = 1):
        """
        Decrement counter metric

        Args:
            name: Counter name
            value: Amount to decrement by
        """
        with self._counter_lock:
            self.counters[name] -= value

    def record_histogram(self, name: str, value: float):
        """
        Record value in histogram

        Args:
            name: Histogram name
            value: Value to record
        """
        with self._histogram_lock:
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
        with self._timeseries_lock:
            self.timeseries[name].append((timestamp, value))

    def record_event(self, event_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record discrete event

        Args:
            event_type: Type of event
            metadata: Optional event metadata
        """
        with self._counter_lock:
            self.counters[f"event_{event_type}"] += 1

        if metadata:
            # Store only the most recent metadata for each event type
            with self._aggregate_lock:
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
        with self._counter_lock:
            return self.counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """
        Get gauge value

        Args:
            name: Gauge name

        Returns:
            Gauge value
        """
        with self._gauge_lock:
            return self.gauges.get(name, 0.0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Uses multiple locks in a consistent order to avoid deadlocks.

        Returns:
            Dictionary with all metrics summaries
        """
        # Acquire locks in consistent order to avoid deadlocks
        with self._counter_lock:
            counters_copy = dict(self.counters)
        
        with self._gauge_lock:
            gauges_copy = dict(self.gauges)
        
        with self._histogram_lock:
            histograms_copy = {name: list(values) for name, values in self.histograms.items()}
        
        summary = {
            "counters": counters_copy,
            "gauges": gauges_copy,
            "histograms_summary": {},
            "rates": {},
            "health_score": self._compute_health_score(),
            "uptime_seconds": time.time() - self._start_time,
            "timestamp": time.time(),
        }

        # Compute histogram summaries (no lock needed, working with copy)
        for name, values in histograms_copy.items():
            if values:
                summary["histograms_summary"][name] = self._compute_histogram_stats(values)

        # Compute rates (no lock needed, working with copy)
        total_actions = counters_copy.get("successful_actions", 0) + counters_copy.get("failed_actions", 0)
        if total_actions > 0:
            summary["rates"]["success_rate"] = counters_copy["successful_actions"] / total_actions
            summary["rates"]["failure_rate"] = counters_copy["failed_actions"] / total_actions
        else:
            summary["rates"]["success_rate"] = 0.0
            summary["rates"]["failure_rate"] = 0.0

        # Compute steps per second
        uptime = time.time() - self._start_time
        if uptime > 0:
            summary["rates"]["steps_per_second"] = counters_copy.get("steps_total", 0) / uptime
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
        with self._histogram_lock:
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
        with self._timeseries_lock:
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
        with self._timeseries_lock:
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
    
    def record_queue_depth(self, depth: int):
        """
        Record current queue depth for health score computation.
        
        Args:
            depth: Current number of jobs in queue
        """
        with self._gauge_lock:
            self.gauges["queue_depth"] = float(depth)
        
        with self._histogram_lock:
            self.histograms["queue_depth"].append(float(depth))

    def _compute_health_score(self) -> float:
        """
        Compute overall system health score (0.0 to 1.0)
        
        GAP 9 FIX: Now includes answer quality metrics, not just throughput/latency.
        ISSUE 7 FIX: Now includes queue depth factor.
        
        Factors considered:
        - Success rate (35% weight) - task completion
        - Uncertainty (15% weight) - model confidence calibration
        - Latency efficiency (15% weight) - response time
        - Answer quality (15% weight) - GAP 9 FIX: actual correctness
        - Queue depth (10% weight) - ISSUE 7 FIX: job backlog indicator
        - Tool diversity (5% weight) - GAP 9 FIX: specialized tools used
        - Routing integrity (5% weight) - GAP 9 FIX: routing decisions respected

        Returns:
            Health score where 1.0 is perfect health
        """
        factors = []
        weights = []

        # Success rate factor (weight: 0.35) - reduced from 0.40 to make room for queue depth
        with self._counter_lock:
            total_actions = self.counters.get("successful_actions", 0) + self.counters.get(
                "failed_actions", 0
            )
            if total_actions > 0:
                success_rate = self.counters["successful_actions"] / total_actions
                factors.append(success_rate)
                weights.append(0.35)

        # Uncertainty factor (weight: 0.15)
        with self._gauge_lock:
            if "current_uncertainty" in self.gauges:
                uncertainty = self.gauges["current_uncertainty"]
                # Lower uncertainty is better
                uncertainty_score = max(0.0, 1.0 - uncertainty)
                factors.append(uncertainty_score)
                weights.append(0.15)

        # Resource efficiency factor (weight: 0.15)
        with self._histogram_lock:
            if (
                "step_duration_ms" in self.histograms
                and len(self.histograms["step_duration_ms"]) > 0
            ):
                recent_durations = list(self.histograms["step_duration_ms"])[-100:]
                if NUMPY_AVAILABLE:
                    avg_time = np.mean(recent_durations)
                else:
                    avg_time = sum(recent_durations) / len(recent_durations)
                # Assume target is 100ms, score decreases as we deviate
                if avg_time > 0:
                    efficiency = min(1.0, 100.0 / avg_time)
                else:
                    efficiency = 0.0
                factors.append(efficiency)
                weights.append(0.15)
        
        # =====================================================================
        # GAP 9 FIX: Answer quality factor (weight: 0.15)
        # Measures actual correctness, not just completion
        # =====================================================================
        with self._histogram_lock:
            if "answer_quality" in self.histograms and len(self.histograms["answer_quality"]) > 0:
                recent_quality = list(self.histograms["answer_quality"])[-100:]
                if NUMPY_AVAILABLE:
                    avg_quality = np.mean(recent_quality)
                else:
                    avg_quality = sum(recent_quality) / len(recent_quality)
                factors.append(avg_quality)
                weights.append(0.15)
            elif "user_satisfaction" in self.histograms and len(self.histograms["user_satisfaction"]) > 0:
                # Fallback to user satisfaction if answer_quality not tracked
                recent_satisfaction = list(self.histograms["user_satisfaction"])[-100:]
                if NUMPY_AVAILABLE:
                    avg_satisfaction = np.mean(recent_satisfaction)
                else:
                    avg_satisfaction = sum(recent_satisfaction) / len(recent_satisfaction)
                factors.append(avg_satisfaction)
                weights.append(0.15)
        
        # =====================================================================
        # ISSUE 7 FIX: Queue depth factor (weight: 0.10)
        # Measures job backlog - critical indicator of system health
        # =====================================================================
        with self._gauge_lock:
            if "queue_depth" in self.gauges:
                current_depth = self.gauges["queue_depth"]
                # Score decreases linearly as queue approaches max_healthy_depth
                # Score is 0 when queue >= max_healthy_depth
                if current_depth <= self.max_healthy_queue_depth:
                    queue_score = 1.0 - (current_depth / self.max_healthy_queue_depth)
                else:
                    queue_score = 0.0
                factors.append(queue_score)
                weights.append(0.10)
        
        # =====================================================================
        # GAP 9 FIX: Tool diversity factor (weight: 0.05) - reduced from 0.10
        # Penalizes over-reliance on fallback tools like world_model
        # =====================================================================
        with self._gauge_lock:
            if "tool_diversity" in self.gauges:
                tool_diversity = self.gauges["tool_diversity"]
                factors.append(tool_diversity)
                weights.append(0.05)
            else:
                # Calculate from tool usage counters if available
                tool_usage = self._compute_tool_diversity()
                if tool_usage is not None:
                    factors.append(tool_usage)
                    weights.append(0.05)
        
        # =====================================================================
        # GAP 9 FIX: Routing integrity factor (weight: 0.05)
        # Measures how often routing decisions are being overridden
        # =====================================================================
        with self._gauge_lock:
            if "routing_integrity" in self.gauges:
                routing_integrity = self.gauges["routing_integrity"]
                factors.append(routing_integrity)
                weights.append(0.05)
        
        with self._counter_lock:
            if "routing_overrides" in self.counters and "routing_total" in self.counters:
                total_routing = self.counters["routing_total"]
                if total_routing > 0:
                    override_rate = self.counters["routing_overrides"] / total_routing
                    routing_integrity = 1.0 - override_rate
                    factors.append(routing_integrity)
                    weights.append(0.05)

        # Compute weighted average
        if factors:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(f * w for f, w in zip(factors, weights))
                return weighted_sum / total_weight

        # Default to neutral score if no data
        return 0.5
    
    def _compute_tool_diversity(self) -> Optional[float]:
        """
        GAP 9 FIX: Compute tool diversity score.
        
        Low diversity (only using world_model/general) = unhealthy
        High diversity (using specialized tools) = healthy
        
        Returns:
            Tool diversity score (0-1) or None if not enough data
        """
        # Count tool usage from counters (called with counter_lock already held)
        tool_counts = {}
        fallback_tools = {'world_model', 'general', 'meta_reasoning'}
        specialized_tools = {'causal', 'probabilistic', 'symbolic', 'mathematical', 
                            'philosophical', 'analogical', 'multimodal', 'cryptographic'}
        
        for key, count in self.counters.items():
            if key.startswith("tool_") and key.endswith("_count"):
                tool_name = key[5:-6]  # Extract tool name from "tool_X_count"
                tool_counts[tool_name] = count
        
        if not tool_counts:
            return None
        
        total_usage = sum(tool_counts.values())
        if total_usage == 0:
            return None
        
        # Calculate fallback vs specialized usage
        fallback_usage = sum(tool_counts.get(t, 0) for t in fallback_tools)
        specialized_usage = sum(tool_counts.get(t, 0) for t in specialized_tools)
        
        # Diversity score: higher when more specialized tools are used
        if specialized_usage + fallback_usage == 0:
            return 0.5  # No data
        
        diversity = specialized_usage / (specialized_usage + fallback_usage)
        
        # Also penalize if only one tool is used (even if specialized)
        unique_tools = sum(1 for c in tool_counts.values() if c > 0)
        if unique_tools == 1:
            diversity *= 0.5  # Penalize single-tool usage
        
        return diversity
    
    # =========================================================================
    # GAP 9 FIX: New methods for tracking answer quality
    # =========================================================================
    
    def record_answer_quality(
        self, 
        query_id: str,
        tools_used: list,
        tools_intended: list,
        confidence: float,
        correctness: Optional[float] = None,
        user_satisfaction: Optional[float] = None
    ):
        """
        GAP 9 FIX: Record answer quality metrics.
        
        Unlike record_step() which only tracks completion, this method
        tracks whether the answer was actually correct/useful.
        
        Args:
            query_id: Unique identifier for the query
            tools_used: List of tools that were actually used
            tools_intended: List of tools that were originally selected
            confidence: Model confidence in the answer
            correctness: Optional correctness score (0-1) if known
            user_satisfaction: Optional user satisfaction score (0-1)
        """
        current_time = time.time()
        
        # Track tool override rate and usage
        was_overridden = set(tools_used) != set(tools_intended)
        with self._counter_lock:
            self.counters["routing_total"] += 1
            if was_overridden:
                self.counters["routing_overrides"] += 1
                logger.debug(
                    f"[Metrics] GAP 9: Tool override detected: "
                    f"intended={tools_intended}, used={tools_used}"
                )
            
            # Track tool usage
            for tool in tools_used:
                self.counters[f"tool_{tool}_count"] += 1
        
        # Track answer quality if provided
        with self._histogram_lock:
            if correctness is not None:
                self.histograms["answer_quality"].append(correctness)
            
            if user_satisfaction is not None:
                self.histograms["user_satisfaction"].append(user_satisfaction)
            
            # Track confidence for calibration
            self.histograms["answer_confidence"].append(confidence)
        
        with self._timeseries_lock:
            if correctness is not None:
                self.timeseries["answer_quality_time"].append((current_time, correctness))
            
            if user_satisfaction is not None:
                self.timeseries["user_satisfaction_time"].append((current_time, user_satisfaction))
        
        # Update routing integrity and tool diversity gauges
        with self._gauge_lock:
            with self._counter_lock:
                total_routing = self.counters.get("routing_total", 1)
                override_count = self.counters.get("routing_overrides", 0)
                self.gauges["routing_integrity"] = 1.0 - (override_count / total_routing)
                
                # Update tool diversity gauge
                diversity = self._compute_tool_diversity()
                if diversity is not None:
                    self.gauges["tool_diversity"] = diversity

    def _perform_cleanup(self):
        """Perform cleanup of old data to maintain memory bounds"""
        current_time = time.time()

        # Clean up old timeseries data (older than 1 hour)
        cutoff_time = current_time - 3600

        with self._timeseries_lock:
            for name, series in list(self.timeseries.items()):
                # Remove old entries
                while series and series[0][0] < cutoff_time:
                    series.popleft()

                # If series is empty, remove it
                if not series:
                    del self.timeseries[name]

        # Clean up old aggregate metadata (older than 1 hour)
        with self._aggregate_lock:
            for key in list(self.aggregates.keys()):
                if isinstance(self.aggregates[key], dict):
                    timestamp = self.aggregates[key].get("timestamp", 0)
                    if timestamp < cutoff_time:
                        del self.aggregates[key]

        with self._cleanup_lock:
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
        with self._counter_lock:
            self.counters.clear()
            logger.info("Counters reset")

    def reset_gauges(self):
        """Reset all gauges to zero"""
        with self._gauge_lock:
            self.gauges.clear()
            logger.info("Gauges reset")

    def reset_histograms(self):
        """Clear all histogram data"""
        with self._histogram_lock:
            self.histograms.clear()
            logger.info("Histograms reset")

    def reset_timeseries(self):
        """Clear all timeseries data"""
        with self._timeseries_lock:
            self.timeseries.clear()
            logger.info("Timeseries reset")

    def reset_all(self):
        """Reset all metrics"""
        with self._counter_lock:
            self.counters.clear()
        with self._gauge_lock:
            self.gauges.clear()
        with self._histogram_lock:
            self.histograms.clear()
        with self._timeseries_lock:
            self.timeseries.clear()
        with self._aggregate_lock:
            self.aggregates.clear()
        logger.info("All metrics reset")

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics in a structured format

        Returns:
            Dictionary with all metrics data
        """
        # Acquire locks in consistent order
        with self._counter_lock:
            counters_copy = dict(self.counters)
        with self._gauge_lock:
            gauges_copy = dict(self.gauges)
        with self._histogram_lock:
            histograms_copy = {
                name: list(values) for name, values in self.histograms.items()
            }
        with self._timeseries_lock:
            timeseries_copy = {
                name: list(series) for name, series in self.timeseries.items()
            }
        with self._aggregate_lock:
            aggregates_copy = dict(self.aggregates)
        
        return {
            "counters": counters_copy,
            "gauges": gauges_copy,
            "histograms": histograms_copy,
            "timeseries": timeseries_copy,
            "aggregates": aggregates_copy,
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
        # Import counters
        if "counters" in data:
            with self._counter_lock:
                self.counters = defaultdict(int, data["counters"])

        # Import gauges
        if "gauges" in data:
            with self._gauge_lock:
                self.gauges = defaultdict(float, data["gauges"])

        # Import histograms
        if "histograms" in data:
            with self._histogram_lock:
                for name, values in data["histograms"].items():
                    self.histograms[name] = deque(
                        values, maxlen=self.max_histogram_size
                    )

        # Import timeseries
        if "timeseries" in data:
            with self._timeseries_lock:
                for name, series in data["timeseries"].items():
                    self.timeseries[name] = deque(
                        series, maxlen=self.max_timeseries_size
                    )

        # Import aggregates
        if "aggregates" in data:
            with self._aggregate_lock:
                self.aggregates = defaultdict(dict, data["aggregates"])

        logger.info("Metrics imported successfully")

    def get_metric_names(self) -> Dict[str, List[str]]:
        """
        Get names of all metrics

        Returns:
            Dictionary mapping metric type to list of metric names
        """
        with self._counter_lock:
            counter_names = list(self.counters.keys())
        with self._gauge_lock:
            gauge_names = list(self.gauges.keys())
        with self._histogram_lock:
            histogram_names = list(self.histograms.keys())
        with self._timeseries_lock:
            timeseries_names = list(self.timeseries.keys())
        
        return {
            "counters": counter_names,
            "gauges": gauge_names,
            "histograms": histogram_names,
            "timeseries": timeseries_names,
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
    max_healthy_queue_depth: int = 100,
) -> EnhancedMetricsCollector:
    """
    Factory function to create metrics collector

    Args:
        max_histogram_size: Maximum number of values in histograms
        max_timeseries_size: Maximum number of points in timeseries
        cleanup_interval: Cleanup interval in seconds
        max_healthy_queue_depth: Maximum queue depth considered healthy for health score

    Returns:
        EnhancedMetricsCollector instance
    """
    return EnhancedMetricsCollector(
        max_histogram_size=max_histogram_size,
        max_timeseries_size=max_timeseries_size,
        cleanup_interval=cleanup_interval,
        max_healthy_queue_depth=max_healthy_queue_depth,
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

# ============================================================
# RESPONSE TIME TRACKER - Performance Monitoring
# ============================================================


class ResponseTimeTracker:
    """Tracks response times for performance monitoring and adaptive scaling.
    
    Maintains a sliding window of response times to compute percentiles
    and detect performance degradation for auto-scaling decisions.
    """
    
    def __init__(self, window_size: int = 1000, alert_threshold_ms: float = 5000.0):
        """
        Initialize response time tracker.
        
        Args:
            window_size: Number of samples to keep in sliding window
            alert_threshold_ms: Response time threshold for alerts (milliseconds)
        """
        self.window_size = window_size
        self.alert_threshold_ms = alert_threshold_ms
        self._samples: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()
        self._degradation_callbacks: List[callable] = []
    
    def record(self, duration_ms: float, job_id: str = None, agent_id: str = None) -> None:
        """Record a response time sample."""
        timestamp = time.time()
        with self._lock:
            self._samples.append({
                "timestamp": timestamp,
                "duration_ms": duration_ms,
                "job_id": job_id,
                "agent_id": agent_id,
            })
        
        # Check for degradation
        if duration_ms > self.alert_threshold_ms:
            self._notify_degradation(duration_ms, job_id, agent_id)
    
    def get_percentile(self, percentile: float) -> float:
        """Get the specified percentile of response times."""
        with self._lock:
            if not self._samples:
                return 0.0
            durations = sorted([s["duration_ms"] for s in self._samples])
            idx = int(len(durations) * percentile / 100.0)
            return durations[min(idx, len(durations) - 1)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive response time statistics."""
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "max_ms": 0.0,
                    "min_ms": 0.0,
                }
            
            durations = [s["duration_ms"] for s in self._samples]
            sorted_durations = sorted(durations)
            
            return {
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "p50_ms": sorted_durations[len(sorted_durations) // 2],
                "p95_ms": sorted_durations[int(len(sorted_durations) * 0.95)],
                "p99_ms": sorted_durations[int(len(sorted_durations) * 0.99)],
                "max_ms": max(durations),
                "min_ms": min(durations),
            }
    
    def register_degradation_callback(self, callback: callable) -> None:
        """Register a callback for performance degradation alerts."""
        self._degradation_callbacks.append(callback)
    
    def _notify_degradation(self, duration_ms: float, job_id: str, agent_id: str) -> None:
        """Notify registered callbacks of performance degradation."""
        for callback in self._degradation_callbacks:
            try:
                callback(duration_ms, job_id, agent_id)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}")
    
    def get_recent_trend(self, window_seconds: float = 60.0) -> float:
        """Get trend of response times in the recent window.
        
        Returns:
            Positive value indicates degradation, negative indicates improvement.
        """
        now = time.time()
        with self._lock:
            recent = [s for s in self._samples if now - s["timestamp"] <= window_seconds]
            
            if len(recent) < 2:
                return 0.0
            
            # Compare first half vs second half
            mid = len(recent) // 2
            first_half_avg = sum(s["duration_ms"] for s in recent[:mid]) / mid
            second_half_avg = sum(s["duration_ms"] for s in recent[mid:]) / (len(recent) - mid)
            
            return second_half_avg - first_half_avg
    
    def trim_to_window_size(self) -> None:
        """Trim samples to window size to prevent memory growth.
        
        PERFORMANCE FIX: Called periodically to ensure the samples deque
        doesn't exceed the configured window size.
        """
        with self._lock:
            # The deque has maxlen, but we explicitly trim for safety
            if len(self._samples) > self.window_size:
                # Keep only the most recent samples
                recent = list(self._samples)[-self.window_size:]
                self._samples.clear()
                self._samples.extend(recent)


# ============================================================
# SYSTEM METRICS - Monitoring and Instrumentation
# ============================================================


class SystemMetrics:
    """
    System metrics for monitoring CuriosityEngine and AgentPool performance.
    
    Tracks:
    - Curiosity engine useful/empty cycles
    - Agent job latency percentiles (p50, p99)
    - Stuck job recoveries
    - Dead letter queue jobs
    """
    
    # Alert threshold for job latency p99 in milliseconds (10 seconds)
    ALERT_LATENCY_THRESHOLD_MS = 10000.0
    
    def __init__(self, alert_threshold_dormancy: float = 0.95):
        """
        Initialize system metrics.
        
        Args:
            alert_threshold_dormancy: Threshold for alerting on curiosity dormancy (0.0-1.0)
        """
        self._lock = threading.RLock()
        self.metrics: Dict[str, Any] = {
            # Curiosity Engine metrics
            "curiosity_useful_cycles": 0,
            "curiosity_empty_cycles": 0,
            "curiosity_total_experiments": 0,
            "curiosity_successful_experiments": 0,
            
            # Agent Pool metrics
            "agent_job_latencies_ms": deque(maxlen=1000),  # Rolling window
            "agent_job_latency_p50": 0.0,
            "agent_job_latency_p99": 0.0,
            "stuck_job_recoveries": 0,
            "dead_letter_jobs": 0,
            
            # Health metrics
            "last_healthy_cycle_timestamp": time.time(),
            "consecutive_alerts": 0,
        }
        self.alert_threshold_dormancy = alert_threshold_dormancy
        
        logger.info("SystemMetrics initialized for monitoring")
    
    def record_curiosity_cycle(self, experiments_run: int, successful: int) -> None:
        """
        Record a curiosity engine learning cycle result.
        
        Args:
            experiments_run: Number of experiments run in this cycle
            successful: Number of successful experiments
        """
        with self._lock:
            if experiments_run > 0:
                self.metrics["curiosity_useful_cycles"] += 1
                self.metrics["last_healthy_cycle_timestamp"] = time.time()
            else:
                self.metrics["curiosity_empty_cycles"] += 1
            
            self.metrics["curiosity_total_experiments"] += experiments_run
            self.metrics["curiosity_successful_experiments"] += successful
    
    def record_job_latency(self, latency_ms: float) -> None:
        """
        Record a job latency measurement.
        
        Args:
            latency_ms: Job latency in milliseconds
        """
        with self._lock:
            self.metrics["agent_job_latencies_ms"].append(latency_ms)
            self._update_latency_percentiles()
    
    def _update_latency_percentiles(self) -> None:
        """Update p50 and p99 latency metrics. Must be called with lock held."""
        latencies = list(self.metrics["agent_job_latencies_ms"])
        if not latencies:
            return
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # Calculate p50 (median) - proper handling for even-length arrays
        if n % 2 == 0:
            # For even-length arrays, median is average of two middle values
            mid = n // 2
            self.metrics["agent_job_latency_p50"] = (
                sorted_latencies[mid - 1] + sorted_latencies[mid]
            ) / 2.0
        else:
            # For odd-length arrays, median is the middle value
            self.metrics["agent_job_latency_p50"] = sorted_latencies[n // 2]
        
        # Calculate p99
        p99_idx = min(int(n * 0.99), n - 1)
        self.metrics["agent_job_latency_p99"] = sorted_latencies[p99_idx]
    
    def record_stuck_job_recovery(self) -> None:
        """Record a stuck job recovery event."""
        with self._lock:
            self.metrics["stuck_job_recoveries"] += 1
    
    def record_dead_letter_job(self) -> None:
        """Record a job moved to dead letter queue."""
        with self._lock:
            self.metrics["dead_letter_jobs"] += 1
    
    def get_dormancy_ratio(self) -> float:
        """
        Get the ratio of empty cycles to total cycles.
        
        Returns:
            Dormancy ratio (0.0-1.0), higher means more dormant
        """
        with self._lock:
            total = (
                self.metrics["curiosity_useful_cycles"] + 
                self.metrics["curiosity_empty_cycles"]
            )
            if total == 0:
                return 0.0
            return self.metrics["curiosity_empty_cycles"] / total
    
    def should_alert(self) -> Optional[str]:
        """
        Check if any metric warrants an alert.
        
        Returns:
            Alert message string if alerting, None otherwise
        """
        with self._lock:
            # Alert if curiosity engine is too dormant
            dormancy = self.get_dormancy_ratio()
            if dormancy > self.alert_threshold_dormancy:
                self.metrics["consecutive_alerts"] += 1
                return f"Curiosity engine stuck in dormancy (ratio={dormancy:.2f})"
            
            # Alert if job latencies spike
            if self.metrics["agent_job_latency_p99"] > self.ALERT_LATENCY_THRESHOLD_MS:
                self.metrics["consecutive_alerts"] += 1
                return (
                    f"Job processing latency spike "
                    f"(p99={self.metrics['agent_job_latency_p99']:.0f}ms)"
                )
            
            # Reset consecutive alerts if healthy
            self.metrics["consecutive_alerts"] = 0
            return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary of all metrics (excludes the raw latencies deque)
        """
        with self._lock:
            return {
                "curiosity_useful_cycles": self.metrics["curiosity_useful_cycles"],
                "curiosity_empty_cycles": self.metrics["curiosity_empty_cycles"],
                "curiosity_total_experiments": self.metrics["curiosity_total_experiments"],
                "curiosity_successful_experiments": self.metrics["curiosity_successful_experiments"],
                "curiosity_dormancy_ratio": self.get_dormancy_ratio(),
                "agent_job_latency_p50": self.metrics["agent_job_latency_p50"],
                "agent_job_latency_p99": self.metrics["agent_job_latency_p99"],
                "stuck_job_recoveries": self.metrics["stuck_job_recoveries"],
                "dead_letter_jobs": self.metrics["dead_letter_jobs"],
                "last_healthy_cycle_timestamp": self.metrics["last_healthy_cycle_timestamp"],
                "consecutive_alerts": self.metrics["consecutive_alerts"],
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics["curiosity_useful_cycles"] = 0
            self.metrics["curiosity_empty_cycles"] = 0
            self.metrics["curiosity_total_experiments"] = 0
            self.metrics["curiosity_successful_experiments"] = 0
            self.metrics["agent_job_latencies_ms"].clear()
            self.metrics["agent_job_latency_p50"] = 0.0
            self.metrics["agent_job_latency_p99"] = 0.0
            self.metrics["stuck_job_recoveries"] = 0
            self.metrics["dead_letter_jobs"] = 0
            self.metrics["last_healthy_cycle_timestamp"] = time.time()
            self.metrics["consecutive_alerts"] = 0


__all__ = [
    "EnhancedMetricsCollector",
    "MetricType",
    "AggregationType",
    "create_metrics_collector",
    "compute_percentile",
    "compute_moving_average",
    "compute_rate",
    "ResponseTimeTracker",
    "SystemMetrics",
]
