"""
System Health Monitor for VULCAN-AGI.

FIX #4: Monitors system health and automatically switches to fast-path mode
when degraded performance is detected. This addresses Arena timeouts, embedding
latency issues, and safety over-blocking by automatically degrading gracefully.

Key Features:
    - Monitors embedding latency (p95 threshold: 1000ms)
    - Tracks Arena timeout rate (threshold: 10% for warning, 30% for skip)
    - Monitors CPU usage (threshold: 90% for batch size reduction)
    - Tracks safety block rate (threshold: 50% for concern)
    - Auto-enables fast path when thresholds exceeded
    - Auto-recovers when metrics improve
    - Provides health status and degradation alerts

Usage:
    from vulcan.routing.system_health_monitor import get_health_monitor, HealthStatus
    
    monitor = get_health_monitor()
    
    # Check if fast path should be used
    if monitor.should_use_fast_path():
        # Use fast path routing
        pass
    
    # Record metrics
    monitor.record_embedding_latency(150.0)  # ms
    monitor.record_arena_result(timed_out=False)
    monitor.record_safety_check(blocked=False)
    
    # Get health status
    status = monitor.get_health_status()
    print(f"System healthy: {status.is_healthy}")
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

# Embedding latency thresholds (milliseconds)
EMBEDDING_P95_WARNING_THRESHOLD_MS = 1000.0
EMBEDDING_P95_CRITICAL_THRESHOLD_MS = 2000.0

# Arena timeout thresholds (percentage)
ARENA_TIMEOUT_WARNING_THRESHOLD_PCT = 10.0
ARENA_TIMEOUT_SKIP_THRESHOLD_PCT = 30.0

# CPU usage thresholds (percentage)
CPU_WARNING_THRESHOLD_PCT = 80.0
CPU_CRITICAL_THRESHOLD_PCT = 90.0

# Safety block rate thresholds (percentage)
SAFETY_BLOCK_WARNING_THRESHOLD_PCT = 30.0
SAFETY_BLOCK_CRITICAL_THRESHOLD_PCT = 50.0

# Metric window sizes (number of samples)
EMBEDDING_WINDOW_SIZE = 100
ARENA_WINDOW_SIZE = 20
SAFETY_WINDOW_SIZE = 50

# Recovery thresholds - must be below these for recovery
EMBEDDING_RECOVERY_THRESHOLD_MS = 500.0
ARENA_RECOVERY_THRESHOLD_PCT = 5.0
CPU_RECOVERY_THRESHOLD_PCT = 70.0

# Minimum samples before making decisions
MIN_SAMPLES_FOR_DECISION = 5

# Auto-recovery cooldown (seconds) - don't recover immediately after degradation
RECOVERY_COOLDOWN_SECONDS = 30.0


class DegradationLevel(Enum):
    """System degradation levels with numeric values for comparison."""
    HEALTHY = 0
    WARNING = 1
    DEGRADED = 2
    CRITICAL = 3


@dataclass
class HealthMetrics:
    """Current health metrics snapshot."""
    embedding_p95_ms: float = 0.0
    embedding_sample_count: int = 0
    arena_timeout_rate_pct: float = 0.0
    arena_sample_count: int = 0
    cpu_usage_pct: float = 0.0
    safety_block_rate_pct: float = 0.0
    safety_sample_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealthStatus:
    """System health status."""
    is_healthy: bool
    degradation_level: DegradationLevel
    fast_path_enabled: bool
    skip_arena: bool
    reduce_batch_size: bool
    use_keyword_routing: bool
    metrics: HealthMetrics
    active_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SystemHealthMonitor:
    """
    Monitors system health and automatically switches to fast-path mode.
    
    This class tracks key performance metrics and automatically enables
    degradation strategies when thresholds are exceeded.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the health monitor.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        
        # Metric storage using deques for efficient windowing
        self._embedding_latencies: Deque[float] = deque(maxlen=EMBEDDING_WINDOW_SIZE)
        self._arena_results: Deque[bool] = deque(maxlen=ARENA_WINDOW_SIZE)  # True = timeout
        self._safety_results: Deque[bool] = deque(maxlen=SAFETY_WINDOW_SIZE)  # True = blocked
        
        # State tracking
        self._fast_path_enabled = False
        self._skip_arena = False
        self._reduce_batch_size = False
        self._use_keyword_routing = False
        self._last_degradation_time: float = 0.0
        self._degradation_level = DegradationLevel.HEALTHY
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event callbacks
        self._degradation_callbacks: List[callable] = []
        self._recovery_callbacks: List[callable] = []
        
        logger.info("[HealthMonitor] Initialized system health monitor")
    
    # ========================================================
    # METRIC RECORDING
    # ========================================================
    
    def record_embedding_latency(self, latency_ms: float) -> None:
        """
        Record an embedding computation latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._embedding_latencies.append(latency_ms)
            self._check_thresholds()
    
    def record_arena_result(self, timed_out: bool, execution_time: float = 0.0) -> None:
        """
        Record an Arena execution result.
        
        Args:
            timed_out: True if the request timed out
            execution_time: Execution time in seconds (for logging)
        """
        with self._lock:
            self._arena_results.append(timed_out)
            if timed_out:
                logger.warning(f"[HealthMonitor] Arena timeout recorded (execution_time={execution_time:.2f}s)")
            self._check_thresholds()
    
    def record_safety_check(self, blocked: bool, reason: str = "") -> None:
        """
        Record a safety check result.
        
        Args:
            blocked: True if the check resulted in a block
            reason: Reason for block (for logging)
        """
        with self._lock:
            self._safety_results.append(blocked)
            if blocked:
                logger.debug(f"[HealthMonitor] Safety block recorded: {reason[:50]}")
            self._check_thresholds()
    
    # ========================================================
    # METRIC CALCULATION
    # ========================================================
    
    def _calculate_p95(self, values: Deque[float]) -> float:
        """Calculate p95 from a deque of values."""
        if len(values) < MIN_SAMPLES_FOR_DECISION:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def _calculate_rate(self, values: Deque[bool]) -> float:
        """Calculate rate (percentage) of True values."""
        if len(values) < MIN_SAMPLES_FOR_DECISION:
            return 0.0
        return sum(1 for v in values if v) / len(values) * 100
    
    def _get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Uses psutil if available, otherwise returns 0.
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            # Try reading from /proc/stat on Linux
            try:
                with open('/proc/loadavg', 'r') as f:
                    load = float(f.read().split()[0])
                    # Estimate CPU usage from load average (rough approximation)
                    cpu_count = os.cpu_count() or 1
                    return min(100.0, load / cpu_count * 100)
            except Exception:
                return 0.0
        except Exception as e:
            logger.debug(f"[HealthMonitor] CPU usage check failed: {e}")
            return 0.0
    
    def _get_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        return HealthMetrics(
            embedding_p95_ms=self._calculate_p95(self._embedding_latencies),
            embedding_sample_count=len(self._embedding_latencies),
            arena_timeout_rate_pct=self._calculate_rate(self._arena_results),
            arena_sample_count=len(self._arena_results),
            cpu_usage_pct=self._get_cpu_usage(),
            safety_block_rate_pct=self._calculate_rate(self._safety_results),
            safety_sample_count=len(self._safety_results),
        )
    
    # ========================================================
    # THRESHOLD CHECKING
    # ========================================================
    
    def _check_thresholds(self) -> None:
        """Check all thresholds and update degradation state."""
        metrics = self._get_metrics()
        issues = []
        recommendations = []
        
        old_level = self._degradation_level
        new_level = DegradationLevel.HEALTHY
        
        # Check embedding latency
        if metrics.embedding_sample_count >= MIN_SAMPLES_FOR_DECISION:
            if metrics.embedding_p95_ms >= EMBEDDING_P95_CRITICAL_THRESHOLD_MS:
                issues.append(f"Embedding p95 critical: {metrics.embedding_p95_ms:.0f}ms")
                recommendations.append("Use keyword routing instead of embeddings")
                self._use_keyword_routing = True
                if DegradationLevel.CRITICAL.value > new_level.value:
                    new_level = DegradationLevel.CRITICAL
            elif metrics.embedding_p95_ms >= EMBEDDING_P95_WARNING_THRESHOLD_MS:
                issues.append(f"Embedding p95 high: {metrics.embedding_p95_ms:.0f}ms")
                if DegradationLevel.WARNING.value > new_level.value:
                    new_level = DegradationLevel.WARNING
        
        # Check Arena timeout rate
        if metrics.arena_sample_count >= MIN_SAMPLES_FOR_DECISION:
            if metrics.arena_timeout_rate_pct >= ARENA_TIMEOUT_SKIP_THRESHOLD_PCT:
                issues.append(f"Arena timeout rate critical: {metrics.arena_timeout_rate_pct:.1f}%")
                recommendations.append("Skip Arena calls")
                self._skip_arena = True
                if DegradationLevel.CRITICAL.value > new_level.value:
                    new_level = DegradationLevel.CRITICAL
            elif metrics.arena_timeout_rate_pct >= ARENA_TIMEOUT_WARNING_THRESHOLD_PCT:
                issues.append(f"Arena timeout rate high: {metrics.arena_timeout_rate_pct:.1f}%")
                if DegradationLevel.WARNING.value > new_level.value:
                    new_level = DegradationLevel.WARNING
        
        # Check CPU usage
        if metrics.cpu_usage_pct >= CPU_CRITICAL_THRESHOLD_PCT:
            issues.append(f"CPU usage critical: {metrics.cpu_usage_pct:.1f}%")
            recommendations.append("Reduce batch sizes")
            self._reduce_batch_size = True
            if DegradationLevel.DEGRADED.value > new_level.value:
                new_level = DegradationLevel.DEGRADED
        elif metrics.cpu_usage_pct >= CPU_WARNING_THRESHOLD_PCT:
            issues.append(f"CPU usage high: {metrics.cpu_usage_pct:.1f}%")
            if DegradationLevel.WARNING.value > new_level.value:
                new_level = DegradationLevel.WARNING
        
        # Check safety block rate
        if metrics.safety_sample_count >= MIN_SAMPLES_FOR_DECISION:
            if metrics.safety_block_rate_pct >= SAFETY_BLOCK_CRITICAL_THRESHOLD_PCT:
                issues.append(f"Safety block rate critical: {metrics.safety_block_rate_pct:.1f}%")
                recommendations.append("Review safety configuration")
                if DegradationLevel.WARNING.value > new_level.value:
                    new_level = DegradationLevel.WARNING
            elif metrics.safety_block_rate_pct >= SAFETY_BLOCK_WARNING_THRESHOLD_PCT:
                issues.append(f"Safety block rate high: {metrics.safety_block_rate_pct:.1f}%")
        
        # Update fast path if any threshold exceeded
        if new_level.value >= DegradationLevel.DEGRADED.value:
            self._fast_path_enabled = True
            self._last_degradation_time = time.time()
        
        # Check for recovery
        elif self._fast_path_enabled:
            self._check_recovery(metrics)
        
        # Log level changes
        if new_level != old_level:
            if new_level.value > old_level.value:
                logger.warning(
                    f"[HealthMonitor] Degradation detected: {old_level.value} -> {new_level.value}. "
                    f"Issues: {', '.join(issues)}"
                )
                self._notify_degradation(new_level, issues)
            else:
                logger.info(
                    f"[HealthMonitor] Partial recovery: {old_level.value} -> {new_level.value}"
                )
        
        self._degradation_level = new_level
    
    def _check_recovery(self, metrics: HealthMetrics) -> None:
        """Check if system has recovered enough to disable fast path."""
        # Don't recover too quickly
        if time.time() - self._last_degradation_time < RECOVERY_COOLDOWN_SECONDS:
            return
        
        can_recover = True
        
        # Check embedding recovery
        if self._use_keyword_routing and metrics.embedding_sample_count >= MIN_SAMPLES_FOR_DECISION:
            if metrics.embedding_p95_ms > EMBEDDING_RECOVERY_THRESHOLD_MS:
                can_recover = False
        
        # Check Arena recovery
        if self._skip_arena and metrics.arena_sample_count >= MIN_SAMPLES_FOR_DECISION:
            if metrics.arena_timeout_rate_pct > ARENA_RECOVERY_THRESHOLD_PCT:
                can_recover = False
        
        # Check CPU recovery
        if self._reduce_batch_size:
            if metrics.cpu_usage_pct > CPU_RECOVERY_THRESHOLD_PCT:
                can_recover = False
        
        if can_recover:
            logger.info("[HealthMonitor] System recovered - disabling fast path mode")
            self._fast_path_enabled = False
            self._skip_arena = False
            self._reduce_batch_size = False
            self._use_keyword_routing = False
            self._degradation_level = DegradationLevel.HEALTHY
            self._notify_recovery()
    
    # ========================================================
    # PUBLIC QUERY METHODS
    # ========================================================
    
    def should_use_fast_path(self) -> bool:
        """Check if fast path should be used due to degradation."""
        with self._lock:
            return self._fast_path_enabled
    
    def should_skip_arena(self) -> bool:
        """Check if Arena should be skipped due to high timeout rate."""
        with self._lock:
            return self._skip_arena
    
    def should_reduce_batch_size(self) -> bool:
        """Check if batch sizes should be reduced due to high CPU."""
        with self._lock:
            return self._reduce_batch_size
    
    def should_use_keyword_routing(self) -> bool:
        """Check if keyword routing should be used instead of embeddings."""
        with self._lock:
            return self._use_keyword_routing
    
    def get_health_status(self) -> HealthStatus:
        """Get comprehensive health status."""
        with self._lock:
            metrics = self._get_metrics()
            issues = []
            recommendations = []
            
            # Collect current issues
            if self._use_keyword_routing:
                issues.append("Embedding latency too high")
                recommendations.append("Using keyword routing")
            if self._skip_arena:
                issues.append("Arena timeout rate too high")
                recommendations.append("Skipping Arena calls")
            if self._reduce_batch_size:
                issues.append("CPU usage too high")
                recommendations.append("Reduced batch sizes")
            
            return HealthStatus(
                is_healthy=self._degradation_level == DegradationLevel.HEALTHY,
                degradation_level=self._degradation_level,
                fast_path_enabled=self._fast_path_enabled,
                skip_arena=self._skip_arena,
                reduce_batch_size=self._reduce_batch_size,
                use_keyword_routing=self._use_keyword_routing,
                metrics=metrics,
                active_issues=issues,
                recommendations=recommendations,
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics for logging/monitoring."""
        with self._lock:
            metrics = self._get_metrics()
            return {
                "degradation_level": self._degradation_level.value,
                "fast_path_enabled": self._fast_path_enabled,
                "skip_arena": self._skip_arena,
                "reduce_batch_size": self._reduce_batch_size,
                "use_keyword_routing": self._use_keyword_routing,
                "embedding_p95_ms": metrics.embedding_p95_ms,
                "embedding_samples": metrics.embedding_sample_count,
                "arena_timeout_rate_pct": metrics.arena_timeout_rate_pct,
                "arena_samples": metrics.arena_sample_count,
                "cpu_usage_pct": metrics.cpu_usage_pct,
                "safety_block_rate_pct": metrics.safety_block_rate_pct,
                "safety_samples": metrics.safety_sample_count,
            }
    
    # ========================================================
    # CALLBACKS
    # ========================================================
    
    def on_degradation(self, callback: callable) -> None:
        """Register a callback for degradation events."""
        self._degradation_callbacks.append(callback)
    
    def on_recovery(self, callback: callable) -> None:
        """Register a callback for recovery events."""
        self._recovery_callbacks.append(callback)
    
    def _notify_degradation(self, level: DegradationLevel, issues: List[str]) -> None:
        """Notify all degradation callbacks."""
        for callback in self._degradation_callbacks:
            try:
                callback(level, issues)
            except Exception as e:
                logger.error(f"[HealthMonitor] Degradation callback failed: {e}")
    
    def _notify_recovery(self) -> None:
        """Notify all recovery callbacks."""
        for callback in self._recovery_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"[HealthMonitor] Recovery callback failed: {e}")
    
    # ========================================================
    # RESET AND CLEAR
    # ========================================================
    
    def reset(self) -> None:
        """Reset all metrics and state."""
        with self._lock:
            self._embedding_latencies.clear()
            self._arena_results.clear()
            self._safety_results.clear()
            self._fast_path_enabled = False
            self._skip_arena = False
            self._reduce_batch_size = False
            self._use_keyword_routing = False
            self._last_degradation_time = 0.0
            self._degradation_level = DegradationLevel.HEALTHY
        
        logger.info("[HealthMonitor] Monitor reset")
    
    def force_fast_path(self, enable: bool = True) -> None:
        """Manually force fast path mode (for testing/emergency)."""
        with self._lock:
            self._fast_path_enabled = enable
            if enable:
                self._last_degradation_time = time.time()
                logger.warning("[HealthMonitor] Fast path FORCED enabled")
            else:
                logger.info("[HealthMonitor] Fast path FORCED disabled")


# ============================================================
# GLOBAL SINGLETON
# ============================================================

_health_monitor: Optional[SystemHealthMonitor] = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> SystemHealthMonitor:
    """Get the global health monitor singleton."""
    global _health_monitor
    
    if _health_monitor is None:
        with _monitor_lock:
            if _health_monitor is None:
                _health_monitor = SystemHealthMonitor()
    
    return _health_monitor


def reset_health_monitor() -> None:
    """Reset the global health monitor."""
    global _health_monitor
    
    with _monitor_lock:
        if _health_monitor is not None:
            _health_monitor.reset()
        _health_monitor = SystemHealthMonitor()


__all__ = [
    "SystemHealthMonitor",
    "HealthStatus",
    "HealthMetrics",
    "DegradationLevel",
    "get_health_monitor",
    "reset_health_monitor",
    # Constants for reference
    "EMBEDDING_P95_WARNING_THRESHOLD_MS",
    "ARENA_TIMEOUT_SKIP_THRESHOLD_PCT",
    "CPU_CRITICAL_THRESHOLD_PCT",
    "SAFETY_BLOCK_CRITICAL_THRESHOLD_PCT",
]
