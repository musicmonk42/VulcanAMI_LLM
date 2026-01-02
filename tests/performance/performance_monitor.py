#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Performance Monitor Module.

A production-grade performance monitoring utility for stress tests and
system health monitoring. Provides comprehensive tracking of system
metrics during test execution.

Key Features:
    - CPU utilization tracking (total and per-core)
    - Memory usage monitoring (RSS, VMS, percentage)
    - Thread and process count tracking
    - SQLite file size and WAL growth monitoring
    - Anomaly detection (CPU spikes, memory leaks)
    - Time-series data export for graphing

Architecture:
    The monitor runs a background thread that samples metrics at
    configurable intervals. All operations are thread-safe.

Usage:
    Context manager (recommended)::

        with PerformanceMonitor(sample_interval_seconds=1.0) as monitor:
            # Run your stress test
            results = runner.run_test()

        metrics = monitor.get_metrics()
        print(f"Avg CPU: {metrics.avg_cpu_percent:.1f}%")

    Manual control::

        monitor = PerformanceMonitor()
        monitor.start()
        # ... run tests ...
        monitor.stop()
        metrics = monitor.get_metrics()

Example:
    >>> with PerformanceMonitor() as monitor:
    ...     time.sleep(5)  # Do work
    >>> metrics = monitor.get_metrics()
    >>> print(f"Samples: {metrics.sample_count}")
    Samples: 5

Note:
    For best results, install psutil: pip install psutil

Author:
    VULCAN-AGI Team

Version:
    1.0.0

License:
    Proprietary - VULCAN AGI Project

See Also:
    - tests.performance.scalability_stress_test: Main stress test module
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Optional, Tuple, TYPE_CHECKING

# Configure logging
logger = logging.getLogger(__name__)

# Module metadata
__version__: Final[str] = "1.0.0"
__author__: Final[str] = "VULCAN-AGI Team"

# Constants
DEFAULT_SAMPLE_INTERVAL: Final[float] = 1.0
DEFAULT_CPU_SPIKE_THRESHOLD: Final[float] = 90.0
DEFAULT_MEMORY_SPIKE_THRESHOLD_MB: Final[float] = 1000.0
MEMORY_GROWTH_THRESHOLD: Final[float] = 1.5  # 50% growth triggers anomaly


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass(frozen=False)
class SystemSnapshot:
    """
    A snapshot of system metrics at a point in time.

    This dataclass captures all measurable system state at a single
    moment, enabling time-series analysis of resource utilization.

    Attributes:
        timestamp: Unix timestamp when snapshot was taken.
        cpu_percent_per_core: CPU utilization per logical core.
        cpu_percent_total: Overall CPU utilization (0-100).
        memory_rss_mb: Resident Set Size in megabytes.
        memory_vms_mb: Virtual Memory Size in megabytes.
        memory_percent: Memory usage as percentage of total.
        thread_count: Number of active threads.
        process_count: Number of child processes (including self).
        sqlite_file_size_mb: SQLite database file size in MB.
        sqlite_wal_size_mb: SQLite WAL file size in MB.

    Example:
        >>> snapshot = SystemSnapshot(timestamp=time.time())
        >>> snapshot.cpu_percent_total = 45.5
        >>> print(snapshot.to_dict()["cpu_percent_total"])
        45.5
    """

    timestamp: float
    cpu_percent_per_core: List[float] = field(default_factory=list)
    cpu_percent_total: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0
    memory_percent: float = 0.0
    thread_count: int = 0
    process_count: int = 0
    sqlite_file_size_mb: float = 0.0
    sqlite_wal_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all snapshot fields.
        """
        return {
            "timestamp": round(self.timestamp, 3),
            "cpu_percent_per_core": [round(c, 1) for c in self.cpu_percent_per_core],
            "cpu_percent_total": round(self.cpu_percent_total, 1),
            "memory_rss_mb": round(self.memory_rss_mb, 2),
            "memory_vms_mb": round(self.memory_vms_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "thread_count": self.thread_count,
            "process_count": self.process_count,
            "sqlite_file_size_mb": round(self.sqlite_file_size_mb, 3),
            "sqlite_wal_size_mb": round(self.sqlite_wal_size_mb, 3),
        }


# ============================================================
# SYSTEM METRICS COLLECTOR
# ============================================================


class SystemMetricsCollector:
    """
    Collects system-level performance metrics.

    Uses psutil for detailed metrics when available, falling back to
    basic OS-level metrics when psutil is not installed.

    Attributes:
        sqlite_db_path: Optional path to SQLite database to monitor.

    Thread Safety:
        This class is thread-safe. Snapshot collection can be called
        from any thread.

    Example:
        >>> collector = SystemMetricsCollector("/path/to/db.sqlite")
        >>> snapshot = collector.collect_snapshot()
        >>> print(f"CPU: {snapshot.cpu_percent_total}%")
    """

    def __init__(self, sqlite_db_path: Optional[str] = None) -> None:
        """
        Initialize the metrics collector.

        Args:
            sqlite_db_path: Optional path to SQLite database to monitor.
                           If provided, SQLite file sizes will be tracked.
        """
        self.sqlite_db_path: Optional[str] = sqlite_db_path
        self._psutil_available: bool = False
        self._process: Optional[Any] = None
        self._psutil_module: Optional[Any] = None

        try:
            import psutil
            self._psutil_module = psutil
            self._psutil_available = True
            self._process = psutil.Process()
            logger.debug("psutil available for detailed metrics")
        except ImportError:
            logger.warning(
                "psutil not available - install with 'pip install psutil' "
                "for detailed metrics"
            )

    def collect_snapshot(self) -> SystemSnapshot:
        """
        Collect a snapshot of current system metrics.

        Returns:
            SystemSnapshot with current system state.
        """
        snapshot = SystemSnapshot(timestamp=time.time())

        if self._psutil_available and self._psutil_module:
            psutil = self._psutil_module

            # CPU metrics (non-blocking call)
            try:
                snapshot.cpu_percent_per_core = psutil.cpu_percent(
                    interval=None, percpu=True
                )
                snapshot.cpu_percent_total = psutil.cpu_percent(interval=None)
            except Exception as e:
                logger.debug(f"Error getting CPU metrics: {e}")

            # Memory metrics for current process
            if self._process:
                try:
                    mem_info = self._process.memory_info()
                    snapshot.memory_rss_mb = mem_info.rss / (1024 * 1024)
                    snapshot.memory_vms_mb = mem_info.vms / (1024 * 1024)
                    snapshot.memory_percent = self._process.memory_percent()
                except Exception as e:
                    logger.debug(f"Error getting memory info: {e}")

                # Thread count for current process
                try:
                    snapshot.thread_count = self._process.num_threads()
                except Exception as e:
                    logger.debug(f"Error getting thread count: {e}")

                # Process count (children + self)
                try:
                    children = self._process.children(recursive=True)
                    snapshot.process_count = len(children) + 1
                except Exception as e:
                    logger.debug(f"Error getting process count: {e}")
        else:
            # Basic fallback without psutil
            snapshot.thread_count = threading.active_count()
            snapshot.cpu_percent_total = 0.0
            snapshot.memory_rss_mb = 0.0

        # SQLite metrics (always available)
        if self.sqlite_db_path:
            snapshot.sqlite_file_size_mb = self._get_sqlite_size()
            snapshot.sqlite_wal_size_mb = self._get_sqlite_wal_size()

        return snapshot

    def _get_sqlite_size(self) -> float:
        """
        Get SQLite database file size in MB.

        Returns:
            File size in megabytes, or 0.0 if file doesn't exist.
        """
        if not self.sqlite_db_path:
            return 0.0
        try:
            if os.path.exists(self.sqlite_db_path):
                return os.path.getsize(self.sqlite_db_path) / (1024 * 1024)
        except (OSError, IOError) as e:
            logger.debug(f"Error getting SQLite size: {e}")
        return 0.0

    def _get_sqlite_wal_size(self) -> float:
        """
        Get SQLite WAL file size in MB.

        Returns:
            WAL file size in megabytes, or 0.0 if file doesn't exist.
        """
        if not self.sqlite_db_path:
            return 0.0
        try:
            wal_path = f"{self.sqlite_db_path}-wal"
            if os.path.exists(wal_path):
                return os.path.getsize(wal_path) / (1024 * 1024)
        except (OSError, IOError) as e:
            logger.debug(f"Error getting SQLite WAL size: {e}")
        return 0.0


# ============================================================
# MONITORING RESULT
# ============================================================


@dataclass
class MonitoringResult:
    """
    Result of a monitoring session.

    Contains all collected snapshots, aggregate metrics, and detected anomalies.

    Attributes:
        start_time: Unix timestamp when monitoring started.
        end_time: Unix timestamp when monitoring stopped.
        duration_seconds: Total monitoring duration.
        snapshots: List of all collected SystemSnapshots.
        sample_count: Number of samples collected.
        anomalies: List of detected anomalies (CPU spikes, memory leaks).
        avg_cpu_percent: Average CPU utilization.
        max_cpu_percent: Maximum CPU utilization observed.
        avg_memory_rss_mb: Average memory usage.
        max_memory_rss_mb: Maximum memory usage observed.
        avg_thread_count: Average thread count.
        max_thread_count: Maximum thread count observed.

    Example:
        >>> result = monitor.get_metrics()
        >>> print(f"Duration: {result.duration_seconds:.1f}s")
        >>> print(f"Avg CPU: {result.avg_cpu_percent:.1f}%")
    """

    start_time: float
    end_time: float
    duration_seconds: float
    snapshots: List[SystemSnapshot]
    sample_count: int
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregated metrics
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    min_cpu_percent: float = 0.0
    avg_memory_rss_mb: float = 0.0
    max_memory_rss_mb: float = 0.0
    min_memory_rss_mb: float = 0.0
    avg_thread_count: float = 0.0
    max_thread_count: int = 0
    min_thread_count: int = 0
    memory_growth_mb: float = 0.0

    def calculate_aggregates(self) -> None:
        """
        Calculate aggregate metrics from snapshots.

        Should be called after monitoring completes to compute
        statistical summaries.
        """
        if not self.snapshots:
            return

        cpu_values = [s.cpu_percent_total for s in self.snapshots]
        memory_values = [s.memory_rss_mb for s in self.snapshots]
        thread_values = [s.thread_count for s in self.snapshots]

        self.avg_cpu_percent = statistics.mean(cpu_values)
        self.max_cpu_percent = max(cpu_values)
        self.min_cpu_percent = min(cpu_values)

        self.avg_memory_rss_mb = statistics.mean(memory_values)
        self.max_memory_rss_mb = max(memory_values)
        self.min_memory_rss_mb = min(memory_values)

        self.avg_thread_count = statistics.mean(thread_values)
        self.max_thread_count = max(thread_values)
        self.min_thread_count = min(thread_values)

        # Calculate memory growth
        if len(memory_values) >= 2:
            self.memory_growth_mb = memory_values[-1] - memory_values[0]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all metrics, suitable for JSON encoding.
        """
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_count": self.sample_count,
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "max_cpu_percent": round(self.max_cpu_percent, 2),
            "min_cpu_percent": round(self.min_cpu_percent, 2),
            "avg_memory_rss_mb": round(self.avg_memory_rss_mb, 2),
            "max_memory_rss_mb": round(self.max_memory_rss_mb, 2),
            "min_memory_rss_mb": round(self.min_memory_rss_mb, 2),
            "avg_thread_count": round(self.avg_thread_count, 2),
            "max_thread_count": self.max_thread_count,
            "min_thread_count": self.min_thread_count,
            "memory_growth_mb": round(self.memory_growth_mb, 2),
            "anomaly_count": len(self.anomalies),
            "anomalies": self.anomalies[:20],  # Limit to first 20
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


# ============================================================
# PERFORMANCE MONITOR
# ============================================================


class PerformanceMonitor:
    """
    Performance monitoring utility for stress tests.

    Implements a context manager interface for easy integration
    and provides background metric collection with anomaly detection.

    Thread Safety:
        All public methods are thread-safe.

    Attributes:
        sample_interval: Time between samples in seconds.
        sqlite_db_path: Path to SQLite database being monitored.
        cpu_spike_threshold: CPU percentage considered a spike.
        memory_spike_threshold_mb: Memory MB considered a spike.

    Example:
        >>> with PerformanceMonitor() as monitor:
        ...     # Run stress test
        ...     results = runner.run_test()
        >>> metrics = monitor.get_metrics()
        >>> print(f"Samples: {metrics.sample_count}")
    """

    def __init__(
        self,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL,
        sqlite_db_path: Optional[str] = None,
        cpu_spike_threshold: float = DEFAULT_CPU_SPIKE_THRESHOLD,
        memory_spike_threshold_mb: float = DEFAULT_MEMORY_SPIKE_THRESHOLD_MB,
    ) -> None:
        """
        Initialize the performance monitor.

        Args:
            sample_interval_seconds: Interval between samples (default: 1.0).
            sqlite_db_path: Optional path to SQLite database to monitor.
            cpu_spike_threshold: CPU percentage to consider a spike (default: 90.0).
            memory_spike_threshold_mb: Memory MB to consider a spike (default: 1000.0).
        """
        self.sample_interval: float = max(0.1, sample_interval_seconds)
        self.sqlite_db_path: Optional[str] = sqlite_db_path
        self.cpu_spike_threshold: float = cpu_spike_threshold
        self.memory_spike_threshold_mb: float = memory_spike_threshold_mb

        self._collector: SystemMetricsCollector = SystemMetricsCollector(sqlite_db_path)
        self._snapshots: List[SystemSnapshot] = []
        self._anomalies: List[Dict[str, Any]] = []
        self._monitoring: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._lock: threading.RLock = threading.RLock()
    
    def __enter__(self) -> "PerformanceMonitor":
        """
        Start monitoring when entering context.

        Returns:
            Self reference for context manager usage.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Stop monitoring when exiting context.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Traceback if an exception occurred.
        """
        self.stop()

    def start(self) -> None:
        """
        Start background monitoring.

        Idempotent - calling multiple times is safe.
        """
        with self._lock:
            if self._monitoring:
                logger.debug("Monitoring already running")
                return

            self._monitoring = True
            self._stop_event.clear()
            self._snapshots = []
            self._anomalies = []
            self._start_time = time.time()

            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="PerformanceMonitor",
            )
            self._monitor_thread.start()
            logger.debug("Performance monitoring started")

    def stop(self) -> None:
        """
        Stop background monitoring.

        Idempotent - calling multiple times is safe.
        """
        with self._lock:
            if not self._monitoring:
                return

            self._stop_event.set()
            self._end_time = time.time()

        # Join outside lock to avoid deadlock
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        with self._lock:
            self._monitoring = False

        logger.debug("Performance monitoring stopped")

    def _monitoring_loop(self) -> None:
        """
        Background thread that collects metrics at intervals.

        This method runs in a daemon thread and collects snapshots
        until stop() is called.
        """
        while not self._stop_event.is_set():
            try:
                snapshot = self._collector.collect_snapshot()

                with self._lock:
                    self._snapshots.append(snapshot)
                    self._detect_anomalies(snapshot)

            except Exception as e:
                logger.debug(f"Error collecting metrics: {e}")

            # Wait for next sample interval
            self._stop_event.wait(timeout=self.sample_interval)

    def _detect_anomalies(self, snapshot: SystemSnapshot) -> None:
        """
        Detect anomalies in the current snapshot.

        Checks for:
        - CPU spikes (above threshold)
        - Memory spikes (above threshold)
        - Memory growth (sudden increases)

        Args:
            snapshot: Current system snapshot to analyze.
        """
        timestamp_iso = datetime.fromtimestamp(
            snapshot.timestamp, tz=timezone.utc
        ).isoformat()

        # CPU spike detection
        if snapshot.cpu_percent_total > self.cpu_spike_threshold:
            self._anomalies.append({
                "type": "cpu_spike",
                "timestamp": timestamp_iso,
                "timestamp_unix": snapshot.timestamp,
                "value": round(snapshot.cpu_percent_total, 1),
                "threshold": self.cpu_spike_threshold,
                "severity": "warning" if snapshot.cpu_percent_total < 95 else "critical",
            })

        # Memory spike detection
        if snapshot.memory_rss_mb > self.memory_spike_threshold_mb:
            self._anomalies.append({
                "type": "memory_spike",
                "timestamp": timestamp_iso,
                "timestamp_unix": snapshot.timestamp,
                "value": round(snapshot.memory_rss_mb, 2),
                "threshold": self.memory_spike_threshold_mb,
                "severity": "warning",
            })

        # Memory growth detection (compare with average of last 10 samples)
        if len(self._snapshots) > 10:
            recent_avg = statistics.mean(
                s.memory_rss_mb for s in self._snapshots[-10:]
            )
            if snapshot.memory_rss_mb > recent_avg * MEMORY_GROWTH_THRESHOLD:
                self._anomalies.append({
                    "type": "memory_growth",
                    "timestamp": timestamp_iso,
                    "timestamp_unix": snapshot.timestamp,
                    "current": round(snapshot.memory_rss_mb, 2),
                    "recent_avg": round(recent_avg, 2),
                    "growth_factor": round(snapshot.memory_rss_mb / recent_avg, 2),
                    "severity": "warning",
                })

    def get_metrics(self) -> MonitoringResult:
        """
        Get the collected metrics and analysis.

        Returns:
            MonitoringResult with all collected data and computed aggregates.
        """
        with self._lock:
            snapshots = list(self._snapshots)
            anomalies = list(self._anomalies)

        end_time = self._end_time or time.time()
        duration = end_time - self._start_time if self._start_time > 0 else 0.0

        result = MonitoringResult(
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=duration,
            snapshots=snapshots,
            sample_count=len(snapshots),
            anomalies=anomalies,
        )
        result.calculate_aggregates()

        return result

    def get_current_snapshot(self) -> SystemSnapshot:
        """
        Get a single snapshot of current metrics.

        Returns:
            SystemSnapshot with current system state.
        """
        return self._collector.collect_snapshot()

    def export_to_json(self, output_path: str) -> None:
        """
        Export metrics to JSON file.

        Args:
            output_path: Path to save JSON file.
        """
        metrics = self.get_metrics()

        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics exported to {output_path}")

    def export_time_series(self, output_path: str) -> None:
        """
        Export time-series data for graphing.

        Creates a CSV file suitable for visualization tools.

        Args:
            output_path: Path to save CSV file.
        """
        with self._lock:
            snapshots = list(self._snapshots)

        if not snapshots:
            logger.warning("No snapshots to export")
            return

        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write header
            f.write(
                "timestamp,datetime,cpu_percent,memory_rss_mb,memory_vms_mb,"
                "thread_count,process_count,sqlite_size_mb,sqlite_wal_mb\n"
            )

            # Write data
            for snapshot in snapshots:
                dt = datetime.fromtimestamp(snapshot.timestamp, tz=timezone.utc)
                f.write(
                    f"{snapshot.timestamp:.3f},"
                    f"{dt.isoformat()},"
                    f"{snapshot.cpu_percent_total:.1f},"
                    f"{snapshot.memory_rss_mb:.2f},"
                    f"{snapshot.memory_vms_mb:.2f},"
                    f"{snapshot.thread_count},"
                    f"{snapshot.process_count},"
                    f"{snapshot.sqlite_file_size_mb:.3f},"
                    f"{snapshot.sqlite_wal_size_mb:.3f}\n"
                )

        logger.info(f"Time series exported to {output_path}")


# ============================================================
# SQLITE PERFORMANCE MONITOR
# ============================================================


class SQLitePerformanceMonitor:
    """
    Monitors SQLite WAL mode performance during concurrent writes.
    
    Tracks:
    - Write latency
    - WAL file growth
    - Checkpoint frequency
    - Lock contentions
    """
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite monitor.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._write_latencies: List[float] = []
        self._checkpoints: int = 0
        self._lock_contentions: int = 0
        self._lock = threading.Lock()
    
    def record_write(self, latency_seconds: float) -> None:
        """Record a write operation latency."""
        with self._lock:
            self._write_latencies.append(latency_seconds)
    
    def record_checkpoint(self) -> None:
        """Record a WAL checkpoint."""
        with self._lock:
            self._checkpoints += 1
    
    def record_lock_contention(self) -> None:
        """Record a lock contention event."""
        with self._lock:
            self._lock_contentions += 1
    
    def get_wal_info(self) -> Dict[str, Any]:
        """Get WAL file information."""
        wal_path = f"{self.db_path}-wal"
        shm_path = f"{self.db_path}-shm"
        
        return {
            "db_exists": os.path.exists(self.db_path),
            "db_size_mb": (
                os.path.getsize(self.db_path) / (1024 * 1024)
                if os.path.exists(self.db_path) else 0.0
            ),
            "wal_exists": os.path.exists(wal_path),
            "wal_size_mb": (
                os.path.getsize(wal_path) / (1024 * 1024)
                if os.path.exists(wal_path) else 0.0
            ),
            "shm_exists": os.path.exists(shm_path),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collected statistics."""
        with self._lock:
            latencies = list(self._write_latencies)
        
        if not latencies:
            return {
                "write_count": 0,
                "checkpoints": self._checkpoints,
                "lock_contentions": self._lock_contentions,
            }
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "write_count": n,
            "avg_latency_ms": sum(latencies) / n * 1000,
            "min_latency_ms": sorted_latencies[0] * 1000,
            "max_latency_ms": sorted_latencies[-1] * 1000,
            "p95_latency_ms": sorted_latencies[int(n * 0.95)] * 1000,
            "checkpoints": self._checkpoints,
            "lock_contentions": self._lock_contentions,
            "wal_info": self.get_wal_info(),
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def run_with_monitoring(
    func,
    *args,
    sample_interval: float = 1.0,
    **kwargs,
) -> Tuple[Any, MonitoringResult]:
    """
    Run a function with performance monitoring.
    
    Args:
        func: Function to run
        *args: Arguments for function
        sample_interval: Monitoring sample interval
        **kwargs: Keyword arguments for function
        
    Returns:
        Tuple of (function result, monitoring result)
    """
    with PerformanceMonitor(sample_interval_seconds=sample_interval) as monitor:
        result = func(*args, **kwargs)
    
    return result, monitor.get_metrics()


def quick_system_check() -> Dict[str, Any]:
    """
    Perform a quick system health check.
    
    Returns:
        Dictionary with current system status
    """
    collector = SystemMetricsCollector()
    snapshot = collector.collect_snapshot()
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu_percent": snapshot.cpu_percent_total,
        "memory_rss_mb": snapshot.memory_rss_mb,
        "memory_percent": snapshot.memory_percent,
        "thread_count": snapshot.thread_count,
        "status": "healthy" if snapshot.cpu_percent_total < 90 else "stressed",
    }


# ============================================================
# MAIN (for testing)
# ============================================================


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VULCAN Performance Monitor")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Monitoring duration in seconds",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample interval in seconds",
    )
    parser.add_argument(
        "--output",
        default="performance_metrics.json",
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    print(f"Monitoring system for {args.duration} seconds...")
    
    with PerformanceMonitor(sample_interval_seconds=args.interval) as monitor:
        time.sleep(args.duration)
    
    metrics = monitor.get_metrics()
    
    print(f"\nMonitoring Results:")
    print(f"  Duration: {metrics.duration_seconds:.1f}s")
    print(f"  Samples: {metrics.sample_count}")
    print(f"  Avg CPU: {metrics.avg_cpu_percent:.1f}%")
    print(f"  Max CPU: {metrics.max_cpu_percent:.1f}%")
    print(f"  Avg Memory: {metrics.avg_memory_rss_mb:.1f}MB")
    print(f"  Max Memory: {metrics.max_memory_rss_mb:.1f}MB")
    print(f"  Avg Threads: {metrics.avg_thread_count:.1f}")
    print(f"  Max Threads: {metrics.max_thread_count}")
    
    if metrics.anomalies:
        print(f"\nAnomalies detected: {len(metrics.anomalies)}")
        for anomaly in metrics.anomalies[:5]:
            print(f"  - {anomaly['type']}: {anomaly.get('value', 'N/A')}")
    
    monitor.export_to_json(args.output)
    print(f"\nResults saved to {args.output}")
