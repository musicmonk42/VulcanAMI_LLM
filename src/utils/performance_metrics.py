"""
Performance Metrics Module

Tracks and compares performance of fallback implementations vs. full implementations.
Provides runtime metrics for:
- Groth16 SNARK proofs (full vs. hash-based)
- spaCy NLP (full vs. TF-IDF)
- FAISS vector search (AVX-512 vs. AVX2 vs. NEON)
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement"""

    operation: str
    implementation: str  # e.g., "full", "fallback", "AVX512", "AVX2"
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    details: Dict = field(default_factory=dict)


class PerformanceTracker:
    """Tracks performance metrics across the system"""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(
        self,
        operation: str,
        implementation: str,
        duration_ms: float,
        success: bool = True,
        **details,
    ) -> None:
        """Record a performance metric"""
        metric = PerformanceMetric(
            operation=operation,
            implementation=implementation,
            duration_ms=duration_ms,
            success=success,
            details=details,
        )

        key = f"{operation}:{implementation}"

        with self._lock:
            self.metrics[key].append(metric)
            self.operation_counts[operation] += 1

    def get_stats(self, operation: str, implementation: str) -> Optional[Dict]:
        """Get statistics for a specific operation and implementation"""
        key = f"{operation}:{implementation}"

        with self._lock:
            if key not in self.metrics or not self.metrics[key]:
                return None
            # Copy to avoid holding lock during computation
            all_metrics = list(self.metrics[key])

        # Compute statistics outside the lock
        successful = [m.duration_ms for m in all_metrics if m.success]
        total = len(all_metrics)

        if not successful:
            return {"count": 0, "total_attempts": total, "failure_rate": 1.0}

        sorted_durations = sorted(successful)
        n = len(sorted_durations)

        stats = {
            "count": n,
            "mean_ms": statistics.mean(successful),
            "median_ms": statistics.median(successful),
            "min_ms": min(successful),
            "max_ms": max(successful),
            "stdev_ms": statistics.stdev(successful) if n > 1 else 0,
            "total_attempts": total,
            "failure_rate": (total - n) / total if total > 0 else 0.0,
        }

        # Add percentiles if enough samples
        # Use proper percentile calculation to avoid truncation errors
        if n >= 20:
            p95_index = min(int(n * 0.95), n - 1)
            stats["p95_ms"] = sorted_durations[p95_index]
        if n >= 100:
            p99_index = min(int(n * 0.99), n - 1)
            stats["p99_ms"] = sorted_durations[p99_index]

        return stats

    def compare_implementations(self, operation: str) -> Dict:
        """Compare all implementations for a given operation"""
        implementations = {}

        for key in self.metrics:
            if key.startswith(f"{operation}:"):
                impl = key.split(":", 1)[1]
                stats = self.get_stats(operation, impl)
                if stats:
                    implementations[impl] = stats

        if not implementations:
            return {}

        # Calculate relative performance
        if "full" in implementations and "fallback" in implementations:
            full_mean = implementations["full"]["mean_ms"]
            fallback_mean = implementations["fallback"]["mean_ms"]

            # Handle edge cases properly
            if full_mean < 0:
                # Negative duration indicates an error
                logger.warning(f"Negative full_mean duration: {full_mean}")
                full_mean = 0

            if full_mean == 0:
                # When full_mean is zero, calculate slowdown appropriately
                if fallback_mean > 0:
                    slowdown_factor = float("inf")
                    slower_by_percent = float("inf")
                else:
                    # Both are zero
                    slowdown_factor = 1.0
                    slower_by_percent = 0.0
            else:
                slowdown_factor = fallback_mean / full_mean
                slower_by_percent = (slowdown_factor - 1.0) * 100

            return {
                "implementations": implementations,
                "comparison": {
                    "fallback_slowdown_factor": slowdown_factor,
                    "fallback_slower_by_ms": fallback_mean - full_mean,
                    "fallback_slower_by_percent": slower_by_percent,
                },
            }

        return {"implementations": implementations}

    def get_summary(self) -> Dict:
        """Get overall performance summary"""
        summary = {}

        operations = set(key.split(":", 1)[0] for key in self.metrics.keys())

        for op in operations:
            comparison = self.compare_implementations(op)
            if comparison:
                summary[op] = comparison

        return summary

    def format_report(self) -> str:
        """Generate a human-readable performance report"""
        summary = self.get_summary()

        if not summary:
            return "No performance data collected yet."

        lines = ["=" * 70, "PERFORMANCE METRICS REPORT", "=" * 70]

        for operation, data in summary.items():
            lines.append(f"\n{operation}:")

            if "implementations" in data:
                for impl, stats in data["implementations"].items():
                    lines.append(f"  {impl}:")
                    lines.append(f"    Mean: {stats['mean_ms']:.2f} ms")
                    lines.append(f"    Median: {stats['median_ms']:.2f} ms")
                    lines.append(
                        f"    Range: {stats['min_ms']:.2f} - {stats['max_ms']:.2f} ms"
                    )
                    lines.append(f"    Samples: {stats['count']}")
                    if "p95_ms" in stats:
                        lines.append(f"    P95: {stats['p95_ms']:.2f} ms")
                    if "p99_ms" in stats:
                        lines.append(f"    P99: {stats['p99_ms']:.2f} ms")
                    if stats.get("total_attempts", 0) > 0:
                        lines.append(f"    Failure rate: {stats['failure_rate']:.1%}")

            if "comparison" in data:
                comp = data["comparison"]
                lines.append(f"  Comparison:")
                if comp["fallback_slowdown_factor"] == float("inf"):
                    lines.append(f"    Fallback is infinitely slower (full_mean is 0)")
                    lines.append(
                        f"    Fallback is slower by {comp['fallback_slower_by_ms']:.2f} ms"
                    )
                else:
                    lines.append(
                        f"    Fallback is {comp['fallback_slowdown_factor']:.2f}x slower"
                    )
                    lines.append(
                        f"    Fallback is slower by {comp['fallback_slower_by_ms']:.2f} ms"
                    )
                    if comp["fallback_slower_by_percent"] != float("inf"):
                        lines.append(
                            f"    Fallback is slower by {comp['fallback_slower_by_percent']:.1f}%"
                        )

        lines.append("=" * 70)
        return "\n".join(lines)

    def clear(self, operation: Optional[str] = None) -> None:
        """Clear metrics, optionally for a specific operation only"""
        with self._lock:
            if operation is None:
                self.metrics.clear()
                self.operation_counts.clear()
            else:
                keys_to_remove = [
                    k for k in self.metrics if k.startswith(f"{operation}:")
                ]
                for key in keys_to_remove:
                    del self.metrics[key]
                self.operation_counts.pop(operation, None)


# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance with thread-safety"""
    global _performance_tracker
    if _performance_tracker is None:
        with _tracker_lock:
            # Double-check pattern to avoid race conditions
            if _performance_tracker is None:
                _performance_tracker = PerformanceTracker()
    return _performance_tracker


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, operation: str, implementation: str, **details):
        self.operation = operation
        self.implementation = implementation
        self.details = details
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.success = exc_type is None

        tracker = get_performance_tracker()
        tracker.record(
            self.operation,
            self.implementation,
            duration_ms,
            self.success,
            **self.details,
        )

        return False  # Don't suppress exceptions

    def __call__(self, func):
        """Use as decorator"""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(self.operation, self.implementation, **self.details):
                return func(*args, **kwargs)

        return wrapper


def log_performance_summary():
    """Log performance summary to logger"""
    tracker = get_performance_tracker()
    report = tracker.format_report()
    logger.info("\n" + report)


# Module-specific performance tracking functions


def track_zk_proof_generation(implementation: str = "fallback"):
    """Decorator/context manager for tracking ZK proof generation performance"""
    return PerformanceTimer("zk_proof_generation", implementation)


def track_analogical_reasoning(implementation: str = "tfidf"):
    """Decorator/context manager for tracking analogical reasoning performance"""
    return PerformanceTimer("analogical_reasoning", implementation)


def track_faiss_search(implementation: str = "avx2"):
    """Decorator/context manager for tracking FAISS search performance"""
    return PerformanceTimer("faiss_search", implementation)
