"""
Performance Instrumentation Utilities for VULCAN

This module provides decorators and utilities for tracking and logging
performance metrics across the token generation pipeline.
"""

import functools
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationPerformanceMetrics:
    """Container for generation-specific performance metrics.
    
    Named specifically to avoid conflicts with other PerformanceMetrics classes
    in the codebase (e.g., in problem_decomposer, stress_tests, etc.).
    """
    
    # Timing statistics (in milliseconds)
    total_encode_time_ms: float = 0.0
    total_logits_time_ms: float = 0.0
    total_sample_time_ms: float = 0.0
    total_context_time_ms: float = 0.0
    total_world_model_time_ms: float = 0.0
    
    # Cache statistics
    encoding_cache_hits: int = 0
    encoding_cache_misses: int = 0
    logits_cache_hits: int = 0
    logits_cache_misses: int = 0
    
    # Token generation statistics
    tokens_generated: int = 0
    generation_errors: int = 0
    
    # Per-operation timing history (for percentile calculation)
    encode_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    logits_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    sample_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def record_encode_time(self, time_ms: float) -> None:
        """Record an encode operation time."""
        self.total_encode_time_ms += time_ms
        self.encode_times.append(time_ms)
    
    def record_logits_time(self, time_ms: float) -> None:
        """Record a logits computation time."""
        self.total_logits_time_ms += time_ms
        self.logits_times.append(time_ms)
    
    def record_sample_time(self, time_ms: float) -> None:
        """Record a sampling time."""
        self.total_sample_time_ms += time_ms
        self.sample_times.append(time_ms)
    
    def get_percentile(self, times: deque, percentile: float) -> float:
        """Calculate percentile from timing history."""
        if not times:
            return 0.0
        sorted_times = sorted(times)
        idx = int(len(sorted_times) * percentile / 100.0)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        total_tokens = max(1, self.tokens_generated)
        
        # Calculate cache hit rates
        encode_total = self.encoding_cache_hits + self.encoding_cache_misses
        logits_total = self.logits_cache_hits + self.logits_cache_misses
        
        return {
            "tokens_generated": self.tokens_generated,
            "errors": self.generation_errors,
            "timing": {
                "total_encode_ms": self.total_encode_time_ms,
                "total_logits_ms": self.total_logits_time_ms,
                "total_sample_ms": self.total_sample_time_ms,
                "total_context_ms": self.total_context_time_ms,
                "avg_encode_ms": self.total_encode_time_ms / total_tokens,
                "avg_logits_ms": self.total_logits_time_ms / total_tokens,
                "avg_sample_ms": self.total_sample_time_ms / total_tokens,
                "p50_encode_ms": self.get_percentile(self.encode_times, 50),
                "p95_encode_ms": self.get_percentile(self.encode_times, 95),
                "p99_encode_ms": self.get_percentile(self.encode_times, 99),
            },
            "cache": {
                "encoding_hit_rate": self.encoding_cache_hits / encode_total if encode_total > 0 else 0.0,
                "logits_hit_rate": self.logits_cache_hits / logits_total if logits_total > 0 else 0.0,
            },
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_encode_time_ms = 0.0
        self.total_logits_time_ms = 0.0
        self.total_sample_time_ms = 0.0
        self.total_context_time_ms = 0.0
        self.total_world_model_time_ms = 0.0
        self.encoding_cache_hits = 0
        self.encoding_cache_misses = 0
        self.logits_cache_hits = 0
        self.logits_cache_misses = 0
        self.tokens_generated = 0
        self.generation_errors = 0
        self.encode_times.clear()
        self.logits_times.clear()
        self.sample_times.clear()


def timed(name: str, threshold_ms: float = 100.0):
    """
    Decorator to time synchronous functions with diagnostic logging.
    
    Args:
        name: Name of the operation for logging
        threshold_ms: Log warning if operation exceeds this threshold
    
    Example:
        @timed("encode_tokens")
        def encode(tokens):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(f"[PERF] {name}: {elapsed_ms:.1f}ms (exceeds {threshold_ms}ms threshold)")
                else:
                    logger.debug(f"[PERF] {name}: {elapsed_ms:.1f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(f"[PERF] {name}: FAILED after {elapsed_ms:.1f}ms - {e}")
                raise
        return wrapper
    return decorator


def timed_async(name: str, threshold_ms: float = 100.0):
    """
    Decorator to time async functions with diagnostic logging.
    
    Args:
        name: Name of the operation for logging
        threshold_ms: Log warning if operation exceeds this threshold
    
    Example:
        @timed_async("fetch_context")
        async def fetch_context():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(f"[PERF] {name}: {elapsed_ms:.1f}ms (exceeds {threshold_ms}ms threshold)")
                else:
                    logger.debug(f"[PERF] {name}: {elapsed_ms:.1f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(f"[PERF] {name}: FAILED after {elapsed_ms:.1f}ms - {e}")
                raise
        return wrapper
    return decorator


class TimingContext:
    """
    Context manager for timing code blocks.
    
    Example:
        with TimingContext("my_operation") as timer:
            do_something()
        print(f"Took {timer.elapsed_ms}ms")
    """
    
    def __init__(self, name: str, threshold_ms: float = 100.0, log_always: bool = False):
        self.name = name
        self.threshold_ms = threshold_ms
        self.log_always = log_always
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self) -> "TimingContext":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        
        if exc_type is not None:
            logger.error(f"[PERF] {self.name}: FAILED after {self.elapsed_ms:.1f}ms")
        elif self.elapsed_ms > self.threshold_ms:
            logger.warning(f"[PERF] {self.name}: {self.elapsed_ms:.1f}ms (exceeds {self.threshold_ms}ms threshold)")
        elif self.log_always:
            logger.info(f"[PERF] {self.name}: {self.elapsed_ms:.1f}ms")


class GenerationPerformanceTracker:
    """
    Global performance tracker for the generation pipeline.
    
    Provides centralized tracking of performance metrics across
    different components of the system.
    
    Named specifically to avoid conflicts with other PerformanceTracker classes
    in the codebase (e.g., in performance_metrics.py, problem_decomposer, etc.).
    """
    
    _instance: Optional["GenerationPerformanceTracker"] = None
    
    def __new__(cls) -> "GenerationPerformanceTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.metrics = GenerationPerformanceMetrics()
        self._operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._enabled = True
    
    def enable(self) -> None:
        """Enable performance tracking."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable performance tracking."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._enabled
    
    def record(self, operation: str, time_ms: float) -> None:
        """Record an operation timing."""
        if not self._enabled:
            return
        self._operation_times[operation].append(time_ms)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        times = self._operation_times.get(operation)
        if not times:
            return {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        
        sorted_times = sorted(times)
        return {
            "count": len(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p50_ms": sorted_times[len(sorted_times) // 2],
            "p95_ms": sorted_times[int(len(sorted_times) * 0.95)],
            "p99_ms": sorted_times[int(len(sorted_times) * 0.99)],
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked operations."""
        return {op: self.get_stats(op) for op in self._operation_times}
    
    def reset(self) -> None:
        """Reset all tracked data."""
        self._operation_times.clear()
        self.metrics.reset()


# Global tracker instance
_tracker = GenerationPerformanceTracker()


def get_generation_performance_tracker() -> GenerationPerformanceTracker:
    """Get the global generation performance tracker instance."""
    return _tracker


__all__ = [
    "GenerationPerformanceMetrics",
    "GenerationPerformanceTracker",
    "TimingContext",
    "get_generation_performance_tracker",
    "timed",
    "timed_async",
]
