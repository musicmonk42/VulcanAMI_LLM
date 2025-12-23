"""
Performance Instrumentation Utilities for VULCAN.

This module provides decorators, context managers, and utilities for tracking
and logging performance metrics across the token generation pipeline.

Features:
    - Thread-safe metrics collection
    - Configurable timing thresholds with warning/debug logging
    - Percentile-based statistics (p50, p95, p99)
    - Singleton pattern for global tracker access
    - Cache hit rate tracking

Example:
    >>> from src.utils.performance_instrumentation import timed, TimingContext
    >>>
    >>> @timed("my_function", threshold_ms=50.0)
    ... def my_function():
    ...     pass
    >>>
    >>> with TimingContext("my_block") as timer:
    ...     pass
    >>> print(timer.elapsed_ms)
"""

import functools
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic callable wrapper
F = TypeVar("F", bound=Callable[..., Any])

# Module constants
DEFAULT_THRESHOLD_MS = 100.0
DEFAULT_HISTORY_SIZE = 1000
PERCENTILE_50 = 50.0
PERCENTILE_95 = 95.0
PERCENTILE_99 = 99.0


@dataclass
class GenerationPerformanceMetrics:
    """Container for generation-specific performance metrics.

    Thread-safe container for tracking timing statistics, cache hit rates,
    and token generation metrics. Named specifically to avoid conflicts with
    other PerformanceMetrics classes in the codebase.

    Attributes:
        total_encode_time_ms: Cumulative encoding time in milliseconds.
        total_logits_time_ms: Cumulative logits computation time in milliseconds.
        total_sample_time_ms: Cumulative sampling time in milliseconds.
        total_context_time_ms: Cumulative context retrieval time in milliseconds.
        total_world_model_time_ms: Cumulative world model update time in ms.
        encoding_cache_hits: Number of encoding cache hits.
        encoding_cache_misses: Number of encoding cache misses.
        logits_cache_hits: Number of logits cache hits.
        logits_cache_misses: Number of logits cache misses.
        tokens_generated: Total number of tokens generated.
        generation_errors: Total number of generation errors.
        encode_times: History of individual encode operation times.
        logits_times: History of individual logits computation times.
        sample_times: History of individual sampling times.
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
    encode_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_SIZE)
    )
    logits_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_SIZE)
    )
    sample_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_SIZE)
    )

    # Thread safety
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def record_encode_time(self, time_ms: float) -> None:
        """Record an encode operation time.

        Args:
            time_ms: Time in milliseconds for the encode operation.
        """
        if not isinstance(time_ms, (int, float)) or time_ms < 0:
            logger.warning("Invalid encode time: %s, skipping", time_ms)
            return
        with self._lock:
            self.total_encode_time_ms += time_ms
            self.encode_times.append(time_ms)

    def record_logits_time(self, time_ms: float) -> None:
        """Record a logits computation time.

        Args:
            time_ms: Time in milliseconds for the logits computation.
        """
        if not isinstance(time_ms, (int, float)) or time_ms < 0:
            logger.warning("Invalid logits time: %s, skipping", time_ms)
            return
        with self._lock:
            self.total_logits_time_ms += time_ms
            self.logits_times.append(time_ms)

    def record_sample_time(self, time_ms: float) -> None:
        """Record a sampling time.

        Args:
            time_ms: Time in milliseconds for the sampling operation.
        """
        if not isinstance(time_ms, (int, float)) or time_ms < 0:
            logger.warning("Invalid sample time: %s, skipping", time_ms)
            return
        with self._lock:
            self.total_sample_time_ms += time_ms
            self.sample_times.append(time_ms)

    def get_percentile(self, times: Deque[float], percentile: float) -> float:
        """Calculate percentile from timing history.

        Args:
            times: Deque of timing values in milliseconds.
            percentile: Percentile to calculate (0-100).

        Returns:
            The percentile value, or 0.0 if no data available.
        """
        if not times:
            return 0.0
        if not 0 <= percentile <= 100:
            logger.warning("Invalid percentile %s, clamping to [0, 100]", percentile)
            percentile = max(0.0, min(100.0, percentile))
        with self._lock:
            sorted_times = sorted(times)
        idx = int(len(sorted_times) * percentile / 100.0)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary containing token counts, timing statistics,
            and cache hit rates.
        """
        with self._lock:
            total_tokens = max(1, self.tokens_generated)

            # Calculate cache hit rates
            encode_total = self.encoding_cache_hits + self.encoding_cache_misses
            logits_total = self.logits_cache_hits + self.logits_cache_misses

            encoding_hit_rate = (
                self.encoding_cache_hits / encode_total if encode_total > 0 else 0.0
            )
            logits_hit_rate = (
                self.logits_cache_hits / logits_total if logits_total > 0 else 0.0
            )

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
                    "p50_encode_ms": self.get_percentile(
                        self.encode_times, PERCENTILE_50
                    ),
                    "p95_encode_ms": self.get_percentile(
                        self.encode_times, PERCENTILE_95
                    ),
                    "p99_encode_ms": self.get_percentile(
                        self.encode_times, PERCENTILE_99
                    ),
                },
                "cache": {
                    "encoding_hit_rate": encoding_hit_rate,
                    "logits_hit_rate": logits_hit_rate,
                },
            }

    def reset(self) -> None:
        """Reset all metrics to initial values."""
        with self._lock:
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


def timed(name: str, threshold_ms: float = DEFAULT_THRESHOLD_MS) -> Callable[[F], F]:
    """Decorator to time synchronous functions with diagnostic logging.

    Logs a warning if the operation exceeds the threshold, otherwise logs
    at debug level. On exception, logs an error with the elapsed time.

    Args:
        name: Name of the operation for logging.
        threshold_ms: Log warning if operation exceeds this threshold.
            Defaults to 100.0 ms.

    Returns:
        Decorator function that wraps the target function.

    Example:
        >>> @timed("encode_tokens", threshold_ms=50.0)
        ... def encode(tokens):
        ...     return process_tokens(tokens)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        "[PERF] %s: %.1fms (exceeds %.1fms threshold)",
                        name,
                        elapsed_ms,
                        threshold_ms,
                    )
                else:
                    logger.debug("[PERF] %s: %.1fms", name, elapsed_ms)
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "[PERF] %s: FAILED after %.1fms - %s", name, elapsed_ms, exc
                )
                raise
        return wrapper  # type: ignore[return-value]
    return decorator


def timed_async(
    name: str, threshold_ms: float = DEFAULT_THRESHOLD_MS
) -> Callable[[F], F]:
    """Decorator to time async functions with diagnostic logging.

    Logs a warning if the operation exceeds the threshold, otherwise logs
    at debug level. On exception, logs an error with the elapsed time.

    Args:
        name: Name of the operation for logging.
        threshold_ms: Log warning if operation exceeds this threshold.
            Defaults to 100.0 ms.

    Returns:
        Decorator function that wraps the target async function.

    Example:
        >>> @timed_async("fetch_context", threshold_ms=200.0)
        ... async def fetch_context():
        ...     return await get_context()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        "[PERF] %s: %.1fms (exceeds %.1fms threshold)",
                        name,
                        elapsed_ms,
                        threshold_ms,
                    )
                else:
                    logger.debug("[PERF] %s: %.1fms", name, elapsed_ms)
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "[PERF] %s: FAILED after %.1fms - %s", name, elapsed_ms, exc
                )
                raise
        return wrapper  # type: ignore[return-value]
    return decorator


class TimingContext:
    """Context manager for timing code blocks.

    Provides a convenient way to measure execution time of code blocks
    with automatic logging based on configurable thresholds.

    Attributes:
        name: Name of the operation for logging.
        threshold_ms: Threshold for warning-level logging.
        log_always: If True, always log (at INFO level when under threshold).
        start_time: Start time of the operation (set on entry).
        elapsed_ms: Elapsed time in milliseconds (set on exit).

    Example:
        >>> with TimingContext("my_operation", threshold_ms=50.0) as timer:
        ...     do_something()
        >>> print(f"Operation took {timer.elapsed_ms:.1f}ms")
    """

    def __init__(
        self,
        name: str,
        threshold_ms: float = DEFAULT_THRESHOLD_MS,
        log_always: bool = False,
    ) -> None:
        """Initialize the timing context.

        Args:
            name: Name of the operation for logging.
            threshold_ms: Log warning if operation exceeds this threshold.
            log_always: If True, always log even if under threshold.
        """
        self.name = name
        self.threshold_ms = threshold_ms
        self.log_always = log_always
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "TimingContext":
        """Enter the context and start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the context, record elapsed time, and log appropriately."""
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            logger.error(
                "[PERF] %s: FAILED after %.1fms", self.name, self.elapsed_ms
            )
        elif self.elapsed_ms > self.threshold_ms:
            logger.warning(
                "[PERF] %s: %.1fms (exceeds %.1fms threshold)",
                self.name,
                self.elapsed_ms,
                self.threshold_ms,
            )
        elif self.log_always:
            logger.info("[PERF] %s: %.1fms", self.name, self.elapsed_ms)


class GenerationPerformanceTracker:
    """Global performance tracker for the generation pipeline.

    Thread-safe singleton that provides centralized tracking of performance
    metrics across different components of the system. Named specifically
    to avoid conflicts with other PerformanceTracker classes in the codebase.

    Features:
        - Singleton pattern ensures single global instance
        - Thread-safe operations with RLock
        - Enable/disable tracking at runtime
        - Percentile-based statistics (p50, p95, p99)

    Example:
        >>> tracker = get_generation_performance_tracker()
        >>> tracker.record("encode", 45.2)
        >>> tracker.record("encode", 52.1)
        >>> stats = tracker.get_stats("encode")
        >>> print(f"Average: {stats['avg_ms']:.1f}ms")
    """

    _instance: Optional["GenerationPerformanceTracker"] = None
    _instance_lock: threading.Lock = threading.Lock()
    _initialized: bool = False  # Class-level default for pylint

    def __new__(cls) -> "GenerationPerformanceTracker":
        """Create or return the singleton instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the tracker (only runs once due to singleton pattern)."""
        # __new__ guarantees _initialized is set before __init__ is called
        if self._initialized:
            return

        self._initialized = True
        self._lock = threading.RLock()
        self.metrics = GenerationPerformanceMetrics()
        self._operation_times: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=DEFAULT_HISTORY_SIZE)
        )
        self._enabled = True

    def enable(self) -> None:
        """Enable performance tracking."""
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable performance tracking."""
        with self._lock:
            self._enabled = False

    def is_enabled(self) -> bool:
        """Check if tracking is enabled.

        Returns:
            True if tracking is enabled, False otherwise.
        """
        with self._lock:
            return self._enabled

    def record(self, operation: str, time_ms: float) -> None:
        """Record an operation timing.

        Args:
            operation: Name of the operation.
            time_ms: Time in milliseconds for the operation.
        """
        with self._lock:
            if not self._enabled:
                return
            if not isinstance(time_ms, (int, float)) or time_ms < 0:
                logger.warning(
                    "Invalid time value for operation '%s': %s", operation, time_ms
                )
                return
            self._operation_times[operation].append(time_ms)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a specific operation.

        Args:
            operation: Name of the operation.

        Returns:
            Dictionary containing count, average, min, max, and percentiles.
        """
        with self._lock:
            times = self._operation_times.get(operation)
            if not times:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                }

            sorted_times = sorted(times)
            length = len(sorted_times)
            return {
                "count": length,
                "avg_ms": sum(times) / length,
                "min_ms": sorted_times[0],
                "max_ms": sorted_times[-1],
                "p50_ms": sorted_times[length // 2],
                "p95_ms": sorted_times[int(length * 0.95)],
                "p99_ms": sorted_times[int(length * 0.99)],
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked operations.

        Returns:
            Dictionary mapping operation names to their statistics.
        """
        with self._lock:
            return {op: self.get_stats(op) for op in self._operation_times}

    def reset(self) -> None:
        """Reset all tracked data."""
        with self._lock:
            self._operation_times.clear()
            self.metrics.reset()


# Global tracker instance - initialized at module load time for thread safety
# Using module-level initialization avoids double-checked locking issues
_tracker: GenerationPerformanceTracker = GenerationPerformanceTracker()


def get_generation_performance_tracker() -> GenerationPerformanceTracker:
    """Get the global generation performance tracker instance.

    Returns:
        The singleton GenerationPerformanceTracker instance.
    """
    return _tracker


class ThrottledLogger:
    """Throttled logger that limits logging frequency for hot paths.

    This class helps reduce logging overhead in performance-critical sections
    by only logging every N calls or every N milliseconds.

    Attributes:
        name: Name prefix for log messages.
        log_interval: Only log every N calls (default: 10).
        time_interval_ms: Only log every N milliseconds (default: 1000).

    Example:
        >>> throttled = ThrottledLogger("step", log_interval=10)
        >>> for i in range(100):
        ...     throttled.debug(f"Processing step {i}")  # Only logs every 10th call
    """

    def __init__(
        self,
        name: str = "",
        log_interval: int = 10,
        time_interval_ms: float = 1000.0,
    ) -> None:
        """Initialize the throttled logger.

        Args:
            name: Name prefix for log messages.
            log_interval: Only log every N calls.
            time_interval_ms: Only log every N milliseconds.
        """
        self.name = name
        self.log_interval = log_interval
        self.time_interval_ms = time_interval_ms
        self._call_count = 0
        self._last_log_time = 0.0
        self._lock = threading.Lock()

    def _should_log(self) -> bool:
        """Check if we should log based on interval or time."""
        with self._lock:
            self._call_count += 1
            current_time = time.time() * 1000  # Convert to ms

            # Log on interval count
            if self._call_count % self.log_interval == 0:
                self._last_log_time = current_time
                return True

            # Log on time interval
            if current_time - self._last_log_time >= self.time_interval_ms:
                self._last_log_time = current_time
                return True

            return False

    def debug(self, message: str) -> None:
        """Log at debug level if throttle allows."""
        if self._should_log():
            prefix = f"[{self.name}] " if self.name else ""
            logger.debug(f"{prefix}{message}")

    def info(self, message: str) -> None:
        """Log at info level if throttle allows."""
        if self._should_log():
            prefix = f"[{self.name}] " if self.name else ""
            logger.info(f"{prefix}{message}")

    def warning(self, message: str) -> None:
        """Log at warning level if throttle allows."""
        if self._should_log():
            prefix = f"[{self.name}] " if self.name else ""
            logger.warning(f"{prefix}{message}")

    def reset(self) -> None:
        """Reset the throttle counters."""
        with self._lock:
            self._call_count = 0
            self._last_log_time = 0.0


# Global throttled loggers for common hot paths
_step_logger = ThrottledLogger("STEP", log_interval=10, time_interval_ms=1000.0)
_token_logger = ThrottledLogger("TOKEN", log_interval=10, time_interval_ms=1000.0)


def get_step_logger() -> ThrottledLogger:
    """Get the global step logger for generation loop logging."""
    return _step_logger


def get_token_logger() -> ThrottledLogger:
    """Get the global token logger for token processing logging."""
    return _token_logger


__all__ = [
    "GenerationPerformanceMetrics",
    "GenerationPerformanceTracker",
    "ThrottledLogger",
    "TimingContext",
    "get_generation_performance_tracker",
    "get_step_logger",
    "get_token_logger",
    "timed",
    "timed_async",
]
