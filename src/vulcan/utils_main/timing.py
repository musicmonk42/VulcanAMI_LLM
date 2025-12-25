# ============================================================
# VULCAN-AGI Timing Utilities Module
# Performance instrumentation decorators and parallel execution
# ============================================================
#
# This module provides decorators and utilities for:
#     - Timing async and sync functions
#     - Logging slow operations for bottleneck detection
#     - Running coroutines in parallel
#
# USAGE:
#     from vulcan.utils_main.timing import timed_async, timed_sync
#     
#     @timed_async
#     async def my_async_function():
#         await asyncio.sleep(0.5)
#     
#     @timed_sync
#     def my_sync_function():
#         time.sleep(0.5)
#
# CONFIGURATION:
#     SLOW_OPERATION_THRESHOLD_MS: Operations taking longer than this
#                                  will be logged as warnings (default: 100ms)
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation
# ============================================================

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Coroutine, List, TypeVar

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Threshold in milliseconds for logging slow operations
SLOW_OPERATION_THRESHOLD_MS = 100

# Type variable for generic function signatures
T = TypeVar('T')


# ============================================================
# TIMING DECORATORS
# ============================================================


def timed_async(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator to time async functions and log slow operations.
    
    PERFORMANCE FIX: This helps identify future bottlenecks by logging
    any async function that takes longer than SLOW_OPERATION_THRESHOLD_MS.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated async function that logs timing information
        
    Example:
        >>> @timed_async
        ... async def slow_operation():
        ...     await asyncio.sleep(0.2)
        ...     return "done"
        >>> # Will log: [SLOW] module.slow_operation took 200.0ms
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > SLOW_OPERATION_THRESHOLD_MS:
                logger.warning(
                    f"[SLOW] {func.__module__}.{func.__name__} took {elapsed_ms:.1f}ms"
                )
    return wrapper


def timed_sync(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time sync functions and log slow operations.
    
    PERFORMANCE FIX: This helps identify future bottlenecks by logging
    any sync function that takes longer than SLOW_OPERATION_THRESHOLD_MS.
    
    Args:
        func: The sync function to decorate
        
    Returns:
        Decorated function that logs timing information
        
    Example:
        >>> @timed_sync
        ... def slow_operation():
        ...     time.sleep(0.2)
        ...     return "done"
        >>> # Will log: [SLOW] module.slow_operation took 200.0ms
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > SLOW_OPERATION_THRESHOLD_MS:
                logger.warning(
                    f"[SLOW] {func.__module__}.{func.__name__} took {elapsed_ms:.1f}ms"
                )
    return wrapper


# ============================================================
# PARALLEL EXECUTION
# ============================================================


async def run_tasks_in_parallel(*coroutines) -> List[Any]:
    """
    Run multiple coroutines in parallel using asyncio.gather.
    
    PERFORMANCE FIX: This allows multiple CPU-bound or I/O-bound operations
    to run concurrently instead of sequentially, reducing total latency.
    
    Args:
        *coroutines: Variable number of coroutines to run in parallel
        
    Returns:
        List of results from each coroutine (in order).
        If a coroutine raises an exception, it will be included in the results.
        
    Example:
        >>> async def fetch_data(n):
        ...     await asyncio.sleep(0.1)
        ...     return f"data_{n}"
        >>> results = await run_tasks_in_parallel(
        ...     fetch_data(1),
        ...     fetch_data(2),
        ...     fetch_data(3),
        ... )
        >>> # Takes ~0.1s instead of ~0.3s sequential
    """
    return await asyncio.gather(*coroutines, return_exceptions=True)


async def run_tasks_with_timeout(
    coroutines: List[Coroutine],
    timeout: float,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Run multiple coroutines in parallel with a timeout.
    
    Args:
        coroutines: List of coroutines to run
        timeout: Maximum time to wait (in seconds)
        return_exceptions: If True, exceptions are returned as results
        
    Returns:
        List of results from completed coroutines
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    return await asyncio.wait_for(
        asyncio.gather(*coroutines, return_exceptions=return_exceptions),
        timeout=timeout
    )


# ============================================================
# TIMING UTILITIES
# ============================================================


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with Timer() as t:
        ...     time.sleep(0.1)
        >>> print(f"Elapsed: {t.elapsed_ms:.1f}ms")
    """
    
    def __init__(self, name: str = None, log_on_exit: bool = False):
        """
        Initialize timer.
        
        Args:
            name: Optional name for logging
            log_on_exit: If True, log elapsed time on context exit
        """
        self.name = name
        self.log_on_exit = log_on_exit
        self._start: float = 0
        self._end: float = 0
    
    def __enter__(self) -> 'Timer':
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._end = time.perf_counter()
        if self.log_on_exit:
            name = self.name or "block"
            if self.elapsed_ms > SLOW_OPERATION_THRESHOLD_MS:
                logger.warning(f"[SLOW] {name} took {self.elapsed_ms:.1f}ms")
            else:
                logger.debug(f"{name} took {self.elapsed_ms:.1f}ms")
        return False
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._end > 0:
            return self._end - self._start
        return time.perf_counter() - self._start
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000


def set_slow_threshold(threshold_ms: int) -> None:
    """
    Set the threshold for logging slow operations.
    
    Args:
        threshold_ms: New threshold in milliseconds
    """
    global SLOW_OPERATION_THRESHOLD_MS
    SLOW_OPERATION_THRESHOLD_MS = threshold_ms
    logger.info(f"Slow operation threshold set to {threshold_ms}ms")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "timed_async",
    "timed_sync",
    "run_tasks_in_parallel",
    "run_tasks_with_timeout",
    "Timer",
    "SLOW_OPERATION_THRESHOLD_MS",
    "set_slow_threshold",
]


# Log module initialization
logger.debug(f"Timing utilities module v{__version__} loaded (threshold: {SLOW_OPERATION_THRESHOLD_MS}ms)")
