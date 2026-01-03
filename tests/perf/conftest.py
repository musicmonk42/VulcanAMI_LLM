# tests/perf/conftest.py
# -*- coding: utf-8 -*-
"""
VULCAN Performance Test Configuration and Fixtures.

This module provides shared configuration, fixtures, and utilities for the
VULCAN performance and boundedness test suite. It implements industry-standard
patterns for performance testing in CI/CD environments.

Architecture:
    Configuration is loaded from environment variables with validated defaults,
    enabling flexible threshold adjustment for different CI environments without
    code changes. All I/O operations are thread-safe and use proper error handling.

Key Features:
    - Environment-variable-based configuration with validation
    - Memory tracking via psutil (graceful degradation if unavailable)
    - Thread-safe result collection and aggregation
    - JSON and Markdown report generation
    - Dependency availability checking

Configuration Environment Variables:
    PERF_MAX_RSS_GROWTH_MB: Maximum allowed RSS memory growth in MB (default: 50)
    PERF_MAX_SLOWDOWN_PCT: Maximum allowed performance slowdown % (default: 20)
    PERF_MAX_P95_REGRESSION_PCT: Maximum p95 latency regression % (default: 25)
    PERF_MAX_RPS_REGRESSION_PCT: Maximum throughput regression % (default: 25)
    PERF_ITERATIONS: Number of iterations for boundedness tests (default: 500)
    PERF_CONCURRENCY_LEVELS: Comma-separated concurrency levels (default: "10,25,50")
    PERF_CONCURRENCY_DURATION: Duration for concurrency tests in seconds (default: 30)
    PERF_OUTPUT_DIR: Output directory for reports (default: "perf")

Example:
    >>> from tests.perf.conftest import get_perf_config, MemoryTracker
    >>> config = get_perf_config()
    >>> print(f"Max RSS growth allowed: {config.max_rss_growth_mb} MB")
    Max RSS growth allowed: 50 MB

Author:
    VULCAN-AGI Performance Engineering Team

Version:
    1.0.0

License:
    Proprietary - VULCAN AGI Project

See Also:
    - tests/perf/test_perf_smoke.py: Performance smoke tests
    - tests/perf/test_boundedness.py: Memory boundedness tests
    - perf/baseline.json: Performance baselines
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

import pytest

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================
# OPTIONAL DEPENDENCY HANDLING
# ============================================================

# Try to import psutil for memory tracking with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE: Final[bool] = True
    logger.debug("psutil available for memory tracking")
except ImportError:
    PSUTIL_AVAILABLE: Final[bool] = False
    psutil = None  # type: ignore[assignment]
    logger.warning(
        "psutil not available - memory tracking will be disabled. "
        "Install with: pip install psutil"
    )


# ============================================================
# CONSTANTS
# ============================================================

# Default configuration values (can be overridden via environment variables)
DEFAULT_MAX_RSS_GROWTH_MB: Final[float] = 50.0
DEFAULT_MAX_SLOWDOWN_PCT: Final[float] = 20.0
DEFAULT_MAX_P95_REGRESSION_PCT: Final[float] = 25.0
DEFAULT_MAX_RPS_REGRESSION_PCT: Final[float] = 25.0
DEFAULT_ITERATIONS: Final[int] = 500
DEFAULT_CONCURRENCY_LEVELS: Final[str] = "10,25,50"
DEFAULT_CONCURRENCY_DURATION: Final[float] = 30.0
DEFAULT_OUTPUT_DIR: Final[str] = "perf"

# Validation bounds
MIN_ITERATIONS: Final[int] = 10
MAX_ITERATIONS: Final[int] = 100000
MIN_CONCURRENCY: Final[int] = 1
MAX_CONCURRENCY: Final[int] = 1000


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class PerfConfigError(Exception):
    """Raised when performance configuration is invalid."""
    pass


class MemoryTrackingError(Exception):
    """Raised when memory tracking operations fail."""
    pass


# ============================================================
# CONFIGURATION
# ============================================================

def _parse_env_float(
    key: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """
    Parse and validate a float from environment variable.

    Args:
        key: Environment variable name.
        default: Default value if not set or invalid.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        Validated float value.

    Raises:
        PerfConfigError: If value is outside allowed range after parsing.
    """
    raw_value = os.environ.get(key, "")
    if not raw_value:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(
            f"Invalid float value for {key}='{raw_value}', using default={default}"
        )
        return default

    if min_val is not None and value < min_val:
        raise PerfConfigError(
            f"{key}={value} is below minimum allowed value {min_val}"
        )
    if max_val is not None and value > max_val:
        raise PerfConfigError(
            f"{key}={value} is above maximum allowed value {max_val}"
        )

    return value


def _parse_env_int(
    key: str,
    default: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> int:
    """
    Parse and validate an integer from environment variable.

    Args:
        key: Environment variable name.
        default: Default value if not set or invalid.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        Validated integer value.

    Raises:
        PerfConfigError: If value is outside allowed range after parsing.
    """
    raw_value = os.environ.get(key, "")
    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            f"Invalid integer value for {key}='{raw_value}', using default={default}"
        )
        return default

    if min_val is not None and value < min_val:
        raise PerfConfigError(
            f"{key}={value} is below minimum allowed value {min_val}"
        )
    if max_val is not None and value > max_val:
        raise PerfConfigError(
            f"{key}={value} is above maximum allowed value {max_val}"
        )

    return value


def _parse_concurrency_levels(raw_value: str) -> List[int]:
    """
    Parse comma-separated concurrency levels.

    Args:
        raw_value: Comma-separated string of integers (e.g., "10,25,50").

    Returns:
        List of validated concurrency levels.

    Raises:
        PerfConfigError: If any value is invalid or outside bounds.
    """
    if not raw_value.strip():
        return [10, 25, 50]  # Default levels

    levels: List[int] = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            level = int(part)
            if level < MIN_CONCURRENCY or level > MAX_CONCURRENCY:
                raise PerfConfigError(
                    f"Concurrency level {level} outside bounds "
                    f"[{MIN_CONCURRENCY}, {MAX_CONCURRENCY}]"
                )
            levels.append(level)
        except ValueError:
            raise PerfConfigError(
                f"Invalid concurrency level '{part}' - must be an integer"
            )

    if not levels:
        return [10, 25, 50]

    return sorted(set(levels))  # Remove duplicates and sort


@dataclass(frozen=False)
class PerfConfig:
    """
    Performance test configuration with environment variable support.

    This dataclass holds all configurable thresholds and parameters for
    performance tests. Values are loaded from environment variables with
    validation and sensible defaults.

    Attributes:
        max_rss_growth_mb: Maximum allowed RSS memory growth in megabytes.
            Tests fail if memory grows beyond this threshold.
        max_slowdown_pct: Maximum allowed performance degradation percentage.
            Compares early vs late iteration performance.
        max_p95_regression_pct: Maximum allowed p95 latency regression percentage
            compared to baseline.
        max_rps_regression_pct: Maximum allowed throughput regression percentage
            compared to baseline.
        iterations: Number of iterations for boundedness/stability tests.
        concurrency_levels: List of concurrency levels to test (workers/tasks).
        concurrency_duration_seconds: Duration for each concurrency test.
        output_dir: Directory path for output reports.

    Example:
        >>> config = PerfConfig()
        >>> print(config.max_rss_growth_mb)
        50.0
        >>> # With environment override:
        >>> # PERF_MAX_RSS_GROWTH_MB=100 python -m pytest tests/perf/

    Thread Safety:
        This class is NOT thread-safe for modification. Create separate
        instances per thread if concurrent configuration is needed.
    """

    # Boundedness thresholds
    max_rss_growth_mb: float = field(
        default_factory=lambda: _parse_env_float(
            "PERF_MAX_RSS_GROWTH_MB",
            DEFAULT_MAX_RSS_GROWTH_MB,
            min_val=0.0,
            max_val=10000.0,
        )
    )
    max_slowdown_pct: float = field(
        default_factory=lambda: _parse_env_float(
            "PERF_MAX_SLOWDOWN_PCT",
            DEFAULT_MAX_SLOWDOWN_PCT,
            min_val=0.0,
            max_val=1000.0,
        )
    )

    # Regression thresholds
    max_p95_regression_pct: float = field(
        default_factory=lambda: _parse_env_float(
            "PERF_MAX_P95_REGRESSION_PCT",
            DEFAULT_MAX_P95_REGRESSION_PCT,
            min_val=0.0,
            max_val=1000.0,
        )
    )
    max_rps_regression_pct: float = field(
        default_factory=lambda: _parse_env_float(
            "PERF_MAX_RPS_REGRESSION_PCT",
            DEFAULT_MAX_RPS_REGRESSION_PCT,
            min_val=0.0,
            max_val=1000.0,
        )
    )

    # Test parameters
    iterations: int = field(
        default_factory=lambda: _parse_env_int(
            "PERF_ITERATIONS",
            DEFAULT_ITERATIONS,
            min_val=MIN_ITERATIONS,
            max_val=MAX_ITERATIONS,
        )
    )
    concurrency_levels: List[int] = field(
        default_factory=lambda: _parse_concurrency_levels(
            os.environ.get("PERF_CONCURRENCY_LEVELS", DEFAULT_CONCURRENCY_LEVELS)
        )
    )
    concurrency_duration_seconds: float = field(
        default_factory=lambda: _parse_env_float(
            "PERF_CONCURRENCY_DURATION",
            DEFAULT_CONCURRENCY_DURATION,
            min_val=1.0,
            max_val=3600.0,
        )
    )

    # Output configuration
    output_dir: str = field(
        default_factory=lambda: os.environ.get("PERF_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure output_dir is a valid path string
        if not self.output_dir:
            self.output_dir = DEFAULT_OUTPUT_DIR

        # Log configuration for debugging
        logger.debug(
            f"PerfConfig initialized: iterations={self.iterations}, "
            f"concurrency_levels={self.concurrency_levels}, "
            f"max_rss_growth_mb={self.max_rss_growth_mb}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary with all configuration values.
        """
        return {
            "max_rss_growth_mb": self.max_rss_growth_mb,
            "max_slowdown_pct": self.max_slowdown_pct,
            "max_p95_regression_pct": self.max_p95_regression_pct,
            "max_rps_regression_pct": self.max_rps_regression_pct,
            "iterations": self.iterations,
            "concurrency_levels": self.concurrency_levels,
            "concurrency_duration_seconds": self.concurrency_duration_seconds,
            "output_dir": self.output_dir,
        }


def get_perf_config() -> PerfConfig:
    """
    Factory function to create a PerfConfig instance.

    This function provides a clean interface for obtaining configuration
    and allows for future enhancements like caching or additional validation.

    Returns:
        Configured PerfConfig instance.

    Raises:
        PerfConfigError: If configuration values are invalid.

    Example:
        >>> config = get_perf_config()
        >>> print(f"Testing with {config.iterations} iterations")
    """
    try:
        return PerfConfig()
    except PerfConfigError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating PerfConfig: {e}")
        raise PerfConfigError(f"Failed to create configuration: {e}") from e


# ============================================================
# PYTEST FIXTURES
# ============================================================

@pytest.fixture(scope="session")
def perf_config() -> PerfConfig:
    """
    Provide performance test configuration for the test session.

    This fixture creates a single PerfConfig instance that is shared
    across all tests in the session. Configuration is loaded from
    environment variables with sensible defaults.

    Yields:
        PerfConfig: Session-scoped configuration instance.

    Example:
        >>> def test_something(perf_config: PerfConfig):
        ...     print(f"Max RSS growth: {perf_config.max_rss_growth_mb} MB")
    """
    return get_perf_config()


@pytest.fixture(scope="session")
def output_dir(perf_config: PerfConfig) -> pathlib.Path:
    """
    Create and return the output directory for performance results.

    Creates the directory if it doesn't exist. The path is determined
    by the PERF_OUTPUT_DIR configuration.

    Args:
        perf_config: Performance configuration fixture.

    Yields:
        pathlib.Path: Path to the output directory.

    Raises:
        OSError: If directory creation fails.
    """
    output_path = pathlib.Path(perf_config.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory created/verified: {output_path}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_path}: {e}")
        raise
    return output_path


@pytest.fixture(scope="function")
def memory_tracker() -> "MemoryTracker":
    """
    Provide a fresh MemoryTracker for each test function.

    Creates a new MemoryTracker instance that can be used to record
    RSS memory samples throughout the test. Tracking is disabled
    gracefully if psutil is not available.

    Yields:
        MemoryTracker: Fresh tracker instance for the test.

    Example:
        >>> def test_memory(memory_tracker: MemoryTracker):
        ...     memory_tracker.sample(label="start")
        ...     # ... do work ...
        ...     memory_tracker.sample(label="end")
        ...     growth = memory_tracker.get_growth_mb()
    """
    return MemoryTracker()


# ============================================================
# MEMORY TRACKING
# ============================================================

class MemoryTracker:
    """
    Thread-safe memory usage tracker for leak detection.

    Tracks Resident Set Size (RSS) memory usage over time, enabling
    detection of memory leaks and unbounded growth. Gracefully handles
    the absence of psutil by recording zero values.

    Attributes:
        samples: List of memory samples with timestamps and labels.

    Thread Safety:
        All public methods are thread-safe via internal locking.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.sample(label="before_work")
        45.2
        >>> # ... perform work ...
        >>> tracker.sample(label="after_work")
        47.8
        >>> print(f"Growth: {tracker.get_growth_mb():.2f} MB")
        Growth: 2.60 MB

    Note:
        If psutil is not available, all RSS values will be 0.0.
        Use PSUTIL_AVAILABLE constant to check availability.
    """

    def __init__(self) -> None:
        """
        Initialize the memory tracker.

        Sets up the psutil Process handle if available, and initializes
        the thread lock for safe concurrent access.
        """
        self._lock = threading.RLock()
        self._samples: List[Dict[str, Any]] = []
        self._process: Optional[Any] = None

        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                self._process = psutil.Process()
                logger.debug("MemoryTracker initialized with psutil")
            except Exception as e:
                logger.warning(f"Failed to initialize psutil Process: {e}")
                self._process = None

    @property
    def samples(self) -> List[Dict[str, Any]]:
        """
        Get a copy of all memory samples.

        Returns:
            List of sample dictionaries, each containing:
            - timestamp (float): Unix timestamp
            - rss_mb (float): RSS memory in megabytes
            - label (Optional[str]): User-provided label
        """
        with self._lock:
            return list(self._samples)

    def sample(self, label: Optional[str] = None) -> float:
        """
        Record a memory sample at the current point in time.

        Takes a snapshot of current RSS memory and stores it with
        a timestamp and optional label for later analysis.

        Args:
            label: Optional descriptive label for this sample
                   (e.g., "before_loop", "iteration_1000").

        Returns:
            Current RSS memory in megabytes, or 0.0 if psutil
            is not available or an error occurs.

        Thread Safety:
            This method is thread-safe.

        Example:
            >>> rss = tracker.sample(label="checkpoint_1")
            >>> print(f"Current RSS: {rss:.2f} MB")
        """
        rss_mb = 0.0
        timestamp = time.time()

        if self._process is not None:
            try:
                mem_info = self._process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
            except Exception as e:
                logger.debug(f"Memory sampling failed: {e}")
                # Continue with rss_mb = 0.0

        sample_data = {
            "timestamp": timestamp,
            "rss_mb": rss_mb,
            "label": label,
        }

        with self._lock:
            self._samples.append(sample_data)

        return rss_mb

    def get_growth_mb(self) -> float:
        """
        Calculate RSS memory growth from first to last sample.

        Returns:
            Memory growth in megabytes (positive = growth, negative = shrink).
            Returns 0.0 if fewer than 2 samples have been recorded.

        Thread Safety:
            This method is thread-safe.

        Example:
            >>> tracker.sample(label="start")
            >>> # ... allocate memory ...
            >>> tracker.sample(label="end")
            >>> growth = tracker.get_growth_mb()
            >>> assert growth < 50.0, "Memory leak detected!"
        """
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            return self._samples[-1]["rss_mb"] - self._samples[0]["rss_mb"]

    def get_max_rss_mb(self) -> float:
        """
        Get the maximum RSS value observed across all samples.

        Returns:
            Maximum RSS in megabytes, or 0.0 if no samples recorded.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            if not self._samples:
                return 0.0
            return max(s["rss_mb"] for s in self._samples)

    def get_min_rss_mb(self) -> float:
        """
        Get the minimum RSS value observed across all samples.

        Returns:
            Minimum RSS in megabytes, or 0.0 if no samples recorded.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            if not self._samples:
                return 0.0
            return min(s["rss_mb"] for s in self._samples)

    def get_sample_count(self) -> int:
        """
        Get the number of samples recorded.

        Returns:
            Number of memory samples.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            return len(self._samples)

    def clear(self) -> None:
        """
        Clear all recorded samples.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            self._samples.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Export memory tracking data for JSON serialization.

        Returns:
            Dictionary containing:
            - sample_count: Number of samples
            - samples: List of sample data
            - growth_mb: Total RSS growth
            - max_rss_mb: Maximum RSS observed
            - min_rss_mb: Minimum RSS observed
            - psutil_available: Whether psutil was used

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            return {
                "sample_count": len(self._samples),
                "samples": list(self._samples),
                "growth_mb": round(self.get_growth_mb(), 2),
                "max_rss_mb": round(self.get_max_rss_mb(), 2),
                "min_rss_mb": round(self.get_min_rss_mb(), 2),
                "psutil_available": PSUTIL_AVAILABLE,
            }


# ============================================================
# RESULT COLLECTION
# ============================================================

class PerfResultCollector:
    """
    Thread-safe collector for aggregating performance test results.

    Collects individual test results and provides aggregation methods
    for calculating percentiles, averages, and generating reports.

    Attributes:
        test_name: Name of the test being collected.
        results: List of collected result dictionaries.
        metadata: Metadata about the collection session.

    Thread Safety:
        All public methods are thread-safe via internal locking.

    Example:
        >>> collector = PerfResultCollector("test_throughput")
        >>> collector.add_result({"latency_seconds": 0.05, "success": True})
        >>> collector.add_result({"latency_seconds": 0.08, "success": True})
        >>> percentiles = collector.calculate_percentiles()
        >>> print(f"p95 latency: {percentiles['p95']:.4f}s")

    Note:
        Results should contain a "latency_seconds" key for percentile
        calculations and a "success" key for success rate tracking.
    """

    def __init__(self, test_name: str) -> None:
        """
        Initialize the result collector.

        Args:
            test_name: Name identifier for the test being collected.

        Raises:
            ValueError: If test_name is empty.
        """
        if not test_name or not test_name.strip():
            raise ValueError("test_name cannot be empty")

        self._lock = threading.RLock()
        self.test_name = test_name.strip()
        self.start_time = time.time()
        self._results: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "test_name": self.test_name,
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "psutil_available": PSUTIL_AVAILABLE,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        }

        logger.debug(f"PerfResultCollector initialized for test: {test_name}")

    @property
    def results(self) -> List[Dict[str, Any]]:
        """
        Get a copy of all collected results.

        Returns:
            List of result dictionaries.
        """
        with self._lock:
            return list(self._results)

    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a test result to the collection.

        Args:
            result: Dictionary containing test result data.
                    Should include "latency_seconds" and "success" keys
                    for full functionality.

        Thread Safety:
            This method is thread-safe.

        Example:
            >>> collector.add_result({
            ...     "latency_seconds": 0.05,
            ...     "success": True,
            ...     "query_type": "mathematical",
            ... })
        """
        if result is None:
            logger.warning("Attempted to add None result, ignoring")
            return

        # Add timestamp if not present
        if "timestamp" not in result:
            result = {**result, "timestamp": time.time()}

        with self._lock:
            self._results.append(result)

    def get_latencies(self) -> List[float]:
        """
        Extract all latency values from collected results.

        Returns:
            List of latency values in seconds. Only includes results
            that have a "latency_seconds" key.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            return [
                r["latency_seconds"]
                for r in self._results
                if "latency_seconds" in r and isinstance(r["latency_seconds"], (int, float))
            ]

    def get_success_count(self) -> int:
        """
        Count successful results.

        Returns:
            Number of results with success=True.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            return sum(1 for r in self._results if r.get("success", False))

    def get_failure_count(self) -> int:
        """
        Count failed results.

        Returns:
            Number of results with success=False or missing success key.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            return sum(1 for r in self._results if not r.get("success", False))

    def calculate_percentiles(self) -> Dict[str, float]:
        """
        Calculate latency percentiles from collected results.

        Uses linear interpolation for percentile calculation, which is
        the industry-standard method (same as numpy.percentile with
        default interpolation).

        Returns:
            Dictionary with keys "p50", "p95", "p99" containing
            the respective percentile values. Returns 0.0 for all
            if no latency data available.

        Thread Safety:
            This method is thread-safe.

        Example:
            >>> percentiles = collector.calculate_percentiles()
            >>> print(f"Median latency: {percentiles['p50']:.4f}s")
        """
        latencies = sorted(self.get_latencies())

        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        n = len(latencies)

        def percentile(p: float) -> float:
            """Calculate percentile using linear interpolation."""
            if n == 1:
                return latencies[0]
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return latencies[f] + (k - f) * (latencies[c] - latencies[f])

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Export all collected data for JSON serialization.

        Generates a comprehensive summary including metadata,
        statistical summaries, and all individual results.

        Returns:
            Dictionary suitable for JSON serialization containing:
            - metadata: Collection metadata
            - summary: Statistical summary
            - results: List of all results

        Thread Safety:
            This method is thread-safe.
        """
        # Update metadata with end time
        self.metadata["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.metadata["duration_seconds"] = round(time.time() - self.start_time, 3)

        latencies = self.get_latencies()
        percentiles = self.calculate_percentiles()

        with self._lock:
            results_copy = list(self._results)
            total_results = len(self._results)
            successful_results = self.get_success_count()

        summary: Dict[str, Any] = {
            "total_results": total_results,
            "successful_results": successful_results,
            "failed_results": total_results - successful_results,
            "success_rate": successful_results / max(1, total_results),
            "latency_p50": round(percentiles["p50"], 6),
            "latency_p95": round(percentiles["p95"], 6),
            "latency_p99": round(percentiles["p99"], 6),
        }

        if latencies:
            summary.update({
                "latency_avg": round(sum(latencies) / len(latencies), 6),
                "latency_min": round(min(latencies), 6),
                "latency_max": round(max(latencies), 6),
                "latency_count": len(latencies),
            })
        else:
            summary.update({
                "latency_avg": 0.0,
                "latency_min": 0.0,
                "latency_max": 0.0,
                "latency_count": 0,
            })

        return {
            "metadata": self.metadata,
            "summary": summary,
            "results": results_copy,
        }


@pytest.fixture
def result_collector(request: pytest.FixtureRequest) -> PerfResultCollector:
    """
    Provide a result collector for the current test function.

    Creates a new PerfResultCollector named after the current test.

    Args:
        request: Pytest fixture request object.

    Yields:
        PerfResultCollector: Collector instance for the test.

    Example:
        >>> def test_throughput(result_collector: PerfResultCollector):
        ...     for i in range(100):
        ...         start = time.perf_counter()
        ...         do_work()
        ...         latency = time.perf_counter() - start
        ...         result_collector.add_result({
        ...             "latency_seconds": latency,
        ...             "success": True,
        ...         })
    """
    return PerfResultCollector(request.node.name)


# ============================================================
# DEPENDENCY AVAILABILITY CHECKS
# ============================================================

def check_dependency(module_name: str) -> bool:
    """
    Check if a Python module is available for import.

    Performs a lightweight check without actually importing the module
    (uses importlib.util.find_spec when possible).

    Args:
        module_name: Name of the module to check (e.g., "numpy", "torch").

    Returns:
        True if the module is importable, False otherwise.

    Example:
        >>> if check_dependency("torch"):
        ...     import torch
        ...     print(f"PyTorch version: {torch.__version__}")
        ... else:
        ...     print("PyTorch not available")
    """
    try:
        # Try lightweight spec check first
        import importlib.util
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return True
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    # Fallback to actual import attempt
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Expected dependencies for perf-lite (fast CI install)
PERF_LITE_DEPS: Final[List[str]] = [
    "psutil",
    "sklearn",  # scikit-learn imports as sklearn
    "networkx",
    "statsmodels",
    "sympy",
    "pandas",
    "yaml",  # PyYAML imports as yaml
    "whoosh",
]

# Additional dependencies for perf-full (includes heavy ML deps)
PERF_FULL_DEPS: Final[List[str]] = PERF_LITE_DEPS + [
    "torch",
    "sentence_transformers",
    "faiss",
]

# Mapping from import name to pip package name
_MODULE_TO_PACKAGE: Final[Dict[str, str]] = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "faiss": "faiss-cpu",
    "sentence_transformers": "sentence-transformers",
}


def get_package_name(module_name: str) -> str:
    """
    Get the pip package name for a module.

    Some modules import with different names than their pip packages
    (e.g., sklearn -> scikit-learn).

    Args:
        module_name: Python import name.

    Returns:
        Pip package name for installation.

    Example:
        >>> get_package_name("sklearn")
        'scikit-learn'
        >>> get_package_name("numpy")
        'numpy'
    """
    return _MODULE_TO_PACKAGE.get(module_name, module_name)


def get_available_deps() -> Dict[str, bool]:
    """
    Get availability status of all performance test dependencies.

    Checks both perf-lite and perf-full dependencies.

    Returns:
        Dictionary mapping module names to availability status.

    Example:
        >>> deps = get_available_deps()
        >>> for name, available in sorted(deps.items()):
        ...     status = "✓" if available else "✗"
        ...     print(f"{status} {name}")
    """
    all_deps = set(PERF_LITE_DEPS + PERF_FULL_DEPS)
    return {dep: check_dependency(dep) for dep in sorted(all_deps)}


def get_missing_deps(dep_list: List[str]) -> List[str]:
    """
    Get list of missing dependencies from a dependency list.

    Args:
        dep_list: List of module names to check.

    Returns:
        List of module names that are not available.

    Example:
        >>> missing = get_missing_deps(PERF_LITE_DEPS)
        >>> if missing:
        ...     print(f"Missing: {', '.join(missing)}")
    """
    return [dep for dep in dep_list if not check_dependency(dep)]


@pytest.fixture(scope="session")
def available_deps() -> Dict[str, bool]:
    """
    Provide dependency availability information for the test session.

    Yields:
        Dictionary mapping module names to availability status.

    Example:
        >>> def test_with_deps(available_deps: Dict[str, bool]):
        ...     if not available_deps.get("torch"):
        ...         pytest.skip("torch not available")
    """
    deps = get_available_deps()
    logger.debug(f"Dependency check: {sum(deps.values())}/{len(deps)} available")
    return deps


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_json_report(
    data: Dict[str, Any],
    output_path: Union[str, pathlib.Path],
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save data to a JSON file with proper error handling.

    Creates parent directories if they don't exist. Uses atomic
    write pattern where possible.

    Args:
        data: Data to serialize to JSON.
        output_path: Path to save the JSON file.
        indent: JSON indentation level (default: 2).
        ensure_ascii: If True, escape non-ASCII characters (default: False).

    Raises:
        OSError: If directory creation or file write fails.
        TypeError: If data is not JSON-serializable.

    Example:
        >>> save_json_report(
        ...     {"results": [1, 2, 3]},
        ...     pathlib.Path("reports/data.json"),
        ... )
    """
    path = pathlib.Path(output_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path.parent}: {e}")
        raise

    try:
        # Serialize first to catch errors before writing
        json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

        logger.debug(f"JSON report saved to: {path}")

    except TypeError as e:
        logger.error(f"Data is not JSON serializable: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        raise


def save_markdown_report(
    content: str,
    output_path: Union[str, pathlib.Path],
) -> None:
    """
    Save markdown content to a file.

    Creates parent directories if they don't exist.

    Args:
        content: Markdown content to save.
        output_path: Path to save the file.

    Raises:
        OSError: If directory creation or file write fails.

    Example:
        >>> save_markdown_report(
        ...     "# Report\\n\\nTest passed!",
        ...     pathlib.Path("reports/summary.md"),
        ... )
    """
    path = pathlib.Path(output_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path.parent}: {e}")
        raise

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Markdown report saved to: {path}")

    except OSError as e:
        logger.error(f"Failed to write markdown to {path}: {e}")
        raise


def generate_summary_markdown(
    results: Dict[str, Any],
    title: str,
    *,
    include_metadata: bool = True,
    include_timestamp: bool = True,
) -> str:
    """
    Generate a markdown summary from test results.

    Creates a well-formatted markdown document with summary statistics
    and optional metadata.

    Args:
        results: Results dictionary containing "summary" and optionally
                 "metadata" keys.
        title: Title for the markdown document.
        include_metadata: Include metadata section if available (default: True).
        include_timestamp: Include generation timestamp (default: True).

    Returns:
        Markdown-formatted string.

    Example:
        >>> results = {
        ...     "summary": {"total_results": 100, "success_rate": 0.95},
        ...     "metadata": {"test_name": "perf_smoke"},
        ... }
        >>> markdown = generate_summary_markdown(results, "Performance Results")
        >>> print(markdown)
        # Performance Results
        ...
    """
    lines: List[str] = [
        f"# {title}",
        "",
    ]

    if include_timestamp:
        lines.extend([
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            "",
        ])

    # Summary section
    summary = results.get("summary", {})
    if summary:
        lines.extend([
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        for key, value in sorted(summary.items()):
            if isinstance(value, float):
                if "rate" in key.lower() or "pct" in key.lower():
                    formatted_value = f"{value:.2%}"
                elif value < 0.001 and value != 0:
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            lines.append(f"| {key} | {formatted_value} |")

        lines.append("")

    # Metadata section
    if include_metadata:
        metadata = results.get("metadata", {})
        if metadata:
            lines.extend([
                "## Metadata",
                "",
            ])
            for key, value in sorted(metadata.items()):
                lines.append(f"- **{key}:** {value}")
            lines.append("")

    return "\n".join(lines)
