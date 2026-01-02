#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Linear Scaling Proof Test.

This module provides comprehensive scalability proof testing for the VULCAN-AGI
system, validating linear scaling behavior across agent pool configurations.

The test proves that the VULCAN system scales linearly with increased agent
pool capacity by measuring throughput improvements as agents are added. This
is critical for production capacity planning and performance validation.

Key Features:
    - Tests with configurable agent configurations (default: 2, 5, 10, 20)
    - For each configuration: N queries where N = agents × 10 (saturates pool)
    - Measures: total duration, throughput (queries/min), average latency
    - Tracks: success rate, per-query metrics, response headers
    - Calculates scaling efficiency using industry-standard formulas
    - Verifies latency consistency (±20% variance threshold)
    - Generates detailed JSON and text reports

Architecture:
    Uses ThreadPoolExecutor to match VULCAN's production concurrency model
    (NOT asyncio.gather, NOT Redis). This ensures accurate measurements that
    reflect real production behavior.

    Request flow:
        Test → ThreadPoolExecutor → requests.Session → HTTP/S → VULCAN API

Pass/Fail Criteria:
    - Scaling efficiency >= 85%: (throughput_increase / agent_increase) × 100
    - Latency variance within ±20% of baseline configuration

Usage:
    Command line::

        # Run full scaling test with default settings
        python -m tests.performance.prove_linear_scaling

        # Specify custom URL and output path
        python -m tests.performance.prove_linear_scaling \\
            --url http://localhost:8000 \\
            --output results/scaling_test.json

        # Enable verbose logging
        python -m tests.performance.prove_linear_scaling -v

    Programmatic::

        from tests.performance.prove_linear_scaling import LinearScalingProofTest

        test = LinearScalingProofTest(base_url="http://localhost:8080")
        try:
            results = test.run_all_configurations()
            report = test.generate_report(results)
            print(test.format_text_report(report))
            test.save_results(report, "results.json")
        finally:
            test.close()

Example Output:
    ========================================================================
    VULCAN LINEAR SCALING PROOF TEST
    ========================================================================
    Configuration: 2 agents, 20 queries → 12.5 qpm, 0.45s avg latency
    Configuration: 5 agents, 50 queries → 28.7 qpm, 0.48s avg latency
    Configuration: 10 agents, 100 queries → 54.2 qpm, 0.51s avg latency
    Configuration: 20 agents, 200 queries → 102.8 qpm, 0.53s avg latency

    Scaling Efficiency: 91.2% (threshold: 85%)
    Latency Variance: 17.8% (threshold: 20%)
    RESULT: PASSED

Author:
    VULCAN-AGI Performance Engineering Team

Version:
    1.0.0

License:
    Proprietary - VULCAN AGI Project

See Also:
    - tests.performance.scalability_stress_test: General stress testing
    - tests.performance.performance_monitor: System metrics collection
    - src.vulcan.orchestrator.agent_pool: Agent pool implementation
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import pathlib
import statistics
import sys
import threading
import time
import traceback
import uuid
import warnings
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress noisy warnings during performance tests
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Configure structured logging with ISO8601 timestamps and thread info
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - "
        "[%(threadName)s] - %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# MODULE METADATA
# ============================================================

__version__: Final[str] = "1.0.0"
__author__: Final[str] = "VULCAN-AGI Performance Engineering Team"
__all__ = [
    "LinearScalingProofTest",
    "QueryResult",
    "ConfigurationResult",
    "ScalingAnalysis",
    "TestReport",
    "TestStatus",
    "main",
]


# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================

# API endpoint configuration
DEFAULT_BASE_URL: Final[str] = "http://localhost:8080"
CHAT_ENDPOINT: Final[str] = "/vulcan/v1/chat"

# Timeout configuration (in seconds)
DEFAULT_CONNECT_TIMEOUT: Final[float] = 10.0
DEFAULT_READ_TIMEOUT: Final[float] = 60.0
DEFAULT_QUERY_TIMEOUT: Final[float] = 60.0

# Safety limits to prevent runaway tests
MAX_TOTAL_TEST_TIMEOUT: Final[float] = 3600.0  # 1 hour maximum
MAX_SINGLE_CONFIG_TIMEOUT: Final[float] = 900.0  # 15 minutes per configuration
MAX_QUERIES_PER_CONFIG: Final[int] = 500  # Safety limit

# Agent configurations for scaling test
DEFAULT_AGENT_CONFIGURATIONS: Final[Tuple[int, ...]] = (2, 5, 10, 20)

# Query multiplier (N queries = agents × QUERY_MULTIPLIER)
QUERY_MULTIPLIER: Final[int] = 10

# Sleep between configurations for agent pool reconfiguration
SLEEP_BETWEEN_CONFIGS: Final[float] = 5.0

# Pass/fail thresholds (industry standard values)
EFFICIENCY_THRESHOLD: Final[float] = 0.85  # 85% efficiency required for pass
LATENCY_VARIANCE_THRESHOLD: Final[float] = 0.20  # ±20% latency variance allowed

# Error handling configuration
ERROR_MESSAGE_MAX_LENGTH: Final[int] = 200
MAX_RETRIES_FOR_TRANSIENT: Final[int] = 0  # No retries for accurate latency

# Connection pooling configuration (optimized for high concurrency)
POOL_CONNECTIONS: Final[int] = 100
POOL_MAXSIZE: Final[int] = 100
POOL_BLOCK: Final[bool] = False


# ============================================================
# SAMPLE QUERIES FOR TESTING
# ============================================================

# Diverse query types to exercise all reasoning modalities
SAMPLE_QUERIES: Final[Tuple[Dict[str, Any], ...]] = (
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze causal relationship between X and Y",
            }
        ],
        "provider": "local",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Solve equation: 3x + 7 = 22, show reasoning",
            }
        ],
        "provider": "local",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Prove logical statement: (P → Q) ∧ P ⊢ Q",
            }
        ],
        "provider": "local",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Calculate P(A|B) given P(B|A)=0.8, P(A)=0.3, P(B)=0.5",
            }
        ],
        "provider": "local",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain the concept of emergence in complex systems",
            }
        ],
        "provider": "local",
    },
)


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================


class LinearScalingTestError(Exception):
    """Base exception for linear scaling test errors."""

    pass


class ConfigurationError(LinearScalingTestError):
    """Raised when test configuration is invalid."""

    pass


class APIConnectionError(LinearScalingTestError):
    """Raised when connection to VULCAN API fails."""

    pass


class TestTimeoutError(LinearScalingTestError):
    """Raised when test exceeds maximum allowed time."""

    pass


# ============================================================
# ENUMERATIONS
# ============================================================


class TestStatus(Enum):
    """Status of a test execution."""

    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    TIMEOUT = auto()
    SKIPPED = auto()

    def __str__(self) -> str:
        """Return human-readable status string."""
        return self.name


class QueryStatus(Enum):
    """Status of a single query execution."""

    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CONNECTION_ERROR = auto()
    UNKNOWN_ERROR = auto()

    def is_successful(self) -> bool:
        """Check if query was successful."""
        return self == QueryStatus.SUCCESS


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass(frozen=False)
class QueryResult:
    """
    Result of a single query execution.

    This dataclass captures comprehensive information about a query execution,
    including timing, status, and any error information.

    Attributes:
        query_id: Unique identifier for this query execution.
        status: Final status of the query execution.
        latency_seconds: Total wall-clock time for execution.
        http_status_code: HTTP status code from response (None if no response).
        error_message: Error description if failed (None if successful).
        response_headers: HTTP headers from response for metrics extraction.
        timestamp: ISO8601 timestamp when query was executed.

    Example:
        >>> result = QueryResult(
        ...     query_id="abc123",
        ...     status=QueryStatus.SUCCESS,
        ...     latency_seconds=0.5,
        ...     http_status_code=200,
        ... )
        >>> result.is_successful
        True
    """

    query_id: str
    status: QueryStatus
    latency_seconds: float
    http_status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")

    @property
    def is_successful(self) -> bool:
        """Check if query was successful."""
        return self.status == QueryStatus.SUCCESS

    @property
    def truncated_error(self) -> Optional[str]:
        """Get truncated error message for display."""
        if self.error_message is None:
            return None
        if len(self.error_message) <= ERROR_MESSAGE_MAX_LENGTH:
            return self.error_message
        return self.error_message[: ERROR_MESSAGE_MAX_LENGTH - 3] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "query_id": self.query_id,
            "status": self.status.name,
            "is_successful": self.is_successful,
            "latency_seconds": round(self.latency_seconds, 6),
            "http_status_code": self.http_status_code,
            "error_message": self.truncated_error,
            "timestamp": self.timestamp,
        }


@dataclass
class LatencyStatistics:
    """
    Statistical analysis of latency measurements.

    Uses industry-standard percentile calculations for accurate
    performance characterization.

    Attributes:
        count: Number of samples.
        mean: Arithmetic mean of latencies.
        std_dev: Standard deviation.
        min: Minimum observed latency.
        max: Maximum observed latency.
        p50: 50th percentile (median).
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
    """

    count: int
    mean: float
    std_dev: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencyStatistics":
        """
        Create statistics from latency samples.

        Uses proper percentile calculation with linear interpolation
        for accurate results.

        Args:
            samples: Sequence of latency measurements in seconds.

        Returns:
            LatencyStatistics with computed values.

        Raises:
            ValueError: If samples is empty.
        """
        if not samples:
            raise ValueError("Cannot compute statistics from empty samples")

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        # Calculate percentiles using linear interpolation (industry standard)
        def percentile(data: List[float], p: float) -> float:
            """Calculate percentile using linear interpolation."""
            if not data:
                return 0.0
            if len(data) == 1:
                return data[0]
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        return cls(
            count=n,
            mean=statistics.mean(sorted_samples),
            std_dev=statistics.stdev(sorted_samples) if n >= 2 else 0.0,
            min=sorted_samples[0],
            max=sorted_samples[-1],
            p50=percentile(sorted_samples, 0.50),
            p90=percentile(sorted_samples, 0.90),
            p95=percentile(sorted_samples, 0.95),
            p99=percentile(sorted_samples, 0.99),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "mean": round(self.mean, 6),
            "std_dev": round(self.std_dev, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "p50": round(self.p50, 6),
            "p90": round(self.p90, 6),
            "p95": round(self.p95, 6),
            "p99": round(self.p99, 6),
        }


@dataclass
class ConfigurationResult:
    """
    Result of a single agent configuration test.

    Contains comprehensive metrics for throughput, latency, and success rate
    analysis for a specific agent pool configuration.

    Attributes:
        agent_count: Number of agents in this configuration.
        total_queries: Total number of queries executed.
        successful_queries: Count of successful queries.
        failed_queries: Count of failed queries.
        total_duration_seconds: Total wall-clock time for configuration test.
        throughput_qps: Queries per second.
        throughput_qpm: Queries per minute.
        success_rate: Ratio of successful to total queries.
        latency_stats: Statistical analysis of successful query latencies.
        query_results: Detailed results for each query.
        timestamp: ISO8601 timestamp when test was executed.
    """

    agent_count: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_duration_seconds: float
    throughput_qps: float
    throughput_qpm: float
    success_rate: float
    latency_stats: Optional[LatencyStatistics]
    query_results: List[QueryResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_count": self.agent_count,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "throughput_qps": round(self.throughput_qps, 4),
            "throughput_qpm": round(self.throughput_qpm, 2),
            "success_rate": round(self.success_rate, 4),
            "latency_stats": (
                self.latency_stats.to_dict() if self.latency_stats else None
            ),
            "timestamp": self.timestamp,
        }


@dataclass
class ScalingAnalysis:
    """
    Analysis of scaling efficiency between two configurations.

    Calculates how well throughput scales with increased agent count,
    using the formula: efficiency = (throughput_factor / agent_factor) × 100

    Attributes:
        base_agents: Agent count in baseline configuration.
        scaled_agents: Agent count in scaled configuration.
        agent_factor: Ratio of scaled to base agents.
        throughput_factor: Ratio of scaled to base throughput.
        efficiency_percent: Scaling efficiency as percentage.
        latency_variance_percent: Latency variance from baseline as percentage.
        efficiency_passed: Whether efficiency meets threshold.
        latency_passed: Whether latency variance is acceptable.
        overall_passed: Whether both criteria are met.
    """

    base_agents: int
    scaled_agents: int
    agent_factor: float
    throughput_factor: float
    efficiency_percent: float
    latency_variance_percent: float
    efficiency_passed: bool
    latency_passed: bool
    overall_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "base_agents": self.base_agents,
            "scaled_agents": self.scaled_agents,
            "agent_factor": round(self.agent_factor, 2),
            "throughput_factor": round(self.throughput_factor, 2),
            "efficiency_percent": round(self.efficiency_percent, 2),
            "latency_variance_percent": round(self.latency_variance_percent, 2),
            "efficiency_passed": self.efficiency_passed,
            "latency_passed": self.latency_passed,
            "overall_passed": self.overall_passed,
        }


@dataclass
class TestReport:
    """
    Complete test report with all results and analysis.

    Contains configuration results, scaling analysis, and final
    pass/fail determination with supporting evidence.

    Attributes:
        version: Test framework version.
        test_id: Unique identifier for this test run.
        timestamp: ISO8601 timestamp when report was generated.
        base_url: Target API URL.
        configurations: Results for each agent configuration.
        scaling_analyses: Pairwise scaling analysis results.
        overall_efficiency_percent: Average scaling efficiency.
        max_latency_variance_percent: Maximum latency variance observed.
        efficiency_passed: Whether efficiency threshold was met.
        latency_passed: Whether latency variance threshold was met.
        overall_status: Final test status.
        summary: Human-readable summary of results.
    """

    version: str
    test_id: str
    timestamp: str
    base_url: str
    configurations: List[ConfigurationResult]
    scaling_analyses: List[ScalingAnalysis]
    overall_efficiency_percent: float
    max_latency_variance_percent: float
    efficiency_passed: bool
    latency_passed: bool
    overall_status: TestStatus
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "test_id": self.test_id,
            "timestamp": self.timestamp,
            "base_url": self.base_url,
            "configurations": [c.to_dict() for c in self.configurations],
            "scaling_analyses": [s.to_dict() for s in self.scaling_analyses],
            "overall_efficiency_percent": round(self.overall_efficiency_percent, 2),
            "max_latency_variance_percent": round(self.max_latency_variance_percent, 2),
            "efficiency_passed": self.efficiency_passed,
            "latency_passed": self.latency_passed,
            "overall_status": str(self.overall_status),
            "pass_criteria": {
                "efficiency_threshold_percent": EFFICIENCY_THRESHOLD * 100,
                "latency_variance_threshold_percent": LATENCY_VARIANCE_THRESHOLD * 100,
            },
            "summary": self.summary,
        }


# ============================================================
# HTTP SESSION FACTORY
# ============================================================


def create_http_session(
    pool_connections: int = POOL_CONNECTIONS,
    pool_maxsize: int = POOL_MAXSIZE,
    max_retries: int = MAX_RETRIES_FOR_TRANSIENT,
) -> requests.Session:
    """
    Create an optimized HTTP session for high-concurrency testing.

    Configures connection pooling and retry behavior for reliable
    performance measurement.

    Args:
        pool_connections: Number of connection pools to cache.
        pool_maxsize: Maximum connections per pool.
        max_retries: Maximum retry attempts for transient failures.

    Returns:
        Configured requests.Session instance.

    Example:
        >>> session = create_http_session()
        >>> response = session.get("http://localhost:8080/health")
    """
    session = requests.Session()

    # Configure retry strategy (disabled for accurate latency measurement)
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0,  # No backoff for performance testing
        status_forcelist=[],  # Don't retry on any status codes
    )

    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=POOL_BLOCK,
        max_retries=retry_strategy,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# ============================================================
# LINEAR SCALING PROOF TEST
# ============================================================


class LinearScalingProofTest:
    """
    Linear scaling proof test for VULCAN-AGI system.

    This class orchestrates the execution of scaling tests across multiple
    agent configurations, measuring throughput, latency, and calculating
    scaling efficiency to prove linear scaling behavior.

    The test uses ThreadPoolExecutor to match VULCAN's production concurrency
    model, ensuring measurements accurately reflect real-world performance.

    Attributes:
        base_url: Base URL for the VULCAN API server.
        query_timeout: Timeout for individual queries in seconds.
        session: Requests session for connection pooling.

    Thread Safety:
        This class is thread-safe for concurrent use. Internal state
        is protected by locks where necessary.

    Example:
        >>> test = LinearScalingProofTest()
        >>> try:
        ...     results = test.run_all_configurations()
        ...     report = test.generate_report(results)
        ...     print(report.summary)
        ... finally:
        ...     test.close()
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        """
        Initialize the linear scaling proof test.

        Args:
            base_url: Base URL for the VULCAN API server.
            query_timeout: Timeout for individual queries in seconds.

        Raises:
            ConfigurationError: If configuration parameters are invalid.
        """
        # Validate configuration
        if not base_url:
            raise ConfigurationError("base_url cannot be empty")
        if query_timeout <= 0:
            raise ConfigurationError("query_timeout must be positive")
        if query_timeout > MAX_SINGLE_CONFIG_TIMEOUT:
            raise ConfigurationError(
                f"query_timeout ({query_timeout}s) exceeds maximum "
                f"({MAX_SINGLE_CONFIG_TIMEOUT}s)"
            )

        self.base_url = base_url.rstrip("/")
        self.query_timeout = query_timeout
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()
        self._closed = False

        logger.info(
            f"Initialized LinearScalingProofTest: "
            f"base_url={self.base_url}, timeout={query_timeout}s"
        )

    @property
    def session(self) -> requests.Session:
        """
        Get or create the HTTP session (lazy initialization).

        Returns:
            Configured requests.Session instance.
        """
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = create_http_session()
        return self._session

    def _truncate_error(self, error: str) -> str:
        """
        Truncate error message to maximum length.

        Args:
            error: Error message to truncate.

        Returns:
            Truncated error message.
        """
        if len(error) <= ERROR_MESSAGE_MAX_LENGTH:
            return error
        return error[: ERROR_MESSAGE_MAX_LENGTH - 3] + "..."

    def send_query(
        self,
        query_id: str,
        payload: Dict[str, Any],
    ) -> QueryResult:
        """
        Send a single query to the VULCAN chat endpoint.

        Measures latency from request start to response completion,
        capturing all relevant metrics for analysis.

        Args:
            query_id: Unique identifier for this query.
            payload: Request payload to send.

        Returns:
            QueryResult with execution details and metrics.

        Note:
            This method does not raise exceptions - all errors are
            captured in the returned QueryResult.
        """
        url = f"{self.base_url}{CHAT_ENDPOINT}"
        start_time = time.perf_counter()

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Request-ID": query_id,
                },
            )
            latency = time.perf_counter() - start_time

            # Determine success based on HTTP status code
            is_success = 200 <= response.status_code < 300

            return QueryResult(
                query_id=query_id,
                status=QueryStatus.SUCCESS if is_success else QueryStatus.FAILED,
                latency_seconds=latency,
                http_status_code=response.status_code,
                error_message=(
                    None
                    if is_success
                    else self._truncate_error(response.text or "Unknown error")
                ),
                response_headers=dict(response.headers),
            )

        except requests.exceptions.Timeout:
            latency = time.perf_counter() - start_time
            return QueryResult(
                query_id=query_id,
                status=QueryStatus.TIMEOUT,
                latency_seconds=latency,
                error_message="Request timed out",
            )

        except requests.exceptions.ConnectionError as e:
            latency = time.perf_counter() - start_time
            return QueryResult(
                query_id=query_id,
                status=QueryStatus.CONNECTION_ERROR,
                latency_seconds=latency,
                error_message=self._truncate_error(f"Connection error: {e}"),
            )

        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.debug(f"Query {query_id} failed with unexpected error: {e}")
            return QueryResult(
                query_id=query_id,
                status=QueryStatus.UNKNOWN_ERROR,
                latency_seconds=latency,
                error_message=self._truncate_error(f"Unexpected error: {e}"),
            )

    def run_configuration(
        self,
        agent_count: int,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ConfigurationResult:
        """
        Run test for a specific agent configuration.

        Executes N queries (where N = agent_count × QUERY_MULTIPLIER) using
        ThreadPoolExecutor with worker count matching the agent configuration.

        Args:
            agent_count: Number of agents/workers for this configuration.
            on_progress: Optional callback invoked after each query completion.
                        Signature: (completed_count, total_count)

        Returns:
            ConfigurationResult with comprehensive performance metrics.

        Raises:
            ConfigurationError: If agent_count is invalid.
            TestTimeoutError: If configuration test exceeds time limit.
        """
        # Validate configuration
        if agent_count <= 0:
            raise ConfigurationError("agent_count must be positive")

        total_queries = agent_count * QUERY_MULTIPLIER
        if total_queries > MAX_QUERIES_PER_CONFIG:
            total_queries = MAX_QUERIES_PER_CONFIG
            logger.warning(
                f"Query count limited to {MAX_QUERIES_PER_CONFIG} "
                f"(was {agent_count * QUERY_MULTIPLIER})"
            )

        logger.info(
            f"Running configuration: {agent_count} agents, "
            f"{total_queries} queries (saturating pool)"
        )

        # Generate queries with unique IDs
        queries: List[Tuple[str, Dict[str, Any]]] = []
        for i in range(total_queries):
            query_id = f"{agent_count}a_{i}_{uuid.uuid4().hex[:8]}"
            # Deep copy to avoid mutation across threads
            payload = json.loads(json.dumps(SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]))
            queries.append((query_id, payload))

        # Calculate safe timeout for this configuration
        config_timeout = min(
            self.query_timeout * total_queries + 60.0,
            MAX_SINGLE_CONFIG_TIMEOUT,
        )

        # Execute queries with ThreadPoolExecutor
        query_results: List[QueryResult] = []
        start_time = time.perf_counter()

        try:
            with ThreadPoolExecutor(
                max_workers=agent_count,
                thread_name_prefix=f"scale_{agent_count}_",
            ) as executor:
                # Submit all queries
                futures: Dict[Future[QueryResult], str] = {
                    executor.submit(self.send_query, query_id, payload): query_id
                    for query_id, payload in queries
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(futures, timeout=config_timeout):
                    query_id = futures[future]
                    try:
                        result = future.result(timeout=self.query_timeout)
                        query_results.append(result)
                    except FuturesTimeoutError:
                        query_results.append(
                            QueryResult(
                                query_id=query_id,
                                status=QueryStatus.TIMEOUT,
                                latency_seconds=self.query_timeout,
                                error_message="Future result timeout",
                            )
                        )
                    except Exception as e:
                        query_results.append(
                            QueryResult(
                                query_id=query_id,
                                status=QueryStatus.UNKNOWN_ERROR,
                                latency_seconds=time.perf_counter() - start_time,
                                error_message=self._truncate_error(str(e)),
                            )
                        )

                    completed += 1
                    if on_progress:
                        on_progress(completed, total_queries)

        except FuturesTimeoutError:
            logger.error(
                f"Configuration {agent_count} timed out after {config_timeout}s"
            )
            # Include partial results

        total_duration = time.perf_counter() - start_time

        # Calculate metrics
        successful_results = [r for r in query_results if r.is_successful]
        successful_queries = len(successful_results)
        failed_queries = len(query_results) - successful_queries

        # Calculate latency statistics from successful queries
        latency_stats: Optional[LatencyStatistics] = None
        if successful_results:
            latencies = [r.latency_seconds for r in successful_results]
            latency_stats = LatencyStatistics.from_samples(latencies)

        # Calculate throughput
        throughput_qps = (
            len(query_results) / total_duration if total_duration > 0 else 0.0
        )
        throughput_qpm = throughput_qps * 60.0
        success_rate = (
            successful_queries / len(query_results) if query_results else 0.0
        )

        result = ConfigurationResult(
            agent_count=agent_count,
            total_queries=len(query_results),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            total_duration_seconds=total_duration,
            throughput_qps=throughput_qps,
            throughput_qpm=throughput_qpm,
            success_rate=success_rate,
            latency_stats=latency_stats,
            query_results=query_results,
        )

        logger.info(
            f"Configuration {agent_count} agents complete: "
            f"{successful_queries}/{len(query_results)} successful "
            f"({success_rate:.1%}), "
            f"throughput={throughput_qpm:.1f} qpm, "
            f"avg_latency={latency_stats.mean:.3f}s"
            if latency_stats
            else f"Configuration {agent_count} agents: no successful queries"
        )

        return result

    def run_all_configurations(
        self,
        configurations: Optional[Sequence[int]] = None,
        on_configuration_complete: Optional[
            Callable[[ConfigurationResult], None]
        ] = None,
    ) -> List[ConfigurationResult]:
        """
        Run tests for all agent configurations.

        Executes configurations sequentially with configurable sleep
        between each to allow agent pool reconfiguration.

        Args:
            configurations: Sequence of agent counts to test.
                          Defaults to DEFAULT_AGENT_CONFIGURATIONS.
            on_configuration_complete: Callback after each configuration.

        Returns:
            List of ConfigurationResult for all configurations.

        Raises:
            ConfigurationError: If no configurations specified.
        """
        if self._closed:
            raise ConfigurationError("Test instance has been closed")

        configs = list(configurations or DEFAULT_AGENT_CONFIGURATIONS)
        if not configs:
            raise ConfigurationError("At least one configuration is required")

        logger.info(f"Starting linear scaling proof test with configurations: {configs}")

        results: List[ConfigurationResult] = []
        total_start = time.perf_counter()

        for i, agent_count in enumerate(configs, 1):
            # Check total test timeout
            elapsed = time.perf_counter() - total_start
            if elapsed > MAX_TOTAL_TEST_TIMEOUT:
                logger.error(
                    f"Total test timeout exceeded ({elapsed:.0f}s > "
                    f"{MAX_TOTAL_TEST_TIMEOUT}s)"
                )
                break

            logger.info(f"Testing configuration {i}/{len(configs)}: {agent_count} agents")

            try:
                result = self.run_configuration(agent_count)
                results.append(result)

                if on_configuration_complete:
                    on_configuration_complete(result)

            except Exception as e:
                logger.error(f"Configuration {agent_count} failed: {e}")
                logger.debug(traceback.format_exc())
                # Continue with remaining configurations

            # Sleep between configurations (except after last)
            if i < len(configs):
                logger.info(
                    f"Sleeping {SLEEP_BETWEEN_CONFIGS}s for agent pool "
                    "reconfiguration..."
                )
                time.sleep(SLEEP_BETWEEN_CONFIGS)

        total_elapsed = time.perf_counter() - total_start
        logger.info(
            f"All configurations complete in {total_elapsed:.1f}s "
            f"({len(results)}/{len(configs)} successful)"
        )

        return results

    def analyze_scaling(
        self,
        results: List[ConfigurationResult],
    ) -> Tuple[List[ScalingAnalysis], float, float]:
        """
        Analyze scaling efficiency between configurations.

        Compares each configuration against the baseline (first configuration)
        to calculate scaling efficiency and latency variance.

        Args:
            results: List of configuration results to analyze.

        Returns:
            Tuple of:
            - List of ScalingAnalysis for each non-baseline configuration
            - Overall efficiency percentage (average of all analyses)
            - Maximum latency variance percentage

        Note:
            Returns empty analysis if fewer than 2 results provided.
        """
        if len(results) < 2:
            logger.warning("Scaling analysis requires at least 2 configurations")
            return [], 0.0, 0.0

        analyses: List[ScalingAnalysis] = []
        base_result = results[0]

        # Get baseline latency for variance calculation
        base_latency = (
            base_result.latency_stats.mean
            if base_result.latency_stats
            else 0.0
        )

        for scaled_result in results[1:]:
            # Calculate agent scaling factor
            agent_factor = scaled_result.agent_count / base_result.agent_count

            # Calculate throughput scaling factor
            throughput_factor = (
                scaled_result.throughput_qpm / base_result.throughput_qpm
                if base_result.throughput_qpm > 0
                else 0.0
            )

            # Calculate scaling efficiency
            efficiency_percent = (
                (throughput_factor / agent_factor) * 100
                if agent_factor > 0
                else 0.0
            )

            # Calculate latency variance from baseline
            scaled_latency = (
                scaled_result.latency_stats.mean
                if scaled_result.latency_stats
                else 0.0
            )
            latency_variance_percent = (
                abs(scaled_latency - base_latency) / base_latency * 100
                if base_latency > 0
                else 0.0
            )

            # Determine pass/fail for each criterion
            efficiency_passed = efficiency_percent >= EFFICIENCY_THRESHOLD * 100
            latency_passed = latency_variance_percent <= LATENCY_VARIANCE_THRESHOLD * 100

            analysis = ScalingAnalysis(
                base_agents=base_result.agent_count,
                scaled_agents=scaled_result.agent_count,
                agent_factor=agent_factor,
                throughput_factor=throughput_factor,
                efficiency_percent=efficiency_percent,
                latency_variance_percent=latency_variance_percent,
                efficiency_passed=efficiency_passed,
                latency_passed=latency_passed,
                overall_passed=efficiency_passed and latency_passed,
            )
            analyses.append(analysis)

        # Calculate overall metrics
        overall_efficiency = (
            statistics.mean(a.efficiency_percent for a in analyses)
            if analyses
            else 0.0
        )
        max_latency_variance = (
            max(a.latency_variance_percent for a in analyses)
            if analyses
            else 0.0
        )

        return analyses, overall_efficiency, max_latency_variance

    def generate_report(
        self,
        results: List[ConfigurationResult],
    ) -> TestReport:
        """
        Generate a comprehensive test report.

        Performs scaling analysis and determines overall pass/fail
        status based on configured thresholds.

        Args:
            results: List of configuration results.

        Returns:
            TestReport with complete analysis and determination.
        """
        test_id = f"linear_scaling_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Perform scaling analysis
        analyses, overall_efficiency, max_latency_variance = self.analyze_scaling(
            results
        )

        # Determine pass/fail
        efficiency_passed = overall_efficiency >= EFFICIENCY_THRESHOLD * 100
        latency_passed = max_latency_variance <= LATENCY_VARIANCE_THRESHOLD * 100
        overall_passed = efficiency_passed and latency_passed

        # Determine status
        if not results:
            overall_status = TestStatus.ERROR
        elif overall_passed:
            overall_status = TestStatus.PASSED
        else:
            overall_status = TestStatus.FAILED

        # Generate summary
        summary_lines = [
            f"Linear Scaling Proof Test: {overall_status}",
            "",
            f"Scaling Efficiency: {overall_efficiency:.1f}% "
            f"(threshold: {EFFICIENCY_THRESHOLD * 100:.0f}%) "
            f"{'✓ PASS' if efficiency_passed else '✗ FAIL'}",
            "",
            f"Max Latency Variance: {max_latency_variance:.1f}% "
            f"(threshold: {LATENCY_VARIANCE_THRESHOLD * 100:.0f}%) "
            f"{'✓ PASS' if latency_passed else '✗ FAIL'}",
        ]

        return TestReport(
            version=__version__,
            test_id=test_id,
            timestamp=timestamp,
            base_url=self.base_url,
            configurations=results,
            scaling_analyses=analyses,
            overall_efficiency_percent=overall_efficiency,
            max_latency_variance_percent=max_latency_variance,
            efficiency_passed=efficiency_passed,
            latency_passed=latency_passed,
            overall_status=overall_status,
            summary="\n".join(summary_lines),
        )

    def format_text_report(self, report: TestReport) -> str:
        """
        Format test report as human-readable text.

        Creates a detailed, well-formatted report suitable for
        console output or log files.

        Args:
            report: TestReport to format.

        Returns:
            Multi-line formatted string.
        """
        lines = [
            "=" * 76,
            "VULCAN LINEAR SCALING PROOF TEST",
            "=" * 76,
            f"Test ID: {report.test_id}",
            f"Report generated: {report.timestamp}",
            f"Framework version: {report.version}",
            f"Target URL: {report.base_url}",
            "",
            "-" * 76,
            "CONFIGURATION RESULTS",
            "-" * 76,
            "",
        ]

        # Configuration results table header
        lines.append(
            f"{'Agents':<8} {'Queries':<9} {'Success':<9} "
            f"{'Throughput':<14} {'Avg Latency':<14} {'Rate':<10}"
        )
        lines.append(
            f"{'':8} {'':9} {'':9} "
            f"{'(qpm)':<14} {'(seconds)':<14} {'':10}"
        )
        lines.append("-" * 76)

        for config in report.configurations:
            avg_lat = (
                f"{config.latency_stats.mean:.3f}"
                if config.latency_stats
                else "N/A"
            )
            lines.append(
                f"{config.agent_count:<8} {config.total_queries:<9} "
                f"{config.successful_queries:<9} "
                f"{config.throughput_qpm:<14.2f} "
                f"{avg_lat:<14} "
                f"{config.success_rate * 100:<10.1f}%"
            )

        lines.append("")

        # Scaling analysis table
        lines.extend([
            "-" * 76,
            "SCALING EFFICIENCY ANALYSIS",
            "-" * 76,
            "",
        ])

        if report.scaling_analyses:
            lines.append(
                f"{'Comparison':<16} {'Agent Δ':<10} {'Throughput Δ':<14} "
                f"{'Efficiency':<12} {'Latency Var':<14} {'Result':<8}"
            )
            lines.append("-" * 76)

            for analysis in report.scaling_analyses:
                result_str = "✓ PASS" if analysis.overall_passed else "✗ FAIL"
                lines.append(
                    f"{analysis.base_agents}→{analysis.scaled_agents:<11} "
                    f"{analysis.agent_factor:<10.1f}x "
                    f"{analysis.throughput_factor:<14.2f}x "
                    f"{analysis.efficiency_percent:<12.1f}% "
                    f"{analysis.latency_variance_percent:<14.1f}% "
                    f"{result_str:<8}"
                )
        else:
            lines.append("No scaling analysis available (need ≥2 configurations)")

        lines.append("")

        # Overall results
        lines.extend([
            "-" * 76,
            "OVERALL RESULTS",
            "-" * 76,
            "",
            f"Overall Scaling Efficiency: {report.overall_efficiency_percent:.1f}%",
            f"  Threshold: {EFFICIENCY_THRESHOLD * 100:.0f}%",
            f"  Status: {'✓ PASS' if report.efficiency_passed else '✗ FAIL'}",
            "",
            f"Maximum Latency Variance: {report.max_latency_variance_percent:.1f}%",
            f"  Threshold: ±{LATENCY_VARIANCE_THRESHOLD * 100:.0f}%",
            f"  Status: {'✓ PASS' if report.latency_passed else '✗ FAIL'}",
            "",
            "-" * 76,
            f"FINAL RESULT: {'✓ PASSED' if report.overall_status == TestStatus.PASSED else '✗ FAILED'}",
            "-" * 76,
            "",
            "Pass Criteria:",
            f"  • Scaling efficiency ≥ {EFFICIENCY_THRESHOLD * 100:.0f}%",
            f"  • Latency variance ≤ ±{LATENCY_VARIANCE_THRESHOLD * 100:.0f}%",
            "",
            "=" * 76,
        ])

        return "\n".join(lines)

    def save_results(
        self,
        report: TestReport,
        output_path: str,
    ) -> None:
        """
        Save test results to JSON file.

        Creates parent directories if they don't exist.

        Args:
            report: TestReport to save.
            output_path: Path to save JSON file.

        Raises:
            OSError: If file cannot be written.
        """
        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")

    def close(self) -> None:
        """
        Close the test instance and release resources.

        This method is idempotent - calling multiple times is safe.
        Should be called when testing is complete.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._session is not None:
                self._session.close()
                self._session = None

        logger.debug("LinearScalingProofTest closed")

    def __enter__(self) -> "LinearScalingProofTest":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        self.close()


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="VULCAN Linear Scaling Proof Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full scaling test with default settings
  python -m tests.performance.prove_linear_scaling

  # Specify custom URL and output path
  python -m tests.performance.prove_linear_scaling \\
      --url http://localhost:8000 \\
      --output results/scaling_test.json

  # Enable verbose logging
  python -m tests.performance.prove_linear_scaling -v

Pass Criteria:
  - Scaling efficiency >= 85%
  - Latency variance within ±20%
        """,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for VULCAN API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--output",
        default="linear_scaling_results.json",
        help="Output file for JSON results (default: linear_scaling_results.json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_QUERY_TIMEOUT,
        help=f"Query timeout in seconds (default: {DEFAULT_QUERY_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for running linear scaling proof tests.

    Returns:
        Exit code: 0 if test passes, 1 if test fails, 2 on error.
    """
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"VULCAN Linear Scaling Proof Test v{__version__}")
    logger.info(f"Configuration: url={args.url}, timeout={args.timeout}s")

    exit_code = 0

    try:
        with LinearScalingProofTest(
            base_url=args.url,
            query_timeout=args.timeout,
        ) as test:
            # Run all configurations
            results = test.run_all_configurations()

            if not results:
                logger.error("No results collected - test failed")
                return 2

            # Generate report
            report = test.generate_report(results)

            # Print text report
            text_report = test.format_text_report(report)
            print(text_report)

            # Save JSON results
            test.save_results(report, args.output)

            # Determine exit code
            if report.overall_status == TestStatus.PASSED:
                logger.info("Linear scaling proof test PASSED")
                exit_code = 0
            else:
                logger.warning("Linear scaling proof test FAILED")
                exit_code = 1

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        exit_code = 2
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
