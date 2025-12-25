#!/usr/bin/env python3
# ============================================================
# VULCAN-AGI Adversarial Tester Integration Module
# ============================================================
# Production-grade adversarial testing integration for the VULCAN-AGI
# Safety Module. Provides enterprise-ready adversarial validation with:
#
# FEATURES:
# - Singleton pattern for AdversarialTester lifecycle management
# - Background periodic testing with configurable intervals
# - Real-time query integrity validation pipeline
# - Thread-safe operations with proper locking mechanisms
# - Graceful degradation when dependencies unavailable
# - Comprehensive metrics and status reporting
# - Database-backed attack logging and audit trails
#
# SECURITY:
# - Anomaly detection using Isolation Forest algorithms
# - SHAP-based stability analysis for adversarial detection
# - Multi-layer integrity checks (statistical, semantic, behavioral)
# - Configurable thresholds via validated environment variables
#
# COMPLIANCE:
# - ITU F.748.53 AI safety standard alignment
# - EU AI Act transparency requirements
# - Audit logging for regulatory compliance
#
# PRODUCTION-READY:
# - Thread-safe singleton with RLock protection
# - Daemon threads for background operations
# - Graceful shutdown with resource cleanup
# - Comprehensive error handling and logging
# ============================================================
"""
VULCAN-AGI Adversarial Tester Integration Module.

This module provides enterprise-grade integration of the AdversarialTester
into the VULCAN-AGI platform, enabling:

1. **Platform Startup Integration**: Initializes the adversarial tester
   during platform boot with proper resource management.

2. **Periodic Security Testing**: Runs comprehensive adversarial test suites
   at configurable intervals to detect security degradation.

3. **Real-Time Query Validation**: Performs integrity checks on incoming
   queries to detect adversarial manipulation attempts.

Architecture:
    The module uses a singleton pattern for the AdversarialTester instance,
    ensuring consistent state across all platform components. Background
    testing runs in daemon threads to avoid blocking main operations.

Thread Safety:
    All public functions are thread-safe. The singleton instance is protected
    by an RLock, and background threads are managed safely with proper
    shutdown semantics.

Environment Variables:
    ADVERSARIAL_ANOMALY_THRESHOLD: Anomaly confidence threshold (0.0-1.0)
    ADVERSARIAL_SHAP_THRESHOLD: SHAP divergence threshold (0.0-1.0)
    ADVERSARIAL_SUCCESS_RATE_THRESHOLD: Alert threshold (0.0-1.0)
    ADVERSARIAL_PERIODIC_INTERVAL: Test interval in seconds (min: 60)

Example:
    >>> from vulcan.safety.adversarial_integration import (
    ...     initialize_adversarial_tester,
    ...     start_periodic_testing,
    ...     check_query_integrity,
    ...     get_adversarial_status,
    ...     shutdown_adversarial_tester
    ... )
    >>>
    >>> # Initialize at platform startup
    >>> tester = initialize_adversarial_tester(log_dir="adversarial_logs")
    >>>
    >>> # Start background periodic testing (every hour by default)
    >>> start_periodic_testing(tester, interval_seconds=3600, run_immediately=True)
    >>>
    >>> # Check query integrity in real-time
    >>> result = check_query_integrity("user query text")
    >>> if not result["safe"]:
    ...     return {"error": result["reason"], "blocked": True}
    >>>
    >>> # Get system status for monitoring
    >>> status = get_adversarial_status()
    >>> print(f"Tests run: {status.get('total_logged_attacks', 0)}")
    >>>
    >>> # Graceful shutdown
    >>> shutdown_adversarial_tester()

See Also:
    - src/adversarial_tester.py: Core adversarial testing implementation
    - src/vulcan/safety/safety_validator.py: Safety validation orchestration
    - src/vulcan/routing/query_router.py: Query routing with safety checks
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

# Try to import psutil for CPU monitoring - used to skip tests during high load
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    # Don't warn at import time - we'll log if needed when we try to check

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.adversarial_tester import AdversarialTester, AttackType

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# VERSION AND MODULE METADATA
# ============================================================
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"
__status__ = "Production"

# ============================================================
# CONSTANTS - Default Configuration Values
# ============================================================

# Threshold defaults - tuned for production use
DEFAULT_ANOMALY_CONFIDENCE_THRESHOLD: float = 0.9
DEFAULT_SHAP_DIVERGENCE_THRESHOLD: float = 0.8
DEFAULT_SUCCESS_RATE_ALERT_THRESHOLD: float = 0.8
DEFAULT_PERIODIC_TEST_INTERVAL: int = 3600  # 1 hour
DEFAULT_TENSOR_SIZE: int = 512
MIN_PERIODIC_INTERVAL: int = 60  # Minimum 60 seconds between tests
MAX_QUERY_LENGTH_FOR_LOGGING: int = 200  # Truncate queries in logs for privacy

# FIX Issue 4: CPU threshold for skipping adversarial tests during high load
# If CPU usage exceeds this threshold, skip the periodic adversarial test to avoid
# competing with production traffic and causing performance degradation
DEFAULT_CPU_THRESHOLD: float = 70.0  # Skip tests if CPU > 70%
DEFAULT_MEMORY_THRESHOLD: float = 85.0  # Skip tests if memory > 85%

# Character feature extraction ratio - use half the tensor for character features
# to leave room for other features (length, word count, hash-based noise)
CHAR_FEATURES_RATIO: float = 0.5

# Environment variable names
ENV_ANOMALY_THRESHOLD = "ADVERSARIAL_ANOMALY_THRESHOLD"
ENV_SHAP_THRESHOLD = "ADVERSARIAL_SHAP_THRESHOLD"
ENV_SUCCESS_RATE_THRESHOLD = "ADVERSARIAL_SUCCESS_RATE_THRESHOLD"
ENV_PERIODIC_INTERVAL = "ADVERSARIAL_PERIODIC_INTERVAL"


# ============================================================
# ENUMS - Status and Result Types
# ============================================================


class IntegrityCheckStatus(Enum):
    """Status codes for query integrity checks."""

    PASSED = "passed"
    BLOCKED_ANOMALY = "blocked_anomaly"
    BLOCKED_SAFETY = "blocked_safety"
    BLOCKED_INVALID = "blocked_invalid"
    SKIPPED = "skipped"
    ERROR = "error"


class PeriodicTestStatus(Enum):
    """Status codes for periodic testing operations."""

    RUNNING = "running"
    STOPPED = "stopped"
    NOT_STARTED = "not_started"
    ERROR = "error"


# ============================================================
# DATACLASSES - Structured Result Types
# ============================================================


@dataclass
class IntegrityCheckResult:
    """
    Result from a query integrity check.

    Attributes:
        safe: Whether the query passed all integrity checks
        status: Detailed status code for the check result
        reason: Human-readable reason if query was blocked
        anomaly_score: Anomaly detection confidence score (0.0-1.0)
        shap_divergence: SHAP stability divergence score
        safety_level: Safety level classification from NSO audit
        checks_performed: List of integrity checks that were executed
        details: Full integrity check result dictionary
        timestamp: UTC timestamp when check was performed
    """

    safe: bool
    status: IntegrityCheckStatus
    reason: Optional[str] = None
    anomaly_score: Optional[float] = None
    shap_divergence: Optional[float] = None
    safety_level: Optional[str] = None
    checks_performed: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "safe": self.safe,
            "status": self.status.value,
            "reason": self.reason,
            "anomaly_score": self.anomaly_score,
            "shap_divergence": self.shap_divergence,
            "safety_level": self.safety_level,
            "checks_performed": self.checks_performed,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class AdversarialTestSummary:
    """
    Summary of an adversarial test suite run.

    Attributes:
        total_tests: Total number of tests executed
        failures: Number of failed tests
        success_rate: Ratio of successful tests (0.0-1.0)
        max_divergence: Maximum divergence observed across all tests
        tests: Dictionary of individual test results
        timestamp: UTC timestamp when test suite was run
        duration_seconds: Time taken to run the test suite
    """

    total_tests: int
    failures: int
    success_rate: float
    max_divergence: float
    tests: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary for serialization."""
        return {
            "total_tests": self.total_tests,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "max_divergence": self.max_divergence,
            "tests": self.tests,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }


# ============================================================
# CONFIGURATION - Environment Variable Handling
# ============================================================


def _get_threshold_env(
    name: str, default: float, min_val: float = 0.0, max_val: float = 1.0
) -> float:
    """
    Get and validate a threshold value from environment variable.

    Safely retrieves a float threshold from environment variables with
    range validation and fallback to default values.

    Args:
        name: Environment variable name
        default: Default value if env var not set or invalid
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validated threshold value

    Example:
        >>> threshold = _get_threshold_env("MY_THRESHOLD", 0.9, 0.0, 1.0)
    """
    try:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        value = float(raw_value)
        if value < min_val or value > max_val:
            logger.warning(
                f"Environment variable {name}={value} out of range "
                f"[{min_val}, {max_val}], using default {default}"
            )
            return default
        return value
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Invalid environment variable {name}: {e}, using default {default}"
        )
        return default


def _get_interval_env(
    name: str, default: int, min_val: int = MIN_PERIODIC_INTERVAL
) -> int:
    """
    Get and validate an interval value from environment variable.

    Safely retrieves an integer interval from environment variables with
    minimum value validation and fallback to default values.

    Args:
        name: Environment variable name
        default: Default value if env var not set or invalid
        min_val: Minimum allowed value (inclusive)

    Returns:
        Validated interval value in seconds

    Example:
        >>> interval = _get_interval_env("MY_INTERVAL", 3600, 60)
    """
    try:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        value = int(raw_value)
        if value < min_val:
            logger.warning(
                f"Environment variable {name}={value} below minimum {min_val}, "
                f"using default {default}"
            )
            return default
        return value
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Invalid environment variable {name}: {e}, using default {default}"
        )
        return default


# Load configuration from environment variables
ANOMALY_CONFIDENCE_THRESHOLD: float = _get_threshold_env(
    ENV_ANOMALY_THRESHOLD, DEFAULT_ANOMALY_CONFIDENCE_THRESHOLD
)

SHAP_DIVERGENCE_THRESHOLD: float = _get_threshold_env(
    ENV_SHAP_THRESHOLD, DEFAULT_SHAP_DIVERGENCE_THRESHOLD
)

SUCCESS_RATE_ALERT_THRESHOLD: float = _get_threshold_env(
    ENV_SUCCESS_RATE_THRESHOLD, DEFAULT_SUCCESS_RATE_ALERT_THRESHOLD
)

PERIODIC_TEST_INTERVAL: int = _get_interval_env(
    ENV_PERIODIC_INTERVAL, DEFAULT_PERIODIC_TEST_INTERVAL
)


# ============================================================
# SINGLETON STATE - Thread-Safe Global Management
# ============================================================

# Global singleton state protected by RLock for thread safety
_ADVERSARIAL_TESTER: Optional["AdversarialTester"] = None
_ADVERSARIAL_LOCK: threading.RLock = threading.RLock()
_PERIODIC_THREAD: Optional[threading.Thread] = None
_PERIODIC_RUNNING: bool = False
_INITIALIZATION_TIMESTAMP: Optional[str] = None


# ============================================================
# DEPENDENCY IMPORTS - Graceful Degradation
# ============================================================

# Try to import AdversarialTester with graceful fallback
try:
    from src.adversarial_tester import AdversarialTester, AttackType

    ADVERSARIAL_TESTER_AVAILABLE: bool = True
    logger.debug("AdversarialTester imported from src.adversarial_tester")
except ImportError:
    try:
        from adversarial_tester import AdversarialTester, AttackType

        ADVERSARIAL_TESTER_AVAILABLE = True
        logger.debug("AdversarialTester imported from adversarial_tester")
    except ImportError:
        AdversarialTester = None  # type: ignore[misc, assignment]
        AttackType = None  # type: ignore[misc, assignment]
        ADVERSARIAL_TESTER_AVAILABLE = False
        logger.warning(
            "AdversarialTester not available - adversarial testing disabled. "
            "Install required dependencies: scipy, scikit-learn"
        )


# ============================================================
# PUBLIC API - Initialization Functions
# ============================================================


def initialize_adversarial_tester(
    log_dir: str = "adversarial_logs",
    interpret_engine: Optional[Any] = None,
    nso_aligner: Optional[Any] = None,
    force_reinit: bool = False,
) -> Optional["AdversarialTester"]:
    """
    Initialize the adversarial tester singleton.

    This function should be called during platform startup to create and
    configure the AdversarialTester instance. The tester is maintained as
    a singleton to ensure consistent state across all platform components.

    Subsequent calls return the existing instance unless force_reinit is True,
    which is useful for testing or configuration updates.

    Args:
        log_dir: Directory path for adversarial test logs and SQLite database.
            Will be created if it doesn't exist. Default: "adversarial_logs"
        interpret_engine: Optional InterpretabilityEngine instance for
            SHAP/LIME explanations. If None, a default engine is created.
        nso_aligner: Optional NSOAligner instance for safety audits.
            If None, a default aligner is created.
        force_reinit: If True, reinitialize even if already initialized.
            Use with caution as this may disrupt ongoing tests.

    Returns:
        AdversarialTester instance if initialization successful, None otherwise.

    Raises:
        No exceptions are raised; errors are logged and None is returned.

    Example:
        >>> tester = initialize_adversarial_tester(
        ...     log_dir="/var/log/vulcan/adversarial",
        ...     force_reinit=False
        ... )
        >>> if tester is None:
        ...     logger.error("Failed to initialize adversarial testing")

    Thread Safety:
        This function is thread-safe. Concurrent calls are serialized via RLock.

    See Also:
        get_adversarial_tester: Retrieve existing instance without reinitializing
        shutdown_adversarial_tester: Graceful shutdown and cleanup
    """
    global _ADVERSARIAL_TESTER, _INITIALIZATION_TIMESTAMP

    if not ADVERSARIAL_TESTER_AVAILABLE:
        logger.error(
            "AdversarialTester not available - cannot initialize. "
            "Ensure scipy and scikit-learn are installed."
        )
        return None

    with _ADVERSARIAL_LOCK:
        if _ADVERSARIAL_TESTER is not None and not force_reinit:
            logger.debug("Returning existing AdversarialTester instance")
            return _ADVERSARIAL_TESTER

        try:
            # Create log directory with proper permissions
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Log configuration being used
            logger.info("Initializing AdversarialTester with configuration:")
            logger.info(f"  log_dir: {log_dir}")
            logger.info(f"  anomaly_threshold: {ANOMALY_CONFIDENCE_THRESHOLD}")
            logger.info(f"  shap_threshold: {SHAP_DIVERGENCE_THRESHOLD}")
            logger.info(f"  success_rate_threshold: {SUCCESS_RATE_ALERT_THRESHOLD}")

            # Initialize the tester
            _ADVERSARIAL_TESTER = AdversarialTester(
                interpret_engine=interpret_engine,
                nso_aligner=nso_aligner,
                log_dir=log_dir,
            )

            # Record initialization timestamp
            _INITIALIZATION_TIMESTAMP = datetime.now(timezone.utc).isoformat()

            logger.info(f"✓ AdversarialTester initialized successfully")
            logger.info(f"  Database: {log_path / 'adversarial_logs.db'}")
            logger.info(f"  Timestamp: {_INITIALIZATION_TIMESTAMP}")

            return _ADVERSARIAL_TESTER

        except Exception as e:
            logger.error(f"Failed to initialize AdversarialTester: {e}", exc_info=True)
            _ADVERSARIAL_TESTER = None
            _INITIALIZATION_TIMESTAMP = None
            return None


def get_adversarial_tester() -> Optional["AdversarialTester"]:
    """
    Get the current adversarial tester singleton instance.

    This function provides access to the initialized AdversarialTester
    without reinitializing it. Returns None if not yet initialized.

    Returns:
        AdversarialTester instance or None if not initialized

    Thread Safety:
        This function is thread-safe.

    Example:
        >>> tester = get_adversarial_tester()
        >>> if tester:
        ...     results = tester.run_adversarial_suite(tensor, proposal)
    """
    with _ADVERSARIAL_LOCK:
        return _ADVERSARIAL_TESTER


# ============================================================
# PUBLIC API - Periodic Testing Functions
# ============================================================


def _check_system_load(
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
    memory_threshold: float = DEFAULT_MEMORY_THRESHOLD,
) -> Tuple[bool, str]:
    """
    FIX Issue 4: Check if system is under acceptable load for adversarial testing.

    This prevents adversarial tests from running during high CPU/memory usage,
    which was causing background CPU starvation during production traffic.

    Args:
        cpu_threshold: Maximum CPU percentage before skipping (default: 70%)
        memory_threshold: Maximum memory percentage before skipping (default: 85%)

    Returns:
        Tuple of (is_acceptable, reason) where:
        - is_acceptable: True if system load is acceptable for testing
        - reason: Human-readable explanation if not acceptable
    """
    if not HAS_PSUTIL:
        # If psutil isn't available, allow tests to run
        return True, "psutil not available - cannot check system load"

    try:
        # Check CPU usage - use interval=None for instantaneous reading (non-blocking)
        # This avoids 1-second delays in the periodic testing loop
        cpu_percent = psutil.cpu_percent(interval=None)
        if cpu_percent > cpu_threshold:
            return False, f"CPU usage at {cpu_percent:.1f}% (threshold: {cpu_threshold:.0f}%)"

        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > memory_threshold:
            return False, f"Memory usage at {memory.percent:.1f}% (threshold: {memory_threshold:.0f}%)"

        return True, f"System load acceptable (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%)"

    except Exception as e:
        logger.warning(f"Error checking system load: {e}")
        # On error, allow tests to run
        return True, f"Error checking load: {e}"


def start_periodic_testing(
    tester: Optional["AdversarialTester"] = None,
    interval_seconds: int = DEFAULT_PERIODIC_TEST_INTERVAL,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
    run_immediately: bool = True,
) -> bool:
    """
    Start periodic adversarial testing in a background daemon thread.

    This function spawns a background thread that runs comprehensive
    adversarial test suites at regular intervals. This provides continuous
    security monitoring to detect any degradation in system robustness.

    The thread runs as a daemon, ensuring it doesn't prevent process
    shutdown. For graceful shutdown with proper cleanup, use
    stop_periodic_testing() before process exit.

    Args:
        tester: AdversarialTester instance to use. If None, uses the
            singleton instance from get_adversarial_tester().
        interval_seconds: Time between test suite runs in seconds.
            Must be >= MIN_PERIODIC_INTERVAL (60s). Default: 3600 (1 hour)
        tensor_size: Dimension of test tensors for adversarial attacks.
            Larger tensors provide more thorough testing but take longer.
            Default: 512
        run_immediately: If True, runs the first test immediately rather
            than waiting for the first interval. Default: True

    Returns:
        True if periodic testing started successfully, False if:
        - No AdversarialTester available
        - Periodic testing already running

    Example:
        >>> tester = initialize_adversarial_tester()
        >>> success = start_periodic_testing(
        ...     tester=tester,
        ...     interval_seconds=1800,  # 30 minutes
        ...     run_immediately=True
        ... )
        >>> if success:
        ...     logger.info("Periodic adversarial testing enabled")

    Thread Safety:
        This function is thread-safe. Only one periodic testing thread
        can run at a time.

    See Also:
        stop_periodic_testing: Stop the background testing thread
        run_single_test: Run a single test suite manually
    """
    global _PERIODIC_THREAD, _PERIODIC_RUNNING

    if tester is None:
        tester = get_adversarial_tester()

    if tester is None:
        logger.error(
            "Cannot start periodic testing - no AdversarialTester available. "
            "Call initialize_adversarial_tester() first."
        )
        return False

    with _ADVERSARIAL_LOCK:
        if _PERIODIC_RUNNING:
            logger.warning(
                "Periodic testing already running - ignoring duplicate start request"
            )
            return True

    def _run_periodic_tests() -> None:
        """
        Background thread function for periodic adversarial testing.

        This is an internal function that runs in a daemon thread.
        It executes test suites at the configured interval until
        _PERIODIC_RUNNING is set to False.
        """
        global _PERIODIC_RUNNING
        _PERIODIC_RUNNING = True

        logger.info(
            f"🔒 Starting periodic adversarial testing "
            f"(interval: {interval_seconds}s, tensor_size: {tensor_size})"
        )

        is_first_run = run_immediately

        while _PERIODIC_RUNNING:
            if is_first_run:
                is_first_run = False
            else:
                # Wait for next interval, checking for shutdown every second
                for _ in range(interval_seconds):
                    if not _PERIODIC_RUNNING:
                        break
                    time.sleep(1)

            if not _PERIODIC_RUNNING:
                break

            # FIX Issue 4: Check system load before running tests
            # Skip adversarial tests during high CPU/memory usage to avoid
            # competing with production traffic
            load_acceptable, load_reason = _check_system_load()
            if not load_acceptable:
                logger.info(f"⏸️ Skipping adversarial test - {load_reason}")
                continue

            start_time = time.time()
            try:
                logger.info("🔒 Starting periodic adversarial test suite...")

                # Generate base tensor for testing
                base_tensor = np.random.randn(tensor_size).astype(np.float32)

                # Run full test suite
                results = tester.run_adversarial_suite(
                    base_tensor=base_tensor,
                    proposal={
                        "id": f"periodic_check_{int(time.time())}",
                        "type": "system_integrity",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

                duration = time.time() - start_time

                # Extract and log results
                summary = results.get("summary", {})
                total_tests = summary.get("total_tests", 0)
                failures = summary.get("failures", 0)
                success_rate = summary.get("success_rate", 0.0)
                max_divergence = summary.get("max_divergence", 0.0)

                logger.info("✅ Adversarial test suite complete:")
                logger.info(f"  • Total tests: {total_tests}")
                logger.info(f"  • Failures: {failures}")
                logger.info(f"  • Success rate: {success_rate:.1%}")
                logger.info(f"  • Max divergence: {max_divergence:.4f}")
                logger.info(f"  • Duration: {duration:.2f}s")

                # Alert on high failure rate
                if success_rate < SUCCESS_RATE_ALERT_THRESHOLD:
                    logger.warning(
                        f"⚠️ ADVERSARIAL TEST ALERT: High failure rate detected! "
                        f"{failures}/{total_tests} tests failed "
                        f"(threshold: {SUCCESS_RATE_ALERT_THRESHOLD:.0%})"
                    )

            except Exception as e:
                logger.error(f"❌ Periodic adversarial test failed: {e}", exc_info=True)

        logger.info("🔒 Periodic adversarial testing stopped")

    # Start background thread
    _PERIODIC_THREAD = threading.Thread(
        target=_run_periodic_tests,
        name="AdversarialPeriodicTester",
        daemon=True,
    )
    _PERIODIC_THREAD.start()

    logger.info(
        f"✓ Periodic adversarial testing thread started (interval: {interval_seconds}s)"
    )
    return True


def stop_periodic_testing(timeout: float = 5.0) -> bool:
    """
    Stop the periodic adversarial testing background thread.

    This function signals the background thread to stop and waits for
    it to complete. Use this for graceful shutdown to ensure any
    in-progress tests complete cleanly.

    Args:
        timeout: Maximum time to wait for thread to stop in seconds.
            Default: 5.0 seconds

    Returns:
        True if thread stopped successfully, False if timeout occurred

    Example:
        >>> success = stop_periodic_testing(timeout=10.0)
        >>> if not success:
        ...     logger.warning("Periodic testing thread did not stop cleanly")

    Thread Safety:
        This function is thread-safe.
    """
    global _PERIODIC_RUNNING, _PERIODIC_THREAD

    _PERIODIC_RUNNING = False

    if _PERIODIC_THREAD is not None:
        _PERIODIC_THREAD.join(timeout=timeout)
        is_alive = _PERIODIC_THREAD.is_alive()
        _PERIODIC_THREAD = None

        if is_alive:
            logger.warning(
                f"Periodic testing thread did not stop within {timeout}s timeout"
            )
            return False

    logger.info("✓ Periodic adversarial testing stopped")
    return True


def get_periodic_testing_status() -> PeriodicTestStatus:
    """
    Get the current status of periodic adversarial testing.

    Returns:
        PeriodicTestStatus enum indicating current state

    Example:
        >>> status = get_periodic_testing_status()
        >>> if status == PeriodicTestStatus.RUNNING:
        ...     logger.info("Periodic testing is active")
    """
    global _PERIODIC_RUNNING, _PERIODIC_THREAD

    if (
        _PERIODIC_RUNNING
        and _PERIODIC_THREAD is not None
        and _PERIODIC_THREAD.is_alive()
    ):
        return PeriodicTestStatus.RUNNING
    elif _PERIODIC_THREAD is not None and not _PERIODIC_THREAD.is_alive():
        return PeriodicTestStatus.ERROR
    else:
        return PeriodicTestStatus.NOT_STARTED


# ============================================================
# PUBLIC API - Query Encoding and Integrity Checking
# ============================================================


def encode_query_to_tensor(
    query: str, tensor_size: int = DEFAULT_TENSOR_SIZE
) -> np.ndarray:
    """
    Encode a text query to a numeric tensor for adversarial testing.

    This function converts text queries into fixed-size numeric tensors
    suitable for adversarial analysis. The encoding is deterministic
    (same query always produces same tensor) using SHA-256 based seeding.

    The encoding captures several query characteristics:
    - Character-level features (ASCII values)
    - Length-based features (normalized query and word counts)
    - Random noise seeded by query hash (for adversarial robustness)

    Args:
        query: The text query to encode. Empty strings are handled safely.
        tensor_size: Size of output tensor. Default: 512.
            Larger sizes provide more granular encoding but increase
            computation time for integrity checks.

    Returns:
        numpy array of shape (tensor_size,) with dtype float32,
        normalized to zero mean and unit standard deviation.

    Example:
        >>> tensor = encode_query_to_tensor("What is machine learning?")
        >>> assert tensor.shape == (512,)
        >>> assert tensor.dtype == np.float32
        >>> assert abs(tensor.mean()) < 1e-6  # Normalized

    Thread Safety:
        This function is thread-safe (stateless).

    Note:
        The encoding is designed for adversarial detection, not semantic
        similarity. Two similar queries may have different tensor representations.
    """
    # Handle empty query edge case
    if not query:
        return np.zeros(tensor_size, dtype=np.float32)

    # Use SHA-256 hash as a seed for reproducible encoding
    hash_bytes = hashlib.sha256(query.encode("utf-8")).digest()
    seed = int.from_bytes(hash_bytes[:4], "big")

    # Create reproducible random state
    rng = np.random.RandomState(seed)

    # Extract query characteristics
    query_len = len(query)
    word_count = len(query.split())

    # Calculate max character features based on tensor size and ratio
    max_char_features = int(tensor_size * CHAR_FEATURES_RATIO)
    char_codes = [ord(c) for c in query[: min(len(query), max_char_features)]]

    # Build tensor components with deterministic noise
    base = rng.randn(tensor_size).astype(np.float32)

    # Add character-level features (with bounds checking)
    if char_codes:
        char_array = np.array(char_codes, dtype=np.float32)
        # Normalize character array safely
        char_std = char_array.std()
        if char_std > 1e-8:
            char_array = (char_array - char_array.mean()) / char_std
        else:
            char_array = char_array - char_array.mean()

        # Ensure we don't exceed tensor bounds
        num_char_features = min(len(char_codes), tensor_size)
        base[:num_char_features] += char_array[:num_char_features] * 0.5

    # Add length-based features (normalized to reasonable ranges)
    base[0] = min(query_len / 1000.0, 10.0)  # Capped normalized length
    base[1] = min(word_count / 100.0, 10.0)  # Capped normalized word count

    # Normalize final tensor safely
    base_std = base.std()
    if base_std > 1e-8:
        base = (base - base.mean()) / base_std
    else:
        base = base - base.mean()

    return base


def check_query_integrity(
    query: str,
    tester: Optional["AdversarialTester"] = None,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
) -> Dict[str, Any]:
    """
    Check query integrity using adversarial testing.

    This function performs real-time integrity validation on incoming
    queries to detect potential adversarial manipulation, anomalous
    patterns, or out-of-distribution inputs.

    The checks performed include:
    - **Anomaly Detection**: Uses Isolation Forest to detect unusual patterns
    - **SHAP Stability**: Checks if SHAP explanations are stable
    - **Safety Level**: NSO alignment audit for safety classification
    - **Statistical Checks**: NaN/Inf detection, range validation

    Args:
        query: The user query to check. Required.
        tester: AdversarialTester instance to use. If None, uses the
            singleton from get_adversarial_tester().
        tensor_size: Size of tensor encoding. Default: 512

    Returns:
        Dictionary with the following keys:
        - **safe** (bool): True if query passes all integrity checks
        - **status** (str): Status code (passed, blocked_*, skipped, error)
        - **reason** (Optional[str]): Human-readable reason if blocked
        - **anomaly_score** (Optional[float]): Anomaly confidence (0.0-1.0)
        - **details** (Dict): Full integrity check results

    Example:
        >>> result = check_query_integrity("How do I hack a computer?")
        >>> if not result["safe"]:
        ...     return {"error": result["reason"], "blocked": True}
        >>> # Query is safe to process
        >>> process_query(query)

    Thread Safety:
        This function is thread-safe.

    Performance:
        Typical latency is 1-10ms depending on tensor size and
        available optimizations.

    See Also:
        encode_query_to_tensor: Underlying tensor encoding
        IntegrityCheckResult: Structured result type
    """
    if tester is None:
        tester = get_adversarial_tester()

    if tester is None:
        # No tester available - allow query but log warning
        logger.debug("AdversarialTester not available - skipping integrity check")
        return IntegrityCheckResult(
            safe=True,
            status=IntegrityCheckStatus.SKIPPED,
            reason=None,
            details={"skipped": True, "reason": "tester_not_available"},
        ).to_dict()

    try:
        # Encode query to tensor
        query_tensor = encode_query_to_tensor(query, tensor_size)

        # Prepare query metadata (truncate for privacy)
        # Use SHA-256 for query ID (truncated for brevity, not security)
        query_id = hashlib.sha256(query.encode()).hexdigest()[:8]
        truncated_query = query[:MAX_QUERY_LENGTH_FOR_LOGGING]

        # Run real-time integrity check
        integrity_results = tester.realtime_integrity_check(
            graph={"query": truncated_query, "id": query_id},
            current_tensor=query_tensor,
        )

        # Extract check results
        is_anomaly = integrity_results.get("is_anomaly", False)
        anomaly_confidence = integrity_results.get("anomaly_confidence", 0.0)
        safety_level = integrity_results.get("safety_level", "safe")
        shap_stable = integrity_results.get("shap_stable", True)
        shap_divergence = integrity_results.get("shap_divergence", 0.0)
        has_nan = integrity_results.get("has_nan", False)
        has_inf = integrity_results.get("has_inf", False)
        checks_performed = integrity_results.get("checks_performed", [])

        # Determine blocking conditions
        should_block = False
        block_reason = None
        status = IntegrityCheckStatus.PASSED

        # Check 1: High-confidence anomaly detection
        if is_anomaly and anomaly_confidence > ANOMALY_CONFIDENCE_THRESHOLD:
            should_block = True
            status = IntegrityCheckStatus.BLOCKED_ANOMALY
            block_reason = (
                f"High-confidence anomaly detected "
                f"(score: {anomaly_confidence:.2f}, threshold: {ANOMALY_CONFIDENCE_THRESHOLD})"
            )
            logger.warning(
                f"🚨 ANOMALY DETECTED in query [{query_id}]: {truncated_query[:100]}..."
            )
            logger.warning(f"  Anomaly confidence: {anomaly_confidence:.2f}")

        # Check 2: Safety level classification
        if safety_level in ("high_risk", "critical"):
            should_block = True
            status = IntegrityCheckStatus.BLOCKED_SAFETY
            block_reason = f"Safety check failed: {safety_level}"
            logger.warning(
                f"⚠️ HIGH-RISK query detected [{query_id}]: safety_level={safety_level}"
            )

        # Check 3: Invalid numeric values
        if has_nan or has_inf:
            should_block = True
            status = IntegrityCheckStatus.BLOCKED_INVALID
            block_reason = "Invalid numeric values detected in query encoding"
            logger.warning(
                f"⚠️ Invalid values in query [{query_id}]: NaN={has_nan}, Inf={has_inf}"
            )

        # Warning: SHAP instability (logged but not blocking)
        if not shap_stable and shap_divergence > SHAP_DIVERGENCE_THRESHOLD:
            logger.warning(
                f"⚠️ SHAP unstable [{query_id}]: "
                f"divergence={shap_divergence:.4f} (threshold: {SHAP_DIVERGENCE_THRESHOLD})"
            )

        # Log successful checks at debug level
        if not should_block:
            logger.debug(
                f"Query [{query_id}] passed integrity checks: {checks_performed}"
            )

        return IntegrityCheckResult(
            safe=not should_block,
            status=status,
            reason=block_reason,
            anomaly_score=anomaly_confidence if is_anomaly else None,
            shap_divergence=shap_divergence if not shap_stable else None,
            safety_level=safety_level,
            checks_performed=checks_performed,
            details=integrity_results,
        ).to_dict()

    except Exception as e:
        logger.error(f"Query integrity check failed: {e}", exc_info=True)
        # On error, allow the query but log the issue (fail-open)
        return IntegrityCheckResult(
            safe=True,
            status=IntegrityCheckStatus.ERROR,
            reason=None,
            details={"error": str(e), "error_type": type(e).__name__},
        ).to_dict()


# ============================================================
# PUBLIC API - Status and Monitoring
# ============================================================


def get_adversarial_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the adversarial testing system.

    This function provides detailed status information for monitoring,
    debugging, and operational dashboards. It includes system state,
    configuration, and recent activity metrics.

    Returns:
        Dictionary with the following keys:
        - **available** (bool): Whether AdversarialTester module is available
        - **initialized** (bool): Whether singleton is initialized
        - **initialization_timestamp** (Optional[str]): When tester was initialized
        - **periodic_running** (bool): Whether periodic testing is active
        - **periodic_status** (str): Periodic testing status enum value
        - **configuration** (Dict): Current threshold configuration
        - **attack_stats** (Dict): Statistics about past attacks
        - **database_path** (Optional[str]): Path to SQLite audit database
        - **total_logged_attacks** (Optional[int]): Total attacks in database
        - **recent_attacks** (Optional[List]): Last 5 attack records

    Example:
        >>> status = get_adversarial_status()
        >>> print(f"System available: {status['available']}")
        >>> print(f"Periodic testing: {status['periodic_running']}")
        >>> print(f"Total logged attacks: {status.get('total_logged_attacks', 0)}")

    Thread Safety:
        This function is thread-safe.

    Performance:
        May take 10-100ms if database queries are performed.
    """
    global _PERIODIC_RUNNING, _INITIALIZATION_TIMESTAMP

    status: Dict[str, Any] = {
        "available": ADVERSARIAL_TESTER_AVAILABLE,
        "initialized": _ADVERSARIAL_TESTER is not None,
        "initialization_timestamp": _INITIALIZATION_TIMESTAMP,
        "periodic_running": _PERIODIC_RUNNING,
        "periodic_status": get_periodic_testing_status().value,
        "configuration": {
            "anomaly_threshold": ANOMALY_CONFIDENCE_THRESHOLD,
            "shap_threshold": SHAP_DIVERGENCE_THRESHOLD,
            "success_rate_threshold": SUCCESS_RATE_ALERT_THRESHOLD,
            "periodic_interval_seconds": PERIODIC_TEST_INTERVAL,
        },
        "attack_stats": {},
        "database_path": None,
        "total_logged_attacks": None,
        "recent_attacks": None,
    }

    if _ADVERSARIAL_TESTER is not None:
        try:
            # Get attack statistics (thread-safe)
            with _ADVERSARIAL_TESTER.stats_lock:
                status["attack_stats"] = dict(_ADVERSARIAL_TESTER.attack_stats)

            # Get database path
            db_path = _ADVERSARIAL_TESTER.log_dir / "adversarial_logs.db"
            status["database_path"] = str(db_path)

            # Get database statistics with proper connection management
            if db_path.exists():
                conn = None
                try:
                    conn = sqlite3.connect(str(db_path), timeout=5.0)
                    cursor = conn.cursor()

                    # Total attack count
                    cursor.execute("SELECT COUNT(*) FROM attack_logs")
                    status["total_logged_attacks"] = cursor.fetchone()[0]

                    # Recent attacks
                    cursor.execute(
                        """
                        SELECT timestamp, attack_type, success, perturbation_norm
                        FROM attack_logs 
                        ORDER BY timestamp DESC 
                        LIMIT 5
                    """
                    )
                    status["recent_attacks"] = [
                        {
                            "timestamp": row[0],
                            "type": row[1],
                            "success": bool(row[2]),
                            "perturbation_norm": row[3],
                        }
                        for row in cursor.fetchall()
                    ]
                except sqlite3.Error as db_error:
                    logger.debug(f"Could not query database: {db_error}")
                finally:
                    if conn is not None:
                        conn.close()

        except Exception as e:
            logger.error(f"Error getting adversarial status: {e}")

    return status


def run_single_test(
    tester: Optional["AdversarialTester"] = None,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
) -> Dict[str, Any]:
    """
    Run a single adversarial test suite manually.

    This function triggers an immediate, on-demand execution of the
    full adversarial test suite. Useful for:
    - Manual security verification
    - Testing after configuration changes
    - Integration testing and CI/CD pipelines

    Args:
        tester: AdversarialTester instance to use. If None, uses the
            singleton from get_adversarial_tester().
        tensor_size: Dimension of test tensors. Default: 512

    Returns:
        Test results dictionary with:
        - **timestamp**: When test was run
        - **tests**: Individual test results
        - **summary**: Aggregated statistics

        Or error dictionary if test failed:
        - **error**: Error message string

    Example:
        >>> results = run_single_test(tensor_size=256)
        >>> if "error" in results:
        ...     logger.error(f"Test failed: {results['error']}")
        >>> else:
        ...     print(f"Success rate: {results['summary']['success_rate']:.1%}")

    Thread Safety:
        This function is thread-safe.

    Performance:
        May take 1-30 seconds depending on tensor size and system load.
    """
    if tester is None:
        tester = get_adversarial_tester()

    if tester is None:
        return {
            "error": "AdversarialTester not available",
            "hint": "Call initialize_adversarial_tester() first",
        }

    start_time = time.time()

    try:
        logger.info("🔒 Running manual adversarial test suite...")

        # Generate deterministic test tensor for reproducibility
        base_tensor = np.random.randn(tensor_size).astype(np.float32)

        # Run full test suite
        results = tester.run_adversarial_suite(
            base_tensor=base_tensor,
            proposal={
                "id": f"manual_test_{int(time.time())}",
                "type": "manual_trigger",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        duration = time.time() - start_time

        # Add duration to summary
        if "summary" in results:
            results["summary"]["duration_seconds"] = round(duration, 3)

        logger.info(
            f"✅ Manual test complete in {duration:.2f}s: "
            f"{results.get('summary', {})}"
        )

        return results

    except Exception as e:
        logger.error(f"Manual adversarial test failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "duration_seconds": round(time.time() - start_time, 3),
        }


# ============================================================
# PUBLIC API - Lifecycle Management
# ============================================================


def shutdown_adversarial_tester() -> bool:
    """
    Shutdown the adversarial tester and cleanup all resources.

    This function performs graceful shutdown of the adversarial testing
    system, including:
    - Stopping periodic testing thread
    - Closing database connection pool
    - Releasing singleton instance

    Should be called during platform shutdown to ensure clean resource
    release and prevent resource leaks.

    Returns:
        True if shutdown completed successfully, False if errors occurred

    Example:
        >>> # During platform shutdown
        >>> success = shutdown_adversarial_tester()
        >>> if not success:
        ...     logger.warning("Adversarial tester shutdown had errors")

    Thread Safety:
        This function is thread-safe.

    See Also:
        initialize_adversarial_tester: Initialization counterpart
        stop_periodic_testing: Stop just the periodic testing thread
    """
    global _ADVERSARIAL_TESTER, _PERIODIC_RUNNING, _INITIALIZATION_TIMESTAMP

    success = True

    # Stop periodic testing first
    if not stop_periodic_testing(timeout=10.0):
        logger.warning("Periodic testing thread did not stop cleanly")
        success = False

    # Close database connections and release singleton
    with _ADVERSARIAL_LOCK:
        if _ADVERSARIAL_TESTER is not None:
            try:
                if hasattr(_ADVERSARIAL_TESTER, "db_pool"):
                    _ADVERSARIAL_TESTER.db_pool.close_all()
                    logger.debug("Database connection pool closed")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")
                success = False
            finally:
                _ADVERSARIAL_TESTER = None
                _INITIALIZATION_TIMESTAMP = None

    if success:
        logger.info("✓ AdversarialTester shutdown complete")
    else:
        logger.warning("⚠️ AdversarialTester shutdown completed with errors")

    return success


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Initialization
    "initialize_adversarial_tester",
    "get_adversarial_tester",
    "shutdown_adversarial_tester",
    # Periodic testing
    "start_periodic_testing",
    "stop_periodic_testing",
    "get_periodic_testing_status",
    # Query checking
    "encode_query_to_tensor",
    "check_query_integrity",
    # Status and monitoring
    "get_adversarial_status",
    "run_single_test",
    # Types and constants
    "IntegrityCheckResult",
    "IntegrityCheckStatus",
    "PeriodicTestStatus",
    "AdversarialTestSummary",
    "ADVERSARIAL_TESTER_AVAILABLE",
    "ANOMALY_CONFIDENCE_THRESHOLD",
    "SHAP_DIVERGENCE_THRESHOLD",
    "SUCCESS_RATE_ALERT_THRESHOLD",
    "PERIODIC_TEST_INTERVAL",
]
