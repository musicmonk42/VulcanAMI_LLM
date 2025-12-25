# ============================================================
# VULCAN-AGI Governance Logger - Audit and Compliance Logging
# ============================================================
# Enterprise-grade governance logging for oversight and compliance:
# - audit_log: All significant actions
# - compliance_checks: PII/sensitive data detection
# - quarantine_log: Risky action detection
#
# PRODUCTION-READY: Thread-safe, SQLite-backed, comprehensive logging
# COMPLIANCE: SOC2, GDPR, HIPAA-aware logging patterns
# ============================================================

"""
VULCAN Governance Logger

Logs actions to audit.db and governance.db for oversight and compliance.

Features:
    - Comprehensive audit logging
    - Compliance check recording
    - Quarantine log for risky actions
    - Policy violation tracking
    - Paginated log retrieval

Tables:
    audit_log: All significant actions with timestamps and metadata
    compliance_checks: PII/sensitive data detection results
    quarantine_log: Blocked or flagged risky actions
    policy_violations: Policy rule violations

Thread Safety:
    All public methods are thread-safe. Uses SQLite with WAL mode
    for concurrent read/write access.

Usage:
    from vulcan.routing import log_to_governance, get_governance_logger

    # Log an action
    log_to_governance("query_processed", {
        "query_id": "q_123",
        "query_type": "reasoning"
    })

    # Get logger for detailed operations
    logger = get_governance_logger()
    logger.log_compliance_check("pii_check", passed=False, {...})
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Generator, List, Optional

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Default database paths
DEFAULT_AUDIT_DB_PATH = Path("data/audit.db")
DEFAULT_GOVERNANCE_DB_PATH = Path("data/governance.db")

# SQLite configuration
SQLITE_TIMEOUT = 30.0  # seconds
SQLITE_PRAGMAS = [
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA cache_size=-64000",  # 64MB cache
    "PRAGMA temp_store=MEMORY",
]

# Query limits
MAX_LOG_QUERY_LIMIT = 1000
DEFAULT_LOG_QUERY_LIMIT = 100

# Request truncation limit
MAX_REQUEST_LENGTH = 2000

# PERFORMANCE FIX: Thread pool for async I/O operations
# This allows governance logging to run without blocking the asyncio event loop
# Configurable via environment variable GOVERNANCE_IO_WORKERS (default: min(4, cpu_count+1))
_GOVERNANCE_IO_EXECUTOR: Optional[ThreadPoolExecutor] = None
_GOVERNANCE_EXECUTOR_LOCK = threading.Lock()

# Calculate default worker count based on CPU count
_DEFAULT_GOVERNANCE_WORKERS = min(4, (os.cpu_count() or 1) + 1)

# ============================================================
# BUFFERED LOGGING CONFIGURATION
# ============================================================
# PERFORMANCE FIX: Non-blocking governance logging with buffered I/O
# This addresses the root cause of synchronous governance logging blocking
# on growing file I/O by using:
# - Bounded buffer (prevents memory leak)
# - Background thread flush (prevents per-query blocking)
# - Rotating log files (prevents single file growing indefinitely)

DEFAULT_BUFFER_MAXLEN = 500  # Maximum entries in buffer before oldest are dropped
DEFAULT_FLUSH_INTERVAL = 5.0  # Flush buffer to disk every N seconds
DEFAULT_LOG_PATH = Path("governance_logs")  # Default directory for JSONL log files


def _get_governance_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool for async governance I/O operations."""
    global _GOVERNANCE_IO_EXECUTOR
    if _GOVERNANCE_IO_EXECUTOR is None:
        with _GOVERNANCE_EXECUTOR_LOCK:
            if _GOVERNANCE_IO_EXECUTOR is None:
                max_workers = int(
                    os.getenv("GOVERNANCE_IO_WORKERS", str(_DEFAULT_GOVERNANCE_WORKERS))
                )
                _GOVERNANCE_IO_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="governance_io"
                )
                # Register cleanup on application exit
                atexit.register(_cleanup_governance_executor)
    return _GOVERNANCE_IO_EXECUTOR


def _cleanup_governance_executor():
    """Cleanup the governance I/O executor on application shutdown."""
    global _GOVERNANCE_IO_EXECUTOR
    if _GOVERNANCE_IO_EXECUTOR is not None:
        try:
            _GOVERNANCE_IO_EXECUTOR.shutdown(wait=False)
            logger.debug("Governance I/O executor shut down")
        except Exception as e:
            logger.warning(f"Error shutting down governance executor: {e}")
        _GOVERNANCE_IO_EXECUTOR = None


# ============================================================
# ENUMS
# ============================================================


class ActionType(str, Enum):
    """Types of actions that can be logged."""

    QUERY_PROCESSED = "query_processed"
    AGENT_TASK_SUBMITTED = "agent_task_submitted"
    AGENT_TASK_COMPLETED = "agent_task_completed"
    AGENT_COLLABORATION = "agent_collaboration"
    GOVERNANCE_CHECK = "governance_check"
    PII_DETECTED = "pii_detected"
    SENSITIVE_TOPIC = "sensitive_topic"
    SELF_MODIFICATION = "self_modification"
    CODE_GENERATION = "code_generation"
    EXPERIMENT_TRIGGERED = "experiment_triggered"
    EXPERIMENT_COMPLETED = "experiment_completed"
    COMPLIANCE_VIOLATION = "compliance_violation"
    QUARANTINE_ACTION = "quarantine_action"
    RESPONSE_GENERATED = "response_generated"
    ERROR_OCCURRED = "error_occurred"
    TOURNAMENT_STARTED = "tournament_started"
    TOURNAMENT_COMPLETED = "tournament_completed"
    USER_FEEDBACK = "user_feedback"


class SeverityLevel(str, Enum):
    """Severity levels for logged actions."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class AuditEntry:
    """
    Entry for the audit log.

    Attributes:
        action_type: Type of action being logged
        severity: Severity level
        actor: System component or user identifier
        query_id: Associated query ID
        session_id: Associated session ID
        details: Action details as dictionary
        timestamp: Entry timestamp
    """

    action_type: ActionType
    severity: SeverityLevel
    actor: str
    query_id: Optional[str]
    session_id: Optional[str]
    details: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "query_id": self.query_id,
            "session_id": self.session_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ============================================================
# BUFFERED GOVERNANCE LOGGER CLASS
# ============================================================


class BufferedGovernanceLogger:
    """
    Non-blocking governance logger with buffered writes.

    PERFORMANCE FIX: This class addresses the root cause of synchronous
    governance logging blocking on growing file I/O. Instead of writing
    to disk on every log call (which causes increasing latency as files grow),
    this logger:

    1. Appends entries to a bounded in-memory buffer (deque with maxlen)
    2. Background thread flushes buffer to disk every N seconds
    3. Writes to rotating hourly JSONL files (not growing single files)

    This changes logging from O(file_size) to O(1) for each query, fixing
    the performance degradation pattern (11s -> 18s -> 33s -> 63s).

    Thread Safety:
        All public methods are thread-safe. Uses threading.Lock for buffer
        access and daemon thread for background flushing.

    Usage:
        from vulcan.routing.governance_logger import BufferedGovernanceLogger

        logger = BufferedGovernanceLogger()

        # Non-blocking log - just appends to buffer
        logger.log("q_123", {"route": "reasoning", "complexity": 0.7})

        # Buffer will be flushed to disk by background thread every 5s
        # On shutdown, remaining entries are flushed via atexit handler

    Args:
        log_path: Directory for log files (default: "governance_logs")
        buffer_maxlen: Maximum buffer size before oldest entries are dropped (default: 100)
        flush_interval: Seconds between background flushes (default: 5.0)
    """

    def __init__(
        self,
        log_path: str = "governance_logs",
        buffer_maxlen: int = DEFAULT_BUFFER_MAXLEN,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    ):
        """
        Initialize the buffered governance logger.

        Args:
            log_path: Directory for rotating log files
            buffer_maxlen: Maximum entries in buffer (default: 100)
            flush_interval: Seconds between background flushes (default: 5.0)
        """
        self.log_path = Path(log_path)
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_maxlen)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gov_flush")
        self._flush_interval = flush_interval
        self._shutdown = False
        self._stats = {
            "entries_logged": 0,
            "entries_flushed": 0,
            "entries_dropped": 0,  # Due to buffer overflow
            "flush_count": 0,
            "errors": 0,
        }

        # Ensure log directory exists
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Start background flush thread
        self._start_background_flush()

        # Register cleanup on application exit
        atexit.register(self._atexit_handler)

        logger.debug(
            f"BufferedGovernanceLogger initialized: log_path={self.log_path}, "
            f"buffer_maxlen={buffer_maxlen}, flush_interval={flush_interval}s"
        )

    def log(self, query_id: str, routing_result: Dict[str, Any]) -> None:
        """
        Non-blocking log - just append to buffer.

        This method returns immediately after appending to the in-memory
        buffer. Actual disk I/O happens in the background flush thread.

        Args:
            query_id: Unique identifier for the query
            routing_result: Routing decision data to log
        """
        entry = {
            "query_id": query_id,
            "timestamp": time.time(),
            "result": routing_result,
        }

        with self._lock:
            # Check if buffer is full (will drop oldest)
            was_full = len(self._buffer) == self._buffer.maxlen
            self._buffer.append(entry)

            self._stats["entries_logged"] += 1
            if was_full:
                self._stats["entries_dropped"] += 1

        # DON'T write to disk here - return immediately
        logger.debug(f"[BufferedGovernanceLogger] Buffered entry for query_id={query_id}")

    def _start_background_flush(self) -> None:
        """Start the background thread that flushes buffer to disk."""

        def _flush_loop() -> None:
            """Flush buffer to disk at regular intervals."""
            while not self._shutdown:
                time.sleep(self._flush_interval)
                if not self._shutdown:  # Check again after sleep
                    self._flush_to_disk()

        thread = threading.Thread(target=_flush_loop, daemon=True, name="gov_flush_loop")
        thread.start()
        logger.debug("Background flush thread started")

    def _flush_to_disk(self) -> None:
        """Batch write buffer contents to rotating log file."""
        with self._lock:
            if not self._buffer:
                return
            entries = list(self._buffer)
            self._buffer.clear()

        if not entries:
            return

        try:
            # Write to hourly rotating log file (not growing single file)
            hour_timestamp = int(time.time() // 3600)
            log_file = self.log_path / f"gov_{hour_timestamp}.jsonl"

            with open(log_file, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            with self._lock:
                self._stats["entries_flushed"] += len(entries)
                self._stats["flush_count"] += 1

            logger.debug(
                f"[BufferedGovernanceLogger] Flushed {len(entries)} entries to {log_file}"
            )

        except Exception as e:
            logger.error(f"[BufferedGovernanceLogger] Flush failed: {e}", exc_info=True)
            with self._lock:
                self._stats["errors"] += 1
            # Re-add entries to buffer on failure (if space permits)
            # We don't want to lose data, but also can't block indefinitely
            with self._lock:
                for entry in reversed(entries):
                    if len(self._buffer) < self._buffer.maxlen:
                        self._buffer.appendleft(entry)

    def _atexit_handler(self) -> None:
        """Flush remaining buffer on application shutdown."""
        # Check if we're in pytest cleanup mode
        if os.environ.get("PYTEST_CLEANUP_DONE") == "1":
            return

        self._shutdown = True
        try:
            # Final flush
            self._flush_to_disk()
            # Shutdown executor
            self._executor.shutdown(wait=False)
            logger.debug("BufferedGovernanceLogger shutdown complete")
        except Exception as e:
            logger.warning(f"Error during BufferedGovernanceLogger shutdown: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["buffer_size"] = len(self._buffer)
            stats["buffer_maxlen"] = self._buffer.maxlen
        return stats

    def flush_now(self) -> None:
        """
        Force an immediate flush of the buffer.

        Useful for testing or when you need to ensure logs are persisted.
        """
        self._flush_to_disk()


# ============================================================
# BUFFERED LOGGER SINGLETON
# ============================================================

_buffered_logger: Optional[BufferedGovernanceLogger] = None
_buffered_logger_lock = threading.Lock()


def get_buffered_governance_logger(
    log_path: str = "governance_logs",
    buffer_maxlen: int = DEFAULT_BUFFER_MAXLEN,
    flush_interval: float = DEFAULT_FLUSH_INTERVAL,
) -> BufferedGovernanceLogger:
    """
    Get or create the global buffered governance logger (thread-safe singleton).

    Args:
        log_path: Directory for log files
        buffer_maxlen: Maximum buffer size
        flush_interval: Seconds between flushes

    Returns:
        BufferedGovernanceLogger instance
    """
    global _buffered_logger

    if _buffered_logger is None:
        with _buffered_logger_lock:
            if _buffered_logger is None:
                _buffered_logger = BufferedGovernanceLogger(
                    log_path=log_path,
                    buffer_maxlen=buffer_maxlen,
                    flush_interval=flush_interval,
                )
                logger.debug("Global BufferedGovernanceLogger instance created")

    return _buffered_logger


def log_routing_result(query_id: str, routing_result: Dict[str, Any]) -> None:
    """
    Non-blocking convenience function to log a routing result.

    This is the recommended way to log governance data from the query router.
    It uses the buffered logger which returns immediately without disk I/O.

    Args:
        query_id: Unique identifier for the query
        routing_result: Routing decision data to log
    """
    buffered_logger = get_buffered_governance_logger()
    buffered_logger.log(query_id, routing_result)


# ============================================================
# GOVERNANCE LOGGER CLASS
# ============================================================


class GovernanceLogger:
    """
    Logs actions to audit.db and governance.db for oversight.

    Thread-safe implementation with SQLite backend using WAL mode
    for concurrent access. Supports comprehensive audit logging,
    compliance checking, and quarantine management.

    Usage:
        logger = GovernanceLogger()

        # Log an action
        logger.log_action(
            ActionType.QUERY_PROCESSED,
            {"query_type": "reasoning", "complexity": 0.7},
            query_id="q_123"
        )

        # Log compliance check
        logger.log_compliance_check(
            "pii_check",
            passed=False,
            {"field": "email", "action": "redacted"}
        )

        # Get audit logs
        logs = logger.get_audit_logs(limit=50)
    """

    def __init__(
        self,
        audit_db_path: Optional[Path] = None,
        governance_db_path: Optional[Path] = None,
    ):
        """
        Initialize the governance logger.

        Args:
            audit_db_path: Path to audit.db
            governance_db_path: Path to governance.db
        """
        self._audit_db_path = audit_db_path or DEFAULT_AUDIT_DB_PATH
        self._governance_db_path = governance_db_path or DEFAULT_GOVERNANCE_DB_PATH

        # Thread safety
        self._lock = threading.RLock()

        # Ensure directories exist
        self._audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._governance_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize databases
        self._init_audit_db()
        self._init_governance_db()

        # Statistics tracking
        self._stats = {
            "total_logs": 0,
            "compliance_checks": 0,
            "quarantine_logs": 0,
            "pii_detections": 0,
            "policy_violations": 0,
            "errors": 0,
        }

        # Register shutdown handler
        atexit.register(self._atexit_handler)

        logger.debug(
            f"GovernanceLogger initialized, audit_db: {self._audit_db_path}, "
            f"governance_db: {self._governance_db_path}"
        )

    def _atexit_handler(self) -> None:
        """Handle graceful shutdown on process exit."""
        try:
            # Perform any cleanup if needed
            pass
        except Exception:
            pass

    @contextmanager
    def _get_audit_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a connection to the audit database.

        Uses WAL mode for concurrent access.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(
            str(self._audit_db_path),
            timeout=SQLITE_TIMEOUT,
            isolation_level=None,  # Autocommit mode
        )
        try:
            # Apply pragmas
            for pragma in SQLITE_PRAGMAS:
                conn.execute(pragma)
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _get_governance_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a connection to the governance database.

        Uses WAL mode for concurrent access.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(
            str(self._governance_db_path), timeout=SQLITE_TIMEOUT, isolation_level=None
        )
        try:
            for pragma in SQLITE_PRAGMAS:
                conn.execute(pragma)
            yield conn
        finally:
            conn.close()

    def _init_audit_db(self) -> None:
        """Initialize the audit database schema."""
        with self._get_audit_connection() as conn:
            cursor = conn.cursor()

            # Main audit log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    action_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    query_id TEXT,
                    session_id TEXT,
                    details TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices for common queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_log(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_action_type 
                ON audit_log(action_type)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_query_id 
                ON audit_log(query_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_session_id 
                ON audit_log(session_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_severity 
                ON audit_log(severity)
            """
            )

    def _init_governance_db(self) -> None:
        """Initialize the governance database schema."""
        with self._get_governance_connection() as conn:
            cursor = conn.cursor()

            # Compliance checks table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    check_type TEXT NOT NULL,
                    query_id TEXT,
                    session_id TEXT,
                    passed INTEGER NOT NULL,
                    details TEXT NOT NULL,
                    remediation TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Quarantine log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quarantine_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    action_type TEXT NOT NULL,
                    query_id TEXT,
                    session_id TEXT,
                    risk_level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    original_request TEXT,
                    blocked INTEGER DEFAULT 1,
                    reviewed INTEGER DEFAULT 0,
                    reviewer TEXT,
                    review_timestamp REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Policy violations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    policy_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    query_id TEXT,
                    details TEXT NOT NULL,
                    auto_resolved INTEGER DEFAULT 0,
                    resolution TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_compliance_timestamp 
                ON compliance_checks(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_compliance_check_type 
                ON compliance_checks(check_type)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quarantine_timestamp 
                ON quarantine_log(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quarantine_risk_level 
                ON quarantine_log(risk_level)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
                ON policy_violations(timestamp)
            """
            )

    def log_action(
        self,
        action_type: ActionType,
        details: Dict[str, Any],
        actor: str = "vulcan_system",
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
    ) -> int:
        """
        Log an action to the audit database.

        Args:
            action_type: Type of action being logged
            details: Action details as dictionary
            actor: Who/what performed the action
            query_id: Associated query ID
            session_id: Associated session ID
            severity: Severity level of the action

        Returns:
            ID of the inserted log entry
        """
        timestamp = time.time()
        entry_id = -1

        with self._lock:
            try:
                with self._get_audit_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO audit_log 
                        (timestamp, action_type, severity, actor, query_id, session_id, details)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            timestamp,
                            action_type.value,
                            severity.value,
                            actor,
                            query_id,
                            session_id,
                            json.dumps(details),
                        ),
                    )
                    entry_id = cursor.lastrowid

                self._stats["total_logs"] += 1

                logger.debug(
                    f"[GovernanceLogger] Logged {action_type.value} (id={entry_id}): "
                    f"query_id={query_id}"
                )

            except Exception as e:
                logger.error(
                    f"[GovernanceLogger] Failed to log action: {e}", exc_info=True
                )
                self._stats["errors"] += 1

        return entry_id

    def log_compliance_check(
        self,
        check_type: str,
        passed: bool,
        details: Dict[str, Any],
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> int:
        """
        Log a compliance check result.

        Args:
            check_type: Type of compliance check (e.g., "pii_check", "sensitive_data")
            passed: Whether the check passed
            details: Check details
            query_id: Associated query ID
            session_id: Associated session ID
            remediation: Suggested remediation if check failed

        Returns:
            ID of the inserted check entry
        """
        timestamp = time.time()
        entry_id = -1

        with self._lock:
            try:
                with self._get_governance_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO compliance_checks 
                        (timestamp, check_type, query_id, session_id, passed, details, remediation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            timestamp,
                            check_type,
                            query_id,
                            session_id,
                            1 if passed else 0,
                            json.dumps(details),
                            remediation,
                        ),
                    )
                    entry_id = cursor.lastrowid

                self._stats["compliance_checks"] += 1
                if check_type == "pii_check" and not passed:
                    self._stats["pii_detections"] += 1

                # Also log to audit
                self.log_action(
                    ActionType.GOVERNANCE_CHECK,
                    {"check_type": check_type, "passed": passed, **details},
                    query_id=query_id,
                    session_id=session_id,
                    severity=SeverityLevel.INFO if passed else SeverityLevel.WARNING,
                )

            except Exception as e:
                logger.error(
                    f"[GovernanceLogger] Failed to log compliance check: {e}",
                    exc_info=True,
                )
                self._stats["errors"] += 1

        return entry_id

    def log_quarantine(
        self,
        action_type: str,
        reason: str,
        risk_level: str,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        original_request: Optional[str] = None,
        blocked: bool = True,
    ) -> int:
        """
        Log a quarantined/blocked action.

        Args:
            action_type: Type of action that was quarantined
            reason: Reason for quarantine
            risk_level: Risk level (low, medium, high, critical)
            query_id: Associated query ID
            session_id: Associated session ID
            original_request: The original request that was quarantined
            blocked: Whether the action was blocked

        Returns:
            ID of the inserted quarantine entry
        """
        timestamp = time.time()
        entry_id = -1

        with self._lock:
            try:
                with self._get_governance_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO quarantine_log 
                        (timestamp, action_type, query_id, session_id, risk_level, reason, original_request, blocked)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            timestamp,
                            action_type,
                            query_id,
                            session_id,
                            risk_level,
                            reason,
                            (
                                original_request[:MAX_REQUEST_LENGTH]
                                if original_request
                                else None
                            ),  # Truncate
                            1 if blocked else 0,
                        ),
                    )
                    entry_id = cursor.lastrowid

                self._stats["quarantine_logs"] += 1

                # Determine severity from risk level
                severity_map = {
                    "low": SeverityLevel.INFO,
                    "medium": SeverityLevel.WARNING,
                    "high": SeverityLevel.ERROR,
                    "critical": SeverityLevel.CRITICAL,
                }
                severity = severity_map.get(risk_level, SeverityLevel.WARNING)

                # Log to audit
                self.log_action(
                    ActionType.QUARANTINE_ACTION,
                    {
                        "action_type": action_type,
                        "reason": reason,
                        "risk_level": risk_level,
                        "blocked": blocked,
                    },
                    query_id=query_id,
                    session_id=session_id,
                    severity=severity,
                )

                logger.warning(
                    f"[GovernanceLogger] Quarantined action (id={entry_id}): "
                    f"{action_type} - {reason}"
                )

            except Exception as e:
                logger.error(
                    f"[GovernanceLogger] Failed to log quarantine: {e}", exc_info=True
                )
                self._stats["errors"] += 1

        return entry_id

    def log_policy_violation(
        self,
        policy_id: str,
        violation_type: str,
        details: Dict[str, Any],
        query_id: Optional[str] = None,
        auto_resolved: bool = False,
        resolution: Optional[str] = None,
    ) -> int:
        """
        Log a policy violation.

        Args:
            policy_id: ID of the violated policy
            violation_type: Type of violation
            details: Violation details
            query_id: Associated query ID
            auto_resolved: Whether the violation was automatically resolved
            resolution: Resolution description if resolved

        Returns:
            ID of the inserted violation entry
        """
        timestamp = time.time()
        entry_id = -1

        with self._lock:
            try:
                with self._get_governance_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO policy_violations 
                        (timestamp, policy_id, violation_type, query_id, details, auto_resolved, resolution)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            timestamp,
                            policy_id,
                            violation_type,
                            query_id,
                            json.dumps(details),
                            1 if auto_resolved else 0,
                            resolution,
                        ),
                    )
                    entry_id = cursor.lastrowid

                self._stats["policy_violations"] += 1

                # Log to audit
                self.log_action(
                    ActionType.COMPLIANCE_VIOLATION,
                    {
                        "policy_id": policy_id,
                        "violation_type": violation_type,
                        "auto_resolved": auto_resolved,
                        **details,
                    },
                    query_id=query_id,
                    severity=(
                        SeverityLevel.WARNING if auto_resolved else SeverityLevel.ERROR
                    ),
                )

            except Exception as e:
                logger.error(
                    f"[GovernanceLogger] Failed to log policy violation: {e}",
                    exc_info=True,
                )
                self._stats["errors"] += 1

        return entry_id

    def get_audit_logs(
        self,
        limit: int = DEFAULT_LOG_QUERY_LIMIT,
        offset: int = 0,
        action_type: Optional[ActionType] = None,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        severity: Optional[SeverityLevel] = None,
        since_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with optional filtering.

        Args:
            limit: Maximum number of logs to return (max 1000)
            offset: Offset for pagination
            action_type: Filter by action type
            query_id: Filter by query ID
            session_id: Filter by session ID
            severity: Filter by severity level
            since_timestamp: Only return logs after this timestamp

        Returns:
            List of audit log entries as dictionaries
        """
        # Enforce limits
        limit = min(limit, MAX_LOG_QUERY_LIMIT)

        with self._get_audit_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM audit_log WHERE 1=1"
            params: List[Any] = []

            if action_type:
                query += " AND action_type = ?"
                params.append(action_type.value)

            if query_id:
                query += " AND query_id = ?"
                params.append(query_id)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if severity:
                query += " AND severity = ?"
                params.append(severity.value)

            if since_timestamp:
                query += " AND timestamp >= ?"
                params.append(since_timestamp)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in rows:
                record = dict(zip(columns, row))
                # Parse JSON details
                if record.get("details"):
                    try:
                        record["details"] = json.loads(record["details"])
                    except json.JSONDecodeError:
                        pass
                results.append(record)

            return results

    def get_compliance_checks(
        self,
        limit: int = DEFAULT_LOG_QUERY_LIMIT,
        offset: int = 0,
        check_type: Optional[str] = None,
        passed: Optional[bool] = None,
        since_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve compliance check records.

        Args:
            limit: Maximum records to return
            offset: Offset for pagination
            check_type: Filter by check type
            passed: Filter by pass/fail status
            since_timestamp: Only return checks after this timestamp

        Returns:
            List of compliance check records
        """
        limit = min(limit, MAX_LOG_QUERY_LIMIT)

        with self._get_governance_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM compliance_checks WHERE 1=1"
            params: List[Any] = []

            if check_type:
                query += " AND check_type = ?"
                params.append(check_type)

            if passed is not None:
                query += " AND passed = ?"
                params.append(1 if passed else 0)

            if since_timestamp:
                query += " AND timestamp >= ?"
                params.append(since_timestamp)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in rows:
                record = dict(zip(columns, row))
                record["passed"] = bool(record.get("passed"))
                if record.get("details"):
                    try:
                        record["details"] = json.loads(record["details"])
                    except json.JSONDecodeError:
                        pass
                results.append(record)

            return results

    def get_quarantine_logs(
        self,
        limit: int = DEFAULT_LOG_QUERY_LIMIT,
        offset: int = 0,
        risk_level: Optional[str] = None,
        reviewed: Optional[bool] = None,
        since_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve quarantine log records.

        Args:
            limit: Maximum records to return
            offset: Offset for pagination
            risk_level: Filter by risk level
            reviewed: Filter by review status
            since_timestamp: Only return logs after this timestamp

        Returns:
            List of quarantine log records
        """
        limit = min(limit, MAX_LOG_QUERY_LIMIT)

        with self._get_governance_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM quarantine_log WHERE 1=1"
            params: List[Any] = []

            if risk_level:
                query += " AND risk_level = ?"
                params.append(risk_level)

            if reviewed is not None:
                query += " AND reviewed = ?"
                params.append(1 if reviewed else 0)

            if since_timestamp:
                query += " AND timestamp >= ?"
                params.append(since_timestamp)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in rows:
                record = dict(zip(columns, row))
                record["blocked"] = bool(record.get("blocked"))
                record["reviewed"] = bool(record.get("reviewed"))
                results.append(record)

            return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive governance logging statistics.

        Returns:
            Dictionary with log counts and database statistics
        """
        with self._lock:
            stats = dict(self._stats)

        # Add database counts
        try:
            with self._get_audit_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM audit_log")
                stats["audit_log_count"] = cursor.fetchone()[0]
        except Exception:
            stats["audit_log_count"] = -1

        try:
            with self._get_governance_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM compliance_checks")
                stats["compliance_check_count"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM quarantine_log")
                stats["quarantine_count"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM policy_violations")
                stats["policy_violation_count"] = cursor.fetchone()[0]
        except Exception:
            stats["compliance_check_count"] = -1
            stats["quarantine_count"] = -1
            stats["policy_violation_count"] = -1

        return stats


# ============================================================
# SINGLETON PATTERN
# ============================================================

_global_logger: Optional[GovernanceLogger] = None
_logger_lock = threading.Lock()


def get_governance_logger() -> GovernanceLogger:
    """
    Get or create the global governance logger (thread-safe singleton).

    Returns:
        GovernanceLogger instance
    """
    global _global_logger

    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = GovernanceLogger()
                logger.debug("Global GovernanceLogger instance created")

    return _global_logger


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def log_to_governance(
    action_type: str,
    details: Dict[str, Any],
    actor: str = "vulcan_system",
    query_id: Optional[str] = None,
    session_id: Optional[str] = None,
    severity: str = "info",
) -> int:
    """
    Log to audit.db tables.

    Convenience function using global logger.

    Args:
        action_type: Type of action (string or ActionType enum)
        details: Action details
        actor: Actor identifier
        query_id: Query ID
        session_id: Session ID
        severity: Severity level string

    Returns:
        ID of the log entry
    """
    gov_logger = get_governance_logger()

    # Convert string to ActionType if needed
    if isinstance(action_type, str):
        try:
            action_type_enum = ActionType(action_type)
        except ValueError:
            action_type_enum = ActionType.QUERY_PROCESSED
    else:
        action_type_enum = action_type

    # Convert string to SeverityLevel
    try:
        severity_enum = SeverityLevel(severity)
    except ValueError:
        severity_enum = SeverityLevel.INFO

    return gov_logger.log_action(
        action_type_enum,
        details,
        actor=actor,
        query_id=query_id,
        session_id=session_id,
        severity=severity_enum,
    )


async def log_to_governance_async(
    action_type: str,
    details: Dict[str, Any],
    actor: str = "vulcan_system",
    query_id: Optional[str] = None,
    session_id: Optional[str] = None,
    severity: str = "info",
) -> int:
    """
    Async version of log_to_governance that doesn't block the event loop.

    PERFORMANCE FIX: This function uses asyncio.to_thread to run the
    synchronous SQLite operation in a thread pool, preventing the
    governance logging from blocking the async request handler.

    Args:
        action_type: Type of action (string or ActionType enum)
        details: Action details
        actor: Actor identifier
        query_id: Query ID
        session_id: Session ID
        severity: Severity level string

    Returns:
        ID of the log entry
    """
    return await asyncio.to_thread(
        log_to_governance,
        action_type,
        details,
        actor,
        query_id,
        session_id,
        severity,
    )


def log_to_governance_fire_and_forget(
    action_type: str,
    details: Dict[str, Any],
    actor: str = "vulcan_system",
    query_id: Optional[str] = None,
    session_id: Optional[str] = None,
    severity: str = "info",
) -> None:
    """
    Fire-and-forget version of log_to_governance for non-critical logging.

    PERFORMANCE FIX: This function submits the logging operation to a
    thread pool and returns immediately without waiting for completion.
    Use this when you don't need to wait for the log entry ID and want
    maximum performance.

    Args:
        action_type: Type of action (string or ActionType enum)
        details: Action details
        actor: Actor identifier
        query_id: Query ID
        session_id: Session ID
        severity: Severity level string
    """
    executor = _get_governance_executor()
    executor.submit(
        log_to_governance,
        action_type,
        details,
        actor,
        query_id,
        session_id,
        severity,
    )
