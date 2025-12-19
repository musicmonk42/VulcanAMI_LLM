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

import atexit
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

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
        governance_db_path: Optional[Path] = None
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
            isolation_level=None  # Autocommit mode
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
            str(self._governance_db_path),
            timeout=SQLITE_TIMEOUT,
            isolation_level=None
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
            cursor.execute("""
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
            """)
            
            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_log(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_action_type 
                ON audit_log(action_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_query_id 
                ON audit_log(query_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session_id 
                ON audit_log(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_severity 
                ON audit_log(severity)
            """)
    
    def _init_governance_db(self) -> None:
        """Initialize the governance database schema."""
        with self._get_governance_connection() as conn:
            cursor = conn.cursor()
            
            # Compliance checks table
            cursor.execute("""
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
            """)
            
            # Quarantine log table
            cursor.execute("""
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
            """)
            
            # Policy violations table
            cursor.execute("""
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
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_compliance_timestamp 
                ON compliance_checks(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_compliance_check_type 
                ON compliance_checks(check_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_timestamp 
                ON quarantine_log(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_risk_level 
                ON quarantine_log(risk_level)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
                ON policy_violations(timestamp)
            """)
    
    def log_action(
        self,
        action_type: ActionType,
        details: Dict[str, Any],
        actor: str = "vulcan_system",
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        severity: SeverityLevel = SeverityLevel.INFO
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
                    cursor.execute("""
                        INSERT INTO audit_log 
                        (timestamp, action_type, severity, actor, query_id, session_id, details)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        action_type.value,
                        severity.value,
                        actor,
                        query_id,
                        session_id,
                        json.dumps(details)
                    ))
                    entry_id = cursor.lastrowid
                
                self._stats["total_logs"] += 1
                
                logger.debug(
                    f"[GovernanceLogger] Logged {action_type.value} (id={entry_id}): "
                    f"query_id={query_id}"
                )
                
            except Exception as e:
                logger.error(f"[GovernanceLogger] Failed to log action: {e}", exc_info=True)
                self._stats["errors"] += 1
        
        return entry_id
    
    def log_compliance_check(
        self,
        check_type: str,
        passed: bool,
        details: Dict[str, Any],
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        remediation: Optional[str] = None
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
                    cursor.execute("""
                        INSERT INTO compliance_checks 
                        (timestamp, check_type, query_id, session_id, passed, details, remediation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        check_type,
                        query_id,
                        session_id,
                        1 if passed else 0,
                        json.dumps(details),
                        remediation
                    ))
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
                    severity=SeverityLevel.INFO if passed else SeverityLevel.WARNING
                )
                
            except Exception as e:
                logger.error(f"[GovernanceLogger] Failed to log compliance check: {e}", exc_info=True)
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
        blocked: bool = True
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
                    cursor.execute("""
                        INSERT INTO quarantine_log 
                        (timestamp, action_type, query_id, session_id, risk_level, reason, original_request, blocked)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        action_type,
                        query_id,
                        session_id,
                        risk_level,
                        reason,
                        original_request[:2000] if original_request else None,  # Truncate
                        1 if blocked else 0
                    ))
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
                        "blocked": blocked
                    },
                    query_id=query_id,
                    session_id=session_id,
                    severity=severity
                )
                
                logger.warning(
                    f"[GovernanceLogger] Quarantined action (id={entry_id}): "
                    f"{action_type} - {reason}"
                )
                
            except Exception as e:
                logger.error(f"[GovernanceLogger] Failed to log quarantine: {e}", exc_info=True)
                self._stats["errors"] += 1
        
        return entry_id
    
    def log_policy_violation(
        self,
        policy_id: str,
        violation_type: str,
        details: Dict[str, Any],
        query_id: Optional[str] = None,
        auto_resolved: bool = False,
        resolution: Optional[str] = None
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
                    cursor.execute("""
                        INSERT INTO policy_violations 
                        (timestamp, policy_id, violation_type, query_id, details, auto_resolved, resolution)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        policy_id,
                        violation_type,
                        query_id,
                        json.dumps(details),
                        1 if auto_resolved else 0,
                        resolution
                    ))
                    entry_id = cursor.lastrowid
                
                self._stats["policy_violations"] += 1
                
                # Log to audit
                self.log_action(
                    ActionType.COMPLIANCE_VIOLATION,
                    {
                        "policy_id": policy_id,
                        "violation_type": violation_type,
                        "auto_resolved": auto_resolved,
                        **details
                    },
                    query_id=query_id,
                    severity=SeverityLevel.WARNING if auto_resolved else SeverityLevel.ERROR
                )
                
            except Exception as e:
                logger.error(f"[GovernanceLogger] Failed to log policy violation: {e}", exc_info=True)
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
        since_timestamp: Optional[float] = None
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
        since_timestamp: Optional[float] = None
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
        since_timestamp: Optional[float] = None
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
    severity: str = "info"
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
        severity=severity_enum
    )
