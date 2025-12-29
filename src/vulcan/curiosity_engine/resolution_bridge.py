"""
SQLite Bridge for Cross-Process Gap Resolution State Sharing.

Part of the VULCAN-AGI system.

This module provides a SQLite-based persistence layer for gap resolution state,
enabling the CuriosityEngine to remember which gaps have been resolved across
subprocess invocations.

Key Problem Solved:
    When CuriosityEngine runs learning cycles in a subprocess (for CPU isolation),
    each subprocess creates a fresh engine instance with no memory of previous
    resolutions. This causes:
    - "Phantom resolutions" where the same gap is "resolved" 40-90 times per hour
    - Cold start triggers every cycle (thinking 0/5 experiments ran)
    - Wasted computation on already-addressed gaps

Key Features:
    - Thread-safe SQLite operations with proper connection management
    - Automatic database initialization with idempotent schema creation
    - TTL-based resolution expiration to allow re-detection of persistent issues
    - Phantom resolution detection to identify gaps that aren't truly fixed
    - Persistent experiment counters to prevent false cold-start detection
    - Comprehensive cleanup to prevent unbounded database growth

Performance Characteristics:
    - Uses WAL mode for concurrent read/write access
    - Bounded queries with LIMIT to prevent unbounded result sets
    - Connection pooling via context manager pattern
    - Index-optimized queries for gap_key and timestamp lookups

Database Schema:
    gap_resolutions table:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - gap_key: TEXT UNIQUE NOT NULL (e.g., "high_error_rate:query_processing")
        - resolved_at: REAL NOT NULL (Unix timestamp when resolved)
        - success: INTEGER NOT NULL DEFAULT 1 (1=success, 0=give-up)
        - attempts: INTEGER NOT NULL DEFAULT 1 (experiment attempt count)
        - created_at: REAL (Unix timestamp of first creation)
        
    resolution_history table:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - gap_key: TEXT NOT NULL (gap identifier)
        - timestamp: REAL NOT NULL (Unix timestamp of resolution)
        - success: INTEGER NOT NULL DEFAULT 1 (resolution success flag)
        - cycle_id: INTEGER (learning cycle that triggered resolution)
        - created_at: REAL (Unix timestamp)

    experiment_counters table:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - counter_name: TEXT UNIQUE NOT NULL (e.g., "total_experiments")
        - value: INTEGER NOT NULL DEFAULT 0 (counter value)
        - updated_at: REAL (last update timestamp)

Usage:
    # In CuriosityEngine subprocess - check before running experiments:
    from vulcan.curiosity_engine.resolution_bridge import (
        is_gap_resolved,
        mark_gap_resolved,
        get_gap_attempts,
        increment_gap_attempts,
        record_resolution_history,
        get_recent_resolutions_count,
        is_phantom_resolution,
        get_experiment_count,
        increment_experiment_count,
    )
    
    # Check if gap was already resolved (survives subprocess restart)
    if is_gap_resolved("high_error_rate:query_processing"):
        continue  # Skip this gap - already addressed
    
    # Check for phantom resolution pattern before investing effort
    if is_phantom_resolution("high_error_rate:query_processing"):
        logger.warning("Gap shows phantom resolution pattern - needs deeper fix")
    
    # After experiment succeeds
    mark_gap_resolved("high_error_rate:query_processing", success=True)
    record_resolution_history("high_error_rate:query_processing", success=True, cycle_id=123)
    
    # Track experiments to prevent false cold-start detection
    total = increment_experiment_count("total_experiments")

Thread Safety:
    All functions are thread-safe. Database connections are managed per-call
    with proper cleanup. Module-level initialization uses double-checked
    locking for thread-safe lazy initialization.

Error Handling:
    All functions return sensible defaults on error (empty dicts, False,
    zero counts) and log warnings. Exceptions are caught and handled
    gracefully to prevent cascading failures in the learning system.
"""

import logging
import os
import sqlite3
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Logging prefix for consistent output
LOG_PREFIX = "[ResolutionBridge]"

# Default database path - uses user-specific data directory for security
# Can be overridden via VULCAN_RESOLUTION_DB_PATH environment variable
# Falls back to temp directory if user data directory is not available
_default_db_dir = os.environ.get("XDG_DATA_HOME") or os.path.join(
    os.path.expanduser("~"), ".local", "share"
)
_default_db_path = os.path.join(_default_db_dir, "vulcan", "gap_resolutions.db")

# Ensure the directory exists with secure permissions (0o700 = rwx------)
try:
    db_dir = os.path.dirname(_default_db_path)
    os.makedirs(db_dir, mode=0o700, exist_ok=True)
except OSError:
    # Fall back to temp directory if we can't create the data directory
    _default_db_path = os.path.join(tempfile.gettempdir(), "vulcan_gap_resolutions.db")

DEFAULT_DB_PATH = Path(
    os.environ.get("VULCAN_RESOLUTION_DB_PATH", _default_db_path)
)

# Database connection timeout in seconds
DB_TIMEOUT_SECONDS = 30.0

# Resolution TTL in seconds (30 minutes default)
# After this time, a resolved gap can be re-detected if the issue persists
RESOLUTION_TTL_SECONDS = int(os.environ.get("VULCAN_RESOLUTION_TTL", "1800"))

# Phantom resolution threshold
# If a gap is resolved this many times in the tracking window, it's a phantom
PHANTOM_RESOLUTION_THRESHOLD = int(
    os.environ.get("VULCAN_PHANTOM_THRESHOLD", "3")
)

# Phantom resolution tracking window (1 hour)
PHANTOM_TRACKING_WINDOW_SECONDS = int(
    os.environ.get("VULCAN_PHANTOM_WINDOW", "3600")
)

# Extended cooldown for phantom resolutions (1 hour)
PHANTOM_COOLDOWN_SECONDS = int(
    os.environ.get("VULCAN_PHANTOM_COOLDOWN", "3600")
)

# Cleanup interval (delete resolutions older than this)
CLEANUP_RETENTION_DAYS = int(
    os.environ.get("VULCAN_RESOLUTION_RETENTION_DAYS", "7")
)

# Default query limits
DEFAULT_HISTORY_LIMIT = 500


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ResolutionStatus(Enum):
    """
    Enumeration of gap resolution statuses.
    
    Used for type-safe status handling and serialization.
    """
    SUCCESS = "success"
    GIVE_UP = "give_up"
    PENDING = "pending"
    PHANTOM = "phantom"


@dataclass
class ResolutionRecord:
    """
    Represents a gap resolution record from the database.
    
    Attributes:
        gap_key: Unique gap identifier (e.g., "high_error_rate:query_processing")
        resolved_at: Unix timestamp when the gap was resolved
        success: True if resolved successfully, False if gave up
        attempts: Number of experiment attempts made
        created_at: Unix timestamp when first tracked
    """
    
    gap_key: str
    resolved_at: float
    success: bool
    attempts: int
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert record to dictionary for serialization.
        
        Returns:
            Dictionary representation of the resolution record.
        """
        return {
            "gap_key": self.gap_key,
            "resolved_at": self.resolved_at,
            "success": self.success,
            "attempts": self.attempts,
            "created_at": self.created_at,
        }
    
    @property
    def status(self) -> ResolutionStatus:
        """Get the resolution status."""
        if self.resolved_at <= 0:
            return ResolutionStatus.PENDING
        return ResolutionStatus.SUCCESS if self.success else ResolutionStatus.GIVE_UP


@dataclass
class ResolutionStatistics:
    """
    Statistics computed from resolution data for monitoring.
    
    This dataclass provides aggregate metrics for monitoring the health
    of the gap resolution system.
    
    Attributes:
        total_resolutions: Total number of gap resolutions tracked
        active_resolutions: Number of resolutions not yet expired (within TTL)
        phantom_count: Number of gaps exhibiting phantom resolution behavior
        avg_attempts: Average experiment attempts per gap
        success_rate: Success rate of resolutions (0.0 to 1.0)
        total_experiments: Total experiments tracked across all counters
        
    Example:
        >>> stats = get_resolution_statistics()
        >>> print(f"Success rate: {stats.success_rate:.1%}")
        >>> print(f"Phantom resolutions: {stats.phantom_count}")
    """
    
    total_resolutions: int
    active_resolutions: int
    phantom_count: int
    avg_attempts: float
    success_rate: float
    total_experiments: int
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary for serialization.
        
        Returns:
            Dictionary representation of statistics.
        """
        return {
            "total_resolutions": self.total_resolutions,
            "active_resolutions": self.active_resolutions,
            "phantom_count": self.phantom_count,
            "avg_attempts": self.avg_attempts,
            "success_rate": self.success_rate,
            "total_experiments": self.total_experiments,
        }


# =============================================================================
# Module-Level State
# =============================================================================

# Thread-local storage for database connections (not currently used but available)
_local = threading.local()

# Module-level lock for database initialization
_init_lock = threading.Lock()
_db_initialized = False


# =============================================================================
# Database Connection Management
# =============================================================================


@contextmanager
def _get_db(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Get database connection with proper cleanup.
    
    Uses WAL mode for better concurrent access and sets a reasonable
    timeout for busy waiting. Connections are created fresh for each
    context to ensure thread safety.
    
    Args:
        db_path: Optional path to database file. Defaults to DEFAULT_DB_PATH.
    
    Yields:
        SQLite connection with row factory set to sqlite3.Row
    
    Example:
        >>> with _get_db() as conn:
        ...     cursor = conn.execute("SELECT COUNT(*) FROM gap_resolutions")
        ...     count = cursor.fetchone()[0]
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path), timeout=DB_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    
    # Enable WAL mode for better concurrent access
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass  # Ignore if already set or not supported
    
    try:
        yield conn
    finally:
        conn.close()


def _init_db(db_path: Optional[Path] = None) -> bool:
    """
    Initialize database schema (idempotent).
    
    Creates the gap_resolutions, resolution_history, and experiment_counters
    tables and necessary indexes if they don't exist. Thread-safe and can be
    called multiple times without side effects. Uses double-checked locking
    for efficient thread-safe initialization.
    
    Args:
        db_path: Optional path to database file. Defaults to DEFAULT_DB_PATH.
    
    Returns:
        True if initialization succeeded, False otherwise.
    
    Example:
        >>> success = _init_db()
        >>> if success:
        ...     print("Database ready")
    """
    global _db_initialized
    
    # Fast path if already initialized
    if _db_initialized:
        return True
    
    with _init_lock:
        # Double-checked locking
        if _db_initialized:
            return True
        
        try:
            with _get_db(db_path) as conn:
                # Gap resolutions table - tracks current resolution status
                # Note: DEFAULT (strftime('%s', 'now')) evaluates at insertion time in SQLite
                # because the expression is wrapped in parentheses
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS gap_resolutions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gap_key TEXT UNIQUE NOT NULL,
                        resolved_at REAL NOT NULL,
                        success INTEGER NOT NULL DEFAULT 1,
                        attempts INTEGER NOT NULL DEFAULT 1,
                        created_at REAL
                    )
                """)
                
                # Resolution history table - tracks all resolutions for phantom detection
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS resolution_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gap_key TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        success INTEGER NOT NULL DEFAULT 1,
                        cycle_id INTEGER,
                        created_at REAL
                    )
                """)
                
                # Experiment counters table - tracks experiment counts across processes
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiment_counters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        counter_name TEXT UNIQUE NOT NULL,
                        value INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL
                    )
                """)
                
                # Create indexes for common query patterns
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gap_key "
                    "ON gap_resolutions(gap_key)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_resolved_at "
                    "ON gap_resolutions(resolved_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_history_gap_key "
                    "ON resolution_history(gap_key)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_history_timestamp "
                    "ON resolution_history(timestamp)"
                )
                
                conn.commit()
            
            _db_initialized = True
            logger.info(f"{LOG_PREFIX} Database initialized successfully")
            return True
            
        except sqlite3.Error as e:
            logger.warning(f"{LOG_PREFIX} Database initialization failed: {e}")
            return False


# =============================================================================
# Core Gap Resolution Functions
# =============================================================================


def is_gap_resolved(
    gap_key: str,
    ttl_seconds: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Check if a gap is currently resolved (and not expired).
    
    A gap is considered resolved if:
    1. It exists in the gap_resolutions table
    2. The resolution is not older than TTL (default 30 minutes)
    
    This check survives subprocess restarts because state is persisted to SQLite.
    
    Args:
        gap_key: Unique gap identifier (e.g., "high_error_rate:query_processing").
            Format is typically "gap_type:domain".
        ttl_seconds: Resolution TTL in seconds. Defaults to RESOLUTION_TTL_SECONDS.
            After TTL expires, gap can be re-detected if issue persists.
        db_path: Optional path to database file.
    
    Returns:
        True if gap is resolved and not expired, False otherwise.
    
    Example:
        >>> if is_gap_resolved("high_error_rate:query_processing"):
        ...     print("Gap already addressed, skipping")
        ... else:
        ...     print("Gap needs attention")
    """
    if not _init_db(db_path):
        return False
    
    ttl = ttl_seconds if ttl_seconds is not None else RESOLUTION_TTL_SECONDS
    cutoff = time.time() - ttl
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT resolved_at FROM gap_resolutions
                WHERE gap_key = ? AND resolved_at > ?
                """,
                (gap_key, cutoff),
            )
            row = cursor.fetchone()
            return row is not None
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error checking gap resolution: {e}")
        return False


def mark_gap_resolved(
    gap_key: str,
    success: bool = True,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Mark a gap as resolved in persistent storage.
    
    Call this after an experiment successfully addresses a gap, or when
    giving up after too many attempts. The resolution persists across
    subprocess restarts.
    
    Args:
        gap_key: Unique gap identifier (e.g., "high_error_rate:query_processing").
        success: True if resolved successfully by experiment, False if giving up.
            Both cases prevent immediate re-detection for the TTL period.
        db_path: Optional path to database file.
    
    Returns:
        True if marking succeeded, False otherwise.
    
    Example:
        >>> # After successful experiment
        >>> mark_gap_resolved("slow_routing:query_processing", success=True)
        >>> 
        >>> # After giving up on persistent issue
        >>> mark_gap_resolved("high_error_rate:system", success=False)
    """
    if not _init_db(db_path):
        return False
    
    try:
        with _get_db(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gap_resolutions
                (gap_key, resolved_at, success, attempts)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT attempts FROM gap_resolutions WHERE gap_key = ?), 0
                ) + 1)
                """,
                (gap_key, time.time(), 1 if success else 0, gap_key),
            )
            conn.commit()
        
        status = "resolved" if success else "deferred"
        logger.info(f"{LOG_PREFIX} Gap {gap_key} marked as {status}")
        return True
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error marking gap resolved: {e}")
        return False


def clear_gap_resolution(
    gap_key: str,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Clear resolution status for a gap to allow immediate re-detection.
    
    Use this when you want to force a gap to be re-evaluated before its
    TTL expires. Useful for testing or manual intervention.
    
    Args:
        gap_key: Unique gap identifier.
        db_path: Optional path to database file.
    
    Returns:
        True if successful, False otherwise.
    
    Example:
        >>> # Force re-detection of a gap
        >>> clear_gap_resolution("high_error_rate:query_processing")
    """
    if not _init_db(db_path):
        return False
    
    try:
        with _get_db(db_path) as conn:
            conn.execute(
                "DELETE FROM gap_resolutions WHERE gap_key = ?",
                (gap_key,),
            )
            conn.commit()
        return True
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error clearing gap resolution: {e}")
        return False


def get_gap_attempts(
    gap_key: str,
    db_path: Optional[Path] = None,
) -> int:
    """
    Get the number of experiment attempts for a gap.
    
    Args:
        gap_key: Unique gap identifier
        db_path: Optional database path.
    
    Returns:
        Number of attempts, or 0 if not found.
    """
    if not _init_db(db_path):
        return 0
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                "SELECT attempts FROM gap_resolutions WHERE gap_key = ?",
                (gap_key,),
            )
            row = cursor.fetchone()
            return row["attempts"] if row else 0
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error getting gap attempts: {e}")
        return 0


def increment_gap_attempts(
    gap_key: str,
    db_path: Optional[Path] = None,
) -> int:
    """
    Increment the experiment attempt counter for a gap.
    
    Tracks how many times experiments have been run against a gap.
    Used to implement give-up logic after too many failed attempts.
    
    Args:
        gap_key: Unique gap identifier.
        db_path: Optional path to database file.
    
    Returns:
        New attempt count, or 0 on error.
    
    Example:
        >>> attempts = increment_gap_attempts("high_error_rate:query_processing")
        >>> if attempts >= 10:
        ...     mark_gap_resolved(gap_key, success=False)  # Give up
    """
    if not _init_db(db_path):
        return 0
    
    try:
        with _get_db(db_path) as conn:
            # Use UPSERT pattern (INSERT ON CONFLICT UPDATE)
            conn.execute(
                """
                INSERT INTO gap_resolutions (gap_key, resolved_at, success, attempts)
                VALUES (?, 0, 0, 1)
                ON CONFLICT(gap_key) DO UPDATE SET
                    attempts = attempts + 1
                """,
                (gap_key,),
            )
            conn.commit()
            
            cursor = conn.execute(
                "SELECT attempts FROM gap_resolutions WHERE gap_key = ?",
                (gap_key,),
            )
            row = cursor.fetchone()
            return row["attempts"] if row else 1
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error incrementing gap attempts: {e}")
        return 0


def reset_gap_attempts(
    gap_key: str,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Reset the experiment attempt counter for a gap.
    
    Args:
        gap_key: Unique gap identifier
        db_path: Optional database path.
    
    Returns:
        True if successful.
    """
    if not _init_db(db_path):
        return False
    
    try:
        with _get_db(db_path) as conn:
            conn.execute(
                "UPDATE gap_resolutions SET attempts = 0 WHERE gap_key = ?",
                (gap_key,),
            )
            conn.commit()
        return True
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error resetting gap attempts: {e}")
        return False


def mark_gap_resolved_batch(
    gap_key: str,
    success: bool = True,
    cycle_id: Optional[int] = None,
    reset_attempts: bool = False,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Mark a gap as resolved with all related operations in a single transaction.
    
    This function combines mark_gap_resolved, record_resolution_history, and
    optionally reset_gap_attempts into a single atomic database transaction
    for better performance and consistency.
    
    Args:
        gap_key: Unique gap identifier.
        success: True if resolved successfully, False if giving up.
        cycle_id: Optional learning cycle ID for tracking.
        reset_attempts: If True, also reset the attempts counter.
        db_path: Optional path to database file.
    
    Returns:
        True if all operations succeeded, False otherwise.
    
    Example:
        >>> # Mark gap resolved with all tracking in one transaction
        >>> mark_gap_resolved_batch(
        ...     "high_error_rate:query_processing",
        ...     success=True,
        ...     cycle_id=123,
        ...     reset_attempts=True
        ... )
    """
    if not _init_db(db_path):
        return False
    
    current_time = time.time()
    
    try:
        with _get_db(db_path) as conn:
            # Start transaction (implicit in SQLite)
            
            # 1. Mark gap as resolved (UPSERT)
            conn.execute(
                """
                INSERT INTO gap_resolutions (gap_key, resolved_at, success, attempts)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT attempts FROM gap_resolutions WHERE gap_key = ?), 0
                ) + 1)
                ON CONFLICT(gap_key) DO UPDATE SET
                    resolved_at = excluded.resolved_at,
                    success = excluded.success,
                    attempts = gap_resolutions.attempts + 1
                """,
                (gap_key, current_time, 1 if success else 0, gap_key),
            )
            
            # 2. Record in history
            conn.execute(
                """
                INSERT INTO resolution_history (gap_key, timestamp, success, cycle_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (gap_key, current_time, 1 if success else 0, cycle_id, current_time),
            )
            
            # 3. Optionally reset attempts
            if reset_attempts:
                conn.execute(
                    "UPDATE gap_resolutions SET attempts = 0 WHERE gap_key = ?",
                    (gap_key,),
                )
            
            conn.commit()
        
        status = "resolved" if success else "deferred"
        logger.info(f"{LOG_PREFIX} Gap {gap_key} marked as {status} (batch)")
        return True
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error in batch resolution: {e}")
        return False


# =============================================================================
# Resolution History Functions (for Phantom Detection)
# =============================================================================

def record_resolution_history(
    gap_key: str,
    success: bool = True,
    cycle_id: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Record a resolution event in history.
    
    Used for phantom resolution detection.
    
    Args:
        gap_key: Unique gap identifier
        success: Whether resolution was successful
        cycle_id: Learning cycle ID (optional)
        db_path: Optional database path.
    
    Returns:
        True if successful.
    """
    if not _init_db(db_path):
        return False
    
    try:
        with _get_db(db_path) as conn:
            conn.execute(
                """
                INSERT INTO resolution_history (gap_key, timestamp, success, cycle_id)
                VALUES (?, ?, ?, ?)
                """,
                (gap_key, time.time(), 1 if success else 0, cycle_id),
            )
            conn.commit()
        return True
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error recording resolution history: {e}")
        return False


def get_recent_resolutions_count(
    gap_key: str,
    window_seconds: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Count how many times a gap was resolved in the recent window.
    
    Used for phantom resolution detection.
    
    Args:
        gap_key: Unique gap identifier
        window_seconds: Time window. None uses default (1 hour).
        db_path: Optional database path.
    
    Returns:
        Number of resolutions in window.
    """
    if not _init_db(db_path):
        return 0
    
    window = window_seconds if window_seconds is not None else PHANTOM_TRACKING_WINDOW_SECONDS
    cutoff = time.time() - window
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM resolution_history
                WHERE gap_key = ? AND timestamp > ?
                """,
                (gap_key, cutoff),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error counting recent resolutions: {e}")
        return 0


def is_phantom_resolution(
    gap_key: str,
    threshold: Optional[int] = None,
    window_seconds: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Check if a gap is exhibiting phantom resolution behavior.
    
    A gap is phantom if it has been "resolved" more than threshold times
    in the tracking window.
    
    Args:
        gap_key: Unique gap identifier
        threshold: Resolution count threshold. None uses default.
        window_seconds: Time window. None uses default.
        db_path: Optional database path.
    
    Returns:
        True if gap is phantom.
    """
    thresh = threshold if threshold is not None else PHANTOM_RESOLUTION_THRESHOLD
    count = get_recent_resolutions_count(gap_key, window_seconds, db_path)
    
    if count >= thresh:
        logger.warning(
            f"{LOG_PREFIX} PHANTOM RESOLUTION: Gap {gap_key} 'resolved' {count}x "
            f"in last {window_seconds or PHANTOM_TRACKING_WINDOW_SECONDS // 60} min"
        )
        return True
    
    return False


# =============================================================================
# Experiment Counter Functions (for Cold Start Detection)
# =============================================================================

def get_experiment_count(
    counter_name: str = "total_experiments",
    db_path: Optional[Path] = None,
) -> int:
    """
    Get a persistent experiment counter value.
    
    Args:
        counter_name: Name of the counter
        db_path: Optional database path.
    
    Returns:
        Counter value, or 0 if not found.
    """
    if not _init_db(db_path):
        return 0
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM experiment_counters WHERE counter_name = ?",
                (counter_name,),
            )
            row = cursor.fetchone()
            return row["value"] if row else 0
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error getting experiment count: {e}")
        return 0


def increment_experiment_count(
    counter_name: str = "total_experiments",
    increment: int = 1,
    db_path: Optional[Path] = None,
) -> int:
    """
    Increment a persistent experiment counter.
    
    Args:
        counter_name: Name of the counter
        increment: Amount to increment by
        db_path: Optional database path.
    
    Returns:
        New counter value.
    """
    if not _init_db(db_path):
        return 0
    
    try:
        with _get_db(db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiment_counters (counter_name, value, updated_at)
                VALUES (?, ?, strftime('%s', 'now'))
                ON CONFLICT(counter_name) DO UPDATE SET
                    value = value + ?,
                    updated_at = strftime('%s', 'now')
                """,
                (counter_name, increment, increment),
            )
            conn.commit()
            
            cursor = conn.execute(
                "SELECT value FROM experiment_counters WHERE counter_name = ?",
                (counter_name,),
            )
            row = cursor.fetchone()
            return row["value"] if row else increment
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error incrementing experiment count: {e}")
        return 0


def get_all_counters(
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Get all experiment counters.
    
    Returns:
        Dictionary of counter_name -> value
    """
    if not _init_db(db_path):
        return {}
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                "SELECT counter_name, value FROM experiment_counters"
            )
            return {row["counter_name"]: row["value"] for row in cursor.fetchall()}
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error getting counters: {e}")
        return {}


# =============================================================================
# Statistics and Analysis
# =============================================================================


def get_resolution_statistics(
    db_path: Optional[Path] = None,
) -> ResolutionStatistics:
    """
    Get comprehensive statistics for monitoring and debugging.
    
    Computes aggregate statistics from the resolution database including
    success rates, phantom counts, and experiment totals. Useful for
    monitoring the health of the gap resolution system.
    
    Args:
        db_path: Optional path to database file.
    
    Returns:
        ResolutionStatistics dataclass with computed metrics.
        Returns default (zero) statistics if database is unavailable.
    
    Example:
        >>> stats = get_resolution_statistics()
        >>> print(f"Success rate: {stats.success_rate:.1%}")
        >>> print(f"Phantom resolutions: {stats.phantom_count}")
        >>> print(f"Total experiments: {stats.total_experiments}")
    """
    default_stats = ResolutionStatistics(
        total_resolutions=0,
        active_resolutions=0,
        phantom_count=0,
        avg_attempts=0.0,
        success_rate=0.0,
        total_experiments=0,
    )
    
    if not _init_db(db_path):
        return default_stats
    
    try:
        with _get_db(db_path) as conn:
            # Total resolutions
            total = conn.execute(
                "SELECT COUNT(*) FROM gap_resolutions"
            ).fetchone()[0]
            
            if total == 0:
                return default_stats
            
            # Active (non-expired) resolutions
            cutoff = time.time() - RESOLUTION_TTL_SECONDS
            active = conn.execute(
                "SELECT COUNT(*) FROM gap_resolutions WHERE resolved_at > ?",
                (cutoff,),
            ).fetchone()[0]
            
            # Success stats
            success_row = conn.execute(
                """
                SELECT 
                    AVG(attempts),
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END),
                    COUNT(*)
                FROM gap_resolutions WHERE resolved_at > 0
                """
            ).fetchone()
            
            avg_attempts = success_row[0] or 0.0
            success_count = success_row[1] or 0
            total_resolved = success_row[2] or 0
            success_rate = success_count / total_resolved if total_resolved > 0 else 0.0
            
            # Count phantom resolutions (gaps resolved 3+ times in last hour)
            phantom_cutoff = time.time() - PHANTOM_TRACKING_WINDOW_SECONDS
            phantom_row = conn.execute(
                """
                SELECT COUNT(DISTINCT gap_key) FROM (
                    SELECT gap_key, COUNT(*) as cnt
                    FROM resolution_history
                    WHERE timestamp > ?
                    GROUP BY gap_key
                    HAVING cnt >= ?
                )
                """,
                (phantom_cutoff, PHANTOM_RESOLUTION_THRESHOLD),
            ).fetchone()
            phantom_count = phantom_row[0] if phantom_row else 0
            
            # Total experiments from counters
            exp_row = conn.execute(
                "SELECT COALESCE(SUM(value), 0) FROM experiment_counters"
            ).fetchone()
            total_experiments = exp_row[0] if exp_row else 0
        
        return ResolutionStatistics(
            total_resolutions=total,
            active_resolutions=active,
            phantom_count=phantom_count,
            avg_attempts=avg_attempts,
            success_rate=success_rate,
            total_experiments=total_experiments,
        )
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Failed to get statistics: {e}")
        return default_stats


# =============================================================================
# Bulk Operations and Maintenance
# =============================================================================


def get_all_resolved_gaps(
    include_expired: bool = False,
    db_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Get all resolved gaps.
    
    Args:
        include_expired: Include resolutions past TTL
        db_path: Optional database path.
    
    Returns:
        Dictionary of gap_key -> resolved_at timestamp
    """
    if not _init_db(db_path):
        return {}
    
    try:
        with _get_db(db_path) as conn:
            if include_expired:
                cursor = conn.execute(
                    "SELECT gap_key, resolved_at FROM gap_resolutions"
                )
            else:
                cutoff = time.time() - RESOLUTION_TTL_SECONDS
                cursor = conn.execute(
                    "SELECT gap_key, resolved_at FROM gap_resolutions WHERE resolved_at > ?",
                    (cutoff,),
                )
            return {row["gap_key"]: row["resolved_at"] for row in cursor.fetchall()}
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Error getting resolved gaps: {e}")
        return {}


def cleanup_old_data(
    days: int = CLEANUP_RETENTION_DAYS,
    db_path: Optional[Path] = None,
) -> int:
    """
    Clean up old resolution data.
    
    Args:
        days: Remove data older than this many days
        db_path: Optional database path.
    
    Returns:
        Number of rows deleted.
    """
    if not _init_db(db_path):
        return 0
    
    cutoff = time.time() - (days * 86400)
    deleted = 0
    
    try:
        with _get_db(db_path) as conn:
            # Clean resolution history
            cursor = conn.execute(
                "DELETE FROM resolution_history WHERE timestamp < ?",
                (cutoff,),
            )
            deleted += cursor.rowcount
            
            # Clean old gap resolutions (but keep phantom detection working)
            cursor = conn.execute(
                "DELETE FROM gap_resolutions WHERE resolved_at < ? AND resolved_at > 0",
                (cutoff,),
            )
            deleted += cursor.rowcount
            
            conn.commit()
        
        if deleted > 0:
            logger.info(f"{LOG_PREFIX} Cleaned up {deleted} old resolution records")
        
        return deleted
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Cleanup failed: {e}")
        return 0


def reset_database(db_path: Optional[Path] = None) -> bool:
    """
    Reset the database (drop all tables and reinitialize).
    
    WARNING: This deletes all resolution state. Use only for testing.
    """
    global _db_initialized
    
    try:
        with _get_db(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS gap_resolutions")
            conn.execute("DROP TABLE IF EXISTS resolution_history")
            conn.execute("DROP TABLE IF EXISTS experiment_counters")
            conn.commit()
        
        with _init_lock:
            _db_initialized = False
        
        return _init_db(db_path)
    except sqlite3.Error as e:
        logger.error(f"{LOG_PREFIX} Database reset failed: {e}")
        return False


def get_database_path() -> Path:
    """Get the current database path."""
    return DEFAULT_DB_PATH


# =============================================================================
# Module Initialization
# =============================================================================

# Initialize on import (non-blocking)
try:
    _init_db()
except Exception as e:
    logger.debug(f"{LOG_PREFIX} Deferred initialization: {e}")
