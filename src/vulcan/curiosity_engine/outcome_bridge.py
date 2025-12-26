"""
SQLite bridge for cross-process query outcome sharing.

Part of the VULCAN-AGI system.

This module provides a SQLite-based communication channel between the main
query processing pipeline and the CuriosityEngine subprocess. The main process
writes query outcomes, and the subprocess reads them for analysis.

Key Features:
- Thread-safe SQLite operations with proper connection management
- Automatic database initialization with idempotent schema creation
- TTL-based cleanup to prevent unbounded database growth
- Analysis functions for gap detection from outcomes

Performance Characteristics:
- Uses WAL mode for concurrent read/write access
- Bounded queries with LIMIT to prevent unbounded result sets
- Connection pooling via context manager pattern

Usage:
    # In main query handler (after query completion):
    from vulcan.curiosity_engine.outcome_bridge import record_query_outcome
    record_query_outcome(
        query_id="q_abc123",
        status="success",
        routing_time_ms=150.0,
        total_time_ms=2500.0,
        complexity=0.45,
        query_type="reasoning",
    )

    # In CuriosityEngine subprocess (during learning cycle):
    from vulcan.curiosity_engine.outcome_bridge import (
        get_recent_outcomes,
        analyze_outcomes_for_gaps,
    )
    outcomes = get_recent_outcomes(minutes=60, limit=500)
    gaps = analyze_outcomes_for_gaps(outcomes)
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default database path - uses /tmp for cross-process access
DEFAULT_DB_PATH = Path("/tmp/vulcan_query_outcomes.db")

# Thread-local storage for database connections
_local = threading.local()

# Module-level lock for database initialization
_init_lock = threading.Lock()
_db_initialized = False


@dataclass
class OutcomeStatistics:
    """Statistics computed from query outcomes."""

    total: int
    unprocessed: int
    avg_routing_ms: float
    max_routing_ms: float
    slow_routing_count: int
    error_count: int
    success_rate: float


@contextmanager
def _get_db(db_path: Optional[Path] = None):
    """
    Get database connection with proper cleanup.

    Uses WAL mode for better concurrent access and sets a reasonable
    timeout for busy waiting.

    Args:
        db_path: Optional path to database file. Defaults to DEFAULT_DB_PATH.

    Yields:
        SQLite connection with row factory set to sqlite3.Row
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for better concurrent access
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass  # Ignore if already set

    try:
        yield conn
    finally:
        conn.close()


def _init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize database schema (idempotent).

    Creates the query_outcomes table and necessary indexes if they don't exist.
    Thread-safe and can be called multiple times without side effects.

    Args:
        db_path: Optional path to database file. Defaults to DEFAULT_DB_PATH.
    """
    global _db_initialized

    # Fast path if already initialized
    if _db_initialized:
        return

    with _init_lock:
        # Double-checked locking
        if _db_initialized:
            return

        try:
            with _get_db(db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT UNIQUE,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        routing_time_ms REAL,
                        total_time_ms REAL,
                        complexity REAL,
                        query_type TEXT,
                        tasks INTEGER DEFAULT 1,
                        error_type TEXT,
                        processed INTEGER DEFAULT 0,
                        created_at REAL DEFAULT (strftime('%s', 'now'))
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp "
                    "ON query_outcomes(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_processed "
                    "ON query_outcomes(processed)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status "
                    "ON query_outcomes(status)"
                )
                conn.commit()

            _db_initialized = True
            logger.info("[OutcomeBridge] Database initialized successfully")

        except sqlite3.Error as e:
            logger.warning(f"[OutcomeBridge] Database initialization failed: {e}")


def record_query_outcome(
    query_id: str,
    status: str,
    routing_time_ms: float,
    total_time_ms: float,
    complexity: float,
    query_type: str,
    tasks: int = 1,
    error_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Record a query outcome to shared storage.

    Call this from the main process after each query completes. The outcome
    will be available for the CuriosityEngine subprocess to analyze.

    Args:
        query_id: Unique query identifier
        status: Query status ("success", "error", "timeout", etc.)
        routing_time_ms: Time spent in query routing phase (ms)
        total_time_ms: Total query processing time (ms)
        complexity: Query complexity score (0.0 to 1.0)
        query_type: Type of query (reasoning, perception, planning, etc.)
        tasks: Number of agent tasks created
        error_type: Error type if status is "error"
        db_path: Optional path to database file

    Returns:
        True if recording succeeded, False otherwise
    """
    _init_db(db_path)

    try:
        with _get_db(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_outcomes 
                (query_id, timestamp, status, routing_time_ms, total_time_ms, 
                 complexity, query_type, tasks, error_type, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    query_id,
                    datetime.utcnow().isoformat(),
                    status,
                    routing_time_ms,
                    total_time_ms,
                    complexity,
                    query_type,
                    tasks,
                    error_type,
                ),
            )
            conn.commit()

        logger.info(
            f"[QueryOutcome] Recorded: {query_id}, status={status}, "
            f"routing={routing_time_ms:.0f}ms, total={total_time_ms:.0f}ms, "
            f"complexity={complexity:.2f}, type={query_type}"
        )
        return True

    except sqlite3.Error as e:
        logger.warning(f"[OutcomeBridge] Failed to record outcome: {e}")
        return False


def get_recent_outcomes(
    minutes: int = 60,
    limit: int = 500,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent outcomes for analysis.

    Call this from the CuriosityEngine subprocess during learning cycles
    to retrieve query outcomes for gap analysis.

    Args:
        minutes: Look back window in minutes (default: 60)
        limit: Maximum number of results (default: 500)
        db_path: Optional path to database file

    Returns:
        List of outcome dictionaries with all fields
    """
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []

    cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()

    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM query_outcomes 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff, limit),
            )
            rows = cursor.fetchall()

        outcomes = [dict(row) for row in rows]

        if outcomes:
            logger.info(
                f"[OutcomeBridge] Loaded {len(outcomes)} recent outcomes "
                f"(last {minutes} min)"
            )

        return outcomes

    except sqlite3.Error as e:
        logger.warning(f"[OutcomeBridge] Failed to load outcomes: {e}")
        return []


def get_unprocessed_outcomes(
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Get outcomes that haven't been processed by the learning system.

    Args:
        limit: Maximum number of results
        db_path: Optional path to database file

    Returns:
        List of unprocessed outcome dictionaries
    """
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []

    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM query_outcomes 
                WHERE processed = 0
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        logger.warning(f"[OutcomeBridge] Failed to load unprocessed outcomes: {e}")
        return []


def mark_outcomes_processed(
    outcome_ids: List[int],
    db_path: Optional[Path] = None,
) -> bool:
    """
    Mark outcomes as processed by the learning system.

    Args:
        outcome_ids: List of outcome IDs to mark as processed
        db_path: Optional path to database file

    Returns:
        True if successful, False otherwise
    """
    if not outcome_ids:
        return True

    try:
        with _get_db(db_path) as conn:
            placeholders = ",".join("?" * len(outcome_ids))
            conn.execute(
                f"UPDATE query_outcomes SET processed = 1 WHERE id IN ({placeholders})",
                outcome_ids,
            )
            conn.commit()

        return True

    except sqlite3.Error as e:
        logger.warning(f"[OutcomeBridge] Failed to mark outcomes processed: {e}")
        return False


def get_outcome_statistics(
    db_path: Optional[Path] = None,
) -> OutcomeStatistics:
    """
    Get statistics for monitoring and gap detection.

    Computes aggregate statistics from the outcomes database including
    success rates, average routing times, and error counts.

    Args:
        db_path: Optional path to database file

    Returns:
        OutcomeStatistics dataclass with computed metrics
    """
    path = db_path or DEFAULT_DB_PATH

    default_stats = OutcomeStatistics(
        total=0,
        unprocessed=0,
        avg_routing_ms=0.0,
        max_routing_ms=0.0,
        slow_routing_count=0,
        error_count=0,
        success_rate=0.0,
    )

    if not path.exists():
        return default_stats

    try:
        with _get_db(db_path) as conn:
            # Total and unprocessed counts
            total = conn.execute("SELECT COUNT(*) FROM query_outcomes").fetchone()[0]
            unprocessed = conn.execute(
                "SELECT COUNT(*) FROM query_outcomes WHERE processed = 0"
            ).fetchone()[0]

            if total == 0:
                return default_stats

            # Compute statistics from last hour
            hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            stats = conn.execute(
                """
                SELECT 
                    AVG(routing_time_ms),
                    MAX(routing_time_ms),
                    COUNT(CASE WHEN routing_time_ms > 10000 THEN 1 END),
                    COUNT(CASE WHEN status != 'success' THEN 1 END),
                    COUNT(CASE WHEN status = 'success' THEN 1 END)
                FROM query_outcomes WHERE timestamp > ?
                """,
                (hour_ago,),
            ).fetchone()

            success_count = stats[4] or 0
            error_count = stats[3] or 0
            total_recent = success_count + error_count
            success_rate = success_count / total_recent if total_recent > 0 else 0.0

        return OutcomeStatistics(
            total=total,
            unprocessed=unprocessed,
            avg_routing_ms=stats[0] or 0.0,
            max_routing_ms=stats[1] or 0.0,
            slow_routing_count=stats[2] or 0,
            error_count=error_count,
            success_rate=success_rate,
        )

    except sqlite3.Error as e:
        logger.warning(f"[OutcomeBridge] Failed to get statistics: {e}")
        return default_stats


def analyze_outcomes_for_gaps(outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze outcomes and return detected knowledge gaps.

    Examines patterns in query outcomes to identify systematic issues
    that should be addressed by the learning system.

    Gap Types Detected:
        - slow_routing: Queries taking >10s to route
        - complex_query_handling: High complexity queries averaging >30s
        - high_error_rate: Error rate exceeding 10%
        - routing_variance: High variance in routing times

    Args:
        outcomes: List of outcome dictionaries from get_recent_outcomes()

    Returns:
        List of gap dictionaries with type, description, severity, and evidence
    """
    if not outcomes:
        return []

    gaps: List[Dict[str, Any]] = []

    # Gap: Slow routing (>10s)
    slow = [o for o in outcomes if (o.get("routing_time_ms") or 0) > 10000]
    if len(slow) >= 3:
        avg_time = sum(o.get("routing_time_ms", 0) for o in slow) / len(slow)
        gaps.append(
            {
                "gap_type": "slow_routing",
                "description": f"{len(slow)} queries with routing > 10s "
                f"(avg: {avg_time / 1000:.1f}s)",
                "severity": min(1.0, len(slow) / 10),
                "evidence": [o.get("query_id") for o in slow[:5]],
                "suggested_action": "Implement embedding cache",
            }
        )

    # Gap: High complexity queries taking too long (>30s average)
    complex_q = [o for o in outcomes if (o.get("complexity") or 0) > 0.5]
    if complex_q:
        avg_time = (
            sum(o.get("total_time_ms", 0) for o in complex_q) / len(complex_q)
        )
        if avg_time > 30000:
            gaps.append(
                {
                    "gap_type": "complex_query_handling",
                    "description": f"Complex queries averaging {avg_time / 1000:.1f}s",
                    "severity": min(1.0, avg_time / 60000),
                    "evidence": [o.get("query_id") for o in complex_q[:5]],
                    "suggested_action": "Optimize reasoning pipeline",
                }
            )

    # Gap: High error rate (>10%)
    if outcomes:
        errors = [o for o in outcomes if o.get("status") != "success"]
        error_rate = len(errors) / len(outcomes)
        if error_rate > 0.1:
            # Group by error type
            error_types = {}
            for e in errors:
                error_type = e.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            gaps.append(
                {
                    "gap_type": "high_error_rate",
                    "description": f"{error_rate * 100:.1f}% error rate",
                    "severity": min(1.0, error_rate * 5),
                    "evidence": list(error_types.keys())[:5],
                    "suggested_action": "Investigate error patterns",
                }
            )

    # Gap: High routing time variance (indicates inconsistent performance)
    if len(outcomes) >= 10:
        routing_times = [
            o.get("routing_time_ms", 0)
            for o in outcomes
            if o.get("routing_time_ms") is not None
        ]
        if routing_times:
            import statistics

            try:
                mean_time = statistics.mean(routing_times)
                std_time = statistics.stdev(routing_times)
                cv = std_time / mean_time if mean_time > 0 else 0

                # Coefficient of variation > 1 indicates high variance
                if cv > 1.0 and mean_time > 1000:
                    gaps.append(
                        {
                            "gap_type": "routing_variance",
                            "description": f"High routing time variance "
                            f"(CV={cv:.2f}, mean={mean_time:.0f}ms)",
                            "severity": min(1.0, cv / 2),
                            "evidence": [
                                f"mean={mean_time:.0f}ms",
                                f"std={std_time:.0f}ms",
                            ],
                            "suggested_action": "Investigate cache effectiveness",
                        }
                    )
            except statistics.StatisticsError:
                pass  # Not enough data

    return gaps


def cleanup_old_outcomes(
    days: int = 7,
    db_path: Optional[Path] = None,
) -> int:
    """
    Remove outcomes older than specified days.

    Should be called periodically to prevent unbounded database growth.

    Args:
        days: Number of days to keep (default: 7)
        db_path: Optional path to database file

    Returns:
        Number of rows deleted
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM query_outcomes WHERE timestamp < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"[OutcomeBridge] Cleaned up {deleted} old outcomes")

        return deleted

    except sqlite3.Error as e:
        logger.warning(f"[OutcomeBridge] Cleanup failed: {e}")
        return 0


# Initialize on import (non-blocking)
try:
    _init_db()
except Exception as e:
    logger.debug(f"[OutcomeBridge] Deferred initialization: {e}")
