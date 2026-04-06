# src/security_audit_engine.py
"""
Graphix IR Security Audit Engine
================================
A robust, self-contained engine for logging auditable events to a
queryable SQLite database, with integrated alerting for critical events.

FIXES APPLIED:
- Thread-safe connection pooling with proper locking
- Transaction management with ACID guarantees
- Removed emojis from alerts (style compliance)
- Context manager support for automatic cleanup
- Database corruption detection and recovery
- Connection timeout handling
- Comprehensive error handling
- Statistics tracking
- Schema migration for backward compatibility
- Async-compatible logging with run_in_executor (Architectural Fix #4)
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Alerting Integration ---
# Gracefully handle missing slack_sdk library
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    SLACK_AVAILABLE = True
except ImportError:
    WebClient = None
    SlackApiError = None
    SLACK_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuditEngineError(Exception):
    """Base exception for audit engine errors."""


class DatabaseCorruptionError(AuditEngineError):
    """Raised when database corruption is detected."""


class ConnectionPool:
    """Thread-safe SQLite connection pool for SecurityAuditEngine."""

    def __init__(self, db_path: Path, max_connections: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self.pool: List[sqlite3.Connection] = []
        self.in_use: set = set()
        self.lock = threading.Lock()
        self.logger = logging.getLogger("ConnectionPool")

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool with timeout."""
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            with self.lock:
                # Try to reuse existing connection
                if self.pool:
                    conn = self.pool.pop()
                    self.in_use.add(id(conn))
                    return conn

                # Create new connection if under limit
                if len(self.in_use) < self.max_connections:
                    conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=self.timeout,
                        check_same_thread=False,
                        isolation_level=None,  # Autocommit mode, handle transactions manually
                    )
                    # Enable WAL mode for better concurrency
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA foreign_keys=ON")

                    self.in_use.add(id(conn))
                    return conn

            # Wait briefly before retrying
            time.sleep(0.01)

        raise AuditEngineError("Connection pool timeout - no connections available")

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        with self.lock:
            conn_id = id(conn)
            if conn_id in self.in_use:
                self.in_use.remove(conn_id)
                if len(self.pool) < self.max_connections:
                    self.pool.append(conn)
                else:
                    conn.close()

    def close_all(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.pool:
                try:
                    conn.close()
                except Exception as e:
                    self.logger.error(f"Error closing connection: {e}")
            self.pool.clear()
            self.in_use.clear()


class SecurityAuditEngine:
    """
    Logs structured audit events to a SQLite database and sends alerts for critical events.

    Configuration for alerting is handled via environment variables:
    - SLACK_BOT_TOKEN: Your Slack App's Bot User OAuth Token.
    - SLACK_ALERT_CHANNEL: The Slack channel ID or name (e.g., #security-alerts) to post to.

    Features:
    - Thread-safe connection pooling
    - Transaction management
    - Database corruption detection
    - Context manager support
    - Statistics tracking
    - Automatic cleanup
    - Schema migration for backward compatibility
    - Async-compatible logging with run_in_executor (Fix #4)
    """

    def __init__(self, db_path: str = "audit.db", max_connections: int = 5):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("SecurityAuditEngine")

        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool
        self.pool = ConnectionPool(self.db_path, max_connections)

        # Thread safety
        self.write_lock = threading.RLock()

        # ThreadPoolExecutor for async DB operations (Architectural Fix #4)
        # This allows SQLite writes to be decoupled from the async event loop,
        # preventing the "distinct pause" seen when the audit log writes to the database.
        # max_workers is configurable via environment variable
        import os
        audit_max_workers = int(os.getenv("AUDIT_DB_MAX_WORKERS", "2"))
        self._executor = ThreadPoolExecutor(max_workers=audit_max_workers, thread_name_prefix="audit_db_")

        # Statistics
        self.stats = {
            "events_logged": 0,
            "alerts_sent": 0,
            "errors": 0,
            "queries_executed": 0,
        }
        self.stats_lock = threading.Lock()

        # Check and migrate schema if needed
        self._check_and_migrate_schema()

        # Initialize database
        self._initialize_db()

        # Verify database integrity
        self._verify_database_integrity()

        # --- Alerting Setup ---
        self.slack_client = None
        self.slack_channel = os.getenv("SLACK_ALERT_CHANNEL")
        slack_token = os.getenv("SLACK_BOT_TOKEN")

        if slack_token and self.slack_channel:
            if SLACK_AVAILABLE:
                try:
                    self.slack_client = WebClient(token=slack_token)
                    self.logger.info(
                        f"Slack alerting enabled. Will send critical alerts to '{self.slack_channel}'."
                    )
                except Exception as e:
                    self.logger.error(f"Failed to initialize Slack client: {e}")
            else:
                self.logger.warning(
                    "SLACK_BOT_TOKEN is set, but 'slack_sdk' is not installed. Alerting is disabled."
                )
        else:
            self.logger.info(
                "Slack environment variables not set. Real-time alerting is disabled."
            )

        self.critical_events = {
            "integrity_failure",
            "unauthorized_access",
            "bias_detected",
            "security_risk",
            "data_breach",
            "authentication_failure",
        }

        self.logger.info(
            f"Audit engine initialized. Logging to database: {self.db_path}"
        )

    def _check_and_migrate_schema(self):
        """
        Check schema version and migrate if needed.
        
        FIXED: Use ALTER TABLE to preserve existing data instead of deleting database.
        This migration adds the 'severity' column if it's missing, preserving all
        existing audit log entries.
        """
        if not self.db_path.exists():
            self.logger.debug("Database does not exist yet, will be created")
            return

        try:
            # Try to open and check schema
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check if audit_log table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
            )
            table_exists = cursor.fetchone()

            if table_exists:
                # Check if severity column exists
                cursor.execute("PRAGMA table_info(audit_log)")
                columns = {row[1] for row in cursor.fetchall()}

                if "severity" not in columns:
                    self.logger.info(
                        "Old schema detected (missing 'severity' column). "
                        "Migrating schema using ALTER TABLE to preserve data..."
                    )
                    
                    try:
                        # CRITICAL FIX: Use ALTER TABLE instead of deleting database
                        # This preserves all existing audit log data
                        cursor.execute(
                            "ALTER TABLE audit_log ADD COLUMN severity TEXT DEFAULT 'info'"
                        )
                        conn.commit()
                        
                        self.logger.info(
                            "Schema migration completed successfully. "
                            "Added 'severity' column with default value 'info'. "
                            "All existing data preserved."
                        )
                    except sqlite3.OperationalError as e:
                        # If ALTER TABLE fails (e.g., column already exists but not detected),
                        # log and continue
                        self.logger.warning(
                            f"ALTER TABLE failed (column may already exist): {e}"
                        )
                else:
                    self.logger.debug("Schema is up to date")

            conn.close()

        except sqlite3.DatabaseError as e:
            self.logger.warning(
                f"Database appears corrupted: {e}. "
                "Database initialization will handle recreation if needed."
            )
        except Exception as e:
            self.logger.debug(f"Schema check encountered issue: {e}")

    @contextmanager
    def _get_connection(self):
        """Context manager for getting and returning connections."""
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            self.pool.return_connection(conn)

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions with ACID guarantees."""
        conn = self.pool.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            self.logger.error(f"Transaction failed, rolled back: {e}")
            raise
        finally:
            self.pool.return_connection(conn)

    def _initialize_db(self):
        """Creates the audit log table and indexes if they don't exist."""
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Create audit log table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        details TEXT,
                        severity TEXT DEFAULT 'info'
                    )
                """
                )

                # Create indexes for faster querying
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_log (event_type)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log (timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_severity ON audit_log (severity)"
                )

                # Create metadata table for tracking
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TEXT
                    )
                """
                )

                # Store initialization timestamp
                cursor.execute(
                    """INSERT OR REPLACE INTO audit_metadata (key, value, updated_at)
                       VALUES (?, ?, ?)""",
                    (
                        "initialized_at",
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat(),
                    ),
                )

            self.logger.info("Database initialized successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise AuditEngineError(f"Failed to initialize database: {e}")

    def _verify_database_integrity(self):
        """Verify database integrity and attempt recovery if corrupted."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Run integrity check
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                if result[0] != "ok":
                    raise DatabaseCorruptionError(
                        f"Database integrity check failed: {result[0]}"
                    )

                self.logger.debug("Database integrity verified")

        except sqlite3.DatabaseError as e:
            self.logger.error(f"Database corruption detected: {e}")
            # Attempt recovery
            self._attempt_recovery()

    def _attempt_recovery(self):
        """Attempt to recover from database corruption."""
        self.logger.warning("Attempting database recovery...")

        try:
            # Create backup of corrupted database
            backup_path = self.db_path.with_suffix(".db.corrupted")
            import shutil

            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Backed up corrupted database to {backup_path}")

            # Close all connections
            self.pool.close_all()

            # Try to dump and restore
            self.db_path.with_suffix(".db.dump")

            # Recreate database
            if self.db_path.exists():
                self.db_path.unlink()

            # Reinitialize
            self.pool = ConnectionPool(self.db_path, self.pool.max_connections)
            self._initialize_db()

            self.logger.info("Database recovery completed")

        except Exception as e:
            self.logger.error(f"Database recovery failed: {e}")
            raise DatabaseCorruptionError(f"Unable to recover database: {e}")

    def log_event(self, event_type: str, details: Dict, severity: str = "info"):
        """
        Logs a structured event to the database and triggers an alert if the event is critical.

        Args:
            event_type: The type of event (e.g., 'graph_submitted', 'integrity_failure').
            details: A dictionary with event-specific information.
            severity: Event severity ('info', 'warning', 'error', 'critical').
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        details_json = json.dumps(details)

        try:
            # Use transaction for ACID guarantees
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO audit_log (timestamp, event_type, details, severity) VALUES (?, ?, ?, ?)",
                    (timestamp, event_type, details_json, severity),
                )

            # Update statistics
            with self.stats_lock:
                self.stats["events_logged"] += 1

            # Trigger alert for critical events
            if event_type in self.critical_events or severity == "critical":
                self._send_alert(event_type, details, timestamp, severity)

            self.logger.debug(f"Logged event: {event_type} ({severity})")

        except sqlite3.Error as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            self.logger.error(f"Failed to write to audit database: {e}")
            raise AuditEngineError(f"Failed to log event: {e}")

    async def log_event_async(self, event_type: str, details: Dict, severity: str = "info"):
        """
        Async version of log_event using run_in_executor for non-blocking DB writes.
        
        This is the recommended method for async code paths (Architectural Fix #4).
        SQLite is synchronous and file-locked, so we decouple the write operation
        from the async event loop to prevent blocking.

        Args:
            event_type: The type of event (e.g., 'graph_submitted', 'integrity_failure').
            details: A dictionary with event-specific information.
            severity: Event severity ('info', 'warning', 'error', 'critical').
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self.log_event,
            event_type,
            details,
            severity
        )

    def _send_alert(
        self, event_type: str, details: Dict, timestamp: str, severity: str = "critical"
    ):
        """Sends a formatted alert to the configured Slack channel."""
        if not self.slack_client or not self.slack_channel:
            return

        try:
            # FIXED: Format message without emojis (style compliance)
            message = (
                f"*Critical Security Event*\n"
                f">*Event Type*: `{event_type}`\n"
                f">*Severity*: `{severity}`\n"
                f">*Timestamp (UTC)*: `{timestamp}`\n"
                f">*Details*:\n"
                f"```{json.dumps(details, indent=2)}```"
            )

            self.slack_client.chat_postMessage(channel=self.slack_channel, text=message)

            # Update statistics
            with self.stats_lock:
                self.stats["alerts_sent"] += 1

            self.logger.info(f"Successfully sent Slack alert for event: {event_type}")

        except SlackApiError as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            self.logger.error(
                f"Failed to send Slack alert: {e.response.get('error', 'Unknown error')}"
            )
        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            self.logger.error(
                f"An unexpected error occurred while sending Slack alert: {e}"
            )

    def query_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Queries the audit log for events with multiple filter options.

        Args:
            event_type: Optional event type filter
            severity: Optional severity filter
            start_time: Optional start timestamp (ISO format)
            end_time: Optional end timestamp (ISO format)
            limit: Maximum number of results

        Returns:
            A list of event dictionaries.
        """
        results = []

        try:
            # Build query dynamically
            query = "SELECT timestamp, event_type, details, severity FROM audit_log WHERE 1=1"
            params = []

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                for row in cursor.fetchall():
                    results.append(
                        {
                            "timestamp": row[0],
                            "event_type": row[1],
                            "details": json.loads(row[2]),
                            "severity": row[3],
                        }
                    )

            # Update statistics
            with self.stats_lock:
                self.stats["queries_executed"] += 1

        except sqlite3.Error as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            self.logger.error(f"Failed to query audit database: {e}")
            raise AuditEngineError(f"Failed to query events: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit engine statistics."""
        with self.stats_lock:
            stats = self.stats.copy()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Total events count
                cursor.execute("SELECT COUNT(*) FROM audit_log")
                stats["total_events"] = cursor.fetchone()[0]

                # Events by type
                cursor.execute(
                    "SELECT event_type, COUNT(*) FROM audit_log GROUP BY event_type"
                )
                stats["events_by_type"] = dict(cursor.fetchall())

                # Events by severity
                cursor.execute(
                    "SELECT severity, COUNT(*) FROM audit_log GROUP BY severity"
                )
                stats["events_by_severity"] = dict(cursor.fetchall())

                # Database size
                stats["db_size_bytes"] = self.db_path.stat().st_size

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")

        return stats

    def cleanup_old_events(self, days: int = 90) -> int:
        """
        Remove events older than specified days.

        Args:
            days: Number of days to retain

        Returns:
            Number of events deleted
        """
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM audit_log WHERE timestamp < ?", (cutoff_date,)
                )
                deleted = cursor.rowcount

            self.logger.info(f"Cleaned up {deleted} events older than {days} days")
            return deleted

        except sqlite3.Error as e:
            self.logger.error(f"Failed to cleanup old events: {e}")
            raise AuditEngineError(f"Failed to cleanup events: {e}")

    def close(self):
        """Closes all database connections and shuts down the executor."""
        try:
            # Shut down the executor for async operations
            if hasattr(self, '_executor') and self._executor is not None:
                self._executor.shutdown(wait=True)
                self.logger.debug("Audit executor shut down.")
            
            self.pool.close_all()
            self.logger.info("Audit database connections closed.")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False


# Global audit log path for transparency report
AUDIT_LOG_PATH = "audit_log.jsonl"


def parse_audit_log_line(line: str) -> Dict:
    """Parse a line from the audit log."""
    return json.loads(line)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SecurityAuditEngine Production Demo")
    print("=" * 70 + "\n")

    # Demo with context manager
    with SecurityAuditEngine(db_path="demo_audit.db") as audit:
        print("--- Test 1: Logging Regular Events ---")

        # Log some events
        audit.log_event(
            "user_login", {"user_id": "user123", "ip": "192.168.1.1"}, severity="info"
        )
        audit.log_event(
            "graph_submitted", {"graph_id": "g001", "size": 100}, severity="info"
        )
        audit.log_event("validation_passed", {"graph_id": "g001"}, severity="info")

        print("Logged 3 regular events")

        print("\n--- Test 2: Logging Critical Event (triggers alert) ---")

        # This should trigger an alert
        audit.log_event(
            "integrity_failure",
            {
                "graph_id": "g002",
                "reason": "signature mismatch",
                "severity_level": "high",
            },
            severity="critical",
        )

        print("Logged critical event (alert triggered if Slack configured)")

        print("\n--- Test 3: Querying Events ---")

        # Query all events
        all_events = audit.query_events(limit=10)
        print(f"Total events retrieved: {len(all_events)}")

        # Query by type
        login_events = audit.query_events(event_type="user_login")
        print(f"Login events: {len(login_events)}")

        # Query by severity
        critical_events = audit.query_events(severity="critical")
        print(f"Critical events: {len(critical_events)}")

        print("\n--- Test 4: Statistics ---")
        stats = audit.get_statistics()
        print(f"Total events logged: {stats['total_events']}")
        print(f"Events by type: {stats.get('events_by_type', {})}")
        print(f"Events by severity: {stats.get('events_by_severity', {})}")
        print(f"Database size: {stats['db_size_bytes'] / 1024:.2f} KB")
        print(f"Queries executed: {stats['queries_executed']}")
        print(f"Alerts sent: {stats['alerts_sent']}")

        print("\n--- Test 5: Thread Safety Test ---")

        import concurrent.futures

        def log_concurrent_event(i):
            audit.log_event(
                "concurrent_test",
                {"thread_id": i, "timestamp": datetime.utcnow().isoformat()},
                severity="info",
            )
            return i

        # Log events from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(log_concurrent_event, i) for i in range(20)]
            concurrent.futures.wait(futures)

        concurrent_events = audit.query_events(event_type="concurrent_test")
        print(f"Concurrent events logged: {len(concurrent_events)}")

        print("\n--- Test 6: Cleanup Old Events ---")

        # This won't delete anything in the demo since all events are new
        deleted = audit.cleanup_old_events(days=1)
        print(f"Old events cleaned up: {deleted}")

    # Clean up demo database
    import os

    if os.path.exists("demo_audit.db"):
        os.remove("demo_audit.db")
        print("\nDemo database cleaned up")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70 + "\n")
