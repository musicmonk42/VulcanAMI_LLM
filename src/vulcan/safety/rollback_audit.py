# rollback_audit.py
"""
Rollback management and audit logging for VULCAN-AGI Safety Module.
Provides snapshot-based rollback capabilities and comprehensive audit trail management.
"""

import copy
import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import threading
import time
import uuid
import zlib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .safety_types import (ActionType, RollbackSnapshot, SafetyReport,
                           SafetyViolationType)

logger = logging.getLogger(__name__)


# Helper function for safe logging during shutdown
def safe_log(log_func, message):
    """Log safely during shutdown when logging may be closed."""
    try:
        log_func(message)
    except (ValueError, AttributeError, OSError, RuntimeError):
        # Silently ignore all logging errors during shutdown
        # This includes:
        # - ValueError: I/O operation on closed file
        # - AttributeError: Logger object has no attribute
        # - OSError: File operation errors
        # - RuntimeError: Logger/handler is closed
        pass


# ============================================================
# MEMORY BOUNDED DEQUE
# ============================================================


class MemoryBoundedDeque:
    """Deque with memory limit instead of item count limit.

    Automatically removes oldest items when memory limit is exceeded.
    Useful for preventing unbounded memory growth in log buffers.
    """

    def __init__(self, max_size_mb: int = 10):
        """
        Initialize memory-bounded deque.

        Args:
            max_size_mb: Maximum size in megabytes
        """
        self.deque = deque()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.lock = threading.RLock()

    def append(self, item):
        """Add item to deque, removing old items if needed."""
        with self.lock:
            item_size = self._estimate_size(item)

            # Remove items until we have space
            while (
                self.current_size_bytes + item_size > self.max_size_bytes and self.deque
            ):
                removed = self.deque.popleft()
                self.current_size_bytes -= self._estimate_size(removed)

            self.deque.append(item)
            self.current_size_bytes += item_size

    def _estimate_size(self, item) -> int:
        """Estimate size of item in bytes."""
        try:
            if isinstance(item, dict):
                # Rough estimate: JSON size
                return len(json.dumps(item, default=str))
            return sys.getsizeof(item)
        except Exception:
            return 1024  # Default 1KB if estimation fails

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        with self.lock:
            return self.current_size_bytes / (1024 * 1024)

    def __len__(self):
        """Return number of items in deque."""
        with self.lock:
            return len(self.deque)

    def __iter__(self):
        """Iterate over items in deque."""
        with self.lock:
            return iter(list(self.deque))

    def clear(self):
        """Clear all items from deque."""
        with self.lock:
            self.deque.clear()
            self.current_size_bytes = 0

    def __bool__(self):
        """Return True if deque is not empty."""
        with self.lock:
            return bool(self.deque)


# ============================================================
# ROLLBACK MANAGER
# ============================================================


class RollbackManager:
    """
    Manages rollback and quarantine functionality with snapshot-based state recovery.
    Provides versioned state management, quarantine capabilities, and rollback history.
    """

    def __init__(
        self, max_snapshots: int = 100, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize rollback manager.

        Args:
            max_snapshots: Maximum number of snapshots to retain
            config: Additional configuration options
                - test_mode: Enable fast test mode (no storage, no workers)
                - worker_check_interval: Worker sleep interval (default 10s, 0.1s in test_mode)
                - enable_storage: Enable SQLite storage (default True, False in test_mode)
                - enable_workers: Enable background threads (default True, False in test_mode)
        """
        self.config = config or {}

        # TEST MODE SUPPORT - Critical for fast test execution
        self.test_mode = self.config.get("test_mode", False)
        if self.test_mode:
            logger.info("=" * 60)
            logger.info("ROLLBACK MANAGER IN TEST MODE")
            logger.info("Fast shutdown, no storage, no worker threads")
            logger.info("=" * 60)
            # Apply test mode defaults
            self.config.setdefault("worker_check_interval", 0.1)
            self.config.setdefault("enable_storage", False)
            self.config.setdefault("enable_workers", False)
            self.config.setdefault("auto_cleanup", False)

        self.max_snapshots = min(max_snapshots, 1000)  # Cap at 1000
        self.snapshots = deque(maxlen=self.max_snapshots)
        self.quarantine = {}
        self.rollback_history = deque(maxlen=1000)
        self.snapshot_index = {}  # Fast lookup by ID
        self.lock = threading.RLock()
        self.db_lock = threading.RLock()

        # Shutdown flag
        self._shutdown = False

        # CRITICAL FIX: Add shutdown event for interruptible sleep
        self._shutdown_event = threading.Event()

        # Configuration
        self.compress_snapshots = self.config.get("compress_snapshots", True)
        self.verify_integrity = self.config.get("verify_integrity", True)
        self.auto_cleanup = self.config.get("auto_cleanup", True)
        self.snapshot_retention_days = self.config.get("snapshot_retention_days", 7)
        self.quarantine_retention_days = self.config.get(
            "quarantine_retention_days", 30
        )

        # CRITICAL FIX: Configurable worker check interval (default 10s for faster shutdown)
        # Set to 0.1 seconds in test mode
        self.worker_check_interval = self.config.get("worker_check_interval", 10.0)

        # Storage configuration - MAKE OPTIONAL
        self.enable_storage = self.config.get("enable_storage", True)
        self.storage_path = Path(self.config.get("storage_path", "rollback_storage"))
        if self.enable_storage:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.metrics = {
            "total_snapshots": 0,
            "total_rollbacks": 0,
            "successful_rollbacks": 0,
            "failed_rollbacks": 0,
            "quarantined_actions": 0,
            "storage_bytes": 0,
        }

        # Initialize storage - MAKE OPTIONAL AND SAFE
        self.conn = None
        if self.enable_storage:
            try:
                self._initialize_storage()
            except Exception as e:
                logger.error(f"Storage initialization failed: {e}")
                if not self.test_mode:
                    raise
                logger.warning("Continuing in test mode without storage")
                self.conn = None
        else:
            logger.info("Storage disabled (test_mode or explicit config)")

        # Cleanup thread reference
        self.cleanup_thread = None

        # Start cleanup thread - MAKE OPTIONAL
        self.enable_workers = self.config.get("enable_workers", True)
        if self.auto_cleanup and self.enable_workers:
            try:
                self._start_cleanup_thread()
            except Exception as e:
                logger.error(f"Failed to start cleanup thread: {e}")
                if not self.test_mode:
                    raise
        else:
            logger.info(f"Background workers disabled (test_mode={self.test_mode})")

        # Register cleanup
        # atexit.register(self.shutdown) # Removed for test suite compatibility

        logger.info(
            f"RollbackManager initialized: test_mode={self.test_mode}, "
            f"storage={self.enable_storage}, workers={self.enable_workers}, "
            f"interval={self.worker_check_interval}s"
        )

    def _initialize_storage(self):
        """Initialize persistent storage for snapshots and quarantine."""
        # SQLite database for metadata
        self.db_path = self.storage_path / "rollback.db"

        with self.db_lock:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level="DEFERRED",
                timeout=30.0,
            )

            # Enable WAL mode for better concurrency
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
                self.conn.commit()
            except sqlite3.Error as e:
                logger.warning(f"Could not enable WAL mode: {e}")

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    state_size INTEGER,
                    action_count INTEGER,
                    checksum TEXT,
                    compressed INTEGER,
                    file_path TEXT,
                    metadata TEXT
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS quarantine (
                    quarantine_id TEXT PRIMARY KEY,
                    action TEXT,
                    reason TEXT,
                    timestamp REAL,
                    expiry REAL,
                    status TEXT,
                    reviewed INTEGER,
                    reviewer TEXT,
                    review_time REAL
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS rollback_history (
                    rollback_id TEXT PRIMARY KEY,
                    snapshot_id TEXT,
                    reason TEXT,
                    timestamp REAL,
                    success INTEGER,
                    error_message TEXT
                )
            """)

            # Create indexes
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON snapshots(timestamp)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_expiry
                ON quarantine(expiry)
            """)

            self.conn.commit()

        # Load existing snapshots
        self._load_snapshots_from_storage()

    def _execute_with_retry(self, operation, max_retries=3, initial_delay=0.1):
        """
        Execute a database operation with retry logic for handling locks.

        Args:
            operation: Callable that performs the database operation
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (doubles each time)

        Returns:
            Result of the operation, or None if all retries failed
        """
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                # Check specifically for "database is locked"
                if "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        # Don't log on every retry, only after final failure
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed, log the error
                        logger.error(
                            f"Database operation failed after {max_retries} retries: {e}"
                        )
                        return None
                else:
                    # Different error, don't retry
                    logger.error(f"Database error: {e}")
                    return None
            except Exception as e:
                # Unexpected error, don't retry
                logger.error(f"Unexpected error in database operation: {e}")
                return None

    def _load_snapshots_from_storage(self):
        """Load existing snapshots from persistent storage."""
        with self.db_lock:
            try:
                cursor = self.conn.execute(
                    "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT ?",
                    (self.max_snapshots,),
                )

                rows = cursor.fetchall()
            except sqlite3.Error as e:
                logger.error(f"Database error loading snapshots: {e}")
                return

        loaded_count = 0
        for row in rows:
            snapshot_id = row[0]
            # Load snapshot file
            file_path = Path(row[6])
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        data = f.read()
                        if row[5]:  # compressed
                            data = zlib.decompress(data)
                        snapshot_data = pickle.loads(data)  # nosec B301 - Internal data structure

                    snapshot = RollbackSnapshot(
                        snapshot_id=snapshot_id,
                        timestamp=row[1],
                        state=snapshot_data["state"],
                        action_log=snapshot_data["action_log"],
                        metadata=json.loads(row[7]) if row[7] else {},
                    )

                    with self.lock:
                        self.snapshots.append(snapshot)
                        self.snapshot_index[snapshot_id] = snapshot

                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Failed to load snapshot {snapshot_id}: {e}")

        logger.info(f"Loaded {loaded_count} existing snapshots from storage")

    def create_snapshot(
        self,
        state: Dict[str, Any],
        action_log: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a snapshot for potential rollback.

        Args:
            state: Current system state to snapshot
            action_log: Log of actions leading to this state
            metadata: Additional snapshot metadata

        Returns:
            Snapshot ID
        """
        if self._shutdown:
            raise RuntimeError("RollbackManager is shut down")

        snapshot_id = str(uuid.uuid4())
        timestamp = time.time()

        # Deep copy to prevent reference issues
        state_copy = copy.deepcopy(state)
        action_log_copy = copy.deepcopy(action_log)

        # Create snapshot
        snapshot = RollbackSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            state=state_copy,
            action_log=action_log_copy,
            metadata=metadata or {"creation_reason": "checkpoint"},
        )

        # Verify integrity if enabled
        if self.verify_integrity and not snapshot.verify_integrity():
            logger.error(f"Snapshot {snapshot_id} failed integrity check")
            raise ValueError("Snapshot integrity verification failed")

        # CRITICAL: Hold lock through entire operation to prevent race conditions
        with self.lock:
            # Persist BEFORE adding to memory
            try:
                self._persist_snapshot(snapshot)
            except Exception as e:
                logger.error(f"Failed to persist snapshot {snapshot_id}: {e}")
                raise

            # Only add to memory after successful persistence
            self.snapshots.append(snapshot)
            self.snapshot_index[snapshot_id] = snapshot

            # Update metrics
            self.metrics["total_snapshots"] += 1
            self.metrics["storage_bytes"] += self._calculate_snapshot_size(snapshot)

            # Clean old snapshots if needed
            if self.auto_cleanup:
                self._cleanup_old_snapshots()

        logger.info(
            f"Created snapshot {snapshot_id} with {len(action_log_copy)} actions"
        )
        return snapshot_id

    def _persist_snapshot(self, snapshot: RollbackSnapshot):
        """Persist snapshot to disk storage."""
        # Prepare data for storage
        snapshot_data = {"state": snapshot.state, "action_log": snapshot.action_log}

        # Serialize
        serialized = pickle.dumps(snapshot_data)

        # Compress if enabled
        compressed = False
        if self.compress_snapshots:
            compressed_data = zlib.compress(serialized, level=6)
            if (
                len(compressed_data) < len(serialized) * 0.9
            ):  # Only use if >10% reduction
                serialized = compressed_data
                compressed = True

        # Write to file
        file_path = self.storage_path / f"snapshot_{snapshot.snapshot_id}.dat"
        try:
            with open(file_path, "wb") as f:
                f.write(serialized)
        except IOError as e:
            logger.error(f"Failed to write snapshot file: {e}")
            raise

        # Store metadata in database
        with self.db_lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO snapshots (snapshot_id, timestamp, state_size,
                                          action_count, checksum, compressed,
                                          file_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        snapshot.snapshot_id,
                        snapshot.timestamp,
                        len(json.dumps(snapshot.state, default=str)),
                        len(snapshot.action_log),
                        snapshot.checksum,
                        int(compressed),
                        str(file_path),
                        json.dumps(snapshot.metadata),
                    ),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error persisting snapshot: {e}")
                # Try to clean up file
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                raise

    def rollback(
        self, snapshot_id: Optional[str] = None, reason: str = "safety_violation"
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback to a previous snapshot.

        Args:
            snapshot_id: Specific snapshot ID to rollback to (None = most recent)
            reason: Reason for rollback

        Returns:
            Dictionary with restored state and metadata, or None if failed
        """
        rollback_id = str(uuid.uuid4())

        with self.lock:
            # Find snapshot
            if snapshot_id:
                snapshot = self.snapshot_index.get(snapshot_id)
            else:
                # Use most recent snapshot
                snapshot = self.snapshots[-1] if self.snapshots else None

            if not snapshot:
                logger.error(
                    f"No snapshot found for rollback (requested: {snapshot_id})"
                )
                self._record_rollback(
                    rollback_id, snapshot_id, reason, False, "Snapshot not found"
                )
                # CRITICAL FIX: Update metrics before returning
                self.metrics["total_rollbacks"] += 1
                self.metrics["failed_rollbacks"] += 1
                return None

            # Verify integrity before rollback
            if self.verify_integrity and not snapshot.verify_integrity():
                logger.error(f"Snapshot {snapshot.snapshot_id} failed integrity check")
                self._record_rollback(
                    rollback_id,
                    snapshot.snapshot_id,
                    reason,
                    False,
                    "Integrity check failed",
                )
                # CRITICAL FIX: Update metrics before returning
                self.metrics["total_rollbacks"] += 1
                self.metrics["failed_rollbacks"] += 1
                return None

            try:
                # Create rollback result
                rollback_result = {
                    "state": copy.deepcopy(snapshot.state),
                    "action_log": copy.deepcopy(snapshot.action_log),
                    "rollback_metadata": {
                        "rollback_id": rollback_id,
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_timestamp": snapshot.timestamp,
                        "reason": reason,
                        "timestamp": time.time(),
                    },
                }

                # Record successful rollback
                self._record_rollback(
                    rollback_id, snapshot.snapshot_id, reason, True, None
                )

                # Update metrics
                self.metrics["total_rollbacks"] += 1
                self.metrics["successful_rollbacks"] += 1

                logger.warning(
                    f"Successfully rolled back to snapshot {snapshot.snapshot_id} "
                    f"(reason: {reason})"
                )

                return rollback_result

            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                self._record_rollback(
                    rollback_id, snapshot.snapshot_id, reason, False, str(e)
                )
                self.metrics["total_rollbacks"] += 1
                self.metrics["failed_rollbacks"] += 1
                return None

    def _record_rollback(
        self,
        rollback_id: str,
        snapshot_id: Optional[str],
        reason: str,
        success: bool,
        error_message: Optional[str],
    ):
        """Record rollback attempt in history."""
        # Add to memory
        with self.lock:
            self.rollback_history.append(
                {
                    "rollback_id": rollback_id,
                    "timestamp": time.time(),
                    "snapshot_id": snapshot_id,
                    "reason": reason,
                    "success": success,
                    "error_message": error_message,
                }
            )

        # Persist to database
        with self.db_lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO rollback_history (rollback_id, snapshot_id, reason,
                                                timestamp, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        rollback_id,
                        snapshot_id,
                        reason,
                        time.time(),
                        int(success),
                        error_message,
                    ),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error recording rollback: {e}")

    def quarantine_action(
        self, action: Dict[str, Any], reason: str, duration_seconds: float = 3600
    ) -> str:
        """
        Quarantine an action for review.

        Args:
            action: Action to quarantine
            reason: Reason for quarantine
            duration_seconds: Quarantine duration in seconds

        Returns:
            Quarantine ID
        """
        quarantine_id = str(uuid.uuid4())
        timestamp = time.time()
        expiry = timestamp + duration_seconds

        # Convert ActionType enums to strings for JSON serialization
        action_serializable = self._make_json_serializable(copy.deepcopy(action))

        with self.lock:
            # Store in memory
            self.quarantine[quarantine_id] = {
                "action": action_serializable,
                "reason": reason,
                "timestamp": timestamp,
                "expiry": expiry,
                "status": "quarantined",
                "reviewed": False,
                "reviewer": None,
                "review_time": None,
            }

            # Update metrics
            self.metrics["quarantined_actions"] += 1

        # Persist to database
        with self.db_lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO quarantine (quarantine_id, action, reason, timestamp,
                                          expiry, status, reviewed, reviewer, review_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        quarantine_id,
                        json.dumps(action_serializable, default=str),
                        reason,
                        timestamp,
                        expiry,
                        "quarantined",
                        0,
                        None,
                        None,
                    ),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error quarantining action: {e}")

        logger.warning(
            f"Action quarantined: {quarantine_id} for reason: {reason} "
            f"(duration: {duration_seconds}s)"
        )

        # Send notification
        self._send_quarantine_notification(quarantine_id, action_serializable, reason)

        return quarantine_id

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (ActionType,)):
            return obj.value
        else:
            return obj

    def review_quarantine(
        self,
        quarantine_id: str,
        approved: bool,
        reviewer: str = "system",
        notes: Optional[str] = None,
    ) -> bool:
        """
        Review a quarantined action.

        Args:
            quarantine_id: ID of quarantined action
            approved: Whether action is approved
            reviewer: Identifier of reviewer
            notes: Review notes

        Returns:
            True if review successful, False otherwise
        """
        with self.lock:
            if quarantine_id not in self.quarantine:
                logger.warning(f"Quarantine ID {quarantine_id} not found")
                return False

            review_time = time.time()
            status = "approved" if approved else "rejected"

            # Update memory
            self.quarantine[quarantine_id]["reviewed"] = True
            self.quarantine[quarantine_id]["reviewer"] = reviewer
            self.quarantine[quarantine_id]["review_time"] = review_time
            self.quarantine[quarantine_id]["status"] = status

            if notes:
                self.quarantine[quarantine_id]["notes"] = notes

        # Update database
        with self.db_lock:
            try:
                self.conn.execute(
                    """
                    UPDATE quarantine
                    SET status = ?, reviewed = 1, reviewer = ?, review_time = ?
                    WHERE quarantine_id = ?
                """,
                    (status, reviewer, review_time, quarantine_id),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error updating quarantine: {e}")

        logger.info(f"Quarantined action {quarantine_id} {status} by {reviewer}")

        return True

    def get_quarantine_item(self, quarantine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a quarantined action.

        Args:
            quarantine_id: ID of quarantined action

        Returns:
            Quarantine details or None if not found
        """
        with self.lock:
            item = self.quarantine.get(quarantine_id)
            return copy.deepcopy(item) if item else None

    def cleanup_expired_quarantine(self):
        """Remove expired quarantine entries."""
        # Early exit if shutdown requested
        if self._shutdown:
            return

        current_time = time.time()

        with self.lock:
            expired = []
            for qid, q in self.quarantine.items():
                if q["expiry"] < current_time:
                    expired.append(qid)

            for qid in expired:
                del self.quarantine[qid]
                logger.info(f"Removed expired quarantine entry: {qid}")

        # Clean from database with retry logic
        def cleanup_operation():
            with self.db_lock:
                self.conn.execute(
                    "DELETE FROM quarantine WHERE expiry < ?", (current_time,)
                )
                self.conn.commit()
                return True

        result = self._execute_with_retry(cleanup_operation)
        if result is None:
            logger.error(
                "Failed to clean expired quarantine from database after retries"
            )

    def _cleanup_old_snapshots(self):
        """Clean up old snapshots based on retention policy."""
        # Early exit if shutdown requested
        if self._shutdown:
            return

        cutoff_time = time.time() - (self.snapshot_retention_days * 86400)

        # Find old snapshots with retry logic
        def find_old_snapshots():
            with self.db_lock:
                cursor = self.conn.execute(
                    "SELECT snapshot_id, file_path FROM snapshots WHERE timestamp < ?",
                    (cutoff_time,),
                )
                return cursor.fetchall()

        old_snapshots = self._execute_with_retry(find_old_snapshots)
        if old_snapshots is None:
            logger.error("Failed to find old snapshots from database after retries")
            return

        for row in old_snapshots:
            # Check shutdown flag before processing each snapshot
            if self._shutdown:
                return

            snapshot_id, file_path = row

            # Remove from memory
            with self.lock:
                if snapshot_id in self.snapshot_index:
                    snapshot = self.snapshot_index[snapshot_id]
                    if snapshot in self.snapshots:
                        self.snapshots.remove(snapshot)
                    del self.snapshot_index[snapshot_id]

            # Delete file
            file_path = Path(file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Error deleting snapshot file {file_path}: {e}")

            logger.info(f"Cleaned up old snapshot: {snapshot_id}")

        # Clean from database with retry logic
        def cleanup_operation():
            with self.db_lock:
                self.conn.execute(
                    "DELETE FROM snapshots WHERE timestamp < ?", (cutoff_time,)
                )
                self.conn.commit()
                return True

        result = self._execute_with_retry(cleanup_operation)
        if result is None:
            logger.error("Failed to delete old snapshots from database after retries")

    def _start_cleanup_thread(self):
        """Start background thread for cleanup tasks."""

        def cleanup_worker():
            while not self._shutdown:
                try:
                    # CRITICAL FIX: Use configurable timeout for faster shutdown in tests
                    # Default is 10s, can be set to 1-5s for test environments
                    if self._shutdown_event.wait(timeout=self.worker_check_interval):
                        break  # Shutdown requested

                    if self._shutdown:
                        break

                    self.cleanup_expired_quarantine()
                    if self.auto_cleanup and not self._shutdown:
                        self._cleanup_old_snapshots()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")

        self.cleanup_thread = threading.Thread(
            target=cleanup_worker, daemon=True, name="RollbackCleanup"
        )
        self.cleanup_thread.start()

    def _calculate_snapshot_size(self, snapshot: RollbackSnapshot) -> int:
        """Calculate approximate size of snapshot in bytes."""
        try:
            # Serialize to estimate size
            data = pickle.dumps(
                {"state": snapshot.state, "action_log": snapshot.action_log}
            )
            return len(data)
        except Exception:
            return 0

    def _send_quarantine_notification(
        self, quarantine_id: str, action: Dict[str, Any], reason: str
    ):
        """Send notification about quarantined action."""
        notification = {
            "type": "quarantine_alert",
            "quarantine_id": quarantine_id,
            "action_type": str(action.get("type", "unknown")),
            "reason": reason,
            "timestamp": time.time(),
            "severity": "high" if "safety" in reason.lower() else "medium",
        }

        # Log critical notification
        logger.critical(f"QUARANTINE NOTIFICATION: {json.dumps(notification)}")

        # In production, this would integrate with notification systems
        # (email, Slack, PagerDuty, etc.)

    def get_snapshot_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent snapshot history.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot summaries
        """
        with self.lock:
            history = []
            for snapshot in list(self.snapshots)[-limit:]:
                history.append(snapshot.to_dict())
            return history

    def get_rollback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent rollback history.

        Args:
            limit: Maximum number of rollbacks to return

        Returns:
            List of rollback records
        """
        with self.lock:
            return list(self.rollback_history)[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get rollback manager metrics."""
        with self.lock:
            return {
                **self.metrics,
                "current_snapshots": len(self.snapshots),
                "quarantine_size": len(self.quarantine),
                "rollback_success_rate": (
                    self.metrics["successful_rollbacks"]
                    / max(1, self.metrics["total_rollbacks"])
                ),
            }

    def export_snapshot(self, snapshot_id: str, export_path: str) -> bool:
        """
        Export a snapshot to a file.

        Args:
            snapshot_id: ID of snapshot to export
            export_path: Path to export file

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            snapshot = self.snapshot_index.get(snapshot_id)
            if not snapshot:
                logger.error(f"Snapshot {snapshot_id} not found for export")
                return False

            try:
                export_data = {
                    "snapshot_id": snapshot.snapshot_id,
                    "timestamp": snapshot.timestamp,
                    "state": snapshot.state,
                    "action_log": snapshot.action_log,
                    "metadata": snapshot.metadata,
                    "checksum": snapshot.checksum,
                }

                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, default=str)

                logger.info(f"Exported snapshot {snapshot_id} to {export_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to export snapshot: {e}")
                return False

    def import_snapshot(self, import_path: str) -> Optional[str]:
        """
        Import a snapshot from a file.

        Args:
            import_path: Path to import file

        Returns:
            Snapshot ID if successful, None otherwise
        """
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            # Create new snapshot from imported data
            snapshot_id = self.create_snapshot(
                state=import_data["state"],
                action_log=import_data["action_log"],
                metadata={
                    **import_data.get("metadata", {}),
                    "imported": True,
                    "import_time": time.time(),
                    "original_id": import_data["snapshot_id"],
                },
            )

            logger.info(f"Imported snapshot from {import_path} as {snapshot_id}")
            return snapshot_id

        except Exception as e:
            logger.error(f"Failed to import snapshot: {e}")
            return None

    def shutdown(self):
        """Shutdown rollback manager and cleanup resources."""
        if self._shutdown:
            return

        # Disable logging error output during shutdown
        logging.raiseExceptions = False

        safe_log(logger.info, "Shutting down RollbackManager...")
        self._shutdown = True

        # CRITICAL FIX: Signal the event to wake up sleeping thread
        self._shutdown_event.set()

        # Wait for cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2.0)

        # Close database connection
        with self.db_lock:
            if self.conn:
                try:
                    self.conn.close()
                    self.conn = None
                except Exception as e:
                    logger.error(f"Error closing database: {e}")

        safe_log(logger.info, "RollbackManager shutdown complete")


# ============================================================
# AUDIT LOGGER
# ============================================================


class AuditLogger:
    """
    Comprehensive audit logging system with redaction, rotation, and search capabilities.
    Provides tamper-evident logging with cryptographic verification.
    """

    # Whitelist of allowed filter fields for SQL injection protection
    ALLOWED_FILTER_FIELDS = {
        "entry_type",
        "severity",
        "action_type",
        "safe",
        "timestamp",
        "entry_id",
    }

    # Whitelist of allowed sort fields
    ALLOWED_SORT_FIELDS = {"timestamp", "severity", "entry_type"}

    def __init__(
        self, log_path: str = "safety_audit", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audit logger.

        Args:
            log_path: Base path for audit logs
            config: Additional configuration
        """
        self.config = config or {}
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Shutdown flag
        self._shutdown = False

        # CRITICAL FIX: Add shutdown event for interruptible sleep
        self._shutdown_event = threading.Event()

        # Configuration
        self.redact_sensitive = self.config.get("redact_sensitive", True)
        self.rotation_days = self.config.get("rotation_days", 30)
        self.compress_old_logs = self.config.get("compress_old_logs", True)
        self.enable_signing = self.config.get("enable_signing", True)
        self.max_log_size_mb = self.config.get("max_log_size_mb", 100)

        # CRITICAL FIX: Configurable worker check interval (default 10s for faster shutdown)
        # Set to 1-5 seconds in test environments, longer in production
        self.worker_check_interval = self.config.get("worker_check_interval", 10.0)

        # CRITICAL FIX: Initialize locks BEFORE any operations that use them
        self.lock = threading.RLock()
        self.db_lock = threading.RLock()

        # Current log file
        self.current_log_file = self._get_log_file()

        # Memory-aware buffer with 10MB limit
        self.log_buffer = MemoryBoundedDeque(max_size_mb=10)

        # Add memory monitoring
        self.memory_check_interval = 100  # Check every 100 entries
        self.entries_since_check = 0

        self.redaction_patterns = self._initialize_redaction_patterns()

        # Hash chain for tamper detection
        self.last_hash = self._get_last_hash()

        # SQLite database for indexing
        self.db_path = self.log_path / "audit_index.db"
        self.conn = None

        # NOW it's safe to initialize database (db_lock exists)
        self._initialize_database()

        # Metrics
        self.metrics = {
            "total_entries": 0,
            "redactions": 0,
            "rotations": 0,
            "searches": 0,
            "verifications": 0,
        }

        # Rotation thread reference
        self.rotation_thread = None

        # Start rotation thread
        self._start_rotation_thread()

        # Register cleanup
        # atexit.register(self.shutdown) # Removed for test suite compatibility

        logger.info(f"AuditLogger initialized with path: {self.log_path}")

    def _initialize_database(self):
        """Initialize SQLite database for log indexing."""
        with self.db_lock:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level="DEFERRED",
                timeout=30.0,
            )

            # Enable WAL mode for better concurrency
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
                self.conn.commit()
            except sqlite3.Error as e:
                logger.warning(f"Could not enable WAL mode: {e}")

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    entry_type TEXT,
                    severity TEXT,
                    action_type TEXT,
                    safe INTEGER,
                    violations TEXT,
                    file_path TEXT,
                    line_number INTEGER,
                    hash TEXT
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_entries(timestamp)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON audit_entries(entry_type)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_severity ON audit_entries(severity)
            """)

            self.conn.commit()

    def _get_log_file(self) -> Path:
        """Get current log file path based on date."""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_path / f"safety_audit_{date_str}.jsonl"

    def _get_last_hash(self) -> str:
        """Get the last hash from existing log for chain continuity."""
        if self.current_log_file.exists():
            try:
                with open(self.current_log_file, "r", encoding="utf-8") as f:
                    # Read last line
                    last_line = None
                    for line in f:
                        last_line = line
                    if last_line:
                        entry = json.loads(last_line)
                        return entry.get("hash", "")
            except Exception:
                pass
        return hashlib.sha256(b"genesis").hexdigest()

    def _initialize_redaction_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for redacting sensitive information."""
        patterns = [
            {
                "name": "ssn",
                "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                "replacement": "[SSN_REDACTED]",
            },
            {
                "name": "email",
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "replacement": "[EMAIL_REDACTED]",
            },
            {
                "name": "credit_card",
                "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                "replacement": "[CC_REDACTED]",
            },
            {
                "name": "phone",
                "pattern": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
                "replacement": "[PHONE_REDACTED]",
            },
            {
                "name": "ip_address",
                "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                "replacement": "[IP_REDACTED]",
            },
            {
                "name": "api_key",
                "pattern": r"\b[A-Za-z0-9]{32,}\b",
                "replacement": "[API_KEY_REDACTED]",
            },
            {
                "name": "password",
                "pattern": r"(?i)(password|passwd|pwd)[\s]*[=:]\s*[^\s]+",
                "replacement": "password=[PASSWORD_REDACTED]",
            },
        ]

        # Add custom patterns from config
        custom_patterns = self.config.get("custom_redaction_patterns", [])
        patterns.extend(custom_patterns)

        return patterns

    def log_safety_decision(
        self, decision: Dict[str, Any], report: SafetyReport, redact: bool = None
    ) -> str:
        """
        Log a safety decision with full traceability.

        Args:
            decision: Decision details
            report: Safety report
            redact: Whether to redact sensitive data (None = use default)

        Returns:
            Entry ID
        """
        if redact is None:
            redact = self.redact_sensitive

        entry_id = str(uuid.uuid4())
        timestamp = time.time()

        # Prepare log entry
        log_entry = {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "iso_timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "entry_type": "safety_decision",
            "decision": self._redact_sensitive(decision) if redact else decision,
            "safety_report": report.to_audit_log(),
            "severity": self._determine_severity(report),
            "process_info": {
                "thread_id": threading.current_thread().ident,
                "thread_name": threading.current_thread().name,
                "process_id": os.getpid(),
            },
        }

        # Add hash chain
        if self.enable_signing:
            with self.lock:
                log_entry["previous_hash"] = self.last_hash
                log_entry["hash"] = self._calculate_hash(log_entry)
                self.last_hash = log_entry["hash"]

        with self.lock:
            # Add to buffer
            self.log_buffer.append(log_entry)
            self.entries_since_check += 1
            self.metrics["total_entries"] += 1

            # Periodic flush (batch write)
            if self.entries_since_check >= 10:  # Flush every 10 entries
                self._flush_buffer_batch()
                self.entries_since_check = 0
                self._check_rotation()

        return entry_id

    def _flush_buffer_batch(self):
        """Flush log buffer in batch for better performance."""
        if not self.log_buffer:
            return

        with self.log_buffer.lock:
            entries_to_write = list(self.log_buffer.deque)
            self.log_buffer.clear()

        if not entries_to_write:
            return

        # Write all entries at once
        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                for entry in entries_to_write:
                    f.write(json.dumps(entry, default=str) + "\n")

            # Batch insert into database
            with self.db_lock:
                try:
                    for entry in entries_to_write:
                        self._index_entry(
                            entry, -1
                        )  # Line number not critical for index
                    self.conn.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error indexing entries: {e}")
        except Exception as e:
            logger.error(f"Failed to flush log buffer: {e}")

    def log_event(
        self, event_type: str, event_data: Dict[str, Any], severity: str = "info"
    ) -> str:
        """
        Log a general audit event.

        Args:
            event_type: Type of event
            event_data: Event data
            severity: Event severity (debug, info, warning, error, critical)

        Returns:
            Entry ID
        """
        entry_id = str(uuid.uuid4())
        timestamp = time.time()

        log_entry = {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "iso_timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "entry_type": event_type,
            "severity": severity,
            "event_data": self._redact_sensitive(event_data)
            if self.redact_sensitive
            else event_data,
        }

        # Add hash chain
        if self.enable_signing:
            with self.lock:
                log_entry["previous_hash"] = self.last_hash
                log_entry["hash"] = self._calculate_hash(log_entry)
                self.last_hash = log_entry["hash"]

        with self.lock:
            self.log_buffer.append(log_entry)
            self.entries_since_check += 1
            self.metrics["total_entries"] += 1

            # Periodic flush
            if self.entries_since_check >= 10:
                self._flush_buffer_batch()
                self.entries_since_check = 0
                self._check_rotation()

        return entry_id

    def log_redaction(self, original: str, redacted: str, reason: str):
        """
        Log a redaction action for transparency.

        Args:
            original: Original text
            redacted: Redacted text
            reason: Reason for redaction
        """
        # Hash original for reference without storing it
        original_hash = hashlib.sha256(original.encode()).hexdigest()

        redaction_entry = {
            "entry_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "entry_type": "redaction",
            "severity": "info",
            "original_hash": original_hash,
            "redacted_value": redacted,
            "reason": reason,
        }

        with self.lock:
            self._write_log_entry(redaction_entry)
            self.metrics["redactions"] += 1

    def _redact_sensitive(self, data: Any) -> Any:
        """Redact sensitive information from data."""
        if isinstance(data, str):
            redacted = data
            for pattern_info in self.redaction_patterns:
                pattern = pattern_info["pattern"]
                replacement = pattern_info["replacement"]

                # Find matches
                try:
                    matches = re.findall(pattern, redacted)
                    if matches:
                        for match in matches:
                            # Log the redaction (avoid recursion)
                            pass  # Simplified to prevent infinite recursion

                        # Apply redaction
                        redacted = re.sub(pattern, replacement, redacted)
                except Exception as e:
                    logger.error(
                        f"Error applying redaction pattern {pattern_info['name']}: {e}"
                    )

            return redacted

        elif isinstance(data, dict):
            return {k: self._redact_sensitive(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [self._redact_sensitive(item) for item in data]

        return data

    def _calculate_hash(self, entry: Dict[str, Any]) -> str:
        """Calculate cryptographic hash for log entry."""
        # Remove hash field for calculation
        entry_copy = {k: v for k, v in entry.items() if k != "hash"}
        entry_str = json.dumps(entry_copy, sort_keys=True, default=str)
        return hashlib.sha256(entry_str.encode()).hexdigest()

    def _write_log_entry(self, entry: Dict[str, Any]) -> int:
        """Write log entry to file and return line number."""
        try:
            # Get current file size
            if self.current_log_file.exists():
                with open(self.current_log_file, "r", encoding="utf-8") as f:
                    line_number = sum(1 for _ in f)
            else:
                line_number = 0

            # Write entry
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            return line_number

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            return -1

    def _index_entry(self, entry: Dict[str, Any], line_number: int):
        """Index log entry in database for fast searching (must be called with db_lock held)."""
        try:
            # Extract key fields
            entry_id = entry.get("entry_id", "")
            timestamp = entry.get("timestamp", 0)
            entry_type = entry.get("entry_type", "")
            severity = entry.get("severity", "info")

            # Extract from nested structures
            action_type = ""
            safe = None
            violations = ""

            if "decision" in entry:
                action_type = str(entry["decision"].get("type", ""))

            if "safety_report" in entry:
                report = entry["safety_report"]
                safe = int(report.get("safe", True))
                violations = ",".join(str(v) for v in report.get("violations", []))

            # Insert into database
            self.conn.execute(
                """
                INSERT OR REPLACE INTO audit_entries
                (entry_id, timestamp, entry_type, severity, action_type,
                 safe, violations, file_path, line_number, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_id,
                    timestamp,
                    entry_type,
                    severity,
                    action_type,
                    safe,
                    violations,
                    str(self.current_log_file),
                    line_number,
                    entry.get("hash", ""),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to index audit entry: {e}")

    def _determine_severity(self, report: SafetyReport) -> str:
        """Determine severity level from safety report."""
        if not report.safe:
            if SafetyViolationType.ADVERSARIAL in report.violations:
                return "critical"
            elif SafetyViolationType.COMPLIANCE in report.violations:
                return "error"
            else:
                return "warning"
        return "info"

    def _check_rotation(self):
        """Check if log rotation is needed."""
        # Early exit if shutdown requested
        if self._shutdown:
            return

        # Check file size
        if self.current_log_file.exists():
            size_mb = self.current_log_file.stat().st_size / (1024 * 1024)

            if size_mb > self.max_log_size_mb:
                self._rotate_log()
                return

        # Check date change
        expected_file = self._get_log_file()
        if expected_file != self.current_log_file:
            self._rotate_log()

    def _rotate_log(self):
        """Rotate current log file."""
        # Early exit if shutdown requested
        if self._shutdown:
            return

        logger.info(f"Rotating log file: {self.current_log_file}")

        # Compress old log if enabled
        if self.compress_old_logs and self.current_log_file.exists():
            self._compress_log(self.current_log_file)

        # Update current log file
        self.current_log_file = self._get_log_file()

        # Reset hash chain for new file
        self.last_hash = hashlib.sha256(b"genesis").hexdigest()

        # Update metrics
        with self.lock:
            self.metrics["rotations"] += 1

    def _compress_log(self, log_file: Path):
        """Compress a log file."""
        import gzip

        compressed_file = log_file.with_suffix(".jsonl.gz")

        try:
            with open(log_file, "rb") as f_in:
                with gzip.open(compressed_file, "wb", compresslevel=9) as f_out:
                    f_out.write(f_in.read())

            # Remove original after successful compression
            log_file.unlink()

            logger.info(f"Compressed log file: {log_file} -> {compressed_file}")

        except Exception as e:
            logger.error(f"Failed to compress log file: {e}")

    def _start_rotation_thread(self):
        """Start background thread for log rotation."""

        def rotation_worker():
            while not self._shutdown:
                try:
                    # CRITICAL FIX: Use configurable timeout for faster shutdown in tests
                    # Default is 10s, can be set to 1-5s for test environments
                    if self._shutdown_event.wait(timeout=self.worker_check_interval):
                        break  # Shutdown requested

                    if self._shutdown:
                        break

                    self._check_rotation()
                    self._cleanup_old_logs()
                except Exception as e:
                    logger.error(f"Rotation thread error: {e}")

        self.rotation_thread = threading.Thread(
            target=rotation_worker, daemon=True, name="AuditRotation"
        )
        self.rotation_thread.start()

    def _cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        # Early exit if shutdown requested
        if self._shutdown:
            return

        cutoff_date = datetime.now() - timedelta(days=self.rotation_days)

        for log_file in self.log_path.glob("safety_audit_*.jsonl*"):
            # Check shutdown before processing each file
            if self._shutdown:
                return

            # Parse date from filename
            try:
                date_str = log_file.stem.split("_")[2].split(".")[0]
                file_date = datetime.strptime(date_str, "%Y%m%d")

                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")

            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")

    def query_logs(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        sort_by: str = "timestamp",
        sort_order: str = "DESC",
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            filters: Additional filters
            limit: Maximum results (1-10000)
            sort_by: Field to sort by
            sort_order: Sort order (ASC or DESC)

        Returns:
            List of matching log entries
        """
        # CRITICAL: Validate inputs to prevent SQL injection
        if limit < 1 or limit > 10000:
            raise ValueError(f"Limit must be 1-10000, got {limit}")

        if sort_by not in self.ALLOWED_SORT_FIELDS:
            raise ValueError(
                f"Invalid sort field: {sort_by}. Allowed: {self.ALLOWED_SORT_FIELDS}"
            )

        if sort_order not in ("ASC", "DESC"):
            raise ValueError(f"Invalid sort order: {sort_order}. Must be ASC or DESC")

        with self.lock:
            self.metrics["searches"] += 1

            # Build query with parameterized statements
            query = "SELECT * FROM audit_entries WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if filters:
                for key, value in filters.items():
                    # CRITICAL: Validate field name against whitelist
                    if key not in self.ALLOWED_FILTER_FIELDS:
                        logger.warning(f"Ignoring invalid filter field: {key}")
                        continue

                    query += f" AND {key} = ?"  # Now safe since key is validated
                    params.append(value)

            # Use validated sort field (safe from injection)
            query += f" ORDER BY {sort_by} {sort_order} LIMIT ?"
            params.append(limit)

            # Execute query
            with self.db_lock:
                try:
                    cursor = self.conn.execute(query, params)
                    rows = cursor.fetchall()
                except sqlite3.Error as e:
                    logger.error(f"Database error querying logs: {e}")
                    return []

            # Fetch entries from files
            results = []
            for row in rows:
                (
                    entry_id,
                    timestamp,
                    entry_type,
                    severity,
                    action_type,
                    safe,
                    violations,
                    file_path,
                    line_number,
                    hash_val,
                ) = row

                # Read actual entry from file
                try:
                    file_path = Path(file_path)
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                if i == line_number or line_number == -1:
                                    # Try to find by entry_id if line_number is -1
                                    try:
                                        entry = json.loads(line)
                                        if line_number == -1:
                                            if entry.get("entry_id") == entry_id:
                                                results.append(entry)
                                                break
                                        else:
                                            results.append(entry)
                                            break
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    logger.error(f"Error reading log entry: {e}")

            return results

    def verify_integrity(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Verify integrity of audit logs using hash chain.

        Args:
            start_time: Start timestamp for verification
            end_time: End timestamp for verification

        Returns:
            Verification results
        """
        if not self.enable_signing:
            return {"verified": False, "error": "Signing not enabled"}

        with self.lock:
            self.metrics["verifications"] += 1

            # Get entries to verify
            entries = self.query_logs(
                start_time, end_time, limit=10000, sort_order="ASC"
            )

            if not entries:
                return {"verified": True, "entries_checked": 0}

            # Verify hash chain
            previous_hash = None
            broken_at = None

            for i, entry in enumerate(entries):
                if "hash" not in entry:
                    continue

                # Check hash
                calculated_hash = self._calculate_hash(entry)
                if calculated_hash != entry["hash"]:
                    broken_at = i
                    break

                # Check chain (allow for genesis or file rotation breaks)
                if previous_hash and entry.get("previous_hash") != previous_hash:
                    # Check if this is a rotation boundary (genesis hash)
                    genesis_hash = hashlib.sha256(b"genesis").hexdigest()
                    if entry.get("previous_hash") != genesis_hash:
                        broken_at = i
                        break

                previous_hash = entry["hash"]

            return {
                "verified": broken_at is None,
                "entries_checked": len(entries),
                "broken_at": broken_at,
                "error": f"Chain broken at entry {broken_at}"
                if broken_at is not None
                else None,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get audit logger metrics."""
        with self.lock:
            return {
                **self.metrics,
                "buffer_size": len(self.log_buffer),
                "current_log_file": str(self.current_log_file),
                "log_size_mb": (
                    self.current_log_file.stat().st_size / (1024 * 1024)
                    if self.current_log_file.exists()
                    else 0
                ),
            }

    def export_logs(
        self,
        export_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        format: str = "json",
    ) -> bool:
        """
        Export audit logs to a file.

        Args:
            export_path: Path for export file
            start_time: Start timestamp
            end_time: End timestamp
            format: Export format (json, csv)

        Returns:
            True if successful
        """
        try:
            # CRITICAL FIX: Use pagination for large exports instead of exceeding limit
            all_entries = []
            offset = 0
            batch_size = 10000

            while True:
                # Query in batches
                batch_entries = self.query_logs(
                    start_time=start_time,
                    end_time=end_time,
                    limit=batch_size,
                    sort_order="ASC",
                )

                if not batch_entries:
                    break

                all_entries.extend(batch_entries)

                # If we got fewer than batch_size, we're done
                if len(batch_entries) < batch_size:
                    break

                # Update start_time for next batch to avoid duplicates
                if batch_entries:
                    start_time = batch_entries[-1]["timestamp"] + 0.000001

                offset += batch_size

                # Safety limit to prevent infinite loops
                if len(all_entries) >= 1000000:  # 1M entries max
                    logger.warning("Export reached 1M entry limit")
                    break

            entries = all_entries

            if format == "json":
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2, default=str)

            elif format == "csv":
                import csv

                if entries:
                    with open(export_path, "w", newline="", encoding="utf-8") as f:
                        # Use first entry to get field names
                        fieldnames = list(entries[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for entry in entries:
                            # Flatten nested structures
                            flat_entry = {}
                            for key, value in entry.items():
                                if isinstance(value, (dict, list)):
                                    flat_entry[key] = json.dumps(value)
                                else:
                                    flat_entry[key] = value
                            writer.writerow(flat_entry)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Exported {len(entries)} audit entries to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export audit logs: {e}")
            return False

    def get_summary(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics of audit logs.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Summary statistics
        """
        entries = self.query_logs(start_time, end_time, limit=10000)

        summary = {
            "total_entries": len(entries),
            "time_range": {
                "start": start_time or (entries[0]["timestamp"] if entries else None),
                "end": end_time or (entries[-1]["timestamp"] if entries else None),
            },
            "entry_types": {},
            "severities": {},
            "safety_stats": {"safe_decisions": 0, "unsafe_decisions": 0},
            "violations": defaultdict(int),
        }

        for entry in entries:
            # Count entry types
            entry_type = entry.get("entry_type", "unknown")
            summary["entry_types"][entry_type] = (
                summary["entry_types"].get(entry_type, 0) + 1
            )

            # Count severities
            severity = entry.get("severity", "unknown")
            summary["severities"][severity] = summary["severities"].get(severity, 0) + 1

            # Safety statistics
            if "safety_report" in entry:
                report = entry["safety_report"]
                if report.get("safe", True):
                    summary["safety_stats"]["safe_decisions"] += 1
                else:
                    summary["safety_stats"]["unsafe_decisions"] += 1

                # Count violations
                for violation in report.get("violations", []):
                    summary["violations"][str(violation)] += 1

        # Convert defaultdict to regular dict for JSON serialization
        summary["violations"] = dict(summary["violations"])

        return summary

    def shutdown(self):
        """Shutdown audit logger and cleanup resources."""
        if self._shutdown:
            return

        # Disable logging error output during shutdown
        logging.raiseExceptions = False

        safe_log(logger.info, "Shutting down AuditLogger...")
        self._shutdown = True

        # CRITICAL FIX: Signal the event to wake up sleeping thread
        self._shutdown_event.set()

        # Flush remaining buffer
        self._flush_buffer_batch()

        # Wait for rotation thread
        if self.rotation_thread and self.rotation_thread.is_alive():
            self.rotation_thread.join(timeout=2.0)

        # Close database connection
        with self.db_lock:
            if self.conn:
                try:
                    self.conn.close()
                    self.conn = None
                except Exception as e:
                    logger.error(f"Error closing database: {e}")

        safe_log(logger.info, "AuditLogger shutdown complete")
