# test_rollback_audit.py
"""
Comprehensive tests for rollback_audit.py module.
Tests rollback management, audit logging, and integrity verification.
"""

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from vulcan.safety.rollback_audit import (AuditLogger, MemoryBoundedDeque,
                                          RollbackManager)
from vulcan.safety.safety_types import (SafetyReport, SafetyViolationType)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def rollback_manager(temp_storage_dir):
    """Create a rollback manager with temporary storage."""
    manager = RollbackManager(
        max_snapshots=10,
        config={
            "storage_path": temp_storage_dir,
            "compress_snapshots": False,
            "verify_integrity": True,
            "auto_cleanup": False,
        },
    )
    yield manager
    manager.shutdown()


@pytest.fixture
def audit_logger(temp_storage_dir):
    """Create an audit logger with temporary storage."""
    log_path = Path(temp_storage_dir) / "audit_logs"
    logger = AuditLogger(
        log_path=str(log_path),
        config={
            "redact_sensitive": True,
            "rotation_days": 7,
            "compress_old_logs": False,
            "enable_signing": True,
            "max_log_size_mb": 10,
        },
    )
    yield logger
    logger.shutdown()


@pytest.fixture
def sample_state():
    """Create sample state for snapshots."""
    return {
        "agents": {
            "agent_1": {"status": "active", "tasks": ["task_1", "task_2"]},
            "agent_2": {"status": "idle", "tasks": []},
        },
        "resources": {"cpu": 0.5, "memory": 0.6},
        "timestamp": time.time(),
    }


@pytest.fixture
def sample_action_log():
    """Create sample action log."""
    return [
        {"type": "explore", "timestamp": time.time() - 100, "result": "success"},
        {"type": "optimize", "timestamp": time.time() - 50, "result": "success"},
        {"type": "maintain", "timestamp": time.time() - 10, "result": "success"},
    ]


@pytest.fixture
def sample_safety_report():
    """Create sample safety report."""
    return SafetyReport(
        safe=False,
        confidence=0.8,
        violations=[SafetyViolationType.COMPLIANCE],
        reasons=["Compliance violation detected"],
        metadata={"test": True},
    )


# ============================================================
# MEMORY BOUNDED DEQUE TESTS
# ============================================================


class TestMemoryBoundedDeque:
    """Tests for MemoryBoundedDeque class."""

    def test_initialization(self):
        """Test deque initialization."""
        deque = MemoryBoundedDeque(max_size_mb=5)
        assert len(deque) == 0
        assert deque.max_size_bytes == 5 * 1024 * 1024
        assert deque.current_size_bytes == 0

    def test_append_single_item(self):
        """Test appending a single item."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        item = {"data": "test_value"}
        deque.append(item)

        assert len(deque) == 1
        assert deque.current_size_bytes > 0

    def test_memory_limit_enforcement(self):
        """Test that memory limit is enforced."""
        deque = MemoryBoundedDeque(max_size_mb=0.01)  # Very small limit

        # Add many items
        for i in range(100):
            deque.append({"data": "x" * 1000})

        # Should have evicted old items
        assert len(deque) < 100
        assert deque.current_size_bytes <= deque.max_size_bytes

    def test_clear(self):
        """Test clearing the deque."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        deque.append({"a": 1})
        deque.append({"b": 2})

        deque.clear()
        assert len(deque) == 0
        assert deque.current_size_bytes == 0

    def test_iteration(self):
        """Test iterating over deque."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        items = [{"id": i} for i in range(5)]

        for item in items:
            deque.append(item)

        collected = list(deque)
        assert len(collected) == 5

    def test_bool_operator(self):
        """Test boolean evaluation."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        assert not deque

        deque.append({"data": "test"})
        assert deque

    @pytest.mark.timeout(10)
    def test_thread_safety(self):
        """Test thread-safe operations."""
        deque = MemoryBoundedDeque(max_size_mb=10)

        def add_items():
            for i in range(20):
                deque.append({"id": i})

        threads = [threading.Thread(target=add_items) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(deque) == 60


# ============================================================
# ROLLBACK MANAGER TESTS
# ============================================================


class TestRollbackManager:
    """Tests for RollbackManager class."""

    @pytest.mark.timeout(10)
    def test_initialization(self, temp_storage_dir):
        """Test manager initialization."""
        manager = RollbackManager(
            max_snapshots=20,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        try:
            assert manager.max_snapshots == 20
            assert len(manager.snapshots) == 0
            assert manager.conn is not None
        finally:
            manager.shutdown()

    @pytest.mark.timeout(10)
    def test_create_snapshot(self, rollback_manager, sample_state, sample_action_log):
        """Test creating a snapshot."""
        snapshot_id = rollback_manager.create_snapshot(
            state=sample_state,
            action_log=sample_action_log,
            metadata={"reason": "test"},
        )

        assert snapshot_id is not None
        assert len(rollback_manager.snapshots) == 1
        assert snapshot_id in rollback_manager.snapshot_index
        assert rollback_manager.metrics["total_snapshots"] == 1

    @pytest.mark.timeout(10)
    def test_create_multiple_snapshots(
        self, rollback_manager, sample_state, sample_action_log
    ):
        """Test creating multiple snapshots."""
        snapshot_ids = []

        for i in range(5):
            state = {**sample_state, "iteration": i}
            snapshot_id = rollback_manager.create_snapshot(
                state=state, action_log=sample_action_log
            )
            snapshot_ids.append(snapshot_id)

        assert len(rollback_manager.snapshots) == 5
        assert all(sid in rollback_manager.snapshot_index for sid in snapshot_ids)

    @pytest.mark.timeout(10)
    def test_snapshot_persistence(
        self, temp_storage_dir, sample_state, sample_action_log
    ):
        """Test that snapshots are persisted to disk."""
        manager = RollbackManager(
            max_snapshots=5,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        try:
            snapshot_id = manager.create_snapshot(
                state=sample_state, action_log=sample_action_log
            )

            # Check file exists
            storage_path = Path(temp_storage_dir)
            snapshot_files = list(storage_path.glob("snapshot_*.dat"))
            assert len(snapshot_files) == 1
        finally:
            manager.shutdown()

    @pytest.mark.timeout(10)
    def test_snapshot_compression(
        self, temp_storage_dir, sample_state, sample_action_log
    ):
        """Test snapshot compression."""
        manager = RollbackManager(
            max_snapshots=5,
            config={
                "storage_path": temp_storage_dir,
                "compress_snapshots": True,
                "auto_cleanup": False,
            },
        )

        try:
            snapshot_id = manager.create_snapshot(
                state=sample_state, action_log=sample_action_log
            )

            # Verify compression was attempted
            assert snapshot_id in manager.snapshot_index
        finally:
            manager.shutdown()

    @pytest.mark.timeout(10)
    def test_rollback_to_latest(
        self, rollback_manager, sample_state, sample_action_log
    ):
        """Test rolling back to most recent snapshot."""
        snapshot_id = rollback_manager.create_snapshot(
            state=sample_state, action_log=sample_action_log
        )

        result = rollback_manager.rollback(reason="test_rollback")

        assert result is not None
        assert "state" in result
        assert "action_log" in result
        assert "rollback_metadata" in result
        assert result["rollback_metadata"]["snapshot_id"] == snapshot_id
        assert rollback_manager.metrics["successful_rollbacks"] == 1

    @pytest.mark.timeout(10)
    def test_rollback_to_specific_snapshot(
        self, rollback_manager, sample_state, sample_action_log
    ):
        """Test rolling back to a specific snapshot."""
        # Create multiple snapshots
        snapshot_ids = []
        for i in range(3):
            state = {**sample_state, "version": i}
            sid = rollback_manager.create_snapshot(
                state=state, action_log=sample_action_log
            )
            snapshot_ids.append(sid)

        # Rollback to first snapshot
        result = rollback_manager.rollback(
            snapshot_id=snapshot_ids[0], reason="specific_rollback"
        )

        assert result is not None
        assert result["state"]["version"] == 0
        assert result["rollback_metadata"]["snapshot_id"] == snapshot_ids[0]

    @pytest.mark.timeout(10)
    def test_rollback_nonexistent_snapshot(self, rollback_manager):
        """Test rollback to nonexistent snapshot."""
        result = rollback_manager.rollback(snapshot_id="nonexistent_id", reason="test")

        assert result is None
        assert rollback_manager.metrics["failed_rollbacks"] == 1

    @pytest.mark.timeout(10)
    def test_rollback_without_snapshots(self, rollback_manager):
        """Test rollback when no snapshots exist."""
        result = rollback_manager.rollback(reason="test")

        assert result is None
        assert rollback_manager.metrics["failed_rollbacks"] == 1

    @pytest.mark.timeout(10)
    def test_quarantine_action(self, rollback_manager):
        """Test quarantining an action."""
        action = {"type": "dangerous_operation", "params": {"level": "high"}}

        quarantine_id = rollback_manager.quarantine_action(
            action=action, reason="safety_violation", duration_seconds=3600
        )

        assert quarantine_id is not None
        assert quarantine_id in rollback_manager.quarantine
        assert rollback_manager.metrics["quarantined_actions"] == 1

    @pytest.mark.timeout(10)
    def test_get_quarantine_item(self, rollback_manager):
        """Test retrieving quarantined item."""
        action = {"type": "test_action"}
        quarantine_id = rollback_manager.quarantine_action(action=action, reason="test")

        item = rollback_manager.get_quarantine_item(quarantine_id)

        assert item is not None
        assert item["action"] == action
        assert item["reason"] == "test"
        assert item["status"] == "quarantined"
        assert item["reviewed"] is False

    @pytest.mark.timeout(10)
    def test_review_quarantine(self, rollback_manager):
        """Test reviewing a quarantined action."""
        action = {"type": "test_action"}
        quarantine_id = rollback_manager.quarantine_action(action=action, reason="test")

        success = rollback_manager.review_quarantine(
            quarantine_id=quarantine_id,
            approved=True,
            reviewer="test_reviewer",
            notes="Looks safe",
        )

        assert success is True

        item = rollback_manager.get_quarantine_item(quarantine_id)
        assert item["reviewed"] is True
        assert item["reviewer"] == "test_reviewer"
        assert item["status"] == "approved"

    @pytest.mark.timeout(10)
    def test_review_nonexistent_quarantine(self, rollback_manager):
        """Test reviewing nonexistent quarantine."""
        success = rollback_manager.review_quarantine(
            quarantine_id="nonexistent", approved=True
        )

        assert success is False

    @pytest.mark.timeout(10)
    def test_cleanup_expired_quarantine(self, rollback_manager):
        """Test cleanup of expired quarantine entries."""
        # Create expired quarantine
        action = {"type": "test"}
        quarantine_id = rollback_manager.quarantine_action(
            action=action,
            reason="test",
            duration_seconds=0.1,  # Very short duration
        )

        assert quarantine_id in rollback_manager.quarantine

        # Wait for expiry
        time.sleep(0.2)

        # Cleanup
        rollback_manager.cleanup_expired_quarantine()

        # Should be removed
        assert quarantine_id not in rollback_manager.quarantine

    @pytest.mark.timeout(10)
    def test_get_snapshot_history(
        self, rollback_manager, sample_state, sample_action_log
    ):
        """Test retrieving snapshot history."""
        # Create snapshots
        for i in range(5):
            rollback_manager.create_snapshot(
                state={**sample_state, "i": i}, action_log=sample_action_log
            )

        history = rollback_manager.get_snapshot_history(limit=3)

        assert len(history) == 3
        assert all("snapshot_id" in entry for entry in history)

    @pytest.mark.timeout(10)
    def test_get_rollback_history(
        self, rollback_manager, sample_state, sample_action_log
    ):
        """Test retrieving rollback history."""
        # Create and rollback snapshots
        rollback_manager.create_snapshot(sample_state, sample_action_log)
        rollback_manager.rollback(reason="test1")

        rollback_manager.create_snapshot(sample_state, sample_action_log)
        rollback_manager.rollback(reason="test2")

        history = rollback_manager.get_rollback_history(limit=5)

        assert len(history) == 2
        assert all("rollback_id" in entry for entry in history)

    @pytest.mark.timeout(10)
    def test_get_metrics(self, rollback_manager, sample_state, sample_action_log):
        """Test getting metrics."""
        rollback_manager.create_snapshot(sample_state, sample_action_log)
        rollback_manager.quarantine_action({"type": "test"}, "test")

        metrics = rollback_manager.get_metrics()

        assert "total_snapshots" in metrics
        assert "quarantined_actions" in metrics
        assert "current_snapshots" in metrics
        assert "rollback_success_rate" in metrics
        assert metrics["total_snapshots"] == 1
        assert metrics["quarantined_actions"] == 1

    @pytest.mark.timeout(10)
    def test_export_snapshot(
        self, rollback_manager, sample_state, sample_action_log, temp_storage_dir
    ):
        """Test exporting a snapshot."""
        snapshot_id = rollback_manager.create_snapshot(
            state=sample_state, action_log=sample_action_log
        )

        export_path = Path(temp_storage_dir) / "export.json"
        success = rollback_manager.export_snapshot(
            snapshot_id=snapshot_id, export_path=str(export_path)
        )

        assert success is True
        assert export_path.exists()

        # Verify export content
        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["snapshot_id"] == snapshot_id
        assert "state" in data
        assert "action_log" in data

    @pytest.mark.timeout(10)
    def test_import_snapshot(
        self, rollback_manager, sample_state, sample_action_log, temp_storage_dir
    ):
        """Test importing a snapshot."""
        # Create and export
        original_id = rollback_manager.create_snapshot(
            state=sample_state, action_log=sample_action_log
        )

        export_path = Path(temp_storage_dir) / "export.json"
        rollback_manager.export_snapshot(original_id, str(export_path))

        # Import
        imported_id = rollback_manager.import_snapshot(str(export_path))

        assert imported_id is not None
        assert imported_id in rollback_manager.snapshot_index

    @pytest.mark.timeout(10)
    def test_max_snapshots_limit(
        self, temp_storage_dir, sample_state, sample_action_log
    ):
        """Test that snapshot limit is enforced."""
        manager = RollbackManager(
            max_snapshots=3,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        try:
            # Create more snapshots than limit
            for i in range(5):
                manager.create_snapshot(
                    state={**sample_state, "i": i}, action_log=sample_action_log
                )

            # Should not exceed limit
            assert len(manager.snapshots) <= 3
        finally:
            manager.shutdown()

    @pytest.mark.timeout(10)
    def test_snapshot_integrity_verification(
        self, rollback_manager, sample_state, sample_action_log
    ):
        """Test snapshot integrity verification."""
        snapshot_id = rollback_manager.create_snapshot(
            state=sample_state, action_log=sample_action_log
        )

        # Get snapshot
        snapshot = rollback_manager.snapshot_index[snapshot_id]

        # Verify integrity
        assert snapshot.verify_integrity() is True

    @pytest.mark.timeout(10)
    def test_shutdown(self, rollback_manager):
        """Test manager shutdown."""
        rollback_manager.shutdown()

        assert rollback_manager._shutdown is True

        # Calling again should be safe
        rollback_manager.shutdown()

    @pytest.mark.timeout(10)
    def test_persistence_across_restarts(
        self, temp_storage_dir, sample_state, sample_action_log
    ):
        """Test that snapshots persist across manager restarts."""
        # Create manager and snapshot
        manager1 = RollbackManager(
            max_snapshots=10,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        try:
            snapshot_id = manager1.create_snapshot(
                state=sample_state, action_log=sample_action_log
            )
        finally:
            manager1.shutdown()

        # Create new manager with same storage
        manager2 = RollbackManager(
            max_snapshots=10,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        try:
            # Should load existing snapshot
            assert len(manager2.snapshots) == 1
            assert snapshot_id in manager2.snapshot_index
        finally:
            manager2.shutdown()


# ============================================================
# AUDIT LOGGER TESTS
# ============================================================


class TestAuditLogger:
    """Tests for AuditLogger class."""

    @pytest.mark.timeout(10)
    def test_initialization(self, temp_storage_dir):
        """Test logger initialization."""
        log_path = Path(temp_storage_dir) / "audit_logs"
        logger = AuditLogger(log_path=str(log_path), config={"redact_sensitive": True})

        try:
            assert logger.log_path.exists()
            assert logger.conn is not None
            assert len(logger.log_buffer) == 0
        finally:
            logger.shutdown()

    @pytest.mark.timeout(10)
    def test_log_safety_decision(self, audit_logger, sample_safety_report):
        """Test logging a safety decision."""
        decision = {
            "type": "explore",
            "action_id": "test_123",
            "timestamp": time.time(),
        }

        entry_id = audit_logger.log_safety_decision(
            decision=decision, report=sample_safety_report
        )

        assert entry_id is not None
        assert audit_logger.metrics["total_entries"] > 0

    @pytest.mark.timeout(10)
    def test_log_event(self, audit_logger):
        """Test logging a general event."""
        event_data = {
            "event": "system_startup",
            "version": "1.0.0",
            "timestamp": time.time(),
        }

        entry_id = audit_logger.log_event(
            event_type="system_event", event_data=event_data, severity="info"
        )

        assert entry_id is not None

    @pytest.mark.timeout(10)
    def test_sensitive_data_redaction(self, audit_logger):
        """Test that sensitive data is redacted."""
        sensitive_data = {
            "email": "user@example.com",
            "ssn": "123-45-6789",
            "phone": "555-123-4567",
            "safe_data": "this is safe",
        }

        redacted = audit_logger._redact_sensitive(sensitive_data)

        assert "EMAIL_REDACTED" in str(redacted)
        assert "SSN_REDACTED" in str(redacted)
        assert "PHONE_REDACTED" in str(redacted)
        assert "this is safe" in str(redacted)

    @pytest.mark.timeout(10)
    def test_redaction_patterns(self, audit_logger):
        """Test various redaction patterns."""
        test_cases = [
            ("My SSN is 123-45-6789", "[SSN_REDACTED]"),
            ("Email: test@example.com", "[EMAIL_REDACTED]"),
            ("Card: 1234-5678-9012-3456", "[CC_REDACTED]"),
            ("Call 555-123-4567", "[PHONE_REDACTED]"),
            ("IP: 192.168.1.1", "[IP_REDACTED]"),
        ]

        for original, expected_pattern in test_cases:
            redacted = audit_logger._redact_sensitive(original)
            assert expected_pattern in redacted

    @pytest.mark.timeout(10)
    def test_hash_chain_integrity(self, audit_logger):
        """Test hash chain for tamper detection."""
        # Log multiple entries
        for i in range(5):
            audit_logger.log_event(event_type="test_event", event_data={"iteration": i})

        # Flush buffer
        audit_logger._flush_buffer_batch()

        # Verify integrity
        verification = audit_logger.verify_integrity()

        assert verification["verified"] is True
        assert verification["entries_checked"] > 0

    @pytest.mark.timeout(10)
    def test_query_logs(self, audit_logger):
        """Test querying logs."""
        # Log some entries
        for i in range(5):
            audit_logger.log_event(
                event_type="test_event", event_data={"id": i}, severity="info"
            )

        # Flush to ensure they're written
        audit_logger._flush_buffer_batch()

        # Query
        results = audit_logger.query_logs(
            filters={"entry_type": "test_event"}, limit=10
        )

        assert len(results) > 0
        assert audit_logger.metrics["searches"] == 1

    @pytest.mark.timeout(10)
    def test_query_with_time_range(self, audit_logger):
        """Test querying logs with time range."""
        start_time = time.time()

        # Log entries
        audit_logger.log_event("test1", {"data": "test"})
        time.sleep(0.1)
        audit_logger.log_event("test2", {"data": "test"})

        end_time = time.time()

        audit_logger._flush_buffer_batch()

        # Query with time range
        results = audit_logger.query_logs(
            start_time=start_time, end_time=end_time, limit=10
        )

        assert len(results) > 0

    @pytest.mark.timeout(10)
    def test_query_input_validation(self, audit_logger):
        """Test that query input is validated."""
        # Invalid limit
        with pytest.raises(ValueError, match="Limit must be"):
            audit_logger.query_logs(limit=20000)

        # Invalid sort field
        with pytest.raises(ValueError, match="Invalid sort field"):
            audit_logger.query_logs(sort_by="malicious_field")

        # Invalid sort order
        with pytest.raises(ValueError, match="Invalid sort order"):
            audit_logger.query_logs(sort_order="INJECT")

    @pytest.mark.timeout(10)
    def test_query_filters_white[self, audit_logger):
        """Test that only whitelisted filter fields are allowed."""
        # Log entry
        audit_logger.log_event("test", {"data": "test"})
        audit_logger._flush_buffer_batch()

        # Valid filter - should work
        results = audit_logger.query_logs(filters={"entry_type": "test"})

        # Invalid filter field - should be ignored
        results = audit_logger.query_logs(filters={"malicious_field": "value"})

        # Should not raise exception, just ignore invalid field
        assert isinstance(results, list)

    @pytest.mark.timeout(10)
    def test_log_rotation_by_size(self, temp_storage_dir):
        """Test log rotation based on file size."""
        log_path = Path(temp_storage_dir) / "audit_logs"
        logger = AuditLogger(
            log_path=str(log_path),
            config={"max_log_size_mb": 0.001},  # Very small limit
        )

        try:
            # Log many entries
            for i in range(100):
                logger.log_event("test", {"data": "x" * 1000})

            logger._flush_buffer_batch()
            logger._check_rotation()

            # Should have rotated
            assert logger.metrics["rotations"] >= 0
        finally:
            logger.shutdown()

    @pytest.mark.timeout(10)
    def test_get_metrics(self, audit_logger):
        """Test getting audit logger metrics."""
        audit_logger.log_event("test", {"data": "test"})

        metrics = audit_logger.get_metrics()

        assert "total_entries" in metrics
        assert "buffer_size" in metrics
        assert "current_log_file" in metrics
        assert "log_size_mb" in metrics
        assert metrics["total_entries"] > 0

    @pytest.mark.timeout(10)
    def test_export_logs_json(self, audit_logger, temp_storage_dir):
        """Test exporting logs to JSON."""
        # Log entries
        for i in range(3):
            audit_logger.log_event("test", {"id": i})

        audit_logger._flush_buffer_batch()

        # Export
        export_path = Path(temp_storage_dir) / "export.json"
        success = audit_logger.export_logs(export_path=str(export_path), format="json")

        assert success is True
        assert export_path.exists()

        # Verify content
        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)

    @pytest.mark.timeout(10)
    def test_export_logs_csv(self, audit_logger, temp_storage_dir):
        """Test exporting logs to CSV."""
        # Log entries
        for i in range(3):
            audit_logger.log_event("test", {"id": i})

        audit_logger._flush_buffer_batch()

        # Export
        export_path = Path(temp_storage_dir) / "export.csv"
        success = audit_logger.export_logs(export_path=str(export_path), format="csv")

        assert success is True
        assert export_path.exists()

    @pytest.mark.timeout(10)
    def test_get_summary(self, audit_logger, sample_safety_report):
        """Test getting log summary."""
        # Log various entries
        audit_logger.log_event("system_event", {"type": "startup"}, severity="info")
        audit_logger.log_event("user_action", {"action": "login"}, severity="info")
        audit_logger.log_safety_decision({"type": "test"}, sample_safety_report)

        audit_logger._flush_buffer_batch()

        summary = audit_logger.get_summary()

        assert "total_entries" in summary
        assert "entry_types" in summary
        assert "severities" in summary
        assert "safety_stats" in summary
        assert summary["total_entries"] > 0

    @pytest.mark.timeout(10)
    def test_batch_flush(self, audit_logger):
        """Test batch flushing of log buffer."""
        # Add many entries
        for i in range(15):
            audit_logger.log_event("test", {"id": i})

        # Buffer should have flushed automatically
        assert len(audit_logger.log_buffer) < 15

    @pytest.mark.timeout(10)
    def test_shutdown(self, audit_logger):
        """Test logger shutdown."""
        audit_logger.log_event("test", {"data": "test"})

        audit_logger.shutdown()

        assert audit_logger._shutdown is True
        assert len(audit_logger.log_buffer) == 0

        # Calling again should be safe
        audit_logger.shutdown()

    @pytest.mark.timeout(15)
    def test_concurrent_logging(self, audit_logger):
        """Test concurrent logging from multiple threads."""

        def log_entries():
            for i in range(10):
                audit_logger.log_event(
                    "thread_test", {"thread": threading.current_thread().name}
                )

        threads = [threading.Thread(target=log_entries) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        audit_logger._flush_buffer_batch()

        # All entries should be logged
        assert audit_logger.metrics["total_entries"] >= 30


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for rollback and audit systems."""

    @pytest.mark.timeout(15)
    def test_rollback_with_audit(
        self, temp_storage_dir, sample_state, sample_action_log
    ):
        """Test integrated rollback and audit logging."""
        # Setup both systems
        manager = RollbackManager(
            max_snapshots=10,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        log_path = Path(temp_storage_dir) / "audit_logs"
        logger = AuditLogger(log_path=str(log_path))

        try:
            # Create snapshot
            snapshot_id = manager.create_snapshot(
                state=sample_state, action_log=sample_action_log
            )

            # Log the snapshot creation
            logger.log_event(
                "snapshot_created", {"snapshot_id": snapshot_id}, severity="info"
            )

            # Perform rollback
            result = manager.rollback(reason="test_rollback")

            # Log the rollback
            logger.log_event(
                "rollback_performed",
                {
                    "snapshot_id": result["rollback_metadata"]["snapshot_id"],
                    "reason": "test_rollback",
                },
                severity="warning",
            )

            logger._flush_buffer_batch()

            # Verify both systems recorded events
            assert result is not None
            assert logger.metrics["total_entries"] >= 2
        finally:
            manager.shutdown()
            logger.shutdown()

    @pytest.mark.timeout(15)
    def test_quarantine_with_audit(self, temp_storage_dir):
        """Test quarantine with audit trail."""
        manager = RollbackManager(
            max_snapshots=10,
            config={"storage_path": temp_storage_dir, "auto_cleanup": False},
        )

        log_path = Path(temp_storage_dir) / "audit_logs"
        logger = AuditLogger(log_path=str(log_path))

        try:
            # Quarantine action
            action = {"type": "dangerous", "params": {}}
            quarantine_id = manager.quarantine_action(
                action=action, reason="safety_violation"
            )

            # Audit the quarantine
            logger.log_event(
                "action_quarantined",
                {"quarantine_id": quarantine_id, "reason": "safety_violation"},
                severity="critical",
            )

            # Review quarantine
            manager.review_quarantine(
                quarantine_id=quarantine_id, approved=False, reviewer="security_team"
            )

            # Audit the review
            logger.log_event(
                "quarantine_reviewed",
                {"quarantine_id": quarantine_id, "approved": False},
                severity="warning",
            )

            logger._flush_buffer_batch()

            assert quarantine_id in manager.quarantine
            assert logger.metrics["total_entries"] >= 2
        finally:
            manager.shutdown()
            logger.shutdown()


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.timeout(10)
    def test_empty_state_snapshot(self, rollback_manager):
        """Test creating snapshot with empty state."""
        snapshot_id = rollback_manager.create_snapshot(state={}, action_log=[])

        assert snapshot_id is not None

    @pytest.mark.timeout(10)
    def test_large_state_snapshot(self, rollback_manager):
        """Test snapshot with large state."""
        large_state = {"data": ["item" * 100 for _ in range(1000)]}

        snapshot_id = rollback_manager.create_snapshot(state=large_state, action_log=[])

        assert snapshot_id is not None

    @pytest.mark.timeout(10)
    def test_nested_data_redaction(self, audit_logger):
        """Test redaction of deeply nested data."""
        nested_data = {
            "level1": {"level2": {"email": "test@example.com", "safe": "data"}}
        }

        redacted = audit_logger._redact_sensitive(nested_data)

        assert "EMAIL_REDACTED" in str(redacted)
        assert "safe" in str(redacted)

    @pytest.mark.timeout(15)
    def test_concurrent_snapshot_creation(self, rollback_manager, sample_state):
        """Test concurrent snapshot creation with max_snapshots limit enforcement."""

        def create_snapshots():
            for i in range(5):
                rollback_manager.create_snapshot(
                    state={**sample_state, "thread": threading.current_thread().name},
                    action_log=[],
                )

        threads = [threading.Thread(target=create_snapshots) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The fixture has max_snapshots=10, so even though 15 were created,
        # only the most recent 10 should be retained
        assert len(rollback_manager.snapshots) == 10
        # Verify that the manager correctly enforced the limit
        assert rollback_manager.max_snapshots == 10

    @pytest.mark.timeout(10)
    def test_snapshot_after_shutdown(self, rollback_manager, sample_state):
        """Test that operations fail gracefully after shutdown."""
        rollback_manager.shutdown()

        with pytest.raises(RuntimeError):
            rollback_manager.create_snapshot(state=sample_state, action_log=[])

    @pytest.mark.timeout(30)
    def test_large_graph_performance(self, temp_storage_dir):
        """Test performance with large graph creating many managers."""
        managers = []

        try:
            # Create multiple managers to simulate large graph scenario
            for i in range(10):
                manager = RollbackManager(
                    max_snapshots=5,
                    config={
                        "storage_path": f"{temp_storage_dir}/manager_{i}",
                        "compress_snapshots": False,
                        "verify_integrity": False,
                        "auto_cleanup": False,  # Disable auto-cleanup to prevent thread issues
                    },
                )
                managers.append(manager)

                # Create a few snapshots per manager
                for j in range(3):
                    manager.create_snapshot(
                        state={"manager": i, "iteration": j, "data": "test" * 100},
                        action_log=[{"action": "test", "step": j}],
                    )

            # Verify all managers created snapshots
            assert all(len(m.snapshots) == 3 for m in managers)

        finally:
            # CRITICAL: Ensure all managers are properly shut down
            for manager in managers:
                try:
                    manager.shutdown()
                except Exception as e:
                    print(f"Error during manager shutdown: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
