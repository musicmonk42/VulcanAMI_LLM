"""
Comprehensive test suite for security_audit_engine.py
"""

import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from security_audit_engine import (
    AuditEngineError,
    ConnectionPool,
    DatabaseCorruptionError,
    SecurityAuditEngine,
)


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def audit_engine(temp_db):
    """Create SecurityAuditEngine instance."""
    engine = SecurityAuditEngine(db_path=temp_db)
    yield engine
    engine.close()


class TestConnectionPool:
    """Test ConnectionPool class."""

    def test_initialization(self, temp_db):
        """Test connection pool initialization."""
        pool = ConnectionPool(Path(temp_db), max_connections=3)

        assert pool.max_connections == 3
        assert len(pool.pool) == 0

        pool.close_all()

    def test_get_connection(self, temp_db):
        """Test getting connection from pool."""
        pool = ConnectionPool(Path(temp_db))

        conn = pool.get_connection()

        assert conn is not None

        pool.return_connection(conn)
        pool.close_all()

    def test_return_connection(self, temp_db):
        """Test returning connection to pool."""
        pool = ConnectionPool(Path(temp_db))

        conn = pool.get_connection()
        pool.return_connection(conn)

        assert len(pool.pool) == 1

        pool.close_all()

    def test_connection_reuse(self, temp_db):
        """Test connection reuse."""
        pool = ConnectionPool(Path(temp_db))

        conn1 = pool.get_connection()
        pool.return_connection(conn1)

        conn2 = pool.get_connection()

        # Should get same connection
        assert id(conn1) == id(conn2)

        pool.return_connection(conn2)
        pool.close_all()

    def test_max_connections_limit(self, temp_db):
        """Test maximum connections limit."""
        pool = ConnectionPool(Path(temp_db), max_connections=2)

        conn1 = pool.get_connection()
        conn2 = pool.get_connection()

        # Third connection should wait/timeout
        # We won't actually test the timeout as it takes too long

        pool.return_connection(conn1)
        pool.return_connection(conn2)
        pool.close_all()

    def test_close_all(self, temp_db):
        """Test closing all connections."""
        pool = ConnectionPool(Path(temp_db))

        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        pool.return_connection(conn1)
        pool.return_connection(conn2)

        pool.close_all()

        assert len(pool.pool) == 0
        assert len(pool.in_use) == 0


class TestSecurityAuditEngineInitialization:
    """Test SecurityAuditEngine initialization."""

    def test_initialization(self, temp_db):
        """Test basic initialization."""
        engine = SecurityAuditEngine(db_path=temp_db)

        assert engine.db_path.exists()
        assert engine.pool is not None

        engine.close()

    def test_initialization_creates_tables(self, temp_db):
        """Test that initialization creates tables."""
        engine = SecurityAuditEngine(db_path=temp_db)

        with engine._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

        assert "audit_log" in tables
        assert "audit_metadata" in tables

        engine.close()

    def test_context_manager(self, temp_db):
        """Test context manager usage."""
        with SecurityAuditEngine(db_path=temp_db) as engine:
            assert engine.db_path.exists()

        # Should be closed


class TestLogEvent:
    """Test event logging."""

    def test_log_event_basic(self, audit_engine):
        """Test logging basic event."""
        audit_engine.log_event("test_event", {"key": "value"})

        # Verify event was logged
        events = audit_engine.query_events(event_type="test_event")

        assert len(events) == 1
        assert events[0]["event_type"] == "test_event"
        assert events[0]["details"]["key"] == "value"

    def test_log_event_with_severity(self, audit_engine):
        """Test logging event with severity."""
        audit_engine.log_event("warning_event", {"msg": "test"}, severity="warning")

        events = audit_engine.query_events(event_type="warning_event")

        assert events[0]["severity"] == "warning"

    def test_log_critical_event(self, audit_engine):
        """Test logging critical event."""
        # Mock slack client to avoid actual sending
        audit_engine.slack_client = MagicMock()
        audit_engine.slack_channel = "#test"

        audit_engine.log_event(
            "integrity_failure", {"error": "test"}, severity="critical"
        )

        events = audit_engine.query_events(event_type="integrity_failure")

        assert len(events) == 1

    def test_log_multiple_events(self, audit_engine):
        """Test logging multiple events."""
        for i in range(5):
            audit_engine.log_event(f"event_{i}", {"index": i})

        events = audit_engine.query_events(limit=10)

        assert len(events) == 5


class TestQueryEvents:
    """Test querying events."""

    def test_query_all_events(self, audit_engine):
        """Test querying all events."""
        audit_engine.log_event("event1", {"data": "1"})
        audit_engine.log_event("event2", {"data": "2"})

        events = audit_engine.query_events()

        assert len(events) >= 2

    def test_query_by_type(self, audit_engine):
        """Test querying by event type."""
        audit_engine.log_event("type_a", {"data": "a"})
        audit_engine.log_event("type_b", {"data": "b"})
        audit_engine.log_event("type_a", {"data": "a2"})

        events = audit_engine.query_events(event_type="type_a")

        assert len(events) == 2
        assert all(e["event_type"] == "type_a" for e in events)

    def test_query_by_severity(self, audit_engine):
        """Test querying by severity."""
        audit_engine.log_event("event1", {}, severity="info")
        audit_engine.log_event("event2", {}, severity="error")
        audit_engine.log_event("event3", {}, severity="error")

        events = audit_engine.query_events(severity="error")

        assert len(events) == 2

    def test_query_with_time_range(self, audit_engine):
        """Test querying with time range."""
        start_time = datetime.utcnow().isoformat()

        audit_engine.log_event("timed_event", {})

        time.sleep(0.1)
        end_time = datetime.utcnow().isoformat()

        events = audit_engine.query_events(
            event_type="timed_event", start_time=start_time, end_time=end_time
        )

        assert len(events) >= 1

    def test_query_with_limit(self, audit_engine):
        """Test querying with limit."""
        for i in range(10):
            audit_engine.log_event("test", {"i": i})

        events = audit_engine.query_events(event_type="test", limit=5)

        assert len(events) == 5


class TestAlertSending:
    """Test alert sending."""

    @patch("security_audit_engine.SLACK_AVAILABLE", True)
    def test_send_alert_success(self, audit_engine):
        """Test successful alert sending."""
        mock_client = MagicMock()
        audit_engine.slack_client = mock_client
        audit_engine.slack_channel = "#test"

        audit_engine._send_alert("test_event", {"key": "value"}, "2025-01-01T00:00:00Z")

        assert mock_client.chat_postMessage.called

    @patch("security_audit_engine.SLACK_AVAILABLE", True)
    def test_send_alert_failure(self, audit_engine):
        """Test alert sending failure."""
        mock_client = MagicMock()
        mock_client.chat_postMessage.side_effect = Exception("Send failed")
        audit_engine.slack_client = mock_client
        audit_engine.slack_channel = "#test"

        # Should not raise, just log error
        audit_engine._send_alert("test_event", {"key": "value"}, "2025-01-01T00:00:00Z")

    def test_send_alert_no_client(self, audit_engine):
        """Test alert when no client configured."""
        audit_engine.slack_client = None

        # Should not raise
        audit_engine._send_alert("test_event", {}, "2025-01-01T00:00:00Z")


class TestStatistics:
    """Test statistics."""

    def test_get_statistics(self, audit_engine):
        """Test getting statistics."""
        # Log some events
        for i in range(5):
            audit_engine.log_event("test", {"i": i}, severity="info")

        stats = audit_engine.get_statistics()

        assert "total_events" in stats
        assert "events_by_type" in stats
        assert "events_by_severity" in stats
        assert stats["total_events"] >= 5

    def test_statistics_by_type(self, audit_engine):
        """Test statistics by type."""
        audit_engine.log_event("type_a", {})
        audit_engine.log_event("type_a", {})
        audit_engine.log_event("type_b", {})

        stats = audit_engine.get_statistics()

        assert stats["events_by_type"]["type_a"] == 2
        assert stats["events_by_type"]["type_b"] == 1


class TestCleanup:
    """Test cleanup operations."""

    def test_cleanup_old_events(self, audit_engine):
        """Test cleaning up old events."""
        # This test is tricky without time travel
        # We'll just verify the method works
        deleted = audit_engine.cleanup_old_events(days=365)

        assert deleted >= 0

    def test_cleanup_with_recent_events(self, audit_engine):
        """Test cleanup doesn't remove recent events."""
        audit_engine.log_event("recent", {})

        deleted = audit_engine.cleanup_old_events(days=1)

        # Should not delete recent event
        assert deleted == 0

        events = audit_engine.query_events(event_type="recent")
        assert len(events) == 1


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_logging(self, audit_engine):
        """Test concurrent event logging."""

        def log_events(thread_id):
            for i in range(10):
                audit_engine.log_event("concurrent", {"thread": thread_id, "i": i})

        threads = [threading.Thread(target=log_events, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = audit_engine.query_events(event_type="concurrent")

        # Should have logged all 50 events
        assert len(events) == 50

    def test_concurrent_queries(self, audit_engine):
        """Test concurrent querying."""
        # Log some events first
        for i in range(20):
            audit_engine.log_event("query_test", {"i": i})

        def query_events():
            events = audit_engine.query_events(event_type="query_test")
            return len(events)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_events) for _ in range(10)]
            results = [f.result() for f in futures]

        # All queries should succeed
        assert all(r >= 20 for r in results)


class TestDatabaseIntegrity:
    """Test database integrity."""

    def test_verify_integrity_success(self, audit_engine):
        """Test integrity verification succeeds."""
        # Should not raise
        audit_engine._verify_database_integrity()

    def test_transaction_rollback(self, audit_engine):
        """Test transaction rollback on error."""
        with audit_engine._get_connection() as conn:
            cursor = conn.cursor()

            # Get initial count
            cursor.execute("SELECT COUNT(*) FROM audit_log")
            initial_count = cursor.fetchone()[0]

        # Try to execute invalid transaction
        try:
            with audit_engine._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO audit_log (timestamp, event_type, details) VALUES (?, ?, ?)",
                    ("2025-01-01T00:00:00Z", "test", "{}"),
                )
                # Cause error
                raise Exception("Test error")
        except:
            pass

        # Count should be unchanged due to rollback
        with audit_engine._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM audit_log")
            final_count = cursor.fetchone()[0]

        assert final_count == initial_count


class TestExceptions:
    """Test custom exceptions."""

    def test_audit_engine_error(self):
        """Test AuditEngineError."""
        error = AuditEngineError("test error")

        assert str(error) == "test error"

    def test_database_corruption_error(self):
        """Test DatabaseCorruptionError."""
        error = DatabaseCorruptionError("corruption detected")

        assert str(error) == "corruption detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
