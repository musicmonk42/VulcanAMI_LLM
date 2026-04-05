"""Tests for canonical AuditProtocol."""
import pytest
from src.protocols.audit import AuditLogger


class TestAuditLogger:
    def setup_method(self):
        self.logger = AuditLogger(max_memory_events=100)

    def test_log_and_retrieve(self):
        self.logger.log_event("test", {"key": "value"})
        events = self.logger.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "test"

    def test_filter_by_type(self):
        self.logger.log_event("type_a", {"x": 1})
        self.logger.log_event("type_b", {"x": 2})
        events = self.logger.get_events(event_type="type_a")
        assert len(events) == 1

    def test_hash_chain_integrity(self):
        self.logger.log_event("e1", {"a": 1})
        self.logger.log_event("e2", {"b": 2})
        self.logger.log_event("e3", {"c": 3})
        assert self.logger.verify_integrity() is True

    def test_tamper_detection(self):
        self.logger.log_event("e1", {"a": 1})
        self.logger.log_event("e2", {"b": 2})
        # Tamper with an event
        self.logger._events[0].data["a"] = 999
        assert self.logger.verify_integrity() is False

    def test_redaction(self):
        logger = AuditLogger(redact_patterns=["password", "secret"])
        logger.log_event("login", {"user": "bob", "password": "hunter2"})
        events = logger.get_events()
        assert events[0]["data"]["password"] == "[REDACTED]"
        assert events[0]["data"]["user"] == "bob"

    def test_memory_limit(self):
        logger = AuditLogger(max_memory_events=5)
        for i in range(10):
            logger.log_event("test", {"i": i})
        assert len(logger.get_events(limit=100)) == 5

    def test_shutdown_clears(self):
        self.logger.log_event("test", {})
        self.logger.shutdown()
        assert len(self.logger.get_events()) == 0
