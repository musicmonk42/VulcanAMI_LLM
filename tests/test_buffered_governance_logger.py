"""
Tests for BufferedGovernanceLogger - Non-blocking governance logging with buffered writes.

These tests verify the performance fix for synchronous governance logging blocking
on growing file I/O.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# Import the BufferedGovernanceLogger
from vulcan.routing.governance_logger import (
    BufferedGovernanceLogger,
    get_buffered_governance_logger,
    log_routing_result,
)


class TestBufferedGovernanceLogger:
    """Tests for BufferedGovernanceLogger class."""

    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Create a temporary directory for log files."""
        log_dir = tmp_path / "governance_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)

    @pytest.fixture
    def logger_instance(self, temp_log_dir):
        """Create a BufferedGovernanceLogger instance for testing."""
        logger = BufferedGovernanceLogger(
            log_path=temp_log_dir,
            buffer_maxlen=10,
            flush_interval=60.0,  # Long interval to prevent auto-flush during tests
        )
        yield logger
        # Cleanup
        logger._shutdown = True

    def test_initialization(self, temp_log_dir):
        """Test that BufferedGovernanceLogger initializes correctly."""
        logger = BufferedGovernanceLogger(
            log_path=temp_log_dir,
            buffer_maxlen=100,
            flush_interval=5.0,
        )

        assert logger.log_path == Path(temp_log_dir)
        assert logger._buffer.maxlen == 100
        assert logger._flush_interval == 5.0
        assert logger._shutdown is False

        # Cleanup
        logger._shutdown = True

    def test_log_is_non_blocking(self, logger_instance, temp_log_dir):
        """Test that log() returns immediately without writing to disk."""
        # Log an entry
        start_time = time.time()
        logger_instance.log("q_001", {"route": "reasoning", "complexity": 0.5})
        elapsed = time.time() - start_time

        # Should be very fast (non-blocking) - less than 10ms
        assert elapsed < 0.01, f"log() took {elapsed:.3f}s, should be < 0.01s"

        # No files should be written yet (buffer not flushed)
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        assert len(log_files) == 0, "Log file should not be created until flush"

    def test_buffer_accumulates_entries(self, logger_instance):
        """Test that entries accumulate in the buffer."""
        # Log multiple entries
        for i in range(5):
            logger_instance.log(f"q_{i:03d}", {"index": i})

        # Check buffer size
        assert len(logger_instance._buffer) == 5

    def test_buffer_bounded_maxlen(self, logger_instance):
        """Test that buffer respects maxlen and drops oldest entries."""
        # Buffer maxlen is 10, log 15 entries
        for i in range(15):
            logger_instance.log(f"q_{i:03d}", {"index": i})

        # Buffer should only contain 10 entries
        assert len(logger_instance._buffer) == 10

        # Should have dropped 5 entries
        stats = logger_instance.get_stats()
        assert stats["entries_dropped"] == 5

    def test_flush_to_disk(self, logger_instance, temp_log_dir):
        """Test that flush_to_disk writes entries to JSONL file."""
        # Log some entries
        for i in range(3):
            logger_instance.log(f"q_{i:03d}", {"index": i, "route": "reasoning"})

        # Manually flush
        logger_instance._flush_to_disk()

        # Check that file was created
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        assert len(log_files) == 1, "Should have created one log file"

        # Check file contents
        with open(log_files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 3, "Should have 3 log entries"

        # Verify JSON format
        for line in lines:
            entry = json.loads(line)
            assert "query_id" in entry
            assert "timestamp" in entry
            assert "result" in entry

    def test_flush_clears_buffer(self, logger_instance):
        """Test that flush_to_disk clears the buffer."""
        # Log some entries
        for i in range(5):
            logger_instance.log(f"q_{i:03d}", {"index": i})

        assert len(logger_instance._buffer) == 5

        # Flush
        logger_instance._flush_to_disk()

        # Buffer should be empty
        assert len(logger_instance._buffer) == 0

    def test_rotating_log_files(self, temp_log_dir):
        """Test that log files rotate by hour."""
        logger = BufferedGovernanceLogger(
            log_path=temp_log_dir,
            buffer_maxlen=100,
            flush_interval=60.0,
        )

        # Log and flush
        logger.log("q_001", {"test": True})
        logger._flush_to_disk()

        # Check file naming pattern (hourly)
        log_files = list(Path(temp_log_dir).glob("gov_*.jsonl"))
        assert len(log_files) == 1

        # File name should be gov_{hour_timestamp}.jsonl
        file_name = log_files[0].name
        assert file_name.startswith("gov_")
        assert file_name.endswith(".jsonl")

        # The timestamp part should be an integer
        timestamp_part = file_name[4:-6]  # Remove "gov_" and ".jsonl"
        assert timestamp_part.isdigit()

        # Cleanup
        logger._shutdown = True

    def test_get_stats(self, logger_instance):
        """Test that get_stats returns correct statistics."""
        # Initial stats
        stats = logger_instance.get_stats()
        assert stats["entries_logged"] == 0
        assert stats["entries_flushed"] == 0
        assert stats["flush_count"] == 0

        # Log some entries
        for i in range(5):
            logger_instance.log(f"q_{i:03d}", {"index": i})

        stats = logger_instance.get_stats()
        assert stats["entries_logged"] == 5
        assert stats["buffer_size"] == 5

        # Flush
        logger_instance._flush_to_disk()

        stats = logger_instance.get_stats()
        assert stats["entries_flushed"] == 5
        assert stats["flush_count"] == 1
        assert stats["buffer_size"] == 0

    def test_flush_now(self, logger_instance, temp_log_dir):
        """Test that flush_now() forces immediate flush."""
        # Log an entry
        logger_instance.log("q_001", {"test": True})

        # Force flush
        logger_instance.flush_now()

        # Check file was created
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        assert len(log_files) == 1

    def test_thread_safety(self, logger_instance):
        """Test that logging is thread-safe."""
        num_threads = 10
        entries_per_thread = 100
        errors = []

        def log_entries(thread_id):
            try:
                for i in range(entries_per_thread):
                    logger_instance.log(f"q_t{thread_id}_{i}", {"thread": thread_id, "index": i})
            except Exception as e:
                errors.append(e)

        # Start threads
        threads = [
            threading.Thread(target=log_entries, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Stats should reflect logged entries (up to maxlen, since buffer is limited)
        stats = logger_instance.get_stats()
        assert stats["entries_logged"] == num_threads * entries_per_thread

    def test_empty_buffer_flush(self, logger_instance, temp_log_dir):
        """Test that flushing an empty buffer does nothing."""
        # Flush empty buffer
        logger_instance._flush_to_disk()

        # No file should be created
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        assert len(log_files) == 0


class TestBufferedGovernanceLoggerSingleton:
    """Tests for the singleton pattern and convenience functions."""

    def test_get_buffered_governance_logger_creates_instance(self, tmp_path):
        """Test that get_buffered_governance_logger creates a singleton."""
        log_dir = str(tmp_path / "test_logs")

        # Reset singleton for testing
        import vulcan.routing.governance_logger as gov_logger_module
        gov_logger_module._buffered_logger = None

        logger1 = get_buffered_governance_logger(log_path=log_dir)
        logger2 = get_buffered_governance_logger(log_path=log_dir)

        # Should be the same instance
        assert logger1 is logger2

        # Cleanup
        logger1._shutdown = True
        gov_logger_module._buffered_logger = None

    def test_log_routing_result_convenience_function(self, tmp_path):
        """Test the log_routing_result convenience function."""
        log_dir = str(tmp_path / "test_logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Reset singleton for testing
        import vulcan.routing.governance_logger as gov_logger_module
        gov_logger_module._buffered_logger = None

        # Create a logger with this path
        logger = BufferedGovernanceLogger(
            log_path=log_dir,
            buffer_maxlen=100,
            flush_interval=60.0,
        )
        gov_logger_module._buffered_logger = logger

        # Use convenience function
        log_routing_result("q_test", {"route": "reasoning", "complexity": 0.7})

        # Check that entry was logged
        stats = logger.get_stats()
        assert stats["entries_logged"] == 1

        # Cleanup
        logger._shutdown = True
        gov_logger_module._buffered_logger = None


class TestBufferedGovernanceLoggerPerformance:
    """Performance tests for BufferedGovernanceLogger."""

    def test_logging_performance(self, tmp_path):
        """Test that logging 1000 entries is fast (non-blocking)."""
        log_dir = str(tmp_path / "perf_test")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logger = BufferedGovernanceLogger(
            log_path=log_dir,
            buffer_maxlen=1000,
            flush_interval=60.0,
        )

        # Time logging 1000 entries
        start = time.time()
        for i in range(1000):
            logger.log(f"q_{i:04d}", {"index": i, "route": "reasoning", "complexity": 0.5})
        elapsed = time.time() - start

        # Should complete in less than 0.5 seconds (non-blocking)
        assert elapsed < 0.5, f"Logging 1000 entries took {elapsed:.3f}s, should be < 0.5s"

        # Cleanup
        logger._shutdown = True

    def test_logging_does_not_create_files_immediately(self, tmp_path):
        """Test that logging doesn't create files (IO happens in background)."""
        log_dir = str(tmp_path / "no_io_test")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logger = BufferedGovernanceLogger(
            log_path=log_dir,
            buffer_maxlen=1000,
            flush_interval=60.0,
        )

        # Log many entries
        for i in range(100):
            logger.log(f"q_{i:03d}", {"index": i})

        # No files should exist yet
        log_files = list(Path(log_dir).glob("*.jsonl"))
        assert len(log_files) == 0, "No files should be created during logging"

        # Cleanup
        logger._shutdown = True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
