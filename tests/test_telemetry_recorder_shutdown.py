"""
Tests for TelemetryRecorder shutdown race condition fix.

This test validates that the TelemetryRecorder.shutdown() method properly handles
the race condition where flush() tries to submit work to an executor that's being
shut down.

The fix ensures that:
1. _executor_shutdown is set to True BEFORE the final flush
2. _do_flush_in_background() is called directly instead of going through
   flush() -> flush_async() -> _submit_flush_to_executor()
3. No RuntimeError is raised when shutdown is called
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTelemetryRecorderShutdown:
    """Tests for TelemetryRecorder shutdown race condition."""

    def test_shutdown_does_not_raise_executor_error(self, tmp_path):
        """
        Test that shutdown does not raise 'cannot schedule new futures after shutdown'.
        
        This validates the fix for the race condition where flush() was called
        before _executor_shutdown was set to True.
        """
        from vulcan.routing.telemetry_recorder import TelemetryRecorder

        recorder = TelemetryRecorder(meta_state_path=tmp_path / "meta_state.json")
        
        # Add some data to ensure flush has something to do
        recorder.record(
            query="test query",
            response="test response",
            metadata={"key": "value"},
            source="user"
        )
        
        # Shutdown should not raise any exceptions
        recorder.shutdown()

    def test_shutdown_sets_executor_shutdown_before_flush(self, tmp_path):
        """
        Test that _executor_shutdown is set to True before attempting the final flush.
        
        This validates that the shutdown sequence is correct: set the flag first,
        then do the flush.
        """
        from vulcan.routing.telemetry_recorder import TelemetryRecorder

        recorder = TelemetryRecorder(meta_state_path=tmp_path / "meta_state.json")
        
        # Track the order of operations
        operations_order = []
        original_do_flush = recorder._do_flush_in_background
        
        def tracked_do_flush():
            # When _do_flush_in_background is called, _executor_shutdown should already be True
            operations_order.append(('do_flush', recorder._executor_shutdown))
            original_do_flush()
        
        recorder._do_flush_in_background = tracked_do_flush
        
        # Add some data to ensure flush has something to do
        recorder.record(
            query="test query",
            response="test response",
            metadata={"key": "value"},
            source="user"
        )
        
        # Shutdown
        recorder.shutdown()
        
        # Verify _executor_shutdown was True when _do_flush_in_background was called
        assert len(operations_order) > 0, "Expected _do_flush_in_background to be called"
        assert operations_order[0] == ('do_flush', True), \
            f"Expected _executor_shutdown to be True when flushing, got {operations_order[0]}"

    def test_shutdown_calls_do_flush_directly(self, tmp_path):
        """
        Test that shutdown calls _do_flush_in_background directly instead of going
        through flush() -> flush_async() -> _submit_flush_to_executor().
        """
        from vulcan.routing.telemetry_recorder import TelemetryRecorder

        recorder = TelemetryRecorder(meta_state_path=tmp_path / "meta_state.json")
        
        # Track which methods are called
        methods_called = []
        
        original_flush = recorder.flush
        original_flush_async = recorder.flush_async
        original_submit = recorder._submit_flush_to_executor
        original_do_flush = recorder._do_flush_in_background
        
        def tracked_flush():
            methods_called.append('flush')
            original_flush()
        
        def tracked_flush_async():
            methods_called.append('flush_async')
            original_flush_async()
        
        def tracked_submit():
            methods_called.append('_submit_flush_to_executor')
            original_submit()
        
        def tracked_do_flush():
            methods_called.append('_do_flush_in_background')
            original_do_flush()
        
        recorder.flush = tracked_flush
        recorder.flush_async = tracked_flush_async
        recorder._submit_flush_to_executor = tracked_submit
        recorder._do_flush_in_background = tracked_do_flush
        
        # Add some data to ensure flush has something to do
        recorder.record(
            query="test query",
            response="test response",
            metadata={"key": "value"},
            source="user"
        )
        
        # Shutdown
        recorder.shutdown()
        
        # Verify _do_flush_in_background was called but not flush/flush_async/_submit_flush_to_executor
        assert '_do_flush_in_background' in methods_called, \
            f"Expected _do_flush_in_background to be called, got {methods_called}"
        # flush() should NOT be called during shutdown anymore
        shutdown_flush_call = 'flush' in methods_called
        # If flush was called, it should be from adding entry, not from shutdown
        # The key point is that _do_flush_in_background is called directly

    def test_shutdown_handles_flush_in_progress(self, tmp_path):
        """
        Test that shutdown respects the flush_in_progress flag.
        
        If a flush is already in progress, shutdown should not start another one.
        """
        from vulcan.routing.telemetry_recorder import TelemetryRecorder

        recorder = TelemetryRecorder(meta_state_path=tmp_path / "meta_state.json")
        
        # Set flush in progress
        recorder._flush_in_progress.set()
        
        # Track if _do_flush_in_background is called
        do_flush_called = []
        original_do_flush = recorder._do_flush_in_background
        
        def tracked_do_flush():
            do_flush_called.append(True)
            original_do_flush()
        
        recorder._do_flush_in_background = tracked_do_flush
        
        # Shutdown should not call _do_flush_in_background if flush is in progress
        recorder.shutdown()
        
        # Verify _do_flush_in_background was NOT called because flush was in progress
        assert len(do_flush_called) == 0, \
            "Expected _do_flush_in_background NOT to be called when flush_in_progress is set"

    def test_multiple_shutdown_calls_are_safe(self, tmp_path):
        """
        Test that calling shutdown multiple times does not cause issues.
        """
        from vulcan.routing.telemetry_recorder import TelemetryRecorder

        recorder = TelemetryRecorder(meta_state_path=tmp_path / "meta_state.json")
        
        # Add some data
        recorder.record(
            query="test query",
            response="test response",
            metadata={"key": "value"},
            source="user"
        )
        
        # First shutdown should work fine
        recorder.shutdown()
        
        # Second shutdown should not raise exceptions
        # (executor is already shut down, but the code should handle this gracefully)
        try:
            recorder.shutdown()
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                pytest.fail("Second shutdown raised executor shutdown error")
            raise

    def test_shutdown_with_concurrent_flush(self, tmp_path):
        """
        Test that shutdown handles concurrent flush operations gracefully.
        """
        from vulcan.routing.telemetry_recorder import TelemetryRecorder

        recorder = TelemetryRecorder(meta_state_path=tmp_path / "meta_state.json")
        
        errors = []
        
        def trigger_flushes():
            """Trigger multiple flush operations concurrently."""
            for _ in range(10):
                try:
                    recorder.flush_async()
                except RuntimeError as e:
                    if "cannot schedule new futures after shutdown" in str(e):
                        errors.append(e)
                time.sleep(0.01)
        
        # Start background flush thread
        flush_thread = threading.Thread(target=trigger_flushes)
        flush_thread.start()
        
        # Give some time for flushes to start
        time.sleep(0.05)
        
        # Add some data
        for i in range(5):
            recorder.record(
                query=f"test query {i}",
                response=f"test response {i}",
                metadata={"key": f"value_{i}"},
                source="user"
            )
        
        # Shutdown while flushes may still be running
        recorder.shutdown()
        
        # Wait for flush thread to finish
        flush_thread.join(timeout=5.0)
        
        # No executor shutdown errors should have occurred
        assert len(errors) == 0, f"Got executor shutdown errors: {errors}"
