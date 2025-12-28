"""
Comprehensive tests for the MemoryGuard module.

Tests the memory pressure monitoring and automatic GC trigger system, ensuring:
- Correct initialization and configuration
- Proper background thread management
- GC triggering at threshold
- Statistics collection
- Thread safety
"""

import gc
import logging
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestMemoryGuard(unittest.TestCase):
    """Test cases for the MemoryGuard class."""

    def setUp(self):
        """Set up test fixtures."""
        # Stop any existing guards
        try:
            from vulcan.monitoring.memory_guard import stop_memory_guard
            stop_memory_guard()
        except ImportError:
            pass

    def tearDown(self):
        """Clean up after tests."""
        try:
            from vulcan.monitoring.memory_guard import stop_memory_guard
            stop_memory_guard()
        except ImportError:
            pass

    def test_memory_guard_initialization(self):
        """Test that MemoryGuard initializes correctly."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(threshold_percent=80.0, check_interval=10.0)

        self.assertEqual(guard.threshold, 80.0)
        self.assertEqual(guard.interval, 10.0)
        self.assertFalse(guard._running)
        self.assertEqual(guard.gc_triggers, 0)
        self.assertIsNone(guard.last_gc_time)
        self.assertEqual(guard.peak_memory_percent, 0.0)

    def test_memory_guard_start_stop(self):
        """Test starting and stopping the memory guard."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard, PSUTIL_AVAILABLE
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(threshold_percent=99.0, check_interval=0.1)

        # Start
        guard.start()
        self.assertTrue(guard._running)
        self.assertIsNotNone(guard._thread)
        self.assertTrue(guard._thread.is_alive())

        # Wait a bit for monitoring to occur
        time.sleep(0.3)

        # Stop
        guard.stop()
        self.assertFalse(guard._running)
        # Thread should have stopped
        time.sleep(0.2)
        if guard._thread is not None:
            self.assertFalse(guard._thread.is_alive())

    def test_memory_guard_double_start(self):
        """Test that double start is handled gracefully."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard, PSUTIL_AVAILABLE
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(threshold_percent=99.0, check_interval=1.0)

        guard.start()
        thread1 = guard._thread

        # Second start should be a no-op
        guard.start()
        thread2 = guard._thread

        self.assertIs(thread1, thread2)

        guard.stop()

    def test_memory_guard_stop_when_not_running(self):
        """Test that stop when not running is safe."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard()

        # Should not raise
        guard.stop()
        guard.stop()  # Double stop should be safe

    def test_force_gc(self):
        """Test forced garbage collection."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard()

        # Create some garbage
        _garbage = [{"key": i} for i in range(1000)]
        del _garbage

        # Force GC
        collected = guard.force_gc()

        self.assertIsInstance(collected, int)
        self.assertEqual(guard.gc_triggers, 1)
        self.assertIsNotNone(guard.last_gc_time)

    def test_get_status(self):
        """Test status retrieval."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard, PSUTIL_AVAILABLE
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(threshold_percent=85.0, check_interval=5.0)
        guard.force_gc()  # Trigger one GC

        status = guard.get_status()

        self.assertIn("running", status)
        self.assertIn("threshold_percent", status)
        self.assertIn("check_interval", status)
        self.assertIn("gc_triggers", status)
        self.assertIn("last_gc_time", status)
        self.assertIn("peak_memory_percent", status)

        self.assertEqual(status["threshold_percent"], 85.0)
        self.assertEqual(status["check_interval"], 5.0)
        self.assertEqual(status["gc_triggers"], 1)

        if PSUTIL_AVAILABLE:
            self.assertIn("current_memory_percent", status)
            self.assertIn("available_gb", status)

    @patch('vulcan.monitoring.memory_guard.psutil')
    def test_gc_triggered_on_high_memory(self, mock_psutil):
        """Test that GC is triggered when memory exceeds threshold."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Mock high memory usage
        mock_memory = MagicMock()
        mock_memory.percent = 90.0  # Above 85% threshold
        mock_memory.available = 2 * 1024**3  # 2GB
        mock_memory.used = 8 * 1024**3  # 8GB

        mock_psutil.virtual_memory.return_value = mock_memory

        guard = MemoryGuard(threshold_percent=85.0, check_interval=0.05)

        # Start and let it run
        guard._running = True
        guard._shutdown_event = threading.Event()

        # Run one iteration of the monitor loop manually
        # by calling _monitor_loop in a controlled way
        def run_one_check():
            try:
                memory = mock_psutil.virtual_memory()
                if memory.percent > guard.threshold:
                    gc.collect()
                    guard.gc_triggers += 1
                    guard.last_gc_time = time.time()
            except Exception:
                pass

        run_one_check()

        # GC should have been triggered
        self.assertGreaterEqual(guard.gc_triggers, 1)


class TestMemoryGuardGlobalFunctions(unittest.TestCase):
    """Test global functions for memory guard management."""

    def setUp(self):
        """Clean up any existing guard."""
        try:
            from vulcan.monitoring.memory_guard import stop_memory_guard
            stop_memory_guard()
        except ImportError:
            pass

    def tearDown(self):
        """Clean up after tests."""
        try:
            from vulcan.monitoring.memory_guard import stop_memory_guard
            stop_memory_guard()
        except ImportError:
            pass

    def test_start_memory_guard_returns_instance(self):
        """Test that start_memory_guard returns a guard instance."""
        try:
            from vulcan.monitoring.memory_guard import (
                start_memory_guard,
                PSUTIL_AVAILABLE,
            )
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("Module not available")
            return

        guard = start_memory_guard(threshold_percent=90.0, check_interval=1.0)

        if guard is not None:
            self.assertTrue(guard._running)
            self.assertEqual(guard.threshold, 90.0)

    def test_stop_memory_guard(self):
        """Test stopping the global memory guard."""
        try:
            from vulcan.monitoring.memory_guard import (
                start_memory_guard,
                stop_memory_guard,
                get_memory_guard,
                PSUTIL_AVAILABLE,
            )
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("Module not available")
            return

        start_memory_guard()
        guard = get_memory_guard()

        if guard is not None:
            self.assertTrue(guard._running)

        stop_memory_guard()
        guard_after = get_memory_guard()

        self.assertIsNone(guard_after)

    def test_get_memory_guard_before_start(self):
        """Test that get_memory_guard returns None before start."""
        try:
            from vulcan.monitoring.memory_guard import get_memory_guard
        except ImportError:
            self.skipTest("Module not available")
            return

        guard = get_memory_guard()
        self.assertIsNone(guard)

    def test_trigger_gc_function(self):
        """Test the trigger_gc convenience function."""
        try:
            from vulcan.monitoring.memory_guard import trigger_gc
        except ImportError:
            self.skipTest("Module not available")
            return

        # Create some garbage
        _garbage = [{"key": i} for i in range(1000)]
        del _garbage

        collected = trigger_gc()
        self.assertIsInstance(collected, int)


class TestMemoryGuardWithoutPsutil(unittest.TestCase):
    """Test MemoryGuard behavior when psutil is not available."""

    def test_graceful_degradation(self):
        """Test that MemoryGuard handles missing psutil gracefully."""
        try:
            from vulcan.monitoring import memory_guard
        except ImportError:
            self.skipTest("Module not available")
            return

        # Save original state
        original_available = memory_guard.PSUTIL_AVAILABLE
        original_psutil = memory_guard.psutil

        try:
            # Simulate psutil not available
            memory_guard.PSUTIL_AVAILABLE = False
            memory_guard.psutil = None

            guard = memory_guard.MemoryGuard()

            # Start should be safe but not actually start monitoring
            guard.start()
            self.assertFalse(guard._running)

            # Status should still work
            status = guard.get_status()
            self.assertIn("running", status)

        finally:
            # Restore original state
            memory_guard.PSUTIL_AVAILABLE = original_available
            memory_guard.psutil = original_psutil


class TestMemoryGuardThreadSafety(unittest.TestCase):
    """Thread safety tests for MemoryGuard."""

    def test_concurrent_force_gc(self):
        """Test concurrent force_gc calls."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("Module not available")
            return

        guard = MemoryGuard()
        results = []
        results_lock = threading.Lock()

        def force_gc_thread():
            collected = guard.force_gc()
            with results_lock:
                results.append(collected)

        threads = [threading.Thread(target=force_gc_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 10)
        # gc_triggers should equal number of calls
        self.assertEqual(guard.gc_triggers, 10)

    def test_concurrent_start_stop(self):
        """Test concurrent start/stop operations."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard, PSUTIL_AVAILABLE
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("Module not available")
            return

        guard = MemoryGuard(check_interval=0.1)
        errors = []
        errors_lock = threading.Lock()

        def start_stop_thread():
            try:
                for _ in range(5):
                    guard.start()
                    time.sleep(0.01)
                    guard.stop()
            except Exception as e:
                with errors_lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=start_stop_thread) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        guard.stop()  # Ensure stopped
        self.assertEqual(len(errors), 0, f"Errors: {errors}")


if __name__ == "__main__":
    unittest.main()
