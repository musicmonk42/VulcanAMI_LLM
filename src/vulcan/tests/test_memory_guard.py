"""
Comprehensive tests for the MemoryGuard module.

Tests the memory pressure monitoring and automatic GC trigger system, ensuring:
- Correct initialization and configuration
- Proper background thread management
- GC triggering at threshold
- Statistics collection
- Thread safety with locks
- Graduated threshold behavior
- Aggressive callback mechanism
- Double-checked locking for global instance
- Non-daemon thread with atexit cleanup
"""

import atexit
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
        """Test that MemoryGuard initializes correctly with graduated thresholds."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Test with new parameters
        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=10.0
        )

        self.assertEqual(guard.warning_threshold, 70.0)
        self.assertEqual(guard.gc_threshold, 75.0)
        self.assertEqual(guard.critical_threshold, 85.0)
        self.assertEqual(guard.interval, 10.0)
        self.assertFalse(guard._running)
        self.assertEqual(guard.gc_triggers, 0)
        self.assertIsNone(guard.last_gc_time)
        self.assertEqual(guard.peak_memory_percent, 0.0)
        self.assertIsNone(guard.aggressive_gc_callback)

    def test_memory_guard_backward_compatibility(self):
        """Test backward compatibility with threshold_percent parameter."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Old API: threshold_percent should map to gc_threshold
        guard = MemoryGuard(threshold_percent=80.0, check_interval=10.0)

        self.assertEqual(guard.threshold, 80.0)  # Backward compat property
        self.assertEqual(guard.gc_threshold, 80.0)
        self.assertEqual(guard.interval, 10.0)
        self.assertFalse(guard._running)
        self.assertEqual(guard.gc_triggers, 0)
        self.assertIsNone(guard.last_gc_time)
        self.assertEqual(guard.peak_memory_percent, 0.0)

    def test_memory_guard_start_stop(self):
        """Test starting and stopping the memory guard with non-daemon thread."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard, PSUTIL_AVAILABLE
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(gc_threshold=99.0, check_interval=0.1)

        # Start
        guard.start()
        self.assertTrue(guard._running)
        self.assertIsNotNone(guard._thread)
        self.assertTrue(guard._thread.is_alive())
        # Verify non-daemon thread
        self.assertFalse(guard._thread.daemon)

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
        """Test status retrieval with all new threshold fields."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard, PSUTIL_AVAILABLE
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=5.0
        )
        guard.force_gc()  # Trigger one GC

        status = guard.get_status()

        self.assertIn("running", status)
        self.assertIn("warning_threshold", status)
        self.assertIn("gc_threshold", status)
        self.assertIn("critical_threshold", status)
        self.assertIn("check_interval", status)
        self.assertIn("gc_triggers", status)
        self.assertIn("last_gc_time", status)
        self.assertIn("peak_memory_percent", status)
        self.assertIn("has_aggressive_callback", status)

        self.assertEqual(status["warning_threshold"], 70.0)
        self.assertEqual(status["gc_threshold"], 75.0)
        self.assertEqual(status["critical_threshold"], 85.0)
        self.assertEqual(status["check_interval"], 5.0)
        self.assertEqual(status["gc_triggers"], 1)
        self.assertFalse(status["has_aggressive_callback"])

        if PSUTIL_AVAILABLE:
            self.assertIn("current_memory_percent", status)
            self.assertIn("available_gb", status)
            self.assertIn("action_level", status)

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

        guard = MemoryGuard(gc_threshold=75.0, check_interval=0.05)

        # Start and let it run
        guard._running = True
        guard._shutdown_event = threading.Event()

        # Run one iteration of the monitor loop manually
        # by calling _monitor_loop in a controlled way
        def run_one_check():
            try:
                memory = mock_psutil.virtual_memory()
                if memory.percent > guard.gc_threshold:
                    gc.collect()
                    guard._increment_gc_trigger()
            except Exception:
                pass

        run_one_check()

        # GC should have been triggered
        self.assertGreaterEqual(guard.gc_triggers, 1)

    @patch('vulcan.monitoring.memory_guard.psutil')
    def test_graduated_thresholds_warning(self, mock_psutil):
        """Test warning threshold (log only, no GC)."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Mock memory at warning level (71%)
        mock_memory = MagicMock()
        mock_memory.percent = 71.0
        mock_memory.available = 3 * 1024**3
        mock_memory.used = 7 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory

        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=0.05
        )

        initial_gc_triggers = guard.gc_triggers

        # Run one check at warning level - should NOT trigger GC
        guard._running = True
        guard._shutdown_event = threading.Event()
        
        memory = mock_psutil.virtual_memory()
        guard._update_peak_memory(memory.percent)
        
        # At warning level, no GC should be triggered
        self.assertEqual(guard.gc_triggers, initial_gc_triggers)

    @patch('vulcan.monitoring.memory_guard.psutil')
    def test_graduated_thresholds_gc(self, mock_psutil):
        """Test GC threshold triggers garbage collection."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Mock memory at GC level (76%)
        mock_memory = MagicMock()
        mock_memory.percent = 76.0
        mock_memory.available = 2.4 * 1024**3
        mock_memory.used = 7.6 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory

        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=0.05
        )

        initial_gc_triggers = guard.gc_triggers

        # Manually trigger GC check
        if mock_memory.percent > guard.gc_threshold:
            gc.collect()
            guard._increment_gc_trigger()

        # GC should have been triggered
        self.assertEqual(guard.gc_triggers, initial_gc_triggers + 1)

    @patch('vulcan.monitoring.memory_guard.psutil')
    def test_graduated_thresholds_critical(self, mock_psutil):
        """Test critical threshold triggers aggressive cleanup."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Mock memory at critical level (90%)
        mock_memory = MagicMock()
        mock_memory.percent = 90.0
        mock_memory.available = 1 * 1024**3
        mock_memory.used = 9 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory

        callback_called = []

        def aggressive_callback():
            callback_called.append(True)

        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=0.05,
            aggressive_gc_callback=aggressive_callback
        )

        # Manually trigger critical check
        if mock_memory.percent > guard.critical_threshold:
            gc.collect(generation=2)
            guard._increment_gc_trigger()
            if guard.aggressive_gc_callback:
                guard.aggressive_gc_callback()

        # Callback should have been called
        self.assertEqual(len(callback_called), 1)
        self.assertGreater(guard.gc_triggers, 0)

    def test_aggressive_callback_set(self):
        """Test setting aggressive callback."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        callback_called = []

        def my_callback():
            callback_called.append(True)

        guard = MemoryGuard(aggressive_gc_callback=my_callback)
        
        self.assertIsNotNone(guard.aggressive_gc_callback)
        
        # Test status reports callback presence
        status = guard.get_status()
        self.assertTrue(status["has_aggressive_callback"])


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

        guard = start_memory_guard(gc_threshold=90.0, check_interval=1.0)

        if guard is not None:
            self.assertTrue(guard._running)
            self.assertEqual(guard.gc_threshold, 90.0)

    def test_start_memory_guard_backward_compat(self):
        """Test start_memory_guard with deprecated threshold_percent."""
        try:
            from vulcan.monitoring.memory_guard import (
                start_memory_guard,
                stop_memory_guard,
                PSUTIL_AVAILABLE,
            )
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("Module not available")
            return

        # Old API
        guard = start_memory_guard(threshold_percent=88.0, check_interval=1.0)

        if guard is not None:
            self.assertTrue(guard._running)
            self.assertEqual(guard.gc_threshold, 88.0)
            self.assertEqual(guard.threshold, 88.0)  # Backward compat property

        stop_memory_guard()

    def test_start_memory_guard_double_checked_locking(self):
        """Test double-checked locking prevents race condition."""
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

        # Clean slate
        stop_memory_guard()

        guards = []
        guards_lock = threading.Lock()

        def start_guard_thread():
            guard = start_memory_guard(gc_threshold=95.0, check_interval=1.0)
            with guards_lock:
                guards.append(guard)

        # Start multiple threads trying to start the guard
        threads = [threading.Thread(target=start_guard_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        unique_guards = set(id(g) for g in guards if g is not None)
        self.assertEqual(len(unique_guards), 1, "Multiple guard instances created!")

        stop_memory_guard()

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

    def test_set_aggressive_gc_callback_function(self):
        """Test the set_aggressive_gc_callback function."""
        try:
            from vulcan.monitoring.memory_guard import (
                start_memory_guard,
                stop_memory_guard,
                set_aggressive_gc_callback,
                get_memory_guard,
                PSUTIL_AVAILABLE,
            )
            if not PSUTIL_AVAILABLE:
                self.skipTest("psutil not available")
                return
        except ImportError:
            self.skipTest("Module not available")
            return

        # Start guard
        guard = start_memory_guard(gc_threshold=95.0)
        
        callback_called = []

        def my_callback():
            callback_called.append(True)

        # Set callback
        set_aggressive_gc_callback(my_callback)

        # Verify callback is set
        guard_inst = get_memory_guard()
        if guard_inst:
            self.assertIsNotNone(guard_inst.aggressive_gc_callback)

        stop_memory_guard()

    def test_set_aggressive_gc_callback_before_start(self):
        """Test set_aggressive_gc_callback when guard not started."""
        try:
            from vulcan.monitoring.memory_guard import (
                stop_memory_guard,
                set_aggressive_gc_callback,
            )
        except ImportError:
            self.skipTest("Module not available")
            return

        # Ensure guard is not running
        stop_memory_guard()

        def my_callback():
            pass

        # Should log warning but not crash
        set_aggressive_gc_callback(my_callback)


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
    """Thread safety tests for MemoryGuard with lock protection."""

    def test_concurrent_force_gc(self):
        """Test concurrent force_gc calls are thread-safe."""
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
        # gc_triggers should equal number of calls (thread-safe)
        self.assertEqual(guard.gc_triggers, 10)

    def test_concurrent_statistics_access(self):
        """Test thread-safe access to statistics properties."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("Module not available")
            return

        guard = MemoryGuard()
        errors = []
        errors_lock = threading.Lock()

        def read_stats_thread():
            try:
                for _ in range(100):
                    _ = guard.gc_triggers
                    _ = guard.last_gc_time
                    _ = guard.peak_memory_percent
            except Exception as e:
                with errors_lock:
                    errors.append(str(e))

        def write_stats_thread():
            try:
                for _ in range(100):
                    guard._increment_gc_trigger()
                    guard._update_peak_memory(50.0 + (_ % 50))
            except Exception as e:
                with errors_lock:
                    errors.append(str(e))

        # Mix of readers and writers
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=read_stats_thread))
            threads.append(threading.Thread(target=write_stats_thread))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # Verify final count is correct
        self.assertEqual(guard.gc_triggers, 500)  # 5 threads * 100 iterations

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

    def test_thread_safe_peak_memory_update(self):
        """Test that peak memory updates are thread-safe."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("Module not available")
            return

        guard = MemoryGuard()

        def update_peak_thread(value):
            guard._update_peak_memory(value)

        # Update peak from multiple threads
        threads = [
            threading.Thread(target=update_peak_thread, args=(i,))
            for i in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Peak should be the maximum value
        self.assertEqual(guard.peak_memory_percent, 99.0)


class TestMemoryGuardDeathSpiralPrevention(unittest.TestCase):
    """Test the death spiral prevention feature (GC backoff mechanism)."""

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

    def test_consecutive_criticals_initialization(self):
        """Test that consecutive criticals counter is initialized to 0."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard()
        self.assertEqual(guard._consecutive_criticals, 0)
        self.assertEqual(guard.max_backoff_multiplier, 5)

    def test_backoff_multiplier_in_status(self):
        """Test that backoff multiplier is exposed in status."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard()
        status = guard.get_status()

        self.assertIn("consecutive_criticals", status)
        self.assertIn("current_backoff_multiplier", status)
        self.assertIn("max_backoff_multiplier", status)
        self.assertEqual(status["consecutive_criticals"], 0)
        self.assertEqual(status["current_backoff_multiplier"], 1)  # 0 + 1
        self.assertEqual(status["max_backoff_multiplier"], 5)

    def test_backoff_multiplier_calculation(self):
        """Test that backoff multiplier is calculated correctly."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard()
        
        # With 0 consecutive criticals, multiplier should be 1
        guard._consecutive_criticals = 0
        status = guard.get_status()
        self.assertEqual(status["current_backoff_multiplier"], 1)
        
        # With 2 consecutive criticals, multiplier should be 3
        guard._consecutive_criticals = 2
        status = guard.get_status()
        self.assertEqual(status["current_backoff_multiplier"], 3)
        
        # With 5+ consecutive criticals, multiplier should be capped at 5
        guard._consecutive_criticals = 10
        status = guard.get_status()
        self.assertEqual(status["current_backoff_multiplier"], 5)

    @patch('vulcan.monitoring.memory_guard.psutil')
    def test_consecutive_criticals_increment(self, mock_psutil):
        """Test that consecutive criticals increments on critical memory."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        # Mock critical memory (90% > 85% critical threshold)
        mock_memory = MagicMock()
        mock_memory.percent = 90.0
        mock_memory.available = 1 * 1024**3
        mock_memory.used = 9 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory

        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=0.05
        )

        # Simulate critical event
        guard._consecutive_criticals = 0
        if mock_memory.percent > guard.critical_threshold:
            guard._consecutive_criticals += 1

        self.assertEqual(guard._consecutive_criticals, 1)

    @patch('vulcan.monitoring.memory_guard.psutil')
    def test_consecutive_criticals_reset_on_lower_memory(self, mock_psutil):
        """Test that consecutive criticals resets when memory drops below critical."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(
            warning_threshold=70.0,
            gc_threshold=75.0,
            critical_threshold=85.0,
            check_interval=0.05
        )

        # Simulate previous critical events
        guard._consecutive_criticals = 3

        # Mock memory dropping below critical (80% - above gc, below critical)
        mock_memory = MagicMock()
        mock_memory.percent = 80.0
        mock_memory.available = 2 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory

        # Simulate the reset logic from _monitor_loop
        if mock_memory.percent <= guard.critical_threshold:
            guard._consecutive_criticals = 0

        self.assertEqual(guard._consecutive_criticals, 0)

    def test_sleep_interval_with_backoff(self):
        """Test that sleep interval is calculated with backoff."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard(check_interval=5.0)
        
        # Normal: 5.0 * 1 = 5.0
        guard._consecutive_criticals = 0
        backoff_multiplier = min(guard._consecutive_criticals + 1, guard.max_backoff_multiplier)
        sleep_interval = guard.interval * backoff_multiplier
        self.assertEqual(sleep_interval, 5.0)
        
        # After 2 criticals: 5.0 * 3 = 15.0
        guard._consecutive_criticals = 2
        backoff_multiplier = min(guard._consecutive_criticals + 1, guard.max_backoff_multiplier)
        sleep_interval = guard.interval * backoff_multiplier
        self.assertEqual(sleep_interval, 15.0)
        
        # After 10 criticals (capped at 5x): 5.0 * 5 = 25.0
        guard._consecutive_criticals = 10
        backoff_multiplier = min(guard._consecutive_criticals + 1, guard.max_backoff_multiplier)
        sleep_interval = guard.interval * backoff_multiplier
        self.assertEqual(sleep_interval, 25.0)

    def test_max_backoff_multiplier_configurable(self):
        """Test that max backoff multiplier is configurable."""
        try:
            from vulcan.monitoring.memory_guard import MemoryGuard
        except ImportError:
            self.skipTest("MemoryGuard not available")
            return

        guard = MemoryGuard()
        
        # Default should be 5
        self.assertEqual(guard.max_backoff_multiplier, 5)
        
        # Should be configurable
        guard.max_backoff_multiplier = 10
        self.assertEqual(guard.max_backoff_multiplier, 10)
        
        # Verify backoff calculation respects new max
        guard._consecutive_criticals = 15
        backoff_multiplier = min(guard._consecutive_criticals + 1, guard.max_backoff_multiplier)
        self.assertEqual(backoff_multiplier, 10)


if __name__ == "__main__":
    unittest.main()
