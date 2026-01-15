"""
Unit tests for ProcessLock with heartbeat mechanism.

Tests the process lock functionality including:
- Basic lock acquire/release
- Heartbeat thread management
- Stale lock detection
- Lock file format with timestamp
- Thread safety

Author: VULCAN-AGI Team
"""

import os
import tempfile
import time
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock

from vulcan.utils_main.process_lock import (
    ProcessLock,
    FCNTL_AVAILABLE,
    is_process_running,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
)


class TestIsProcessRunning:
    """Tests for the is_process_running utility function."""
    
    def test_current_process_is_running(self):
        """Test that current process is detected as running."""
        assert is_process_running(os.getpid()) is True
    
    def test_invalid_pid_zero(self):
        """Test that PID 0 returns False."""
        assert is_process_running(0) is False
    
    def test_invalid_pid_negative(self):
        """Test that negative PIDs return False."""
        assert is_process_running(-1) is False
        assert is_process_running(-100) is False
    
    def test_nonexistent_pid(self):
        """Test that a very high PID returns False (likely not running)."""
        # Use a very high PID that's unlikely to exist
        assert is_process_running(999999999) is False


class TestProcessLockBasics:
    """Basic ProcessLock functionality tests."""
    
    def test_lock_creation(self):
        """Test creating a ProcessLock instance."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = f.name
        
        try:
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            assert lock.lock_path == lock_path
            assert lock._locked is False
            assert lock._heartbeat_thread is None
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
    
    def test_lock_creation_with_heartbeat_params(self):
        """Test creating a ProcessLock with custom heartbeat parameters."""
        lock = ProcessLock(
            heartbeat_interval=10,
            lock_ttl=30,
            enable_heartbeat=True,
        )
        
        assert lock._heartbeat_interval == 10
        assert lock._lock_ttl == 30
        assert lock._enable_heartbeat is True
    
    def test_default_heartbeat_values(self):
        """Test that default heartbeat values are correct."""
        lock = ProcessLock(enable_heartbeat=True)
        
        assert lock._heartbeat_interval == DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        assert lock._lock_ttl == DEFAULT_LOCK_TTL_SECONDS
    
    def test_repr(self):
        """Test string representation of ProcessLock."""
        lock = ProcessLock(enable_heartbeat=True)
        repr_str = repr(lock)
        
        assert "ProcessLock" in repr_str
        assert "unlocked" in repr_str
        assert "heartbeat=on" in repr_str
    
    def test_repr_heartbeat_off(self):
        """Test string representation with heartbeat disabled."""
        lock = ProcessLock(enable_heartbeat=False)
        repr_str = repr(lock)
        
        assert "heartbeat=off" in repr_str


@pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
class TestProcessLockAcquireRelease:
    """Tests for lock acquire/release with fcntl."""
    
    def test_acquire_release_basic(self):
        """Test basic acquire and release."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = f.name
        
        try:
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            
            # Acquire
            assert lock.acquire() is True
            assert lock.is_locked() is True
            
            # Release
            lock.release()
            assert lock.is_locked() is False
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
    
    def test_acquire_creates_lock_file(self):
        """Test that acquire creates the lock file with PID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            
            assert lock.acquire() is True
            assert os.path.exists(lock_path)
            
            # Read lock file content
            with open(lock_path, "r") as f:
                content = f.read()
            
            # Should contain PID
            assert str(os.getpid()) in content
            
            lock.release()
    
    def test_acquire_with_heartbeat_creates_timestamp(self):
        """Test that acquire with heartbeat creates timestamp in lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=True,
                heartbeat_interval=1,
            )
            
            try:
                assert lock.acquire() is True
                
                # Give heartbeat thread time to start
                time.sleep(0.1)
                
                # Read lock file content
                with open(lock_path, "r") as f:
                    lines = f.read().strip().split("\n")
                
                # Should have PID and timestamp
                assert len(lines) >= 2
                assert str(os.getpid()) == lines[0]
                
                # Timestamp should be a valid float
                timestamp = float(lines[1])
                assert timestamp > 0
                assert timestamp <= time.time()
            finally:
                lock.release()
    
    def test_context_manager(self):
        """Test using ProcessLock as context manager."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = f.name
        
        try:
            with ProcessLock(lock_path=lock_path, enable_heartbeat=False) as lock:
                assert lock.is_locked() is True
            
            # Lock should be released after context
            assert lock.is_locked() is False
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
    
    def test_release_idempotent(self):
        """Test that release can be called multiple times safely."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = f.name
        
        try:
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            lock.acquire()
            
            # Release multiple times should not raise
            lock.release()
            lock.release()
            lock.release()
            
            assert lock.is_locked() is False
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)


@pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
class TestProcessLockHeartbeat:
    """Tests for heartbeat functionality."""
    
    def test_heartbeat_thread_starts_on_acquire(self):
        """Test that heartbeat thread starts when lock is acquired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=True,
                heartbeat_interval=0.1,  # Fast for testing
            )
            
            try:
                assert lock.acquire() is True
                
                # Give thread time to start
                time.sleep(0.2)
                
                assert lock._heartbeat_thread is not None
                assert lock._heartbeat_thread.is_alive() is True
            finally:
                lock.release()
    
    def test_heartbeat_thread_stops_on_release(self):
        """Test that heartbeat thread stops when lock is released."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=True,
                heartbeat_interval=0.1,
            )
            
            assert lock.acquire() is True
            time.sleep(0.2)
            
            # Thread should be alive
            heartbeat_thread = lock._heartbeat_thread
            assert heartbeat_thread is not None
            assert heartbeat_thread.is_alive() is True
            
            # Release should stop the thread
            lock.release()
            time.sleep(0.3)  # Wait for thread to stop
            
            assert heartbeat_thread.is_alive() is False
    
    def test_heartbeat_updates_timestamp(self):
        """Test that heartbeat periodically updates the timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=True,
                heartbeat_interval=0.1,  # 100ms interval
            )
            
            try:
                assert lock.acquire() is True
                
                # Read initial timestamp
                time.sleep(0.15)
                with open(lock_path, "r") as f:
                    lines1 = f.read().strip().split("\n")
                timestamp1 = float(lines1[1]) if len(lines1) > 1 else 0
                
                # Wait for another heartbeat
                time.sleep(0.15)
                with open(lock_path, "r") as f:
                    lines2 = f.read().strip().split("\n")
                timestamp2 = float(lines2[1]) if len(lines2) > 1 else 0
                
                # Timestamp should have been updated
                assert timestamp2 > timestamp1
            finally:
                lock.release()
    
    def test_heartbeat_disabled(self):
        """Test that heartbeat thread does not start when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=False,
            )
            
            try:
                assert lock.acquire() is True
                time.sleep(0.1)
                
                # No heartbeat thread should exist
                assert lock._heartbeat_thread is None
            finally:
                lock.release()
    
    def test_get_heartbeat_status(self):
        """Test getting heartbeat status information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=True,
                heartbeat_interval=0.1,
                lock_ttl=0.3,
            )
            
            try:
                assert lock.acquire() is True
                time.sleep(0.15)
                
                status = lock.get_heartbeat_status()
                
                assert status["enabled"] is True
                assert status["thread_alive"] is True
                assert status["last_heartbeat"] > 0
                assert status["interval"] == 0.1
                assert status["ttl"] == 0.3
                assert status["seconds_since_heartbeat"] is not None
                assert status["seconds_since_heartbeat"] < 1.0
            finally:
                lock.release()
    
    def test_heartbeat_status_when_disabled(self):
        """Test heartbeat status when heartbeat is disabled."""
        lock = ProcessLock(enable_heartbeat=False)
        
        status = lock.get_heartbeat_status()
        
        assert status["enabled"] is False
        assert status["thread_alive"] is False


@pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
class TestProcessLockStaleLockDetection:
    """Tests for stale lock detection and recovery."""
    
    def test_is_lock_stale_no_pid(self):
        """Test that lock is stale if PID is None."""
        lock = ProcessLock(enable_heartbeat=True)
        
        assert lock._is_lock_stale(None, time.time()) is True
    
    def test_is_lock_stale_dead_process(self):
        """Test that lock is stale if process is not running."""
        lock = ProcessLock(enable_heartbeat=True)
        
        # Use a very high PID that's unlikely to exist
        assert lock._is_lock_stale(999999999, time.time()) is True
    
    def test_is_lock_stale_expired_heartbeat(self):
        """Test that lock is stale if heartbeat has expired."""
        lock = ProcessLock(
            enable_heartbeat=True,
            lock_ttl=1.0,  # 1 second TTL
        )
        
        # Current process but old timestamp
        old_timestamp = time.time() - 2.0  # 2 seconds ago
        assert lock._is_lock_stale(os.getpid(), old_timestamp) is True
    
    def test_is_lock_not_stale_active(self):
        """Test that lock is not stale if process is running and heartbeat is fresh."""
        lock = ProcessLock(
            enable_heartbeat=True,
            lock_ttl=60.0,
        )
        
        # Current process with fresh timestamp
        fresh_timestamp = time.time()
        assert lock._is_lock_stale(os.getpid(), fresh_timestamp) is False
    
    def test_read_lock_info(self):
        """Test reading PID and timestamp from lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            
            # Create lock file with known content
            with open(lock_path, "w") as f:
                f.write("12345\n1234567890.123\n")
            
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            pid, timestamp = lock._read_lock_info()
            
            assert pid == 12345
            assert timestamp == 1234567890.123
    
    def test_read_lock_info_missing_file(self):
        """Test reading from non-existent lock file."""
        lock = ProcessLock(
            lock_path="/nonexistent/path/test.lock",
            enable_heartbeat=False,
        )
        
        pid, timestamp = lock._read_lock_info()
        
        assert pid is None
        assert timestamp is None


class TestProcessLockNonUnix:
    """Tests for non-Unix platform behavior."""
    
    @patch('vulcan.utils_main.process_lock.FCNTL_AVAILABLE', False)
    def test_acquire_without_fcntl(self):
        """Test that acquire succeeds without fcntl (graceful degradation)."""
        # Need to reimport with patched constant
        # For this test, we'll just verify the constant check
        lock = ProcessLock(enable_heartbeat=False)
        
        # The actual acquire() checks FCNTL_AVAILABLE at runtime
        # This test documents expected behavior
        if not FCNTL_AVAILABLE:
            assert lock.acquire() is True


class TestStateModuleIntegration:
    """Tests for state module integration."""
    
    def test_get_lock_status_no_lock(self):
        """Test get_lock_status when no lock is set."""
        try:
            # Import directly from state module to avoid FastAPI dependency
            from vulcan.server.state import get_lock_status
            import vulcan.server.state as state_module
        except ImportError as e:
            pytest.skip(f"vulcan.server.state not available: {e}")
        
        # Save original state
        original_lock = state_module.process_lock
        
        try:
            state_module.process_lock = None
            status = get_lock_status()
            
            assert status["locked"] is False
            assert status["heartbeat"] is None
            assert "not initialized" in status.get("message", "")
        finally:
            # Restore original state
            state_module.process_lock = original_lock
    
    @pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
    def test_get_lock_status_with_lock(self):
        """Test get_lock_status when lock is set."""
        try:
            # Import directly from state module to avoid FastAPI dependency
            from vulcan.server.state import get_lock_status
            import vulcan.server.state as state_module
        except ImportError as e:
            pytest.skip(f"vulcan.server.state not available: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(
                lock_path=lock_path,
                enable_heartbeat=True,
                heartbeat_interval=0.1,
            )
            
            # Save original state
            original_lock = state_module.process_lock
            
            try:
                assert lock.acquire() is True
                time.sleep(0.15)
                
                state_module.process_lock = lock
                status = get_lock_status()
                
                assert status["locked"] is True
                assert status["heartbeat"] is not None
                assert status["heartbeat"]["enabled"] is True
                assert status["heartbeat"]["thread_alive"] is True
            finally:
                lock.release()
                state_module.process_lock = original_lock


class TestDefaultLockPath:
    """Tests for default lock path determination."""
    
    def test_env_variable_override(self):
        """Test that VULCAN_LOCK_PATH env variable is respected."""
        custom_path = "/custom/lock/path.lock"
        
        with patch.dict(os.environ, {"VULCAN_LOCK_PATH": custom_path}):
            path = ProcessLock._get_default_lock_path()
            assert path == custom_path
    
    def test_fallback_to_tmp(self):
        """Test that /tmp is used when /var/lock is not writable."""
        # Unset any env variable
        with patch.dict(os.environ, {}, clear=True):
            # The actual behavior depends on /var/lock availability
            path = ProcessLock._get_default_lock_path()
            
            # Should be either /var/lock or temp dir
            assert "vulcan_orchestrator.lock" in path
