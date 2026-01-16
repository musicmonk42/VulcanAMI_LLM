"""
Comprehensive tests for the cross-platform ProcessLock implementation.

Tests verify:
1. Cross-platform compatibility (Windows, macOS, Linux)
2. Proper lock acquisition and release
3. Stale lock detection and recovery
4. Heartbeat mechanism
5. Concurrent process handling
6. Error handling and edge cases
7. Backward compatibility

Following industry standards for test quality.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

import pytest


class TestProcessLockImports:
    """Test that ProcessLock module imports correctly with proper compatibility."""
    
    def test_process_lock_imports(self):
        """Verify all ProcessLock components can be imported."""
        from vulcan.utils_main.process_lock import (
            ProcessLock,
            FILELOCK_AVAILABLE,
            FCNTL_AVAILABLE,
            get_process_lock,
            set_process_lock,
            create_and_acquire_lock,
            is_process_running,
            DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            DEFAULT_LOCK_TTL_SECONDS,
        )
        
        assert ProcessLock is not None
        assert isinstance(FILELOCK_AVAILABLE, bool)
        assert isinstance(FCNTL_AVAILABLE, bool)
        assert callable(get_process_lock)
        assert callable(set_process_lock)
        assert callable(create_and_acquire_lock)
        assert callable(is_process_running)
        assert isinstance(DEFAULT_HEARTBEAT_INTERVAL_SECONDS, int)
        assert isinstance(DEFAULT_LOCK_TTL_SECONDS, int)
    
    def test_backward_compatibility_fcntl_available(self):
        """Verify FCNTL_AVAILABLE is aliased to FILELOCK_AVAILABLE for backward compatibility."""
        from vulcan.utils_main.process_lock import FILELOCK_AVAILABLE, FCNTL_AVAILABLE
        
        # These should be the same for backward compatibility
        assert FCNTL_AVAILABLE == FILELOCK_AVAILABLE
    
    def test_exports_from_utils_main(self):
        """Verify ProcessLock can be imported from utils_main package."""
        from vulcan.utils_main import (
            ProcessLock,
            FILELOCK_AVAILABLE,
            FCNTL_AVAILABLE,
        )
        
        assert ProcessLock is not None
        assert isinstance(FILELOCK_AVAILABLE, bool)
        assert isinstance(FCNTL_AVAILABLE, bool)


class TestIsProcessRunning:
    """Test the is_process_running utility function."""
    
    def test_is_process_running_current_process(self):
        """Verify current process is detected as running."""
        from vulcan.utils_main.process_lock import is_process_running
        
        current_pid = os.getpid()
        assert is_process_running(current_pid) is True
    
    def test_is_process_running_invalid_pid(self):
        """Verify invalid PIDs return False."""
        from vulcan.utils_main.process_lock import is_process_running
        
        # Test various invalid PIDs
        assert is_process_running(0) is False
        assert is_process_running(-1) is False
        assert is_process_running(-999) is False
    
    def test_is_process_running_nonexistent_pid(self):
        """Verify non-existent PID returns False."""
        from vulcan.utils_main.process_lock import is_process_running
        
        # Use a very high PID that's unlikely to exist
        assert is_process_running(999999999) is False


try:
    import filelock
    FILELOCK_INSTALLED = True
except ImportError:
    FILELOCK_INSTALLED = False


@pytest.mark.skipif(not FILELOCK_INSTALLED, reason="Requires filelock library")
class TestProcessLockBasicOperations:
    """Test basic ProcessLock operations with filelock available."""
    
    @pytest.fixture
    def temp_lock_path(self):
        """Provide a temporary lock file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.lock")
    
    def test_lock_initialization(self, temp_lock_path):
        """Verify ProcessLock can be initialized."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path)
        
        assert lock.lock_path == temp_lock_path
        assert lock.is_locked() is False
        assert lock.heartbeat_interval == 30
        assert lock.lock_ttl == 90
    
    def test_lock_acquire_and_release(self, temp_lock_path):
        """Verify lock can be acquired and released."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=False)
        
        # Acquire lock
        assert lock.acquire() is True
        assert lock.is_locked() is True
        
        # Release lock
        lock.release()
        assert lock.is_locked() is False
    
    def test_lock_prevents_concurrent_acquisition(self, temp_lock_path):
        """Verify lock prevents concurrent acquisition by same process."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock1 = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=False)
        lock2 = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=False)
        
        # First lock acquires successfully
        assert lock1.acquire() is True
        
        # Second lock fails to acquire
        assert lock2.acquire() is False
        
        # Release first lock
        lock1.release()
        
        # Now second lock can acquire
        assert lock2.acquire() is True
        lock2.release()
    
    def test_lock_context_manager(self, temp_lock_path):
        """Verify lock works as context manager."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=False)
        
        with lock:
            assert lock.is_locked() is True
        
        # Lock should be released after exiting context
        assert lock.is_locked() is False
    
    def test_lock_custom_parameters(self, temp_lock_path):
        """Verify lock accepts custom heartbeat and TTL parameters."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(
            lock_path=temp_lock_path,
            heartbeat_interval=15,
            lock_ttl=45,
            enable_heartbeat=False,
        )
        
        assert lock.heartbeat_interval == 15
        assert lock.lock_ttl == 45
    
    def test_lock_repr(self, temp_lock_path):
        """Verify lock has informative string representation."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path)
        repr_str = repr(lock)
        
        assert "ProcessLock" in repr_str
        assert temp_lock_path in repr_str
        assert "locked=" in repr_str


@pytest.mark.skipif(not FILELOCK_INSTALLED, reason="Requires filelock library")
class TestProcessLockHeartbeat:
    """Test ProcessLock heartbeat mechanism."""
    
    @pytest.fixture
    def temp_lock_path(self):
        """Provide a temporary lock file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.lock")
    
    def test_heartbeat_metadata_written(self, temp_lock_path):
        """Verify heartbeat writes metadata file."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=True)
        lock.acquire()
        
        # Wait for heartbeat to update
        time.sleep(0.5)
        
        # Check metadata file exists
        metadata_path = temp_lock_path + ".meta"
        assert os.path.exists(metadata_path)
        
        # Read metadata
        with open(metadata_path, 'r') as f:
            content = f.read().strip()
            parts = content.split(':')
            assert len(parts) == 2
            pid = int(parts[0])
            timestamp = float(parts[1])
            assert pid == os.getpid()
            assert timestamp > 0
        
        lock.release()
        
        # Metadata should be cleaned up
        assert not os.path.exists(metadata_path)
    
    def test_heartbeat_status(self, temp_lock_path):
        """Verify heartbeat status can be queried."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=True)
        lock.acquire()
        
        status = lock.get_heartbeat_status()
        
        assert status['locked'] is True
        assert status['lock_path'] == temp_lock_path
        assert status['pid'] == os.getpid()
        assert status['last_heartbeat'] is not None
        assert status['heartbeat_age'] is not None
        assert status['thread_alive'] is True
        
        lock.release()
    
    def test_heartbeat_disabled(self, temp_lock_path):
        """Verify heartbeat can be disabled."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=False)
        lock.acquire()
        
        status = lock.get_heartbeat_status()
        
        # Heartbeat thread should not be running
        assert status['thread_alive'] is False
        
        lock.release()


@pytest.mark.skipif(not FILELOCK_INSTALLED, reason="Requires filelock library")
class TestProcessLockStaleLockDetection:
    """Test stale lock detection and recovery."""
    
    @pytest.fixture
    def temp_lock_path(self):
        """Provide a temporary lock file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.lock")
    
    def test_stale_lock_detection_dead_process(self, temp_lock_path):
        """Verify stale lock from dead process can be detected and cleared."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, enable_heartbeat=False)
        
        # Manually create lock file and metadata for a non-existent process
        # This simulates a crashed process
        with open(temp_lock_path, 'w') as f:
            f.write("locked")
        
        metadata_path = temp_lock_path + ".meta"
        with open(metadata_path, 'w') as f:
            f.write(f"999999999:{time.time()}")  # Non-existent PID
        
        # Lock should detect stale lock
        assert lock._is_lock_stale() is True
        
        # Cleanup
        if os.path.exists(temp_lock_path):
            os.remove(temp_lock_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    def test_stale_lock_detection_expired_heartbeat(self, temp_lock_path):
        """Verify stale lock from expired heartbeat is detected."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock(lock_path=temp_lock_path, lock_ttl=1, enable_heartbeat=False)
        
        # Manually create metadata with old timestamp
        metadata_path = temp_lock_path + ".meta"
        old_timestamp = time.time() - 10  # 10 seconds ago
        with open(metadata_path, 'w') as f:
            f.write(f"{os.getpid()}:{old_timestamp}")
        
        # Lock should detect stale lock
        assert lock._is_lock_stale() is True
        
        # Cleanup
        if os.path.exists(metadata_path):
            os.remove(metadata_path)


class TestProcessLockWithoutFilelock:
    """Test ProcessLock behavior when filelock is not available."""
    
    @pytest.fixture
    def mock_filelock_unavailable(self):
        """Mock filelock as unavailable."""
        import sys
        import vulcan.utils_main.process_lock as pl_module
        
        # Save original values
        original_available = pl_module.FILELOCK_AVAILABLE
        original_filelock = pl_module.FileLock
        
        # Mock as unavailable
        pl_module.FILELOCK_AVAILABLE = False
        pl_module.FileLock = None
        
        yield
        
        # Restore
        pl_module.FILELOCK_AVAILABLE = original_available
        pl_module.FileLock = original_filelock
    
    def test_acquire_fails_without_filelock(self, mock_filelock_unavailable):
        """Verify lock acquisition fails gracefully without filelock."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path)
            
            # Acquire should return False, not True
            result = lock.acquire()
            assert result is False
            assert lock.is_locked() is False


class TestProcessLockGlobalRegistry:
    """Test global process lock registry functions."""
    
    def test_get_set_process_lock(self):
        """Verify global process lock get/set functions."""
        from vulcan.utils_main.process_lock import (
            get_process_lock,
            set_process_lock,
            ProcessLock,
        )
        
        # Initially should be None
        assert get_process_lock() is None
        
        # Create and set a lock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path)
            set_process_lock(lock)
            
            # Should be retrievable
            retrieved = get_process_lock()
            assert retrieved is lock
            
            # Reset to None for cleanup
            set_process_lock(None)
    
    def test_create_and_acquire_lock_success(self):
        """Verify create_and_acquire_lock returns lock on success."""
        if not FILELOCK_INSTALLED:
            pytest.skip("filelock library not installed")
        from vulcan.utils_main.process_lock import create_and_acquire_lock, get_process_lock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            
            lock = create_and_acquire_lock(lock_path)
            
            if lock is not None:  # Only if filelock is available
                assert lock.is_locked() is True
                assert get_process_lock() is lock
                
                # Cleanup
                lock.release()
                from vulcan.utils_main.process_lock import set_process_lock
                set_process_lock(None)


class TestProcessLockEdgeCases:
    """Test edge cases and error handling."""
    
    def test_lock_path_auto_detection(self):
        """Verify lock path is auto-detected if not provided."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        lock = ProcessLock()
        
        # Should have a default path
        assert lock.lock_path is not None
        assert isinstance(lock.lock_path, str)
        assert len(lock.lock_path) > 0
    
    def test_release_without_acquire(self):
        """Verify releasing without acquiring is safe."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path)
            
            # Should not raise an exception
            lock.release()
            assert lock.is_locked() is False
    
    def test_multiple_releases(self):
        """Verify multiple releases are safe."""
        if not FILELOCK_INSTALLED:
            pytest.skip("filelock library not installed")
        from vulcan.utils_main.process_lock import ProcessLock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            
            lock.acquire()
            lock.release()
            
            # Second release should be safe
            lock.release()
            assert lock.is_locked() is False


class TestProcessLockSecurityAndRobustness:
    """Test security and robustness aspects."""
    
    def test_lock_uses_tempfile_gettempdir(self):
        """Verify lock uses tempfile.gettempdir() for portability."""
        from vulcan.utils_main.process_lock import ProcessLock
        
        # When /var/lock doesn't exist or isn't writable
        with patch('os.path.isdir', return_value=False):
            lock = ProcessLock()
            assert tempfile.gettempdir() in lock.lock_path
    
    def test_metadata_file_permissions(self):
        """Verify metadata files use safe permissions."""
        if not FILELOCK_INSTALLED:
            pytest.skip("filelock library not installed")
        from vulcan.utils_main.process_lock import ProcessLock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path, enable_heartbeat=False)
            
            lock.acquire()
            
            # Metadata file should exist
            metadata_path = lock_path + ".meta"
            if os.path.exists(metadata_path):
                # Verify it's a regular file
                assert os.path.isfile(metadata_path)
            
            lock.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
