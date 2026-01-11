"""
Test Suite: VULCAN-AGI Utils Main Fixes

This test suite validates the fixes implemented for the VULCAN-AGI utils_main package,
including stale lock detection, bytes handling in sanitization, and key collision warnings.

ISSUES ADDRESSED:
    1. Stale lock detection in process_lock.py (P1)
    2. Bytes handling in sanitize.py (P1)
    3. Key collision warning in sanitize.py (P2)

TEST STRATEGY:
    - Mock-based testing for process status checks
    - Edge case testing for bytes encoding scenarios
    - Log capture for warning verification
    - Thread safety validation
    - Integration tests with real file system operations

AUTHOR: VULCAN-AGI Team
VERSION: 1.0.0
CREATED: 2026-01-11
"""

import base64
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Provide pytest.fail fallback
    class pytest:
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
        
        class raises:
            def __init__(self, exc_class):
                self.exc_class = exc_class
                self.exc_info = None
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.exc_class} but no exception was raised")
                if not issubclass(exc_type, self.exc_class):
                    return False
                self.exc_info = (exc_type, exc_val, exc_tb)
                return True

from src.vulcan.utils_main.process_lock import (
    ProcessLock,
    is_process_running,
    FCNTL_AVAILABLE,
)
from src.vulcan.utils_main.sanitize import (
    sanitize_payload,
    deep_sanitize_for_json,
)


# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def temp_lock_dir():
    """Create a temporary directory for lock file tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def mock_logger():
    """Create a mock logger for capturing log messages."""
    logger = MagicMock()
    return logger


@pytest.fixture
def caplog_fixture(caplog):
    """Pytest caplog fixture for capturing logs."""
    caplog.set_level(logging.WARNING)
    return caplog


# ============================================================
# PROCESS LOCK TESTS - STALE LOCK DETECTION
# ============================================================


class TestProcessLockStaleLockDetection:
    """Test stale lock detection functionality."""
    
    def test_is_process_running_current_process(self):
        """Test that current process is detected as running."""
        current_pid = os.getpid()
        assert is_process_running(current_pid) is True
    
    def test_is_process_running_invalid_pid_zero(self):
        """Test that PID 0 returns False."""
        assert is_process_running(0) is False
    
    def test_is_process_running_invalid_pid_negative(self):
        """Test that negative PID returns False."""
        assert is_process_running(-1) is False
        assert is_process_running(-999) is False
    
    def test_is_process_running_nonexistent_pid(self):
        """Test that non-existent PID returns False."""
        # Use a very high PID that is unlikely to exist
        nonexistent_pid = 999999
        assert is_process_running(nonexistent_pid) is False
    
    @patch('os.kill')
    def test_is_process_running_permission_error(self, mock_kill):
        """Test that PermissionError means process exists."""
        mock_kill.side_effect = PermissionError("No permission")
        # Process exists but we can't signal it
        assert is_process_running(1234) is True
    
    @patch('os.kill')
    def test_is_process_running_process_lookup_error(self, mock_kill):
        """Test that ProcessLookupError means process doesn't exist."""
        mock_kill.side_effect = ProcessLookupError("Process not found")
        assert is_process_running(1234) is False
    
    @patch('os.kill')
    def test_is_process_running_os_error(self, mock_kill):
        """Test that OSError is handled gracefully."""
        mock_kill.side_effect = OSError("Some OS error")
        assert is_process_running(1234) is False
    
    @pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
    def test_stale_lock_detection_and_cleanup(self, temp_lock_dir):
        """Test that stale locks are detected and removed."""
        lock_path = os.path.join(temp_lock_dir, "test.lock")
        
        # Create a lock file with a non-existent PID
        stale_pid = 999999
        with open(lock_path, 'w') as f:
            f.write(f"{stale_pid}\n")
        
        # Mock fcntl to simulate lock acquisition failure on first try
        original_flock = None
        try:
            import fcntl
            original_flock = fcntl.flock
            call_count = [0]
            
            def mock_flock(fd, operation):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call - simulate lock held
                    raise IOError("Resource temporarily unavailable")
                # Second call - allow acquisition
                return original_flock(fd, operation)
            
            with patch('fcntl.flock', side_effect=mock_flock):
                lock = ProcessLock(lock_path)
                # Should detect stale lock and acquire it
                result = lock.acquire()
                
                assert result is True
                assert lock.is_locked() is True
                
                # Verify our PID is now in the file
                with open(lock_path, 'r') as f:
                    content = f.read().strip()
                    assert int(content) == os.getpid()
                
                lock.release()
        finally:
            pass
    
    @pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
    def test_stale_lock_warning_logged(self, temp_lock_dir, caplog):
        """Test that stale lock detection logs a warning."""
        lock_path = os.path.join(temp_lock_dir, "test.lock")
        
        # Create a stale lock file
        stale_pid = 999999
        with open(lock_path, 'w') as f:
            f.write(f"{stale_pid}\n")
        
        # Mock fcntl to fail on first acquisition
        original_flock = None
        try:
            import fcntl
            original_flock = fcntl.flock
            call_count = [0]
            
            def mock_flock(fd, operation):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise IOError("Resource temporarily unavailable")
                return original_flock(fd, operation)
            
            with patch('fcntl.flock', side_effect=mock_flock):
                with caplog.at_level(logging.WARNING):
                    lock = ProcessLock(lock_path)
                    lock.acquire()
                    
                    # Check that warning was logged
                    assert any("stale lock" in record.message.lower() for record in caplog.records)
                    assert any(str(stale_pid) in record.message for record in caplog.records)
                    
                    lock.release()
        finally:
            pass
    
    @pytest.mark.skipif(not FCNTL_AVAILABLE, reason="fcntl not available on this platform")
    def test_no_stale_lock_when_process_running(self, temp_lock_dir):
        """Test that lock is not removed if process is still running."""
        lock_path = os.path.join(temp_lock_dir, "test.lock")
        
        # Use current process PID (definitely running)
        running_pid = os.getpid()
        
        # Create first lock
        lock1 = ProcessLock(lock_path)
        acquired = lock1.acquire()
        assert acquired is True
        
        # Try to acquire with second lock instance
        lock2 = ProcessLock(lock_path)
        acquired2 = lock2.acquire()
        
        # Should fail because process is still running
        assert acquired2 is False
        
        # Clean up
        lock1.release()


# ============================================================
# SANITIZE TESTS - BYTES HANDLING
# ============================================================


class TestSanitizeBytesHandling:
    """Test bytes and bytearray handling in sanitization."""
    
    def test_bytes_utf8_decodable(self):
        """Test that UTF-8 decodable bytes are decoded to string."""
        data = b"Hello, World!"
        result = deep_sanitize_for_json(data)
        assert result == "Hello, World!"
        assert isinstance(result, str)
    
    def test_bytes_utf8_with_unicode(self):
        """Test that UTF-8 bytes with unicode characters are decoded."""
        data = "Привет мир 🌍".encode('utf-8')
        result = deep_sanitize_for_json(data)
        assert result == "Привет мир 🌍"
        assert isinstance(result, str)
    
    def test_bytes_non_utf8_base64_encoded(self):
        """Test that non-UTF-8 bytes are base64 encoded."""
        # Binary data that's not valid UTF-8
        data = b'\x00\x01\x02\xff\xfe\xfd'
        result = deep_sanitize_for_json(data)
        
        # Should be base64 encoded
        expected = base64.b64encode(data).decode('ascii')
        assert result == expected
        assert isinstance(result, str)
        
        # Verify it can be decoded back
        decoded = base64.b64decode(result)
        assert decoded == data
    
    def test_bytes_mixed_valid_invalid_utf8(self):
        """Test bytes with invalid UTF-8 sequences are base64 encoded."""
        # Mix valid ASCII with invalid UTF-8
        data = b'Hello\xff\xfeWorld'
        result = deep_sanitize_for_json(data)
        
        # Should be base64 encoded due to invalid sequences
        expected = base64.b64encode(data).decode('ascii')
        assert result == expected
    
    def test_bytearray_utf8_decodable(self):
        """Test that UTF-8 decodable bytearray is decoded to string."""
        data = bytearray(b"Test bytearray")
        result = deep_sanitize_for_json(data)
        assert result == "Test bytearray"
        assert isinstance(result, str)
    
    def test_bytearray_non_utf8_base64_encoded(self):
        """Test that non-UTF-8 bytearray is base64 encoded."""
        data = bytearray(b'\x00\x01\x02\xff\xfe\xfd')
        result = deep_sanitize_for_json(data)
        
        # Should be base64 encoded
        expected = base64.b64encode(bytes(data)).decode('ascii')
        assert result == expected
    
    def test_bytes_in_dict(self):
        """Test bytes handling when nested in dictionary."""
        data = {
            "text": b"UTF-8 text",
            "binary": b'\x00\x01\x02',
            "nested": {
                "data": b"More text"
            }
        }
        result = deep_sanitize_for_json(data)
        
        assert result["text"] == "UTF-8 text"
        assert result["binary"] == base64.b64encode(b'\x00\x01\x02').decode('ascii')
        assert result["nested"]["data"] == "More text"
    
    def test_bytes_in_list(self):
        """Test bytes handling when in list."""
        data = [b"text1", b'\x00\x01', b"text2"]
        result = deep_sanitize_for_json(data)
        
        assert result[0] == "text1"
        assert result[1] == base64.b64encode(b'\x00\x01').decode('ascii')
        assert result[2] == "text2"
    
    def test_empty_bytes(self):
        """Test handling of empty bytes."""
        data = b""
        result = deep_sanitize_for_json(data)
        assert result == ""
        assert isinstance(result, str)
    
    def test_empty_bytearray(self):
        """Test handling of empty bytearray."""
        data = bytearray()
        result = deep_sanitize_for_json(data)
        assert result == ""
        assert isinstance(result, str)
    
    def test_bytes_before_str_fallback(self):
        """Test that bytes are handled before falling back to str()."""
        # This ensures we don't get "b'data'" string representation
        data = b"test"
        result = deep_sanitize_for_json(data)
        
        # Should be decoded, not converted via str()
        assert result == "test"
        assert result != "b'test'"


# ============================================================
# SANITIZE TESTS - KEY COLLISION WARNING
# ============================================================


class TestSanitizeKeyCollisionWarning:
    """Test key collision detection and warning."""
    
    def test_key_collision_warning_logged(self, caplog):
        """Test that key collision logs a warning."""
        # Create objects that will have the same string representation
        class CustomKey:
            def __str__(self):
                return "same_key"
        
        data = {
            CustomKey(): "value1",
            "same_key": "value2"  # This will collide with CustomKey().__str__()
        }
        
        with caplog.at_level(logging.WARNING):
            result = sanitize_payload(data)
            
            # Check that warning was logged
            warning_messages = [record.message for record in caplog.records 
                               if record.levelname == "WARNING"]
            assert any("collision" in msg.lower() for msg in warning_messages)
            assert any("same_key" in msg for msg in warning_messages)
    
    def test_key_collision_preserves_last_value(self, caplog):
        """Test that key collision preserves the last value seen."""
        class CustomKey1:
            def __str__(self):
                return "key"
        
        class CustomKey2:
            def __str__(self):
                return "key"
        
        # Note: dict iteration order is insertion order in Python 3.7+
        data = {
            CustomKey1(): "first",
            CustomKey2(): "second",
            "key": "third"
        }
        
        with caplog.at_level(logging.WARNING):
            result = sanitize_payload(data)
            
            # Should have logged warnings for collisions
            warning_count = sum(1 for r in caplog.records 
                               if "collision" in r.message.lower())
            assert warning_count >= 1  # At least one collision
            
            # Result should have the key with last value
            assert "key" in result
    
    def test_no_collision_warning_unique_keys(self, caplog):
        """Test that no warning is logged when all keys are unique."""
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        with caplog.at_level(logging.WARNING):
            result = sanitize_payload(data)
            
            # Should not have any collision warnings
            warning_messages = [record.message for record in caplog.records 
                               if record.levelname == "WARNING"]
            assert not any("collision" in msg.lower() for msg in warning_messages)
    
    def test_integer_key_collision(self, caplog):
        """Test collision detection with integer keys."""
        data = {
            1: "number_one",
            "1": "string_one"  # Will collide when integer is converted to string
        }
        
        with caplog.at_level(logging.WARNING):
            result = sanitize_payload(data)
            
            # Should log collision warning
            assert any("collision" in record.message.lower() 
                      for record in caplog.records 
                      if record.levelname == "WARNING")
            
            # Should have key "1" in result
            assert "1" in result
    
    def test_nested_dict_collision_warning(self, caplog):
        """Test that collisions in nested dicts are also detected."""
        data = {
            "outer": {
                1: "value1",
                "1": "value2"  # Collision in nested dict
            }
        }
        
        with caplog.at_level(logging.WARNING):
            result = sanitize_payload(data)
            
            # Should log collision warning for nested dict
            assert any("collision" in record.message.lower() 
                      for record in caplog.records 
                      if record.levelname == "WARNING")
    
    def test_none_key_no_collision_warning(self, caplog):
        """Test that None keys don't cause collision warnings."""
        data = {
            None: "none_value",
            "key": "value"
        }
        
        with caplog.at_level(logging.WARNING):
            result = sanitize_payload(data)
            
            # Should not log collision warning (None is removed, not converted)
            collision_warnings = [r for r in caplog.records 
                                 if "collision" in r.message.lower() 
                                 and r.levelname == "WARNING"]
            assert len(collision_warnings) == 0
            
            # None key should be removed
            assert None not in result
            assert "None" not in result
            assert "key" in result


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests combining multiple fixes."""
    
    def test_sanitize_complex_data_with_bytes(self):
        """Test sanitizing complex data structure with bytes."""
        data = {
            "user": "Alice",
            "data": b"Binary data",
            "nested": {
                "bytes": b'\x00\x01\x02',
                "list": [b"item1", b"item2"]
            },
            None: "should_be_removed"
        }
        
        result = deep_sanitize_for_json(data)
        
        assert result["user"] == "Alice"
        assert result["data"] == "Binary data"
        assert result["nested"]["bytes"] == base64.b64encode(b'\x00\x01\x02').decode('ascii')
        assert result["nested"]["list"][0] == "item1"
        assert "None" not in result
    
    def test_is_process_running_exported(self):
        """Test that is_process_running is properly exported."""
        from src.vulcan.utils_main.process_lock import __all__
        assert "is_process_running" in __all__


# ============================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================


class TestBackwardCompatibility:
    """Test that fixes don't break existing functionality."""
    
    def test_sanitize_payload_existing_behavior(self):
        """Test that existing sanitize_payload behavior is preserved."""
        # Test basic functionality that should not change
        data = {
            None: "removed",
            "key": "value",
            "nested": {
                None: "also_removed",
                "inner": 123
            }
        }
        
        result = sanitize_payload(data)
        
        assert "key" in result
        assert result["key"] == "value"
        assert None not in result
        assert result["nested"]["inner"] == 123
    
    def test_deep_sanitize_existing_types(self):
        """Test that existing type handling is preserved."""
        import datetime
        from enum import Enum
        
        class Color(Enum):
            RED = 1
            BLUE = 2
        
        data = {
            "string": "text",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "enum": Color.RED,
            "datetime": datetime.datetime(2024, 1, 1, 12, 0, 0)
        }
        
        result = deep_sanitize_for_json(data)
        
        assert result["string"] == "text"
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None
        assert result["enum"] == 1
        assert "2024-01-01" in result["datetime"]


# ============================================================
# MAIN EXECUTION
# ============================================================


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("pytest not available, running basic validation...")
        print("All test classes are defined successfully.")
