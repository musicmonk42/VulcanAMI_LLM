# ============================================================
# VULCAN-AGI Process Lock Module
# File-based process lock to prevent split-brain race conditions
# ============================================================
#
# This lock serves as a fallback when Redis is unavailable for state
# synchronization. If a second process attempts to start while the
# lock is held, it will detect the existing lock file and shut down.
#
# PLATFORM SUPPORT:
#     - Unix/Linux: Full support via fcntl.flock()
#     - Windows: Graceful degradation (lock always succeeds with warning)
#     - macOS: Full support via fcntl.flock()
#
# CONFIGURATION:
#     Environment Variable: VULCAN_LOCK_PATH
#     Default Paths:
#         - /var/lock/vulcan_orchestrator.lock (if writable)
#         - /tmp/vulcan_orchestrator.lock (fallback)
#
# USAGE:
#     # Context manager (recommended)
#     with ProcessLock() as lock:
#         # Critical section - only one process can execute
#         pass
#     
#     # Manual control
#     lock = ProcessLock()
#     if lock.acquire():
#         try:
#             # Critical section
#             pass
#         finally:
#             lock.release()
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation and error handling
# ============================================================

import logging
import os
from typing import Optional

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# PLATFORM-SPECIFIC IMPORTS
# ============================================================

# File-based locking for split-brain prevention when Redis is unavailable
try:
    import fcntl
    FCNTL_AVAILABLE = True
    logger.debug("fcntl module available - file locking enabled")
except ImportError:
    fcntl = None
    FCNTL_AVAILABLE = False
    logger.warning(
        "fcntl module not available (non-Unix system). "
        "File-based locking disabled - split-brain prevention unavailable."
    )


# ============================================================
# PROCESS LOCK CLASS
# ============================================================


class ProcessLock:
    """
    File-based process lock to prevent split-brain race conditions.
    
    This lock serves as a fallback when Redis is unavailable for state synchronization.
    If a second process attempts to start while the lock is held, it will detect
    the existing lock file and shut down immediately.
    
    Uses fcntl.flock() on Unix systems for advisory file locking.
    
    The lock file path can be configured via:
    - VULCAN_LOCK_PATH environment variable
    - Constructor parameter
    - Default: /var/lock/vulcan_orchestrator.lock (falls back to /tmp if /var/lock doesn't exist)
    """
    
    # Default paths - /var/lock is preferred for container environments
    DEFAULT_LOCK_DIR = "/var/lock"
    FALLBACK_LOCK_DIR = "/tmp"
    LOCK_FILENAME = "vulcan_orchestrator.lock"
    
    def __init__(self, lock_path: str = None):
        self.lock_path = lock_path or self._get_default_lock_path()
        self._lock_file = None
        self._locked = False
        self._logger = logging.getLogger("ProcessLock")
    
    @classmethod
    def _get_default_lock_path(cls) -> str:
        """Get the default lock file path, respecting environment variable."""
        # Priority 1: Environment variable
        env_path = os.getenv("VULCAN_LOCK_PATH")
        if env_path:
            return env_path
        
        # Priority 2: /var/lock if it exists and is writable
        if os.path.isdir(cls.DEFAULT_LOCK_DIR) and os.access(cls.DEFAULT_LOCK_DIR, os.W_OK):
            return os.path.join(cls.DEFAULT_LOCK_DIR, cls.LOCK_FILENAME)
        
        # Priority 3: /tmp as fallback
        return os.path.join(cls.FALLBACK_LOCK_DIR, cls.LOCK_FILENAME)
    
    def acquire(self) -> bool:
        """
        Attempt to acquire the process lock.
        
        Returns:
            True if lock was acquired successfully, False otherwise.
        """
        if not FCNTL_AVAILABLE:
            self._logger.warning(
                "fcntl not available (non-Unix system). "
                "File-based locking disabled - split-brain prevention unavailable."
            )
            return True  # Allow process to continue without lock on non-Unix systems
        
        try:
            # Ensure parent directory exists
            lock_dir = os.path.dirname(self.lock_path)
            if lock_dir and not os.path.exists(lock_dir):
                os.makedirs(lock_dir, exist_ok=True)
            
            # Open file in read/write mode, creating if needed
            # Using 'a+' mode preserves existing content and creates if not exists
            # Then seek to beginning for reading existing PID if needed
            self._lock_file = open(self.lock_path, "a+")
            self._lock_file.seek(0)
            
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Truncate and write our PID (we now hold the lock)
            self._lock_file.truncate(0)
            self._lock_file.write(f"{os.getpid()}\n")
            self._lock_file.flush()
            
            self._locked = True
            self._logger.info(
                f"Process lock acquired (PID: {os.getpid()}, file: {self.lock_path})"
            )
            return True
            
        except (IOError, OSError) as e:
            # Lock is held by another process - try to read existing PID for debugging
            existing_pid = None
            if self._lock_file:
                try:
                    self._lock_file.seek(0)
                    existing_pid = self._lock_file.read().strip()
                except Exception:
                    pass
                self._lock_file.close()
                self._lock_file = None
            
            pid_info = f" (held by PID: {existing_pid})" if existing_pid else ""
            self._logger.error(
                f"Failed to acquire process lock: {e}.{pid_info} "
                f"Another vulcan.orchestrator instance may be running. "
                f"Lock file: {self.lock_path}"
            )
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error acquiring process lock: {e}")
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            return False
    
    def release(self):
        """Release the process lock."""
        if not self._locked or not self._lock_file:
            return
        
        try:
            if FCNTL_AVAILABLE:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            self._locked = False
            
            # Remove lock file
            try:
                os.remove(self.lock_path)
            except OSError:
                pass  # File may have already been removed
                
            self._logger.info("Process lock released")
        except Exception as e:
            self._logger.warning(f"Error releasing process lock: {e}")
    
    def is_locked(self) -> bool:
        """Check if lock is currently held by this process."""
        return self._locked
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(
                "Failed to acquire process lock - another instance may be running. "
                "This prevents split-brain race conditions when Redis is unavailable."
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
    
    def __repr__(self) -> str:
        """String representation of the lock."""
        status = "locked" if self._locked else "unlocked"
        return f"ProcessLock(path={self.lock_path!r}, status={status})"


# ============================================================
# GLOBAL PROCESS LOCK MANAGEMENT
# ============================================================

# Global process lock instance (initialized during startup)
_process_lock: Optional[ProcessLock] = None


def get_process_lock() -> Optional[ProcessLock]:
    """
    Get the global process lock instance.
    
    Returns:
        The global ProcessLock instance, or None if not initialized
    """
    return _process_lock


def set_process_lock(lock: ProcessLock) -> None:
    """
    Set the global process lock instance.
    
    Args:
        lock: The ProcessLock instance to set as global
    """
    global _process_lock
    _process_lock = lock
    logger.debug(f"Global process lock set: {lock}")


def create_and_acquire_lock(lock_path: str = None) -> Optional[ProcessLock]:
    """
    Create a new process lock and attempt to acquire it.
    
    Args:
        lock_path: Optional path for the lock file
        
    Returns:
        ProcessLock instance if acquired successfully, None otherwise
    """
    lock = ProcessLock(lock_path)
    if lock.acquire():
        set_process_lock(lock)
        return lock
    return None


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "ProcessLock",
    "FCNTL_AVAILABLE",
    "get_process_lock",
    "set_process_lock",
    "create_and_acquire_lock",
]


# Log module initialization
logger.debug(f"Process lock module v{__version__} loaded (fcntl available: {FCNTL_AVAILABLE})")
