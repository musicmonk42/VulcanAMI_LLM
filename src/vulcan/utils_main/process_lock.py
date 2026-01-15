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
# HEARTBEAT (v1.1.0):
#     The lock now supports a heartbeat mechanism for distributed systems.
#     When enabled, the lock file is periodically updated with a timestamp
#     to indicate the process is still alive. Other processes can detect
#     stale locks by checking if the timestamp is older than the TTL.
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation and error handling
#     1.1.0 - Added heartbeat mechanism for self-healing distributed systems
# ============================================================

import logging
import os
import tempfile
import time
import threading
from typing import Optional

# Module metadata
__version__ = "1.1.0"
__author__ = "VULCAN-AGI Team"

# ============================================================
# HEARTBEAT CONFIGURATION
# ============================================================

# Default heartbeat interval in seconds - how often to refresh the lock timestamp
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
"""
Heartbeat interval: How frequently the lock timestamp is updated.

Rationale: 30 seconds provides a good balance between responsiveness
(detecting dead processes quickly) and avoiding excessive I/O overhead.
"""

# Default TTL in seconds - how long before a lock is considered stale
DEFAULT_LOCK_TTL_SECONDS = 90
"""
Lock TTL: How long after the last heartbeat before a lock is considered stale.

Rationale: 3x the heartbeat interval (90 seconds) allows for some timing
variance and brief I/O delays while still detecting crashed processes
within a reasonable timeframe.
"""

logger = logging.getLogger(__name__)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def is_process_running(pid: int) -> bool:
    """
    Check if a process with given PID is running.
    
    Uses os.kill with signal 0 (null signal) to check process existence.
    This is a standard Unix technique for checking if a process is alive.
    
    Args:
        pid: Process ID to check
        
    Returns:
        True if process exists and is running, False otherwise
    """
    if pid <= 0:
        return False
    try:
        # Signal 0 doesn't send a signal but checks if process exists
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # Process does not exist
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    except OSError:
        # Other OS errors - assume process doesn't exist
        return False


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
    
    Heartbeat Support (v1.1.0):
        The lock now supports a heartbeat mechanism that periodically updates
        the lock file with a timestamp. This allows other processes to detect
        stale locks from crashed processes and safely acquire them. The heartbeat
        runs in a daemon thread that automatically stops on shutdown.
    
    The lock file path can be configured via:
    - VULCAN_LOCK_PATH environment variable
    - Constructor parameter
    - Default: /var/lock/vulcan_orchestrator.lock (falls back to /tmp if /var/lock doesn't exist)
    
    Lock File Format:
        Line 1: PID
        Line 2: Timestamp (Unix time) - added by heartbeat
    """
    
    # Default paths - /var/lock is preferred for container environments
    DEFAULT_LOCK_DIR = "/var/lock"
    # Use tempfile.gettempdir() instead of hardcoded /tmp for security (B108)
    FALLBACK_LOCK_DIR = tempfile.gettempdir()
    LOCK_FILENAME = "vulcan_orchestrator.lock"
    
    def __init__(
        self,
        lock_path: str = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        lock_ttl: float = DEFAULT_LOCK_TTL_SECONDS,
        enable_heartbeat: bool = True,
    ):
        """
        Initialize the process lock.
        
        Args:
            lock_path: Optional custom path for the lock file
            heartbeat_interval: How often to update the lock timestamp (seconds)
            lock_ttl: How long before a lock is considered stale (seconds)
            enable_heartbeat: Whether to start the heartbeat thread on acquire
        """
        self.lock_path = lock_path or self._get_default_lock_path()
        self._lock_file = None
        self._locked = False
        self._logger = logging.getLogger("ProcessLock")
        
        # Heartbeat configuration
        self._heartbeat_interval = heartbeat_interval
        self._lock_ttl = lock_ttl
        self._enable_heartbeat = enable_heartbeat
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._last_heartbeat: float = 0.0
    
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
    
    def _read_lock_info(self) -> tuple[Optional[int], Optional[float]]:
        """
        Read PID and timestamp from lock file.
        
        Returns:
            Tuple of (pid, timestamp) where either may be None if not available.
            
        Thread Safety:
            This method should only be called when holding the file lock
            or when the lock file is known to be stable.
        """
        try:
            with open(self.lock_path, "r") as f:
                lines = f.read().strip().split("\n")
                pid = int(lines[0]) if lines and lines[0] else None
                timestamp = float(lines[1]) if len(lines) > 1 and lines[1] else None
                return pid, timestamp
        except (FileNotFoundError, ValueError, IndexError, OSError):
            return None, None
    
    def _write_lock_info(self, pid: int, timestamp: float) -> None:
        """
        Write PID and timestamp to lock file.
        
        Args:
            pid: Process ID to write
            timestamp: Unix timestamp to write
            
        Thread Safety:
            This method should only be called when holding the file lock.
        """
        if self._lock_file:
            self._lock_file.seek(0)
            self._lock_file.truncate(0)
            self._lock_file.write(f"{pid}\n{timestamp}\n")
            self._lock_file.flush()
            # Ensure data is written to disk for durability
            os.fsync(self._lock_file.fileno())
    
    def _is_lock_stale(self, pid: Optional[int], timestamp: Optional[float]) -> bool:
        """
        Check if a lock is stale based on PID and timestamp.
        
        A lock is considered stale if:
        1. The process is no longer running, OR
        2. The timestamp is older than the TTL (heartbeat expired)
        
        Args:
            pid: Process ID from lock file
            timestamp: Timestamp from lock file
            
        Returns:
            True if lock is stale and can be safely acquired, False otherwise.
        """
        # If no PID, lock file is corrupted - treat as stale
        if pid is None:
            self._logger.warning("Lock file has no PID - treating as stale")
            return True
        
        # Check if process is still running
        if not is_process_running(pid):
            self._logger.warning(
                f"Lock held by PID {pid} which is no longer running - treating as stale"
            )
            return True
        
        # If heartbeat is enabled, check timestamp
        if self._enable_heartbeat and timestamp is not None:
            age = time.time() - timestamp
            if age > self._lock_ttl:
                self._logger.warning(
                    f"Lock heartbeat expired: last update {age:.1f}s ago "
                    f"(TTL: {self._lock_ttl}s) - treating as stale. "
                    f"PID {pid} may be a zombie or frozen process."
                )
                return True
        
        return False
    
    def _heartbeat_loop(self) -> None:
        """
        Background thread that periodically updates the lock timestamp.
        
        This provides a heartbeat mechanism for distributed systems:
        - Other processes can detect if this process has crashed by checking
          if the timestamp is older than the TTL
        - Enables self-healing: crashed processes' locks can be safely acquired
        
        Thread Safety:
            This method runs in its own daemon thread and uses the stop event
            for clean shutdown. File operations are atomic via fsync.
        """
        self._logger.debug(
            f"Heartbeat thread started (interval: {self._heartbeat_interval}s, "
            f"TTL: {self._lock_ttl}s)"
        )
        
        consecutive_failures = 0
        max_failures = 3
        
        while not self._heartbeat_stop_event.is_set():
            try:
                if self._locked and self._lock_file:
                    current_time = time.time()
                    self._write_lock_info(os.getpid(), current_time)
                    self._last_heartbeat = current_time
                    consecutive_failures = 0
                    self._logger.debug(
                        f"Heartbeat updated: PID {os.getpid()}, timestamp {current_time:.3f}"
                    )
            except Exception as e:
                consecutive_failures += 1
                self._logger.warning(
                    f"Heartbeat update failed ({consecutive_failures}/{max_failures}): {e}"
                )
                if consecutive_failures >= max_failures:
                    self._logger.error(
                        f"Heartbeat failed {max_failures} consecutive times. "
                        "Lock may become stale. Consider investigating disk I/O issues."
                    )
                    # Don't break - keep trying, the issue may be transient
            
            # Wait for next heartbeat interval or until stopped
            self._heartbeat_stop_event.wait(self._heartbeat_interval)
        
        self._logger.debug("Heartbeat thread stopped")
    
    def _start_heartbeat(self) -> None:
        """
        Start the heartbeat background thread.
        
        Thread Safety:
            Safe to call multiple times - will only start one thread.
        """
        if not self._enable_heartbeat:
            return
        
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._logger.debug("Heartbeat thread already running")
            return
        
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="ProcessLock-Heartbeat",
            daemon=True  # Daemon thread - won't prevent process exit
        )
        self._heartbeat_thread.start()
        self._logger.info(
            f"Heartbeat thread started (interval={self._heartbeat_interval}s, "
            f"ttl={self._lock_ttl}s)"
        )
    
    def _stop_heartbeat(self) -> None:
        """
        Stop the heartbeat background thread gracefully.
        
        Thread Safety:
            Safe to call multiple times. Uses event signaling for clean shutdown.
        """
        if self._heartbeat_thread is None:
            return
        
        self._heartbeat_stop_event.set()
        
        # Wait for thread to finish with timeout
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self._heartbeat_interval + 1)
            if self._heartbeat_thread.is_alive():
                self._logger.warning(
                    "Heartbeat thread did not stop within timeout. "
                    "Thread will be terminated on process exit (daemon thread)."
                )
        
        self._heartbeat_thread = None
        self._logger.debug("Heartbeat thread stopped")
    
    def acquire(self) -> bool:
        """
        Attempt to acquire the process lock.
        
        This method implements a robust lock acquisition strategy:
        1. Try to acquire the file lock (non-blocking)
        2. If lock is held, check if it's stale (dead process or expired heartbeat)
        3. If stale, remove and retry
        4. If acquired, start heartbeat thread (if enabled)
        
        Returns:
            True if lock was acquired successfully, False otherwise.
            
        Thread Safety:
            This method is not thread-safe. Only one thread per process
            should attempt to acquire the lock.
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
            self._lock_file = open(self.lock_path, "a+")
            self._lock_file.seek(0)
            
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # We now hold the lock - write our PID and initial timestamp
            current_time = time.time()
            self._write_lock_info(os.getpid(), current_time)
            self._last_heartbeat = current_time
            
            self._locked = True
            self._logger.info(
                f"Process lock acquired (PID: {os.getpid()}, file: {self.lock_path})"
            )
            
            # Start heartbeat thread if enabled
            self._start_heartbeat()
            
            return True
            
        except (IOError, OSError) as e:
            # Lock is held by another process - check if it's stale
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            
            # Read existing lock info
            existing_pid, existing_timestamp = self._read_lock_info()
            
            # Check if lock is stale
            if self._is_lock_stale(existing_pid, existing_timestamp):
                self._logger.warning(
                    f"Detected stale lock (PID: {existing_pid}, "
                    f"timestamp: {existing_timestamp}). "
                    "Removing stale lock and retrying acquisition."
                )
                
                # Remove stale lock file
                try:
                    os.remove(self.lock_path)
                except OSError:
                    pass  # File may have already been removed
                
                # Retry lock acquisition once
                try:
                    self._lock_file = open(self.lock_path, "a+")
                    self._lock_file.seek(0)
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    current_time = time.time()
                    self._write_lock_info(os.getpid(), current_time)
                    self._last_heartbeat = current_time
                    
                    self._locked = True
                    self._logger.info(
                        f"Process lock acquired after removing stale lock "
                        f"(PID: {os.getpid()}, file: {self.lock_path})"
                    )
                    
                    # Start heartbeat thread if enabled
                    self._start_heartbeat()
                    
                    return True
                except (IOError, OSError) as retry_error:
                    if self._lock_file:
                        self._lock_file.close()
                        self._lock_file = None
                    self._logger.error(
                        f"Failed to acquire lock after removing stale lock: {retry_error}"
                    )
                    return False
            
            # Lock is held by an active process
            pid_info = f" (held by PID: {existing_pid})" if existing_pid else ""
            timestamp_info = ""
            if existing_timestamp:
                age = time.time() - existing_timestamp
                timestamp_info = f", last heartbeat: {age:.1f}s ago"
            
            self._logger.error(
                f"Failed to acquire process lock: {e}.{pid_info}{timestamp_info} "
                f"Another vulcan.orchestrator instance is running. "
                f"Lock file: {self.lock_path}"
            )
            return False
            
        except Exception as e:
            self._logger.error(f"Unexpected error acquiring process lock: {e}")
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            return False
    
    def release(self) -> None:
        """
        Release the process lock.
        
        This method:
        1. Stops the heartbeat thread (if running)
        2. Releases the file lock
        3. Removes the lock file
        
        Thread Safety:
            Safe to call multiple times. Subsequent calls are no-ops.
        """
        if not self._locked:
            return
        
        # Stop heartbeat thread first
        self._stop_heartbeat()
        
        try:
            if self._lock_file:
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
            self._locked = False
    
    def is_locked(self) -> bool:
        """
        Check if lock is currently held by this process.
        
        Returns:
            True if this process holds the lock, False otherwise.
        """
        return self._locked
    
    def get_heartbeat_status(self) -> dict:
        """
        Get current heartbeat status for monitoring/diagnostics.
        
        Returns:
            Dictionary with heartbeat status information:
            - enabled: Whether heartbeat is enabled
            - thread_alive: Whether heartbeat thread is running
            - last_heartbeat: Timestamp of last successful heartbeat
            - interval: Configured heartbeat interval
            - ttl: Configured lock TTL
        """
        return {
            "enabled": self._enable_heartbeat,
            "thread_alive": (
                self._heartbeat_thread is not None and 
                self._heartbeat_thread.is_alive()
            ),
            "last_heartbeat": self._last_heartbeat,
            "seconds_since_heartbeat": (
                time.time() - self._last_heartbeat if self._last_heartbeat > 0 else None
            ),
            "interval": self._heartbeat_interval,
            "ttl": self._lock_ttl,
        }
    
    def __enter__(self):
        """Context manager entry - acquire lock."""
        if not self.acquire():
            raise RuntimeError(
                "Failed to acquire process lock - another instance may be running. "
                "This prevents split-brain race conditions when Redis is unavailable."
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock."""
        self.release()
        return False
    
    def __repr__(self) -> str:
        """String representation of the lock."""
        status = "locked" if self._locked else "unlocked"
        heartbeat = "heartbeat=on" if self._enable_heartbeat else "heartbeat=off"
        return f"ProcessLock(path={self.lock_path!r}, status={status}, {heartbeat})"


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
    "is_process_running",
    "get_process_lock",
    "set_process_lock",
    "create_and_acquire_lock",
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_LOCK_TTL_SECONDS",
]


# Log module initialization
logger.debug(f"Process lock module v{__version__} loaded (fcntl available: {FCNTL_AVAILABLE})")
