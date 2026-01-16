"""
Process Lock Module - Cross-Platform Implementation

Provides file-based process locking that works on Windows, macOS, and Linux.
Uses the `filelock` library for cross-platform compatibility.
"""

import logging
import os
import tempfile
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0"
__author__ = "VULCAN-AGI Team"

# Configuration
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_LOCK_TTL_SECONDS = 90

# Check for filelock availability
try:
    from filelock import FileLock, Timeout as FileLockTimeout
    FILELOCK_AVAILABLE = True
except ImportError:
    FileLock = None
    FileLockTimeout = None
    FILELOCK_AVAILABLE = False
    logger.warning(
        "filelock library not available. Install with: pip install filelock. "
        "Process locking will be disabled, risking split-brain conditions."
    )

# Backwards compatibility alias
FCNTL_AVAILABLE = FILELOCK_AVAILABLE


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
        return True
    except OSError:
        return False
    except Exception:
        return False


class ProcessLock:
    """
    Cross-platform file-based process lock for split-brain prevention.
    
    Uses the `filelock` library which works on Windows, macOS, and Linux.
    Includes heartbeat mechanism for detecting stale locks from crashed processes.
    
    Usage:
        lock = ProcessLock()
        if lock.acquire():
            try:
                # Protected code here
                pass
            finally:
                lock.release()
        
        # Or as context manager:
        with ProcessLock() as lock:
            if lock.is_locked():
                # Protected code here
                pass
    
    Attributes:
        lock_path: Path to the lock file
        heartbeat_interval: Seconds between heartbeat updates
        lock_ttl: Seconds before a lock is considered stale
    """
    
    DEFAULT_LOCK_DIR = "/var/lock"
    FALLBACK_LOCK_DIR = tempfile.gettempdir()
    LOCK_FILENAME = "vulcan_orchestrator.lock"
    
    def __init__(
        self,
        lock_path: Optional[str] = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        lock_ttl: float = DEFAULT_LOCK_TTL_SECONDS,
        enable_heartbeat: bool = True,
    ):
        """
        Initialize the process lock.
        
        Args:
            lock_path: Path to lock file (default: auto-detect)
            heartbeat_interval: Seconds between heartbeat updates
            lock_ttl: Seconds before lock is considered stale
            enable_heartbeat: Whether to run heartbeat thread
        """
        self.lock_path = lock_path or self._get_default_lock_path()
        self.heartbeat_interval = heartbeat_interval
        self.lock_ttl = lock_ttl
        self.enable_heartbeat = enable_heartbeat
        
        self._lock: Optional[FileLock] = None
        self._locked = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._pid = os.getpid()
        
        # Metadata file for heartbeat (separate from lock file)
        self._metadata_path = self.lock_path + ".meta"
        
        if FILELOCK_AVAILABLE:
            self._lock = FileLock(self.lock_path, timeout=0)
    
    @classmethod
    def _get_default_lock_path(cls) -> str:
        """Get the default lock file path."""
        # Try /var/lock first (Linux standard), fall back to temp dir
        if os.path.isdir(cls.DEFAULT_LOCK_DIR) and os.access(cls.DEFAULT_LOCK_DIR, os.W_OK):
            return os.path.join(cls.DEFAULT_LOCK_DIR, cls.LOCK_FILENAME)
        return os.path.join(cls.FALLBACK_LOCK_DIR, cls.LOCK_FILENAME)
    
    def _read_metadata(self) -> tuple[Optional[int], Optional[float]]:
        """Read PID and timestamp from metadata file."""
        try:
            if os.path.exists(self._metadata_path):
                with open(self._metadata_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        parts = content.split(':')
                        if len(parts) == 2:
                            return int(parts[0]), float(parts[1])
        except (ValueError, IOError, OSError) as e:
            logger.debug(f"Could not read lock metadata: {e}")
        return None, None
    
    def _write_metadata(self) -> None:
        """Write PID and timestamp to metadata file."""
        try:
            with open(self._metadata_path, 'w') as f:
                f.write(f"{self._pid}:{time.time()}")
        except (IOError, OSError) as e:
            logger.warning(f"Could not write lock metadata: {e}")
    
    def _clear_metadata(self) -> None:
        """Remove the metadata file."""
        try:
            if os.path.exists(self._metadata_path):
                os.remove(self._metadata_path)
        except (IOError, OSError) as e:
            logger.debug(f"Could not remove lock metadata: {e}")
    
    def _is_lock_stale(self) -> bool:
        """Check if an existing lock is stale (from crashed process)."""
        pid, timestamp = self._read_metadata()
        
        if pid is None or timestamp is None:
            return True  # No metadata = stale
        
        # Check if holding process is still alive
        if not is_process_running(pid):
            logger.info(f"Lock holder (PID {pid}) is no longer running - lock is stale")
            return True
        
        # Check if heartbeat is too old
        age = time.time() - timestamp
        if age > self.lock_ttl:
            logger.info(f"Lock heartbeat is {age:.1f}s old (TTL: {self.lock_ttl}s) - lock is stale")
            return True
        
        return False
    
    def _heartbeat_loop(self) -> None:
        """Background thread that updates the heartbeat timestamp."""
        logger.debug("Heartbeat thread started")
        while not self._heartbeat_stop_event.is_set():
            try:
                self._write_metadata()
            except Exception as e:
                logger.warning(f"Heartbeat update failed: {e}")
            
            # Wait for interval or stop event
            self._heartbeat_stop_event.wait(self.heartbeat_interval)
        logger.debug("Heartbeat thread stopped")
    
    def _start_heartbeat(self) -> None:
        """Start the heartbeat background thread."""
        if not self.enable_heartbeat:
            return
        
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="ProcessLock-Heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
    
    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat background thread."""
        if self._heartbeat_thread is not None:
            self._heartbeat_stop_event.set()
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
    
    def acquire(self) -> bool:
        """
        Attempt to acquire the process lock.
        
        Returns:
            True if lock was acquired, False if another process holds it.
        
        Note:
            If the existing lock is stale (holder crashed), it will be
            forcibly acquired after clearing the stale lock.
        """
        if not FILELOCK_AVAILABLE:
            logger.error(
                "CRITICAL: filelock library not available! "
                "Install with: pip install filelock. "
                "Running without lock protection - RISK OF DATA CORRUPTION!"
            )
            # Return False to indicate lock failure, not True (which would be a lie)
            return False
        
        try:
            # Try to acquire the lock (non-blocking)
            self._lock.acquire(timeout=0)
            self._locked = True
            self._write_metadata()
            self._start_heartbeat()
            logger.info(f"Process lock acquired: {self.lock_path} (PID: {self._pid})")
            return True
            
        except FileLockTimeout:
            # Lock is held by another process - check if stale
            if self._is_lock_stale():
                logger.warning("Stale lock detected - attempting to remove lock file")
                try:
                    # Remove the lock file directly (stale lock)
                    # Note: This is safe because we've verified the holder is dead/stale
                    if os.path.exists(self.lock_path):
                        try:
                            os.remove(self.lock_path)
                        except OSError as e:
                            logger.error(f"Failed to remove stale lock file: {e}")
                            return False
                    
                    self._clear_metadata()
                    
                    # Try again with fresh FileLock
                    self._lock = FileLock(self.lock_path, timeout=0)
                    self._lock.acquire(timeout=0)
                    self._locked = True
                    self._write_metadata()
                    self._start_heartbeat()
                    logger.info(f"Process lock acquired (after clearing stale): {self.lock_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to acquire lock after clearing stale: {e}")
                    return False
            else:
                pid, _ = self._read_metadata()
                logger.warning(
                    f"Process lock held by another process (PID: {pid}). "
                    f"Cannot start - would cause split-brain condition."
                )
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error acquiring lock: {e}")
            return False
    
    def release(self) -> None:
        """Release the process lock."""
        self._stop_heartbeat()
        
        if self._lock is not None and self._locked:
            try:
                self._lock.release()
                self._clear_metadata()
                logger.info(f"Process lock released: {self.lock_path}")
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self._locked = False
    
    def is_locked(self) -> bool:
        """Check if this instance holds the lock."""
        return self._locked
    
    def get_heartbeat_status(self) -> dict:
        """Get status information about the heartbeat."""
        pid, timestamp = self._read_metadata()
        # Store thread reference to avoid race condition
        thread = self._heartbeat_thread
        return {
            "locked": self._locked,
            "lock_path": self.lock_path,
            "pid": pid,
            "last_heartbeat": timestamp,
            "heartbeat_age": time.time() - timestamp if timestamp else None,
            "thread_alive": thread.is_alive() if thread is not None else False,
        }
    
    def __enter__(self) -> 'ProcessLock':
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.release()
        return False
    
    def __repr__(self) -> str:
        return f"ProcessLock(path={self.lock_path}, locked={self._locked})"


# Module-level singleton
_process_lock: Optional[ProcessLock] = None


def get_process_lock() -> Optional[ProcessLock]:
    """Get the global process lock instance."""
    return _process_lock


def set_process_lock(lock: ProcessLock) -> None:
    """Set the global process lock instance."""
    global _process_lock
    _process_lock = lock


def create_and_acquire_lock(lock_path: str = None) -> Optional[ProcessLock]:
    """Create a process lock and attempt to acquire it."""
    lock = ProcessLock(lock_path=lock_path)
    if lock.acquire():
        set_process_lock(lock)
        return lock
    return None


__all__ = [
    "ProcessLock",
    "FILELOCK_AVAILABLE",
    "FCNTL_AVAILABLE",  # Backwards compatibility
    "get_process_lock",
    "set_process_lock",
    "create_and_acquire_lock",
    "is_process_running",
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_LOCK_TTL_SECONDS",
]
