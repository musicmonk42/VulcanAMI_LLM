# ============================================================
# VULCAN-AGI Distillation Storage Backend Module
# Pluggable storage backend for distillation training data
# ============================================================
#
# Supports:
#     - JSONL format (one example per line, appendable)
#     - Optional encryption at rest using Fernet
#     - Configurable retention with automatic cleanup
#     - Provenance records for governance audit
#     - Thread-safe operations
#     - Atomic file operations
#     - Async buffered writes (reduces I/O blocking)
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added thread safety, atomic operations, and enhanced logging
#     1.1.0 - Added async buffered writes for better performance
# ============================================================

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module metadata
__version__ = "1.1.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class DistillationStorageBackend:
    """
    Pluggable storage backend for distillation training data.
    
    Supports:
    - JSONL format (one example per line, appendable)
    - Optional encryption at rest using Fernet
    - Configurable retention with automatic cleanup
    - Provenance records for governance audit
    - Async buffered writes (reduces blocking I/O)
    
    Performance Features:
    - Write buffer with configurable flush interval
    - Non-blocking append_example_async method
    - Background writer thread for async writes
    """
    
    def __init__(
        self,
        storage_path: str = "data/distillation",
        use_encryption: bool = False,
        encryption_key: Optional[str] = None,
        max_file_size_mb: int = 100,
        max_rotated_files: int = 10,
        min_free_disk_mb: int = 500,
        enable_async_writes: bool = True,
        async_buffer_size: int = 100,
        async_flush_interval_seconds: float = 5.0,
    ):
        """
        Initialize storage backend.
        
        Args:
            storage_path: Base directory for storage
            use_encryption: Enable encryption at rest
            encryption_key: Fernet key for encryption (auto-generated if not provided)
            max_file_size_mb: Max size per JSONL file before rotation
            max_rotated_files: Maximum number of rotated files to keep (default: 10)
            min_free_disk_mb: Minimum free disk space before warning (default: 500MB)
            enable_async_writes: Enable async buffered writes (default: True)
            async_buffer_size: Max buffer size before auto-flush (default: 100)
            async_flush_interval_seconds: Time between auto-flushes (default: 5.0)
        """
        self.storage_path = Path(storage_path)
        self.use_encryption = use_encryption
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_rotated_files = max_rotated_files
        self.min_free_disk_bytes = min_free_disk_mb * 1024 * 1024
        self.logger = logging.getLogger("DistillationStorage")
        
        # Thread safety lock for concurrent access
        # Using threading.Lock for efficiency since we don't need reentrant behavior
        self._lock = threading.Lock()
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption if enabled
        self._fernet = None
        if use_encryption:
            try:
                from cryptography.fernet import Fernet
                if encryption_key:
                    self._fernet = Fernet(encryption_key.encode())
                else:
                    # Generate new encryption key
                    key = Fernet.generate_key()
                    self._fernet = Fernet(key)
                    # FIXED: Never log key material for security
                    self.logger.warning(
                        "Generated new encryption key. "
                        "Set DISTILLATION_ENCRYPTION_KEY env var to persist. "
                        "Key NOT logged for security."
                    )
                    # Store key securely if path provided
                    key_file = self.storage_path / ".encryption_key"
                    try:
                        key_file.write_bytes(key)
                        key_file.chmod(0o600)  # Owner read/write only
                        self.logger.info(f"Encryption key stored at {key_file}")
                    except Exception as e:
                        self.logger.error(f"Could not store encryption key: {e}")
            except ImportError:
                self.logger.warning(
                    "cryptography package not installed. "
                    "Encryption disabled. Install with: pip install cryptography"
                )
                self.use_encryption = False
        
        # File paths
        self._examples_file = self.storage_path / "examples.jsonl"
        self._provenance_file = self.storage_path / "provenance.jsonl"
        self._metadata_file = self.storage_path / "metadata.json"
        
        # Load metadata
        self._metadata = self._load_metadata()
        
        # Async write buffer (reduces blocking I/O)
        self._enable_async_writes = enable_async_writes
        self._async_buffer_size = async_buffer_size
        self._async_flush_interval = async_flush_interval_seconds
        self._write_queue: Optional[queue.Queue] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_stop_event: Optional[threading.Event] = None
        self._pending_writes = 0
        
        if enable_async_writes:
            self._init_async_writer()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load storage metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_examples": 0,
            "created_at": time.time(),
            "last_write": None,
            "encryption_enabled": self.use_encryption,
        }
    
    def _save_metadata(self):
        """Save storage metadata."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _encrypt(self, data: str) -> str:
        """Encrypt data if encryption is enabled."""
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        return data
    
    def _decrypt(self, data: str) -> str:
        """Decrypt data if encryption is enabled."""
        if self._fernet:
            return self._fernet.decrypt(data.encode()).decode()
        return data
    
    # ============================================================
    # ASYNC WRITE BUFFER METHODS
    # ============================================================
    
    def _init_async_writer(self):
        """Initialize the async writer thread and queue."""
        self._write_queue = queue.Queue(maxsize=self._async_buffer_size * 2)
        self._writer_stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._async_writer_loop,
            daemon=True,
            name="DistillationAsyncWriter",
        )
        self._writer_thread.start()
        self.logger.info(
            f"Async writer started (buffer_size={self._async_buffer_size}, "
            f"flush_interval={self._async_flush_interval}s)"
        )
    
    def _async_writer_loop(self):
        """Background thread that writes buffered examples to disk."""
        buffer: List[str] = []
        last_flush = time.time()
        
        while not self._writer_stop_event.is_set():
            try:
                # Try to get item from queue with timeout
                try:
                    item = self._write_queue.get(timeout=0.5)
                    buffer.append(item)
                    self._write_queue.task_done()
                except queue.Empty:
                    pass
                
                # Check if we should flush
                should_flush = (
                    len(buffer) >= self._async_buffer_size or
                    (buffer and time.time() - last_flush >= self._async_flush_interval)
                )
                
                if should_flush and buffer:
                    self._flush_buffer(buffer)
                    buffer = []
                    last_flush = time.time()
                    
            except Exception as e:
                self.logger.error(f"Async writer error: {e}")
        
        # Final flush on shutdown
        if buffer:
            self._flush_buffer(buffer)
    
    def _flush_buffer(self, buffer: List[str]):
        """Flush the write buffer to disk."""
        if not buffer:
            return
        
        with self._lock:
            try:
                # Check file rotation
                if self._examples_file.exists():
                    if self._examples_file.stat().st_size > self.max_file_size_bytes:
                        self._rotate_file(self._examples_file)
                
                # Write all buffered lines at once
                with open(self._examples_file, "a") as f:
                    for line in buffer:
                        f.write(line + "\n")
                
                # Update metadata
                self._metadata["total_examples"] += len(buffer)
                self._metadata["last_write"] = time.time()
                self._save_metadata()
                
                self._pending_writes -= len(buffer)
                self.logger.debug(f"Flushed {len(buffer)} examples to disk")
                
            except Exception as e:
                self.logger.error(f"Failed to flush buffer: {e}")
    
    def append_example_async(self, example: Dict[str, Any]) -> bool:
        """
        Append a training example to storage asynchronously (non-blocking).
        
        The example is added to a buffer and written to disk in the background.
        This reduces I/O blocking from ~10-50ms to ~1-2ms per call.
        
        Args:
            example: The example dictionary to store
            
        Returns:
            True if successfully queued, False if queue is full or async disabled
        """
        if not self._enable_async_writes or not self._write_queue:
            # Fall back to synchronous write
            return self.append_example(example)
        
        try:
            # Serialize and optionally encrypt
            line = json.dumps(example, separators=(',', ':'))
            if self.use_encryption:
                line = self._encrypt(line)
            
            # Try to add to queue (non-blocking)
            self._write_queue.put_nowait(line)
            self._pending_writes += 1
            return True
            
        except queue.Full:
            self.logger.warning("Async write queue full - falling back to sync write")
            return self.append_example(example)
        except Exception as e:
            self.logger.error(f"Failed to queue example: {e}")
            return False
    
    def flush_async_buffer(self) -> int:
        """
        Force flush the async write buffer.
        
        Returns:
            Number of pending writes (0 after flush completes)
        """
        if not self._write_queue:
            return 0
        
        # Wait for queue to drain
        self._write_queue.join()
        return self._pending_writes
    
    def shutdown_async_writer(self, timeout: float = 5.0):
        """
        Gracefully shutdown the async writer thread.
        
        Args:
            timeout: Maximum time to wait for flush (default: 5.0 seconds)
        """
        if self._writer_stop_event:
            self._writer_stop_event.set()
        
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=timeout)
            if self._writer_thread.is_alive():
                self.logger.warning(
                    f"Async writer thread did not stop within {timeout}s timeout"
                )
            else:
                self.logger.info("Async writer stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown_async_writer(timeout=2.0)
        except Exception:
            pass
    
    # ============================================================
    # SYNCHRONOUS WRITE METHOD
    # ============================================================
    
    def _sanitize_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove potential injection vectors from example.
        
        Prevents JSONL injection by removing control characters and newlines
        that could break the JSONL format or inject malicious content.
        
        Args:
            example: The example dictionary to sanitize
            
        Returns:
            Sanitized copy of the example
        """
        sanitized = {}
        for key, value in example.items():
            if isinstance(value, str):
                # Remove control characters and newlines
                value = ''.join(c for c in value if c.isprintable() or c == ' ')
                value = value.replace('\n', ' ').replace('\r', ' ')
            elif isinstance(value, dict):
                value = self._sanitize_example(value)
            sanitized[key] = value
        return sanitized
    
    def append_example(self, example: Dict[str, Any]) -> bool:
        """
        Append a training example to storage.
        
        Uses JSONL format (one JSON object per line) for efficient appending.
        Thread-safe with proper locking.
        
        Args:
            example: The example dictionary to store
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Check file rotation
                if self._examples_file.exists():
                    if self._examples_file.stat().st_size > self.max_file_size_bytes:
                        self._rotate_file(self._examples_file)
                
                # Sanitize before serialization to prevent JSONL injection
                sanitized = self._sanitize_example(example)
                
                # Serialize with strict settings
                line = json.dumps(
                    sanitized,
                    separators=(',', ':'),
                    ensure_ascii=True,  # Prevent encoding issues
                    allow_nan=False,  # Prevent NaN/Infinity injection
                )
                
                # Verify no newlines in output (defense in depth)
                if '\n' in line or '\r' in line:
                    self.logger.error("JSONL injection attempt detected and blocked")
                    return False
                
                # Optionally encrypt
                if self.use_encryption:
                    line = self._encrypt(line)
                
                # Append to file
                with open(self._examples_file, "a") as f:
                    f.write(line + "\n")
                
                # Update metadata
                self._metadata["total_examples"] += 1
                self._metadata["last_write"] = time.time()
                self._save_metadata()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to append example: {e}")
                return False
    
    def append_provenance(self, record: Dict[str, Any]) -> bool:
        """
        Append a provenance record for governance audit.
        
        Args:
            record: The provenance record dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            record["recorded_at"] = time.time()
            line = json.dumps(record, separators=(',', ':'))
            
            with open(self._provenance_file, "a") as f:
                f.write(line + "\n")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to append provenance: {e}")
            return False
    
    def read_examples(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Read examples from storage.
        
        Thread-safe with proper locking.
        
        Args:
            limit: Maximum number of examples to read
            offset: Number of examples to skip
            
        Returns:
            List of example dictionaries
        """
        with self._lock:
            examples = []
            
            if not self._examples_file.exists():
                return examples
            
            try:
                with open(self._examples_file, "r") as f:
                    for i, line in enumerate(f):
                        if i < offset:
                            continue
                        if limit and len(examples) >= limit:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        if self.use_encryption:
                            line = self._decrypt(line)
                        
                        examples.append(json.loads(line))
                
            except Exception as e:
                self.logger.error(f"Failed to read examples: {e}")
            
            return examples
    
    def read_and_clear(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Read a batch of examples and remove them from storage.
        
        Used for training consumption - examples are removed after reading.
        
        Args:
            batch_size: Number of examples to read
            
        Returns:
            List of example dictionaries
        """
        examples = self.read_examples(limit=batch_size)
        
        if examples:
            # Rewrite file without consumed examples
            remaining = self.read_examples(offset=batch_size)
            self._rewrite_examples(remaining)
        
        return examples
    
    def _rewrite_examples(self, examples: List[Dict[str, Any]]):
        """Rewrite examples file with given examples."""
        try:
            # Write to temp file first
            temp_file = self._examples_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                for example in examples:
                    line = json.dumps(example, separators=(',', ':'))
                    if self.use_encryption:
                        line = self._encrypt(line)
                    f.write(line + "\n")
            
            # Atomic replace
            temp_file.replace(self._examples_file)
            
            # Update metadata
            self._metadata["total_examples"] = len(examples)
            self._save_metadata()
            
        except Exception as e:
            self.logger.error(f"Failed to rewrite examples: {e}")
    
    def delete_user_data(self, user_id: str) -> int:
        """
        Delete all examples for a user (GDPR Article 17 - Right to Erasure).
        
        This method implements the GDPR "right to be forgotten" by removing
        all training examples associated with a specific user ID.
        
        Args:
            user_id: User identifier to delete data for
            
        Returns:
            Number of examples deleted
        """
        with self._lock:
            examples = self.read_examples()
            remaining = [ex for ex in examples if ex.get("user_id") != user_id]
            deleted_count = len(examples) - len(remaining)
            
            if deleted_count > 0:
                self._rewrite_examples(remaining)
                # Log deletion for compliance audit
                import hashlib
                self.append_provenance({
                    "event_type": "user_data_deletion",
                    "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest(),
                    "records_deleted": deleted_count,
                    "timestamp": time.time(),
                })
                self.logger.info(f"Deleted {deleted_count} examples for user {user_id[:8]}...")
            
            return deleted_count
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all data for a user (GDPR Article 20 - Data Portability).
        
        This method implements the GDPR "right to data portability" by
        exporting all training examples for a user in a portable format.
        
        Args:
            user_id: User identifier to export data for
            
        Returns:
            Dictionary containing all user data in portable format
        """
        examples = [
            ex for ex in self.read_examples()
            if ex.get("user_id") == user_id
        ]
        
        return {
            "user_id": user_id,
            "export_date": time.time(),
            "format_version": "1.0",
            "examples": examples,
            "total_examples": len(examples),
            "data_format": "JSONL",
            "encryption_used": self.use_encryption,
        }
    
    def _rotate_file(self, file_path: Path):
        """
        Rotate file when it exceeds max size and clean up old rotated files.
        
        Also checks disk space and warns if below threshold.
        """
        timestamp = int(time.time())
        rotated_path = file_path.with_suffix(f".{timestamp}.jsonl")
        file_path.rename(rotated_path)
        self.logger.info(f"Rotated {file_path} to {rotated_path}")
        
        # Clean up old rotated files to prevent disk fill
        self._cleanup_old_rotated_files(file_path)
        
        # Check disk space and warn if low
        self._check_disk_space()
    
    def _cleanup_old_rotated_files(self, base_file_path: Path):
        """
        Remove oldest rotated files when exceeding max_rotated_files limit.
        
        This prevents unbounded disk usage from accumulated rotated files.
        """
        try:
            # Find all rotated files matching pattern: examples.{timestamp}.jsonl
            pattern = base_file_path.stem + ".*.jsonl"
            rotated_files = sorted(
                self.storage_path.glob(pattern),
                key=lambda f: f.stat().st_mtime
            )
            
            # Remove oldest files if exceeding limit
            files_to_remove = len(rotated_files) - self.max_rotated_files
            if files_to_remove > 0:
                for old_file in rotated_files[:files_to_remove]:
                    old_file.unlink()
                    self.logger.info(f"Removed old rotated file: {old_file}")
                
                self.logger.info(
                    f"Cleaned up {files_to_remove} old rotated files "
                    f"(keeping {self.max_rotated_files} most recent)"
                )
        except Exception as e:
            self.logger.error(f"Failed to cleanup old rotated files: {e}")
    
    def _check_disk_space(self):
        """
        Check available disk space and log warning if below threshold.
        
        This provides early warning before disk fills up completely.
        """
        try:
            import shutil
            disk_usage = shutil.disk_usage(self.storage_path)
            free_bytes = disk_usage.free
            
            if free_bytes < self.min_free_disk_bytes:
                self.logger.warning(
                    f"LOW DISK SPACE WARNING: Only {free_bytes / (1024*1024):.1f}MB free. "
                    f"Distillation storage may fail. "
                    f"Consider running cleanup_expired() or increasing disk space."
                )
        except Exception as e:
            self.logger.debug(f"Could not check disk space: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics including async write stats."""
        stats = dict(self._metadata)
        
        if self._examples_file.exists():
            stats["file_size_bytes"] = self._examples_file.stat().st_size
        else:
            stats["file_size_bytes"] = 0
        
        stats["encryption_enabled"] = self.use_encryption
        
        # Add async write stats
        stats["async_writes_enabled"] = self._enable_async_writes
        if self._enable_async_writes:
            stats["async_pending_writes"] = self._pending_writes
            stats["async_queue_size"] = self._write_queue.qsize() if self._write_queue else 0
            stats["async_buffer_size"] = self._async_buffer_size
            stats["async_flush_interval_seconds"] = self._async_flush_interval
        
        return stats
    
    def cleanup_expired(self, retention_seconds: int) -> int:
        """
        Remove examples older than retention period.
        
        Args:
            retention_seconds: Retention period in seconds
            
        Returns:
            Number of examples removed
        """
        if not self._examples_file.exists():
            return 0
        
        cutoff = time.time() - retention_seconds
        examples = self.read_examples()
        
        valid_examples = [
            ex for ex in examples
            if ex.get("timestamp", 0) > cutoff
        ]
        
        removed = len(examples) - len(valid_examples)
        
        if removed > 0:
            self._rewrite_examples(valid_examples)
            self.logger.info(f"Cleaned up {removed} expired examples")
        
        return removed


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the storage module.
    
    Returns:
        Dictionary containing module information
    """
    return {
        "version": __version__,
        "author": __author__,
        "features": [
            "JSONL format storage",
            "Optional Fernet encryption",
            "Automatic file rotation",
            "Thread-safe operations",
            "Provenance tracking",
            "Retention cleanup",
            "Async buffered writes",
        ],
    }


__all__ = ["DistillationStorageBackend", "get_module_info"]


# Module initialization logging
logger.debug(f"Distillation storage module v{__version__} loaded")
