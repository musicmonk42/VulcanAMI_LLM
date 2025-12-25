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
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module metadata
__version__ = "1.0.0"
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
    """
    
    def __init__(
        self,
        storage_path: str = "data/distillation",
        use_encryption: bool = False,
        encryption_key: Optional[str] = None,
        max_file_size_mb: int = 100,
    ):
        """
        Initialize storage backend.
        
        Args:
            storage_path: Base directory for storage
            use_encryption: Enable encryption at rest
            encryption_key: Fernet key for encryption (auto-generated if not provided)
            max_file_size_mb: Max size per JSONL file before rotation
        """
        self.storage_path = Path(storage_path)
        self.use_encryption = use_encryption
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.logger = logging.getLogger("DistillationStorage")
        
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
                    # Generate and log key (should be stored securely in production)
                    key = Fernet.generate_key()
                    self._fernet = Fernet(key)
                    self.logger.warning(
                        "Generated new encryption key. Store this securely: "
                        f"{key.decode()[:20]}..."
                    )
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
    
    def append_example(self, example: Dict[str, Any]) -> bool:
        """
        Append a training example to storage.
        
        Uses JSONL format (one JSON object per line) for efficient appending.
        
        Args:
            example: The example dictionary to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check file rotation
            if self._examples_file.exists():
                if self._examples_file.stat().st_size > self.max_file_size_bytes:
                    self._rotate_file(self._examples_file)
            
            # Serialize and optionally encrypt
            line = json.dumps(example, separators=(',', ':'))
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
        
        Args:
            limit: Maximum number of examples to read
            offset: Number of examples to skip
            
        Returns:
            List of example dictionaries
        """
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
    
    def _rotate_file(self, file_path: Path):
        """Rotate file when it exceeds max size."""
        timestamp = int(time.time())
        rotated_path = file_path.with_suffix(f".{timestamp}.jsonl")
        file_path.rename(rotated_path)
        self.logger.info(f"Rotated {file_path} to {rotated_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = dict(self._metadata)
        
        if self._examples_file.exists():
            stats["file_size_bytes"] = self._examples_file.stat().st_size
        else:
            stats["file_size_bytes"] = 0
        
        stats["encryption_enabled"] = self.use_encryption
        
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


__all__ = ["DistillationStorageBackend"]
