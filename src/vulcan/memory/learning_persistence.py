"""
Learning State Persistence Module.

This module provides enterprise-grade persistence for the VULCAN learning system's
state, including tool weights, concepts, and contraindications. State is saved
to disk after every update and loaded on initialization to ensure learning
persists across queries and server restarts.

Key Features:
    - Atomic file writes to prevent corruption
    - Automatic backup rotation with configurable retention
    - Thread-safe operations with proper locking
    - Graceful degradation when storage is unavailable
    - Schema versioning for forward/backward compatibility
    - Comprehensive validation and error handling
    - In-memory caching for performance
    - Detailed logging for observability

Storage Configuration:
    - Path: Environment variable VULCAN_STORAGE_PATH (default: /mnt/vulcan-data)
    - File: learning_state.json
    - Backups: learning_state.json.backup.{N}

Example Usage:
    >>> persistence = LearningStatePersistence()
    >>> persistence.update_tool_weights({"general": 0.035, "code": 0.020})
    True
    >>> weights = persistence.get_tool_weights()
    >>> print(weights)
    {'general': 0.035, 'code': 0.020}

Security Considerations:
    - No sensitive data is stored (only statistical weights)
    - File permissions follow system defaults
    - No network access required

Copyright (c) 2024 VULCAN-AGI Project
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Default storage path (Railway volume mount point)
DEFAULT_STORAGE_PATH = "/mnt/vulcan-data"

# Schema version for migration support
SCHEMA_VERSION = "1.0.0"

# Backup configuration
MAX_BACKUP_COUNT = 5
BACKUP_SUFFIX = ".backup"

# Validation constraints
MAX_TOOL_NAME_LENGTH = 256
MAX_TOOL_WEIGHT_VALUE = 1.0
MIN_TOOL_WEIGHT_VALUE = -1.0
MAX_TOOL_COUNT = 10000


# =============================================================================
# Type Definitions
# =============================================================================


class StateMetadata(TypedDict, total=False):
    """Type definition for state metadata."""
    
    version: str
    schema_version: str
    created_at: float
    updated_at: float
    load_count: int
    save_count: int
    cleared_at: Optional[float]
    checksum: Optional[str]
    hostname: Optional[str]


class LearningState(TypedDict, total=False):
    """Type definition for the complete learning state."""
    
    tool_weights: Dict[str, float]
    concept_library: Dict[str, Any]
    contraindications: Dict[str, Any]
    metadata: StateMetadata


@dataclass
class PersistenceStats:
    """Statistics about the persistence layer."""
    
    storage_path: str
    file_exists: bool
    file_size_bytes: int
    tool_weights_count: int
    concept_library_count: int
    contraindications_count: int
    backup_count: int
    last_save_time: Optional[float]
    last_load_time: Optional[float]
    total_saves: int
    total_loads: int
    errors_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Custom Exceptions
# =============================================================================


class PersistenceError(Exception):
    """Base exception for persistence-related errors."""
    pass


class ValidationError(PersistenceError):
    """Raised when state validation fails."""
    pass


class StorageError(PersistenceError):
    """Raised when storage operations fail."""
    pass


# =============================================================================
# Main Persistence Class
# =============================================================================


class LearningStatePersistence:
    """
    Enterprise-grade persistence for VULCAN learning system state.
    
    This class provides thread-safe, atomic persistence of learning state
    including tool weights, concept libraries, and contraindications. It
    implements industry best practices including:
    
    - **Atomic Writes**: Uses write-to-temp-then-rename pattern to prevent
      corruption from interrupted writes or power failures.
    
    - **Backup Rotation**: Automatically maintains backup files with
      configurable retention policy.
    
    - **Thread Safety**: All public methods are protected by reentrant locks
      for safe concurrent access.
    
    - **Graceful Degradation**: Operations continue (in-memory only) even
      when disk storage is unavailable.
    
    - **Schema Versioning**: Supports forward/backward compatibility through
      versioned schemas with migration support.
    
    - **Validation**: All input is validated before persistence to prevent
      corrupt state.
    
    - **Observability**: Comprehensive logging and statistics for monitoring.
    
    Attributes:
        storage_path: Directory path for storage.
        filename: Name of the state file.
        state_file: Full path to the state file.
    
    Example:
        >>> # Initialize with default path
        >>> persistence = LearningStatePersistence()
        
        >>> # Or with custom path for testing
        >>> persistence = LearningStatePersistence(storage_path="/tmp/test")
        
        >>> # Update tool weights (automatically persists)
        >>> persistence.update_tool_weights({"general": 0.035})
        True
        
        >>> # Get current weights
        >>> weights = persistence.get_tool_weights()
        {'general': 0.035}
        
        >>> # Get statistics
        >>> stats = persistence.get_stats()
    """
    
    __slots__ = (
        "storage_path",
        "filename", 
        "state_file",
        "_lock",
        "_cached_state",
        "_dirty",
        "_storage_available",
        "_stats",
    )
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        filename: str = "learning_state.json",
    ) -> None:
        """
        Initialize the persistence layer.
        
        Creates the storage directory if it doesn't exist and loads any
        existing state from disk into the in-memory cache.
        
        Args:
            storage_path: Directory for storage. Defaults to VULCAN_STORAGE_PATH
                         environment variable, or /mnt/vulcan-data if not set.
            filename: Name of the state file. Defaults to "learning_state.json".
        
        Raises:
            ValueError: If filename is empty or contains invalid characters.
        
        Example:
            >>> persistence = LearningStatePersistence()
            >>> persistence = LearningStatePersistence(storage_path="/custom/path")
            >>> persistence = LearningStatePersistence(filename="custom_state.json")
        """
        # Validate filename
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")
        if not filename.endswith(".json"):
            raise ValueError("filename must end with .json")
        if "/" in filename or "\\" in filename:
            raise ValueError("filename cannot contain path separators")
        
        # Determine storage path from environment or default
        if storage_path is None:
            storage_path = os.environ.get("VULCAN_STORAGE_PATH", DEFAULT_STORAGE_PATH)
        
        self.storage_path: Path = Path(storage_path)
        self.filename: str = filename
        self.state_file: Path = self.storage_path / filename
        
        # Thread safety - use RLock for reentrant locking
        self._lock: threading.RLock = threading.RLock()
        
        # In-memory state cache (reduces disk reads)
        self._cached_state: Optional[LearningState] = None
        self._dirty: bool = False
        
        # Track storage availability for graceful degradation
        self._storage_available: bool = False
        
        # Statistics for observability
        self._stats: Dict[str, Any] = {
            "total_saves": 0,
            "total_loads": 0,
            "errors_count": 0,
            "last_save_time": None,
            "last_load_time": None,
        }
        
        # Initialize storage directory
        self._storage_available = self._ensure_storage_directory()
        
        logger.info(
            f"[LearningStatePersistence] Initialized: path={self.state_file}, "
            f"storage_available={self._storage_available}"
        )
    
    # =========================================================================
    # Public API - State Operations
    # =========================================================================
    
    def load_state(self) -> LearningState:
        """
        Load learning state from disk.
        
        Attempts to load state from the primary file, falling back to
        backups if the primary is corrupted. Returns cached state if
        available to minimize disk I/O.
        
        Returns:
            Dictionary containing the learning state with keys:
            - tool_weights: Dict[str, float]
            - concept_library: Dict[str, Any]
            - contraindications: Dict[str, Any]
            - metadata: StateMetadata
        
        Note:
            If no state file exists or loading fails, returns a default
            empty state. This method never raises exceptions - it always
            returns a valid state dictionary.
        
        Example:
            >>> state = persistence.load_state()
            >>> print(state["tool_weights"])
            {'general': 0.035}
        """
        with self._lock:
            # Return cached state if available (copy to prevent mutation)
            if self._cached_state is not None:
                return copy.deepcopy(self._cached_state)
            
            default_state = self._create_default_state()
            
            # Try to load from primary file
            if self.state_file.exists():
                try:
                    state = self._load_from_file(self.state_file)
                    if state is not None:
                        # Validate and migrate schema if needed
                        state = self._validate_and_migrate(state, default_state)
                        self._cached_state = state
                        self._update_load_stats()
                        return copy.deepcopy(state)
                except Exception as e:
                    logger.error(
                        f"[LearningStatePersistence] Failed to load primary state: {e}"
                    )
                    self._stats["errors_count"] += 1
            
            # Try backup files in order
            backup_state = self._try_load_from_backups(default_state)
            if backup_state is not None:
                self._cached_state = backup_state
                self._update_load_stats()
                return copy.deepcopy(backup_state)
            
            # No existing state found - use default
            logger.info(
                f"[LearningStatePersistence] No existing state at {self.state_file}, "
                "starting fresh"
            )
            self._cached_state = default_state
            self._update_load_stats()
            return copy.deepcopy(default_state)
    
    def save_state(self, state: LearningState) -> bool:
        """
        Save learning state to disk with atomic write.
        
        Uses a write-to-temp-then-rename pattern to ensure atomicity.
        Creates a backup of the existing file before overwriting.
        
        Args:
            state: Dictionary containing the learning state to save.
                  Must include tool_weights, concept_library, contraindications,
                  and metadata keys.
        
        Returns:
            True if save succeeded, False otherwise.
        
        Note:
            Even if disk write fails, the in-memory cache is updated.
            This allows the system to continue operating with recent
            data even when storage is temporarily unavailable.
        
        Example:
            >>> state = persistence.load_state()
            >>> state["tool_weights"]["new_tool"] = 0.05
            >>> success = persistence.save_state(state)
            >>> print(success)
            True
        """
        with self._lock:
            # Validate state before saving
            try:
                self._validate_state(state)
            except ValidationError as e:
                logger.error(f"[LearningStatePersistence] Validation failed: {e}")
                self._stats["errors_count"] += 1
                return False
            
            # Update metadata
            state = self._update_metadata_for_save(state)
            
            # Update in-memory cache (always succeeds)
            self._cached_state = copy.deepcopy(state)
            self._dirty = False
            
            # Attempt disk write
            if not self._storage_available:
                logger.warning(
                    "[LearningStatePersistence] Storage unavailable, "
                    "state cached in memory only"
                )
                return True  # Cache updated successfully
            
            try:
                # Create backup of existing file
                if self.state_file.exists():
                    self._rotate_backups()
                
                # Atomic write
                success = self._atomic_write(state)
                
                if success:
                    self._stats["total_saves"] += 1
                    self._stats["last_save_time"] = time.time()
                    
                    tool_count = len(state.get("tool_weights", {}))
                    save_count = state.get("metadata", {}).get("save_count", 0)
                    logger.info(
                        f"[LearningStatePersistence] Saved: {tool_count} tool weights, "
                        f"save_count={save_count}"
                    )
                
                return success
                
            except Exception as e:
                logger.error(
                    f"[LearningStatePersistence] Save failed: {e}"
                )
                self._stats["errors_count"] += 1
                return False
    
    def update_tool_weights(self, tool_weights: Dict[str, float]) -> bool:
        """
        Update tool weights and persist to disk.
        
        This is the primary method for updating tool weights. It loads
        the current state, updates the weights, and saves atomically.
        
        Args:
            tool_weights: Dictionary mapping tool names to weight adjustments.
                         Keys must be non-empty strings, values must be floats
                         within the valid range.
        
        Returns:
            True if update and save succeeded, False otherwise.
        
        Raises:
            ValidationError: If tool_weights contains invalid data.
        
        Example:
            >>> success = persistence.update_tool_weights({
            ...     "general": 0.035,
            ...     "code": 0.020,
            ...     "search": -0.005
            ... })
            >>> print(success)
            True
        """
        # Validate tool weights
        self._validate_tool_weights(tool_weights)
        
        with self._lock:
            state = self.load_state()
            state["tool_weights"] = copy.deepcopy(tool_weights)
            return self.save_state(state)
    
    def get_tool_weights(self) -> Dict[str, float]:
        """
        Get the persisted tool weights.
        
        Returns:
            Dictionary mapping tool names to weight adjustments.
            Returns empty dict if no weights are persisted.
        
        Example:
            >>> weights = persistence.get_tool_weights()
            >>> print(weights.get("general", 0.0))
            0.035
        """
        state = self.load_state()
        return copy.deepcopy(state.get("tool_weights", {}))
    
    def update_concept_library(self, concept_library: Dict[str, Any]) -> bool:
        """
        Update concept library and persist to disk.
        
        Args:
            concept_library: Dictionary of learned concepts.
        
        Returns:
            True if update and save succeeded, False otherwise.
        """
        with self._lock:
            state = self.load_state()
            state["concept_library"] = copy.deepcopy(concept_library)
            return self.save_state(state)
    
    def update_contraindications(self, contraindications: Dict[str, Any]) -> bool:
        """
        Update contraindications and persist to disk.
        
        Args:
            contraindications: Dictionary of learned contraindications.
        
        Returns:
            True if update and save succeeded, False otherwise.
        """
        with self._lock:
            state = self.load_state()
            state["contraindications"] = copy.deepcopy(contraindications)
            return self.save_state(state)
    
    def clear_state(self) -> bool:
        """
        Clear all persisted state and reset to defaults.
        
        Creates a backup before clearing and resets all data including
        tool weights, concept library, and contraindications.
        
        Returns:
            True if clear and save succeeded, False otherwise.
        
        Example:
            >>> persistence.clear_state()
            True
            >>> weights = persistence.get_tool_weights()
            >>> print(weights)
            {}
        """
        with self._lock:
            default_state = self._create_default_state()
            default_state["metadata"]["cleared_at"] = time.time()
            
            logger.info("[LearningStatePersistence] Clearing state")
            return self.save_state(default_state)
    
    # =========================================================================
    # Public API - Statistics and Information
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the persistence layer.
        
        Returns:
            Dictionary containing:
            - storage_path: Path to state file
            - file_exists: Whether state file exists
            - file_size_bytes: Size of state file
            - tool_weights_count: Number of tool weights
            - concept_library_count: Number of concepts
            - contraindications_count: Number of contraindications
            - backup_count: Number of backup files
            - total_saves: Total successful saves
            - total_loads: Total loads
            - errors_count: Total errors encountered
            - last_save_time: Timestamp of last save
            - last_load_time: Timestamp of last load
            - storage_available: Whether storage is accessible
            - metadata: State metadata
        
        Example:
            >>> stats = persistence.get_stats()
            >>> print(f"Saves: {stats['total_saves']}")
            Saves: 42
        """
        state = self.load_state()
        
        # Get file size
        file_size = 0
        if self.state_file.exists():
            try:
                file_size = self.state_file.stat().st_size
            except OSError:
                pass
        
        # Count backups
        backup_count = len(self._list_backups())
        
        return {
            "storage_path": str(self.state_file),
            "file_exists": self.state_file.exists(),
            "file_size_bytes": file_size,
            "tool_weights_count": len(state.get("tool_weights", {})),
            "concept_library_count": len(state.get("concept_library", {})),
            "contraindications_count": len(state.get("contraindications", {})),
            "backup_count": backup_count,
            "total_saves": self._stats["total_saves"],
            "total_loads": self._stats["total_loads"],
            "errors_count": self._stats["errors_count"],
            "last_save_time": self._stats["last_save_time"],
            "last_load_time": self._stats["last_load_time"],
            "storage_available": self._storage_available,
            "metadata": state.get("metadata", {}),
        }
    
    def is_storage_available(self) -> bool:
        """
        Check if persistent storage is available.
        
        Returns:
            True if storage directory is writable, False otherwise.
        """
        return self._storage_available
    
    # =========================================================================
    # Private Methods - File Operations
    # =========================================================================
    
    def _ensure_storage_directory(self) -> bool:
        """
        Ensure the storage directory exists and is writable.
        
        Returns:
            True if directory exists/was created and is writable, False otherwise.
        """
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Test write access with a temporary file
            test_file = self.storage_path / ".write_test"
            try:
                test_file.write_text("test", encoding="utf-8")
                test_file.unlink()
            except (OSError, PermissionError):
                logger.warning(
                    f"[LearningStatePersistence] Storage directory not writable: "
                    f"{self.storage_path}"
                )
                return False
            
            return True
            
        except (OSError, PermissionError) as e:
            logger.warning(
                f"[LearningStatePersistence] Cannot create storage directory "
                f"{self.storage_path}: {e}"
            )
            return False
    
    def _load_from_file(self, file_path: Path) -> Optional[LearningState]:
        """
        Load state from a specific file.
        
        Args:
            file_path: Path to the file to load.
        
        Returns:
            Loaded state dictionary or None if loading fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    return None
                state = json.loads(content)
                
            # Verify checksum if present
            if "metadata" in state and "checksum" in state["metadata"]:
                stored_checksum = state["metadata"]["checksum"]
                state_copy = copy.deepcopy(state)
                state_copy["metadata"].pop("checksum", None)
                computed_checksum = self._compute_checksum(state_copy)
                
                if stored_checksum != computed_checksum:
                    logger.warning(
                        f"[LearningStatePersistence] Checksum mismatch in {file_path}"
                    )
                    # Continue anyway - data may still be usable
            
            return state
            
        except json.JSONDecodeError as e:
            logger.error(
                f"[LearningStatePersistence] Invalid JSON in {file_path}: {e}"
            )
            return None
        except (OSError, IOError) as e:
            logger.error(
                f"[LearningStatePersistence] Failed to read {file_path}: {e}"
            )
            return None
    
    def _atomic_write(self, state: LearningState) -> bool:
        """
        Atomically write state to disk.
        
        Uses write-to-temp-then-rename pattern for atomicity.
        
        Args:
            state: State dictionary to write.
        
        Returns:
            True if write succeeded, False otherwise.
        """
        temp_file = self.state_file.with_suffix(".json.tmp")
        
        try:
            # Compute and add checksum
            state_with_checksum = copy.deepcopy(state)
            checksum = self._compute_checksum(state)
            state_with_checksum["metadata"]["checksum"] = checksum
            
            # Write to temp file with explicit encoding
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state_with_checksum, f, indent=2, default=str, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename (POSIX guarantees atomicity)
            temp_file.replace(self.state_file)
            
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"[LearningStatePersistence] Atomic write failed: {e}")
            
            # Clean up temp file
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass
            
            return False
    
    # =========================================================================
    # Private Methods - Backup Management
    # =========================================================================
    
    def _rotate_backups(self) -> None:
        """
        Rotate backup files, keeping only MAX_BACKUP_COUNT most recent.
        """
        if not self.state_file.exists():
            return
        
        try:
            # Get existing backups sorted by modification time (oldest first)
            backups = self._list_backups()
            
            # Remove excess backups
            while len(backups) >= MAX_BACKUP_COUNT:
                oldest = backups.pop(0)
                try:
                    oldest.unlink()
                except OSError as e:
                    logger.warning(f"[LearningStatePersistence] Failed to remove backup: {e}")
            
            # Create new backup
            backup_name = f"{self.filename}{BACKUP_SUFFIX}.{int(time.time())}"
            backup_path = self.storage_path / backup_name
            shutil.copy2(self.state_file, backup_path)
            
        except (OSError, IOError) as e:
            logger.warning(f"[LearningStatePersistence] Backup rotation failed: {e}")
    
    def _list_backups(self) -> List[Path]:
        """
        List all backup files sorted by modification time (oldest first).
        
        Returns:
            List of backup file paths.
        """
        pattern = f"{self.filename}{BACKUP_SUFFIX}.*"
        try:
            backups = list(self.storage_path.glob(pattern))
            backups.sort(key=lambda p: p.stat().st_mtime)
            return backups
        except OSError:
            return []
    
    def _try_load_from_backups(self, default_state: LearningState) -> Optional[LearningState]:
        """
        Attempt to load state from backup files.
        
        Tries backups in reverse chronological order (newest first).
        
        Args:
            default_state: Default state to use for validation/migration.
        
        Returns:
            Loaded state or None if all backups fail.
        """
        backups = self._list_backups()
        backups.reverse()  # Try newest first
        
        for backup_path in backups:
            try:
                state = self._load_from_file(backup_path)
                if state is not None:
                    state = self._validate_and_migrate(state, default_state)
                    logger.info(
                        f"[LearningStatePersistence] Recovered from backup: {backup_path}"
                    )
                    return state
            except Exception as e:
                logger.warning(
                    f"[LearningStatePersistence] Backup {backup_path} failed: {e}"
                )
                continue
        
        return None
    
    # =========================================================================
    # Private Methods - Validation
    # =========================================================================
    
    def _validate_state(self, state: LearningState) -> None:
        """
        Validate state structure and contents.
        
        Args:
            state: State dictionary to validate.
        
        Raises:
            ValidationError: If state is invalid.
        """
        if not isinstance(state, dict):
            raise ValidationError("State must be a dictionary")
        
        # Validate tool weights if present
        if "tool_weights" in state:
            self._validate_tool_weights(state["tool_weights"])
    
    def _validate_tool_weights(self, weights: Dict[str, float]) -> None:
        """
        Validate tool weights dictionary.
        
        Args:
            weights: Tool weights to validate.
        
        Raises:
            ValidationError: If weights are invalid.
        """
        if not isinstance(weights, dict):
            raise ValidationError("tool_weights must be a dictionary")
        
        if len(weights) > MAX_TOOL_COUNT:
            raise ValidationError(f"Too many tools: {len(weights)} > {MAX_TOOL_COUNT}")
        
        for tool_name, weight in weights.items():
            if not isinstance(tool_name, str):
                raise ValidationError(f"Tool name must be string: {tool_name}")
            
            if not tool_name or len(tool_name) > MAX_TOOL_NAME_LENGTH:
                raise ValidationError(
                    f"Invalid tool name length: {len(tool_name) if tool_name else 0}"
                )
            
            if not isinstance(weight, (int, float)):
                raise ValidationError(f"Weight must be numeric: {weight}")
            
            if weight < MIN_TOOL_WEIGHT_VALUE or weight > MAX_TOOL_WEIGHT_VALUE:
                raise ValidationError(
                    f"Weight out of range [{MIN_TOOL_WEIGHT_VALUE}, "
                    f"{MAX_TOOL_WEIGHT_VALUE}]: {weight}"
                )
    
    def _validate_and_migrate(
        self,
        state: LearningState,
        default_state: LearningState
    ) -> LearningState:
        """
        Validate loaded state and migrate schema if needed.
        
        Args:
            state: Loaded state to validate.
            default_state: Default state for filling missing keys.
        
        Returns:
            Validated and migrated state.
        """
        # Ensure all required keys exist
        for key in default_state:
            if key not in state:
                state[key] = copy.deepcopy(default_state[key])
        
        # Ensure metadata exists
        if "metadata" not in state:
            state["metadata"] = {}
        
        # Update load count
        state["metadata"]["load_count"] = state["metadata"].get("load_count", 0) + 1
        
        # Schema migration would go here for future versions
        current_schema = state.get("metadata", {}).get("schema_version", "1.0.0")
        if current_schema != SCHEMA_VERSION:
            logger.info(
                f"[LearningStatePersistence] Migrating schema from "
                f"{current_schema} to {SCHEMA_VERSION}"
            )
            state["metadata"]["schema_version"] = SCHEMA_VERSION
        
        return state
    
    # =========================================================================
    # Private Methods - Utilities
    # =========================================================================
    
    def _create_default_state(self) -> LearningState:
        """
        Create a default empty state.
        
        Returns:
            Default state dictionary with all required keys.
        """
        return {
            "tool_weights": {},
            "concept_library": {},
            "contraindications": {},
            "metadata": {
                "version": "1.0",
                "schema_version": SCHEMA_VERSION,
                "created_at": time.time(),
                "updated_at": time.time(),
                "load_count": 0,
                "save_count": 0,
            }
        }
    
    def _update_metadata_for_save(self, state: LearningState) -> LearningState:
        """
        Update metadata fields before saving.
        
        Args:
            state: State to update.
        
        Returns:
            State with updated metadata.
        """
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["version"] = "1.0"
        state["metadata"]["schema_version"] = SCHEMA_VERSION
        state["metadata"]["updated_at"] = time.time()
        state["metadata"]["save_count"] = state["metadata"].get("save_count", 0) + 1
        
        # Add hostname for debugging multi-instance deployments
        try:
            import socket
            state["metadata"]["hostname"] = socket.gethostname()
        except Exception:
            pass
        
        return state
    
    def _update_load_stats(self) -> None:
        """Update statistics after a successful load."""
        self._stats["total_loads"] += 1
        self._stats["last_load_time"] = time.time()
    
    def _compute_checksum(self, state: LearningState) -> str:
        """
        Compute SHA-256 checksum of state for integrity verification.
        
        Args:
            state: State to checksum.
        
        Returns:
            Hex-encoded SHA-256 checksum.
        """
        # Create a stable string representation
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()[:16]
    
    def __repr__(self) -> str:
        """Return string representation of the persistence instance."""
        return (
            f"LearningStatePersistence("
            f"path={self.state_file}, "
            f"available={self._storage_available})"
        )
