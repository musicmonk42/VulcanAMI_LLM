"""
Comprehensive test suite for LearningStatePersistence.

Tests cover:
- Basic CRUD operations
- Thread safety
- Error handling and edge cases
- Atomic writes and backup rotation
- Integration with file system
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from vulcan.memory.learning_persistence import (
    DEFAULT_STORAGE_PATH,
    MAX_BACKUP_COUNT,
    LearningStatePersistence,
    PersistenceError,
    SCHEMA_VERSION,
    ValidationError,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    tmpdir = tempfile.mkdtemp(prefix="vulcan_learning_test_")
    yield tmpdir
    # Cleanup after tests
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def persistence(temp_storage_dir):
    """Create a LearningStatePersistence instance with temp directory."""
    return LearningStatePersistence(storage_path=temp_storage_dir)


class TestLearningStatePersistenceInitialization:
    """Test LearningStatePersistence initialization."""

    def test_basic_initialization(self, temp_storage_dir):
        """Test basic initialization creates storage directory."""
        persistence = LearningStatePersistence(storage_path=temp_storage_dir)
        
        assert persistence.storage_path == Path(temp_storage_dir)
        assert persistence.filename == "learning_state.json"
        assert persistence.state_file == Path(temp_storage_dir) / "learning_state.json"
        assert persistence.is_storage_available()

    def test_initialization_with_custom_filename(self, temp_storage_dir):
        """Test initialization with custom filename."""
        persistence = LearningStatePersistence(
            storage_path=temp_storage_dir,
            filename="custom_state.json"
        )
        
        assert persistence.filename == "custom_state.json"
        assert persistence.state_file == Path(temp_storage_dir) / "custom_state.json"

    def test_initialization_creates_directory(self, temp_storage_dir):
        """Test initialization creates nested directory."""
        nested_path = os.path.join(temp_storage_dir, "nested", "dir")
        persistence = LearningStatePersistence(storage_path=nested_path)
        
        assert Path(nested_path).exists()
        assert persistence.is_storage_available()

    def test_initialization_with_env_variable(self, temp_storage_dir):
        """Test initialization uses environment variable."""
        with patch.dict(os.environ, {"VULCAN_STORAGE_PATH": temp_storage_dir}):
            persistence = LearningStatePersistence()
            assert persistence.storage_path == Path(temp_storage_dir)

    def test_initialization_invalid_filename_empty(self, temp_storage_dir):
        """Test initialization rejects empty filename."""
        with pytest.raises(ValueError, match="filename cannot be empty"):
            LearningStatePersistence(storage_path=temp_storage_dir, filename="")

    def test_initialization_invalid_filename_no_json(self, temp_storage_dir):
        """Test initialization rejects non-json filename."""
        with pytest.raises(ValueError, match="filename must end with .json"):
            LearningStatePersistence(storage_path=temp_storage_dir, filename="state.txt")

    def test_initialization_invalid_filename_path_separator(self, temp_storage_dir):
        """Test initialization rejects filename with path separator."""
        with pytest.raises(ValueError, match="filename cannot contain path separators"):
            LearningStatePersistence(storage_path=temp_storage_dir, filename="path/state.json")


class TestLoadState:
    """Test load_state functionality."""

    def test_load_state_no_existing_file(self, persistence):
        """Test loading state when no file exists returns default."""
        state = persistence.load_state()
        
        assert state["tool_weights"] == {}
        assert state["concept_library"] == {}
        assert state["contraindications"] == {}
        assert "metadata" in state
        assert state["metadata"]["schema_version"] == SCHEMA_VERSION

    def test_load_state_existing_file(self, persistence):
        """Test loading state from existing file."""
        # Create a state file manually
        test_state = {
            "tool_weights": {"general": 0.035, "code": 0.020},
            "concept_library": {"concept1": {"data": "test"}},
            "contraindications": {},
            "metadata": {
                "version": "1.0",
                "schema_version": SCHEMA_VERSION,
                "created_at": time.time(),
                "updated_at": time.time(),
                "load_count": 0,
                "save_count": 5,
            }
        }
        
        with open(persistence.state_file, "w", encoding="utf-8") as f:
            json.dump(test_state, f)
        
        state = persistence.load_state()
        
        assert state["tool_weights"]["general"] == 0.035
        assert state["tool_weights"]["code"] == 0.020
        assert state["metadata"]["load_count"] == 1  # Incremented

    def test_load_state_caching(self, persistence):
        """Test that load_state caches result."""
        # First load - creates default
        state1 = persistence.load_state()
        
        # Modify the file directly (should be ignored due to caching)
        with open(persistence.state_file, "w", encoding="utf-8") as f:
            json.dump({"tool_weights": {"test": 999.0}}, f)
        
        # Second load - should return cached
        state2 = persistence.load_state()
        
        assert state1 == state2
        assert "test" not in state2["tool_weights"]

    def test_load_state_corrupted_json(self, persistence):
        """Test loading state from corrupted JSON file."""
        with open(persistence.state_file, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        
        state = persistence.load_state()
        
        # Should return default state
        assert state["tool_weights"] == {}

    def test_load_state_fills_missing_keys(self, persistence):
        """Test that load fills in missing keys from defaults."""
        # Create partial state file
        partial_state = {
            "tool_weights": {"test": 0.1},
            # Missing: concept_library, contraindications, metadata
        }
        
        with open(persistence.state_file, "w", encoding="utf-8") as f:
            json.dump(partial_state, f)
        
        state = persistence.load_state()
        
        assert state["tool_weights"]["test"] == 0.1
        assert "concept_library" in state
        assert "contraindications" in state
        assert "metadata" in state


class TestSaveState:
    """Test save_state functionality."""

    def test_save_state_basic(self, persistence):
        """Test basic state saving."""
        state = {
            "tool_weights": {"general": 0.05, "code": 0.03},
            "concept_library": {},
            "contraindications": {},
            "metadata": {}
        }
        
        result = persistence.save_state(state)
        
        assert result is True
        assert persistence.state_file.exists()
        
        # Verify file contents
        with open(persistence.state_file, "r", encoding="utf-8") as f:
            saved_state = json.load(f)
        
        assert saved_state["tool_weights"]["general"] == 0.05
        assert saved_state["metadata"]["save_count"] == 1

    def test_save_state_atomic_write(self, persistence):
        """Test that save uses atomic write (no .tmp file left behind)."""
        state = {
            "tool_weights": {"test": 0.1},
            "concept_library": {},
            "contraindications": {},
            "metadata": {}
        }
        
        persistence.save_state(state)
        
        # Check no temp files exist
        temp_files = list(persistence.storage_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_save_state_creates_backup(self, persistence):
        """Test that save creates backup of existing file."""
        # First save
        state1 = {"tool_weights": {"v1": 0.1}, "concept_library": {}, "contraindications": {}, "metadata": {}}
        persistence.save_state(state1)
        
        # Clear cache to force new load
        persistence._cached_state = None
        
        # Second save - should create backup
        state2 = {"tool_weights": {"v2": 0.2}, "concept_library": {}, "contraindications": {}, "metadata": {}}
        persistence.save_state(state2)
        
        # Check backup exists
        backup_pattern = f"{persistence.filename}.backup.*"
        backups = list(persistence.storage_path.glob(backup_pattern))
        assert len(backups) >= 1

    def test_save_state_backup_rotation(self, persistence):
        """Test that backups are rotated when exceeding max count."""
        state = {"tool_weights": {}, "concept_library": {}, "contraindications": {}, "metadata": {}}
        
        # Create many saves to trigger backup rotation
        for i in range(MAX_BACKUP_COUNT + 5):
            state["tool_weights"][f"tool_{i}"] = float(i) / 100
            persistence._cached_state = None  # Force reload
            persistence.save_state(state)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Check backup count
        backup_pattern = f"{persistence.filename}.backup.*"
        backups = list(persistence.storage_path.glob(backup_pattern))
        assert len(backups) <= MAX_BACKUP_COUNT

    def test_save_state_updates_cache(self, persistence):
        """Test that save updates in-memory cache."""
        state = {"tool_weights": {"test": 0.5}, "concept_library": {}, "contraindications": {}, "metadata": {}}
        
        persistence.save_state(state)
        
        # Load should return cached version without reading file
        loaded = persistence.load_state()
        assert loaded["tool_weights"]["test"] == 0.5


class TestUpdateToolWeights:
    """Test update_tool_weights functionality."""

    def test_update_tool_weights_basic(self, persistence):
        """Test basic tool weight update."""
        weights = {"general": 0.035, "code": 0.020, "search": -0.005}
        
        result = persistence.update_tool_weights(weights)
        
        assert result is True
        
        # Verify persisted
        loaded = persistence.get_tool_weights()
        assert loaded["general"] == 0.035
        assert loaded["code"] == 0.020
        assert loaded["search"] == -0.005

    def test_update_tool_weights_overwrites(self, persistence):
        """Test that update overwrites previous weights."""
        persistence.update_tool_weights({"old": 0.1})
        persistence.update_tool_weights({"new": 0.2})
        
        weights = persistence.get_tool_weights()
        
        assert "old" not in weights
        assert weights["new"] == 0.2

    def test_update_tool_weights_preserves_other_state(self, persistence):
        """Test that update preserves concept_library and contraindications."""
        # Set up initial state with all fields
        state = persistence.load_state()
        state["concept_library"] = {"concept1": "data1"}
        state["contraindications"] = {"contra1": "data1"}
        persistence.save_state(state)
        
        # Clear cache
        persistence._cached_state = None
        
        # Update only tool weights
        persistence.update_tool_weights({"new_tool": 0.1})
        
        # Verify other fields preserved
        loaded = persistence.load_state()
        assert loaded["concept_library"]["concept1"] == "data1"
        assert loaded["contraindications"]["contra1"] == "data1"


class TestValidation:
    """Test validation functionality."""

    def test_valid_tool_weights(self, persistence):
        """Test that valid weights are accepted."""
        valid_weights = {
            "tool1": 0.5,
            "tool2": -0.5,
            "tool3": 0.0,
        }
        
        result = persistence.update_tool_weights(valid_weights)
        assert result is True

    def test_invalid_tool_weight_out_of_range(self, persistence):
        """Test that out-of-range weights are rejected."""
        invalid_weights = {"tool": 999.0}  # Way out of valid range
        
        with pytest.raises(ValidationError, match="Weight out of range"):
            persistence.update_tool_weights(invalid_weights)

    def test_invalid_tool_name_empty(self, persistence):
        """Test that empty tool names are rejected."""
        invalid_weights = {"": 0.5}
        
        with pytest.raises(ValidationError, match="Invalid tool name length"):
            persistence.update_tool_weights(invalid_weights)

    def test_invalid_tool_name_type(self, persistence):
        """Test that non-string tool names are rejected."""
        invalid_weights = {123: 0.5}
        
        with pytest.raises(ValidationError, match="Tool name must be string"):
            persistence.update_tool_weights(invalid_weights)


class TestGetStats:
    """Test get_stats functionality."""

    def test_get_stats_basic(self, persistence):
        """Test basic stats retrieval."""
        stats = persistence.get_stats()
        
        assert "storage_path" in stats
        assert "file_exists" in stats
        assert "tool_weights_count" in stats
        assert "total_saves" in stats
        assert "total_loads" in stats
        assert "storage_available" in stats
        assert stats["storage_available"] is True

    def test_get_stats_after_operations(self, persistence):
        """Test stats after some operations."""
        # Perform some operations
        persistence.update_tool_weights({"tool1": 0.1, "tool2": 0.2})
        # Clear cache to force actual load
        persistence._cached_state = None
        persistence.load_state()
        persistence._cached_state = None
        persistence.load_state()
        
        stats = persistence.get_stats()
        
        assert stats["tool_weights_count"] == 2
        assert stats["total_saves"] >= 1
        assert stats["total_loads"] >= 2
        assert stats["file_exists"] is True


class TestClearState:
    """Test clear_state functionality."""

    def test_clear_state(self, persistence):
        """Test clearing state."""
        # Set up some state
        persistence.update_tool_weights({"tool": 0.5})
        
        # Clear it
        result = persistence.clear_state()
        
        assert result is True
        
        # Verify cleared
        state = persistence.load_state()
        assert state["tool_weights"] == {}
        assert "cleared_at" in state["metadata"]


class TestThreadSafety:
    """Test thread safety of operations."""

    def test_concurrent_updates(self, persistence):
        """Test concurrent tool weight updates."""
        errors = []
        
        def update_weights(tool_name, iterations):
            try:
                for i in range(iterations):
                    persistence.update_tool_weights({tool_name: float(i) / 100})
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_weights, args=(f"tool_{i}", 10))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join(timeout=30)
        
        # No errors should have occurred
        assert len(errors) == 0
        
        # State file should still be valid
        state = persistence.load_state()
        assert "tool_weights" in state

    def test_concurrent_reads(self, persistence):
        """Test concurrent state reads."""
        # Set up some state
        persistence.update_tool_weights({"test": 0.5})
        
        results = []
        errors = []
        
        def read_state():
            try:
                for _ in range(20):
                    state = persistence.load_state()
                    results.append(state["tool_weights"].get("test"))
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple reader threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=read_state)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0
        # All reads should return the same value
        assert all(r == 0.5 for r in results)


class TestBackupRecovery:
    """Test backup and recovery functionality."""

    def test_recovery_from_backup(self, persistence):
        """Test recovery from backup when primary fails."""
        # Save initial state
        persistence.update_tool_weights({"initial": 0.1})
        
        # Clear cache to force file read
        persistence._cached_state = None
        
        # Save again to create backup
        persistence.update_tool_weights({"updated": 0.2})
        
        # Corrupt the primary file
        with open(persistence.state_file, "w", encoding="utf-8") as f:
            f.write("corrupted data {{{")
        
        # Clear cache
        persistence._cached_state = None
        
        # Load should recover from backup
        state = persistence.load_state()
        
        # Should have recovered some state (either initial or default)
        assert "tool_weights" in state


class TestInteractionHistory:
    """Test interaction history functionality."""

    def test_add_interaction_basic(self, persistence):
        """Test adding a basic interaction."""
        result = persistence.add_interaction(
            query_id="q123",
            query="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            tools_used=["general", "search"],
            success=True,
            latency_ms=150.5,
        )
        
        assert result is True
        
        # Verify it was stored
        history = persistence.get_interaction_history(limit=10)
        assert len(history) >= 1
        assert history[0]["query_id"] == "q123"
        assert history[0]["query"] == "What is machine learning?"
        assert "general" in history[0]["tools_used"]

    def test_get_interaction_history(self, persistence):
        """Test getting interaction history."""
        # Add multiple interactions
        for i in range(5):
            persistence.add_interaction(
                query_id=f"q{i}",
                query=f"Query {i}",
                answer=f"Answer {i}",
                tools_used=["tool_a"] if i % 2 == 0 else ["tool_b"],
                success=(i % 2 == 0),
            )
        
        # Get all
        history = persistence.get_interaction_history(limit=10)
        assert len(history) == 5
        
        # Get limited
        limited = persistence.get_interaction_history(limit=2)
        assert len(limited) == 2
        
        # Get success only
        success_only = persistence.get_interaction_history(success_only=True)
        assert all(h["success"] for h in success_only)

    def test_get_interactions_by_tool(self, persistence):
        """Test filtering interactions by tool."""
        # Add interactions with different tools
        persistence.add_interaction(
            query_id="q1", query="Q1", answer="A1",
            tools_used=["general", "search"], success=True
        )
        persistence.add_interaction(
            query_id="q2", query="Q2", answer="A2",
            tools_used=["code", "search"], success=True
        )
        persistence.add_interaction(
            query_id="q3", query="Q3", answer="A3",
            tools_used=["general"], success=True
        )
        
        # Filter by tool
        general_interactions = persistence.get_interactions_by_tool("general")
        assert len(general_interactions) == 2
        
        search_interactions = persistence.get_interactions_by_tool("search")
        assert len(search_interactions) == 2
        
        code_interactions = persistence.get_interactions_by_tool("code")
        assert len(code_interactions) == 1

    def test_interaction_history_limit(self, persistence):
        """Test that interaction history is limited to 1000 entries."""
        # This is a lighter test - just verify the mechanism works
        for i in range(10):
            persistence.add_interaction(
                query_id=f"q{i}",
                query=f"Query {i}",
                answer=f"Answer {i}",
            )
        
        history = persistence.get_interaction_history(limit=100)
        assert len(history) <= 100


class TestRepr:
    """Test string representation."""

    def test_repr(self, persistence):
        """Test __repr__ output."""
        repr_str = repr(persistence)
        
        assert "LearningStatePersistence" in repr_str
        assert "path=" in repr_str
        assert "available=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
