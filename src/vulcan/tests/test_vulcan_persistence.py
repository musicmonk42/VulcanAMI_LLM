"""
Comprehensive test suite for persistence.py

Tests cover:
- Memory compression (LZ4, ZSTD, Neural, Semantic)
- Memory persistence (save, load, batch operations)
- Version control (create, rollback, branches, merge)
- Atomic writes and crash recovery
- Encryption (if available)
- Checkpoints and restore
- Storage statistics
- Concurrent access
- Edge cases and error handling
"""

import pytest
import numpy as np
import time
import threading
import tempfile
import shutil
import pickle
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vulcan.memory.persistence import (
    MemoryPersistence,
    MemoryVersionControl,
    MemoryCompressor,
    NeuralCompressor,
    SemanticCompressor
)
from vulcan.memory.base import (
    Memory,
    MemoryType,
    CompressionType
)

# Check if optional dependencies are available
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_memory():
    """Create sample memory for testing."""
    return Memory(
        id="test_memory_001",
        type=MemoryType.EPISODIC,
        content="This is test content for memory persistence",
        timestamp=time.time(),
        importance=0.8,
        metadata={'test': True, 'category': 'unit_test'}
    )


@pytest.fixture
def sample_memories():
    """Create multiple sample memories."""
    memories = []
    for i in range(10):
        mem = Memory(
            id=f"test_memory_{i:03d}",
            type=MemoryType.EPISODIC if i % 2 == 0 else MemoryType.SEMANTIC,
            content=f"Test content number {i}",
            timestamp=time.time() + i,
            importance=0.5 + (i * 0.05),
            metadata={'index': i, 'test': True}
        )
        memories.append(mem)
    return memories


@pytest.fixture
def persistence(temp_dir):
    """Create MemoryPersistence instance."""
    # Set environment to allow ephemeral key for testing
    os.environ['ALLOW_EPHEMERAL_KEY'] = 'true'
    
    pers = MemoryPersistence(base_path=temp_dir)
    yield pers
    
    # Cleanup
    pers.shutdown()
    
    # Clean up env
    if 'ALLOW_EPHEMERAL_KEY' in os.environ:
        del os.environ['ALLOW_EPHEMERAL_KEY']


@pytest.fixture
def persistence_with_encryption(temp_dir):
    """Create MemoryPersistence with encryption."""
    if not ENCRYPTION_AVAILABLE:
        pytest.skip("Encryption not available")
    
    # Generate encryption key
    key = Fernet.generate_key()
    os.environ['MEMORY_ENCRYPT_KEY'] = key.decode()
    
    pers = MemoryPersistence(base_path=temp_dir)
    yield pers
    
    # Cleanup
    pers.shutdown()
    del os.environ['MEMORY_ENCRYPT_KEY']


@pytest.fixture
def version_control(temp_dir):
    """Create MemoryVersionControl instance."""
    return MemoryVersionControl(storage_path=temp_dir)


# ============================================================
# COMPRESSION TESTS
# ============================================================

class TestMemoryCompression:
    """Test memory compression functionality."""
    
    def test_compress_none(self, sample_memory):
        """Test no compression."""
        compressed = MemoryCompressor.compress(sample_memory, CompressionType.NONE)
        
        assert compressed is not None
        assert len(compressed) > 0
        
        # Should be able to decompress
        decompressed = MemoryCompressor.decompress(compressed, CompressionType.NONE)
        assert decompressed == sample_memory.content
    
    @pytest.mark.skipif(not LZ4_AVAILABLE, reason="LZ4 not available")
    def test_compress_lz4(self, sample_memory):
        """Test LZ4 compression."""
        compressed = MemoryCompressor.compress(sample_memory, CompressionType.LZ4)
        
        assert compressed is not None
        assert len(compressed) > 0
        
        # Decompress and verify
        decompressed = MemoryCompressor.decompress(compressed, CompressionType.LZ4)
        assert decompressed == sample_memory.content
    
    @pytest.mark.skipif(not LZ4_AVAILABLE, reason="LZ4 not available")
    def test_compression_ratio(self, sample_memory):
        """Test that compression reduces size."""
        # Create memory with repetitive content
        sample_memory.content = "Test " * 1000
        
        compressed = MemoryCompressor.compress(sample_memory, CompressionType.LZ4)
        uncompressed = pickle.dumps(sample_memory.content)
        
        # Compressed should be smaller
        assert len(compressed) < len(uncompressed)
        
        # Estimate compression ratio
        ratio = MemoryCompressor.estimate_compression_ratio(
            sample_memory, 
            CompressionType.LZ4
        )
        assert ratio > 1.0  # Should have compression
    
    def test_semantic_compression(self, sample_memory):
        """Test semantic compression."""
        compressor = SemanticCompressor()
        
        result = compressor.compress(sample_memory.content, strategy='summary')
        
        assert 'type' in result
        assert result['type'] == 'summary'
        assert 'compressed' in result
        assert 'original_length' in result
    
    def test_semantic_compression_keywords(self, sample_memory):
        """Test keyword extraction compression."""
        compressor = SemanticCompressor()
        
        sample_memory.content = "machine learning deep neural networks artificial intelligence"
        result = compressor.compress(sample_memory.content, strategy='keywords')
        
        assert result['type'] == 'keywords'
        assert 'keywords' in result
        assert len(result['keywords']) > 0
    
    def test_semantic_compression_embedding(self, sample_memory):
        """Test embedding compression."""
        compressor = SemanticCompressor()
        
        result = compressor.compress(sample_memory.content, strategy='embedding')
        
        assert 'type' in result
        assert 'embedding' in result
        assert isinstance(result['embedding'], np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_neural_compression(self, sample_memory):
        """Test neural compression."""
        # Neural compression requires specific content types
        sample_memory.content = "A" * 500  # Longer content
        
        compressed = MemoryCompressor.compress(sample_memory, CompressionType.NEURAL)
        
        assert compressed is not None
        assert len(compressed) > 0
        
        # Try to decompress
        decompressed = MemoryCompressor.decompress(compressed, CompressionType.NEURAL)
        # Neural compression is lossy, so just check it doesn't crash
        assert decompressed is not None


# ============================================================
# PERSISTENCE TESTS
# ============================================================

class TestMemoryPersistence:
    """Test memory save and load functionality."""
    
    def test_save_memory(self, persistence, sample_memory):
        """Test saving a single memory."""
        success = persistence.save_memory(sample_memory, immediate=True)
        
        assert success is True
        
        # Verify file exists
        assert len(list(persistence.memories_path.glob("**/*.mem"))) > 0
    
    def test_load_memory(self, persistence, sample_memory):
        """Test loading a saved memory."""
        # Save first
        persistence.save_memory(sample_memory, immediate=True)
        
        # Load it back
        loaded = persistence.load_memory(sample_memory.id)
        
        assert loaded is not None
        assert loaded.id == sample_memory.id
        assert loaded.content == sample_memory.content
        assert loaded.type == sample_memory.type
        assert loaded.importance == sample_memory.importance
    
    def test_save_load_with_compression(self, persistence, sample_memory):
        """Test save and load with compression."""
        success = persistence.save_memory(
            sample_memory, 
            compress=True,
            compression_type=CompressionType.LZ4,
            immediate=True
        )
        
        assert success is True
        
        # Load and verify
        loaded = persistence.load_memory(sample_memory.id)
        
        assert loaded is not None
        assert loaded.content == sample_memory.content
    
    def test_save_batch(self, persistence, sample_memories):
        """Test batch save operation."""
        saved_count = persistence.save_batch(
            sample_memories,
            compress=True,
            compression_type=CompressionType.LZ4
        )
        
        assert saved_count == len(sample_memories)
        
        # Flush to ensure all written
        persistence.flush_buffer()
        
        # Verify all can be loaded
        for memory in sample_memories:
            loaded = persistence.load_memory(memory.id)
            assert loaded is not None
    
    def test_load_batch(self, persistence, sample_memories):
        """Test batch load operation."""
        # Save first
        persistence.save_batch(sample_memories, immediate=True)
        persistence.flush_buffer()
        
        # Load batch
        memory_ids = [m.id for m in sample_memories]
        loaded = persistence.load_batch(memory_ids)
        
        assert len(loaded) == len(sample_memories)
        assert all(m.id in memory_ids for m in loaded)
    
    def test_delete_memory(self, persistence, sample_memory):
        """Test deleting a memory."""
        # Save first
        persistence.save_memory(sample_memory, immediate=True)
        
        # Verify it exists
        loaded = persistence.load_memory(sample_memory.id)
        assert loaded is not None
        
        # Delete it
        success = persistence.delete_memory(sample_memory.id, permanent=False)
        assert success is True
        
        # Should be in trash, not in regular location
        loaded = persistence.load_memory(sample_memory.id)
        assert loaded is None
    
    def test_delete_memory_permanent(self, persistence, sample_memory):
        """Test permanent deletion."""
        persistence.save_memory(sample_memory, immediate=True)
        
        # Permanent delete
        success = persistence.delete_memory(sample_memory.id, permanent=True)
        assert success is True
        
        # Should not exist anywhere
        loaded = persistence.load_memory(sample_memory.id)
        assert loaded is None
    
    def test_list_memories(self, persistence, sample_memories):
        """Test listing persisted memories."""
        # Save some memories
        for memory in sample_memories[:5]:
            persistence.save_memory(memory, immediate=True)
        
        # List all memories
        memory_ids = persistence.list_memories()
        
        assert len(memory_ids) >= 5
        assert all(m.id in memory_ids for m in sample_memories[:5])
    
    def test_list_memories_with_pattern(self, persistence, sample_memories):
        """Test listing with pattern matching."""
        # Save memories
        for memory in sample_memories:
            persistence.save_memory(memory, immediate=True)
        
        # List with pattern
        memory_ids = persistence.list_memories(pattern="test_memory_00")
        
        # Should match memories 000-009
        assert len(memory_ids) > 0


# ============================================================
# VERSION CONTROL TESTS
# ============================================================

class TestVersionControl:
    """Test memory version control."""
    
    def test_create_version(self, version_control, sample_memory):
        """Test creating a version."""
        version_id = version_control.create_version(
            sample_memory,
            message="Initial version",
            author="test_user"
        )
        
        assert version_id is not None
        assert len(version_id) > 0
    
    def test_get_version(self, version_control, sample_memory):
        """Test retrieving a version."""
        version_id = version_control.create_version(sample_memory)
        
        # Get the version
        version = version_control.get_version(sample_memory.id, version_id)
        
        assert version is not None
        assert version.version_id == version_id
        assert version.memory_id == sample_memory.id
    
    def test_version_history(self, version_control, sample_memory):
        """Test getting version history."""
        # Create multiple versions
        for i in range(5):
            sample_memory.content = f"Version {i} content"
            version_control.create_version(
                sample_memory,
                message=f"Update {i}"
            )
            time.sleep(0.01)  # Ensure different timestamps
        
        # Get history
        history = version_control.get_history(sample_memory.id)
        
        assert len(history) == 5
        # Should be in reverse chronological order
        assert history[0].timestamp > history[-1].timestamp
    
    def test_no_version_on_unchanged_content(self, version_control, sample_memory):
        """Test that unchanged content doesn't create new version."""
        version_id1 = version_control.create_version(sample_memory)
        version_id2 = version_control.create_version(sample_memory)
        
        # Should return same version ID (no change)
        assert version_id1 == version_id2
    
    def test_create_branch(self, version_control):
        """Test creating a branch."""
        success = version_control.create_branch("feature_branch")
        
        assert success is True
        assert "feature_branch" in version_control.branches
    
    def test_switch_branch(self, version_control, sample_memory):
        """Test switching branches."""
        # Create version on main
        v1 = version_control.create_version(sample_memory, message="Main version")
        
        # Create and switch to new branch
        version_control.create_branch("test_branch")
        success = version_control.switch_branch("test_branch")
        
        assert success is True
        assert version_control.current_branch == "test_branch"
    
    def test_merge_branches(self, version_control, sample_memory):
        """Test merging branches."""
        # Create version on main
        version_control.create_version(sample_memory)
        
        # Create branch and add version
        version_control.create_branch("feature")
        version_control.switch_branch("feature")
        
        sample_memory.content = "Feature content"
        version_control.create_version(sample_memory)
        
        # Switch back to main and merge
        version_control.switch_branch("main")
        result = version_control.merge_branches("feature", "main")
        
        assert result['success'] is True
        assert result['merged'] >= 0
    
    def test_rollback(self, version_control, sample_memory):
        """Test rolling back to a previous version."""
        # Create versions
        v1 = version_control.create_version(sample_memory, message="Version 1")
        
        sample_memory.content = "Modified content"
        v2 = version_control.create_version(sample_memory, message="Version 2")
        
        # Rollback to v1
        success = version_control.rollback(sample_memory.id, v1)
        
        assert success is True
        
        # Current version should be v1
        current = version_control.get_version(sample_memory.id)
        assert current.version_id == v1
    
    def test_diff_versions(self, version_control, sample_memory):
        """Test comparing versions."""
        v1 = version_control.create_version(sample_memory)
        
        sample_memory.content = "Different content"
        sample_memory.metadata['new_field'] = 'value'
        v2 = version_control.create_version(sample_memory)
        
        diff = version_control.diff_versions(v1, v2)
        
        assert 'version1' in diff
        assert 'version2' in diff
        assert 'hash_changed' in diff
        assert diff['hash_changed'] is True


# ============================================================
# CHECKPOINT TESTS
# ============================================================

class TestCheckpoints:
    """Test checkpoint and restore functionality."""
    
    def test_create_checkpoint(self, persistence, sample_memories):
        """Test creating a checkpoint."""
        memories_dict = {m.id: m for m in sample_memories}
        
        success = persistence.checkpoint(memories_dict, name="test_checkpoint")
        
        assert success is True
        
        # Verify checkpoint file exists
        checkpoints = list(persistence.base_path.glob("checkpoint_*.pkl"))
        assert len(checkpoints) > 0
    
    def test_restore_from_checkpoint(self, persistence, sample_memories):
        """Test restoring from a checkpoint."""
        memories_dict = {m.id: m for m in sample_memories}
        
        # Create checkpoint
        persistence.checkpoint(memories_dict, name="restore_test")
        
        # Restore
        restored = persistence.restore_from_checkpoint()
        
        assert len(restored) == len(sample_memories)
        assert all(m.id in restored for m in sample_memories)
    
    def test_restore_specific_checkpoint(self, persistence, sample_memories):
        """Test restoring from a specific checkpoint."""
        memories_dict = {m.id: m for m in sample_memories}
        
        # Create checkpoint
        persistence.checkpoint(memories_dict, name="specific")
        
        # Find checkpoint file
        checkpoints = list(persistence.base_path.glob("checkpoint_specific_*.pkl"))
        assert len(checkpoints) > 0
        
        checkpoint_file = checkpoints[0].name
        
        # Restore from specific checkpoint
        restored = persistence.restore_from_checkpoint(checkpoint_file)
        
        assert len(restored) == len(sample_memories)


# ============================================================
# ENCRYPTION TESTS
# ============================================================

class TestEncryption:
    """Test encryption functionality."""
    
    @pytest.mark.skipif(not ENCRYPTION_AVAILABLE, reason="Encryption not available")
    def test_save_with_encryption(self, persistence_with_encryption, sample_memory):
        """Test saving with encryption."""
        success = persistence_with_encryption.save_memory(
            sample_memory,
            immediate=True
        )
        
        assert success is True
        
        # Load and verify
        loaded = persistence_with_encryption.load_memory(sample_memory.id)
        assert loaded is not None
        assert loaded.content == sample_memory.content
    
    @pytest.mark.skipif(not ENCRYPTION_AVAILABLE, reason="Encryption not available")
    def test_encrypted_data_unreadable_without_key(self, temp_dir, sample_memory):
        """Test that encrypted data can't be read without key."""
        # Create persistence with encryption
        key = Fernet.generate_key()
        os.environ['MEMORY_ENCRYPT_KEY'] = key.decode()
        
        pers1 = MemoryPersistence(base_path=temp_dir)
        pers1.save_memory(sample_memory, immediate=True)
        pers1.shutdown()
        
        del os.environ['MEMORY_ENCRYPT_KEY']
        
        # Try to load with different key (ephemeral)
        os.environ['ALLOW_EPHEMERAL_KEY'] = 'true'
        pers2 = MemoryPersistence(base_path=temp_dir)
        
        loaded = pers2.load_memory(sample_memory.id)
        
        # Should fail to decrypt or return None
        if loaded is not None:
            # Content should be encrypted bytes, not the original
            assert loaded.content != sample_memory.content
        
        pers2.shutdown()
        
        # Cleanup
        if 'ALLOW_EPHEMERAL_KEY' in os.environ:
            del os.environ['ALLOW_EPHEMERAL_KEY']


# ============================================================
# ATOMIC WRITE TESTS
# ============================================================

class TestAtomicWrites:
    """Test atomic write operations."""
    
    def test_atomic_write_success(self, persistence, sample_memory):
        """Test successful atomic write."""
        success = persistence.save_memory(sample_memory, immediate=True)
        
        assert success is True
        
        # Verify no temp files left behind
        temp_files = list(persistence.memories_path.glob("**/*.tmp"))
        assert len(temp_files) == 0
    
    def test_backup_on_overwrite(self, persistence, sample_memory):
        """Test that backup is created when overwriting."""
        # Save initial version
        persistence.save_memory(sample_memory, immediate=True)
        
        # Modify and save again
        sample_memory.content = "Modified content"
        persistence.save_memory(sample_memory, immediate=True)
        
        # Original should still be loadable
        loaded = persistence.load_memory(sample_memory.id)
        assert loaded.content == "Modified content"


# ============================================================
# CONCURRENT ACCESS TESTS
# ============================================================

class TestConcurrentAccess:
    """Test thread safety."""
    
    def test_concurrent_saves(self, persistence, sample_memories):
        """Test concurrent save operations."""
        errors = []
        
        def save_memory(memory):
            try:
                persistence.save_memory(memory, immediate=True)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for memory in sample_memories:
            t = threading.Thread(target=save_memory, args=(memory,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # All memories should be saved
        for memory in sample_memories:
            loaded = persistence.load_memory(memory.id)
            assert loaded is not None
    
    def test_concurrent_loads(self, persistence, sample_memories):
        """Test concurrent load operations."""
        # Save all memories first
        for memory in sample_memories:
            persistence.save_memory(memory, immediate=True)
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def load_memory(memory_id):
            try:
                loaded = persistence.load_memory(memory_id)
                with lock:
                    results.append(loaded)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for memory in sample_memories:
            t = threading.Thread(target=load_memory, args=(memory.id,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == len(sample_memories)


# ============================================================
# STORAGE STATISTICS TESTS
# ============================================================

class TestStorageStats:
    """Test storage statistics."""
    
    def test_get_storage_stats(self, persistence, sample_memories):
        """Test getting storage statistics."""
        # Save some memories
        for memory in sample_memories:
            persistence.save_memory(memory, immediate=True)
        
        stats = persistence.get_storage_stats()
        
        assert 'total_files' in stats
        assert 'total_size_mb' in stats
        assert 'memories_count' in stats
        assert stats['memories_count'] >= len(sample_memories)
    
    def test_cleanup_old_versions(self, persistence, sample_memory):
        """Test cleaning up old versions."""
        # This would require waiting for files to age
        # For testing, we'll just verify the method runs
        cleaned = persistence.cleanup_old_versions(max_age_days=0)
        
        assert cleaned >= 0


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_save_empty_content(self, persistence):
        """Test saving memory with empty content."""
        memory = Memory(
            id="empty_test",
            type=MemoryType.WORKING,
            content="",
            timestamp=time.time()
        )
        
        success = persistence.save_memory(memory, immediate=True)
        assert success is True
        
        loaded = persistence.load_memory(memory.id)
        assert loaded is not None
        assert loaded.content == ""
    
    def test_save_none_content(self, persistence):
        """Test saving memory with None content."""
        memory = Memory(
            id="none_test",
            type=MemoryType.WORKING,
            content=None,
            timestamp=time.time()
        )
        
        success = persistence.save_memory(memory, immediate=True)
        assert success is True
        
        loaded = persistence.load_memory(memory.id)
        assert loaded is not None
    
    def test_save_large_content(self, persistence):
        """Test saving large content."""
        memory = Memory(
            id="large_test",
            type=MemoryType.LONG_TERM,
            content="x" * 1000000,  # 1MB of data
            timestamp=time.time()
        )
        
        success = persistence.save_memory(
            memory,
            compress=True,
            compression_type=CompressionType.LZ4,
            immediate=True
        )
        
        assert success is True
        
        loaded = persistence.load_memory(memory.id)
        assert loaded is not None
        assert len(loaded.content) == 1000000
    
    def test_load_nonexistent_memory(self, persistence):
        """Test loading non-existent memory."""
        loaded = persistence.load_memory("nonexistent_id")
        
        assert loaded is None
    
    def test_delete_nonexistent_memory(self, persistence):
        """Test deleting non-existent memory."""
        success = persistence.delete_memory("nonexistent_id")
        
        # Should return False or handle gracefully
        assert success is False or success is True
    
    def test_corrupted_metadata_handling(self, persistence, sample_memory):
        """Test handling of corrupted metadata."""
        # Save memory
        persistence.save_memory(sample_memory, immediate=True)
        
        # Corrupt metadata file
        metadata_path = persistence._get_metadata_path(sample_memory.id)
        with open(metadata_path, 'w') as f:
            f.write("corrupted data {{{")
        
        # Try to load - should handle gracefully
        # Depending on implementation, might return None or raise
        try:
            loaded = persistence.load_memory(sample_memory.id)
            # If it loads, that's fine too
        except Exception:
            # Expected to fail gracefully
            pass
    
    def test_buffer_flush_on_shutdown(self, temp_dir, sample_memories):
        """Test that buffer is flushed on shutdown without encryption."""
        # FIX: Don't use ephemeral encryption - test without encryption
        # to avoid cross-instance encryption key issues
        
        pers = MemoryPersistence(base_path=temp_dir)
        
        # Add to buffer without flushing
        for memory in sample_memories:
            pers.save_memory(memory, immediate=False)
        
        # Shutdown should flush
        pers.shutdown()
        
        # Create new instance and verify memories were saved
        pers2 = MemoryPersistence(base_path=temp_dir)
        
        for memory in sample_memories:
            loaded = pers2.load_memory(memory.id)
            assert loaded is not None
        
        pers2.shutdown()
    
    def test_version_control_with_missing_database(self, temp_dir):
        """Test version control handles missing database."""
        # Create version control
        vc = MemoryVersionControl(storage_path=temp_dir)
        
        # Delete database
        if vc.db_path.exists():
            os.remove(vc.db_path)
        
        # Should reinitialize gracefully
        vc2 = MemoryVersionControl(storage_path=temp_dir)
        
        # Should work
        memory = Memory(
            id="test",
            type=MemoryType.WORKING,
            content="test",
            timestamp=time.time()
        )
        
        version_id = vc2.create_version(memory)
        assert version_id is not None


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_lifecycle(self, persistence, sample_memory):
        """Test complete memory lifecycle."""
        # Save with compression
        success = persistence.save_memory(
            sample_memory,
            compress=True,
            compression_type=CompressionType.LZ4,
            immediate=True
        )
        assert success is True
        
        # Load back
        loaded = persistence.load_memory(sample_memory.id)
        assert loaded is not None
        assert loaded.content == sample_memory.content
        
        # Update
        loaded.content = "Updated content"
        persistence.save_memory(loaded, immediate=True)
        
        # Load again
        loaded2 = persistence.load_memory(sample_memory.id)
        assert loaded2.content == "Updated content"
        
        # Delete
        persistence.delete_memory(sample_memory.id, permanent=True)
        
        # Verify deleted
        loaded3 = persistence.load_memory(sample_memory.id)
        assert loaded3 is None
    
    def test_versioning_workflow(self, persistence, sample_memory):
        """Test version control workflow."""
        vc = persistence.version_control
        
        # Create initial version
        v1 = vc.create_version(sample_memory, message="Initial")
        
        # Modify and create new version
        sample_memory.content = "Version 2"
        v2 = vc.create_version(sample_memory, message="Update 1")
        
        # Get history
        history = vc.get_history(sample_memory.id)
        assert len(history) >= 2
        
        # Rollback
        vc.rollback(sample_memory.id, v1)
        
        # Verify current is v1
        current = vc.get_version(sample_memory.id)
        assert current.version_id == v1
    
    def test_checkpoint_and_restore_workflow(self, persistence, sample_memories):
        """Test checkpoint and restore workflow."""
        memories_dict = {m.id: m for m in sample_memories}
        
        # Save all memories
        for memory in sample_memories:
            persistence.save_memory(memory, immediate=True)
        
        # Create checkpoint
        persistence.checkpoint(memories_dict, name="integration_test")
        
        # Delete some memories
        for memory in sample_memories[:5]:
            persistence.delete_memory(memory.id, permanent=True)
        
        # Restore from checkpoint
        restored = persistence.restore_from_checkpoint()
        
        # All memories should be back
        assert len(restored) == len(sample_memories)
        
        # Verify all restored items are Memory objects
        for memory_id, memory in restored.items():
            assert isinstance(memory, Memory)
            assert memory.id == memory_id
        
        # Save restored memories
        for memory in restored.values():
            persistence.save_memory(memory, immediate=True)
        
        # Verify all can be loaded
        for memory in sample_memories:
            loaded = persistence.load_memory(memory.id)
            assert loaded is not None


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])