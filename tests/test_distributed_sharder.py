"""
Comprehensive test suite for distributed_sharder.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from distributed_sharder import (
    MAX_SHARD_SIZE_MB,
    CompressionType,
    DistributedSharder,
    PruningStrategy,
    ShardMetadata,
    create_sharder,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sharder(temp_dir):
    """Create sharder instance."""
    return create_sharder(dry_run=False, checkpoint_dir=temp_dir)


@pytest.fixture
def test_tensor():
    """Create test tensor."""
    return np.random.randn(100, 50).astype(np.float32)


class TestShardMetadata:
    """Test ShardMetadata dataclass."""

    def test_initialization(self):
        """Test metadata initialization."""
        meta = ShardMetadata(
            axis=0,
            original_shape=(100, 50),
            shard_shapes=[(25, 50), (25, 50), (25, 50), (25, 50)],
            shard_slices=[(0, 25), (25, 50), (50, 75), (75, 100)],
            dtype="float32",
            compressed=False,
            compressor="none",
            num_nodes=4,
        )

        assert meta.axis == 0
        assert meta.num_nodes == 4
        assert not meta.compressed

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = ShardMetadata(
            axis=0,
            original_shape=(100, 50),
            shard_shapes=[(50, 50), (50, 50)],
            shard_slices=[(0, 50), (50, 100)],
            dtype="float32",
            compressed=True,
            compressor="gzip",
            num_nodes=2,
        )

        d = meta.to_dict()

        assert d["axis"] == 0
        assert d["compressed"] is True
        assert d["compressor"] == "gzip"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "axis": 0,
            "original_shape": (100, 50),
            "shard_shapes": [(50, 50), (50, 50)],
            "shard_slices": [(0, 50), (50, 100)],
            "dtype": "float32",
            "compressed": False,
            "compressor": "none",
            "num_nodes": 2,
        }

        meta = ShardMetadata.from_dict(d)

        assert meta.axis == 0
        assert meta.num_nodes == 2


class TestDistributedSharder:
    """Test DistributedSharder class."""

    def test_initialization(self, temp_dir):
        """Test sharder initialization."""
        sharder = create_sharder(dry_run=False, checkpoint_dir=temp_dir)

        assert sharder.backend == "local"
        assert not sharder.dry_run
        assert sharder.checkpoint_dir.exists()

    def test_initialization_dry_run(self):
        """Test dry run initialization."""
        sharder = create_sharder(dry_run=True)

        assert sharder.dry_run

    def test_shard_tensor_basic(self, sharder, test_tensor):
        """Test basic tensor sharding."""
        shards, meta = sharder.shard_tensor(test_tensor, num_nodes=4, axis=0)

        assert len(shards) == 4
        assert meta.num_nodes == 4
        assert meta.axis == 0
        assert meta.original_shape == test_tensor.shape

    def test_shard_tensor_invalid_type(self, sharder):
        """Test sharding with invalid tensor type."""
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            sharder.shard_tensor([1, 2, 3], num_nodes=2)

    def test_shard_tensor_invalid_axis(self, sharder, test_tensor):
        """Test sharding with invalid axis."""
        with pytest.raises(ValueError, match="out of bounds"):
            sharder.shard_tensor(test_tensor, num_nodes=2, axis=10)

    def test_shard_tensor_too_many_nodes(self, sharder):
        """Test sharding with more nodes than axis length."""
        small_tensor = np.array([1, 2, 3])

        shards, meta = sharder.shard_tensor(small_tensor, num_nodes=10, axis=0)

        # Should reduce to actual size
        assert len(shards) == 3

    def test_shard_tensor_compression_gzip(self, sharder, test_tensor):
        """Test sharding with gzip compression."""
        shards, meta = sharder.shard_tensor(
            test_tensor, num_nodes=4, compress=True, compression_type="gzip"
        )

        assert meta.compressed
        assert meta.compressor == "gzip"
        assert all(isinstance(s, bytes) for s in shards)

    def test_shard_tensor_negative_axis(self, sharder, test_tensor):
        """Test sharding with negative axis."""
        shards, meta = sharder.shard_tensor(test_tensor, num_nodes=2, axis=-1)

        # Should normalize to positive
        assert meta.axis == 1

    def test_unshard_uncompressed(self, sharder, test_tensor):
        """Test unsharding uncompressed shards."""
        shards, meta = sharder.shard_tensor(test_tensor, num_nodes=4, axis=0)

        reconstructed = sharder.unshard(shards, meta)

        assert reconstructed.shape == test_tensor.shape
        np.testing.assert_array_almost_equal(reconstructed, test_tensor)

    def test_unshard_compressed(self, sharder, test_tensor):
        """Test unsharding compressed shards."""
        shards, meta = sharder.shard_tensor(
            test_tensor, num_nodes=4, compress=True, compression_type="gzip"
        )

        reconstructed = sharder.unshard(shards, meta)

        assert reconstructed.shape == test_tensor.shape
        np.testing.assert_array_almost_equal(reconstructed, test_tensor, decimal=5)

    def test_unshard_empty_shards(self, sharder):
        """Test unsharding with empty shards list."""
        with pytest.raises(ValueError, match="shards list is empty"):
            sharder.unshard([])

    def test_unshard_type_mismatch(self, sharder, test_tensor):
        """Test unsharding with mixed shard types."""
        shards = [test_tensor[:25], b"bytes_shard"]

        with pytest.raises(ValueError, match="type mismatch"):
            sharder.unshard(shards)

    def test_prune_tokens_magnitude(self, sharder):
        """Test magnitude-based pruning."""
        tensor = np.random.randn(50, 50).astype(np.float32)

        pruned, mask, threshold = sharder.prune_tokens(
            tensor, strategy="magnitude", target_sparsity=0.5
        )

        assert pruned.shape == tensor.shape
        assert mask.shape == tensor.shape
        assert mask.dtype == bool

        # Check sparsity is close to target
        actual_sparsity = 1.0 - (np.count_nonzero(pruned) / pruned.size)
        assert 0.4 <= actual_sparsity <= 0.6

    def test_prune_tokens_random(self, sharder):
        """Test random pruning."""
        tensor = np.random.randn(50, 50).astype(np.float32)

        pruned, mask, threshold = sharder.prune_tokens(
            tensor, strategy="random", target_sparsity=0.7
        )

        actual_sparsity = 1.0 - (np.count_nonzero(pruned) / pruned.size)
        assert 0.6 <= actual_sparsity <= 0.8

    def test_prune_tokens_structured(self, sharder):
        """Test structured pruning."""
        tensor = np.random.randn(50, 50).astype(np.float32)

        pruned, mask, threshold = sharder.prune_tokens(
            tensor, strategy="structured", target_sparsity=0.5
        )

        assert pruned.shape == tensor.shape

    def test_prune_tokens_invalid_type(self, sharder):
        """Test pruning with invalid tensor type."""
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            sharder.prune_tokens([1, 2, 3], strategy="magnitude")

    def test_prune_tokens_invalid_strategy(self, sharder):
        """Test pruning with invalid strategy."""
        tensor = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="Invalid strategy"):
            sharder.prune_tokens(tensor, strategy="invalid")

    def test_prune_tokens_invalid_sparsity(self, sharder):
        """Test pruning with invalid target sparsity."""
        tensor = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="must be in"):
            sharder.prune_tokens(tensor, strategy="magnitude", target_sparsity=1.5)

    def test_dynamic_batch(self, sharder):
        """Test dynamic batching."""
        tensors = [np.random.randn(10, 10).astype(np.float32) for _ in range(25)]

        batches = sharder.dynamic_batch(tensors, max_batch_size=8)

        assert len(batches) > 0
        assert all(isinstance(b, np.ndarray) for b in batches)

        # Check all original tensors are in batches
        total_tensors = sum(b.shape[0] for b in batches)
        assert total_tensors == 25

    def test_dynamic_batch_empty(self, sharder):
        """Test batching with empty list."""
        with pytest.raises(ValueError, match="empty"):
            sharder.dynamic_batch([])

    def test_dynamic_batch_invalid_size(self, sharder):
        """Test batching with invalid batch size."""
        tensors = [np.random.randn(10, 10) for _ in range(5)]

        with pytest.raises(ValueError, match="must be >= 1"):
            sharder.dynamic_batch(tensors, max_batch_size=0)

    def test_dynamic_batch_shape_validation(self, sharder):
        """Test batch shape validation."""
        tensors = [np.random.randn(10, 10), np.random.randn(10, 5)]  # Different shape

        with pytest.raises(ValueError, match="incompatible shape"):
            sharder.dynamic_batch(tensors, validate_shapes=True)

    def test_dynamic_batch_memory_limit(self, sharder):
        """Test batching with memory limit."""
        tensors = [np.random.randn(100, 100).astype(np.float32) for _ in range(10)]

        # Each tensor is ~40KB, limit to ~100KB per batch
        batches = sharder.dynamic_batch(tensors, max_memory_mb=0.1)

        assert len(batches) > 1  # Should create multiple batches

    def test_save_checkpoint(self, sharder, test_tensor):
        """Test saving checkpoint."""
        shards, meta = sharder.shard_tensor(test_tensor, num_nodes=4)

        path = sharder.save_checkpoint(shards, meta, name="test_ckpt")

        assert path.exists()

    def test_load_checkpoint(self, sharder, test_tensor):
        """Test loading checkpoint."""
        shards, meta = sharder.shard_tensor(test_tensor, num_nodes=4)
        sharder.save_checkpoint(shards, meta, name="test_ckpt")

        loaded_shards, loaded_meta = sharder.load_checkpoint("test_ckpt")

        assert len(loaded_shards) == len(shards)
        assert loaded_meta.num_nodes == meta.num_nodes

    def test_load_nonexistent_checkpoint(self, sharder):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            sharder.load_checkpoint("nonexistent")

    def test_get_stats(self, sharder, test_tensor):
        """Test getting statistics."""
        sharder.shard_tensor(test_tensor, num_nodes=4)
        sharder.shard_tensor(test_tensor, num_nodes=2)

        stats = sharder.get_stats()

        assert stats["shards_created"] == 6  # 4 + 2

    def test_reset_stats(self, sharder, test_tensor):
        """Test resetting statistics."""
        sharder.shard_tensor(test_tensor, num_nodes=4)

        sharder.reset_stats()

        stats = sharder.get_stats()
        assert stats["shards_created"] == 0

    def test_get_last_metadata(self, sharder, test_tensor):
        """Test getting last metadata."""
        _, meta = sharder.shard_tensor(test_tensor, num_nodes=4)

        last_meta = sharder.get_last_metadata()

        assert last_meta is not None
        assert last_meta.num_nodes == 4

    def test_get_metadata_history(self, sharder, test_tensor):
        """Test getting metadata history."""
        sharder.shard_tensor(test_tensor, num_nodes=4)
        sharder.shard_tensor(test_tensor, num_nodes=2)

        history = sharder.get_metadata_history()

        assert len(history) == 2


class TestDryRunMode:
    """Test dry run mode."""

    def test_dry_run_shard(self):
        """Test sharding in dry run mode."""
        sharder = create_sharder(dry_run=True)
        tensor = np.random.randn(100, 50)

        shards, meta = sharder.shard_tensor(tensor, num_nodes=4)

        assert sharder.stats["dry_run_operations"] == 1

    def test_dry_run_unshard(self):
        """Test unsharding in dry run mode."""
        sharder = create_sharder(dry_run=True)

        result = sharder.unshard([np.array([1, 2, 3])])

        assert sharder.stats["dry_run_operations"] == 1

    def test_dry_run_prune(self):
        """Test pruning in dry run mode."""
        sharder = create_sharder(dry_run=True)
        tensor = np.random.randn(10, 10)

        pruned, mask, thr = sharder.prune_tokens(tensor, strategy="magnitude")

        assert sharder.stats["dry_run_operations"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
