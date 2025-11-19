<<<<<<< HEAD
"""
Comprehensive tests for lsm.py module.

Tests cover:
- BloomFilter operations
- MerkleLSMDAG functionality
- MerkleLSM tree operations
- Compaction strategies
- Point and range queries
- Async operations
"""

import pytest
import asyncio
import numpy as np
import pickle
import zlib
from unittest.mock import Mock, patch
from typing import List

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from lsm import (
    BloomFilter,
    Packfile,
    MerkleNode,
    MerkleLSMDAG,
    MerkleLSM
)


class TestBloomFilter:
    """Test suite for BloomFilter class."""
    
    def test_initialization(self):
        """Test BloomFilter initialization."""
        bf = BloomFilter(size=1000, num_hashes=3)
        
        assert bf.size == 1000
        assert bf.num_hashes == 3
        assert bf.item_count == 0
        assert len(bf.bit_array) == 1000
    
    def test_add_and_contains(self):
        """Test adding items and checking membership."""
        bf = BloomFilter(size=10000)
        
        bf.add("item1")
        bf.add("item2")
        bf.add("item3")
        
        assert bf.contains("item1") is True
        assert bf.contains("item2") is True
        assert bf.contains("item3") is True
        assert bf.item_count == 3
    
    def test_false_negatives_impossible(self):
        """Test that false negatives are impossible."""
        bf = BloomFilter(size=10000)
        
        items = [f"item{i}" for i in range(100)]
        
        for item in items:
            bf.add(item)
        
        # All added items should be found
        for item in items:
            assert bf.contains(item) is True
    
    def test_false_positive_rate(self):
        """Test false positive rate calculation."""
        bf = BloomFilter(size=10000, num_hashes=3)
        
        # Initially zero
        assert bf.false_positive_rate() == 0.0
        
        # Add items
        for i in range(100):
            bf.add(f"item{i}")
        
        # Should have some positive rate
        fpr = bf.false_positive_rate()
        assert 0.0 < fpr < 1.0
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        bf = BloomFilter(size=1000, num_hashes=3)
        
        bf.add("test1")
        bf.add("test2")
        
        # Serialize
        data = bf.serialize()
        assert isinstance(data, bytes)
        
        # Deserialize
        bf2 = BloomFilter.deserialize(data)
        
        assert bf2.size == bf.size
        assert bf2.num_hashes == bf.num_hashes
        assert bf2.item_count == bf.item_count
        assert bf2.contains("test1") is True
        assert bf2.contains("test2") is True
    
    def test_hash_consistency(self):
        """Test that hashes are consistent."""
        bf = BloomFilter()
        
        hashes1 = bf._hashes("test")
        hashes2 = bf._hashes("test")
        
        assert hashes1 == hashes2
    
    def test_empty_bloom_filter(self):
        """Test empty bloom filter."""
        bf = BloomFilter()
        
        assert bf.contains("anything") is False
        assert bf.false_positive_rate() == 0.0


class TestPackfile:
    """Test suite for Packfile class."""
    
    def test_initialization(self):
        """Test Packfile initialization."""
        data = b"test data"
        pack = Packfile(
            pack_id="pack-123",
            level=0,
            data=data
        )
        
        assert pack.pack_id == "pack-123"
        assert pack.level == 0
        assert pack.data == data
        assert pack.checksum is not None
    
    def test_checksum_generation(self):
        """Test automatic checksum generation."""
        data = b"test data"
        pack = Packfile(
            pack_id="pack-123",
            level=0,
            data=data
        )
        
        import hashlib
        expected_checksum = hashlib.sha256(data).hexdigest()
        
        assert pack.checksum == expected_checksum
    
    def test_with_bloom_filter(self):
        """Test Packfile with bloom filter."""
        bf = BloomFilter()
        bf.add("key1")
        
        pack = Packfile(
            pack_id="pack-123",
            level=0,
            data=b"data",
            bloom_filter=bf,
            min_key="a",
            max_key="z"
        )
        
        assert pack.bloom_filter is not None
        assert pack.min_key == "a"
        assert pack.max_key == "z"


class TestMerkleLSMDAG:
    """Test suite for MerkleLSMDAG class."""
    
    def test_initialization(self):
        """Test MerkleLSMDAG initialization."""
        dag = MerkleLSMDAG()
        
        assert dag.nodes == {}
        assert dag.head is None
        assert dag.branches == {"main": None}
    
    def test_add_node(self):
        """Test adding nodes to DAG."""
        dag = MerkleLSMDAG()
        
        node_hash = dag.add_node(
            pack_ids=["pack1", "pack2"],
            metadata={"level": 0}
        )
        
        assert node_hash in dag.nodes
        assert dag.head == node_hash
        assert dag.branches["main"] == node_hash
    
    def test_add_node_with_parents(self):
        """Test adding node with parent references."""
        dag = MerkleLSMDAG()
        
        # Add first node
        parent_hash = dag.add_node(pack_ids=["pack1"])
        
        # Add child node
        child_hash = dag.add_node(
            pack_ids=["pack2"],
            parent_hashes=[parent_hash]
        )
        
        node = dag.nodes[child_hash]
        assert parent_hash in node.parent_hashes
    
    def test_get_lineage(self):
        """Test getting node lineage."""
        dag = MerkleLSMDAG()
        
        # Build chain: root -> middle -> head
        root = dag.add_node(pack_ids=["pack1"])
        middle = dag.add_node(pack_ids=["pack2"], parent_hashes=[root])
        head = dag.add_node(pack_ids=["pack3"], parent_hashes=[middle])
        
        lineage = dag.get_lineage(head)
        
        assert len(lineage) == 3
        assert lineage[0] == head
        assert lineage[-1] == root
    
    def test_verify_integrity(self):
        """Test DAG integrity verification."""
        dag = MerkleLSMDAG()
        
        # Add nodes
        parent = dag.add_node(pack_ids=["pack1"])
        child = dag.add_node(pack_ids=["pack2"], parent_hashes=[parent])
        
        # Valid node
        assert dag.verify_integrity(child) is True
        
        # Invalid node
        assert dag.verify_integrity("nonexistent") is False
    
    def test_get_dag_stats(self):
        """Test DAG statistics."""
        dag = MerkleLSMDAG()
        
        dag.add_node(pack_ids=["pack1"])
        dag.add_node(pack_ids=["pack2"])
        
        stats = dag.get_dag_stats()
        
        assert stats["total_nodes"] == 2
        assert stats["head"] is not None
        assert "branches" in stats
        assert "max_depth" in stats


class TestMerkleLSM:
    """Test suite for MerkleLSM class."""
    
    def test_initialization(self):
        """Test MerkleLSM initialization."""
        lsm = MerkleLSM(
            packfile_size_mb=32,
            compaction_strategy="adaptive"
        )
        
        assert lsm.packfile_size_mb == 32
        assert lsm.compaction_strategy == "adaptive"
        assert lsm.memtable == {}
        assert lsm.wal == []
        assert lsm.dag is not None
    
    def test_put_and_get(self):
        """Test putting and getting values."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.put("key2", "value2")
        
        assert lsm.get("key1") == "value1"
        assert lsm.get("key2") == "value2"
        assert lsm.get("nonexistent") is None
    
    def test_put_updates_existing_key(self):
        """Test updating existing key."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.put("key1", "value2")
        
        assert lsm.get("key1") == "value2"
    
    def test_delete(self):
        """Test deleting keys."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.delete("key1")
        
        assert lsm.get("key1") is None
    
    def test_memtable_flush(self):
        """Test memtable flushing."""
        lsm = MerkleLSM(
            packfile_size_mb=1,
            background_compaction=False
        )
        
        # Fill memtable
        for i in range(100):
            lsm.put(f"key{i}", f"value{i}" * 100)
        
        # Should create packfiles at level 0
        assert 0 in lsm.packfiles
        assert len(lsm.packfiles[0]) > 0
    
    def test_compaction_leveled(self):
        """Test leveled compaction."""
        lsm = MerkleLSM(
            compaction_strategy="leveled",
            background_compaction=False
        )
        
        # Create multiple packfiles at level 0
        for i in range(5):
            items = [(f"key{i}_{j}", f"value{j}") for j in range(10)]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        # Trigger compaction
        lsm.compact_level(0)
        
        # Should have moved to level 1
        assert 1 in lsm.packfiles
    
    def test_compaction_adaptive(self):
        """Test adaptive compaction."""
        lsm = MerkleLSM(
            compaction_strategy="adaptive",
            background_compaction=False
        )
        
        # Create packfiles
        for i in range(3):
            items = [(f"key{i}_{j}", f"value{j}") for j in range(10)]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        # Trigger compaction
        lsm.compact_level(0)
        
        # Should have compacted
        assert len(lsm.packfiles[0]) < 3 or 1 in lsm.packfiles
    
    def test_range_query(self):
        """Test range queries."""
        lsm = MerkleLSM(background_compaction=False)
        
        # Insert sorted data
        for i in range(100):
            lsm.put(f"key{i:03d}", f"value{i}")
        
        # Flush to packfiles
        lsm.flush_memtable()
        
        # Query range
        results = lsm.range_query("key010", "key020")
        
        assert len(results) > 0
        assert all("key010" <= k < "key020" for k, v in results)
    
    def test_pattern_match(self):
        """Test pattern matching."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("user_123", "alice")
        lsm.put("user_456", "bob")
        lsm.put("admin_789", "charlie")
        
        # Pattern match
        results = lsm.pattern_match("user_*")
        
        assert len(results) >= 2
        keys = [k for k, v in results]
        assert "user_123" in keys
        assert "user_456" in keys
    
    def test_scan(self):
        """Test scanning all keys."""
        lsm = MerkleLSM(background_compaction=False)
        
        for i in range(10):
            lsm.put(f"key{i}", f"value{i}")
        
        results = list(lsm.scan())
        
        assert len(results) >= 10
    
    def test_snapshot_creation(self):
        """Test snapshot creation."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.put("key2", "value2")
        
        snapshot_id = lsm.create_snapshot()
        
        assert snapshot_id in lsm.snapshots
        assert len(lsm.snapshots[snapshot_id]) == 2
    
    def test_snapshot_restoration(self):
        """Test snapshot restoration."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        snapshot_id = lsm.create_snapshot()
        
        # Modify data
        lsm.put("key2", "value2")
        
        # Restore
        lsm.restore_snapshot(snapshot_id)
        
        assert lsm.get("key1") == "value1"
        assert lsm.get("key2") is None
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.flush_memtable()
        
        stats = lsm.get_statistics()
        
        assert "memtable_size" in stats
        assert "packfile_count" in stats
        assert "levels" in stats
        assert "compaction_stats" in stats
    
    @pytest.mark.asyncio
    async def test_async_get(self):
        """Test async get operation."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        
        value = await lsm.get_async("key1")
        
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_async_put(self):
        """Test async put operation."""
        lsm = MerkleLSM(background_compaction=False)
        
        await lsm.put_async("key1", "value1")
        
        assert lsm.get("key1") == "value1"
    
    def test_bloom_filter_optimization(self):
        """Test bloom filter optimization."""
        lsm = MerkleLSM(
            bloom_filter=True,
            background_compaction=False
        )
        
        # Add data and flush
        for i in range(100):
            lsm.put(f"key{i}", f"value{i}")
        lsm.flush_memtable()
        
        # Packfiles should have bloom filters
        if 0 in lsm.packfiles and lsm.packfiles[0]:
            assert lsm.packfiles[0][0].bloom_filter is not None
    
    def test_tombstone_handling(self):
        """Test tombstone markers."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.delete("key1")
        
        # Should return None
        assert lsm.get("key1") is None
    
    def test_close(self):
        """Test resource cleanup."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        
        # Should not raise
        lsm.close()


class TestCompactionStrategies:
    """Test different compaction strategies."""
    
    def test_leveled_compaction_strategy(self):
        """Test leveled compaction selection."""
        lsm = MerkleLSM(
            compaction_strategy="leveled",
            background_compaction=False
        )
        
        # Create test packfiles
        for i in range(3):
            items = [(f"key{i}", "value")]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        selected = lsm._select_leveled_compaction(0)
        
        # Should select all packfiles at level
        assert len(selected) == 3
    
    def test_adaptive_compaction_strategy(self):
        """Test adaptive compaction selection."""
        lsm = MerkleLSM(
            compaction_strategy="adaptive",
            background_compaction=False
        )
        
        # Create test packfiles
        for i in range(5):
            items = [(f"key{i}", "value")]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        selected = lsm._select_adaptive_compaction(0)
        
        # Should select subset
        assert len(selected) > 0
        assert len(selected) <= 5
    
    def test_compaction_trigger_conditions(self):
        """Test compaction trigger conditions."""
        lsm = MerkleLSM(background_compaction=False)
        
        # Level 0 should compact at 4 packfiles
        for i in range(4):
            items = [(f"key{i}", "value")]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        assert lsm._should_compact(0) is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_lsm(self):
        """Test operations on empty LSM."""
        lsm = MerkleLSM(background_compaction=False)
        
        assert lsm.get("any_key") is None
        assert list(lsm.scan()) == []
        assert lsm.range_query("a", "z") == []
    
    def test_large_values(self):
        """Test with large values."""
        lsm = MerkleLSM(background_compaction=False)
        
        large_value = "x" * (1024 * 1024)  # 1MB
        lsm.put("large_key", large_value)
        
        assert lsm.get("large_key") == large_value
    
    def test_many_small_values(self):
        """Test with many small values."""
        lsm = MerkleLSM(background_compaction=False)
        
        for i in range(10000):
            lsm.put(f"key{i}", f"value{i}")
        
        # Sample check
        assert lsm.get("key0") == "value0"
        assert lsm.get("key9999") == "value9999"
    
    def test_unicode_keys_and_values(self):
        """Test with unicode data."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key_你好", "值_你好")
        lsm.put("key_مرحبا", "值_مرحبا")
        
        assert lsm.get("key_你好") == "値_你好"
        assert lsm.get("key_مرحبا") == "值_مرحبا"
    
    def test_special_characters_in_keys(self):
        """Test with special characters."""
        lsm = MerkleLSM(background_compaction=False)
        
        special_keys = [
            "key with spaces",
            "key/with/slashes",
            "key.with.dots",
            "key_with_underscores",
            "key-with-dashes"
        ]
        
        for key in special_keys:
            lsm.put(key, "value")
            assert lsm.get(key) == "value"


if __name__ == "__main__":
=======
"""
Comprehensive tests for lsm.py module.

Tests cover:
- BloomFilter operations
- MerkleLSMDAG functionality
- MerkleLSM tree operations
- Compaction strategies
- Point and range queries
- Async operations
"""

import pytest
import asyncio
import numpy as np
import pickle
import zlib
from unittest.mock import Mock, patch
from typing import List

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from lsm import (
    BloomFilter,
    Packfile,
    MerkleNode,
    MerkleLSMDAG,
    MerkleLSM
)


class TestBloomFilter:
    """Test suite for BloomFilter class."""
    
    def test_initialization(self):
        """Test BloomFilter initialization."""
        bf = BloomFilter(size=1000, num_hashes=3)
        
        assert bf.size == 1000
        assert bf.num_hashes == 3
        assert bf.item_count == 0
        assert len(bf.bit_array) == 1000
    
    def test_add_and_contains(self):
        """Test adding items and checking membership."""
        bf = BloomFilter(size=10000)
        
        bf.add("item1")
        bf.add("item2")
        bf.add("item3")
        
        assert bf.contains("item1") is True
        assert bf.contains("item2") is True
        assert bf.contains("item3") is True
        assert bf.item_count == 3
    
    def test_false_negatives_impossible(self):
        """Test that false negatives are impossible."""
        bf = BloomFilter(size=10000)
        
        items = [f"item{i}" for i in range(100)]
        
        for item in items:
            bf.add(item)
        
        # All added items should be found
        for item in items:
            assert bf.contains(item) is True
    
    def test_false_positive_rate(self):
        """Test false positive rate calculation."""
        bf = BloomFilter(size=10000, num_hashes=3)
        
        # Initially zero
        assert bf.false_positive_rate() == 0.0
        
        # Add items
        for i in range(100):
            bf.add(f"item{i}")
        
        # Should have some positive rate
        fpr = bf.false_positive_rate()
        assert 0.0 < fpr < 1.0
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        bf = BloomFilter(size=1000, num_hashes=3)
        
        bf.add("test1")
        bf.add("test2")
        
        # Serialize
        data = bf.serialize()
        assert isinstance(data, bytes)
        
        # Deserialize
        bf2 = BloomFilter.deserialize(data)
        
        assert bf2.size == bf.size
        assert bf2.num_hashes == bf.num_hashes
        assert bf2.item_count == bf.item_count
        assert bf2.contains("test1") is True
        assert bf2.contains("test2") is True
    
    def test_hash_consistency(self):
        """Test that hashes are consistent."""
        bf = BloomFilter()
        
        hashes1 = bf._hashes("test")
        hashes2 = bf._hashes("test")
        
        assert hashes1 == hashes2
    
    def test_empty_bloom_filter(self):
        """Test empty bloom filter."""
        bf = BloomFilter()
        
        assert bf.contains("anything") is False
        assert bf.false_positive_rate() == 0.0


class TestPackfile:
    """Test suite for Packfile class."""
    
    def test_initialization(self):
        """Test Packfile initialization."""
        data = b"test data"
        pack = Packfile(
            pack_id="pack-123",
            level=0,
            data=data
        )
        
        assert pack.pack_id == "pack-123"
        assert pack.level == 0
        assert pack.data == data
        assert pack.checksum is not None
    
    def test_checksum_generation(self):
        """Test automatic checksum generation."""
        data = b"test data"
        pack = Packfile(
            pack_id="pack-123",
            level=0,
            data=data
        )
        
        import hashlib
        expected_checksum = hashlib.sha256(data).hexdigest()
        
        assert pack.checksum == expected_checksum
    
    def test_with_bloom_filter(self):
        """Test Packfile with bloom filter."""
        bf = BloomFilter()
        bf.add("key1")
        
        pack = Packfile(
            pack_id="pack-123",
            level=0,
            data=b"data",
            bloom_filter=bf,
            min_key="a",
            max_key="z"
        )
        
        assert pack.bloom_filter is not None
        assert pack.min_key == "a"
        assert pack.max_key == "z"


class TestMerkleLSMDAG:
    """Test suite for MerkleLSMDAG class."""
    
    def test_initialization(self):
        """Test MerkleLSMDAG initialization."""
        dag = MerkleLSMDAG()
        
        assert dag.nodes == {}
        assert dag.head is None
        assert dag.branches == {"main": None}
    
    def test_add_node(self):
        """Test adding nodes to DAG."""
        dag = MerkleLSMDAG()
        
        node_hash = dag.add_node(
            pack_ids=["pack1", "pack2"],
            metadata={"level": 0}
        )
        
        assert node_hash in dag.nodes
        assert dag.head == node_hash
        assert dag.branches["main"] == node_hash
    
    def test_add_node_with_parents(self):
        """Test adding node with parent references."""
        dag = MerkleLSMDAG()
        
        # Add first node
        parent_hash = dag.add_node(pack_ids=["pack1"])
        
        # Add child node
        child_hash = dag.add_node(
            pack_ids=["pack2"],
            parent_hashes=[parent_hash]
        )
        
        node = dag.nodes[child_hash]
        assert parent_hash in node.parent_hashes
    
    def test_get_lineage(self):
        """Test getting node lineage."""
        dag = MerkleLSMDAG()
        
        # Build chain: root -> middle -> head
        root = dag.add_node(pack_ids=["pack1"])
        middle = dag.add_node(pack_ids=["pack2"], parent_hashes=[root])
        head = dag.add_node(pack_ids=["pack3"], parent_hashes=[middle])
        
        lineage = dag.get_lineage(head)
        
        assert len(lineage) == 3
        assert lineage[0] == head
        assert lineage[-1] == root
    
    def test_verify_integrity(self):
        """Test DAG integrity verification."""
        dag = MerkleLSMDAG()
        
        # Add nodes
        parent = dag.add_node(pack_ids=["pack1"])
        child = dag.add_node(pack_ids=["pack2"], parent_hashes=[parent])
        
        # Valid node
        assert dag.verify_integrity(child) is True
        
        # Invalid node
        assert dag.verify_integrity("nonexistent") is False
    
    def test_get_dag_stats(self):
        """Test DAG statistics."""
        dag = MerkleLSMDAG()
        
        dag.add_node(pack_ids=["pack1"])
        dag.add_node(pack_ids=["pack2"])
        
        stats = dag.get_dag_stats()
        
        assert stats["total_nodes"] == 2
        assert stats["head"] is not None
        assert "branches" in stats
        assert "max_depth" in stats


class TestMerkleLSM:
    """Test suite for MerkleLSM class."""
    
    def test_initialization(self):
        """Test MerkleLSM initialization."""
        lsm = MerkleLSM(
            packfile_size_mb=32,
            compaction_strategy="adaptive"
        )
        
        assert lsm.packfile_size_mb == 32
        assert lsm.compaction_strategy == "adaptive"
        assert lsm.memtable == {}
        assert lsm.wal == []
        assert lsm.dag is not None
    
    def test_put_and_get(self):
        """Test putting and getting values."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.put("key2", "value2")
        
        assert lsm.get("key1") == "value1"
        assert lsm.get("key2") == "value2"
        assert lsm.get("nonexistent") is None
    
    def test_put_updates_existing_key(self):
        """Test updating existing key."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.put("key1", "value2")
        
        assert lsm.get("key1") == "value2"
    
    def test_delete(self):
        """Test deleting keys."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.delete("key1")
        
        assert lsm.get("key1") is None
    
    def test_memtable_flush(self):
        """Test memtable flushing."""
        lsm = MerkleLSM(
            packfile_size_mb=1,
            background_compaction=False
        )
        
        # Fill memtable
        for i in range(100):
            lsm.put(f"key{i}", f"value{i}" * 100)
        
        # Should create packfiles at level 0
        assert 0 in lsm.packfiles
        assert len(lsm.packfiles[0]) > 0
    
    def test_compaction_leveled(self):
        """Test leveled compaction."""
        lsm = MerkleLSM(
            compaction_strategy="leveled",
            background_compaction=False
        )
        
        # Create multiple packfiles at level 0
        for i in range(5):
            items = [(f"key{i}_{j}", f"value{j}") for j in range(10)]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        # Trigger compaction
        lsm.compact_level(0)
        
        # Should have moved to level 1
        assert 1 in lsm.packfiles
    
    def test_compaction_adaptive(self):
        """Test adaptive compaction."""
        lsm = MerkleLSM(
            compaction_strategy="adaptive",
            background_compaction=False
        )
        
        # Create packfiles
        for i in range(3):
            items = [(f"key{i}_{j}", f"value{j}") for j in range(10)]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        # Trigger compaction
        lsm.compact_level(0)
        
        # Should have compacted
        assert len(lsm.packfiles[0]) < 3 or 1 in lsm.packfiles
    
    def test_range_query(self):
        """Test range queries."""
        lsm = MerkleLSM(background_compaction=False)
        
        # Insert sorted data
        for i in range(100):
            lsm.put(f"key{i:03d}", f"value{i}")
        
        # Flush to packfiles
        lsm.flush_memtable()
        
        # Query range
        results = lsm.range_query("key010", "key020")
        
        assert len(results) > 0
        assert all("key010" <= k < "key020" for k, v in results)
    
    def test_pattern_match(self):
        """Test pattern matching."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("user_123", "alice")
        lsm.put("user_456", "bob")
        lsm.put("admin_789", "charlie")
        
        # Pattern match
        results = lsm.pattern_match("user_*")
        
        assert len(results) >= 2
        keys = [k for k, v in results]
        assert "user_123" in keys
        assert "user_456" in keys
    
    def test_scan(self):
        """Test scanning all keys."""
        lsm = MerkleLSM(background_compaction=False)
        
        for i in range(10):
            lsm.put(f"key{i}", f"value{i}")
        
        results = list(lsm.scan())
        
        assert len(results) >= 10
    
    def test_snapshot_creation(self):
        """Test snapshot creation."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.put("key2", "value2")
        
        snapshot_id = lsm.create_snapshot()
        
        assert snapshot_id in lsm.snapshots
        assert len(lsm.snapshots[snapshot_id]) == 2
    
    def test_snapshot_restoration(self):
        """Test snapshot restoration."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        snapshot_id = lsm.create_snapshot()
        
        # Modify data
        lsm.put("key2", "value2")
        
        # Restore
        lsm.restore_snapshot(snapshot_id)
        
        assert lsm.get("key1") == "value1"
        assert lsm.get("key2") is None
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.flush_memtable()
        
        stats = lsm.get_statistics()
        
        assert "memtable_size" in stats
        assert "packfile_count" in stats
        assert "levels" in stats
        assert "compaction_stats" in stats
    
    @pytest.mark.asyncio
    async def test_async_get(self):
        """Test async get operation."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        
        value = await lsm.get_async("key1")
        
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_async_put(self):
        """Test async put operation."""
        lsm = MerkleLSM(background_compaction=False)
        
        await lsm.put_async("key1", "value1")
        
        assert lsm.get("key1") == "value1"
    
    def test_bloom_filter_optimization(self):
        """Test bloom filter optimization."""
        lsm = MerkleLSM(
            bloom_filter=True,
            background_compaction=False
        )
        
        # Add data and flush
        for i in range(100):
            lsm.put(f"key{i}", f"value{i}")
        lsm.flush_memtable()
        
        # Packfiles should have bloom filters
        if 0 in lsm.packfiles and lsm.packfiles[0]:
            assert lsm.packfiles[0][0].bloom_filter is not None
    
    def test_tombstone_handling(self):
        """Test tombstone markers."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        lsm.delete("key1")
        
        # Should return None
        assert lsm.get("key1") is None
    
    def test_close(self):
        """Test resource cleanup."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key1", "value1")
        
        # Should not raise
        lsm.close()


class TestCompactionStrategies:
    """Test different compaction strategies."""
    
    def test_leveled_compaction_strategy(self):
        """Test leveled compaction selection."""
        lsm = MerkleLSM(
            compaction_strategy="leveled",
            background_compaction=False
        )
        
        # Create test packfiles
        for i in range(3):
            items = [(f"key{i}", "value")]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        selected = lsm._select_leveled_compaction(0)
        
        # Should select all packfiles at level
        assert len(selected) == 3
    
    def test_adaptive_compaction_strategy(self):
        """Test adaptive compaction selection."""
        lsm = MerkleLSM(
            compaction_strategy="adaptive",
            background_compaction=False
        )
        
        # Create test packfiles
        for i in range(5):
            items = [(f"key{i}", "value")]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        selected = lsm._select_adaptive_compaction(0)
        
        # Should select subset
        assert len(selected) > 0
        assert len(selected) <= 5
    
    def test_compaction_trigger_conditions(self):
        """Test compaction trigger conditions."""
        lsm = MerkleLSM(background_compaction=False)
        
        # Level 0 should compact at 4 packfiles
        for i in range(4):
            items = [(f"key{i}", "value")]
            pack = lsm._create_packfile(items, level=0)
            lsm.packfiles[0].append(pack)
        
        assert lsm._should_compact(0) is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_lsm(self):
        """Test operations on empty LSM."""
        lsm = MerkleLSM(background_compaction=False)
        
        assert lsm.get("any_key") is None
        assert list(lsm.scan()) == []
        assert lsm.range_query("a", "z") == []
    
    def test_large_values(self):
        """Test with large values."""
        lsm = MerkleLSM(background_compaction=False)
        
        large_value = "x" * (1024 * 1024)  # 1MB
        lsm.put("large_key", large_value)
        
        assert lsm.get("large_key") == large_value
    
    def test_many_small_values(self):
        """Test with many small values."""
        lsm = MerkleLSM(background_compaction=False)
        
        for i in range(10000):
            lsm.put(f"key{i}", f"value{i}")
        
        # Sample check
        assert lsm.get("key0") == "value0"
        assert lsm.get("key9999") == "value9999"
    
    def test_unicode_keys_and_values(self):
        """Test with unicode data."""
        lsm = MerkleLSM(background_compaction=False)
        
        lsm.put("key_你好", "值_你好")
        lsm.put("key_مرحبا", "值_مرحبا")
        
        assert lsm.get("key_你好") == "値_你好"
        assert lsm.get("key_مرحبا") == "值_مرحبا"
    
    def test_special_characters_in_keys(self):
        """Test with special characters."""
        lsm = MerkleLSM(background_compaction=False)
        
        special_keys = [
            "key with spaces",
            "key/with/slashes",
            "key.with.dots",
            "key_with_underscores",
            "key-with-dashes"
        ]
        
        for key in special_keys:
            lsm.put(key, "value")
            assert lsm.get(key) == "value"


if __name__ == "__main__":
>>>>>>> ea7a1e4 (LLM training)
    pytest.main([__file__, "-v", "--tb=short"])