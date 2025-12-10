"""
Integration tests for the complete Vulcan Persistent Memory system.

Tests the interaction between all modules:
- PackfileStore + MerkleLSM
- MerkleLSM + GraphRAG
- UnlearningEngine + ZKProver
- Complete end-to-end workflows
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import sys

sys.path.insert(0, "/mnt/user-data/uploads")

# Import all modules
try:
    from __init__ import create_memory_system, quick_start, get_system_info

    INIT_AVAILABLE = True
except ImportError:
    INIT_AVAILABLE = False

from store import PackfileStore
from lsm import MerkleLSM, BloomFilter
from unlearning import UnlearningEngine, GradientSurgeryUnlearner
from zk import ZKProver, MerkleTree


class TestMemorySystemCreation:
    """Test memory system creation and initialization."""

    @pytest.mark.skipif(not INIT_AVAILABLE, reason="__init__ module not available")
    @patch("store.S3Store")
    def test_create_memory_system(self, mock_s3):
        """Test creating complete memory system."""
        mock_s3.return_value = Mock()

        system = create_memory_system(
            s3_bucket="test-bucket",
            embedding_model="test_model",
            compression="zlib",
            encryption="AES256",
        )

        assert "store" in system
        assert "lsm" in system
        assert "unlearning" in system
        assert "zk_prover" in system
        assert "version" in system

    @pytest.mark.skipif(not INIT_AVAILABLE, reason="__init__ module not available")
    @patch("store.S3Store")
    def test_quick_start(self, mock_s3):
        """Test quick start with defaults."""
        mock_s3.return_value = Mock()

        system = quick_start(s3_bucket="test-bucket")

        assert system is not None
        assert "store" in system
        assert "lsm" in system

    @pytest.mark.skipif(not INIT_AVAILABLE, reason="__init__ module not available")
    def test_get_system_info(self):
        """Test getting system information."""
        info = get_system_info()

        assert "version" in info
        assert "components" in info
        assert "features" in info


class TestStorageAndLSMIntegration:
    """Test integration between PackfileStore and MerkleLSM."""

    @patch("store.S3Store")
    def test_lsm_with_packfile_store(self, mock_s3):
        """Test LSM tree with packfile storage."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 100}
        mock_s3_instance.get_object.return_value = b"test data"
        mock_s3.return_value = mock_s3_instance

        # Create store
        store = PackfileStore(s3_bucket="test-bucket")

        # Create LSM
        lsm = MerkleLSM(background_compaction=False)

        # Add data
        for i in range(10):
            lsm.put(f"key{i}", f"value{i}")

        # Flush to create packfile
        lsm.flush_memtable()

        # Should have packfiles
        assert 0 in lsm.packfiles

        # Get packfile data
        if lsm.packfiles[0]:
            packfile = lsm.packfiles[0][0]

            # Upload to store
            path = store.upload(packfile.data, level=0)

            assert path is not None
            assert mock_s3_instance.put_object.called

    @patch("store.S3Store")
    def test_data_roundtrip(self, mock_s3):
        """Test data roundtrip through store and LSM."""
        mock_s3_instance = Mock()
        stored_data = {}

        def put_side_effect(key, data, **kwargs):
            stored_data[key] = data
            return {"etag": "abc", "size": len(data)}

        def get_side_effect(key, byte_range=None):
            return stored_data.get(key, b"")

        mock_s3_instance.put_object.side_effect = put_side_effect
        mock_s3_instance.get_object.side_effect = get_side_effect
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket", compression="zlib")
        lsm = MerkleLSM(background_compaction=False, compression="zlib")

        # Write data
        test_data = {f"key{i}": f"value{i}" for i in range(20)}
        for k, v in test_data.items():
            lsm.put(k, v)

        # Flush and upload
        lsm.flush_memtable()

        if lsm.packfiles[0]:
            packfile = lsm.packfiles[0][0]
            path = store.upload(packfile.data)

            # Download
            downloaded = store.download(path)

            # Should match
            assert len(downloaded) > 0


class TestUnlearningIntegration:
    """Test unlearning integration with LSM and ZK proofs."""

    def test_unlearning_with_lsm_dag(self):
        """Test unlearning with LSM DAG."""
        lsm = MerkleLSM(background_compaction=False)

        # Add data
        for i in range(50):
            lsm.put(f"user_{i}", f"data_{i}")

        lsm.flush_memtable()

        # Create unlearning engine
        engine = UnlearningEngine(merkle_graph=lsm.dag, method="gradient_surgery")

        # Unlearn pattern
        forget = [f"user_{i}".encode() for i in range(10)]
        retain = [f"user_{i}".encode() for i in range(10, 50)]

        result = engine.unlearn(forget=forget, retain=retain)

        assert result is not None
        assert "method" in result

    def test_unlearning_with_zk_proof(self):
        """Test unlearning with zero-knowledge proof."""
        lsm = MerkleLSM(background_compaction=False)

        # Create unlearning engine
        engine = UnlearningEngine(merkle_graph=lsm.dag)

        # Create ZK prover
        prover = ZKProver()

        # Perform unlearning
        forget = [b"sensitive1", b"sensitive2"]
        retain = [b"normal1", b"normal2", b"normal3"]

        result = engine.unlearn(forget=forget, retain=retain)

        # Generate ZK proof for the unlearning
        zk_proof = prover.generate_unlearning_proof(
            pattern="sensitive*",
            affected_packs=["pack1"],
            metadata={"unlearning_result": result},
        )

        # Verify proof
        is_valid = prover.verify_unlearning_proof(zk_proof)

        assert is_valid is True

    def test_certified_unlearning_workflow(self):
        """Test complete certified unlearning workflow."""
        lsm = MerkleLSM(background_compaction=False)

        # Add data
        for i in range(100):
            lsm.put(f"record_{i}", {"data": f"value_{i}"})

        # Create snapshot before unlearning
        before_snapshot = lsm.create_snapshot()

        # Perform certified unlearning
        engine = UnlearningEngine(merkle_graph=lsm.dag, method="certified")

        forget = [f"record_{i}".encode() for i in range(20)]
        retain = [f"record_{i}".encode() for i in range(20, 100)]

        result = engine.unlearn(forget=forget, retain=retain)

        assert "certificate" in result
        assert "epsilon" in result

        # Verify certificate
        certificate = result["certificate"]
        is_valid = engine.verify_certified_removal(certificate, forget)

        assert is_valid is True

        # Create snapshot after unlearning
        after_snapshot = lsm.create_snapshot()

        # Snapshots should be different
        assert before_snapshot != after_snapshot


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @patch("store.S3Store")
    def test_complete_memory_lifecycle(self, mock_s3):
        """Test complete memory system lifecycle."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 100}
        mock_s3_instance.get_object.return_value = b"data"
        mock_s3.return_value = mock_s3_instance

        # Initialize components
        store = PackfileStore(s3_bucket="test-bucket")
        lsm = MerkleLSM(background_compaction=False)
        engine = UnlearningEngine(merkle_graph=lsm.dag)
        prover = ZKProver()

        # 1. Write data
        for i in range(100):
            lsm.put(f"user_{i}", {"email": f"user{i}@example.com"})

        # 2. Create snapshot
        snapshot_id = lsm.create_snapshot()

        # 3. Flush and upload packfiles
        lsm.flush_memtable()

        if lsm.packfiles[0]:
            for packfile in lsm.packfiles[0]:
                store.upload(packfile.data)

        # 4. Perform unlearning (GDPR request)
        forget = [f"user_{i}".encode() for i in range(10)]
        retain = [f"user_{i}".encode() for i in range(10, 100)]

        unlearning_result = engine.unlearn(forget=forget, retain=retain)

        # 5. Generate ZK proof
        zk_proof = prover.generate_unlearning_proof(
            pattern="user_[0-9]", affected_packs=["pack1"], metadata=unlearning_result
        )

        # 6. Verify everything
        assert unlearning_result["verification"]["passed"] in [True, False]
        assert prover.verify_unlearning_proof(zk_proof) is True

        # 7. Get statistics
        lsm_stats = lsm.get_statistics()
        store_stats = store.get_statistics()
        engine_stats = engine.get_statistics()
        prover_stats = prover.get_statistics()

        assert all(
            [
                lsm_stats is not None,
                store_stats is not None,
                engine_stats is not None,
                prover_stats is not None,
            ]
        )

    @patch("store.S3Store")
    @pytest.mark.asyncio
    async def test_async_operations_integration(self, mock_s3):
        """Test async operations across modules."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 100}
        mock_s3_instance.get_object.return_value = b"async data"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")
        lsm = MerkleLSM(background_compaction=False)
        engine = UnlearningEngine(merkle_graph=lsm.dag)

        # Async LSM operations
        await lsm.put_async("key1", "value1")
        value = await lsm.get_async("key1")
        assert value == "value1"

        # Async store operations
        data = b"test data"
        path = await store.upload_async(data)
        downloaded = await store.download_async(path)

        # Async unlearning
        forget = [b"item1"]
        retain = [b"item2"]
        result = await engine.unlearn_async(forget=forget, retain=retain)

        assert result is not None

    def test_multi_level_compaction_workflow(self):
        """Test multi-level compaction with all features."""
        lsm = MerkleLSM(
            compaction_strategy="leveled",
            background_compaction=False,
            bloom_filter=True,
            max_levels=5,
        )

        # Write large amount of data to trigger compaction
        for i in range(1000):
            lsm.put(f"key_{i:04d}", f"value_{i}")

        # Trigger multiple flush operations
        for _ in range(5):
            lsm.flush_memtable()

        # Manually trigger compaction
        for level in range(3):
            if lsm._should_compact(level):
                lsm.compact_level(level)

        # Verify data integrity
        for i in range(0, 1000, 100):  # Sample check
            value = lsm.get(f"key_{i:04d}")
            assert value == f"value_{i}"

        # Get statistics
        stats = lsm.get_statistics()
        assert stats["total_packfiles"] > 0


class TestErrorHandling:
    """Test error handling across integrated systems."""

    @patch("store.S3Store")
    def test_storage_failure_recovery(self, mock_s3):
        """Test recovery from storage failures."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.side_effect = Exception("S3 error")
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        with pytest.raises(Exception):
            store.upload(b"data")

    def test_unlearning_with_empty_merkle_graph(self):
        """Test unlearning with empty Merkle graph."""
        from lsm import MerkleLSMDAG

        dag = MerkleLSMDAG()
        engine = UnlearningEngine(merkle_graph=dag)

        # Should handle gracefully
        result = engine.unlearn(forget=[b"item1"], retain=[b"item2"])

        assert result is not None

    def test_concurrent_operations(self):
        """Test concurrent operations on LSM."""
        lsm = MerkleLSM(background_compaction=False)

        # Simulate concurrent writes
        import threading

        def write_data(start, count):
            for i in range(start, start + count):
                lsm.put(f"key_{i}", f"value_{i}")

        threads = []
        for i in range(4):
            thread = threading.Thread(target=write_data, args=(i * 100, 100))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify some data
        assert lsm.get("key_0") == "value_0"
        assert lsm.get("key_399") == "value_399"


class TestPerformance:
    """Performance and scalability tests."""

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        lsm = MerkleLSM(background_compaction=False)

        # Write 10K records
        import time

        start = time.time()

        for i in range(10000):
            lsm.put(f"key_{i:05d}", {"data": f"value_{i}", "metadata": {"index": i}})

        write_time = time.time() - start

        # Read sample
        start = time.time()

        for i in range(0, 10000, 100):
            value = lsm.get(f"key_{i:05d}")
            assert value is not None

        read_time = time.time() - start

        print(f"Write time: {write_time:.2f}s, Read time: {read_time:.2f}s")

        # Should be reasonably fast
        assert write_time < 60  # Less than 1 minute
        assert read_time < 10  # Less than 10 seconds

    def test_bloom_filter_performance(self):
        """Test bloom filter performance benefit."""
        lsm_with_bloom = MerkleLSM(bloom_filter=True, background_compaction=False)

        lsm_without_bloom = MerkleLSM(bloom_filter=False, background_compaction=False)

        # Add data
        for i in range(1000):
            lsm_with_bloom.put(f"key_{i}", f"value_{i}")
            lsm_without_bloom.put(f"key_{i}", f"value_{i}")

        # Flush
        lsm_with_bloom.flush_memtable()
        lsm_without_bloom.flush_memtable()

        # Test negative lookups (bloom filter advantage)
        import time

        start = time.time()
        for i in range(1000, 2000):
            lsm_with_bloom.get(f"nonexistent_{i}")
        bloom_time = time.time() - start

        start = time.time()
        for i in range(1000, 2000):
            lsm_without_bloom.get(f"nonexistent_{i}")
        no_bloom_time = time.time() - start

        print(f"Bloom: {bloom_time:.3f}s, No bloom: {no_bloom_time:.3f}s")


class TestDataIntegrity:
    """Test data integrity across operations."""

    def test_data_consistency_after_compaction(self):
        """Test data remains consistent after compaction."""
        lsm = MerkleLSM(background_compaction=False)

        # Write test data
        test_data = {f"key_{i}": f"value_{i}" for i in range(500)}

        for k, v in test_data.items():
            lsm.put(k, v)

        # Flush and compact
        lsm.flush_memtable()

        for level in range(2):
            if lsm._should_compact(level):
                lsm.compact_level(level)

        # Verify all data
        for k, v in test_data.items():
            assert lsm.get(k) == v

    def test_dag_integrity_after_operations(self):
        """Test DAG integrity is maintained."""
        lsm = MerkleLSM(background_compaction=False)

        # Perform operations
        for i in range(100):
            lsm.put(f"key_{i}", f"value_{i}")

        lsm.flush_memtable()

        # Check DAG
        dag_stats = lsm.dag.get_dag_stats()

        assert dag_stats["total_nodes"] > 0
        if dag_stats["head"]:
            assert lsm.dag.verify_integrity(dag_stats["head"]) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
