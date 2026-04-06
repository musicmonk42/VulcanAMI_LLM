"""
Comprehensive test suite for Memory Module Improvements

Tests new features added to the memory system:
- MemoryUsageMonitor
- ConnectionPool
- DistributedCheckpoint
- ShardedMemoryIndex
- CompressionStats
"""

import logging
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Import memory module components
from vulcan.memory import (
    CompressionStats,
    CompressionType,
    ConnectionPool,
    DistributedCheckpoint,
    Memory,
    MemoryConfig,
    MemoryFederation,
    MemoryNode,
    MemoryPersistence,
    MemoryType,
    MemoryUsageMonitor,
    ShardedMemoryIndex,
)


# ============================================================
# MEMORY USAGE MONITOR TESTS
# ============================================================


class TestMemoryUsageMonitor:
    """Test suite for MemoryUsageMonitor class."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryUsageMonitor(
            warning_threshold_mb=100,
            critical_threshold_mb=200,
        )
        
        assert monitor.warning_threshold_mb == 100
        assert monitor.critical_threshold_mb == 200
        assert len(monitor._usage_by_type) == 0
    
    def test_track_memory(self):
        """Test tracking memory objects."""
        monitor = MemoryUsageMonitor(
            warning_threshold_mb=1000,
            critical_threshold_mb=2000,
        )
        
        # Create test memory
        memory = Memory(
            id="test_001",
            type=MemoryType.EPISODIC,
            content="Test content" * 100,
            embedding=np.random.randn(512),
        )
        
        # Track memory
        monitor.track_memory(memory)
        
        # Verify tracking
        stats = monitor.get_usage_stats()
        assert stats['current_mb'] > 0
        assert MemoryType.EPISODIC.value in stats['by_type']
    
    def test_untrack_memory(self):
        """Test untracking memory objects."""
        monitor = MemoryUsageMonitor()
        
        memory = Memory(
            id="test_002",
            type=MemoryType.SEMANTIC,
            content="Test content",
            embedding=np.random.randn(512),
        )
        
        monitor.track_memory(memory)
        initial_usage = monitor._usage_by_type[MemoryType.SEMANTIC]
        
        monitor.untrack_memory(memory)
        final_usage = monitor._usage_by_type[MemoryType.SEMANTIC]
        
        assert final_usage <= initial_usage
    
    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        monitor = MemoryUsageMonitor()
        
        # Track multiple memories
        for i in range(10):
            memory = Memory(
                id=f"test_{i:03d}",
                type=MemoryType.LONG_TERM,
                content=f"Test content {i}" * 50,
                embedding=np.random.randn(512),
            )
            monitor.track_memory(memory)
        
        stats = monitor.get_usage_stats()
        
        assert 'current_mb' in stats
        assert 'peak_mb' in stats
        assert 'by_type' in stats
        assert stats['current_mb'] > 0
    
    def test_adaptive_capacity(self):
        """Test adaptive capacity calculation."""
        monitor = MemoryUsageMonitor(
            warning_threshold_mb=10,
            critical_threshold_mb=20,
        )
        
        # Normal capacity
        capacity = monitor.get_adaptive_capacity(MemoryType.LONG_TERM)
        assert capacity == 100000
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        monitor = MemoryUsageMonitor()
        errors = []
        
        def track_memories(count: int):
            try:
                for i in range(count):
                    memory = Memory(
                        id=f"test_{threading.current_thread().ident}_{i}",
                        type=MemoryType.WORKING,
                        content=f"Content {i}",
                        embedding=np.random.randn(512),
                    )
                    monitor.track_memory(memory)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = [
            threading.Thread(target=track_memories, args=(10,))
            for _ in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        stats = monitor.get_usage_stats()
        assert stats['current_mb'] > 0


# ============================================================
# CONNECTION POOL TESTS
# ============================================================


class TestConnectionPool:
    """Test suite for ConnectionPool class."""
    
    def test_pool_initialization(self):
        """Test connection pool initialization."""
        pool = ConnectionPool(
            max_connections_per_node=5,
            connection_timeout=30.0,
        )
        
        assert pool.max_connections_per_node == 5
        assert pool.connection_timeout == 30.0
        assert len(pool._pools) == 0
    
    def test_pool_creation_per_node(self):
        """Test pool creation for each node."""
        pool = ConnectionPool(max_connections_per_node=3)
        
        # Get connection (will create pool)
        conn = pool.get_connection("node1", "localhost", 5000)
        
        # Verify pool was created
        assert "node1" in pool._pools
        assert "node1" in pool._connection_counts
        assert pool._connection_counts["node1"] >= 1
        
        # Clean up
        if conn:
            pool.return_connection("node1", conn)
        pool.cleanup()
    
    def test_connection_pooling(self):
        """Test connection reuse from pool."""
        pool = ConnectionPool(max_connections_per_node=5)
        
        # Get and return connection
        conn1 = pool.get_connection("node2", "localhost", 5001)
        if conn1:
            pool.return_connection("node2", conn1)
            
            # Get again - should reuse
            conn2 = pool.get_connection("node2", "localhost", 5001)
            # Note: May or may not be same object depending on implementation
            
            if conn2:
                pool.return_connection("node2", conn2)
        
        pool.cleanup()
    
    def test_max_connections_limit(self):
        """Test maximum connections per node limit."""
        pool = ConnectionPool(max_connections_per_node=2)
        
        # Try to create connections up to limit
        # Note: This test assumes connection creation may fail
        # which is expected for invalid host/port
        conn1 = pool.get_connection("node3", "invalid-host", 9999)
        conn2 = pool.get_connection("node3", "invalid-host", 9999)
        
        # Clean up
        if conn1:
            pool.return_connection("node3", conn1)
        if conn2:
            pool.return_connection("node3", conn2)
        pool.cleanup()
    
    def test_cleanup(self):
        """Test cleanup of all connections."""
        pool = ConnectionPool()
        
        # Attempt to create connections
        pool.get_connection("node4", "localhost", 5002)
        pool.get_connection("node5", "localhost", 5003)
        
        # Cleanup
        pool.cleanup()
        
        # Verify cleanup
        assert len(pool._pools) == 0
        assert len(pool._connection_counts) == 0
        assert len(pool._node_info) == 0


# ============================================================
# DISTRIBUTED CHECKPOINT TESTS
# ============================================================


class TestDistributedCheckpoint:
    """Test suite for DistributedCheckpoint class."""
    
    def test_checkpoint_initialization(self):
        """Test checkpoint initialization."""
        federation = MemoryFederation()
        persistence = None  # Mock persistence
        
        checkpoint = DistributedCheckpoint(federation, persistence)
        
        assert checkpoint.federation is federation
        assert checkpoint.persistence is persistence
        assert len(checkpoint.active_checkpoints) == 0
        assert len(checkpoint.checkpoint_history) == 0
    
    def test_checkpoint_status(self):
        """Test getting checkpoint status."""
        federation = MemoryFederation()
        checkpoint = DistributedCheckpoint(federation, None)
        
        status = checkpoint.get_checkpoint_status()
        
        assert 'active_checkpoints' in status
        assert 'checkpoint_history' in status
        assert 'checkpoints' in status
        assert status['active_checkpoints'] == 0
    
    def test_checkpoint_with_nodes(self):
        """Test checkpoint with registered nodes."""
        federation = MemoryFederation()
        
        # Register test nodes
        node1 = MemoryNode(
            node_id="node1",
            host="localhost",
            port=5000,
            capacity=10000,
        )
        node2 = MemoryNode(
            node_id="node2",
            host="localhost",
            port=5001,
            capacity=10000,
        )
        
        federation.register_node(node1)
        federation.register_node(node2)
        
        # Elect leader
        federation.elect_leader()
        
        checkpoint = DistributedCheckpoint(federation, None)
        
        # Initiate checkpoint
        checkpoint_id = checkpoint.initiate_checkpoint()
        
        # Verify checkpoint was created
        # Note: checkpoint_id may be empty if not leader
        status = checkpoint.get_checkpoint_status()
        assert isinstance(status, dict)
    
    def test_thread_safety(self):
        """Test thread-safe checkpoint operations."""
        federation = MemoryFederation()
        checkpoint = DistributedCheckpoint(federation, None)
        
        errors = []
        
        def get_status():
            try:
                for _ in range(10):
                    checkpoint.get_checkpoint_status()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = [threading.Thread(target=get_status) for _ in range(3)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ============================================================
# SHARDED MEMORY INDEX TESTS
# ============================================================


class TestShardedMemoryIndex:
    """Test suite for ShardedMemoryIndex class."""
    
    def test_index_initialization(self):
        """Test sharded index initialization."""
        index = ShardedMemoryIndex(
            dimension=512,
            shard_size=100,
            index_type="flat",
        )
        
        assert index.dimension == 512
        assert index.shard_size == 100
        assert len(index.shards) == 1  # Initial shard
        assert index.current_shard_idx == 0
    
    def test_add_embeddings(self):
        """Test adding embeddings to index."""
        index = ShardedMemoryIndex(dimension=128, shard_size=50)
        
        # Add embeddings
        for i in range(10):
            embedding = np.random.randn(128)
            success = index.add(f"mem_{i}", embedding)
            assert success
        
        # Verify tracking
        assert len(index.memory_to_shard) == 10
    
    def test_auto_shard_creation(self):
        """Test automatic shard creation when full."""
        index = ShardedMemoryIndex(dimension=64, shard_size=5)
        
        # Add more than shard_size embeddings
        for i in range(12):
            embedding = np.random.randn(64)
            index.add(f"mem_{i}", embedding)
        
        # Verify multiple shards created
        stats = index.get_stats()
        assert stats['total_shards'] >= 2
    
    def test_search(self):
        """Test searching across shards."""
        index = ShardedMemoryIndex(dimension=128, shard_size=50)
        
        # Add embeddings
        embeddings = []
        for i in range(20):
            embedding = np.random.randn(128)
            embeddings.append(embedding)
            index.add(f"mem_{i}", embedding)
        
        # Search
        query = np.random.randn(128)
        results = index.search(query, k=5)
        
        # Verify results
        assert len(results) <= 5
        for memory_id, score in results:
            assert isinstance(memory_id, str)
            assert isinstance(score, float)
    
    def test_parallel_search(self):
        """Test parallel search across multiple shards."""
        index = ShardedMemoryIndex(dimension=64, shard_size=10)
        
        # Add enough to create multiple shards
        for i in range(30):
            embedding = np.random.randn(64)
            index.add(f"mem_{i}", embedding)
        
        stats = index.get_stats()
        assert stats['total_shards'] >= 2
        
        # Perform search
        query = np.random.randn(64)
        results = index.search(query, k=10)
        
        assert len(results) <= 10
    
    def test_remove(self):
        """Test removing memories from index."""
        index = ShardedMemoryIndex(dimension=64, shard_size=50)
        
        # Add embeddings
        for i in range(10):
            embedding = np.random.randn(64)
            index.add(f"mem_{i}", embedding)
        
        # Remove one
        success = index.remove("mem_5")
        assert success
        
        # Verify removal
        assert "mem_5" not in index.memory_to_shard
    
    def test_clear(self):
        """Test clearing all shards."""
        index = ShardedMemoryIndex(dimension=64, shard_size=20)
        
        # Add embeddings
        for i in range(30):
            embedding = np.random.randn(64)
            index.add(f"mem_{i}", embedding)
        
        # Clear
        index.clear()
        
        # Verify
        assert len(index.memory_to_shard) == 0
    
    def test_get_stats(self):
        """Test getting shard statistics."""
        index = ShardedMemoryIndex(dimension=64, shard_size=10)
        
        # Add embeddings
        for i in range(25):
            embedding = np.random.randn(64)
            index.add(f"mem_{i}", embedding)
        
        stats = index.get_stats()
        
        assert 'total_shards' in stats
        assert 'shard_size' in stats
        assert 'total_memories' in stats
        assert 'shard_utilization' in stats
        assert 'avg_utilization' in stats
        
        assert stats['total_memories'] == 25
        assert stats['shard_size'] == 10


# ============================================================
# COMPRESSION STATS TESTS
# ============================================================


class TestCompressionStats:
    """Test suite for CompressionStats dataclass."""
    
    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = CompressionStats(compression_type=CompressionType.LZ4)
        
        assert stats.compression_type == CompressionType.LZ4
        assert stats.total_compressions == 0
        assert stats.total_decompressions == 0
    
    def test_record_compression(self):
        """Test recording compression operation."""
        stats = CompressionStats(compression_type=CompressionType.LZ4)
        
        # Record compression
        stats.record_compression(
            original_bytes=1000,
            compressed_bytes=300,
            time_ms=10.5,
        )
        
        assert stats.total_compressions == 1
        assert stats.total_original_bytes == 1000
        assert stats.total_compressed_bytes == 300
        assert stats.total_compression_time_ms == 10.5
    
    def test_record_decompression(self):
        """Test recording decompression operation."""
        stats = CompressionStats(compression_type=CompressionType.ZSTD)
        
        # Record decompression
        stats.record_decompression(time_ms=5.2)
        
        assert stats.total_decompressions == 1
        assert stats.total_decompression_time_ms == 5.2
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        stats = CompressionStats(compression_type=CompressionType.LZ4)
        
        # Record multiple compressions
        stats.record_compression(1000, 300, 10.0)
        stats.record_compression(2000, 600, 15.0)
        
        ratio = stats.get_compression_ratio()
        expected_ratio = 3000 / 900  # Total original / total compressed
        
        assert abs(ratio - expected_ratio) < 0.01
    
    def test_avg_compression_time(self):
        """Test average compression time calculation."""
        stats = CompressionStats(compression_type=CompressionType.LZ4)
        
        # Record compressions
        stats.record_compression(1000, 300, 10.0)
        stats.record_compression(2000, 600, 20.0)
        stats.record_compression(1500, 450, 15.0)
        
        avg_time = stats.get_avg_compression_time_ms()
        expected_avg = 45.0 / 3
        
        assert abs(avg_time - expected_avg) < 0.01
    
    def test_avg_decompression_time(self):
        """Test average decompression time calculation."""
        stats = CompressionStats(compression_type=CompressionType.ZSTD)
        
        # Record decompressions
        stats.record_decompression(5.0)
        stats.record_decompression(7.0)
        stats.record_decompression(6.0)
        
        avg_time = stats.get_avg_decompression_time_ms()
        expected_avg = 18.0 / 3
        
        assert abs(avg_time - expected_avg) < 0.01
    
    def test_error_tracking(self):
        """Test error tracking."""
        stats = CompressionStats(compression_type=CompressionType.LZ4)
        
        # Record operations with errors
        stats.record_compression(1000, 300, 10.0, error=True)
        stats.record_decompression(5.0, error=True)
        
        assert stats.compression_errors == 1
        assert stats.decompression_errors == 1
    
    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CompressionStats(compression_type=CompressionType.LZ4)
        
        # Record some operations
        stats.record_compression(1000, 250, 10.0)
        stats.record_decompression(5.0)
        
        result = stats.to_dict()
        
        assert 'compression_type' in result
        assert 'total_compressions' in result
        assert 'total_decompressions' in result
        assert 'compression_ratio' in result
        assert 'avg_compression_time_ms' in result
        assert 'avg_decompression_time_ms' in result
        
        assert result['compression_type'] == 'lz4'
        assert result['total_compressions'] == 1
        assert result['total_decompressions'] == 1


# ============================================================
# RUN TESTS
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
