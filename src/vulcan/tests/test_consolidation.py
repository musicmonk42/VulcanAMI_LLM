"""Test suite for consolidation.py - Memory consolidation and optimization"""

import pytest
import numpy as np
import time
import hashlib
import pickle
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
import copy

# Import the module to test
from vulcan.memory.consolidation import (
    ConsolidationStrategy,
    ClusteringAlgorithm,
    KMeansClustering,
    DBSCANClustering,
    HierarchicalClustering,
    MemoryConsolidator,
    MemoryOptimizer,
    IndexManager,
    CacheManager,
)
from vulcan.memory.base import Memory, MemoryType

# Try to import optional dependencies for testing
try:
    import sklearn

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_memories():
    """Create sample memories for testing."""
    memories = []
    for i in range(10):
        memory = Memory(
            id=f"mem_{i}",
            type=MemoryType.EPISODIC if i % 2 == 0 else MemoryType.SEMANTIC,
            content=f"Content {i}",
            timestamp=time.time() - (10 - i) * 3600,  # Spread over 10 hours
            importance=0.5 + (i % 3) * 0.2,
            access_count=i,
        )
        # Add embeddings to some memories
        if i % 2 == 0:
            memory.embedding = np.random.rand(128)
        memories.append(memory)
    return memories


@pytest.fixture
def memories_with_embeddings():
    """Create memories with embeddings for clustering tests."""
    memories = []
    # Create 3 clusters of memories
    for cluster in range(3):
        cluster_center = np.random.rand(128)
        for i in range(5):
            memory = Memory(
                id=f"mem_c{cluster}_i{i}",
                type=MemoryType.SEMANTIC,
                content=f"Cluster {cluster} Item {i}",
                timestamp=time.time(),
                importance=0.5 + np.random.rand() * 0.5,
            )
            # Add noise to cluster center
            memory.embedding = cluster_center + np.random.randn(128) * 0.1
            memory.embedding = memory.embedding / np.linalg.norm(memory.embedding)
            memories.append(memory)
    return memories


@pytest.fixture
def consolidator():
    """Create a MemoryConsolidator instance."""
    return MemoryConsolidator()


@pytest.fixture
def optimizer():
    """Create a MemoryOptimizer instance."""
    return MemoryOptimizer()


# ============================================================
# CONSOLIDATION STRATEGY ENUM TESTS
# ============================================================


class TestConsolidationStrategy:
    """Test ConsolidationStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies have expected values."""
        assert ConsolidationStrategy.IMPORTANCE_BASED.value == "importance"
        assert ConsolidationStrategy.FREQUENCY_BASED.value == "frequency"
        assert ConsolidationStrategy.RECENCY_BASED.value == "recency"
        assert ConsolidationStrategy.SEMANTIC_CLUSTERING.value == "semantic"
        assert ConsolidationStrategy.CAUSAL_CHAINS.value == "causal"
        assert ConsolidationStrategy.INFORMATION_THEORETIC.value == "information"
        assert ConsolidationStrategy.ADAPTIVE.value == "adaptive"
        assert ConsolidationStrategy.HIERARCHICAL.value == "hierarchical"
        assert ConsolidationStrategy.GRAPH_BASED.value == "graph"


# ============================================================
# CLUSTERING ALGORITHM TESTS
# ============================================================


class TestKMeansClustering:
    """Test K-means clustering implementation."""

    def test_cluster_basic(self):
        """Test basic K-means clustering."""
        # Create simple 2D data for easy verification
        embeddings = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],  # Cluster 1
                [5, 5],
                [5, 6],
                [6, 5],  # Cluster 2
            ]
        )

        labels = KMeansClustering.cluster(embeddings, n_clusters=2)

        assert len(labels) == 6
        # Check that points are grouped correctly
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_simple_kmeans_fallback(self):
        """Test simple K-means implementation without sklearn."""
        embeddings = np.random.rand(20, 10)

        labels = KMeansClustering._simple_kmeans(embeddings, n_clusters=3)

        assert len(labels) == 20
        assert len(set(labels)) <= 3
        assert all(0 <= label < 3 for label in labels)

    def test_single_cluster(self):
        """Test K-means with single cluster."""
        embeddings = np.random.rand(10, 5)

        labels = KMeansClustering.cluster(embeddings, n_clusters=1)

        assert len(labels) == 10
        assert all(label == 0 for label in labels)


class TestDBSCANClustering:
    """Test DBSCAN clustering implementation."""

    def test_cluster_with_noise(self):
        """Test DBSCAN clustering with noise points."""
        # Create clusters with noise
        embeddings = np.array(
            [
                [0, 0],
                [0, 0.1],
                [0.1, 0],  # Dense cluster
                [5, 5],
                [5, 5.1],
                [5.1, 5],  # Another dense cluster
                [10, 10],  # Noise point
            ]
        )

        labels = DBSCANClustering.cluster(embeddings, eps=0.5, min_samples=2)

        assert len(labels) == 7
        # Noise point should be labeled -1
        assert -1 in labels

    def test_simple_dbscan_fallback(self):
        """Test simple DBSCAN implementation."""
        embeddings = np.random.rand(15, 5)

        labels = DBSCANClustering._simple_dbscan(embeddings, eps=0.5, min_samples=3)

        assert len(labels) == 15
        # All labels should be >= -1 (including noise)
        assert all(label >= -1 for label in labels)


class TestHierarchicalClustering:
    """Test hierarchical clustering implementation."""

    def test_cluster_basic(self):
        """Test basic hierarchical clustering."""
        embeddings = np.random.rand(10, 5)

        labels = HierarchicalClustering.cluster(embeddings, n_clusters=3)

        assert len(labels) == 10
        assert len(set(labels)) <= 3

    def test_simple_hierarchical_fallback(self):
        """Test simple hierarchical clustering implementation."""
        embeddings = np.random.rand(8, 3)

        labels = HierarchicalClustering._simple_hierarchical(embeddings, n_clusters=2)

        assert len(labels) == 8
        assert len(set(labels)) == 2


# ============================================================
# MEMORY CONSOLIDATOR TESTS
# ============================================================


class TestMemoryConsolidator:
    """Test MemoryConsolidator class."""

    def test_initialization(self, consolidator):
        """Test consolidator initialization."""
        assert len(consolidator.strategies) == 9
        assert ConsolidationStrategy.IMPORTANCE_BASED in consolidator.strategies
        assert len(consolidator.consolidation_history) == 0
        assert isinstance(consolidator.performance_metrics, defaultdict)

    def test_consolidate_empty(self, consolidator):
        """Test consolidation with empty memory list."""
        result = consolidator.consolidate([])
        assert result == []

    def test_consolidate_by_importance(self, consolidator, sample_memories):
        """Test importance-based consolidation."""
        result = consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.IMPORTANCE_BASED,
            target_count=5,
        )

        assert len(result) == 5
        # Should keep memories with highest salience
        assert all(isinstance(m, Memory) for m in result)

    def test_consolidate_by_frequency(self, consolidator, sample_memories):
        """Test frequency-based consolidation."""
        result = consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.FREQUENCY_BASED,
            target_count=5,
        )

        assert len(result) <= len(sample_memories)
        # Higher access counts should be preferred
        access_counts = [m.access_count for m in result]
        assert access_counts == sorted(access_counts, reverse=True)

    def test_consolidate_by_recency(self, consolidator, sample_memories):
        """Test recency-based consolidation."""
        result = consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.RECENCY_BASED,
            time_window=5 * 3600,  # 5 hours
        )

        assert len(result) <= len(sample_memories)
        # Should keep recent memories
        current_time = time.time()
        for memory in result:
            age = current_time - memory.timestamp
            assert age <= 6 * 3600  # Allow some margin

    def test_consolidate_by_semantics(self, consolidator, memories_with_embeddings):
        """Test semantic clustering consolidation."""
        result = consolidator.consolidate(
            memories_with_embeddings,
            strategy=ConsolidationStrategy.SEMANTIC_CLUSTERING,
            target_count=3,
        )

        assert len(result) <= len(memories_with_embeddings)
        # Should have representatives from clusters
        assert len(result) >= 1

    def test_consolidate_by_causality_simple(self, consolidator, sample_memories):
        """Test causal chain consolidation without NetworkX."""
        # Patch NETWORKX_AVAILABLE to force simple implementation
        with patch("vulcan.memory.consolidation.NETWORKX_AVAILABLE", False):
            result = consolidator.consolidate(
                sample_memories,
                strategy=ConsolidationStrategy.CAUSAL_CHAINS,
                target_count=5,
            )

        assert len(result) <= len(sample_memories)

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
    def test_consolidate_by_causality_graph(self, consolidator, sample_memories):
        """Test causal chain consolidation with NetworkX."""
        result = consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.CAUSAL_CHAINS,
            target_count=5,
        )

        assert len(result) <= len(sample_memories)

    def test_consolidate_by_information(self, consolidator, sample_memories):
        """Test information-theoretic consolidation."""
        result = consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.INFORMATION_THEORETIC,
            target_count=5,
        )

        assert len(result) == 5
        assert all(isinstance(m, Memory) for m in result)

    def test_consolidate_hierarchical(self, consolidator, sample_memories):
        """Test hierarchical consolidation."""
        result = consolidator.consolidate(
            sample_memories, strategy=ConsolidationStrategy.HIERARCHICAL, target_count=5
        )

        assert len(result) <= len(sample_memories)
        # Should have memories from different types
        types = set(m.type for m in result)
        assert len(types) >= 1

    def test_consolidate_graph_based_fallback(self, consolidator, sample_memories):
        """Test graph-based consolidation fallback when NetworkX not available."""
        with patch("vulcan.memory.consolidation.NETWORKX_AVAILABLE", False):
            result = consolidator.consolidate(
                sample_memories,
                strategy=ConsolidationStrategy.GRAPH_BASED,
                target_count=5,
            )

        # Should fall back to importance-based
        assert len(result) == 5

    def test_adaptive_strategy_selection(self, consolidator, memories_with_embeddings):
        """Test adaptive strategy selection."""
        # Mock _select_best_strategy to verify it's called
        with patch.object(consolidator, "_select_best_strategy") as mock_select:
            mock_select.return_value = ConsolidationStrategy.SEMANTIC_CLUSTERING

            result = consolidator.consolidate(
                memories_with_embeddings,
                strategy=ConsolidationStrategy.ADAPTIVE,
                target_count=5,
            )

            mock_select.assert_called_once()

    def test_consolidation_history_tracking(self, consolidator, sample_memories):
        """Test that consolidation history is tracked."""
        initial_history_len = len(consolidator.consolidation_history)

        consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.IMPORTANCE_BASED,
            target_count=5,
        )

        assert len(consolidator.consolidation_history) == initial_history_len + 1

        last_history = consolidator.consolidation_history[-1]
        assert "timestamp" in last_history
        assert "strategy" in last_history
        assert "input_count" in last_history
        assert "output_count" in last_history
        assert "compression_ratio" in last_history

    def test_performance_metrics_tracking(self, consolidator, sample_memories):
        """Test that performance metrics are tracked."""
        strategy = ConsolidationStrategy.IMPORTANCE_BASED

        consolidator.consolidate(sample_memories, strategy=strategy, target_count=5)

        assert strategy in consolidator.performance_metrics
        assert len(consolidator.performance_metrics[strategy]) > 0

        metrics = consolidator.performance_metrics[strategy][-1]
        assert "compression_ratio" in metrics
        assert "time" in metrics
        assert "quality" in metrics

    def test_find_optimal_clusters(self, consolidator, memories_with_embeddings):
        """Test optimal cluster number finding."""
        embeddings = np.array(
            [m.embedding for m in memories_with_embeddings if m.embedding is not None]
        )

        n_clusters = consolidator._find_optimal_clusters(embeddings)

        assert isinstance(n_clusters, int)
        assert 1 <= n_clusters <= min(10, len(embeddings))

    def test_select_cluster_representative(
        self, consolidator, memories_with_embeddings
    ):
        """Test cluster representative selection."""
        cluster_memories = memories_with_embeddings[:5]
        cluster_embeddings = np.array([m.embedding for m in cluster_memories])

        representative = consolidator._select_cluster_representative(
            cluster_memories, cluster_embeddings
        )

        assert isinstance(representative, Memory)
        assert representative in cluster_memories or representative.id.startswith(
            "merged_"
        )

    def test_remove_near_duplicates(self, consolidator):
        """Test near-duplicate removal."""
        # Create memories with duplicate content
        memories = [
            Memory(id="1", type=MemoryType.SEMANTIC, content="Same content"),
            Memory(id="2", type=MemoryType.SEMANTIC, content="Same content"),
            Memory(id="3", type=MemoryType.SEMANTIC, content="Different content"),
        ]

        unique = consolidator._remove_near_duplicates(memories)

        assert len(unique) == 2
        content_set = set(m.content for m in unique)
        assert "Same content" in content_set
        assert "Different content" in content_set

    def test_calculate_similarity(self, consolidator):
        """Test memory similarity calculation."""
        memory1 = Memory(
            id="1",
            type=MemoryType.SEMANTIC,
            content="Content 1",
            timestamp=time.time(),
            embedding=np.random.rand(128),
        )
        memory2 = Memory(
            id="2",
            type=MemoryType.SEMANTIC,
            content="Content 2",
            timestamp=time.time() + 100,
            embedding=np.random.rand(128),
        )

        similarity = consolidator._calculate_similarity(memory1, memory2)

        assert 0 <= similarity <= 1

    def test_calculate_causal_score(self, consolidator):
        """Test causal relationship scoring."""
        cause = Memory(
            id="cause",
            type=MemoryType.EPISODIC,
            content="Cause event",
            timestamp=time.time(),
        )
        effect = Memory(
            id="effect",
            type=MemoryType.EPISODIC,
            content="Effect event",
            timestamp=time.time() + 1800,  # 30 minutes later
        )

        score = consolidator._calculate_causal_score(cause, effect)

        assert 0 <= score <= 1

    def test_merge_memories(self, consolidator, sample_memories):
        """Test memory merging."""
        memories_to_merge = sample_memories[:3]

        merged = consolidator._merge_memories(memories_to_merge)

        assert isinstance(merged, Memory)
        assert merged.id.startswith("merged_")
        assert "merged_from" in merged.content
        assert len(merged.content["merged_from"]) == 3

    def test_calculate_entropy(self, consolidator):
        """Test entropy calculation."""
        # High entropy text (random)
        high_entropy_text = "abcdefghijklmnop"
        high_entropy = consolidator._calculate_entropy(high_entropy_text)

        # Low entropy text (repetitive)
        low_entropy_text = "aaaaaaaaaa"
        low_entropy = consolidator._calculate_entropy(low_entropy_text)

        assert 0 <= high_entropy <= 1
        assert 0 <= low_entropy <= 1
        assert high_entropy > low_entropy

    def test_calculate_information_content(self, consolidator):
        """Test information content calculation."""
        memory = Memory(
            id="test",
            type=MemoryType.SEMANTIC,
            content="Test content with some information",
            timestamp=time.time(),
            importance=0.7,
            access_count=5,
        )

        info_content = consolidator._calculate_information_content(memory)

        assert 0 <= info_content <= 1

    def test_evaluate_consolidation_quality(self, consolidator, sample_memories):
        """Test consolidation quality evaluation."""
        original = sample_memories
        consolidated = sample_memories[:5]

        quality = consolidator._evaluate_consolidation_quality(original, consolidated)

        assert 0 <= quality <= 1


# ============================================================
# MEMORY OPTIMIZER TESTS
# ============================================================


class TestMemoryOptimizer:
    """Test MemoryOptimizer class."""

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert len(optimizer.optimization_history) == 0
        assert isinstance(optimizer.index_manager, IndexManager)
        assert isinstance(optimizer.cache_manager, CacheManager)

    def test_optimize_basic(self, optimizer, sample_memories):
        """Test basic memory optimization."""
        memories_dict = {m.id: m for m in sample_memories}

        optimized = optimizer.optimize(memories_dict)

        assert isinstance(optimized, dict)
        assert len(optimized) <= len(memories_dict)

    def test_remove_duplicates(self, optimizer):
        """Test duplicate removal in optimization."""
        memories = {
            "1": Memory(
                id="1", type=MemoryType.SEMANTIC, content="Same", importance=0.5
            ),
            "2": Memory(
                id="2", type=MemoryType.SEMANTIC, content="Same", importance=0.7
            ),
            "3": Memory(
                id="3", type=MemoryType.SEMANTIC, content="Different", importance=0.6
            ),
        }

        unique = optimizer._remove_duplicates(memories)

        assert len(unique) == 2
        # Should keep the one with higher importance
        assert "2" in unique
        assert "3" in unique

    def test_compress_large_memories(self, optimizer):
        """Test marking large memories for compression."""
        # Create a large memory
        large_content = {"data": "x" * 20000}
        memories = {
            "1": Memory(id="1", type=MemoryType.SEMANTIC, content=large_content),
            "2": Memory(id="2", type=MemoryType.SEMANTIC, content="Small"),
        }

        result = optimizer._compress_large_memories(memories)

        assert "needs_compression" in result["1"].metadata
        assert "needs_compression" not in result["2"].metadata

    def test_update_indices(self, optimizer):
        """Test index updating."""
        memories = {"1": Memory(id="1", type=MemoryType.SEMANTIC, content="Test")}

        result = optimizer._update_indices(memories)

        assert result["1"].metadata["indexed"] is True
        assert "index_version" in result["1"].metadata

    def test_rebalance_memory_types(self, optimizer):
        """Test memory type rebalancing."""
        # Create unbalanced distribution
        memories = {}
        for i in range(20):
            memory_type = MemoryType.SEMANTIC if i < 18 else MemoryType.EPISODIC
            memories[str(i)] = Memory(
                id=str(i), type=memory_type, content=f"Content {i}"
            )

        result = optimizer._rebalance_memory_types(memories)

        # Should return memories (actual rebalancing would require type conversion)
        assert len(result) == len(memories)

    def test_optimization_history_tracking(self, optimizer, sample_memories):
        """Test that optimization history is tracked."""
        memories_dict = {m.id: m for m in sample_memories}

        optimizer.optimize(memories_dict)

        assert len(optimizer.optimization_history) == 1

        history = optimizer.optimization_history[0]
        assert "timestamp" in history
        assert "duration_ms" in history
        assert "input_count" in history
        assert "output_count" in history
        assert "reduction_ratio" in history


# ============================================================
# INDEX MANAGER TESTS
# ============================================================


class TestIndexManager:
    """Test IndexManager class."""

    def test_initialization(self):
        """Test index manager initialization."""
        manager = IndexManager()

        assert manager.version == 1
        assert "content_hash" in manager.indices
        assert "type_index" in manager.indices
        assert "timestamp_index" in manager.indices
        assert "importance_index" in manager.indices

    def test_update_indices(self, sample_memories):
        """Test updating indices."""
        manager = IndexManager()
        memories_dict = {m.id: m for m in sample_memories}

        manager.update(memories_dict)

        # Check content hash index
        assert len(manager.indices["content_hash"]) > 0

        # Check type index
        assert MemoryType.EPISODIC in manager.indices["type_index"]
        assert MemoryType.SEMANTIC in manager.indices["type_index"]

        # Check sorted indices
        assert len(manager.indices["timestamp_index"]) == len(memories_dict)
        assert len(manager.indices["importance_index"]) == len(memories_dict)

        # Check version increment
        assert manager.version == 2

    def test_indices_sorting(self):
        """Test that indices are properly sorted."""
        manager = IndexManager()
        memories = {
            "1": Memory(
                id="1",
                type=MemoryType.SEMANTIC,
                content="A",
                timestamp=100,
                importance=0.5,
            ),
            "2": Memory(
                id="2",
                type=MemoryType.SEMANTIC,
                content="B",
                timestamp=200,
                importance=0.8,
            ),
            "3": Memory(
                id="3",
                type=MemoryType.SEMANTIC,
                content="C",
                timestamp=150,
                importance=0.3,
            ),
        }

        manager.update(memories)

        # Check timestamp index is sorted ascending
        timestamps = [t for t, _ in manager.indices["timestamp_index"]]
        assert timestamps == sorted(timestamps)

        # Check importance index is sorted descending
        importances = [i for i, _ in manager.indices["importance_index"]]
        assert importances == sorted(importances, reverse=True)


# ============================================================
# CACHE MANAGER TESTS
# ============================================================


class TestCacheManager:
    """Test CacheManager class."""

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager()

        assert len(manager.access_cache) == 0
        assert len(manager.embedding_cache) == 0
        assert manager.max_cache_size == 1000

    def test_optimize_caches(self, sample_memories):
        """Test cache optimization."""
        manager = CacheManager()
        memories_dict = {m.id: m for m in sample_memories}

        manager.optimize(memories_dict)

        # Check access cache populated with high-access memories
        assert len(manager.access_cache) > 0
        assert len(manager.access_cache) <= manager.max_cache_size // 2

        # Check embedding cache populated
        memories_with_embeddings = sum(
            1 for m in sample_memories if m.embedding is not None
        )
        assert len(manager.embedding_cache) <= min(
            memories_with_embeddings, manager.max_cache_size // 2
        )

    def test_cache_size_limits(self):
        """Test that cache size limits are respected."""
        manager = CacheManager()
        manager.max_cache_size = 10

        # Create many memories
        memories = {}
        for i in range(20):
            memory = Memory(
                id=str(i),
                type=MemoryType.SEMANTIC,
                content=f"Content {i}",
                access_count=i,
            )
            if i % 2 == 0:
                memory.embedding = np.random.rand(10)
            memories[str(i)] = memory

        manager.optimize(memories)

        assert len(manager.access_cache) <= manager.max_cache_size // 2
        assert len(manager.embedding_cache) <= manager.max_cache_size // 2


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for consolidation system."""

    def test_consolidation_with_optimization(
        self, consolidator, optimizer, sample_memories
    ):
        """Test consolidation followed by optimization."""
        # First consolidate
        consolidated = consolidator.consolidate(
            sample_memories,
            strategy=ConsolidationStrategy.IMPORTANCE_BASED,
            target_count=5,
        )

        # Then optimize
        consolidated_dict = {m.id: m for m in consolidated}
        optimized = optimizer.optimize(consolidated_dict)

        assert len(optimized) <= len(consolidated_dict)
        assert all(isinstance(m, Memory) for m in optimized.values())

    def test_multiple_consolidations(self, consolidator, sample_memories):
        """Test multiple consecutive consolidations."""
        result = sample_memories

        for strategy in [
            ConsolidationStrategy.IMPORTANCE_BASED,
            ConsolidationStrategy.FREQUENCY_BASED,
            ConsolidationStrategy.SEMANTIC_CLUSTERING,
        ]:
            if len(result) > 3:
                result = consolidator.consolidate(
                    result, strategy=strategy, target_count=max(3, len(result) // 2)
                )

        assert len(result) <= len(sample_memories)
        assert len(result) >= 1

    def test_end_to_end_workflow(self, memories_with_embeddings):
        """Test complete consolidation workflow."""
        consolidator = MemoryConsolidator()
        optimizer = MemoryOptimizer()

        # Start with many memories
        initial_count = len(memories_with_embeddings)

        # Consolidate using adaptive strategy
        consolidated = consolidator.consolidate(
            memories_with_embeddings,
            strategy=ConsolidationStrategy.ADAPTIVE,
            target_count=initial_count // 3,
        )

        # Optimize the consolidated memories
        consolidated_dict = {m.id: m for m in consolidated}
        optimized = optimizer.optimize(consolidated_dict)

        # Verify reduction and quality
        assert len(optimized) < initial_count
        assert len(consolidator.consolidation_history) > 0
        assert len(optimizer.optimization_history) > 0


# ============================================================
# PERFORMANCE TESTS
# ============================================================


class TestPerformance:
    """Performance tests for consolidation."""

    def test_large_memory_consolidation(self, consolidator):
        """Test consolidation with large number of memories."""
        # Create 1000 memories
        memories = []
        for i in range(1000):
            memory = Memory(
                id=f"perf_{i}",
                type=MemoryType.SEMANTIC,
                content=f"Content {i}",
                timestamp=time.time() - i,
                importance=np.random.rand(),
            )
            if i % 3 == 0:
                memory.embedding = np.random.rand(128)
            memories.append(memory)

        start_time = time.time()
        result = consolidator.consolidate(
            memories, strategy=ConsolidationStrategy.IMPORTANCE_BASED, target_count=100
        )
        elapsed = time.time() - start_time

        assert len(result) == 100
        assert elapsed < 5.0  # Should complete within 5 seconds

    def test_clustering_performance(self, consolidator):
        """Test clustering performance with many embeddings."""
        embeddings = np.random.rand(500, 128)

        start_time = time.time()
        n_clusters = consolidator._find_optimal_clusters(embeddings)
        elapsed = time.time() - start_time

        assert isinstance(n_clusters, int)
        assert elapsed < 2.0  # Should complete within 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
