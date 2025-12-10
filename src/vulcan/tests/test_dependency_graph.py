"""
test_dependency_graph.py - Comprehensive tests for DependencyGraph
Part of the VULCAN-AGI system

Tests cover:
- Graph storage and structure
- Cycle detection and breaking
- Path finding and traversal
- Caching and performance
- Thread safety
- Edge cases and error handling
"""

import threading
import time

import pytest

from vulcan.curiosity_engine.dependency_graph import (
    CacheManager, CycleAwareDependencyGraph, CycleDetector, DependencyAnalyzer,
    DependencyEdge, DependencyType, GraphStorage, PathFinder, ROICalculator,
    TopologicalSorter)
from vulcan.curiosity_engine.gap_analyzer import KnowledgeGap

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_gap():
    """Create a mock knowledge gap"""
    gap = KnowledgeGap(
        type="decomposition",
        domain="test_domain",
        priority=0.8,
        estimated_cost=10.0,
        complexity=0.5,
    )
    return gap


@pytest.fixture
def graph_storage():
    """Create a GraphStorage instance"""
    return GraphStorage(max_nodes=100, max_edges=200)


@pytest.fixture
def dependency_graph():
    """Create a CycleAwareDependencyGraph instance"""
    return CycleAwareDependencyGraph(max_nodes=100, max_edges=200)


@pytest.fixture
def dependency_analyzer():
    """Create a DependencyAnalyzer instance"""
    return DependencyAnalyzer()


@pytest.fixture
def roi_calculator():
    """Create an ROICalculator instance"""
    return ROICalculator()


# ============================================================================
# Test DependencyType
# ============================================================================


class TestDependencyType:
    """Tests for DependencyType enum"""

    def test_dependency_types_exist(self):
        """Test that all dependency types exist"""
        assert DependencyType.CAUSAL
        assert DependencyType.PREREQUISITE
        assert DependencyType.RELATED
        assert DependencyType.WEAK

    def test_dependency_type_values(self):
        """Test dependency type values"""
        assert DependencyType.CAUSAL.value == "causal"
        assert DependencyType.PREREQUISITE.value == "prerequisite"
        assert DependencyType.RELATED.value == "related"
        assert DependencyType.WEAK.value == "weak"


# ============================================================================
# Test DependencyEdge
# ============================================================================


class TestDependencyEdge:
    """Tests for DependencyEdge class"""

    def test_create_edge(self, mock_gap):
        """Test creating a dependency edge"""
        gap1 = mock_gap
        gap2 = KnowledgeGap(
            type="causal", domain="test2", priority=0.7, estimated_cost=15.0
        )

        edge = DependencyEdge(
            source=gap1,
            target=gap2,
            dependency_type=DependencyType.PREREQUISITE,
            strength=0.8,
        )

        assert edge.source == gap1
        assert edge.target == gap2
        assert edge.dependency_type == DependencyType.PREREQUISITE
        assert edge.strength == 0.8

    def test_edge_cost_calculation(self, mock_gap):
        """Test edge cost calculation for different types"""
        gap1 = mock_gap
        gap2 = KnowledgeGap(
            type="test", domain="test2", priority=0.5, estimated_cost=10.0
        )

        # Weak edge should have low cost
        weak_edge = DependencyEdge(
            source=gap1, target=gap2, dependency_type=DependencyType.WEAK, strength=1.0
        )
        weak_cost = weak_edge.get_cost()

        # Causal edge should have high cost
        causal_edge = DependencyEdge(
            source=gap1,
            target=gap2,
            dependency_type=DependencyType.CAUSAL,
            strength=1.0,
        )
        causal_cost = causal_edge.get_cost()

        assert weak_cost < causal_cost
        assert weak_cost == 0.1  # 0.1 * 1.0
        assert causal_cost == 5.0  # 5.0 * 1.0

    def test_edge_age(self):
        """Test edge age calculation"""
        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.RELATED
        )

        time.sleep(0.1)
        age = edge.age()

        assert age >= 0.1
        assert age < 1.0


# ============================================================================
# Test GraphStorage
# ============================================================================


class TestGraphStorage:
    """Tests for GraphStorage class"""

    def test_add_node(self, graph_storage):
        """Test adding a node"""
        result = graph_storage.add_node("node1", {"data": "test"})

        assert result is True
        assert "node1" in graph_storage.nodes
        assert graph_storage.node_count() == 1

    def test_add_duplicate_node(self, graph_storage):
        """Test adding duplicate node"""
        graph_storage.add_node("node1", {"data": "test1"})
        graph_storage.add_node("node1", {"data": "test2"})

        # Should update existing node
        assert graph_storage.node_count() == 1

    def test_add_edge(self, graph_storage, mock_gap):
        """Test adding an edge"""
        gap1 = mock_gap
        gap2 = KnowledgeGap(
            type="test", domain="test2", priority=0.5, estimated_cost=10.0
        )

        graph_storage.add_node("gap1", gap1)
        graph_storage.add_node("gap2", gap2)

        edge = DependencyEdge(
            source=gap1, target=gap2, dependency_type=DependencyType.PREREQUISITE
        )

        result = graph_storage.add_edge("gap1", "gap2", edge)

        assert result is True
        assert graph_storage.edge_count() == 1
        assert ("gap1", "gap2") in graph_storage.edges

    def test_remove_edge(self, graph_storage, mock_gap):
        """Test removing an edge"""
        gap1 = mock_gap
        gap2 = KnowledgeGap(
            type="test", domain="test2", priority=0.5, estimated_cost=10.0
        )

        graph_storage.add_node("gap1", gap1)
        graph_storage.add_node("gap2", gap2)

        edge = DependencyEdge(
            source=gap1, target=gap2, dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.remove_edge("gap1", "gap2")

        assert graph_storage.edge_count() == 0
        assert ("gap1", "gap2") not in graph_storage.edges

    def test_remove_node(self, graph_storage, mock_gap):
        """Test removing a node"""
        gap1 = mock_gap
        gap2 = KnowledgeGap(
            type="test", domain="test2", priority=0.5, estimated_cost=10.0
        )

        graph_storage.add_node("gap1", gap1)
        graph_storage.add_node("gap2", gap2)

        edge = DependencyEdge(
            source=gap1, target=gap2, dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)

        # Remove node should also remove its edges
        graph_storage.remove_node("gap1")

        assert graph_storage.node_count() == 1
        assert graph_storage.edge_count() == 0

    def test_get_children(self, graph_storage, mock_gap):
        """Test getting children of a node"""
        graph_storage.add_node("gap1", mock_gap)
        graph_storage.add_node("gap2", mock_gap)
        graph_storage.add_node("gap3", mock_gap)

        edge1 = DependencyEdge(
            source=mock_gap,
            target=mock_gap,
            dependency_type=DependencyType.PREREQUISITE,
        )

        graph_storage.add_edge("gap1", "gap2", edge1)
        graph_storage.add_edge("gap1", "gap3", edge1)

        children = graph_storage.get_children("gap1")

        assert len(children) == 2
        assert "gap2" in children
        assert "gap3" in children

    def test_get_parents(self, graph_storage, mock_gap):
        """Test getting parents of a node"""
        graph_storage.add_node("gap1", mock_gap)
        graph_storage.add_node("gap2", mock_gap)
        graph_storage.add_node("gap3", mock_gap)

        edge = DependencyEdge(
            source=mock_gap,
            target=mock_gap,
            dependency_type=DependencyType.PREREQUISITE,
        )

        graph_storage.add_edge("gap1", "gap3", edge)
        graph_storage.add_edge("gap2", "gap3", edge)

        parents = graph_storage.get_parents("gap3")

        assert len(parents) == 2
        assert "gap1" in parents
        assert "gap2" in parents

    def test_max_nodes_limit(self):
        """Test max nodes limit enforcement"""
        storage = GraphStorage(max_nodes=3, max_edges=10)

        assert storage.add_node("node1", {"data": "test1"})
        assert storage.add_node("node2", {"data": "test2"})
        assert storage.add_node("node3", {"data": "test3"})

        # Fourth node should fail
        result = storage.add_node("node4", {"data": "test4"})
        assert result is False
        assert storage.node_count() == 3

    def test_max_edges_limit(self, graph_storage):
        """Test max edges limit enforcement"""
        storage = GraphStorage(max_nodes=10, max_edges=2)

        storage.add_node("gap1", {"data": "test"})
        storage.add_node("gap2", {"data": "test"})
        storage.add_node("gap3", {"data": "test"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        assert storage.add_edge("gap1", "gap2", edge)
        assert storage.add_edge("gap2", "gap3", edge)

        # Third edge should fail
        result = storage.add_edge("gap1", "gap3", edge)
        assert result is False
        assert storage.edge_count() == 2

    def test_thread_safety(self, graph_storage):
        """Test thread safety of graph storage"""
        errors = []

        def add_nodes(thread_id):
            try:
                for i in range(10):
                    graph_storage.add_node(f"node_{thread_id}_{i}", {"data": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Test PathFinder
# ============================================================================


class TestPathFinder:
    """Tests for PathFinder class"""

    def test_has_path_direct(self, graph_storage):
        """Test direct path detection"""
        graph_storage.add_node("gap1", {"data": "test"})
        graph_storage.add_node("gap2", {"data": "test"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)

        path_finder = PathFinder(graph_storage)

        assert path_finder.has_path("gap1", "gap2") is True
        assert path_finder.has_path("gap2", "gap1") is False

    def test_has_path_indirect(self, graph_storage):
        """Test indirect path detection"""
        graph_storage.add_node("gap1", {"data": "test"})
        graph_storage.add_node("gap2", {"data": "test"})
        graph_storage.add_node("gap3", {"data": "test"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap3", edge)

        path_finder = PathFinder(graph_storage)

        assert path_finder.has_path("gap1", "gap3") is True
        assert path_finder.has_path("gap3", "gap1") is False

    def test_get_descendants(self, graph_storage):
        """Test getting all descendants"""
        # Create chain: gap1 -> gap2 -> gap3
        for i in range(1, 4):
            graph_storage.add_node(f"gap{i}", {"data": f"test{i}"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap3", edge)

        path_finder = PathFinder(graph_storage)
        descendants = path_finder.get_descendants("gap1")

        assert len(descendants) == 2
        assert "gap2" in descendants
        assert "gap3" in descendants

    def test_get_ancestors(self, graph_storage):
        """Test getting all ancestors"""
        # Create chain: gap1 -> gap2 -> gap3
        for i in range(1, 4):
            graph_storage.add_node(f"gap{i}", {"data": f"test{i}"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap3", edge)

        path_finder = PathFinder(graph_storage)
        ancestors = path_finder.get_ancestors("gap3")

        assert len(ancestors) == 2
        assert "gap1" in ancestors
        assert "gap2" in ancestors


# ============================================================================
# Test CycleDetector
# ============================================================================


class TestCycleDetector:
    """Tests for CycleDetector class"""

    def test_would_create_cycle_simple(self, graph_storage):
        """Test cycle detection for simple case"""
        graph_storage.add_node("gap1", {"data": "test"})
        graph_storage.add_node("gap2", {"data": "test"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)

        detector = CycleDetector(graph_storage)

        # Adding reverse edge would create cycle
        assert detector.would_create_cycle("gap2", "gap1") is True
        assert detector.would_create_cycle("gap1", "gap2") is False

    def test_would_create_cycle_complex(self, graph_storage):
        """Test cycle detection for complex case"""
        # Create chain: gap1 -> gap2 -> gap3
        for i in range(1, 4):
            graph_storage.add_node(f"gap{i}", {"data": f"test{i}"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap3", edge)

        detector = CycleDetector(graph_storage)

        # Adding gap3 -> gap1 would create cycle
        assert detector.would_create_cycle("gap3", "gap1") is True

    def test_self_cycle(self, graph_storage):
        """Test self-cycle detection"""
        graph_storage.add_node("gap1", {"data": "test"})

        detector = CycleDetector(graph_storage)

        # Node pointing to itself is a cycle
        assert detector.would_create_cycle("gap1", "gap1") is True

    def test_has_cycles_empty(self, graph_storage):
        """Test cycle detection on empty graph"""
        detector = CycleDetector(graph_storage)

        assert detector.has_cycles() is False

    def test_has_cycles_with_cycle(self, graph_storage):
        """Test cycle detection on graph with cycle"""
        graph_storage.add_node("gap1", {"data": "test"})
        graph_storage.add_node("gap2", {"data": "test"})

        edge = DependencyEdge(
            source="gap1", target="gap2", dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap1", edge)

        detector = CycleDetector(graph_storage)

        assert detector.has_cycles() is True

    def test_break_cycles(self, graph_storage):
        """Test breaking cycles"""
        # Create cycle: gap1 -> gap2 -> gap1
        graph_storage.add_node("gap1", {"data": "test"})
        graph_storage.add_node("gap2", {"data": "test"})

        weak_edge = DependencyEdge(
            source="gap1",
            target="gap2",
            dependency_type=DependencyType.WEAK,
            strength=1.0,
        )

        strong_edge = DependencyEdge(
            source="gap2",
            target="gap1",
            dependency_type=DependencyType.CAUSAL,
            strength=1.0,
        )

        graph_storage.add_edge("gap1", "gap2", weak_edge)
        graph_storage.add_edge("gap2", "gap1", strong_edge)

        detector = CycleDetector(graph_storage)

        assert detector.has_cycles() is True

        removed = detector.break_cycles_minimum_cost()

        # Should remove weak edge (lower cost)
        assert len(removed) > 0
        assert detector.has_cycles() is False


# ============================================================================
# Test TopologicalSorter
# ============================================================================


class TestTopologicalSorter:
    """Tests for TopologicalSorter class"""

    def test_topological_sort_simple(self, graph_storage):
        """Test topological sort on simple graph"""
        gap1 = {"id": "gap1", "data": "test1"}
        gap2 = {"id": "gap2", "data": "test2"}
        gap3 = {"id": "gap3", "data": "test3"}

        graph_storage.add_node("gap1", gap1)
        graph_storage.add_node("gap2", gap2)
        graph_storage.add_node("gap3", gap3)

        edge = DependencyEdge(
            source=gap1, target=gap2, dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap3", edge)

        sorter = TopologicalSorter(graph_storage)
        sorted_nodes = sorter.topological_sort()

        assert len(sorted_nodes) == 3

        # gap1 should come before gap2, gap2 before gap3
        gap1_idx = sorted_nodes.index(gap1)
        gap2_idx = sorted_nodes.index(gap2)
        gap3_idx = sorted_nodes.index(gap3)

        assert gap1_idx < gap2_idx < gap3_idx

    def test_topological_sort_with_cycle(self, graph_storage):
        """Test topological sort on graph with cycle"""
        gap1 = {"id": "gap1"}
        gap2 = {"id": "gap2"}

        graph_storage.add_node("gap1", gap1)
        graph_storage.add_node("gap2", gap2)

        edge = DependencyEdge(
            source=gap1, target=gap2, dependency_type=DependencyType.PREREQUISITE
        )

        graph_storage.add_edge("gap1", "gap2", edge)
        graph_storage.add_edge("gap2", "gap1", edge)

        sorter = TopologicalSorter(graph_storage)

        # Should raise error or return empty on cycle
        with pytest.raises(ValueError):
            sorter.topological_sort()


# ============================================================================
# Test CacheManager
# ============================================================================


class TestCacheManager:
    """Tests for CacheManager class"""

    def test_path_cache(self):
        """Test path caching"""
        cache = CacheManager(cache_ttl=60)

        # Put value in cache
        cache.put_path(("gap1", "gap2"), True)

        # Retrieve from cache
        result = cache.get_path(("gap1", "gap2"))

        assert result is True
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0

    def test_descendants_cache(self):
        """Test descendants caching"""
        cache = CacheManager(cache_ttl=60)

        descendants = {"gap2", "gap3", "gap4"}
        cache.put_descendants("gap1", descendants)

        result = cache.get_descendants("gap1")

        assert result == descendants
        assert cache.cache_hits == 1

    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = CacheManager(cache_ttl=0.1)  # 100ms TTL

        cache.put_path(("gap1", "gap2"), True)

        # Immediate retrieval should work
        result = cache.get_path(("gap1", "gap2"))
        assert result is True

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        result = cache.get_path(("gap1", "gap2"))
        assert result is None
        assert cache.cache_misses == 1

    def test_cache_invalidation(self):
        """Test cache invalidation"""
        cache = CacheManager(cache_ttl=60)

        cache.put_path(("gap1", "gap2"), True)
        cache.put_descendants("gap1", {"gap2"})

        cache.invalidate_all()

        # Both should be invalidated
        assert cache.get_path(("gap1", "gap2")) is None
        assert cache.get_descendants("gap1") is None

    def test_cache_statistics(self):
        """Test cache statistics"""
        cache = CacheManager(cache_ttl=60)

        cache.put_path(("gap1", "gap2"), True)
        cache.get_path(("gap1", "gap2"))  # Hit
        cache.get_path(("gap1", "gap3"))  # Miss

        stats = cache.get_statistics()

        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == 0.5


# ============================================================================
# Test CycleAwareDependencyGraph
# ============================================================================


class TestCycleAwareDependencyGraph:
    """Tests for CycleAwareDependencyGraph class"""

    def test_initialization(self, dependency_graph):
        """Test graph initialization"""
        assert dependency_graph is not None
        assert dependency_graph.storage is not None
        assert dependency_graph.cache is not None

    def test_add_node(self, dependency_graph, mock_gap):
        """Test adding a node to graph"""
        node_id = dependency_graph.add_node(mock_gap)

        assert node_id is not None
        assert node_id in dependency_graph.nodes

    def test_add_edge_without_cycle(self, dependency_graph):
        """Test adding edge that doesn't create cycle"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)

        result = dependency_graph.add_edge(gap1, gap2)

        assert result is True

    def test_add_edge_with_cycle(self, dependency_graph):
        """Test that adding edge that creates cycle is prevented"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)

        # Add edge gap1 -> gap2
        dependency_graph.add_edge(gap1, gap2)

        # Try to add edge gap2 -> gap1 (would create cycle)
        result = dependency_graph.add_edge(gap2, gap1)

        assert result is False

    def test_add_weak_edge(self, dependency_graph):
        """Test adding weak edge"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)

        result = dependency_graph.add_weak_edge(gap1, gap2)

        assert result is True

    def test_descendants(self, dependency_graph):
        """Test getting descendants"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )
        gap3 = KnowledgeGap(
            type="test3", domain="d3", priority=0.6, estimated_cost=12.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)
        dependency_graph.add_node(gap3)

        dependency_graph.add_edge(gap1, gap2)
        dependency_graph.add_edge(gap2, gap3)

        descendants = dependency_graph.descendants(gap1)

        # Should have gap2 and gap3 as descendants
        assert len(descendants) >= 1

    def test_ancestors(self, dependency_graph):
        """Test getting ancestors"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )
        gap3 = KnowledgeGap(
            type="test3", domain="d3", priority=0.6, estimated_cost=12.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)
        dependency_graph.add_node(gap3)

        dependency_graph.add_edge(gap1, gap2)
        dependency_graph.add_edge(gap2, gap3)

        ancestors = dependency_graph.ancestors(gap3)

        # Should have gap1 and gap2 as ancestors
        assert len(ancestors) >= 1

    def test_topological_sort(self, dependency_graph):
        """Test topological sorting"""
        gaps = []
        for i in range(3):
            gap = KnowledgeGap(
                type=f"test{i}",
                domain=f"d{i}",
                priority=0.8 - i * 0.1,
                estimated_cost=10.0 + i * 5,
            )
            gaps.append(gap)
            dependency_graph.add_node(gap)

        dependency_graph.add_edge(gaps[0], gaps[1])
        dependency_graph.add_edge(gaps[1], gaps[2])

        sorted_gaps = dependency_graph.topological_sort()

        # Should return all gaps in dependency order
        assert len(sorted_gaps) == 3

    def test_break_cycles(self, dependency_graph):
        """Test cycle breaking"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)

        # Add weak edges to create cycle
        dependency_graph.add_weak_edge(gap1, gap2)
        dependency_graph.add_weak_edge(gap2, gap1)

        # Should have cycle
        assert dependency_graph.has_cycles() is True

        # Break cycles
        removed = dependency_graph.break_cycles_minimum_cost()

        # Should break the cycle
        assert len(removed) > 0
        assert dependency_graph.has_cycles() is False

    def test_get_statistics(self, dependency_graph):
        """Test getting graph statistics"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)
        dependency_graph.add_edge(gap1, gap2)

        stats = dependency_graph.get_statistics()

        assert "nodes" in stats
        assert "edges" in stats
        assert stats["nodes"] >= 2
        assert stats["edges"] >= 1

    def test_thread_safety(self, dependency_graph):
        """Test thread safety of graph operations"""
        errors = []

        def add_gaps_and_edges(thread_id):
            try:
                gaps = []
                for i in range(5):
                    gap = KnowledgeGap(
                        type=f"test_{thread_id}_{i}",
                        domain=f"domain_{thread_id}",
                        priority=0.8,
                        estimated_cost=10.0,
                    )
                    gaps.append(gap)
                    dependency_graph.add_node(gap)

                # Add some edges
                for i in range(len(gaps) - 1):
                    dependency_graph.add_edge(gaps[i], gaps[i + 1])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_gaps_and_edges, args=(i,)) for i in range(3):
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Test DependencyAnalyzer
# ============================================================================


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer class"""

    def test_initialization(self, dependency_analyzer):
        """Test analyzer initialization"""
        assert dependency_analyzer is not None

    def test_find_dependencies_decomposition(self, dependency_analyzer):
        """Test finding dependencies for decomposition gap"""
        gap = KnowledgeGap(
            type="decomposition", domain="test", priority=0.8, estimated_cost=10.0
        )

        # Add missing concepts
        gap.missing_concepts = ["concept1", "concept2"]

        dependencies = dependency_analyzer.find_dependencies(gap)

        # Should find semantic dependencies
        assert len(dependencies) >= 2

    def test_find_dependencies_causal(self, dependency_analyzer):
        """Test finding dependencies for causal gap"""
        gap = KnowledgeGap(
            type="causal", domain="test", priority=0.8, estimated_cost=10.0
        )

        gap.variables = ["var1", "var2", "var3"]

        dependencies = dependency_analyzer.find_dependencies(gap)

        # Should find correlation dependencies
        assert len(dependencies) >= 2

    def test_find_dependencies_transfer(self, dependency_analyzer):
        """Test finding dependencies for transfer gap"""
        gap = KnowledgeGap(
            type="transfer", domain="test", priority=0.8, estimated_cost=10.0
        )

        gap.source_domain = "domain1"
        gap.target_domain = "domain2"

        dependencies = dependency_analyzer.find_dependencies(gap)

        # Should find domain bridge dependency
        assert len(dependencies) >= 1

    def test_identify_root_causes(self, dependency_analyzer, dependency_graph):
        """Test identifying root causes"""
        # Create gaps with dependencies
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.9, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.8, estimated_cost=15.0
        )
        gap3 = KnowledgeGap(
            type="test3", domain="d3", priority=0.7, estimated_cost=12.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)
        dependency_graph.add_node(gap3)

        # gap1 is root (no dependencies)
        # gap2 depends on gap1
        # gap3 depends on gap2
        dependency_graph.add_edge(gap2, gap1)
        dependency_graph.add_edge(gap3, gap2)

        root_causes = dependency_analyzer.identify_root_causes(dependency_graph)

        # gap1 should be identified as root cause
        assert len(root_causes) >= 1

    def test_calculate_unlock_potential(self, dependency_analyzer, dependency_graph):
        """Test calculating unlock potential"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.9, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.8, estimated_cost=15.0
        )
        gap3 = KnowledgeGap(
            type="test3", domain="d3", priority=0.7, estimated_cost=12.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)
        dependency_graph.add_node(gap3)

        dependency_graph.add_edge(gap1, gap2)
        dependency_graph.add_edge(gap1, gap3)

        potential = dependency_analyzer.calculate_unlock_potential(
            gap1, dependency_graph
        )

        # Should have non-zero potential
        assert potential > 0


# ============================================================================
# Test ROICalculator
# ============================================================================


class TestROICalculator:
    """Tests for ROICalculator class"""

    def test_initialization(self, roi_calculator):
        """Test ROI calculator initialization"""
        assert roi_calculator is not None
        assert roi_calculator.unlock_weight > 0
        assert roi_calculator.dependency_weight > 0

    def test_calculate_base_roi(self, roi_calculator):
        """Test base ROI calculation"""
        gap = KnowledgeGap(
            type="decomposition", domain="test", priority=0.8, estimated_cost=10.0
        )

        base_roi = roi_calculator.calculate_base_roi(gap)

        assert base_roi > 0
        assert base_roi == pytest.approx(
            0.8 / 10.0 * 1.1, rel=0.01
        )  # 1.1 is decomposition bonus

    def test_apply_unlock_bonus(self, roi_calculator):
        """Test applying unlock bonus"""
        gap = KnowledgeGap(
            type="test", domain="test", priority=0.8, estimated_cost=10.0
        )

        roi_with_bonus = roi_calculator.apply_unlock_bonus(gap, descendants_count=5)
        base_roi = roi_calculator.calculate_base_roi(gap)

        # ROI with bonus should be higher
        assert roi_with_bonus > base_roi

    def test_apply_dependency_penalty(self, roi_calculator):
        """Test applying dependency penalty"""
        gap = KnowledgeGap(
            type="test", domain="test", priority=0.8, estimated_cost=10.0
        )

        roi_with_penalty = roi_calculator.apply_dependency_penalty(
            gap, ancestors_count=5
        )
        base_roi = roi_calculator.calculate_base_roi(gap)

        # ROI with penalty should be lower
        assert roi_with_penalty < base_roi

    def test_get_adjusted_roi(self, roi_calculator, dependency_graph):
        """Test getting adjusted ROI"""
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.9, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.8, estimated_cost=15.0
        )
        gap3 = KnowledgeGap(
            type="test3", domain="d3", priority=0.7, estimated_cost=12.0
        )

        dependency_graph.add_node(gap1)
        dependency_graph.add_node(gap2)
        dependency_graph.add_node(gap3)

        dependency_graph.add_edge(gap1, gap2)
        dependency_graph.add_edge(gap1, gap3)

        roi = roi_calculator.get_adjusted_roi(gap1, dependency_graph)

        # Should return positive ROI
        assert roi > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_graph_workflow(self):
        """Test complete workflow with graph operations"""
        graph = CycleAwareDependencyGraph(max_nodes=100, max_edges=200)

        # Create gaps
        gaps = []
        for i in range(5):
            gap = KnowledgeGap(
                type=f"test{i}",
                domain=f"domain{i}",
                priority=0.9 - i * 0.1,
                estimated_cost=10.0 + i * 5,
            )
            gaps.append(gap)
            graph.add_node(gap)

        # Add dependencies
        for i in range(len(gaps) - 1):
            graph.add_edge(gaps[i], gaps[i + 1])

        # Get topological sort
        sorted_gaps = graph.topological_sort()

        # Get statistics
        stats = graph.get_statistics()

        assert len(sorted_gaps) == 5
        assert stats["nodes"] == 5
        assert stats["edges"] == 4

    def test_cycle_detection_and_breaking(self):
        """Test detecting and breaking cycles"""
        graph = CycleAwareDependencyGraph(max_nodes=100, max_edges=200)

        # Create cycle
        gap1 = KnowledgeGap(
            type="test1", domain="d1", priority=0.8, estimated_cost=10.0
        )
        gap2 = KnowledgeGap(
            type="test2", domain="d2", priority=0.7, estimated_cost=15.0
        )

        graph.add_node(gap1)
        graph.add_node(gap2)

        graph.add_weak_edge(gap1, gap2)
        graph.add_weak_edge(gap2, gap1)

        # Detect cycle
        assert graph.has_cycles() is True

        # Break cycle
        removed = graph.break_cycles_minimum_cost()

        # Verify cycle is broken
        assert len(removed) > 0
        assert graph.has_cycles() is False

        # Should now be able to sort
        sorted_gaps = graph.topological_sort()
        assert len(sorted_gaps) == 2


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance tests"""

    def test_large_graph_performance(self):
        """Test performance with large graph"""
        graph = CycleAwareDependencyGraph(max_nodes=1000, max_edges=5000)

        start_time = time.time()

        # Add many nodes
        gaps = []
        for i in range(100):
            gap = KnowledgeGap(
                type=f"test{i}",
                domain=f"domain{i % 10}",
                priority=0.8,
                estimated_cost=10.0,
            )
            gaps.append(gap)
            graph.add_node(gap)

        # Add edges
        for i in range(len(gaps) - 1):
            if i % 10 != 9:  # Avoid some edges to prevent too dense graph
                graph.add_edge(gaps[i], gaps[i + 1])

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 10.0

    def test_cache_effectiveness(self):
        """Test cache effectiveness"""
        graph = CycleAwareDependencyGraph(max_nodes=100, max_edges=200)

        gaps = []
        for i in range(10):
            gap = KnowledgeGap(
                type=f"test{i}", domain=f"domain{i}", priority=0.8, estimated_cost=10.0
            )
            gaps.append(gap)
            graph.add_node(gap)

        # Add edges
        for i in range(len(gaps) - 1):
            graph.add_edge(gaps[i], gaps[i + 1])

        # First call - should populate cache
        descendants1 = graph.descendants(gaps[0])

        # Second call - should use cache
        descendants2 = graph.descendants(gaps[0])

        assert descendants1 == descendants2

        # Check cache statistics
        stats = graph.get_statistics()
        assert stats["cache_hits"] > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
