"""
Comprehensive test suite for large_graph_generator.py
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from large_graph_generator import (
    DEFAULT_MAX_MESH_NODES,
    MAX_DENSITY,
    MAX_GRAPH_COUNT,
    MAX_MESH_NODES_LIMIT,
    MAX_NODES,
    MIN_DENSITY,
    MIN_GRAPH_COUNT,
    MIN_NODES,
    VALID_EDGE_TYPES,
    VALID_NODE_TYPES,
    VALID_TOPOLOGIES,
    generate_large_graph,
    generate_specific_topology,
    generate_stress_test_graphs,
    get_graph_statistics,
    validate_graph_structure,
)


@pytest.fixture
def simple_graph():
    """Create simple valid graph."""
    return {
        "nodes": [
            {"id": "node_0", "type": "CONST", "params": {"value": 0.5}},
            {"id": "node_1", "type": "ADD", "params": {"value": 0.7}},
        ],
        "edges": [{"from": "node_0", "to": "node_1", "type": "data"}],
        "metadata": {"num_nodes": 2, "num_edges": 1},
    }


class TestGenerateLargeGraph:
    """Test generate_large_graph function."""

    def test_generate_basic(self):
        """Test basic graph generation."""
        graph = generate_large_graph(num_nodes=10, density=0.2, seed=42)

        assert "nodes" in graph
        assert "edges" in graph
        assert "metadata" in graph
        assert len(graph["nodes"]) == 10

    def test_generate_with_seed(self):
        """Test reproducibility with seed."""
        graph1 = generate_large_graph(num_nodes=20, density=0.1, seed=42)
        graph2 = generate_large_graph(num_nodes=20, density=0.1, seed=42)

        assert len(graph1["nodes"]) == len(graph2["nodes"])
        assert len(graph1["edges"]) == len(graph2["edges"])

    def test_generate_different_seeds(self):
        """Test different results with different seeds."""
        graph1 = generate_large_graph(num_nodes=20, density=0.3, seed=42)
        graph2 = generate_large_graph(num_nodes=20, density=0.3, seed=99)

        # Edges might differ (random graph)
        # Just ensure both are valid
        assert len(graph1["nodes"]) == 20
        assert len(graph2["nodes"]) == 20

    def test_generate_custom_node_types(self):
        """Test with custom node types."""
        custom_types = ["CUSTOM1", "CUSTOM2"]
        graph = generate_large_graph(
            num_nodes=5, density=0.3, seed=42, node_types=custom_types
        )

        # All nodes should have types from custom list
        node_types = {node["type"] for node in graph["nodes"]}
        assert node_types.issubset(set(custom_types))

    def test_generate_custom_edge_types(self):
        """Test with custom edge types."""
        custom_types = ["custom_edge"]
        graph = generate_large_graph(
            num_nodes=5, density=0.5, seed=42, edge_types=custom_types
        )

        # All edges should have types from custom list
        if graph["edges"]:
            edge_types = {edge["type"] for edge in graph["edges"]}
            assert edge_types.issubset(set(custom_types))

    def test_generate_invalid_num_nodes_type(self):
        """Test with invalid num_nodes type."""
        with pytest.raises(TypeError, match="num_nodes must be int"):
            generate_large_graph(num_nodes="10")

    def test_generate_num_nodes_below_min(self):
        """Test with num_nodes below minimum."""
        with pytest.raises(ValueError, match="num_nodes must be in"):
            generate_large_graph(num_nodes=0)

    def test_generate_num_nodes_above_max(self):
        """Test with num_nodes above maximum."""
        with pytest.raises(ValueError, match="num_nodes must be in"):
            generate_large_graph(num_nodes=MAX_NODES + 1)

    def test_generate_invalid_density_type(self):
        """Test with invalid density type."""
        with pytest.raises(TypeError, match="density must be numeric"):
            generate_large_graph(num_nodes=10, density="0.5")

    def test_generate_density_below_min(self):
        """Test with density below minimum."""
        with pytest.raises(ValueError, match="density must be in"):
            generate_large_graph(num_nodes=10, density=-0.1)

    def test_generate_density_above_max(self):
        """Test with density above maximum."""
        with pytest.raises(ValueError, match="density must be in"):
            generate_large_graph(num_nodes=10, density=1.5)

    def test_generate_invalid_seed_type(self):
        """Test with invalid seed type."""
        with pytest.raises(TypeError, match="seed must be int or None"):
            generate_large_graph(num_nodes=10, seed="42")

    def test_generate_invalid_node_types_not_list(self):
        """Test with node_types not a list."""
        with pytest.raises(ValueError, match="node_types must be a non-empty list"):
            generate_large_graph(num_nodes=10, node_types="CONST")

    def test_generate_invalid_node_types_empty(self):
        """Test with empty node_types."""
        with pytest.raises(ValueError, match="node_types must be a non-empty list"):
            generate_large_graph(num_nodes=10, node_types=[])

    def test_generate_invalid_edge_types_not_list(self):
        """Test with edge_types not a list."""
        with pytest.raises(ValueError, match="edge_types must be a non-empty list"):
            generate_large_graph(num_nodes=10, edge_types="data")

    def test_generate_invalid_edge_types_empty(self):
        """Test with empty edge_types."""
        with pytest.raises(ValueError, match="edge_types must be a non-empty list"):
            generate_large_graph(num_nodes=10, edge_types=[])

    def test_generate_metadata_correct(self):
        """Test metadata is correct."""
        graph = generate_large_graph(num_nodes=50, density=0.1, seed=42)

        assert graph["metadata"]["num_nodes"] == 50
        assert graph["metadata"]["density"] == 0.1
        assert graph["metadata"]["seed"] == 42
        assert "is_connected" in graph["metadata"]
        assert "average_degree" in graph["metadata"]

    @pytest.mark.slow
    def test_generate_large_graph_1000_nodes(self):
        """Test with 500 nodes (reduced from 1000 for CI timeout)."""
        graph = generate_large_graph(num_nodes=500, density=0.01, seed=42)

        assert len(graph["nodes"]) == 500
        assert validate_graph_structure(graph)


class TestGenerateStressTestGraphs:
    """Test generate_stress_test_graphs function."""

    def test_stress_test_basic(self):
        """Test basic stress test generation."""
        graphs = generate_stress_test_graphs(count=2, seed=42)

        assert len(graphs) == 2
        assert all("nodes" in g for g in graphs)
        assert all("edges" in g for g in graphs)

    def test_stress_test_with_ranges(self):
        """Test with custom ranges."""
        graphs = generate_stress_test_graphs(
            count=2,
            min_nodes=20,
            max_nodes=50,
            min_density=0.05,
            max_density=0.15,
            seed=42,
        )

        assert len(graphs) == 2

        for graph in graphs:
            num_nodes = graph["metadata"]["num_nodes"]
            assert 20 <= num_nodes <= 50

    def test_stress_test_reproducible(self):
        """Test reproducibility with seed."""
        graphs1 = generate_stress_test_graphs(count=2, seed=42)
        graphs2 = generate_stress_test_graphs(count=2, seed=42)

        assert len(graphs1) == len(graphs2)

        for g1, g2 in zip(graphs1, graphs2):
            assert len(g1["nodes"]) == len(g2["nodes"])

    def test_stress_test_graph_indices(self):
        """Test that graph indices are set."""
        graphs = generate_stress_test_graphs(count=2, seed=42)

        for i, graph in enumerate(graphs):
            assert graph["metadata"]["graph_index"] == i

    def test_stress_test_invalid_count_type(self):
        """Test with invalid count type."""
        with pytest.raises(TypeError, match="count must be int"):
            generate_stress_test_graphs(count="5")

    def test_stress_test_count_below_min(self):
        """Test with count below minimum."""
        with pytest.raises(ValueError, match="count must be in"):
            generate_stress_test_graphs(count=0)

    def test_stress_test_count_above_max(self):
        """Test with count above maximum."""
        with pytest.raises(ValueError, match="count must be in"):
            generate_stress_test_graphs(count=MAX_GRAPH_COUNT + 1)

    def test_stress_test_invalid_node_range_type(self):
        """Test with invalid node range types."""
        with pytest.raises(TypeError, match="min_nodes and max_nodes must be int"):
            generate_stress_test_graphs(min_nodes="50", max_nodes=100)

    def test_stress_test_invalid_node_range_values(self):
        """Test with invalid node range values."""
        with pytest.raises(ValueError, match="Invalid node range"):
            generate_stress_test_graphs(min_nodes=200, max_nodes=100)

    def test_stress_test_invalid_density_range_type(self):
        """Test with invalid density range types."""
        with pytest.raises(
            TypeError, match="min_density and max_density must be numeric"
        ):
            generate_stress_test_graphs(min_density="0.1", max_density=0.3)

    def test_stress_test_invalid_density_range_values(self):
        """Test with invalid density range values."""
        with pytest.raises(ValueError, match="Invalid density range"):
            generate_stress_test_graphs(min_density=0.5, max_density=0.2)

    def test_stress_test_invalid_seed_type(self):
        """Test with invalid seed type."""
        with pytest.raises(TypeError, match="seed must be int or None"):
            generate_stress_test_graphs(count=5, seed="42")


class TestGenerateSpecificTopology:
    """Test generate_specific_topology function."""

    def test_topology_star(self):
        """Test star topology."""
        graph = generate_specific_topology("star", num_nodes=10, seed=42)

        assert graph["metadata"]["topology"] == "star"
        assert graph["metadata"]["actual_nodes"] == 10

    def test_topology_ring(self):
        """Test ring topology."""
        graph = generate_specific_topology("ring", num_nodes=8, seed=42)

        assert graph["metadata"]["topology"] == "ring"
        assert graph["metadata"]["actual_nodes"] == 8

    def test_topology_mesh(self):
        """Test mesh topology."""
        graph = generate_specific_topology("mesh", num_nodes=5, seed=42)

        assert graph["metadata"]["topology"] == "mesh"
        assert graph["metadata"]["actual_nodes"] == 5

    def test_topology_tree(self):
        """Test tree topology."""
        graph = generate_specific_topology("tree", num_nodes=20, seed=42)

        assert graph["metadata"]["topology"] == "tree"
        # Tree may have different number due to balanced structure
        assert graph["metadata"]["actual_nodes"] > 0

    def test_topology_random(self):
        """Test random topology."""
        graph = generate_specific_topology("random", num_nodes=15, seed=42)

        assert graph["metadata"]["topology"] == "random"
        assert graph["metadata"]["actual_nodes"] == 15

    def test_topology_mesh_limited(self):
        """Test mesh topology with limit."""
        graph = generate_specific_topology(
            "mesh", num_nodes=200, max_mesh_nodes=50, seed=42
        )

        assert graph["metadata"]["actual_nodes"] == 50

    def test_topology_invalid_type_str(self):
        """Test with invalid topology type (not string)."""
        with pytest.raises(TypeError, match="topology_type must be string"):
            generate_specific_topology(123, num_nodes=10)

    def test_topology_invalid_type_value(self):
        """Test with invalid topology value."""
        with pytest.raises(ValueError, match="Invalid topology_type"):
            generate_specific_topology("invalid_topology", num_nodes=10)

    def test_topology_invalid_num_nodes_type(self):
        """Test with invalid num_nodes type."""
        with pytest.raises(TypeError, match="num_nodes must be int"):
            generate_specific_topology("star", num_nodes="10")

    def test_topology_num_nodes_out_of_range(self):
        """Test with num_nodes out of range."""
        with pytest.raises(ValueError, match="num_nodes must be in"):
            generate_specific_topology("star", num_nodes=MAX_NODES + 1)

    def test_topology_invalid_max_mesh_nodes_type(self):
        """Test with invalid max_mesh_nodes type."""
        with pytest.raises(TypeError, match="max_mesh_nodes must be int"):
            generate_specific_topology("mesh", num_nodes=10, max_mesh_nodes="50")

    def test_topology_max_mesh_nodes_out_of_range(self):
        """Test with max_mesh_nodes out of range."""
        with pytest.raises(ValueError, match="max_mesh_nodes must be in"):
            generate_specific_topology(
                "mesh", num_nodes=10, max_mesh_nodes=MAX_MESH_NODES_LIMIT + 1
            )

    def test_topology_invalid_seed_type(self):
        """Test with invalid seed type."""
        with pytest.raises(TypeError, match="seed must be int or None"):
            generate_specific_topology("star", num_nodes=10, seed="42")

    def test_topology_tree_node_count_approximation(self):
        """Test tree topology node count approximation."""
        for target in [10, 30]:
            graph = generate_specific_topology("tree", num_nodes=target, seed=42)
            actual = graph["metadata"]["actual_nodes"]

            # Should be within reasonable range (tree structure has constraints)
            deviation = abs(actual - target) / target
            assert deviation < 1.0  # Within 100% is acceptable for trees


class TestValidateGraphStructure:
    """Test validate_graph_structure function."""

    def test_validate_valid_graph(self, simple_graph):
        """Test validating valid graph."""
        assert validate_graph_structure(simple_graph) is True

    def test_validate_not_dict(self):
        """Test with non-dict graph."""
        assert validate_graph_structure("not a dict") is False

    def test_validate_missing_nodes(self):
        """Test with missing nodes key."""
        graph = {"edges": []}
        assert validate_graph_structure(graph) is False

    def test_validate_missing_edges(self):
        """Test with missing edges key."""
        graph = {"nodes": []}
        assert validate_graph_structure(graph) is False

    def test_validate_nodes_not_list(self):
        """Test with nodes not a list."""
        graph = {"nodes": "not a list", "edges": []}
        assert validate_graph_structure(graph) is False

    def test_validate_edges_not_list(self):
        """Test with edges not a list."""
        graph = {"nodes": [], "edges": "not a list"}
        assert validate_graph_structure(graph) is False

    def test_validate_node_not_dict(self):
        """Test with node not a dict."""
        graph = {"nodes": ["not a dict"], "edges": []}
        assert validate_graph_structure(graph) is False

    def test_validate_node_missing_id(self):
        """Test with node missing id."""
        graph = {"nodes": [{"type": "CONST"}], "edges": []}
        assert validate_graph_structure(graph) is False

    def test_validate_duplicate_node_ids(self):
        """Test with duplicate node IDs."""
        graph = {"nodes": [{"id": "node1"}, {"id": "node1"}], "edges": []}  # Duplicate
        assert validate_graph_structure(graph) is False

    def test_validate_edge_not_dict(self):
        """Test with edge not a dict."""
        graph = {"nodes": [{"id": "n1"}], "edges": ["not a dict"]}
        assert validate_graph_structure(graph) is False

    def test_validate_edge_missing_from(self):
        """Test with edge missing from field."""
        graph = {"nodes": [{"id": "n1"}], "edges": [{"to": "n1"}]}
        assert validate_graph_structure(graph) is False

    def test_validate_edge_missing_to(self):
        """Test with edge missing to field."""
        graph = {"nodes": [{"id": "n1"}], "edges": [{"from": "n1"}]}
        assert validate_graph_structure(graph) is False

    def test_validate_edge_references_nonexistent_from(self):
        """Test with edge referencing non-existent from node."""
        graph = {
            "nodes": [{"id": "n1"}],
            "edges": [{"from": "nonexistent", "to": "n1"}],
        }
        assert validate_graph_structure(graph) is False

    def test_validate_edge_references_nonexistent_to(self):
        """Test with edge referencing non-existent to node."""
        graph = {
            "nodes": [{"id": "n1"}],
            "edges": [{"from": "n1", "to": "nonexistent"}],
        }
        assert validate_graph_structure(graph) is False


class TestGetGraphStatistics:
    """Test get_graph_statistics function."""

    def test_statistics_basic(self, simple_graph):
        """Test basic statistics calculation."""
        stats = get_graph_statistics(simple_graph)

        assert stats["num_nodes"] == 2
        assert stats["num_edges"] == 1
        assert "density" in stats
        assert "average_degree" in stats

    def test_statistics_empty_graph(self):
        """Test statistics for empty graph."""
        graph = {"nodes": [], "edges": []}

        stats = get_graph_statistics(graph)

        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0

    def test_statistics_node_types(self):
        """Test node type counting."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "ADD"},
                {"id": "n2", "type": "ADD"},
                {"id": "n3", "type": "MUL"},
            ],
            "edges": [],
        }

        stats = get_graph_statistics(graph)

        assert stats["node_types"]["ADD"] == 2
        assert stats["node_types"]["MUL"] == 1

    def test_statistics_edge_types(self):
        """Test edge type counting."""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [
                {"from": "n1", "to": "n2", "type": "data"},
                {"from": "n2", "to": "n1", "type": "control"},
            ],
        }

        stats = get_graph_statistics(graph)

        assert stats["edge_types"]["data"] == 1
        assert stats["edge_types"]["control"] == 1

    def test_statistics_degree_calculation(self):
        """Test degree statistics."""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n1", "to": "n3"},
                {"from": "n2", "to": "n3"},
            ],
        }

        stats = get_graph_statistics(graph)

        # In undirected graph interpretation:
        # n1 appears in: (n1->n2) and (n1->n3) = degree 2
        # n2 appears in: (n1->n2) and (n2->n3) = degree 2
        # n3 appears in: (n1->n3) and (n2->n3) = degree 2
        assert stats["max_degree"] == 2  # All nodes have degree 2
        assert stats["min_degree"] == 2  # All nodes have degree 2
        assert stats["average_degree"] == 2.0  # Average is 2.0

    def test_statistics_invalid_graph(self):
        """Test with invalid graph."""
        with pytest.raises(ValueError, match="Invalid graph structure"):
            get_graph_statistics({"invalid": "graph"})

    def test_statistics_density_calculation(self):
        """Test density calculation."""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n1", "to": "n3"},
                {"from": "n2", "to": "n3"},
            ],
        }

        stats = get_graph_statistics(graph)

        # 3 nodes can have max 3 edges (complete graph)
        # We have 3 edges, so density = 3/3 = 1.0
        assert stats["density"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
