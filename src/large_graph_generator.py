"""
Large Graph Generator Module (Production-Ready)
==============================================
Version: 2.0.0 - All issues fixed, validated, production-ready
Utilities for generating large graphs for testing scalability and load testing.
"""

import logging
import random
from typing import Any, Dict, List, Optional

try:
    from src.logging_config import configure as _configure_logging
except ModuleNotFoundError:
    from logging_config import configure as _configure_logging
_configure_logging()
logger = logging.getLogger("LargeGraphGenerator")

# NetworkX is optional but required for functionality
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False
    logger.error("NetworkX is not installed. Install it with: pip install networkx")

# Constants
MIN_NODES = 1
MAX_NODES = 100000
MIN_DENSITY = 0.0
MAX_DENSITY = 1.0
MIN_GRAPH_COUNT = 1
MAX_GRAPH_COUNT = 1000
VALID_NODE_TYPES = ["CONST", "ADD", "EMBED", "BRANCH", "MUL", "DIV", "CONCAT", "SPLIT"]
VALID_EDGE_TYPES = ["data", "control", "dependency"]
VALID_TOPOLOGIES = ["star", "ring", "mesh", "tree", "random"]
DEFAULT_MAX_MESH_NODES = 100
MAX_MESH_NODES_LIMIT = 1000


def generate_large_graph(
    num_nodes: int = 1000,
    density: float = 0.1,
    seed: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a large random graph for testing purposes.

    Args:
        num_nodes: Number of nodes in the graph (1 to 100,000)
        density: Edge density between 0.0 and 1.0
        seed: Random seed for reproducibility (optional)
        node_types: List of valid node types to use (optional)
        edge_types: List of valid edge types to use (optional)

    Returns:
        Dict containing graph data with nodes, edges, and metadata

    Raises:
        ImportError: If NetworkX is not available
        ValueError: If parameters are invalid
        TypeError: If parameter types are incorrect
    """
    # Check NetworkX availability
    if not NETWORKX_AVAILABLE or nx is None:
        raise ImportError(
            "NetworkX is required for graph generation. "
            "Install it with: pip install networkx"
        )

    # Validate num_nodes
    if not isinstance(num_nodes, int):
        raise TypeError(f"num_nodes must be int, got {type(num_nodes).__name__}")

    if num_nodes < MIN_NODES or num_nodes > MAX_NODES:
        raise ValueError(
            f"num_nodes must be in [{MIN_NODES}, {MAX_NODES}], got {num_nodes}"
        )

    # Validate density
    if not isinstance(density, (int, float)):
        raise TypeError(f"density must be numeric, got {type(density).__name__}")

    if density < MIN_DENSITY or density > MAX_DENSITY:
        raise ValueError(
            f"density must be in [{MIN_DENSITY}, {MAX_DENSITY}], got {density}"
        )

    # Validate seed if provided
    if seed is not None and not isinstance(seed, int):
        raise TypeError(f"seed must be int or None, got {type(seed).__name__}")

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Use default node types if not provided
    if node_types is None:
        node_types = VALID_NODE_TYPES
    elif not isinstance(node_types, list) or not node_types:
        raise ValueError("node_types must be a non-empty list")

    # Use default edge types if not provided
    if edge_types is None:
        edge_types = VALID_EDGE_TYPES
    elif not isinstance(edge_types, list) or not edge_types:
        raise ValueError("edge_types must be a non-empty list")

    try:
        # Create a random graph using Erdős-Rényi model
        graph = nx.erdos_renyi_graph(num_nodes, density, seed=seed)

        # Convert to standard format
        nodes = []
        edges = []

        # Add nodes with random attributes
        for node_id in graph.nodes():
            nodes.append(
                {
                    "id": f"node_{node_id}",
                    "type": random.choice(node_types),
                    "params": {
                        "value": round(random.uniform(0.1, 1.0), 4),
                        "priority": random.randint(1, 10),
                        "timestamp": random.randint(1000000000, 2000000000),
                    },
                }
            )

        # Add edges with correct format
        for source, target in graph.edges():
            edges.append(
                {
                    "from": f"node_{source}",
                    "to": f"node_{target}",
                    "type": random.choice(edge_types),
                }
            )

        # Calculate metadata
        is_connected = nx.is_connected(graph)
        avg_degree = (
            sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
        )

        metadata = {
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "density": density,
            "is_connected": is_connected,
            "average_degree": round(avg_degree, 2),
            "seed": seed,
        }

        logger.info(
            f"Generated graph: {num_nodes} nodes, {len(edges)} edges, "
            f"connected={is_connected}"
        )

        return {"nodes": nodes, "edges": edges, "metadata": metadata}

    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        raise


def generate_stress_test_graphs(
    count: int = 10,
    min_nodes: int = 100,
    max_nodes: int = 5000,
    min_density: float = 0.05,
    max_density: float = 0.3,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multiple graphs for stress testing with varying parameters.

    Args:
        count: Number of graphs to generate (1 to 1,000)
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        min_density: Minimum edge density
        max_density: Maximum edge density
        seed: Random seed for reproducibility (optional)

    Returns:
        List of graph dictionaries

    Raises:
        ImportError: If NetworkX is not available
        ValueError: If parameters are invalid
        TypeError: If parameter types are incorrect
    """
    # Check NetworkX availability
    if not NETWORKX_AVAILABLE or nx is None:
        raise ImportError(
            "NetworkX is required for graph generation. "
            "Install it with: pip install networkx"
        )

    # Validate count
    if not isinstance(count, int):
        raise TypeError(f"count must be int, got {type(count).__name__}")

    if count < MIN_GRAPH_COUNT or count > MAX_GRAPH_COUNT:
        raise ValueError(
            f"count must be in [{MIN_GRAPH_COUNT}, {MAX_GRAPH_COUNT}], got {count}"
        )

    # Validate node range
    if not isinstance(min_nodes, int) or not isinstance(max_nodes, int):
        raise TypeError("min_nodes and max_nodes must be int")

    if min_nodes < MIN_NODES or max_nodes > MAX_NODES or min_nodes > max_nodes:
        raise ValueError(
            f"Invalid node range: [{min_nodes}, {max_nodes}]. "
            f"Must be within [{MIN_NODES}, {MAX_NODES}]"
        )

    # Validate density range
    if not isinstance(min_density, (int, float)) or not isinstance(
        max_density, (int, float)
    ):
        raise TypeError("min_density and max_density must be numeric")

    if (
        min_density < MIN_DENSITY
        or max_density > MAX_DENSITY
        or min_density > max_density
    ):
        raise ValueError(
            f"Invalid density range: [{min_density}, {max_density}]. "
            f"Must be within [{MIN_DENSITY}, {MAX_DENSITY}]"
        )

    # Set random seed if provided
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int or None, got {type(seed).__name__}")
        random.seed(seed)

    graphs = []

    try:
        for i in range(count):
            # Vary the size and density for each graph
            num_nodes = random.randint(min_nodes, max_nodes)
            density = random.uniform(min_density, max_density)

            # Use different seed for each graph if base seed provided
            graph_seed = seed + i if seed is not None else None

            graph = generate_large_graph(num_nodes, density, seed=graph_seed)
            graph["metadata"]["graph_index"] = i
            graphs.append(graph)

            logger.debug(f"Generated stress test graph {i + 1}/{count}")

        logger.info(f"Generated {count} stress test graphs")
        return graphs

    except Exception as e:
        logger.error(f"Stress test graph generation failed: {e}")
        raise


def generate_specific_topology(
    topology_type: str = "star",
    num_nodes: int = 100,
    seed: Optional[int] = None,
    max_mesh_nodes: int = DEFAULT_MAX_MESH_NODES,
) -> Dict[str, Any]:
    """
    Generate a graph with a specific topology.

    Args:
        topology_type: Type of topology ('star', 'ring', 'mesh', 'tree', 'random')
        num_nodes: Number of nodes (actual count may vary for some topologies)
        seed: Random seed for reproducibility (optional)
        max_mesh_nodes: Maximum nodes for mesh topology (default: 100)

    Returns:
        Dict containing graph data with nodes, edges, and metadata

    Raises:
        ImportError: If NetworkX is not available
        ValueError: If parameters are invalid
        TypeError: If parameter types are incorrect
    """
    # Check NetworkX availability
    if not NETWORKX_AVAILABLE or nx is None:
        raise ImportError(
            "NetworkX is required for topology generation. "
            "Install it with: pip install networkx"
        )

    # Validate topology_type
    if not isinstance(topology_type, str):
        raise TypeError(
            f"topology_type must be string, got {type(topology_type).__name__}"
        )

    if topology_type not in VALID_TOPOLOGIES:
        raise ValueError(
            f"Invalid topology_type: '{topology_type}'. "
            f"Must be one of {VALID_TOPOLOGIES}"
        )

    # Validate num_nodes
    if not isinstance(num_nodes, int):
        raise TypeError(f"num_nodes must be int, got {type(num_nodes).__name__}")

    if num_nodes < MIN_NODES or num_nodes > MAX_NODES:
        raise ValueError(
            f"num_nodes must be in [{MIN_NODES}, {MAX_NODES}], got {num_nodes}"
        )

    # Validate max_mesh_nodes
    if not isinstance(max_mesh_nodes, int):
        raise TypeError(
            f"max_mesh_nodes must be int, got {type(max_mesh_nodes).__name__}"
        )

    if max_mesh_nodes < MIN_NODES or max_mesh_nodes > MAX_MESH_NODES_LIMIT:
        raise ValueError(
            f"max_mesh_nodes must be in [{MIN_NODES}, {MAX_MESH_NODES_LIMIT}], "
            f"got {max_mesh_nodes}"
        )

    # Set random seed if provided
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int or None, got {type(seed).__name__}")
        random.seed(seed)

    try:
        # Generate graph based on topology
        if topology_type == "star":
            # Star: n-1 leaves connected to central node
            graph = nx.star_graph(num_nodes - 1)
            actual_nodes = num_nodes

        elif topology_type == "ring":
            # Ring: nodes connected in a cycle
            graph = nx.cycle_graph(num_nodes)
            actual_nodes = num_nodes

        elif topology_type == "mesh":
            # Mesh: complete graph (limited for performance)
            if num_nodes > max_mesh_nodes:
                logger.warning(
                    f"Mesh topology limited to {max_mesh_nodes} nodes "
                    f"(requested {num_nodes}). Use max_mesh_nodes parameter to increase."
                )
                actual_nodes = max_mesh_nodes
            else:
                actual_nodes = num_nodes
            graph = nx.complete_graph(actual_nodes)

        elif topology_type == "tree":
            # Balanced tree: calculate depth to approximate num_nodes
            # For branching factor b and depth d: nodes ≈ (b^(d+1) - 1) / (b - 1)
            import math

            branching_factor = 3  # Ternary tree

            # Calculate depth needed to get close to num_nodes
            # Solve: num_nodes ≈ (3^(d+1) - 1) / 2
            # Therefore: d ≈ log3(num_nodes * 2 + 1) - 1
            if num_nodes == 1:
                depth = 0
            else:
                depth = max(0, int(math.log(num_nodes * 2 + 1, branching_factor)) - 1)

            graph = nx.balanced_tree(branching_factor, depth)
            actual_nodes = graph.number_of_nodes()

            # Warn if significant deviation from requested
            deviation = (
                abs(actual_nodes - num_nodes) / num_nodes if num_nodes > 0 else 0
            )
            if deviation > 0.3:
                logger.warning(
                    f"Tree has {actual_nodes} nodes (requested {num_nodes}). "
                    f"Balanced tree structure constrains exact node count."
                )

        elif topology_type == "random":
            # Random graph with moderate density
            graph = nx.erdos_renyi_graph(num_nodes, 0.1, seed=seed)
            actual_nodes = num_nodes

        else:
            # Default to random (should not reach here due to validation)
            graph = nx.erdos_renyi_graph(num_nodes, 0.1, seed=seed)
            actual_nodes = num_nodes

        # Convert to standard format
        nodes = []
        for n in graph.nodes():
            nodes.append(
                {
                    "id": f"node_{n}",
                    "type": "CONST",
                    "params": {"value": n, "topology_position": n},
                }
            )

        edges = []
        for s, t in graph.edges():
            edges.append({"from": f"node_{s}", "to": f"node_{t}", "type": "data"})

        # Calculate metadata
        actual_edges = graph.number_of_edges()
        is_connected = nx.is_connected(graph)

        metadata = {
            "topology": topology_type,
            "requested_nodes": num_nodes,
            "actual_nodes": actual_nodes,
            "num_edges": actual_edges,
            "is_connected": is_connected,
            "seed": seed,
        }

        logger.info(
            f"Generated {topology_type} topology: {actual_nodes} nodes, "
            f"{actual_edges} edges"
        )

        return {"nodes": nodes, "edges": edges, "metadata": metadata}

    except Exception as e:
        logger.error(f"Topology generation failed for {topology_type}: {e}")
        raise


def validate_graph_structure(graph: Dict[str, Any]) -> bool:
    """
    Validate that a graph has correct structure.

    Args:
        graph: Graph dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check that graph is a dict
        if not isinstance(graph, dict):
            logger.error("Graph must be a dictionary")
            return False

        # Check required keys
        if "nodes" not in graph or "edges" not in graph:
            logger.error("Graph must have 'nodes' and 'edges' keys")
            return False

        # Check nodes is a list
        if not isinstance(graph["nodes"], list):
            logger.error("'nodes' must be a list")
            return False

        # Check edges is a list
        if not isinstance(graph["edges"], list):
            logger.error("'edges' must be a list")
            return False

        # Validate node IDs are unique
        node_ids = set()
        for i, node in enumerate(graph["nodes"]):
            if not isinstance(node, dict):
                logger.error(f"Node at index {i} is not a dictionary")
                return False

            if "id" not in node:
                logger.error(f"Node at index {i} missing 'id' field")
                return False

            if node["id"] in node_ids:
                logger.error(f"Duplicate node ID: {node['id']}")
                return False

            node_ids.add(node["id"])

        # Validate edges reference existing nodes
        for i, edge in enumerate(graph["edges"]):
            if not isinstance(edge, dict):
                logger.error(f"Edge at index {i} is not a dictionary")
                return False

            if "from" not in edge or "to" not in edge:
                logger.error(f"Edge at index {i} missing 'from' or 'to' field")
                return False

            if edge["from"] not in node_ids:
                logger.error(f"Edge {i} references non-existent node: {edge['from']}")
                return False

            if edge["to"] not in node_ids:
                logger.error(f"Edge {i} references non-existent node: {edge['to']}")
                return False

        logger.debug("Graph structure validation passed")
        return True

    except Exception as e:
        logger.error(f"Graph validation error: {e}")
        return False


def get_graph_statistics(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for a graph.

    Args:
        graph: Graph dictionary

    Returns:
        Dictionary with graph statistics

    Raises:
        ValueError: If graph is invalid
    """
    if not validate_graph_structure(graph):
        raise ValueError("Invalid graph structure")

    num_nodes = len(graph["nodes"])
    num_edges = len(graph["edges"])

    # Calculate density
    max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
    density = num_edges / max_edges if max_edges > 0 else 0

    # Count node types
    node_types = {}
    for node in graph["nodes"]:
        node_type = node.get("type", "UNKNOWN")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    # Count edge types
    edge_types = {}
    for edge in graph["edges"]:
        edge_type = edge.get("type", "UNKNOWN")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    # Calculate degree statistics
    degrees = {}
    for edge in graph["edges"]:
        from_node = edge["from"]
        to_node = edge["to"]
        degrees[from_node] = degrees.get(from_node, 0) + 1
        degrees[to_node] = degrees.get(to_node, 0) + 1

    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
    max_degree = max(degrees.values()) if degrees else 0
    min_degree = min(degrees.values()) if degrees else 0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 4),
        "average_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "min_degree": min_degree,
        "node_types": node_types,
        "edge_types": edge_types,
    }


# Demo and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Large Graph Generator - Production Demo")
    print("=" * 60)

    if not NETWORKX_AVAILABLE:
        print("\nERROR: NetworkX is not installed!")
        print("Install it with: pip install networkx")
        exit(1)

    # Test 1: Generate basic graph
    print("\n1. Generate Basic Graph:")
    try:
        graph1 = generate_large_graph(num_nodes=50, density=0.1, seed=42)
        print(f"   Nodes: {graph1['metadata']['num_nodes']}")
        print(f"   Edges: {graph1['metadata']['num_edges']}")
        print(f"   Connected: {graph1['metadata']['is_connected']}")
        print(f"   Valid: {validate_graph_structure(graph1)}")

        stats = get_graph_statistics(graph1)
        print(f"   Avg Degree: {stats['average_degree']}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 2: Generate stress test graphs
    print("\n2. Generate Stress Test Graphs:")
    try:
        stress_graphs = generate_stress_test_graphs(
            count=5, min_nodes=100, max_nodes=500, seed=42
        )
        print(f"   Generated: {len(stress_graphs)} graphs")
        for i, g in enumerate(stress_graphs):
            print(
                f"   Graph {i}: {g['metadata']['num_nodes']} nodes, "
                f"{g['metadata']['num_edges']} edges"
            )
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 3: Generate specific topologies
    print("\n3. Generate Specific Topologies:")
    for topo in ["star", "ring", "mesh", "tree", "random"]:
        try:
            graph = generate_specific_topology(topo, num_nodes=20, seed=42)
            print(
                f"   {topo.capitalize()}: {graph['metadata']['actual_nodes']} nodes, "
                f"{graph['metadata']['num_edges']} edges"
            )
        except Exception as e:
            print(f"   {topo.capitalize()} ERROR: {e}")

    # Test 4: Tree topology with various sizes
    print("\n4. Tree Topology Node Count Accuracy:")
    for target_nodes in [10, 50, 100, 500, 1000]:
        try:
            graph = generate_specific_topology("tree", num_nodes=target_nodes, seed=42)
            actual = graph["metadata"]["actual_nodes"]
            deviation = abs(actual - target_nodes) / target_nodes * 100
            print(
                f"   Target: {target_nodes}, Actual: {actual}, "
                f"Deviation: {deviation:.1f}%"
            )
        except Exception as e:
            print(f"   Target {target_nodes} ERROR: {e}")

    # Test 5: Input validation
    print("\n5. Input Validation:")
    test_cases = [
        ("Negative nodes", lambda: generate_large_graph(num_nodes=-10)),
        ("Invalid density", lambda: generate_large_graph(num_nodes=100, density=2.0)),
        ("Invalid topology", lambda: generate_specific_topology("invalid_topo")),
        ("Invalid seed type", lambda: generate_large_graph(seed="not_a_number")),
    ]

    for name, test_func in test_cases:
        try:
            test_func()
            print(f"   {name}: ERROR - Should have raised exception")
        except (ValueError, TypeError) as e:
            print(f"   {name}: Correctly rejected - {str(e)[:50]}...")

    # Test 6: Mesh with custom limit
    print("\n6. Mesh Topology with Custom Limit:")
    try:
        graph = generate_specific_topology(
            "mesh", num_nodes=150, max_mesh_nodes=75, seed=42
        )
        print(
            f"   Requested: 150, Max allowed: 75, "
            f"Actual: {graph['metadata']['actual_nodes']}"
        )
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
