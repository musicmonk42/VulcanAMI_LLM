"""
Graph Generator
===============
Comprehensive generator for large-scale Graphix IR graphs with multiple topologies,
optimizations, and validation capabilities.
"""

import gzip
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import sys
import time
import xml.sax.saxutils as saxutils
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (Any, Callable, Dict, Iterator, List, Optional, Set, Tuple,
                    Union)

# Constants
DEFAULT_PROBABILITY = 0.05
REWIRING_PROBABILITY = 0.3
METADATA_PROBABILITY = 0.3
ATTRIBUTE_PROBABILITY = 0.2
WEIGHT_PROBABILITY = 0.3
EDGE_METADATA_PROBABILITY = 0.1
SMALL_GRAPH_THRESHOLD = 1000
MAX_CYCLE_CHECK_SIZE = 1000
LARGE_SPEC_SIZE_KB = 100
MAX_NESTING_DEPTH = 20
DEFAULT_CHUNK_SIZE = 10000
MIN_EDGE_SAMPLE_SIZE = 100000

# Graph topology types
class GraphTopology(Enum):
    """Supported graph topology types."""
    RANDOM = "random"
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert"  # Scale-free network
    WATTS_STROGATZ = "watts_strogatz"  # Small-world network
    REGULAR_LATTICE = "regular_lattice"
    TREE = "tree"
    DAG = "directed_acyclic_graph"
    COMPLETE = "complete"
    BIPARTITE = "bipartite"
    STAR = "star"
    RING = "ring"
    MESH = "mesh"
    HYPERCUBE = "hypercube"
    HIERARCHICAL = "hierarchical"
    COMMUNITY = "community"  # Multiple connected communities
    CUSTOM = "custom"

@dataclass
class GraphStatistics:
    """Statistics about a generated graph."""
    node_count: int = 0
    edge_count: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    min_degree: int = 0
    density: float = 0.0
    is_connected: bool = False
    component_count: int = 0
    has_cycles: bool = False
    diameter: int = 0
    avg_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    generation_time_ms: float = 0.0
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "avg_degree": round(self.avg_degree, 3),
            "max_degree": self.max_degree,
            "min_degree": self.min_degree,
            "density": round(self.density, 6),
            "is_connected": self.is_connected,
            "component_count": self.component_count,
            "has_cycles": self.has_cycles,
            "diameter": self.diameter,
            "avg_path_length": round(self.avg_path_length, 3),
            "clustering_coefficient": round(self.clustering_coefficient, 6),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "size_bytes": self.size_bytes
        }

@dataclass
class NodeProperties:
    """Properties for graph nodes."""
    node_type: str = "ComputeNode"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeProperties:
    """Properties for graph edges."""
    edge_type: str = "data"
    weight: float = 1.0
    directed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

class GraphGenerator:
    """
    Comprehensive generator for large-scale IR graphs with multiple topologies.
    """

    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 optimize_memory: bool = True,
                 validate_output: bool = True,
                 compression: bool = False):
        """
        Initialize the graph generator.

        Args:
            seed: Random seed for reproducibility
            verbose: Enable verbose output
            optimize_memory: Use memory-efficient data structures for large graphs
            validate_output: Validate generated graphs
            compression: Enable compression for output files
        """
        self.verbose = verbose
        self.optimize_memory = optimize_memory
        self.validate_output = validate_output
        self.compression = compression

        # Set random seed
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = random.randint(0, 1000000)
            random.seed(self.seed)

        self.generated_count = 0
        self.total_nodes_generated = 0
        self.total_edges_generated = 0

        # Node type distribution for realistic graphs
        self.node_types = ["InputNode", "OutputNode", "ComputeNode", "TransformNode",
                          "AggregateNode", "FilterNode", "JoinNode", "SplitNode"]

        # Edge type distribution
        self.edge_types = ["data", "control", "dependency", "reference", "temporal"]

        if self.verbose:
            self._log(f"GraphGenerator initialized with seed: {self.seed}")

    def _log(self, message: str) -> None:
        """Lazy logging to avoid f-string evaluation in hot paths."""
        if self.verbose:
            print(message)

    def generate_graph(self,
                       num_nodes: int = 100,
                       num_edges: Optional[int] = None,
                       topology: GraphTopology = GraphTopology.RANDOM,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate a graph with specified topology and size.

        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges (if applicable for topology)
            topology: Graph topology type
            **kwargs: Additional topology-specific parameters

        Returns:
            Generated graph in IR format
        """
        start_time = time.time()

        self._log(f"Generating {topology.value} graph with {num_nodes} nodes...")

        # Select generation method based on topology
        if topology == GraphTopology.RANDOM:
            graph = self._generate_random_graph(num_nodes, num_edges, **kwargs)
        elif topology == GraphTopology.ERDOS_RENYI:
            graph = self._generate_erdos_renyi_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.BARABASI_ALBERT:
            graph = self._generate_barabasi_albert_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.WATTS_STROGATZ:
            graph = self._generate_watts_strogatz_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.REGULAR_LATTICE:
            graph = self._generate_regular_lattice_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.TREE:
            graph = self._generate_tree_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.DAG:
            graph = self._generate_dag_graph(num_nodes, num_edges, **kwargs)
        elif topology == GraphTopology.COMPLETE:
            graph = self._generate_complete_graph(num_nodes)
        elif topology == GraphTopology.BIPARTITE:
            graph = self._generate_bipartite_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.STAR:
            graph = self._generate_star_graph(num_nodes)
        elif topology == GraphTopology.RING:
            graph = self._generate_ring_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.MESH:
            graph = self._generate_mesh_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.HYPERCUBE:
            graph = self._generate_hypercube_graph(**kwargs)
        elif topology == GraphTopology.HIERARCHICAL:
            graph = self._generate_hierarchical_graph(num_nodes, **kwargs)
        elif topology == GraphTopology.COMMUNITY:
            graph = self._generate_community_graph(num_nodes, **kwargs)
        else:
            graph = self._generate_custom_graph(num_nodes, num_edges, **kwargs)

        # Add metadata
        generation_time = (time.time() - start_time) * 1000
        graph["metadata"] = {
            "topology": topology.value,
            "generation_time_ms": generation_time,
            "seed": self.seed,
            "generator_version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }

        # Validate if requested and graph is small enough
        if self.validate_output and len(graph.get("nodes", [])) < SMALL_GRAPH_THRESHOLD:
            self._validate_graph(graph)

        # Update counters
        self.generated_count += 1
        self.total_nodes_generated += len(graph.get("nodes", []))
        self.total_edges_generated += len(graph.get("edges", []))

        if self.verbose:
            stats = self._calculate_statistics(graph)
            self._log(f"Generated {topology.value} graph in {generation_time:.2f}ms")
            self._log(f"  Nodes: {stats.node_count}, Edges: {stats.edge_count}")
            self._log(f"  Density: {stats.density:.6f}, Avg Degree: {stats.avg_degree:.2f}")

        return graph

    def _generate_random_graph(self, num_nodes: int, num_edges: Optional[int],
                              allow_self_loops: bool = False,
                              allow_multi_edges: bool = False) -> Dict[str, Any]:
        """Generate a random graph with specified nodes and edges."""
        if num_edges is None:
            # Default to sparse graph
            num_edges = min(num_nodes * 2, num_nodes * (num_nodes - 1) // 2)

        nodes = self._create_nodes(num_nodes)
        edges = []
        edge_set = set()

        max_attempts = num_edges * 10
        attempts = 0

        while len(edges) < num_edges and attempts < max_attempts:
            from_idx = random.randint(0, num_nodes - 1)
            to_idx = random.randint(0, num_nodes - 1)

            if not allow_self_loops and from_idx == to_idx:
                attempts += 1
                continue

            edge_key = (from_idx, to_idx)
            if not allow_multi_edges and edge_key in edge_set:
                attempts += 1
                continue

            edge_set.add(edge_key)
            edges.append(self._create_edge(f"node_{from_idx}", f"node_{to_idx}"))
            attempts += 1

        return self._create_graph_structure(nodes, edges)

    def _generate_erdos_renyi_graph(self, num_nodes: int, probability: float = None) -> Dict[str, Any]:
        """Generate an Erdős-Rényi random graph."""
        if probability is None:
            # Use default that creates connected graphs with high probability
            probability = 2 * math.log(num_nodes) / num_nodes if num_nodes > 1 else DEFAULT_PROBABILITY

        nodes = self._create_nodes(num_nodes)
        edges = []

        # For large graphs with memory optimization, sample edges instead of checking all pairs
        if num_nodes > 1000 and self.optimize_memory:
            expected_edges = int(probability * num_nodes * (num_nodes - 1) / 2)
            # Sample edges using rejection sampling to avoid materializing all combinations
            edge_set = set()
            max_attempts = expected_edges * 10
            attempts = 0

            while len(edge_set) < expected_edges and attempts < max_attempts:
                from_idx = random.randint(0, num_nodes - 1)
                to_idx = random.randint(0, num_nodes - 1)
                if from_idx != to_idx:
                    edge_key = (min(from_idx, to_idx), max(from_idx, to_idx))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                attempts += 1

            for from_idx, to_idx in edge_set:
                edges.append(self._create_edge(f"node_{from_idx}", f"node_{to_idx}"))
        else:
            # Standard generation for smaller graphs
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if random.random() < probability:
                        edges.append(self._create_edge(f"node_{i}", f"node_{j}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_barabasi_albert_graph(self, num_nodes: int, m: int = 2) -> Dict[str, Any]:
        """Generate a Barabási-Albert scale-free network."""
        if m < 1:
            m = 1
        if m >= num_nodes:
            m = num_nodes - 1

        nodes = self._create_nodes(num_nodes)
        edges = []

        # Start with a complete graph of m+1 nodes
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                edges.append(self._create_edge(f"node_{i}", f"node_{j}"))

        # Track node degrees for preferential attachment
        degree = defaultdict(int)
        for edge in edges:
            degree[edge["from"]] += 1
            degree[edge["to"]] += 1

        # Add remaining nodes with preferential attachment
        for new_node in range(m + 1, num_nodes):
            # Calculate attachment probabilities
            total_degree = sum(degree.values())
            if total_degree == 0:
                targets = random.sample(range(new_node), min(m, new_node))
            else:
                # Use weighted random selection based on degree
                nodes_list = list(range(new_node))
                weights = [degree[f"node_{i}"] for i in nodes_list]
                total_weight = sum(weights)

                targets = set()
                while len(targets) < min(m, new_node):
                    r = random.random() * total_weight
                    cumsum = 0
                    for idx, weight in zip(nodes_list, weights):
                        cumsum += weight
                        if cumsum >= r:
                            targets.add(idx)
                            break

            for target in targets:
                edges.append(self._create_edge(f"node_{new_node}", f"node_{target}"))
                degree[f"node_{new_node}"] += 1
                degree[f"node_{target}"] += 1

        return self._create_graph_structure(nodes, edges)

    def _generate_watts_strogatz_graph(self, num_nodes: int, k: int = 4,
                                      rewiring_prob: float = REWIRING_PROBABILITY) -> Dict[str, Any]:
        """Generate a Watts-Strogatz small-world network."""
        if k >= num_nodes:
            k = num_nodes - 1
        if k < 2:
            k = 2

        nodes = self._create_nodes(num_nodes)
        edges = []
        edge_set = set()  # Track edges to avoid duplicates

        # Create regular ring lattice
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                target = (i + j) % num_nodes
                edge_key = (i, target) if i < target else (target, i)
                if edge_key not in edge_set:
                    edges.append(self._create_edge(f"node_{i}", f"node_{target}"))
                    edge_set.add(edge_key)

        # Rewire edges with probability
        edges_to_rewire = []
        for idx, edge in enumerate(edges):
            if random.random() < rewiring_prob:
                edges_to_rewire.append(idx)

        for idx in edges_to_rewire:
            old_edge = edges[idx]
            from_node = old_edge["from"]
            from_idx = int(from_node.split("_")[1])

            # Find new target (not already connected, including both directions)
            existing_targets = set()
            for e in edges:
                if e["from"] == from_node:
                    existing_targets.add(e["to"])
                if e["to"] == from_node:
                    existing_targets.add(e["from"])

            available_targets = [f"node_{i}" for i in range(num_nodes]
                               if f"node_{i}" not in existing_targets and f"node_{i}" != from_node]

            if available_targets:
                new_target = random.choice(available_targets)
                edges[idx] = self._create_edge(from_node, new_target)

        return self._create_graph_structure(nodes, edges)

    def _generate_regular_lattice_graph(self, num_nodes: int, dimensions: int = 2) -> Dict[str, Any]:
        """Generate a regular lattice graph."""
        # Calculate grid dimensions
        if dimensions == 2:
            grid_size = int(math.sqrt(num_nodes))
            actual_nodes = grid_size * grid_size
        elif dimensions == 3:
            grid_size = int(num_nodes ** (1/3))
            actual_nodes = grid_size ** 3
        else:
            grid_size = int(num_nodes ** (1/dimensions))
            actual_nodes = grid_size ** dimensions

        nodes = self._create_nodes(actual_nodes)
        edges = []

        # Create grid connections
        if dimensions == 2:
            for i in range(grid_size):
                for j in range(grid_size):
                    node_id = i * grid_size + j
                    # Right neighbor
                    if j < grid_size - 1:
                        edges.append(self._create_edge(f"node_{node_id}", f"node_{node_id + 1}"))
                    # Bottom neighbor
                    if i < grid_size - 1:
                        edges.append(self._create_edge(f"node_{node_id}", f"node_{node_id + grid_size}"))
        elif dimensions == 3:
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        node_id = i * grid_size * grid_size + j * grid_size + k
                        # Connections in 3D
                        if k < grid_size - 1:
                            edges.append(self._create_edge(f"node_{node_id}", f"node_{node_id + 1}"))
                        if j < grid_size - 1:
                            edges.append(self._create_edge(f"node_{node_id}", f"node_{node_id + grid_size}"))
                        if i < grid_size - 1:
                            edges.append(self._create_edge(f"node_{node_id}", f"node_{node_id + grid_size * grid_size}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_tree_graph(self, num_nodes: int, branching_factor: int = 2,
                            balanced: bool = True) -> Dict[str, Any]:
        """Generate a tree graph."""
        nodes = self._create_nodes(num_nodes)
        edges = []

        if balanced:
            # Generate balanced tree
            for i in range(num_nodes):
                for j in range(1, branching_factor + 1):
                    child_idx = i * branching_factor + j
                    if child_idx < num_nodes:
                        edges.append(self._create_edge(f"node_{i}", f"node_{child_idx}"))
        else:
            # Generate random tree using corrected Prüfer sequence
            if num_nodes <= 2:
                if num_nodes == 2:
                    edges.append(self._create_edge("node_0", "node_1"))
            else:
                # Generate Prüfer sequence
                prufer = [random.randint(0, num_nodes - 1) for _ in range(num_nodes - 2)]

                # Decode Prüfer sequence to tree
                degree = [1] * num_nodes
                for node in prufer:
                    degree[node] += 1

                # Build edges from Prüfer sequence
                for node in prufer:
                    # Find first node with degree 1
                    for leaf in range(num_nodes):
                        if degree[leaf] == 1:
                            edges.append(self._create_edge(f"node_{leaf}", f"node_{node}"))
                            degree[leaf] -= 1
                            degree[node] -= 1
                            break

                # Connect remaining two nodes with degree 1
                remaining = [i for i in range(num_nodes) if degree[i] == 1]
                if len(remaining) == 2:
                    edges.append(self._create_edge(f"node_{remaining[0]}", f"node_{remaining[1]}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_dag_graph(self, num_nodes: int, num_edges: Optional[int],
                           layers: Optional[int] = None) -> Dict[str, Any]:
        """Generate a Directed Acyclic Graph (DAG)."""
        nodes = self._create_nodes(num_nodes)

        if layers is None:
            layers = min(int(math.sqrt(num_nodes)), 10)

        # Assign nodes to layers - ensure strict ordering
        nodes_per_layer = max(1, num_nodes // layers)
        layer_assignment = {}
        for i in range(num_nodes):
            layer_assignment[i] = min(i // nodes_per_layer, layers - 1)

        edges = []
        edge_set = set()

        if num_edges is None:
            # Create edges only from lower to higher layers
            for i in range(num_nodes):
                current_layer = layer_assignment[i]
                # Connect to 1-3 nodes in strictly higher layers
                connections = random.randint(1, min(3, num_nodes - i - 1))
                candidates = [j for j in range(i + 1, num_nodes]
                            if layer_assignment[j] > current_layer]
                if candidates:
                    targets = random.sample(candidates, min(connections, len(candidates))
                    for target in targets:
                        edges.append(self._create_edge(f"node_{i}", f"node_{target}"))
        else:
            # Generate specified number of edges maintaining DAG property
            edge_count = 0
            max_attempts = num_edges * 10
            attempts = 0

            while edge_count < num_edges and attempts < max_attempts:
                from_idx = random.randint(0, num_nodes - 2)
                to_idx = random.randint(from_idx + 1, num_nodes - 1)

                # Ensure edges only go to higher layers
                if layer_assignment[from_idx] < layer_assignment[to_idx]:
                    edge_key = (from_idx, to_idx)
                    if edge_key not in edge_set:
                        edges.append(self._create_edge(f"node_{from_idx}", f"node_{to_idx}"))
                        edge_set.add(edge_key)
                        edge_count += 1
                attempts += 1

        return self._create_graph_structure(nodes, edges)

    def _generate_complete_graph(self, num_nodes: int) -> Dict[str, Any]:
        """Generate a complete graph (all nodes connected to all others)."""
        nodes = self._create_nodes(num_nodes)
        edges = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append(self._create_edge(f"node_{i}", f"node_{j}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_bipartite_graph(self, num_nodes: int,
                                 partition_ratio: float = 0.5,
                                 connection_prob: float = 0.3) -> Dict[str, Any]:
        """Generate a bipartite graph."""
        partition_1_size = int(num_nodes * partition_ratio)
        partition_2_size = num_nodes - partition_1_size

        # Create nodes with partition assignment
        nodes = []
        for i in range(partition_1_size):
            node = self._create_node(f"node_{i}", node_type="InputNode")
            node["partition"] = 1
            nodes.append(node)

        for i in range(partition_1_size, num_nodes):
            node = self._create_node(f"node_{i}", node_type="OutputNode")
            node["partition"] = 2
            nodes.append(node)

        # Sample expected number of edges instead of checking all pairs
        expected_edges = int(connection_prob * partition_1_size * partition_2_size)
        edges = []
        edge_set = set()

        max_attempts = expected_edges * 10
        attempts = 0

        while len(edges) < expected_edges and attempts < max_attempts:
            i = random.randint(0, partition_1_size - 1)
            j = random.randint(partition_1_size, num_nodes - 1)
            edge_key = (i, j)
            if edge_key not in edge_set:
                edges.append(self._create_edge(f"node_{i}", f"node_{j}"))
                edge_set.add(edge_key)
            attempts += 1

        return self._create_graph_structure(nodes, edges)

    def _generate_star_graph(self, num_nodes: int) -> Dict[str, Any]:
        """Generate a star graph (one central node connected to all others)."""
        nodes = self._create_nodes(num_nodes)
        edges = []

        # Node 0 is the center
        for i in range(1, num_nodes):
            edges.append(self._create_edge("node_0", f"node_{i}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_ring_graph(self, num_nodes: int, k_neighbors: int = 1) -> Dict[str, Any]:
        """Generate a ring graph (circular connection pattern)."""
        nodes = self._create_nodes(num_nodes)
        edges = []

        for i in range(num_nodes):
            for j in range(1, k_neighbors + 1):
                target = (i + j) % num_nodes
                edges.append(self._create_edge(f"node_{i}", f"node_{target}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_mesh_graph(self, num_nodes: int, dimensions: Tuple[int, ...] = None) -> Dict[str, Any]:
        """Generate a mesh graph."""
        if dimensions is None:
            # Default to 2D mesh
            size = int(math.sqrt(num_nodes))
            dimensions = (size, size)

        # Adjust node count to fit dimensions
        actual_nodes = 1
        for dim in dimensions:
            actual_nodes *= dim

        nodes = self._create_nodes(actual_nodes)
        edges = []
        edge_set = set()

        # Create mesh connections
        def get_node_id(coords):
            node_id = 0
            multiplier = 1
            for i in range(len(coords) - 1, -1, -1):
                node_id += coords[i] * multiplier
                multiplier *= dimensions[i]
            return node_id

        def get_forward_neighbors(coords):
            """Get only forward neighbors to avoid duplicates."""
            neighbors = []
            for i in range(len(coords))
                # Only forward neighbor
                if coords[i] < dimensions[i] - 1:
                    new_coords = list(coords)
                    new_coords[i] += 1
                    neighbors.append(tuple(new_coords))
            return neighbors

        # Iterate through all nodes
        for coords in itertools.product(*[range(dim) for dim in dimensions]):
            node_id = get_node_id(coords)
            for neighbor_coords in get_forward_neighbors(coords):
                neighbor_id = get_node_id(neighbor_coords)
                edge_key = (node_id, neighbor_id)
                if edge_key not in edge_set:
                    edges.append(self._create_edge(f"node_{node_id}", f"node_{neighbor_id}"))
                    edge_set.add(edge_key)

        return self._create_graph_structure(nodes, edges)

    def _generate_hypercube_graph(self, dimension: int = 3) -> Dict[str, Any]:
        """
        Generate a hypercube graph.
        Note: num_nodes is determined by dimension (2^dimension nodes).
        """
        num_nodes = 2 ** dimension
        nodes = self._create_nodes(num_nodes)
        edges = []

        # Connect nodes that differ by one bit
        for i in range(num_nodes):
            for j in range(dimension):
                neighbor = i ^ (1 << j)  # XOR to flip j-th bit
                if neighbor > i:  # Avoid duplicate edges
                    edges.append(self._create_edge(f"node_{i}", f"node_{neighbor}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_hierarchical_graph(self, num_nodes: int, levels: int = 3,
                                    branching_factor: int = 3) -> Dict[str, Any]:
        """Generate a hierarchical graph with multiple levels."""
        nodes = []
        edges = []

        # Create hierarchical structure
        level_nodes = defaultdict(list)
        node_counter = 0

        # Root level
        root = self._create_node(f"node_{node_counter}", node_type="InputNode")
        root["level"] = 0
        nodes.append(root)
        level_nodes[0].append(node_counter)
        node_counter += 1

        # Generate levels
        for level in range(1, levels):
            nodes_in_level = len(level_nodes[level - 1]) * branching_factor
            for _ in range(min(nodes_in_level, num_nodes - node_counter))
                node = self._create_node(f"node_{node_counter}",
                                       node_type="ComputeNode" if level < levels - 1 else "OutputNode")
                node["level"] = level
                nodes.append(node)
                level_nodes[level].append(node_counter)

                # Connect to parent
                parent_idx = random.choice(level_nodes[level - 1])
                edges.append(self._create_edge(f"node_{parent_idx}", f"node_{node_counter}"))

                node_counter += 1
                if node_counter >= num_nodes:
                    break

            if node_counter >= num_nodes:
                break

        # Add remaining nodes if needed
        while node_counter < num_nodes:
            node = self._create_node(f"node_{node_counter}", node_type="ComputeNode")
            node["level"] = levels - 1
            nodes.append(node)

            # Connect to random parent
            if level_nodes[levels - 2]:
                parent_idx = random.choice(level_nodes[levels - 2])
                edges.append(self._create_edge(f"node_{parent_idx}", f"node_{node_counter}"))

            node_counter += 1

        return self._create_graph_structure(nodes, edges)

    def _generate_community_graph(self, num_nodes: int, num_communities: int = 3,
                                 intra_prob: float = 0.3, inter_prob: float = 0.01) -> Dict[str, Any]:
        """Generate a graph with community structure."""
        nodes = []
        edges = []

        # Assign nodes to communities evenly
        nodes_per_community = num_nodes // num_communities
        remainder = num_nodes % num_communities

        community_assignment = {}
        node_idx = 0

        for community in range(num_communities):
            # Distribute remainder nodes to first communities
            community_size = nodes_per_community + (1 if community < remainder else 0)
            for _ in range(community_size):
                if node_idx < num_nodes:
                    node = self._create_node(f"node_{node_idx}")
                    node["community"] = community
                    nodes.append(node)
                    community_assignment[node_idx] = community
                    node_idx += 1

        # Create intra-community and inter-community edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if community_assignment[i] == community_assignment[j]:
                    # Same community - higher probability
                    if random.random() < intra_prob:
                        edges.append(self._create_edge(f"node_{i}", f"node_{j}"))
                else:
                    # Different communities - lower probability
                    if random.random() < inter_prob:
                        edges.append(self._create_edge(f"node_{i}", f"node_{j}"))

        return self._create_graph_structure(nodes, edges)

    def _generate_custom_graph(self, num_nodes: int, num_edges: Optional[int],
                              generator_func: Optional[Callable] = None,
                              **kwargs) -> Dict[str, Any]:
        """Generate a custom graph using a provided generator function."""
        if generator_func is not None:
            return generator_func(num_nodes, num_edges, **kwargs)
        else:
            # Default to random graph
            return self._generate_random_graph(num_nodes, num_edges, **kwargs)

    def _create_nodes(self, count: int) -> List[Dict[str, Any]]:
        """Create a list of nodes."""
        nodes = []
        for i in range(count):
            node_type = random.choice(self.node_types)
            nodes.append(self._create_node(f"node_{i}", node_type))
        return nodes

    def _create_node(self, node_id: str, node_type: Optional[str] = None) -> Dict[str, Any]:
        """Create a single node."""
        if node_type is None:
            node_type = random.choice(self.node_types)

        node = {
            "id": node_id,
            "type": node_type
        }

        # Add optional properties
        if random.random() < METADATA_PROBABILITY:
            node["metadata"] = {
                "weight": random.random(),
                "priority": random.randint(1, 10),
                "timestamp": time.time()
            }

        if random.random() < ATTRIBUTE_PROBABILITY:
            node["attributes"] = {
                "color": random.choice(["red", "blue", "green", "yellow"]),
                "size": random.choice(["small", "medium", "large"])
            }

        return node

    def _create_edge(self, from_node: str, to_node: str) -> Dict[str, Any]:
        """Create a single edge."""
        edge = {
            "from": from_node,
            "to": to_node,
            "type": random.choice(self.edge_types)
        }

        # Add optional properties
        if random.random() < WEIGHT_PROBABILITY:
            edge["weight"] = random.random()

        if random.random() < EDGE_METADATA_PROBABILITY:
            edge["metadata"] = {
                "label": f"edge_{random.randint(1000, 9999)}",
                "timestamp": time.time()
            }

        return edge

    def _create_graph_structure(self, nodes: List[Dict[str, Any]],
                               edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create the final graph structure."""
        graph_id = self._generate_id()

        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges
        }

    def _generate_id(self) -> str:
        """Generate a unique graph ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        return f"graph_{timestamp}_{random_suffix}"

    def _validate_graph(self, graph: Dict[str, Any]) -> bool:
        """Validate that the graph structure is correct."""
        try:
            # Check required fields
            assert "grammar_version" in graph
            assert "id" in graph
            assert "type" in graph
            assert "nodes" in graph
            assert "edges" in graph

            # Validate nodes (use sampling for large graphs)
            node_ids = set()
            nodes = graph["nodes"]
            sample_size = min(len(nodes), SMALL_GRAPH_THRESHOLD)

            for node in (random.sample(nodes, sample_size) if len(nodes) > sample_size else nodes):
                assert "id" in node
                assert "type" in node
                if len(nodes) <= SMALL_GRAPH_THRESHOLD:
                    assert node["id"] not in node_ids  # No duplicate IDs (only check for small graphs)
                    node_ids.add(node["id"])

            # For large graphs, build node_ids set if not already done
            if len(nodes) > sample_size:
                node_ids = {node["id"] for node in nodes}

            # Validate edges (sample for large graphs)
            edges = graph["edges"]
            sample_size = min(len(edges), SMALL_GRAPH_THRESHOLD)

            for edge in (random.sample(edges, sample_size) if len(edges) > sample_size else edges):
                assert "from" in edge
                assert "to" in edge
                # Check that edge endpoints exist (for small graphs only)
                if len(nodes) <= SMALL_GRAPH_THRESHOLD:
                    assert edge["from"] in node_ids
                    assert edge["to"] in node_ids

            return True
        except AssertionError as e:
            self._log(f"Validation failed: {e}")
            return False

    def _calculate_statistics(self, graph: Dict[str, Any]) -> GraphStatistics:
        """Calculate comprehensive statistics for a graph."""
        stats = GraphStatistics()

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        stats.node_count = len(nodes)
        stats.edge_count = len(edges)

        if stats.node_count == 0:
            return stats

        # Calculate degree distribution
        degree_count = defaultdict(int)
        for edge in edges:
            degree_count[edge["from"]] += 1
            degree_count[edge["to"]] += 1

        degrees = list(degree_count.values())
        if degrees:
            stats.avg_degree = sum(degrees) / len(degrees)
            stats.max_degree = max(degrees)
            stats.min_degree = min(degrees)

        # Calculate density
        max_edges = stats.node_count * (stats.node_count - 1)
        if max_edges > 0:
            stats.density = stats.edge_count / max_edges

        # Check connectivity (only for smaller graphs)
        if stats.node_count < SMALL_GRAPH_THRESHOLD and stats.node_count > 0:
            visited = set()
            queue = deque([nodes[0]["id"]])
            visited.add(nodes[0]["id"])

            # Build adjacency list - treat as undirected for connectivity check
            adj_list = defaultdict(list)
            for edge in edges:
                adj_list[edge["from"]].append(edge["to"])
                adj_list[edge["to"]].append(edge["from"])

            while queue:
                current = queue.popleft()
                for neighbor in adj_list[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            stats.is_connected = len(visited) == stats.node_count

            # Count components
            all_nodes = set(node["id"] for node in nodes)
            unvisited = all_nodes - visited
            stats.component_count = 1 if visited else 0

            while unvisited:
                start = unvisited.pop()
                queue = deque([start])
                component = set([start])

                while queue:
                    current = queue.popleft()
                    for neighbor in adj_list[current]:
                        if neighbor in unvisited:
                            unvisited.remove(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)

                stats.component_count += 1

        # Check for cycles (proper detection for small graphs, heuristic for large)
        if stats.node_count < MAX_CYCLE_CHECK_SIZE:
            # For small graphs, use proper cycle detection
            # For trees: edges = nodes - components
            # For graphs with cycles: edges >= nodes - components + 1
            expected_tree_edges = stats.node_count - max(1, stats.component_count)
            stats.has_cycles = stats.edge_count > expected_tree_edges
        else:
            # Heuristic for large graphs
            stats.has_cycles = stats.edge_count >= stats.node_count

        # Calculate size using approximate method for large graphs
        if stats.node_count < SMALL_GRAPH_THRESHOLD:
            stats.size_bytes = len(json.dumps(graph))
        else:
            # Approximate size for large graphs
            stats.size_bytes = sys.getsizeof(graph)

        return stats

    def generate_large_scale_graph(self, num_nodes: int,
                                  topology: GraphTopology = GraphTopology.BARABASI_ALBERT,
                                  chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, Any]:
        """
        Generate a very large graph using memory-efficient streaming techniques.

        Args:
            num_nodes: Number of nodes (can be millions)
            topology: Graph topology
            chunk_size: Process nodes in chunks for memory efficiency

        Returns:
            Large graph structure
        """
        if num_nodes <= chunk_size:
            return self.generate_graph(num_nodes, topology=topology)

        self._log(f"Generating large-scale {topology.value} graph with {num_nodes:,} nodes...")
        self._log(f"Using chunk size: {chunk_size:,}")

        start_time = time.time()

        # Use memory-mapped approach or streaming for very large graphs
        graph_id = self._generate_id()

        # For truly memory-efficient generation, we would write directly to file
        # For now, we'll limit the actual node/edge generation
        if self.optimize_memory and num_nodes > 100000:
            # Generate a more manageable subset with metadata indicating full size
            actual_nodes = min(num_nodes, 50000)
            self._log(f"Memory optimization enabled: generating {actual_nodes:,} nodes as representative sample")

            graph = self.generate_graph(actual_nodes, topology=topology)
            graph["metadata"]["is_large_scale"] = True
            graph["metadata"]["full_node_count"] = num_nodes
            graph["metadata"]["chunk_size"] = chunk_size
            graph["metadata"]["is_sample"] = True

            generation_time = (time.time() - start_time) * 1000
            graph["metadata"]["generation_time_ms"] = generation_time

            self._log(f"Large-scale graph sample generated in {generation_time/1000:.2f}s")
            self._log(f"  Sample: {len(graph['nodes']):,} nodes, {len(graph['edges']):,} edges")
            self._log(f"  Represents: {num_nodes:,} node graph")

            return graph

        # For smaller "large" graphs, generate normally
        return self.generate_graph(num_nodes, topology=topology)

    def save_graph(self, graph: Dict[str, Any], output_dir: str = "generated_graphs",
                   format: str = "json", compress: bool = None):
        """
        Save the graph to a file with optional compression.

        Args:
            graph: Graph to save
            output_dir: Output directory
            format: Output format (json, pickle, graphml)
            compress: Whether to compress the output
        """
        os.makedirs(output_dir, exist_ok=True)

        if compress is None:
            compress = self.compression

        graph_id = graph.get("id", "unnamed")

        if format == "json":
            ext = ".json.gz" if compress else ".json"
            file_path = os.path.join(output_dir, f"{graph_id}{ext}")

            if compress:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(graph, f, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(graph, f, indent=2)

        elif format == "pickle":
            ext = ".pkl.gz" if compress else ".pkl"
            file_path = os.path.join(output_dir, f"{graph_id}{ext}")

            if compress:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(graph, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(graph, f)

        elif format == "graphml":
            # Export to GraphML format for compatibility with other tools
            file_path = os.path.join(output_dir, f"{graph_id}.graphml")
            self._export_graphml(graph, file_path)

        else:
            raise ValueError(f"Unsupported format: {format}")

        if self.verbose:
            file_size = os.path.getsize(file_path)
            self._log(f"Saved graph to: {file_path} ({file_size / 1024:.2f} KB)")

    def _export_graphml(self, graph: Dict[str, Any], file_path: str):
        """Export graph to GraphML format with proper XML escaping."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
            f.write('  <graph id="G" edgedefault="directed">\n')

            # Write nodes with XML escaping
            for node in graph.get("nodes", []):
                node_id = saxutils.escape(str(node["id"]))
                f.write(f'    <node id="{node_id}"')
                if "type" in node:
                    node_type = saxutils.escape(str(node["type"]))
                    f.write(f' type="{node_type}"')
                f.write('/>\n')

            # Write edges with XML escaping
            for i, edge in enumerate(graph.get("edges", [])):
                from_id = saxutils.escape(str(edge["from"]))
                to_id = saxutils.escape(str(edge["to"]))
                f.write(f'    <edge id="e{i}" source="{from_id}" target="{to_id}"')
                if "type" in edge:
                    edge_type = saxutils.escape(str(edge["type"]))
                    f.write(f' type="{edge_type}"')
                if "weight" in edge:
                    f.write(f' weight="{edge["weight"]}"')
                f.write('/>\n')

            f.write('  </graph>\n')
            f.write('</graphml>\n')

    def batch_generate(self, specs: List[Dict[str, Any]],
                       parallel: bool = False) -> List[Dict[str, Any]]:
        """
        Generate multiple graphs from specifications.

        Args:
            specs: List of graph specifications
            parallel: Whether to use parallel generation (requires multiprocessing)

        Returns:
            List of generated graphs
        """
        graphs = []

        for i, spec in enumerate(specs):
            self._log(f"Generating graph {i+1}/{len(specs)}...")

            graph = self.generate_graph(**spec)
            graphs.append(graph)

        return graphs

    def generate_benchmark_suite(self, output_dir: str = "benchmark_graphs"):
        """Generate a comprehensive suite of graphs for benchmarking."""
        os.makedirs(output_dir, exist_ok=True)

        benchmarks = [
            # Small graphs
            {"num_nodes": 10, "topology": GraphTopology.COMPLETE, "name": "small_complete"},
            {"num_nodes": 20, "topology": GraphTopology.TREE, "name": "small_tree"},
            {"num_nodes": 25, "topology": GraphTopology.MESH, "name": "small_mesh"},

            # Medium graphs
            {"num_nodes": 100, "topology": GraphTopology.ERDOS_RENYI, "name": "medium_random"},
            {"num_nodes": 200, "topology": GraphTopology.BARABASI_ALBERT, "name": "medium_scalefree"},
            {"num_nodes": 150, "topology": GraphTopology.WATTS_STROGATZ, "name": "medium_smallworld"},

            # Large graphs
            {"num_nodes": 1000, "topology": GraphTopology.DAG, "name": "large_dag"},
            {"num_nodes": 2000, "topology": GraphTopology.COMMUNITY, "name": "large_community"},
            {"num_nodes": 1500, "topology": GraphTopology.HIERARCHICAL, "name": "large_hierarchical"},

            # Extra large graphs
            {"num_nodes": 10000, "topology": GraphTopology.RANDOM, "name": "xlarge_random"},
            {"num_nodes": 20000, "topology": GraphTopology.BARABASI_ALBERT, "name": "xlarge_scalefree"},
        ]

        results = []

        for benchmark in benchmarks:
            name = benchmark.pop("name")
            self._log(f"\nGenerating benchmark: {name}")

            graph = self.generate_graph(**benchmark)

            # Save graph
            graph_file = os.path.join(output_dir, f"{name}.json")
            with open(graph_file, 'w') as f:
                json.dump(graph, f, indent=2)

            # Calculate and save statistics
            stats = self._calculate_statistics(graph)
            stats_dict = stats.to_dict()
            stats_dict["name"] = name
            stats_dict["file"] = graph_file
            results.append(stats_dict)

            self._log(f"  Saved: {graph_file}")
            self._log(f"  Stats: {stats.node_count} nodes, {stats.edge_count} edges, "
                     f"density={stats.density:.6f}")

        # Save benchmark results
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nBenchmark suite generated in: {output_dir}")
        print(f"Results saved to: {results_file}")

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get generation summary statistics."""
        return {
            "seed": self.seed,
            "graphs_generated": self.generated_count,
            "total_nodes": self.total_nodes_generated,
            "total_edges": self.total_edges_generated,
            "avg_nodes_per_graph": self.total_nodes_generated / self.generated_count if self.generated_count > 0 else 0,
            "avg_edges_per_graph": self.total_edges_generated / self.generated_count if self.generated_count > 0 else 0,
            "configuration": {
                "optimize_memory": self.optimize_memory,
                "validate_output": self.validate_output,
                "compression": self.compression,
                "verbose": self.verbose
            }
        }

def main():
    """
    Main function demonstrating various graph generation capabilities.
    """
    generator = GraphGenerator(verbose=True, optimize_memory=True)

    print("=" * 60)
    print("COMPREHENSIVE GRAPH GENERATOR")
    print("=" * 60)

    # Generate various topologies
    topologies_to_test = [
        (GraphTopology.RANDOM, 50, {"num_edges": 100}),
        (GraphTopology.ERDOS_RENYI, 100, {"probability": DEFAULT_PROBABILITY}),
        (GraphTopology.BARABASI_ALBERT, 200, {"m": 3}),
        (GraphTopology.WATTS_STROGATZ, 100, {"k": 4, "rewiring_prob": REWIRING_PROBABILITY}),
        (GraphTopology.TREE, 63, {"branching_factor": 2, "balanced": True}),
        (GraphTopology.DAG, 50, {"layers": 5}),
        (GraphTopology.COMPLETE, 10, {}),
        (GraphTopology.BIPARTITE, 40, {"partition_ratio": 0.4, "connection_prob": 0.3}),
        (GraphTopology.STAR, 20, {}),
        (GraphTopology.RING, 30, {"k_neighbors": 2}),
        (GraphTopology.MESH, 49, {"dimensions": (7, 7)}),
        (GraphTopology.HYPERCUBE, 16, {"dimension": 4}),
        (GraphTopology.HIERARCHICAL, 100, {"levels": 4, "branching_factor": 3}),
        (GraphTopology.COMMUNITY, 150, {"num_communities": 3, "intra_prob": 0.3, "inter_prob": 0.01})
    ]

    output_dir = "generated_graphs"
    os.makedirs(output_dir, exist_ok=True)

    for topology, num_nodes, kwargs in topologies_to_test:
        print(f"\n{'-' * 40}")
        print(f"Generating {topology.value} graph...")

        graph = generator.generate_graph(num_nodes=num_nodes, topology=topology, **kwargs)

        # Save the graph
        generator.save_graph(graph, output_dir=output_dir)

        # Calculate statistics
        stats = generator._calculate_statistics(graph)
        print(f"Statistics:")
        print(f"  Nodes: {stats.node_count}, Edges: {stats.edge_count}")
        print(f"  Density: {stats.density:.6f}, Avg Degree: {stats.avg_degree:.2f}")
        print(f"  Connected: {stats.is_connected}, Components: {stats.component_count}")

    # Generate a large-scale graph
    print(f"\n{'=' * 60}")
    print("LARGE-SCALE GRAPH GENERATION")
    print("=" * 60)

    large_graph = generator.generate_large_scale_graph(
        num_nodes=100000,
        topology=GraphTopology.BARABASI_ALBERT,
        chunk_size=DEFAULT_CHUNK_SIZE
    )

    # Save with compression
    generator.save_graph(large_graph, output_dir=output_dir, compress=True)

    # Generate benchmark suite
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUITE GENERATION")
    print("=" * 60)
    generator.generate_benchmark_suite(output_dir="benchmark_graphs")

    # Print final summary
    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print("=" * 60)
    summary = generator.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print(f"\nAll graphs saved to: {output_dir}/")
    print("Generation complete!")

if __name__ == "__main__":
    main()
