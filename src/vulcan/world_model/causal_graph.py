"""
causal_graph.py - Causal DAG structure and operations for World Model
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
Integrated with comprehensive safety validation.
FIXED: API compatibility, circular import prevention, thread safety, complete NetworkX integration
"""

import copy
import heapq
import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path as FilePath
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Lazy import safety validator to prevent circular dependency
# DO NOT import at module level - import inside __init__ instead
SAFETY_VALIDATOR_AVAILABLE = False
EnhancedSafetyValidator = None
SafetyConfig = None

logger = logging.getLogger(__name__)


def _lazy_import_safety_validator():
    """Lazy import safety validator to avoid circular dependency"""
    global SAFETY_VALIDATOR_AVAILABLE, EnhancedSafetyValidator, SafetyConfig

    if EnhancedSafetyValidator is not None:
        return  # Already imported

    try:
        from ..safety.safety_types import SafetyConfig as SC
        from ..safety.safety_validator import EnhancedSafetyValidator as ESV

        EnhancedSafetyValidator = ESV
        SafetyConfig = SC
        SAFETY_VALIDATOR_AVAILABLE = True
        logger.info("Safety validator lazy loaded successfully")
    except ImportError as e:
        SAFETY_VALIDATOR_AVAILABLE = False
        logger.warning(f"Safety validator not available: {e}")
    except Exception as e:
        SAFETY_VALIDATOR_AVAILABLE = False
        logger.error(f"Failed to lazy load safety validator: {e}")


# Protected imports with fallbacks
try:
    pass

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, using fallback implementations")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning(
        "networkx not available, using comprehensive fallback graph implementation"
    )

    # Comprehensive fallback graph implementation
    class SimpleDiGraph:
        """
        Complete directed graph implementation for when NetworkX is not available.
        Implements all necessary graph algorithms including Tarjan's SCC and Dijkstra's shortest path.
        """

        def __init__(self):
            self.nodes_dict = {}
            self.edges_dict = defaultdict(list)
            self.edge_attrs = {}
            self.in_edges_dict = defaultdict(list)
            self._lock = threading.Lock()

        def add_node(self, node, **attrs):
            """Add a node with attributes"""
            with self._lock:
                if node not in self.nodes_dict:
                    self.nodes_dict[node] = attrs
                else:
                    self.nodes_dict[node].update(attrs)

        def add_edge(self, u, v, **attrs):
            """Add an edge with attributes"""
            with self._lock:
                if u not in self.nodes_dict:
                    self.add_node(u)
                if v not in self.nodes_dict:
                    self.add_node(v)
                if v not in self.edges_dict[u]:
                    self.edges_dict[u].append(v)
                    self.in_edges_dict[v].append(u)
                self.edge_attrs[(u, v)] = attrs

        def remove_edge(self, u, v):
            """Remove an edge"""
            with self._lock:
                if u in self.edges_dict and v in self.edges_dict[u]:
                    self.edges_dict[u].remove(v)
                if v in self.in_edges_dict and u in self.in_edges_dict[v]:
                    self.in_edges_dict[v].remove(u)
                if (u, v) in self.edge_attrs:
                    del self.edge_attrs[(u, v)]

        def remove_node(self, node):
            """Remove a node and all its edges"""
            with self._lock:
                if node not in self.nodes_dict:
                    return

                # Remove all outgoing edges
                for target in list(self.edges_dict.get(node, [])):
                    self.remove_edge(node, target)

                # Remove all incoming edges
                for source in list(self.in_edges_dict.get(node, [])):
                    self.remove_edge(source, node)

                # Remove node
                del self.nodes_dict[node]
                if node in self.edges_dict:
                    del self.edges_dict[node]
                if node in self.in_edges_dict:
                    del self.in_edges_dict[node]

        def has_edge(self, u, v):
            """Check if edge exists"""
            with self._lock:
                return u in self.edges_dict and v in self.edges_dict[u]

        def get_edge_data(self, u, v, default=None):
            """Get edge attributes"""
            with self._lock:
                return self.edge_attrs.get((u, v), default)

        def nodes(self, data=False):
            """Get all nodes"""
            with self._lock:
                if data:
                    return list((n, attrs) for n, attrs in self.nodes_dict.items()]
                return list(self.nodes_dict.keys())

        def edges(self, data=False):
            """Get all edges"""
            with self._lock:
                if data:
                    edges = []
                    for u, vs in self.edges_dict.items():
                        for v in vs:
                            edges.append((u, v, self.edge_attrs.get((u, v), {})))
                    return edges
                else:
                    edges = []
                    for u, vs in self.edges_dict.items():
                        for v in vs:
                            edges.append((u, v))
                    return edges

        def successors(self, node):
            """Get successor nodes (outgoing edges)"""
            with self._lock:
                return list(self.edges_dict.get(node, []))

        def predecessors(self, node):
            """Get predecessor nodes (incoming edges)"""
            with self._lock:
                return list(self.in_edges_dict.get(node, []))

        def has_node(self, node):
            """Check if node exists"""
            with self._lock:
                return node in self.nodes_dict

        def number_of_nodes(self):
            """Get number of nodes"""
            with self._lock:
                return len(self.nodes_dict)

        def number_of_edges(self):
            """Get number of edges"""
            with self._lock:
                return sum(len(vs) for vs in self.edges_dict.values())

        def out_degree(self, node):
            """Get out-degree of node"""
            with self._lock:
                return len(self.edges_dict.get(node, []))

        def in_degree(self, node):
            """Get in-degree of node"""
            with self._lock:
                return len(self.in_edges_dict.get(node, []))

        def degree(self, node):
            """Get total degree of node"""
            return self.in_degree(node) + self.out_degree(node)

        def subgraph(self, nodes):
            """Create a subgraph with given nodes"""
            sub = SimpleDiGraph()
            with self._lock:
                for node in nodes:
                    if node in self.nodes_dict:
                        sub.add_node(node, **self.nodes_dict[node])

                for node in nodes:
                    for successor in self.edges_dict.get(node, []):
                        if successor in nodes:
                            sub.add_edge(
                                node,
                                successor,
                                **self.edge_attrs.get((node, successor), {}),
                            )

            return sub

        def copy(self):
            """Create a deep copy of the graph"""
            new_graph = SimpleDiGraph()
            with self._lock:
                new_graph.nodes_dict = copy.deepcopy(self.nodes_dict)
                new_graph.edges_dict = copy.deepcopy(self.edges_dict)
                new_graph.edge_attrs = copy.deepcopy(self.edge_attrs)
                new_graph.in_edges_dict = copy.deepcopy(self.in_edges_dict)
            return new_graph

        def reverse(self):
            """Return a reversed copy of the graph"""
            rev = SimpleDiGraph()
            with self._lock:
                for node in self.nodes_dict:
                    rev.add_node(node, **self.nodes_dict[node])

                for u, vs in self.edges_dict.items():
                    for v in vs:
                        rev.add_edge(v, u, **self.edge_attrs.get((u, v), {}))

            return rev

    # Complete NetworkX replacement with all algorithms
    class MockNX:
        """
        Complete NetworkX replacement implementing all necessary graph algorithms.
        Includes Tarjan's algorithm for SCCs, cycle detection, topological sort, and Dijkstra's algorithm.
        """

        DiGraph = SimpleDiGraph

        @staticmethod
        def is_directed_acyclic_graph(graph):
            """
            Check if graph is a DAG using DFS-based cycle detection.
            Time complexity: O(V + E)
            """
            visited = set()
            rec_stack = set()

            def has_cycle_from(node):
                visited.add(node)
                rec_stack.add(node)

                for neighbor in graph.edges_dict.get(node, []):
                    if neighbor not in visited:
                        if has_cycle_from(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            for node in graph.nodes_dict:
                if node not in visited:
                    if has_cycle_from(node):
                        return False
            return True

        @staticmethod
        def strongly_connected_components(graph):
            """
            Find strongly connected components using Tarjan's algorithm.
            Time complexity: O(V + E)
            Returns: Generator of sets of nodes in each SCC
            """
            index_counter = [0]
            stack = []
            lowlinks = {}
            index = {}
            on_stack = set()
            sccs = []

            def strongconnect(node):
                index[node] = index_counter[0]
                lowlinks[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)

                for successor in graph.edges_dict.get(node, []):
                    if successor not in index:
                        strongconnect(successor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                    elif successor in on_stack:
                        lowlinks[node] = min(lowlinks[node], index[successor])

                if lowlinks[node] == index[node]:
                    component = set()
                    while True:
                        successor = stack.pop()
                        on_stack.remove(successor)
                        component.add(successor)
                        if successor == node:
                            break
                    sccs.append(component)

            for node in graph.nodes_dict:
                if node not in index:
                    strongconnect(node)

            return iter(sccs)

        @staticmethod
        def dag_longest_path_length(graph):
            """
            Find longest path length in DAG using dynamic programming.
            Time complexity: O(V + E)
            Returns: Number of edges in longest path (not weighted sum)
            """
            if not MockNX.is_directed_acyclic_graph(graph):
                raise ValueError("Graph contains cycles")

            if not graph.nodes_dict:
                return 0

            # Compute in-degrees
            in_degree = defaultdict(int)
            for u in graph.edges_dict:
                for v in graph.edges_dict[u]:
                    in_degree[v] += 1

            # Initialize queue with nodes of in-degree 0
            queue = deque([n for n in graph.nodes_dict if in_degree[n] == 0])
            topo_order = []

            # Topological sort
            while queue:
                node = queue.popleft()
                topo_order.append(node)
                for neighbor in graph.edges_dict.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if len(topo_order) != len(graph.nodes_dict):
                raise ValueError("Graph has cycles")

            # Dynamic programming to find longest path (count edges, not weights)
            dist = defaultdict(int)
            for node in topo_order:
                for neighbor in graph.edges_dict.get(node, []):
                    dist[neighbor] = max(dist[neighbor], dist[node] + 1)

            return max(dist.values()) if dist else 0

        @staticmethod
        def topological_sort(graph):
            """
            Topological sort using Kahn's algorithm.
            Time complexity: O(V + E)
            Returns: List of nodes in topological order
            """
            if not MockNX.is_directed_acyclic_graph(graph):
                raise ValueError("Graph contains cycles")

            in_degree = defaultdict(int)
            for u in graph.edges_dict:
                for v in graph.edges_dict[u]:
                    in_degree[v] += 1

            queue = deque([n for n in graph.nodes_dict if in_degree[n] == 0])
            result = []

            while queue:
                node = queue.popleft()
                result.append(node)
                for neighbor in graph.edges_dict.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if len(result) != len(graph.nodes_dict):
                raise ValueError("Graph has cycles")

            return result

        @staticmethod
        def simple_cycles(graph):
            """
            Find all simple cycles using Johnson's algorithm (simplified).
            Time complexity: O((V + E)(C + 1)) where C is number of cycles
            Returns: List of cycles (each cycle is a list of nodes)
            """

            def unblock(node, blocked, block_map):
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current in blocked:
                        blocked.remove(current)
                        stack.extend(block_map[current])
                        block_map[current].clear()

            def find_cycles_from(start, node, stack, blocked, block_map, sccs):
                cycles_found = []
                stack.append(node)
                blocked.add(node)
                closed = False

                for neighbor in graph.edges_dict.get(node, []):
                    if neighbor == start:
                        cycles_found.append(stack[:])
                        closed = True
                    elif neighbor not in blocked and neighbor in sccs:
                        result = find_cycles_from(
                            start, neighbor, stack, blocked, block_map, sccs
                        )
                        cycles_found.extend(result)
                        if result:
                            closed = True

                if closed:
                    unblock(node, blocked, block_map)
                else:
                    for neighbor in graph.edges_dict.get(node, []):
                        if neighbor in sccs:
                            if node not in block_map[neighbor]:
                                block_map[neighbor].add(node)

                stack.pop()
                return cycles_found

            all_cycles = []
            sccs = list(MockNX.strongly_connected_components(graph))

            for scc in sccs:
                if len(scc) == 1:
                    node = next(iter(scc))
                    if node in graph.edges_dict.get(node, []):
                        all_cycles.append([node])
                    continue

                subgraph_nodes = sorted(scc)
                for start in subgraph_nodes:
                    blocked = set()
                    block_map = defaultdict(set)
                    cycles = find_cycles_from(start, start, [], blocked, block_map, scc)
                    all_cycles.extend(cycles)

            return all_cycles

        @staticmethod
        def d_separated(graph, x, y, z):
            """
            Check if x and y are d-separated given conditioning set z using Bayes-Ball algorithm.
            Time complexity: O(V + E)
            """
            if not isinstance(x, set):
                x = {x}
            if not isinstance(y, set):
                y = {y}
            if not isinstance(z, set):
                z = set(z) if z else set()

            # Bayes-Ball algorithm
            visited_up = set()
            visited_down = set()
            schedule = []

            # Initialize from x nodes going up and down
            for node in x:
                schedule.append((node, "up"))
                schedule.append((node, "down"))

            while schedule:
                node, direction = schedule.pop()

                if direction == "up":
                    if (node, "up") in visited_up:
                        continue
                    visited_up.add((node, "up"))

                    if node not in z and node not in x:
                        for parent in graph.in_edges_dict.get(node, []):
                            schedule.append((parent, "up"))
                        for child in graph.edges_dict.get(node, []):
                            schedule.append((child, "down"))

                    if node in z:
                        for parent in graph.in_edges_dict.get(node, []):
                            schedule.append((parent, "up"))

                else:  # direction == 'down'
                    if (node, "down") in visited_down:
                        continue
                    visited_down.add((node, "down"))

                    if node not in z:
                        for child in graph.edges_dict.get(node, []):
                            schedule.append((child, "down"))

                    if node in z:
                        for child in graph.edges_dict.get(node, []):
                            schedule.append((child, "up"))

            # Check if any y node was reached
            for y_node in y:
                if (y_node, "up") in visited_up or (y_node, "down") in visited_down:
                    return False

            return True

        @staticmethod
        def dijkstra_path(graph, source, target, weight="weight"):
            """
            Find shortest path using Dijkstra's algorithm.
            Time complexity: O((V + E) log V)
            Returns: List of nodes in shortest path
            """
            if source not in graph.nodes_dict or target not in graph.nodes_dict:
                raise ValueError("Source or target not in graph")

            distances = {node: float("inf") for node in graph.nodes_dict}
            distances[source] = 0
            previous = {}
            pq = [(0, source)]
            visited = set()

            while pq:
                current_dist, current = heapq.heappop(pq)

                if current in visited:
                    continue

                visited.add(current)

                if current == target:
                    break

                if current_dist > distances[current]:
                    continue

                for neighbor in graph.edges_dict.get(current, []):
                    edge_data = graph.edge_attrs.get((current, neighbor), {})
                    edge_weight = edge_data.get(weight, 1)

                    distance = current_dist + edge_weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (distance, neighbor))

            if target not in previous and target != source:
                raise ValueError(f"No path from {source} to {target}")

            # Reconstruct path
            path = []
            current = target
            while current in previous:
                path.append(current)
                current = previous[current]
            path.append(source)
            path.reverse()

            return path

        @staticmethod
        def dijkstra_path_length(graph, source, target, weight="weight"):
            """
            Get shortest path length using Dijkstra's algorithm.
            Time complexity: O((V + E) log V)
            Returns: Path length (sum of weights)
            """
            path = MockNX.dijkstra_path(graph, source, target, weight)

            total_weight = 0
            for i in range(len(path) - 1):
                edge_data = graph.edge_attrs.get((path[i], path[i + 1]), {})
                total_weight += edge_data.get(weight, 1)

            return total_weight

        @staticmethod
        def all_simple_paths(graph, source, target, cutoff=None):
            """
            Find all simple paths from source to target.
            Time complexity: O(V!) in worst case
            Returns: Generator of paths (each path is a list of nodes)
            """
            if source not in graph.nodes_dict or target not in graph.nodes_dict:
                return

            if cutoff is None:
                cutoff = len(graph.nodes_dict)

            visited = {source}
            stack = [(source, iter(graph.edges_dict.get(source, [])))]

            while stack:
                parent, children = stack[-1]

                try:
                    child = next(children)

                    if child == target:
                        yield [node for node, _ in stack] + [target]

                    elif len(stack) < cutoff:
                        if child not in visited:
                            visited.add(child)
                            stack.append((child, iter(graph.edges_dict.get(child, []))))

                except StopIteration:
                    stack.pop()
                    if stack:
                        visited.remove(parent)

        @staticmethod
        def shortest_path(graph, source, target, weight=None):
            """
            Find shortest path (unweighted BFS or weighted Dijkstra).
            Returns: List of nodes in shortest path
            """
            if weight is None:
                # Unweighted BFS
                if source not in graph.nodes_dict or target not in graph.nodes_dict:
                    raise ValueError("Source or target not in graph")

                if source == target:
                    return [source]

                visited = {source}
                queue = deque([(source, [source])])

                while queue:
                    node, path = queue.popleft()

                    for neighbor in graph.edges_dict.get(node, []):
                        if neighbor == target:
                            return path + [neighbor]

                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))

                raise ValueError(f"No path from {source} to {target}")
            else:
                # Weighted Dijkstra
                return MockNX.dijkstra_path(graph, source, target, weight)

        @staticmethod
        def has_path(graph, source, target):
            """
            Check if path exists from source to target using BFS.
            Time complexity: O(V + E)
            """
            if source not in graph.nodes_dict or target not in graph.nodes_dict:
                return False

            if source == target:
                return True

            visited = {source}
            queue = deque([source])

            while queue:
                node = queue.popleft()

                for neighbor in graph.edges_dict.get(node, []):
                    if neighbor == target:
                        return True

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            return False

        @staticmethod
        def ancestors(graph, node):
            """
            Get all ancestors (nodes with paths to given node).
            Time complexity: O(V + E)
            """
            if node not in graph.nodes_dict:
                return set()

            ancestors = set()
            visited = set()
            queue = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                for parent in graph.in_edges_dict.get(current, []):
                    ancestors.add(parent)
                    queue.append(parent)

            return ancestors

        @staticmethod
        def descendants(graph, node):
            """
            Get all descendants (nodes reachable from given node).
            Time complexity: O(V + E)
            """
            if node not in graph.nodes_dict:
                return set()

            descendants = set()
            visited = set()
            queue = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                for child in graph.edges_dict.get(current, []):
                    descendants.add(child)
                    queue.append(child)

            return descendants

    nx = MockNX()


class EvidenceType(Enum):
    """Types of evidence for causal relationships"""

    INTERVENTION = "intervention"  # Experimental intervention
    CORRELATION = "correlation"  # Observational correlation
    EXPERT = "expert"  # Expert knowledge
    THEORETICAL = "theoretical"  # Theoretical reasoning
    MIXED = "mixed"  # Multiple evidence types


@dataclass
class ProbabilityDistribution:
    """Probability distribution for stochastic edges"""

    distribution_type: str  # "normal", "uniform", "beta", etc.
    parameters: Dict[str, float]

    def sample(self) -> float:
        """Sample from the distribution"""
        if self.distribution_type == "normal":
            return np.random.normal(
                self.parameters.get("mean", 0), self.parameters.get("std", 1)
            )
        elif self.distribution_type == "uniform":
            return np.random.uniform(
                self.parameters.get("low", 0), self.parameters.get("high", 1)
            )
        elif self.distribution_type == "beta":
            return np.random.beta(
                self.parameters.get("alpha", 1), self.parameters.get("beta", 1)
            )
        else:
            return np.random.random()

    def mean(self) -> float:
        """Get mean of distribution"""
        if self.distribution_type == "normal":
            return self.parameters.get("mean", 0)
        elif self.distribution_type == "uniform":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            return (low + high) / 2
        elif self.distribution_type == "beta":
            alpha = self.parameters.get("alpha", 1)
            beta = self.parameters.get("beta", 1)
            return alpha / (alpha + beta)
        else:
            return 0.5


@dataclass
class CausalEdge:
    """Single causal relationship"""

    cause: str
    effect: str
    strength: float
    evidence_type: EvidenceType
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    is_stochastic: bool = False
    probability_distribution: Optional[ProbabilityDistribution] = None

    def get_edge_key(self) -> Tuple[str, str]:
        """Get unique edge identifier"""
        return (self.cause, self.effect)

    def sample_strength(self) -> float:
        """Sample strength for stochastic edges"""
        if self.is_stochastic and self.probability_distribution:
            return self.probability_distribution.sample()
        return self.strength


@dataclass
class CausalPath:
    """
    Causal path structure - defined here to prevent circular dependency
    """

    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # (from, to, strength)
    total_strength: float = 1.0
    confidence: float = 1.0
    evidence_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Get path length"""
        return len(self.nodes)

    def get_strengths(self) -> List[float]:
        """Get list of edge strengths"""
        return [strength for _, _, strength in self.edges]


class LRUCache:
    """Thread-safe LRU cache implementation"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        """Put value in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()

    def __len__(self):
        """Get cache size"""
        with self.lock:
            return len(self.cache)


class GraphStructure:
    """Manages the graph structure with complete thread safety"""

    def __init__(self):
        self.nodes = set()
        self.edges = {}  # (cause, effect) -> CausalEdge
        self.adjacency_list = defaultdict(set)  # cause -> set of effects
        self.reverse_adjacency = defaultdict(set)  # effect -> set of causes

        # NetworkX graph for advanced operations
        if NETWORKX_AVAILABLE:
            self.nx_graph = nx.DiGraph()
        else:
            self.nx_graph = SimpleDiGraph()

        self.lock = threading.RLock()

    def add_edge(self, edge: CausalEdge) -> bool:
        """Add edge to structure with full thread safety"""

        with self.lock:
            edge_key = edge.get_edge_key()

            # Add to structures
            self.edges[edge_key] = edge
            self.nodes.add(edge.cause)
            self.nodes.add(edge.effect)
            self.adjacency_list[edge.cause].add(edge.effect)
            self.reverse_adjacency[edge.effect].add(edge.cause)

            # Add to NetworkX graph with weight
            self.nx_graph.add_edge(
                edge.cause, edge.effect, weight=edge.strength, edge_object=edge
            )

            return True

    def remove_edge(self, cause: str, effect: str) -> bool:
        """Remove edge from structure with full thread safety"""

        with self.lock:
            edge_key = (cause, effect)

            if edge_key not in self.edges:
                return False

            # Remove from structures
            del self.edges[edge_key]
            self.adjacency_list[cause].discard(effect)
            self.reverse_adjacency[effect].discard(cause)

            # Remove isolated nodes
            if not self.adjacency_list[cause] and cause not in self.reverse_adjacency:
                self.nodes.discard(cause)
            if not self.reverse_adjacency[effect] and effect not in self.adjacency_list:
                self.nodes.discard(effect)

            # Remove from NetworkX graph
            if self.nx_graph.has_edge(cause, effect):
                self.nx_graph.remove_edge(cause, effect)

            return True

    def update_edge_strength(
        self, cause: str, effect: str, new_strength: float
    ) -> bool:
        """Update edge strength without removing and re-adding"""

        with self.lock:
            edge_key = (cause, effect)

            if edge_key not in self.edges:
                return False

            # Update edge strength
            edge = self.edges[edge_key]
            edge.strength = new_strength
            edge.timestamp = time.time()

            # Update NetworkX graph
            if self.nx_graph.has_edge(cause, effect):
                edge_data = self.nx_graph.get_edge_data(cause, effect)
                if edge_data:
                    edge_data["weight"] = new_strength

            return True

    def has_edge(self, cause: str, effect: str) -> bool:
        """Check if edge exists with thread safety"""

        with self.lock:
            return (cause, effect) in self.edges

    def get_edge(self, cause: str, effect: str) -> Optional[CausalEdge]:
        """Get edge object with thread safety"""

        with self.lock:
            edge = self.edges.get((cause, effect))
            return copy.deepcopy(edge) if edge else None

    def get_parents(self, node: str) -> Set[str]:
        """Get parent nodes with thread safety"""

        with self.lock:
            return self.reverse_adjacency.get(node, set()).copy()

    def get_children(self, node: str) -> Set[str]:
        """Get child nodes with thread safety"""

        with self.lock:
            return self.adjacency_list.get(node, set()).copy()

    def get_all_nodes(self) -> Set[str]:
        """Get all nodes with thread safety"""

        with self.lock:
            return self.nodes.copy()

    def get_all_edges(self) -> Dict[Tuple[str, str], CausalEdge]:
        """Get all edges with thread safety"""

        with self.lock:
            return {k: copy.deepcopy(v) for k, v in self.edges.items()}


class CycleDetector:
    """Detects and manages cycles with improved algorithms"""

    def __init__(self, structure: GraphStructure):
        self.structure = structure
        self.cycle_cache = LRUCache(capacity=1000)
        self.lock = threading.RLock()

    def would_create_cycle(self, cause: str, effect: str) -> bool:
        """Check if adding edge would create cycle with caching"""

        if cause == effect:
            return True

        with self.lock:
            # Check cache
            cache_key = (effect, cause)
            cached_result = self.cycle_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Check if there's already a path from effect to cause
            result = self._has_path(effect, cause)

            # Cache result
            self.cycle_cache.put(cache_key, result)

            return result

    def has_cycles(self) -> bool:
        """Check if graph has cycles using efficient algorithm"""

        with self.lock:
            if NETWORKX_AVAILABLE:
                try:
                    return not nx.is_directed_acyclic_graph(self.structure.nx_graph)
                except Exception as e:
                    logger.warning(f"NetworkX cycle check failed: {e}, using fallback")
                    return self._has_cycles_dfs()
            else:
                return not MockNX.is_directed_acyclic_graph(self.structure.nx_graph)

    def find_all_cycles(self) -> List[List[str]]:
        """Find all cycles in graph using efficient algorithm"""

        with self.lock:
            if NETWORKX_AVAILABLE:
                try:
                    return list(nx.simple_cycles(self.structure.nx_graph))
                except Exception as e:
                    logger.warning(
                        f"NetworkX cycle finding failed: {e}, using fallback"
                    )
                    return []
            else:
                return MockNX.simple_cycles(self.structure.nx_graph)

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components using Tarjan's algorithm"""

        with self.lock:
            if NETWORKX_AVAILABLE:
                try:
                    sccs = list(
                        nx.strongly_connected_components(self.structure.nx_graph)
                    )
                    return [scc for scc in sccs if len(scc) > 1]
                except Exception as e:
                    logger.warning(f"NetworkX SCC failed: {e}, using fallback")
                    sccs = self._tarjan_scc()
                    return [scc for scc in sccs if len(scc) > 1]
            else:
                sccs = list(
                    MockNX.strongly_connected_components(self.structure.nx_graph)
                )
                return [scc for scc in sccs if len(scc) > 1]

    def _tarjan_scc(self) -> List[Set[str]]:
        """
        Tarjan's algorithm for finding strongly connected components.
        Time complexity: O(V + E)
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        sccs = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            for successor in self.structure.adjacency_list.get(node, []):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    lowlinks[node] = min(lowlinks[node], index[successor])

            if lowlinks[node] == index[node]:
                component = set()
                while True:
                    successor = stack.pop()
                    on_stack.remove(successor)
                    component.add(successor)
                    if successor == node:
                        break
                sccs.append(component)

        for node in self.structure.nodes:
            if node not in index:
                strongconnect(node)

        return sccs

    def break_cycles_minimum_feedback(self) -> List[Tuple[str, str]]:
        """
        Break cycles by removing minimum feedback arc set.
        Uses greedy approach to minimize total weight of removed edges.
        """

        removed_edges = []

        with self.lock:
            while self.has_cycles():
                # EXAMINE: Find cycles
                cycles = self.find_all_cycles()
                if not cycles:
                    break

                # SELECT: Find minimum weight edge across all cycles
                min_edge = self._find_minimum_weight_edge_in_cycles(cycles)

                if min_edge:
                    # APPLY: Remove edge
                    self.structure.remove_edge(min_edge[0], min_edge[1])
                    removed_edges.append(min_edge)
                    self.cycle_cache.clear()

                    # REMEMBER: Log removal
                    logger.info(
                        f"Removed edge {min_edge[0]} -> {min_edge[1]} to break cycle"
                    )
                else:
                    # Safety break if we can't find an edge to remove
                    logger.error("Could not find edge to break cycles")
                    break

        return removed_edges

    def _has_path(self, source: str, target: str) -> bool:
        """Check if path exists from source to target using BFS"""

        if source not in self.structure.nodes or target not in self.structure.nodes:
            return False

        if source == target:
            return True

        visited = set()
        queue = deque([source])

        while queue:
            current = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            if current == target:
                return True

            for neighbor in self.structure.adjacency_list.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        return False

    def _has_cycles_dfs(self) -> bool:
        """Check for cycles using DFS with recursion stack"""

        visited = set()
        rec_stack = set()

        def visit(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.structure.adjacency_list.get(node, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.structure.nodes:
            if node not in visited:
                if visit(node):
                    return True

        return False

    def _find_minimum_weight_edge_in_cycles(
        self, cycles: List[List[str]]
    ) -> Optional[Tuple[str, str]]:
        """Find edge with minimum weight across all cycles"""

        min_edge = None
        min_weight = float("inf")

        for cycle in cycles:
            for i in range(len(cycle))
                j = (i + 1) % len(cycle)
                edge_key = (cycle[i], cycle[j])

                edge = self.structure.edges.get(edge_key)
                if edge and edge.strength < min_weight:
                    min_weight = edge.strength
                    min_edge = edge_key

        return min_edge


class PathFinder:
    """
    Finds paths in the graph with advanced algorithms including Dijkstra's shortest path
    """

    def __init__(self, structure: GraphStructure):
        self.structure = structure
        self.path_cache = LRUCache(capacity=5000)
        self.lock = threading.RLock()

    def find_paths(
        self, source: str, targets: Union[str, List[str]], max_length: int = 5
    ) -> List[CausalPath]:
        """
        Find causal paths from source to targets using BFS.
        Returns CausalPath objects with full edge information.
        """

        if isinstance(targets, str):
            targets = [targets]

        with self.lock:
            # Check cache
            cache_key = (source, tuple(sorted(targets)), max_length)
            cached_result = self.path_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # EXAMINE & SELECT: Find paths
            all_paths = []

            for target in targets:
                if source in self.structure.nodes and target in self.structure.nodes:
                    # Find raw paths (node sequences)
                    raw_paths = self._find_paths_bfs(source, target, max_length)

                    # Convert to CausalPath objects
                    for node_sequence in raw_paths:
                        causal_path = self._convert_to_causal_path(node_sequence)
                        if causal_path:
                            all_paths.append(causal_path)

            # REMEMBER: Cache result
            self.path_cache.put(cache_key, all_paths)

            return all_paths

    def find_all_paths(
        self, source: str, targets: Union[str, List[str]]
    ) -> List[CausalPath]:
        """
        Find all causal paths from source to targets (no length limit).
        Returns CausalPath objects.
        """

        if isinstance(targets, str):
            targets = [targets]

        with self.lock:
            all_paths = []

            for target in targets:
                if source in self.structure.nodes and target in self.structure.nodes:
                    if NETWORKX_AVAILABLE:
                        try:
                            # Use NetworkX for efficient all paths search
                            for path in nx.all_simple_paths(
                                self.structure.nx_graph, source, target
                            ):
                                causal_path = self._convert_to_causal_path(path)
                                if causal_path:
                                    all_paths.append(causal_path)
                        except Exception as e:
                            logger.warning(
                                f"NetworkX all_simple_paths failed: {e}, using fallback"
                            )
                            raw_paths = self._find_paths_bfs(
                                source, target, len(self.structure.nodes)
                            )
                            for node_sequence in raw_paths:
                                causal_path = self._convert_to_causal_path(
                                    node_sequence
                                )
                                if causal_path:
                                    all_paths.append(causal_path)
                    else:
                        # Use fallback implementation
                        for path in MockNX.all_simple_paths(
                            self.structure.nx_graph, source, target
                        ):
                            causal_path = self._convert_to_causal_path(path)
                            if causal_path:
                                all_paths.append(causal_path)

            return all_paths

    def find_shortest_path(
        self, source: str, target: str, weighted: bool = True
    ) -> Optional[CausalPath]:
        """
        Find shortest path using Dijkstra's algorithm (if weighted) or BFS (if unweighted).
        Returns single CausalPath object.
        """

        with self.lock:
            if source not in self.structure.nodes or target not in self.structure.nodes:
                return None

            try:
                if weighted:
                    # Use Dijkstra's algorithm for weighted shortest path
                    if NETWORKX_AVAILABLE:
                        path = nx.shortest_path(
                            self.structure.nx_graph, source, target, weight="weight"
                        )
                    else:
                        path = MockNX.dijkstra_path(
                            self.structure.nx_graph, source, target, weight="weight"
                        )
                else:
                    # Use BFS for unweighted shortest path
                    if NETWORKX_AVAILABLE:
                        path = nx.shortest_path(self.structure.nx_graph, source, target)
                    else:
                        path = MockNX.shortest_path(
                            self.structure.nx_graph, source, target
                        )

                return self._convert_to_causal_path(path)

            except Exception as e:
                logger.warning(f"Shortest path finding failed: {e}")
                return None

    def get_shortest_path_length(
        self, source: str, target: str, weighted: bool = True
    ) -> Optional[float]:
        """
        Get shortest path length (sum of edge weights).
        """

        path = self.find_shortest_path(source, target, weighted)
        if path:
            if weighted:
                return path.total_strength
            else:
                return len(path.nodes) - 1
        return None

    def _convert_to_causal_path(self, node_sequence: List[str]) -> Optional[CausalPath]:
        """Convert node sequence to CausalPath object with full edge information"""

        if len(node_sequence) < 2:
            return None

        edges = []
        total_strength = 1.0
        evidence_types = []

        # Extract edges and calculate total strength
        for i in range(len(node_sequence) - 1):
            from_node = node_sequence[i]
            to_node = node_sequence[i + 1]

            edge = self.structure.edges.get((from_node, to_node))
            if edge:
                edges.append((from_node, to_node, edge.strength))
                total_strength *= edge.strength
                evidence_types.append(edge.evidence_type.value)
            else:
                # Edge doesn't exist - path is invalid
                return None

        # Calculate confidence (using total strength as proxy)
        confidence = total_strength

        return CausalPath(
            nodes=node_sequence,
            edges=edges,
            total_strength=total_strength,
            confidence=confidence,
            evidence_types=evidence_types,
            metadata={"path_length": len(node_sequence)},
        )

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node (nodes with paths to this node)"""

        with self.lock:
            if node not in self.structure.nodes:
                return set()

            if NETWORKX_AVAILABLE:
                try:
                    return nx.ancestors(self.structure.nx_graph, node)
                except Exception as e:
                    logger.warning(f"NetworkX ancestors failed: {e}, using fallback")

            ancestors = set()
            visited = set()
            queue = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                parents = self.structure.get_parents(current)
                ancestors.update(parents)
                queue.extend(parents)

            return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node (nodes reachable from this node)"""

        with self.lock:
            if node not in self.structure.nodes:
                return set()

            if NETWORKX_AVAILABLE:
                try:
                    return nx.descendants(self.structure.nx_graph, node)
                except Exception as e:
                    logger.warning(f"NetworkX descendants failed: {e}, using fallback")

            descendants = set()
            visited = set()
            queue = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                children = self.structure.get_children(current)
                descendants.update(children)
                queue.extend(children)

            return descendants

    def get_longest_path_length(self) -> int:
        """
        Get length of longest path in DAG (counts edges, not weighted sum).
        Returns the number of edges in the longest path.
        """

        with self.lock:
            # Always use our own implementation to ensure we count edges, not weights
            # This avoids issues with NetworkX potentially using edge weights
            return self._longest_path_dfs()

    def _find_paths_bfs(
        self, source: str, target: str, max_length: int
    ) -> List[List[str]]:
        """Find paths using BFS - returns node sequences"""

        paths = []
        queue = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_length:
                continue

            if current == target:
                paths.append(path)
                continue

            for neighbor in self.structure.adjacency_list.get(current, []):
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def _longest_path_dfs(self) -> int:
        """
        Find longest path using DFS with memoization.
        Returns the number of edges in the longest path (not weighted sum).
        """

        if not self.structure.nodes:
            return 0

        memo = {}

        def dfs(node):
            if node in memo:
                return memo[node]

            max_length = 0
            for child in self.structure.adjacency_list.get(node, []):
                max_length = max(max_length, 1 + dfs(child))

            memo[node] = max_length
            return max_length

        return max(dfs(node) for node in self.structure.nodes)

    def clear_cache(self):
        """Clear path cache atomically"""

        with self.lock:
            self.path_cache.clear()


class DSeparationChecker:
    """Checks d-separation using Bayes-Ball algorithm"""

    def __init__(self, structure: GraphStructure, path_finder: PathFinder):
        self.structure = structure
        self.path_finder = path_finder
        self.lock = threading.RLock()

    def is_d_separated(self, x: str, y: str, conditioning_set: Set[str]) -> bool:
        """
        Check if x and y are d-separated given conditioning set using Bayes-Ball algorithm.
        Thread-safe implementation.
        """

        with self.lock:
            if x not in self.structure.nodes or y not in self.structure.nodes:
                return True

            if x == y:
                return False

            # Use NetworkX's d-separation algorithm if available
            if NETWORKX_AVAILABLE:
                try:
                    return nx.d_separated(
                        self.structure.nx_graph, {x}, {y}, conditioning_set
                    )
                except Exception as e:
                    logger.warning(f"NetworkX d_separated failed: {e}, using fallback")
                    return self._bayes_ball_d_separation(x, y, conditioning_set)
            else:
                return MockNX.d_separated(
                    self.structure.nx_graph, {x}, {y}, conditioning_set
                )

    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Get Markov blanket of a node (parents, children, and co-parents).
        Thread-safe implementation.
        """

        with self.lock:
            if node not in self.structure.nodes:
                return set()

            markov_blanket = set()

            # Add parents
            parents = self.structure.get_parents(node)
            markov_blanket.update(parents)

            # Add children
            children = self.structure.get_children(node)
            markov_blanket.update(children)

            # Add co-parents (other parents of children)
            for child in children:
                co_parents = self.structure.get_parents(child)
                markov_blanket.update(co_parents)

            # Remove the node itself
            markov_blanket.discard(node)

            return markov_blanket

    def _bayes_ball_d_separation(
        self, x: str, y: str, conditioning_set: Set[str]
    ) -> bool:
        """
        Bayes-Ball algorithm for d-separation.
        More accurate than simple path blocking.
        """

        visited_up = set()
        visited_down = set()
        schedule = [(x, "up"), (x, "down")]

        while schedule:
            node, direction = schedule.pop()

            if direction == "up":
                if (node, "up") in visited_up:
                    continue
                visited_up.add((node, "up"))

                if node not in conditioning_set and node != x:
                    for parent in self.structure.get_parents(node):
                        schedule.append((parent, "up"))
                    for child in self.structure.get_children(node):
                        schedule.append((child, "down"))

                if node in conditioning_set:
                    for parent in self.structure.get_parents(node):
                        schedule.append((parent, "up"))

            else:  # direction == 'down'
                if (node, "down") in visited_down:
                    continue
                visited_down.add((node, "down"))

                if node not in conditioning_set:
                    for child in self.structure.get_children(node):
                        schedule.append((child, "down"))

                if node in conditioning_set:
                    for child in self.structure.get_children(node):
                        schedule.append((child, "up"))

        # Check if y was reached
        return not ((y, "up") in visited_up or (y, "down") in visited_down)


class TopologicalSorter:
    """Handles topological sorting with thread safety"""

    def __init__(self, structure: GraphStructure):
        self.structure = structure
        self.lock = threading.RLock()

    def topological_sort(self) -> List[str]:
        """
        Get topological ordering of nodes using Kahn's algorithm.
        Thread-safe implementation.
        """

        with self.lock:
            if NETWORKX_AVAILABLE:
                try:
                    return list(nx.topological_sort(self.structure.nx_graph))
                except Exception as e:
                    logger.warning(
                        f"NetworkX topological_sort failed: {e}, using fallback"
                    )
                    return self._topological_sort_kahn()
            else:
                try:
                    return MockNX.topological_sort(self.structure.nx_graph)
                except Exception as e:
                    logger.warning(
                        f"MockNX topological_sort failed: {e}, using fallback"
                    )
                    return self._topological_sort_kahn()

    def _topological_sort_kahn(self) -> List[str]:
        """Kahn's algorithm for topological sorting"""

        in_degree = defaultdict(int)

        for node in self.structure.nodes:
            for child in self.structure.adjacency_list.get(node, []):
                in_degree[child] += 1

        queue = deque([n for n in self.structure.nodes if in_degree[n] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for child in self.structure.adjacency_list.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.structure.nodes):
            raise ValueError("Graph has cycles - cannot perform topological sort")

        return result


class CausalDAG:
    """
    Directed acyclic graph for causal relationships.
    Complete implementation with safety validation, advanced algorithms, and full thread safety.
    """

    def __init__(self, safety_config: Optional[Dict[str, Any]] = None):
        """
        Initialize causal DAG with optional safety configuration.

        Args:
            safety_config: Optional safety configuration dictionary
        """

        # Lazy import safety validator
        _lazy_import_safety_validator()

        # Initialize safety validator
        if SAFETY_VALIDATOR_AVAILABLE and EnhancedSafetyValidator is not None:
            try:
                if isinstance(safety_config, dict) and safety_config:
                    self.safety_validator = EnhancedSafetyValidator(
                        SafetyConfig.from_dict(safety_config)
                    )
                else:
                    self.safety_validator = EnhancedSafetyValidator()
                logger.info("CausalDAG: Safety validator initialized")
            except Exception as e:
                self.safety_validator = None
                logger.warning(f"CausalDAG: Failed to initialize safety validator: {e}")
        else:
            self.safety_validator = None
            logger.warning(
                "CausalDAG: Safety validator not available - operating without safety checks"
            )

        # Initialize components with separation of concerns
        self.structure = GraphStructure()
        self.cycle_detector = CycleDetector(self.structure)
        self.path_finder = PathFinder(self.structure)
        self.d_separator = DSeparationChecker(self.structure, self.path_finder)
        self.topological_sorter = TopologicalSorter(self.structure)

        # Statistics tracking
        self.edge_count = 0
        self.cycle_checks = 0
        self.path_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Safety tracking
        self.safety_blocks = defaultdict(int)
        self.safety_corrections = defaultdict(int)

        # Thread safety - use RLock for reentrant locking
        self.lock = threading.RLock()

        logger.info("CausalDAG initialized with complete implementation")

    def add_edge(
        self,
        cause: str,
        effect: str,
        strength: float,
        evidence_type: Union[str, EvidenceType],
        confidence_interval: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add causal edge to graph with safety validation and cycle prevention.

        Args:
            cause: Source node
            effect: Target node
            strength: Edge strength (0-1)
            evidence_type: Type of evidence (intervention, correlation, etc.)
            confidence_interval: Optional confidence interval
            metadata: Optional metadata dictionary

        Returns:
            True if edge was added successfully, False otherwise
        """

        with self.lock:
            # SAFETY: Validate edge parameters
            if self.safety_validator:
                edge_check = self._validate_edge_safety(cause, effect, strength)
                if not edge_check["safe"]:
                    logger.warning(
                        f"Blocked unsafe edge {cause} -> {effect}: {edge_check['reason']}"
                    )
                    self.safety_blocks["edge"] += 1
                    return False

            # EXAMINE: Check for cycle
            if self.cycle_detector.would_create_cycle(cause, effect):
                logger.warning(f"Edge {cause} -> {effect} would create cycle")
                return False

            # SAFETY: Check graph complexity limits
            if self.safety_validator and len(self.structure.nodes) >= 10000:
                logger.warning("Maximum node count reached, blocking edge addition")
                self.safety_blocks["max_nodes"] += 1
                return False

            # SELECT: Convert evidence type if string
            if isinstance(evidence_type, str):
                try:
                    evidence_type = EvidenceType(evidence_type)
                except ValueError:
                    evidence_type = EvidenceType.CORRELATION

            # APPLY: Create and add edge
            edge = CausalEdge(
                cause=cause,
                effect=effect,
                strength=strength,
                evidence_type=evidence_type,
                confidence_interval=confidence_interval
                or (strength * 0.9, min(1.0, strength * 1.1)),
                metadata=metadata or {},
            )

            self.structure.add_edge(edge)

            # Invalidate caches atomically
            self._invalidate_caches()

            # REMEMBER: Update statistics
            self.edge_count += 1
            logger.debug(f"Added edge {cause} -> {effect} (strength={strength:.2f})")

            return True

    def add_stochastic_edge(
        self,
        cause: str,
        effect: str,
        probability_distribution: Union[ProbabilityDistribution, Dict[str, Any]],
    ) -> bool:
        """
        Add stochastic causal edge with probability distribution.

        Args:
            cause: Source node
            effect: Target node
            probability_distribution: ProbabilityDistribution object or dict with type and parameters

        Returns:
            True if edge was added successfully, False otherwise
        """

        with self.lock:
            # Convert dict to ProbabilityDistribution
            if isinstance(probability_distribution, dict):
                probability_distribution = ProbabilityDistribution(
                    distribution_type=probability_distribution.get("type", "normal"),
                    parameters=probability_distribution.get("parameters", {}),
                )

            # SAFETY: Validate distribution parameters
            if self.safety_validator:
                dist_check = self._validate_distribution_safety(
                    probability_distribution
                )
                if not dist_check["safe"]:
                    logger.warning(
                        f"Blocked unsafe stochastic edge {cause} -> {effect}: {dist_check['reason']}"
                    )
                    self.safety_blocks["stochastic_edge"] += 1
                    return False

            # EXAMINE: Check for cycle
            if self.cycle_detector.would_create_cycle(cause, effect):
                logger.warning(
                    f"Stochastic edge {cause} -> {effect} would create cycle"
                )
                return False

            # APPLY: Create stochastic edge
            edge = CausalEdge(
                cause=cause,
                effect=effect,
                strength=probability_distribution.mean(),
                evidence_type=EvidenceType.INTERVENTION,
                is_stochastic=True,
                probability_distribution=probability_distribution,
            )

            self.structure.add_edge(edge)

            # Invalidate caches atomically
            self._invalidate_caches()

            # REMEMBER: Update statistics
            self.edge_count += 1
            logger.debug(f"Added stochastic edge {cause} -> {effect}")

            return True

    def remove_edge(self, cause: str, effect: str) -> bool:
        """
        Remove causal edge from graph.

        Args:
            cause: Source node
            effect: Target node

        Returns:
            True if edge was removed, False if edge didn't exist
        """

        with self.lock:
            result = self.structure.remove_edge(cause, effect)

            if result:
                # Invalidate caches atomically
                self._invalidate_caches()

                self.edge_count -= 1
                logger.debug(f"Removed edge {cause} -> {effect}")

            return result

    def update_edge_strength(
        self, cause: str, effect: str, new_strength: float
    ) -> bool:
        """
        Update edge strength without removing and re-adding edge.
        More efficient than remove + add for strength updates.

        Args:
            cause: Source node
            effect: Target node
            new_strength: New strength value (0-1)

        Returns:
            True if edge was updated, False if edge doesn't exist
        """

        with self.lock:
            # SAFETY: Validate new strength
            if self.safety_validator:
                if (
                    not np.isfinite(new_strength)
                    or new_strength < 0
                    or new_strength > 1
                ):
                    logger.warning(f"Invalid strength value: {new_strength}")
                    return False

            result = self.structure.update_edge_strength(cause, effect, new_strength)

            if result:
                # Invalidate caches atomically
                self._invalidate_caches()
                logger.debug(
                    f"Updated edge {cause} -> {effect} strength to {new_strength:.2f}"
                )

            return result

    def batch_add_edges(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple edges in batch with deferred cache invalidation.
        More efficient than adding edges one at a time.

        Args:
            edges: List of edge dictionaries with keys: cause, effect, strength, evidence_type

        Returns:
            Dictionary with success count and failed edges
        """

        with self.lock:
            success_count = 0
            failed_edges = []

            for edge_dict in edges:
                try:
                    result = self.add_edge(
                        cause=edge_dict["cause"],
                        effect=edge_dict["effect"],
                        strength=edge_dict["strength"],
                        evidence_type=edge_dict.get("evidence_type", "correlation"),
                        confidence_interval=edge_dict.get("confidence_interval"),
                        metadata=edge_dict.get("metadata"),
                    )

                    if result:
                        success_count += 1
                    else:
                        failed_edges.append(edge_dict)

                except Exception as e:
                    logger.error(f"Failed to add edge: {e}")
                    failed_edges.append(edge_dict)

            # Single cache invalidation at the end
            self._invalidate_caches()

            return {
                "success_count": success_count,
                "failed_count": len(failed_edges),
                "failed_edges": failed_edges,
            }

    def _validate_edge_safety(
        self, cause: str, effect: str, strength: float
    ) -> Dict[str, Any]:
        """
        Validate edge parameters for safety.

        Args:
            cause: Source node
            effect: Target node
            strength: Edge strength

        Returns:
            Dictionary with 'safe' boolean and optional 'reason' string
        """

        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check node names
        if not cause or not effect:
            violations.append("Empty node name")

        if len(cause) > 1000 or len(effect) > 1000:
            violations.append("Node name too long (max 1000 characters)")

        # Check for invalid characters
        if any(char in cause for char in ["\n", "\r", "\0"]):
            violations.append("Cause node contains invalid characters")
        if any(char in effect for char in ["\n", "\r", "\0"]):
            violations.append("Effect node contains invalid characters")

        # Check strength
        if not np.isfinite(strength):
            violations.append(f"Non-finite strength: {strength}")

        if strength < 0 or strength > 1:
            violations.append(f"Strength out of bounds [0, 1]: {strength}")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _validate_distribution_safety(
        self, distribution: ProbabilityDistribution
    ) -> Dict[str, Any]:
        """
        Validate probability distribution parameters for safety.

        Args:
            distribution: ProbabilityDistribution object

        Returns:
            Dictionary with 'safe' boolean and optional 'reason' string
        """

        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check distribution type
        valid_types = ["normal", "uniform", "beta"]
        if distribution.distribution_type not in valid_types:
            violations.append(
                f"Invalid distribution type: {distribution.distribution_type}"
            )

        # Check parameters exist and are valid
        for key, value in distribution.parameters.items():
            if not isinstance(value, (int, float)):
                violations.append(f"Invalid parameter type for {key}: {type(value)}")
            elif not np.isfinite(value):
                violations.append(f"Non-finite parameter {key}: {value}")

        # Type-specific validation
        if distribution.distribution_type == "normal":
            std = distribution.parameters.get("std", 1)
            if std <= 0:
                violations.append("Normal distribution std must be positive")
            if std > 1000:
                violations.append("Normal distribution std too large (max 1000)")

        elif distribution.distribution_type == "uniform":
            low = distribution.parameters.get("low", 0)
            high = distribution.parameters.get("high", 1)
            if low >= high:
                violations.append("Uniform distribution low must be less than high")

        elif distribution.distribution_type == "beta":
            alpha = distribution.parameters.get("alpha", 1)
            beta = distribution.parameters.get("beta", 1)
            if alpha <= 0 or beta <= 0:
                violations.append("Beta distribution parameters must be positive")
            if alpha > 1000 or beta > 1000:
                violations.append("Beta distribution parameters too large (max 1000)")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _invalidate_caches(self):
        """Atomically invalidate all caches after graph modifications"""

        self.path_finder.clear_cache()
        self.cycle_detector.cycle_cache.clear()

    def find_paths(
        self, source: str, targets: Union[str, List[str]], max_length: int = 5
    ) -> List[CausalPath]:
        """
        Find causal paths from source to targets with length limit.

        Args:
            source: Source node
            targets: Target node(s)
            max_length: Maximum path length

        Returns:
            List of CausalPath objects
        """

        self.path_queries += 1
        return self.path_finder.find_paths(source, targets, max_length)

    def find_all_paths(
        self, source: str, targets: Union[str, List[str]]
    ) -> List[CausalPath]:
        """
        Find all causal paths from source to targets (no length limit).

        Args:
            source: Source node
            targets: Target node(s)

        Returns:
            List of CausalPath objects
        """

        self.path_queries += 1
        return self.path_finder.find_all_paths(source, targets)

    def find_shortest_path(
        self, source: str, target: str, weighted: bool = True
    ) -> Optional[CausalPath]:
        """
        Find shortest path using Dijkstra's algorithm (weighted) or BFS (unweighted).

        Args:
            source: Source node
            target: Target node
            weighted: If True, use edge weights; if False, use hop count

        Returns:
            CausalPath object or None if no path exists
        """

        self.path_queries += 1
        return self.path_finder.find_shortest_path(source, target, weighted)

    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Get Markov blanket of a node (parents, children, and co-parents).

        Args:
            node: Node name

        Returns:
            Set of node names in Markov blanket
        """

        return self.d_separator.get_markov_blanket(node)

    def is_d_separated(self, x: str, y: str, conditioning_set: Set[str]) -> bool:
        """
        Check if x and y are d-separated given conditioning set.
        Uses Bayes-Ball algorithm for accurate d-separation testing.

        Args:
            x: First node
            y: Second node
            conditioning_set: Set of conditioning nodes

        Returns:
            True if x and y are d-separated, False otherwise
        """

        return self.d_separator.is_d_separated(x, y, conditioning_set)

    def has_edge(self, cause: str, effect: str) -> bool:
        """
        Check if edge exists in graph.

        Args:
            cause: Source node
            effect: Target node

        Returns:
            True if edge exists
        """

        return self.structure.has_edge(cause, effect)

    def get_edge(self, cause: str, effect: str) -> Optional[CausalEdge]:
        """
        Get edge object with all attributes.

        Args:
            cause: Source node
            effect: Target node

        Returns:
            CausalEdge object or None if edge doesn't exist
        """

        return self.structure.get_edge(cause, effect)

    def get_parents(self, node: str) -> Set[str]:
        """
        Get parent nodes (nodes with edges to this node).

        Args:
            node: Node name

        Returns:
            Set of parent node names
        """

        return self.structure.get_parents(node)

    def get_children(self, node: str) -> Set[str]:
        """
        Get child nodes (nodes with edges from this node).

        Args:
            node: Node name

        Returns:
            Set of child node names
        """

        return self.structure.get_children(node)

    def get_ancestors(self, node: str) -> Set[str]:
        """
        Get all ancestors of a node (all nodes with paths to this node).

        Args:
            node: Node name

        Returns:
            Set of ancestor node names
        """

        return self.path_finder.get_ancestors(node)

    def get_descendants(self, node: str) -> Set[str]:
        """
        Get all descendants of a node (all nodes reachable from this node).

        Args:
            node: Node name

        Returns:
            Set of descendant node names
        """

        return self.path_finder.get_descendants(node)

    def has_cycles(self) -> bool:
        """
        Check if graph contains cycles.

        Returns:
            True if graph has cycles
        """

        self.cycle_checks += 1
        return self.cycle_detector.has_cycles()

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.

        Returns:
            List of sets, each set contains nodes in one SCC
        """

        return self.cycle_detector.find_strongly_connected_components()

    def get_longest_path_length(self) -> int:
        """
        Get length of longest path in DAG (number of edges, not weighted sum).

        Returns:
            Length of longest path, or -1 if graph has cycles
        """

        if self.has_cycles():
            return -1

        return self.path_finder.get_longest_path_length()

    def break_cycles_minimum_feedback(self) -> List[Tuple[str, str]]:
        """
        Break all cycles by removing minimum feedback arc set.
        Uses greedy algorithm to minimize total weight of removed edges.

        Returns:
            List of removed edges (cause, effect) tuples
        """

        removed = self.cycle_detector.break_cycles_minimum_feedback()

        if removed:
            self.edge_count -= len(removed)
            self._invalidate_caches()

        return removed

    def topological_sort(self) -> List[str]:
        """
        Get topological ordering of nodes using Kahn's algorithm.

        Returns:
            List of nodes in topological order

        Raises:
            ValueError: If graph contains cycles
        """

        if self.has_cycles():
            raise ValueError("Cannot topologically sort graph with cycles")

        return self.topological_sorter.topological_sort()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the causal DAG.

        Returns:
            Dictionary with various statistics
        """

        stats = {
            "node_count": len(self.structure.nodes),
            "edge_count": self.edge_count,
            "cycle_checks": self.cycle_checks,
            "path_queries": self.path_queries,
            "has_cycles": self.has_cycles(),
            "longest_path": self.get_longest_path_length()
            if not self.has_cycles()
            else -1,
            "cache_size": len(self.path_finder.path_cache),
        }

        # Add degree statistics
        if self.structure.nodes:
            in_degrees = [
                len(self.structure.get_parents(n)) for n in self.structure.nodes
            ]
            out_degrees = [
                len(self.structure.get_children(n)) for n in self.structure.nodes
            ]

            stats["degree_stats"] = {
                "avg_in_degree": np.mean(in_degrees),
                "max_in_degree": max(in_degrees),
                "avg_out_degree": np.mean(out_degrees),
                "max_out_degree": max(out_degrees),
            }

        # Add safety statistics
        if self.safety_validator:
            stats["safety"] = {
                "enabled": True,
                "blocks": dict(self.safety_blocks),
                "corrections": dict(self.safety_corrections),
                "total_blocks": sum(self.safety_blocks.values()),
                "total_corrections": sum(self.safety_corrections.values()),
            }
        else:
            stats["safety"] = {"enabled": False}

        return stats

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export graph to dictionary format for serialization.

        Returns:
            Dictionary representation of the graph
        """

        with self.lock:
            nodes_list = list(self.structure.nodes)
            edges_list = []

            for (cause, effect), edge in self.structure.edges.items():
                edge_dict = {
                    "cause": cause,
                    "effect": effect,
                    "strength": edge.strength,
                    "evidence_type": edge.evidence_type.value,
                    "confidence_interval": edge.confidence_interval,
                    "metadata": edge.metadata,
                    "timestamp": edge.timestamp,
                    "is_stochastic": edge.is_stochastic,
                }

                if edge.probability_distribution:
                    edge_dict["probability_distribution"] = {
                        "type": edge.probability_distribution.distribution_type,
                        "parameters": edge.probability_distribution.parameters,
                    }

                edges_list.append(edge_dict)

            return {
                "nodes": nodes_list,
                "edges": edges_list,
                "statistics": self.get_statistics(),
            }

    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Import graph from dictionary format.

        Args:
            data: Dictionary with 'nodes' and 'edges' keys

        Returns:
            True if import successful
        """

        with self.lock:
            try:
                # Clear existing graph
                self.structure = GraphStructure()
                self.cycle_detector = CycleDetector(self.structure)
                self.path_finder = PathFinder(self.structure)
                self.d_separator = DSeparationChecker(self.structure, self.path_finder)
                self.topological_sorter = TopologicalSorter(self.structure)

                self.edge_count = 0

                # Import edges
                edges = data.get("edges", [])
                for edge_dict in edges:
                    if (
                        edge_dict.get("is_stochastic")
                        and "probability_distribution" in edge_dict
                    ):
                        # Import stochastic edge
                        self.add_stochastic_edge(
                            cause=edge_dict["cause"],
                            effect=edge_dict["effect"],
                            probability_distribution=edge_dict[
                                "probability_distribution"
                            ],
                        )
                    else:
                        # Import regular edge
                        self.add_edge(
                            cause=edge_dict["cause"],
                            effect=edge_dict["effect"],
                            strength=edge_dict["strength"],
                            evidence_type=edge_dict.get("evidence_type", "correlation"),
                            confidence_interval=edge_dict.get("confidence_interval"),
                            metadata=edge_dict.get("metadata"),
                        )

                logger.info(
                    f"Imported graph with {len(self.structure.nodes)} nodes and {self.edge_count} edges"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to import graph: {e}")
                return False

    def save_to_file(self, filepath: Union[str, FilePath]) -> bool:
        """
        Save graph to JSON file.

        Args:
            filepath: Path to output file

        Returns:
            True if save successful
        """

        try:
            data = self.export_to_dict()

            filepath = FilePath(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved graph to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save graph to {filepath}: {e}")
            return False

    def load_from_file(self, filepath: Union[str, FilePath]) -> bool:
        """
        Load graph from JSON file.

        Args:
            filepath: Path to input file

        Returns:
            True if load successful
        """

        try:
            filepath = FilePath(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = self.import_from_dict(data)

            if result:
                logger.info(f"Loaded graph from {filepath}")

            return result

        except Exception as e:
            logger.error(f"Failed to load graph from {filepath}: {e}")
            return False

    @property
    def nodes(self) -> Set[str]:
        """Get all nodes in graph"""
        return self.structure.get_all_nodes()

    @property
    def edges(self) -> Dict[Tuple[str, str], CausalEdge]:
        """Get all edges in graph"""
        return self.structure.get_all_edges()

    def __len__(self) -> int:
        """Get number of nodes in graph"""
        return len(self.structure.nodes)

    def __contains__(self, node: str) -> bool:
        """Check if node exists in graph"""
        return node in self.structure.nodes

    def __repr__(self) -> str:
        """String representation of graph"""
        return f"CausalDAG(nodes={len(self.structure.nodes)}, edges={self.edge_count}, has_cycles={self.has_cycles()})"
