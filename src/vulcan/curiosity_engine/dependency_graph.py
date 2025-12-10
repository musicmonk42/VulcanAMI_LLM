"""
dependency_graph.py - Dependency graph and analysis for Curiosity Engine
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies between gaps"""

    CAUSAL = "causal"  # One gap causes another
    PREREQUISITE = "prerequisite"  # Must solve one before another
    RELATED = "related"  # Related but not strictly dependent
    WEAK = "weak"  # Weak dependency (can be broken if needed)


@dataclass
class DependencyEdge:
    """Edge in dependency graph"""

    source: Any  # Source gap
    target: Any  # Target gap
    dependency_type: DependencyType
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def get_cost(self) -> float:
        """Get cost of breaking this dependency"""
        try:
            if self.dependency_type == DependencyType.WEAK:
                return 0.1 * self.strength
            elif self.dependency_type == DependencyType.RELATED:
                return 0.5 * self.strength
            elif self.dependency_type == DependencyType.PREREQUISITE:
                return 2.0 * self.strength
            else:  # CAUSAL
                return 5.0 * self.strength
        except Exception as e:
            logger.warning("Error calculating edge cost: %s", e)
            return 1.0

    def age(self) -> float:
        """Get age of edge in seconds"""
        try:
            return time.time() - self.created_at
        except Exception as e:
            return 0.0


class GraphStorage:
    """Manages graph storage and structure - SEPARATED CONCERN"""

    def __init__(self, max_nodes: int = 10000, max_edges: int = 50000):
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.nodes = {}  # gap_id -> gap object
        self.edges = {}  # (source_id, target_id) -> DependencyEdge
        self.adjacency = defaultdict(set)  # source_id -> set of target_ids
        self.reverse_adjacency = defaultdict(set)  # target_id -> set of source_ids
        self.weak_edges = set()  # set of (source_id, target_id) tuples

        # Track node creation times for proper eviction
        self.node_creation_times = {}  # node_id -> timestamp

        # NetworkX graph for algorithms
        self.nx_graph = nx.DiGraph()

        # Thread safety
        self.lock = threading.RLock()

    def add_node(self, node_id: str, node_data: Any) -> bool:
        """Add node to storage"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)

                if len(self.nodes) >= self.max_nodes:
                    return False

                self.nodes[node_id] = node_data
                self.node_creation_times[node_id] = time.time()
                self.nx_graph.add_node(node_id, data=node_data)
                return True
            except Exception as e:
                logger.error("Error adding node: %s", e)
                return False

    def add_edge(self, source_id: str, target_id: str, edge: DependencyEdge) -> bool:
        """Add edge to storage"""
        with self.lock:
            try:
                if not isinstance(source_id, str):
                    source_id = str(source_id)
                if not isinstance(target_id, str):
                    target_id = str(target_id)

                if len(self.edges) >= self.max_edges:
                    return False

                edge_key = (source_id, target_id)

                # Update or add edge
                if edge_key in self.edges:
                    self.edges[edge_key].strength = max(
                        self.edges[edge_key].strength, edge.strength
                    )
                else:
                    self.edges[edge_key] = edge
                    self.adjacency[source_id].add(target_id)
                    self.reverse_adjacency[target_id].add(source_id)

                    if edge.dependency_type == DependencyType.WEAK:
                        self.weak_edges.add(edge_key)

                    self.nx_graph.add_edge(source_id, target_id, weight=edge.strength)

                return True
            except Exception as e:
                logger.error("Error adding edge: %s", e)
                return False

    def remove_edge(self, source_id: str, target_id: str):
        """Remove edge from storage"""
        with self.lock:
            try:
                if not isinstance(source_id, str):
                    source_id = str(source_id)
                if not isinstance(target_id, str):
                    target_id = str(target_id)

                edge_key = (source_id, target_id)

                if edge_key in self.edges:
                    del self.edges[edge_key]

                self.adjacency[source_id].discard(target_id)
                self.reverse_adjacency[target_id].discard(source_id)
                self.weak_edges.discard(edge_key)

                if self.nx_graph.has_edge(source_id, target_id):
                    self.nx_graph.remove_edge(source_id, target_id)
            except Exception as e:
                logger.error("Error removing edge: %s", e)

    def remove_node(self, node_id: str):
        """Remove node and all its edges"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)

                if node_id not in self.nodes:
                    return

                # Remove all edges involving this node
                edges_to_remove = []
                for source, target in list(self.edges.keys()):
                    if source == node_id or target == node_id:
                        edges_to_remove.append((source, target))

                for source, target in edges_to_remove:
                    self.remove_edge(source, target)

                # Remove node
                del self.nodes[node_id]

                # Remove creation time
                if node_id in self.node_creation_times:
                    del self.node_creation_times[node_id]

                if self.nx_graph.has_node(node_id):
                    self.nx_graph.remove_node(node_id)
            except Exception as e:
                logger.error("Error removing node: %s", e)

    def get_node(self, node_id: str) -> Optional[Any]:
        """Get node data"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)
                return self.nodes.get(node_id)
            except Exception as e:
                return None

    def get_edge(self, source_id: str, target_id: str) -> Optional[DependencyEdge]:
        """Get edge data"""
        with self.lock:
            try:
                if not isinstance(source_id, str):
                    source_id = str(source_id)
                if not isinstance(target_id, str):
                    target_id = str(target_id)
                return self.edges.get((source_id, target_id))
            except Exception as e:
                return None

    def get_children(self, node_id: str) -> Set[str]:
        """Get immediate children of node"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)
                return self.adjacency.get(node_id, set()).copy()
            except Exception as e:
                return set()

    def get_parents(self, node_id: str) -> Set[str]:
        """Get immediate parents of node"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)
                return self.reverse_adjacency.get(node_id, set()).copy()
            except Exception as e:
                return set()

    def node_count(self) -> int:
        """Get number of nodes"""
        with self.lock:
            return len(self.nodes)

    def edge_count(self) -> int:
        """Get number of edges"""
        with self.lock:
            return len(self.edges)

    def get_all_nodes(self) -> Dict[str, Any]:
        """Get all nodes"""
        with self.lock:
            return self.nodes.copy()

    def get_all_edges(self) -> Dict[Tuple[str, str], DependencyEdge]:
        """Get all edges"""
        with self.lock:
            return self.edges.copy()

    def get_node_age(self, node_id: str) -> float:
        """Get age of node in seconds"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)
                created = self.node_creation_times.get(node_id, time.time())
                return time.time() - created
            except Exception as e:
                return 0.0


class PathFinder:
    """Finds paths and relationships in graph - SEPARATED CONCERN"""

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.lock = threading.RLock()

    def has_path(self, source_id: str, target_id: str) -> bool:
        """Check if path exists from source to target"""
        with self.lock:
            try:
                if not isinstance(source_id, str):
                    source_id = str(source_id)
                if not isinstance(target_id, str):
                    target_id = str(target_id)

                if source_id == target_id:
                    return True

                if not self.storage.get_node(source_id) or not self.storage.get_node(
                    target_id
                ):
                    return False

                # BFS with early termination
                visited = set()
                queue = deque([source_id])

                while queue:
                    current = queue.popleft()
                    if current == target_id:
                        return True

                    if current in visited:
                        continue
                    visited.add(current)

                    # Add unvisited neighbors
                    neighbors = self.storage.get_children(current)
                    queue.extend(neighbors - visited)

                return False
            except Exception as e:
                logger.error("Error checking path: %s", e)
                return False

    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendants of a node"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)

                descendants = set()
                visited = set()
                queue = deque([node_id])

                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)

                    # Add children
                    children = self.storage.get_children(current)
                    descendants.update(children)
                    queue.extend(children - visited)

                return descendants
            except Exception as e:
                logger.error("Error getting descendants: %s", e)
                return set()

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestors of a node"""
        with self.lock:
            try:
                if not isinstance(node_id, str):
                    node_id = str(node_id)

                ancestors = set()
                visited = set()
                queue = deque([node_id])

                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)

                    # Add parents
                    parents = self.storage.get_parents(current)
                    ancestors.update(parents)
                    queue.extend(parents - visited)

                return ancestors
            except Exception as e:
                logger.error("Error getting ancestors: %s", e)
                return set()

    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between nodes"""
        with self.lock:
            try:
                if not isinstance(source_id, str):
                    source_id = str(source_id)
                if not isinstance(target_id, str):
                    target_id = str(target_id)

                return nx.shortest_path(self.storage.nx_graph, source_id, target_id)
            except Exception as e:
                return None


class CycleDetector:
    """Detects and manages cycles in graph - SEPARATED CONCERN"""

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.cycle_count = 0
        self.lock = threading.RLock()

    def would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding edge would create cycle"""
        with self.lock:
            try:
                if not isinstance(source_id, str):
                    source_id = str(source_id)
                if not isinstance(target_id, str):
                    target_id = str(target_id)

                if source_id == target_id:
                    return True

                # Check if there's already a path from target to source
                path_finder = PathFinder(self.storage)
                return path_finder.has_path(target_id, source_id)
            except Exception as e:
                logger.error("Error checking cycle: %s", e)
                return True  # Conservative: assume cycle to be safe

    def has_cycles(self) -> bool:
        """Check if graph has cycles"""
        with self.lock:
            try:
                return not nx.is_directed_acyclic_graph(self.storage.nx_graph)
            except Exception as e:
                return self._has_cycles_dfs()

    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in graph"""
        with self.lock:
            try:
                cycles = list(nx.simple_cycles(self.storage.nx_graph))
                return cycles[:100]  # Limit to first 100
            except Exception as e:
                return []

    def break_cycles_minimum_cost(self) -> List[Tuple[str, str]]:
        """Break cycles by removing minimum cost edges"""
        with self.lock:
            removed = []
            max_iterations = 100
            iteration = 0

            while self.has_cycles() and iteration < max_iterations:
                iteration += 1

                # EXAMINE: Find all cycles
                cycles = self.find_cycles()
                if not cycles:
                    break

                # SELECT: Find minimum cost edge to remove
                min_edge = None
                min_cost = float("inf")

                for cycle in cycles:
                    for i in range(len(cycle)):
                        j = (i + 1) % len(cycle)
                        edge = self.storage.get_edge(cycle[i], cycle[j])

                        if edge:
                            cost = edge.get_cost()
                            if cost < min_cost:
                                min_cost = cost
                                min_edge = (cycle[i], cycle[j])

                # APPLY: Remove edge
                if min_edge:
                    self.storage.remove_edge(min_edge[0], min_edge[1])
                    removed.append(min_edge)
                    self.cycle_count += 1

                    # REMEMBER
                    logger.info(
                        "Removed edge %s -> %s to break cycle (cost=%.2f)",
                        min_edge[0],
                        min_edge[1],
                        min_cost,
                    )
                else:
                    # No edge found, break to avoid infinite loop
                    break

            if iteration >= max_iterations:
                logger.warning("Max iterations reached in cycle breaking")

            return removed

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components"""
        with self.lock:
            try:
                sccs = list(nx.strongly_connected_components(self.storage.nx_graph))
                return [item for item in sccs if len(scc] > 1)
            except Exception as e:
                return self._find_sccs_tarjan()

    def _has_cycles_dfs(self) -> bool:
        """Check for cycles using DFS"""
        visited = set()
        rec_stack = set()

        def visit(node):
            visited.add(node)
            rec_stack.add(node)

            try:
                for neighbor in self.storage.get_children(node):
                    if neighbor not in visited:
                        if visit(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        self.cycle_count += 1
                        return True
                return False
            finally:
                # FIX: Always cleanup recursion stack
                rec_stack.discard(node)

        for node in self.storage.get_all_nodes():
            if node not in visited:
                if visit(node):
                    return True

        return False

    def _find_sccs_tarjan(self) -> List[Set[str]]:
        """Find strongly connected components using Tarjan's algorithm"""
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()
        sccs = []

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for child in self.storage.get_children(node):
                if child not in indices:
                    strongconnect(child)
                    lowlinks[node] = min(lowlinks[node], lowlinks[child])
                elif child in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[child])

            if lowlinks[node] == indices[node]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.add(w)
                    if w == node:
                        break
                if len(scc) > 1:
                    sccs.append(scc)

        for node in self.storage.get_all_nodes():
            if node not in indices:
                strongconnect(node)

        return sccs


class TopologicalSorter:
    """Performs topological sorting - SEPARATED CONCERN"""

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.lock = threading.RLock()

    def topological_sort(self) -> List[Any]:
        """Get topological ordering of nodes"""
        with self.lock:
            try:
                # Use NetworkX topological sort
                sorted_ids = list(nx.topological_sort(self.storage.nx_graph))
                return [
                    self.storage.get_node(node_id)
                    for node_id in sorted_ids
                    if self.storage.get_node(node_id) is not None
                ]
            except Exception as e:
                # Fallback to Kahn's algorithm
                return self._topological_sort_kahn()

    def _topological_sort_kahn(self) -> List[Any]:
        """Topological sort using Kahn's algorithm"""
        in_degree = defaultdict(int)

        # FIX: Initialize all nodes first
        all_nodes = self.storage.get_all_nodes()
        for node_id in all_nodes:
            in_degree[node_id] = 0

        # Calculate in-degrees
        for node_id in all_nodes:
            for child in self.storage.get_children(node_id):
                in_degree[child] += 1

        # Find nodes with no incoming edges
        queue = deque([n for n in all_nodes if in_degree[n] == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            node = self.storage.get_node(node_id)
            if node:
                result.append(node)

            # Remove edges from this node
            for child in self.storage.get_children(node_id):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != self.storage.node_count():
            raise ValueError("Graph has cycles")

        return result


class CacheManager:
    """Manages caching for graph operations - SEPARATED CONCERN"""

    def __init__(self, cache_ttl: int = 60):
        self.cache_ttl = cache_ttl

        # Different cache types
        self._path_cache = {}  # (source, target) -> bool
        self._descendants_cache = {}  # node_id -> Set[str]
        self._ancestors_cache = {}  # node_id -> Set[str]
        self._topological_cache = None
        self._topological_cache_time = 0

        # Cache timestamps
        self._cache_times = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

        self.lock = threading.RLock()

    def get_path(self, key: Tuple[str, str]) -> Optional[bool]:
        """Get cached path result"""
        with self.lock:
            try:
                cache_key = f"path_{key[0]}_{key[1]}"
                if self._is_valid(cache_key) and key in self._path_cache:
                    self.cache_hits += 1
                    return self._path_cache[key]
                self.cache_misses += 1
                return None
            except Exception as e:
                self.cache_misses += 1
                return None

    def put_path(self, key: Tuple[str, str], value: bool):
        """Cache path result"""
        with self.lock:
            try:
                cache_key = f"path_{key[0]}_{key[1]}"
                self._path_cache[key] = value
                self._cache_times[cache_key] = time.time()
                self._limit_cache_size(self._path_cache, 10000)
            except Exception as e:
                logger.warning("Error caching path: %s", e)

    def get_descendants(self, node_id: str) -> Optional[Set[str]]:
        """Get cached descendants"""
        with self.lock:
            try:
                cache_key = f"desc_{node_id}"
                if self._is_valid(cache_key) and node_id in self._descendants_cache:
                    self.cache_hits += 1
                    return self._descendants_cache[node_id].copy()
                self.cache_misses += 1
                return None
            except Exception as e:
                self.cache_misses += 1
                return None

    def put_descendants(self, node_id: str, descendants: Set[str]):
        """Cache descendants"""
        with self.lock:
            try:
                cache_key = f"desc_{node_id}"
                self._descendants_cache[node_id] = descendants.copy()
                self._cache_times[cache_key] = time.time()
                self._limit_cache_size(self._descendants_cache, 1000)
            except Exception as e:
                logger.warning("Error caching descendants: %s", e)

    def get_ancestors(self, node_id: str) -> Optional[Set[str]]:
        """Get cached ancestors"""
        with self.lock:
            try:
                cache_key = f"anc_{node_id}"
                if self._is_valid(cache_key) and node_id in self._ancestors_cache:
                    self.cache_hits += 1
                    return self._ancestors_cache[node_id].copy()
                self.cache_misses += 1
                return None
            except Exception as e:
                self.cache_misses += 1
                return None

    def put_ancestors(self, node_id: str, ancestors: Set[str]):
        """Cache ancestors"""
        with self.lock:
            try:
                cache_key = f"anc_{node_id}"
                self._ancestors_cache[node_id] = ancestors.copy()
                self._cache_times[cache_key] = time.time()
                self._limit_cache_size(self._ancestors_cache, 1000)
            except Exception as e:
                logger.warning("Error caching ancestors: %s", e)

    def get_topological(self) -> Optional[List[Any]]:
        """Get cached topological sort"""
        with self.lock:
            try:
                if self._is_valid("topological", self._topological_cache_time):
                    self.cache_hits += 1
                    return self._topological_cache
                self.cache_misses += 1
                return None
            except Exception as e:
                self.cache_misses += 1
                return None

    def put_topological(self, result: List[Any]):
        """Cache topological sort"""
        with self.lock:
            try:
                self._topological_cache = result
                self._topological_cache_time = time.time()
            except Exception as e:
                logger.warning("Error caching topological sort: %s", e)

    def invalidate_all(self):
        """Invalidate all caches"""
        with self.lock:
            self._path_cache.clear()
            self._descendants_cache.clear()
            self._ancestors_cache.clear()
            self._topological_cache = None
            self._topological_cache_time = 0
            self._cache_times.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.cache_hits + self.cache_misses
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(1, total),
                "path_cache_size": len(self._path_cache),
                "descendants_cache_size": len(self._descendants_cache),
                "ancestors_cache_size": len(self._ancestors_cache),
            }

    def _is_valid(self, key: str, timestamp: Optional[float] = None) -> bool:
        """Check if cache entry is still valid"""
        try:
            if timestamp is not None:
                return time.time() - timestamp < self.cache_ttl

            cached_time = self._cache_times.get(key, 0)
            return time.time() - cached_time < self.cache_ttl
        except Exception as e:
            return False

    def _limit_cache_size(self, cache: Dict, max_size: int):
        """Limit cache size by removing oldest entries using LRU"""
        try:
            if len(cache) > max_size:
                # FIX: Use actual cache_times for LRU eviction
                num_to_remove = max_size // 10

                # Build list of (key, time) tuples for cache entries
                cache_entries = []
                for key in cache.keys():
                    # Generate cache key for timestamp lookup
                    if isinstance(key, tuple):
                        cache_key = f"path_{key[0]}_{key[1]}"
                    else:
                        # Check which cache this is
                        if key in self._descendants_cache:
                            cache_key = f"desc_{key}"
                        elif key in self._ancestors_cache:
                            cache_key = f"anc_{key}"
                        else:
                            cache_key = str(key)

                    timestamp = self._cache_times.get(cache_key, 0)
                    cache_entries.append((key, timestamp))

                # Sort by timestamp (oldest first)
                cache_entries.sort(key=lambda x: x[1])

                # Remove oldest entries
                for key, _ in cache_entries[:num_to_remove]:
                    cache.pop(key, None)

                    # Clean up cache_times
                    if isinstance(key, tuple):
                        cache_key = f"path_{key[0]}_{key[1]}"
                    else:
                        if key in self._descendants_cache:
                            cache_key = f"desc_{key}"
                        elif key in self._ancestors_cache:
                            cache_key = f"anc_{key}"
                        else:
                            cache_key = str(key)

                    self._cache_times.pop(cache_key, None)
        except Exception as e:
            logger.warning("Error limiting cache size: %s", e)


class EvictionManager:
    """Manages node and edge eviction - SEPARATED CONCERN"""

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.lock = threading.RLock()

    def evict_oldest_nodes(self, percentage: float = 0.1):
        """Evict oldest nodes to maintain size limit"""
        with self.lock:
            try:
                if self.storage.node_count() < self.storage.max_nodes:
                    return

                # Find nodes to evict
                num_to_evict = int(self.storage.max_nodes * percentage)

                # FIX: Sort by actual creation time
                nodes_with_age = [
                    (node_id, self.storage.node_creation_times.get(node_id, 0))
                    for node_id in self.storage.get_all_nodes()
                ]
                nodes_with_age.sort(key=lambda x: x[1])  # Oldest first

                # Remove oldest nodes
                for node_id, _ in nodes_with_age[:num_to_evict]:
                    self.storage.remove_node(node_id)

                logger.info("Evicted %d oldest nodes", num_to_evict)
            except Exception as e:
                logger.error("Error evicting oldest nodes: %s", e)

    def evict_weakest_edges(self, percentage: float = 0.1):
        """Evict weakest edges to maintain size limit"""
        with self.lock:
            try:
                if self.storage.edge_count() < self.storage.max_edges:
                    return

                # Find edges to evict
                num_to_evict = int(self.storage.max_edges * percentage)

                # Sort edges by cost (weakest first)
                edge_costs = []
                for edge_key, edge in self.storage.get_all_edges().items():
                    edge_costs.append((edge_key, edge.get_cost()))

                edge_costs.sort(key=lambda x: x[1])

                # Remove weakest edges
                for (source_id, target_id), _ in edge_costs[:num_to_evict]:
                    self.storage.remove_edge(source_id, target_id)

                logger.info("Evicted %d weakest edges", num_to_evict)
            except Exception as e:
                logger.error("Error evicting weakest edges: %s", e)


class CycleAwareDependencyGraph:
    """Dependency graph with cycle detection and caching - REFACTORED"""

    def __init__(self, max_nodes: int = 10000, max_edges: int = 50000):
        """
        Initialize dependency graph

        Args:
            max_nodes: Maximum number of nodes
            max_edges: Maximum number of edges
        """
        # Components
        self.storage = GraphStorage(max_nodes, max_edges)
        self.path_finder = PathFinder(self.storage)
        self.cycle_detector = CycleDetector(self.storage)
        self.sorter = TopologicalSorter(self.storage)
        self.cache = CacheManager()
        self.eviction_manager = EvictionManager(self.storage)

        # Statistics
        self.edges_broken = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info("CycleAwareDependencyGraph initialized (refactored)")

    def add_node(self, gap) -> str:
        """Add node to graph - REFACTORED"""
        with self.lock:
            try:
                # EXAMINE: Check size limit
                if self.storage.node_count() >= self.storage.max_nodes:
                    self.eviction_manager.evict_oldest_nodes()

                # Get node ID (immutable)
                node_id = self._get_node_id(gap)

                # APPLY: Add node
                if not self.storage.add_node(node_id, gap):
                    # Try again after eviction
                    self.eviction_manager.evict_oldest_nodes()
                    self.storage.add_node(node_id, gap)

                # REMEMBER: Invalidate caches
                self.cache.invalidate_all()

                logger.debug("Added node %s to dependency graph", node_id)
                return node_id
            except Exception as e:
                logger.error("Error adding node: %s", e)
                return ""

    def add_edge(
        self,
        source,
        target,
        dependency_type: DependencyType = DependencyType.PREREQUISITE,
        strength: float = 1.0,
    ) -> bool:
        """Add edge to graph - REFACTORED"""
        with self.lock:
            try:
                # EXAMINE: Check size limit
                if self.storage.edge_count() >= self.storage.max_edges:
                    self.eviction_manager.evict_weakest_edges()

                source_id = self._get_node_id(source)
                target_id = self._get_node_id(target)

                # Check for existing edge
                existing = self.storage.get_edge(source_id, target_id)
                if existing:
                    existing.strength = max(existing.strength, strength)
                    return True

                # SELECT: Check for cycle
                if self.would_create_cycle(source_id, target_id):
                    logger.warning(
                        "Edge %s -> %s would create cycle", source_id, target_id
                    )
                    return False

                # APPLY: Create and add edge
                edge = DependencyEdge(
                    source=source,
                    target=target,
                    dependency_type=dependency_type,
                    strength=strength,
                )

                if not self.storage.add_edge(source_id, target_id, edge):
                    # Try again after eviction
                    self.eviction_manager.evict_weakest_edges()
                    self.storage.add_edge(source_id, target_id, edge)

                # REMEMBER: Invalidate caches
                self.cache.invalidate_all()

                logger.debug(
                    "Added edge %s -> %s (type=%s, strength=%.2f)",
                    source_id,
                    target_id,
                    dependency_type.value,
                    strength,
                )
                return True
            except Exception as e:
                logger.error("Error adding edge: %s", e)
                return False

    def add_weak_edge(self, source, target, strength: float = 0.5) -> bool:
        """Add weak edge (can be broken if creates cycle) - REFACTORED"""
        with self.lock:
            try:
                source_id = self._get_node_id(source)
                target_id = self._get_node_id(target)

                # Create weak edge (don't check for cycles)
                edge = DependencyEdge(
                    source=source,
                    target=target,
                    dependency_type=DependencyType.WEAK,
                    strength=strength,
                )

                self.storage.add_edge(source_id, target_id, edge)
                self.cache.invalidate_all()

                logger.debug("Added weak edge %s -> %s", source_id, target_id)
                return True
            except Exception as e:
                logger.error("Error adding weak edge: %s", e)
                return False

    def would_create_cycle(self, source, target) -> bool:
        """Check if adding edge would create cycle - DELEGATED"""
        try:
            source_id = self._get_node_id(source)
            target_id = self._get_node_id(target)

            # Check cache first
            cache_key = (target_id, source_id)
            cached = self.cache.get_path(cache_key)

            if cached is not None:
                return cached

            # Check for cycle
            result = self.cycle_detector.would_create_cycle(source_id, target_id)

            # Cache result
            self.cache.put_path(cache_key, result)

            return result
        except Exception as e:
            logger.error("Error checking cycle: %s", e)
            return True  # Conservative: assume cycle

    def topological_sort(self) -> List[Any]:
        """Get topological ordering of nodes - DELEGATED"""
        try:
            # Check cache
            cached = self.cache.get_topological()
            if cached is not None:
                return cached

            if self.has_cycles():
                raise ValueError("Cannot topologically sort graph with cycles")

            result = self.sorter.topological_sort()

            # Cache result
            self.cache.put_topological(result)

            return result
        except Exception as e:
            logger.error("Error in topological sort: %s", e)
            return []

    def break_cycles_minimum_cost(self) -> List[Tuple[str, str]]:
        """Break cycles by removing minimum cost edges - DELEGATED"""
        try:
            removed = self.cycle_detector.break_cycles_minimum_cost()
            self.edges_broken += len(removed)
            self.cache.invalidate_all()
            return removed
        except Exception as e:
            logger.error("Error breaking cycles: %s", e)
            return []

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components - DELEGATED"""
        try:
            return self.cycle_detector.find_strongly_connected_components()
        except Exception as e:
            logger.error("Error finding SCCs: %s", e)
            return []

    def descendants(self, gap) -> Set[str]:
        """Get all descendants of a gap - DELEGATED"""
        try:
            gap_id = self._get_node_id(gap)

            # Check cache
            cached = self.cache.get_descendants(gap_id)
            if cached is not None:
                return cached

            # Get descendants
            descendants = self.path_finder.get_descendants(gap_id)

            # Cache result
            self.cache.put_descendants(gap_id, descendants)

            return descendants
        except Exception as e:
            logger.error("Error getting descendants: %s", e)
            return set()

    def ancestors(self, gap) -> Set[str]:
        """Get all ancestors of a gap - DELEGATED"""
        try:
            gap_id = self._get_node_id(gap)

            # Check cache
            cached = self.cache.get_ancestors(gap_id)
            if cached is not None:
                return cached

            # Get ancestors
            ancestors = self.path_finder.get_ancestors(gap_id)

            # Cache result
            self.cache.put_ancestors(gap_id, ancestors)

            return ancestors
        except Exception as e:
            logger.error("Error getting ancestors: %s", e)
            return set()

    def has_cycles(self) -> bool:
        """Check if graph has cycles - DELEGATED"""
        try:
            return self.cycle_detector.has_cycles()
        except Exception as e:
            logger.error("Error checking cycles: %s", e)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        try:
            cache_stats = self.cache.get_statistics()

            return {
                "nodes": self.storage.node_count(),
                "edges": self.storage.edge_count(),
                "weak_edges": len(self.storage.weak_edges),
                "cycles_detected": self.cycle_detector.cycle_count,
                "edges_broken": self.edges_broken,
                **cache_stats,
            }
        except Exception as e:
            logger.error("Error getting statistics: %s", e)
            return {}

    def _get_node_id(self, node) -> str:
        """Get stable, immutable node ID from node object or ID string"""
        try:
            if isinstance(node, str):
                return node
            elif hasattr(node, "id") and node.id:
                return str(node.id)
            elif hasattr(node, "gap_id") and node.gap_id:
                # Ensure id attribute exists for consistency
                if not hasattr(node, "id") or not node.id:
                    node.id = node.gap_id
                return str(node.gap_id)
            else:
                # Generate stable ID based on object hash
                node_id = f"gap_{hash(node) % 1000000}"
                if hasattr(node, "__dict__"):
                    node.id = node_id
                return node_id
        except Exception as e:
            logger.warning("Error getting node ID: %s", e)
            return f"gap_{id(node)}"

    # Compatibility properties for backward compatibility
    @property
    def nodes(self) -> Dict[str, Any]:
        """Get all nodes for backward compatibility"""
        return self.storage.get_all_nodes()

    @property
    def edges(self) -> Dict[Tuple[str, str], DependencyEdge]:
        """Get all edges for backward compatibility"""
        return self.storage.get_all_edges()

    @property
    def weak_edges(self) -> Set[Tuple[str, str]]:
        """Get weak edges for backward compatibility"""
        return self.storage.weak_edges

    @property
    def adjacency(self) -> Dict[str, Set[str]]:
        """Get adjacency for backward compatibility"""
        return self.storage.adjacency

    @property
    def reverse_adjacency(self) -> Dict[str, Set[str]]:
        """Get reverse adjacency for backward compatibility"""
        return self.storage.reverse_adjacency

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Get NetworkX graph for backward compatibility"""
        return self.storage.nx_graph

    @property
    def cycle_count(self) -> int:
        """Get cycle count for backward compatibility"""
        return self.cycle_detector.cycle_count

    @property
    def cache_hits(self) -> int:
        """Get cache hits for backward compatibility"""
        return self.cache.cache_hits

    @property
    def cache_misses(self) -> int:
        """Get cache misses for backward compatibility"""
        return self.cache.cache_misses


class DependencyAnalyzer:
    """Analyzes relationships between gaps - UNCHANGED (already focused)"""

    def __init__(self):
        """Initialize dependency analyzer"""
        self.dependency_patterns = {}
        self.root_causes = {}
        self.pattern_cache = {}
        self.cache_ttl = 300  # 5 minutes

        logger.info("DependencyAnalyzer initialized")

    def find_dependencies(self, gap) -> List[Any]:
        """
        Find dependencies for a gap

        Args:
            gap: Gap to analyze

        Returns:
            List of gaps that this gap depends on
        """
        try:
            # Check cache
            gap_id = self._get_gap_id(gap)
            cache_key = f"deps_{gap_id}"

            if cache_key in self.pattern_cache:
                cache_time, deps = self.pattern_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return deps

            dependencies = []

            # Analyze based on gap type
            if hasattr(gap, "type"):
                if gap.type == "decomposition":
                    # Decomposition gaps depend on semantic understanding
                    dependencies.extend(self._find_semantic_dependencies(gap))
                elif gap.type == "causal":
                    # Causal gaps depend on correlation data
                    dependencies.extend(self._find_correlation_dependencies(gap))
                elif gap.type == "transfer":
                    # Transfer gaps depend on domain knowledge
                    dependencies.extend(self._find_domain_dependencies(gap))
                elif gap.type == "latent":
                    # Latent gaps might depend on multiple types
                    dependencies.extend(self._find_latent_dependencies(gap))

            # Check for explicit dependencies
            if hasattr(gap, "dependencies"):
                dependencies.extend(gap.dependencies)

            # Remove duplicates while preserving order
            unique_deps = []
            seen = set()
            for dep in dependencies:
                dep_id = self._get_gap_id(dep)
                if dep_id not in seen:
                    unique_deps.append(dep)
                    seen.add(dep_id)

            # Cache result
            self.pattern_cache[cache_key] = (time.time(), unique_deps)

            return unique_deps
        except Exception as e:
            logger.error("Error finding dependencies: %s", e)
            return []

    def identify_root_causes(self, gap_graph: CycleAwareDependencyGraph) -> List[Any]:
        """
        Identify root cause gaps

        Args:
            gap_graph: Dependency graph

        Returns:
            List of root cause gaps
        """
        try:
            root_causes = []

            for node_id, gap in gap_graph.nodes.items():
                # Root cause has no ancestors or only weak ancestors
                ancestors = gap_graph.ancestors(node_id)

                if not ancestors:
                    root_causes.append(gap)
                else:
                    # Check if all ancestors are weak
                    all_weak = True
                    for ancestor_id in ancestors:
                        edge_key = (ancestor_id, node_id)
                        if edge_key not in gap_graph.weak_edges:
                            all_weak = False
                            break

                    if all_weak:
                        root_causes.append(gap)

            # Sort by priority if available
            if root_causes and hasattr(root_causes[0], "priority"):
                root_causes.sort(key=lambda g: g.priority, reverse=True)

            return root_causes
        except Exception as e:
            logger.error("Error identifying root causes: %s", e)
            return []

    def calculate_unlock_potential(
        self, gap, graph: CycleAwareDependencyGraph
    ) -> float:
        """
        Calculate how many gaps would be unlocked by solving this gap

        Args:
            gap: Gap to evaluate
            graph: Dependency graph

        Returns:
            Unlock potential score
        """
        try:
            gap_id = self._get_gap_id(gap)

            # Get descendants
            descendants = graph.descendants(gap_id)

            # Calculate potential
            potential = 0.0

            for desc_id in descendants:
                if desc_id in graph.nodes:
                    desc_gap = graph.nodes[desc_id]

                    # Weight by descendant priority
                    if hasattr(desc_gap, "priority"):
                        potential += desc_gap.priority
                    else:
                        potential += 1.0

                    # Bonus for gaps with many descendants (with limit)
                    desc_descendants = graph.descendants(desc_id)
                    potential += min(5, len(desc_descendants)) * 0.1

            return potential
        except Exception as e:
            logger.error("Error calculating unlock potential: %s", e)
            return 0.0

    def _get_gap_id(self, gap) -> str:
        """Get ID from gap object"""
        try:
            if isinstance(gap, str):
                return gap
            elif hasattr(gap, "id") and gap.id:
                return str(gap.id)
            elif hasattr(gap, "gap_id") and gap.gap_id:
                return str(gap.gap_id)
            else:
                return str(id(gap))
        except Exception as e:
            return str(id(gap))

    def _find_semantic_dependencies(self, gap) -> List[Any]:
        """Find semantic dependencies for decomposition gap"""
        try:
            dependencies = []

            # Check for missing concepts
            if hasattr(gap, "missing_concepts"):
                for concept in gap.missing_concepts:
                    # Create semantic gap
                    semantic_gap = self._create_gap(
                        gap_type="semantic",
                        concept=concept,
                        priority=0.8,
                        gap_id=f"semantic_{concept}",
                    )
                    dependencies.append(semantic_gap)

            # Check metadata for semantic requirements
            if hasattr(gap, "metadata") and "requires_concepts" in gap.metadata:
                for concept in gap.metadata["requires_concepts"]:
                    semantic_gap = self._create_gap(
                        gap_type="semantic",
                        concept=concept,
                        priority=0.7,
                        gap_id=f"semantic_req_{concept}",
                    )
                    dependencies.append(semantic_gap)

            return dependencies
        except Exception as e:
            logger.error("Error finding semantic dependencies: %s", e)
            return []

    def _find_correlation_dependencies(self, gap) -> List[Any]:
        """Find correlation dependencies for causal gap"""
        try:
            dependencies = []

            # Check for required variables
            if hasattr(gap, "variables"):
                for var in gap.variables[:2]:  # Limit to avoid too many deps
                    corr_gap = self._create_gap(
                        gap_type="correlation",
                        variable=var,
                        priority=0.6,
                        gap_id=f"corr_{var}",
                    )
                    dependencies.append(corr_gap)

            # Check metadata for correlation requirements
            if hasattr(gap, "metadata"):
                if "requires_correlation" in gap.metadata:
                    for var_pair in gap.metadata["requires_correlation"]:
                        corr_gap = self._create_gap(
                            gap_type="correlation",
                            variables=var_pair,
                            priority=0.5,
                            gap_id=f"corr_{'_'.join(var_pair)}",
                        )
                        dependencies.append(corr_gap)

            return dependencies
        except Exception as e:
            logger.error("Error finding correlation dependencies: %s", e)
            return []

    def _find_domain_dependencies(self, gap) -> List[Any]:
        """Find domain dependencies for transfer gap"""
        try:
            dependencies = []

            # Check for source and target domains
            source_domain = None
            target_domain = None

            if hasattr(gap, "source_domain"):
                source_domain = gap.source_domain
            elif hasattr(gap, "metadata") and "source_domain" in gap.metadata:
                source_domain = gap.metadata["source_domain"]

            if hasattr(gap, "target_domain"):
                target_domain = gap.target_domain
            elif hasattr(gap, "metadata") and "target_domain" in gap.metadata:
                target_domain = gap.metadata["target_domain"]

            if source_domain and target_domain:
                bridge_gap = self._create_gap(
                    gap_type="domain_bridge",
                    source=source_domain,
                    target=target_domain,
                    priority=0.7,
                    gap_id=f"bridge_{source_domain}_{target_domain}",
                )
                dependencies.append(bridge_gap)

            return dependencies
        except Exception as e:
            logger.error("Error finding domain dependencies: %s", e)
            return []

    def _find_latent_dependencies(self, gap) -> List[Any]:
        """Find dependencies for latent gap"""
        try:
            dependencies = []

            if hasattr(gap, "pattern"):
                # Need to understand the pattern first
                pattern_gap = self._create_gap(
                    gap_type="pattern",
                    pattern=gap.pattern,
                    priority=0.5,
                    gap_id=f"pattern_{id(gap.pattern)}",
                )
                dependencies.append(pattern_gap)

            # Check for exploratory dependencies
            if hasattr(gap, "metadata") and "requires_exploration" in gap.metadata:
                for area in gap.metadata["requires_exploration"]:
                    explore_gap = self._create_gap(
                        gap_type="exploration",
                        area=area,
                        priority=0.4,
                        gap_id=f"explore_{area}",
                    )
                    dependencies.append(explore_gap)

            return dependencies
        except Exception as e:
            logger.error("Error finding latent dependencies: %s", e)
            return []

    def _create_gap(self, gap_type: str, priority: float, gap_id: str, **kwargs):
        """Create a gap object"""
        try:
            gap = type(
                "Gap",
                (),
                {
                    "type": gap_type,
                    "priority": priority,
                    "id": gap_id,
                    "gap_id": gap_id,
                    "estimated_cost": 10.0,
                    "metadata": kwargs,
                },
            )()

            # Add any additional kwargs as attributes
            for key, value in kwargs.items():
                setattr(gap, key, value)

            return gap
        except Exception as e:
            logger.error("Error creating gap: %s", e)
            return None


class ROICalculator:
    """Calculates adjusted ROI with dependencies - UNCHANGED (already focused)"""

    def __init__(self, unlock_weight: float = 0.3, dependency_weight: float = 0.2):
        """
        Initialize ROI calculator

        Args:
            unlock_weight: Weight for unlock bonus
            dependency_weight: Weight for dependency penalty
        """
        self.unlock_weight = unlock_weight
        self.dependency_weight = dependency_weight
        self.roi_cache = {}
        self.cache_ttl = 60

        logger.info("ROICalculator initialized")

    def calculate_base_roi(self, gap) -> float:
        """
        Calculate base ROI for a gap

        Args:
            gap: Gap to evaluate

        Returns:
            Base ROI score
        """
        try:
            # Get priority and cost
            priority = gap.priority if hasattr(gap, "priority") else 1.0
            cost = gap.estimated_cost if hasattr(gap, "estimated_cost") else 1.0

            # Prevent division by zero
            cost = max(cost, 0.01)

            # Base ROI is priority / cost
            base_roi = priority / cost

            # Apply type-specific modifiers
            if hasattr(gap, "type"):
                if gap.type == "causal":
                    base_roi *= 1.3  # Causal gaps are valuable
                elif gap.type == "latent":
                    base_roi *= 1.2  # Latent gaps reveal hidden knowledge
                elif gap.type == "decomposition":
                    base_roi *= 1.1  # Decomposition enables problem solving

            return base_roi
        except Exception as e:
            logger.error("Error calculating base ROI: %s", e)
            return 1.0

    def apply_unlock_bonus(self, gap, descendants_count: int) -> float:
        """
        Apply bonus for gaps that unlock others

        Args:
            gap: Gap to evaluate
            descendants_count: Number of descendant gaps

        Returns:
            ROI with unlock bonus
        """
        try:
            base_roi = self.calculate_base_roi(gap)

            # Calculate unlock bonus
            unlock_bonus = descendants_count * self.unlock_weight

            # Apply with diminishing returns
            unlock_multiplier = 1.0 + np.log1p(unlock_bonus)

            return base_roi * unlock_multiplier
        except Exception as e:
            logger.error("Error applying unlock bonus: %s", e)
            return self.calculate_base_roi(gap)

    def apply_dependency_penalty(self, gap, ancestors_count: int) -> float:
        """
        Apply penalty for gaps with many dependencies

        Args:
            gap: Gap to evaluate
            ancestors_count: Number of ancestor gaps

        Returns:
            ROI with dependency penalty
        """
        try:
            base_roi = self.calculate_base_roi(gap)

            # Calculate dependency penalty
            dependency_penalty = ancestors_count * self.dependency_weight

            # Apply with diminishing effect
            penalty_multiplier = 1.0 / (1.0 + np.log1p(dependency_penalty))

            return base_roi * penalty_multiplier
        except Exception as e:
            logger.error("Error applying dependency penalty: %s", e)
            return self.calculate_base_roi(gap)

    def get_adjusted_roi(
        self, gap, dependency_graph: CycleAwareDependencyGraph
    ) -> float:
        """
        Get fully adjusted ROI considering dependencies

        Args:
            gap: Gap to evaluate
            dependency_graph: Dependency graph

        Returns:
            Adjusted ROI score
        """
        try:
            # Check cache
            gap_id = gap.id if hasattr(gap, "id") else str(gap)
            cache_key = f"roi_{gap_id}_{dependency_graph.edges_broken}"

            if cache_key in self.roi_cache:
                cache_time, roi = self.roi_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return roi

            # Get dependency counts
            descendants = dependency_graph.descendants(gap_id)
            ancestors = dependency_graph.ancestors(gap_id)

            descendants_count = len(descendants)
            ancestors_count = len(ancestors)

            # Calculate base ROI
            base_roi = self.calculate_base_roi(gap)

            # Apply unlock bonus
            unlock_multiplier = 1.0 + (descendants_count * self.unlock_weight)

            # Apply dependency penalty
            penalty_multiplier = 1.0 - min(
                0.5, ancestors_count * self.dependency_weight
            )

            # Calculate final adjusted ROI
            adjusted_roi = base_roi * unlock_multiplier * penalty_multiplier

            # Add small random noise to break ties
            adjusted_roi += np.random.uniform(-0.001, 0.001)

            roi = max(0.01, adjusted_roi)

            # Cache result
            self.roi_cache[cache_key] = (time.time(), roi)

            return roi
        except Exception as e:
            logger.error("Error calculating adjusted ROI: %s", e)
            return 1.0
