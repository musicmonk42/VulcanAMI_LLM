"""
decomposition_strategies.py - Decomposition strategies for problem decomposer
Part of the VULCAN-AGI system
"""

import numpy as np
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import json
from abc import ABC, abstractmethod
import copy
import hashlib
from enum import Enum

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, graph features will be limited")
    
    # Mock NetworkX for basic functionality
    class MockGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = []
            self._adjacency = defaultdict(list)
            
        def nodes(self):
            return self._nodes.keys()
            
        def edges(self):
            return self._edges
            
        def add_node(self, node_id, **attrs):
            self._nodes[node_id] = attrs
            
        def add_edge(self, source, target, **attrs):
            self._edges.append((source, target))
            self._adjacency[source].append(target)
            
        def degree(self, node=None):
            if node is not None:
                return len(self._adjacency.get(node, []))
            return {n: len(self._adjacency[n]) for n in self._nodes}
            
        def in_degree(self, node):
            count = 0
            for edges in self._adjacency.values():
                if node in edges:
                    count += 1
            return count
            
        def predecessors(self, node):
            preds = []
            for n, edges in self._adjacency.items():
                if node in edges:
                    preds.append(n)
            return preds
            
        def successors(self, node):
            return self._adjacency.get(node, [])
            
        def subgraph(self, nodes):
            sub = MockGraph()
            for node in nodes:
                if node in self._nodes:
                    sub.add_node(node, **self._nodes[node])
            for source, target in self._edges:
                if source in nodes and target in nodes:
                    sub.add_edge(source, target)
            return sub
    
    class MockNX:
        Graph = MockGraph
        DiGraph = MockGraph
        
        @staticmethod
        def is_directed_acyclic_graph(graph):
            # Actual cycle detection using DFS
            try:
                visited = set()
                rec_stack = set()
                
                def has_cycle(node):
                    visited.add(node)
                    rec_stack.add(node)
                    
                    for neighbor in graph._adjacency.get(node, []):
                        if neighbor not in visited:
                            if has_cycle(neighbor):
                                return True
                        elif neighbor in rec_stack:
                            return True
                    
                    rec_stack.remove(node)
                    return False
                
                for node in graph.nodes():
                    if node not in visited:
                        if has_cycle(node):
                            return False
                return True
            except:
                return True  # Fallback
            
        @staticmethod
        def density(graph):
            n = len(list(graph.nodes()))
            if n <= 1:
                return 0
            e = len(list(graph.edges()))
            return e / (n * (n - 1))
            
        @staticmethod
        def single_source_shortest_path_length(graph, source):
            # Simplified - just return depths
            return {node: 1 for node in graph.nodes()}
            
        @staticmethod
        def strongly_connected_components(graph):
            # Simplified - treat each node as its own component
            return [{node} for node in graph.nodes()]
            
        @staticmethod
        def weakly_connected_components(graph):
            # Simplified - treat all as one component
            if hasattr(graph, 'nodes'):
                return [set(graph.nodes())]
            return []
            
        @staticmethod
        def simple_cycles(graph):
            # Simplified - no cycles
            return []
            
        class algorithms:
            class isomorphism:
                @staticmethod
                def DiGraphMatcher(g1, g2):
                    class Matcher:
                        def subgraph_is_isomorphic(self):
                            return False
                    return Matcher()
    
    nx = MockNX()

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, using fallback cosine similarity")
    
    # Fallback cosine similarity
    def cosine_similarity(X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        
        X_norm[X_norm == 0] = 1
        Y_norm[Y_norm == 0] = 1
        
        X_normalized = X / X_norm
        Y_normalized = Y / Y_norm
        
        return np.dot(X_normalized, Y_normalized.T)

try:
    from scipy.spatial.distance import euclidean, hamming
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, using fallback distance functions")
    
    # Fallback distance functions
    def euclidean(u, v):
        u = np.asarray(u)
        v = np.asarray(v)
        return np.sqrt(np.sum((u - v) ** 2))
    
    def hamming(u, v):
        u = np.asarray(u)
        v = np.asarray(v)
        return np.mean(u != v)

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of decomposition strategies"""
    EXACT = "exact"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    SYNTHETIC = "synthetic"
    ANALOGICAL = "analogical"
    BRUTE_FORCE = "brute_force"


@dataclass
class DecompositionResult:
    """Result from decomposition strategy"""
    components: List[Any] = field(default_factory=list)
    confidence: float = 0.0
    strategy_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def is_complete(self) -> bool:
        """Check if decomposition is complete"""
        return len(self.components) > 0 and self.confidence > 0.5


@dataclass
class PatternMatch:
    """Pattern matching result"""
    pattern_id: str
    similarity_score: float
    matched_nodes: List[str] = field(default_factory=list)
    transformation: Dict[str, Any] = field(default_factory=dict)


class DecompositionStrategy(ABC):
    """Base class for all strategies"""
    
    def __init__(self, name: str = None, strategy_type: StrategyType = None):
        """
        Initialize decomposition strategy
        
        Args:
            name: Strategy name
            strategy_type: Type of strategy
        """
        self.name = name or self.__class__.__name__
        self.strategy_type = strategy_type or StrategyType.EXACT
        self.pattern_library = {}
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        
        logger.info("Initialized strategy: %s", self.name)
    
    @abstractmethod
    def apply(self, problem_graph) -> DecompositionResult:
        """
        Apply decomposition strategy to problem
        
        Args:
            problem_graph: Problem to decompose
            
        Returns:
            Decomposition result
        """
        pass
    
    def decompose(self, problem_graph) -> List[Dict[str, Any]]:
        """
        Decompose problem (alias for apply)
        
        Args:
            problem_graph: Problem to decompose
            
        Returns:
            List of decomposition steps
        """
        result = self.apply(problem_graph)
        
        # Convert result to list of steps
        steps = []
        
        if result.components:
            for i, component in enumerate(result.components):
                step = {
                    'step_id': f"{self.name}_{i}",
                    'type': self.strategy_type.value,
                    'component': component,
                    'confidence': result.confidence,
                    'action_type': component.get('type', 'process') if isinstance(component, dict) else 'process',
                    'description': self._generate_step_description(component, i),
                    'dependencies': [],
                    'estimated_complexity': 1.0,
                    'required_resources': {}
                }
                steps.append(step)
        else:
            # FIX: If no components found, create fallback steps
            # This ensures we always return at least one step
            logger.warning("%s produced no components, generating fallback step", self.name)
            
            # Try to extract nodes from problem graph
            nodes = []
            if hasattr(problem_graph, 'nodes') and problem_graph.nodes:
                nodes = list(problem_graph.nodes.keys())[:10]  # Limit to 10 nodes
            
            if nodes:
                # Create step for each node or group of nodes
                for i, node in enumerate(nodes):
                    step = {
                        'step_id': f"{self.name}_fallback_{i}",
                        'type': self.strategy_type.value,
                        'component': {
                            'type': 'node_processing',
                            'node': node,
                            'fallback': True
                        },
                        'confidence': 0.3,  # Low confidence for fallback
                        'action_type': 'process',
                        'description': f"Process node {node}",
                        'dependencies': [],
                        'estimated_complexity': 1.0,
                        'required_resources': {}
                    }
                    steps.append(step)
            else:
                # Create generic fallback step
                step = {
                    'step_id': f"{self.name}_fallback_generic",
                    'type': self.strategy_type.value,
                    'component': {
                        'type': 'generic_processing',
                        'problem': 'unknown',
                        'fallback': True
                    },
                    'confidence': 0.2,  # Very low confidence
                    'action_type': 'process',
                    'description': 'Generic problem processing',
                    'dependencies': [],
                    'estimated_complexity': 1.0,
                    'required_resources': {}
                }
                steps.append(step)
        
        return steps
    
    def _generate_step_description(self, component: Any, index: int) -> str:
        """Generate description for a decomposition step"""
        if isinstance(component, dict):
            comp_type = component.get('type', 'unknown')
            return f"Step {index}: {comp_type} operation"
        else:
            return f"Step {index}: Process component"
    
    def calculate_confidence(self, result: Any) -> float:
        """
        Calculate confidence score for result
        
        Args:
            result: Decomposition result
            
        Returns:
            Confidence score [0, 1]
        """
        if isinstance(result, DecompositionResult):
            # Base confidence on completeness and component count
            base_confidence = 0.5
            
            if result.is_complete():
                base_confidence += 0.2
            
            # More components = higher confidence (up to a point)
            component_factor = min(0.3, len(result.components) * 0.05)
            base_confidence += component_factor
            
            return min(1.0, base_confidence)
        
        return 0.5
    
    def is_parallelizable(self) -> bool:
        """Check if strategy can be parallelized"""
        return False  # Default to sequential
    
    def is_deterministic(self) -> bool:
        """Check if strategy is deterministic"""
        return True  # Default to deterministic
    
    def get_success_rate(self) -> float:
        """Get strategy success rate"""
        if self.execution_count == 0:
            return 0.5
        return self.success_count / self.execution_count
    
    def get_average_execution_time(self) -> float:
        """Get average execution time"""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count


class ExactDecomposition(DecompositionStrategy):
    """Direct pattern matching"""
    
    def __init__(self):
        """Initialize exact decomposition"""
        super().__init__("ExactDecomposition", StrategyType.EXACT)
        self.pattern_library = self._load_pattern_library()
        self.match_threshold = 0.95
        
    def apply(self, problem_graph) -> DecompositionResult:
        """Apply exact pattern matching"""
        start_time = time.time()
        self.execution_count += 1
        
        # Find exact matches
        matches = self.find_exact_pattern_matches(problem_graph)
        
        # Create components from matches
        components = []
        for match in matches:
            component = {
                'pattern_id': match.pattern_id,
                'nodes': match.matched_nodes,
                'type': 'exact_match',
                'confidence': match.similarity_score
            }
            components.append(component)
        
        # FIX: If no matches, create default decomposition based on graph structure
        if not components:
            logger.debug("No exact matches found, generating structural fallback")
            
            # Try to break down by graph structure
            if hasattr(problem_graph, 'nodes') and problem_graph.nodes:
                nodes = list(problem_graph.nodes.keys())
                
                # Group nodes into manageable chunks
                chunk_size = max(1, len(nodes) // 3) if len(nodes) > 3 else 1
                
                for i in range(0, len(nodes), chunk_size):
                    chunk = nodes[i:i + chunk_size]
                    component = {
                        'pattern_id': 'sequential_processing',
                        'nodes': chunk,
                        'type': 'node_group',
                        'confidence': 0.4,  # Low confidence for fallback
                        'fallback': True
                    }
                    components.append(component)
        
        # Calculate overall confidence
        if matches:
            confidence = np.mean([m.similarity_score for m in matches])
            self.success_count += 1
        else:
            confidence = 0.4 if components else 0.2  # Low confidence for fallback
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        return DecompositionResult(
            components=components,
            confidence=confidence,
            strategy_type=self.strategy_type.value,
            execution_time=execution_time,
            metadata={
                'matches_found': len(matches),
                'used_fallback': len(matches) == 0
            }
        )
    
    def find_exact_pattern_matches(self, problem_graph) -> List[PatternMatch]:
        """
        Find exact pattern matches in problem graph
        
        Args:
            problem_graph: Graph to search
            
        Returns:
            List of pattern matches
        """
        matches = []
        
        # Convert problem graph to NetworkX if needed
        if hasattr(problem_graph, 'to_networkx'):
            G = problem_graph.to_networkx()
        else:
            G = problem_graph
        
        # Search for each pattern in library
        for pattern_id, pattern in self.pattern_library.items():
            # Try to find isomorphic subgraph
            if self._is_subgraph_match(G, pattern):
                match = PatternMatch(
                    pattern_id=pattern_id,
                    similarity_score=1.0,
                    matched_nodes=list(G.nodes()) if hasattr(G, 'nodes') else []
                )
                matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        
        return matches
    
    def _load_pattern_library(self) -> Dict[str, Any]:
        """Load pattern library for exact matching"""
        library = {}
        
        # Create some common patterns
        patterns = {
            'linear': self._create_linear_pattern(4),
            'tree': self._create_tree_pattern(3),
            'cycle': self._create_cycle_pattern(4),
            'star': self._create_star_pattern(5)
        }
        
        for name, pattern in patterns.items():
            library[name] = pattern
        
        return library
    
    def _is_subgraph_match(self, G, pattern) -> bool:
        """Check if pattern is subgraph of G"""
        if NETWORKX_AVAILABLE:
            try:
                # Use NetworkX subgraph isomorphism
                from networkx.algorithms import isomorphism
                matcher = isomorphism.DiGraphMatcher(G, pattern)
                return matcher.subgraph_is_isomorphic()
            except:
                pass
        
        # Fallback to simple check
        if hasattr(pattern, 'nodes') and hasattr(G, 'nodes'):
            return len(list(pattern.nodes())) <= len(list(G.nodes()))
        return False
    
    def _create_linear_pattern(self, size: int):
        """Create linear chain pattern"""
        if NETWORKX_AVAILABLE:
            G = nx.DiGraph()
        else:
            G = MockGraph()
        
        for i in range(size - 1):
            G.add_edge(i, i + 1)
        return G
    
    def _create_tree_pattern(self, depth: int):
        """Create tree pattern"""
        if NETWORKX_AVAILABLE:
            G = nx.DiGraph()
        else:
            G = MockGraph()
            
        node_id = 0
        queue = [(0, 0)]  # (node, level)
        
        while queue and queue[0][1] < depth:
            parent, level = queue.pop(0)
            for _ in range(2):  # Binary tree
                node_id += 1
                G.add_edge(parent, node_id)
                if level + 1 < depth:
                    queue.append((node_id, level + 1))
        
        return G
    
    def _create_cycle_pattern(self, size: int):
        """Create cycle pattern"""
        if NETWORKX_AVAILABLE:
            G = nx.DiGraph()
        else:
            G = MockGraph()
            
        for i in range(size):
            G.add_edge(i, (i + 1) % size)
        return G
    
    def _create_star_pattern(self, size: int):
        """Create star pattern"""
        if NETWORKX_AVAILABLE:
            G = nx.DiGraph()
        else:
            G = MockGraph()
            
        for i in range(1, size):
            G.add_edge(0, i)
        return G


class SemanticDecomposition(DecompositionStrategy):
    """Semantic similarity matching"""
    
    def __init__(self, semantic_model=None):
        """Initialize semantic decomposition"""
        super().__init__("SemanticDecomposition", StrategyType.SEMANTIC)
        self.semantic_model = semantic_model
        self.similarity_threshold = 0.7
        self.embedding_cache = {}
        self._cache_lock = threading.RLock()
        self.max_cache_size = 1000
        
    def apply(self, problem_graph) -> DecompositionResult:
        """Apply semantic decomposition"""
        start_time = time.time()
        self.execution_count += 1
        
        # Find semantic matches
        matches = self.find_semantic_matches(problem_graph)
        
        # Create components
        components = []
        for match in matches:
            component = {
                'type': 'semantic_match',
                'similarity': match['similarity'],
                'concept': match['concept'],
                'nodes': match['nodes']
            }
            components.append(component)
        
        # FIX: If no semantic matches, create basic decomposition
        if not components:
            logger.debug("No semantic matches found, generating basic decomposition")
            
            if hasattr(problem_graph, 'nodes') and problem_graph.nodes:
                # Group all nodes as one semantic unit
                component = {
                    'type': 'semantic_group',
                    'similarity': 0.5,
                    'concept': 'general_processing',
                    'nodes': list(problem_graph.nodes.keys()),
                    'fallback': True
                }
                components.append(component)
        
        # Calculate confidence
        if matches:
            confidence = np.mean([m['similarity'] for m in matches])
            self.success_count += 1
        else:
            confidence = 0.3  # Low confidence for fallback
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        return DecompositionResult(
            components=components,
            confidence=confidence,
            strategy_type=self.strategy_type.value,
            execution_time=execution_time,
            metadata={'matches_found': len(matches), 'used_fallback': len(matches) == 0}
        )
    
    def find_semantic_matches(self, remaining_graph) -> List[Dict[str, Any]]:
        """
        Find semantic matches in graph
        
        Args:
            remaining_graph: Graph to search
            
        Returns:
            List of semantic matches
        """
        matches = []
        
        # Get node embeddings
        node_embeddings = self._get_node_embeddings(remaining_graph)
        
        # Cluster semantically similar nodes
        clusters = self._cluster_by_similarity(node_embeddings)
        
        # Create matches from clusters
        for cluster_id, cluster_nodes in clusters.items():
            if len(cluster_nodes) >= 2:
                # Calculate cluster cohesion
                cohesion = self._calculate_cluster_cohesion(
                    cluster_nodes,
                    node_embeddings
                )
                
                if cohesion >= self.similarity_threshold:
                    match = {
                        'concept': f"semantic_cluster_{cluster_id}",
                        'nodes': cluster_nodes,
                        'similarity': cohesion
                    }
                    matches.append(match)
        
        return matches
    
    def _get_node_embeddings(self, graph) -> Dict[str, np.ndarray]:
        """Get semantic embeddings for nodes"""
        embeddings = {}
        
        # Convert graph to dict if needed
        if hasattr(graph, 'nodes'):
            nodes_method = getattr(graph, 'nodes', lambda: {})
            if callable(nodes_method):
                nodes = {n: {} for n in nodes_method()}
            else:
                nodes = nodes_method
        else:
            nodes = graph.get('nodes', {})
        
        for node_id, node_data in nodes.items():
            # Check cache
            cache_key = str(node_data)
            
            with self._cache_lock:
                if cache_key in self.embedding_cache:
                    embeddings[node_id] = self.embedding_cache[cache_key]
                else:
                    # Generate embedding
                    embedding = self._generate_embedding(node_data)
                    embeddings[node_id] = embedding
                    
                    # FIXED: More aggressive cache limiting - check EVERY time and remove entries until under limit
                    while len(self.embedding_cache) >= self.max_cache_size:
                        # Remove oldest entry
                        if self.embedding_cache:
                            oldest_key = next(iter(self.embedding_cache))
                            del self.embedding_cache[oldest_key]
                        else:
                            break
                    
                    # Now add the new entry
                    self.embedding_cache[cache_key] = embedding
        
        return embeddings
    
    def _generate_embedding(self, node_data: Any) -> np.ndarray:
        """Generate semantic embedding for node"""
        # Simple embedding based on node properties
        # In production, would use actual semantic model
        
        text = str(node_data)
        
        # Create feature vector
        features = []
        
        # Text length feature
        features.append(len(text) / 100.0)
        
        # Character distribution features
        for char in 'aeiou':
            features.append(text.lower().count(char) / max(1, len(text)))
        
        # Hash-based features
        hash_val = hashlib.md5(text.encode()).hexdigest()
        for i in range(0, 32, 4):
            features.append(int(hash_val[i:i+4], 16) / 65535.0)
        
        # Pad to fixed size
        embedding_size = 128
        while len(features) < embedding_size:
            features.append(0.0)
        
        return np.array(features[:embedding_size])
    
    def _cluster_by_similarity(self, embeddings: Dict[str, np.ndarray]) -> Dict[int, List[str]]:
        """Cluster nodes by semantic similarity"""
        if not embeddings:
            return {}
        
        # Limit processing size
        if len(embeddings) > 1000:
            logger.warning("Too many nodes for clustering: %d, limiting to 1000", len(embeddings))
            node_ids = list(embeddings.keys())[:1000]
        else:
            node_ids = list(embeddings.keys())
        
        # Convert to matrix
        embedding_matrix = np.array([embeddings[nid] for nid in node_ids])
        
        # Simple clustering using similarity threshold
        clusters = {}
        cluster_id = 0
        assigned = set()
        max_clusters = 100
        
        for i, node_id in enumerate(node_ids):
            if node_id in assigned:
                continue
            
            # FIXED: Check cluster limit BEFORE creating new cluster
            if cluster_id >= max_clusters:
                logger.warning("Reached maximum cluster limit: %d", max_clusters)
                break
            
            # Start new cluster
            cluster = [node_id]
            assigned.add(node_id)
            
            # Find similar nodes
            for j, other_id in enumerate(node_ids):
                if i != j and other_id not in assigned:
                    if SKLEARN_AVAILABLE:
                        similarity = cosine_similarity(
                            embedding_matrix[i:i+1],
                            embedding_matrix[j:j+1]
                        )[0, 0]
                    else:
                        # Use fallback
                        similarity = cosine_similarity(
                            embedding_matrix[i:i+1],
                            embedding_matrix[j:j+1]
                        )[0, 0]
                    
                    if similarity >= self.similarity_threshold:
                        cluster.append(other_id)
                        assigned.add(other_id)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        return clusters
    
    def _calculate_cluster_cohesion(self, cluster_nodes: List[str],
                                   embeddings: Dict[str, np.ndarray]) -> float:
        """Calculate semantic cohesion of cluster"""
        if len(cluster_nodes) < 2:
            return 1.0
        
        similarities = []
        for i, node1 in enumerate(cluster_nodes):
            for node2 in cluster_nodes[i+1:]:
                if node1 in embeddings and node2 in embeddings:
                    if SKLEARN_AVAILABLE:
                        sim = cosine_similarity(
                            embeddings[node1].reshape(1, -1),
                            embeddings[node2].reshape(1, -1)
                        )[0, 0]
                    else:
                        # Use fallback
                        sim = cosine_similarity(
                            embeddings[node1].reshape(1, -1),
                            embeddings[node2].reshape(1, -1)
                        )[0, 0]
                    similarities.append(sim)
        
        if similarities:
            return np.mean(similarities)
        return 0.0


class StructuralDecomposition(DecompositionStrategy):
    """Structure-based matching"""
    
    def __init__(self):
        """Initialize structural decomposition"""
        super().__init__("StructuralDecomposition", StrategyType.STRUCTURAL)
        self.structural_patterns = self._load_structural_patterns()
        
    def apply(self, problem_graph) -> DecompositionResult:
        """Apply structural decomposition"""
        start_time = time.time()
        self.execution_count += 1
        
        # Find structural matches
        matches = self.find_structural_matches(problem_graph)
        
        # Create components
        components = []
        for match in matches:
            component = {
                'type': 'structural_match',
                'structure': match['structure'],
                'nodes': match['nodes'],
                'confidence': match['confidence']
            }
            components.append(component)
        
        # FIX: If no structural matches, create basic sequential decomposition
        if not components:
            logger.debug("No structural matches found, generating sequential decomposition")
            
            if hasattr(problem_graph, 'nodes') and problem_graph.nodes:
                nodes = list(problem_graph.nodes.keys())
                
                # Create sequential processing components
                chunk_size = max(1, len(nodes) // 3) if len(nodes) > 3 else len(nodes)
                
                for i in range(0, len(nodes), chunk_size):
                    chunk = nodes[i:i + chunk_size]
                    component = {
                        'type': 'sequential_group',
                        'structure': 'sequential',
                        'nodes': chunk,
                        'confidence': 0.5,
                        'fallback': True
                    }
                    components.append(component)
        
        # Calculate confidence
        if matches:
            confidence = np.mean([m['confidence'] for m in matches])
            self.success_count += 1
        else:
            confidence = 0.5  # Moderate confidence for fallback
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        return DecompositionResult(
            components=components,
            confidence=confidence,
            strategy_type=self.strategy_type.value,
            execution_time=execution_time,
            metadata={'matches_found': len(matches), 'used_fallback': len(matches) == 0}
        )
    
    def find_structural_matches(self, remaining_graph) -> List[Dict[str, Any]]:
        """
        Find structural patterns in graph
        
        Args:
            remaining_graph: Graph to analyze
            
        Returns:
            List of structural matches
        """
        matches = []
        
        # Convert to graph object
        if hasattr(remaining_graph, 'to_networkx'):
            G = remaining_graph.to_networkx()
        elif NETWORKX_AVAILABLE:
            G = nx.DiGraph()
        else:
            G = MockGraph()
        
        # Check for each structural pattern
        for pattern_name, pattern_checker in self.structural_patterns.items():
            match_result = pattern_checker(G)
            if match_result:
                matches.append({
                    'structure': pattern_name,
                    'nodes': match_result['nodes'],
                    'confidence': match_result['confidence']
                })
        
        return matches
    
    def _load_structural_patterns(self) -> Dict[str, callable]:
        """Load structural pattern checkers"""
        return {
            'hierarchical': self._check_hierarchical,
            'modular': self._check_modular,
            'pipeline': self._check_pipeline,
            'recursive': self._check_recursive,
            'parallel': self._check_parallel
        }
    
    def _check_hierarchical(self, G) -> Optional[Dict[str, Any]]:
        """Check for hierarchical structure"""
        if not hasattr(G, 'nodes'):
            return None
            
        nodes_list = list(G.nodes())
        if not nodes_list:
            return None
        
        # Check if DAG
        if NETWORKX_AVAILABLE:
            if not nx.is_directed_acyclic_graph(G):
                return None
        
        # Find root nodes (no incoming edges)
        roots = []
        for n in nodes_list:
            has_incoming = False
            if hasattr(G, 'in_degree'):
                in_deg = G.in_degree(n) if callable(G.in_degree) else G.in_degree.get(n, 0)
                if in_deg == 0:
                    roots.append(n)
            else:
                # Fallback: check edges manually
                for source, target in (G.edges() if hasattr(G, 'edges') else []):
                    if target == n:
                        has_incoming = True
                        break
                if not has_incoming:
                    roots.append(n)
        
        if roots:
            # Calculate hierarchy depth
            if NETWORKX_AVAILABLE and hasattr(nx, 'single_source_shortest_path_length'):
                depths = nx.single_source_shortest_path_length(G, roots[0])
            else:
                depths = {n: 1 for n in nodes_list}
            
            max_depth = max(depths.values()) if depths else 0
            
            if max_depth >= 2:
                return {
                    'nodes': nodes_list,
                    'confidence': min(1.0, max_depth / 5.0),
                    'roots': roots,
                    'depth': max_depth
                }
        
        return None
    
    def _check_modular(self, G) -> Optional[Dict[str, Any]]:
        """Check for modular structure"""
        if not hasattr(G, 'nodes'):
            return None
            
        nodes_list = list(G.nodes())
        if len(nodes_list) < 4:
            return None
        
        # Find strongly connected components
        if NETWORKX_AVAILABLE:
            sccs = list(nx.strongly_connected_components(G))
        else:
            # Simplified - treat each node as component
            sccs = [{n} for n in nodes_list]
        
        if len(sccs) >= 2:
            # Calculate modularity score
            modularity = len(sccs) / max(1, len(nodes_list) / 3)
            
            return {
                'nodes': nodes_list,
                'confidence': min(1.0, modularity),
                'modules': [list(scc) for scc in sccs]
            }
        
        return None
    
    def _check_pipeline(self, G) -> Optional[Dict[str, Any]]:
        """Check for pipeline structure"""
        if not hasattr(G, 'nodes'):
            return None
            
        nodes_list = list(G.nodes())
        if not nodes_list:
            return None
        
        # Check for linear chain
        if hasattr(G, 'degree'):
            if callable(G.degree):
                degrees = dict(G.degree())
            else:
                degrees = dict(G.degree)
        else:
            degrees = {n: 2 for n in nodes_list}  # Default
        
        # Count nodes with degree 2 (middle of pipeline)
        middle_nodes = sum(1 for d in degrees.values() if d == 2)
        
        if middle_nodes >= len(nodes_list) * 0.6:
            return {
                'nodes': nodes_list,
                'confidence': middle_nodes / len(nodes_list)
            }
        
        return None
    
    def _check_recursive(self, G) -> Optional[Dict[str, Any]]:
        """Check for recursive structure"""
        if not hasattr(G, 'nodes'):
            return None
            
        nodes_list = list(G.nodes())
        if not nodes_list:
            return None
        
        # Look for cycles (recursion indicator)
        if NETWORKX_AVAILABLE:
            try:
                cycles = list(nx.simple_cycles(G))
                if cycles:
                    return {
                        'nodes': nodes_list,
                        'confidence': min(1.0, len(cycles) / len(nodes_list)),
                        'cycles': cycles
                    }
            except:
                pass
        
        return None
    
    def _check_parallel(self, G) -> Optional[Dict[str, Any]]:
        """Check for parallel structure"""
        if not hasattr(G, 'nodes'):
            return None
            
        nodes_list = list(G.nodes())
        if len(nodes_list) < 3:
            return None
        
        # Look for nodes with same predecessors and successors
        parallel_groups = defaultdict(list)
        
        for node in nodes_list:
            if hasattr(G, 'predecessors') and hasattr(G, 'successors'):
                preds = tuple(sorted(G.predecessors(node)))
                succs = tuple(sorted(G.successors(node)))
                key = (preds, succs)
                parallel_groups[key].append(node)
        
        # Find groups with multiple nodes
        parallel_sets = [nodes for nodes in parallel_groups.values() if len(nodes) > 1]
        
        if parallel_sets:
            total_parallel = sum(len(s) for s in parallel_sets)
            confidence = total_parallel / len(nodes_list)
            
            return {
                'nodes': nodes_list,
                'confidence': confidence,
                'parallel_groups': parallel_sets
            }
        
        return None


class SyntheticBridging(DecompositionStrategy):
    """Generate synthetic bridges for unknowns"""
    
    def __init__(self):
        """Initialize synthetic bridging"""
        super().__init__("SyntheticBridging", StrategyType.SYNTHETIC)
        self.mutation_rate = 0.1
        self.bridge_templates = self._load_bridge_templates()
        
    def apply(self, problem_graph) -> DecompositionResult:
        """Apply synthetic bridging"""
        start_time = time.time()
        self.execution_count += 1
        
        # Identify unknown subgraphs
        unknown_subgraphs = self._identify_unknown_subgraphs(problem_graph)
        
        # Generate bridges
        bridges = []
        for subgraph in unknown_subgraphs:
            bridge = self.generate_synthetic_bridges(subgraph)
            if bridge:
                bridges.extend(bridge)
        
        # FIX: If no bridges generated, create default synthetic solution
        if not bridges:
            logger.debug("No synthetic bridges generated, creating default")
            
            if hasattr(problem_graph, 'nodes') and problem_graph.nodes:
                nodes = list(problem_graph.nodes.keys())
                
                # Create synthetic bridge for all nodes
                bridge = {
                    'type': 'synthetic_bridge',
                    'template': 'generic',
                    'nodes': nodes,
                    'structure': {'type': 'synthetic', 'method': 'default'},
                    'confidence': 0.4,
                    'synthetic': True,
                    'fallback': True
                }
                bridges.append(bridge)
        
        # Calculate confidence (lower for synthetic)
        confidence = 0.4 if bridges else 0.2
        
        if bridges:
            self.success_count += 1
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        return DecompositionResult(
            components=bridges,
            confidence=confidence,
            strategy_type=self.strategy_type.value,
            execution_time=execution_time,
            metadata={'bridges_generated': len(bridges)}
        )
    
    def generate_synthetic_bridges(self, unknown_subgraph) -> List[Dict[str, Any]]:
        """
        Generate synthetic bridges for unknown subgraph
        
        Args:
            unknown_subgraph: Subgraph needing bridge
            
        Returns:
            List of synthetic bridge components
        """
        bridges = []
        
        # Analyze subgraph structure
        structure = self._analyze_structure(unknown_subgraph)
        
        # Select appropriate template
        template = self._select_template(structure)
        
        if template:
            # Mutate template to fit
            mutated = self.mutate_pattern(template, structure)
            
            # Create bridge component
            bridge = {
                'type': 'synthetic_bridge',
                'template': template['name'],
                'structure': mutated,
                'confidence': 0.5,
                'synthetic': True
            }
            bridges.append(bridge)
        
        # Try to generate alternative bridges
        alternatives = self._generate_alternatives(structure)
        bridges.extend(alternatives)
        
        return bridges
    
    def mutate_pattern(self, pattern: Dict[str, Any], target_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate pattern to match target structure
        
        Args:
            pattern: Base pattern
            target_structure: Target structure to match
            
        Returns:
            Mutated pattern
        """
        mutated = copy.deepcopy(pattern)
        
        # Adjust size
        if 'size' in target_structure:
            mutated['size'] = target_structure['size']
        
        # Apply random mutations
        if np.random.random() < self.mutation_rate:
            mutated = self._apply_random_mutation(mutated)
        
        # Adjust connectivity
        if 'connectivity' in target_structure:
            mutated['connectivity'] = target_structure['connectivity']
        
        return mutated
    
    def _identify_unknown_subgraphs(self, problem_graph) -> List[Any]:
        """Identify subgraphs that are unknown/unmatched"""
        unknown = []
        
        # Simple heuristic: disconnected components or low-confidence regions
        if hasattr(problem_graph, 'to_networkx'):
            G = problem_graph.to_networkx()
        elif NETWORKX_AVAILABLE:
            G = problem_graph if isinstance(problem_graph, nx.Graph) else nx.DiGraph()
        else:
            return []
        
        # Find weakly connected components
        if NETWORKX_AVAILABLE:
            components = list(nx.weakly_connected_components(G))
        else:
            # Simplified - treat whole graph as one component
            if hasattr(G, 'nodes'):
                components = [set(G.nodes())]
            else:
                components = []
        
        for component in components:
            if len(component) <= 3:  # Small components are likely unknown
                subgraph = G.subgraph(component) if hasattr(G, 'subgraph') else G
                unknown.append(subgraph)
        
        return unknown
    
    def _analyze_structure(self, subgraph) -> Dict[str, Any]:
        """Analyze structure of subgraph"""
        structure = {
            'size': 1,
            'edges': 0,
            'connectivity': 'sparse'
        }
        
        # FIXED: Handle both callable and dict/attribute access for nodes and edges
        if hasattr(subgraph, 'nodes') and hasattr(subgraph, 'edges'):
            # Get nodes list
            if callable(subgraph.nodes):
                nodes_list = list(subgraph.nodes())
            else:
                nodes_list = list(subgraph.nodes.keys()) if isinstance(subgraph.nodes, dict) else list(subgraph.nodes)
            
            # Get edges list
            if callable(subgraph.edges):
                edges_list = list(subgraph.edges())
            else:
                edges_list = list(subgraph.edges) if hasattr(subgraph.edges, '__iter__') else []
            
            nodes_count = len(nodes_list)
            edges_count = len(edges_list)
            
            structure['size'] = nodes_count
            structure['edges'] = edges_count
            
            if nodes_count > 1:
                density = edges_count / (nodes_count * (nodes_count - 1) + 1)
                if density > 0.5:
                    structure['connectivity'] = 'dense'
                elif density > 0.2:
                    structure['connectivity'] = 'medium'
        
        return structure
    
    def _select_template(self, structure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select appropriate bridge template"""
        best_template = None
        best_score = 0
        
        for template in self.bridge_templates:
            score = self._calculate_template_fit(template, structure)
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template if best_score > 0.3 else None
    
    def _calculate_template_fit(self, template: Dict[str, Any], 
                               structure: Dict[str, Any]) -> float:
        """Calculate how well template fits structure"""
        score = 0.5  # Base score
        
        # Size similarity
        if 'size' in template and 'size' in structure:
            size_diff = abs(template['size'] - structure['size'])
            score -= size_diff * 0.1
        
        # Connectivity match
        if template.get('connectivity') == structure.get('connectivity'):
            score += 0.3
        
        return max(0, min(1, score))
    
    def _generate_alternatives(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative synthetic bridges"""
        alternatives = []
        
        # Simple linear bridge
        if structure['size'] <= 5:
            alternatives.append({
                'type': 'synthetic_bridge',
                'template': 'linear',
                'structure': {'type': 'sequence', 'length': structure['size']},
                'confidence': 0.3,
                'synthetic': True
            })
        
        # Parallel bridge for larger structures
        if structure['size'] >= 3:
            alternatives.append({
                'type': 'synthetic_bridge',
                'template': 'parallel',
                'structure': {'type': 'parallel', 'branches': min(3, structure['size'])},
                'confidence': 0.3,
                'synthetic': True
            })
        
        return alternatives
    
    def _apply_random_mutation(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutation to pattern"""
        mutated = copy.deepcopy(pattern)
        
        # Random mutation types
        mutation_type = np.random.choice(['size', 'structure', 'property'])
        
        if mutation_type == 'size' and 'size' in mutated:
            mutated['size'] = max(1, mutated['size'] + np.random.randint(-2, 3))
        elif mutation_type == 'structure':
            mutated['variant'] = np.random.choice(['A', 'B', 'C'])
        elif mutation_type == 'property':
            mutated['modified'] = True
        
        return mutated
    
    def _load_bridge_templates(self) -> List[Dict[str, Any]]:
        """Load bridge templates"""
        return [
            {'name': 'simple', 'size': 2, 'connectivity': 'sparse'},
            {'name': 'chain', 'size': 4, 'connectivity': 'medium'},
            {'name': 'hub', 'size': 5, 'connectivity': 'dense'},
            {'name': 'mesh', 'size': 6, 'connectivity': 'dense'}
        ]


class AnalogicalDecomposition(DecompositionStrategy):
    """Analogy-based decomposition"""
    
    def __init__(self):
        """Initialize analogical decomposition"""
        super().__init__("AnalogicalDecomposition", StrategyType.ANALOGICAL)
        self.analogy_database = self._load_analogy_database()
        self.similarity_threshold = 0.6
        
    def apply(self, problem_graph) -> DecompositionResult:
        """Apply analogical decomposition"""
        start_time = time.time()
        self.execution_count += 1
        
        # Find analogies
        analogies = self._find_analogies(problem_graph)
        
        # Create components from analogies
        components = []
        for analogy in analogies:
            component = {
                'type': 'analogical',
                'source_domain': analogy['source_domain'],
                'target_mapping': analogy['mapping'],
                'confidence': analogy['confidence']
            }
            components.append(component)
        
        # FIX: If no analogies found, create generic analogy-based decomposition
        if not components:
            logger.debug("No analogies found, creating generic analogy-based decomposition")
            
            # Create generic analogical component
            component = {
                'type': 'analogical',
                'source_domain': 'general',
                'target_mapping': {'method': 'generic_analogy'},
                'confidence': 0.3,
                'fallback': True
            }
            components.append(component)
        
        # Calculate confidence
        if analogies:
            confidence = np.mean([a['confidence'] for a in analogies])
            self.success_count += 1
        else:
            confidence = 0.3  # Low confidence for fallback
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        return DecompositionResult(
            components=components,
            confidence=confidence,
            strategy_type=self.strategy_type.value,
            execution_time=execution_time,
            metadata={'analogies_found': len(analogies), 'used_fallback': len(analogies) == 0}
        )
    
    def _find_analogies(self, problem_graph) -> List[Dict[str, Any]]:
        """Find analogies for problem"""
        analogies = []
        
        # Get problem features
        problem_features = self._extract_features(problem_graph)
        
        # Search analogy database
        for source_domain, source_cases in self.analogy_database.items():
            for case in source_cases:
                similarity = self._calculate_similarity(problem_features, case['features'])
                
                if similarity >= self.similarity_threshold:
                    # Create mapping
                    mapping = self._create_mapping(problem_features, case['features'])
                    
                    analogy = {
                        'source_domain': source_domain,
                        'source_case': case['name'],
                        'mapping': mapping,
                        'confidence': similarity
                    }
                    analogies.append(analogy)
        
        # Sort by confidence
        analogies.sort(key=lambda a: a['confidence'], reverse=True)
        
        return analogies[:5]  # Return top 5 analogies
    
    def _extract_features(self, problem_graph) -> Dict[str, Any]:
        """Extract features for analogy matching"""
        features = {
            'size': 0,
            'complexity': 0,
            'structure_type': 'unknown',
            'has_cycles': False,
            'is_hierarchical': False
        }
        
        if hasattr(problem_graph, 'to_networkx'):
            G = problem_graph.to_networkx()
            if hasattr(G, 'nodes') and hasattr(G, 'edges'):
                features['size'] = len(list(G.nodes()))
                features['complexity'] = len(list(G.edges())) / max(1, features['size'])
                
                if NETWORKX_AVAILABLE:
                    features['has_cycles'] = not nx.is_directed_acyclic_graph(G)
                    
                    # Check if hierarchical
                    if nx.is_directed_acyclic_graph(G):
                        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
                        features['is_hierarchical'] = len(roots) <= 2
        
        return features
    
    def _calculate_similarity(self, features1: Dict[str, Any], 
                            features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature sets"""
        similarity = 0.0
        count = 0
        
        for key in features1:
            if key in features2:
                if isinstance(features1[key], bool):
                    if features1[key] == features2[key]:
                        similarity += 1.0
                elif isinstance(features1[key], (int, float)):
                    diff = abs(features1[key] - features2[key])
                    similarity += max(0, 1 - diff / 10)
                elif features1[key] == features2[key]:
                    similarity += 1.0
                
                count += 1
        
        return similarity / max(1, count)
    
    def _create_mapping(self, target_features: Dict[str, Any],
                       source_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create mapping from source to target"""
        mapping = {
            'feature_correspondence': {},
            'transformations': []
        }
        
        # Map corresponding features
        for key in target_features:
            if key in source_features:
                mapping['feature_correspondence'][key] = {
                    'source': source_features[key],
                    'target': target_features[key]
                }
        
        # Identify needed transformations
        if target_features.get('size', 0) > source_features.get('size', 0):
            mapping['transformations'].append('scale_up')
        elif target_features.get('size', 0) < source_features.get('size', 0):
            mapping['transformations'].append('scale_down')
        
        return mapping
    
    def _load_analogy_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load database of analogies"""
        return {
            'sorting': [
                {
                    'name': 'merge_sort',
                    'features': {'size': 100, 'complexity': 2, 'is_hierarchical': True}
                },
                {
                    'name': 'quick_sort',
                    'features': {'size': 100, 'complexity': 2, 'is_hierarchical': False}
                }
            ],
            'optimization': [
                {
                    'name': 'gradient_descent',
                    'features': {'size': 50, 'complexity': 3, 'has_cycles': True}
                }
            ],
            'search': [
                {
                    'name': 'breadth_first',
                    'features': {'size': 30, 'complexity': 1.5, 'is_hierarchical': False}
                }
            ]
        }


class BruteForceSearch(DecompositionStrategy):
    """Last resort exhaustive search"""
    
    def __init__(self, max_depth: int = 3):
        """Initialize brute force search"""
        super().__init__("BruteForceSearch", StrategyType.BRUTE_FORCE)
        self.max_depth = max_depth
        self.max_iterations = 1000
        self._iteration_count = 0
        
    def apply(self, problem_graph) -> DecompositionResult:
        """Apply brute force decomposition"""
        start_time = time.time()
        self.execution_count += 1
        
        # Exhaustive search for decomposition
        best_decomposition = self._exhaustive_search(problem_graph)
        
        # Create components
        components = []
        if best_decomposition:
            for i, part in enumerate(best_decomposition):
                component = {
                    'type': 'brute_force',
                    'part': i,
                    'content': part,
                    'confidence': 0.3  # Low confidence for brute force
                }
                components.append(component)
            self.success_count += 1
        
        # FIX: Ensure we always have at least one component
        if not components:
            logger.debug("Brute force produced no decomposition, creating fallback")
            
            # Create single component containing whole problem
            component = {
                'type': 'brute_force',
                'part': 0,
                'content': problem_graph,
                'confidence': 0.2,
                'fallback': True
            }
            components.append(component)
        
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        
        return DecompositionResult(
            components=components,
            confidence=0.3 if components else 0.2,
            strategy_type=self.strategy_type.value,
            execution_time=execution_time,
            metadata={'iterations': min(self.max_iterations, self._iteration_count)}
        )
    
    def is_deterministic(self) -> bool:
        """Brute force is non-deterministic due to cutoffs"""
        return False
    
    def _exhaustive_search(self, problem_graph) -> Optional[List[Any]]:
        """Perform exhaustive search for decomposition"""
        self._iteration_count = 0
        
        # Get problem size
        problem_size = 1
        if hasattr(problem_graph, 'to_networkx'):
            G = problem_graph.to_networkx()
            if hasattr(G, 'nodes'):
                problem_size = len(list(G.nodes()))
        elif hasattr(problem_graph, 'nodes'):
            problem_size = len(list(problem_graph.nodes()))
        
        # Try different decomposition sizes
        for num_parts in range(2, min(problem_size + 1, self.max_depth + 1)):
            decomposition = self._try_decomposition(problem_graph, num_parts)
            
            if decomposition and self._is_valid_decomposition(decomposition):
                return decomposition
            
            self._iteration_count += 1
            if self._iteration_count >= self.max_iterations:
                break
        
        # Fallback to trivial decomposition
        return [problem_graph]
    
    def _try_decomposition(self, problem_graph, num_parts: int) -> Optional[List[Any]]:
        """Try to decompose into num_parts"""
        if hasattr(problem_graph, 'to_networkx'):
            G = problem_graph.to_networkx()
        elif hasattr(problem_graph, 'nodes'):
            G = problem_graph
        else:
            return None
            
        if hasattr(G, 'nodes'):
            nodes = list(G.nodes())
            
            if len(nodes) < num_parts:
                return None
            
            # Simple partitioning
            parts = []
            partition_size = len(nodes) // num_parts
            
            for i in range(num_parts):
                start = i * partition_size
                end = start + partition_size if i < num_parts - 1 else len(nodes)
                part_nodes = nodes[start:end]
                
                if part_nodes:
                    if hasattr(G, 'subgraph'):
                        subgraph = G.subgraph(part_nodes)
                    else:
                        subgraph = part_nodes
                    parts.append(subgraph)
            
            return parts if parts else None
        
        return None
    
    def _is_valid_decomposition(self, decomposition: List[Any]) -> bool:
        """Check if decomposition is valid"""
        if not decomposition:
            return False
        
        # Check that parts are non-empty
        for part in decomposition:
            if hasattr(part, 'nodes'):
                nodes_list = list(part.nodes())
                if len(nodes_list) == 0:
                    return False
        
        return True