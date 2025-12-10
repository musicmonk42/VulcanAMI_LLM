"""
decomposition_library.py - Pattern and principle library for problem decomposer
Part of the VULCAN-AGI system
"""

import numpy as np
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import time
import json
import pickle
from pathlib import Path
from enum import Enum
import hashlib
import heapq

# Optional imports with fallbacks
try:
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, using fallback cosine similarity")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, graph features will be limited")

    # Create a simple mock for nx when not available
    class MockGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = []

        def nodes(self):
            return self._nodes.keys()

        def edges(self):
            return self._edges

        def add_node(self, node_id, **attrs):
            self._nodes[node_id] = attrs

        def add_edge(self, source, target, **attrs):
            self._edges.append((source, target, attrs))

        def degree(self):
            return []

    class MockNX:
        Graph = MockGraph
        DiGraph = MockGraph

        @staticmethod
        def density(graph):
            if hasattr(graph, "nodes") and hasattr(graph, "edges"):
                n = len(graph.nodes())
                if n <= 1:
                    return 0
                e = len(graph.edges())
                return e / (n * (n - 1))
            return 0

        @staticmethod
        def is_directed(graph):
            return True

        @staticmethod
        def is_directed_acyclic_graph(graph):
            return True

        @staticmethod
        def dag_longest_path_length(graph):
            return 1

    nx = MockNX()

logger = logging.getLogger(__name__)


# Fallback cosine similarity implementation
def cosine_similarity_fallback(X, Y):
    """Simple cosine similarity implementation"""
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

    similarity = np.dot(X_normalized, Y_normalized.T)
    return similarity


class PatternStatus(Enum):
    """Status of decomposition patterns"""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    PROVEN = "proven"
    FAILED = "failed"


class DomainCategory(Enum):
    """Categories of domains"""

    FREQUENT = "frequent"
    COMMON = "common"
    RARE = "rare"
    NOVEL = "novel"


@dataclass
class Pattern:
    """Pattern representation"""

    pattern_id: str
    structure: Any  # Can be nx.DiGraph or MockGraph
    features: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_signature(self) -> str:
        """Get unique signature for pattern"""
        nodes_count = 0
        edges_count = 0

        if hasattr(self.structure, "nodes") and hasattr(self.structure, "edges"):
            nodes_count = len(list(self.structure.nodes()))
            edges_count = len(list(self.structure.edges()))

        content = json.dumps(
            {
                "nodes": nodes_count,
                "edges": edges_count,
                "features": sorted(self.features.keys()),
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()


@dataclass
class Context:
    """Context for pattern application"""

    domain: str
    problem_type: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)

    def matches(self, other: "Context") -> float:
        """Calculate match score with another context"""
        score = 0.0

        if self.domain == other.domain:
            score += 0.5
        elif self.domain == "general" or other.domain == "general":
            score += 0.2

        if self.problem_type == other.problem_type:
            score += 0.3

        # Check constraint overlap
        if self.constraints and other.constraints:
            common_keys = set(self.constraints.keys()) & set(other.constraints.keys())
            if common_keys:
                matches = sum(
                    1
                    for k in common_keys
                    if self.constraints[k] == other.constraints[k]
                )
                score += 0.2 * (matches / len(common_keys))

        return min(1.0, score)


@dataclass
class DecompositionPrinciple:
    """Reusable decomposition principle"""

    principle_id: str
    name: str
    pattern: Pattern
    applicable_contexts: List[Context] = field(default_factory=list)
    success_rate: float = 0.5
    contraindications: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_success_rate(self):
        """Update success rate based on usage"""
        total = self.success_count + self.failure_count
        if total > 0:
            self.success_rate = self.success_count / total

    def is_applicable(self, context: Context) -> Tuple[bool, float]:
        """Check if principle is applicable to context"""
        # Check contraindications
        for contraindication in self.contraindications:
            if contraindication in str(context):
                return False, 0.0

        # Check context match
        if not self.applicable_contexts:
            return True, 0.5  # No specific contexts means generally applicable

        max_match = 0.0
        for applicable_context in self.applicable_contexts:
            match_score = applicable_context.matches(context)
            max_match = max(max_match, match_score)

        return max_match > 0.3, max_match


@dataclass
class PatternPerformance:
    """Performance tracking for patterns"""

    pattern_signature: str
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    avg_execution_time: float = 0.0
    domains_used: Set[str] = field(default_factory=set)
    last_performance: Optional[float] = None
    failure_reasons: List[str] = field(default_factory=list)

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self.successful_uses + self.failed_uses
        if total == 0:
            return 0.5
        return self.successful_uses / total

    def update(
        self,
        success: bool,
        execution_time: float,
        domain: str = None,
        failure_reason: str = None,
    ):
        """Update performance metrics"""
        self.total_uses += 1

        if success:
            self.successful_uses += 1
            self.last_performance = 1.0
        else:
            self.failed_uses += 1
            self.last_performance = 0.0
            if failure_reason:
                self.failure_reasons.append(failure_reason)
                # Limit failure reasons list
                if len(self.failure_reasons) > 100:
                    self.failure_reasons = self.failure_reasons[-100:]

        # Update average execution time
        self.avg_execution_time = (
            self.avg_execution_time * (self.total_uses - 1) + execution_time
        ) / self.total_uses

        if domain:
            self.domains_used.add(domain)


class DecompositionLibrary:
    """Manages decomposition patterns and principles"""

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize decomposition library

        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = storage_path or Path("decomposition_library")

        # Pattern storage
        self.patterns = {}  # pattern_id -> Pattern
        self.pattern_index = {}  # signature -> pattern_id
        self.pattern_embeddings = {}  # pattern_id -> embedding

        # Principle storage
        self.principles = {}  # principle_id -> DecompositionPrinciple

        # Performance tracking - FIXED: Don't use defaultdict with dataclass requiring args
        self.performance = {}  # signature -> PatternPerformance

        # Domain tracking
        self.domain_patterns = defaultdict(set)  # domain -> set of pattern_ids
        self.pattern_domains = defaultdict(set)  # pattern_id -> set of domains

        # Similarity cache
        self.similarity_cache = {}
        self.cache_size = 1000

        # Statistics
        self.total_patterns = 0
        self.total_principles = 0

        # Thread safety
        self._lock = threading.RLock()

        # Load existing library if available
        self._load_library()

        logger.info(
            "DecompositionLibrary initialized with %d patterns and %d principles",
            len(self.patterns),
            len(self.principles),
        )

    def find_similar(self, subgraph, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar patterns in library

        Args:
            subgraph: Graph to find similar patterns for
            top_k: Number of similar patterns to return

        Returns:
            List of (pattern_id, similarity_score) tuples
        """
        # Extract features from subgraph
        query_features = self._extract_features(subgraph)
        query_embedding = self._get_embedding(query_features)

        # Calculate similarities
        similarities = []

        with self._lock:
            # Enforce cache size limit before adding new entries
            if len(self.similarity_cache) >= self.cache_size:
                # Remove 10% oldest entries (FIFO approximation)
                items = list(self.similarity_cache.items())
                cutoff = int(len(items) * 0.9)
                self.similarity_cache = dict(items[cutoff:])

            for pattern_id, pattern in self.patterns.items():
                # Check cache
                cache_key = (
                    pattern_id,
                    hashlib.md5(str(query_features).encode(), usedforsecurity=False).hexdigest(),
                )

                if cache_key in self.similarity_cache:
                    similarity = self.similarity_cache[cache_key]
                else:
                    # Calculate similarity
                    pattern_embedding = self.pattern_embeddings.get(pattern_id)

                    if pattern_embedding is not None:
                        similarity = self._calculate_similarity(
                            query_embedding, pattern_embedding
                        )
                    else:
                        similarity = self._calculate_structural_similarity(
                            subgraph, pattern.structure
                        )

                    # Cache result
                    if len(self.similarity_cache) < self.cache_size:
                        self.similarity_cache[cache_key] = similarity

                similarities.append((pattern_id, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def reinforce_pattern(self, signature: str, decomposition: Any, performance: float):
        """
        Reinforce successful pattern

        Args:
            signature: Pattern signature
            decomposition: Decomposition that was successful
            performance: Performance score
        """
        with self._lock:
            # FIXED: Create performance object if doesn't exist
            if signature not in self.performance:
                self.performance[signature] = PatternPerformance(
                    pattern_signature=signature
                )

            # Update performance tracking
            perf = self.performance[signature]
            perf.update(
                success=performance > 0.5,
                execution_time=0,  # Would be provided in real implementation
                domain=decomposition.get("domain", "general")
                if isinstance(decomposition, dict)
                else "general",
            )

            # If pattern exists, update its metadata
            if signature in self.pattern_index:
                pattern_id = self.pattern_index[signature]
                pattern = self.patterns.get(pattern_id)

                if pattern:
                    # Update pattern metadata
                    pattern.metadata["reinforcement_count"] = (
                        pattern.metadata.get("reinforcement_count", 0) + 1
                    )
                    pattern.metadata["avg_performance"] = (
                        pattern.metadata.get("avg_performance", 0.5)
                        * pattern.metadata.get("reinforcement_count", 1)
                        + performance
                    ) / (pattern.metadata["reinforcement_count"] + 1)

                    # Promote to proven if consistently successful
                    if perf.get_success_rate() > 0.8 and perf.total_uses > 10:
                        pattern.metadata["status"] = PatternStatus.PROVEN.value

            logger.debug(
                "Reinforced pattern %s with performance %.2f",
                signature[:8],
                performance,
            )

    def mark_failed_pattern(self, signature: str, decomposition: Any, reason: str):
        """
        Mark pattern as failed

        Args:
            signature: Pattern signature
            decomposition: Failed decomposition
            reason: Failure reason
        """
        with self._lock:
            # FIXED: Create performance object if doesn't exist
            if signature not in self.performance:
                self.performance[signature] = PatternPerformance(
                    pattern_signature=signature
                )

            # Update performance tracking
            perf = self.performance[signature]
            perf.update(success=False, execution_time=0, failure_reason=reason)

            # Update pattern status if exists
            if signature in self.pattern_index:
                pattern_id = self.pattern_index[signature]
                pattern = self.patterns.get(pattern_id)

                if pattern:
                    pattern.metadata["failure_count"] = (
                        pattern.metadata.get("failure_count", 0) + 1
                    )

                    # Mark as failed if too many failures
                    if perf.failed_uses > 5 and perf.get_success_rate() < 0.2:
                        pattern.metadata["status"] = PatternStatus.FAILED.value
                    elif perf.get_success_rate() < 0.5:
                        pattern.metadata["status"] = PatternStatus.EXPERIMENTAL.value

            logger.debug("Marked pattern %s as failed: %s", signature[:8], reason)

    def add_principle(self, principle: DecompositionPrinciple):
        """
        Add decomposition principle to library

        Args:
            principle: Principle to add
        """
        with self._lock:
            # FIX: Handle both object and dict-like principle
            if isinstance(principle, dict):
                principle_id = principle.get(
                    "principle_id", principle.get("id", f"principle_{int(time.time())}")
                )
            elif hasattr(principle, "principle_id"):
                principle_id = principle.principle_id
            else:
                principle_id = getattr(principle, "id", f"principle_{int(time.time())}")

            self.principles[principle_id] = principle
            self.total_principles += 1

            # Add associated pattern - handle both object and dict
            pattern = None
            if isinstance(principle, dict):
                pattern = principle.get("pattern")
            elif hasattr(principle, "pattern"):
                pattern = principle.pattern

            if pattern:
                self._add_pattern(pattern)

                # Get pattern_id for domain tracking
                if isinstance(pattern, dict):
                    pattern_id = pattern.get(
                        "pattern_id", f"pattern_{int(time.time())}"
                    )
                elif hasattr(pattern, "pattern_id"):
                    pattern_id = pattern.pattern_id
                else:
                    pattern_id = f"pattern_{int(time.time())}"

                # FIXED: Update domain tracking with pattern_id
                # Get applicable contexts
                applicable_contexts = []
                if isinstance(principle, dict):
                    applicable_contexts = principle.get("applicable_contexts", [])
                    # Handle applicable_domains as well
                    if not applicable_contexts and "applicable_domains" in principle:
                        domains = principle["applicable_domains"]
                        if isinstance(domains, list):
                            for domain in domains:
                                applicable_contexts.append({"domain": domain})
                elif hasattr(principle, "applicable_contexts"):
                    applicable_contexts = principle.applicable_contexts

                for context in applicable_contexts:
                    if isinstance(context, dict):
                        domain = context.get("domain", "general")
                    elif hasattr(context, "domain"):
                        domain = context.domain
                    else:
                        domain = "general"

                    self.domain_patterns[domain].add(pattern_id)
                    self.pattern_domains[pattern_id].add(domain)

            logger.info("Added principle %s to library", principle_id)

    def get_principle(self, principle_id: str) -> Optional[DecompositionPrinciple]:
        """Get principle by ID"""
        with self._lock:
            return self.principles.get(principle_id)

    def get_applicable_principles(
        self, context: Context
    ) -> List[DecompositionPrinciple]:
        """
        Get principles applicable to context

        Args:
            context: Context to match

        Returns:
            List of applicable principles
        """
        applicable = []

        with self._lock:
            for principle in self.principles.values():
                is_applicable, match_score = principle.is_applicable(context)

                if is_applicable:
                    applicable.append((principle, match_score))

        # Sort by match score and success rate
        applicable.sort(key=lambda x: x[1] * x[0].success_rate, reverse=True)

        return [p for p, _ in applicable]

    def _add_pattern(self, pattern):
        """
        Add pattern to library - handles both Pattern objects and dict-like objects

        Args:
            pattern: Pattern object or dict-like object with pattern data
        """
        # Should be called within lock from add_principle

        # FIX: Handle both Pattern objects and dict-like objects
        if isinstance(pattern, dict):
            # Pattern is a dict - extract pattern_id
            pattern_id = pattern.get(
                "pattern_id", f"pattern_{int(time.time())}_{id(pattern)}"
            )

            # Ensure pattern_id is in the dict for future reference
            if "pattern_id" not in pattern:
                pattern["pattern_id"] = pattern_id

            # Store the pattern
            self.patterns[pattern_id] = pattern

            # Get or create signature
            if "signature" in pattern:
                signature = pattern["signature"]
            else:
                # Generate signature from available data
                structure_info = {}
                if "structure" in pattern:
                    struct = pattern["structure"]
                    if hasattr(struct, "nodes") and hasattr(struct, "edges"):
                        structure_info["nodes"] = len(list(struct.nodes()))
                        structure_info["edges"] = len(list(struct.edges()))

                features = pattern.get("features", {})
                if isinstance(features, dict):
                    structure_info["features"] = sorted(features.keys())

                content = json.dumps(structure_info, sort_keys=True)
                signature = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

            # Index by signature
            self.pattern_index[signature] = pattern_id

            # Generate and store embedding from features
            features = pattern.get("features", {})
            if not isinstance(features, dict):
                features = {}
            embedding = self._get_embedding(features)
            self.pattern_embeddings[pattern_id] = embedding

        elif hasattr(pattern, "pattern_id"):
            # Pattern is an object with pattern_id attribute
            pattern_id = pattern.pattern_id

            self.patterns[pattern_id] = pattern

            # Index by signature
            if hasattr(pattern, "get_signature") and callable(pattern.get_signature):
                signature = pattern.get_signature()
            else:
                # Generate signature
                signature = f"signature_{pattern_id}"

            self.pattern_index[signature] = pattern_id

            # Generate and store embedding
            features = pattern.features if hasattr(pattern, "features") else {}
            embedding = self._get_embedding(features)
            self.pattern_embeddings[pattern_id] = embedding

        else:
            # Pattern is some other object - try to extract what we can
            logger.warning(
                "Pattern is neither dict nor has pattern_id attribute: %s",
                type(pattern),
            )

            # Generate a pattern_id
            pattern_id = f"pattern_{int(time.time())}_{id(pattern)}"

            # Try to set pattern_id on the object if possible
            if hasattr(pattern, "__dict__"):
                pattern.__dict__["pattern_id"] = pattern_id

            # Store the pattern as-is
            self.patterns[pattern_id] = pattern

            # Generate a signature
            signature = f"signature_{pattern_id}"
            self.pattern_index[signature] = pattern_id

            # Generate embedding with empty features
            embedding = self._get_embedding({})
            self.pattern_embeddings[pattern_id] = embedding

        self.total_patterns += 1

        logger.debug("Added pattern %s to library", pattern_id)

    def _extract_features(self, graph) -> Dict[str, Any]:
        """Extract features from graph"""
        features = {}

        if NETWORKX_AVAILABLE and isinstance(graph, nx.Graph):
            features["node_count"] = len(graph.nodes())
            features["edge_count"] = len(graph.edges())
            features["density"] = nx.density(graph) if len(graph.nodes()) > 1 else 0

            # Structural features
            if nx.is_directed(graph):
                features["is_dag"] = nx.is_directed_acyclic_graph(graph)
                if features["is_dag"]:
                    try:
                        features["longest_path"] = nx.dag_longest_path_length(graph)
                    except Exception:
                        features["longest_path"] = 1

            # Degree statistics
            degrees = dict(graph.degree())
            if degrees:
                features["avg_degree"] = np.mean(list(degrees.values()))
                features["max_degree"] = max(degrees.values())
        elif hasattr(graph, "nodes") and hasattr(graph, "edges"):
            # Fallback for mock or custom graph
            features["node_count"] = len(list(graph.nodes()))
            features["edge_count"] = len(list(graph.edges()))
            features["density"] = 0.5  # Default

        return features

    def _get_embedding(self, features: Dict[str, Any]) -> np.ndarray:
        """Generate embedding from features"""
        # Fixed-size embedding
        embedding_size = 64
        embedding = np.zeros(embedding_size)

        # Encode features into embedding
        feature_str = json.dumps(features, sort_keys=True)
        hash_val = hashlib.md5(feature_str.encode(), usedforsecurity=False).hexdigest()

        # Convert hash to numeric features
        for i in range(0, min(len(hash_val), embedding_size * 2), 2):
            idx = (i // 2) % embedding_size
            val = int(hash_val[i : i + 2], 16) / 255.0
            embedding[idx] = val

        # Add numeric features directly
        if "node_count" in features:
            embedding[0] = features["node_count"] / 100.0
        if "edge_count" in features:
            embedding[1] = features["edge_count"] / 100.0
        if "density" in features:
            embedding[2] = features["density"]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate similarity between embeddings"""
        if SKLEARN_AVAILABLE:
            return float(
                cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[
                    0, 0
                ]
            )
        else:
            # Use fallback
            return float(
                cosine_similarity_fallback(
                    embedding1.reshape(1, -1), embedding2.reshape(1, -1)
                )[0, 0]
            )

    def _calculate_structural_similarity(self, graph1, graph2) -> float:
        """Calculate structural similarity between graphs"""
        # Check if both are graphs
        has_nodes1 = hasattr(graph1, "nodes")
        has_nodes2 = hasattr(graph2, "nodes")

        if not (has_nodes1 and has_nodes2):
            return 0.0

        # Get node counts
        try:
            nodes1 = len(list(graph1.nodes()))
            nodes2 = len(list(graph2.nodes()))
        except Exception:
            return 0.0

        if nodes1 == 0 and nodes2 == 0:
            return 1.0
        if nodes1 == 0 or nodes2 == 0:
            return 0.0

        # Simple structural similarity based on size and density
        size_sim = 1.0 - abs(nodes1 - nodes2) / max(nodes1, nodes2, 1)

        # Calculate densities
        if (
            NETWORKX_AVAILABLE
            and isinstance(graph1, nx.Graph)
            and isinstance(graph2, nx.Graph)
        ):
            density1 = nx.density(graph1) if nodes1 > 1 else 0
            density2 = nx.density(graph2) if nodes2 > 1 else 0
        else:
            # Simple density calculation
            edges1 = len(list(graph1.edges())) if hasattr(graph1, "edges") else 0
            edges2 = len(list(graph2.edges())) if hasattr(graph2, "edges") else 0
            max_edges1 = nodes1 * (nodes1 - 1) if nodes1 > 1 else 1
            max_edges2 = nodes2 * (nodes2 - 1) if nodes2 > 1 else 1
            density1 = edges1 / max_edges1
            density2 = edges2 / max_edges2

        density_sim = 1.0 - abs(density1 - density2)

        return (size_sim + density_sim) / 2

    def _save_library(self):
        """Save library to disk"""
        if not self.storage_path:
            return

        with self._lock:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Save patterns
            patterns_file = self.storage_path / "patterns.pkl"
            try:
                with open(patterns_file, "wb") as f:
                    pickle.dump(
                        {
                            "patterns": self.patterns,
                            "pattern_index": self.pattern_index,
                            "pattern_embeddings": self.pattern_embeddings,
                        },
                        f,
                    )
            except Exception as e:
                logger.error("Failed to save patterns: %s", e)

            # Save principles
            principles_file = self.storage_path / "principles.pkl"
            try:
                with open(principles_file, "wb") as f:
                    pickle.dump(self.principles, f)
            except Exception as e:
                logger.error("Failed to save principles: %s", e)

            # Save performance data
            performance_file = self.storage_path / "performance.json"
            try:
                with open(performance_file, "w") as f:
                    perf_data = {
                        sig: {
                            "total_uses": p.total_uses,
                            "successful_uses": p.successful_uses,
                            "failed_uses": p.failed_uses,
                            "avg_execution_time": p.avg_execution_time,
                        }
                        for sig, p in self.performance.items()
                    }
                    json.dump(perf_data, f)
            except Exception as e:
                logger.error("Failed to save performance data: %s", e)

    def _load_library(self):
        """Load library from disk"""
        if not self.storage_path or not self.storage_path.exists():
            return

        with self._lock:
            # Load patterns
            patterns_file = self.storage_path / "patterns.pkl"
            if patterns_file.exists():
                try:
                    # Check file size before loading
                    file_size = patterns_file.stat().st_size
                    if file_size > 100_000_000:  # 100MB limit
                        logger.error("Patterns file too large: %d bytes", file_size)
                        return

                    with open(patterns_file, "rb") as f:
                        data = pickle.load(f)

                        # Validate loaded data
                        if not isinstance(data, dict):
                            logger.error("Invalid patterns data type: %s", type(data))
                            return

                        self.patterns = data.get("patterns", {})
                        if not isinstance(self.patterns, dict):
                            logger.error("Invalid patterns structure")
                            self.patterns = {}
                            return

                        self.pattern_index = data.get("pattern_index", {})
                        self.pattern_embeddings = data.get("pattern_embeddings", {})
                except Exception as e:
                    logger.warning("Failed to load patterns: %s", e)

            # Load principles
            principles_file = self.storage_path / "principles.pkl"
            if principles_file.exists():
                try:
                    # Check file size before loading
                    file_size = principles_file.stat().st_size
                    if file_size > 100_000_000:  # 100MB limit
                        logger.error("Principles file too large: %d bytes", file_size)
                        return

                    with open(principles_file, "rb") as f:
                        data = pickle.load(f)

                        # Validate loaded data
                        if not isinstance(data, dict):
                            logger.error("Invalid principles data type: %s", type(data))
                            return

                        self.principles = data
                except Exception as e:
                    logger.warning("Failed to load principles: %s", e)


class StratifiedDecompositionLibrary(DecompositionLibrary):
    """Library with stratified domain organization"""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize stratified library"""
        super().__init__(storage_path)

        # Frequency tracking
        self.pattern_frequency = defaultdict(int)  # pattern_id -> usage count

        # Cross-domain tracking
        self.cross_domain_patterns = set()  # pattern_ids used across multiple domains

        # Domain categorization
        self.domain_categories = self._categorize_domains()

        logger.info("StratifiedDecompositionLibrary initialized")

    def get_patterns_by_frequency(self, min_count: int) -> List[Tuple[str, Pattern]]:
        """
        Get patterns by usage frequency

        Args:
            min_count: Minimum usage count

        Returns:
            List of (pattern_id, pattern) tuples
        """
        with self._lock:
            frequent_patterns = []

            for pattern_id, count in self.pattern_frequency.items():
                if count >= min_count:
                    pattern = self.patterns.get(pattern_id)
                    if pattern:
                        frequent_patterns.append((pattern_id, pattern))

            # Sort by frequency
            frequent_patterns.sort(
                key=lambda x: self.pattern_frequency[x[0]], reverse=True
            )

            return frequent_patterns

    def get_patterns_by_domain(self, domain: str) -> List[Pattern]:
        """
        Get patterns for specific domain

        Args:
            domain: Domain name

        Returns:
            List of patterns
        """
        with self._lock:
            patterns = []

            # Get pattern IDs for domain
            pattern_ids = self.domain_patterns.get(domain, set())

            for pattern_id in pattern_ids:
                pattern = self.patterns.get(pattern_id)
                if pattern:
                    patterns.append(pattern)

            # Sort by performance in this domain
            patterns.sort(
                key=lambda p: self.performance.get(
                    self._get_pattern_signature(p),
                    PatternPerformance(
                        pattern_signature=self._get_pattern_signature(p)
                    ),
                ).get_success_rate(),
                reverse=True,
            )

            return patterns

    def _get_pattern_signature(self, pattern) -> str:
        """Get signature from pattern object or dict"""
        if isinstance(pattern, dict):
            return pattern.get(
                "signature", f"sig_{pattern.get('pattern_id', 'unknown')}"
            )
        elif hasattr(pattern, "get_signature") and callable(pattern.get_signature):
            return pattern.get_signature()
        elif hasattr(pattern, "pattern_id"):
            return f"sig_{pattern.pattern_id}"
        else:
            return f"sig_{id(pattern)}"

    def get_cross_domain_patterns(self, min_domains: int = 3) -> List[Pattern]:
        """
        Get patterns used across multiple domains

        Args:
            min_domains: Minimum number of domains

        Returns:
            List of cross-domain patterns
        """
        with self._lock:
            cross_domain = []

            for pattern_id, domains in self.pattern_domains.items():
                if len(domains) >= min_domains:
                    pattern = self.patterns.get(pattern_id)
                    if pattern:
                        cross_domain.append(pattern)
                        self.cross_domain_patterns.add(pattern_id)

            # Sort by number of domains
            cross_domain.sort(
                key=lambda p: len(self.pattern_domains[self._get_pattern_id(p)]),
                reverse=True,
            )

            return cross_domain

    def _get_pattern_id(self, pattern) -> str:
        """Get pattern_id from pattern object or dict"""
        if isinstance(pattern, dict):
            return pattern.get("pattern_id", f"pattern_{id(pattern)}")
        elif hasattr(pattern, "pattern_id"):
            return pattern.pattern_id
        else:
            return f"pattern_{id(pattern)}"

    def get_strategy_by_type(self, strategy_type: str):
        """
        Get decomposition strategy by type

        Args:
            strategy_type: Type of strategy

        Returns:
            Strategy object or None
        """
        # This would return actual strategy objects in full implementation
        # For now, return a mock strategy
        try:
            from .decomposition_strategies import (
                DecompositionStrategy,
                ExactDecomposition,
                SemanticDecomposition,
                StructuralDecomposition,
            )
        except ImportError:
            try:
                from decomposition_strategies import (
                    DecompositionStrategy,
                    ExactDecomposition,
                    SemanticDecomposition,
                    StructuralDecomposition,
                )
            except ImportError:
                logger.warning("Could not import decomposition strategies")
                return None

        strategy_map = {
            "exact": ExactDecomposition,
            "semantic": SemanticDecomposition,
            "structural": StructuralDecomposition,
            "simple": StructuralDecomposition,
            "hierarchical": StructuralDecomposition,
            "hybrid": StructuralDecomposition,
        }

        strategy_class = strategy_map.get(strategy_type)
        if strategy_class:
            return strategy_class()

        return None

    def get_strategy(self, strategy_name: str):
        """
        Get strategy by name

        Args:
            strategy_name: Name of strategy

        Returns:
            Strategy object or None
        """
        # Map strategy names to types
        name_to_type = {
            "ExactDecomposition": "exact",
            "SemanticDecomposition": "semantic",
            "StructuralDecomposition": "structural",
        }

        strategy_type = name_to_type.get(strategy_name, strategy_name.lower())
        return self.get_strategy_by_type(strategy_type)

    def update_usage_statistics(self, pattern_id: str, domain: str):
        """Update usage statistics for pattern"""
        with self._lock:
            # Update frequency
            self.pattern_frequency[pattern_id] += 1

            # Update domain tracking
            self.pattern_domains[pattern_id].add(domain)
            self.domain_patterns[domain].add(pattern_id)

            # Check if cross-domain
            if len(self.pattern_domains[pattern_id]) >= 3:
                self.cross_domain_patterns.add(pattern_id)

    def get_domain_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per domain"""
        with self._lock:
            stats = {}

            for domain in self.domain_patterns:
                pattern_ids = self.domain_patterns[domain]

                # Calculate domain statistics
                success_rates = []
                for pattern_id in pattern_ids:
                    pattern = self.patterns.get(pattern_id)
                    if pattern:
                        signature = self._get_pattern_signature(pattern)

                        # FIXED: Create performance object if doesn't exist
                        if signature not in self.performance:
                            self.performance[signature] = PatternPerformance(
                                pattern_signature=signature
                            )

                        perf = self.performance[signature]
                        if perf.total_uses > 0:
                            success_rates.append(perf.get_success_rate())

                stats[domain] = {
                    "pattern_count": len(pattern_ids),
                    "avg_success_rate": np.mean(success_rates)
                    if success_rates
                    else 0.5,
                    "category": self._categorize_domain(domain, len(pattern_ids)),
                }

            return stats

    def _categorize_domains(self) -> Dict[DomainCategory, Set[str]]:
        """Categorize domains by usage"""
        with self._lock:
            categories = defaultdict(set)

            for domain, pattern_ids in self.domain_patterns.items():
                category = self._categorize_domain(domain, len(pattern_ids))
                categories[category].add(domain)

            return dict(categories)

    def _categorize_domain(self, domain: str, pattern_count: int) -> DomainCategory:
        """Categorize single domain"""
        if pattern_count >= 100:
            return DomainCategory.FREQUENT
        elif pattern_count >= 20:
            return DomainCategory.COMMON
        elif pattern_count >= 5:
            return DomainCategory.RARE
        else:
            return DomainCategory.NOVEL
