"""Memory consolidation and optimization"""

import copy
import hashlib
import logging
import pickle
import threading
import time
from collections import Counter, defaultdict
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .base import Memory, MemoryType

# Try to import optional dependencies
try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using fallback clustering")

try:
    pass

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available, graph analysis limited")

logger = logging.getLogger(__name__)

# ============================================================
# CONSOLIDATION STRATEGIES
# ============================================================


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies."""

    IMPORTANCE_BASED = "importance"
    FREQUENCY_BASED = "frequency"
    RECENCY_BASED = "recency"
    SEMANTIC_CLUSTERING = "semantic"
    CAUSAL_CHAINS = "causal"
    INFORMATION_THEORETIC = "information"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    GRAPH_BASED = "graph"


# ============================================================
# CLUSTERING ALGORITHMS
# ============================================================


class ClusteringAlgorithm:
    """Base class for clustering algorithms."""

    @staticmethod
    def cluster(embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Cluster embeddings."""
        raise NotImplementedError


class KMeansClustering(ClusteringAlgorithm):
    """K-means clustering implementation."""

    @staticmethod
    def cluster(embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Cluster using K-means."""
        if SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings).tolist()
        else:
            # Simple K-means implementation
            return KMeansClustering._simple_kmeans(embeddings, n_clusters)

    @staticmethod
    def _simple_kmeans(
        embeddings: np.ndarray, n_clusters: int, max_iter: int = 100
    ) -> List[int]:
        """Simple K-means implementation without sklearn."""
        n_samples = len(embeddings)

        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = embeddings[indices].copy()

        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            old_labels = labels.copy()

            # Assign points to nearest centroid
            for i, point in enumerate(embeddings):
                distances = np.linalg.norm(centroids - point, axis=1)
                labels[i] = np.argmin(distances)

            # Update centroids
            for k in range(n_clusters):
                cluster_points = embeddings[labels == k]
                if len(cluster_points) > 0:
                    centroids[k] = np.mean(cluster_points, axis=0)

            # Check convergence
            if np.array_equal(labels, old_labels):
                break

        return labels.tolist()


class DBSCANClustering(ClusteringAlgorithm):
    """DBSCAN clustering for automatic cluster discovery."""

    @staticmethod
    def cluster(
        embeddings: np.ndarray, eps: float = 0.3, min_samples: int = 5
    ) -> List[int]:
        """Cluster using DBSCAN."""
        if SKLEARN_AVAILABLE:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            return dbscan.fit_predict(embeddings).tolist()
        else:
            # Simple DBSCAN implementation
            return DBSCANClustering._simple_dbscan(embeddings, eps, min_samples)

    @staticmethod
    def _simple_dbscan(
        embeddings: np.ndarray, eps: float, min_samples: int
    ) -> List[int]:
        """Simple DBSCAN implementation."""
        n_samples = len(embeddings)
        labels = np.full(n_samples, -1)  # -1 for noise
        cluster_id = 0
        visited = set()

        def get_neighbors(idx):
            distances = np.linalg.norm(embeddings - embeddings[idx], axis=1)
            return np.where(distances <= eps)[0]

        for i in range(n_samples):
            if i in visited:
                continue

            visited.add(i)
            neighbors = get_neighbors(i)

            if len(neighbors) < min_samples:
                continue  # Noise point

            # Start new cluster
            labels[i] = cluster_id
            seed_set = set(neighbors)
            seed_set.discard(i)

            while seed_set:
                j = seed_set.pop()
                if j not in visited:
                    visited.add(j)
                    j_neighbors = get_neighbors(j)
                    if len(j_neighbors) >= min_samples:
                        seed_set.update(j_neighbors)

                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        return labels.tolist()


class HierarchicalClustering(ClusteringAlgorithm):
    """Hierarchical clustering for dendrogram-based grouping."""

    @staticmethod
    def cluster(embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Cluster using hierarchical clustering."""
        if SKLEARN_AVAILABLE:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            return clustering.fit_predict(embeddings).tolist()
        else:
            # Simple hierarchical clustering
            return HierarchicalClustering._simple_hierarchical(embeddings, n_clusters)

    @staticmethod
    def _simple_hierarchical(embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Simple hierarchical clustering implementation."""
        n_samples = len(embeddings)

        # Start with each point as its own cluster
        clusters = [[i] for i in range(n_samples)]

        # Compute pairwise distances
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances[i, j] = distances[j, i] = dist

        # Merge clusters until we have n_clusters
        while len(clusters) > n_clusters:
            # Find closest pair of clusters
            min_dist = float("inf")
            merge_i, merge_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage
                    cluster_dist = 0
                    count = 0
                    for idx_i in clusters[i]:
                        for idx_j in clusters[j]:
                            cluster_dist += distances[idx_i, idx_j]
                            count += 1

                    if count > 0:
                        cluster_dist /= count
                        if cluster_dist < min_dist:
                            min_dist = cluster_dist
                            merge_i, merge_j = i, j

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]

        # Assign labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_id

        return labels.tolist()


# ============================================================
# ENHANCED MEMORY CONSOLIDATOR
# ============================================================


class MemoryConsolidator:
    """Advanced memory consolidation with multiple strategies."""

    def __init__(self):
        self.strategies = {
            ConsolidationStrategy.IMPORTANCE_BASED: self._consolidate_by_importance,
            ConsolidationStrategy.FREQUENCY_BASED: self._consolidate_by_frequency,
            ConsolidationStrategy.RECENCY_BASED: self._consolidate_by_recency,
            ConsolidationStrategy.SEMANTIC_CLUSTERING: self._consolidate_by_semantics,
            ConsolidationStrategy.CAUSAL_CHAINS: self._consolidate_by_causality,
            ConsolidationStrategy.INFORMATION_THEORETIC: self._consolidate_by_information,
            ConsolidationStrategy.ADAPTIVE: self._consolidate_adaptive,
            ConsolidationStrategy.HIERARCHICAL: self._consolidate_hierarchical,
            ConsolidationStrategy.GRAPH_BASED: self._consolidate_graph_based,
        }

        self.consolidation_history = []
        self.performance_metrics = defaultdict(list)

        # Clustering algorithms
        self.clustering_algorithms = {
            "kmeans": KMeansClustering(),
            "dbscan": DBSCANClustering(),
            "hierarchical": HierarchicalClustering(),
        }

        # Cache for expensive computations
        self.cache = {}
        self.cache_lock = threading.Lock()

    def consolidate(
        self,
        memories: List[Memory],
        strategy: ConsolidationStrategy = ConsolidationStrategy.ADAPTIVE,
        target_count: Optional[int] = None,
        **kwargs,
    ) -> List[Memory]:
        """Consolidate memories using specified strategy."""

        if not memories:
            return []

        if strategy not in self.strategies:
            strategy = ConsolidationStrategy.ADAPTIVE

        start_time = time.time()

        # Use adaptive strategy if specified
        if strategy == ConsolidationStrategy.ADAPTIVE:
            strategy = self._select_best_strategy(memories, target_count)

        consolidation_func = self.strategies[strategy]
        consolidated = consolidation_func(memories, target_count, **kwargs)

        # Calculate metrics
        elapsed_time = time.time() - start_time
        compression_ratio = len(memories) / max(1, len(consolidated))

        # Record consolidation
        self.consolidation_history.append(
            {
                "timestamp": time.time(),
                "strategy": strategy.value,
                "input_count": len(memories),
                "output_count": len(consolidated),
                "compression_ratio": compression_ratio,
                "elapsed_time": elapsed_time,
            }
        )

        # Update performance metrics
        self.performance_metrics[strategy].append(
            {
                "compression_ratio": compression_ratio,
                "time": elapsed_time,
                "quality": self._evaluate_consolidation_quality(memories, consolidated),
            }
        )

        return consolidated

    def _select_best_strategy(
        self, memories: List[Memory], target_count: Optional[int]
    ) -> ConsolidationStrategy:
        """Select best consolidation strategy based on memory characteristics."""

        # Analyze memory characteristics
        has_embeddings = sum(1 for m in memories if m.embedding is not None) / len(
            memories
        )
        avg_access_count = np.mean([m.access_count for m in memories])
        time_span = max(m.timestamp for m in memories) - min(
            m.timestamp for m in memories
        )
        importance_variance = np.var([m.importance for m in memories])

        # Rule-based strategy selection
        if has_embeddings > 0.8:
            return ConsolidationStrategy.SEMANTIC_CLUSTERING
        elif avg_access_count > 10:
            return ConsolidationStrategy.FREQUENCY_BASED
        elif time_span < 3600:  # Less than 1 hour
            return ConsolidationStrategy.RECENCY_BASED
        elif importance_variance > 0.2:
            return ConsolidationStrategy.IMPORTANCE_BASED
        else:
            return ConsolidationStrategy.INFORMATION_THEORETIC

    def _consolidate_by_importance(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Keep most important memories with decay consideration."""
        current_time = time.time()

        # Calculate effective importance with time decay
        scored_memories = []
        for memory in memories:
            # Combine importance with salience
            effective_importance = memory.compute_salience(current_time)
            scored_memories.append((memory, effective_importance))

        # Sort by effective importance
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        if target_count:
            return [m for m, _ in scored_memories[:target_count]]

        # Dynamic threshold based on distribution
        scores = [s for _, s in scored_memories]
        threshold = np.percentile(scores, 50)

        # Add minimum count constraint
        min_count = max(5, len(memories) // 10)
        result = [m for m, s in scored_memories if s >= threshold]

        if len(result) < min_count:
            result = [m for m, _ in scored_memories[:min_count]]

        return result

    def _consolidate_by_frequency(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Keep frequently accessed memories with recency weighting."""
        current_time = time.time()

        # Calculate weighted access score
        scored_memories = []
        for memory in memories:
            # Weight recent accesses more
            recency_factor = np.exp(
                -(current_time - memory.timestamp) / (7 * 24 * 3600)
            )
            weighted_access = memory.access_count * (0.5 + 0.5 * recency_factor)
            scored_memories.append((memory, weighted_access))

        # Sort by weighted access
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        if target_count:
            return [m for m, _ in scored_memories[:target_count]]

        # Keep memories with above-average weighted access
        scores = [s for _, s in scored_memories]
        threshold = np.mean(scores)

        return [m for m, s in scored_memories if s >= threshold]

    def _consolidate_by_recency(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Keep recent memories with importance weighting."""
        current_time = time.time()
        window = kwargs.get("time_window", 24 * 3600)  # Default 24 hours

        # Score by recency and importance
        scored_memories = []
        for memory in memories:
            age = current_time - memory.timestamp
            if age <= window:
                # Exponential decay within window
                recency_score = np.exp(-age / window)
                combined_score = recency_score * (0.7 + 0.3 * memory.importance)
                scored_memories.append((memory, combined_score))

        # Sort by combined score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        if target_count:
            return [m for m, _ in scored_memories[:target_count]]

        return [m for m, _ in scored_memories]

    def _consolidate_by_semantics(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Consolidate by semantic clustering with multiple algorithms."""
        if not memories:
            return []

        # Extract embeddings
        embeddings = []
        valid_memories = []

        for memory in memories:
            if memory.embedding is not None:
                embeddings.append(memory.embedding)
                valid_memories.append(memory)

        if not embeddings:
            # Fallback to importance-based
            return self._consolidate_by_importance(memories, target_count)

        embeddings_array = np.array(embeddings)

        # Determine number of clusters
        if target_count:
            n_clusters = min(target_count, len(valid_memories))
        else:
            # Automatic cluster selection using elbow method
            n_clusters = self._find_optimal_clusters(embeddings_array)

        # Choose clustering algorithm
        algorithm = kwargs.get("clustering", "kmeans")

        if algorithm == "dbscan":
            clusters = DBSCANClustering.cluster(embeddings_array)
        elif algorithm == "hierarchical":
            clusters = HierarchicalClustering.cluster(embeddings_array, n_clusters)
        else:
            clusters = KMeansClustering.cluster(embeddings_array, n_clusters)

        # Select representative from each cluster
        consolidated = []
        cluster_groups = defaultdict(list)

        for i, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(valid_memories[i])

        for cluster_id, cluster_memories in cluster_groups.items():
            if cluster_id == -1:  # DBSCAN noise points
                # Keep all noise points if important enough
                for memory in cluster_memories:
                    if memory.importance > 0.7:
                        consolidated.append(memory)
            else:
                # Select best representative
                representative = self._select_cluster_representative(
                    cluster_memories,
                    embeddings_array[
                        [i for i, c in enumerate(clusters) if c == cluster_id]
                    ],
                )
                consolidated.append(representative)

        return consolidated

    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        max_clusters = min(10, len(embeddings))

        if SKLEARN_AVAILABLE:
            # Use silhouette score
            scores = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                scores.append(score)

            # Find elbow point
            if scores:
                # Convert numpy integer to Python int
                return int(np.argmax(scores) + 2)

        # Default to sqrt(n) rule - ensure Python int
        return int(np.sqrt(len(embeddings)))

    def _select_cluster_representative(
        self, cluster_memories: List[Memory], cluster_embeddings: np.ndarray
    ) -> Memory:
        """Select best representative from cluster."""
        if len(cluster_memories) == 1:
            return cluster_memories[0]

        # Find medoid (most central point)
        if len(cluster_embeddings) > 0:
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            np.argmin(distances)

            # Weight by importance and centrality
            scores = []
            for i, memory in enumerate(cluster_memories):
                centrality = 1.0 / (1.0 + distances[i])
                score = memory.importance * 0.6 + centrality * 0.4
                scores.append(score)

            best_idx = np.argmax(scores)
            representative = cluster_memories[best_idx]
        else:
            # Fallback to most important
            representative = max(cluster_memories, key=lambda m: m.importance)

        # Merge information from cluster
        merged = self._merge_memories(cluster_memories)

        # Use representative ID but merged content
        representative.content = merged.content
        representative.metadata.update(merged.metadata)

        return representative

    def _consolidate_by_causality(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Keep memories that form causal chains."""
        if NETWORKX_AVAILABLE:
            return self._consolidate_causality_graph(memories, target_count)
        else:
            return self._consolidate_causality_simple(memories, target_count)

    def _consolidate_causality_graph(
        self, memories: List[Memory], target_count: Optional[int]
    ) -> List[Memory]:
        """Use graph algorithms for causal chain detection."""
        import networkx as nx

        # Build directed graph
        G = nx.DiGraph()

        # Add nodes
        for memory in memories:
            G.add_node(memory.id, memory=memory)

        # Add edges based on causal relationships
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)

        for i, memory in enumerate(sorted_memories):
            # Look at subsequent memories
            for j in range(i + 1, min(i + 10, len(sorted_memories)):
                next_memory = sorted_memories[j]

                # Check temporal and semantic relationship
                time_delta = next_memory.timestamp - memory.timestamp
                if 0 < time_delta < 3600:  # Within 1 hour
                    # Calculate causal score
                    causal_score = self._calculate_causal_score(memory, next_memory)
                    if causal_score > 0.5:
                        G.add_edge(memory.id, next_memory.id, weight=causal_score)

        # Find important paths
        important_nodes = set()

        # Use PageRank to find important nodes
        if len(G.nodes) > 0:
            pagerank = nx.pagerank(G)
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

            # Add nodes with high PageRank
            for node_id, score in sorted_nodes:
                if target_count and len(important_nodes) >= target_count:
                    break
                important_nodes.add(node_id)

        # Extract memories
        result = []
        for node_id in important_nodes:
            if node_id in G.nodes:
                result.append(G.nodes[node_id]["memory"])

        return result

    def _consolidate_causality_simple(
        self, memories: List[Memory], target_count: Optional[int]
    ) -> List[Memory]:
        """Simple causal chain detection without NetworkX."""
        # Build causal graph
        causal_graph = self._build_causal_graph(memories)

        # Find important chains
        chains = self._find_causal_chains(causal_graph)

        # Score chains
        scored_chains = []
        for chain in chains:
            score = sum(m.importance for m in chain) * len(chain)
            scored_chains.append((chain, score))

        scored_chains.sort(key=lambda x: x[1], reverse=True)

        # Select memories from top chains
        consolidated = list(]
        seen = set()

        for chain, _ in scored_chains:
            for memory in chain:
                if memory.id not in seen:
                    consolidated.append(memory)
                    seen.add(memory.id)

                    if target_count and len(consolidated) >= target_count:
                        return consolidated

        return consolidated

    def _calculate_causal_score(self, cause: Memory, effect: Memory) -> float:
        """Calculate causal relationship score between memories."""
        score = 0.0

        # Temporal factor
        time_delta = effect.timestamp - cause.timestamp
        if time_delta > 0:
            temporal_score = np.exp(-time_delta / 3600)  # Decay over 1 hour
            score += temporal_score * 0.3

        # Semantic similarity
        if cause.embedding is not None and effect.embedding is not None:
            similarity = np.dot(cause.embedding, effect.embedding) / (
                np.linalg.norm(cause.embedding) * np.linalg.norm(effect.embedding)
                + 1e-10
            )
            score += max(0, similarity) * 0.5

        # Metadata overlap
        shared_keys = set(cause.metadata.keys()) & set(effect.metadata.keys())
        if shared_keys:
            overlap_ratio = len(shared_keys) / max(
                len(cause.metadata), len(effect.metadata)
            )
            score += overlap_ratio * 0.2

        return min(1.0, score)

    def _consolidate_by_information(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Keep memories with highest information content."""
        # Calculate information scores
        scored_memories = []

        for memory in memories:
            score = self._calculate_information_content(memory)
            scored_memories.append((memory, score))

        # Sort by information content
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        if target_count:
            return [m for m, _ in scored_memories[:target_count]]

        # Use entropy-based threshold
        scores = [s for _, s in scored_memories]
        threshold = np.percentile(scores, 40)

        return [m for m, s in scored_memories if s >= threshold]

    def _consolidate_adaptive(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Adaptive consolidation using multiple strategies."""
        # This is handled by _select_best_strategy
        strategy = self._select_best_strategy(memories, target_count)
        return self.strategies[strategy](memories, target_count, **kwargs)

    def _consolidate_hierarchical(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Hierarchical consolidation with multiple levels."""
        if not memories:
            return []

        # First level: Remove duplicates
        unique_memories = self._remove_near_duplicates(memories)

        if target_count and len(unique_memories) <= target_count:
            return unique_memories

        # Second level: Cluster by type
        by_type = defaultdict(list)
        for memory in unique_memories:
            by_type[memory.type].append(memory)

        # Third level: Consolidate each type
        consolidated = []

        for mem_type, type_memories in by_type.items():
            # Allocate quota proportionally
            if target_count:
                type_quota = max(
                    1, int(target_count * len(type_memories) / len(unique_memories))
                )
            else:
                type_quota = None

            # Use appropriate strategy for type
            if mem_type == MemoryType.EPISODIC:
                strategy = ConsolidationStrategy.RECENCY_BASED
            elif mem_type == MemoryType.SEMANTIC:
                strategy = ConsolidationStrategy.SEMANTIC_CLUSTERING
            else:
                strategy = ConsolidationStrategy.IMPORTANCE_BASED

            type_consolidated = self.strategies[strategy](
                type_memories, type_quota, **kwargs
            )
            consolidated.extend(type_consolidated)

        return consolidated

    def _consolidate_graph_based(
        self, memories: List[Memory], target_count: Optional[int], **kwargs
    ) -> List[Memory]:
        """Graph-based consolidation using memory relationships."""
        # Check flag and wrap in try-except
        if not NETWORKX_AVAILABLE:
            logger.info(
                "NetworkX not available, falling back to importance-based consolidation"
            )
            return self._consolidate_by_importance(memories, target_count, **kwargs)

        try:
            import networkx as nx  # Import inside try block

            # Build memory graph
            G = nx.Graph()

            for memory in memories:
                G.add_node(memory.id, memory=memory)

            # Add edges based on similarity
            for i, mem1 in enumerate(memories):
                for mem2 in memories[i + 1 :]:
                    similarity = self._calculate_similarity(mem1, mem2)
                    if similarity > 0.5:
                        G.add_edge(mem1.id, mem2.id, weight=similarity)

            # Find communities
            if len(G.edges) > 0:
                communities = nx.community.greedy_modularity_communities(G)
            else:
                communities = [{node} for node in G.nodes]

            # Select representative from each community
            consolidated = []

            for community in communities:
                community_memories = [G.nodes[node]["memory"] for node in community]

                if community_memories:
                    # Find most central node
                    if len(community) > 1:
                        subgraph = G.subgraph(community)
                        centrality = nx.degree_centrality(subgraph)
                        central_node = max(centrality, key=centrality.get)
                        representative = G.nodes[central_node]["memory"]
                    else:
                        representative = community_memories[0]

                    consolidated.append(representative)

            # Limit to target count
            if target_count and len(consolidated) > target_count:
                consolidated.sort(key=lambda m: m.importance, reverse=True)
                consolidated = consolidated[:target_count]

            return consolidated

        except ImportError as e:
            logger.warning(f"NetworkX import failed despite availability flag: {e}")
            return self._consolidate_by_importance(memories, target_count, **kwargs)
        except Exception as e:
            logger.error(f"Graph-based consolidation error: {e}")
            return self._consolidate_by_importance(memories, target_count, **kwargs)

    def _remove_near_duplicates(
        self, memories: List[Memory], threshold: float = 0.95
    ) -> List[Memory]:
        """Remove near-duplicate memories."""
        unique = list(]
        seen_hashes = set()
        seen_embeddings = list(]

        for memory in memories:
            # Check content hash
            content_hash = hashlib.sha256(
                pickle.dumps(memory.content, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()

            if content_hash in seen_hashes:
                continue

            # Check embedding similarity
            is_duplicate = False
            if memory.embedding is not None:
                for seen_emb in seen_embeddings:
                    similarity = np.dot(memory.embedding, seen_emb) / (
                        np.linalg.norm(memory.embedding) * np.linalg.norm(seen_emb)
                        + 1e-10
                    )
                    if similarity > threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(memory)
                seen_hashes.add(content_hash)
                if memory.embedding is not None:
                    seen_embeddings.append(memory.embedding)

        return unique

    def _calculate_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate overall similarity between memories."""
        similarity = 0.0
        weights_sum = 0.0

        # Embedding similarity
        if mem1.embedding is not None and mem2.embedding is not None:
            emb_sim = np.dot(mem1.embedding, mem2.embedding) / (
                np.linalg.norm(mem1.embedding) * np.linalg.norm(mem2.embedding) + 1e-10
            )
            similarity += emb_sim * 0.5
            weights_sum += 0.5

        # Type similarity
        if mem1.type == mem2.type:
            similarity += 0.2
        weights_sum += 0.2

        # Temporal proximity
        time_diff = abs(mem1.timestamp - mem2.timestamp)
        temporal_sim = np.exp(-time_diff / (24 * 3600))  # Decay over 1 day
        similarity += temporal_sim * 0.2
        weights_sum += 0.2

        # Metadata overlap
        if mem1.metadata and mem2.metadata:
            shared = set(mem1.metadata.keys()) & set(mem2.metadata.keys())
            if shared:
                overlap = len(shared) / max(len(mem1.metadata), len(mem2.metadata))
                similarity += overlap * 0.1
            weights_sum += 0.1

        return similarity / weights_sum if weights_sum > 0 else 0.0

    def _cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Cluster embeddings using best available method."""
        # Ensure valid number of clusters
        n_clusters = min(n_clusters, len(embeddings))
        n_clusters = max(1, n_clusters)

        if n_clusters == 1:
            return [0] * len(embeddings)

        # Try different algorithms
        if SKLEARN_AVAILABLE:
            # Use sklearn with optimization
            best_score = -1
            best_labels = None

            for algorithm in ["kmeans", "hierarchical"]:
                if algorithm == "kmeans":
                    clusterer = KMeans(
                        n_clusters=n_clusters, random_state=42, n_init=10
                    )
                else:
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)

                labels = clusterer.fit_predict(embeddings)

                # Evaluate clustering quality
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels

            if best_labels is not None:
                return best_labels.tolist()

        # Fallback to simple k-means
        return KMeansClustering.cluster(embeddings, n_clusters)

    def _merge_memories(self, memories: List[Memory]) -> Memory:
        """Merge multiple memories into one consolidated memory."""
        if len(memories) == 1:
            return copy.deepcopy(memories[0])

        # Create merged memory
        merged = Memory(
            id=f"merged_{hashlib.sha256(''.join(m.id for m in memories).encode()).hexdigest()[:16]}",
            type=Counter([m.type for m in memories]).most_common(1)[0][0],
            content={
                "merged_from": [m.id for m in memories],
                "contents": [m.content for m in memories],
                "summary": self._generate_summary(memories),
            },
            timestamp=max(m.timestamp for m in memories),
            importance=max(m.importance for m in memories),
            access_count=sum(m.access_count for m in memories),
        )

        # Merge embeddings
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        if embeddings:
            # Weighted average based on importance
            weights = np.array(
                [m.importance for m in memories if m.embedding is not None]
            )
            weights = weights / weights.sum()
            merged.embedding = np.average(embeddings, axis=0, weights=weights)

        # Merge metadata
        merged.metadata = {}
        for memory in memories:
            for key, value in memory.metadata.items():
                if key not in merged.metadata:
                    merged.metadata[key] = []
                merged.metadata[key].append(value)

        # Flatten single-item lists in metadata
        for key, value in merged.metadata.items():
            if isinstance(value, list) and len(value) == 1:
                merged.metadata[key] = value[0]

        return merged

    def _generate_summary(self, memories: List[Memory]) -> str:
        """Generate summary of multiple memories."""
        # Simple extractive summary
        contents = []
        for memory in memories[:5]:  # Limit to prevent too long summary
            content_str = str(memory.content)[:100]
            contents.append(content_str)

        return " | ".join(contents)

    def _build_causal_graph(self, memories: List[Memory]) -> Dict[str, List[Memory]]:
        """Build causal relationships between memories."""
        graph = {}

        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)

        # Build edges based on temporal proximity and similarity
        for i, memory in enumerate(sorted_memories):
            effects = []

            # Look at subsequent memories
            for j in range(i + 1, min(i + 10, len(sorted_memories)):
                next_memory = sorted_memories[j]

                # Calculate causal score
                causal_score = self._calculate_causal_score(memory, next_memory)
                if causal_score > 0.5:
                    effects.append(next_memory)

            if effects:
                graph[memory.id] = effects

        return graph

    def _find_causal_chains(self, graph: Dict[str, List[Memory]]) -> List[List[Memory]]:
        """Find causal chains in memory graph."""
        chains = []
        visited = set()

        def dfs(memory_id, chain, memory_map):
            if memory_id in visited:
                return

            visited.add(memory_id)

            if memory_id in graph:
                for next_memory in graph[memory_id]:
                    dfs(next_memory.id, chain + [next_memory], memory_map)
            else:
                if len(chain) > 1:
                    chains.append(chain)

        # Create memory map
        memory_map = {}
        for memory_id, effects in graph.items():
            for memory in effects:
                memory_map[memory.id] = memory

        # Start DFS from each root node
        for memory_id in graph.keys():
            if memory_id not in visited:
                dfs(memory_id, [], memory_map)

        # Sort chains by total importance
        chains.sort(key=lambda c: sum(m.importance for m in c), reverse=True)

        return chains

    def _calculate_information_content(self, memory: Memory) -> float:
        """Calculate information content of memory."""
        # Multiple factors for information content

        # Content complexity (entropy)
        content_str = str(memory.content)
        entropy = self._calculate_entropy(content_str)

        # Uniqueness (inverse of similarity to other memories)
        uniqueness = 1.0  # Would need comparison with other memories

        # Temporal relevance
        age = time.time() - memory.timestamp
        temporal_relevance = np.exp(-age / (30 * 24 * 3600))  # 30-day decay

        # Access pattern entropy
        access_entropy = np.log(1 + memory.access_count) / 10

        # Combine factors
        score = (
            entropy * 0.3
            + memory.importance * 0.3
            + uniqueness * 0.2
            + temporal_relevance * 0.1
            + access_entropy * 0.1
        )

        return float(np.clip(score, 0, 1))

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        # Character frequency
        char_freq = Counter(text)
        total = len(text)

        entropy = 0.0
        for count in char_freq.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)

        # Normalize to [0, 1]
        max_entropy = np.log2(len(char_freq))
        if max_entropy > 0:
            entropy = entropy / max_entropy

        return float(entropy)

    def _evaluate_consolidation_quality(
        self, original: List[Memory], consolidated: List[Memory]
    ) -> float:
        """Evaluate quality of consolidation."""
        if not original or not consolidated:
            return 0.0

        # Information preservation
        orig_info = sum(self._calculate_information_content(m) for m in original)
        cons_info = sum(self._calculate_information_content(m) for m in consolidated)
        info_preservation = min(1.0, cons_info / orig_info) if orig_info > 0 else 0

        # Diversity preservation
        if len(consolidated) > 1:
            diversity = 1.0 - np.mean(
                [
                    self._calculate_similarity(consolidated[i], consolidated[j])
                    for i in range(len(consolidated)):
                    for j in range(i + 1, len(consolidated)):
                ]
            )
        else:
            diversity = 0.5

        # Coverage (how well consolidated memories represent original)
        coverage = len(consolidated) / len(original)

        quality = info_preservation * 0.5 + diversity * 0.3 + coverage * 0.2

        return float(np.clip(quality, 0, 1))


# ============================================================
# MEMORY OPTIMIZER
# ============================================================


class MemoryOptimizer:
    """Optimizes memory storage and retrieval."""

    def __init__(self):
        self.optimization_history = []
        self.index_manager = IndexManager()
        self.cache_manager = CacheManager()

    def optimize(self, memories: Dict[str, Memory]) -> Dict[str, Memory]:
        """Optimize memory collection."""
        start_time = time.time()

        optimized = memories.copy()

        # Step 1: Remove duplicates
        optimized = self._remove_duplicates(optimized)

        # Step 2: Compress large memories
        optimized = self._compress_large_memories(optimized)

        # Step 3: Update indices
        optimized = self._update_indices(optimized)

        # Step 4: Optimize cache
        self.cache_manager.optimize(optimized)

        # Step 5: Rebalance memory types
        optimized = self._rebalance_memory_types(optimized)

        # Record optimization
        elapsed = time.time() - start_time
        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "duration_ms": elapsed * 1000,
                "input_count": len(memories),
                "output_count": len(optimized),
                "reduction_ratio": 1 - len(optimized) / max(1, len(memories)),
            }
        )

        return optimized

    def _remove_duplicates(self, memories: Dict[str, Memory]) -> Dict[str, Memory]:
        """Remove duplicate memories keeping best version."""
        seen_hashes = {}
        unique = {}

        for memory_id, memory in memories.items():
            # Compute content hash
            content_hash = hashlib.sha256(
                pickle.dumps(memory.content, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()

            if content_hash not in seen_hashes:
                unique[memory_id] = memory
                seen_hashes[content_hash] = memory_id
            else:
                # Compare with existing
                existing_id = seen_hashes[content_hash]
                existing = unique[existing_id]

                # Keep the better one
                if (
                    memory.importance > existing.importance
                    or memory.access_count > existing.access_count
                ):
                    del unique[existing_id]
                    unique[memory_id] = memory
                    seen_hashes[content_hash] = memory_id

        logger.info(f"Removed {len(memories) - len(unique)} duplicate memories")
        return unique

    def _compress_large_memories(
        self, memories: Dict[str, Memory]
    ) -> Dict[str, Memory]:
        """Mark large memories for compression."""
        import sys

        compression_candidates = []

        for memory_id, memory in memories.items():
            if memory.content is not None and not memory.compressed:
                size = sys.getsizeof(pickle.dumps(memory.content))

                if size > 10000:  # 10KB threshold
                    compression_candidates.append((memory_id, size))
                    memory.metadata["needs_compression"] = True
                    memory.metadata["original_size"] = size

        if compression_candidates:
            logger.info(
                f"Marked {len(compression_candidates)} memories for compression"
            )

        return memories

    def _update_indices(self, memories: Dict[str, Memory]) -> Dict[str, Memory]:
        """Update memory indices for faster retrieval."""
        # Update various indices
        self.index_manager.update(memories)

        # Add index metadata
        for memory_id, memory in memories.items():
            memory.metadata["indexed"] = True
            memory.metadata["index_version"] = self.index_manager.version

        return memories

    def _rebalance_memory_types(self, memories: Dict[str, Memory]) -> Dict[str, Memory]:
        """Rebalance distribution of memory types."""
        # Count by type
        type_counts = Counter(m.type for m in memories.values())
        total = len(memories)

        # Define target ratios
        target_ratios = {
            MemoryType.WORKING: 0.05,
            MemoryType.EPISODIC: 0.25,
            MemoryType.SEMANTIC: 0.35,
            MemoryType.PROCEDURAL: 0.15,
            MemoryType.LONG_TERM: 0.20,
        }

        # Check if rebalancing needed
        needs_rebalancing = False
        for mem_type, target_ratio in target_ratios.items():
            current_ratio = type_counts.get(mem_type, 0) / max(1, total)
            if abs(current_ratio - target_ratio) > 0.1:
                needs_rebalancing = True
                break

        if needs_rebalancing:
            logger.info("Rebalancing memory type distribution")
            # Would implement type conversion/migration here

        return memories


class IndexManager:
    """Manages memory indices."""

    def __init__(self):
        self.version = 1
        self.indices = {
            "content_hash": {},
            "type_index": defaultdict(set),
            "timestamp_index": [],
            "importance_index": [],
        }

    def update(self, memories: Dict[str, Memory]):
        """Update all indices."""
        # Clear existing
        for index in self.indices.values():
            if isinstance(index, dict):
                index.clear()
            elif isinstance(index, list):
                index.clear()

        # Rebuild indices
        for memory_id, memory in memories.items():
            # Content hash index
            content_hash = hashlib.sha256(
                pickle.dumps(memory.content, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()[:16]
            self.indices["content_hash"][content_hash] = memory_id

            # Type index
            self.indices["type_index"][memory.type].add(memory_id)

            # Timestamp index (sorted)
            self.indices["timestamp_index"].append((memory.timestamp, memory_id))

            # Importance index (sorted)
            self.indices["importance_index"].append((memory.importance, memory_id))

        # Sort indices
        self.indices["timestamp_index"].sort()
        self.indices["importance_index"].sort(reverse=True)

        self.version += 1


class CacheManager:
    """Manages memory caches."""

    def __init__(self):
        self.access_cache = {}
        self.embedding_cache = {}
        self.max_cache_size = 1000

    def optimize(self, memories: Dict[str, Memory]):
        """Optimize caches based on memory access patterns."""
        # Track most accessed
        access_counts = [(m.access_count, mid) for mid, m in memories.items()]
        access_counts.sort(reverse=True)

        # Cache top accessed memories
        self.access_cache.clear()
        for count, memory_id in access_counts[: self.max_cache_size // 2]:
            if memory_id in memories:
                self.access_cache[memory_id] = memories[memory_id]

        # Cache memories with embeddings
        self.embedding_cache.clear()
        for memory_id, memory in memories.items():
            if (
                memory.embedding is not None
                and len(self.embedding_cache) < self.max_cache_size // 2
            ):
                self.embedding_cache[memory_id] = memory.embedding
