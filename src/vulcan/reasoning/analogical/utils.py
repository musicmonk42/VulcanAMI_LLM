"""
Utility functions for analogical reasoning.

This module provides helper functions for working with analogical mappings,
including conceptual distance computation, knowledge graph path finding,
analogy clustering, and explanation generation.

Key Functions:
    - compute_conceptual_distance: Semantic distance between concepts
    - find_conceptual_path: Shortest path in knowledge graphs
    - cluster_analogies: Group similar analogies using K-means
    - explain_analogy_differences: Generate human-readable comparisons
    - test_semantic_similarity: Demonstration and testing function

Module: vulcan.reasoning.analogical.utils
Author: Vulcan AI Team
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    # Avoid circular imports at runtime
    from .semantic_enricher import SemanticEnricher
    from .types import AnalogicalMapping

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None  # type: ignore
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, graph-based features disabled")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None  # type: ignore
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, clustering disabled")


# Constants
DEFAULT_N_CLUSTERS = 3
CLUSTERING_RANDOM_SEED = 42
SCORE_DIFF_THRESHOLD = 0.1  # Threshold for "significant" difference
MAX_ENTITIES_TO_DISPLAY = 3  # Maximum entities to show in explanations


def compute_conceptual_distance(
    concept1: str, concept2: str, enricher: SemanticEnricher
) -> float:
    """
    Compute conceptual distance between two concepts in semantic space.
    
    Distance is defined as 1 - similarity, so:
    - Distance 0.0 = identical concepts
    - Distance 1.0 = completely dissimilar concepts
    
    Args:
        concept1: First concept (term or phrase)
        concept2: Second concept (term or phrase)
        enricher: SemanticEnricher instance for similarity computation
        
    Returns:
        Conceptual distance in range [0.0, 1.0].
        
    Examples:
        >>> from vulcan.reasoning.analogical import SemanticEnricher
        >>> enricher = SemanticEnricher()
        >>> distance = compute_conceptual_distance("car", "automobile", enricher)
        >>> print(f"Distance: {distance:.3f}")  # Low distance (similar)
        >>> distance = compute_conceptual_distance("car", "banana", enricher)
        >>> print(f"Distance: {distance:.3f}")  # High distance (dissimilar)
    
    Note:
        This function is the semantic analog of edit distance for strings,
        but operates in embedding space rather than character space.
    """
    similarity = enricher.compute_semantic_similarity(concept1, concept2)
    return 1.0 - similarity


def find_conceptual_path(
    start_concept: str,
    end_concept: str,
    knowledge_graph: Optional[object] = None,
) -> List[str]:
    """
    Find shortest conceptual path between two concepts in a knowledge graph.
    
    Uses NetworkX shortest path algorithm to traverse the knowledge graph
    and find the minimal connection between concepts.
    
    Args:
        start_concept: Starting concept node
        end_concept: Target concept node
        knowledge_graph: NetworkX DiGraph representing concept relationships
        
    Returns:
        List of concepts forming the shortest path, or empty list if:
        - NetworkX is not available
        - Graph is None
        - No path exists between concepts
        - Either concept is not in the graph
        
    Examples:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([("dog", "animal"), ("animal", "living_thing")])
        >>> path = find_conceptual_path("dog", "living_thing", G)
        >>> print(path)  # ["dog", "animal", "living_thing"]
    
    Note:
        Requires NetworkX to be installed. Gracefully returns empty list
        if NetworkX is unavailable.
    """
    if not NETWORKX_AVAILABLE or knowledge_graph is None:
        return []
    
    try:
        # Verify both concepts exist in graph
        if start_concept in knowledge_graph and end_concept in knowledge_graph:
            path = nx.shortest_path(knowledge_graph, start_concept, end_concept)
            return path
    except nx.NetworkXNoPath:
        # No path exists between concepts
        logger.debug(f"No path found between '{start_concept}' and '{end_concept}'")
        return []
    except Exception as e:
        logger.warning(f"Path finding failed: {e}")
        return []
    
    return []


def cluster_analogies(
    analogies: List[AnalogicalMapping],
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> Dict[int, List[AnalogicalMapping]]:
    """
    Cluster analogical mappings by similarity using K-means.
    
    Groups analogies based on their quantitative features:
    - mapping_score: Overall quality
    - confidence: Reliability score
    - semantic_coherence: Semantic consistency
    - Entity/relation counts: Structural size
    - structural_depth: Complexity measure
    
    Args:
        analogies: List of analogical mappings to cluster
        n_clusters: Number of clusters to create (default: 3)
        
    Returns:
        Dictionary mapping cluster IDs to lists of analogies.
        If clustering fails or is unavailable, returns {0: analogies}.
        
    Examples:
        >>> mappings = [mapping1, mapping2, mapping3, ...]
        >>> clusters = cluster_analogies(mappings, n_clusters=2)
        >>> for cluster_id, cluster_mappings in clusters.items():
        ...     print(f"Cluster {cluster_id}: {len(cluster_mappings)} analogies")
    
    Note:
        Requires scikit-learn to be installed. Falls back to single cluster
        if unavailable or if clustering fails.
    """
    if not SKLEARN_AVAILABLE or not analogies:
        return {0: analogies}
    
    try:
        # Create feature vectors from mappings
        features = []
        for mapping in analogies:
            feature_vec = [
                mapping.mapping_score,
                mapping.confidence,
                mapping.semantic_coherence,
                len(mapping.entity_mappings),
                len(mapping.relation_mappings),
                mapping.structural_depth,
            ]
            features.append(feature_vec)
        
        features = np.array(features)
        
        # Ensure n_clusters doesn't exceed number of analogies
        n_clusters = min(n_clusters, len(analogies))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=CLUSTERING_RANDOM_SEED)
        labels = kmeans.fit_predict(features)
        
        # Group analogies by cluster
        clusters: Dict[int, List[AnalogicalMapping]] = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(analogies[i])
        
        return dict(clusters)
    
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        # Fallback: return all analogies in single cluster
        return {0: analogies}


def explain_analogy_differences(
    mapping1: AnalogicalMapping,
    mapping2: AnalogicalMapping,
) -> str:
    """
    Generate human-readable explanation of differences between two analogies.
    
    Analyzes and explains differences in:
    - Overall similarity scores
    - Semantic coherence
    - Entity coverage
    - Structural depth
    
    Useful for understanding why one analogy might be preferred over another
    or for debugging analogy selection logic.
    
    Args:
        mapping1: First analogical mapping
        mapping2: Second analogical mapping
        
    Returns:
        Multi-line string explaining the key differences.
        
    Examples:
        >>> explanation = explain_analogy_differences(good_mapping, poor_mapping)
        >>> print(explanation)
        Comparison of analogical mappings:
        
        The first mapping has a higher similarity score (0.85 vs 0.62).
        The first mapping is more semantically coherent.
        First mapping includes: sun, planet, orbit
        
        The first mapping captures deeper structural relations.
    
    Note:
        The explanation focuses on actionable differences that exceed
        the SCORE_DIFF_THRESHOLD (0.1) to avoid noise from trivial variations.
    """
    explanation = "Comparison of analogical mappings:\n\n"
    
    # Score comparison
    score_diff = mapping1.mapping_score - mapping2.mapping_score
    if abs(score_diff) > SCORE_DIFF_THRESHOLD:
        better = "first" if score_diff > 0 else "second"
        explanation += f"The {better} mapping has a higher similarity score "
        explanation += (
            f"({mapping1.mapping_score:.2f} vs {mapping2.mapping_score:.2f}).\n"
        )
    else:
        explanation += "Both mappings have similar similarity scores.\n"
    
    # Semantic coherence comparison
    coh_diff = mapping1.semantic_coherence - mapping2.semantic_coherence
    if abs(coh_diff) > SCORE_DIFF_THRESHOLD:
        better = "first" if coh_diff > 0 else "second"
        explanation += f"The {better} mapping is more semantically coherent.\n"
    
    # Entity mapping comparison
    entities1 = set(mapping1.entity_mappings.keys())
    entities2 = set(mapping2.entity_mappings.keys())
    
    only_in_1 = entities1 - entities2
    only_in_2 = entities2 - entities1
    
    if only_in_1:
        entities_display = ', '.join(list(only_in_1)[:MAX_ENTITIES_TO_DISPLAY])
        explanation += f"\nFirst mapping includes: {entities_display}\n"
    if only_in_2:
        entities_display = ', '.join(list(only_in_2)[:MAX_ENTITIES_TO_DISPLAY])
        explanation += f"Second mapping includes: {entities_display}\n"
    
    # Structural depth comparison
    if mapping1.structural_depth != mapping2.structural_depth:
        deeper = (
            "first"
            if mapping1.structural_depth > mapping2.structural_depth
            else "second"
        )
        explanation += f"\nThe {deeper} mapping captures deeper structural relations.\n"
    
    return explanation


def test_semantic_similarity() -> None:
    """
    Test and demonstrate semantic similarity computation.
    
    This function serves dual purposes:
    1. Verify that semantic enrichment is working correctly
    2. Provide examples of semantic similarity for different concept pairs
    
    Prints similarity scores for several concept pairs, demonstrating:
    - High similarity for related concepts (cat/dog, car/automobile)
    - Medium similarity for conceptually related terms (cat/feline)
    - Low similarity for unrelated concepts (cat/quantum)
    
    Examples:
        >>> from vulcan.reasoning.analogical.utils import test_semantic_similarity
        >>> test_semantic_similarity()
        Testing semantic similarity:
        cat vs dog: 0.654
        cat vs feline: 0.782
        cat vs quantum: 0.123
        running vs runner: 0.891
        car vs automobile: 0.876
        happy vs sad: 0.234
        
        Using TF-IDF fallback embeddings (good quality)
    
    Note:
        This is primarily a diagnostic function. The actual similarity values
        will vary depending on whether sentence-transformers is available
        or if the TF-IDF fallback is being used.
    """
    # Import here to avoid circular dependency at module level
    from .semantic_enricher import SemanticEnricher
    
    enricher = SemanticEnricher()
    
    # Test concept pairs with expected similarity patterns
    test_pairs = [
        ("cat", "dog"),  # Related animals
        ("cat", "feline"),  # Specific to general
        ("cat", "quantum"),  # Unrelated
        ("running", "runner"),  # Morphologically related
        ("car", "automobile"),  # Synonyms
        ("happy", "sad"),  # Antonyms
    ]
    
    print("Testing semantic similarity:")
    for term1, term2 in test_pairs:
        similarity = enricher.compute_semantic_similarity(term1, term2)
        print(f"{term1} vs {term2}: {similarity:.3f}")
    
    # Indicate which embedding method is being used
    if enricher.embedding_model:
        print("\nUsing sentence-transformers embeddings (best quality)")
    else:
        print("\nUsing TF-IDF fallback embeddings (good quality)")


# Export public API
__all__ = [
    "compute_conceptual_distance",
    "find_conceptual_path",
    "cluster_analogies",
    "explain_analogy_differences",
    "test_semantic_similarity",
    "NETWORKX_AVAILABLE",
    "SKLEARN_AVAILABLE",
]
