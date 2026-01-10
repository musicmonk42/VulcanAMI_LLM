"""
Type definitions for the analogical reasoning system.

This module contains all dataclasses, enums, and type definitions used throughout
the analogical reasoning subsystem. It follows the single responsibility principle
by isolating type definitions from business logic.

Design Principles:
    - Immutable where possible (frozen dataclasses for value objects)
    - Rich type annotations for IDE support and type checking
    - Comprehensive docstrings following Google style
    - Self-contained with minimal external dependencies

Module: vulcan.reasoning.analogical.types
Author: Vulcan AI Team
License: Proprietary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt


class MappingType(Enum):
    """
    Types of analogical mappings based on Structure-Mapping Theory.
    
    Attributes:
        SURFACE: Surface-level similarity (superficial features)
        STRUCTURAL: Deep structural similarity (relations between elements)
        PRAGMATIC: Goal-directed, context-dependent similarity
        SEMANTIC: Meaning-based similarity using semantic spaces
    
    References:
        Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
        Cognitive Science, 7(2), 155-170.
    """

    SURFACE = "surface"
    STRUCTURAL = "structural"
    PRAGMATIC = "pragmatic"
    SEMANTIC = "semantic"


class SemanticRelationType(Enum):
    """
    Types of semantic relations between concepts.
    
    Based on cognitive linguistics and WordNet relation taxonomy.
    Used for enriching analogical mappings with semantic structure.
    
    Attributes:
        HYPERNYM: Is-a relationship (cat is-a animal)
        HYPONYM: Inverse of hypernym (animal has-hyponym cat)
        MERONYM: Part-of relationship (wheel is part-of car)
        HOLONYM: Has-part relationship (car has-part wheel)
        SYNONYM: Same meaning (happy ≈ joyful)
        ANTONYM: Opposite meaning (hot ↔ cold)
        CAUSE: Causation (virus causes disease)
        EFFECT: Result relationship (disease is-effect-of virus)
        ATTRIBUTE: Property relationship (sky has-attribute blue)
    """

    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    MERONYM = "meronym"
    HOLONYM = "holonym"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    CAUSE = "cause"
    EFFECT = "effect"
    ATTRIBUTE = "attribute"


@dataclass
class Entity:
    """
    Represents an entity in analogical reasoning with rich semantic enrichment.
    
    Entities are the basic building blocks of analogical mappings. Each entity
    can be enriched with embeddings, semantic categories, and syntactic information
    to enable sophisticated similarity computations.
    
    Attributes:
        name: Human-readable name of the entity
        attributes: Dictionary of entity attributes (e.g., {"color": "red"})
        entity_type: Semantic type (e.g., "person", "object", "concept")
        embedding: Optional semantic embedding vector (384-dim for compatibility)
        semantic_category: Broad semantic category for quick filtering
        pos_tag: Part-of-speech tag (if available from NLP pipeline)
        dependency_role: Syntactic role in sentence structure
    
    Examples:
        >>> entity = Entity(name="car", entity_type="vehicle")
        >>> entity.attributes = {"wheels": 4, "motor": "combustion"}
        >>> similarity = entity.similarity_to(other_entity)
    
    Note:
        Entities are hashable and can be used as dictionary keys.
        Equality is based on (name, entity_type) tuple.
    """

    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    entity_type: str = "object"
    embedding: Optional[npt.NDArray[np.float32]] = None
    semantic_category: Optional[str] = None
    pos_tag: Optional[str] = None
    dependency_role: Optional[str] = None

    def __hash__(self) -> int:
        """
        Make Entity hashable for use in sets and as dict keys.
        
        Returns:
            Hash based on name and entity_type tuple.
        """
        return hash((self.name, self.entity_type))

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on name and entity_type.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if both name and entity_type match, False otherwise.
        """
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type

    def similarity_to(self, other: Entity, use_embeddings: bool = True) -> float:
        """
        Compute sophisticated multi-faceted similarity to another entity.
        
        This method combines multiple similarity signals:
        - Embedding-based semantic similarity (when available)
        - Type compatibility (entity_type match)
        - Semantic category match
        - Attribute structural similarity
        - Lexical name similarity (fallback)
        
        Args:
            other: Entity to compare with
            use_embeddings: Whether to use embedding-based similarity
            
        Returns:
            Similarity score in range [0.0, 1.0] where 1.0 is identical.
            
        Examples:
            >>> cat = Entity("cat", entity_type="animal")
            >>> dog = Entity("dog", entity_type="animal")
            >>> cat.similarity_to(dog)  # ~0.7 (same type, related concepts)
            >>> car = Entity("car", entity_type="vehicle")
            >>> cat.similarity_to(car)  # ~0.0 (different types)
        """
        if not isinstance(other, Entity):
            return 0.0

        # Identity check: comparing to self
        if self is other or (
            self.name == other.name and self.entity_type == other.entity_type
        ):
            return 1.0

        # Different entity types have zero similarity (strong type system)
        if self.entity_type != other.entity_type:
            return 0.0

        # Type compatibility (same type at this point)
        type_match = 1.0

        # Embedding-based semantic similarity (best signal when available)
        if (
            use_embeddings
            and self.embedding is not None
            and other.embedding is not None
        ):
            embedding_sim = self._embedding_similarity(self.embedding, other.embedding)
        else:
            embedding_sim = 0.0

        # Semantic category match (categorical similarity)
        category_match = (
            1.0
            if (
                self.semantic_category
                and self.semantic_category == other.semantic_category
            )
            else 0.0
        )

        # Attribute structural similarity
        attr_sim = self._attribute_similarity(self.attributes, other.attributes)

        # Lexical name similarity (fallback for no embeddings)
        name_sim = self._lexical_similarity(self.name, other.name)

        # Weighted combination based on available signals
        if self.embedding is not None and other.embedding is not None:
            # Prefer embeddings when available
            weights = [0.4, 0.3, 0.15, 0.15]  # embedding, type, category, attr
            scores = [embedding_sim, type_match, category_match, attr_sim]
        else:
            # Fall back to lexical and structural features
            weights = [0.3, 0.3, 0.2, 0.2]  # name, type, category, attr
            scores = [name_sim, type_match, category_match, attr_sim]

        return sum(w * s for w, s in zip(weights, scores))

    def _embedding_similarity(
        self, emb1: npt.NDArray[np.float32], emb2: npt.NDArray[np.float32]
    ) -> float:
        """
        Compute cosine similarity between embedding vectors.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity in range [-1.0, 1.0], clipped to [0.0, 1.0].
            Returns 0.0 on any error.
        """
        try:
            emb1_flat = emb1.flatten()
            emb2_flat = emb2.flatten()

            norm1 = np.linalg.norm(emb1_flat)
            norm2 = np.linalg.norm(emb2_flat)

            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0

            cos_sim = np.dot(emb1_flat, emb2_flat) / (norm1 * norm2)
            return float(np.clip(cos_sim, 0.0, 1.0))
        except Exception:
            return 0.0

    def _attribute_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """
        Compute deep attribute similarity with type awareness.
        
        Considers both key overlap and value similarity for common keys.
        Handles numerical, string, and exact-match comparisons appropriately.
        
        Args:
            attrs1: First attribute dictionary
            attrs2: Second attribute dictionary
            
        Returns:
            Similarity score in range [0.0, 1.0].
        """
        if not attrs1 and not attrs2:
            return 1.0  # Both empty
        if not attrs1 or not attrs2:
            return 0.0  # One empty, one not

        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        all_keys = set(attrs1.keys()) | set(attrs2.keys())

        if not all_keys:
            return 1.0

        # Jaccard similarity for key overlap
        key_sim = len(common_keys) / len(all_keys)

        # Value similarity for common keys
        value_sims = []
        for key in common_keys:
            v1, v2 = attrs1[key], attrs2[key]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Normalized numerical difference
                max_val = max(abs(v1), abs(v2), 1e-10)
                value_sims.append(1.0 - min(1.0, abs(v1 - v2) / max_val))
            elif isinstance(v1, str) and isinstance(v2, str):
                # String similarity using lexical comparison
                value_sims.append(self._lexical_similarity(v1, v2))
            else:
                # Exact match for other types
                value_sims.append(1.0 if v1 == v2 else 0.0)

        value_sim = float(np.mean(value_sims)) if value_sims else 0.0

        return (key_sim + value_sim) / 2.0

    def _lexical_similarity(self, s1: str, s2: str) -> float:
        """
        Multi-level lexical similarity combining multiple metrics.
        
        Combines:
        - Exact match check
        - Edit distance (Levenshtein)
        - Character n-gram overlap (Jaccard)
        - Token set overlap
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score in range [0.0, 1.0].
        """
        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Exact match
        if s1_lower == s2_lower:
            return 1.0

        # Normalized edit distance
        edit_dist = self._levenshtein_distance(s1_lower, s2_lower)
        max_len = max(len(s1_lower), len(s2_lower), 1)
        edit_sim = 1.0 - (edit_dist / max_len)

        # Character n-gram similarity
        ngram_sim = self._ngram_similarity(s1_lower, s2_lower, n=2)

        # Token overlap (word-level)
        tokens1 = set(s1_lower.split())
        tokens2 = set(s2_lower.split())
        if tokens1 or tokens2:
            token_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            token_sim = 0.0

        # Averaged combination
        return (edit_sim + ngram_sim + token_sim) / 3.0

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein edit distance between two strings.
        
        Uses dynamic programming with O(min(m,n)) space optimization.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Minimum number of single-character edits required.
        """
        # Ensure s1 is the longer string for optimization
        if len(s1) < len(s2):
            return Entity._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        # Use rolling array for space optimization
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def _ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
        """
        Compute n-gram Jaccard similarity between strings.
        
        Args:
            s1: First string
            s2: Second string
            n: N-gram size (default: 2 for bigrams)
            
        Returns:
            Jaccard similarity of n-gram sets in range [0.0, 1.0].
        """
        if len(s1) < n or len(s2) < n:
            return 1.0 if s1 == s2 else 0.0

        ngrams1 = set(s1[i : i + n] for i in range(len(s1) - n + 1))
        ngrams2 = set(s2[i : i + n] for i in range(len(s2) - n + 1))

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union


@dataclass
class Relation:
    """
    Represents a relation between entities with semantic metadata.
    
    Relations encode structured knowledge about how entities interact or relate.
    They can be enriched with semantic types and confidence scores.
    
    Attributes:
        predicate: The relation name (e.g., "causes", "is-a", "has-part")
        arguments: List of entity names involved in this relation
        relation_type: Type classification (e.g., "binary", "ternary")
        order: Relation order (1 for first-order, 2 for second-order, etc.)
        metadata: Additional structured information about the relation
        semantic_type: Optional semantic classification of the relation
        confidence: Confidence score for this relation [0.0, 1.0]
    
    Examples:
        >>> rel = Relation(
        ...     predicate="causes",
        ...     arguments=["smoking", "cancer"],
        ...     semantic_type=SemanticRelationType.CAUSE
        ... )
    """

    predicate: str
    arguments: List[str]
    relation_type: str = "binary"
    order: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_type: Optional[SemanticRelationType] = None
    confidence: float = 1.0

    def matches(self, other: Relation, entity_mapping: Dict[str, str]) -> bool:
        """
        Check if this relation matches another given an entity mapping.
        
        Used in structure mapping to verify relation consistency.
        
        Args:
            other: Relation to match against
            entity_mapping: Dictionary mapping source entities to target entities
            
        Returns:
            True if relations match under the given entity mapping.
        """
        if self.predicate != other.predicate:
            return False

        if len(self.arguments) != len(other.arguments):
            return False

        for arg1, arg2 in zip(self.arguments, other.arguments):
            if arg1 in entity_mapping:
                if entity_mapping[arg1] != arg2:
                    return False
            elif arg1 != arg2:
                return False

        return True

    def semantic_similarity_to(self, other: Relation) -> float:
        """
        Compute semantic similarity to another relation.
        
        Considers predicate match, semantic type, and relation type.
        
        Args:
            other: Relation to compare with
            
        Returns:
            Similarity score in range [0.0, 1.0].
        """
        # Predicate similarity (exact or partial match)
        pred_sim = 1.0 if self.predicate == other.predicate else 0.5

        # Semantic type match (if both have semantic types)
        type_sim = (
            1.0
            if (self.semantic_type and self.semantic_type == other.semantic_type)
            else 0.0
        )

        # Relation type match (structural compatibility)
        rel_type_sim = 1.0 if self.relation_type == other.relation_type else 0.5

        return (pred_sim + type_sim + rel_type_sim) / 3.0


@dataclass
class AnalogicalMapping:
    """
    Represents a complete analogical mapping with rich evaluation metrics.
    
    An analogical mapping connects a source domain to a target domain by
    establishing correspondences between entities and relations. Quality
    metrics help rank and select the best mappings.
    
    Attributes:
        source_domain: Name/description of the source domain
        target_domain: Name/description of the target domain
        entity_mappings: Dictionary mapping source entities to target entities
        relation_mappings: List of (source_relation, target_relation) pairs
        mapping_score: Overall quality score [0.0, 1.0]
        mapping_type: Type of this mapping (surface, structural, etc.)
        confidence: Confidence in this mapping [0.0, 1.0]
        explanation: Human-readable explanation of the mapping
        semantic_coherence: Semantic consistency score [0.0, 1.0]
        structural_depth: Number of relation levels captured
    
    Examples:
        >>> mapping = AnalogicalMapping(
        ...     source_domain="solar_system",
        ...     target_domain="atom",
        ...     entity_mappings={"sun": "nucleus", "planet": "electron"},
        ...     relation_mappings=[],
        ...     mapping_score=0.85,
        ...     mapping_type=MappingType.STRUCTURAL
        ... )
    """

    source_domain: str
    target_domain: str
    entity_mappings: Dict[str, str]
    relation_mappings: List[Tuple[Relation, Relation]]
    mapping_score: float
    mapping_type: MappingType
    confidence: float = 0.0
    explanation: str = ""
    semantic_coherence: float = 0.0
    structural_depth: int = 0


# Export all public types
__all__ = [
    "MappingType",
    "SemanticRelationType",
    "Entity",
    "Relation",
    "AnalogicalMapping",
]
