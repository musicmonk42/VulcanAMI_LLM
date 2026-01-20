"""
Goal relevance analysis for analogical reasoning.

This module provides sophisticated goal-oriented analysis capabilities for
analogical reasoning. It determines which entity mappings are most relevant
to achieving a specified goal by combining NLP, semantic similarity, causal
reasoning, and dependency parsing.

Key Features:
    - Multi-faceted relevance scoring combining 5 different signals
    - NLP-based entity and concept extraction using spaCy
    - Semantic similarity using embeddings
    - Causal relevance analysis
    - Dependency parsing for syntactic roles
    - Graceful fallback when NLP tools unavailable

Module: vulcan.reasoning.analogical.goal_analyzer
Author: Vulcan AI Team
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Set

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .semantic_enricher import SemanticEnricher
    from .types import Relation

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
    
    # Import the lazy-loading function from semantic_enricher
    from .semantic_enricher import get_nlp
except ImportError:
    SPACY_AVAILABLE = False
    
    def get_nlp():
        """Return None when spaCy unavailable."""
        return None


# Constants
DEFAULT_RELEVANCE_THRESHOLD = 0.5
DIRECT_MENTION_SCORE = 1.0
ENTITY_OVERLAP_SCORE = 0.8
FALLBACK_RELEVANCE_SCORE = 0.5
HIGH_DEPENDENCY_SCORE = 0.8
LOW_DEPENDENCY_SCORE = 0.3
NEUTRAL_DEPENDENCY_SCORE = 0.5

# Weights for relevance scoring (sum to 1.0)
WEIGHT_DIRECT_MENTION = 0.3
WEIGHT_ENTITY_OVERLAP = 0.2
WEIGHT_SEMANTIC = 0.25
WEIGHT_CAUSAL = 0.15
WEIGHT_DEPENDENCY = 0.1

# Weights when dependency analysis unavailable
WEIGHTS_NO_DEPENDENCY = [0.35, 0.25, 0.25, 0.15]

# Action indicators for concept extraction
ACTION_INDICATORS = [
    "find", "get", "make", "create", "solve", "achieve",
    "reach", "obtain", "maximize", "minimize",
]

# Important syntactic roles for dependency parsing
IMPORTANT_SYNTACTIC_ROLES = {"nsubj", "dobj", "pobj", "ROOT"}


class GoalRelevanceAnalyzer:
    """
    Analyzes goal relevance using NLP and semantic understanding.
    
    This class provides sophisticated analysis of which entity mappings are most
    relevant to achieving a specified goal. It combines multiple signals:
    - Direct mention in goal text
    - Entity overlap with extracted goal entities
    - Semantic similarity using embeddings
    - Causal relevance through relation analysis
    - Dependency parsing for syntactic roles (when spaCy available)
    
    The relevance scores help prioritize analogical mappings that are most
    likely to be useful for goal achievement.
    
    Attributes:
        semantic_enricher: SemanticEnricher instance for embeddings
        use_spacy: Whether spaCy is available and loaded
    
    Examples:
        >>> from vulcan.reasoning.analogical import SemanticEnricher, GoalRelevanceAnalyzer
        >>> enricher = SemanticEnricher()
        >>> analyzer = GoalRelevanceAnalyzer(enricher)
        >>> 
        >>> mappings = {"sun": "nucleus", "planet": "electron"}
        >>> goal = "understand atomic structure"
        >>> source = {"entities": [...], "relations": [...]}
        >>> target = {"entities": [...], "relations": [...]}
        >>> 
        >>> relevance = analyzer.analyze_goal_relevance(mappings, goal, source, target)
        >>> print(relevance)  # {"sun": 0.85, "planet": 0.78}
        >>> 
        >>> # Filter by relevance threshold
        >>> filtered = analyzer.filter_by_relevance(mappings, relevance, threshold=0.5)
    
    Note:
        Gracefully degrades when spaCy is unavailable, using heuristic fallbacks
        for entity and concept extraction.
    """
    
    def __init__(self, semantic_enricher: SemanticEnricher):
        """
        Initialize the goal relevance analyzer.
        
        Args:
            semantic_enricher: SemanticEnricher instance for computing embeddings
                and semantic similarities.
        """
        self.semantic_enricher = semantic_enricher
        # Check if spaCy is available AND model can be loaded
        self.use_spacy = SPACY_AVAILABLE and get_nlp() is not None
    
    def analyze_goal_relevance(
        self,
        mappings: Dict[str, str],
        goal: Any,
        source: Dict,
        target: Dict,
    ) -> Dict[str, float]:
        """
        Analyze which entity mappings are relevant to achieving the goal.
        
        Computes relevance scores for each entity mapping using multiple signals:
        1. Direct mention in goal text (highest weight)
        2. Overlap with extracted goal entities
        3. Semantic similarity to goal
        4. Causal relevance through relations
        5. Syntactic importance via dependency parsing (optional)
        
        Args:
            mappings: Dictionary mapping source entities to target entities
            goal: Goal description (any type, converted to string)
            source: Source domain dictionary with 'entities' and 'relations'
            target: Target domain dictionary with 'entities' and 'relations'
            
        Returns:
            Dictionary mapping entity names to relevance scores [0.0, 1.0].
            Higher scores indicate greater relevance to goal achievement.
            
        Examples:
            >>> mappings = {"sun": "nucleus", "planet": "electron"}
            >>> goal = "explain atomic orbits"
            >>> relevance = analyzer.analyze_goal_relevance(mappings, goal, src, tgt)
            >>> print(relevance["planet"])  # High score (orbit is key concept)
        
        Note:
            If no goal is specified (None or empty), returns 1.0 for all mappings
            (assumes all mappings are equally relevant).
        """
        if not goal:
            # No goal specified, all mappings equally relevant
            return {k: 1.0 for k in mappings.keys()}
        
        goal_text = str(goal)
        
        # Extract goal entities and concepts
        goal_entities = self._extract_goal_entities(goal_text)
        goal_concepts = self._extract_goal_concepts(goal_text)
        goal_embedding = self.semantic_enricher._get_embedding(goal_text)
        
        relevance_scores = {}
        
        for source_entity, target_entity in mappings.items():
            score = self._compute_entity_goal_relevance(
                source_entity,
                target_entity,
                goal_text,
                goal_entities,
                goal_concepts,
                goal_embedding,
                source,
                target,
            )
            relevance_scores[source_entity] = score
        
        return relevance_scores
    
    def _extract_goal_entities(self, goal_text: str) -> Set[str]:
        """
        Extract named entities and key nouns from goal text.
        
        Uses spaCy for accurate NLP-based extraction when available,
        with heuristic fallback for capitalized words and long words.
        
        Args:
            goal_text: Goal description text
            
        Returns:
            Set of lowercase entity strings.
        """
        entities = set()
        
        if self.use_spacy:
            nlp_instance = get_nlp()
            if nlp_instance:
                try:
                    doc = nlp_instance(goal_text)
                    
                    # Named entities
                    for ent in doc.ents:
                        entities.add(ent.text.lower())
                    
                    # Important nouns (non-stop words)
                    for token in doc:
                        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                            entities.add(token.text.lower())
                except Exception as e:
                    logger.debug(f"Failed to extract entities from text: {e}")
        
        # Fallback: extract capitalized words and longer words heuristically
        words = goal_text.split()
        for word in words:
            if word and (word[0].isupper() or len(word) > 5):
                entities.add(word.lower())
        
        return entities
    
    def _extract_goal_concepts(self, goal_text: str) -> Set[str]:
        """
        Extract key concepts and action verbs from goal text.
        
        Focuses on verbs (actions) and key adjectives that indicate
        the nature of the goal.
        
        Args:
            goal_text: Goal description text
            
        Returns:
            Set of lowercase concept strings.
        """
        concepts = set()
        
        if self.use_spacy:
            nlp_instance = get_nlp()
            if nlp_instance:
                try:
                    doc = nlp_instance(goal_text)
                    
                    # Verbs (actions) - use lemma for canonical form
                    for token in doc:
                        if token.pos_ == "VERB" and not token.is_stop:
                            concepts.add(token.lemma_.lower())
                    
                    # Key adjectives (descriptors)
                    for token in doc:
                        if token.pos_ == "ADJ" and not token.is_stop:
                            concepts.add(token.text.lower())
                except Exception as e:
                    logger.debug(f"Failed to extract concepts from text: {e}")
        
        # Fallback: detect common action words
        goal_lower = goal_text.lower()
        for action in ACTION_INDICATORS:
            if action in goal_lower:
                concepts.add(action)
        
        return concepts
    
    def _compute_entity_goal_relevance(
        self,
        source_entity: str,
        target_entity: str,
        goal_text: str,
        goal_entities: Set[str],
        goal_concepts: Set[str],
        goal_embedding: npt.NDArray[np.float32],
        source: Dict,
        target: Dict,
    ) -> float:
        """
        Compute comprehensive relevance score for an entity mapping.
        
        Combines 5 different relevance signals with optimized weights:
        1. Direct mention (0.30)
        2. Entity overlap (0.20)
        3. Semantic similarity (0.25)
        4. Causal relevance (0.15)
        5. Dependency role (0.10, optional)
        
        Args:
            source_entity: Source domain entity name
            target_entity: Target domain entity name
            goal_text: Goal description text
            goal_entities: Extracted goal entities
            goal_concepts: Extracted goal concepts
            goal_embedding: Goal text embedding vector
            source: Source domain dictionary
            target: Target domain dictionary
            
        Returns:
            Weighted relevance score in range [0.0, 1.0].
        """
        scores = []
        
        # 1. Direct mention in goal (highest weight)
        source_lower = source_entity.lower()
        target_lower = target_entity.lower()
        goal_lower = goal_text.lower()
        
        direct_mention_score = (
            DIRECT_MENTION_SCORE
            if (source_lower in goal_lower or target_lower in goal_lower)
            else 0.0
        )
        scores.append(direct_mention_score)
        
        # 2. Entity overlap with goal entities
        entity_overlap_score = (
            ENTITY_OVERLAP_SCORE
            if (source_lower in goal_entities or target_lower in goal_entities)
            else 0.0
        )
        scores.append(entity_overlap_score)
        
        # 3. Semantic similarity to goal
        source_emb = self.semantic_enricher._get_embedding(source_entity)
        target_emb = self.semantic_enricher._get_embedding(target_entity)
        
        source_goal_sim = self._embedding_similarity(source_emb, goal_embedding)
        target_goal_sim = self._embedding_similarity(target_emb, goal_embedding)
        semantic_score = max(source_goal_sim, target_goal_sim)
        scores.append(semantic_score)
        
        # 4. Causal relevance (entity in goal-relevant relations)
        causal_score = self._compute_causal_relevance(
            source_entity, target_entity, goal_concepts, source, target
        )
        scores.append(causal_score)
        
        # 5. Dependency analysis (if spaCy available)
        if self.use_spacy:
            dependency_score = self._compute_dependency_relevance(
                source_entity, target_entity, goal_text
            )
            scores.append(dependency_score)
        
        # Weighted average
        weights = (
            [WEIGHT_DIRECT_MENTION, WEIGHT_ENTITY_OVERLAP, WEIGHT_SEMANTIC,
             WEIGHT_CAUSAL, WEIGHT_DEPENDENCY]
            if len(scores) == 5
            else WEIGHTS_NO_DEPENDENCY
        )
        weights = weights[:len(scores)]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _embedding_similarity(
        self, emb1: npt.NDArray[np.float32], emb2: npt.NDArray[np.float32]
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity in range [0.0, 1.0].
            Returns 0.0 if either vector has near-zero norm.
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return float(np.clip(np.dot(emb1, emb2) / (norm1 * norm2), 0.0, 1.0))
    
    def _compute_causal_relevance(
        self,
        source_entity: str,
        target_entity: str,
        goal_concepts: Set[str],
        source: Dict,
        target: Dict,
    ) -> float:
        """
        Compute causal relevance by checking goal-relevant relations.
        
        Analyzes whether the entity participates in relations whose predicates
        contain goal-relevant concepts (verbs, actions).
        
        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            goal_concepts: Set of goal concepts to match
            source: Source domain dictionary
            target: Target domain dictionary
            
        Returns:
            Proportion of entity's relations that are goal-relevant [0.0, 1.0].
            Returns 0.5 if entity has no relations (neutral).
        """
        # Check if entity participates in relations with goal-relevant predicates
        source_rels = source.get("relations", [])
        
        relevant_count = 0
        total_count = 0
        
        # Import Relation type here to avoid circular import
        from .types import Relation
        
        for rel in source_rels:
            if isinstance(rel, Relation):
                if source_entity in rel.arguments:
                    total_count += 1
                    pred_lower = rel.predicate.lower()
                    if any(concept in pred_lower for concept in goal_concepts):
                        relevant_count += 1
            elif isinstance(rel, tuple) and len(rel) >= 3:
                # Tuple format: (entity1, predicate, entity2)
                if source_entity in [rel[0], rel[2]]:
                    total_count += 1
                    pred_lower = str(rel[1]).lower()
                    if any(concept in pred_lower for concept in goal_concepts):
                        relevant_count += 1
        
        return (
            relevant_count / total_count
            if total_count > 0
            else FALLBACK_RELEVANCE_SCORE  # Neutral score when no relations
        )
    
    def _compute_dependency_relevance(
        self, source_entity: str, target_entity: str, goal_text: str
    ) -> float:
        """
        Use dependency parsing to determine syntactic relevance.
        
        Entities playing important syntactic roles (subject, object, root)
        in the goal sentence receive higher relevance scores.
        
        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            goal_text: Goal description text
            
        Returns:
            Relevance score based on syntactic role:
            - 0.8: Important role (subject, object, root)
            - 0.5: Mentioned but not in important role
            - 0.3: Not mentioned or spaCy unavailable
        """
        nlp_instance = get_nlp()
        if not nlp_instance:
            return LOW_DEPENDENCY_SCORE  # Fallback when spaCy not available
        
        try:
            doc = nlp_instance(goal_text)
            
            # Find if entities play important syntactic roles
            source_lower = source_entity.lower()
            target_lower = target_entity.lower()
            
            for token in doc:
                if (source_lower in token.text.lower() or
                    target_lower in token.text.lower()):
                    if token.dep_ in IMPORTANT_SYNTACTIC_ROLES:
                        return HIGH_DEPENDENCY_SCORE
                    else:
                        return NEUTRAL_DEPENDENCY_SCORE
            
            return LOW_DEPENDENCY_SCORE
        except Exception:
            return NEUTRAL_DEPENDENCY_SCORE
    
    def filter_by_relevance(
        self,
        mappings: Dict[str, str],
        relevance_scores: Dict[str, float],
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> Dict[str, str]:
        """
        Filter entity mappings by relevance threshold.
        
        Keeps only mappings whose relevance score meets or exceeds the threshold.
        Useful for focusing on the most goal-relevant analogical mappings.
        
        Args:
            mappings: Dictionary mapping source entities to target entities
            relevance_scores: Dictionary of relevance scores from analyze_goal_relevance
            threshold: Minimum relevance score to keep (default: 0.5)
            
        Returns:
            Filtered dictionary containing only relevant mappings.
            
        Examples:
            >>> mappings = {"a": "x", "b": "y", "c": "z"}
            >>> scores = {"a": 0.8, "b": 0.4, "c": 0.6}
            >>> filtered = analyzer.filter_by_relevance(mappings, scores, threshold=0.5)
            >>> print(filtered)  # {"a": "x", "c": "z"}
        """
        return {
            entity: target
            for entity, target in mappings.items()
            if relevance_scores.get(entity, 0.0) >= threshold
        }


# Export public API
__all__ = [
    "GoalRelevanceAnalyzer",
    "SPACY_AVAILABLE",
    "get_nlp",
]
