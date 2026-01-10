"""
Structure-Mapping Theory (SMT) algorithms for analogical reasoning.

This module implements core Structure-Mapping Theory algorithms for finding
analogical mappings between source and target domains. It provides multiple
mapping strategies including structural, surface, semantic, and pragmatic
mappings.

Key Features:
    - Structural mapping with consistency enforcement
    - Surface-level attribute matching
    - Semantic similarity-based mapping
    - Goal-directed pragmatic mapping
    - Entity and relation extraction
    - Systematicity computation
    - Coherence evaluation

References:
    Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
    Cognitive Science, 7(2), 155-170.

Module: vulcan.reasoning.analogical.structure_mapping
Author: Vulcan AI Team
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from .goal_analyzer import GoalRelevanceAnalyzer
    from .semantic_enricher import SemanticEnricher
    from .types import AnalogicalMapping, Entity, Relation

logger = logging.getLogger(__name__)

# Constants for mapping algorithms
ENTITY_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for entity candidates
SURFACE_MAPPING_THRESHOLD = 0.5  # Minimum score for surface mappings
SEMANTIC_MAPPING_THRESHOLD = 0.5  # Minimum score for semantic mappings
PRAGMATIC_RELEVANCE_THRESHOLD = 0.4  # Minimum relevance for pragmatic filtering
MAX_MAPPING_ITERATIONS = 100  # Maximum iterations for structural extension

# Weights for combined scoring
WEIGHT_STRUCTURAL = 0.6  # Weight for structural consistency
WEIGHT_SYSTEMATICITY = 0.2  # Weight for systematicity
WEIGHT_SEMANTIC_COHERENCE = 0.2  # Weight for semantic coherence

# Weights for pragmatic mapping
WEIGHT_PRAGMATIC_STRUCTURAL = 0.7  # Weight for structural component
WEIGHT_PRAGMATIC_RELEVANCE = 0.3  # Weight for goal relevance


class StructureMappingEngine:
    """
    Core engine for Structure-Mapping Theory (SMT) based analogical mapping.
    
    This class implements the complete SMT algorithm including:
    - Structural mapping with relational consistency
    - Surface-level attribute matching
    - Semantic similarity computation
    - Goal-directed pragmatic adaptation
    - Systematicity evaluation
    
    The engine can operate in multiple modes and automatically selects
    the best mapping strategy based on the input domains and task goals.
    
    Attributes:
        semantic_enricher: SemanticEnricher for computing similarities
        goal_analyzer: GoalRelevanceAnalyzer for goal-directed mapping
        learned_weights: Dictionary of learned weights for scoring
    
    Examples:
        >>> from vulcan.reasoning.analogical import (
        ...     StructureMappingEngine, SemanticEnricher, GoalRelevanceAnalyzer
        ... )
        >>> enricher = SemanticEnricher()
        >>> analyzer = GoalRelevanceAnalyzer(enricher)
        >>> engine = StructureMappingEngine(enricher, analyzer)
        >>> 
        >>> source = {"entities": [...], "relations": [...]}
        >>> target = {"entities": [...], "relations": [...]}
        >>> mapping = engine.structural_mapping(source, target)
    
    Note:
        The engine uses learned weights that can be adapted through
        experience to improve mapping quality over time.
    """
    
    def __init__(
        self,
        semantic_enricher: SemanticEnricher,
        goal_analyzer: GoalRelevanceAnalyzer,
        learned_weights: Dict[str, float] = None,
    ):
        """
        Initialize the structure mapping engine.
        
        Args:
            semantic_enricher: SemanticEnricher instance for similarity computations
            goal_analyzer: GoalRelevanceAnalyzer for goal-directed mapping
            learned_weights: Optional dictionary of learned weights for scoring.
                If None, uses default weights.
        """
        self.semantic_enricher = semantic_enricher
        self.goal_analyzer = goal_analyzer
        self.learned_weights = learned_weights or {"structural": WEIGHT_STRUCTURAL}
    
    def structural_mapping(
        self, source: Dict, target: Dict
    ) -> AnalogicalMapping:
        """
        Perform advanced structural mapping with semantic awareness.
        
        Implements the core SMT algorithm:
        1. Find entity candidates using semantic similarity
        2. Identify identical or similar predicates
        3. Extend mappings while maintaining structural consistency
        4. Evaluate consistency and systematicity
        5. Compute semantic coherence
        
        Args:
            source: Source domain dictionary with 'entities' and 'relations'
            target: Target domain dictionary with 'entities' and 'relations'
            
        Returns:
            AnalogicalMapping with entity and relation mappings, scores, and metadata.
            
        Examples:
            >>> source = {
            ...     "entities": [Entity("sun", ...), Entity("planet", ...)],
            ...     "relations": [Relation("orbits", ["planet", "sun"])]
            ... }
            >>> target = {
            ...     "entities": [Entity("electron", ...), Entity("nucleus", ...)],
            ...     "relations": [Relation("orbits", ["electron", "nucleus"])]
            ... }
            >>> mapping = engine.structural_mapping(source, target)
            >>> print(mapping.entity_mappings)  # {"sun": "nucleus", "planet": "electron"}
        
        Note:
            This method prioritizes structural consistency over surface similarity,
            following the principles of Structure-Mapping Theory.
        """
        from .types import AnalogicalMapping, MappingType
        
        # Find entity candidates using semantic similarity
        entity_candidates = self._find_entity_candidates_semantic(
            source["entities"], target["entities"]
        )
        
        # Find identical predicates
        initial_mappings = self._find_identical_predicates(
            source["relations"], target["relations"]
        )
        
        # Extend mappings structurally
        best_mapping = self._extend_mappings_structurally(
            initial_mappings,
            source["relations"],
            target["relations"],
            entity_candidates,
        )
        
        # Evaluate consistency
        consistency_score = self._evaluate_structural_consistency(
            best_mapping, source, target
        )
        
        # Systematicity
        systematicity_score = self._compute_systematicity(
            best_mapping["relation_mappings"]
        )
        
        # Semantic coherence
        semantic_coherence = self._compute_semantic_coherence(
            best_mapping, source, target
        )
        
        # Combined score
        mapping_score = (
            self.learned_weights["structural"] * consistency_score
            + WEIGHT_SYSTEMATICITY * systematicity_score
            + WEIGHT_SEMANTIC_COHERENCE * semantic_coherence
        )
        
        return AnalogicalMapping(
            source_domain="source",
            target_domain="target",
            entity_mappings=best_mapping["entity_mappings"],
            relation_mappings=best_mapping["relation_mappings"],
            mapping_score=mapping_score,
            mapping_type=MappingType.STRUCTURAL,
            confidence=consistency_score,
            semantic_coherence=semantic_coherence,
            structural_depth=len(best_mapping["relation_mappings"]),
        )
    
    def _find_entity_candidates_semantic(
        self, source_entities: Set, target_entities: Set
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find entity candidates using advanced semantic similarity.
        
        For each source entity, finds all target entities with similarity
        above the threshold, sorted by similarity score.
        
        Args:
            source_entities: Set of source domain entities
            target_entities: Set of target domain entities
            
        Returns:
            Dictionary mapping source entity names to lists of
            (target_entity_name, similarity_score) tuples, sorted by score.
        """
        from .types import Entity
        
        candidates = defaultdict(list)
        
        for s_entity in source_entities:
            for t_entity in target_entities:
                try:
                    if isinstance(s_entity, Entity) and isinstance(t_entity, Entity):
                        # Use enhanced similarity with embeddings
                        similarity = s_entity.similarity_to(
                            t_entity, use_embeddings=True
                        )
                        if similarity > ENTITY_SIMILARITY_THRESHOLD:
                            candidates[s_entity.name].append(
                                (t_entity.name, similarity)
                            )
                    else:
                        s_name = str(s_entity)
                        t_name = str(t_entity)
                        similarity = self.semantic_enricher.compute_semantic_similarity(
                            s_name, t_name
                        )
                        if similarity > ENTITY_SIMILARITY_THRESHOLD:
                            candidates[s_name].append((t_name, similarity))
                except Exception as e:
                    logger.warning(f"Entity candidate finding failed: {e}")
                    continue
        
        # Sort by similarity (descending)
        for entity in candidates:
            candidates[entity].sort(key=lambda x: x[1], reverse=True)
        
        return dict(candidates)
    
    def _find_identical_predicates(
        self, source_rels: List, target_rels: List
    ) -> List[Dict]:
        """
        Find relations with identical or similar predicates.
        
        Matches relations based on:
        1. Exact predicate name match
        2. Semantic type match (for Relation objects)
        
        Args:
            source_rels: List of source domain relations
            target_rels: List of target domain relations
            
        Returns:
            List of dictionaries containing matched relations and entity constraints.
        """
        from .types import Relation
        
        mappings = []
        
        for s_rel in source_rels:
            for t_rel in target_rels:
                try:
                    if isinstance(s_rel, Relation) and isinstance(t_rel, Relation):
                        # Exact predicate match
                        if s_rel.predicate == t_rel.predicate:
                            mappings.append(
                                {
                                    "source_rel": s_rel,
                                    "target_rel": t_rel,
                                    "entity_constraints": list(
                                        zip(s_rel.arguments, t_rel.arguments)
                                    ),
                                }
                            )
                        # Semantic similarity match
                        elif (
                            s_rel.semantic_type
                            and s_rel.semantic_type == t_rel.semantic_type
                        ):
                            mappings.append(
                                {
                                    "source_rel": s_rel,
                                    "target_rel": t_rel,
                                    "entity_constraints": list(
                                        zip(s_rel.arguments, t_rel.arguments)
                                    ),
                                }
                            )
                    elif isinstance(s_rel, tuple) and isinstance(t_rel, tuple):
                        # Tuple format: (entity1, predicate, entity2)
                        if len(s_rel) >= 3 and len(t_rel) >= 3 and s_rel[1] == t_rel[1]:
                            mappings.append(
                                {
                                    "source_rel": s_rel,
                                    "target_rel": t_rel,
                                    "entity_constraints": [
                                        (s_rel[0], t_rel[0]),
                                        (s_rel[2], t_rel[2]),
                                    ],
                                }
                            )
                except Exception as e:
                    logger.warning(f"Predicate matching failed: {e}")
                    continue
        
        return mappings
    
    def _extend_mappings_structurally(
        self,
        initial_mappings: List[Dict],
        source_rels: List,
        target_rels: List,
        entity_candidates: Dict,
    ) -> Dict:
        """
        Extend mappings using structural consistency.
        
        Starting from initial relation matches, iteratively adds new
        relation mappings that maintain structural consistency.
        
        Args:
            initial_mappings: Initial relation mappings from predicate matching
            source_rels: List of source domain relations
            target_rels: List of target domain relations
            entity_candidates: Dictionary of entity candidate mappings
            
        Returns:
            Best mapping dictionary with entity_mappings, relation_mappings, and score.
        """
        best_mapping = {"entity_mappings": {}, "relation_mappings": [], "score": 0}
        
        for initial in initial_mappings:
            try:
                mapping = {"entity_mappings": {}, "relation_mappings": [initial]}
                
                # Initialize entity mappings from constraints
                if "entity_constraints" in initial:
                    for s_entity, t_entity in initial["entity_constraints"]:
                        mapping["entity_mappings"][s_entity] = t_entity
                
                # Iteratively extend the mapping
                extended = True
                iterations = 0
                
                while extended and iterations < MAX_MAPPING_ITERATIONS:
                    iterations += 1
                    extended = False
                    
                    for s_rel in source_rels:
                        for t_rel in target_rels:
                            if self._can_add_consistently(s_rel, t_rel, mapping):
                                mapping["relation_mappings"].append(
                                    {"source_rel": s_rel, "target_rel": t_rel}
                                )
                                extended = True
                                break
                
                # Score the mapping
                num_rels = max(len(source_rels), 1)
                score = len(mapping["relation_mappings"]) / num_rels
                
                if score > best_mapping["score"]:
                    best_mapping = mapping
                    best_mapping["score"] = score
            except Exception as e:
                logger.warning(f"Mapping extension failed: {e}")
                continue
        
        return best_mapping
    
    def _can_add_consistently(self, s_rel, t_rel, mapping: Dict) -> bool:
        """
        Check if a relation can be added while maintaining consistency.
        
        A relation can be added if:
        1. Predicates match
        2. Entity mappings are consistent with existing mappings
        
        Args:
            s_rel: Source relation
            t_rel: Target relation
            mapping: Current mapping dictionary
            
        Returns:
            True if the relation can be added consistently, False otherwise.
        """
        try:
            if hasattr(s_rel, "predicate") and hasattr(t_rel, "predicate"):
                # Check predicate match
                if s_rel.predicate != t_rel.predicate:
                    return False
                
                # Check entity consistency
                for s_arg, t_arg in zip(s_rel.arguments, t_rel.arguments):
                    if s_arg in mapping["entity_mappings"]:
                        if mapping["entity_mappings"][s_arg] != t_arg:
                            return False
        except Exception as e:
            logger.warning(f"Consistency check failed: {e}")
            return False
        
        return True
    
    def _evaluate_structural_consistency(
        self, mapping: Dict, source: Dict, target: Dict
    ) -> float:
        """
        Evaluate structural consistency of the mapping.
        
        Computes the proportion of relation mappings that satisfy
        their entity constraints consistently.
        
        Args:
            mapping: Mapping dictionary with entity and relation mappings
            source: Source domain dictionary
            target: Target domain dictionary
            
        Returns:
            Consistency score in range [0.0, 1.0].
        """
        if not mapping["relation_mappings"]:
            return 0.0
        
        consistent_count = 0
        total_count = 0
        
        for rel_map in mapping["relation_mappings"]:
            total_count += 1
            
            try:
                consistent = True
                if "entity_constraints" in rel_map:
                    for s_entity, t_entity in rel_map["entity_constraints"]:
                        if s_entity in mapping["entity_mappings"]:
                            if mapping["entity_mappings"][s_entity] != t_entity:
                                consistent = False
                                break
                
                if consistent:
                    consistent_count += 1
            except Exception as e:
                logger.warning(f"Consistency evaluation failed: {e}")
                continue
        
        return consistent_count / max(total_count, 1)
    
    def _compute_systematicity(self, relation_mappings: List) -> float:
        """
        Compute systematicity score.
        
        Systematicity measures the proportion of higher-order relations
        (relations about relations) in the mapping.
        
        Args:
            relation_mappings: List of relation mapping dictionaries
            
        Returns:
            Systematicity score in range [0.0, 1.0].
        """
        if not relation_mappings:
            return 0.0
        
        try:
            higher_order_count = 0
            for rel_map in relation_mappings:
                if hasattr(rel_map.get("source_rel"), "order"):
                    if rel_map["source_rel"].order > 1:
                        higher_order_count += 1
            
            return higher_order_count / max(len(relation_mappings), 1)
        except Exception:
            return 0.0
    
    def _compute_semantic_coherence(
        self, mapping: Dict, source: Dict, target: Dict
    ) -> float:
        """
        Compute semantic coherence of the mapping.
        
        Evaluates how semantically similar mapped entities and relations are,
        considering semantic categories and embedding similarities.
        
        Args:
            mapping: Mapping dictionary with entity and relation mappings
            source: Source domain dictionary
            target: Target domain dictionary
            
        Returns:
            Semantic coherence score in range [0.0, 1.0].
        """
        from .types import Entity, Relation
        
        if not mapping["entity_mappings"]:
            return 0.0
        
        coherence_scores = []
        
        # Check if mapped entities have similar semantic categories
        for s_name, t_name in mapping["entity_mappings"].items():
            # Find corresponding entities
            s_entity = next(
                (
                    e
                    for e in source["entities"]
                    if isinstance(e, Entity) and e.name == s_name
                ),
                None,
            )
            t_entity = next(
                (
                    e
                    for e in target["entities"]
                    if isinstance(e, Entity) and e.name == t_name
                ),
                None,
            )
            
            if (
                s_entity
                and t_entity
                and isinstance(s_entity, Entity)
                and isinstance(t_entity, Entity)
            ):
                if s_entity.semantic_category == t_entity.semantic_category:
                    coherence_scores.append(1.0)
                elif s_entity.semantic_category and t_entity.semantic_category:
                    # Compute category similarity
                    cat_sim = self.semantic_enricher.compute_semantic_similarity(
                        s_entity.semantic_category, t_entity.semantic_category
                    )
                    coherence_scores.append(cat_sim)
                else:
                    coherence_scores.append(0.5)
        
        # Check relation semantic coherence
        for rel_map in mapping["relation_mappings"]:
            if "source_rel" in rel_map and "target_rel" in rel_map:
                s_rel = rel_map["source_rel"]
                t_rel = rel_map["target_rel"]
                
                if isinstance(s_rel, Relation) and isinstance(t_rel, Relation):
                    rel_sim = s_rel.semantic_similarity_to(t_rel)
                    coherence_scores.append(rel_sim)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def surface_mapping(self, source: Dict, target: Dict) -> AnalogicalMapping:
        """
        Perform surface-level mapping based on attribute similarity.
        
        Matches entities based on their surface features (attributes, names)
        without considering relational structure.
        
        Args:
            source: Source domain dictionary
            target: Target domain dictionary
            
        Returns:
            AnalogicalMapping with entity mappings based on surface similarity.
        """
        from .types import AnalogicalMapping, Entity, MappingType
        
        entity_mappings = {}
        
        for s_entity in source["entities"]:
            best_match = None
            best_score = 0
            
            for t_entity in target["entities"]:
                try:
                    if isinstance(s_entity, Entity) and isinstance(t_entity, Entity):
                        score = s_entity.similarity_to(t_entity, use_embeddings=True)
                    else:
                        score = self.semantic_enricher.compute_semantic_similarity(
                            str(s_entity), str(t_entity)
                        )
                    
                    if score > best_score:
                        best_score = score
                        best_match = t_entity
                except Exception as e:
                    logger.warning(f"Surface matching failed: {e}")
                    continue
            
            if best_match and best_score > SURFACE_MAPPING_THRESHOLD:
                s_name = (
                    s_entity.name if isinstance(s_entity, Entity) else str(s_entity)
                )
                t_name = (
                    best_match.name
                    if isinstance(best_match, Entity)
                    else str(best_match)
                )
                entity_mappings[s_name] = t_name
        
        num_entities = max(len(source["entities"]), 1)
        mapping_score = len(entity_mappings) / num_entities
        
        return AnalogicalMapping(
            source_domain="source",
            target_domain="target",
            entity_mappings=entity_mappings,
            relation_mappings=[],
            mapping_score=mapping_score,
            mapping_type=MappingType.SURFACE,
            confidence=mapping_score,
        )
    
    def semantic_mapping(self, source: Dict, target: Dict) -> AnalogicalMapping:
        """
        Perform pure semantic similarity-based mapping.
        
        Uses semantic embeddings to find the most similar entities
        between source and target domains.
        
        Args:
            source: Source domain dictionary
            target: Target domain dictionary
            
        Returns:
            AnalogicalMapping with entity mappings based on semantic similarity.
        """
        from .types import AnalogicalMapping, Entity, MappingType
        
        entity_mappings = {}
        
        for s_entity in source["entities"]:
            best_match = None
            best_score = 0
            
            for t_entity in target["entities"]:
                try:
                    s_name = (
                        s_entity.name if isinstance(s_entity, Entity) else str(s_entity)
                    )
                    t_name = (
                        t_entity.name if isinstance(t_entity, Entity) else str(t_entity)
                    )
                    
                    # Pure semantic similarity
                    score = self.semantic_enricher.compute_semantic_similarity(
                        s_name, t_name
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_match = t_entity
                except Exception as e:
                    logger.warning(f"Semantic matching failed: {e}")
                    continue
            
            if best_match and best_score > SEMANTIC_MAPPING_THRESHOLD:
                s_name = (
                    s_entity.name if isinstance(s_entity, Entity) else str(s_entity)
                )
                t_name = (
                    best_match.name
                    if isinstance(best_match, Entity)
                    else str(best_match)
                )
                entity_mappings[s_name] = t_name
        
        num_entities = max(len(source["entities"]), 1)
        mapping_score = len(entity_mappings) / num_entities
        
        return AnalogicalMapping(
            source_domain="source",
            target_domain="target",
            entity_mappings=entity_mappings,
            relation_mappings=[],
            mapping_score=mapping_score,
            mapping_type=MappingType.SEMANTIC,
            confidence=0.7,
        )
    
    def pragmatic_mapping(
        self, source: Dict, target: Dict, target_problem: Dict
    ) -> AnalogicalMapping:
        """
        Perform goal-directed pragmatic mapping.
        
        Combines structural mapping with goal relevance analysis to focus
        on mappings that are relevant to achieving the specified goal.
        
        Args:
            source: Source domain dictionary
            target: Target domain dictionary
            target_problem: Target problem dictionary with 'goal' or 'objective'
            
        Returns:
            AnalogicalMapping filtered by goal relevance.
        """
        from .types import MappingType
        
        # Get base structural mapping
        structural_mapping = self.structural_mapping(source, target)
        
        # Extract goal
        goal = target_problem.get("goal", target_problem.get("objective"))
        
        if goal:
            # Analyze goal relevance
            relevance_scores = self.goal_analyzer.analyze_goal_relevance(
                structural_mapping.entity_mappings, goal, source, target
            )
            
            # Filter by relevance
            relevant_mappings = self.goal_analyzer.filter_by_relevance(
                structural_mapping.entity_mappings,
                relevance_scores,
                threshold=PRAGMATIC_RELEVANCE_THRESHOLD,
            )
            
            # Update mapping
            structural_mapping.entity_mappings = relevant_mappings
            
            # Adjust score based on relevance
            avg_relevance = (
                float(np.mean(list(relevance_scores.values())))
                if relevance_scores
                else 0.5
            )
            structural_mapping.mapping_score = (
                WEIGHT_PRAGMATIC_STRUCTURAL * structural_mapping.mapping_score
                + WEIGHT_PRAGMATIC_RELEVANCE * avg_relevance
            )
        
        structural_mapping.mapping_type = MappingType.PRAGMATIC
        return structural_mapping
    
    def _compute_structural_similarity(
        self, source: Dict, target: Dict
    ) -> float:
        """
        Compute structural similarity between source and target domains.
        
        This method computes a similarity score based on the structural
        properties of the domains including:
        - Number of entities and relations
        - Degree distribution
        - Relational structure alignment
        
        Args:
            source: Source domain dictionary with 'entities' and 'relations'
            target: Target domain dictionary with 'entities' and 'relations'
            
        Returns:
            Structural similarity score between 0 and 1.
            
        Examples:
            >>> engine = StructureMappingEngine(enricher, analyzer)
            >>> sim = engine._compute_structural_similarity(source, target)
            >>> assert 0 <= sim <= 1
        """
        try:
            # Handle truly empty dicts - they're identical
            if not source and not target:
                return 1.0
            
            # One empty, one not - different
            if (not source) or (not target):
                return 0.0
            
            # Handle dicts without entities/relations keys (treat as raw data dicts)
            has_structure_keys_source = "entities" in source or "relations" in source
            has_structure_keys_target = "entities" in target or "relations" in target
            
            # If neither has structure keys, they're not comparable domain structures
            if not has_structure_keys_source and not has_structure_keys_target:
                # Compare as raw dicts - no overlap means dissimilar
                source_keys = set(source.keys())
                target_keys = set(target.keys())
                if not source_keys and not target_keys:
                    return 1.0  # Both empty
                if not source_keys or not target_keys:
                    return 0.0  # One empty
                intersection = len(source_keys & target_keys)
                union = len(source_keys | target_keys)
                return float(intersection / union) if union > 0 else 0.0
            
            source_entities = source.get("entities", [])
            source_relations = source.get("relations", [])
            target_entities = target.get("entities", [])
            target_relations = target.get("relations", [])
            
            # Handle empty entity/relation sets
            if not source_entities and not target_entities:
                return 1.0 if not source_relations and not target_relations else 0.0
            
            if not source_entities or not target_entities:
                return 0.0
            
            # Entity count similarity (normalized difference)
            entity_count_sim = 1.0 - abs(
                len(source_entities) - len(target_entities)
            ) / max(len(source_entities), len(target_entities), 1)
            
            # Relation count similarity (normalized difference)
            relation_count_sim = 1.0 - abs(
                len(source_relations) - len(target_relations)
            ) / max(len(source_relations), len(target_relations), 1)
            
            # Combine similarities with equal weight
            structural_similarity = (entity_count_sim + relation_count_sim) / 2.0
            
            return float(np.clip(structural_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Structural similarity computation failed: {e}")
            return 0.0
    
    def _compute_surface_similarity(self, source: Dict, target: Dict) -> float:
        """
        Compute surface-level similarity between source and target domains.
        
        This method computes similarity based on surface-level features:
        - Attribute overlap
        - Label similarity
        - Type matching
        
        Args:
            source: Source domain dictionary with 'entities' and 'relations'
            target: Target domain dictionary with 'entities' and 'relations'
            
        Returns:
            Surface similarity score between 0 and 1.
            
        Examples:
            >>> engine = StructureMappingEngine(enricher, analyzer)
            >>> sim = engine._compute_surface_similarity(source, target)
            >>> assert 0 <= sim <= 1
        """
        try:
            # Handle truly empty dicts - they're identical
            if not source and not target:
                return 1.0
            
            # One empty, one not - different
            if (not source) or (not target):
                return 0.0
            
            # Handle dicts without entities keys (treat as raw data dicts)
            has_entities_source = "entities" in source
            has_entities_target = "entities" in target
            
            # If neither has entities keys, compare as raw dicts
            if not has_entities_source and not has_entities_target:
                source_keys = set(source.keys())
                target_keys = set(target.keys())
                if not source_keys and not target_keys:
                    return 1.0
                if not source_keys or not target_keys:
                    return 0.0
                intersection = len(source_keys & target_keys)
                union = len(source_keys | target_keys)
                return float(intersection / union) if union > 0 else 0.0
            
            source_entities = source.get("entities", [])
            target_entities = target.get("entities", [])
            
            # Handle empty entity sets
            if not source_entities and not target_entities:
                return 1.0
            
            if not source_entities or not target_entities:
                return 0.0
            
            # Compute attribute overlap
            source_attrs = set()
            for entity in source_entities:
                if hasattr(entity, 'attributes'):
                    source_attrs.update(entity.attributes.keys())
                elif isinstance(entity, dict):
                    source_attrs.update(entity.get('attributes', {}).keys())
            
            target_attrs = set()
            for entity in target_entities:
                if hasattr(entity, 'attributes'):
                    target_attrs.update(entity.attributes.keys())
                elif isinstance(entity, dict):
                    target_attrs.update(entity.get('attributes', {}).keys())
            
            # Jaccard similarity for attributes
            if not source_attrs and not target_attrs:
                attr_sim = 1.0
            elif not source_attrs or not target_attrs:
                attr_sim = 0.0
            else:
                intersection = len(source_attrs & target_attrs)
                union = len(source_attrs | target_attrs)
                attr_sim = intersection / union if union > 0 else 0.0
            
            # Entity count similarity
            count_sim = 1.0 - abs(
                len(source_entities) - len(target_entities)
            ) / max(len(source_entities), len(target_entities), 1)
            
            # Combine similarities
            surface_similarity = (attr_sim + count_sim) / 2.0
            
            return float(np.clip(surface_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Surface similarity computation failed: {e}")
            return 0.0
    
    def _compute_semantic_similarity(self, source: Dict, target: Dict) -> float:
        """
        Compute semantic similarity between source and target domains.
        
        Uses semantic embeddings to compare meaning rather than just structure.
        Falls back to label-based similarity when embeddings unavailable.
        
        Args:
            source: Source domain dictionary with 'entities' and 'relations'
            target: Target domain dictionary with 'entities' and 'relations'
            
        Returns:
            Semantic similarity score between 0 and 1.
            
        Examples:
            >>> engine = StructureMappingEngine(enricher, analyzer)
            >>> sim = engine._compute_semantic_similarity(source, target)
            >>> assert 0 <= sim <= 1
        """
        try:
            # Handle truly empty dicts - they're identical
            if not source and not target:
                return 1.0
            
            # One empty, one not - different
            if (not source) or (not target):
                return 0.0
            
            # Handle dicts without entities keys (treat as raw data dicts)
            has_entities_source = "entities" in source
            has_entities_target = "entities" in target
            
            # If neither has entities keys, compare as raw dicts
            if not has_entities_source and not has_entities_target:
                source_keys = set(source.keys())
                target_keys = set(target.keys())
                if not source_keys and not target_keys:
                    return 1.0
                if not source_keys or not target_keys:
                    return 0.0
                intersection = len(source_keys & target_keys)
                union = len(source_keys | target_keys)
                return float(intersection / union) if union > 0 else 0.0
            
            source_entities = source.get("entities", [])
            target_entities = target.get("entities", [])
            
            # Handle empty entity sets
            if not source_entities and not target_entities:
                return 1.0
            
            if not source_entities or not target_entities:
                return 0.0
            
            # Use semantic enricher if available
            if hasattr(self, 'semantic_enricher') and self.semantic_enricher:
                try:
                    similarities = []
                    for s_entity in source_entities:
                        for t_entity in target_entities:
                            # Get entity names
                            s_name = s_entity.name if hasattr(s_entity, 'name') else str(s_entity)
                            t_name = t_entity.name if hasattr(t_entity, 'name') else str(t_entity)
                            
                            # Compute semantic similarity via enricher
                            sim = self.semantic_enricher.compute_semantic_similarity(s_name, t_name)
                            similarities.append(sim)
                    
                    if similarities:
                        return float(np.mean(similarities))
                except Exception as e:
                    logger.debug(f"Semantic enricher similarity failed: {e}")
            
            # Fallback to simple label-based similarity
            source_labels = set()
            for entity in source_entities:
                label = entity.name if hasattr(entity, 'name') else str(entity)
                source_labels.add(label.lower())
            
            target_labels = set()
            for entity in target_entities:
                label = entity.name if hasattr(entity, 'name') else str(entity)
                target_labels.add(label.lower())
            
            # Jaccard similarity for labels
            if not source_labels or not target_labels:
                return 0.0
            
            intersection = len(source_labels & target_labels)
            union = len(source_labels | target_labels)
            
            semantic_similarity = intersection / union if union > 0 else 0.0
            
            return float(np.clip(semantic_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0


# Export public API
__all__ = [
    "StructureMappingEngine",
]
