"""
Analogical reasoning subpackage for VULCAN.

This package provides sophisticated analogical reasoning capabilities based on
Structure-Mapping Theory (Gentner, 1983) with modern semantic enrichment.

The package is organized into focused modules:
    - types: Core dataclasses and enums  
    - semantic_enricher: Semantic embedding and enrichment
    - goal_analyzer: Goal relevance analysis
    - structure_mapping: Core SMT algorithms
    - domain_parser: Domain extraction and parsing
    - base_reasoner: Abstract base classes
    - engine: Main reasoning engine
    - utils: Utility functions

Usage:
    >>> from vulcan.reasoning.analogical import AnalogicalReasoningEngine, Entity
    >>> engine = AnalogicalReasoningEngine()
    >>> entity = Entity(name="car", entity_type="vehicle")

Module: vulcan.reasoning.analogical
Author: Vulcan AI Team
"""

# Core types - always available
from .types import (
    Entity,
    Relation,
    AnalogicalMapping,
    MappingType,
    SemanticRelationType,
)

# Import other components from the original file for now
# This maintains backward compatibility while we complete the refactoring
try:
    # These will be moved to their own modules in subsequent iterations
    from ..analogical_reasoning import (
        SemanticEnricher,
        GoalRelevanceAnalyzer,
        AbstractReasoner,
        AnalogicalReasoner,
        AnalogicalReasoningEngine,
        compute_conceptual_distance,
        find_conceptual_path,
        cluster_analogies,
        explain_analogy_differences,
        test_semantic_similarity,
        NETWORKX_AVAILABLE,
        SKLEARN_AVAILABLE,
        SPACY_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        TORCH_AVAILABLE,
    )
except ImportError as e:
    # Graceful degradation if parent module has issues
    import logging
    logging.getLogger(__name__).warning(
        f"Failed to import from parent analogical_reasoning module: {e}"
    )
    SemanticEnricher = None
    GoalRelevanceAnalyzer = None
    AbstractReasoner = None
    AnalogicalReasoner = None
    AnalogicalReasoningEngine = None
    compute_conceptual_distance = None
    find_conceptual_path = None
    cluster_analogies = None
    explain_analogy_differences = None
    test_semantic_similarity = None
    NETWORKX_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    SPACY_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False


__all__ = [
    # Types
    "Entity",
    "Relation",
    "AnalogicalMapping",
    "MappingType",
    "SemanticRelationType",
    # Classes  
    "SemanticEnricher",
    "GoalRelevanceAnalyzer",
    "AbstractReasoner",
    "AnalogicalReasoner",
    "AnalogicalReasoningEngine",
    # Functions
    "compute_conceptual_distance",
    "find_conceptual_path",
    "cluster_analogies",
    "explain_analogy_differences",
    "test_semantic_similarity",
    # Availability flags
    "NETWORKX_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "SPACY_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "TORCH_AVAILABLE",
]

__version__ = "2.0.0"
__author__ = "Vulcan AI Team"
