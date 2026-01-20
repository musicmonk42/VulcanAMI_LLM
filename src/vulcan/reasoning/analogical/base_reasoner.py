"""
Base reasoner classes for analogical reasoning.

This module provides the abstract base classes and core AnalogicalReasoner
implementation that forms the foundation of the analogical reasoning system.

The AnalogicalReasoner class integrates:
- Semantic enrichment via SemanticEnricher
- Structure mapping via StructureMappingEngine
- Goal-directed analysis via GoalRelevanceAnalyzer
- Domain parsing and knowledge management
- Caching and performance optimization
- Learning from successful mappings

Classes:
    AbstractReasoner: Base class for abstract reasoning with basic abstraction/concretization
    AnalogicalReasoner: Complete analogical reasoning implementation with semantic understanding

Author: VulcanAMI Team
Date: 2026-01-10
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, List, Tuple

if TYPE_CHECKING:
    from vulcan.reasoning.analogical.types import Entity, AnalogicalMapping, MappingType

from vulcan.reasoning.analogical.semantic_enricher import (
    SemanticEnricher,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)
from vulcan.reasoning.analogical.goal_analyzer import GoalRelevanceAnalyzer
from vulcan.reasoning.analogical.structure_mapping import StructureMappingEngine
from vulcan.reasoning.analogical.types import MappingType

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Cache size limits
MAX_CACHE_SIZE: int = 1000
MAX_ANALOGY_CACHE_SIZE: int = 500
MAX_SUCCESSFUL_MAPPINGS: int = 1000

# Default weights for mapping strategies
DEFAULT_STRUCTURAL_WEIGHT: float = 0.6
DEFAULT_SURFACE_WEIGHT: float = 0.2
DEFAULT_SEMANTIC_WEIGHT: float = 0.2

# Default similarity threshold
DEFAULT_SIMILARITY_THRESHOLD: float = 0.7

# Thread pool configuration
MAX_WORKERS: int = 4


class AbstractReasoner:
    """
    Base class for abstract reasoning.
    
    Provides fundamental abstraction and concretization operations that can be
    overridden by subclasses to implement specific reasoning strategies.
    
    This is a minimal base class following the Template Method pattern, allowing
    subclasses to customize the abstraction process while maintaining a consistent
    interface.
    
    Attributes:
        abstractions: Dictionary storing concept abstractions
        
    Examples:
        >>> reasoner = AbstractReasoner()
        >>> abstract_concept = reasoner.abstract("complex_concept")
        >>> concrete = reasoner.concretize(abstract_concept)
        
    Note:
        This class is designed to be subclassed. The default implementations
        are identity functions (return the input unchanged).
    """

    def __init__(self) -> None:
        """Initialize the abstract reasoner with empty abstraction storage."""
        self.abstractions: Dict[Any, Any] = {}

    def abstract(self, concept: Any) -> Any:
        """
        Create an abstraction of a concept.
        
        Default implementation returns the concept unchanged. Subclasses should
        override this method to implement domain-specific abstraction logic.
        
        Args:
            concept: The concept to abstract (can be any type)
            
        Returns:
            The abstracted concept (same type as input by default)
            
        Examples:
            >>> reasoner = AbstractReasoner()
            >>> result = reasoner.abstract("detailed_concept")
            >>> assert result == "detailed_concept"  # Identity by default
        """
        return concept

    def concretize(self, abstraction: Any) -> Any:
        """
        Make an abstraction concrete.
        
        Default implementation returns the abstraction unchanged. Subclasses should
        override this method to implement domain-specific concretization logic.
        
        Args:
            abstraction: The abstraction to concretize (can be any type)
            
        Returns:
            The concretized concept (same type as input by default)
            
        Examples:
            >>> reasoner = AbstractReasoner()
            >>> result = reasoner.concretize("abstract_concept")
            >>> assert result == "abstract_concept"  # Identity by default
        """
        return abstraction


class AnalogicalReasoner(AbstractReasoner):
    """
    Enhanced analogical reasoning with advanced semantic understanding.
    
    This class implements Structure-Mapping Theory (Gentner, 1983) with modern
    semantic enrichment using sentence transformers. It provides a complete
    analogical reasoning system that can:
    
    - Store and manage domain knowledge
    - Find structural analogies between domains
    - Map entities based on semantic similarity and structural role
    - Learn from successful mappings to improve future performance
    - Cache results for efficiency
    - Track performance statistics
    
    The reasoner integrates three key components:
    1. SemanticEnricher: For computing semantic similarity between concepts
    2. StructureMappingEngine: For finding structural correspondences
    3. GoalRelevanceAnalyzer: For goal-directed analogical reasoning
    
    Attributes:
        semantic_enricher: SemanticEnricher instance for computing embeddings
        goal_analyzer: GoalRelevanceAnalyzer for relevance scoring
        structure_mapper: StructureMappingEngine for structural mapping
        domain_knowledge: Storage for domain structures
        analogy_cache: Cache of computed analogies
        mapping_cache: Cache of entity mappings
        successful_mappings: Recent successful mappings for learning
        enable_caching: Whether to use caching
        enable_learning: Whether to learn from mappings
        stats: Performance tracking statistics
        
    Examples:
        >>> reasoner = AnalogicalReasoner(enable_caching=True)
        >>> reasoner.add_domain("biology", {"entities": [...], "relations": [...]})
        >>> mapping = reasoner.find_structural_analogy("software", "biology")
        >>> print(f"Confidence: {mapping.confidence:.2f}")
        
    References:
        Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
        Cognitive Science, 7(2), 155-170.
    """

    # Semantic similarity threshold for entity matching
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.3
    
    # Epsilon for numerical stability in similarity computations
    COSINE_SIMILARITY_EPSILON: float = 1e-8

    def __init__(
        self, 
        enable_caching: bool = True, 
        enable_learning: bool = True
    ) -> None:
        """
        Initialize the analogical reasoner with all components.
        
        Args:
            enable_caching: If True, cache computed analogies for performance
            enable_learning: If True, learn from successful mappings to improve weights
            
        Note:
            Initialization creates several caches and instantiates heavy components
            (SemanticEnricher with embedding model). Consider reusing instances rather
            than creating new ones frequently.
        """
        super().__init__()

        # ===================================================================
        # Core Components
        # ===================================================================
        
        # Semantic enrichment (uses singleton pattern for embedding model)
        self.semantic_enricher = SemanticEnricher()
        
        # Goal-directed analysis
        self.goal_analyzer = GoalRelevanceAnalyzer(self.semantic_enricher)
        
        # Structure mapping engine
        self.structure_mapper = StructureMappingEngine(
            self.semantic_enricher,
            self.goal_analyzer
        )

        # ===================================================================
        # Domain Knowledge Storage
        # ===================================================================
        
        self.domain_knowledge: Dict[str, Any] = {}
        self.domain_graphs: Dict[str, Any] = {}

        # ===================================================================
        # Caching System
        # ===================================================================
        
        self.enable_caching = enable_caching
        self.analogy_cache: Dict[str, Any] = {}
        self.mapping_cache: Dict[str, Any] = {}
        self.successful_mappings: deque = deque(maxlen=MAX_SUCCESSFUL_MAPPINGS)
        
        # Cache size limits
        self.max_cache_size = MAX_CACHE_SIZE
        self.max_analogy_cache_size = MAX_ANALOGY_CACHE_SIZE

        # ===================================================================
        # Mapping Configuration
        # ===================================================================
        
        # Similarity threshold for considering a match
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        
        # Weights for different mapping strategies
        self.structural_weight = DEFAULT_STRUCTURAL_WEIGHT
        self.surface_weight = DEFAULT_SURFACE_WEIGHT
        self.semantic_weight = DEFAULT_SEMANTIC_WEIGHT

        # ===================================================================
        # Learning System
        # ===================================================================
        
        self.enable_learning = enable_learning
        if enable_learning:
            self.mapping_patterns: defaultdict = defaultdict(int)
            self.domain_similarities: defaultdict = defaultdict(float)
            self.learned_weights: Dict[str, float] = {
                "structural": DEFAULT_STRUCTURAL_WEIGHT,
                "surface": DEFAULT_SURFACE_WEIGHT,
                "semantic": DEFAULT_SEMANTIC_WEIGHT,
            }

        # ===================================================================
        # Performance Tracking
        # ===================================================================
        
        self.stats: Dict[str, Any] = {
            "total_mappings": 0,
            "successful_mappings": 0,
            "cache_hits": 0,
            "average_mapping_score": 0.0,
            "semantic_enrichments": 0,
            "embedding_method": (
                "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "tfidf"
            ),
            "num_domains": 0,
        }

        # ===================================================================
        # Parallel Processing
        # ===================================================================
        
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # ===================================================================
        # Persistence
        # ===================================================================
        
        self.model_path = Path("analogical_models")
        self.model_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized AnalogicalReasoner with semantic understanding")
        logger.info(f"Embedding method: {self.stats['embedding_method']}")

    def add_domain(self, domain_name: str, structure: Dict[str, Any]) -> None:
        """
        Add domain knowledge with semantic enrichment.
        
        Processes and stores a domain structure, enriching entities with semantic
        embeddings for improved matching. The structure can contain entities,
        relations, and attributes.
        
        Args:
            domain_name: Unique identifier for the domain
            structure: Domain structure containing:
                - entities: List of Entity objects or entity names
                - relations: List of Relation objects or tuples
                - attributes: Optional dict of entity attributes
                
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> structure = {
            ...     "entities": ["node", "edge", "graph"],
            ...     "relations": [("node", "connects_to", "edge")],
            ...     "attributes": {"node": ["has_data", "has_neighbors"]}
            ... }
            >>> reasoner.add_domain("graph_theory", structure)
            >>> assert "graph_theory" in reasoner.domain_knowledge
            
        Note:
            Entity enrichment with embeddings can take 100-500ms depending on the
            number of entities and whether the embedding model is already loaded.
        """
        from vulcan.reasoning.analogical.types import Entity
        from vulcan.reasoning.analogical.domain_parser import (
            extract_entities,
            extract_relations,
            extract_attributes,
        )

        # Parse structure using domain_parser functions
        entities = extract_entities(structure)
        relations = extract_relations(structure)
        attributes = extract_attributes(structure)

        # Enrich entities semantically
        enriched_entities: Set[Entity] = set()
        for entity in entities:
            if isinstance(entity, Entity):
                enriched = self.semantic_enricher.enrich_entity(entity)
                enriched_entities.add(enriched)
            else:
                # Convert string to Entity and enrich
                entity_obj = Entity(name=str(entity), entity_type="concept")
                enriched = self.semantic_enricher.enrich_entity(entity_obj)
                enriched_entities.add(enriched)

        # Store enriched domain knowledge
        self.domain_knowledge[domain_name] = {
            "entities": enriched_entities,
            "relations": relations,
            "attributes": attributes,
            "structure": structure,
        }
        
        # Update statistics
        self.stats["num_domains"] = len(self.domain_knowledge)

        # Build graph representation for structural analysis
        from vulcan.reasoning.analogical.domain_parser import build_domain_graph
        
        self.domain_graphs[domain_name] = build_domain_graph(
            enriched_entities, relations
        )

        logger.info(
            f"Added domain '{domain_name}' with {len(enriched_entities)} "
            f"enriched entities and {len(relations)} relations"
        )

    def compute_similarity(self, source: Dict, target: Dict) -> float:
        """
        Compute overall similarity between two domain structures.
        
        Uses a weighted combination of:
        - Structural similarity (graph structure)
        - Surface similarity (attribute matching)
        - Semantic similarity (embedding cosine similarity)
        
        Args:
            source: Source domain structure
            target: Target domain structure
            
        Returns:
            Similarity score in range [0.0, 1.0]
            
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> source = {"entities": ["A", "B"], "relations": [("A", "r", "B")]}
            >>> target = {"entities": ["X", "Y"], "relations": [("X", "s", "Y")]}
            >>> similarity = reasoner.compute_similarity(source, target)
            >>> assert 0.0 <= similarity <= 1.0
        """
        # Delegate to structure mapper
        structural_sim = self.structure_mapper._compute_structural_similarity(
            source, target
        )
        surface_sim = self.structure_mapper._compute_surface_similarity(source, target)
        semantic_sim = self.structure_mapper._compute_semantic_similarity(
            source, target
        )

        # Weighted combination
        weights = self.learned_weights if self.enable_learning else {
            "structural": self.structural_weight,
            "surface": self.surface_weight,
            "semantic": self.semantic_weight,
        }

        total_similarity = (
            weights["structural"] * structural_sim
            + weights["surface"] * surface_sim
            + weights["semantic"] * semantic_sim
        )

        return total_similarity

    def find_structural_analogy(
        self,
        source_domain: Any,
        target_problem: Any,
        mapping_type: Optional[MappingType] = None,
    ) -> Dict[str, Any]:
        """
        Find structural analogy between source domain and target problem.
        
        This is the main entry point for analogical reasoning. It delegates to
        the StructureMappingEngine to perform the actual mapping based on the
        specified strategy.
        
        Args:
            source_domain: Source domain name or structure
            target_problem: Target problem description or structure
            mapping_type: Type of mapping to perform (structural, surface, semantic, pragmatic)
            
        Returns:
            Dict containing:
                - found: Whether an analogy was found
                - mapping: AnalogicalMapping object with entity correspondences
                - confidence: Confidence score [0.0, 1.0]
                - explanation: Human-readable explanation
                
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> reasoner.add_domain("software", {...})
            >>> reasoner.add_domain("biology", {...})
            >>> result = reasoner.find_structural_analogy("software", "biology")
            >>> if result["found"]:
            ...     print(f"Confidence: {result['confidence']:.2f}")
        """
        from vulcan.reasoning.analogical.types import MappingType as MT
        
        # Default to structural mapping
        if mapping_type is None:
            mapping_type = MT.STRUCTURAL

        # Check cache first
        if self.enable_caching:
            cache_key = f"{source_domain}:{target_problem}:{mapping_type.value}"
            if cache_key in self.analogy_cache:
                self.stats["cache_hits"] += 1
                self.stats["total_mappings"] += 1  # Count cache hits as mappings
                return self.analogy_cache[cache_key]

        # Validate source domain
        if isinstance(source_domain, str) and source_domain not in self.domain_knowledge:
            # Check if it's a description vs unknown domain
            if len(source_domain) < 50:  # Likely a domain name
                return {
                    "found": False,
                    "mapping": None,
                    "confidence": 0.0,
                    "explanation": f"Unknown source domain: {source_domain}",
                    "reason": f"Unknown source domain: {source_domain}",
                    "mapping_type": mapping_type.value,
                    "score": 0.0,
                    "mappings": {},
                    "solution": None,
                }

        # Prepare source and target structures
        source_struct = self._prepare_structure(source_domain)
        target_struct = self._prepare_structure(target_problem)

        # Delegate to structure mapper
        if mapping_type == MT.STRUCTURAL:
            mapping = self.structure_mapper.structural_mapping(source_struct, target_struct)
        elif mapping_type == MT.SURFACE:
            mapping = self.structure_mapper.surface_mapping(source_struct, target_struct)
        elif mapping_type == MT.SEMANTIC:
            mapping = self.structure_mapper.semantic_mapping(source_struct, target_struct)
        elif mapping_type == MT.PRAGMATIC:
            # Pragmatic mapping requires goal context
            goal_text = target_struct.get("goal", "")
            mapping = self.structure_mapper.pragmatic_mapping(
                source_struct, target_struct, goal_text
            )
        else:
            mapping = self.structure_mapper.structural_mapping(source_struct, target_struct)

        # Build result with backward compatibility
        result = {
            "found": len(mapping.entity_mappings) > 0,
            "mapping": mapping,
            "confidence": mapping.confidence,
            "explanation": self._generate_explanation(mapping),
            "mapping_type": mapping_type.value,
            # Backward compatibility fields
            "score": mapping.mapping_score,
            "mappings": mapping.entity_mappings,
            "solution": mapping.entity_mappings if len(mapping.entity_mappings) > 0 else None,
            "reason": "No analogical mapping found." if len(mapping.entity_mappings) == 0 else None,
        }

        # Update statistics
        self.stats["total_mappings"] += 1
        if result["found"]:
            self.stats["successful_mappings"] += 1
            self.successful_mappings.append(mapping)

        # Cache result
        if self.enable_caching:
            if len(self.analogy_cache) >= self.max_analogy_cache_size:
                # Evict oldest entry
                self.analogy_cache.pop(next(iter(self.analogy_cache)))
            self.analogy_cache[cache_key] = result

        return result

    def _prepare_structure(self, domain: Any) -> Dict[str, Any]:
        """
        Prepare a domain structure for mapping.
        
        Converts various input formats (string, dict, domain name) into a
        standardized structure dict suitable for mapping.
        
        Args:
            domain: Domain name, structure dict, or description string
            
        Returns:
            Standardized structure dict with entities, relations, and attributes
        """
        if isinstance(domain, str):
            # Check if it's a known domain name
            if domain in self.domain_knowledge:
                struct = self.domain_knowledge[domain]["structure"]
            else:
                # Parse as natural language description
                from vulcan.reasoning.analogical.domain_parser import (
                    extract_domain_structure,
                )
                struct = extract_domain_structure(domain, domain)
        elif isinstance(domain, dict):
            struct = domain
        else:
            struct = {}
        
        # Ensure all required keys are present
        if "entities" not in struct:
            struct["entities"] = []
        if "relations" not in struct:
            struct["relations"] = []
        if "attributes" not in struct:
            struct["attributes"] = {}
            
        return struct

    def _structural_mapping(
        self, 
        source: Dict, 
        target: Dict
    ) -> AnalogicalMapping:
        """
        Perform structural mapping between source and target (compatibility wrapper).
        
        This method provides backward compatibility for code that expects
        _structural_mapping to be a method on the reasoner.
        
        Args:
            source: Source domain structure
            target: Target domain structure
            
        Returns:
            AnalogicalMapping object
        """
        return self.structure_mapper.structural_mapping(source, target)

    def _surface_mapping(
        self,
        source: Dict,
        target: Dict
    ) -> AnalogicalMapping:
        """
        Perform surface mapping between source and target (compatibility wrapper).
        
        Args:
            source: Source domain structure
            target: Target domain structure
            
        Returns:
            AnalogicalMapping object
        """
        return self.structure_mapper.surface_mapping(source, target)

    def _generate_explanation(self, mapping: AnalogicalMapping) -> str:
        """
        Generate human-readable explanation of the mapping.
        
        Args:
            mapping: AnalogicalMapping object
            
        Returns:
            Explanation string describing the analogy
        """
        if not mapping.entity_mappings:
            return "No analogical mapping found."

        explanation_parts = [
            f"Found {len(mapping.entity_mappings)} entity correspondences:",
        ]

        # Show up to 5 most confident mappings
        sorted_mappings = sorted(
            mapping.entity_mappings.items(),
            key=lambda x: mapping.confidence,
            reverse=True,
        )[:5]

        for source_entity, target_entity in sorted_mappings:
            explanation_parts.append(f"  - {source_entity} ↔ {target_entity}")

        if mapping.structural_depth > 0:
            explanation_parts.append(
                f"\nStructural depth: {mapping.structural_depth}"
            )

        explanation_parts.append(f"\nOverall confidence: {mapping.confidence:.2f}")

        return "\n".join(explanation_parts)

    def update_cache(
        self, source: Any, target: Any, mapping: Dict, confidence: float
    ) -> None:
        """
        Update mapping cache with a new result.
        
        Args:
            source: Source domain
            target: Target domain
            mapping: Entity mapping dictionary
            confidence: Confidence score
        """
        if not self.enable_caching:
            return

        cache_key = f"{source}:{target}"
        
        # Check cache size limit
        if len(self.mapping_cache) >= self.max_cache_size:
            # Evict oldest entry (FIFO)
            self.mapping_cache.pop(next(iter(self.mapping_cache)))

        self.mapping_cache[cache_key] = {
            "mapping": mapping,
            "confidence": confidence,
        }

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def find_multiple_analogies(
        self,
        target_problem: Any,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top-k analogies for a target problem from stored domains.
        
        Args:
            target_problem: Target problem description or structure
            k: Number of analogies to return (default: 5)
            
        Returns:
            List of dicts with domain, mapping, confidence, and explanation
        """
        target_struct = self._prepare_structure(target_problem)
        
        results = []
        for domain_name in self.domain_knowledge:
            source_struct = self.domain_knowledge[domain_name]["structure"]
            
            # Find analogy
            result = self.find_structural_analogy(
                domain_name,
                target_struct,
                MappingType.STRUCTURAL
            )
            
            if result.get("found"):
                results.append({
                    "domain": domain_name,
                    "mapping": result.get("mapping"),
                    "confidence": result.get("confidence", 0.0),
                    "explanation": result.get("explanation", ""),
                })
        
        # Sort by confidence and return top k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:k]

    def cross_domain_transfer(
        self,
        source_domain: str,
        target_domain: str,
        concept: str
    ) -> Dict[str, Any]:
        """
        Transfer a concept from source domain to target domain via analogy.
        
        Args:
            source_domain: Name of source domain
            target_domain: Name of target domain
            concept: Concept to transfer
            
        Returns:
            Dict with transferred concept and explanation
        """
        # Find structural analogy
        result = self.find_structural_analogy(
            source_domain,
            target_domain,
            MappingType.STRUCTURAL
        )
        
        if not result.get("found"):
            return {
                "success": False,
                "reason": "No analogy found between domains",
            }
        
        mapping = result["mapping"].entity_mappings
        
        # Find the target concept
        if concept in mapping:
            transferred = mapping[concept]
            return {
                "success": True,
                "source_concept": concept,
                "target_concept": transferred,
                "explanation": f"'{concept}' in {source_domain} corresponds to '{transferred}' in {target_domain}",
            }
        else:
            return {
                "success": False,
                "reason": f"Concept '{concept}' not found in mapping",
            }

    def _learn_from_mapping(
        self, 
        source_domain: str, 
        mapping: AnalogicalMapping
    ) -> None:
        """
        Learn from a successful mapping to improve future mappings.
        
        Args:
            source_domain: Source domain name
            mapping: Analogical mapping to learn from
        """
        if not self.enable_learning:
            return
        
        # Track successful mapping patterns
        pattern_key = f"{source_domain}:{mapping.target_domain}"
        self.mapping_patterns[pattern_key] += 1
        
        # Update domain similarities
        self.domain_similarities[pattern_key] = mapping.confidence

    def save_model(self, name: str = "default") -> None:
        """
        Save the reasoner model to disk.
        
        Args:
            name: Model name (default: "default")
        """
        import pickle
        
        model_file = self.model_path / f"{name}.pkl"
        
        # Save key components
        model_data = {
            "domain_knowledge": self.domain_knowledge,
            "mapping_patterns": dict(self.mapping_patterns) if self.enable_learning else {},
            "domain_similarities": dict(self.domain_similarities) if self.enable_learning else {},
            "stats": self.stats,
        }
        
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {model_file}")

    def load_model(self, name: str = "default") -> None:
        """
        Load a saved reasoner model from disk.
        
        Args:
            name: Model name (default: "default")
        """
        import pickle
        
        model_file = self.model_path / f"{name}.pkl"
        
        if not model_file.exists():
            logger.warning(f"Model file {model_file} does not exist")
            return
        
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
        
        # Restore components
        self.domain_knowledge = model_data.get("domain_knowledge", {})
        if self.enable_learning:
            self.mapping_patterns = defaultdict(int, model_data.get("mapping_patterns", {}))
            self.domain_similarities = defaultdict(float, model_data.get("domain_similarities", {}))
        self.stats = model_data.get("stats", self.stats)
        
        logger.info(f"Loaded model from {model_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict with statistics
        """
        return self.stats.copy()

    # Compatibility methods for compute_similarity components
    def _compute_structural_similarity(self, source: Dict, target: Dict) -> float:
        """Compute structural similarity (compatibility wrapper)."""
        return self.structure_mapper._compute_structural_similarity(source, target)
    
    def _compute_surface_similarity(self, source: Dict, target: Dict) -> float:
        """Compute surface similarity (compatibility wrapper)."""
        return self.structure_mapper._compute_surface_similarity(source, target)
    
    def _compute_semantic_similarity(self, source: Dict, target: Dict) -> float:
        """Compute semantic similarity (compatibility wrapper)."""
        return self.structure_mapper._compute_semantic_similarity(source, target)

    # =========================================================================
    # ISSUE #7 FIX: Complete Analogical Mapping for Batch Queries
    # =========================================================================
    # These methods implement complete mapping for multiple concepts, ensuring
    # all requested mappings are processed rather than returning early after
    # the first match. This follows industry-standard practices from:
    # - Structure-Mapping Engine (SME) by Forbus et al.
    # - Analogical Retrieval by MAC/FAC (Forbus, Gentner, Law, 1995)
    # =========================================================================

    def extract_mapping_targets(self, query: str) -> List[str]:
        """
        Extract all concepts user wants mapped from a natural language query.
        
        ISSUE #7 FIX: Parses queries to extract ALL requested mapping targets,
        not just the first one. Supports multiple query formats:
        
        - Comma-separated lists: "Map A, B, C to domain X"
        - Numbered lists: "1. concept A\\n2. concept B\\n3. concept C"
        - Arrow notation: "A → ?\\nB → ?\\nC → ?"
        - Bullet points: "• concept A\\n• concept B"
        - Question format: "What corresponds to A, B, and C?"
        
        Args:
            query: Natural language query requesting concept mappings
            
        Returns:
            List of concept strings to be mapped
            
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> concepts = reasoner.extract_mapping_targets(
            ...     "Map leader election, quorum, fencing token to biology"
            ... )
            >>> assert concepts == ["leader election", "quorum", "fencing token"]
            
            >>> concepts = reasoner.extract_mapping_targets(
            ...     "1. Leader election → ?\\n2. Quorum → ?\\n3. Split brain → ?"
            ... )
            >>> assert len(concepts) == 3
        """
        concepts = []
        
        # Normalize whitespace
        query = query.strip()
        query_lower = query.lower()
        
        # Pattern 1: "Map X, Y, Z to domain" or "Map X, Y and Z to domain"
        # Domain names can contain spaces, hyphens, underscores (e.g., "cellular biology")
        map_pattern = r'map\s+(.+?)\s+(?:to|in|into|onto)\s+[\w\s\-_]+'
        match = re.search(map_pattern, query_lower, re.IGNORECASE)
        if match:
            concepts_str = match.group(1)
            # Split by commas, "and", and semicolons
            parts = re.split(r'[,;]|\s+and\s+', concepts_str)
            concepts = [p.strip() for p in parts if p.strip()]
            if concepts:
                return concepts
        
        # Pattern 2: Numbered list "1. concept\n2. concept\n..."
        numbered_pattern = r'^\s*\d+[.)]\s*(.+?)(?:\s*→\s*\?|\s*$)'
        numbered_matches = re.findall(numbered_pattern, query, re.MULTILINE)
        if numbered_matches:
            concepts = [m.strip() for m in numbered_matches if m.strip()]
            if concepts:
                return concepts
        
        # Pattern 3: Arrow notation "concept → ?"
        arrow_pattern = r'([^→\n]+?)\s*→\s*\?'
        arrow_matches = re.findall(arrow_pattern, query)
        if arrow_matches:
            concepts = [m.strip() for m in arrow_matches if m.strip()]
            if concepts:
                return concepts
        
        # Pattern 4: Bullet points "• concept" or "- concept"
        bullet_pattern = r'^[\s]*[•\-\*]\s*(.+?)$'
        bullet_matches = re.findall(bullet_pattern, query, re.MULTILINE)
        if bullet_matches:
            concepts = [m.strip() for m in bullet_matches if m.strip()]
            if concepts:
                return concepts
        
        # Pattern 5: "What corresponds to X, Y, Z" or "How does X, Y, Z map"
        correspond_pattern = r'(?:corresponds?\s+to|map(?:s|ped)?(?:\s+to)?|equivalent\s+to)\s+(.+?)(?:\?|$|\s+in\s+)'
        correspond_match = re.search(correspond_pattern, query_lower, re.IGNORECASE)
        if correspond_match:
            concepts_str = correspond_match.group(1)
            parts = re.split(r'[,;]|\s+and\s+', concepts_str)
            concepts = [p.strip() for p in parts if p.strip()]
            if concepts:
                return concepts
        
        # Pattern 6: Colon-separated concepts "Concepts: A, B, C"
        colon_pattern = r'concepts?\s*:\s*(.+?)(?:\s+to\s+|\s+in\s+|\?|$)'
        colon_match = re.search(colon_pattern, query_lower, re.IGNORECASE)
        if colon_match:
            concepts_str = colon_match.group(1)
            parts = re.split(r'[,;]|\s+and\s+', concepts_str)
            concepts = [p.strip() for p in parts if p.strip()]
            if concepts:
                return concepts
        
        # Fallback: No structured pattern found, return empty list
        logger.debug(f"Could not extract mapping targets from query: {query[:100]}...")
        return []

    def map_all_concepts(
        self,
        source_concepts: List[str],
        target_domain: str,
        mapping_type: Optional[MappingType] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Map ALL source concepts to target domain, ensuring complete mapping.
        
        ISSUE #7 FIX: Maps each concept individually to ensure no concepts are
        missed, unlike the previous behavior which could return early after
        finding the first match.
        
        Args:
            source_concepts: List of concepts to map from source domain
            target_domain: Name of target domain to map to
            mapping_type: Type of mapping to use (default: STRUCTURAL)
            
        Returns:
            Dict mapping each source concept to its mapping result (or None if not found)
            Each result contains:
                - target_concept: The mapped concept in target domain
                - confidence: Confidence score for this mapping
                - explanation: Human-readable explanation
                - found: Whether a mapping was found
                
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> reasoner.add_domain("distributed_systems", {...})
            >>> reasoner.add_domain("biology", {...})
            >>> results = reasoner.map_all_concepts(
            ...     ["leader election", "quorum", "split brain"],
            ...     "biology"
            ... )
            >>> for concept, result in results.items():
            ...     if result and result["found"]:
            ...         print(f"{concept} → {result['target_concept']}")
        """
        from vulcan.reasoning.analogical.types import MappingType as MT
        
        if mapping_type is None:
            mapping_type = MT.STRUCTURAL
        
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        
        for concept in source_concepts:
            try:
                # Create a minimal source structure for this concept
                source_struct = {
                    "name": concept,
                    "entities": [{"name": concept, "type": "concept", "attributes": []}],
                    "relations": [],
                    "attributes": [concept],
                }
                
                # Find mapping to target domain
                mapping_result = self.find_structural_analogy(
                    source_struct,
                    target_domain,
                    mapping_type
                )
                
                if mapping_result.get("found") and mapping_result.get("mapping"):
                    # Extract the best target mapping for this concept
                    entity_mappings = mapping_result["mapping"].entity_mappings
                    
                    # Find the target concept that corresponds to our source
                    target_concept = None
                    if concept in entity_mappings:
                        target_concept = entity_mappings[concept]
                    elif entity_mappings:
                        # Use first available mapping as approximation
                        target_concept = list(entity_mappings.values())[0]
                    
                    results[concept] = {
                        "found": target_concept is not None,
                        "target_concept": target_concept,
                        "confidence": mapping_result.get("confidence", 0.0),
                        "explanation": mapping_result.get("explanation", ""),
                        "mapping": entity_mappings,
                    }
                else:
                    # No mapping found
                    results[concept] = {
                        "found": False,
                        "target_concept": None,
                        "confidence": 0.0,
                        "explanation": f"No mapping found for '{concept}' in {target_domain}",
                        "mapping": {},
                    }
                    logger.info(
                        f"[AnalogicalReasoner] Could not find mapping for "
                        f"'{concept}' in domain '{target_domain}'"
                    )
                    
            except Exception as e:
                logger.error(f"Error mapping concept '{concept}': {e}")
                results[concept] = {
                    "found": False,
                    "target_concept": None,
                    "confidence": 0.0,
                    "explanation": f"Error during mapping: {str(e)}",
                    "mapping": {},
                }
        
        return results

    def check_mapping_completeness(
        self,
        requested: List[str],
        mapped: Dict[str, Optional[Dict[str, Any]]]
    ) -> Tuple[bool, List[str], float]:
        """
        Verify that all requested concepts were successfully mapped.
        
        ISSUE #7 FIX: Validates mapping completeness and reports which concepts
        could not be mapped, enabling proper user feedback.
        
        Args:
            requested: List of concepts that were requested to be mapped
            mapped: Dict of mapping results from map_all_concepts()
            
        Returns:
            Tuple of (is_complete, unmapped_concepts, completeness_ratio):
                - is_complete: True if all concepts were mapped
                - unmapped_concepts: List of concepts that couldn't be mapped
                - completeness_ratio: Fraction of concepts successfully mapped (0.0 to 1.0)
                
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> requested = ["A", "B", "C", "D", "E"]
            >>> mapped = {"A": {..., "found": True}, "B": {..., "found": True}, 
            ...           "C": None, "D": {..., "found": False}, "E": {..., "found": True}}
            >>> is_complete, unmapped, ratio = reasoner.check_mapping_completeness(requested, mapped)
            >>> assert not is_complete
            >>> assert set(unmapped) == {"C", "D"}
            >>> assert ratio == 0.6  # 3 out of 5 mapped
        """
        unmapped = []
        successful_count = 0
        
        for concept in requested:
            result = mapped.get(concept)
            if result is None:
                unmapped.append(concept)
            elif not result.get("found", False):
                unmapped.append(concept)
            else:
                successful_count += 1
        
        is_complete = len(unmapped) == 0
        completeness_ratio = successful_count / len(requested) if requested else 1.0
        
        if not is_complete:
            logger.warning(
                f"Incomplete mapping: {len(unmapped)}/{len(requested)} concepts unmapped: "
                f"{unmapped}"
            )
        
        return is_complete, unmapped, completeness_ratio

    def format_mapping_response(
        self,
        mappings: Dict[str, Optional[Dict[str, Any]]],
        source_domain: str,
        target_domain: str,
        unmapped: List[str] = None
    ) -> str:
        """
        Format complete mapping results as human-readable text.
        
        ISSUE #7 FIX: Creates a comprehensive response showing ALL requested
        mappings, including those that couldn't be found, ensuring users
        receive complete feedback on their query.
        
        Args:
            mappings: Dict of mapping results from map_all_concepts()
            source_domain: Name of source domain
            target_domain: Name of target domain
            unmapped: Optional list of unmapped concepts (for explicit reporting)
            
        Returns:
            Human-readable string with complete mapping results
            
        Examples:
            >>> response = reasoner.format_mapping_response(
            ...     {"A": {"found": True, "target_concept": "X", "confidence": 0.85},
            ...      "B": {"found": False, "target_concept": None, "confidence": 0.0}},
            ...     "distributed_systems",
            ...     "biology"
            ... )
            >>> print(response)
            Analogical Mappings: distributed_systems → biology
            ============================================================
            
            1. A → X (confidence: 0.85)
            2. B → (no mapping found)
            
            ============================================================
            Summary: 1/2 concepts successfully mapped (50.0%)
        """
        lines = [
            f"Analogical Mappings: {source_domain} → {target_domain}",
            "=" * 60,
            "",
        ]
        
        successful_count = 0
        total_count = len(mappings)
        total_confidence = 0.0
        
        for i, (concept, result) in enumerate(mappings.items(), 1):
            if result and result.get("found"):
                target = result.get("target_concept", "unknown")
                confidence = result.get("confidence", 0.0)
                lines.append(f"{i}. {concept} → {target} (confidence: {confidence:.2f})")
                successful_count += 1
                total_confidence += confidence
                
                # Include explanation if available
                explanation = result.get("explanation", "")
                if explanation and len(explanation) < 200:
                    # Indent explanation on next line
                    lines.append(f"   Rationale: {explanation[:150]}")
            else:
                lines.append(f"{i}. {concept} → (no mapping found)")
                if result and result.get("explanation"):
                    lines.append(f"   Note: {result['explanation'][:100]}")
        
        # Add summary
        lines.append("")
        lines.append("=" * 60)
        
        completeness = (successful_count / total_count * 100) if total_count > 0 else 0
        avg_confidence = (total_confidence / successful_count) if successful_count > 0 else 0
        
        lines.append(
            f"Summary: {successful_count}/{total_count} concepts successfully mapped "
            f"({completeness:.1f}%)"
        )
        
        if successful_count > 0:
            lines.append(f"Average confidence: {avg_confidence:.2f}")
        
        # Note unmapped concepts explicitly
        if unmapped:
            lines.append("")
            lines.append(f"Note: Could not find mappings for: {', '.join(unmapped)}")
            lines.append(
                "Consider providing more context about these concepts or checking "
                "if they exist in the source domain."
            )
        
        return "\n".join(lines)

    def reason_with_complete_mapping(
        self,
        query: str,
        source_domain: str,
        target_domain: str,
        mapping_type: Optional[MappingType] = None
    ) -> Dict[str, Any]:
        """
        Perform analogical reasoning with complete mapping of all concepts.
        
        ISSUE #7 FIX: This is the main entry point for batch analogical reasoning
        that ensures ALL concepts in a query are mapped, not just the first one.
        
        This method:
        1. Extracts all mapping targets from the query
        2. Maps each concept individually
        3. Validates completeness
        4. Returns formatted results with all mappings
        
        Args:
            query: Natural language query with concepts to map
            source_domain: Name of source domain (must be added via add_domain)
            target_domain: Name of target domain (must be added via add_domain)
            mapping_type: Type of mapping to use (default: STRUCTURAL)
            
        Returns:
            Dict containing:
                - success: Whether the operation completed
                - complete: Whether all concepts were mapped
                - mappings: Dict of concept → result
                - unmapped: List of concepts that couldn't be mapped
                - completeness_ratio: Fraction of concepts mapped
                - formatted_response: Human-readable response string
                - concepts_found: Number of concepts extracted from query
                
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> reasoner.add_domain("distributed_systems", {...})
            >>> reasoner.add_domain("biology", {...})
            >>> result = reasoner.reason_with_complete_mapping(
            ...     "Map leader election, quorum, fencing token, split brain, "
            ...     "write divergence to biology",
            ...     "distributed_systems",
            ...     "biology"
            ... )
            >>> if result["success"]:
            ...     print(result["formatted_response"])
            ...     if not result["complete"]:
            ...         print(f"Unmapped: {result['unmapped']}")
        """
        from vulcan.reasoning.analogical.types import MappingType as MT
        
        if mapping_type is None:
            mapping_type = MT.STRUCTURAL
        
        # Step 1: Extract all concepts to map
        concepts = self.extract_mapping_targets(query)
        
        if not concepts:
            # Fallback: try to extract noun phrases or key terms from query
            # This is better than using the entire query as a single concept
            logger.info(
                f"No structured concepts found in query, attempting keyword extraction"
            )
            # Simple keyword extraction: split on common delimiters and filter
            # This extracts potential concept terms rather than the whole query
            potential_terms = re.split(r'[,;:?!.\n]+', query)
            potential_terms = [t.strip() for t in potential_terms if t.strip() and len(t.strip()) > 2]
            
            if potential_terms:
                # Filter out common stop words and very short terms
                stop_words = {'map', 'to', 'from', 'the', 'a', 'an', 'in', 'on', 'for', 'and', 'or', 'what', 'how', 'is', 'are'}
                concepts = [t for t in potential_terms if t.lower() not in stop_words and len(t) > 3]
            
            if not concepts:
                # Last resort: use the cleaned query, but limit length
                clean_query = query.strip()[:100]  # Limit to 100 chars
                concepts = [clean_query] if clean_query else []
        
        logger.info(
            f"[AnalogicalReasoner] Extracted {len(concepts)} concepts to map: {concepts}"
        )
        
        # Step 2: Map all concepts
        mappings = self.map_all_concepts(concepts, target_domain, mapping_type)
        
        # Step 3: Check completeness
        is_complete, unmapped, completeness_ratio = self.check_mapping_completeness(
            concepts, mappings
        )
        
        # Step 4: Format response
        formatted_response = self.format_mapping_response(
            mappings, source_domain, target_domain, unmapped
        )
        
        return {
            "success": True,
            "complete": is_complete,
            "mappings": mappings,
            "unmapped": unmapped,
            "completeness_ratio": completeness_ratio,
            "formatted_response": formatted_response,
            "concepts_found": len(concepts),
            "concepts_mapped": len(concepts) - len(unmapped),
            "source_domain": source_domain,
            "target_domain": target_domain,
        }

    # =========================================================================
    # Interface Compatibility Methods for AnalogicalToolWrapper
    # =========================================================================
    # These methods provide the interface expected by AnalogicalToolWrapper
    # in src/vulcan/reasoning/selection/tool_selector.py (lines 2506-2536).
    # They delegate to the existing implementation methods.
    # =========================================================================

    def find_analogies(
        self,
        query: Any,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Find analogies for a query (interface method for AnalogicalToolWrapper).
        
        This method provides the interface expected by AnalogicalToolWrapper.
        It delegates to find_multiple_analogies() which returns top-k analogies
        from stored domains.
        
        Args:
            query: Query string, dict, or domain structure
            k: Number of analogies to return (default: 5)
            
        Returns:
            Dict containing:
                - analogies: List of analogy results
                - query: The original query
                - count: Number of analogies found
                -
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> reasoner.add_domain("biology", {...})
            >>> result = reasoner.find_analogies("How does a neuron work?")
            >>> print(f"Found {result['count']} analogies")
        """
        # Extract query string if needed
        if isinstance(query, dict):
            query_text = query.get("query") or query.get("text") or str(query)
        else:
            query_text = str(query)
        
        # Use find_multiple_analogies to get top-k analogies from all domains
        analogies = self.find_multiple_analogies(query_text, k=k)
        
        return {
            "analogies": analogies,
            "query": query_text,
            "count": len(analogies),
            "success": len(analogies) > 0,
        }
    
    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Perform analogical reasoning on a problem (interface method for AnalogicalToolWrapper).
        
        This method provides the interface expected by AnalogicalToolWrapper.
        It delegates to find_structural_analogy() for structural mapping.
        
        Args:
            problem: Problem description (string or dict)
            
        Returns:
            Dict containing:
                - found: Whether an analogy was found
                - mapping: AnalogicalMapping object
                - confidence: Confidence score
                - explanation: Human-readable explanation
                
        Examples:
            >>> reasoner = AnalogicalReasoner()
            >>> reasoner.add_domain("software", {...})
            >>> result = reasoner.reason("How to design a distributed system?")
            >>> if result["found"]:
            ...     print(result["explanation"])
        """
        # Extract query from problem
        if isinstance(problem, dict):
            query = problem.get("query") or problem.get("text") or str(problem)
            target_domain = problem.get("target_domain")
        else:
            query = str(problem)
            target_domain = None
        
        # If target domain is specified, use it
        if target_domain:
            return self.find_structural_analogy(
                query,
                target_domain,
                mapping_type=MappingType.STRUCTURAL
            )
        
        # Otherwise, find analogies from all domains and return the best one
        analogies = self.find_multiple_analogies(query, k=1)
        
        if analogies:
            best = analogies[0]
            # Convert to expected format
            return {
                "found": True,
                "mapping": best.get("mapping"),
                "confidence": best.get("confidence", 0.0),
                "explanation": best.get("explanation", ""),
                "domain": best.get("domain"),
            }
        else:
            return {
                "found": False,
                "mapping": None,
                "confidence": 0.0,
                "explanation": "No analogies found in stored domains.",
                "domain": None,
            }
