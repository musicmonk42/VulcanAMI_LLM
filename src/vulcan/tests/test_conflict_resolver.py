"""
test_conflict_resolver.py - Comprehensive tests for EvidenceWeightedResolver
Part of the VULCAN-AGI system

Tests cover:
- Conflict resolution strategies
- Evidence-weighted decision making
- Domain-specific evidence weights
- Concept merging and variant creation
- Resolution reversal
- Semantic similarity comparison
- Safety integration
- World model integration
- Size limits and eviction
"""

import pytest
import numpy as np
import time
import threading
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import deque
from enum import Enum


# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_bridge.conflict_resolver import (
    EvidenceWeightedResolver,
    ConflictResolution,
    ConflictType,
    ResolutionAction,
    Evidence,
    EvidenceType
)


# Mock ConceptConflict since it's in semantic_bridge_core (circular dependency)
@dataclass
class ConceptConflict:
    """Mock ConceptConflict for testing"""
    new_concept: Any
    existing_concept: Any
    conflict_type: str
    severity: float
    resolution_options: List[str] = field(default_factory=list)


# Mock classes for testing
@dataclass
class MockConcept:
    """Mock concept for testing"""
    concept_id: str
    pattern_signature: str
    confidence: float = 0.7
    success_rate: float = 0.8
    usage_count: int = 10
    domains: Set[str] = field(default_factory=set)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_signature(self) -> str:
        return self.pattern_signature
    
    def update_usage(self, success: bool):
        self.usage_count += 1
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate


class MockWorldModel:
    """Mock world model for testing"""
    def __init__(self):
        self.causal_graph = MockCausalGraph()


class MockCausalGraph:
    """Mock causal graph for testing"""
    def __init__(self):
        self.nodes = set()
        self.edges = {}
    
    def has_node(self, node):
        return node in self.nodes
    
    def add_node(self, node):
        self.nodes.add(node)
    
    def add_edge(self, source, target, **kwargs):
        edge_key = f"{source}->{target}"
        self.edges[edge_key] = kwargs
    
    def has_edge(self, source, target):
        edge_key = f"{source}->{target}"
        return edge_key in self.edges


class MockDomainRegistry:
    """Mock domain registry for testing"""
    def __init__(self):
        self.domains = {}


class TestEvidenceWeightedResolverBasics:
    """Test basic resolver functionality"""
    
    def test_initialization(self):
        """Test resolver initialization"""
        resolver = EvidenceWeightedResolver()
        
        assert len(resolver.evidence_store) == 0
        assert len(resolver.resolution_history) == 0
        assert len(resolver.concept_relationships) == 0
        assert resolver.total_resolutions == 0
        assert resolver.successful_resolutions == 0
    
    def test_initialization_with_world_model(self):
        """Test initialization with world model"""
        world_model = MockWorldModel()
        resolver = EvidenceWeightedResolver(world_model=world_model)
        
        assert resolver.world_model is world_model
    
    def test_initialization_with_domain_registry(self):
        """Test initialization with domain registry"""
        domain_registry = MockDomainRegistry()
        resolver = EvidenceWeightedResolver(domain_registry=domain_registry)
        
        assert resolver.domain_registry is domain_registry
    
    def test_initialization_with_safety_config(self):
        """Test initialization with safety config"""
        # FIXED: Use valid SafetyConfig parameters (empty dict is valid)
        # SafetyConfig doesn't accept 'max_risk_score' parameter
        safety_config = {}  # Empty dict is valid and uses defaults
        resolver = EvidenceWeightedResolver(safety_config=safety_config)
        
        assert resolver is not None
        assert hasattr(resolver, 'safety_validator')
    
    def test_size_limits(self):
        """Test size limits are properly set"""
        resolver = EvidenceWeightedResolver()
        
        assert resolver.max_evidence_concepts == 10000
        assert resolver.max_evidence_per_concept == 5000
        assert resolver.max_relationship_concepts == 10000
        assert resolver.max_relationships_per_concept == 100


class TestEvidenceManagement:
    """Test evidence tracking and management"""
    
    def test_add_evidence(self):
        """Test adding evidence for a concept"""
        resolver = EvidenceWeightedResolver()
        
        evidence = Evidence(
            evidence_id="ev_001",
            evidence_type=EvidenceType.EMPIRICAL,
            source="test_source",
            strength=0.9,
            confidence=0.85
        )
        
        resolver.add_evidence("concept_001", evidence)
        
        assert "concept_001" in resolver.evidence_store
        assert len(resolver.evidence_store["concept_001"]) == 1
    
    def test_add_multiple_evidence(self):
        """Test adding multiple evidence pieces"""
        resolver = EvidenceWeightedResolver()
        
        for i in range(5):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test_source",
                strength=0.8 + i * 0.02
            )
            resolver.add_evidence("concept_001", evidence)
        
        assert len(resolver.evidence_store["concept_001"]) == 5
    
    def test_evidence_weight_calculation(self):
        """Test evidence weight calculation"""
        resolver = EvidenceWeightedResolver()
        
        evidence = Evidence(
            evidence_id="ev_001",
            evidence_type=EvidenceType.EMPIRICAL,
            source="test",
            strength=0.9,
            confidence=0.8,
            timestamp=time.time()
        )
        
        weight = evidence.get_weight()
        
        assert 0 <= weight <= 1.0
        assert weight > 0
    
    def test_evidence_weight_with_domain(self):
        """Test domain-specific evidence weighting"""
        resolver = EvidenceWeightedResolver()
        
        evidence = Evidence(
            evidence_id="ev_001",
            evidence_type=EvidenceType.THEORETICAL,
            source="test",
            strength=0.9,
            confidence=0.8
        )
        
        weight_physics = evidence.get_weight(
            domain='theoretical_physics',
            domain_weights=resolver.domain_evidence_weights
        )
        weight_engineering = evidence.get_weight(
            domain='engineering',
            domain_weights=resolver.domain_evidence_weights
        )
        
        assert weight_physics >= weight_engineering
    
    def test_evidence_recency_decay(self):
        """Test that old evidence has lower weight"""
        resolver = EvidenceWeightedResolver()
        
        recent = Evidence(
            evidence_id="ev_recent",
            evidence_type=EvidenceType.EMPIRICAL,
            source="test",
            strength=0.9,
            confidence=0.8,
            timestamp=time.time()
        )
        
        old = Evidence(
            evidence_id="ev_old",
            evidence_type=EvidenceType.EMPIRICAL,
            source="test",
            strength=0.9,
            confidence=0.8,
            timestamp=time.time() - (365 * 24 * 3600)
        )
        
        recent_weight = recent.get_weight()
        old_weight = old.get_weight()
        
        assert recent_weight > old_weight
    
    def test_calculate_evidence_weight_for_concept(self):
        """Test calculating total evidence weight for concept"""
        resolver = EvidenceWeightedResolver()
        
        concept = MockConcept(
            concept_id="concept_001",
            pattern_signature="test_pattern",
            confidence=0.8,
            success_rate=0.9,
            usage_count=50
        )
        
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            resolver.add_evidence(concept.concept_id, evidence)
        
        weight = resolver.calculate_evidence_weight(concept)
        
        assert weight > 0
        assert weight > 0.5
    
    def test_evidence_store_size_limit(self):
        """Test evidence store respects size limits"""
        resolver = EvidenceWeightedResolver()
        resolver.max_evidence_concepts = 5
        
        for i in range(10):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            resolver.add_evidence(f"concept_{i}", evidence)
        
        assert len(resolver.evidence_store) <= resolver.max_evidence_concepts
    
    def test_evidence_per_concept_limit(self):
        """Test evidence per concept respects limit"""
        resolver = EvidenceWeightedResolver()
        resolver.max_evidence_per_concept = 10
        
        for i in range(15):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            resolver.add_evidence("concept_001", evidence)
        
        assert len(resolver.evidence_store["concept_001"]) <= resolver.max_evidence_per_concept


class TestConflictResolution:
    """Test conflict resolution logic"""
    
    def test_resolve_conflict_basic(self):
        """Test basic conflict resolution"""
        resolver = EvidenceWeightedResolver()
        
        new_concept = MockConcept(
            concept_id="new_001",
            pattern_signature="pattern_new",
            confidence=0.8
        )
        
        existing_concept = MockConcept(
            concept_id="existing_001",
            pattern_signature="pattern_existing",
            confidence=0.7
        )
        
        conflict = ConceptConflict(
            new_concept=new_concept,
            existing_concept=existing_concept,
            conflict_type="overlap",
            severity=0.6
        )
        
        resolution = resolver.resolve_conflict(conflict)
        
        assert 'action' in resolution
        assert 'confidence' in resolution
        assert 'reasoning' in resolution
        assert resolution['action'] in [
            'merge', 'replace', 'coexist', 'reject', 'variant'
        ]
    
    def test_resolve_conflict_with_dict(self):
        """Test conflict resolution with dictionary input"""
        resolver = EvidenceWeightedResolver()
        
        new_concept = MockConcept(
            concept_id="new_001",
            pattern_signature="pattern_new"
        )
        
        existing_concept = MockConcept(
            concept_id="existing_001",
            pattern_signature="pattern_existing"
        )
        
        conflict_dict = {
            'new_concept': new_concept,
            'existing_concepts': [existing_concept],
            'conflict_type': 'overlap'
        }
        
        resolution = resolver.resolve_conflict(conflict_dict)
        
        assert resolution is not None
        assert 'action' in resolution
    
    def test_resolve_high_evidence_new_concept(self):
        """Test resolution favors new concept with strong evidence"""
        resolver = EvidenceWeightedResolver()
        
        new_concept = MockConcept(
            concept_id="new_strong",
            pattern_signature="pattern_new",
            confidence=0.95,
            success_rate=0.95,
            usage_count=100
        )
        
        for i in range(5):
            evidence = Evidence(
                evidence_id=f"ev_new_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.9
            )
            resolver.add_evidence(new_concept.concept_id, evidence)
        
        existing_concept = MockConcept(
            concept_id="existing_weak",
            pattern_signature="pattern_existing",
            confidence=0.6,
            success_rate=0.6,
            usage_count=10
        )
        
        conflict = ConceptConflict(
            new_concept=new_concept,
            existing_concept=existing_concept,
            conflict_type="contradiction",
            severity=0.8
        )
        
        resolution = resolver.resolve_conflict(conflict)
        
        assert resolution['action'] != 'reject'
    
    def test_resolution_tracking(self):
        """Test that resolutions are tracked in history"""
        resolver = EvidenceWeightedResolver()
        
        new_concept = MockConcept(
            concept_id="new_001",
            pattern_signature="pattern_new"
        )
        
        existing_concept = MockConcept(
            concept_id="existing_001",
            pattern_signature="pattern_existing"
        )
        
        conflict = ConceptConflict(
            new_concept=new_concept,
            existing_concept=existing_concept,
            conflict_type="overlap",
            severity=0.6
        )
        
        initial_count = len(resolver.resolution_history)
        initial_total = resolver.total_resolutions
        
        resolver.resolve_conflict(conflict)
        
        assert len(resolver.resolution_history) > initial_count
        assert resolver.total_resolutions > initial_total


class TestConceptMerging:
    """Test concept merging functionality"""
    
    def test_merge_basic_concepts(self):
        """Test basic concept merging"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a",
            confidence=0.8,
            success_rate=0.85,
            usage_count=50
        )
        concept_a.features = {'feature1': 10, 'feature2': 20}
        concept_a.domains = {'optimization'}
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b",
            confidence=0.75,
            success_rate=0.80,
            usage_count=30
        )
        concept_b.features = {'feature2': 25, 'feature3': 30}
        concept_b.domains = {'control'}
        
        merged = resolver.merge_concepts(concept_a, concept_b)
        
        assert merged is not None
        assert 'optimization' in merged.domains
        assert 'control' in merged.domains
        assert merged.usage_count == 80
    
    def test_merge_preserves_best_attributes(self):
        """Test merging preserves best attributes"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a",
            success_rate=0.9,
            usage_count=100
        )
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b",
            success_rate=0.7,
            usage_count=20
        )
        
        merged = resolver.merge_concepts(concept_a, concept_b)
        
        assert 0.7 <= merged.success_rate <= 0.9
        assert merged.usage_count == 120
    
    def test_merge_combines_features(self):
        """Test feature combination in merge"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a"
        )
        concept_a.features = {
            'numeric': 10.0,
            'set_data': {1, 2, 3},
            'list_data': [1, 2],
            'unique_a': 'value_a'
        }
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {
            'numeric': 20.0,
            'set_data': {3, 4, 5},
            'list_data': [3, 4],
            'unique_b': 'value_b'
        }
        
        merged = resolver.merge_concepts(concept_a, concept_b)
        
        assert 'numeric' in merged.features
        assert 'unique_a' in merged.features
        assert 'unique_b' in merged.features
        
        assert merged.features['numeric'] == 15.0
        
        if 'set_data' in merged.features:
            assert len(merged.features['set_data']) >= 5
        
        if 'list_data' in merged.features:
            assert len(merged.features['list_data']) >= 2


class TestConceptVariants:
    """Test concept variant creation"""
    
    def test_create_variant_basic(self):
        """Test basic variant creation"""
        resolver = EvidenceWeightedResolver()
        
        base_concept = MockConcept(
            concept_id="base_001",
            pattern_signature="base_pattern",
            confidence=0.8
        )
        base_concept.features = {'feature1': 10}
        
        new_pattern = {'feature1': 15, 'feature2': 20}
        
        variant = resolver.create_concept_variant(base_concept, new_pattern)
        
        assert variant is not None
        assert variant.concept_id != base_concept.concept_id
        assert '_var_' in variant.concept_id
    
    def test_variant_has_lower_confidence(self):
        """Test variants start with lower confidence"""
        resolver = EvidenceWeightedResolver()
        
        base_concept = MockConcept(
            concept_id="base_002",
            pattern_signature="base_pattern",
            confidence=0.9
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        assert variant.confidence < base_concept.confidence
    
    def test_variant_resets_usage(self):
        """Test variant resets usage statistics"""
        resolver = EvidenceWeightedResolver()
        
        base_concept = MockConcept(
            concept_id="base_003",
            pattern_signature="base_pattern",
            usage_count=100
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        assert variant.usage_count == 0
    
    def test_variant_relationship_tracking(self):
        """Test variant relationships are tracked"""
        resolver = EvidenceWeightedResolver()
        
        base_concept = MockConcept(
            concept_id="base_004",
            pattern_signature="base_pattern"
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        if base_concept.concept_id in resolver.concept_relationships:
            relationships = resolver.concept_relationships[base_concept.concept_id]
            assert variant.concept_id in relationships


class TestSemanticSimilarity:
    """Test semantic similarity calculation"""
    
    def test_semantic_similarity_identical(self):
        """Test similarity of identical concepts"""
        resolver = EvidenceWeightedResolver()
        
        concept = MockConcept(
            concept_id="concept_001",
            pattern_signature="pattern"
        )
        concept.features = {'a': 1, 'b': 2, 'c': 3}
        
        similarity = resolver._calculate_semantic_similarity(concept, concept)
        
        assert similarity > 0.9
    
    def test_semantic_similarity_different(self):
        """Test similarity of very different concepts"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a"
        )
        concept_a.features = {'a': 1, 'b': 2}
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {'x': 10, 'y': 20, 'z': 30}
        
        similarity = resolver._calculate_semantic_similarity(concept_a, concept_b)
        
        assert similarity < 0.5
    
    def test_semantic_similarity_partial_overlap(self):
        """Test similarity with partial feature overlap"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a"
        )
        concept_a.features = {'a': 10, 'b': 20, 'c': 30}
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {'a': 11, 'b': 21, 'd': 40}
        
        similarity = resolver._calculate_semantic_similarity(concept_a, concept_b)
        
        assert 0.3 < similarity < 0.9


class TestConceptComparison:
    """Test concept comparison functionality"""
    
    def test_compare_concepts_basic(self):
        """Test basic concept comparison"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a",
            success_rate=0.9
        )
        concept_a.features = {'a': 1, 'b': 2}
        concept_a.domains = {'optimization'}
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b",
            success_rate=0.7
        )
        concept_b.features = {'a': 1, 'c': 3}
        concept_b.domains = {'control'}
        
        comparison = resolver.compare_concepts(concept_a, concept_b)
        
        assert 'similarity' in comparison
        assert 'evidence_ratio' in comparison
        assert 'feature_overlap' in comparison
        assert 'domain_overlap' in comparison
        assert 'performance_diff' in comparison
    
    def test_compare_concepts_performance_diff(self):
        """Test performance difference calculation"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a",
            success_rate=0.95
        )
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b",
            success_rate=0.70
        )
        
        comparison = resolver.compare_concepts(concept_a, concept_b)
        
        assert comparison['performance_diff'] == pytest.approx(0.25, abs=0.01)


class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_evidence_addition(self):
        """Test concurrent evidence addition"""
        resolver = EvidenceWeightedResolver()
        
        def add_evidence(thread_id):
            for i in range(10):
                evidence = Evidence(
                    evidence_id=f"ev_{thread_id}_{i}",
                    evidence_type=EvidenceType.EMPIRICAL,
                    source="test",
                    strength=0.8
                )
                resolver.add_evidence(f"concept_{thread_id}", evidence)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_evidence, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(resolver.evidence_store) == 3
    
    def test_concurrent_conflict_resolution(self):
        """Test concurrent conflict resolution"""
        resolver = EvidenceWeightedResolver()
        
        def resolve_conflicts(thread_id):
            for i in range(5):
                new_concept = MockConcept(
                    concept_id=f"new_{thread_id}_{i}",
                    pattern_signature=f"pattern_new_{thread_id}_{i}"
                )
                
                existing_concept = MockConcept(
                    concept_id=f"existing_{thread_id}_{i}",
                    pattern_signature=f"pattern_existing_{thread_id}_{i}"
                )
                
                conflict = ConceptConflict(
                    new_concept=new_concept,
                    existing_concept=existing_concept,
                    conflict_type="overlap",
                    severity=0.5
                )
                
                resolver.resolve_conflict(conflict)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=resolve_conflicts, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert resolver.total_resolutions == 15


class TestStatistics:
    """Test statistics and reporting"""
    
    def test_get_statistics_empty(self):
        """Test getting statistics from empty resolver"""
        resolver = EvidenceWeightedResolver()
        
        stats = resolver.get_statistics()
        
        assert 'total_resolutions' in stats
        assert 'successful_resolutions' in stats
        assert 'success_rate' in stats
        assert 'evidence_store_concepts' in stats
        assert stats['total_resolutions'] == 0
    
    def test_get_statistics(self):
        """Test getting resolver statistics"""
        resolver = EvidenceWeightedResolver()
        
        # Manually add evidence for 3 concepts
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            resolver.add_evidence(f"concept_{i}", evidence)
        
        # Resolve 2 conflicts - this will auto-generate evidence for the concepts involved
        for i in range(2):
            new_concept = MockConcept(
                concept_id=f"new_{i}",
                pattern_signature=f"pattern_new_{i}"
            )
            
            existing_concept = MockConcept(
                concept_id=f"existing_{i}",
                pattern_signature=f"pattern_existing_{i}"
            )
            
            conflict = ConceptConflict(
                new_concept=new_concept,
                existing_concept=existing_concept,
                conflict_type="overlap",
                severity=0.5
            )
            
            resolver.resolve_conflict(conflict)
        
        stats = resolver.get_statistics()
        
        assert stats['total_resolutions'] == 2
        # FIXED: The evidence store will contain the 3 manually added concepts
        # plus evidence auto-generated during conflict resolution for the existing concepts
        # So we check that we have at least the 3 we manually added
        assert stats['evidence_store_concepts'] >= 3
        assert 0 <= stats['success_rate'] <= 1


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_merge_with_missing_attributes(self):
        """Test merging concepts with missing attributes"""
        resolver = EvidenceWeightedResolver()
        
        concept_a = MockConcept(
            concept_id="concept_a",
            pattern_signature="pattern_a"
        )
        
        concept_b = MockConcept(
            concept_id="concept_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {'a': 1}
        
        merged = resolver.merge_concepts(concept_a, concept_b)
        assert merged is not None
    
    def test_calculate_evidence_weight_no_evidence(self):
        """Test calculating weight with no evidence"""
        resolver = EvidenceWeightedResolver()
        
        concept = MockConcept(
            concept_id="concept_001",
            pattern_signature="pattern",
            confidence=0.7,
            success_rate=0.8
        )
        
        weight = resolver.calculate_evidence_weight(concept)
        assert weight > 0
    
    def test_empty_feature_overlap(self):
        """Test feature overlap with empty features"""
        resolver = EvidenceWeightedResolver()
        
        overlap = resolver._calculate_feature_overlap({}, {})
        
        assert overlap == 0.0
    
    def test_resolution_to_dict(self):
        """Test resolution serialization"""
        resolution = ConflictResolution(
            action=ResolutionAction.MERGE,
            confidence=0.85,
            justification="Test merge",
            affected_concepts=["concept_a", "concept_b"]
        )
        
        resolution_dict = resolution.to_dict()
        
        assert 'action' in resolution_dict
        assert 'confidence' in resolution_dict
        assert 'justification' in resolution_dict
        assert resolution_dict['action'] == 'merge'
        assert resolution_dict['confidence'] == 0.85


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
