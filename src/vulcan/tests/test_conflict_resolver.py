"""
test_conflict_resolver.py - OPTIMIZED VERSION
Comprehensive tests for EvidenceWeightedResolver

OPTIMIZATION: Uses module-scoped fixtures to avoid re-initializing
the resolver for every test (which was causing extreme slowness).
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


# ============================================================
# MOCK CLASSES
# ============================================================

@dataclass
class ConceptConflict:
    """Mock ConceptConflict for testing"""
    new_concept: Any
    existing_concept: Any
    conflict_type: str
    severity: float
    resolution_options: List[str] = field(default_factory=list)


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


# ============================================================
# SHARED FIXTURES - KEY OPTIMIZATION
# ============================================================

@pytest.fixture(scope="module")
def resolver():
    """
    Module-scoped resolver - created once per test module.
    This is the KEY optimization - avoids re-initializing for every test.
    """
    return EvidenceWeightedResolver()


@pytest.fixture(scope="module")
def resolver_with_world_model():
    """Module-scoped resolver with world model"""
    world_model = MockWorldModel()
    return EvidenceWeightedResolver(world_model=world_model)


@pytest.fixture(scope="module")
def resolver_with_domain_registry():
    """Module-scoped resolver with domain registry"""
    domain_registry = MockDomainRegistry()
    return EvidenceWeightedResolver(domain_registry=domain_registry)


@pytest.fixture
def fresh_resolver():
    """
    Function-scoped resolver for tests that need clean state.
    Only use when absolutely necessary.
    """
    return EvidenceWeightedResolver()


# ============================================================
# TEST: BASIC FUNCTIONALITY
# ============================================================

class TestEvidenceWeightedResolverBasics:
    """Test basic resolver functionality"""
    
    def test_initialization(self, resolver):
        """Test resolver initialization"""
        assert hasattr(resolver, 'evidence_store')
        assert hasattr(resolver, 'resolution_history')
        assert hasattr(resolver, 'concept_relationships')
    
    def test_initialization_with_world_model(self, resolver_with_world_model):
        """Test initialization with world model"""
        assert resolver_with_world_model.world_model is not None
    
    def test_initialization_with_domain_registry(self, resolver_with_domain_registry):
        """Test initialization with domain registry"""
        assert resolver_with_domain_registry.domain_registry is not None
    
    def test_initialization_with_safety_config(self):
        """Test initialization with safety config - needs fresh instance"""
        safety_config = {}
        resolver = EvidenceWeightedResolver(safety_config=safety_config)
        assert resolver is not None
        assert hasattr(resolver, 'safety_validator')
    
    def test_size_limits(self, resolver):
        """Test size limits are properly set"""
        assert resolver.max_evidence_concepts == 10000
        assert resolver.max_evidence_per_concept == 5000
        assert resolver.max_relationship_concepts == 10000
        assert resolver.max_relationships_per_concept == 100


# ============================================================
# TEST: EVIDENCE MANAGEMENT
# ============================================================

class TestEvidenceManagement:
    """Test evidence tracking and management"""
    
    def test_add_evidence(self, fresh_resolver):
        """Test adding evidence for a concept"""
        evidence = Evidence(
            evidence_id="ev_001",
            evidence_type=EvidenceType.EMPIRICAL,
            source="test_source",
            strength=0.9,
            confidence=0.85
        )
        
        fresh_resolver.add_evidence("concept_001", evidence)
        
        assert "concept_001" in fresh_resolver.evidence_store
        assert len(fresh_resolver.evidence_store["concept_001"]) == 1
    
    def test_add_multiple_evidence(self, fresh_resolver):
        """Test adding multiple evidence pieces"""
        for i in range(5):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test_source",
                strength=0.8 + i * 0.02
            )
            fresh_resolver.add_evidence("concept_multi", evidence)
        
        assert len(fresh_resolver.evidence_store["concept_multi"]) == 5
    
    def test_evidence_weight_calculation(self, resolver):
        """Test evidence weight calculation"""
        evidence = Evidence(
            evidence_id="ev_weight",
            evidence_type=EvidenceType.EMPIRICAL,
            source="test",
            strength=0.9,
            confidence=0.8,
            timestamp=time.time()
        )
        
        weight = evidence.get_weight()
        
        assert 0 <= weight <= 1.0
        assert weight > 0
    
    def test_evidence_weight_with_domain(self, resolver):
        """Test domain-specific evidence weighting"""
        evidence = Evidence(
            evidence_id="ev_domain",
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
    
    def test_evidence_recency_decay(self, resolver):
        """Test that old evidence has lower weight"""
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
    
    def test_calculate_evidence_weight_for_concept(self, fresh_resolver):
        """Test calculating total evidence weight for concept"""
        concept = MockConcept(
            concept_id="concept_weight_calc",
            pattern_signature="test_pattern",
            confidence=0.8,
            success_rate=0.9,
            usage_count=50
        )
        
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"ev_calc_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            fresh_resolver.add_evidence(concept.concept_id, evidence)
        
        weight = fresh_resolver.calculate_evidence_weight(concept)
        
        assert weight > 0
        assert weight > 0.5
    
    def test_evidence_store_size_limit(self, fresh_resolver):
        """Test evidence store respects size limits"""
        fresh_resolver.max_evidence_concepts = 5
        
        for i in range(10):
            evidence = Evidence(
                evidence_id=f"ev_limit_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            fresh_resolver.add_evidence(f"concept_limit_{i}", evidence)
        
        assert len(fresh_resolver.evidence_store) <= fresh_resolver.max_evidence_concepts
    
    def test_evidence_per_concept_limit(self, fresh_resolver):
        """Test evidence per concept respects limit"""
        fresh_resolver.max_evidence_per_concept = 10
        
        for i in range(15):
            evidence = Evidence(
                evidence_id=f"ev_per_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            fresh_resolver.add_evidence("concept_per_limit", evidence)
        
        assert len(fresh_resolver.evidence_store["concept_per_limit"]) <= fresh_resolver.max_evidence_per_concept


# ============================================================
# TEST: CONFLICT RESOLUTION
# ============================================================

class TestConflictResolution:
    """Test conflict resolution logic"""
    
    def test_resolve_conflict_basic(self, resolver):
        """Test basic conflict resolution"""
        new_concept = MockConcept(
            concept_id="new_basic",
            pattern_signature="pattern_new",
            confidence=0.8
        )
        
        existing_concept = MockConcept(
            concept_id="existing_basic",
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
        assert resolution['action'] in ['merge', 'replace', 'coexist', 'reject', 'variant']
    
    def test_resolve_conflict_with_dict(self, resolver):
        """Test conflict resolution with dictionary input"""
        new_concept = MockConcept(
            concept_id="new_dict",
            pattern_signature="pattern_new"
        )
        
        existing_concept = MockConcept(
            concept_id="existing_dict",
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
    
    def test_resolve_high_evidence_new_concept(self, fresh_resolver):
        """Test resolution favors new concept with strong evidence"""
        new_concept = MockConcept(
            concept_id="new_strong",
            pattern_signature="pattern_new",
            confidence=0.95,
            success_rate=0.95,
            usage_count=100
        )
        
        for i in range(5):
            evidence = Evidence(
                evidence_id=f"ev_strong_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.9
            )
            fresh_resolver.add_evidence(new_concept.concept_id, evidence)
        
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
        
        resolution = fresh_resolver.resolve_conflict(conflict)
        
        assert resolution['action'] != 'reject'
    
    def test_resolution_tracking(self, fresh_resolver):
        """Test that resolutions are tracked in history"""
        new_concept = MockConcept(
            concept_id="new_track",
            pattern_signature="pattern_new"
        )
        
        existing_concept = MockConcept(
            concept_id="existing_track",
            pattern_signature="pattern_existing"
        )
        
        conflict = ConceptConflict(
            new_concept=new_concept,
            existing_concept=existing_concept,
            conflict_type="overlap",
            severity=0.6
        )
        
        initial_count = len(fresh_resolver.resolution_history)
        initial_total = fresh_resolver.total_resolutions
        
        fresh_resolver.resolve_conflict(conflict)
        
        assert len(fresh_resolver.resolution_history) > initial_count
        assert fresh_resolver.total_resolutions > initial_total


# ============================================================
# TEST: CONCEPT MERGING
# ============================================================

class TestConceptMerging:
    """Test concept merging functionality"""
    
    def test_merge_basic_concepts(self, resolver):
        """Test basic concept merging"""
        concept_a = MockConcept(
            concept_id="merge_a",
            pattern_signature="pattern_a",
            confidence=0.8,
            success_rate=0.85,
            usage_count=50
        )
        concept_a.features = {'feature1': 10, 'feature2': 20}
        concept_a.domains = {'optimization'}
        
        concept_b = MockConcept(
            concept_id="merge_b",
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
    
    def test_merge_preserves_best_attributes(self, resolver):
        """Test merging preserves best attributes"""
        concept_a = MockConcept(
            concept_id="best_a",
            pattern_signature="pattern_a",
            success_rate=0.9,
            usage_count=100
        )
        
        concept_b = MockConcept(
            concept_id="best_b",
            pattern_signature="pattern_b",
            success_rate=0.7,
            usage_count=20
        )
        
        merged = resolver.merge_concepts(concept_a, concept_b)
        
        assert 0.7 <= merged.success_rate <= 0.9
        assert merged.usage_count == 120
    
    def test_merge_combines_features(self, resolver):
        """Test feature combination in merge"""
        concept_a = MockConcept(
            concept_id="feat_a",
            pattern_signature="pattern_a"
        )
        concept_a.features = {
            'numeric': 10.0,
            'set_data': {1, 2, 3},
            'list_data': [1, 2],
            'unique_a': 'value_a'
        }
        
        concept_b = MockConcept(
            concept_id="feat_b",
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


# ============================================================
# TEST: CONCEPT VARIANTS
# ============================================================

class TestConceptVariants:
    """Test concept variant creation"""
    
    def test_create_variant_basic(self, resolver):
        """Test basic variant creation"""
        base_concept = MockConcept(
            concept_id="base_var",
            pattern_signature="base_pattern",
            confidence=0.8
        )
        base_concept.features = {'feature1': 10}
        
        new_pattern = {'feature1': 15, 'feature2': 20}
        
        variant = resolver.create_concept_variant(base_concept, new_pattern)
        
        assert variant is not None
        assert variant.concept_id != base_concept.concept_id
        assert '_var_' in variant.concept_id
    
    def test_variant_has_lower_confidence(self, resolver):
        """Test variants start with lower confidence"""
        base_concept = MockConcept(
            concept_id="base_conf",
            pattern_signature="base_pattern",
            confidence=0.9
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        assert variant.confidence < base_concept.confidence
    
    def test_variant_resets_usage(self, resolver):
        """Test variant resets usage statistics"""
        base_concept = MockConcept(
            concept_id="base_usage",
            pattern_signature="base_pattern",
            usage_count=100
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        assert variant.usage_count == 0
    
    def test_variant_relationship_tracking(self, fresh_resolver):
        """Test variant relationships are tracked"""
        base_concept = MockConcept(
            concept_id="base_rel",
            pattern_signature="base_pattern"
        )
        
        variant = fresh_resolver.create_concept_variant(base_concept, {})
        
        if base_concept.concept_id in fresh_resolver.concept_relationships:
            relationships = fresh_resolver.concept_relationships[base_concept.concept_id]
            assert variant.concept_id in relationships


# ============================================================
# TEST: SEMANTIC SIMILARITY
# ============================================================

class TestSemanticSimilarity:
    """Test semantic similarity calculation"""
    
    def test_semantic_similarity_identical(self, resolver):
        """Test similarity of identical concepts"""
        concept = MockConcept(
            concept_id="sim_identical",
            pattern_signature="pattern"
        )
        concept.features = {'a': 1, 'b': 2, 'c': 3}
        
        similarity = resolver._calculate_semantic_similarity(concept, concept)
        
        assert similarity > 0.9
    
    def test_semantic_similarity_different(self, resolver):
        """Test similarity of very different concepts"""
        concept_a = MockConcept(
            concept_id="sim_diff_a",
            pattern_signature="pattern_a"
        )
        concept_a.features = {'a': 1, 'b': 2}
        
        concept_b = MockConcept(
            concept_id="sim_diff_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {'x': 10, 'y': 20, 'z': 30}
        
        similarity = resolver._calculate_semantic_similarity(concept_a, concept_b)
        
        assert similarity < 0.5
    
    def test_semantic_similarity_partial_overlap(self, resolver):
        """Test similarity with partial feature overlap"""
        concept_a = MockConcept(
            concept_id="sim_partial_a",
            pattern_signature="pattern_a"
        )
        concept_a.features = {'a': 10, 'b': 20, 'c': 30}
        
        concept_b = MockConcept(
            concept_id="sim_partial_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {'a': 11, 'b': 21, 'd': 40}
        
        similarity = resolver._calculate_semantic_similarity(concept_a, concept_b)
        
        assert 0.3 < similarity < 0.9


# ============================================================
# TEST: CONCEPT COMPARISON
# ============================================================

class TestConceptComparison:
    """Test concept comparison functionality"""
    
    def test_compare_concepts_basic(self, resolver):
        """Test basic concept comparison"""
        concept_a = MockConcept(
            concept_id="cmp_a",
            pattern_signature="pattern_a",
            success_rate=0.9
        )
        concept_a.features = {'a': 1, 'b': 2}
        concept_a.domains = {'optimization'}
        
        concept_b = MockConcept(
            concept_id="cmp_b",
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
    
    def test_compare_concepts_performance_diff(self, resolver):
        """Test performance difference calculation"""
        concept_a = MockConcept(
            concept_id="perf_a",
            pattern_signature="pattern_a",
            success_rate=0.95
        )
        
        concept_b = MockConcept(
            concept_id="perf_b",
            pattern_signature="pattern_b",
            success_rate=0.70
        )
        
        comparison = resolver.compare_concepts(concept_a, concept_b)
        
        assert comparison['performance_diff'] == pytest.approx(0.25, abs=0.01)


# ============================================================
# TEST: THREAD SAFETY
# ============================================================

class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_evidence_addition(self, fresh_resolver):
        """Test concurrent evidence addition"""
        def add_evidence(thread_id):
            for i in range(10):
                evidence = Evidence(
                    evidence_id=f"ev_thread_{thread_id}_{i}",
                    evidence_type=EvidenceType.EMPIRICAL,
                    source="test",
                    strength=0.8
                )
                fresh_resolver.add_evidence(f"concept_thread_{thread_id}", evidence)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_evidence, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(fresh_resolver.evidence_store) == 3
    
    def test_concurrent_conflict_resolution(self, fresh_resolver):
        """Test concurrent conflict resolution"""
        def resolve_conflicts(thread_id):
            for i in range(5):
                new_concept = MockConcept(
                    concept_id=f"new_concurrent_{thread_id}_{i}",
                    pattern_signature=f"pattern_new_{thread_id}_{i}"
                )
                
                existing_concept = MockConcept(
                    concept_id=f"existing_concurrent_{thread_id}_{i}",
                    pattern_signature=f"pattern_existing_{thread_id}_{i}"
                )
                
                conflict = ConceptConflict(
                    new_concept=new_concept,
                    existing_concept=existing_concept,
                    conflict_type="overlap",
                    severity=0.5
                )
                
                fresh_resolver.resolve_conflict(conflict)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=resolve_conflicts, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert fresh_resolver.total_resolutions == 15


# ============================================================
# TEST: STATISTICS
# ============================================================

class TestStatistics:
    """Test statistics and reporting"""
    
    def test_get_statistics_empty(self, fresh_resolver):
        """Test getting statistics from empty resolver"""
        stats = fresh_resolver.get_statistics()
        
        assert 'total_resolutions' in stats
        assert 'successful_resolutions' in stats
        assert 'success_rate' in stats
        assert 'evidence_store_concepts' in stats
        assert stats['total_resolutions'] == 0
    
    def test_get_statistics(self, fresh_resolver):
        """Test getting resolver statistics"""
        # Add evidence for 3 concepts
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"ev_stats_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            fresh_resolver.add_evidence(f"concept_stats_{i}", evidence)
        
        # Resolve 2 conflicts
        for i in range(2):
            new_concept = MockConcept(
                concept_id=f"new_stats_{i}",
                pattern_signature=f"pattern_new_{i}"
            )
            
            existing_concept = MockConcept(
                concept_id=f"existing_stats_{i}",
                pattern_signature=f"pattern_existing_{i}"
            )
            
            conflict = ConceptConflict(
                new_concept=new_concept,
                existing_concept=existing_concept,
                conflict_type="overlap",
                severity=0.5
            )
            
            fresh_resolver.resolve_conflict(conflict)
        
        stats = fresh_resolver.get_statistics()
        
        assert stats['total_resolutions'] == 2
        assert stats['evidence_store_concepts'] >= 3
        assert 0 <= stats['success_rate'] <= 1


# ============================================================
# TEST: EDGE CASES
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_merge_with_missing_attributes(self, resolver):
        """Test merging concepts with missing attributes"""
        concept_a = MockConcept(
            concept_id="edge_a",
            pattern_signature="pattern_a"
        )
        
        concept_b = MockConcept(
            concept_id="edge_b",
            pattern_signature="pattern_b"
        )
        concept_b.features = {'a': 1}
        
        merged = resolver.merge_concepts(concept_a, concept_b)
        assert merged is not None
    
    def test_calculate_evidence_weight_no_evidence(self, resolver):
        """Test calculating weight with no evidence"""
        concept = MockConcept(
            concept_id="no_evidence",
            pattern_signature="pattern",
            confidence=0.7,
            success_rate=0.8
        )
        
        weight = resolver.calculate_evidence_weight(concept)
        assert weight > 0
    
    def test_empty_feature_overlap(self, resolver):
        """Test feature overlap with empty features"""
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
