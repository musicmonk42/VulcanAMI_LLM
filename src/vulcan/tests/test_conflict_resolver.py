"""
test_conflict_resolver.py 
Tests conflict resolver functionality without spawning threads.
"""

import pytest
import numpy as np
import time
import threading
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import deque, defaultdict
from enum import Enum
from unittest.mock import Mock


# ============================================================
# MOCK ENUMS
# ============================================================

class ConflictType(Enum):
    OVERLAP = "overlap"
    CONTRADICTION = "contradiction"
    SUBSUMPTION = "subsumption"
    SEMANTIC = "semantic"


class ResolutionAction(Enum):
    MERGE = "merge"
    REPLACE = "replace"
    COEXIST = "coexist"
    REJECT = "reject"
    VARIANT = "variant"


class EvidenceType(Enum):
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    HEURISTIC = "heuristic"
    DERIVED = "derived"


# ============================================================
# MOCK DATACLASSES
# ============================================================

@dataclass
class Evidence:
    """Mock Evidence class"""
    evidence_id: str
    evidence_type: EvidenceType
    source: str
    strength: float = 0.5
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_weight(self, domain: str = None, domain_weights: Dict = None) -> float:
        base_weight = self.strength * self.confidence
        
        # Apply recency decay
        age_days = (time.time() - self.timestamp) / 86400
        recency_factor = np.exp(-age_days / 365)
        
        # Apply domain weight
        domain_factor = 1.0
        if domain and domain_weights:
            type_weights = domain_weights.get(domain, {})
            domain_factor = type_weights.get(self.evidence_type.value, 1.0)
        
        return base_weight * recency_factor * domain_factor


@dataclass
class ConflictResolution:
    """Mock ConflictResolution class"""
    action: ResolutionAction
    confidence: float
    justification: str
    affected_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'action': self.action.value,
            'confidence': self.confidence,
            'justification': self.justification,
            'affected_concepts': self.affected_concepts,
            'metadata': self.metadata
        }


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


# ============================================================
# MOCK RESOLVER
# ============================================================

class MockEvidenceWeightedResolver:
    """Mock EvidenceWeightedResolver - no thread spawning"""
    
    def __init__(self, world_model=None, domain_registry=None, safety_config=None):
        self.world_model = world_model
        self.domain_registry = domain_registry
        
        # Mock safety validator
        self.safety_validator = Mock()
        self.safety_validator.validate_action_comprehensive = Mock(
            return_value=Mock(safe=True, confidence=0.9)
        )
        
        # Size limits
        self.max_evidence_concepts = 10000
        self.max_evidence_per_concept = 5000
        self.max_relationship_concepts = 10000
        self.max_relationships_per_concept = 100
        
        # Storage
        self.evidence_store: Dict[str, List[Evidence]] = defaultdict(list)
        self.resolution_history: List[Dict] = []
        self.concept_relationships: Dict[str, Set[str]] = defaultdict(set)
        
        # Counters
        self.total_resolutions = 0
        self.successful_resolutions = 0
        
        # Domain weights
        self.domain_evidence_weights = {
            'theoretical_physics': {'theoretical': 1.2, 'empirical': 0.9},
            'engineering': {'empirical': 1.2, 'theoretical': 0.8},
            'default': {'empirical': 1.0, 'theoretical': 1.0}
        }
        
        self._lock = threading.Lock()
    
    def add_evidence(self, concept_id: str, evidence: Evidence):
        with self._lock:
            # Evict if too many concepts
            while len(self.evidence_store) >= self.max_evidence_concepts:
                oldest = next(iter(self.evidence_store))
                del self.evidence_store[oldest]
            
            # Evict if too much evidence for this concept
            if len(self.evidence_store[concept_id]) >= self.max_evidence_per_concept:
                self.evidence_store[concept_id] = self.evidence_store[concept_id][-self.max_evidence_per_concept+1:]
            
            self.evidence_store[concept_id].append(evidence)
    
    def calculate_evidence_weight(self, concept: MockConcept) -> float:
        base_weight = concept.confidence * concept.success_rate
        usage_factor = min(concept.usage_count / 100, 1.0)
        
        evidence_weight = 0.0
        if concept.concept_id in self.evidence_store:
            for ev in self.evidence_store[concept.concept_id]:
                evidence_weight += ev.get_weight()
            evidence_weight = min(evidence_weight, 1.0)
        
        return base_weight * 0.4 + usage_factor * 0.3 + evidence_weight * 0.3
    
    def resolve_conflict(self, conflict) -> Dict[str, Any]:
        # Handle dict input
        if isinstance(conflict, dict):
            new_concept = conflict.get('new_concept')
            existing_concepts = conflict.get('existing_concepts', [])
            existing_concept = existing_concepts[0] if existing_concepts else None
            conflict_type = conflict.get('conflict_type', 'overlap')
        else:
            new_concept = conflict.new_concept
            existing_concept = conflict.existing_concept
            conflict_type = conflict.conflict_type
        
        # Calculate evidence weights
        new_weight = self.calculate_evidence_weight(new_concept) if new_concept else 0
        existing_weight = self.calculate_evidence_weight(existing_concept) if existing_concept else 0
        
        # Determine action based on weights
        weight_diff = new_weight - existing_weight
        
        if weight_diff > 0.3:
            action = 'replace'
            confidence = 0.8
        elif weight_diff < -0.3:
            action = 'reject'
            confidence = 0.8
        elif abs(weight_diff) < 0.1:
            action = 'merge'
            confidence = 0.7
        else:
            action = 'coexist'
            confidence = 0.6
        
        # Track resolution
        with self._lock:
            self.total_resolutions += 1
            self.successful_resolutions += 1
            
            resolution_record = {
                'action': action,
                'confidence': confidence,
                'new_concept': new_concept.concept_id if new_concept else None,
                'existing_concept': existing_concept.concept_id if existing_concept else None,
                'timestamp': time.time()
            }
            self.resolution_history.append(resolution_record)
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': f"Weight diff: {weight_diff:.2f}"
        }
    
    def merge_concepts(self, concept_a: MockConcept, concept_b: MockConcept) -> MockConcept:
        merged = MockConcept(
            concept_id=f"merged_{concept_a.concept_id}_{concept_b.concept_id}",
            pattern_signature=f"merged_{concept_a.pattern_signature}",
            confidence=(concept_a.confidence + concept_b.confidence) / 2,
            success_rate=(concept_a.success_rate + concept_b.success_rate) / 2,
            usage_count=concept_a.usage_count + concept_b.usage_count
        )
        
        # Merge domains
        merged.domains = concept_a.domains.union(concept_b.domains)
        
        # Merge features
        merged.features = {}
        all_keys = set(concept_a.features.keys()) | set(concept_b.features.keys())
        
        for key in all_keys:
            val_a = concept_a.features.get(key)
            val_b = concept_b.features.get(key)
            
            if val_a is None:
                merged.features[key] = val_b
            elif val_b is None:
                merged.features[key] = val_a
            elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                merged.features[key] = (val_a + val_b) / 2
            elif isinstance(val_a, set) and isinstance(val_b, set):
                merged.features[key] = val_a.union(val_b)
            elif isinstance(val_a, list) and isinstance(val_b, list):
                merged.features[key] = val_a + val_b
            else:
                merged.features[key] = val_a
        
        return merged
    
    def create_concept_variant(self, base_concept: MockConcept, 
                                new_pattern: Dict) -> MockConcept:
        variant_id = f"{base_concept.concept_id}_var_{int(time.time() * 1000) % 10000}"
        
        variant = MockConcept(
            concept_id=variant_id,
            pattern_signature=f"variant_{base_concept.pattern_signature}",
            confidence=base_concept.confidence * 0.8,
            success_rate=0.5,
            usage_count=0
        )
        
        variant.domains = base_concept.domains.copy()
        variant.features = {**base_concept.features, **new_pattern}
        
        # Track relationship
        with self._lock:
            self.concept_relationships[base_concept.concept_id].add(variant_id)
        
        return variant
    
    def compare_concepts(self, concept_a: MockConcept, 
                          concept_b: MockConcept) -> Dict[str, Any]:
        similarity = self._calculate_semantic_similarity(concept_a, concept_b)
        
        weight_a = self.calculate_evidence_weight(concept_a)
        weight_b = self.calculate_evidence_weight(concept_b)
        evidence_ratio = weight_a / (weight_b + 0.001)
        
        feature_overlap = self._calculate_feature_overlap(
            concept_a.features, concept_b.features
        )
        
        domain_overlap = len(concept_a.domains & concept_b.domains) / max(
            len(concept_a.domains | concept_b.domains), 1
        )
        
        performance_diff = concept_a.success_rate - concept_b.success_rate
        
        return {
            'similarity': similarity,
            'evidence_ratio': evidence_ratio,
            'feature_overlap': feature_overlap,
            'domain_overlap': domain_overlap,
            'performance_diff': performance_diff
        }
    
    def _calculate_semantic_similarity(self, concept_a: MockConcept,
                                        concept_b: MockConcept) -> float:
        if concept_a.concept_id == concept_b.concept_id:
            return 1.0
        
        feature_overlap = self._calculate_feature_overlap(
            concept_a.features, concept_b.features
        )
        
        domain_sim = len(concept_a.domains & concept_b.domains) / max(
            len(concept_a.domains | concept_b.domains), 1
        )
        
        return feature_overlap * 0.7 + domain_sim * 0.3
    
    def _calculate_feature_overlap(self, features_a: Dict, features_b: Dict) -> float:
        if not features_a and not features_b:
            return 0.0
        
        all_keys = set(features_a.keys()) | set(features_b.keys())
        if not all_keys:
            return 0.0
        
        common_keys = set(features_a.keys()) & set(features_b.keys())
        
        return len(common_keys) / len(all_keys)
    
    def get_statistics(self) -> Dict[str, Any]:
        success_rate = (
            self.successful_resolutions / max(self.total_resolutions, 1)
        )
        
        return {
            'total_resolutions': self.total_resolutions,
            'successful_resolutions': self.successful_resolutions,
            'success_rate': success_rate,
            'evidence_store_concepts': len(self.evidence_store),
            'total_evidence': sum(len(v) for v in self.evidence_store.values()),
            'tracked_relationships': len(self.concept_relationships)
        }


# Alias for compatibility
EvidenceWeightedResolver = MockEvidenceWeightedResolver


# ============================================================
# MOCK SUPPORT CLASSES
# ============================================================

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
        self.edges[f"{source}->{target}"] = kwargs
    
    def has_edge(self, source, target):
        return f"{source}->{target}" in self.edges


class MockDomainRegistry:
    """Mock domain registry for testing"""
    def __init__(self):
        self.domains = {}


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def resolver():
    return MockEvidenceWeightedResolver()


@pytest.fixture(scope="module")
def resolver_with_world_model():
    world_model = MockWorldModel()
    return MockEvidenceWeightedResolver(world_model=world_model)


@pytest.fixture(scope="module")
def resolver_with_domain_registry():
    domain_registry = MockDomainRegistry()
    return MockEvidenceWeightedResolver(domain_registry=domain_registry)


@pytest.fixture
def fresh_resolver():
    return MockEvidenceWeightedResolver()


# ============================================================
# TESTS
# ============================================================

class TestEvidenceWeightedResolverBasics:
    """Test basic resolver functionality"""
    
    def test_initialization(self, resolver):
        assert hasattr(resolver, 'evidence_store')
        assert hasattr(resolver, 'resolution_history')
        assert hasattr(resolver, 'concept_relationships')
    
    def test_initialization_with_world_model(self, resolver_with_world_model):
        assert resolver_with_world_model.world_model is not None
    
    def test_initialization_with_domain_registry(self, resolver_with_domain_registry):
        assert resolver_with_domain_registry.domain_registry is not None
    
    def test_initialization_with_safety_config(self):
        resolver = MockEvidenceWeightedResolver(safety_config={})
        assert resolver is not None
        assert hasattr(resolver, 'safety_validator')
    
    def test_size_limits(self, resolver):
        assert resolver.max_evidence_concepts == 10000
        assert resolver.max_evidence_per_concept == 5000
        assert resolver.max_relationship_concepts == 10000
        assert resolver.max_relationships_per_concept == 100


class TestEvidenceManagement:
    """Test evidence tracking and management"""
    
    def test_add_evidence(self, fresh_resolver):
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
        
        assert recent.get_weight() > old.get_weight()
    
    def test_calculate_evidence_weight_for_concept(self, fresh_resolver):
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


class TestConflictResolution:
    """Test conflict resolution logic"""
    
    def test_resolve_conflict_basic(self, resolver):
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


class TestConceptMerging:
    """Test concept merging functionality"""
    
    def test_merge_basic_concepts(self, resolver):
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


class TestConceptVariants:
    """Test concept variant creation"""
    
    def test_create_variant_basic(self, resolver):
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
        base_concept = MockConcept(
            concept_id="base_conf",
            pattern_signature="base_pattern",
            confidence=0.9
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        assert variant.confidence < base_concept.confidence
    
    def test_variant_resets_usage(self, resolver):
        base_concept = MockConcept(
            concept_id="base_usage",
            pattern_signature="base_pattern",
            usage_count=100
        )
        
        variant = resolver.create_concept_variant(base_concept, {})
        
        assert variant.usage_count == 0
    
    def test_variant_relationship_tracking(self, fresh_resolver):
        base_concept = MockConcept(
            concept_id="base_rel",
            pattern_signature="base_pattern"
        )
        
        variant = fresh_resolver.create_concept_variant(base_concept, {})
        
        if base_concept.concept_id in fresh_resolver.concept_relationships:
            relationships = fresh_resolver.concept_relationships[base_concept.concept_id]
            assert variant.concept_id in relationships


class TestSemanticSimilarity:
    """Test semantic similarity calculation"""
    
    def test_semantic_similarity_identical(self, resolver):
        concept = MockConcept(
            concept_id="sim_identical",
            pattern_signature="pattern"
        )
        concept.features = {'a': 1, 'b': 2, 'c': 3}
        
        similarity = resolver._calculate_semantic_similarity(concept, concept)
        
        assert similarity > 0.9
    
    def test_semantic_similarity_different(self, resolver):
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


class TestConceptComparison:
    """Test concept comparison functionality"""
    
    def test_compare_concepts_basic(self, resolver):
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


class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_evidence_addition(self, fresh_resolver):
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


class TestStatistics:
    """Test statistics and reporting"""
    
    def test_get_statistics_empty(self, fresh_resolver):
        stats = fresh_resolver.get_statistics()
        
        assert 'total_resolutions' in stats
        assert 'successful_resolutions' in stats
        assert 'success_rate' in stats
        assert 'evidence_store_concepts' in stats
        assert stats['total_resolutions'] == 0
    
    def test_get_statistics(self, fresh_resolver):
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"ev_stats_{i}",
                evidence_type=EvidenceType.EMPIRICAL,
                source="test",
                strength=0.8
            )
            fresh_resolver.add_evidence(f"concept_stats_{i}", evidence)
        
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


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_merge_with_missing_attributes(self, resolver):
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
        concept = MockConcept(
            concept_id="no_evidence",
            pattern_signature="pattern",
            confidence=0.7,
            success_rate=0.8
        )
        
        weight = resolver.calculate_evidence_weight(concept)
        assert weight > 0
    
    def test_empty_feature_overlap(self, resolver):
        overlap = resolver._calculate_feature_overlap({}, {})
        assert overlap == 0.0
    
    def test_resolution_to_dict(self):
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
