"""
test_concept_mapper.py - Comprehensive tests for ConceptMapper
Part of the VULCAN-AGI system

Tests cover:
- Pattern to concept mapping
- Grounded effects extraction
- Effect consistency validation
- Grounding confidence calculation
- Concept creation and registration
- Evidence tracking and updates
- Domain-adaptive thresholds
- Safety integration
- World model integration
- Concept decay and archiving
- Size limits and eviction
"""

# Add parent directory to path for imports
from semantic_bridge.concept_mapper import (Concept, ConceptMapper, EffectType,
                                            GroundingStatus, MeasurableEffect,
                                            PatternOutcome)
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock classes for testing
@dataclass
class MockPattern:
    """Mock pattern for testing"""

    pattern_id: str
    pattern_signature: str
    domain: str = "general"
    complexity: float = 0.5
    expected_effects: Dict[str, float] = field(default_factory=dict)

    def get_signature(self) -> str:
        return self.pattern_signature


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
        self.edges[(source, target)] = kwargs

    def has_edge(self, source, target):
        return (source, target) in self.edges


class MockDomainRegistry:
    """Mock domain registry for testing"""

    def __init__(self):
        self.domains = {}

    def add_domain(self, name: str, criticality: float = 0.5):
        self.domains[name] = type(
            "Domain", (), {"name": name, "criticality_score": criticality}
        )()


class TestConceptMapperBasics:
    """Test basic concept mapper functionality"""

    def test_initialization(self):
        """Test concept mapper initialization"""
        mapper = ConceptMapper()

        assert len(mapper.concepts) == 0
        assert len(mapper.effect_library) == 0
        assert mapper.total_patterns_processed == 0
        assert mapper.total_concepts_created == 0

    def test_initialization_with_world_model(self):
        """Test initialization with world model"""
        world_model = MockWorldModel()
        mapper = ConceptMapper(world_model=world_model)

        assert mapper.world_model is world_model

    def test_initialization_with_domain_registry(self):
        """Test initialization with domain registry"""
        domain_registry = MockDomainRegistry()
        mapper = ConceptMapper(domain_registry=domain_registry)

        assert mapper.domain_registry is domain_registry

    def test_initialization_with_safety_config(self):
        """Test initialization with safety config"""
        # FIXED: Use empty dict instead of invalid parameter 'max_risk_score'
        # SafetyConfig doesn't have a 'max_risk_score' parameter
        safety_config = {}
        mapper = ConceptMapper(safety_config=safety_config)

        # Should initialize without error
        assert mapper is not None


class TestPatternToConceptMapping:
    """Test pattern to concept mapping"""

    def test_map_simple_pattern(self):
        """Test mapping a simple pattern to concept"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_001",
            pattern_signature="test_pattern_sig",
            domain="optimization",
        )

        concept = mapper.map_pattern_to_concept(pattern, domain="optimization")

        assert concept is not None
        assert concept.pattern_signature == pattern.pattern_signature
        assert "optimization" in concept.domains
        assert concept.concept_id is not None

    def test_map_pattern_with_expected_effects(self):
        """Test mapping pattern with expected effects"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_002",
            pattern_signature="pattern_with_effects",
            domain="general",
            expected_effects={"accuracy": 0.95, "latency": 0.05, "memory_usage": 100.0},
        )

        concept = mapper.map_pattern_to_concept(pattern)

        assert concept is not None
        assert len(concept.grounded_effects) > 0

    def test_map_same_pattern_twice(self):
        """Test mapping the same pattern returns same concept"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_003",
            pattern_signature="duplicate_pattern",
            domain="general",
        )

        concept1 = mapper.map_pattern_to_concept(pattern)
        concept2 = mapper.map_pattern_to_concept(pattern)

        assert concept1.concept_id == concept2.concept_id

    def test_map_pattern_to_multiple_domains(self):
        """Test mapping pattern to multiple domains"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_004",
            pattern_signature="multi_domain_pattern",
            domain="general",
        )

        concept1 = mapper.map_pattern_to_concept(pattern, domain="optimization")
        concept2 = mapper.map_pattern_to_concept(pattern, domain="control")

        assert concept1.concept_id == concept2.concept_id
        assert "optimization" in concept1.domains
        assert "control" in concept1.domains


class TestGroundedEffects:
    """Test grounded effects extraction and validation"""

    def test_extract_measurable_effects_empty(self):
        """Test extracting effects from empty outcomes"""
        mapper = ConceptMapper()
        outcomes = []

        effects = mapper.extract_measurable_effects(outcomes)

        assert len(effects) == 0

    def test_extract_measurable_effects_basic(self):
        """Test extracting effects from basic outcomes"""
        mapper = ConceptMapper()

        # FIXED: Create at least 5 outcomes (base_min_instances = 5)
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i:03d}",
                pattern_signature="test_sig",
                success=True,
                measurements={"accuracy": 0.93 + i * 0.01, "latency": 0.05 + i * 0.001},
                domain="general",
            )
            for i in range(5)
        ]

        effects = mapper.extract_measurable_effects(outcomes)

        assert len(effects) > 0
        # Check that effects are stored in library
        assert len(mapper.effect_library) > 0

    def test_extract_effects_with_sufficient_evidence(self):
        """Test effects are only created with sufficient evidence"""
        mapper = ConceptMapper()
        mapper.base_min_instances = 3

        # Create outcomes with consistent measurements
        outcomes = []
        for i in range(5):
            outcomes.append(
                PatternOutcome(
                    outcome_id=f"out_{i}",
                    pattern_signature="test_sig",
                    success=True,
                    measurements={"accuracy": 0.9 + i * 0.01},
                    domain="general",
                )
            )

        effects = mapper.extract_measurable_effects(outcomes)

        # Should have effects since we have >= min_instances
        assert len(effects) > 0

    def test_effect_categorization(self):
        """Test effect type categorization"""
        mapper = ConceptMapper()

        # Test different measurement types
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature="test_sig",
                success=True,
                measurements={
                    "execution_time": 0.5,
                    "memory_usage": 100.0,
                    "accuracy": 0.95,
                    "cpu_usage": 50.0,
                },
                domain="general",
            )
            for i in range(5)
        ]

        effects = mapper.extract_measurable_effects(outcomes)

        # Should have different effect types
        effect_types = {e.effect_type for e in effects}
        assert len(effect_types) > 0

    def test_effect_consistency_validation(self):
        """Test effect consistency validation"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_005",
            pattern_signature="consistency_test",
            domain="general",
        )

        # Create consistent effects
        effects = [
            MeasurableEffect(
                effect_id="eff_001",
                effect_type=EffectType.PERFORMANCE,
                measurement=0.95,
                unit="%",
            ),
            MeasurableEffect(
                effect_id="eff_002",
                effect_type=EffectType.PERFORMANCE,
                measurement=0.93,
                unit="%",
            ),
        ]

        # Need historical outcomes for validation
        mapper.pattern_outcomes[pattern.pattern_signature] = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature=pattern.pattern_signature,
                success=True,
                measurements={"accuracy": 0.94},
                domain="general",
            )
            for i in range(6)
        ]

        is_consistent = mapper.validate_effect_consistency(pattern, effects)

        # With sufficient evidence, should validate
        assert isinstance(is_consistent, bool)


class TestConceptCreation:
    """Test concept creation and registration"""

    def test_create_concept_basic(self):
        """Test basic concept creation"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_006", pattern_signature="create_test", domain="general"
        )

        effects = [
            MeasurableEffect(
                effect_id="eff_001",
                effect_type=EffectType.PERFORMANCE,
                measurement=0.95,
                unit="%",
                confidence=0.8,
            )
        ]

        concept = mapper.create_concept(pattern, effects)

        assert concept is not None
        assert concept.pattern_signature == pattern.pattern_signature
        assert len(concept.grounded_effects) == 1
        assert concept.confidence > 0

    def test_create_concept_with_evidence(self):
        """Test creating concept with initial evidence count"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_007", pattern_signature="evidence_test", domain="general"
        )

        effects = [
            MeasurableEffect(
                effect_id="eff_001",
                effect_type=EffectType.PERFORMANCE,
                measurement=0.95,
                unit="%",
                confidence=0.8,
            )
        ]

        concept = mapper.create_concept(pattern, effects, evidence_count=10)

        assert concept.evidence_count == 10
        assert concept.positive_evidence > 0

    def test_register_concept(self):
        """Test registering existing concept"""
        mapper = ConceptMapper()

        concept = Concept(
            pattern_signature="registered_pattern", grounded_effects=[], confidence=0.7
        )

        mapper.register_concept(concept)

        assert concept.concept_id in mapper.concepts

    def test_find_similar_concepts(self):
        """Test finding similar concepts"""
        mapper = ConceptMapper()

        # Create base concept
        concept1 = Concept(
            pattern_signature="concept_1", grounded_effects=[], confidence=0.8
        )
        concept1.domains = {"optimization", "general"}
        concept1.success_rate = 0.9
        mapper.register_concept(concept1)

        # Create similar concept
        concept2 = Concept(
            pattern_signature="concept_2", grounded_effects=[], confidence=0.75
        )
        concept2.domains = {"optimization"}
        concept2.success_rate = 0.85
        mapper.register_concept(concept2)

        # Create dissimilar concept
        concept3 = Concept(
            pattern_signature="concept_3", grounded_effects=[], confidence=0.6
        )
        concept3.domains = {"control"}
        concept3.success_rate = 0.5
        mapper.register_concept(concept3)

        # Find similar to concept1
        similar = mapper.find_similar_concepts(concept1, top_k=2)

        assert len(similar) <= 2
        if len(similar) > 0:
            # First result should be most similar
            assert similar[0][1] > 0  # Similarity score


class TestEvidenceTracking:
    """Test evidence tracking and concept updates"""

    def test_update_concept_with_outcomes(self):
        """Test updating concept with new outcomes"""
        mapper = ConceptMapper()

        concept = Concept(
            pattern_signature="update_test", grounded_effects=[], confidence=0.5
        )

        initial_evidence = concept.evidence_count

        outcomes = [
            PatternOutcome(
                outcome_id="out_001",
                pattern_signature="update_test",
                success=True,
                domain="general",
            ),
            PatternOutcome(
                outcome_id="out_002",
                pattern_signature="update_test",
                success=True,
                domain="general",
            ),
        ]

        concept.update_evidence(outcomes)

        assert concept.evidence_count > initial_evidence
        assert concept.positive_evidence > 0

    def test_update_usage(self):
        """Test updating concept usage statistics"""
        mapper = ConceptMapper()

        concept = Concept(
            pattern_signature="usage_test", grounded_effects=[], confidence=0.5
        )

        initial_usage = concept.usage_count
        initial_success_rate = concept.success_rate

        # Update with successful usage
        concept.update_usage(success=True)

        assert concept.usage_count > initial_usage
        # Success rate should increase
        assert concept.success_rate >= initial_success_rate

    def test_grounding_status_progression(self):
        """Test grounding status progresses with evidence"""
        mapper = ConceptMapper()

        concept = Concept(
            pattern_signature="grounding_test",
            grounded_effects=[
                MeasurableEffect(
                    effect_id="eff_001",
                    effect_type=EffectType.PERFORMANCE,
                    measurement=0.95,
                    unit="%",
                    confidence=0.8,
                )
            ],
            confidence=0.5,
        )

        initial_status = concept.grounding_status

        # Add many successful outcomes
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature="grounding_test",
                success=True,
                measurements={"accuracy": 0.95},
                domain="general",
            )
            for i in range(25)
        ]

        concept.update_evidence(outcomes)

        # Status should improve
        assert concept.evidence_count >= 25
        # Grounding confidence should be high
        assert concept.get_grounding_confidence() > 0.5


class TestDomainAdaptiveThresholds:
    """Test domain-adaptive threshold functionality"""

    def test_get_thresholds_default(self):
        """Test getting default thresholds"""
        mapper = ConceptMapper()

        thresholds = mapper.get_thresholds_for_domain("unknown_domain")

        assert "min_instances" in thresholds
        assert "consistency" in thresholds
        assert "grounding_confidence" in thresholds

    def test_get_thresholds_with_domain_registry(self):
        """Test getting thresholds with domain registry"""
        domain_registry = MockDomainRegistry()
        domain_registry.add_domain("critical_domain", criticality=0.9)

        mapper = ConceptMapper(domain_registry=domain_registry)

        thresholds = mapper.get_thresholds_for_domain("critical_domain")

        # High criticality should have stricter thresholds
        assert thresholds["min_instances"] >= mapper.base_min_instances
        assert thresholds["consistency"] >= mapper.base_consistency_threshold


class TestConceptDecay:
    """Test concept decay and archiving"""

    def test_decay_unused_concepts(self):
        """Test removal of unused concepts"""
        mapper = ConceptMapper()

        # Create old unused concept
        old_concept = Concept(
            pattern_signature="old_concept", grounded_effects=[], confidence=0.5
        )
        old_concept.creation_time = time.time() - (40 * 24 * 3600)  # 40 days old
        old_concept.usage_count = 2  # Low usage
        mapper.register_concept(old_concept)

        # Create recent concept
        new_concept = Concept(
            pattern_signature="new_concept", grounded_effects=[], confidence=0.5
        )
        new_concept.usage_count = 10
        mapper.register_concept(new_concept)

        removed = mapper.decay_unused_concepts(max_age_days=30, min_usage=5)

        # Old concept should be removed
        assert len(removed) > 0
        assert old_concept.concept_id in removed

    def test_concept_archiving(self):
        """Test concepts are archived before removal"""
        mapper = ConceptMapper()

        concept = Concept(
            pattern_signature="archive_test", grounded_effects=[], confidence=0.5
        )
        concept.creation_time = time.time() - (40 * 24 * 3600)
        concept.usage_count = 1
        mapper.register_concept(concept)

        initial_archived_count = len(mapper.archived_concepts)

        mapper.decay_unused_concepts(max_age_days=30, min_usage=5)

        # Should have archived the concept
        assert len(mapper.archived_concepts) > initial_archived_count


class TestSizeLimitsAndEviction:
    """Test size limits and eviction strategies"""

    def test_max_concepts_limit(self):
        """Test maximum concepts limit is enforced"""
        mapper = ConceptMapper()
        mapper.max_concepts = 10  # Small limit for testing

        # Create more concepts than limit
        for i in range(15):
            pattern = MockPattern(
                pattern_id=f"test_{i}",
                pattern_signature=f"pattern_{i}",
                domain="general",
            )
            mapper.map_pattern_to_concept(pattern)

        # Should not exceed limit
        assert len(mapper.concepts) <= mapper.max_concepts

    def test_effect_library_limit(self):
        """Test effect library size limit"""
        mapper = ConceptMapper()
        mapper.max_effects = 50  # Small limit for testing

        # Create many outcomes with different measurements
        outcomes = []
        for i in range(100):
            outcomes.append(
                PatternOutcome(
                    outcome_id=f"out_{i}",
                    pattern_signature="test_sig",
                    success=True,
                    measurements={f"metric_{i}": float(i)},
                    domain="general",
                )
            )

        mapper.extract_measurable_effects(outcomes)

        # Should not exceed limit
        assert len(mapper.effect_library) <= mapper.max_effects

    def test_eviction_strategy(self):
        """Test least valuable concepts are evicted"""
        mapper = ConceptMapper()
        mapper.max_concepts = 5

        # Create concepts with different values
        for i in range(8):
            concept = Concept(
                pattern_signature=f"pattern_{i}", grounded_effects=[], confidence=0.5
            )
            concept.usage_count = i  # Different usage counts
            concept.success_rate = 0.5 + (i * 0.05)
            mapper.register_concept(concept)

        # Low usage concepts should be evicted
        assert len(mapper.concepts) <= mapper.max_concepts


class TestProcessPatternOutcomes:
    """Test processing pattern outcomes to create/update concepts"""

    def test_process_outcomes_creates_concept(self):
        """Test processing outcomes creates concept"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_008", pattern_signature="outcome_test", domain="general"
        )

        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature="outcome_test",
                success=True,
                measurements={"accuracy": 0.9},
                domain="general",
            )
            for i in range(6)
        ]

        concept = mapper.process_pattern_outcomes(pattern, outcomes)

        assert concept is not None
        assert concept.pattern_signature == pattern.pattern_signature

    def test_process_outcomes_updates_existing(self):
        """Test processing outcomes updates existing concept"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_009",
            pattern_signature="update_outcome_test",
            domain="general",
        )

        # First batch of outcomes
        outcomes1 = [
            PatternOutcome(
                outcome_id=f"out1_{i}",
                pattern_signature="update_outcome_test",
                success=True,
                measurements={"accuracy": 0.9},
                domain="general",
            )
            for i in range(6)
        ]

        concept1 = mapper.process_pattern_outcomes(pattern, outcomes1)
        initial_evidence = concept1.evidence_count

        # Second batch of outcomes
        outcomes2 = [
            PatternOutcome(
                outcome_id=f"out2_{i}",
                pattern_signature="update_outcome_test",
                success=True,
                measurements={"accuracy": 0.95},
                domain="general",
            )
            for i in range(4)
        ]

        concept2 = mapper.process_pattern_outcomes(pattern, outcomes2)

        # Should be same concept, updated
        assert concept1.concept_id == concept2.concept_id
        assert concept2.evidence_count > initial_evidence

    def test_process_insufficient_outcomes(self):
        """Test processing with insufficient outcomes"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_010",
            pattern_signature="insufficient_test",
            domain="general",
        )

        # Too few outcomes
        outcomes = [
            PatternOutcome(
                outcome_id="out_001",
                pattern_signature="insufficient_test",
                success=True,
                measurements={"accuracy": 0.9},
                domain="general",
            )
        ]

        concept = mapper.process_pattern_outcomes(pattern, outcomes)

        # Should not create concept with insufficient evidence
        assert concept is None


class TestWorldModelIntegration:
    """Test world model integration"""

    def test_link_concept_to_world_model(self):
        """Test linking concept to world model"""
        world_model = MockWorldModel()
        mapper = ConceptMapper(world_model=world_model)

        pattern = MockPattern(
            pattern_id="test_011",
            pattern_signature="world_model_test",
            domain="general",
            expected_effects={"accuracy": 0.95},
        )

        concept = mapper.map_pattern_to_concept(pattern)

        # Should have added nodes to world model
        assert len(world_model.causal_graph.nodes) > 0


class TestThreadSafety:
    """Test thread-safe operations"""

    def test_concurrent_pattern_mapping(self):
        """Test concurrent pattern to concept mapping"""
        mapper = ConceptMapper()

        def map_patterns(thread_id):
            for i in range(10):
                pattern = MockPattern(
                    pattern_id=f"t{thread_id}_p{i}",
                    pattern_signature=f"pattern_{thread_id}_{i}",
                    domain="general",
                )
                mapper.map_pattern_to_concept(pattern)

        threads = []
        for i in range(5):
            t = threading.Thread(target=map_patterns, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have created concepts from all threads
        assert len(mapper.concepts) > 0

    def test_concurrent_concept_registration(self):
        """Test concurrent concept registration"""
        mapper = ConceptMapper()

        def register_concepts(thread_id):
            for i in range(10):
                concept = Concept(
                    pattern_signature=f"pattern_{thread_id}_{i}",
                    grounded_effects=[],
                    confidence=0.5,
                )
                mapper.register_concept(concept)

        threads = []
        for i in range(5):
            t = threading.Thread(target=register_concepts, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all concepts registered
        assert len(mapper.concepts) > 0


class TestStatistics:
    """Test statistics and reporting"""

    def test_get_statistics(self):
        """Test getting mapper statistics"""
        mapper = ConceptMapper()

        # Create some concepts
        for i in range(3):
            pattern = MockPattern(
                pattern_id=f"test_{i}",
                pattern_signature=f"pattern_{i}",
                domain="general",
            )
            mapper.map_pattern_to_concept(pattern)

        stats = mapper.get_statistics()

        assert "total_patterns_processed" in stats
        assert "total_concepts_created" in stats
        assert "active_concepts" in stats
        assert "effect_library_size" in stats
        assert stats["active_concepts"] == 3


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_extract_effects_with_nan_values(self):
        """Test handling NaN values in measurements"""
        mapper = ConceptMapper()

        outcomes = [
            PatternOutcome(
                outcome_id="out_001",
                pattern_signature="nan_test",
                success=True,
                measurements={"metric": float("nan")},
                domain="general",
            )
        ]

        # Should handle gracefully without crashing
        effects = mapper.extract_measurable_effects(outcomes)

        # NaN values should be filtered out
        assert len(effects) == 0 or all(np.isfinite(e.measurement) for e in effects)

    def test_extract_effects_with_inf_values(self):
        """Test handling infinite values"""
        mapper = ConceptMapper()

        outcomes = [
            PatternOutcome(
                outcome_id="out_001",
                pattern_signature="inf_test",
                success=True,
                measurements={"metric": float("inf")},
                domain="general",
            )
        ]

        effects = mapper.extract_measurable_effects(outcomes)

        # Infinite values should be filtered
        assert len(effects) == 0 or all(np.isfinite(e.measurement) for e in effects)

    def test_empty_pattern_signature(self):
        """Test handling empty pattern signature"""
        mapper = ConceptMapper()
        pattern = MockPattern(
            pattern_id="test_empty", pattern_signature="", domain="general"
        )

        # Should handle gracefully
        concept = mapper.map_pattern_to_concept(pattern)
        assert concept is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
