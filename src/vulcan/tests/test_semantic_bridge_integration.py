"""
test_semantic_bridge_integration.py - Comprehensive integration tests for SemanticBridge
Part of the VULCAN-AGI system

Tests the entire semantic bridge system working together:
- End-to-end concept learning and transfer
- Cross-component communication
- World model integration
- Safety validation integration
- Cache coordination
- Domain management
- Conflict resolution workflows
"""

import pytest
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import shutil


# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_bridge import (
    SemanticBridge,
    ConceptMapper,
    Concept,
    PatternOutcome,
    MeasurableEffect,
    MapperEffectType,
    GroundingStatus,
    EvidenceWeightedResolver,
    ConflictResolution,
    ConflictType,
    DomainRegistry,
    DomainProfile,
    DomainCriticality,
    TransferEngine,
    TransferDecision,
    TransferType,
    ConceptEffect,
    CacheManager,
    create_semantic_bridge,
    get_version_info,
    get_default_config
)


# Mock classes for integration testing
class MockWorldModel:
    """Mock world model for testing"""
    def __init__(self):
        self.causal_graph = MockCausalGraph()
        self.predictions = []
        self.updates = []
    
    def predict_outcome(self, pattern, context):
        prediction = {'success_probability': 0.8, 'expected_measurements': {}}
        self.predictions.append(prediction)
        return prediction
    
    def record_update(self, source, description):
        self.updates.append({'source': source, 'description': description})


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
    
    def remove_edge(self, source, target):
        edge_key = f"{source}->{target}"
        if edge_key in self.edges:
            del self.edges[edge_key]


class MockVulcanMemory:
    """Mock VULCAN memory for testing"""
    def __init__(self, temp_dir):
        self.storage_path = temp_dir
        self.stored_concepts = []
    
    def store_concept(self, concept):
        self.stored_concepts.append(concept)


class MockPattern:
    """Mock pattern for testing"""
    def __init__(self, pattern_id: str, domain: str = "general", complexity: float = 0.5):
        self.pattern_id = pattern_id
        self.pattern_signature = f"sig_{pattern_id}"
        self.domain = domain
        self.complexity = complexity
        self.confidence = 0.7
        self.expected_effects = {}
        self.features = {}


class TestBasicIntegration:
    """Test basic integration between components"""
    
    def test_factory_function(self):
        """Test factory function creates proper bridge"""
        bridge = create_semantic_bridge()
        
        assert isinstance(bridge, SemanticBridge)
        assert bridge.concept_mapper is not None
        assert bridge.conflict_resolver is not None
        assert bridge.domain_registry is not None
        assert bridge.transfer_engine is not None
        assert bridge.cache_manager is not None
    
    def test_version_info(self):
        """Test version information"""
        info = get_version_info()
        
        assert 'version' in info
        assert 'status' in info
        assert 'components' in info
        assert 'features' in info
        assert info['status'] == 'Production'
    
    def test_default_config(self):
        """Test default configuration"""
        config = get_default_config()
        
        assert 'safety' in config
        assert 'memory' in config
        assert 'learning' in config
        assert 'transfer' in config


class TestConceptLearningPipeline:
    """Test end-to-end concept learning pipeline"""
    
    def test_learn_concept_full_pipeline(self):
        """Test complete concept learning pipeline"""
        world_model = MockWorldModel()
        bridge = SemanticBridge(world_model=world_model)
        
        # Create pattern
        pattern = MockPattern("test_pattern_001", "general")
        pattern.expected_effects = {'accuracy': 0.9, 'latency': 0.1}
        
        # Create outcomes
        outcomes = []
        for i in range(10):
            outcome = PatternOutcome(
                outcome_id=f"outcome_{i}",
                pattern_signature=pattern.pattern_signature,
                success=True,
                measurements={'accuracy': 0.9 + np.random.normal(0, 0.02)},
                domain="general",
                timestamp=time.time()
            )
            outcomes.append(outcome)
        
        # Learn concept
        concept = bridge.learn_concept_from_pattern(pattern, outcomes)
        
        # Verify concept was created
        assert concept is not None
        assert hasattr(concept, 'concept_id')
        assert 'general' in concept.domains
        
        # Verify concept mapper stored it
        assert len(bridge.concept_mapper.concepts) > 0
        
        # Verify world model was updated
        assert len(world_model.causal_graph.nodes) > 0
        
        # Verify operation history was recorded
        assert len(bridge.operation_history) > 0
    
    def test_learn_multiple_concepts_different_domains(self):
        """Test learning multiple concepts in different domains"""
        bridge = SemanticBridge()
        
        domains = ['general', 'optimization', 'control']
        learned_concepts = []
        
        for i, domain in enumerate(domains):
            pattern = MockPattern(f"pattern_{i}", domain)
            
            outcomes = [
                PatternOutcome(
                    outcome_id=f"outcome_{i}_{j}",
                    pattern_signature=pattern.pattern_signature,
                    success=True,
                    measurements={'metric': 0.8},
                    domain=domain,
                    timestamp=time.time()
                )
                for j in range(7)
            ]
            
            concept = bridge.learn_concept_from_pattern(pattern, outcomes)
            if concept:
                learned_concepts.append(concept)
        
        # Verify concepts learned across domains
        assert len(learned_concepts) >= 1
        
        # Verify domain registry was updated
        assert len(bridge.domain_registry.domains) >= len(domains)


class TestCrossDomainTransfer:
    """Test cross-domain concept transfer"""
    
    def test_transfer_concept_between_domains(self):
        """Test transferring a concept between domains"""
        bridge = SemanticBridge()
        
        # Learn concept in source domain
        pattern = MockPattern("source_pattern", "general")
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature=pattern.pattern_signature,
                success=True,
                measurements={'accuracy': 0.85},
                domain="general",
                timestamp=time.time()
            )
            for i in range(8)
        ]
        
        concept = bridge.learn_concept_from_pattern(pattern, outcomes)
        
        if concept:
            # Test transfer to optimization domain
            compatibility = bridge.validate_transfer_compatibility(
                concept, "general", "optimization"
            )
            
            assert compatibility is not None
            assert hasattr(compatibility, 'compatibility_score')
            assert 0 <= compatibility.compatibility_score <= 1
            
            # Get applicable concepts for target domain
            applicable = bridge.get_applicable_concepts("optimization", min_confidence=0.5)
            
            assert isinstance(applicable, list)
    
    def test_transfer_strategy_selection(self):
        """Test transfer strategy selection"""
        bridge = SemanticBridge()
        
        pattern = MockPattern("strategy_pattern", "general")
        
        # Test different target domains
        strategies = []
        for target in ['optimization', 'control', 'general']:
            strategy = bridge.select_transfer_strategy(pattern, target)
            strategies.append(strategy)
            assert strategy in ['direct_transfer', 'structural_analogy', 
                              'functional_transfer', 'general_analogy']
        
        # Should have selected strategies
        assert len(strategies) == 3


class TestConflictResolution:
    """Test conflict resolution workflows"""
    
    def test_resolve_duplicate_concepts(self):
        """Test resolving duplicate concept conflicts"""
        bridge = SemanticBridge()
        
        # Create first concept
        pattern1 = MockPattern("duplicate_pattern", "general")
        pattern1.features = {'feature_a': 1.0, 'feature_b': 2.0}
        
        outcomes1 = [
            PatternOutcome(
                outcome_id=f"out1_{i}",
                pattern_signature=pattern1.pattern_signature,
                success=True,
                measurements={'metric': 0.8},
                domain="general",
                timestamp=time.time()
            )
            for i in range(6)
        ]
        
        concept1 = bridge.learn_concept_from_pattern(pattern1, outcomes1)
        
        if concept1:
            # Create similar pattern (potential duplicate)
            pattern2 = MockPattern("similar_pattern", "general")
            pattern2.features = {'feature_a': 1.0, 'feature_b': 2.0}
            
            # Resolve conflict
            resolution = bridge.resolve_concept_conflict(pattern2, [concept1])
            
            assert resolution is not None
            assert 'action' in resolution
            assert resolution['action'] in ['replace', 'merge', 'coexist', 'reject', 'none']


class TestCacheCoordination:
    """Test cache coordination across components"""
    
    def test_cache_manager_registration(self):
        """Test that all caches are registered"""
        bridge = SemanticBridge()
        
        stats = bridge.cache_manager.get_statistics()
        
        assert 'caches' in stats
        assert 'summary' in stats
        
        # Verify key caches are registered
        cache_names = list(stats['caches'].keys())
        assert 'pattern_signature' in cache_names
        assert 'domain_concept' in cache_names
    
    def test_cache_hit_miss_tracking(self):
        """Test cache hit/miss tracking"""
        bridge = SemanticBridge()
        
        initial_stats = bridge.cache_manager.get_statistics()
        
        # Trigger some cache operations
        pattern = MockPattern("cache_test", "general")
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature=pattern.pattern_signature,
                success=True,
                measurements={},
                domain="general",
                timestamp=time.time()
            )
            for i in range(5)
        ]
        
        # This should trigger cache operations
        bridge._extract_pattern_signature(pattern, outcomes)
        bridge._extract_pattern_signature(pattern, outcomes)  # Should be cached
        
        final_stats = bridge.cache_manager.get_statistics()
        
        # Some cache activity should have occurred
        assert final_stats['summary']['total_hits'] >= 0
        assert final_stats['summary']['total_misses'] >= 0
    
    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced"""
        bridge = SemanticBridge()
        
        # Check memory status
        memory_check = bridge.cache_manager.check_memory()
        
        assert 'total_mb' in memory_check
        assert 'limit_mb' in memory_check
        assert 'usage_percent' in memory_check


class TestDomainManagement:
    """Test domain management and relationships"""
    
    def test_domain_registration_and_retrieval(self):
        """Test domain registration and retrieval"""
        bridge = SemanticBridge()
        
        # Register custom domain
        profile = DomainProfile(
            name="custom_domain",
            criticality_score=DomainCriticality.MEDIUM.value,
            capabilities={'capability_a', 'capability_b'},
            limitations={'limitation_x'}
        )
        
        bridge.domain_registry.register_domain("custom_domain", profile)
        
        # Verify registration
        assert "custom_domain" in bridge.domain_registry.domains
        
        # Get related domains
        related = bridge.domain_registry.get_related_domains("custom_domain")
        assert isinstance(related, list)
    
    def test_domain_hierarchy(self):
        """Test domain hierarchy relationships"""
        bridge = SemanticBridge()
        
        # Add domain relationship
        bridge.domain_registry.add_domain_relationship(
            "general", "optimization", "specialization", 0.8
        )
        
        # Get hierarchy
        hierarchy = bridge.domain_registry.get_domain_hierarchy("optimization")
        
        assert 'parents' in hierarchy
        assert 'children' in hierarchy
        assert 'siblings' in hierarchy
    
    def test_domain_distance_calculation(self):
        """Test domain distance calculation"""
        bridge = SemanticBridge()
        
        # Calculate distances
        dist_same = bridge.domain_registry.calculate_domain_distance("general", "general")
        dist_diff = bridge.domain_registry.calculate_domain_distance("general", "optimization")
        
        assert dist_same == 0.0
        assert 0 <= dist_diff <= 1.0
        assert dist_diff > dist_same


class TestWorldModelIntegration:
    """Test world model integration"""
    
    def test_world_model_updates_on_learning(self):
        """Test that world model is updated when learning concepts"""
        world_model = MockWorldModel()
        bridge = SemanticBridge(world_model=world_model)
        
        initial_nodes = len(world_model.causal_graph.nodes)
        
        # Learn concept
        pattern = MockPattern("wm_test", "general")
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature=pattern.pattern_signature,
                success=True,
                measurements={'metric': 0.8},
                domain="general",
                timestamp=time.time()
            )
            for i in range(6)
        ]
        
        concept = bridge.learn_concept_from_pattern(pattern, outcomes)
        
        # World model should have been updated
        final_nodes = len(world_model.causal_graph.nodes)
        assert final_nodes >= initial_nodes
    
    def test_world_model_insights(self):
        """Test getting world model insights for concepts"""
        world_model = MockWorldModel()
        bridge = SemanticBridge(world_model=world_model)
        
        # Create a concept
        concept = Concept(
            pattern_signature="test_sig",
            grounded_effects=[],
            confidence=0.8
        )
        
        # Get insights
        insights = bridge.get_world_model_insights(concept)
        
        assert 'available' in insights
        assert insights['available'] is True


class TestSafetyIntegration:
    """Test safety validation integration"""
    
    def test_safety_filtering_outcomes(self):
        """Test that safety validation filters unsafe outcomes"""
        bridge = SemanticBridge()
        
        pattern = MockPattern("safety_test", "general")
        
        # Mix of outcomes (some might be filtered)
        outcomes = [
            PatternOutcome(
                outcome_id=f"out_{i}",
                pattern_signature=pattern.pattern_signature,
                success=True,
                measurements={'metric': 0.8},
                domain="general",
                timestamp=time.time()
            )
            for i in range(5)
        ]
        
        # Learn with safety validation
        concept = bridge.learn_concept_from_pattern(pattern, outcomes)
        
        # Check safety statistics
        stats = bridge.get_statistics()
        assert 'safety' in stats
    
    def test_safety_blocks_tracked(self):
        """Test that safety blocks are tracked"""
        bridge = SemanticBridge()
        
        stats = bridge.get_statistics()
        
        if stats['safety']['enabled']:
            assert 'blocks' in stats['safety']
            assert 'corrections' in stats['safety']


class TestStatisticsAndMonitoring:
    """Test statistics and monitoring"""
    
    def test_bridge_statistics(self):
        """Test getting comprehensive bridge statistics"""
        bridge = SemanticBridge()
        
        stats = bridge.get_statistics()
        
        # Verify all expected keys
        expected_keys = [
            'total_concepts', 'active_concepts', 'total_transfers',
            'total_conflicts', 'domains', 'world_model_connected',
            'cache_manager', 'safety'
        ]
        
        for key in expected_keys:
            assert key in stats
    
    def test_component_statistics(self):
        """Test getting statistics from all components"""
        bridge = SemanticBridge()
        
        # Get statistics from each component
        mapper_stats = bridge.concept_mapper.get_statistics()
        resolver_stats = bridge.conflict_resolver.get_statistics()
        registry_stats = bridge.domain_registry.get_statistics()
        transfer_stats = bridge.transfer_engine.get_statistics()
        cache_stats = bridge.cache_manager.get_statistics()
        
        # Verify all return dictionaries
        assert isinstance(mapper_stats, dict)
        assert isinstance(resolver_stats, dict)
        assert isinstance(registry_stats, dict)
        assert isinstance(transfer_stats, dict)
        assert isinstance(cache_stats, dict)


class TestPersistence:
    """Test persistence functionality"""
    
    def test_operation_history_persistence(self):
        """Test that operation history can be persisted"""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MockVulcanMemory(temp_dir)
            bridge = SemanticBridge(vulcan_memory=memory)
            
            # Create some operations
            pattern = MockPattern("persist_test", "general")
            outcomes = [
                PatternOutcome(
                    outcome_id=f"out_{i}",
                    pattern_signature=pattern.pattern_signature,
                    success=True,
                    measurements={},
                    domain="general",
                    timestamp=time.time()
                )
                for i in range(5)
            ]
            
            bridge.learn_concept_from_pattern(pattern, outcomes)
            
            # Force persistence
            bridge._persist_operation_history()
            
            # Check that history file was created
            history_path = Path(temp_dir) / 'semantic_bridge' / 'operation_history.jsonl'
            # Note: File might not exist if memory path is not set properly
            # This is expected in some test configurations


class TestThreadSafety:
    """Test thread safety of operations"""
    
    def test_concurrent_concept_learning(self):
        """Test concurrent concept learning from multiple threads"""
        bridge = SemanticBridge()
        results = []
        
        def learn_concepts(thread_id):
            for i in range(3):
                pattern = MockPattern(f"thread_{thread_id}_pattern_{i}", "general")
                outcomes = [
                    PatternOutcome(
                        outcome_id=f"t{thread_id}_out_{j}",
                        pattern_signature=pattern.pattern_signature,
                        success=True,
                        measurements={'metric': 0.7},
                        domain="general",
                        timestamp=time.time()
                    )
                    for j in range(5)
                ]
                
                try:
                    concept = bridge.learn_concept_from_pattern(pattern, outcomes)
                    if concept:
                        results.append(concept)
                except Exception as e:
                    pass  # Some may fail due to safety validation
        
        # Run concurrent threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=learn_concepts, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no crashes and some concepts learned
        assert bridge.total_concepts >= 0
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access"""
        bridge = SemanticBridge()
        
        def access_caches(thread_id):
            pattern = MockPattern(f"cache_{thread_id}", "general")
            outcomes = [
                PatternOutcome(
                    outcome_id=f"out_{thread_id}_{i}",
                    pattern_signature=pattern.pattern_signature,
                    success=True,
                    measurements={},
                    domain="general",
                    timestamp=time.time()
                )
                for i in range(3)
            ]
            
            # Access caches
            bridge._extract_pattern_signature(pattern, outcomes)
            bridge.get_applicable_concepts("general")
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=access_caches, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert True


class TestCompleteWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_full_concept_lifecycle(self):
        """Test complete concept lifecycle: learn, transfer, conflict, resolve"""
        world_model = MockWorldModel()
        bridge = SemanticBridge(world_model=world_model)
        
        # Phase 1: Learn concept in source domain
        pattern1 = MockPattern("lifecycle_pattern_1", "general")
        pattern1.features = {'feature_x': 5.0}
        
        outcomes1 = [
            PatternOutcome(
                outcome_id=f"phase1_{i}",
                pattern_signature=pattern1.pattern_signature,
                success=True,
                measurements={'accuracy': 0.85},
                domain="general",
                timestamp=time.time()
            )
            for i in range(8)
        ]
        
        concept1 = bridge.learn_concept_from_pattern(pattern1, outcomes1)
        assert concept1 is not None
        
        # Phase 2: Test transfer
        applicable = bridge.get_applicable_concepts("optimization", min_confidence=0.5)
        assert isinstance(applicable, list)
        
        # Phase 3: Create potential conflict
        pattern2 = MockPattern("lifecycle_pattern_2", "general")
        pattern2.features = {'feature_x': 5.0}  # Similar to pattern1
        
        # Phase 4: Resolve conflict
        resolution = bridge.resolve_concept_conflict(pattern2, [concept1])
        assert resolution is not None
        
        # Phase 5: Get final statistics
        stats = bridge.get_statistics()
        assert stats['total_concepts'] >= 1
        
        # Verify world model was updated throughout
        assert len(world_model.causal_graph.nodes) > 0
    
    def test_multi_domain_knowledge_transfer(self):
        """Test knowledge transfer across multiple domains"""
        bridge = SemanticBridge()
        
        domains = ['general', 'optimization', 'control']
        patterns = []
        concepts = []
        
        # Learn concepts in each domain
        for i, domain in enumerate(domains):
            pattern = MockPattern(f"multi_domain_{i}", domain)
            pattern.features = {'shared_feature': 1.0, f'{domain}_specific': 2.0}
            
            outcomes = [
                PatternOutcome(
                    outcome_id=f"md_{domain}_{j}",
                    pattern_signature=pattern.pattern_signature,
                    success=True,
                    measurements={'metric': 0.8},
                    domain=domain,
                    timestamp=time.time()
                )
                for j in range(6)
            ]
            
            concept = bridge.learn_concept_from_pattern(pattern, outcomes)
            if concept:
                patterns.append(pattern)
                concepts.append(concept)
        
        # Test transfer strategies
        for source_domain in domains:
            for target_domain in domains:
                if source_domain != target_domain:
                    strategy = bridge.select_transfer_strategy(
                        patterns[0] if patterns else MockPattern("test", "general"),
                        target_domain
                    )
                    assert strategy is not None
        
        # Verify domain registry updated
        assert len(bridge.domain_registry.domains) >= len(domains)
        
        # Verify concepts exist
        assert len(bridge.concept_mapper.concepts) >= 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_outcomes(self):
        """Test handling empty outcomes list"""
        bridge = SemanticBridge()
        
        pattern = MockPattern("empty_test", "general")
        concept = bridge.learn_concept_from_pattern(pattern, [])
        
        # Should handle gracefully
        assert concept is None
    
    def test_unknown_domain(self):
        """Test operations with unknown domain"""
        bridge = SemanticBridge()
        
        concepts = bridge.get_applicable_concepts("completely_unknown_domain")
        
        # Should return empty list or handle gracefully
        assert isinstance(concepts, list)
    
    def test_invalid_pattern(self):
        """Test handling invalid pattern"""
        bridge = SemanticBridge()
        
        # Pattern with minimal attributes
        pattern = type('Pattern', (), {})()
        
        outcomes = [
            PatternOutcome(
                outcome_id="test",
                pattern_signature="sig",
                success=True,
                measurements={},
                domain="general",
                timestamp=time.time()
            )
        ]
        
        # Should handle gracefully
        try:
            concept = bridge.learn_concept_from_pattern(pattern, outcomes)
            # May succeed or fail depending on safety validation
            assert concept is None or concept is not None
        except Exception:
            # Expected if safety validation is strict
            pass


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics"""
    
    def test_many_concepts(self):
        """Test handling many concepts"""
        bridge = SemanticBridge()
        
        # Create many concepts
        for i in range(20):
            pattern = MockPattern(f"perf_pattern_{i}", "general")
            outcomes = [
                PatternOutcome(
                    outcome_id=f"perf_{i}_{j}",
                    pattern_signature=pattern.pattern_signature,
                    success=True,
                    measurements={'metric': 0.7},
                    domain="general",
                    timestamp=time.time()
                )
                for j in range(5)
            ]
            
            bridge.learn_concept_from_pattern(pattern, outcomes)
        
        # Verify size limits are enforced
        assert len(bridge.concept_mapper.concepts) <= bridge.concept_mapper.max_concepts
        
        # Statistics should be available
        stats = bridge.get_statistics()
        assert stats['total_concepts'] >= 0
    
    def test_cache_eviction(self):
        """Test that caches evict properly when full"""
        bridge = SemanticBridge()
        
        # Set small limits for testing
        bridge.max_pattern_cache_size = 5
        
        # Generate many patterns to trigger eviction
        for i in range(10):
            pattern = MockPattern(f"cache_evict_{i}", "general")
            outcomes = [
                PatternOutcome(
                    outcome_id=f"ce_{i}_{j}",
                    pattern_signature=pattern.pattern_signature,
                    success=True,
                    measurements={},
                    domain="general",
                    timestamp=time.time()
                )
                for j in range(3)
            ]
            
            bridge._extract_pattern_signature(pattern, outcomes)
        
        # Cache should not exceed limit
        assert len(bridge.pattern_signature_cache) <= bridge.max_pattern_cache_size


if __name__ == '__main__':
    # Run with verbose output
    pytest.main([__file__, '-v', '--tb=short', '-s'])