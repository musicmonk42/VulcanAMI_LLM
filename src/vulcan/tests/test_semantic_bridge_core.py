"""
test_semantic_bridge_real_imports.py - Verify real implementations are used, not stubs
Part of the VULCAN-AGI system

This test file specifically verifies that SemanticBridge is using the real
implementations of its components, not the fallback stubs.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRealImportsVerification:
    """Verify that real implementations are being used, not stubs"""
    
    def test_concept_mapper_is_real_implementation(self):
        """Test that ConceptMapper is the real implementation"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        from semantic_bridge.concept_mapper import ConceptMapper as RealConceptMapper
        
        bridge = SemanticBridge()
        
        # Check that the concept mapper is the real class
        assert type(bridge.concept_mapper).__name__ == 'ConceptMapper'
        assert isinstance(bridge.concept_mapper, RealConceptMapper)
        
        # Real implementation should have these methods
        assert hasattr(bridge.concept_mapper, 'map_pattern_to_concept')
        assert hasattr(bridge.concept_mapper, 'extract_measurable_effects')
        assert hasattr(bridge.concept_mapper, 'process_pattern_outcomes')
        
        # Real implementation should have these attributes
        assert hasattr(bridge.concept_mapper, 'concepts')
        assert hasattr(bridge.concept_mapper, 'effect_library')
        assert hasattr(bridge.concept_mapper, 'world_model')
        
        print("✓ ConceptMapper is real implementation")
    
    def test_conflict_resolver_is_real_implementation(self):
        """Test that EvidenceWeightedResolver is the real implementation"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        from semantic_bridge.conflict_resolver import EvidenceWeightedResolver as RealResolver
        
        bridge = SemanticBridge()
        
        # Check that the resolver is the real class
        assert type(bridge.conflict_resolver).__name__ == 'EvidenceWeightedResolver'
        assert isinstance(bridge.conflict_resolver, RealResolver)
        
        # Real implementation should have these methods
        assert hasattr(bridge.conflict_resolver, 'resolve_conflict')
        assert hasattr(bridge.conflict_resolver, 'merge_concepts')
        assert hasattr(bridge.conflict_resolver, 'create_concept_variant')
        assert hasattr(bridge.conflict_resolver, 'calculate_evidence_weight')
        
        # Real implementation should have these attributes
        assert hasattr(bridge.conflict_resolver, 'evidence_store')
        assert hasattr(bridge.conflict_resolver, 'resolution_history')
        assert hasattr(bridge.conflict_resolver, 'world_model')
        
        print("✓ EvidenceWeightedResolver is real implementation")
    
    def test_domain_registry_is_real_implementation(self):
        """Test that DomainRegistry is the real implementation"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        from semantic_bridge.domain_registry import DomainRegistry as RealDomainRegistry
        
        bridge = SemanticBridge()
        
        # Check that the domain registry is the real class
        assert type(bridge.domain_registry).__name__ == 'DomainRegistry'
        assert isinstance(bridge.domain_registry, RealDomainRegistry)
        
        # Real implementation should have these methods
        assert hasattr(bridge.domain_registry, 'register_domain')
        assert hasattr(bridge.domain_registry, 'get_domain_hierarchy')
        assert hasattr(bridge.domain_registry, 'calculate_domain_distance')
        assert hasattr(bridge.domain_registry, 'get_similar_domains')
        assert hasattr(bridge.domain_registry, 'get_related_domains')
        
        # Real implementation should have these attributes
        assert hasattr(bridge.domain_registry, 'domains')
        assert hasattr(bridge.domain_registry, 'domain_graph')
        assert hasattr(bridge.domain_registry, 'world_model')
        assert hasattr(bridge.domain_registry, 'distance_cache')
        
        print("✓ DomainRegistry is real implementation")
    
    def test_transfer_engine_is_real_implementation(self):
        """Test that TransferEngine is the real implementation"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        from semantic_bridge.transfer_engine import TransferEngine as RealTransferEngine
        
        bridge = SemanticBridge()
        
        # Check that the transfer engine is the real class
        assert type(bridge.transfer_engine).__name__ == 'TransferEngine'
        assert isinstance(bridge.transfer_engine, RealTransferEngine)
        
        # Real implementation should have these methods
        assert hasattr(bridge.transfer_engine, 'calculate_effect_overlap')
        assert hasattr(bridge.transfer_engine, 'validate_full_transfer')
        assert hasattr(bridge.transfer_engine, 'validate_partial_transfer')
        assert hasattr(bridge.transfer_engine, 'execute_transfer')
        
        # Real implementation should have these attributes
        assert hasattr(bridge.transfer_engine, 'transfer_history')
        assert hasattr(bridge.transfer_engine, 'world_model')
        assert hasattr(bridge.transfer_engine, 'compatibility_cache')
        
        print("✓ TransferEngine is real implementation")
    
    def test_cache_manager_is_real_implementation(self):
        """Test that CacheManager is the real implementation"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        from semantic_bridge.cache_manager import CacheManager as RealCacheManager
        
        bridge = SemanticBridge()
        
        # Check that the cache manager is the real class
        assert type(bridge.cache_manager).__name__ == 'CacheManager'
        assert isinstance(bridge.cache_manager, RealCacheManager)
        
        # Real implementation should have these methods
        assert hasattr(bridge.cache_manager, 'register_cache')
        assert hasattr(bridge.cache_manager, 'check_memory')
        assert hasattr(bridge.cache_manager, 'record_hit')
        assert hasattr(bridge.cache_manager, 'record_miss')
        assert hasattr(bridge.cache_manager, 'get_statistics')
        
        # Real implementation should have these attributes
        assert hasattr(bridge.cache_manager, 'caches')
        assert hasattr(bridge.cache_manager, 'max_memory')
        
        print("✓ CacheManager is real implementation")
    
    def test_concept_class_is_real_implementation(self):
        """Test that Concept class is the real implementation"""
        from semantic_bridge.concept_mapper import Concept
        
        # Create a concept instance
        concept = Concept(
            pattern_signature="test_pattern",
            grounded_effects=[],
            confidence=0.8
        )
        
        # Real implementation should have these attributes
        assert hasattr(concept, 'concept_id')
        assert hasattr(concept, 'pattern_signature')
        assert hasattr(concept, 'grounded_effects')
        assert hasattr(concept, 'confidence')
        assert hasattr(concept, 'domains')
        assert hasattr(concept, 'usage_count')
        assert hasattr(concept, 'success_rate')
        assert hasattr(concept, 'grounding_status')
        assert hasattr(concept, 'evidence_count')
        assert hasattr(concept, 'creation_time')
        
        # Real implementation should have these methods
        assert hasattr(concept, 'update_usage')
        assert hasattr(concept, 'update_evidence')
        assert hasattr(concept, 'calculate_stability_score')
        assert hasattr(concept, 'get_grounding_confidence')
        assert hasattr(concept, 'to_dict')
        
        print("✓ Concept is real implementation")
    
    def test_pattern_outcome_is_real_implementation(self):
        """Test that PatternOutcome class is the real implementation"""
        from semantic_bridge.concept_mapper import PatternOutcome
        import time
        
        # Create a pattern outcome instance
        outcome = PatternOutcome(
            outcome_id="test_001",
            pattern_signature="test_pattern",
            success=True,
            measurements={'accuracy': 0.9},
            domain="general",
            timestamp=time.time()
        )
        
        # Real implementation should have these attributes
        assert hasattr(outcome, 'outcome_id')
        assert hasattr(outcome, 'pattern_signature')
        assert hasattr(outcome, 'success')
        assert hasattr(outcome, 'measurements')
        assert hasattr(outcome, 'domain')
        assert hasattr(outcome, 'timestamp')
        assert hasattr(outcome, 'errors')  # List, not error_message
        assert hasattr(outcome, 'context')
        
        print("✓ PatternOutcome is real implementation")
    
    def test_all_components_receive_world_model(self):
        """Test that world_model is passed to all components"""
        class MockWorldModel:
            def __init__(self):
                self.causal_graph = None
        
        world_model = MockWorldModel()
        
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge(world_model=world_model)
        
        # Verify world_model is passed to main bridge
        assert bridge.world_model is world_model
        
        # Verify world_model is passed to all components
        assert bridge.concept_mapper.world_model is world_model
        assert bridge.conflict_resolver.world_model is world_model
        assert bridge.transfer_engine.world_model is world_model
        assert bridge.domain_registry.world_model is world_model
        
        print("✓ World model correctly passed to all components")
    
    def test_all_components_receive_safety_config(self):
        """Test that safety_config is passed to all components"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        # Use empty config (valid for SafetyConfig)
        safety_config = {}
        bridge = SemanticBridge(safety_config=safety_config)
        
        # Verify all components have safety_validator
        assert hasattr(bridge, 'safety_validator')
        assert hasattr(bridge.concept_mapper, 'safety_validator')
        assert hasattr(bridge.conflict_resolver, 'safety_validator')
        assert hasattr(bridge.transfer_engine, 'safety_validator')
        assert hasattr(bridge.domain_registry, 'safety_validator')
        
        print("✓ Safety config correctly passed to all components")
    
    def test_no_stub_methods_present(self):
        """Test that stub-specific markers are not present"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Check ConceptMapper doesn't have stub signature
        # Stub version only has 3 methods, real has many more
        mapper_methods = [m for m in dir(bridge.concept_mapper) if not m.startswith('_')]
        assert len(mapper_methods) > 5, "ConceptMapper appears to be stub (too few methods)"
        
        # Check DomainRegistry doesn't have stub signature
        # Stub version only has 3 methods, real has many more
        registry_methods = [m for m in dir(bridge.domain_registry) if not m.startswith('_')]
        assert len(registry_methods) > 5, "DomainRegistry appears to be stub (too few methods)"
        
        # Check TransferEngine doesn't have stub signature
        # Stub version only has 2 methods, real has more
        transfer_methods = [m for m in dir(bridge.transfer_engine) if not m.startswith('_')]
        assert len(transfer_methods) > 3, "TransferEngine appears to be stub (too few methods)"
        
        print("✓ No stub implementations detected")
    
    def test_real_implementation_has_size_limits(self):
        """Test that real implementations have proper size limits"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Real ConceptMapper should have size limits
        assert hasattr(bridge.concept_mapper, 'max_concepts')
        assert bridge.concept_mapper.max_concepts > 0
        assert hasattr(bridge.concept_mapper, 'max_effects')
        
        # Real DomainRegistry should have size limits
        assert hasattr(bridge.domain_registry, 'max_domains')
        assert bridge.domain_registry.max_domains > 0
        assert hasattr(bridge.domain_registry, 'max_effect_domains')
        
        # Real CacheManager should have size limits
        assert hasattr(bridge.cache_manager, 'max_memory')
        assert bridge.cache_manager.max_memory > 0
        
        print("✓ Real implementations have proper size limits")
    
    def test_real_implementation_has_threading_locks(self):
        """Test that real implementations have thread safety"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Real implementations should have locks
        assert hasattr(bridge, '_concept_lock')
        assert hasattr(bridge.concept_mapper, '_lock')
        assert hasattr(bridge.domain_registry, '_lock')
        
        print("✓ Real implementations have thread safety locks")
    
    def test_component_statistics_are_detailed(self):
        """Test that real implementations return detailed statistics"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Get statistics from main bridge
        bridge_stats = bridge.get_statistics()
        
        # Real implementation should have many stat fields
        assert len(bridge_stats) > 10
        assert 'total_concepts' in bridge_stats
        assert 'active_concepts' in bridge_stats
        assert 'total_transfers' in bridge_stats
        assert 'cache_manager' in bridge_stats
        assert 'world_model_connected' in bridge_stats
        
        # Component stats should be detailed
        cache_stats = bridge.cache_manager.get_statistics()
        assert isinstance(cache_stats, dict)
        
        print("✓ Real implementations provide detailed statistics")


class TestStubsAreNotUsed:
    """Negative tests - verify stub behavior is NOT present"""
    
    def test_concept_mapper_not_stub(self):
        """Test that ConceptMapper is not the stub version"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Stub only has basic map_pattern_to_concept
        # Real has extract_measurable_effects, process_pattern_outcomes, etc.
        assert hasattr(bridge.concept_mapper, 'extract_measurable_effects')
        assert hasattr(bridge.concept_mapper, 'process_pattern_outcomes')
        assert hasattr(bridge.concept_mapper, 'effect_library')
        
        # Stub doesn't have these
        assert not (
            len([m for m in dir(bridge.concept_mapper) if not m.startswith('_')]) <= 5
        ), "Appears to be stub (too few public methods)"
        
        print("✓ ConceptMapper is definitely not stub")
    
    def test_domain_registry_not_stub(self):
        """Test that DomainRegistry is not the stub version"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Stub only has register_domain, get_related_domains, update_domain_performance
        # Real has many more methods
        assert hasattr(bridge.domain_registry, 'calculate_domain_distance')
        assert hasattr(bridge.domain_registry, 'get_domain_hierarchy')
        assert hasattr(bridge.domain_registry, 'get_similar_domains')
        assert hasattr(bridge.domain_registry, 'merge_domains')
        
        print("✓ DomainRegistry is definitely not stub")
    
    def test_transfer_engine_not_stub(self):
        """Test that TransferEngine is not the stub version"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Stub only has assess_transfer and transfer_concept
        # Real has validate_full_transfer, validate_partial_transfer, execute_transfer, etc.
        assert hasattr(bridge.transfer_engine, 'validate_full_transfer')
        assert hasattr(bridge.transfer_engine, 'validate_partial_transfer')
        assert hasattr(bridge.transfer_engine, 'execute_transfer')
        
        # Real has compatibility_cache
        assert hasattr(bridge.transfer_engine, 'compatibility_cache')
        
        print("✓ TransferEngine is definitely not stub")
    
    def test_conflict_resolver_not_stub(self):
        """Test that EvidenceWeightedResolver is not the stub version"""
        from semantic_bridge.semantic_bridge_core import SemanticBridge
        
        bridge = SemanticBridge()
        
        # Stub only has resolve_conflict with minimal implementation
        # Real has merge_concepts, create_concept_variant, calculate_evidence_weight
        assert hasattr(bridge.conflict_resolver, 'merge_concepts')
        assert hasattr(bridge.conflict_resolver, 'create_concept_variant')
        assert hasattr(bridge.conflict_resolver, 'calculate_evidence_weight')
        assert hasattr(bridge.conflict_resolver, 'evidence_store')
        
        print("✓ EvidenceWeightedResolver is definitely not stub")


class TestImportDiagnostics:
    """Diagnostic tests to verify imports"""
    
    def test_all_modules_importable(self):
        """Test that all semantic_bridge modules can be imported"""
        try:
            from semantic_bridge import concept_mapper
            assert concept_mapper is not None
            print("✓ concept_mapper imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import concept_mapper: {e}")
        
        try:
            from semantic_bridge import conflict_resolver
            assert conflict_resolver is not None
            print("✓ conflict_resolver imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import conflict_resolver: {e}")
        
        try:
            from semantic_bridge import domain_registry
            assert domain_registry is not None
            print("✓ domain_registry imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import domain_registry: {e}")
        
        try:
            from semantic_bridge import transfer_engine
            assert transfer_engine is not None
            print("✓ transfer_engine imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import transfer_engine: {e}")
        
        try:
            from semantic_bridge import cache_manager
            assert cache_manager is not None
            print("✓ cache_manager imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import cache_manager: {e}")
    
    def test_semantic_bridge_core_imports_check(self):
        """Test which classes are actually imported in semantic_bridge_core"""
        import semantic_bridge.semantic_bridge_core as core
        
        # Check what's actually available in the module
        available_classes = [name for name in dir(core) if not name.startswith('_')]
        
        print(f"\n📋 Available classes in semantic_bridge_core: {len(available_classes)}")
        print(f"   Classes: {', '.join(sorted(available_classes)[:20])}")
        
        # Verify key classes are available
        assert 'SemanticBridge' in available_classes
        assert 'ConceptConflict' in available_classes
        assert 'PatternOutcome' in available_classes
        assert 'Concept' in available_classes


if __name__ == '__main__':
    # Run with verbose output to see all the checkmarks
    pytest.main([__file__, '-v', '-s'])
