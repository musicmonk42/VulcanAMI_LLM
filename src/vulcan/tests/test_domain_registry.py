"""
test_domain_registry.py - Comprehensive tests for DomainRegistry
Part of the VULCAN-AGI system

Tests cover:
- Domain registration and management
- Domain profiles and characteristics
- Domain relationships and hierarchy
- Domain distance calculation
- Adaptive cache sizing
- Effect management
- Risk adjustment
- World model integration
- Safety integration
- Size limits and eviction
"""

import pytest
import numpy as np
import time
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_bridge.domain_registry import (
    DomainRegistry,
    DomainProfile,
    DomainEffect,
    DomainCriticality,
    EffectCategory,
    Pattern,
    PatternType,
    RiskAdjuster,
    DomainRelationship
)


# Mock classes for testing
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


class TestDomainRegistryBasics:
    """Test basic domain registry functionality"""
    
    def test_initialization(self):
        """Test domain registry initialization"""
        registry = DomainRegistry()
        
        assert len(registry.domains) > 0  # Has default domains
        assert registry.total_domains > 0
        assert registry.max_domains == 10000
        assert registry.max_effect_domains == 5000
    
    def test_initialization_with_world_model(self):
        """Test initialization with world model"""
        world_model = MockWorldModel()
        registry = DomainRegistry(world_model=world_model)
        
        assert registry.world_model is world_model
    
    def test_initialization_with_safety_config(self):
        """Test initialization with safety config"""
        safety_config = {'max_risk_score': 0.8}
        registry = DomainRegistry(safety_config=safety_config)
        
        assert registry is not None
        assert hasattr(registry, 'safety_validator')
    
    def test_default_domains_initialized(self):
        """Test default domains are initialized"""
        registry = DomainRegistry()
        
        # Should have default domains
        assert 'general' in registry.domains
        assert 'safety_critical' in registry.domains
        assert 'optimization' in registry.domains
        assert 'real_time' in registry.domains
        assert 'machine_learning' in registry.domains
    
    def test_size_limits(self):
        """Test size limits are properly set"""
        registry = DomainRegistry()
        
        assert registry.max_domains == 10000
        assert registry.max_effect_domains == 5000
        assert registry.max_effects_per_domain == 1000
        assert registry.max_cache_size == 1000
        assert registry.max_effect_categories == 100


class TestDomainProfile:
    """Test DomainProfile functionality"""
    
    def test_create_domain_profile(self):
        """Test creating domain profile"""
        profile = DomainProfile(
            name="test_domain",
            criticality_score=0.7
        )
        
        assert profile.name == "test_domain"
        assert profile.criticality_score == 0.7
        assert len(profile.effect_types) == 0
        assert len(profile.typical_patterns) == 0
    
    def test_domain_profile_with_attributes(self):
        """Test domain profile with attributes"""
        profile = DomainProfile(
            name="test_domain",
            criticality_score=0.8,
            effect_types={"compute", "analyze"},
            capabilities={"pattern_matching", "optimization"},
            limitations={"memory_constraints"}
        )
        
        assert len(profile.effect_types) == 2
        assert len(profile.capabilities) == 2
        assert len(profile.limitations) == 1
    
    def test_add_pattern_to_profile(self):
        """Test adding pattern to domain profile"""
        profile = DomainProfile(name="test_domain")
        
        pattern = Pattern(
            pattern_id="pat_001",
            pattern_type=PatternType.STRUCTURAL,
            description="Test pattern",
            complexity=0.6
        )
        
        profile.add_pattern(pattern)
        
        assert len(profile.typical_patterns) == 1
        assert profile.typical_patterns[0].pattern_id == "pat_001"
    
    def test_update_performance(self):
        """Test updating performance metrics"""
        profile = DomainProfile(name="test_domain")
        
        # Update once
        profile.update_performance('accuracy', 0.9)
        assert profile.performance_metrics['accuracy'] == 0.9
        
        # Update again (should use exponential moving average)
        profile.update_performance('accuracy', 0.8)
        assert profile.performance_metrics['accuracy'] != 0.8
        assert 0.8 < profile.performance_metrics['accuracy'] < 0.9
    
    def test_get_risk_level(self):
        """Test risk level calculation"""
        # FIXED: Test values now align with actual thresholds
        # Thresholds: LOW < 0.3, MEDIUM_LOW >= 0.3, MEDIUM >= 0.5, 
        # MEDIUM_HIGH >= 0.7, HIGH >= 0.9, SAFETY_CRITICAL >= 0.95
        levels = [
            (0.05, "LOW"),
            (0.35, "MEDIUM_LOW"),
            (0.55, "MEDIUM"),
            (0.75, "MEDIUM_HIGH"),
            (0.92, "HIGH"),
            (0.96, "SAFETY_CRITICAL")
        ]
        
        for score, expected_level in levels:
            profile = DomainProfile(
                name="test",
                criticality_score=score
            )
            assert profile.get_risk_level() == expected_level
    
    def test_profile_to_dict(self):
        """Test converting profile to dictionary"""
        profile = DomainProfile(
            name="test_domain",
            criticality_score=0.7,
            effect_types={"compute"},
            capabilities={"optimization"}
        )
        
        profile_dict = profile.to_dict()
        
        assert 'name' in profile_dict
        assert 'criticality_score' in profile_dict
        assert 'risk_level' in profile_dict
        assert 'effect_types' in profile_dict
        assert profile_dict['name'] == "test_domain"


class TestDomainRegistration:
    """Test domain registration functionality"""
    
    def test_register_new_domain(self):
        """Test registering a new domain"""
        registry = DomainRegistry()
        initial_count = len(registry.domains)
        
        profile = DomainProfile(
            name="custom_domain",
            criticality_score=0.6
        )
        
        registry.register_domain("custom_domain", profile)
        
        assert len(registry.domains) == initial_count + 1
        assert "custom_domain" in registry.domains
    
    def test_register_domain_without_profile(self):
        """Test registering domain without explicit profile"""
        registry = DomainRegistry()
        
        registry.register_domain("new_domain")
        
        assert "new_domain" in registry.domains
        assert registry.domains["new_domain"].name == "new_domain"
    
    def test_register_domain_with_characteristics(self):
        """Test registering domain with characteristics dict"""
        registry = DomainRegistry()
        
        characteristics = {
            'adaptability': 'high',
            'complexity': 'medium',
            'criticality': 'high'
        }
        
        registry.register_domain("test_domain", characteristics=characteristics)
        
        assert "test_domain" in registry.domains
        profile = registry.domains["test_domain"]
        assert 'adaptability' in profile.metadata
        assert profile.criticality_score == DomainCriticality.HIGH.value
    
    def test_register_duplicate_domain(self):
        """Test registering domain with existing name"""
        registry = DomainRegistry()
        
        profile1 = DomainProfile(
            name="duplicate",
            criticality_score=0.5
        )
        
        profile2 = DomainProfile(
            name="duplicate",
            criticality_score=0.8
        )
        
        registry.register_domain("duplicate", profile1)
        registry.register_domain("duplicate", profile2)
        
        # Should update to new profile
        assert registry.domains["duplicate"].criticality_score == 0.8
    
    def test_criticality_score_validation(self):
        """Test criticality score is validated and clamped"""
        registry = DomainRegistry()
        
        # Test out of range criticality
        profile = DomainProfile(
            name="test",
            criticality_score=1.5  # Invalid
        )
        
        registry.register_domain("test", profile)
        
        # Should be clamped to valid range
        assert 0 <= registry.domains["test"].criticality_score <= 1


class TestDomainRelationships:
    """Test domain relationship management"""
    
    def test_add_domain_relationship(self):
        """Test adding relationship between domains"""
        registry = DomainRegistry()
        
        registry.register_domain("parent_domain")
        registry.register_domain("child_domain")
        
        initial_rel_count = len(registry.relationships)
        
        registry.add_domain_relationship(
            "parent_domain",
            "child_domain",
            "parent",
            strength=0.8
        )
        
        assert len(registry.relationships) > initial_rel_count
    
    def test_parent_child_relationship(self):
        """Test parent-child relationship updates profiles"""
        registry = DomainRegistry()
        
        registry.register_domain("parent")
        registry.register_domain("child")
        
        registry.add_domain_relationship("parent", "child", "parent", 0.9)
        
        # Check parent profile
        assert "child" in registry.domains["parent"].child_domains
        
        # Check child profile
        assert "parent" in registry.domains["child"].parent_domains
    
    def test_get_domain_hierarchy(self):
        """Test getting domain hierarchy"""
        registry = DomainRegistry()
        
        # Create hierarchy
        registry.register_domain("grandparent")
        registry.register_domain("parent")
        registry.register_domain("child")
        
        registry.add_domain_relationship("grandparent", "parent", "parent")
        registry.add_domain_relationship("parent", "child", "parent")
        
        hierarchy = registry.get_domain_hierarchy("child")
        
        assert 'parents' in hierarchy
        assert 'children' in hierarchy
        assert 'ancestors' in hierarchy
        assert 'descendants' in hierarchy
        assert "parent" in hierarchy['parents']
    
    def test_get_related_domains(self):
        """Test getting related domains"""
        registry = DomainRegistry()
        
        registry.register_domain("domain_a")
        registry.register_domain("domain_b")
        registry.register_domain("domain_c")
        
        registry.add_domain_relationship("domain_a", "domain_b", "related")
        
        related = registry.get_related_domains("domain_a")
        
        assert isinstance(related, list)
        # Should not include self
        assert "domain_a" not in related


class TestDomainDistance:
    """Test domain distance calculation"""
    
    def test_distance_same_domain(self):
        """Test distance for same domain is zero"""
        registry = DomainRegistry()
        
        distance = registry.calculate_domain_distance("general", "general")
        
        assert distance == 0.0
    
    def test_distance_different_domains(self):
        """Test distance between different domains"""
        registry = DomainRegistry()
        
        distance = registry.calculate_domain_distance(
            "general",
            "safety_critical"
        )
        
        assert 0 < distance <= 1.0
    
    def test_distance_caching(self):
        """Test distance results are cached"""
        registry = DomainRegistry()
        
        # First call - cache miss
        initial_cache_size = len(registry.distance_cache)
        distance1 = registry.calculate_domain_distance("general", "optimization")
        
        # Cache should have grown
        assert len(registry.distance_cache) > initial_cache_size
        
        # Second call - cache hit
        distance2 = registry.calculate_domain_distance("general", "optimization")
        
        # Should return same distance
        assert distance1 == distance2
    
    def test_distance_unknown_domain(self):
        """Test distance with unknown domain"""
        registry = DomainRegistry()
        
        distance = registry.calculate_domain_distance(
            "general",
            "nonexistent_domain"
        )
        
        # Should return maximum distance
        assert distance == 1.0
    
    def test_get_similar_domains(self):
        """Test getting similar domains"""
        registry = DomainRegistry()
        
        similar = registry.get_similar_domains("general", top_k=3)
        
        assert len(similar) <= 3
        # Should be list of tuples (domain_name, similarity_score)
        if len(similar) > 0:
            assert isinstance(similar[0], tuple)
            assert len(similar[0]) == 2
            assert isinstance(similar[0][0], str)
            assert isinstance(similar[0][1], float)
            # Similarity scores should be sorted descending
            if len(similar) > 1:
                assert similar[0][1] >= similar[1][1]


class TestAdaptiveCache:
    """Test adaptive cache sizing"""
    
    def test_cache_hit_tracking(self):
        """Test cache hit tracking"""
        registry = DomainRegistry()
        
        # Generate cache hits
        for _ in range(3):
            registry.calculate_domain_distance("general", "optimization")
        
        assert registry.cache_hit_count > 0
    
    def test_cache_miss_tracking(self):
        """Test cache miss tracking"""
        registry = DomainRegistry()
        
        initial_misses = registry.cache_miss_count
        
        # Generate cache miss
        registry.calculate_domain_distance("general", "optimization")
        
        assert registry.cache_miss_count > initial_misses
    
    def test_adaptive_cache_resize(self):
        """Test adaptive cache resizing based on hit rate"""
        registry = DomainRegistry()
        registry.max_cache_size = 100
        
        # Simulate high hit rate
        registry.cache_hit_count = 90
        registry.cache_miss_count = 10
        
        initial_cache_size = registry.max_cache_size
        registry._adaptive_cache_resize()
        
        # Should increase cache size with high hit rate
        assert registry.max_cache_size >= initial_cache_size


class TestDomainEffects:
    """Test domain effect management"""
    
    def test_get_domain_effects(self):
        """Test getting effects for a domain"""
        registry = DomainRegistry()
        
        effects = registry.get_domain_effects("general")
        
        assert isinstance(effects, list)
        if len(effects) > 0:
            assert isinstance(effects[0], DomainEffect)
    
    def test_effects_generated_from_profile(self):
        """Test effects are generated from domain profile"""
        registry = DomainRegistry()
        
        profile = DomainProfile(
            name="test_domain",
            effect_types={"compute", "analyze", "transform"}
        )
        
        registry.register_domain("test_domain", profile)
        
        effects = registry.get_domain_effects("test_domain")
        
        # Should have effects for each effect type
        assert len(effects) >= 3
    
    def test_effects_from_patterns(self):
        """Test effects are generated from patterns"""
        registry = DomainRegistry()
        
        profile = DomainProfile(name="test_domain")
        pattern = Pattern(
            pattern_id="pat_001",
            pattern_type=PatternType.STRUCTURAL,
            description="Test pattern"
        )
        profile.add_pattern(pattern)
        
        registry.register_domain("test_domain", profile)
        
        effects = registry.get_domain_effects("test_domain")
        
        # Should have effect from pattern
        assert len(effects) > 0
    
    def test_effects_caching(self):
        """Test effects are cached after first generation"""
        registry = DomainRegistry()
        
        # First call generates effects
        effects1 = registry.get_domain_effects("general")
        
        # Second call should return cached effects
        effects2 = registry.get_domain_effects("general")
        
        # Should be same reference (cached)
        assert effects1 is effects2
    
    def test_effect_statistics_tracking(self):
        """Test effect statistics are tracked"""
        registry = DomainRegistry()
        
        initial_stats_count = len(registry.effect_statistics)
        
        # Generate effects
        registry.get_domain_effects("general")
        
        # Should have updated statistics
        assert len(registry.effect_statistics) >= initial_stats_count


class TestDomainManagement:
    """Test domain management operations"""
    
    def test_update_domain_criticality(self):
        """Test updating domain criticality"""
        registry = DomainRegistry()
        
        registry.register_domain("test_domain")
        
        initial_criticality = registry.domains["test_domain"].criticality_score
        
        registry.update_domain_criticality("test_domain", 0.9)
        
        assert registry.domains["test_domain"].criticality_score == 0.9
        assert registry.domains["test_domain"].criticality_score != initial_criticality
    
    def test_update_domain_performance(self):
        """Test updating domain performance"""
        registry = DomainRegistry()
        
        registry.register_domain("test_domain")
        
        # Update with success
        registry.update_domain_performance("test_domain", success=True)
        
        profile = registry.domains["test_domain"]
        assert 'success_rate' in profile.performance_metrics
        assert 'usage_count' in profile.performance_metrics
        assert profile.performance_metrics['usage_count'] == 1
    
    def test_update_performance_nonexistent_domain(self):
        """Test updating performance for nonexistent domain"""
        registry = DomainRegistry()
        
        # Should not raise error
        registry.update_domain_performance("nonexistent", success=True)
    
    def test_merge_domains(self):
        """Test merging two domains"""
        registry = DomainRegistry()
        
        profile_a = DomainProfile(
            name="domain_a",
            criticality_score=0.6,
            effect_types={"compute"},
            capabilities={"optimization"}
        )
        
        profile_b = DomainProfile(
            name="domain_b",
            criticality_score=0.8,
            effect_types={"analyze"},
            capabilities={"pattern_matching"}
        )
        
        registry.register_domain("domain_a", profile_a)
        registry.register_domain("domain_b", profile_b)
        
        merged = registry.merge_domains("domain_a", "domain_b", "merged_domain")
        
        assert merged is not None
        assert "merged_domain" in registry.domains
        # Should have combined effect types
        assert len(merged.effect_types) == 2
        # Should have max criticality
        assert merged.criticality_score == 0.8
    
    def test_merge_nonexistent_domains(self):
        """Test merging nonexistent domains raises error"""
        registry = DomainRegistry()
        
        with pytest.raises(ValueError):
            registry.merge_domains("nonexistent_a", "nonexistent_b")


class TestRiskAdjuster:
    """Test RiskAdjuster functionality"""
    
    def test_risk_adjuster_initialization(self):
        """Test risk adjuster initialization"""
        adjuster = RiskAdjuster()
        
        assert hasattr(adjuster, 'base_thresholds')
        assert hasattr(adjuster, 'criticality_multipliers')
        assert 'confidence' in adjuster.base_thresholds
    
    def test_get_dynamic_thresholds(self):
        """Test getting dynamic thresholds"""
        adjuster = RiskAdjuster()
        
        low_risk_profile = DomainProfile(
            name="low_risk",
            criticality_score=0.2
        )
        
        high_risk_profile = DomainProfile(
            name="high_risk",
            criticality_score=0.9
        )
        
        low_thresholds = adjuster.get_dynamic_thresholds(low_risk_profile)
        high_thresholds = adjuster.get_dynamic_thresholds(high_risk_profile)
        
        # High risk should have stricter thresholds
        assert high_thresholds['confidence'] > low_thresholds['confidence']
        assert high_thresholds['validation'] > low_thresholds['validation']
    
    def test_calculate_safety_margin(self):
        """Test safety margin calculation"""
        adjuster = RiskAdjuster()
        
        # Test different criticality levels
        margins = [
            (0.1, 0.0),  # Low criticality
            (0.5, 0.05),  # Medium
            (0.7, 0.1),   # Medium-high
            (0.9, 0.15),  # High
            (0.96, 0.2)   # Safety critical
        ]
        
        for criticality, expected_margin in margins:
            margin = adjuster.calculate_safety_margin(criticality)
            assert margin == expected_margin
    
    def test_assess_risk(self):
        """Test risk assessment"""
        adjuster = RiskAdjuster()
        
        profile = DomainProfile(
            name="test",
            criticality_score=0.9
        )
        
        assessment = adjuster.assess_risk(profile, "transfer")
        
        assert 'risk_level' in assessment
        assert 'criticality_score' in assessment
        assert 'requires_validation' in assessment
        assert 'recommended_actions' in assessment
        assert assessment['risk_level'] == "HIGH"
    
    def test_get_validation_requirements(self):
        """Test getting validation requirements"""
        adjuster = RiskAdjuster()
        
        # Low criticality
        low_reqs = adjuster.get_validation_requirements(0.3)
        assert low_reqs['min_test_coverage'] == 0.7
        assert low_reqs['validation_passes'] == 1
        
        # Safety critical
        high_reqs = adjuster.get_validation_requirements(0.95)
        assert high_reqs['min_test_coverage'] == 0.95
        assert high_reqs['validation_passes'] == 3
        assert 'safety' in high_reqs['required_tests']
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "risk_config.json"
            
            # Create and save config
            adjuster1 = RiskAdjuster()
            adjuster1.base_thresholds['confidence'] = 0.85
            adjuster1.save_config(config_path)
            
            # Load config
            adjuster2 = RiskAdjuster(config_path=config_path)
            
            assert adjuster2.base_thresholds['confidence'] == 0.85


class TestSizeLimitsAndEviction:
    """Test size limits and eviction strategies"""
    
    def test_max_domains_limit(self):
        """Test maximum domains limit is enforced"""
        registry = DomainRegistry()
        registry.max_domains = 10  # Small limit for testing
        
        # Create more domains than limit
        for i in range(15):
            registry.register_domain(f"domain_{i}")
        
        # Should not exceed limit
        assert len(registry.domains) <= registry.max_domains
    
    def test_effect_domain_limit(self):
        """Test effect domain storage limit"""
        registry = DomainRegistry()
        registry.max_effect_domains = 5
        
        # Generate effects for more domains than limit
        for i in range(10):
            domain_name = f"domain_{i}"
            registry.register_domain(domain_name)
            registry.get_domain_effects(domain_name)
        
        # Should not exceed limit
        assert len(registry.domain_effects) <= registry.max_effect_domains
    
    def test_cache_size_limit(self):
        """Test distance cache size limit"""
        registry = DomainRegistry()
        registry.max_cache_size = 5
        
        # Register domains
        for i in range(10):
            registry.register_domain(f"domain_{i}")
        
        # Calculate distances
        for i in range(8):
            registry.calculate_domain_distance(f"domain_{i}", f"domain_{i+1}")
        
        # Should not exceed cache limit
        assert len(registry.distance_cache) <= registry.max_cache_size


class TestWorldModelIntegration:
    """Test world model integration"""
    
    def test_registry_without_world_model(self):
        """Test registry works without world model"""
        registry = DomainRegistry(world_model=None)
        
        # Should work fine
        registry.register_domain("test_domain")
        assert "test_domain" in registry.domains
    
    def test_link_domain_to_world_model(self):
        """Test linking domain to world model"""
        world_model = MockWorldModel()
        registry = DomainRegistry(world_model=world_model)
        
        profile = DomainProfile(
            name="test_domain",
            capabilities={"optimization", "analysis"}
        )
        
        initial_nodes = len(world_model.causal_graph.nodes)
        
        registry.register_domain("test_domain", profile)
        
        # Should have added nodes to world model
        assert len(world_model.causal_graph.nodes) > initial_nodes


class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_domain_registration(self):
        """Test concurrent domain registration"""
        registry = DomainRegistry()
        
        def register_domains(thread_id):
            for i in range(5):
                registry.register_domain(f"domain_{thread_id}_{i}")
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=register_domains, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have registered domains from all threads
        # At least some should be present (may evict due to limits)
        assert len(registry.domains) > 5
    
    def test_concurrent_distance_calculation(self):
        """Test concurrent distance calculation"""
        registry = DomainRegistry()
        
        results = []
        
        def calculate_distances(thread_id):
            for _ in range(5):
                dist = registry.calculate_domain_distance("general", "optimization")
                results.append(dist)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=calculate_distances, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All distances should be same value
        assert len(set(results)) == 1


class TestStatistics:
    """Test statistics and reporting"""
    
    def test_get_statistics_empty(self):
        """Test getting statistics"""
        registry = DomainRegistry()
        
        stats = registry.get_statistics()
        
        assert 'total_domains' in stats
        assert 'active_domains' in stats
        assert 'total_relationships' in stats
        assert 'criticality_distribution' in stats
        assert stats['total_domains'] > 0  # Has defaults
    
    def test_criticality_distribution(self):
        """Test criticality distribution calculation"""
        registry = DomainRegistry()
        
        stats = registry.get_statistics()
        distribution = stats['criticality_distribution']
        
        # Should have some distribution
        assert isinstance(distribution, dict)
        assert len(distribution) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_pattern_get_signature(self):
        """Test pattern signature generation"""
        pattern = Pattern(
            pattern_id="pat_001",
            pattern_type=PatternType.STRUCTURAL,
            description="Test pattern",
            complexity=0.5
        )
        
        signature = pattern.get_signature()
        
        assert signature is not None
        assert len(signature) == 12  # MD5 hash truncated
    
    def test_domain_effect_to_dict(self):
        """Test domain effect serialization"""
        effect = DomainEffect(
            effect_id="eff_001",
            category=EffectCategory.COMPUTATION,
            description="Test effect",
            importance=0.8
        )
        
        effect_dict = effect.to_dict()
        
        assert 'effect_id' in effect_dict
        assert 'category' in effect_dict
        assert 'description' in effect_dict
        assert effect_dict['category'] == 'computation'
    
    def test_empty_domain_hierarchy(self):
        """Test hierarchy for domain without relationships"""
        registry = DomainRegistry()
        
        registry.register_domain("isolated_domain")
        
        hierarchy = registry.get_domain_hierarchy("isolated_domain")
        
        assert len(hierarchy['parents']) == 0
        assert len(hierarchy['children']) == 0
    
    def test_nonexistent_domain_hierarchy(self):
        """Test hierarchy for nonexistent domain"""
        registry = DomainRegistry()
        
        hierarchy = registry.get_domain_hierarchy("nonexistent")
        
        # Should return empty hierarchy
        assert len(hierarchy['parents']) == 0
        assert len(hierarchy['children']) == 0


class TestPersistence:
    """Test persistence functionality"""
    
    def test_save_and_load_registry(self):
        """Test saving and loading registry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry"
            
            # Create and save registry
            registry1 = DomainRegistry(storage_path=storage_path)
            registry1.register_domain("custom_domain")
            registry1._save_registry()
            
            # Load registry
            registry2 = DomainRegistry(storage_path=storage_path)
            
            # Should have loaded custom domain
            assert "custom_domain" in registry2.domains


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])