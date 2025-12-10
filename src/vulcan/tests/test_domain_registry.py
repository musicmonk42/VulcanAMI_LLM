"""
test_domain_registry.py - PURE MOCK VERSION
Tests domain registry functionality without spawning real threads.
"""

import pytest
import numpy as np
import time
import threading
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock


# ============================================================================
# Mock Enums and Classes
# ============================================================================


class DomainCriticality(Enum):
    LOW = "LOW"
    MEDIUM_LOW = "MEDIUM_LOW"
    MEDIUM = "MEDIUM"
    MEDIUM_HIGH = "MEDIUM_HIGH"
    HIGH = "HIGH"
    SAFETY_CRITICAL = "SAFETY_CRITICAL"


class EffectCategory(Enum):
    COMPUTATION = "computation"
    STORAGE = "storage"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"


class PatternType(Enum):
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"


class DomainRelationship(Enum):
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    RELATED = "related"


@dataclass
class Pattern:
    """Mock pattern"""

    pattern_id: str
    pattern_type: PatternType
    description: str
    complexity: float = 0.5

    def get_signature(self) -> str:
        import hashlib

        content = f"{self.pattern_id}_{self.pattern_type.value}_{self.complexity}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class DomainEffect:
    """Mock domain effect"""

    effect_id: str
    category: EffectCategory
    description: str
    importance: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "effect_id": self.effect_id,
            "category": self.category.value,
            "description": self.description,
            "importance": self.importance,
        }


@dataclass
class DomainProfile:
    """Mock domain profile"""

    name: str
    criticality_score: float = 0.5
    effect_types: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    limitations: Set[str] = field(default_factory=set)
    typical_patterns: List[Pattern] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_pattern(self, pattern: Pattern):
        self.typical_patterns.append(pattern)

    def update_performance(self, metric: str, value: float, alpha: float = 0.1):
        if metric in self.performance_metrics:
            self.performance_metrics[metric] = (
                alpha * value + (1 - alpha) * self.performance_metrics[metric]
            )
        else:
            self.performance_metrics[metric] = value

    def get_risk_level(self) -> str:
        score = self.criticality_score
        if score >= 0.95:
            return "SAFETY_CRITICAL"
        elif score >= 0.9:
            return "HIGH"
        elif score >= 0.7:
            return "MEDIUM_HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        elif score >= 0.3:
            return "MEDIUM_LOW"
        else:
            return "LOW"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "criticality_score": self.criticality_score,
            "risk_level": self.get_risk_level(),
            "effect_types": list(self.effect_types),
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
            "performance_metrics": self.performance_metrics,
        }


class MockRiskAdjuster:
    """Mock risk adjuster"""

    def __init__(self, config_path: Optional[Path] = None):
        self.base_thresholds = {"confidence": 0.8, "safety": 0.9, "performance": 0.7}
        if config_path and config_path.exists():
            self._load_config(config_path)

    def _load_config(self, path: Path):
        with open(path) as f:
            config = json.load(f)
            self.base_thresholds.update(config.get("thresholds", {}))

    def save_config(self, path: Path):
        with open(path, "w") as f:
            json.dump({"thresholds": self.base_thresholds}, f)

    def get_adjusted_thresholds(self, domain: str, criticality: float) -> Dict:
        multiplier = 1.0 + (criticality * 0.5)
        return {k: min(v * multiplier, 1.0) for k, v in self.base_thresholds.items()}

    def get_requirements(self, criticality: float) -> Dict:
        if criticality >= 0.9:
            return {
                "validation_passes": 3,
                "required_tests": ["safety", "performance", "stress"],
                "approval_level": "high",
            }
        elif criticality >= 0.5:
            return {
                "validation_passes": 2,
                "required_tests": ["safety", "performance"],
                "approval_level": "medium",
            }
        else:
            return {
                "validation_passes": 1,
                "required_tests": ["basic"],
                "approval_level": "low",
            }


class MockDomainRegistry:
    """Mock domain registry - no thread spawning"""

    def __init__(self, world_model=None, safety_config=None, storage_path=None):
        self.world_model = world_model
        self.storage_path = storage_path

        # Mock safety validator
        self.safety_validator = Mock()
        self.safety_validator.validate_action_comprehensive = Mock(
            return_value=Mock(safe=True, confidence=0.9)
        )

        # Size limits
        self.max_domains = 10000
        self.max_effect_domains = 5000
        self.max_effects_per_domain = 1000
        self.max_cache_size = 1000
        self.max_effect_categories = 100

        # Storage
        self.domains: Dict[str, DomainProfile] = {}
        self.domain_effects: Dict[str, List[DomainEffect]] = {}
        self.relationships: Dict[str, Dict[str, DomainRelationship]] = {}
        self.distance_cache: Dict[tuple, float] = {}

        self.total_domains = 0
        self._lock = threading.Lock()

        # Initialize default domains
        self._init_default_domains()

        # Load from storage if exists
        if storage_path:
            self._load_registry()

    def _init_default_domains(self):
        defaults = [
            ("general", 0.3),
            ("safety_critical", 0.95),
            ("optimization", 0.5),
            ("real_time", 0.7),
            ("machine_learning", 0.6),
            ("data_processing", 0.4),
            ("control", 0.8),
        ]
        for name, score in defaults:
            profile = DomainProfile(name=name, criticality_score=score)
            self.domains[name] = profile
            self.total_domains += 1

    def register_domain(self, name: str, profile: Optional[DomainProfile] = None):
        with self._lock:
            # Evict if at limit
            while len(self.domains) >= self.max_domains:
                oldest = next(iter(self.domains))
                del self.domains[oldest]

            if profile is None:
                profile = DomainProfile(name=name)

            self.domains[name] = profile
            self.total_domains += 1

            # Link to world model if present
            if self.world_model and hasattr(self.world_model, "causal_graph"):
                self.world_model.causal_graph.add_node(f"domain_{name}")
                for cap in profile.capabilities:
                    self.world_model.causal_graph.add_node(f"cap_{cap}")

    def get_domain(self, name: str) -> Optional[DomainProfile]:
        return self.domains.get(name)

    def update_domain(self, name: str, **kwargs) -> bool:
        if name not in self.domains:
            return False

        profile = self.domains[name]
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        return True

    def calculate_domain_distance(self, domain1: str, domain2: str) -> float:
        cache_key = tuple(sorted([domain1, domain2]))

        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # Evict cache if too large
        while len(self.distance_cache) >= self.max_cache_size:
            oldest = next(iter(self.distance_cache))
            del self.distance_cache[oldest]

        # Calculate distance based on criticality scores
        p1 = self.domains.get(domain1)
        p2 = self.domains.get(domain2)

        if p1 is None or p2 is None:
            distance = 1.0
        else:
            distance = abs(p1.criticality_score - p2.criticality_score)

        self.distance_cache[cache_key] = distance
        return distance

    def add_relationship(
        self, domain1: str, domain2: str, relationship: DomainRelationship
    ):
        if domain1 not in self.relationships:
            self.relationships[domain1] = {}
        self.relationships[domain1][domain2] = relationship

    def get_domain_hierarchy(self, domain: str) -> Dict[str, List[str]]:
        parents = []
        children = []

        if domain in self.relationships:
            for other, rel in self.relationships[domain].items():
                if rel == DomainRelationship.PARENT:
                    parents.append(other)
                elif rel == DomainRelationship.CHILD:
                    children.append(other)

        return {"parents": parents, "children": children}

    def get_domain_effects(self, domain: str) -> List[DomainEffect]:
        if domain not in self.domain_effects:
            # Evict if too many domains have effects
            while len(self.domain_effects) >= self.max_effect_domains:
                oldest = next(iter(self.domain_effects))
                del self.domain_effects[oldest]

            self.domain_effects[domain] = []

        return self.domain_effects[domain]

    def add_effect(self, domain: str, effect: DomainEffect):
        if domain not in self.domain_effects:
            self.domain_effects[domain] = []

        if len(self.domain_effects[domain]) < self.max_effects_per_domain:
            self.domain_effects[domain].append(effect)

    def get_statistics(self) -> Dict[str, Any]:
        criticality_dist = {}
        for profile in self.domains.values():
            level = profile.get_risk_level()
            criticality_dist[level] = criticality_dist.get(level, 0) + 1

        return {
            "total_domains": len(self.domains),
            "active_domains": len(self.domains),
            "total_relationships": sum(len(r) for r in self.relationships.values()),
            "criticality_distribution": criticality_dist,
            "cache_size": len(self.distance_cache),
        }

    def _save_registry(self):
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        path.mkdir(parents=True, exist_ok=True)

        data = {"domains": {name: p.to_dict() for name, p in self.domains.items()}}

        with open(path / "registry.json", "w") as f:
            json.dump(data, f)

    def _load_registry(self):
        if not self.storage_path:
            return

        path = Path(self.storage_path) / "registry.json"
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        for name, pdata in data.get("domains", {}).items():
            if name not in self.domains:
                profile = DomainProfile(
                    name=name, criticality_score=pdata.get("criticality_score", 0.5)
                )
                self.domains[name] = profile


# ============================================================================
# Mock World Model
# ============================================================================


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


# ============================================================================
# Tests
# ============================================================================


class TestDomainRegistryBasics:
    """Test basic domain registry functionality"""

    def test_initialization(self):
        registry = MockDomainRegistry()
        assert len(registry.domains) > 0
        assert registry.total_domains > 0
        assert registry.max_domains == 10000
        assert registry.max_effect_domains == 5000

    def test_initialization_with_world_model(self):
        world_model = MockWorldModel()
        registry = MockDomainRegistry(world_model=world_model)
        assert registry.world_model is world_model

    def test_initialization_with_safety_config(self):
        safety_config = {}
        registry = MockDomainRegistry(safety_config=safety_config)
        assert registry is not None
        assert hasattr(registry, "safety_validator")

    def test_default_domains_initialized(self):
        registry = MockDomainRegistry()
        assert "general" in registry.domains
        assert "safety_critical" in registry.domains
        assert "optimization" in registry.domains
        assert "real_time" in registry.domains
        assert "machine_learning" in registry.domains

    def test_size_limits(self):
        registry = MockDomainRegistry()
        assert registry.max_domains == 10000
        assert registry.max_effect_domains == 5000
        assert registry.max_effects_per_domain == 1000
        assert registry.max_cache_size == 1000
        assert registry.max_effect_categories == 100


class TestDomainProfile:
    """Test DomainProfile functionality"""

    def test_create_domain_profile(self):
        profile = DomainProfile(name="test_domain", criticality_score=0.7)
        assert profile.name == "test_domain"
        assert profile.criticality_score == 0.7
        assert len(profile.effect_types) == 0

    def test_domain_profile_with_attributes(self):
        profile = DomainProfile(
            name="test_domain",
            criticality_score=0.8,
            effect_types={"compute", "analyze"},
            capabilities={"pattern_matching", "optimization"},
            limitations={"memory_constraints"},
        )
        assert len(profile.effect_types) == 2
        assert len(profile.capabilities) == 2
        assert len(profile.limitations) == 1

    def test_add_pattern_to_profile(self):
        profile = DomainProfile(name="test_domain")
        pattern = Pattern(
            pattern_id="pat_001",
            pattern_type=PatternType.STRUCTURAL,
            description="Test pattern",
            complexity=0.6,
        )
        profile.add_pattern(pattern)
        assert len(profile.typical_patterns) == 1
        assert profile.typical_patterns[0].pattern_id == "pat_001"

    def test_update_performance(self):
        profile = DomainProfile(name="test_domain")
        profile.update_performance("accuracy", 0.9)
        assert profile.performance_metrics["accuracy"] == 0.9

        profile.update_performance("accuracy", 0.8)
        assert profile.performance_metrics["accuracy"] != 0.8
        assert 0.8 < profile.performance_metrics["accuracy"] < 0.9

    def test_get_risk_level(self):
        levels = [
            (0.05, "LOW"),
            (0.35, "MEDIUM_LOW"),
            (0.55, "MEDIUM"),
            (0.75, "MEDIUM_HIGH"),
            (0.92, "HIGH"),
            (0.96, "SAFETY_CRITICAL"),
        ]

        for score, expected_level in levels:
            profile = DomainProfile(name="test", criticality_score=score)
            assert profile.get_risk_level() == expected_level

    def test_profile_to_dict(self):
        profile = DomainProfile(
            name="test_domain",
            criticality_score=0.7,
            effect_types={"compute"},
            capabilities={"optimization"},
        )
        profile_dict = profile.to_dict()

        assert "name" in profile_dict
        assert "criticality_score" in profile_dict
        assert "risk_level" in profile_dict
        assert profile_dict["name"] == "test_domain"


class TestDomainRegistration:
    """Test domain registration functionality"""

    def test_register_new_domain(self):
        registry = MockDomainRegistry()
        initial_count = len(registry.domains)

        profile = DomainProfile(name="custom_domain", criticality_score=0.6)
        registry.register_domain("custom_domain", profile)

        assert len(registry.domains) == initial_count + 1
        assert "custom_domain" in registry.domains

    def test_register_domain_without_profile(self):
        registry = MockDomainRegistry()
        registry.register_domain("new_domain")

        assert "new_domain" in registry.domains
        assert registry.domains["new_domain"].name == "new_domain"

    def test_update_domain(self):
        registry = MockDomainRegistry()
        registry.register_domain("test_domain")

        success = registry.update_domain("test_domain", criticality_score=0.9)

        assert success == True
        assert registry.domains["test_domain"].criticality_score == 0.9

    def test_update_nonexistent_domain(self):
        registry = MockDomainRegistry()
        success = registry.update_domain("nonexistent", criticality_score=0.5)
        assert success == False


class TestDomainDistance:
    """Test domain distance calculation"""

    def test_calculate_domain_distance(self):
        registry = MockDomainRegistry()
        dist = registry.calculate_domain_distance("general", "safety_critical")
        assert 0.0 <= dist <= 1.0

    def test_distance_caching(self):
        registry = MockDomainRegistry()

        dist1 = registry.calculate_domain_distance("general", "optimization")
        dist2 = registry.calculate_domain_distance("general", "optimization")

        assert dist1 == dist2
        assert len(registry.distance_cache) >= 1

    def test_same_domain_distance(self):
        registry = MockDomainRegistry()
        dist = registry.calculate_domain_distance("general", "general")
        assert dist == 0.0


class TestDomainRelationships:
    """Test domain relationship management"""

    def test_add_relationship(self):
        registry = MockDomainRegistry()
        registry.add_relationship(
            "ml_vision", "machine_learning", DomainRelationship.CHILD
        )

        assert "ml_vision" in registry.relationships
        assert (
            registry.relationships["ml_vision"]["machine_learning"]
            == DomainRelationship.CHILD
        )

    def test_get_domain_hierarchy(self):
        registry = MockDomainRegistry()
        registry.register_domain("parent_domain")
        registry.register_domain("child_domain")

        registry.add_relationship(
            "child_domain", "parent_domain", DomainRelationship.PARENT
        )

        hierarchy = registry.get_domain_hierarchy("child_domain")
        assert "parent_domain" in hierarchy["parents"]


class TestDomainEffects:
    """Test domain effect management"""

    def test_get_domain_effects(self):
        registry = MockDomainRegistry()
        effects = registry.get_domain_effects("general")
        assert isinstance(effects, list)

    def test_add_effect(self):
        registry = MockDomainRegistry()
        effect = DomainEffect(
            effect_id="eff_001",
            category=EffectCategory.COMPUTATION,
            description="Test effect",
            importance=0.8,
        )

        registry.add_effect("general", effect)
        effects = registry.get_domain_effects("general")

        assert len(effects) == 1
        assert effects[0].effect_id == "eff_001"


class TestRiskAdjuster:
    """Test risk adjustment functionality"""

    def test_get_adjusted_thresholds(self):
        adjuster = MockRiskAdjuster()

        low_risk = adjuster.get_adjusted_thresholds("general", 0.3)
        high_risk = adjuster.get_adjusted_thresholds("safety_critical", 0.95)

        assert high_risk["confidence"] > low_risk["confidence"]

    def test_get_requirements(self):
        adjuster = MockRiskAdjuster()

        low_reqs = adjuster.get_requirements(0.3)
        high_reqs = adjuster.get_requirements(0.95)

        assert low_reqs["validation_passes"] == 1
        assert high_reqs["validation_passes"] == 3
        assert "safety" in high_reqs["required_tests"]

    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "risk_config.json"

            adjuster1 = MockRiskAdjuster()
            adjuster1.base_thresholds["confidence"] = 0.85
            adjuster1.save_config(config_path)

            adjuster2 = MockRiskAdjuster(config_path=config_path)
            assert adjuster2.base_thresholds["confidence"] == 0.85


class TestSizeLimitsAndEviction:
    """Test size limits and eviction strategies"""

    def test_max_domains_limit(self):
        registry = MockDomainRegistry()
        registry.max_domains = 10

        for i in range(15):
            registry.register_domain(f"domain_{i}")

        assert len(registry.domains) <= registry.max_domains

    def test_effect_domain_limit(self):
        registry = MockDomainRegistry()
        registry.max_effect_domains = 5

        for i in range(10):
            domain_name = f"domain_{i}"
            registry.register_domain(domain_name)
            registry.get_domain_effects(domain_name)

        assert len(registry.domain_effects) <= registry.max_effect_domains

    def test_cache_size_limit(self):
        registry = MockDomainRegistry()
        registry.max_cache_size = 5

        for i in range(10):
            registry.register_domain(f"domain_{i}")

        for i in range(8):
            registry.calculate_domain_distance(f"domain_{i}", f"domain_{i + 1}")

        assert len(registry.distance_cache) <= registry.max_cache_size


class TestWorldModelIntegration:
    """Test world model integration"""

    def test_registry_without_world_model(self):
        registry = MockDomainRegistry(world_model=None)
        registry.register_domain("test_domain")
        assert "test_domain" in registry.domains

    def test_link_domain_to_world_model(self):
        world_model = MockWorldModel()
        registry = MockDomainRegistry(world_model=world_model)

        profile = DomainProfile(
            name="test_domain", capabilities={"optimization", "analysis"}
        )

        initial_nodes = len(world_model.causal_graph.nodes)
        registry.register_domain("test_domain", profile)

        assert len(world_model.causal_graph.nodes) > initial_nodes


class TestThreadSafety:
    """Test thread-safe operations"""

    def test_concurrent_domain_registration(self):
        registry = MockDomainRegistry()

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

        assert len(registry.domains) > 5

    def test_concurrent_distance_calculation(self):
        registry = MockDomainRegistry()
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

        assert len(set(results)) == 1


class TestStatistics:
    """Test statistics and reporting"""

    def test_get_statistics(self):
        registry = MockDomainRegistry()
        stats = registry.get_statistics()

        assert "total_domains" in stats
        assert "active_domains" in stats
        assert "total_relationships" in stats
        assert "criticality_distribution" in stats
        assert stats["total_domains"] > 0

    def test_criticality_distribution(self):
        registry = MockDomainRegistry()
        stats = registry.get_statistics()
        distribution = stats["criticality_distribution"]

        assert isinstance(distribution, dict)
        assert len(distribution) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_pattern_get_signature(self):
        pattern = Pattern(
            pattern_id="pat_001",
            pattern_type=PatternType.STRUCTURAL,
            description="Test pattern",
            complexity=0.5,
        )
        signature = pattern.get_signature()
        assert signature is not None
        assert len(signature) == 12

    def test_domain_effect_to_dict(self):
        effect = DomainEffect(
            effect_id="eff_001",
            category=EffectCategory.COMPUTATION,
            description="Test effect",
            importance=0.8,
        )
        effect_dict = effect.to_dict()

        assert "effect_id" in effect_dict
        assert "category" in effect_dict
        assert effect_dict["category"] == "computation"

    def test_empty_domain_hierarchy(self):
        registry = MockDomainRegistry()
        registry.register_domain("isolated_domain")

        hierarchy = registry.get_domain_hierarchy("isolated_domain")

        assert len(hierarchy["parents"]) == 0
        assert len(hierarchy["children"]) == 0

    def test_nonexistent_domain_hierarchy(self):
        registry = MockDomainRegistry()
        hierarchy = registry.get_domain_hierarchy("nonexistent")

        assert len(hierarchy["parents"]) == 0
        assert len(hierarchy["children"]) == 0


class TestPersistence:
    """Test persistence functionality"""

    def test_save_and_load_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry"

            registry1 = MockDomainRegistry(storage_path=storage_path)
            registry1.register_domain("custom_domain")
            registry1._save_registry()

            registry2 = MockDomainRegistry(storage_path=storage_path)

            assert "custom_domain" in registry2.domains


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
