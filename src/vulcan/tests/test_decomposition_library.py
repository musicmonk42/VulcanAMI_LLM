"""
test_decomposition_library.py - Comprehensive tests for decomposition library
Part of the VULCAN-AGI system

Tests:
- Pattern storage and retrieval
- Principle management
- Performance tracking
- Domain stratification
- Similarity search
- Cross-domain patterns
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components to test
from problem_decomposer.decomposition_library import (
    Context, DecompositionLibrary, DecompositionPrinciple, DomainCategory,
    Pattern, PatternPerformance, PatternStatus, StratifiedDecompositionLibrary)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def temp_storage():
    """Create temporary storage directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def library(temp_storage):
    """Create basic library instance"""
    return DecompositionLibrary(storage_path=temp_storage / "library")


@pytest.fixture
def stratified_library(temp_storage):
    """Create stratified library instance"""
    return StratifiedDecompositionLibrary(storage_path=temp_storage / "stratified")


@pytest.fixture
def sample_pattern():
    """Create sample pattern"""
    return Pattern(
        pattern_id="test_pattern_1",
        structure=None,
        features={
            "node_count": 5,
            "edge_count": 4,
            "density": 0.4,
            "type": "hierarchical",
        },
        metadata={"created": "test"},
    )


@pytest.fixture
def sample_context():
    """Create sample context"""
    return Context(
        domain="optimization", problem_type="continuous", constraints={"bounded": True}
    )


@pytest.fixture
def sample_principle(sample_pattern, sample_context):
    """Create sample principle"""
    return DecompositionPrinciple(
        principle_id="test_principle_1",
        name="Test Hierarchical Decomposition",
        pattern=sample_pattern,
        applicable_contexts=[sample_context],
        success_rate=0.7,
        contraindications=["flat_structure"],
    )


# ============================================================
# PATTERN TESTS
# ============================================================


class TestPattern:
    """Test Pattern class"""

    def test_pattern_creation(self, sample_pattern):
        """Test pattern creation"""
        assert sample_pattern.pattern_id == "test_pattern_1"
        assert sample_pattern.features["node_count"] == 5
        assert sample_pattern.features["edge_count"] == 4

        logger.info("✓ Pattern creation test passed")

    def test_pattern_signature(self, sample_pattern):
        """Test pattern signature generation"""
        signature = sample_pattern.get_signature()

        assert signature is not None
        assert isinstance(signature, str)
        assert len(signature) == 32  # MD5 hash length

        # Same pattern should give same signature
        signature2 = sample_pattern.get_signature()
        assert signature == signature2

        logger.info(f"✓ Pattern signature test passed: {signature}")

    def test_pattern_signature_consistency(self):
        """Test signature consistency across identical patterns"""
        pattern1 = Pattern(
            pattern_id="p1", structure=None, features={"node_count": 3, "edge_count": 2}
        )

        pattern2 = Pattern(
            pattern_id="p2",  # Different ID
            structure=None,
            features={"node_count": 3, "edge_count": 2},  # Same features
        )

        # Same features should give same signature
        assert pattern1.get_signature() == pattern2.get_signature()

        logger.info("✓ Pattern signature consistency test passed")


# ============================================================
# CONTEXT TESTS
# ============================================================


class TestContext:
    """Test Context class"""

    def test_context_creation(self, sample_context):
        """Test context creation"""
        assert sample_context.domain == "optimization"
        assert sample_context.problem_type == "continuous"
        assert sample_context.constraints["bounded"] == True

        logger.info("✓ Context creation test passed")

    def test_context_matching_identical(self, sample_context):
        """Test context matching with identical context"""
        other = Context(
            domain="optimization",
            problem_type="continuous",
            constraints={"bounded": True},
        )

        score = sample_context.matches(other)

        assert score >= 0.8  # Should be high match

        logger.info(f"✓ Identical context matching test passed: score={score:.2f}")

    def test_context_matching_different_domain(self, sample_context):
        """Test context matching with different domain"""
        other = Context(
            domain="classification",
            problem_type="continuous",
            constraints={"bounded": True},
        )

        score = sample_context.matches(other)

        assert score < 0.8  # Should be lower match

        logger.info(f"✓ Different domain matching test passed: score={score:.2f}")

    def test_context_matching_general_domain(self, sample_context):
        """Test context matching with general domain"""
        other = Context(domain="general", problem_type="continuous", constraints={})

        score = sample_context.matches(other)

        assert score > 0.3  # General should match somewhat

        logger.info(f"✓ General domain matching test passed: score={score:.2f}")


# ============================================================
# PRINCIPLE TESTS
# ============================================================


class TestDecompositionPrinciple:
    """Test DecompositionPrinciple class"""

    def test_principle_creation(self, sample_principle):
        """Test principle creation"""
        assert sample_principle.principle_id == "test_principle_1"
        assert sample_principle.name == "Test Hierarchical Decomposition"
        assert sample_principle.success_rate == 0.7

        logger.info("✓ Principle creation test passed")

    def test_principle_update_success_rate(self, sample_principle):
        """Test success rate updates"""
        sample_principle.success_count = 7
        sample_principle.failure_count = 3

        sample_principle.update_success_rate()

        assert sample_principle.success_rate == 0.7

        logger.info(
            f"✓ Success rate update test passed: {sample_principle.success_rate:.2f}"
        )

    def test_principle_applicability(self, sample_principle, sample_context):
        """Test principle applicability checking"""
        is_applicable, match_score = sample_principle.is_applicable(sample_context)

        assert is_applicable == True
        assert match_score > 0.5

        logger.info(
            f"✓ Applicability test passed: applicable={is_applicable}, score={match_score:.2f}"
        )

    def test_principle_contraindications(self, sample_principle):
        """Test contraindication checking"""
        # Context with contraindication
        bad_context = Context(
            domain="flat_structure", problem_type="continuous", constraints={}
        )

        is_applicable, match_score = sample_principle.is_applicable(bad_context)

        assert is_applicable == False

        logger.info("✓ Contraindication test passed")


# ============================================================
# LIBRARY BASIC TESTS
# ============================================================


class TestDecompositionLibrary:
    """Test DecompositionLibrary class"""

    def test_library_initialization(self, library):
        """Test library initialization"""
        assert library is not None
        assert hasattr(library, "patterns")
        assert hasattr(library, "principles")
        assert hasattr(library, "performance")

        logger.info("✓ Library initialization test passed")

    def test_add_principle(self, library, sample_principle):
        """Test adding principle to library"""
        library.add_principle(sample_principle)

        assert sample_principle.principle_id in library.principles
        assert library.total_principles == 1

        # Pattern should also be added
        assert len(library.patterns) > 0

        logger.info("✓ Add principle test passed")

    def test_get_principle(self, library, sample_principle):
        """Test retrieving principle"""
        library.add_principle(sample_principle)

        retrieved = library.get_principle(sample_principle.principle_id)

        assert retrieved is not None
        assert retrieved.principle_id == sample_principle.principle_id

        logger.info("✓ Get principle test passed")

    def test_get_nonexistent_principle(self, library):
        """Test retrieving nonexistent principle"""
        retrieved = library.get_principle("nonexistent")

        assert retrieved is None

        logger.info("✓ Get nonexistent principle test passed")

    def test_get_applicable_principles(self, library, sample_principle, sample_context):
        """Test getting applicable principles"""
        library.add_principle(sample_principle)

        applicable = library.get_applicable_principles(sample_context)

        assert len(applicable) > 0
        assert sample_principle in applicable

        logger.info(f"✓ Get applicable principles test passed: {len(applicable)} found")


# ============================================================
# PATTERN PERFORMANCE TESTS
# ============================================================


class TestPatternPerformance:
    """Test PatternPerformance tracking"""

    def test_performance_initialization(self):
        """Test performance object initialization"""
        perf = PatternPerformance(pattern_signature="test_sig")

        assert perf.pattern_signature == "test_sig"
        assert perf.total_uses == 0
        assert perf.successful_uses == 0
        assert perf.failed_uses == 0

        logger.info("✓ Performance initialization test passed")

    def test_performance_update_success(self):
        """Test updating performance with success"""
        perf = PatternPerformance(pattern_signature="test_sig")

        perf.update(success=True, execution_time=1.5, domain="optimization")

        assert perf.total_uses == 1
        assert perf.successful_uses == 1
        assert perf.failed_uses == 0
        assert perf.last_performance == 1.0
        assert "optimization" in perf.domains_used

        logger.info("✓ Performance success update test passed")

    def test_performance_update_failure(self):
        """Test updating performance with failure"""
        perf = PatternPerformance(pattern_signature="test_sig")

        perf.update(success=False, execution_time=0.5, failure_reason="timeout")

        assert perf.total_uses == 1
        assert perf.successful_uses == 0
        assert perf.failed_uses == 1
        assert perf.last_performance == 0.0
        assert len(perf.failure_reasons) == 1

        logger.info("✓ Performance failure update test passed")

    def test_performance_success_rate(self):
        """Test success rate calculation"""
        perf = PatternPerformance(pattern_signature="test_sig")

        # Add some successes and failures
        for _ in range(7):
            perf.update(success=True, execution_time=1.0)
        for _ in range(3):
            perf.update(success=False, execution_time=1.0)

        success_rate = perf.get_success_rate()

        assert success_rate == 0.7

        logger.info(f"✓ Success rate calculation test passed: {success_rate:.2f}")

    def test_performance_failure_reasons_limit(self):
        """Test failure reasons list limiting"""
        perf = PatternPerformance(pattern_signature="test_sig")

        # Add many failures
        for i in range(150):
            perf.update(success=False, execution_time=1.0, failure_reason=f"error_{i}")

        # Should be limited to 100
        assert len(perf.failure_reasons) == 100

        logger.info("✓ Failure reasons limiting test passed")


# ============================================================
# SIMILARITY SEARCH TESTS
# ============================================================


class TestSimilaritySearch:
    """Test pattern similarity search"""

    def test_find_similar_basic(self, library):
        """Test basic similarity search"""
        # Add some patterns
        for i in range(5):
            pattern = Pattern(
                pattern_id=f"pattern_{i}",
                structure=None,
                features={
                    "node_count": 5 + i,
                    "edge_count": 4 + i,
                    "density": 0.4 + i * 0.1,
                },
            )
            library._add_pattern(pattern)

        # Search for similar
        query_pattern = Pattern(
            pattern_id="query",
            structure=None,
            features={"node_count": 6, "edge_count": 5, "density": 0.5},
        )

        similar = library.find_similar(query_pattern, top_k=3)

        assert len(similar) <= 3
        assert all(isinstance(s, tuple) for s in similar)
        assert all(len(s) == 2 for s in similar)

        logger.info(
            f"✓ Basic similarity search test passed: found {len(similar)} similar patterns"
        )

    def test_find_similar_empty_library(self, library):
        """Test similarity search in empty library"""
        query_pattern = Pattern(
            pattern_id="query",
            structure=None,
            features={"node_count": 5, "edge_count": 4},
        )

        similar = library.find_similar(query_pattern, top_k=5)

        assert len(similar) == 0

        logger.info("✓ Empty library similarity search test passed")

    def test_similarity_caching(self, library):
        """Test similarity result caching"""
        # Add a pattern
        pattern = Pattern(
            pattern_id="pattern_1",
            structure=None,
            features={"node_count": 5, "edge_count": 4},
        )
        library._add_pattern(pattern)

        # Search twice
        query_pattern = Pattern(
            pattern_id="query",
            structure=None,
            features={"node_count": 5, "edge_count": 4},
        )

        similar1 = library.find_similar(query_pattern, top_k=3)
        cache_size_before = len(library.similarity_cache)

        similar2 = library.find_similar(query_pattern, top_k=3)
        cache_size_after = len(library.similarity_cache)

        # Results should be same
        assert similar1 == similar2

        logger.info(f"✓ Similarity caching test passed: cache size {cache_size_after}")


# ============================================================
# REINFORCEMENT TESTS
# ============================================================


class TestPatternReinforcement:
    """Test pattern reinforcement mechanisms"""

    def test_reinforce_pattern(self, library, sample_pattern):
        """Test pattern reinforcement"""
        library._add_pattern(sample_pattern)
        signature = sample_pattern.get_signature()

        # Reinforce pattern
        library.reinforce_pattern(
            signature, {"domain": "optimization"}, performance=0.8
        )

        # Check performance was updated
        perf = library.performance[signature]
        assert perf.successful_uses > 0

        logger.info("✓ Pattern reinforcement test passed")

    def test_mark_failed_pattern(self, library, sample_pattern):
        """Test marking pattern as failed"""
        library._add_pattern(sample_pattern)
        signature = sample_pattern.get_signature()

        # Mark as failed
        library.mark_failed_pattern(
            signature, {"domain": "optimization"}, reason="timeout"
        )

        # Check performance was updated
        perf = library.performance[signature]
        assert perf.failed_uses > 0
        assert "timeout" in perf.failure_reasons

        logger.info("✓ Mark failed pattern test passed")

    def test_pattern_promotion_to_proven(self, library, sample_pattern):
        """Test pattern promotion to proven status"""
        library._add_pattern(sample_pattern)
        signature = sample_pattern.get_signature()

        # Reinforce many times with success
        for _ in range(15):
            library.reinforce_pattern(signature, {}, performance=0.9)

        # Check if promoted to proven
        pattern = library.patterns[sample_pattern.pattern_id]
        if "status" in pattern.metadata:
            assert pattern.metadata["status"] == PatternStatus.PROVEN.value

        logger.info("✓ Pattern promotion test passed")


# ============================================================
# STRATIFIED LIBRARY TESTS
# ============================================================


class TestStratifiedLibrary:
    """Test StratifiedDecompositionLibrary"""

    def test_stratified_initialization(self, stratified_library):
        """Test stratified library initialization"""
        assert stratified_library is not None
        assert hasattr(stratified_library, "pattern_frequency")
        assert hasattr(stratified_library, "cross_domain_patterns")

        logger.info("✓ Stratified library initialization test passed")

    def test_update_usage_statistics(self, stratified_library, sample_pattern):
        """Test updating usage statistics"""
        stratified_library._add_pattern(sample_pattern)

        # Update usage
        stratified_library.update_usage_statistics(
            sample_pattern.pattern_id, domain="optimization"
        )

        assert stratified_library.pattern_frequency[sample_pattern.pattern_id] == 1
        assert (
            "optimization"
            in stratified_library.pattern_domains[sample_pattern.pattern_id]
        )

        logger.info("✓ Usage statistics update test passed")

    def test_get_patterns_by_frequency(self, stratified_library):
        """Test getting patterns by frequency"""
        # Add patterns with different frequencies
        for i in range(5):
            pattern = Pattern(
                pattern_id=f"pattern_{i}", structure=None, features={"node_count": i}
            )
            stratified_library._add_pattern(pattern)

            # Use pattern i times
            for _ in range(i):
                stratified_library.update_usage_statistics(
                    pattern.pattern_id, domain="test"
                )

        # Get frequent patterns (min_count=2)
        frequent = stratified_library.get_patterns_by_frequency(min_count=2)

        assert len(frequent) >= 3  # patterns 2, 3, 4

        logger.info(
            f"✓ Get patterns by frequency test passed: {len(frequent)} frequent patterns"
        )

    def test_get_patterns_by_domain(self, stratified_library, sample_principle):
        """Test getting patterns by domain"""
        stratified_library.add_principle(sample_principle)

        patterns = stratified_library.get_patterns_by_domain("optimization")

        assert len(patterns) > 0

        logger.info(f"✓ Get patterns by domain test passed: {len(patterns)} patterns")

    def test_get_cross_domain_patterns(self, stratified_library):
        """Test getting cross-domain patterns"""
        # Add pattern used in multiple domains
        pattern = Pattern(
            pattern_id="cross_domain_pattern",
            structure=None,
            features={"node_count": 5},
        )
        stratified_library._add_pattern(pattern)

        # Use in multiple domains
        domains = ["optimization", "classification", "planning", "analysis"]
        for domain in domains:
            stratified_library.update_usage_statistics(
                pattern.pattern_id, domain=domain
            )

        # Get cross-domain patterns
        cross_domain = stratified_library.get_cross_domain_patterns(min_domains=3)

        assert len(cross_domain) > 0

        logger.info(
            f"✓ Get cross-domain patterns test passed: {len(cross_domain)} patterns"
        )

    def test_get_domain_statistics(self, stratified_library, sample_principle):
        """Test domain statistics"""
        stratified_library.add_principle(sample_principle)

        # Update some usage
        if sample_principle.pattern:
            stratified_library.update_usage_statistics(
                sample_principle.pattern.pattern_id, domain="optimization"
            )

        stats = stratified_library.get_domain_statistics()

        assert isinstance(stats, dict)

        logger.info(f"✓ Domain statistics test passed: {len(stats)} domains")

    def test_domain_categorization(self, stratified_library):
        """Test domain categorization"""
        # Add patterns to different domains with different frequencies
        for i in range(3):
            pattern = Pattern(
                pattern_id=f"frequent_{i}", structure=None, features={"node_count": i}
            )
            stratified_library._add_pattern(pattern)

            # Use many times in 'frequent_domain'
            for _ in range(50):
                stratified_library.update_usage_statistics(
                    pattern.pattern_id, domain="frequent_domain"
                )

        # Get domain categories
        categories = stratified_library.domain_categories

        assert isinstance(categories, dict)

        logger.info(
            f"✓ Domain categorization test passed: {len(categories)} categories"
        )


# ============================================================
# PERSISTENCE TESTS
# ============================================================


class TestPersistence:
    """Test library persistence"""

    def test_save_and_load_patterns(self, temp_storage):
        """Test saving and loading patterns"""
        # Create library and add patterns
        lib1 = DecompositionLibrary(storage_path=temp_storage / "persist_test")

        pattern = Pattern(
            pattern_id="persist_pattern",
            structure=None,
            features={"node_count": 5, "edge_count": 4},
        )
        lib1._add_pattern(pattern)

        # Save
        lib1._save_library()

        # Create new library and load
        lib2 = DecompositionLibrary(storage_path=temp_storage / "persist_test")

        # Check pattern was loaded
        assert "persist_pattern" in lib2.patterns

        logger.info("✓ Save and load patterns test passed")

    def test_save_and_load_principles(self, temp_storage, sample_principle):
        """Test saving and loading principles"""
        # Create library and add principle
        lib1 = DecompositionLibrary(storage_path=temp_storage / "persist_test2")
        lib1.add_principle(sample_principle)

        # Save
        lib1._save_library()

        # Create new library and load
        lib2 = DecompositionLibrary(storage_path=temp_storage / "persist_test2")

        # Check principle was loaded
        assert sample_principle.principle_id in lib2.principles

        logger.info("✓ Save and load principles test passed")


# ============================================================
# STRATEGY RETRIEVAL TESTS
# ============================================================


class TestStrategyRetrieval:
    """Test strategy retrieval from library"""

    def test_get_strategy_by_type(self, stratified_library):
        """Test getting strategy by type"""
        strategy = stratified_library.get_strategy_by_type("structural")

        assert strategy is not None

        logger.info("✓ Get strategy by type test passed")

    def test_get_strategy_by_name(self, stratified_library):
        """Test getting strategy by name"""
        strategy = stratified_library.get_strategy("StructuralDecomposition")

        assert strategy is not None

        logger.info("✓ Get strategy by name test passed")

    def test_get_nonexistent_strategy(self, stratified_library):
        """Test getting nonexistent strategy"""
        strategy = stratified_library.get_strategy_by_type("nonexistent")

        assert strategy is None

        logger.info("✓ Get nonexistent strategy test passed")


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DECOMPOSITION LIBRARY TESTS")
    print("=" * 70)

    # Run with pytest
    pytest.main(
        [
            __file__,
            "-v",  # Verbose
            "-s",  # Show print statements
            "--tb=short",  # Short traceback format
            "--color=yes",  # Colored output
        ]
    )
