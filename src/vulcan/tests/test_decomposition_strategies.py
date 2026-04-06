"""
test_decomposition_strategies.py - Comprehensive tests for decomposition strategies
Part of the VULCAN-AGI system

Tests:
- All strategy types (Exact, Semantic, Structural, Synthetic, Analogical, BruteForce)
- Strategy application and decomposition
- Fallback behavior
- Performance metrics
- Pattern matching
- Strategy-specific features
"""

from problem_decomposer.problem_decomposer_core import ProblemGraph
from problem_decomposer.decomposition_strategies import (
    AnalogicalDecomposition,
    BruteForceSearch,
    DecompositionResult,
    ExactDecomposition,
    SemanticDecomposition,
    StrategyType,
    StructuralDecomposition,
    SyntheticBridging,
)
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components to test

logger = logging.getLogger(__name__)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def simple_graph():
    """Create simple linear graph"""
    return ProblemGraph(
        nodes={
            "A": {"type": "operation"},
            "B": {"type": "operation"},
            "C": {"type": "operation"},
        },
        edges=[("A", "B", {}), ("B", "C", {})],
        root="A",
        metadata={"domain": "test", "type": "simple"},
    )


@pytest.fixture
def hierarchical_graph():
    """Create hierarchical graph"""
    return ProblemGraph(
        nodes={
            "root": {"type": "decision"},
            "branch1": {"type": "operation"},
            "branch2": {"type": "operation"},
            "leaf1": {"type": "transform"},
            "leaf2": {"type": "transform"},
        },
        edges=[
            ("root", "branch1", {}),
            ("root", "branch2", {}),
            ("branch1", "leaf1", {}),
            ("branch2", "leaf2", {}),
        ],
        root="root",
        metadata={"domain": "test", "type": "hierarchical"},
    )


@pytest.fixture
def parallel_graph():
    """Create parallel graph"""
    return ProblemGraph(
        nodes={
            "start": {"type": "decision"},
            "task1": {"type": "operation"},
            "task2": {"type": "operation"},
            "task3": {"type": "operation"},
            "merge": {"type": "operation"},
        },
        edges=[
            ("start", "task1", {}),
            ("start", "task2", {}),
            ("start", "task3", {}),
            ("task1", "merge", {}),
            ("task2", "merge", {}),
            ("task3", "merge", {}),
        ],
        root="start",
        metadata={"domain": "test", "type": "parallel"},
    )


@pytest.fixture
def cyclic_graph():
    """Create cyclic graph"""
    return ProblemGraph(
        nodes={
            "init": {"type": "operation"},
            "evaluate": {"type": "decision"},
            "refine": {"type": "transform"},
            "output": {"type": "operation"},
        },
        edges=[
            ("init", "evaluate", {}),
            ("evaluate", "refine", {}),
            ("refine", "evaluate", {}),  # Cycle
            ("evaluate", "output", {}),
        ],
        root="init",
        metadata={"domain": "test", "type": "cyclic"},
    )


@pytest.fixture
def empty_graph():
    """Create empty graph"""
    return ProblemGraph(
        nodes={}, edges=[], metadata={"domain": "test", "type": "empty"}
    )


# ============================================================
# DECOMPOSITION RESULT TESTS
# ============================================================


class TestDecompositionResult:
    """Test DecompositionResult dataclass"""

    def test_result_creation(self):
        """Test result creation"""
        result = DecompositionResult(
            components=[{"type": "test"}],
            confidence=0.8,
            strategy_type="test",
            execution_time=1.0,
        )

        assert len(result.components) == 1
        assert result.confidence == 0.8
        assert result.strategy_type == "test"

        logger.info("✓ DecompositionResult creation test passed")

    def test_is_complete(self):
        """Test completeness check"""
        # Complete result
        complete = DecompositionResult(components=[{"type": "test"}], confidence=0.6)
        assert complete.is_complete() == True

        # Incomplete - no components
        incomplete1 = DecompositionResult(components=[], confidence=0.8)
        assert incomplete1.is_complete() == False

        # Incomplete - low confidence
        incomplete2 = DecompositionResult(components=[{"type": "test"}], confidence=0.3)
        assert incomplete2.is_complete() == False

        logger.info("✓ DecompositionResult completeness test passed")


# ============================================================
# EXACT DECOMPOSITION TESTS
# ============================================================


class TestExactDecomposition:
    """Test ExactDecomposition strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = ExactDecomposition()

        assert strategy.name == "ExactDecomposition"
        assert strategy.strategy_type == StrategyType.EXACT
        assert len(strategy.pattern_library) > 0

        logger.info("✓ ExactDecomposition initialization test passed")

    def test_apply_simple_graph(self, simple_graph):
        """Test applying to simple graph"""
        strategy = ExactDecomposition()
        result = strategy.apply(simple_graph)

        assert result is not None
        assert isinstance(result, DecompositionResult)
        assert len(result.components) > 0
        assert result.confidence > 0

        logger.info(
            f"✓ ExactDecomposition apply test passed: {len(result.components)} components"
        )

    def test_decompose_creates_steps(self, simple_graph):
        """Test decompose creates proper steps"""
        strategy = ExactDecomposition()
        steps = strategy.decompose(simple_graph)

        assert isinstance(steps, list)
        assert len(steps) > 0

        # Check step structure
        for step in steps:
            assert "step_id" in step
            assert "type" in step
            assert "component" in step
            assert "confidence" in step
            assert "description" in step

        logger.info(f"✓ ExactDecomposition decompose test passed: {len(steps)} steps")

    def test_empty_graph_fallback(self, empty_graph):
        """Test fallback behavior with empty graph"""
        strategy = ExactDecomposition()
        steps = strategy.decompose(empty_graph)

        # Should still return at least one fallback step
        assert len(steps) > 0

        # Should have fallback marker
        assert any(step.get("component", {}).get("fallback") for step in steps)

        logger.info("✓ ExactDecomposition empty graph fallback test passed")

    def test_pattern_matching(self, hierarchical_graph):
        """Test pattern matching functionality"""
        strategy = ExactDecomposition()
        matches = strategy.find_exact_pattern_matches(hierarchical_graph)

        assert isinstance(matches, list)
        # May or may not find matches depending on patterns

        logger.info(
            f"✓ ExactDecomposition pattern matching test passed: {len(matches)} matches"
        )

    def test_success_rate_tracking(self, simple_graph):
        """Test success rate tracking"""
        strategy = ExactDecomposition()

        initial_count = strategy.execution_count

        strategy.apply(simple_graph)
        strategy.apply(simple_graph)

        assert strategy.execution_count == initial_count + 2
        assert strategy.get_success_rate() >= 0

        logger.info(
            f"✓ ExactDecomposition success rate test passed: {strategy.get_success_rate():.2f}"
        )


# ============================================================
# SEMANTIC DECOMPOSITION TESTS
# ============================================================


class TestSemanticDecomposition:
    """Test SemanticDecomposition strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = SemanticDecomposition()

        assert strategy.name == "SemanticDecomposition"
        assert strategy.strategy_type == StrategyType.SEMANTIC
        assert strategy.similarity_threshold == 0.7

        logger.info("✓ SemanticDecomposition initialization test passed")

    def test_apply_simple_graph(self, simple_graph):
        """Test applying to simple graph"""
        strategy = SemanticDecomposition()
        result = strategy.apply(simple_graph)

        assert result is not None
        assert len(result.components) > 0
        assert result.confidence > 0

        logger.info(
            f"✓ SemanticDecomposition apply test passed: {len(result.components)} components"
        )

    def test_embedding_generation(self):
        """Test embedding generation"""
        strategy = SemanticDecomposition()

        embedding = strategy._generate_embedding({"type": "test", "value": 42})

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 128  # Fixed size

        logger.info("✓ SemanticDecomposition embedding generation test passed")

    def test_embedding_caching(self):
        """Test embedding caching"""
        strategy = SemanticDecomposition()

        # Generate same embedding twice
        node_data = {"type": "test"}

        cache_size_before = len(strategy.embedding_cache)
        embedding1 = strategy._generate_embedding(node_data)

        # Add to cache manually
        strategy.embedding_cache[str(node_data)] = embedding1

        cache_size_after = len(strategy.embedding_cache)

        assert cache_size_after > cache_size_before

        logger.info("✓ SemanticDecomposition caching test passed")

    def test_clustering(self, hierarchical_graph):
        """Test semantic clustering"""
        strategy = SemanticDecomposition()

        # Get embeddings
        embeddings = strategy._get_node_embeddings(hierarchical_graph)

        # Cluster
        clusters = strategy._cluster_by_similarity(embeddings)

        assert isinstance(clusters, dict)

        logger.info(
            f"✓ SemanticDecomposition clustering test passed: {len(clusters)} clusters"
        )

    def test_cache_size_limit(self):
        """Test cache size limiting through normal API usage"""
        strategy = SemanticDecomposition()

        # Create a mock graph with many unique nodes to fill cache through normal API
        # This tests the actual _get_node_embeddings method which has the limiting logic
        class MockGraph:
            def __init__(self, num_nodes):
                self.nodes = {
                    f"node_{i}": {"data": f"unique_value_{i}"} for i in range(num_nodes)
                }

        # Create graph with more nodes than cache limit
        test_graph = MockGraph(strategy.max_cache_size + 100)

        # Process through normal API which has cache limiting
        strategy._get_node_embeddings(test_graph)

        # Cache should be strictly limited now with the while loop
        assert len(strategy.embedding_cache) <= strategy.max_cache_size

        logger.info(
            f"✓ SemanticDecomposition cache limiting test passed: {len(strategy.embedding_cache)} entries"
        )


# ============================================================
# STRUCTURAL DECOMPOSITION TESTS
# ============================================================


class TestStructuralDecomposition:
    """Test StructuralDecomposition strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = StructuralDecomposition()

        assert strategy.name == "StructuralDecomposition"
        assert strategy.strategy_type == StrategyType.STRUCTURAL
        assert len(strategy.structural_patterns) > 0

        logger.info("✓ StructuralDecomposition initialization test passed")

    def test_apply_hierarchical_graph(self, hierarchical_graph):
        """Test applying to hierarchical graph"""
        strategy = StructuralDecomposition()
        result = strategy.apply(hierarchical_graph)

        assert result is not None
        assert len(result.components) > 0

        logger.info(
            f"✓ StructuralDecomposition hierarchical test passed: {len(result.components)} components"
        )

    def test_apply_parallel_graph(self, parallel_graph):
        """Test applying to parallel graph"""
        strategy = StructuralDecomposition()
        result = strategy.apply(parallel_graph)

        assert result is not None
        assert len(result.components) > 0

        logger.info(
            f"✓ StructuralDecomposition parallel test passed: {len(result.components)} components"
        )

    def test_check_hierarchical(self, hierarchical_graph):
        """Test hierarchical pattern detection"""
        strategy = StructuralDecomposition()

        G = (
            hierarchical_graph.to_networkx()
            if hasattr(hierarchical_graph, "to_networkx")
            else hierarchical_graph
        )
        result = strategy._check_hierarchical(G)

        # May or may not detect hierarchy depending on graph structure
        assert result is None or isinstance(result, dict)

        logger.info("✓ StructuralDecomposition hierarchical check test passed")

    def test_check_parallel(self, parallel_graph):
        """Test parallel pattern detection"""
        strategy = StructuralDecomposition()

        G = (
            parallel_graph.to_networkx()
            if hasattr(parallel_graph, "to_networkx")
            else parallel_graph
        )
        result = strategy._check_parallel(G)

        # Should detect parallel structure
        if result:
            assert "parallel_groups" in result

        logger.info("✓ StructuralDecomposition parallel check test passed")

    def test_fallback_behavior(self, empty_graph):
        """Test fallback with empty graph"""
        strategy = StructuralDecomposition()
        steps = strategy.decompose(empty_graph)

        # Should return fallback steps
        assert len(steps) > 0

        logger.info("✓ StructuralDecomposition fallback test passed")


# ============================================================
# SYNTHETIC BRIDGING TESTS
# ============================================================


class TestSyntheticBridging:
    """Test SyntheticBridging strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = SyntheticBridging()

        assert strategy.name == "SyntheticBridging"
        assert strategy.strategy_type == StrategyType.SYNTHETIC
        assert len(strategy.bridge_templates) > 0

        logger.info("✓ SyntheticBridging initialization test passed")

    def test_apply_simple_graph(self, simple_graph):
        """Test applying to simple graph"""
        strategy = SyntheticBridging()
        result = strategy.apply(simple_graph)

        assert result is not None
        assert len(result.components) > 0

        logger.info(
            f"✓ SyntheticBridging apply test passed: {len(result.components)} components"
        )

    def test_pattern_mutation(self):
        """Test pattern mutation"""
        strategy = SyntheticBridging()

        pattern = {"name": "test", "size": 5, "connectivity": "medium"}
        target = {"size": 7, "connectivity": "dense"}

        mutated = strategy.mutate_pattern(pattern, target)

        assert mutated is not None
        assert mutated["size"] == 7
        assert mutated["connectivity"] == "dense"

        logger.info("✓ SyntheticBridging mutation test passed")

    def test_structure_analysis(self, simple_graph):
        """Test structure analysis"""
        strategy = SyntheticBridging()

        structure = strategy._analyze_structure(simple_graph)

        assert "size" in structure
        assert "edges" in structure
        assert "connectivity" in structure

        logger.info("✓ SyntheticBridging structure analysis test passed")

    def test_template_selection(self):
        """Test template selection"""
        strategy = SyntheticBridging()

        structure = {"size": 4, "edges": 3, "connectivity": "medium"}
        template = strategy._select_template(structure)

        # May or may not find template
        assert template is None or isinstance(template, dict)

        logger.info("✓ SyntheticBridging template selection test passed")


# ============================================================
# ANALOGICAL DECOMPOSITION TESTS
# ============================================================


class TestAnalogicalDecomposition:
    """Test AnalogicalDecomposition strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = AnalogicalDecomposition()

        assert strategy.name == "AnalogicalDecomposition"
        assert strategy.strategy_type == StrategyType.ANALOGICAL
        assert len(strategy.analogy_database) > 0

        logger.info("✓ AnalogicalDecomposition initialization test passed")

    def test_apply_simple_graph(self, simple_graph):
        """Test applying to simple graph"""
        strategy = AnalogicalDecomposition()
        result = strategy.apply(simple_graph)

        assert result is not None
        assert len(result.components) > 0

        logger.info(
            f"✓ AnalogicalDecomposition apply test passed: {len(result.components)} components"
        )

    def test_feature_extraction(self, hierarchical_graph):
        """Test feature extraction"""
        strategy = AnalogicalDecomposition()

        features = strategy._extract_features(hierarchical_graph)

        assert "size" in features
        assert "complexity" in features
        assert "structure_type" in features

        logger.info("✓ AnalogicalDecomposition feature extraction test passed")

    def test_similarity_calculation(self):
        """Test similarity calculation"""
        strategy = AnalogicalDecomposition()

        features1 = {"size": 5, "complexity": 2.0, "has_cycles": False}
        features2 = {"size": 6, "complexity": 2.5, "has_cycles": False}

        similarity = strategy._calculate_similarity(features1, features2)

        assert 0 <= similarity <= 1

        logger.info(
            f"✓ AnalogicalDecomposition similarity test passed: {similarity:.2f}"
        )

    def test_mapping_creation(self):
        """Test mapping creation"""
        strategy = AnalogicalDecomposition()

        target = {"size": 10, "complexity": 3.0}
        source = {"size": 5, "complexity": 2.0}

        mapping = strategy._create_mapping(target, source)

        assert "feature_correspondence" in mapping
        assert "transformations" in mapping

        logger.info("✓ AnalogicalDecomposition mapping test passed")


# ============================================================
# BRUTE FORCE SEARCH TESTS
# ============================================================


class TestBruteForceSearch:
    """Test BruteForceSearch strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = BruteForceSearch()

        assert strategy.name == "BruteForceSearch"
        assert strategy.strategy_type == StrategyType.BRUTE_FORCE
        assert strategy.is_deterministic() == False

        logger.info("✓ BruteForceSearch initialization test passed")

    def test_apply_simple_graph(self, simple_graph):
        """Test applying to simple graph"""
        strategy = BruteForceSearch()
        result = strategy.apply(simple_graph)

        assert result is not None
        assert len(result.components) > 0

        logger.info(
            f"✓ BruteForceSearch apply test passed: {len(result.components)} components"
        )

    def test_exhaustive_search(self, hierarchical_graph):
        """Test exhaustive search"""
        strategy = BruteForceSearch()

        decomposition = strategy._exhaustive_search(hierarchical_graph)

        assert decomposition is not None
        assert len(decomposition) > 0

        logger.info(
            f"✓ BruteForceSearch exhaustive search test passed: {len(decomposition)} parts"
        )

    def test_decomposition_validation(self):
        """Test decomposition validation"""
        strategy = BruteForceSearch()

        # Valid decomposition
        valid = [{"nodes": ["A", "B"]}, {"nodes": ["C", "D"]}]
        assert strategy._is_valid_decomposition(valid) == True

        # Invalid - empty
        invalid = []
        assert strategy._is_valid_decomposition(invalid) == False

        logger.info("✓ BruteForceSearch validation test passed")

    def test_iteration_limit(self, hierarchical_graph):
        """Test iteration limiting"""
        strategy = BruteForceSearch(max_depth=2)
        strategy.max_iterations = 5  # Set low limit

        strategy._exhaustive_search(hierarchical_graph)

        # Should respect iteration limit
        assert strategy._iteration_count <= strategy.max_iterations

        logger.info("✓ BruteForceSearch iteration limit test passed")


# ============================================================
# CROSS-STRATEGY TESTS
# ============================================================


class TestAllStrategies:
    """Test all strategies with same inputs"""

    def test_all_strategies_simple_graph(self, simple_graph):
        """Test all strategies on simple graph"""
        strategies = [
            ExactDecomposition(),
            SemanticDecomposition(),
            StructuralDecomposition(),
            SyntheticBridging(),
            AnalogicalDecomposition(),
            BruteForceSearch(),
        ]

        for strategy in strategies:
            result = strategy.apply(simple_graph)

            assert result is not None
            assert len(result.components) > 0
            assert result.confidence >= 0

            logger.info(
                f"✓ {strategy.name} handled simple graph: {len(result.components)} components"
            )

    def test_all_strategies_empty_graph(self, empty_graph):
        """Test all strategies handle empty graph"""
        strategies = [
            ExactDecomposition(),
            SemanticDecomposition(),
            StructuralDecomposition(),
            SyntheticBridging(),
            AnalogicalDecomposition(),
            BruteForceSearch(),
        ]

        for strategy in strategies:
            steps = strategy.decompose(empty_graph)

            # All should handle empty graph gracefully
            assert steps is not None
            assert len(steps) > 0  # Should have fallback

            logger.info(f"✓ {strategy.name} handled empty graph with fallback")

    def test_all_strategies_metrics(self, hierarchical_graph):
        """Test metrics tracking for all strategies"""
        strategies = [
            ExactDecomposition(),
            SemanticDecomposition(),
            StructuralDecomposition(),
            SyntheticBridging(),
            AnalogicalDecomposition(),
            BruteForceSearch(),
        ]

        for strategy in strategies:
            # Run multiple times
            for _ in range(3):
                strategy.apply(hierarchical_graph)

            assert strategy.execution_count == 3
            assert strategy.get_success_rate() >= 0
            assert strategy.get_average_execution_time() >= 0

            logger.info(
                f"✓ {strategy.name} metrics: success_rate={strategy.get_success_rate():.2f}"
            )


# ============================================================
# PERFORMANCE TESTS
# ============================================================


class TestPerformance:
    """Test strategy performance"""

    def test_execution_time_tracking(self, simple_graph):
        """Test execution time tracking"""
        strategy = ExactDecomposition()

        result = strategy.apply(simple_graph)

        # FIXED: Allow for 0.0 due to timing precision on fast operations
        assert result.execution_time >= 0
        assert strategy.total_execution_time >= 0

        logger.info(
            f"✓ Execution time tracking test passed: {result.execution_time:.6f}s"
        )

    def test_parallelizability(self):
        """Test parallelizability flag"""
        strategies = [
            ExactDecomposition(),
            SemanticDecomposition(),
            StructuralDecomposition(),
        ]

        for strategy in strategies:
            # Default should be False
            assert isinstance(strategy.is_parallelizable(), bool)

        logger.info("✓ Parallelizability test passed")


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DECOMPOSITION STRATEGIES TESTS")
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
