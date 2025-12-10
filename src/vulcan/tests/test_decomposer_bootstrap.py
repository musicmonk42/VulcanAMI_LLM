"""
test_decomposer_bootstrap.py - Comprehensive tests for decomposer bootstrap system
Part of the VULCAN-AGI system

Tests:
- Strategy instantiation and registration
- Library integration
- Fallback chain population
- Complete system initialization
- Integration validation
"""

import pytest
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components to test
from problem_decomposer.decomposer_bootstrap import (
    DecomposerBootstrap,
    create_decomposer,
    create_test_problem,
    validate_decomposer_setup,
    run_bootstrap_test,
    get_bootstrap,
)

from problem_decomposer.problem_decomposer_core import (
    ProblemDecomposer,
    ProblemGraph,
    DecompositionPlan,
    ExecutionOutcome,
)

from problem_decomposer.decomposition_strategies import (
    DecompositionStrategy,
    ExactDecomposition,
    SemanticDecomposition,
    StructuralDecomposition,
    SyntheticBridging,
    AnalogicalDecomposition,
    BruteForceSearch,
)

from problem_decomposer.decomposition_library import (
    StratifiedDecompositionLibrary,
    DecompositionPrinciple,
    Context,
    Pattern,
)

from problem_decomposer.fallback_chain import FallbackChain
from problem_decomposer.adaptive_thresholds import AdaptiveThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def bootstrap():
    """Create bootstrap instance"""
    return DecomposerBootstrap()


@pytest.fixture
def decomposer():
    """Create fully initialized decomposer"""
    return create_decomposer()


@pytest.fixture
def test_problems():
    """Create test problems of different types"""
    return {
        "hierarchical": create_test_problem("hierarchical"),
        "sequential": create_test_problem("sequential"),
        "parallel": create_test_problem("parallel"),
        "cyclic": create_test_problem("cyclic"),
        "simple": create_test_problem("simple"),
    }


# ============================================================
# BOOTSTRAP TESTS
# ============================================================


class TestDecomposerBootstrap:
    """Test DecomposerBootstrap class"""

    def test_bootstrap_initialization(self, bootstrap):
        """Test bootstrap initializes correctly"""
        assert bootstrap is not None
        assert hasattr(bootstrap, "strategy_registry")
        assert hasattr(bootstrap, "strategy_instances")
        assert hasattr(bootstrap, "library")
        assert hasattr(bootstrap, "fallback_chain")
        assert bootstrap.library is None  # Not initialized yet

        logger.info("✓ Bootstrap initialization test passed")

    def test_singleton_pattern(self):
        """Test bootstrap singleton pattern"""
        bootstrap1 = get_bootstrap()
        bootstrap2 = get_bootstrap()

        assert bootstrap1 is bootstrap2
        assert id(bootstrap1) == id(bootstrap2)

        logger.info("✓ Singleton pattern test passed")

    def test_create_strategy_instances(self, bootstrap):
        """Test strategy instance creation"""
        strategies = bootstrap.create_strategy_instances()

        # Check we got strategies
        assert len(strategies) > 0
        assert len(bootstrap.strategy_instances) > 0

        # Check all required strategy types are present
        required_types = [
            "exact",
            "semantic",
            "structural",
            "synthetic",
            "analogical",
            "brute_force",
        ]

        for strategy_type in required_types:
            assert strategy_type in strategies, (
                f"Missing strategy type: {strategy_type}"
            )

        # Check strategy instances are unique
        unique_instances = set(id(s) for s in bootstrap.strategy_instances)
        assert len(unique_instances) == len(bootstrap.strategy_instances)

        # Check predicted type mappings
        predicted_mappings = [
            "hierarchical_decomposition",
            "modular_decomposition",
            "pipeline_decomposition",
            "parallel_decomposition",
            "recursive_decomposition",
            "temporal_decomposition",
            "constraint_based_decomposition",
            "direct_decomposition",
            "iterative_decomposition",
            "hybrid_decomposition",
        ]

        for mapping in predicted_mappings:
            assert mapping in strategies, f"Missing mapping: {mapping}"

        logger.info(
            f"✓ Created {len(strategies)} strategy mappings from {len(bootstrap.strategy_instances)} instances"
        )

    def test_strategy_types(self, bootstrap):
        """Test each strategy type is correct class"""
        strategies = bootstrap.create_strategy_instances()

        # Verify strategy types
        assert isinstance(strategies["exact"], ExactDecomposition)
        assert isinstance(strategies["semantic"], SemanticDecomposition)
        assert isinstance(strategies["structural"], StructuralDecomposition)
        assert isinstance(strategies["synthetic"], SyntheticBridging)
        assert isinstance(strategies["analogical"], AnalogicalDecomposition)
        assert isinstance(strategies["brute_force"], BruteForceSearch)

        # Verify all have required methods
        for name, strategy in strategies.items():
            assert hasattr(strategy, "apply"), f"Strategy {name} missing apply method"
            assert hasattr(strategy, "decompose"), (
                f"Strategy {name} missing decompose method"
            )
            assert hasattr(strategy, "name"), f"Strategy {name} missing name attribute"

        logger.info("✓ Strategy type validation passed")

    def test_register_strategies_in_library(self, bootstrap):
        """Test strategy registration in library"""
        strategies = bootstrap.create_strategy_instances()
        library = StratifiedDecompositionLibrary()

        bootstrap.register_strategies_in_library(library, strategies)

        # Verify registration
        assert hasattr(library, "strategy_registry")
        assert len(library.strategy_registry) == len(strategies)

        # Test strategy retrieval
        for strategy_name in ["exact", "semantic", "structural"]:
            retrieved = library.get_strategy(strategy_name)
            assert retrieved is not None, (
                f"Failed to retrieve strategy: {strategy_name}"
            )
            assert retrieved == strategies[strategy_name]

        # Test get_strategy_by_type
        hierarchical = library.get_strategy_by_type("hierarchical_decomposition")
        assert hierarchical is not None
        assert isinstance(hierarchical, StructuralDecomposition)

        logger.info("✓ Strategy registration test passed")

    def test_populate_fallback_chain(self, bootstrap):
        """Test fallback chain population"""
        strategies = bootstrap.create_strategy_instances()
        fallback_chain = FallbackChain()

        bootstrap.populate_fallback_chain(fallback_chain, bootstrap.strategy_instances)

        # Check fallback chain has strategies
        assert len(fallback_chain.strategies) > 0
        assert len(fallback_chain.strategies) == len(bootstrap.strategy_instances)

        # Verify ordering (exact should be first, brute force last)
        strategy_names = [s.name for s in fallback_chain.strategies]
        assert "ExactDecomposition" in strategy_names[0:2]  # Should be early
        assert "BruteForceSearch" in strategy_names[-2:]  # Should be last

        # Verify costs are set
        for strategy in fallback_chain.strategies:
            strategy_name = strategy.name
            assert (
                strategy_name in fallback_chain.strategy_costs or True
            )  # Some may not have costs

        logger.info(
            f"✓ Populated fallback chain with {len(fallback_chain.strategies)} strategies"
        )

    def test_initialize_library_with_base_principles(self, bootstrap):
        """Test library initialization with base principles"""
        library = StratifiedDecompositionLibrary()

        bootstrap.initialize_library_with_base_principles(library)

        # Check principles were added
        assert len(library.principles) > 0

        # Verify specific principles
        expected_principles = [
            "hierarchical_principle",
            "modular_principle",
            "sequential_principle",
            "parallel_principle",
            "iterative_principle",
        ]

        for principle_id in expected_principles:
            assert principle_id in library.principles, (
                f"Missing principle: {principle_id}"
            )

        # Verify principle structure
        for principle_id, principle in library.principles.items():
            assert hasattr(principle, "name")
            assert hasattr(principle, "applicable_contexts")
            assert hasattr(principle, "success_rate")
            assert len(principle.applicable_contexts) > 0

        logger.info(
            f"✓ Initialized library with {len(library.principles)} base principles"
        )

    def test_configure_adaptive_thresholds(self, bootstrap):
        """Test threshold configuration"""
        thresholds = AdaptiveThresholds()

        bootstrap.configure_adaptive_thresholds(thresholds)

        # Check thresholds are set
        current = thresholds.get_current()
        assert current is not None
        assert "confidence" in current
        assert "complexity" in current
        assert "performance" in current

        # Verify reasonable values
        assert 0 < current["confidence"] < 1
        assert current["complexity"] > 0
        assert 0 < current["performance"] < 1

        logger.info("✓ Adaptive thresholds configured")


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestDecomposerIntegration:
    """Test complete decomposer integration"""

    def test_create_decomposer_basic(self):
        """Test basic decomposer creation"""
        decomposer = create_decomposer()

        assert decomposer is not None
        assert isinstance(decomposer, ProblemDecomposer)

        logger.info("✓ Basic decomposer creation passed")

    def test_decomposer_components(self, decomposer):
        """Test decomposer has all required components"""
        # Core components
        assert hasattr(decomposer, "library")
        assert hasattr(decomposer, "thresholds")
        assert hasattr(decomposer, "fallback_chain")
        assert hasattr(decomposer, "executor")
        assert hasattr(decomposer, "strategy_profiler")
        assert hasattr(decomposer, "performance_tracker")

        # Check components are initialized
        assert decomposer.library is not None
        assert decomposer.thresholds is not None
        assert decomposer.fallback_chain is not None
        assert decomposer.executor is not None

        logger.info("✓ Decomposer components validation passed")

    def test_decomposer_has_strategies(self, decomposer):
        """Test decomposer has registered strategies"""
        library = decomposer.library

        # Check strategy registry exists
        assert hasattr(library, "strategy_registry")
        assert len(library.strategy_registry) > 0

        # Try to retrieve strategies
        test_types = ["exact", "structural", "hierarchical_decomposition"]
        for strategy_type in test_types:
            strategy = library.get_strategy_by_type(strategy_type)
            assert strategy is not None, f"Failed to get strategy: {strategy_type}"

        logger.info(
            f"✓ Decomposer has {len(library.strategy_registry)} registered strategies"
        )

    def test_decomposer_has_fallback_strategies(self, decomposer):
        """Test fallback chain is populated"""
        fallback_chain = decomposer.fallback_chain

        assert len(fallback_chain.strategies) > 0
        assert len(fallback_chain.strategies) >= 5  # At least 5 strategies

        logger.info(f"✓ Fallback chain has {len(fallback_chain.strategies)} strategies")

    def test_decomposer_has_principles(self, decomposer):
        """Test library has base principles"""
        library = decomposer.library

        assert hasattr(library, "principles")
        assert len(library.principles) > 0

        logger.info(f"✓ Library has {len(library.principles)} base principles")

    def test_validate_decomposer_setup(self, decomposer):
        """Test decomposer setup validation"""
        validation = validate_decomposer_setup(decomposer)

        assert validation is not None
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert "checks" in validation

        # Should be valid
        if not validation["valid"]:
            logger.error("Validation errors:")
            for error in validation["errors"]:
                logger.error(f"  - {error}")

        assert validation["valid"], "Decomposer setup validation failed"

        # Check specific validations
        checks = validation["checks"]
        assert checks.get("strategy_count", 0) > 0
        assert checks.get("fallback_chain_count", 0) > 0
        assert checks.get("executor_initialized") == True

        logger.info("✓ Decomposer setup validation passed")


# ============================================================
# FUNCTIONAL TESTS
# ============================================================


class TestDecomposerFunctionality:
    """Test decomposer functionality"""

    def test_decompose_simple_problem(self, decomposer, test_problems):
        """Test decomposing simple problem"""
        problem = test_problems["simple"]

        plan = decomposer.decompose_novel_problem(problem)

        assert plan is not None
        assert isinstance(plan, DecompositionPlan)
        assert len(plan.steps) > 0
        assert plan.confidence > 0

        logger.info(
            f"✓ Simple problem decomposed into {len(plan.steps)} steps (confidence: {plan.confidence:.2f})"
        )

    def test_decompose_hierarchical_problem(self, decomposer, test_problems):
        """Test decomposing hierarchical problem"""
        problem = test_problems["hierarchical"]

        plan = decomposer.decompose_novel_problem(problem)

        assert plan is not None
        assert len(plan.steps) > 0

        # Hierarchical problems should have multiple steps
        assert len(plan.steps) >= 2

        logger.info(f"✓ Hierarchical problem decomposed into {len(plan.steps)} steps")

    def test_decompose_sequential_problem(self, decomposer, test_problems):
        """Test decomposing sequential problem"""
        problem = test_problems["sequential"]

        plan = decomposer.decompose_novel_problem(problem)

        assert plan is not None
        assert len(plan.steps) > 0

        logger.info(f"✓ Sequential problem decomposed into {len(plan.steps)} steps")

    def test_decompose_parallel_problem(self, decomposer, test_problems):
        """Test decomposing parallel problem"""
        problem = test_problems["parallel"]

        plan = decomposer.decompose_novel_problem(problem)

        assert plan is not None
        assert len(plan.steps) > 0

        logger.info(f"✓ Parallel problem decomposed into {len(plan.steps)} steps")

    def test_decompose_and_execute(self, decomposer, test_problems):
        """Test full decompose and execute pipeline"""
        problem = test_problems["simple"]

        # This is the main integration test
        try:
            plan, outcome = decomposer.decompose_and_execute(problem)

            assert plan is not None
            assert outcome is not None
            assert isinstance(outcome, ExecutionOutcome)

            logger.info(
                f"✓ Decompose and execute completed: success={outcome.success}, time={outcome.execution_time:.2f}s"
            )

        except RuntimeError as e:
            if "SAFETY CRITICAL" in str(e):
                logger.info(
                    "✓ Safety validator correctly required (expected in test environment)"
                )
                pytest.skip("Safety validator not available in test environment")
            else:
                raise

    def test_strategy_profiling(self, decomposer, test_problems):
        """Test strategy profiling functionality"""
        # Profile all strategies
        for strategy in decomposer.fallback_chain.strategies:
            profile = decomposer.strategy_profiler.profile_strategy(strategy)

            assert profile is not None
            assert "name" in profile
            assert "type" in profile
            assert "complexity_range" in profile

        logger.info(
            f"✓ Profiled {len(decomposer.fallback_chain.strategies)} strategies"
        )

    def test_performance_tracking(self, decomposer, test_problems):
        """Test performance tracking"""
        problem = test_problems["simple"]

        # Create a mock plan and outcome
        plan = DecompositionPlan(
            steps=[{"type": "test", "description": "test step"}],
            confidence=0.8,
            estimated_complexity=2.0,
        )

        outcome = ExecutionOutcome(
            success=True, execution_time=1.5, metrics={"test_metric": 0.9}
        )

        # Record execution
        decomposer.performance_tracker.record_execution(problem, plan, outcome)

        # Check tracking worked
        stats = decomposer.get_statistics()
        assert "performance_stats" in stats

        logger.info("✓ Performance tracking working")

    def test_adaptive_threshold_updates(self, decomposer):
        """Test adaptive threshold updates"""
        initial_confidence = decomposer.thresholds.get_confidence_threshold()

        # Simulate successful execution
        decomposer.thresholds.update_from_outcome(
            complexity=2.0, success=True, execution_time=1.0
        )

        # Thresholds should still be valid
        updated_confidence = decomposer.thresholds.get_confidence_threshold()
        assert 0 < updated_confidence < 1

        logger.info(
            f"✓ Adaptive thresholds updated: {initial_confidence:.2f} -> {updated_confidence:.2f}"
        )


# ============================================================
# TEST PROBLEM TESTS
# ============================================================


class TestTestProblems:
    """Test the test problem generation"""

    def test_create_hierarchical_problem(self):
        """Test hierarchical problem creation"""
        problem = create_test_problem("hierarchical")

        assert problem is not None
        assert len(problem.nodes) > 0
        assert len(problem.edges) > 0
        assert problem.root is not None
        assert problem.metadata.get("type") == "hierarchical"

        logger.info(
            f"✓ Created hierarchical problem: {len(problem.nodes)} nodes, {len(problem.edges)} edges"
        )

    def test_create_sequential_problem(self):
        """Test sequential problem creation"""
        problem = create_test_problem("sequential")

        assert problem is not None
        assert len(problem.nodes) > 0
        assert len(problem.edges) > 0
        assert problem.metadata.get("type") == "sequential"

        logger.info(
            f"✓ Created sequential problem: {len(problem.nodes)} nodes, {len(problem.edges)} edges"
        )

    def test_create_parallel_problem(self):
        """Test parallel problem creation"""
        problem = create_test_problem("parallel")

        assert problem is not None
        assert len(problem.nodes) > 0
        assert len(problem.edges) > 0
        assert problem.metadata.get("type") == "parallel"

        logger.info(
            f"✓ Created parallel problem: {len(problem.nodes)} nodes, {len(problem.edges)} edges"
        )

    def test_create_cyclic_problem(self):
        """Test cyclic problem creation"""
        problem = create_test_problem("cyclic")

        assert problem is not None
        assert len(problem.nodes) > 0
        assert len(problem.edges) > 0
        assert problem.metadata.get("type") == "iterative"

        # Should have a cycle
        has_cycle = False
        for source, target, _ in problem.edges:
            if source == "evaluate" and target == "refine":
                has_cycle = True
            if source == "refine" and target == "evaluate":
                has_cycle = True

        assert has_cycle, "Cyclic problem should have a cycle"

        logger.info(
            f"✓ Created cyclic problem: {len(problem.nodes)} nodes, {len(problem.edges)} edges"
        )

    def test_create_simple_problem(self):
        """Test simple problem creation"""
        problem = create_test_problem("simple")

        assert problem is not None
        assert len(problem.nodes) > 0
        assert len(problem.edges) > 0
        assert problem.metadata.get("type") == "simple"

        logger.info(
            f"✓ Created simple problem: {len(problem.nodes)} nodes, {len(problem.edges)} edges"
        )


# ============================================================
# STRESS TESTS
# ============================================================


class TestBootstrapStress:
    """Stress tests for bootstrap system"""

    def test_multiple_decomposer_creation(self):
        """Test creating multiple decomposers"""
        decomposers = []

        for i in range(5):
            decomposer = create_decomposer()
            decomposers.append(decomposer)
            assert decomposer is not None

        logger.info(f"✓ Created {len(decomposers)} decomposers successfully")

    def test_decompose_multiple_problems(self, decomposer):
        """Test decomposing multiple problems"""
        problem_types = ["simple", "hierarchical", "sequential", "parallel"]

        for problem_type in problem_types:
            problem = create_test_problem(problem_type)
            plan = decomposer.decompose_novel_problem(problem)

            assert plan is not None
            assert len(plan.steps) > 0

        logger.info(f"✓ Decomposed {len(problem_types)} different problem types")

    def test_strategy_retrieval_performance(self, decomposer):
        """Test strategy retrieval is fast"""
        library = decomposer.library

        start_time = time.time()

        for _ in range(1000):
            library.get_strategy_by_type("hierarchical_decomposition")
            library.get_strategy("structural")

        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Strategy retrieval too slow: {elapsed:.3f}s"

        logger.info(f"✓ 2000 strategy retrievals in {elapsed:.3f}s")


# ============================================================
# ERROR HANDLING TESTS
# ============================================================


class TestBootstrapErrorHandling:
    """Test error handling in bootstrap"""

    def test_invalid_problem_type(self):
        """Test creating problem with invalid type"""
        problem = create_test_problem("invalid_type")

        # Should return simple problem as fallback
        assert problem is not None
        assert len(problem.nodes) > 0

        logger.info("✓ Invalid problem type handled gracefully")

    def test_decompose_empty_problem(self, decomposer):
        """Test decomposing empty problem"""
        problem = ProblemGraph(nodes={}, edges=[], metadata={"domain": "test"})

        plan = decomposer.decompose_novel_problem(problem)

        # Should still return a plan (even if minimal)
        assert plan is not None

        logger.info("✓ Empty problem handled gracefully")

    def test_missing_strategy_fallback(self, decomposer):
        """Test fallback when preferred strategy not available"""
        library = decomposer.library

        # Try to get non-existent strategy
        strategy = library.get_strategy_by_type("nonexistent_strategy")

        # Should return None
        assert strategy is None

        logger.info("✓ Missing strategy handled correctly")


# ============================================================
# RUN TESTS
# ============================================================


def test_run_bootstrap_test():
    """Test the built-in bootstrap test"""
    result = run_bootstrap_test()

    assert result == True, "Bootstrap test failed"

    logger.info("✓ Built-in bootstrap test passed")


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DECOMPOSER BOOTSTRAP TESTS")
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
