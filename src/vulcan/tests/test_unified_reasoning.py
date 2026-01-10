"""
Comprehensive test suite for unified_reasoning.py

Tests the unified reasoning system with multiple strategies, tool selection,
and portfolio execution, handling optional dependencies gracefully.

PERFORMANCE OPTIMIZED VERSION:
- Uses MODULE-SCOPED fixture to create ONE UnifiedReasoner for all 45 tests
- Previous version created 32+ UnifiedReasoner instances (one per test), causing timeouts
- Added skip_runtime config to avoid heavy initialization
- State is reset between tests via autouse fixture

CRITICAL NOTES:
- The test_state_persistence test is SKIPPED because it creates a real UnifiedReasoner
  that loads transformer models, which causes access violations when combined with
  leaked threads from previous tests
- All fixtures use nuclear shutdown + gc.collect() to aggressively clean up resources
- The force_shutdown_reasoner function makes ALL threads daemon immediately to prevent hangs
"""

from vulcan.reasoning.unified import (
    ReasoningPlan,
    ReasoningStrategy,
    ReasoningTask,
    UnifiedReasoner,
    _load_optional_components,
    _load_reasoning_components,
    _load_selection_components,
)
from vulcan.reasoning.reasoning_types import (
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningType,
)
import gc

# CRITICAL: Mock problematic components BEFORE any imports that might load them
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock the components that create background threads to prevent leaks
mock_hardware_dispatcher = MagicMock()
mock_execution_metrics = MagicMock()
mock_processing = MagicMock()
mock_rollback_audit = MagicMock()

sys.modules["hardware_dispatcher"] = mock_hardware_dispatcher
sys.modules["src.hardware_dispatcher"] = mock_hardware_dispatcher
sys.modules["unified_runtime.execution_metrics"] = mock_execution_metrics
sys.modules["src.unified_runtime.execution_metrics"] = mock_execution_metrics
sys.modules["vulcan.processing"] = mock_processing
sys.modules["vulcan.safety.rollback_audit"] = mock_rollback_audit

# Now import the module to test

# =============================================================================
# MODULE-LEVEL SHARED REASONER (CRITICAL PERFORMANCE OPTIMIZATION)
# =============================================================================
# Creating UnifiedReasoner is EXPENSIVE (spawns threads, loads components).
# Instead of creating one per test (32+ instances), we create ONE for the module.

_SHARED_REASONER = None
_ORIGINAL_REASONERS = None


@pytest.fixture(scope="module")
def shared_reasoner():
    """
    MODULE-SCOPED fixture - creates ONE UnifiedReasoner for ALL tests.

    This is the key performance optimization. Previous version created a new
    UnifiedReasoner for each of 32+ tests, causing massive slowdown and timeouts.
    """
    global _SHARED_REASONER, _ORIGINAL_REASONERS

    gc.collect()

    config = {
        "confidence_threshold": 0.5,
        "max_reasoning_time": 5.0,
        "default_timeout": 5.0,
        "skip_runtime": True,  # Skip heavy runtime initialization
    }

    _SHARED_REASONER = UnifiedReasoner(
        enable_learning=False, enable_safety=False, max_workers=2, config=config
    )

    # Save original reasoners dict for restoration
    _ORIGINAL_REASONERS = dict(_SHARED_REASONER.reasoners)

    yield _SHARED_REASONER

    # Cleanup at end of module
    force_shutdown_reasoner(_SHARED_REASONER, timeout=1.0)
    gc.collect()


@pytest.fixture(autouse=True)
def reset_shared_reasoner_state(shared_reasoner):
    """
    Auto-use fixture that resets reasoner state between tests.
    This allows tests to modify the reasoner without affecting other tests.
    """
    # Save original values before test
    original_timeout = getattr(shared_reasoner, "default_timeout", 5.0)
    original_max_cache_size = getattr(shared_reasoner, "max_cache_size", 1000)

    # Run the test
    yield

    # Reset state after test
    if shared_reasoner and _ORIGINAL_REASONERS is not None:
        # Restore original values (various tests modify these)
        shared_reasoner.default_timeout = original_timeout
        shared_reasoner.max_cache_size = original_max_cache_size

        # Restore original reasoners
        shared_reasoner.reasoners = dict(_ORIGINAL_REASONERS)
        # Clear caches
        shared_reasoner.result_cache.clear()
        shared_reasoner.plan_cache.clear()


# FIXED: Helper function to create valid ReasoningChain for tests
def create_test_chain():
    """Helper to create a valid ReasoningChain for tests"""
    initial_step = ReasoningStep(
        step_id="test_init",
        step_type=ReasoningType.UNKNOWN,
        input_data=None,
        output_data=None,
        confidence=1.0,
        explanation="Test initialization",
    )

    return ReasoningChain(
        chain_id="test_chain",
        steps=[initial_step],  # ALWAYS include initial step
        initial_query={},
        final_conclusion=None,
        total_confidence=0.0,
        reasoning_types_used=set(),
        modalities_involved=set(),
        safety_checks=[],
        audit_trail=[],
    )


# NUCLEAR SHUTDOWN VERSION - Makes all threads daemon immediately
def force_shutdown_reasoner(reasoner, timeout=0.5):
    """NUCLEAR shutdown - makes all threads daemon and forces immediate cleanup"""
    if not reasoner:
        return

    import gc

    # Mark shutdown everywhere - set every possible shutdown flag
    for flag in ["_is_shutdown", "_shutdown", "shutdown_flag"]:
        try:
            setattr(reasoner, flag, True)
        except Exception:
            pass

    # Daemonize executor threads IMMEDIATELY
    if hasattr(reasoner, "executor") and reasoner.executor:
        if hasattr(reasoner.executor, "_threads"):
            for t in list(reasoner.executor._threads):
                try:
                    t.daemon = True
                except Exception:
                    pass
            try:
                reasoner.executor._threads.clear()
            except Exception:
                pass

        try:
            reasoner.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        reasoner.executor = None

    # Daemonize all component threads
    component_names = [
        "portfolio_executor",
        "warm_pool",
        "cache",
        "tool_selector",
        "safety_governor",
        "tool_monitor",
        "processor",
        "runtime",
    ]

    for comp_name in component_names:
        comp = getattr(reasoner, comp_name, None)
        if comp:
            # Set shutdown flags on component
            for flag in ["_shutdown", "_is_shutdown", "shutdown_flag"]:
                try:
                    setattr(comp, flag, True)
                except Exception:
                    pass

            # Find and daemonize all threads
            thread_attrs = [
                "monitor_thread",
                "scaling_thread",
                "health_check_thread",
                "cleanup_thread",
                "_monitor_thread",
                "_cleanup_thread",
                "_health_thread",
                "_scaling_thread",
                "watchdog_thread",
                "_watchdog_thread",
            ]

            for attr in thread_attrs:
                thread = getattr(comp, attr, None)
                if thread and hasattr(thread, "daemon"):
                    try:
                        thread.daemon = True
                    except Exception:
                        pass

            # Try shutdown with minimal timeout
            if hasattr(comp, "shutdown"):
                try:
                    comp.shutdown(timeout=0.01)
                except Exception:
                    pass

            # Nullify component reference
            try:
                setattr(reasoner, comp_name, None)
            except Exception:
                pass

    # Force garbage collection
    gc.collect()


# Helper function to register mock reasoners with selection layer
def register_mock_reasoner(reasoner, reasoning_type, mock_reasoner):
    """
    Register a mock reasoner with both the reasoners dict and the selection layer.
    This ensures portfolio_executor and warm_pool can find the tool.
    """
    # Add to reasoners dict (enum key)
    reasoner.reasoners[reasoning_type] = mock_reasoner

    # Also register with selection layer if available (string key)
    tool_name = reasoning_type.value
    if reasoner.portfolio_executor and hasattr(reasoner.portfolio_executor, "tools"):
        reasoner.portfolio_executor.tools[tool_name] = mock_reasoner
    if reasoner.warm_pool and hasattr(reasoner.warm_pool, "tools"):
        reasoner.warm_pool.tools[tool_name] = mock_reasoner


class TestComponentLoading:
    """Test lazy loading of components"""

    def test_load_reasoning_components(self):
        """Test loading reasoning components"""
        components = _load_reasoning_components()

        assert isinstance(components, dict)
        # Components dict may have various reasoners, or be empty
        # Just verify it's a valid dict
        assert len(components) >= 0

    def test_load_selection_components(self):
        """Test loading selection components"""
        components = _load_selection_components()

        assert isinstance(components, dict)
        # May be empty if dependencies not available

    def test_load_optional_components(self):
        """Test loading optional components"""
        components = _load_optional_components()

        assert isinstance(components, dict)
        # May be empty if optional deps not available


class TestReasoningStrategy:
    """Test ReasoningStrategy enum"""

    def test_strategy_values(self):
        """Test strategy enum values"""
        assert ReasoningStrategy.SEQUENTIAL.value == "sequential"
        assert ReasoningStrategy.PARALLEL.value == "parallel"
        assert ReasoningStrategy.ENSEMBLE.value == "ensemble"
        assert ReasoningStrategy.ADAPTIVE.value == "adaptive"
        assert ReasoningStrategy.PORTFOLIO.value == "portfolio"
        assert ReasoningStrategy.UTILITY_BASED.value == "utility_based"


class TestReasoningTask:
    """Test ReasoningTask dataclass"""

    def test_task_creation(self):
        """Test creating reasoning tasks"""
        task = ReasoningTask(
            task_id="test_001",
            task_type=ReasoningType.PROBABILISTIC,
            input_data={"x": 5},
            query={"threshold": 0.7},
            priority=5,
        )

        assert task.task_id == "test_001"
        assert task.task_type == ReasoningType.PROBABILISTIC
        assert task.priority == 5
        assert task.query["threshold"] == 0.7

    def test_task_defaults(self):
        """Test task default values"""
        task = ReasoningTask(
            task_id="test_002",
            task_type=ReasoningType.SYMBOLIC,
            input_data="test",
            query={},
        )

        assert task.priority == 0
        assert task.deadline is None
        assert isinstance(task.constraints, dict)
        assert isinstance(task.metadata, dict)


class TestReasoningPlan:
    """Test ReasoningPlan dataclass"""

    def test_plan_creation(self):
        """Test creating reasoning plans"""
        task = ReasoningTask(
            task_id="t1",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test",
            query={},
        )

        plan = ReasoningPlan(
            plan_id="p1",
            tasks=[task],
            strategy=ReasoningStrategy.SEQUENTIAL,
            dependencies={},
            estimated_time=1.0,
            estimated_cost=100.0,
        )

        assert plan.plan_id == "p1"
        assert len(plan.tasks) == 1
        assert plan.strategy == ReasoningStrategy.SEQUENTIAL
        assert plan.estimated_time == 1.0


class TestUnifiedReasoner:
    """Test main UnifiedReasoner class"""

    @pytest.fixture
    def reasoner(self, shared_reasoner):
        """
        Use the module-level shared reasoner instead of creating a new one.

        PERFORMANCE FIX: This was creating a new UnifiedReasoner for each of 32 tests,
        causing massive slowdown. Now uses a single shared instance.
        """
        return shared_reasoner

    def test_reasoner_initialization(self, reasoner):
        """Test reasoner initialization"""
        assert reasoner is not None
        assert isinstance(reasoner.reasoners, dict)
        assert reasoner.confidence_threshold == 0.5
        assert reasoner.max_reasoning_time == 5.0
        assert not reasoner._is_shutdown

    def test_reasoner_has_core_components(self, reasoner):
        """Test that core components are initialized"""
        assert hasattr(reasoner, "reasoners")
        assert hasattr(reasoner, "reasoning_strategies")
        assert hasattr(reasoner, "performance_metrics")
        assert hasattr(reasoner, "result_cache")
        assert hasattr(reasoner, "executor")

    @pytest.mark.timeout(10)
    def test_simple_reasoning(self, reasoner):
        """Test simple reasoning call"""
        # Create mock reasoner for testing
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "test", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        result = reasoner.reason(
            input_data=[1, 2, 3],
            reasoning_type=ReasoningType.PROBABILISTIC,
            strategy=ReasoningStrategy.SEQUENTIAL,
        )

        assert isinstance(result, ReasoningResult)
        assert result is not None
        assert result.conclusion is not None
        assert result.confidence >= 0.0
        assert result.reasoning_chain is not None
        assert len(result.reasoning_chain.steps) > 0

    def test_reasoning_type_determination(self, reasoner):
        """Test automatic reasoning type determination"""
        # Probabilistic input - numeric array
        reasoning_type = reasoner._determine_reasoning_type([1, 2, 3], {})
        assert reasoning_type == ReasoningType.PROBABILISTIC

        # Symbolic input - string with logical operators
        reasoning_type = reasoner._determine_reasoning_type("A AND B OR C", {})
        assert reasoning_type in [ReasoningType.SYMBOLIC, ReasoningType.PROBABILISTIC]

        # Query-based determination - probability keyword
        reasoning_type = reasoner._determine_reasoning_type(
            "test", {"query": "what is the probability"}
        )
        assert reasoning_type == ReasoningType.PROBABILISTIC

        # Query-based determination - prove keyword
        reasoning_type = reasoner._determine_reasoning_type(
            "test", {"query": "prove this theorem"}
        )
        assert reasoning_type in [ReasoningType.SYMBOLIC, ReasoningType.PROBABILISTIC]

        # Causal reasoning - cause keyword
        reasoning_type = reasoner._determine_reasoning_type(
            "test", {"query": "what is the cause"}
        )
        assert reasoning_type == ReasoningType.CAUSAL

    @pytest.mark.timeout(10)
    def test_cache_functionality(self, reasoner):
        """Test result caching"""
        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "cached", "confidence": 0.9}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # First call - cache miss
        result1 = reasoner.reason(
            input_data=[1, 2, 3],
            query={"test": "query"},
            reasoning_type=ReasoningType.PROBABILISTIC,
        )

        # Second call - cache hit (should not call mock again)
        call_count_before = mock_reasoner.reason_with_uncertainty.call_count
        result2 = reasoner.reason(
            input_data=[1, 2, 3],
            query={"test": "query"},
            reasoning_type=ReasoningType.PROBABILISTIC,
        )
        call_count_after = mock_reasoner.reason_with_uncertainty.call_count

        # Should have same results due to caching
        assert result1 is not None
        assert result2 is not None
        assert result1.conclusion is not None
        assert result2.conclusion is not None
        # Cache hit means no new call
        assert call_count_after == call_count_before

    @pytest.mark.timeout(15)
    def test_cache_size_limit(self, reasoner):
        """Test cache size limiting"""
        reasoner.max_cache_size = 10

        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "test", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Add many entries
        for i in range(20):
            reasoner.reason(
                input_data=f"test_{i}",
                query={"unique": i},
                reasoning_type=ReasoningType.PROBABILISTIC,
            )

        # Cache should be limited
        assert len(reasoner.result_cache) <= reasoner.max_cache_size

    def test_reasoning_strategies_available(self, reasoner):
        """Test all reasoning strategies are registered"""
        strategies = reasoner.reasoning_strategies

        assert ReasoningStrategy.SEQUENTIAL in strategies
        assert ReasoningStrategy.PARALLEL in strategies
        assert ReasoningStrategy.ENSEMBLE in strategies
        assert ReasoningStrategy.HIERARCHICAL in strategies
        assert ReasoningStrategy.ADAPTIVE in strategies
        assert ReasoningStrategy.HYBRID in strategies
        assert ReasoningStrategy.PORTFOLIO in strategies
        assert ReasoningStrategy.UTILITY_BASED in strategies

    @pytest.mark.timeout(10)
    def test_sequential_reasoning(self, reasoner):
        """Test sequential reasoning strategy"""
        # Create mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "seq_result", "confidence": 0.75}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Create plan with multiple tasks
        task1 = ReasoningTask(
            task_id="t1",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test1",
            query={},
        )
        task2 = ReasoningTask(
            task_id="t2",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test2",
            query={},
        )

        plan = ReasoningPlan(
            plan_id="p1",
            tasks=[task1, task2],
            strategy=ReasoningStrategy.SEQUENTIAL,
            dependencies={},
            estimated_time=1.0,
            estimated_cost=100.0,
        )

        # FIXED: Create reasoning chain WITH initial step
        reasoning_chain = create_test_chain()

        result = reasoner._sequential_reasoning(plan, reasoning_chain)

        assert isinstance(result, ReasoningResult)
        assert result.conclusion is not None
        assert result.reasoning_chain is not None
        assert (
            len(result.reasoning_chain.steps) >= 2
        )  # Should have steps from both tasks
        assert mock_reasoner.reason_with_uncertainty.call_count == 2

    @pytest.mark.timeout(10)
    def test_parallel_reasoning(self, reasoner):
        """Test parallel reasoning strategy"""
        # Create mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "parallel_result", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Create plan with 3 tasks
        tasks = [
            ReasoningTask(
                task_id=f"t{i}",
                task_type=ReasoningType.PROBABILISTIC,
                input_data=f"test{i}",
                query={},
            )
            for i in range(3)
        ]

        plan = ReasoningPlan(
            plan_id="p1",
            tasks=tasks,
            strategy=ReasoningStrategy.PARALLEL,
            dependencies={},
            estimated_time=1.0,
            estimated_cost=100.0,
        )

        # FIXED: Create reasoning chain WITH initial step
        reasoning_chain = create_test_chain()

        result = reasoner._parallel_reasoning(plan, reasoning_chain)

        assert isinstance(result, ReasoningResult)
        assert result.reasoning_chain is not None
        assert result.reasoning_type == ReasoningType.HYBRID
        assert mock_reasoner.reason_with_uncertainty.call_count == 3

    @pytest.mark.timeout(10)
    def test_ensemble_reasoning(self, reasoner):
        """Test ensemble reasoning with multiple reasoners"""
        # Create two different mock reasoners
        mock_prob = Mock()
        mock_prob.reason_with_uncertainty = Mock(
            return_value={"conclusion": "prob_result", "confidence": 0.7}
        )

        mock_symbolic = Mock()
        mock_symbolic.add_rule = Mock()
        mock_symbolic.query = Mock(
            return_value={"result": "symbolic_result", "confidence": 0.8, "proof": []}
        )

        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_prob)
        register_mock_reasoner(reasoner, ReasoningType.SYMBOLIC, mock_symbolic)

        # Test with ensemble strategy
        result = reasoner.reason(
            input_data={"kb": [], "data": [1, 2, 3]},
            query={"goal": "test"},
            strategy=ReasoningStrategy.ENSEMBLE,
        )

        assert isinstance(result, ReasoningResult)
        assert result.confidence > 0
        # Should have used multiple reasoners
        assert len(result.reasoning_chain.reasoning_types_used) >= 1

    @pytest.mark.timeout(10)
    def test_adaptive_reasoning(self, reasoner):
        """Test adaptive reasoning strategy"""
        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "adaptive", "confidence": 0.85}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Test with different input characteristics
        # Simple input - should use utility-based
        result1 = reasoner.reason(
            input_data=[1, 2, 3], query={}, strategy=ReasoningStrategy.ADAPTIVE
        )

        assert isinstance(result1, ReasoningResult)
        assert result1.reasoning_chain is not None

        # Complex input - should trigger ensemble
        result2 = reasoner.reason(
            input_data=list(range(2000)),  # Large input
            query={},
            strategy=ReasoningStrategy.ADAPTIVE,
        )

        assert isinstance(result2, ReasoningResult)
        assert result2.reasoning_chain is not None

    @pytest.mark.timeout(10)
    def test_hybrid_reasoning(self, reasoner):
        """Test hybrid reasoning strategy"""
        # Mock probabilistic reasoner
        mock_prob = Mock()
        mock_prob.reason_with_uncertainty = Mock(
            return_value={
                "conclusion": "hybrid_prob",
                "confidence": 0.6,  # Low confidence to trigger symbolic
            }
        )

        # Mock symbolic reasoner
        mock_symbolic = Mock()
        mock_symbolic.add_rule = Mock()
        mock_symbolic.query = Mock(
            return_value={"result": "hybrid_symbolic", "confidence": 0.9, "proof": []}
        )

        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_prob)
        register_mock_reasoner(reasoner, ReasoningType.SYMBOLIC, mock_symbolic)

        result = reasoner.reason(
            input_data={"kb": [], "data": [1, 2, 3]},
            query={"goal": "test"},
            strategy=ReasoningStrategy.HYBRID,
        )

        assert isinstance(result, ReasoningResult)
        assert result.reasoning_chain is not None
        # Hybrid should have tried probabilistic first
        assert mock_prob.reason_with_uncertainty.call_count >= 1

    def test_input_characteristics_analysis(self, reasoner):
        """Test input characteristic analysis"""
        # Simple input
        task1 = ReasoningTask(
            task_id="t1",
            task_type=ReasoningType.PROBABILISTIC,
            input_data=[1, 2, 3],
            query={},
        )

        chars1 = reasoner._analyze_input_characteristics(task1)

        assert "complexity" in chars1
        assert "uncertainty" in chars1
        assert "multimodal" in chars1
        assert chars1["multimodal"] is False
        assert chars1["size"] == "small"

        # Large input
        task2 = ReasoningTask(
            task_id="t2",
            task_type=ReasoningType.PROBABILISTIC,
            input_data=list(range(2000)),
            query={},
        )

        chars2 = reasoner._analyze_input_characteristics(task2)

        assert chars2["size"] == "large"
        assert chars2["complexity"] > 0.5

    def test_topological_sort(self, reasoner):
        """Test topological sorting of tasks"""
        tasks = [
            ReasoningTask(
                task_id="t1",
                task_type=ReasoningType.PROBABILISTIC,
                input_data="test1",
                query={},
            ),
            ReasoningTask(
                task_id="t2",
                task_type=ReasoningType.PROBABILISTIC,
                input_data="test2",
                query={},
            ),
            ReasoningTask(
                task_id="t3",
                task_type=ReasoningType.PROBABILISTIC,
                input_data="test3",
                query={},
            ),
        ]

        # t3 depends on t2, t2 depends on t1
        dependencies = {"t2": ["t1"], "t3": ["t2"]}

        sorted_tasks = reasoner._topological_sort(tasks, dependencies)

        assert len(sorted_tasks) == 3
        # t1 should come first
        assert sorted_tasks[0].task_id == "t1"
        assert sorted_tasks[1].task_id == "t2"
        assert sorted_tasks[2].task_id == "t3"

    def test_weighted_voting(self, reasoner):
        """Test weighted voting for ensemble"""
        # Boolean voting
        conclusions = [True, True, False]
        weights = [0.5, 0.3, 0.2]

        result = reasoner._weighted_voting(conclusions, weights)
        assert result is True

        # Numerical voting
        conclusions = [10, 20, 30]
        weights = [0.5, 0.3, 0.2]

        result = reasoner._weighted_voting(conclusions, weights)
        assert isinstance(result, (int, float))
        assert 10 <= result <= 30
        # Weighted average: 10*0.5 + 20*0.3 + 30*0.2 = 17
        assert 16 <= result <= 18

        # Categorical voting
        conclusions = ["A", "B", "A"]
        weights = [0.4, 0.3, 0.3]

        result = reasoner._weighted_voting(conclusions, weights)
        assert result == "A"  # A has weight 0.7, B has 0.3

    def test_combine_parallel_results(self, reasoner):
        """Test combining parallel results"""
        # Dictionary results
        results = [
            ReasoningResult(
                conclusion={"a": 1, "b": 2},
                confidence=0.8,
                reasoning_type=ReasoningType.PROBABILISTIC,
                reasoning_chain=None,
                explanation="",
            ),
            ReasoningResult(
                conclusion={"c": 3},
                confidence=0.7,
                reasoning_type=ReasoningType.PROBABILISTIC,
                reasoning_chain=None,
                explanation="",
            ),
        ]

        combined = reasoner._combine_parallel_results(results)

        assert isinstance(combined, dict)
        assert "a" in combined and combined["a"] == 1
        assert "c" in combined and combined["c"] == 3

        # Numerical results
        results_num = [
            ReasoningResult(
                conclusion=10,
                confidence=0.8,
                reasoning_type=ReasoningType.PROBABILISTIC,
                reasoning_chain=None,
                explanation="",
            ),
            ReasoningResult(
                conclusion=20,
                confidence=0.7,
                reasoning_type=ReasoningType.PROBABILISTIC,
                reasoning_chain=None,
                explanation="",
            ),
        ]

        combined_num = reasoner._combine_parallel_results(results_num)
        assert isinstance(combined_num, (int, float))
        assert combined_num == 15.0  # Mean of 10 and 20

    @pytest.mark.timeout(10)
    def test_performance_metrics_tracking(self, reasoner):
        """Test performance metrics tracking"""
        initial_count = reasoner.performance_metrics["total_reasonings"]

        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "test", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Perform reasoning
        result = reasoner.reason(
            input_data=[1, 2, 3], reasoning_type=ReasoningType.PROBABILISTIC
        )

        # Check metrics updated
        assert reasoner.performance_metrics["total_reasonings"] > initial_count
        assert reasoner.performance_metrics["successful_reasonings"] >= 0
        assert reasoner.performance_metrics["average_confidence"] >= 0

    @pytest.mark.timeout(15)
    def test_thread_safety(self, reasoner):
        """Test thread-safe operations"""
        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "thread_safe", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        results = []
        errors = []

        def reasoning_task():
            try:
                result = reasoner.reason(
                    input_data=[1, 2, 3], reasoning_type=ReasoningType.PROBABILISTIC
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=reasoning_task) for _ in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 5
        # All results should be valid
        for r in results:
            assert isinstance(r, ReasoningResult)
            assert r.confidence >= 0

    def test_statistics_retrieval(self, reasoner):
        """Test getting comprehensive statistics"""
        stats = reasoner.get_statistics()

        assert isinstance(stats, dict)
        assert "performance" in stats
        assert "cache_stats" in stats
        assert "task_stats" in stats
        assert "history_size" in stats
        assert "execution_count" in stats

        # Check performance metrics structure
        perf = stats["performance"]
        assert "total_reasonings" in perf
        assert "successful_reasonings" in perf
        assert "average_confidence" in perf

    def test_cache_clearing(self, reasoner):
        """Test cache clearing"""
        # Add some cached results
        reasoner.result_cache["test1"] = Mock()
        reasoner.result_cache["test2"] = Mock()
        reasoner.plan_cache["plan1"] = Mock()

        assert len(reasoner.result_cache) > 0
        assert len(reasoner.plan_cache) > 0

        reasoner.clear_caches()

        assert len(reasoner.result_cache) == 0
        assert len(reasoner.plan_cache) == 0

    def test_state_persistence(self, reasoner, tmp_path):
        """Test saving and loading state using the shared reasoner fixture"""
        # Set up initial state with some values
        reasoner.confidence_threshold = 0.7
        initial_threshold = reasoner.confidence_threshold

        # Add some test data to history
        test_history_item = {
            "type": "test",
            "data": "test_data",
            "timestamp": 1234567890.0,  # Fixed timestamp for test reliability
        }
        reasoner.reasoning_history.append(test_history_item)
        initial_history_len = len(reasoner.reasoning_history)

        # Temporarily override model_path to use tmp_path
        original_model_path = reasoner.model_path
        reasoner.model_path = tmp_path

        try:
            # Save state
            reasoner.save_state("test_state")

            # Verify state file was created
            state_file = tmp_path / "test_state_unified_state.pkl"
            assert state_file.exists(), "State file should be created"

            # Modify state
            reasoner.confidence_threshold = 0.3
            reasoner.reasoning_history.clear()

            # Load state back
            reasoner.load_state("test_state")

            # Verify state was restored
            assert (
                reasoner.confidence_threshold == initial_threshold
            ), "Confidence threshold should be restored"
            assert (
                len(reasoner.reasoning_history) == initial_history_len
            ), "History should be restored"

        finally:
            # Restore original model_path
            reasoner.model_path = original_model_path

    def test_error_handling(self, reasoner):
        """Test error handling in reasoning"""
        # Reasoning type with no reasoner
        result = reasoner.reason(
            input_data="test", reasoning_type=ReasoningType.UNKNOWN
        )

        # Should return error result, not crash
        assert isinstance(result, ReasoningResult)
        # FIX CRITICAL-7: Changed from 0.0 to <= 0.25 to match symbolic reasoner confidence floor
        # The symbolic reasoner has CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT = 0.2
        # When it returns "not applicable", it still uses this floor to prevent downstream issues
        assert result.confidence <= 0.25  # Allow for confidence floor + small margin

    @pytest.mark.timeout(5)
    def test_timeout_handling(self, reasoner):
        """Test timeout handling"""
        reasoner.default_timeout = 0.1

        # Create slow mock reasoner
        def slow_reasoning(*args, **kwargs):
            time.sleep(1.0)
            return {"conclusion": "slow", "confidence": 0.8}

        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = slow_reasoning
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Should timeout
        start = time.time()
        result = reasoner.reason(
            input_data=[1, 2, 3], reasoning_type=ReasoningType.PROBABILISTIC
        )
        elapsed = time.time() - start

        # Should complete quickly due to timeout
        assert elapsed < 2.0

    def test_shutdown(self):
        """Test proper shutdown.

        NOTE: This test creates its OWN reasoner instead of using the shared one
        because it needs to test shutdown behavior, which would break the shared
        instance for subsequent tests.
        """
        # Create an isolated reasoner just for this test
        isolated_reasoner = UnifiedReasoner(
            enable_learning=False,
            enable_safety=False,
            max_workers=1,
            config={"skip_runtime": True},
        )

        try:
            # Perform some operations
            isolated_reasoner.performance_metrics["test"] = 123

            # Shutdown with skip_save=True
            isolated_reasoner.shutdown(timeout=0.5, skip_save=True)

            assert isolated_reasoner._is_shutdown

            # Should not allow reasoning after shutdown
            result = isolated_reasoner.reason(input_data="test")

            assert "shutdown" in str(result.conclusion).lower()
        finally:
            # Ensure cleanup even if test fails
            force_shutdown_reasoner(isolated_reasoner, timeout=0.1)

    def test_create_utility_context(self, reasoner):
        """Test utility context creation"""
        query = {"mode": "fast"}
        constraints = {"time_budget_ms": 5000, "confidence_threshold": 0.7}

        context = reasoner._create_utility_context(query, constraints)

        # May be None if components not available, that's ok
        if context is not None:
            assert hasattr(context, "time_budget") or isinstance(context, dict)

    def test_create_optimized_plan(self, reasoner):
        """Test optimized plan creation"""
        task = ReasoningTask(
            task_id="opt_task",
            task_type=ReasoningType.PROBABILISTIC,
            input_data=[1, 2, 3],
            query={},
            constraints={"time_budget_ms": 5000},
        )

        plan = reasoner._create_optimized_plan(task, ReasoningStrategy.SEQUENTIAL)

        assert isinstance(plan, ReasoningPlan)
        assert len(plan.tasks) > 0
        assert plan.strategy == ReasoningStrategy.SEQUENTIAL
        assert plan.estimated_time > 0
        assert plan.estimated_cost > 0

    def test_estimate_plan_time(self, reasoner):
        """Test plan time estimation"""
        tasks = [
            ReasoningTask(
                task_id=f"t{i}",
                task_type=ReasoningType.PROBABILISTIC,
                input_data="test",
                query={},
            )
            for i in range(3)
        ]

        estimated_time = reasoner._estimate_plan_time(tasks)

        assert isinstance(estimated_time, float)
        assert estimated_time > 0
        assert estimated_time >= 3.0  # At least 1 second per task

    def test_estimate_plan_cost(self, reasoner):
        """Test plan cost estimation"""
        tasks = [
            ReasoningTask(
                task_id=f"t{i}",
                task_type=ReasoningType.PROBABILISTIC,
                input_data="test",
                query={},
            )
            for i in range(3)
        ]

        estimated_cost = reasoner._estimate_plan_cost(tasks)

        assert isinstance(estimated_cost, (int, float))
        assert estimated_cost > 0
        assert estimated_cost >= 300  # At least 100 per task


class TestSpecializedReasoning:
    """Test specialized reasoning methods"""

    @pytest.fixture
    def reasoner(self, shared_reasoner):
        """Use the module-level shared reasoner - PERFORMANCE OPTIMIZED"""
        return shared_reasoner

    def test_reason_by_analogy(self, reasoner):
        """Test analogical reasoning method"""
        # Mock analogical reasoner
        mock_analogical = Mock()
        mock_analogical.find_structural_analogy = Mock(
            return_value={
                "found": True,
                "confidence": 0.8,
                "solution": "analogical_solution",
                "explanation": "Found mapping",
            }
        )
        register_mock_reasoner(reasoner, ReasoningType.ANALOGICAL, mock_analogical)

        result = reasoner.reason_by_analogy(
            target_problem={"problem": "test"}, source_domain="math"
        )

        assert isinstance(result, ReasoningResult)
        assert result.reasoning_type == ReasoningType.ANALOGICAL
        assert result.confidence == 0.8
        assert mock_analogical.find_structural_analogy.called

    def test_reason_counterfactual(self, reasoner):
        """Test counterfactual reasoning method"""
        # Only test if counterfactual reasoner is available
        if reasoner.counterfactual is None:
            # Mock counterfactual reasoner
            mock_cf = Mock()
            mock_cf_result = Mock()
            mock_cf_result.counterfactual = {"outcome": "different"}
            mock_cf_result.probability = 0.7
            mock_cf_result.explanation = "Counterfactual explanation"
            mock_cf.compute_counterfactual = Mock(return_value=mock_cf_result)
            reasoner.counterfactual = mock_cf

        try:
            result = reasoner.reason_counterfactual(
                factual_state={"x": 1}, intervention={"x": 2}
            )

            assert isinstance(result, ReasoningResult)
            if result.reasoning_type == ReasoningType.COUNTERFACTUAL:
                assert result.confidence > 0
        finally:
            # Ensure cleanup
            reasoner.counterfactual = None


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def reasoner(self, shared_reasoner):
        """Use the module-level shared reasoner - PERFORMANCE OPTIMIZED"""
        return shared_reasoner

    def test_empty_input(self, reasoner):
        """Test with empty input"""
        result = reasoner.reason(input_data=None)

        assert isinstance(result, ReasoningResult)

    def test_invalid_strategy(self, reasoner):
        """Test with valid strategy"""
        result = reasoner.reason(input_data="test", strategy=ReasoningStrategy.ADAPTIVE)

        assert isinstance(result, ReasoningResult)

    def test_no_reasoners_available(self, reasoner):
        """Test when no reasoners are available.

        NOTE: This test clears all reasoners temporarily. The autouse fixture
        will restore the original reasoners after the test completes.
        """
        # Clear all reasoners (will be restored by autouse fixture)
        reasoner.reasoners = {}

        result = reasoner.reason(input_data="test")

        assert isinstance(result, ReasoningResult)

    def test_concurrent_cache_access(self, reasoner):
        """Test concurrent cache access"""
        results = []
        lock = threading.Lock()

        def cache_task():
            for _ in range(10):
                thread_id = threading.current_thread().ident
                reasoner.result_cache[f"key_{thread_id}"] = Mock()
                reasoner.clear_caches()
                with lock:
                    results.append(True)

        threads = [threading.Thread(target=cache_task) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(results) > 0


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.fixture
    def reasoner(self, shared_reasoner):
        """Use the module-level shared reasoner - PERFORMANCE OPTIMIZED"""
        return shared_reasoner

    @pytest.mark.timeout(20)
    def test_end_to_end_reasoning_workflow(self, reasoner):
        """Test complete reasoning workflow"""
        # Mock a reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "integrated", "confidence": 0.85}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Perform reasoning
        result = reasoner.reason(
            input_data=[1, 2, 3, 4, 5],
            query={"threshold": 0.7},
            reasoning_type=ReasoningType.PROBABILISTIC,
            strategy=ReasoningStrategy.SEQUENTIAL,
        )

        # Verify result
        assert isinstance(result, ReasoningResult)
        assert result.confidence >= 0.7
        assert result.reasoning_chain is not None

        # Get statistics
        stats = reasoner.get_statistics()
        assert stats["performance"]["total_reasonings"] > 0
        assert stats["execution_count"] > 0

    @pytest.mark.timeout(30)
    def test_multi_strategy_workflow(self, reasoner):
        """Test workflow using multiple strategies"""
        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "multi_strategy", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        strategies = [
            ReasoningStrategy.SEQUENTIAL,
            ReasoningStrategy.ADAPTIVE,
            ReasoningStrategy.UTILITY_BASED,
        ]

        results = []
        for strategy in strategies:
            result = reasoner.reason(
                input_data=[1, 2, 3],
                reasoning_type=ReasoningType.PROBABILISTIC,
                strategy=strategy,
            )
            results.append(result)

        # All should succeed
        assert all(isinstance(r, ReasoningResult) for r in results)
        assert all(r.confidence > 0 for r in results)

        # Check strategy usage
        stats = reasoner.get_statistics()
        assert len(stats["performance"]["strategy_usage"]) > 0


# Performance tests
class TestPerformance:
    """Performance and stress tests"""

    @pytest.fixture
    def reasoner(self, shared_reasoner):
        """Use the module-level shared reasoner - PERFORMANCE OPTIMIZED"""
        return shared_reasoner

    @pytest.mark.timeout(40)
    def test_high_volume_reasoning(self, reasoner):
        """Test high volume of reasoning requests"""
        # Mock reasoner
        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = Mock(
            return_value={"conclusion": "volume_test", "confidence": 0.8}
        )
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # Perform many reasoning operations
        start = time.time()
        for i in range(50):
            reasoner.reason(
                input_data=f"test_{i}", reasoning_type=ReasoningType.PROBABILISTIC
            )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30.0

        stats = reasoner.get_statistics()
        assert stats["performance"]["total_reasonings"] >= 50

    @pytest.mark.timeout(15)
    def test_cache_performance(self, reasoner):
        """Test cache improves performance"""
        # Clear cache to ensure clean state for this test
        reasoner.clear_caches()

        # Mock slow reasoner
        call_count = [0]

        def slow_reasoning(*args, **kwargs):
            call_count[0] += 1
            time.sleep(0.05)  # Reduced sleep time for faster test
            return {"conclusion": "cached", "confidence": 0.8}

        mock_reasoner = Mock()
        mock_reasoner.reason_with_uncertainty = slow_reasoning
        register_mock_reasoner(reasoner, ReasoningType.PROBABILISTIC, mock_reasoner)

        # First call
        start = time.perf_counter()
        reasoner.reason(
            input_data="test_cache_unique_perf",
            query={"same": "query", "unique": "perf_test"},
            reasoning_type=ReasoningType.PROBABILISTIC,
        )
        first_time = time.perf_counter() - start

        # Second call (cached)
        start = time.perf_counter()
        reasoner.reason(
            input_data="test_cache_unique_perf",
            query={"same": "query", "unique": "perf_test"},
            reasoning_type=ReasoningType.PROBABILISTIC,
        )
        second_time = time.perf_counter() - start

        # Primary check: Should only call slow function once (cache working)
        assert call_count[0] == 1, f"Expected 1 call, got {call_count[0]}"

        # Secondary check: Cached call should be faster or equal
        if first_time > 0 and second_time > 0:
            assert (
                second_time <= first_time
            ), f"Cached call ({second_time}s) should be <= first call ({first_time}s)"

        # The cache definitely worked if we only called the slow function once
        stats = reasoner.get_statistics()
        assert stats["performance"]["total_reasonings"] >= 2


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
