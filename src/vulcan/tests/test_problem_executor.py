"""
test_problem_executor.py - Comprehensive tests for problem_executor module
Tests execution of decomposition plans with safety validation
"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vulcan.problem_decomposer.problem_decomposer_core import (
    DecompositionPlan, ExecutionOutcome, ProblemGraph)
from vulcan.problem_decomposer.problem_executor import (ExecutionStrategy,
                                                        ProblemExecutor)

# Import or define Principle
try:
    from vulcan.validation.validation_engine import DomainTestCase, Principle
except ImportError:
    from dataclasses import dataclass, field
    from typing import Any, Callable, Optional

    @dataclass
    class Principle:
        id: str
        core_pattern: Any
        confidence: float
        execution_logic: Optional[Callable] = None
        execution_type: str = "function"
        applicable_domains: List[str] = field(default_factory=list)

        def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            if self.execution_logic and callable(self.execution_logic):
                return self.execution_logic(inputs)
            raise NotImplementedError(f"No execution logic for principle {self.id}")

    @dataclass
    class DomainTestCase:
        domain: str
        test_id: str
        inputs: Dict[str, Any]
        expected_outputs: Any


# Test Fixtures


@pytest.fixture
def mock_safety_validator():
    """Mock safety validator"""
    validator = Mock()
    validator.validate_state.return_value = {"safe": True}
    return validator


@pytest.fixture
def mock_semantic_bridge():
    """Mock semantic bridge"""
    bridge = Mock()
    return bridge


@pytest.fixture
def basic_executor(mock_safety_validator):
    """Basic executor with mocked safety validator"""
    with patch(
        "vulcan.problem_decomposer.problem_executor.SAFETY_VALIDATOR_AVAILABLE", True
    ):
        with patch(
            "vulcan.problem_decomposer.problem_executor.EnhancedSafetyValidator",
            return_value=mock_safety_validator,
        ):
            executor = ProblemExecutor()
            executor.safety_validator = mock_safety_validator
            return executor


@pytest.fixture
def executor_no_safety():
    """Executor without safety validator"""
    with patch(
        "vulcan.problem_decomposer.problem_executor.SAFETY_VALIDATOR_AVAILABLE", False
    ):
        executor = ProblemExecutor()
        return executor


@pytest.fixture
def simple_problem_graph():
    """Simple problem graph for testing"""
    return ProblemGraph(
        nodes={
            "node1": {
                "type": "operation",
                "operation": "sum",
                "parameters": {"input": "data"},
            },
            "node2": {
                "type": "transform",
                "transform": "scale",
                "parameters": {"factor": 2.0},
            },
            "node3": {
                "type": "decision",
                "condition": {"variable": "x", "operator": ">", "threshold": 5},
            },
        },
        edges=[("node1", "node2", {}), ("node2", "node3", {})],
        metadata={"domain": "general", "description": "Simple test problem"},
    )


@pytest.fixture
def hierarchical_problem_graph():
    """Hierarchical problem graph"""
    return ProblemGraph(
        nodes={
            "root": {"type": "operation", "operation": "sum"},
            "child1": {"type": "operation", "operation": "product"},
            "child2": {"type": "transform", "transform": "normalize"},
            "leaf1": {"type": "operation", "operation": "filter"},
            "leaf2": {"type": "operation", "operation": "map"},
        },
        edges=[
            ("root", "child1", {}),
            ("root", "child2", {}),
            ("child1", "leaf1", {}),
            ("child2", "leaf2", {}),
        ],
        metadata={"domain": "optimization"},
    )


@pytest.fixture
def simple_plan():
    """Simple decomposition plan"""
    return DecompositionPlan(
        steps=[
            {
                "type": "structural_match",
                "structure": "hierarchical",
                "nodes": ["node1", "node2"],
                "confidence": 0.8,
            },
            {
                "type": "exact_match",
                "pattern_id": "linear",
                "nodes": ["node3"],
                "confidence": 0.9,
            },
        ],
        strategy=Mock(is_parallelizable=lambda: False),
        confidence=0.85,
    )


@pytest.fixture
def complex_plan():
    """Complex multi-step plan"""
    return DecompositionPlan(
        steps=[
            {
                "type": "structural_match",
                "structure": "modular",
                "nodes": ["node1"],
                "confidence": 0.7,
            },
            {
                "type": "semantic_match",
                "concept": "semantic_cluster",
                "similarity": 0.75,
                "nodes": ["node2"],
                "confidence": 0.6,
            },
            {"type": "synthetic_bridge", "template": "parallel", "confidence": 0.8},
            {
                "type": "analogical",
                "source_domain": "sorting",
                "target_mapping": {"input": "data"},
                "confidence": 0.7,
            },
        ],
        strategy=None,
        confidence=0.7,
    )


# Test Initialization


def test_executor_initialization_with_safety():
    """Test executor initializes with safety validator"""
    with patch(
        "vulcan.problem_decomposer.problem_executor.SAFETY_VALIDATOR_AVAILABLE", True
    ):
        with patch(
            "vulcan.problem_decomposer.problem_executor.EnhancedSafetyValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            executor = ProblemExecutor()

            assert executor.safety_validator is not None
            assert executor.total_executions == 0
            assert executor.successful_executions == 0
            assert len(executor.solvers) > 0


def test_executor_initialization_without_safety():
    """Test executor initializes without safety validator"""
    with patch(
        "vulcan.problem_decomposer.problem_executor.SAFETY_VALIDATOR_AVAILABLE", False
    ):
        executor = ProblemExecutor()

        assert executor.safety_validator is None
        assert executor.total_executions == 0


def test_executor_with_custom_config():
    """Test executor with custom safety config"""
    config = {"max_recursion_depth": 100, "timeout": 60}

    with patch(
        "vulcan.problem_decomposer.problem_executor.SAFETY_VALIDATOR_AVAILABLE", True
    ):
        with patch(
            "vulcan.problem_decomposer.problem_executor.SafetyConfig"
        ) as mock_config_class:
            with patch(
                "vulcan.problem_decomposer.problem_executor.EnhancedSafetyValidator"
            ) as mock_validator_class:
                # Set up the mock chain properly
                mock_config = Mock()
                mock_config_class.from_dict.return_value = mock_config
                mock_validator = Mock()
                mock_validator_class.return_value = mock_validator

                executor = ProblemExecutor(safety_config=config)

                mock_config_class.from_dict.assert_called_once_with(config)
                assert executor.safety_validator is not None


# Test Plan Execution


def test_execute_plan_requires_safety_validator():
    """Test that execute_plan requires safety validator"""
    executor = ProblemExecutor()
    executor.safety_validator = None

    graph = ProblemGraph(nodes={}, edges=[])
    plan = DecompositionPlan(steps=[], confidence=0.5)

    with pytest.raises(RuntimeError, match="SAFETY CRITICAL"):
        executor.execute_plan(graph, plan)


def test_execute_plan_blocks_unsafe_plan(basic_executor, simple_problem_graph):
    """Test that unsafe plans are blocked"""
    # Create plan with unsafe step type
    unsafe_plan = DecompositionPlan(
        steps=[{"type": "shell_command", "command": "rm -rf /", "confidence": 0.9}],
        confidence=0.9,
    )

    outcome = basic_executor.execute_plan(simple_problem_graph, unsafe_plan)

    assert not outcome.success
    assert "safety_blocked" in outcome.metrics
    assert len(outcome.errors) > 0
    assert "unsafe step type" in outcome.errors[0].lower()


def test_execute_plan_blocks_excessive_steps(basic_executor, simple_problem_graph):
    """Test that plans with too many steps are blocked"""
    # Create plan with excessive steps
    excessive_plan = DecompositionPlan(
        steps=[{"type": "generic", "confidence": 0.5} for _ in range(150)],
        confidence=0.5,
    )

    outcome = basic_executor.execute_plan(simple_problem_graph, excessive_plan)

    assert not outcome.success
    assert "safety_blocked" in outcome.metrics


def test_execute_plan_simple_success(basic_executor, simple_problem_graph, simple_plan):
    """Test successful execution of simple plan"""
    outcome = basic_executor.execute_plan(simple_problem_graph, simple_plan)

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.execution_time >= 0
    assert "plan_confidence" in outcome.metrics
    assert "num_steps" in outcome.metrics
    assert outcome.metrics["safety_validated"] == True


def test_execute_plan_caching(basic_executor, simple_problem_graph, simple_plan):
    """Test that solutions are cached"""
    # First execution
    basic_executor.execute_plan(simple_problem_graph, simple_plan)

    # Second execution should use cache
    outcome2 = basic_executor.execute_plan(simple_problem_graph, simple_plan)

    assert "from_cache" in outcome2.metadata
    assert outcome2.metadata["from_cache"] == True


def test_execute_plan_updates_statistics(
    basic_executor, simple_problem_graph, simple_plan
):
    """Test that execution updates statistics"""
    initial_count = basic_executor.total_executions

    basic_executor.execute_plan(simple_problem_graph, simple_plan)

    assert basic_executor.total_executions == initial_count + 1
    assert len(basic_executor.execution_history) > 0


# Test Step Conversion


def test_convert_structural_step(basic_executor, simple_problem_graph):
    """Test conversion of structural step to principle"""
    steps = [
        {
            "type": "structural_match",
            "structure": "hierarchical",
            "nodes": ["node1", "node2"],
            "confidence": 0.8,
        }
    ]

    principles = basic_executor._convert_steps_to_principles(
        steps, simple_problem_graph
    )

    assert len(principles) == 1
    assert principles[0].id.startswith("step_0_structural_match")
    assert callable(principles[0].execution_logic)


def test_convert_semantic_step(basic_executor, simple_problem_graph):
    """Test conversion of semantic step to principle"""
    steps = [
        {
            "type": "semantic_match",
            "concept": "clustering",
            "similarity": 0.75,
            "nodes": ["node1"],
            "confidence": 0.7,
        }
    ]

    principles = basic_executor._convert_steps_to_principles(
        steps, simple_problem_graph
    )

    assert len(principles) == 1
    assert "semantic_match" in principles[0].id


def test_convert_all_step_types(basic_executor, simple_problem_graph):
    """Test conversion of all step types"""
    steps = [
        {
            "type": "structural_match",
            "structure": "hierarchical",
            "nodes": ["node1"],
            "confidence": 0.8,
        },
        {
            "type": "semantic_match",
            "concept": "test",
            "similarity": 0.7,
            "nodes": ["node2"],
            "confidence": 0.7,
        },
        {
            "type": "exact_match",
            "pattern_id": "linear",
            "nodes": ["node3"],
            "confidence": 0.9,
        },
        {"type": "synthetic_bridge", "template": "simple", "confidence": 0.6},
        {
            "type": "analogical",
            "source_domain": "sorting",
            "target_mapping": {},
            "confidence": 0.7,
        },
        {"type": "brute_force", "part": 1, "content": {}, "confidence": 0.5},
    ]

    principles = basic_executor._convert_steps_to_principles(
        steps, simple_problem_graph
    )

    assert len(principles) == 6
    for principle in principles:
        assert callable(principle.execution_logic)


def test_convert_fallback_on_error(basic_executor, simple_problem_graph):
    """Test fallback principle creation on error"""
    steps = [{"type": "unknown_type", "confidence": 0.5}]

    principles = basic_executor._convert_steps_to_principles(
        steps, simple_problem_graph
    )

    assert len(principles) == 1


# Test Execution Strategies


def test_determine_strategy_parallel(basic_executor):
    """Test parallel strategy determination"""
    plan = DecompositionPlan(
        steps=[{"type": "generic"}],
        strategy=Mock(is_parallelizable=lambda: True),
        confidence=0.8,
    )

    principles = [Mock() for _ in range(3)]

    strategy = basic_executor._determine_execution_strategy(plan, principles)

    assert strategy == ExecutionStrategy.PARALLEL


def test_determine_strategy_iterative(basic_executor):
    """Test iterative strategy determination"""
    plan = DecompositionPlan(
        steps=[{"type": "recursive", "structure": "recursive"}],
        strategy=None,
        confidence=0.8,
    )

    principles = [Mock()]

    strategy = basic_executor._determine_execution_strategy(plan, principles)

    assert strategy == ExecutionStrategy.ITERATIVE


def test_determine_strategy_sequential(basic_executor):
    """Test sequential strategy determination"""
    plan = DecompositionPlan(steps=[{"type": "generic"}], strategy=None, confidence=0.8)

    principles = [Mock()]

    strategy = basic_executor._determine_execution_strategy(plan, principles)

    assert strategy == ExecutionStrategy.SEQUENTIAL


# Test Sequential Execution


def test_execute_sequential_success(basic_executor, simple_problem_graph):
    """Test successful sequential execution"""
    # Create principles with mock execution
    principles = []
    for i in range(3):
        principle = Principle(
            id=f"test_principle_{i}",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True, "result": i},
        )
        principles.append(principle)

    outcome = basic_executor._execute_sequential(principles, simple_problem_graph)

    assert outcome.success
    assert len(outcome.sub_results) == 3
    assert all(r["success"] for r in outcome.sub_results)


def test_execute_sequential_failure(basic_executor, simple_problem_graph):
    """Test sequential execution stops on failure"""

    # Create principles with one that fails
    def failing_logic(inputs):
        raise ValueError("Test failure")

    principles = [
        Principle(
            id="p1",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True},
        ),
        Principle(
            id="p2", core_pattern={}, confidence=0.8, execution_logic=failing_logic
        ),
        Principle(
            id="p3",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True},
        ),
    ]

    outcome = basic_executor._execute_sequential(principles, simple_problem_graph)

    assert not outcome.success
    assert len(outcome.sub_results) == 2  # Only first two executed
    assert len(outcome.errors) > 0


# Test Parallel Execution


def test_execute_parallel_all_success(basic_executor, simple_problem_graph):
    """Test parallel execution with all successes"""
    principles = [
        Principle(
            id=f"p{i}",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True, "result": i},
        )
        for i in range(3):
    ]

    outcome = basic_executor._execute_parallel(principles, simple_problem_graph)

    assert outcome.success
    assert len(outcome.sub_results) == 3


def test_execute_parallel_with_failures(basic_executor, simple_problem_graph):
    """Test parallel execution with some failures"""

    def failing_logic(inputs):
        raise ValueError("Test failure")

    principles = [
        Principle(
            id="p1",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True},
        ),
        Principle(
            id="p2", core_pattern={}, confidence=0.8, execution_logic=failing_logic
        ),
        Principle(
            id="p3",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True},
        ),
    ]

    outcome = basic_executor._execute_parallel(principles, simple_problem_graph)

    assert not outcome.success
    assert len(outcome.sub_results) == 3  # All attempted


# Test Iterative Execution


def test_execute_iterative_converges(basic_executor, simple_problem_graph):
    """Test iterative execution converges"""
    call_count = {"count": 0}

    def converging_logic(inputs):
        call_count["count"] += 1
        return {"success": True, "converged": call_count["count"] >= 2}

    principles = [
        Principle(
            id="iter_p",
            core_pattern={},
            confidence=0.8,
            execution_logic=converging_logic,
        )
    ]

    outcome = basic_executor._execute_iterative(principles, simple_problem_graph)

    assert outcome.success
    assert "iterations" in outcome.metrics


def test_execute_iterative_max_iterations(basic_executor, simple_problem_graph):
    """Test iterative execution respects max iterations"""
    principles = list(
        Principle(
            id="iter_p",
            core_pattern={},
            confidence=0.8,
            execution_logic=lambda inputs: {"success": True},
        )
    ]

    outcome = basic_executor._execute_iterative(principles, simple_problem_graph)

    assert "iterations" in outcome.metrics
    assert outcome.metrics["iterations"] <= 10


# Test Structure Solving


def test_solve_hierarchical(basic_executor, hierarchical_problem_graph):
    """Test hierarchical structure solving"""
    nodes = list(hierarchical_problem_graph.nodes.keys())
    inputs = {"data": [1, 2, 3]}

    result = basic_executor._solve_hierarchical(
        nodes, inputs, hierarchical_problem_graph
    )

    assert result["structure"] == "hierarchical"
    assert "results" in result
    assert "root_nodes" in result
    assert result["processed_count"] > 0


def test_solve_modular(basic_executor, simple_problem_graph):
    """Test modular structure solving"""
    nodes = ["node1", "node2", "node3"]
    inputs = {"data": [1, 2, 3]}

    result = basic_executor._solve_modular(nodes, inputs, simple_problem_graph)

    assert result["structure"] == "modular"
    assert "modules" in result
    assert result["module_count"] > 0


def test_solve_pipeline(basic_executor, simple_problem_graph):
    """Test pipeline structure solving"""
    nodes = ["node1", "node2", "node3"]
    inputs = {"data": [1, 2, 3]}

    result = basic_executor._solve_pipeline(nodes, inputs, simple_problem_graph)

    assert result["structure"] == "pipeline"
    assert "stages" in result
    assert "final_output" in result


def test_solve_recursive_with_protection(basic_executor, simple_problem_graph):
    """Test recursive solving with stack protection"""
    nodes = ["node1"]
    inputs = {"size": 10}

    result = basic_executor._solve_recursive(nodes, inputs, simple_problem_graph)

    assert result["structure"] == "recursive"
    assert "result" in result


def test_solve_recursive_max_depth(basic_executor, simple_problem_graph):
    """Test recursive solving respects max depth"""
    nodes = ["node1"]
    inputs = {"size": 1000}  # Would recurse deeply

    # Should not raise due to depth limits
    result = basic_executor._solve_recursive(nodes, inputs, simple_problem_graph)

    assert "result" in result


def test_solve_parallel_structure(basic_executor, simple_problem_graph):
    """Test parallel structure solving"""
    nodes = ["node1", "node2"]
    inputs = {"data": [1, 2, 3]}

    result = basic_executor._solve_parallel(nodes, inputs, simple_problem_graph)

    assert result["structure"] == "parallel"
    assert "parallel_results" in result
    assert "aggregated" in result


# Test Node Processing


def test_process_operation_node(basic_executor):
    """Test processing operation node"""
    node_data = {
        "type": "operation",
        "operation": "sum",
        "parameters": {"input": "data"},
    }
    inputs = {"data": [1, 2, 3, 4, 5]}

    result = basic_executor._execute_operation(node_data, inputs)

    assert result["operation"] == "sum"
    assert result["result"] == 15


def test_process_decision_node(basic_executor):
    """Test processing decision node"""
    node_data = {
        "type": "decision",
        "condition": {"variable": "x", "operator": ">", "threshold": 5},
    }
    inputs = {"x": 10}

    result = basic_executor._execute_decision(node_data, inputs)

    assert result["decision"] == True


def test_process_transform_node(basic_executor):
    """Test processing transform node"""
    node_data = {"type": "transform", "transform": "normalize", "parameters": {}}
    inputs = {"data": [1, 2, 3, 4, 5]}

    result = basic_executor._execute_transform(node_data, inputs)

    assert result["transform"] == "normalize"
    assert len(result["result"]) == 5


# Test Safety Validation


def test_validate_plan_safety_valid(basic_executor, simple_plan, simple_problem_graph):
    """Test validation of safe plan"""
    validation = basic_executor._validate_plan_safety(simple_plan, simple_problem_graph)

    assert validation["safe"] == True


def test_validate_plan_safety_invalid_confidence(basic_executor, simple_problem_graph):
    """Test validation catches invalid confidence"""
    bad_plan = DecompositionPlan(
        steps=[],
        confidence=1.5,  # Invalid
    )

    validation = basic_executor._validate_plan_safety(bad_plan, simple_problem_graph)

    assert validation["safe"] == False
    assert "confidence" in validation["reason"].lower()


def test_validate_principles_safety_valid(basic_executor):
    """Test validation of safe principles"""
    principles = [
        Principle(id="p1", core_pattern={}, confidence=0.8, execution_logic=lambda x: x)
    ]

    validation = basic_executor._validate_principles_safety(principles)

    assert validation["safe"] == True


def test_validate_principles_safety_no_logic(basic_executor):
    """Test validation catches missing execution logic"""
    principles = [
        Principle(id="p1", core_pattern={}, confidence=0.8, execution_logic=None)
    ]

    validation = basic_executor._validate_principles_safety(principles)

    assert validation["safe"] == False
    assert "no execution logic" in validation["reason"].lower()


def test_validate_outcome_safety_valid(basic_executor):
    """Test validation of safe outcome"""
    outcome = ExecutionOutcome(
        success=True, execution_time=1.5, sub_results=[], metrics={"accuracy": 0.95}
    )

    validation = basic_executor._validate_outcome_safety(outcome)

    assert validation["safe"] == True


def test_validate_outcome_safety_excessive_time(basic_executor):
    """Test validation catches excessive execution time"""
    outcome = ExecutionOutcome(
        success=True,
        execution_time=4000.0,  # Over 1 hour
        sub_results=[],
        metrics={},
    )

    validation = basic_executor._validate_outcome_safety(outcome)

    assert validation["safe"] == False


def test_apply_outcome_corrections(basic_executor):
    """Test safety corrections are applied"""
    outcome = ExecutionOutcome(
        success=True,
        execution_time=5000.0,
        sub_results=[],
        metrics={"bad_metric": float("inf")},
    )

    validation = {"safe": False, "reason": "excessive time and invalid metrics"}

    corrected = basic_executor._apply_outcome_corrections(outcome, validation)

    assert corrected.execution_time <= 3600
    assert np.isfinite(corrected.metrics["bad_metric"])
    assert corrected.metrics["safety_corrected"] == True


# Test Execute and Validate


def test_execute_and_validate_success(
    basic_executor, simple_problem_graph, simple_plan
):
    """Test execute and validate with successful outcome"""
    basic_executor.validator = Mock()

    outcome, validation = basic_executor.execute_and_validate(
        simple_problem_graph, simple_plan
    )

    assert isinstance(outcome, ExecutionOutcome)
    assert isinstance(validation, dict)


def test_execute_and_validate_no_validator(
    basic_executor, simple_problem_graph, simple_plan
):
    """Test execute and validate without validator"""
    basic_executor.validator = None

    outcome, validation = basic_executor.execute_and_validate(
        simple_problem_graph, simple_plan
    )

    assert validation["validated"] == False
    assert validation["reason"] == "no_validator"


# Test Statistics


def test_get_statistics(basic_executor, simple_problem_graph, simple_plan):
    """Test statistics collection"""
    # Execute some plans
    basic_executor.execute_plan(simple_problem_graph, simple_plan)
    basic_executor.execute_plan(simple_problem_graph, simple_plan)

    stats = basic_executor.get_statistics()

    assert "total_executions" in stats
    assert "successful_executions" in stats
    assert "success_rate" in stats
    assert "safety" in stats
    assert stats["safety"]["enabled"] == True


def test_statistics_without_safety():
    """Test statistics when safety is disabled"""
    with patch(
        "vulcan.problem_decomposer.problem_executor.SAFETY_VALIDATOR_AVAILABLE", False
    ):
        executor = ProblemExecutor()
        stats = executor.get_statistics()

        assert stats["safety"]["enabled"] == False


# Test Cache Management


def test_cache_size_limit(basic_executor):
    """Test that cache respects size limit"""
    basic_executor.cache_size = 5

    # Create many different problems
    for i in range(10):
        graph = ProblemGraph(nodes={f"node_{i}": {}}, edges=[], metadata={"id": i})
        plan = DecompositionPlan(steps=[], confidence=0.5)

        basic_executor.execute_plan(graph, plan)

    assert len(basic_executor.solution_cache) <= basic_executor.cache_size


# Test Thread Safety


def test_thread_safety_cache_access(basic_executor, simple_problem_graph, simple_plan):
    """Test thread-safe cache access"""
    import threading

    results = []

    def execute():
        outcome = basic_executor.execute_plan(simple_problem_graph, simple_plan)
        results.append(outcome)

    threads = [threading.Thread(target=execute) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 5


# Test Recursion Protection


def test_recursion_depth_protection(basic_executor):
    """Test that recursion depth is protected"""
    basic_executor.max_recursion_depth = 10

    # Create deeply recursive problem
    graph = ProblemGraph(nodes={"n1": {}}, edges=[])
    inputs = {"size": 100}

    # This should not cause stack overflow
    result = basic_executor._solve_recursive(["n1"], inputs, graph)

    assert "result" in result


# Test Edge Cases


def test_empty_plan(basic_executor, simple_problem_graph):
    """Test execution with empty plan"""
    empty_plan = DecompositionPlan(steps=[], confidence=0.5)

    outcome = basic_executor.execute_plan(simple_problem_graph, empty_plan)

    assert not outcome.success
    assert len(outcome.errors) > 0


def test_malformed_step(basic_executor, simple_problem_graph):
    """Test handling of malformed step"""
    bad_plan = DecompositionPlan(
        steps=[
            {"type": "structural_match"}  # Missing required fields
        ],
        confidence=0.5,
    )

    outcome = basic_executor.execute_plan(simple_problem_graph, bad_plan)

    # Should handle gracefully
    assert isinstance(outcome, ExecutionOutcome)


def test_empty_problem_graph(basic_executor):
    """Test execution with empty problem graph"""
    empty_graph = ProblemGraph(nodes={}, edges=[])
    plan = DecompositionPlan(steps=list(], confidence=0.5)

    outcome = basic_executor.execute_plan(empty_graph, plan)

    assert isinstance(outcome, ExecutionOutcome)


# Integration Tests


def test_full_execution_pipeline(basic_executor, hierarchical_problem_graph):
    """Test complete execution pipeline"""
    plan = DecompositionPlan(
        steps=list(
            {
                "type": "structural_match",
                "structure": "hierarchical",
                "nodes": list(hierarchical_problem_graph.nodes.keys()),
                "confidence": 0.8,
            },
            {
                "type": "exact_match",
                "pattern_id": "tree",
                "nodes": ["root"],
                "confidence": 0.9,
            },
        ],
        strategy=None,
        confidence=0.85,
    )

    outcome = basic_executor.execute_plan(hierarchical_problem_graph, plan)

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.metrics["safety_validated"] == True
    assert "plan_confidence" in outcome.metrics


def test_execution_with_all_solver_types(basic_executor, simple_problem_graph):
    """Test execution using all solver types"""
    comprehensive_plan = DecompositionPlan(
        steps=[
            {
                "type": "structural_match",
                "structure": "hierarchical",
                "nodes": ["node1"],
                "confidence": 0.8,
            },
            {
                "type": "semantic_match",
                "concept": "test",
                "similarity": 0.7,
                "nodes": ["node2"],
                "confidence": 0.7,
            },
            {
                "type": "exact_match",
                "pattern_id": "linear",
                "nodes": ["node3"],
                "confidence": 0.9,
            },
            {"type": "synthetic_bridge", "template": "simple", "confidence": 0.6},
            {
                "type": "analogical",
                "source_domain": "sorting",
                "target_mapping": {},
                "confidence": 0.7,
            },
            {"type": "brute_force", "part": 1, "content": {}, "confidence": 0.5},
        ],
        strategy=None,
        confidence=0.7,
    )

    outcome = basic_executor.execute_plan(simple_problem_graph, comprehensive_plan)

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.metrics["num_steps"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
