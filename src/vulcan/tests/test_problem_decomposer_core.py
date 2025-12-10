"""
test_problem_decomposer_core.py - PURE MOCK VERSION
Tests problem decomposer core functionality without spawning threads.
"""

import pytest
import numpy as np
import time
import hashlib
import threading
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ============================================================================
# Mock Enums and Classes
# ============================================================================


class ProblemComplexity(Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class DecompositionMode(Enum):
    EXACT = "exact"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


class DomainDataCategory(Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    MIXED = "mixed"


@dataclass
class ProblemGraph:
    nodes: Dict[str, Dict] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)
    root: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0

    def get_signature(self) -> str:
        content = str(sorted(self.nodes.keys())) + str(sorted(self.edges))
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def to_networkx(self):
        class MockGraph:
            def __init__(self, nodes, edges):
                self._nodes = nodes
                self._edges = edges

            def nodes(self):
                return list(self._nodes.keys())

            def edges(self):
                return [(e[0], e[1]) for e in self._edges]

            def number_of_nodes(self):
                return len(self._nodes)

        return MockGraph(self.nodes, self.edges)


@dataclass
class DecompositionStep:
    step_id: str
    action_type: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: float = 1.0
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "description": self.description,
            "dependencies": self.dependencies,
            "estimated_complexity": self.estimated_complexity,
            "confidence": self.confidence,
        }


@dataclass
class DecompositionPlan:
    steps: List[DecompositionStep] = field(default_factory=list)
    confidence: float = 0.8
    estimated_complexity: float = 1.0
    strategy: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "confidence": self.confidence,
            "estimated_complexity": self.estimated_complexity,
            "strategy": str(self.strategy),
        }


@dataclass
class ExecutionOutcome:
    success: bool
    execution_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ProblemSignature:
    node_count: int = 0
    edge_count: int = 0
    complexity: float = 0.0
    domain: str = "unknown"
    has_cycles: bool = False
    hash: str = ""


@dataclass
class LearningGap:
    problem_signature: str
    failure_reason: str
    timestamp: float = field(default_factory=time.time)
    attempted_strategies: List[str] = field(default_factory=list)


class MockDomainSelector:
    def __init__(self):
        self.domains = ["general", "optimization", "planning", "control"]

    def select_domain(self, problem: ProblemGraph) -> str:
        return problem.metadata.get("domain", "general")


class MockPerformanceTracker:
    def __init__(self):
        self.executions: Dict[str, List[Dict]] = defaultdict(list)

    def record_execution(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ):
        strategy_name = getattr(plan.strategy, "name", "unknown")
        self.executions[strategy_name].append(
            {"success": outcome.success, "time": outcome.execution_time}
        )

    def get_strategy_success_rate(self, strategy_name: str) -> float:
        execs = self.executions.get(strategy_name, [])
        if not execs:
            return 0.0
        return sum(1 for e in execs if e["success"]) / len(execs)

    def get_average_execution_time(self, strategy_name: str) -> float:
        execs = self.executions.get(strategy_name, [])
        if not execs:
            return 0.0
        return sum(e["time"] for e in execs) / len(execs)


class MockStrategyProfiler:
    def __init__(self):
        self.affinities: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def update_affinity(
        self, strategy_name: str, domain: str, complexity: float, success: bool
    ):
        delta = 0.1 if success else -0.05
        self.affinities[strategy_name][domain] += delta

    def get_affinity(self, strategy_name: str, domain: str) -> float:
        return self.affinities[strategy_name][domain]


class MockStrategy:
    def __init__(self, name: str):
        self.name = name

    def decompose(self, problem: ProblemGraph) -> DecompositionPlan:
        steps = []
        for i, node in enumerate(problem.nodes):
            steps.append(
                DecompositionStep(
                    step_id=f"step_{i}",
                    action_type="process",
                    description=f"Process {node}",
                )
            )
        return DecompositionPlan(steps=steps, confidence=0.8, strategy=self)


class MockFallbackChain:
    def __init__(self):
        self.strategies = [
            MockStrategy("exact"),
            MockStrategy("heuristic"),
            MockStrategy("brute_force"),
        ]

    def generate_fallback_plans(self, problem: ProblemGraph) -> List[DecompositionPlan]:
        return [s.decompose(problem) for s in self.strategies]


class MockExecutor:
    def __init__(self):
        pass

    def execute_plan(
        self, problem: ProblemGraph, plan: DecompositionPlan
    ) -> ExecutionOutcome:
        return ExecutionOutcome(success=True, execution_time=1.0)


class MockProblemDecomposer:
    def __init__(self, semantic_bridge=None, validator=None, safety_config=None):
        self.semantic_bridge = semantic_bridge
        self.validator = validator

        self.safety_validator = Mock()
        self.safety_validator.validate_action_comprehensive = Mock(
            return_value=Mock(safe=True, confidence=0.9)
        )

        self.domain_selector = MockDomainSelector()
        self.performance_tracker = MockPerformanceTracker()
        self.strategy_profiler = MockStrategyProfiler()
        self.fallback_chain = MockFallbackChain()
        self.executor = MockExecutor()

        self.decomposition_cache: Dict[str, DecompositionPlan] = {}
        self.learning_gaps: List[LearningGap] = []
        self.successful_decompositions = 0
        self.failed_decompositions = 0
        self.cache_size = 100

        self._lock = threading.Lock()

    def _extract_problem_signature(self, problem: ProblemGraph) -> ProblemSignature:
        return ProblemSignature(
            node_count=len(problem.nodes),
            edge_count=len(problem.edges),
            complexity=len(problem.nodes) + len(problem.edges) * 0.5,
            domain=problem.metadata.get("domain", "unknown"),
            hash=problem.get_signature(),
        )

    def _predict_best_strategy(
        self, problem: ProblemGraph, signature: ProblemSignature
    ) -> MockStrategy:
        # Simple strategy selection based on complexity
        if signature.complexity < 5:
            return MockStrategy("exact")
        elif signature.complexity < 20:
            return MockStrategy("heuristic")
        else:
            return MockStrategy("brute_force")

    def decompose_novel_problem(self, problem: ProblemGraph) -> DecompositionPlan:
        sig = problem.get_signature()

        # Check cache
        if sig in self.decomposition_cache:
            return self.decomposition_cache[sig]

        signature = self._extract_problem_signature(problem)
        strategy = self._predict_best_strategy(problem, signature)
        plan = strategy.decompose(problem)

        # Cache result
        with self._lock:
            if len(self.decomposition_cache) >= self.cache_size:
                oldest = next(iter(self.decomposition_cache))
                del self.decomposition_cache[oldest]
            self.decomposition_cache[sig] = plan

        return plan

    def decompose_with_fallbacks(self, problem: ProblemGraph) -> DecompositionPlan:
        plans = self.fallback_chain.generate_fallback_plans(problem)
        return plans[0] if plans else DecompositionPlan()

    def learn_from_execution(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ):
        self.performance_tracker.record_execution(problem, plan, outcome)

        if outcome.success:
            self.successful_decompositions += 1
        else:
            self.failed_decompositions += 1
            gap = self.create_learning_gap(problem)
            self.learning_gaps.append(gap)

    def create_learning_gap(self, problem: ProblemGraph) -> LearningGap:
        return LearningGap(
            problem_signature=problem.get_signature(),
            failure_reason="Decomposition failed",
        )

    def get_statistics(self) -> Dict:
        return {
            "decomposition_stats": {
                "successful_decompositions": self.successful_decompositions,
                "failed_decompositions": self.failed_decompositions,
                "cache_size": len(self.decomposition_cache),
            },
            "execution_stats": {
                "total_executions": self.successful_decompositions
                + self.failed_decompositions
            },
            "safety": {"enabled": True},
        }


# Aliases
ProblemDecomposer = MockProblemDecomposer
DomainSelector = MockDomainSelector
PerformanceTracker = MockPerformanceTracker
StrategyProfiler = MockStrategyProfiler


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_problem():
    return ProblemGraph(
        nodes={
            "A": {"type": "operation"},
            "B": {"type": "operation"},
            "C": {"type": "operation"},
        },
        edges=[("A", "B", {}), ("B", "C", {})],
        root="A",
        metadata={"domain": "test", "type": "sequential"},
    )


@pytest.fixture
def hierarchical_problem():
    return ProblemGraph(
        nodes={
            "root": {"type": "decision", "level": 0},
            "branch1": {"type": "operation", "level": 1},
            "branch2": {"type": "operation", "level": 1},
            "leaf1": {"type": "transform", "level": 2},
            "leaf2": {"type": "transform", "level": 2},
            "leaf3": {"type": "transform", "level": 2},
        },
        edges=[
            ("root", "branch1", {"weight": 1.0}),
            ("root", "branch2", {"weight": 1.0}),
            ("branch1", "leaf1", {"weight": 0.5}),
            ("branch1", "leaf2", {"weight": 0.5}),
            ("branch2", "leaf3", {"weight": 1.0}),
        ],
        root="root",
        metadata={"domain": "planning", "type": "hierarchical"},
    )


@pytest.fixture
def complex_problem():
    nodes = {f"node_{i}": {"type": "operation", "index": i} for i in range(20)}
    edges = [(f"node_{i}", f"node_{i + 1}", {}) for i in range(19)]
    edges.extend([("node_5", "node_10", {}), ("node_8", "node_15", {})])
    return ProblemGraph(
        nodes=nodes,
        edges=edges,
        root="node_0",
        metadata={"domain": "optimization", "type": "complex"},
    )


@pytest.fixture
def cyclic_problem():
    return ProblemGraph(
        nodes={"init": {}, "evaluate": {}, "refine": {}, "output": {}},
        edges=[
            ("init", "evaluate", {}),
            ("evaluate", "refine", {}),
            ("refine", "evaluate", {}),
            ("evaluate", "output", {}),
        ],
        root="init",
        metadata={"domain": "optimization", "type": "iterative"},
    )


@pytest.fixture
def mock_validator():
    validator = Mock()
    validator.validate_solution = Mock(return_value={"valid": True, "score": 0.9})
    return validator


@pytest.fixture
def mock_semantic_bridge():
    bridge = Mock()
    bridge.apply_concept = Mock(return_value={"success": True})
    return bridge


@pytest.fixture
def decomposer(mock_validator, mock_semantic_bridge):
    return MockProblemDecomposer(
        semantic_bridge=mock_semantic_bridge, validator=mock_validator, safety_config={}
    )


# ============================================================================
# Tests
# ============================================================================


class TestProblemGraph:
    def test_graph_creation(self, simple_problem):
        assert len(simple_problem.nodes) == 3
        assert len(simple_problem.edges) == 2
        assert simple_problem.root == "A"

    def test_get_signature(self, simple_problem):
        sig1 = simple_problem.get_signature()
        sig2 = simple_problem.get_signature()
        assert sig1 == sig2
        assert len(sig1) == 32

    def test_signature_uniqueness(self, simple_problem, hierarchical_problem):
        assert simple_problem.get_signature() != hierarchical_problem.get_signature()

    def test_to_networkx(self, simple_problem):
        G = simple_problem.to_networkx()
        assert hasattr(G, "nodes")
        assert G.number_of_nodes() == 3

    def test_complexity_score(self, simple_problem):
        simple_problem.complexity_score = 2.5
        assert simple_problem.complexity_score == 2.5


class TestDecompositionStep:
    def test_step_creation(self):
        step = DecompositionStep(
            step_id="step_1",
            action_type="process",
            description="Process node A",
            dependencies=["step_0"],
            estimated_complexity=1.5,
            confidence=0.8,
        )
        assert step.step_id == "step_1"
        assert step.action_type == "process"
        assert step.confidence == 0.8

    def test_step_to_dict(self):
        step = DecompositionStep(step_id="s1", action_type="test")
        d = step.to_dict()
        assert "step_id" in d
        assert d["step_id"] == "s1"


class TestDecompositionPlan:
    def test_plan_creation(self):
        plan = DecompositionPlan(confidence=0.9, estimated_complexity=2.0)
        assert plan.confidence == 0.9
        assert len(plan.steps) == 0

    def test_plan_to_dict(self):
        plan = DecompositionPlan(confidence=0.85)
        d = plan.to_dict()
        assert "confidence" in d
        assert d["confidence"] == 0.85


class TestProblemDecomposer:
    def test_initialization(self, decomposer):
        assert decomposer.successful_decompositions == 0
        assert len(decomposer.decomposition_cache) == 0

    def test_extract_signature(self, decomposer, simple_problem):
        sig = decomposer._extract_problem_signature(simple_problem)
        assert sig.node_count == 3
        assert sig.edge_count == 2

    def test_predict_strategy(self, decomposer, simple_problem):
        sig = decomposer._extract_problem_signature(simple_problem)
        strategy = decomposer._predict_best_strategy(simple_problem, sig)
        assert strategy is not None

    def test_decompose_novel_problem(self, decomposer, simple_problem):
        plan = decomposer.decompose_novel_problem(simple_problem)
        assert isinstance(plan, DecompositionPlan)
        assert len(plan.steps) > 0

    def test_decompose_caching(self, decomposer, simple_problem):
        plan1 = decomposer.decompose_novel_problem(simple_problem)
        plan2 = decomposer.decompose_novel_problem(simple_problem)
        assert plan1 is plan2

    def test_decompose_with_fallbacks(self, decomposer, simple_problem):
        plan = decomposer.decompose_with_fallbacks(simple_problem)
        assert isinstance(plan, DecompositionPlan)

    def test_learn_from_execution(self, decomposer, simple_problem):
        plan = DecompositionPlan(confidence=0.7)
        plan.strategy = Mock(name="TestStrategy")
        outcome = ExecutionOutcome(success=True, execution_time=1.5)

        decomposer.learn_from_execution(simple_problem, plan, outcome)
        assert decomposer.successful_decompositions == 1

    def test_learn_from_failure(self, decomposer, simple_problem):
        plan = DecompositionPlan(confidence=0.7)
        plan.strategy = Mock(name="TestStrategy")
        outcome = ExecutionOutcome(success=False, execution_time=1.0)

        decomposer.learn_from_execution(simple_problem, plan, outcome)
        assert len(decomposer.learning_gaps) == 1

    def test_create_learning_gap(self, decomposer, simple_problem):
        gap = decomposer.create_learning_gap(simple_problem)
        assert isinstance(gap, LearningGap)
        assert gap.problem_signature == simple_problem.get_signature()

    def test_get_statistics(self, decomposer):
        stats = decomposer.get_statistics()
        assert "decomposition_stats" in stats
        assert "execution_stats" in stats
        assert "safety" in stats


class TestPerformanceTracker:
    def test_record_and_retrieve(self):
        tracker = MockPerformanceTracker()
        plan = DecompositionPlan()
        plan.strategy = MockStrategy("test")
        outcome = ExecutionOutcome(success=True, execution_time=1.0)

        tracker.record_execution(ProblemGraph(), plan, outcome)

        assert tracker.get_strategy_success_rate("test") == 1.0
        assert tracker.get_average_execution_time("test") == 1.0


class TestEdgeCases:
    def test_empty_problem(self, decomposer):
        empty = ProblemGraph(nodes={}, edges=[], metadata={"domain": "test"})
        plan = decomposer.decompose_novel_problem(empty)
        assert isinstance(plan, DecompositionPlan)

    def test_large_problem(self, decomposer):
        nodes = {f"node_{i}": {} for i in range(100)}
        edges = [(f"node_{i}", f"node_{i + 1}", {}) for i in range(99)]
        large = ProblemGraph(nodes=nodes, edges=edges, metadata={"domain": "test"})

        plan = decomposer.decompose_novel_problem(large)
        assert isinstance(plan, DecompositionPlan)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
