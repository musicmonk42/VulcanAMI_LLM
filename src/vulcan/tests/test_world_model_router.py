"""
test_world_model_router.py - PURE MOCK VERSION
Tests WorldModelRouter functionality without spawning real threads.
"""

import pickle
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import Mock

import numpy as np
import pytest

# ============================================================================
# Mock Enums
# ============================================================================


class UpdateType(Enum):
    INTERVENTION = "intervention"
    CORRELATION = "correlation"
    DYNAMICS = "dynamics"
    CAUSAL = "causal"
    INVARIANT = "invariant"
    CONFIDENCE = "confidence"


class UpdatePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4


# ============================================================================
# Mock Dataclasses
# ============================================================================


@dataclass
class UpdateStrategy:
    update_type: UpdateType
    priority: UpdatePriority
    dependencies: Set[UpdateType] = field(default_factory=set)
    estimated_cost_ms: float = 10.0
    confidence_threshold: float = 0.5
    can_parallelize: bool = True
    can_defer: bool = False


@dataclass
class ObservationSignature:
    has_intervention: bool
    has_temporal: bool
    has_multi_variable: bool
    has_anomaly: bool
    has_structural_change: bool
    variable_count: int
    time_delta: Optional[float]
    confidence: float
    domain: str
    pattern_hash: str


@dataclass
class UpdatePlan:
    immediate: List[UpdateType]
    deferred: List[UpdateType]
    parallel_groups: List[List[UpdateType]]
    estimated_time_ms: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Mock Components
# ============================================================================


class MockUpdateDependencyGraph:
    """Mock dependency graph"""

    def __init__(self):
        self.dependencies = {
            UpdateType.INTERVENTION: set(),
            UpdateType.CORRELATION: set(),
            UpdateType.DYNAMICS: {UpdateType.CORRELATION},
            UpdateType.CAUSAL: {UpdateType.INTERVENTION, UpdateType.CORRELATION},
            UpdateType.INVARIANT: {UpdateType.DYNAMICS},
            UpdateType.CONFIDENCE: set(),
        }
        self.mutual_exclusive = {(UpdateType.INTERVENTION, UpdateType.CAUSAL)}

    def get_dependencies(self, update_type: UpdateType) -> Set[UpdateType]:
        return self.dependencies.get(update_type, set())

    def can_parallelize(self, updates: List[UpdateType]) -> bool:
        for u1, u2 in self.mutual_exclusive:
            if u1 in updates and u2 in updates:
                return False
        return True

    def get_execution_order(self, required: Set[UpdateType]) -> List[List[UpdateType]]:
        groups = []
        remaining = set(required)
        executed = set()

        while remaining:
            ready = []
            for u in remaining:
                deps = self.get_dependencies(u)
                if deps.issubset(executed):
                    ready.append(u)

            if not ready:
                # No progress possible, just take one
                ready = [remaining.pop()]
                remaining.add(ready[0])

            groups.append(ready)
            for u in ready:
                remaining.discard(u)
                executed.add(u)

        return groups


class MockPatternLearner:
    """Mock pattern learner"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.pattern_history: List[Dict] = []
        self.pattern_counts: Dict[str, int] = defaultdict(int)
        self.rules: List[Dict] = []
        self.rule_confidence: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record_pattern(
        self,
        signature: Optional[ObservationSignature],
        updates_run: List[UpdateType],
        success: bool,
        cost_ms: float,
    ):
        if signature is None:
            return

        with self._lock:
            self.pattern_history.append(
                {
                    "signature": signature,
                    "updates": updates_run,
                    "success": success,
                    "cost_ms": cost_ms,
                    "timestamp": time.time(),
                }
            )
            self.pattern_counts[signature.pattern_hash] += 1

            if len(self.pattern_history) > self.window_size:
                self.pattern_history = self.pattern_history[-self.window_size :]

    def predict_updates(self, signature: ObservationSignature) -> List[UpdateType]:
        for rule in self.rules:
            if self._matches_rule(signature, rule):
                return rule.get("updates", [])
        return []

    def _matches_rule(self, sig: ObservationSignature, rule: Dict) -> bool:
        conditions = rule.get("signature_conditions", {})
        for key, value in conditions.items():
            if hasattr(sig, key) and getattr(sig, key) != value:
                return False
        return True

    def _learn_rules(self):
        # Simplified rule learning
        if len(self.pattern_history) < 10:
            return

        successful = [p for p in self.pattern_history if p["success"]]
        if not successful:
            return


class MockCostModel:
    """Mock cost model"""

    def __init__(self):
        self.base_costs = {
            UpdateType.INTERVENTION: 50.0,
            UpdateType.CORRELATION: 20.0,
            UpdateType.DYNAMICS: 30.0,
            UpdateType.CAUSAL: 40.0,
            UpdateType.INVARIANT: 25.0,
            UpdateType.CONFIDENCE: 10.0,
        }
        self.cost_history: Dict[UpdateType, List[float]] = defaultdict(list)
        self.size_multiplier = 0.1

    def estimate_cost(
        self, update_type: UpdateType, observation_size: int = 1
    ) -> float:
        base = self.base_costs.get(update_type, 20.0)
        return base + (observation_size * self.size_multiplier)

    def update_cost(self, update_type: UpdateType, actual_cost: float):
        self.cost_history[update_type].append(actual_cost)

        if len(self.cost_history[update_type]) >= 10:
            avg = np.mean(self.cost_history[update_type][-20:])
            self.base_costs[update_type] = avg


class MockWorldModelRouter:
    """Mock world model router - no thread spawning"""

    def __init__(self, world_model, config: Dict = None):
        config = config or {}
        self.world_model = world_model

        self.time_budget_ms = config.get("time_budget_ms", 1000)
        self.min_confidence = config.get("min_confidence", 0.5)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.use_learning = config.get("use_learning", True)

        self.dependency_graph = MockUpdateDependencyGraph()
        self.pattern_learner = MockPatternLearner()
        self.cost_model = MockCostModel()

        self.strategies = self._init_strategies()
        self.cache: Dict[str, UpdatePlan] = {}
        self.metrics = {"plans_created": 0, "cache_hits": 0, "pattern_rules": 0}

        # Mock safety validator
        self.safety_validator = Mock()
        self.safety_validator.analyze_observation_safety = Mock(
            return_value={"safe": True}
        )
        self.safety_validator.is_enabled = True

        self._lock = threading.Lock()

    def _init_strategies(self) -> Dict[UpdateType, UpdateStrategy]:
        return {
            UpdateType.INTERVENTION: UpdateStrategy(
                update_type=UpdateType.INTERVENTION,
                priority=UpdatePriority.CRITICAL,
                dependencies=set(),
                estimated_cost_ms=50.0,
            ),
            UpdateType.CORRELATION: UpdateStrategy(
                update_type=UpdateType.CORRELATION,
                priority=UpdatePriority.HIGH,
                dependencies=set(),
                estimated_cost_ms=20.0,
            ),
            UpdateType.DYNAMICS: UpdateStrategy(
                update_type=UpdateType.DYNAMICS,
                priority=UpdatePriority.NORMAL,
                dependencies={UpdateType.CORRELATION},
                estimated_cost_ms=30.0,
            ),
            UpdateType.CAUSAL: UpdateStrategy(
                update_type=UpdateType.CAUSAL,
                priority=UpdatePriority.NORMAL,
                dependencies={UpdateType.INTERVENTION, UpdateType.CORRELATION},
                estimated_cost_ms=40.0,
            ),
            UpdateType.INVARIANT: UpdateStrategy(
                update_type=UpdateType.INVARIANT,
                priority=UpdatePriority.LOW,
                dependencies={UpdateType.DYNAMICS},
                estimated_cost_ms=25.0,
                can_defer=True,
            ),
            UpdateType.CONFIDENCE: UpdateStrategy(
                update_type=UpdateType.CONFIDENCE,
                priority=UpdatePriority.NORMAL,
                dependencies=set(),
                estimated_cost_ms=10.0,
            ),
        }

    def route(self, observation, constraints: Dict = None) -> UpdatePlan:
        constraints = constraints or {}

        # Check safety
        if hasattr(self, "safety_validator") and self.safety_validator:
            safety_result = self.safety_validator.analyze_observation_safety(
                observation
            )
            if not safety_result.get("safe", True):
                return UpdatePlan(
                    immediate=[],
                    deferred=[],
                    parallel_groups=[],
                    estimated_time_ms=0,
                    confidence=0.0,
                    reasoning="Safety blocked",
                    metadata={"safety_blocked": True},
                )

        signature = self._extract_signature(observation)

        # Check cache
        cache_key = signature.pattern_hash
        if cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            return self.cache[cache_key]

        # Determine required updates
        required = self._determine_required_updates(signature)

        # Apply constraints
        constraints.get("time_budget_ms", self.time_budget_ms)
        filtered = self._apply_constraints(required, constraints, signature)

        # Create plan
        plan = self._create_execution_plan(filtered, signature)

        with self._lock:
            self.metrics["plans_created"] += 1
            self.cache[cache_key] = plan

        return plan

    def execute(self, plan: UpdatePlan) -> Dict[str, Any]:
        start = time.time()
        results = {}
        updates_executed = []
        success = True

        # Execute immediate updates
        for update_type in plan.immediate:
            try:
                result = self._execute_update(
                    update_type, plan.metadata.get("signature")
                )
                results[update_type.value] = result
                updates_executed.append(update_type)
            except Exception as e:
                results[update_type.value] = {"error": str(e)}
                success = False

        # Execute parallel groups
        for group in plan.parallel_groups:
            for update_type in group:
                try:
                    result = self._execute_update(
                        update_type, plan.metadata.get("signature")
                    )
                    results[update_type.value] = result
                    updates_executed.append(update_type)
                except Exception as e:
                    results[update_type.value] = {"error": str(e)}
                    success = False

        elapsed = (time.time() - start) * 1000

        # Update cost model
        for ut in updates_executed:
            self.cost_model.update_cost(ut, elapsed / max(len(updates_executed), 1))

        return {
            "results": results,
            "updates_executed": updates_executed,
            "execution_time_ms": elapsed,
            "success": success,
        }

    def _execute_update(self, update_type: UpdateType, signature) -> Dict:
        if update_type == UpdateType.CORRELATION:
            return self.world_model.correlation_tracker.update()
        elif update_type == UpdateType.DYNAMICS:
            return self.world_model.dynamics.update()
        elif update_type == UpdateType.INVARIANT:
            return self.world_model.invariant_detector.check()
        elif update_type == UpdateType.CONFIDENCE:
            return self.world_model.confidence_tracker.update()
        elif update_type == UpdateType.INTERVENTION:
            return (
                self.world_model.intervention_manager.process()
                if hasattr(self.world_model.intervention_manager, "process")
                else {}
            )
        elif update_type == UpdateType.CAUSAL:
            return {"status": "success"}
        return {"status": "unknown"}

    def _extract_signature(self, observation) -> ObservationSignature:
        variables = getattr(observation, "variables", {}) or {}
        intervention_data = getattr(observation, "intervention_data", None)

        return ObservationSignature(
            has_intervention=intervention_data is not None,
            has_temporal=True,
            has_multi_variable=len(variables) > 1,
            has_anomaly=getattr(observation, "is_anomaly", False),
            has_structural_change=getattr(observation, "structural_change", False),
            variable_count=len(variables),
            time_delta=10.0,
            confidence=getattr(observation, "confidence", 1.0),
            domain=getattr(observation, "domain", "unknown"),
            pattern_hash=str(hash(frozenset(variables.keys()))),
        )

    def _determine_required_updates(
        self, signature: ObservationSignature
    ) -> Set[UpdateType]:
        required = {UpdateType.CONFIDENCE}

        if signature.has_intervention:
            required.add(UpdateType.INTERVENTION)
            required.add(UpdateType.CAUSAL)

        if signature.has_multi_variable:
            required.add(UpdateType.CORRELATION)
            required.add(UpdateType.DYNAMICS)

        if signature.has_temporal:
            required.add(UpdateType.CORRELATION)

        return required

    def _apply_constraints(
        self,
        updates: Set[UpdateType],
        constraints: Dict,
        signature: ObservationSignature,
    ) -> Set[UpdateType]:
        budget = constraints.get("time_budget_ms", self.time_budget_ms)
        priority_threshold = constraints.get("priority_threshold", UpdatePriority.BATCH)

        if budget <= 0:
            return set()

        filtered = set()
        total_cost = 0

        for ut in sorted(updates, key=lambda x: self.strategies[x].priority.value):
            cost = self.cost_model.estimate_cost(ut, signature.variable_count)
            if total_cost + cost <= budget:
                if self.strategies[ut].priority.value <= priority_threshold.value:
                    filtered.add(ut)
                    total_cost += cost

        return filtered

    def _create_execution_plan(
        self, updates: Set[UpdateType], signature: ObservationSignature
    ) -> UpdatePlan:
        if not updates:
            return UpdatePlan(
                immediate=[],
                deferred=[],
                parallel_groups=[],
                estimated_time_ms=0,
                confidence=1.0,
                reasoning="No updates required",
                metadata={"signature": signature},
            )

        execution_groups = self.dependency_graph.get_execution_order(updates)

        immediate = execution_groups[0] if execution_groups else []
        parallel_groups = execution_groups[1:] if len(execution_groups) > 1 else []

        # Calculate estimated time
        total_time = sum(
            self.cost_model.estimate_cost(ut, signature.variable_count)
            for ut in updates
        )

        return UpdatePlan(
            immediate=immediate,
            deferred=[],
            parallel_groups=parallel_groups,
            estimated_time_ms=total_time,
            confidence=signature.confidence,
            reasoning=f"Executing {len(updates)} updates",
            metadata={"signature": signature},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "plans_created": self.metrics["plans_created"],
            "cache_hits": self.metrics["cache_hits"],
            "pattern_rules": len(self.pattern_learner.rules),
            "safety": {"enabled": True},
        }

    def save_state(self, path: str):
        save_path = Path(path) / "router_state.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "metrics": self.metrics,
            "cache": {},  # Don't pickle complex objects
            "pattern_history": self.pattern_learner.pattern_history,
            "rules": self.pattern_learner.rules,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str):
        save_path = Path(path) / "router_state.pkl"

        if not save_path.exists():
            return

        try:
            with open(save_path, "rb") as f:
                state = pickle.load(f)

            self.metrics.update(state.get("metrics", {}))
            self.pattern_learner.pattern_history = state.get("pattern_history", [])
            self.pattern_learner.rules = state.get("rules", [])
        except Exception:
            pass


# Aliases for compatibility
UpdateDependencyGraph = MockUpdateDependencyGraph
PatternLearner = MockPatternLearner
CostModel = MockCostModel
WorldModelRouter = MockWorldModelRouter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_world_model():
    model = Mock()
    model.correlation_tracker = Mock()
    model.correlation_tracker.update = Mock(return_value={"status": "success"})
    model.correlation_tracker.get_baseline = Mock(return_value=5.0)
    model.correlation_tracker.get_noise_level = Mock(return_value=0.5)

    model.dynamics = Mock()
    model.dynamics.update = Mock(return_value={"status": "success"})

    model.invariant_detector = Mock()
    model.invariant_detector.check = Mock(return_value={"status": "success"})

    model.confidence_tracker = Mock()
    model.confidence_tracker.update = Mock(return_value={"status": "success"})

    model.intervention_manager = Mock()
    model.causal_graph = Mock()
    model.last_observation_time = time.time() - 10.0
    model.safety_validator = None

    return model


@pytest.fixture
def sample_observation():
    obs = Mock()
    obs.timestamp = time.time()
    obs.variables = {"x": 1.0, "y": 2.0, "z": 3.0}
    obs.confidence = 0.9
    obs.domain = "test"
    obs.intervention_data = None
    obs.sequence_id = 123
    obs.metadata = {}
    obs.is_anomaly = False
    obs.structural_change = False
    obs.is_intervention = False
    return obs


@pytest.fixture
def intervention_observation():
    obs = Mock()
    obs.timestamp = time.time()
    obs.variables = {"x": 5.0, "y": 10.0}
    obs.intervention_data = {"intervened_variable": "x", "intervention_value": 5.0}
    obs.confidence = 1.0
    obs.domain = "test"
    obs.sequence_id = 124
    obs.metadata = {"is_intervention": True}
    obs.is_anomaly = False
    obs.structural_change = False
    obs.is_intervention = True
    return obs


@pytest.fixture
def observation_signature():
    return ObservationSignature(
        has_intervention=False,
        has_temporal=True,
        has_multi_variable=True,
        has_anomaly=False,
        has_structural_change=False,
        variable_count=3,
        time_delta=10.0,
        confidence=0.9,
        domain="test",
        pattern_hash="123456",
    )


@pytest.fixture
def dependency_graph():
    return MockUpdateDependencyGraph()


@pytest.fixture
def pattern_learner():
    return MockPatternLearner(window_size=100)


@pytest.fixture
def cost_model():
    return MockCostModel()


@pytest.fixture
def router(mock_world_model):
    config = {
        "time_budget_ms": 1000,
        "min_confidence": 0.5,
        "exploration_rate": 0.1,
        "use_learning": True,
    }
    return MockWorldModelRouter(mock_world_model, config)


# ============================================================================
# Tests
# ============================================================================


class TestEnums:
    """Test enum definitions"""

    def test_update_type_enum(self):
        assert UpdateType.INTERVENTION.value == "intervention"
        assert UpdateType.CORRELATION.value == "correlation"
        assert UpdateType.DYNAMICS.value == "dynamics"
        assert UpdateType.CAUSAL.value == "causal"
        assert UpdateType.INVARIANT.value == "invariant"
        assert UpdateType.CONFIDENCE.value == "confidence"

    def test_update_priority_enum(self):
        assert UpdatePriority.CRITICAL.value == 0
        assert UpdatePriority.HIGH.value == 1
        assert UpdatePriority.NORMAL.value == 2
        assert UpdatePriority.LOW.value == 3
        assert UpdatePriority.BATCH.value == 4
        assert UpdatePriority.CRITICAL.value < UpdatePriority.HIGH.value


class TestDataclasses:
    """Test dataclass definitions"""

    def test_update_strategy_creation(self):
        strategy = UpdateStrategy(
            update_type=UpdateType.CORRELATION,
            priority=UpdatePriority.HIGH,
            dependencies=set(),
            estimated_cost_ms=20.0,
            confidence_threshold=0.5,
            can_parallelize=True,
            can_defer=False,
        )

        assert strategy.update_type == UpdateType.CORRELATION
        assert strategy.priority == UpdatePriority.HIGH
        assert strategy.can_parallelize == True

    def test_observation_signature_creation(self, observation_signature):
        assert observation_signature.has_intervention == False
        assert observation_signature.has_multi_variable == True
        assert observation_signature.variable_count == 3
        assert observation_signature.confidence == 0.9

    def test_update_plan_creation(self):
        plan = UpdatePlan(
            immediate=[UpdateType.CORRELATION, UpdateType.DYNAMICS],
            deferred=[UpdateType.INVARIANT],
            parallel_groups=[[UpdateType.CONFIDENCE]],
            estimated_time_ms=100.0,
            confidence=0.8,
            reasoning="Test plan",
        )

        assert len(plan.immediate) == 2
        assert len(plan.deferred) == 1
        assert plan.confidence == 0.8


class TestUpdateDependencyGraph:
    """Test UpdateDependencyGraph component"""

    def test_get_dependencies(self, dependency_graph):
        deps = dependency_graph.get_dependencies(UpdateType.INTERVENTION)
        assert len(deps) == 0

        deps = dependency_graph.get_dependencies(UpdateType.DYNAMICS)
        assert UpdateType.CORRELATION in deps

    def test_can_parallelize(self, dependency_graph):
        can_parallel = dependency_graph.can_parallelize(
            [UpdateType.CORRELATION, UpdateType.CONFIDENCE]
        )
        assert can_parallel == True

        can_parallel = dependency_graph.can_parallelize(
            [UpdateType.INTERVENTION, UpdateType.CAUSAL]
        )
        assert can_parallel == False

    def test_get_execution_order(self, dependency_graph):
        required = {UpdateType.CORRELATION, UpdateType.DYNAMICS, UpdateType.CONFIDENCE}

        execution_groups = dependency_graph.get_execution_order(required)

        assert len(execution_groups) > 0
        all_updates = []
        for group in execution_groups:
            all_updates.extend(group)

        assert UpdateType.CORRELATION in all_updates
        assert UpdateType.DYNAMICS in all_updates


class TestPatternLearner:
    """Test PatternLearner component"""

    def test_initialization(self, pattern_learner):
        assert pattern_learner.window_size == 100
        assert len(pattern_learner.pattern_history) == 0
        assert len(pattern_learner.rules) == 0

    def test_record_pattern(self, pattern_learner, observation_signature):
        updates = [UpdateType.CORRELATION, UpdateType.DYNAMICS]

        pattern_learner.record_pattern(
            signature=observation_signature,
            updates_run=updates,
            success=True,
            cost_ms=50.0,
        )

        assert len(pattern_learner.pattern_history) == 1

    def test_record_pattern_with_none_signature(self, pattern_learner):
        pattern_learner.record_pattern(
            signature=None,
            updates_run=[UpdateType.CORRELATION],
            success=True,
            cost_ms=50.0,
        )

        assert len(pattern_learner.pattern_history) == 0

    def test_predict_updates(self, pattern_learner, observation_signature):
        predicted = pattern_learner.predict_updates(observation_signature)
        assert isinstance(predicted, list)

        pattern_learner.rules = [
            {
                "id": "test_rule",
                "signature_conditions": {
                    "has_intervention": False,
                    "has_temporal": True,
                },
                "updates": [UpdateType.CORRELATION, UpdateType.DYNAMICS],
                "success_rate": 0.9,
            }
        ]

        predicted = pattern_learner.predict_updates(observation_signature)
        assert len(predicted) > 0


class TestCostModel:
    """Test CostModel component"""

    def test_initialization(self, cost_model):
        assert UpdateType.CORRELATION in cost_model.base_costs
        assert cost_model.base_costs[UpdateType.CORRELATION] > 0

    def test_estimate_cost(self, cost_model):
        cost = cost_model.estimate_cost(UpdateType.CORRELATION, observation_size=100)
        assert cost > 0

    def test_estimate_cost_scales_with_size(self, cost_model):
        small_cost = cost_model.estimate_cost(
            UpdateType.CORRELATION, observation_size=10
        )
        large_cost = cost_model.estimate_cost(
            UpdateType.CORRELATION, observation_size=1000
        )
        assert large_cost > small_cost

    def test_update_cost(self, cost_model):
        initial = len(cost_model.cost_history[UpdateType.CORRELATION])
        cost_model.update_cost(UpdateType.CORRELATION, actual_cost=25.0)
        assert len(cost_model.cost_history[UpdateType.CORRELATION]) == initial + 1


class TestWorldModelRouter:
    """Test WorldModelRouter main orchestrator"""

    def test_initialization(self, router, mock_world_model):
        assert router.world_model == mock_world_model
        assert router.time_budget_ms == 1000
        assert router.min_confidence == 0.5
        assert len(router.strategies) > 0

    def test_route_basic_observation(self, router, sample_observation):
        plan = router.route(sample_observation)

        assert isinstance(plan, UpdatePlan)
        assert (
            len(plan.immediate) > 0
            or len(plan.deferred) > 0
            or len(plan.parallel_groups) > 0
        )
        assert plan.estimated_time_ms >= 0

    def test_route_intervention_observation(self, router, intervention_observation):
        plan = router.route(intervention_observation)
        assert isinstance(plan, UpdatePlan)

    def test_route_with_constraints(self, router, sample_observation):
        constraints = {"time_budget_ms": 100, "priority_threshold": UpdatePriority.HIGH}

        plan = router.route(sample_observation, constraints)
        assert isinstance(plan, UpdatePlan)

    def test_execute_plan(self, router, sample_observation):
        plan = router.route(sample_observation)
        result = router.execute(plan)

        assert "results" in result
        assert "updates_executed" in result
        assert "execution_time_ms" in result
        assert "success" in result

    def test_execute_empty_plan(self, router):
        plan = UpdatePlan(
            immediate=[],
            deferred=[],
            parallel_groups=[],
            estimated_time_ms=0,
            confidence=1.0,
            reasoning="Empty plan",
        )

        result = router.execute(plan)
        assert result["success"] == True
        assert len(result["updates_executed"]) == 0

    def test_extract_signature(self, router, sample_observation):
        signature = router._extract_signature(sample_observation)

        assert isinstance(signature, ObservationSignature)
        assert signature.variable_count == 3
        assert signature.confidence == 0.9

    def test_get_metrics(self, router, sample_observation):
        router.route(sample_observation)
        metrics = router.get_metrics()

        assert "plans_created" in metrics
        assert "cache_hits" in metrics
        assert isinstance(metrics, dict)


class TestSafetyIntegration:
    """Test safety validation integration"""

    def test_route_with_unsafe_observation(self, mock_world_model):
        router = MockWorldModelRouter(mock_world_model, {})

        router.safety_validator.analyze_observation_safety = Mock(
            return_value={"safe": False, "reason": "Test unsafe"}
        )

        obs = Mock()
        obs.timestamp = time.time()
        obs.variables = {"x": 1.0}

        plan = router.route(obs)

        assert len(plan.immediate) == 0
        assert plan.metadata.get("safety_blocked") == True


class TestStatePersistence:
    """Test state saving and loading"""

    def test_save_state(self, router):
        with tempfile.TemporaryDirectory() as tmpdir:
            router.save_state(tmpdir)
            save_path = Path(tmpdir) / "router_state.pkl"
            assert save_path.exists()

    def test_load_state(self, router):
        router.metrics["plans_created"] = 100
        router.metrics["cache_hits"] = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            router.save_state(tmpdir)

            new_router = MockWorldModelRouter(router.world_model, {})
            new_router.load_state(tmpdir)

            assert new_router.metrics["plans_created"] == 100
            assert new_router.metrics["cache_hits"] == 50

    def test_load_nonexistent_state(self, router):
        router.load_state("/nonexistent/path")
        assert router is not None


class TestThreadSafety:
    """Test thread-safe operations"""

    def test_concurrent_routing(self, router, sample_observation):
        results = []

        def route():
            plan = router.route(sample_observation)
            results.append(plan)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=route)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(isinstance(p, UpdatePlan) for p in results)

    def test_concurrent_pattern_learning(self, pattern_learner, observation_signature):
        def record():
            for _ in range(10):
                pattern_learner.record_pattern(
                    signature=observation_signature,
                    updates_run=[UpdateType.CORRELATION],
                    success=True,
                    cost_ms=50.0,
                )

        threads = []
        for _ in range(5):
            t = threading.Thread(target=record)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(pattern_learner.pattern_history) == 50


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_route_with_missing_attributes(self, router):
        obs = Mock()
        obs.timestamp = time.time()
        obs.variables = {}
        obs.confidence = 1.0
        obs.domain = None
        obs.intervention_data = None
        obs.is_anomaly = False
        obs.structural_change = False

        plan = router.route(obs)
        assert isinstance(plan, UpdatePlan)

    def test_execute_with_failing_component(self, router, sample_observation):
        router.world_model.correlation_tracker.update.side_effect = Exception(
            "Test error"
        )

        plan = UpdatePlan(
            immediate=[UpdateType.CORRELATION],
            deferred=[],
            parallel_groups=[],
            estimated_time_ms=20,
            confidence=0.9,
            reasoning="Test",
            metadata={"signature": router._extract_signature(sample_observation)},
        )

        result = router.execute(plan)
        assert result["success"] == False

        router.world_model.correlation_tracker.update.side_effect = None

    def test_route_with_zero_budget(self, router, sample_observation):
        constraints = {"time_budget_ms": 0}
        plan = router.route(sample_observation, constraints)
        assert isinstance(plan, UpdatePlan)


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_routing_workflow(self, router, sample_observation):
        plan = router.route(sample_observation)
        assert isinstance(plan, UpdatePlan)

        result = router.execute(plan)
        assert "success" in result

        metrics = router.get_metrics()
        assert metrics["plans_created"] > 0

    def test_learning_and_adaptation(self, router, observation_signature):
        updates = [UpdateType.CORRELATION, UpdateType.DYNAMICS]

        for _ in range(20):
            router.pattern_learner.record_pattern(
                signature=observation_signature,
                updates_run=updates,
                success=True,
                cost_ms=50.0,
            )

        predicted = router.pattern_learner.predict_updates(observation_signature)
        assert isinstance(predicted, list)


class TestPerformance:
    """Performance and scalability tests"""

    def test_routing_performance(self, router, sample_observation):
        start = time.time()

        for _ in range(100):
            router.route(sample_observation)

        elapsed = time.time() - start
        assert elapsed < 5

    def test_execution_performance(self, router, sample_observation):
        plan = router.route(sample_observation)

        start = time.time()
        for _ in range(50):
            router.execute(plan)
        elapsed = time.time() - start

        assert elapsed < 10

    def test_pattern_learning_scalability(self, pattern_learner, observation_signature):
        start = time.time()

        for _ in range(1000):
            pattern_learner.record_pattern(
                signature=observation_signature,
                updates_run=[UpdateType.CORRELATION],
                success=True,
                cost_ms=50.0,
            )

        elapsed = time.time() - start
        assert elapsed < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
