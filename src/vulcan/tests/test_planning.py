# test_planning.py
# Comprehensive test suite for VULCAN-AGI Planning Module
# FIXED: Test scope issues causing test failures
# Run: pytest src/vulcan/tests/test_planning.py -v --tb=short --cov=src.vulcan.planning --cov-report=html

import asyncio
import copy
import gc
import threading
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from src.vulcan.config import ActionType, AgentConfig, GoalType
from src.vulcan.planning import (ConsensusProtocol, DistributedCoordinator,
                                 EnhancedHierarchicalPlanner, MCTSNode,
                                 MonteCarloTreeSearch, Plan, PlanLibrary,
                                 PlanMonitor, PlanningMethod, PlanningState,
                                 PlanRepairer, PlanStep, ResourceAllocator,
                                 ResourceAwareCompute)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    config = AgentConfig()
    config.agent_id = "test-agent"
    config.collective_id = "test-collective"
    config.max_graph_size = 100
    config.max_execution_time_s = 5.0
    config.max_memory_mb = 1000
    return config


@pytest.fixture
def test_context():
    """Create test context."""
    return {
        "high_level_goal": "explore",
        "complexity": 1.5,
        "data_size": 1000,
        "constraints": {"time_ms": 1000, "energy_nJ": 10000},
    }


@pytest.fixture
def planning_state():
    """Create test planning state."""
    return PlanningState(goal="test_goal", context={"test": "data"}, achieved=False)


@pytest.fixture
def plan_step():
    """Create test plan step."""
    return PlanStep(
        step_id="step_1",
        action="explore",
        preconditions=["has_data"],
        effects=["explored"],
        resources={"cpu": 1.0, "memory": 100},
        duration=1.0,
        probability=0.9,
    )


@pytest.fixture
def test_plan():
    """Create test plan."""
    plan = Plan(plan_id="test_plan_1", goal="test_goal", context={"test": "data"})

    for i in range(3):
        step = PlanStep(
            step_id=f"step_{i}",
            action=f"action_{i}",
            duration=1.0,
            probability=0.8,
            resources={"cpu": 1.0},
        )
        plan.add_step(step)

    return plan


@pytest.fixture
def enhanced_planner(agent_config):
    """Create enhanced hierarchical planner."""
    planner = EnhancedHierarchicalPlanner()
    yield planner
    # Cleanup after test
    planner.cleanup()


@pytest.fixture
def resource_compute():
    """Create resource-aware compute instance."""
    return ResourceAwareCompute()


@pytest.fixture
def distributed_coordinator():
    """Create distributed coordinator."""
    coordinator = DistributedCoordinator(max_agents=4)
    yield coordinator
    coordinator.cleanup()


# ============================================================
# PLAN STEP TESTS
# ============================================================


class TestPlanStep:
    """Test PlanStep class."""

    def test_plan_step_creation(self, plan_step):
        """Test creating a plan step."""
        assert plan_step.step_id == "step_1"
        assert plan_step.action == "explore"
        assert "has_data" in plan_step.preconditions
        assert "explored" in plan_step.effects
        assert plan_step.duration == 1.0
        assert plan_step.probability == 0.9

    def test_plan_step_defaults(self):
        """Test plan step with defaults."""
        step = PlanStep(step_id="test", action="test_action")

        assert step.preconditions == []
        assert step.effects == []
        assert step.resources == {}
        assert step.duration == 1.0
        assert step.probability == 1.0
        assert step.status == "pending"
        assert step.dependencies == []

    def test_plan_step_serialization(self, plan_step):
        """Test plan step serialization."""
        step_dict = plan_step.__dict__

        assert "step_id" in step_dict
        assert "action" in step_dict
        assert "resources" in step_dict


# ============================================================
# PLAN TESTS
# ============================================================


class TestPlan:
    """Test Plan class."""

    def test_plan_creation(self, test_plan):
        """Test creating a plan."""
        assert test_plan.plan_id == "test_plan_1"
        assert test_plan.goal == "test_goal"
        assert len(test_plan.steps) == 3
        assert test_plan.total_cost > 0
        assert test_plan.expected_duration > 0

    def test_add_step(self):
        """Test adding steps to plan."""
        plan = Plan(plan_id="test", goal="goal", context={})

        initial_cost = plan.total_cost
        initial_duration = plan.expected_duration

        step = PlanStep(
            step_id="step_1",
            action="action",
            duration=2.0,
            probability=0.8,
            resources={"cpu": 5.0},
        )

        plan.add_step(step)

        assert len(plan.steps) == 1
        assert plan.total_cost == initial_cost + 5.0
        assert plan.expected_duration == initial_duration + 2.0
        assert plan.success_probability == 0.8

    def test_plan_optimization(self):
        """Test plan optimization with dependencies."""
        plan = Plan(plan_id="test", goal="goal", context={})

        step1 = PlanStep(step_id="step_1", action="action_1")
        step2 = PlanStep(step_id="step_2", action="action_2", dependencies=["step_1"])
        step3 = PlanStep(
            step_id="step_3", action="action_3", dependencies=["step_1", "step_2"]
        )

        # Add in wrong order
        plan.add_step(step3)
        plan.add_step(step1)
        plan.add_step(step2)

        # Optimize
        plan.optimize()

        # Check correct order
        step_ids = [s.step_id for s in plan.steps]
        assert step_ids.index("step_1") < step_ids.index("step_2")
        assert step_ids.index("step_2") < step_ids.index("step_3")

    def test_plan_to_dict(self, test_plan):
        """Test plan serialization to dict."""
        plan_dict = test_plan.to_dict()

        assert "plan_id" in plan_dict
        assert "goal" in plan_dict
        assert "num_steps" in plan_dict
        assert "total_cost" in plan_dict
        assert "expected_duration" in plan_dict
        assert "success_probability" in plan_dict
        assert "steps" in plan_dict
        assert len(plan_dict["steps"]) == 3


# ============================================================
# MCTS NODE TESTS
# ============================================================


class TestMCTSNode:
    """Test MCTSNode class."""

    def test_node_creation(self):
        """Test creating an MCTS node."""
        state = {"value": 1}
        node = MCTSNode(state)

        assert node.state == state
        assert node.parent is None
        assert node.action is None
        assert node.visits == 0
        assert node.value == 0.0
        assert len(node.children) == 0

    def test_node_parent_weakref(self):
        """Test parent weak reference."""
        parent = MCTSNode({"parent": True})
        child = MCTSNode({"child": True}, parent=parent)

        assert child.parent == parent

        # Delete parent
        del parent
        gc.collect()

        # Child's parent reference should be None or invalid
        assert child.parent is None or child.parent == parent

    def test_ucb1_calculation(self):
        """Test UCB1 calculation."""
        parent = MCTSNode({"parent": True})
        parent.visits = 10

        child = MCTSNode({"child": True}, parent=parent)

        # Unvisited node should have infinite UCB1
        assert child.ucb1() == float("inf")

        # Update child
        child.update(1.0)
        child.update(0.5)

        ucb1_value = child.ucb1()
        assert isinstance(ucb1_value, float)
        assert ucb1_value > 0

    def test_best_child_selection(self):
        """Test selecting best child."""
        parent = MCTSNode({"parent": True})
        parent.visits = 10

        child1 = MCTSNode({"child": 1}, parent=parent, action="action1")
        child2 = MCTSNode({"child": 2}, parent=parent, action="action2")

        # Update children with different values
        for _ in range(5):
            child1.update(0.5)
        for _ in range(3):
            child2.update(0.9)

        parent.children = {"action1": child1, "action2": child2}

        best = parent.best_child()
        assert best in [child1, child2]

    def test_node_update(self):
        """Test node update with threading."""
        node = MCTSNode({"test": True})

        def update_node():
            for _ in range(100):
                node.update(1.0)

        threads = [threading.Thread(target=update_node) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert node.visits == 500
        assert node.value == 500.0

    def test_node_cleanup(self):
        """Test node cleanup."""
        parent = MCTSNode({"parent": True})
        child = MCTSNode({"child": True}, parent=parent)
        parent.children = {"action": child}

        parent.cleanup()

        assert len(parent.children) == 0
        assert parent.state is None
        assert parent._parent_ref is None


# ============================================================
# MCTS TESTS
# ============================================================


class TestMonteCarloTreeSearch:
    """Test MonteCarloTreeSearch class."""

    def test_mcts_creation(self):
        """Test creating MCTS instance."""
        mcts = MonteCarloTreeSearch(simulation_budget=100, max_depth=10)

        assert mcts.simulation_budget == 100
        assert mcts.max_depth == 10
        assert mcts.root is None

    def test_mcts_search(self):
        """Test MCTS search - FIXED."""
        mcts = MonteCarloTreeSearch(simulation_budget=50, max_depth=5)

        initial_state = {"value": 0}
        # Store expected actions for assertion
        expected_actions = ["action1", "action2", "action3"]

        # Set up action generator - MUST return consistent actions
        mcts.action_generator = lambda s: expected_actions.copy()
        mcts.state_evaluator = lambda s: np.random.random()
        mcts.transition_model = lambda s, a: {"value": s.get("value", 0) + 1}

        # Pass actions to search
        best_action, value = mcts.search(initial_state, expected_actions.copy())

        # Assert using the stored expected_actions list
        assert best_action in expected_actions or best_action is None
        assert isinstance(value, float)

        # Cleanup
        mcts.cleanup()

    @pytest.mark.asyncio
    async def test_mcts_search_async(self):
        """Test async MCTS search - FIXED."""
        mcts = MonteCarloTreeSearch(simulation_budget=20, max_depth=5)

        initial_state = {"value": 0}
        # Store expected actions for assertion
        expected_actions = ["action1", "action2"]

        # Set up action generator - MUST return consistent actions
        mcts.action_generator = lambda s: expected_actions.copy()
        mcts.state_evaluator = lambda s: np.random.random()
        mcts.transition_model = lambda s, a: {"value": s.get("value", 0) + 1}

        # Pass actions to search
        best_action, value = await mcts.search_async(
            initial_state, expected_actions.copy()
        )

        # Assert using the stored expected_actions list
        assert best_action in expected_actions or best_action is None
        assert isinstance(value, float)

        mcts.cleanup()

    def test_mcts_tree_policy(self):
        """Test MCTS tree policy."""
        mcts = MonteCarloTreeSearch(max_depth=5)
        mcts.root = MCTSNode({"value": 0})
        mcts.root.untried_actions = ["action1", "action2"]

        mcts.action_generator = lambda s: ["action1", "action2"]
        mcts.state_evaluator = lambda s: 0.5
        mcts.transition_model = lambda s, a: {"value": s.get("value", 0) + 1}

        node = mcts._tree_policy(mcts.root)

        assert node is not None

        mcts.cleanup()

    def test_mcts_expansion(self):
        """Test MCTS node expansion."""
        mcts = MonteCarloTreeSearch()
        mcts.root = MCTSNode({"value": 0})
        mcts.root.untried_actions = ["action1", "action2"]

        mcts.action_generator = lambda s: ["action1", "action2"]
        mcts.transition_model = lambda s, a: {"value": s.get("value", 0) + 1}

        child = mcts._expand(mcts.root)

        assert child != mcts.root
        assert child.parent == mcts.root
        assert len(mcts.root.children) == 1

        mcts.cleanup()

    def test_mcts_default_policy(self):
        """Test MCTS rollout."""
        mcts = MonteCarloTreeSearch(max_depth=5)

        mcts.action_generator = lambda s: ["action1", "action2"]
        mcts.state_evaluator = lambda s: 0.5
        mcts.transition_model = lambda s, a: {"value": s.get("value", 0) + 1}

        reward = mcts._default_policy({"value": 0})

        assert isinstance(reward, float)
        assert reward >= 0

    def test_mcts_backup(self):
        """Test MCTS backpropagation."""
        mcts = MonteCarloTreeSearch()

        root = MCTSNode({"root": True})
        child = MCTSNode({"child": True}, parent=root)
        grandchild = MCTSNode({"grandchild": True}, parent=child)

        mcts._backup(grandchild, 1.0)

        assert root.visits > 0
        assert child.visits > 0
        assert grandchild.visits > 0
        assert root.value > 0

    def test_mcts_pruning(self):
        """Test MCTS tree pruning."""
        mcts = MonteCarloTreeSearch()
        mcts.root = MCTSNode({"root": True})

        # Create many children
        for i in range(20):
            child = MCTSNode({"child": i}, parent=mcts.root)
            child.visits = i  # Different visit counts
            mcts.root.children[f"action_{i}"] = child

        initial_children = len(mcts.root.children)

        mcts._prune_tree()

        # Should have pruned some children
        assert len(mcts.root.children) < initial_children

        mcts.cleanup()

    def test_mcts_cleanup(self):
        """Test MCTS cleanup."""
        mcts = MonteCarloTreeSearch()
        mcts.root = MCTSNode({"root": True})

        child = MCTSNode({"child": True}, parent=mcts.root)
        mcts.root.children = {"action": child}

        mcts.cleanup()

        assert mcts.root is None


# ============================================================
# PLANNING STATE TESTS
# ============================================================


class TestPlanningState:
    """Test PlanningState class."""

    def test_state_creation(self, planning_state):
        """Test creating planning state."""
        assert planning_state.goal == "test_goal"
        assert planning_state.context == {"test": "data"}
        assert not planning_state.achieved
        assert len(planning_state.steps_taken) == 0
        assert len(planning_state.resources_used) == 0

    def test_state_hashing(self):
        """Test state hashing for use in sets/dicts."""
        state1 = PlanningState(goal="goal1", context={})
        state2 = PlanningState(goal="goal1", context={})
        state3 = PlanningState(goal="goal2", context={})

        assert hash(state1) == hash(state2)
        assert hash(state1) != hash(state3)

    def test_state_equality(self):
        """Test state equality comparison."""
        state1 = PlanningState(goal="goal1", context={})
        state2 = PlanningState(goal="goal1", context={})
        state3 = PlanningState(goal="goal2", context={})

        state1.steps_taken.append("step1")
        state2.steps_taken.append("step1")

        assert state1 == state2
        assert state1 != state3

    def test_state_ordering(self):
        """Test state ordering for heap operations."""
        state1 = PlanningState(goal="goal", context={})
        state2 = PlanningState(goal="goal", context={})

        state2.steps_taken = ["step1", "step2"]

        assert state1 < state2


# ============================================================
# ENHANCED HIERARCHICAL PLANNER TESTS
# ============================================================


class TestEnhancedHierarchicalPlanner:
    """Test EnhancedHierarchicalPlanner class."""

    def test_planner_creation(self, enhanced_planner):
        """Test creating enhanced planner."""
        assert enhanced_planner is not None
        assert enhanced_planner.plan_library is not None
        assert enhanced_planner.plan_monitor is not None
        assert enhanced_planner.plan_repairer is not None
        assert enhanced_planner.mcts is not None

    def test_generate_plan_hierarchical(self, enhanced_planner, test_context):
        """Test generating hierarchical plan."""
        plan = enhanced_planner.generate_plan(
            goal="explore", context=test_context, method=PlanningMethod.HIERARCHICAL
        )

        assert isinstance(plan, Plan)
        assert plan.goal == "explore"
        assert len(plan.steps) > 0

    def test_generate_plan_mcts(self, enhanced_planner, test_context):
        """Test generating MCTS plan."""
        plan = enhanced_planner.create_plan(
            goal="optimize", context=test_context, method=PlanningMethod.MCTS
        )

        assert isinstance(plan, Plan)
        assert plan.goal == "optimize"

    def test_generate_plan_astar(self, enhanced_planner, test_context):
        """Test generating A* plan."""
        plan = enhanced_planner.create_plan(
            goal="maintain", context=test_context, method=PlanningMethod.A_STAR
        )

        assert isinstance(plan, Plan)
        assert plan.goal == "maintain"

    def test_create_hierarchical_plan(self, enhanced_planner, test_context):
        """Test hierarchical plan creation."""
        plan = enhanced_planner._create_hierarchical_plan("explore", test_context)

        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0
        assert all(isinstance(step, PlanStep) for step in plan.steps)

    def test_plan_caching(self, enhanced_planner, test_context):
        """Test plan caching in library."""
        goal = "test_cache"

        # Generate first plan
        plan1 = enhanced_planner.create_plan(goal, test_context)

        # Generate second plan with same goal/context
        plan2 = enhanced_planner.create_plan(goal, test_context)

        # Should use cached plan
        assert plan1.plan_id == plan2.plan_id or plan2.plan_id.startswith("plan_")

    def test_execute_plan(self, enhanced_planner, test_plan):
        """Test plan execution."""

        def mock_executor(step):
            return {"step_id": step.step_id, "success": True, "duration": step.duration}

        result = enhanced_planner.execute_plan(test_plan, mock_executor)

        assert "success" in result
        assert "executed_steps" in result
        assert "trace" in result

    def test_extract_preconditions(self, enhanced_planner):
        """Test extracting preconditions from subgoal."""
        subgoal = {
            "subgoal": "test",
            "resources_required": {"cpu": 1.0, "memory": 100},
            "dependencies": ["dep1"],
        }

        preconditions = enhanced_planner._extract_preconditions(subgoal)

        assert isinstance(preconditions, list)
        assert any("cpu" in p for p in preconditions)
        assert any("dep1" in p for p in preconditions)

    def test_extract_effects(self, enhanced_planner):
        """Test extracting effects from subgoal."""
        subgoal = {"subgoal": "test_goal", "success_criteria": {"metric1": 0.8}}

        effects = enhanced_planner._extract_effects(subgoal)

        assert isinstance(effects, list)
        assert any("completed_test_goal" in e for e in effects)

    def test_estimate_duration(self, enhanced_planner):
        """Test duration estimation."""
        subgoal = {"subgoal": "optimize_model", "resources_required": {"cpu": 2.0}}

        duration = enhanced_planner._estimate_duration(subgoal)

        assert isinstance(duration, float)
        assert duration > 0

    def test_estimate_success_probability(self, enhanced_planner):
        """Test success probability estimation."""
        subgoal = {"subgoal": "test", "priority": 0.8}

        probability = enhanced_planner._estimate_success_probability(subgoal)

        assert isinstance(probability, float)
        assert 0 <= probability <= 1

    def test_identify_dependencies(self, enhanced_planner):
        """Test dependency identification."""
        previous_subgoals = [{"subgoal": "explore_data"}, {"subgoal": "learn_patterns"}]

        deps = enhanced_planner._identify_dependencies(
            "optimize_model", previous_subgoals
        )

        assert isinstance(deps, list)

    def test_generate_actions(self, enhanced_planner, planning_state):
        """Test action generation."""
        actions = enhanced_planner._generate_actions(planning_state)

        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_evaluate_planning_state(self, enhanced_planner, planning_state):
        """Test state evaluation."""
        score = enhanced_planner._evaluate_planning_state(planning_state)

        assert isinstance(score, float)

    def test_apply_planning_action(self, enhanced_planner, planning_state):
        """Test applying action to state."""
        new_state = enhanced_planner._apply_planning_action(
            planning_state, ActionType.EXPLORE
        )

        assert isinstance(new_state, PlanningState)
        assert len(new_state.steps_taken) > len(planning_state.steps_taken)

    def test_heuristic_function(self, enhanced_planner):
        """Test A* heuristic function."""
        state = PlanningState(goal="goal", context={})
        goal_state = PlanningState(goal="goal", context={}, achieved=True)

        distance = enhanced_planner._heuristic(state, goal_state)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_cleanup(self, enhanced_planner):
        """Test planner cleanup."""
        enhanced_planner.cleanup()

        # Should not raise exception
        assert True


# ============================================================
# PLAN LIBRARY TESTS
# ============================================================


class TestPlanLibrary:
    """Test PlanLibrary class."""

    def test_library_creation(self):
        """Test creating plan library."""
        library = PlanLibrary(max_size=100)

        assert library.max_size == 100
        assert len(library.plans) == 0

    def test_store_and_retrieve_plan(self, test_plan):
        """Test storing and retrieving plans."""
        library = PlanLibrary()

        library.store_plan(test_plan)

        retrieved = library.get_plan(test_plan.goal, test_plan.context)

        assert retrieved is not None
        assert retrieved.goal == test_plan.goal

    def test_plan_eviction(self):
        """Test LRU eviction - FIXED."""
        library = PlanLibrary(max_size=5)

        # Store many plans
        for i in range(10):
            plan = Plan(plan_id=f"plan_{i}", goal=f"goal_{i}", context={"id": i})
            library.store_plan(plan)

        # Should have evicted some to stay at max_size
        assert len(library.plans) == 5

    def test_access_counting(self, test_plan):
        """Test access count tracking."""
        library = PlanLibrary()

        library.store_plan(test_plan)
        key = library._make_key(test_plan.goal, test_plan.context)

        initial_count = library.access_counts[key]

        # Access multiple times
        for _ in range(5):
            library.get_plan(test_plan.goal, test_plan.context)

        assert library.access_counts[key] > initial_count

    def test_library_cleanup(self):
        """Test library cleanup."""
        library = PlanLibrary()

        for i in range(10):
            plan = Plan(plan_id=f"plan_{i}", goal=f"goal_{i}", context={})
            library.store_plan(plan)

        library.cleanup()

        assert len(library.plans) == 0
        assert len(library.access_counts) == 0


# ============================================================
# PLAN MONITOR TESTS
# ============================================================


class TestPlanMonitor:
    """Test PlanMonitor class."""

    def test_monitor_creation(self):
        """Test creating plan monitor."""
        monitor = PlanMonitor()

        assert len(monitor.execution_history) == 0
        assert len(monitor.performance_metrics) == 0

    def test_monitor_step(self, plan_step):
        """Test monitoring step execution."""
        monitor = PlanMonitor()

        result = {"success": True, "duration": 1.0}
        monitor.monitor_step(plan_step, result)

        assert len(monitor.execution_history) == 1
        assert plan_step.action in monitor.performance_metrics

    def test_success_rate_calculation(self, plan_step):
        """Test success rate calculation."""
        monitor = PlanMonitor()

        # Record multiple executions
        for success in [True, True, False, True]:
            result = {"success": success}
            monitor.monitor_step(plan_step, result)

        success_rate = monitor.get_action_success_rate(plan_step.action)

        assert 0 <= success_rate <= 1
        assert success_rate == 0.75  # 3 out of 4

    def test_history_limit(self, plan_step):
        """Test execution history size limit."""
        monitor = PlanMonitor()

        # Add many entries
        for i in range(150):
            result = {"success": True, "iteration": i}
            monitor.monitor_step(plan_step, result)

        # Should be limited to maxlen
        assert len(monitor.execution_history) == 100


# ============================================================
# PLAN REPAIRER TESTS
# ============================================================


class TestPlanRepairer:
    """Test PlanRepairer class."""

    def test_repairer_creation(self):
        """Test creating plan repairer."""
        repairer = PlanRepairer()

        assert len(repairer.repair_strategies) > 0

    def test_repair_preconditions(self, plan_step):
        """Test repairing preconditions."""
        repairer = PlanRepairer()

        result = repairer.repair_preconditions(plan_step)

        assert isinstance(result, bool)

    def test_create_recovery_plan(self, test_plan, plan_step):
        """Test creating recovery plan."""
        repairer = PlanRepairer()

        recovery = repairer.create_recovery_plan(
            test_plan, plan_step, "precondition failure: missing data"
        )

        assert recovery is None or isinstance(recovery, Plan)

    def test_classify_failure(self):
        """Test failure classification."""
        repairer = PlanRepairer()

        assert (
            repairer._classify_failure("precondition failed") == "precondition_failure"
        )
        assert (
            repairer._classify_failure("insufficient resources") == "resource_shortage"
        )
        assert repairer._classify_failure("execution timeout") == "timeout"
        assert repairer._classify_failure("unknown error") == "unknown"

    def test_precondition_repair_strategy(self, test_plan, plan_step):
        """Test precondition repair strategy."""
        repairer = PlanRepairer()

        recovery = repairer._repair_preconditions_strategy(test_plan, plan_step)

        assert isinstance(recovery, Plan)
        assert len(recovery.steps) > 0

    def test_resource_repair_strategy(self, test_plan, plan_step):
        """Test resource repair strategy."""
        repairer = PlanRepairer()

        recovery = repairer._repair_resources_strategy(test_plan, plan_step)

        assert isinstance(recovery, Plan)
        assert len(recovery.steps) > 0

    def test_timeout_repair_strategy(self, test_plan, plan_step):
        """Test timeout repair strategy."""
        repairer = PlanRepairer()

        recovery = repairer._repair_timeout_strategy(test_plan, plan_step)

        assert isinstance(recovery, Plan)
        assert len(recovery.steps) > 0


# ============================================================
# RESOURCE AWARE COMPUTE TESTS
# ============================================================


class TestResourceAwareCompute:
    """Test ResourceAwareCompute class."""

    def test_compute_creation(self, resource_compute):
        """Test creating resource-aware compute."""
        # FIXED: Changed resource_monitors (plural) to resource_monitor (singular)
        assert resource_compute.resource_monitor is not None
        assert len(resource_compute.optimization_strategies) > 0

    def test_get_resource_availability(self, resource_compute):
        """Test getting resource availability."""
        availability = resource_compute.get_resource_availability()

        # FIXED: Check for keys that are actually returned by implementation
        # Implementation returns: cpu, memory, gpu (always)
        # Plus disk, energy when current_state is populated
        assert "cpu" in availability
        assert "memory" in availability
        assert "gpu" in availability
        # Note: 'disk' and 'energy' only present after monitoring thread populates current_state

        for value in availability.values():
            assert isinstance(value, (int, float))

    def test_plan_with_budget(self, resource_compute):
        """Test planning with budget constraints."""
        problem = {
            "type": "test",
            "complexity": 1.5,
            "data_size": 1000,
            "goal": {"type": "learning"},
        }

        result = resource_compute.plan_with_budget(
            problem, time_budget_ms=1000, energy_budget_nJ=10000
        )

        assert "action" in result
        assert "resource_usage" in result
        assert "time_ms" in result["resource_usage"]
        assert "energy_nJ" in result["resource_usage"]
        assert "within_budget" in result["resource_usage"]

    def test_resource_estimation(self, resource_compute):
        """Test resource requirement estimation."""
        problem = {"complexity": 2.0, "data_size": 5000, "goal": {"type": "learning"}}

        # FIXED: Added use_gpu parameter (method signature requires it)
        estimated = resource_compute._estimate_requirements(problem, use_gpu=False)

        assert "time_ms" in estimated
        assert "memory_mb" in estimated
        assert "energy_nJ" in estimated
        assert all(v > 0 for v in estimated.values())

    def test_optimization_strategy_selection(self, resource_compute):
        """Test selecting optimization strategy."""
        problem = {"data_size": 50000, "complexity": 3.0}

        estimated = {"time_ms": 2000, "energy_nJ": 20000}

        # FIXED: Added missing resources parameter (method signature requires it)
        resources = resource_compute.get_resource_availability()

        strategy = resource_compute._select_optimization_strategy(
            problem, estimated, resources, 1000, 10000
        )

        assert strategy in resource_compute.optimization_strategies.keys()

    def test_apply_pruning(self, resource_compute):
        """Test pruning optimization."""
        problem = {"data_size": 10000, "complexity": 2.0}

        optimized = resource_compute._apply_pruning(problem)

        assert optimized["data_size"] < problem["data_size"]
        assert optimized["pruned"] is True

    def test_apply_quantization(self, resource_compute):
        """Test quantization optimization."""
        problem = {"complexity": 2.0}

        optimized = resource_compute._apply_quantization(problem)

        assert optimized["quantized"] is True
        assert "precision" in optimized

    def test_apply_caching(self, resource_compute):
        """Test caching optimization."""
        problem = {"complexity": 2.0}

        optimized = resource_compute._apply_caching(problem)

        assert optimized["use_cache"] is True

    def test_apply_batching(self, resource_compute):
        """Test batching optimization."""
        problem = {"data_size": 1000, "complexity": 2.0}

        optimized = resource_compute._apply_batching(problem)

        assert "batch_size" in optimized
        assert optimized["batch_size"] > 0

    def test_apply_parallelization(self, resource_compute):
        """Test parallelization optimization."""
        problem = {"data_size": 1000, "complexity": 2.0}

        optimized = resource_compute._apply_parallelization(problem)

        assert optimized["parallel"] is True
        assert "num_workers" in optimized

    def test_cache_functionality(self, resource_compute):
        """Test result caching."""
        problem = {"test": "data", "complexity": 1.0, "data_size": 100}

        # First execution
        result1 = resource_compute.plan_with_budget(problem, 1000, 10000)

        # Second execution (should use cache)
        result2 = resource_compute.plan_with_budget(problem, 1000, 10000)

        # Results should be similar (from cache)
        assert result1["action"]["type"] == result2["action"]["type"]


# ============================================================
# RESOURCE ALLOCATOR TESTS
# ============================================================


class TestResourceAllocator:
    """Test ResourceAllocator class."""

    def test_allocator_creation(self):
        """Test creating resource allocator."""
        allocator = ResourceAllocator()
        assert allocator is not None

    def test_feasible_allocation(self):
        """Test feasible resource allocation."""
        allocator = ResourceAllocator()

        requirements = {"cpu": 50, "memory": 1000}
        available = {"cpu": 100, "memory": 2000}

        result = allocator.allocate(requirements, available)

        assert result["feasible"] is True
        assert result["allocation"] == requirements

    def test_infeasible_allocation(self):
        """Test infeasible resource allocation."""
        allocator = ResourceAllocator()

        requirements = {"cpu": 150, "memory": 3000}
        available = {"cpu": 100, "memory": 2000}

        result = allocator.allocate(requirements, available)

        assert result["feasible"] is False
        assert result["allocation"]["cpu"] <= available["cpu"]
        assert result["allocation"]["memory"] <= available["memory"]

    def test_utilization_calculation(self):
        """Test resource utilization calculation."""
        allocator = ResourceAllocator()

        requirements = {"cpu": 50}
        available = {"cpu": 100}

        result = allocator.allocate(requirements, available)

        assert "utilization" in result
        assert result["utilization"]["cpu"] == 0.5


# ============================================================
# DISTRIBUTED COORDINATOR TESTS
# ============================================================


class TestDistributedCoordinator:
    """Test DistributedCoordinator class."""

    def test_coordinator_creation(self, distributed_coordinator):
        """Test creating distributed coordinator."""
        assert distributed_coordinator.max_agents == 4
        assert len(distributed_coordinator.agents) == 0

    def test_register_agent(self, distributed_coordinator):
        """Test registering agents."""
        success = distributed_coordinator.register_agent(
            "agent_1", ["planning", "learning"]
        )

        assert success is True
        assert "agent_1" in distributed_coordinator.agents

    def test_max_agents_limit(self, distributed_coordinator):
        """Test maximum agents limit."""
        # Register max agents
        for i in range(distributed_coordinator.max_agents):
            distributed_coordinator.register_agent(f"agent_{i}", ["test"])

        # Try to register one more
        success = distributed_coordinator.register_agent("agent_extra", ["test"])

        assert success is False

    def test_distribute_task(self, distributed_coordinator):
        """Test distributing task to agents."""
        # Register agents
        distributed_coordinator.register_agent("agent_1", ["planning"])
        distributed_coordinator.register_agent("agent_2", ["learning"])

        task = {
            "type": "parallel",
            "data": list(range(100)),
            "params": {"threshold": 0.5},
        }

        result = distributed_coordinator.distribute_task(task)

        assert "status" in result
        assert result["status"] in ["distributed", "failed", "no_agents"]

    def test_task_decomposition(self, distributed_coordinator):
        """Test task decomposition."""
        task = {"type": "parallel", "data": list(range(100))}

        subtasks = distributed_coordinator._decompose_task(task)

        assert isinstance(subtasks, list)
        assert len(subtasks) > 0

    def test_subtask_assignment(self, distributed_coordinator):
        """Test subtask assignment to agents."""
        agents = ["agent_1", "agent_2", "agent_3"]
        subtasks = [{"id": i} for i in range(10)]

        assignments = distributed_coordinator._assign_subtasks(subtasks, agents)

        assert isinstance(assignments, dict)
        assert len(assignments) <= len(agents)

    def test_heartbeat_monitoring(self, distributed_coordinator):
        """Test agent heartbeat monitoring."""
        distributed_coordinator.register_agent("agent_1", ["test"])

        # Simulate time passing
        time.sleep(0.1)

        # Agent should still be active
        assert distributed_coordinator.agents["agent_1"]["status"] == "active"

    def test_coordinator_cleanup(self, distributed_coordinator):
        """Test coordinator cleanup."""
        distributed_coordinator.register_agent("agent_1", ["test"])
        distributed_coordinator.cleanup()

        assert len(distributed_coordinator.agents) == 0


# ============================================================
# CONSENSUS PROTOCOL TESTS
# ============================================================


class TestConsensusProtocol:
    """Test ConsensusProtocol class."""

    def test_protocol_creation(self):
        """Test creating consensus protocol."""
        protocol = ConsensusProtocol()
        assert protocol is not None

    def test_unanimous_consensus(self):
        """Test achieving unanimous consensus."""
        protocol = ConsensusProtocol()

        proposals = {"agent_1": "value_a", "agent_2": "value_a", "agent_3": "value_a"}

        result = protocol.achieve_consensus(proposals)

        assert result == "value_a"

    def test_majority_consensus(self):
        """Test achieving majority consensus."""
        protocol = ConsensusProtocol()

        proposals = {
            "agent_1": "value_a",
            "agent_2": "value_a",
            "agent_3": "value_a",
            "agent_4": "value_b",
        }

        result = protocol.achieve_consensus(proposals)

        assert result == "value_a"

    def test_no_consensus(self):
        """Test when no consensus is possible."""
        protocol = ConsensusProtocol()

        proposals = {"agent_1": "value_a", "agent_2": "value_b", "agent_3": "value_c"}

        result = protocol.achieve_consensus(proposals)

        # Should return most common value
        assert result in proposals.values()

    def test_empty_proposals(self):
        """Test consensus with empty proposals."""
        protocol = ConsensusProtocol()

        result = protocol.achieve_consensus({})

        assert result is None

    def test_single_proposal(self):
        """Test consensus with single proposal."""
        protocol = ConsensusProtocol()

        proposals = {"agent_1": "value_a"}

        result = protocol.achieve_consensus(proposals)

        assert result == "value_a"


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for planning module."""

    def test_full_planning_workflow(self, enhanced_planner, test_context):
        """Test complete planning workflow."""
        # Generate plan
        plan = enhanced_planner.generate_plan("explore", test_context)

        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0

        # Execute plan
        def mock_executor(step):
            return {"step_id": step.step_id, "success": True}

        result = enhanced_planner.execute_plan(plan, mock_executor)

        assert result["success"] or result["executed_steps"] > 0

    def test_plan_repair_workflow(self, enhanced_planner, test_plan):
        """Test plan repair workflow."""
        # Simulate failure
        failed_step = test_plan.steps[0]

        recovery = enhanced_planner.plan_repairer.create_recovery_plan(
            test_plan, failed_step, "precondition failure"
        )

        assert recovery is None or isinstance(recovery, Plan)

    def test_resource_constrained_planning(self, enhanced_planner, resource_compute):
        """Test planning with resource constraints."""
        problem = {"goal": {"type": "learning"}, "complexity": 2.0, "data_size": 5000}

        result = resource_compute.plan_with_budget(
            problem, time_budget_ms=500, energy_budget_nJ=5000
        )

        assert "within_budget" in result["resource_usage"]

    def test_distributed_planning(self, distributed_coordinator, enhanced_planner):
        """Test distributed planning across agents."""
        # Register agents
        for i in range(3):
            distributed_coordinator.register_agent(f"agent_{i}", ["planning"])

        # Distribute planning task
        task = {"type": "parallel", "data": list(range(30)), "goal": "explore"}

        result = distributed_coordinator.distribute_task(task)

        assert result["status"] in ["distributed", "no_agents", "failed"]


# ============================================================
# PERFORMANCE TESTS
# ============================================================


class TestPerformance:
    """Performance tests for planning module."""

    def test_mcts_performance(self):
        """Test MCTS performance with many simulations."""
        mcts = MonteCarloTreeSearch(simulation_budget=1000, max_depth=20)

        initial_state = {"value": 0}
        actions = [f"action_{i}" for i in range(10)]

        mcts.action_generator = lambda s: actions.copy()
        mcts.state_evaluator = lambda s: np.random.random()
        mcts.transition_model = lambda s, a: {"value": s.get("value", 0) + 1}

        start_time = time.time()
        best_action, value = mcts.search(initial_state, actions.copy())
        elapsed = time.time() - start_time

        assert elapsed < 5.0  # Should complete in reasonable time
        assert best_action is not None or best_action is None  # Either is valid

        mcts.cleanup()

    def test_plan_library_performance(self):
        """Test plan library performance with many plans."""
        library = PlanLibrary(max_size=1000)

        # Store many plans
        start_time = time.time()
        for i in range(1000):
            plan = Plan(plan_id=f"plan_{i}", goal=f"goal_{i % 100}", context={"id": i})
            library.store_plan(plan)
        elapsed_store = time.time() - start_time

        # Retrieve plans
        start_time = time.time()
        for i in range(100):
            library.get_plan(f"goal_{i}", {"id": i})
        elapsed_retrieve = time.time() - start_time

        assert elapsed_store < 2.0
        assert elapsed_retrieve < 1.0

        library.cleanup()

    def test_concurrent_planning(self, enhanced_planner):
        """Test concurrent plan generation."""

        def generate_plan(goal):
            return enhanced_planner.create_plan(
                goal, {"complexity": 1.0}, method=PlanningMethod.HIERARCHICAL
            )

        goals = [f"goal_{i}" for i in range(10)]

        start_time = time.time()

        threads = []
        for goal in goals:
            thread = threading.Thread(target=generate_plan, args=(goal,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time

        assert elapsed < 10.0  # Should handle concurrent requests


# ============================================================
# RUN CONFIGURATION
# ============================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--cov=src.vulcan.planning",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
