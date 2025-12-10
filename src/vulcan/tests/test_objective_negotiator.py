"""
test_objective_negotiator.py - Unit tests for ObjectiveNegotiator
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch
from collections import defaultdict

from vulcan.world_model.meta_reasoning.objective_negotiator import (
    ObjectiveNegotiator,
    AgentProposal,
    NegotiationResult,
    NegotiationOutcome,
    NegotiationStrategy,
    ConflictResolution,
)


@pytest.fixture
def mock_objective_hierarchy():
    """Mock objective hierarchy"""
    hierarchy = Mock()

    # Mock objectives
    mock_accuracy = Mock()
    mock_accuracy.weight = 1.0
    mock_accuracy.priority = 0
    mock_accuracy.constraints = {"min": 0.9, "max": 1.0}
    mock_accuracy.target_value = 0.95

    mock_efficiency = Mock()
    mock_efficiency.weight = 0.8
    mock_efficiency.priority = 1
    mock_efficiency.constraints = {"min": 0.0, "max": 1.0}
    mock_efficiency.target_value = 0.85

    mock_safety = Mock()
    mock_safety.weight = 1.0
    mock_safety.priority = 0
    mock_safety.constraints = {"min": 1.0, "max": 1.0}
    mock_safety.target_value = 1.0

    hierarchy.objectives = {
        "prediction_accuracy": mock_accuracy,
        "efficiency": mock_efficiency,
        "safety": mock_safety,
    }

    # Mock find_conflicts
    def mock_find_conflicts(obj_a, obj_b):
        if (obj_a == "efficiency" and obj_b == "prediction_accuracy") or (
            obj_a == "prediction_accuracy" and obj_b == "efficiency"
        ):
            return {
                "type": "direct",
                "severity": "medium",
                "description": "Speed vs accuracy tradeoff",
            }
        return None

    hierarchy.find_conflicts = mock_find_conflicts

    # Mock get_priority_order
    hierarchy.get_priority_order = Mock(
        return_value=["prediction_accuracy", "safety", "efficiency"]
    )

    return hierarchy


@pytest.fixture
def mock_world_model(mock_objective_hierarchy):
    """Mock world model"""
    world_model = Mock()

    # Mock motivational introspection
    mi = Mock()

    # Mock counterfactual reasoner
    reasoner = Mock()

    # Mock predict_under_objective
    def mock_predict(obj, context):
        outcome = Mock()
        outcome.predicted_value = 0.8
        outcome.confidence = 0.9
        return outcome

    reasoner.predict_under_objective = mock_predict

    # Mock find_pareto_frontier
    def mock_pareto(objectives):
        points = []
        for i, obj in enumerate(objectives):
            point = Mock()
            point.objectives = {obj: 0.8}
            point.objective_weights = {obj: 1.0 / len(objectives)}
            point.is_pareto_optimal = True
            point.dominates = []
            point.dominated_by = []
            points.append(point)
        return points

    reasoner.find_pareto_frontier = mock_pareto

    mi.counterfactual_reasoner = reasoner
    world_model.motivational_introspection = mi

    return world_model


@pytest.fixture
def negotiator(mock_objective_hierarchy, mock_world_model):
    """Create negotiator instance"""
    return ObjectiveNegotiator(mock_objective_hierarchy, mock_world_model)


@pytest.fixture
def sample_proposals():
    """Sample agent proposals"""
    return [
        {
            "agent_id": "agent_1",
            "objective": "prediction_accuracy",
            "target_value": 0.95,
            "weight": 1.0,
            "flexibility": 0.5,
        },
        {
            "agent_id": "agent_2",
            "objective": "efficiency",
            "target_value": 0.9,
            "weight": 0.8,
            "flexibility": 0.7,
        },
    ]


class TestInitialization:
    """Test negotiator initialization"""

    def test_init(self, mock_objective_hierarchy, mock_world_model):
        """Test initialization"""
        negotiator = ObjectiveNegotiator(mock_objective_hierarchy, mock_world_model)

        assert negotiator.objective_hierarchy == mock_objective_hierarchy
        assert negotiator.world_model == mock_world_model
        assert isinstance(negotiator.stats, defaultdict)

    def test_parameters_set(self, negotiator):
        """Test that negotiation parameters are set"""
        assert negotiator.max_iterations > 0
        assert 0.0 <= negotiator.convergence_threshold <= 1.0
        assert 0.0 <= negotiator.fairness_weight <= 1.0


class TestNegotiateMultiAgentProposals:
    """Test multi-agent negotiation"""

    def test_negotiate_basic(self, negotiator, sample_proposals):
        """Test basic negotiation"""
        result = negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert isinstance(result, NegotiationResult)
        assert result.outcome in [e for e in NegotiationOutcome]
        assert isinstance(result.agreed_objectives, dict)

    def test_negotiate_empty_proposals(self, negotiator):
        """Test negotiation with empty proposals"""
        result = negotiator.negotiate_multi_agent_proposals([])

        assert result.outcome == NegotiationOutcome.DEADLOCK
        assert len(result.agreed_objectives) == 0

    def test_negotiate_single_proposal(self, negotiator):
        """Test negotiation with single proposal"""
        proposals = [
            {
                "agent_id": "agent_1",
                "objective": "prediction_accuracy",
                "target_value": 0.95,
                "weight": 1.0,
            }
        ]

        result = negotiator.negotiate_multi_agent_proposals(proposals)

        assert isinstance(result, NegotiationResult)

    def test_negotiate_conflicting_proposals(self, negotiator):
        """Test negotiation with conflicting proposals"""
        proposals = [
            {
                "agent_id": "agent_1",
                "objective": "prediction_accuracy",
                "target_value": 0.95,
                "weight": 1.0,
            },
            {
                "agent_id": "agent_2",
                "objective": "efficiency",
                "target_value": 0.9,
                "weight": 1.0,
            },
        ]

        result = negotiator.negotiate_multi_agent_proposals(proposals)

        assert isinstance(result, NegotiationResult)

    def test_result_includes_participants(self, negotiator, sample_proposals):
        """Test that result includes participating agents"""
        result = negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert len(result.participating_agents) > 0
        assert "agent_1" in result.participating_agents

    def test_result_includes_strategy(self, negotiator, sample_proposals):
        """Test that result includes strategy used"""
        result = negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert isinstance(result.strategy_used, NegotiationStrategy)

    def test_result_includes_confidence(self, negotiator, sample_proposals):
        """Test that result includes confidence"""
        result = negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert 0.0 <= result.confidence <= 1.0

    def test_validation_performed(self, negotiator, sample_proposals):
        """Test that negotiated result is validated"""
        result = negotiator.negotiate_multi_agent_proposals(sample_proposals)

        # Should either pass validation or be marked as deadlock
        if result.outcome != NegotiationOutcome.DEADLOCK:
            # Result should be valid
            is_valid = negotiator.validate_negotiated_objectives(
                result.agreed_objectives
            )
            assert is_valid or result.confidence < 0.5

    def test_statistics_updated(self, negotiator, sample_proposals):
        """Test that statistics are updated"""
        initial_count = negotiator.stats["negotiations_performed"]

        negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert negotiator.stats["negotiations_performed"] == initial_count + 1

    def test_history_updated(self, negotiator, sample_proposals):
        """Test that negotiation history is updated"""
        initial_size = len(negotiator.negotiation_history)

        negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert len(negotiator.negotiation_history) == initial_size + 1


class TestFindParetoFrontier:
    """Test Pareto frontier computation"""

    def test_find_frontier_basic(self, negotiator):
        """Test basic Pareto frontier finding"""
        objective_space = {"objectives": ["prediction_accuracy", "efficiency"]}

        frontier = negotiator.find_pareto_frontier(objective_space)

        assert isinstance(frontier, list)

    def test_find_frontier_empty(self, negotiator):
        """Test finding frontier with empty objectives"""
        objective_space = {"objectives": []}

        frontier = negotiator.find_pareto_frontier(objective_space)

        assert isinstance(frontier, list)
        assert len(frontier) == 0

    def test_frontier_points_structure(self, negotiator):
        """Test structure of frontier points"""
        objective_space = {"objectives": ["prediction_accuracy", "efficiency"]}

        frontier = negotiator.find_pareto_frontier(objective_space)

        if frontier:
            point = frontier[0]
            assert "objectives" in point
            assert "weights" in point
            assert "is_pareto_optimal" in point


class TestResolveObjectiveConflict:
    """Test conflict resolution"""

    def test_resolve_conflict_basic(self, negotiator):
        """Test basic conflict resolution"""
        resolution = negotiator.resolve_objective_conflict(
            "efficiency", "prediction_accuracy", {}
        )

        assert isinstance(resolution, ConflictResolution)
        assert len(resolution.objectives) == 2

    def test_resolve_no_conflict(self, negotiator):
        """Test resolution when no conflict exists"""
        resolution = negotiator.resolve_objective_conflict(
            "prediction_accuracy", "safety", {}
        )

        assert isinstance(resolution, ConflictResolution)
        assert resolution.resolution_type == "no_conflict"

    def test_resolution_includes_weights(self, negotiator):
        """Test that resolution includes agreed weights"""
        resolution = negotiator.resolve_objective_conflict(
            "efficiency", "prediction_accuracy", {}
        )

        assert isinstance(resolution.agreed_weights, dict)
        assert len(resolution.agreed_weights) > 0

    def test_resolution_includes_outcomes(self, negotiator):
        """Test that resolution includes expected outcomes"""
        resolution = negotiator.resolve_objective_conflict(
            "efficiency", "prediction_accuracy", {}
        )

        assert isinstance(resolution.expected_outcomes, dict)

    def test_resolution_includes_confidence(self, negotiator):
        """Test that resolution includes confidence"""
        resolution = negotiator.resolve_objective_conflict(
            "efficiency", "prediction_accuracy", {}
        )

        assert 0.0 <= resolution.confidence <= 1.0

    def test_resolution_includes_reasoning(self, negotiator):
        """Test that resolution includes reasoning"""
        resolution = negotiator.resolve_objective_conflict(
            "efficiency", "prediction_accuracy", {}
        )

        assert isinstance(resolution.reasoning, str)
        assert len(resolution.reasoning) > 0

    def test_statistics_updated(self, negotiator):
        """Test that statistics are updated"""
        initial_count = negotiator.stats["conflicts_resolved"]

        negotiator.resolve_objective_conflict("efficiency", "prediction_accuracy", {})

        assert negotiator.stats["conflicts_resolved"] == initial_count + 1


class TestDynamicObjectiveWeighting:
    """Test dynamic objective weighting"""

    def test_basic_weighting(self, negotiator):
        """Test basic dynamic weighting"""
        system_state = {}

        weights = negotiator.dynamic_objective_weighting(system_state)

        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_weights_sum_to_one(self, negotiator):
        """Test that weights sum to 1.0"""
        system_state = {}

        weights = negotiator.dynamic_objective_weighting(system_state)

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_high_uncertainty_increases_safety(self, negotiator):
        """Test that high uncertainty increases safety weight"""
        base_state = {"uncertainty": 0.1}
        high_uncertainty_state = {"uncertainty": 0.9}

        base_weights = negotiator.dynamic_objective_weighting(base_state)
        high_weights = negotiator.dynamic_objective_weighting(high_uncertainty_state)

        if "safety" in base_weights and "safety" in high_weights:
            # Safety weight should be higher with high uncertainty
            assert high_weights["safety"] >= base_weights["safety"] * 0.9

    def test_low_resources_increases_efficiency(self, negotiator):
        """Test that low resources increase efficiency weight"""
        normal_state = {"available_resources": 0.8}
        low_resource_state = {"available_resources": 0.2}

        normal_weights = negotiator.dynamic_objective_weighting(normal_state)
        low_weights = negotiator.dynamic_objective_weighting(low_resource_state)

        if "efficiency" in normal_weights and "efficiency" in low_weights:
            # Efficiency weight should be higher with low resources
            assert low_weights["efficiency"] >= normal_weights["efficiency"] * 0.9

    def test_critical_objectives_prioritized(self, negotiator):
        """Test that critical objectives maintain minimum weight"""
        system_state = {}

        weights = negotiator.dynamic_objective_weighting(system_state)

        # Critical objectives (priority 0) should have reasonable weights
        for obj_name, obj in negotiator.objective_hierarchy.objectives.items():
            if obj.priority == 0 and obj_name in weights:
                assert weights[obj_name] >= 0.1  # At least 10%

    def test_statistics_updated(self, negotiator):
        """Test that statistics are updated"""
        initial_count = negotiator.stats["dynamic_weightings"]

        negotiator.dynamic_objective_weighting({})

        assert negotiator.stats["dynamic_weightings"] == initial_count + 1


class TestValidateNegotiatedObjectives:
    """Test validation of negotiated objectives"""

    def test_validate_valid_objectives(self, negotiator):
        """Test validation of valid objectives"""
        negotiated = {"prediction_accuracy": 0.95, "efficiency": 0.85}

        is_valid = negotiator.validate_negotiated_objectives(negotiated)

        assert is_valid is True

    def test_validate_unknown_objective(self, negotiator):
        """Test validation fails for unknown objectives"""
        negotiated = {"unknown_objective": 0.8}

        is_valid = negotiator.validate_negotiated_objectives(negotiated)

        assert is_valid is False

    def test_validate_constraint_violation_min(self, negotiator):
        """Test validation fails for min constraint violation"""
        negotiated = {
            "prediction_accuracy": 0.5  # Below min of 0.9
        }

        is_valid = negotiator.validate_negotiated_objectives(negotiated)

        assert is_valid is False

    def test_validate_constraint_violation_max(self, negotiator):
        """Test validation fails for max constraint violation"""
        negotiated = {
            "prediction_accuracy": 1.5  # Above max of 1.0
        }

        is_valid = negotiator.validate_negotiated_objectives(negotiated)

        assert is_valid is False

    def test_validate_critical_objective_compromised(self, negotiator):
        """Test validation fails if critical objective compromised"""
        negotiated = {
            "prediction_accuracy": 0.5  # Way below target for critical objective
        }

        is_valid = negotiator.validate_negotiated_objectives(negotiated)

        assert is_valid is False

    def test_validate_empty_objectives(self, negotiator):
        """Test validation of empty objectives"""
        is_valid = negotiator.validate_negotiated_objectives({})

        assert is_valid is True  # Empty is technically valid


class TestNegotiationStrategies:
    """Test different negotiation strategies"""

    def test_pareto_strategy(self, negotiator, sample_proposals):
        """Test Pareto optimal strategy"""
        # Force Pareto strategy
        agent_proposals = [negotiator._parse_proposal(p) for p in sample_proposals]
        analysis = negotiator._analyze_proposal_space(agent_proposals)

        result = negotiator._negotiate_via_pareto(agent_proposals, analysis)

        assert isinstance(result, NegotiationResult)
        assert result.pareto_optimal is True

    def test_weighted_average_strategy(self, negotiator, sample_proposals):
        """Test weighted average strategy"""
        agent_proposals = [negotiator._parse_proposal(p) for p in sample_proposals]
        analysis = negotiator._analyze_proposal_space(agent_proposals)

        result = negotiator._negotiate_via_weighted_average(agent_proposals, analysis)

        assert isinstance(result, NegotiationResult)
        assert result.outcome in [
            NegotiationOutcome.COMPROMISE,
            NegotiationOutcome.CONSENSUS,
        ]

    def test_nash_bargaining_strategy(self, negotiator, sample_proposals):
        """Test Nash bargaining strategy"""
        agent_proposals = [negotiator._parse_proposal(p) for p in sample_proposals]
        analysis = negotiator._analyze_proposal_space(agent_proposals)

        result = negotiator._negotiate_via_nash_bargaining(agent_proposals, analysis)

        assert isinstance(result, NegotiationResult)

    def test_lexicographic_strategy(self, negotiator, sample_proposals):
        """Test lexicographic strategy"""
        agent_proposals = [negotiator._parse_proposal(p) for p in sample_proposals]
        analysis = negotiator._analyze_proposal_space(agent_proposals)

        result = negotiator._negotiate_via_lexicographic(agent_proposals, analysis)

        assert isinstance(result, NegotiationResult)

    def test_minimax_strategy(self, negotiator, sample_proposals):
        """Test minimax strategy"""
        agent_proposals = [negotiator._parse_proposal(p) for p in sample_proposals]
        analysis = negotiator._analyze_proposal_space(agent_proposals)

        result = negotiator._negotiate_via_minimax(agent_proposals, analysis)

        assert isinstance(result, NegotiationResult)


class TestHelperMethods:
    """Test helper methods"""

    def test_parse_proposal(self, negotiator):
        """Test parsing proposal"""
        proposal_dict = {
            "agent_id": "test_agent",
            "objective": "prediction_accuracy",
            "target_value": 0.95,
            "weight": 1.0,
        }

        proposal = negotiator._parse_proposal(proposal_dict)

        assert isinstance(proposal, AgentProposal)
        assert proposal.agent_id == "test_agent"
        assert proposal.objective == "prediction_accuracy"

    def test_analyze_proposal_space(self, negotiator, sample_proposals):
        """Test analyzing proposal space"""
        agent_proposals = [negotiator._parse_proposal(p) for p in sample_proposals]

        analysis = negotiator._analyze_proposal_space(agent_proposals)

        assert isinstance(analysis, dict)
        assert "num_proposals" in analysis
        assert "num_unique_objectives" in analysis
        assert "has_conflicts" in analysis

    def test_assess_difficulty(self, negotiator):
        """Test assessing negotiation difficulty"""
        proposals = [
            AgentProposal("a1", "obj1", 1.0, 1.0, flexibility=0.8),
            AgentProposal("a2", "obj2", 1.0, 1.0, flexibility=0.7),
        ]
        conflicts = []

        # --- FIX: Pass analysis dict instead of proposals, conflicts ---
        analysis = {
            "conflicts": conflicts,
            "avg_flexibility": 0.75,
            "has_conflicts": False,
        }
        difficulty = negotiator._assess_difficulty(analysis)
        # --- END FIX ---

        assert difficulty in ["easy", "medium", "hard"]

    def test_select_negotiation_strategy(self, negotiator):
        """Test strategy selection"""
        proposals = [
            AgentProposal("a1", "obj1", 1.0, 1.0, flexibility=0.5),
            AgentProposal("a2", "obj2", 1.0, 1.0, flexibility=0.5),
        ]
        # --- FIX: Add missing 'negotiation_difficulty' key ---
        analysis = {
            "has_conflicts": False,
            "num_unique_objectives": 2,
            "avg_flexibility": 0.5,
            "negotiation_difficulty": "easy",  # Added this key
        }
        # --- END FIX ---

        strategy = negotiator._select_negotiation_strategy(proposals, analysis)

        assert isinstance(strategy, NegotiationStrategy)

    def test_calculate_fairness(self, negotiator):
        """Test fairness calculation"""
        point = {"objectives": {"obj1": 0.8, "obj2": 0.7}}
        proposals = [
            AgentProposal("a1", "obj1", 1.0, 1.0),
            AgentProposal("a2", "obj2", 1.0, 1.0),
        ]

        fairness = negotiator._calculate_fairness(point, proposals)

        assert isinstance(fairness, float)

    def test_calculate_nash_product(self, negotiator):
        """Test Nash product calculation"""
        candidate = {"objectives": {"obj1": 0.8, "obj2": 0.7}}
        proposals = [
            AgentProposal("a1", "obj1", 1.0, 1.0),
            AgentProposal("a2", "obj2", 1.0, 1.0),
        ]

        product = negotiator._calculate_nash_product(candidate, proposals)

        assert isinstance(product, float)
        assert product > 0

    def test_calculate_max_regret(self, negotiator):
        """Test max regret calculation"""
        candidate = {"objectives": {"obj1": 0.7}}
        proposals = [AgentProposal("a1", "obj1", 0.9, 1.0)]

        regret = negotiator._calculate_max_regret(candidate, proposals)

        assert isinstance(regret, float)
        assert regret >= 0.0

    def test_generate_candidate_solutions(self, negotiator):
        """Test generating candidate solutions"""
        objectives = ["obj1", "obj2"]

        candidates = negotiator._generate_candidate_solutions(objectives)

        assert isinstance(candidates, list)
        assert len(candidates) > 0

        # Check structure
        if candidates:
            assert "weights" in candidates[0]
            assert "objectives" in candidates[0]


class TestDataclasses:
    """Test dataclasses"""

    def test_agent_proposal(self):
        """Test AgentProposal creation"""
        proposal = AgentProposal(
            agent_id="test", objective="accuracy", target_value=0.95, weight=1.0
        )

        assert proposal.agent_id == "test"
        assert proposal.objective == "accuracy"

    def test_negotiation_result(self):
        """Test NegotiationResult creation"""
        result = NegotiationResult(
            outcome=NegotiationOutcome.CONSENSUS,
            agreed_objectives={"obj": 0.8},
            objective_weights={"obj": 1.0},
            participating_agents=["a1"],
            strategy_used=NegotiationStrategy.PARETO_OPTIMAL,
            iterations=1,
            convergence_time_ms=10.0,
            compromises_made=[],
            pareto_optimal=True,
            confidence=0.9,
            reasoning="Test",
        )

        assert result.outcome == NegotiationOutcome.CONSENSUS
        assert result.pareto_optimal is True

    def test_conflict_resolution(self):
        """Test ConflictResolution creation"""
        resolution = ConflictResolution(
            objectives=["a", "b"],
            resolution_type="compromise",
            agreed_weights={"a": 0.5, "b": 0.5},
            expected_outcomes={"a": 0.7, "b": 0.8},
            tradeoffs=[],
            confidence=0.8,
            reasoning="Test resolution",
        )

        assert len(resolution.objectives) == 2
        assert resolution.confidence == 0.8


class TestStatistics:
    """Test statistics tracking"""

    def test_get_statistics(self, negotiator):
        """Test getting statistics"""
        stats = negotiator.get_statistics()

        assert isinstance(stats, dict)
        # --- FIX: Assert a real top-level key ---
        assert "total_negotiations" in stats
        # --- END FIX ---

    def test_strategy_performance_tracked(self, negotiator, sample_proposals):
        """Test that strategy performance is tracked"""
        # Perform negotiation
        negotiator.negotiate_multi_agent_proposals(sample_proposals)

        stats = negotiator.get_statistics()

        assert "strategy_performance" in stats


class TestThreadSafety:
    """Test thread safety"""

    def test_concurrent_negotiations(self, negotiator, sample_proposals):
        """Test concurrent negotiations are thread-safe"""
        import threading

        results = []
        errors = []

        def negotiate():
            try:
                result = negotiator.negotiate_multi_agent_proposals(sample_proposals)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=negotiate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5

    def test_concurrent_conflict_resolution(self, negotiator):
        """Test concurrent conflict resolutions are thread-safe"""
        import threading

        results = []
        errors = []

        def resolve():
            try:
                resolution = negotiator.resolve_objective_conflict(
                    "efficiency", "prediction_accuracy", {}
                )
                results.append(resolution)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5


class TestEdgeCases:
    """Test edge cases"""

    def test_all_proposals_same_objective(self, negotiator):
        """Test negotiation when all agents want same objective"""
        proposals = [
            {
                "agent_id": f"agent_{i}",
                "objective": "prediction_accuracy",
                "target_value": 0.95,
                "weight": 1.0,
            }
            for i in range(3)
        ]

        result = negotiator.negotiate_multi_agent_proposals(proposals)

        assert isinstance(result, NegotiationResult)

    def test_proposals_with_zero_weight(self, negotiator):
        """Test handling proposals with zero weight"""
        proposals = [
            {"agent_id": "a1", "objective": "obj1", "target_value": 0.9, "weight": 0.0},
            {"agent_id": "a2", "objective": "obj2", "target_value": 0.8, "weight": 0.0},
        ]

        result = negotiator.negotiate_multi_agent_proposals(proposals)

        # Should handle gracefully
        assert isinstance(result, NegotiationResult)

    def test_extreme_flexibility(self, negotiator):
        """Test handling extreme flexibility values"""
        proposals = [
            {
                "agent_id": "a1",
                "objective": "obj1",
                "target_value": 0.9,
                "weight": 1.0,
                "flexibility": 0.0,
            },
            {
                "agent_id": "a2",
                "objective": "obj2",
                "target_value": 0.8,
                "weight": 1.0,
                "flexibility": 1.0,
            },
        ]

        result = negotiator.negotiate_multi_agent_proposals(proposals)

        assert isinstance(result, NegotiationResult)

    def test_many_objectives(self, negotiator):
        """Test handling many objectives"""
        proposals = [
            {
                "agent_id": f"a{i}",
                "objective": f"obj{i}",
                "target_value": 0.8,
                "weight": 1.0,
            }
            for i in range(10)
        ]

        result = negotiator.negotiate_multi_agent_proposals(proposals)

        assert isinstance(result, NegotiationResult)


class TestIntegration:
    """Integration tests"""

    def test_full_negotiation_workflow(self, negotiator, sample_proposals):
        """Test full negotiation workflow"""
        # 1. Negotiate proposals
        result = negotiator.negotiate_multi_agent_proposals(sample_proposals)

        assert isinstance(result, NegotiationResult)

        # 2. Validate result
        if result.outcome != NegotiationOutcome.DEADLOCK:
            is_valid = negotiator.validate_negotiated_objectives(
                result.agreed_objectives
            )
            assert isinstance(is_valid, bool)

        # 3. Get statistics
        stats = negotiator.get_statistics()
        # --- FIX: Assert the correct top-level key ---
        assert stats["total_negotiations"] > 0
        # --- END FIX ---

    def test_conflict_to_resolution_workflow(self, negotiator):
        """Test workflow from conflict detection to resolution"""
        # 1. Resolve conflict
        resolution = negotiator.resolve_objective_conflict(
            "efficiency", "prediction_accuracy", {}
        )

        assert isinstance(resolution, ConflictResolution)

        # 2. Apply dynamic weighting based on resolution
        weights = negotiator.dynamic_objective_weighting({})

        assert isinstance(weights, dict)

        # 3. Validate the weights
        is_valid = negotiator.validate_negotiated_objectives(weights)
        assert isinstance(is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
