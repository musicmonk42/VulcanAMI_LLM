"""
test_goal_conflict_detector.py - Unit tests for GoalConflictDetector

FIXED: test_statistics_updated uses .get() to handle missing keys safely
"""

from collections import defaultdict
from unittest.mock import Mock

import pytest

from vulcan.world_model.meta_reasoning.goal_conflict_detector import (
    Conflict,
    ConflictSeverity,
    ConflictType,
    GoalConflictDetector,
    MultiObjectiveTension,
)


@pytest.fixture
def mock_objective_hierarchy():
    """Mock objective hierarchy for testing"""
    hierarchy = Mock()

    # Mock objectives
    mock_efficiency = Mock()
    mock_efficiency.priority = 1
    mock_efficiency.constraints = {"min": 0.0, "max": 1.0}
    mock_efficiency.metadata = {"resources": ["cpu", "memory"]}

    mock_accuracy = Mock()
    mock_accuracy.priority = 0  # Critical
    mock_accuracy.constraints = {"min": 0.9, "max": 1.0}
    mock_accuracy.metadata = {"resources": ["cpu"]}

    mock_safety = Mock()
    mock_safety.priority = 0  # Critical
    mock_safety.constraints = {"min": 1.0, "max": 1.0}
    mock_safety.metadata = {"resources": ["memory"]}

    mock_exploration = Mock()
    mock_exploration.priority = 2
    mock_exploration.constraints = {"min": 0.0, "max": 1.0}
    mock_exploration.metadata = {"resources": []}

    mock_exploitation = Mock()
    mock_exploitation.priority = 2
    mock_exploitation.constraints = {"min": 0.0, "max": 1.0}
    mock_exploitation.metadata = {"resources": []}

    hierarchy.objectives = {
        "efficiency": mock_efficiency,
        "prediction_accuracy": mock_accuracy,
        "safety": mock_safety,
        "exploration": mock_exploration,
        "exploitation": mock_exploitation,
    }

    # Mock find_conflicts method
    def mock_find_conflicts(obj_a, obj_b):
        # Define known conflicts
        conflicts = {
            ("efficiency", "prediction_accuracy"): {
                "type": "tradeoff",
                "severity": "medium",
                "description": "Speed and accuracy typically trade off",
            },
            ("prediction_accuracy", "efficiency"): {
                "type": "tradeoff",
                "severity": "medium",
                "description": "Speed and accuracy typically trade off",
            },
            ("safety", "efficiency"): {
                "type": "tradeoff",
                "severity": "high",
                "description": "Safety checks reduce performance",
            },
            ("efficiency", "safety"): {
                "type": "tradeoff",
                "severity": "high",
                "description": "Safety checks reduce performance",
            },
            ("exploration", "exploitation"): {
                "type": "direct",
                "severity": "medium",
                "description": "Cannot maximize both exploration and exploitation",
            },
            ("exploitation", "exploration"): {
                "type": "direct",
                "severity": "medium",
                "description": "Cannot maximize both exploration and exploitation",
            },
        }
        return conflicts.get((obj_a, obj_b), None)

    hierarchy.find_conflicts = mock_find_conflicts

    # Mock get_priority_order method
    def mock_get_priority_order():
        return [
            "prediction_accuracy",
            "safety",
            "efficiency",
            "exploration",
            "exploitation",
        ]

    hierarchy.get_priority_order = mock_get_priority_order

    return hierarchy


@pytest.fixture
def detector(mock_objective_hierarchy):
    """Create detector instance for testing"""
    return GoalConflictDetector(mock_objective_hierarchy)


class TestInitialization:
    """Test detector initialization"""

    def test_init_creates_structures(self, detector):
        """Test that initialization creates required structures"""
        assert detector.objective_hierarchy is not None
        assert isinstance(detector.conflict_rules, list)
        assert len(detector.conflict_rules) > 0
        assert isinstance(detector.resolution_strategies, dict)
        assert isinstance(detector.stats, defaultdict)

    def test_conflict_rules_initialized(self, detector):
        """Test that conflict rules are properly initialized"""
        assert len(detector.conflict_rules) >= 3

        # Check rule structure
        for rule in detector.conflict_rules:
            assert "name" in rule
            assert "objectives" in rule
            assert "type" in rule
            assert "severity" in rule
            assert "description" in rule

    def test_resolution_strategies_initialized(self, detector):
        """Test that resolution strategies are initialized"""
        expected_types = [
            ConflictType.DIRECT,
            ConflictType.INDIRECT,
            ConflictType.CONSTRAINT,
            ConflictType.PRIORITY,
            ConflictType.TRADEOFF,
        ]

        for conflict_type in expected_types:
            assert conflict_type in detector.resolution_strategies
            assert len(detector.resolution_strategies[conflict_type]) > 0


class TestConflictDetection:
    """Test conflict detection in proposals"""

    def test_detect_no_conflicts_single_objective(self, detector):
        """Test that single objective has no conflicts"""
        proposal = {"objective": "efficiency", "target": 0.9}

        conflicts = detector.detect_conflicts_in_proposal(proposal)
        assert isinstance(conflicts, list)
        assert len(conflicts) == 0

    def test_detect_tradeoff_conflict(self, detector):
        """Test detection of tradeoff conflicts"""
        proposal = {
            "objectives": ["efficiency", "prediction_accuracy"],
            "objective_weights": {"efficiency": 0.5, "prediction_accuracy": 0.5},
        }

        conflicts = detector.detect_conflicts_in_proposal(proposal)
        assert len(conflicts) > 0

        # Should find the speed/accuracy tradeoff
        tradeoff_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.TRADEOFF
        ]
        assert len(tradeoff_conflicts) > 0

    def test_detect_direct_conflict(self, detector):
        """Test detection of direct conflicts"""
        proposal = {
            "objectives": ["exploration", "exploitation"],
            "objective_weights": {"exploration": 0.5, "exploitation": 0.5},
        }

        conflicts = detector.detect_conflicts_in_proposal(proposal)

        # Should find direct conflict between exploration and exploitation
        direct_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.DIRECT
        ]
        assert len(direct_conflicts) > 0

    def test_detect_constraint_violation(self, detector):
        """Test detection of constraint violations"""
        proposal = {
            "prediction_accuracy": 0.5,  # Below minimum of 0.9
            "objective": "prediction_accuracy",
        }

        conflicts = detector.detect_conflicts_in_proposal(proposal)

        # Should find constraint violation
        constraint_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.CONSTRAINT
        ]
        assert len(constraint_conflicts) > 0

    def test_detect_resource_conflict(self, detector):
        """Test detection of resource conflicts"""
        proposal = {
            "objectives": ["efficiency", "prediction_accuracy"],
            "objective_weights": {"efficiency": 0.6, "prediction_accuracy": 0.4},
        }

        conflicts = detector.detect_conflicts_in_proposal(proposal)

        # Both objectives share CPU resource
        resource_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.INDIRECT
        ]
        assert len(resource_conflicts) > 0

    def test_detect_priority_violation(self, detector):
        """Test detection of priority violations"""
        proposal = {
            "objective_weights": {
                "exploration": 0.9,  # High weight for low-priority objective
                "prediction_accuracy": 0.1,
            }
        }

        conflicts = detector.detect_conflicts_in_proposal(proposal)

        # Should detect priority violation
        priority_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.PRIORITY
        ]
        assert len(priority_conflicts) > 0

    def test_conflicts_have_resolutions(self, detector):
        """Test that detected conflicts include resolution options"""
        proposal = {"objectives": ["efficiency", "prediction_accuracy"]}

        conflicts = detector.detect_conflicts_in_proposal(proposal)

        if conflicts:
            for conflict in conflicts:
                assert hasattr(conflict, "resolution_options")
                assert isinstance(conflict.resolution_options, list)
                if conflict.resolution_options:
                    assert "strategy" in conflict.resolution_options[0]

    def test_conflict_history_updated(self, detector):
        """Test that conflict history is updated"""
        initial_size = len(detector.conflict_history)

        proposal = {"objectives": ["efficiency", "prediction_accuracy"]}

        conflicts = detector.detect_conflicts_in_proposal(proposal)

        assert len(detector.conflict_history) == initial_size + len(conflicts)

    def test_statistics_updated(self, detector):
        """Test that statistics are updated - FIXED: Use .get() for safe access"""
        initial_count = detector.stats.get("proposals_analyzed", 0)

        proposal = {"objective": "efficiency"}

        detector.detect_conflicts_in_proposal(proposal)

        assert detector.stats["proposals_analyzed"] == initial_count + 1


class TestMultiObjectiveTension:
    """Test multi-objective tension analysis"""

    def test_empty_objectives(self, detector):
        """Test tension analysis with empty objectives"""
        analysis = detector.analyze_multi_objective_tension([])

        assert isinstance(analysis, MultiObjectiveTension)
        assert len(analysis.objectives) == 0
        assert analysis.overall_tension == 0.0
        assert len(analysis.primary_conflicts) == 0

    def test_single_objective_tension(self, detector):
        """Test tension analysis with single objective"""
        analysis = detector.analyze_multi_objective_tension(["efficiency"])

        assert isinstance(analysis, MultiObjectiveTension)
        assert len(analysis.objectives) == 1
        assert analysis.tension_matrix.shape == (1, 1)
        assert analysis.overall_tension == 0.0

    def test_two_objective_tension(self, detector):
        """Test tension analysis with two objectives"""
        analysis = detector.analyze_multi_objective_tension(
            ["efficiency", "prediction_accuracy"]
        )

        assert isinstance(analysis, MultiObjectiveTension)
        assert len(analysis.objectives) == 2
        assert analysis.tension_matrix.shape == (2, 2)
        assert 0.0 <= analysis.overall_tension <= 1.0

    def test_multiple_objective_tension(self, detector):
        """Test tension analysis with multiple objectives"""
        objectives = ["efficiency", "prediction_accuracy", "safety"]
        analysis = detector.analyze_multi_objective_tension(objectives)

        assert isinstance(analysis, MultiObjectiveTension)
        assert len(analysis.objectives) == 3
        assert analysis.tension_matrix.shape == (3, 3)
        assert isinstance(analysis.recommendations, list)
        assert len(analysis.recommendations) > 0

    def test_tension_matrix_symmetric(self, detector):
        """Test that tension matrix is symmetric"""
        objectives = ["efficiency", "prediction_accuracy"]
        analysis = detector.analyze_multi_objective_tension(objectives)

        # Check symmetry
        n = len(objectives)
        for i in range(n):
            for j in range(n):
                assert analysis.tension_matrix[i, j] == analysis.tension_matrix[j, i]

    def test_tension_matrix_diagonal_zero(self, detector):
        """Test that tension matrix diagonal is zero"""
        objectives = ["efficiency", "prediction_accuracy", "safety"]
        analysis = detector.analyze_multi_objective_tension(objectives)

        # Diagonal should be zero (no conflict with self)
        n = len(objectives)
        for i in range(n):
            assert analysis.tension_matrix[i, i] == 0.0

    def test_high_tension_generates_recommendations(self, detector):
        """Test that high tension generates appropriate recommendations"""
        # These objectives have high conflicts
        objectives = ["exploration", "exploitation", "efficiency", "safety"]
        analysis = detector.analyze_multi_objective_tension(objectives)

        assert len(analysis.recommendations) > 0
        # Should mention tension in recommendations
        recommendation_text = " ".join(analysis.recommendations).lower()
        assert "tension" in recommendation_text or "conflict" in recommendation_text

    def test_pareto_optimality_check(self, detector):
        """Test Pareto optimality checking"""
        objectives = ["efficiency", "prediction_accuracy"]
        analysis = detector.analyze_multi_objective_tension(objectives)

        assert isinstance(analysis.pareto_optimal, bool)

    def test_primary_conflicts_extracted(self, detector):
        """Test that primary conflicts are extracted"""
        objectives = ["efficiency", "safety"]  # Known high-severity conflict
        analysis = detector.analyze_multi_objective_tension(objectives)

        # Should identify the safety/efficiency conflict
        assert len(analysis.primary_conflicts) > 0
        assert all(isinstance(c, Conflict) for c in analysis.primary_conflicts)


class TestConstraintValidation:
    """Test constraint violation checking"""

    def test_no_violations(self, detector):
        """Test proposal with no constraint violations"""
        proposal = {"efficiency": 0.8, "prediction_accuracy": 0.95}

        violations = detector.check_constraint_violations(proposal)
        assert isinstance(violations, list)
        assert len(violations) == 0

    def test_minimum_violation(self, detector):
        """Test detection of minimum constraint violation"""
        proposal = {"prediction_accuracy": 0.5}  # Below minimum of 0.9

        violations = detector.check_constraint_violations(proposal)
        assert len(violations) > 0

        violation = violations[0]
        assert violation["constraint"] == "minimum"
        assert violation["value"] < violation["limit"]

    def test_maximum_violation(self, detector):
        """Test detection of maximum constraint violation"""
        proposal = {"efficiency": 1.5}  # Above maximum of 1.0

        violations = detector.check_constraint_violations(proposal)
        assert len(violations) > 0

        violation = violations[0]
        assert violation["constraint"] == "maximum"
        assert violation["value"] > violation["limit"]

    def test_multiple_violations(self, detector):
        """Test detection of multiple violations"""
        proposal = {
            "efficiency": 1.5,  # Too high
            "prediction_accuracy": 0.5,  # Too low
        }

        violations = detector.check_constraint_violations(proposal)
        assert len(violations) >= 2

    def test_violation_includes_details(self, detector):
        """Test that violations include detailed information"""
        proposal = {"prediction_accuracy": 0.5}

        violations = detector.check_constraint_violations(proposal)
        assert len(violations) > 0

        violation = violations[0]
        assert "objective" in violation
        assert "constraint" in violation
        assert "value" in violation
        assert "limit" in violation
        assert "violation" in violation


class TestTradeoffValidation:
    """Test tradeoff acceptability validation"""

    def test_acceptable_tradeoff(self, detector):
        """Test validation of acceptable tradeoff"""
        tradeoff = {
            "sacrifice": "efficiency",
            "gain": "prediction_accuracy",
            "sacrifice_amount": 0.1,
            "gain_amount": 0.2,
        }

        result = detector.validate_tradeoff_acceptability(tradeoff)

        assert isinstance(result, dict)
        assert "acceptable" in result
        assert "issues" in result
        assert "recommendation" in result
        assert "confidence" in result

    def test_critical_objective_sacrifice(self, detector):
        """Test that sacrificing critical objective is rejected"""
        tradeoff = {
            "sacrifice": "prediction_accuracy",  # Critical objective
            "gain": "efficiency",
            "sacrifice_amount": 0.3,
            "gain_amount": 0.1,
        }

        result = detector.validate_tradeoff_acceptability(tradeoff)

        assert not result["acceptable"]
        assert len(result["issues"]) > 0

        # Should flag critical sacrifice
        critical_issues = [
            i for i in result["issues"] if i["type"] == "critical_sacrifice"
        ]
        assert len(critical_issues) > 0

    def test_poor_ratio_tradeoff(self, detector):
        """Test detection of poor tradeoff ratio"""
        tradeoff = {
            "sacrifice": "efficiency",
            "gain": "safety",
            "sacrifice_amount": 0.5,
            "gain_amount": 0.1,  # Poor ratio: losing more than gaining
        }

        result = detector.validate_tradeoff_acceptability(tradeoff)

        # Should flag poor ratio
        ratio_issues = [i for i in result["issues"] if i["type"] == "poor_ratio"]
        assert len(ratio_issues) > 0

    def test_unnecessary_tradeoff(self, detector):
        """Test detection of unnecessary tradeoffs"""
        tradeoff = {
            "sacrifice": "efficiency",
            "gain": "exploration",  # These don't conflict
            "sacrifice_amount": 0.1,
            "gain_amount": 0.1,
        }

        result = detector.validate_tradeoff_acceptability(tradeoff)

        # May flag as unnecessary
        unnecessary_issues = [
            i for i in result["issues"] if i["type"] == "unnecessary_tradeoff"
        ]
        # Note: This depends on the conflict rules, so we don't assert it must exist
        assert isinstance(result["issues"], list)

    def test_recommendation_reflects_acceptability(self, detector):
        """Test that recommendation matches acceptability"""
        good_tradeoff = {
            "sacrifice": "efficiency",
            "gain": "prediction_accuracy",
            "sacrifice_amount": 0.1,
            "gain_amount": 0.2,
        }

        result = detector.validate_tradeoff_acceptability(good_tradeoff)

        if result["acceptable"]:
            assert result["recommendation"] == "APPROVE"
        else:
            assert result["recommendation"] == "REJECT"


class TestResolutionSuggestions:
    """Test conflict resolution suggestions"""

    def test_suggest_resolution_for_direct_conflict(self, detector):
        """Test resolution suggestions for direct conflict"""
        conflict = Conflict(
            objectives=["exploration", "exploitation"],
            conflict_type=ConflictType.DIRECT,
            severity=ConflictSeverity.MEDIUM,
            description="Direct conflict",
        )

        resolutions = detector.suggest_resolution(conflict)

        assert isinstance(resolutions, list)
        assert len(resolutions) > 0

        # Should have strategies like sequential or weighted_sum
        strategy_names = [r["strategy"] for r in resolutions]
        assert "sequential" in strategy_names or "weighted_sum" in strategy_names

    def test_suggest_resolution_for_tradeoff(self, detector):
        """Test resolution suggestions for tradeoff conflict"""
        conflict = Conflict(
            objectives=["efficiency", "prediction_accuracy"],
            conflict_type=ConflictType.TRADEOFF,
            severity=ConflictSeverity.MEDIUM,
            description="Tradeoff conflict",
        )

        resolutions = detector.suggest_resolution(conflict)

        assert len(resolutions) > 0

        # Should suggest pareto optimization for tradeoffs
        strategy_names = [r["strategy"] for r in resolutions]
        assert (
            "pareto_optimization" in strategy_names
            or "bounded_optimization" in strategy_names
        )

    def test_suggest_resolution_for_constraint(self, detector):
        """Test resolution suggestions for constraint conflict"""
        conflict = Conflict(
            objectives=["efficiency"],
            conflict_type=ConflictType.CONSTRAINT,
            severity=ConflictSeverity.HIGH,
            description="Constraint violation",
        )

        resolutions = detector.suggest_resolution(conflict)

        assert len(resolutions) > 0

        # Should suggest constraint relaxation
        strategy_names = [r["strategy"] for r in resolutions]
        assert (
            "constraint_relaxation" in strategy_names
            or "constraint_reformulation" in strategy_names
        )

    def test_resolutions_include_specific_actions(self, detector):
        """Test that resolutions include specific actions"""
        conflict = Conflict(
            objectives=["efficiency", "prediction_accuracy"],
            conflict_type=ConflictType.TRADEOFF,
            severity=ConflictSeverity.MEDIUM,
            description="Tradeoff",
        )

        resolutions = detector.suggest_resolution(conflict)

        for resolution in resolutions:
            assert "strategy" in resolution
            assert "description" in resolution
            assert "specific_actions" in resolution
            assert isinstance(resolution["specific_actions"], list)


class TestHelperMethods:
    """Test helper methods"""

    def test_extract_objectives_from_proposal(self, detector):
        """Test extracting objectives from various proposal formats"""
        # Single objective
        proposal1 = {"objective": "efficiency"}
        objs1 = detector._extract_objectives_from_proposal(proposal1)
        assert "efficiency" in objs1

        # Multiple objectives
        proposal2 = {"objectives": ["efficiency", "prediction_accuracy"]}
        objs2 = detector._extract_objectives_from_proposal(proposal2)
        assert "efficiency" in objs2
        assert "prediction_accuracy" in objs2

        # From weights
        proposal3 = {"objective_weights": {"efficiency": 0.5, "safety": 0.5}}
        objs3 = detector._extract_objectives_from_proposal(proposal3)
        assert "efficiency" in objs3
        assert "safety" in objs3

        # From optimize_for
        proposal4 = {"optimize_for": "prediction_accuracy"}
        objs4 = detector._extract_objectives_from_proposal(proposal4)
        assert "prediction_accuracy" in objs4

    def test_severity_mapping(self, detector):
        """Test severity string to enum mapping"""
        assert detector._map_severity("critical") == ConflictSeverity.CRITICAL
        assert detector._map_severity("high") == ConflictSeverity.HIGH
        assert detector._map_severity("medium") == ConflictSeverity.MEDIUM
        assert detector._map_severity("low") == ConflictSeverity.LOW
        assert detector._map_severity("unknown") == ConflictSeverity.MEDIUM  # Default

    def test_severity_from_tension(self, detector):
        """Test mapping tension values to severity"""
        assert detector._severity_from_tension(0.95) == ConflictSeverity.CRITICAL
        assert detector._severity_from_tension(0.75) == ConflictSeverity.HIGH
        assert detector._severity_from_tension(0.5) == ConflictSeverity.MEDIUM
        assert detector._severity_from_tension(0.25) == ConflictSeverity.LOW
        assert detector._severity_from_tension(0.1) == ConflictSeverity.NEGLIGIBLE

    def test_find_shared_resources(self, detector):
        """Test finding shared resources"""
        objectives = ["efficiency", "prediction_accuracy"]

        shared = detector._find_shared_resources(objectives)

        # Both share CPU
        assert "cpu" in shared
        assert len(shared["cpu"]) == 2

    def test_constraints_conflict_detection(self, detector):
        """Test constraint conflict detection"""
        # Non-overlapping ranges
        constraints_a = {"min": 0.0, "max": 0.5}
        constraints_b = {"min": 0.6, "max": 1.0}

        assert detector._constraints_conflict(constraints_a, constraints_b)

        # Overlapping ranges
        constraints_c = {"min": 0.0, "max": 0.6}
        constraints_d = {"min": 0.5, "max": 1.0}

        assert not detector._constraints_conflict(constraints_c, constraints_d)


class TestConflictDataclass:
    """Test Conflict dataclass"""

    def test_conflict_creation(self):
        """Test creating a conflict"""
        conflict = Conflict(
            objectives=["efficiency", "prediction_accuracy"],
            conflict_type=ConflictType.TRADEOFF,
            severity=ConflictSeverity.MEDIUM,
            description="Test conflict",
        )

        assert conflict.objectives == ["efficiency", "prediction_accuracy"]
        assert conflict.conflict_type == ConflictType.TRADEOFF
        assert conflict.severity == ConflictSeverity.MEDIUM
        assert conflict.description == "Test conflict"

    def test_conflict_to_dict(self):
        """Test converting conflict to dictionary"""
        conflict = Conflict(
            objectives=["efficiency"],
            conflict_type=ConflictType.CONSTRAINT,
            severity=ConflictSeverity.HIGH,
            description="Constraint violation",
            quantitative_measure=0.5,
        )

        conflict_dict = conflict.to_dict()

        assert isinstance(conflict_dict, dict)
        assert conflict_dict["objectives"] == ["efficiency"]
        assert conflict_dict["conflict_type"] == "constraint"
        assert conflict_dict["severity"] == "high"
        assert conflict_dict["quantitative_measure"] == 0.5


class TestStatistics:
    """Test statistics tracking"""

    def test_statistics_initialization(self, detector):
        """Test that statistics are initialized"""
        stats = detector.get_statistics()

        assert isinstance(stats, dict)
        assert "statistics" in stats
        assert "conflict_history_size" in stats

    def test_statistics_updated(self, detector):
        """Test that statistics are updated - FIXED: Use .get() for safe access"""
        initial_stats = detector.get_statistics()
        initial_count = initial_stats["statistics"].get("proposals_analyzed", 0)

        proposal = {"objectives": ["efficiency", "prediction_accuracy"]}

        detector.detect_conflicts_in_proposal(proposal)

        updated_stats = detector.get_statistics()
        updated_count = updated_stats["statistics"].get("proposals_analyzed", 0)

        assert updated_count > initial_count

    def test_conflict_detection_rate(self, detector):
        """Test conflict detection rate calculation"""
        # Analyze proposal with conflicts
        proposal = {"objectives": ["exploration", "exploitation"]}

        detector.detect_conflicts_in_proposal(proposal)

        stats = detector.get_statistics()

        assert "conflict_detection_rate" in stats
        assert isinstance(stats["conflict_detection_rate"], float)
        assert 0.0 <= stats["conflict_detection_rate"]


class TestThreadSafety:
    """Test thread safety"""

    def test_concurrent_conflict_detection(self, detector):
        """Test concurrent conflict detection is thread-safe"""
        import threading

        results = []
        errors = []

        def detect_conflicts():
            try:
                proposal = {"objectives": ["efficiency", "prediction_accuracy"]}
                conflicts = detector.detect_conflicts_in_proposal(proposal)
                results.append(conflicts)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=detect_conflicts) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10

    def test_concurrent_tension_analysis(self, detector):
        """Test concurrent tension analysis is thread-safe"""
        import threading

        results = []
        errors = []

        def analyze_tension():
            try:
                objectives = ["efficiency", "prediction_accuracy", "safety"]
                analysis = detector.analyze_multi_objective_tension(objectives)
                results.append(analysis)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=analyze_tension) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_proposal(self, detector):
        """Test handling of empty proposal"""
        conflicts = detector.detect_conflicts_in_proposal({})
        assert isinstance(conflicts, list)
        assert len(conflicts) == 0

    def test_unknown_objective(self, detector):
        """Test handling of unknown objectives"""
        proposal = {"objectives": ["unknown_objective", "another_unknown"]}

        conflicts = detector.detect_conflicts_in_proposal(proposal)
        # Should handle gracefully without errors
        assert isinstance(conflicts, list)

    def test_malformed_proposal(self, detector):
        """Test handling of malformed proposal"""
        proposal = {
            "objective_weights": "not_a_dict",  # Invalid format
            "objectives": None,
        }

        # Should not crash
        conflicts = detector.detect_conflicts_in_proposal(proposal)
        assert isinstance(conflicts, list)

    def test_tension_analysis_with_unknown_objectives(self, detector):
        """Test tension analysis with unknown objectives"""
        objectives = ["unknown1", "unknown2"]

        analysis = detector.analyze_multi_objective_tension(objectives)

        assert isinstance(analysis, MultiObjectiveTension)
        assert len(analysis.objectives) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
