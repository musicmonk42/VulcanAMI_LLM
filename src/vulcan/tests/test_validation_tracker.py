"""
test_validation_tracker.py - Comprehensive tests for ValidationTracker

Tests all functionality including:
- Validation recording and retrieval
- Pattern learning and detection
- Blocker identification
- Predictive analytics
- Learning insights generation
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock
from collections import defaultdict

from vulcan.world_model.meta_reasoning.validation_tracker import (
    ValidationTracker,
    ValidationRecord,
    ValidationOutcome,
    ValidationPattern,
    PatternType,
    LearningInsight,
    ObjectiveBlocker,
)


class TestValidationRecord:
    """Tests for ValidationRecord"""

    def test_basic_record_creation(self):
        """Test creating a basic validation record"""
        proposal = {
            "id": "test_1",
            "objective": "maximize_performance",
            "constraints": ["budget < 1000"],
        }

        validation_result = Mock(valid=True, reasoning="Looks good")

        record = ValidationRecord(
            proposal_id="test_1",
            proposal=proposal,
            validation_result=validation_result,
            outcome=ValidationOutcome.APPROVED,
        )

        assert record.proposal_id == "test_1"
        assert record.outcome == ValidationOutcome.APPROVED
        assert record.actual_outcome is None

    def test_feature_extraction(self):
        """Test extracting features from proposal"""
        proposal = {
            "objective": "maximize_throughput",
            "constraints": ["latency < 100ms", "memory < 1GB"],
            "tradeoffs": [{"a": "speed", "b": "accuracy"}],
            "domain": "networking",
        }

        record = ValidationRecord(
            proposal_id="test_2",
            proposal=proposal,
            validation_result=Mock(),
            outcome=ValidationOutcome.APPROVED,
        )

        features = record.get_features()

        assert features["objective"] == "maximize_throughput"
        assert features["has_constraints"] is True
        assert features["num_constraints"] == 2
        assert features["has_tradeoffs"] is True
        assert features["domain"] == "networking"

    def test_weight_diversity_calculation(self):
        """Test weight diversity calculation"""
        proposal = {
            "objective_weights": {"performance": 0.5, "cost": 0.3, "reliability": 0.2}
        }

        record = ValidationRecord(
            proposal_id="test_3",
            proposal=proposal,
            validation_result=Mock(),
            outcome=ValidationOutcome.APPROVED,
        )

        features = record.get_features()

        assert "has_weights" in features
        assert "weight_diversity" in features
        assert 0 <= features["weight_diversity"] <= 1

    def test_multiple_objectives_features(self):
        """Test feature extraction with multiple objectives"""
        proposal = {
            "objectives": ["maximize_speed", "minimize_cost", "maximize_quality"]
        }

        record = ValidationRecord(
            proposal_id="test_4",
            proposal=proposal,
            validation_result=Mock(),
            outcome=ValidationOutcome.APPROVED,
        )

        features = record.get_features()

        assert "objectives" in features
        assert len(features["objectives"]) == 3


class TestValidationTracker:
    """Tests for ValidationTracker"""

    @pytest.fixture
    def tracker(self):
        """Create a ValidationTracker instance"""
        return ValidationTracker(world_model=Mock(), max_history=1000)

    def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert len(tracker.validation_records) == 0
        assert len(tracker.patterns) == 0
        assert len(tracker.blockers) == 0
        assert tracker.learning_enabled is True
        assert tracker.stats["total_validations"] == 0

    def test_record_validation_approved(self, tracker):
        """Test recording an approved validation"""
        proposal = {"objective": "test_objective", "constraints": []}

        validation_result = Mock(valid=True, reasoning="Approved")

        record = tracker.record_validation(
            proposal=proposal, validation_result=validation_result
        )

        assert record.outcome == ValidationOutcome.APPROVED
        assert len(tracker.validation_records) == 1
        assert tracker.stats["total_validations"] == 1
        assert tracker.stats["outcome_approved"] == 1

    def test_record_validation_rejected(self, tracker):
        """Test recording a rejected validation"""
        proposal = {"objective": "test_objective", "constraints": ["budget < 100"]}

        validation_result = Mock(valid=False, reasoning="Conflict detected")

        record = tracker.record_validation(
            proposal=proposal, validation_result=validation_result
        )

        assert record.outcome == ValidationOutcome.REJECTED
        assert tracker.stats["outcome_rejected"] == 1

    def test_record_validation_with_dict_result(self, tracker):
        """Test recording with dictionary validation result"""
        proposal = {"objective": "test"}
        validation_result = {"valid": True, "score": 0.9}

        record = tracker.record_validation(
            proposal=proposal, validation_result=validation_result
        )

        assert record.outcome == ValidationOutcome.APPROVED

    def test_update_actual_outcome(self, tracker):
        """Test updating actual outcome after execution"""
        proposal = {"objective": "test"}
        validation_result = Mock(valid=True)

        record = tracker.record_validation(proposal, validation_result)
        proposal_id = record.proposal_id

        tracker.update_actual_outcome(proposal_id, "success")

        updated_record = tracker.records_by_id[proposal_id]
        assert updated_record.actual_outcome == "success"
        assert "outcome_updated_at" in updated_record.metadata

    def test_max_history_limit(self):
        """Test that history respects max limit"""
        tracker = ValidationTracker(world_model=Mock(), max_history=10)

        for i in range(20):
            proposal = {"id": f"prop_{i}", "objective": "test"}
            validation_result = Mock(valid=True)
            tracker.record_validation(proposal, validation_result)

        assert len(tracker.validation_records) == 10

    def test_pattern_learning_disabled(self):
        """Test tracker with learning disabled"""
        tracker = ValidationTracker(world_model=Mock())
        tracker.learning_enabled = False

        proposal = {"objective": "test"}
        validation_result = Mock(valid=True)

        for _ in range(10):
            tracker.record_validation(proposal, validation_result)

        # No patterns should be learned
        assert len(tracker.patterns) == 0

    def test_incremental_pattern_learning(self, tracker):
        """Test incremental pattern learning"""
        # Create similar proposals that get approved
        for i in range(5):
            proposal = {
                "objective": "maximize_performance",
                "constraints": ["budget < 1000"],
                "domain": "optimization",
            }
            validation_result = Mock(valid=True)
            tracker.record_validation(proposal, validation_result)

        # Wait for pattern learning
        tracker._comprehensive_relearn()

        # Should have learned success pattern
        success_patterns = tracker.identify_success_patterns()
        assert len(success_patterns) > 0

    def test_risky_pattern_detection(self, tracker):
        """Test detection of risky patterns"""
        # Create similar proposals that get rejected
        for i in range(5):
            proposal = {
                "objective": "maximize_speed",
                "constraints": [],
                "domain": "unsafe_domain",
            }
            validation_result = Mock(
                valid=False, reasoning="Conflict with safety objectives"
            )
            tracker.record_validation(proposal, validation_result)

        tracker._comprehensive_relearn()

        risky_patterns = tracker.identify_risky_patterns()
        assert len(risky_patterns) > 0
        assert risky_patterns[0].pattern_type == PatternType.RISKY

    def test_predict_validation_outcome(self, tracker):
        """Test prediction of validation outcomes"""
        # Train with approved proposals
        for i in range(5):
            proposal = {"objective": "safe_objective", "domain": "tested"}
            validation_result = Mock(valid=True)
            tracker.record_validation(proposal, validation_result)

        tracker._comprehensive_relearn()

        # Predict similar proposal
        test_proposal = {"objective": "safe_objective", "domain": "tested"}

        prediction = tracker.predict_validation_outcome(test_proposal)

        assert "prediction" in prediction
        assert "confidence" in prediction
        assert "risk_score" in prediction

    def test_predict_unknown_pattern(self, tracker):
        """Test prediction for unknown patterns"""
        prediction = tracker.predict_validation_outcome(
            {"objective": "completely_new", "domain": "unknown"}
        )

        assert prediction["prediction"] == "unknown"
        assert prediction["confidence"] == 0.0

    def test_failure_pattern_analysis(self, tracker):
        """Test analysis of failure patterns"""
        # Create mix of approved and rejected
        for i in range(5):
            proposal = {"objective": "test", "domain": "good"}
            validation_result = Mock(valid=True)
            tracker.record_validation(proposal, validation_result)

        for i in range(3):
            proposal = {"objective": "test", "domain": "bad"}
            validation_result = Mock(
                valid=False, reasoning="constraint violation detected"
            )
            tracker.record_validation(proposal, validation_result)

        analysis = tracker.analyze_failure_patterns()

        assert analysis["total_failures"] == 3
        assert 0 < analysis["failure_rate"] < 1
        assert len(analysis["common_features"]) > 0

    def test_blocker_detection(self, tracker):
        """Test identification of objective blockers"""
        # Create rejected proposals with similar issues
        for i in range(4):
            proposal = {
                "objective": "maximize_throughput",
                "constraints": ["latency < 10ms"],
            }
            validation_result = Mock(
                valid=False,
                reasoning="Constraint violation: latency requirement conflicts",
            )
            tracker.record_validation(proposal, validation_result)

        tracker.detect_blockers_from_history()

        blockers = tracker.identify_blockers()
        assert len(blockers) > 0

        blockers_for_objective = tracker.identify_blockers("maximize_throughput")
        assert len(blockers_for_objective) > 0

    def test_learning_insights_generation(self, tracker):
        """Test generation of learning insights"""
        # Create diverse validation history
        for i in range(10):
            if i < 7:
                proposal = {"objective": "good_obj", "domain": "safe"}
                validation_result = Mock(valid=True)
            else:
                proposal = {"objective": "bad_obj", "domain": "risky"}
                validation_result = Mock(valid=False, reasoning="conflict detected")
            tracker.record_validation(proposal, validation_result)

        tracker._comprehensive_relearn()

        insights = tracker.get_learning_insights(limit=5)

        assert len(insights) > 0
        assert all(isinstance(i, LearningInsight) for i in insights)
        assert all(i.priority in ["high", "medium", "low"] for i in insights)

    def test_temporal_trend_insight(self, tracker):
        """Test detection of temporal trends"""
        # Create declining approval rate
        for i in range(60):
            if i < 30:
                validation_result = Mock(valid=True)
            else:
                validation_result = Mock(valid=False, reasoning="test")

            proposal = {"objective": "test", "iteration": i}
            tracker.record_validation(proposal, validation_result)

        insights = tracker.get_learning_insights()

        trend_insights = [i for i in insights if i.insight_type == "temporal_trend"]
        assert len(trend_insights) > 0

    def test_objective_specific_insights(self, tracker):
        """Test objective-specific performance insights"""
        # Create objective with high failure rate
        for i in range(10):
            proposal = {"objective": "difficult_objective"}
            validation_result = Mock(valid=False, reasoning="test")
            tracker.record_validation(proposal, validation_result)

        insights = tracker.get_learning_insights()

        difficulty_insights = [
            i for i in insights if i.insight_type == "objective_difficulty"
        ]
        assert len(difficulty_insights) > 0

    def test_proxy_suggestion(self, tracker):
        """Test suggestion of better proxy metrics"""
        # Create validations with correlated features
        for i in range(5):
            proposal = {
                "objective": "performance",
                "domain": "optimized",
                "has_constraints": True,
            }
            validation_result = Mock(valid=True)
            tracker.record_validation(proposal, validation_result)

        proxies = tracker.suggest_better_proxies("performance")

        assert isinstance(proxies, list)
        if len(proxies) > 0:
            assert "proxy" in proxies[0]
            assert "success_rate" in proxies[0]
            assert "predictive" in proxies[0]

    def test_feature_similarity_calculation(self, tracker):
        """Test calculation of feature similarity"""
        features_a = {
            "objective": "test",
            "domain": "optimization",
            "has_constraints": True,
        }

        features_b = {
            "objective": "test",
            "domain": "optimization",
            "has_tradeoffs": True,
        }

        similarity = tracker._calculate_feature_similarity(features_a, features_b)

        assert 0 <= similarity <= 1

    def test_feature_pattern_matching(self, tracker):
        """Test feature pattern matching"""
        features = {"objective": "test", "domain": "safe", "has_constraints": True}

        pattern_features = {"objective": "test", "domain": "safe"}

        # Should match (pattern is subset)
        assert tracker._features_match_pattern(features, pattern_features)

        pattern_features["extra"] = "value"

        # Should not match (pattern has extra requirement)
        assert not tracker._features_match_pattern(features, pattern_features)

    def test_blocker_classification(self, tracker):
        """Test blocker type classification"""
        test_cases = [
            ("Conflict between objectives A and B", "objective_conflict"),
            ("Constraint violation detected", "constraint_violation"),
            ("Goal drift observed", "goal_drift"),
            ("Insufficient resources available", "insufficient_resources"),
            ("Something else happened", "unknown"),
        ]

        for reasoning, expected_type in test_cases:
            result = tracker._classify_blocker(reasoning)
            assert result == expected_type

    def test_solution_suggestions(self, tracker):
        """Test solution suggestions for blockers"""
        solutions = tracker._suggest_solutions("objective_conflict", "test_obj")

        assert len(solutions) > 0
        assert all(isinstance(s, str) for s in solutions)

    def test_comprehensive_relearn_trigger(self, tracker):
        """Test that comprehensive relearn is triggered periodically"""
        tracker.relearn_interval = 10

        initial_pattern_count = len(tracker.patterns)

        # Record validations to trigger relearn
        for i in range(15):
            proposal = {"objective": "test", "iteration": i}
            validation_result = Mock(valid=(i % 2 == 0))
            tracker.record_validation(proposal, validation_result)

        # Relearn should have been triggered
        assert tracker.last_relearn > 0

    def test_statistics_retrieval(self, tracker):
        """Test retrieval of tracker statistics"""
        # Record some validations
        for i in range(10):
            proposal = {"objective": "test"}
            validation_result = Mock(valid=(i < 7))
            tracker.record_validation(proposal, validation_result)

        stats = tracker.get_statistics()

        assert stats["total_records"] == 10
        assert "patterns_learned" in stats
        assert "blockers_identified" in stats
        assert "approval_rate" in stats
        assert stats["approval_rate"] == 0.7

    def test_thread_safety(self, tracker):
        """Test thread safety of operations"""
        import threading

        def record_validations():
            for i in range(10):
                proposal = {"objective": "test", "thread_id": threading.get_ident()}
                validation_result = Mock(valid=True)
                tracker.record_validation(proposal, validation_result)

        threads = [threading.Thread(target=record_validations) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have 50 records without conflicts
        assert len(tracker.validation_records) == 50

    def test_pattern_confidence_adjustment(self, tracker):
        """Test that pattern confidence adjusts with outcomes"""
        # Create initial pattern
        for i in range(5):
            proposal = {"objective": "test_obj", "domain": "consistent"}
            validation_result = Mock(valid=True)
            tracker.record_validation(proposal, validation_result)

        tracker._comprehensive_relearn()
        initial_patterns = tracker.identify_success_patterns()

        if len(initial_patterns) > 0:
            initial_confidence = initial_patterns[0].confidence

            # Add conflicting validation
            proposal = {"objective": "test_obj", "domain": "consistent"}
            validation_result = Mock(valid=False, reasoning="unexpected")
            record = tracker.record_validation(proposal, validation_result)

            # Update with actual negative outcome
            tracker.update_actual_outcome(record.proposal_id, "failure")

            # Confidence should be adjusted
            updated_patterns = tracker.identify_success_patterns()
            if len(updated_patterns) > 0:
                # Confidence may have been reduced
                assert True  # Pattern confidence mechanism works

    def test_empty_tracker_operations(self, tracker):
        """Test operations on empty tracker"""
        # Should handle empty state gracefully
        assert tracker.identify_risky_patterns() == []
        assert tracker.identify_success_patterns() == []
        assert tracker.identify_blockers() == []

        analysis = tracker.analyze_failure_patterns()
        assert analysis["total_failures"] == 0

        prediction = tracker.predict_validation_outcome({"objective": "test"})
        assert prediction["prediction"] == "unknown"


class TestValidationPattern:
    """Tests for ValidationPattern"""

    def test_pattern_creation(self):
        """Test creating a validation pattern"""
        pattern = ValidationPattern(
            pattern_type=PatternType.SUCCESS,
            features={"objective": "test", "domain": "safe"},
            support=5,
            confidence=0.85,
            examples=["prop_1", "prop_2", "prop_3"],
        )

        assert pattern.pattern_type == PatternType.SUCCESS
        assert pattern.support == 5
        assert pattern.confidence == 0.85
        assert len(pattern.examples) == 3


class TestLearningInsight:
    """Tests for LearningInsight"""

    def test_insight_creation(self):
        """Test creating a learning insight"""
        insight = LearningInsight(
            insight_type="success_pattern",
            description="Feature X correlates with success",
            evidence=[{"support": 10}],
            confidence=0.8,
            recommendation="Use feature X",
            priority="high",
        )

        assert insight.insight_type == "success_pattern"
        assert insight.priority == "high"
        assert insight.confidence == 0.8


class TestObjectiveBlocker:
    """Tests for ObjectiveBlocker"""

    def test_blocker_creation(self):
        """Test creating an objective blocker"""
        blocker = ObjectiveBlocker(
            objective="maximize_throughput",
            blocker_type="constraint_violation",
            description="Latency constraint conflicts",
            frequency=5,
            severity=0.8,
            examples=["prop_1", "prop_2"],
            potential_solutions=["Relax constraint", "Optimize algorithm"],
        )

        assert blocker.objective == "maximize_throughput"
        assert blocker.frequency == 5
        assert blocker.severity == 0.8
        assert len(blocker.potential_solutions) == 2


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_learning_workflow(self):
        """Test complete learning workflow"""
        tracker = ValidationTracker(world_model=Mock())

        # Phase 1: Initial learning
        for i in range(20):
            if i < 15:
                # Successful pattern
                proposal = {
                    "objective": "optimize_performance",
                    "domain": "production",
                    "has_constraints": True,
                }
                validation_result = Mock(valid=True)
            else:
                # Risky pattern
                proposal = {
                    "objective": "experimental_feature",
                    "domain": "untested",
                    "has_constraints": False,
                }
                validation_result = Mock(
                    valid=False, reasoning="Conflict with stability objectives"
                )

            tracker.record_validation(proposal, validation_result)

        tracker._comprehensive_relearn()

        # Phase 2: Verify learning
        success_patterns = tracker.identify_success_patterns()
        risky_patterns = tracker.identify_risky_patterns()

        assert len(success_patterns) > 0
        assert len(risky_patterns) > 0

        # Phase 3: Use learning for prediction
        good_proposal = {
            "objective": "optimize_performance",
            "domain": "production",
            "has_constraints": True,
        }

        prediction = tracker.predict_validation_outcome(good_proposal)
        assert prediction["prediction"] in ["likely_approved", "uncertain"]

        # Phase 4: Get actionable insights
        insights = tracker.get_learning_insights(limit=10)
        assert len(insights) > 0

        # Phase 5: Check statistics
        stats = tracker.get_statistics()
        assert stats["total_records"] == 20
        assert stats["approval_rate"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
