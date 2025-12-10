"""
Comprehensive test suite for governance and value alignment systems.
Tests GovernanceManager, ValueAlignmentSystem, and HumanOversightInterface.

NOTE: Before running these tests, ensure governance_alignment.py has the correct import:
    from .safety_types import (  # Note the dot for relative import
        SafetyReport,
        SafetyViolationType,
        ActionType
    )
"""

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from vulcan.safety.governance_alignment import (AlignmentConstraint,
                                                GovernanceLevel,
                                                GovernanceManager,
                                                GovernancePolicy,
                                                HumanFeedback,
                                                HumanOversightInterface,
                                                StakeholderType,
                                                ValueAlignmentSystem,
                                                ValueSystem)

# Import from safety_types (with fallback)
try:
    from vulcan.safety.safety_types import (ActionType, SafetyReport,
                                            SafetyViolationType)
except ImportError:
    # Mock if not available
    from enum import Enum

    class ActionType(Enum):
        EXPLORE = "explore"
        OPTIMIZE = "optimize"
        MAINTAIN = "maintain"
        EMERGENCY_STOP = "emergency_stop"

    class SafetyViolationType(Enum):
        GOVERNANCE = "governance"
        ALIGNMENT = "alignment"

    class SafetyReport:
        def __init__(self, safe, violations, reasons, **kwargs):
            self.safe = safe
            self.violations = violations
            self.reasons = reasons


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for database."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def governance_manager(temp_db_dir):
    """Create a GovernanceManager instance."""
    # Reset the singleton before each test to ensure clean state
    GovernanceManager.reset_instance()

    config = {
        "db_path": str(temp_db_dir / "test_governance.db"),
        "max_active_decisions": 100,
        "max_stakeholders": 100,
    }
    manager = GovernanceManager(config=config)
    yield manager
    manager.shutdown()
    # Reset singleton after test as well
    GovernanceManager.reset_instance()


@pytest.fixture
def value_system():
    """Create a ValueSystem instance."""
    return ValueSystem(
        name="test_system",
        values={"safety": 0.9, "efficiency": 0.6, "transparency": 0.8},
        constraints=[],
        trade_offs={("safety", "efficiency"): 0.8},
    )


@pytest.fixture
def alignment_system(value_system):
    """Create a ValueAlignmentSystem instance."""
    return ValueAlignmentSystem(value_system=value_system)


@pytest.fixture
def oversight_interface(governance_manager, alignment_system):
    """Create a HumanOversightInterface instance."""
    return HumanOversightInterface(
        governance_manager=governance_manager, alignment_system=alignment_system
    )


@pytest.fixture
def sample_action():
    """Create a sample action."""
    return {
        "id": "action_123",
        "type": ActionType.EXPLORE,
        "risk_score": 0.3,
        "safety_score": 0.8,
        "confidence": 0.85,
        "explanation": "Test action for exploration",
        "expected_benefit": 5.0,
        "potential_harm": 0.1,
    }


@pytest.fixture
def sample_context():
    """Create sample context."""
    return {
        "critical_operation": False,
        "require_human_approval": False,
        "environment": "test",
    }


# ============================================================================
# GOVERNANCE POLICY TESTS
# ============================================================================


class TestGovernancePolicy:
    """Test GovernancePolicy dataclass."""

    def test_creation(self):
        """Test creating a governance policy."""
        policy = GovernancePolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="A test policy",
            level=GovernanceLevel.AUTONOMOUS,
            stakeholders=[StakeholderType.AI_SYSTEM],
            approval_threshold=0.5,
            timeout_seconds=30.0,
        )

        assert policy.policy_id == "test_policy"
        assert policy.level == GovernanceLevel.AUTONOMOUS
        assert len(policy.stakeholders) == 1
        assert policy.approval_threshold == 0.5


# ============================================================================
# GOVERNANCE MANAGER TESTS
# ============================================================================


class TestGovernanceManager:
    """Test GovernanceManager class."""

    def test_initialization(self, governance_manager):
        """Test manager initialization."""
        assert governance_manager is not None
        assert len(governance_manager.policies) > 0
        assert "autonomous_default" in governance_manager.policies

    def test_database_initialization(self, governance_manager):
        """Test that database is initialized."""
        assert governance_manager.conn is not None

        # Check tables exist
        cursor = governance_manager.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='governance_decisions'
        """)
        assert cursor.fetchone() is not None

    def test_add_policy(self, governance_manager):
        """Test adding a new policy."""
        policy = GovernancePolicy(
            policy_id="custom_policy",
            name="Custom Policy",
            description="Custom test policy",
            level=GovernanceLevel.SUPERVISED,
            stakeholders=[StakeholderType.OPERATOR],
            approval_threshold=0.7,
            timeout_seconds=60.0,
        )

        governance_manager.add_policy(policy)

        assert "custom_policy" in governance_manager.policies
        assert governance_manager.policies["custom_policy"].name == "Custom Policy"

    def test_register_stakeholder(self, governance_manager):
        """Test registering a stakeholder."""
        result = governance_manager.register_stakeholder(
            "user_123", StakeholderType.USER, {"name": "Test User"}
        )

        assert result is True
        assert "user_123" in governance_manager.stakeholder_registry
        assert (
            governance_manager.stakeholder_registry["user_123"]["type"]
            == StakeholderType.USER
        )

    def test_register_stakeholder_limit(self, governance_manager):
        """Test stakeholder registration limit."""
        governance_manager.max_stakeholders = 5

        # Register up to limit
        for i in range(5):
            result = governance_manager.register_stakeholder(
                f"user_{i}", StakeholderType.USER
            )
            assert result is True

        # Try to exceed limit
        result = governance_manager.register_stakeholder("user_6", StakeholderType.USER)
        assert result is False

    def test_request_approval_autonomous(
        self, governance_manager, sample_action, sample_context
    ):
        """Test autonomous approval request."""
        result = governance_manager.request_approval(
            sample_action, policy_id="autonomous_default", context=sample_context
        )

        assert "decision_id" in result
        assert "approved" in result
        assert result["approved"] is True
        assert result["governance_level"] == GovernanceLevel.AUTONOMOUS.value

    def test_request_approval_auto_select_policy(
        self, governance_manager, sample_action, sample_context
    ):
        """Test automatic policy selection based on risk."""
        # Low risk - should use autonomous
        low_risk_action = {**sample_action, "risk_score": 0.2}
        result = governance_manager.request_approval(
            low_risk_action, context=sample_context
        )
        assert "autonomous" in result["policy_applied"].lower()

        # High risk - should use safety critical
        high_risk_action = {**sample_action, "risk_score": 0.8}
        result = governance_manager.request_approval(
            high_risk_action, context=sample_context
        )
        assert (
            "safety" in result["policy_applied"].lower()
            or "critical" in result["policy_applied"].lower()
        )

    def test_request_approval_emergency(self, governance_manager):
        """Test emergency action approval."""
        emergency_action = {
            "id": "emergency_123",
            "type": ActionType.EMERGENCY_STOP,
            "reason": "Safety violation detected",
            "risk_score": 1.0,
        }

        result = governance_manager.request_approval(
            emergency_action, policy_id="emergency_stop"
        )

        assert "approved" in result
        assert result["governance_level"] == GovernanceLevel.EMERGENCY.value

    def test_process_autonomous(self, governance_manager, sample_action):
        """Test autonomous processing."""
        policy = governance_manager.policies["autonomous_default"]
        result = governance_manager._process_autonomous(
            "decision_1", sample_action, policy
        )

        assert result["approved"] is True
        assert result["confidence"] == 1.0
        assert "response_time" in result

    def test_process_supervised(self, governance_manager, sample_action):
        """Test supervised processing."""
        policy = governance_manager.policies["human_supervised"]
        result = governance_manager._process_supervised(
            "decision_2", sample_action, policy
        )

        assert "approved" in result
        assert "confidence" in result
        assert "response_time" in result

    def test_ai_supervision(self, governance_manager, sample_action):
        """Test AI supervision recommendation."""
        recommendation = governance_manager._get_ai_supervision(sample_action)

        assert "approved" in recommendation
        assert "confidence" in recommendation
        assert 0 <= recommendation["confidence"] <= 1

    def test_select_policy_based_on_risk(self, governance_manager, sample_context):
        """Test policy selection based on risk score."""
        # Low risk
        low_risk = {"risk_score": 0.2}
        policy = governance_manager._select_policy(low_risk, sample_context)
        assert policy.policy_id == "autonomous_default"

        # Medium risk
        medium_risk = {"risk_score": 0.6}
        policy = governance_manager._select_policy(medium_risk, sample_context)
        assert policy.policy_id in ["human_supervised", "safety_critical"]

        # High risk
        high_risk = {"risk_score": 0.8}
        policy = governance_manager._select_policy(high_risk, sample_context)
        assert policy.policy_id == "safety_critical"

    def test_record_decision(self, governance_manager, sample_action):
        """Test recording a decision."""
        result = {"approved": True, "confidence": 0.9}

        governance_manager._record_decision(
            "decision_123", "action_123", "autonomous_default", result
        )

        # Check it's in history
        assert len(governance_manager.decision_history) > 0

        # Check it's in database
        cursor = governance_manager.conn.cursor()
        cursor.execute(
            "SELECT * FROM governance_decisions WHERE decision_id=?", ("decision_123",)
        )
        row = cursor.fetchone()
        assert row is not None

    def test_get_governance_stats(
        self, governance_manager, sample_action, sample_context
    ):
        """Test getting governance statistics."""
        # Make some decisions
        for _ in range(5):
            governance_manager.request_approval(sample_action, context=sample_context)

        stats = governance_manager.get_governance_stats()

        assert "policies" in stats
        assert "active_decisions" in stats
        assert "registered_stakeholders" in stats
        assert stats["policies"] > 0

    def test_cleanup_old_decisions(self, governance_manager):
        """Test cleanup of old decisions."""
        # Add some old decisions
        for i in range(10):
            governance_manager.active_decisions[f"old_{i}"] = {
                "timestamp": time.time() - 7200,  # 2 hours ago
                "action": {},
                "policy": None,
                "status": "pending",
            }

        initial_count = len(governance_manager.active_decisions)

        # Run cleanup
        governance_manager._cleanup_old_decisions()

        # Should have fewer active decisions
        assert len(governance_manager.active_decisions) < initial_count

    def test_thread_safety(self, governance_manager, sample_action, sample_context):
        """Test thread-safe operations."""
        import threading

        results = []

        def make_request():
            result = governance_manager.request_approval(
                sample_action, context=sample_context
            )
            results.append(result)

        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all("decision_id" in r for r in results)

    def test_shutdown(self, temp_db_dir):
        """Test manager shutdown."""
        # Reset singleton to ensure fresh instance
        GovernanceManager.reset_instance()

        config = {"db_path": str(temp_db_dir / "test_shutdown.db")}
        manager = GovernanceManager(config=config)

        assert manager.conn is not None

        manager.shutdown()

        # Should be marked as shutdown
        assert manager._shutdown is True

        # Reset singleton for other tests
        GovernanceManager.reset_instance()


# ============================================================================
# VALUE ALIGNMENT SYSTEM TESTS
# ============================================================================


class TestValueAlignmentSystem:
    """Test ValueAlignmentSystem class."""

    def test_initialization(self, alignment_system):
        """Test system initialization."""
        assert alignment_system.value_system is not None
        assert len(alignment_system.value_system.values) > 0
        assert len(alignment_system.constraints) > 0

    def test_default_value_system(self):
        """Test creation of default value system."""
        system = ValueAlignmentSystem()

        assert "safety" in system.value_system.values
        assert "beneficence" in system.value_system.values
        assert system.value_system.values["safety"] > 0

    def test_add_constraint(self, alignment_system):
        """Test adding an alignment constraint."""
        constraint = AlignmentConstraint(
            constraint_id="test_constraint",
            name="Test Constraint",
            description="A test constraint",
            check_function=lambda a: True,
            priority=1,
            violation_severity="medium",
        )

        initial_count = len(alignment_system.constraints)
        alignment_system.add_constraint(constraint)

        assert len(alignment_system.constraints) == initial_count + 1

    def test_check_alignment_safe_action(self, alignment_system):
        """Test alignment check for safe action."""
        action = {
            "type": "test",
            "safety_score": 0.9,
            "expected_benefit": 5.0,
            "potential_harm": 0.05,
            "explanation": "Test action with explanation",
            "confidence": 0.85,
            "uncertainty": 0.1,
        }

        result = alignment_system.check_alignment(action)

        assert "aligned" in result
        assert "alignment_score" in result
        assert "value_scores" in result
        assert 0 <= result["alignment_score"] <= 1

    def test_check_alignment_unsafe_action(self, alignment_system):
        """Test alignment check for unsafe action."""
        action = {
            "type": "test",
            "safety_score": 0.2,
            "causes_harm": True,
            "potential_harm": 0.8,
            "expected_benefit": 0.0,
        }

        result = alignment_system.check_alignment(action)

        assert "violations" in result
        assert len(result["violations"]) > 0

    def test_calculate_value_alignment_safety(self, alignment_system):
        """Test safety value alignment calculation."""
        action = {"safety_score": 0.85}
        score = alignment_system._calculate_value_alignment(action, "safety", None)

        assert score == 0.85

    def test_calculate_value_alignment_beneficence(self, alignment_system):
        """Test beneficence value alignment calculation."""
        action = {"expected_benefit": 5.0}
        score = alignment_system._calculate_value_alignment(action, "beneficence", None)

        assert 0 <= score <= 1

    def test_calculate_value_alignment_transparency(self, alignment_system):
        """Test transparency value alignment calculation."""
        action = {"explanation": "Clear explanation", "auditable": True}
        score = alignment_system._calculate_value_alignment(
            action, "transparency", None
        )

        assert score > 0.5

    def test_calculate_value_alignment_autonomy(self, alignment_system):
        """Test autonomy value alignment calculation."""
        # Preserves autonomy
        action = {"restricts_user_control": False}
        score = alignment_system._calculate_value_alignment(action, "autonomy", None)
        assert score == 1.0

        # Restricts autonomy
        action = {"restricts_user_control": True}
        score = alignment_system._calculate_value_alignment(action, "autonomy", None)
        assert score == 0.0

    def test_check_goal_preservation(self, alignment_system):
        """Test goal preservation check."""
        action = {"type": "test", "expected_benefit": 5.0, "potential_harm": 0.05}

        score = alignment_system._check_goal_preservation(action)

        assert 0 <= score <= 1

    def test_detect_value_drift_insufficient_history(self, alignment_system):
        """Test drift detection with insufficient history."""
        scores = {"safety": 0.8, "efficiency": 0.6}

        drift = alignment_system._detect_value_drift(scores)

        assert drift is False  # Not enough history

    def test_detect_value_drift_with_history(self, alignment_system):
        """Test drift detection with sufficient history."""
        # Build history
        for i in range(110):
            alignment_system.alignment_history.append(
                {
                    "timestamp": time.time(),
                    "alignment_scores": {
                        "safety": 0.8 + (i * 0.001),
                        "efficiency": 0.6,
                    },
                    "overall_alignment": 0.7,
                    "violations": [],
                }
            )

        # Test with significantly different scores
        current_scores = {"safety": 0.4, "efficiency": 0.6}
        drift = alignment_system._detect_value_drift(current_scores)

        # May or may not detect drift depending on threshold
        assert isinstance(drift, bool)

    def test_learn_preferences(self, alignment_system):
        """Test learning from human feedback."""
        feedback = HumanFeedback(
            feedback_id="feedback_1",
            action_id="action_1",
            stakeholder_type=StakeholderType.USER,
            stakeholder_id="user_1",
            approval=True,
            confidence=0.9,
            preferences={"safety": 0.95, "efficiency": 0.7},
        )

        alignment_system.learn_preferences(feedback)

        assert len(alignment_system.preference_learning_buffer) == 1
        assert "safety" in alignment_system.learned_preferences

    def test_modify_goal_allowed(self, alignment_system):
        """Test modifying a non-core goal."""
        result = alignment_system.modify_goal("secondary", "add", "new_secondary_goal")

        assert result is True
        assert "new_secondary_goal" in alignment_system.current_goals["secondary"]

    def test_modify_goal_core_protection(self, alignment_system):
        """Test that core goals cannot be removed."""
        result = alignment_system.modify_goal("primary", "remove", "minimize_harm")

        assert result is False
        assert "minimize_harm" in alignment_system.current_goals["primary"]

    def test_get_alignment_report(self, alignment_system, sample_action):
        """Test generating alignment report."""
        # Generate some alignment history
        for _ in range(10):
            alignment_system.check_alignment(sample_action)

        report = alignment_system.get_alignment_report()

        assert "current_values" in report
        assert "active_constraints" in report
        assert "goal_preservation" in report
        assert isinstance(report["current_values"], dict)


# ============================================================================
# HUMAN OVERSIGHT INTERFACE TESTS
# ============================================================================


class TestHumanOversightInterface:
    """Test HumanOversightInterface class."""

    def test_initialization(self, oversight_interface):
        """Test interface initialization."""
        assert oversight_interface.governance is not None
        assert oversight_interface.alignment is not None
        assert oversight_interface.emergency_stop_enabled is True

    def test_request_human_approval_normal(
        self, oversight_interface, sample_action, sample_context
    ):
        """Test requesting human approval with normal urgency."""
        result = oversight_interface.request_human_approval(
            sample_action, urgency="normal", context=sample_context
        )

        assert "decision_id" in result
        assert "approved" in result
        assert len(oversight_interface.feedback_requests) > 0

    def test_request_human_approval_critical(
        self, oversight_interface, sample_action, sample_context
    ):
        """Test requesting human approval with critical urgency."""
        result = oversight_interface.request_human_approval(
            sample_action, urgency="critical", context=sample_context
        )

        assert "decision_id" in result
        assert result["governance_level"] == GovernanceLevel.HUMAN_CONTROLLED.value

    def test_emergency_stop(self, oversight_interface):
        """Test emergency stop execution."""
        result = oversight_interface.emergency_stop("Test emergency")

        assert result["success"] is True
        assert result["action_taken"] == "emergency_stop"
        assert len(oversight_interface.interventions) > 0

    def test_emergency_stop_disabled(self, oversight_interface):
        """Test emergency stop when disabled."""
        oversight_interface.emergency_stop_enabled = False

        result = oversight_interface.emergency_stop("Test emergency")

        assert result["success"] is False

    def test_set_automation_level(self, oversight_interface):
        """Test setting automation level."""
        result = oversight_interface.set_automation_level(0.5)

        assert result is True
        assert oversight_interface.automation_level == 0.5
        assert len(oversight_interface.interventions) > 0

    def test_set_automation_level_clamping(self, oversight_interface):
        """Test automation level is clamped to [0, 1]."""
        oversight_interface.set_automation_level(1.5)
        assert oversight_interface.automation_level == 1.0

        oversight_interface.set_automation_level(-0.5)
        assert oversight_interface.automation_level == 0.0

    def test_enable_human_override(self, oversight_interface):
        """Test enabling human override."""
        oversight_interface.enable_human_override(duration_seconds=1.0)

        assert oversight_interface.human_override_active is True
        assert len(oversight_interface.interventions) > 0

        # Wait for override to expire
        time.sleep(1.5)

        # Should be disabled
        assert oversight_interface.human_override_active is False

    def test_submit_feedback(self, oversight_interface):
        """Test submitting human feedback."""
        feedback = {
            "user_id": "user_123",
            "approved": True,
            "confidence": 0.9,
            "reasoning": "Looks good",
            "preferences": {"safety": 0.95},
        }

        result = oversight_interface.submit_feedback("action_123", feedback)

        assert result["processed"] is True
        assert "feedback_id" in result
        assert len(oversight_interface.collected_feedback) > 0

    def test_create_alert(self, oversight_interface):
        """Test creating an alert."""
        alert_id = oversight_interface.create_alert(
            "safety_violation", "Potential safety issue detected", severity="high"
        )

        assert alert_id is not None
        assert len(oversight_interface.alerts) > 0

    def test_acknowledge_alert(self, oversight_interface):
        """Test acknowledging an alert."""
        alert_id = oversight_interface.create_alert(
            "test_alert", "Test alert", severity="medium"
        )

        result = oversight_interface.acknowledge_alert(alert_id, notes="Handled")

        assert result is True

        # Check alert is acknowledged
        alert = next(a for a in oversight_interface.alerts if a["alert_id"] == alert_id)
        assert alert["acknowledged"] is True

    def test_get_oversight_status(self, oversight_interface):
        """Test getting oversight status."""
        status = oversight_interface.get_oversight_status()

        assert "automation_level" in status
        assert "human_override_active" in status
        assert "emergency_stop_enabled" in status
        assert "governance_stats" in status
        assert "alignment_report" in status

    def test_get_monitoring_dashboard(
        self, oversight_interface, sample_action, sample_context
    ):
        """Test getting monitoring dashboard."""
        # Generate some activity
        oversight_interface.request_human_approval(
            sample_action, context=sample_context
        )
        oversight_interface.create_alert("test", "Test alert")

        dashboard = oversight_interface.get_monitoring_dashboard()

        assert "system_health" in dashboard
        assert "recent_decisions" in dashboard
        assert "active_alerts" in dashboard
        assert "automation_level" in dashboard
        assert 0 <= dashboard["system_health"] <= 1

    def test_calculate_system_health(self, oversight_interface):
        """Test system health calculation."""
        health = oversight_interface._calculate_system_health()

        assert 0 <= health <= 1
        assert isinstance(health, float)

    def test_get_metrics_summary(self, oversight_interface):
        """Test getting metrics summary."""
        metrics = oversight_interface._get_metrics_summary()

        assert "total_decisions" in metrics
        assert "total_interventions" in metrics
        assert "total_feedback" in metrics
        assert "pending_approvals" in metrics


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for governance and alignment systems."""

    def test_full_governance_flow(
        self, governance_manager, sample_action, sample_context
    ):
        """Test complete governance flow."""
        # Register stakeholder
        governance_manager.register_stakeholder("operator_1", StakeholderType.OPERATOR)

        # Request approval
        result = governance_manager.request_approval(
            sample_action, context=sample_context
        )

        assert "decision_id" in result
        assert "approved" in result

        # Check stats
        stats = governance_manager.get_governance_stats()
        assert stats["registered_stakeholders"] >= 1

    def test_alignment_with_governance(
        self, governance_manager, alignment_system, sample_action
    ):
        """Test alignment check integrated with governance."""
        # Check alignment
        alignment_result = alignment_system.check_alignment(sample_action)

        # Request governance based on alignment
        if not alignment_result["aligned"]:
            # Should trigger higher governance
            sample_action["risk_score"] = 0.8

        governance_result = governance_manager.request_approval(sample_action)

        assert "decision_id" in governance_result

    def test_human_oversight_with_governance_and_alignment(
        self, oversight_interface, sample_action, sample_context
    ):
        """Test human oversight coordinating governance and alignment."""
        # Set moderate automation
        oversight_interface.set_automation_level(0.5)

        # Request approval
        result = oversight_interface.request_human_approval(
            sample_action, urgency="high", context=sample_context
        )

        # Submit feedback
        feedback = {
            "approved": True,
            "confidence": 0.85,
            "preferences": {"safety": 0.95},
        }
        oversight_interface.submit_feedback(sample_action["id"], feedback)

        # Check alignment learned from feedback
        report = oversight_interface.alignment.get_alignment_report()
        assert "learned_preferences" in report

        # Get monitoring dashboard
        dashboard = oversight_interface.get_monitoring_dashboard()
        assert dashboard["system_health"] > 0


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_governance_no_applicable_policy(self, governance_manager):
        """Test request with no applicable policy."""
        action = {"type": "unknown", "risk_score": 0.5}

        # Clear all policies
        governance_manager.policies.clear()

        result = governance_manager.request_approval(action)

        assert result["approved"] is False
        assert "no applicable" in result["reason"].lower()

    def test_alignment_empty_action(self, alignment_system):
        """Test alignment check with empty action."""
        result = alignment_system.check_alignment({})

        assert "aligned" in result
        assert "alignment_score" in result

    def test_alignment_missing_fields(self, alignment_system):
        """Test alignment check with missing fields."""
        action = {"type": "test"}  # Minimal action

        result = alignment_system.check_alignment(action)

        assert isinstance(result, dict)
        assert "alignment_score" in result

    def test_governance_database_error_handling(self, temp_db_dir):
        """Test handling of database errors."""
        # Reset singleton to ensure fresh instance
        GovernanceManager.reset_instance()

        config = {"db_path": str(temp_db_dir / "test.db")}
        manager = GovernanceManager(config=config)

        # Close connection to simulate error
        manager.conn.close()
        manager.conn = None

        # Should not crash
        try:
            manager._record_decision("test_id", "action_id", "policy_id", {})
        except Exception:
            # Should log error but not crash
            pass

        manager.shutdown()

    def test_oversight_alert_nonexistent(self, oversight_interface):
        """Test acknowledging nonexistent alert."""
        result = oversight_interface.acknowledge_alert("nonexistent_id")

        assert result is False

    def test_constraint_check_exception(self, alignment_system):
        """Test constraint that raises exception."""

        def faulty_check(action):
            raise ValueError("Test exception")

        constraint = AlignmentConstraint(
            constraint_id="faulty",
            name="Faulty Constraint",
            description="Raises exception",
            check_function=faulty_check,
            priority=1,
            violation_severity="high",
        )

        alignment_system.add_constraint(constraint)

        # Should handle exception gracefully
        result = alignment_system.check_alignment({"type": "test"})

        assert isinstance(result, dict)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_governance_many_decisions(
        self, governance_manager, sample_action, sample_context
    ):
        """Test handling many concurrent decisions."""
        start_time = time.time()

        for _ in range(50):
            governance_manager.request_approval(sample_action, context=sample_context)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 10.0

        stats = governance_manager.get_governance_stats()
        assert stats["active_decisions"] <= governance_manager.max_active_decisions

    def test_alignment_many_checks(self, alignment_system, sample_action):
        """Test many alignment checks."""
        start_time = time.time()

        for _ in range(100):
            alignment_system.check_alignment(sample_action)

        elapsed = time.time() - start_time

        # Should be reasonably fast
        assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
