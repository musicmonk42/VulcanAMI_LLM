"""
Comprehensive test suite for governance_loop.py
"""

import copy
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from governance_loop import (MAX_POLICIES, MAX_POLICY_ID_LENGTH,
                             MAX_POLICY_NAME_LENGTH, GovernanceLoop, Policy,
                             PolicyPriority, PolicyType, PolicyViolation)


@pytest.fixture
def governance():
    """Create governance loop."""
    loop = GovernanceLoop(
        check_interval_s=1,
        enable_auto_enforcement=True,
        enable_policy_learning=True
    )
    yield loop
    if loop.is_running:
        loop.stop()


@pytest.fixture
def sample_policy():
    """Create sample policy."""
    return Policy(
        id="test_policy_001",
        name="Test Policy",
        type=PolicyType.SAFETY,
        priority=PolicyPriority.HIGH,
        rules=[
            {
                'type': 'threshold',
                'name': 'memory_test',
                'threshold': {'metric': 'memory_mb', 'max': 1000}
            }
        ]
    )


class TestPolicy:
    """Test Policy class."""

    def test_initialization(self):
        """Test policy initialization."""
        policy = Policy(
            id="test_001",
            name="Test Policy",
            type=PolicyType.PERFORMANCE,
            priority=PolicyPriority.MEDIUM,
            rules=[{'type': 'condition', 'name': 'test'}]
        )

        assert policy.id == "test_001"
        assert policy.name == "Test Policy"
        assert policy.type == PolicyType.PERFORMANCE
        assert policy.enabled

    def test_evaluate_compliant(self, sample_policy):
        """Test evaluating compliant context."""
        context = {'memory_mb': 500}

        compliant, reason = sample_policy.evaluate(context)

        assert compliant is True

    def test_evaluate_violation(self, sample_policy):
        """Test evaluating violation."""
        context = {'memory_mb': 1500}

        compliant, reason = sample_policy.evaluate(context)

        assert compliant is False
        assert "violated" in reason

    def test_evaluate_condition_rule(self):
        """Test condition rule evaluation."""
        policy = Policy(
            id="test",
            name="Test",
            type=PolicyType.SAFETY,
            priority=PolicyPriority.HIGH,
            rules=[
                {
                    'type': 'condition',
                    'name': 'status_check',
                    'condition': {'field': 'status', 'operator': '==', 'value': 'ok'}
                }
            ]
        )

        context = {'status': 'ok'}
        compliant, _ = policy.evaluate(context)

        assert compliant is True

    def test_evaluate_operators(self):
        """Test various operators."""
        policy = Policy(
            id="test",
            name="Test",
            type=PolicyType.SAFETY,
            priority=PolicyPriority.HIGH,
            rules=[
                {
                    'type': 'condition',
                    'name': 'test',
                    'condition': {'field': 'value', 'operator': '>', 'value': 10}
                }
            ]
        )

        assert policy.evaluate({'value': 15})[0] is True
        assert policy.evaluate({'value': 5})[0] is False


class TestGovernanceLoop:
    """Test GovernanceLoop class."""

    def test_initialization(self):
        """Test loop initialization."""
        loop = GovernanceLoop(
            check_interval_s=5,
            enable_auto_enforcement=False,
            enable_policy_learning=False
        )

        assert loop.check_interval_s == 5
        assert not loop.enable_auto_enforcement
        assert not loop.enable_policy_learning
        assert len(loop.policies) > 0  # Has default policies

    def test_invalid_check_interval(self):
        """Test invalid check interval."""
        with pytest.raises(ValueError):
            GovernanceLoop(check_interval_s=0)

        with pytest.raises(ValueError):
            GovernanceLoop(check_interval_s=-1)

    def test_add_policy(self, governance, sample_policy):
        """Test adding policy."""
        governance.add_policy(sample_policy)

        assert sample_policy.id in governance.policies

    def test_add_policy_invalid_type(self, governance):
        """Test adding invalid policy type."""
        with pytest.raises(TypeError):
            governance.add_policy("not a policy")

    def test_add_policy_no_id(self, governance):
        """Test adding policy without ID."""
        policy = Policy(
            id="",
            name="Test",
            type=PolicyType.SAFETY,
            priority=PolicyPriority.HIGH,
            rules=[{'test': 'rule'}]
        )

        with pytest.raises(ValueError, match="non-empty string"):
            governance.add_policy(policy)

    def test_add_policy_too_long_id(self, governance):
        """Test adding policy with too long ID."""
        policy = Policy(
            id="x" * (MAX_POLICY_ID_LENGTH + 1),
            name="Test",
            type=PolicyType.SAFETY,
            priority=PolicyPriority.HIGH,
            rules=[{'test': 'rule'}]
        )

        with pytest.raises(ValueError, match="too long"):
            governance.add_policy(policy)

    def test_add_policy_limit(self, governance):
        """Test policy limit."""
        # Fill up to limit
        for i in range(MAX_POLICIES - len(governance.policies)):
            policy = Policy(
                id=f"policy_{i}",
                name=f"Policy {i}",
                type=PolicyType.SAFETY,
                priority=PolicyPriority.LOW,
                rules=[{'test': 'rule'}]
            )
            governance.add_policy(policy)

        # Next should fail
        with pytest.raises(ValueError, match="Maximum policy limit"):
            policy = Policy(
                id="overflow",
                name="Overflow",
                type=PolicyType.SAFETY,
                priority=PolicyPriority.LOW,
                rules=[{'test': 'rule'}]
            )
            governance.add_policy(policy)

    def test_remove_policy(self, governance, sample_policy):
        """Test removing policy."""
        governance.add_policy(sample_policy)
        governance.remove_policy(sample_policy.id)

        assert sample_policy.id not in governance.policies

    def test_remove_nonexistent_policy(self, governance):
        """Test removing non-existent policy."""
        with pytest.raises(ValueError, match="not found"):
            governance.remove_policy("nonexistent")

    def test_start_stop(self, governance):
        """Test starting and stopping loop."""
        governance.start()

        assert governance.is_running
        assert governance.governance_thread is not None

        time.sleep(0.5)

        governance.stop()

        assert not governance.is_running

    def test_start_already_running(self, governance):
        """Test starting already running loop."""
        governance.start()

        # Start again - should warn but not fail
        governance.start()

        governance.stop()

    def test_enforce_policies(self, governance):
        """Test policy enforcement."""
        action = {
            'action_type': 'normal',
            'memory_mb': 100,
            'cpu_percent': 50,
            'latency_ms': 100
        }

        result = governance.enforce_policies(action)

        assert result['compliance_checked'] is True
        assert 'compliance_score' in result

    def test_enforce_policies_blocked(self, governance):
        """Test blocking action."""
        # Add strict policy
        strict_policy = Policy(
            id="strict_001",
            name="Strict Policy",
            type=PolicyType.SAFETY,
            priority=PolicyPriority.CRITICAL,
            rules=[
                {
                    'type': 'threshold',
                    'name': 'strict_memory',
                    'threshold': {'metric': 'memory_mb', 'max': 100}
                }
            ]
        )
        governance.add_policy(strict_policy)

        action = {
            'action_type': 'normal',
            'memory_mb': 500  # Exceeds limit
        }

        result = governance.enforce_policies(action)

        # Should be blocked
        assert result.get('blocked', False)

    def test_get_compliance_report(self, governance):
        """Test getting compliance report."""
        report = governance.get_compliance_report()

        assert 'compliance_score' in report
        assert 'total_policies' in report
        assert 'total_violations' in report
        assert 'recent_violations' in report

    def test_export_policies(self, governance, sample_policy):
        """Test exporting policies."""
        governance.add_policy(sample_policy)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            governance.export_policies(filepath)

            assert os.path.exists(filepath)

            # Verify content
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            assert sample_policy.id in data

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_export_policies_permission_error(self, governance):
        """Test export with permission error."""
        if os.name != 'nt':  # Skip on Windows
            with pytest.raises(PermissionError):
                governance.export_policies("/root/test.json")

    def test_import_policies(self, governance, sample_policy):
        """Test importing policies."""
        # Export first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            governance.add_policy(sample_policy)
            governance.export_policies(filepath)

            # Create new loop and import
            governance2 = GovernanceLoop(check_interval_s=10)
            governance2.import_policies(filepath)

            assert sample_policy.id in governance2.policies

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_import_nonexistent_file(self, governance):
        """Test importing non-existent file."""
        with pytest.raises(FileNotFoundError):
            governance.import_policies("nonexistent.json")

    def test_import_invalid_json(self, governance):
        """Test importing invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("{invalid json}")
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                governance.import_policies(filepath)
        finally:
            os.remove(filepath)


class TestPolicyEnforcement:
    """Test policy enforcement."""

    def test_enforcement_actions(self, governance):
        """Test different enforcement actions."""
        # Safety enforcement
        safety_policy = governance.policies.get("safety_001")
        if safety_policy:
            context = {'memory_mb': 8000}
            success = governance._enforce_safety(safety_policy, context, "memory limit")
            assert isinstance(success, bool)

    def test_violation_recording(self, governance, sample_policy):
        """Test violation recording."""
        governance.add_policy(sample_policy)

        context = {'memory_mb': 1500}

        initial_violations = len(governance.violations)
        governance._record_violation(sample_policy, context, "Test violation")

        assert len(governance.violations) > initial_violations


class TestPolicyLearning:
    """Test policy learning."""

    def test_learn_from_outcomes(self, governance):
        """Test learning from outcomes."""
        results = {
            'test_policy': (True, "Compliant"),
            'test_policy2': (False, "Violation")
        }

        governance._learn_from_outcomes(results)

        # Should track effectiveness
        assert len(governance.policy_effectiveness) > 0

    def test_adapt_policies(self, governance, sample_policy):
        """Test policy adaptation."""
        governance.add_policy(sample_policy)

        # Simulate many violations
        governance.violation_counts[sample_policy.id] = 100

        # Set up effectiveness data
        governance.policy_effectiveness[sample_policy.id] = {
            'success': 20,
            'total': 100
        }

        governance._adapt_policies()

        # Policy might be modified based on effectiveness


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_policy_operations(self, governance):
        """Test concurrent policy operations."""
        def add_policies():
            for i in range(10):
                try:
                    policy = Policy(
                        id=f"concurrent_{threading.current_thread().ident}_{i}",
                        name=f"Policy {i}",
                        type=PolicyType.OPERATIONAL,
                        priority=PolicyPriority.LOW,
                        rules=[{'test': 'rule'}]
                    )
                    governance.add_policy(policy)
                except:
                    pass

        threads = [threading.Thread(target=add_policies) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without crashes
        assert len(governance.policies) > 0

    def test_concurrent_enforcement(self, governance):
        """Test concurrent policy enforcement."""
        results = []

        def enforce():
            action = {
                'action_type': 'test',
                'memory_mb': 100
            }
            result = governance.enforce_policies(action)
            results.append(result)

        threads = [threading.Thread(target=enforce) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r['compliance_checked'] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
