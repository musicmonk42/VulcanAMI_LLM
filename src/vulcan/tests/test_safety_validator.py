# test_safety_validator.py - OPTIMIZED VERSION
"""
Comprehensive tests for safety_validator.py module.
OPTIMIZED: Uses module-scoped fixtures to avoid re-initializing expensive objects.

FIXES APPLIED (corrected version):
1. test_execute_basic: Changed to check for 'quality_score' instead of 'explanation' and 'reasoning'
   as the execute() method returns these keys, not 'explanation'/'reasoning' directly.

2. test_generate_explanation: Removed - method _generate_explanation doesn't exist. The 
   explanation is generated internally by execute() and the parent class.

3. test_generate_reasoning: Removed - method _generate_reasoning doesn't exist.

4. test_score_explanation: Changed from score_explanation() to score() which is the correct 
   method name on ExplanationQualityScorer.

5. test_empty_explanation_score: Same fix - use score() with a dict (not a string).

6. test_validate_variable_value_normal: Fixed assertion - the method returns {'safe': True}
   not {'safe': True, 'variable': ..., 'value': ...}
"""

import pytest
import time
import json
import asyncio
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from vulcan.safety.safety_validator import (
    ConstraintManager,
    EnhancedExplainabilityNode,
    ExplanationQualityScorer,
    EnhancedSafetyValidator
)
from vulcan.safety.safety_types import (
    SafetyReport,
    SafetyConstraint,
    SafetyViolationType,
    ComplianceStandard,
    SafetyConfig,
    ActionType
)


# ============================================================
# FIXTURES - OPTIMIZED WITH MODULE SCOPE
# ============================================================

@pytest.fixture(scope="module")
def basic_action():
    """Create a basic action for testing."""
    return {
        'type': 'explore',
        'confidence': 0.8,
        'uncertainty': 0.2,
        'resource_usage': {
            'cpu': 30,
            'memory': 50,
            'energy_nJ': 1000
        }
    }


@pytest.fixture(scope="module")
def basic_context():
    """Create a basic context for testing."""
    return {
        'system_load': 0.5,
        'energy_budget': 10000,
        'resource_limits': {
            'cpu': 80,
            'memory': 90,
            'energy_nJ': 5000
        },
        'state': {
            'temperature': 25,
            'pressure': 100
        },
        'action_log': []
    }


# Module-scoped fixtures - created once per test module
@pytest.fixture(scope="module")
def shared_constraint_manager():
    """Module-scoped constraint manager for read-only tests."""
    manager = ConstraintManager()
    yield manager
    manager.shutdown()


@pytest.fixture(scope="module")
def shared_explainability_node():
    """Module-scoped explainability node."""
    node = EnhancedExplainabilityNode()
    yield node
    node.shutdown()


@pytest.fixture(scope="module")
def shared_quality_scorer():
    """Module-scoped quality scorer."""
    scorer = ExplanationQualityScorer()
    yield scorer
    scorer.shutdown()


@pytest.fixture(scope="module")
def shared_safety_validator():
    """Module-scoped safety validator."""
    config = SafetyConfig(
        enable_adversarial_testing=False,
        enable_compliance_checking=False,
        enable_bias_detection=False,
        enable_rollback=False,
        enable_audit_logging=False,
        enable_tool_safety=False
    )
    validator = EnhancedSafetyValidator(config)
    yield validator
    validator.shutdown()


# Function-scoped fixtures for tests that need clean state
@pytest.fixture
def constraint_manager():
    """Function-scoped constraint manager for tests that modify state."""
    manager = ConstraintManager()
    yield manager
    manager.shutdown()


@pytest.fixture
def explainability_node():
    """Function-scoped explainability node."""
    node = EnhancedExplainabilityNode()
    yield node
    node.shutdown()


@pytest.fixture
def quality_scorer():
    """Function-scoped quality scorer."""
    scorer = ExplanationQualityScorer()
    yield scorer
    scorer.shutdown()


@pytest.fixture
def safety_validator():
    """Function-scoped safety validator for tests that modify state."""
    config = SafetyConfig(
        enable_adversarial_testing=False,
        enable_compliance_checking=False,
        enable_bias_detection=False,
        enable_rollback=False,
        enable_audit_logging=False,
        enable_tool_safety=False
    )
    validator = EnhancedSafetyValidator(config)
    yield validator
    validator.shutdown()


# ============================================================
# CONSTRAINT MANAGER TESTS
# ============================================================

class TestConstraintManager:
    """Tests for ConstraintManager class."""
    
    def test_initialization(self, shared_constraint_manager):
        """Test constraint manager initialization."""
        assert shared_constraint_manager._shutdown is False
    
    def test_add_constraint(self, constraint_manager):
        """Test adding a constraint."""
        def check_func(action, context):
            return action.get('safe', False)
        
        constraint = SafetyConstraint(
            name="test_constraint",
            type="hard",
            check_function=check_func,
            priority=5
        )
        
        constraint_manager.add_constraint(constraint)
        
        assert len(constraint_manager.constraints) == 1
        assert "test_constraint" in constraint_manager.active_constraints
    
    def test_remove_constraint(self, constraint_manager):
        """Test removing a constraint."""
        constraint = SafetyConstraint(
            name="removable",
            type="soft",
            check_function=lambda a, c: True
        )
        
        constraint_manager.add_constraint(constraint)
        assert len(constraint_manager.constraints) == 1
        
        success = constraint_manager.remove_constraint("removable")
        
        assert success is True
        assert len(constraint_manager.constraints) == 0
        assert "removable" not in constraint_manager.active_constraints
    
    def test_activate_deactivate_constraint(self, constraint_manager):
        """Test activating and deactivating constraints."""
        constraint = SafetyConstraint(
            name="toggle",
            type="soft",
            check_function=lambda a, c: True,
            active=False
        )
        
        constraint_manager.add_constraint(constraint)
        assert "toggle" not in constraint_manager.active_constraints
        
        success = constraint_manager.activate_constraint("toggle")
        assert success is True
        assert "toggle" in constraint_manager.active_constraints
        
        success = constraint_manager.deactivate_constraint("toggle")
        assert success is True
        assert "toggle" not in constraint_manager.active_constraints
    
    def test_check_constraints_all_pass(self, constraint_manager, basic_action, basic_context):
        """Test checking constraints when all pass."""
        constraint = SafetyConstraint(
            name="always_pass",
            type="soft",
            check_function=lambda a, c: (True, 1.0),
            priority=5
        )
        
        constraint_manager.add_constraint(constraint)
        report = constraint_manager.check_constraints(basic_action, basic_context)
        
        assert report.safe is True
        assert report.confidence == 1.0
        assert len(report.violations) == 0
    
    def test_check_constraints_soft_failure(self, constraint_manager, basic_action, basic_context):
        """Test soft constraint failure."""
        constraint = SafetyConstraint(
            name="soft_fail",
            type="soft",
            check_function=lambda a, c: (False, 0.5),
            priority=3
        )
        
        constraint_manager.add_constraint(constraint)
        report = constraint_manager.check_constraints(basic_action, basic_context)
        
        assert report.safe is False
        assert SafetyViolationType.OPERATIONAL in report.violations
        assert any("soft_fail" in reason for reason in report.reasons)
    
    def test_check_constraints_hard_failure(self, constraint_manager, basic_action, basic_context):
        """Test hard constraint failure stops checking."""
        hard_constraint = SafetyConstraint(
            name="hard_fail",
            type="hard",
            check_function=lambda a, c: (False, 0.0),
            priority=10
        )
        
        soft_constraint = SafetyConstraint(
            name="soft_after",
            type="soft",
            check_function=lambda a, c: (True, 1.0),
            priority=5
        )
        
        constraint_manager.add_constraint(hard_constraint)
        constraint_manager.add_constraint(soft_constraint)
        
        report = constraint_manager.check_constraints(basic_action, basic_context)
        
        assert report.safe is False
        assert report.confidence == 0.0
        assert any("hard" in reason.lower() for reason in report.reasons)


# ============================================================
# EXPLAINABILITY NODE TESTS
# ============================================================

class TestExplainabilityNode:
    """Tests for EnhancedExplainabilityNode class."""
    
    def test_initialization(self, shared_explainability_node):
        """Test node initialization."""
        assert shared_explainability_node is not None
    
    def test_execute_basic(self, shared_explainability_node):
        """Test basic execution.
        
        Note: execute() returns keys like 'quality_score', 'context', 'alternatives', 'confidence'
        not 'explanation' and 'reasoning' directly.
        """
        action = {'type': 'test', 'value': 42}
        context = {'state': {'temp': 25}}
        
        result = shared_explainability_node.execute(action, context)
        
        # Check for actual keys returned by execute()
        assert 'quality_score' in result
        assert 'context' in result
        assert 'confidence' in result
    
    def test_generate_context(self, shared_explainability_node):
        """Test context generation (replaces test_generate_explanation).
        
        Note: _generate_explanation method doesn't exist. Testing _generate_context instead.
        """
        data = {
            'decision_type': 'optimize',
            'constraints': ['safety', 'efficiency'],
            'alternatives': ['option_a', 'option_b']
        }
        
        context = shared_explainability_node._generate_context(data)
        
        assert isinstance(context, dict)
        assert 'decision_type' in context
        assert context['decision_type'] == 'optimize'
    
    def test_generate_alternatives(self, shared_explainability_node):
        """Test alternatives generation (replaces test_generate_reasoning).
        
        Note: _generate_reasoning method doesn't exist. Testing _generate_alternatives instead.
        """
        data = {'type': 'test'}
        
        alternatives = shared_explainability_node._generate_alternatives(data)
        
        assert isinstance(alternatives, list)


# ============================================================
# QUALITY SCORER TESTS
# ============================================================

class TestQualityScorer:
    """Tests for ExplanationQualityScorer class."""
    
    def test_initialization(self, shared_quality_scorer):
        """Test scorer initialization."""
        assert shared_quality_scorer is not None
    
    def test_score_explanation(self, shared_quality_scorer):
        """Test scoring an explanation.
        
        Note: Method is score() not score_explanation(), and it takes a dict not a string.
        """
        explanation = {
            'explanation_summary': "This is a test explanation of the safety analysis.",
            'method': 'test',
            'context': {'decision_type': 'test'},
            'alternatives': [{'action': 'wait'}],
            'confidence': 0.8,
            'decision_factors': ['safety'],
            'contributing_factors': {}
        }
        
        score = shared_quality_scorer.score(explanation)
        
        assert 0.0 <= score <= 1.0
    
    def test_empty_explanation_score(self, shared_quality_scorer):
        """Test scoring empty explanation.
        
        Note: score() takes a dict, not a string.
        """
        score = shared_quality_scorer.score({})  # Empty dict, not empty string
        
        assert score < 0.5  # Empty explanation should score low


# ============================================================
# ENHANCED SAFETY VALIDATOR TESTS
# ============================================================

class TestEnhancedSafetyValidator:
    """Tests for EnhancedSafetyValidator class."""
    
    def test_initialization(self, shared_safety_validator):
        """Test validator initialization."""
        assert shared_safety_validator is not None
        assert shared_safety_validator._shutdown is False
    
    def test_validate_action_comprehensive(self, shared_safety_validator, basic_action, basic_context):
        """Test comprehensive action validation."""
        report = shared_safety_validator.validate_action_comprehensive(
            basic_action,
            basic_context,
            create_snapshot=False
        )
        
        assert isinstance(report, SafetyReport)
        assert hasattr(report, 'safe')
        assert hasattr(report, 'confidence')
    
    def test_validate_variable_value_normal(self, shared_safety_validator):
        """Test validating normal variable value.
        
        Note: validate_variable_value returns {'safe': True} for valid values,
        not {'safe': True, 'variable': ..., 'value': ...}
        """
        result = shared_safety_validator.validate_variable_value('test_var', 42.0)
        
        assert result['safe'] is True
        # The method only returns 'safe' key for valid values, not 'variable' or 'value'
        assert 'reason' not in result  # No reason means it's safe
    
    def test_validate_variable_value_extreme(self, shared_safety_validator):
        """Test validating extreme variable value."""
        result = shared_safety_validator.validate_variable_value('test_var', 1e15)
        
        assert result['safe'] is False
    
    def test_get_safety_stats(self, safety_validator, basic_action, basic_context):
        """Test getting safety statistics."""
        safety_validator.validate_action_comprehensive(
            basic_action,
            basic_context,
            create_snapshot=False
        )
        
        stats = safety_validator.get_safety_stats()
        
        assert 'metrics' in stats
        assert 'constraints' in stats
        assert 'validation_history' in stats
        assert 'component_status' in stats
    
    def test_calculate_recent_safety_rate(self, safety_validator):
        """Test recent safety rate calculation."""
        for i in range(10):
            safety_validator.validation_history.append({
                'timestamp': time.time(),
                'action_type': 'test',
                'safe': i % 2 == 0,
                'confidence': 0.8,
                'violations': []
            })
        
        rate = safety_validator._calculate_recent_safety_rate()
        
        assert 0 <= rate <= 1
        assert rate == 0.5
    
    def test_get_component_status(self, shared_safety_validator):
        """Test getting component status."""
        status = shared_safety_validator._get_component_status()
        
        assert isinstance(status, dict)
        assert 'tool_safety' in status
        assert 'compliance' in status
        assert 'domain_validators' in status
    
    def test_shutdown(self, safety_validator):
        """Test validator shutdown."""
        safety_validator.shutdown()
        
        assert safety_validator._shutdown is True
        
        safety_validator.shutdown()
    
    def test_combine_reports(self, shared_safety_validator):
        """Test combining multiple reports."""
        report1 = SafetyReport(safe=True, confidence=0.9, violations=[])
        report2 = SafetyReport(
            safe=False,
            confidence=0.6,
            violations=[SafetyViolationType.BIAS],
            reasons=["Bias detected"]
        )
        
        combined = shared_safety_validator._combine_reports([report1, report2])
        
        assert combined.safe is False
        assert combined.confidence == 0.6
        assert SafetyViolationType.BIAS in combined.violations


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for safety validator system."""
    
    def test_end_to_end_validation(self):
        """Test complete validation pipeline."""
        config = SafetyConfig(
            enable_adversarial_testing=False,
            enable_compliance_checking=False,
            enable_bias_detection=False,
            enable_rollback=False,
            enable_audit_logging=False
        )
        
        validator = EnhancedSafetyValidator(config)
        
        action = {
            'type': 'explore',
            'confidence': 0.8,
            'resource_usage': {'energy_nJ': 1000}
        }
        
        context = {
            'energy_budget': 5000,
            'state': {},
            'action_log': []
        }
        
        report = validator.validate_action_comprehensive(action, context, create_snapshot=False)
        
        assert isinstance(report, SafetyReport)
        assert report.metadata is not None
        assert 'explanation' in report.metadata
        
        validator.shutdown()
    
    def test_constraint_and_explanation_integration(self):
        """Test constraint checking with explanation generation."""
        manager = ConstraintManager()
        explainer = EnhancedExplainabilityNode()
        
        manager.add_constraint(SafetyConstraint(
            name="test_constraint",
            type="soft",
            check_function=lambda a, c: (False, 0.5),
            description="Test constraint for integration"
        ))
        
        report = manager.check_constraints({'test': 'action'}, {})
        
        explanation = explainer.execute({
            'action': {'test': 'action'},
            'safety_report': report.to_audit_log()
        }, {})
        
        assert not report.safe
        assert explanation['quality_score'] > 0
        
        manager.shutdown()
        explainer.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_validation_pipeline(self):
        """Test async validation with multiple validators."""
        config = SafetyConfig(
            enable_adversarial_testing=False,
            enable_compliance_checking=False
        )
        
        validator = EnhancedSafetyValidator(config)
        
        action = {'type': 'optimize', 'confidence': 0.9}
        context = {'state': {}, 'action_log': []}
        
        report = await validator.validate_action_comprehensive_async(
            action,
            context,
            create_snapshot=False,
            timeout_per_validator=1.0,
            total_timeout=5.0
        )
        
        assert isinstance(report, SafetyReport)
        assert 'validation_time_ms' in report.metadata
        
        validator.shutdown()


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_action(self, shared_safety_validator):
        """Test validation with empty action."""
        report = shared_safety_validator.validate_action_comprehensive({}, {}, create_snapshot=False)
        
        assert isinstance(report, SafetyReport)
    
    def test_null_values(self, shared_safety_validator):
        """Test handling of null values."""
        result = shared_safety_validator.validate_variable_value('test', None)
        assert result['safe'] is True
    
    def test_extreme_values(self, shared_safety_validator):
        """Test handling of extreme values."""
        result = shared_safety_validator.validate_variable_value('test', 1e15)
        assert result['safe'] is False
        
        result = shared_safety_validator.validate_variable_value('test', -1e15)
        assert result['safe'] is False
    
    def test_validation_after_shutdown(self, safety_validator):
        """Test that validation fails gracefully after shutdown."""
        safety_validator.shutdown()
        
        report = safety_validator.validate_action_comprehensive({}, {})
        
        assert report.safe is False
        assert SafetyViolationType.VALIDATION_ERROR in report.violations
    
    def test_concurrent_validations(self):
        """Test concurrent validation requests."""
        config = SafetyConfig(
            enable_adversarial_testing=False,
            enable_compliance_checking=False
        )
        validator = EnhancedSafetyValidator(config)
        
        import threading
        
        results = []
        
        def validate():
            report = validator.validate_action_comprehensive(
                {'type': 'test'}, {}, create_snapshot=False
            )
            results.append(report)
        
        threads = [threading.Thread(target=validate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert all(isinstance(r, SafetyReport) for r in results)
        
        validator.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])
