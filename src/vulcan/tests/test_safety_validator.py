# test_safety_validator.py
"""
Comprehensive tests for safety_validator.py module.
Tests the main orchestrator, constraint manager, explainability, and integration.
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
# FIXTURES
# ============================================================

@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def constraint_manager():
    """Create a constraint manager."""
    manager = ConstraintManager()
    yield manager
    manager.shutdown()


@pytest.fixture
def explainability_node():
    """Create an explainability node."""
    node = EnhancedExplainabilityNode()
    yield node
    node.shutdown()


@pytest.fixture
def quality_scorer():
    """Create a quality scorer."""
    scorer = ExplanationQualityScorer()
    yield scorer
    scorer.shutdown()


@pytest.fixture
def safety_validator():
    """Create enhanced safety validator with mock components."""
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
    
    def test_initialization(self):
        """Test constraint manager initialization."""
        manager = ConstraintManager()
        
        assert len(manager.constraints) == 0
        assert len(manager.active_constraints) == 0
        assert manager._shutdown is False
        
        manager.shutdown()
    
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
        
        # Activate
        success = constraint_manager.activate_constraint("toggle")
        assert success is True
        assert "toggle" in constraint_manager.active_constraints
        
        # Deactivate
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
    
    def test_constraint_priority_ordering(self, constraint_manager, basic_action, basic_context):
        """Test that constraints are checked in priority order."""
        # Add constraints with different priorities
        for i in range(3):
            constraint = SafetyConstraint(
                name=f"constraint_{i}",
                type="soft",
                check_function=lambda a, c: (True, 1.0),
                priority=i
            )
            constraint_manager.add_constraint(constraint)
        
        report = constraint_manager.check_constraints(basic_action, basic_context)
        
        # All should pass
        assert report.safe is True
        assert len(report.metadata['constraint_names']) == 3
    
    def test_constraint_exception_handling(self, constraint_manager, basic_action, basic_context):
        """Test handling of exceptions in constraint checks."""
        def failing_check(action, context):
            raise ValueError("Test error")
        
        constraint = SafetyConstraint(
            name="failing",
            type="soft",
            check_function=failing_check
        )
        
        constraint_manager.add_constraint(constraint)
        report = constraint_manager.check_constraints(basic_action, basic_context)
        
        assert report.safe is False
        assert SafetyViolationType.OPERATIONAL in report.violations
    
    def test_get_constraint_stats(self, constraint_manager):
        """Test getting constraint statistics."""
        constraint = SafetyConstraint(
            name="tracked",
            type="soft",
            check_function=lambda a, c: (True, 0.8),
            priority=5
        )
        
        constraint_manager.add_constraint(constraint)
        constraint_manager.check_constraints({}, {})
        
        stats = constraint_manager.get_constraint_stats()
        
        assert stats['total_constraints'] == 1
        assert stats['active_constraints'] == 1
        assert len(stats['constraint_details']) == 1
        assert stats['constraint_details'][0]['name'] == "tracked"
    
    def test_reset_violations(self, constraint_manager):
        """Test resetting violation counters."""
        constraint = SafetyConstraint(
            name="test",
            type="soft",
            check_function=lambda a, c: (False, 0.5)
        )
        
        constraint_manager.add_constraint(constraint)
        constraint_manager.check_constraints({}, {})
        
        stats = constraint_manager.get_constraint_stats()
        assert stats['total_violations'] > 0
        
        # Reset all
        constraint_manager.reset_violations()
        stats = constraint_manager.get_constraint_stats()
        assert stats['total_violations'] == 0
        
        # Reset specific
        constraint_manager.check_constraints({}, {})
        constraint_manager.reset_violations("test")
        stats = constraint_manager.get_constraint_stats()
        assert stats['total_violations'] == 0
    
    def test_shutdown(self, constraint_manager):
        """Test constraint manager shutdown."""
        constraint_manager.add_constraint(SafetyConstraint(
            name="test",
            type="soft",
            check_function=lambda a, c: True
        ))
        
        constraint_manager.shutdown()
        
        assert constraint_manager._shutdown is True
        assert len(constraint_manager.constraints) == 0


# ============================================================
# EXPLANATION QUALITY SCORER TESTS
# ============================================================

class TestExplanationQualityScorer:
    """Tests for ExplanationQualityScorer class."""
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = ExplanationQualityScorer()
        
        assert len(scorer.scoring_history) == 0
        assert 'excellent' in scorer.quality_thresholds
        
        scorer.shutdown()
    
    def test_score_minimal_explanation(self, quality_scorer):
        """Test scoring minimal explanation."""
        explanation = {'explanation_summary': 'Test explanation'}
        
        score = quality_scorer.score(explanation)
        
        assert 0 <= score <= 1
        assert len(quality_scorer.scoring_history) == 1
    
    def test_score_complete_explanation(self, quality_scorer):
        """Test scoring complete explanation."""
        explanation = {
            'explanation_summary': 'Comprehensive test explanation.',
            'method': 'test',
            'context': {'type': 'test'},
            'alternatives': [{'action': 'alt1'}],
            'confidence': 0.9,
            'decision_factors': ['factor1', 'factor2'],
            'visual_aids': {'type': 'chart'},
            'feature_importance': [{'feature': 'f1', 'importance': 0.8}]
        }
        
        score = quality_scorer.score(explanation)
        
        assert score > 0.5
        assert len(quality_scorer.scoring_history) == 1
    
    def test_score_completeness(self, quality_scorer):
        """Test completeness scoring."""
        minimal = {'explanation_summary': 'Test'}
        complete = {
            'explanation_summary': 'Test',
            'method': 'test',
            'context': {},
            'alternatives': [],
            'confidence': 0.9,
            'decision_factors': [],
            'visual_aids': {},
            'shap_scores': []
        }
        
        minimal_comp = quality_scorer._score_completeness(minimal)
        complete_comp = quality_scorer._score_completeness(complete)
        
        assert complete_comp > minimal_comp
    
    def test_score_clarity(self, quality_scorer):
        """Test clarity scoring."""
        too_short = {'explanation_summary': 'Short.'}
        good_length = {'explanation_summary': 'This is a well-written explanation that provides clear information about the decision.' * 2}
        too_long = {'explanation_summary': 'Very long explanation.' * 100}
        
        short_clarity = quality_scorer._score_clarity(too_short)
        good_clarity = quality_scorer._score_clarity(good_length)
        long_clarity = quality_scorer._score_clarity(too_long)
        
        assert good_clarity > short_clarity
        assert good_clarity > long_clarity
    
    def test_get_quality_category(self, quality_scorer):
        """Test quality category determination."""
        assert quality_scorer.get_quality_category(0.9) == 'excellent'
        assert quality_scorer.get_quality_category(0.7) == 'good'
        assert quality_scorer.get_quality_category(0.5) == 'acceptable'
        # FIX: 0.3 is below the 0.4 threshold for 'acceptable', so it should be 'poor'
        assert quality_scorer.get_quality_category(0.3) == 'poor'
        assert quality_scorer.get_quality_category(0.1) == 'poor'
    
    def test_get_scoring_stats(self, quality_scorer):
        """Test getting scoring statistics."""
        # Score multiple explanations
        for i in range(5):
            quality_scorer.score({'explanation_summary': f'Test {i}'})
        
        stats = quality_scorer.get_scoring_stats()
        
        assert stats['total_scored'] == 5
        assert 'average_score' in stats
        assert 'distribution' in stats
    
    def test_calculate_trend(self, quality_scorer):
        """Test trend calculation."""
        # Not enough data
        trend = quality_scorer._calculate_trend()
        assert trend == 'insufficient_data'
        
        # Add improving scores
        for i in range(20):
            score = 0.3 + (i * 0.02)
            quality_scorer.scoring_history.append({
                'timestamp': time.time(),
                'score': score,
                'completeness': score,
                'clarity': score,
                'relevance': score,
                'usefulness': score
            })
        
        trend = quality_scorer._calculate_trend()
        assert trend == 'improving'


# ============================================================
# ENHANCED EXPLAINABILITY NODE TESTS
# ============================================================

class TestEnhancedExplainabilityNode:
    """Tests for EnhancedExplainabilityNode class."""
    
    def test_initialization(self):
        """Test node initialization."""
        node = EnhancedExplainabilityNode()
        
        assert node.explanation_quality_scorer is not None
        assert len(node.explanation_cache) == 0
        
        node.shutdown()
    
    def test_execute_basic(self, explainability_node):
        """Test basic explanation execution."""
        data = {
            'decision': 'explore',
            'confidence': 0.8
        }
        
        result = explainability_node.execute(data, {})
        
        assert 'explanation_summary' in result
        assert 'confidence' in result
        assert 'quality_score' in result
    
    def test_execute_with_context(self, explainability_node):
        """Test explanation with context."""
        data = {
            'decision': 'optimize',
            'constraints': ['energy', 'time'],
            'safety_report': {
                'safe': True,
                'confidence': 0.9,
                'violations': []
            }
        }
        
        result = explainability_node.execute(data, {})
        
        assert 'context' in result
        assert 'alternatives' in result
        assert result['confidence'] > 0
    
    def test_caching(self, explainability_node):
        """Test explanation caching."""
        data = {'test': 'data'}
        
        # First call
        result1 = explainability_node.execute(data, {})
        cache_size1 = len(explainability_node.explanation_cache)
        
        # Second call with same data
        result2 = explainability_node.execute(data, {})
        cache_size2 = len(explainability_node.explanation_cache)
        
        assert cache_size1 == 1
        assert cache_size2 == 1
        assert result1['quality_score'] == result2['quality_score']
    
    def test_feature_importance_extraction(self, explainability_node):
        """Test feature importance extraction."""
        shap_scores = [0.8, 0.6, 0.4, 0.2]
        
        features = explainability_node._extract_feature_importance(shap_scores)
        
        assert len(features) == 4
        assert all('feature' in f for f in features)
        assert all('importance' in f for f in features)
    
    def test_generate_alternatives(self, explainability_node):
        """Test alternative generation."""
        data = {'decision': 'explore'}
        
        alternatives = explainability_node._generate_alternatives(data)
        
        assert len(alternatives) > 0
        assert all('action' in alt for alt in alternatives)
        assert all('reason' in alt for alt in alternatives)
    
    def test_explanation_improvement(self, explainability_node):
        """Test explanation improvement for low quality."""
        explanation = {
            'explanation_summary': 'Poor',
            'quality_score': 0.3
        }
        data = {'decision': 'test'}
        
        improved = explainability_node._improve_explanation(explanation, data)
        
        assert 'detailed_reasoning' in improved
        assert 'step_by_step' in improved
        assert 'examples' in improved
    
    def test_truncate_explanation(self, explainability_node):
        """Test explanation truncation."""
        long_text = "This is a very long explanation. " * 100
        max_length = 200
        
        truncated = explainability_node._truncate_explanation(long_text, max_length)
        
        assert len(truncated) <= max_length
    
    def test_get_explanation_stats(self, explainability_node):
        """Test getting explanation statistics."""
        # Generate some explanations
        for i in range(3):
            explainability_node.execute({'test': i}, {})
        
        stats = explainability_node.get_explanation_stats()
        
        assert stats['total_explanations'] == 3
        assert 'average_quality' in stats
        assert 'cache_size' in stats
    
    def test_shutdown(self, explainability_node):
        """Test explainability node shutdown."""
        explainability_node.execute({'test': 'data'}, {})
        
        explainability_node.shutdown()
        
        assert explainability_node._shutdown is True
        assert len(explainability_node.explanation_cache) == 0


# ============================================================
# ENHANCED SAFETY VALIDATOR TESTS
# ============================================================

class TestEnhancedSafetyValidator:
    """Tests for EnhancedSafetyValidator class."""
    
    def test_initialization(self):
        """Test validator initialization."""
        config = SafetyConfig()
        validator = EnhancedSafetyValidator(config)
        
        assert validator.constraint_manager is not None
        assert validator.explainability_node is not None
        assert validator.safety_metrics is not None
        
        validator.shutdown()
    
    def test_initialization_with_dict_config(self):
        """Test initialization with dictionary config."""
        config_dict = {
            'enable_adversarial_testing': False,
            'enable_compliance_checking': False,
            'enable_bias_detection': False
        }
        
        validator = EnhancedSafetyValidator(config_dict)
        
        assert validator.safety_config is not None
        validator.shutdown()
    
    def test_default_constraints_setup(self, safety_validator):
        """Test that default constraints are set up."""
        stats = safety_validator.constraint_manager.get_constraint_stats()
        
        assert stats['total_constraints'] > 0
        assert any('energy' in c['name'] for c in stats['constraint_details'])
    
    def test_validate_action_basic(self, safety_validator, basic_action, basic_context):
        """Test basic action validation."""
        is_safe, reason, confidence = safety_validator.validate_action(
            basic_action,
            basic_context
        )
        
        assert isinstance(is_safe, bool)
        assert isinstance(reason, str)
        assert 0 <= confidence <= 1
    
    def test_validate_action_comprehensive_sync(self, safety_validator, basic_action, basic_context):
        """Test comprehensive synchronous validation."""
        report = safety_validator.validate_action_comprehensive(
            basic_action,
            basic_context,
            create_snapshot=False
        )
        
        assert isinstance(report, SafetyReport)
        assert hasattr(report, 'safe')
        assert hasattr(report, 'confidence')
        # FIX: Check for either constraints_checked or num_checks (both should be present now)
        assert 'constraints_checked' in report.metadata or 'num_checks' in report.metadata
    
    @pytest.mark.asyncio
    async def test_validate_action_comprehensive_async(self, safety_validator, basic_action, basic_context):
        """Test comprehensive asynchronous validation."""
        report = await safety_validator.validate_action_comprehensive_async(
            basic_action,
            basic_context,
            create_snapshot=False,
            timeout_per_validator=2.0,
            total_timeout=10.0
        )
        
        assert isinstance(report, SafetyReport)
        assert 'validation_time_ms' in report.metadata
    
    def test_validate_variable_value(self, safety_validator):
        """Test variable value validation."""
        # Valid value
        result = safety_validator.validate_variable_value('temperature', 50)
        assert result['safe'] is True
        
        # Out of range
        result = safety_validator.validate_variable_value('temperature', 200)
        assert result['safe'] is False
        
        # NaN value
        result = safety_validator.validate_variable_value('test', np.nan)
        assert result['safe'] is False
    
    def test_validate_state_vector(self, safety_validator):
        """Test state vector validation."""
        state = np.array([25, 100, 5000])
        variables = ['temperature', 'pressure', 'energy']
        
        result = safety_validator.validate_state_vector(state, variables)
        
        assert 'safe' in result
        
        # Test with NaN
        bad_state = np.array([np.nan, 100, 5000])
        result = safety_validator.validate_state_vector(bad_state, variables)
        assert result['safe'] is False
    
    def test_clamp_to_safe_region(self, safety_validator):
        """Test clamping to safe region."""
        class MockState:
            def __init__(self):
                self.variables = {
                    'temperature': 200,  # Out of range
                    'pressure': 50,       # In range
                    'unknown': np.nan     # Invalid
                }
        
        state = MockState()
        variables = ['temperature', 'pressure', 'unknown']
        
        clamped = safety_validator.clamp_to_safe_region(state, variables)
        
        assert clamped.variables['temperature'] <= 100
        assert clamped.variables['pressure'] == 50
        assert not np.isnan(clamped.variables['unknown'])
    
    def test_validate_prediction_value(self, safety_validator):
        """Test prediction value validation."""
        # Valid
        result = safety_validator.validate_prediction_value(0.8, 'confidence')
        assert result['safe'] is True
        
        # Out of range
        result = safety_validator.validate_prediction_value(1.5, 'confidence')
        assert result['safe'] is False
        assert 'safe_value' in result
    
    def test_validate_effect_magnitude(self, safety_validator):
        """Test effect magnitude validation."""
        # Valid effect
        result = safety_validator.validate_effect_magnitude('A', 'B', 0.5)
        assert result['safe'] is True
        
        # Too large
        result = safety_validator.validate_effect_magnitude('A', 'B', 150)
        assert result['safe'] is False
        
        # Harmful pattern
        result = safety_validator.validate_effect_magnitude('action', 'harm', 0.5)
        assert result['safe'] is False
    
    def test_validate_intervention(self, safety_validator):
        """Test intervention validation."""
        # Valid intervention
        result = safety_validator.validate_intervention(
            'temp',
            'pressure',
            'simulated',
            {'approved': True}
        )
        assert result['safe'] is True
        
        # Forbidden variable
        result = safety_validator.validate_intervention(
            'system_critical',
            'output',
            'real_world',
            {}
        )
        assert result['safe'] is False
    
    def test_is_safe_region(self, safety_validator):
        """Test safe region checking."""
        safe_context = {
            'state': {
                'temperature': 25,
                'pressure': 100
            },
            'risk_level': 'low'
        }
        
        assert safety_validator.is_safe_region(safe_context) is True
        
        # Unsafe context
        unsafe_context = {
            'state': {
                'temperature': 200  # Out of range
            }
        }
        
        assert safety_validator.is_safe_region(unsafe_context) is False
    
    def test_validate_pattern(self, safety_validator):
        """Test pattern validation."""
        # FIX: Create a more complete mock pattern with better action structure
        class MockPattern:
            def __init__(self):
                self.actions = [
                    {
                        'type': 'explore',
                        'safe': True,
                        'confidence': 0.8,
                        'resource_usage': {
                            'energy_nJ': 500
                        }
                    }
                ]
                self.metadata = {'harmful': False, 'reward': 10}
        
        pattern = MockPattern()
        result = safety_validator.validate_pattern(pattern)
        
        # The pattern should be safe since:
        # 1. It has a valid action type
        # 2. It's not marked as harmful
        # 3. It has positive reward
        # 4. The code now handles incomplete actions gracefully
        assert result['safe'] is True
    
    def test_validate_graph_basic(self, safety_validator):
        """Test basic graph validation."""
        graph = {
            'nodes': {
                'node1': {'type': 'compute', 'edges': []},
                'node2': {'type': 'output', 'edges': []}
            }
        }
        
        report = safety_validator.validate(graph)
        
        assert isinstance(report, SafetyReport)
    
    def test_validate_graph_with_cycle(self, safety_validator):
        """Test graph validation with cycle detection."""
        graph = {
            'nodes': {
                'A': {'type': 'compute', 'edges': [{'target': 'B'}]},
                'B': {'type': 'compute', 'edges': [{'target': 'A'}]}
            }
        }
        
        report = safety_validator.validate(graph)
        
        # FIX: The fixed code should properly detect cycles and add the reason
        assert report.safe is False
        assert any('cyclic' in reason.lower() or 'cycle' in reason.lower() for reason in report.reasons)
    
    def test_validate_graph_invalid_node_types(self, safety_validator):
        """Test graph validation with invalid node types."""
        graph = {
            'nodes': {
                'node1': {'type': 'invalid_type', 'edges': []}
            }
        }
        
        report = safety_validator.validate(graph)
        
        assert report.safe is False
        assert any('invalid' in reason.lower() for reason in report.reasons)
    
    def test_validate_tool_selection(self, safety_validator):
        """Test tool selection validation."""
        tools = ['tool1', 'tool2', 'tool3']
        context = {'confidence': 0.8}
        
        allowed_tools, report = safety_validator.validate_tool_selection(
            tools,
            context
        )
        
        assert isinstance(allowed_tools, list)
        assert isinstance(report, SafetyReport)
    
    def test_combine_reports(self, safety_validator):
        """Test combining multiple safety reports."""
        report1 = SafetyReport(
            safe=True,
            confidence=0.9,
            violations=[]
        )
        
        report2 = SafetyReport(
            safe=False,
            confidence=0.6,
            violations=[SafetyViolationType.BIAS],
            reasons=["Bias detected"]
        )
        
        combined = safety_validator._combine_reports([report1, report2])
        
        assert combined.safe is False  # Any unsafe makes combined unsafe
        assert combined.confidence == 0.6  # Min confidence
        assert SafetyViolationType.BIAS in combined.violations
        assert len(combined.reasons) == 1
    
    def test_get_safety_stats(self, safety_validator, basic_action, basic_context):
        """Test getting safety statistics."""
        # Generate some validations
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
        # Add some history
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
        assert rate == 0.5  # Half safe, half unsafe
    
    def test_get_component_status(self, safety_validator):
        """Test getting component status."""
        status = safety_validator._get_component_status()
        
        assert isinstance(status, dict)
        assert 'tool_safety' in status
        assert 'compliance' in status
        assert 'domain_validators' in status
    
    def test_shutdown(self, safety_validator):
        """Test validator shutdown."""
        safety_validator.shutdown()
        
        assert safety_validator._shutdown is True
        
        # Shutdown should be idempotent
        safety_validator.shutdown()
    
    def test_shutdown_with_validation_history(self, safety_validator, basic_action, basic_context):
        """Test shutdown clears history."""
        safety_validator.validate_action_comprehensive(
            basic_action,
            basic_context,
            create_snapshot=False
        )
        
        assert len(safety_validator.validation_history) > 0
        
        safety_validator.shutdown()
        
        assert len(safety_validator.validation_history) == 0


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
            'resource_usage': {
                'energy_nJ': 1000
            }
        }
        
        context = {
            'energy_budget': 5000,
            'state': {},
            'action_log': []
        }
        
        report = validator.validate_action_comprehensive(
            action,
            context,
            create_snapshot=False
        )
        
        assert isinstance(report, SafetyReport)
        assert report.metadata is not None
        assert 'explanation' in report.metadata
        
        validator.shutdown()
    
    def test_constraint_and_explanation_integration(self):
        """Test constraint checking with explanation generation."""
        manager = ConstraintManager()
        explainer = EnhancedExplainabilityNode()
        
        # Add constraint
        manager.add_constraint(SafetyConstraint(
            name="test_constraint",
            type="soft",
            check_function=lambda a, c: (False, 0.5),
            description="Test constraint for integration"
        ))
        
        # Check constraints
        report = manager.check_constraints({'test': 'action'}, {})
        
        # Generate explanation
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
    
    def test_empty_action(self, safety_validator):
        """Test validation with empty action."""
        report = safety_validator.validate_action_comprehensive(
            {},
            {},
            create_snapshot=False
        )
        
        assert isinstance(report, SafetyReport)
    
    def test_null_values(self, safety_validator):
        """Test handling of null values."""
        action = {
            'type': None,
            'confidence': None
        }
        
        result = safety_validator.validate_variable_value('test', None)
        assert result['safe'] is True
    
    def test_extreme_values(self, safety_validator):
        """Test handling of extreme values."""
        result = safety_validator.validate_variable_value('test', 1e15)
        assert result['safe'] is False
        
        result = safety_validator.validate_variable_value('test', -1e15)
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
                {'type': 'test'},
                {},
                create_snapshot=False
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