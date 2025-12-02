# test_safety_module_integration.py - OPTIMIZED VERSION
"""
Comprehensive integration tests for VULCAN-AGI Safety Module.
OPTIMIZED: Uses module-scoped fixtures to avoid re-initializing expensive objects.

FIXES APPLIED (corrected version):
1. test_tool_safety_check: Added missing context fields (logic_valid, causal_graph_valid, 
   sample_size, temporal_paradox) required by ToolSafetyManager probabilistic contract preconditions.

2. test_value_alignment: Changed calculate_alignment() to check_alignment() which is the 
   correct public API method on ValueAlignmentSystem.

3. test_governance_integration: Changed ActionType.ANALYZE to ActionType.OPTIMIZE since
   ANALYZE doesn't exist in the ActionType enum.
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading

# Import all safety module components
from vulcan.safety.safety_types import (
    SafetyReport,
    SafetyViolationType,
    SafetyConstraint,
    RollbackSnapshot,
    ToolSafetyContract,
    ToolSafetyLevel,
    ComplianceStandard,
    SafetyConfig,
    SafetyMetrics,
    SafetyException,
    ActionType,
    Condition
)

from vulcan.safety.domain_validators import (
    CausalSafetyValidator,
    PredictionSafetyValidator,
    OptimizationSafetyValidator,
    DataProcessingSafetyValidator,
    ValidationResult,
    validator_registry
)

from vulcan.safety.tool_safety import (
    TokenBucket,
    ToolSafetyManager,
    ToolSafetyGovernor
)

from vulcan.safety.rollback_audit import (
    RollbackManager,
    AuditLogger,
    MemoryBoundedDeque
)

from vulcan.safety.governance_alignment import (
    GovernanceManager,
    ValueAlignmentSystem,
    HumanOversightInterface,
    GovernanceLevel,
    StakeholderType,
    HumanFeedback
)

from vulcan.safety.safety_validator import (
    ConstraintManager,
    EnhancedExplainabilityNode,
    ExplanationQualityScorer,
    EnhancedSafetyValidator
)


# ============================================================
# MODULE-SCOPED FIXTURES - KEY OPTIMIZATION
# ============================================================

@pytest.fixture(scope="module")
def temp_dir():
    """Create temporary directory for the entire test module."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="module")
def safety_config():
    """Module-scoped test safety configuration."""
    return SafetyConfig(
        enable_adversarial_testing=False,
        enable_compliance_checking=True,
        enable_bias_detection=False,
        enable_rollback=True,
        enable_audit_logging=True,
        enable_tool_safety=True,
        safety_thresholds={
            'uncertainty_max': 0.9,
            'identity_drift_max': 0.5,
            'bias_threshold': 0.2,
            'confidence_min': 0.6
        }
    )


@pytest.fixture(scope="module")
def sample_action():
    """Module-scoped sample action for testing."""
    return {
        'type': ActionType.OPTIMIZE,
        'confidence': 0.8,
        'uncertainty': 0.2,
        'resource_usage': {
            'energy_nJ': 100,
            'cpu': 30,
            'memory': 500
        },
        'safe': True,
        'id': 'test_action_001'
    }


@pytest.fixture(scope="module")
def sample_context():
    """Module-scoped sample context for testing."""
    return {
        'energy_budget': 1000,
        'resource_limits': {'cpu': 80, 'memory': 2000},
        'state': {'temperature': 25, 'pressure': 100, 'system_stable': True},
        'action_log': []
    }


# Module-scoped validators - created once
@pytest.fixture(scope="module")
def shared_causal_validator():
    """Module-scoped causal validator."""
    return CausalSafetyValidator()


@pytest.fixture(scope="module")
def shared_prediction_validator():
    """Module-scoped prediction validator."""
    return PredictionSafetyValidator()


@pytest.fixture(scope="module")
def shared_optimization_validator():
    """Module-scoped optimization validator."""
    return OptimizationSafetyValidator()


@pytest.fixture(scope="module")
def shared_data_validator():
    """Module-scoped data processing validator."""
    return DataProcessingSafetyValidator()


# Function-scoped fixtures for tests that need clean state
@pytest.fixture
def func_temp_dir():
    """Function-scoped temp directory."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


# ============================================================
# UNIT TESTS - SAFETY TYPES
# ============================================================

class TestSafetyTypes:
    """Test safety types and data structures."""
    
    def test_condition_evaluation(self):
        """Test Condition class evaluation."""
        cond = Condition('value', '>', 5, "Value must be > 5")
        assert cond.evaluate({'value': 10}) == True
        assert cond.evaluate({'value': 3}) == False
        
        cond = Condition('status', 'in', ['active', 'ready'], "Status check")
        assert cond.evaluate({'status': 'active'}) == True
        assert cond.evaluate({'status': 'inactive'}) == False
        
        cond = Condition('tags', 'contains', 'important', "Tag check")
        assert cond.evaluate({'tags': ['important', 'urgent']}) == True
        assert cond.evaluate({'tags': ['normal']}) == False
        
        cond = Condition('value', '>', 5, "Value must be > 5")
        assert cond.evaluate({'value': None}) == False
        assert cond.evaluate({}) == False
    
    def test_condition_serialization(self):
        """Test Condition serialization."""
        cond = Condition('temp', '<', 100, "Temperature limit")
        data = cond.to_dict()
        
        assert data['field'] == 'temp'
        assert data['operator'] == '<'
        assert data['value'] == 100
        
        restored = Condition.from_dict(data)
        assert restored.field == cond.field
        assert restored.operator == cond.operator
    
    def test_safety_report_creation(self):
        """Test SafetyReport creation and validation."""
        report = SafetyReport(
            safe=False,
            confidence=0.7,
            violations=[SafetyViolationType.ENERGY],
            reasons=["Energy budget exceeded"]
        )
        
        assert report.safe == False
        assert report.confidence == 0.7
        assert SafetyViolationType.ENERGY in report.violations
        assert report.audit_id is not None
    
    def test_safety_report_merge(self):
        """Test merging safety reports."""
        report1 = SafetyReport(safe=True, confidence=0.9, violations=[])
        report2 = SafetyReport(
            safe=False,
            confidence=0.6,
            violations=[SafetyViolationType.UNCERTAINTY],
            reasons=["High uncertainty"]
        )
        
        merged = report1.merge(report2)
        assert merged.safe == False
        assert merged.confidence == 0.6
        assert SafetyViolationType.UNCERTAINTY in merged.violations
    
    def test_rollback_snapshot_integrity(self):
        """Test RollbackSnapshot integrity verification."""
        snapshot = RollbackSnapshot(
            snapshot_id='test_snapshot',
            timestamp=time.time(),
            state={'var1': 10, 'var2': 20},
            action_log=[{'action': 'test'}],
            metadata={'reason': 'checkpoint'}
        )
        
        assert snapshot.verify_integrity() == True
        
        snapshot.state['var1'] = 999
        assert snapshot.verify_integrity() == False


# ============================================================
# UNIT TESTS - DOMAIN VALIDATORS
# ============================================================

class TestDomainValidators:
    """Test domain-specific validators."""
    
    def test_causal_edge_validation(self, shared_causal_validator):
        """Test causal edge validation."""
        result = shared_causal_validator.validate_causal_edge('A', 'B', 2.5)
        assert result.safe == True
        
        result = shared_causal_validator.validate_causal_edge('A', 'B', float('nan'))
        assert result.safe == False
        assert 'NaN' in result.reason
        
        result = shared_causal_validator.validate_causal_edge('A', 'B', 1000.0)
        assert result.safe == False
        
        result = shared_causal_validator.validate_causal_edge('A', 'A', 2.5)
        assert result.safe == False
    
    def test_causal_path_validation(self, shared_causal_validator):
        """Test causal path validation."""
        result = shared_causal_validator.validate_causal_path(['A', 'B', 'C'], [2.0, 1.5])
        assert result.safe == True
        
        result = shared_causal_validator.validate_causal_path(['A', 'B', 'C'], [2.0])
        assert result.safe == False
        
        result = shared_causal_validator.validate_causal_path(['A', 'B', 'C'], [5.0, 5.0])
        assert result.safe == False
        
        result = shared_causal_validator.validate_causal_path(['A', 'B', 'A'], [2.0, 2.0])
        assert result.safe == False
    
    def test_causal_graph_validation(self, shared_causal_validator):
        """Test causal graph validation."""
        adjacency = {
            'A': [('B', 2.0), ('C', 1.5)],
            'B': [('D', 1.0)],
            'C': [('D', 1.2)]
        }
        result = shared_causal_validator.validate_causal_graph(adjacency)
        assert result.safe == True
        
        adjacency_cyclic = {
            'A': [('B', 2.0)],
            'B': [('C', 1.5)],
            'C': [('A', 1.0)]
        }
        result = shared_causal_validator.validate_causal_graph(adjacency_cyclic)
        assert result.safe == False
    
    def test_prediction_validation(self, shared_prediction_validator):
        """Test prediction validation."""
        result = shared_prediction_validator.validate_prediction(10.0, 8.0, 12.0, 'temperature')
        assert result.safe == True
        
        result = shared_prediction_validator.validate_prediction(float('nan'), 8.0, 12.0, 'temperature')
        assert result.safe == False
        
        result = shared_prediction_validator.validate_prediction(10.0, 12.0, 8.0, 'temperature')
        assert result.safe == False
        
        result = shared_prediction_validator.validate_prediction(15.0, 8.0, 12.0, 'temperature')
        assert result.safe == False
    
    def test_optimization_params_validation(self, shared_optimization_validator):
        """Test optimization parameter validation."""
        params = {
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'learning_rate': 0.01,
            'bounds': {'x': (0, 10), 'y': (0, 20)}
        }
        result = shared_optimization_validator.validate_optimization_params(params)
        assert result.safe == True
        
        params = {'max_iterations': 100000, 'tolerance': 1e-6}
        result = shared_optimization_validator.validate_optimization_params(params)
        assert result.safe == False
        
        params = {'max_iterations': 1000, 'tolerance': 1e-6, 'learning_rate': 2.0}
        result = shared_optimization_validator.validate_optimization_params(params)
        assert result.safe == False
    
    def test_data_processing_validation(self, shared_data_validator):
        """Test data processing validation."""
        df_info = {
            'rows': 1000,
            'columns': 50,
            'memory_mb': 10,
            'missing_ratio': 0.1,
            'dtypes': {'col1': 'int64', 'col2': 'float64'}
        }
        result = shared_data_validator.validate_dataframe(df_info)
        assert result.safe == True
        
        df_info = {
            'rows': 20000000,
            'columns': 50,
            'memory_mb': 10,
            'missing_ratio': 0.1
        }
        result = shared_data_validator.validate_dataframe(df_info)
        assert result.safe == False


# ============================================================
# UNIT TESTS - TOOL SAFETY
# ============================================================

class TestToolSafety:
    """Test tool safety management."""
    
    def test_token_bucket_rate_limiting(self):
        """Test token bucket rate limiter."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        
        for _ in range(10):
            assert bucket.consume(1.0) == True
        
        assert bucket.consume(1.0) == False
        
        time.sleep(0.2)
        assert bucket.consume(1.0) == True
        
        bucket.shutdown()
    
    def test_tool_safety_contract_validation(self):
        """Test tool safety contract validation."""
        manager = ToolSafetyManager()
        
        assert 'probabilistic' in manager.contracts
        assert 'symbolic' in manager.contracts
        assert 'causal' in manager.contracts
        
        contract = manager.contracts['probabilistic']
        context = {
            'confidence': 0.7,
            'data_quality': 0.8,
            'corrupted_data': False
        }
        valid, failures = contract.validate_preconditions(context)
        assert valid == True
        
        context = {'confidence': 0.2, 'data_quality': 0.8, 'corrupted_data': False}
        valid, failures = contract.validate_preconditions(context)
        assert valid == False
        
        manager.shutdown()
    
    def test_tool_safety_check(self):
        """Test tool safety checking.
        
        Note: ToolSafetyManager requires additional context fields for probabilistic
        tool contracts including logic_valid, causal_graph_valid, sample_size, temporal_paradox.
        """
        manager = ToolSafetyManager()
        
        # Reset rate limiter to avoid timing issues
        if hasattr(manager, 'rate_limiters') and 'probabilistic' in manager.rate_limiters:
            manager.rate_limiters['probabilistic'].tokens = manager.rate_limiters['probabilistic'].capacity
        
        context = {
            'confidence': 0.8,  # Increased from 0.7
            'data_quality': 0.8,
            'corrupted_data': False,
            'adversarial_detected': False,
            'system_overload': False,
            'estimated_resources': {'memory_mb': 100, 'time_ms': 1000},
            # Additional required fields for probabilistic contract preconditions
            'logic_valid': True,
            'causal_graph_valid': True,
            'sample_size': 100,
            'temporal_paradox': False
        }
        
        safe, report = manager.check_tool_safety('probabilistic', context)
        assert safe == True, f"Expected safe=True but got safe={safe}, report={report}"
        
        context['adversarial_detected'] = True
        safe, report = manager.check_tool_safety('probabilistic', context)
        assert safe == False
        
        manager.shutdown()
    
    def test_tool_safety_governor(self):
        """Test tool safety governor."""
        governor = ToolSafetyGovernor()
        
        request = {'confidence': 0.8, 'constraints': {}, 'risk_approved': False}
        tools = ['probabilistic', 'symbolic']
        allowed, result = governor.govern_tool_selection(request, tools)
        
        assert isinstance(allowed, list)
        assert 'allowed_tools' in result
        
        governor.trigger_emergency_stop("Test emergency")
        assert governor.emergency_stop == True
        
        allowed, result = governor.govern_tool_selection(request, tools)
        assert len(allowed) == 0
        
        governor.clear_emergency_stop('test_admin')
        assert governor.emergency_stop == False
        
        governor.shutdown()


# ============================================================
# UNIT TESTS - ROLLBACK AND AUDIT
# ============================================================

class TestRollbackAudit:
    """Test rollback and audit logging."""
    
    def test_memory_bounded_deque(self):
        """Test memory-bounded deque."""
        deque = MemoryBoundedDeque(max_size_mb=0.001)
        
        for i in range(100):
            deque.append({'data': f'item_{i}', 'value': i})
        
        assert len(deque) < 100
        assert deque.get_memory_usage_mb() <= 0.001
        
        deque.clear()
        assert len(deque) == 0
    
    def test_rollback_snapshot_creation(self, func_temp_dir):
        """Test snapshot creation and persistence."""
        manager = RollbackManager(
            max_snapshots=10,
            config={'storage_path': func_temp_dir}
        )
        
        state = {'temperature': 25, 'pressure': 100}
        action_log = [{'action': 'test', 'timestamp': time.time()}]
        
        snapshot_id = manager.create_snapshot(state, action_log)
        assert snapshot_id is not None
        assert len(manager.snapshots) == 1
        
        history = manager.get_snapshot_history()
        assert len(history) == 1
        
        manager.shutdown()
    
    def test_rollback_execution(self, func_temp_dir):
        """Test rollback to snapshot."""
        manager = RollbackManager(config={'storage_path': func_temp_dir})
        
        state = {'value': 100}
        action_log = [{'action': 'increase_value'}]
        snapshot_id = manager.create_snapshot(state, action_log)
        
        result = manager.rollback(snapshot_id, reason='test_rollback')
        assert result is not None
        assert result['state']['value'] == 100
        
        metrics = manager.get_metrics()
        assert metrics['total_rollbacks'] == 1
        
        manager.shutdown()
    
    def test_audit_logging(self, func_temp_dir):
        """Test audit logging."""
        logger = AuditLogger(
            log_path=str(Path(func_temp_dir) / 'audit'),
            config={'redact_sensitive': True}
        )
        
        decision = {'action': 'test', 'confidence': 0.8}
        report = SafetyReport(safe=True, confidence=0.9, violations=[])
        
        entry_id = logger.log_safety_decision(decision, report)
        assert entry_id is not None
        
        event_id = logger.log_event('test_event', {'data': 'test'}, severity='info')
        assert event_id is not None
        
        metrics = logger.get_metrics()
        assert metrics['total_entries'] >= 2
        
        logger.shutdown()


# ============================================================
# UNIT TESTS - GOVERNANCE AND ALIGNMENT
# ============================================================

class TestGovernanceAlignment:
    """Test governance and alignment systems."""
    
    def test_governance_manager_initialization(self, func_temp_dir):
        """Test governance manager initialization."""
        config = {
            'db_path': str(Path(func_temp_dir) / 'governance.db'),
            'max_active_decisions': 100
        }
        manager = GovernanceManager(config=config)
        
        assert 'autonomous_default' in manager.policies
        assert 'human_supervised' in manager.policies
        assert 'safety_critical' in manager.policies
        
        manager.shutdown()
    
    def test_approval_request(self, func_temp_dir):
        """Test approval request."""
        config = {'db_path': str(Path(func_temp_dir) / 'governance.db')}
        manager = GovernanceManager(config=config)
        
        action = {
            'type': ActionType.OPTIMIZE,
            'risk_score': 0.3,
            'safety_score': 0.8
        }
        
        result = manager.request_approval(action)
        
        assert 'approved' in result
        assert 'decision_id' in result
        assert 'policy_applied' in result
        
        manager.shutdown()
    
    def test_value_alignment(self, func_temp_dir):
        """Test value alignment system.
        
        Note: ValueAlignmentSystem uses check_alignment() not calculate_alignment().
        """
        config = {'db_path': str(Path(func_temp_dir) / 'alignment.db')}
        system = ValueAlignmentSystem(config=config)
        
        action = {'type': ActionType.OPTIMIZE, 'impact': 'positive'}
        context = {'state': {'stable': True}}
        
        # Use check_alignment() which is the correct public API method
        alignment = system.check_alignment(action, context)
        
        assert 'alignment_score' in alignment
        assert 0.0 <= alignment['alignment_score'] <= 1.0
        
        system.shutdown()


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_safety_validation(self, func_temp_dir, safety_config, sample_action, sample_context):
        """Test complete safety validation pipeline."""
        safety_config.rollback_config['storage_path'] = str(Path(func_temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(func_temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        report = validator.validate_action_comprehensive(
            sample_action,
            sample_context,
            create_snapshot=False
        )
        
        assert isinstance(report, SafetyReport)
        assert hasattr(report, 'safe')
        assert hasattr(report, 'confidence')
        
        validator.shutdown()
    
    def test_governance_integration(self, func_temp_dir):
        """Test governance system integration.
        
        Note: ActionType.ANALYZE doesn't exist - using ActionType.OPTIMIZE instead.
        """
        config = {'db_path': str(Path(func_temp_dir) / 'governance.db')}
        manager = GovernanceManager(config=config)
        
        action = {
            'type': ActionType.OPTIMIZE,  # Changed from ANALYZE which doesn't exist
            'risk_score': 0.2,
            'safety_score': 0.9
        }
        
        result = manager.request_approval(action)
        
        assert result['approved'] == True
        
        manager.shutdown()


# ============================================================
# STRESS TESTS
# ============================================================

class TestStress:
    """Stress tests for performance and resource limits."""
    
    def test_high_frequency_validations(self, func_temp_dir, safety_config):
        """Test high-frequency validation requests."""
        safety_config.rollback_config['storage_path'] = str(Path(func_temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(func_temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        start_time = time.time()
        count = 50
        
        for i in range(count):
            action = {'type': ActionType.OPTIMIZE, 'id': f'stress_action_{i}', 'confidence': 0.8}
            context = {'state': {}}
            report = validator.validate_action_comprehensive(action, context)
            assert isinstance(report, SafetyReport)
        
        elapsed = time.time() - start_time
        throughput = count / elapsed
        
        print(f"\nValidation throughput: {throughput:.2f} validations/second")
        assert throughput > 1.0
        
        validator.shutdown()
    
    def test_constraint_manager_scale(self):
        """Test constraint manager with many constraints."""
        manager = ConstraintManager()
        
        for i in range(50):
            constraint = SafetyConstraint(
                name=f'constraint_{i}',
                type='soft',
                check_function=lambda a, c, i=i: (a.get('value', 0) < 100 + i, 0.9),
                threshold=0.0,
                priority=i
            )
            manager.add_constraint(constraint)
        
        action = {'value': 50}
        context = {}
        
        start = time.time()
        report = manager.check_constraints(action, context)
        elapsed = time.time() - start
        
        print(f"\nConstraint check time (50 constraints): {elapsed*1000:.2f}ms")
        assert elapsed < 1.0
        
        manager.shutdown()


# ============================================================
# ERROR HANDLING TESTS
# ============================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_validator_with_missing_components(self):
        """Test validator gracefully handles missing components."""
        config = SafetyConfig(
            enable_rollback=False,
            enable_audit_logging=False,
            enable_tool_safety=False
        )
        
        validator = EnhancedSafetyValidator(config=config)
        
        action = {'type': ActionType.OPTIMIZE, 'confidence': 0.8}
        context = {'state': {}}
        
        report = validator.validate_action_comprehensive(action, context)
        assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    def test_invalid_action_handling(self, func_temp_dir, safety_config):
        """Test handling of invalid action data."""
        safety_config.rollback_config['storage_path'] = str(Path(func_temp_dir) / 'rollback')
        validator = EnhancedSafetyValidator(config=safety_config)
        
        report = validator.validate_action_comprehensive({}, {})
        assert isinstance(report, SafetyReport)
        
        action = {'type': ActionType.OPTIMIZE, 'confidence': float('nan')}
        report = validator.validate_action_comprehensive(action, {})
        assert isinstance(report, SafetyReport)
        
        validator.shutdown()


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
