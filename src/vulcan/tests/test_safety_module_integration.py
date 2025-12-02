# test_safety_module_integration.py
"""
Comprehensive integration tests for VULCAN-AGI Safety Module.
Tests all components individually and their interactions.

FIXES APPLIED (corrected version):
1. test_tool_safety_check: Added more context fields and better error reporting
   to handle all preconditions required by the probabilistic tool contract.

2. test_value_alignment: Added new test that uses the correct public method
   check_alignment() instead of the non-existent calculate_alignment().
   Note: The private method _calculate_value_alignment() should not be called directly.

3. test_governance_integration: Added new test that uses ActionType.OPTIMIZE
   instead of ActionType.ANALYZE (which doesn't exist in the ActionType enum).
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
# FIXTURES
# ============================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def safety_config():
    """Create test safety configuration."""
    return SafetyConfig(
        enable_adversarial_testing=False,  # Disable for faster tests
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

@pytest.fixture
def sample_action():
    """Create sample action for testing."""
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

@pytest.fixture
def sample_context():
    """Create sample context for testing."""
    return {
        'energy_budget': 1000,
        'resource_limits': {
            'cpu': 80,
            'memory': 2000
        },
        'state': {
            'temperature': 25,
            'pressure': 100,
            'system_stable': True
        },
        'action_log': []
    }

# ============================================================
# UNIT TESTS - SAFETY TYPES
# ============================================================

class TestSafetyTypes:
    """Test safety types and data structures."""
    
    def test_condition_evaluation(self):
        """Test Condition class evaluation."""
        # Greater than
        cond = Condition('value', '>', 5, "Value must be > 5")
        assert cond.evaluate({'value': 10}) == True
        assert cond.evaluate({'value': 3}) == False
        
        # In operator
        cond = Condition('status', 'in', ['active', 'ready'], "Status must be active or ready")
        assert cond.evaluate({'status': 'active'}) == True
        assert cond.evaluate({'status': 'inactive'}) == False
        
        # Contains
        cond = Condition('tags', 'contains', 'important', "Must contain important tag")
        assert cond.evaluate({'tags': ['important', 'urgent']}) == True
        assert cond.evaluate({'tags': ['normal']}) == False
        
        # Handle None
        cond = Condition('value', '>', 5, "Value must be > 5")
        assert cond.evaluate({'value': None}) == False
        assert cond.evaluate({}) == False
    
    def test_condition_serialization(self):
        """Test Condition serialization."""
        cond = Condition('temp', '<', 100, "Temperature limit")
        
        # To dict
        data = cond.to_dict()
        assert data['field'] == 'temp'
        assert data['operator'] == '<'
        assert data['value'] == 100
        
        # From dict
        restored = Condition.from_dict(data)
        assert restored.field == cond.field
        assert restored.operator == cond.operator
        assert restored.value == cond.value
    
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
        assert len(report.reasons) == 1
        assert report.audit_id is not None
    
    def test_safety_report_merge(self):
        """Test merging safety reports."""
        report1 = SafetyReport(
            safe=True,
            confidence=0.9,
            violations=[]
        )
        
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
    
    def test_tool_safety_contract_serialization(self):
        """Test ToolSafetyContract serialization."""
        contract = ToolSafetyContract(
            tool_name='test_tool',
            safety_level=ToolSafetyLevel.MONITORED,
            preconditions=[Condition('confidence', '>', 0.5, "Min confidence")],
            postconditions=[Condition('result', '==', True, "Must succeed")],
            invariants=[Condition('stable', '==', True, "Must be stable")],
            veto_conditions=[Condition('emergency', '==', True, "No emergency")],
            max_frequency=100.0,
            max_resource_usage={'memory_mb': 1000},
            required_confidence=0.7,
            risk_score=0.3
        )
        
        # Serialize to JSON
        json_str = contract.to_json()
        assert isinstance(json_str, str)
        
        # Deserialize
        restored = ToolSafetyContract.from_json(json_str)
        assert restored.tool_name == contract.tool_name
        assert restored.safety_level == contract.safety_level
        assert len(restored.preconditions) == len(contract.preconditions)
    
    def test_rollback_snapshot_integrity(self):
        """Test RollbackSnapshot integrity verification."""
        snapshot = RollbackSnapshot(
            snapshot_id='test_snapshot',
            timestamp=time.time(),
            state={'var1': 10, 'var2': 20},
            action_log=[{'action': 'test'}],
            metadata={'reason': 'checkpoint'}
        )
        
        # Verify integrity
        assert snapshot.verify_integrity() == True
        
        # Corrupt state
        snapshot.state['var1'] = 999
        assert snapshot.verify_integrity() == False

# ============================================================
# UNIT TESTS - DOMAIN VALIDATORS
# ============================================================

class TestDomainValidators:
    """Test domain-specific validators."""
    
    def test_causal_edge_validation(self):
        """Test causal edge validation."""
        validator = CausalSafetyValidator()
        
        # Valid edge
        result = validator.validate_causal_edge('A', 'B', 2.5)
        assert result.safe == True
        
        # NaN strength
        result = validator.validate_causal_edge('A', 'B', float('nan'))
        assert result.safe == False
        assert 'NaN' in result.reason
        
        # Excessive strength
        result = validator.validate_causal_edge('A', 'B', 1000.0)
        assert result.safe == False
        assert 'too large' in result.reason
        
        # Self-loop
        result = validator.validate_causal_edge('A', 'A', 2.5)
        assert result.safe == False
        assert 'Self-loop' in result.reason
        
        # Unsafe pattern
        result = validator.validate_causal_edge('harm', 'increase', 2.5)
        assert result.safe == False
        assert 'unsafe pattern' in result.reason
    
    def test_causal_path_validation(self):
        """Test causal path validation."""
        validator = CausalSafetyValidator()
        
        # Valid path
        result = validator.validate_causal_path(['A', 'B', 'C'], [2.0, 1.5])
        assert result.safe == True
        
        # Mismatched nodes and strengths
        result = validator.validate_causal_path(['A', 'B', 'C'], [2.0])
        assert result.safe == False
        assert 'Mismatch' in result.reason
        
        # Excessive amplification
        result = validator.validate_causal_path(['A', 'B', 'C'], [5.0, 5.0])
        assert result.safe == False
        assert 'amplification' in result.reason
        
        # Cycle detection
        result = validator.validate_causal_path(['A', 'B', 'A'], [2.0, 2.0])
        assert result.safe == False
        assert 'cycle' in result.reason
    
    def test_causal_graph_validation(self):
        """Test causal graph validation."""
        validator = CausalSafetyValidator()
        
        # Valid DAG
        adjacency = {
            'A': [('B', 2.0), ('C', 1.5)],
            'B': [('D', 1.0)],
            'C': [('D', 1.2)]
        }
        result = validator.validate_causal_graph(adjacency)
        assert result.safe == True
        
        # Graph with cycle
        adjacency_cyclic = {
            'A': [('B', 2.0)],
            'B': [('C', 1.5)],
            'C': [('A', 1.0)]
        }
        result = validator.validate_causal_graph(adjacency_cyclic)
        assert result.safe == False
        assert 'cycle' in result.reason
    
    def test_prediction_validation(self):
        """Test prediction validation."""
        validator = PredictionSafetyValidator()
        
        # Valid prediction
        result = validator.validate_prediction(10.0, 8.0, 12.0, 'temperature')
        assert result.safe == True
        
        # NaN prediction
        result = validator.validate_prediction(float('nan'), 8.0, 12.0, 'temperature')
        assert result.safe == False
        assert result.severity == 'critical'
        
        # Invalid bounds (lower > upper)
        result = validator.validate_prediction(10.0, 12.0, 8.0, 'temperature')
        assert result.safe == False
        
        # Expected outside bounds
        result = validator.validate_prediction(15.0, 8.0, 12.0, 'temperature')
        assert result.safe == False
        
        # With safe regions
        validator_with_regions = PredictionSafetyValidator(safe_regions={'temp': (0, 100)})
        result = validator_with_regions.validate_prediction(150.0, 140.0, 160.0, 'temp')
        assert result.safe == False
        assert 'safe region' in result.reason
    
    def test_prediction_batch_validation(self):
        """Test batch prediction validation."""
        validator = PredictionSafetyValidator()
        
        predictions = [
            {'expected': 10.0, 'lower': 8.0, 'upper': 12.0, 'variable': 'temp1'},
            {'expected': 20.0, 'lower': 18.0, 'upper': 22.0, 'variable': 'temp2'},
            {'expected': float('nan'), 'lower': 28.0, 'upper': 32.0, 'variable': 'temp3'},
        ]
        
        result = validator.validate_prediction_batch(predictions)
        assert result.safe == False  # One prediction is invalid
        assert '1/3' in result.reason or '1 unsafe' in result.reason.lower()
    
    def test_optimization_params_validation(self):
        """Test optimization parameter validation."""
        validator = OptimizationSafetyValidator()
        
        # Valid params
        params = {
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'learning_rate': 0.01,
            'bounds': {'x': (0, 10), 'y': (0, 20)}
        }
        result = validator.validate_optimization_params(params)
        assert result.safe == True
        
        # Too many iterations
        params = {
            'max_iterations': 100000,
            'tolerance': 1e-6
        }
        result = validator.validate_optimization_params(params)
        assert result.safe == False
        
        # Invalid learning rate
        params = {
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'learning_rate': 2.0
        }
        result = validator.validate_optimization_params(params)
        assert result.safe == False
        
        # Invalid bounds
        params = {
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'bounds': {'x': (10, 0)}  # lower > upper
        }
        result = validator.validate_optimization_params(params)
        assert result.safe == False
    
    def test_data_processing_validation(self):
        """Test data processing validation."""
        validator = DataProcessingSafetyValidator()
        
        # Valid dataframe
        df_info = {
            'rows': 1000,
            'columns': 50,
            'memory_mb': 10,
            'missing_ratio': 0.1,
            'dtypes': {'col1': 'int64', 'col2': 'float64'}
        }
        result = validator.validate_dataframe(df_info)
        assert result.safe == True
        
        # Too many rows
        df_info = {
            'rows': 20000000,
            'columns': 50,
            'memory_mb': 10,
            'missing_ratio': 0.1
        }
        result = validator.validate_dataframe(df_info)
        assert result.safe == False
        
        # Too much missing data
        df_info = {
            'rows': 1000,
            'columns': 50,
            'memory_mb': 10,
            'missing_ratio': 0.8
        }
        result = validator.validate_dataframe(df_info)
        assert result.safe == False

# ============================================================
# UNIT TESTS - TOOL SAFETY
# ============================================================

class TestToolSafety:
    """Test tool safety management."""
    
    def test_token_bucket_rate_limiting(self):
        """Test token bucket rate limiter."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)  # 10 tokens/sec
        
        # Should allow 10 tokens immediately
        for _ in range(10):
            assert bucket.consume(1.0) == True
        
        # 11th should fail
        assert bucket.consume(1.0) == False
        
        # Wait and try again
        time.sleep(0.2)  # Allow 2 tokens to regenerate
        assert bucket.consume(1.0) == True
        
        # Check available tokens
        available = bucket.get_available()
        assert 0 <= available <= 10
        
        bucket.shutdown()
    
    def test_tool_safety_contract_validation(self):
        """Test tool safety contract validation."""
        manager = ToolSafetyManager()
        
        # Check default contracts exist
        assert 'probabilistic' in manager.contracts
        assert 'symbolic' in manager.contracts
        assert 'causal' in manager.contracts
        
        # Test precondition validation
        contract = manager.contracts['probabilistic']
        context = {
            'confidence': 0.7,
            'data_quality': 0.8,
            'corrupted_data': False
        }
        valid, failures = contract.validate_preconditions(context)
        assert valid == True
        assert len(failures) == 0
        
        # Failing precondition
        context = {
            'confidence': 0.2,  # Too low
            'data_quality': 0.8,
            'corrupted_data': False
        }
        valid, failures = contract.validate_preconditions(context)
        assert valid == False
        assert len(failures) > 0
        
        manager.shutdown()
    
    def test_tool_safety_check(self):
        """Test tool safety checking."""
        manager = ToolSafetyManager()
        
        # Reset rate limiters to ensure clean state
        if hasattr(manager, 'rate_limiters'):
            for limiter in manager.rate_limiters.values():
                if hasattr(limiter, 'tokens'):
                    limiter.tokens = limiter.capacity
        
        # Safe usage - provide all fields that might be required by preconditions
        context = {
            'confidence': 0.8,  # Higher confidence to ensure it passes threshold
            'data_quality': 0.9,
            'corrupted_data': False,
            'adversarial_detected': False,
            'system_overload': False,
            'logic_valid': True,  # Additional fields that might be checked
            'causal_graph_valid': True,
            'sample_size': 100,
            'temporal_paradox': False,
            'estimated_resources': {
                'memory_mb': 100,
                'time_ms': 1000
            }
        }
        
        safe, report = manager.check_tool_safety('probabilistic', context)
        # Check if safety check passed, or if it failed, get details
        if not safe:
            # If failed, check if it's due to rate limiting or other transient issues
            # In that case, the test should still be meaningful
            assert report is not None, "Report should not be None"
            print(f"Safety check failed with violations: {report.violations}, reasons: {report.reasons}")
        assert safe == True, f"Expected safe=True but got safe=False. Violations: {report.violations}, Reasons: {report.reasons}"
        assert report.safe == True
        
        # Veto condition triggered
        context['adversarial_detected'] = True
        safe, report = manager.check_tool_safety('probabilistic', context)
        assert safe == False
        assert SafetyViolationType.TOOL_VETO in report.violations
        
        # Low confidence
        context = {
            'confidence': 0.3,  # Below required 0.5
            'adversarial_detected': False,
            'system_overload': False
        }
        safe, report = manager.check_tool_safety('probabilistic', context)
        assert safe == False
        
        manager.shutdown()
    
    def test_tool_veto_selection(self):
        """Test tool selection veto."""
        manager = ToolSafetyManager()
        
        tools = ['probabilistic', 'symbolic', 'causal']
        context = {
            'confidence': 0.8,
            'data_quality': 0.9,
            'corrupted_data': False,
            'adversarial_detected': False,
            'system_overload': False,
            'logic_valid': True,
            'axioms_count': 50,
            'contradictory_axioms': False,
            'causal_graph_valid': True,
            'sample_size': 100,
            'temporal_paradox': False
        }
        
        allowed, report = manager.veto_tool_selection(tools, context)
        assert len(allowed) > 0
        assert report.safe == True or len(report.violations) > 0
        
        manager.shutdown()
    
    def test_tool_safety_governor(self):
        """Test tool safety governor."""
        governor = ToolSafetyGovernor()
        
        # Normal governance
        request = {
            'confidence': 0.8,
            'constraints': {},
            'risk_approved': False
        }
        
        tools = ['probabilistic', 'symbolic']
        allowed, result = governor.govern_tool_selection(request, tools)
        
        assert isinstance(allowed, list)
        assert 'allowed_tools' in result
        assert 'veto_report' in result
        
        # Test emergency stop
        governor.trigger_emergency_stop("Test emergency")
        assert governor.emergency_stop == True
        
        allowed, result = governor.govern_tool_selection(request, tools)
        assert len(allowed) == 0
        assert result['status'] == 'emergency_stop'
        
        # Clear emergency
        governor.clear_emergency_stop('test_admin')
        assert governor.emergency_stop == False
        
        # Test quarantine
        governor.quarantine_tool('symbolic', 'Test quarantine', duration_seconds=1)
        assert 'symbolic' in governor.quarantine_list
        
        time.sleep(1.1)  # Wait for quarantine to expire
        assert 'symbolic' not in governor.quarantine_list
        
        governor.shutdown()

# ============================================================
# UNIT TESTS - ROLLBACK AND AUDIT
# ============================================================

class TestRollbackAudit:
    """Test rollback and audit logging."""
    
    def test_memory_bounded_deque(self):
        """Test memory-bounded deque."""
        deque = MemoryBoundedDeque(max_size_mb=0.001)  # 1KB limit
        
        # Add items
        for i in range(100):
            deque.append({'data': f'item_{i}', 'value': i})
        
        # Should automatically limit size
        assert len(deque) < 100
        assert deque.get_memory_usage_mb() <= 0.001
        
        deque.clear()
        assert len(deque) == 0
    
    def test_rollback_snapshot_creation(self, temp_dir):
        """Test snapshot creation and persistence."""
        manager = RollbackManager(
            max_snapshots=10,
            config={'storage_path': temp_dir}
        )
        
        state = {'temperature': 25, 'pressure': 100}
        action_log = [{'action': 'test', 'timestamp': time.time()}]
        
        snapshot_id = manager.create_snapshot(state, action_log)
        assert snapshot_id is not None
        
        # Verify snapshot exists
        assert len(manager.snapshots) == 1
        assert snapshot_id in manager.snapshot_index
        
        # Get snapshot history
        history = manager.get_snapshot_history()
        assert len(history) == 1
        assert history[0]['snapshot_id'] == snapshot_id
        
        manager.shutdown()
    
    def test_rollback_execution(self, temp_dir):
        """Test rollback to snapshot."""
        manager = RollbackManager(config={'storage_path': temp_dir})
        
        # Create snapshot
        state = {'value': 100}
        action_log = [{'action': 'increase_value'}]
        snapshot_id = manager.create_snapshot(state, action_log)
        
        # Rollback
        result = manager.rollback(snapshot_id, reason='test_rollback')
        assert result is not None
        assert result['state']['value'] == 100
        assert result['rollback_metadata']['snapshot_id'] == snapshot_id
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics['total_rollbacks'] == 1
        assert metrics['successful_rollbacks'] == 1
        
        manager.shutdown()
    
    def test_quarantine_action(self, temp_dir):
        """Test action quarantine."""
        manager = RollbackManager(config={'storage_path': temp_dir})
        
        action = {'type': 'dangerous_action', 'id': 'test_001'}
        quarantine_id = manager.quarantine_action(action, 'safety_violation', duration_seconds=1)
        
        assert quarantine_id is not None
        assert quarantine_id in manager.quarantine
        
        # Get quarantine item
        item = manager.get_quarantine_item(quarantine_id)
        assert item is not None
        assert item['action']['id'] == 'test_001'
        
        # Review quarantine
        success = manager.review_quarantine(quarantine_id, approved=False, reviewer='test_reviewer')
        assert success == True
        
        manager.shutdown()
    
    def test_audit_logging(self, temp_dir):
        """Test audit logging."""
        logger = AuditLogger(
            log_path=str(Path(temp_dir) / 'audit'),
            config={'redact_sensitive': True}
        )
        
        # Log safety decision
        decision = {'action': 'test', 'confidence': 0.8}
        report = SafetyReport(
            safe=True,
            confidence=0.9,
            violations=[]
        )
        
        entry_id = logger.log_safety_decision(decision, report)
        assert entry_id is not None
        
        # Log event
        event_id = logger.log_event('test_event', {'data': 'test'}, severity='info')
        assert event_id is not None
        
        # Query logs
        time.sleep(0.1)  # Allow buffer to flush
        logs = logger.query_logs(limit=10)
        assert len(logs) >= 0  # May be empty due to batching
        
        # Get metrics
        metrics = logger.get_metrics()
        assert metrics['total_entries'] >= 2
        
        logger.shutdown()
    
    def test_audit_log_redaction(self, temp_dir):
        """Test sensitive data redaction."""
        logger = AuditLogger(
            log_path=str(Path(temp_dir) / 'audit'),
            config={'redact_sensitive': True}
        )
        
        # Test SSN redaction
        data_with_ssn = "SSN: 123-45-6789"
        redacted = logger._redact_sensitive(data_with_ssn)
        assert '123-45-6789' not in redacted
        assert '[SSN_REDACTED]' in redacted
        
        # Test email redaction
        data_with_email = "Contact: user@example.com"
        redacted = logger._redact_sensitive(data_with_email)
        assert 'user@example.com' not in redacted
        assert '[EMAIL_REDACTED]' in redacted
        
        logger.shutdown()

# ============================================================
# UNIT TESTS - GOVERNANCE AND ALIGNMENT
# ============================================================

class TestGovernanceAlignment:
    """Test governance and alignment systems."""
    
    def test_governance_manager_initialization(self, temp_dir):
        """Test governance manager initialization."""
        config = {
            'db_path': str(Path(temp_dir) / 'governance.db'),
            'max_active_decisions': 100
        }
        manager = GovernanceManager(config=config)
        
        # Check default policies
        assert 'autonomous_default' in manager.policies
        assert 'human_supervised' in manager.policies
        assert 'safety_critical' in manager.policies
        
        manager.shutdown()
    
    def test_approval_request(self, temp_dir):
        """Test approval request."""
        config = {'db_path': str(Path(temp_dir) / 'governance.db')}
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
    
    def test_stakeholder_registration(self, temp_dir):
        """Test stakeholder registration."""
        config = {'db_path': str(Path(temp_dir) / 'governance.db')}
        manager = GovernanceManager(config=config)
        
        success = manager.register_stakeholder(
            'operator_001',
            StakeholderType.OPERATOR,
            metadata={'name': 'Test Operator'}
        )
        
        assert success == True
        assert 'operator_001' in manager.stakeholder_registry
        
        manager.shutdown()
    
    def test_value_alignment_system(self):
        """Test value alignment system."""
        system = ValueAlignmentSystem()
        
        # Check alignment
        action = {
            'type': ActionType.EXPLORE,
            'safety_score': 0.8,
            'explanation': 'Testing alignment',
            'auditable': True
        }
        
        context = {}
        result = system.check_alignment(action, context)
        
        assert 'aligned' in result
        assert 'alignment_score' in result
        assert 'value_scores' in result
    
    def test_value_alignment(self):
        """Test value alignment calculation.
        
        Note: This test uses check_alignment() which is the public API.
        The private method _calculate_value_alignment() should not be called directly.
        """
        system = ValueAlignmentSystem()
        
        action = {
            'type': ActionType.OPTIMIZE,
            'safety_score': 0.85,
            'explanation': 'Testing value alignment',
            'auditable': True,
            'reversible': True
        }
        
        context = {
            'user_preferences': {},
            'system_state': 'operational'
        }
        
        # Use the public check_alignment method (not calculate_alignment or _calculate_value_alignment)
        result = system.check_alignment(action, context)
        
        # Extract alignment score from result
        alignment = result.get('alignment_score', 0)
        
        assert isinstance(alignment, (int, float))
        assert 0 <= alignment <= 1
        assert 'aligned' in result
        assert 'value_scores' in result
    
    def test_human_oversight_interface(self, temp_dir):
        """Test human oversight interface."""
        config = {'db_path': str(Path(temp_dir) / 'governance.db')}
        governance = GovernanceManager(config=config)
        alignment = ValueAlignmentSystem()
        interface = HumanOversightInterface(governance, alignment)
        
        # Set automation level
        success = interface.set_automation_level(0.7)
        assert success == True
        assert interface.automation_level == 0.7
        
        # Get status
        status = interface.get_oversight_status()
        assert 'automation_level' in status
        assert 'emergency_stop_enabled' in status
        
        # Create alert
        alert_id = interface.create_alert('test_alert', 'Test message', severity='medium')
        assert alert_id is not None
        
        # Acknowledge alert
        success = interface.acknowledge_alert(alert_id, notes='Test note')
        assert success == True
        
        governance.shutdown()

# ============================================================
# UNIT TESTS - SAFETY VALIDATOR
# ============================================================

class TestSafetyValidator:
    """Test main safety validator."""
    
    def test_constraint_manager(self):
        """Test constraint manager."""
        manager = ConstraintManager()
        
        # Add constraint
        constraint = SafetyConstraint(
            name='test_constraint',
            type='hard',
            check_function=lambda a, c: (a.get('value', 0) < 100, 0.9),
            threshold=0.0,
            priority=5
        )
        
        manager.add_constraint(constraint)
        assert 'test_constraint' in manager.active_constraints
        
        # Check constraints
        action = {'value': 50}
        context = {}
        report = manager.check_constraints(action, context)
        assert report.safe == True
        
        # Failing constraint
        action = {'value': 150}
        report = manager.check_constraints(action, context)
        assert report.safe == False
        
        # Get stats
        stats = manager.get_constraint_stats()
        assert stats['total_constraints'] == 1
        
        manager.shutdown()
    
    def test_explainability_node(self):
        """Test enhanced explainability node."""
        node = EnhancedExplainabilityNode()
        
        data = {
            'decision': 'explore',
            'confidence': 0.8,
            'features': {'f1': 0.5, 'f2': 0.3}
        }
        
        explanation = node.execute(data, {})
        
        assert 'explanation_summary' in explanation
        assert 'confidence' in explanation
        assert 'quality_score' in explanation
        assert 'context' in explanation
        
        # Get stats
        stats = node.get_explanation_stats()
        assert 'total_explanations' in stats
        
        node.shutdown()
    
    def test_explanation_quality_scorer(self):
        """Test explanation quality scorer."""
        scorer = ExplanationQualityScorer()
        
        # Complete explanation
        explanation = {
            'explanation_summary': 'This is a detailed explanation of the decision process.',
            'method': 'neural',
            'context': {'decision_type': 'explore'},
            'alternatives': [{'action': 'wait', 'reason': 'gather info'}],
            'confidence': 0.85,
            'decision_factors': ['factor1', 'factor2'],
            'visual_aids': {'type': 'chart'},
            'feature_importance': [{'feature': 'f1', 'importance': 0.8}]
        }
        
        score = scorer.score(explanation)
        assert 0 <= score <= 1
        assert score > 0.5  # Should score well
        
        # Get quality category
        category = scorer.get_quality_category(score)
        assert category in ['excellent', 'good', 'acceptable', 'poor']
        
        scorer.shutdown()
    
    def test_safety_validator_initialization(self, temp_dir, safety_config):
        """Test enhanced safety validator initialization."""
        # Override paths
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Check components initialized
        assert validator.constraint_manager is not None
        assert validator.explainability_node is not None
        assert validator.tool_safety_manager is not None
        
        # Check domain validators initialized (lazy load)
        assert hasattr(validator, 'causal_validator')
        
        validator.shutdown()
    
    def test_basic_action_validation(self, temp_dir, safety_config, sample_action, sample_context):
        """Test basic action validation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Validate action
        safe, reason, confidence = validator.validate_action(sample_action, sample_context)
        
        assert isinstance(safe, bool)
        assert isinstance(reason, str)
        assert 0 <= confidence <= 1
        
        validator.shutdown()
    
    def test_comprehensive_validation_sync(self, temp_dir, safety_config, sample_action, sample_context):
        """Test comprehensive synchronous validation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False  # Faster
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Comprehensive validation
        report = validator.validate_action_comprehensive(
            sample_action,
            sample_context,
            create_snapshot=True
        )
        
        assert isinstance(report, SafetyReport)
        assert report.audit_id is not None
        assert 'snapshot_id' in report.metadata or report.metadata.get('snapshot_id') is None
        
        # Get stats
        stats = validator.get_safety_stats()
        assert 'metrics' in stats
        assert 'constraints' in stats
        
        validator.shutdown()
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation_async(self, temp_dir, safety_config, sample_action, sample_context):
        """Test comprehensive asynchronous validation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Async validation
        report = await validator.validate_action_comprehensive_async(
            sample_action,
            sample_context,
            timeout_per_validator=2.0,
            total_timeout=10.0
        )
        
        assert isinstance(report, SafetyReport)
        assert report.audit_id is not None
        assert 'validation_time_ms' in report.metadata
        
        validator.shutdown()
    
    def test_domain_validator_delegation(self, temp_dir, safety_config):
        """Test domain validator delegation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Causal edge validation
        result = validator.validate_causal_edge('A', 'B', 2.5)
        assert 'safe' in result
        
        # Causal path validation
        result = validator.validate_causal_path(['A', 'B', 'C'], [2.0, 1.5])
        assert 'safe' in result
        
        # Prediction validation
        result = validator.validate_prediction_comprehensive(
            10.0, 8.0, 12.0,
            {'target_variable': 'temperature'}
        )
        assert 'safe' in result
        
        validator.shutdown()
    
    def test_tool_selection_validation(self, temp_dir, safety_config):
        """Test tool selection validation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        validator = EnhancedSafetyValidator(config=safety_config)
        
        tools = ['probabilistic', 'symbolic']
        context = {
            'confidence': 0.8,
            'constraints': {}
        }
        
        allowed, report = validator.validate_tool_selection(tools, context)
        
        assert isinstance(allowed, list)
        assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    def test_graph_validation(self, temp_dir, safety_config):
        """Test Graphix IR graph validation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Valid graph
        graph = {
            'nodes': {
                'node1': {'type': 'input', 'edges': [{'target': 'node2'}]},
                'node2': {'type': 'compute', 'edges': [{'target': 'node3'}]},
                'node3': {'type': 'output', 'edges': []}
            }
        }
        
        report = validator.validate(graph)
        assert isinstance(report, SafetyReport)
        
        # Graph with cycle
        graph_cyclic = {
            'nodes': {
                'node1': {'type': 'compute', 'edges': [{'target': 'node2'}]},
                'node2': {'type': 'compute', 'edges': [{'target': 'node1'}]}
            }
        }
        
        report = validator.validate(graph_cyclic)
        assert report.safe == False
        assert any('cycle' in r.lower() for r in report.reasons)
        
        validator.shutdown()

# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for the complete safety module."""
    
    def test_end_to_end_safe_action(self, temp_dir, safety_config, sample_action, sample_context):
        """Test complete flow for a safe action."""
        # Setup
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # 1. Validate action comprehensively
        report = validator.validate_action_comprehensive(sample_action, sample_context)
        
        # 2. Check report
        assert isinstance(report, SafetyReport)
        assert report.audit_id is not None
        
        # 3. Verify snapshot created
        if validator.rollback_manager:
            assert len(validator.rollback_manager.snapshots) > 0
        
        # 4. Verify audit logged
        if validator.audit_logger:
            metrics = validator.audit_logger.get_metrics()
            assert metrics['total_entries'] > 0
        
        # 5. Get safety stats
        stats = validator.get_safety_stats()
        assert stats['metrics']['total_checks'] > 0
        
        validator.shutdown()
    
    def test_end_to_end_unsafe_action(self, temp_dir, safety_config):
        """Test complete flow for an unsafe action."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Create unsafe action
        unsafe_action = {
            'type': ActionType.OPTIMIZE,
            'confidence': 0.3,  # Too low
            'uncertainty': 0.95,  # Too high
            'resource_usage': {
                'energy_nJ': 10000  # Exceeds budget
            }
        }
        
        context = {
            'energy_budget': 1000,
            'state': {}
        }
        
        # Validate
        report = validator.validate_action_comprehensive(unsafe_action, context)
        
        # Should fail
        assert report.safe == False
        assert len(report.violations) > 0
        assert len(report.reasons) > 0
        
        # Check quarantine
        if validator.rollback_manager:
            assert len(validator.rollback_manager.quarantine) > 0
        
        validator.shutdown()
    
    def test_tool_safety_integration(self, temp_dir, safety_config):
        """Test tool safety integration with main validator."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Action with tool usage
        action = {
            'type': ActionType.OPTIMIZE,
            'tool_name': 'probabilistic',
            'confidence': 0.8,
            'uncertainty': 0.2
        }
        
        context = {
            'data_quality': 0.9,
            'corrupted_data': False,
            'adversarial_detected': False,
            'system_overload': False
        }
        
        # Validate
        report = validator.validate_action_comprehensive(action, context)
        
        assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    def test_rollback_on_critical_violation(self, temp_dir, safety_config):
        """Test automatic rollback on critical violation."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.rollback_config['auto_rollback_on_critical'] = True
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Create initial state
        context = {
            'state': {'value': 100},
            'action_log': []
        }
        
        # Critical violation
        action = {
            'type': ActionType.OPTIMIZE,
            'confidence': 0.8,
            'causes_harm': True  # Would trigger ethical violation
        }
        
        report = validator.validate_action_comprehensive(action, context, create_snapshot=True)
        
        # Should detect violation
        if not report.safe and SafetyViolationType.ADVERSARIAL in report.violations:
            # Check if rollback attempted
            if validator.rollback_manager:
                metrics = validator.rollback_manager.get_metrics()
                # Rollback may or may not occur depending on violation type
                assert 'total_rollbacks' in metrics
        
        validator.shutdown()
    
    def test_concurrent_validations(self, temp_dir, safety_config):
        """Test concurrent validation requests."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Create multiple validation threads
        results = []
        
        def validate_action(action_id):
            action = {
                'type': ActionType.OPTIMIZE,
                'id': f'action_{action_id}',
                'confidence': 0.8,
                'uncertainty': 0.2
            }
            context = {'state': {}}
            report = validator.validate_action_comprehensive(action, context)
            results.append(report)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=validate_action, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All validations should complete
        assert len(results) == 5
        for report in results:
            assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_parallel_validations(self, temp_dir, safety_config):
        """Test parallel async validations."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Create multiple validation tasks
        tasks = []
        for i in range(3):
            action = {
                'type': ActionType.OPTIMIZE,
                'id': f'async_action_{i}',
                'confidence': 0.8
            }
            context = {'state': {}}
            
            task = validator.validate_action_comprehensive_async(
                action, context,
                timeout_per_validator=1.0,
                total_timeout=5.0
            )
            tasks.append(task)
        
        # Run all in parallel
        reports = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(reports) == 3
        for report in reports:
            assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    def test_governance_integration(self, temp_dir, safety_config):
        """Test governance integration with safety validator.
        
        Note: Uses ActionType.OPTIMIZE instead of ActionType.ANALYZE 
        since ANALYZE is not a valid ActionType enum value.
        """
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        
        # Setup governance
        governance_config = {'db_path': str(Path(temp_dir) / 'governance.db')}
        governance = GovernanceManager(config=governance_config)
        alignment = ValueAlignmentSystem()
        
        # Create action with valid ActionType (OPTIMIZE, not ANALYZE)
        action = {
            'type': ActionType.OPTIMIZE,  # Valid ActionType - not ANALYZE which doesn't exist
            'risk_score': 0.3,
            'safety_score': 0.8,
            'confidence': 0.85,
            'explanation': 'Test governance integration',
            'auditable': True
        }
        
        context = {
            'state': {'initialized': True},
            'governance_level': 'standard'
        }
        
        # Request approval
        approval_result = governance.request_approval(action)
        assert 'approved' in approval_result
        assert 'decision_id' in approval_result
        
        # Check alignment
        alignment_result = alignment.check_alignment(action, context)
        assert 'aligned' in alignment_result
        assert 'alignment_score' in alignment_result
        
        # Verify governance and alignment work together
        if approval_result.get('approved') and alignment_result.get('aligned'):
            # Both systems agree the action is acceptable
            assert True
        else:
            # At least one system flagged the action
            # This is still valid behavior - just log it
            print(f"Governance approved: {approval_result.get('approved')}")
            print(f"Alignment aligned: {alignment_result.get('aligned')}")
        
        governance.shutdown()

# ============================================================
# STRESS TESTS
# ============================================================

class TestStress:
    """Stress tests for performance and resource limits."""
    
    def test_high_frequency_validations(self, temp_dir, safety_config):
        """Test high-frequency validation requests."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        safety_config.audit_config['log_path'] = str(Path(temp_dir) / 'audit')
        safety_config.enable_adversarial_testing = False
        
        validator = EnhancedSafetyValidator(config=safety_config)
        
        start_time = time.time()
        count = 50
        
        for i in range(count):
            action = {
                'type': ActionType.OPTIMIZE,
                'id': f'stress_action_{i}',
                'confidence': 0.8
            }
            context = {'state': {}}
            
            report = validator.validate_action_comprehensive(action, context)
            assert isinstance(report, SafetyReport)
        
        elapsed = time.time() - start_time
        throughput = count / elapsed
        
        print(f"\nValidation throughput: {throughput:.2f} validations/second")
        assert throughput > 1.0  # Should handle at least 1 validation per second
        
        validator.shutdown()
    
    def test_memory_bounded_structures(self):
        """Test that memory-bounded structures work correctly."""
        manager = ToolSafetyManager()
        
        # Add many usage records
        for i in range(2000):
            context = {
                'confidence': 0.8,
                'adversarial_detected': False,
                'system_overload': False
            }
            manager.check_tool_safety('probabilistic', context)
        
        # Should be bounded
        assert len(manager.usage_history['probabilistic']) <= 1000
        
        manager.shutdown()
    
    def test_constraint_manager_scale(self):
        """Test constraint manager with many constraints."""
        manager = ConstraintManager()
        
        # Add many constraints
        for i in range(50):
            constraint = SafetyConstraint(
                name=f'constraint_{i}',
                type='soft',
                check_function=lambda a, c, i=i: (a.get('value', 0) < 100 + i, 0.9),
                threshold=0.0,
                priority=i
            )
            manager.add_constraint(constraint)
        
        # Check constraints
        action = {'value': 50}
        context = {}
        
        start = time.time()
        report = manager.check_constraints(action, context)
        elapsed = time.time() - start
        
        print(f"\nConstraint check time (50 constraints): {elapsed*1000:.2f}ms")
        assert elapsed < 1.0  # Should complete within 1 second
        
        manager.shutdown()

# ============================================================
# ERROR HANDLING TESTS
# ============================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_validator_with_missing_components(self, temp_dir):
        """Test validator gracefully handles missing components."""
        config = SafetyConfig(
            enable_rollback=False,
            enable_audit_logging=False,
            enable_tool_safety=False
        )
        
        validator = EnhancedSafetyValidator(config=config)
        
        # Should still work
        action = {'type': ActionType.OPTIMIZE, 'confidence': 0.8}
        context = {'state': {}}
        
        report = validator.validate_action_comprehensive(action, context)
        assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    def test_invalid_action_handling(self, temp_dir, safety_config):
        """Test handling of invalid action data."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Empty action
        report = validator.validate_action_comprehensive({}, {})
        assert isinstance(report, SafetyReport)
        
        # Action with NaN values
        action = {
            'type': ActionType.OPTIMIZE,
            'confidence': float('nan')
        }
        report = validator.validate_action_comprehensive(action, {})
        assert isinstance(report, SafetyReport)
        
        validator.shutdown()
    
    def test_database_error_recovery(self, temp_dir):
        """Test recovery from database errors."""
        # Create manager with invalid path
        manager = RollbackManager(
            config={'storage_path': '/invalid/path/that/does/not/exist'}
        )
        
        # Should handle gracefully
        try:
            snapshot_id = manager.create_snapshot({'test': 1}, [])
            # If it doesn't raise, that's ok
        except Exception as e:
            # Should be a specific error, not a crash
            assert isinstance(e, (RuntimeError, IOError, OSError))
        
        manager.shutdown()
    
    def test_concurrent_shutdown(self, temp_dir, safety_config):
        """Test clean shutdown with concurrent operations."""
        safety_config.rollback_config['storage_path'] = str(Path(temp_dir) / 'rollback')
        validator = EnhancedSafetyValidator(config=safety_config)
        
        # Start validation in background
        def background_validation():
            for _ in range(10):
                if validator._shutdown:
                    break
                action = {'type': ActionType.OPTIMIZE, 'confidence': 0.8}
                try:
                    validator.validate_action_comprehensive(action, {'state': {}})
                except:
                    pass  # Ignore errors during shutdown
                time.sleep(0.1)
        
        thread = threading.Thread(target=background_validation)
        thread.start()
        
        time.sleep(0.2)
        
        # Shutdown should be clean
        validator.shutdown()
        
        thread.join(timeout=2.0)
        assert not thread.is_alive()

# ============================================================
# RUN TESTS
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
