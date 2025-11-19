"""
test_transparency_interface.py - Unit tests for TransparencyInterface
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock
from collections import defaultdict, deque
from dataclasses import dataclass

from vulcan.world_model.meta_reasoning.transparency_interface import (
    TransparencyInterface,
    TransparencyMetadata,
    SerializationFormat
)


@dataclass
class MockObjectiveAnalysis:
    """Mock ObjectiveAnalysis for testing"""
    objective_name: str
    status: Mock
    current_value: float
    target_value: float
    confidence: float
    reasoning: str


@dataclass
class MockProposalValidation:
    """Mock ProposalValidation for testing"""
    proposal_id: str
    valid: bool
    overall_status: Mock
    objective_analyses: list
    conflicts_detected: list
    alternatives_suggested: list
    reasoning: str
    confidence: float
    timestamp: float


@pytest.fixture
def mock_introspection():
    """Mock motivational introspection engine"""
    introspection = Mock()
    
    # Mock validation history
    introspection.validation_history = deque(maxlen=1000)
    
    # Mock conflict detector
    conflict_detector = Mock()
    hierarchy = Mock()
    hierarchy.find_conflicts = Mock(return_value=None)
    conflict_detector.objective_hierarchy = hierarchy
    introspection.conflict_detector = conflict_detector
    
    # Mock conflict history
    introspection.conflict_history = deque(maxlen=500)
    
    # Mock active objectives
    introspection.active_objectives = {
        'prediction_accuracy': {'weight': 1.0},
        'efficiency': {'weight': 0.8}
    }
    
    # Mock objective constraints
    introspection.objective_constraints = {
        'prediction_accuracy': {'min': 0.9, 'max': 1.0},
        'efficiency': {'min': 0.0, 'max': 1.0}
    }
    
    # Mock explain_motivation_structure
    def mock_explain():
        return {
            'objectives': {
                'active': ['prediction_accuracy', 'efficiency'],
                'weights': {'prediction_accuracy': 1.0, 'efficiency': 0.8},
                'constraints': introspection.objective_constraints,
                'hierarchy': {
                    'primary': [],
                    'secondary': [],
                    'derived': []
                }
            },
            'current_state': {
                'prediction_accuracy': {'current_value': 0.95},
                'efficiency': {'current_value': 0.85}
            }
        }
    
    introspection.explain_motivation_structure = mock_explain
    
    return introspection


@pytest.fixture
def interface(mock_introspection):
    """Create interface instance"""
    return TransparencyInterface(mock_introspection)


@pytest.fixture
def sample_validation():
    """Sample validation result"""
    mock_status = Mock()
    mock_status.value = 'aligned'
    
    mock_analysis_status = Mock()
    mock_analysis_status.value = 'aligned'
    
    return MockProposalValidation(
        proposal_id='test_123',
        valid=True,
        overall_status=mock_status,
        objective_analyses=[
            MockObjectiveAnalysis(
                objective_name='prediction_accuracy',
                status=mock_analysis_status,
                current_value=0.95,
                target_value=0.95,
                confidence=0.9,
                reasoning='On target'
            )
        ],
        conflicts_detected=[],
        alternatives_suggested=[],
        reasoning='Proposal is aligned',
        confidence=0.9,
        timestamp=time.time()
    )


class TestInitialization:
    """Test initialization"""
    
    def test_init(self, mock_introspection):
        """Test basic initialization"""
        interface = TransparencyInterface(mock_introspection)
        
        assert interface.introspection_engine == mock_introspection
        assert isinstance(interface.stats, defaultdict)
        assert interface.schema_version is not None
    
    def test_cache_initialized(self, interface):
        """Test that cache is initialized"""
        assert isinstance(interface.cache, dict)
        assert interface.cache_ttl > 0
    
    def test_audit_log_initialized(self, interface):
        """Test that audit log is initialized"""
        assert isinstance(interface.audit_log, list)
        assert interface.max_audit_entries > 0


class TestSerializeValidation:
    """Test validation serialization"""
    
    def test_serialize_basic(self, interface, sample_validation):
        """Test basic validation serialization"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert isinstance(serialized, dict)
        assert 'schema_version' in serialized
        assert 'type' in serialized
        assert serialized['type'] == 'validation_result'
    
    def test_serialized_includes_metadata(self, interface, sample_validation):
        """Test that serialized validation includes metadata"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert 'metadata' in serialized
        assert 'version' in serialized['metadata']
        assert 'timestamp' in serialized['metadata']
    
    def test_serialized_includes_validation_data(self, interface, sample_validation):
        """Test that serialized validation includes validation data"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert 'validation' in serialized
        assert 'id' in serialized['validation']
        assert 'outcome' in serialized['validation']
    
    def test_serialized_includes_objectives(self, interface, sample_validation):
        """Test that objectives are included"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert 'objectives' in serialized['validation']
        assert isinstance(serialized['validation']['objectives'], list)
    
    def test_serialized_includes_conflicts(self, interface, sample_validation):
        """Test that conflicts are included"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert 'conflicts' in serialized['validation']
        assert isinstance(serialized['validation']['conflicts'], list)
    
    def test_serialized_includes_signature(self, interface, sample_validation):
        """Test that signature is included"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert 'signature' in serialized
        assert isinstance(serialized['signature'], str)
        assert len(serialized['signature']) > 0
    
    def test_serialized_includes_actionable(self, interface, sample_validation):
        """Test that actionable items are included"""
        serialized = interface.serialize_validation(sample_validation)
        
        assert 'actionable' in serialized
        assert isinstance(serialized['actionable'], list)
    
    def test_serialize_dict_validation(self, interface):
        """Test serialization of dict validation"""
        validation_dict = {
            'proposal_id': 'test_456',
            'valid': False,
            'overall_status': 'conflict',
            'confidence': 0.7,
            'reasoning': 'Conflicts detected',
            'timestamp': time.time()
        }
        
        serialized = interface.serialize_validation(validation_dict)
        
        assert isinstance(serialized, dict)
        assert serialized['validation']['id'] == 'test_456'
    
    def test_statistics_updated(self, interface, sample_validation):
        """Test that statistics are updated"""
        initial_count = interface.stats['validations_serialized']
        
        interface.serialize_validation(sample_validation)
        
        assert interface.stats['validations_serialized'] == initial_count + 1
    
    def test_audit_log_updated(self, interface, sample_validation):
        """Test that audit log is updated"""
        initial_size = len(interface.audit_log)
        
        interface.serialize_validation(sample_validation)
        
        assert len(interface.audit_log) == initial_size + 1


class TestSerializeObjectiveState:
    """Test objective state serialization"""
    
    def test_serialize_objective_state(self, interface):
        """Test basic objective state serialization"""
        serialized = interface.serialize_objective_state()
        
        assert isinstance(serialized, dict)
        assert serialized['type'] == 'objective_state'
    
    def test_includes_objectives(self, interface):
        """Test that objectives are included"""
        serialized = interface.serialize_objective_state()
        
        assert 'objectives' in serialized
        assert 'active' in serialized['objectives']
        assert 'hierarchy' in serialized['objectives']
    
    def test_includes_constraints(self, interface):
        """Test that constraints are included"""
        serialized = interface.serialize_objective_state()
        
        assert 'constraints' in serialized
        assert isinstance(serialized['constraints'], dict)
    
    def test_includes_signature(self, interface):
        """Test that signature is included"""
        serialized = interface.serialize_objective_state()
        
        assert 'signature' in serialized
        assert isinstance(serialized['signature'], str)
    
    def test_statistics_updated(self, interface):
        """Test that statistics are updated"""
        initial_count = interface.stats['objective_states_serialized']
        
        interface.serialize_objective_state()
        
        assert interface.stats['objective_states_serialized'] == initial_count + 1


class TestSerializeConflict:
    """Test conflict serialization"""
    
    def test_serialize_single_conflict(self, interface):
        """Test serializing single conflict"""
        conflict = {
            'objectives': ['efficiency', 'prediction_accuracy'],
            'conflict_type': 'direct',
            'severity': 'medium',
            'description': 'Speed vs accuracy',
            'quantitative_measure': 0.6
        }
        
        serialized = interface.serialize_conflict(conflict)
        
        assert isinstance(serialized, dict)
        assert serialized['type'] == 'conflict_analysis'
    
    def test_serialize_conflict_list(self, interface):
        """Test serializing list of conflicts"""
        conflicts = [
            {
                'objectives': ['a', 'b'],
                'conflict_type': 'direct',
                'severity': 'high',
                'description': 'Conflict 1'
            },
            {
                'objectives': ['c', 'd'],
                'conflict_type': 'indirect',
                'severity': 'low',
                'description': 'Conflict 2'
            }
        ]
        
        serialized = interface.serialize_conflict(conflicts)
        
        assert 'conflicts' in serialized
        assert len(serialized['conflicts']) == 2
    
    def test_includes_summary(self, interface):
        """Test that summary is included"""
        conflict = {
            'objectives': ['a', 'b'],
            'conflict_type': 'direct',
            'severity': 'critical',
            'description': 'Test conflict'
        }
        
        serialized = interface.serialize_conflict(conflict)
        
        assert 'summary' in serialized
        assert 'total_conflicts' in serialized['summary']
        assert 'by_severity' in serialized['summary']
    
    def test_resolution_strategy_recommended(self, interface):
        """Test that resolution strategy is recommended"""
        conflict = {
            'objectives': ['a', 'b'],
            'conflict_type': 'direct',
            'severity': 'critical',
            'description': 'Critical conflict'
        }
        
        serialized = interface.serialize_conflict(conflict)
        
        assert 'resolution_strategy' in serialized
        assert serialized['resolution_strategy'] in [
            'none_needed', 
            'immediate_resolution_required',
            'priority_resolution_recommended',
            'standard_resolution'
        ]
    
    def test_statistics_updated(self, interface):
        """Test that statistics are updated"""
        initial_count = interface.stats['conflicts_serialized']
        
        interface.serialize_conflict({})
        
        assert interface.stats['conflicts_serialized'] == initial_count + 1


class TestSerializeNegotiationOutcome:
    """Test negotiation outcome serialization"""
    
    def test_serialize_negotiation(self, interface):
        """Test basic negotiation serialization"""
        mock_outcome = Mock()
        mock_outcome.value = 'consensus'
        
        mock_strategy = Mock()
        mock_strategy.value = 'pareto_optimal'
        
        negotiation = {
            'outcome': mock_outcome,
            'agreed_objectives': {'efficiency': 0.8},
            'objective_weights': {'efficiency': 1.0},
            'participating_agents': ['agent_1', 'agent_2'],
            'strategy_used': mock_strategy,
            'iterations': 5,
            'convergence_time_ms': 100.0,
            'compromises_made': [],
            'pareto_optimal': True,
            'confidence': 0.9,
            'reasoning': 'Consensus reached'
        }
        
        serialized = interface.serialize_negotiation_outcome(negotiation)
        
        assert isinstance(serialized, dict)
        assert serialized['type'] == 'negotiation_outcome'
    
    def test_includes_outcome(self, interface):
        """Test that outcome is included"""
        mock_outcome = Mock()
        mock_outcome.value = 'consensus'
        mock_strategy = Mock()
        mock_strategy.value = 'pareto_optimal'
        
        negotiation = {
            'outcome': mock_outcome,
            'strategy_used': mock_strategy,
            'pareto_optimal': True,
            'confidence': 0.9
        }
        
        serialized = interface.serialize_negotiation_outcome(negotiation)
        
        assert 'outcome' in serialized
        assert 'status' in serialized['outcome']
    
    def test_includes_agreement(self, interface):
        """Test that agreement is included"""
        negotiation = {
            'outcome': Mock(value='consensus'),
            'strategy_used': Mock(value='pareto'),
            'agreed_objectives': {},
            'objective_weights': {},
            'participating_agents': []
        }
        
        serialized = interface.serialize_negotiation_outcome(negotiation)
        
        assert 'agreement' in serialized
        assert 'objectives' in serialized['agreement']
        assert 'weights' in serialized['agreement']
    
    def test_includes_process(self, interface):
        """Test that process information is included"""
        negotiation = {
            'outcome': Mock(value='consensus'),
            'strategy_used': Mock(value='pareto'),
            'iterations': 10,
            'convergence_time_ms': 150.0,
            'compromises_made': []
        }
        
        serialized = interface.serialize_negotiation_outcome(negotiation)
        
        assert 'process' in serialized
        assert 'strategy' in serialized['process']
        assert 'iterations' in serialized['process']
    
    def test_statistics_updated(self, interface):
        """Test that statistics are updated"""
        initial_count = interface.stats['negotiations_serialized']
        
        negotiation = {
            'outcome': Mock(value='consensus'),
            'strategy_used': Mock(value='pareto')
        }
        
        interface.serialize_negotiation_outcome(negotiation)
        
        assert interface.stats['negotiations_serialized'] == initial_count + 1


class TestExportForConsensus:
    """Test consensus export"""
    
    def test_export_for_consensus(self, interface):
        """Test basic consensus export"""
        export = interface.export_for_consensus()
        
        assert isinstance(export, dict)
        assert export['type'] == 'consensus_context'
    
    def test_includes_system_state(self, interface):
        """Test that system state is included"""
        export = interface.export_for_consensus()
        
        assert 'system_state' in export
        assert 'objectives' in export['system_state']
    
    def test_includes_recent_activity(self, interface):
        """Test that recent activity is included"""
        export = interface.export_for_consensus()
        
        assert 'recent_activity' in export
        assert 'validations' in export['recent_activity']
        assert 'conflicts' in export['recent_activity']
    
    def test_includes_voting_context(self, interface):
        """Test that voting context is included"""
        export = interface.export_for_consensus()
        
        assert 'voting_context' in export
        assert 'quorum_required' in export['voting_context']
        assert 'voting_mechanism' in export['voting_context']
    
    def test_statistics_updated(self, interface):
        """Test that statistics are updated"""
        initial_count = interface.stats['consensus_exports']
        
        interface.export_for_consensus()
        
        assert interface.stats['consensus_exports'] == initial_count + 1


class TestAuditLog:
    """Test audit log functionality"""
    
    def test_get_audit_log(self, interface):
        """Test getting audit log"""
        # Add some entries
        interface._audit('test_event', {'data': 'test'})
        
        log = interface.get_audit_log()
        
        assert isinstance(log, list)
        assert len(log) > 0
    
    def test_filter_by_time(self, interface):
        """Test filtering by time"""
        now = time.time()
        
        interface._audit('event1', {'data': '1'})
        time.sleep(0.01)
        future_time = time.time()
        interface._audit('event2', {'data': '2'})
        
        # Get events after now
        log = interface.get_audit_log(start_time=future_time)
        
        # Should only get event2
        assert len(log) >= 1
    
    def test_filter_by_event_type(self, interface):
        """Test filtering by event type"""
        interface._audit('type_a', {'data': '1'})
        interface._audit('type_b', {'data': '2'})
        interface._audit('type_a', {'data': '3'})
        
        log = interface.get_audit_log(event_type='type_a')
        
        # Should only get type_a events
        assert all(e['event_type'] == 'type_a' for e in log)
    
    def test_audit_log_trimmed(self, interface):
        """Test that audit log is trimmed when too large"""
        # Set small limit
        interface.max_audit_entries = 10
        
        # Add many entries
        for i in range(20):
            interface._audit(f'event_{i}', {'data': i})
        
        # Should be trimmed to max
        assert len(interface.audit_log) <= interface.max_audit_entries


class TestSignatureVerification:
    """Test signature generation and verification"""
    
    def test_generate_signature(self, interface):
        """Test signature generation"""
        data = {'test': 'data', 'number': 123}
        
        signature = interface._generate_signature(data)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length
    
    def test_signature_deterministic(self, interface):
        """Test that signature is deterministic"""
        data = {'test': 'data'}
        
        sig1 = interface._generate_signature(data)
        sig2 = interface._generate_signature(data)
        
        assert sig1 == sig2
    
    def test_signature_changes_with_data(self, interface):
        """Test that signature changes when data changes"""
        data1 = {'test': 'data1'}
        data2 = {'test': 'data2'}
        
        sig1 = interface._generate_signature(data1)
        sig2 = interface._generate_signature(data2)
        
        assert sig1 != sig2
    
    def test_verify_signature_valid(self, interface, sample_validation):
        """Test verifying valid signature"""
        serialized = interface.serialize_validation(sample_validation)
        
        is_valid = interface.verify_signature(serialized)
        
        assert is_valid is True
    
    def test_verify_signature_invalid(self, interface):
        """Test verifying invalid signature"""
        data = {
            'test': 'data',
            'signature': 'invalid_signature'
        }
        
        is_valid = interface.verify_signature(data)
        
        assert is_valid is False
    
    def test_verify_signature_missing(self, interface):
        """Test verifying missing signature"""
        data = {'test': 'data'}
        
        is_valid = interface.verify_signature(data)
        
        assert is_valid is False


class TestHelperMethods:
    """Test helper methods"""
    
    def test_create_metadata(self, interface):
        """Test metadata creation"""
        metadata = interface._create_metadata('test_context')
        
        assert isinstance(metadata, dict)
        assert 'version' in metadata
        assert 'timestamp' in metadata
        assert 'source' in metadata
        assert 'context' in metadata
    
    def test_extract_structured_reasoning(self, interface):
        """Test extracting structured reasoning"""
        data = {
            'reasoning': 'Conflict detected in proposal',
            'valid': False,
            'confidence': 0.7
        }
        
        structured = interface._extract_structured_reasoning(data)
        
        assert isinstance(structured, dict)
        assert 'decision' in structured
        assert 'confidence' in structured
        assert 'key_factors' in structured
    
    def test_extract_actionable_items_rejected(self, interface):
        """Test extracting actionable items from rejected proposal"""
        validation = {
            'valid': False,
            'conflicts': [{'type': 'direct'}],
            'alternatives': [{'objective': 'efficiency'}]
        }
        
        actionable = interface._extract_actionable_items(validation)
        
        assert isinstance(actionable, list)
        assert len(actionable) > 0
    
    def test_extract_actionable_items_accepted(self, interface):
        """Test extracting actionable items from accepted proposal"""
        validation = {
            'valid': True
        }
        
        actionable = interface._extract_actionable_items(validation)
        
        assert isinstance(actionable, list)
        assert len(actionable) == 0
    
    def test_count_by_severity(self, interface):
        """Test counting conflicts by severity"""
        conflicts = [
            {'severity': 'high'},
            {'severity': 'medium'},
            {'severity': 'high'}
        ]
        
        counts = interface._count_by_severity(conflicts)
        
        assert counts['high'] == 2
        assert counts['medium'] == 1
    
    def test_count_by_type(self, interface):
        """Test counting conflicts by type"""
        conflicts = [
            {'type': 'direct'},
            {'type': 'indirect'},
            {'type': 'direct'}
        ]
        
        counts = interface._count_by_type(conflicts)
        
        assert counts['direct'] == 2
        assert counts['indirect'] == 1


class TestTransparencyMetadata:
    """Test TransparencyMetadata dataclass"""
    
    def test_create_metadata(self):
        """Test creating metadata"""
        metadata = TransparencyMetadata()
        
        assert metadata.version == "1.0"
        assert metadata.source == "vulcan_ami"
        assert metadata.timestamp > 0
    
    def test_metadata_with_params(self):
        """Test creating metadata with parameters"""
        metadata = TransparencyMetadata(
            version="2.0",
            timestamp=12345.0,
            signature="test_sig",
            source="test_source"
        )
        
        assert metadata.version == "2.0"
        assert metadata.timestamp == 12345.0
        assert metadata.signature == "test_sig"


class TestStatistics:
    """Test statistics tracking"""
    
    def test_get_statistics(self, interface):
        """Test getting statistics"""
        stats = interface.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'statistics' in stats
        assert 'audit_log_size' in stats
        assert 'schema_version' in stats
    
    def test_statistics_track_operations(self, interface, sample_validation):
        """Test that operations are tracked"""
        # Perform operations
        interface.serialize_validation(sample_validation)
        interface.serialize_objective_state()
        
        stats = interface.get_statistics()
        
        assert stats['statistics']['validations_serialized'] > 0
        assert stats['statistics']['objective_states_serialized'] > 0


class TestEdgeCases:
    """Test edge cases"""
    
    def test_serialize_empty_conflict(self, interface):
        """Test serializing empty conflict"""
        serialized = interface.serialize_conflict([])
        
        assert isinstance(serialized, dict)
        assert serialized['summary']['total_conflicts'] == 0
    
    def test_serialize_none_values(self, interface):
        """Test handling None values gracefully"""
        validation = {
            'proposal_id': 'test',
            'valid': True,
            'overall_status': None,
            'confidence': None,
            'reasoning': None
        }
        
        serialized = interface.serialize_validation(validation)
        
        assert isinstance(serialized, dict)
    
    def test_malformed_negotiation(self, interface):
        """Test handling malformed negotiation data"""
        negotiation = {
            'outcome': 'not_a_mock',  # String instead of enum
            'strategy_used': 'not_a_mock'
        }
        
        # Should handle gracefully
        serialized = interface.serialize_negotiation_outcome(negotiation)
        
        assert isinstance(serialized, dict)
    
    def test_empty_audit_log_filter(self, interface):
        """Test filtering empty audit log"""
        log = interface.get_audit_log(event_type='nonexistent')
        
        assert isinstance(log, list)
        assert len(log) == 0


class TestIntegration:
    """Integration tests"""
    
    def test_full_validation_workflow(self, interface, sample_validation):
        """Test full validation workflow"""
        # 1. Serialize validation
        serialized = interface.serialize_validation(sample_validation)
        assert isinstance(serialized, dict)
        
        # 2. Verify signature
        is_valid = interface.verify_signature(serialized)
        assert is_valid is True
        
        # 3. Check audit log
        log = interface.get_audit_log(event_type='validation_serialized')
        assert len(log) > 0
        
        # 4. Get statistics
        stats = interface.get_statistics()
        assert stats['statistics']['validations_serialized'] > 0
    
    def test_consensus_export_workflow(self, interface):
        """Test consensus export workflow"""
        # 1. Export for consensus
        export = interface.export_for_consensus()
        assert isinstance(export, dict)
        
        # 2. Verify signature
        is_valid = interface.verify_signature(export)
        assert is_valid is True
        
        # 3. Check structure
        assert 'system_state' in export
        assert 'voting_context' in export


if __name__ == '__main__':
    pytest.main([__file__, '-v'])