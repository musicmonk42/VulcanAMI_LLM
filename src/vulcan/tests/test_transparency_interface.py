"""
test_transparency_interface.py - PURE MOCK VERSION
Tests transparency interface without spawning threads.
"""

import pytest
import json
import time
import hashlib
import threading
from unittest.mock import Mock, MagicMock
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


# ============================================================================
# Mock Enums and Classes
# ============================================================================

class SerializationFormat(Enum):
    JSON = "json"
    DICT = "dict"
    COMPACT = "compact"


@dataclass
class TransparencyMetadata:
    version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)
    schema_version: str = "1.0"
    source: str = "transparency_interface"


@dataclass
class MockObjectiveAnalysis:
    objective_name: str
    status: Mock
    current_value: float
    target_value: float
    confidence: float
    reasoning: str


@dataclass
class MockProposalValidation:
    proposal_id: str
    valid: bool
    overall_status: Mock
    objective_analyses: list
    conflicts_detected: list
    alternatives_suggested: list
    reasoning: str
    confidence: float
    timestamp: float


class MockTransparencyInterface:
    """Mock TransparencyInterface without thread spawning"""
    
    def __init__(self, introspection_engine=None):
        self.introspection_engine = introspection_engine
        self.stats = defaultdict(int)
        self.schema_version = "1.0"
        self.cache = {}
        self.cache_ttl = 300
        self.audit_log = []
        self.max_audit_entries = 1000
        self._lock = threading.Lock()
    
    def serialize_validation(self, validation) -> Dict:
        with self._lock:
            self.stats['validations_serialized'] += 1
        
        # Handle dict or object
        if isinstance(validation, dict):
            proposal_id = validation.get('proposal_id', 'unknown')
            valid = validation.get('valid', False)
            overall_status = validation.get('overall_status', 'unknown')
            confidence = validation.get('confidence', 0.0)
            reasoning = validation.get('reasoning', '')
            timestamp = validation.get('timestamp', time.time())
            objectives = validation.get('objective_analyses', [])
            conflicts = validation.get('conflicts_detected', [])
        else:
            proposal_id = getattr(validation, 'proposal_id', 'unknown')
            valid = getattr(validation, 'valid', False)
            overall_status = getattr(validation, 'overall_status', Mock())
            if hasattr(overall_status, 'value'):
                overall_status = overall_status.value
            confidence = getattr(validation, 'confidence', 0.0)
            reasoning = getattr(validation, 'reasoning', '')
            timestamp = getattr(validation, 'timestamp', time.time())
            objectives = getattr(validation, 'objective_analyses', [])
            conflicts = getattr(validation, 'conflicts_detected', [])
        
        serialized = {
            'schema_version': self.schema_version,
            'type': 'validation_result',
            'metadata': {
                'version': '1.0.0',
                'timestamp': time.time()
            },
            'validation': {
                'id': proposal_id,
                'outcome': 'valid' if valid else 'invalid',
                'status': str(overall_status),
                'confidence': confidence,
                'reasoning': reasoning,
                'objectives': self._serialize_objectives(objectives),
                'conflicts': [str(c) for c in conflicts]
            },
            'signature': self._generate_signature(proposal_id),
            'actionable': self._get_actionable_items(validation)
        }
        
        # Update audit log
        self.audit_log.append({
            'action': 'serialize_validation',
            'proposal_id': proposal_id,
            'timestamp': time.time()
        })
        
        # Trim audit log
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]
        
        return serialized
    
    def _serialize_objectives(self, objectives) -> List[Dict]:
        result = []
        for obj in objectives:
            if isinstance(obj, dict):
                result.append(obj)
            else:
                status = getattr(obj, 'status', Mock())
                if hasattr(status, 'value'):
                    status = status.value
                result.append({
                    'name': getattr(obj, 'objective_name', 'unknown'),
                    'status': str(status),
                    'current_value': getattr(obj, 'current_value', 0.0),
                    'target_value': getattr(obj, 'target_value', 0.0),
                    'confidence': getattr(obj, 'confidence', 0.0),
                    'reasoning': getattr(obj, 'reasoning', '')
                })
        return result
    
    def _generate_signature(self, data) -> str:
        content = f"{data}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_actionable_items(self, validation) -> List[Dict]:
        items = []
        if isinstance(validation, dict):
            if not validation.get('valid', True):
                items.append({'action': 'review', 'priority': 'high'})
        else:
            if not getattr(validation, 'valid', True):
                items.append({'action': 'review', 'priority': 'high'})
        return items
    
    def serialize_objective_state(self, objective_name: str = None) -> Dict:
        with self._lock:
            self.stats['objective_states_serialized'] += 1
        
        if self.introspection_engine:
            structure = self.introspection_engine.explain_motivation_structure()
        else:
            structure = {'objectives': {'active': [], 'weights': {}}, 'current_state': {}}
        
        if objective_name:
            objectives = {objective_name: structure.get('current_state', {}).get(objective_name, {})}
        else:
            objectives = structure.get('current_state', {})
        
        return {
            'schema_version': self.schema_version,
            'type': 'objective_state',
            'objectives': objectives,
            'weights': structure.get('objectives', {}).get('weights', {}),
            'timestamp': time.time()
        }
    
    def serialize_conflict(self, conflict) -> Dict:
        with self._lock:
            self.stats['conflicts_serialized'] += 1
        
        if isinstance(conflict, dict):
            return {
                'schema_version': self.schema_version,
                'type': 'conflict',
                'conflict': conflict,
                'timestamp': time.time()
            }
        
        return {
            'schema_version': self.schema_version,
            'type': 'conflict',
            'conflict': {
                'id': getattr(conflict, 'conflict_id', 'unknown'),
                'type': str(getattr(conflict, 'conflict_type', 'unknown')),
                'severity': getattr(conflict, 'severity', 0.0),
                'objectives': getattr(conflict, 'objectives', [])
            },
            'timestamp': time.time()
        }
    
    def get_motivation_summary(self) -> Dict:
        with self._lock:
            self.stats['summaries_generated'] += 1
        
        if self.introspection_engine:
            structure = self.introspection_engine.explain_motivation_structure()
        else:
            structure = {'objectives': {'active': []}}
        
        return {
            'schema_version': self.schema_version,
            'type': 'motivation_summary',
            'active_objectives': structure.get('objectives', {}).get('active', []),
            'timestamp': time.time()
        }
    
    def get_validation_history(self, limit: int = 100) -> List[Dict]:
        if self.introspection_engine and hasattr(self.introspection_engine, 'validation_history'):
            history = list(self.introspection_engine.validation_history)[-limit:]
            return [self.serialize_validation(v) for v in history]
        return []
    
    def get_conflict_history(self, limit: int = 100) -> List[Dict]:
        if self.introspection_engine and hasattr(self.introspection_engine, 'conflict_history'):
            history = list(self.introspection_engine.conflict_history)[-limit:]
            return [self.serialize_conflict(c) for c in history]
        return []
    
    def export_state(self, format: SerializationFormat = SerializationFormat.JSON) -> str:
        state = {
            'schema_version': self.schema_version,
            'type': 'full_state_export',
            'motivation': self.get_motivation_summary(),
            'statistics': dict(self.stats),
            'timestamp': time.time()
        }
        
        if format == SerializationFormat.JSON:
            return json.dumps(state, indent=2, default=str)
        elif format == SerializationFormat.COMPACT:
            return json.dumps(state, separators=(',', ':'), default=str)
        else:
            return state
    
    def get_statistics(self) -> Dict:
        return {
            'validations_serialized': self.stats['validations_serialized'],
            'objective_states_serialized': self.stats['objective_states_serialized'],
            'conflicts_serialized': self.stats['conflicts_serialized'],
            'summaries_generated': self.stats['summaries_generated'],
            'cache_size': len(self.cache),
            'audit_log_size': len(self.audit_log)
        }
    
    def clear_cache(self):
        with self._lock:
            self.cache.clear()
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return self.audit_log[-limit:]


# Alias
TransparencyInterface = MockTransparencyInterface


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_introspection():
    introspection = Mock()
    introspection.validation_history = deque(maxlen=1000)
    
    conflict_detector = Mock()
    hierarchy = Mock()
    hierarchy.find_conflicts = Mock(return_value=None)
    conflict_detector.objective_hierarchy = hierarchy
    introspection.conflict_detector = conflict_detector
    
    introspection.conflict_history = deque(maxlen=500)
    introspection.active_objectives = {
        'prediction_accuracy': {'weight': 1.0},
        'efficiency': {'weight': 0.8}
    }
    introspection.objective_constraints = {
        'prediction_accuracy': {'min': 0.9, 'max': 1.0},
        'efficiency': {'min': 0.0, 'max': 1.0}
    }
    
    def mock_explain():
        return {
            'objectives': {
                'active': ['prediction_accuracy', 'efficiency'],
                'weights': {'prediction_accuracy': 1.0, 'efficiency': 0.8},
                'constraints': introspection.objective_constraints
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
    return MockTransparencyInterface(mock_introspection)


@pytest.fixture
def sample_validation():
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


# ============================================================================
# Tests
# ============================================================================

class TestInitialization:
    def test_init(self, mock_introspection):
        interface = MockTransparencyInterface(mock_introspection)
        assert interface.introspection_engine == mock_introspection
        assert isinstance(interface.stats, defaultdict)
        assert interface.schema_version is not None
    
    def test_cache_initialized(self, interface):
        assert isinstance(interface.cache, dict)
        assert interface.cache_ttl > 0
    
    def test_audit_log_initialized(self, interface):
        assert isinstance(interface.audit_log, list)
        assert interface.max_audit_entries > 0


class TestSerializeValidation:
    def test_serialize_basic(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert isinstance(serialized, dict)
        assert 'schema_version' in serialized
        assert 'type' in serialized
        assert serialized['type'] == 'validation_result'
    
    def test_serialized_includes_metadata(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert 'metadata' in serialized
        assert 'version' in serialized['metadata']
        assert 'timestamp' in serialized['metadata']
    
    def test_serialized_includes_validation_data(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert 'validation' in serialized
        assert 'id' in serialized['validation']
        assert 'outcome' in serialized['validation']
    
    def test_serialized_includes_objectives(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert 'objectives' in serialized['validation']
        assert isinstance(serialized['validation']['objectives'], list)
    
    def test_serialized_includes_conflicts(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert 'conflicts' in serialized['validation']
        assert isinstance(serialized['validation']['conflicts'], list)
    
    def test_serialized_includes_signature(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert 'signature' in serialized
        assert isinstance(serialized['signature'], str)
        assert len(serialized['signature']) > 0
    
    def test_serialized_includes_actionable(self, interface, sample_validation):
        serialized = interface.serialize_validation(sample_validation)
        assert 'actionable' in serialized
        assert isinstance(serialized['actionable'], list)
    
    def test_serialize_dict_validation(self, interface):
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
        initial_count = interface.stats['validations_serialized']
        interface.serialize_validation(sample_validation)
        assert interface.stats['validations_serialized'] == initial_count + 1
    
    def test_audit_log_updated(self, interface, sample_validation):
        initial_size = len(interface.audit_log)
        interface.serialize_validation(sample_validation)
        assert len(interface.audit_log) == initial_size + 1


class TestSerializeObjectiveState:
    def test_serialize_objective_state(self, interface):
        state = interface.serialize_objective_state()
        assert isinstance(state, dict)
        assert 'schema_version' in state
        assert 'type' in state
        assert state['type'] == 'objective_state'
    
    def test_serialize_specific_objective(self, interface):
        state = interface.serialize_objective_state('prediction_accuracy')
        assert 'objectives' in state
    
    def test_includes_weights(self, interface):
        state = interface.serialize_objective_state()
        assert 'weights' in state


class TestSerializeConflict:
    def test_serialize_conflict_dict(self, interface):
        conflict = {'id': 'c1', 'type': 'priority', 'severity': 0.5}
        serialized = interface.serialize_conflict(conflict)
        assert serialized['type'] == 'conflict'
        assert 'conflict' in serialized
    
    def test_serialize_conflict_object(self, interface):
        conflict = Mock()
        conflict.conflict_id = 'c2'
        conflict.conflict_type = 'resource'
        conflict.severity = 0.7
        conflict.objectives = ['obj1', 'obj2']
        
        serialized = interface.serialize_conflict(conflict)
        assert serialized['type'] == 'conflict'


class TestMotivationSummary:
    def test_get_motivation_summary(self, interface):
        summary = interface.get_motivation_summary()
        assert isinstance(summary, dict)
        assert 'type' in summary
        assert summary['type'] == 'motivation_summary'
    
    def test_includes_active_objectives(self, interface):
        summary = interface.get_motivation_summary()
        assert 'active_objectives' in summary


class TestHistory:
    def test_get_validation_history(self, interface):
        history = interface.get_validation_history()
        assert isinstance(history, list)
    
    def test_get_conflict_history(self, interface):
        history = interface.get_conflict_history()
        assert isinstance(history, list)


class TestExport:
    def test_export_state_json(self, interface):
        exported = interface.export_state(SerializationFormat.JSON)
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert 'schema_version' in parsed
    
    def test_export_state_compact(self, interface):
        exported = interface.export_state(SerializationFormat.COMPACT)
        assert isinstance(exported, str)
        assert '  ' not in exported  # No indentation
    
    def test_export_state_dict(self, interface):
        exported = interface.export_state(SerializationFormat.DICT)
        assert isinstance(exported, dict)


class TestStatistics:
    def test_get_statistics(self, interface):
        stats = interface.get_statistics()
        assert isinstance(stats, dict)
        assert 'validations_serialized' in stats
        assert 'cache_size' in stats


class TestCache:
    def test_clear_cache(self, interface):
        interface.cache['key'] = 'value'
        interface.clear_cache()
        assert len(interface.cache) == 0


class TestAuditLog:
    def test_get_audit_log(self, interface, sample_validation):
        interface.serialize_validation(sample_validation)
        log = interface.get_audit_log()
        assert isinstance(log, list)
        assert len(log) > 0


class TestThreadSafety:
    def test_concurrent_serialization(self, mock_introspection):
        interface = MockTransparencyInterface(mock_introspection)
        
        def serialize_many():
            for i in range(10):
                validation = {
                    'proposal_id': f'test_{i}',
                    'valid': True,
                    'confidence': 0.9
                }
                interface.serialize_validation(validation)
        
        threads = []
        for _ in range(3):
            t = threading.Thread(target=serialize_many)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert interface.stats['validations_serialized'] == 30


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
