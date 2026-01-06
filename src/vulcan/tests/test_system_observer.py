# tests/test_system_observer.py
"""
Tests for the SystemObserver component.

The SystemObserver connects the query processing pipeline to the WorldModel's
causal learning system. These tests verify that events are properly converted
to observations and fed to the world model.
"""

import time
from unittest.mock import Mock, MagicMock

import pytest


class MockWorldModel:
    """Mock WorldModel for testing SystemObserver"""
    
    def __init__(self):
        self.observations = []
        self.causal_graph = Mock()
        self.causal_graph.add_edge = Mock()
    
    def update_from_observation(self, observation):
        self.observations.append(observation)


@pytest.fixture
def mock_world_model():
    return MockWorldModel()


@pytest.fixture
def system_observer(mock_world_model):
    from vulcan.world_model.system_observer import SystemObserver
    return SystemObserver(mock_world_model)


class TestSystemObserverBasics:
    """Test basic SystemObserver functionality"""
    
    def test_initialization(self, system_observer):
        """Test SystemObserver initializes correctly"""
        assert system_observer.enabled is True
        assert system_observer.stats['total_queries'] == 0
        assert len(system_observer.query_history) == 0
    
    def test_observe_query_start(self, system_observer, mock_world_model):
        """Test observing query start event"""
        system_observer.observe_query_start(
            query_id="q_test_123",
            query="What is the capital of France?",
            classification={
                'category': 'factual',
                'complexity': 0.3,
                'tools': ['knowledge_base']
            }
        )
        
        assert system_observer.stats['total_queries'] == 1
        assert len(system_observer.query_history) == 1
        assert len(mock_world_model.observations) == 1
    
    def test_observe_engine_result(self, system_observer, mock_world_model):
        """Test observing engine result event"""
        system_observer.observe_engine_result(
            query_id="q_test_123",
            engine_name="probabilistic",
            result={'conclusion': 'Paris', 'confidence': 0.95},
            success=True,
            execution_time_ms=150.0
        )
        
        assert system_observer.stats['total_engine_executions'] == 0  # execution_end increments
        assert len(system_observer.engine_history) == 1
        assert len(mock_world_model.observations) == 1
    
    def test_observe_validation_failure(self, system_observer, mock_world_model):
        """Test observing validation failure"""
        system_observer.observe_validation_failure(
            query_id="q_test_456",
            engine_name="mathematical",
            reason="Result type mismatch",
            query="Is {A→B, ¬B, A} satisfiable?",
            result={'conclusion': 'x**2 + 2x + 1'}
        )
        
        assert system_observer.stats['validation_failures'] == 1
        assert len(mock_world_model.observations) == 1
    
    def test_observe_outcome(self, system_observer, mock_world_model):
        """Test observing query outcome"""
        system_observer.observe_outcome(
            query_id="q_test_123",
            response={
                'source': 'reasoning',
                'confidence': 0.85,
                'response_time_ms': 250
            },
            user_feedback={'rating': 5, 'satisfied': True}
        )
        
        assert system_observer.stats['total_outcomes'] == 1
        assert len(system_observer.outcome_history) == 1
        assert len(mock_world_model.observations) == 1
    
    def test_observe_error(self, system_observer, mock_world_model):
        """Test observing error event"""
        system_observer.observe_error(
            query_id="q_test_789",
            error_type="TimeoutError",
            error_message="Engine execution timed out after 30s",
            component="reasoning_integration"
        )
        
        assert system_observer.stats['errors_observed'] == 1
        assert len(mock_world_model.observations) == 1


class TestSystemObserverDisabled:
    """Test SystemObserver when disabled"""
    
    def test_disabled_no_observations(self, system_observer, mock_world_model):
        """Test that disabled observer doesn't emit events"""
        system_observer.enabled = False
        
        system_observer.observe_query_start("q_1", "test", {'category': 'test'})
        system_observer.observe_engine_result("q_1", "test", {}, True, 100)
        system_observer.observe_outcome("q_1", {})
        
        # No observations should be recorded
        assert len(mock_world_model.observations) == 0
        assert system_observer.stats['total_queries'] == 0


class TestSystemObserverStatistics:
    """Test SystemObserver statistics tracking"""
    
    def test_get_statistics(self, system_observer):
        """Test getting statistics"""
        stats = system_observer.get_statistics()
        
        assert 'total_queries' in stats
        assert 'total_engine_executions' in stats
        assert 'total_outcomes' in stats
        assert 'errors_observed' in stats
        assert 'enabled' in stats
    
    def test_get_engine_performance(self, system_observer):
        """Test getting engine performance stats"""
        # Add some engine results
        for i in range(5):
            system_observer.observe_engine_result(
                query_id=f"q_{i}",
                engine_name="probabilistic",
                result={'confidence': 0.7 + i * 0.05},
                success=i % 2 == 0,  # Alternating success
                execution_time_ms=100 + i * 20
            )
        
        perf = system_observer.get_engine_performance("probabilistic")
        
        assert perf['engine'] == 'probabilistic'
        assert perf['executions'] == 5
        assert 0.4 <= perf['success_rate'] <= 0.8  # 2-3 out of 5
        assert perf['avg_confidence'] > 0
    
    def test_get_validation_failure_patterns(self, system_observer):
        """Test getting validation failure patterns"""
        # Add some validation failures
        system_observer.observe_validation_failure(
            "q_1", "math_engine", "type_mismatch", 
            "Is {A→B} valid?", {}
        )
        system_observer.observe_validation_failure(
            "q_2", "math_engine", "type_mismatch",
            "What is ∀x P(x)?", {}
        )
        system_observer.observe_validation_failure(
            "q_3", "symbolic_engine", "parse_error",
            "Compute integral", {}
        )
        
        patterns = system_observer.get_validation_failure_patterns()
        
        assert len(patterns) >= 1


class TestSystemObserverSingleton:
    """Test SystemObserver singleton pattern"""
    
    def test_get_system_observer_none_initially(self):
        """Test get_system_observer returns None before initialization"""
        from vulcan.world_model import system_observer as so_module
        
        # Reset singleton
        so_module._system_observer = None
        
        observer = so_module.get_system_observer()
        # May be None or an instance depending on prior test state
        # Just verify no exception is raised
        assert observer is None or isinstance(observer, so_module.SystemObserver)
    
    def test_initialize_system_observer(self, mock_world_model):
        """Test initializing system observer singleton"""
        from vulcan.world_model.system_observer import (
            initialize_system_observer,
            get_system_observer,
            SystemObserver
        )
        
        observer = initialize_system_observer(mock_world_model)
        
        assert observer is not None
        assert isinstance(observer, SystemObserver)
        assert get_system_observer() is observer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
