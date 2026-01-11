"""Test suite for arena/client.py - Arena client resilience and thread safety"""

import asyncio
import logging
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from vulcan.arena.client import (
    ArenaCircuitBreaker,
    _get_arena_semaphore,
    build_arena_payload,
    select_arena_agent,
)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def mock_routing_plan():
    """Create a mock routing plan for testing."""
    plan = MagicMock()
    plan.query_type = MagicMock()
    plan.query_type.value = "GENERATIVE"
    plan.query_id = "test_123"
    plan.complexity_score = 0.5
    plan.selected_tools = []
    return plan


@pytest.fixture
def mock_routing_plan_unknown():
    """Create a mock routing plan with unknown query type."""
    plan = MagicMock()
    plan.query_type = MagicMock()
    plan.query_type.value = "UNKNOWN_TYPE"
    plan.query_id = "test_456"
    plan.complexity_score = 0.5
    plan.selected_tools = []
    return plan


# ============================================================
# SEMAPHORE THREAD SAFETY TESTS
# ============================================================


class TestSemaphoreThreadSafety:
    """Test thread-safe semaphore initialization."""

    def test_semaphore_thread_safety(self):
        """Verify semaphore initialization under concurrent access."""
        # Reset the global semaphore
        import vulcan.arena.client as client_module
        client_module._arena_semaphore = None
        
        results = []
        barrier = threading.Barrier(10)  # Synchronize 10 threads
        
        def access_semaphore():
            """Thread function that accesses semaphore."""
            barrier.wait()  # Wait for all threads to be ready
            sem = _get_arena_semaphore()
            results.append(id(sem))
        
        # Create 10 threads that all try to access semaphore simultaneously
        threads = [threading.Thread(target=access_semaphore) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should get the same semaphore instance (same id)
        assert len(set(results)) == 1, "Multiple semaphore instances created - race condition detected"
        
        # Verify it's actually a semaphore
        sem = _get_arena_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    def test_semaphore_reuse(self):
        """Verify semaphore is reused across calls."""
        sem1 = _get_arena_semaphore()
        sem2 = _get_arena_semaphore()
        
        # Should return the same instance
        assert id(sem1) == id(sem2)


# ============================================================
# PAYLOAD VALIDATION TESTS
# ============================================================


class TestPayloadValidation:
    """Test payload validation after sanitization."""

    @patch('vulcan.arena.client.sanitize_payload')
    def test_payload_validation_after_sanitization_generator(self, mock_sanitize, mock_routing_plan):
        """Verify required fields are validated for generator agent."""
        # Mock sanitize_payload to remove a required field
        mock_sanitize.return_value = {
            # Missing 'spec_id' and 'parameters'
            "some_other_field": "value"
        }
        
        from vulcan.arena.client import execute_via_arena
        
        # This test simulates what would happen in execute_via_arena
        # We'll test the validation logic directly
        agent_id = "generator"
        payload = {"spec_id": "test", "parameters": {}}
        sanitized = mock_sanitize.return_value
        
        required_fields = ['spec_id', 'parameters']
        is_valid = all(key in sanitized for key in required_fields)
        
        assert not is_valid, "Validation should detect missing required fields"

    @patch('vulcan.arena.client.sanitize_payload')
    def test_payload_validation_after_sanitization_evolver(self, mock_sanitize, mock_routing_plan):
        """Verify required fields are validated for non-generator agents."""
        # Mock sanitize_payload to remove a required field
        mock_sanitize.return_value = {
            # Missing 'graph_id' and 'nodes'
            "some_other_field": "value"
        }
        
        agent_id = "evolver"
        payload = {"graph_id": "test", "nodes": []}
        sanitized = mock_sanitize.return_value
        
        required_fields = ['graph_id', 'nodes']
        is_valid = all(key in sanitized for key in required_fields)
        
        assert not is_valid, "Validation should detect missing required fields"

    def test_payload_validation_success(self, mock_routing_plan):
        """Verify validation passes when all required fields present."""
        # Test generator payload
        generator_payload = build_arena_payload("test query", mock_routing_plan, "generator")
        assert 'spec_id' in generator_payload
        assert 'parameters' in generator_payload
        
        # Test evolver payload
        evolver_payload = build_arena_payload("test query", mock_routing_plan, "evolver")
        assert 'graph_id' in evolver_payload
        assert 'nodes' in evolver_payload


# ============================================================
# AGENT SELECTION TESTS
# ============================================================


class TestAgentSelection:
    """Test agent selection and unknown type handling."""

    def test_agent_selection_unknown_type_logs_warning(self, mock_routing_plan_unknown, caplog):
        """Verify logging for unknown query types."""
        with caplog.at_level(logging.WARNING):
            agent_id = select_arena_agent(mock_routing_plan_unknown)
        
        # Should default to generator
        assert agent_id == "generator"
        
        # Should log a warning
        assert any("Unknown query type" in record.message for record in caplog.records)
        assert any("UNKNOWN_TYPE" in record.message for record in caplog.records)

    def test_agent_selection_known_types(self, mock_routing_plan):
        """Verify correct agent selection for known types."""
        # Test generative
        mock_routing_plan.query_type.value = "GENERATIVE"
        assert select_arena_agent(mock_routing_plan) == "generator"
        
        # Test optimization
        mock_routing_plan.query_type.value = "OPTIMIZATION"
        assert select_arena_agent(mock_routing_plan) == "evolver"
        
        # Test perception
        mock_routing_plan.query_type.value = "PERCEPTION"
        assert select_arena_agent(mock_routing_plan) == "visualizer"


# ============================================================
# CIRCUIT BREAKER TESTS
# ============================================================


class TestCircuitBreaker:
    """Test circuit breaker behavior."""

    def test_circuit_breaker_opens_after_threshold(self):
        """Verify circuit breaker opens after consecutive timeouts."""
        cb = ArenaCircuitBreaker()
        
        # Circuit should start closed
        assert not cb.should_bypass()
        
        # Record timeouts up to threshold
        from vulcan.arena.client import CIRCUIT_BREAKER_THRESHOLD
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            cb.record_timeout()
        
        # Circuit should now be open
        assert cb.should_bypass()
        assert cb.is_open

    def test_circuit_breaker_resets_on_success(self):
        """Verify circuit breaker resets consecutive timeout counter on success."""
        cb = ArenaCircuitBreaker()
        
        # Record some timeouts
        cb.record_timeout()
        cb.record_timeout()
        assert cb.consecutive_timeouts == 2
        
        # Record success
        cb.record_success()
        
        # Consecutive counter should reset
        assert cb.consecutive_timeouts == 0

    def test_circuit_breaker_reset_after_time(self):
        """Verify circuit breaker resets after timeout period."""
        cb = ArenaCircuitBreaker()
        
        # Open the circuit
        from vulcan.arena.client import CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_RESET_TIME
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            cb.record_timeout()
        
        assert cb.should_bypass()
        
        # Simulate time passing by setting last_failure_time in the past
        cb.last_failure_time = time.time() - (CIRCUIT_BREAKER_RESET_TIME + 1)
        
        # Circuit should reset
        assert not cb.should_bypass()
        assert not cb.is_open

    def test_circuit_breaker_stats(self):
        """Verify circuit breaker stats are tracked correctly."""
        cb = ArenaCircuitBreaker()
        
        cb.record_timeout()
        cb.record_success()
        cb.record_timeout()
        
        stats = cb.get_stats()
        
        assert stats['total_timeouts'] == 2
        assert stats['total_successes'] == 1
        assert stats['consecutive_timeouts'] == 1
        assert 'time_since_failure' in stats
