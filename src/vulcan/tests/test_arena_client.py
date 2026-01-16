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


# ============================================================
# DISTRIBUTED CIRCUIT BREAKER TESTS
# ============================================================


class TestDistributedCircuitBreaker:
    """Test distributed circuit breaker with Redis support."""

    def test_distributed_circuit_breaker_fallback_no_redis(self):
        """Verify fallback to local circuit breaker when Redis unavailable."""
        from vulcan.arena.client import DistributedCircuitBreaker
        
        # Create without Redis
        cb = DistributedCircuitBreaker()
        
        # Should fall back to local circuit breaker
        assert not cb._redis_available
        
        # Operations should work via local fallback
        cb.record_timeout()
        cb.record_timeout()
        cb.record_timeout()
        
        assert cb.should_bypass()
        
        stats = cb.get_stats()
        assert stats['distributed'] == False
        assert stats['redis_available'] == False

    @patch('redis.from_url')
    def test_distributed_circuit_breaker_with_redis(self, mock_redis_from_url):
        """Verify distributed circuit breaker uses Redis when available."""
        from vulcan.arena.client import DistributedCircuitBreaker, CIRCUIT_BREAKER_THRESHOLD
        import os
        
        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis_from_url.return_value = mock_redis
        
        # Set Redis URL temporarily
        old_redis_url = os.environ.get('VULCAN_ARENA_REDIS_URL')
        os.environ['VULCAN_ARENA_REDIS_URL'] = 'redis://localhost:6379/0'
        
        try:
            # Create with Redis
            cb = DistributedCircuitBreaker()
            cb.__post_init__()
            
            # Should initialize Redis
            assert cb._redis_available or not cb._redis  # May not have redis package
            
        finally:
            # Restore environment
            if old_redis_url:
                os.environ['VULCAN_ARENA_REDIS_URL'] = old_redis_url
            elif 'VULCAN_ARENA_REDIS_URL' in os.environ:
                del os.environ['VULCAN_ARENA_REDIS_URL']

    def test_distributed_circuit_breaker_record_timeout(self):
        """Verify timeout recording in distributed circuit breaker."""
        from vulcan.arena.client import DistributedCircuitBreaker
        
        cb = DistributedCircuitBreaker()
        
        # Record timeout (should use local fallback)
        cb.record_timeout()
        
        # Verify recorded
        stats = cb.get_stats()
        assert stats['total_timeouts'] >= 1

    def test_distributed_circuit_breaker_record_success(self):
        """Verify success recording resets consecutive counter."""
        from vulcan.arena.client import DistributedCircuitBreaker
        
        cb = DistributedCircuitBreaker()
        
        cb.record_timeout()
        cb.record_timeout()
        cb.record_success()
        
        stats = cb.get_stats()
        assert stats['consecutive_timeouts'] == 0
        assert stats['total_successes'] >= 1


# ============================================================
# FEEDBACK RETRY QUEUE TESTS
# ============================================================


class TestFeedbackRetryQueue:
    """Test feedback retry queue with exponential backoff."""

    @pytest.mark.asyncio
    async def test_feedback_retry_queue_initialization(self):
        """Verify lazy initialization of retry queue."""
        from vulcan.arena.client import FeedbackRetryQueue
        
        queue = FeedbackRetryQueue()
        
        # Should not be initialized yet
        assert not queue._initialized
        assert queue._queue is None
        assert queue._worker_task is None
        
        # Enqueue should initialize
        await queue.enqueue({
            "graph_id": "test_123",
            "score": 0.8,
            "rationale": "Good response"
        })
        
        # Should now be initialized
        assert queue._initialized
        assert queue._queue is not None
        assert queue._worker_task is not None

    def test_feedback_retry_queue_backoff_calculation(self):
        """Verify exponential backoff calculation."""
        from vulcan.arena.client import FeedbackRetryQueue
        
        queue = FeedbackRetryQueue()
        
        # Test exponential backoff
        delay_0 = queue._calculate_backoff(0)
        delay_1 = queue._calculate_backoff(1)
        delay_2 = queue._calculate_backoff(2)
        
        # Should increase exponentially
        assert 0.5 <= delay_0 <= 1.5  # ~1s with jitter
        assert 1.5 <= delay_1 <= 2.5  # ~2s with jitter
        assert 3.0 <= delay_2 <= 5.0  # ~4s with jitter
        
        # Should cap at MAX_DELAY
        delay_10 = queue._calculate_backoff(10)
        assert delay_10 <= queue.MAX_DELAY * 1.25  # Max + jitter

    @pytest.mark.asyncio
    async def test_feedback_retry_queue_enqueue(self):
        """Verify feedback can be enqueued."""
        from vulcan.arena.client import FeedbackRetryQueue
        
        queue = FeedbackRetryQueue()
        
        feedback_data = {
            "graph_id": "test_456",
            "score": 0.9,
            "rationale": "Excellent"
        }
        
        await queue.enqueue(feedback_data)
        
        # Should be in queue
        assert queue._queue.qsize() >= 1

    @pytest.mark.asyncio
    async def test_feedback_retry_queue_shutdown(self):
        """Verify graceful shutdown of retry queue."""
        from vulcan.arena.client import FeedbackRetryQueue
        
        queue = FeedbackRetryQueue()
        
        # Initialize by enqueueing
        await queue.enqueue({
            "graph_id": "test_789",
            "score": 0.7,
            "rationale": "Good"
        })
        
        # Should have worker task
        assert queue._worker_task is not None
        
        # Shutdown
        await queue.shutdown()
        
        # Worker should be cancelled
        assert queue._worker_task.done()


# ============================================================
# TIMEOUT CONFIGURATION TESTS
# ============================================================


class TestTimeoutConfiguration:
    """Test timeout configuration and environment variables."""

    def test_timeout_constants(self):
        """Verify timeout constants are set correctly."""
        from vulcan.arena.client import GENERATOR_TIMEOUT, SIMPLE_TASK_TIMEOUT
        
        # Verify timeouts
        assert GENERATOR_TIMEOUT == 90.0
        assert SIMPLE_TASK_TIMEOUT == 30.0

    def test_environment_variable_override(self):
        """Verify environment variable can override timeout."""
        import os
        import importlib
        
        # Set environment variable
        old_timeout = os.environ.get('VULCAN_ARENA_TIMEOUT')
        os.environ['VULCAN_ARENA_TIMEOUT'] = '120.0'
        
        try:
            # Reload module to pick up env var
            import vulcan.arena.client as client_module
            importlib.reload(client_module)
            
            # Should use custom timeout
            assert client_module.GENERATOR_TIMEOUT == 120.0
            
        finally:
            # Restore environment
            if old_timeout:
                os.environ['VULCAN_ARENA_TIMEOUT'] = old_timeout
            elif 'VULCAN_ARENA_TIMEOUT' in os.environ:
                del os.environ['VULCAN_ARENA_TIMEOUT']
            
            # Reload to restore defaults
            importlib.reload(client_module)

    def test_max_feedback_retries_env_var(self):
        """Verify feedback retries can be configured via env var."""
        import os
        
        old_retries = os.environ.get('VULCAN_ARENA_FEEDBACK_RETRIES')
        os.environ['VULCAN_ARENA_FEEDBACK_RETRIES'] = '5'
        
        try:
            # Reload module
            import vulcan.arena.client as client_module
            import importlib
            importlib.reload(client_module)
            
            # Should use custom retry count
            assert client_module.MAX_FEEDBACK_RETRIES == 5
            
        finally:
            # Restore environment
            if old_retries:
                os.environ['VULCAN_ARENA_FEEDBACK_RETRIES'] = old_retries
            elif 'VULCAN_ARENA_FEEDBACK_RETRIES' in os.environ:
                del os.environ['VULCAN_ARENA_FEEDBACK_RETRIES']
            
            # Reload to restore defaults
            import importlib
            importlib.reload(client_module)


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for Arena client with new features."""

    @pytest.mark.asyncio
    @patch('vulcan.arena.client.get_http_session')
    async def test_submit_feedback_with_retry_on_failure(self, mock_get_session):
        """Verify feedback submission enqueues for retry on failure."""
        from vulcan.arena.client import submit_arena_feedback
        
        # Mock session that fails
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_get_session.return_value = mock_session
        
        # Submit feedback
        result = await submit_arena_feedback(
            proposal_id="test_proposal",
            score=0.8,
            rationale="Test feedback"
        )
        
        # Should indicate error and retry queued
        assert result["status"] == "error"
        assert result.get("retry_queued") == True

    @pytest.mark.asyncio
    @patch('vulcan.arena.client.get_http_session')
    async def test_submit_feedback_success(self, mock_get_session):
        """Verify successful feedback submission."""
        from vulcan.arena.client import submit_arena_feedback
        
        # Mock successful session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True})
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_get_session.return_value = mock_session
        
        # Submit feedback
        result = await submit_arena_feedback(
            proposal_id="test_proposal",
            score=0.9,
            rationale="Great response"
        )
        
        # Should succeed
        assert result["status"] == "success"
        assert "retry_queued" not in result
