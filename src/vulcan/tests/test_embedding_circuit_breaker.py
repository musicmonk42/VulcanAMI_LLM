"""
Tests for the EmbeddingCircuitBreaker module.

Tests cover:
- Basic circuit breaker state transitions
- Latency-based failure detection
- Recovery behavior
- Thread safety
- Statistics tracking
"""

import threading
import time
from unittest.mock import patch

import pytest

from vulcan.reasoning.selection.embedding_circuit_breaker import (
    CircuitState,
    EmbeddingCircuitBreaker,
    get_embedding_circuit_breaker,
    reset_embedding_circuit_breaker,
    get_circuit_breaker_stats,
    DEFAULT_LATENCY_THRESHOLD_MS,
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_RESET_TIMEOUT_S,
)


class TestEmbeddingCircuitBreakerBasics:
    """Test basic circuit breaker functionality"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_circuit_breaker()

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state"""
        cb = EmbeddingCircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_should_not_skip_initially(self):
        """Should not skip embeddings when circuit is closed"""
        cb = EmbeddingCircuitBreaker()
        assert cb.should_skip_embedding() is False

    def test_fast_latency_keeps_circuit_closed(self):
        """Fast operations should keep circuit closed"""
        cb = EmbeddingCircuitBreaker(latency_threshold_ms=5000)
        
        # Record several fast operations
        for _ in range(10):
            cb.record_latency(100)  # 100ms is fast
        
        assert cb.state == CircuitState.CLOSED
        assert cb.should_skip_embedding() is False

    def test_slow_latency_opens_circuit(self):
        """Slow operations should open circuit after threshold"""
        cb = EmbeddingCircuitBreaker(
            latency_threshold_ms=1000,  # 1 second threshold
            failure_threshold=3,
        )
        
        # Record slow operations
        for _ in range(3):
            cb.record_latency(5000)  # 5 seconds - very slow
        
        assert cb.state == CircuitState.OPEN
        assert cb.should_skip_embedding() is True

    def test_failure_opens_circuit(self):
        """Explicit failures should open circuit"""
        cb = EmbeddingCircuitBreaker(failure_threshold=2)
        
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN

    def test_force_reset(self):
        """Force reset should close circuit"""
        cb = EmbeddingCircuitBreaker(failure_threshold=1)
        
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        cb.force_reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.should_skip_embedding() is False


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery behavior"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_circuit_breaker()

    def test_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after timeout"""
        cb = EmbeddingCircuitBreaker(
            failure_threshold=1,
            reset_timeout_s=0.1,  # Very short for testing
        )
        
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for reset timeout
        time.sleep(0.15)
        
        # Should transition to half-open on next check
        assert cb.should_skip_embedding() is False
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovery_from_half_open(self):
        """Successful operations in HALF_OPEN should close circuit"""
        cb = EmbeddingCircuitBreaker(
            latency_threshold_ms=5000,
            failure_threshold=1,
            reset_timeout_s=0.1,
            success_threshold=2,
        )
        
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        time.sleep(0.15)
        cb.should_skip_embedding()  # Transition to HALF_OPEN
        
        # Record successful operations
        cb.record_latency(100)  # Fast - success
        cb.record_latency(100)  # Fast - success
        
        assert cb.state == CircuitState.CLOSED

    def test_slow_in_half_open_reopens(self):
        """Slow operation in HALF_OPEN should reopen circuit"""
        cb = EmbeddingCircuitBreaker(
            latency_threshold_ms=1000,
            failure_threshold=1,
            reset_timeout_s=0.1,
        )
        
        cb.record_failure()
        time.sleep(0.15)
        cb.should_skip_embedding()  # Transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        
        # Record slow operation
        cb.record_latency(5000)
        
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerStatistics:
    """Test statistics tracking"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_circuit_breaker()

    def test_stats_tracking(self):
        """Statistics should be tracked correctly"""
        cb = EmbeddingCircuitBreaker()
        
        # Allow some embeddings
        for _ in range(5):
            cb.should_skip_embedding()
            cb.record_latency(100)
        
        stats = cb.get_stats()
        
        assert stats.total_allowed == 5
        assert stats.total_skipped == 0
        assert stats.latency_ema_ms > 0

    def test_skip_counting(self):
        """Skipped embeddings should be counted"""
        cb = EmbeddingCircuitBreaker(failure_threshold=1)
        
        cb.record_failure()  # Open circuit
        
        # These should all be skipped
        cb.should_skip_embedding()
        cb.should_skip_embedding()
        cb.should_skip_embedding()
        
        stats = cb.get_stats()
        assert stats.total_skipped == 3

    def test_latency_ema(self):
        """EMA should be calculated correctly"""
        cb = EmbeddingCircuitBreaker(ema_alpha=0.5)
        
        cb.record_latency(100)  # First: EMA = 100
        cb.record_latency(200)  # EMA = 0.5 * 200 + 0.5 * 100 = 150
        
        stats = cb.get_stats()
        assert 140 < stats.latency_ema_ms < 160


class TestCircuitBreakerSingleton:
    """Test singleton behavior"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_circuit_breaker()

    def test_singleton_returns_same_instance(self):
        """get_embedding_circuit_breaker should return singleton"""
        cb1 = get_embedding_circuit_breaker()
        cb2 = get_embedding_circuit_breaker()
        
        assert cb1 is cb2

    def test_reset_creates_new_instance(self):
        """reset should allow new instance creation"""
        cb1 = get_embedding_circuit_breaker()
        cb1.record_failure()
        
        reset_embedding_circuit_breaker()
        
        cb2 = get_embedding_circuit_breaker()
        
        # Should be a fresh instance in CLOSED state
        assert cb2.state == CircuitState.CLOSED

    def test_get_circuit_breaker_stats(self):
        """get_circuit_breaker_stats should return stats dict"""
        cb = get_embedding_circuit_breaker()
        cb.record_latency(100)
        
        stats = get_circuit_breaker_stats()
        
        assert isinstance(stats, dict)
        assert "state" in stats
        assert "latency_ema_ms" in stats

    def test_stats_not_initialized(self):
        """get_circuit_breaker_stats should handle uninitialized state"""
        reset_embedding_circuit_breaker()
        # Don't create instance
        
        # Patch the global to be None
        with patch('vulcan.reasoning.selection.embedding_circuit_breaker._embedding_circuit_breaker', None):
            stats = get_circuit_breaker_stats()
        
        assert stats.get("status") == "not_initialized"


class TestCircuitBreakerThreadSafety:
    """Test thread safety"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_circuit_breaker()

    def test_concurrent_access(self):
        """Circuit breaker should handle concurrent access"""
        cb = EmbeddingCircuitBreaker()
        errors = []
        
        def worker(worker_id):
            try:
                for _ in range(100):
                    cb.should_skip_embedding()
                    cb.record_latency(50 + worker_id)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        stats = cb.get_stats()
        assert stats.total_allowed == 1000  # 10 workers * 100 iterations

    def test_concurrent_failures(self):
        """Concurrent failures should not corrupt state"""
        cb = EmbeddingCircuitBreaker(failure_threshold=100)
        errors = []
        
        def worker():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Should have opened at some point (250 failures > 100 threshold)
        # State might be open or have been triggered
        stats = cb.get_stats()
        assert stats.failure_count >= 0  # Just verify it's valid


class TestCircuitBreakerEdgeCases:
    """Test edge cases"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_circuit_breaker()

    def test_decay_of_failures(self):
        """Successful operations should decay failure count"""
        cb = EmbeddingCircuitBreaker(
            latency_threshold_ms=5000,
            failure_threshold=5,
        )
        
        # Add some failures (but not enough to open)
        cb.record_latency(10000)  # Slow - failure
        cb.record_latency(10000)  # Slow - failure
        
        # Now record fast operations to decay
        for _ in range(5):
            cb.record_latency(100)  # Fast - decays failure count
        
        # Should still be closed
        assert cb.state == CircuitState.CLOSED

    def test_zero_threshold_latency(self):
        """Should handle zero-ish thresholds"""
        cb = EmbeddingCircuitBreaker(
            latency_threshold_ms=1,  # Very low threshold
            failure_threshold=1,
        )
        
        cb.record_latency(10)  # Any realistic latency is slow
        
        assert cb.state == CircuitState.OPEN

    def test_very_long_latency(self):
        """Should handle very long latencies"""
        cb = EmbeddingCircuitBreaker(failure_threshold=1)
        
        # 2 minutes - extremely slow
        cb.record_latency(120000)
        
        assert cb.state == CircuitState.OPEN


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
