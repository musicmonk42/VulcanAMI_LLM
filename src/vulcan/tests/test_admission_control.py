"""
Comprehensive test suite for admission_control.py

Tests rate limiting, circuit breakers, resource monitoring, priority queues,
and adaptive admission control with proper thread safety validation.
"""

# Import the module to test
from vulcan.reasoning.selection.admission_control import (
    AdaptiveAdmissionController,
    AdmissionControlIntegration,
    AdmissionDecision,
    AdmissionMetrics,
    CircuitBreaker,
    PriorityQueue,
    Request,
    RequestPriority,
    ResourceMonitor,
    SlidingWindowRateLimiter,
    SystemHealth,
    TokenBucketRateLimiter,
)
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEnums:
    """Test enum definitions"""

    def test_request_priority_values(self):
        """Test RequestPriority enum values"""
        assert RequestPriority.CRITICAL.value == 0
        assert RequestPriority.HIGH.value == 1
        assert RequestPriority.NORMAL.value == 2
        assert RequestPriority.LOW.value == 3
        assert RequestPriority.BATCH.value == 4

    def test_admission_decision_values(self):
        """Test AdmissionDecision enum values"""
        assert AdmissionDecision.ADMIT.value == "admit"
        assert AdmissionDecision.REJECT.value == "reject"
        assert AdmissionDecision.DEFER.value == "defer"
        assert AdmissionDecision.REDIRECT.value == "redirect"

    def test_system_health_values(self):
        """Test SystemHealth enum values"""
        assert SystemHealth.HEALTHY.value == "healthy"
        assert SystemHealth.DEGRADED.value == "degraded"
        assert SystemHealth.OVERLOADED.value == "overloaded"
        assert SystemHealth.CRITICAL.value == "critical"


class TestRequest:
    """Test Request dataclass"""

    def test_request_creation(self):
        """Test creating requests"""
        request = Request(
            request_id="test_001",
            priority=RequestPriority.HIGH,
            estimated_cost={"time_ms": 100, "energy_mj": 50},
            context={"user": "test"},
        )

        assert request.request_id == "test_001"
        assert request.priority == RequestPriority.HIGH
        assert request.estimated_cost["time_ms"] == 100
        assert "user" in request.context

    def test_request_defaults(self):
        """Test request default values"""
        request = Request(
            request_id="test_002",
            priority=RequestPriority.NORMAL,
            estimated_cost={},
            context={},
        )

        assert isinstance(request.timestamp, float)
        assert request.timestamp > 0
        assert request.deadline is None
        assert isinstance(request.metadata, dict)


class TestAdmissionMetrics:
    """Test AdmissionMetrics dataclass"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = AdmissionMetrics()

        assert metrics.total_requests == 0
        assert metrics.admitted_requests == 0
        assert metrics.rejected_requests == 0
        assert metrics.current_queue_depth == 0
        assert metrics.rejection_rate == 0.0


class TestTokenBucketRateLimiter:
    """Test TokenBucketRateLimiter"""

    def test_initialization(self):
        """Test rate limiter initialization"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=20)

        assert limiter.rate == 10
        assert limiter.capacity == 20
        assert limiter.tokens == 20

    def test_consume_success(self):
        """Test successful token consumption"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=20)

        assert limiter.consume(5) is True
        assert limiter.tokens == 15

    def test_consume_failure(self):
        """Test failed token consumption"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=20)

        # Consume all tokens
        limiter.consume(20)

        # Should fail
        assert limiter.consume(1) is False

    def test_token_refill(self):
        """Test token refill over time"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=20)

        # Consume all tokens
        limiter.consume(20)

        # Wait for refill
        time.sleep(0.5)

        # Should have some tokens back (rate=10/s, so ~5 tokens in 0.5s)
        assert limiter.available_tokens() > 0

    def test_available_tokens(self):
        """Test available tokens check"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=20)

        available = limiter.available_tokens()
        assert available == 20

        limiter.consume(10)
        available = limiter.available_tokens()
        # Use approximate equality due to time elapsed and floating-point precision
        assert abs(available - 10) < 0.1  # Allow small tolerance for time-based refill

    def test_thread_safety(self):
        """Test thread-safe token consumption"""
        limiter = TokenBucketRateLimiter(rate=100, capacity=100)

        results = []

        def consume_tokens():
            for _ in range(10):
                results.append(limiter.consume(1))

        threads = [threading.Thread(target=consume_tokens) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have mix of success and failure
        assert True in results
        # Total consumption should not exceed capacity
        successes = sum(1 for r in results if r)
        assert successes <= 100


class TestSlidingWindowRateLimiter:
    """Test SlidingWindowRateLimiter"""

    def test_initialization(self):
        """Test sliding window initialization"""
        limiter = SlidingWindowRateLimiter(window_size_seconds=1, max_requests=10)

        assert limiter.window_size == 1
        assert limiter.max_requests == 10

    def test_allow_within_limit(self):
        """Test allowing requests within limit"""
        limiter = SlidingWindowRateLimiter(window_size_seconds=1, max_requests=5)

        for _ in range(5):
            assert limiter.allow_request() is True

    def test_reject_over_limit(self):
        """Test rejecting requests over limit"""
        limiter = SlidingWindowRateLimiter(window_size_seconds=1, max_requests=5)

        # Use up limit
        for _ in range(5):
            limiter.allow_request()

        # Should reject
        assert limiter.allow_request() is False

    def test_window_sliding(self):
        """Test that window slides over time"""
        limiter = SlidingWindowRateLimiter(window_size_seconds=1, max_requests=5)

        # Fill up window
        for _ in range(5):
            limiter.allow_request()

        # Wait for window to slide
        time.sleep(1.1)

        # Should allow again
        assert limiter.allow_request() is True

    def test_current_rate(self):
        """Test current rate calculation"""
        limiter = SlidingWindowRateLimiter(window_size_seconds=1, max_requests=10)

        # Add some requests
        for _ in range(3):
            limiter.allow_request()

        rate = limiter.current_rate()
        assert rate == 3.0


class TestCircuitBreaker:
    """Test CircuitBreaker"""

    def test_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=10)

        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_remains_closed_under_threshold(self):
        """Test circuit remains closed under failure threshold"""
        cb = CircuitBreaker(failure_threshold=5)

        for _ in range(4):
            cb.record_failure()

        assert cb.state == "closed"
        assert cb.is_available() is True

    def test_opens_at_threshold(self):
        """Test circuit opens at failure threshold"""
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.is_available() is False

    def test_success_reduces_failures(self):
        """Test successes reduce failure count"""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 1

    def test_half_open_transition(self):
        """Test transition to half-open state"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Wait for recovery timeout
        time.sleep(0.6)

        # Should transition to half-open
        assert cb.is_available() is True
        assert cb.state == "half-open"

    def test_recovery_after_successes(self):
        """Test circuit closes after successes in half-open"""
        cb = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.2, success_threshold=2
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Wait and transition to half-open
        time.sleep(0.3)
        cb.is_available()

        # Record successes
        cb.record_success()
        cb.record_success()

        assert cb.state == "closed"


class TestResourceMonitor:
    """Test ResourceMonitor"""

    @pytest.fixture
    def monitor(self):
        """Create monitor for testing"""
        monitor = ResourceMonitor(
            cpu_threshold=80, memory_threshold=80, queue_threshold=100
        )
        yield monitor
        monitor.stop()

    def test_initialization(self, monitor):
        """Test resource monitor initialization"""
        assert monitor.cpu_threshold == 80
        assert monitor.memory_threshold == 80
        assert not monitor._shutdown_event.is_set()
        assert monitor.monitor_thread.is_alive()

    def test_get_current_load(self, monitor):
        """Test getting current system load"""
        # Wait for monitor to collect some data
        time.sleep(0.2)

        load = monitor.get_current_load()

        assert "cpu_percent" in load
        assert "memory_percent" in load
        assert "cpu_peak" in load
        assert "memory_peak" in load
        assert load["cpu_percent"] >= 0

    def test_overload_detection(self, monitor):
        """Test overload detection"""
        # With normal thresholds, system should not be overloaded
        is_overloaded, reason = monitor.is_overloaded()

        assert isinstance(is_overloaded, bool)
        assert isinstance(reason, str)

    def test_queue_depth_tracking(self, monitor):
        """Test queue depth tracking"""
        monitor.update_queue_depth("test_queue", 50)
        monitor.update_queue_depth("test_queue", 75)
        monitor.update_queue_depth("test_queue", 100)

        # Queue depths should be tracked
        assert "test_queue" in monitor.queue_depths

    def test_monitor_thread_running(self, monitor):
        """Test that monitor thread is running"""
        assert monitor.monitor_thread.is_alive()

    def test_stop_monitoring(self, monitor):
        """Test stopping the monitor"""
        monitor.stop()

        assert monitor._shutdown_event.is_set()


class TestPriorityQueue:
    """Test PriorityQueue"""

    def test_initialization(self):
        """Test priority queue initialization"""
        queue = PriorityQueue(max_size=100)

        assert queue.max_size == 100
        assert queue.size() == 0

    def test_put_and_get(self):
        """Test putting and getting requests"""
        queue = PriorityQueue(max_size=100)

        request = Request(
            request_id="test_1",
            priority=RequestPriority.NORMAL,
            estimated_cost={},
            context={},
        )

        assert queue.put(request) is True
        assert queue.size() == 1

        retrieved = queue.get(timeout=1.0)
        assert retrieved is not None
        assert retrieved.request_id == "test_1"

    def test_priority_ordering(self):
        """Test that higher priority requests come first"""
        queue = PriorityQueue(max_size=100)

        # Add requests in different order
        low = Request("low", RequestPriority.LOW, {}, {})
        high = Request("high", RequestPriority.HIGH, {}, {})
        critical = Request("critical", RequestPriority.CRITICAL, {}, {})

        queue.put(low)
        queue.put(high)
        queue.put(critical)

        # Should get in priority order
        first = queue.get(timeout=0.1)
        assert first.request_id == "critical"

        second = queue.get(timeout=0.1)
        assert second.request_id == "high"

        third = queue.get(timeout=0.1)
        assert third.request_id == "low"

    def test_max_size_limit(self):
        """Test queue size limiting"""
        queue = PriorityQueue(max_size=5)

        # Fill queue
        for i in range(5):
            request = Request(f"req_{i}", RequestPriority.NORMAL, {}, {})
            assert queue.put(request) is True

        # Should reject when full
        overflow = Request("overflow", RequestPriority.NORMAL, {}, {})
        assert queue.put(overflow) is False

    def test_get_timeout(self):
        """Test get with timeout"""
        queue = PriorityQueue(max_size=100)

        # Queue is empty, should timeout
        start = time.time()
        result = queue.get(timeout=0.5)
        elapsed = time.time() - start

        assert result is None
        assert 0.4 < elapsed < 0.7

    def test_remove(self):
        """Test removing requests"""
        queue = PriorityQueue(max_size=100)

        request = Request("test", RequestPriority.NORMAL, {}, {})
        queue.put(request)

        assert queue.size() == 1
        assert queue.remove("test") is True
        assert queue.size() == 0

    def test_clear_expired(self):
        """Test clearing expired requests"""
        queue = PriorityQueue(max_size=100)

        # Add request with past deadline
        expired = Request(
            "expired", RequestPriority.NORMAL, {}, {}, deadline=time.time() - 10
        )
        queue.put(expired)

        # Add request with future deadline
        valid = Request(
            "valid", RequestPriority.NORMAL, {}, {}, deadline=time.time() + 100
        )
        queue.put(valid)

        expired_count = queue.clear_expired()

        assert expired_count == 1
        assert queue.size() == 1


class TestAdaptiveAdmissionController:
    """Test AdaptiveAdmissionController"""

    @pytest.fixture
    def controller(self):
        """Create controller for testing"""
        config = {
            "global_rate": 100,
            "burst_capacity": 200,
            "cpu_threshold": 90,
            "memory_threshold": 90,
            "max_queue_size": 100,
        }
        controller = AdaptiveAdmissionController(config)
        yield controller
        controller.shutdown(timeout=2.0)

    def test_initialization(self, controller):
        """Test controller initialization"""
        assert controller.global_rate_limiter is not None
        assert controller.circuit_breaker is not None
        assert controller.resource_monitor is not None
        assert controller.queue is not None

    def test_admit_normal_request(self, controller):
        """Test admitting normal request"""
        request = Request(
            request_id="test_normal",
            priority=RequestPriority.NORMAL,
            estimated_cost={"time_ms": 100, "energy_mj": 50},
            context={},
        )

        decision, info = controller.admit(request)

        assert decision == AdmissionDecision.ADMIT
        assert isinstance(info, dict)

    def test_reject_when_circuit_open(self, controller, encoding="utf-8"):
        """Test rejection when circuit breaker is open"""
        # Force circuit breaker open
        for _ in range(10):
            controller.circuit_breaker.record_failure()

        request = Request("test", RequestPriority.NORMAL, {}, {})

        decision, info = controller.admit(request)

        assert decision == AdmissionDecision.REJECT
        assert "circuit_breaker" in info["reason"]

    def test_priority_handling(self, controller):
        """Test that critical requests get priority"""
        critical = Request("critical", RequestPriority.CRITICAL, {"time_ms": 100}, {})

        decision, info = controller.admit(critical)

        # Critical requests should be admitted even under load
        assert decision == AdmissionDecision.ADMIT

    def test_rate_limiting(self, controller):
        """Test rate limiting"""
        # Rapidly send many requests
        admitted = 0
        rejected = 0

        for i in range(300):
            request = Request(f"req_{i}", RequestPriority.NORMAL, {"time_ms": 10}, {})

            decision, _ = controller.admit(request)

            if decision == AdmissionDecision.ADMIT:
                admitted += 1
            else:
                rejected += 1

        # Should have rate limited some requests
        assert rejected > 0
        assert admitted > 0

    def test_get_metrics(self, controller):
        """Test getting metrics"""
        # Admit a few requests
        for i in range(5):
            request = Request(f"req_{i}", RequestPriority.NORMAL, {}, {})
            controller.admit(request)

        metrics = controller.get_metrics()

        assert isinstance(metrics, AdmissionMetrics)
        assert metrics.total_requests >= 5

    def test_reset_metrics(self, controller):
        """Test resetting metrics"""
        # Generate some metrics
        request = Request("test", RequestPriority.NORMAL, {}, {})
        controller.admit(request)

        controller.reset_metrics()

        metrics = controller.get_metrics()
        assert metrics.total_requests == 0

    def test_system_health_check(self, controller):
        """Test system health checking"""
        health = controller._check_system_health()

        assert isinstance(health, SystemHealth)

    def test_resource_availability_check(self, controller):
        """Test resource availability checking"""
        request = Request(
            "test", RequestPriority.NORMAL, {"cpu_percent": 5, "memory_mb": 100}, {}
        )

        available = controller._check_resource_availability(request)

        assert isinstance(available, bool)

    def test_shutdown(self, controller):
        """Test controller shutdown"""
        controller.shutdown(timeout=2.0)

        assert controller._is_shutdown is True

        # Should reject after shutdown
        request = Request("test", RequestPriority.NORMAL, {}, {})
        decision, info = controller.admit(request)

        assert decision == AdmissionDecision.REJECT


class TestAdmissionControlIntegration:
    """Test AdmissionControlIntegration"""

    @pytest.fixture
    def integration(self):
        """Create integration for testing"""
        config = {"global_rate": 100, "max_queue_size": 50}
        integration = AdmissionControlIntegration(config)
        yield integration
        integration.shutdown(timeout=2.0)

    def test_initialization(self, integration):
        """Test integration initialization"""
        assert integration.controller is not None
        assert isinstance(integration.request_map, dict)

    def test_check_admission_success(self, integration):
        """Test successful admission check"""
        problem = "test_problem"
        constraints = {"time_budget": 1000, "energy_budget": 500}

        admitted, info = integration.check_admission(
            problem, constraints, priority=RequestPriority.NORMAL
        )

        assert isinstance(admitted, bool)
        assert isinstance(info, dict)

    def test_check_admission_with_callback(self, integration):
        """Test admission with callback"""
        problem = "test_problem"
        constraints = {"time_budget": 1000}
        callback = Mock()

        admitted, info = integration.check_admission(
            problem, constraints, callback=callback
        )

        # Callback should be stored if admitted
        if admitted:
            assert len(integration.request_map) > 0

    def test_process_next(self, integration):
        """Test processing next request"""
        # Admit a request first
        problem = "test_problem"
        constraints = {"time_budget": 1000}

        integration.check_admission(problem, constraints)

        # Process it
        result = integration.process_next(timeout=1.0)

        # May or may not get result depending on timing
        if result:
            assert "request" in result

    def test_get_system_status(self, integration):
        """Test getting system status"""
        status = integration.get_system_status()

        assert isinstance(status, dict)
        if status:  # May be empty on error
            assert "metrics" in status or "resources" in status

    def test_shutdown(self, integration):
        """Test integration shutdown"""
        integration.shutdown(timeout=2.0)

        # Controller should be shutdown
        assert integration.controller._is_shutdown


class TestThreadSafety:
    """Test thread safety of components"""

    def test_concurrent_admissions(self):
        """Test concurrent admission requests"""
        controller = AdaptiveAdmissionController()

        try:
            results = []
            errors = []

            def admit_requests():
                try:
                    for i in range(10):
                        request = Request(
                            f"req_{threading.current_thread().ident}_{i}",
                            RequestPriority.NORMAL,
                            {"time_ms": 100},
                            {},
                        )
                        decision, info = controller.admit(request)
                        results.append(decision)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=admit_requests) for _ in range(5)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have no errors
            assert len(errors) == 0
            # Should have results
            assert len(results) > 0

        finally:
            controller.shutdown()

    def test_concurrent_priority_queue_ops(self):
        """Test concurrent priority queue operations"""
        queue = PriorityQueue(max_size=100)

        def producer():
            for i in range(10):
                request = Request(
                    f"req_{threading.current_thread().ident}_{i}",
                    RequestPriority.NORMAL,
                    {},
                    {},
                )
                queue.put(request)

        def consumer():
            for _ in range(10):
                queue.get(timeout=1.0)

        producers = [threading.Thread(target=producer) for _ in range(3)]
        consumers = [threading.Thread(target=consumer) for _ in range(3)]

        for t in producers + consumers:
            t.start()
        for t in producers + consumers:
            t.join()

        # Should complete without deadlock
        assert True


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_end_to_end_admission_workflow(self):
        """Test complete admission workflow"""
        integration = AdmissionControlIntegration(
            {"global_rate": 50, "max_queue_size": 20}
        )

        try:
            # Submit requests
            admitted_count = 0
            for i in range(10):
                admitted, info = integration.check_admission(
                    problem=f"problem_{i}",
                    constraints={"time_budget": 500},
                    priority=RequestPriority.NORMAL,
                )
                if admitted:
                    admitted_count += 1

            # Process requests
            processed = 0
            while processed < admitted_count:
                result = integration.process_next(timeout=0.5)
                if result:
                    processed += 1
                else:
                    break

            # Get status
            status = integration.get_system_status()
            assert isinstance(status, dict)

        finally:
            integration.shutdown()

    def test_overload_handling(self):
        """Test handling of system overload"""
        controller = AdaptiveAdmissionController(
            {
                "global_rate": 10,  # Very low rate
                "max_queue_size": 5,
            }
        )

        try:
            # Flood with requests
            admitted = 0
            rejected = 0

            for i in range(50):
                request = Request(
                    f"flood_{i}", RequestPriority.NORMAL, {"time_ms": 100}, {}
                )

                decision, _ = controller.admit(request)

                if decision == AdmissionDecision.ADMIT:
                    admitted += 1
                else:
                    rejected += 1

            # Should have rejected some due to rate limiting
            assert rejected > 0

            # Get metrics
            metrics = controller.get_metrics()
            assert metrics.rejected_requests > 0

        finally:
            controller.shutdown()


class TestPerformance:
    """Performance tests"""

    def test_high_throughput_admission(self):
        """Test admission controller under high load"""
        controller = AdaptiveAdmissionController(
            {"global_rate": 1000, "max_queue_size": 500}
        )

        try:
            start = time.time()

            for i in range(100):
                request = Request(
                    f"perf_{i}", RequestPriority.NORMAL, {"time_ms": 10}, {}
                )
                controller.admit(request)

            elapsed = time.time() - start

            # Should process 100 requests quickly
            assert elapsed < 5.0

            metrics = controller.get_metrics()
            assert metrics.total_requests == 100

        finally:
            controller.shutdown()

    def test_priority_queue_performance(self):
        """Test priority queue performance"""
        queue = PriorityQueue(max_size=1000)

        start = time.time()

        # Add many requests
        for i in range(500):
            request = Request(f"req_{i}", RequestPriority.NORMAL, {}, {})
            queue.put(request)

        # Remove them all
        for _ in range(500):
            queue.get(timeout=0.1)

        elapsed = time.time() - start

        # Should be fast
        assert elapsed < 5.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
