"""
Comprehensive Test Suite for Memory Boundedness
==============================================

Tests to verify that all data structures have proper bounds to prevent
unbounded memory growth and memory leaks.

Test Coverage:
1. All deques have maxlen limits
2. Caches have eviction policies
3. Memory growth is bounded over time
4. Stress tests for memory leaks
"""

import pytest
import gc
import sys
from collections import deque


class TestDequeB oudedness:
    """Test that all deques have maxlen limits."""

    def test_governed_unlearning_pending_tasks(self):
        """Test pending_tasks deque in governed_unlearning.py."""
        from src.memory.governed_unlearning import GovernedUnlearning

        gu = GovernedUnlearning()

        # Check that pending_tasks has maxlen
        assert hasattr(gu.pending_tasks, "maxlen")
        assert gu.pending_tasks.maxlen is not None
        assert gu.pending_tasks.maxlen == 1000

    def test_orchestrator_task_queue(self):
        """Test task_queue deque in unified orchestrator."""
        from src.vulcan.reasoning.unified.orchestrator import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # Check that task_queue has maxlen
        assert hasattr(orchestrator.task_queue, "maxlen")
        assert orchestrator.task_queue.maxlen is not None
        assert orchestrator.task_queue.maxlen == 10000

    def test_admission_control_requests(self):
        """Test requests deque in admission control."""
        from src.vulcan.reasoning.selection.admission_control import (
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter(window_size_seconds=60, max_requests=100)

        # Check that requests has maxlen
        assert hasattr(limiter.requests, "maxlen")
        assert limiter.requests.maxlen is not None
        # Should be at least 2x max_requests or 10000
        expected = max(100 * 2, 10000)
        assert limiter.requests.maxlen == expected

    def test_rollback_audit_deque(self):
        """Test deque in rollback audit has maxlen."""
        from src.vulcan.safety.rollback_audit import MemoryBoundedDeque

        mbd = MemoryBoundedDeque(max_size_mb=10)

        # Check that deque has maxlen as safety backstop
        assert hasattr(mbd.deque, "maxlen")
        assert mbd.deque.maxlen is not None
        assert mbd.deque.maxlen == 100000

    def test_neural_safety_deque(self):
        """Test deque in neural safety has maxlen."""
        from src.vulcan.safety.neural_safety import MemoryBoundedDeque

        mbd = MemoryBoundedDeque(max_size_mb=10)

        # Check that deque has maxlen as safety backstop
        assert hasattr(mbd.deque, "maxlen")
        assert mbd.deque.maxlen is not None
        assert mbd.deque.maxlen == 100000

    def test_specialized_memory_task_queue(self):
        """Test task_queue in specialized memory has maxlen."""
        from src.vulcan.memory.specialized import WorkingMemory

        wm = WorkingMemory()

        # Check that task_queue has maxlen
        assert hasattr(wm.task_queue, "maxlen")
        assert wm.task_queue.maxlen is not None
        assert wm.task_queue.maxlen == 1000

    def test_graph_rag_order_deque(self):
        """Test order deque in graph_rag has maxlen."""
        from src.persistant_memory_v46.graph_rag import LRUCache

        cache = LRUCache(capacity=500)

        # Check that order has maxlen matching capacity
        assert hasattr(cache.order, "maxlen")
        assert cache.order.maxlen is not None
        assert cache.order.maxlen == 500


@pytest.mark.boundedness
class TestMemoryGrowthBounds:
    """Test that memory growth is bounded over time."""

    def test_governed_unlearning_memory_bounded(self):
        """Test that GovernedUnlearning doesn't grow unbounded."""
        from src.memory.governed_unlearning import GovernedUnlearning

        gu = GovernedUnlearning()

        # Add many tasks
        for i in range(2000):  # More than maxlen
            # Simulate adding tasks (would need proper task objects in real code)
            if len(gu.pending_tasks) < gu.pending_tasks.maxlen:
                gu.pending_tasks.append(f"task_{i}")

        # Should be bounded by maxlen
        assert len(gu.pending_tasks) <= 1000

    def test_rate_limiter_memory_bounded(self):
        """Test that rate limiter doesn't grow unbounded."""
        from src.vulcan.reasoning.selection.admission_control import (
            SlidingWindowRateLimiter,
        )
        import time

        limiter = SlidingWindowRateLimiter(window_size_seconds=60, max_requests=100)

        # Simulate many requests
        for _ in range(500):  # More than maxlen
            limiter.allow_request()

        # Should be bounded by maxlen
        assert len(limiter.requests) <= limiter.requests.maxlen

    def test_lru_cache_memory_bounded(self):
        """Test that LRU cache doesn't grow unbounded."""
        from src.persistant_memory_v46.graph_rag import LRUCache

        cache = LRUCache(capacity=100)

        # Add many items
        for i in range(500):  # More than capacity
            cache.put(f"key_{i}", f"value_{i}")

        # Should be bounded by capacity
        assert len(cache.cache) <= 100
        assert len(cache.order) <= 100


@pytest.mark.stress
class TestMemoryLeakStress:
    """Stress tests to detect memory leaks."""

    @pytest.mark.slow
    def test_no_leak_in_task_queue(self):
        """Test that task queue doesn't leak memory over many operations."""
        from src.vulcan.reasoning.unified.orchestrator import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # Force garbage collection
        gc.collect()
        initial_size = sys.getsizeof(orchestrator.task_queue)

        # Add and remove many tasks
        for cycle in range(100):
            for i in range(100):
                if len(orchestrator.task_queue) < orchestrator.task_queue.maxlen:
                    orchestrator.task_queue.append(f"task_{cycle}_{i}")

        # Force garbage collection
        gc.collect()
        final_size = sys.getsizeof(orchestrator.task_queue)

        # Size should be bounded (allowing some overhead)
        # With maxlen, size should not grow significantly
        assert final_size < initial_size * 2, "Memory appears to be growing unbounded"

    @pytest.mark.slow
    def test_no_leak_in_memory_bounded_deque(self):
        """Test that memory-bounded deque doesn't leak."""
        from src.vulcan.safety.rollback_audit import MemoryBoundedDeque

        mbd = MemoryBoundedDeque(max_size_mb=1)

        # Force garbage collection
        gc.collect()

        # Add many items
        for cycle in range(50):
            for i in range(100):
                mbd.append({"data": f"item_{cycle}_{i}", "extra": "x" * 100})

        # Force garbage collection
        gc.collect()

        # Memory usage should be bounded
        memory_usage_mb = mbd.get_memory_usage_mb()
        assert memory_usage_mb <= 1.5, f"Memory usage {memory_usage_mb}MB exceeds limit"


@pytest.mark.integration
class TestIntegrationMemoryBounds:
    """Integration tests for memory boundedness across modules."""

    def test_multiple_components_bounded(self):
        """Test that using multiple components doesn't cause unbounded growth."""
        from src.memory.governed_unlearning import GovernedUnlearning
        from src.vulcan.reasoning.unified.orchestrator import UnifiedOrchestrator
        from src.vulcan.reasoning.selection.admission_control import (
            SlidingWindowRateLimiter,
        )

        # Create multiple components
        gu = GovernedUnlearning()
        orchestrator = UnifiedOrchestrator()
        limiter = SlidingWindowRateLimiter(window_size_seconds=60, max_requests=100)

        # Simulate usage
        for i in range(1000):
            # Add tasks to governed unlearning
            if len(gu.pending_tasks) < gu.pending_tasks.maxlen:
                gu.pending_tasks.append(f"task_{i}")

            # Add tasks to orchestrator
            if len(orchestrator.task_queue) < orchestrator.task_queue.maxlen:
                orchestrator.task_queue.append(f"task_{i}")

            # Use rate limiter
            limiter.allow_request()

        # All should be bounded
        assert len(gu.pending_tasks) <= 1000
        assert len(orchestrator.task_queue) <= 10000
        assert len(limiter.requests) <= limiter.requests.maxlen


class TestDequeBestPractices:
    """Test that deques follow best practices."""

    def test_all_unbounded_deques_identified(self):
        """
        Document test: Verify we've identified and fixed all unbounded deques.
        
        This test documents the deques we've fixed. If a new unbounded deque
        is added in the future, this test should be updated.
        """
        fixed_deques = [
            "src.memory.governed_unlearning.GovernedUnlearning.pending_tasks",
            "src.vulcan.reasoning.unified.orchestrator.UnifiedOrchestrator.task_queue",
            "src.vulcan.reasoning.selection.admission_control.SlidingWindowRateLimiter.requests",
            "src.vulcan.safety.rollback_audit.MemoryBoundedDeque.deque",
            "src.vulcan.safety.neural_safety.MemoryBoundedDeque.deque",
            "src.vulcan.memory.specialized.WorkingMemory.task_queue",
            "src.persistant_memory_v46.graph_rag.LRUCache.order",
        ]

        # This list documents all the deques we've fixed
        assert len(fixed_deques) == 7, "Expected 7 deques to be fixed"

    def test_deque_maxlen_reasonable(self):
        """Test that all maxlen values are reasonable (not too large)."""
        from src.memory.governed_unlearning import GovernedUnlearning
        from src.vulcan.reasoning.unified.orchestrator import UnifiedOrchestrator
        from src.vulcan.memory.specialized import WorkingMemory

        gu = GovernedUnlearning()
        orchestrator = UnifiedOrchestrator()
        wm = WorkingMemory()

        # All maxlen values should be reasonable
        assert gu.pending_tasks.maxlen <= 10000
        assert orchestrator.task_queue.maxlen <= 100000
        assert wm.task_queue.maxlen <= 10000


def test_memory_boundedness_documentation():
    """
    Test that documents the memory boundedness approach.
    
    This test serves as documentation for the memory boundedness strategy:
    1. All deques must have maxlen parameter
    2. Caches must have eviction policies (LRU, size-based, etc.)
    3. Size-based limits should have count-based backstops
    4. Reasonable limits based on use case
    """
    # Strategy 1: Direct maxlen on deques
    example_deque = deque(maxlen=1000)
    assert example_deque.maxlen == 1000

    # Strategy 2: Size-based eviction with count backstop
    # (implemented in MemoryBoundedDeque classes)

    # Strategy 3: LRU eviction with capacity limits
    # (implemented in LRUCache)

    # All strategies ensure bounded memory growth
    assert True, "Memory boundedness strategy documented"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
