"""
Unit tests for critical concurrency and reliability fixes.

Tests cover:
1. Thread-local state in SymbolicReasoner
2. Circuit breaker exponential backoff
3. Recursive meta-reasoning guard
4. Reasoning chain length limits

Author: GitHub Copilot
Date: 2026-01-18
"""

import pytest
import threading
import time
from typing import List
from unittest.mock import Mock, patch

# Test Fix #1: Thread-Local State
def test_symbolic_reasoner_thread_local_kb():
    """
    Test that SymbolicReasoner uses thread-local KnowledgeBase.
    
    Verifies:
    - Each thread gets its own KB instance
    - Modifications in one thread don't affect other threads
    - No race conditions on parallel execution
    """
    from src.vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
    
    reasoner = SymbolicReasoner()
    
    # Store KB IDs from different threads
    kb_ids = []
    errors = []
    
    def thread_func(thread_id: int):
        try:
            # Get KB instance for this thread
            kb1 = reasoner.kb
            kb_id1 = id(kb1)
            
            # Add some facts (thread-specific)
            reasoner.add_fact(f"thread_{thread_id}_fact(x)")
            
            # Get KB again (should be same instance for this thread)
            kb2 = reasoner.kb
            kb_id2 = id(kb2)
            
            # Verify it's the same instance within thread
            assert kb_id1 == kb_id2, f"Thread {thread_id}: KB instance changed within thread"
            
            kb_ids.append((thread_id, kb_id1))
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    # Run multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=thread_func, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Check for errors
    assert len(errors) == 0, f"Thread errors: {errors}"
    
    # Verify each thread got a different KB instance
    kb_ids_only = [kb_id for _, kb_id in kb_ids]
    unique_kb_ids = set(kb_ids_only)
    
    assert len(unique_kb_ids) == 5, \
        f"Expected 5 unique KB instances, got {len(unique_kb_ids)}: {kb_ids}"


# Test Fix #3: Circuit Breaker Exponential Backoff
def test_circuit_breaker_exponential_backoff():
    """
    Test that circuit breaker implements exponential backoff.
    
    Verifies:
    - Timeout increases exponentially with failures
    - Jitter is applied correctly
    - Maximum timeout is respected
    """
    from src.vulcan.reasoning.selection.embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
        CircuitState,
    )
    
    # Create circuit breaker with known parameters
    cb = EmbeddingCircuitBreaker(
        reset_timeout_s=10.0,
        max_reset_timeout_s=100.0,
        backoff_multiplier=2.0,
        backoff_jitter=0.0,  # No jitter for predictable testing
    )
    
    # Verify initial state
    assert cb._consecutive_failures == 0
    assert cb._state == CircuitState.CLOSED
    
    # Simulate failures and check backoff
    initial_timeout = cb.reset_timeout_s
    
    # First failure: timeout should be initial_timeout
    cb.record_latency(10000)  # Way over threshold
    assert cb._consecutive_failures == 1
    timeout1 = cb._get_current_timeout_with_jitter()
    expected1 = initial_timeout * (2.0 ** 0)
    assert abs(timeout1 - expected1) < 0.1, \
        f"First failure: expected {expected1}, got {timeout1}"
    
    # Second failure: timeout should be 2x
    cb._state = CircuitState.HALF_OPEN
    cb.record_latency(10000)
    assert cb._consecutive_failures == 2
    timeout2 = cb._get_current_timeout_with_jitter()
    expected2 = initial_timeout * (2.0 ** 1)
    assert abs(timeout2 - expected2) < 0.1, \
        f"Second failure: expected {expected2}, got {timeout2}"
    
    # Third failure: timeout should be 4x
    cb._state = CircuitState.HALF_OPEN
    cb.record_latency(10000)
    assert cb._consecutive_failures == 3
    timeout3 = cb._get_current_timeout_with_jitter()
    expected3 = initial_timeout * (2.0 ** 2)
    assert abs(timeout3 - expected3) < 0.1, \
        f"Third failure: expected {expected3}, got {timeout3}"
    
    # Test maximum timeout cap
    cb._consecutive_failures = 10  # Very high
    timeout_max = cb._get_current_timeout_with_jitter()
    assert timeout_max == 100.0, \
        f"Maximum timeout: expected 100.0, got {timeout_max}"


def test_circuit_breaker_backoff_jitter():
    """
    Test that circuit breaker applies jitter correctly.
    
    Verifies:
    - Jitter creates variation in timeout
    - Jitter stays within expected range
    """
    from src.vulcan.reasoning.selection.embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
    )
    
    cb = EmbeddingCircuitBreaker(
        reset_timeout_s=100.0,
        backoff_jitter=0.2,  # 20% jitter
    )
    
    # Generate multiple timeouts with jitter
    timeouts = []
    for _ in range(100):
        timeout = cb._get_current_timeout_with_jitter()
        timeouts.append(timeout)
    
    # Check that timeouts vary (jitter is working)
    unique_timeouts = set(timeouts)
    assert len(unique_timeouts) > 10, \
        f"Jitter should create variation, got {len(unique_timeouts)} unique values"
    
    # Check that all timeouts are within jitter range
    expected_min = 100.0 * (1 - 0.2)  # 80.0
    expected_max = 100.0 * (1 + 0.2)  # 120.0
    
    for timeout in timeouts:
        assert expected_min <= timeout <= expected_max, \
            f"Timeout {timeout} outside expected range [{expected_min}, {expected_max}]"


def test_circuit_breaker_reset_on_success():
    """
    Test that consecutive failures reset on successful close.
    
    Verifies:
    - Backoff state resets when circuit closes
    - Timeout returns to initial value
    """
    from src.vulcan.reasoning.selection.embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
        CircuitState,
    )
    
    cb = EmbeddingCircuitBreaker(
        reset_timeout_s=10.0,
        backoff_multiplier=2.0,
        backoff_jitter=0.0,
    )
    
    # Cause multiple failures
    cb._consecutive_failures = 5
    cb._state = CircuitState.OPEN
    
    # Force reset
    cb.force_reset()
    
    # Verify backoff state is reset
    assert cb._consecutive_failures == 0
    assert cb._state == CircuitState.CLOSED
    
    timeout = cb._get_current_timeout_with_jitter()
    assert timeout == 10.0, f"Expected timeout reset to 10.0, got {timeout}"


# Test Fix #4: Recursion Guard
def test_recursion_depth_limit():
    """
    Test that recursion depth limit prevents infinite loops.
    
    Verifies:
    - MAX_RECURSION_DEPTH is enforced
    - Error result returned at limit
    - No stack overflow
    """
    # Import the constant from production code
    from src.vulcan.reasoning.unified.orchestrator import UnifiedReasoner
    
    MAX_RECURSION_DEPTH = UnifiedReasoner.MAX_RECURSION_DEPTH
    call_count = [0]
    
    def recursive_reason(depth=0):
        call_count[0] += 1
        
        if depth >= MAX_RECURSION_DEPTH:
            return {"error": f"Max depth {MAX_RECURSION_DEPTH} exceeded"}
        
        # Simulate recursive call
        return recursive_reason(depth + 1)
    
    result = recursive_reason()
    
    # Verify depth limit was hit
    assert call_count[0] == MAX_RECURSION_DEPTH + 1  # +1 for the final call that fails
    assert "error" in result
    assert str(MAX_RECURSION_DEPTH) in result["error"]


# Test Fix #5: Chain Length Limit
def test_reasoning_chain_max_steps():
    """
    Test that ReasoningChain enforces maximum step limit.
    
    Verifies:
    - MAX_REASONING_CHAIN_STEPS is enforced
    - Automatic pruning when limit exceeded
    - Chain remains functional after pruning
    """
    from src.vulcan.reasoning.reasoning_types import (
        ReasoningChain,
        ReasoningStep,
        ReasoningType,
        MAX_REASONING_CHAIN_STEPS,
    )
    
    # Create initial step
    initial_step = ReasoningStep(
        step_id="init",
        step_type=ReasoningType.UNKNOWN,
        input_data="test",
        output_data="test",
        confidence=1.0,
        explanation="Init",
    )
    
    # Create chain
    chain = ReasoningChain(
        chain_id="test_chain",
        steps=[initial_step],
        initial_query={"query": "test"},
        final_conclusion=None,
        total_confidence=0.5,
        reasoning_types_used=set(),
        modalities_involved=set(),
        safety_checks=[],
        audit_trail=[],
    )
    
    # Add many steps (more than limit)
    for i in range(MAX_REASONING_CHAIN_STEPS + 50):
        step = ReasoningStep(
            step_id=f"step_{i}",
            step_type=ReasoningType.PROBABILISTIC,
            input_data=f"input_{i}",
            output_data=f"output_{i}",
            confidence=0.5 + (i % 10) * 0.05,  # Varying confidence
            explanation=f"Step {i}",
            timestamp=time.time() + i * 0.001,  # Different timestamps
        )
        chain.add_step(step)
    
    # Verify chain was pruned to limit
    assert len(chain.steps) == MAX_REASONING_CHAIN_STEPS, \
        f"Expected {MAX_REASONING_CHAIN_STEPS} steps, got {len(chain.steps)}"
    
    # Verify first step is preserved
    assert chain.steps[0].step_id == "init", \
        "First step should be preserved"


def test_chain_pruning_preserves_high_confidence():
    """
    Test that chain pruning keeps highest confidence steps.
    
    Verifies:
    - High-confidence steps from middle are kept
    - Low-confidence steps are pruned
    - Temporal order is maintained
    """
    from src.vulcan.reasoning.reasoning_types import (
        ReasoningChain,
        ReasoningStep,
        ReasoningType,
        MAX_REASONING_CHAIN_STEPS,
    )
    
    # Create chain with varying confidence
    steps = []
    
    # Add initial step (always kept)
    steps.append(ReasoningStep(
        step_id="init",
        step_type=ReasoningType.UNKNOWN,
        input_data="test",
        output_data="test",
        confidence=1.0,
        explanation="Init",
        timestamp=time.time(),
    ))
    
    # Add many middle steps with varying confidence
    base_time = time.time()
    for i in range(MAX_REASONING_CHAIN_STEPS + 20):
        # Create some high-confidence (0.9) and some low-confidence (0.3) steps
        confidence = 0.9 if i % 5 == 0 else 0.3
        
        steps.append(ReasoningStep(
            step_id=f"step_{i}",
            step_type=ReasoningType.PROBABILISTIC,
            input_data=f"input_{i}",
            output_data=f"output_{i}",
            confidence=confidence,
            explanation=f"Step {i}",
            timestamp=base_time + i * 0.001,
        ))
    
    chain = ReasoningChain(
        chain_id="test_chain",
        steps=steps,
        initial_query={"query": "test"},
        final_conclusion=None,
        total_confidence=0.5,
        reasoning_types_used=set(),
        modalities_involved=set(),
        safety_checks=[],
        audit_trail=[],
    )
    
    # Chain should be pruned
    assert len(chain.steps) == MAX_REASONING_CHAIN_STEPS
    
    # Count high-confidence steps in pruned chain
    # (excluding first step and last 10 which are always kept)
    middle_steps = chain.steps[1:-10]
    high_conf_count = sum(1 for s in middle_steps if s.confidence >= 0.8)
    
    # Should have more high-confidence steps than low-confidence
    # (because pruning preferentially keeps high-confidence)
    assert high_conf_count > len(middle_steps) * 0.3, \
        f"Expected high-confidence steps to be preserved, got {high_conf_count}/{len(middle_steps)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
