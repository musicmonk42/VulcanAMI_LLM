"""
Tests for _consume_async_result timeout handling in GraphixVulcanLLM.

These tests verify that the per-token timeout protection works correctly
to prevent the 60-second hang issue when the first token never arrives.

NOTE: These tests use a mock implementation of _consume_async_result rather than
importing from graphix_vulcan_llm.py because:
1. The actual implementation has complex dependencies (CognitiveLoop, bridge, transformer)
2. We want to test the timeout logic in isolation without those dependencies
3. The mock implementation mirrors the critical timeout logic that we're testing
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, List, Dict
from unittest.mock import MagicMock

import pytest


# Custom exception for first-token timeout (mirrors the one in graphix_vulcan_llm.py)
# This is intentionally duplicated here because importing from graphix_vulcan_llm
# would bring in many heavy dependencies that we don't need for these unit tests.
class FirstTokenTimeoutError(Exception):
    """Raised when the first token never arrives within the timeout period."""
    pass


@dataclass
class MockCognitiveLoopResult:
    """Mock result from CognitiveLoop."""
    tokens: List[Any]
    text: str
    reasoning_trace: List[Dict[str, Any]]
    safety_events: List[Dict[str, Any]]
    audit_records: List[Dict[str, Any]]
    beam_metadata: Any = None
    speculative_stats: Any = None
    metrics: Dict[str, Any] = None
    completed: bool = True
    stopped_reason: str = "max_tokens_reached"
    duration_seconds: float = 0.1


async def mock_fast_generator() -> AsyncGenerator[Dict[str, Any], None]:
    """A mock async generator that produces items quickly."""
    for i in range(3):
        yield {"token": f"token_{i}", "info": {}}
        await asyncio.sleep(0.01)  # Small delay
    # Final item is the complete result
    yield MockCognitiveLoopResult(
        tokens=["token_0", "token_1", "token_2"],
        text="token_0 token_1 token_2",
        reasoning_trace=[],
        safety_events=[],
        audit_records=[],
        metrics={},
    )


async def mock_slow_first_token_generator() -> AsyncGenerator[Dict[str, Any], None]:
    """A mock async generator that hangs on the first token (simulates the bug)."""
    # This simulates the hanging behavior - first token never arrives
    await asyncio.sleep(100)  # Hang for 100 seconds (will be interrupted by timeout)
    yield {"token": "should_not_reach", "info": {}}


async def mock_slow_subsequent_token_generator() -> AsyncGenerator[Dict[str, Any], None]:
    """A mock async generator that produces first token quickly but hangs on second."""
    yield {"token": "first_token", "info": {}}
    # Hang on subsequent token
    await asyncio.sleep(100)
    yield {"token": "should_not_reach", "info": {}}


class MockGraphixVulcanLLM:
    """Minimal mock of GraphixVulcanLLM with just the _consume_async_result method.
    
    This mock implements the core timeout logic from the real GraphixVulcanLLM class,
    allowing us to test the per-token timeout behavior in isolation without the
    heavy dependencies required by the full implementation (CognitiveLoop, transformer,
    bridge, etc.).
    
    The implementation here should be kept in sync with the actual _consume_async_result
    method in graphix_vulcan_llm.py to ensure the tests accurately reflect the real behavior.
    """
    
    _MAX_GENERATOR_ITEMS = 10000
    _FIRST_TOKEN_TIMEOUT_SECONDS = 2.0  # Short timeout for testing
    
    def __init__(self):
        self.logger = MagicMock()
        
    async def _consume_async_result(self, gen_result, timeout: float = 60.0) -> Any:
        """Copy of the actual implementation for testing."""
        import inspect
        
        async def _consume():
            result = gen_result
            
            # If it's a coroutine, await it first
            if inspect.iscoroutine(result):
                result = await result
            
            # Now check if we have an async generator (streaming mode)
            if hasattr(result, "__anext__"):
                items = []
                count = 0
                
                first_token_timeout = self._FIRST_TOKEN_TIMEOUT_SECONDS
                subsequent_token_timeout = 1.0  # Short for testing
                
                async_iter = result.__aiter__()
                while True:
                    try:
                        token_timeout = first_token_timeout if count == 0 else subsequent_token_timeout
                        
                        item = await asyncio.wait_for(
                            async_iter.__anext__(),
                            timeout=token_timeout
                        )
                        
                        count += 1
                        items.append(item)
                        
                        if count > self._MAX_GENERATOR_ITEMS:
                            break
                            
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        if count == 0:
                            raise FirstTokenTimeoutError(
                                f"First token never arrived after {first_token_timeout}s - "
                                "async generator blocked on first await"
                            ) from None
                        else:
                            break
                
                if not items:
                    raise ValueError("Generator yielded no items!")
                
                return items[-1]
            
            # If result already has tokens, it's the final result
            if hasattr(result, "tokens"):
                return result
            
            raise TypeError(f"Unknown result type: {type(result).__name__}")
        
        try:
            if timeout:
                try:
                    return await asyncio.wait_for(_consume(), timeout=timeout)
                except FirstTokenTimeoutError:
                    raise
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Generation timed out after {timeout} seconds") from None
            else:
                return await _consume()
        except FirstTokenTimeoutError as e:
            # Convert to standard TimeoutError for API compatibility
            raise TimeoutError(str(e)) from e


@pytest.mark.asyncio
async def test_consume_fast_generator():
    """Test that fast generators complete successfully."""
    llm = MockGraphixVulcanLLM()
    
    result = await llm._consume_async_result(mock_fast_generator(), timeout=10.0)
    
    assert result is not None
    assert hasattr(result, 'tokens')
    assert result.tokens == ["token_0", "token_1", "token_2"]


@pytest.mark.asyncio
async def test_consume_slow_first_token_raises_timeout():
    """Test that slow first token raises TimeoutError quickly (not after 60s).
    
    This test verifies that when the first token never arrives, we get a quick
    timeout error instead of hanging for the full 60-second overall timeout.
    """
    llm = MockGraphixVulcanLLM()
    llm._FIRST_TOKEN_TIMEOUT_SECONDS = 1.0  # 1 second timeout for first token
    
    start = time.time()
    
    with pytest.raises(TimeoutError) as exc_info:
        # Use a much longer overall timeout (30s) so only the per-token timeout fires
        await llm._consume_async_result(mock_slow_first_token_generator(), timeout=30.0)
    
    elapsed = time.time() - start
    # The test should complete in about 1 second (first token timeout), not 30 seconds
    # Allow some buffer for async overhead
    assert elapsed < 5.0, f"Test took {elapsed:.1f}s, should have timed out in ~1s"
    # Verify the error message mentions first token
    assert "First token never arrived" in str(exc_info.value)


@pytest.mark.asyncio
async def test_consume_slow_subsequent_token_returns_partial():
    """Test that slow subsequent tokens return partial results instead of hanging."""
    llm = MockGraphixVulcanLLM()
    llm._FIRST_TOKEN_TIMEOUT_SECONDS = 5.0  # Long enough for first token
    
    # This should complete quickly, returning the first token that was successfully received
    result = await llm._consume_async_result(
        mock_slow_subsequent_token_generator(), 
        timeout=10.0
    )
    
    # Should return the first item since it was received
    assert result is not None
    assert result["token"] == "first_token"


@pytest.mark.asyncio 
async def test_consume_direct_result():
    """Test that direct results (non-generator) are handled correctly."""
    llm = MockGraphixVulcanLLM()
    
    direct_result = MockCognitiveLoopResult(
        tokens=["a", "b", "c"],
        text="a b c",
        reasoning_trace=[],
        safety_events=[],
        audit_records=[],
        metrics={},
    )
    
    result = await llm._consume_async_result(direct_result, timeout=10.0)
    
    assert result is direct_result
    assert result.tokens == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_consume_coroutine_returning_result():
    """Test that coroutines returning direct results are handled correctly."""
    llm = MockGraphixVulcanLLM()
    
    async def coro_returning_result():
        return MockCognitiveLoopResult(
            tokens=["x", "y"],
            text="x y",
            reasoning_trace=[],
            safety_events=[],
            audit_records=[],
            metrics={},
        )
    
    result = await llm._consume_async_result(coro_returning_result(), timeout=10.0)
    
    assert result is not None
    assert result.tokens == ["x", "y"]


@pytest.mark.asyncio
async def test_consume_coroutine_returning_generator():
    """Test that coroutines returning async generators are handled correctly."""
    llm = MockGraphixVulcanLLM()
    
    async def coro_returning_generator():
        return mock_fast_generator()
    
    result = await llm._consume_async_result(coro_returning_generator(), timeout=10.0)
    
    assert result is not None
    assert hasattr(result, 'tokens')


@pytest.mark.asyncio
async def test_overall_timeout_respected():
    """Test that the overall timeout is respected even if per-token timeouts are longer."""
    llm = MockGraphixVulcanLLM()
    llm._FIRST_TOKEN_TIMEOUT_SECONDS = 100.0  # Very long first token timeout
    
    start = time.time()
    
    with pytest.raises(TimeoutError) as exc_info:
        # Overall timeout of 0.5s should trigger before the 100s first token timeout
        await llm._consume_async_result(mock_slow_first_token_generator(), timeout=0.5)
    
    elapsed = time.time() - start
    
    # Should be the overall timeout message, not the first token message
    assert "timed out after 0.5 seconds" in str(exc_info.value)
    # Should have completed quickly (around 0.5s, not 100s)
    assert elapsed < 2.0, f"Test took {elapsed:.1f}s, should have timed out in ~0.5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

