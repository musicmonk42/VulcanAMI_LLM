"""
Test suite for VulcanReasoningOutput and hybrid output formatting.

These tests verify:
1. VulcanReasoningOutput dataclass functionality
2. HybridLLMExecutor structured output handling
3. OpenAI formatting integration (with mocking)
4. Fallback formatting when OpenAI is unavailable
5. Error handling for reasoning failures
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vulcan.llm.hybrid_executor import (
    HybridLLMExecutor,
    VulcanReasoningOutput,
    VULCAN_HARD_TIMEOUT,
    PER_TOKEN_TIMEOUT,
)


class TestVulcanReasoningOutput:
    """Test VulcanReasoningOutput dataclass."""

    def test_create_successful_output(self):
        """Test creating a successful reasoning output."""
        output = VulcanReasoningOutput(
            query_id="test-001",
            success=True,
            result=42,
            result_type="mathematical",
            method_used="symbolic_computation",
            confidence=0.95,
            reasoning_trace=["Step 1: Parse", "Step 2: Compute"],
        )
        
        assert output.success is True
        assert output.result == 42
        assert output.result_type == "mathematical"
        assert output.confidence == 0.95
        assert len(output.reasoning_trace) == 2
        assert output.error is None
        assert output.is_valid() is True

    def test_create_failed_output(self):
        """Test creating a failed reasoning output."""
        output = VulcanReasoningOutput(
            query_id="test-002",
            success=False,
            result=None,
            result_type="unknown",
            method_used="symbolic_computation",
            confidence=0.0,
            error="Timeout during computation",
        )
        
        assert output.success is False
        assert output.result is None
        assert output.error == "Timeout during computation"
        assert output.is_valid() is False

    def test_to_dict(self):
        """Test converting output to dictionary."""
        output = VulcanReasoningOutput(
            query_id="test-003",
            success=True,
            result={"answer": "pi/3"},
            result_type="mathematical",
            method_used="calculus",
            confidence=0.9,
            metadata={"query": "integrate sin(x)"},
        )
        
        d = output.to_dict()
        
        assert isinstance(d, dict)
        assert d["query_id"] == "test-003"
        assert d["success"] is True
        assert d["result"] == {"answer": "pi/3"}
        assert d["metadata"]["query"] == "integrate sin(x)"

    def test_repr(self):
        """Test string representation."""
        output_success = VulcanReasoningOutput(
            query_id="test-004",
            success=True,
            result=100,
            result_type="factual",
            method_used="lookup",
            confidence=0.99,
        )
        
        output_fail = VulcanReasoningOutput(
            query_id="test-005",
            success=False,
            result=None,
            result_type="unknown",
            method_used="unknown",
            confidence=0.0,
        )
        
        # Success should show ✓
        assert "✓" in repr(output_success)
        assert "test-004" in repr(output_success)
        
        # Failure should show ✗
        assert "✗" in repr(output_fail)

    def test_default_values(self):
        """Test default values are set correctly."""
        output = VulcanReasoningOutput(
            query_id="test-006",
            success=True,
            result="some result",
        )
        
        assert output.result_type == "unknown"
        assert output.method_used == "unknown"
        assert output.confidence == 0.0
        assert output.reasoning_trace == []
        assert output.error is None
        assert output.metadata == {}


class TestTimeoutConfiguration:
    """Test timeout configuration constants."""

    def test_hard_timeout_value(self):
        """Test VULCAN_HARD_TIMEOUT is set to a reasonable value."""
        # Should be at least 60 seconds for CPU-intensive reasoning
        assert VULCAN_HARD_TIMEOUT >= 60.0
        # Default should be 120 seconds
        assert VULCAN_HARD_TIMEOUT == 120.0

    def test_per_token_timeout_value(self):
        """Test PER_TOKEN_TIMEOUT is set correctly."""
        # Should be at least 10 seconds per token for CPU execution
        assert PER_TOKEN_TIMEOUT >= 10.0
        # Default should be 30 seconds
        assert PER_TOKEN_TIMEOUT == 30.0


class TestHybridExecutorStructuredOutput:
    """Test HybridLLMExecutor with structured output."""

    @pytest.fixture
    def mock_local_llm(self):
        """Create a mock local LLM."""
        mock = MagicMock()
        mock.generate.return_value = MagicMock(text="Mock LLM response")
        return mock

    @pytest.fixture
    def executor(self, mock_local_llm):
        """Create HybridLLMExecutor with mocks."""
        # Mock the OpenAI client getter
        mock_openai_getter = MagicMock(return_value=None)
        
        executor = HybridLLMExecutor(
            local_llm=mock_local_llm,
            openai_client_getter=mock_openai_getter,
            mode="parallel",
            timeout=60.0,
        )
        return executor

    @pytest.mark.asyncio
    async def test_execute_with_structured_output_success(self, executor):
        """Test execution with successful structured output."""
        reasoning_output = VulcanReasoningOutput(
            query_id="test-exec-001",
            success=True,
            result="x**3/3",
            result_type="mathematical",
            method_used="symbolic_integration",
            confidence=0.95,
            reasoning_trace=["Parsed integral", "Applied power rule"],
        )
        
        result = await executor.execute_with_structured_output(
            prompt="What is the integral of x^2?",
            reasoning_output=reasoning_output,
            use_openai_formatting=False,  # Use internal formatting
        )
        
        assert result is not None
        assert "text" in result
        assert "source" in result
        assert "vulcan" in result["source"].lower()
        assert result["metadata"]["reasoning_output"]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_structured_output_error(self, executor):
        """Test execution with failed structured output."""
        reasoning_output = VulcanReasoningOutput(
            query_id="test-exec-002",
            success=False,
            result=None,
            result_type="unknown",
            method_used="symbolic_computation",
            confidence=0.0,
            error="Timeout during computation",
        )
        
        result = await executor.execute_with_structured_output(
            prompt="Complex query that times out",
            reasoning_output=reasoning_output,
        )
        
        assert result is not None
        assert result.get("error") is True
        assert "vulcan_reasoning_error" in result["source"]
        assert "timeout" in result["text"].lower() or "issue" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_fallback_to_legacy(self, executor):
        """Test that missing structured output falls back to legacy execution."""
        # Mock the legacy execute method
        executor.execute = AsyncMock(return_value={
            "text": "Legacy response",
            "source": "local",
            "systems_used": ["vulcan_internal"],
        })
        
        result = await executor.execute_with_structured_output(
            prompt="Test query",
            reasoning_output=None,  # No structured output
        )
        
        # Should have called legacy execute
        executor.execute.assert_called_once()
        assert result["text"] == "Legacy response"

    @pytest.mark.asyncio
    async def test_execute_with_context_containing_output(self, executor):
        """Test that reasoning_output can be passed via context dict."""
        reasoning_output = VulcanReasoningOutput(
            query_id="test-exec-003",
            success=True,
            result={"answer": 42},
            result_type="factual",
            method_used="lookup",
            confidence=1.0,
        )
        
        result = await executor.execute_with_structured_output(
            prompt="What is the answer?",
            context={"reasoning_output": reasoning_output},
            use_openai_formatting=False,
        )
        
        assert result is not None
        assert result["metadata"]["reasoning_output"]["result"]["answer"] == 42


class TestInternalFormatting:
    """Test internal formatting without OpenAI."""

    @pytest.fixture
    def executor(self):
        """Create HybridLLMExecutor with mocks."""
        executor = HybridLLMExecutor(
            local_llm=MagicMock(),
            openai_client_getter=MagicMock(return_value=None),
            mode="parallel",
        )
        return executor

    def test_format_mathematical_result(self, executor):
        """Test formatting mathematical results."""
        output = VulcanReasoningOutput(
            query_id="fmt-001",
            success=True,
            result="x**3/3 + C",
            result_type="mathematical",
            method_used="integration",
            confidence=0.95,
        )
        
        formatted = executor._format_structured_output_sync(output)
        
        assert "x**3/3" in formatted
        assert "answer" in formatted.lower() or "result" in formatted.lower()

    def test_format_symbolic_result(self, executor):
        """Test formatting symbolic results."""
        output = VulcanReasoningOutput(
            query_id="fmt-002",
            success=True,
            result="The proposition is valid",
            result_type="symbolic",
            method_used="logical_reasoning",
            confidence=0.9,
        )
        
        formatted = executor._format_structured_output_sync(output)
        
        assert "symbolic" in formatted.lower() or "proposition" in formatted.lower()

    def test_format_error_result(self, executor):
        """Test formatting error results."""
        output = VulcanReasoningOutput(
            query_id="fmt-003",
            success=False,
            result=None,
            result_type="unknown",
            method_used="unknown",
            confidence=0.0,
            error="Division by zero",
        )
        
        formatted = executor._format_structured_output_sync(output)
        
        assert "issue" in formatted.lower() or "error" in formatted.lower()

    def test_format_dict_result(self, executor):
        """Test formatting dict results."""
        output = VulcanReasoningOutput(
            query_id="fmt-004",
            success=True,
            result={"x": 1, "y": 2, "z": 3},
            result_type="factual",
            method_used="computation",
            confidence=0.8,
        )
        
        formatted = executor._format_structured_output_sync(output)
        
        # Should contain the dict data in some form
        assert "x" in formatted or "1" in formatted


class TestOpenAIFormatting:
    """Test OpenAI formatting integration."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Formatted by OpenAI: The answer is 42."))]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def executor_with_openai(self, mock_openai_client):
        """Create executor with mocked OpenAI."""
        executor = HybridLLMExecutor(
            local_llm=MagicMock(),
            openai_client_getter=lambda: mock_openai_client,
            mode="parallel",
        )
        return executor

    @pytest.mark.asyncio
    async def test_format_with_openai_success(self, executor_with_openai):
        """Test successful OpenAI formatting."""
        # Mock the _call_openai method to return a formatted string
        executor_with_openai._call_openai = AsyncMock(
            return_value="The integral of x² is **x³/3 + C**."
        )
        
        reasoning_output = VulcanReasoningOutput(
            query_id="openai-fmt-001",
            success=True,
            result="x**3/3 + C",
            result_type="mathematical",
            method_used="integration",
            confidence=0.95,
        )
        
        loop = asyncio.get_event_loop()
        formatted = await executor_with_openai._format_with_openai(
            reasoning_output, "What is the integral of x^2?", loop
        )
        
        assert formatted is not None
        assert "x³/3" in formatted or "x**3/3" in formatted

    @pytest.mark.asyncio
    async def test_format_with_openai_failure_fallback(self, executor_with_openai):
        """Test fallback when OpenAI formatting fails."""
        # Mock OpenAI to fail
        executor_with_openai._call_openai = AsyncMock(return_value=None)
        
        reasoning_output = VulcanReasoningOutput(
            query_id="openai-fmt-002",
            success=True,
            result=42,
            result_type="mathematical",
            method_used="computation",
            confidence=0.99,
        )
        
        loop = asyncio.get_event_loop()
        formatted = await executor_with_openai._format_with_openai(
            reasoning_output, "What is 6*7?", loop
        )
        
        # Should return None when OpenAI fails
        assert formatted is None


class TestErrorFormatting:
    """Test error message formatting."""

    @pytest.fixture
    def executor(self):
        """Create executor for error formatting tests."""
        return HybridLLMExecutor(
            local_llm=MagicMock(),
            openai_client_getter=MagicMock(return_value=None),
            mode="parallel",
        )

    def test_format_timeout_error(self, executor):
        """Test formatting timeout errors."""
        output = VulcanReasoningOutput(
            query_id="err-001",
            success=False,
            result=None,
            result_type="unknown",
            method_used="unknown",
            confidence=0.0,
            error="Timeout exceeded after 30 seconds",
        )
        
        error_text = executor._format_reasoning_error(output)
        
        assert "timeout" in error_text.lower() or "longer than expected" in error_text.lower()
        assert "suggestions" in error_text.lower()

    def test_format_memory_error(self, executor):
        """Test formatting memory errors."""
        output = VulcanReasoningOutput(
            query_id="err-002",
            success=False,
            result=None,
            result_type="unknown",
            method_used="unknown",
            confidence=0.0,
            error="Out of memory during computation",
        )
        
        error_text = executor._format_reasoning_error(output)
        
        assert "resource" in error_text.lower() or "memory" in error_text.lower()

    def test_format_generic_error(self, executor):
        """Test formatting generic errors."""
        output = VulcanReasoningOutput(
            query_id="err-003",
            success=False,
            result=None,
            result_type="unknown",
            method_used="unknown",
            confidence=0.0,
            error="Unknown error occurred",
        )
        
        error_text = executor._format_reasoning_error(output)
        
        assert "issue" in error_text.lower() or "error" in error_text.lower()
        # Should include error reference
        assert any(c.isupper() and c.isalnum() for c in error_text)

    def test_format_error_no_message(self, executor):
        """Test formatting errors with no error message."""
        output = VulcanReasoningOutput(
            query_id="err-004",
            success=False,
            result=None,
            result_type="unknown",
            method_used="unknown",
            confidence=0.0,
            error=None,  # No error message
        )
        
        error_text = executor._format_reasoning_error(output)
        
        # Should still produce helpful text
        assert "issue" in error_text.lower() or "could not" in error_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
