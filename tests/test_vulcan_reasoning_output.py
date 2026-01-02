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
        # Should include error reference code (12 uppercase hex characters like "A1B2C3D4E5F6")
        # The error reference helps users report issues for support investigation
        import re
        assert re.search(r'\*\*[A-F0-9]{12}\*\*', error_text), \
            f"Error reference not found in: {error_text}"

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


class TestFormatOutputForUser:
    """Test the new format_output_for_user method for OpenAI language formatting."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Based on VULCAN's analysis, the answer is 42."))
        ]
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

    @pytest.fixture
    def executor_without_openai(self):
        """Create executor without OpenAI access."""
        executor = HybridLLMExecutor(
            local_llm=MagicMock(),
            openai_client_getter=lambda: None,  # No OpenAI
            mode="parallel",
        )
        return executor

    @pytest.mark.asyncio
    async def test_format_output_with_dict_input(self, executor_with_openai):
        """Test format_output_for_user with dict input."""
        # Mock the OpenAI formatting call
        executor_with_openai._call_openai_formatting = AsyncMock(
            return_value="VULCAN calculated the result: **42**"
        )
        
        reasoning_output = {
            "success": True,
            "result": 42,
            "method": "calculation",
            "confidence": 0.95,
            "reasoning_trace": ["Step 1", "Step 2"],
        }
        
        result = await executor_with_openai.format_output_for_user(
            reasoning_output=reasoning_output,
            original_prompt="What is 6 times 7?",
        )
        
        assert result is not None
        assert "text" in result
        assert result["source"] == "openai_formatting"
        assert "openai_formatting" in result["systems_used"]
        assert result["metadata"]["openai_role"] == "formatting_only"

    @pytest.mark.asyncio
    async def test_format_output_with_vulcan_reasoning_output(self, executor_with_openai):
        """Test format_output_for_user with VulcanReasoningOutput input."""
        executor_with_openai._call_openai_formatting = AsyncMock(
            return_value="The integral of x² is x³/3 + C."
        )
        
        reasoning_output = VulcanReasoningOutput(
            query_id="test-001",
            success=True,
            result="x**3/3 + C",
            result_type="mathematical",
            method_used="symbolic_integration",
            confidence=0.95,
        )
        
        result = await executor_with_openai.format_output_for_user(
            reasoning_output=reasoning_output,
            original_prompt="What is the integral of x^2?",
        )
        
        assert result["source"] == "openai_formatting"
        assert "vulcan_reasoning" in result["systems_used"]
        assert result["metadata"]["reasoning_output"]["result"] == "x**3/3 + C"

    @pytest.mark.asyncio
    async def test_format_output_fallback_when_openai_unavailable(self, executor_without_openai):
        """Test fallback to internal formatting when OpenAI is unavailable."""
        reasoning_output = {
            "success": True,
            "result": 42,
            "confidence": 0.9,
        }
        
        result = await executor_without_openai.format_output_for_user(
            reasoning_output=reasoning_output,
            original_prompt="What is 6*7?",
        )
        
        assert result is not None
        assert result["source"] == "internal_formatting"
        assert "internal_formatting" in result["systems_used"]
        assert result["metadata"]["fallback_reason"] == "openai_unavailable"

    @pytest.mark.asyncio
    async def test_format_output_error_handling(self, executor_with_openai):
        """Test error handling for failed reasoning."""
        reasoning_output = {
            "success": False,
            "result": None,
            "error": "Timeout during computation",
        }
        
        result = await executor_with_openai.format_output_for_user(
            reasoning_output=reasoning_output,
            original_prompt="Complex query",
        )
        
        assert result is not None
        assert result.get("error") is True
        assert result["source"] == "vulcan_reasoning_error"
        assert "timeout" in result["text"].lower() or "issue" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_format_output_distillation_capture(self, executor_with_openai):
        """Test that distillation capture is tracked."""
        executor_with_openai._call_openai_formatting = AsyncMock(
            return_value="The answer based on VULCAN's reasoning is: **100**"
        )
        executor_with_openai._distillation_enabled = True
        
        # Mock the capture method to track calls
        capture_calls = []
        original_capture = executor_with_openai._capture_formatting_for_distillation
        def mock_capture(*args, **kwargs):
            capture_calls.append((args, kwargs))
            return original_capture(*args, **kwargs)
        executor_with_openai._capture_formatting_for_distillation = mock_capture
        
        reasoning_output = {
            "success": True,
            "result": 100,
            "confidence": 0.99,
        }
        
        result = await executor_with_openai.format_output_for_user(
            reasoning_output=reasoning_output,
            original_prompt="What is 10*10?",
        )
        
        assert result["distillation_captured"] is True
        # Capture is called from _format_with_openai_for_output
        assert len(capture_calls) >= 1

    @pytest.mark.asyncio
    async def test_format_output_with_converted_type(self, executor_with_openai):
        """Test handling of non-standard input types (converted to string)."""
        executor_with_openai._call_openai_formatting = AsyncMock(
            return_value="Result: raw string value"
        )
        
        # Pass an unexpected type (e.g., a plain string)
        result = await executor_with_openai.format_output_for_user(
            reasoning_output="plain string result",
            original_prompt="Test query",
        )
        
        assert result is not None
        # Should handle gracefully
        assert "text" in result


class TestOpenAILanguageFormattingConfig:
    """Test OPENAI_LANGUAGE_FORMATTING configuration."""

    def test_config_constants_exist(self):
        """Test that configuration constants are defined."""
        from vulcan.llm.hybrid_executor import (
            OPENAI_LANGUAGE_FORMATTING,
            OPENAI_LANGUAGE_POLISH,
        )
        
        # Should be boolean values (False by default without env vars)
        assert isinstance(OPENAI_LANGUAGE_FORMATTING, bool)
        assert isinstance(OPENAI_LANGUAGE_POLISH, bool)

    def test_config_default_values(self):
        """Test default configuration values."""
        from vulcan.llm.hybrid_executor import (
            OPENAI_LANGUAGE_FORMATTING,
            OPENAI_LANGUAGE_POLISH,
        )
        
        # FIX: OPENAI_LANGUAGE_FORMATTING now defaults to True to prevent 60-second timeouts
        # This matches .env.example documentation and enables fast OpenAI formatting (~2-5s)
        # Note: This test assumes no env vars are set
        assert OPENAI_LANGUAGE_FORMATTING is True  # Changed from False to True
        assert OPENAI_LANGUAGE_POLISH is False


class TestParallelExecution:
    """Test the restored true parallel execution mode.
    
    These tests verify:
    1. True parallel execution runs both backends simultaneously
    2. First successful response wins
    3. Graceful fallback when OpenAI is unavailable
    4. Task cancellation after first success
    5. Timeout handling
    """

    @pytest.fixture
    def mock_local_llm(self):
        """Create a mock local LLM."""
        mock = MagicMock()
        mock.generate.return_value = MagicMock(text="Local LLM response")
        return mock

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def executor_with_both(self, mock_local_llm, mock_openai_client):
        """Create HybridLLMExecutor with both backends available."""
        executor = HybridLLMExecutor(
            local_llm=mock_local_llm,
            openai_client_getter=lambda: mock_openai_client,
            mode="parallel",
            timeout=30.0,
        )
        return executor

    @pytest.fixture
    def executor_without_openai(self, mock_local_llm):
        """Create HybridLLMExecutor without OpenAI access."""
        executor = HybridLLMExecutor(
            local_llm=mock_local_llm,
            openai_client_getter=lambda: None,  # No OpenAI
            mode="parallel",
            timeout=30.0,
        )
        return executor

    def test_openai_available_method(self, executor_with_both, executor_without_openai):
        """Test _openai_available helper method."""
        assert executor_with_both._openai_available() is True
        assert executor_without_openai._openai_available() is False

    @pytest.mark.asyncio
    async def test_parallel_mode_with_openai_available(self, executor_with_both, mock_openai_client):
        """Test parallel execution when OpenAI is available."""
        # Mock both _call_openai and _call_local_llm
        executor_with_both._call_openai = AsyncMock(return_value="OpenAI wins!")
        executor_with_both._call_local_llm = AsyncMock(return_value=None)  # Local fails
        
        result = await executor_with_both.execute(
            prompt="Hello, how are you?",
            max_tokens=50,
            temperature=0.7,
        )
        
        assert result is not None
        assert "text" in result
        # OpenAI should win since local failed
        assert "parallel" in result.get("source", "") or "openai" in result.get("source", "").lower()

    @pytest.mark.asyncio
    async def test_parallel_fallback_without_openai(self, executor_without_openai):
        """Test that parallel mode falls back to local_first when OpenAI unavailable."""
        # Since OpenAI is not available, it should fall back to local_first
        result = await executor_without_openai.execute(
            prompt="Test query",
            max_tokens=50,
        )
        
        assert result is not None
        # Should have used local execution
        assert "text" in result

    @pytest.mark.asyncio
    async def test_run_local_llm_wrapper(self, executor_with_both):
        """Test _run_local_llm async wrapper returns correct format."""
        executor_with_both._call_local_llm = AsyncMock(return_value="Local LLM result")
        
        loop = asyncio.get_event_loop()
        result = await executor_with_both._run_local_llm(loop, "test prompt", 100)
        
        assert result is not None
        assert "text" in result
        assert result["text"] == "Local LLM result"
        assert result["source"] == "local"
        assert "vulcan_local_llm" in result["systems_used"]

    @pytest.mark.asyncio
    async def test_run_openai_wrapper(self, executor_with_both):
        """Test _run_openai async wrapper returns correct format."""
        executor_with_both._call_openai = AsyncMock(return_value="OpenAI result")
        
        loop = asyncio.get_event_loop()
        result = await executor_with_both._run_openai(
            loop, "test prompt", 100, 0.7, "system prompt", None
        )
        
        assert result is not None
        assert "text" in result
        assert result["text"] == "OpenAI result"
        assert result["source"] == "openai"
        assert "openai" in result["systems_used"]

    @pytest.mark.asyncio
    async def test_run_local_llm_failure_returns_empty(self, executor_with_both):
        """Test _run_local_llm returns empty result on failure."""
        executor_with_both._call_local_llm = AsyncMock(return_value=None)
        
        loop = asyncio.get_event_loop()
        result = await executor_with_both._run_local_llm(loop, "test prompt", 100)
        
        assert result is not None
        assert result["text"] is None
        assert "failed" in result["source"]

    @pytest.mark.asyncio
    async def test_run_openai_failure_returns_empty(self, executor_with_both):
        """Test _run_openai returns empty result on failure."""
        executor_with_both._call_openai = AsyncMock(return_value=None)
        
        loop = asyncio.get_event_loop()
        result = await executor_with_both._run_openai(
            loop, "test prompt", 100, 0.7, "system prompt", None
        )
        
        assert result is not None
        assert result["text"] is None
        assert "failed" in result["source"]

    @pytest.mark.asyncio
    async def test_parallel_first_wins(self, executor_with_both):
        """Test that the first successful response wins in parallel mode."""
        import asyncio
        
        # Make OpenAI return quickly, local return slowly
        async def slow_local(*args, **kwargs):
            await asyncio.sleep(5)  # Very slow
            return "Slow local result"
        
        async def fast_openai(*args, **kwargs):
            await asyncio.sleep(0.01)  # Very fast
            return "Fast OpenAI result"
        
        executor_with_both._call_local_llm = slow_local
        executor_with_both._call_openai = fast_openai
        
        result = await executor_with_both.execute(
            prompt="Test query",
            max_tokens=50,
        )
        
        assert result is not None
        # OpenAI should win since it's faster
        if "parallel" in result.get("source", ""):
            assert "openai" in result.get("source", "").lower()


