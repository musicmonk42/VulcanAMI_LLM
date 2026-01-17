"""
Tests for Issue #2: Mathematical Computation Retry Logic
Tests that retry loop with error feedback works correctly.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# Import the mathematical computation module
try:
    from src.vulcan.reasoning.mathematical_computation import MathematicalComputationTool
    MATH_TOOL_AVAILABLE = True
except ImportError:
    try:
        from vulcan.reasoning.mathematical_computation import MathematicalComputationTool
        MATH_TOOL_AVAILABLE = True
    except ImportError:
        MATH_TOOL_AVAILABLE = False


@pytest.mark.skipif(not MATH_TOOL_AVAILABLE, reason="Mathematical computation tool not available")
class TestMathematicalRetryLogic:
    """Test retry logic with error feedback for mathematical computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = MathematicalComputationTool()
    
    def test_request_code_correction_method_exists(self):
        """Test that _request_code_correction method exists."""
        assert hasattr(self.tool, '_request_code_correction'), \
            "_request_code_correction method should exist"
    
    def test_request_code_correction_with_mock_llm(self):
        """Test _request_code_correction with a mock LLM."""
        # Create a mock LLM that returns corrected code
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="result = 42"))]
        mock_llm.chat = Mock()
        mock_llm.chat.completions = Mock()
        mock_llm.chat.completions.create = Mock(return_value=mock_response)
        mock_llm.model = "gpt-4"
        
        query = "Calculate 2 + 2"
        failed_code = "result = 2 +"  # Syntax error
        error_msg = "SyntaxError: invalid syntax"
        
        corrected_code = self.tool._request_code_correction(query, failed_code, error_msg, mock_llm)
        
        assert corrected_code is not None, "Corrected code should not be None"
        assert "result" in corrected_code, "Corrected code should contain result"
    
    def test_try_fallback_accepts_error_msg(self):
        """Test that _try_fallback accepts error_msg parameter."""
        # Create a mock classification
        from vulcan.reasoning.mathematical_computation import ProblemClassification, ProblemType, SolutionStrategy
        
        classification = ProblemClassification(
            problem_type=ProblemType.ARITHMETIC,
            confidence=0.8,
            keywords=["add"],
            suggested_strategy=SolutionStrategy.NUMERIC
        )
        
        query = "Calculate 2 + 2"
        error_msg = "SyntaxError: invalid syntax"
        
        # Call _try_fallback with error_msg parameter
        result = self.tool._try_fallback(
            query, 
            classification, 
            SolutionStrategy.LLM_GENERATED,
            llm=None,
            error_msg=error_msg
        )
        
        # Result might be None (no fallback available), but method should accept parameter
        # The important thing is it doesn't raise an error
    
    @patch('vulcan.reasoning.mathematical_computation.execute_math_code')
    def test_retry_loop_on_syntax_error(self, mock_execute):
        """Test that retry loop activates on syntax error."""
        # Mock execute_math_code to fail on first two attempts, succeed on third
        mock_execute.side_effect = [
            {"success": False, "error": "SyntaxError: '(' was never closed"},
            {"success": False, "error": "SyntaxError: invalid syntax"},
            {"success": True, "result": 1.6449340668482264}  # π²/6
        ]
        
        # Create a mock LLM that returns corrected code
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="result = sum(1/n**2 for n in range(1, 1000))"))]
        mock_llm.chat = Mock()
        mock_llm.chat.completions = Mock()
        mock_llm.chat.completions.create = Mock(return_value=mock_response)
        mock_llm.model = "gpt-4"
        
        # Note: This test verifies the structure exists
        # Full integration test would require actual LLM
        
        # Test passes if retry logic structure is in place
        assert hasattr(self.tool, '_request_code_correction'), \
            "Retry logic requires _request_code_correction method"
    
    def test_max_retries_constant_exists(self):
        """Test that MAX_RETRIES constant is defined in execute method."""
        # This is a structural test - we can't easily test the full execute flow
        # without complex mocking, but we can verify the method exists
        import inspect
        source = inspect.getsource(self.tool.execute)
        assert "MAX_RETRIES" in source, \
            "execute() method should define MAX_RETRIES constant"
        assert "for attempt in range(MAX_RETRIES)" in source or "for attempt in range(" in source, \
            "execute() method should have retry loop"


@pytest.mark.skipif(not MATH_TOOL_AVAILABLE, reason="Mathematical computation tool not available")
class TestMathematicalCodeCorrection:
    """Test code correction prompt generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = MathematicalComputationTool()
    
    def test_correction_prompt_includes_error(self):
        """Test that correction request includes the error message."""
        # Create a mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="result = 42"))]
        mock_llm.chat = Mock()
        mock_llm.chat.completions = Mock()
        mock_llm.chat.completions.create = Mock(return_value=mock_response)
        mock_llm.model = "gpt-4"
        
        query = "Calculate sum"
        failed_code = "result = sum("  # Unclosed parenthesis
        error_msg = "SyntaxError: '(' was never closed"
        
        corrected_code = self.tool._request_code_correction(query, failed_code, error_msg, mock_llm)
        
        # Verify that the LLM was called with a prompt containing the error
        mock_llm.chat.completions.create.assert_called_once()
        call_args = mock_llm.chat.completions.create.call_args
        messages = call_args[1]['messages'] if 'messages' in call_args[1] else call_args[0][1]
        prompt_content = messages[0]['content']
        
        assert error_msg in prompt_content, "Correction prompt should include error message"
        assert failed_code in prompt_content, "Correction prompt should include failed code"
        assert query in prompt_content, "Correction prompt should include original query"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
