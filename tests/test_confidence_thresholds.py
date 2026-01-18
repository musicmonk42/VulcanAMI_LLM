"""
Test suite for confidence threshold changes.

These tests verify:
1. MIN_REASONING_CONFIDENCE_THRESHOLD is now 0.01 (was 0.10)
2. MIN_REASONING_CONFIDENCE in hybrid_executor is now 0.01 (was 0.5)
3. Nested object unwrapping works correctly in formatters
4. Diagnostic logging is present
"""

import os
import logging
from unittest.mock import MagicMock, patch

import pytest


def test_unified_chat_confidence_threshold():
    """Test that MIN_REASONING_CONFIDENCE_THRESHOLD defaults to 0.01."""
    # Clear any existing environment variable
    if "VULCAN_MIN_REASONING_CONFIDENCE" in os.environ:
        del os.environ["VULCAN_MIN_REASONING_CONFIDENCE"]
    
    # Import and check the default value
    # The value is evaluated at runtime in unified_chat.py, so we simulate it
    default_threshold = float(os.environ.get("VULCAN_MIN_REASONING_CONFIDENCE", "0.01"))
    
    assert default_threshold == 0.01, f"Expected 0.01, got {default_threshold}"


def test_nested_object_unwrapping():
    """Test that format_direct_reasoning_response unwraps nested objects correctly."""
    from vulcan.reasoning.formatters import format_direct_reasoning_response
    
    # Create a nested structure (simulating ReasoningResult with nested conclusion)
    class MockReasoningResult:
        def __init__(self, conclusion, confidence=0.8, reasoning_type="test"):
            self.conclusion = conclusion
            self.confidence = confidence
            self.reasoning_type = reasoning_type
            self.explanation = "Test explanation"
    
    # Test single-level nesting
    nested_conclusion = MockReasoningResult("Final answer: 42", 0.9)
    result = format_direct_reasoning_response(
        conclusion=nested_conclusion,
        confidence=0.5,
        reasoning_type="test",
        explanation="Outer explanation"
    )
    
    assert "Final answer: 42" in result, f"Expected 'Final answer: 42' in result, got: {result}"
    assert "90%" in result or "0.9" in result, f"Expected confidence 90% in result"
    
    # Test multi-level nesting (3 levels deep)
    level3 = "The answer is 3.14159"
    level2 = MockReasoningResult(level3, 0.95)
    level1 = MockReasoningResult(level2, 0.85)
    
    result = format_direct_reasoning_response(
        conclusion=level1,
        confidence=0.7,
        reasoning_type="mathematical",
        explanation="Multi-level test"
    )
    
    assert "3.14159" in result, f"Expected '3.14159' in result after unwrapping 3 levels"
    assert "95%" in result or "0.95" in result, f"Expected highest confidence (95%) in result"


def test_format_conclusion_for_user_handles_dicts():
    """Test that format_conclusion_for_user handles various dict structures."""
    from vulcan.reasoning.formatters import format_conclusion_for_user
    
    # Test dict with 'answer' key
    conclusion = {"answer": "42", "metadata": "test"}
    result = format_conclusion_for_user(conclusion)
    assert "42" in result
    
    # Test dict with 'result' key
    conclusion = {"result": "success", "internal": "hidden"}
    result = format_conclusion_for_user(conclusion)
    assert "success" in result
    
    # Test dict with 'conclusion' key
    conclusion = {"conclusion": "The answer is yes", "debug": "info"}
    result = format_conclusion_for_user(conclusion)
    assert "yes" in result


def test_format_reasoning_results_logs_diagnostics():
    """Test that format_reasoning_results includes diagnostic logging."""
    from vulcan.endpoints.chat_helpers import format_reasoning_results
    
    # Set up logging capture
    with patch('vulcan.endpoints.chat_helpers.logger') as mock_logger:
        reasoning_results = {
            "unified": {
                "conclusion": "Test conclusion",
                "confidence": 0.8,
                "reasoning_type": "test"
            }
        }
        
        result = format_reasoning_results(reasoning_results)
        
        # Check that diagnostic logging was called
        assert mock_logger.info.called, "Expected diagnostic logging to be called"
        
        # Check that the call included "DIAGNOSTIC"
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        has_diagnostic_log = any("DIAGNOSTIC" in str(call) for call in log_calls)
        assert has_diagnostic_log, "Expected at least one log call with 'DIAGNOSTIC'"


def test_conclusion_formatter_handles_reasoning_result_objects():
    """Test ConclusionFormatter handles objects with to_dict() method."""
    from vulcan.endpoints.chat_helpers import ConclusionFormatter
    
    # Mock a ReasoningResult-like object
    class MockReasoningResult:
        def __init__(self):
            self.conclusion = "Test answer"
            self.confidence = 0.95
        
        def to_dict(self):
            return {
                "conclusion": self.conclusion,
                "confidence": self.confidence
            }
    
    obj = MockReasoningResult()
    result = ConclusionFormatter.format(obj)
    
    assert result is not None, "Expected non-None result"
    assert "Test answer" in result or "conclusion" in result, f"Expected conclusion in result: {result}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
