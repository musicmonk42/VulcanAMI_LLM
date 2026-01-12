#!/usr/bin/env python3
"""
Test for _get_reasoning_attr helper function in unified_chat.py.

This test validates the fix for Bug 1: Missing _get_reasoning_attr function
that was causing NameError exceptions and 500 Internal Server Errors.

The function safely extracts attributes from reasoning output that can be
either a dict or an object.
"""

import pytest
from vulcan.endpoints.unified_chat import _get_reasoning_attr


class MockReasoningResult:
    """Mock reasoning result object for testing."""
    
    def __init__(self):
        self.conclusion = "Test conclusion"
        self.confidence = 0.85
        self.reasoning_type = "causal_reasoning"
        self.explanation = "Test explanation"


class TestGetReasoningAttr:
    """Test suite for _get_reasoning_attr helper function."""

    def test_function_exists(self):
        """Verify _get_reasoning_attr function exists and is callable."""
        assert callable(_get_reasoning_attr), \
            "_get_reasoning_attr must be a callable function"

    def test_extract_from_dict(self):
        """Test extracting attributes from dictionary."""
        reasoning_dict = {
            "conclusion": "The answer is 42",
            "confidence": 0.95,
            "reasoning_type": "mathematical",
            "explanation": "Calculated using formula X"
        }
        
        assert _get_reasoning_attr(reasoning_dict, "conclusion") == "The answer is 42"
        assert _get_reasoning_attr(reasoning_dict, "confidence") == 0.95
        assert _get_reasoning_attr(reasoning_dict, "reasoning_type") == "mathematical"
        assert _get_reasoning_attr(reasoning_dict, "explanation") == "Calculated using formula X"

    def test_extract_from_object(self):
        """Test extracting attributes from object."""
        reasoning_obj = MockReasoningResult()
        
        assert _get_reasoning_attr(reasoning_obj, "conclusion") == "Test conclusion"
        assert _get_reasoning_attr(reasoning_obj, "confidence") == 0.85
        assert _get_reasoning_attr(reasoning_obj, "reasoning_type") == "causal_reasoning"
        assert _get_reasoning_attr(reasoning_obj, "explanation") == "Test explanation"

    def test_missing_attr_returns_default_none(self):
        """Test that missing attributes return None by default."""
        reasoning_dict = {"conclusion": "test"}
        
        assert _get_reasoning_attr(reasoning_dict, "missing_key") is None
        
        reasoning_obj = MockReasoningResult()
        assert _get_reasoning_attr(reasoning_obj, "missing_attr") is None

    def test_missing_attr_returns_custom_default(self):
        """Test that missing attributes return custom default value."""
        reasoning_dict = {"conclusion": "test"}
        
        assert _get_reasoning_attr(reasoning_dict, "missing", default="N/A") == "N/A"
        assert _get_reasoning_attr(reasoning_dict, "missing", default=0.0) == 0.0
        assert _get_reasoning_attr(reasoning_dict, "missing", default=[]) == []

    def test_none_input_returns_default(self):
        """Test that None input returns default value."""
        assert _get_reasoning_attr(None, "any_attr") is None
        assert _get_reasoning_attr(None, "any_attr", default="fallback") == "fallback"

    def test_empty_dict_returns_default(self):
        """Test that empty dict returns default for missing keys."""
        assert _get_reasoning_attr({}, "conclusion") is None
        assert _get_reasoning_attr({}, "conclusion", default="") == ""

    def test_real_world_usage_pattern(self):
        """Test real-world usage pattern from unified_chat.py line 1372."""
        # Simulate agent_reasoning_output from agent pool
        agent_reasoning_output = {
            "conclusion": "Earthquakes are caused by tectonic plate movement",
            "confidence": 0.92,
            "reasoning_type": "causal_reasoning",
            "explanation": "Analysis of seismic data and geological models"
        }
        
        # Extract values as done in unified_chat.py
        extracted_conclusion = _get_reasoning_attr(agent_reasoning_output, "conclusion")
        extracted_confidence = _get_reasoning_attr(agent_reasoning_output, "confidence")
        extracted_type = _get_reasoning_attr(agent_reasoning_output, "reasoning_type")
        extracted_explanation = _get_reasoning_attr(agent_reasoning_output, "explanation")
        
        # Build result dict as in unified_chat.py
        reasoning_results = {
            "agent_reasoning": {
                "conclusion": extracted_conclusion,
                "confidence": extracted_confidence,
                "reasoning_type": extracted_type,
                "explanation": extracted_explanation,
            }
        }
        
        # Verify extraction worked
        assert reasoning_results["agent_reasoning"]["conclusion"] == \
            "Earthquakes are caused by tectonic plate movement"
        assert reasoning_results["agent_reasoning"]["confidence"] == 0.92
        assert reasoning_results["agent_reasoning"]["reasoning_type"] == "causal_reasoning"

    def test_handles_mixed_types(self):
        """Test handling various data types as attribute values."""
        reasoning_dict = {
            "string_value": "text",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "data"},
            "none_value": None
        }
        
        assert _get_reasoning_attr(reasoning_dict, "string_value") == "text"
        assert _get_reasoning_attr(reasoning_dict, "int_value") == 42
        assert _get_reasoning_attr(reasoning_dict, "float_value") == 3.14
        assert _get_reasoning_attr(reasoning_dict, "bool_value") is True
        assert _get_reasoning_attr(reasoning_dict, "list_value") == [1, 2, 3]
        assert _get_reasoning_attr(reasoning_dict, "dict_value") == {"nested": "data"}
        assert _get_reasoning_attr(reasoning_dict, "none_value") is None

    def test_defensive_against_attribute_error(self):
        """Test that function doesn't raise AttributeError."""
        # This would raise AttributeError with getattr() on dict
        # but should return default with _get_reasoning_attr
        reasoning_dict = {"key": "value"}
        
        # Should NOT raise AttributeError
        result = _get_reasoning_attr(reasoning_dict, "nonexistent_key", default="safe")
        assert result == "safe"

    def test_defensive_against_key_error(self):
        """Test that function doesn't raise KeyError."""
        # This would raise KeyError with dict[] access
        # but should return default with _get_reasoning_attr
        reasoning_dict = {"key": "value"}
        
        # Should NOT raise KeyError
        result = _get_reasoning_attr(reasoning_dict, "missing", default="safe")
        assert result == "safe"

    def test_object_without_attribute(self):
        """Test object that doesn't have the requested attribute."""
        class SimpleObject:
            def __init__(self):
                self.existing = "value"
        
        obj = SimpleObject()
        
        # Should return default, not raise AttributeError
        assert _get_reasoning_attr(obj, "nonexistent", default="fallback") == "fallback"
        assert _get_reasoning_attr(obj, "existing") == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
