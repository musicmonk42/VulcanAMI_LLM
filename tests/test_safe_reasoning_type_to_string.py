"""
Tests for safe_reasoning_type_to_string serialization fix.

This test suite verifies the fix for the critical bug:
    "'ReasoningType' object has no attribute 'replace'"

The bug occurs when code calls string methods like .replace() on
ReasoningType enum objects instead of their string values.

Industry Standard: This test suite follows defensive programming patterns
and ensures type-safe serialization of enum values.
"""

import pytest
from enum import Enum
from typing import Any, Optional


# Import the function under test
try:
    from vulcan.endpoints.chat_helpers import (
        safe_reasoning_type_to_string,
        format_reasoning_type_for_display,
    )
    HELPERS_AVAILABLE = True
except ImportError:
    safe_reasoning_type_to_string = None
    format_reasoning_type_for_display = None
    HELPERS_AVAILABLE = False


# Import ReasoningType enum if available
try:
    from vulcan.reasoning.reasoning_types import ReasoningType
    REASONING_TYPE_AVAILABLE = True
except ImportError:
    ReasoningType = None
    REASONING_TYPE_AVAILABLE = False


class TestSafeReasoningTypeToString:
    """Test suite for safe_reasoning_type_to_string helper."""

    def test_import_available(self):
        """Test that required functions can be imported."""
        assert HELPERS_AVAILABLE, "safe_reasoning_type_to_string should be available"

    def test_none_returns_default(self):
        """Test that None returns the default value."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string(None)
        assert result == "unknown"
        
        result = safe_reasoning_type_to_string(None, default="hybrid")
        assert result == "hybrid"

    def test_string_passthrough(self):
        """Test that strings pass through unchanged."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string("probabilistic")
        assert result == "probabilistic"
        
        result = safe_reasoning_type_to_string("causal_reasoning")
        assert result == "causal_reasoning"
        
        result = safe_reasoning_type_to_string("")
        assert result == ""

    def test_enum_extracts_value(self):
        """Test that Enum objects have their .value extracted."""
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        # Test with ReasoningType enum
        result = safe_reasoning_type_to_string(ReasoningType.PROBABILISTIC)
        assert result == "probabilistic"
        
        result = safe_reasoning_type_to_string(ReasoningType.SYMBOLIC)
        assert result == "symbolic"
        
        result = safe_reasoning_type_to_string(ReasoningType.PHILOSOPHICAL)
        assert result == "philosophical"

    def test_custom_enum(self):
        """Test with custom enum to verify generic enum handling."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        class CustomEnum(Enum):
            FOO = "foo_value"
            BAR = "bar_value"
        
        result = safe_reasoning_type_to_string(CustomEnum.FOO)
        assert result == "foo_value"
        
        result = safe_reasoning_type_to_string(CustomEnum.BAR)
        assert result == "bar_value"

    def test_object_with_value_attribute(self):
        """Test with objects that have .value attribute (enum-like)."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        class EnumLike:
            def __init__(self, val):
                self.value = val
        
        obj = EnumLike("custom_value")
        result = safe_reasoning_type_to_string(obj)
        assert result == "custom_value"

    def test_fallback_to_str(self):
        """Test fallback to str() for unknown types."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string(123)
        assert result == "123"
        
        result = safe_reasoning_type_to_string(3.14)
        assert result == "3.14"

    def test_prevents_replace_error(self):
        """
        CRITICAL TEST: Verify that the function prevents the AttributeError.
        
        This is the exact bug that was causing:
            "'ReasoningType' object has no attribute 'replace'"
        """
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        # This should NOT raise AttributeError
        reasoning_type = ReasoningType.PROBABILISTIC
        type_str = safe_reasoning_type_to_string(reasoning_type)
        
        # Now we can safely call string methods
        formatted = type_str.replace("_", " ")
        assert formatted == "probabilistic"
        
        # Test with formatting
        formatted = type_str.replace("_", " ").title()
        assert formatted == "Probabilistic"


class TestFormatReasoningTypeForDisplay:
    """Test suite for format_reasoning_type_for_display helper."""

    def test_import_available(self):
        """Test that required functions can be imported."""
        assert HELPERS_AVAILABLE, "format_reasoning_type_for_display should be available"

    def test_none_returns_default(self):
        """Test that None returns the default display value."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = format_reasoning_type_for_display(None)
        assert result == "Hybrid"
        
        result = format_reasoning_type_for_display(None, default="Unknown")
        assert result == "Unknown"

    def test_string_formatting(self):
        """Test proper formatting of string values."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = format_reasoning_type_for_display("probabilistic")
        assert result == "Probabilistic"
        
        result = format_reasoning_type_for_display("causal_reasoning")
        assert result == "Causal Reasoning"
        
        result = format_reasoning_type_for_display("meta_reasoning")
        assert result == "Meta Reasoning"

    def test_enum_formatting(self):
        """Test proper formatting of enum values."""
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        result = format_reasoning_type_for_display(ReasoningType.PROBABILISTIC)
        assert result == "Probabilistic"
        
        result = format_reasoning_type_for_display(ReasoningType.CAUSAL)
        assert result == "Causal"

    def test_prevents_replace_error_end_to_end(self):
        """
        CRITICAL TEST: End-to-end verification that formatting doesn't crash.
        
        This simulates the exact scenario where the bug occurred.
        """
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        # Simulate the problematic code path
        reasoning_type = ReasoningType.PHILOSOPHICAL
        
        # This was failing before the fix:
        # reasoning_type_display = reasoning_type.replace("_", " ").title()
        
        # Now it works:
        reasoning_type_display = format_reasoning_type_for_display(reasoning_type)
        assert reasoning_type_display == "Philosophical"
        
        # Also test with None (edge case)
        reasoning_type_display = format_reasoning_type_for_display(None)
        assert reasoning_type_display == "Hybrid"


class TestMetadataSerialization:
    """
    Test suite for metadata serialization scenarios.
    
    These tests verify that reasoning_type values can be safely
    serialized to JSON-compatible dictionaries.
    """

    def test_dict_serialization_with_enum(self):
        """Test that enum values can be safely added to dicts."""
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        reasoning_type = ReasoningType.SYMBOLIC
        
        # This is how metadata should be built
        metadata = {
            "reasoning_type": safe_reasoning_type_to_string(reasoning_type),
            "confidence": 0.95,
        }
        
        assert metadata["reasoning_type"] == "symbolic"
        assert isinstance(metadata["reasoning_type"], str)

    def test_json_serializable(self):
        """Test that converted values are JSON serializable."""
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        import json
        
        reasoning_type = ReasoningType.MATHEMATICAL
        type_str = safe_reasoning_type_to_string(reasoning_type)
        
        # This should not raise
        metadata = {
            "reasoning_type": type_str,
            "tool": type_str,
        }
        json_str = json.dumps(metadata)
        
        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed["reasoning_type"] == "mathematical"

    def test_pydantic_model_compatible(self):
        """Test compatibility with Pydantic models (used by FastAPI)."""
        if not HELPERS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        try:
            from pydantic import BaseModel
            from typing import Dict, Any
        except ImportError:
            pytest.skip("Pydantic not available")
        
        class ResponseModel(BaseModel):
            response: str
            metadata: Dict[str, Any]
        
        reasoning_type = ReasoningType.CAUSAL
        
        # Build response with safe serialization
        response = ResponseModel(
            response="Test response",
            metadata={
                "reasoning_type": safe_reasoning_type_to_string(reasoning_type),
            }
        )
        
        # This should not crash
        json_dict = response.model_dump()
        assert json_dict["metadata"]["reasoning_type"] == "causal"


class TestEdgeCases:
    """Test edge cases and defensive programming scenarios."""

    def test_empty_string(self):
        """Test handling of empty string."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string("")
        assert result == ""
        
        result = safe_reasoning_type_to_string("", default="fallback")
        assert result == ""  # Empty string is not None, so no default

    def test_whitespace_string(self):
        """Test handling of whitespace-only string."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string("  ")
        assert result == "  "

    def test_numeric_types(self):
        """Test handling of numeric types."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string(42)
        assert result == "42"
        
        result = safe_reasoning_type_to_string(0)
        assert result == "0"

    def test_boolean_types(self):
        """Test handling of boolean types."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string(True)
        assert result == "True"
        
        result = safe_reasoning_type_to_string(False)
        assert result == "False"

    def test_list_type_fallback(self):
        """Test that list types fall back to str()."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string(["a", "b"])
        assert result == "['a', 'b']"

    def test_dict_type_fallback(self):
        """Test that dict types fall back to str()."""
        if not HELPERS_AVAILABLE:
            pytest.skip("Helper functions not available")
        
        result = safe_reasoning_type_to_string({"key": "value"})
        assert result == "{'key': 'value'}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
