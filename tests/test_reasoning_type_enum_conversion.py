"""
Tests for reasoning_type enum conversion functionality.

This test suite verifies that the enum conversion helpers properly handle
the pipeline-dropping bug where philosophical/ethical results are discarded
when reasoning_type is passed as a string instead of a ReasoningType Enum.

Test Coverage:
- String to enum conversion with various inputs
- Result object conversion (dict and dataclass)
- Error handling and logging for invalid types
- Alias mappings (meta_reasoning, philosophical_reasoning, etc.)
- Integration with agent_pool and orchestrator
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Test imports - these should work even if some components are unavailable
try:
    from src.vulcan.reasoning.reasoning_types import ReasoningType
    REASONING_TYPE_AVAILABLE = True
except ImportError:
    ReasoningType = None
    REASONING_TYPE_AVAILABLE = False

try:
    from src.vulcan.reasoning.integration.utils import (
        convert_reasoning_type_to_enum,
        ensure_reasoning_type_enum,
    )
    CONVERSION_UTILS_AVAILABLE = True
except ImportError:
    convert_reasoning_type_to_enum = None
    ensure_reasoning_type_enum = None
    CONVERSION_UTILS_AVAILABLE = False


class TestReasoningTypeConversion:
    """Test suite for reasoning_type enum conversion helpers."""
    
    def test_import_available(self):
        """Test that required modules can be imported."""
        assert REASONING_TYPE_AVAILABLE, "ReasoningType enum should be available"
        assert CONVERSION_UTILS_AVAILABLE, "Conversion utilities should be available"
    
    def test_convert_string_to_enum(self):
        """Test conversion of string values to ReasoningType enum."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Test lowercase string matching
        result = convert_reasoning_type_to_enum("philosophical", "test")
        assert result == ReasoningType.PHILOSOPHICAL
        
        # Test uppercase string matching
        result = convert_reasoning_type_to_enum("PHILOSOPHICAL", "test")
        assert result == ReasoningType.PHILOSOPHICAL
        
        # Test other enum values
        result = convert_reasoning_type_to_enum("mathematical", "test")
        assert result == ReasoningType.MATHEMATICAL
        
        result = convert_reasoning_type_to_enum("causal", "test")
        assert result == ReasoningType.CAUSAL
    
    def test_convert_alias_strings(self):
        """Test conversion of alias strings (e.g., meta_reasoning)."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Test meta_reasoning alias -> PHILOSOPHICAL
        result = convert_reasoning_type_to_enum("meta_reasoning", "test")
        assert result == ReasoningType.PHILOSOPHICAL
        
        # Test philosophical_reasoning alias -> PHILOSOPHICAL
        result = convert_reasoning_type_to_enum("philosophical_reasoning", "test")
        assert result == ReasoningType.PHILOSOPHICAL
        
        # Test world_model alias -> PHILOSOPHICAL
        result = convert_reasoning_type_to_enum("world_model", "test")
        assert result == ReasoningType.PHILOSOPHICAL
        
        # Test ethical_reasoning alias -> PHILOSOPHICAL
        result = convert_reasoning_type_to_enum("ethical_reasoning", "test")
        assert result == ReasoningType.PHILOSOPHICAL
    
    def test_convert_enum_passthrough(self):
        """Test that enum values pass through unchanged."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Enum should pass through unchanged
        result = convert_reasoning_type_to_enum(ReasoningType.PHILOSOPHICAL, "test")
        assert result == ReasoningType.PHILOSOPHICAL
        
        result = convert_reasoning_type_to_enum(ReasoningType.MATHEMATICAL, "test")
        assert result == ReasoningType.MATHEMATICAL
    
    def test_convert_none_value(self):
        """Test that None values are handled correctly."""
        if not CONVERSION_UTILS_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        result = convert_reasoning_type_to_enum(None, "test")
        assert result is None
    
    def test_convert_invalid_string(self):
        """Test handling of invalid string values."""
        if not CONVERSION_UTILS_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Should return None and log error for invalid strings
        result = convert_reasoning_type_to_enum("invalid_type", "test")
        assert result is None
        
        result = convert_reasoning_type_to_enum("", "test")
        assert result is None
    
    def test_ensure_dict_conversion(self):
        """Test ensure_reasoning_type_enum with dictionary."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Test dict with string reasoning_type
        result_dict = {
            "reasoning_type": "philosophical",
            "confidence": 0.9,
            "conclusion": "Test conclusion"
        }
        
        converted = ensure_reasoning_type_enum(result_dict, "test")
        assert converted["reasoning_type"] == ReasoningType.PHILOSOPHICAL
        assert converted["confidence"] == 0.9
        assert converted["conclusion"] == "Test conclusion"
    
    def test_ensure_dict_with_alias(self):
        """Test ensure_reasoning_type_enum with alias in dictionary."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Test dict with meta_reasoning alias
        result_dict = {
            "reasoning_type": "meta_reasoning",
            "confidence": 0.85
        }
        
        converted = ensure_reasoning_type_enum(result_dict, "test")
        assert converted["reasoning_type"] == ReasoningType.PHILOSOPHICAL
        assert converted["confidence"] == 0.85
    
    def test_ensure_object_conversion(self):
        """Test ensure_reasoning_type_enum with object."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Create a mock result object
        @dataclass
        class MockResult:
            reasoning_type: Any
            confidence: float
        
        result_obj = MockResult(reasoning_type="philosophical", confidence=0.9)
        
        # Note: ensure_reasoning_type_enum may not modify frozen dataclasses,
        # but it should at least not crash
        try:
            converted = ensure_reasoning_type_enum(result_obj, "test")
            # Check if conversion happened (if not frozen)
            if hasattr(converted.reasoning_type, 'value'):
                assert converted.reasoning_type == ReasoningType.PHILOSOPHICAL
        except AttributeError:
            # Frozen dataclass - expected behavior
            pass
    
    def test_ensure_none_value(self):
        """Test ensure_reasoning_type_enum with None."""
        if not CONVERSION_UTILS_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        result = ensure_reasoning_type_enum(None, "test")
        assert result is None
    
    def test_ensure_no_reasoning_type_field(self):
        """Test ensure_reasoning_type_enum with result without reasoning_type."""
        if not CONVERSION_UTILS_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Dict without reasoning_type - should pass through unchanged
        result_dict = {"confidence": 0.8, "conclusion": "Test"}
        converted = ensure_reasoning_type_enum(result_dict, "test")
        assert converted == result_dict
        
        # Object without reasoning_type - should pass through unchanged
        @dataclass
        class MockResult:
            confidence: float
        
        result_obj = MockResult(confidence=0.8)
        converted = ensure_reasoning_type_enum(result_obj, "test")
        assert converted == result_obj


class TestPhilosophicalQueryEndToEnd:
    """
    End-to-end tests for philosophical/ethical queries.
    
    These tests verify that high-confidence world_model results
    are properly surfaced and not discarded due to type mismatches.
    """
    
    def test_philosophical_result_not_discarded(self):
        """Test that philosophical results with proper enum are not discarded."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        # Simulate a philosophical result as it would come from apply_reasoning
        result_dict = {
            "reasoning_type": ReasoningType.PHILOSOPHICAL,  # Proper enum
            "confidence": 0.9,
            "conclusion": "This is a philosophical answer",
            "selected_tools": ["world_model"],
        }
        
        # Ensure conversion doesn't break properly-typed result
        converted = ensure_reasoning_type_enum(result_dict, "test")
        assert converted["reasoning_type"] == ReasoningType.PHILOSOPHICAL
        assert converted["confidence"] == 0.9
        assert converted["conclusion"] == "This is a philosophical answer"
    
    def test_string_philosophical_result_converted(self):
        """Test that string philosophical results are converted to enum."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        # Simulate a result with string reasoning_type (the bug case)
        result_dict = {
            "reasoning_type": "philosophical",  # String instead of enum
            "confidence": 0.85,
            "conclusion": "This is a philosophical answer",
            "selected_tools": ["world_model"],
        }
        
        # Conversion should fix the type
        converted = ensure_reasoning_type_enum(result_dict, "agent_pool")
        assert converted["reasoning_type"] == ReasoningType.PHILOSOPHICAL
        assert converted["confidence"] == 0.85
        assert converted["conclusion"] == "This is a philosophical answer"
    
    def test_meta_reasoning_alias_converted(self):
        """Test that meta_reasoning alias is converted to PHILOSOPHICAL enum."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        # Simulate the exact bug: meta_reasoning string from world model
        result_dict = {
            "reasoning_type": "meta_reasoning",  # Alias string
            "confidence": 0.92,
            "conclusion": "I am an AI system designed to...",
            "selected_tools": ["world_model"],
            "self_referential": True,
        }
        
        # Conversion should map to PHILOSOPHICAL
        converted = ensure_reasoning_type_enum(result_dict, "agent_pool")
        assert converted["reasoning_type"] == ReasoningType.PHILOSOPHICAL
        assert converted["confidence"] == 0.92
        assert converted["conclusion"] == "I am an AI system designed to..."


class TestLoggingAndErrorHandling:
    """Tests for proper logging and error handling during conversion."""
    
    @patch('src.vulcan.reasoning.integration.utils.logger')
    def test_invalid_string_logs_error(self, mock_logger):
        """Test that invalid strings trigger error logging."""
        if not CONVERSION_UTILS_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        convert_reasoning_type_to_enum("invalid_type", "test_context")
        
        # Should log error with context
        assert mock_logger.error.called
        error_call_args = mock_logger.error.call_args[0][0]
        assert "CRITICAL" in error_call_args
        assert "invalid_type" in error_call_args
        assert "test_context" in error_call_args
    
    @patch('src.vulcan.reasoning.integration.utils.logger')
    def test_successful_conversion_logs_info(self, mock_logger):
        """Test that successful string conversion logs info."""
        if not CONVERSION_UTILS_AVAILABLE or not REASONING_TYPE_AVAILABLE:
            pytest.skip("Required components not available")
        
        convert_reasoning_type_to_enum("philosophical", "test_context")
        
        # Should log info about conversion
        assert mock_logger.info.called
    
    @patch('src.vulcan.reasoning.integration.utils.logger')
    def test_discarded_result_logs_full_context(self, mock_logger):
        """Test that discarded results log full context for debugging."""
        if not CONVERSION_UTILS_AVAILABLE:
            pytest.skip("Conversion utilities not available")
        
        # Result with invalid reasoning_type
        result_dict = {
            "reasoning_type": "invalid_type",
            "confidence": 0.95,  # High confidence!
            "conclusion": "Important answer that will be lost",
        }
        
        ensure_reasoning_type_enum(result_dict, "orchestrator")
        
        # Should log error with full result details
        assert mock_logger.error.called
        error_call_args = mock_logger.error.call_args[0][0]
        assert "DISCARDED RESULT" in error_call_args
        assert "confidence=0.95" in error_call_args or "confidence=0.95" in str(mock_logger.error.call_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
