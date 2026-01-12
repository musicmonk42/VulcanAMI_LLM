#!/usr/bin/env python3
"""
Test for missing methods in ReasoningIntegration that were causing 500 errors.

This test validates the fix for two critical bugs:
1. Missing _select_with_tool_selector method
2. Missing _record_selection_time method

These bugs were causing AttributeError exceptions that propagated to users
as 500 Internal Server Errors.
"""

import pytest
from unittest.mock import MagicMock, patch
from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
from vulcan.reasoning.integration.types import ReasoningResult


class TestReasoningIntegrationMissingMethods:
    """Test suite for missing ReasoningIntegration methods."""

    def test_select_with_tool_selector_method_exists(self):
        """Verify _select_with_tool_selector method exists and is callable."""
        integration = ReasoningIntegration()
        
        # Check method exists
        assert hasattr(integration, '_select_with_tool_selector'), \
            "ReasoningIntegration must have _select_with_tool_selector method"
        
        # Check it's callable
        assert callable(integration._select_with_tool_selector), \
            "_select_with_tool_selector must be callable"

    def test_record_selection_time_method_exists(self):
        """Verify _record_selection_time method exists and is callable."""
        integration = ReasoningIntegration()
        
        # Check method exists
        assert hasattr(integration, '_record_selection_time'), \
            "ReasoningIntegration must have _record_selection_time method"
        
        # Check it's callable
        assert callable(integration._record_selection_time), \
            "_record_selection_time must be callable"

    def test_select_with_tool_selector_returns_reasoning_result(self):
        """Verify _select_with_tool_selector returns a ReasoningResult."""
        integration = ReasoningIntegration()
        
        # Call the method with minimal args
        result = integration._select_with_tool_selector(
            query="test query",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Verify return type
        assert isinstance(result, ReasoningResult), \
            "_select_with_tool_selector must return ReasoningResult"
        
        # Verify result has required attributes
        assert hasattr(result, 'selected_tools'), "Result must have selected_tools"
        assert hasattr(result, 'reasoning_strategy'), "Result must have reasoning_strategy"
        assert hasattr(result, 'confidence'), "Result must have confidence"
        assert hasattr(result, 'rationale'), "Result must have rationale"
        assert hasattr(result, 'metadata'), "Result must have metadata"
        
        # Verify selected_tools is not empty
        assert len(result.selected_tools) > 0, \
            "selected_tools must contain at least one tool"

    def test_select_with_tool_selector_with_context(self):
        """Verify _select_with_tool_selector handles context properly."""
        integration = ReasoningIntegration()
        
        context = {
            "conversation_id": "test_123",
            "constraints": {"max_time_ms": 1000}
        }
        
        result = integration._select_with_tool_selector(
            query="What causes earthquakes?",
            query_type="reasoning",
            complexity=0.7,
            context=context
        )
        
        assert isinstance(result, ReasoningResult)
        assert result.confidence >= 0.0 and result.confidence <= 1.0

    def test_record_selection_time_updates_statistics(self):
        """Verify _record_selection_time updates statistics correctly."""
        integration = ReasoningIntegration()
        
        # Record some selection times
        times = [10.5, 15.3, 20.1, 12.8, 18.5]
        for time_ms in times:
            integration._record_selection_time(time_ms)
        
        # Get statistics
        stats = integration.get_statistics()
        
        # Verify avg_selection_time_ms is calculated
        assert 'avg_selection_time_ms' in stats, \
            "Statistics must include avg_selection_time_ms"
        
        # Verify average is reasonable (should be close to mean of recorded times)
        expected_avg = sum(times) / len(times)
        assert abs(stats['avg_selection_time_ms'] - expected_avg) < 0.1, \
            f"Average should be ~{expected_avg}, got {stats['avg_selection_time_ms']}"

    def test_record_selection_time_handles_single_value(self):
        """Verify _record_selection_time works with single value."""
        integration = ReasoningIntegration()
        
        integration._record_selection_time(42.5)
        
        stats = integration.get_statistics()
        assert stats['avg_selection_time_ms'] == 42.5

    def test_record_selection_time_thread_safe(self):
        """Verify _record_selection_time is thread-safe with locking."""
        integration = ReasoningIntegration()
        
        # Record multiple times rapidly (simulating concurrent access)
        for i in range(100):
            integration._record_selection_time(float(i))
        
        stats = integration.get_statistics()
        # Should have recorded all 100 values
        assert stats['avg_selection_time_ms'] >= 0

    @patch('vulcan.reasoning.integration.selection_strategies.select_with_tool_selector')
    def test_select_with_tool_selector_delegates_correctly(self, mock_select):
        """Verify _select_with_tool_selector delegates to selection_strategies."""
        integration = ReasoningIntegration()
        
        # Mock the return value
        mock_result = ReasoningResult(
            selected_tools=['test_tool'],
            reasoning_strategy='test_strategy',
            confidence=0.8,
            rationale='Test rationale',
            metadata={}
        )
        mock_select.return_value = mock_result
        
        # Call the method
        result = integration._select_with_tool_selector(
            query="test",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Verify delegation happened
        mock_select.assert_called_once()
        
        # Verify correct arguments were passed
        call_args = mock_select.call_args
        assert call_args.kwargs['orchestrator'] == integration
        assert call_args.kwargs['query'] == "test"
        assert call_args.kwargs['query_type'] == "general"
        assert call_args.kwargs['complexity'] == 0.5
        assert call_args.kwargs['context'] is None
        
        # Verify result is returned
        assert result == mock_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
