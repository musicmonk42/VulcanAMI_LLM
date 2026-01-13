#!/usr/bin/env python3
"""
Test for SelectionRequest parameter fix.

Validates that SelectionRequest is instantiated with correct parameter names:
- problem (not query)
- context (not query_type)

This test ensures the fix for the reasoning system parameter mismatch bug.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from vulcan.reasoning.integration.orchestrator import ReasoningIntegration


class TestSelectionRequestParameters:
    """Test suite for SelectionRequest parameter fix."""

    @patch('vulcan.reasoning.integration.selection_strategies.SelectionRequest')
    def test_selection_request_uses_correct_parameters(self, mock_selection_request):
        """Verify SelectionRequest is called with 'problem' and 'context' parameters."""
        # Setup
        integration = ReasoningIntegration()
        
        # Mock the tool selector
        mock_selector = MagicMock()
        mock_selection_result = MagicMock()
        mock_selection_result.selected_tools = ['mathematical']
        mock_selection_result.strategy = 'direct'
        mock_selection_result.confidence = 0.8
        mock_selection_result.rationale = 'Test selection'
        mock_selector.select_and_execute.return_value = mock_selection_result
        integration._tool_selector = mock_selector
        
        # Mock the SelectionRequest instance
        mock_request_instance = MagicMock()
        mock_selection_request.return_value = mock_request_instance
        
        # Call the method
        result = integration._select_with_tool_selector(
            query="Compute sum from 1 to n",
            query_type="mathematical",
            complexity=0.6,
            context=None
        )
        
        # Verify SelectionRequest was called with correct parameters
        mock_selection_request.assert_called_once()
        call_kwargs = mock_selection_request.call_args[1]
        
        # Assert correct parameter names are used
        assert 'problem' in call_kwargs, "SelectionRequest must use 'problem' parameter"
        assert 'query' not in call_kwargs, "SelectionRequest should NOT use 'query' parameter"
        
        # Assert context contains query_type
        assert 'context' in call_kwargs, "SelectionRequest must have 'context' parameter"
        assert isinstance(call_kwargs['context'], dict), "context must be a dict"
        assert 'query_type' in call_kwargs['context'], "context must contain 'query_type'"
        assert call_kwargs['context']['query_type'] == 'mathematical', \
            "context['query_type'] must match the query_type argument"
        
        # Verify the query is passed as problem
        assert call_kwargs['problem'] == "Compute sum from 1 to n", \
            "problem parameter must contain the query text"

    @patch('vulcan.reasoning.integration.selection_strategies.SelectionRequest')
    def test_selection_request_includes_mode_and_constraints(self, mock_selection_request):
        """Verify SelectionRequest includes mode and constraints parameters."""
        # Setup
        integration = ReasoningIntegration()
        
        # Mock the tool selector
        mock_selector = MagicMock()
        mock_selection_result = MagicMock()
        mock_selection_result.selected_tools = ['probabilistic']
        mock_selection_result.strategy = 'direct'
        mock_selection_result.confidence = 0.7
        mock_selection_result.rationale = 'Test selection'
        mock_selector.select_and_execute.return_value = mock_selection_result
        integration._tool_selector = mock_selector
        
        # Mock the SelectionRequest instance
        mock_request_instance = MagicMock()
        mock_selection_request.return_value = mock_request_instance
        
        # Call with constraints in context
        context_with_constraints = {
            'constraints': {'time_budget_ms': 5000}
        }
        
        result = integration._select_with_tool_selector(
            query="What is the probability?",
            query_type="probabilistic",
            complexity=0.8,
            context=context_with_constraints
        )
        
        # Verify SelectionRequest was called with mode and constraints
        call_kwargs = mock_selection_request.call_args[1]
        
        assert 'mode' in call_kwargs, "SelectionRequest must include 'mode' parameter"
        assert 'constraints' in call_kwargs, "SelectionRequest must include 'constraints' parameter"
        assert call_kwargs['constraints'] == {'time_budget_ms': 5000}, \
            "constraints must be passed from context"

    def test_selection_request_without_tool_selector(self):
        """Verify fallback behavior when tool selector is not available."""
        # Setup
        integration = ReasoningIntegration()
        integration._tool_selector = None
        
        # Call the method
        result = integration._select_with_tool_selector(
            query="Test query",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Verify it returns a ReasoningResult with fallback values
        from vulcan.reasoning.integration.types import ReasoningResult
        assert isinstance(result, ReasoningResult)
        assert len(result.selected_tools) > 0
        assert result.confidence >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
