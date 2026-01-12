#!/usr/bin/env python3
"""
Test for infinite recursion fix in tool selection.

This test validates that the infinite recursion bug between
orchestrator._select_with_tool_selector() and 
selection_strategies.select_with_tool_selector() has been fixed.

Bug: Lines 348-352 in selection_strategies.py created a circular delegation
Fix: Removed the delegation check so the standalone function contains the implementation
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, '/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src')

from vulcan.reasoning.integration.selection_strategies import select_with_tool_selector
from vulcan.reasoning.integration.types import ReasoningResult


class TestInfiniteRecursionFix(unittest.TestCase):
    """Test suite for the infinite recursion fix."""

    def test_no_infinite_recursion_when_called_from_orchestrator(self):
        """
        Verify that calling select_with_tool_selector with an orchestrator
        that has _select_with_tool_selector method does NOT cause infinite recursion.
        
        This is the core test for the bug fix.
        """
        # Mock the SelectionMode import that happens inside the function
        import sys
        from types import SimpleNamespace
        
        # Create a mock module for the import
        mock_tool_selector_module = SimpleNamespace()
        mock_tool_selector_module.SelectionMode = SimpleNamespace(
            BALANCED='BALANCED',
            ACCURATE='ACCURATE',
            FAST='FAST'
        )
        sys.modules['vulcan.reasoning.selection.tool_selector'] = mock_tool_selector_module
        
        try:
            # Create a mock orchestrator with the _select_with_tool_selector method
            mock_orchestrator = Mock()
            mock_orchestrator._select_with_tool_selector = Mock()
            
            # Create a mock tool selector that returns a proper result
            mock_tool_selector = Mock()
            mock_selection = Mock()
            mock_selection.selected_tools = ['test_tool']
            mock_selection.strategy = 'test_strategy'
            mock_selection.confidence = 0.8
            mock_selection.rationale = 'Test selection'
            mock_tool_selector.select_tools = Mock(return_value=mock_selection)
            
            mock_orchestrator._tool_selector = mock_tool_selector
            
            # Call the function - this should NOT recurse into orchestrator._select_with_tool_selector
            result = select_with_tool_selector(
                orchestrator=mock_orchestrator,
                query="test query",
                query_type="general",
                complexity=0.5,
                context=None
            )
            
            # Verify the result is correct
            self.assertIsInstance(result, ReasoningResult)
            self.assertEqual(result.selected_tools, ['test_tool'])
            
            # CRITICAL: Verify that orchestrator._select_with_tool_selector was NOT called
            # This proves we didn't delegate back to the orchestrator (which would cause recursion)
            mock_orchestrator._select_with_tool_selector.assert_not_called()
            
            # Verify we DID call the tool selector directly
            mock_tool_selector.select_tools.assert_called_once()
        finally:
            # Clean up the mock module
            if 'vulcan.reasoning.selection.tool_selector' in sys.modules:
                del sys.modules['vulcan.reasoning.selection.tool_selector']

    def test_directly_uses_tool_selector_implementation(self):
        """
        Verify that select_with_tool_selector directly uses orchestrator._tool_selector
        instead of delegating to orchestrator._select_with_tool_selector.
        """
        # Mock the SelectionMode import that happens inside the function
        import sys
        from types import SimpleNamespace
        
        # Create a mock module for the import
        mock_tool_selector_module = SimpleNamespace()
        mock_tool_selector_module.SelectionMode = SimpleNamespace(
            BALANCED='BALANCED',
            ACCURATE='ACCURATE',
            FAST='FAST'
        )
        sys.modules['vulcan.reasoning.selection.tool_selector'] = mock_tool_selector_module
        
        try:
            # Create mock orchestrator with both methods
            mock_orchestrator = Mock()
            mock_orchestrator._select_with_tool_selector = Mock(
                side_effect=Exception("Should not be called - would cause recursion!")
            )
            
            # Set up tool selector to work properly
            mock_tool_selector = Mock()
            mock_selection = Mock()
            mock_selection.selected_tools = ['direct_tool']
            mock_selection.strategy = 'direct_strategy'
            mock_selection.confidence = 0.9
            mock_selection.rationale = 'Direct selection'
            mock_tool_selector.select_tools = Mock(return_value=mock_selection)
            
            mock_orchestrator._tool_selector = mock_tool_selector
            
            # This should succeed without calling _select_with_tool_selector
            result = select_with_tool_selector(
                orchestrator=mock_orchestrator,
                query="direct test",
                query_type="reasoning",
                complexity=0.7,
                context=None
            )
            
            # Verify success
            self.assertIsInstance(result, ReasoningResult)
            self.assertEqual(result.selected_tools, ['direct_tool'])
            
            # Verify we used the tool selector directly, not the orchestrator method
            mock_tool_selector.select_tools.assert_called_once()
            mock_orchestrator._select_with_tool_selector.assert_not_called()
        finally:
            # Clean up the mock module
            if 'vulcan.reasoning.selection.tool_selector' in sys.modules:
                del sys.modules['vulcan.reasoning.selection.tool_selector']

    def test_fallback_when_tool_selector_unavailable(self):
        """
        Verify fallback behavior when tool selector is None.
        """
        # Create orchestrator without tool selector
        mock_orchestrator = Mock()
        mock_orchestrator._tool_selector = None
        mock_orchestrator._select_with_tool_selector = Mock()
        
        # Call the function
        result = select_with_tool_selector(
            orchestrator=mock_orchestrator,
            query="fallback test",
            query_type="general",
            complexity=0.4,
            context=None
        )
        
        # Should return a result using fallback logic
        self.assertIsInstance(result, ReasoningResult)
        self.assertIsNotNone(result.selected_tools)
        self.assertTrue(len(result.selected_tools) > 0)
        
        # Should NOT have called the orchestrator method (no recursion risk)
        mock_orchestrator._select_with_tool_selector.assert_not_called()

    def test_handles_tool_selector_exception(self):
        """
        Verify graceful degradation when tool selector raises an exception.
        """
        # Create orchestrator with failing tool selector
        mock_orchestrator = Mock()
        mock_orchestrator._select_with_tool_selector = Mock()
        
        mock_tool_selector = Mock()
        mock_tool_selector.select_tools = Mock(side_effect=RuntimeError("Tool selector failed"))
        mock_orchestrator._tool_selector = mock_tool_selector
        
        # Should handle the exception and return fallback result
        result = select_with_tool_selector(
            orchestrator=mock_orchestrator,
            query="error test",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Should return a fallback result
        self.assertIsInstance(result, ReasoningResult)
        self.assertIsNotNone(result.selected_tools)
        
        # Should NOT have recursed into orchestrator method
        mock_orchestrator._select_with_tool_selector.assert_not_called()


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
