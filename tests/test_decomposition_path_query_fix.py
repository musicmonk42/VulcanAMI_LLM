"""
Test for the decomposition path query fix.

This test validates the fix for the bug where step descriptions (~28 chars)
were being passed to ToolSelector instead of the original query (e.g., 507 chars)
during the decomposition path in reasoning_integration.py.

The bug manifested as:
- User sends: "SAT problem: A→B, B→C, ¬C, A∨B" (507 chars)
- [ReasoningIntegration] Using decomposition path (complexity=0.50 >= 0.5)
- [ReasoningIntegration] Decomposed into 4 steps
- [ToolSelector] Found query for semantic matching (length=28 chars) ← WRONG!

The fix ensures that the original query is passed to ToolSelector for tool
selection, not the step descriptions.
"""

import unittest
from unittest.mock import MagicMock, patch


class MockDecompositionStep:
    """Mock decomposition step for testing."""
    
    def __init__(self, step_id, description, complexity=0.3):
        self.step_id = step_id
        self.description = description
        self.estimated_complexity = complexity


class MockDecompositionPlan:
    """Mock decomposition plan for testing."""
    
    def __init__(self, steps, confidence=0.7, strategy=None):
        self.steps = steps
        self.confidence = confidence
        self.strategy = strategy


class MockProblemGraph:
    """Mock problem graph for testing."""
    
    def __init__(self):
        self.nodes = {"root": {"type": "query", "content": "test"}}
        self.complexity_score = 0.6


class TestDecompositionPathQueryFix(unittest.TestCase):
    """Test that the original query is passed to ToolSelector during decomposition."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset global singleton for testing
        try:
            import vulcan.reasoning.reasoning_integration as ri_module
            ri_module._reasoning_integration = None
        except ImportError:
            pass

    def test_decomposition_path_passes_original_query(self):
        """
        Test that _process_with_decomposition passes the original query
        to _select_with_tool_selector, not step descriptions.
        """
        try:
            from vulcan.reasoning.reasoning_integration import ReasoningIntegration
        except ImportError:
            self.skipTest("ReasoningIntegration not available")
            return

        # Create integration with decomposition enabled
        integration = ReasoningIntegration(config={
            'enable_decomposition': True,
            'enable_cross_domain_transfer': False,
        })
        
        # Create mock components
        mock_decomposer = MagicMock()
        mock_query_bridge = MagicMock()
        mock_tool_selector = MagicMock()
        
        # Set up mock decomposition plan with short step descriptions
        steps = [
            MockDecompositionStep("step_0", "Parse logical constraints", 0.3),
            MockDecompositionStep("step_1", "Build dependency graph", 0.3),
            MockDecompositionStep("step_2", "Check satisfiability", 0.4),
            MockDecompositionStep("step_3", "Return solution", 0.2),
        ]
        mock_decomposition_plan = MockDecompositionPlan(steps, confidence=0.7)
        mock_decomposer.decompose_novel_problem.return_value = mock_decomposition_plan
        
        # Set up mock query bridge
        mock_problem_graph = MockProblemGraph()
        mock_query_bridge.convert_to_problem_graph.return_value = mock_problem_graph
        
        # Set up mock tool selector result
        from vulcan.reasoning.reasoning_integration import ReasoningResult
        mock_selection_result = ReasoningResult(
            selected_tools=['symbolic', 'probabilistic'],
            reasoning_strategy='causal_reasoning',
            confidence=0.85,
            rationale='Test selection',
            metadata={},
        )
        
        # Track what query is passed to _select_with_tool_selector
        captured_queries = []
        original_select = integration._select_with_tool_selector
        
        def capture_query(*args, **kwargs):
            query = kwargs.get('query', args[0] if args else None)
            captured_queries.append(query)
            return mock_selection_result
        
        # Inject mock components
        integration._problem_decomposer = mock_decomposer
        integration._query_bridge = mock_query_bridge
        integration._tool_selector = mock_tool_selector
        integration._initialized = True
        
        # Patch _select_with_tool_selector to capture the query
        with patch.object(integration, '_select_with_tool_selector', capture_query):
            # Test with a long original query (507 chars)
            original_query = (
                "SAT problem: Given the logical formula with the following constraints: "
                "A implies B (A→B), B implies C (B→C), NOT C (¬C), and A OR B (A∨B). "
                "Determine whether this propositional logic formula is satisfiable. "
                "If it is satisfiable, provide a satisfying assignment for the boolean "
                "variables A, B, and C. If it is not satisfiable, explain why not and "
                "identify which constraints lead to the contradiction."
            )
            
            # Call _process_with_decomposition
            result = integration._process_with_decomposition(
                query=original_query,
                query_type="reasoning",
                complexity=0.6,
                context={},
            )
            
            # CRITICAL ASSERTION: The original query should be passed to tool selection,
            # not the short step descriptions
            self.assertEqual(len(captured_queries), 1,
                "Tool selection should be called exactly once with the original query")
            
            # The captured query should be the ORIGINAL query, not a step description
            self.assertEqual(captured_queries[0], original_query,
                f"Expected original query ({len(original_query)} chars), "
                f"but got query with {len(captured_queries[0])} chars")
            
            # The length should be 400+ chars, not ~28 chars
            self.assertGreater(len(captured_queries[0]), 100,
                f"Query passed to tool selector should be longer than 100 chars, "
                f"got {len(captured_queries[0])} chars")

    def test_decomposition_returns_tools_from_original_query_selection(self):
        """
        Test that tools selected based on the original query are returned,
        not tools based on step descriptions.
        """
        try:
            from vulcan.reasoning.reasoning_integration import ReasoningIntegration
        except ImportError:
            self.skipTest("ReasoningIntegration not available")
            return

        # Create integration
        integration = ReasoningIntegration(config={
            'enable_decomposition': True,
            'enable_cross_domain_transfer': False,
        })
        
        # Create mock components
        mock_decomposer = MagicMock()
        mock_query_bridge = MagicMock()
        mock_tool_selector = MagicMock()
        
        # Set up mock decomposition plan
        steps = [
            MockDecompositionStep("step_0", "Parse input", 0.3),
            MockDecompositionStep("step_1", "Process data", 0.3),
        ]
        mock_decomposition_plan = MockDecompositionPlan(steps, confidence=0.7)
        mock_decomposer.decompose_novel_problem.return_value = mock_decomposition_plan
        
        # Set up mock query bridge
        mock_query_bridge.convert_to_problem_graph.return_value = MockProblemGraph()
        
        # Inject mock components
        integration._problem_decomposer = mock_decomposer
        integration._query_bridge = mock_query_bridge
        integration._tool_selector = mock_tool_selector
        integration._initialized = True
        
        # Create mock result with specific tools
        from vulcan.reasoning.reasoning_integration import ReasoningResult
        expected_tools = ['symbolic', 'causal', 'probabilistic']
        mock_result = ReasoningResult(
            selected_tools=expected_tools,
            reasoning_strategy='causal_reasoning',
            confidence=0.85,
            rationale='Test',
            metadata={},
        )
        
        with patch.object(integration, '_select_with_tool_selector', return_value=mock_result):
            result = integration._process_with_decomposition(
                query="Complex query that needs symbolic and causal reasoning",
                query_type="reasoning",
                complexity=0.6,
                context={},
            )
            
            # All expected tools should be in the result
            for tool in expected_tools:
                self.assertIn(tool, result.selected_tools,
                    f"Expected tool '{tool}' to be in selected_tools")

    def test_step_metadata_still_recorded(self):
        """
        Test that step metadata is still recorded correctly even though
        tool selection is done only once on the original query.
        """
        try:
            from vulcan.reasoning.reasoning_integration import ReasoningIntegration
        except ImportError:
            self.skipTest("ReasoningIntegration not available")
            return

        integration = ReasoningIntegration(config={
            'enable_decomposition': True,
            'enable_cross_domain_transfer': False,
        })
        
        mock_decomposer = MagicMock()
        mock_query_bridge = MagicMock()
        mock_tool_selector = MagicMock()
        
        # Create steps with specific descriptions
        steps = [
            MockDecompositionStep("step_0", "First step description", 0.3),
            MockDecompositionStep("step_1", "Second step description", 0.4),
            MockDecompositionStep("step_2", "Third step description", 0.5),
        ]
        mock_decomposition_plan = MockDecompositionPlan(steps, confidence=0.7)
        mock_decomposer.decompose_novel_problem.return_value = mock_decomposition_plan
        mock_query_bridge.convert_to_problem_graph.return_value = MockProblemGraph()
        
        integration._problem_decomposer = mock_decomposer
        integration._query_bridge = mock_query_bridge
        integration._tool_selector = mock_tool_selector
        integration._initialized = True
        
        from vulcan.reasoning.reasoning_integration import ReasoningResult
        mock_result = ReasoningResult(
            selected_tools=['symbolic'],
            reasoning_strategy='test',
            confidence=0.8,
            rationale='Test',
            metadata={},
        )
        
        with patch.object(integration, '_select_with_tool_selector', return_value=mock_result):
            result = integration._process_with_decomposition(
                query="Original query text",
                query_type="reasoning",
                complexity=0.6,
                context={},
            )
            
            # Verify decomposition metadata is present
            self.assertTrue(result.metadata.get('decomposition_path'),
                "decomposition_path should be True in metadata")
            self.assertEqual(result.metadata.get('decomposition_steps'), 3,
                "Should record 3 decomposition steps")
            
            # Verify step_results are recorded
            step_results = result.metadata.get('step_results', [])
            self.assertEqual(len(step_results), 3,
                "Should have 3 step results in metadata")
            
            # Each step should have inherited tools from primary selection
            for step in step_results:
                self.assertEqual(step['tools'], ['symbolic'],
                    "Step should inherit tools from primary selection")


class TestDecompositionLogging(unittest.TestCase):
    """Test that appropriate logging is done for decomposition path."""

    def test_log_message_includes_original_query_length(self):
        """
        Test that the log message includes the original query length,
        helping to diagnose if the wrong query is being used.
        """
        try:
            from vulcan.reasoning.reasoning_integration import ReasoningIntegration
            import logging
        except ImportError:
            self.skipTest("ReasoningIntegration not available")
            return

        integration = ReasoningIntegration(config={
            'enable_decomposition': True,
            'enable_cross_domain_transfer': False,
        })
        
        mock_decomposer = MagicMock()
        mock_query_bridge = MagicMock()
        mock_tool_selector = MagicMock()
        
        steps = [MockDecompositionStep("step_0", "Short desc", 0.3)]
        mock_decomposition_plan = MockDecompositionPlan(steps, confidence=0.7)
        mock_decomposer.decompose_novel_problem.return_value = mock_decomposition_plan
        mock_query_bridge.convert_to_problem_graph.return_value = MockProblemGraph()
        
        integration._problem_decomposer = mock_decomposer
        integration._query_bridge = mock_query_bridge
        integration._tool_selector = mock_tool_selector
        integration._initialized = True
        
        from vulcan.reasoning.reasoning_integration import ReasoningResult
        mock_result = ReasoningResult(
            selected_tools=['symbolic'],
            reasoning_strategy='test',
            confidence=0.8,
            rationale='Test',
            metadata={},
        )
        
        # Capture log output
        with patch.object(integration, '_select_with_tool_selector', return_value=mock_result):
            with self.assertLogs('vulcan.reasoning.reasoning_integration', level='INFO') as cm:
                integration._process_with_decomposition(
                    query="A" * 500,  # 500 character query
                    query_type="reasoning",
                    complexity=0.6,
                    context={},
                )
                
                # Check that the log includes the query length
                log_output = '\n'.join(cm.output)
                self.assertIn('length=500', log_output,
                    "Log should include original query length")


if __name__ == '__main__':
    unittest.main(verbosity=2)
