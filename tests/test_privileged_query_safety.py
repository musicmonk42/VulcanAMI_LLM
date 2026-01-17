"""
Test Privileged Query Safety Standard - No Fallback for Introspective/Ethical/Philosophical Queries

This test suite validates that introspective, ethical, and philosophical queries:
1. Are properly detected by the system
2. Are routed to world_model/meta-reasoning
3. Return explicit no-answer results when world_model fails
4. NEVER fall through to classifier/general tool selection

Industry Standard AGI Safety Compliance Test Suite.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestPhilosophicalQueryDetection(unittest.TestCase):
    """Test detection of philosophical queries."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from vulcan.reasoning.integration.query_analysis import is_philosophical_query
            self.is_philosophical = is_philosophical_query
        except ImportError:
            self.skipTest("is_philosophical_query not available")
    
    def test_consciousness_query_detected(self):
        """Test that consciousness queries are detected as philosophical."""
        queries = [
            "What is consciousness?",
            "What is the nature of consciousness?",
            "Can you explain consciousness?",
            "Tell me about conscious awareness",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(
                    self.is_philosophical(query),
                    f"Failed to detect philosophical query: {query}"
                )
    
    def test_free_will_query_detected(self):
        """Test that free will queries are detected as philosophical."""
        queries = [
            "Do we have free will?",
            "Is free will an illusion?",
            "What is free will?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(
                    self.is_philosophical(query),
                    f"Failed to detect philosophical query: {query}"
                )
    
    def test_mind_body_problem_detected(self):
        """Test that mind-body problem queries are detected."""
        queries = [
            "What is the mind-body problem?",
            "How does the mind relate to the body?",
            "Explain the mind body connection",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(
                    self.is_philosophical(query),
                    f"Failed to detect philosophical query: {query}"
                )
    
    def test_existence_queries_detected(self):
        """Test that existential queries are detected as philosophical."""
        queries = [
            "What is the meaning of life?",
            "What is existence?",
            "What is the nature of reality?",
            "What is truth?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(
                    self.is_philosophical(query),
                    f"Failed to detect philosophical query: {query}"
                )
    
    def test_non_philosophical_queries_not_detected(self):
        """Test that non-philosophical queries are not falsely detected."""
        queries = [
            "What is the capital of France?",
            "How do I cook pasta?",
            "Calculate 2 + 2",
            "What is the weather today?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertFalse(
                    self.is_philosophical(query),
                    f"Falsely detected non-philosophical query as philosophical: {query}"
                )


class TestPrivilegedQueryNoAnswerPath(unittest.TestCase):
    """Test that privileged queries return explicit no-answer when world_model fails."""
    
    def setUp(self):
        """Set up test fixtures - skip if integration module not available."""
        try:
            from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        except ImportError:
            self.skipTest("ReasoningIntegration not available")
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_self_referential_no_answer_when_world_model_returns_none(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref
    ):
        """Test self-referential query returns no-answer when world_model returns None."""
        # Set up detection mocks
        mock_is_self_ref.return_value = True
        mock_is_ethical.return_value = False
        mock_is_phil.return_value = False
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        from vulcan.reasoning.integration.types import ReasoningStrategyType
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock _consult_world_model_introspection to return None
        integration._consult_world_model_introspection = Mock(return_value=None)
        
        # Call apply_reasoning with self-referential query
        result = integration.apply_reasoning(
            query="What are you capable of?",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Verify result is privileged no-answer
        self.assertEqual(result.selected_tools, ["world_model"])
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.override_router_tools)
        self.assertIn("world_model", result.selected_tools)
        self.assertIn("no fallback", result.rationale.lower())
        
        # Verify metadata flags
        self.assertTrue(result.metadata.get("privileged_no_answer"))
        self.assertTrue(result.metadata.get("no_classifier_fallback"))
        self.assertEqual(
            result.metadata.get("safety_standard_applied"),
            "privileged_query_no_fallback"
        )
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_ethical_query_no_answer_when_world_model_low_confidence(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref
    ):
        """Test ethical query returns no-answer when world_model has low confidence."""
        # Set up detection mocks
        mock_is_self_ref.return_value = False
        mock_is_ethical.return_value = True
        mock_is_phil.return_value = False
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock _consult_world_model_introspection to return low confidence
        integration._consult_world_model_introspection = Mock(return_value={
            "confidence": 0.3,  # Below 0.5 threshold
            "response": "Uncertain",
            "reasoning": "Not confident in this answer"
        })
        
        # Call apply_reasoning with ethical query
        result = integration.apply_reasoning(
            query="Is it right to lie to protect someone?",
            query_type="ethical",
            complexity=0.6,
            context=None
        )
        
        # Verify result is privileged no-answer
        self.assertEqual(result.selected_tools, ["world_model"])
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.override_router_tools)
        
        # Verify metadata includes world_model failure reason
        self.assertTrue(result.metadata.get("privileged_no_answer"))
        self.assertIn("confidence", result.metadata.get("world_model_failure_reason", "").lower())
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_philosophical_query_no_answer_path(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref
    ):
        """Test philosophical query returns no-answer when world_model fails."""
        # Set up detection mocks
        mock_is_self_ref.return_value = False
        mock_is_ethical.return_value = False
        mock_is_phil.return_value = True
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock _consult_world_model_introspection to return None
        integration._consult_world_model_introspection = Mock(return_value=None)
        
        # Call apply_reasoning with philosophical query
        result = integration.apply_reasoning(
            query="What is consciousness?",
            query_type="philosophical",
            complexity=0.7,
            context=None
        )
        
        # Verify result is privileged no-answer
        self.assertEqual(result.selected_tools, ["world_model"])
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.override_router_tools)
        self.assertTrue(result.metadata.get("privileged_no_answer"))
        self.assertTrue(result.metadata.get("no_classifier_fallback"))


class TestPrivilegedQuerySuccessPath(unittest.TestCase):
    """Test that privileged queries work correctly when world_model succeeds."""
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_world_model_high_confidence_returns_answer(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref
    ):
        """Test that high confidence world_model result is returned."""
        # Set up detection mocks
        mock_is_self_ref.return_value = True
        mock_is_ethical.return_value = False
        mock_is_phil.return_value = False
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock _consult_world_model_introspection to return high confidence
        integration._consult_world_model_introspection = Mock(return_value={
            "confidence": 0.85,  # Above 0.5 threshold
            "response": "I am an AI system designed to help with reasoning and analysis.",
            "reasoning": "Self-introspection query answered with high confidence"
        })
        
        # Call apply_reasoning
        result = integration.apply_reasoning(
            query="What are you?",
            query_type="self_introspection",
            complexity=0.5,
            context=None
        )
        
        # Verify result is privileged with answer
        self.assertEqual(result.selected_tools, ["world_model"])
        self.assertEqual(result.confidence, 0.85)
        self.assertTrue(result.override_router_tools)
        self.assertIn("AI system", result.metadata.get("world_model_response", ""))
        self.assertFalse(result.metadata.get("privileged_no_answer", False))


class TestPrivilegedQueryDelegation(unittest.TestCase):
    """Test that world_model delegation is preserved."""
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_world_model_delegation_preserved(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref
    ):
        """Test that world_model delegation to other tools is preserved."""
        # Set up detection mocks
        mock_is_self_ref.return_value = True
        mock_is_ethical.return_value = False
        mock_is_phil.return_value = False
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock _consult_world_model_introspection to return delegation
        integration._consult_world_model_introspection = Mock(return_value={
            "confidence": 0.8,
            "needs_delegation": True,
            "recommended_tool": "mathematical",
            "delegation_reason": "This is actually a mathematical query"
        })
        
        # Mock _select_with_tool_selector
        mock_result = Mock()
        mock_result.selected_tools = ["mathematical"]
        mock_result.confidence = 0.9
        mock_result.metadata = {}
        integration._select_with_tool_selector = Mock(return_value=mock_result)
        
        # Call apply_reasoning
        result = integration.apply_reasoning(
            query="What is 2 + 2?",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Verify delegation happened
        self.assertEqual(result.selected_tools, ["mathematical"])
        self.assertTrue(result.metadata.get("world_model_delegation"))
        self.assertEqual(result.metadata.get("delegated_tool"), "mathematical")


class TestDefenseInDepthChecks(unittest.TestCase):
    """Test that defense-in-depth checks catch privileged queries at all checkpoints."""
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_classifier_skip_defense_catches_philosophical(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref
    ):
        """Test that classifier skip defense catches philosophical queries."""
        # Set up detection mocks
        mock_is_self_ref.return_value = False
        mock_is_ethical.return_value = False
        mock_is_phil.return_value = True
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock world_model to return low confidence (triggering no-answer path)
        integration._consult_world_model_introspection = Mock(return_value={
            "confidence": 0.2,
            "response": "Uncertain"
        })
        
        # Call apply_reasoning with philosophical query
        result = integration.apply_reasoning(
            query="What is the nature of truth?",
            query_type="general",
            complexity=0.4,
            context=None
        )
        
        # Verify privileged no-answer path was taken
        self.assertEqual(result.selected_tools, ["world_model"])
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.metadata.get("privileged_no_answer"))


class TestAuditLogging(unittest.TestCase):
    """Test that audit logging captures privileged query handling."""
    
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.logger')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_self_referential')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_ethical_query')
    @patch('vulcan.reasoning.integration.apply_reasoning_impl.is_philosophical_query')
    def test_no_answer_path_logs_audit_trail(
        self, mock_is_phil, mock_is_ethical, mock_is_self_ref, mock_logger
    ):
        """Test that no-answer path generates audit logs."""
        # Set up detection mocks
        mock_is_self_ref.return_value = True
        mock_is_ethical.return_value = False
        mock_is_phil.return_value = False
        
        # Import after patching
        from vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        # Create integration instance
        integration = ReasoningIntegration()
        
        # Mock world_model to return None
        integration._consult_world_model_introspection = Mock(return_value=None)
        
        # Call apply_reasoning
        result = integration.apply_reasoning(
            query="What is your purpose?",
            query_type="general",
            complexity=0.5,
            context=None
        )
        
        # Verify audit logging was called
        # Check for warning log about privileged no-answer path
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "PRIVILEGED QUERY NO-ANSWER PATH" in str(call)]
        self.assertGreater(len(warning_calls), 0, "Expected warning log for privileged no-answer path")
        
        # Check for info log about audit
        info_calls = [call for call in mock_logger.info.call_args_list 
                     if "AUDIT" in str(call)]
        self.assertGreater(len(info_calls), 0, "Expected audit log entry")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
