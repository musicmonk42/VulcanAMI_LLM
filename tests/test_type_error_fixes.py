"""
Test Type Error Fixes

This test validates the fixes for two production errors:
1. unified_chat.py: AttributeError when direct_conclusion is dict instead of string
2. apply_reasoning_impl.py: Missing _is_self_referential attribute in ReasoningIntegration

Both fixes follow industry best practices with defensive programming,
proper type handling, and clear documentation.
"""

import unittest
from typing import Any, Dict, Optional


class TestUnifiedChatConclusionNormalization(unittest.TestCase):
    """Test the conclusion normalization helper function in unified_chat."""
    
    def test_normalize_conclusion_string(self):
        """Test that string conclusions are returned as-is."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        result = _normalize_conclusion_to_string("This is a conclusion")
        self.assertEqual(result, "This is a conclusion")
        self.assertIsInstance(result, str)
    
    def test_normalize_conclusion_none(self):
        """Test that None returns None."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        result = _normalize_conclusion_to_string(None)
        self.assertIsNone(result)
    
    def test_normalize_conclusion_dict_with_conclusion_key(self):
        """Test that dict with 'conclusion' key extracts the value."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        conclusion_dict = {"conclusion": "The answer is 42"}
        result = _normalize_conclusion_to_string(conclusion_dict)
        self.assertEqual(result, "The answer is 42")
        self.assertIsInstance(result, str)
    
    def test_normalize_conclusion_dict_with_result_key(self):
        """Test that dict with 'result' key extracts the value."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        conclusion_dict = {"result": "Success"}
        result = _normalize_conclusion_to_string(conclusion_dict)
        self.assertEqual(result, "Success")
        self.assertIsInstance(result, str)
    
    def test_normalize_conclusion_dict_with_response_key(self):
        """Test that dict with 'response' key extracts the value."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        conclusion_dict = {"response": "Here is the answer"}
        result = _normalize_conclusion_to_string(conclusion_dict)
        self.assertEqual(result, "Here is the answer")
        self.assertIsInstance(result, str)
    
    def test_normalize_conclusion_dict_nested(self):
        """Test that nested dicts are handled correctly."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        conclusion_dict = {
            "metadata": {"status": "ok"},
            "conclusion": "Final answer"
        }
        result = _normalize_conclusion_to_string(conclusion_dict)
        self.assertEqual(result, "Final answer")
    
    def test_normalize_conclusion_dict_no_standard_keys(self):
        """Test that dict without standard keys converts to string."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        conclusion_dict = {"custom_key": "custom_value"}
        result = _normalize_conclusion_to_string(conclusion_dict)
        # Should convert entire dict to string
        self.assertIsInstance(result, str)
        self.assertIn("custom_key", result)
        self.assertIn("custom_value", result)
    
    def test_normalize_conclusion_other_types(self):
        """Test that other types (int, float, bool) convert to string."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        # Integer
        result = _normalize_conclusion_to_string(42)
        self.assertEqual(result, "42")
        self.assertIsInstance(result, str)
        
        # Float
        result = _normalize_conclusion_to_string(3.14)
        self.assertEqual(result, "3.14")
        self.assertIsInstance(result, str)
        
        # Boolean
        result = _normalize_conclusion_to_string(True)
        self.assertEqual(result, "True")
        self.assertIsInstance(result, str)
    
    def test_normalize_conclusion_empty_string(self):
        """Test that empty string is preserved."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        result = _normalize_conclusion_to_string("")
        self.assertEqual(result, "")
        self.assertIsInstance(result, str)
    
    def test_normalize_conclusion_whitespace_string(self):
        """Test that whitespace-only string is preserved."""
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        result = _normalize_conclusion_to_string("   ")
        self.assertEqual(result, "   ")
        self.assertIsInstance(result, str)


class TestReasoningIntegrationWrapperMethods(unittest.TestCase):
    """Test the wrapper methods in ReasoningIntegration class."""
    
    def test_is_self_referential_method_exists(self):
        """Test that _is_self_referential method exists on ReasoningIntegration."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        self.assertTrue(hasattr(integration, '_is_self_referential'))
        self.assertTrue(callable(getattr(integration, '_is_self_referential')))
    
    def test_is_ethical_query_method_exists(self):
        """Test that _is_ethical_query method exists on ReasoningIntegration."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        self.assertTrue(hasattr(integration, '_is_ethical_query'))
        self.assertTrue(callable(getattr(integration, '_is_ethical_query')))
    
    def test_consult_world_model_introspection_method_exists(self):
        """Test that _consult_world_model_introspection method exists."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        self.assertTrue(hasattr(integration, '_consult_world_model_introspection'))
        self.assertTrue(callable(getattr(integration, '_consult_world_model_introspection')))
    
    def test_is_self_referential_detects_self_ref_queries(self):
        """Test that _is_self_referential correctly identifies self-referential queries."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        
        # Self-referential queries
        self.assertTrue(integration._is_self_referential("What can you do?"))
        self.assertTrue(integration._is_self_referential("What are your capabilities?"))
        self.assertTrue(integration._is_self_referential("How do you work?"))
        
        # Non-self-referential queries
        self.assertFalse(integration._is_self_referential("What is photosynthesis?"))
        self.assertFalse(integration._is_self_referential("Explain quantum mechanics"))
    
    def test_is_ethical_query_detects_ethical_queries(self):
        """Test that _is_ethical_query correctly identifies ethical queries."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        
        # Ethical queries
        self.assertTrue(integration._is_ethical_query("Is it right to lie?"))
        self.assertTrue(integration._is_ethical_query("Should I help someone?"))
        
        # Non-ethical queries
        self.assertFalse(integration._is_ethical_query("What is the capital of France?"))
        self.assertFalse(integration._is_ethical_query("Calculate 2+2"))
    
    def test_consult_world_model_returns_dict_or_none(self):
        """Test that _consult_world_model_introspection returns dict or None."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        
        # Capability query should return dict
        result = integration._consult_world_model_introspection("What can you do?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('response', result)
        self.assertIn('confidence', result)
        
        # Non-introspective query should return None
        result = integration._consult_world_model_introspection("What is 2+2?")
        self.assertIsNone(result)
    
    def test_wrapper_methods_delegate_correctly(self):
        """Test that wrapper methods delegate to standalone functions."""
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        from src.vulcan.reasoning.integration.query_analysis import (
            is_self_referential,
            is_ethical_query,
            consult_world_model_introspection,
        )
        
        integration = ReasoningIntegration()
        query = "What can you do?"
        
        # Results should match standalone function results
        self.assertEqual(
            integration._is_self_referential(query),
            is_self_referential(query)
        )
        
        self.assertEqual(
            integration._is_ethical_query(query),
            is_ethical_query(query)
        )
        
        self.assertEqual(
            integration._consult_world_model_introspection(query),
            consult_world_model_introspection(query)
        )


class TestIntegrationErrorScenarios(unittest.TestCase):
    """Integration tests for the error scenarios from production."""
    
    def test_direct_conclusion_dict_no_longer_crashes(self):
        """
        Test that the original error scenario no longer causes AttributeError.
        
        Original error:
        AttributeError: 'dict' object has no attribute 'strip'
        at line: if direct_conclusion is not None and direct_conclusion.strip():
        """
        from src.vulcan.endpoints.unified_chat import _normalize_conclusion_to_string
        
        # Simulate the problematic scenario from production
        direct_conclusion = {
            "conclusion": "This is the answer",
            "metadata": {"tool": "world_model"}
        }
        
        # This should not raise AttributeError
        normalized = _normalize_conclusion_to_string(direct_conclusion)
        
        # Verify we can now safely call string methods
        self.assertIsInstance(normalized, str)
        self.assertTrue(normalized.strip())  # Should not raise AttributeError
        self.assertEqual(normalized, "This is the answer")
    
    def test_apply_reasoning_has_required_methods(self):
        """
        Test that apply_reasoning can call the required methods.
        
        Original error:
        AttributeError: 'ReasoningIntegration' object has no attribute '_is_self_referential'
        at line: is_self_ref = self._is_self_referential(query)
        """
        from src.vulcan.reasoning.integration.orchestrator import ReasoningIntegration
        
        integration = ReasoningIntegration()
        
        # Verify all required methods exist and can be called
        query = "What can you do?"
        
        # These should not raise AttributeError
        is_self_ref = integration._is_self_referential(query)
        is_ethical = integration._is_ethical_query(query)
        wm_result = integration._consult_world_model_introspection(query)
        
        # Verify return types
        self.assertIsInstance(is_self_ref, bool)
        self.assertIsInstance(is_ethical, bool)
        self.assertTrue(wm_result is None or isinstance(wm_result, dict))


if __name__ == '__main__':
    unittest.main()
