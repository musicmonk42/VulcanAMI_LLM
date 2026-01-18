"""
Unit tests for Bug #2 fixes: Conclusion content loss prevention.

Tests the fixes for high-confidence reasoning results that were losing
their conclusions during weighted voting and extraction.

Author: VulcanAMI Team
Date: 2026-01-18
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict


# ==============================================================================
# Test weighted_voting fixes
# ==============================================================================

class TestWeightedVotingFixes(unittest.TestCase):
    """Test suite for weighted_voting None/empty filtering fixes."""

    def test_weighted_voting_filters_none_values(self):
        """Test that weighted_voting filters out None values before voting."""
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Test case: None value with high weight should be filtered
        conclusions = ["Valid answer", None, "Another valid answer"]
        weights = [0.3, 0.5, 0.2]  # None has highest weight
        
        result = weighted_voting(conclusions, weights)
        
        # Should return one of the valid conclusions, not None
        self.assertIsNotNone(result)
        self.assertIn(result, ["Valid answer", "Another valid answer"])
        # With weights 0.3 vs 0.2 (normalized to 0.6 vs 0.4), should pick first
        self.assertEqual(result, "Valid answer")

    def test_weighted_voting_filters_string_none(self):
        """Test that weighted_voting filters out string 'None' values."""
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Test case: String "None" with high weight should be filtered
        conclusions = ["Valid", "None", "Answer"]
        weights = [0.3, 0.5, 0.2]  # "None" has highest weight
        
        result = weighted_voting(conclusions, weights)
        
        # Should return one of the valid conclusions, not "None" string
        self.assertNotEqual(result, "None")
        self.assertIn(result, ["Valid", "Answer"])

    def test_weighted_voting_filters_empty_strings(self):
        """Test that weighted_voting filters out empty/whitespace strings."""
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Test case: Empty and whitespace strings should be filtered
        conclusions = ["Valid", "", "  ", "Answer"]
        weights = [0.2, 0.3, 0.3, 0.2]
        
        result = weighted_voting(conclusions, weights)
        
        # Should return one of the non-empty conclusions
        self.assertIn(result, ["Valid", "Answer"])
        self.assertNotEqual(result, "")
        self.assertTrue(result.strip())

    def test_weighted_voting_all_invalid_returns_none(self):
        """Test that weighted_voting returns None when all conclusions are invalid."""
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Test case: All invalid conclusions
        conclusions = [None, "None", "", "  ", None]
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        result = weighted_voting(conclusions, weights)
        
        # Should return None when all conclusions are invalid
        self.assertIsNone(result)

    def test_weighted_voting_boolean_with_none(self):
        """Test that weighted_voting handles boolean conclusions with None values."""
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Test case: Boolean conclusions with None (should be filtered)
        conclusions = [True, None, False, True]
        weights = [0.3, 0.3, 0.2, 0.2]  # None has 0.3 weight
        
        result = weighted_voting(conclusions, weights)
        
        # Should return boolean, not None
        self.assertIsInstance(result, bool)
        # After filtering None: True has weights 0.3 + 0.2 = 0.5 (normalized to 5/7 ≈ 0.714)
        # False has weight 0.2 (normalized to 2/7 ≈ 0.286)
        # True weight 0.714 > 0.5 threshold, so should return True
        self.assertTrue(result)

    def test_weighted_voting_numeric_with_none(self):
        """Test that weighted_voting handles numeric conclusions with None values."""
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Test case: Numeric conclusions with None (should be filtered)
        conclusions = [10.0, None, 20.0, 30.0]
        weights = [0.3, 0.3, 0.2, 0.2]  # None has 0.3 weight
        
        result = weighted_voting(conclusions, weights)
        
        # Should return numeric average of valid values
        self.assertIsInstance(result, (int, float))
        self.assertIsNotNone(result)
        # After filtering None (weight 0.3), remaining weights [0.3, 0.2, 0.2] sum to 0.7
        # Normalized weights are [3/7, 2/7, 2/7]
        # Weighted average: 10*(3/7) + 20*(2/7) + 30*(2/7) = (30+40+60)/7 ≈ 18.57
        self.assertGreater(result, 18.0)
        self.assertLess(result, 19.0)


# ==============================================================================
# Test _extract_conclusion_from_dict fixes
# ==============================================================================

class TestConclusionExtractionFixes(unittest.TestCase):
    """Test suite for _extract_conclusion_from_dict enhancements."""

    def test_extract_conclusion_from_dict_basic(self):
        """Test basic conclusion extraction from dict."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        # Create a mock pool manager (we only need the method)
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # Test standard keys
        data1 = {"conclusion": "Test answer"}
        self.assertEqual(pool._extract_conclusion_from_dict(data1), "Test answer")
        
        data2 = {"response": "Another answer"}
        self.assertEqual(pool._extract_conclusion_from_dict(data2), "Another answer")
        
        data3 = {"result": "Result answer"}
        self.assertEqual(pool._extract_conclusion_from_dict(data3), "Result answer")

    def test_extract_conclusion_from_dict_filters_none(self):
        """Test that extraction filters None and 'None' string values."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # Should skip None values and find valid ones
        data1 = {"conclusion": None, "response": "Valid answer"}
        self.assertEqual(pool._extract_conclusion_from_dict(data1), "Valid answer")
        
        # Should skip "None" string values
        data2 = {"conclusion": "None", "response": "Valid answer"}
        self.assertEqual(pool._extract_conclusion_from_dict(data2), "Valid answer")
        
        # Should skip empty strings
        data3 = {"conclusion": "", "response": "Valid answer"}
        self.assertEqual(pool._extract_conclusion_from_dict(data3), "Valid answer")

    def test_extract_conclusion_from_dict_nested(self):
        """Test that extraction handles nested dictionaries recursively."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # Test nested dict with conclusion inside
        data = {
            "response": {
                "conclusion": "Nested in response"
            }
        }
        result = pool._extract_conclusion_from_dict(data)
        self.assertEqual(result, "Nested in response")

    def test_extract_conclusion_from_dict_priority_order(self):
        """Test that extraction uses correct priority order."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # world_model_response should have highest priority
        data = {
            "world_model_response": "World model answer",
            "conclusion": "Generic conclusion",
            "response": "Generic response",
        }
        self.assertEqual(pool._extract_conclusion_from_dict(data), "World model answer")
        
        # conclusion should be second priority
        data2 = {
            "conclusion": "Conclusion answer",
            "response": "Generic response",
            "result": "Generic result",
        }
        self.assertEqual(pool._extract_conclusion_from_dict(data2), "Conclusion answer")


# ==============================================================================
# Test _is_valid_conclusion fixes
# ==============================================================================

class TestValidConclusionCheck(unittest.TestCase):
    """Test suite for _is_valid_conclusion enhancements."""

    def test_is_valid_conclusion_none_detection(self):
        """Test that _is_valid_conclusion correctly detects None values."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # None should be invalid
        self.assertFalse(pool._is_valid_conclusion(None))
        
        # String "None" should be invalid
        self.assertFalse(pool._is_valid_conclusion("None"))
        self.assertFalse(pool._is_valid_conclusion("none"))
        self.assertFalse(pool._is_valid_conclusion("NONE"))
        
        # Valid string should be valid
        self.assertTrue(pool._is_valid_conclusion("Valid answer"))

    def test_is_valid_conclusion_empty_strings(self):
        """Test that _is_valid_conclusion detects empty/whitespace strings."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # Empty and whitespace strings should be invalid
        self.assertFalse(pool._is_valid_conclusion(""))
        self.assertFalse(pool._is_valid_conclusion("   "))
        self.assertFalse(pool._is_valid_conclusion("\t\n"))
        
        # Non-empty strings should be valid
        self.assertTrue(pool._is_valid_conclusion("Valid"))
        self.assertTrue(pool._is_valid_conclusion("  Valid  "))

    def test_is_valid_conclusion_empty_containers(self):
        """Test that _is_valid_conclusion detects empty containers."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # Empty containers should be invalid
        self.assertFalse(pool._is_valid_conclusion({}))
        self.assertFalse(pool._is_valid_conclusion([]))
        self.assertFalse(pool._is_valid_conclusion(()))
        
        # Non-empty containers should be valid
        self.assertTrue(pool._is_valid_conclusion({"key": "value"}))
        self.assertTrue(pool._is_valid_conclusion([1, 2, 3]))
        self.assertTrue(pool._is_valid_conclusion((1,)))

    def test_is_valid_conclusion_numeric_values(self):
        """Test that _is_valid_conclusion handles numeric values correctly."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # All numeric values should be valid (including 0)
        self.assertTrue(pool._is_valid_conclusion(0))
        self.assertTrue(pool._is_valid_conclusion(0.0))
        self.assertTrue(pool._is_valid_conclusion(42))
        self.assertTrue(pool._is_valid_conclusion(3.14159))
        self.assertTrue(pool._is_valid_conclusion(-10))

    def test_is_valid_conclusion_boolean_values(self):
        """Test that _is_valid_conclusion handles boolean values correctly."""
        from vulcan.orchestrator.agent_pool import AgentPoolManager
        
        pool = AgentPoolManager.__new__(AgentPoolManager)
        
        # Both True and False should be valid
        self.assertTrue(pool._is_valid_conclusion(True))
        self.assertTrue(pool._is_valid_conclusion(False))


# ==============================================================================
# Integration tests
# ==============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete fix."""

    def test_integration_high_confidence_with_valid_conclusion(self):
        """
        Integration test: High-confidence result with valid conclusion
        should be properly extracted and delivered.
        """
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Simulate ensemble reasoning with one engine returning None
        # This was causing the bug: None with high weight would win
        conclusions = [
            "Detailed mathematical answer: P(D|+) = 0.166667",
            None,
            "Probabilistic reasoning suggests: 16.7%"
        ]
        weights = [0.4, 0.3, 0.3]
        
        result = weighted_voting(conclusions, weights)
        
        # Should return a valid conclusion, not None
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should be one of the valid answers
        self.assertTrue("16" in result.lower() or "answer" in result.lower())

    def test_integration_all_none_with_high_confidence(self):
        """
        Integration test: When all conclusions are None but confidence is high,
        system should handle gracefully.
        """
        from vulcan.reasoning.unified.strategies import weighted_voting
        
        # Simulate problematic case from production logs
        conclusions = [None, None, None]
        weights = [0.33, 0.33, 0.34]
        
        result = weighted_voting(conclusions, weights)
        
        # Should return None (can't manufacture conclusion from nothing)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

