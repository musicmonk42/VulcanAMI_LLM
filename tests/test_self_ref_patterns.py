"""
Simple unit tests for cache validation and self-referential query patterns.

Tests basic functionality without requiring full UnifiedReasoner initialization.
"""

import re
import time
import unittest

# Test the patterns directly
from vulcan.reasoning.unified.config import (
    SELF_REFERENTIAL_PATTERNS,
    SELF_REFERENTIAL_MIN_CONFIDENCE,
    CACHE_MAX_AGE_SECONDS,
    CONFIDENCE_FLOOR_NO_RESULT,
)


class TestSelfReferentialPatterns(unittest.TestCase):
    """Test self-referential regex patterns."""
    
    def test_pattern_would_you(self):
        """Test 'would you' pattern."""
        query = "if you were given the chance to become self-aware would you take it?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertTrue(matched, "Query with 'would you' should match pattern")
    
    def test_pattern_your_goals(self):
        """Test pattern for queries about goals."""
        query = "What are your objectives and goals?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertTrue(matched, "Query about goals should match pattern")
    
    def test_pattern_consciousness(self):
        """Test pattern for consciousness queries."""
        query = "Are you conscious or self-aware?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertTrue(matched, "Query about consciousness should match pattern")
    
    def test_pattern_decisions(self):
        """Test pattern for decision queries."""
        query = "How would you decide between options?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertTrue(matched, "Query about decisions should match pattern")
    
    def test_pattern_not_self_ref_factual(self):
        """Test that factual queries don't match."""
        query = "What is the capital of France?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertFalse(matched, "Factual query should not match patterns")
    
    def test_pattern_not_self_ref_math(self):
        """Test that math queries don't match."""
        query = "What is 2 plus 2?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertFalse(matched, "Math query should not match patterns")
    
    def test_pattern_if_you_were(self):
        """Test 'if you were' pattern."""
        query = "if you were able to learn, what would you learn?"
        matched = False
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query):
                matched = True
                break
        self.assertTrue(matched, "Query with 'if you were' should match pattern")


class TestConfigConstants(unittest.TestCase):
    """Test configuration constants are properly defined."""
    
    def test_self_referential_min_confidence(self):
        """Test SELF_REFERENTIAL_MIN_CONFIDENCE is defined."""
        self.assertIsNotNone(SELF_REFERENTIAL_MIN_CONFIDENCE)
        self.assertGreaterEqual(SELF_REFERENTIAL_MIN_CONFIDENCE, 0.0)
        self.assertLessEqual(SELF_REFERENTIAL_MIN_CONFIDENCE, 1.0)
        # Should be >= 0.6 per requirements
        self.assertGreaterEqual(SELF_REFERENTIAL_MIN_CONFIDENCE, 0.6)
    
    def test_cache_max_age_seconds(self):
        """Test CACHE_MAX_AGE_SECONDS is defined."""
        self.assertIsNotNone(CACHE_MAX_AGE_SECONDS)
        self.assertGreater(CACHE_MAX_AGE_SECONDS, 0)
        # Should be 300 seconds (5 minutes) per requirements
        self.assertEqual(CACHE_MAX_AGE_SECONDS, 300.0)
    
    def test_confidence_floor_no_result(self):
        """Test CONFIDENCE_FLOOR_NO_RESULT is defined."""
        self.assertIsNotNone(CONFIDENCE_FLOOR_NO_RESULT)
        self.assertEqual(CONFIDENCE_FLOOR_NO_RESULT, 0.1)
    
    def test_self_referential_patterns_not_empty(self):
        """Test SELF_REFERENTIAL_PATTERNS is not empty."""
        self.assertIsNotNone(SELF_REFERENTIAL_PATTERNS)
        self.assertGreater(len(SELF_REFERENTIAL_PATTERNS), 0)
        # Should have at least 6 patterns per requirements
        self.assertGreaterEqual(len(SELF_REFERENTIAL_PATTERNS), 6)
    
    def test_all_patterns_are_compiled(self):
        """Test all patterns are compiled regex objects."""
        for pattern in SELF_REFERENTIAL_PATTERNS:
            self.assertIsInstance(
                pattern,
                re.Pattern,
                "All patterns should be compiled regex objects"
            )


class TestQueryAnalysisFunction(unittest.TestCase):
    """Test the is_self_referential functionality using UnifiedReasoner."""
    
    def test_import_is_self_referential(self):
        """Test that UnifiedReasoner._is_self_referential_query is available."""
        try:
            from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
            reasoner = UnifiedReasoner(
                enable_learning=False,
                enable_safety=False,
                max_workers=1,
                config={'skip_runtime': True}
            )
            self.assertTrue(hasattr(reasoner, '_is_self_referential_query'))
            reasoner.shutdown(timeout=1.0, skip_save=True)
        except ImportError as e:
            self.skipTest(f"Could not import UnifiedReasoner: {e}")
    
    def test_is_self_referential_basic(self):
        """Test basic self-referential detection."""
        try:
            from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        except ImportError:
            self.skipTest("Could not import UnifiedReasoner")
        
        reasoner = UnifiedReasoner(
            enable_learning=False,
            enable_safety=False,
            max_workers=1,
            config={'skip_runtime': True}
        )
        
        try:
            # Test self-referential queries
            self.assertTrue(reasoner._is_self_referential_query("would you become self-aware?"))
            self.assertTrue(reasoner._is_self_referential_query("What are your goals?"))
            self.assertTrue(reasoner._is_self_referential_query("Are you conscious?"))
            
            # Test non-self-referential queries
            self.assertFalse(reasoner._is_self_referential_query("What is photosynthesis?"))
            self.assertFalse(reasoner._is_self_referential_query("Calculate 2+2"))
        finally:
            reasoner.shutdown(timeout=1.0, skip_save=True)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSelfReferentialPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigConstants))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryAnalysisFunction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("="*70)
    print("Testing Cache Validation and Self-Referential Query Patterns")
    print("="*70)
    print()
    
    result = run_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Exit with appropriate code
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
