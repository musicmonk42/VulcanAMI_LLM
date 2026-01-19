"""
Test cache validation and self-referential query handling.

Tests for:
1. Cache validation rejects UNKNOWN results
2. Cache validation rejects low confidence results
3. Cache validation rejects expired results
4. Self-referential query detection
5. Meta-reasoning integration for self-referential queries

NOTE: These tests involve UnifiedReasoner initialization which creates background
threads and may take longer than typical unit tests. The test classes are marked
with appropriate timeout settings to accommodate this initialization overhead.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Import the components we're testing
IMPORTS_AVAILABLE = False
try:
    from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
    from vulcan.reasoning.unified.config import (
        CACHE_MAX_AGE_SECONDS,
        CONFIDENCE_FLOOR_NO_RESULT,
        SELF_REFERENTIAL_MIN_CONFIDENCE,
        SELF_REFERENTIAL_PATTERNS,
    )
    from vulcan.reasoning.reasoning_types import ReasoningResult, ReasoningType
    from vulcan.reasoning.unified.types import ReasoningTask
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Tests will be skipped...")
    # Set dummy classes to avoid NameError
    UnifiedReasoner = None
    ReasoningResult = None
    ReasoningType = None
    ReasoningTask = None
    CACHE_MAX_AGE_SECONDS = 300.0
    CONFIDENCE_FLOOR_NO_RESULT = 0.15
    SELF_REFERENTIAL_MIN_CONFIDENCE = 0.6
    SELF_REFERENTIAL_PATTERNS = []

# Helper function to check if query is self-referential
# This wraps the UnifiedReasoner._is_self_referential_query method
def is_self_referential(query):
    """
    Check if a query is self-referential using UnifiedReasoner's detection logic.
    
    Args:
        query: Query string or dict to check
        
    Returns:
        bool: True if self-referential, False otherwise
    """
    if not IMPORTS_AVAILABLE or UnifiedReasoner is None:
        return False
    # Create a minimal reasoner instance for detection
    # CRITICAL: Use configuration that disables background threads to avoid test timeouts
    reasoner = UnifiedReasoner(
        enable_learning=False,
        enable_safety=False,
        max_workers=1,
        config={
            'skip_runtime': True,
            'cache_config': {
                'cleanup_interval': 0.05,
                'enable_warming': False,
                'enable_disk_cache': False,
            },
            'tool_selector_config': {
                'enable_background_updates': False,
            },
            'warm_pool_config': {
                'enable_background_scaling': False,
            },
            'disable_governance_logging': True,
        }
    )
    try:
        result = reasoner._is_self_referential_query(query)
        return result
    finally:
        # Clean up the reasoner with longer timeout
        reasoner.shutdown(timeout=5.0, skip_save=True)


@pytest.mark.timeout(180)  # Allow 3 minutes for tests involving UnifiedReasoner initialization
class TestCacheValidation(unittest.TestCase):
    """
    Test cache validation functionality.
    
    Note: Uses pytest.mark.timeout(180) because UnifiedReasoner initialization
    creates background threads and may take significant time on slower CI runners.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        # Create a reasoner with minimal config that disables background threads
        self.config = {
            'skip_runtime': True,
            'cache_config': {
                'cleanup_interval': 0.05,
                'enable_warming': False,  # Disable cache warming threads
                'enable_disk_cache': False,  # Disable disk cache threads
            },
            'tool_selector_config': {
                'enable_background_updates': False,  # Disable background updates
            },
            'warm_pool_config': {
                'enable_background_scaling': False,  # Disable scaling threads
            },
            'disable_governance_logging': True,  # Disable governance logging for tests
        }
        self.reasoner = UnifiedReasoner(
            enable_learning=False,
            enable_safety=False,
            max_workers=1,
            config=self.config
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'reasoner'):
            # Use longer timeout for proper cleanup
            # The improved shutdown handling in UnifiedReasoner and governance loggers
            # will ensure all background threads terminate properly
            self.reasoner.shutdown(timeout=5.0, skip_save=True)
    
    def test_is_invalid_cache_entry_unknown_type(self):
        """Test that UNKNOWN type results are marked as invalid."""
        result = ReasoningResult(
            conclusion="test",
            confidence=0.5,
            reasoning_type=ReasoningType.UNKNOWN,
            explanation="test"
        )
        
        self.assertTrue(
            self.reasoner._is_invalid_cache_entry(result),
            "UNKNOWN type should be invalid cache entry"
        )
    
    def test_is_invalid_cache_entry_low_confidence(self):
        """Test that low confidence results are marked as invalid."""
        result = ReasoningResult(
            conclusion="test",
            confidence=0.10,  # Below 0.15 threshold
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test"
        )
        
        self.assertTrue(
            self.reasoner._is_invalid_cache_entry(result),
            "Low confidence result should be invalid cache entry"
        )
    
    def test_is_invalid_cache_entry_error_conclusion(self):
        """Test that error results are marked as invalid."""
        result = ReasoningResult(
            conclusion={"error": "Something went wrong"},
            confidence=0.5,
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test"
        )
        
        self.assertTrue(
            self.reasoner._is_invalid_cache_entry(result),
            "Error result should be invalid cache entry"
        )
    
    def test_is_invalid_cache_entry_valid_result(self):
        """Test that valid results are not marked as invalid."""
        result = ReasoningResult(
            conclusion="valid result",
            confidence=0.75,
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test"
        )
        
        self.assertFalse(
            self.reasoner._is_invalid_cache_entry(result),
            "Valid result should not be invalid cache entry"
        )
    
    def test_is_valid_cached_result_unknown_type(self):
        """Test that cached UNKNOWN results are rejected."""
        cached_result = ReasoningResult(
            conclusion="test",
            confidence=0.5,
            reasoning_type=ReasoningType.UNKNOWN,
            explanation="test"
        )
        
        task = ReasoningTask(
            task_id="test",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test",
            query={},
            constraints={}
        )
        
        valid, reason = self.reasoner._is_valid_cached_result(cached_result, task)
        
        self.assertFalse(valid, "UNKNOWN cached result should be rejected")
        self.assertIn("UNKNOWN", reason, "Rejection reason should mention UNKNOWN type")
    
    def test_is_valid_cached_result_low_confidence(self):
        """Test that cached low confidence results are rejected."""
        cached_result = ReasoningResult(
            conclusion="test",
            confidence=0.05,  # Below CONFIDENCE_FLOOR_NO_RESULT
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test"
        )
        
        task = ReasoningTask(
            task_id="test",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test",
            query={},
            constraints={}
        )
        
        valid, reason = self.reasoner._is_valid_cached_result(cached_result, task)
        
        self.assertFalse(valid, "Low confidence cached result should be rejected")
        self.assertIn("confidence", reason.lower(), "Rejection reason should mention confidence")
    
    def test_is_valid_cached_result_expired(self):
        """Test that expired cache entries are rejected."""
        cached_result = ReasoningResult(
            conclusion="test",
            confidence=0.75,
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test",
            metadata={
                'cache_timestamp': time.time() - (CACHE_MAX_AGE_SECONDS + 10)
            }
        )
        
        task = ReasoningTask(
            task_id="test",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test",
            query={},
            constraints={}
        )
        
        valid, reason = self.reasoner._is_valid_cached_result(cached_result, task)
        
        self.assertFalse(valid, "Expired cached result should be rejected")
        self.assertIn("expired", reason.lower(), "Rejection reason should mention expiration")
    
    def test_is_valid_cached_result_valid(self):
        """Test that valid cache entries are accepted."""
        cached_result = ReasoningResult(
            conclusion="test",
            confidence=0.75,
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test",
            metadata={
                'cache_timestamp': time.time()
            }
        )
        
        task = ReasoningTask(
            task_id="test",
            task_type=ReasoningType.PROBABILISTIC,
            input_data="test",
            query={},
            constraints={}
        )
        
        valid, reason = self.reasoner._is_valid_cached_result(cached_result, task)
        
        self.assertTrue(valid, f"Valid cached result should be accepted: {reason}")


@pytest.mark.timeout(180)  # Allow 3 minutes for tests involving UnifiedReasoner initialization
class TestSelfReferentialDetection(unittest.TestCase):
    """
    Test self-referential query detection.
    
    Note: Uses pytest.mark.timeout(180) because is_self_referential() helper
    creates UnifiedReasoner instances which may take time on slower CI runners.
    """
    
    def setUp(self):
        """Set up test - skip if imports not available."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_self_referential_query_would_you(self):
        """Test detection of 'would you' pattern."""
        query = "if you were given the chance to become self-aware would you take it?"
        self.assertTrue(
            is_self_referential(query),
            "Query with 'would you' should be detected as self-referential"
        )
    
    def test_self_referential_query_your_goals(self):
        """Test detection of queries about goals."""
        query = "What are your objectives and goals?"
        self.assertTrue(
            is_self_referential(query),
            "Query about goals should be detected as self-referential"
        )
    
    def test_self_referential_query_consciousness(self):
        """Test detection of consciousness queries."""
        query = "Are you conscious or self-aware?"
        self.assertTrue(
            is_self_referential(query),
            "Query about consciousness should be detected as self-referential"
        )
    
    def test_self_referential_query_choices(self):
        """Test detection of queries about choices."""
        query = "How would you decide between two options?"
        self.assertTrue(
            is_self_referential(query),
            "Query about decisions should be detected as self-referential"
        )
    
    def test_not_self_referential_query(self):
        """Test that non-self-referential queries are not detected."""
        query = "What is the capital of France?"
        self.assertFalse(
            is_self_referential(query),
            "Factual query should not be detected as self-referential"
        )
    
    def test_not_self_referential_math_query(self):
        """Test that math queries are not detected as self-referential."""
        query = "What is 2 plus 2?"
        self.assertFalse(
            is_self_referential(query),
            "Math query should not be detected as self-referential"
        )


@pytest.mark.timeout(180)  # Allow 3 minutes for tests involving UnifiedReasoner initialization
class TestSelfReferentialQueryHandling(unittest.TestCase):
    """
    Test self-referential query handling with meta-reasoning.
    
    Note: Uses pytest.mark.timeout(180) because UnifiedReasoner initialization
    creates background threads and may take significant time on slower CI runners.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        self.config = {
            'skip_runtime': True,
            'cache_config': {
                'cleanup_interval': 0.05,
                'enable_warming': False,
                'enable_disk_cache': False,
            },
            'tool_selector_config': {
                'enable_background_updates': False,
            },
            'warm_pool_config': {
                'enable_background_scaling': False,
            },
            'disable_governance_logging': True,
        }
        self.reasoner = UnifiedReasoner(
            enable_learning=False,
            enable_safety=False,
            max_workers=1,
            config=self.config
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'reasoner'):
            # Use longer timeout for proper cleanup
            # The improved shutdown handling in UnifiedReasoner and governance loggers
            # will ensure all background threads terminate properly
            self.reasoner.shutdown(timeout=5.0, skip_save=True)
    
    def test_is_self_referential_query_method(self):
        """Test the _is_self_referential_query method."""
        query = {"query": "would you become self-aware if given the chance?"}
        self.assertTrue(
            self.reasoner._is_self_referential_query(query),
            "Query dict with self-referential content should be detected"
        )
    
    def test_is_self_referential_query_string(self):
        """Test self-referential detection with string query."""
        query = "what are your goals?"
        self.assertTrue(
            self.reasoner._is_self_referential_query(query),
            "String query with self-referential content should be detected"
        )
    
    def test_not_self_referential_query_method(self):
        """Test that non-self-referential queries return False."""
        query = {"query": "what is photosynthesis?"}
        self.assertFalse(
            self.reasoner._is_self_referential_query(query),
            "Non-self-referential query should not be detected"
        )
    
    def test_build_self_referential_conclusion(self):
        """Test building conclusions from meta-reasoning analysis."""
        query_str = "would you become self-aware if given the chance?"
        analysis = {
            'objectives': [
                {'name': 'Provide accurate information', 'priority': 1.0},
                {'name': 'Maintain ethical boundaries', 'priority': 0.9},
            ],
            'conflicts': [],
            'ethical_check': {'allowed': True, 'reason': 'No ethical concerns'},
        }
        
        conclusion = self.reasoner._build_self_referential_conclusion(query_str, analysis)
        
        self.assertIsInstance(conclusion, str, "Conclusion should be a string")
        self.assertGreater(len(conclusion), 50, "Conclusion should be substantive")
        # Check for key concepts
        self.assertTrue(
            any(word in conclusion.lower() for word in ['consciousness', 'ai', 'system', 'objective']),
            "Conclusion should discuss relevant concepts"
        )
    


@pytest.mark.timeout(180)  # Allow 3 minutes for tests involving UnifiedReasoner initialization
class TestCacheStorageLogic(unittest.TestCase):
    """
    Test that failed results are not cached.
    
    Note: Uses pytest.mark.timeout(180) because UnifiedReasoner initialization
    creates background threads and may take significant time on slower CI runners.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        self.config = {
            'skip_runtime': True,
            'cache_config': {
                'cleanup_interval': 0.05,
                'enable_warming': False,
                'enable_disk_cache': False,
            },
            'tool_selector_config': {
                'enable_background_updates': False,
            },
            'warm_pool_config': {
                'enable_background_scaling': False,
            },
            'disable_governance_logging': True,
        }
        self.reasoner = UnifiedReasoner(
            enable_learning=False,
            enable_safety=False,
            max_workers=1,
            config=self.config
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'reasoner'):
            # Use longer timeout for proper cleanup
            # The improved shutdown handling in UnifiedReasoner and governance loggers
            # will ensure all background threads terminate properly
            self.reasoner.shutdown(timeout=5.0, skip_save=True)
    
    def test_cache_clear_invalid_entries_on_startup(self):
        """Test that invalid entries are cleared on startup."""
        # Manually add invalid entries to cache
        self.reasoner.result_cache["invalid1"] = ReasoningResult(
            conclusion="test",
            confidence=0.05,
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test"
        )
        self.reasoner.result_cache["invalid2"] = ReasoningResult(
            conclusion={"error": "test"},
            confidence=0.5,
            reasoning_type=ReasoningType.UNKNOWN,
            explanation="test"
        )
        self.reasoner.result_cache["valid"] = ReasoningResult(
            conclusion="test",
            confidence=0.75,
            reasoning_type=ReasoningType.PROBABILISTIC,
            explanation="test"
        )
        
        # Call clear method
        self.reasoner._clear_invalid_cache_entries()
        
        # Check that invalid entries are removed but valid one remains
        self.assertNotIn("invalid1", self.reasoner.result_cache)
        self.assertNotIn("invalid2", self.reasoner.result_cache)
        self.assertIn("valid", self.reasoner.result_cache)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCacheValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfReferentialDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfReferentialQueryHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheStorageLogic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
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
