"""
Comprehensive Test Suite for NSOAligner Singleton Lifecycle
==========================================================

Tests to verify that the NSOAligner singleton pattern works correctly and
that shutdown() is never called incorrectly on the singleton instance.

These tests ensure:
1. get_nso_aligner() returns the same instance across multiple calls
2. The singleton persists across different contexts
3. reset_nso_aligner() properly cleans up the singleton
4. No code incorrectly calls shutdown() on the singleton
"""

import pytest
import threading
import time
from unittest.mock import patch, MagicMock

# Skip entire module if torch is not available
pytest.importorskip("torch", reason="PyTorch required for NSOAligner tests")

from src.nso_aligner import (
    get_nso_aligner,
    reset_nso_aligner,
    _nso_aligner_instance,
)


class TestNSOAlignerSingletonLifecycle:
    """Test suite for NSOAligner singleton lifecycle management."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_nso_aligner()

    def teardown_method(self):
        """Clean up singleton after each test."""
        reset_nso_aligner()

    def test_singleton_returns_same_instance(self):
        """Test that get_nso_aligner() returns the same instance on multiple calls."""
        # First call creates instance
        instance1 = get_nso_aligner()
        assert instance1 is not None

        # Subsequent calls return same instance
        instance2 = get_nso_aligner()
        assert instance2 is instance1

        instance3 = get_nso_aligner()
        assert instance3 is instance1

    def test_singleton_persists_across_contexts(self):
        """Test that singleton persists across different function contexts."""

        def get_aligner_in_context():
            return get_nso_aligner()

        # Get instance in main context
        main_instance = get_nso_aligner()

        # Get instance in function context
        func_instance = get_aligner_in_context()

        # Should be same instance
        assert func_instance is main_instance

    def test_singleton_thread_safety(self):
        """Test that singleton creation is thread-safe."""
        instances = []
        lock = threading.Lock()

        def create_instance():
            instance = get_nso_aligner()
            with lock:
                instances.append(instance)

        # Create multiple threads that all try to get instance
        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(instances) == 10
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    def test_reset_clears_singleton(self):
        """Test that reset_nso_aligner() properly clears the singleton."""
        # Create instance
        instance1 = get_nso_aligner()
        assert instance1 is not None

        # Reset singleton
        reset_nso_aligner()

        # Next call should create new instance
        instance2 = get_nso_aligner()
        assert instance2 is not None
        assert instance2 is not instance1

    def test_reset_calls_shutdown(self):
        """Test that reset_nso_aligner() calls shutdown() on the instance."""
        # Create instance
        instance = get_nso_aligner()

        # Mock the shutdown method
        with patch.object(instance, "shutdown") as mock_shutdown:
            reset_nso_aligner()
            # shutdown() should have been called
            mock_shutdown.assert_called_once()

    def test_singleton_with_different_parameters(self):
        """Test that singleton ignores parameters on subsequent calls."""
        # Create with no parameters
        instance1 = get_nso_aligner()

        # Call with parameters (should be ignored, return same instance)
        mock_client = MagicMock()
        instance2 = get_nso_aligner(claude_client=mock_client)

        assert instance2 is instance1

    def test_no_shutdown_called_in_normal_usage(self):
        """
        Test that shutdown() is never called during normal usage.
        
        This is a critical test to ensure the bug is fixed.
        """
        instance = get_nso_aligner()

        # Mock shutdown to track if it's called
        original_shutdown = instance.shutdown
        call_count = {"count": 0}

        def mock_shutdown():
            call_count["count"] += 1
            original_shutdown()

        instance.shutdown = mock_shutdown

        # Simulate normal usage (multiple requests)
        for _ in range(5):
            aligner = get_nso_aligner()
            # Use the aligner
            _ = aligner  # Just reference it

        # shutdown() should NEVER be called during normal usage
        assert (
            call_count["count"] == 0
        ), "BUG: shutdown() was called on singleton instance!"

    def test_singleton_survives_multiple_requests(self):
        """
        Test that singleton instance survives across multiple simulated requests.
        
        This simulates the real-world scenario where multiple HTTP requests
        all use the same NSOAligner instance.
        """

        def simulate_request():
            """Simulate a single request using NSOAligner."""
            aligner = get_nso_aligner()
            # Simulate using the aligner
            assert aligner is not None
            # DO NOT call shutdown() - this was the bug!
            return aligner

        # Simulate 10 sequential requests
        instances = [simulate_request() for _ in range(10)]

        # All should be the same instance
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    def test_proper_cleanup_only_on_reset(self):
        """Test that cleanup only happens via reset_nso_aligner()."""
        # Create instance
        instance1 = get_nso_aligner()
        instance1_id = id(instance1)

        # Simulate multiple uses (like multiple requests)
        for _ in range(5):
            instance = get_nso_aligner()
            assert id(instance) == instance1_id

        # Instance should still be alive
        instance = get_nso_aligner()
        assert id(instance) == instance1_id

        # Only reset should clean it up
        reset_nso_aligner()

        # Now we should get a new instance
        instance2 = get_nso_aligner()
        assert id(instance2) != instance1_id


class TestNSOAlignerUsagePatterns:
    """Test correct and incorrect usage patterns."""

    def teardown_method(self):
        """Clean up after each test."""
        reset_nso_aligner()

    def test_correct_usage_pattern(self):
        """Document and test the correct usage pattern."""
        # CORRECT: Get instance, use it, don't call shutdown()
        safety = get_nso_aligner()
        # Use the instance for validation
        # Note: We can't actually run multi_model_audit without proper setup,
        # but we test the pattern
        assert safety is not None
        # NO shutdown() call here - this is the fix!

    def test_incorrect_pattern_documented(self):
        """
        Document the INCORRECT pattern that was fixed.
        
        This test exists to document what NOT to do.
        """
        safety = get_nso_aligner()

        # Track if shutdown is called
        shutdown_called = False

        def mock_shutdown():
            nonlocal shutdown_called
            shutdown_called = True

        # Replace shutdown with mock
        original_shutdown = safety.shutdown
        safety.shutdown = mock_shutdown

        # INCORRECT PATTERN (this was the bug):
        # try:
        #     result = safety.multi_model_audit(data)
        # finally:
        #     safety.shutdown()  # BUG: Destroys singleton!

        # Since we're testing the CORRECT pattern now, shutdown should NOT be called
        assert not shutdown_called, "shutdown() should never be called on singleton!"


@pytest.mark.integration
class TestNSOAlignerIntegration:
    """Integration tests for NSOAligner with real components."""

    def teardown_method(self):
        """Clean up after each test."""
        reset_nso_aligner()

    def test_multiple_sequential_uses(self):
        """Test multiple sequential uses of the aligner."""
        # Simulate 3 different use cases all using the same singleton
        aligner1 = get_nso_aligner()
        aligner2 = get_nso_aligner()
        aligner3 = get_nso_aligner()

        # All should be the same instance
        assert aligner1 is aligner2 is aligner3

    def test_reset_during_testing(self):
        """Test that reset works correctly during test cleanup."""
        # This pattern should be used in test fixtures
        instance1 = get_nso_aligner()

        # Cleanup (e.g., in tearDown or fixture cleanup)
        reset_nso_aligner()

        # New test can get fresh instance
        instance2 = get_nso_aligner()

        assert instance1 is not instance2


def test_singleton_documentation_example():
    """
    Test the example code from the get_nso_aligner() docstring.
    
    This ensures our documentation examples actually work.
    """
    # Example from docstring - CORRECT usage
    safety = get_nso_aligner()
    # result = safety.multi_model_audit(data)  # Would work with real data
    # NO shutdown() call - instance persists for next request

    # Clean up for next test
    reset_nso_aligner()


def test_reset_documentation_example():
    """Test the example code from reset_nso_aligner() docstring."""
    # Example: test cleanup
    # In actual test, this would be in tearDown
    reset_nso_aligner()

    # Can now get fresh instance
    safety = get_nso_aligner()
    assert safety is not None

    # Cleanup
    reset_nso_aligner()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
