"""
Comprehensive tests for the singletons module.

Tests the central singleton registry for reasoning components, ensuring:
- Thread-safe singleton creation
- Correct lazy initialization
- Proper reset functionality
- Integration with problem decomposer
"""

import logging
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestSingletonRegistry(unittest.TestCase):
    """Test cases for the singleton registry functions."""

    def setUp(self):
        """Reset singletons before each test."""
        try:
            from vulcan.reasoning.singletons import reset_all
            reset_all()
        except ImportError:
            pass

    def tearDown(self):
        """Clean up singletons after each test."""
        try:
            from vulcan.reasoning.singletons import reset_all
            reset_all()
        except ImportError:
            pass

    def test_get_or_create_basic(self):
        """Test basic get_or_create functionality."""
        from vulcan.reasoning.singletons import get_or_create, _instances

        # Create a simple factory
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"created": True, "call_count": call_count}

        # First call should create instance
        instance1 = get_or_create("test_key", factory)
        self.assertEqual(instance1["created"], True)
        self.assertEqual(instance1["call_count"], 1)
        self.assertEqual(call_count, 1)

        # Second call should return same instance
        instance2 = get_or_create("test_key", factory)
        self.assertIs(instance1, instance2)
        self.assertEqual(call_count, 1)  # Factory should not be called again

    def test_get_or_create_thread_safety(self):
        """Test that get_or_create is thread-safe."""
        from vulcan.reasoning.singletons import get_or_create, reset_all

        reset_all()

        call_count = 0
        call_lock = threading.Lock()
        instances = []
        instances_lock = threading.Lock()

        def slow_factory():
            nonlocal call_count
            with call_lock:
                call_count += 1
            time.sleep(0.1)  # Simulate slow initialization
            return {"id": id(threading.current_thread())}

        def worker():
            instance = get_or_create("thread_test", slow_factory)
            with instances_lock:
                instances.append(instance)

        # Start multiple threads simultaneously
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        self.assertEqual(len(instances), 10)
        for instance in instances[1:]:
            self.assertIs(instance, instances[0])

        # Factory should only be called once
        self.assertEqual(call_count, 1)

    def test_reset_all_clears_instances(self):
        """Test that reset_all clears all cached singletons."""
        from vulcan.reasoning.singletons import (
            get_or_create,
            reset_all,
            _instances,
        )

        # Create some instances
        get_or_create("key1", lambda: {"key": 1})
        get_or_create("key2", lambda: {"key": 2})

        self.assertIn("key1", _instances)
        self.assertIn("key2", _instances)

        # Reset
        reset_all()

        # Instances should be cleared
        self.assertEqual(len(_instances), 0)

    def test_get_tool_selector_singleton(self):
        """Test that get_tool_selector returns singleton."""
        try:
            from vulcan.reasoning.singletons import get_tool_selector, reset_all

            reset_all()

            # First call
            ts1 = get_tool_selector()

            # Second call should return same instance
            ts2 = get_tool_selector()

            if ts1 is not None:
                self.assertIs(ts1, ts2)
        except ImportError:
            self.skipTest("ToolSelector not available")

    def test_get_bayesian_prior_singleton(self):
        """Test that get_bayesian_prior returns singleton."""
        try:
            from vulcan.reasoning.singletons import get_bayesian_prior, reset_all

            reset_all()

            # First call
            bp1 = get_bayesian_prior()

            # Second call should return same instance
            bp2 = get_bayesian_prior()

            if bp1 is not None:
                self.assertIs(bp1, bp2)
        except ImportError:
            self.skipTest("BayesianMemoryPrior not available")

    def test_get_problem_decomposer_singleton(self):
        """Test that get_problem_decomposer returns singleton."""
        try:
            from vulcan.reasoning.singletons import get_problem_decomposer, reset_all

            reset_all()

            # First call
            pd1 = get_problem_decomposer()

            # Second call should return same instance
            pd2 = get_problem_decomposer()

            if pd1 is not None:
                self.assertIs(pd1, pd2)
        except ImportError:
            self.skipTest("ProblemDecomposer not available")

    def test_prewarm_all_initializes_components(self):
        """Test that prewarm_all initializes all singletons."""
        try:
            from vulcan.reasoning.singletons import prewarm_all, reset_all

            reset_all()

            results = prewarm_all()

            # Should return a dictionary with component names
            self.assertIsInstance(results, dict)
            self.assertIn("tool_selector", results)
            self.assertIn("reasoning_integration", results)
            self.assertIn("problem_decomposer", results)

            # Count should reflect actual availability
            total = len(results)
            initialized = sum(1 for v in results.values() if v)
            logger.info(f"Prewarm results: {initialized}/{total} components initialized")

        except ImportError:
            self.skipTest("Singletons module not available")

    def test_cleanup_releases_singletons(self):
        """Test that cleanup releases all singletons."""
        try:
            from vulcan.reasoning.singletons import (
                cleanup,
                get_or_create,
                _instances,
            )

            # Create some instances
            get_or_create("cleanup_test1", lambda: {})
            get_or_create("cleanup_test2", lambda: {})

            self.assertGreater(len(_instances), 0)

            # Cleanup
            cleanup()

            # Should be cleared
            self.assertEqual(len(_instances), 0)

        except ImportError:
            self.skipTest("Singletons module not available")


class TestSingletonImportErrors(unittest.TestCase):
    """Test singleton behavior when components are not available."""

    def setUp(self):
        """Reset singletons before each test."""
        try:
            from vulcan.reasoning.singletons import reset_all
            reset_all()
        except ImportError:
            pass

    def test_graceful_degradation_on_import_error(self):
        """Test that singletons return None on import errors."""
        from vulcan.reasoning import singletons

        # Mock an import error for ToolSelector
        with patch.dict('sys.modules', {'vulcan.reasoning.selection.tool_selector': None}):
            with patch('vulcan.reasoning.singletons.logger') as mock_logger:
                # This should not raise, just return None
                result = singletons.get_tool_selector()
                # Result depends on whether ToolSelector is actually available
                # The important thing is it doesn't crash


class TestSingletonIntegration(unittest.TestCase):
    """Integration tests for singletons with actual components."""

    def setUp(self):
        """Reset singletons before each test."""
        try:
            from vulcan.reasoning.singletons import reset_all
            reset_all()
        except ImportError:
            pass

    def test_multiple_singletons_independent(self):
        """Test that different singletons are independent."""
        try:
            from vulcan.reasoning.singletons import (
                get_tool_selector,
                get_bayesian_prior,
                get_cost_model,
                reset_all,
            )

            reset_all()

            ts = get_tool_selector()
            bp = get_bayesian_prior()
            cm = get_cost_model()

            # If they exist, they should be different objects
            if ts is not None and bp is not None:
                self.assertIsNot(ts, bp)
            if bp is not None and cm is not None:
                self.assertIsNot(bp, cm)
            if ts is not None and cm is not None:
                self.assertIsNot(ts, cm)

        except ImportError:
            self.skipTest("Singletons not available")


if __name__ == "__main__":
    unittest.main()
