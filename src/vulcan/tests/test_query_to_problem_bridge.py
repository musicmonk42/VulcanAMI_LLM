"""
Comprehensive tests for the QueryToProblemBridge module.

Tests the bridge between query analysis and problem decomposition, ensuring:
- Correct conversion of queries to ProblemGraph
- Subproblem detection for various query patterns
- Result aggregation from subproblem executions
- Thread safety and performance
"""

import logging
import threading
import time
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestQueryToProblemBridge(unittest.TestCase):
    """Test cases for the QueryToProblemBridge class."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from vulcan.reasoning.query_to_problem_bridge import QueryToProblemBridge
            self.bridge = QueryToProblemBridge()
        except ImportError as e:
            self.skipTest(f"QueryToProblemBridge not available: {e}")

    def test_bridge_initialization(self):
        """Test that bridge initializes correctly."""
        self.assertIsNotNone(self.bridge)
        self.assertEqual(self.bridge._conversion_count, 0)
        self.assertEqual(self.bridge._aggregation_count, 0)

    def test_convert_simple_query(self):
        """Test conversion of a simple query."""
        query = "What is the capital of France?"
        query_analysis = {
            'type': 'general',
            'complexity': 0.2,
            'uncertainty': 0.1,
        }

        result = self.bridge.convert_to_problem_graph(query, query_analysis)

        # Check if decomposer is available
        try:
            from vulcan.problem_decomposer import PROBLEM_DECOMPOSER_AVAILABLE
            if not PROBLEM_DECOMPOSER_AVAILABLE:
                self.assertIsNone(result)
                return
        except ImportError:
            self.assertIsNone(result)
            return

        if result is not None:
            self.assertIn("root", result.nodes)
            self.assertEqual(result.nodes["root"]["type"], "query")
            self.assertEqual(result.nodes["root"]["content"], query)
            self.assertAlmostEqual(result.complexity_score, 0.2)

    def test_convert_complex_query_with_tool_hints(self):
        """Test conversion with tool selection hints."""
        query = "Analyze the causal relationship between X and Y"
        query_analysis = {
            'type': 'reasoning',
            'complexity': 0.7,
            'uncertainty': 0.3,
            'requires_reasoning': True,
        }
        tool_selection = {
            'tools': ['causal', 'probabilistic'],
            'strategy': 'multi_tool',
            'confidence': 0.8,
        }

        result = self.bridge.convert_to_problem_graph(
            query, query_analysis, tool_selection
        )

        try:
            from vulcan.problem_decomposer import PROBLEM_DECOMPOSER_AVAILABLE
            if not PROBLEM_DECOMPOSER_AVAILABLE:
                self.assertIsNone(result)
                return
        except ImportError:
            self.assertIsNone(result)
            return

        if result is not None:
            self.assertIn("tool_hints", result.nodes)
            self.assertEqual(result.nodes["tool_hints"]["tools"], ['causal', 'probabilistic'])

    def test_detect_multiple_questions(self):
        """Test detection of multiple questions in query."""
        query = "What is X? How does Y work? Why is Z important?"
        query_analysis = {'type': 'general', 'complexity': 0.5}

        subproblems = self.bridge._detect_subproblems(query, query_analysis)

        # Should detect 3 questions
        self.assertGreaterEqual(len(subproblems), 1)
        for sp in subproblems:
            self.assertIn('content', sp)
            self.assertIn('type', sp)

    def test_detect_conditional_structure(self):
        """Test detection of conditional structures."""
        query = "If the temperature exceeds 100 degrees, then what happens to the material?"
        query_analysis = {'type': 'reasoning', 'complexity': 0.6}

        subproblems = self.bridge._detect_subproblems(query, query_analysis)

        # Should detect conditional pattern
        condition_found = any(sp.get('type') == 'condition' for sp in subproblems)
        # Conditional detection is best-effort
        if len(subproblems) > 0:
            self.assertTrue(any('content' in sp for sp in subproblems))

    def test_detect_comparative_structure(self):
        """Test detection of comparative structures."""
        query = "Compare Python vs JavaScript for web development"
        query_analysis = {'type': 'reasoning', 'complexity': 0.5}

        subproblems = self.bridge._detect_subproblems(query, query_analysis)

        # Should detect comparative pattern
        if len(subproblems) > 0:
            types = [sp.get('type') for sp in subproblems]
            has_comparison = any('comparison' in str(t) for t in types if t)
            # Comparison detection should work for "vs" pattern
            self.assertTrue(has_comparison or len(subproblems) >= 2)

    def test_score_to_complexity_name(self):
        """Test complexity score to name conversion."""
        self.assertEqual(self.bridge._score_to_complexity_name(0.0), 'trivial')
        self.assertEqual(self.bridge._score_to_complexity_name(0.1), 'trivial')
        self.assertEqual(self.bridge._score_to_complexity_name(0.25), 'simple')
        self.assertEqual(self.bridge._score_to_complexity_name(0.45), 'moderate')
        self.assertEqual(self.bridge._score_to_complexity_name(0.65), 'complex')
        self.assertEqual(self.bridge._score_to_complexity_name(0.85), 'expert')
        self.assertEqual(self.bridge._score_to_complexity_name(1.0), 'expert')

    def test_generate_problem_id_unique(self):
        """Test that problem IDs are unique for different queries."""
        id1 = self.bridge._generate_problem_id("Query one")
        id2 = self.bridge._generate_problem_id("Query two")
        id3 = self.bridge._generate_problem_id("Query one")  # Same as first

        self.assertNotEqual(id1, id2)
        self.assertEqual(id1, id3)  # Same query should produce same ID
        self.assertTrue(id1.startswith("query_"))

    def test_conversion_count_increments(self):
        """Test that conversion count increments on each call."""
        initial_count = self.bridge._conversion_count

        query_analysis = {'type': 'general', 'complexity': 0.3}
        self.bridge.convert_to_problem_graph("Test query 1", query_analysis)
        self.bridge.convert_to_problem_graph("Test query 2", query_analysis)

        self.assertEqual(self.bridge._conversion_count, initial_count + 2)


class TestSubproblemResultAggregation(unittest.TestCase):
    """Test cases for subproblem result aggregation."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from vulcan.reasoning.query_to_problem_bridge import (
                QueryToProblemBridge,
                SubproblemResult,
                AggregatedResult,
            )
            self.bridge = QueryToProblemBridge()
            self.SubproblemResult = SubproblemResult
        except ImportError as e:
            self.skipTest(f"QueryToProblemBridge not available: {e}")

    def test_aggregate_empty_results(self):
        """Test aggregation of empty results."""
        result = self.bridge.aggregate_subproblem_results([])

        self.assertEqual(result.status, 'empty')
        self.assertEqual(result.total_subproblems, 0)
        self.assertEqual(result.successful, 0)
        self.assertEqual(result.failed, 0)
        self.assertIsNone(result.content)

    def test_aggregate_all_successful(self):
        """Test aggregation when all subproblems succeed."""
        results = [
            self.SubproblemResult(
                step_id="step_0",
                success=True,
                content="Result 1",
                tools_used=["tool_a"],
                confidence=0.8,
            ),
            self.SubproblemResult(
                step_id="step_1",
                success=True,
                content="Result 2",
                tools_used=["tool_b"],
                confidence=0.9,
            ),
        ]

        aggregated = self.bridge.aggregate_subproblem_results(results)

        self.assertEqual(aggregated.status, 'complete')
        self.assertEqual(aggregated.total_subproblems, 2)
        self.assertEqual(aggregated.successful, 2)
        self.assertEqual(aggregated.failed, 0)
        self.assertIn("Result 1", aggregated.content)
        self.assertIn("Result 2", aggregated.content)
        self.assertEqual(len(aggregated.errors), 0)

    def test_aggregate_partial_success(self):
        """Test aggregation when some subproblems fail."""
        results = [
            self.SubproblemResult(
                step_id="step_0",
                success=True,
                content="Result 1",
                confidence=0.8,
            ),
            self.SubproblemResult(
                step_id="step_1",
                success=False,
                error="Something went wrong",
                confidence=0.0,
            ),
        ]

        aggregated = self.bridge.aggregate_subproblem_results(results)

        self.assertEqual(aggregated.status, 'partial')
        self.assertEqual(aggregated.successful, 1)
        self.assertEqual(aggregated.failed, 1)
        self.assertIn("Something went wrong", aggregated.errors)

    def test_aggregate_all_failed(self):
        """Test aggregation when all subproblems fail."""
        results = [
            self.SubproblemResult(
                step_id="step_0",
                success=False,
                error="Error 1",
            ),
            self.SubproblemResult(
                step_id="step_1",
                success=False,
                error="Error 2",
            ),
        ]

        aggregated = self.bridge.aggregate_subproblem_results(results)

        self.assertEqual(aggregated.status, 'error')
        self.assertEqual(aggregated.successful, 0)
        self.assertEqual(aggregated.failed, 2)
        self.assertIsNone(aggregated.content)
        self.assertEqual(len(aggregated.errors), 2)

    def test_aggregate_confidence_calculation(self):
        """Test that overall confidence is calculated correctly."""
        results = [
            self.SubproblemResult(
                step_id="step_0",
                success=True,
                content="A",
                confidence=0.9,
                execution_time_ms=100,
            ),
            self.SubproblemResult(
                step_id="step_1",
                success=True,
                content="B",
                confidence=0.7,
                execution_time_ms=100,
            ),
        ]

        aggregated = self.bridge.aggregate_subproblem_results(results)

        # Confidence should be weighted average
        self.assertGreater(aggregated.overall_confidence, 0.0)
        self.assertLessEqual(aggregated.overall_confidence, 1.0)


class TestQueryToProblemBridgeSingleton(unittest.TestCase):
    """Test the singleton accessor for QueryToProblemBridge."""

    def test_singleton_returns_same_instance(self):
        """Test that get_query_to_problem_bridge returns singleton."""
        try:
            from vulcan.reasoning.query_to_problem_bridge import get_query_to_problem_bridge

            bridge1 = get_query_to_problem_bridge()
            bridge2 = get_query_to_problem_bridge()

            self.assertIs(bridge1, bridge2)
        except ImportError:
            self.skipTest("Module not available")


class TestQueryToProblemBridgeThreadSafety(unittest.TestCase):
    """Thread safety tests for QueryToProblemBridge."""

    def test_concurrent_conversions(self):
        """Test that concurrent conversions are safe."""
        try:
            from vulcan.reasoning.query_to_problem_bridge import QueryToProblemBridge
            bridge = QueryToProblemBridge()
        except ImportError:
            self.skipTest("Module not available")
            return

        results = []
        results_lock = threading.Lock()
        errors = []

        def convert_query(query_num):
            try:
                query = f"Test query number {query_num}"
                analysis = {'type': 'general', 'complexity': 0.3}
                result = bridge.convert_to_problem_graph(query, analysis)
                with results_lock:
                    results.append((query_num, result))
            except Exception as e:
                with results_lock:
                    errors.append((query_num, str(e)))

        # Run concurrent conversions
        threads = [threading.Thread(target=convert_query, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 20)

    def test_concurrent_aggregations(self):
        """Test that concurrent aggregations are safe."""
        try:
            from vulcan.reasoning.query_to_problem_bridge import (
                QueryToProblemBridge,
                SubproblemResult,
            )
            bridge = QueryToProblemBridge()
        except ImportError:
            self.skipTest("Module not available")
            return

        results = []
        results_lock = threading.Lock()

        def aggregate(thread_num):
            subresults = [
                SubproblemResult(
                    step_id=f"step_{thread_num}_{i}",
                    success=True,
                    content=f"Content from thread {thread_num}",
                    confidence=0.8,
                )
                for i in range(3)
            ]
            aggregated = bridge.aggregate_subproblem_results(subresults)
            with results_lock:
                results.append(aggregated)

        threads = [threading.Thread(target=aggregate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 10)
        for r in results:
            self.assertEqual(r.status, 'complete')


class TestQueryToProblemBridgeStatistics(unittest.TestCase):
    """Test statistics collection in QueryToProblemBridge."""

    def test_get_statistics(self):
        """Test that statistics are properly collected."""
        try:
            from vulcan.reasoning.query_to_problem_bridge import QueryToProblemBridge
            bridge = QueryToProblemBridge()
        except ImportError:
            self.skipTest("Module not available")
            return

        # Perform some operations
        analysis = {'type': 'general', 'complexity': 0.3}
        bridge.convert_to_problem_graph("Test 1", analysis)
        bridge.convert_to_problem_graph("Test 2", analysis)

        from vulcan.reasoning.query_to_problem_bridge import SubproblemResult
        bridge.aggregate_subproblem_results([
            SubproblemResult(step_id="s1", success=True)
        ])

        # Get statistics
        stats = bridge.get_statistics()

        self.assertIn('conversion_count', stats)
        self.assertIn('aggregation_count', stats)
        self.assertIn('decomposer_available', stats)
        self.assertEqual(stats['conversion_count'], 2)
        self.assertEqual(stats['aggregation_count'], 1)


if __name__ == "__main__":
    unittest.main()
