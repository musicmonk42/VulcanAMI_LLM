# ============================================================
# VULCAN-AGI Orchestrator - Variants Tests (FIXED)
# Comprehensive test suite for variants.py
# ============================================================

import unittest
import sys
import time
import asyncio
import threading
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Add src directory to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import components to test
from vulcan.orchestrator.variants import (
    ParallelOrchestrator,
    FaultTolerantOrchestrator,
    AdaptiveOrchestrator,
    PerformanceMonitor,
    StrategySelector,
    PerceptionError,
    ReasoningError,
    ExecutionError,
    shutdown_executor_with_timeout,
    SUPPORTS_EXECUTOR_TIMEOUT,
    PYTHON_VERSION,
)


# ============================================================
# TEST HELPERS
# ============================================================


def create_mock_config():
    """Create a properly configured mock config"""
    config = Mock()
    config.max_parallel_processes = 2
    config.max_parallel_threads = 4
    config.max_retries = 3
    config.performance_window_size = 50

    # Agent pool configuration
    config.min_agents = 1
    config.max_agents = 4
    config.task_queue_type = "zmq"
    config.agent_lifecycle_tracking = True
    config.agent_health_check_interval = 60

    return config


def create_mock_deps():
    """Create a properly configured mock dependencies"""
    deps = Mock()

    # Metrics
    deps.metrics = Mock()
    deps.metrics.record_step = Mock()
    deps.metrics.increment_counter = Mock()

    # Multimodal
    deps.multimodal = Mock()
    deps.multimodal.clear_cache = Mock()

    # Long-term memory
    deps.ltm = Mock()
    deps.ltm.search = Mock(return_value=[])

    # Compressed memory
    deps.compressed_memory = Mock()
    deps.compressed_memory.compress_batch = Mock()

    return deps


# ============================================================
# TEST: CUSTOM EXCEPTIONS
# ============================================================


class TestCustomExceptions(unittest.TestCase):
    """Test custom exception classes"""

    def test_perception_error(self):
        """Test PerceptionError exception"""
        error = PerceptionError("Test perception error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test perception error")

    def test_reasoning_error(self):
        """Test ReasoningError exception"""
        error = ReasoningError("Test reasoning error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test reasoning error")

    def test_execution_error(self):
        """Test ExecutionError exception"""
        error = ExecutionError("Test execution error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test execution error")


# ============================================================
# TEST: EXECUTOR SHUTDOWN HELPER
# ============================================================


def simple_task():
    """Simple task for process executor"""
    return 42


class TestExecutorShutdown(unittest.TestCase):
    """Test executor shutdown helper function"""

    def test_shutdown_thread_executor(self):
        """Test shutting down ThreadPoolExecutor"""
        executor = ThreadPoolExecutor(max_workers=2)

        # Submit a simple task
        future = executor.submit(lambda: time.sleep(0.1))
        future.result()

        # Shutdown with timeout
        shutdown_executor_with_timeout(executor, "Test ThreadExecutor", timeout=2.0)

        # Verify executor is shutdown by attempting to submit a new task
        with self.assertRaises(RuntimeError):
            executor.submit(lambda: None)

    def test_shutdown_process_executor(self):
        """Test shutting down ProcessPoolExecutor"""
        executor = ProcessPoolExecutor(max_workers=2)

        # Submit a simple task (use a module-level function for pickling)
        future = executor.submit(simple_task)
        result = future.result()
        self.assertEqual(result, 42)

        # Shutdown with timeout
        shutdown_executor_with_timeout(executor, "Test ProcessExecutor", timeout=2.0)

        # Verify executor is shutdown by attempting to submit a new task
        with self.assertRaises(RuntimeError):
            executor.submit(simple_task)

    def test_shutdown_already_shutdown_executor(self):
        """Test shutting down an already shutdown executor"""
        executor = ThreadPoolExecutor(max_workers=2)
        executor.shutdown(wait=True)

        # Should not raise exception
        shutdown_executor_with_timeout(executor, "Test Executor", timeout=1.0)

    def test_python_version_flag(self):
        """Test Python version flag"""
        self.assertIsInstance(SUPPORTS_EXECUTOR_TIMEOUT, bool)
        self.assertIsNotNone(PYTHON_VERSION)
        self.assertGreaterEqual(PYTHON_VERSION.major, 3)


# ============================================================
# TEST: PERFORMANCE MONITOR
# ============================================================


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class"""

    def test_initialization(self):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor(window_size=50)

        self.assertEqual(monitor.window_size, 50)
        self.assertEqual(len(monitor.metrics_history), 0)
        self.assertIsNotNone(monitor._lock)

    def test_record_metrics(self):
        """Test recording metrics"""
        monitor = PerformanceMonitor(window_size=10)

        metrics = {"latency": 100.0, "error": False, "reward": 0.8}

        monitor.record(metrics)

        self.assertEqual(len(monitor.metrics_history), 1)
        self.assertIn("timestamp", monitor.metrics_history[0])
        self.assertEqual(monitor.metrics_history[0]["latency"], 100.0)

    def test_window_size_limit(self):
        """Test that metrics history is bounded by window size"""
        monitor = PerformanceMonitor(window_size=5)

        # Add more metrics than window size
        for i in range(10):
            monitor.record({"value": i})

        # Should only keep last 5
        self.assertEqual(len(monitor.metrics_history), 5)
        self.assertEqual(monitor.metrics_history[-1]["value"], 9)

    def test_get_recent_metrics_empty(self):
        """Test getting metrics from empty history"""
        monitor = PerformanceMonitor()

        metrics = monitor.get_recent_metrics()

        self.assertEqual(metrics["avg_latency"], 0.0)
        self.assertEqual(metrics["error_rate"], 0.0)
        self.assertEqual(metrics["avg_reward"], 0.0)
        self.assertEqual(metrics["uncertainty"], 0.5)

    def test_get_recent_metrics_with_data(self):
        """Test getting metrics with data"""
        monitor = PerformanceMonitor()

        # Add some metrics
        for i in range(5):
            monitor.record(
                {
                    "latency": 100.0 + i * 10,
                    "error": i % 2 == 0,
                    "reward": 0.5 + i * 0.1,
                    "uncertainty": 0.3,
                }
            )

        metrics = monitor.get_recent_metrics(n=5)

        self.assertGreater(metrics["avg_latency"], 0)
        self.assertGreater(metrics["error_rate"], 0)
        self.assertGreater(metrics["avg_reward"], 0)
        self.assertEqual(metrics["uncertainty"], 0.3)

    def test_thread_safety(self):
        """Test thread-safe recording"""
        monitor = PerformanceMonitor()

        def record_metrics():
            for i in range(10):
                monitor.record({"value": i})

        threads = [threading.Thread(target=record_metrics) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have recorded all metrics
        self.assertEqual(len(monitor.metrics_history), 50)


# ============================================================
# TEST: STRATEGY SELECTOR
# ============================================================


class TestStrategySelector(unittest.TestCase):
    """Test StrategySelector class"""

    def test_initialization(self):
        """Test StrategySelector initialization"""
        selector = StrategySelector()
        self.assertIsNotNone(selector)

    def test_select_strategy_empty_metrics(self):
        """Test strategy selection with empty metrics"""
        selector = StrategySelector()

        strategy = selector.select_strategy({})
        self.assertEqual(strategy, "balanced")

    def test_select_strategy_high_error_rate(self):
        """Test strategy selection with high error rate"""
        selector = StrategySelector()

        metrics = {"error_rate": 0.15, "avg_latency": 100, "avg_reward": 0.5}

        strategy = selector.select_strategy(metrics)
        self.assertEqual(strategy, "careful")

    def test_select_strategy_high_latency(self):
        """Test strategy selection with high latency"""
        selector = StrategySelector()

        metrics = {"error_rate": 0.05, "avg_latency": 1500, "avg_reward": 0.5}

        strategy = selector.select_strategy(metrics)
        self.assertEqual(strategy, "fast")

    def test_select_strategy_low_reward(self):
        """Test strategy selection with low reward"""
        selector = StrategySelector()

        metrics = {"error_rate": 0.05, "avg_latency": 100, "avg_reward": 0.2}

        strategy = selector.select_strategy(metrics)
        self.assertEqual(strategy, "exploratory")

    def test_select_strategy_balanced(self):
        """Test strategy selection with balanced metrics"""
        selector = StrategySelector()

        metrics = {"error_rate": 0.05, "avg_latency": 500, "avg_reward": 0.7}

        strategy = selector.select_strategy(metrics)
        self.assertEqual(strategy, "balanced")


# ============================================================
# TEST: PARALLEL ORCHESTRATOR
# ============================================================


class TestParallelOrchestrator(unittest.TestCase):
    """Test ParallelOrchestrator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = create_mock_config()
        self.mock_sys = Mock()
        self.mock_sys.provenance_chain = []
        self.mock_deps = create_mock_deps()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_initialization(self, mock_agent_pool):
        """Test ParallelOrchestrator initialization"""
        orchestrator = ParallelOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        self.assertIsNotNone(orchestrator)
        self.assertIsNotNone(orchestrator.process_executor)
        self.assertIsNotNone(orchestrator.thread_executor)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_shutdown(self, mock_agent_pool):
        """Test ParallelOrchestrator shutdown"""
        orchestrator = ParallelOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Shutdown should not raise exception
        orchestrator.shutdown()

        # Verify executors are shut down by attempting to submit new tasks
        with self.assertRaises(RuntimeError):
            orchestrator.process_executor.submit(simple_task)

        with self.assertRaises(RuntimeError):
            orchestrator.thread_executor.submit(lambda: None)

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_destructor(self, mock_agent_pool):
        """Test ParallelOrchestrator destructor"""
        orchestrator = ParallelOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Trigger destructor
        orchestrator.__del__()

        # Should not raise exception
        self.assertTrue(True)

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_update_memory_async(self, mock_agent_pool):
        """Test async memory update"""
        orchestrator = ParallelOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        orchestrator.execution_history = deque(maxlen=100)

        # Add some history
        for i in range(15):
            orchestrator.execution_history.append({"step": i})

        # Update memory
        result = orchestrator._update_memory_async([])

        # Should complete successfully
        self.assertTrue(result)

        # Cleanup
        orchestrator.shutdown()


# ============================================================
# TEST: FAULT TOLERANT ORCHESTRATOR
# ============================================================


class TestFaultTolerantOrchestrator(unittest.TestCase):
    """Test FaultTolerantOrchestrator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = create_mock_config()
        self.mock_sys = Mock()
        self.mock_sys.provenance_chain = []
        self.mock_deps = create_mock_deps()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_initialization(self, mock_agent_pool):
        """Test FaultTolerantOrchestrator initialization"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        self.assertIsNotNone(orchestrator)
        self.assertIsNotNone(orchestrator.fallback_strategies)
        self.assertIsNotNone(orchestrator.error_history)
        self.assertEqual(len(orchestrator.error_history), 0)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_step_with_recovery_success(self, mock_agent_pool):
        """Test successful execution with recovery"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Mock successful step
        orchestrator.step = Mock(return_value={"success": True})

        result = orchestrator.step_with_recovery([], {})

        self.assertTrue(result["success"])
        self.assertEqual(len(orchestrator.error_history), 1)
        self.assertTrue(orchestrator.error_history[0]["success"])

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_step_with_recovery_perception_error(self, mock_agent_pool):
        """Test recovery from PerceptionError"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Mock step to raise PerceptionError then succeed
        call_count = [0]

        def step_side_effect(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PerceptionError("Test error")
            return {"success": True}

        orchestrator.step = Mock(side_effect=step_side_effect)
        orchestrator._perception_fallback = Mock(return_value=None)

        result = orchestrator.step_with_recovery([], {})

        # Should have recovered
        self.assertTrue(result["success"])
        self.assertGreater(len(orchestrator.error_history), 0)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_get_error_statistics_empty(self, mock_agent_pool):
        """Test error statistics with no errors"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        stats = orchestrator.get_error_statistics()

        self.assertEqual(stats["total_attempts"], 0)
        self.assertEqual(stats["total_errors"], 0)
        self.assertEqual(stats["success_rate"], 0.0)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_get_error_statistics_with_errors(self, mock_agent_pool):
        """Test error statistics with errors"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Add some error history
        orchestrator.error_history.append(
            {"timestamp": time.time(), "attempt": 0, "success": True}
        )
        orchestrator.error_history.append(
            {
                "timestamp": time.time(),
                "attempt": 0,
                "error_type": "perception",
                "error": "test",
            }
        )
        orchestrator.error_history.append(
            {"timestamp": time.time(), "attempt": 0, "success": True}
        )

        stats = orchestrator.get_error_statistics()

        self.assertEqual(stats["total_attempts"], 3)
        self.assertEqual(stats["total_errors"], 1)
        self.assertAlmostEqual(stats["success_rate"], 2 / 3, places=2)
        self.assertIn("perception", stats["error_types"])

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_perception_fallback(self, mock_agent_pool):
        """Test perception fallback"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Mock dependencies
        orchestrator.deps.ltm.search = Mock(return_value=[])

        result = orchestrator._perception_fallback(Exception("test"))

        # Should return None when no recent memories
        self.assertIsNone(result)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_reasoning_fallback(self, mock_agent_pool):
        """Test reasoning fallback"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Mock methods
        orchestrator._create_wait_plan = Mock(return_value={"action": "wait"})
        orchestrator._validate_and_ensure_safety = Mock(
            return_value={"action": "wait", "safe": True}
        )
        orchestrator._execute_action = Mock(return_value={"success": True})

        result = orchestrator._reasoning_fallback(Exception("test"))

        # Should return result
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_execution_fallback(self, mock_agent_pool):
        """Test execution fallback"""
        orchestrator = FaultTolerantOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        orchestrator._create_fallback_result = Mock(return_value={"fallback": True})

        result = orchestrator._execution_fallback(Exception("test"))

        # Should return fallback result
        self.assertIsNotNone(result)

        # Cleanup
        orchestrator.shutdown()


# ============================================================
# TEST: ADAPTIVE ORCHESTRATOR
# ============================================================


class TestAdaptiveOrchestrator(unittest.TestCase):
    """Test AdaptiveOrchestrator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = create_mock_config()
        self.mock_sys = Mock()
        self.mock_sys.provenance_chain = []
        self.mock_deps = create_mock_deps()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_initialization(self, mock_agent_pool):
        """Test AdaptiveOrchestrator initialization"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        self.assertIsNotNone(orchestrator)
        self.assertIsNotNone(orchestrator.performance_monitor)
        self.assertIsNotNone(orchestrator.strategy_selector)
        self.assertIsNotNone(orchestrator.adaptation_history)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_adaptive_step_with_mocks(self, mock_agent_pool):
        """Test adaptive step with mocked execution"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Mock methods
        orchestrator.step = Mock(return_value={"success": True, "reward": 0.8})
        orchestrator._fast_step = Mock(return_value={"success": True, "reward": 0.7})
        orchestrator._careful_step = Mock(return_value={"success": True, "reward": 0.9})
        orchestrator._exploratory_step = Mock(
            return_value={"success": True, "reward": 0.6}
        )

        # Execute adaptive step
        result = orchestrator.adaptive_step([], {})

        # Should have executed successfully
        self.assertTrue(result["success"])

        # Should have recorded adaptation
        self.assertGreater(len(orchestrator.adaptation_history), 0)

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_fast_step(self, mock_agent_pool):
        """Test fast execution mode"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        orchestrator.step = Mock(return_value={"success": True})

        result = orchestrator._fast_step([], {})

        # Should have called step with modified context
        self.assertTrue(result["success"])
        orchestrator.step.assert_called_once()

        # Check that context was modified
        call_args = orchestrator.step.call_args[0]
        context = call_args[1]
        self.assertEqual(context.get("quality"), "fast")

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_careful_step(self, mock_agent_pool):
        """Test careful execution mode"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        orchestrator.step = Mock(return_value={"success": True})

        result = orchestrator._careful_step([], {})

        # Should have called step with modified context
        self.assertTrue(result["success"])
        orchestrator.step.assert_called_once()

        # Check that context was modified
        call_args = orchestrator.step.call_args[0]
        context = call_args[1]
        self.assertEqual(context.get("quality"), "high")
        self.assertEqual(context.get("safety_level"), "strict")

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_exploratory_step(self, mock_agent_pool):
        """Test exploratory execution mode"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        orchestrator.step = Mock(return_value={"success": True})

        result = orchestrator._exploratory_step([], {})

        # Should have called step with modified context
        self.assertTrue(result["success"])
        orchestrator.step.assert_called_once()

        # Check that context was modified
        call_args = orchestrator.step.call_args[0]
        context = call_args[1]
        self.assertEqual(context.get("high_level_goal"), "explore")

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_get_adaptation_statistics_empty(self, mock_agent_pool):
        """Test adaptation statistics with no adaptations"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        stats = orchestrator.get_adaptation_statistics()

        self.assertEqual(stats["total_adaptations"], 0)
        self.assertEqual(stats["strategy_distribution"], {})
        self.assertEqual(stats["current_strategy"], "balanced")

        # Cleanup
        orchestrator.shutdown()

    @patch("vulcan.orchestrator.collective.AgentPoolManager")
    def test_get_adaptation_statistics_with_data(self, mock_agent_pool):
        """Test adaptation statistics with data"""
        orchestrator = AdaptiveOrchestrator(
            self.mock_config, self.mock_sys, self.mock_deps
        )

        # Add some adaptations
        orchestrator.adaptation_history.append(
            {"strategy": "fast", "metrics": {}, "timestamp": time.time()}
        )
        orchestrator.adaptation_history.append(
            {"strategy": "careful", "metrics": {}, "timestamp": time.time()}
        )
        orchestrator.adaptation_history.append(
            {"strategy": "fast", "metrics": {}, "timestamp": time.time()}
        )

        stats = orchestrator.get_adaptation_statistics()

        self.assertEqual(stats["total_adaptations"], 3)
        self.assertEqual(stats["strategy_distribution"]["fast"], 2)
        self.assertEqual(stats["strategy_distribution"]["careful"], 1)
        self.assertEqual(stats["current_strategy"], "fast")

        # Cleanup
        orchestrator.shutdown()


# ============================================================
# TEST SUITE RUNNER
# ============================================================


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestCustomExceptions))
    test_suite.addTest(unittest.makeSuite(TestExecutorShutdown))
    test_suite.addTest(unittest.makeSuite(TestPerformanceMonitor))
    test_suite.addTest(unittest.makeSuite(TestStrategySelector))
    test_suite.addTest(unittest.makeSuite(TestParallelOrchestrator))
    test_suite.addTest(unittest.makeSuite(TestFaultTolerantOrchestrator))
    test_suite.addTest(unittest.makeSuite(TestAdaptiveOrchestrator))

    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
