"""
Unit Test for the VULCAN Reasoning Selection Submodule

This test file provides granular, focused tests for each major component
of the `vulcan.reasoning.selection` package, including:
- AdmissionControl: Verifies rate limiting and overload protection.
- PortfolioExecutor: Verifies different multi-tool execution strategies.
- WarmPool: Verifies tool instance acquisition and release.
- UtilityModel: Verifies context-dependent utility calculations.

This is a complement to the broader integration test.

To run: `pytest src/vulcan/tests/test_selection_submodule.py`
"""

import logging
import sys
import time
import unittest
from pathlib import Path


# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from vulcan.reasoning.selection.admission_control import (
        AdmissionControlIntegration,
        Request,
        RequestPriority,
    )
    from vulcan.reasoning.selection.portfolio_executor import (
        ExecutionStrategy,
        PortfolioExecutor,
        PortfolioResult,
    )
    from vulcan.reasoning.selection.utility_model import (
        ContextMode,
        UtilityContext,
        UtilityModel,
    )
    from vulcan.reasoning.selection.warm_pool import WarmStartPool
except ImportError as e:
    print(f"Failed to import selection modules: {e}")
    print(
        "Please ensure this test script is placed correctly and all dependencies are available."
    )
    sys.exit(1)

logger = logging.getLogger(__name__)


# --- Mock Tools for Testing ---
class MockTool:
    """A mock tool that simulates work by sleeping."""

    def __init__(self, name: str, execution_time: float, confidence: float):
        self.name = name
        self.execution_time = execution_time
        self.confidence = confidence

    def reason(self, problem: any):
        time.sleep(self.execution_time)
        return {"result": f"Result from {self.name}", "confidence": self.confidence}


def mock_tool_factory(name, exec_time, conf):
    """A factory function to create mock tools for the warm pool."""
    return MockTool(name, exec_time, conf)


class TestAdmissionControl(unittest.TestCase):
    """Tests for the AdmissionControlIntegration."""

    def setUp(self):
        # High rate limit for most tests, low for specific rate limit test
        config = {"global_rate": 1000, "burst_capacity": 2000}
        self.admission_control = AdmissionControlIntegration(config)

    def tearDown(self):
        self.admission_control.shutdown()

    def test_admit_healthy_system(self):
        logger.info("--- Testing AdmissionControl: Healthy System ---")
        request = Request(
            request_id="test-01",
            priority=RequestPriority.NORMAL,
            estimated_cost={"time_ms": 100},
            context={},
        )
        admitted, info = self.admission_control.check_admission(
            problem="test problem", constraints={}, priority=RequestPriority.NORMAL
        )
        self.assertTrue(admitted)
        self.assertIn("estimated_wait_ms", info)

    def test_reject_rate_limit(self):
        logger.info("--- Testing AdmissionControl: Rate Limiting ---")
        # Use a controller with a very low rate limit
        low_rate_config = {"global_rate": 2, "burst_capacity": 2}
        limiter = AdmissionControlIntegration(low_rate_config)

        # Consume the burst capacity
        self.assertTrue(limiter.check_admission("p1", {})[0])
        self.assertTrue(limiter.check_admission("p2", {})[0])

        # This one should be rejected
        admitted, info = limiter.check_admission("p3", {})
        self.assertFalse(admitted)
        self.assertEqual(info.get("reason"), "rate_limit_exceeded")
        limiter.shutdown()

    def test_reject_overloaded_system(self):
        logger.info("--- Testing AdmissionControl: CPU Overload ---")

        # OPTION 1: Directly inject high CPU values into the history
        # This is the most reliable approach since it bypasses the background thread
        with self.admission_control.controller.resource_monitor._history_lock:
            # Clear existing history and inject high CPU values
            self.admission_control.controller.resource_monitor.cpu_history.clear()
            self.admission_control.controller.resource_monitor.memory_history.clear()

            # Fill with high CPU values to simulate overload
            for _ in range(60):  # Fill the entire deque
                self.admission_control.controller.resource_monitor.cpu_history.append(
                    96.0
                )
                self.admission_control.controller.resource_monitor.memory_history.append(
                    50.0
                )

        # Set the threshold
        self.admission_control.controller.resource_monitor.cpu_threshold = 90.0

        # Now check system health
        health = self.admission_control.controller._check_system_health()
        self.assertIn(health.value, ["overloaded", "critical"])

        # Test admission
        admitted, info = self.admission_control.check_admission("problem", {})
        self.assertFalse(admitted)
        self.assertIn(info.get("reason"), ["system_critical", "insufficient_resources"])


class TestPortfolioExecutor(unittest.TestCase):
    """Tests for the PortfolioExecutor."""

    def setUp(self):
        self.mock_tools = {
            "fast_tool": MockTool(name="fast_tool", execution_time=0.1, confidence=0.8),
            "slow_tool": MockTool(
                name="slow_tool", execution_time=0.5, confidence=0.95
            ),
            "cheap_tool": MockTool(
                name="cheap_tool", execution_time=0.05, confidence=0.6
            ),
        }
        self.executor = PortfolioExecutor(tools=self.mock_tools, max_workers=4)

    def tearDown(self):
        self.executor.shutdown()

    def test_execute_single(self):
        logger.info("--- Testing PortfolioExecutor: SINGLE Strategy ---")
        result = self.executor.execute(
            strategy=ExecutionStrategy.SINGLE,
            tool_names=["fast_tool"],
            problem="test",
            constraints={"time_budget_ms": 1000},
        )
        self.assertIsInstance(result, PortfolioResult)
        self.assertEqual(len(result.tools_used), 1)
        self.assertEqual(result.tools_used[0], "fast_tool")
        self.assertIn("Result from fast_tool", str(result.primary_result))

    def test_speculative_parallel_fastest_wins(self):
        logger.info("--- Testing PortfolioExecutor: SPECULATIVE_PARALLEL Strategy ---")
        # The fast tool has a confidence (0.8) that meets the minimum (0.7),
        # so it should terminate early without waiting for the slow tool.
        result = self.executor.execute(
            strategy=ExecutionStrategy.SPECULATIVE_PARALLEL,
            tool_names=["slow_tool", "fast_tool"],
            problem="test",
            constraints={"time_budget_ms": 2000, "min_confidence": 0.7},
        )
        self.assertIsInstance(result, PortfolioResult)
        self.assertLess(
            result.execution_time, 0.4
        )  # Should be much faster than the slow tool (0.5s)
        self.assertEqual(result.primary_result["result"], "Result from fast_tool")
        self.assertTrue(result.metadata.get("early_termination"))


class TestWarmPool(unittest.TestCase):
    """Tests for the WarmStartPool."""

    def setUp(self):
        self.mock_tool_factories = {
            "test_tool": lambda: mock_tool_factory("test_tool", 0.01, 0.9)
        }
        self.pool = WarmStartPool(
            tools=self.mock_tool_factories,
            config={"min_pool_size": 1, "max_pool_size": 2},
        )
        time.sleep(0.5)  # Allow time for initial instance to warm up

    def tearDown(self):
        self.pool.shutdown()

    def test_acquire_and_release(self):
        logger.info("--- Testing WarmPool: Acquire and Release ---")
        stats_before = self.pool.pools["test_tool"].get_statistics()
        self.assertGreaterEqual(stats_before["ready"], 1)

        # Acquire
        instance_id, tool_instance = self.pool.acquire_tool("test_tool", timeout=1.0)
        self.assertIsNotNone(instance_id)
        self.assertIsNotNone(tool_instance)

        stats_during = self.pool.pools["test_tool"].get_statistics()
        self.assertEqual(stats_during["busy"], 1)

        # Release
        self.pool.release_tool("test_tool", instance_id)

        stats_after = self.pool.pools["test_tool"].get_statistics()
        self.assertEqual(stats_after["busy"], 0)
        self.assertGreaterEqual(stats_after["ready"], 1)


class TestUtilityModel(unittest.TestCase):
    """Tests for the UtilityModel."""

    def setUp(self):
        self.utility_model = UtilityModel()

    def test_rush_mode_prioritizes_time(self):
        logger.info("--- Testing UtilityModel: RUSH Mode ---")
        context = UtilityContext(
            mode=ContextMode.RUSH,
            time_budget=1000,
            energy_budget=1000,
            min_quality=0.5,
            max_risk=0.5,
        )

        # Option 1: Fast, low quality
        utility_fast = self.utility_model.compute_utility(
            quality=0.6, time=100, energy=100, risk=0.4, context=context
        )

        # Option 2: Slow, high quality
        utility_slow = self.utility_model.compute_utility(
            quality=0.95, time=900, energy=500, risk=0.1, context=context
        )

        self.assertGreater(
            utility_fast,
            utility_slow,
            "In RUSH mode, the faster option should have higher utility.",
        )

    def test_accurate_mode_prioritizes_quality(self):
        logger.info("--- Testing UtilityModel: ACCURATE Mode ---")
        context = UtilityContext(
            mode=ContextMode.ACCURATE,
            time_budget=1000,
            energy_budget=1000,
            min_quality=0.5,
            max_risk=0.5,
        )

        # Option 1: Fast, low quality
        utility_fast = self.utility_model.compute_utility(
            quality=0.6, time=100, energy=100, risk=0.4, context=context
        )

        # Option 2: Slow, high quality
        utility_slow = self.utility_model.compute_utility(
            quality=0.95, time=900, energy=500, risk=0.1, context=context
        )

        self.assertGreater(
            utility_slow,
            utility_fast,
            "In ACCURATE mode, the higher quality option should have higher utility.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
