"""
Comprehensive test suite for portfolio_executor.py

Tests all execution strategies, timeout handling, resource management,
and proper future cleanup.
"""

# Import the module to test
from vulcan.reasoning.selection.portfolio_executor import (ExecutionMonitor,
                                                           ExecutionStatus,
                                                           ExecutionStrategy,
                                                           PortfolioExecutor,
                                                           PortfolioResult,
                                                           ToolExecution)
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Mock tool classes for testing
@dataclass
class MockResult:
    """Mock result from a tool"""

    confidence: float
    value: Any

    def __str__(self):
        return f"MockResult(confidence={self.confidence}, value={self.value})"


class MockTool:
    """Mock tool for testing"""

    def __init__(
        self,
        name: str,
        execution_time: float = 0.1,
        confidence: float = 0.8,
        should_fail: bool = False,
    ):
        self.name = name
        self.execution_time = execution_time
        self.confidence = confidence
        self.should_fail = should_fail
        self.call_count = 0

    def reason(self, problem: Any) -> MockResult:
        """Mock reasoning method"""
        self.call_count += 1

        if self.should_fail:
            raise ValueError(f"{self.name} failed as expected")

        time.sleep(self.execution_time)

        return MockResult(
            confidence=self.confidence, value=f"{self.name}_result_{problem}"
        )

    def __call__(self, problem: Any) -> MockResult:
        """Allow tool to be called directly"""
        return self.reason(problem)


class TestEnums:
    """Test enum definitions"""

    def test_execution_strategy_values(self):
        """Test ExecutionStrategy enum"""
        assert ExecutionStrategy.SINGLE.value == "single"
        assert ExecutionStrategy.SPECULATIVE_PARALLEL.value == "speculative_parallel"
        assert ExecutionStrategy.SEQUENTIAL_REFINEMENT.value == "sequential_refinement"
        assert ExecutionStrategy.COMMITTEE_CONSENSUS.value == "committee_consensus"
        assert ExecutionStrategy.CASCADE.value == "cascade"
        assert ExecutionStrategy.TOURNAMENT.value == "tournament"
        assert ExecutionStrategy.ADAPTIVE_MIX.value == "adaptive_mix"

    def test_execution_status_values(self):
        """Test ExecutionStatus enum"""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.TIMEOUT.value == "timeout"


class TestToolExecution:
    """Test ToolExecution dataclass"""

    def test_creation(self):
        """Test creating tool execution"""
        tool = MockTool("test_tool")

        execution = ToolExecution(
            tool_name="test_tool", tool_instance=tool, problem="test_problem"
        )

        assert execution.tool_name == "test_tool"
        assert execution.tool_instance == tool
        assert execution.problem == "test_problem"
        assert execution.status == ExecutionStatus.PENDING

    def test_defaults(self):
        """Test default values"""
        tool = MockTool("test_tool")

        execution = ToolExecution(
            tool_name="test_tool", tool_instance=tool, problem="test"
        )

        assert execution.end_time is None
        assert execution.result is None
        assert execution.error is None
        assert execution.future is None
        assert isinstance(execution.metadata, dict)


class TestPortfolioResult:
    """Test PortfolioResult dataclass"""

    def test_creation(self):
        """Test creating portfolio result"""
        result = PortfolioResult(
            strategy=ExecutionStrategy.SINGLE,
            primary_result="result",
            all_results={"tool_a": "result"},
            execution_time=1.5,
            energy_used=100.0,
            tools_used=["tool_a"],
        )

        assert result.strategy == ExecutionStrategy.SINGLE
        assert result.primary_result == "result"
        assert result.execution_time == 1.5
        assert result.tools_used == ["tool_a"]


class TestExecutionMonitor:
    """Test ExecutionMonitor"""

    def test_initialization(self):
        """Test monitor initialization"""
        monitor = ExecutionMonitor(
            time_budget_ms=5000, energy_budget_mj=1000, min_confidence=0.7
        )

        assert monitor.time_budget == 5.0  # Converted to seconds
        assert monitor.energy_budget == 1000
        assert monitor.min_confidence == 0.7

    def test_time_remaining(self):
        """Test time remaining calculation"""
        monitor = ExecutionMonitor(time_budget_ms=1000, energy_budget_mj=1000)

        time.sleep(0.1)

        remaining = monitor.time_remaining()
        assert remaining < 1.0
        assert remaining > 0.8

    def test_is_timeout(self):
        """Test timeout detection"""
        monitor = ExecutionMonitor(time_budget_ms=100, energy_budget_mj=1000)

        assert not monitor.is_timeout()

        time.sleep(0.15)

        assert monitor.is_timeout()

    def test_energy_tracking(self):
        """Test energy tracking"""
        monitor = ExecutionMonitor(time_budget_ms=5000, energy_budget_mj=1000)

        monitor.record_execution("tool_a", 100, 300)
        assert monitor.energy_remaining() == 700

        monitor.record_execution("tool_b", 100, 500)
        assert monitor.energy_remaining() == 200

    def test_is_energy_exceeded(self):
        """Test energy budget exceeded"""
        monitor = ExecutionMonitor(time_budget_ms=5000, energy_budget_mj=500)

        assert not monitor.is_energy_exceeded()

        monitor.record_execution("tool_a", 100, 600)

        assert monitor.is_energy_exceeded()

    def test_should_continue(self):
        """Test should continue check"""
        monitor = ExecutionMonitor(
            time_budget_ms=1000, energy_budget_mj=1000, min_confidence=0.8
        )

        # Initially should continue
        assert monitor.should_continue()

        # With good result
        good_result = MockResult(confidence=0.9, value="good")
        assert monitor.should_continue(good_result)

        # After timeout
        time.sleep(1.1)
        assert not monitor.should_continue()


class TestPortfolioExecutor:
    """Test PortfolioExecutor"""

    @pytest.fixture
    def tools(self):
        """Create mock tools for testing"""
        return {
            "fast_tool": MockTool("fast_tool", 0.05, 0.7),
            "slow_tool": MockTool("slow_tool", 0.2, 0.9),
            "accurate_tool": MockTool("accurate_tool", 0.15, 0.95),
            "unreliable_tool": MockTool("unreliable_tool", 0.1, 0.5),
        }

    @pytest.fixture
    def executor(self, tools):
        """Create executor for testing"""
        executor = PortfolioExecutor(tools=tools, max_workers=4)
        yield executor
        executor.shutdown(timeout=2.0)

    def test_initialization(self, executor, tools):
        """Test executor initialization"""
        assert executor.tools == tools
        assert executor.max_workers == 4
        assert len(executor.strategies) == 7  # All strategies registered

    def test_single_execution(self, executor):
        """Test single tool execution"""
        constraints = {
            "time_budget_ms": 5000,
            "energy_budget_mj": 1000,
            "min_confidence": 0.5,
        }

        result = executor.execute(
            strategy=ExecutionStrategy.SINGLE,
            tool_names=["fast_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.SINGLE
        assert result.primary_result is not None
        assert "fast_tool" in result.tools_used

    def test_single_execution_timeout(self, executor):
        """Test single execution with timeout"""
        # Very short timeout
        constraints = {
            "time_budget_ms": 10,  # 10ms
            "energy_budget_mj": 1000,
        }

        result = executor.execute(
            strategy=ExecutionStrategy.SINGLE,
            tool_names=["slow_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        # Should timeout or complete
        assert isinstance(result, PortfolioResult)

    def test_speculative_parallel(self, executor):
        """Test speculative parallel execution"""
        constraints = {
            "time_budget_ms": 5000,
            "energy_budget_mj": 2000,
            "min_confidence": 0.6,
        }

        result = executor.execute(
            strategy=ExecutionStrategy.SPECULATIVE_PARALLEL,
            tool_names=["fast_tool", "slow_tool", "accurate_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.SPECULATIVE_PARALLEL
        assert len(result.tools_used) > 0
        # Should have used at least one tool
        assert result.primary_result is not None

    def test_sequential_refinement(self, executor):
        """Test sequential refinement strategy"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 2000}

        result = executor.execute(
            strategy=ExecutionStrategy.SEQUENTIAL_REFINEMENT,
            tool_names=["fast_tool", "accurate_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.SEQUENTIAL_REFINEMENT
        # Should use multiple tools
        assert len(result.tools_used) >= 1

    def test_committee_consensus(self, executor):
        """Test committee consensus strategy"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 3000}

        result = executor.execute(
            strategy=ExecutionStrategy.COMMITTEE_CONSENSUS,
            tool_names=["fast_tool", "slow_tool", "accurate_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.COMMITTEE_CONSENSUS
        assert result.consensus_confidence is not None
        assert 0 <= result.consensus_confidence <= 1

    def test_cascade_execution(self, executor):
        """Test cascade strategy"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 2000}

        result = executor.execute(
            strategy=ExecutionStrategy.CASCADE,
            tool_names=["fast_tool", "accurate_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.CASCADE

    def test_tournament_execution(self, executor):
        """Test tournament strategy"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 3000}

        result = executor.execute(
            strategy=ExecutionStrategy.TOURNAMENT,
            tool_names=["fast_tool", "slow_tool", "accurate_tool", "unreliable_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.TOURNAMENT
        assert "winner" in result.metadata or len(result.tools_used) > 0

    def test_adaptive_mix(self, executor):
        """Test adaptive mix strategy"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 2000}

        result = executor.execute(
            strategy=ExecutionStrategy.ADAPTIVE_MIX,
            tool_names=["fast_tool", "accurate_tool"],
            problem="test_problem",
            constraints=constraints,
        )

        assert isinstance(result, PortfolioResult)
        assert result.strategy == ExecutionStrategy.ADAPTIVE_MIX

    def test_invalid_tools(self, executor):
        """Test execution with invalid tools"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 1000}

        result = executor.execute(
            strategy=ExecutionStrategy.SINGLE,
            tool_names=["nonexistent_tool"],
            problem="test",
            constraints=constraints,
        )

        # Should return error result
        assert "error" in result.metadata

    def test_tool_failure_handling(self):
        """Test handling of tool failures"""
        tools = {
            "failing_tool": MockTool("failing_tool", 0.1, 0.8, should_fail=True),
            "working_tool": MockTool("working_tool", 0.1, 0.8, should_fail=False),
        }

        executor = PortfolioExecutor(tools=tools)

        try:
            constraints = {"time_budget_ms": 5000, "energy_budget_mj": 1000}

            # Single failing tool should return error
            result = executor.execute(
                strategy=ExecutionStrategy.SINGLE,
                tool_names=["failing_tool"],
                problem="test",
                constraints=constraints,
            )

            assert "error" in result.metadata or result.primary_result is None
        finally:
            executor.shutdown()

    def test_statistics_tracking(self, executor):
        """Test execution statistics tracking"""
        constraints = {"time_budget_ms": 5000, "energy_budget_mj": 1000}

        # Execute several times
        for _ in range(3):
            executor.execute(
                strategy=ExecutionStrategy.SINGLE,
                tool_names=["fast_tool"],
                problem="test",
                constraints=constraints,
            )

        stats = executor.get_statistics()

        assert "strategy_performance" in stats
        assert "total_executions" in stats
        assert stats["total_executions"] >= 3
        assert ExecutionStrategy.SINGLE.value in stats["strategy_performance"]

    def test_monitor_integration(self, executor):
        """Test integration with ExecutionMonitor"""
        monitor = ExecutionMonitor(
            time_budget_ms=2000, energy_budget_mj=1000, min_confidence=0.7
        )

        constraints = {"time_budget_ms": 2000, "energy_budget_mj": 1000}

        result = executor.execute(
            strategy=ExecutionStrategy.SINGLE,
            tool_names=["fast_tool"],
            problem="test",
            constraints=constraints,
            monitor=monitor,
        )

        assert isinstance(result, PortfolioResult)
        # Monitor should have tracked execution
        assert len(monitor.executions) > 0 or result.execution_time > 0

    def test_shutdown(self, executor):
        """Test executor shutdown"""
        executor.shutdown(timeout=2.0)

        assert executor._is_shutdown

        # Should not accept new executions
        constraints = {"time_budget_ms": 1000, "energy_budget_mj": 1000}

        result = executor.execute(
            strategy=ExecutionStrategy.SINGLE,
            tool_names=["fast_tool"],
            problem="test",
            constraints=constraints,
        )

        assert "shutdown" in result.metadata.get("error", "").lower()

    def test_helper_methods(self, executor):
        """Test helper methods"""
        # Test result scoring
        good_result = MockResult(confidence=0.9, value="good")
        score = executor._score_result(good_result)
        assert 0 <= score <= 1
        assert score == 0.9

        # Test result acceptability
        assert executor._is_acceptable_result(good_result, 0.8)
        assert not executor._is_acceptable_result(good_result, 0.95)

        # Test tool speed ranking
        rank = executor._get_tool_speed_rank("probabilistic")
        assert isinstance(rank, int)

        # Test energy estimation
        energy = executor._estimate_energy("symbolic", 1.0)
        assert energy > 0


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_portfolio_workflow(self):
        """Test complete portfolio execution workflow"""
        tools = {
            "tool_a": MockTool("tool_a", 0.05, 0.7),
            "tool_b": MockTool("tool_b", 0.1, 0.85),
            "tool_c": MockTool("tool_c", 0.15, 0.95),
        }

        executor = PortfolioExecutor(tools=tools)

        try:
            constraints = {
                "time_budget_ms": 5000,
                "energy_budget_mj": 2000,
                "min_confidence": 0.7,
            }

            # Test each strategy
            strategies = [
                ExecutionStrategy.SINGLE,
                ExecutionStrategy.SPECULATIVE_PARALLEL,
                ExecutionStrategy.COMMITTEE_CONSENSUS,
            ]

            results = []
            for strategy in strategies:
                result = executor.execute(
                    strategy=strategy,
                    tool_names=["tool_a", "tool_b", "tool_c"],
                    problem="integration_test",
                    constraints=constraints,
                )
                results.append(result)

            # All should succeed
            assert all(isinstance(r, PortfolioResult) for r in results)

            # Get final statistics
            stats = executor.get_statistics()
            assert stats["total_executions"] >= len(strategies)

        finally:
            executor.shutdown()

    def test_resource_constraints(self):
        """Test execution under resource constraints"""
        tools = {
            "fast": MockTool("fast", 0.05, 0.7),
            "slow": MockTool("slow", 0.5, 0.9),
        }

        executor = PortfolioExecutor(tools=tools)

        try:
            # Very tight constraints
            constraints = {
                "time_budget_ms": 100,  # 100ms
                "energy_budget_mj": 100,
            }

            result = executor.execute(
                strategy=ExecutionStrategy.SPECULATIVE_PARALLEL,
                tool_names=["fast", "slow"],
                problem="constrained_test",
                constraints=constraints,
            )

            # Should complete within constraints
            assert result.execution_time < 1.0  # Should timeout fast

        finally:
            executor.shutdown()

    def test_concurrent_executions(self):
        """Test concurrent portfolio executions"""
        tools = {f"tool_{i}": MockTool(f"tool_{i}", 0.05, 0.8) for i in range(5)}

        executor = PortfolioExecutor(tools=tools, max_workers=4)

        try:
            results = []
            errors = []

            def run_execution(tool_name):
                try:
                    constraints = {"time_budget_ms": 2000, "energy_budget_mj": 1000}
                    result = executor.execute(
                        strategy=ExecutionStrategy.SINGLE,
                        tool_names=[tool_name],
                        problem="concurrent_test",
                        constraints=constraints,
                    )
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=run_execution, args=(f"tool_{i}",))
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have no errors
            assert len(errors) == 0
            assert len(results) == 5

        finally:
            executor.shutdown()


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_tool_list(self):
        """Test execution with empty tool list"""
        executor = PortfolioExecutor(tools={})

        try:
            constraints = {"time_budget_ms": 1000, "energy_budget_mj": 1000}

            result = executor.execute(
                strategy=ExecutionStrategy.SINGLE,
                tool_names=[],
                problem="test",
                constraints=constraints,
            )

            # Should return error result
            assert "error" in result.metadata
        finally:
            executor.shutdown()

    def test_zero_timeout(self):
        """Test execution with zero timeout"""
        tools = {"tool": MockTool("tool", 0.1, 0.8)}
        executor = PortfolioExecutor(tools=tools)

        try:
            constraints = {"time_budget_ms": 0, "energy_budget_mj": 1000}

            result = executor.execute(
                strategy=ExecutionStrategy.SINGLE,
                tool_names=["tool"],
                problem="test",
                constraints=constraints,
            )

            # Should handle gracefully
            assert isinstance(result, PortfolioResult)
        finally:
            executor.shutdown()

    def test_all_tools_fail(self):
        """Test when all tools fail"""
        tools = {
            f"fail_{i}": MockTool(f"fail_{i}", 0.1, 0.8, should_fail=True)
            for i in range(3)
        }

        executor = PortfolioExecutor(tools=tools)

        try:
            constraints = {"time_budget_ms": 5000, "energy_budget_mj": 1000}

            result = executor.execute(
                strategy=ExecutionStrategy.COMMITTEE_CONSENSUS,
                tool_names=list(tools.keys()),
                problem="test",
                constraints=constraints,
            )

            # Should return error result
            assert "error" in result.metadata or result.primary_result is None
        finally:
            executor.shutdown()


class TestPerformance:
    """Performance tests"""

    def test_parallel_speedup(self):
        """Test that parallel execution is faster"""
        tools = {f"tool_{i}": MockTool(f"tool_{i}", 0.2, 0.8) for i in range(3)}

        executor = PortfolioExecutor(tools=tools)

        try:
            constraints = {"time_budget_ms": 10000, "energy_budget_mj": 5000}

            # Parallel execution
            start = time.time()
            parallel_result = executor.execute(
                strategy=ExecutionStrategy.SPECULATIVE_PARALLEL,
                tool_names=list(tools.keys()),
                problem="perf_test",
                constraints=constraints,
            )
            parallel_time = time.time() - start

            # Should complete faster than sequential execution of all tools
            assert parallel_time < 0.6  # Less than 3 * 0.2

        finally:
            executor.shutdown()

    def test_many_executions(self):
        """Test performance with many executions"""
        tools = {"tool": MockTool("tool", 0.01, 0.8)}
        executor = PortfolioExecutor(tools=tools)

        try:
            constraints = {"time_budget_ms": 1000, "energy_budget_mj": 1000}

            start = time.time()

            for _ in range(20):
                executor.execute(
                    strategy=ExecutionStrategy.SINGLE,
                    tool_names=["tool"],
                    problem="perf_test",
                    constraints=constraints,
                )

            elapsed = time.time() - start

            # Should complete reasonably fast
            assert elapsed < 5.0

        finally:
            executor.shutdown()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
