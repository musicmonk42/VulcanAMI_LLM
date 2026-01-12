"""
Comprehensive test suite for cost_optimizer.py

This module tests the cost optimization functionality for memory systems,
including cost analysis, optimization strategies, and budget management.
"""

import os
import sys
import threading
import time
from unittest.mock import Mock

import pytest

# Add src to path properly to enable imports
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.memory.cost_optimizer import (
    CostAnalyzer,
    CostBreakdown,
    CostOptimizer,
    OptimizationMetrics,
    OptimizationPhase,
    OptimizationReport,
    OptimizationStrategy,
)


# ============================================================
# TEST DATA CLASSES
# ============================================================


class TestCostBreakdown:
    def test_initialization_defaults(self):
        breakdown = CostBreakdown()
        assert breakdown.storage_cost == 0.0
        assert breakdown.total_cost == 0.0
        assert isinstance(breakdown.timestamp, float)

    def test_to_dict(self):
        breakdown = CostBreakdown(storage_cost=10.0, total_cost=30.0)
        result = breakdown.to_dict()
        assert result["storage_cost"] == 10.0
        assert "timestamp" in result


class TestOptimizationReport:
    def test_initialization(self):
        report = OptimizationReport(
            optimization_id="test-123",
            strategy=OptimizationStrategy.BALANCED,
            phase=OptimizationPhase.ANALYSIS,
            started_at=time.time(),
        )
        assert report.optimization_id == "test-123"
        assert report.completed_at is None

    def test_get_duration_complete(self):
        start = time.time()
        report = OptimizationReport(
            optimization_id="test",
            strategy=OptimizationStrategy.BALANCED,
            phase=OptimizationPhase.COMPLETED,
            started_at=start,
            completed_at=start + 5.0,
        )
        duration = report.get_duration()
        assert 4.9 <= duration <= 5.1


class TestOptimizationMetrics:
    def test_get_success_rate(self):
        metrics = OptimizationMetrics(
            total_optimizations=10, successful_optimizations=8
        )
        assert metrics.get_success_rate() == 0.8


# ============================================================
# TEST COST ANALYZER
# ============================================================


class TestCostAnalyzer:
    @pytest.fixture
    def mock_memory(self):
        memory = Mock()
        memory.get_storage_usage.return_value = {"total_gb": 100.0}
        memory.get_bandwidth_usage.return_value = {"total_gb": 500.0}
        return memory

    def test_initialization(self, mock_memory):
        analyzer = CostAnalyzer(mock_memory)
        assert analyzer.memory == mock_memory
        assert hasattr(analyzer, "pricing")

    def test_analyze_current_costs(self, mock_memory):
        analyzer = CostAnalyzer(mock_memory)
        costs = analyzer.analyze_current_costs()
        assert isinstance(costs, CostBreakdown)
        assert costs.total_cost >= 0


# ============================================================
# TEST COST OPTIMIZER
# ============================================================


class TestCostOptimizer:
    @pytest.fixture
    def mock_memory(self):
        memory = Mock()
        memory.get_storage_usage.return_value = {"total_gb": 100.0}
        memory.get_bandwidth_usage.return_value = {"total_gb": 500.0}
        memory.dedup_engine = Mock()
        memory.dedup_engine.estimate_duplicates.return_value = {
            "potential_savings_gb": 20.0
        }
        memory.dedup_engine.fold_ir_atoms.return_value = None
        return memory

    def test_initialization(self, mock_memory):
        optimizer = CostOptimizer(mock_memory, auto_optimize=False)
        assert optimizer.memory == mock_memory
        assert isinstance(optimizer.metrics, OptimizationMetrics)
        assert hasattr(optimizer, "budget_config")
        optimizer.shutdown()

    def test_optimize_storage(self, mock_memory):
        optimizer = CostOptimizer(mock_memory, auto_optimize=False)
        report = optimizer.optimize_storage(strategy=OptimizationStrategy.AGGRESSIVE)
        assert isinstance(report, OptimizationReport)
        assert report.phase == OptimizationPhase.COMPLETED
        optimizer.shutdown()

    def test_optimize_full(self, mock_memory):
        optimizer = CostOptimizer(mock_memory, auto_optimize=False)
        report = optimizer.optimize_full(strategy=OptimizationStrategy.BALANCED)
        assert isinstance(report, OptimizationReport)
        assert report.cost_before is not None
        optimizer.shutdown()

    def test_check_budget(self, mock_memory):
        optimizer = CostOptimizer(mock_memory, auto_optimize=False)
        status = optimizer.check_budget()
        assert "status" in status
        assert "usage_percentage" in status
        optimizer.shutdown()

    def test_get_metrics(self, mock_memory):
        optimizer = CostOptimizer(mock_memory, auto_optimize=False)
        metrics = optimizer.get_metrics()
        assert isinstance(metrics, OptimizationMetrics)
        optimizer.shutdown()

    def test_thread_safety(self, mock_memory):
        optimizer = CostOptimizer(mock_memory, auto_optimize=False)

        def run_optimization():
            optimizer.optimize_storage(strategy=OptimizationStrategy.BALANCED)

        threads = [threading.Thread(target=run_optimization) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = optimizer.get_metrics()
        assert metrics.total_optimizations >= 3
        optimizer.shutdown()


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    def test_full_workflow(self):
        memory = Mock()
        memory.get_storage_usage.return_value = {"total_gb": 100.0}
        memory.get_bandwidth_usage.return_value = {"total_gb": 500.0}
        memory.dedup_engine = Mock()
        memory.dedup_engine.estimate_duplicates.return_value = {
            "potential_savings_gb": 20.0
        }

        optimizer = CostOptimizer(memory, auto_optimize=False)
        report = optimizer.optimize_full(strategy=OptimizationStrategy.AGGRESSIVE)

        assert report.phase == OptimizationPhase.COMPLETED
        metrics = optimizer.get_metrics()
        assert metrics.total_optimizations > 0

        optimizer.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
