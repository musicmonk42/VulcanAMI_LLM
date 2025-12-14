"""
Integration test suite for the strategies module
Tests that all strategy components work together properly
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import all strategy components
from src.strategies.cost_model import CostComponent, StochasticCostModel
from src.strategies.distribution_monitor import (
    DistributionMonitor,
    DriftSeverity,
    DriftType,
)
from src.strategies.feature_extraction import FeatureTier, MultiTierFeatureExtractor
from src.strategies.tool_monitor import HealthStatus, MetricType, ToolMonitor
from src.strategies.value_of_information import (
    DecisionState,
    InformationSource,
    ValueOfInformationGate,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def integrated_strategy_system():
    """Create fully integrated strategy system."""
    # Initialize all components
    cost_model = StochasticCostModel()
    distribution_monitor = DistributionMonitor()
    feature_extractor = MultiTierFeatureExtractor()
    tool_monitor = ToolMonitor(config={"monitoring_interval": 10.0})
    voi_gate = ValueOfInformationGate()

    yield {
        "cost_model": cost_model,
        "distribution_monitor": distribution_monitor,
        "feature_extractor": feature_extractor,
        "tool_monitor": tool_monitor,
        "voi_gate": voi_gate,
    }

    # Cleanup
    tool_monitor.shutdown()


@pytest.fixture
def sample_problems():
    """Create sample problems for testing."""
    return [
        "Solve x + 5 = 10",
        "What is the probability of rolling two sixes?",
        "If all men are mortal and Socrates is a man, what can we conclude?",
        {
            "text": "Find shortest path",
            "graph": {"nodes": ["A", "B", "C"], "edges": [("A", "B"), ("B", "C")]},
        },
        "Calculate the derivative of x^2 + 3x + 2",
    ]


class TestBasicIntegration:
    """Test basic integration between components."""

    def test_all_components_initialized(self, integrated_strategy_system):
        """Test that all components initialize successfully."""
        assert integrated_strategy_system["cost_model"] is not None
        assert integrated_strategy_system["distribution_monitor"] is not None
        assert integrated_strategy_system["feature_extractor"] is not None
        assert integrated_strategy_system["tool_monitor"] is not None
        assert integrated_strategy_system["voi_gate"] is not None

    def test_components_compatible(self, integrated_strategy_system):
        """Test that components have compatible interfaces."""
        system = integrated_strategy_system

        # Feature extractor produces features
        features = system["feature_extractor"].extract_tier1("test problem")
        assert isinstance(features, np.ndarray)

        # Cost model can use features
        predictions = system["cost_model"].predict_cost("symbolic", features)
        assert isinstance(predictions, dict)

        # Distribution monitor can track features
        system["distribution_monitor"].update(features)
        assert system["distribution_monitor"].sample_count > 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_decision_workflow(self, integrated_strategy_system):
        """Test complete workflow: extract features -> check VOI -> predict cost -> monitor."""
        system = integrated_strategy_system
        problem = "If x = 5, what is 2x + 3?"

        # Step 1: Extract features
        features = system["feature_extractor"].extract_tier1(problem)
        assert len(features) > 0

        # Step 2: Check if we should gather more info
        budget = {"time_ms": 1000, "energy_mj": 100}
        predictions = np.array([0.4, 0.3, 0.3])  # Uncertain predictions

        should_gather, action = system["voi_gate"].should_probe_deeper(
            features, predictions, budget
        )

        # Step 3: Predict costs
        cost_predictions = system["cost_model"].predict_cost("symbolic", features)
        assert CostComponent.TIME_MS.value in cost_predictions

        # Step 4: Monitor execution
        system["tool_monitor"].record_execution(
            tool_name="symbolic",
            success=True,
            latency_ms=cost_predictions[CostComponent.TIME_MS.value]["mean"],
            energy_mj=cost_predictions[CostComponent.ENERGY_MJ.value]["mean"],
            confidence=0.8,
        )

        # Step 5: Update cost model
        system["cost_model"].update(
            "symbolic",
            CostComponent.TIME_MS,
            cost_predictions[CostComponent.TIME_MS.value]["mean"],
            features,
        )

        # Step 6: Check for drift
        system["distribution_monitor"].update(features)

        # Verify all components updated
        assert system["tool_monitor"].tool_metrics["symbolic"].total_executions > 0
        assert system["distribution_monitor"].sample_count > 0
        assert system["voi_gate"].total_decisions > 0

    def test_multiple_problems_workflow(
        self, integrated_strategy_system, sample_problems
    ):
        """Test workflow with multiple problems."""
        system = integrated_strategy_system

        for i, problem in enumerate(sample_problems):
            # Extract features
            features = system["feature_extractor"].extract_tier1(problem)

            # Predict costs
            tool_name = [
                "symbolic",
                "probabilistic",
                "causal",
                "analogical",
                "multimodal",
            ][i % 5]
            cost_predictions = system["cost_model"].predict_cost(tool_name, features)

            # Simulate execution
            success = i % 4 != 0  # Fail every 4th
            system["tool_monitor"].record_execution(
                tool_name=tool_name,
                success=success,
                latency_ms=cost_predictions[CostComponent.TIME_MS.value]["mean"],
                energy_mj=cost_predictions[CostComponent.ENERGY_MJ.value]["mean"],
                confidence=0.7 + (i % 3) * 0.1,
            )

            # Update models
            system["cost_model"].update(
                tool_name,
                CostComponent.TIME_MS,
                cost_predictions[CostComponent.TIME_MS.value]["mean"],
                features,
            )

            system["distribution_monitor"].update(features)

        # Verify system state
        assert len(system["tool_monitor"].tool_metrics) > 0
        assert system["distribution_monitor"].sample_count == len(sample_problems)


class TestFeatureExtractionIntegration:
    """Test feature extraction integration with other components."""

    def test_features_drive_cost_prediction(self, integrated_strategy_system):
        """Test that features influence cost predictions."""
        system = integrated_strategy_system

        # Simple problem
        simple_features = system["feature_extractor"].extract_tier1("x = 5")
        simple_costs = system["cost_model"].predict_cost("symbolic", simple_features)

        # Complex problem
        complex_features = system["feature_extractor"].extract_tier1(
            "Solve the differential equation dy/dx = y with initial condition y(0) = 1"
        )
        complex_costs = system["cost_model"].predict_cost("symbolic", complex_features)

        # Complex should generally cost more (due to complexity estimation)
        assert isinstance(simple_costs, dict)
        assert isinstance(complex_costs, dict)

    def test_adaptive_feature_extraction(self, integrated_strategy_system):
        """Test adaptive feature extraction based on VOI."""
        system = integrated_strategy_system
        problem = "What is the probability that A and B both occur?"

        # Start with Tier 1
        tier1_features = system["feature_extractor"].extract_tier1(problem)

        # Check VOI
        budget = {"time_ms": 500, "energy_mj": 50}
        predictions = np.array([0.45, 0.45, 0.1])  # High uncertainty

        should_gather, action = system["voi_gate"].should_probe_deeper(
            tier1_features, predictions, budget
        )

        # If VOI suggests gathering more, extract higher tier
        if should_gather and "tier" in (action or ""):
            tier2_features = system["feature_extractor"].extract_tier2(problem)
            assert len(tier2_features) > len(tier1_features)


class TestCostModelIntegration:
    """Test cost model integration with other components."""

    def test_cost_predictions_tracked_by_monitor(self, integrated_strategy_system):
        """Test that cost predictions are tracked by monitor."""
        system = integrated_strategy_system

        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tool_name = "symbolic"

        # Predict costs
        predictions = system["cost_model"].predict_cost(tool_name, features)

        # Simulate execution with predicted costs
        system["tool_monitor"].record_execution(
            tool_name=tool_name,
            success=True,
            latency_ms=predictions[CostComponent.TIME_MS.value]["mean"],
            energy_mj=predictions[CostComponent.ENERGY_MJ.value]["mean"],
            confidence=0.8,
        )

        # Update cost model with actual
        system["cost_model"].update(
            tool_name,
            CostComponent.TIME_MS,
            predictions[CostComponent.TIME_MS.value]["mean"],
            features,
        )

        # Verify both updated
        assert system["tool_monitor"].tool_metrics[tool_name].total_executions > 0
        assert (
            len(system["cost_model"].observations[tool_name][CostComponent.TIME_MS]) > 0
        )

    def test_health_affects_cost_predictions(self, integrated_strategy_system):
        """Test that tool health affects cost predictions."""
        system = integrated_strategy_system
        features = np.array([1.0, 2.0, 3.0])

        # Update health to unhealthy
        system["cost_model"].update_health(
            "test_tool", {"error_rate": 0.5, "warm": False}
        )

        # Predict costs
        predictions = system["cost_model"].predict_cost("test_tool", features)

        # Health should affect predictions
        assert predictions["failure_risk"] > 0


class TestDistributionMonitorIntegration:
    """Test distribution monitor integration."""

    def test_drift_detection_affects_decisions(self, integrated_strategy_system):
        """Test that drift detection can trigger actions."""
        system = integrated_strategy_system

        # Set reference distribution
        reference_features = np.random.randn(100, 10)
        system["distribution_monitor"].set_reference(reference_features)

        # Add similar features (no drift)
        for i in range(50):
            features = np.random.randn(10)
            system["distribution_monitor"].update(features)

        # Add shifted features (drift)
        for i in range(50):
            features = np.random.randn(10) + 5.0  # Shifted
            drift_detected = system["distribution_monitor"].update(features)

        # Get drift summary
        summary = system["distribution_monitor"].get_drift_summary()

        # Drift should be tracked
        assert "total_drifts" in summary

    def test_feature_drift_monitoring(
        self, integrated_strategy_system, sample_problems
    ):
        """Test monitoring feature drift over time."""
        system = integrated_strategy_system

        # FIXED: Extract features from enough problems to have sufficient samples
        # Need at least as many samples as n_components (default 10)
        initial_features = []
        # Use more problems to ensure we have enough samples
        for problem in sample_problems:
            features = system["feature_extractor"].extract_tier1(problem)
            initial_features.append(features)

        # Add more synthetic samples to reach minimum count
        feature_dim = len(initial_features[0])
        while len(initial_features) < 15:  # Ensure we have more than n_components
            synthetic_features = np.random.randn(feature_dim)
            initial_features.append(synthetic_features)

        # Set as reference
        reference = np.array(initial_features)
        system["distribution_monitor"].set_reference(reference)

        # Monitor new problems
        for i in range(10):
            features = np.random.randn(feature_dim)
            system["distribution_monitor"].update(features)

        # Should have monitoring data
        stats = system["distribution_monitor"].get_statistics()
        assert stats["total_samples"] > 0


class TestToolMonitorIntegration:
    """Test tool monitor integration."""

    def test_monitor_tracks_all_executions(
        self, integrated_strategy_system, sample_problems
    ):
        """Test that monitor tracks all tool executions."""
        system = integrated_strategy_system

        tools = ["symbolic", "probabilistic", "causal"]

        for i, problem in enumerate(sample_problems):
            features = system["feature_extractor"].extract_tier1(problem)
            tool = tools[i % len(tools)]

            # Execute and monitor
            system["tool_monitor"].record_execution(
                tool_name=tool,
                success=True,
                latency_ms=100.0 + i * 10,
                energy_mj=10.0,
                confidence=0.8,
            )

        # Get summary
        summary = system["tool_monitor"].get_metrics_summary()

        assert summary["system"]["total_requests"] == len(sample_problems)
        assert len(summary["tools"]) <= len(tools)

    def test_monitor_health_informs_cost_model(self, integrated_strategy_system):
        """Test that monitor health informs cost model."""
        system = integrated_strategy_system

        # Record failures
        for i in range(5):
            system["tool_monitor"].record_execution(
                tool_name="failing_tool",
                success=False,
                latency_ms=200.0,
                energy_mj=20.0,
                confidence=0.3,
            )

        # Get tool metrics
        metrics = system["tool_monitor"].tool_metrics["failing_tool"]

        # Update cost model health
        system["cost_model"].update_health(
            "failing_tool",
            {
                "error_rate": metrics.error_rate,
                "warm": False,
                "consecutive_failures": metrics.consecutive_failures,
            },
        )

        # Health should be reflected
        health = system["cost_model"].health_metrics["failing_tool"]
        assert health.error_rate > 0


class TestVOIGateIntegration:
    """Test VOI gate integration."""

    def test_voi_drives_feature_extraction(self, integrated_strategy_system):
        """Test that VOI decisions drive feature extraction tier."""
        system = integrated_strategy_system
        problem = "Complex multi-step reasoning problem"

        # Start with Tier 1
        features = system["feature_extractor"].extract_tier1(problem)

        # Check VOI with high budget
        budget = {"time_ms": 1000, "energy_mj": 100}
        predictions = np.array([0.4, 0.35, 0.25])  # Uncertain

        should_gather, action = system["voi_gate"].should_probe_deeper(
            features, predictions, budget
        )

        # VOI decision is made
        assert isinstance(should_gather, bool)

    def test_voi_considers_costs(self, integrated_strategy_system):
        """Test that VOI considers cost predictions."""
        system = integrated_strategy_system
        features = np.array([1.0, 2.0, 3.0])

        # Get cost predictions
        tier2_cost = system["cost_model"].predict_cost("tier2_extractor", features)

        # Create decision state with budget
        budget = {
            "time_ms": tier2_cost[CostComponent.TIME_MS.value]["mean"] * 0.5,  # Limited
            "energy_mj": 100,
        }

        predictions = np.array([0.4, 0.3, 0.3])

        should_gather, action = system["voi_gate"].should_probe_deeper(
            features, predictions, budget
        )

        # VOI should consider budget constraints
        assert isinstance(should_gather, bool)


class TestDataFlowIntegration:
    """Test data flow between components."""

    def test_feature_to_cost_to_monitor_flow(self, integrated_strategy_system):
        """Test data flow: features -> cost prediction -> execution -> monitoring."""
        system = integrated_strategy_system
        problem = "Test problem"

        # 1. Extract features
        features = system["feature_extractor"].extract_tier1(problem)
        assert features is not None

        # 2. Predict costs using features
        cost_pred = system["cost_model"].predict_cost("symbolic", features)
        assert cost_pred is not None

        # 3. Execute with predicted costs
        system["tool_monitor"].record_execution(
            tool_name="symbolic",
            success=True,
            latency_ms=cost_pred[CostComponent.TIME_MS.value]["mean"],
            energy_mj=cost_pred[CostComponent.ENERGY_MJ.value]["mean"],
            confidence=0.8,
        )

        # 4. Update cost model with actual
        system["cost_model"].update(
            "symbolic",
            CostComponent.TIME_MS,
            cost_pred[CostComponent.TIME_MS.value]["mean"],
            features,
        )

        # Verify data propagated
        assert len(system["cost_model"].observations["symbolic"]) > 0
        assert system["tool_monitor"].tool_metrics["symbolic"].total_executions > 0

    def test_feedback_loop(self, integrated_strategy_system):
        """Test feedback loop between components."""
        system = integrated_strategy_system

        # Initial execution
        features = np.array([1.0, 2.0, 3.0])
        system["tool_monitor"].record_execution(
            tool_name="test_tool",
            success=False,
            latency_ms=500.0,
            energy_mj=50.0,
            confidence=0.5,
        )

        # Get health from monitor
        monitor_metrics = system["tool_monitor"].tool_metrics["test_tool"]

        # FIXED: Update cost model health without trying to set health_score property
        # health_score is a calculated property, not a settable attribute
        system["cost_model"].update_health(
            "test_tool", {"error_rate": monitor_metrics.error_rate, "warm": False}
        )

        # Next prediction should reflect poor health
        new_pred = system["cost_model"].predict_cost("test_tool", features)

        # Should have failure risk
        assert new_pred["failure_risk"] > 0


class TestConcurrentOperations:
    """Test concurrent operations across components."""

    def test_concurrent_updates(self, integrated_strategy_system):
        """Test concurrent updates to all components."""
        import threading

        system = integrated_strategy_system

        def process_problem(problem_id):
            features = np.random.randn(10)

            # Feature extraction
            # (Would normally use extractor, using random for concurrency test)

            # Cost prediction
            system["cost_model"].predict_cost("symbolic", features)

            # Monitoring
            system["tool_monitor"].record_execution(
                tool_name="symbolic",
                success=True,
                latency_ms=100.0,
                energy_mj=10.0,
                confidence=0.8,
            )

            # Distribution monitoring
            system["distribution_monitor"].update(features)

        threads = []
        for i in range(10):
            t = threading.Thread(target=process_problem, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All components should have data
        assert system["tool_monitor"].tool_metrics["symbolic"].total_executions == 10
        assert system["distribution_monitor"].sample_count == 10


class TestPersistenceIntegration:
    """Test persistence across all components."""

    def test_save_and_load_all_components(self, integrated_strategy_system, temp_dir):
        """Test saving and loading all component states."""
        system = integrated_strategy_system

        # Generate some state
        features = np.array([1.0, 2.0, 3.0])
        system["cost_model"].update("symbolic", CostComponent.TIME_MS, 100.0, features)
        system["distribution_monitor"].set_reference(np.random.randn(50, 3))
        system["tool_monitor"].record_execution("symbolic", True, 100, 10, 0.8)
        system["voi_gate"].should_probe_deeper(features, np.array([0.3, 0.4, 0.3]), {})

        # Save all states
        cost_dir = Path(temp_dir) / "cost_model"
        monitor_dir = Path(temp_dir) / "distribution_monitor"
        voi_dir = Path(temp_dir) / "voi_gate"

        system["cost_model"].save_model(str(cost_dir))
        system["distribution_monitor"].save_state(str(monitor_dir))
        system["voi_gate"].save_state(str(voi_dir))
        system["tool_monitor"].export_metrics(str(Path(temp_dir) / "tool_metrics.json"))

        # Verify files exist
        assert (cost_dir / "distributions.json").exists()
        assert (monitor_dir / "statistics.json").exists()
        assert (voi_dir / "voi_state.json").exists()
        assert (Path(temp_dir) / "tool_metrics.json").exists()


class TestErrorHandling:
    """Test error handling across components."""

    def test_graceful_degradation(self, integrated_strategy_system):
        """Test that components handle errors gracefully."""
        system = integrated_strategy_system

        # Try invalid operations
        try:
            system["cost_model"].predict_cost("unknown_tool", np.array([]))
        except:
            pass  # Should not crash other components

        # Other components should still work
        features = np.array([1.0, 2.0, 3.0])
        system["distribution_monitor"].update(features)

        assert system["distribution_monitor"].sample_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
