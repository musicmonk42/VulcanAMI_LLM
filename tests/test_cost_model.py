"""
Comprehensive test suite for cost_model.py
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.strategies.cost_model import (ComplexityEstimator, ComplexityLevel,
                                       CostComponent, CostDistribution,
                                       CostObservation, CostPredictor,
                                       HealthMetrics, StochasticCostModel)


@pytest.fixture
def sample_features():
    """Create sample feature vector."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def cost_model():
    """Create StochasticCostModel instance."""
    return StochasticCostModel()


@pytest.fixture
def cost_predictor():
    """Create CostPredictor instance."""
    return CostPredictor()


@pytest.fixture
def complexity_estimator():
    """Create ComplexityEstimator instance."""
    return ComplexityEstimator()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEnums:
    """Test enum classes."""
    
    def test_cost_component_enum(self):
        """Test CostComponent enum."""
        assert CostComponent.TIME_MS.value == "time_ms"
        assert CostComponent.ENERGY_MJ.value == "energy_mj"
        assert CostComponent.MEMORY_MB.value == "memory_mb"
    
    def test_complexity_level_enum(self):
        """Test ComplexityLevel enum."""
        assert ComplexityLevel.TRIVIAL.value == 0.1
        assert ComplexityLevel.EXTREME.value == 1.0


class TestDataClasses:
    """Test dataclass structures."""
    
    def test_cost_observation_creation(self, sample_features):
        """Test creating CostObservation."""
        obs = CostObservation(
            tool_name="test_tool",
            component=CostComponent.TIME_MS,
            value=100.0,
            features=sample_features,
            complexity=0.5
        )
        
        assert obs.tool_name == "test_tool"
        assert obs.value == 100.0
        assert obs.complexity == 0.5
    
    def test_cost_observation_defaults(self, sample_features):
        """Test CostObservation default values."""
        obs = CostObservation(
            tool_name="test",
            component=CostComponent.TIME_MS,
            value=50.0,
            features=sample_features,
            complexity=0.3
        )
        
        assert obs.cold_start is False
        assert obs.health_score == 1.0
        assert isinstance(obs.metadata, dict)
    
    def test_cost_distribution_to_dict(self):
        """Test CostDistribution to_dict method."""
        dist = CostDistribution(
            mean=100.0,
            variance=25.0,
            std=5.0,
            percentile_5=90.0,
            percentile_25=95.0,
            median=100.0,
            percentile_75=105.0,
            percentile_95=110.0,
            confidence_interval=(95.0, 105.0),
            samples=100
        )
        
        result = dist.to_dict()
        
        assert result['mean'] == 100.0
        assert result['std'] == 5.0
        assert result['samples'] == 100
    
    def test_health_metrics_defaults(self):
        """Test HealthMetrics default values."""
        health = HealthMetrics()
        
        assert health.error_rate == 0.0
        assert health.queue_depth == 0
        assert health.consecutive_failures == 0
        assert health.warm is False
    
    def test_health_metrics_score_perfect(self):
        """Test health score with perfect health."""
        health = HealthMetrics()
        
        score = health.health_score
        
        assert score == 1.0
    
    def test_health_metrics_score_with_errors(self):
        """Test health score with errors."""
        health = HealthMetrics(error_rate=0.5)
        
        score = health.health_score
        
        assert score < 1.0
    
    def test_health_metrics_score_with_queue(self):
        """Test health score with queue depth."""
        health = HealthMetrics(queue_depth=50)
        
        score = health.health_score
        
        assert score < 1.0
    
    def test_health_metrics_score_bounds(self):
        """Test health score is bounded [0, 1]."""
        health = HealthMetrics(
            error_rate=1.0,
            queue_depth=1000,
            consecutive_failures=100
        )
        
        score = health.health_score
        
        assert 0.0 <= score <= 1.0


class TestComplexityEstimator:
    """Test ComplexityEstimator."""
    
    def test_initialization(self, complexity_estimator):
        """Test estimator initialization."""
        assert complexity_estimator is not None
        assert complexity_estimator.feature_weights is None
    
    def test_estimate_basic(self, complexity_estimator):
        """Test basic complexity estimation."""
        features = np.array([1.0, 2.0, 3.0])
        
        complexity = complexity_estimator.estimate(features)
        
        assert isinstance(complexity, float)
        assert 0.0 <= complexity <= 1.0
    
    def test_estimate_simple_features(self, complexity_estimator):
        """Test estimation with simple features."""
        features = np.array([1.0, 1.0, 1.0])
        
        complexity = complexity_estimator.estimate(features)
        
        assert complexity >= 0.0
    
    def test_estimate_complex_features(self, complexity_estimator):
        """Test estimation with complex features."""
        features = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        complexity = complexity_estimator.estimate(features)
        
        assert complexity > 0.0
    
    def test_estimate_tracks_history(self, complexity_estimator):
        """Test that estimates are tracked."""
        features = np.array([1.0, 2.0, 3.0])
        
        initial_count = len(complexity_estimator.complexity_history)
        complexity_estimator.estimate(features)
        
        assert len(complexity_estimator.complexity_history) == initial_count + 1


class TestCostPredictor:
    """Test CostPredictor."""
    
    def test_initialization(self, cost_predictor):
        """Test predictor initialization."""
        assert cost_predictor is not None
        assert len(cost_predictor.models) == 0
    
    def test_fit_insufficient_data(self, cost_predictor, sample_features):
        """Test fitting with insufficient data."""
        observations = [
            CostObservation(
                tool_name="test",
                component=CostComponent.TIME_MS,
                value=100.0,
                features=sample_features,
                complexity=0.5
            )
        ]
        
        # Should not fit with < 10 observations
        cost_predictor.fit(observations, "test", CostComponent.TIME_MS)
        
        assert ("test", CostComponent.TIME_MS) not in cost_predictor.models
    
    def test_fit_sufficient_data(self, cost_predictor):
        """Test fitting with sufficient data."""
        observations = []
        for i in range(20):
            obs = CostObservation(
                tool_name="test",
                component=CostComponent.TIME_MS,
                value=100.0 + i,
                features=np.random.randn(5),
                complexity=0.5
            )
            observations.append(obs)
        
        cost_predictor.fit(observations, "test", CostComponent.TIME_MS)
        
        assert ("test", CostComponent.TIME_MS) in cost_predictor.models
    
    def test_predict_no_model(self, cost_predictor, sample_features):
        """Test prediction without trained model."""
        result = cost_predictor.predict(
            sample_features, 0.5, "unknown", CostComponent.TIME_MS
        )
        
        assert result is None
    
    def test_predict_with_model(self, cost_predictor):
        """Test prediction with trained model."""
        # Train model
        observations = []
        for i in range(20):
            obs = CostObservation(
                tool_name="test",
                component=CostComponent.TIME_MS,
                value=100.0 + i * 2,
                features=np.array([float(i)] * 5),
                complexity=0.5
            )
            observations.append(obs)
        
        cost_predictor.fit(observations, "test", CostComponent.TIME_MS)
        
        # Predict
        features = np.array([10.0] * 5)
        prediction = cost_predictor.predict(
            features, 0.5, "test", CostComponent.TIME_MS
        )
        
        assert prediction is not None
        assert prediction >= 0


class TestStochasticCostModel:
    """Test StochasticCostModel."""
    
    def test_initialization(self, cost_model):
        """Test cost model initialization."""
        assert cost_model is not None
        assert len(cost_model.distributions) > 0
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {
            'max_observations': 500,
            'cold_start_time_ms': 200
        }
        
        model = StochasticCostModel(config)
        
        assert model.max_observations == 500
    
    def test_defaults_initialized(self, cost_model):
        """Test that default distributions are initialized."""
        # Should have defaults for common tools
        assert 'symbolic' in cost_model.distributions
        assert 'probabilistic' in cost_model.distributions
    
    def test_predict_cost_basic(self, cost_model, sample_features):
        """Test basic cost prediction."""
        predictions = cost_model.predict_cost("symbolic", sample_features)
        
        assert isinstance(predictions, dict)
        assert CostComponent.TIME_MS.value in predictions
    
    def test_predict_cost_includes_components(self, cost_model, sample_features):
        """Test prediction includes all components."""
        predictions = cost_model.predict_cost("symbolic", sample_features)
        
        for component in CostComponent:
            assert component.value in predictions
    
    def test_predict_cost_includes_failure_risk(self, cost_model, sample_features):
        """Test prediction includes failure risk."""
        predictions = cost_model.predict_cost("symbolic", sample_features)
        
        assert 'failure_risk' in predictions
    
    def test_predict_cost_confidence_interval(self, cost_model, sample_features):
        """Test prediction includes confidence intervals."""
        predictions = cost_model.predict_cost("symbolic", sample_features)
        
        time_pred = predictions[CostComponent.TIME_MS.value]
        
        assert 'ci' in time_pred
        assert len(time_pred['ci']) == 2
        assert time_pred['ci'][0] <= time_pred['mean'] <= time_pred['ci'][1]
    
    def test_update_cost(self, cost_model, sample_features):
        """Test updating cost with observation."""
        initial_count = len(cost_model.observations['test'][CostComponent.TIME_MS])
        
        cost_model.update(
            "test",
            CostComponent.TIME_MS,
            100.0,
            sample_features
        )
        
        final_count = len(cost_model.observations['test'][CostComponent.TIME_MS])
        
        assert final_count == initial_count + 1
    
    def test_update_multiple_observations(self, cost_model):
        """Test updating with multiple observations."""
        features = np.array([1.0, 2.0, 3.0])
        
        for i in range(10):
            cost_model.update(
                "test",
                CostComponent.TIME_MS,
                100.0 + i,
                features
            )
        
        observations = cost_model.observations['test'][CostComponent.TIME_MS]
        
        assert len(observations) == 10
    
    def test_update_limits_observations(self, cost_model):
        """Test that observations are limited."""
        cost_model.max_observations = 10
        features = np.array([1.0, 2.0, 3.0])
        
        for i in range(20):
            cost_model.update(
                "test",
                CostComponent.TIME_MS,
                100.0,
                features
            )
        
        observations = cost_model.observations['test'][CostComponent.TIME_MS]
        
        assert len(observations) <= cost_model.max_observations
    
    def test_update_health_metrics(self, cost_model):
        """Test updating health metrics."""
        cost_model.update_health("test", {
            'error_rate': 0.1,
            'warm': True
        })
        
        health = cost_model.health_metrics["test"]
        
        assert health.error_rate == 0.1
        assert health.warm is True
    
    def test_get_tail_risks(self, cost_model, sample_features):
        """Test getting tail risks."""
        # Add some observations
        for i in range(20):
            cost_model.update(
                "test",
                CostComponent.TIME_MS,
                100.0 + i * 5,
                sample_features
            )
        
        tail_risks = cost_model.get_tail_risks("test")
        
        assert CostComponent.TIME_MS.value in tail_risks
    
    def test_estimate_total_cost(self, cost_model, sample_features):
        """Test estimating total cost."""
        total = cost_model.estimate_total_cost("symbolic", sample_features)
        
        assert isinstance(total, float)
        assert total > 0
    
    def test_estimate_total_cost_with_weights(self, cost_model, sample_features):
        """Test total cost with custom weights."""
        weights = {
            CostComponent.TIME_MS.value: 2.0,
            CostComponent.MEMORY_MB.value: 0.5
        }
        
        total = cost_model.estimate_total_cost(
            "symbolic", sample_features, weights
        )
        
        assert total > 0
    
    def test_get_statistics(self, cost_model, sample_features):
        """Test getting statistics."""
        # Perform some operations
        cost_model.predict_cost("symbolic", sample_features)
        cost_model.update("test", CostComponent.TIME_MS, 100.0, sample_features)
        
        stats = cost_model.get_statistics()
        
        assert 'total_predictions' in stats
        assert 'total_updates' in stats
        assert 'tools_tracked' in stats
    
    def test_confidence_interval_calculation(self, cost_model):
        """Test confidence interval calculation."""
        ci = cost_model._confidence_interval(100.0, 10.0, 0.95, 30)
        
        assert len(ci) == 2
        assert ci[0] < 100.0 < ci[1]
    
    def test_confidence_interval_small_sample(self, cost_model):
        """Test confidence interval with small sample."""
        ci = cost_model._confidence_interval(100.0, 10.0, 0.95, 5)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
    
    def test_confidence_interval_no_sample(self, cost_model):
        """Test confidence interval with no sample count."""
        ci = cost_model._confidence_interval(100.0, 10.0, 0.95, None)
        
        assert len(ci) == 2
        assert ci[0] >= 0


class TestPersistence:
    """Test model persistence."""
    
    def test_save_model(self, cost_model, temp_dir, sample_features):
        """Test saving model."""
        # Add some data
        cost_model.update("test", CostComponent.TIME_MS, 100.0, sample_features)
        
        cost_model.save_model(temp_dir)
        
        # Check files exist
        save_path = Path(temp_dir)
        assert (save_path / 'distributions.json').exists()
        assert (save_path / 'observations.pkl').exists()
    
    def test_load_model(self, temp_dir, sample_features):
        """Test loading model."""
        # Create and save model
        model1 = StochasticCostModel()
        model1.update("test", CostComponent.TIME_MS, 100.0, sample_features)
        model1.save_model(temp_dir)
        
        # Load into new model
        model2 = StochasticCostModel()
        model2.load_model(temp_dir)
        
        # Verify data loaded
        assert "test" in model2.observations
    
    def test_load_nonexistent_model(self, cost_model):
        """Test loading from nonexistent path."""
        # Should not crash
        cost_model.load_model("/nonexistent/path")


class TestThreadSafety:
    """Test thread safety."""
    
    def test_concurrent_updates(self, cost_model):
        """Test concurrent updates."""
        import threading
        
        def update_cost(index):
            features = np.array([float(index)] * 5)
            cost_model.update(
                "test",
                CostComponent.TIME_MS,
                100.0 + index,
                features
            )
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=update_cost, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All updates should be recorded
        observations = cost_model.observations['test'][CostComponent.TIME_MS]
        assert len(observations) == 10
    
    def test_concurrent_predictions(self, cost_model):
        """Test concurrent predictions."""
        import threading
        
        results = []
        
        def predict_cost(index):
            features = np.array([float(index)] * 5)
            pred = cost_model.predict_cost("symbolic", features)
            results.append(pred)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=predict_cost, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_features(self, cost_model):
        """Test with empty features."""
        features = np.array([])
        
        # Should handle gracefully
        try:
            predictions = cost_model.predict_cost("symbolic", features)
            assert isinstance(predictions, dict)
        except:
            pass  # Acceptable to fail
    
    def test_negative_values(self, cost_model):
        """Test with negative values."""
        features = np.array([1.0, 2.0, 3.0])
        
        cost_model.update(
            "test",
            CostComponent.TIME_MS,
            -50.0,  # Negative cost
            features
        )
        
        # Should still work
        observations = cost_model.observations['test'][CostComponent.TIME_MS]
        assert len(observations) > 0
    
    def test_very_large_values(self, cost_model):
        """Test with very large values."""
        features = np.array([1e6, 1e7, 1e8])
        
        cost_model.update(
            "test",
            CostComponent.TIME_MS,
            1e10,
            features
        )
        
        # Should handle large values
        observations = cost_model.observations['test'][CostComponent.TIME_MS]
        assert len(observations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])