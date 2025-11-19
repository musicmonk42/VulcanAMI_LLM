"""
Comprehensive test suite for value_of_information.py
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from value_of_information import (
    InformationSource,
    VOIAction,
    InformationCost,
    InformationValue,
    DecisionState,
    UncertaintyEstimator,
    InformationGainCalculator,
    CostEstimator,
    ValueCalculator,
    ValueOfInformationGate,
)


@pytest.fixture
def sample_features():
    """Create sample feature vector."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    return np.array([0.3, 0.5, 0.2])


@pytest.fixture
def uncertainty_estimator():
    """Create UncertaintyEstimator instance."""
    return UncertaintyEstimator()


@pytest.fixture
def gain_calculator():
    """Create InformationGainCalculator instance."""
    return InformationGainCalculator()


@pytest.fixture
def cost_estimator():
    """Create CostEstimator instance."""
    return CostEstimator()


@pytest.fixture
def value_calculator():
    """Create ValueCalculator instance."""
    return ValueCalculator()


@pytest.fixture
def voi_gate():
    """Create ValueOfInformationGate instance."""
    return ValueOfInformationGate()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEnums:
    """Test enum classes."""
    
    def test_information_source_enum(self):
        """Test InformationSource enum."""
        assert InformationSource.TIER2_FEATURES.value == "tier2_structural"
        assert InformationSource.PROBE_TOOL.value == "probe_tool"
    
    def test_voi_action_enum(self):
        """Test VOIAction enum."""
        assert VOIAction.PROCEED.value == "proceed"
        assert VOIAction.GATHER_MORE.value == "gather_more"


class TestDataClasses:
    """Test dataclass structures."""
    
    def test_information_cost_creation(self):
        """Test creating InformationCost."""
        cost = InformationCost(
            time_ms=100.0,
            energy_mj=10.0,
            monetary=1.0,
            opportunity=0.5
        )
        
        assert cost.time_ms == 100.0
        assert cost.energy_mj == 10.0
    
    def test_information_cost_total(self):
        """Test calculating total cost."""
        cost = InformationCost(
            time_ms=100.0,
            energy_mj=10.0,
            monetary=1.0,
            opportunity=0.5
        )
        
        total = cost.total_cost()
        
        assert total > 0
    
    def test_information_value_creation(self):
        """Test creating InformationValue."""
        value = InformationValue(
            expected_value=10.0,
            information_gain=0.5,
            cost=InformationCost(100, 10),
            net_value=5.0,
            recommendation=VOIAction.GATHER_MORE,
            source=InformationSource.TIER2_FEATURES,
            confidence=0.8
        )
        
        assert value.expected_value == 10.0
        assert value.net_value == 5.0
    
    def test_decision_state_creation(self, sample_features):
        """Test creating DecisionState."""
        state = DecisionState(
            features=sample_features,
            uncertainty=0.5,
            current_best_tool="tool1",
            current_confidence=0.7
        )
        
        assert state.uncertainty == 0.5
        assert state.current_best_tool == "tool1"


class TestUncertaintyEstimator:
    """Test UncertaintyEstimator."""
    
    def test_initialization(self, uncertainty_estimator):
        """Test estimator initialization."""
        assert uncertainty_estimator is not None
    
    def test_estimate_uncertainty_features_only(self, uncertainty_estimator, sample_features):
        """Test estimating uncertainty from features only."""
        uncertainty = uncertainty_estimator.estimate_uncertainty(sample_features)
        
        assert isinstance(uncertainty, float)
        assert 0 <= uncertainty <= 1
    
    def test_estimate_uncertainty_with_predictions(self, uncertainty_estimator, sample_features, sample_predictions):
        """Test estimating uncertainty with predictions."""
        uncertainty = uncertainty_estimator.estimate_uncertainty(
            sample_features,
            sample_predictions
        )
        
        assert isinstance(uncertainty, float)
        assert 0 <= uncertainty <= 1
    
    def test_feature_uncertainty(self, uncertainty_estimator, sample_features):
        """Test feature-based uncertainty."""
        uncertainty = uncertainty_estimator._feature_uncertainty(sample_features)
        
        assert 0 <= uncertainty <= 1
    
    def test_feature_uncertainty_sparse(self, uncertainty_estimator):
        """Test feature uncertainty with sparse features."""
        sparse_features = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        uncertainty = uncertainty_estimator._feature_uncertainty(sparse_features)
        
        # Sparse features should have higher uncertainty
        assert uncertainty > 0.3
    
    def test_entropy_uncertainty(self, uncertainty_estimator, sample_predictions):
        """Test entropy-based uncertainty."""
        uncertainty = uncertainty_estimator._entropy_uncertainty(sample_predictions)
        
        assert 0 <= uncertainty <= 1
    
    def test_entropy_uncertainty_uniform(self, uncertainty_estimator):
        """Test entropy with uniform distribution."""
        uniform = np.array([0.33, 0.33, 0.34])
        
        uncertainty = uncertainty_estimator._entropy_uncertainty(uniform)
        
        # Uniform distribution should have high uncertainty
        assert uncertainty > 0.8
    
    def test_variance_uncertainty(self, uncertainty_estimator, sample_predictions):
        """Test variance-based uncertainty."""
        uncertainty = uncertainty_estimator._variance_uncertainty(sample_predictions)
        
        assert 0 <= uncertainty <= 1


class TestInformationGainCalculator:
    """Test InformationGainCalculator."""
    
    def test_initialization(self, gain_calculator):
        """Test calculator initialization."""
        assert gain_calculator is not None
    
    def test_calculate_gain(self, gain_calculator):
        """Test calculating information gain."""
        current = np.array([0.5, 0.5])
        posterior = np.array([0.9, 0.1])
        
        gain = gain_calculator.calculate_gain(current, posterior)
        
        assert gain > 0
    
    def test_expected_gain_features(self, gain_calculator, sample_features):
        """Test expected gain from features."""
        gain = gain_calculator.expected_gain_features(sample_features, 2)
        
        assert gain > 0
        assert gain < 1
    
    def test_expected_gain_features_higher_tier(self, gain_calculator, sample_features):
        """Test that higher tiers give more gain."""
        gain2 = gain_calculator.expected_gain_features(sample_features, 2)
        gain4 = gain_calculator.expected_gain_features(sample_features, 4)
        
        assert gain4 > gain2
    
    def test_expected_gain_probe(self, gain_calculator):
        """Test expected gain from probing."""
        gain = gain_calculator.expected_gain_probe("tool1", 0.5)
        
        assert gain > 0
    
    def test_expected_gain_probe_high_confidence(self, gain_calculator):
        """Test that high confidence reduces gain from probing."""
        gain_low = gain_calculator.expected_gain_probe("tool1", 0.5)
        gain_high = gain_calculator.expected_gain_probe("tool1", 0.95)
        
        assert gain_low > gain_high


class TestCostEstimator:
    """Test CostEstimator."""
    
    def test_initialization(self, cost_estimator):
        """Test estimator initialization."""
        assert cost_estimator is not None
    
    def test_estimate_cost_basic(self, cost_estimator):
        """Test basic cost estimation."""
        cost = cost_estimator.estimate_cost(InformationSource.TIER2_FEATURES)
        
        assert isinstance(cost, InformationCost)
        assert cost.time_ms > 0
    
    def test_estimate_cost_all_sources(self, cost_estimator):
        """Test cost estimation for all sources."""
        for source in InformationSource:
            cost = cost_estimator.estimate_cost(source)
            
            assert isinstance(cost, InformationCost)
            assert cost.time_ms > 0
    
    def test_estimate_cost_with_context(self, cost_estimator):
        """Test cost estimation with context."""
        context = {'complexity': 0.5, 'urgency': 0.8}
        
        cost = cost_estimator.estimate_cost(InformationSource.TIER2_FEATURES, context)
        
        assert cost.time_ms > 0
    
    def test_update_cost(self, cost_estimator):
        """Test updating cost estimates."""
        source = InformationSource.TIER2_FEATURES
        actual_cost = InformationCost(150.0, 15.0)
        
        cost_estimator.update_cost(source, actual_cost)
        
        assert len(cost_estimator.historical_costs[source]) > 0


class TestValueCalculator:
    """Test ValueCalculator."""
    
    def test_initialization(self, value_calculator):
        """Test calculator initialization."""
        assert value_calculator is not None
    
    def test_expected_value_current(self, value_calculator):
        """Test current expected value."""
        value = value_calculator.expected_value_current(0.8, 0.9)
        
        assert value > 0
    
    def test_expected_value_with_info(self, value_calculator):
        """Test expected value with additional information."""
        value = value_calculator.expected_value_with_info(0.5, 0.3, 0.8)
        
        assert value > 0
    
    def test_value_improves_with_info(self, value_calculator):
        """Test that value improves with information."""
        current = value_calculator.expected_value_current(0.5, 0.8)
        with_info = value_calculator.expected_value_with_info(0.5, 0.3, 0.8)
        
        assert with_info > current
    
    def test_value_of_perfect_information(self, value_calculator):
        """Test EVPI calculation."""
        evpi = value_calculator.value_of_perfect_information(0.6, 0.9)
        
        # Use pytest.approx for floating-point comparison
        assert evpi == pytest.approx(0.3, rel=1e-9)


class TestValueOfInformationGate:
    """Test ValueOfInformationGate main class."""
    
    def test_initialization(self, voi_gate):
        """Test gate initialization."""
        assert voi_gate is not None
        assert voi_gate.uncertainty_estimator is not None
    
    def test_should_probe_deeper_low_uncertainty(self, voi_gate, sample_features, sample_predictions):
        """Test decision with low uncertainty."""
        budget = {'time_ms': 1000, 'energy_mj': 100}
        
        # With high confidence predictions
        confident_predictions = np.array([0.95, 0.03, 0.02])
        
        should_gather, action = voi_gate.should_probe_deeper(
            sample_features,
            confident_predictions,
            budget
        )
        
        # Should probably not gather more
        assert isinstance(should_gather, bool)
    
    def test_should_probe_deeper_high_uncertainty(self, voi_gate, sample_features):
        """Test decision with high uncertainty."""
        budget = {'time_ms': 1000, 'energy_mj': 100}
        
        # With uncertain predictions
        uncertain_predictions = np.array([0.4, 0.3, 0.3])
        
        should_gather, action = voi_gate.should_probe_deeper(
            sample_features,
            uncertain_predictions,
            budget
        )
        
        # Decision depends on VOI calculation
        assert isinstance(should_gather, bool)
    
    def test_should_probe_deeper_limited_budget(self, voi_gate, sample_features, sample_predictions):
        """Test decision with limited budget."""
        budget = {'time_ms': 10, 'energy_mj': 1}  # Very limited
        
        should_gather, action = voi_gate.should_probe_deeper(
            sample_features,
            sample_predictions,
            budget
        )
        
        # Should probably not gather with limited budget
        assert isinstance(should_gather, bool)
    
    def test_evaluate_information_source(self, voi_gate, sample_features):
        """Test evaluating specific information source."""
        state = DecisionState(
            features=sample_features,
            uncertainty=0.5,
            remaining_budget={'time_ms': 1000, 'energy_mj': 100}
        )
        
        value = voi_gate.evaluate_information_source(
            state,
            InformationSource.TIER2_FEATURES
        )
        
        if value:
            assert isinstance(value, InformationValue)
            assert hasattr(value, 'net_value')
    
    def test_calculate_evpi(self, voi_gate):
        """Test EVPI calculation."""
        distribution = np.array([0.3, 0.5, 0.2])
        utilities = np.array([10.0, 15.0, 8.0])
        
        evpi = voi_gate.calculate_evpi(distribution, utilities)
        
        assert evpi >= 0
    
    def test_calculate_evsi(self, voi_gate):
        """Test EVSI calculation."""
        current = np.array([0.3, 0.5, 0.2])
        sample = np.array([0.2, 0.7, 0.1])
        utilities = np.array([10.0, 15.0, 8.0])
        
        evsi = voi_gate.calculate_evsi(current, sample, utilities)
        
        assert evsi >= 0
    
    def test_multi_stage_voi_myopic(self, voi_gate, sample_features):
        """Test myopic multi-stage VOI."""
        state = DecisionState(
            features=sample_features,
            uncertainty=0.5,
            remaining_budget={'time_ms': 1000}
        )
        
        sequence = voi_gate.multi_stage_voi(state, horizon=3)
        
        assert isinstance(sequence, list)
    
    def test_update_with_outcome(self, voi_gate):
        """Test updating with actual outcome."""
        actual_cost = InformationCost(120.0, 12.0)
        
        voi_gate.update_with_outcome(
            InformationSource.TIER2_FEATURES,
            0.4,
            actual_cost
        )
        
        # Should not crash
    
    def test_get_statistics(self, voi_gate, sample_features, sample_predictions):
        """Test getting statistics."""
        # Make some decisions
        budget = {'time_ms': 1000, 'energy_mj': 100}
        voi_gate.should_probe_deeper(sample_features, sample_predictions, budget)
        
        stats = voi_gate.get_statistics()
        
        assert 'total_decisions' in stats
        assert 'gather_rate' in stats
    
    def test_visualize_decisions(self, voi_gate, sample_features, sample_predictions):
        """Test visualization data."""
        # Make some decisions
        budget = {'time_ms': 1000, 'energy_mj': 100}
        for i in range(10):
            voi_gate.should_probe_deeper(sample_features, sample_predictions, budget)
        
        viz_data = voi_gate.visualize_decisions()
        
        assert isinstance(viz_data, dict)


class TestPersistence:
    """Test state persistence."""
    
    def test_save_state(self, voi_gate, temp_dir, sample_features, sample_predictions):
        """Test saving VOI state."""
        # Make some decisions
        budget = {'time_ms': 1000, 'energy_mj': 100}
        voi_gate.should_probe_deeper(sample_features, sample_predictions, budget)
        
        voi_gate.save_state(temp_dir)
        
        save_path = Path(temp_dir)
        assert (save_path / 'voi_state.json').exists()
    
    def test_load_state(self, temp_dir, sample_features, sample_predictions):
        """Test loading VOI state."""
        # Create and save
        gate1 = ValueOfInformationGate()
        budget = {'time_ms': 1000, 'energy_mj': 100}
        gate1.should_probe_deeper(sample_features, sample_predictions, budget)
        gate1.save_state(temp_dir)
        
        # Load into new instance
        gate2 = ValueOfInformationGate()
        gate2.load_state(temp_dir)
        
        # Should have loaded statistics
        assert gate2.total_decisions > 0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_predictions(self, voi_gate, sample_features):
        """Test with zero predictions."""
        zero_predictions = np.array([0.0, 0.0, 0.0])
        budget = {'time_ms': 1000}
        
        should_gather, action = voi_gate.should_probe_deeper(
            sample_features,
            zero_predictions,
            budget
        )
        
        assert isinstance(should_gather, bool)
    
    def test_empty_budget(self, voi_gate, sample_features, sample_predictions):
        """Test with empty budget."""
        empty_budget = {}
        
        should_gather, action = voi_gate.should_probe_deeper(
            sample_features,
            sample_predictions,
            empty_budget
        )
        
        assert isinstance(should_gather, bool)
    
    def test_negative_values(self, value_calculator):
        """Test handling negative values."""
        # Should handle gracefully
        value = value_calculator.expected_value_current(-0.5, 0.8)
        
        # Value might be negative but shouldn't crash
        assert isinstance(value, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])