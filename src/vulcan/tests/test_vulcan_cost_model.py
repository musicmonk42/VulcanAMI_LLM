"""
Comprehensive test suite for StochasticCostModel.
Tests EWMA tracking, ML predictions, online learning, and adapter interfaces.
"""

from vulcan.reasoning.selection.cost_model import (
    EWMA,
    LGBM_AVAILABLE,
    SKLEARN_AVAILABLE,
    ContextMode,
    CostComponent,
    CostEstimate,
    CostModel,
    ExecutionRecord,
    FeatureExtractor,
    StochasticCostModel,
    get_cost_model,
)
import shutil

# Import the module under test
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model persistence."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def basic_config():
    """Basic configuration for cost model."""
    return {
        "model_type": "ewma",  # Start with EWMA for faster tests
        "ewma_alpha": 0.3,
        "min_samples_for_ml": 50,
        "batch_size": 32,
        "default_time_ms": 100.0,
        "default_energy_mj": 2.0,
        "default_quality": 0.7,
        "default_risk": 0.15,
    }


@pytest.fixture
def cost_model(basic_config):
    """Create a basic cost model instance."""
    return StochasticCostModel(config=basic_config)


@pytest.fixture
def sample_context():
    """Create sample context dictionary."""
    return {
        "mode": ContextMode.BALANCED,
        "input_size": 100.0,
        "deadline_ms": 1000.0,
        "complexity": 1.5,
        "data_volume": 50.0,
    }


@pytest.fixture
def sample_cost_estimate():
    """Create sample cost estimate."""
    return CostEstimate(
        time_ms=150.0,
        energy_mj=2.5,
        quality=0.85,
        risk=0.10,
        time_lower=100.0,
        time_upper=200.0,
        quality_lower=0.75,
        quality_upper=0.95,
        confidence=0.9,
        sample_size=50,
    )


# ============================================================================
# EWMA TESTS
# ============================================================================


class TestEWMA:
    """Test EWMA tracking class."""

    def test_initialization(self):
        """Test EWMA initialization."""
        ewma = EWMA(alpha=0.25)
        assert ewma.alpha == 0.25
        assert ewma.mean is None
        assert ewma.n == 0

    def test_initialization_with_values(self):
        """Test EWMA initialization with initial values."""
        ewma = EWMA(alpha=0.25, init_mean=10.0, init_var=4.0)
        assert ewma.mean == 10.0
        assert ewma.var == 4.0

    def test_first_update(self):
        """Test first EWMA update."""
        ewma = EWMA(alpha=0.25)
        mean, std = ewma.update(5.0)
        assert ewma.mean == 5.0
        assert ewma.n == 1
        assert mean == 5.0

    def test_multiple_updates(self):
        """Test multiple EWMA updates."""
        ewma = EWMA(alpha=0.25)
        values = [5.0, 6.0, 5.5, 6.5, 5.8]

        for val in values:
            mean, std = ewma.update(val)

        assert ewma.n == 5
        assert 5.0 < ewma.mean < 7.0
        assert std >= 0

    def test_get_stats(self):
        """Test getting statistics."""
        ewma = EWMA(alpha=0.25)
        ewma.update(10.0)
        ewma.update(12.0)
        ewma.update(11.0)

        mean, std, n = ewma.get_stats()
        assert n == 3
        assert 10.0 <= mean <= 12.0
        assert std >= 0

    def test_convergence(self):
        """Test EWMA converges to stable value."""
        ewma = EWMA(alpha=0.2)

        # Feed constant value
        for _ in range(50):
            ewma.update(100.0)

        mean, std, _ = ewma.get_stats()
        assert abs(mean - 100.0) < 1.0
        assert std < 5.0


# ============================================================================
# FEATURE EXTRACTOR TESTS
# ============================================================================


class TestFeatureExtractor:
    """Test feature extraction."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert len(extractor.feature_names) > 0

    def test_extract_basic(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        context = {
            "input_size": 100.0,
            "mode": ContextMode.BALANCED,
            "complexity": 1.5,
        }

        features = extractor.extract("test_tool", context)

        assert isinstance(features, np.ndarray)
        assert len(features) == len(extractor.feature_names)
        assert features.dtype == np.float32

    def test_extract_with_deadline(self):
        """Test feature extraction with deadline."""
        extractor = FeatureExtractor()
        context = {
            "input_size": 100.0,
            "deadline_ms": 500.0,
            "estimated_time_ms": 300.0,
            "mode": ContextMode.RUSH,
        }

        features = extractor.extract("test_tool", context)

        # Check deadline features
        assert features[2] == 1.0  # has_deadline
        assert 0.0 < features[3] <= 1.0  # deadline_pressure

    def test_extract_different_modes(self):
        """Test feature extraction with different modes."""
        extractor = FeatureExtractor()

        for mode in [ContextMode.RUSH, ContextMode.ACCURATE, ContextMode.EFFICIENT]:
            context = {"mode": mode, "input_size": 50.0}
            features = extractor.extract("test_tool", context)

            # Check that mode one-hot encoding is present
            mode_features = features[4:9]  # Mode features
            assert np.sum(mode_features) == 1.0  # Exactly one mode active

    def test_extract_empty_context(self):
        """Test feature extraction with empty context."""
        extractor = FeatureExtractor()
        features = extractor.extract("test_tool", None)

        assert isinstance(features, np.ndarray)
        assert len(features) == len(extractor.feature_names)

    def test_time_of_day_features(self):
        """Test time-of-day cyclical features."""
        extractor = FeatureExtractor()
        context = {"input_size": 100.0}

        features = extractor.extract("test_tool", context)

        # Check time features (last two)
        time_sin = features[-2]
        time_cos = features[-1]

        # Should be valid sin/cos values
        assert -1.0 <= time_sin <= 1.0
        assert -1.0 <= time_cos <= 1.0

        # Should satisfy sin^2 + cos^2 = 1
        assert abs(time_sin**2 + time_cos**2 - 1.0) < 0.01


# ============================================================================
# COST ESTIMATE TESTS
# ============================================================================


class TestCostEstimate:
    """Test CostEstimate data class."""

    def test_creation(self):
        """Test creating a cost estimate."""
        estimate = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)

        assert estimate.time_ms == 100.0
        assert estimate.energy_mj == 2.0
        assert estimate.quality == 0.8
        assert estimate.risk == 0.1

    def test_to_dict(self, sample_cost_estimate):
        """Test conversion to dictionary."""
        d = sample_cost_estimate.to_dict()

        assert isinstance(d, dict)
        assert "time_ms" in d
        assert "quality" in d
        assert d["time_ms"] == sample_cost_estimate.time_ms

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "time_ms": 150.0,
            "energy_mj": 2.5,
            "quality": 0.85,
            "risk": 0.10,
            "confidence": 0.9,
            "sample_size": 50,
        }

        estimate = CostEstimate.from_dict(data)

        assert estimate.time_ms == 150.0
        assert estimate.quality == 0.85
        assert estimate.confidence == 0.9

    def test_round_trip(self, sample_cost_estimate):
        """Test dictionary round-trip conversion."""
        d = sample_cost_estimate.to_dict()
        estimate2 = CostEstimate.from_dict(d)

        assert estimate2.time_ms == sample_cost_estimate.time_ms
        assert estimate2.quality == sample_cost_estimate.quality


# ============================================================================
# EXECUTION RECORD TESTS
# ============================================================================


class TestExecutionRecord:
    """Test ExecutionRecord data class."""

    def test_creation(self):
        """Test creating an execution record."""
        record = ExecutionRecord(
            tool_name="test_tool",
            context={"mode": "balanced"},
            timestamp=time.time(),
            actual_time_ms=120.0,
            actual_energy_mj=2.3,
            actual_quality=0.82,
            actual_risk=0.12,
            success=True,
        )

        assert record.tool_name == "test_tool"
        assert record.actual_time_ms == 120.0
        assert record.success is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = ExecutionRecord(
            tool_name="test_tool",
            context={"mode": "balanced"},
            timestamp=time.time(),
            actual_time_ms=120.0,
            actual_energy_mj=2.3,
            actual_quality=0.82,
            actual_risk=0.12,
            success=True,
            features={"input_size": 100.0},
        )

        d = record.to_dict()

        assert isinstance(d, dict)
        assert d["tool_name"] == "test_tool"
        assert d["actual_time_ms"] == 120.0
        assert "features" in d


# ============================================================================
# STOCHASTIC COST MODEL TESTS
# ============================================================================


class TestStochasticCostModel:
    """Test main cost model class."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        model = StochasticCostModel()
        assert model.ewma_alpha > 0
        assert model.min_samples_for_ml > 0

    def test_initialization_with_config(self, basic_config):
        """Test initialization with config."""
        model = StochasticCostModel(config=basic_config)
        assert model.ewma_alpha == 0.3
        assert model.default_time_ms == 100.0

    def test_predict_cold_start(self, cost_model, sample_context):
        """Test prediction with no training data (cold start)."""
        estimate = cost_model.predict("new_tool", sample_context)

        assert isinstance(estimate, CostEstimate)
        assert estimate.time_ms > 0
        assert 0 <= estimate.quality <= 1
        assert 0 <= estimate.risk <= 1
        assert estimate.confidence >= 0

    def test_predict_ewma(self, cost_model, sample_context):
        """Test prediction using EWMA."""
        # Train with some data
        for i in range(10):
            actual = CostEstimate(
                time_ms=100.0 + i * 5, energy_mj=2.0, quality=0.8, risk=0.1
            )
            cost_model.observe("test_tool", sample_context, actual)

        # Predict
        estimate = cost_model.predict("test_tool", sample_context)

        assert isinstance(estimate, CostEstimate)
        assert estimate.metadata.get("predictor") == "ewma"
        assert estimate.sample_size > 0

    def test_observe_updates_ewma(
        self, cost_model, sample_context, sample_cost_estimate
    ):
        """Test that observe updates EWMA statistics."""
        # Get initial stats
        initial_stats = cost_model.get_statistics("test_tool")
        initial_count = initial_stats["sample_count"]

        # Observe
        cost_model.observe("test_tool", sample_context, sample_cost_estimate)

        # Check updated stats
        updated_stats = cost_model.get_statistics("test_tool")
        assert updated_stats["sample_count"] == initial_count + 1

    def test_observe_multiple(self, cost_model, sample_context):
        """Test multiple observations."""
        for i in range(20):
            actual = CostEstimate(
                time_ms=100.0 + np.random.randn() * 10,
                energy_mj=2.0 + np.random.randn() * 0.5,
                quality=0.8 + np.random.randn() * 0.05,
                risk=0.1 + np.random.randn() * 0.02,
            )
            cost_model.observe("test_tool", sample_context, actual)

        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] == 20
        assert 80 < stats["time_ms"] < 120  # Should be near 100

    def test_mode_specific_tracking(self, cost_model):
        """Test that different modes are tracked separately."""
        contexts = [
            {"mode": ContextMode.RUSH},
            {"mode": ContextMode.ACCURATE},
            {"mode": ContextMode.EFFICIENT},
        ]

        for ctx in contexts:
            for i in range(5):
                actual = CostEstimate(
                    time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1
                )
                cost_model.observe("test_tool", ctx, actual)

        # Each mode should have separate statistics
        rush_stats = cost_model.get_statistics("test_tool", "rush")
        accurate_stats = cost_model.get_statistics("test_tool", "accurate")

        assert rush_stats["sample_count"] == 5
        assert accurate_stats["sample_count"] == 5

    def test_mode_nudges(self, cost_model, sample_context):
        """Test that mode applies appropriate nudges to predictions."""
        # Train with balanced mode
        for _ in range(10):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", sample_context, actual)

        # Predict with different modes
        rush_ctx = {**sample_context, "mode": ContextMode.RUSH}
        accurate_ctx = {**sample_context, "mode": ContextMode.ACCURATE}

        rush_est = cost_model.predict("test_tool", rush_ctx)
        accurate_est = cost_model.predict("test_tool", accurate_ctx)

        # Rush should be faster but lower quality
        assert rush_est.time_ms < accurate_est.time_ms
        assert rush_est.quality < accurate_est.quality

    def test_deadline_penalty(self, cost_model):
        """Test deadline penalty for late predictions."""
        context_no_deadline = {"input_size": 100.0}
        context_tight_deadline = {"input_size": 100.0, "deadline_ms": 50.0}

        # Train
        for _ in range(10):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", context_no_deadline, actual)

        # Predict with tight deadline
        est = cost_model.predict("test_tool", context_tight_deadline)

        # Should have higher risk due to deadline violation
        assert est.risk > 0.1

    def test_input_size_scaling(self, cost_model):
        """Test that input size scales predictions."""
        small_ctx = {"input_size": 10.0}
        large_ctx = {"input_size": 1000.0}

        # Train
        for _ in range(10):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", small_ctx, actual)

        # Predict for different sizes
        small_est = cost_model.predict("test_tool", small_ctx)
        large_est = cost_model.predict("test_tool", large_ctx)

        # Larger input should take more time
        assert large_est.time_ms > small_est.time_ms

    def test_confidence_intervals(self, cost_model, sample_context):
        """Test that confidence intervals are generated."""
        # Train
        for _ in range(20):
            actual = CostEstimate(
                time_ms=100.0 + np.random.randn() * 20,
                energy_mj=2.0,
                quality=0.8 + np.random.randn() * 0.1,
                risk=0.1,
            )
            cost_model.observe("test_tool", sample_context, actual)

        # Predict
        est = cost_model.predict("test_tool", sample_context)

        # Check intervals
        assert est.time_lower < est.time_ms < est.time_upper
        assert est.quality_lower < est.quality < est.quality_upper
        assert est.time_upper - est.time_lower > 0

    def test_confidence_increases_with_samples(self, cost_model, sample_context):
        """Test that confidence increases with more samples."""
        # Few samples
        for _ in range(5):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", sample_context, actual)

        est_low = cost_model.predict("test_tool", sample_context)

        # Many more samples
        for _ in range(100):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", sample_context, actual)

        est_high = cost_model.predict("test_tool", sample_context)

        assert est_high.confidence > est_low.confidence


# ============================================================================
# ADAPTER INTERFACE TESTS
# ============================================================================


class TestAdapterInterface:
    """Test adapter interface for UnifiedReasoning integration."""

    def test_update_time_component(self, cost_model):
        """Test updating TIME_MS component."""
        context = {"mode": ContextMode.BALANCED}

        cost_model.update("test_tool", CostComponent.TIME_MS, 150.0, context)

        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] > 0

    def test_update_energy_component(self, cost_model):
        """Test updating ENERGY_MJ component."""
        context = {"mode": ContextMode.BALANCED}

        cost_model.update("test_tool", CostComponent.ENERGY_MJ, 2.5, context)

        stats = cost_model.get_statistics("test_tool")
        assert stats["energy_mj"] > 0

    def test_update_quality_component(self, cost_model):
        """Test updating QUALITY component."""
        context = {"mode": ContextMode.BALANCED}

        cost_model.update("test_tool", CostComponent.QUALITY, 0.85, context)

        stats = cost_model.get_statistics("test_tool")
        assert 0 <= stats["quality"] <= 1

    def test_update_risk_component(self, cost_model):
        """Test updating RISK component."""
        context = {"mode": ContextMode.BALANCED}

        cost_model.update("test_tool", CostComponent.RISK, 0.15, context)

        stats = cost_model.get_statistics("test_tool")
        assert 0 <= stats["risk"] <= 1

    def test_update_quality_clamping(self, cost_model):
        """Test that quality values are clamped to bounds."""
        context = {"mode": ContextMode.BALANCED}

        # Try updating with out-of-bounds value
        cost_model.update("test_tool", CostComponent.QUALITY, 1.5, context)

        stats = cost_model.get_statistics("test_tool")
        assert stats["quality"] <= cost_model.max_quality

    def test_update_risk_clamping(self, cost_model):
        """Test that risk values are clamped to bounds."""
        context = {"mode": ContextMode.BALANCED}

        # Try updating with out-of-bounds value
        cost_model.update("test_tool", CostComponent.RISK, -0.5, context)

        stats = cost_model.get_statistics("test_tool")
        assert stats["risk"] >= cost_model.min_risk

    def test_update_unknown_component(self, cost_model):
        """Test that unknown components are handled gracefully."""
        context = {"mode": ContextMode.BALANCED}

        # Should not crash
        class FakeComponent:
            value = "unknown"

        cost_model.update("test_tool", FakeComponent(), 100.0, context)

    def test_update_multiple_components(self, cost_model):
        """Test updating multiple components."""
        context = {"mode": ContextMode.BALANCED}

        cost_model.update("test_tool", CostComponent.TIME_MS, 150.0, context)
        cost_model.update("test_tool", CostComponent.QUALITY, 0.85, context)
        cost_model.update("test_tool", CostComponent.RISK, 0.12, context)

        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] > 0

    def test_update_affects_predictions(self, cost_model):
        """Test that component updates affect predictions."""
        context = {"mode": ContextMode.BALANCED}

        # Update with specific values
        for _ in range(10):
            cost_model.update("test_tool", CostComponent.TIME_MS, 200.0, context)
            cost_model.update("test_tool", CostComponent.QUALITY, 0.9, context)

        # Predict
        est = cost_model.predict("test_tool", context)

        # Should be influenced by updates
        assert 150 < est.time_ms < 250
        assert 0.8 < est.quality < 0.95


# ============================================================================
# ML MODEL TESTS (if LightGBM available)
# ============================================================================


@pytest.mark.skipif(not LGBM_AVAILABLE, reason="LightGBM not available")
class TestMLModels:
    """Test ML model functionality (requires LightGBM)."""

    def test_ml_model_training(self, sample_context):
        """Test that ML models are trained after sufficient data."""
        config = {
            "model_type": "lgbm",
            "min_samples_for_ml": 20,
            "batch_size": 20,
        }
        model = StochasticCostModel(config=config)

        # Generate sufficient training data
        for i in range(30):
            actual = CostEstimate(
                time_ms=100.0 + i * 2,
                energy_mj=2.0 + i * 0.1,
                quality=0.8 + np.random.randn() * 0.05,
                risk=0.1 + np.random.randn() * 0.02,
            )
            model.observe("test_tool", sample_context, actual)

        # Models should be trained
        assert model.models["time_ms"] is not None

    def test_ml_prediction(self, sample_context):
        """Test ML-based prediction."""
        config = {
            "model_type": "lgbm",
            "min_samples_for_ml": 20,
            "batch_size": 20,
        }
        model = StochasticCostModel(config=config)

        # Train
        for i in range(30):
            actual = CostEstimate(
                time_ms=100.0 + i, energy_mj=2.0, quality=0.8, risk=0.1
            )
            model.observe("test_tool", sample_context, actual)

        # Predict
        est = model.predict("test_tool", sample_context)

        assert est.metadata.get("predictor") == "lgbm"

    def test_ml_calibration(self, sample_context):
        """Test ML model calibration."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not available")

        config = {
            "model_type": "lgbm",
            "min_samples_for_ml": 20,
            "batch_size": 30,
            "enable_calibration": True,
        }
        model = StochasticCostModel(config=config)

        # Train with enough data for calibration
        for i in range(40):
            actual = CostEstimate(
                time_ms=100.0,
                energy_mj=2.0,
                quality=0.8 + np.random.randn() * 0.1,
                risk=0.1 + np.random.randn() * 0.05,
            )
            model.observe("test_tool", sample_context, actual)

        # Check if calibrators were created
        # (They might not be if training failed, but test should not crash)
        assert isinstance(model.calibrators, dict)


# ============================================================================
# PERSISTENCE TESTS
# ============================================================================


class TestPersistence:
    """Test model persistence and loading."""

    def test_save_load_ewma(self, temp_model_dir, sample_context):
        """Test saving and loading EWMA statistics."""
        config = {"model_type": "ewma", "model_save_path": str(temp_model_dir)}

        # Train and save
        model1 = StochasticCostModel(config=config)
        for _ in range(10):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            model1.observe("test_tool", sample_context, actual)

        model1._save_models()

        # Load in new model
        model2 = StochasticCostModel(config=config)

        # Statistics should be loaded
        stats = model2.get_statistics("test_tool")
        assert stats["sample_count"] > 0

    @pytest.mark.skipif(not LGBM_AVAILABLE, reason="LightGBM not available")
    def test_save_load_ml_models(self, temp_model_dir, sample_context):
        """Test saving and loading ML models."""
        config = {
            "model_type": "lgbm",
            "min_samples_for_ml": 20,
            "batch_size": 20,
            "model_save_path": str(temp_model_dir),
        }

        # Train and save
        model1 = StochasticCostModel(config=config)
        for i in range(30):
            actual = CostEstimate(
                time_ms=100.0 + i, energy_mj=2.0, quality=0.8, risk=0.1
            )
            model1.observe("test_tool", sample_context, actual)

        # Verify models were saved
        assert (temp_model_dir / "cost_model_time_ms.pkl").exists()

        # Load in new model
        model2 = StochasticCostModel(config=config)

        # Should have loaded models
        assert model2.models["time_ms"] is not None


# ============================================================================
# STATISTICS AND UTILITIES TESTS
# ============================================================================


class TestStatisticsAndUtilities:
    """Test statistics and utility methods."""

    def test_get_statistics_no_data(self, cost_model):
        """Test getting statistics with no training data."""
        stats = cost_model.get_statistics("unknown_tool")

        assert stats["sample_count"] == 0
        assert stats["time_ms"] == cost_model.default_time_ms

    def test_get_statistics_with_data(self, cost_model, sample_context):
        """Test getting statistics after training."""
        # Train
        for _ in range(10):
            actual = CostEstimate(time_ms=150.0, energy_mj=2.5, quality=0.85, risk=0.12)
            cost_model.observe("test_tool", sample_context, actual)

        stats = cost_model.get_statistics("test_tool")

        assert stats["sample_count"] == 10
        assert "time_ms" in stats
        assert "time_std" in stats
        assert stats["model_type"] == cost_model.effective_model_type

    def test_reset_statistics_all(self, cost_model, sample_context):
        """Test resetting all statistics."""
        # Train
        for _ in range(5):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", sample_context, actual)

        # Reset
        cost_model.reset_statistics()

        # Statistics should be cleared
        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] == 0

    def test_reset_statistics_specific_tool(self, cost_model, sample_context):
        """Test resetting statistics for specific tool."""
        # Train two tools
        for tool in ["tool1", "tool2"]:
            for _ in range(5):
                actual = CostEstimate(
                    time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1
                )
                cost_model.observe(tool, sample_context, actual)

        # Reset only tool1
        cost_model.reset_statistics(tool_name="tool1")

        # tool1 should be reset, tool2 should remain
        stats1 = cost_model.get_statistics("tool1")
        stats2 = cost_model.get_statistics("tool2")

        assert stats1["sample_count"] == 0
        assert stats2["sample_count"] > 0

    def test_reset_statistics_specific_mode(self, cost_model):
        """Test resetting statistics for specific mode."""
        contexts = [
            {"mode": ContextMode.RUSH},
            {"mode": ContextMode.ACCURATE},
        ]

        # Train both modes
        for ctx in contexts:
            for _ in range(5):
                actual = CostEstimate(
                    time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1
                )
                cost_model.observe("test_tool", ctx, actual)

        # Reset specific mode
        cost_model.reset_statistics(tool_name="test_tool", mode="rush")

        # Rush should be reset, accurate should remain
        rush_stats = cost_model.get_statistics("test_tool", "rush")
        accurate_stats = cost_model.get_statistics("test_tool", "accurate")

        assert rush_stats["sample_count"] == 0
        assert accurate_stats["sample_count"] > 0


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_observations(self, cost_model, sample_context):
        """Test concurrent observations from multiple threads."""

        def observe_many():
            for _ in range(20):
                actual = CostEstimate(
                    time_ms=100.0 + np.random.randn() * 10,
                    energy_mj=2.0,
                    quality=0.8,
                    risk=0.1,
                )
                cost_model.observe("test_tool", sample_context, actual)

        threads = [threading.Thread(target=observe_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all observations
        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] == 100

    def test_concurrent_predictions(self, cost_model, sample_context):
        """Test concurrent predictions from multiple threads."""
        # Train first
        for _ in range(10):
            actual = CostEstimate(time_ms=100.0, energy_mj=2.0, quality=0.8, risk=0.1)
            cost_model.observe("test_tool", sample_context, actual)

        results = []

        def predict_many():
            for _ in range(10):
                est = cost_model.predict("test_tool", sample_context)
                results.append(est)

        threads = [threading.Thread(target=predict_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all predictions
        assert len(results) == 50
        assert all(isinstance(r, CostEstimate) for r in results)

    def test_concurrent_updates(self, cost_model):
        """Test concurrent adapter updates."""
        context = {"mode": ContextMode.BALANCED}

        def update_many():
            for _ in range(20):
                cost_model.update("test_tool", CostComponent.TIME_MS, 150.0, context)

        threads = [threading.Thread(target=update_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all updates
        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] == 100


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_none_context(self, cost_model):
        """Test prediction with None context."""
        est = cost_model.predict("test_tool", None)
        assert isinstance(est, CostEstimate)

    def test_observe_none_context(self, cost_model, sample_cost_estimate):
        """Test observation with None context."""
        # Should not crash
        cost_model.observe("test_tool", None, sample_cost_estimate)

        stats = cost_model.get_statistics("test_tool")
        assert stats["sample_count"] > 0

    def test_extreme_values(self, cost_model, sample_context):
        """Test handling of extreme values."""
        # Very large values
        actual = CostEstimate(time_ms=1e6, energy_mj=1e4, quality=0.999, risk=0.001)

        cost_model.observe("test_tool", sample_context, actual)

        # Should not crash
        est = cost_model.predict("test_tool", sample_context)
        assert isinstance(est, CostEstimate)

    def test_negative_values_clamped(self, cost_model):
        """Test that negative quality/risk values are handled."""
        context = {"mode": ContextMode.BALANCED}

        # Try negative values
        cost_model.update("test_tool", CostComponent.QUALITY, -0.5, context)
        cost_model.update("test_tool", CostComponent.RISK, -0.2, context)

        stats = cost_model.get_statistics("test_tool")
        assert stats["quality"] >= cost_model.min_quality
        assert stats["risk"] >= cost_model.min_risk

    def test_empty_history(self, cost_model):
        """Test operations with empty history."""
        assert len(cost_model.history) == 0

        # Should not crash
        stats = cost_model.get_statistics("unknown_tool")
        assert isinstance(stats, dict)


# ============================================================================
# FACTORY AND EXPORTS TESTS
# ============================================================================


class TestFactoryAndExports:
    """Test factory functions and module exports."""

    def test_get_cost_model_factory(self):
        """Test get_cost_model factory function."""
        model = get_cost_model()
        assert isinstance(model, StochasticCostModel)

    def test_get_cost_model_with_config(self, basic_config):
        """Test factory with configuration."""
        model = get_cost_model(config=basic_config)
        assert model.ewma_alpha == basic_config["ewma_alpha"]

    def test_cost_model_alias(self):
        """Test CostModel alias."""
        assert CostModel is StochasticCostModel

    def test_module_exports(self):
        """Test that expected symbols are exported."""
        from vulcan.reasoning.selection.cost_model import __all__

        expected = [
            "StochasticCostModel",
            "CostModel",
            "CostEstimate",
            "ExecutionRecord",
            "CostComponent",
            "EWMA",
            "FeatureExtractor",
            "get_cost_model",
        ]

        for symbol in expected:
            assert symbol in __all__


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for realistic workflows."""

    def test_complete_learning_cycle(self, cost_model):
        """Test complete learning cycle: predict -> observe -> predict."""
        context = {"mode": ContextMode.BALANCED, "input_size": 1.0}

        # Initial prediction (cold start)
        est1 = cost_model.predict("test_tool", context)
        initial_confidence = est1.confidence

        # Simulate execution and observation
        for i in range(20):
            actual = CostEstimate(
                time_ms=120.0 + np.random.randn() * 15,
                energy_mj=2.3 + np.random.randn() * 0.3,
                quality=0.82 + np.random.randn() * 0.08,
                risk=0.13 + np.random.randn() * 0.03,
            )
            cost_model.observe("test_tool", context, actual)

        # Prediction after learning
        est2 = cost_model.predict("test_tool", context)

        # Should have higher confidence
        assert est2.confidence > initial_confidence

        # Predictions should be closer to observed values
        assert 90 < est2.time_ms < 150
        assert 0.7 < est2.quality < 0.9

    def test_multi_tool_multi_mode(self, cost_model):
        """Test tracking multiple tools in multiple modes."""
        tools = ["tool1", "tool2", "tool3"]
        modes = [ContextMode.RUSH, ContextMode.ACCURATE, ContextMode.EFFICIENT]

        # Train all combinations
        for tool in tools:
            for mode in modes:
                context = {"mode": mode, "input_size": 100.0}
                for _ in range(5):
                    actual = CostEstimate(
                        time_ms=100.0 + np.random.randn() * 10,
                        energy_mj=2.0,
                        quality=0.8,
                        risk=0.1,
                    )
                    cost_model.observe(tool, context, actual)

        # Verify all tracked
        for tool in tools:
            for mode in modes:
                mode_str = mode.value if hasattr(mode, "value") else str(mode)
                stats = cost_model.get_statistics(tool, mode_str)
                assert stats["sample_count"] > 0

    def test_adapter_integration(self, cost_model):
        """Test adapter interface in realistic scenario."""
        context = {"mode": ContextMode.BALANCED, "input_size": 1.0}

        # Simulate UnifiedReasoning updates
        for i in range(10):
            # After each execution, update individual components
            cost_model.update("symbolic", CostComponent.TIME_MS, 150.0 + i * 5, context)
            cost_model.update(
                "symbolic", CostComponent.QUALITY, 0.85 + i * 0.01, context
            )
            cost_model.update("symbolic", CostComponent.RISK, 0.1 - i * 0.005, context)

        # Predict should reflect updates
        est = cost_model.predict("symbolic", context)

        assert 150 < est.time_ms < 250
        assert 0.8 < est.quality < 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
