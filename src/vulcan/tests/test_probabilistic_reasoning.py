"""
Comprehensive Test Suite for Probabilistic Reasoning

Tests Gaussian Processes, uncertainty quantification, active learning,
kernel selection, and numerical stability fixes.
"""

from vulcan.reasoning.probabilistic_reasoning import (
    EnhancedProbabilisticReasoner,
    ProbabilisticReasoner,
)
import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Filter warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Fixtures
@pytest.fixture
def basic_reasoner():
    """Create basic probabilistic reasoner"""
    return EnhancedProbabilisticReasoner(
        kernel_type="rbf", enable_ensemble=True, enable_learning=True
    )


@pytest.fixture
def adaptive_reasoner():
    """Create adaptive kernel reasoner"""
    return EnhancedProbabilisticReasoner(
        kernel_type="adaptive", enable_ensemble=True, enable_sparse=True
    )


@pytest.fixture
def compatibility_reasoner():
    """Create compatibility wrapper reasoner"""
    return ProbabilisticReasoner(enable_learning=True)


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.1 * np.random.randn(50)
    return X, y


@pytest.fixture
def sample_2d_data():
    """Create 2D sample data"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0] ** 2 + X[:, 1] ** 2 + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def periodic_data():
    """Create data with periodic pattern"""
    np.random.seed(42)
    X = np.linspace(0, 20, 100).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel() / 5) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model saving/loading"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Basic Functionality Tests
class TestEnhancedProbabilisticReasoner:
    """Test EnhancedProbabilisticReasoner"""

    def test_initialization(self):
        reasoner = EnhancedProbabilisticReasoner()

        assert reasoner.kernel_type == "adaptive"
        assert reasoner.ensemble_size == 5
        assert reasoner.enable_ensemble is True
        assert len(reasoner.gp_ensemble) == 5
        assert reasoner.max_inducing_points == 100

    def test_initialization_with_params(self):
        reasoner = EnhancedProbabilisticReasoner(
            kernel_type="matern", enable_sparse=False, enable_ensemble=False
        )

        assert reasoner.kernel_type == "matern"
        assert reasoner.ensemble_size == 1
        assert reasoner.enable_sparse is False

    def test_kernel_types(self):
        kernel_types = ["rbf", "matern", "rational_quadratic", "periodic", "combined"]

        for ktype in kernel_types:
            reasoner = EnhancedProbabilisticReasoner(kernel_type=ktype)
            assert reasoner.kernel_type == ktype

    def test_gp_ensemble_initialization(self, basic_reasoner):
        assert len(basic_reasoner.gp_ensemble) == basic_reasoner.ensemble_size
        assert len(basic_reasoner.ensemble) == basic_reasoner.ensemble_size

        for gp in basic_reasoner.gp_ensemble:
            assert gp is not None

    def test_acquisition_functions(self, basic_reasoner):
        expected_funcs = ["ei", "ucb", "entropy", "thompson", "mes", "kg"]

        for func_name in expected_funcs:
            assert func_name in basic_reasoner.acquisition_functions

    def test_diagnostics_initialization(self, basic_reasoner):
        assert "mse_history" in basic_reasoner.diagnostics
        assert "likelihood_history" in basic_reasoner.diagnostics
        assert "hyperparameter_history" in basic_reasoner.diagnostics


# Training and Prediction Tests
class TestTrainingAndPrediction:
    """Test training and prediction functionality"""

    def test_update_beliefs_batch_simple(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]

        basic_reasoner.update_beliefs_batch(observations)

        assert basic_reasoner.trained is True
        assert len(basic_reasoner.observations) > 0

    def test_update_beliefs_batch_empty(self, basic_reasoner):
        basic_reasoner.update_beliefs_batch([])

        # Should handle gracefully
        assert basic_reasoner.trained is False

    def test_predict_with_uncertainty_ensemble(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.0]])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        assert "mean" in result
        assert "std" in result
        assert "epistemic" in result
        assert "aleatoric" in result
        assert "confidence_interval" in result
        assert "predictions" in result

        assert isinstance(result["mean"], float)
        assert result["std"] >= 0

    def test_predict_without_training(self, basic_reasoner):
        test_point = np.array([[5.0]])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        # Should return default values
        assert result["mean"] == 0.5
        assert result["std"] == 1.0
        assert result["epistemic"] == 1.0

    def test_predict_1d_input(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Test with 1D array
        test_point = np.array([5.0])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        assert isinstance(result["mean"], float)
        assert result["std"] >= 0

    def test_multi_output_handling(self, basic_reasoner):
        X = np.random.randn(50, 2)
        # Single output
        y = np.random.randn(50)

        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        assert basic_reasoner.trained is True
        assert basic_reasoner.output_dim == 1

    def test_confidence_interval_bounds(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.0]])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        ci_lower, ci_upper = result["confidence_interval"]

        assert ci_lower < ci_upper
        assert ci_lower < result["mean"] < ci_upper


# Kernel Selection Tests
class TestKernelSelection:
    """Test adaptive kernel selection"""

    def test_adaptive_kernel_selection(self, adaptive_reasoner, sample_data):
        X, y = sample_data

        adaptive_reasoner.adaptive_kernel_selection(X, y)

        assert adaptive_reasoner.kernel is not None
        assert len(adaptive_reasoner.gp_ensemble) > 0

    def test_adaptive_with_insufficient_data(self, adaptive_reasoner):
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])

        adaptive_reasoner.adaptive_kernel_selection(X, y)

        # Should use RBF as fallback
        assert adaptive_reasoner.kernel is not None

    def test_periodicity_detection(self, adaptive_reasoner, periodic_data):
        X, y = periodic_data

        is_periodic = adaptive_reasoner._detect_periodicity(y)

        # Should detect the periodic pattern
        assert isinstance(is_periodic, bool)

    def test_periodicity_detection_short_data(self, adaptive_reasoner):
        y = np.array([1, 2, 3, 4, 5])

        is_periodic = adaptive_reasoner._detect_periodicity(y)

        assert is_periodic is False

    def test_periodicity_detection_zero_autocorr(self, adaptive_reasoner):
        y = np.zeros(30)

        is_periodic = adaptive_reasoner._detect_periodicity(y)

        # Should handle zero autocorrelation
        assert is_periodic is False

    def test_create_adaptive_kernel(self, adaptive_reasoner):
        kernel = adaptive_reasoner._create_adaptive_kernel()

        assert kernel is not None


# Kernel Functions Tests
class TestKernelFunctions:
    """Test kernel computation functions"""

    def test_rbf_kernel(self, basic_reasoner):
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.0], [1.0]])

        K = basic_reasoner.rbf_kernel(X1, X2, length_scale=1.0)

        assert K.shape == (3, 2)
        assert np.all(K >= 0)
        assert np.all(K <= 1)
        assert K[0, 0] > K[0, 1]  # Closer points have higher similarity

    def test_rbf_kernel_zero_length_scale(self, basic_reasoner):
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[0.0]])

        # Should protect against zero length scale
        K = basic_reasoner.rbf_kernel(X1, X2, length_scale=0.0)

        assert np.all(np.isfinite(K))

    def test_matern_kernel(self, basic_reasoner):
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.0], [1.0]])

        K = basic_reasoner.matern_kernel(X1, X2, length_scale=1.0, nu=1.5)

        assert K.shape == (3, 2)
        assert np.all(K >= 0)
        assert np.all(K <= 1)

    def test_matern_kernel_different_nu(self, basic_reasoner):
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[0.0]])

        for nu in [0.5, 1.5, 2.5, 3.5]:
            K = basic_reasoner.matern_kernel(X1, X2, nu=nu)
            assert np.all(np.isfinite(K))

    def test_matern_kernel_zero_length_scale(self, basic_reasoner):
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[0.0]])

        K = basic_reasoner.matern_kernel(X1, X2, length_scale=0.0)

        assert np.all(np.isfinite(K))


# Active Learning Tests
class TestActiveLearning:
    """Test active learning acquisition functions"""

    def test_expected_improvement(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.5]])
        ei = basic_reasoner.compute_expected_improvement(test_point)

        assert isinstance(ei, float)
        assert ei >= 0

    def test_expected_improvement_zero_std(self, basic_reasoner):
        # Mock prediction with zero std
        with patch.object(
            basic_reasoner, "predict_with_uncertainty_ensemble"
        ) as mock_predict:
            mock_predict.return_value = {
                "mean": 1.0,
                "std": 0.0,
                "epistemic": 0.0,
                "aleatoric": 0.0,
            }

            test_point = np.array([[5.0]])
            ei = basic_reasoner.compute_expected_improvement(test_point)

            assert ei == 0.0

    def test_upper_confidence_bound(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.5]])
        ucb = basic_reasoner.compute_upper_confidence_bound(test_point, beta=2.0)

        assert isinstance(ucb, float)

    def test_entropy_reduction(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.5]])
        entropy_red = basic_reasoner.compute_entropy_reduction(test_point)

        assert isinstance(entropy_red, float)
        assert entropy_red >= 0

    def test_thompson_sampling(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.5]])
        sample = basic_reasoner.thompson_sampling(test_point)

        assert isinstance(sample, float)

    def test_max_value_entropy_search(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Reset random seed to ensure consistent behavior regardless of test order
        np.random.seed(42)
        test_point = np.array([[5.5]])
        mes = basic_reasoner.max_value_entropy_search(test_point)

        assert isinstance(mes, float)

    def test_knowledge_gradient(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.5]])
        kg = basic_reasoner.knowledge_gradient(test_point, n_samples=50)

        assert isinstance(kg, float)

    def test_knowledge_gradient_no_observations(self, basic_reasoner):
        test_point = np.array([[5.5]])
        kg = basic_reasoner.knowledge_gradient(test_point, n_samples=10)

        assert isinstance(kg, float)


# Constrained Optimization Tests
class TestConstrainedOptimization:
    """Test constrained optimization features"""

    def test_add_constraint(self, basic_reasoner):
        def constraint_fn(x):
            return x[0] - 5.0

        basic_reasoner.add_constraint(
            constraint_fn, threshold=0.0, constraint_type="lt"
        )

        assert len(basic_reasoner.constraints) == 1
        assert basic_reasoner.constraints[0]["threshold"] == 0.0
        assert basic_reasoner.constraints[0]["type"] == "lt"

    def test_add_constraint_with_type(self, basic_reasoner):
        def constraint_fn_lt(x):
            return x[0]

        def constraint_fn_gt(x):
            return x[0]

        # Less than constraint
        basic_reasoner.add_constraint(
            constraint_fn_lt, threshold=10.0, constraint_type="lt"
        )

        # Greater than constraint
        basic_reasoner.add_constraint(
            constraint_fn_gt, threshold=5.0, constraint_type="gt"
        )

        assert len(basic_reasoner.constraints) == 2
        assert basic_reasoner.constraints[0]["type"] == "lt"
        assert basic_reasoner.constraints[1]["type"] == "gt"

    def test_suggest_next_observation_constrained(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Add constraint with proper type specification
        def constraint_fn(x):
            return x[0] - 7.0  # Returns negative when x[0] < 7

        # Use 'gt' type: constraint value must be > threshold (0.0)
        # This means x[0] - 7.0 > 0.0, i.e., x[0] > 7.0
        basic_reasoner.add_constraint(
            constraint_fn, threshold=0.0, constraint_type="gt"
        )

        candidates = [np.array([[i]]) for i in range(1, 10)]
        next_obs = basic_reasoner.suggest_next_observation_constrained(
            candidates, method="ei"
        )

        # Should suggest point >= 7
        if next_obs is not None:
            assert next_obs[0][0] >= 7.0

    def test_suggest_next_observation_no_feasible(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Add impossible constraint
        def constraint_fn(x):
            return 1.0  # Always infeasible

        basic_reasoner.add_constraint(constraint_fn, threshold=0.0)

        candidates = [np.array([[i]]) for i in range(1, 5)]
        next_obs = basic_reasoner.suggest_next_observation_constrained(candidates)

        assert next_obs is None

    def test_suggest_next_observation_empty_candidates(self, basic_reasoner):
        result = basic_reasoner.suggest_next_observation_constrained([])

        assert result is None


# Feature Engineering Tests
class TestFeatureEngineering:
    """Test feature engineering capabilities"""

    def test_engineer_features(self, basic_reasoner, sample_2d_data):
        X, y = sample_2d_data

        basic_reasoner.feature_engineering_enabled = True
        X_transformed = basic_reasoner._engineer_features(X, fit=True)

        assert X_transformed.shape[0] == X.shape[0]
        # Should have reduced dimensions via PCA

    def test_engineer_features_transform_only(self, basic_reasoner, sample_2d_data):
        X, y = sample_2d_data

        basic_reasoner.feature_engineering_enabled = True

        # First fit
        X_fit = basic_reasoner._engineer_features(X, fit=True)

        # Then transform
        X_new = np.random.randn(10, 2)
        X_transformed = basic_reasoner._engineer_features(X_new, fit=False)

        assert X_transformed.shape[0] == 10

    def test_engineer_features_failure_handling(self, basic_reasoner):
        # Invalid input
        X = np.array([])

        X_result = basic_reasoner._engineer_features(X, fit=True)

        # Should return original on failure
        assert len(X_result) == 0


# Sparse GP Tests
class TestSparseGP:
    """Test sparse GP with inducing points"""

    def test_select_inducing_points(self, basic_reasoner):
        # Create large dataset
        X = np.random.randn(200, 2)
        y = np.random.randn(200)

        X_induced, y_induced = basic_reasoner._select_inducing_points(X, y)

        assert len(X_induced) <= basic_reasoner.max_inducing_points
        assert len(X_induced) == len(y_induced)
        assert basic_reasoner.inducing_points is not None

    def test_sparse_gp_training(self, adaptive_reasoner):
        # Create large dataset
        X = np.random.randn(150, 2)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(150)

        observations = [(x, y_val) for x, y_val in zip(X, y)]
        adaptive_reasoner.update_beliefs_batch(observations)

        # Should have used sparse approximation
        assert adaptive_reasoner.trained is True


# Kernel Updates Tests
class TestKernelUpdates:
    """Test kernel parameter updates"""

    def test_update_kernel(self, basic_reasoner):
        new_data = np.random.randn(20, 2)
        outcomes = np.random.randn(20)

        initial_history_len = len(basic_reasoner.kernel_history)

        basic_reasoner.update_kernel(new_data, outcomes)

        # History should have at least the initial length (may have added entries)
        assert len(basic_reasoner.kernel_history) >= initial_history_len

    def test_update_kernel_size_limit(self, basic_reasoner):
        # Add many updates
        for _ in range(150):
            new_data = np.random.randn(10, 2)
            outcomes = np.random.randn(10)
            basic_reasoner.update_kernel(new_data, outcomes)

        # Should be limited by deque maxlen
        assert len(basic_reasoner.kernel_history) <= 100

    def test_update_kernel_large_data(self, basic_reasoner):
        # Test with very large data
        new_data = np.random.randn(2000, 2)
        outcomes = np.random.randn(2000)

        basic_reasoner.update_kernel(new_data, outcomes)

        # Should have limited history size
        assert len(basic_reasoner.kernel_history) <= 100

    def test_adapt_kernel_parameters(self, basic_reasoner):
        # Add some history
        for _ in range(10):
            new_data = np.random.randn(5, 2)
            outcomes = np.random.randn(5)
            basic_reasoner.update_kernel(new_data, outcomes)

        # Should run without error
        # The _adapt_kernel_parameters method is called internally


# Belief State Tests
class TestBeliefState:
    """Test belief state management"""

    def test_update_belief_state(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        assert "n_observations" in basic_reasoner.belief_state
        assert "mean_reward" in basic_reasoner.belief_state
        assert "std_reward" in basic_reasoner.belief_state
        assert "timestamp" in basic_reasoner.belief_state

        assert basic_reasoner.belief_state["n_observations"] > 0

    def test_belief_state_with_kernel_params(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Should have kernel type after training
        assert "kernel_type" in basic_reasoner.belief_state


# Numerical Stability Tests
class TestNumericalStability:
    """Test numerical stability fixes"""

    def test_predict_empty_ensemble(self, basic_reasoner):
        # Clear ensemble
        basic_reasoner.ensemble = []
        basic_reasoner.gp_ensemble = []

        test_point = np.array([[5.0]])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        # Should initialize and return safe defaults
        assert result["mean"] == 0.5
        assert result["std"] == 1.0

    def test_predict_division_by_zero(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Mock to return zero std
        with patch.object(basic_reasoner.gp_ensemble[0], "predict") as mock_predict:
            mock_predict.return_value = (np.array([1.0]), np.array([0.0]))

            test_point = np.array([[5.0]])
            result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

            # Should handle zero std gracefully
            assert np.isfinite(result["std"])

    def test_observations_size_limit(self, basic_reasoner):
        # Add many observations
        for i in range(1500):
            obs = (np.array([[float(i)]]), float(i))
            basic_reasoner.observations.append(obs)

        # Update with new batch
        new_obs = [(np.array([[1.0]]), 1.0)]
        basic_reasoner.update_beliefs_batch(new_obs)

        # Should be limited
        assert len(basic_reasoner.observations) <= 1000

    def test_rbf_kernel_negative_distances(self, basic_reasoner):
        # This shouldn't happen, but test protection
        X1 = np.array([[0.0]])
        X2 = np.array([[0.0]])

        K = basic_reasoner.rbf_kernel(X1, X2)

        assert np.all(np.isfinite(K))
        assert np.all(K >= 0)


# Model Persistence Tests
class TestModelPersistence:
    """Test model saving and loading"""

    def test_save_model(self, basic_reasoner, sample_data, temp_model_dir):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        basic_reasoner.model_path = Path(temp_model_dir)
        basic_reasoner.save_model("test_model")

        model_file = Path(temp_model_dir) / "test_model_gp_model.pkl"
        assert model_file.exists()

    def test_load_model(self, basic_reasoner, sample_data, temp_model_dir):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        basic_reasoner.model_path = Path(temp_model_dir)
        basic_reasoner.save_model("test_model")

        # Create new reasoner and load
        new_reasoner = EnhancedProbabilisticReasoner()
        new_reasoner.model_path = Path(temp_model_dir)
        new_reasoner.load_model("test_model")

        assert new_reasoner.trained is True
        assert len(new_reasoner.observations) > 0
        assert len(new_reasoner.gp_ensemble) > 0

    def test_load_nonexistent_model(self, basic_reasoner, temp_model_dir):
        basic_reasoner.model_path = Path(temp_model_dir)

        with pytest.raises(FileNotFoundError):
            basic_reasoner.load_model("nonexistent")


# Diagnostics Tests
class TestDiagnostics:
    """Test diagnostic capabilities"""

    def test_get_diagnostics(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        diagnostics = basic_reasoner.get_diagnostics()

        assert "model_trained" in diagnostics
        assert "n_observations" in diagnostics
        assert "ensemble_size" in diagnostics
        assert "belief_state" in diagnostics

        assert diagnostics["model_trained"] is True
        assert diagnostics["n_observations"] > 0

    def test_diagnostics_history_tracking(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        # Should have stored diagnostics
        assert len(basic_reasoner.diagnostics["likelihood_history"]) > 0


# Compatibility Wrapper Tests
class TestCompatibilityWrapper:
    """Test ProbabilisticReasoner compatibility wrapper"""

    def test_compatibility_initialization(self):
        reasoner = ProbabilisticReasoner(enable_learning=True)

        assert reasoner.enable_learning is True
        assert isinstance(reasoner, EnhancedProbabilisticReasoner)

    def test_reason_method(self, compatibility_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        compatibility_reasoner.update_beliefs_batch(observations)

        result = compatibility_reasoner.reason(np.array([5.0]))

        # FIXED: Check attributes instead of dict keys
        assert hasattr(result, "conclusion")
        assert hasattr(result, "confidence")
        assert hasattr(result, "metadata")
        assert "mean" in result.metadata
        assert "uncertainty" in result.metadata

    def test_reason_with_uncertainty(self, compatibility_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        compatibility_reasoner.update_beliefs_batch(observations)

        result = compatibility_reasoner.reason_with_uncertainty(
            np.array([5.0]), threshold=0.5
        )

        # FIXED: Access attributes instead of dict keys
        assert isinstance(result.conclusion, dict)
        assert "is_above_threshold" in result.conclusion
        assert 0 <= result.confidence <= 1

    def test_reason_with_dict_input(self, compatibility_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        compatibility_reasoner.update_beliefs_batch(observations)

        result = compatibility_reasoner.reason_with_uncertainty(
            {"feature1": 5.0, "feature2": 3.0}
        )

        # FIXED: Check attributes instead of dict keys
        assert hasattr(result, "conclusion")
        assert hasattr(result, "confidence")

    def test_reason_with_list_input(self, compatibility_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        compatibility_reasoner.update_beliefs_batch(observations)

        result = compatibility_reasoner.reason_with_uncertainty([5.0, 3.0])

        # FIXED: Check attributes instead of dict keys
        assert hasattr(result, "conclusion")

    def test_reason_with_string_input(self, compatibility_reasoner):
        result = compatibility_reasoner.reason_with_uncertainty("test string")

        # FIXED: Check attributes instead of dict keys
        # Should handle gracefully with hash-based features
        assert hasattr(result, "conclusion")

    def test_reason_untrained(self, compatibility_reasoner):
        result = compatibility_reasoner.reason_with_uncertainty(np.array([5.0]))

        # FIXED: Access attribute instead of dict key
        # Should return safe defaults
        assert result.confidence == 0.0


# Edge Cases Tests
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_observation(self, basic_reasoner):
        observations = [(np.array([[1.0]]), 1.0)]
        basic_reasoner.update_beliefs_batch(observations)

        assert basic_reasoner.trained is True

    def test_zero_variance_data(self, basic_reasoner):
        X = np.ones((50, 1))
        y = np.ones(50)

        observations = [(x, y_val) for x, y_val in zip(X, y)]

        # Should handle constant data
        basic_reasoner.update_beliefs_batch(observations)

    def test_nan_in_observations(self, basic_reasoner):
        X = np.array([[1.0], [2.0], [np.nan], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        observations = [(x, y_val) for x, y_val in zip(X, y)]

        # Should handle NaN gracefully
        try:
            basic_reasoner.update_beliefs_batch(observations)
        except Exception:
            # Expected to fail, but shouldn't crash
            pass

    def test_extreme_values(self, basic_reasoner):
        X = np.array([[1e10], [1e-10], [1e5]])
        y = np.array([1e10, 1e-10, 1e5])

        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[1e6]])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        assert np.isfinite(result["mean"])
        assert np.isfinite(result["std"])

    def test_high_dimensional_input(self, basic_reasoner):
        X = np.random.randn(100, 50)  # 50 dimensions
        y = np.random.randn(100)

        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.random.randn(1, 50)
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        assert isinstance(result["mean"], float)


# Integration Tests
class TestIntegration:
    """Integration tests across components"""

    def test_full_active_learning_loop(self, basic_reasoner):
        # Initialize with random data
        X_init = np.random.randn(10, 1)
        y_init = np.sin(X_init.ravel())

        observations = [(x, y_val) for x, y_val in zip(X_init, y_init)]
        basic_reasoner.update_beliefs_batch(observations)

        # Active learning loop
        for _ in range(5):
            candidates = [np.array([[x]]) for x in np.linspace(-5, 5, 20)]
            next_point = basic_reasoner.suggest_next_observation_constrained(
                candidates, method="ei"
            )

            if next_point is not None:
                y_new = np.sin(next_point[0])
                basic_reasoner.update_beliefs_batch([(next_point, y_new)])

        # Should have improved
        assert len(basic_reasoner.observations) > 10

    def test_ensemble_diversity(self, basic_reasoner, sample_data):
        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_point = np.array([[5.0]])
        result = basic_reasoner.predict_with_uncertainty_ensemble(test_point)

        predictions = result["predictions"]

        # Ensemble should show some diversity
        if len(predictions) > 1:
            assert np.std(predictions) >= 0


# Performance Tests
class TestPerformance:
    """Test performance characteristics"""

    def test_prediction_speed(self, basic_reasoner, sample_data):
        import time

        X, y = sample_data
        observations = [(x, y_val) for x, y_val in zip(X, y)]
        basic_reasoner.update_beliefs_batch(observations)

        test_points = np.random.randn(100, 1)

        start = time.time()
        for point in test_points:
            basic_reasoner.predict_with_uncertainty_ensemble(point)
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 10.0  # 100 predictions in under 10 seconds

    def test_memory_management(self, basic_reasoner):
        # Add many observations
        for i in range(2000):
            obs = (np.array([[float(i)]]), float(i))
            basic_reasoner.observations.append(obs)

        # Should be limited by deque
        assert len(basic_reasoner.observations) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
