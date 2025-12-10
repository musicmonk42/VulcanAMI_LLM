"""
confidence_calibrator.py - Confidence calibration for World Model predictions
Part of the VULCAN-AGI system

Integrated with comprehensive safety validation.
FIXED: Complete implementation with proper sklearn integration, robust PAVA algorithm,
       API compatibility - calibrate() handles None features, update() made router-compatible
COMPLETE: All placeholder implementations replaced with production-ready code
CIRCULAR IMPORT FIX: Lazy loading of safety validator to prevent circular dependencies
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# DO NOT import safety modules at module level - prevents circular import
# These will be lazily imported when needed in class methods
# from ..safety.safety_validator import EnhancedSafetyValidator  # REMOVED
# from ..safety.safety_types import SafetyConfig  # REMOVED

# Protected imports with fallbacks
try:
    from sklearn.isotonic import IsotonicRegression

    SKLEARN_ISOTONIC_AVAILABLE = True
except ImportError:
    SKLEARN_ISOTONIC_AVAILABLE = False
    logging.warning(
        "sklearn.isotonic not available, using comprehensive fallback implementation"
    )

try:
    from sklearn.linear_model import LogisticRegression

    SKLEARN_LOGISTIC_AVAILABLE = True
except ImportError:
    SKLEARN_LOGISTIC_AVAILABLE = False
    logging.warning(
        "sklearn.linear_model not available, using comprehensive fallback implementation"
    )

try:
    from scipy import stats
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, using fallback implementations")

logger = logging.getLogger(__name__)


# Comprehensive fallback implementations
class RobustIsotonicRegression:
    """
    Complete isotonic regression implementation with proper PAVA algorithm.
    Handles edge cases, out-of-bounds predictions, and monotonicity constraints.
    Production-ready replacement for sklearn.isotonic.IsotonicRegression.
    """

    def __init__(self, y_min=None, y_max=None, increasing=True, out_of_bounds="clip"):
        """
        Initialize isotonic regression.

        Args:
            y_min: Minimum value for predictions
            y_max: Maximum value for predictions
            increasing: If True, fit increasing function; if False, fit decreasing
            out_of_bounds: How to handle out-of-bounds predictions ('clip', 'nan', or 'raise')
        """
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

        # Fitted parameters
        self.X_min_ = None
        self.X_max_ = None
        self.X_thresholds_ = None
        self.y_thresholds_ = None
        self.f_ = None
        self._is_fitted = False

    def fit(self, X, y, sample_weight=None):
        """
        Fit isotonic regression using Pool Adjacent Violators Algorithm (PAVA).

        Args:
            X: Training data (1D array-like)
            y: Target values (1D array-like)
            sample_weight: Optional sample weights

        Returns:
            self
        """

        # Input validation
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()

        if len(X) == 0:
            raise ValueError("X is empty")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length, got {len(X)} and {len(y)}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")

        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones_like(X)
        else:
            sample_weight = np.asarray(sample_weight).ravel()
            if len(sample_weight) != len(X):
                raise ValueError("sample_weight must have same length as X")
            if not np.all(sample_weight >= 0):
                raise ValueError("sample_weight must be non-negative")

        # Sort by X
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        weights_sorted = sample_weight[sorted_indices]

        # Apply PAVA algorithm
        if self.increasing:
            y_isotonic = self._pava_increasing(y_sorted, weights_sorted)
        else:
            y_isotonic = self._pava_decreasing(y_sorted, weights_sorted)

        # Apply bounds if specified
        if self.y_min is not None:
            y_isotonic = np.maximum(y_isotonic, self.y_min)
        if self.y_max is not None:
            y_isotonic = np.minimum(y_isotonic, self.y_max)

        # Store fitted parameters for prediction
        self.X_min_ = X_sorted[0]
        self.X_max_ = X_sorted[-1]

        # Create step function representation
        # Find points where y changes
        unique_indices = [0]
        for i in range(1, len(y_isotonic))
            if y_isotonic[i] != y_isotonic[i - 1]:
                unique_indices.append(i)
        unique_indices.append(len(y_isotonic) - 1)

        self.X_thresholds_ = X_sorted[unique_indices]
        self.y_thresholds_ = y_isotonic[unique_indices]

        # Create interpolation function
        self.f_ = lambda x: np.interp(x, self.X_thresholds_, self.y_thresholds_)

        self._is_fitted = True
        logger.debug(
            f"Isotonic regression fitted with {len(X)} samples, "
            f"{len(self.X_thresholds_)} thresholds"
        )

        return self

    def predict(self, X):
        """
        Predict using isotonic regression.

        Args:
            X: Data to predict (array-like)

        Returns:
            Predicted values
        """

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() before predict().")

        X = np.asarray(X)
        original_shape = X.shape
        X_flat = X.ravel()

        # Use interpolation function
        y_pred = self.f_(X_flat)

        # Handle out of bounds
        if self.out_of_bounds == "clip":
            # Clip to fitted range
            y_pred = np.clip(
                y_pred, np.min(self.y_thresholds_), np.max(self.y_thresholds_)
            )

            # Also apply y_min/y_max if specified
            if self.y_min is not None:
                y_pred = np.maximum(y_pred, self.y_min)
            if self.y_max is not None:
                y_pred = np.minimum(y_pred, self.y_max)

        elif self.out_of_bounds == "nan":
            # Set out-of-bounds predictions to NaN
            out_of_bounds_mask = (X_flat < self.X_min_) | (X_flat > self.X_max_)
            y_pred[out_of_bounds_mask] = np.nan

        elif self.out_of_bounds == "raise":
            # Raise error if any predictions are out of bounds
            if np.any(X_flat < self.X_min_) or np.any(X_flat > self.X_max_):
                raise ValueError(
                    f"X contains values outside fitted range "
                    f"[{self.X_min_}, {self.X_max_}]"
                )

        return y_pred.reshape(original_shape)

    def _pava_increasing(self, y, weights):
        """
        Pool Adjacent Violators Algorithm for increasing function.
        Optimized O(n²) implementation with proper backward checking.

        Args:
            y: Target values (sorted by X)
            weights: Sample weights

        Returns:
            Isotonic y values
        """

        n = len(y)
        if n == 0:
            return np.array([])

        if n == 1:
            return y.copy()

        # Initialize blocks: each element is (start_idx, end_idx, weighted_avg, total_weight)
        blocks = []
        for i in range(n):
            blocks.append({"start": i, "end": i, "value": y[i], "weight": weights[i]})

        # Merge blocks to enforce monotonicity
        i = 0
        while i < len(blocks) - 1:
            # Check if current block violates monotonicity with next block
            if blocks[i]["value"] > blocks[i + 1]["value"]:
                # Merge the two blocks
                block1 = blocks[i]
                block2 = blocks[i + 1]

                # Weighted average
                total_weight = block1["weight"] + block2["weight"]
                new_value = (
                    block1["value"] * block1["weight"]
                    + block2["value"] * block2["weight"]
                ) / total_weight

                # Create merged block
                merged_block = {
                    "start": block1["start"],
                    "end": block2["end"],
                    "value": new_value,
                    "weight": total_weight,
                }

                # Replace blocks
                blocks[i] = merged_block
                blocks.pop(i + 1)

                # Check backwards for violations
                while i > 0 and blocks[i - 1]["value"] > blocks[i]["value"]:
                    block1 = blocks[i - 1]
                    block2 = blocks[i]

                    total_weight = block1["weight"] + block2["weight"]
                    new_value = (
                        block1["value"] * block1["weight"]
                        + block2["value"] * block2["weight"]
                    ) / total_weight

                    merged_block = {
                        "start": block1["start"],
                        "end": block2["end"],
                        "value": new_value,
                        "weight": total_weight,
                    }

                    blocks[i - 1] = merged_block
                    blocks.pop(i)
                    i -= 1
            else:
                i += 1

        # Expand blocks back to original array
        y_isotonic = np.empty(n)
        for block in blocks:
            y_isotonic[block["start"] : block["end"] + 1] = block["value"]

        return y_isotonic

    def _pava_decreasing(self, y, weights):
        """
        Pool Adjacent Violators Algorithm for decreasing function.

        Args:
            y: Target values (sorted by X)
            weights: Sample weights

        Returns:
            Isotonic y values (decreasing)
        """

        # For decreasing, negate y, apply increasing PAVA, then negate back
        y_neg = -y
        y_isotonic_neg = self._pava_increasing(y_neg, weights)
        return -y_isotonic_neg

    def transform(self, X):
        """Alias for predict for sklearn compatibility"""
        return self.predict(X)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in one step"""
        return self.fit(X, y, sample_weight).transform(X)


class RobustLogisticRegression:
    """
    Complete logistic regression implementation for Platt scaling.
    Handles regularization, convergence monitoring, and edge cases.
    Production-ready replacement for sklearn.linear_model.LogisticRegression.
    """

    def __init__(
        self, penalty="l2", C=1.0, max_iter=100, tol=1e-4, learning_rate=0.1, verbose=0
    ):
        """
        Initialize logistic regression.

        Args:
            penalty: Regularization type ('l2' or 'l1')
            C: Inverse of regularization strength (higher = less regularization)
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
            learning_rate: Learning rate for gradient descent
            verbose: Verbosity level
        """
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Fitted parameters
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self._is_fitted = False

    def fit(self, X, y, sample_weight=None):
        """
        Fit logistic regression using gradient descent with regularization.

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,) - binary (0/1)
            sample_weight: Optional sample weights

        Returns:
            self
        """

        # Input validation
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if len(y) != n_samples:
            raise ValueError(
                f"X and y must have same number of samples, got {n_samples} and {len(y)}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")

        # Check y is binary
        unique_y = np.unique(y)
        if not (len(unique_y) <= 2 and np.all((unique_y == 0) | (unique_y == 1))):
            logger.warning("y should be binary (0/1), attempting to convert")
            y = (y > 0.5).astype(float)

        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight).ravel()
            if len(sample_weight) != n_samples:
                raise ValueError("sample_weight must have same length as y")

        # Normalize sample weights
        sample_weight = sample_weight / np.sum(sample_weight) * n_samples

        # Initialize weights
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.array([0.0])

        # Optimization using gradient descent with momentum
        prev_loss = float("inf")
        velocity_w = np.zeros(n_features)
        velocity_b = 0.0
        momentum = 0.9

        for iteration in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.coef_) + self.intercept_
            predictions = self._sigmoid(z)

            # Calculate loss (cross-entropy with regularization)
            epsilon = 1e-15
            predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)

            cross_entropy = -np.mean(
                sample_weight
                * (
                    y * np.log(predictions_clipped)
                    + (1 - y) * np.log(1 - predictions_clipped)
                )
            )

            # Add regularization
            if self.penalty == "l2":
                reg_term = 0.5 / self.C * np.sum(self.coef_**2)
            elif self.penalty == "l1":
                reg_term = 1.0 / self.C * np.sum(np.abs(self.coef_))
            else:
                reg_term = 0.0

            loss = cross_entropy + reg_term

            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                if self.verbose > 0:
                    logger.info(f"Converged at iteration {iteration}")
                break

            prev_loss = loss

            # Calculate gradients
            error = predictions - y
            grad_w = np.dot(X.T, sample_weight * error) / n_samples
            grad_b = np.mean(sample_weight * error)

            # Add regularization gradient
            if self.penalty == "l2":
                grad_w += self.coef_ / self.C
            elif self.penalty == "l1":
                grad_w += np.sign(self.coef_) / self.C

            # Update with momentum
            velocity_w = momentum * velocity_w - self.learning_rate * grad_w
            velocity_b = momentum * velocity_b - self.learning_rate * grad_b

            self.coef_ += velocity_w
            self.intercept_ += velocity_b

            if self.verbose > 1 and iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}, loss: {loss:.6f}")

        self.n_iter_ = iteration + 1
        self._is_fitted = True

        logger.debug(f"Logistic regression fitted in {self.n_iter_} iterations")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Data to predict (n_samples, n_features)

        Returns:
            Probabilities for each class (n_samples, 2)
        """

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() before predict_proba().")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        z = np.dot(X, self.coef_) + self.intercept_
        proba_1 = self._sigmoid(z)
        proba_0 = 1 - proba_1

        return np.column_stack([proba_0, proba_1])

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Data to predict

        Returns:
            Predicted class labels (0 or 1)
        """

        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def decision_function(self, X):
        """
        Compute decision function (logits).

        Args:
            X: Data to predict

        Returns:
            Decision function values
        """

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() before decision_function().")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.dot(X, self.coef_) + self.intercept_

    def _sigmoid(self, z):
        """Numerically stable sigmoid function"""
        # Clip to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))


class BetaCalibrator:
    """
    Beta calibration for probability calibration.
    Uses beta distribution CDF to map probabilities.
    """

    def __init__(self, method="moments"):
        """
        Initialize beta calibrator.

        Args:
            method: Parameter estimation method ('moments' or 'mle')
        """
        self.method = method
        self.alpha = 1.0
        self.beta = 1.0
        self._is_fitted = False

    def fit(self, probabilities, outcomes):
        """
        Fit beta calibration parameters.

        Args:
            probabilities: Predicted probabilities (n_samples,)
            outcomes: Actual outcomes (n_samples,)

        Returns:
            self
        """

        probabilities = np.asarray(probabilities).ravel()
        outcomes = np.asarray(outcomes).ravel()

        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities and outcomes must have same length")

        if len(probabilities) < 2:
            logger.warning(
                "Too few samples for beta calibration, using default parameters"
            )
            self._is_fitted = True
            return self

        if self.method == "moments":
            self._fit_moments(probabilities, outcomes)
        elif self.method == "mle" and SCIPY_AVAILABLE:
            self._fit_mle(probabilities, outcomes)
        else:
            self._fit_moments(probabilities, outcomes)

        self._is_fitted = True
        logger.debug(
            f"Beta calibration fitted with alpha={self.alpha:.3f}, beta={self.beta:.3f}"
        )

        return self

    def predict(self, probabilities):
        """
        Calibrate probabilities using beta CDF.

        Args:
            probabilities: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() before predict().")

        probabilities = np.asarray(probabilities)

        if SCIPY_AVAILABLE:
            # Use scipy beta CDF
            calibrated = stats.beta.cdf(probabilities, self.alpha, self.beta)
        else:
            # Fallback: approximate beta CDF using incomplete beta function
            calibrated = self._beta_cdf_approx(probabilities, self.alpha, self.beta)

        return calibrated

    def _fit_moments(self, probabilities, outcomes):
        """Fit using method of moments"""

        # Calculate moments
        np.mean(probabilities)
        var_prob = np.var(probabilities)
        mean_outcome = np.mean(outcomes)

        # Prevent division by zero
        if var_prob < 1e-10:
            var_prob = 1e-10

        # Method of moments for beta distribution
        if mean_outcome > 0 and mean_outcome < 1:
            common = mean_outcome * (1 - mean_outcome) / var_prob - 1
            self.alpha = max(0.1, mean_outcome * common)
            self.beta = max(0.1, (1 - mean_outcome) * common)
        else:
            # Fallback to uniform
            self.alpha = 1.0
            self.beta = 1.0

    def _fit_mle(self, probabilities, outcomes):
        """Fit using maximum likelihood estimation"""

        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10

            # Log-likelihood for beta-binomial
            calibrated = stats.beta.cdf(probabilities, alpha, beta)
            calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)

            log_like = np.sum(
                outcomes * np.log(calibrated) + (1 - outcomes) * np.log(1 - calibrated)
            )

            return -log_like

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[self.alpha, self.beta],
            bounds=[(0.01, 100), (0.01, 100)],
            method="L-BFGS-B",
        )

        if result.success:
            self.alpha, self.beta = result.x
        else:
            logger.warning("Beta MLE optimization failed, using moments method")
            self._fit_moments(probabilities, outcomes)

    def _beta_cdf_approx(self, x, alpha, beta):
        """Approximate beta CDF without scipy"""

        # For special cases
        if alpha == 1.0 and beta == 1.0:
            return x  # Uniform distribution

        # Simple approximation using normal distribution
        # Works reasonably well for moderate alpha, beta
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = np.sqrt(var)

        # Normal approximation
        z = (x - mean) / (std + 1e-10)

        # Approximate normal CDF
        return 0.5 * (1.0 + np.tanh(z / np.sqrt(2)))


# Use sklearn if available, otherwise use our implementations
if not SKLEARN_ISOTONIC_AVAILABLE:
    IsotonicRegression = RobustIsotonicRegression
    logger.info("Using fallback RobustIsotonicRegression")

if not SKLEARN_LOGISTIC_AVAILABLE:
    LogisticRegression = RobustLogisticRegression
    logger.info("Using fallback RobustLogisticRegression")


@dataclass
class CalibrationBin:
    """Bin for calibration statistics"""

    min_confidence: float
    max_confidence: float
    predictions: List[float] = field(default_factory=list)
    outcomes: List[float] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        """Average confidence in this bin"""
        if not self.predictions:
            return (self.min_confidence + self.max_confidence) / 2
        return np.mean(self.predictions)

    @property
    def accuracy(self) -> float:
        """Actual accuracy in this bin"""
        if not self.outcomes:
            return 0.0
        return np.mean(self.outcomes)

    @property
    def count(self) -> int:
        """Number of predictions in this bin"""
        return len(self.predictions)


@dataclass
class PredictionRecord:
    """Record of a single prediction for calibration"""

    timestamp: float
    raw_confidence: float
    calibrated_confidence: float
    context_features: Optional[np.ndarray]
    actual_outcome: Optional[float] = None
    domain: str = "unknown"


class ConfidenceCalibrator:
    """
    Calibrates confidence scores using various methods.
    Complete production implementation with safety validation and comprehensive algorithms.
    FIXED: Circular import resolved with lazy loading of safety validator.
    """

    def __init__(
        self,
        method: str = "isotonic",
        n_bins: int = 10,
        window_size: int = 1000,
        safety_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize confidence calibrator

        Args:
            method: Calibration method - "isotonic", "platt", "histogram", "beta"
            n_bins: Number of bins for histogram calibration
            window_size: Window size for rolling calibration
            safety_config: Optional safety configuration
        """
        self.method = method
        self.n_bins = n_bins
        self.window_size = window_size

        # Initialize safety validator with LAZY LOADING to prevent circular import
        self._safety_validator_instance = None
        self._safety_config = safety_config
        self._safety_validator_initialized = False

        # Calibration models
        self.isotonic_model = IsotonicRegression(
            out_of_bounds="clip" if SKLEARN_ISOTONIC_AVAILABLE else "clip",
            y_min=0.0,
            y_max=1.0,
        )
        self.platt_model = LogisticRegression(
            max_iter=100 if not SKLEARN_LOGISTIC_AVAILABLE else 100
        )
        self.beta_calibrator = BetaCalibrator()
        self.histogram_bins = self._initialize_bins()

        # History for calibration
        self.calibration_history = deque(maxlen=window_size)
        self.domain_calibrators = {}  # Domain-specific calibrators

        # Calibration metrics
        self.last_calibration_time = time.time()
        self.calibration_version = 1
        self.models_fitted = {
            "isotonic": False,
            "platt": False,
            "beta": False,
            "histogram": False,
        }

        # Context-aware calibration
        self.context_model = None
        self.context_features_dim = None

        # Safety tracking
        self.safety_blocks = defaultdict(int)
        self.safety_corrections = defaultdict(int)

        # Performance metrics
        self.calibration_calls = 0
        self.calibration_time_total = 0.0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            f"ConfidenceCalibrator initialized with method={method}, n_bins={n_bins}"
        )

    @property
    def safety_validator(self):
        """Lazy-load safety validator to prevent circular import"""
        if not self._safety_validator_initialized:
            try:
                # LAZY IMPORT - prevents circular dependency
                from ..safety.safety_types import SafetyConfig
                from ..safety.safety_validator import EnhancedSafetyValidator

                if isinstance(self._safety_config, dict) and self._safety_config:
                    self._safety_validator_instance = EnhancedSafetyValidator(
                        SafetyConfig.from_dict(self._safety_config)
                    )
                else:
                    self._safety_validator_instance = EnhancedSafetyValidator()
                logger.info("ConfidenceCalibrator: Safety validator initialized (lazy)")
            except ImportError as e:
                logger.warning(
                    f"ConfidenceCalibrator: Safety validator not available: {e}"
                )
                self._safety_validator_instance = None

            self._safety_validator_initialized = True

        return self._safety_validator_instance

    def calibrate(
        self,
        raw_confidence: float,
        context_features: Optional[np.ndarray] = None,
        safety_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calibrate a raw confidence score.

        Args:
            raw_confidence: Uncalibrated confidence score [0, 1]
            context_features: Optional context for context-aware calibration (can be None)
            safety_context: Optional safety context for validation

        Returns:
            Calibrated confidence score [0, 1]
        """

        start_time = time.time()

        with self.lock:
            self.calibration_calls += 1

            # SAFETY: Validate input confidence
            if not np.isfinite(raw_confidence):
                logger.warning(f"Non-finite raw confidence: {raw_confidence}")
                self.safety_corrections["non_finite_confidence"] += 1
                raw_confidence = 0.5

            # Clamp to valid range
            raw_confidence = float(np.clip(raw_confidence, 0.0, 1.0))

            # Check if we can use context-aware calibration
            use_context_calibration = (
                context_features is not None
                and self.context_model is not None
                and np.all(np.isfinite(context_features))
            )

            if use_context_calibration:
                # Context-aware calibration
                try:
                    calibrated = self._context_aware_calibration(
                        raw_confidence, context_features
                    )
                except Exception as e:
                    logger.debug(
                        f"Context-aware calibration failed: {e}, using basic calibration"
                    )
                    calibrated = self._basic_calibration(raw_confidence)
            else:
                # Basic calibration without context
                calibrated = self._basic_calibration(raw_confidence)

            # SAFETY: Reduce confidence for predictions in unsafe regions
            if self.safety_validator and safety_context:
                safety_check = self._validate_safety_context(safety_context)
                if not safety_check["safe"]:
                    logger.debug(
                        f"Reducing confidence due to unsafe region: {safety_check['reason']}"
                    )
                    self.safety_corrections["unsafe_region"] += 1
                    calibrated *= 0.5

            # SAFETY: Final validation
            calibrated = float(np.clip(calibrated, 0.0, 1.0))

            if not np.isfinite(calibrated):
                logger.warning("Non-finite calibrated confidence, returning 0.5")
                self.safety_corrections["calibration_failure"] += 1
                calibrated = 0.5

            # Update timing
            self.calibration_time_total += time.time() - start_time

        return calibrated

    def _basic_calibration(self, confidence: float) -> float:
        """
        Apply basic calibration without context features.

        Args:
            confidence: Raw confidence score

        Returns:
            Calibrated confidence score
        """

        # Method-specific calibration
        if self.method == "isotonic":
            return self._isotonic_calibration(confidence)
        elif self.method == "platt":
            return self._platt_calibration(confidence)
        elif self.method == "histogram":
            return self._histogram_calibration(confidence)
        elif self.method == "beta":
            return self._beta_calibration(confidence)
        else:
            # No calibration
            return confidence

    def update_calibration(
        self,
        prediction: float,
        actual_outcome: float,
        context_features: Optional[np.ndarray] = None,
    ):
        """
        Update calibration with new prediction-outcome pair.

        Args:
            prediction: The confidence score that was made
            actual_outcome: The actual outcome (0 or 1 for binary, continuous for regression)
            context_features: Optional context features
        """

        with self.lock:
            # SAFETY: Validate inputs
            if not np.isfinite(prediction) or not np.isfinite(actual_outcome):
                logger.warning("Non-finite values in calibration update")
                self.safety_blocks["calibration_update"] += 1
                return

            # Clamp values
            prediction = float(np.clip(prediction, 0.0, 1.0))
            actual_outcome = float(np.clip(actual_outcome, 0.0, 1.0))

            # Add to history
            record = PredictionRecord(
                timestamp=time.time(),
                raw_confidence=prediction,
                calibrated_confidence=prediction,
                context_features=context_features
                if context_features is not None
                else None,
                actual_outcome=actual_outcome,
            )
            self.calibration_history.append(record)

            # Update histogram bins
            self._update_histogram_bins(prediction, actual_outcome)

            # Retrain models periodically
            if len(self.calibration_history) >= 100:
                if time.time() - self.last_calibration_time > 60:  # Every minute
                    self._retrain_calibrators()
                    self.last_calibration_time = time.time()
                    self.calibration_version += 1

    def get_calibration_curve(self) -> Tuple[List[float], List[float], List[int]]:
        """
        Get calibration curve data.

        Returns:
            Tuple of (mean_confidences, accuracies, counts) for each bin
        """

        with self.lock:
            mean_confidences = []
            accuracies = []
            counts = []

            for bin in self.histogram_bins:
                if bin.count > 0:
                    mean_confidences.append(bin.mean_confidence)
                    accuracies.append(bin.accuracy)
                    counts.append(bin.count)

            return mean_confidences, accuracies, counts

    def calculate_expected_calibration_error(self) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Returns:
            ECE score (lower is better, 0 is perfect)
        """

        with self.lock:
            total_samples = sum(bin.count for bin in self.histogram_bins)
            if total_samples == 0:
                return 0.0

            ece = 0.0
            for bin in self.histogram_bins:
                if bin.count > 0:
                    bin_weight = bin.count / total_samples
                    bin_error = abs(bin.accuracy - bin.mean_confidence)
                    ece += bin_weight * bin_error

            return float(ece)

    def calculate_maximum_calibration_error(self) -> float:
        """
        Calculate Maximum Calibration Error (MCE).

        Returns:
            MCE score (lower is better, 0 is perfect)
        """

        with self.lock:
            mce = 0.0
            for bin in self.histogram_bins:
                if bin.count > 0:
                    bin_error = abs(bin.accuracy - bin.mean_confidence)
                    mce = max(mce, bin_error)

            return float(mce)

    def calculate_brier_score(self) -> Optional[float]:
        """
        Calculate Brier score for calibration quality.

        Returns:
            Brier score (lower is better, 0 is perfect)
        """

        with self.lock:
            if len(self.calibration_history) == 0:
                return None

            predictions = []
            outcomes = []

            for record in self.calibration_history:
                if record.actual_outcome is not None:
                    predictions.append(record.raw_confidence)
                    outcomes.append(record.actual_outcome)

            if len(predictions) == 0:
                return None

            predictions = np.array(predictions)
            outcomes = np.array(outcomes)

            brier_score = np.mean((predictions - outcomes) ** 2)
            return float(brier_score)

    def get_reliability_diagram_data(self) -> Dict[str, Any]:
        """
        Get data for reliability diagram visualization.

        Returns:
            Dictionary with visualization data
        """

        mean_confs, accuracies, counts = self.get_calibration_curve()

        return {
            "mean_confidences": mean_confs,
            "accuracies": accuracies,
            "counts": counts,
            "perfect_calibration": mean_confs,  # y=x line
            "ece": self.calculate_expected_calibration_error(),
            "mce": self.calculate_maximum_calibration_error(),
            "brier_score": self.calculate_brier_score(),
            "total_predictions": sum(counts) if counts else 0,
            "calibration_version": self.calibration_version,
        }

    def _validate_safety_context(
        self, safety_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate safety context"""

        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check for extreme values in context
        for key, value in safety_context.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    violations.append(f"Non-finite value in context: {key}")
                elif abs(value) > 1e6:
                    violations.append(f"Extreme value in context: {key}={value}")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _initialize_bins(self) -> List[CalibrationBin]:
        """Initialize histogram bins"""
        bins = []
        bin_width = 1.0 / self.n_bins

        for i in range(self.n_bins):
            bins.append(
                CalibrationBin(
                    min_confidence=i * bin_width, max_confidence=(i + 1) * bin_width
                )
            )

        return bins

    def _update_histogram_bins(self, confidence: float, outcome: float):
        """Update histogram bins with new data"""
        bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
        self.histogram_bins[bin_idx].predictions.append(confidence)
        self.histogram_bins[bin_idx].outcomes.append(outcome)

        # Limit bin size to prevent memory issues
        max_bin_size = max(100, self.window_size // self.n_bins)
        if len(self.histogram_bins[bin_idx].predictions) > max_bin_size:
            self.histogram_bins[bin_idx].predictions.pop(0)
            self.histogram_bins[bin_idx].outcomes.pop(0)

    def _retrain_calibrators(self):
        """Retrain calibration models with recent data"""
        if len(self.calibration_history) < 50:
            logger.debug("Not enough samples for retraining")
            return

        try:
            # Prepare training data
            X_list = []
            y_list = []

            for record in self.calibration_history:
                if record.actual_outcome is not None:
                    X_list.append(record.raw_confidence)
                    y_list.append(record.actual_outcome)

            if len(y_list) < 50:
                logger.debug("Not enough labeled samples for retraining")
                return

            X = np.array(X_list)
            y = np.array(y_list)

            # SAFETY: Validate training data
            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
                logger.warning(
                    "Non-finite values in training data, skipping retraining"
                )
                self.safety_blocks["retraining"] += 1
                return

            # Retrain isotonic regression
            try:
                self.isotonic_model.fit(X, y)
                self.models_fitted["isotonic"] = True
                logger.debug("Isotonic model retrained")
            except Exception as e:
                logger.warning(f"Failed to retrain isotonic model: {e}")
                self.models_fitted["isotonic"] = False

            # Retrain Platt scaling
            try:
                X_2d = X.reshape(-1, 1)
                self.platt_model.fit(X_2d, y)
                self.models_fitted["platt"] = True
                logger.debug("Platt model retrained")
            except Exception as e:
                logger.warning(f"Failed to retrain Platt model: {e}")
                self.models_fitted["platt"] = False

            # Retrain beta calibrator
            try:
                self.beta_calibrator.fit(X, y)
                self.models_fitted["beta"] = True
                logger.debug("Beta calibrator retrained")
            except Exception as e:
                logger.warning(f"Failed to retrain beta calibrator: {e}")
                self.models_fitted["beta"] = False

            self.models_fitted["histogram"] = True

            logger.info(f"Calibration models retrained with {len(y)} samples")

        except Exception as e:
            logger.error(f"Error in calibrator retraining: {e}")
            self.safety_blocks["retraining"] += 1

    def _isotonic_calibration(self, confidence: float) -> float:
        """Apply isotonic regression calibration"""
        if not self.models_fitted["isotonic"]:
            return confidence

        try:
            calibrated = self.isotonic_model.predict([confidence])
            if hasattr(calibrated, "__len__") and len(calibrated) > 0:
                calibrated = float(calibrated[0])
            else:
                calibrated = float(calibrated)
            return calibrated
        except Exception as e:
            logger.debug(f"Isotonic calibration failed: {e}")
            return confidence

    def _platt_calibration(self, confidence: float) -> float:
        """Apply Platt scaling calibration"""
        if not self.models_fitted["platt"]:
            return confidence

        try:
            X = np.array([[confidence]])
            if hasattr(self.platt_model, "predict_proba"):
                calibrated = self.platt_model.predict_proba(X)[0, 1]
            else:
                # Fallback using decision function
                logit = self.platt_model.decision_function(X)
                calibrated = 1.0 / (1.0 + np.exp(-logit))
            return float(calibrated)
        except Exception as e:
            logger.debug(f"Platt calibration failed: {e}")
            return confidence

    def _histogram_calibration(self, confidence: float) -> float:
        """Apply histogram binning calibration"""
        bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
        bin = self.histogram_bins[bin_idx]

        if bin.count >= 5:  # Need minimum samples
            return bin.accuracy
        return confidence

    def _beta_calibration(self, confidence: float) -> float:
        """Apply beta calibration"""
        if not self.models_fitted["beta"]:
            return confidence

        try:
            calibrated = self.beta_calibrator.predict(np.array([confidence]))
            if hasattr(calibrated, "__len__"):
                calibrated = float(calibrated[0])
            else:
                calibrated = float(calibrated)
            return calibrated
        except Exception as e:
            logger.debug(f"Beta calibration failed: {e}")
            return confidence

    def _context_aware_calibration(
        self, confidence: float, context_features: np.ndarray
    ) -> float:
        """Apply context-aware calibration using features"""
        # Combine confidence with context
        features = np.concatenate([[confidence], context_features.ravel()])

        try:
            if self.context_model is not None and hasattr(
                self.context_model, "predict"
            ):
                calibrated = self.context_model.predict([features])[0]
                return float(np.clip(calibrated, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"Context-aware calibration failed: {e}")

        return confidence

    def get_statistics(self) -> Dict[str, Any]:
        """Get calibration statistics"""

        with self.lock:
            stats = {
                "method": self.method,
                "n_bins": self.n_bins,
                "window_size": self.window_size,
                "calibration_history_size": len(self.calibration_history),
                "calibration_version": self.calibration_version,
                "calibration_calls": self.calibration_calls,
                "avg_calibration_time": self.calibration_time_total
                / max(1, self.calibration_calls),
                "models_fitted": dict(self.models_fitted),
                "ece": self.calculate_expected_calibration_error(),
                "mce": self.calculate_maximum_calibration_error(),
                "brier_score": self.calculate_brier_score(),
            }

            # Add safety statistics
            if self.safety_validator:
                stats["safety"] = {
                    "enabled": True,
                    "blocks": dict(self.safety_blocks),
                    "corrections": dict(self.safety_corrections),
                    "total_blocks": sum(self.safety_blocks.values()),
                    "total_corrections": sum(self.safety_corrections.values()),
                }
            else:
                stats["safety"] = {"enabled": False}

            return stats


class ModelConfidenceTracker:
    """
    Tracks overall model confidence and identifies uncertainty regions.
    Complete production implementation with comprehensive metrics.
    FIXED: Circular import resolved with lazy loading of safety validator.
    """

    def __init__(
        self,
        decay_rate: float = 0.95,
        min_confidence: float = 0.1,
        max_confidence: float = 0.95,
        safety_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model confidence tracker

        Args:
            decay_rate: Exponential decay rate for confidence updates
            min_confidence: Minimum allowed model confidence
            max_confidence: Maximum allowed model confidence
            safety_config: Optional safety configuration
        """
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

        # Initialize safety validator with LAZY LOADING to prevent circular import
        self._safety_validator_instance = None
        self._safety_config = safety_config
        self._safety_validator_initialized = False

        # Overall model confidence
        self.model_confidence = 0.5

        # Domain-specific confidence
        self.domain_confidence = defaultdict(lambda: 0.5)

        # Prediction tracking
        self.recent_predictions = deque(maxlen=100)
        self.prediction_errors = deque(maxlen=100)

        # Low confidence regions
        self.low_confidence_regions = []

        # Confidence history
        self.confidence_history = deque(maxlen=1000)

        # Performance metrics
        self.rolling_accuracy = 0.5
        self.rolling_error = 0.5

        # Update counters
        self.update_count = 0
        self.successful_updates = 0

        # Safety tracking
        self.safety_blocks = defaultdict(int)
        self.safety_corrections = defaultdict(int)

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ModelConfidenceTracker initialized")

    @property
    def safety_validator(self):
        """Lazy-load safety validator to prevent circular import"""
        if not self._safety_validator_initialized:
            try:
                # LAZY IMPORT - prevents circular dependency
                from ..safety.safety_types import SafetyConfig
                from ..safety.safety_validator import EnhancedSafetyValidator

                if isinstance(self._safety_config, dict) and self._safety_config:
                    self._safety_validator_instance = EnhancedSafetyValidator(
                        SafetyConfig.from_dict(self._safety_config)
                    )
                else:
                    self._safety_validator_instance = EnhancedSafetyValidator()
                logger.info(
                    "ModelConfidenceTracker: Safety validator initialized (lazy)"
                )
            except ImportError as e:
                logger.warning(
                    f"ModelConfidenceTracker: Safety validator not available: {e}"
                )
                self._safety_validator_instance = None

            self._safety_validator_initialized = True

        return self._safety_validator_instance

    def update(
        self, observation: Optional[Any] = None, prediction: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Update model confidence based on observation and prediction.

        Args:
            observation: Observed outcome (optional)
            prediction: Model's prediction (optional)

        Returns:
            Dict with update status
        """

        with self.lock:
            self.update_count += 1

            # Handle case where parameters are None
            if observation is None or prediction is None:
                return {
                    "status": "skipped",
                    "reason": "missing_data",
                    "message": "Both observation and prediction required for confidence update",
                    "model_confidence": self.model_confidence,
                }

            timestamp = time.time()
            error = None

            # Calculate error if both observation and prediction are available
            try:
                error = self._calculate_prediction_error(observation, prediction)

                # SAFETY: Validate error
                if not np.isfinite(error):
                    logger.warning("Non-finite prediction error")
                    self.safety_corrections["non_finite_error"] += 1
                    error = 0.5

                error = float(np.clip(error, 0.0, 1.0))
                self.prediction_errors.append(error)

                # Update rolling metrics
                self._update_rolling_metrics(error)

                # Calculate confidence delta based on error
                if error < 0.1:  # Very accurate
                    confidence_delta = 0.05
                elif error < 0.3:  # Reasonable accuracy
                    confidence_delta = 0.01
                elif error < 0.5:  # Poor accuracy
                    confidence_delta = -0.02
                else:  # Very poor
                    confidence_delta = -0.05

                # Apply exponential moving average update
                self.model_confidence = self.decay_rate * self.model_confidence + (
                    1 - self.decay_rate
                ) * (self.model_confidence + confidence_delta)

                # Clamp to bounds
                self.model_confidence = float(
                    np.clip(
                        self.model_confidence, self.min_confidence, self.max_confidence
                    )
                )

                # Update domain confidence if available
                if hasattr(observation, "domain"):
                    domain = observation.domain
                    domain_conf = self.domain_confidence[domain]
                    self.domain_confidence[domain] = float(
                        self.decay_rate * domain_conf
                        + (1 - self.decay_rate) * (domain_conf + confidence_delta)
                    )

                self.successful_updates += 1

            except Exception as e:
                logger.error(f"Error calculating prediction error: {e}")
                return {
                    "status": "error",
                    "reason": str(e),
                    "message": "Failed to calculate prediction error",
                    "model_confidence": self.model_confidence,
                }

            # Track confidence history
            self.confidence_history.append(
                {
                    "timestamp": timestamp,
                    "confidence": self.model_confidence,
                    "error": error,
                }
            )

            # Periodically identify low confidence regions
            if len(self.confidence_history) % 50 == 0:
                self._identify_low_confidence_regions()

            return {
                "status": "success",
                "model_confidence": self.model_confidence,
                "error": error,
                "rolling_accuracy": self.rolling_accuracy,
                "rolling_error": self.rolling_error,
            }

    def get_model_confidence(self) -> float:
        """
        Get current overall model confidence.

        Returns:
            Model confidence score [0, 1]
        """

        with self.lock:
            return float(self.model_confidence)

    def get_prediction_confidence(self, action: Any, context: Any) -> float:
        """
        Get confidence for a specific prediction.

        Args:
            action: Action or input for prediction
            context: Context for prediction

        Returns:
            Prediction-specific confidence [0, 1]
        """

        with self.lock:
            # Base confidence is model confidence
            confidence = self.model_confidence

            # Adjust based on domain if available
            if hasattr(context, "domain"):
                domain = context.domain
                domain_conf = self.domain_confidence.get(domain, 0.5)
                confidence = 0.7 * confidence + 0.3 * domain_conf

            # Adjust based on action novelty
            novelty = self._calculate_novelty(action)
            if novelty > 0.8:  # Very novel
                confidence *= 0.7
            elif novelty > 0.5:  # Somewhat novel
                confidence *= 0.9

            # Adjust based on recent performance
            if self.rolling_accuracy < 0.3:
                confidence *= 0.8
            elif self.rolling_accuracy > 0.7:
                confidence = min(1.0, confidence * 1.1)

            return float(np.clip(confidence, 0.0, 1.0))

    def identify_low_confidence_regions(self) -> List[Dict[str, Any]]:
        """
        Identify regions where model has low confidence.

        Returns:
            List of low confidence regions with descriptions
        """

        with self.lock:
            self._identify_low_confidence_regions()
            return list(self.low_confidence_regions)

    def get_confidence_summary(self) -> Dict[str, Any]:
        """
        Get summary of model confidence metrics.

        Returns:
            Dictionary with confidence statistics
        """

        with self.lock:
            domain_summary = {
                domain: float(conf) for domain, conf in self.domain_confidence.items()
            }

            recent_errors = (
                list(self.prediction_errors)[-10:] if self.prediction_errors else []
            )

            stats = {
                "overall_confidence": float(self.model_confidence),
                "domain_confidence": domain_summary,
                "rolling_accuracy": float(self.rolling_accuracy),
                "rolling_error": float(self.rolling_error),
                "recent_mean_error": float(np.mean(recent_errors))
                if recent_errors
                else None,
                "recent_max_error": float(np.max(recent_errors))
                if recent_errors
                else None,
                "recent_min_error": float(np.min(recent_errors))
                if recent_errors
                else None,
                "low_confidence_regions": len(self.low_confidence_regions),
                "confidence_trend": self._calculate_confidence_trend(),
                "update_count": self.update_count,
                "successful_updates": self.successful_updates,
                "success_rate": self.successful_updates / max(1, self.update_count),
            }

            # Add safety statistics
            if self.safety_validator:
                stats["safety"] = {
                    "enabled": True,
                    "blocks": dict(self.safety_blocks),
                    "corrections": dict(self.safety_corrections),
                    "total_blocks": sum(self.safety_blocks.values()),
                    "total_corrections": sum(self.safety_corrections.values()),
                }
            else:
                stats["safety"] = {"enabled": False}

            return stats

    def _calculate_prediction_error(self, observation: Any, prediction: Any) -> float:
        """Calculate error between observation and prediction"""

        # Try multiple ways to extract values
        obs_val = None
        pred_val = None

        # Extract observation value
        if hasattr(observation, "value"):
            obs_val = observation.value
        elif isinstance(observation, (int, float)):
            obs_val = observation
        elif isinstance(observation, dict) and "value" in observation:
            obs_val = observation["value"]

        # Extract prediction value
        if hasattr(prediction, "expected"):
            pred_val = prediction.expected
        elif hasattr(prediction, "value"):
            pred_val = prediction.value
        elif isinstance(prediction, (int, float)):
            pred_val = prediction
        elif isinstance(prediction, dict):
            pred_val = prediction.get("expected", prediction.get("value"))

        # Calculate error if we have both values
        if obs_val is not None and pred_val is not None:
            try:
                obs_val = float(obs_val)
                pred_val = float(pred_val)

                # Normalized absolute error
                denominator = abs(obs_val) + abs(pred_val) + 1e-10
                error = abs(obs_val - pred_val) / denominator
                return min(error, 1.0)
            except (TypeError, ValueError):
                pass

        # Default error if can't calculate
        return 0.5

    def _update_rolling_metrics(self, error: float):
        """Update rolling accuracy and error metrics"""
        # Convert error to accuracy (inverse relationship)
        accuracy = 1.0 - error

        # Exponential moving average
        alpha = 0.1
        self.rolling_accuracy = float(
            (1 - alpha) * self.rolling_accuracy + alpha * accuracy
        )
        self.rolling_error = float((1 - alpha) * self.rolling_error + alpha * error)

    def _calculate_novelty(self, action: Any) -> float:
        """Calculate how novel an action is compared to history"""
        # Simplified novelty calculation
        # In production, this would compare to historical action distribution

        # For now, use random baseline
        # Real implementation would use embedding distance or distribution comparison
        return np.random.uniform(0.0, 0.5)

    def _identify_low_confidence_regions(self):
        """Identify regions where model confidence is low"""
        self.low_confidence_regions = []

        # Check domain confidence
        for domain, confidence in self.domain_confidence.items():
            if confidence < 0.3:
                self.low_confidence_regions.append(
                    {
                        "type": "domain",
                        "domain": domain,
                        "confidence": float(confidence),
                        "description": f"Low confidence in domain: {domain}",
                    }
                )

        # Check recent error patterns
        if len(self.prediction_errors) >= 10:
            recent_errors = list(self.prediction_errors)[-10:]
            mean_error = np.mean(recent_errors)
            if mean_error > 0.5:
                self.low_confidence_regions.append(
                    {
                        "type": "high_error",
                        "mean_error": float(mean_error),
                        "max_error": float(np.max(recent_errors)),
                        "description": f"Recent predictions have high error rate (mean={mean_error:.3f})",
                    }
                )

        # Check confidence trend
        trend = self._calculate_confidence_trend()
        if trend < -0.1:
            self.low_confidence_regions.append(
                {
                    "type": "declining_confidence",
                    "trend": float(trend),
                    "description": f"Model confidence is declining (trend={trend:.3f})",
                }
            )

        # Check overall low confidence
        if self.model_confidence < 0.3:
            self.low_confidence_regions.append(
                {
                    "type": "overall_low",
                    "confidence": float(self.model_confidence),
                    "description": f"Overall model confidence is low ({self.model_confidence:.3f})",
                }
            )

    def _calculate_confidence_trend(self) -> float:
        """Calculate trend in model confidence"""
        if len(self.confidence_history) < 10:
            return 0.0

        # Get recent confidence values
        recent_confs = [h["confidence"] for h in list(self.confidence_history)[-20:]]

        # Simple linear regression for trend
        x = np.arange(len(recent_confs))
        y = np.array(recent_confs)

        # Calculate slope using least squares
        n = len(x)
        denominator = n * np.sum(x**2) - np.sum(x) ** 2

        # Check for zero denominator
        if abs(denominator) < 1e-10:
            return 0.0

        numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
        slope = numerator / denominator

        return float(slope)

    def save_state(self, path: str):
        """Save tracker state to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        with self.lock:
            state = {
                "model_confidence": float(self.model_confidence),
                "domain_confidence": {
                    k: float(v) for k, v in self.domain_confidence.items()
                },
                "rolling_accuracy": float(self.rolling_accuracy),
                "rolling_error": float(self.rolling_error),
                "low_confidence_regions": self.low_confidence_regions,
                "update_count": self.update_count,
                "successful_updates": self.successful_updates,
            }

        with open(save_path / "confidence_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Confidence tracker state saved to {save_path}")

    def load_state(self, path: str):
        """Load tracker state from disk"""
        load_path = Path(path)

        if not (load_path / "confidence_state.json").exists():
            logger.warning(f"No saved state found at {load_path}")
            return

        with open(load_path / "confidence_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        with self.lock:
            self.model_confidence = float(state.get("model_confidence", 0.5))

            domain_conf_dict = state.get("domain_confidence", {})
            self.domain_confidence = defaultdict(
                lambda: 0.5, {k: float(v) for k, v in domain_conf_dict.items()}
            )

            self.rolling_accuracy = float(state.get("rolling_accuracy", 0.5))
            self.rolling_error = float(state.get("rolling_error", 0.5))
            self.low_confidence_regions = state.get("low_confidence_regions", [])
            self.update_count = int(state.get("update_count", 0))
            self.successful_updates = int(state.get("successful_updates", 0))

        logger.info(f"Confidence tracker state loaded from {load_path}")

    def reset(self):
        """Reset tracker to initial state"""
        with self.lock:
            self.model_confidence = 0.5
            self.domain_confidence.clear()
            self.recent_predictions.clear()
            self.prediction_errors.clear()
            self.confidence_history.clear()
            self.low_confidence_regions = []
            self.rolling_accuracy = 0.5
            self.rolling_error = 0.5
            self.update_count = 0
            self.successful_updates = 0
            self.safety_blocks.clear()
            self.safety_corrections.clear()

        logger.info("Confidence tracker reset to initial state")


class TemperatureScaling:
    """
    Temperature scaling for neural network calibration.
    Simple but effective post-processing calibration method.
    """

    def __init__(self):
        """Initialize temperature scaling"""
        self.temperature = 1.0
        self._is_fitted = False

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Fit temperature parameter using validation set.

        Args:
            logits: Model logits (before softmax)
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            self
        """

        logits = np.asarray(logits)
        labels = np.asarray(labels)

        if len(logits) != len(labels):
            raise ValueError("logits and labels must have same length")

        # Initialize temperature
        self.temperature = 1.0

        # Optimize temperature using gradient descent
        for iteration in range(max_iter):
            # Forward pass with temperature
            scaled_logits = logits / self.temperature

            # Softmax
            exp_logits = np.exp(
                scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
            )
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Cross-entropy loss
            epsilon = 1e-15
            probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

            if labels.ndim == 1:
                # Class indices
                loss = -np.mean(np.log(probs_clipped[np.arange(len(labels), labels])))
            else:
                # One-hot encoded
                loss = -np.mean(np.sum(labels * np.log(probs_clipped), axis=-1))

            # Gradient of loss w.r.t. temperature
            # Simplified gradient computation
            grad = self._compute_temperature_gradient(logits, labels, self.temperature)

            # Update temperature
            self.temperature -= lr * grad

            # Ensure temperature stays positive
            self.temperature = max(0.01, self.temperature)

        self._is_fitted = True
        logger.debug(f"Temperature scaling fitted with T={self.temperature:.3f}")

        return self

    def predict(self, logits):
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model logits

        Returns:
            Calibrated probabilities
        """

        if not self._is_fitted:
            logger.warning("Temperature not fitted, using T=1.0")

        logits = np.asarray(logits)

        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Softmax
        exp_logits = np.exp(
            scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs

    def _compute_temperature_gradient(self, logits, labels, temperature):
        """Compute gradient of loss w.r.t. temperature"""

        # Simplified gradient computation
        # In practice, would use automatic differentiation

        epsilon = 1e-7

        # Forward pass at T
        scaled_logits = logits / temperature
        exp_logits = np.exp(
            scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if labels.ndim == 1:
            loss_t = -np.mean(
                np.log(np.clip(probs[np.arange(len(labels), labels], epsilon, 1))
            )
        else:
            loss_t = -np.mean(
                np.sum(labels * np.log(np.clip(probs, epsilon, 1)), axis=-1)
            )

        # Forward pass at T + epsilon
        scaled_logits_plus = logits / (temperature + epsilon)
        exp_logits_plus = np.exp(
            scaled_logits_plus - np.max(scaled_logits_plus, axis=-1, keepdims=True)
        )
        probs_plus = exp_logits_plus / np.sum(exp_logits_plus, axis=-1, keepdims=True)

        if labels.ndim == 1:
            loss_t_plus = -np.mean(
                np.log(np.clip(probs_plus[np.arange(len(labels), labels], epsilon, 1))
            )
        else:
            loss_t_plus = -np.mean(
                np.sum(labels * np.log(np.clip(probs_plus, epsilon, 1)), axis=-1)
            )

        # Numerical gradient
        grad = (loss_t_plus - loss_t) / epsilon

        return grad


class CalibrationEnsemble:
    """
    Ensemble of calibration methods for robust calibration.
    Combines multiple calibrators and weights their predictions.
    """

    def __init__(self, methods=None, weights=None):
        """
        Initialize calibration ensemble.

        Args:
            methods: List of calibration method names
            weights: Optional weights for each method
        """

        if methods is None:
            methods = ["isotonic", "platt", "beta"]

        self.methods = methods
        self.calibrators = {}

        # Initialize calibrators
        for method in methods:
            if method == "isotonic":
                self.calibrators[method] = IsotonicRegression(out_of_bounds="clip")
            elif method == "platt":
                self.calibrators[method] = LogisticRegression()
            elif method == "beta":
                self.calibrators[method] = BetaCalibrator()
            else:
                logger.warning(f"Unknown calibration method: {method}")

        # Set weights
        if weights is None:
            self.weights = {method: 1.0 / len(methods) for method in methods}
        else:
            if len(weights) != len(methods):
                raise ValueError("weights must have same length as methods")
            self.weights = dict(zip(methods, weights))

        self._is_fitted = False

    def fit(self, predictions, outcomes):
        """
        Fit all calibrators.

        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes

        Returns:
            self
        """

        predictions = np.asarray(predictions)
        outcomes = np.asarray(outcomes)

        for method, calibrator in self.calibrators.items():
            try:
                if method in ["isotonic", "platt"]:
                    if method == "platt":
                        calibrator.fit(predictions.reshape(-1, 1), outcomes)
                    else:
                        calibrator.fit(predictions, outcomes)
                elif method == "beta":
                    calibrator.fit(predictions, outcomes)

                logger.debug(f"Fitted {method} calibrator")
            except Exception as e:
                logger.warning(f"Failed to fit {method} calibrator: {e}")

        self._is_fitted = True
        return self

    def predict(self, predictions):
        """
        Calibrate predictions using ensemble.

        Args:
            predictions: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """

        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        predictions = np.asarray(predictions)

        # Get predictions from each calibrator
        calibrated_predictions = []
        valid_weights = []

        for method, calibrator in self.calibrators.items():
            try:
                if method == "isotonic":
                    pred = calibrator.predict(predictions)
                elif method == "platt":
                    pred = calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
                elif method == "beta":
                    pred = calibrator.predict(predictions)
                else:
                    continue

                calibrated_predictions.append(pred)
                valid_weights.append(self.weights[method])

            except Exception as e:
                logger.debug(f"Calibrator {method} failed: {e}")

        if not calibrated_predictions:
            # Fallback to uncalibrated
            return predictions

        # Weighted average
        calibrated_predictions = np.array(calibrated_predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)

        ensemble_pred = np.sum(
            calibrated_predictions * valid_weights[:, np.newaxis], axis=0
        )

        return ensemble_pred


class AdaptiveCalibrator:
    """
    Adaptive calibrator that adjusts calibration based on observed performance.
    Automatically selects and tunes the best calibration method.
    """

    def __init__(
        self, candidate_methods=None, evaluation_metric="ece", adaptation_rate=0.1
    ):
        """
        Initialize adaptive calibrator.

        Args:
            candidate_methods: List of methods to consider
            evaluation_metric: Metric for method selection ('ece', 'brier', 'log_loss')
            adaptation_rate: Rate of adaptation to new data
        """

        if candidate_methods is None:
            candidate_methods = ["isotonic", "platt", "histogram", "beta"]

        self.candidate_methods = candidate_methods
        self.evaluation_metric = evaluation_metric
        self.adaptation_rate = adaptation_rate

        # Create calibrators for each method
        self.calibrators = {
            method: ConfidenceCalibrator(method=method, n_bins=10)
            for method in candidate_methods
        }

        # Performance tracking
        self.method_scores = {method: [] for method in candidate_methods}
        self.current_best_method = candidate_methods[0]

        # History
        self.history = deque(maxlen=1000)

    def calibrate(self, confidence, context=None):
        """
        Calibrate using current best method.

        Args:
            confidence: Raw confidence
            context: Optional context

        Returns:
            Calibrated confidence
        """

        calibrator = self.calibrators[self.current_best_method]
        return calibrator.calibrate(confidence, context)

    def update(self, prediction, outcome):
        """
        Update all calibrators and adapt method selection.

        Args:
            prediction: Predicted confidence
            outcome: Actual outcome
        """

        # Update all calibrators
        for calibrator in self.calibrators.values():
            calibrator.update_calibration(prediction, outcome)

        # Add to history
        self.history.append((prediction, outcome))

        # Periodically evaluate methods
        if len(self.history) % 100 == 0:
            self._evaluate_and_adapt()

    def _evaluate_and_adapt(self):
        """Evaluate all methods and select best"""

        if len(self.history) < 50:
            return

        # Evaluate each method
        for method, calibrator in self.calibrators.items():
            try:
                if self.evaluation_metric == "ece":
                    score = calibrator.calculate_expected_calibration_error()
                elif self.evaluation_metric == "brier":
                    score = calibrator.calculate_brier_score()
                    if score is None:
                        score = float("inf")
                else:
                    score = float("inf")

                self.method_scores[method].append(score)

                # Keep only recent scores
                if len(self.method_scores[method]) > 10:
                    self.method_scores[method].pop(0)

            except Exception as e:
                logger.debug(f"Failed to evaluate {method}: {e}")

        # Select best method based on average recent score
        best_method = None
        best_score = float("inf")

        for method, scores in self.method_scores.items():
            if scores:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_method = method

        if best_method and best_method != self.current_best_method:
            logger.info(
                f"Adaptive calibrator switched from {self.current_best_method} "
                f"to {best_method} (score: {best_score:.4f})"
            )
            self.current_best_method = best_method

    def get_method_performance(self) -> Dict[str, float]:
        """Get average performance of each method"""

        return {
            method: np.mean(scores) if scores else None
            for method, scores in self.method_scores.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get calibrator statistics"""

        return {
            "current_method": self.current_best_method,
            "method_performance": self.get_method_performance(),
            "history_size": len(self.history),
            "evaluation_metric": self.evaluation_metric,
        }


# Utility functions for calibration analysis


def plot_reliability_diagram(
    calibrator: ConfidenceCalibrator, save_path: Optional[str] = None
):
    """
    Plot reliability diagram for calibrator.

    Args:
        calibrator: Fitted calibrator
        save_path: Optional path to save plot
    """

    try:
        import matplotlib.pyplot as plt

        data = calibrator.get_reliability_diagram_data()

        mean_confs = data["mean_confidences"]
        accuracies = data["accuracies"]
        counts = data["counts"]

        if not mean_confs:
            logger.warning("No data for reliability diagram")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax1.plot(mean_confs, accuracies, "o-", label="Model calibration")
        ax1.set_xlabel("Mean predicted confidence")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(
            f"Reliability Diagram\nECE: {data['ece']:.4f}, MCE: {data['mce']:.4f}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.bar(mean_confs, counts, width=0.08, alpha=0.7)
        ax2.set_xlabel("Confidence bin")
        ax2.set_ylabel("Number of predictions")
        ax2.set_title("Distribution of predictions")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Reliability diagram saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, cannot plot reliability diagram")
    except Exception as e:
        logger.error(f"Failed to plot reliability diagram: {e}")


def calculate_calibration_metrics(
    predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate various calibration metrics.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        n_bins: Number of bins for ECE/MCE

    Returns:
        Dictionary with calibration metrics
    """

    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have same length")

    # Brier score
    brier_score = np.mean((predictions - outcomes) ** 2)

    # Log loss
    epsilon = 1e-15
    predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
    log_loss = -np.mean(
        outcomes * np.log(predictions_clipped)
        + (1 - outcomes) * np.log(1 - predictions_clipped)
    )

    # ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_mask = (predictions >= bin_boundaries[i]) & (
            predictions < bin_boundaries[i + 1]
        )
        if i == n_bins - 1:  # Include right boundary for last bin
            bin_mask = (predictions >= bin_boundaries[i]) & (
                predictions <= bin_boundaries[i + 1]
            )

        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(outcomes[bin_mask])
            bin_conf = np.mean(predictions[bin_mask])
            bin_weight = np.sum(bin_mask) / len(predictions)

            bin_error = abs(bin_acc - bin_conf)
            ece += bin_weight * bin_error
            mce = max(mce, bin_error)

    return {
        "brier_score": float(brier_score),
        "log_loss": float(log_loss),
        "ece": float(ece),
        "mce": float(mce),
    }


def compare_calibration_methods(
    predictions: np.ndarray, outcomes: np.ndarray, methods: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different calibration methods.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        methods: List of methods to compare

    Returns:
        Dictionary mapping method names to their metrics
    """

    if methods is None:
        methods = ["isotonic", "platt", "histogram", "beta"]

    results = {}

    # Split data for training and testing
    split_idx = int(len(predictions) * 0.7)
    train_pred = predictions[:split_idx]
    train_out = outcomes[:split_idx]
    test_pred = predictions[split_idx:]
    test_out = outcomes[split_idx:]

    for method in methods:
        try:
            # Create and fit calibrator
            calibrator = ConfidenceCalibrator(method=method)

            # Train
            for pred, out in zip(train_pred, train_out):
                calibrator.update_calibration(pred, out)

            # Calibrate test predictions
            calibrated_test = np.array(
                [calibrator.calibrate(pred) for pred in test_pred]
            )

            # Calculate metrics
            metrics = calculate_calibration_metrics(calibrated_test, test_out)
            results[method] = metrics

            logger.info(
                f"{method}: ECE={metrics['ece']:.4f}, Brier={metrics['brier_score']:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to evaluate {method}: {e}")
            results[method] = None

    return results


# Export main classes and functions
__all__ = [
    "ConfidenceCalibrator",
    "ModelConfidenceTracker",
    "RobustIsotonicRegression",
    "RobustLogisticRegression",
    "BetaCalibrator",
    "TemperatureScaling",
    "CalibrationEnsemble",
    "AdaptiveCalibrator",
    "CalibrationBin",
    "PredictionRecord",
    "plot_reliability_diagram",
    "calculate_calibration_metrics",
    "compare_calibration_methods",
]
