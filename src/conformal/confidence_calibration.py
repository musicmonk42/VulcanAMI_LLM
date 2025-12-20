"""
Confidence Calibration for Tool Selection System

Implements various calibration methods to ensure confidence scores are well-calibrated,
including temperature scaling, isotonic regression, and conformal prediction.
"""

import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .security_fixes import safe_pickle_load

# Make seaborn optional since it's not actually used in the code
try:
    import seaborn as sns
except ImportError:
    sns = None

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Data point for calibration"""

    prediction: float  # Predicted confidence/probability
    actual: bool  # Actual outcome (success/failure)
    features: Optional[np.ndarray] = None
    tool_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality"""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score
    log_loss: float  # Negative log-likelihood
    reliability: float  # Reliability component
    resolution: float  # Resolution component
    uncertainty: float  # Uncertainty component
    sharpness: float  # Sharpness (concentration of predictions)

    def to_dict(self) -> Dict[str, float]:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "sharpness": self.sharpness,
        }


class TemperatureScaling:
    """Temperature scaling for calibration"""

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Fit temperature parameter

        Args:
            logits: Raw model outputs (before softmax)
            labels: True labels
        """

        def nll_loss(temp):
            """Negative log-likelihood loss"""
            scaled_logits = logits / temp
            probs = self._softmax(scaled_logits)
            # Clip for numerical stability
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            if len(probs.shape) == 1:
                # Binary case
                loss = -np.mean(
                    labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
                )
            else:
                # Multi-class case
                loss = -np.mean(np.sum(labels * np.log(probs), axis=1))

            return loss

        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, bounds=[(0.01, 10.0)], method="L-BFGS-B")

        self.temperature = result.x[0]
        self.fitted = True

        logger.info(f"Temperature scaling fitted: T={self.temperature:.3f}")

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""

        if not self.fitted:
            return self._softmax(logits)

        scaled_logits = logits / self.temperature
        return self._softmax(scaled_logits)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation"""

        if len(x.shape) == 1:
            # Binary case - sigmoid
            return 1 / (1 + np.exp(-x))
        else:
            # Multi-class case
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class IsotonicCalibration:
    """Isotonic regression calibration"""

    def __init__(self):
        self.isotonic = None
        self.fitted = False

    def fit(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Fit isotonic regression

        Args:
            predictions: Predicted probabilities
            labels: True labels
        """

        self.isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

        self.isotonic.fit(predictions, labels)
        self.fitted = True

        logger.info("Isotonic regression calibration fitted")

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration"""

        if not self.fitted:
            return predictions

        # Handle edge cases
        predictions = np.clip(predictions, 0, 1)

        return self.isotonic.transform(predictions)


class PlattScaling:
    """Platt scaling (sigmoid calibration)"""

    def __init__(self):
        self.model = None
        self.fitted = False

    def fit(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling

        Args:
            predictions: Predicted probabilities
            labels: True labels
        """

        # Use logistic regression
        self.model = LogisticRegression()

        # Reshape for sklearn
        X = predictions.reshape(-1, 1)

        self.model.fit(X, labels)
        self.fitted = True

        logger.info("Platt scaling calibration fitted")

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Platt scaling"""

        if not self.fitted:
            return predictions

        X = predictions.reshape(-1, 1)

        # Get calibrated probabilities
        calibrated = self.model.predict_proba(X)[:, 1]

        return calibrated


class BetaCalibration:
    """Beta calibration for probability calibration"""

    def __init__(self):
        self.alpha = 1.0
        self.beta_param = 1.0
        self.fitted = False

    def fit(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Fit Beta distribution parameters

        Args:
            predictions: Predicted probabilities
            labels: True labels
        """

        # Separate into positive and negative classes
        pos_preds = predictions[labels == 1]
        neg_preds = predictions[labels == 0]

        if len(pos_preds) > 0 and len(neg_preds) > 0:
            # Fit Beta distributions
            self.alpha = np.mean(pos_preds) * 10  # Simple heuristic
            self.beta_param = np.mean(1 - neg_preds) * 10

            self.fitted = True
            logger.info(
                f"Beta calibration fitted: α={self.alpha:.3f}, β={self.beta_param:.3f}"
            )

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Beta calibration"""

        if not self.fitted:
            return predictions

        # Apply Beta CDF transformation
        calibrated = beta.cdf(predictions, self.alpha, self.beta_param)

        return calibrated


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification"""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor

        Args:
            alpha: Significance level (1 - alpha is coverage guarantee)
        """
        self.alpha = alpha
        self.calibration_scores = []
        self.fitted = False

        logger.info(
            f"ConformalPredictor initialized (alpha={alpha}, coverage_guarantee={1-alpha:.1%})"
        )

    def fit(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Fit conformal predictor

        Args:
            predictions: Predicted probabilities
            labels: True labels
        """

        # Compute nonconformity scores
        self.calibration_scores = []

        for pred, label in zip(predictions, labels):
            if label == 1:
                score = 1 - pred  # Error when predicting positive
            else:
                score = pred  # Error when predicting negative

            self.calibration_scores.append(score)

        self.calibration_scores = np.array(self.calibration_scores)
        self.fitted = True

        logger.info(
            f"Conformal predictor fitted with {len(self.calibration_scores)} scores"
        )

    def predict_set(self, prediction: float) -> Tuple[bool, bool, float]:
        """
        Predict set with coverage guarantee

        Returns:
            (include_negative, include_positive, p_value)
        """

        if not self.fitted:
            return True, True, 0.5

        # Compute p-values for each class
        p_neg = self._compute_p_value(prediction, 0)
        p_pos = self._compute_p_value(prediction, 1)

        # Include in prediction set if p-value > alpha
        include_neg = p_neg > self.alpha
        include_pos = p_pos > self.alpha

        return include_neg, include_pos, max(p_neg, p_pos)

    def _compute_p_value(self, prediction: float, label: int) -> float:
        """Compute p-value for a label"""

        if label == 1:
            score = 1 - prediction
        else:
            score = prediction

        # Proportion of calibration scores >= current score
        p_value = np.mean(self.calibration_scores >= score)

        return p_value


class CalibratedDecisionMaker:
    """
    Main calibration system integrating multiple methods
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize calibrated decision maker

        Args:
            n_bins: Number of bins for ECE calculation
        """
        self.n_bins = n_bins

        # Calibrators for each tool
        self.calibrators = defaultdict(dict)

        # Calibration data
        self.calibration_data = defaultdict(list)

        # Metrics
        self.metrics = defaultdict(lambda: None)

        # Default calibration method priorities
        self.calibration_methods = ["isotonic", "temperature", "platt", "beta"]

    def add_observation(
        self,
        tool_name: str,
        prediction: float,
        actual: bool,
        features: Optional[np.ndarray] = None,
    ):
        """Add calibration observation"""

        data_point = CalibrationData(
            prediction=prediction, actual=actual, features=features, tool_name=tool_name
        )

        self.calibration_data[tool_name].append(data_point)

    def fit_calibration(self, tool_name: str, method: Optional[str] = None):
        """
        Fit calibration for a tool

        Args:
            tool_name: Name of the tool
            method: Specific calibration method or None for auto-select
        """

        if tool_name not in self.calibration_data:
            logger.warning(f"No calibration data for {tool_name}")
            return

        data = self.calibration_data[tool_name]

        if len(data) < 50:
            logger.warning(f"Insufficient data for calibration: {len(data)} samples")
            return

        # Extract predictions and labels
        predictions = np.array([d.prediction for d in data])
        labels = np.array([d.actual for d in data])

        # Fit different calibrators
        if method == "temperature" or method is None:
            temp_cal = TemperatureScaling()
            # Need logits for temperature scaling
            logits = self._predictions_to_logits(predictions)
            temp_cal.fit(logits, labels)
            self.calibrators[tool_name]["temperature"] = temp_cal

        if method == "isotonic" or method is None:
            iso_cal = IsotonicCalibration()
            iso_cal.fit(predictions, labels)
            self.calibrators[tool_name]["isotonic"] = iso_cal

        if method == "platt" or method is None:
            platt_cal = PlattScaling()
            platt_cal.fit(predictions, labels)
            self.calibrators[tool_name]["platt"] = platt_cal

        if method == "beta" or method is None:
            beta_cal = BetaCalibration()
            beta_cal.fit(predictions, labels)
            self.calibrators[tool_name]["beta"] = beta_cal

        # Fit conformal predictor
        conf_cal = ConformalPredictor()
        conf_cal.fit(predictions, labels)
        self.calibrators[tool_name]["conformal"] = conf_cal

        # Compute metrics
        self.metrics[tool_name] = self.compute_metrics(predictions, labels)

        logger.info(
            f"Calibration fitted for {tool_name}: ECE={self.metrics[tool_name].ece:.4f}"
        )

    def calibrate_confidence(
        self,
        tool_name: str,
        raw_confidence: float,
        features: Optional[np.ndarray] = None,
        method: Optional[str] = None,
    ) -> float:
        """
        Calibrate raw confidence score

        Args:
            tool_name: Name of the tool
            raw_confidence: Raw confidence score
            features: Optional features for context
            method: Specific calibration method or None for auto-select

        Returns:
            Calibrated confidence
        """

        if tool_name not in self.calibrators:
            return raw_confidence

        # Select calibration method
        if method is None:
            # Use first available method in priority order
            for method_name in self.calibration_methods:
                if method_name in self.calibrators[tool_name]:
                    method = method_name
                    break

        if method not in self.calibrators[tool_name]:
            return raw_confidence

        calibrator = self.calibrators[tool_name][method]

        # Apply calibration
        if method == "temperature":
            logit = self._prediction_to_logit(raw_confidence)
            calibrated = calibrator.calibrate(np.array([logit]))[0]
        else:
            calibrated = calibrator.calibrate(np.array([raw_confidence]))[0]

        # Ensure valid probability
        calibrated = np.clip(calibrated, 0.001, 0.999)

        return float(calibrated)

    def get_prediction_set(
        self, tool_name: str, confidence: float, alpha: float = 0.1
    ) -> Tuple[bool, bool, float]:
        """
        Get conformal prediction set

        Returns:
            (include_negative, include_positive, p_value)
        """

        if tool_name not in self.calibrators:
            return True, True, 0.5

        if "conformal" not in self.calibrators[tool_name]:
            return True, True, 0.5

        conf_predictor = self.calibrators[tool_name]["conformal"]
        conf_predictor.alpha = alpha

        return conf_predictor.predict_set(confidence)

    def compute_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> CalibrationMetrics:
        """Compute calibration metrics"""

        # FIXED: Validate input arrays are not empty
        if len(predictions) == 0 or len(labels) == 0:
            return CalibrationMetrics(
                ece=0.0,
                mce=0.0,
                brier_score=0.0,
                log_loss=float("inf"),
                reliability=0.0,
                resolution=0.0,
                uncertainty=0.0,
                sharpness=0.0,
            )

        # Expected Calibration Error
        ece = self._compute_ece(predictions, labels)

        # Maximum Calibration Error
        mce = self._compute_mce(predictions, labels)

        # Brier Score
        brier_score = np.mean((predictions - labels) ** 2)

        # Log Loss
        eps = 1e-10
        log_loss = -np.mean(
            labels * np.log(predictions + eps)
            + (1 - labels) * np.log(1 - predictions + eps)
        )

        # Reliability-Resolution-Uncertainty decomposition
        rel, res, unc = self._reliability_resolution_uncertainty(predictions, labels)

        # Sharpness (concentration of predictions)
        sharpness = np.std(predictions)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            log_loss=log_loss,
            reliability=rel,
            resolution=res,
            uncertainty=unc,
            sharpness=sharpness,
        )

    def _compute_ece(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Expected Calibration Error"""

        # FIXED: Protect against empty array division
        if len(predictions) == 0:
            return 0.0

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            # Get predictions in this bin
            in_bin = (predictions > bin_boundaries[i]) & (
                predictions <= bin_boundaries[i + 1]
            )

            if np.sum(in_bin) > 0:
                # Compute accuracy and confidence in bin
                bin_accuracy = np.mean(labels[in_bin])
                bin_confidence = np.mean(predictions[in_bin])
                bin_weight = np.sum(in_bin) / len(predictions)

                # Add to ECE
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        return ece

    def _compute_mce(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Maximum Calibration Error"""

        # FIXED: Protect against empty array
        if len(predictions) == 0:
            return 0.0

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (
                predictions <= bin_boundaries[i + 1]
            )

            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(labels[in_bin])
                bin_confidence = np.mean(predictions[in_bin])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce

    def _reliability_resolution_uncertainty(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute reliability, resolution, and uncertainty"""

        # FIXED: Protect against empty array division
        if len(predictions) == 0:
            return 0.0, 0.0, 0.0

        # Overall accuracy
        base_rate = np.mean(labels)

        # Uncertainty
        uncertainty = base_rate * (1 - base_rate)

        # Resolution and Reliability
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        reliability = 0.0
        resolution = 0.0

        for i in range(self.n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (
                predictions <= bin_boundaries[i + 1]
            )

            if np.sum(in_bin) > 0:
                bin_weight = np.sum(in_bin) / len(predictions)
                bin_accuracy = np.mean(labels[in_bin])
                bin_confidence = np.mean(predictions[in_bin])

                reliability += bin_weight * (bin_confidence - bin_accuracy) ** 2
                resolution += bin_weight * (bin_accuracy - base_rate) ** 2

        return reliability, resolution, uncertainty

    def plot_reliability_diagram(self, tool_name: str, save_path: Optional[str] = None):
        """Plot reliability diagram"""

        if tool_name not in self.calibration_data:
            logger.warning(f"No data for {tool_name}")
            return

        data = self.calibration_data[tool_name]
        predictions = np.array([d.prediction for d in data])
        labels = np.array([d.actual for d in data])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        accuracies = []
        confidences = []
        counts = []

        for i in range(self.n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (
                predictions <= bin_boundaries[i + 1]
            )

            if np.sum(in_bin) > 0:
                accuracies.append(np.mean(labels[in_bin]))
                confidences.append(np.mean(predictions[in_bin]))
                counts.append(np.sum(in_bin))
            else:
                accuracies.append(bin_centers[i])
                confidences.append(bin_centers[i])
                counts.append(0)

        # Plot diagonal
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        # Plot calibration
        ax1.scatter(confidences, accuracies, s=np.array(counts) * 2, alpha=0.7)
        ax1.plot(confidences, accuracies, "b-", alpha=0.7, label="Actual")

        ax1.set_xlabel("Mean Predicted Confidence")
        ax1.set_ylabel("Actual Accuracy")
        ax1.set_title(f"Reliability Diagram - {tool_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence histogram
        ax2.hist(predictions, bins=20, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Count")
        ax2.set_title("Confidence Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def _predictions_to_logits(self, predictions: np.ndarray) -> np.ndarray:
        """Convert predictions to logits"""

        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

        # Inverse sigmoid
        logits = np.log(predictions / (1 - predictions))

        return logits

    def _prediction_to_logit(self, prediction: float) -> float:
        """Convert single prediction to logit"""

        prediction = np.clip(prediction, 1e-10, 1 - 1e-10)
        return np.log(prediction / (1 - prediction))

    def save_calibration(self, path: str):
        """Save calibration models"""

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save calibrators
        with open(save_path / "calibrators.pkl", "wb") as f:
            pickle.dump(dict(self.calibrators), f)

        # Save metrics
        metrics_dict = {
            tool: metrics.to_dict() if metrics else None
            for tool, metrics in self.metrics.items()
        }

        with open(save_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Calibration saved to {save_path}")

    def load_calibration(self, path: str):
        """Load calibration models"""

        load_path = Path(path)

        # Load calibrators
        calibrators_file = load_path / "calibrators.pkl"
        if calibrators_file.exists():
            self.calibrators = defaultdict(
                dict, safe_pickle_load(str(calibrators_file))
            )

        # Load metrics
        metrics_file = load_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)

                for tool, metrics in metrics_dict.items():
                    if metrics:
                        self.metrics[tool] = CalibrationMetrics(**metrics)

        logger.info(f"Calibration loaded from {load_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get calibration statistics"""

        stats = {}

        for tool_name in self.calibration_data.keys():
            tool_stats = {
                "n_samples": len(self.calibration_data[tool_name]),
                "calibrators": list(self.calibrators[tool_name].keys()),
            }

            if self.metrics[tool_name]:
                tool_stats["metrics"] = self.metrics[tool_name].to_dict()

            stats[tool_name] = tool_stats

        return stats
