# cost_model.py
"""
Stochastic Cost Model for Tool Selection System

Models execution costs (time, energy, memory) with uncertainty quantification,
health-aware adjustments, and predictive capabilities.
"""

import json
import logging
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)


class CostComponent(Enum):
    """Cost components tracked by the model"""

    TIME_MS = "time_ms"
    ENERGY_MJ = "energy_mj"
    MEMORY_MB = "memory_mb"
    CPU_CYCLES = "cpu_cycles"
    NETWORK_KB = "network_kb"


class ComplexityLevel(Enum):
    """Problem complexity levels"""

    TRIVIAL = 0.1
    SIMPLE = 0.3
    MODERATE = 0.5
    COMPLEX = 0.8
    EXTREME = 1.0


@dataclass
class CostObservation:
    """Single cost observation"""

    tool_name: str
    component: CostComponent
    value: float
    features: np.ndarray
    complexity: float
    timestamp: float = field(default_factory=time.time)
    cold_start: bool = False
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostDistribution:
    """Cost distribution parameters"""

    mean: float
    variance: float
    std: float
    percentile_5: float
    percentile_25: float
    median: float
    percentile_75: float
    percentile_95: float
    confidence_interval: Tuple[float, float]
    samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "p5": self.percentile_5,
            "p25": self.percentile_25,
            "median": self.median,
            "p75": self.percentile_75,
            "p95": self.percentile_95,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "samples": self.samples,
        }


@dataclass
class HealthMetrics:
    """Tool health metrics"""

    error_rate: float = 0.0
    queue_depth: int = 0
    last_success: float = field(default_factory=time.time)
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    avg_response_time: float = 0.0
    warm: bool = False
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    @property
    def health_score(self) -> float:
        """Compute overall health score [0, 1]"""
        score = 1.0

        # Penalize errors
        score -= self.error_rate * 0.3

        # Penalize queue depth
        score -= min(0.2, self.queue_depth / 100)

        # Penalize consecutive failures
        score -= min(0.3, self.consecutive_failures * 0.1)

        # Penalize staleness
        time_since_success = time.time() - self.last_success
        if time_since_success > 300:  # 5 minutes
            score -= min(0.1, time_since_success / 3600)

        # Resource usage penalties
        if self.cpu_usage > 80:
            score -= 0.1
        if self.memory_usage > 80:
            score -= 0.1

        return max(0.0, min(1.0, score))


class CostPredictor:
    """Predicts costs using regression models"""

    def __init__(self):
        self.models = {}  # (tool, component) -> model
        self.feature_importance = defaultdict(dict)

    def fit(
        self,
        observations: List[CostObservation],
        tool_name: str,
        component: CostComponent,
    ):
        """Fit prediction model for tool/component"""

        if len(observations) < 10:
            return

        # Extract features and targets
        X = np.array([obs.features for obs in observations])
        y = np.array([obs.value for obs in observations])

        # Add complexity as feature
        complexity = np.array([obs.complexity for obs in observations]).reshape(-1, 1)
        X = np.hstack([X, complexity])

        # Simple linear regression for now
        # Could use more sophisticated models
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        self.models[(tool_name, component)] = model

        # Store feature importance (coefficients)
        self.feature_importance[tool_name][component] = model.coef_

    def predict(
        self,
        features: np.ndarray,
        complexity: float,
        tool_name: str,
        component: CostComponent,
    ) -> Optional[float]:
        """Predict cost for given features"""

        if (tool_name, component) not in self.models:
            return None

        model = self.models[(tool_name, component)]

        # Prepare features
        X = np.hstack([features.reshape(1, -1), [[complexity]]])

        prediction = model.predict(X)[0]

        return max(0, prediction)  # Ensure non-negative


class StochasticCostModel:
    """
    Main stochastic cost model with variance tracking and health awareness
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Cost distributions per tool and component
        self.distributions = defaultdict(lambda: defaultdict(dict))

        # Observations for distribution fitting
        self.observations = defaultdict(lambda: defaultdict(list))
        self.max_observations = config.get("max_observations", 1000)

        # Health tracking
        self.health_metrics = defaultdict(HealthMetrics)

        # Cold start penalties
        self.cold_start_penalties = {
            CostComponent.TIME_MS: config.get("cold_start_time_ms", 100),
            CostComponent.ENERGY_MJ: config.get("cold_start_energy_mj", 50),
            CostComponent.MEMORY_MB: config.get("cold_start_memory_mb", 100),
        }

        # Complexity estimation
        self.complexity_estimator = ComplexityEstimator()

        # Cost predictor
        self.predictor = CostPredictor()

        # CPU capabilities for cost adjustment
        try:
            from src.utils.cpu_capabilities import get_cpu_capabilities
            
            self._cpu_caps = get_cpu_capabilities()
            self._cpu_cost_multiplier = self._calculate_cpu_multiplier()
            logger.info(
                f"[CostModel] CPU tier: {self._cpu_caps.get_performance_tier()}, "
                f"multiplier: {self._cpu_cost_multiplier:.2f}"
            )
        except Exception as e:
            logger.warning(f"[CostModel] CPU detection failed: {e}, using default multiplier")
            self._cpu_caps = None
            self._cpu_cost_multiplier = 1.0

        # Define compute-intensive tools that benefit from better CPU capabilities
        self._compute_intensive_tools = {
            "symbolic", "causal", "multimodal", "visualizer", 
            "generator", "evolver", "analogical"
        }

        # Default distributions
        self._initialize_defaults()

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.prediction_errors = deque(maxlen=100)
        self.total_predictions = 0
        self.total_updates = 0

    def _calculate_cpu_multiplier(self) -> float:
        """Calculate cost multiplier based on CPU capabilities.
        
        Higher performance CPUs (AVX-512, SVE2) execute compute-intensive operations
        faster, reducing time costs. This multiplier adjusts predictions to reflect
        hardware capabilities.
        
        Returns:
            float: Cost multiplier where <1.0 means faster (better CPU), 
                   >1.0 means slower (basic CPU)
        """
        if self._cpu_caps is None:
            return 1.0
            
        tier = self._cpu_caps.get_performance_tier()
        
        # Higher performance = lower cost multiplier (faster execution)
        # These multipliers are empirically derived from benchmarks showing
        # that AVX-512 can be 2-4x faster than scalar for vectorizable workloads
        multipliers = {
            "High Performance": 0.7,      # AVX-512/SVE2 - significant speedup
            "Medium Performance": 0.85,   # AVX2/SVE/NEON - moderate speedup
            "Standard Performance": 1.0,  # AVX/SSE4 - baseline
            "Basic Performance": 1.5,     # Scalar only - slower
        }
        
        return multipliers.get(tier, 1.0)

    def _initialize_defaults(self):
        """Initialize default cost distributions
        
        Note: Added Arena agent defaults with realistic cold-start priors.
        Production logs show visualizer taking 30-60s under CPU load,
        so defaults should reflect actual performance, not optimistic estimates.
        """

        default_costs = {
            "symbolic": {
                CostComponent.TIME_MS: (500, 100),  # (mean, std)
                CostComponent.ENERGY_MJ: (50, 10),
                CostComponent.MEMORY_MB: (200, 50),
            },
            "probabilistic": {
                CostComponent.TIME_MS: (300, 50),
                CostComponent.ENERGY_MJ: (100, 20),
                CostComponent.MEMORY_MB: (150, 30),
            },
            "causal": {
                CostComponent.TIME_MS: (1000, 200),
                CostComponent.ENERGY_MJ: (200, 40),
                CostComponent.MEMORY_MB: (300, 60),
            },
            "analogical": {
                CostComponent.TIME_MS: (200, 40),
                CostComponent.ENERGY_MJ: (80, 15),
                CostComponent.MEMORY_MB: (100, 20),
            },
            "multimodal": {
                CostComponent.TIME_MS: (1500, 300),
                CostComponent.ENERGY_MJ: (300, 60),
                CostComponent.MEMORY_MB: (500, 100),
            },
            # Note: Add Arena agent defaults with realistic values
            # Production logs show these agents take 30-60s under CPU load
            "visualizer": {
                CostComponent.TIME_MS: (45000, 20000),  # 45s mean, high variance
                CostComponent.ENERGY_MJ: (500, 200),
                CostComponent.MEMORY_MB: (800, 300),
            },
            "generator": {
                CostComponent.TIME_MS: (30000, 15000),  # 30s mean
                CostComponent.ENERGY_MJ: (400, 150),
                CostComponent.MEMORY_MB: (600, 200),
            },
            "evolver": {
                CostComponent.TIME_MS: (40000, 18000),  # 40s mean
                CostComponent.ENERGY_MJ: (450, 180),
                CostComponent.MEMORY_MB: (700, 250),
            },
            "photonic_optimizer": {
                CostComponent.TIME_MS: (60000, 25000),  # 60s mean - hardware is slow
                CostComponent.ENERGY_MJ: (600, 250),
                CostComponent.MEMORY_MB: (500, 150),
            },
            "automl_optimizer": {
                CostComponent.TIME_MS: (50000, 22000),  # 50s mean
                CostComponent.ENERGY_MJ: (550, 220),
                CostComponent.MEMORY_MB: (900, 350),
            },
        }

        for tool_name, components in default_costs.items():
            for component, (mean, std) in components.items():
                self.distributions[tool_name][component] = self._create_distribution(
                    mean,
                    std * std,
                    [mean],  # variance = std^2
                )

    def predict_cost(
        self, tool_name: str, features: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Predict cost distribution with confidence intervals
        
        Applies CPU-aware adjustments for compute-intensive operations based on
        detected instruction set capabilities (AVX-512, AVX2, etc.)

        Returns:
            Dictionary with cost predictions for each component
        """

        with self.lock:
            self.total_predictions += 1

            # Estimate complexity
            complexity = self.complexity_estimator.estimate(features)

            # Get health metrics
            health = self.health_metrics[tool_name]

            predictions = {}

            for component in CostComponent:
                # Get base distribution
                if component in self.distributions[tool_name]:
                    dist = self.distributions[tool_name][component]
                else:
                    # Use default
                    dist = self._create_distribution(100, 100, [100])

                # Adjust for complexity
                adjusted_mean = dist.mean * (1 + complexity)
                adjusted_var = dist.variance * (1 + complexity * 1.5)

                # Adjust for health
                health_multiplier = 1 + (1 - health.health_score) * 0.5
                adjusted_mean *= health_multiplier
                adjusted_var *= health_multiplier**2

                # Add cold start penalty if not warm
                if not health.warm:
                    if component in self.cold_start_penalties:
                        adjusted_mean += self.cold_start_penalties[component]
                        adjusted_var += (
                            self.cold_start_penalties[component] * 0.2
                        ) ** 2

                # Apply CPU multiplier for compute-intensive operations
                # Only affects time and energy, not memory
                if (tool_name in self._compute_intensive_tools and 
                    component in [CostComponent.TIME_MS, CostComponent.ENERGY_MJ]):
                    adjusted_mean *= self._cpu_cost_multiplier
                    adjusted_var *= (self._cpu_cost_multiplier ** 2)

                # Try predictive model
                predicted = self.predictor.predict(
                    features, complexity, tool_name, component
                )
                if predicted is not None:
                    # Blend with statistical estimate
                    adjusted_mean = 0.7 * adjusted_mean + 0.3 * predicted

                # Compute confidence interval
                ci = self._confidence_interval(
                    adjusted_mean, np.sqrt(adjusted_var), confidence_level
                )

                predictions[component.value] = {
                    "mean": adjusted_mean,
                    "var": adjusted_var,
                    "std": np.sqrt(adjusted_var),
                    "ci": ci,
                    "confidence_level": confidence_level,
                }

            # Add aggregate prediction
            predictions["failure_risk"] = health.error_rate * (1 + complexity)
            
            # Add CPU adjustment metadata for monitoring
            if tool_name in self._compute_intensive_tools:
                predictions["cpu_adjusted"] = True
                predictions["cpu_multiplier"] = self._cpu_cost_multiplier
                if self._cpu_caps:
                    predictions["cpu_tier"] = self._cpu_caps.get_performance_tier()

            return predictions

    def update(
        self,
        tool_name: str,
        component: CostComponent,
        value: float,
        features: np.ndarray,
        cold_start: bool = False,
    ):
        """Update cost model with new observation"""

        with self.lock:
            self.total_updates += 1

            # Estimate complexity
            complexity = self.complexity_estimator.estimate(features)

            # Create observation
            obs = CostObservation(
                tool_name=tool_name,
                component=component,
                value=value,
                features=features,
                complexity=complexity,
                cold_start=cold_start,
                health_score=self.health_metrics[tool_name].health_score,
            )

            # Store observation
            self.observations[tool_name][component].append(obs)

            # Limit observations
            if len(self.observations[tool_name][component]) > self.max_observations:
                self.observations[tool_name][component].pop(0)

            # Update distribution
            self._update_distribution(tool_name, component)

            # Update predictor periodically
            if len(self.observations[tool_name][component]) % 50 == 0:
                self.predictor.fit(
                    self.observations[tool_name][component], tool_name, component
                )

            # Track prediction error if we had predicted this
            self._track_prediction_error(tool_name, component, value, features)

    def update_health(self, tool_name: str, health_update: Dict[str, Any]):
        """Update health metrics for a tool"""

        with self.lock:
            health = self.health_metrics[tool_name]

            for key, value in health_update.items():
                if hasattr(health, key):
                    setattr(health, key, value)

    def _update_distribution(self, tool_name: str, component: CostComponent):
        """Update cost distribution from observations"""

        observations = self.observations[tool_name][component]

        if len(observations) < 2:
            return

        # Extract values, adjusting for cold starts and health
        values = []
        for obs in observations:
            value = obs.value

            # Normalize for cold start
            if obs.cold_start and component in self.cold_start_penalties:
                value -= self.cold_start_penalties[component]

            # Normalize for complexity
            value /= 1 + obs.complexity

            # Normalize for health
            if obs.health_score > 0:
                value /= 1 + (1 - obs.health_score) * 0.5

            values.append(max(1, value))  # Ensure positive

        # Create new distribution
        self.distributions[tool_name][component] = self._create_distribution(
            np.mean(values), np.var(values), values
        )

    def _create_distribution(
        self, mean: float, variance: float, samples: List[float]
    ) -> CostDistribution:
        """Create cost distribution from statistics"""

        if len(samples) < 2:
            # Not enough data, use simple estimates
            return CostDistribution(
                mean=mean,
                variance=variance,
                std=np.sqrt(variance),
                percentile_5=mean * 0.5,
                percentile_25=mean * 0.75,
                median=mean,
                percentile_75=mean * 1.25,
                percentile_95=mean * 1.5,
                confidence_interval=(mean * 0.8, mean * 1.2),
                samples=len(samples),
            )

        # Compute full statistics
        samples_array = np.array(samples)

        return CostDistribution(
            mean=np.mean(samples_array),
            variance=np.var(samples_array),
            std=np.std(samples_array),
            percentile_5=np.percentile(samples_array, 5),
            percentile_25=np.percentile(samples_array, 25),
            median=np.median(samples_array),
            percentile_75=np.percentile(samples_array, 75),
            percentile_95=np.percentile(samples_array, 95),
            confidence_interval=self._confidence_interval(
                np.mean(samples_array), np.std(samples_array), 0.95, len(samples_array)
            ),
            samples=len(samples_array),
        )

    def _confidence_interval(
        self, mean: float, std: float, confidence: float, n: Optional[int] = None
    ) -> Tuple[float, float]:
        """Compute confidence interval"""

        # FIXED: Added validation for degrees of freedom
        if n is None or n < 2:
            # Not enough samples for t-distribution, use conservative estimate
            se = std if n is None else std / np.sqrt(max(n, 1))
            t_stat = 2.576  # Conservative z-score for 99% CI
        elif n < 30:
            # Use t-distribution for small samples (n >= 2)
            se = std / np.sqrt(n)
            df = n - 1  # Degrees of freedom, guaranteed to be >= 1
            t_stat = stats.t.ppf((1 + confidence) / 2, df)
        else:
            # Use normal distribution for large samples
            se = std / np.sqrt(n)
            t_stat = stats.norm.ppf((1 + confidence) / 2)

        margin = t_stat * se

        return (max(0, mean - margin), mean + margin)

    def _track_prediction_error(
        self,
        tool_name: str,
        component: CostComponent,
        actual: float,
        features: np.ndarray,
    ):
        """Track prediction error for monitoring"""

        # Get prediction
        prediction = self.predict_cost(tool_name, features)

        if component.value in prediction:
            predicted = prediction[component.value]["mean"]
            error = abs(actual - predicted) / max(actual, 1)
            self.prediction_errors.append(
                {
                    "tool": tool_name,
                    "component": component.value,
                    "error": error,
                    "timestamp": time.time(),
                }
            )

    def get_tail_risks(
        self, tool_name: str, percentile: float = 95
    ) -> Dict[str, float]:
        """Get tail risk estimates (worst-case scenarios)"""

        tail_risks = {}

        for component in CostComponent:
            if component in self.distributions[tool_name]:
                dist = self.distributions[tool_name][component]

                if percentile == 95:
                    tail_value = dist.percentile_95
                else:
                    # Estimate from distribution
                    samples = self.observations[tool_name][component]
                    if samples:
                        values = [obs.value for obs in samples]
                        tail_value = np.percentile(values, percentile)
                    else:
                        tail_value = dist.mean * (1 + percentile / 100)

                tail_risks[component.value] = tail_value

        return tail_risks

    def estimate_total_cost(
        self,
        tool_name: str,
        features: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimate total weighted cost

        Args:
            tool_name: Tool name
            features: Feature vector
            weights: Component weights for aggregation

        Returns:
            Total weighted cost estimate
        """

        if weights is None:
            weights = {
                CostComponent.TIME_MS.value: 1.0,
                CostComponent.ENERGY_MJ.value: 0.1,
                CostComponent.MEMORY_MB.value: 0.01,
            }

        predictions = self.predict_cost(tool_name, features)

        total = 0
        for component, weight in weights.items():
            if component in predictions:
                total += predictions[component]["mean"] * weight

        return total

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""

        with self.lock:
            stats = {
                "total_predictions": self.total_predictions,
                "total_updates": self.total_updates,
                "tools_tracked": list(self.distributions.keys()),
                "observations_count": {
                    tool: {
                        comp.value: len(obs_list)
                        for comp, obs_list in components.items()
                    }
                    for tool, components in self.observations.items()
                },
                "health_scores": {
                    tool: health.health_score
                    for tool, health in self.health_metrics.items()
                },
            }

            # Add prediction error statistics
            if self.prediction_errors:
                errors = [e["error"] for e in self.prediction_errors]
                stats["prediction_performance"] = {
                    "mean_error": np.mean(errors),
                    "std_error": np.std(errors),
                    "max_error": np.max(errors),
                }

            return stats

    def save_model(self, path: str):
        """Save cost model to disk"""

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        with self.lock:
            # Save distributions
            dist_data = {}
            for tool, components in self.distributions.items():
                dist_data[tool] = {
                    comp.value: dist.to_dict() for comp, dist in components.items()
                }

            with open(save_path / "distributions.json", "w", encoding="utf-8") as f:
                json.dump(dist_data, f, indent=2)

            # Save observations
            with open(save_path / "observations.pkl", "wb") as f:
                pickle.dump(dict(self.observations), f)

            # Save predictor models
            with open(save_path / "predictor.pkl", "wb") as f:
                pickle.dump(self.predictor, f)

            # Save health metrics (excluding computed properties)
            health_data = {
                tool: {
                    "error_rate": health.error_rate,
                    "queue_depth": health.queue_depth,
                    "last_success": health.last_success,
                    "last_failure": health.last_failure,
                    "consecutive_failures": health.consecutive_failures,
                    "avg_response_time": health.avg_response_time,
                    "warm": health.warm,
                    "cpu_usage": health.cpu_usage,
                    "memory_usage": health.memory_usage,
                }
                for tool, health in self.health_metrics.items()
            }

            with open(save_path / "health.json", "w", encoding="utf-8") as f:
                json.dump(health_data, f, indent=2)

        logger.info(f"Cost model saved to {save_path}")

    def load_model(self, path: str):
        """Load cost model from disk"""

        load_path = Path(path)

        if not load_path.exists():
            logger.warning(f"Model path {load_path} not found")
            return

        with self.lock:
            # Load distributions
            dist_file = load_path / "distributions.json"
            if dist_file.exists():
                with open(dist_file, "r", encoding="utf-8") as f:
                    dist_data = json.load(f)

                for tool, components in dist_data.items():
                    for comp_str, dist_dict in components.items():
                        # Reconstruct distribution
                        self.distributions[tool][CostComponent(comp_str)] = (
                            CostDistribution(
                                mean=dist_dict["mean"],
                                variance=dist_dict["variance"],
                                std=dist_dict["std"],
                                percentile_5=dist_dict["p5"],
                                percentile_25=dist_dict["p25"],
                                median=dist_dict["median"],
                                percentile_75=dist_dict["p75"],
                                percentile_95=dist_dict["p95"],
                                confidence_interval=(
                                    dist_dict["ci_lower"],
                                    dist_dict["ci_upper"],
                                ),
                                samples=dist_dict["samples"],
                            )
                        )

            # Load observations
            obs_file = load_path / "observations.pkl"
            if obs_file.exists():
                with open(obs_file, "rb") as f:
                    self.observations = defaultdict(
                        lambda: defaultdict(list), safe_pickle_load(f)
                    )

            # Load predictor
            pred_file = load_path / "predictor.pkl"
            if pred_file.exists():
                with open(pred_file, "rb") as f:
                    self.predictor = safe_pickle_load(f)

            # Load health metrics (skip computed properties like health_score)
            health_file = load_path / "health.json"
            if health_file.exists():
                with open(health_file, "r", encoding="utf-8") as f:
                    health_data = json.load(f)

                for tool, metrics in health_data.items():
                    health = self.health_metrics[tool]
                    for key, value in metrics.items():
                        # Skip computed properties that don't have setters
                        if key == "health_score":
                            continue
                        if hasattr(health, key):
                            setattr(health, key, value)

        logger.info(f"Cost model loaded from {load_path}")


class ComplexityEstimator:
    """Estimates problem complexity from features"""

    def __init__(self):
        self.feature_weights = None
        self.complexity_history = deque(maxlen=1000)

    def estimate(self, features: np.ndarray) -> float:
        """
        Estimate complexity from features

        Returns:
            Complexity score [0, 1]
        """

        # Simple heuristics for now
        # Could be learned from data

        # Feature magnitude
        magnitude = np.linalg.norm(features)

        # Feature sparsity
        # Note: Check for zero length before division to prevent RuntimeWarning
        if len(features) > 0:
            sparsity = np.sum(features != 0) / len(features)
        else:
            sparsity = 0.0

        # Feature entropy (if positive)
        if np.all(features >= 0) and np.sum(features) > 0:
            probs = features / np.sum(features)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropy_norm = entropy / np.log(len(features))
        else:
            entropy_norm = 0.5

        # Combine factors
        complexity = (
            0.3 * min(1.0, magnitude / 10) + 0.3 * sparsity + 0.4 * entropy_norm
        )

        # Track for learning
        self.complexity_history.append(
            {"features": features, "complexity": complexity, "timestamp": time.time()}
        )

        return min(1.0, max(0.0, complexity))
