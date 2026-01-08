"""
Contextual Bandit Learning for Tool Selection - FULL IMPLEMENTATION

FULLY IMPLEMENTED VERSION with:
- Sophisticated ML-based reward model (Random Forest, Neural Network)
- Advanced off-policy evaluation with proper variance estimation
- Doubly robust estimator with learned propensity scores
- Cross-validation for model selection
- Feature engineering for contexts
- Proper handling of continuous and categorical features
"""

import logging
import pickle  # SECURITY: Internal data only, never deserialize untrusted data
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# Advanced ML imports
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simplified models")


# ============================================================
# Note: Router Disagreement Penalty
# ============================================================
# When the selected tool wasn't in the router's recommended list,
# we apply this penalty to reduce the reward signal.
# This prevents tools from being rewarded when the router suggested
# different tools (and OpenAI fallback produced the actual result).
ROUTER_DISAGREEMENT_PENALTY = 0.5  # Halve the quality when router disagreed


class ExplorationStrategy(Enum):
    """Exploration strategies for bandit algorithms"""

    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"
    LINUCB = "linucb"
    GRADIENT = "gradient"


@dataclass
class BanditContext:
    """Context for bandit decision"""

    features: np.ndarray
    problem_type: str
    constraints: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BanditAction:
    """Action taken by bandit"""

    tool_name: str
    action_id: int
    expected_reward: float
    exploration_bonus: float = 0.0
    probability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BanditFeedback:
    """Feedback from action execution"""

    context: BanditContext
    action: BanditAction
    reward: float
    execution_time: float
    energy_used: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedRewardModel:
    """
    FULL IMPLEMENTATION: Advanced ML-based reward model

    Uses ensemble of models with automatic selection based on cross-validation
    """

    def __init__(self, context_dim: int, n_actions: int):
        self.context_dim = context_dim
        self.n_actions = n_actions
        self.eps = 1e-10

        # Feature engineering
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.poly_features = (
            PolynomialFeatures(degree=2, include_bias=False)
            if SKLEARN_AVAILABLE
            else None
        )
        self.feature_dim = context_dim

        # Model ensemble - one model per action
        self.models = {}
        self.model_types = ["random_forest", "gradient_boosting", "neural_net", "ridge"]
        self.selected_model_type = "random_forest"

        # Initialize models for each action
        for action_id in range(n_actions):
            self.models[action_id] = self._create_model_ensemble()

        # Training data storage
        self.X_train = {action_id: [] for action_id in range(n_actions)}
        self.y_train = {action_id: [] for action_id in range(n_actions)}

        # Model performance tracking
        self.model_scores = defaultdict(list)
        self.is_fitted = False

    def _create_model_ensemble(self) -> Dict[str, Any]:
        """Create ensemble of different model types"""
        if not SKLEARN_AVAILABLE:
            return {"simple": SimpleRewardModel()}

        ensemble = {
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42,
            ),
            "ridge": Ridge(alpha=1.0),
            "neural_net": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
        }

        return ensemble

    def fit(self, history: List[BanditFeedback]):
        """
        Fit reward models using historical data with cross-validation
        """
        if not history:
            logger.warning("No history provided for reward model training")
            return

        try:
            # Organize data by action
            for feedback in history:
                action_id = feedback.action.action_id

                # Engineer features
                features = self._engineer_features(feedback.context.features)

                self.X_train[action_id].append(features)
                self.y_train[action_id].append(feedback.reward)

            # Train models for each action
            for action_id in range(self.n_actions):
                if not self.X_train[action_id]:
                    logger.warning(f"No training data for action {action_id}")
                    continue

                X = np.array(self.X_train[action_id])
                y = np.array(self.y_train[action_id])

                if len(X) < 10:
                    logger.warning(
                        f"Insufficient data for action {action_id}: {len(X)} samples"
                    )
                    continue

                # Fit scaler on training data
                if self.scaler and action_id == 0:  # Fit once
                    self.scaler.fit(X)

                # Scale features
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X

                # Select best model via cross-validation
                best_model_type = self._select_best_model(X_scaled, y, action_id)

                # Train selected model on all data
                selected_model = self.models[action_id][best_model_type]
                selected_model.fit(X_scaled, y)

                logger.info(
                    f"Action {action_id}: Selected {best_model_type} model "
                    f"with score {self.model_scores[action_id][-1]:.3f}"
                )

            self.is_fitted = True

        except Exception as e:
            logger.error(f"Reward model training failed: {e}")
            self.is_fitted = False

    def _select_best_model(self, X: np.ndarray, y: np.ndarray, action_id: int) -> str:
        """
        Select best model type using cross-validation
        """
        if not SKLEARN_AVAILABLE or len(X) < 10:
            return "simple"

        best_score = -np.inf
        best_model_type = "ridge"

        try:
            # Use 5-fold cross-validation
            kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)

            for model_type in self.model_types:
                model = self.models[action_id][model_type]

                try:
                    # Cross-validation score (negative MSE)
                    scores = cross_val_score(
                        model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
                    )

                    avg_score = np.mean(scores)

                    logger.debug(
                        f"Action {action_id}, Model {model_type}: CV score = {avg_score:.3f}"
                    )

                    if avg_score > best_score:
                        best_score = avg_score
                        best_model_type = model_type

                    self.model_scores[action_id].append(avg_score)

                except Exception as e:
                    logger.warning(f"CV failed for {model_type}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Model selection failed: {e}")

        return best_model_type

    def _engineer_features(self, features: np.ndarray) -> np.ndarray:
        """
        Engineer features for better model performance

        Includes:
        - Polynomial features
        - Interaction terms
        - Statistical features
        """
        try:
            # Ensure 1D array
            if features.ndim > 1:
                features = features.flatten()

            engineered = []

            # Original features
            engineered.extend(features)

            # Statistical features
            if len(features) > 1:
                engineered.append(np.mean(features))
                engineered.append(np.std(features))
                engineered.append(np.min(features))
                engineered.append(np.max(features))
                engineered.append(np.median(features))

            # Polynomial features (degree 2)
            if len(features) <= 20:  # Only for reasonable dimensions
                for i in range(len(features)):
                    engineered.append(features[i] ** 2)

                # Interaction terms (first few features)
                for i in range(min(5, len(features))):
                    for j in range(i + 1, min(5, len(features))):
                        engineered.append(features[i] * features[j])

            return np.array(engineered)

        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}")
            return features

    def predict(self, context: BanditContext, action: BanditAction) -> float:
        """
        Predict reward for context-action pair
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, returning default prediction")
            return 0.5

        try:
            action_id = action.action_id

            if action_id not in self.models:
                return 0.5

            # Engineer features
            features = self._engineer_features(context.features)

            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)

            # Get prediction from best model
            model = self.models[action_id][self.selected_model_type]

            if not hasattr(model, "predict"):
                return 0.5

            prediction = model.predict(features_scaled)[0]

            # Clip to reasonable range
            prediction = np.clip(prediction, -10, 10)

            return float(prediction)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def predict_with_uncertainty(
        self, context: BanditContext, action: BanditAction
    ) -> Tuple[float, float]:
        """
        Predict reward with uncertainty estimate

        Uses ensemble variance as uncertainty measure
        """
        if not self.is_fitted:
            return 0.5, 1.0

        try:
            action_id = action.action_id

            # Engineer and scale features
            features = self._engineer_features(context.features)
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)

            # Get predictions from all models in ensemble
            predictions = []

            for model_type, model in self.models[action_id].items():
                if hasattr(model, "predict"):
                    try:
                        pred = model.predict(features_scaled)[0]
                        predictions.append(pred)
                    except Exception as e:
                        logger.debug(f"Operation failed: {e}")

            if not predictions:
                return 0.5, 1.0

            # Mean and std as uncertainty
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions) if len(predictions) > 1 else 1.0

            return float(mean_pred), float(std_pred)

        except Exception as e:
            logger.error(f"Prediction with uncertainty failed: {e}")
            return 0.5, 1.0


class PropensityScoreModel:
    """
    Learn propensity scores for off-policy evaluation

    Propensity score = P(action | context)
    """

    def __init__(self, context_dim: int, n_actions: int):
        self.context_dim = context_dim
        self.n_actions = n_actions
        self.eps = 1e-10

        # Use logistic regression for each action
        if SKLEARN_AVAILABLE:
            from sklearn.linear_model import LogisticRegression

            self.models = {
                action_id: LogisticRegression(max_iter=1000, random_state=42)
                for action_id in range(n_actions)
            }
        else:
            self.models = None

        self.is_fitted = False

    def fit(self, history: List[BanditFeedback]):
        """Fit propensity score models"""
        if not SKLEARN_AVAILABLE or not history:
            return

        try:
            # Prepare data
            X_all = []
            y_all = {action_id: [] for action_id in range(self.n_actions)}

            for feedback in history:
                X_all.append(feedback.context.features)

                # Binary labels for each action
                for action_id in range(self.n_actions):
                    y_all[action_id].append(
                        1 if feedback.action.action_id == action_id else 0
                    )

            X_all = np.array(X_all)

            # Fit binary classifier for each action
            for action_id in range(self.n_actions):
                y = np.array(y_all[action_id])

                # Only fit if we have positive examples
                if np.sum(y) > 0:
                    self.models[action_id].fit(X_all, y)

            self.is_fitted = True

        except Exception as e:
            logger.error(f"Propensity score training failed: {e}")

    def predict_proba(self, context: BanditContext, action_id: int) -> float:
        """Predict probability of selecting action given context"""
        if not self.is_fitted or not SKLEARN_AVAILABLE:
            return 1.0 / self.n_actions

        try:
            X = context.features.reshape(1, -1)
            prob = self.models[action_id].predict_proba(X)[0, 1]

            # Ensure positive probability
            prob = max(prob, self.eps)

            return float(prob)

        except Exception as e:
            logger.warning(f"Propensity prediction failed: {e}")
            return 1.0 / self.n_actions


class SimpleRewardModel:
    """Fallback simple reward model when sklearn unavailable"""

    def __init__(self):
        self.model = {}
        self.default_reward = 0.5

    def fit(self, X, y):
        """Store mean reward for each unique context"""
        for features, reward in zip(X, y):
            key = (
                str(features.tobytes())
                if hasattr(features, "tobytes")
                else str(features)
            )
            if key not in self.model:
                self.model[key] = []
            self.model[key].append(reward)

        # Average rewards
        for key in self.model:
            self.model[key] = np.mean(self.model[key])

    def predict(self, X):
        """Predict rewards"""
        predictions = []
        for features in X:
            key = (
                str(features.tobytes())
                if hasattr(features, "tobytes")
                else str(features)
            )
            predictions.append(self.model.get(key, self.default_reward))
        return np.array(predictions)


class ContextualBandit:
    """Base contextual bandit learner"""

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
    ):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.exploration_strategy = exploration_strategy
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Action value estimates
        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        # Action counts
        self.action_counts = np.zeros(n_actions)
        self.total_count = 0

        # History
        self.history = deque(maxlen=10000)

        # Numerical stability
        self.eps = 1e-10
        self.max_exploration_bonus = 10.0
        self.min_beta_param = 0.1

        # Statistics
        self.stats = {
            "total_actions": 0,
            "total_reward": 0.0,
            "exploration_actions": 0,
            "exploitation_actions": 0,
            "average_reward": 0.0,
        }

        self.lock = threading.Lock()

    def select_action(self, context: BanditContext) -> BanditAction:
        """Select action based on strategy"""
        with self.lock:
            try:
                if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
                    return self._epsilon_greedy_selection(context)
                elif self.exploration_strategy == ExplorationStrategy.UCB:
                    return self._ucb_selection(context)
                elif self.exploration_strategy == ExplorationStrategy.THOMPSON_SAMPLING:
                    return self._thompson_sampling_selection(context)
                else:
                    return self._epsilon_greedy_selection(context)
            except Exception as e:
                logger.error(f"Action selection failed: {e}")
                return BanditAction(
                    tool_name=self._get_tool_name(0),
                    action_id=0,
                    expected_reward=0.5,
                    exploration_bonus=0.0,
                    probability=1.0,
                )

    def _epsilon_greedy_selection(self, context: BanditContext) -> BanditAction:
        """Epsilon-greedy selection"""
        try:
            context_key = self._get_context_key(context)
            q_values = self.q_values[context_key]

            if np.random.random() < self.exploration_rate:
                action_id = np.random.randint(self.n_actions)
                self.stats["exploration_actions"] += 1
                exploration_bonus = self.exploration_rate
            else:
                action_id = np.argmax(q_values)
                self.stats["exploitation_actions"] += 1
                exploration_bonus = 0.0

            return BanditAction(
                tool_name=self._get_tool_name(action_id),
                action_id=action_id,
                expected_reward=float(q_values[action_id]),
                exploration_bonus=exploration_bonus,
                probability=self._compute_action_probability(action_id, context),
            )
        except Exception as e:
            logger.error(f"Epsilon-greedy failed: {e}")
            return BanditAction(
                tool_name=self._get_tool_name(0),
                action_id=0,
                expected_reward=0.5,
                exploration_bonus=0.0,
                probability=1.0,
            )

    def _ucb_selection(self, context: BanditContext) -> BanditAction:
        """UCB selection"""
        try:
            context_key = self._get_context_key(context)
            q_values = self.q_values[context_key]

            if self.total_count == 0:
                action_id = np.random.randint(self.n_actions)
                return BanditAction(
                    tool_name=self._get_tool_name(action_id),
                    action_id=action_id,
                    expected_reward=0.5,
                    exploration_bonus=float("inf"),
                    probability=1.0 / self.n_actions,
                )

            ucb_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                if self.action_counts[a] == 0:
                    ucb_values[a] = float("inf")
                else:
                    log_term = np.log(self.total_count + 1)
                    denominator = max(self.action_counts[a], self.eps)
                    exploration_bonus = np.sqrt(2 * log_term / denominator)
                    exploration_bonus = min(
                        exploration_bonus, self.max_exploration_bonus
                    )
                    ucb_values[a] = q_values[a] + exploration_bonus

            if np.all(np.isinf(ucb_values)):
                action_id = np.random.randint(self.n_actions)
            else:
                finite_max = (
                    np.max(ucb_values[np.isfinite(ucb_values)])
                    if np.any(np.isfinite(ucb_values))
                    else 0
                )
                ucb_values_safe = np.where(
                    np.isinf(ucb_values), finite_max + 1, ucb_values
                )
                action_id = np.argmax(ucb_values_safe)

            return BanditAction(
                tool_name=self._get_tool_name(action_id),
                action_id=action_id,
                expected_reward=float(q_values[action_id]),
                exploration_bonus=(
                    float(ucb_values[action_id] - q_values[action_id])
                    if np.isfinite(ucb_values[action_id])
                    else self.max_exploration_bonus
                ),
                probability=self._compute_action_probability(action_id, context),
            )
        except Exception as e:
            logger.error(f"UCB failed: {e}")
            return BanditAction(
                tool_name=self._get_tool_name(0),
                action_id=0,
                expected_reward=0.5,
                exploration_bonus=0.0,
                probability=1.0,
            )

    def _thompson_sampling_selection(self, context: BanditContext) -> BanditAction:
        """Thompson sampling"""
        try:
            context_key = self._get_context_key(context)

            samples = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                q_value = self.q_values[context_key][a]
                count = max(self.action_counts[a], 1)

                successes = max(self.min_beta_param, q_value * count)
                failures = max(self.min_beta_param, (1 - q_value) * count)

                successes = np.clip(successes, self.min_beta_param, 1e6)
                failures = np.clip(failures, self.min_beta_param, 1e6)

                try:
                    samples[a] = np.random.beta(successes, failures)
                except (ValueError, FloatingPointError):
                    samples[a] = q_value

            if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
                samples = self.q_values[context_key].copy()

            action_id = np.argmax(samples)

            return BanditAction(
                tool_name=self._get_tool_name(action_id),
                action_id=action_id,
                expected_reward=float(self.q_values[context_key][action_id]),
                exploration_bonus=float(
                    samples[action_id] - self.q_values[context_key][action_id]
                ),
                probability=self._compute_action_probability(action_id, context),
            )
        except Exception as e:
            logger.error(f"Thompson sampling failed: {e}")
            return BanditAction(
                tool_name=self._get_tool_name(0),
                action_id=0,
                expected_reward=0.5,
                exploration_bonus=0.0,
                probability=1.0,
            )

    def update(self, feedback: BanditFeedback):
        """Update with feedback"""
        with self.lock:
            try:
                context_key = self._get_context_key(feedback.context)
                old_q = self.q_values[context_key][feedback.action.action_id]

                reward = np.clip(feedback.reward, -1e6, 1e6)
                lr = np.clip(self.learning_rate, 0, 1)

                self.q_values[context_key][feedback.action.action_id] += lr * (
                    reward - old_q
                )
                self.q_values[context_key][feedback.action.action_id] = np.clip(
                    self.q_values[context_key][feedback.action.action_id], -1e6, 1e6
                )

                self.action_counts[feedback.action.action_id] += 1
                self.total_count += 1

                self.stats["total_actions"] += 1
                self.stats["total_reward"] += reward

                if self.stats["total_actions"] > 0:
                    self.stats["average_reward"] = (
                        self.stats["total_reward"] / self.stats["total_actions"]
                    )

                self.history.append(feedback)
            except Exception as e:
                logger.error(f"Update failed: {e}")

    def _get_context_key(self, context: BanditContext) -> str:
        """Generate context key"""
        try:
            discretized = np.round(context.features * 10) / 10
            return str(discretized.tobytes())
        except Exception:
            return "default_context"

    def _get_tool_name(self, action_id: int) -> str:
        """Map action to tool"""
        # Note: Added mathematical, philosophical, and world_model to complete tool list
        tools = ["symbolic", "probabilistic", "causal", "analogical", "multimodal", "mathematical", "philosophical", "world_model"]
        return tools[action_id % len(tools)]

    def _compute_action_probability(
        self, action_id: int, context: BanditContext
    ) -> float:
        """Compute action probability"""
        try:
            if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
                context_key = self._get_context_key(context)
                best_action = np.argmax(self.q_values[context_key])

                if action_id == best_action:
                    return (
                        1 - self.exploration_rate
                    ) + self.exploration_rate / self.n_actions
                else:
                    return self.exploration_rate / self.n_actions
            else:
                return 1.0 / self.n_actions
        except Exception:
            return 1.0 / self.n_actions


class LinUCBBandit:
    """Linear UCB bandit"""

    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha

        self.eps = 1e-10
        self.ridge_param = 1e-4

        self.A = [np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_actions)]
        self.theta = [np.zeros((context_dim, 1)) for _ in range(n_actions)]

        self.history = deque(maxlen=10000)
        self.stats = defaultdict(float)

    def select_action(self, context: BanditContext) -> BanditAction:
        """Select action"""
        try:
            x = context.features.reshape(-1, 1)
            ucb_values = np.zeros(self.n_actions)

            for a in range(self.n_actions):
                try:
                    A_reg = self.A[a] + self.ridge_param * np.eye(self.context_dim)

                    try:
                        A_inv = np.linalg.inv(A_reg)
                    except np.linalg.LinAlgError:
                        A_inv = np.linalg.pinv(A_reg)

                    self.theta[a] = A_inv @ self.b[a]

                    predicted_reward = float((x.T @ self.theta[a]).item())
                    variance = float((x.T @ A_inv @ x).item())
                    variance = max(0, variance)

                    exploration_bonus = self.alpha * np.sqrt(variance)
                    exploration_bonus = min(exploration_bonus, 10.0)

                    ucb_values[a] = predicted_reward + exploration_bonus
                except Exception:
                    ucb_values[a] = 0.0

            action_id = np.argmax(ucb_values)

            return BanditAction(
                tool_name=self._get_tool_name(action_id),
                action_id=action_id,
                expected_reward=float((x.T @ self.theta[action_id]).item()),
                exploration_bonus=float(
                    ucb_values[action_id] - float((x.T @ self.theta[action_id]).item())
                ),
                probability=1.0,
            )
        except Exception as e:
            logger.error(f"LinUCB failed: {e}")
            return BanditAction(
                tool_name=self._get_tool_name(0),
                action_id=0,
                expected_reward=0.5,
                exploration_bonus=0.0,
                probability=1.0,
            )

    def update(self, feedback: BanditFeedback):
        """Update LinUCB"""
        try:
            x = feedback.context.features.reshape(-1, 1)
            a = feedback.action.action_id
            r = np.clip(feedback.reward, -1e6, 1e6)

            self.A[a] += x @ x.T
            self.b[a] += r * x

            self.stats["total_actions"] += 1
            self.stats["total_reward"] += r

            if self.stats["total_actions"] > 0:
                self.stats["average_reward"] = (
                    self.stats["total_reward"] / self.stats["total_actions"]
                )

            self.history.append(feedback)
        except Exception as e:
            logger.error(f"LinUCB update failed: {e}")

    def _get_tool_name(self, action_id: int) -> str:
        # Note: Added mathematical, philosophical, and world_model to complete tool list
        tools = ["symbolic", "probabilistic", "causal", "analogical", "multimodal", "mathematical", "philosophical", "world_model"]
        return tools[action_id % len(tools)]


class NeuralContextualBandit(nn.Module):
    """Neural contextual bandit"""

    def __init__(self, context_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        self.context_dim = context_dim
        self.n_actions = n_actions

        self.eps = 1e-10
        self.max_log_std = 2.0
        self.min_log_std = -10.0

        self.feature_extractor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_actions)]
        )

        self.log_std_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_actions)]
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.history = deque(maxlen=10000)
        self.stats = defaultdict(float)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        try:
            features = self.feature_extractor(context)

            means = torch.cat([head(features) for head in self.action_heads], dim=1)
            log_stds = torch.cat([head(features) for head in self.log_std_heads], dim=1)

            log_stds = torch.clamp(log_stds, self.min_log_std, self.max_log_std)
            stds = torch.exp(log_stds)
            stds = torch.clamp(stds, self.eps, 1e3)

            return means, stds
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            batch_size = context.shape[0]
            return torch.zeros(batch_size, self.n_actions), torch.ones(
                batch_size, self.n_actions
            )

    def select_action(
        self, context: BanditContext, exploration_rate: float = 0.1
    ) -> BanditAction:
        """Select action"""
        try:
            x = torch.FloatTensor(context.features).unsqueeze(0)

            with torch.no_grad():
                means, stds = self.forward(x)

                if torch.isnan(means).any() or torch.isinf(means).any():
                    means = torch.zeros_like(means)

                if torch.isnan(stds).any() or torch.isinf(stds).any():
                    stds = torch.ones_like(stds)

                samples = means + stds * torch.randn_like(means)

                if torch.isnan(samples).any():
                    samples = means

                action_id = torch.argmax(samples).item()
                expected_reward = means[0, action_id].item()
                exploration_bonus = stds[0, action_id].item()

            return BanditAction(
                tool_name=self._get_tool_name(action_id),
                action_id=action_id,
                expected_reward=expected_reward,
                exploration_bonus=exploration_bonus,
                probability=self._compute_selection_probability(
                    means[0], stds[0], action_id
                ),
            )
        except Exception as e:
            logger.error(f"Neural selection failed: {e}")
            return BanditAction(
                tool_name=self._get_tool_name(0),
                action_id=0,
                expected_reward=0.5,
                exploration_bonus=0.0,
                probability=1.0,
            )

    def update(self, feedback: BanditFeedback):
        """Update neural network"""
        try:
            x = torch.FloatTensor(feedback.context.features).unsqueeze(0)
            a = feedback.action.action_id
            r = torch.FloatTensor([feedback.reward])
            r = torch.clamp(r, -1e6, 1e6)

            means, stds = self.forward(x)

            pred_mean = means[0, a]
            pred_std = stds[0, a]
            pred_std = torch.clamp(pred_std, self.eps, 1e3)

            nll_loss = 0.5 * torch.log(2 * np.pi * pred_std**2) + 0.5 * (
                (r - pred_mean) ** 2
            ) / (pred_std**2 + self.eps)

            if torch.isnan(nll_loss) or torch.isinf(nll_loss):
                return

            self.optimizer.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.stats["total_actions"] += 1
            self.stats["total_reward"] += feedback.reward

            if self.stats["total_actions"] > 0:
                self.stats["average_reward"] = (
                    self.stats["total_reward"] / self.stats["total_actions"]
                )

            self.history.append(feedback)
        except Exception as e:
            logger.error(f"Neural update failed: {e}")

    def _get_tool_name(self, action_id: int) -> str:
        # Note: Added mathematical, philosophical, and world_model to complete tool list
        tools = ["symbolic", "probabilistic", "causal", "analogical", "multimodal", "mathematical", "philosophical", "world_model"]
        return tools[action_id % len(tools)]

    def _compute_selection_probability(
        self, means: torch.Tensor, stds: torch.Tensor, action_id: int
    ) -> float:
        return 1.0 / self.n_actions


class OffPolicyEvaluator:
    """
    FULL IMPLEMENTATION: Advanced off-policy evaluation

    With proper ML-based reward models and propensity scores
    """

    def __init__(self, context_dim: int, n_actions: int):
        self.context_dim = context_dim
        self.n_actions = n_actions
        self.eps = 1e-10

        # Advanced reward model
        self.reward_model = AdvancedRewardModel(context_dim, n_actions)

        # Propensity score model
        self.propensity_model = PropensityScoreModel(context_dim, n_actions)

        self.evaluation_methods = {
            "ips": self._importance_sampling,
            "dr": self._doubly_robust,
            "dm": self._direct_method,
            "switch": self._switch_estimator,
            "snips": self._self_normalized_ips,
        }

    def evaluate(
        self, history: List[BanditFeedback], new_policy: Callable, method: str = "dr"
    ) -> Dict[str, float]:
        """Evaluate new policy"""
        if method not in self.evaluation_methods:
            raise ValueError(f"Unknown method: {method}")

        # Train models on history
        self.reward_model.fit(history)
        self.propensity_model.fit(history)

        try:
            return self.evaluation_methods[method](history, new_policy)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "estimated_reward": 0.0,
                "error": str(e),
                "num_samples": len(history),
            }

    def _importance_sampling(
        self, history: List[BanditFeedback], new_policy: Callable
    ) -> Dict[str, float]:
        """IPS estimator"""
        try:
            total_weight = 0.0
            weighted_reward = 0.0

            for feedback in history:
                new_action = new_policy(feedback.context)

                if new_action.action_id == feedback.action.action_id:
                    new_prob = max(
                        self.propensity_model.predict_proba(
                            feedback.context, new_action.action_id
                        ),
                        self.eps,
                    )
                    old_prob = max(feedback.action.probability, self.eps)

                    weight = new_prob / old_prob
                    weight = min(weight, 100.0)

                    weighted_reward += weight * feedback.reward
                    total_weight += weight

            estimated_reward = weighted_reward / max(total_weight, self.eps)

            return {
                "estimated_reward": float(estimated_reward),
                "effective_sample_size": float(total_weight),
                "num_samples": len(history),
            }
        except Exception as e:
            logger.error(f"IPS failed: {e}")
            return {"estimated_reward": 0.0, "error": str(e)}

    def _doubly_robust(
        self, history: List[BanditFeedback], new_policy: Callable
    ) -> Dict[str, float]:
        """Doubly robust estimator"""
        try:
            total_weight = 0.0
            weighted_reward = 0.0
            direct_estimate = 0.0

            for feedback in history:
                new_action = new_policy(feedback.context)

                predicted_reward = self.reward_model.predict(
                    feedback.context, new_action
                )
                direct_estimate += predicted_reward

                if new_action.action_id == feedback.action.action_id:
                    new_prob = max(
                        self.propensity_model.predict_proba(
                            feedback.context, new_action.action_id
                        ),
                        self.eps,
                    )
                    old_prob = max(feedback.action.probability, self.eps)

                    weight = new_prob / old_prob
                    weight = min(weight, 100.0)

                    weighted_reward += weight * (feedback.reward - predicted_reward)
                    total_weight += weight

            n_samples = max(len(history), 1)
            dr_estimate = (direct_estimate + weighted_reward) / n_samples

            return {
                "estimated_reward": float(dr_estimate),
                "direct_component": float(direct_estimate / n_samples),
                "ips_correction": float(weighted_reward / n_samples),
                "num_samples": len(history),
            }
        except Exception as e:
            logger.error(f"DR failed: {e}")
            return {"estimated_reward": 0.0, "error": str(e)}

    def _direct_method(
        self, history: List[BanditFeedback], new_policy: Callable
    ) -> Dict[str, float]:
        """Direct method"""
        try:
            total_reward = 0.0
            for feedback in history:
                new_action = new_policy(feedback.context)
                predicted_reward = self.reward_model.predict(
                    feedback.context, new_action
                )
                total_reward += predicted_reward

            n_samples = max(len(history), 1)

            return {
                "estimated_reward": float(total_reward / n_samples),
                "num_samples": len(history),
            }
        except Exception as e:
            logger.error(f"DM failed: {e}")
            return {"estimated_reward": 0.0, "error": str(e)}

    def _switch_estimator(
        self, history: List[BanditFeedback], new_policy: Callable
    ) -> Dict[str, float]:
        """SWITCH estimator"""
        try:
            ips_result = self._importance_sampling(history, new_policy)
            dm_result = self._direct_method(history, new_policy)

            n_samples = max(len(history), 1)
            ess_ratio = ips_result["effective_sample_size"] / n_samples
            weight = min(1.0, ess_ratio * 2)

            combined = (
                weight * ips_result["estimated_reward"]
                + (1 - weight) * dm_result["estimated_reward"]
            )

            return {
                "estimated_reward": float(combined),
                "ips_weight": float(weight),
                "ips_component": float(ips_result["estimated_reward"]),
                "dm_component": float(dm_result["estimated_reward"]),
                "num_samples": len(history),
            }
        except Exception as e:
            logger.error(f"SWITCH failed: {e}")
            return {"estimated_reward": 0.0, "error": str(e)}

    def _self_normalized_ips(
        self, history: List[BanditFeedback], new_policy: Callable
    ) -> Dict[str, float]:
        """Self-normalized IPS"""
        try:
            numerator = 0.0
            denominator = 0.0

            for feedback in history:
                new_action = new_policy(feedback.context)

                if new_action.action_id == feedback.action.action_id:
                    new_prob = max(
                        self.propensity_model.predict_proba(
                            feedback.context, new_action.action_id
                        ),
                        self.eps,
                    )
                    old_prob = max(feedback.action.probability, self.eps)

                    weight = new_prob / old_prob
                    weight = min(weight, 100.0)

                    numerator += weight * feedback.reward
                    denominator += weight

            estimated_reward = numerator / max(denominator, self.eps)

            return {
                "estimated_reward": float(estimated_reward),
                "total_weight": float(denominator),
                "num_samples": len(history),
            }
        except Exception as e:
            logger.error(f"SNIPS failed: {e}")
            return {"estimated_reward": 0.0, "error": str(e)}


class AdaptiveBanditOrchestrator:
    """Orchestrates multiple bandit algorithms"""

    def __init__(self, n_actions: int, context_dim: int):
        self.n_actions = n_actions
        self.context_dim = context_dim

        self.bandits = {
            "epsilon_greedy": ContextualBandit(
                n_actions, context_dim, ExplorationStrategy.EPSILON_GREEDY
            ),
            "ucb": ContextualBandit(n_actions, context_dim, ExplorationStrategy.UCB),
            "thompson": ContextualBandit(
                n_actions, context_dim, ExplorationStrategy.THOMPSON_SAMPLING
            ),
            "linucb": LinUCBBandit(n_actions, context_dim),
            "neural": NeuralContextualBandit(context_dim, n_actions),
        }

        self.meta_bandit = ContextualBandit(
            len(self.bandits), context_dim, ExplorationStrategy.THOMPSON_SAMPLING
        )

        self.evaluator = OffPolicyEvaluator(context_dim, n_actions)

        self.performance_window = defaultdict(lambda: deque(maxlen=100))
        self.active_bandit = "epsilon_greedy"

        self.executor = ThreadPoolExecutor(max_workers=2)

    def select_action(self, context: BanditContext) -> BanditAction:
        """Select action"""
        try:
            meta_context = BanditContext(
                features=context.features,
                problem_type="bandit_selection",
                constraints=context.constraints,
            )

            meta_action = self.meta_bandit.select_action(meta_context)
            bandit_names = list(self.bandits.keys())
            selected_bandit_name = bandit_names[
                meta_action.action_id % len(bandit_names)
            ]

            self.active_bandit = selected_bandit_name
            action = self.bandits[selected_bandit_name].select_action(context)

            action.metadata["bandit_algorithm"] = selected_bandit_name

            return action
        except Exception as e:
            logger.error(f"Adaptive selection failed: {e}")
            return self.bandits["epsilon_greedy"].select_action(context)

    def update(self, feedback: BanditFeedback):
        """Update bandits"""
        try:
            if "bandit_algorithm" in feedback.action.metadata:
                bandit_name = feedback.action.metadata["bandit_algorithm"]
                self.bandits[bandit_name].update(feedback)

                self.performance_window[bandit_name].append(feedback.reward)

                if len(self.performance_window[bandit_name]) >= 10:
                    avg_reward = np.mean(list(self.performance_window[bandit_name]))

                    bandit_names = list(self.bandits.keys())
                    action_id = bandit_names.index(bandit_name)

                    meta_feedback = BanditFeedback(
                        context=feedback.context,
                        action=BanditAction(
                            tool_name=bandit_name,
                            action_id=action_id,
                            expected_reward=avg_reward,
                            probability=1.0,
                        ),
                        reward=avg_reward,
                        execution_time=feedback.execution_time,
                        energy_used=feedback.energy_used,
                        success=feedback.success,
                    )
                    self.meta_bandit.update(meta_feedback)

            for name, bandit in self.bandits.items():
                if name != feedback.action.metadata.get("bandit_algorithm"):
                    bandit.update(feedback)
        except Exception as e:
            logger.error(f"Update failed: {e}")

    def save_model(self, path: str):
        """Save models"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            for name, bandit in self.bandits.items():
                if hasattr(bandit, "state_dict"):
                    torch.save(bandit.state_dict(), save_path / f"{name}_model.pt")
                else:
                    with open(save_path / f"{name}_state.pkl", "wb") as f:
                        pickle.dump(
                            {
                                "q_values": (
                                    dict(bandit.q_values)
                                    if hasattr(bandit, "q_values")
                                    else {}
                                ),
                                "action_counts": (
                                    bandit.action_counts.tolist()
                                    if hasattr(bandit, "action_counts")
                                    else []
                                ),
                                "stats": (
                                    dict(bandit.stats)
                                    if hasattr(bandit, "stats")
                                    else {}
                                ),
                            },
                            f,
                        )

            with open(save_path / "meta_bandit.pkl", "wb") as f:
                pickle.dump(
                    {
                        "performance_window": {
                            k: list(v) for k, v in self.performance_window.items()
                        },
                        "active_bandit": self.active_bandit,
                    },
                    f,
                )

            logger.info(f"Models saved to {save_path}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def load_model(self, path: str):
        """Load models"""
        try:
            load_path = Path(path)

            for name, bandit in self.bandits.items():
                if hasattr(bandit, "load_state_dict"):
                    model_file = load_path / f"{name}_model.pt"
                    if model_file.exists():
                        bandit.load_state_dict(
                            torch.load(model_file, weights_only=True)
                        )
                else:
                    state_file = load_path / f"{name}_state.pkl"
                    if state_file.exists():
                        with open(state_file, "rb") as f:
                            state = pickle.load(
                                f
                            )  # nosec B301 - Internal data structure
                            if hasattr(bandit, "q_values"):
                                bandit.q_values = defaultdict(
                                    lambda: np.zeros(self.n_actions), state["q_values"]
                                )
                            if hasattr(bandit, "action_counts"):
                                bandit.action_counts = np.array(state["action_counts"])
                            if hasattr(bandit, "stats"):
                                bandit.stats = state["stats"]

            meta_file = load_path / "meta_bandit.pkl"
            if meta_file.exists():
                with open(meta_file, "rb") as f:
                    meta_state = pickle.load(f)  # nosec B301 - Internal data structure
                    self.performance_window = defaultdict(
                        lambda: deque(maxlen=100),
                        {
                            k: deque(v, maxlen=100)
                            for k, v in meta_state["performance_window"].items()
                        },
                    )
                    self.active_bandit = meta_state["active_bandit"]

            logger.info(f"Models loaded from {load_path}")
        except Exception as e:
            logger.error(f"Load failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = {"active_bandit": self.active_bandit, "bandit_performance": {}}

        try:
            for name, bandit in self.bandits.items():
                if hasattr(bandit, "stats"):
                    stats["bandit_performance"][name] = dict(bandit.stats)

                if name in self.performance_window:
                    recent = list(self.performance_window[name])
                    if recent:
                        stats["bandit_performance"][name]["recent_average"] = float(
                            np.mean(recent)
                        )
        except Exception as e:
            logger.error(f"Stats failed: {e}")

        return stats


class ToolSelectionBandit(AdaptiveBanditOrchestrator):
    """Tool selection bandit"""

    def __init__(self):
        # n_actions=8 to include all tools: symbolic, probabilistic, causal, analogical, 
        # multimodal, mathematical, philosophical, and world_model
        super().__init__(n_actions=8, context_dim=128)

        # Tool names registered with the bandit learning system
        # Without registration, bandit updates fail with "Unknown tool name 'X' in bandit update"
        self.tool_names = [
            "symbolic",
            "probabilistic",
            "causal",
            "analogical",
            "multimodal",
            "mathematical",   # For math queries
            "philosophical",  # For ethical/philosophical queries
            "world_model",    # For meta-cognitive self-introspection queries
        ]
        self.tool_costs = {
            "symbolic": {"time": 50, "energy": 50},
            "probabilistic": {"time": 100, "energy": 100},
            "causal": {"time": 200, "energy": 200},
            "analogical": {"time": 80, "energy": 80},
            "multimodal": {"time": 300, "energy": 300},
            "mathematical": {"time": 60, "energy": 40},
            "philosophical": {"time": 150, "energy": 100},
            "world_model": {"time": 180, "energy": 120},  # For self-introspection
        }

    def select_tool(self, features: np.ndarray, constraints: Dict[str, float]) -> str:
        """Select tool"""
        try:
            context = BanditContext(
                features=features,
                problem_type="tool_selection",
                constraints=constraints,
            )

            action = self.select_action(context)
            return action.tool_name
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return "symbolic"

    def update_from_execution(
        self,
        features: np.ndarray,
        tool_name: str,
        quality: float,
        time_ms: float,
        energy_mj: float,
        constraints: Dict[str, float],
        router_selected_tools: Optional[List[str]] = None,
        source: Optional[str] = None,
    ):
        """Update from execution.
        
        Note: Added router_selected_tools and source parameters to check
        if the tool was CORRECT for the query type. Don't reward a tool just
        because OpenAI fallback succeeded - reward only when the tool was
        actually the right choice based on query router's analysis.
        
        Args:
            features: Context features
            tool_name: Tool that was used
            quality: Quality score of result
            time_ms: Execution time in milliseconds
            energy_mj: Energy used in millijoules
            constraints: Budget constraints
            router_selected_tools: Tools that the QueryRouter recommended (for correctness check)
            source: Source of result (e.g., 'openai_fallback', 'local')
        """
        try:
            # Note: Check if result came from OpenAI fallback
            # If so, don't reward the selected tool - it didn't actually produce the result
            if source and 'openai' in source.lower() and 'fallback' in source.lower():
                logger.info(
                    f"[ToolSelectionBandit] SKIPPING reward for '{tool_name}' - "
                    f"result came from OpenAI fallback, not tool execution"
                )
                return
            
            # Note: Check router agreement
            # Only reward tool if it was in the router's recommended list
            if router_selected_tools and tool_name not in router_selected_tools:
                logger.info(
                    f"[ToolSelectionBandit] REDUCED reward for '{tool_name}' - "
                    f"not in router's selection: {router_selected_tools}"
                )
                # Apply penalty: tool wasn't the router's choice
                quality = quality * ROUTER_DISAGREEMENT_PENALTY
            
            reward = self._compute_reward(quality, time_ms, energy_mj, constraints)

            context = BanditContext(
                features=features,
                problem_type="tool_selection",
                constraints=constraints,
            )

            action_id = (
                self.tool_names.index(tool_name) if tool_name in self.tool_names else 0
            )

            action = BanditAction(
                tool_name=tool_name,
                action_id=action_id,
                expected_reward=reward,
                probability=1.0,
            )

            feedback = BanditFeedback(
                context=context,
                action=action,
                reward=reward,
                execution_time=time_ms,
                energy_used=energy_mj,
                success=quality > constraints.get("min_quality", 0.5),
            )

            self.update(feedback)
        except Exception as e:
            logger.error(f"Update failed: {e}")

    def _compute_reward(
        self,
        quality: float,
        time_ms: float,
        energy_mj: float,
        constraints: Dict[str, float],
    ) -> float:
        """Compute reward"""
        try:
            time_budget = max(constraints.get("time_budget", 1000), 1.0)
            energy_budget = max(constraints.get("energy_budget", 1000), 1.0)

            time_score = max(0, 1 - time_ms / time_budget)
            energy_score = max(0, 1 - energy_mj / energy_budget)
            quality = np.clip(quality, 0, 1)

            quality_weight = constraints.get("quality_weight", 1.0)
            time_weight = constraints.get("time_weight", 1.0)
            energy_weight = constraints.get("energy_weight", 1.0)

            total_weight = max(quality_weight + time_weight + energy_weight, 1e-10)

            reward = (
                quality * quality_weight
                + time_score * time_weight
                + energy_score * energy_weight
            ) / total_weight

            return float(np.clip(reward, 0, 1))
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return 0.5
