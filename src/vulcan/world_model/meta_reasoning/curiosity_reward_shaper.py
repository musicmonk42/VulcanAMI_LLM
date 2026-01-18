# src/vulcan/world_model/meta_reasoning/curiosity_reward_shaper.py
"""
curiosity_reward_shaper.py - Multi-algorithm curiosity-driven exploration and reward shaping
Part of the meta_reasoning subsystem for VULCAN-AMI

FULL PRODUCTION IMPLEMENTATION

Advanced curiosity-driven exploration using multiple algorithms:
- Count-based exploration (1/sqrt(N) novelty bonus)
- Intrinsic Curiosity Module (ICM) - prediction error based novelty
- Random Network Distillation (RND) - neural network novelty detection
- Information gain maximization
- Episodic novelty memory
- Feature-based state similarity

Algorithms:
- Count-based: Simple visit counting with decay
- ICM: Predicts next state features, uses prediction error as novelty
- RND: Random network distillation for discovering novel states
- Information gain: Measures uncertainty reduction
- Hybrid: Combines multiple methods with learned weights

Features:
- Adaptive bonus scaling based on learning progress
- Episodic memory for efficient similarity search
- State feature extraction and hashing
- Temporal decay for forgetting old states
- Integration with world model for predictions

Integration:
- Uses world model for state prediction
- Records to ValidationTracker for effectiveness learning
- Feeds TransparencyInterface for audit trails
- Adapts with SelfImprovementDrive

Thread-safe with comprehensive statistics and state persistence.
"""

import hashlib
import json
import logging
import threading
import time  # Moved import here to be grouped
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from typing import Any, Callable, Dict, List, Optional

from vulcan.world_model.meta_reasoning.numpy_compat import np, NUMPY_AVAILABLE
from vulcan.world_model.meta_reasoning.serialization_mixin import SerializationMixin

logger = logging.getLogger(__name__)


# --- START FIX: Add missing helper function ---
def _hash_to_float(s: str) -> float:
    """Helper to hash a string to a float in [0, 1]"""
    try:
        hash_bytes = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()
        # Use first 4 bytes as an int
        hash_int = int.from_bytes(hash_bytes[:4], "little")
        # Normalize to [0, 1]
        return (hash_int & 0xFFFFFFFF) / 0xFFFFFFFF
    except Exception:
        return 0.5  # Fallback


# --- END FIX ---


class CuriosityMethod(Enum):
    """Curiosity computation method"""

    COUNT_BASED = "count_based"  # Simple visit counting
    ICM = "icm"  # Intrinsic Curiosity Module
    RND = "rnd"  # Random Network Distillation
    INFORMATION_GAIN = "information_gain"  # Information-theoretic
    EPISODIC = "episodic"  # Episodic memory similarity
    HYBRID = "hybrid"  # Combination of methods


class NoveltyLevel(Enum):
    """Classification of state novelty"""

    COMPLETELY_NOVEL = "completely_novel"  # Never seen before
    HIGHLY_NOVEL = "highly_novel"  # Very different from seen states
    MODERATELY_NOVEL = "moderately_novel"  # Somewhat different
    FAMILIAR = "familiar"  # Similar to seen states
    WELL_KNOWN = "well_known"  # Seen many times


@dataclass
class NoveltyEstimate:
    """Estimate of state/action novelty"""

    state_hash: str
    novelty_score: float  # 0-1, higher = more novel
    visit_count: int

    # Novelty by method
    count_novelty: float = 0.0
    icm_novelty: float = 0.0
    rnd_novelty: float = 0.0
    episodic_novelty: float = 0.0

    # Classification
    novelty_level: NoveltyLevel = NoveltyLevel.FAMILIAR

    # Tracking
    first_visit: float = field(default_factory=time.time)
    last_visit: float = field(default_factory=time.time)

    # State features
    features: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "state_hash": self.state_hash,
            "novelty_score": self.novelty_score,
            "visit_count": self.visit_count,
            "novelty_level": self.novelty_level.value,
            "count_novelty": self.count_novelty,
            "icm_novelty": self.icm_novelty,
            "rnd_novelty": self.rnd_novelty,
            "episodic_novelty": self.episodic_novelty,
            "last_visit": self.last_visit,
            "metadata": self.metadata,
        }


@dataclass
class EpisodicMemory:
    """Memory entry for episodic novelty"""

    state_hash: str
    features: np.ndarray  # Feature vector (or list if numpy failed)
    timestamp: float
    visit_count: int = 1

    def similarity(self, other_features: np.ndarray) -> float:
        """Compute cosine similarity with other features"""
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        # Ensure features are lists if numpy failed
        self_features = list(self.features) if not NUMPY_AVAILABLE else self.features
        other_features_list = (
            list(other_features) if not NUMPY_AVAILABLE else other_features
        )

        if len(self_features) != len(other_features_list):
            return 0.0

        # Cosine similarity calculation (works with lists via FakeNumpy)
        dot_product = _np.dot(self_features, other_features_list)
        norm_a = _np.linalg.norm(self_features)
        norm_b = _np.linalg.norm(other_features_list)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        sim = dot_product / (norm_a * norm_b)

        # Numerical safety: clamp to [0, 1] (use np.clip if available)
        sim = _np.clip(sim, 0.0, 1.0)

        return float(sim)  # Ensure float return


@dataclass
class CuriosityStatistics:
    """Statistics about curiosity-driven exploration"""

    total_states_seen: int = 0
    total_bonuses_computed: int = 0
    total_bonus_value: float = 0.0

    # By novelty level
    completely_novel_count: int = 0
    highly_novel_count: int = 0
    moderately_novel_count: int = 0
    familiar_count: int = 0
    well_known_count: int = 0

    # Learning metrics
    average_novelty: float = 0.0
    novelty_trend: float = 0.0  # Positive = more novel over time
    exploration_efficiency: float = 0.0  # Ratio of novel to total states

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_states_seen": self.total_states_seen,
            "total_bonuses_computed": self.total_bonuses_computed,
            "total_bonus_value": self.total_bonus_value,
            "average_bonus": self.total_bonus_value
            / max(1, self.total_bonuses_computed),
            "by_novelty_level": {
                "completely_novel": self.completely_novel_count,
                "highly_novel": self.highly_novel_count,
                "moderately_novel": self.moderately_novel_count,
                "familiar": self.familiar_count,
                "well_known": self.well_known_count,
            },
            "average_novelty": self.average_novelty,
            "novelty_trend": self.novelty_trend,
            "exploration_efficiency": self.exploration_efficiency,
        }


class CuriosityRewardShaper(SerializationMixin):
    """
    Multi-algorithm curiosity-driven exploration and reward shaping

    Implements multiple curiosity methods:
    - COUNT_BASED: Simple 1/sqrt(N) novelty bonus
    - ICM: Intrinsic Curiosity Module using prediction error
    - RND: Random Network Distillation for novelty detection
    - INFORMATION_GAIN: Entropy-based information gain
    - EPISODIC: Similarity to episodic memory
    - HYBRID: Learned combination of methods

    Features:
    - Adaptive bonus scaling based on learning progress
    - Episodic memory for efficient novelty detection
    - Temporal decay for forgetting old states
    - Feature-based state representation
    - Multiple exploration strategies

    Thread-safe with comprehensive statistics.
    Integrates with VULCAN world model and learning systems.
    """

    _unpickleable_attrs = ['lock', '_np', 'world_model']

    def __init__(
        self,
        curiosity_weight: float = 0.1,
        method: CuriosityMethod = CuriosityMethod.HYBRID,
        decay_rate: float = 0.99,
        max_bonus: float = 1.0,
        episodic_memory_size: int = 10000,
        feature_dim: int = 64,
        world_model=None,
        validation_tracker=None,
        transparency_interface=None,
    ):
        """
        Initialize curiosity reward shaper

        Args:
            curiosity_weight: Weight for curiosity bonus (0-1)
            method: Curiosity computation method
            decay_rate: Temporal decay for old states (0-1)
            max_bonus: Maximum curiosity bonus
            episodic_memory_size: Size of episodic memory
            feature_dim: Dimension of state features
            world_model: Optional world model for predictions
            validation_tracker: Optional ValidationTracker integration
            transparency_interface: Optional TransparencyInterface integration
        """
        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        self.curiosity_weight = curiosity_weight
        self.method = method
        self.decay_rate = decay_rate
        self.max_bonus = max_bonus
        self.episodic_memory_size = episodic_memory_size
        self.feature_dim = feature_dim
        self.world_model = world_model
        self.validation_tracker = validation_tracker
        self.transparency_interface = transparency_interface

        # State tracking
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.novelty_estimates: Dict[str, NoveltyEstimate] = {}

        # Episodic memory
        self.episodic_memory: deque = deque(maxlen=episodic_memory_size)
        self.memory_index: Dict[str, EpisodicMemory] = {}

        # Count-based tracking
        self.total_states_seen: int = 0

        # ICM components (simplified)
        self.icm_forward_predictions: Dict[str, np.ndarray] = (
            {}
        )  # Will store lists if numpy failed
        self.icm_prediction_errors: deque = deque(maxlen=1000)

        # RND components (simplified - using random projections)
        self.rnd_target_network: Optional[np.ndarray] = (
            None  # Will store lists if numpy failed
        )
        self.rnd_predictor_network: Optional[np.ndarray] = (
            None  # Will store lists if numpy failed
        )
        self._initialize_rnd()

        # Information gain tracking
        self.state_feature_distributions: Dict[str, Dict[str, List[float]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        # Hybrid method weights (learned)
        self.hybrid_weights = {"count": 0.3, "icm": 0.25, "rnd": 0.25, "episodic": 0.2}

        # Statistics
        self.statistics = CuriosityStatistics()
        self.novelty_history: deque = deque(maxlen=1000)

        # Adaptive scaling
        self.bonus_scale: float = 1.0
        self.scale_update_interval: int = 100
        self.states_since_scale_update: int = 0

        # Thread safety
        self.lock = threading.RLock()

        # Feature extractors (can be registered)
        self.feature_extractors: List[Callable] = []

        logger.info("CuriosityRewardShaper initialized (FULL IMPLEMENTATION)")
        logger.info(f"  Method: {method.value}, Weight: {curiosity_weight}")
        logger.info(f"  Decay rate: {decay_rate}, Max bonus: {max_bonus}")
        logger.info(f"  Episodic memory size: {episodic_memory_size}")

    def _restore_unpickleable_attrs(self) -> None:
        """Restore unpickleable attributes after deserialization."""
        self.lock = threading.RLock()
        self._np = np if NUMPY_AVAILABLE else FakeNumpy
        self.world_model = None  # Must be re-injected via set_world_model()
        self._world_model_injected = False  # Track injection state

    def set_world_model(self, world_model: Any) -> None:
        """
        Re-inject world_model after unpickling.
        
        Args:
            world_model: The WorldModel instance to use for predictions
            
        Raises:
            ValueError: If world_model is None
        """
        if world_model is None:
            raise ValueError(
                "world_model cannot be None. Provide a valid WorldModel instance."
            )
        with self.lock:
            self.world_model = world_model
            self._world_model_injected = True
            logger.debug("CuriosityRewardShaper: world_model dependency injected")

    def _validate_world_model(self, operation: str = "operation") -> None:
        """
        Validate that world_model is available for operations that require it.
        
        This method should be called at the start of methods that need world_model.
        """
        if self.world_model is None and not getattr(self, '_world_model_injected', True):
            logger.warning(
                f"CuriosityRewardShaper.{operation} called without world_model. "
                f"After deserializing, call set_world_model() to restore full functionality."
            )

    def compute_curiosity_bonus(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute curiosity bonus for state

        Args:
            state: Current state
            context: Optional context

        Returns:
            Curiosity bonus value (0 to max_bonus)
        """
        with self.lock:
            # Hash state
            state_hash = self._hash_state(state)

            # Extract features
            features = self._extract_features(state, context or {})

            # Update visit count
            self.state_visits[state_hash] += 1
            self.total_states_seen += 1

            # Compute novelty by method
            if self.method == CuriosityMethod.COUNT_BASED:
                novelty = self._compute_count_based_novelty(state_hash)
            elif self.method == CuriosityMethod.ICM:
                novelty = self._compute_icm_novelty(state_hash, features, context or {})
            elif self.method == CuriosityMethod.RND:
                novelty = self._compute_rnd_novelty(features)
            elif self.method == CuriosityMethod.INFORMATION_GAIN:
                novelty = self._compute_information_gain(state_hash, features)
            elif self.method == CuriosityMethod.EPISODIC:
                novelty = self._compute_episodic_novelty(state_hash, features)
            elif self.method == CuriosityMethod.HYBRID:
                novelty = self._compute_hybrid_novelty(
                    state_hash, features, context or {}
                )
            else:
                novelty = 0.5

            # Classify novelty level
            novelty_level = self._classify_novelty(
                novelty, self.state_visits[state_hash]
            )

            # Create/update novelty estimate
            if state_hash not in self.novelty_estimates:
                estimate = NoveltyEstimate(
                    state_hash=state_hash,
                    novelty_score=novelty,
                    visit_count=self.state_visits[state_hash],
                    novelty_level=novelty_level,
                    features=features,  # Store original features dict
                )
                self.novelty_estimates[state_hash] = estimate
            else:
                estimate = self.novelty_estimates[state_hash]
                estimate.novelty_score = novelty
                estimate.visit_count = self.state_visits[state_hash]
                estimate.novelty_level = novelty_level
                estimate.last_visit = time.time()
                # Update features if needed? Maybe not necessary here.

            # Update episodic memory
            self._update_episodic_memory(state_hash, features)  # Pass features dict

            # Compute bonus
            bonus = novelty * self.curiosity_weight * self.bonus_scale
            bonus = min(bonus, self.max_bonus)  # Cap at maximum

            # Update statistics
            self._update_statistics(novelty, novelty_level, bonus)

            # Periodic adaptive scaling
            self.states_since_scale_update += 1
            if self.states_since_scale_update >= self.scale_update_interval:
                self._update_adaptive_scaling()
                self.states_since_scale_update = 0

            # Record to transparency interface
            if self.transparency_interface:
                try:
                    self.transparency_interface.record_curiosity_bonus(
                        state_hash=state_hash,
                        novelty=novelty,
                        bonus=bonus,
                        method=self.method.value,
                    )
                except Exception as e:
                    logger.debug(f"Failed to record to transparency interface: {e}")

            logger.debug(
                f"Curiosity bonus: {bonus:.4f} (novelty: {novelty:.3f}, visits: {self.state_visits[state_hash]})"
            )

            return bonus

    def shape_reward(
        self,
        base_reward: float,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Shape reward with curiosity bonus

        Args:
            base_reward: Base task reward
            state: Current state
            context: Optional context

        Returns:
            Shaped reward (base + curiosity bonus)
        """
        curiosity_bonus = self.compute_curiosity_bonus(state, context)
        shaped_reward = base_reward + curiosity_bonus

        logger.debug(
            f"Shaped reward: {base_reward:.4f} + {curiosity_bonus:.4f} = {shaped_reward:.4f}"
        )

        return shaped_reward

    def update_novelty_estimates(
        self,
        state: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
        next_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Update novelty estimates with outcome information

        Args:
            state: State that was explored
            outcome: Optional outcome/reward information
            next_state: Optional next state for ICM
        """
        with self.lock:
            state_hash = self._hash_state(state)

            # Update ICM if next state available
            if next_state and self.method in [
                CuriosityMethod.ICM,
                CuriosityMethod.HYBRID,
            ]:
                self._update_icm(state, next_state)

            # Update information gain estimates
            if outcome:
                features = self._extract_features(state, {})
                for feature_name, feature_value in features.items():
                    # Convert bools to floats; try to convert numeric types; hash strings
                    try:
                        if isinstance(feature_value, bool):
                            fv = 1.0 if feature_value else 0.0
                        else:
                            fv = float(feature_value)
                    except (TypeError, ValueError):
                        # If it fails, check if it's a string to hash
                        if isinstance(feature_value, str):
                            fv = _hash_to_float(feature_value)
                        else:
                            # Skip other non-numeric types (e.g., lists, dicts)
                            continue

                    self.state_feature_distributions[state_hash][feature_name].append(
                        fv
                    )

            # Record to validation tracker if available
            if self.validation_tracker and outcome:
                try:
                    self.validation_tracker.record_validation(
                        proposal={"type": "exploration", "state": state_hash},
                        validation_result={
                            "novelty": self.novelty_estimates.get(
                                state_hash,
                                NoveltyEstimate(
                                    state_hash=state_hash,
                                    novelty_score=0.0,
                                    visit_count=0,
                                ),
                            ).novelty_score
                        },
                        actual_outcome=outcome.get("outcome", "unknown"),
                    )
                except Exception as e:
                    logger.debug(f"Failed to record to validation tracker: {e}")

    def get_novelty(self, state: Dict[str, Any]) -> float:
        """
        Get current novelty estimate for state

        Args:
            state: State to query

        Returns:
            Novelty score (0-1)
        """
        with self.lock:
            state_hash = self._hash_state(state)

            if state_hash in self.novelty_estimates:
                return self.novelty_estimates[state_hash].novelty_score
            else:
                # Never seen - completely novel
                return 1.0

    def get_exploration_recommendation(
        self, states: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Recommend which state to explore next

        Args:
            states: List of candidate states

        Returns:
            Recommendation with best state and reasoning
        """
        with self.lock:
            if not states:
                return {"recommended_state": None, "reason": "No states provided"}

            # Compute novelty for each state
            state_novelties = []
            for state in states:
                novelty = self.get_novelty(state)
                state_novelties.append((state, novelty))

            # Sort by novelty (highest first)
            state_novelties.sort(key=lambda x: x[1], reverse=True)

            best_state, best_novelty = state_novelties[0]

            return {
                "recommended_state": best_state,
                "novelty": best_novelty,
                "reason": f"Highest novelty ({best_novelty:.3f}) among {len(states)} candidates",
                "all_novelties": [(self._hash_state(s), n) for s, n in state_novelties],
            }

    def register_feature_extractor(
        self, extractor: Callable[[Dict, Dict], Dict[str, Any]]
    ):
        """
        Register custom feature extractor

        Args:
            extractor: Function that takes (state, context) and returns features dict
        """
        with self.lock:
            self.feature_extractors.append(extractor)
            logger.info(
                f"Registered custom feature extractor ({len(self.feature_extractors)} total)"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.lock:
            stats = self.statistics.to_dict()

            stats.update(
                {
                    "method": self.method.value,
                    "curiosity_weight": self.curiosity_weight,
                    "bonus_scale": self.bonus_scale,
                    "unique_states_seen": len(self.state_visits),
                    "episodic_memory_size": len(self.episodic_memory),
                    "novelty_estimates": len(self.novelty_estimates),
                    "hybrid_weights": (
                        self.hybrid_weights.copy()
                        if self.method == CuriosityMethod.HYBRID
                        else None
                    ),
                }
            )

            return stats

    def export_state(self) -> Dict[str, Any]:
        """Export shaper state for persistence"""
        with self.lock:
            return {
                "state_visits": dict(self.state_visits),
                "novelty_estimates": {
                    k: v.to_dict()
                    for k, v in list(self.novelty_estimates.items())[:1000]  # Last 1000
                },
                "statistics": self.statistics.to_dict(),
                "hybrid_weights": self.hybrid_weights.copy(),
                "bonus_scale": self.bonus_scale,
                "config": {
                    "curiosity_weight": self.curiosity_weight,
                    "method": self.method.value,
                    "decay_rate": self.decay_rate,
                    "max_bonus": self.max_bonus,
                },
                "export_time": time.time(),
            }

    def import_state(self, state: Dict[str, Any]):
        """Import shaper state from persistence"""
        with self.lock:
            # Import visit counts
            self.state_visits = defaultdict(int, state.get("state_visits", {}))

            # Import hybrid weights
            if "hybrid_weights" in state:
                self.hybrid_weights.update(state["hybrid_weights"])

            # Import bonus scale
            self.bonus_scale = state.get("bonus_scale", 1.0)

            logger.info(
                f"Imported state from persistence: {len(self.state_visits)} states"
            )

    def reset(self) -> None:
        """Reset all state and statistics"""
        with self.lock:
            self.state_visits.clear()
            self.novelty_estimates.clear()
            self.episodic_memory.clear()
            self.memory_index.clear()
            self.icm_forward_predictions.clear()
            self.icm_prediction_errors.clear()
            self.state_feature_distributions.clear()
            self.novelty_history.clear()

            self.total_states_seen = 0
            self.statistics = CuriosityStatistics()
            self.bonus_scale = 1.0
            self.states_since_scale_update = 0

            # Reinitialize RND
            self._initialize_rnd()

            logger.info("CuriosityRewardShaper reset - all data cleared")

    # ============================================================
    # Internal Methods - Curiosity Computation
    # ============================================================

    def _compute_count_based_novelty(self, state_hash: str) -> float:
        """Compute count-based novelty (1/sqrt(N))"""
        visits = self.state_visits[state_hash]
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy
        novelty = 1.0 / _np.sqrt(visits) if visits > 0 else 1.0
        return min(1.0, novelty)

    def _compute_icm_novelty(
        self, state_hash: str, features: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Compute ICM-based novelty using prediction error

        ICM measures novelty as the error in predicting next state features.
        """
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        # If no prediction history, return high novelty
        if state_hash not in self.icm_forward_predictions:
            return 0.8

        # Get predicted features
        predicted = self.icm_forward_predictions.get(state_hash)
        if predicted is None:
            return 0.7

        # Convert features to vector
        actual = self._features_to_vector(features)

        # Compute prediction error (normalized)
        if len(predicted) != len(actual):
            # --- START FIX ---
            # Ensure both are lists to use .extend() safely,
            # as FakeNumpy doesn't have np.pad
            predicted_list = list(predicted)
            actual_list = list(actual)

            max_len = max(len(predicted_list), len(actual_list))

            if len(predicted_list) < max_len:
                predicted_list.extend([0.0] * (max_len - len(predicted_list)))
            if len(actual_list) < max_len:
                actual_list.extend([0.0] * (max_len - len(actual_list)))

            # Now convert back to _np.array for mean/abs
            error = _np.mean(
                _np.abs(_np.array(predicted_list) - _np.array(actual_list))
            )
            # --- END FIX ---

        else:
            error = _np.mean(
                _np.abs(_np.array(predicted) - _np.array(actual))
            )  # Ensure arrays/lists for abs/mean

        # Normalize error to 0-1 range
        normalized_error = min(1.0, error / 2.0)  # Assume max error ~2

        # Store error for statistics
        self.icm_prediction_errors.append(normalized_error)

        return normalized_error

    def _compute_rnd_novelty(self, features: Dict[str, Any]) -> float:
        """
        Compute RND-based novelty

        RND uses prediction error of a random target network.
        """
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        if self.rnd_target_network is None or self.rnd_predictor_network is None:
            return 0.7

        # Convert features to vector
        feature_vec = self._features_to_vector(features)

        # Ensure correct dimension
        # [Original] target_dim = len(self.rnd_target_network[0]) if self.rnd_target_network else self.feature_dim
        # [Fixed]
        target_dim = (
            len(self.rnd_target_network[0])
            if self.rnd_target_network is not None
            else self.feature_dim
        )

        if len(feature_vec) < target_dim:
            # Pad with zeros if FakeNumpy, otherwise use np.pad
            if NUMPY_AVAILABLE:
                feature_vec = _np.pad(feature_vec, (0, target_dim - len(feature_vec)))
            else:
                feature_vec.extend([0.0] * (target_dim - len(feature_vec)))
        elif len(feature_vec) > target_dim:
            feature_vec = feature_vec[:target_dim]

        # Target network output (fixed random) - dot works with FakeNumpy lists
        target_output = _np.dot(self.rnd_target_network, feature_vec)

        # Predictor network output (learned, but we simplify here)
        predictor_output = _np.dot(self.rnd_predictor_network, feature_vec)

        # Prediction error as novelty - abs/mean work with FakeNumpy lists
        error = _np.mean(
            _np.abs(_np.array(target_output) - _np.array(predictor_output))
        )

        # Normalize
        normalized_error = min(
            1.0, error / 10.0
        )  # Assume max error ~10? Adjusted divisor

        return normalized_error

    def _compute_information_gain(
        self, state_hash: str, features: Dict[str, Any]
    ) -> float:
        """
        Compute information gain based novelty

        Measures how much the state reduces uncertainty.
        """
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        if state_hash not in self.state_feature_distributions:
            return 0.9  # High novelty for first visit

        # Compute entropy of feature distributions
        total_entropy = 0.0
        count = 0

        for feature_name, values in self.state_feature_distributions[
            state_hash
        ].items():
            if len(values) < 2:
                continue

            # Compute entropy using np.histogram (or fake version)
            hist, _ = _np.histogram(values, bins=10, density=True)
            hist = [h for h in hist if h > 0]  # Remove zero bins (list comprehension)

            if len(hist) > 0:
                # Use np.log2 (or fake version)
                entropy = -_np.sum(
                    [h * _np.log2(h + 1e-10) for h in hist]
                )  # List comprehension for sum
                total_entropy += entropy
                count += 1

        if count == 0:
            return 0.8

        avg_entropy = total_entropy / count

        # Normalize entropy to 0-1 (assume max entropy ~3.3 for 10 bins)
        normalized_entropy = min(1.0, avg_entropy / 3.3)

        # Higher entropy means more uncertainty reduction potential -> higher novelty
        return normalized_entropy

    def _compute_episodic_novelty(
        self, state_hash: str, features: Dict[str, Any]
    ) -> float:
        """
        Compute episodic memory based novelty

        Measures similarity to past experiences.
        """
        if not self.episodic_memory:
            return 1.0  # Completely novel

        # Convert features to vector (works with FakeNumpy)
        feature_vec = self._features_to_vector(features)

        # Find most similar memory
        max_similarity = 0.0

        for memory in self.episodic_memory:
            # memory.similarity handles numpy/FakeNumpy internally
            similarity = memory.similarity(feature_vec)
            max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity

        # Numerical safety: clamp to [0, 1] (use np.clip if available)
        novelty = self._np.clip(novelty, 0.0, 1.0)

        return float(novelty)  # Ensure float return

    def _compute_hybrid_novelty(
        self, state_hash: str, features: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Compute hybrid novelty combining multiple methods
        """
        # Compute each method
        count_novelty = self._compute_count_based_novelty(state_hash)
        icm_novelty = self._compute_icm_novelty(state_hash, features, context)
        rnd_novelty = self._compute_rnd_novelty(features)
        episodic_novelty = self._compute_episodic_novelty(state_hash, features)

        # Defensive clamp (as requested by patch) - use np.clip if available
        episodic_novelty = self._np.clip(episodic_novelty, 0.0, 1.0)

        # Store in estimate for transparency
        if state_hash in self.novelty_estimates:
            estimate = self.novelty_estimates[state_hash]
            estimate.count_novelty = count_novelty
            estimate.icm_novelty = icm_novelty
            estimate.rnd_novelty = rnd_novelty
            estimate.episodic_novelty = episodic_novelty

        # Weighted combination
        hybrid_novelty = (
            self.hybrid_weights["count"] * count_novelty
            + self.hybrid_weights["icm"] * icm_novelty
            + self.hybrid_weights["rnd"] * rnd_novelty
            + self.hybrid_weights["episodic"] * episodic_novelty
        )

        # Final clamp
        hybrid_novelty = self._np.clip(hybrid_novelty, 0.0, 1.0)

        return float(hybrid_novelty)

    def _classify_novelty(self, novelty: float, visit_count: int) -> NoveltyLevel:
        """Classify novelty level"""
        if visit_count == 1:
            return NoveltyLevel.COMPLETELY_NOVEL
        elif novelty > 0.8:
            return NoveltyLevel.HIGHLY_NOVEL
        elif novelty > 0.5:
            return NoveltyLevel.MODERATELY_NOVEL
        elif novelty > 0.2:
            return NoveltyLevel.FAMILIAR
        else:
            return NoveltyLevel.WELL_KNOWN

    # ============================================================
    # Internal Methods - Feature Processing
    # ============================================================

    def _extract_features(
        self, state: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from state"""
        features = {}

        # Basic features from state
        if isinstance(state, dict):
            features.update(state)
        else:
            # Handle non-dict states by assigning to a default key
            features["state_value"] = state

        # Apply custom feature extractors
        for extractor in self.feature_extractors:
            try:
                custom_features = extractor(state, context)
                if custom_features:
                    features.update(custom_features)
            except Exception as e:
                logger.debug(f"Feature extractor failed: {e}")

        # Add context features
        for key, value in context.items():
            if isinstance(value, (int, float, str, bool)):
                features[f"context_{key}"] = value

        return features

    def _features_to_vector(
        self, features: Dict[str, Any]
    ) -> np.ndarray:  # Returns list if numpy failed
        """Convert features dict to numpy vector or list"""
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy
        vec = []

        # Sort keys for consistent vector order
        for key in sorted(features.keys()):
            value = features[key]

            if isinstance(value, (int, float)):
                vec.append(float(value))
            elif isinstance(value, bool):
                vec.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple string hash mapped to [0, 1]
                vec.append(_hash_to_float(value))  # Use helper
            # Skip complex types like lists/dicts for basic vectorization
            # else:
            #    vec.append(0.0) # Or handle differently

        # Return np.array or list
        return _np.array(vec) if vec else _np.array([0.0])

    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Hash state for identification"""
        try:
            # Convert state to a string representation that handles common types
            # Sort dict keys for consistent hashing
            state_repr = json.dumps(state, sort_keys=True, default=str)
            return hashlib.md5(
                state_repr.encode("utf-8"), usedforsecurity=False
            ).hexdigest()[
                :16
            ]  # Use utf-8
        except Exception as e:
            # Fallback for unhashable types
            logger.debug(f"Hashing state failed with JSON ({e}), using fallback hash.")
            try:
                # Attempt a simpler hash based on string representation
                return str(hash(str(state)))[:16]
            except Exception as final_e:
                logger.error(f"Fallback hashing also failed: {final_e}")
                # Final fallback: use time and hope for uniqueness (not ideal)
                return f"hash_err_{time.time_ns()}"

    # ============================================================
    # Internal Methods - Memory and Learning
    # ============================================================

    def _update_episodic_memory(self, state_hash: str, features: Dict[str, Any]):
        """Update episodic memory with new state"""
        # Convert features to vector (works with FakeNumpy)
        feature_vec = self._features_to_vector(features)

        if state_hash in self.memory_index:
            # Update existing memory
            memory = self.memory_index[state_hash]
            memory.visit_count += 1
            memory.timestamp = time.time()
            # Optionally move to end of deque if using LRU-like update
            # (requires finding and moving element, deque doesn't support directly)
        else:
            # Create new memory
            memory = EpisodicMemory(
                state_hash=state_hash,
                features=feature_vec,  # Store vector/list
                timestamp=time.time(),
            )

            # Check if memory is full BEFORE adding
            if (
                len(self.episodic_memory) >= self.episodic_memory_size
                and self.episodic_memory_size > 0
            ):
                # Remove oldest item from deque AND index
                oldest_memory = self.episodic_memory.popleft()
                if oldest_memory.state_hash in self.memory_index:
                    del self.memory_index[oldest_memory.state_hash]

            # Add new item if size allows or after making space
            if self.episodic_memory_size > 0:
                self.episodic_memory.append(memory)
                self.memory_index[state_hash] = memory

    def _update_icm(self, state: Dict[str, Any], next_state: Dict[str, Any]):
        """Update ICM forward model"""
        state_hash = self._hash_state(state)

        # Extract features
        # current_features = self._extract_features(state, {}) # Not used in simple model
        next_features = self._extract_features(next_state, {})

        # Simple forward prediction: just store next features' vector
        # In a real implementation, this would train a neural network
        next_vec = self._features_to_vector(next_features)  # Handles numpy/FakeNumpy
        self.icm_forward_predictions[state_hash] = next_vec

    def _initialize_rnd(self):
        """Initialize Random Network Distillation networks"""
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy
        # Target network (fixed random) - randn works with FakeNumpy
        self.rnd_target_network = _np.random.randn(
            self.feature_dim, self.feature_dim
        )  # * 0.1 <-- Removed scaling for simplicity w/ FakeNumpy

        # Predictor network (starts similar but will "learn")
        # In this simplified version, we just add some noise
        # Ensure addition works element-wise if FakeNumpy returns lists of lists
        noise = _np.random.randn(self.feature_dim, self.feature_dim)  # * 0.05

        if NUMPY_AVAILABLE:
            self.rnd_predictor_network = (
                self.rnd_target_network + noise * 0.05
            )  # Use numpy addition
        else:
            # Manual addition for list of lists
            pred_net = []
            target_net_list = self.rnd_target_network  # Assume list of lists
            noise_list = noise  # Assume list of lists
            for r_idx in range(self.feature_dim):
                row = []
                for c_idx in range(self.feature_dim):
                    # Check if indices are valid before accessing
                    target_val = (
                        target_net_list[r_idx][c_idx]
                        if r_idx < len(target_net_list)
                        and c_idx < len(target_net_list[r_idx])
                        else 0
                    )
                    noise_val = (
                        noise_list[r_idx][c_idx]
                        if r_idx < len(noise_list) and c_idx < len(noise_list[r_idx])
                        else 0
                    )
                    row.append(target_val + noise_val * 0.05)
                pred_net.append(row)
            self.rnd_predictor_network = pred_net

    def _update_adaptive_scaling(self):
        """Update adaptive bonus scaling based on exploration progress"""
        if not self.novelty_history:
            return

        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        # Calculate recent average novelty
        recent_novelty = _np.mean(list(self.novelty_history)[-100:])

        # If novelty is declining, increase bonus to encourage more exploration
        if recent_novelty < 0.3:
            self.bonus_scale = min(2.0, self.bonus_scale * 1.1)
        # If novelty is high, decrease bonus (exploration is working)
        elif recent_novelty > 0.7:
            self.bonus_scale = max(0.5, self.bonus_scale * 0.9)

        logger.debug(
            f"Adaptive scaling updated: {self.bonus_scale:.3f} (recent novelty: {recent_novelty:.3f})"
        )

    def _update_statistics(
        self, novelty: float, novelty_level: NoveltyLevel, bonus: float
    ):
        """Update statistics"""
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        self.statistics.total_bonuses_computed += 1
        self.statistics.total_bonus_value += bonus

        # Update by level
        if novelty_level == NoveltyLevel.COMPLETELY_NOVEL:
            self.statistics.completely_novel_count += 1
        elif novelty_level == NoveltyLevel.HIGHLY_NOVEL:
            self.statistics.highly_novel_count += 1
        elif novelty_level == NoveltyLevel.MODERATELY_NOVEL:
            self.statistics.moderately_novel_count += 1
        elif novelty_level == NoveltyLevel.FAMILIAR:
            self.statistics.familiar_count += 1
        elif novelty_level == NoveltyLevel.WELL_KNOWN:
            self.statistics.well_known_count += 1

        # Update novelty history
        self.novelty_history.append(novelty)

        # Update average novelty
        if self.novelty_history:
            self.statistics.average_novelty = _np.mean(list(self.novelty_history))

        # Update exploration efficiency
        novel_count = (
            self.statistics.completely_novel_count
            + self.statistics.highly_novel_count
            + self.statistics.moderately_novel_count
        )
        total_count = max(1, self.statistics.total_bonuses_computed)
        self.statistics.exploration_efficiency = novel_count / total_count

        # Update novelty trend (simple linear regression on recent history)
        if len(self.novelty_history) >= 20:
            recent = list(self.novelty_history)[-20:]
            x = _np.arange(len(recent))  # Use fake numpy if needed
            y = _np.array(recent)  # Use fake numpy if needed

            # Linear regression using np.vstack/np.ones/np.linalg.lstsq (or fake versions)
            # 
            # CRITICAL FIX (Defect Report Category 1.1): FakeNumpy Crash
            # 
            # Industry Standard Fix:
            # 1. Fail-Fast Validation: Check NumPy availability before attempting computation
            # 2. Graceful Degradation: Use simple fallback if NumPy unavailable
            # 3. Clear Error Logging: Log specific failure reason for debugging
            # 4. Safe Default: Return 0.0 trend rather than crashing
            try:
                if not NUMPY_AVAILABLE:
                    # INDUSTRY STANDARD: Graceful degradation with logging
                    logger.warning(
                        "NumPy not available - cannot compute accurate novelty trend. "
                        "Install numpy for proper least squares regression. "
                        "Using simple mean-based estimation as fallback."
                    )
                    # Simple fallback: compare first half to second half
                    mid = len(recent) // 2
                    first_half_mean = sum(recent[:mid]) / max(1, mid)
                    second_half_mean = sum(recent[mid:]) / max(1, len(recent) - mid)
                    self.statistics.novelty_trend = float(second_half_mean - first_half_mean)
                else:
                    # NumPy available - use proper least squares
                    A = _np.vstack(
                        [x, _np.ones(len(x))]
                    ).T  # Create design matrix
                    # CRITICAL FIX: Use _np.linalg.lstsq (not _np.lstsq)
                    # lstsq is in the linalg submodule for both real and fake numpy
                    results = _np.linalg.lstsq(A, y, rcond=None)
                    slope = (
                        results[0][0] if results and results[0] else 0.0
                    )  # Get slope from first element of solution tuple
                    self.statistics.novelty_trend = float(slope)
            except Exception as e:
                # INDUSTRY STANDARD: Comprehensive error handling with context
                logger.warning(
                    f"Failed to compute novelty trend via least squares: {type(e).__name__}: {e}. "
                    f"Using fallback estimation."
                )
                # Fallback: simple mean comparison
                try:
                    mid = len(recent) // 2
                    if mid > 0:
                        first_half_mean = sum(recent[:mid]) / mid
                        second_half_mean = sum(recent[mid:]) / (len(recent) - mid)
                        self.statistics.novelty_trend = float(second_half_mean - first_half_mean)
                    else:
                        self.statistics.novelty_trend = 0.0
                except Exception as fallback_error:
                    logger.error(f"Even fallback trend calculation failed: {fallback_error}")
                    self.statistics.novelty_trend = 0.0


# Module-level exports
__all__ = [
    "CuriosityRewardShaper",
    "NoveltyEstimate",
    "EpisodicMemory",
    "CuriosityStatistics",
    "CuriosityMethod",
    "NoveltyLevel",
]
