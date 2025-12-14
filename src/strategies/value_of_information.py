"""
Value of Information (VOI) Gate for Tool Selection System

Determines whether gathering additional information (deeper feature extraction,
additional tool probing) would improve decision quality enough to justify the cost.
"""

import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)


class InformationSource(Enum):
    """Types of information that can be gathered"""

    TIER2_FEATURES = "tier2_structural"
    TIER3_FEATURES = "tier3_semantic"
    TIER4_FEATURES = "tier4_multimodal"
    PROBE_TOOL = "probe_tool"
    ENSEMBLE_PROBE = "ensemble_probe"
    MEMORY_LOOKUP = "memory_lookup"
    EXTERNAL_API = "external_api"


class VOIAction(Enum):
    """Actions based on VOI analysis"""

    PROCEED = "proceed"
    GATHER_MORE = "gather_more"
    PROBE_SPECIFIC = "probe_specific"
    FULL_ANALYSIS = "full_analysis"


@dataclass
class InformationCost:
    """Cost of gathering information"""

    time_ms: float
    energy_mj: float
    monetary: float = 0.0
    opportunity: float = 0.0

    def total_cost(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate total weighted cost"""
        if weights is None:
            weights = {"time": 1.0, "energy": 0.1, "monetary": 10.0, "opportunity": 0.5}

        return (
            weights.get("time", 1.0) * self.time_ms
            + weights.get("energy", 0.1) * self.energy_mj
            + weights.get("monetary", 10.0) * self.monetary
            + weights.get("opportunity", 0.5) * self.opportunity
        )


@dataclass
class InformationValue:
    """Value of information analysis result"""

    expected_value: float
    information_gain: float
    cost: InformationCost
    net_value: float
    recommendation: VOIAction
    source: InformationSource
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionState:
    """Current state of decision-making"""

    features: np.ndarray
    uncertainty: float
    current_best_tool: Optional[str] = None
    current_confidence: float = 0.5
    gathered_information: List[InformationSource] = field(default_factory=list)
    remaining_budget: Dict[str, float] = field(default_factory=dict)
    prior_distribution: Optional[np.ndarray] = None


class UncertaintyEstimator:
    """Estimates decision uncertainty"""

    def __init__(self):
        self.feature_importance = None
        self.calibration_data = deque(maxlen=1000)

    def estimate_uncertainty(
        self, features: np.ndarray, predictions: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate uncertainty in current decision

        Returns:
            Uncertainty score [0, 1]
        """

        uncertainties = []

        # Feature-based uncertainty
        feature_uncertainty = self._feature_uncertainty(features)
        uncertainties.append(feature_uncertainty)

        # Prediction entropy
        if predictions is not None:
            entropy_uncertainty = self._entropy_uncertainty(predictions)
            uncertainties.append(entropy_uncertainty)

        # Variance in predictions
        if predictions is not None and len(predictions) > 1:
            variance_uncertainty = self._variance_uncertainty(predictions)
            uncertainties.append(variance_uncertainty)

        # Combine uncertainties
        return np.mean(uncertainties)

    def _feature_uncertainty(self, features: np.ndarray) -> float:
        """Uncertainty based on feature quality"""

        # Check for missing or sparse features
        sparsity = np.sum(features == 0) / len(features)

        # Check for extreme values
        if np.std(features) > 0:
            z_scores = np.abs((features - np.mean(features)) / np.std(features))
            extremity = np.mean(z_scores > 3)
        else:
            extremity = 0

        # Feature quality score
        quality = 1 - (sparsity * 0.5 + extremity * 0.5)

        # Uncertainty is inverse of quality
        return 1 - quality

    def _entropy_uncertainty(self, predictions: np.ndarray) -> float:
        """Uncertainty based on prediction entropy"""

        # FIXED: Check for zero sum before normalization
        total = np.sum(np.abs(predictions))
        if total == 0:
            return 0.5  # Maximum uncertainty when all predictions are zero

        # Normalize predictions to probabilities
        probs = np.abs(predictions) / total

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by maximum entropy
        max_entropy = np.log(len(predictions))

        if max_entropy > 0:
            return entropy / max_entropy

        return 0.5

    def _variance_uncertainty(self, predictions: np.ndarray) -> float:
        """Uncertainty based on prediction variance"""

        # Coefficient of variation
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        if mean_pred > 0:
            cv = std_pred / mean_pred
            return min(1.0, cv)

        return 0.5


class InformationGainCalculator:
    """Calculates expected information gain"""

    def __init__(self):
        self.prior_entropy = {}
        self.conditional_entropy = {}

    def calculate_gain(
        self, current_distribution: np.ndarray, expected_posterior: np.ndarray
    ) -> float:
        """
        Calculate expected information gain

        Returns:
            Information gain in bits
        """

        # Current entropy
        current_entropy = self._entropy(current_distribution)

        # Expected posterior entropy
        posterior_entropy = self._entropy(expected_posterior)

        # Information gain
        return current_entropy - posterior_entropy

    def expected_gain_features(
        self, current_features: np.ndarray, feature_tier: int
    ) -> float:
        """
        Expected information gain from additional features

        Args:
            current_features: Current feature vector
            feature_tier: Target feature tier (2, 3, or 4)

        Returns:
            Expected information gain
        """

        # Estimate based on feature tier
        # Higher tiers provide more information
        tier_gains = {
            1: 0.1,  # Already have tier 1
            2: 0.3,  # Structural features
            3: 0.5,  # Semantic features
            4: 0.7,  # Multimodal features
        }

        base_gain = tier_gains.get(feature_tier, 0.2)

        # Adjust based on current feature quality
        feature_quality = 1 - (np.sum(current_features == 0) / len(current_features))

        # Less gain if already have good features
        adjusted_gain = base_gain * (1 - feature_quality * 0.5)

        return adjusted_gain

    def expected_gain_probe(self, tool_name: str, current_confidence: float) -> float:
        """
        Expected information gain from probing a tool

        Returns:
            Expected information gain
        """

        # More gain if current confidence is near 0.5 (maximum uncertainty)
        uncertainty = 1 - abs(current_confidence - 0.5) * 2

        # Base gain from probing
        base_gain = 0.4

        return base_gain * uncertainty

    def _entropy(self, distribution: np.ndarray) -> float:
        """Calculate entropy of distribution"""

        # FIXED: Check for zero sum before normalization
        total = np.sum(np.abs(distribution))
        if total == 0:
            return 0.0  # Zero entropy when distribution is all zeros

        # Ensure valid probability distribution
        probs = np.abs(distribution) / total

        # Calculate entropy
        return -np.sum(probs * np.log(probs + 1e-10))


class CostEstimator:
    """Estimates costs of information gathering"""

    def __init__(self):
        self.historical_costs = defaultdict(list)
        self.cost_models = {}

    def estimate_cost(
        self, source: InformationSource, context: Optional[Dict[str, Any]] = None
    ) -> InformationCost:
        """
        Estimate cost of gathering information

        Returns:
            InformationCost object
        """

        # Base costs for each source
        base_costs = {
            InformationSource.TIER2_FEATURES: InformationCost(50, 5, 0, 0.1),
            InformationSource.TIER3_FEATURES: InformationCost(200, 20, 0, 0.2),
            InformationSource.TIER4_FEATURES: InformationCost(500, 50, 0, 0.3),
            InformationSource.PROBE_TOOL: InformationCost(100, 10, 0, 0.15),
            InformationSource.ENSEMBLE_PROBE: InformationCost(300, 30, 0, 0.25),
            InformationSource.MEMORY_LOOKUP: InformationCost(10, 1, 0, 0.05),
            InformationSource.EXTERNAL_API: InformationCost(500, 5, 0.01, 0.2),
        }

        cost = base_costs.get(source, InformationCost(100, 10, 0, 0.1))

        # Adjust based on context
        if context:
            if "complexity" in context:
                complexity_factor = 1 + context["complexity"]
                cost.time_ms *= complexity_factor
                cost.energy_mj *= complexity_factor

            if "urgency" in context:
                # Higher opportunity cost if urgent
                cost.opportunity *= 1 + context["urgency"]

        # Use historical data if available
        if source in self.historical_costs and self.historical_costs[source]:
            historical_times = [c.time_ms for c in self.historical_costs[source]]
            cost.time_ms = np.mean(historical_times[-10:])  # Use recent history

        return cost

    def update_cost(self, source: InformationSource, actual_cost: InformationCost):
        """Update cost estimates with actual data"""

        self.historical_costs[source].append(actual_cost)

        # Keep only recent history
        if len(self.historical_costs[source]) > 100:
            self.historical_costs[source] = self.historical_costs[source][-100:]


class ValueCalculator:
    """Calculates value of decisions"""

    def __init__(self):
        self.utility_function = self._default_utility
        self.value_history = deque(maxlen=1000)

    def expected_value_current(
        self, confidence: float, expected_quality: float
    ) -> float:
        """
        Expected value of current decision

        Returns:
            Expected value
        """

        # Simple expected value
        value = confidence * expected_quality

        # Risk adjustment
        risk_penalty = (1 - confidence) * 0.2

        return value - risk_penalty

    def expected_value_with_info(
        self,
        current_confidence: float,
        expected_confidence_gain: float,
        expected_quality: float,
    ) -> float:
        """
        Expected value after gathering information

        Returns:
            Expected value
        """

        new_confidence = min(1.0, current_confidence + expected_confidence_gain)

        # Improved value from better decision
        value = new_confidence * expected_quality

        # Reduced risk from higher confidence
        risk_penalty = (1 - new_confidence) * 0.2

        return value - risk_penalty

    def value_of_perfect_information(
        self, current_value: float, perfect_value: float
    ) -> float:
        """
        Calculate value of perfect information (EVPI)

        Returns:
            EVPI value
        """

        return max(0, perfect_value - current_value)

    def _default_utility(self, outcome: float, confidence: float) -> float:
        """Default utility function"""

        # Risk-adjusted utility
        return outcome * confidence - (1 - confidence) * 0.1


class ValueOfInformationGate:
    """
    Main VOI gate for tool selection decisions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Components
        self.uncertainty_estimator = UncertaintyEstimator()
        self.gain_calculator = InformationGainCalculator()
        self.cost_estimator = CostEstimator()
        self.value_calculator = ValueCalculator()

        # Configuration
        self.voi_threshold = config.get("voi_threshold", 0.1)
        self.max_iterations = config.get("max_iterations", 3)
        self.myopic = config.get("myopic", True)  # Myopic vs non-myopic

        # State tracking
        self.decision_history = deque(maxlen=1000)
        self.voi_calculations = deque(maxlen=1000)

        # Statistics
        self.total_decisions = 0
        self.gather_decisions = 0
        self.proceed_decisions = 0
        self.total_value_gained = 0.0

        logger.info("VOI Gate initialized")

    def should_probe_deeper(
        self,
        features: np.ndarray,
        predictions: Optional[np.ndarray],
        budget_remaining: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if more information should be gathered

        Args:
            features: Current feature vector
            predictions: Current predictions from tools
            budget_remaining: Remaining time/energy budget

        Returns:
            (should_gather, action_type)
        """

        self.total_decisions += 1

        # Create decision state
        state = DecisionState(
            features=features,
            uncertainty=self.uncertainty_estimator.estimate_uncertainty(
                features, predictions
            ),
            remaining_budget=budget_remaining,
        )

        # Evaluate all information sources
        best_source = None
        best_value = -float("inf")

        for source in InformationSource:
            value = self.evaluate_information_source(state, source)

            if value and value.net_value > best_value:
                best_value = value.net_value
                best_source = value

        # Decision threshold
        if best_source and best_source.net_value > self.voi_threshold:
            self.gather_decisions += 1

            # Record decision
            self._record_decision(state, best_source, True)

            return True, best_source.source.value
        else:
            self.proceed_decisions += 1

            # Record decision
            self._record_decision(state, best_source, False)

            return False, None

    def evaluate_information_source(
        self, state: DecisionState, source: InformationSource
    ) -> Optional[InformationValue]:
        """
        Evaluate value of specific information source

        Returns:
            InformationValue or None if not feasible
        """

        # Estimate cost
        cost = self.cost_estimator.estimate_cost(
            source,
            {
                "complexity": np.mean(state.features),
                "urgency": 1.0 - state.remaining_budget.get("time_ms", 1000) / 1000,
            },
        )

        # Check budget constraints
        if not self._check_budget(state.remaining_budget, cost):
            return None

        # Calculate expected information gain
        if source in [
            InformationSource.TIER2_FEATURES,
            InformationSource.TIER3_FEATURES,
            InformationSource.TIER4_FEATURES,
        ]:
            # FIXED: Extract tier number from enum name (e.g., "TIER2_FEATURES" -> 2)
            tier = int(source.name.split("TIER")[1][0])
            info_gain = self.gain_calculator.expected_gain_features(
                state.features, tier
            )
        elif source == InformationSource.PROBE_TOOL:
            info_gain = self.gain_calculator.expected_gain_probe(
                state.current_best_tool or "unknown", state.current_confidence
            )
        else:
            info_gain = 0.3  # Default gain

        # Calculate expected value improvement
        current_value = self.value_calculator.expected_value_current(
            state.current_confidence,
            0.7,  # Expected quality placeholder
        )

        # Expected confidence gain from information
        confidence_gain = info_gain * 0.3  # Heuristic conversion

        value_with_info = self.value_calculator.expected_value_with_info(
            state.current_confidence,
            confidence_gain,
            0.7,  # Expected quality
        )

        expected_value = value_with_info - current_value

        # Net value
        net_value = expected_value - cost.total_cost() / 1000  # Normalize cost

        # Recommendation
        if net_value > self.voi_threshold:
            recommendation = VOIAction.GATHER_MORE
        else:
            recommendation = VOIAction.PROCEED

        return InformationValue(
            expected_value=expected_value,
            information_gain=info_gain,
            cost=cost,
            net_value=net_value,
            recommendation=recommendation,
            source=source,
            confidence=min(0.95, state.current_confidence + confidence_gain),
        )

    def _check_budget(self, budget: Dict[str, float], cost: InformationCost) -> bool:
        """Check if cost fits within budget"""

        if "time_ms" in budget and cost.time_ms > budget["time_ms"]:
            return False

        if "energy_mj" in budget and cost.energy_mj > budget["energy_mj"]:
            return False

        if "monetary" in budget and cost.monetary > budget["monetary"]:
            return False

        return True

    def calculate_evpi(
        self, current_distribution: np.ndarray, tool_utilities: np.ndarray
    ) -> float:
        """
        Calculate Expected Value of Perfect Information

        Args:
            current_distribution: Current probability distribution over tools
            tool_utilities: Utility of each tool if chosen

        Returns:
            EVPI value
        """

        # Expected value under current information
        current_value = np.sum(current_distribution * tool_utilities)

        # Value under perfect information (always choose best)
        perfect_value = np.max(tool_utilities)

        # EVPI
        evpi = perfect_value - current_value

        return max(0, evpi)

    def calculate_evsi(
        self,
        current_distribution: np.ndarray,
        sample_distribution: np.ndarray,
        tool_utilities: np.ndarray,
    ) -> float:
        """
        Calculate Expected Value of Sample Information

        Args:
            current_distribution: Current distribution
            sample_distribution: Distribution after sampling
            tool_utilities: Utility values

        Returns:
            EVSI value
        """

        # Current expected value
        current_value = np.sum(current_distribution * tool_utilities)

        # Expected value with sample information
        sample_value = np.sum(sample_distribution * tool_utilities)

        # EVSI
        evsi = sample_value - current_value

        return max(0, evsi)

    def multi_stage_voi(
        self, state: DecisionState, horizon: int = 3
    ) -> List[InformationValue]:
        """
        Non-myopic multi-stage VOI analysis

        Args:
            state: Current decision state
            horizon: Planning horizon

        Returns:
            Sequence of information gathering actions
        """

        if self.myopic or horizon <= 1:
            # Myopic decision
            best_value = None
            for source in InformationSource:
                value = self.evaluate_information_source(state, source)
                if value and (
                    best_value is None or value.net_value > best_value.net_value
                ):
                    best_value = value

            return [best_value] if best_value else []

        # Dynamic programming for multi-stage
        return self._backward_induction(state, horizon)

    def _backward_induction(
        self, state: DecisionState, horizon: int
    ) -> List[InformationValue]:
        """
        Backward induction for optimal information gathering sequence
        """

        # Simplified implementation
        sequence = []
        current_state = state

        for stage in range(horizon):
            best_action = None
            best_value = -float("inf")

            for source in InformationSource:
                # Skip if already gathered
                if source in current_state.gathered_information:
                    continue

                value = self.evaluate_information_source(current_state, source)

                if value and value.net_value > best_value:
                    best_value = value.net_value
                    best_action = value

            if best_action and best_action.net_value > self.voi_threshold:
                sequence.append(best_action)

                # Update state (simplified)
                current_state.gathered_information.append(best_action.source)
                current_state.current_confidence = best_action.confidence
                current_state.uncertainty *= 0.7  # Reduce uncertainty
            else:
                break

        return sequence

    def _record_decision(
        self, state: DecisionState, value: Optional[InformationValue], gather: bool
    ):
        """Record VOI decision for analysis"""

        decision_record = {
            "timestamp": time.time(),
            "uncertainty": state.uncertainty,
            "confidence": state.current_confidence,
            "gather_decision": gather,
            "source": value.source.value if value else None,
            "expected_value": value.expected_value if value else 0,
            "net_value": value.net_value if value else 0,
            "budget_remaining": state.remaining_budget,
        }

        self.decision_history.append(decision_record)

        if value:
            self.voi_calculations.append(value)
            if gather:
                self.total_value_gained += value.expected_value

    def update_with_outcome(
        self,
        source: InformationSource,
        actual_gain: float,
        actual_cost: InformationCost,
    ):
        """
        Update VOI models with actual outcomes

        Args:
            source: Information source used
            actual_gain: Actual information gain achieved
            actual_cost: Actual cost incurred
        """

        # Update cost model
        self.cost_estimator.update_cost(source, actual_cost)

        # Could update gain estimates here as well
        logger.debug(f"VOI outcome: {source.value} gain={actual_gain:.3f}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get VOI statistics"""

        gather_rate = self.gather_decisions / max(1, self.total_decisions)

        recent_decisions = list(self.decision_history)[-100:]
        recent_gather = sum(1 for d in recent_decisions if d["gather_decision"])

        return {
            "total_decisions": self.total_decisions,
            "gather_decisions": self.gather_decisions,
            "proceed_decisions": self.proceed_decisions,
            "gather_rate": gather_rate,
            "total_value_gained": self.total_value_gained,
            "avg_value_per_gather": self.total_value_gained
            / max(1, self.gather_decisions),
            "recent_gather_rate": recent_gather / max(1, len(recent_decisions)),
            "threshold": self.voi_threshold,
            "myopic": self.myopic,
        }

    def visualize_decisions(self) -> Dict[str, Any]:
        """Create visualization data for VOI decisions"""

        if not self.decision_history:
            return {}

        # Extract decision data
        uncertainties = [d["uncertainty"] for d in self.decision_history]
        confidences = [d["confidence"] for d in self.decision_history]
        gather_decisions = [d["gather_decision"] for d in self.decision_history]
        net_values = [d["net_value"] for d in self.decision_history]

        # Source distribution
        source_counts = defaultdict(int)
        for d in self.decision_history:
            if d["source"]:
                source_counts[d["source"]] += 1

        return {
            "uncertainty_distribution": {
                "mean": np.mean(uncertainties),
                "std": np.std(uncertainties),
                "histogram": np.histogram(uncertainties, bins=10)[0].tolist(),
            },
            "confidence_distribution": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "histogram": np.histogram(confidences, bins=10)[0].tolist(),
            },
            "gather_pattern": {
                "gather_when_uncertain": (
                    np.mean(
                        [g for g, u in zip(gather_decisions, uncertainties) if u > 0.5]
                    )
                    if any(u > 0.5 for u in uncertainties)
                    else 0
                ),
                "gather_when_confident": (
                    np.mean(
                        [g for g, c in zip(gather_decisions, confidences) if c > 0.7]
                    )
                    if any(c > 0.7 for c in confidences)
                    else 0
                ),
            },
            "source_distribution": dict(source_counts),
            "value_distribution": {
                "positive_value_rate": sum(1 for v in net_values if v > 0)
                / len(net_values),
                "mean_positive_value": (
                    np.mean([v for v in net_values if v > 0])
                    if any(v > 0 for v in net_values)
                    else 0
                ),
            },
        }

    def save_state(self, path: str):
        """Save VOI gate state"""

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        state = {
            "statistics": self.get_statistics(),
            "decision_history": list(self.decision_history),
            "thresholds": {
                "voi_threshold": self.voi_threshold,
                "max_iterations": self.max_iterations,
            },
        }

        with open(save_path / "voi_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        # Save cost history
        with open(save_path / "cost_history.pkl", "wb") as f:
            pickle.dump(dict(self.cost_estimator.historical_costs), f)

        logger.info(f"VOI state saved to {save_path}")

    def load_state(self, path: str):
        """Load VOI gate state"""

        load_path = Path(path)

        if not load_path.exists():
            logger.warning(f"VOI state path {load_path} not found")
            return

        # Load configuration
        state_file = load_path / "voi_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.voi_threshold = state["thresholds"]["voi_threshold"]
            self.max_iterations = state["thresholds"]["max_iterations"]

            # Restore statistics
            stats = state["statistics"]
            self.total_decisions = stats["total_decisions"]
            self.gather_decisions = stats["gather_decisions"]
            self.proceed_decisions = stats["proceed_decisions"]
            self.total_value_gained = stats["total_value_gained"]

        # Load cost history
        cost_file = load_path / "cost_history.pkl"
        if cost_file.exists():
            self.cost_estimator.historical_costs = defaultdict(
                list, safe_pickle_load(str(cost_file))
            )

        logger.info(f"VOI state loaded from {load_path}")
