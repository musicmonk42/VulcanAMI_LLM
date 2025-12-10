"""
Core reasoning types and data structures

Fixed version with comprehensive validation and error handling.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Import ModalityType - with fallback if config.py doesn't exist
try:
    from ..config import ModalityType
except ImportError:
    # Fallback: Define ModalityType here if config.py is missing
    class ModalityType(Enum):
        """Modality types for multimodal reasoning"""

        TEXT = "text"
        VISION = "vision"
        AUDIO = "audio"
        VIDEO = "video"
        CODE = "code"
        NUMERIC = "numeric"
        GRAPH = "graph"
        TABULAR = "tabular"
        SENSOR = "sensor"
        UNKNOWN = "unknown"


class AbstractReasoner(ABC):
    """Base class for all reasoner implementations"""

    @abstractmethod
    def reason(
        self, problem: Any, context: Optional["ReasoningContext"] = None
    ) -> "ReasoningResult":
        """Execute reasoning on the given problem"""

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return reasoner capabilities"""

    def validate_input(self, problem: Any) -> bool:
        """Validate input problem (optional override)"""
        return True

    def warm_up(self):
        """Warm up reasoner (optional override)"""

    def shutdown(self):
        """Clean shutdown (optional override)"""


class ReasoningType(Enum):
    """Types of reasoning supported."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    COUNTERFACTUAL = "counterfactual"
    SYMBOLIC = "symbolic"
    MULTIMODAL = "multimodal"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"
    ABSTRACT = "abstract"
    HIERARCHICAL = "hierarchical"  # FINAL FIX: Added missing enum member
    UNKNOWN = "unknown"


class SelectionMode(Enum):
    """Tool selection modes for optimization"""

    FAST = "fast"  # Optimize for speed
    ACCURATE = "accurate"  # Optimize for accuracy
    EFFICIENT = "efficient"  # Optimize for energy
    BALANCED = "balanced"  # Balance all factors
    SAFE = "safe"  # Maximum safety checks


class PortfolioStrategy(Enum):
    """Portfolio execution strategies"""

    SEQUENTIAL = "sequential"  # Execute tools in sequence
    PARALLEL = "parallel"  # Execute tools in parallel
    SPECULATIVE_PARALLEL = "speculative_parallel"  # Speculative parallel execution
    CASCADE = "cascade"  # Cascade with early stopping
    COMMITTEE_CONSENSUS = "committee_consensus"  # Committee voting
    SEQUENTIAL_REFINEMENT = "sequential_refinement"  # Iterative refinement
    HEDGE = "hedge"  # Hedging strategy
    ADAPTIVE = "adaptive"  # Adaptive selection


class UtilityContext(Enum):
    """Context modes for utility calculation"""

    RUSH = "rush"  # Prioritize speed
    ACCURATE = "accurate"  # Prioritize quality
    EFFICIENT = "efficient"  # Prioritize energy efficiency
    BALANCED = "balanced"  # Balance all factors
    EXPLORATORY = "exploratory"  # Exploration/learning mode
    CONSERVATIVE = "conservative"  # Risk-averse mode


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain"""

    step_id: str
    step_type: ReasoningType
    input_data: Any
    output_data: Any
    confidence: float
    explanation: str
    modality: Optional[ModalityType] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate reasoning step data"""
        # Validate confidence is in valid range [0, 1]
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"Confidence must be numeric, got {type(self.confidence)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        # Validate step_id is not empty
        if not self.step_id or not isinstance(self.step_id, str):
            raise ValueError("step_id must be a non-empty string")

        # Validate step_type is ReasoningType enum
        if not isinstance(self.step_type, ReasoningType):
            raise TypeError(
                f"step_type must be ReasoningType enum, got {type(self.step_type)}"
            )

        # Validate timestamp is reasonable
        if (
            self.timestamp < 0 or self.timestamp > time.time() + 86400
        ):  # Not more than 1 day in future
            raise ValueError(f"Invalid timestamp: {self.timestamp}")


@dataclass
class ReasoningChain:
    """Complete reasoning chain with audit trail"""

    chain_id: str
    steps: List[ReasoningStep]
    initial_query: Dict[str, Any]
    final_conclusion: Any
    total_confidence: float
    reasoning_types_used: Set[ReasoningType]
    modalities_involved: Set[ModalityType]
    safety_checks: List[Dict[str, Any]]
    audit_trail: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate reasoning chain"""
        # Validate chain_id
        if not self.chain_id or not isinstance(self.chain_id, str):
            raise ValueError("chain_id must be a non-empty string")

        # Validate confidence range
        if not isinstance(self.total_confidence, (int, float)):
            raise TypeError(
                f"total_confidence must be numeric, got {type(self.total_confidence)}"
            )

        if not 0.0 <= self.total_confidence <= 1.0:
            raise ValueError(
                f"total_confidence must be in [0, 1], got {self.total_confidence}"
            )

        # Validate steps list is not empty
        if not self.steps:
            raise ValueError("ReasoningChain must have at least one step")

        if not isinstance(self.steps, list):
            raise TypeError("steps must be a list")

        # Validate all steps are ReasoningStep instances
        for i, step in enumerate(self.steps):
            if not isinstance(step, ReasoningStep):
                raise TypeError(f"Step {i} must be a ReasoningStep instance")

        # Validate reasoning_types_used is a set
        if not isinstance(self.reasoning_types_used, set):
            raise TypeError("reasoning_types_used must be a set")

        # Validate modalities_involved is a set
        if not isinstance(self.modalities_involved, set):
            raise TypeError("modalities_involved must be a set")

        # Validate safety_checks and audit_trail are lists
        if not isinstance(self.safety_checks, list):
            raise TypeError("safety_checks must be a list")

        if not isinstance(self.audit_trail, list):
            raise TypeError("audit_trail must be a list")


@dataclass
class ReasoningResult:
    """Enhanced result from a reasoning process."""

    conclusion: Any
    confidence: float
    reasoning_type: ReasoningType
    evidence: List[Any] = field(default_factory=list)
    explanation: str = ""
    uncertainty: float = 0.0
    reasoning_chain: Optional[ReasoningChain] = None
    safety_status: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate reasoning result"""
        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(self.confidence)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        # Validate uncertainty
        if not isinstance(self.uncertainty, (int, float)):
            raise TypeError(
                f"uncertainty must be numeric, got {type(self.uncertainty)}"
            )

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(f"uncertainty must be in [0, 1], got {self.uncertainty}")

        # Validate reasoning_type
        if not isinstance(self.reasoning_type, ReasoningType):
            raise TypeError(
                f"reasoning_type must be ReasoningType enum, got {type(self.reasoning_type)}"
            )

        # Validate evidence is a list
        if not isinstance(self.evidence, list):
            raise TypeError("evidence must be a list")

        # Validate optional reasoning_chain
        if self.reasoning_chain is not None and not isinstance()
            self.reasoning_chain, ReasoningChain
        ):
            raise TypeError("reasoning_chain must be a ReasoningChain instance or None")


@dataclass
class SelectionResult:
    """Result from tool selection process"""

    selected_tool: str
    execution_result: Any
    confidence: float
    calibrated_confidence: float
    execution_time_ms: float
    energy_used_mj: float
    strategy_used: PortfolioStrategy
    all_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    utility_score: Optional[float] = None
    cost_breakdown: Optional[Dict[str, float]] = None
    voi_analysis: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate selection result"""
        # Validate selected_tool
        if not self.selected_tool or not isinstance(self.selected_tool, str):
            raise ValueError("selected_tool must be a non-empty string")

        # Validate confidence values
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(self.confidence)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        if not isinstance(self.calibrated_confidence, (int, float)):
            raise TypeError(
                f"calibrated_confidence must be numeric, got {type(self.calibrated_confidence)}"
            )

        if not 0.0 <= self.calibrated_confidence <= 1.0:
            raise ValueError(
                f"calibrated_confidence must be in [0, 1], got {self.calibrated_confidence}"
            )

        # Validate execution metrics are non-negative
        if self.execution_time_ms < 0:
            raise ValueError(
                f"execution_time_ms must be non-negative, got {self.execution_time_ms}"
            )

        if self.energy_used_mj < 0:
            raise ValueError(
                f"energy_used_mj must be non-negative, got {self.energy_used_mj}"
            )

        # Validate strategy
        if not isinstance(self.strategy_used, PortfolioStrategy):
            raise TypeError(
                f"strategy_used must be PortfolioStrategy enum, got {type(self.strategy_used)}"
            )

        # Validate all_results is a dict
        if not isinstance(self.all_results, dict):
            raise TypeError("all_results must be a dictionary")

        # Validate optional utility_score
        if self.utility_score is not None:
            if not isinstance(self.utility_score, (int, float)):
                raise TypeError("utility_score must be numeric or None")


@dataclass
class PortfolioResult:
    """Result from portfolio execution"""

    primary_result: Any
    all_results: Dict[str, Any]
    strategy: PortfolioStrategy
    tools_used: List[str]
    execution_time_ms: float
    energy_used: float
    confidence_scores: Dict[str, float]
    consensus_achieved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate portfolio result"""
        # Validate strategy
        if not isinstance(self.strategy, PortfolioStrategy):
            raise TypeError(
                f"strategy must be PortfolioStrategy enum, got {type(self.strategy)}"
            )

        # Validate all_results is a dict
        if not isinstance(self.all_results, dict):
            raise TypeError("all_results must be a dictionary")

        # Validate tools_used is a list
        if not isinstance(self.tools_used, list):
            raise TypeError("tools_used must be a list")

        # Validate execution metrics are non-negative
        if self.execution_time_ms < 0:
            raise ValueError(
                f"execution_time_ms must be non-negative, got {self.execution_time_ms}"
            )

        if self.energy_used < 0:
            raise ValueError(
                f"energy_used must be non-negative, got {self.energy_used}"
            )

        # Validate confidence_scores
        if not isinstance(self.confidence_scores, dict):
            raise TypeError("confidence_scores must be a dictionary")

        for tool, score in self.confidence_scores.items():
            if not isinstance(score, (int, float)):
                raise TypeError(f"Confidence score for {tool} must be numeric")
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Confidence score for {tool} must be in [0, 1], got {score}"
                )

        # Validate consensus_achieved is boolean
        if not isinstance(self.consensus_achieved, bool):
            raise TypeError("consensus_achieved must be a boolean")


@dataclass
class CostEstimate:
    """Estimated costs for execution"""

    time_ms: float
    energy_mj: float
    memory_mb: float
    confidence_interval: Tuple[float, float]
    percentiles: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cost estimate"""
        # Validate all costs are non-negative
        if self.time_ms < 0:
            raise ValueError(f"time_ms must be non-negative, got {self.time_ms}")

        if self.energy_mj < 0:
            raise ValueError(f"energy_mj must be non-negative, got {self.energy_mj}")

        if self.memory_mb < 0:
            raise ValueError(f"memory_mb must be non-negative, got {self.memory_mb}")

        # Validate confidence_interval
        if (
            not isinstance(self.confidence_interval, tuple)
            or len(self.confidence_interval) != 2
        ):
            raise ValueError("confidence_interval must be a tuple of length 2")

        lower, upper = self.confidence_interval
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise TypeError("confidence_interval values must be numeric")

        if lower < 0 or upper < 0:
            raise ValueError("confidence_interval values must be non-negative")

        if lower > upper:
            raise ValueError(
                f"confidence_interval lower bound ({lower}) must be <= upper bound ({upper})"
            )

        # Validate percentiles
        if not isinstance(self.percentiles, dict):
            raise TypeError("percentiles must be a dictionary")

        for percentile, value in self.percentiles.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Percentile value for {percentile} must be numeric")
            if value < 0:
                raise ValueError(
                    f"Percentile value for {percentile} must be non-negative"
                )


@dataclass
class SafetyAssessment:
    """Safety assessment result"""

    is_safe: bool
    safety_level: str
    violations: List[str]
    mitigations: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate safety assessment"""
        # Validate is_safe is boolean
        if not isinstance(self.is_safe, bool):
            raise TypeError("is_safe must be a boolean")

        # Validate safety_level
        if not self.safety_level or not isinstance(self.safety_level, str):
            raise ValueError("safety_level must be a non-empty string")

        valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL", "UNKNOWN"}
        if self.safety_level.upper() not in valid_levels:
            raise ValueError(
                f"safety_level must be one of {valid_levels}, got {self.safety_level}"
            )

        # Validate lists
        if not isinstance(self.violations, list):
            raise TypeError("violations must be a list")

        if not isinstance(self.mitigations, list):
            raise TypeError("mitigations must be a list")

        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(self.confidence)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class CalibrationData:
    """Data for confidence calibration"""

    raw_confidence: float
    calibrated_confidence: float
    actual_outcome: bool
    tool_name: str
    timestamp: float = field(default_factory=time.time)
    features: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate calibration data"""
        # Validate confidence values
        if not isinstance(self.raw_confidence, (int, float)):
            raise TypeError(
                f"raw_confidence must be numeric, got {type(self.raw_confidence)}"
            )

        if not 0.0 <= self.raw_confidence <= 1.0:
            raise ValueError(
                f"raw_confidence must be in [0, 1], got {self.raw_confidence}"
            )

        if not isinstance(self.calibrated_confidence, (int, float)):
            raise TypeError(
                f"calibrated_confidence must be numeric, got {type(self.calibrated_confidence)}"
            )

        if not 0.0 <= self.calibrated_confidence <= 1.0:
            raise ValueError(
                f"calibrated_confidence must be in [0, 1], got {self.calibrated_confidence}"
            )

        # Validate actual_outcome is boolean
        if not isinstance(self.actual_outcome, bool):
            raise TypeError("actual_outcome must be a boolean")

        # Validate tool_name
        if not self.tool_name or not isinstance(self.tool_name, str):
            raise ValueError("tool_name must be a non-empty string")

        # Validate timestamp
        if self.timestamp < 0 or self.timestamp > time.time() + 86400:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")

        # Validate optional features
        if self.features is not None:
            if not isinstance(self.features, list):
                raise TypeError("features must be a list or None")
            for i, feature in enumerate(self.features):
                if not isinstance(feature, (int, float)):
                    raise TypeError(f"Feature {i} must be numeric")


@dataclass
class MonitoringData:
    """Data for system monitoring"""

    tool_name: str
    latency_ms: float
    throughput: float
    error_rate: float
    resource_usage: Dict[str, float]
    health_score: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate monitoring data"""
        # Validate tool_name
        if not self.tool_name or not isinstance(self.tool_name, str):
            raise ValueError("tool_name must be a non-empty string")

        # Validate latency_ms is non-negative
        if not isinstance(self.latency_ms, (int, float)):
            raise TypeError("latency_ms must be numeric")

        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")

        # Validate throughput is non-negative
        if not isinstance(self.throughput, (int, float)):
            raise TypeError("throughput must be numeric")

        if self.throughput < 0:
            raise ValueError(f"throughput must be non-negative, got {self.throughput}")

        # Validate error_rate is in [0, 1]
        if not isinstance(self.error_rate, (int, float)):
            raise TypeError("error_rate must be numeric")

        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {self.error_rate}")

        # Validate health_score is in [0, 1]
        if not isinstance(self.health_score, (int, float)):
            raise TypeError("health_score must be numeric")

        if not 0.0 <= self.health_score <= 1.0:
            raise ValueError(f"health_score must be in [0, 1], got {self.health_score}")

        # Validate resource_usage
        if not isinstance(self.resource_usage, dict):
            raise TypeError("resource_usage must be a dictionary")

        for resource, usage in self.resource_usage.items():
            if not isinstance(usage, (int, float)):
                raise TypeError(f"Resource usage for {resource} must be numeric")
            if usage < 0:
                raise ValueError(f"Resource usage for {resource} must be non-negative")

        # Validate timestamp
        if self.timestamp < 0 or self.timestamp > time.time() + 86400:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")


@dataclass
class ValueOfInformation:
    """Value of information analysis result"""

    expected_value: float
    information_gain: float
    cost: float
    net_value: float
    recommendation: str
    source: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate VOI data"""
        # Validate all numeric fields
        for field_name in ["expected_value", "information_gain", "cost", "net_value"]:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric, got {type(value)}")

        # Validate cost is non-negative
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")

        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(self.confidence)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        # Validate string fields
        if not self.recommendation or not isinstance(self.recommendation, str):
            raise ValueError("recommendation must be a non-empty string")

        if not self.source or not isinstance(self.source, str):
            raise ValueError("source must be a non-empty string")


@dataclass
class DistributionShift:
    """Distribution shift detection result"""

    drift_detected: bool
    drift_type: str
    severity: str
    affected_features: List[int]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate distribution shift data"""
        # Validate drift_detected is boolean
        if not isinstance(self.drift_detected, bool):
            raise TypeError("drift_detected must be a boolean")

        # Validate drift_type
        if not self.drift_type or not isinstance(self.drift_type, str):
            raise ValueError("drift_type must be a non-empty string")

        valid_drift_types = {"SUDDEN", "GRADUAL", "INCREMENTAL", "RECURRING", "NONE"}
        if self.drift_type.upper() not in valid_drift_types:
            raise ValueError(
                f"drift_type must be one of {valid_drift_types}, got {self.drift_type}"
            )

        # Validate severity
        if not self.severity or not isinstance(self.severity, str):
            raise ValueError("severity must be a non-empty string")

        valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"}
        if self.severity.upper() not in valid_severities:
            raise ValueError(
                f"severity must be one of {valid_severities}, got {self.severity}"
            )

        # Validate affected_features is a list of integers
        if not isinstance(self.affected_features, list):
            raise TypeError("affected_features must be a list")

        for i, feature_idx in enumerate(self.affected_features):
            if not isinstance(feature_idx, int):
                raise TypeError(
                    f"affected_features[{i}] must be an integer, got {type(feature_idx)}"
                )
            if feature_idx < 0:
                raise ValueError(
                    f"affected_features[{i}] must be non-negative, got {feature_idx}"
                )

        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(self.confidence)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        # Validate timestamp
        if self.timestamp < 0 or self.timestamp > time.time() + 86400:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")
