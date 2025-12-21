"""
gap_analyzer.py - Knowledge gap analysis for Curiosity Engine
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
"""

import hashlib
import logging
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Optional imports with fallbacks
try:
    from scipy import stats
    from scipy.stats import zscore

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, some statistical features disabled")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, anomaly detection features limited")

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of knowledge gaps"""

    DECOMPOSITION = "decomposition"
    CAUSAL = "causal"
    SEMANTIC_BRIDGE = "semantic_bridge"
    LATENT = "latent"
    TRANSFER = "transfer"
    PREDICTION = "prediction"
    PATTERN = "pattern"
    SEMANTIC = "semantic"
    CORRELATION = "correlation"
    DOMAIN_BRIDGE = "domain_bridge"
    EXPLORATION = "exploration"


@dataclass
class Pattern:
    """Pattern representation for gap analysis"""

    pattern_id: str
    pattern_type: str
    frequency: float
    components: List[Any] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Ensure pattern has an ID"""
        if not self.pattern_id:
            self.pattern_id = f"pattern_{self.pattern_type}_{int(self.created_at)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "components": self.components,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def similarity_to(self, other: "Pattern") -> float:
        """Calculate similarity to another pattern"""
        try:
            if self.pattern_type != other.pattern_type:
                return 0.0

            # Compare components
            if self.components and other.components:
                self_set = set(map(str, self.components))
                other_set = set(map(str, other.components))

                # FIX: Check for empty sets
                if not self_set and not other_set:
                    return 1.0

                if self_set or other_set:
                    intersection = len(self_set & other_set)
                    union = len(self_set | other_set)

                    # FIX: Check division by zero
                    jaccard = intersection / union if union > 0 else 0.0

                    # Weight by confidence
                    similarity = jaccard * (self.confidence + other.confidence) / 2
                    return min(1.0, similarity)

            return 0.0
        except Exception as e:
            logger.warning("Error calculating similarity: %s", e)
            return 0.0


@dataclass
class KnowledgeGap:
    """Single knowledge gap representation"""

    type: str  # Gap type
    domain: str
    priority: float
    estimated_cost: float
    missing_capability: Optional[str] = None
    gap_id: Optional[str] = None
    id: Optional[str] = None  # Alias for gap_id
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    addressed: bool = False
    complexity: float = 0.5
    dependencies: List[str] = field(default_factory=list)
    adjusted_roi: Optional[float] = None

    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.gap_id and not self.id:
            # Generate unique ID
            content = f"{self.type}_{self.domain}_{self.timestamp}"
            hash_val = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[
                :8
            ]
            self.gap_id = f"{self.type}_{hash_val}"
            self.id = self.gap_id
        elif self.gap_id and not self.id:
            self.id = self.gap_id
        elif self.id and not self.gap_id:
            self.gap_id = self.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "gap_id": self.gap_id,
            "id": self.id,
            "type": self.type,
            "domain": self.domain,
            "priority": self.priority,
            "estimated_cost": self.estimated_cost,
            "missing_capability": self.missing_capability,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "addressed": self.addressed,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "adjusted_roi": self.adjusted_roi,
        }

    def mark_addressed(self):
        """Mark gap as addressed"""
        self.addressed = True

    def __hash__(self):
        """Make gap hashable"""
        return hash(self.gap_id)

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, KnowledgeGap):
            return False
        return self.gap_id == other.gap_id


@dataclass
class LatentGap(KnowledgeGap):
    """Gap discovered through anomaly detection"""

    pattern: Optional[Pattern] = None
    frequency: float = 0.0
    impact: float = 0.0
    detection_confidence: float = 0.5
    anomaly_score: float = 0.0

    def __post_init__(self):
        """Initialize latent gap"""
        super().__post_init__()
        self.type = "latent"

        # Calculate priority based on impact and frequency
        if self.impact > 0 and self.frequency > 0:
            self.priority = self.impact * self.frequency * self.detection_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "pattern": self.pattern.to_dict() if self.pattern else None,
                "frequency": self.frequency,
                "impact": self.impact,
                "detection_confidence": self.detection_confidence,
                "anomaly_score": self.anomaly_score,
            }
        )
        return base_dict


class SimpleAnomalyDetector:
    """Simple anomaly detector for when sklearn is not available"""

    def __init__(self, contamination: float = 0.1):
        """
        Initialize simple anomaly detector

        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.threshold = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """Fit the detector"""
        try:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-10

            # Calculate distances
            distances = np.sum(((X - self.mean) / self.std) ** 2, axis=1)

            # Set threshold based on contamination
            sorted_distances = np.sort(distances)
            threshold_idx = int(len(distances) * (1 - self.contamination))
            self.threshold = (
                sorted_distances[threshold_idx]
                if threshold_idx < len(sorted_distances)
                else np.inf
            )

            return self
        except Exception as e:
            logger.error("Error fitting detector: %s", e)
            return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        try:
            if self.mean is None:
                return np.ones(len(X))

            distances = np.sum(((X - self.mean) / self.std) ** 2, axis=1)
            predictions = np.ones(len(X))
            predictions[distances > self.threshold] = -1

            return predictions
        except Exception as e:
            logger.error("Error predicting: %s", e)
            return np.ones(len(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (negative for anomalies)"""
        try:
            if self.mean is None:
                return np.zeros(len(X))

            distances = np.sum(((X - self.mean) / self.std) ** 2, axis=1)
            # Convert to negative scores (more negative = more anomalous)
            return -distances
        except Exception as e:
            logger.error("Error scoring samples: %s", e)
            return np.zeros(len(X))


class FailureTracker:
    """Tracks different types of failures - SEPARATED CONCERN"""

    def __init__(self, max_history: int = 1000):
        self.decomposition_failures = deque(maxlen=max_history)
        self.prediction_errors = deque(maxlen=max_history)
        self.transfer_failures = deque(maxlen=max_history)
        self.lock = threading.RLock()

    def record_failure(self, failure_type: str, failure_data: Dict[str, Any]):
        """Record a failure for analysis"""
        with self.lock:
            try:
                failure_data["timestamp"] = time.time()
                failure_data["type"] = failure_type

                if failure_type == "decomposition":
                    self.decomposition_failures.append(failure_data)
                elif failure_type == "prediction":
                    self.prediction_errors.append(failure_data)
                elif failure_type == "transfer":
                    self.transfer_failures.append(failure_data)
            except Exception as e:
                logger.error("Error recording failure: %s", e)

    def get_decomposition_failures(self) -> List[Dict[str, Any]]:
        """Get decomposition failures"""
        with self.lock:
            return list(self.decomposition_failures)

    def get_prediction_errors(self) -> List[Dict[str, Any]]:
        """Get prediction errors"""
        with self.lock:
            return list(self.prediction_errors)

    def get_transfer_failures(self) -> List[Dict[str, Any]]:
        """Get transfer failures"""
        with self.lock:
            return list(self.transfer_failures)

    def get_statistics(self) -> Dict[str, int]:
        """Get failure statistics"""
        with self.lock:
            return {
                "decomposition_failures": len(self.decomposition_failures),
                "prediction_errors": len(self.prediction_errors),
                "transfer_failures": len(self.transfer_failures),
            }


class PatternTracker:
    """Tracks patterns for analysis - SEPARATED CONCERN"""

    def __init__(self, max_history: int = 1000):
        # FIX: Use regular dict instead of defaultdict for thread safety
        self.observed_patterns = {}
        self.max_history = max_history
        self.lock = threading.RLock()

    def _get_pattern_deque(self, pattern_key: str) -> deque:
        """Thread-safe pattern deque retrieval"""
        with self.lock:
            if pattern_key not in self.observed_patterns:
                self.observed_patterns[pattern_key] = deque(maxlen=self.max_history)
            return self.observed_patterns[pattern_key]

    def record_pattern(self, pattern_key: str, observation: Any):
        """Record a pattern observation"""
        with self.lock:
            try:
                pattern_deque = self._get_pattern_deque(pattern_key)
                pattern_deque.append(observation)
            except Exception as e:
                logger.error("Error recording pattern: %s", e)

    def get_patterns(self, pattern_type: Optional[str] = None) -> Dict[str, List[Any]]:
        """Get observed patterns"""
        with self.lock:
            try:
                if pattern_type:
                    if pattern_type in self.observed_patterns:
                        return {
                            pattern_type: list(self.observed_patterns[pattern_type])
                        }
                    return {}

                return {k: list(v) for k, v in self.observed_patterns.items()}
            except Exception as e:
                logger.error("Error getting patterns: %s", e)
                return {}

    def get_pattern_count(self) -> int:
        """Get number of pattern types"""
        with self.lock:
            return len(self.observed_patterns)


class GapRegistry:
    """Manages gap registration and tracking - SEPARATED CONCERN"""

    def __init__(self, max_history: int = 10000):
        self.identified_gaps = {}
        self.gap_history = deque(maxlen=max_history)
        self.total_gaps_found = 0
        # FIX: Use regular dict instead of defaultdict
        self.gaps_by_type = {}
        self.gap_success_rate = {}
        self.lock = threading.RLock()

    def _get_type_count(self, gap_type: str) -> int:
        """Thread-safe type count retrieval"""
        with self.lock:
            return self.gaps_by_type.get(gap_type, 0)

    def _increment_type_count(self, gap_type: str):
        """Thread-safe type count increment"""
        with self.lock:
            self.gaps_by_type[gap_type] = self._get_type_count(gap_type) + 1

    def _get_success_stats(self, gap_type: str) -> Dict[str, int]:
        """Thread-safe success stats retrieval"""
        with self.lock:
            if gap_type not in self.gap_success_rate:
                self.gap_success_rate[gap_type] = {"success": 0, "total": 0}
            return self.gap_success_rate[gap_type]

    def register_gap(self, gap: KnowledgeGap) -> bool:
        """Register a discovered gap"""
        with self.lock:
            try:
                if gap.gap_id not in self.identified_gaps:
                    self.identified_gaps[gap.gap_id] = gap
                    self.gap_history.append(gap.to_dict())
                    self.total_gaps_found += 1
                    self._increment_type_count(gap.type)
                    return True
                return False
            except Exception as e:
                logger.error("Error registering gap: %s", e)
                return False

    def update_gap_success(self, gap_id: str, success: bool):
        """Update success tracking for a gap"""
        with self.lock:
            try:
                if gap_id in self.identified_gaps:
                    gap = self.identified_gaps[gap_id]
                    stats = self._get_success_stats(gap.type)
                    stats["total"] += 1
                    if success:
                        stats["success"] += 1
                        gap.mark_addressed()
            except Exception as e:
                logger.error("Error updating gap success: %s", e)

    def get_gaps(self, addressed: Optional[bool] = None) -> List[KnowledgeGap]:
        """Get gaps with optional filter"""
        with self.lock:
            try:
                if addressed is None:
                    return list(self.identified_gaps.values())

                return [
                    g for g in self.identified_gaps.values() if g.addressed == addressed
                ]
            except Exception as e:
                logger.error("Error getting gaps: %s", e)
                return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self.lock:
            try:
                success_rates = {}
                # FIX: Copy items before iterating
                success_items = list(self.gap_success_rate.items())
                for gap_type, stats in success_items:
                    if stats["total"] > 0:
                        success_rates[gap_type] = stats["success"] / stats["total"]

                return {
                    "total_gaps_found": self.total_gaps_found,
                    "gaps_by_type": dict(self.gaps_by_type),
                    "active_gaps": len(
                        [g for g in self.identified_gaps.values() if not g.addressed]
                    ),
                    "addressed_gaps": len(
                        [g for g in self.identified_gaps.values() if g.addressed]
                    ),
                    "success_rates": success_rates,
                }
            except Exception as e:
                logger.error("Error getting statistics: %s", e)
                return {}


class DecompositionAnalyzer:
    """Analyzes decomposition failures - SEPARATED CONCERN"""

    def __init__(self, min_frequency: float = 0.1):
        self.min_frequency = min_frequency

    def analyze_failures(self, failures: List[Dict[str, Any]]) -> List[KnowledgeGap]:
        """Analyze decomposition failures"""
        gaps = []

        try:
            # FIX: Validate input
            if not failures or not isinstance(failures, list):
                return gaps

            # EXAMINE: Group failures by pattern
            failure_patterns = {}

            for failure in failures:
                if isinstance(failure, dict):
                    pattern_key = failure.get("pattern", "unknown")
                    if pattern_key not in failure_patterns:
                        failure_patterns[pattern_key] = []
                    failure_patterns[pattern_key].append(failure)

            # SELECT & APPLY: Create gaps for frequent patterns
            total_failures = max(1, len(failures))

            for pattern_key, pattern_failures in failure_patterns.items():
                # FIX: Check division by zero
                frequency = (
                    len(pattern_failures) / total_failures if total_failures > 0 else 0
                )

                if frequency >= self.min_frequency:
                    # Extract common properties
                    domains = set()
                    complexities = []
                    missing_concepts = set()

                    for failure in pattern_failures:
                        if "domain" in failure:
                            domains.add(failure["domain"])
                        if "complexity" in failure:
                            complexities.append(failure["complexity"])
                        if "missing_concepts" in failure:
                            missing_concepts.update(failure["missing_concepts"])

                    # Create gap for each domain
                    for domain in domains or {"unknown"}:
                        gap = KnowledgeGap(
                            type="decomposition",
                            domain=domain,
                            priority=min(1.0, frequency * 0.8),
                            estimated_cost=(
                                20 * np.mean(complexities) if complexities else 20
                            ),
                            missing_capability=f"decompose_{pattern_key}",
                            complexity=np.mean(complexities) if complexities else 0.5,
                            metadata={
                                "pattern": pattern_key,
                                "failure_count": len(pattern_failures),
                                "frequency": frequency,
                                "missing_concepts": list(missing_concepts),
                            },
                        )

                        gaps.append(gap)

            # REMEMBER: Add structural analysis
            structural_gaps = self._analyze_structural_failures(failures)
            gaps.extend(structural_gaps)

        except Exception as e:
            logger.error("Error analyzing failures: %s", e)

        return gaps

    def _analyze_structural_failures(
        self, failures: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """Analyze structural decomposition failures"""
        gaps = []

        try:
            # Look for patterns in failed structures
            structure_types = {}
            structure_domains = {}

            for failure in failures:
                if "structure" in failure:
                    structure = failure["structure"]
                    structure_types[structure] = structure_types.get(structure, 0) + 1

                    if structure not in structure_domains:
                        structure_domains[structure] = set()
                    if "domain" in failure:
                        structure_domains[structure].add(failure["domain"])

            # Create gaps for problematic structures
            for structure, count in structure_types.items():
                if count >= 5:  # Threshold for structural issue
                    domains = structure_domains.get(structure, {"structural"})

                    for domain in domains:
                        gap = KnowledgeGap(
                            type="decomposition",
                            domain=domain,
                            priority=0.6 + min(0.3, count / 20),
                            estimated_cost=15,
                            missing_capability=f"handle_{structure}",
                            metadata={
                                "structure_type": structure,
                                "failure_count": count,
                            },
                        )
                        gaps.append(gap)

        except Exception as e:
            logger.error("Error analyzing structural failures: %s", e)

        return gaps


class PredictionAnalyzer:
    """Analyzes prediction errors - SEPARATED CONCERN"""

    def analyze_errors(self, errors: List[Dict[str, Any]]) -> List[KnowledgeGap]:
        """Analyze prediction errors"""
        gaps = []

        try:
            # FIX: Validate input
            if not errors or not isinstance(errors, list):
                return gaps

            # EXAMINE: Group errors by variable pairs
            error_patterns = {}

            for error in errors:
                if isinstance(error, dict):
                    cause = error.get("cause", "unknown")
                    effect = error.get("effect", "unknown")
                    pair_key = f"{cause}->{effect}"
                    if pair_key not in error_patterns:
                        error_patterns[pair_key] = []
                    error_patterns[pair_key].append(error)

            # SELECT & APPLY: Create gaps for high-error pairs
            for pair_key, pair_errors in error_patterns.items():
                if len(pair_errors) < 3:  # Need minimum samples
                    continue

                # Calculate error statistics
                error_magnitudes = []
                variables = set()

                for error in pair_errors:
                    if "magnitude" in error:
                        error_magnitudes.append(error["magnitude"])
                    if "variables" in error:
                        variables.update(error["variables"])

                if error_magnitudes:
                    avg_error = np.mean(error_magnitudes)
                    std_error = (
                        np.std(error_magnitudes) if len(error_magnitudes) > 1 else 0
                    )

                    # Create gap if error is significant
                    if avg_error > 0.3:  # 30% error threshold
                        parts = pair_key.split("->")

                        gap = KnowledgeGap(
                            type="causal",
                            domain=pair_errors[0].get("domain", "unknown"),
                            priority=min(1.0, avg_error),
                            estimated_cost=30,  # Causal discovery is expensive
                            missing_capability=f"causal_relation_{pair_key}",
                            metadata={
                                "cause": parts[0] if len(parts) > 0 else "unknown",
                                "effect": parts[1] if len(parts) > 1 else "unknown",
                                "variables": list(variables),
                                "avg_error": avg_error,
                                "std_error": std_error,
                                "sample_count": len(pair_errors),
                            },
                        )

                        gaps.append(gap)

            # REMEMBER: Add systematic error analysis
            systematic_gaps = self._analyze_systematic_errors(errors)
            gaps.extend(systematic_gaps)

        except Exception as e:
            logger.error("Error analyzing prediction errors: %s", e)

        return gaps

    def _analyze_systematic_errors(
        self, errors: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """Analyze systematic prediction errors"""
        gaps = []

        try:
            if len(errors) >= 20:
                recent_errors = errors[-100:]

                # Check for consistent over/under prediction
                signed_errors = []
                domains = set()

                for error in recent_errors:
                    if "signed_error" in error:
                        signed_errors.append(error["signed_error"])
                    if "domain" in error:
                        domains.add(error["domain"])

                if signed_errors:
                    mean_error = np.mean(signed_errors)
                    std_error = np.std(signed_errors) if len(signed_errors) > 1 else 0

                    if abs(mean_error) > 0.2:
                        for domain in domains or {"systematic_bias"}:
                            gap = KnowledgeGap(
                                type="causal",
                                domain=domain,
                                priority=0.7,
                                estimated_cost=20,
                                missing_capability="bias_correction",
                                metadata={
                                    "mean_error": mean_error,
                                    "std_error": std_error,
                                    "error_type": (
                                        "over_prediction"
                                        if mean_error > 0
                                        else "under_prediction"
                                    ),
                                    "sample_count": len(signed_errors),
                                },
                            )
                            gaps.append(gap)

        except Exception as e:
            logger.error("Error analyzing systematic errors: %s", e)

        return gaps


class TransferAnalyzer:
    """Analyzes transfer failures - SEPARATED CONCERN"""

    def __init__(self, min_frequency: float = 0.1):
        self.min_frequency = min_frequency

    def analyze_failures(self, failures: List[Dict[str, Any]]) -> List[KnowledgeGap]:
        """Analyze transfer failures"""
        gaps = []

        try:
            # FIX: Validate input
            if not failures or not isinstance(failures, list):
                return gaps

            # EXAMINE: Group failures by domain pairs
            transfer_patterns = {}

            for failure in failures:
                if isinstance(failure, dict):
                    source = failure.get("source_domain", "unknown")
                    target = failure.get("target_domain", "unknown")
                    pair_key = f"{source}_to_{target}"
                    if pair_key not in transfer_patterns:
                        transfer_patterns[pair_key] = []
                    transfer_patterns[pair_key].append(failure)

            # SELECT & APPLY: Create gaps for failed transfers
            total_failures = max(1, len(failures))

            for pair_key, pair_failures in transfer_patterns.items():
                # FIX: Check division by zero
                frequency = (
                    len(pair_failures) / total_failures if total_failures > 0 else 0
                )

                if frequency >= self.min_frequency:
                    parts = pair_key.split("_to_")

                    # Calculate transfer difficulty
                    success_rates = []
                    for failure in pair_failures:
                        if "success_rate" in failure:
                            success_rates.append(failure["success_rate"])

                    avg_success = np.mean(success_rates) if success_rates else 0.0

                    gap = KnowledgeGap(
                        type="transfer",
                        domain=f"{parts[0]}_transfer",
                        priority=1.0 - avg_success,
                        estimated_cost=25,
                        missing_capability=f"bridge_{pair_key}",
                        metadata={
                            "source_domain": parts[0] if len(parts) > 0 else "unknown",
                            "target_domain": parts[1] if len(parts) > 1 else "unknown",
                            "failure_count": len(pair_failures),
                            "avg_success_rate": avg_success,
                            "frequency": frequency,
                        },
                    )

                    gaps.append(gap)

            # REMEMBER: Add domain boundary analysis
            boundary_gaps = self._analyze_domain_boundaries(failures)
            gaps.extend(boundary_gaps)

        except Exception as e:
            logger.error("Error analyzing transfer failures: %s", e)

        return gaps

    def _analyze_domain_boundaries(
        self, failures: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """Analyze domain boundary issues"""
        gaps = []

        try:
            # Look for domain pairs with high failure rates
            domain_pairs = {}

            for failure in failures:
                if "source_domain" in failure and "target_domain" in failure:
                    pair = (failure["source_domain"], failure["target_domain"])
                    if pair not in domain_pairs:
                        domain_pairs[pair] = {"success": 0, "failure": 0}

                    if failure.get("success", False):
                        domain_pairs[pair]["success"] += 1
                    else:
                        domain_pairs[pair]["failure"] += 1

            # Create gaps for problematic boundaries
            for (source, target), stats in domain_pairs.items():
                total = stats["success"] + stats["failure"]
                if total >= 5:
                    # FIX: Check division by zero
                    failure_rate = stats["failure"] / total if total > 0 else 0
                    if failure_rate > 0.6:
                        gap = KnowledgeGap(
                            type="semantic_bridge",
                            domain=f"{source}_{target}_boundary",
                            priority=failure_rate,
                            estimated_cost=25,
                            missing_capability=f"bridge_{source}_{target}",
                            metadata={
                                "source_domain": source,
                                "target_domain": target,
                                "failure_rate": failure_rate,
                                "attempts": total,
                            },
                        )
                        gaps.append(gap)

        except Exception as e:
            logger.error("Error analyzing domain boundaries: %s", e)

        return gaps


class AnomalyAnalyzer:
    """Handles anomaly detection - SEPARATED CONCERN"""

    def __init__(self, anomaly_threshold: float = 0.2):
        self.anomaly_threshold = anomaly_threshold

        if SKLEARN_AVAILABLE:
            self.anomaly_detector = None
            self.scaler = StandardScaler()
        else:
            self.anomaly_detector = SimpleAnomalyDetector(
                contamination=anomaly_threshold
            )
            self.scaler = None

    def detect_anomalies(
        self, predictions: List[Dict[str, Any]], threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in predictions"""
        try:
            # FIX: Validate input
            if (
                not predictions
                or not isinstance(predictions, list)
                or len(predictions) < 5
            ):
                return []

            if threshold is None:
                threshold = self.anomaly_threshold

            anomalies = []

            # Extract features from predictions
            features = []
            valid_indices = []

            for i, pred in enumerate(predictions):
                if isinstance(pred, dict):
                    feature_vec = [
                        pred.get("value", 0),
                        pred.get("confidence", 0.5),
                        pred.get("variance", 0),
                        pred.get("error", 0),
                    ]
                    features.append(feature_vec)
                    valid_indices.append(i)

            if len(features) < 5:
                return []

            features = np.array(features)

            if SKLEARN_AVAILABLE:
                # Use sklearn if available
                if self.anomaly_detector is None:
                    self.anomaly_detector = IsolationForest(
                        contamination=threshold, random_state=42
                    )

                # Scale features if scaler available
                if self.scaler:
                    features_scaled = self.scaler.fit_transform(features)
                else:
                    features_scaled = features

                self.anomaly_detector.fit(features_scaled)
                anomaly_labels = self.anomaly_detector.predict(features_scaled)
                anomaly_scores = self.anomaly_detector.score_samples(features_scaled)
            else:
                # Use simple detector
                if self.anomaly_detector is None:
                    self.anomaly_detector = SimpleAnomalyDetector(
                        contamination=threshold
                    )

                self.anomaly_detector.fit(features)
                anomaly_labels = self.anomaly_detector.predict(features)
                anomaly_scores = self.anomaly_detector.score_samples(features)

            # Collect anomalies
            for idx, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
                if label == -1:  # Anomaly
                    original_idx = valid_indices[idx]
                    anomaly = {
                        "index": original_idx,
                        "prediction": predictions[original_idx],
                        "anomaly_score": abs(score),  # Convert to positive
                        "features": features[idx].tolist(),
                    }
                    anomalies.append(anomaly)

        except Exception as e:
            logger.warning("Anomaly detection failed: %s", e)
            return []

        return anomalies

    def detect_pattern_anomalies(self, observations: List[Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in pattern observations"""
        anomalies = []

        try:
            if len(observations) < 10:
                return anomalies

            # Extract numeric features
            features = []
            valid_indices = []

            for i, obs in enumerate(observations):
                if isinstance(obs, dict):
                    feature = obs.get("value", obs.get("priority", obs.get("cost", 0)))
                    if isinstance(feature, (int, float)):
                        features.append(feature)
                        valid_indices.append(i)

            if len(features) >= 10:
                features_array = np.array(features)
                mean = np.mean(features_array)
                std = np.std(features_array)

                # FIX: Check for zero std
                if std > 1e-10:
                    # Find outliers using z-score
                    for idx, value in enumerate(features):
                        z_score = abs(value - mean) / std
                        if z_score > 3:  # 3-sigma rule
                            original_idx = valid_indices[idx]
                            anomalies.append(
                                {
                                    "index": original_idx,
                                    "value": value,
                                    "z_score": z_score,
                                    "frequency": (
                                        1 / len(features) if len(features) > 0 else 0.1
                                    ),
                                    "confidence": min(0.99, z_score / 10),
                                    "score": z_score / 10,
                                    "components": (
                                        [observations[original_idx]]
                                        if original_idx < len(observations)
                                        else []
                                    ),
                                }
                            )

        except Exception as e:
            logger.warning("Pattern anomaly detection failed: %s", e)

        return anomalies


class LatentGapDetector:
    """Detects latent gaps through pattern analysis - SEPARATED CONCERN"""

    def detect_from_patterns(
        self, patterns: Dict[str, List[Any]], anomaly_analyzer: AnomalyAnalyzer
    ) -> List[LatentGap]:
        """Detect latent gaps from patterns"""
        gaps = []

        try:
            # FIX: Validate input
            if not patterns or not isinstance(patterns, dict):
                return gaps

            # EXAMINE: Analyze patterns for anomalies
            for pattern_type, observations in patterns.items():
                if len(observations) < 10:  # Need minimum data
                    continue

                # SELECT: Detect anomalies
                anomalies = anomaly_analyzer.detect_pattern_anomalies(
                    list(observations)
                )

                # APPLY: Create gaps from anomalies
                for anomaly in anomalies:
                    impact = self._calculate_anomaly_impact(anomaly)

                    if impact > 0.2:  # Impact threshold
                        pattern = Pattern(
                            pattern_id=f"pattern_{pattern_type}_{int(time.time())}",
                            pattern_type=pattern_type,
                            frequency=anomaly.get("frequency", 0.1),
                            components=anomaly.get("components", []),
                            confidence=anomaly.get("confidence", 0.5),
                        )

                        gap = LatentGap(
                            domain=pattern_type,
                            priority=impact,
                            estimated_cost=15,
                            pattern=pattern,
                            frequency=anomaly.get("frequency", 0.1),
                            impact=impact,
                            detection_confidence=anomaly.get("confidence", 0.5),
                            anomaly_score=anomaly.get("score", 0.0),
                        )

                        gaps.append(gap)

        except Exception as e:
            logger.error("Error detecting latent gaps: %s", e)

        return gaps

    def _calculate_anomaly_impact(self, anomaly: Dict[str, Any]) -> float:
        """Calculate impact of an anomaly"""
        try:
            # Extract factors
            z_score = anomaly.get("z_score", 0)
            frequency = anomaly.get("frequency", 0.1)
            confidence = anomaly.get("confidence", 0.5)

            # Impact increases with deviation and confidence
            # FIX: Ensure division is safe
            base_impact = min(1.0, (z_score / 10) * confidence) if z_score > 0 else 0

            # Adjust for frequency (rare events can be more impactful)
            frequency_factor = 1 + (1 - frequency) * 0.5

            impact = min(1.0, base_impact * frequency_factor)

            return impact
        except Exception as e:
            logger.warning("Error calculating anomaly impact: %s", e)
            return 0.5


class GapAnalyzer:
    """Identifies different types of knowledge gaps - REFACTORED"""

    def __init__(
        self,
        anomaly_threshold: float = 0.2,
        min_frequency: float = 0.1,
        max_history: int = 10000,
    ):
        """
        Initialize gap analyzer

        Args:
            anomaly_threshold: Threshold for anomaly detection
            min_frequency: Minimum frequency for gap reporting
            max_history: Maximum history size
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_frequency = min_frequency

        # Components
        self.failure_tracker = FailureTracker(max_history)
        self.pattern_tracker = PatternTracker(max_history)
        self.gap_registry = GapRegistry(max_history)

        self.decomposition_analyzer = DecompositionAnalyzer(min_frequency)
        self.prediction_analyzer = PredictionAnalyzer()
        self.transfer_analyzer = TransferAnalyzer(min_frequency)
        self.anomaly_analyzer = AnomalyAnalyzer(anomaly_threshold)
        self.latent_detector = LatentGapDetector()

        # Caching
        self._gap_cache = {}
        self._cache_ttl = 60

        # Thread safety
        self._lock = threading.RLock()

        logger.info("GapAnalyzer initialized (refactored)")

    def analyze_decomposition_failures(self) -> List[KnowledgeGap]:
        """Analyze failures in problem decomposition - DELEGATED"""
        with self._lock:
            try:
                failures = self.failure_tracker.get_decomposition_failures()
                gaps = self.decomposition_analyzer.analyze_failures(failures)

                # Register gaps
                for gap in gaps:
                    self.gap_registry.register_gap(gap)

                logger.debug("Found %d decomposition gaps", len(gaps))
                return gaps
            except Exception as e:
                logger.error("Error analyzing decomposition failures: %s", e)
                return []

    def analyze_prediction_errors(self) -> List[KnowledgeGap]:
        """Analyze errors in predictions - DELEGATED"""
        with self._lock:
            try:
                errors = self.failure_tracker.get_prediction_errors()
                gaps = self.prediction_analyzer.analyze_errors(errors)

                # Register gaps
                for gap in gaps:
                    self.gap_registry.register_gap(gap)

                logger.debug("Found %d prediction gaps", len(gaps))
                return gaps
            except Exception as e:
                logger.error("Error analyzing prediction errors: %s", e)
                return []

    def analyze_transfer_failures(self) -> List[KnowledgeGap]:
        """Analyze failures in knowledge transfer - DELEGATED"""
        with self._lock:
            try:
                failures = self.failure_tracker.get_transfer_failures()
                gaps = self.transfer_analyzer.analyze_failures(failures)

                # Register gaps
                for gap in gaps:
                    self.gap_registry.register_gap(gap)

                logger.debug("Found %d transfer gaps", len(gaps))
                return gaps
            except Exception as e:
                logger.error("Error analyzing transfer failures: %s", e)
                return []

    def detect_latent_gaps(self) -> List[LatentGap]:
        """Detect latent gaps through pattern analysis - DELEGATED"""
        with self._lock:
            try:
                patterns = self.pattern_tracker.get_patterns()
                gaps = self.latent_detector.detect_from_patterns(
                    patterns, self.anomaly_analyzer
                )

                # Register gaps
                for gap in gaps:
                    self.gap_registry.register_gap(gap)

                logger.debug("Found %d latent gaps", len(gaps))
                return gaps
            except Exception as e:
                logger.error("Error detecting latent gaps: %s", e)
                return []

    def detect_anomalies(
        self, predictions: List[Dict[str, Any]], threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in predictions - DELEGATED"""
        return self.anomaly_analyzer.detect_anomalies(predictions, threshold)

    def record_failure(self, failure_type: str, failure_data: Dict[str, Any]):
        """Record a failure for analysis - DELEGATED"""
        with self._lock:
            try:
                # Record in failure tracker
                self.failure_tracker.record_failure(failure_type, failure_data)

                # Also track as pattern
                pattern_key = f"{failure_type}_{failure_data.get('domain', 'unknown')}"
                self.pattern_tracker.record_pattern(pattern_key, failure_data)
            except Exception as e:
                logger.error("Error recording failure: %s", e)

    def get_all_gaps(self) -> List[KnowledgeGap]:
        """Get all identified gaps - REFACTORED"""
        with self._lock:
            try:
                # FIX: Atomic cache check
                cache_key = "all_gaps"
                current_time = time.time()

                if cache_key in self._gap_cache:
                    cached_time, cached_gaps = self._gap_cache[cache_key]
                    if current_time - cached_time < self._cache_ttl:
                        return cached_gaps

                # EXAMINE: Analyze all failure types
                all_gaps = []

                # SELECT & APPLY: Get gaps from each analyzer
                all_gaps.extend(self.analyze_decomposition_failures())
                all_gaps.extend(self.analyze_prediction_errors())
                all_gaps.extend(self.analyze_transfer_failures())
                all_gaps.extend(self.detect_latent_gaps())

                # Sort by priority
                all_gaps.sort(key=lambda g: g.priority, reverse=True)

                # REMEMBER: Cache result atomically
                self._gap_cache[cache_key] = (current_time, all_gaps)

                return all_gaps
            except Exception as e:
                logger.error("Error getting all gaps: %s", e)
                return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get gap analysis statistics - DELEGATED"""
        with self._lock:
            try:
                failure_stats = self.failure_tracker.get_statistics()
                registry_stats = self.gap_registry.get_statistics()

                return {
                    **failure_stats,
                    **registry_stats,
                    "pattern_count": self.pattern_tracker.get_pattern_count(),
                    "cache_size": len(self._gap_cache),
                }
            except Exception as e:
                logger.error("Error getting statistics: %s", e)
                return {}

    def update_gap_success(self, gap_id: str, success: bool):
        """Update success tracking for a gap - DELEGATED"""
        self.gap_registry.update_gap_success(gap_id, success)

    def clear_cache(self):
        """Clear the gap cache"""
        with self._lock:
            self._gap_cache.clear()
