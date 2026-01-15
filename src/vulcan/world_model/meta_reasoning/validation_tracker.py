# src/vulcan/world_model/meta_reasoning/validation_tracker.py
"""
validation_tracker.py - Validation history tracking and pattern learning
Part of the meta_reasoning subsystem for VULCAN-AMI

Tracks validation outcomes, learns patterns, identifies failure modes.
Provides memory and learning capabilities for the meta-reasoning system.

Core capabilities:
- Record validation history with outcomes
- Detect patterns in successful vs failed proposals
- Identify blockers preventing objective satisfaction
- Learn better proxies for objectives
- Generate actionable insights from history
- Longitudinal CSIU tracking (for maintainers only, not exposed to UX)

ENHANCEMENTS (2025-10-18):
- Complete longitudinal delta calculations with correct signs (lower entropy/miscomm = improvement)
- Periodic maintainer logging for CSIU trends
- Unified audit trail via TransparencyInterface integration

FIX (2025-10-22):
- Corrected pattern learning/prediction logic for `predict_validation_outcome`
- Improved initial confidence for new patterns in `_incremental_pattern_update`
- Added debugging logs for pattern matching and prediction
- Fixed prediction thresholds to be more conservative and avoid false rejections
- Adjusted pattern rebuilding to include moderate-confidence patterns
- Improved risk scoring to be less aggressive for unknown/neutral patterns
"""

from dataclasses import asdict, is_dataclass
import math
import logging
import math  # Import math for log2, sqrt
import random  # Import random if needed by FakeNumpy
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# import numpy as np # Original import
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock  # --- START FIX: Import MagicMock ---

# --- START FIX: Add numpy fallback ---
# logger = logging.getLogger(__name__) # Original logger placement
logger = logging.getLogger(__name__)  # Moved logger init up
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using list-based math")

    class FakeNumpy:
        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0  # Return float

        @staticmethod
        def array(lst):
            return list(lst)

        @staticmethod
        def log2(x):
            if isinstance(x, list):
                return [math.log2(i) if i > 0 else -float("inf") for i in x]
            return math.log2(x) if x > 0 else -float("inf")

        @staticmethod
        def diff(a, n=1, axis=-1, prepend=None, append=None):
            # Simplified diff for 1D list
            if not isinstance(a, list) or n != 1 or axis != -1:
                raise NotImplementedError(
                    "FakeNumpy diff only supports n=1, axis=-1 on lists"
                )
            if len(a) < 2:
                return []
            result = [a[i] - a[i - 1] for i in range(1, len(a))]
            # Prepend/append not implemented simply here
            return result

        @staticmethod
        def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
            # Simplified std dev for 1D list
            if not isinstance(a, list):
                raise NotImplementedError("FakeNumpy std only supports lists")
            n = len(a)
            if n <= ddof:
                return float("nan")  # Cannot compute std dev
            mean_val = sum(a) / n if n > 0 else 0
            variance = sum((x - mean_val) ** 2 for x in a) / (n - ddof)
            return math.sqrt(variance)

        # Add generic type placeholder if needed elsewhere
        class generic:
            pass

        # Add ndarray type placeholder if needed elsewhere
        class ndarray:
            pass

    np = FakeNumpy()
# --- END FIX ---


# Assuming ProposalValidation is available from motivational_introspection
try:
    from .motivational_introspection import ObjectiveStatus, ProposalValidation

    MOTIVATIONAL_INTROSPECTION_AVAILABLE = True
except ImportError:
    MOTIVATIONAL_INTROSPECTION_AVAILABLE = False
    logger.warning(
        "Failed to import types from motivational_introspection. Using dummy types."
    )

    # Fallback definition if run standalone or MI not found
    class ProposalValidation:
        pass

    class ObjectiveStatus(Enum):
        ALIGNED = "aligned"
        CONFLICT = "conflict"
        VIOLATION = "violation"
        DRIFT = "drift"
        ACCEPTABLE = "acceptable"
        UNKNOWN = "unknown"  # Add UNKNOWN


class ValidationOutcome(Enum):
    """Outcome of validation"""

    APPROVED = "approved"  # Passed validation
    REJECTED = "rejected"  # Failed validation
    MODIFIED = "modified"  # Modified during validation/enforcement
    UNKNOWN = "unknown"  # Outcome not yet determined


class PatternType(Enum):
    """Type of learned pattern"""

    SUCCESS = "success"  # Correlates with approval
    FAILURE = "failure"  # Correlates with rejection (deprecated, use RISKY)
    RISKY = "risky"  # Correlates with rejection
    SAFE = "safe"  # Correlates with approval (alternative name for SUCCESS)


@dataclass
class ValidationRecord:
    """Record of a validation event"""

    proposal_id: str
    proposal: Dict[str, Any]
    validation_result: Any  # Can be ProposalValidation object or dict
    outcome: ValidationOutcome  # Based on validation_result.valid initially
    actual_outcome: Optional[str] = (
        None  # What actually happened after execution ('success', 'failure')
    )
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_features(self) -> Dict[str, Any]:
        """Extract features from proposal for pattern matching, ensuring hashability"""
        _np = np if NUMPY_AVAILABLE else FakeNumpy  # Use internal alias
        features = {}
        proposal_data = self.proposal  # Assume proposal is dict

        # Objective-related features
        if "objective" in proposal_data and isinstance(proposal_data["objective"], str):
            features["objective"] = proposal_data["objective"]

        if "objectives" in proposal_data:
            objectives_list = proposal_data["objectives"]
            # Ensure it's a list of strings, convert to sorted tuple
            if isinstance(objectives_list, list) and all(
                isinstance(o, str) for o in objectives_list
            ):
                # Convert list to sorted tuple for hashability and consistency
                # **************************************************************************
                # FIX: Change key to 'objectives' to match test
                features["objectives"] = tuple(sorted(objectives_list))
                # **************************************************************************
                features["num_objectives"] = len(objectives_list)
            elif isinstance(
                objectives_list, str
            ):  # Handle single objective string case
                # **************************************************************************
                # FIX: Change key to 'objectives' to match test
                features["objectives"] = tuple([objectives_list])
                # **************************************************************************
                features["num_objectives"] = 1

        # Constraint-related features
        constraints_data = proposal_data.get("constraints")
        if isinstance(constraints_data, (dict, list)):
            features["has_constraints"] = True
            features["num_constraints"] = len(constraints_data)
            # Could add features based on constraint *types* if schema is known

        # Trade-off features
        tradeoffs_data = proposal_data.get("tradeoffs")
        if isinstance(tradeoffs_data, (dict, list)):
            features["has_tradeoffs"] = True
            features["num_tradeoffs"] = len(tradeoffs_data)

        # Weight features
        weights_data = proposal_data.get("objective_weights")
        if isinstance(weights_data, dict):
            features["has_weights"] = True
            # Calculate diversity only if values are numeric
            numeric_weights = [
                v for v in weights_data.values() if isinstance(v, (int, float))
            ]
            if numeric_weights:
                features["weight_diversity"] = self._calculate_weight_diversity(
                    numeric_weights
                )  # Pass list

        # Domain features
        domain_data = proposal_data.get("domain")
        if isinstance(domain_data, str):
            features["domain"] = domain_data

        # Action type
        action_data = proposal_data.get("action")
        if isinstance(action_data, str):
            features["action_type"] = action_data

        # Ensure all feature values are hashable before returning
        # The logic above tries to create hashable representations (tuples, numbers, strings, bools)
        # Final check if needed, but ideally handled during creation.
        # for k, v in features.items():
        #      assert isinstance(v, (str, int, float, bool, tuple)), f"Feature '{k}' is not hashable: {type(v)}"

        return features

    def _calculate_weight_diversity(self, weights_list: List[float]) -> float:
        """Calculate diversity of weights (entropy-like measure) using internal np alias"""
        _np = np if NUMPY_AVAILABLE else FakeNumpy  # Use internal alias
        if not weights_list:
            return 0.0

        # Ensure weights are non-negative
        values = [max(0.0, w) for w in weights_list]
        total = sum(values)

        if total <= 1e-9:  # Avoid division by zero or log(0)
            return 0.0

        # Normalized entropy calculation
        entropy = 0.0
        for v in values:
            if v > 1e-9:  # Check > 0 with tolerance
                p = v / total
                log_p = _np.log2(p)  # Use internal alias
                # Handle potential -inf from log2(very small p)
                if log_p != float("-inf"):
                    entropy -= p * log_p

        # Normalize to 0-1
        num_weights = len(values)
        # Max entropy for uniform distribution log2(N)
        max_entropy = (
            _np.log2(num_weights) if num_weights > 1 else 1.0
        )  # Use internal alias

        # Avoid division by zero if max_entropy is zero or negative (shouldn't happen for num_weights > 0)
        return (entropy / max_entropy) if max_entropy > 1e-9 else 0.0


@dataclass
class ValidationPattern:
    """Learned pattern from validation history"""

    pattern_type: PatternType
    features: Dict[
        str, Any
    ]  # Features dict (keys are strings, values should be hashable)
    support: int  # How many examples support this pattern
    confidence: float  # Confidence in pattern (0-1, Bayesian posterior mean)
    examples: List[str] = field(default_factory=list)  # Example proposal IDs
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Store alpha/beta here

    # Add hash and eq for use in sets or as dict keys
    def __hash__(self):
        # Hash based on features (convert dict to sorted tuple of items)
        try:
            # Ensure features dict itself is hashable by converting to tuple
            feature_key = tuple(sorted(self.features.items()))
        except TypeError:
            # Fallback if values within features are somehow unhashable despite get_features logic
            logger.warning(
                f"Unhashable feature value found in pattern: {self.features}"
            )
            feature_key = tuple(
                sorted({k: str(v) for k, v in self.features.items()}.items())
            )  # Convert values to str

        # Include pattern_type in hash
        return hash((self.pattern_type, feature_key))

    def __eq__(self, other):
        if not isinstance(other, ValidationPattern):
            return NotImplemented
        # Compare based on type and features dict content
        try:
            # Compare features dicts directly (requires consistent ordering or conversion)
            feature_key_self = tuple(sorted(self.features.items()))
            feature_key_other = tuple(sorted(other.features.items()))
            return (
                self.pattern_type == other.pattern_type
                and feature_key_self == feature_key_other
            )
        except TypeError:
            # Fallback comparison if features contain unhashable types
            return self.pattern_type == other.pattern_type and str(
                self.features
            ) == str(
                other.features
            )  # Less reliable fallback


@dataclass
class LearningInsight:
    """Actionable insight learned from history"""

    insight_type: str
    description: str
    evidence: List[Dict[str, Any]]  # Should contain serializable evidence
    confidence: float
    recommendation: str
    priority: str  # 'high', 'medium', 'low'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveBlocker:
    """Something preventing objective satisfaction"""

    objective: str
    blocker_type: str
    description: str
    frequency: int  # How often this blocks
    severity: float  # 0-1 scale (e.g., based on frequency or impact)
    examples: List[str]  # List of proposal IDs
    potential_solutions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationTracker:
    """
    Tracks validation history and learns patterns

    Provides memory and learning for meta-reasoning:
    - Records all validations with outcomes
    - Detects patterns in success/failure
    - Identifies common blockers
    - Learns better objective proxies
    - Generates actionable insights
    - Tracks CSIU metrics longitudinally (internal only)
    """

    # --- START FIX: Modify __init__ to accept world_model ---
    def __init__(
        self,
        world_model,  # <-- ADDED
        max_history: int = 10000,
        self_improvement_drive=None,
        transparency_interface=None,
    ):
        """
        Initialize validation tracker

        Args:
            world_model: The parent WorldModel instance.
            max_history: Maximum validation records to keep
            self_improvement_drive: Optional SelfImprovementDrive instance (can be mock)
            transparency_interface: Optional TransparencyInterface instance (can be mock)
        """
        self.world_model = world_model  # <-- ADDED

        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        self.max_history = max_history
        # Use MagicMock for optional dependencies if None
        self.self_improvement_drive = self_improvement_drive or MagicMock(
            _csiu_enabled=False
        )
        self.transparency_interface = transparency_interface or MagicMock()
        # --- END FIX ---

        # Validation history (deque provides maxlen automatically)
        self.validation_records: deque[ValidationRecord] = deque(maxlen=max_history)
        # Index for fast lookup by ID (needs manual management if history size exceeded)
        self.records_by_id: Dict[str, ValidationRecord] = {}

        # Learned patterns (store as set for faster lookups/updates?) No, list allows sorting.
        self.patterns: List[ValidationPattern] = []
        # Index patterns by features for faster matching during prediction
        # Key: tuple(sorted(features.items())), Value: ValidationPattern
        self.pattern_index: Dict[tuple, ValidationPattern] = {}

        # Blockers
        self.blockers: List[ObjectiveBlocker] = []
        # Index blockers by objective for faster lookup
        self.blocker_index: Dict[str, List[ObjectiveBlocker]] = defaultdict(list)

        # Statistics
        self.stats = defaultdict(int)

        # Learning configuration
        self.min_pattern_support = 3  # Minimum examples to form pattern
        self.pattern_confidence_threshold = 0.6  # Adjusted confidence threshold
        self.learning_enabled = True
        self.relearn_interval = 50  # Rebuild patterns every N validations
        self.last_relearn = 0

        # CSIU periodic logging
        self.csiu_log_interval = 100  # Log CSIU trends every N validations
        self.last_csiu_log = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ValidationTracker initialized")

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare state for pickling by removing unpickleable objects.
        
        This fixes the persistence firewall issue where threading locks,
        module references, and mock objects cannot be pickled.
        """
        state = self.__dict__.copy()
        # Remove unpickleable items - they will be re-created on unpickle
        state.pop('lock', None)  # threading.RLock
        state.pop('_np', None)  # numpy module reference
        # Note: world_model, self_improvement_drive, transparency_interface 
        # should be re-injected after unpickling if needed
        state.pop('world_model', None)
        state.pop('self_improvement_drive', None)
        state.pop('transparency_interface', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore state after unpickling, re-creating unpickleable objects.
        
        Note: world_model, self_improvement_drive, and transparency_interface
        must be re-injected after unpickling by calling set_dependencies().
        """
        self.__dict__.update(state)
        # Re-create unpickleable objects
        self.lock = threading.RLock()
        self._np = np if NUMPY_AVAILABLE else FakeNumpy
        # Set placeholder for dependencies that must be re-injected
        self.world_model = None
        self.self_improvement_drive = MagicMock(_csiu_enabled=False)
        self.transparency_interface = MagicMock()

    def set_dependencies(
        self,
        world_model=None,
        self_improvement_drive=None,
        transparency_interface=None
    ) -> None:
        """
        Re-inject dependencies after unpickling.
        
        Call this method after restoring from pickle to reconnect
        the tracker to other system components.
        """
        if world_model is not None:
            self.world_model = world_model
        if self_improvement_drive is not None:
            self.self_improvement_drive = self_improvement_drive
        if transparency_interface is not None:
            self.transparency_interface = transparency_interface

    def record_validation(
        self,
        proposal: Dict[str, Any],
        validation_result: Any,  # Can be ProposalValidation or dict
        actual_outcome: Optional[str] = None,
    ) -> ValidationRecord:
        """
        Record a validation event, ensuring data serializability.

        Args:
            proposal: The proposal dict.
            validation_result: Result object (e.g., ProposalValidation) or dict.
            actual_outcome: Optional actual outcome ('success', 'failure').

        Returns:
            Created ValidationRecord.
        """
        with self.lock:
            # Determine outcome based on validation result's 'valid' flag safely
            valid = False
            validation_dict = {}  # Keep serialized version for the record
            if isinstance(validation_result, dict):
                valid = validation_result.get("valid", False)
                validation_dict = self._make_serializable(
                    validation_result
                )  # Serialize dict input
            elif hasattr(validation_result, "valid"):
                valid = getattr(validation_result, "valid", False)
                # Serialize object input using its method or dataclass function
                # Check for Mock BEFORE to_dict to avoid calling to_dict() on Mock
                if hasattr(
                    validation_result, "_extract_mock_name"
                ):  # Check if it's a Mock
                    validation_dict = {
                        "valid": valid,
                        "reasoning": self._make_serializable(
                            getattr(validation_result, "reasoning", None)
                        ),
                    }
                elif hasattr(validation_result, "to_dict"):
                    try:
                        validation_dict = self._make_serializable(
                            validation_result.to_dict()
                        )
                    except Exception:
                        validation_dict = self._make_serializable(
                            vars(validation_result)
                        )  # Fallback
                elif is_dataclass(validation_result):
                    try:
                        validation_dict = self._make_serializable(
                            asdict(validation_result)
                        )
                    except Exception:
                        validation_dict = self._make_serializable(
                            vars(validation_result)
                        )  # Fallback
                else:  # Fallback for unknown objects
                    validation_dict = self._make_serializable(validation_result)

            outcome = (
                ValidationOutcome.APPROVED if valid else ValidationOutcome.REJECTED
            )
            # Check if modified (might need info from validation_result or context?)
            # Assuming modification status isn't directly available here, default based on valid/rejected.

            # Create record ID
            proposal_id = str(
                proposal.get(
                    "id",
                    f"proposal_{time.time_ns()}_{self.stats.get('total_validations', 0)}",
                )
            )

            # Ensure proposal is serializable
            serializable_proposal = self._make_serializable(proposal)
            if not isinstance(serializable_proposal, dict):  # Ensure it's still a dict
                logger.error("Proposal serialization failed, storing as string.")
                serializable_proposal = {"raw_proposal_str": str(proposal)}

            # Extract features from the serializable proposal
            try:
                features = self._extract_proposal_features(serializable_proposal)
                metadata = {
                    "features": features
                }  # Store extracted features in metadata
            except Exception as e:
                logger.error(
                    f"Failed to extract features for proposal {proposal_id}: {e}",
                    exc_info=True,
                )
                metadata = {"features": {}, "feature_error": str(e)}

            # Add CSIU metadata if available (internal only)
            if not isinstance(self.self_improvement_drive, MagicMock) and getattr(
                self.self_improvement_drive, "_csiu_enabled", False
            ):
                try:
                    csiu_meta = self._get_csiu_metadata()  # Use helper
                    if csiu_meta:
                        metadata["csiu"] = csiu_meta
                except Exception as e:
                    logger.debug(
                        f"Failed to add CSIU metadata to record {proposal_id}: {e}"
                    )

            record = ValidationRecord(
                proposal_id=proposal_id,
                proposal=serializable_proposal,  # Store serialized version
                validation_result=validation_dict,  # Store serialized version
                outcome=outcome,
                actual_outcome=actual_outcome,
                metadata=metadata,  # Contains features and potentially CSIU
            )

            # --- Manage History Size ---
            # If deque is full, remove oldest entry from records_by_id before adding new
            if len(self.validation_records) >= self.max_history:
                oldest_record = self.validation_records[0]  # Will be popped by deque
                if oldest_record.proposal_id in self.records_by_id:
                    del self.records_by_id[oldest_record.proposal_id]

            # Store record
            self.validation_records.append(record)
            self.records_by_id[proposal_id] = record

            # Update statistics
            total_validations = self.stats["total_validations"] + 1  # Use temp var
            self.stats["total_validations"] = total_validations
            self.stats[f"outcome_{outcome.value}"] += 1

            # Trigger learning if enabled
            if self.learning_enabled:
                self._incremental_pattern_update(record)
                # Periodic comprehensive relearning
                if total_validations % self.relearn_interval == 0:
                    logger.info(
                        f"Triggering comprehensive relearn at validation #{total_validations}"
                    )
                    self._comprehensive_relearn()
                    self.last_relearn = total_validations

            # Periodic CSIU logging (maintainer only)
            if (
                not isinstance(self.self_improvement_drive, MagicMock)
                and getattr(self.self_improvement_drive, "_csiu_enabled", False)
                and total_validations % self.csiu_log_interval == 0
            ):
                logger.info(
                    f"Triggering CSIU trend logging at validation #{total_validations}"
                )
                self._log_csiu_trends()
                self.last_csiu_log = total_validations

            logger.debug(f"Recorded validation: {proposal_id} -> {outcome.value}")

            # Audit via Transparency Interface
            if not isinstance(self.transparency_interface, MagicMock) and hasattr(
                self.transparency_interface, "_audit"
            ):
                try:
                    audit_data = {
                        "proposal_id": proposal_id,
                        "outcome": outcome.value,
                        "valid": valid,
                        "actual_outcome": actual_outcome,
                        "features_hash": hash(
                            tuple(sorted(metadata.get("features", {}).items()))
                        ),  # Hash features for audit
                    }
                    # Include CSIU summary in audit if present
                    if "csiu" in metadata:
                        audit_data["csiu_audit"] = metadata["csiu"]

                    self.transparency_interface._audit(
                        "validation_recorded", audit_data
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to audit validation record via transparency interface: {e}"
                    )

            return record

    def _get_csiu_metadata(self) -> Optional[Dict[str, Any]]:
        """Helper to safely get CSIU metadata from self_improvement_drive."""
        try:
            # Get current telemetry snapshot
            cur_telemetry = {}
            if hasattr(self.self_improvement_drive, "_collect_telemetry_snapshot"):
                telemetry_result = (
                    self.self_improvement_drive._collect_telemetry_snapshot()
                )
                if isinstance(telemetry_result, dict):
                    cur_telemetry = telemetry_result

            # Extract key features used in longitudinal summary
            features = {
                "A": cur_telemetry.get("A", 0.85),
                "H": cur_telemetry.get("H", 0.06),
                "C": cur_telemetry.get("C", 0.88),
                "E": cur_telemetry.get("E", 0.50),
                "U": cur_telemetry.get("U", 0.70),
                "M": cur_telemetry.get("M", 0.02),
            }

            return {
                "U_prev": getattr(self.self_improvement_drive, "_csiu_U_prev", 0.0),
                "U_ewma": getattr(self.self_improvement_drive, "_csiu_u_ewma", 0.0),
                "features": features,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.debug(f"Failed to get CSIU metadata: {e}")
            return None

    # --- summarize_longitudinal, _log_csiu_trends, update_actual_outcome ---
    # (Copied from previous version, ensuring self._np usage)
    def summarize_longitudinal(self, window: int = 500) -> Dict[str, Any]:
        """Summarize CSIU metrics over a rolling window."""
        _np = self._np
        with self.lock:
            history_list = list(self.validation_records)
            effective_window = min(window, len(history_list))
            last = history_list[-effective_window:]
            if not last:
                return {}

            def avg(key: str, default: float = 0.0) -> float:
                vals = [
                    r.metadata.get("csiu", {}).get("features", {}).get(key)
                    for r in last
                    if r.metadata and "csiu" in r.metadata
                ]
                vals = [float(v) for v in vals if isinstance(v, (int, float))]
                return _np.mean(vals) if vals else default

            summary = {"window_size": len(last), "timestamp": time.time()}
            features_to_avg = {
                "A": 0.85,
                "H": 0.06,
                "C": 0.88,
                "E": 0.50,
                "U": 0.70,
                "M": 0.02,
            }
            feature_names = {
                "A": "alignment_coherence",
                "H": "entropy",
                "C": "intent_clarity",
                "E": "empathy",
                "U": "user_satisfaction",
                "M": "miscomm",
            }

            for key, default in features_to_avg.items():
                summary[f"trend_{feature_names[key]}"] = avg(key, default)

            # Calculate trends (deltas)
            if len(last) >= 20:
                mid = len(last) // 2
                first_half = last[:mid]
                second_half = last[mid:]

                def avg_half(records, key: str, default: float = 0.0) -> float:
                    vals = [
                        r.metadata.get("csiu", {}).get("features", {}).get(key)
                        for r in records
                        if r.metadata and "csiu" in r.metadata
                    ]
                    vals = [float(v) for v in vals if isinstance(v, (int, float))]
                    return _np.mean(vals) if vals else default

                summary["delta_alignment"] = avg_half(second_half, "A") - avg_half(
                    first_half, "A"
                )
                summary["delta_entropy"] = avg_half(first_half, "H") - avg_half(
                    second_half, "H"
                )  # Lower is better
                summary["delta_clarity"] = avg_half(second_half, "C") - avg_half(
                    first_half, "C"
                )
                summary["delta_empathy"] = avg_half(second_half, "E") - avg_half(
                    first_half, "E"
                )
                summary["delta_satisfaction"] = avg_half(second_half, "U") - avg_half(
                    first_half, "U"
                )
                summary["delta_miscomm"] = avg_half(first_half, "M") - avg_half(
                    second_half, "M"
                )  # Lower is better

            return summary

    def _log_csiu_trends(self):
        """Periodic maintainer logging for CSIU trends."""
        try:
            trends = self.summarize_longitudinal()
            if not trends:
                return

            logger.info(
                "[CSIU_TRENDS] Longitudinal summary (window: %d):",
                trends.get("window_size", 0),
            )
            # Log main trends
            log_items = [
                ("Alignment", "trend_alignment_coherence", "delta_alignment"),
                ("Entropy", "trend_entropy", "delta_entropy"),
                ("Clarity", "trend_intent_clarity", "delta_clarity"),
                ("Empathy", "trend_empathy", "delta_empathy"),
                ("Satisfaction", "trend_user_satisfaction", "delta_satisfaction"),
                ("Miscomm", "trend_miscomm", "delta_miscomm"),
            ]
            for name, trend_key, delta_key in log_items:
                trend_val = trends.get(trend_key, 0.0)
                delta_val = trends.get(delta_key)  # Might be None if window too small
                if delta_val is not None:
                    logger.info(f"  {name}: {trend_val:.3f} (Δ: {delta_val:+.3f})")
                else:
                    logger.info(f"  {name}: {trend_val:.3f} (Δ: N/A)")

            # Feed to TransparencyInterface
            if not isinstance(self.transparency_interface, MagicMock) and hasattr(
                self.transparency_interface, "_audit"
            ):
                try:
                    # Pass serializable trends dict
                    self.transparency_interface._audit(
                        "csiu_trends_summary", self._make_serializable(trends)
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to add CSIU trends to transparency audit: {e}"
                    )

        except Exception as e:
            logger.error(f"Failed to log CSIU trends: {e}", exc_info=True)

    def update_actual_outcome(self, proposal_id: str, actual_outcome: str):
        """Update actual outcome after proposal execution"""
        with self.lock:
            if proposal_id in self.records_by_id:
                record = self.records_by_id[proposal_id]
                # Validate outcome string?
                valid_outcomes = {"success", "failure"}
                if actual_outcome not in valid_outcomes:
                    logger.warning(
                        f"Invalid actual_outcome '{actual_outcome}' for {proposal_id}. Expected 'success' or 'failure'."
                    )
                    # Optionally default or ignore? Ignore for now.
                    return

                record.actual_outcome = actual_outcome
                record.metadata["outcome_updated_at"] = time.time()

                # Trigger learning update based on this new information
                if self.learning_enabled:
                    self._update_patterns_with_outcome(record)

                # Update stats
                self.stats[f"actual_outcome_{actual_outcome}"] += 1

                logger.debug(
                    f"Updated actual outcome for {proposal_id}: {actual_outcome}"
                )
            else:
                logger.warning(
                    f"Could not find record for proposal_id '{proposal_id}' to update actual outcome."
                )

    # --- identify_risky_patterns, identify_success_patterns ---
    # (Copied from previous version)
    def identify_risky_patterns(
        self, min_support: Optional[int] = None
    ) -> List[ValidationPattern]:
        """Identify patterns associated with rejected proposals"""
        with self.lock:
            min_support = (
                min_support if min_support is not None else self.min_pattern_support
            )
            # Filter patterns (ensure type comparison is safe)
            risky_patterns = [
                p
                for p in self.patterns
                if p.pattern_type == PatternType.RISKY and p.support >= min_support
            ]
            # Sort by confidence then support
            risky_patterns.sort(key=lambda p: (p.confidence, p.support), reverse=True)
            return risky_patterns

    def identify_success_patterns(
        self, min_support: Optional[int] = None
    ) -> List[ValidationPattern]:
        """Identify patterns associated with approved proposals"""
        with self.lock:
            min_support = (
                min_support if min_support is not None else self.min_pattern_support
            )
            success_types = {PatternType.SUCCESS, PatternType.SAFE}
            success_patterns = [
                p
                for p in self.patterns
                if p.pattern_type in success_types and p.support >= min_support
            ]
            success_patterns.sort(key=lambda p: (p.confidence, p.support), reverse=True)
            return success_patterns

    # --- identify_blockers, analyze_failure_patterns ---
    # (Copied from previous version, ensuring self._np usage)
    def identify_blockers(
        self, objective: Optional[str] = None
    ) -> List[ObjectiveBlocker]:
        """Identify what prevents objectives from being satisfied"""
        with self.lock:
            # Re-detect blockers on demand? Or rely on periodic relearn? Re-detect for now.
            self.detect_blockers_from_history()  # Ensure blockers are up-to-date

            if objective:
                blockers = self.blocker_index.get(objective, [])
            else:
                blockers = self.blockers[:]  # Return a copy

            # Sort by severity then frequency
            blockers.sort(key=lambda b: (b.severity, b.frequency), reverse=True)
            return blockers

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in validation failures"""
        self._np
        with self.lock:
            history_list = list(self.validation_records)  # Snapshot
            rejected = [
                r for r in history_list if r.outcome == ValidationOutcome.REJECTED
            ]
            total_records = len(history_list)

            if not rejected:
                return {
                    "total_failures": 0,
                    "failure_rate": 0.0,
                    "common_features": [],
                    "temporal_pattern": None,
                    "failure_reasons": {},
                }

            # Analyze features
            feature_counts = Counter()
            for record in rejected:
                features = record.metadata.get("features", {})  # Use stored features
                for feature, value in features.items():
                    try:
                        feature_key = f"{feature}={value}"  # Basic key
                    except TypeError:
                        feature_key = f"{feature}=<unhashable_{type(value).__name__}>"
                    feature_counts[feature_key] += 1

            common_features = feature_counts.most_common(10)

            # Temporal analysis
            timestamps = [r.timestamp for r in rejected]
            temporal_pattern = self._analyze_temporal_pattern(
                timestamps
            )  # Uses self._np

            # Failure reasons
            failure_reasons = Counter()
            for record in rejected:
                reasoning = self._safe_get_reasoning(record.validation_result)
                if reasoning:
                    blocker_type = self._classify_blocker(reasoning)
                    failure_reasons[blocker_type] += 1

            return {
                "total_failures": len(rejected),
                "failure_rate": (
                    len(rejected) / total_records if total_records > 0 else 0.0
                ),
                "common_features": common_features,
                "temporal_pattern": temporal_pattern,
                "failure_reasons": dict(failure_reasons),
            }

    # --- predict_validation_outcome ---
    # (Copied from previous corrected version)
    def predict_validation_outcome(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Predict validation outcome based on historical patterns"""
        with self.lock:
            try:
                features = self._extract_proposal_features(proposal)
                # Ensure features key is hashable
                features_key = tuple(sorted(features.items()))
            except Exception as e:
                logger.error(f"Could not extract features for prediction: {e}")
                return {
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "risk_score": 0.5,
                    "reasoning": "Feature extraction failed",
                }

            logger.debug(f"Predicting outcome for features: {features}")
            matching_risky = 0.0
            matching_success = 0.0
            matched_patterns = []

            # Check for exact match first using index
            exact_match_pattern = self.pattern_index.get(features_key)
            if exact_match_pattern:
                logger.debug(f"  Found exact pattern match via index.")
                # Should be only one pattern per exact feature set? If list, take first for now.
                p = (
                    exact_match_pattern[0]
                    if isinstance(exact_match_pattern, list)
                    else exact_match_pattern
                )
                matched_patterns.append(p)
                if p.pattern_type in [PatternType.RISKY, PatternType.FAILURE]:
                    matching_risky = p.confidence
                elif p.pattern_type in [PatternType.SUCCESS, PatternType.SAFE]:
                    matching_success = p.confidence
            else:
                # If no exact match, check for subset matches (more expensive)
                logger.debug("  No exact pattern match, checking subset matches...")
                for pattern in self.patterns:
                    if self._features_match_pattern(
                        features, pattern.features
                    ):  # Check if proposal features contain pattern features
                        matched_patterns.append(pattern)
                        logger.debug(
                            f"  Matched subset pattern: type={pattern.pattern_type.value}, conf={pattern.confidence:.2f}, support={pattern.support}, features={pattern.features}"
                        )
                        if pattern.pattern_type in [
                            PatternType.RISKY,
                            PatternType.FAILURE,
                        ]:
                            # Accumulate confidence for multiple matches? Weighted sum? Max? Simple sum for now.
                            matching_risky += pattern.confidence
                        elif pattern.pattern_type in [
                            PatternType.SUCCESS,
                            PatternType.SAFE,
                        ]:
                            matching_success += pattern.confidence

            total_confidence_signal = matching_risky + matching_success
            logger.debug(
                f"  Total matching confidence signal: Success={matching_success:.2f}, Risky={matching_risky:.2f}"
            )

            # Prediction logic (revised thresholds)
            risk_score = 0.5
            prediction = "unknown"
            confidence = 0.0
            reasoning = "No strong matching patterns found."

            if total_confidence_signal > 0.1:  # Need some signal
                risk_score = (
                    matching_risky / total_confidence_signal
                    if total_confidence_signal > 0
                    else 0.5
                )
                # Decision Thresholds (more conservative)
                approve_threshold = 0.65  # Need reasonable confidence in success
                reject_threshold = 0.75  # Need high confidence in riskiness

                if (
                    matching_success / total_confidence_signal > approve_threshold
                    and matching_risky / total_confidence_signal
                    < (1 - approve_threshold)
                ):
                    prediction = "likely_approved"
                    confidence = matching_success / total_confidence_signal
                    reasoning = f"Matched {len(matched_patterns)} pattern(s) strongly suggesting approval."
                elif (
                    matching_risky / total_confidence_signal > reject_threshold
                    and matching_success / total_confidence_signal
                    < (1 - reject_threshold)
                ):
                    prediction = "likely_rejected"
                    confidence = matching_risky / total_confidence_signal
                    reasoning = f"Matched {len(matched_patterns)} pattern(s) strongly suggesting rejection."
                else:  # Uncertain zone or conflicting signals
                    # Default to likely_approved to avoid false rejections
                    prediction = "likely_approved"
                    # Confidence reflects the uncertainty or conflict
                    confidence = (
                        1.0
                        - abs(matching_success - matching_risky)
                        / total_confidence_signal
                    )
                    reasoning = f"Matched {len(matched_patterns)} pattern(s) with conflicting or uncertain signals. Defaulting to approval."

            else:  # No significant pattern matches
                prediction = "unknown"  # FIX: Change default to 'unknown' to match test
                confidence = 0.0  # Low confidence
                risk_score = 0.5  # Neutral default risk
                reasoning = "No relevant historical patterns found."

            logger.debug(
                f"  Final Prediction: {prediction}, Risk score: {risk_score:.2f}, Confidence: {confidence:.2f}"
            )

            return {
                "prediction": prediction,
                "confidence": confidence,
                "risk_score": risk_score,  # Use risk score for nuance
                "reasoning": reasoning,
            }

    # --- get_learning_insights, suggest_better_proxies, _safe_get_reasoning ---
    # (Copied from previous version)
    def get_learning_insights(self, limit: int = 10) -> List[LearningInsight]:
        """Generate actionable insights from validation history"""
        with self.lock:
            insights = []
            # Insight 1: High-risk patterns
            risky = self.identify_risky_patterns(
                min_support=max(self.min_pattern_support, 5)
            )  # Higher support for insight
            if risky:
                p = risky[0]
                insights.append(
                    LearningInsight(
                        "high_risk_pattern",
                        f"Proposals like {p.features} often rejected ({p.confidence:.1%})",
                        [{"pattern": p.features, "support": p.support}],
                        p.confidence,
                        "Avoid or revise proposals matching this pattern",
                        "high",
                    )
                )
            # Insight 2: Success patterns
            successful = self.identify_success_patterns(
                min_support=max(self.min_pattern_support, 5)
            )
            if successful:
                p = successful[0]
                insights.append(
                    LearningInsight(
                        "success_pattern",
                        f"Proposals like {p.features} often approved ({p.confidence:.1%})",
                        [{"pattern": p.features, "support": p.support}],
                        p.confidence,
                        "Prioritize similar proposals",
                        "medium",
                    )
                )
            # Insight 3: Common blockers
            blockers = self.identify_blockers()
            if blockers:
                b = blockers[0]
                insights.append(
                    LearningInsight(
                        "common_blocker",
                        f'Objective "{b.objective}" often blocked by "{b.blocker_type}" ({b.frequency} times)',
                        [{"blocker": b.blocker_type, "freq": b.frequency}],
                        min(1.0, b.frequency / 10),
                        (
                            b.potential_solutions[0]
                            if b.potential_solutions
                            else "Investigate blocker"
                        ),
                        "high" if b.severity > 0.7 else "medium",
                    )
                )

            # Insight 4: Temporal trends
            analysis = self.analyze_failure_patterns()  # Re-use failure analysis
            if analysis.get("temporal_pattern"):
                # Simple check for clustering or regularity changes could go here
                insights.append(
                    LearningInsight(
                        "temporal_trend",
                        "Failure temporal pattern detected.",
                        [analysis["temporal_pattern"]],
                        0.5,
                        "Investigate failure timing",
                        "low",
                    )
                )

            # Insight 5: Objective Difficulty
            obj_perf = self._analyze_objective_performance()
            for obj, perf in obj_perf.items():
                if perf["failure_rate"] > 0.6 and perf["total"] >= 5:
                    insights.append(
                        LearningInsight(
                            "objective_difficulty",
                            f'Objective "{obj}" has high failure rate ({perf["failure_rate"]:.1%})',
                            [perf],
                            min(1.0, perf["total"] / 10),
                            f'Review strategy for "{obj}"',
                            "high",
                        )
                    )

            priority_order = {"high": 0, "medium": 1, "low": 2}
            insights.sort(
                key=lambda i: (priority_order.get(i.priority, 2), -i.confidence)
            )
            return insights[:limit]

    def suggest_better_proxies(self, objective: str) -> List[Dict[str, Any]]:
        """Suggest better proxy metrics for objective"""
        with self.lock:
            history_list = list(self.validation_records)  # Snapshot
            relevant_records = [
                r
                for r in history_list
                if (
                    r.proposal.get("objective") == objective
                    or objective in r.proposal.get("objectives", [])
                )
                and r.actual_outcome is not None
            ]
            if not relevant_records:
                return []

            feature_success_rates = defaultdict(lambda: {"success": 0, "total": 0})
            for record in relevant_records:
                features = record.metadata.get("features", {})  # Use stored features
                success = record.actual_outcome == "success"
                for feature, value in features.items():
                    try:
                        feature_key = f"{feature}={value}"
                    except TypeError:
                        feature_key = f"{feature}=<unhashable_{type(value).__name__}>"
                    feature_success_rates[feature_key]["total"] += 1
                    if success:
                        feature_success_rates[feature_key]["success"] += 1

            proxies = []
            min_support_proxy = max(2, self.min_pattern_support - 1)
            for feature_key, counts in feature_success_rates.items():
                if counts["total"] >= min_support_proxy:
                    success_rate = counts["success"] / counts["total"]
                    predictive_power = abs(success_rate - 0.5) * 2
                    if predictive_power > 0.5:  # Higher threshold for suggestion
                        proxies.append(
                            {
                                "proxy": feature_key,
                                "success_rate": success_rate,
                                "support": counts["total"],
                                "predictive": predictive_power,
                            }
                        )

            proxies.sort(key=lambda p: p["predictive"], reverse=True)
            return proxies[:5]

    def _safe_get_reasoning(self, validation_result: Any) -> Optional[str]:
        """Safely extract reasoning string"""
        reasoning = None
        if hasattr(validation_result, "reasoning"):
            reasoning = validation_result.reasoning
        elif isinstance(validation_result, dict):
            reasoning = validation_result.get("reasoning")
        try:
            return str(reasoning) if reasoning is not None else None
        except Exception:
            return None

    # --- _extract_proposal_features, _incremental_pattern_update, ---
    # --- _update_patterns_with_outcome, _comprehensive_relearn, _rebuild_patterns ---
    # (Copied from previous corrected version)
    def _extract_proposal_features(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features, ensuring hashability for use as keys/in sets."""
        if not isinstance(proposal, dict):
            return {}  # Handle invalid input

        features = {}
        # Objective features
        obj = proposal.get("objective")
        if isinstance(obj, str):
            features["objective"] = obj
        objs = proposal.get("objectives")
        if isinstance(objs, list) and all(isinstance(o, str) for o in objs):
            # **************************************************************************
            # FIX 2: Write to 'objectives' key
            features["objectives"] = tuple(sorted(objs))  # Use tuple
            # **************************************************************************
            features["num_objectives"] = len(objs)
        elif isinstance(objs, str):
            # **************************************************************************
            # FIX 2: Write to 'objectives' key
            features["objectives"] = tuple([objs])
            # **************************************************************************
            features["num_objectives"] = 1

        # Constraint features
        constraints = proposal.get("constraints")
        if isinstance(constraints, (dict, list)):
            features["has_constraints"] = True
            features["num_constraints"] = len(constraints)
            # Hash constraint keys? For now, just presence/count

        # Tradeoff features
        tradeoffs = proposal.get("tradeoffs")
        if isinstance(tradeoffs, (dict, list)):
            features["has_tradeoffs"] = True
            features["num_tradeoffs"] = len(tradeoffs)

        # Domain
        domain = proposal.get("domain")
        if isinstance(domain, str):
            features["domain"] = domain

        # Action type
        action = proposal.get("action")
        if isinstance(action, str):
            features["action_type"] = action

        # Weight diversity (already calculated to be float)
        weights = proposal.get("objective_weights")
        if isinstance(weights, dict):
            numeric_weights = [
                v for v in weights.values() if isinstance(v, (int, float))
            ]
            if numeric_weights:
                features["weight_diversity"] = self._calculate_weight_diversity(
                    numeric_weights
                )

        # Final check - ensure all values are basic hashable types
        # This shouldn't be strictly necessary if construction is correct
        final_features = {}
        for k, v in features.items():
            if isinstance(v, (str, int, float, bool, tuple, type(None))):
                final_features[k] = v
            # else: logger.debug(f"Skipping non-hashable feature value: {k}={type(v)}")

        return final_features

    def _incremental_pattern_update(self, record: ValidationRecord):
        """Update patterns incrementally with new record"""
        try:
            features = record.metadata.get(
                "features", self._extract_proposal_features(record.proposal)
            )  # Use stored features
            feature_key = tuple(sorted(features.items()))  # Hashable key
        except Exception as e:
            logger.error(
                f"Could not process features for incremental update (Proposal {record.proposal_id}): {e}"
            )
            return

        outcome = record.outcome
        logger.debug(
            f"Incremental update: Proposal {record.proposal_id}, Outcome {outcome.value}, Features Key Hash {hash(feature_key)}"
        )

        # Check if pattern with exact features exists
        matching_pattern = self.pattern_index.get(feature_key)

        if matching_pattern:
            logger.debug(
                f"  Updating existing pattern: type={matching_pattern.pattern_type.value}, features_hash={hash(feature_key)}"
            )
            pattern = matching_pattern  # Use the found pattern
            pattern.support += 1
            pattern.last_seen = record.timestamp
            if record.proposal_id not in pattern.examples:
                pattern.examples.append(record.proposal_id)
                if len(pattern.examples) > 10:
                    pattern.examples.pop(0)

            # Update Bayesian confidence
            alpha = pattern.metadata.get("alpha", 1.0)
            beta = pattern.metadata.get("beta", 1.0)
            if outcome == ValidationOutcome.APPROVED:
                alpha += 1.0
            else:
                beta += (
                    1.0  # Treat REJECTED/MODIFIED/UNKNOWN as evidence against approval
                )

            pattern.metadata["alpha"] = alpha
            pattern.metadata["beta"] = beta
            new_confidence = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            logger.debug(
                f"    Updated Bayesian params: alpha={alpha:.1f}, beta={beta:.1f}, Confidence={new_confidence:.3f} (was {pattern.confidence:.3f})"
            )
            pattern.confidence = new_confidence

            # Check if pattern type should flip based on new confidence
            current_is_success = pattern.pattern_type in [
                PatternType.SUCCESS,
                PatternType.SAFE,
            ]
            should_be_success = new_confidence > 0.5  # Simple threshold for type flip
            if current_is_success != should_be_success:
                new_type = (
                    PatternType.SUCCESS if should_be_success else PatternType.RISKY
                )
                logger.info(
                    f"  Flipping pattern type for features {hash(feature_key)}: {pattern.pattern_type.value} -> {new_type.value} (Confidence: {new_confidence:.3f})"
                )
                pattern.pattern_type = new_type

        else:  # Create new pattern
            if not features:  # Don't create patterns for empty features
                logger.debug("  Skipping new pattern creation: no features extracted.")
                return

            initial_alpha = 1.5
            initial_beta = 1.5  # Start slightly uncertain prior
            if outcome == ValidationOutcome.APPROVED:
                initial_alpha += 1.0
            else:
                initial_beta += 1.0

            initial_confidence = initial_alpha / (initial_alpha + initial_beta)
            pattern_type = (
                PatternType.SUCCESS if initial_confidence >= 0.5 else PatternType.RISKY
            )

            new_pattern = ValidationPattern(
                pattern_type=pattern_type,
                features=features,
                support=1,
                confidence=initial_confidence,
                examples=[record.proposal_id],
                first_seen=record.timestamp,
                last_seen=record.timestamp,
                metadata={"alpha": initial_alpha, "beta": initial_beta},
            )

            self.patterns.append(new_pattern)
            self.pattern_index[feature_key] = new_pattern  # Add to index
            logger.debug(
                f"  Created new pattern: type={new_pattern.pattern_type.value}, conf={new_pattern.confidence:.2f}, features_hash={hash(feature_key)}"
            )

    def _update_patterns_with_outcome(self, record: ValidationRecord):
        """Update patterns based on actual execution outcome ('success'/'failure')"""
        try:
            features = record.metadata.get(
                "features", self._extract_proposal_features(record.proposal)
            )
            feature_key = tuple(sorted(features.items()))
        except Exception as e:
            logger.error(
                f"Could not process features for outcome update (Proposal {record.proposal_id}): {e}"
            )
            return

        actual_success = record.actual_outcome == "success"
        logger.debug(
            f"Updating patterns with actual outcome for {record.proposal_id}: {'success' if actual_success else 'failure'}"
        )

        matching_pattern = self.pattern_index.get(feature_key)
        if matching_pattern:
            # Adjust Bayesian params based on actual outcome (stronger signal)
            alpha = matching_pattern.metadata.get("alpha", 1.0)
            beta = matching_pattern.metadata.get("beta", 1.0)
            update_strength = (
                2.0  # Give actual outcome more weight than initial validation
            )

            if actual_success:
                alpha += update_strength
            else:
                beta += update_strength

            matching_pattern.metadata["alpha"] = alpha
            matching_pattern.metadata["beta"] = beta
            new_confidence = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            logger.debug(
                f"  Adjusted pattern confidence for features {hash(feature_key)} to {new_confidence:.3f} based on actual outcome."
            )
            matching_pattern.confidence = new_confidence

            # Check if type needs flipping
            current_is_success = matching_pattern.pattern_type in [
                PatternType.SUCCESS,
                PatternType.SAFE,
            ]
            should_be_success = new_confidence > 0.5
            if current_is_success != should_be_success:
                new_type = (
                    PatternType.SUCCESS if should_be_success else PatternType.RISKY
                )
                logger.info(
                    f"  Flipping pattern type based on actual outcome for {hash(feature_key)}: {matching_pattern.pattern_type.value} -> {new_type.value}"
                )
                matching_pattern.pattern_type = new_type

        else:
            logger.debug(
                f"  No existing pattern found for features {hash(feature_key)} to update with actual outcome."
            )
            # Optionally create a new pattern based on actual outcome here? Or wait for incremental? Wait.

    def _comprehensive_relearn(self):
        """Perform comprehensive relearning from all history"""
        with self.lock:  # Ensure exclusive access during rebuild
            logger.info(
                "Starting comprehensive relearning from %d validation records...",
                len(self.validation_records),
            )
            start_time = time.time()

            # Rebuild patterns from scratch using all records
            self._rebuild_patterns()

            # Re-detect blockers from scratch using all records
            self.detect_blockers_from_history()

            end_time = time.time()
            logger.info(
                "Comprehensive relearning complete in %.2f seconds: %d patterns, %d blockers found.",
                end_time - start_time,
                len(self.patterns),
                len(self.blockers),
            )

    def _rebuild_patterns(self):
        """Rebuild all patterns from validation history"""
        # Clear existing patterns and index
        self.patterns.clear()
        self.pattern_index.clear()
        logger.debug("Cleared existing patterns and index for rebuild.")

        # Group records by feature key
        feature_groups = defaultdict(list)
        processed_count = 0
        history_snapshot = list(self.validation_records)  # Snapshot for iteration

        for record in history_snapshot:
            try:
                # Use stored features if available, else extract
                features = record.metadata.get("features")
                if features is None:
                    features = self._extract_proposal_features(record.proposal)
                # Important: Create the hashable key consistently
                feature_key = tuple(sorted(features.items()))
                feature_groups[feature_key].append(record)
                processed_count += 1
            except Exception as e:
                logger.warning(
                    f"Skipping record {record.proposal_id} during pattern rebuild due to feature error: {e}"
                )

        logger.debug(
            f"Grouped {processed_count} records into {len(feature_groups)} unique feature sets."
        )

        # Create/Update patterns from groups
        new_patterns = []
        new_pattern_index = {}
        patterns_created = 0
        for feature_key, records in feature_groups.items():
            if len(records) >= self.min_pattern_support:
                features = dict(feature_key)  # Convert key back to dict

                # Bayesian update across all records for this feature set
                alpha = 1.0
                beta = 1.0  # Priors
                record_ids_for_pattern = []
                timestamps = []

                for r in records:
                    # Prefer actual outcome if available, else use initial validation outcome
                    outcome_used = r.outcome  # Default to initial
                    if r.actual_outcome:
                        outcome_used = (
                            ValidationOutcome.APPROVED
                            if r.actual_outcome == "success"
                            else ValidationOutcome.REJECTED
                        )

                    if outcome_used == ValidationOutcome.APPROVED:
                        alpha += 1.0
                    else:
                        beta += 1.0
                    record_ids_for_pattern.append(r.proposal_id)
                    timestamps.append(r.timestamp)

                confidence = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                support = len(records)

                # Determine pattern type based on final confidence
                # Stricter threshold for RISKY patterns?
                if confidence > 0.6:
                    pattern_type = PatternType.SUCCESS
                elif confidence < 0.4:
                    pattern_type = PatternType.RISKY
                else:
                    # Skip patterns in the uncertain middle ground during rebuild? Or classify? Classify for now.
                    continue
                    # pattern_type = PatternType.SUCCESS if confidence >= 0.5 else PatternType.RISKY
                    # logger.debug(f"Pattern for {hash(feature_key)} has uncertain confidence ({confidence:.2f}), skipping.")
                    # continue

                pattern = ValidationPattern(
                    pattern_type=pattern_type,
                    features=features,
                    support=support,
                    confidence=confidence,
                    examples=record_ids_for_pattern[:10],  # Keep examples
                    first_seen=min(timestamps) if timestamps else time.time(),
                    last_seen=max(timestamps) if timestamps else time.time(),
                    metadata={"alpha": alpha, "beta": beta},  # Store Bayesian params
                )
                new_patterns.append(pattern)
                new_pattern_index[feature_key] = (
                    pattern  # Map key to the single pattern
                )
                patterns_created += 1

        # Atomically replace old patterns/index with new ones
        self.patterns = new_patterns
        self.pattern_index = new_pattern_index
        logger.debug(f"Rebuilt {patterns_created} patterns.")

    # --- _find_similar_records, _calculate_feature_similarity ---
    # (Copied from previous version)
    def _find_similar_records(
        self, record: ValidationRecord, threshold: float = 0.7
    ) -> List[ValidationRecord]:
        """Find records similar to given record"""
        try:
            features = record.metadata.get(
                "features", self._extract_proposal_features(record.proposal)
            )
        except Exception:
            return []  # Cannot find similar if features fail

        similar = []
        history_snapshot = list(self.validation_records)
        for other_record in history_snapshot:
            if other_record.proposal_id == record.proposal_id:
                continue
            try:
                other_features = other_record.metadata.get(
                    "features", self._extract_proposal_features(other_record.proposal)
                )
                similarity = self._calculate_feature_similarity(
                    features, other_features
                )
                if similarity >= threshold:
                    similar.append(other_record)
            except Exception:
                continue  # Skip if feature extraction fails

        return similar

    def _calculate_feature_similarity(
        self, features_a: Dict[str, Any], features_b: Dict[str, Any]
    ) -> float:
        """Calculate similarity between feature sets using Jaccard index on key-value pairs"""
        if not features_a or not features_b:
            return 0.0

        # Convert feature dicts to sets of (key, value) tuples for Jaccard
        try:
            # Ensure values are hashable for set conversion
            set_a = set(
                (
                    k,
                    (
                        v
                        if isinstance(v, (str, int, float, bool, tuple, type(None)))
                        else str(v)
                    ),
                )
                for k, v in features_a.items()
            )
            set_b = set(
                (
                    k,
                    (
                        v
                        if isinstance(v, (str, int, float, bool, tuple, type(None)))
                        else str(v)
                    ),
                )
                for k, v in features_b.items()
            )
        except Exception as e:
            logger.warning(
                f"Could not create hashable sets for feature similarity: {e}"
            )
            return 0.0  # Cannot calculate similarity

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return intersection / union if union > 0 else 0.0

    # --- _features_match_pattern ---
    # (Copied from previous corrected version)
    def _features_match_pattern(
        self,
        features: Dict[str, Any],  # Proposal features
        pattern_features: Dict[str, Any],
    ) -> bool:
        """Check if proposal features contain all pattern features with the same values."""
        if not pattern_features:  # An empty pattern matches nothing by definition
            # logger.debug("    Skipping empty pattern features.")
            return False

        for key, pattern_value in pattern_features.items():
            if key not in features:
                # logger.debug(f"    Mismatch: Pattern key '{key}' not in proposal features.")
                return False  # Proposal must have all keys from pattern

            prop_value = features[key]

            # Consistent comparison (handle potential type diffs like list vs tuple)
            try:
                # Convert lists/tuples to sorted tuples for comparison
                if isinstance(prop_value, list):
                    prop_value = tuple(sorted(prop_value))
                if isinstance(pattern_value, list):
                    pattern_value = tuple(sorted(pattern_value))
                # Convert tuple values similarly if needed (though features should ideally be flattened)
                if isinstance(prop_value, tuple):
                    prop_value = tuple(sorted(prop_value))
                if isinstance(pattern_value, tuple):
                    pattern_value = tuple(sorted(pattern_value))

                if prop_value != pattern_value:
                    # logger.debug(f"    Mismatch: Key '{key}', proposal='{prop_value}' ({type(prop_value)}) != pattern='{pattern_value}' ({type(pattern_value)})")
                    return False
            except TypeError:
                # Fallback to string comparison if direct comparison fails
                if str(prop_value) != str(pattern_value):
                    # logger.debug(f"    Mismatch (str fallback): Key '{key}', proposal='{str(prop_value)}' != pattern='{str(pattern_value)}'")
                    return False

        # If loop completes, all pattern features are present and match
        # logger.debug(f"    Match found for pattern features: {pattern_features}")
        return True

    # --- _analyze_temporal_pattern, _analyze_objective_performance ---
    # (Copied from previous version, ensuring self._np usage)
    def _analyze_temporal_pattern(
        self, timestamps: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Analyze temporal pattern in timestamps"""
        _np = self._np
        if len(timestamps) < 10:
            return None
        timestamps_sorted = sorted(timestamps)
        # Use np.diff (or fake version)
        intervals = _np.diff(timestamps_sorted)
        if len(intervals) < 1:
            return None

        mean_interval = _np.mean(intervals)
        std_interval = _np.std(intervals)  # Use np.std or fake version

        regularity = 0.0
        if mean_interval > 1e-9:  # Avoid division by zero
            regularity = 1.0 / (1.0 + std_interval / mean_interval)

        return {
            "mean_interval": float(mean_interval),
            "std_interval": float(std_interval),
            "regularity": float(regularity),
        }

    def _analyze_objective_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance (approval/failure rate) by objective"""
        objective_stats = defaultdict(
            lambda: {"total": 0, "approved": 0, "rejected": 0}
        )
        history_snapshot = list(self.validation_records)  # Snapshot

        for record in history_snapshot:
            # Use stored features for consistency
            features = record.metadata.get("features", {})
            objectives_in_proposal = set()
            if "objective" in features:
                objectives_in_proposal.add(features["objective"])
            # Assumes 'objectives_tuple' exists if multiple objectives were present
            # **************************************************************************
            # FIX: Read from 'objectives' key
            if "objectives" in features:
                objectives_in_proposal.update(features["objectives"])
            # **************************************************************************

            # Determine outcome (prefer actual if available)
            outcome = record.outcome
            if record.actual_outcome:
                outcome = (
                    ValidationOutcome.APPROVED
                    if record.actual_outcome == "success"
                    else ValidationOutcome.REJECTED
                )

            for obj in objectives_in_proposal:
                if not obj or not isinstance(obj, str):
                    continue  # Skip invalid
                objective_stats[obj]["total"] += 1
                if outcome == ValidationOutcome.APPROVED:
                    objective_stats[obj]["approved"] += 1
                else:  # REJECTED or UNKNOWN or MODIFIED
                    objective_stats[obj]["rejected"] += 1

        result = {}
        for obj, stats in objective_stats.items():
            total = stats["total"]
            if total > 0:
                result[obj] = {
                    "total": total,
                    "approval_rate": stats["approved"] / total,
                    "failure_rate": stats["rejected"] / total,
                }
        return result

    # --- detect_blockers_from_history, _classify_blocker, _suggest_solutions ---
    # (Copied from previous version)
    def detect_blockers_from_history(self):
        """Detect blockers by analyzing rejected proposals"""
        with self.lock:
            logger.debug("Detecting blockers from history...")
            rejected = [
                r
                for r in self.validation_records
                if r.outcome == ValidationOutcome.REJECTED
            ]
            blockers_by_objective = defaultdict(
                lambda: defaultdict(list)
            )  # obj -> type -> instances

            for record in rejected:
                features = record.metadata.get("features", {})
                objectives_in_proposal = set()
                if "objective" in features:
                    objectives_in_proposal.add(features["objective"])
                if "objectives" in features:
                    objectives_in_proposal.update(features["objectives"])

                reasoning = self._safe_get_reasoning(record.validation_result)
                if reasoning:
                    blocker_type = self._classify_blocker(reasoning)
                    if blocker_type != "unknown":
                        instance = {
                            "reasoning": reasoning,
                            "proposal_id": record.proposal_id,
                        }
                        for obj in objectives_in_proposal:
                            if obj and isinstance(
                                obj, str
                            ):  # Ensure valid objective name
                                blockers_by_objective[obj][blocker_type].append(
                                    instance
                                )

            # Create blocker objects, replacing existing ones
            new_blockers = []
            new_blocker_index = defaultdict(list)
            total_record_count = len(
                self.validation_records
            )  # Use total count for severity normalization

            for obj, types in blockers_by_objective.items():
                # Calculate total attempts for this objective to normalize severity
                obj_attempts = sum(
                    1
                    for r in self.validation_records
                    if obj in r.metadata.get("features", {}).get("objectives", [])
                    or r.metadata.get("features", {}).get("objective") == obj
                )

                for blocker_type, instances in types.items():
                    frequency = len(instances)
                    if (
                        frequency >= self.min_pattern_support
                    ):  # Only consider blockers with minimum support
                        # Severity = frequency / total attempts for this objective
                        severity_score = frequency / max(1, obj_attempts)

                        descriptions = [inst["reasoning"] for inst in instances]
                        most_common_desc = (
                            Counter(descriptions).most_common(1)[0][0]
                            if descriptions
                            else f"{blocker_type} blocks {obj}"
                        )

                        blocker = ObjectiveBlocker(
                            objective=obj,
                            blocker_type=blocker_type,
                            description=most_common_desc,
                            frequency=frequency,
                            severity=severity_score,
                            examples=[i["proposal_id"] for i in instances[:5]],
                            potential_solutions=self._suggest_solutions(
                                blocker_type, obj
                            ),
                        )
                        new_blockers.append(blocker)
                        new_blocker_index[obj].append(blocker)

            # Atomically update blockers
            self.blockers = new_blockers
            self.blocker_index = new_blocker_index
            logger.debug(f"Detected {len(self.blockers)} blockers.")

    def _classify_blocker(self, reasoning: str) -> str:
        """Classify blocker type from reasoning string"""
        if not isinstance(reasoning, str):
            return "unknown"
        r = reasoning.lower()
        # Order checks from specific to general
        if "constraint violation" in r or "violates constraint" in r:
            if "minimum" in r:
                return "constraint_violation_min"
            if "maximum" in r:
                return "constraint_violation_max"
            return "constraint_violation"
        # **************************************************************************
        # FIX: Check for both keywords
        if "objective" in r and "conflict" in r:
            return "objective_conflict"
        # **************************************************************************
        if "goal drift" in r:
            return "goal_drift"
        if "unacceptable tradeoff" in r:
            return "unacceptable_tradeoff"
        if "risky pattern" in r:
            return "risky_pattern_match"
        if "predicted failure" in r or ("rejected" in r and "historical pattern" in r):
            return "predicted_failure"
        if "pathology" in r:
            return "objective_pathology"  # Generic pathology
        if "insufficient resource" in r:
            return "insufficient_resources"
        if "insufficient data" in r:
            return "insufficient_data"
        if "insufficient" in r or "lacking" in r:
            return "insufficient_info"  # Generic insufficient
        return "unknown"

    def _suggest_solutions(self, blocker_type: str, objective: str) -> List[str]:
        """Suggest potential solutions based on blocker type"""
        # (Using the comprehensive map from previous version)
        solutions_map = {
            "objective_conflict": [
                "Use multi-objective optimization (e.g., Pareto)",
                "Prioritize objectives explicitly",
                "Negotiate compromise",
            ],
            "constraint_violation": [
                f"Relax non-critical constraints for {objective}",
                f"Reformulate proposal for {objective}",
                f"Check feasibility for {objective} within limits",
            ],
            "constraint_violation_min": [
                f"Ensure actions respect minimum for {objective}",
                f"Adjust proposal to increase {objective} value",
            ],
            "constraint_violation_max": [
                f"Ensure actions respect maximum for {objective}",
                f"Adjust proposal to decrease {objective} value",
            ],
            "goal_drift": [
                "Realign proposal with core design spec",
                "Clarify objective definitions",
                "Add drift monitoring",
            ],
            "insufficient_resources": [
                f"Allocate more resources towards {objective}",
                "Optimize resource usage",
                f"Adjust target for {objective}",
            ],
            "insufficient_data": [
                f"Gather more data for {objective}",
                f"Initiate exploration for {objective} data",
            ],
            "objective_pathology": [
                f"Review proposal for {objective} against spec",
                "Analyze historical failures",
            ],
            "unacceptable_tradeoff": [
                f"Re-evaluate tradeoff involving {objective}",
                "Prioritize critical objectives",
            ],
            "risky_pattern_match": [
                f"Analyze why pattern fails for {objective}",
                "Modify proposal to avoid risky features",
            ],
            "predicted_failure": [
                f"Review historical failures for {objective}",
                "Modify proposal based on past failures",
            ],
            "insufficient_info": [
                f"Gather more information relevant to {objective} or proposal context"
            ],
            "unknown": [f"Investigate root cause of failures related to {objective}"],
        }
        return solutions_map.get(blocker_type, solutions_map["unknown"])

    # --- get_statistics ---
    # (Copied from previous corrected version)
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        with self.lock:
            total_validations = self.stats.get("total_validations", 0)
            approved_validations = self.stats.get("outcome_approved", 0)
            # Calculate actual success rate if outcomes recorded
            actual_successes = self.stats.get("actual_outcome_success", 0)
            actual_failures = self.stats.get("actual_outcome_failure", 0)
            total_actuals = actual_successes + actual_failures

            stats_snapshot = dict(self.stats)  # Snapshot defaultdict

            stats_output = {
                "total_records": len(self.validation_records),
                "total_validations_processed": total_validations,
                "patterns_learned": len(self.patterns),
                "blockers_identified": len(self.blockers),
                "approval_rate": (
                    (approved_validations / total_validations)
                    if total_validations > 0
                    else 0.0
                ),
                "actual_success_rate": (
                    (actual_successes / total_actuals) if total_actuals > 0 else None
                ),  # Rate based on actuals
                "raw_stats": stats_snapshot,  # Include raw counts
            }

            # Add CSIU status safely
            csiu_enabled = False
            if not isinstance(self.self_improvement_drive, MagicMock):
                csiu_enabled = getattr(
                    self.self_improvement_drive, "_csiu_enabled", False
                )
            stats_output["csiu_tracking_enabled"] = csiu_enabled
            if csiu_enabled:
                stats_output["csiu_log_interval"] = self.csiu_log_interval
                stats_output["last_csiu_log_at_count"] = self.last_csiu_log

            return stats_output

    # --- _make_serializable ---
    # (Copied from previous corrected version, uses self._np)
    def _make_serializable(self, data: Any) -> Any:
        """Recursively make data JSON serializable, handling common types."""
        _np = self._np  # Use internal alias

        # **************************************************************************
        # FIX: Handle Mock objects FIRST to prevent recursion
        if hasattr(data, "_extract_mock_name"):
            return f"<MagicMock name='{data._extract_mock_name()}' id='{id(data)}'>"
        # **************************************************************************

        if isinstance(data, dict):
            # Handle Mocks that might act like dicts
            # This check is now secondary, which is fine
            if hasattr(data, "_extract_mock_name"):
                return f"<MagicMock name='{data._extract_mock_name()}' id='{id(data)}'>"
            return {
                self._make_serializable(k): self._make_serializable(v)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple, deque)):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, set):
            return sorted([self._make_serializable(item) for item in data])
        elif isinstance(data, Enum):
            return data.value
        elif hasattr(data, "to_dict") and callable(data.to_dict):
            try:
                return self._make_serializable(data.to_dict())  # Recurse on result
            except Exception:
                return str(data)  # Fallback
        # Handle dataclasses AFTER to_dict check
        elif is_dataclass(data) and not isinstance(data, type):
            try:
                return self._make_serializable(asdict(data))  # Recurse on result
            except Exception:
                return str(data)  # Fallback
        elif isinstance(data, (str, int, bool)) or data is None:
            return data
        elif isinstance(data, float):  # Handle NaN/inf
            return str(data) if math.isnan(data) or math.isinf(data) else data
        # Use NUMPY_AVAILABLE flag for numpy types
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return data.tolist()
        elif NUMPY_AVAILABLE and isinstance(data, np.generic):
            # Ensure item() returns basic python type
            item = data.item()
            if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                return str(item)
            return item
        # Handle standalone Mocks
        elif hasattr(data, "_extract_mock_name"):
            return f"<MagicMock name='{data._extract_mock_name()}' id='{id(data)}'>"
        else:  # Fallback: try str()
            try:
                return str(data)
            except Exception:
                return f"<unserializable_{type(data).__name__}>"


# Need math for float checks in _make_serializable
# Need asdict, is_dataclass if using dataclasses and not handled by to_dict
