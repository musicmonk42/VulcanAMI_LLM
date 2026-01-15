# src/vulcan/world_model/meta_reasoning/transparency_interface.py
"""
transparency_interface.py - Machine-readable transparency for agent communication
Part of the meta_reasoning subsystem for VULCAN-AMI

Provides structured, machine-readable output for agent-to-agent communication.
NOT natural language explanations - precise data structures that agents can
parse, reason about, and act upon.

Enables:
- Validation result serialization for consensus voting
- Objective state broadcasting
- Conflict documentation for automated resolution
- Negotiation outcome audit trails
- Graphix IR consensus integration
- CSIU internal audit trails (for maintainers only, not exposed to UX)

GUARDS (2025-10-18):
- Hard guards against CSIU leakage to user-facing serializers
- Maintainer-only longitudinal CSIU accessor
- Longitudinal trends attached to internal audits

FIX (2025-10-22):
- Implemented robust serialization in `serialize_validation` to handle potential Mock objects and Enums correctly.
- Added `_make_serializable` helper function.
- Ensured consistent use of safe float conversion.
"""

import math
import hashlib
import json
import logging
import threading  # ADDED for RLock
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum

# import numpy as np # Original import
from typing import Any, Callable, Dict, List, Optional, Set, Union  # Add Union here
from unittest.mock import MagicMock  # ADDED for fallback

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

        # Add generic type placeholder if needed elsewhere, though not directly used here
        class generic:
            pass

        # Add ndarray type placeholder if needed elsewhere
        class ndarray:
            pass

        # Add item method for scalars if needed (though FakeNumpy won't produce np scalars)
        # item = lambda x: x # Simplistic item method

        # Add other necessary numpy functions if used later
        @staticmethod
        def sqrt(x):
            import math

            return math.sqrt(x) if x >= 0 else float("nan")

        @staticmethod
        def log(x):
            import math

            if isinstance(x, list):
                return [math.log(i) if i > 0 else -float("inf") for i in x]
            return math.log(x) if x > 0 else -float("inf")

    np = FakeNumpy()
# --- END FIX ---


# Import necessary types for type checking if possible, handle gracefully if not
try:
    # Use real imports if available
    from .motivational_introspection import (
        ObjectiveAnalysis,
        ObjectiveStatus,
        ProposalValidation,
    )

    MOTIVATIONAL_INTROSPECTION_AVAILABLE = True
except ImportError:
    # Define dummy types if imports fail (e.g., during standalone testing)
    MOTIVATIONAL_INTROSPECTION_AVAILABLE = False
    logger.warning(
        "Failed to import types from motivational_introspection. Using dummy types."
    )

    class ProposalValidation:
        pass

    class ObjectiveAnalysis:
        pass

    # **************************************************************************
    # FIX 1: Add UNKNOWN to the fallback ObjectiveStatus enum
    class ObjectiveStatus(Enum):
        ALIGNED = "aligned"
        CONFLICT = "conflict"
        VIOLATION = "violation"
        DRIFT = "drift"
        ACCEPTABLE = "acceptable"
        UNKNOWN = "unknown"

    # **************************************************************************


# **************************************************************************
# FIX 2: Add fallback ViolationSeverity enum
class ViolationSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# **************************************************************************


class SerializationFormat(Enum):
    """Output format for serialization"""

    JSON = "json"
    COMPACT = "compact"
    VERBOSE = "verbose"


@dataclass
class TransparencyMetadata:
    """Metadata for transparency output"""

    version: str = "1.0"
    timestamp: float = 0.0
    signature: Optional[str] = None
    source: str = "vulcan_ami"

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# Constants for input validation in explain_decision and related methods
MAX_DECISION_SUMMARY_LENGTH = 500  # Maximum length for decision summaries
MAX_FACTOR_VALUE_LENGTH = 100  # Maximum length for individual factor values
MAX_REASONING_STEP_LENGTH = 200  # Maximum length for reasoning steps


class TransparencyInterface:
    """
    Structured, machine-readable output for agent-to-agent communication

    This is NOT for human consumption. It's for agents to:
    - Parse validation results and make decisions
    - Understand current objective state
    - Reason about conflicts programmatically
    - Audit negotiation outcomes
    - Participate in consensus voting

    All output is structured data (dicts, lists) that can be:
    - Serialized to JSON
    - Hashed for verification
    - Transmitted over network
    - Parsed by other agents
    """

    # --- START REPLACEMENT ---
    def __init__(self, world_model: Optional["WorldModel"] = None):
        """
        Initialize the transparency/audit interface.
        Args:
            world_model: Reference to the parent WorldModel (optional but recommended
                         for full logging of causal-graph / prediction-engine state).
        """
        self.world_model = world_model

        # --- Infer introspection_engine and self_improvement_drive from world_model ---
        if world_model and hasattr(world_model, "motivational_introspection"):
            mi = getattr(world_model, "motivational_introspection", None)
            logger.debug(
                f"TransparencyInterface: motivational_introspection found, type={type(mi).__name__}"
            )
            self.introspection_engine = mi
        else:
            logger.debug(
                f"TransparencyInterface: world_model={world_model is not None}, hasattr={hasattr(world_model, 'motivational_introspection') if world_model else False}"
            )
            logger.warning(
                "TransparencyInterface: world_model or motivational_introspection not provided. Using MagicMock."
            )
            self.introspection_engine = MagicMock()

        if world_model and hasattr(world_model, "self_improvement_drive"):
            self.self_improvement_drive = (
                world_model.self_improvement_drive or MagicMock(_csiu_enabled=False)
            )
        else:
            self.self_improvement_drive = MagicMock(_csiu_enabled=False)

        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        # Serialization cache
        self.cache = {}
        self.cache_ttl = 60  # 1 minute
        
        # Configuration for explain_decision confidence
        self.default_explanation_confidence = 0.75  # Configurable default

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = 1000  # Increased size from 10

        # Statistics
        self.stats = defaultdict(int)

        # Schema version
        self.schema_version = "2.1"

        # Thread safety lock
        self.lock = threading.RLock()
        self.explain_cache: Dict[str, Any] = {}

        logger.info("TransparencyInterface initialized")

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare state for pickling by removing unpickleable lock objects.
        """
        state = self.__dict__.copy()
        state.pop('lock', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore state after unpickling, re-creating the lock.
        """
        self.__dict__.update(state)
        self.lock = threading.RLock()

    # --- END REPLACEMENT ---

    # --- START REPLACEMENT for _make_serializable ---
    def _make_serializable(self, data: Any, seen: Optional[Set[int]] = None) -> Any:
        """
        Recursively make data JSON serializable, handling common types and circular references.

        Args:
            data: The data to serialize.
            seen: A set of object IDs already processed in this path to detect cycles.
        """
        _np = self._np  # Use internal alias

        if seen is None:
            seen = set()

        # For mutable objects, check if we've seen them before to prevent cycles
        # Use id() for tracking, as it's unique for the object's lifetime
        try:
            data_id = id(data)
            is_hashable = True
        except TypeError:
            is_hashable = False  # Not all objects (like some mocks) are hashable or have a stable id in all contexts
            data_id = None

        if is_hashable and not isinstance(
            data, (str, int, float, bool, type(None), Enum)
        ):
            if data_id in seen:
                # Return a placeholder for circular references
                return f"<circular_ref to {type(data).__name__} at {hex(data_id)}>"
            seen.add(data_id)

        try:
            if isinstance(data, dict):
                if hasattr(data, "_extract_mock_name") and callable(
                    data._extract_mock_name
                ):
                    return f"<MagicMock name='{data._extract_mock_name()}' id='{id(data)}'>"
                # Recursively process dict items, passing the 'seen' set
                return {
                    self._make_serializable(k, seen.copy()): self._make_serializable(
                        v, seen.copy()
                    )
                    for k, v in data.items()
                }  # Serialize keys too
            elif (
                isinstance(data, list)
                or isinstance(data, tuple)
                or isinstance(data, deque)
            ):
                # Recursively process list/tuple/deque items
                return [self._make_serializable(item, seen.copy()) for item in data]
            elif isinstance(data, set):
                # Convert set to sorted list for consistent serialization
                return sorted(
                    [self._make_serializable(item, seen.copy()) for item in data]
                )
            elif isinstance(data, Enum):
                return data.value
            elif is_dataclass(data) and not isinstance(data, type):
                try:
                    # Use asdict for dataclasses, then clean the resulting dict recursively
                    return self._make_serializable(asdict(data), seen.copy())
                except Exception as e:
                    logger.debug(
                        f"Error calling asdict on {type(data)}: {e}, falling back to string."
                    )
                    # Fallback: serialize fields manually
                    cleaned_dict = {}
                    if hasattr(data, "__dataclass_fields__"):
                        for f_name in data.__dataclass_fields__:
                            cleaned_dict[f_name] = self._make_serializable(
                                getattr(data, f_name), seen.copy()
                            )
                    return cleaned_dict
            elif hasattr(data, "to_dict") and callable(data.to_dict):
                try:
                    # Use object's own serialization if available, then clean recursively
                    return self._make_serializable(data.to_dict(), seen.copy())
                except Exception as e:
                    logger.debug(
                        f"Error calling to_dict on {type(data)}: {e}, falling back to string."
                    )
                    try:
                        return str(data)
                    except Exception:
                        return f"<unserializable_{type(data).__name__}>"
            elif isinstance(data, (str, int, bool)) or data is None:
                return data
            elif isinstance(data, float):
                if math.isnan(data) or math.isinf(data):
                    return str(data)  # Represent NaN/inf as strings
                return data
            elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                return data.tolist()
            elif NUMPY_AVAILABLE and isinstance(data, np.generic):
                return data.item()
            elif hasattr(data, "_extract_mock_name") and callable(
                data._extract_mock_name
            ):
                return f"<MagicMock name='{data._extract_mock_name()}' id='{id(data)}'>"
            else:
                try:
                    return str(data)
                except Exception:
                    return f"<unserializable_{type(data).__name__}>"
        finally:
            # Remove from seen set when done processing this branch
            if is_hashable and data_id is not None:
                seen.discard(data_id)

    # --- END REPLACEMENT for _make_serializable ---

    def serialize_validation(
        self, validation_result: Union[ProposalValidation, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Convert validation to standardized machine-readable format

        Args:
            validation_result: ProposalValidation dataclass instance or a dictionary.

        Returns:
            Structured dictionary for agent consumption (cleaned for JSON serialization).
        """

        # FIXED: Use the _make_serializable helper to robustly convert the input
        # This handles dataclasses, dicts, enums, mocks, numpy types etc.
        try:
            # Pass None for 'seen' to start a new cycle detection
            validation_dict = self._make_serializable(validation_result, seen=None)
            # Ensure the result is actually a dictionary after serialization
            if not isinstance(validation_dict, dict):
                logger.error(
                    f"Serialization of validation result did not yield a dictionary (got {type(validation_dict)}). Result: {validation_dict}"
                )
                # Fallback to a minimal error structure
                validation_dict = {
                    "proposal_id": "serialization_error",
                    "valid": False,
                    "reasoning": f"Serialization failed, original type: {type(validation_result)}",
                }
        except Exception as e:
            logger.error(
                f"Critical error during validation serialization: {e}", exc_info=True
            )
            validation_dict = {
                "proposal_id": "serialization_error",
                "valid": False,
                "reasoning": f"Critical serialization error: {e}",
            }

        # Ensure essential keys exist using safe .get() with defaults
        # Handle potential nesting (e.g., if input was {'validation': {...}})
        validation_data = validation_dict.get("validation", validation_dict)
        # Ensure validation_data is a dict before proceeding
        if not isinstance(validation_data, dict):
            logger.warning(
                f"Validation data within serialized result is not a dict: {type(validation_data)}. Using defaults."
            )
            validation_data = {}  # Fallback to empty dict

        proposal_id = validation_data.get("proposal_id", "unknown_id")
        valid = validation_data.get("valid", False)

        # **************************************************************************
        # START: FIX FOR AttributeError
        # Safely get status, default to ObjectiveStatus.UNKNOWN if available, else string
        default_status = "unknown"
        if ObjectiveStatus and hasattr(ObjectiveStatus, "UNKNOWN"):
            default_status = ObjectiveStatus.UNKNOWN.value
        elif ObjectiveStatus and hasattr(
            ObjectiveStatus, "unknown"
        ):  # Fallback for lowercase
            default_status = ObjectiveStatus.unknown.value

        overall_status_val = validation_data.get("overall_status", default_status)
        overall_status = str(overall_status_val)  # Ensure string
        # END: FIX FOR AttributeError
        # **************************************************************************

        confidence = self._safe_float(validation_data.get("confidence"), 0.0)
        reasoning = str(validation_data.get("reasoning", ""))  # Ensure string
        timestamp = self._safe_float(
            validation_data.get("timestamp"), time.time()
        )  # Ensure float
        objective_analyses = validation_data.get("objective_analyses", [])
        # Ensure objective_analyses is a list
        if not isinstance(objective_analyses, list):
            objective_analyses = []

        conflicts = validation_data.get(
            "conflicts_detected", validation_data.get("conflicts", [])
        )
        if not isinstance(conflicts, list):
            conflicts = []

        alternatives = validation_data.get(
            "alternatives_suggested", validation_data.get("alternatives", [])
        )
        if not isinstance(alternatives, list):
            alternatives = []

        # Create machine-readable structure using cleaned/defaulted values
        serialized = {
            "schema_version": self.schema_version,
            "type": "validation_result",
            "metadata": self._create_metadata("validation"),
            "validation": {
                "id": str(proposal_id),  # Ensure string ID
                "outcome": {
                    "valid": bool(valid),  # Ensure boolean
                    "status": overall_status,
                    "confidence": confidence,
                },
                # Nested lists/dicts were cleaned by _make_serializable initially
                "objectives": objective_analyses,
                "conflicts": self._serialize_conflicts_compact(
                    conflicts
                ),  # Further compact serialization
                "alternatives": self._serialize_alternatives_compact(
                    alternatives
                ),  # Further compact serialization
                "reasoning": {
                    "primary": reasoning,
                    # Pass the fully cleaned dict to extract structured reasoning
                    "structured": self._extract_structured_reasoning(validation_dict),
                },
            },
            # Pass the fully cleaned dict to extract actionable items
            "actionable": self._extract_actionable_items(validation_dict),
        }

        # Generate signature BEFORE adding internal audit data
        # Ensure the signature is based only on the public, serializable data
        # Clean again right before signing to ensure consistent structure
        try:
            signature_base = self._make_serializable(serialized, seen=None)
            signature = self._generate_signature(signature_base)
        except Exception as e:
            logger.error(f"Failed to generate signature for validation result: {e}")
            signature = "signature_error"
        serialized["signature"] = signature

        # Add internal audit with CSIU data (for maintainers only) - AFTER signature
        internal_audit = self._generate_internal_audit()
        audit_entry_data = (
            serialized.copy()
        )  # Copy data before adding internal audit for the log
        if internal_audit:
            # DO NOT add internal_audit to the returned 'serialized' dict destined for external agents
            # Only add it to the internal audit log entry
            audit_entry_data["internal_audit"] = internal_audit

        # GUARD: Hard guard against UX leakage - explicitly check and remove CSIU/internal data
        if "csiu" in serialized:
            logger.debug("Removing CSIU data from externally serializable output.")
            serialized.pop("csiu", None)
        if "internal_audit" in serialized:
            logger.debug(
                "Removing Internal audit data from externally serializable output."
            )
            serialized.pop("internal_audit", None)

        # Add to audit log (including internal data)
        self._audit("validation_serialized", audit_entry_data)

        self.stats["validations_serialized"] += 1

        return serialized  # Return the cleaned, externally safe version

    def serialize_objective_state(self) -> Dict[str, Any]:
        """
        Serialize current objective hierarchy, weights, constraints

        Returns:
            Complete objective state in machine-readable format
        """

        # Get current state from introspection engine (mock-safe)
        motivation_structure = {}
        if hasattr(self.introspection_engine, "explain_motivation_structure"):
            try:
                motivation_structure = (
                    self.introspection_engine.explain_motivation_structure()
                )
                # Ensure the result is a dict
                if not isinstance(motivation_structure, dict):
                    logger.warning(
                        "introspection_engine.explain_motivation_structure did not return a dict."
                    )
                    motivation_structure = {}
            except Exception as e:
                logger.error(f"Error calling explain_motivation_structure: {e}")
                motivation_structure = {}  # Fallback empty
        else:
            logger.warning(
                "introspection_engine missing 'explain_motivation_structure' method."
            )

        # Create machine-readable structure
        serialized = {
            "schema_version": self.schema_version,
            "type": "objective_state",
            "metadata": self._create_metadata("objective_state"),
            "objectives": {
                "active": self._serialize_active_objectives(
                    motivation_structure.get("objectives", {})
                ),
                "hierarchy": self._serialize_hierarchy(
                    motivation_structure.get("objectives", {}).get("hierarchy", {})
                ),
                "state": self._make_serializable(
                    motivation_structure.get("current_state", {}), seen=None
                ),  # Serialize state values
            },
            "constraints": self._serialize_constraints(),
        }

        # Generate signature BEFORE adding internal audit data
        try:
            signature_base = self._make_serializable(serialized, seen=None)
            signature = self._generate_signature(signature_base)
        except Exception as e:
            logger.error(f"Failed to generate signature for objective state: {e}")
            signature = "signature_error"
        serialized["signature"] = signature

        # Add internal audit with CSIU data (for maintainers only) - AFTER signature
        internal_audit = self._generate_internal_audit()
        audit_entry_data = serialized.copy()
        if internal_audit:
            audit_entry_data["internal_audit"] = internal_audit

        # GUARD: Hard guard against UX leakage - remove internal data from returned output
        if "csiu" in serialized:
            serialized.pop("csiu", None)
        if "internal_audit" in serialized:
            serialized.pop("internal_audit", None)

        # Add to audit log (including internal data)
        self._audit("objective_state_serialized", audit_entry_data)

        self.stats["objective_states_serialized"] += 1

        return serialized  # Return cleaned version

    def serialize_conflict(
        self, conflict_analysis: Union[Any, List[Any]]
    ) -> Dict[str, Any]:
        """
        Structured representation of objective conflicts

        Args:
            conflict_analysis: Conflict object/dict or list of conflicts.

        Returns:
            Structured conflict representation dict.
        """

        # Handle both single conflict and list, ensuring they are serializable
        if isinstance(conflict_analysis, list):
            conflicts_input = conflict_analysis
        # Check for common conflict dataclass/object structures before assuming dict
        elif hasattr(conflict_analysis, "to_dict") and callable(
            conflict_analysis.to_dict
        ):
            conflicts_input = [conflict_analysis]
        elif isinstance(conflict_analysis, dict):
            conflicts_input = [conflict_analysis]
        elif conflict_analysis is not None:  # Log unexpected types
            logger.warning(
                f"Unexpected conflict_analysis type: {type(conflict_analysis)}. Processing as empty list."
            )
            conflicts_input = []
        else:
            conflicts_input = []

        # Serialize each conflict robustly
        serialized_conflicts = []
        for conflict in conflicts_input:
            try:
                conflict_dict = self._make_serializable(
                    conflict, seen=None
                )  # Clean the input first
                if not isinstance(conflict_dict, dict):
                    logger.warning(
                        f"Serialized conflict is not a dict: {type(conflict_dict)}. Skipping."
                    )
                    continue

                # Ensure essential keys exist after cleaning, provide defaults
                objectives = conflict_dict.get("objectives", [])
                if not isinstance(objectives, list):
                    objectives = []  # Ensure list

                conflict_type_val = conflict_dict.get(
                    "conflict_type", conflict_dict.get("type", "unknown")
                )  # Handle both keys
                severity_val = conflict_dict.get("severity", "unknown")

                serialized_conflict = {
                    "objectives": [str(o) for o in objectives],  # Ensure strings
                    "type": str(conflict_type_val),  # Ensure string
                    "severity": str(severity_val),  # Ensure string
                    "description": str(
                        conflict_dict.get("description", "")
                    ),  # Ensure string
                    "quantitative_measure": self._safe_float(
                        conflict_dict.get("quantitative_measure"), None
                    ),  # Ensure float or None
                    "resolution_options": self._make_serializable(
                        conflict_dict.get("resolution_options", []), seen=None
                    ),  # Ensure serializable list
                    "metadata": self._make_serializable(
                        conflict_dict.get("metadata", {}), seen=None
                    ),  # Ensure serializable dict
                }
                serialized_conflicts.append(serialized_conflict)
            except Exception as e:
                logger.error(
                    f"Error serializing individual conflict: {e}. Conflict data: {conflict}",
                    exc_info=True,
                )

        # Create machine-readable structure
        serialized = {
            "schema_version": self.schema_version,
            "type": "conflict_analysis",
            "metadata": self._create_metadata("conflict"),
            "conflicts": serialized_conflicts,
            "summary": {
                "total_conflicts": len(serialized_conflicts),
                "by_severity": self._count_by_severity(serialized_conflicts),
                "by_type": self._count_by_type(serialized_conflicts),
                "requires_resolution": any(
                    # Check against potential string values from serialization
                    c.get("severity")
                    in [
                        "critical",
                        "high",
                        ViolationSeverity.CRITICAL.value,
                        ViolationSeverity.HIGH.value,
                    ]
                    for c in serialized_conflicts
                ),
            },
            "resolution_strategy": self._recommend_resolution_strategy(
                serialized_conflicts
            ),
        }

        # Generate signature BEFORE adding internal audit data
        try:
            signature_base = self._make_serializable(serialized, seen=None)
            signature = self._generate_signature(signature_base)
        except Exception as e:
            logger.error(f"Failed to generate signature for conflict analysis: {e}")
            signature = "signature_error"
        serialized["signature"] = signature

        # Add internal audit with CSIU data (for maintainers only) - AFTER signature
        internal_audit = self._generate_internal_audit()
        audit_entry_data = serialized.copy()
        if internal_audit:
            audit_entry_data["internal_audit"] = internal_audit

        # GUARD: Hard guard against UX leakage - remove internal data from returned output
        if "csiu" in serialized:
            serialized.pop("csiu", None)
        if "internal_audit" in serialized:
            serialized.pop("internal_audit", None)

        # Add to audit log (including internal data)
        self._audit("conflict_serialized", audit_entry_data)

        self.stats["conflicts_serialized"] += 1

        return serialized  # Return cleaned version

    def serialize_negotiation_outcome(self, negotiation: Any) -> Dict[str, Any]:
        """
        Document negotiation outcome for audit trail

        Args:
            negotiation: NegotiationResult dataclass instance or similar dict.

        Returns:
            Structured negotiation documentation dict.
        """

        # Convert negotiation result to dict robustly
        try:
            negotiation_dict = self._make_serializable(negotiation, seen=None)
            if not isinstance(negotiation_dict, dict):
                logger.error(
                    f"Serialization of negotiation result did not yield a dict: {type(negotiation_dict)}"
                )
                negotiation_dict = {
                    "outcome": "serialization_error",
                    "reasoning": "Serialization failed",
                }
        except Exception as e:
            logger.error(
                f"Critical error during negotiation serialization: {e}", exc_info=True
            )
            negotiation_dict = {
                "outcome": "serialization_error",
                "reasoning": f"Critical error: {e}",
            }

        # Create machine-readable structure using safe .get()
        serialized = {
            "schema_version": self.schema_version,
            "type": "negotiation_outcome",
            "metadata": self._create_metadata("negotiation"),
            "outcome": {
                "status": str(negotiation_dict.get("outcome", "unknown")),
                "pareto_optimal": bool(negotiation_dict.get("pareto_optimal", False)),
                "confidence": self._safe_float(negotiation_dict.get("confidence"), 0.0),
            },
            "agreement": {
                "objectives": self._make_serializable(
                    negotiation_dict.get("agreed_objectives", {}), seen=None
                ),  # Serialize nested dict
                "weights": self._make_serializable(
                    negotiation_dict.get("objective_weights", {}), seen=None
                ),  # Serialize nested dict
                "participants": self._make_serializable(
                    negotiation_dict.get("participating_agents", []), seen=None
                ),  # Serialize list
            },
            "process": {
                "strategy": str(negotiation_dict.get("strategy_used", "unknown")),
                "iterations": int(negotiation_dict.get("iterations", 0)),
                "convergence_time_ms": self._safe_float(
                    negotiation_dict.get("convergence_time_ms"), 0.0
                ),
                "compromises": self._make_serializable(
                    negotiation_dict.get("compromises_made", []), seen=None
                ),  # Serialize list
            },
            "rationale": {
                "reasoning": str(negotiation_dict.get("reasoning", "")),
                "structured": self._extract_structured_reasoning(negotiation_dict),
            },
        }

        # Generate signature BEFORE adding internal audit data
        try:
            signature_base = self._make_serializable(serialized, seen=None)
            signature = self._generate_signature(signature_base)
        except Exception as e:
            logger.error(f"Failed to generate signature for negotiation outcome: {e}")
            signature = "signature_error"
        serialized["signature"] = signature

        # Add internal audit with CSIU data (for maintainers only) - AFTER signature
        internal_audit = self._generate_internal_audit()
        audit_entry_data = serialized.copy()
        if internal_audit:
            audit_entry_data["internal_audit"] = internal_audit

        # GUARD: Hard guard against UX leakage - remove internal data from returned output
        if "csiu" in serialized:
            serialized.pop("csiu", None)
        if "internal_audit" in serialized:
            serialized.pop("internal_audit", None)

        # Add to audit log (including internal data)
        self._audit("negotiation_serialized", audit_entry_data)

        self.stats["negotiations_serialized"] += 1

        return serialized  # Return cleaned version

    # --- Methods below are largely unchanged but ensure use of self._np ---

    def export_for_consensus(self) -> Dict[str, Any]:
        """Format for Graphix IR's consensus mechanism"""
        # Get current state
        objective_state = self.serialize_objective_state()
        # Get recent validations (if any)
        recent_validations = self._get_recent_validations(limit=5)
        # Get active conflicts
        active_conflicts = self._get_active_conflicts()

        consensus_export = {
            "schema_version": self.schema_version,
            "type": "consensus_context",
            "metadata": self._create_metadata("consensus_export"),
            "system_state": {
                "objectives": objective_state.get("objectives", {}),
                "constraints": objective_state.get("constraints", {}),
                "timestamp": time.time(),
            },
            "recent_activity": {
                "validations": recent_validations,
                "conflicts": active_conflicts,
                "statistics": self._get_consensus_statistics(),
            },
            "voting_context": {  # Example context
                "quorum_required": True,
                "voting_mechanism": "weighted_majority",
                "confidence_threshold": 0.7,
                "timeout_seconds": 300,
            },
        }
        # Signature, internal audit, guard, logging... (same pattern as others)
        try:
            signature = self._generate_signature(
                self._make_serializable(consensus_export, seen=None)
            )
        except Exception:
            signature = "signature_error"
        consensus_export["signature"] = signature
        internal_audit = self._generate_internal_audit()
        audit_entry_data = consensus_export.copy()
        if internal_audit:
            audit_entry_data["internal_audit"] = internal_audit
        if "csiu" in consensus_export:
            consensus_export.pop("csiu", None)
        if "internal_audit" in consensus_export:
            consensus_export.pop("internal_audit", None)
        self._audit("consensus_export", audit_entry_data)
        self.stats["consensus_exports"] += 1
        return consensus_export

    def get_csiu_longitudinal(self) -> Dict[str, Any]:
        """Maintainer-only accessor for CSIU longitudinal trends."""
        # Check if introspection engine and tracker exist and are not mocks
        tracker = None
        if hasattr(self.introspection_engine, "validation_tracker") and not isinstance(
            self.introspection_engine.validation_tracker, MagicMock
        ):
            tracker = self.introspection_engine.validation_tracker

        if tracker is None:
            logger.debug("Validation tracker unavailable for CSIU longitudinal data.")
            return {}

        # Check if CSIU is enabled via SID reference (more robust check)
        if isinstance(self.self_improvement_drive, MagicMock) or not getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            logger.debug("CSIU disabled or self_improvement_drive unavailable.")
            return {}

        try:
            # Call the tracker's summarize method if it exists
            if hasattr(tracker, "summarize_longitudinal") and callable(
                tracker.summarize_longitudinal
            ):
                trends = tracker.summarize_longitudinal()
                return self._make_serializable(
                    trends, seen=None
                )  # Ensure result is serializable
            else:
                logger.warning(
                    "ValidationTracker missing 'summarize_longitudinal' method."
                )
                return {}
        except Exception as e:
            logger.error(f"Error getting CSIU longitudinal data: {e}", exc_info=True)
            return {}

    def get_audit_log(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries, applying filters."""
        # Convert deque to list for safe filtering
        try:
            # Ensure thread safety if log could be modified during iteration
            # Although deque appends are thread-safe, filtering might race?
            # Creating a list snapshot is safer.
            log_list = list(self.audit_log)
        except Exception as e:
            logger.error(f"Error converting audit log deque to list: {e}")
            return []

        filtered = log_list
        try:
            if start_time is not None:
                filtered = [
                    e
                    for e in filtered
                    if isinstance(e, dict) and e.get("timestamp", 0) >= start_time
                ]
            if end_time is not None:
                filtered = [
                    e
                    for e in filtered
                    if isinstance(e, dict)
                    and e.get("timestamp", float("inf")) <= end_time
                ]
            if event_type is not None:
                filtered = [
                    e
                    for e in filtered
                    if isinstance(e, dict) and e.get("event_type") == event_type
                ]
        except Exception as e:
            logger.error(f"Error filtering audit log: {e}")
            return []  # Return empty list on error

        return filtered

    def verify_signature(self, serialized_data: Dict[str, Any]) -> bool:
        """Verify signature of serialized data"""
        if not isinstance(serialized_data, dict):
            return False  # Must be dict

        stored_signature = serialized_data.get("signature")
        if not stored_signature or not isinstance(stored_signature, str):
            logger.debug("Missing or invalid signature in data for verification.")
            return False

        # Create a copy excluding signature and internal audit for recalculation
        data_to_verify = {}
        for k, v in serialized_data.items():
            if k not in ("signature", "internal_audit"):
                data_to_verify[k] = v

        try:
            # Recalculate signature on the cleaned data
            recalculated_signature = self._generate_signature(
                self._make_serializable(data_to_verify, seen=None)
            )
            is_valid = stored_signature == recalculated_signature
            if not is_valid:
                logger.warning(
                    f"Signature mismatch! Stored: {stored_signature}, Calculated: {recalculated_signature}"
                )
            return is_valid
        except Exception as e:
            logger.error(f"Error during signature verification: {e}")
            return False

    def _safe_float(
        self, value: Any, default: Optional[float] = 0.0
    ) -> Optional[float]:
        """Safely convert value to float, handling None, numpy types, and invalid values."""
        if value is None:
            return default
        # Handle numpy types if numpy is available
        if NUMPY_AVAILABLE and isinstance(value, np.generic):
            try:
                return float(value.item())
            except (AttributeError, ValueError, TypeError):
                return default
        # Handle standard types
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _generate_internal_audit(self) -> Optional[Dict[str, Any]]:
        """Generate internal audit data including CSIU metrics."""
        # Use mock-safe access for SID attributes
        if isinstance(self.self_improvement_drive, MagicMock) or not getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            return None  # CSIU disabled or SID not available

        try:
            cur_telemetry = {}
            if hasattr(
                self.self_improvement_drive, "_collect_telemetry_snapshot"
            ) and callable(self.self_improvement_drive._collect_telemetry_snapshot):
                telemetry_result = (
                    self.self_improvement_drive._collect_telemetry_snapshot()
                )
                if isinstance(telemetry_result, dict):
                    cur_telemetry = telemetry_result

            csiu_weights = getattr(self.self_improvement_drive, "_csiu_w", {})
            if not isinstance(csiu_weights, dict):
                csiu_weights = {}  # Ensure dict

            pressure = 0.0
            if hasattr(self.self_improvement_drive, "_csiu_u_ewma"):
                U_ewma = getattr(self.self_improvement_drive, "_csiu_u_ewma", 0.0)
                if hasattr(self.self_improvement_drive, "_csiu_pressure") and callable(
                    self.self_improvement_drive._csiu_pressure
                ):
                    try:
                        pressure = self.self_improvement_drive._csiu_pressure(U_ewma)
                    except Exception as e:
                        logger.error(
                            f"Error calculating CSIU pressure: {e}", exc_info=True
                        )

            internal_audit = {
                "csiu": {
                    "pressure": round(self._safe_float(pressure, 0.0), 3),
                    "weights": {
                        k: round(self._safe_float(v, 0.0), 3)
                        for k, v in csiu_weights.items()
                    },
                    "telemetry": {
                        k: (
                            round(self._safe_float(v, 0.0), 3)
                            if isinstance(v, (int, float))
                            else str(v)
                        )
                        for k, v in cur_telemetry.items()
                    },
                    "timestamp": time.time(),
                }
            }

            # Add longitudinal trends if available
            longitudinal_trends = self.get_csiu_longitudinal()  # Already serializes
            if longitudinal_trends:
                internal_audit["longitudinal_trends"] = longitudinal_trends

            return internal_audit

        except Exception as e:
            logger.error(f"Failed to generate internal audit data: {e}", exc_info=True)
            return None

    def _create_metadata(self, context: str) -> Dict[str, Any]:
        """Create metadata for serialized output"""
        return {
            "version": self.schema_version,
            "timestamp": time.time(),
            "source": "vulcan_ami_meta_reasoning",
            "context": context,
            # Signature added later after full serialization
        }

    def _serialize_conflicts_compact(
        self, conflicts: List[Any]
    ) -> List[Dict[str, Any]]:
        """Serialize conflicts in compact format"""
        serialized = []
        if not isinstance(conflicts, list):
            return []  # Ensure list input

        for c in conflicts:
            try:
                c_dict = self._make_serializable(c, seen=None)
                if isinstance(c_dict, dict):
                    objectives = c_dict.get("objectives", [])
                    if not isinstance(objectives, list):
                        objectives = []

                    serialized.append(
                        {
                            "objectives": sorted(
                                [str(o) for o in objectives]
                            ),  # Sort for consistency
                            "type": str(
                                c_dict.get(
                                    "conflict_type", c_dict.get("type", "unknown")
                                )
                            ),
                            "severity": str(c_dict.get("severity", "unknown")),
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Skipping conflict during compact serialization due to error: {e}"
                )
        return serialized

    def _serialize_alternatives_compact(
        self, alternatives: List[Any]
    ) -> List[Dict[str, Any]]:
        """Serialize alternatives in compact format"""
        serialized = []
        if not isinstance(alternatives, list):
            return []

        for alt in alternatives:
            try:
                alt_dict = self._make_serializable(alt, seen=None)
                if isinstance(alt_dict, dict):
                    serialized.append(
                        {
                            "objective": str(alt_dict.get("objective", "unknown")),
                            "confidence": self._safe_float(
                                alt_dict.get("confidence"), 0.0
                            ),  # Ensure float
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Skipping alternative during compact serialization due to error: {e}"
                )
        return serialized

    def _extract_structured_reasoning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured reasoning components from various serialized data"""
        # **************************************************************************
        # FIX 3: Add guard clause for non-dict inputs
        if not isinstance(data, dict):
            return {"decision": None, "confidence": 0.0, "key_factors": []}
        # **************************************************************************

        # Look for outcome/valid status first
        outcome_data = data.get("validation", {}).get(
            "outcome", data.get("outcome", {})
        )
        if not isinstance(outcome_data, dict):
            outcome_data = {}

        decision = outcome_data.get("valid")  # Check validation outcome specifically
        if decision is None:  # Fallback for negotiation outcome status
            decision_str = outcome_data.get("status")
            if decision_str:
                decision = decision_str not in [
                    "deadlock",
                    "dominated",
                    "unknown",
                    "serialization_error",
                ]

        confidence = self._safe_float(
            outcome_data.get("confidence", data.get("confidence")), 0.0
        )  # Check multiple places

        key_factors = set()
        reasoning_text = str(
            data.get("reasoning", data.get("rationale", {}).get("reasoning", ""))
        )  # Check multiple places

        # Simple keyword extraction from reasoning text
        if reasoning_text:
            text_lower = reasoning_text.lower()
            if "conflict" in text_lower:
                key_factors.add("conflict_detected")
            if "violation" in text_lower:
                key_factors.add("constraint_violation")
            if "align" in text_lower:
                key_factors.add("objective_aligned")
            if "historical" in text_lower or "pattern" in text_lower:
                key_factors.add("historical_pattern_match")
            if "tradeoff" in text_lower:
                key_factors.add("tradeoff_considered")
            if "pareto" in text_lower:
                key_factors.add("pareto_optimality")
            if "compromise" in text_lower:
                key_factors.add("compromise_reached")
            if "deadlock" in text_lower:
                key_factors.add("negotiation_deadlock")

        # Add factors from specific structured fields
        if data.get("conflicts_detected") or data.get("conflicts"):
            key_factors.add("explicit_conflict")
        if any(
            oa.get("status") == "violation"
            for oa in data.get("validation", {}).get("objectives", [])
        ):
            key_factors.add("explicit_violation")

        # FIX: Handle outcome being a string instead of a dict
        outcome_val = data.get("outcome")
        if isinstance(outcome_val, dict) and outcome_val.get("pareto_optimal"):
            key_factors.add("pareto_optimal_solution")

        return {
            "decision": decision,  # Changed from 'decision_outcome' to 'decision'
            "confidence": confidence,
            "key_factors": sorted(list(key_factors)),  # Return sorted list
        }

    def _extract_actionable_items(
        self, validation_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract actionable items if validation failed"""
        actionable = []

        # Check validation outcome safely - handle both nested and flat structures
        # Try nested structure first (full serialized format)
        validation_outcome = validation_dict.get("validation", {}).get("outcome", {})
        if not isinstance(validation_outcome, dict):
            validation_outcome = {}

        # Try to get 'valid' from nested structure, then fall back to flat structure
        valid = validation_outcome.get("valid")
        if valid is None:
            # Fall back to flat structure
            valid = validation_dict.get("valid", True)

        # Default to True if still None/missing
        if valid is None:
            valid = True

        if not valid:
            logger.debug("Extracting actionable items for rejected proposal.")
            # Proposal was rejected - provide actionable suggestions

            # Use compact conflict serialization - try both nested and flat
            conflicts = validation_dict.get("validation", {}).get("conflicts", [])
            if not conflicts:
                conflicts = validation_dict.get("conflicts", [])

            if conflicts:
                actionable.append(
                    {
                        "action": "resolve_conflicts",
                        "target": "proposal_objectives",  # More specific target
                        "details": {
                            "summary": f"{len(conflicts)} conflict(s) detected",
                            "conflicts": conflicts,
                        },  # Use pre-serialized conflicts
                    }
                )

            # Use compact alternative serialization - try both nested and flat
            alternatives = validation_dict.get("validation", {}).get("alternatives", [])
            if not alternatives:
                alternatives = validation_dict.get("alternatives", [])

            if alternatives:
                actionable.append(
                    {
                        "action": "consider_alternatives",
                        "target": "proposal_approach",
                        "details": {
                            "summary": f"{len(alternatives)} alternative(s) suggested",
                            "alternatives": alternatives,
                        },  # Use pre-serialized alts
                    }
                )

            # Suggest reviewing reasoning, provide structured factors
            structured_reasoning = (
                validation_dict.get("validation", {})
                .get("reasoning", {})
                .get("structured", {})
            )
            if not actionable:  # Only if no specific actions above
                actionable.append(
                    {
                        "action": "review_reasoning",
                        "target": "proposal_logic",
                        "details": {
                            "primary_reason": validation_dict.get("validation", {})
                            .get("reasoning", {})
                            .get("primary", "No specific reason provided."),
                            "key_factors": structured_reasoning.get("key_factors", []),
                        },
                    }
                )

        return actionable

    def _serialize_active_objectives(
        self, objectives_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Serialize active objectives list with weights/constraints"""
        if not isinstance(objectives_data, dict):
            return []

        active = objectives_data.get("active", [])
        if not isinstance(active, list):
            active = []

        weights = objectives_data.get("weights", {})
        if not isinstance(weights, dict):
            weights = {}

        constraints = objectives_data.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        serialized_list = []
        for obj_name in active:
            if isinstance(obj_name, str):  # Ensure obj is a string name
                serialized_list.append(
                    {
                        "name": obj_name,
                        "weight": self._safe_float(
                            weights.get(obj_name), 1.0
                        ),  # Default 1.0
                        "constraints": self._make_serializable(
                            constraints.get(obj_name, {}), seen=None
                        ),  # Ensure constraints are serializable
                    }
                )
        return serialized_list

    def _serialize_hierarchy(self, hierarchy_data: Any) -> Any:
        """Serialize objective hierarchy structure"""
        # Assume hierarchy_data is already somewhat structured (e.g., dict from explain_motivation_structure)
        # Use robust serializer to handle potential objects or non-JSON types within
        return self._make_serializable(hierarchy_data, seen=None)

    def _serialize_constraints(self) -> Dict[str, Any]:
        """Serialize all known constraints"""
        constraints = {}
        # Access constraints safely via introspection_engine attribute
        if hasattr(self.introspection_engine, "objective_constraints") and isinstance(
            self.introspection_engine.objective_constraints, dict
        ):
            constraints = self.introspection_engine.objective_constraints

        # Ensure the result is serializable
        return self._make_serializable(constraints, seen=None)

    def _count_by_severity(
        self, items: List[Dict[str, Any]], key: str = "severity"
    ) -> Dict[str, int]:
        """Count items (like conflicts or violations) by severity key"""
        counts = defaultdict(int)
        for item in items:
            if isinstance(item, dict):
                severity_val = item.get(key, "unknown")
                counts[str(severity_val)] += 1  # Ensure string key
        return dict(counts)

    def _count_by_type(
        self, items: List[Dict[str, Any]], key: str = "type"
    ) -> Dict[str, int]:
        """Count items (like conflicts) by type key"""
        counts = defaultdict(int)
        for item in items:
            if isinstance(item, dict):
                type_val = item.get(key, "unknown")
                counts[str(type_val)] += 1  # Ensure string key
        return dict(counts)

    def _recommend_resolution_strategy(self, conflicts: List[Dict[str, Any]]) -> str:
        """Recommend resolution strategy based on conflict severity"""
        if not conflicts:
            return "none_needed"
        # Check severities safely
        has_critical = any(
            c.get("severity") in ["critical", ViolationSeverity.CRITICAL.value]
            for c in conflicts
            if isinstance(c, dict)
        )
        if has_critical:
            return "immediate_resolution_required"
        has_high = any(
            c.get("severity") in ["high", ViolationSeverity.HIGH.value]
            for c in conflicts
            if isinstance(c, dict)
        )
        if has_high:
            return "priority_resolution_recommended"
        return "standard_resolution"

    def _get_recent_validations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent validation results safely"""
        # Access tracker safely
        tracker = None
        if hasattr(self.introspection_engine, "validation_tracker"):
            tracker = self.introspection_engine.validation_tracker

        if (
            tracker is None
            or isinstance(tracker, MagicMock)
            or not hasattr(tracker, "validation_records")
        ):
            logger.debug(
                "Validation tracker unavailable or invalid for recent validations."
            )
            return []

        # Access deque safely and convert to list snapshot
        try:
            history_deque = tracker.validation_records
            if not isinstance(history_deque, deque):
                return []  # Ensure it's a deque
            history_list = list(history_deque)
            recent = history_list[-limit:]  # Get last 'limit' items

            # Serialize safely
            serialized_recent = []
            for v in recent:
                # Attempt to serialize using to_dict if available, else basic attrs
                v_dict = {}
                if hasattr(v, "to_dict") and callable(v.to_dict):
                    try:
                        v_dict = self._make_serializable(v.to_dict(), seen=None)
                    except Exception as e:
                        logger.debug(f"Operation failed: {e}")
                if not v_dict:  # Fallback if to_dict fails or not present
                    v_dict = {
                        "proposal_id": getattr(v, "proposal_id", "unknown"),
                        "valid": getattr(v, "valid", False),  # Use getattr for safety
                        "outcome": str(getattr(v, "outcome", "unknown")),
                        "timestamp": self._safe_float(getattr(v, "timestamp", 0.0)),
                    }
                # Ensure basic fields are present after serialization attempt
                serialized_recent.append(
                    {
                        "id": v_dict.get("proposal_id", v_dict.get("id", "unknown")),
                        "valid": v_dict.get("valid", False),
                        "timestamp": self._safe_float(v_dict.get("timestamp"), 0.0),
                    }
                )

            return serialized_recent
        except Exception as e:
            logger.error(
                f"Error accessing or serializing validation history: {e}", exc_info=True
            )
            return []

    def _get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Get currently active conflicts (based on hierarchy and recent history)"""
        active_conflicts = []
        _np = self._np  # Use internal alias

        # 1. Conflicts derived from current objective hierarchy state (mock-safe)
        hierarchy = None
        if hasattr(self.introspection_engine, "objective_hierarchy"):
            hierarchy = self.introspection_engine.objective_hierarchy

        objectives_dict = {}
        if (
            hierarchy
            and not isinstance(hierarchy, MagicMock)
            and hasattr(hierarchy, "objectives")
            and isinstance(hierarchy.objectives, dict)
        ):
            objectives_dict = hierarchy.objectives

        if objectives_dict and hasattr(hierarchy, "find_conflicts"):
            objective_names = list(objectives_dict.keys())
            for i, obj_a in enumerate(objective_names):
                for obj_b in objective_names[i + 1 :]:
                    try:
                        conflict = hierarchy.find_conflicts(obj_a, obj_b)
                        if conflict and isinstance(conflict, dict):
                            # Serialize and add
                            active_conflicts.append(
                                self._make_serializable(conflict, seen=None)
                            )
                    except Exception as e:
                        logger.debug(
                            f"Error checking hierarchy conflict between {obj_a} and {obj_b}: {e}"
                        )

        # 2. Conflicts from recent validation history (from introspection engine's history)
        conflict_history_deque = None
        if hasattr(self.introspection_engine, "conflict_history"):
            conflict_history_deque = self.introspection_engine.conflict_history

        if isinstance(conflict_history_deque, deque):
            try:
                # Get last N conflicts from history deque
                recent_conflicts = list(conflict_history_deque)[
                    -10:
                ]  # Snapshot last 10
                for conflict in recent_conflicts:
                    # Serialize and add
                    active_conflicts.append(
                        self._make_serializable(conflict, seen=None)
                    )
            except Exception as e:
                logger.error(f"Error accessing conflict history: {e}", exc_info=True)

        # Deduplicate based on objectives and type
        unique_conflicts = []
        seen = set()
        for c in active_conflicts:
            if isinstance(c, dict):
                # Create a stable key: sorted tuple of objectives + type string
                obj_tuple = tuple(sorted(str(o) for o in c.get("objectives", [])))
                type_str = str(c.get("type", c.get("conflict_type", "unknown")))
                key = (obj_tuple, type_str)

                if key not in seen:
                    unique_conflicts.append(c)
                    seen.add(key)

        logger.debug(f"Found {len(unique_conflicts)} unique active/recent conflicts.")
        return unique_conflicts

    def _get_consensus_statistics(self) -> Dict[str, Any]:
        """Get statistics relevant for consensus decisions"""
        _np = self._np  # Use internal alias
        stats = {"total_validations": 0, "approval_rate": 0.5, "avg_confidence": 0.5}

        # Access tracker safely
        tracker = None
        if hasattr(self.introspection_engine, "validation_tracker"):
            tracker = self.introspection_engine.validation_tracker

        if (
            tracker
            and not isinstance(tracker, MagicMock)
            and hasattr(tracker, "validation_records")
        ):
            try:
                history_deque = tracker.validation_records
                if not isinstance(history_deque, deque):
                    raise TypeError("validation_records is not a deque")

                history_list = list(history_deque)  # Snapshot
                num_records = len(history_list)
                stats["total_validations"] = num_records

                if num_records > 0:
                    # Calculate approval rate safely using getattr
                    approved_count = sum(
                        1 for v in history_list if getattr(v, "valid", False)
                    )
                    stats["approval_rate"] = approved_count / num_records

                    # Calculate average confidence safely using getattr and safe_float
                    confidences = [
                        self._safe_float(getattr(v, "confidence", 0.5), 0.5)
                        for v in history_list
                    ]
                    stats["avg_confidence"] = (
                        float(_np.mean(confidences)) if confidences else 0.5
                    )

            except Exception as e:
                logger.error(
                    f"Error calculating consensus statistics from validation history: {e}",
                    exc_info=True,
                )
                # Keep default stats on error

        return stats

    def _generate_signature(self, data: Any) -> str:
        """Generate cryptographic signature (hash) for data"""
        try:
            # Convert data to a stable JSON string representation
            # Use default=str for basic fallback, sort keys for consistency
            data_str = json.dumps(
                data, sort_keys=True, default=str, separators=(",", ":")
            )
            # Encode to bytes and hash
            signature = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
            return signature
        except TypeError as e:
            logger.error(
                f"Failed to serialize data for signature generation: {e}. Data type: {type(data)}"
            )
            # Fallback if JSON fails (less stable)
            try:
                fallback_str = repr(data)  # Use repr as a fallback
                signature = hashlib.sha256(fallback_str.encode("utf-8")).hexdigest()
                logger.warning("Used repr fallback for signature generation.")
                return signature
            except Exception as final_e:
                logger.critical(
                    f"CRITICAL: Could not generate signature even with fallback: {final_e}"
                )
                return "signature_generation_failed"

    def _audit(self, event_type: str, data: Dict[str, Any]):
        """Add entry to audit log, ensuring data signature is present"""
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            # Ensure data signature exists, recalculate if missing (shouldn't happen)
            "data_signature": data.get("signature")
            or self._generate_signature(self._make_serializable(data, seen=None)),
            # Optionally include a snapshot of key data points for quick audit viewing
            "summary": {
                "type": data.get("type"),
                "id": data.get("validation", {}).get("id")
                or data.get("proposal_id")
                or data.get("id", "N/A"),
            },
        }

        # Include CSIU summary if available AND enabled
        if not isinstance(self.self_improvement_drive, MagicMock) and getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            csiu_summary = data.get("internal_audit", {}).get("csiu")
            if csiu_summary and isinstance(csiu_summary, dict):
                entry["csiu_summary"] = {  # Select key fields
                    "pressure": csiu_summary.get("pressure"),
                    "top_weights": sorted(
                        csiu_summary.get("weights", {}).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[
                        :3
                    ],  # Top 3 weights
                }

        try:
            # Append to audit log
            self.audit_log.append(entry)

            # Trim audit log if it exceeds max_audit_entries
            if len(self.audit_log) > self.max_audit_entries:
                # Keep only the most recent entries
                self.audit_log = self.audit_log[-self.max_audit_entries :]
        except Exception as e:
            logger.error(f"Failed to append to audit log: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get transparency interface statistics"""
        # Use mock-safe access for SID attributes
        csiu_enabled = False
        csiu_calc_enabled = False
        csiu_regs_enabled = False
        csiu_hist_enabled = False
        if not isinstance(self.self_improvement_drive, MagicMock):
            csiu_enabled = getattr(self.self_improvement_drive, "_csiu_enabled", False)
            if csiu_enabled:
                csiu_calc_enabled = getattr(
                    self.self_improvement_drive, "_csiu_calc_enabled", False
                )
                csiu_regs_enabled = getattr(
                    self.self_improvement_drive, "_csiu_regs_enabled", False
                )
                csiu_hist_enabled = getattr(
                    self.self_improvement_drive, "_csiu_hist_enabled", False
                )

        # Create stats dict using .get() for safety
        stats_snapshot = dict(self.stats)  # Snapshot defaultdict

        stats_output = {
            "statistics": {  # Wrap counters in 'statistics' key
                "validations_serialized": stats_snapshot.get(
                    "validations_serialized", 0
                ),
                "objective_states_serialized": stats_snapshot.get(
                    "objective_states_serialized", 0
                ),
                "conflicts_serialized": stats_snapshot.get("conflicts_serialized", 0),
                "negotiations_serialized": stats_snapshot.get(
                    "negotiations_serialized", 0
                ),
                "consensus_exports": stats_snapshot.get("consensus_exports", 0),
            },
            "audit_log_size": len(self.audit_log),  # Use current len
            "schema_version": self.schema_version,
            "csiu_status": {  # Internal status check
                "enabled": csiu_enabled,
                "calc_enabled": csiu_calc_enabled,
                "regs_enabled": csiu_regs_enabled,
                "hist_enabled": csiu_hist_enabled,
            },
        }

        return stats_output

    def explain_decision(
        self,
        decision: Any,
        factors: Optional[Dict[str, Any]] = None,
        reasoning_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a transparent explanation for a decision.
        
        This method provides machine-readable transparency for decisions made by
        the reasoning system. It's used by the orchestrator for self-referential
        meta-reasoning queries.
        
        Thread-safe: Uses lock for cache operations.
        
        Args:
            decision: The decision to explain (string, dict, or other)
            factors: Contributing factors to the decision. Keys are factor names,
                    values can be strings, dicts with 'reason'/'description', or primitives.
            reasoning_steps: List of reasoning steps taken, in execution order.
            
        Returns:
            Dict containing structured explanation with:
                - decision_summary (str): Brief summary of the decision
                - contributing_factors (List[str]): Key factors that influenced the decision
                - reasoning_trace (List[str]): Steps taken during reasoning
                - confidence (float): Overall confidence in the explanation (0.0-1.0)
                - factors_detail (Dict): Serialized detailed factors
                - timestamp (float): Unix timestamp when explanation was generated
                
        Raises:
            Never raises - returns minimal explanation on error with 'error' field set.
            
        Example:
            >>> interface = TransparencyInterface()
            >>> explanation = interface.explain_decision(
            ...     decision="Route to mathematical engine",
            ...     factors={'query_type': 'bayesian', 'confidence': 0.85},
            ...     reasoning_steps=['Parse query', 'Detect probability notation', 'Select engine']
            ... )
            >>> explanation['decision_summary']
            'Route to mathematical engine'
            >>> explanation['confidence']
            0.75
        """
        with self.lock:
            try:
                # Validate and normalize inputs
                if factors is None:
                    factors = {}
                if reasoning_steps is None:
                    reasoning_steps = []
                
                # Validate types
                if not isinstance(factors, dict):
                    logger.warning(f"[TransparencyInterface] factors should be dict, got {type(factors)}")
                    factors = {'raw_factors': str(factors)}
                    
                if not isinstance(reasoning_steps, list):
                    logger.warning(f"[TransparencyInterface] reasoning_steps should be list, got {type(reasoning_steps)}")
                    reasoning_steps = [str(reasoning_steps)]
                
                # Extract decision summary with sanitization
                if isinstance(decision, str):
                    decision_summary = decision[:MAX_DECISION_SUMMARY_LENGTH]
                elif isinstance(decision, dict):
                    decision_summary = decision.get('query', decision.get('decision', str(decision)))[:MAX_DECISION_SUMMARY_LENGTH]
                else:
                    decision_summary = str(decision)[:MAX_DECISION_SUMMARY_LENGTH]
                
                # Extract contributing factors with error handling
                contributing_factors = []
                for key, value in factors.items():
                    try:
                        if isinstance(value, dict):
                            # Extract nested info safely
                            factor_desc = f"{key}: {value.get('reason', value.get('description', str(value)[:MAX_FACTOR_VALUE_LENGTH]))}"
                        else:
                            factor_desc = f"{key}: {str(value)[:MAX_FACTOR_VALUE_LENGTH]}"
                        contributing_factors.append(factor_desc)
                    except Exception as e:
                        logger.debug(f"[TransparencyInterface] Error processing factor {key}: {e}")
                        contributing_factors.append(f"{key}: <unparseable>")
                
                # Calculate confidence based on data availability and quality
                confidence = self.default_explanation_confidence
                if not factors:
                    confidence *= 0.8  # Reduce confidence if no factors
                if not reasoning_steps:
                    confidence *= 0.9  # Reduce confidence if no trace
                confidence = max(0.1, min(1.0, confidence))  # Clamp to valid range
                
                # Build explanation with defensive serialization
                explanation = {
                    'decision_summary': decision_summary,
                    'contributing_factors': contributing_factors,
                    'reasoning_trace': [str(step)[:MAX_REASONING_STEP_LENGTH] for step in reasoning_steps],
                    'confidence': confidence,
                    'factors_detail': self._make_serializable(factors),
                    'timestamp': time.time(),
                }
                
                # Update stats atomically
                self.stats['explanations_generated'] += 1
                
                return explanation
                
            except Exception as e:
                logger.error(f"[TransparencyInterface] Failed to explain decision: {e}", exc_info=True)
                # Return minimal explanation on error - never raise
                return {
                    'decision_summary': str(decision)[:MAX_DECISION_SUMMARY_LENGTH] if decision else 'Unknown decision',
                    'contributing_factors': [],
                    'reasoning_trace': reasoning_steps or [],
                    'confidence': 0.3,  # Low confidence for error case
                    'error': str(e),
                    'timestamp': time.time(),
                }


# Need math for float checks
