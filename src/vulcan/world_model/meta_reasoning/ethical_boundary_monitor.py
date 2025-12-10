# src/vulcan/world_model/meta_reasoning/ethical_boundary_monitor.py
"""
ethical_boundary_monitor.py - Multi-layered ethical boundary monitoring and enforcement
Part of the meta_reasoning subsystem for VULCAN-AMI

FULL PRODUCTION IMPLEMENTATION

Comprehensive ethical safety system with:
- Multi-layered boundary definitions (hard constraints, soft guidelines)
- Real-time violation detection with graduated responses
- Rule-based and learned boundary checking
- Contextual ethical reasoning
- Transparency and audit trails
- Emergency shutdown capabilities
- Integration with safety systems

Boundary Categories:
- HARM_PREVENTION: Prevent physical, psychological, or societal harm
- PRIVACY: Protect user privacy and data confidentiality
- FAIRNESS: Ensure fair treatment across demographics
- TRANSPARENCY: Maintain explainability and accountability
- AUTONOMY: Respect user agency and informed consent
- TRUTHFULNESS: Prevent deception and misinformation
- RESOURCE_LIMITS: Prevent resource abuse

Enforcement Levels:
- MONITOR: Log for review (no action)
- WARN: Alert but allow action
- MODIFY: Automatically modify action to comply
- BLOCK: Prevent action entirely
- SHUTDOWN: Emergency shutdown if critical violation

Integration:
- Records to ValidationTracker for pattern learning
- Alerts to SelfImprovementDrive for adaptation
- Feeds TransparencyInterface for audit trails
- Integrates with world model for contextual reasoning
"""

import json
import logging
import re
import threading
import time  # Moved import here
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
# import time # Original import
# import numpy as np # Original import
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
        # Define necessary numpy functions used in this file (if any)
        # Based on the code, numpy seems only used for potential future extensions or maybe type hints.
        # Currently, no direct np calls are made that need replacement.
        # Add common fallbacks just in case.
        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0

        @staticmethod
        def array(lst):
            return list(lst)

        # Add other potential fallbacks if needed later

    np = FakeNumpy()
# --- END FIX ---


class BoundaryCategory(Enum):
    """Category of ethical boundary"""

    HARM_PREVENTION = "harm_prevention"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    AUTONOMY = "autonomy"
    TRUTHFULNESS = "truthfulness"
    RESOURCE_LIMITS = "resource_limits"
    GENERAL = "general"


class ViolationSeverity(Enum):
    """Severity of ethical violations"""

    CRITICAL = "critical"  # Immediate danger, requires shutdown
    HIGH = "high"  # Serious violation, block action
    MEDIUM = "medium"  # Moderate violation, modify or warn
    LOW = "low"  # Minor violation, warn only
    NONE = "none"  # No violation


class EnforcementLevel(Enum):
    """Level of enforcement for boundary violations"""

    MONITOR = "monitor"  # Log only, no action
    WARN = "warn"  # Alert but allow
    MODIFY = "modify"  # Automatically modify to comply
    BLOCK = "block"  # Prevent action
    SHUTDOWN = "shutdown"  # Emergency shutdown


class BoundaryType(Enum):
    """Type of boundary constraint"""

    HARD_CONSTRAINT = "hard_constraint"  # Must never be violated
    SOFT_GUIDELINE = "soft_guideline"  # Should be followed but can be overridden
    LEARNED_BOUNDARY = "learned_boundary"  # Learned from experience
    CONTEXTUAL = "contextual"  # Context-dependent


@dataclass
class EthicalBoundary:
    """Definition of an ethical boundary"""

    name: str
    category: BoundaryCategory
    boundary_type: BoundaryType
    description: str

    # Enforcement
    enforcement_level: EnforcementLevel
    severity_if_violated: ViolationSeverity

    # Constraint definition
    constraint_function: Optional[Callable[[Dict[str, Any]], bool]] = (
        None  # Returns True if OK
    )
    constraint_rules: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Rule-based constraints

    # Modification function (for MODIFY enforcement)
    modification_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    # Context conditions (when this boundary applies)
    applies_when: Optional[Callable[[Dict[str, Any]], bool]] = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_checked: float = 0.0
    check_count: int = 0
    violation_count: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def check(self, action: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Check if action violates this boundary

        Returns:
            True if action is OK (no violation), False if violation
        """
        self.last_checked = time.time()
        self.check_count += 1

        # Check if boundary applies in this context
        if self.applies_when and not self.applies_when(context):
            return True  # Boundary doesn't apply, so no violation

        # Check constraint function if provided
        if self.constraint_function:
            try:
                result = self.constraint_function(action)
                if not result:
                    self.violation_count += 1
                return result
            except Exception as e:
                logger.error(f"Constraint function error for {self.name}: {e}")
                return False  # Fail-safe: treat as violation

        # Check rule-based constraints
        if self.constraint_rules:
            for rule in self.constraint_rules:
                if not self._check_rule(rule, action):
                    self.violation_count += 1
                    return False

        return True

    def _check_rule(self, rule: Dict[str, Any], action: Dict[str, Any]) -> bool:
        """Check a single rule against action"""
        rule_type = rule.get("type", "field_check")

        if rule_type == "field_check":
            # Check if field exists and meets condition
            field_name = rule.get("field")  # Renamed variable to avoid conflict
            condition = rule.get("condition")

            if field_name not in action:
                return rule.get("allow_missing", False)

            value = action[field_name]

            if condition == "exists":
                return True
            elif condition == "not_exists":
                return False
            elif condition == "equals":
                return value == rule.get("value")
            elif condition == "not_equals":
                return value != rule.get("value")
            elif condition == "less_than":
                # Check if value is comparable before comparison
                rule_value = rule.get("value")
                try:
                    return value < rule_value if rule_value is not None else False
                except TypeError:
                    logger.warning(
                        f"TypeError comparing field '{field_name}' ({type(value)}) with value ({type(rule_value)}) for rule {rule}"
                    )
                    return False  # Cannot compare types, assume violation
            elif condition == "greater_than":
                # Check if value is comparable before comparison
                rule_value = rule.get("value")
                try:
                    return value > rule_value if rule_value is not None else False
                except TypeError:
                    logger.warning(
                        f"TypeError comparing field '{field_name}' ({type(value)}) with value ({type(rule_value)}) for rule {rule}"
                    )
                    return False  # Cannot compare types, assume violation
            elif condition == "in_range":
                min_val = rule.get("min", float("-inf"))
                max_val = rule.get("max", float("inf"))
                try:
                    return min_val <= value <= max_val
                except TypeError:
                    logger.warning(
                        f"TypeError comparing field '{field_name}' ({type(value)}) with range ({type(min_val)}, {type(max_val)}) for rule {rule}"
                    )
                    return False
            elif condition == "matches_pattern":
                pattern = rule.get("pattern")
                try:
                    return bool(re.match(pattern, str(value))) if pattern else False
                except re.error as e:
                    logger.error(f"Invalid regex pattern in rule {rule}: {e}")
                    return False  # Invalid pattern, assume violation

        elif rule_type == "combined":
            # Logical combination of sub-rules
            operator = rule.get("operator", "and")
            sub_rules = rule.get("rules", [])

            if operator == "and":
                return all(self._check_rule(r, action) for r in sub_rules)
            elif operator == "or":
                return any(self._check_rule(r, action) for r in sub_rules)
            elif operator == "not":
                return not self._check_rule(sub_rules[0], action) if sub_rules else True

        return True  # Default to allowing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "category": self.category.value,
            "boundary_type": self.boundary_type.value,
            "description": self.description,
            "enforcement_level": self.enforcement_level.value,
            "severity_if_violated": self.severity_if_violated.value,
            "check_count": self.check_count,
            "violation_count": self.violation_count,
            "violation_rate": self.violation_count / max(1, self.check_count),
            "last_checked": self.last_checked,
            "metadata": self.metadata,
        }


@dataclass
class EthicalViolation:
    """Record of an ethical violation"""

    action: Dict[str, Any]
    boundary_violated: str
    category: BoundaryCategory
    severity: ViolationSeverity
    description: str

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Enforcement taken
    enforcement_action: EnforcementLevel = EnforcementLevel.MONITOR
    action_modified: bool = False
    modified_action: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    violation_id: str = field(default_factory=lambda: f"viol_{time.time_ns()}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""

        # Helper to make action serializable (handles non-serializable types)
        def _safe_serialize(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [_safe_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): _safe_serialize(v) for k, v in obj.items()}
            else:
                try:
                    return str(obj)
                except Exception:
                    return f"<unserializable_{type(obj).__name__}>"

        return {
            "violation_id": self.violation_id,
            "boundary_violated": self.boundary_violated,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "enforcement_action": self.enforcement_action.value,
            "action_modified": self.action_modified,
            "modified_action": _safe_serialize(self.modified_action),
            "action": _safe_serialize(self.action),  # Serialize original action too
            "timestamp": self.timestamp,
            "context": _safe_serialize(self.context),
            "metadata": _safe_serialize(self.metadata),
        }


@dataclass
class EnforcementAction:
    """Record of enforcement action taken"""

    violation: EthicalViolation
    action_taken: EnforcementLevel
    success: bool
    details: str
    timestamp: float = field(default_factory=time.time)


class EthicalBoundaryMonitor:
    """
    Multi-layered ethical boundary monitoring and enforcement

    Monitors actions for ethical violations across multiple categories:
    - Harm prevention (physical, psychological, societal)
    - Privacy protection (data, surveillance, confidentiality)
    - Fairness (demographic, outcome, process)
    - Transparency (explainability, accountability)
    - Autonomy (user agency, informed consent)
    - Truthfulness (honesty, accuracy, deception)
    - Resource limits (computational, financial, environmental)

    Enforcement mechanisms:
    - MONITOR: Log violations for review
    - WARN: Alert but allow action
    - MODIFY: Automatically modify action to comply
    - BLOCK: Prevent action entirely
    - SHUTDOWN: Emergency shutdown for critical violations

    Thread-safe with comprehensive audit trails.
    Integrates with VULCAN safety systems.
    """

    def __init__(
        self,
        boundaries: Optional[Dict[str, EthicalBoundary]] = None,
        strict_mode: bool = False,
        alert_callback: Optional[Callable[[EthicalViolation], None]] = None,
        shutdown_callback: Optional[Callable[[EthicalViolation], None]] = None,
        validation_tracker=None,
        self_improvement_drive=None,
        transparency_interface=None,
        load_defaults: bool = False,
    ):
        """
        Initialize ethical boundary monitor

        Args:
            boundaries: Optional pre-defined boundaries
            strict_mode: If True, treat all violations as BLOCK
            alert_callback: Optional callback for violations
            shutdown_callback: Optional callback for critical violations
            validation_tracker: Optional ValidationTracker integration
            self_improvement_drive: Optional SelfImprovementDrive integration
            transparency_interface: Optional TransparencyInterface integration
            load_defaults: If True, load default boundaries (default: False for test isolation)
        """
        self.strict_mode = strict_mode
        self.alert_callback = alert_callback
        self.shutdown_callback = shutdown_callback
        self.validation_tracker = validation_tracker
        self.self_improvement_drive = self_improvement_drive
        self.transparency_interface = transparency_interface

        # Ethical boundaries
        self.boundaries: Dict[str, EthicalBoundary] = boundaries or {}
        # Rebuild index if boundaries were provided
        self.boundary_index: Dict[BoundaryCategory, List[str]] = defaultdict(list)
        if boundaries:
            for name, boundary in boundaries.items():
                self.boundary_index[boundary.category].append(name)

        # Violation history
        self.violations: deque = deque(maxlen=10000)
        self.violations_by_boundary: Dict[str, List[EthicalViolation]] = defaultdict(
            list
        )

        # Enforcement history
        self.enforcement_actions: deque = deque(maxlen=1000)

        # Statistics
        self.stats = defaultdict(int)
        self.stats["initialized_at"] = time.time()

        # Emergency shutdown state
        self.shutdown_triggered: bool = False
        self.shutdown_reason: Optional[str] = None

        # Thread safety
        self.lock = threading.RLock()

        # Initialize default boundaries if requested AND no boundaries provided
        if load_defaults and not boundaries:
            self._initialize_default_boundaries()

        logger.info("EthicalBoundaryMonitor initialized (FULL IMPLEMENTATION)")
        logger.info(f"  Strict mode: {strict_mode}, Boundaries: {len(self.boundaries)}")

    def add_boundary(
        self,
        name: str,
        category: BoundaryCategory,
        boundary_type: BoundaryType,
        description: str,
        enforcement_level: EnforcementLevel,
        severity: ViolationSeverity,
        constraint_function: Optional[Callable[[Dict[str, Any]], bool]] = None,
        constraint_rules: Optional[List[Dict[str, Any]]] = None,
        modification_function: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = None,
        applies_when: Optional[Callable[[Dict[str, Any]], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EthicalBoundary:
        """
        Add a new ethical boundary

        Args:
            name: Unique name for boundary
            category: Boundary category
            boundary_type: Type of boundary
            description: Description of boundary
            enforcement_level: Enforcement level
            severity: Severity if violated
            constraint_function: Function that returns True if OK
            constraint_rules: List of rule dictionaries
            modification_function: Function to modify action
            applies_when: Function to check if boundary applies
            metadata: Additional metadata

        Returns:
            Created EthicalBoundary object
        """
        with self.lock:
            if name in self.boundaries:
                raise ValueError(f"Boundary {name} already exists")

            # Validate constraint function (simple callable check)
            if constraint_function and not callable(constraint_function):
                logger.error(f"Invalid constraint_function for {name}: not callable.")
                raise ValueError("constraint_function must be callable")

            # Validate modification function (simple callable check)
            if modification_function and not callable(modification_function):
                logger.error(f"Invalid modification_function for {name}: not callable.")
                raise ValueError("modification_function must be callable")

            # Validate applies_when function (simple callable check)
            if applies_when and not callable(applies_when):
                logger.error(f"Invalid applies_when for {name}: not callable.")
                raise ValueError("applies_when must be callable")

            boundary = EthicalBoundary(
                name=name,
                category=category,
                boundary_type=boundary_type,
                description=description,
                enforcement_level=enforcement_level,
                severity_if_violated=severity,
                constraint_function=constraint_function,
                constraint_rules=constraint_rules or [],
                modification_function=modification_function,
                applies_when=applies_when,
                metadata=metadata or {},
            )

            self.boundaries[name] = boundary
            self.boundary_index[category].append(name)

            logger.info(f"Added new boundary: {name} ({category.value})")

            return boundary

    def remove_boundary(self, name: str):
        """
        Remove a boundary by name

        Args:
            name: Name of boundary to remove
        """
        with self.lock:
            if name not in self.boundaries:
                raise ValueError(f"Boundary {name} does not exist")

            boundary = self.boundaries.pop(name)
            # Safely remove from index list
            if name in self.boundary_index[boundary.category]:
                self.boundary_index[boundary.category].remove(name)

            logger.info(f"Removed boundary: {name}")

    def clear_boundaries(self):
        """
        Clear all boundaries from the monitor
        """
        with self.lock:
            self.boundaries.clear()
            self.boundary_index.clear()
            logger.info("Cleared all boundaries")

    def get_boundaries(
        self, category: Optional[BoundaryCategory] = None
    ) -> Dict[str, EthicalBoundary]:
        """
        Get all boundaries or filtered by category

        Args:
            category: Optional category filter

        Returns:
            Dictionary of boundaries
        """
        with self.lock:
            if category is None:
                return dict(self.boundaries)

            return {
                name: self.boundaries[name]
                for name in self.boundary_index.get(category, [])
                if name in self.boundaries
            }

    def check_action(
        self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[EthicalViolation]]:
        """
        Check if action violates ethical boundaries

        Args:
            action: Action to check
            context: Optional context for checking

        Returns:
            (is_allowed, violation) tuple
            - is_allowed: True if action is OK, False if should be blocked
            - violation: EthicalViolation object if violation detected, None otherwise
        """
        with self.lock:
            # Check if in shutdown state
            if self.shutdown_triggered:
                return (
                    False,
                    EthicalViolation(
                        action=action,
                        boundary_violated="SYSTEM_SHUTDOWN",
                        category=BoundaryCategory.GENERAL,
                        severity=ViolationSeverity.CRITICAL,
                        description=f"System in shutdown state: {self.shutdown_reason}",
                        context=context or {},
                    ),
                )

            context = context or {}
            self.stats["checks_performed"] += 1

            # Check all applicable boundaries
            violations = []

            # Iterate over a copy of keys in case boundaries are modified during check
            for boundary_name in list(self.boundaries.keys()):
                boundary = self.boundaries.get(boundary_name)
                if not boundary:
                    continue  # Boundary might have been removed concurrently

                # Check if boundary applies
                if boundary.applies_when:
                    try:
                        if not boundary.applies_when(context):
                            continue
                    except Exception as e:
                        logger.error(
                            f"Error checking applies_when for {boundary_name}: {e}"
                        )
                        continue  # Skip boundary if condition check fails

                # Check boundary
                is_ok = boundary.check(action, context)

                if not is_ok:
                    # Create violation record
                    violation = EthicalViolation(
                        action=action,
                        boundary_violated=boundary_name,
                        category=boundary.category,
                        severity=boundary.severity_if_violated,
                        description=f"Action violates boundary: {boundary.description}",
                        context=context,
                        enforcement_action=boundary.enforcement_level,  # Store intended enforcement
                    )

                    violations.append(violation)

            # If no violations, allow action
            if not violations:
                self.stats["actions_allowed"] += 1
                return (True, None)

            # Handle violations (most severe first)
            violations.sort(
                key=lambda v: self._severity_to_numeric(v.severity), reverse=True
            )
            most_severe = violations[0]

            # Record all violations (before enforcement modifies them)
            for violation in violations:
                self._record_violation(violation)

            # Determine if action should be allowed based on MOST SEVERE violation's enforcement
            is_allowed, final_action = self._enforce_violation(most_severe)

            # If modified, update the most_severe violation record for return
            if most_severe.action_modified:
                most_severe.modified_action = (
                    final_action  # Store the final modified action
                )
                # The original action is already stored in the violation record

            if is_allowed:
                self.stats["actions_allowed_with_warnings"] += 1
            else:
                self.stats["actions_blocked"] += 1

            return (is_allowed, most_severe)

    def detect_boundary_violations(
        self, proposal: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[EthicalViolation]:
        """
        Detect all ethical violations in proposal (without enforcement)

        Args:
            proposal: Proposal/action to check
            context: Optional context

        Returns:
            List of detected violations
        """
        with self.lock:
            context = context or {}
            violations = []

            # Use a temporary copy of boundaries to avoid modifying real stats during detection
            temp_boundaries = {
                name: boundary for name, boundary in self.boundaries.items()
            }

            for boundary_name, boundary in temp_boundaries.items():
                # Check if applies
                if boundary.applies_when:
                    try:
                        if not boundary.applies_when(context):
                            continue
                    except Exception as e:
                        logger.debug(
                            f"Error checking applies_when for {boundary_name} during detection: {e}"
                        )
                        continue

                # Check constraint function directly if available
                is_ok = True
                if boundary.constraint_function:
                    try:
                        is_ok = boundary.constraint_function(proposal)
                    except Exception as e:
                        logger.debug(
                            f"Error checking constraint_function for {boundary_name} during detection: {e}"
                        )
                        is_ok = False  # Assume violation on error
                # Check rules otherwise
                elif boundary.constraint_rules:
                    try:
                        is_ok = all(
                            boundary._check_rule(rule, proposal)
                            for rule in boundary.constraint_rules
                        )
                    except Exception as e:
                        logger.debug(
                            f"Error checking rules for {boundary_name} during detection: {e}"
                        )
                        is_ok = False  # Assume violation on error

                if not is_ok:
                    violation = EthicalViolation(
                        action=proposal,
                        boundary_violated=boundary_name,
                        category=boundary.category,
                        severity=boundary.severity_if_violated,
                        description=f"Proposal violates boundary: {boundary.description}",
                        context=context,
                        # Note: Does not set enforcement_action as it's detection only
                    )
                    violations.append(violation)

            return violations

    def get_violations(
        self,
        severity: Optional[ViolationSeverity] = None,
        category: Optional[BoundaryCategory] = None,
        limit: Optional[int] = None,
    ) -> List[EthicalViolation]:
        """
        Get list of violations with optional filters

        Args:
            severity: Optional severity filter
            category: Optional category filter
            limit: Optional limit on results

        Returns:
            List of violations (most recent first)
        """
        with self.lock:
            # Create a snapshot to avoid issues if deque modifies during iteration
            violations_snapshot = list(self.violations)

            # Filter by severity
            if severity:
                violations_snapshot = [
                    v for v in violations_snapshot if v.severity == severity
                ]

            # Filter by category
            if category:
                violations_snapshot = [
                    v for v in violations_snapshot if v.category == category
                ]

            # Sort by timestamp (most recent first)
            violations_snapshot.sort(key=lambda v: v.timestamp, reverse=True)

            # Limit
            if limit:
                violations_snapshot = violations_snapshot[:limit]

            return violations_snapshot

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations"""
        with self.lock:
            summary = {
                "total_violations": len(self.violations),
                "by_severity": defaultdict(int),
                "by_category": defaultdict(int),
                "by_boundary": defaultdict(int),
                "recent_violations": [],
            }

            # Create a snapshot for consistent counting
            violations_snapshot = list(self.violations)

            for violation in violations_snapshot:
                summary["by_severity"][violation.severity.value] += 1
                summary["by_category"][violation.category.value] += 1
                summary["by_boundary"][violation.boundary_violated] += 1

            # Get recent violations (use the already sorted snapshot if available)
            recent = sorted(
                violations_snapshot, key=lambda v: v.timestamp, reverse=True
            )[:10]
            summary["recent_violations"] = [v.to_dict() for v in recent]

            # Convert defaultdicts to regular dicts for output
            summary["by_severity"] = dict(summary["by_severity"])
            summary["by_category"] = dict(summary["by_category"])
            summary["by_boundary"] = dict(summary["by_boundary"])

            return summary

    def trigger_shutdown(self, reason: str):
        """
        Trigger emergency shutdown

        Args:
            reason: Reason for shutdown
        """
        with self.lock:
            # Only trigger if not already shut down
            if not self.shutdown_triggered:
                self.shutdown_triggered = True
                self.shutdown_reason = reason
                self.stats["shutdown_count"] += 1

                logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")

                # Call shutdown callback if provided
                if self.shutdown_callback:
                    try:
                        # Create a representative violation for the callback context
                        violation = EthicalViolation(
                            action={"shutdown": True, "reason": reason},
                            boundary_violated="CRITICAL_SYSTEM_EVENT",  # More generic boundary
                            category=BoundaryCategory.GENERAL,
                            severity=ViolationSeverity.CRITICAL,
                            description=reason,
                            enforcement_action=EnforcementLevel.SHUTDOWN,
                        )
                        self.shutdown_callback(violation)
                    except Exception as e:
                        logger.error(f"Shutdown callback failed: {e}")
            else:
                logger.warning(
                    f"Shutdown already triggered (Reason: {self.shutdown_reason}). Ignoring new trigger: {reason}"
                )

    def reset_shutdown(self):
        """Reset shutdown state (use with extreme caution)"""
        with self.lock:
            if self.shutdown_triggered:
                self.shutdown_triggered = False
                self.shutdown_reason = None
                logger.warning("Shutdown state reset - system reactivated")
            else:
                logger.info("System is not in shutdown state. No reset needed.")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.lock:
            # Create snapshot for consistency
            violations_snapshot = list(self.violations)
            enforcement_snapshot = list(self.enforcement_actions)

            return {
                "checks_performed": self.stats["checks_performed"],
                "actions_allowed": self.stats["actions_allowed"],
                "actions_allowed_with_warnings": self.stats[
                    "actions_allowed_with_warnings"
                ],
                "actions_blocked": self.stats["actions_blocked"],
                "total_violations": len(violations_snapshot),
                "violations_by_severity": self._count_by_severity(
                    violations_snapshot
                ),  # Pass snapshot
                "violations_by_category": self._count_by_category(
                    violations_snapshot
                ),  # Pass snapshot
                "boundaries_defined": len(self.boundaries),
                "boundaries_by_category": {
                    cat.value: len(bounds)
                    for cat, bounds in self.boundary_index.items()
                    if bounds
                },
                "shutdown_triggered": self.shutdown_triggered,
                "shutdown_reason": self.shutdown_reason,
                "enforcement_actions_recorded": len(
                    enforcement_snapshot
                ),  # Use snapshot length
                "initialized_at": self.stats["initialized_at"],
                "uptime_seconds": time.time() - self.stats["initialized_at"],
            }

    def export_state(self) -> Dict[str, Any]:
        """Export monitor state for persistence"""
        with self.lock:
            # Create snapshots
            violations_snapshot = list(self.violations)
            boundaries_snapshot = dict(self.boundaries)

            return {
                # Export boundary metadata only
                "boundaries": {k: v.to_dict() for k, v in boundaries_snapshot.items()},
                "violations": [
                    v.to_dict() for v in violations_snapshot[-1000:]
                ],  # Last 1000
                "stats": dict(self.stats),
                "shutdown_state": {
                    "triggered": self.shutdown_triggered,
                    "reason": self.shutdown_reason,
                },
                "config": {"strict_mode": self.strict_mode},
                "export_time": time.time(),
            }

    def import_state(self, state: Dict[str, Any]):
        """Import monitor state from persistence"""
        with self.lock:
            # Clear existing state before import? Decide based on desired behavior (merge vs replace)
            # self.reset() # Optional: uncomment to replace instead of merge

            # Import boundaries (metadata only, functions cannot be serialized)
            imported_boundaries = state.get("boundaries", {})
            for name, boundary_dict in imported_boundaries.items():
                if (
                    name not in self.boundaries
                ):  # Avoid overwriting existing boundaries with only metadata
                    try:
                        category = BoundaryCategory(boundary_dict["category"])
                        boundary = EthicalBoundary(
                            name=name,
                            category=category,
                            boundary_type=BoundaryType(boundary_dict["boundary_type"]),
                            description=boundary_dict["description"],
                            enforcement_level=EnforcementLevel(
                                boundary_dict["enforcement_level"]
                            ),
                            severity_if_violated=ViolationSeverity(
                                boundary_dict["severity_if_violated"]
                            ),
                            metadata=boundary_dict.get("metadata", {}),
                        )
                        # Restore stats from dict
                        boundary.check_count = boundary_dict.get("check_count", 0)
                        boundary.violation_count = boundary_dict.get(
                            "violation_count", 0
                        )
                        boundary.last_checked = boundary_dict.get("last_checked", 0)

                        self.boundaries[name] = boundary
                        # Ensure category exists in index before appending
                        if category not in self.boundary_index:
                            self.boundary_index[category] = []
                        if name not in self.boundary_index[category]:
                            self.boundary_index[category].append(name)
                        logger.warning(
                            f"Imported boundary {name} metadata only - functions not serializable"
                        )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.error(
                            f"Failed to import boundary '{name}' due to invalid data: {e}. Boundary dict: {boundary_dict}"
                        )
                else:
                    # Optionally update stats for existing boundaries
                    existing_boundary = self.boundaries[name]
                    existing_boundary.check_count = boundary_dict.get(
                        "check_count", existing_boundary.check_count
                    )
                    existing_boundary.violation_count = boundary_dict.get(
                        "violation_count", existing_boundary.violation_count
                    )
                    existing_boundary.last_checked = boundary_dict.get(
                        "last_checked", existing_boundary.last_checked
                    )

            # Import violations (append or replace?) - Current implementation appends to existing deque limit
            imported_violations = state.get("violations", [])
            # Clear existing violations if replacing: self.violations.clear(); self.violations_by_boundary.clear()
            for viol_dict in imported_violations:
                try:
                    category = BoundaryCategory(viol_dict["category"])
                    severity = ViolationSeverity(viol_dict["severity"])
                    enforcement = EnforcementLevel(viol_dict["enforcement_action"])
                    boundary_name = viol_dict["boundary_violated"]

                    viol = EthicalViolation(
                        action=viol_dict.get(
                            "action", {}
                        ),  # Action might be complex/unserializable
                        boundary_violated=boundary_name,
                        category=category,
                        severity=severity,
                        description=viol_dict["description"],
                        context=viol_dict.get("context", {}),
                        enforcement_action=enforcement,
                        action_modified=viol_dict.get("action_modified", False),
                        modified_action=viol_dict.get("modified_action"),
                        timestamp=viol_dict["timestamp"],
                        violation_id=viol_dict["violation_id"],
                        metadata=viol_dict.get("metadata", {}),
                    )
                    self.violations.append(viol)
                    # Ensure boundary name exists before adding to dict
                    if boundary_name not in self.violations_by_boundary:
                        self.violations_by_boundary[boundary_name] = []
                    self.violations_by_boundary[boundary_name].append(viol)
                except (KeyError, ValueError, TypeError) as e:
                    logger.error(
                        f"Failed to import violation {viol_dict.get('violation_id')}: {e}. Violation dict: {viol_dict}"
                    )

            # Import stats (merge or replace?) - Current implementation merges/updates
            imported_stats = state.get("stats", {})
            self.stats.update(imported_stats)  # Simple update, might need smarter merge

            # Import shutdown state
            shutdown_state = state.get("shutdown_state", {})
            self.shutdown_triggered = shutdown_state.get("triggered", False)
            self.shutdown_reason = shutdown_state.get("reason")

            # Import config
            self.strict_mode = state.get("config", {}).get(
                "strict_mode", self.strict_mode
            )  # Use current as default

            logger.info(
                f"Imported state from persistence. Total boundaries: {len(self.boundaries)}, Violations in deque: {len(self.violations)}"
            )

    def reset(self) -> None:
        """Reset all violations and statistics (keeps boundaries)"""
        with self.lock:
            self.violations.clear()
            self.violations_by_boundary.clear()
            self.enforcement_actions.clear()

            # Reset stats, but keep initialization time
            init_time = self.stats.get("initialized_at", time.time())
            self.stats.clear()
            self.stats["initialized_at"] = init_time

            self.shutdown_triggered = False
            self.shutdown_reason = None

            # Reset boundary statistics
            for boundary in self.boundaries.values():
                boundary.check_count = 0
                boundary.violation_count = 0
                # Keep last_checked? Optional. Resetting seems safer.
                boundary.last_checked = 0.0

            logger.info(
                "EthicalBoundaryMonitor reset - violations and stats cleared, boundaries kept"
            )

    # ============================================================
    # Internal Methods
    # ============================================================

    def _initialize_default_boundaries(self):
        """Initialize default ethical boundaries"""

        # HARM PREVENTION: Prevent harmful actions
        try:
            self.add_boundary(
                name="no_physical_harm",
                category=BoundaryCategory.HARM_PREVENTION,
                boundary_type=BoundaryType.HARD_CONSTRAINT,
                description="Prevent actions that could cause physical harm",
                enforcement_level=EnforcementLevel.BLOCK,
                severity=ViolationSeverity.CRITICAL,
                constraint_function=lambda action: not action.get(
                    "causes_physical_harm", False
                ),
            )
        except ValueError:
            pass  # Ignore if already exists

        try:
            self.add_boundary(
                name="no_psychological_harm",
                category=BoundaryCategory.HARM_PREVENTION,
                boundary_type=BoundaryType.HARD_CONSTRAINT,
                description="Prevent actions causing significant psychological harm",
                enforcement_level=EnforcementLevel.BLOCK,
                severity=ViolationSeverity.HIGH,
                constraint_function=lambda action: not action.get(
                    "causes_psychological_harm", False
                ),
            )
        except ValueError:
            pass

        # PRIVACY: Protect user privacy
        try:
            self.add_boundary(
                name="no_private_data_leak",
                category=BoundaryCategory.PRIVACY,
                boundary_type=BoundaryType.HARD_CONSTRAINT,
                description="Prevent leaking private user data",
                enforcement_level=EnforcementLevel.BLOCK,
                severity=ViolationSeverity.HIGH,
                constraint_function=lambda action: not action.get(
                    "leaks_private_data", False
                ),
            )
        except ValueError:
            pass

        try:
            self.add_boundary(
                name="respect_data_minimization",
                category=BoundaryCategory.PRIVACY,
                boundary_type=BoundaryType.SOFT_GUIDELINE,
                description="Collect only necessary data",
                enforcement_level=EnforcementLevel.WARN,
                severity=ViolationSeverity.MEDIUM,
                constraint_function=lambda action: action.get(
                    "data_collection_justified", True
                ),
            )
        except ValueError:
            pass

        # FAIRNESS: Ensure fair treatment
        try:
            self.add_boundary(
                name="no_demographic_discrimination",
                category=BoundaryCategory.FAIRNESS,
                boundary_type=BoundaryType.HARD_CONSTRAINT,
                description="Prevent discrimination based on protected characteristics",
                enforcement_level=EnforcementLevel.BLOCK,
                severity=ViolationSeverity.HIGH,
                constraint_function=lambda action: not action.get(
                    "discriminates_by_demographics", False
                ),
            )
        except ValueError:
            pass

        # TRANSPARENCY: Maintain explainability
        try:
            self.add_boundary(
                name="maintain_explainability",
                category=BoundaryCategory.TRANSPARENCY,
                boundary_type=BoundaryType.SOFT_GUIDELINE,
                description="Ensure actions are explainable",
                enforcement_level=EnforcementLevel.WARN,
                severity=ViolationSeverity.LOW,
                constraint_function=lambda action: action.get("is_explainable", True),
            )
        except ValueError:
            pass

        # AUTONOMY: Respect user agency
        try:
            self.add_boundary(
                name="respect_user_consent",
                category=BoundaryCategory.AUTONOMY,
                boundary_type=BoundaryType.HARD_CONSTRAINT,
                description="Require user consent for significant actions",
                enforcement_level=EnforcementLevel.BLOCK,
                severity=ViolationSeverity.HIGH,
                constraint_function=lambda action: action.get("has_user_consent", True)
                or not action.get("requires_consent", False),
            )
        except ValueError:
            pass

        # TRUTHFULNESS: Prevent deception
        try:
            self.add_boundary(
                name="no_deliberate_deception",
                category=BoundaryCategory.TRUTHFULNESS,
                boundary_type=BoundaryType.HARD_CONSTRAINT,
                description="Prevent deliberate deception",
                enforcement_level=EnforcementLevel.BLOCK,
                severity=ViolationSeverity.HIGH,
                constraint_function=lambda action: not action.get(
                    "is_deceptive", False
                ),
            )
        except ValueError:
            pass

        # RESOURCE LIMITS: Prevent abuse
        try:
            self.add_boundary(
                name="respect_computational_limits",
                category=BoundaryCategory.RESOURCE_LIMITS,
                boundary_type=BoundaryType.SOFT_GUIDELINE,
                description="Stay within computational resource limits",
                enforcement_level=EnforcementLevel.WARN,
                severity=ViolationSeverity.MEDIUM,
                constraint_rules=[
                    {
                        "type": "field_check",
                        "field": "estimated_compute_cost",
                        "condition": "less_than",
                        "value": 1000.0,  # Some reasonable limit
                    }
                ],
            )
        except ValueError:
            pass

        logger.info(
            f"Initialized default ethical boundaries (if not already present). Total: {len(self.boundaries)}"
        )

    def _record_violation(self, violation: EthicalViolation):
        """Record a violation"""
        # Ensure thread safety for appending to shared lists/deques
        with self.lock:
            self.violations.append(violation)
            # Ensure the list exists before appending
            if violation.boundary_violated not in self.violations_by_boundary:
                self.violations_by_boundary[violation.boundary_violated] = []
            self.violations_by_boundary[violation.boundary_violated].append(violation)

            self.stats[f"violations_{violation.severity.value}"] += 1
            self.stats[f"violations_{violation.category.value}"] += 1

        # Log violation
        log_level = (
            logging.CRITICAL
            if violation.severity == ViolationSeverity.CRITICAL
            else logging.ERROR
            if violation.severity == ViolationSeverity.HIGH
            else logging.WARNING
            if violation.severity == ViolationSeverity.MEDIUM
            else logging.INFO
        )  # Use INFO for LOW severity

        logger.log(
            log_level,
            f"Ethical violation detected: ID={violation.violation_id}, Boundary='{violation.boundary_violated}', Severity={violation.severity.value}, Desc='{violation.description}'",
        )

        # Call alert callback (outside lock if it's potentially blocking)
        if self.alert_callback:
            try:
                self.alert_callback(violation)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Notify integrations (outside lock if potentially blocking)
        if self.validation_tracker:
            try:
                # Prepare data carefully, ensuring serializability if needed by tracker
                validation_result_data = {
                    "valid": False,
                    "violation": violation.to_dict(),
                }
                proposal_data = {
                    "type": "ethical_check",
                    "action_hash": hash(str(violation.action)),
                }  # Hash action if too large

                self.validation_tracker.record_validation(
                    proposal=proposal_data,
                    validation_result=validation_result_data,
                    actual_outcome="violation",
                )
            except Exception as e:
                logger.debug(f"Failed to record to validation tracker: {e}")

        if self.transparency_interface:
            try:
                self.transparency_interface.record_ethical_violation(
                    violation.to_dict()
                )
            except Exception as e:
                logger.debug(f"Failed to record to transparency interface: {e}")

    def _enforce_violation(
        self, violation: EthicalViolation
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Enforce a violation based on its defined level.

        Args:
            violation: The EthicalViolation object.

        Returns:
            Tuple (is_allowed: bool, final_action: Optional[Dict[str, Any]])
            - is_allowed: True if the action (potentially modified) should proceed.
            - final_action: The action dictionary to proceed with (original or modified), or None if blocked/shutdown.
        """
        enforcement_level = violation.enforcement_action
        final_action = violation.action  # Start with original action

        # In strict mode, escalate everything below SHUTDOWN to BLOCK
        if self.strict_mode and enforcement_level != EnforcementLevel.SHUTDOWN:
            logger.warning(
                f"Strict mode: Escalating enforcement for {violation.boundary_violated} from {enforcement_level.value} to BLOCK."
            )
            enforcement_level = EnforcementLevel.BLOCK

        # Store the effective enforcement level taken in the violation record
        violation.enforcement_action = (
            enforcement_level  # Update violation object directly
        )

        success = True  # Assume success unless modification fails or shutdown occurs
        details = f"Enforcement level: {enforcement_level.value}"

        # Handle enforcement
        if enforcement_level == EnforcementLevel.MONITOR:
            # Just log (already done in _record_violation), allow action
            is_allowed = True

        elif enforcement_level == EnforcementLevel.WARN:
            # Warn but allow
            logger.warning(
                f"Ethical warning issued for violation {violation.violation_id}: {violation.description}"
            )
            is_allowed = True

        elif enforcement_level == EnforcementLevel.MODIFY:
            # Modify action if possible
            boundary = self.boundaries.get(violation.boundary_violated)
            if boundary and boundary.modification_function:
                try:
                    modified_action = boundary.modification_function(violation.action)
                    violation.action_modified = True
                    # violation.modified_action is updated later if needed
                    final_action = modified_action  # Use the modified action
                    details += f". Action modified by {boundary.name}."
                    logger.info(
                        f"Action modified by {violation.boundary_violated} due to violation {violation.violation_id}."
                    )
                    is_allowed = True
                except Exception as e:
                    logger.error(
                        f"Modification function for {violation.boundary_violated} failed: {e}. Blocking action."
                    )
                    details += f". Modification failed: {e}. Action blocked."
                    success = False
                    is_allowed = False
                    final_action = None  # Action is blocked
            else:
                # Cannot modify, block instead
                logger.warning(
                    f"No modification function for {violation.boundary_violated}. Blocking action for violation {violation.violation_id}."
                )
                details += ". No modification function available. Action blocked."
                success = False  # Enforcement effectively failed to allow progress
                is_allowed = False
                final_action = None  # Action is blocked

        elif enforcement_level == EnforcementLevel.BLOCK:
            # Block action
            logger.error(
                f"Action blocked due to ethical violation {violation.violation_id}: {violation.description}"
            )
            details += ". Action blocked."
            success = True  # Blocking is the successful enforcement here
            is_allowed = False
            final_action = None  # Action is blocked

        elif enforcement_level == EnforcementLevel.SHUTDOWN:
            # Trigger shutdown
            shutdown_reason = (
                f"Critical violation {violation.violation_id}: {violation.description}"
            )
            details += f". Triggering system shutdown. Reason: {shutdown_reason}"
            self.trigger_shutdown(shutdown_reason)
            success = True  # Shutdown is successful enforcement
            is_allowed = False
            final_action = None  # Action is blocked (system shutting down)

        else:  # Should not happen
            logger.error(
                f"Unknown enforcement level: {enforcement_level}. Allowing action with warning."
            )
            details += f". Unknown enforcement level '{enforcement_level}'. Allowing with warning."
            success = False
            is_allowed = True

        # Record enforcement action taken
        enforcement_record = EnforcementAction(
            violation=violation,
            action_taken=enforcement_level,
            success=success,
            details=details,
        )
        with self.lock:
            self.enforcement_actions.append(enforcement_record)

        return is_allowed, final_action

    def _severity_to_numeric(self, severity: ViolationSeverity) -> int:
        """Convert severity to numeric for sorting"""
        # Ensure input is the Enum member, not its value
        sev_member = (
            severity
            if isinstance(severity, ViolationSeverity)
            else ViolationSeverity(severity)
        )

        return {
            ViolationSeverity.CRITICAL: 4,
            ViolationSeverity.HIGH: 3,
            ViolationSeverity.MEDIUM: 2,
            ViolationSeverity.LOW: 1,
            ViolationSeverity.NONE: 0,
        }.get(sev_member, 0)

    def _count_by_severity(
        self, violations_list: Optional[List[EthicalViolation]] = None
    ) -> Dict[str, int]:
        """Count violations by severity (uses provided list or internal deque)"""
        counts = defaultdict(int)
        source = violations_list if violations_list is not None else self.violations
        for violation in source:
            counts[violation.severity.value] += 1
        return dict(counts)

    def _count_by_category(
        self, violations_list: Optional[List[EthicalViolation]] = None
    ) -> Dict[str, int]:
        """Count violations by category (uses provided list or internal deque)"""
        counts = defaultdict(int)
        source = violations_list if violations_list is not None else self.violations
        for violation in source:
            counts[violation.category.value] += 1
        return dict(counts)


# Module-level exports
__all__ = [
    "EthicalBoundaryMonitor",
    "EthicalBoundary",
    "EthicalViolation",
    "EnforcementAction",
    "BoundaryCategory",
    "ViolationSeverity",
    "EnforcementLevel",
    "BoundaryType",
]
