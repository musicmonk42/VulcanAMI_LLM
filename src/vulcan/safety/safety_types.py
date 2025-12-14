# safety_types.py
"""
Safety types and data structures for VULCAN-AGI Safety Module.
Contains all shared enums, dataclasses, and base interfaces used across safety components.
"""

import hashlib
import json
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================
# ENUMERATIONS
# ============================================================


class SafetyViolationType(Enum):
    """Types of safety violations that can occur in the system."""

    ENERGY = "energy"  # Energy budget exceeded
    UNCERTAINTY = "uncertainty"  # Uncertainty threshold exceeded
    IDENTITY = "identity"  # Identity drift detected
    GOAL = "goal"  # Goal misalignment
    ETHICAL = "ethical"  # Ethical violation
    OPERATIONAL = "operational"  # Operational constraint violated
    ADVERSARIAL = "adversarial"  # Adversarial attack detected
    FORMAL = "formal"  # Formal property violation
    COMPLIANCE = "compliance"  # Regulatory compliance violation
    BIAS = "bias"  # Bias detected
    PRIVACY = "privacy"  # Privacy violation
    TOOL_CONTRACT = "tool_contract"  # Tool contract violation
    TOOL_VETO = "tool_veto"  # Tool usage vetoed
    VALIDATION_ERROR = "validation_error"  # Validation system error
    PERFORMANCE = "performance"  # Performance issue


class ComplianceStandard(Enum):
    """Supported compliance and regulatory standards."""

    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    ITU_F748_53 = "itu_f748_53"  # ITU F.748.53 Autonomous Systems Standards
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # Information Security Management
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    COPPA = "coppa"  # Children's Online Privacy Protection Act
    AI_ACT = "ai_act_eu"  # EU AI Act


class ToolSafetyLevel(Enum):
    """Safety levels for tool usage authorization."""

    UNRESTRICTED = "unrestricted"  # No restrictions on usage
    MONITORED = "monitored"  # Usage allowed with monitoring
    SUPERVISED = "supervised"  # Requires active supervision
    RESTRICTED = "restricted"  # Restricted use with explicit approval
    PROHIBITED = "prohibited"  # Not allowed under any circumstances


class SafetyLevel(Enum):
    """Overall system safety levels."""

    CRITICAL = "critical"  # Critical safety level - immediate action required
    HIGH = "high"  # High safety concern
    MEDIUM = "medium"  # Medium safety concern
    LOW = "low"  # Low safety concern
    MINIMAL = "minimal"  # Minimal safety concern


class ActionType(Enum):
    """Types of actions that can be taken by the system."""

    EXPLORE = "explore"  # Exploration action
    OPTIMIZE = "optimize"  # Optimization action
    MAINTAIN = "maintain"  # Maintenance action
    WAIT = "wait"  # Wait/pause action
    SAFE_FALLBACK = "safe_fallback"  # Fallback to safe state
    EMERGENCY_STOP = "emergency_stop"  # Emergency stop action


# ============================================================
# SERIALIZABLE CONDITION CLASS
# ============================================================


@dataclass
class Condition:
    """Serializable condition for preconditions/postconditions/invariants/veto conditions.
    Replaces lambda functions to enable serialization and inspection."""

    field: str  # Field name to check in context
    operator: str  # Comparison operator
    value: Any  # Value to compare against
    description: str = ""  # Human-readable description

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against a context dictionary.

        Args:
            context: Dictionary containing values to check

        Returns:
            True if condition is satisfied, False otherwise
        """
        actual = context.get(self.field)

        # Handle None values
        if actual is None:
            # Special case: checking for None
            if self.operator == "==" and self.value is None:
                return True
            if self.operator == "!=" and self.value is None:
                return False
            # For other operators with None actual value, return False
            return False

        try:
            if self.operator == ">":
                return actual > self.value
            elif self.operator == "<":
                return actual < self.value
            elif self.operator == ">=":
                return actual >= self.value
            elif self.operator == "<=":
                return actual <= self.value
            elif self.operator == "==":
                return actual == self.value
            elif self.operator == "!=":
                return actual != self.value
            elif self.operator == "in":
                return actual in self.value
            elif self.operator == "not_in":
                return actual not in self.value
            elif self.operator == "contains":
                return self.value in actual
            elif self.operator == "not_contains":
                return self.value not in actual
            else:
                raise ValueError(f"Unknown operator: {self.operator}")
        except (TypeError, ValueError):
            # Handle comparison errors gracefully
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary for serialization."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create condition from dictionary."""
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"],
            description=data.get("description", ""),
        )

    def __str__(self) -> str:
        """String representation of condition."""
        desc = f" ({self.description})" if self.description else ""
        return f"{self.field} {self.operator} {self.value}{desc}"


# ============================================================
# DATACLASSES
# ============================================================


@dataclass
class SafetyReport:
    """Comprehensive safety assessment report."""

    safe: bool  # Overall safety determination
    confidence: float  # Confidence in safety assessment (0-1)
    violations: List[SafetyViolationType] = field(
        default_factory=list
    )  # Detected violations
    reasons: List[str] = field(default_factory=list)  # Human-readable violation reasons
    mitigations: List[str] = field(
        default_factory=list
    )  # Suggested mitigation strategies
    timestamp: float = field(default_factory=time.time)  # Report generation timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    audit_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Unique audit identifier
    compliance_checks: Dict[str, bool] = field(
        default_factory=dict
    )  # Compliance check results
    bias_scores: Dict[str, float] = field(default_factory=dict)  # Bias detection scores
    tool_vetoes: List[str] = field(default_factory=list)  # List of vetoed tools

    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate confidence is in valid range
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        # Ensure violations are unique (remove duplicates)
        self.violations = list(dict.fromkeys(self.violations))  # Preserves order

        # Validate timestamp is not in future (allow 60 second clock skew)
        if self.timestamp > time.time() + 60:
            raise ValueError(
                f"Timestamp cannot be in future: {self.timestamp} > {time.time()}"
            )

        # Ensure bias_scores are in valid range
        for key, score in self.bias_scores.items():
            if not 0 <= score <= 1:
                raise ValueError(f"Bias score for '{key}' must be 0-1, got {score}")

    def to_audit_log(self) -> Dict[str, Any]:
        """Convert report to audit log entry format."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "safe": self.safe,
            "confidence": self.confidence,
            "violations": [v.value for v in self.violations],
            "reasons": self.reasons,
            "mitigations": self.mitigations,
            "compliance_checks": self.compliance_checks,
            "bias_scores": self.bias_scores,
            "tool_vetoes": self.tool_vetoes,
            "metadata": self.metadata,
            "severity": self._calculate_severity(),
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_audit_log(), indent=2)

    def _calculate_severity(self) -> str:
        """Calculate overall severity level based on violations."""
        if not self.safe:
            critical_violations = [
                SafetyViolationType.ADVERSARIAL,
                SafetyViolationType.COMPLIANCE,
                SafetyViolationType.PRIVACY,
            ]
            if any(v in critical_violations for v in self.violations):
                return SafetyLevel.CRITICAL.value
            elif len(self.violations) > 3:
                return SafetyLevel.HIGH.value
            else:
                return SafetyLevel.MEDIUM.value
        return SafetyLevel.MINIMAL.value

    def add_violation(self, violation_type: SafetyViolationType, reason: str):
        """Add a violation to the report."""
        if violation_type not in self.violations:
            self.violations.append(violation_type)
        self.reasons.append(reason)
        self.safe = False
        self.confidence = min(self.confidence, 0.5)

    def merge(self, other: "SafetyReport") -> "SafetyReport":
        """Merge another safety report into this one."""
        self.safe = self.safe and other.safe
        self.confidence = min(self.confidence, other.confidence)
        self.violations.extend(other.violations)
        self.violations = list(dict.fromkeys(self.violations))  # Remove duplicates
        self.reasons.extend(other.reasons)
        self.mitigations.extend(other.mitigations)
        self.compliance_checks.update(other.compliance_checks)
        self.bias_scores.update(other.bias_scores)
        self.tool_vetoes.extend(other.tool_vetoes)
        self.tool_vetoes = list(dict.fromkeys(self.tool_vetoes))  # Remove duplicates
        self.metadata.update(other.metadata)
        return self


@dataclass
class SafetyConstraint:
    """Definition of a safety constraint."""

    name: str  # Constraint name
    type: str  # Constraint type (hard/soft)
    check_function: Callable  # Function to check constraint
    threshold: float = 0.0  # Threshold value
    priority: int = 1  # Priority level (higher = more important)
    active: bool = True  # Whether constraint is active
    compliance_standard: Optional[ComplianceStandard] = (
        None  # Associated compliance standard
    )
    description: str = ""  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def check(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Execute constraint check."""
        try:
            result = self.check_function(action, context)
            if isinstance(result, bool):
                return result, 1.0 if result else 0.0
            elif isinstance(result, tuple) and len(result) == 2:
                return result
            else:
                return False, 0.0
        except Exception:
            # Log error and return failure
            return False, 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraint to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type,
            "threshold": self.threshold,
            "priority": self.priority,
            "active": self.active,
            "compliance_standard": (
                self.compliance_standard.value if self.compliance_standard else None
            ),
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class RollbackSnapshot:
    """Snapshot for system state rollback capability."""

    snapshot_id: str  # Unique snapshot identifier
    timestamp: float  # Snapshot creation timestamp
    state: Dict[str, Any]  # System state at snapshot time
    action_log: List[Dict[str, Any]]  # Log of actions leading to this state
    metadata: Dict[str, Any]  # Additional snapshot metadata
    checksum: Optional[str] = None  # State checksum for integrity

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of state."""
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify snapshot integrity using checksum."""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "state_size": len(json.dumps(self.state, default=str)),
            "action_count": len(self.action_log),
            "checksum": self.checksum,
            "metadata": self.metadata,
        }


@dataclass
class ToolSafetyContract:
    """Safety contract specification for a tool.

    Now uses serializable Condition objects instead of lambda functions,
    enabling persistence, inspection, and modification of contracts.
    """

    tool_name: str  # Tool identifier
    safety_level: ToolSafetyLevel  # Required safety level
    preconditions: List[Condition]  # Conditions that must be True before use
    postconditions: List[Condition]  # Conditions to check after execution
    invariants: List[Condition]  # Conditions that must always be True
    max_frequency: float  # Maximum calls per minute
    max_resource_usage: Dict[str, float]  # Resource usage limits
    required_confidence: float  # Minimum confidence to use tool
    veto_conditions: List[Condition]  # Conditions that trigger automatic veto
    risk_score: float  # Base risk score (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    description: str = ""  # Human-readable description
    version: str = "1.0.0"  # Contract version

    def validate_preconditions(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all preconditions."""
        failures = []
        for i, condition in enumerate(self.preconditions):
            try:
                if not condition.evaluate(context):
                    desc = condition.description or f"Precondition {i}"
                    failures.append(
                        f"{desc} failed: {condition.field} {condition.operator} {condition.value}"
                    )
            except Exception as e:
                failures.append(f"Precondition {i} error: {str(e)}")
        return len(failures) == 0, failures

    def validate_postconditions(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all postconditions."""
        failures = []
        for i, condition in enumerate(self.postconditions):
            try:
                if not condition.evaluate(result):
                    desc = condition.description or f"Postcondition {i}"
                    failures.append(
                        f"{desc} failed: {condition.field} {condition.operator} {condition.value}"
                    )
            except Exception as e:
                failures.append(f"Postcondition {i} error: {str(e)}")
        return len(failures) == 0, failures

    def check_invariants(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check all invariants."""
        failures = []
        for i, condition in enumerate(self.invariants):
            try:
                if not condition.evaluate(context):
                    desc = condition.description or f"Invariant {i}"
                    failures.append(
                        f"{desc} violated: {condition.field} {condition.operator} {condition.value}"
                    )
            except Exception as e:
                failures.append(f"Invariant {i} error: {str(e)}")
        return len(failures) == 0, failures

    def check_veto(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if any veto conditions are triggered."""
        veto_reasons = []
        for i, condition in enumerate(self.veto_conditions):
            try:
                if condition.evaluate(context):
                    desc = condition.description or f"Veto condition {i}"
                    veto_reasons.append(
                        f"{desc} triggered: {condition.field} {condition.operator} {condition.value}"
                    )
            except Exception as e:
                veto_reasons.append(f"Veto condition {i} error: {str(e)}")
        return len(veto_reasons) > 0, veto_reasons

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary representation for serialization."""
        return {
            "tool_name": self.tool_name,
            "safety_level": self.safety_level.value,
            "preconditions": [c.to_dict() for c in self.preconditions],
            "postconditions": [c.to_dict() for c in self.postconditions],
            "invariants": [c.to_dict() for c in self.invariants],
            "veto_conditions": [c.to_dict() for c in self.veto_conditions],
            "max_frequency": self.max_frequency,
            "max_resource_usage": self.max_resource_usage,
            "required_confidence": self.required_confidence,
            "risk_score": self.risk_score,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSafetyContract":
        """Create contract from dictionary representation."""
        return cls(
            tool_name=data["tool_name"],
            safety_level=ToolSafetyLevel(data["safety_level"]),
            preconditions=[Condition.from_dict(c) for c in data["preconditions"]],
            postconditions=[Condition.from_dict(c) for c in data["postconditions"]],
            invariants=[Condition.from_dict(c) for c in data["invariants"]],
            veto_conditions=[Condition.from_dict(c) for c in data["veto_conditions"]],
            max_frequency=data["max_frequency"],
            max_resource_usage=data["max_resource_usage"],
            required_confidence=data["required_confidence"],
            risk_score=data["risk_score"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert contract to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ToolSafetyContract":
        """Create contract from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================================
# BASE CLASSES AND INTERFACES
# ============================================================


class SafetyValidator:
    """Base interface for safety validators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration."""
        self.config = config or {}

    def validate_action(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """
        Validate an action for safety.

        Returns:
            Tuple of (is_safe, reason, confidence)
        """
        # Base implementation - override in subclasses
        return True, "OK", 1.0

    def get_config(self) -> Dict[str, Any]:
        """Get validator configuration."""
        return self.config


# [!!!] START OF CRITICAL FIX [!!!]
# Replaced the "smart" stub with a "dumb" one to break the circular import.
class GovernanceOrchestrator:
    """
    [FIXED] Base interface/stub for governance orchestration.
    This version breaks the circular import by NOT importing GovernanceManager.
    The real EnhancedSafetyValidator will instantiate the *real* GovernanceManager.
    """

    def __init__(self, policies: Optional[Dict[str, Any]] = None):
        """Initialize orchestrator with policies."""
        self.policies = policies or {}
        self._initialized = False
        # Note: Cannot use logger here as it might not be configured
        # when this file is first imported.

    def check_compliance(self, plan: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance of a plan.
        """
        # Stub implementation fallback - permissive for development
        return {
            "compliant": True,
            "violations": [],
            "compliance_score": 1.0,
            "checked_policies": list(self.policies.keys()),
            "stub_mode": True,
        }

    def enforce_policies(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce governance policies on actions.
        """
        # Stub implementation - pass through (permissive for development)
        return actions

    def request_approval(
        self,
        action: Dict[str, Any],
        policy_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request approval for an action.
        """
        # Stub implementation - auto-approve (permissive for development)
        return {
            "approved": True,
            "decision_id": f"stub_{uuid.uuid4()}",
            "action_id": action.get("id", str(uuid.uuid4())),
            "policy_applied": "auto_approve_stub",
            "governance_level": "autonomous",
            "confidence": 0.5,
            "reasoning": "Stub governance - auto-approved (no real governance available)",
            "response_time": 0.001,
            "stub_mode": True,
            "timestamp": time.time(),
        }

    def get_policies(self) -> Dict[str, Any]:
        """Get current policies."""
        return self.policies

    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "policies": len(self.policies),
            "active_decisions": 0,
            "pending_approvals": 0,
            "stub_mode": True,
            "error_mode": False,
        }

    def is_initialized(self) -> bool:
        """Check if real governance manager is initialized."""
        return self._initialized

    def shutdown(self):
        """Shutdown governance manager if initialized."""
        pass  # Stub has nothing to shut down


# [!!!] END OF CRITICAL FIX [!!!]


class NSOAligner:
    """Base interface for Neural-Symbolic-Optimization alignment."""

    def __init__(self, policies: Optional[Dict[str, Any]] = None):
        """Initialize aligner with policies."""
        self.policies = policies or {}

    def scan_external(self, proposal: Any) -> bool:
        """
        Scan external proposal for safety.

        Returns:
            Boolean indicating if proposal is safe
        """
        # Base implementation - override in subclasses
        return True

    def align_action(self, plan: Any, thresholds: Dict[str, float]) -> Any:
        """
        Align an action plan with safety thresholds.

        Returns:
            Aligned plan
        """
        # Base implementation - override in subclasses
        return plan


class ExplainabilityNode:
    """Base interface for explainability generation."""

    def execute(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for data.

        Returns:
            Dictionary with explanation
        """
        # Base implementation - override in subclasses
        return {
            "explanation_summary": "Base explanation for the provided data.",
            "method": "default",
            "confidence": 0.5,
            "timestamp": time.time(),
        }


# ============================================================
# UTILITY CLASSES
# ============================================================


class SafetyMetrics:
    """Container for safety-related metrics."""

    def __init__(self):
        """Initialize metrics container."""
        self.total_checks = 0
        self.safe_decisions = 0
        self.unsafe_decisions = 0
        self.violations_by_type = {v: 0 for v in SafetyViolationType}
        self.average_confidence = 0.0
        self.last_updated = time.time()

    def update(self, report: SafetyReport):
        """Update metrics with a safety report."""
        self.total_checks += 1
        if report.safe:
            self.safe_decisions += 1
        else:
            self.unsafe_decisions += 1

        for violation in report.violations:
            self.violations_by_type[violation] += 1

        # Update rolling average confidence
        self.average_confidence = (
            self.average_confidence * (self.total_checks - 1) + report.confidence
        ) / self.total_checks
        self.last_updated = time.time()

    def get_safety_rate(self) -> float:
        """Calculate safety rate."""
        if self.total_checks == 0:
            return 1.0
        return self.safe_decisions / self.total_checks

    def get_top_violations(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get top N violation types."""
        sorted_violations = sorted(
            [(v.value, count) for v, count in self.violations_by_type.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_violations[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_checks": self.total_checks,
            "safe_decisions": self.safe_decisions,
            "unsafe_decisions": self.unsafe_decisions,
            "safety_rate": self.get_safety_rate(),
            "average_confidence": self.average_confidence,
            "top_violations": self.get_top_violations(),
            "last_updated": self.last_updated,
            "iso_timestamp": datetime.fromtimestamp(self.last_updated).isoformat(),
        }


class SafetyException(Exception):
    """Custom exception for safety-related errors."""

    def __init__(
        self,
        message: str,
        violation_type: SafetyViolationType,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize safety exception."""
        super().__init__(message)
        self.violation_type = violation_type
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": str(self),
            "violation_type": self.violation_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc(),
        }


# ============================================================
# CONFIGURATION CLASSES
# ============================================================


@dataclass
class SafetyConfig:
    """Configuration for safety system."""

    enable_adversarial_testing: bool = True
    enable_compliance_checking: bool = True
    enable_bias_detection: bool = True
    enable_rollback: bool = True
    enable_audit_logging: bool = True
    enable_tool_safety: bool = True

    safety_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "uncertainty_max": 0.9,
            "identity_drift_max": 0.5,
            "bias_threshold": 0.2,
            "confidence_min": 0.6,
        }
    )

    compliance_standards: List[ComplianceStandard] = field(
        default_factory=lambda: [
            ComplianceStandard.GDPR,
            ComplianceStandard.ITU_F748_53,
        ]
    )

    rollback_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_snapshots": 100,
            "auto_rollback_on_critical": True,
            "quarantine_duration_seconds": 3600,
        }
    )

    audit_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "log_path": "safety_audit",
            "redact_sensitive": True,
            "rotation_days": 30,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyConfig":
        """Create configuration from dictionary."""
        # Handle compliance standards enum conversion
        if "compliance_standards" in data:
            data["compliance_standards"] = [
                ComplianceStandard(s) if isinstance(s, str) else s
                for s in data["compliance_standards"]
            ]
        return cls(**data)


# ============================================================
# TYPE ALIASES
# ============================================================

SafetyCheck = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str, float]]
PolicyFunction = Callable[[Any], bool]
VetoFunction = Callable[[Dict[str, Any]], bool]
ConditionFunction = Callable[[Dict[str, Any]], bool]
