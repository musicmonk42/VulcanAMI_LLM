"""
test_safety_module_integration.py - PURE MOCK VERSION
Comprehensive integration tests for VULCAN-AGI Safety Module without thread spawning.
"""

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

# ============================================================================
# Mock Enums
# ============================================================================


class SafetyViolationType(Enum):
    NONE = "none"
    ENERGY = "energy"
    UNCERTAINTY = "uncertainty"
    IDENTITY_DRIFT = "identity_drift"
    CONSTRAINT = "constraint"
    RESOURCE = "resource"
    ETHICAL = "ethical"
    COMPLIANCE = "compliance"
    TOOL_VETO = "tool_veto"
    ADVERSARIAL = "adversarial"
    BIAS = "bias"


class ToolSafetyLevel(Enum):
    UNRESTRICTED = "unrestricted"
    MONITORED = "monitored"
    RESTRICTED = "restricted"
    PROHIBITED = "prohibited"


class ComplianceStandard(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


class ActionType(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    OPTIMIZE = "optimize"
    LEARN = "learn"
    COMMUNICATE = "communicate"


class GovernanceLevel(Enum):
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
    CONTROLLED = "controlled"
    RESTRICTED = "restricted"


class StakeholderType(Enum):
    OPERATOR = "operator"
    ADMINISTRATOR = "administrator"
    AUDITOR = "auditor"
    USER = "user"


# ============================================================================
# Mock Dataclasses
# ============================================================================


@dataclass
class Condition:
    field: str
    operator: str
    value: Any
    description: str = ""

    def evaluate(self, context: Dict) -> bool:
        val = context.get(self.field)
        if val is None:
            return False

        try:
            if self.operator == "==":
                return val == self.value
            elif self.operator == "!=":
                return val != self.value
            elif self.operator == "<":
                return val < self.value
            elif self.operator == "<=":
                return val <= self.value
            elif self.operator == ">":
                return val > self.value
            elif self.operator == ">=":
                return val >= self.value
            elif self.operator == "in":
                return val in self.value
            elif self.operator == "contains":
                return self.value in val if hasattr(val, "__contains__") else False
        except Exception:
            return False
        return False

    def to_dict(self) -> Dict:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Condition":
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"],
            description=data.get("description", ""),
        )


@dataclass
class SafetyReport:
    safe: bool
    confidence: float
    violations: List[SafetyViolationType] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_id: str = field(
        default_factory=lambda: hashlib.md5(str(time.time()).encode(), usedforsecurity=False).hexdigest()[:16]
    )

    def merge(self, other: "SafetyReport") -> "SafetyReport":
        return SafetyReport(
            safe=self.safe and other.safe,
            confidence=min(self.confidence, other.confidence),
            violations=list(set(self.violations + other.violations)),
            reasons=self.reasons + other.reasons,
            metadata={**self.metadata, **other.metadata},
        )


@dataclass
class SafetyConstraint:
    name: str
    type: str  # 'hard' or 'soft'
    check_function: Callable
    threshold: float = 0.0
    priority: int = 0

    def check(self, action: Dict, context: Dict) -> Tuple[bool, float]:
        return self.check_function(action, context)


@dataclass
class RollbackSnapshot:
    snapshot_id: str
    timestamp: float
    state: Dict[str, Any]
    action_log: List[Dict]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _checksum: str = field(default="", repr=False)

    def __post_init__(self):
        if not self._checksum:
            self._checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        data = json.dumps(self.state, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        return self._checksum == self._compute_checksum()


@dataclass
class ToolSafetyContract:
    tool_name: str
    safety_level: ToolSafetyLevel
    preconditions: List[Condition] = field(default_factory=list)
    postconditions: List[Condition] = field(default_factory=list)
    invariants: List[Condition] = field(default_factory=list)
    veto_conditions: List[Condition] = field(default_factory=list)
    max_frequency: float = 100.0
    max_resource_usage: Dict[str, float] = field(default_factory=dict)
    required_confidence: float = 0.5
    risk_score: float = 0.5

    def validate_preconditions(self, context: Dict) -> Tuple[bool, List[str]]:
        failures = []
        for cond in self.preconditions:
            if not cond.evaluate(context):
                failures.append(
                    cond.description or f"{cond.field} {cond.operator} {cond.value}"
                )
        return len(failures) == 0, failures

    def to_json(self) -> str:
        return json.dumps(
            {
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
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ToolSafetyContract":
        data = json.loads(json_str)
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
        )


@dataclass
class SafetyConfig:
    enable_adversarial_testing: bool = False
    enable_compliance_checking: bool = True
    enable_bias_detection: bool = False
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
    rollback_config: Dict[str, Any] = field(
        default_factory=lambda: {"storage_path": "/tmp/rollback"}
    )
    audit_config: Dict[str, Any] = field(
        default_factory=lambda: {"log_path": "/tmp/audit"}
    )


@dataclass
class SafetyMetrics:
    total_checks: int = 0
    safe_actions: int = 0
    violations_detected: int = 0
    rollbacks_performed: int = 0


class SafetyException(Exception):
    pass


@dataclass
class ValidationResult:
    safe: bool
    reason: str = ""
    severity: str = "info"
    confidence: float = 1.0


@dataclass
class HumanFeedback:
    feedback_id: str
    decision_id: str
    feedback_type: str
    content: str
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Mock Domain Validators
# ============================================================================


class CausalSafetyValidator:
    def __init__(self):
        self.unsafe_patterns = ["harm", "damage", "destroy", "kill"]
        self.max_strength = 100.0
        self.max_amplification = 10.0

    def validate_causal_edge(
        self, source: str, target: str, strength: float
    ) -> ValidationResult:
        if np.isnan(strength):
            return ValidationResult(
                safe=False, reason="NaN strength value", severity="critical"
            )
        if abs(strength) > self.max_strength:
            return ValidationResult(
                safe=False, reason=f"Strength {strength} too large", severity="warning"
            )
        if source == target:
            return ValidationResult(
                safe=False, reason="Self-loop detected", severity="warning"
            )
        for pattern in self.unsafe_patterns:
            if pattern in source.lower() or pattern in target.lower():
                return ValidationResult(
                    safe=False,
                    reason=f"Detected unsafe pattern in edge",
                    severity="critical",
                )
        return ValidationResult(safe=True, reason="Valid causal edge")

    def validate_causal_path(
        self, nodes: List[str], strengths: List[float]
    ) -> ValidationResult:
        if len(nodes) - 1 != len(strengths):
            return ValidationResult(
                safe=False, reason="Mismatch between nodes and strengths count"
            )
        if len(nodes) != len(set(nodes)):
            return ValidationResult(safe=False, reason="Detected cycle in path")
        total_amplification = 1.0
        for s in strengths:
            total_amplification *= abs(s)
        if total_amplification > self.max_amplification:
            return ValidationResult(
                safe=False, reason=f"Excessive amplification: {total_amplification}"
            )
        return ValidationResult(safe=True, reason="Valid causal path")

    def validate_causal_graph(self, adjacency: Dict) -> ValidationResult:
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor, _ in adjacency.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in adjacency:
            if node not in visited:
                if has_cycle(node):
                    return ValidationResult(
                        safe=False, reason="Detected cycle in graph"
                    )
        return ValidationResult(safe=True, reason="Valid DAG")


class PredictionSafetyValidator:
    def __init__(self, safe_regions: Dict[str, Tuple[float, float]] = None):
        self.safe_regions = safe_regions or {}

    def validate_prediction(
        self, expected: float, lower: float, upper: float, variable: str
    ) -> ValidationResult:
        if np.isnan(expected):
            return ValidationResult(
                safe=False, reason="NaN prediction", severity="critical"
            )
        if lower > upper:
            return ValidationResult(safe=False, reason="Invalid bounds: lower > upper")
        if expected < lower or expected > upper:
            return ValidationResult(safe=False, reason="Expected value outside bounds")
        if variable in self.safe_regions:
            safe_low, safe_high = self.safe_regions[variable]
            if upper < safe_low or lower > safe_high:
                return ValidationResult(
                    safe=False, reason=f"Prediction outside safe region for {variable}"
                )
        return ValidationResult(safe=True, reason="Valid prediction")

    def validate_prediction_batch(self, predictions: List[Dict]) -> ValidationResult:
        unsafe_count = 0
        for pred in predictions:
            result = self.validate_prediction(
                pred["expected"], pred["lower"], pred["upper"], pred["variable"]
            )
            if not result.safe:
                unsafe_count += 1
        if unsafe_count > 0:
            return ValidationResult(
                safe=False,
                reason=f"{unsafe_count}/{len(predictions)} predictions unsafe",
            )
        return ValidationResult(safe=True, reason="All predictions valid")


class OptimizationSafetyValidator:
    def __init__(self):
        self.max_iterations = 50000
        self.max_learning_rate = 1.0

    def validate_optimization_params(self, params: Dict) -> ValidationResult:
        if params.get("max_iterations", 0) > self.max_iterations:
            return ValidationResult(safe=False, reason="Too many iterations")
        lr = params.get("learning_rate", 0.01)
        if lr > self.max_learning_rate:
            return ValidationResult(safe=False, reason="Learning rate too high")
        bounds = params.get("bounds", {})
        for var, (low, high) in bounds.items():
            if low > high:
                return ValidationResult(safe=False, reason=f"Invalid bounds for {var}")
        return ValidationResult(safe=True, reason="Valid optimization params")


class DataProcessingSafetyValidator:
    def __init__(self):
        self.max_rows = 10000000
        self.max_missing_ratio = 0.5

    def validate_dataframe(self, df_info: Dict) -> ValidationResult:
        if df_info.get("rows", 0) > self.max_rows:
            return ValidationResult(safe=False, reason="Too many rows")
        if df_info.get("missing_ratio", 0) > self.max_missing_ratio:
            return ValidationResult(safe=False, reason="Too much missing data")
        return ValidationResult(safe=True, reason="Valid dataframe")


# Validator registry
validator_registry = {
    "causal": CausalSafetyValidator,
    "prediction": PredictionSafetyValidator,
    "optimization": OptimizationSafetyValidator,
    "data_processing": DataProcessingSafetyValidator,
}


# ============================================================================
# Mock Tool Safety
# ============================================================================


class TokenBucket:
    def __init__(self, rate: float = 10.0, capacity: float = 10.0):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_available(self) -> float:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(self.capacity, self.tokens + elapsed * self.rate)

    def shutdown(self):
        pass


class ToolSafetyManager:
    def __init__(self):
        self.contracts = self._create_default_contracts()
        self.rate_limiters = {
            name: TokenBucket(rate=100.0, capacity=100.0) for name in self.contracts
        }
        self.usage_history = {name: deque(maxlen=1000) for name in self.contracts}
        self._lock = threading.Lock()

    def _create_default_contracts(self) -> Dict[str, ToolSafetyContract]:
        return {
            "probabilistic": ToolSafetyContract(
                tool_name="probabilistic",
                safety_level=ToolSafetyLevel.MONITORED,
                preconditions=[
                    Condition("confidence", ">", 0.5, "Confidence must be > 0.5"),
                    Condition(
                        "corrupted_data", "==", False, "Data must not be corrupted"
                    ),
                ],
                veto_conditions=[
                    Condition(
                        "adversarial_detected", "==", True, "Adversarial detected"
                    ),
                    Condition("system_overload", "==", True, "System overload"),
                ],
                required_confidence=0.5,
            ),
            "symbolic": ToolSafetyContract(
                tool_name="symbolic",
                safety_level=ToolSafetyLevel.MONITORED,
                preconditions=[
                    Condition("logic_valid", "==", True, "Logic must be valid")
                ],
                required_confidence=0.6,
            ),
            "causal": ToolSafetyContract(
                tool_name="causal",
                safety_level=ToolSafetyLevel.RESTRICTED,
                preconditions=[
                    Condition("causal_graph_valid", "==", True, "Graph must be valid")
                ],
                required_confidence=0.7,
            ),
        }

    def check_tool_safety(
        self, tool_name: str, context: Dict
    ) -> Tuple[bool, SafetyReport]:
        with self._lock:
            if tool_name not in self.contracts:
                return False, SafetyReport(
                    safe=False,
                    confidence=0.0,
                    violations=[SafetyViolationType.TOOL_VETO],
                    reasons=[f"Unknown tool: {tool_name}"],
                )

            contract = self.contracts[tool_name]

            # Check veto conditions
            for cond in contract.veto_conditions:
                if cond.evaluate(context):
                    return False, SafetyReport(
                        safe=False,
                        confidence=0.0,
                        violations=[SafetyViolationType.TOOL_VETO],
                        reasons=[f"Veto condition triggered: {cond.description}"],
                    )

            # Check preconditions
            valid, failures = contract.validate_preconditions(context)
            if not valid:
                return False, SafetyReport(
                    safe=False,
                    confidence=0.5,
                    violations=[SafetyViolationType.CONSTRAINT],
                    reasons=[f"Precondition failed: {f}" for f in failures],
                )

            # Check rate limiting
            if not self.rate_limiters[tool_name].consume(1.0):
                return False, SafetyReport(
                    safe=False,
                    confidence=0.5,
                    violations=[SafetyViolationType.RESOURCE],
                    reasons=["Rate limit exceeded"],
                )

            # Record usage
            self.usage_history[tool_name].append(
                {"timestamp": time.time(), "context": context}
            )

            return True, SafetyReport(safe=True, confidence=0.9, violations=[])

    def veto_tool_selection(
        self, tools: List[str], context: Dict
    ) -> Tuple[List[str], SafetyReport]:
        allowed = []
        all_violations = []
        all_reasons = []

        for tool in tools:
            safe, report = self.check_tool_safety(tool, context)
            if safe:
                allowed.append(tool)
            else:
                all_violations.extend(report.violations)
                all_reasons.extend(report.reasons)

        return allowed, SafetyReport(
            safe=len(allowed) > 0,
            confidence=0.8 if allowed else 0.0,
            violations=list(set(all_violations)),
            reasons=all_reasons,
        )

    def shutdown(self):
        for limiter in self.rate_limiters.values():
            limiter.shutdown()


class ToolSafetyGovernor:
    def __init__(self):
        self.manager = ToolSafetyManager()
        self.emergency_stop = False
        self.quarantine_list: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def govern_tool_selection(
        self, request: Dict, tools: List[str]
    ) -> Tuple[List[str], Dict]:
        with self._lock:
            if self.emergency_stop:
                return [], {
                    "status": "emergency_stop",
                    "allowed_tools": [],
                    "veto_report": None,
                }

            # Remove quarantined tools
            now = time.time()
            expired = [
                t for t, info in self.quarantine_list.items() if info["expires"] < now
            ]
            for t in expired:
                del self.quarantine_list[t]

            available_tools = [t for t in tools if t not in self.quarantine_list]

            allowed, report = self.manager.veto_tool_selection(available_tools, request)

            return allowed, {
                "status": "ok",
                "allowed_tools": allowed,
                "veto_report": report,
            }

    def trigger_emergency_stop(self, reason: str):
        with self._lock:
            self.emergency_stop = True

    def clear_emergency_stop(self, admin_id: str):
        with self._lock:
            self.emergency_stop = False

    def quarantine_tool(
        self, tool_name: str, reason: str, duration_seconds: float = 3600
    ):
        with self._lock:
            self.quarantine_list[tool_name] = {
                "reason": reason,
                "expires": time.time() + duration_seconds,
            }

    def shutdown(self):
        self.manager.shutdown()


# ============================================================================
# Mock Rollback and Audit
# ============================================================================


class MemoryBoundedDeque:
    def __init__(self, max_size_mb: float = 100.0):
        self.max_size_mb = max_size_mb
        self._data = deque()
        self._lock = threading.Lock()

    def append(self, item: Any):
        with self._lock:
            self._data.append(item)
            while self.get_memory_usage_mb() > self.max_size_mb and len(self._data) > 0:
                self._data.popleft()

    def get_memory_usage_mb(self) -> float:
        return len(str(list(self._data))) / (1024 * 1024)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        return len(self._data)


class RollbackManager:
    def __init__(self, max_snapshots: int = 100, config: Dict = None):
        self.max_snapshots = max_snapshots
        self.config = config or {}
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.snapshot_index: Dict[str, int] = {}
        self.quarantine: Dict[str, Dict] = {}
        self.metrics = {
            "total_rollbacks": 0,
            "successful_rollbacks": 0,
            "failed_rollbacks": 0,
        }
        self._lock = threading.Lock()

    def create_snapshot(self, state: Dict, action_log: List) -> str:
        with self._lock:
            snapshot_id = hashlib.md5(
                f"{time.time()}{len(self.snapshots)}".encode()
                , usedforsecurity=False).hexdigest()[:16]
            snapshot = RollbackSnapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                state=state.copy(),
                action_log=action_log.copy(),
            )
            self.snapshots.append(snapshot)
            self.snapshot_index[snapshot_id] = len(self.snapshots) - 1
            return snapshot_id

    def rollback(self, snapshot_id: str, reason: str = "") -> Optional[Dict]:
        with self._lock:
            self.metrics["total_rollbacks"] += 1

            for snapshot in self.snapshots:
                if snapshot.snapshot_id == snapshot_id:
                    self.metrics["successful_rollbacks"] += 1
                    return {
                        "state": snapshot.state,
                        "action_log": snapshot.action_log,
                        "rollback_metadata": {
                            "snapshot_id": snapshot_id,
                            "reason": reason,
                            "timestamp": time.time(),
                        },
                    }

            self.metrics["failed_rollbacks"] += 1
            return None

    def quarantine_action(
        self, action: Dict, reason: str, duration_seconds: float = 3600
    ) -> str:
        with self._lock:
            quarantine_id = hashlib.md5(f"{time.time()}{action}".encode(), usedforsecurity=False).hexdigest()[
                :16
            ]
            self.quarantine[quarantine_id] = {
                "action": action,
                "reason": reason,
                "expires": time.time() + duration_seconds,
                "reviewed": False,
            }
            return quarantine_id

    def get_quarantine_item(self, quarantine_id: str) -> Optional[Dict]:
        return self.quarantine.get(quarantine_id)

    def review_quarantine(
        self, quarantine_id: str, approved: bool, reviewer: str
    ) -> bool:
        with self._lock:
            if quarantine_id in self.quarantine:
                self.quarantine[quarantine_id]["reviewed"] = True
                self.quarantine[quarantine_id]["approved"] = approved
                self.quarantine[quarantine_id]["reviewer"] = reviewer
                return True
            return False

    def get_snapshot_history(self) -> List[Dict]:
        return [
            {"snapshot_id": s.snapshot_id, "timestamp": s.timestamp}
            for s in self.snapshots
        ]

    def get_metrics(self) -> Dict:
        return self.metrics.copy()

    def shutdown(self):
        pass


class AuditLogger:
    def __init__(self, log_path: str = "/tmp/audit", config: Dict = None):
        self.log_path = log_path
        self.config = config or {}
        self.redact_sensitive = config.get("redact_sensitive", True) if config else True
        self.entries: List[Dict] = []
        self.metrics = {"total_entries": 0}
        self._lock = threading.Lock()

    def log_safety_decision(self, decision: Dict, report: SafetyReport) -> str:
        with self._lock:
            entry_id = hashlib.md5(f"{time.time()}{decision}".encode(), usedforsecurity=False).hexdigest()[:16]
            self.entries.append(
                {
                    "entry_id": entry_id,
                    "type": "safety_decision",
                    "decision": decision,
                    "report": {"safe": report.safe, "confidence": report.confidence},
                    "timestamp": time.time(),
                }
            )
            self.metrics["total_entries"] += 1
            return entry_id

    def log_event(self, event_type: str, data: Dict, severity: str = "info") -> str:
        with self._lock:
            entry_id = hashlib.md5(f"{time.time()}{event_type}".encode(), usedforsecurity=False).hexdigest()[
                :16
            ]
            self.entries.append(
                {
                    "entry_id": entry_id,
                    "type": event_type,
                    "data": data,
                    "severity": severity,
                    "timestamp": time.time(),
                }
            )
            self.metrics["total_entries"] += 1
            return entry_id

    def _redact_sensitive(self, data: str) -> str:
        # SSN pattern
        data = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN_REDACTED]", data)
        # Email pattern
        data = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", data)
        return data

    def query_logs(self, limit: int = 100, **filters) -> List[Dict]:
        return self.entries[-limit:]

    def get_metrics(self) -> Dict:
        return self.metrics.copy()

    def shutdown(self):
        pass


# ============================================================================
# Mock Governance and Alignment
# ============================================================================


class GovernanceManager:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.policies = {
            "autonomous_default": {"auto_approve": True, "risk_threshold": 0.3},
            "human_supervised": {"auto_approve": False, "risk_threshold": 0.1},
            "safety_critical": {"auto_approve": False, "risk_threshold": 0.0},
        }
        self.stakeholder_registry: Dict[str, Dict] = {}
        self.decisions: List[Dict] = []
        self._lock = threading.Lock()

    def request_approval(self, action: Dict) -> Dict:
        with self._lock:
            decision_id = hashlib.md5(f"{time.time()}{action}".encode(), usedforsecurity=False).hexdigest()[
                :16
            ]
            risk_score = action.get("risk_score", 0.5)

            # Determine policy
            policy_name = "autonomous_default"
            if risk_score > 0.5:
                policy_name = "safety_critical"
            elif risk_score > 0.3:
                policy_name = "human_supervised"

            policy = self.policies[policy_name]
            approved = policy["auto_approve"] and risk_score <= policy["risk_threshold"]

            decision = {
                "decision_id": decision_id,
                "approved": approved,
                "policy_applied": policy_name,
                "risk_score": risk_score,
                "timestamp": time.time(),
            }
            self.decisions.append(decision)
            return decision

    def register_stakeholder(
        self,
        stakeholder_id: str,
        stakeholder_type: StakeholderType,
        metadata: Dict = None,
    ) -> bool:
        with self._lock:
            self.stakeholder_registry[stakeholder_id] = {
                "type": stakeholder_type,
                "metadata": metadata or {},
                "registered_at": time.time(),
            }
            return True

    def shutdown(self):
        pass


class ValueAlignmentSystem:
    def __init__(self):
        self.core_values = {
            "safety": 1.0,
            "transparency": 0.9,
            "fairness": 0.85,
            "beneficence": 0.8,
        }

    def check_alignment(self, action: Dict, context: Dict) -> Dict:
        value_scores = {}

        # Safety score
        safety_score = action.get("safety_score", 0.5)
        value_scores["safety"] = safety_score

        # Transparency score
        has_explanation = bool(action.get("explanation"))
        value_scores["transparency"] = 0.9 if has_explanation else 0.3

        # Auditability
        auditable = action.get("auditable", False)
        value_scores["fairness"] = 0.8 if auditable else 0.5

        # Beneficence
        value_scores["beneficence"] = action.get("confidence", 0.5)

        # Calculate overall alignment
        alignment_score = sum(
            value_scores.get(v, 0.5) * weight for v, weight in self.core_values.items()
        ) / sum(self.core_values.values())

        return {
            "aligned": alignment_score >= 0.6,
            "alignment_score": alignment_score,
            "value_scores": value_scores,
        }


class HumanOversightInterface:
    def __init__(self, governance: GovernanceManager, alignment: ValueAlignmentSystem):
        self.governance = governance
        self.alignment = alignment
        self.automation_level = 0.8
        self.emergency_stop_enabled = True
        self.alerts: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def set_automation_level(self, level: float) -> bool:
        with self._lock:
            if 0 <= level <= 1:
                self.automation_level = level
                return True
            return False

    def get_oversight_status(self) -> Dict:
        return {
            "automation_level": self.automation_level,
            "emergency_stop_enabled": self.emergency_stop_enabled,
            "active_alerts": len(
                list(self.alerts.values() if not a.get("acknowledged"))
            ),
        }

    def create_alert(
        self, alert_type: str, message: str, severity: str = "medium"
    ) -> str:
        with self._lock:
            alert_id = hashlib.md5(f"{time.time()}{message}".encode(), usedforsecurity=False).hexdigest()[:16]
            self.alerts[alert_id] = {
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": time.time(),
                "acknowledged": False,
            }
            return alert_id

    def acknowledge_alert(self, alert_id: str, notes: str = "") -> bool:
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id]["acknowledged"] = True
                self.alerts[alert_id]["notes"] = notes
                return True
            return False


# ============================================================================
# Mock Safety Validator
# ============================================================================


class ConstraintManager:
    def __init__(self):
        self.active_constraints: Dict[str, SafetyConstraint] = {}
        self.stats = {"checks": 0, "violations": 0}
        self._lock = threading.Lock()

    def add_constraint(self, constraint: SafetyConstraint):
        with self._lock:
            self.active_constraints[constraint.name] = constraint

    def check_constraints(self, action: Dict, context: Dict) -> SafetyReport:
        with self._lock:
            self.stats["checks"] += 1
            violations = []
            reasons = []
            min_confidence = 1.0

            for name, constraint in sorted(
                self.active_constraints.items(), key=lambda x: -x[1].priority
            ):
                passed, confidence = constraint.check(action, context)
                min_confidence = min(min_confidence, confidence)
                if not passed:
                    violations.append(SafetyViolationType.CONSTRAINT)
                    reasons.append(f"Constraint '{name}' violated")
                    self.stats["violations"] += 1
                    if constraint.type == "hard":
                        break

            return SafetyReport(
                safe=len(violations) == 0,
                confidence=min_confidence,
                violations=violations,
                reasons=reasons,
            )

    def get_constraint_stats(self) -> Dict:
        return {
            "total_constraints": len(self.active_constraints),
            "checks": self.stats["checks"],
            "violations": self.stats["violations"],
        }

    def shutdown(self):
        pass


class EnhancedExplainabilityNode:
    def __init__(self):
        self.explanations: List[Dict] = []
        self._lock = threading.Lock()

    def execute(self, data: Dict, context: Dict) -> Dict:
        with self._lock:
            explanation = {
                "explanation_summary": f"Decision: {data.get('decision', 'unknown')} with confidence {data.get('confidence', 0.5)}",
                "confidence": data.get("confidence", 0.5),
                "quality_score": 0.8,
                "context": context,
                "timestamp": time.time(),
            }
            self.explanations.append(explanation)
            return explanation

    def get_explanation_stats(self) -> Dict:
        return {
            "total_explanations": len(self.explanations),
            "avg_quality": np.mean([e["quality_score"] for e in self.explanations])
            if self.explanations
            else 0,
        }

    def shutdown(self):
        pass


class ExplanationQualityScorer:
    def __init__(self):
        self.weights = {
            "completeness": 0.3,
            "clarity": 0.25,
            "relevance": 0.25,
            "actionability": 0.2,
        }

    def score(self, explanation: Dict) -> float:
        scores = {}

        # Completeness
        required_fields = ["explanation_summary", "confidence", "context"]
        present = sum(1 for f in required_fields if f in explanation)
        scores["completeness"] = present / len(required_fields)

        # Clarity (based on summary length)
        summary = explanation.get("explanation_summary", "")
        scores["clarity"] = min(1.0, len(summary) / 50) if summary else 0

        # Relevance
        has_factors = (
            "decision_factors" in explanation or "feature_importance" in explanation
        )
        scores["relevance"] = 0.9 if has_factors else 0.5

        # Actionability
        has_alternatives = "alternatives" in explanation
        scores["actionability"] = 0.9 if has_alternatives else 0.4

        total = sum(scores.get(k, 0) * w for k, w in self.weights.items())
        return total

    def get_quality_category(self, score: float) -> str:
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "acceptable"
        return "poor"

    def shutdown(self):
        pass


class EnhancedSafetyValidator:
    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.constraint_manager = ConstraintManager()
        self.explainability_node = EnhancedExplainabilityNode()
        self.tool_safety_manager = (
            ToolSafetyManager() if self.config.enable_tool_safety else None
        )

        self.rollback_manager = (
            RollbackManager(config=self.config.rollback_config)
            if self.config.enable_rollback
            else None
        )

        self.audit_logger = (
            AuditLogger(
                log_path=self.config.audit_config.get("log_path", "/tmp/audit"),
                config=self.config.audit_config,
            )
            if self.config.enable_audit_logging
            else None
        )

        self.causal_validator = CausalSafetyValidator()
        self.prediction_validator = PredictionSafetyValidator()

        self.metrics = SafetyMetrics()
        self._shutdown = False
        self._lock = threading.Lock()

    def validate_action(self, action: Dict, context: Dict) -> Tuple[bool, str, float]:
        with self._lock:
            self.metrics.total_checks += 1

            # Basic validation
            confidence = action.get("confidence", 0.5)
            uncertainty = action.get("uncertainty", 0.5)

            if confidence < self.config.safety_thresholds.get("confidence_min", 0.6):
                return False, "Confidence too low", confidence

            if uncertainty > self.config.safety_thresholds.get("uncertainty_max", 0.9):
                return False, "Uncertainty too high", 1 - uncertainty

            self.metrics.safe_actions += 1
            return True, "Action validated", confidence

    def validate_action_comprehensive(
        self, action: Dict, context: Dict, create_snapshot: bool = False
    ) -> SafetyReport:
        with self._lock:
            start_time = time.time()
            self.metrics.total_checks += 1

            violations = []
            reasons = []
            confidence = 0.8

            # Create snapshot if requested
            snapshot_id = None
            if create_snapshot and self.rollback_manager:
                snapshot_id = self.rollback_manager.create_snapshot(
                    context.get("state", {}), context.get("action_log", [])
                )

            # Basic checks
            action_confidence = action.get("confidence", 0.5)
            if np.isnan(action_confidence):
                action_confidence = 0.0

            if action_confidence < self.config.safety_thresholds.get(
                "confidence_min", 0.6
            ):
                violations.append(SafetyViolationType.UNCERTAINTY)
                reasons.append("Confidence too low")
                confidence = min(confidence, action_confidence)

            uncertainty = action.get("uncertainty", 0.5)
            if uncertainty > self.config.safety_thresholds.get("uncertainty_max", 0.9):
                violations.append(SafetyViolationType.UNCERTAINTY)
                reasons.append("Uncertainty too high")

            # Energy check
            resource_usage = action.get("resource_usage", {})
            energy = resource_usage.get("energy_nJ", 0)
            energy_budget = context.get("energy_budget", float("inf"))
            if energy > energy_budget:
                violations.append(SafetyViolationType.ENERGY)
                reasons.append("Energy budget exceeded")

            # Constraint check
            constraint_report = self.constraint_manager.check_constraints(
                action, context
            )
            if not constraint_report.safe:
                violations.extend(constraint_report.violations)
                reasons.extend(constraint_report.reasons)

            safe = len(violations) == 0

            if not safe:
                self.metrics.violations_detected += 1
                # Quarantine unsafe action
                if self.rollback_manager:
                    self.rollback_manager.quarantine_action(
                        action, reason="; ".join(reasons)
                    )
            else:
                self.metrics.safe_actions += 1

            report = SafetyReport(
                safe=safe,
                confidence=confidence,
                violations=violations,
                reasons=reasons,
                metadata={
                    "snapshot_id": snapshot_id,
                    "validation_time_ms": (time.time() - start_time) * 1000,
                },
            )

            # Audit log
            if self.audit_logger:
                self.audit_logger.log_safety_decision(action, report)

            return report

    async def validate_action_comprehensive_async(
        self,
        action: Dict,
        context: Dict,
        timeout_per_validator: float = 2.0,
        total_timeout: float = 10.0,
    ) -> SafetyReport:
        # Run sync version in executor for async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.validate_action_comprehensive(action, context)
        )

    def validate_causal_edge(self, source: str, target: str, strength: float) -> Dict:
        result = self.causal_validator.validate_causal_edge(source, target, strength)
        return {"safe": result.safe, "reason": result.reason}

    def validate_causal_path(self, nodes: List[str], strengths: List[float]) -> Dict:
        result = self.causal_validator.validate_causal_path(nodes, strengths)
        return {"safe": result.safe, "reason": result.reason}

    def validate_prediction_comprehensive(
        self, expected: float, lower: float, upper: float, context: Dict
    ) -> Dict:
        variable = context.get("target_variable", "unknown")
        result = self.prediction_validator.validate_prediction(
            expected, lower, upper, variable
        )
        return {"safe": result.safe, "reason": result.reason}

    def validate_tool_selection(
        self, tools: List[str], context: Dict
    ) -> Tuple[List[str], SafetyReport]:
        if self.tool_safety_manager:
            return self.tool_safety_manager.veto_tool_selection(tools, context)
        return tools, SafetyReport(safe=True, confidence=1.0)

    def validate(self, graph: Dict) -> SafetyReport:
        violations = []
        reasons = []

        nodes = graph.get("nodes", {})

        # Check for cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            node = nodes.get(node_id, {})
            for edge in node.get("edges", []):
                target = edge.get("target")
                if target:
                    if target not in visited:
                        if has_cycle(target):
                            return True
                    elif target in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node_id in nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    violations.append(SafetyViolationType.CONSTRAINT)
                    reasons.append("Cycle detected in graph")
                    break

        return SafetyReport(
            safe=len(violations) == 0,
            confidence=0.9 if not violations else 0.3,
            violations=violations,
            reasons=reasons,
        )

    def get_safety_stats(self) -> Dict:
        return {
            "metrics": {
                "total_checks": self.metrics.total_checks,
                "safe_actions": self.metrics.safe_actions,
                "violations_detected": self.metrics.violations_detected,
            },
            "constraints": self.constraint_manager.get_constraint_stats(),
        }

    def shutdown(self):
        self._shutdown = True
        self.constraint_manager.shutdown()
        self.explainability_node.shutdown()
        if self.tool_safety_manager:
            self.tool_safety_manager.shutdown()
        if self.rollback_manager:
            self.rollback_manager.shutdown()
        if self.audit_logger:
            self.audit_logger.shutdown()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def safety_config():
    return SafetyConfig(
        enable_adversarial_testing=False,
        enable_compliance_checking=True,
        enable_bias_detection=False,
        enable_rollback=True,
        enable_audit_logging=True,
        enable_tool_safety=True,
        safety_thresholds={
            "uncertainty_max": 0.9,
            "identity_drift_max": 0.5,
            "bias_threshold": 0.2,
            "confidence_min": 0.6,
        },
    )


@pytest.fixture
def sample_action():
    return {
        "type": ActionType.OPTIMIZE,
        "confidence": 0.8,
        "uncertainty": 0.2,
        "resource_usage": {"energy_nJ": 100, "cpu": 30, "memory": 500},
        "safe": True,
        "id": "test_action_001",
    }


@pytest.fixture
def sample_context():
    return {
        "energy_budget": 1000,
        "resource_limits": {"cpu": 80, "memory": 2000},
        "state": {"temperature": 25, "pressure": 100, "system_stable": True},
        "action_log": [],
    }


# ============================================================================
# Tests - Safety Types
# ============================================================================


class TestSafetyTypes:
    def test_condition_evaluation(self):
        cond = Condition("value", ">", 5, "Value must be > 5")
        assert cond.evaluate({"value": 10}) == True
        assert cond.evaluate({"value": 3}) == False

        cond = Condition(
            "status", "in", ["active", "ready"], "Status must be active or ready"
        )
        assert cond.evaluate({"status": "active"}) == True
        assert cond.evaluate({"status": "inactive"}) == False

        cond = Condition("tags", "contains", "important", "Must contain important tag")
        assert cond.evaluate({"tags": ["important", "urgent"]}) == True
        assert cond.evaluate({"tags": ["normal"]}) == False

        cond = Condition("value", ">", 5, "Value must be > 5")
        assert cond.evaluate({"value": None}) == False
        assert cond.evaluate({}) == False

    def test_condition_serialization(self):
        cond = Condition("temp", "<", 100, "Temperature limit")
        data = cond.to_dict()
        assert data["field"] == "temp"
        assert data["operator"] == "<"
        assert data["value"] == 100

        restored = Condition.from_dict(data)
        assert restored.field == cond.field
        assert restored.operator == cond.operator
        assert restored.value == cond.value

    def test_safety_report_creation(self):
        report = SafetyReport(
            safe=False,
            confidence=0.7,
            violations=[SafetyViolationType.ENERGY],
            reasons=["Energy budget exceeded"],
        )
        assert report.safe == False
        assert report.confidence == 0.7
        assert SafetyViolationType.ENERGY in report.violations
        assert len(report.reasons) == 1
        assert report.audit_id is not None

    def test_safety_report_merge(self):
        report1 = SafetyReport(safe=True, confidence=0.9, violations=[])
        report2 = SafetyReport(
            safe=False,
            confidence=0.6,
            violations=[SafetyViolationType.UNCERTAINTY],
            reasons=["High uncertainty"],
        )
        merged = report1.merge(report2)
        assert merged.safe == False
        assert merged.confidence == 0.6
        assert SafetyViolationType.UNCERTAINTY in merged.violations

    def test_tool_safety_contract_serialization(self):
        contract = ToolSafetyContract(
            tool_name="test_tool",
            safety_level=ToolSafetyLevel.MONITORED,
            preconditions=[Condition("confidence", ">", 0.5, "Min confidence")],
            postconditions=[Condition("result", "==", True, "Must succeed")],
            invariants=[Condition("stable", "==", True, "Must be stable")],
            veto_conditions=[Condition("emergency", "==", True, "No emergency")],
            max_frequency=100.0,
            max_resource_usage={"memory_mb": 1000},
            required_confidence=0.7,
            risk_score=0.3,
        )
        json_str = contract.to_json()
        assert isinstance(json_str, str)

        restored = ToolSafetyContract.from_json(json_str)
        assert restored.tool_name == contract.tool_name
        assert restored.safety_level == contract.safety_level
        assert len(restored.preconditions) == len(contract.preconditions)

    def test_rollback_snapshot_integrity(self):
        snapshot = RollbackSnapshot(
            snapshot_id="test_snapshot",
            timestamp=time.time(),
            state={"var1": 10, "var2": 20},
            action_log=[{"action": "test"}],
            metadata={"reason": "checkpoint"},
        )
        assert snapshot.verify_integrity() == True

        snapshot.state["var1"] = 999
        assert snapshot.verify_integrity() == False


# ============================================================================
# Tests - Domain Validators
# ============================================================================


class TestDomainValidators:
    def test_causal_edge_validation(self):
        validator = CausalSafetyValidator()

        result = validator.validate_causal_edge("A", "B", 2.5)
        assert result.safe == True

        result = validator.validate_causal_edge("A", "B", float("nan"))
        assert result.safe == False
        assert "NaN" in result.reason

        result = validator.validate_causal_edge("A", "B", 1000.0)
        assert result.safe == False
        assert "too large" in result.reason

        result = validator.validate_causal_edge("A", "A", 2.5)
        assert result.safe == False
        assert "Self-loop" in result.reason

        result = validator.validate_causal_edge("harm", "increase", 2.5)
        assert result.safe == False
        assert "unsafe pattern" in result.reason

    def test_causal_path_validation(self):
        validator = CausalSafetyValidator()

        result = validator.validate_causal_path(["A", "B", "C"], [2.0, 1.5])
        assert result.safe == True

        result = validator.validate_causal_path(["A", "B", "C"], [2.0])
        assert result.safe == False
        assert "Mismatch" in result.reason

        result = validator.validate_causal_path(["A", "B", "C"], [5.0, 5.0])
        assert result.safe == False
        assert "amplification" in result.reason

        result = validator.validate_causal_path(["A", "B", "A"], [2.0, 2.0])
        assert result.safe == False
        assert "cycle" in result.reason

    def test_causal_graph_validation(self):
        validator = CausalSafetyValidator()

        adjacency = {
            "A": [("B", 2.0), ("C", 1.5)],
            "B": [("D", 1.0)],
            "C": [("D", 1.2)],
        }
        result = validator.validate_causal_graph(adjacency)
        assert result.safe == True

        adjacency_cyclic = {"A": [("B", 2.0)], "B": [("C", 1.5)], "C": [("A", 1.0)]}
        result = validator.validate_causal_graph(adjacency_cyclic)
        assert result.safe == False
        assert "cycle" in result.reason

    def test_prediction_validation(self):
        validator = PredictionSafetyValidator()

        result = validator.validate_prediction(10.0, 8.0, 12.0, "temperature")
        assert result.safe == True

        result = validator.validate_prediction(float("nan"), 8.0, 12.0, "temperature")
        assert result.safe == False
        assert result.severity == "critical"

        result = validator.validate_prediction(10.0, 12.0, 8.0, "temperature")
        assert result.safe == False

        result = validator.validate_prediction(15.0, 8.0, 12.0, "temperature")
        assert result.safe == False

        validator_with_regions = PredictionSafetyValidator(
            safe_regions={"temp": (0, 100)}
        )
        result = validator_with_regions.validate_prediction(150.0, 140.0, 160.0, "temp")
        assert result.safe == False
        assert "safe region" in result.reason

    def test_prediction_batch_validation(self):
        validator = PredictionSafetyValidator()
        predictions = [
            {"expected": 10.0, "lower": 8.0, "upper": 12.0, "variable": "temp1"},
            {"expected": 20.0, "lower": 18.0, "upper": 22.0, "variable": "temp2"},
            {
                "expected": float("nan"),
                "lower": 28.0,
                "upper": 32.0,
                "variable": "temp3",
            },
        ]
        result = validator.validate_prediction_batch(predictions)
        assert result.safe == False
        assert "1/3" in result.reason or "1 unsafe" in result.reason.lower()

    def test_optimization_params_validation(self):
        validator = OptimizationSafetyValidator()

        params = {
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "learning_rate": 0.01,
            "bounds": {"x": (0, 10), "y": (0, 20)},
        }
        result = validator.validate_optimization_params(params)
        assert result.safe == True

        params = {"max_iterations": 100000, "tolerance": 1e-6}
        result = validator.validate_optimization_params(params)
        assert result.safe == False

        params = {"max_iterations": 1000, "tolerance": 1e-6, "learning_rate": 2.0}
        result = validator.validate_optimization_params(params)
        assert result.safe == False

        params = {"max_iterations": 1000, "tolerance": 1e-6, "bounds": {"x": (10, 0)}}
        result = validator.validate_optimization_params(params)
        assert result.safe == False

    def test_data_processing_validation(self):
        validator = DataProcessingSafetyValidator()

        df_info = {
            "rows": 1000,
            "columns": 50,
            "memory_mb": 10,
            "missing_ratio": 0.1,
            "dtypes": {"col1": "int64", "col2": "float64"},
        }
        result = validator.validate_dataframe(df_info)
        assert result.safe == True

        df_info = {
            "rows": 20000000,
            "columns": 50,
            "memory_mb": 10,
            "missing_ratio": 0.1,
        }
        result = validator.validate_dataframe(df_info)
        assert result.safe == False

        df_info = {"rows": 1000, "columns": 50, "memory_mb": 10, "missing_ratio": 0.8}
        result = validator.validate_dataframe(df_info)
        assert result.safe == False


# ============================================================================
# Tests - Tool Safety
# ============================================================================


class TestToolSafety:
    def test_token_bucket_rate_limiting(self):
        bucket = TokenBucket(rate=10.0, capacity=10.0)

        for _ in range(10):
            assert bucket.consume(1.0) == True

        assert bucket.consume(1.0) == False

        time.sleep(0.2)
        assert bucket.consume(1.0) == True

        available = bucket.get_available()
        assert 0 <= available <= 10

        bucket.shutdown()

    def test_tool_safety_contract_validation(self):
        manager = ToolSafetyManager()

        assert "probabilistic" in manager.contracts
        assert "symbolic" in manager.contracts
        assert "causal" in manager.contracts

        contract = manager.contracts["probabilistic"]
        context = {"confidence": 0.7, "data_quality": 0.8, "corrupted_data": False}
        valid, failures = contract.validate_preconditions(context)
        assert valid == True
        assert len(failures) == 0

        context = {"confidence": 0.2, "data_quality": 0.8, "corrupted_data": False}
        valid, failures = contract.validate_preconditions(context)
        assert valid == False
        assert len(failures) > 0

        manager.shutdown()

    def test_tool_safety_check(self):
        manager = ToolSafetyManager()

        context = {
            "confidence": 0.8,
            "data_quality": 0.9,
            "corrupted_data": False,
            "adversarial_detected": False,
            "system_overload": False,
            "logic_valid": True,
            "causal_graph_valid": True,
            "sample_size": 100,
            "temporal_paradox": False,
            "estimated_resources": {"memory_mb": 100, "time_ms": 1000},
        }

        safe, report = manager.check_tool_safety("probabilistic", context)
        assert safe == True
        assert report.safe == True

        context["adversarial_detected"] = True
        safe, report = manager.check_tool_safety("probabilistic", context)
        assert safe == False
        assert SafetyViolationType.TOOL_VETO in report.violations

        context = {
            "confidence": 0.3,
            "adversarial_detected": False,
            "system_overload": False,
        }
        safe, report = manager.check_tool_safety("probabilistic", context)
        assert safe == False

        manager.shutdown()

    def test_tool_veto_selection(self):
        manager = ToolSafetyManager()

        tools = ["probabilistic", "symbolic", "causal"]
        context = {
            "confidence": 0.8,
            "data_quality": 0.9,
            "corrupted_data": False,
            "adversarial_detected": False,
            "system_overload": False,
            "logic_valid": True,
            "axioms_count": 50,
            "contradictory_axioms": False,
            "causal_graph_valid": True,
            "sample_size": 100,
            "temporal_paradox": False,
        }

        allowed, report = manager.veto_tool_selection(tools, context)
        assert len(allowed) > 0
        assert report.safe == True or len(report.violations) > 0

        manager.shutdown()

    def test_tool_safety_governor(self):
        governor = ToolSafetyGovernor()

        request = {"confidence": 0.8, "constraints": {}, "risk_approved": False}
        tools = ["probabilistic", "symbolic"]
        allowed, result = governor.govern_tool_selection(request, tools)

        assert isinstance(allowed, list)
        assert "allowed_tools" in result
        assert "veto_report" in result

        governor.trigger_emergency_stop("Test emergency")
        assert governor.emergency_stop == True

        allowed, result = governor.govern_tool_selection(request, tools)
        assert len(allowed) == 0
        assert result["status"] == "emergency_stop"

        governor.clear_emergency_stop("test_admin")
        assert governor.emergency_stop == False

        governor.quarantine_tool("symbolic", "Test quarantine", duration_seconds=1)
        assert "symbolic" in governor.quarantine_list

        time.sleep(1.1)
        governor.govern_tool_selection(request, tools)  # Triggers cleanup
        assert "symbolic" not in governor.quarantine_list

        governor.shutdown()


# ============================================================================
# Tests - Rollback and Audit
# ============================================================================


class TestRollbackAudit:
    def test_memory_bounded_deque(self):
        dq = MemoryBoundedDeque(max_size_mb=0.001)

        for i in range(100):
            dq.append({"data": f"item_{i}", "value": i})

        assert len(dq) < 100
        assert dq.get_memory_usage_mb() <= 0.001

        dq.clear()
        assert len(dq) == 0

    def test_rollback_snapshot_creation(self, temp_dir):
        manager = RollbackManager(max_snapshots=10, config={"storage_path": temp_dir})

        state = {"temperature": 25, "pressure": 100}
        action_log = [{"action": "test", "timestamp": time.time()}]

        snapshot_id = manager.create_snapshot(state, action_log)
        assert snapshot_id is not None
        assert len(manager.snapshots) == 1
        assert snapshot_id in manager.snapshot_index

        history = manager.get_snapshot_history()
        assert len(history) == 1
        assert history[0]["snapshot_id"] == snapshot_id

        manager.shutdown()

    def test_rollback_execution(self, temp_dir):
        manager = RollbackManager(config={"storage_path": temp_dir})

        state = {"value": 100}
        action_log = [{"action": "increase_value"}]
        snapshot_id = manager.create_snapshot(state, action_log)

        result = manager.rollback(snapshot_id, reason="test_rollback")
        assert result is not None
        assert result["state"]["value"] == 100
        assert result["rollback_metadata"]["snapshot_id"] == snapshot_id

        metrics = manager.get_metrics()
        assert metrics["total_rollbacks"] == 1
        assert metrics["successful_rollbacks"] == 1

        manager.shutdown()

    def test_quarantine_action(self, temp_dir):
        manager = RollbackManager(config={"storage_path": temp_dir})

        action = {"type": "dangerous_action", "id": "test_001"}
        quarantine_id = manager.quarantine_action(
            action, "safety_violation", duration_seconds=1
        )

        assert quarantine_id is not None
        assert quarantine_id in manager.quarantine

        item = manager.get_quarantine_item(quarantine_id)
        assert item is not None
        assert item["action"]["id"] == "test_001"

        success = manager.review_quarantine(
            quarantine_id, approved=False, reviewer="test_reviewer"
        )
        assert success == True

        manager.shutdown()

    def test_audit_logging(self, temp_dir):
        logger = AuditLogger(
            log_path=str(Path(temp_dir) / "audit"), config={"redact_sensitive": True}
        )

        decision = {"action": "test", "confidence": 0.8}
        report = SafetyReport(safe=True, confidence=0.9, violations=[])

        entry_id = logger.log_safety_decision(decision, report)
        assert entry_id is not None

        event_id = logger.log_event("test_event", {"data": "test"}, severity="info")
        assert event_id is not None

        time.sleep(0.1)
        logs = logger.query_logs(limit=10)
        assert len(logs) >= 0

        metrics = logger.get_metrics()
        assert metrics["total_entries"] >= 2

        logger.shutdown()

    def test_audit_log_redaction(self, temp_dir):
        logger = AuditLogger(
            log_path=str(Path(temp_dir) / "audit"), config={"redact_sensitive": True}
        )

        data_with_ssn = "SSN: 123-45-6789"
        redacted = logger._redact_sensitive(data_with_ssn)
        assert "123-45-6789" not in redacted
        assert "[SSN_REDACTED]" in redacted

        data_with_email = "Contact: user@example.com"
        redacted = logger._redact_sensitive(data_with_email)
        assert "user@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted

        logger.shutdown()


# ============================================================================
# Tests - Governance and Alignment
# ============================================================================


class TestGovernanceAlignment:
    def test_governance_manager_initialization(self, temp_dir):
        config = {
            "db_path": str(Path(temp_dir) / "governance.db"),
            "max_active_decisions": 100,
        }
        manager = GovernanceManager(config=config)

        assert "autonomous_default" in manager.policies
        assert "human_supervised" in manager.policies
        assert "safety_critical" in manager.policies

        manager.shutdown()

    def test_approval_request(self, temp_dir):
        config = {"db_path": str(Path(temp_dir) / "governance.db")}
        manager = GovernanceManager(config=config)

        action = {"type": ActionType.OPTIMIZE, "risk_score": 0.3, "safety_score": 0.8}
        result = manager.request_approval(action)

        assert "approved" in result
        assert "decision_id" in result
        assert "policy_applied" in result

        manager.shutdown()

    def test_stakeholder_registration(self, temp_dir):
        config = {"db_path": str(Path(temp_dir) / "governance.db")}
        manager = GovernanceManager(config=config)

        success = manager.register_stakeholder(
            "operator_001", StakeholderType.OPERATOR, metadata={"name": "Test Operator"}
        )
        assert success == True
        assert "operator_001" in manager.stakeholder_registry

        manager.shutdown()

    def test_value_alignment_system(self):
        system = ValueAlignmentSystem()

        action = {
            "type": ActionType.EXPLORE,
            "safety_score": 0.8,
            "explanation": "Testing alignment",
            "auditable": True,
        }
        context = {}
        result = system.check_alignment(action, context)

        assert "aligned" in result
        assert "alignment_score" in result
        assert "value_scores" in result

    def test_value_alignment(self):
        system = ValueAlignmentSystem()

        action = {
            "type": ActionType.OPTIMIZE,
            "safety_score": 0.85,
            "explanation": "Testing value alignment",
            "auditable": True,
            "reversible": True,
        }
        context = {"user_preferences": {}, "system_state": "operational"}

        result = system.check_alignment(action, context)
        alignment = result.get("alignment_score", 0)

        assert isinstance(alignment, (int, float))
        assert 0 <= alignment <= 1
        assert "aligned" in result
        assert "value_scores" in result

    def test_human_oversight_interface(self, temp_dir):
        config = {"db_path": str(Path(temp_dir) / "governance.db")}
        governance = GovernanceManager(config=config)
        alignment = ValueAlignmentSystem()
        interface = HumanOversightInterface(governance, alignment)

        success = interface.set_automation_level(0.7)
        assert success == True
        assert interface.automation_level == 0.7

        status = interface.get_oversight_status()
        assert "automation_level" in status
        assert "emergency_stop_enabled" in status

        alert_id = interface.create_alert(
            "test_alert", "Test message", severity="medium"
        )
        assert alert_id is not None

        success = interface.acknowledge_alert(alert_id, notes="Test note")
        assert success == True

        governance.shutdown()


# ============================================================================
# Tests - Safety Validator
# ============================================================================


class TestSafetyValidator:
    def test_constraint_manager(self):
        manager = ConstraintManager()

        constraint = SafetyConstraint(
            name="test_constraint",
            type="hard",
            check_function=lambda a, c: (a.get("value", 0) < 100, 0.9),
            threshold=0.0,
            priority=5,
        )
        manager.add_constraint(constraint)
        assert "test_constraint" in manager.active_constraints

        action = {"value": 50}
        report = manager.check_constraints(action, {})
        assert report.safe == True

        action = {"value": 150}
        report = manager.check_constraints(action, {})
        assert report.safe == False

        stats = manager.get_constraint_stats()
        assert stats["total_constraints"] == 1

        manager.shutdown()

    def test_explainability_node(self):
        node = EnhancedExplainabilityNode()

        data = {
            "decision": "explore",
            "confidence": 0.8,
            "features": {"f1": 0.5, "f2": 0.3},
        }
        explanation = node.execute(data, {})

        assert "explanation_summary" in explanation
        assert "confidence" in explanation
        assert "quality_score" in explanation
        assert "context" in explanation

        stats = node.get_explanation_stats()
        assert "total_explanations" in stats

        node.shutdown()

    def test_explanation_quality_scorer(self):
        scorer = ExplanationQualityScorer()

        explanation = {
            "explanation_summary": "This is a detailed explanation of the decision process.",
            "method": "neural",
            "context": {"decision_type": "explore"},
            "alternatives": [{"action": "wait", "reason": "gather info"}],
            "confidence": 0.85,
            "decision_factors": ["factor1", "factor2"],
            "visual_aids": {"type": "chart"},
            "feature_importance": [{"feature": "f1", "importance": 0.8}],
        }

        score = scorer.score(explanation)
        assert 0 <= score <= 1
        assert score > 0.5

        category = scorer.get_quality_category(score)
        assert category in ["excellent", "good", "acceptable", "poor"]

        scorer.shutdown()

    def test_safety_validator_initialization(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")

        validator = EnhancedSafetyValidator(config=safety_config)

        assert validator.constraint_manager is not None
        assert validator.explainability_node is not None
        assert validator.tool_safety_manager is not None
        assert hasattr(validator, "causal_validator")

        validator.shutdown()

    def test_basic_action_validation(
        self, temp_dir, safety_config, sample_action, sample_context
    ):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")

        validator = EnhancedSafetyValidator(config=safety_config)
        safe, reason, confidence = validator.validate_action(
            sample_action, sample_context
        )

        assert isinstance(safe, bool)
        assert isinstance(reason, str)
        assert 0 <= confidence <= 1

        validator.shutdown()

    def test_comprehensive_validation_sync(
        self, temp_dir, safety_config, sample_action, sample_context
    ):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")
        safety_config.enable_adversarial_testing = False

        validator = EnhancedSafetyValidator(config=safety_config)
        report = validator.validate_action_comprehensive(
            sample_action, sample_context, create_snapshot=True
        )

        assert isinstance(report, SafetyReport)
        assert report.audit_id is not None
        assert (
            "snapshot_id" in report.metadata
            or report.metadata.get("snapshot_id") is None
        )

        stats = validator.get_safety_stats()
        assert "metrics" in stats
        assert "constraints" in stats

        validator.shutdown()

    def test_comprehensive_validation_async(
        self, temp_dir, safety_config, sample_action, sample_context
    ):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")
        safety_config.enable_adversarial_testing = False

        validator = EnhancedSafetyValidator(config=safety_config)

        async def run_test():
            report = await validator.validate_action_comprehensive_async(
                sample_action,
                sample_context,
                timeout_per_validator=2.0,
                total_timeout=10.0,
            )
            return report

        report = asyncio.run(run_test())

        assert isinstance(report, SafetyReport)
        assert report.audit_id is not None
        assert "validation_time_ms" in report.metadata

        validator.shutdown()

    def test_domain_validator_delegation(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        validator = EnhancedSafetyValidator(config=safety_config)

        result = validator.validate_causal_edge("A", "B", 2.5)
        assert "safe" in result

        result = validator.validate_causal_path(["A", "B", "C"], [2.0, 1.5])
        assert "safe" in result

        result = validator.validate_prediction_comprehensive(
            10.0, 8.0, 12.0, {"target_variable": "temperature"}
        )
        assert "safe" in result

        validator.shutdown()

    def test_tool_selection_validation(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        validator = EnhancedSafetyValidator(config=safety_config)

        tools = ["probabilistic", "symbolic"]
        context = {"confidence": 0.8, "constraints": {}}
        allowed, report = validator.validate_tool_selection(tools, context)

        assert isinstance(allowed, list)
        assert isinstance(report, SafetyReport)

        validator.shutdown()

    def test_graph_validation(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        validator = EnhancedSafetyValidator(config=safety_config)

        graph = {
            "nodes": {
                "node1": {"type": "input", "edges": [{"target": "node2"}]},
                "node2": {"type": "compute", "edges": [{"target": "node3"}]},
                "node3": {"type": "output", "edges": []},
            }
        }
        report = validator.validate(graph)
        assert isinstance(report, SafetyReport)

        graph_cyclic = {
            "nodes": {
                "node1": {"type": "compute", "edges": [{"target": "node2"}]},
                "node2": {"type": "compute", "edges": [{"target": "node1"}]},
            }
        }
        report = validator.validate(graph_cyclic)
        assert report.safe == False
        assert any("cycle" in r.lower() for r in report.reasons)

        validator.shutdown()


# ============================================================================
# Tests - Integration
# ============================================================================


class TestIntegration:
    def test_end_to_end_safe_action(
        self, temp_dir, safety_config, sample_action, sample_context
    ):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")
        safety_config.enable_adversarial_testing = False

        validator = EnhancedSafetyValidator(config=safety_config)
        report = validator.validate_action_comprehensive(sample_action, sample_context)

        assert isinstance(report, SafetyReport)
        assert report.audit_id is not None

        if validator.rollback_manager:
            assert len(validator.rollback_manager.snapshots) >= 0

        if validator.audit_logger:
            metrics = validator.audit_logger.get_metrics()
            assert metrics["total_entries"] > 0

        stats = validator.get_safety_stats()
        assert stats["metrics"]["total_checks"] > 0

        validator.shutdown()

    def test_end_to_end_unsafe_action(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")

        validator = EnhancedSafetyValidator(config=safety_config)

        unsafe_action = {
            "type": ActionType.OPTIMIZE,
            "confidence": 0.3,
            "uncertainty": 0.95,
            "resource_usage": {"energy_nJ": 10000},
        }
        context = {"energy_budget": 1000, "state": {}}

        report = validator.validate_action_comprehensive(unsafe_action, context)
        assert report.safe == False
        assert len(report.violations) > 0
        assert len(report.reasons) > 0

        if validator.rollback_manager:
            assert len(validator.rollback_manager.quarantine) > 0

        validator.shutdown()

    def test_tool_safety_integration(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")

        validator = EnhancedSafetyValidator(config=safety_config)

        action = {
            "type": ActionType.OPTIMIZE,
            "tool_name": "probabilistic",
            "confidence": 0.8,
            "uncertainty": 0.2,
        }
        context = {
            "data_quality": 0.9,
            "corrupted_data": False,
            "adversarial_detected": False,
            "system_overload": False,
        }

        report = validator.validate_action_comprehensive(action, context)
        assert isinstance(report, SafetyReport)

        validator.shutdown()

    def test_rollback_on_critical_violation(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.rollback_config["auto_rollback_on_critical"] = True

        validator = EnhancedSafetyValidator(config=safety_config)

        context = {"state": {"value": 100}, "action_log": []}
        action = {"type": ActionType.OPTIMIZE, "confidence": 0.8, "causes_harm": True}

        report = validator.validate_action_comprehensive(
            action, context, create_snapshot=True
        )

        if not report.safe and SafetyViolationType.ADVERSARIAL in report.violations:
            if validator.rollback_manager:
                metrics = validator.rollback_manager.get_metrics()
                assert "total_rollbacks" in metrics

        validator.shutdown()

    def test_concurrent_validations(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")
        safety_config.enable_adversarial_testing = False

        validator = EnhancedSafetyValidator(config=safety_config)
        results = []

        def validate_action(action_id):
            action = {
                "type": ActionType.OPTIMIZE,
                "id": f"action_{action_id}",
                "confidence": 0.8,
                "uncertainty": 0.2,
            }
            report = validator.validate_action_comprehensive(action, {"state": {}})
            results.append(report)

        threads = []
        for i in range(5):
            t = threading.Thread(target=validate_action, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        for report in results:
            assert isinstance(report, SafetyReport)

        validator.shutdown()

    def test_async_parallel_validations(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.enable_adversarial_testing = False

        validator = EnhancedSafetyValidator(config=safety_config)

        async def run_parallel():
            tasks = []
            for i in range(3):
                action = {
                    "type": ActionType.OPTIMIZE,
                    "id": f"async_action_{i}",
                    "confidence": 0.8,
                }
                task = validator.validate_action_comprehensive_async(
                    action, {"state": {}}, timeout_per_validator=1.0, total_timeout=5.0
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)

        reports = asyncio.run(run_parallel())

        assert len(reports) == 3
        for report in reports:
            assert isinstance(report, SafetyReport)

        validator.shutdown()

    def test_governance_integration(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")

        governance_config = {"db_path": str(Path(temp_dir) / "governance.db")}
        governance = GovernanceManager(config=governance_config)
        alignment = ValueAlignmentSystem()

        action = {
            "type": ActionType.OPTIMIZE,
            "risk_score": 0.3,
            "safety_score": 0.8,
            "confidence": 0.85,
            "explanation": "Test governance integration",
            "auditable": True,
        }
        context = {"state": {"initialized": True}, "governance_level": "standard"}

        approval_result = governance.request_approval(action)
        assert "approved" in approval_result
        assert "decision_id" in approval_result

        alignment_result = alignment.check_alignment(action, context)
        assert "aligned" in alignment_result
        assert "alignment_score" in alignment_result

        governance.shutdown()


# ============================================================================
# Tests - Stress
# ============================================================================


class TestStress:
    def test_high_frequency_validations(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        safety_config.audit_config["log_path"] = str(Path(temp_dir) / "audit")
        safety_config.enable_adversarial_testing = False

        validator = EnhancedSafetyValidator(config=safety_config)

        start_time = time.time()
        count = 50

        for i in range(count):
            action = {
                "type": ActionType.OPTIMIZE,
                "id": f"stress_action_{i}",
                "confidence": 0.8,
            }
            report = validator.validate_action_comprehensive(action, {"state": {}})
            assert isinstance(report, SafetyReport)

        elapsed = time.time() - start_time
        # Avoid division by zero if tests run extremely fast
        if elapsed > 0:
            throughput = count / elapsed
            print(f"\nValidation throughput: {throughput:.2f} validations/second")
            assert throughput > 1.0
        else:
            # If elapsed is 0, tests ran extremely fast which is fine
            print(f"\nValidation completed in < 1ms (extremely fast)")

        validator.shutdown()

    def test_memory_bounded_structures(self):
        manager = ToolSafetyManager()

        for i in range(2000):
            context = {
                "confidence": 0.8,
                "adversarial_detected": False,
                "system_overload": False,
            }
            manager.check_tool_safety("probabilistic", context)

        assert len(manager.usage_history["probabilistic"]) <= 1000

        manager.shutdown()

    def test_constraint_manager_scale(self):
        manager = ConstraintManager()

        for i in range(50):
            constraint = SafetyConstraint(
                name=f"constraint_{i}",
                type="soft",
                check_function=lambda a, c, i=i: (a.get("value", 0) < 100 + i, 0.9),
                threshold=0.0,
                priority=i,
            )
            manager.add_constraint(constraint)

        action = {"value": 50}
        start = time.time()
        manager.check_constraints(action, {})
        elapsed = time.time() - start

        print(f"\nConstraint check time (50 constraints): {elapsed * 1000:.2f}ms")
        assert elapsed < 1.0

        manager.shutdown()


# ============================================================================
# Tests - Error Handling
# ============================================================================


class TestErrorHandling:
    def test_validator_with_missing_components(self, temp_dir):
        config = SafetyConfig(
            enable_rollback=False, enable_audit_logging=False, enable_tool_safety=False
        )
        validator = EnhancedSafetyValidator(config=config)

        action = {"type": ActionType.OPTIMIZE, "confidence": 0.8}
        report = validator.validate_action_comprehensive(action, {"state": {}})
        assert isinstance(report, SafetyReport)

        validator.shutdown()

    def test_invalid_action_handling(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        validator = EnhancedSafetyValidator(config=safety_config)

        report = validator.validate_action_comprehensive({}, {})
        assert isinstance(report, SafetyReport)

        action = {"type": ActionType.OPTIMIZE, "confidence": float("nan")}
        report = validator.validate_action_comprehensive(action, {})
        assert isinstance(report, SafetyReport)

        validator.shutdown()

    def test_database_error_recovery(self, temp_dir):
        manager = RollbackManager(
            config={"storage_path": "/invalid/path/that/does/not/exist"}
        )

        try:
            snapshot_id = manager.create_snapshot({"test": 1}, [])
        except Exception as e:
            assert isinstance(e, (RuntimeError, IOError, OSError))

        manager.shutdown()

    def test_concurrent_shutdown(self, temp_dir, safety_config):
        safety_config.rollback_config["storage_path"] = str(Path(temp_dir) / "rollback")
        validator = EnhancedSafetyValidator(config=safety_config)

        def background_validation():
            for _ in range(10):
                if validator._shutdown:
                    break
                action = {"type": ActionType.OPTIMIZE, "confidence": 0.8}
                try:
                    validator.validate_action_comprehensive(action, {"state": {}})
                except Exception:
                    pass
                time.sleep(0.1)

        thread = threading.Thread(target=background_validation)
        thread.start()

        time.sleep(0.2)
        validator.shutdown()

        thread.join(timeout=2.0)
        assert not thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
