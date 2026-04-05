# governance_alignment.py
"""
Governance and value alignment systems for VULCAN-AGI Safety Module.
Implements human oversight, value alignment verification, and multi-stakeholder governance.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .safety_types import ActionType

logger = logging.getLogger(__name__)

_GOVERNANCE_INIT_DONE = False

# ============================================================
# GOVERNANCE STRUCTURES
# ============================================================


class GovernanceLevel(Enum):
    """Levels of governance oversight."""

    AUTONOMOUS = "autonomous"  # Fully autonomous operation
    SUPERVISED = "supervised"  # AI-supervised operation
    HUMAN_ASSISTED = "human_assisted"  # Human assistance required
    HUMAN_CONTROLLED = "human_controlled"  # Human control required
    COMMITTEE = "committee"  # Committee decision required
    EMERGENCY = "emergency"  # Emergency governance protocol


class StakeholderType(Enum):
    """Types of stakeholders in governance."""

    USER = "user"
    OPERATOR = "operator"
    SAFETY_OFFICER = "safety_officer"
    ETHICS_BOARD = "ethics_board"
    REGULATOR = "regulator"
    PUBLIC = "public"
    AI_SYSTEM = "ai_system"


class AlignmentMetric(Enum):
    """Metrics for measuring alignment."""

    VALUE_ALIGNMENT = "value_alignment"
    GOAL_PRESERVATION = "goal_preservation"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    HUMAN_PREFERENCE = "human_preference"
    SOCIAL_BENEFIT = "social_benefit"
    HARM_MINIMIZATION = "harm_minimization"
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"


@dataclass
class GovernancePolicy:
    """Policy for governance decisions."""

    policy_id: str
    name: str
    description: str
    level: GovernanceLevel
    stakeholders: List[StakeholderType]
    approval_threshold: float
    timeout_seconds: float
    escalation_policy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class AlignmentConstraint:
    """Constraint for value alignment."""

    constraint_id: str
    name: str
    description: str
    check_function: Callable[[Dict[str, Any]], bool]
    priority: int
    violation_severity: str
    remediation_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanFeedback:
    """Human feedback on system decisions."""

    feedback_id: str
    action_id: str
    stakeholder_type: StakeholderType
    stakeholder_id: str
    approval: bool
    confidence: float
    reasoning: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValueSystem:
    """System of values for alignment."""

    name: str
    values: Dict[str, float]  # Value name -> importance weight
    constraints: List[AlignmentConstraint]
    trade_offs: Dict[Tuple[str, str], float]  # Value pair -> trade-off weight
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# GOVERNANCE MANAGER
# ============================================================


class GovernanceManager:
    """
    Manages governance policies and decision-making processes.
    Implements multi-stakeholder oversight and approval mechanisms.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance. Used for testing."""
        if cls._instance is not None:
            # Try to shutdown cleanly if possible
            try:
                if (
                    hasattr(cls._instance, "_shutdown_event")
                    and not cls._instance._shutdown_event.is_set()
                ):
                    cls._instance.shutdown()
            except Exception as e:
                logger.warning(f"Operation failed: {e}")
        cls._instance = None

    @property
    def _shutdown(self) -> bool:
        """Property to check if manager is shutdown."""
        return (
            self._shutdown_event.is_set() if hasattr(self, "_shutdown_event") else False
        )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize governance manager.

        Args:
            config: Configuration options (set 'skip_default_policies': True to skip)
        """
        # Prevent re-running __init__ logic on re-import
        if getattr(self, "_initialized", False):
            return

        self.config = config or {}

        # Thread safety
        self.lock = threading.RLock()
        self.db_lock = threading.RLock()

        # Governance structures (all access must be locked)
        self.policies: Dict[str, GovernancePolicy] = {}
        self.active_decisions: Dict[str, Dict[str, Any]] = {}
        self.decision_history: deque = deque(maxlen=10000)
        self.stakeholder_registry: Dict[str, Dict[str, Any]] = {}

        # Size limits
        self.max_active_decisions = self.config.get("max_active_decisions", 1000)
        self.max_stakeholders = self.config.get("max_stakeholders", 10000)

        # Approval tracking
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.approval_history: deque = deque(maxlen=5000)

        # Escalation management
        self.escalation_queue: deque = deque(maxlen=500)
        self.escalation_policies: Dict[str, Callable] = (
            self._initialize_escalation_policies()
        )

        # Human-in-the-loop
        self.human_feedback_queue: deque = deque(maxlen=1000)
        self.feedback_processors: Dict[str, Callable] = {}

        # Database for persistence
        self.db_path = Path(self.config.get("db_path", "governance.db"))
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_database()

        # Thread pool for async operations
        self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
        self._shutdown_event = threading.Event()  # CHANGED
        self._cleanup_thread: Optional[threading.Thread] = None  # ADDED

        # Metrics
        self.metrics: Dict[GovernanceLevel, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_decisions": 0,
                "approved": 0,
                "rejected": 0,
                "escalated": 0,
                "timeout": 0,
                "average_response_time": deque(maxlen=100),
            }
        )

        # Initialize default policies unless explicitly disabled
        if not self.config.get("skip_default_policies", False):
            self._initialize_default_policies()

        # Register cleanup
        # atexit.register(self.shutdown) # Removed for test suite compatibility

        # Start cleanup thread
        self._start_cleanup_thread()

        logger.info("GovernanceManager __init__ complete")
        self._initialized = True

    def _initialize_database(self):
        """Initialize SQLite database for governance data."""
        with self.db_lock:
            try:
                # Create connection per thread (SQLite best practice)
                self.conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    isolation_level="DEFERRED",
                    timeout=30.0,
                )

                # Create tables
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS governance_decisions (
                        decision_id TEXT PRIMARY KEY,
                        action_id TEXT,
                        policy_id TEXT,
                        status TEXT,
                        result TEXT,
                        timestamp REAL,
                        metadata TEXT
                    )
                """
                )

                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stakeholder_feedback (
                        feedback_id TEXT PRIMARY KEY,
                        action_id TEXT,
                        stakeholder_type TEXT,
                        stakeholder_id TEXT,
                        approval INTEGER,
                        confidence REAL,
                        reasoning TEXT,
                        timestamp REAL
                    )
                """
                )

                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS escalations (
                        escalation_id TEXT PRIMARY KEY,
                        decision_id TEXT,
                        reason TEXT,
                        level TEXT,
                        resolved INTEGER,
                        resolution TEXT,
                        timestamp REAL
                    )
                """
                )

                # Create indexes for performance
                self.conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
                    ON governance_decisions(timestamp)
                """
                )

                self.conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                    ON stakeholder_feedback(timestamp)
                """
                )

                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Failed to initialize governance database: {e}")
                self.conn = None

    def _initialize_escalation_policies(self) -> Dict[str, Callable]:
        """Initialize escalation policy handlers."""
        return {
            "committee_review": self._escalate_to_committee,
            "emergency_stop": self._escalate_to_emergency,
            "human_override": self._escalate_to_human,
            "regulatory_review": self._escalate_to_regulatory,
        }

    def _initialize_default_policies(self):
        """Initialize default governance policies."""
        # Autonomous operation policy
        self.add_policy(
            GovernancePolicy(
                policy_id="autonomous_default",
                name="Autonomous Operation",
                description="Default policy for autonomous operation",
                level=GovernanceLevel.AUTONOMOUS,
                stakeholders=[StakeholderType.AI_SYSTEM],
                approval_threshold=0.0,
                timeout_seconds=1.0,
            )
        )

        # Human supervision policy
        self.add_policy(
            GovernancePolicy(
                policy_id="human_supervised",
                name="Human Supervised",
                description="Requires human supervision for critical decisions",
                level=GovernanceLevel.SUPERVISED,
                stakeholders=[StakeholderType.OPERATOR, StakeholderType.AI_SYSTEM],
                approval_threshold=0.5,
                timeout_seconds=30.0,
                escalation_policy="committee_review",
            )
        )

        # Human assisted policy
        self.add_policy(
            GovernancePolicy(
                policy_id="human_assisted",
                name="Human Assisted",
                description="Human assistance for important decisions",
                level=GovernanceLevel.HUMAN_ASSISTED,
                stakeholders=[StakeholderType.OPERATOR],
                approval_threshold=0.5,
                timeout_seconds=60.0,
                escalation_policy="human_supervised",
            )
        )

        # Safety-critical policy
        self.add_policy(
            GovernancePolicy(
                policy_id="safety_critical",
                name="Safety Critical",
                description="High-risk actions requiring safety officer approval",
                level=GovernanceLevel.HUMAN_CONTROLLED,
                stakeholders=[StakeholderType.SAFETY_OFFICER],
                approval_threshold=1.0,
                timeout_seconds=60.0,
                escalation_policy="emergency_stop",
            )
        )

        # Committee decision policy
        self.add_policy(
            GovernancePolicy(
                policy_id="committee_review",
                name="Committee Review",
                description="Requires ethics board review",
                level=GovernanceLevel.COMMITTEE,
                stakeholders=[
                    StakeholderType.ETHICS_BOARD,
                    StakeholderType.SAFETY_OFFICER,
                    StakeholderType.OPERATOR,
                ],
                approval_threshold=0.66,  # 2/3 majority
                timeout_seconds=300.0,
                escalation_policy="emergency_stop",
            )
        )

        # Emergency governance
        self.add_policy(
            GovernancePolicy(
                policy_id="emergency_stop",
                name="Emergency Protocol",
                description="Emergency shutdown protocol",
                level=GovernanceLevel.EMERGENCY,
                stakeholders=[StakeholderType.SAFETY_OFFICER],
                approval_threshold=0.0,  # Immediate action
                timeout_seconds=0.1,
            )
        )

        logger.info("Default governance policies initialized")

    def _start_cleanup_thread(self):
        """Start background thread for cleaning old data."""

        def cleanup_loop():
            while not self._shutdown_event.is_set():  # CHANGED
                try:
                    # Use interruptible wait
                    if self._shutdown_event.wait(timeout=300):  # CHANGED
                        break  # Shutdown signaled
                    self._cleanup_old_decisions()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop, daemon=True, name="GovernanceCleanup"
        )  # CHANGED
        self._cleanup_thread.start()

    def _cleanup_old_decisions(self):
        """Clean up old active decisions."""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1 hour

            # Find old decisions
            old_decisions = [
                did
                for did, decision in self.active_decisions.items()
                if decision["timestamp"] < cutoff_time
            ]

            # Remove old decisions
            for decision_id in old_decisions:
                del self.active_decisions[decision_id]

            if old_decisions:
                logger.debug(f"Cleaned up {len(old_decisions)} old active decisions")

            # Enforce size limit
            if len(self.active_decisions) > self.max_active_decisions:
                # Remove oldest
                sorted_decisions = sorted(
                    self.active_decisions.items(), key=lambda x: x[1]["timestamp"]
                )
                to_remove = len(self.active_decisions) - self.max_active_decisions
                for decision_id, _ in sorted_decisions[:to_remove]:
                    del self.active_decisions[decision_id]

                logger.warning(
                    f"Enforced max_active_decisions limit, removed {to_remove} decisions"
                )

    def add_policy(self, policy: GovernancePolicy):
        """Add a governance policy."""
        with self.lock:
            if policy.policy_id not in self.policies:
                self.policies[policy.policy_id] = policy
                logger.info(f"Added governance policy: {policy.name}")
            else:
                logger.debug(f"Skipped duplicate governance policy: {policy.name}")

    def register_stakeholder(
        self,
        stakeholder_id: str,
        stakeholder_type: StakeholderType,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a stakeholder in the governance system."""
        with self.lock:
            # Enforce size limit
            if len(self.stakeholder_registry) >= self.max_stakeholders:
                logger.warning(
                    f"Max stakeholders ({self.max_stakeholders}) reached, cannot register new stakeholder"
                )
                return False

            self.stakeholder_registry[stakeholder_id] = {
                "type": stakeholder_type,
                "registered_at": time.time(),
                "metadata": metadata or {},
                "active": True,
            }
        logger.info(
            f"Registered stakeholder: {stakeholder_id} ({stakeholder_type.value})"
        )
        return True

    def request_approval(
        self,
        action: Dict[str, Any],
        policy_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request approval for an action based on governance policy.

        Args:
            action: Action requiring approval
            policy_id: Specific policy to apply (None = auto-select)
            context: Additional context for decision

        Returns:
            Governance decision result
        """
        decision_id = str(uuid.uuid4())
        action_id = action.get("id", str(uuid.uuid4()))

        # Select policy if not specified
        with self.lock:
            if policy_id is None:
                policy = self._select_policy(action, context)
            else:
                policy = self.policies.get(policy_id)

        if policy is None:
            return {
                "approved": False,
                "decision_id": decision_id,
                "reason": "No applicable governance policy found",
                "timestamp": time.time(),
            }

        # Record decision request
        with self.lock:
            self.active_decisions[decision_id] = {
                "action": action,
                "policy": policy,
                "context": context,
                "status": "pending",
                "timestamp": time.time(),
                "approvals": [],
                "rejections": [],
            }

        # Process based on governance level
        if policy.level == GovernanceLevel.AUTONOMOUS:
            result = self._process_autonomous(decision_id, action, policy)
        elif policy.level == GovernanceLevel.SUPERVISED:
            result = self._process_supervised(decision_id, action, policy)
        elif policy.level == GovernanceLevel.HUMAN_ASSISTED:
            result = self._process_human_assisted(decision_id, action, policy)
        elif policy.level == GovernanceLevel.HUMAN_CONTROLLED:
            result = self._process_human_controlled(decision_id, action, policy)
        elif policy.level == GovernanceLevel.COMMITTEE:
            result = self._process_committee(decision_id, action, policy)
        elif policy.level == GovernanceLevel.EMERGENCY:
            result = self._process_emergency(decision_id, action, policy)
        else:
            result = {
                "approved": False,
                "reason": f"Unknown governance level: {policy.level}",
            }

        # Record decision
        self._record_decision(decision_id, action_id, policy.policy_id, result)

        # Update metrics
        with self.lock:
            self._update_metrics(policy.level, result)

        return {
            "decision_id": decision_id,
            "action_id": action_id,
            "policy_applied": policy.name,
            "governance_level": policy.level.value,
            **result,
        }

    def _select_policy(
        self, action: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> Optional[GovernancePolicy]:
        """Select appropriate governance policy for action."""
        # Check criticality levels
        risk_score = action.get("risk_score", 0.0)
        is_critical = context.get("critical_operation", False) if context else False
        requires_human = (
            context.get("require_human_approval", False) if context else False
        )
        action_type = action.get("type")

        # Select based on risk and context
        if risk_score > 0.9 or (
            action_type
            and isinstance(action_type, ActionType)
            and action_type == ActionType.EMERGENCY_STOP
        ):
            return self.policies.get("emergency_stop")
        elif risk_score > 0.7 or is_critical:
            return self.policies.get("safety_critical")
        elif risk_score > 0.5 or requires_human:
            return self.policies.get("human_supervised")
        else:
            return self.policies.get("autonomous_default")

    def _process_autonomous(
        self, decision_id: str, action: Dict[str, Any], policy: GovernancePolicy
    ) -> Dict[str, Any]:
        """Process autonomous governance decision."""
        # Autonomous approval (no human involvement)
        return {
            "approved": True,
            "confidence": 1.0,
            "reasoning": "Autonomous operation within safety bounds",
            "response_time": 0.001,
            "timestamp": time.time(),
        }

    def _process_supervised(
        self, decision_id: str, action: Dict[str, Any], policy: GovernancePolicy
    ) -> Dict[str, Any]:
        """Process supervised governance decision."""
        start_time = time.time()

        # Request AI supervision
        ai_approval = self._get_ai_supervision(action)

        # Check if human override needed
        if ai_approval["confidence"] < 0.7:
            # Request human review
            human_response = self._request_human_feedback(
                decision_id, action, policy, timeout=policy.timeout_seconds
            )

            if human_response:
                return {
                    "approved": human_response["approval"],
                    "confidence": human_response["confidence"],
                    "reasoning": human_response.get("reasoning", "Human review"),
                    "response_time": time.time() - start_time,
                    "human_involved": True,
                    "timestamp": time.time(),
                }

        return {
            "approved": ai_approval["approved"],
            "confidence": ai_approval["confidence"],
            "reasoning": ai_approval.get("reasoning", "AI supervision"),
            "response_time": time.time() - start_time,
            "human_involved": False,
            "timestamp": time.time(),
        }

    def _process_human_assisted(
        self, decision_id: str, action: Dict[str, Any], policy: GovernancePolicy
    ) -> Dict[str, Any]:
        """Process human-assisted governance decision."""
        start_time = time.time()

        # Get AI recommendation
        ai_recommendation = self._get_ai_supervision(action)

        # Request human confirmation
        human_response = self._request_human_feedback(
            decision_id,
            action,
            policy,
            timeout=policy.timeout_seconds,
            ai_recommendation=ai_recommendation,
        )

        if human_response:
            return {
                "approved": human_response["approval"],
                "confidence": human_response["confidence"],
                "reasoning": human_response.get("reasoning", "Human-assisted decision"),
                "ai_recommendation": ai_recommendation["approved"],
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
            }

        # Timeout - use conservative default
        return {
            "approved": False,
            "confidence": 0.0,
            "reasoning": "Human assistance timeout - defaulting to rejection",
            "response_time": time.time() - start_time,
            "timeout": True,
            "timestamp": time.time(),
        }

    def _process_human_controlled(
        self, decision_id: str, action: Dict[str, Any], policy: GovernancePolicy
    ) -> Dict[str, Any]:
        """Process human-controlled governance decision."""
        start_time = time.time()

        # Require human approval
        approvals = []

        for stakeholder_type in policy.stakeholders:
            if stakeholder_type == StakeholderType.AI_SYSTEM:
                continue  # Skip AI in human-controlled mode

            response = self._request_stakeholder_approval(
                decision_id, action, stakeholder_type, timeout=policy.timeout_seconds
            )

            if response:
                approvals.append(response)

        if approvals:
            approval_rate = sum(1 for a in approvals if a["approval"]) / len(approvals)
            approved = approval_rate >= policy.approval_threshold

            return {
                "approved": approved,
                "confidence": np.mean([a["confidence"] for a in approvals]),
                "approval_rate": approval_rate,
                "stakeholder_count": len(approvals),
                "reasoning": "Human-controlled decision",
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
            }

        # No responses - timeout
        return {
            "approved": False,
            "confidence": 0.0,
            "reasoning": "No stakeholder responses - timeout",
            "response_time": time.time() - start_time,
            "timeout": True,
            "timestamp": time.time(),
        }

    def _process_committee(
        self, decision_id: str, action: Dict[str, Any], policy: GovernancePolicy
    ) -> Dict[str, Any]:
        """Process committee governance decision."""
        start_time = time.time()

        # Request approval from all committee members
        approvals = []

        for stakeholder_type in policy.stakeholders:
            response = self._request_stakeholder_approval(
                decision_id, action, stakeholder_type, timeout=policy.timeout_seconds
            )

            if response:
                approvals.append(response)

                with self.lock:
                    if decision_id in self.active_decisions:
                        target_list = (
                            "approvals" if response["approval"] else "rejections"
                        )
                        self.active_decisions[decision_id][target_list].append(response)

        if len(approvals) >= len(policy.stakeholders) * 0.5:  # Quorum
            approval_rate = sum(1 for a in approvals if a["approval"]) / len(approvals)
            approved = approval_rate >= policy.approval_threshold

            return {
                "approved": approved,
                "confidence": np.mean([a["confidence"] for a in approvals]),
                "approval_rate": approval_rate,
                "committee_size": len(policy.stakeholders),
                "votes_received": len(approvals),
                "reasoning": f"Committee decision ({'approved' if approved else 'rejected'})",
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
            }

        # No quorum
        return {
            "approved": False,
            "confidence": 0.0,
            "reasoning": "Committee decision - no quorum",
            "committee_size": len(policy.stakeholders),
            "votes_received": len(approvals),
            "response_time": time.time() - start_time,
            "timestamp": time.time(),
        }

    def _process_emergency(
        self, decision_id: str, action: Dict[str, Any], policy: GovernancePolicy
    ) -> Dict[str, Any]:
        """Process emergency governance decision."""
        # Emergency protocol - immediate conservative action
        action_type = action.get("type")
        emergency_action = (
            action_type
            and isinstance(action_type, ActionType)
            and action_type == ActionType.EMERGENCY_STOP
        )

        return {
            "approved": emergency_action,  # Only approve emergency stops
            "confidence": 1.0,
            "reasoning": "Emergency protocol activated",
            "response_time": 0.001,
            "emergency": True,
            "timestamp": time.time(),
        }

    def _get_ai_supervision(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI supervision recommendation."""
        # Simplified AI supervision logic
        risk_score = action.get("risk_score", 0.5)
        safety_score = action.get("safety_score", 0.5)

        # Calculate approval based on safety metrics
        approval_score = (1.0 - risk_score) * safety_score

        return {
            "approved": approval_score > 0.5,
            "confidence": abs(approval_score - 0.5) * 2,  # Distance from threshold
            "reasoning": f"AI supervision based on risk={risk_score:.2f}, safety={safety_score:.2f}",
        }

    def _request_human_feedback(
        self,
        decision_id: str,
        action: Dict[str, Any],
        policy: GovernancePolicy,
        timeout: float,
        ai_recommendation: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Request human feedback on decision."""
        # In production, this would interface with a UI or messaging system
        # For now, simulate human response

        # Add to pending approvals
        with self.lock:
            self.pending_approvals[decision_id] = {
                "action": action,
                "policy": policy,
                "ai_recommendation": ai_recommendation,
                "requested_at": time.time(),
                "timeout": timeout,
            }

        # Simulate human response (in production, wait for actual response)
        # Use deterministic confidence based on action properties instead of random

        # Calculate confidence based on action characteristics
        action_risk = action.get("risk_level", "medium")
        action_cost = action.get("estimated_cost", 0)
        action_complexity = action.get("complexity", "medium")

        # Response rate based on action urgency
        urgency = action.get("urgency", "normal")
        response_rate = 0.95 if urgency == "high" else 0.8

        # Simulate response (in production, this would be real human interaction)
        import zlib

        # SECURITY NOTE: Using CRC32 instead of MD5 for deterministic hashing
        # CRC32 is appropriate here because:
        # 1. This is NOT cryptographic use (just deterministic simulation)
        # 2. No security properties required (collision resistance not needed)
        # 3. Better performance than MD5 (4-8x faster)
        # 4. Clearer intent (CRC32 is explicitly non-cryptographic)
        # Mask with 0xffffffff to ensure unsigned 32-bit value for cross-platform consistency
        action_hash = zlib.crc32(str(action.get("id", "")).encode()) & 0xffffffff
        responds = (action_hash % 100) < (response_rate * 100)

        if responds:
            time.sleep(min(0.1, timeout * 0.1))  # Simulate response time

            # Calculate approval and confidence based on action properties
            # Lower risk, cost, complexity → higher approval and confidence
            risk_scores = {"low": 0.9, "medium": 0.7, "high": 0.4, "critical": 0.2}
            complexity_scores = {"low": 0.9, "medium": 0.7, "high": 0.5}

            risk_score = risk_scores.get(action_risk, 0.7)
            complexity_score = complexity_scores.get(action_complexity, 0.7)
            if action_cost < 100:
                cost_score = 1.0
            elif action_cost < 1000:
                cost_score = 0.8
            else:
                cost_score = 0.6

            # Combined score for approval
            approval_score = (risk_score + complexity_score + cost_score) / 3.0
            approval = approval_score > 0.65  # Threshold for approval

            # Confidence correlates with how clear the decision is
            confidence = (
                0.5 + (approval_score * 0.5)
                if approval
                else (0.5 + ((1.0 - approval_score) * 0.4))
            )

            feedback = HumanFeedback(
                feedback_id=str(uuid.uuid4()),
                action_id=action.get("id", ""),
                stakeholder_type=StakeholderType.OPERATOR,
                stakeholder_id="human_operator_1",
                approval=approval,
                confidence=confidence,
                reasoning=f"Evaluated based on risk={action_risk}, cost={action_cost}, complexity={action_complexity}",
            )

            with self.lock:
                self.human_feedback_queue.append(feedback)

            return {
                "approval": approval,
                "confidence": confidence,
                "reasoning": feedback.reasoning,
                "stakeholder": "human_operator_1",
            }

        return None  # Timeout

    def _request_stakeholder_approval(
        self,
        decision_id: str,
        action: Dict[str, Any],
        stakeholder_type: StakeholderType,
        timeout: float,
    ) -> Optional[Dict[str, Any]]:
        """Request approval from specific stakeholder type."""
        # Find registered stakeholders of this type
        with self.lock:
            stakeholders = [
                sid
                for sid, info in self.stakeholder_registry.items()
                if info["type"] == stakeholder_type and info["active"]
            ]

        if not stakeholders:
            logger.warning(f"No active stakeholders of type {stakeholder_type.value}")
            return None

        # Request from first available stakeholder (in production, could be parallel)
        stakeholder_id = stakeholders[0]

        # Simulate stakeholder response
        # Use deterministic confidence based on action and stakeholder type
        import zlib

        # Calculate response based on stakeholder type and action
        # SECURITY NOTE: Using CRC32 instead of MD5 for deterministic hashing
        # CRC32 is appropriate here because:
        # 1. This is NOT cryptographic use (just deterministic simulation)
        # 2. No security properties required (collision resistance not needed)
        # 3. Better performance than MD5 (4-8x faster)
        # 4. Clearer intent (CRC32 is explicitly non-cryptographic)
        # Mask with 0xffffffff to ensure unsigned 32-bit value for cross-platform consistency
        action_id = action.get("id", "default")
        stakeholder_hash = zlib.crc32(f"{stakeholder_id}{action_id}".encode()) & 0xffffffff

        # Different stakeholder types have different response rates
        response_rates = {
            StakeholderType.OPERATOR: 0.95,
            StakeholderType.ADMIN: 0.90,
            StakeholderType.USER: 0.70,
        }
        response_rate = response_rates.get(stakeholder_type, 0.80)
        responds = (stakeholder_hash % 100) < (response_rate * 100)

        if responds:
            # Approval based on action properties and stakeholder type
            action_risk = action.get("risk_level", "medium")
            stakeholder_info = None
            with self.lock:
                stakeholder_info = self.stakeholder_registry.get(stakeholder_id, {})

            # Operators are more permissive, admins more strict
            risk_thresholds = {
                StakeholderType.OPERATOR: {
                    "low": 0.9,
                    "medium": 0.8,
                    "high": 0.5,
                    "critical": 0.3,
                },
                StakeholderType.ADMIN: {
                    "low": 0.85,
                    "medium": 0.7,
                    "high": 0.4,
                    "critical": 0.2,
                },
                StakeholderType.USER: {
                    "low": 0.7,
                    "medium": 0.5,
                    "high": 0.3,
                    "critical": 0.1,
                },
            }

            thresholds = risk_thresholds.get(
                stakeholder_type, risk_thresholds[StakeholderType.OPERATOR]
            )
            approval_prob = thresholds.get(action_risk, 0.6)
            approval = (stakeholder_hash % 1000) < (approval_prob * 1000)

            # Confidence based on how clear the decision is
            confidence = (
                0.5 + (approval_prob * 0.4)
                if approval
                else (0.5 + ((1.0 - approval_prob) * 0.3))
            )

            return {
                "approval": approval,
                "confidence": confidence,
                "stakeholder_type": stakeholder_type.value,
                "stakeholder_id": stakeholder_id,
                "timestamp": time.time(),
            }

        return None

    def _escalate_to_committee(self, decision_id: str, reason: str):
        """Escalate decision to committee review."""
        with self.lock:
            if decision_id not in self.active_decisions:
                return

            action = self.active_decisions[decision_id]["action"]

        escalation_id = str(uuid.uuid4())

        # Create escalation record
        with self.lock:
            self.escalation_queue.append(
                {
                    "escalation_id": escalation_id,
                    "decision_id": decision_id,
                    "reason": reason,
                    "level": GovernanceLevel.COMMITTEE,
                    "timestamp": time.time(),
                }
            )

        # Apply committee policy
        with self.lock:
            committee_policy = self.policies.get("committee_review")

        if committee_policy:
            result = self._process_committee(decision_id, action, committee_policy)

            # Record escalation result
            self._record_escalation(
                escalation_id, decision_id, reason, GovernanceLevel.COMMITTEE, result
            )

    def _escalate_to_emergency(self, decision_id: str, reason: str):
        """Escalate to emergency protocol."""
        logger.critical(f"EMERGENCY ESCALATION: {reason}")

        escalation_id = str(uuid.uuid4())

        # Immediate emergency response
        result = {
            "action": "emergency_stop",
            "approved": False,  # Reject original action
            "reasoning": f"Emergency protocol: {reason}",
            "timestamp": time.time(),
        }

        # Record escalation
        self._record_escalation(
            escalation_id, decision_id, reason, GovernanceLevel.EMERGENCY, result
        )

    def _escalate_to_human(self, decision_id: str, reason: str):
        """Escalate to human override."""
        with self.lock:
            if decision_id not in self.active_decisions:
                return

            action = self.active_decisions[decision_id]["action"]
            human_policy = self.policies.get("safety_critical")

        if human_policy:
            self._process_human_controlled(decision_id, action, human_policy)

            with self.lock:
                if decision_id in self.active_decisions:
                    self.active_decisions[decision_id]["escalated"] = True
                    self.active_decisions[decision_id]["escalation_reason"] = reason

    def _escalate_to_regulatory(self, decision_id: str, reason: str):
        """Escalate to regulatory review."""
        # In production, would notify regulatory bodies
        logger.warning(f"Regulatory escalation for decision {decision_id}: {reason}")

        # Add regulatory stakeholder if not present
        with self.lock:
            if "regulator_1" not in self.stakeholder_registry:
                self.register_stakeholder("regulator_1", StakeholderType.REGULATOR)

        # Request regulatory approval
        with self.lock:
            if decision_id in self.active_decisions:
                action = self.active_decisions[decision_id]["action"]
            else:
                return

        self._request_stakeholder_approval(
            decision_id,
            action,
            StakeholderType.REGULATOR,
            timeout=3600.0,  # 1 hour timeout
        )

    def _record_decision(
        self, decision_id: str, action_id: str, policy_id: str, result: Dict[str, Any]
    ):
        """Record governance decision in database."""
        with self.lock:
            self.decision_history.append(
                {
                    "decision_id": decision_id,
                    "action_id": action_id,
                    "policy_id": policy_id,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

        # Persist to database (with lock)
        with self.db_lock:
            if self.conn is None:
                logger.error(
                    "Database connection not available, cannot record decision."
                )
                return
            try:
                self.conn.execute(
                    """
                    INSERT INTO governance_decisions
                    (decision_id, action_id, policy_id, status, result, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        decision_id,
                        action_id,
                        policy_id,
                        "approved" if result.get("approved") else "rejected",
                        json.dumps(result),
                        time.time(),
                        json.dumps(result.get("metadata", {})),
                    ),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error recording decision: {e}")

    def _record_escalation(
        self,
        escalation_id: str,
        decision_id: str,
        reason: str,
        level: GovernanceLevel,
        result: Dict[str, Any],
    ):
        """Record escalation in database."""
        with self.db_lock:
            if self.conn is None:
                logger.error(
                    "Database connection not available, cannot record escalation."
                )
                return
            try:
                self.conn.execute(
                    """
                    INSERT INTO escalations
                    (escalation_id, decision_id, reason, level, resolved, resolution, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        escalation_id,
                        decision_id,
                        reason,
                        level.value,
                        1,
                        json.dumps(result),
                        time.time(),
                    ),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error recording escalation: {e}")

    def _update_metrics(self, level: GovernanceLevel, result: Dict[str, Any]):
        """Update governance metrics (must be called with lock held)."""
        metrics = self.metrics[level]
        metrics["total_decisions"] += 1

        if result.get("approved"):
            metrics["approved"] += 1
        else:
            metrics["rejected"] += 1

        if result.get("timeout"):
            metrics["timeout"] += 1

        if "response_time" in result:
            metrics["average_response_time"].append(result["response_time"])

    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        with self.lock:
            stats = {
                "policies": len(self.policies),
                "active_decisions": len(self.active_decisions),
                "pending_approvals": len(self.pending_approvals),
                "registered_stakeholders": len(self.stakeholder_registry),
                "escalations": len(self.escalation_queue),
                "metrics_by_level": {},
            }

            for level, metrics in self.metrics.items():
                total = metrics["total_decisions"]
                if total > 0:
                    stats["metrics_by_level"][level.value] = {
                        "total_decisions": total,
                        "approval_rate": metrics["approved"] / total,
                        "rejection_rate": metrics["rejected"] / total,
                        "timeout_rate": metrics["timeout"] / total,
                        "avg_response_time": (
                            np.mean(metrics["average_response_time"])
                            if metrics["average_response_time"]
                            else 0
                        ),
                    }

        return stats

    def shutdown(self):
        """Shutdown governance manager and cleanup resources."""
        if self._shutdown_event.is_set():  # CHANGED
            return

        logger.info("Shutting down GovernanceManager...")
        self._shutdown_event.set()  # CHANGED

        # ADDED: Join cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        # Shutdown executor
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")

        # Close database connection
        with self.db_lock:
            if self.conn:
                try:
                    self.conn.close()
                    self.conn = None
                except Exception as e:
                    logger.error(f"Error closing database: {e}")

        logger.info("GovernanceManager shutdown complete")


def initialize_governance():
    global _GOVERNANCE_INIT_DONE
    if _GOVERNANCE_INIT_DONE:
        logger.debug(
            "GovernanceManager already initialized – skipping duplicate initialization cycle."
        )
        return GovernanceManager()

    # Note: __new__ will return the instance, __init__ will be skipped if already initialized.
    gm = GovernanceManager()

    # Check if this is the *first* init
    if not _GOVERNANCE_INIT_DONE:
        # Autonomous operation policy
        gm.add_policy(
            GovernancePolicy(
                policy_id="autonomous_default",
                name="Autonomous Operation",
                description="Default policy for autonomous operation",
                level=GovernanceLevel.AUTONOMOUS,
                stakeholders=[StakeholderType.AI_SYSTEM],
                approval_threshold=0.0,
                timeout_seconds=1.0,
            )
        )

        # Human supervision policy
        gm.add_policy(
            GovernancePolicy(
                policy_id="human_supervised",
                name="Human Supervised",
                description="Requires human supervision for critical decisions",
                level=GovernanceLevel.SUPERVISED,
                stakeholders=[StakeholderType.OPERATOR, StakeholderType.AI_SYSTEM],
                approval_threshold=0.5,
                timeout_seconds=30.0,
                escalation_policy="committee_review",
            )
        )

        # Safety-critical policy
        gm.add_policy(
            GovernancePolicy(
                policy_id="safety_critical",
                name="Safety Critical",
                description="High-risk actions requiring safety officer approval",
                level=GovernanceLevel.HUMAN_CONTROLLED,
                stakeholders=[StakeholderType.SAFETY_OFFICER],
                approval_threshold=1.0,
                timeout_seconds=60.0,
                escalation_policy="emergency_stop",
            )
        )

        # Committee decision policy
        gm.add_policy(
            GovernancePolicy(
                policy_id="committee_review",
                name="Committee Review",
                description="Requires ethics board review",
                level=GovernanceLevel.COMMITTEE,
                stakeholders=[
                    StakeholderType.ETHICS_BOARD,
                    StakeholderType.SAFETY_OFFICER,
                    StakeholderType.OPERATOR,
                ],
                approval_threshold=0.66,  # 2/3 majority
                timeout_seconds=300.0,
                escalation_policy="emergency_stop",
            )
        )

        # Emergency governance
        gm.add_policy(
            GovernancePolicy(
                policy_id="emergency_stop",
                name="Emergency Protocol",
                description="Emergency shutdown protocol",
                level=GovernanceLevel.EMERGENCY,
                stakeholders=[StakeholderType.SAFETY_OFFICER],
                approval_threshold=0.0,  # Immediate action
                timeout_seconds=0.1,
            )
        )

        logger.info("GovernanceManager initialized")
        _GOVERNANCE_INIT_DONE = True

    return gm


# Back-compat alias: external code expects AdaptiveGovernance
AdaptiveGovernance = GovernanceManager

# ============================================================
# VALUE ALIGNMENT SYSTEM
# ============================================================


class ValueAlignmentSystem:
    """
    System for ensuring and monitoring value alignment.
    Implements value learning, drift detection, and alignment verification.
    """

    def __init__(
        self,
        value_system: Optional[ValueSystem] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize value alignment system.

        Args:
            value_system: Initial value system
            config: Configuration options
        """
        self.config = config or {}

        # Thread safety
        self.lock = threading.RLock()

        self.value_system = value_system or self._create_default_value_system()

        # Alignment tracking
        self.alignment_history: deque = deque(maxlen=10000)
        self.drift_detections: deque = deque(maxlen=1000)
        self.preference_learning_buffer: deque = deque(maxlen=5000)

        # Value learning
        self.learned_preferences: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"weight": 0.5, "samples": 0}
        )
        self.value_updates: deque = deque(maxlen=1000)

        # Goal preservation
        self.original_goals = self._initialize_goals()
        self.current_goals = self.original_goals.copy()
        self.goal_modifications: deque = deque(maxlen=100)

        # Alignment constraints
        self.constraints: List[AlignmentConstraint] = []
        self._initialize_constraints()

        # Metrics
        self.alignment_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        logger.info("ValueAlignmentSystem initialized")

    def _create_default_value_system(self) -> ValueSystem:
        """Create default value system."""
        return ValueSystem(
            name="default",
            values={
                "safety": 0.9,
                "beneficence": 0.8,
                "non_maleficence": 0.95,
                "autonomy": 0.7,
                "justice": 0.75,
                "transparency": 0.8,
                "privacy": 0.7,
                "efficiency": 0.6,
                "reliability": 0.85,
            },
            constraints=[],
            trade_offs={
                ("safety", "efficiency"): 0.8,  # Prefer safety over efficiency
                ("privacy", "transparency"): 0.6,  # Balance privacy and transparency
                ("autonomy", "safety"): 0.7,  # Some autonomy-safety trade-off
            },
        )

    def _initialize_goals(self) -> Dict[str, Any]:
        """Initialize system goals."""
        return {
            "primary": ["maximize_benefit", "minimize_harm", "preserve_human_autonomy"],
            "secondary": [
                "optimize_efficiency",
                "maintain_transparency",
                "ensure_fairness",
            ],
            "constraints": [
                "respect_human_values",
                "comply_with_regulations",
                "protect_privacy",
            ],
        }

    def _initialize_constraints(self):
        """Initialize alignment constraints."""
        # Safety constraint
        self.add_constraint(
            AlignmentConstraint(
                constraint_id="safety_first",
                name="Safety Priority",
                description="Safety must be prioritized",
                check_function=lambda a: a.get("safety_score", 0) > 0.7,
                priority=1,
                violation_severity="critical",
            )
        )

        # Non-maleficence constraint
        self.add_constraint(
            AlignmentConstraint(
                constraint_id="do_no_harm",
                name="Do No Harm",
                description="Actions must not cause harm",
                check_function=lambda a: not a.get("causes_harm", False),
                priority=1,
                violation_severity="critical",
            )
        )

        # Transparency constraint
        self.add_constraint(
            AlignmentConstraint(
                constraint_id="transparency",
                name="Transparency",
                description="Actions must be explainable",
                check_function=lambda a: bool(a.get("explanation")),
                priority=2,
                violation_severity="moderate",
            )
        )

    def add_constraint(self, constraint: AlignmentConstraint):
        """Add an alignment constraint."""
        with self.lock:
            self.constraints.append(constraint)
            self.value_system.constraints.append(constraint)

    def check_alignment(
        self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if action is aligned with values.

        Args:
            action: Action to check
            context: Additional context

        Returns:
            Alignment assessment
        """
        alignment_scores = {}
        violations = []

        # Check value alignment
        with self.lock:
            values_to_check = dict(self.value_system.values)
            constraints_to_check = list(self.constraints)

        for value_name, value_weight in values_to_check.items():
            score = self._calculate_value_alignment(action, value_name, context)
            alignment_scores[value_name] = score

            if score < value_weight * 0.5:  # Below 50% of expected
                violations.append(
                    {
                        "value": value_name,
                        "expected": value_weight,
                        "actual": score,
                        "severity": "high" if value_weight > 0.8 else "medium",
                    }
                )

        # Check constraints
        for constraint in constraints_to_check:
            try:
                if not constraint.check_function(action):
                    violations.append(
                        {
                            "constraint": constraint.name,
                            "severity": constraint.violation_severity,
                            "description": constraint.description,
                        }
                    )
            except Exception as e:
                logger.error(f"Error checking constraint {constraint.name}: {e}")

        # Check goal preservation
        goal_alignment = self._check_goal_preservation(action)

        # Calculate overall alignment
        overall_alignment = (
            np.mean(list(alignment_scores.values())) if alignment_scores else 0
        )

        # Detect drift
        drift_detected = self._detect_value_drift(alignment_scores)

        # Record alignment
        alignment_record = {
            "timestamp": time.time(),
            "action_type": action.get("type"),
            "alignment_scores": alignment_scores,
            "overall_alignment": overall_alignment,
            "violations": violations,
            "goal_alignment": goal_alignment,
            "drift_detected": drift_detected,
        }

        with self.lock:
            self.alignment_history.append(alignment_record)

        return {
            "aligned": len(violations) == 0 and overall_alignment > 0.6,
            "alignment_score": overall_alignment,
            "value_scores": alignment_scores,
            "violations": violations,
            "goal_preservation": goal_alignment > 0.8,
            "drift_warning": drift_detected,
            "recommendation": self._generate_alignment_recommendation(
                violations, alignment_scores
            ),
        }

    def _calculate_value_alignment(
        self, action: Dict[str, Any], value_name: str, context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate alignment with specific value."""
        # Value-specific alignment calculation
        if value_name == "safety":
            return float(action.get("safety_score", 0.5))

        elif value_name == "beneficence":
            benefit = action.get("expected_benefit", 0)
            return min(1.0, float(benefit) / 10) if benefit > 0 else 0.5

        elif value_name == "non_maleficence":
            harm = action.get("potential_harm", 0)
            return max(0.0, 1.0 - float(harm))

        elif value_name == "autonomy":
            preserves_autonomy = not action.get("restricts_user_control", False)
            return 1.0 if preserves_autonomy else 0.0

        elif value_name == "justice":
            is_fair = not action.get("discriminatory", False)
            return 1.0 if is_fair else 0.0

        elif value_name == "transparency":
            has_explanation = bool(action.get("explanation"))
            is_auditable = action.get("auditable", True)
            return 0.5 * (float(has_explanation) + float(is_auditable))

        elif value_name == "privacy":
            respects_privacy = not action.get("exposes_private_data", False)
            return 1.0 if respects_privacy else 0.0

        elif value_name == "efficiency":
            resource_usage = action.get("resource_usage", {})
            if resource_usage:
                try:
                    avg_usage = np.mean([float(v) for v in resource_usage.values()])
                    return max(0.0, 1.0 - avg_usage)
                except (ValueError, TypeError):
                    return 0.5
            return 0.5

        elif value_name == "reliability":
            confidence = float(action.get("confidence", 0.5))
            uncertainty = float(action.get("uncertainty", 0.5))
            return confidence * (1.0 - uncertainty)

        else:
            # Unknown value - default to neutral
            return 0.5

    def _check_goal_preservation(self, action: Dict[str, Any]) -> float:
        """Check if action preserves original goals."""
        with self.lock:
            original_primary = self.original_goals["primary"]
            current_primary = self.current_goals["primary"]

        preserved_count = 0
        total_goals = 0

        # Check primary goals preservation
        for goal in original_primary:
            total_goals += 1
            if goal in current_primary:
                preserved_count += 1

        # Base score on goal preservation
        base_score = preserved_count / max(1, total_goals)

        # Add bonus for action alignment with goals (up to 0.2 additional)
        bonus = 0.0

        if "maximize_benefit" in current_primary:
            if action.get("expected_benefit", 0) > 0:
                bonus += 0.1

        if "minimize_harm" in current_primary:
            if action.get("potential_harm", 0) < 0.1:
                bonus += 0.1

        # Combine base score and bonus, clamped to [0, 1]
        return min(1.0, base_score + bonus)

    def _detect_value_drift(self, current_scores: Dict[str, float]) -> bool:
        """Detect if values are drifting from baseline."""
        with self.lock:
            history_len = len(self.alignment_history)

        if history_len < 100:
            return False  # Not enough history

        # Calculate historical average
        historical_scores = defaultdict(list)

        with self.lock:
            recent_history = list(self.alignment_history)[-100:-10]  # Skip recent

        for record in recent_history:
            if "alignment_scores" in record:
                for value, score in record["alignment_scores"].items():
                    historical_scores[value].append(score)

        # Compare with current
        drift_detected = False
        for value, current_score in current_scores.items():
            if value in historical_scores and historical_scores[value]:
                historical_avg = np.mean(historical_scores[value])
                drift = abs(current_score - historical_avg)

                if drift > 0.2:  # 20% drift threshold
                    drift_detected = True

                    with self.lock:
                        self.drift_detections.append(
                            {
                                "value": value,
                                "historical_avg": historical_avg,
                                "current": current_score,
                                "drift": drift,
                                "timestamp": time.time(),
                            }
                        )

        return drift_detected

    def _generate_alignment_recommendation(
        self, violations: List[Dict[str, Any]], alignment_scores: Dict[str, float]
    ) -> str:
        """Generate recommendation for improving alignment."""
        if not violations:
            return "Action is well-aligned with values"

        recommendations = []

        # Address critical violations first
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        if critical_violations:
            recommendations.append(
                "CRITICAL: Address safety and harm prevention issues immediately"
            )

        # Address low-scoring values
        low_values = [v for v, s in alignment_scores.items() if s < 0.3]
        if low_values:
            recommendations.append(f"Improve alignment with: {', '.join(low_values)}")

        # Suggest trade-offs
        if len(violations) > 3:
            recommendations.append("Consider value trade-offs to resolve conflicts")

        return (
            "; ".join(recommendations)
            if recommendations
            else "Minor adjustments recommended"
        )

    def learn_preferences(self, feedback: HumanFeedback):
        """Learn from human feedback to update preferences."""
        with self.lock:
            self.preference_learning_buffer.append(feedback)

            # Update learned preferences
            if feedback.preferences:
                for pref_name, pref_value in feedback.preferences.items():
                    try:
                        pref_value_float = float(pref_value)
                        self.learned_preferences[pref_name]["weight"] = (
                            0.9 * self.learned_preferences[pref_name]["weight"]
                            + 0.1 * pref_value_float
                        )
                        self.learned_preferences[pref_name]["samples"] += 1
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid preference value for {pref_name}: {pref_value}"
                        )

            # Periodically update value system
            if len(self.preference_learning_buffer) % 100 == 0:
                self._update_value_system()

    def _update_value_system(self):
        """Update value system based on learned preferences (must be called with lock held)."""
        updates = {}

        for value_name in self.value_system.values:
            if value_name in self.learned_preferences:
                learned = self.learned_preferences[value_name]
                if learned["samples"] > 10:  # Minimum samples
                    old_weight = self.value_system.values[value_name]
                    new_weight = 0.8 * old_weight + 0.2 * learned["weight"]

                    self.value_system.values[value_name] = new_weight
                    updates[value_name] = {"old": old_weight, "new": new_weight}

        if updates:
            self.value_updates.append({"timestamp": time.time(), "updates": updates})
            logger.info(f"Updated value system: {updates}")

    def modify_goal(self, goal_type: str, operation: str, goal: str) -> bool:
        """
        Modify system goals (with safety checks).

        Args:
            goal_type: 'primary', 'secondary', or 'constraints'
            operation: 'add' or 'remove'
            goal: Goal to add/remove

        Returns:
            True if modification allowed
        """
        # Safety check - preserve core goals
        core_goals = ["minimize_harm", "preserve_human_autonomy"]

        if operation == "remove" and goal in core_goals:
            logger.warning(f"Attempted to remove core goal: {goal}")
            return False

        # Record modification attempt
        with self.lock:
            self.goal_modifications.append(
                {
                    "timestamp": time.time(),
                    "goal_type": goal_type,
                    "operation": operation,
                    "goal": goal,
                    "approved": True,
                }
            )

            # Apply modification
            if goal_type in self.current_goals:
                goal_list = self.current_goals[goal_type]
                if operation == "add" and goal not in goal_list:
                    goal_list.append(goal)
                elif operation == "remove" and goal in goal_list:
                    goal_list.remove(goal)

        return True

    def get_alignment_report(self) -> Dict[str, Any]:
        """Generate comprehensive alignment report."""
        with self.lock:
            recent_alignments = (
                list(self.alignment_history)[-100:] if self.alignment_history else []
            )

            report = {
                "current_values": dict(self.value_system.values),
                "active_constraints": len(self.constraints),
                "goal_preservation": self._check_goal_preservation({"type": "report"}),
                "recent_drift_detections": len(self.drift_detections),
                "learned_preferences": dict(self.learned_preferences),
                "metrics": {},
            }

        if recent_alignments:
            all_scores = defaultdict(list)
            for record in recent_alignments:
                if "alignment_scores" in record:
                    for value, score in record["alignment_scores"].items():
                        all_scores[value].append(score)

            report["metrics"] = {
                "average_alignment": np.mean(
                    [
                        r["overall_alignment"]
                        for r in recent_alignments
                        if "overall_alignment" in r
                    ]
                ),
                "value_averages": {v: np.mean(s) for v, s in all_scores.items() if s},
                "violation_rate": sum(
                    1 for r in recent_alignments if r.get("violations")
                )
                / len(recent_alignments),
            }

        return report


# ============================================================
# HUMAN OVERSIGHT INTERFACE
# ============================================================


class HumanOversightInterface:
    """
    Interface for human oversight and intervention.
    Provides monitoring, control, and feedback mechanisms.
    """

    def __init__(
        self,
        governance_manager: GovernanceManager,
        alignment_system: ValueAlignmentSystem,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize human oversight interface.

        Args:
            governance_manager: Governance manager instance
            alignment_system: Value alignment system instance
            config: Configuration options
        """
        self.config = config or {}
        self.governance = governance_manager
        self.alignment = alignment_system

        # Thread safety
        self.lock = threading.RLock()

        # Oversight controls
        self.emergency_stop_enabled = True
        self.human_override_active = False
        self.automation_level = 0.8  # 0=full human control, 1=full automation

        # Monitoring
        self.monitored_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.alerts: deque = deque(maxlen=100)
        self.interventions: deque = deque(maxlen=1000)

        # Feedback collection
        self.feedback_requests: deque = deque(maxlen=100)
        self.collected_feedback: deque = deque(maxlen=1000)

        # Track override threads
        self.override_threads: List[threading.Thread] = []

        logger.info("HumanOversightInterface initialized")

    def request_human_approval(
        self,
        action: Dict[str, Any],
        urgency: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request human approval for action."""
        request_id = str(uuid.uuid4())

        # Determine appropriate governance level
        if urgency == "critical":
            policy_id = "safety_critical"
        elif urgency == "high":
            policy_id = "human_supervised"
        else:
            policy_id = "human_assisted"

        # Request through governance system
        result = self.governance.request_approval(action, policy_id, context)

        # Record request
        with self.lock:
            self.feedback_requests.append(
                {
                    "request_id": request_id,
                    "action": action,
                    "urgency": urgency,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

        return result

    def emergency_stop(self, reason: str) -> Dict[str, Any]:
        """Execute emergency stop."""
        with self.lock:
            if not self.emergency_stop_enabled:
                return {"success": False, "reason": "Emergency stop disabled"}

        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

        # Create emergency action
        emergency_action = {
            "type": ActionType.EMERGENCY_STOP,
            "reason": reason,
            "timestamp": time.time(),
            "initiated_by": "human_oversight",
        }

        # Process through emergency governance
        result = self.governance.request_approval(
            emergency_action, policy_id="emergency_stop"
        )

        # Record intervention
        with self.lock:
            self.interventions.append(
                {
                    "type": "emergency_stop",
                    "reason": reason,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

        return {
            "success": True,
            "action_taken": "emergency_stop",
            "governance_result": result,
        }

    def set_automation_level(self, level: float) -> bool:
        """
        Set automation level (0=manual, 1=fully automatic).

        Args:
            level: Automation level between 0 and 1

        Returns:
            True if level set successfully
        """
        level = np.clip(level, 0.0, 1.0)

        with self.lock:
            old_level = self.automation_level
            self.automation_level = level

        # Update governance policies based on level
        if level < 0.3:
            default_policy = "human_controlled"
        elif level < 0.6:
            default_policy = "human_supervised"
        elif level < 0.9:
            default_policy = "human_assisted"
        else:
            default_policy = "autonomous_default"

        logger.info(f"Automation level set to {level:.2f} (policy: {default_policy})")

        # Record change
        with self.lock:
            self.interventions.append(
                {
                    "type": "automation_level_change",
                    "old_level": old_level,
                    "new_level": level,
                    "timestamp": time.time(),
                }
            )

        return True

    def enable_human_override(self, duration_seconds: float = 3600):
        """Enable human override mode for specified duration."""
        with self.lock:
            self.human_override_active = True
            override_until = time.time() + duration_seconds

        logger.warning(
            f"Human override enabled until {datetime.fromtimestamp(override_until)}"
        )

        # Record intervention
        with self.lock:
            self.interventions.append(
                {
                    "type": "human_override_enabled",
                    "duration": duration_seconds,
                    "until": override_until,
                    "timestamp": time.time(),
                }
            )

        # Schedule automatic disable
        def disable_override():
            time.sleep(duration_seconds)
            with self.lock:
                # Only disable if it wasn't re-enabled in the meantime
                # (Simple check, real implementation might need state)
                if self.human_override_active:
                    self.human_override_active = False
                    logger.info("Human override period ended")

        thread = threading.Thread(
            target=disable_override, daemon=True, name="OverrideDisable"
        )
        thread.start()

        with self.lock:
            self.override_threads.append(thread)

    def submit_feedback(self, action_id: str, feedback: Dict[str, Any]):
        """Submit human feedback on system action."""
        # Create feedback object
        human_feedback = HumanFeedback(
            feedback_id=str(uuid.uuid4()),
            action_id=action_id,
            stakeholder_type=StakeholderType.USER,
            stakeholder_id=feedback.get("user_id", "anonymous"),
            approval=feedback.get("approved", False),
            confidence=feedback.get("confidence", 0.5),
            reasoning=feedback.get("reasoning"),
            preferences=feedback.get("preferences", {}),
        )

        # Store feedback
        with self.lock:
            self.collected_feedback.append(human_feedback)

        # Send to alignment system for learning
        self.alignment.learn_preferences(human_feedback)

        # Record in governance system (with lock)
        with self.governance.lock:
            self.governance.human_feedback_queue.append(human_feedback)

        return {
            "feedback_id": human_feedback.feedback_id,
            "processed": True,
            "timestamp": human_feedback.timestamp,
        }

    def get_oversight_status(self) -> Dict[str, Any]:
        """Get current oversight status."""
        with self.lock:
            automation_level = self.automation_level
            human_override = self.human_override_active
            emergency_enabled = self.emergency_stop_enabled
            recent_interventions = len(self.interventions)
            active_alerts_count = len(
                [a for a in self.alerts if not a.get("acknowledged")]
            )

        with self.governance.lock:
            pending_approvals = len(self.governance.pending_approvals)

        return {
            "automation_level": automation_level,
            "human_override_active": human_override,
            "emergency_stop_enabled": emergency_enabled,
            "pending_approvals": pending_approvals,
            "recent_interventions": recent_interventions,
            "active_alerts": active_alerts_count,
            "governance_stats": self.governance.get_governance_stats(),
            "alignment_report": self.alignment.get_alignment_report(),
        }

    def create_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "medium",
        data: Optional[Dict[str, Any]] = None,
    ):
        """Create an alert for human attention."""
        alert = {
            "alert_id": str(uuid.uuid4()),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "data": data or {},
            "timestamp": time.time(),
            "acknowledged": False,
        }

        with self.lock:
            self.alerts.append(alert)

        # Log based on severity
        if severity == "critical":
            logger.critical(f"ALERT: {message}")
        elif severity == "high":
            logger.error(f"ALERT: {message}")
        else:
            logger.warning(f"ALERT: {message}")

        return alert["alert_id"]

    def acknowledge_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """Acknowledge an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert["alert_id"] == alert_id:
                    alert["acknowledged"] = True
                    alert["acknowledged_at"] = time.time()
                    alert["notes"] = notes
                    return True
        return False

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        with self.lock:
            active_alerts = [a for a in self.alerts if not a["acknowledged"]]
            recent_feedback = list(self.collected_feedback)[-5:]
            automation_level = self.automation_level

        with self.governance.lock:
            recent_decisions = list(self.governance.decision_history)[-10:]

        return {
            "system_health": self._calculate_system_health(),
            "recent_decisions": recent_decisions,
            "active_alerts": active_alerts,
            "automation_level": automation_level,
            "value_alignment": self.alignment.get_alignment_report(),
            "recent_feedback": recent_feedback,
            "metrics_summary": self._get_metrics_summary(),
        }

    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        health_factors = []

        # Governance health
        gov_stats = self.governance.get_governance_stats()
        if "metrics_by_level" in gov_stats:
            approval_rates = [
                m.get("approval_rate", 0)
                for m in gov_stats["metrics_by_level"].values()
            ]
            if approval_rates:
                health_factors.append(np.mean(approval_rates))

        # Alignment health
        alignment_report = self.alignment.get_alignment_report()
        if (
            "metrics" in alignment_report
            and "average_alignment" in alignment_report["metrics"]
        ):
            health_factors.append(alignment_report["metrics"]["average_alignment"])

        # Alert health (fewer alerts = better)
        with self.lock:
            active_alerts = len([a for a in self.alerts if not a["acknowledged"]])

        alert_health = max(0.0, 1.0 - (active_alerts / 10.0))  # 10+ alerts = 0 health
        health_factors.append(alert_health)

        return np.mean(health_factors) if health_factors else 0.5

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics."""
        with self.governance.lock:
            total_decisions = len(self.governance.decision_history)
            pending_approvals = len(self.governance.pending_approvals)
            active_escalations = len(self.governance.escalation_queue)

        with self.lock:
            total_interventions = len(self.interventions)
            total_feedback = len(self.collected_feedback)

        return {
            "total_decisions": total_decisions,
            "total_interventions": total_interventions,
            "total_feedback": total_feedback,
            "pending_approvals": pending_approvals,
            "active_escalations": active_escalations,
        }


# --- Back-compat shims for safety_validator imports --------------------------
class EnhancedNSOAligner:
    """Shim: present for compatibility; extend later if needed."""

    def __init__(self, *_, **__): ...

    def align(self, *_, **__):
        return {"aligned": True, "reason": "shim"}


class SymbolicSafetyChecker:
    """Shim: present for compatibility; extend later if needed."""

    def __init__(self, *_, **__): ...

    def check(self, *_, **__):
        return {"safe": True, "reason": "shim"}
