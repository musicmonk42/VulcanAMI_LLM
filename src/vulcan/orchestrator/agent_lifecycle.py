# ============================================================
# VULCAN-AGI Orchestrator - Agent Lifecycle Module
# Agent states, capabilities, metadata, and job provenance tracking
# FULLY FIXED VERSION - Enhanced with state validation and safety checks
# ============================================================

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# AGENT STATE ENUMERATION
# ============================================================


class AgentState(Enum):
    """Agent lifecycle states with strict state machine semantics"""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    RECOVERING = "recovering"
    RETIRING = "retiring"
    TERMINATED = "terminated"
    ERROR = "error"
    SUSPENDED = "suspended"
    CLEANUP = "cleanup"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"AgentState.{self.name}"

    def is_active(self) -> bool:
        """Check if agent is in an active state"""
        return self in {
            AgentState.INITIALIZING,
            AgentState.IDLE,
            AgentState.WORKING,
            AgentState.RECOVERING,
        }

    def is_terminal(self) -> bool:
        """Check if agent is in a terminal state"""
        return self in {AgentState.TERMINATED}

    def is_error_state(self) -> bool:
        """Check if agent is in an error state"""
        return self in {AgentState.ERROR, AgentState.RECOVERING}

    def can_accept_work(self) -> bool:
        """Check if agent can accept new work"""
        return self == AgentState.IDLE


# ============================================================
# AGENT CAPABILITY ENUMERATION
# ============================================================


class AgentCapability(Enum):
    """Agent capability types with hierarchical relationships
    
    AGENT POOL CONFIGURATION FIX: Added specialized reasoning engine capabilities
    to enable proper routing of queries to the correct reasoning engines.
    
    Previously, only PERCEPTION and GENERAL were effectively used, causing ~45%
    of queries to fail due to missing agent capabilities. This update adds:
    - PROBABILISTIC: Bayesian inference, probability calculations
    - SYMBOLIC: SAT solving, logical inference, proof verification
    - PHILOSOPHICAL: Ethical dilemmas, MEC analysis, deontic logic
    - MATHEMATICAL: Symbolic math, calculus, induction proofs
    - CAUSAL: Causal graphs, intervention analysis
    - ANALOGICAL: Structure mapping, analogical inference
    - CRYPTOGRAPHIC: Hash computation, encryption operations
    - WORLD_MODEL: Self-introspection, counterfactual reasoning
    
    Note: These capabilities map to reasoning engines stored in _AVAILABLE_ENGINES
    in portfolio_executor.py.
    """

    # Basic capabilities
    PERCEPTION = "perception"
    REASONING = "reasoning"
    LEARNING = "learning"
    PLANNING = "planning"
    EXECUTION = "execution"
    MEMORY = "memory"
    SAFETY = "safety"
    GENERAL = "general"
    
    # AGENT POOL FIX: Specialized reasoning engine capabilities
    # These map directly to reasoning engines registered in portfolio_executor.py
    PROBABILISTIC = "probabilistic"      # Maps to ProbabilisticReasoner
    SYMBOLIC = "symbolic"                # Maps to SymbolicReasoner
    PHILOSOPHICAL = "philosophical"      # Maps to WorldModel (mode='philosophical')
    MATHEMATICAL = "mathematical"        # Maps to MathematicalComputationTool
    CAUSAL = "causal"                    # Maps to CausalReasoner
    ANALOGICAL = "analogical"            # Maps to AnalogicalReasoningEngine
    CRYPTOGRAPHIC = "cryptographic"      # Maps to CryptographicEngine
    WORLD_MODEL = "world_model"          # Maps to WorldModel
    MULTIMODAL = "multimodal"            # Maps to MultimodalReasoner

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"AgentCapability.{self.name}"

    def is_specialized(self) -> bool:
        """Check if this is a specialized capability"""
        return self != AgentCapability.GENERAL

    def is_reasoning_capability(self) -> bool:
        """Check if this is a specialized reasoning capability.
        
        AGENT POOL FIX: Identifies capabilities that map to specific reasoning engines.
        """
        return self in {
            AgentCapability.PROBABILISTIC,
            AgentCapability.SYMBOLIC,
            AgentCapability.PHILOSOPHICAL,
            AgentCapability.MATHEMATICAL,
            AgentCapability.CAUSAL,
            AgentCapability.ANALOGICAL,
            AgentCapability.CRYPTOGRAPHIC,
            AgentCapability.WORLD_MODEL,
            AgentCapability.MULTIMODAL,
            AgentCapability.LANGUAGE,
        }

    def can_handle_capability(self, required: "AgentCapability") -> bool:
        """Check if this agent can handle the required capability"""
        # GENERAL agents can handle any capability
        if self == AgentCapability.GENERAL:
            return True
        # REASONING agents can handle any reasoning capability
        if self == AgentCapability.REASONING and required.is_reasoning_capability():
            return True
        # Otherwise, must match exactly
        return self == required


# ============================================================
# STATE TRANSITION RULES
# ============================================================


class StateTransitionRules:
    """Defines valid state transitions for the agent lifecycle state machine"""

    # Valid transitions: current_state -> set of allowed next states
    VALID_TRANSITIONS: Dict[AgentState, FrozenSet[AgentState]] = {
        AgentState.UNINITIALIZED: frozenset(
            {
                AgentState.INITIALIZING,
                AgentState.TERMINATED,  # Can terminate before initialization
            }
        ),
        AgentState.INITIALIZING: frozenset(
            {AgentState.IDLE, AgentState.ERROR, AgentState.TERMINATED}
        ),
        AgentState.IDLE: frozenset(
            {
                AgentState.WORKING,
                AgentState.RETIRING,
                AgentState.SUSPENDED,
                AgentState.TERMINATED,
            }
        ),
        AgentState.WORKING: frozenset(
            {
                AgentState.IDLE,
                AgentState.ERROR,
                AgentState.RETIRING,
                AgentState.TERMINATED,
            }
        ),
        AgentState.RECOVERING: frozenset(
            {AgentState.IDLE, AgentState.ERROR, AgentState.TERMINATED}
        ),
        AgentState.RETIRING: frozenset({AgentState.TERMINATED, AgentState.CLEANUP}),
        AgentState.TERMINATED: frozenset(),  # Terminal state - no valid transitions
        AgentState.ERROR: frozenset({AgentState.RECOVERING, AgentState.TERMINATED}),
        AgentState.SUSPENDED: frozenset(
            {AgentState.IDLE, AgentState.RECOVERING, AgentState.TERMINATED}
        ),
        AgentState.CLEANUP: frozenset({AgentState.TERMINATED}),
    }

    @classmethod
    def is_valid_transition(cls, from_state: AgentState, to_state: AgentState) -> bool:
        """Check if a state transition is valid"""
        if from_state not in cls.VALID_TRANSITIONS:
            logger.error(f"Unknown from_state: {from_state}")
            return False

        allowed_states = cls.VALID_TRANSITIONS[from_state]
        return to_state in allowed_states

    @classmethod
    def get_allowed_transitions(cls, from_state: AgentState) -> FrozenSet[AgentState]:
        """Get all allowed transitions from a given state"""
        return cls.VALID_TRANSITIONS.get(from_state, frozenset())

    @classmethod
    def validate_transition(
        cls, from_state: AgentState, to_state: AgentState, agent_id: str = "unknown"
    ) -> bool:
        """Validate and log state transition"""
        if cls.is_valid_transition(from_state, to_state):
            logger.debug(
                f"Agent {agent_id}: Valid transition {from_state} -> {to_state}"
            )
            return True
        else:
            logger.warning(
                f"Agent {agent_id}: Invalid transition {from_state} -> {to_state}. "
                f"Allowed transitions: {cls.get_allowed_transitions(from_state)}"
            )
            return False


# ============================================================
# AGENT METADATA
# ============================================================


@dataclass
class AgentMetadata:
    """Metadata for tracking agents with enhanced validation and metrics"""

    agent_id: str
    state: AgentState
    capability: AgentCapability
    created_at: float
    last_active: float
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_runtime_s: float = 0.0
    hardware_spec: Dict[str, Any] = field(default_factory=dict)
    location: str = "local"
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Enhanced tracking fields
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    total_state_transitions: int = 0
    last_error_time: Optional[float] = None
    consecutive_errors: int = 0
    last_successful_task_time: Optional[float] = None
    average_task_duration_s: float = 0.0

    def __post_init__(self):
        """Validate initial state"""
        if not isinstance(self.state, AgentState):
            raise ValueError(f"state must be AgentState, got {type(self.state)}")
        if not isinstance(self.capability, AgentCapability):
            raise ValueError(
                f"capability must be AgentCapability, got {type(self.capability)}"
            )

        # Initialize state history
        self._record_state_change(self.state, "initialization")

    def transition_state(self, new_state: AgentState, reason: str = "") -> bool:
        """
        Transition to a new state with validation

        Args:
            new_state: Target state
            reason: Reason for transition

        Returns:
            True if transition successful, False otherwise
        """
        # Special case: Allow idempotent TERMINATED -> TERMINATED for graceful handling
        # This bypasses validation since TERMINATED is a true terminal state
        if self.state == AgentState.TERMINATED and new_state == AgentState.TERMINATED:
            logger.debug(
                f"Agent {self.agent_id} already terminated, allowing idempotent transition"
            )
            return True

        if StateTransitionRules.validate_transition(
            self.state, new_state, self.agent_id
        ):
            old_state = self.state
            self.state = new_state
            self.last_active = time.time()
            self.total_state_transitions += 1

            self._record_state_change(new_state, reason, old_state)

            # Update error tracking
            if new_state == AgentState.ERROR:
                self.last_error_time = time.time()
                self.consecutive_errors += 1
            elif new_state == AgentState.IDLE and old_state == AgentState.WORKING:
                # Successfully completed work
                self.consecutive_errors = 0
                self.last_successful_task_time = time.time()

            return True
        else:
            logger.error(
                f"Agent {self.agent_id}: Rejected transition "
                f"{self.state} -> {new_state} (reason: {reason})"
            )
            return False

    def _record_state_change(
        self, new_state: AgentState, reason: str, old_state: Optional[AgentState] = None
    ):
        """Record state change in history"""
        self.state_history.append(
            {
                "timestamp": time.time(),
                "old_state": old_state.value if old_state else None,
                "new_state": new_state.value,
                "reason": reason,
            }
        )

        # Keep only last 100 state changes to prevent unbounded growth
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

    def record_task_completion(self, success: bool, duration_s: float):
        """Record task completion with metrics update"""
        if success:
            self.tasks_completed += 1
            self.consecutive_errors = 0
            self.last_successful_task_time = time.time()

            # Update average task duration (exponential moving average)
            alpha = 0.3  # Smoothing factor
            if self.average_task_duration_s == 0.0:
                self.average_task_duration_s = duration_s
            else:
                self.average_task_duration_s = (
                    alpha * duration_s + (1 - alpha) * self.average_task_duration_s
                )
        else:
            self.tasks_failed += 1
            self.consecutive_errors += 1
            self.last_error_time = time.time()

        self.total_runtime_s += duration_s
        self.last_active = time.time()

        # Update performance metrics
        self._update_performance_metrics()

    def _update_performance_metrics(self):
        """Update calculated performance metrics"""
        total_tasks = self.tasks_completed + self.tasks_failed

        if total_tasks > 0:
            self.performance_metrics["success_rate"] = (
                self.tasks_completed / total_tasks
            )
            self.performance_metrics["failure_rate"] = self.tasks_failed / total_tasks
        else:
            self.performance_metrics["success_rate"] = 0.0
            self.performance_metrics["failure_rate"] = 0.0

        self.performance_metrics["total_tasks"] = total_tasks
        self.performance_metrics["consecutive_errors"] = self.consecutive_errors
        self.performance_metrics["average_task_duration_s"] = (
            self.average_task_duration_s
        )

        # Calculate uptime
        uptime_s = time.time() - self.created_at
        self.performance_metrics["uptime_s"] = uptime_s

        # Calculate utilization
        if uptime_s > 0:
            self.performance_metrics["utilization"] = self.total_runtime_s / uptime_s
        else:
            self.performance_metrics["utilization"] = 0.0

    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record error with context"""
        error_record = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        self.error_history.append(error_record)

        # Keep only last 50 errors to prevent unbounded growth
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]

        self.last_error_time = time.time()
        self.consecutive_errors += 1

    def should_recover(self) -> bool:
        """Determine if agent should attempt recovery"""
        # Don't recover if too many consecutive errors
        if self.consecutive_errors >= 5:
            return False

        # Don't recover if in terminal state
        if self.state.is_terminal():
            return False

        # Recover if in error state and not too many errors
        if self.state == AgentState.ERROR:
            return True

        return False

    def should_retire(self) -> bool:
        """Determine if agent should be retired"""
        # Retire if too many consecutive errors
        if self.consecutive_errors >= 5:
            return True

        # Retire if failure rate is too high (> 50%) and enough samples
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks >= 10:
            failure_rate = self.tasks_failed / total_tasks
            if failure_rate > 0.5:
                return True

        # Retire if idle for too long (> 5 minutes)
        idle_time = time.time() - self.last_active
        if self.state == AgentState.IDLE and idle_time > 300:
            return True

        return False

    def get_health_score(self) -> float:
        """
        Calculate agent health score (0.0 to 1.0)

        Returns:
            Health score where 1.0 is perfect health
        """
        total_tasks = self.tasks_completed + self.tasks_failed

        if total_tasks == 0:
            # New agent, assume healthy
            return 0.8

        # Success rate component (0.0 to 0.5)
        success_rate = self.tasks_completed / total_tasks
        success_component = success_rate * 0.5

        # Error recency component (0.0 to 0.3)
        if self.last_error_time:
            time_since_error = time.time() - self.last_error_time
            # Errors decay over 1 hour
            error_decay = min(1.0, time_since_error / 3600.0)
            error_component = error_decay * 0.3
        else:
            error_component = 0.3

        # Consecutive errors component (0.0 to 0.2)
        if self.consecutive_errors == 0:
            consecutive_component = 0.2
        else:
            # Penalize heavily for consecutive errors
            consecutive_component = max(0.0, 0.2 - (self.consecutive_errors * 0.05))

        health_score = success_component + error_component + consecutive_component

        return max(0.0, min(1.0, health_score))

    def is_healthy(self, threshold: float = 0.5) -> bool:
        """Check if agent is healthy based on health score"""
        return self.get_health_score() >= threshold

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of agent metadata"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "capability": self.capability.value,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_runtime_s": self.total_runtime_s,
            "location": self.location,
            "consecutive_errors": self.consecutive_errors,
            "health_score": self.get_health_score(),
            "performance_metrics": self.performance_metrics,
            "can_accept_work": self.state.can_accept_work(),
        }


# ============================================================
# JOB PROVENANCE
# ============================================================


@dataclass
class JobProvenance:
    """Complete provenance for a job with enhanced tracking"""

    job_id: str
    agent_id: str
    graph_id: str
    parameters: Dict[str, Any]
    hardware_used: Dict[str, Any]
    start_time: float
    end_time: Optional[float]
    outcome: Optional[str]  # 'success', 'failed', 'timeout', 'cancelled'
    result: Optional[Any]
    error: Optional[str]
    resource_consumption: Dict[str, float]
    checkpoint_paths: List[str] = field(default_factory=list)
    parent_job_id: Optional[str] = None
    child_job_ids: List[str] = field(default_factory=list)

    # Enhanced tracking fields
    retry_count: int = 0
    original_job_id: Optional[str] = None  # For retried jobs
    priority: int = 0
    timeout_seconds: Optional[float] = None
    actual_duration: Optional[float] = None
    queue_time: Optional[float] = None  # Time spent in queue
    execution_start_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields"""
        if self.end_time and self.start_time:
            self.actual_duration = self.end_time - self.start_time

    def complete(self, outcome: str, result: Any = None, error: str = None):
        """Mark job as complete"""
        self.end_time = time.time()
        self.outcome = outcome
        self.result = result
        self.error = error

        if self.start_time:
            self.actual_duration = self.end_time - self.start_time

        # Calculate queue time if execution start time was recorded
        if self.execution_start_time and self.start_time:
            self.queue_time = self.execution_start_time - self.start_time

    def start_execution(self):
        """Mark when execution actually started (after queuing)"""
        self.execution_start_time = time.time()

        if self.start_time:
            self.queue_time = self.execution_start_time - self.start_time

    def is_complete(self) -> bool:
        """Check if job is complete"""
        return self.outcome is not None

    def is_successful(self) -> bool:
        """Check if job completed successfully"""
        return self.outcome == "success"

    def is_failed(self) -> bool:
        """Check if job failed"""
        return self.outcome in ["failed", "timeout", "cancelled"]

    def should_retry(self, max_retries: int = 3) -> bool:
        """Determine if job should be retried"""
        if self.retry_count >= max_retries:
            return False

        # Retry on failure, but not on cancellation
        if self.outcome in ["failed", "timeout"]:
            return True

        return False

    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.actual_duration is not None:
            return self.actual_duration

        if self.end_time and self.start_time:
            return self.end_time - self.start_time

        return None

    def get_execution_duration(self) -> Optional[float]:
        """Get actual execution duration (excluding queue time)"""
        if self.end_time and self.execution_start_time:
            return self.end_time - self.execution_start_time

        return None

    def add_checkpoint(self, checkpoint_path: str):
        """Add checkpoint path"""
        if checkpoint_path not in self.checkpoint_paths:
            self.checkpoint_paths.append(checkpoint_path)

    def add_child_job(self, child_job_id: str):
        """Add child job ID"""
        if child_job_id not in self.child_job_ids:
            self.child_job_ids.append(child_job_id)

    def update_resource_consumption(self, resources: Dict[str, float]):
        """Update resource consumption metrics"""
        self.resource_consumption.update(resources)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of job provenance including result data"""
        summary = {
            "job_id": self.job_id,
            "agent_id": self.agent_id,
            "graph_id": self.graph_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "outcome": self.outcome,
            "status": self.outcome,  # Alias for backwards compatibility
            "result": self.result,  # Include full result for reasoning output access
            "duration": self.get_duration(),
            "execution_duration": self.get_execution_duration(),
            "queue_time": self.queue_time,
            "retry_count": self.retry_count,
            "priority": self.priority,
            "resource_consumption": self.resource_consumption,
            "has_error": self.error is not None,
            "num_checkpoints": len(self.checkpoint_paths),
            "num_children": len(self.child_job_ids),
            "has_parent": self.parent_job_id is not None,
        }

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "job_id": self.job_id,
            "agent_id": self.agent_id,
            "graph_id": self.graph_id,
            "parameters": self.parameters,
            "hardware_used": self.hardware_used,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "outcome": self.outcome,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "resource_consumption": self.resource_consumption,
            "checkpoint_paths": self.checkpoint_paths,
            "parent_job_id": self.parent_job_id,
            "child_job_ids": self.child_job_ids,
            "retry_count": self.retry_count,
            "original_job_id": self.original_job_id,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "actual_duration": self.actual_duration,
            "queue_time": self.queue_time,
            "execution_start_time": self.execution_start_time,
            "metadata": self.metadata,
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def create_agent_metadata(
    agent_id: str,
    capability: AgentCapability = AgentCapability.GENERAL,
    location: str = "local",
    hardware_spec: Optional[Dict[str, Any]] = None,
) -> AgentMetadata:
    """
    Factory function to create agent metadata

    Args:
        agent_id: Unique agent identifier
        capability: Agent capability
        location: Agent location (local, remote, cloud)
        hardware_spec: Hardware specification

    Returns:
        AgentMetadata instance
    """
    current_time = time.time()

    return AgentMetadata(
        agent_id=agent_id,
        state=AgentState.INITIALIZING,
        capability=capability,
        created_at=current_time,
        last_active=current_time,
        location=location,
        hardware_spec=hardware_spec or {},
    )


def create_job_provenance(
    job_id: str,
    graph_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    timeout_seconds: Optional[float] = None,
    parent_job_id: Optional[str] = None,
) -> JobProvenance:
    """
    Factory function to create job provenance

    Args:
        job_id: Unique job identifier
        graph_id: Graph identifier
        parameters: Job parameters
        priority: Job priority
        timeout_seconds: Timeout in seconds
        parent_job_id: Parent job ID if this is a child job

    Returns:
        JobProvenance instance
    """
    return JobProvenance(
        job_id=job_id,
        agent_id="",  # Will be assigned later
        graph_id=graph_id,
        parameters=parameters or {},
        hardware_used={},
        start_time=time.time(),
        end_time=None,
        outcome=None,
        result=None,
        error=None,
        resource_consumption={},
        priority=priority,
        timeout_seconds=timeout_seconds,
        parent_job_id=parent_job_id,
    )


def validate_state_machine():
    """Validate that the state machine is properly configured"""
    all_states = set(AgentState)
    defined_states = set(StateTransitionRules.VALID_TRANSITIONS.keys())

    if all_states != defined_states:
        missing = all_states - defined_states
        extra = defined_states - all_states

        error_msg = []
        if missing:
            error_msg.append(f"Missing state transitions for: {missing}")
        if extra:
            error_msg.append(f"Extra state transitions for: {extra}")

        raise ValueError("State machine validation failed: " + "; ".join(error_msg))

    logger.info("Agent lifecycle state machine validated successfully")
    return True


# Validate state machine on module import
try:
    validate_state_machine()
except Exception as e:
    logger.error(f"State machine validation failed: {e}")
    raise


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "AgentState",
    "AgentCapability",
    "AgentMetadata",
    "JobProvenance",
    "StateTransitionRules",
    "create_agent_metadata",
    "create_job_provenance",
    "validate_state_machine",
]
