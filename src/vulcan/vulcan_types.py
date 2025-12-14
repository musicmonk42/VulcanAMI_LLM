# ============================================================
# VULCAN-AGI Type Definitions Module
# Complete type system with validation, versioning, and schema enforcement
# FIXED VERSION - Added missing types (SystemState, Episode, ProvRecord, SA_Latents, HealthSnapshot)
# ============================================================

import inspect
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar

import numpy as np

# Type validation libraries
try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from vulcan.config import ActionType, ModalityType, SafetyLevel

logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar("T")
StateType = TypeVar("StateType", bound="SystemState")
NodeType = TypeVar("NodeType", bound="IRNode")
EventType = TypeVar("EventType", bound="Event")

# ============================================================
# VERSION MANAGEMENT
# ============================================================


class SchemaVersion:
    """Schema version management."""

    MAJOR: int = 1
    MINOR: int = 3
    PATCH: int = 1

    @classmethod
    def get_version(cls) -> str:
        """Get current schema version."""
        return f"{cls.MAJOR}.{cls.MINOR}.{cls.PATCH}"

    @classmethod
    def is_compatible(cls, version: str) -> bool:
        """Check if version is compatible."""
        parts = version.split(".")
        if len(parts) != 3:
            return False

        major, minor, patch = map(int, parts)

        # Major version must match
        if major != cls.MAJOR:
            return False

        # Minor version can be lower or equal
        if minor > cls.MINOR:
            return False

        return True


# ============================================================
# IR NODE TYPES
# ============================================================


class IRNodeType(Enum):
    """Complete set of IR node types."""

    # Core compute nodes
    COMPUTE = "compute"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    MAP = "map"
    REDUCE = "reduce"
    FILTER = "filter"

    # Data flow nodes
    INPUT = "input"
    OUTPUT = "output"
    BUFFER = "buffer"
    CACHE = "cache"
    QUEUE = "queue"
    STREAM = "stream"

    # Control flow nodes
    CONDITIONAL = "conditional"
    LOOP = "loop"
    BRANCH = "branch"
    MERGE = "merge"
    BARRIER = "barrier"
    SYNCHRONIZE = "synchronize"

    # Memory nodes
    LOAD = "load"
    STORE = "store"
    ALLOCATE = "allocate"
    FREE = "free"
    COPY = "copy"
    MOVE = "move"

    # Communication nodes
    SEND = "send"
    RECEIVE = "receive"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    GATHER = "gather"
    ALL_REDUCE = "all_reduce"

    # Learning nodes
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZE = "optimize"
    LOSS = "loss"
    GRADIENT = "gradient"
    UPDATE = "update"

    # Reasoning nodes
    INFER = "infer"
    PROVE = "prove"
    HYPOTHESIZE = "hypothesize"
    VALIDATE = "validate"
    EXPLAIN = "explain"
    ABSTRACT = "abstract"

    # Planning nodes
    PLAN = "plan"
    SCHEDULE = "schedule"
    ALLOCATE_RESOURCES = "allocate_resources"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    DECIDE = "decide"

    # Safety nodes
    SAFETY_CHECK = "safety_check"
    VALIDATE_CONSTRAINTS = "validate_constraints"
    ROLLBACK = "rollback"
    CHECKPOINT = "checkpoint"
    AUDIT = "audit"
    ENCRYPT = "encrypt"

    # Monitoring nodes
    MONITOR = "monitor"
    MEASURE = "measure"
    PROFILE = "profile"
    TRACE = "trace"
    LOG = "log"
    ALERT = "alert"

    # Custom nodes
    CUSTOM = "custom"
    PLUGIN = "plugin"
    EXTERNAL = "external"


class IREdgeType(Enum):
    """Types of edges in IR graph."""

    DATA = "data"  # Data dependency
    CONTROL = "control"  # Control dependency
    MEMORY = "memory"  # Memory dependency
    TEMPORAL = "temporal"  # Temporal ordering
    CAUSAL = "causal"  # Causal relationship
    RESOURCE = "resource"  # Resource constraint
    SAFETY = "safety"  # Safety constraint
    PRIORITY = "priority"  # Priority ordering


@dataclass
class IRNode:
    """Base IR node with full validation."""

    id: str
    type: IRNodeType
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution properties
    device: str = "cpu"
    priority: int = 0
    timeout_ms: Optional[int] = None
    retry_count: int = 0

    # Resource requirements
    memory_mb: float = 0
    compute_flops: float = 0
    bandwidth_mbps: float = 0
    energy_nj: float = 0

    # Validation
    schema_version: str = field(default_factory=SchemaVersion.get_version)
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate node after initialization."""
        self.validate()

    def validate(self) -> bool:
        """Validate node structure and parameters."""
        self.validation_errors = []

        # Validate ID format
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.id):
            self.validation_errors.append(f"Invalid node ID format: {self.id}")

        # Validate type-specific parameters
        if not self._validate_params():
            self.validation_errors.append(
                f"Parameter validation failed for type {self.type}"
            )

        # Validate resource constraints
        if self.memory_mb < 0:
            self.validation_errors.append(
                f"Invalid memory requirement: {self.memory_mb}"
            )

        if self.energy_nj < 0:
            self.validation_errors.append(
                f"Invalid energy requirement: {self.energy_nj}"
            )

        self.validated = len(self.validation_errors) == 0
        return self.validated

    def _validate_params(self) -> bool:
        """Validate parameters based on node type."""
        required_params = {
            IRNodeType.COMPUTE: ["operation"],
            IRNodeType.TRANSFORM: ["transformation"],
            IRNodeType.CONDITIONAL: ["condition"],
            IRNodeType.LOOP: ["condition", "max_iterations"],
            IRNodeType.SAFETY_CHECK: ["safety_level", "constraints"],
        }

        if self.type in required_params:
            for param in required_params[self.type]:
                if param not in self.params:
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with validation."""
        if not self.validated:
            self.validate()

        return {
            "id": self.id,
            "type": self.type.value,
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "device": self.device,
            "priority": self.priority,
            "timeout_ms": self.timeout_ms,
            "retry_count": self.retry_count,
            "resources": {
                "memory_mb": self.memory_mb,
                "compute_flops": self.compute_flops,
                "bandwidth_mbps": self.bandwidth_mbps,
                "energy_nj": self.energy_nj,
            },
            "schema_version": self.schema_version,
            "validated": self.validated,
        }


@dataclass
class IREdge:
    """Edge in IR graph."""

    source: str
    target: str
    type: IREdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate edge."""
        if not self.source or not self.target:
            return False
        if self.weight < 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class IRGraph:
    """Complete IR graph with validation and versioning."""

    grammar_version: str
    nodes: List[IRNode]
    edges: List[IREdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Graph properties
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)
    subgraphs: Dict[str, "IRGraph"] = field(default_factory=dict)

    # Execution properties
    execution_order: List[str] = field(default_factory=list)
    parallelization_groups: List[List[str]] = field(default_factory=list)

    # Validation
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate entire graph."""
        self.validation_errors = []

        # Validate version compatibility
        if not SchemaVersion.is_compatible(self.grammar_version):
            self.validation_errors.append(
                f"Incompatible grammar version: {self.grammar_version}"
            )

        # Validate all nodes
        node_ids = set()
        for node in self.nodes:
            if not node.validate():
                self.validation_errors.extend(node.validation_errors)

            if node.id in node_ids:
                self.validation_errors.append(f"Duplicate node ID: {node.id}")
            node_ids.add(node.id)

        # Validate all edges
        for edge in self.edges:
            if not edge.validate():
                self.validation_errors.append(
                    f"Invalid edge: {edge.source} -> {edge.target}"
                )

            if edge.source not in node_ids:
                self.validation_errors.append(f"Edge source not found: {edge.source}")

            if edge.target not in node_ids:
                self.validation_errors.append(f"Edge target not found: {edge.target}")

        # Check for cycles in safety-critical paths
        if self._has_safety_cycles():
            self.validation_errors.append("Cycle detected in safety-critical path")

        self.validated = len(self.validation_errors) == 0
        return self.validated

    def _has_safety_cycles(self) -> bool:
        """Check for cycles in safety-critical paths."""
        # Build adjacency list for safety edges
        adj = defaultdict(list)
        for edge in self.edges:
            if edge.type == IREdgeType.SAFETY:
                adj[edge.source].append(edge.target)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node_id in adj.keys():
            if node_id not in visited:
                if has_cycle(node_id):
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with validation."""
        if not self.validated:
            self.validate()

        return {
            "grammar_version": self.grammar_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
            "entry_points": self.entry_points,
            "exit_points": self.exit_points,
            "execution_order": self.execution_order,
            "validated": self.validated,
        }


# ============================================================
# AGENT ROLES
# ============================================================


class AgentRole(Enum):
    """Complete set of agent roles."""

    # Core roles
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    PLANNER = "planner"
    LEARNER = "learner"
    REASONER = "reasoner"
    MONITOR = "monitor"

    # Specialized roles
    SAFETY_OFFICER = "safety_officer"
    RESOURCE_MANAGER = "resource_manager"
    MEMORY_KEEPER = "memory_keeper"
    COMMUNICATOR = "communicator"
    TRANSLATOR = "translator"
    VALIDATOR = "validator"

    # Research roles
    EXPLORER = "explorer"
    EXPERIMENTER = "experimenter"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    INNOVATOR = "innovator"

    # Support roles
    AUDITOR = "auditor"
    DEBUGGER = "debugger"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    LOAD_BALANCER = "load_balancer"
    CACHE_MANAGER = "cache_manager"


@dataclass
class AgentCapability:
    """Agent capability specification."""

    name: str
    category: str
    level: int  # 0-10 proficiency level
    certified: bool = False
    certification_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate capability."""
        if not 0 <= self.level <= 10:
            return False
        if self.certified and not self.certification_date:
            return False
        return True


@dataclass
class AgentProfile:
    """Complete agent profile."""

    agent_id: str
    role: AgentRole
    capabilities: List[AgentCapability] = field(default_factory=list)

    # Performance metrics
    success_rate: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    tasks_completed: int = 0

    # Resource limits
    max_memory_mb: int = 1000
    max_cpu_cores: int = 4
    max_gpu_memory_mb: int = 0
    energy_budget_nj: float = 1e9

    # Permissions
    allowed_actions: Set[ActionType] = field(default_factory=set)
    prohibited_actions: Set[ActionType] = field(default_factory=set)
    security_clearance: int = 0  # 0-5

    # Status
    status: str = "active"  # active, suspended, terminated
    last_active: datetime = field(default_factory=datetime.now)

    def validate(self) -> bool:
        """Validate agent profile."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.agent_id):
            return False

        for capability in self.capabilities:
            if not capability.validate():
                return False

        if not 0 <= self.success_rate <= 1:
            return False

        if not 0 <= self.error_rate <= 1:
            return False

        if not 0 <= self.security_clearance <= 5:
            return False

        return True


# ============================================================
# ACTION TYPES
# ============================================================


class ActionCategory(Enum):
    """Action categories for classification."""

    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    STORAGE = "storage"
    LEARNING = "learning"
    REASONING = "reasoning"
    PLANNING = "planning"
    SAFETY = "safety"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"


@dataclass
class ActionSpecification:
    """Complete action specification."""

    type: ActionType
    category: ActionCategory
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Constraints
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)

    # Resource requirements
    min_resources: Dict[str, float] = field(default_factory=dict)
    max_resources: Dict[str, float] = field(default_factory=dict)

    # Safety properties
    safety_level_required: SafetyLevel = SafetyLevel.STANDARD
    requires_validation: bool = True
    requires_audit: bool = False
    reversible: bool = False

    # Timing
    min_duration_ms: float = 0
    max_duration_ms: float = float("inf")
    timeout_ms: Optional[float] = None

    # Metadata
    version: str = "1.0.0"
    deprecated: bool = False
    replacement: Optional[str] = None

    def validate(self) -> bool:
        """Validate action specification."""
        # Check resource constraints
        for resource, min_val in self.min_resources.items():
            if resource in self.max_resources:
                if min_val > self.max_resources[resource]:
                    return False

        # Check timing constraints
        if self.min_duration_ms > self.max_duration_ms:
            return False

        if self.timeout_ms and self.timeout_ms < self.min_duration_ms:
            return False

        return True


@dataclass
class ActionResult:
    """Result of action execution."""

    action_id: str
    action_type: ActionType
    status: str  # success, failure, timeout, cancelled

    # Execution details
    start_time: float
    end_time: float
    duration_ms: float

    # Results
    outputs: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)

    # Resources used
    resources_consumed: Dict[str, float] = field(default_factory=dict)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Audit trail
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    safety_checks: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate action result."""
        if self.end_time < self.start_time:
            return False

        calculated_duration = (self.end_time - self.start_time) * 1000
        if abs(calculated_duration - self.duration_ms) > 1:
            return False

        return True


# ============================================================
# EVENT SYSTEM
# ============================================================


class EventPriority(IntEnum):
    """Event priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    DEBUG = 4


class EventCategory(Enum):
    """Event categories."""

    SYSTEM = "system"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    LEARNING = "learning"
    REASONING = "reasoning"
    PLANNING = "planning"
    COMMUNICATION = "communication"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class EventMetadata:
    """Event metadata."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    causation_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Routing information
    source_agent: Optional[str] = None
    target_agents: List[str] = field(default_factory=list)
    broadcast: bool = False

    # Timing
    ttl_ms: Optional[float] = None
    expiry_time: Optional[float] = None

    # Security
    encrypted: bool = False
    signed: bool = False
    signature: Optional[str] = None


@dataclass
class Event:
    """Enhanced event with full metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    category: EventCategory = EventCategory.INFO
    priority: EventPriority = EventPriority.NORMAL

    # Event data
    data: Any = None
    error: Optional[Exception] = None

    # Timing
    timestamp: float = field(default_factory=time.time)

    # Metadata
    metadata: EventMetadata = field(default_factory=EventMetadata)

    # Validation
    schema_version: str = field(default_factory=SchemaVersion.get_version)
    validated: bool = False

    def validate(self) -> bool:
        """Validate event."""
        if not self.type:
            return False

        if self.metadata.expiry_time and self.metadata.expiry_time < self.timestamp:
            return False

        self.validated = True
        return True

    def is_expired(self) -> bool:
        """Check if event has expired."""
        if self.metadata.expiry_time:
            return time.time() > self.metadata.expiry_time

        if self.metadata.ttl_ms:
            return (time.time() - self.timestamp) * 1000 > self.metadata.ttl_ms

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "category": self.category.value,
            "priority": self.priority,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": asdict(self.metadata),
            "schema_version": self.schema_version,
        }


# ============================================================
# SYSTEM STATE
# ============================================================


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    component_name: str
    status: str  # healthy, degraded, unhealthy, offline

    # Metrics
    uptime_seconds: float = 0
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    last_check: float = field(default_factory=time.time)

    # Performance
    avg_latency_ms: float = 0
    p99_latency_ms: float = 0
    throughput_per_second: float = 0

    # Resources
    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == "healthy"


@dataclass
class SystemHealth:
    """Complete system health state."""

    overall_status: str = "healthy"
    components: Dict[str, ComponentHealth] = field(default_factory=dict)

    # System metrics
    total_uptime_seconds: float = 0
    total_requests: int = 0
    failed_requests: int = 0

    # Resource usage
    total_memory_mb: float = 0
    total_cpu_percent: float = 0
    total_gpu_percent: float = 0
    total_energy_nj: float = 0

    # Performance
    avg_system_latency_ms: float = 0
    system_throughput: float = 0

    def update_component(self, name: str, health: ComponentHealth):
        """Update component health."""
        self.components[name] = health
        self._recalculate_overall_status()

    def _recalculate_overall_status(self):
        """Recalculate overall system status."""
        if not self.components:
            self.overall_status = "unknown"
            return

        statuses = [c.status for c in self.components.values()]

        if all(s == "healthy" for s in statuses):
            self.overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            self.overall_status = "unhealthy"
        elif any(s == "degraded" for s in statuses):
            self.overall_status = "degraded"
        else:
            self.overall_status = "unknown"


@dataclass
class KnowledgeState:
    """System knowledge state."""

    # Concept counts
    total_concepts: int = 0
    active_concepts: int = 0

    # Relationship counts
    causal_links: int = 0
    correlations: int = 0
    patterns: int = 0

    # Memory statistics
    episodic_memories: int = 0
    semantic_facts: int = 0
    procedural_skills: int = 0

    # Learning statistics
    total_experiences: int = 0
    positive_experiences: int = 0
    negative_experiences: int = 0

    # Model statistics
    models_loaded: int = 0
    model_parameters: int = 0
    model_version: str = "1.0.0"


@dataclass
class CommunicationState:
    """System communication state."""

    # Connection statistics
    active_connections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0

    # Protocol usage
    protocol_usage: Dict[str, int] = field(default_factory=dict)

    # Error statistics
    failed_sends: int = 0
    failed_receives: int = 0
    timeout_count: int = 0

    # Bandwidth usage
    bandwidth_used_mbps: float = 0
    peak_bandwidth_mbps: float = 0


@dataclass
class SecurityState:
    """System security state."""

    # Authentication
    authenticated_users: int = 0
    failed_auth_attempts: int = 0

    # Authorization
    access_granted: int = 0
    access_denied: int = 0

    # Threats
    threats_detected: int = 0
    threats_mitigated: int = 0

    # Encryption
    encrypted_messages: int = 0
    encryption_failures: int = 0

    # Audit
    audit_entries: int = 0
    compliance_violations: int = 0


@dataclass
class CompleteSystemState:
    """Complete system state with all subsystems."""

    # Identification
    system_id: str
    version: str

    # Core states
    health: SystemHealth = field(default_factory=SystemHealth)
    knowledge: KnowledgeState = field(default_factory=KnowledgeState)
    communication: CommunicationState = field(default_factory=CommunicationState)
    security: SecurityState = field(default_factory=SecurityState)

    # Execution state
    current_step: int = 0
    lifecycle_phase: str = (
        "initialization"  # initialization, running, maintenance, shutdown
    )

    # Active entities
    active_agents: Dict[str, AgentProfile] = field(default_factory=dict)
    active_plans: List[str] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)

    # Performance tracking
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Timestamps
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    def update(self):
        """Update state timestamp."""
        self.last_update = time.time()
        self.current_step += 1


# ============================================================
# ADDITIONAL TYPES FOR ORCHESTRATOR (FIXED - PREVIOUSLY MISSING)
# ============================================================


@dataclass
class SA_Latents:
    """Self-awareness latent state."""

    uncertainty: float = 0.5
    identity_drift: float = 0.0
    learning_efficiency: float = 0.5
    metacognitive_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "uncertainty": self.uncertainty,
            "identity_drift": self.identity_drift,
            "learning_efficiency": self.learning_efficiency,
            "metacognitive_confidence": self.metacognitive_confidence,
        }


@dataclass
class HealthSnapshot:
    """Health metrics snapshot."""

    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0
    latency_ms: float = 0
    error_rate: float = 0
    energy_budget_left_nJ: float = 1e9

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
            "energy_budget_left_nJ": self.energy_budget_left_nJ,
        }


@dataclass
class Episode:
    """Experience episode for memory."""

    t: float
    context: Dict[str, Any]
    action_bundle: Dict[str, Any]
    observation: Any
    reward_vec: Dict[str, float]
    SA_latents: SA_Latents
    expl_uri: str
    prov_sig: str
    modalities_used: Set[ModalityType]
    uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "t": self.t,
            "context": self.context,
            "action_bundle": self.action_bundle,
            "observation": str(self.observation),
            "reward_vec": self.reward_vec,
            "SA_latents": self.SA_latents.to_dict(),
            "expl_uri": self.expl_uri,
            "prov_sig": self.prov_sig,
            "modalities_used": [m.value for m in self.modalities_used],
            "uncertainty": self.uncertainty,
        }


@dataclass
class ProvRecord:
    """Provenance record for auditability."""

    t: float
    graph_id: str
    agent_version: str
    policy_versions: Dict[str, str]
    input_hash: str
    kernel_sig: Optional[str]
    explainer_uri: str
    ecdsa_sig: str
    modality: ModalityType
    uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "t": self.t,
            "graph_id": self.graph_id,
            "agent_version": self.agent_version,
            "policy_versions": self.policy_versions,
            "input_hash": self.input_hash,
            "kernel_sig": self.kernel_sig,
            "explainer_uri": self.explainer_uri,
            "ecdsa_sig": self.ecdsa_sig,
            "modality": self.modality.value,
            "uncertainty": self.uncertainty,
        }


@dataclass
class SystemState:
    """Main system state (simplified from CompleteSystemState for orchestrator)."""

    CID: str  # Context ID
    step: int = 0
    policies: Dict[str, str] = field(default_factory=dict)
    SA: SA_Latents = field(default_factory=SA_Latents)
    health: HealthSnapshot = field(default_factory=HealthSnapshot)
    active_modalities: Set[ModalityType] = field(default_factory=set)
    uncertainty_estimates: Dict[str, float] = field(default_factory=dict)
    provenance_chain: List[ProvRecord] = field(default_factory=list)
    last_obs: Any = None
    last_reward: Optional[float] = None

    def update_step(self):
        """Increment step counter."""
        self.step += 1

    def add_provenance(self, record: ProvRecord):
        """Add provenance record."""
        self.provenance_chain.append(record)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "CID": self.CID,
            "step": self.step,
            "policies": self.policies,
            "SA": self.SA.to_dict(),
            "health": self.health.to_dict(),
            "active_modalities": [m.value for m in self.active_modalities],
            "uncertainty_estimates": self.uncertainty_estimates,
            "provenance_chain": [p.to_dict() for p in self.provenance_chain],
            "last_obs": str(self.last_obs) if self.last_obs else None,
            "last_reward": self.last_reward,
        }


# ============================================================
# TYPE VALIDATION
# ============================================================


class TypeValidator:
    """Universal type validator for all system types."""

    @staticmethod
    def validate_type(obj: Any, expected_type: Type) -> Tuple[bool, List[str]]:
        """Validate object against expected type."""
        errors = []

        # Handle None
        if obj is None:
            if expected_type is type(None):
                return True, []
            else:
                errors.append(f"Expected {expected_type}, got None")
                return False, errors

        # Check basic type
        if not isinstance(obj, expected_type):
            errors.append(f"Expected {expected_type}, got {type(obj)}")
            return False, errors

        # Additional validation for dataclasses
        if is_dataclass(expected_type):
            return TypeValidator._validate_dataclass(obj, errors)

        # Additional validation for enums
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            return TypeValidator._validate_enum(obj, expected_type, errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_dataclass(obj: Any, errors: List[str]) -> Tuple[bool, List[str]]:
        """Validate dataclass fields."""
        if hasattr(obj, "validate"):
            if not obj.validate():
                if hasattr(obj, "validation_errors"):
                    errors.extend(obj.validation_errors)
                else:
                    errors.append(f"Validation failed for {type(obj).__name__}")

        return len(errors) == 0, errors

    @staticmethod
    def _validate_enum(
        obj: Any, enum_type: Type[Enum], errors: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate enum value."""
        if obj not in enum_type:
            errors.append(f"Invalid enum value: {obj} not in {enum_type.__name__}")

        return len(errors) == 0, errors


# ============================================================
# API TYPE ENFORCEMENT
# ============================================================


def enforce_types(func: Callable) -> Callable:
    """Decorator to enforce type checking on function inputs/outputs."""

    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validate input types
        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                valid, errors = TypeValidator.validate_type(
                    param_value, param.annotation
                )
                if not valid:
                    raise TypeError(f"Parameter {param_name}: {', '.join(errors)}")

        # Execute function
        result = func(*args, **kwargs)

        # Validate return type
        if sig.return_annotation != inspect.Signature.empty:
            valid, errors = TypeValidator.validate_type(result, sig.return_annotation)
            if not valid:
                raise TypeError(f"Return value: {', '.join(errors)}")

        return result

    return wrapper


# ============================================================
# SCHEMA DEFINITIONS
# ============================================================


class IRSchemas:
    """Versioned IR schemas."""

    SCHEMAS = {
        "1.3.1": {
            "type": "object",
            "required": ["grammar_version", "nodes", "edges"],
            "properties": {
                "grammar_version": {"type": "string"},
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "type"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "params": {"type": "object"},
                            "inputs": {"type": "array"},
                            "outputs": {"type": "array"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["source", "target", "type"],
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "type": {"type": "string"},
                            "weight": {"type": "number"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
            },
        }
    }

    @classmethod
    def validate_graph(cls, graph_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate graph against schema."""
        if not JSONSCHEMA_AVAILABLE:
            return True, []  # Skip validation if jsonschema not available

        version = graph_dict.get("grammar_version", "1.3.1")
        schema = cls.SCHEMAS.get(version)

        if not schema:
            return False, [f"No schema found for version {version}"]

        try:
            jsonschema.validate(graph_dict, schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [str(e)]


# ============================================================
# MIGRATION SUPPORT
# ============================================================


class SchemaMigrator:
    """Handle schema version migrations."""

    MIGRATIONS = {
        ("1.0.0", "1.1.0"): lambda d: SchemaMigrator._migrate_1_0_to_1_1(d),
        ("1.1.0", "1.2.0"): lambda d: SchemaMigrator._migrate_1_1_to_1_2(d),
        ("1.2.0", "1.3.0"): lambda d: SchemaMigrator._migrate_1_2_to_1_3(d),
        ("1.3.0", "1.3.1"): lambda d: d,  # No changes needed
    }

    @classmethod
    def migrate(
        cls, data: Dict[str, Any], target_version: str = None
    ) -> Dict[str, Any]:
        """Migrate data to target version."""
        current_version = data.get("schema_version", "1.0.0")
        target_version = target_version or SchemaVersion.get_version()

        if current_version == target_version:
            return data

        # Find migration path
        path = cls._find_migration_path(current_version, target_version)

        if not path:
            raise ValueError(
                f"No migration path from {current_version} to {target_version}"
            )

        # Apply migrations
        migrated_data = data.copy()
        for from_ver, to_ver in path:
            migration_func = cls.MIGRATIONS.get((from_ver, to_ver))
            if migration_func:
                migrated_data = migration_func(migrated_data)
                migrated_data["schema_version"] = to_ver

        return migrated_data

    @classmethod
    def _find_migration_path(
        cls, from_version: str, to_version: str
    ) -> List[Tuple[str, str]]:
        """Find migration path between versions."""
        # Build version graph
        graph = defaultdict(list)
        for (from_ver, to_ver), _ in cls.MIGRATIONS.items():
            graph[from_ver].append(to_ver)

        # BFS to find path
        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current, path = queue.popleft()

            if current == to_version:
                return path

            for next_ver in graph[current]:
                if next_ver not in visited:
                    visited.add(next_ver)
                    new_path = path + [(current, next_ver)]
                    queue.append((next_ver, new_path))

        return []

    @staticmethod
    def _migrate_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.0.0 to 1.1.0."""
        # Add metadata field if missing
        if "metadata" not in data:
            data["metadata"] = {}

        # Update node structure
        for node in data.get("nodes", []):
            if "device" not in node:
                node["device"] = "cpu"

        return data

    @staticmethod
    def _migrate_1_1_to_1_2(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.1.0 to 1.2.0."""
        # Add resource requirements to nodes
        for node in data.get("nodes", []):
            if "resources" not in node:
                node["resources"] = {
                    "memory_mb": 0,
                    "compute_flops": 0,
                    "bandwidth_mbps": 0,
                    "energy_nj": 0,
                }

        return data

    @staticmethod
    def _migrate_1_2_to_1_3(data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.2.0 to 1.3.0."""
        # Add validation fields
        for node in data.get("nodes", []):
            if "validated" not in node:
                node["validated"] = False

        return data


# ============================================================
# TYPE REGISTRY
# ============================================================


class TypeRegistry:
    """Global type registry for dynamic type resolution."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, type_class: Type):
        """Register a type."""
        cls._registry[name] = type_class

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a registered type."""
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """Create an instance of a registered type."""
        type_class = cls.get(name)
        if type_class:
            return type_class(**kwargs)
        raise ValueError(f"Type {name} not registered")


# Register all types
TypeRegistry.register("IRNode", IRNode)
TypeRegistry.register("IREdge", IREdge)
TypeRegistry.register("IRGraph", IRGraph)
TypeRegistry.register("Event", Event)
TypeRegistry.register("AgentProfile", AgentProfile)
TypeRegistry.register("ActionSpecification", ActionSpecification)
TypeRegistry.register("ActionResult", ActionResult)
TypeRegistry.register("CompleteSystemState", CompleteSystemState)
TypeRegistry.register("SystemState", SystemState)
TypeRegistry.register("Episode", Episode)
TypeRegistry.register("ProvRecord", ProvRecord)
TypeRegistry.register("SA_Latents", SA_Latents)
TypeRegistry.register("HealthSnapshot", HealthSnapshot)

# ============================================================
# SERIALIZATION SUPPORT
# ============================================================


class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for all custom types."""

    def default(self, obj):
        if is_dataclass(obj):
            return {
                "__type__": type(obj).__name__,
                "__module__": type(obj).__module__,
                "__data__": asdict(obj),
            }
        elif isinstance(obj, Enum):
            return {
                "__type__": "Enum",
                "__class__": type(obj).__name__,
                "__value__": obj.value,
            }
        elif isinstance(obj, datetime):
            return {"__type__": "datetime", "__value__": obj.isoformat()}
        elif isinstance(obj, uuid.UUID):
            return {"__type__": "UUID", "__value__": str(obj)}
        elif isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "__value__": obj.tolist(),
                "__dtype__": str(obj.dtype),
                "__shape__": obj.shape,
            }
        elif isinstance(obj, Exception):
            return {
                "__type__": "Exception",
                "__class__": type(obj).__name__,
                "__message__": str(obj),
            }
        elif isinstance(obj, set):
            return {"__type__": "set", "__value__": list(obj)}

        return super().default(obj)


class EnhancedJSONDecoder(json.JSONDecoder):
    """Enhanced JSON decoder for all custom types."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "__type__" not in obj:
            return obj

        type_name = obj["__type__"]

        if type_name == "datetime":
            return datetime.fromisoformat(obj["__value__"])
        elif type_name == "UUID":
            return uuid.UUID(obj["__value__"])
        elif type_name == "ndarray":
            arr = np.array(obj["__value__"], dtype=obj["__dtype__"])
            return arr.reshape(obj["__shape__"])
        elif type_name == "set":
            return set(obj["__value__"])
        elif type_name == "Enum":
            # Need to resolve enum class
            return obj["__value__"]  # Return value for now
        elif type_name == "Exception":
            # Reconstruct exception
            exc_class = type(obj["__class__"], (Exception,), {})
            return exc_class(obj["__message__"])
        else:
            # Try to resolve from type registry
            type_class = TypeRegistry.get(type_name)
            if type_class and "__data__" in obj:
                return type_class(**obj["__data__"])

        return obj


# ============================================================
# TYPE EXPORTS
# ============================================================

__all__ = [
    # Version management
    "SchemaVersion",
    # IR types
    "IRNodeType",
    "IREdgeType",
    "IRNode",
    "IREdge",
    "IRGraph",
    # Agent types
    "AgentRole",
    "AgentCapability",
    "AgentProfile",
    # Action types
    "ActionCategory",
    "ActionSpecification",
    "ActionResult",
    # Event types
    "EventPriority",
    "EventCategory",
    "EventMetadata",
    "Event",
    # System state types
    "ComponentHealth",
    "SystemHealth",
    "KnowledgeState",
    "CommunicationState",
    "SecurityState",
    "CompleteSystemState",
    # Orchestrator-specific types (FIXED - ADDED)
    "SystemState",
    "Episode",
    "ProvRecord",
    "SA_Latents",
    "HealthSnapshot",
    # Validation
    "TypeValidator",
    "enforce_types",
    # Schemas
    "IRSchemas",
    "SchemaMigrator",
    # Registry
    "TypeRegistry",
    # Serialization
    "EnhancedJSONEncoder",
    "EnhancedJSONDecoder",
]
