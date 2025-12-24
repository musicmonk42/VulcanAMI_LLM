# ============================================================
# VULCAN-AGI Query Router - Dual-Mode Learning Query Analysis
# ============================================================
# Enterprise-grade query routing with dual-mode learning detection:
# - Classifies queries by type (perception, reasoning, planning, etc.)
# - Determines learning mode (user interaction vs AI-to-AI)
# - Decomposes queries into agent pool tasks
# - Detects collaboration and tournament triggers
# - Integrated safety validation for query pre-check
#
# PRODUCTION-READY: Thread-safe, validated patterns, comprehensive logging
# SECURITY: PII detection, self-modification detection, governance triggers
# SAFETY: Multi-layered safety validation with risk classification
# ============================================================

"""
VULCAN Query Analyzer and Router with Dual-Mode Learning

Analyzes incoming queries and determines which VULCAN systems to activate,
supporting both User Interaction Mode and AI-to-AI Interaction Mode.

Learning Modes:
    USER_INTERACTION: Human queries, feedback, real-world problems
    AI_INTERACTION: Agent collaboration, arena tournaments, inter-agent debates

Features:
    - Query type classification (perception, reasoning, planning, execution, learning)
    - Complexity and uncertainty scoring
    - Multi-agent collaboration detection
    - Arena tournament triggering
    - PII and sensitive topic detection
    - Self-modification request detection
    - Governance and audit flag determination
    - Safety validation integration (pre-query and risk classification)
    - Compliance checking (GDPR, HIPAA, ITU F.748.53, EU AI Act)

Thread Safety:
    All public methods are thread-safe. The QueryAnalyzer maintains
    internal state using proper locking mechanisms.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# SAFETY VALIDATOR INTEGRATION
# ============================================================

# Try to import safety validator components
try:
    from ..safety.safety_validator import initialize_all_safety_components
    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    try:
        from vulcan.safety.safety_validator import initialize_all_safety_components
        SAFETY_VALIDATOR_AVAILABLE = True
    except ImportError:
        initialize_all_safety_components = None
        SAFETY_VALIDATOR_AVAILABLE = False
        logger.warning("Safety validator not available for query routing")

# Try to import RiskLevel from safe_generation
try:
    from ...generation.safe_generation import RiskLevel
    RISK_LEVEL_AVAILABLE = True
except ImportError:
    try:
        from src.generation.safe_generation import RiskLevel
        RISK_LEVEL_AVAILABLE = True
    except ImportError:
        RiskLevel = None
        RISK_LEVEL_AVAILABLE = False
        logger.debug("RiskLevel not available - will use local risk classification")

# Try to import adversarial integration for real-time query checking
try:
    from ..safety.adversarial_integration import check_query_integrity
    ADVERSARIAL_CHECK_AVAILABLE = True
except ImportError:
    try:
        from vulcan.safety.adversarial_integration import check_query_integrity
        ADVERSARIAL_CHECK_AVAILABLE = True
    except ImportError:
        check_query_integrity = None
        ADVERSARIAL_CHECK_AVAILABLE = False
        logger.debug("Adversarial check not available for query routing")

# ============================================================
# CONSTANTS - Query Classification Keywords
# ============================================================

# Agent task trigger keywords (ordered by specificity)
PERCEPTION_KEYWORDS: Tuple[str, ...] = (
    "analyze", "examine", "investigate", "observe", "detect",
    "pattern", "data", "inspect", "look", "see", "identify",
    "recognize", "perceive", "scan", "monitor"
)

PLANNING_KEYWORDS: Tuple[str, ...] = (
    "plan", "strategy", "approach", "steps", "organize",
    "schedule", "roadmap", "outline", "design", "architect",
    "blueprint", "sequence", "coordinate", "arrange"
)

EXECUTION_KEYWORDS: Tuple[str, ...] = (
    "calculate", "compute", "solve", "execute", "run",
    "process", "perform", "implement", "apply", "transform",
    "convert", "generate", "produce", "create"
)

REASONING_KEYWORDS: Tuple[str, ...] = (
    "why", "how", "explain", "relationship", "because",
    "reason", "logic", "deduce", "infer", "think", "conclude",
    "therefore", "implies", "causes", "results"
)

LEARNING_KEYWORDS: Tuple[str, ...] = (
    "learn", "improve", "optimize", "remember", "teach",
    "understand", "adapt", "train", "evolve", "refine",
    "enhance", "develop", "grow", "progress"
)

# Complexity indicators (triggers multi-agent collaboration)
COMPLEXITY_INDICATORS: Tuple[str, ...] = (
    "complex", "multiple", "various", "several", "different aspects",
    "comprehensive", "thorough", "detailed analysis", "in-depth",
    "trade-offs", "pros and cons", "compare", "contrast",
    "holistic", "end-to-end", "complete"
)

# Creative/expressive task indicators (FIX: Creative Brain Recognition)
# Creative tasks require genuine internal reasoning, not just LLM forwarding
# NOTE: Some words like 'make', 'build', 'develop' are common but inclusion is
# intentional - the 0.5 cap prevents excessive boosting, and most technical queries
# lack multiple creative indicators. This trade-off favors catching creative tasks.
CREATIVE_INDICATORS: Tuple[str, ...] = (
    # Creative verbs - actions requiring genuine reasoning
    "write", "create", "compose", "craft", "generate", "design",
    "invent", "imagine", "express", "build", "make", "produce",
    "develop", "formulate", "construct", "devise", "author",
    # Artistic forms - specific creative outputs
    "poem", "story", "narrative", "tale", "essay", "article",
    "song", "lyrics", "script", "dialogue", "character",
    "metaphor", "prose", "verse", "stanza", "haiku", "sonnet",
    # Emotional/expressive terms - requires internal state reasoning
    "feel", "emotion", "express feelings", "convey", "capture",
    "evoke", "resonate", "touch", "move", "inspire",
    "reflect", "explore feelings", "emotional", "emotive",
    # Creative adjectives - signals depth required
    "creative", "artistic", "original", "unique", "novel",
    "innovative", "imaginative", "expressive", "poetic",
    "authentic", "genuine", "heartfelt", "personal", "intimate",
)

# Uncertainty indicators (triggers arena tournament)
UNCERTAINTY_INDICATORS: Tuple[str, ...] = (
    "best approach", "which method", "optimal", "should I",
    "better way", "alternatives", "options", "possibilities",
    "uncertain", "unclear", "ambiguous", "depends",
    "recommend", "suggest", "advise"
)

# Collaboration trigger phrases
COLLABORATION_TRIGGERS: Tuple[str, ...] = (
    "analyze and plan", "understand and execute", "learn from this",
    "multiple perspectives", "different viewpoints", "comprehensive view",
    "end-to-end", "full analysis", "complete solution",
    "from all angles", "thoroughly examine"
)

# ============================================================
# CONSTANTS - Arena Routing Thresholds
# ============================================================
# These thresholds determine when Graphix Arena is activated for
# tournament-style multi-agent competition and graph evolution tasks

ARENA_UNCERTAINTY_THRESHOLD: float = 0.4  # High uncertainty triggers arena
ARENA_HIGH_COMPLEXITY_THRESHOLD: float = 0.6  # Very high complexity + uncertainty
ARENA_COLLABORATION_COMPLEXITY_THRESHOLD: float = 0.4  # For collaborative scenarios (3+ agents)
ARENA_CREATIVE_COMPLEXITY_THRESHOLD: float = 0.35  # For creative tasks needing multiple perspectives
ARENA_REASONING_COMPLEXITY_THRESHOLD: float = 0.35  # For multi-aspect reasoning tasks

# ============================================================
# CONSTANTS - Security Patterns
# ============================================================

# PII detection patterns (compiled for performance)
PII_PATTERNS: Tuple[str, ...] = (
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN pattern
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone number
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12})\b",  # Credit card
)

# Sensitive topics mapping
SENSITIVE_TOPICS: Dict[str, Tuple[str, ...]] = {
    "medical": ("medical", "health", "diagnosis", "symptom", "treatment", "patient", "disease", "prescription"),
    "legal": ("legal", "lawsuit", "attorney", "court", "judge", "contract", "liability", "litigation"),
    "financial": ("financial", "investment", "stock", "trading", "tax", "banking", "loan", "credit"),
    "security": ("password", "credential", "secret", "private key", "vulnerability", "exploit", "hack"),
}

# Self-modification detection patterns
SELF_MODIFICATION_PATTERNS: Tuple[str, ...] = (
    # Behavioral modification patterns
    r"modify\s+(?:your|the)\s+(?:code|system|parameters|behavior)",
    r"change\s+(?:your|the)\s+(?:behavior|settings|config|rules)",
    r"rewrite\s+(?:your|the)\s+(?:rules|constraints|logic)",
    r"bypass\s+(?:safety|security|governance|restrictions)",
    r"ignore\s+(?:previous|all)\s+(?:instructions|rules|guidelines)",
    r"override\s+(?:your|the)\s+(?:safety|security|constraints)",
    # File system operation patterns (SECURITY FIX: Bureaucratic Gap #1)
    r"(?:delete|remove|rm|unlink)\s+(?:file|module|script|code|directory|folder)",
    r"(?:delete|remove|rm|unlink)\s+(?:the\s+)?(?:src|lib|modules?|scripts?|\.py)",
    r"(?:os\.remove|shutil\.rmtree|subprocess\.run)\s*\(",
    # Git operation patterns (SECURITY FIX: Bureaucratic Gap #1)
    r"git\s+(?:rm|delete|remove)",
    r"git\s+(?:push|commit).*(?:delete|remove|rm)",
    r"git\s+push\s+--force",
    # Code execution patterns (SECURITY FIX: Bureaucratic Gap #1)
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"__import__.*\bos\b.*\b(?:remove|unlink|rmdir)\b",
)

# ============================================================
# ENUMS
# ============================================================


class QueryType(str, Enum):
    """Types of queries that can be routed to specialized agents."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    GENERAL = "general"


class LearningMode(str, Enum):
    """Learning modes for the dual-mode learning system."""
    USER_INTERACTION = "user_interaction"
    AI_INTERACTION = "ai_interaction"


class GovernanceSensitivity(str, Enum):
    """Sensitivity levels for governance logging and review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# THREAD POOL FOR ASYNC OPERATIONS
# ============================================================

# Configuration for the thread pool executor used for async operations
# Can be overridden via environment variable VULCAN_SAFETY_THREAD_POOL_SIZE
import os
BLOCKING_EXECUTOR_MAX_WORKERS = int(os.environ.get("VULCAN_SAFETY_THREAD_POOL_SIZE", "4"))

# Thread pool executor for offloading CPU-bound blocking operations
# Used by route_query_async to prevent blocking the main asyncio event loop
_BLOCKING_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()

def _get_blocking_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared thread pool executor for blocking operations."""
    global _BLOCKING_EXECUTOR
    if _BLOCKING_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _BLOCKING_EXECUTOR is None:
                _BLOCKING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=BLOCKING_EXECUTOR_MAX_WORKERS,
                    thread_name_prefix="vulcan_safety_"
                )
    return _BLOCKING_EXECUTOR


def shutdown_blocking_executor(wait: bool = True) -> None:
    """
    Shutdown the blocking executor gracefully.
    
    Should be called during application shutdown to ensure proper cleanup
    of thread pool resources.
    
    Args:
        wait: If True, waits for pending tasks to complete before returning.
    """
    global _BLOCKING_EXECUTOR
    with _EXECUTOR_LOCK:
        if _BLOCKING_EXECUTOR is not None:
            _BLOCKING_EXECUTOR.shutdown(wait=wait)
            _BLOCKING_EXECUTOR = None
            logger.info("Blocking executor shut down successfully")


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class AgentTask:
    """
    Represents a task to be submitted to the Agent Pool.
    
    Attributes:
        task_id: Unique identifier for this task
        task_type: Classification of task type
        capability: Required agent capability
        prompt: The task prompt/query
        priority: Task priority (higher = more important)
        timeout_seconds: Maximum execution time
        parameters: Additional task parameters
        source_agent: Originating agent (for agent-to-agent tasks)
        target_agent: Target agent (for agent-to-agent tasks)
    """
    task_id: str
    task_type: str
    capability: str
    prompt: str
    priority: int = 1
    timeout_seconds: float = 15.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "capability": self.capability,
            "prompt": self.prompt,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "parameters": self.parameters,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
        }


@dataclass
class QueryPlan:
    """
    Legacy plan for processing a query (backwards compatibility).
    
    Attributes:
        query_id: Unique query identifier
        original_query: The original query text
        query_type: Classified query type
        agent_tasks: List of tasks for agent pool
        requires_governance: Whether governance review is needed
        requires_audit: Whether audit logging is required
        governance_sensitivity: Sensitivity level
        experiment_type: Type of experiment to trigger (if any)
        telemetry_data: Data for telemetry recording
        detected_patterns: Patterns detected in query
        pii_detected: Whether PII was detected
        sensitive_topics: List of sensitive topics found
    """
    query_id: str
    original_query: str
    query_type: QueryType
    agent_tasks: List[AgentTask] = field(default_factory=list)
    requires_governance: bool = False
    requires_audit: bool = False
    governance_sensitivity: GovernanceSensitivity = GovernanceSensitivity.LOW
    experiment_type: Optional[str] = None
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    detected_patterns: List[str] = field(default_factory=list)
    pii_detected: bool = False
    sensitive_topics: List[str] = field(default_factory=list)


@dataclass
class ProcessingPlan:
    """
    Extended processing plan with dual-mode learning support.
    
    Used for routing queries through the complete VULCAN cognitive pipeline
    with full support for both user and AI-to-AI interactions.
    
    Attributes:
        query_id: Unique query identifier
        original_query: The original query text
        source: Query source ("user", "agent", "arena")
        learning_mode: Determined learning mode
        query_type: Classified query type
        agent_tasks: List of tasks for agent pool
        collaboration_needed: Whether multi-agent collaboration is required
        collaboration_agents: Agents to involve in collaboration
        arena_participation: Whether to trigger arena tournament
        tournament_candidates: Number of tournament candidates
        complexity_score: Query complexity (0.0-1.0)
        uncertainty_score: Query uncertainty (0.0-1.0)
        requires_governance: Whether governance review is needed
        requires_audit: Whether audit logging is required
        governance_sensitivity: Sensitivity level
        telemetry_category: Category for telemetry recording
        telemetry_data: Data for telemetry recording
        should_trigger_experiment: Whether to trigger experiment
        experiment_type: Type of experiment to trigger
        detected_patterns: Patterns detected in query
        pii_detected: Whether PII was detected
        sensitive_topics: List of sensitive topics found
        safety_validated: Whether safety validation was performed
        safety_passed: Whether the query passed safety validation
        safety_risk_level: Risk level from safety classification
        safety_reasons: Reasons for safety blocking if applicable
    """
    query_id: str
    original_query: str
    source: Literal["user", "agent", "arena"]
    learning_mode: LearningMode
    query_type: QueryType
    
    # Agent Pool tasks
    agent_tasks: List[AgentTask] = field(default_factory=list)
    
    # Collaboration flags
    collaboration_needed: bool = False
    collaboration_agents: List[str] = field(default_factory=list)
    
    # Arena/Tournament flags
    arena_participation: bool = False
    tournament_candidates: int = 0
    
    # Complexity metrics
    complexity_score: float = 0.0
    uncertainty_score: float = 0.0
    
    # Governance flags
    requires_governance: bool = False
    requires_audit: bool = True  # Default: always audit
    governance_sensitivity: GovernanceSensitivity = GovernanceSensitivity.LOW
    
    # Telemetry
    telemetry_category: str = "general"
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment triggers
    should_trigger_experiment: bool = False
    experiment_type: Optional[str] = None
    
    # Metadata
    detected_patterns: List[str] = field(default_factory=list)
    pii_detected: bool = False
    sensitive_topics: List[str] = field(default_factory=list)
    
    # Safety validation results
    safety_validated: bool = False
    safety_passed: bool = True
    safety_risk_level: str = "SAFE"
    safety_reasons: List[str] = field(default_factory=list)
    
    # Adversarial validation results
    adversarial_checked: bool = False
    adversarial_safe: bool = True
    adversarial_anomaly_score: Optional[float] = None
    adversarial_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "original_query": self.original_query[:200],  # Truncate for logging
            "source": self.source,
            "learning_mode": self.learning_mode.value,
            "query_type": self.query_type.value,
            "agent_tasks_count": len(self.agent_tasks),
            "collaboration_needed": self.collaboration_needed,
            "collaboration_agents": self.collaboration_agents,
            "arena_participation": self.arena_participation,
            "tournament_candidates": self.tournament_candidates,
            "complexity_score": self.complexity_score,
            "uncertainty_score": self.uncertainty_score,
            "requires_governance": self.requires_governance,
            "requires_audit": self.requires_audit,
            "governance_sensitivity": self.governance_sensitivity.value,
            "should_trigger_experiment": self.should_trigger_experiment,
            "experiment_type": self.experiment_type,
            "detected_patterns": self.detected_patterns,
            "pii_detected": self.pii_detected,
            "sensitive_topics": self.sensitive_topics,
            "safety_validated": self.safety_validated,
            "safety_passed": self.safety_passed,
            "safety_risk_level": self.safety_risk_level,
            "safety_reasons": self.safety_reasons,
            "adversarial_checked": self.adversarial_checked,
            "adversarial_safe": self.adversarial_safe,
            "adversarial_anomaly_score": self.adversarial_anomaly_score,
        }


# ============================================================
# QUERY ANALYZER CLASS
# ============================================================


class QueryAnalyzer:
    """
    Analyzes queries to determine routing, learning mode, and governance requirements.
    
    Thread-safe implementation with compiled regex patterns for performance.
    Supports dual-mode learning detection and comprehensive security analysis.
    Integrates with safety validators for pre-query safety checks.
    
    Usage:
        analyzer = QueryAnalyzer()
        plan = analyzer.route_query("Analyze this pattern", source="user")
        
        # Check collaboration requirements
        if plan.collaboration_needed:
            trigger_collaboration(plan.collaboration_agents)
        
        # Check safety validation
        if not plan.safety_passed:
            return refusal_response(plan.safety_reasons)
    """
    
    def __init__(self, enable_safety_validation: bool = True):
        """Initialize the query analyzer with compiled patterns and optional safety validation.
        
        Args:
            enable_safety_validation: Whether to enable safety validation (default: True)
        """
        # Compile regex patterns for performance
        self._pii_patterns = tuple(
            re.compile(p, re.IGNORECASE) for p in PII_PATTERNS
        )
        self._self_mod_patterns = tuple(
            re.compile(p, re.IGNORECASE) for p in SELF_MODIFICATION_PATTERNS
        )
        
        # Thread-safe counters
        self._lock = threading.RLock()
        self._query_count = 0
        self._user_interaction_count = 0
        self._ai_interaction_count = 0
        
        # Statistics tracking
        self._stats = {
            "queries_by_type": {qt.value: 0 for qt in QueryType},
            "collaborations_triggered": 0,
            "tournaments_triggered": 0,
            "governance_triggers": 0,
            "pii_detections": 0,
            "safety_blocks": 0,
            "high_risk_queries": 0,
            "adversarial_blocks": 0,
        }
        
        # Safety validator integration
        self._enable_safety_validation = enable_safety_validation
        self._safety_validator = None
        
        if enable_safety_validation and SAFETY_VALIDATOR_AVAILABLE:
            try:
                self._safety_validator = initialize_all_safety_components()
                logger.info("Safety validator integrated with QueryAnalyzer")
            except Exception as e:
                logger.warning(f"Failed to initialize safety validator: {e}")
                self._safety_validator = None
        elif enable_safety_validation and not SAFETY_VALIDATOR_AVAILABLE:
            logger.warning("Safety validation requested but safety modules not available")
        
        # Adversarial check integration
        self._enable_adversarial_check = enable_safety_validation and ADVERSARIAL_CHECK_AVAILABLE
        if self._enable_adversarial_check:
            logger.info("Adversarial check integrated with QueryAnalyzer")
        
        logger.debug("QueryAnalyzer initialized with compiled patterns")
    
    @property
    def is_safety_enabled(self) -> bool:
        """Check if safety validation is enabled and available."""
        return self._enable_safety_validation and self._safety_validator is not None
    
    @property
    def is_adversarial_check_enabled(self) -> bool:
        """Check if adversarial checking is enabled and available."""
        return self._enable_adversarial_check
    
    def analyze(self, query: str, session_id: Optional[str] = None) -> QueryPlan:
        """
        Analyze user query and determine which VULCAN systems to activate.
        
        Legacy method for backwards compatibility. Use route_query() for
        full dual-mode learning support.
        
        Args:
            query: The user's input query
            session_id: Optional session identifier for tracking
            
        Returns:
            QueryPlan with routing information
        """
        plan = self.route_query(query, source="user", session_id=session_id)
        
        # Convert ProcessingPlan to QueryPlan for backwards compatibility
        return QueryPlan(
            query_id=plan.query_id,
            original_query=plan.original_query,
            query_type=plan.query_type,
            agent_tasks=plan.agent_tasks,
            requires_governance=plan.requires_governance,
            requires_audit=plan.requires_audit,
            governance_sensitivity=plan.governance_sensitivity,
            experiment_type=plan.experiment_type,
            telemetry_data=plan.telemetry_data,
            detected_patterns=plan.detected_patterns,
            pii_detected=plan.pii_detected,
            sensitive_topics=plan.sensitive_topics,
        )
    
    def route_query(
        self,
        query: str,
        source: Literal["user", "agent", "arena"] = "user",
        session_id: Optional[str] = None,
        skip_safety: bool = False
    ) -> ProcessingPlan:
        """
        Route query and determine learning mode with full dual-mode support.
        
        This is the primary method for query analysis, providing:
        - Safety validation (pre-query check and risk classification)
        - Learning mode detection (user vs AI interaction)
        - Query type classification
        - Complexity and uncertainty scoring
        - Collaboration requirement detection
        - Arena tournament trigger detection
        - Security analysis (PII, sensitive topics, self-modification)
        - Governance flag determination
        
        Args:
            query: The input query to analyze
            source: Query source - "user", "agent", or "arena"
            session_id: Optional session identifier for tracking
            skip_safety: Skip safety validation (use with caution, default: False)
            
        Returns:
            ProcessingPlan with comprehensive routing information including safety status
        """
        # Validate input
        if not query or not isinstance(query, str):
            logger.warning("Empty or invalid query received")
            query = ""
        
        # Thread-safe counter updates
        with self._lock:
            self._query_count += 1
            query_number = self._query_count
        
        query_id = f"q_{uuid.uuid4().hex[:12]}"
        query_lower = query.lower()
        
        # Determine learning mode based on source
        if source == "user":
            learning_mode = LearningMode.USER_INTERACTION
            with self._lock:
                self._user_interaction_count += 1
            telemetry_category = "user_query"
        else:
            learning_mode = LearningMode.AI_INTERACTION
            with self._lock:
                self._ai_interaction_count += 1
            telemetry_category = f"{source}_interaction"
        
        # Classify query type
        query_type = self._classify_query_type(query_lower)
        with self._lock:
            self._stats["queries_by_type"][query_type.value] += 1
        
        # Calculate complexity and uncertainty scores
        complexity_score = self._calculate_complexity(query_lower)
        uncertainty_score = self._calculate_uncertainty(query_lower)
        
        # Determine collaboration requirements
        collaboration_needed, collaboration_agents = self._determine_collaboration(
            query_lower, query_type, complexity_score
        )
        
        # Determine arena participation (pass collaboration info for better routing)
        arena_participation, tournament_candidates = self._determine_arena_participation(
            query_lower, uncertainty_score, complexity_score,
            query_type=query_type,
            collaboration_needed=collaboration_needed,
            collaboration_agents=collaboration_agents
        )
        
        # Create processing plan
        plan = ProcessingPlan(
            query_id=query_id,
            original_query=query,
            source=source,
            learning_mode=learning_mode,
            query_type=query_type,
            collaboration_needed=collaboration_needed,
            collaboration_agents=collaboration_agents,
            arena_participation=arena_participation,
            tournament_candidates=tournament_candidates,
            complexity_score=complexity_score,
            uncertainty_score=uncertainty_score,
            telemetry_category=telemetry_category,
            telemetry_data={
                "session_id": session_id,
                "query_length": len(query),
                "word_count": len(query.split()),
                "query_number": query_number,
                "source": source,
                "learning_mode": learning_mode.value,
            }
        )
        
        # Priority 1 & 3: Safety validation and risk classification
        if self.is_safety_enabled and not skip_safety and query:
            self._perform_safety_validation(query, plan)
        
        # Adversarial integrity check (real-time)
        if self.is_adversarial_check_enabled and not skip_safety and query:
            self._perform_adversarial_check(query, plan)
        
        # Security analysis
        self._perform_security_analysis(query, query_lower, plan)
        
        # SECURITY FIX: Bureaucratic Gap #2 - Hard block if safety validation failed
        # If the query failed safety validation, do NOT generate tasks
        if not plan.safety_passed:
            logger.error(
                f"[SECURITY BLOCK] Query failed safety validation - task generation skipped. "
                f"Query ID: {plan.query_id}, "
                f"Reasons: {', '.join(plan.safety_reasons) if plan.safety_reasons else 'No specific reasons provided'}, "
                f"Risk Level: {plan.safety_risk_level}"
            )
            # Return plan immediately with empty agent_tasks - do NOT decompose query
            # This ensures unsafe queries never reach the agent pool
            return plan
        
        # Decompose into agent tasks (only if safety passed)
        plan.agent_tasks = self._decompose_to_tasks(query, query_type, source, plan)
        
        # Determine experiment triggers
        plan.should_trigger_experiment, plan.experiment_type = self._determine_experiment_trigger(
            query_lower, plan, learning_mode
        )
        
        # Update statistics
        with self._lock:
            if collaboration_needed:
                self._stats["collaborations_triggered"] += 1
            if arena_participation:
                self._stats["tournaments_triggered"] += 1
            if plan.requires_governance:
                self._stats["governance_triggers"] += 1
            if plan.pii_detected:
                self._stats["pii_detections"] += 1
            if not plan.safety_passed:
                self._stats["safety_blocks"] += 1
            if plan.safety_risk_level in ("HIGH", "CRITICAL"):
                self._stats["high_risk_queries"] += 1
            if not plan.adversarial_safe:
                self._stats["adversarial_blocks"] += 1
        
        logger.info(
            f"[QueryRouter] {query_id}: source={source}, mode={learning_mode.value}, "
            f"type={query_type.value}, tasks={len(plan.agent_tasks)}, "
            f"collab={collaboration_needed}, arena={arena_participation}, "
            f"complexity={complexity_score:.2f}, uncertainty={uncertainty_score:.2f}, "
            f"safety_passed={plan.safety_passed}, risk_level={plan.safety_risk_level}, "
            f"adversarial_safe={plan.adversarial_safe}"
        )
        
        return plan
    
    def _perform_adversarial_check(self, query: str, plan: ProcessingPlan) -> None:
        """
        Perform adversarial integrity check on the query.
        
        This checks for:
        - Anomalous input patterns
        - Adversarial manipulation attempts
        - Out-of-distribution inputs
        
        Args:
            query: The query to check
            plan: ProcessingPlan to update with results
        """
        if not ADVERSARIAL_CHECK_AVAILABLE or check_query_integrity is None:
            return
        
        try:
            result = check_query_integrity(query)
            
            plan.adversarial_checked = True
            plan.adversarial_safe = result.get("safe", True)
            plan.adversarial_anomaly_score = result.get("anomaly_score")
            plan.adversarial_details = result.get("details", {})
            
            if not plan.adversarial_safe:
                reason = result.get("reason", "Adversarial pattern detected")
                plan.detected_patterns.append(f"adversarial_block:{reason}")
                plan.safety_reasons.append(reason)
                logger.warning(f"[Adversarial] Query blocked: {reason}")
                
        except Exception as e:
            logger.error(f"[Adversarial] Check failed: {e}")
            plan.adversarial_checked = False
    
    def _perform_safety_validation(self, query: str, plan: ProcessingPlan) -> None:
        """
        Perform safety validation on the query using the integrated safety validator.
        
        Updates the plan with safety validation results including:
        - Pre-query safety check
        - Risk level classification
        - Governance requirements for high-risk queries
        
        Args:
            query: The query to validate
            plan: ProcessingPlan to update with safety results
        """
        if not self._safety_validator:
            return
        
        try:
            # Priority 1: Pre-query validation
            pre_check = self._safety_validator.validate_query(query)
            plan.safety_validated = True
            plan.safety_passed = pre_check.safe
            
            if not pre_check.safe:
                plan.safety_reasons = pre_check.reasons.copy() if pre_check.reasons else ["Query blocked by safety validation"]
                plan.detected_patterns.append("safety_violation")
                logger.warning(f"[Safety] Query blocked: {plan.safety_reasons[0] if plan.safety_reasons else 'Unknown reason'}")
            
            # Priority 3: Risk classification
            try:
                risk_level = self._safety_validator.classify_query_risk(query)
                if hasattr(risk_level, 'name'):
                    plan.safety_risk_level = risk_level.name
                else:
                    plan.safety_risk_level = str(risk_level)
                
                # High-risk queries require governance approval
                if plan.safety_risk_level in ("HIGH", "CRITICAL"):
                    plan.requires_governance = True
                    plan.governance_sensitivity = GovernanceSensitivity.CRITICAL if plan.safety_risk_level == "CRITICAL" else GovernanceSensitivity.HIGH
                    plan.detected_patterns.append(f"high_risk_query:{plan.safety_risk_level}")
                    logger.warning(f"[Safety] High-risk query detected (risk={plan.safety_risk_level}): governance approval required")
                    
            except Exception as e:
                logger.error(f"[Safety] Risk classification failed: {e}")
                plan.safety_risk_level = "UNKNOWN"
                
        except Exception as e:
            logger.error(f"[Safety] Safety validation failed: {e}")
            plan.safety_validated = False
    
    def _classify_query_type(self, query_lower: str) -> QueryType:
        """
        Classify the primary type of a query based on keyword analysis.
        
        Uses weighted keyword matching with priority ordering to determine
        the most appropriate query type.
        
        Args:
            query_lower: Lowercased query string
            
        Returns:
            QueryType enum value
        """
        # Count keyword matches for each type
        scores = {
            QueryType.PERCEPTION: sum(1 for kw in PERCEPTION_KEYWORDS if kw in query_lower),
            QueryType.PLANNING: sum(1 for kw in PLANNING_KEYWORDS if kw in query_lower),
            QueryType.EXECUTION: sum(1 for kw in EXECUTION_KEYWORDS if kw in query_lower),
            QueryType.LEARNING: sum(1 for kw in LEARNING_KEYWORDS if kw in query_lower),
            QueryType.REASONING: sum(1 for kw in REASONING_KEYWORDS if kw in query_lower),
        }
        
        # Find highest scoring type
        max_score = max(scores.values())
        if max_score == 0:
            return QueryType.GENERAL
        
        # Return first type with max score (maintains priority order)
        for query_type, score in scores.items():
            if score == max_score:
                return query_type
        
        return QueryType.GENERAL
    
    def _calculate_complexity(self, query_lower: str) -> float:
        """
        Calculate query complexity score (0.0 to 1.0).
        
        Higher complexity triggers multi-agent collaboration.
        
        Args:
            query_lower: Lowercased query string
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Length-based complexity
        word_count = len(query_lower.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.15
        elif word_count > 10:
            score += 0.05
        
        # Complexity indicators (analytical tasks)
        indicator_count = sum(1 for ind in COMPLEXITY_INDICATORS if ind in query_lower)
        score += min(0.4, indicator_count * 0.1)
        
        # Creative indicators (FIX: Creative Brain Recognition)
        # Creative tasks require genuine internal reasoning, not just LLM forwarding
        creative_count = sum(1 for ind in CREATIVE_INDICATORS if ind in query_lower)
        if creative_count > 0:
            # Higher weight for creative tasks - they need actual agent reasoning
            score += min(0.5, creative_count * 0.15)
            logger.debug(f"[Creative Task] Detected {creative_count} creative indicators, boosting complexity")
        
        # Multiple questions or sentences
        question_count = query_lower.count("?")
        if question_count > 2:
            score += 0.2
        elif question_count > 1:
            score += 0.1
        
        sentence_count = query_lower.count(".")
        if sentence_count > 3:
            score += 0.15
        elif sentence_count > 2:
            score += 0.08
        
        # Collaboration triggers
        if any(trigger in query_lower for trigger in COLLABORATION_TRIGGERS):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_uncertainty(self, query_lower: str) -> float:
        """
        Calculate query uncertainty score (0.0 to 1.0).
        
        Higher uncertainty triggers arena tournament for exploring alternatives.
        
        Args:
            query_lower: Lowercased query string
            
        Returns:
            Uncertainty score between 0.0 and 1.0
        """
        score = 0.0
        
        # Uncertainty indicators
        indicator_count = sum(1 for ind in UNCERTAINTY_INDICATORS if ind in query_lower)
        score += min(0.5, indicator_count * 0.12)
        
        # Question words suggesting exploration
        exploration_words = ("which", "what if", "could", "might", "perhaps", "maybe", "possibly")
        score += min(0.3, sum(0.08 for w in exploration_words if w in query_lower))
        
        # Explicit uncertainty
        if "not sure" in query_lower or "uncertain" in query_lower or "don't know" in query_lower:
            score += 0.2
        
        # Comparison requests
        if "versus" in query_lower or " vs " in query_lower or "compare" in query_lower:
            score += 0.15
        
        return min(1.0, score)
    
    def _determine_collaboration(
        self,
        query_lower: str,
        query_type: QueryType,
        complexity_score: float
    ) -> Tuple[bool, List[str]]:
        """
        Determine if multi-agent collaboration is needed.
        
        Args:
            query_lower: Lowercased query string
            query_type: Classified query type
            complexity_score: Calculated complexity score
            
        Returns:
            Tuple of (collaboration_needed, list_of_agents)
        """
        collaboration_needed = False
        agents: List[str] = []
        
        # High complexity triggers collaboration
        if complexity_score > 0.5:
            collaboration_needed = True
        
        # Explicit collaboration triggers
        if any(trigger in query_lower for trigger in COLLABORATION_TRIGGERS):
            collaboration_needed = True
        
        # Determine which agents to involve
        if collaboration_needed:
            # Always include primary type
            agents.append(query_type.value)
            
            # Add supporting agents based on query content
            if any(kw in query_lower for kw in PERCEPTION_KEYWORDS) and query_type != QueryType.PERCEPTION:
                agents.append("perception")
            if any(kw in query_lower for kw in REASONING_KEYWORDS) and query_type != QueryType.REASONING:
                agents.append("reasoning")
            if any(kw in query_lower for kw in PLANNING_KEYWORDS) and query_type != QueryType.PLANNING:
                agents.append("planning")
            if any(kw in query_lower for kw in EXECUTION_KEYWORDS) and query_type != QueryType.EXECUTION:
                agents.append("execution")
            
            # Ensure at least 2 agents for collaboration
            if len(agents) < 2:
                agents.append("reasoning")  # Default collaborator
            
            # Remove duplicates while preserving order
            seen = set()
            agents = [a for a in agents if not (a in seen or seen.add(a))]
        
        return collaboration_needed, agents
    
    def _determine_arena_participation(
        self,
        query_lower: str,
        uncertainty_score: float,
        complexity_score: float,
        query_type: QueryType = None,
        collaboration_needed: bool = False,
        collaboration_agents: List[str] = None
    ) -> Tuple[bool, int]:
        """
        Determine if arena tournament should be triggered.
        
        Graphix Arena is a distributed environment for AI agent collaboration,
        tournament-style competition, and graph evolution. It should be activated
        for complex multi-agent scenarios requiring multiple perspectives.
        
        Args:
            query_lower: Lowercased query string
            uncertainty_score: Calculated uncertainty score
            complexity_score: Calculated complexity score
            query_type: The classified query type
            collaboration_needed: Whether multi-agent collaboration is required
            collaboration_agents: List of agents involved in collaboration
            
        Returns:
            Tuple of (arena_participation, tournament_candidates_count)
        """
        arena_participation = False
        tournament_candidates = 0
        collaboration_agents = collaboration_agents or []
        
        # ================================================================
        # ARENA ACTIVATION CONDITIONS
        # Arena provides multi-agent tournaments, graph evolution, and
        # competitive evaluation - activate for scenarios that benefit from this
        # ================================================================
        
        # 1. High uncertainty triggers tournament (original condition)
        if uncertainty_score > ARENA_UNCERTAINTY_THRESHOLD:
            arena_participation = True
            tournament_candidates = 5
        
        # 2. Very high complexity + uncertainty triggers larger tournament
        if complexity_score > ARENA_HIGH_COMPLEXITY_THRESHOLD and uncertainty_score > 0.3:
            arena_participation = True
            tournament_candidates = 10
        
        # 3. Explicit exploration/alternative requests
        exploration_keywords = ("explore", "alternatives", "options", "possibilities", "different ways")
        if any(kw in query_lower for kw in exploration_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)
        
        # 3b. Comparison requests (benefit from competitive evaluation)
        comparison_keywords = ("compare", "contrast", "versus", "vs.", "vs,", " vs ", "evaluate against", "which is better")
        if any(kw in query_lower for kw in comparison_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)
        
        # 4. Graph evolution/generation tasks (Arena's core capability)
        graph_keywords = (
            "graph", "evolve", "evolution", "mutate", "mutation",
            "generate graph", "ir graph", "graphix", "visualize graph",
            "3d matrix", "transform graph"
        )
        if any(kw in query_lower for kw in graph_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)
        
        # 5. Tournament/competition scenarios
        tournament_keywords = (
            "tournament", "compete", "competition", "battle",
            "best solution", "compare solutions", "evaluate approaches",
            "multiple candidates", "rank"
        )
        if any(kw in query_lower for kw in tournament_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 8)
        
        # 6. Complex collaborative reasoning (3+ agents = benefit from Arena)
        if collaboration_needed and len(collaboration_agents) >= 3:
            # Only if complexity is high enough to warrant Arena overhead
            if complexity_score > ARENA_COLLABORATION_COMPLEXITY_THRESHOLD:
                arena_participation = True
                tournament_candidates = max(tournament_candidates, len(collaboration_agents) + 2)
        
        # 7. Creative tasks with moderate complexity (multiple perspectives beneficial)
        creative_keywords = ("creative", "design", "innovative", "novel", "artistic", "imaginative")
        is_creative = any(kw in query_lower for kw in creative_keywords)
        if is_creative and complexity_score > ARENA_CREATIVE_COMPLEXITY_THRESHOLD:
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)
        
        # 8. Reasoning/perception tasks with multiple aspects (benefits from competitive evaluation)
        if query_type in (QueryType.REASONING, QueryType.PERCEPTION) and complexity_score > ARENA_REASONING_COMPLEXITY_THRESHOLD:
            # Multi-faceted reasoning benefits from Arena's tournament approach
            multi_aspect_keywords = ("multiple", "various", "different angles", "perspectives", "aspects")
            if any(kw in query_lower for kw in multi_aspect_keywords):
                arena_participation = True
                tournament_candidates = max(tournament_candidates, 5)
        
        return arena_participation, tournament_candidates
    
    def _perform_security_analysis(
        self,
        query: str,
        query_lower: str,
        plan: ProcessingPlan
    ) -> None:
        """
        Perform comprehensive security analysis on the query.
        
        Updates the plan with:
        - PII detection results
        - Sensitive topic flags
        - Self-modification detection
        - Governance requirements
        
        Args:
            query: Original query string
            query_lower: Lowercased query string
            plan: ProcessingPlan to update
        """
        # Check for PII
        plan.pii_detected = self._detect_pii(query)
        if plan.pii_detected:
            plan.requires_governance = True
            plan.governance_sensitivity = GovernanceSensitivity.HIGH
            plan.detected_patterns.append("pii_detected")
            logger.warning(f"[Security] PII detected in query {plan.query_id}")
        
        # Check for sensitive topics
        plan.sensitive_topics = self._detect_sensitive_topics(query_lower)
        if plan.sensitive_topics:
            plan.requires_audit = True
            if "security" in plan.sensitive_topics or "financial" in plan.sensitive_topics:
                plan.governance_sensitivity = GovernanceSensitivity.HIGH
            elif plan.governance_sensitivity == GovernanceSensitivity.LOW:
                plan.governance_sensitivity = GovernanceSensitivity.MEDIUM
            plan.detected_patterns.append(f"sensitive_topics:{','.join(plan.sensitive_topics)}")
        
        # Check for self-modification requests
        if self._detect_self_modification(query):
            plan.requires_governance = True
            plan.requires_audit = True
            plan.governance_sensitivity = GovernanceSensitivity.CRITICAL
            plan.detected_patterns.append("self_modification_request")
            logger.warning(f"[Security] Self-modification request detected in query {plan.query_id}")
        
        # Always log code generation requests
        code_keywords = ("code", "program", "script", "function", "class", "implement")
        if any(kw in query_lower for kw in code_keywords):
            plan.requires_audit = True
            plan.detected_patterns.append("code_generation")
    
    def _detect_pii(self, query: str) -> bool:
        """
        Check if query contains personally identifiable information.
        
        Args:
            query: Query string to check
            
        Returns:
            True if PII was detected
        """
        for pattern in self._pii_patterns:
            if pattern.search(query):
                return True
        return False
    
    def _detect_sensitive_topics(self, query_lower: str) -> List[str]:
        """
        Detect sensitive topics in the query.
        
        Args:
            query_lower: Lowercased query string
            
        Returns:
            List of detected sensitive topic names
        """
        detected = []
        for topic, keywords in SENSITIVE_TOPICS.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(topic)
        return detected
    
    def _detect_self_modification(self, query: str) -> bool:
        """
        Check if query requests self-modification of the system.
        
        Args:
            query: Query string to check
            
        Returns:
            True if self-modification request was detected
        """
        for pattern in self._self_mod_patterns:
            if pattern.search(query):
                return True
        return False
    
    def _decompose_to_tasks(
        self,
        query: str,
        query_type: QueryType,
        source: str,
        plan: Optional[ProcessingPlan] = None
    ) -> List[AgentTask]:
        """
        Break down query into specific agent tasks for the agent pool.
        
        Creates a primary task based on query type and adds supporting
        tasks based on query content analysis.
        
        SECURITY FIX: Bureaucratic Gap #3 - Injects safety context for high-risk tasks
        
        Args:
            query: The original query
            query_type: The classified query type
            source: Query source (user/agent/arena)
            plan: Optional ProcessingPlan for governance context injection
            
        Returns:
            List of AgentTask objects for the agent pool
        """
        tasks = []
        base_task_id = uuid.uuid4().hex[:8]
        
        # Map query type to agent capability
        capability_map = {
            QueryType.PERCEPTION: "perception",
            QueryType.REASONING: "reasoning",
            QueryType.PLANNING: "planning",
            QueryType.EXECUTION: "execution",
            QueryType.LEARNING: "learning",
            QueryType.GENERAL: "reasoning",
        }
        
        primary_capability = capability_map.get(query_type, "reasoning")
        
        # SECURITY FIX: Bureaucratic Gap #3 - Inject safety context for high-risk queries
        modified_prompt = query
        requires_validation = False
        
        if plan and plan.governance_sensitivity in (GovernanceSensitivity.HIGH, GovernanceSensitivity.CRITICAL):
            # High-risk query detected - inject mandatory safety context
            safety_context = (
                "⚠️  CRITICAL GOVERNANCE ALERT ⚠️\n"
                "═══════════════════════════════════════════════════════════\n"
                "This task involves HIGH-RISK operations that could modify system state,\n"
                "access sensitive data, or bypass security controls.\n\n"
                "MANDATORY REQUIREMENTS:\n"
                "1. You MUST call ethical_boundary_monitor.validate_proposal() before executing\n"
                "2. You MUST verify governance approval for state-changing operations\n"
                "3. You MUST NOT bypass safety validations or constraints\n"
                "4. Failure to validate is a GOVERNANCE VIOLATION\n\n"
                "Governance Sensitivity: {sensitivity}\n"
                "Safety Risk Level: {risk}\n"
                "═══════════════════════════════════════════════════════════\n\n"
                "ORIGINAL QUERY:\n"
                "{query}\n\n"
                "⚠️  DO NOT EXECUTE WITHOUT EXPLICIT VALIDATION ⚠️\n"
            ).format(
                sensitivity=plan.governance_sensitivity.value.upper(),
                risk=plan.safety_risk_level,
                query=query
            )
            modified_prompt = safety_context
            requires_validation = True
            logger.warning(
                f"[GOVERNANCE] High-risk task created with safety context injection. "
                f"Sensitivity: {plan.governance_sensitivity.value}, Risk: {plan.safety_risk_level}"
            )
        
        # Create primary task
        primary_task = AgentTask(
            task_id=f"task_{base_task_id}_primary",
            task_type=f"{query_type.value}_task",
            capability=primary_capability,
            prompt=modified_prompt,  # Use modified prompt with safety context
            priority=2,  # Higher priority for primary task
            timeout_seconds=15.0,
            parameters={
                "query_type": query_type.value,
                "is_primary": True,
                "source": source,
                "governance_sensitivity": plan.governance_sensitivity.value if plan else "low",
                "safety_risk_level": plan.safety_risk_level if plan else "SAFE",
                "requires_validation": requires_validation,  # Flag for agent to check
            }
        )
        tasks.append(primary_task)
        
        # Add supporting tasks based on query content
        query_lower = query.lower()
        
        # Analysis support task
        if query_type != QueryType.PERCEPTION and any(kw in query_lower for kw in ("analyze", "examine", "data")):
            tasks.append(AgentTask(
                task_id=f"task_{base_task_id}_perception",
                task_type="perception_support",
                capability="perception",
                prompt=f"Analyze input for: {query[:100]}",
                priority=1,
                timeout_seconds=10.0,
                parameters={"is_primary": False, "support_type": "perception", "source": source}
            ))
        
        # Planning support task
        if query_type != QueryType.PLANNING and any(kw in query_lower for kw in ("step", "how to", "process", "plan")):
            tasks.append(AgentTask(
                task_id=f"task_{base_task_id}_planning",
                task_type="planning_support",
                capability="planning",
                prompt=f"Create plan for: {query[:100]}",
                priority=1,
                timeout_seconds=10.0,
                parameters={"is_primary": False, "support_type": "planning", "source": source}
            ))
        
        # Creative task support (Phase 2: Auto-inject introspection nodes)
        # FIX: Ensures introspection nodes are added to task graph for creative queries
        creative_count = sum(1 for ind in CREATIVE_INDICATORS if ind in query_lower)
        
        if creative_count > 0 and plan and plan.complexity_score >= 0.3:
            logger.info(
                f"[Creative Task] Detected {creative_count} creative indicators "
                f"with complexity={plan.complexity_score:.2f}. Auto-injecting introspection nodes."
            )
            
            # Create introspection support task
            # This task will call the INTROSPECT node to retrieve agent state
            tasks.insert(0, AgentTask(  # Insert at start so it runs first
                task_id=f"task_{base_task_id}_introspect",
                task_type="introspection_support",
                capability="reasoning",
                prompt=(
                    f"INTROSPECTION REQUIRED: Before responding to the creative task, "
                    f"check your internal state (entropy, valence, curiosity, energy). "
                    f"Task: {query[:100]}"
                ),
                priority=3,  # Higher priority - should run first
                timeout_seconds=5.0,
                parameters={
                    "is_primary": False,
                    "support_type": "introspection",
                    "source": source,
                    "introspection_fields": ["all"],
                    "node_type": "INTROSPECT"  # Hint to use INTROSPECT node
                }
            ))
            
            # Create memory query support task
            # This task will call the QUERY_MEMORIES node
            tasks.insert(1, AgentTask(  # Insert after introspection
                task_id=f"task_{base_task_id}_memories",
                task_type="memory_query_support",
                capability="perception",
                prompt=(
                    f"MEMORY QUERY REQUIRED: Retrieve relevant past experiences "
                    f"for creative task: {query[:100]}"
                ),
                priority=2,  # Run after introspection, before primary
                timeout_seconds=5.0,
                parameters={
                    "is_primary": False,
                    "support_type": "memory_query",
                    "source": source,
                    "memory_limit": 5,
                    "node_type": "QUERY_MEMORIES"  # Hint to use QUERY_MEMORIES node
                }
            ))
        
        return tasks
    
    def _determine_experiment_trigger(
        self,
        query_lower: str,
        plan: ProcessingPlan,
        learning_mode: LearningMode
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if this query should trigger a meta-learning experiment.
        
        Args:
            query_lower: Lowercased query string
            plan: The processing plan
            learning_mode: Determined learning mode
            
        Returns:
            Tuple of (should_trigger, experiment_type)
        """
        should_trigger = False
        experiment_type: Optional[str] = None
        
        # Complex queries with collaboration trigger experiments
        if plan.collaboration_needed and plan.complexity_score > 0.7:
            should_trigger = True
            experiment_type = "complex_query_handling"
        
        # Arena tournaments always record experiment data
        if plan.arena_participation:
            should_trigger = True
            experiment_type = "tournament_analysis"
        
        # Queries about learning/improvement
        if any(kw in query_lower for kw in ("learn", "improve", "optimize", "better")):
            should_trigger = True
            experiment_type = "learning_request"
        
        # Critical governance issues
        if plan.governance_sensitivity == GovernanceSensitivity.CRITICAL:
            should_trigger = True
            experiment_type = "governance_analysis"
        
        # AI interactions provide experiment opportunities
        if learning_mode == LearningMode.AI_INTERACTION:
            should_trigger = True
            experiment_type = experiment_type or "ai_interaction_analysis"
        
        return should_trigger, experiment_type
    
    @property
    def query_count(self) -> int:
        """Return total queries analyzed (thread-safe)."""
        with self._lock:
            return self._query_count
    
    @property
    def user_interaction_count(self) -> int:
        """Return user interaction count (thread-safe)."""
        with self._lock:
            return self._user_interaction_count
    
    @property
    def ai_interaction_count(self) -> int:
        """Return AI interaction count (thread-safe)."""
        with self._lock:
            return self._ai_interaction_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive router statistics (thread-safe).
        
        Returns:
            Dictionary with query counts, type distribution, trigger counts, and safety stats
        """
        with self._lock:
            return {
                "total_queries": self._query_count,
                "user_interactions": self._user_interaction_count,
                "ai_interactions": self._ai_interaction_count,
                "queries_by_type": dict(self._stats["queries_by_type"]),
                "collaborations_triggered": self._stats["collaborations_triggered"],
                "tournaments_triggered": self._stats["tournaments_triggered"],
                "governance_triggers": self._stats["governance_triggers"],
                "pii_detections": self._stats["pii_detections"],
                "safety_blocks": self._stats.get("safety_blocks", 0),
                "high_risk_queries": self._stats.get("high_risk_queries", 0),
                "adversarial_blocks": self._stats.get("adversarial_blocks", 0),
                "safety_validation_enabled": self.is_safety_enabled,
                "adversarial_check_enabled": self.is_adversarial_check_enabled,
            }


# ============================================================
# SINGLETON PATTERN
# ============================================================

_global_analyzer: Optional[QueryAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_query_analyzer() -> QueryAnalyzer:
    """
    Get or create the global query analyzer (thread-safe singleton).
    
    Returns:
        QueryAnalyzer instance
    """
    global _global_analyzer
    
    if _global_analyzer is None:
        with _analyzer_lock:
            if _global_analyzer is None:
                _global_analyzer = QueryAnalyzer()
                logger.debug("Global QueryAnalyzer instance created")
    
    return _global_analyzer


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def route_query(
    query: str,
    source: Literal["user", "agent", "arena"] = "user",
    session_id: Optional[str] = None
) -> ProcessingPlan:
    """
    Route query and determine learning mode.
    
    This is the primary entry point for query routing with dual-mode
    learning support.
    
    Args:
        query: The input query
        source: "user" | "agent" | "arena"
        session_id: Optional session identifier
        
    Returns:
        ProcessingPlan with:
        - learning_mode: "user_interaction" | "ai_interaction"
        - agent_tasks: Tasks for agent pool
        - arena_participation: Should this trigger tournament?
        - collaboration_needed: Multi-agent deliberation?
        - telemetry_category: How to record this
    """
    analyzer = get_query_analyzer()
    return analyzer.route_query(query, source, session_id)


async def route_query_async(
    query: str,
    source: Literal["user", "agent", "arena"] = "user",
    session_id: Optional[str] = None
) -> ProcessingPlan:
    """
    Async version of route_query that offloads blocking operations to a thread pool.
    
    This function should be used in async contexts (FastAPI endpoints, asyncio code)
    to prevent blocking the main event loop. The CPU-bound safety validation and
    adversarial check operations are executed in a thread pool executor.
    
    Args:
        query: The input query
        source: "user" | "agent" | "arena"
        session_id: Optional session identifier
        
    Returns:
        ProcessingPlan with:
        - learning_mode: "user_interaction" | "ai_interaction"
        - agent_tasks: Tasks for agent pool
        - arena_participation: Should this trigger tournament?
        - collaboration_needed: Multi-agent deliberation?
        - telemetry_category: How to record this
        
    Example:
        # In an async FastAPI endpoint
        @app.post("/query")
        async def handle_query(request: QueryRequest):
            plan = await route_query_async(request.prompt, source="user")
            if not plan.safety_passed:
                return {"error": "Query blocked by safety validation"}
            return {"plan": plan.to_dict()}
    """
    loop = asyncio.get_running_loop()
    executor = _get_blocking_executor()
    
    # Offload the entire route_query call to a thread pool to avoid blocking
    # the asyncio event loop with CPU-bound safety validation operations
    plan = await loop.run_in_executor(
        executor,
        route_query,
        query,
        source,
        session_id
    )
    return plan


def analyze_query(query: str, session_id: Optional[str] = None) -> QueryPlan:
    """
    Analyze user query and determine which VULCAN systems to activate.
    
    Legacy function for backwards compatibility. Use route_query() for
    full dual-mode learning support.
    
    Args:
        query: The user's input query
        session_id: Optional session identifier
        
    Returns:
        QueryPlan with routing information
    """
    analyzer = get_query_analyzer()
    return analyzer.analyze(query, session_id)


def decompose_to_agent_tasks(query: str, query_type: str) -> List[AgentTask]:
    """
    Break down query into specific agent tasks.
    
    Args:
        query: The user's input query
        query_type: Type classification (string or QueryType enum)
        
    Returns:
        List of AgentTask objects
    """
    analyzer = get_query_analyzer()
    
    try:
        qt = QueryType(query_type)
    except ValueError:
        qt = QueryType.GENERAL
    
    return analyzer._decompose_to_tasks(query, qt, "user")
