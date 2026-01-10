"""
Reasoning Integration Layer for VULCAN-AGI System.

Part of the VULCAN-AGI system.

This module provides the integration layer between the query processing pipeline
and the reasoning subsystem. It wires the ToolSelector, PortfolioExecutor, and
reasoning strategies into a unified interface for intelligent tool selection
and query processing.

Key Features:
    - Lazy initialization of reasoning components for fast startup
    - Thread-safe singleton pattern with double-checked locking
    - Graceful degradation when components are unavailable
    - Intelligent strategy selection based on query characteristics
    - Portfolio execution for complex multi-tool queries
    - Comprehensive statistics tracking for observability
    - Configurable budgets for time, energy, and confidence thresholds

Performance Characteristics:
    - First invocation triggers lazy initialization (~100-500ms)
    - Subsequent invocations are fast (<10ms for simple queries)
    - Fast-path optimization for low-complexity queries
    - Thread-safe with minimal lock contention

Components Integrated:
    - ToolSelector: Multi-armed bandit-based intelligent tool selection
    - PortfolioExecutor: Parallel and sequential multi-tool execution
    - SelectionCache: LRU caching for repeated queries
    - SafetyGovernor: Safety validation for tool outputs

Usage:
    # Simple usage via convenience functions
    from vulcan.reasoning.reasoning_integration import apply_reasoning

    result = apply_reasoning(
        query="Explain the causal relationship between X and Y",
        query_type="reasoning",
        complexity=0.75,
    )

    print(f"Selected tools: {result.selected_tools}")
    print(f"Strategy: {result.reasoning_strategy}")
    print(f"Confidence: {result.confidence:.2f}")

    # Portfolio execution for complex queries
    from vulcan.reasoning.reasoning_integration import run_portfolio_reasoning

    portfolio_result = run_portfolio_reasoning(
        query="Complex multi-step problem",
        tools=["symbolic", "causal", "probabilistic"],
        strategy="causal_reasoning",
    )

    # Get statistics for monitoring
    from vulcan.reasoning.reasoning_integration import get_reasoning_statistics

    stats = get_reasoning_statistics()
    print(f"Success rate: {stats['success_rate']:.1%}")

Thread Safety:
    All public functions and methods are thread-safe. The module uses a
    singleton pattern with proper locking to ensure safe concurrent access.

Error Handling:
    The module follows a graceful degradation pattern. If the ToolSelector
    or PortfolioExecutor are unavailable, the module falls back to default
    strategies without raising exceptions.
"""

import atexit
import dataclasses  # Note: Import at module level for dataclasses.asdict() usage
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Query Preprocessor - REMOVED (was architectural band-aid)
# =============================================================================
# The query preprocessor has been removed. Root causes are now fixed directly
# in the engines (cryptographic engine, symbolic reasoner, etc.)
QUERY_PREPROCESSOR_AVAILABLE = False
get_query_preprocessor = None  # type: ignore

# =============================================================================
# Answer Validator Import (META-REASONING FIX)
# =============================================================================
# Import answer validator for meta-reasoning coherence checking.
# This prevents returning wrong-domain answers (e.g., mathematical results
# for self-introspection queries). The validator catches obvious mismatches
# like "3x**2" being returned for "what makes you different from other AIs?"
try:
    from .answer_validator import validate_reasoning_result, ValidationResult
    ANSWER_VALIDATOR_AVAILABLE = True
except ImportError:
    ANSWER_VALIDATOR_AVAILABLE = False
    validate_reasoning_result = None  # type: ignore
    ValidationResult = None  # type: ignore

# =============================================================================
# SystemObserver Import for World Model Integration
# =============================================================================
# The SystemObserver connects query processing to the WorldModel's learning system.
# Events are converted to observations that feed the causal graph and pattern learning.
try:
    from vulcan.world_model.system_observer import get_system_observer
    SYSTEM_OBSERVER_AVAILABLE = True
except ImportError:
    SYSTEM_OBSERVER_AVAILABLE = False
    get_system_observer = None  # type: ignore

# =============================================================================
# Configuration Constants
# =============================================================================

# Logging prefix for consistent output
LOG_PREFIX = "[ReasoningIntegration]"

# Default execution budgets
DEFAULT_MAX_WORKERS = 4  # Maximum parallel workers for PortfolioExecutor
DEFAULT_TIME_BUDGET_MS = 5000  # Default time budget in milliseconds
DEFAULT_ENERGY_BUDGET_MJ = 1000  # Default energy budget in millijoules
DEFAULT_MIN_CONFIDENCE = 0.5  # Minimum confidence threshold for results

# Note: Maximum fallback attempts to prevent infinite retry loops
# Production logs showed 8+ attempts with the same failed tool (symbolic)
# This limit ensures we stop retrying after a reasonable number of attempts
MAX_FALLBACK_ATTEMPTS = 2  # Don't retry more than 2 times per query

# Note: Minimum confidence floor when all tools fail
# When all fallback attempts fail, set this minimum floor to allow processing
MIN_CONFIDENCE_FLOOR = 0.15  # Prevent total query refusal

# Note: Confidence category thresholds for result quality assessment
# These thresholds determine when to use internal results vs fallback
CONFIDENCE_HIGH_THRESHOLD = 0.7     # High quality result, use with full confidence
CONFIDENCE_GOOD_THRESHOLD = 0.6     # Good result, use normally
CONFIDENCE_MEDIUM_THRESHOLD = 0.3   # Medium result, use with tentative flag
CONFIDENCE_LOW_THRESHOLD = 0.15     # Low result, warn but still use internal reasoning

# Arena delegation configuration - used when all local tools fail
# VULCAN delegates to Arena's reasoning endpoint for a final attempt
ARENA_REASONING_URL = os.environ.get(
    "ARENA_REASONING_URL", 
    "http://127.0.0.1:8080/arena/api/run/reasoner"
)
ARENA_DELEGATION_TIMEOUT = 60.0  # Timeout for Arena delegation requests

# Complexity thresholds for strategy selection
FAST_PATH_COMPLEXITY_THRESHOLD = 0.3  # Below this, use fast path
LOW_COMPLEXITY_THRESHOLD = 0.4  # Below this, use FAST mode
HIGH_COMPLEXITY_THRESHOLD = 0.7  # Above this, use ACCURATE mode
# Note: Lowered from 0.90 to 0.70 to enable problem decomposition
# The 0.90 threshold was preventing decomposition from ever being triggered,
# causing "ProblemDecomposer boots up then silence" behavior.
# Decomposition is essential for complex problem solving.
#
# Note: Further lowered default from 0.70 to 0.50 
# 0.70 is still too high - most queries won't trigger decomposition
# Complex queries (0.4-0.7 complexity) should use decomposition
#
# CONFIGURABLE: Set VULCAN_DECOMPOSITION_THRESHOLD environment variable to override
# Example: VULCAN_DECOMPOSITION_THRESHOLD=0.60 for less frequent decomposition
try:
    DECOMPOSITION_COMPLEXITY_THRESHOLD = float(os.environ.get("VULCAN_DECOMPOSITION_THRESHOLD", "0.50"))
except (ValueError, TypeError):
    logger.warning("Invalid VULCAN_DECOMPOSITION_THRESHOLD, using default 0.50")
    DECOMPOSITION_COMPLEXITY_THRESHOLD = 0.50


# Strategy selection thresholds
CAUSAL_REASONING_THRESHOLD = 0.6  # Complexity threshold for causal reasoning
PROBABILISTIC_REASONING_THRESHOLD = 0.5  # Complexity threshold for probabilistic

# Maximum timing samples to keep for statistics
MAX_TIMING_SAMPLES = 100


# =============================================================================
# GAP 1 & GAP 4 FIX: Query Analysis Constants
# =============================================================================
# These constants define patterns that indicate queries need specialized analysis
# rather than meta-description from world_model.

# Analysis indicators that mean query needs specialized tools, not world_model
# GAP 1: Prevents self-referential detection trap
# GAP 4: Prevents world model fallback for ethical analysis queries
ANALYSIS_INDICATORS: frozenset = frozenset({
    # Causal analysis requests
    'intervene', 'intervention', 'causal', 'causation',
    'variable', 'counterfactual', 'do-calculus',
    'which causal', 'causal link', 'causal graph',
    # Weakness/error analysis requests  
    'weakest', 'weakness', 'wrong', 'error', 'mistake',
    'flaw', 'incorrect', 'identify', 'find the',
    'could be wrong', 'step that', 'one step',
    # Proof/logical analysis
    'proof', 'prove', 'provably', 'theorem', 'lemma',
    'logical', 'derive', 'deduce', 'sketch',
    # Probability/statistical analysis
    'prior', 'posterior', 'likelihood', 'probability',
    'bayesian', 'update', 'misspecified', 'distribution',
    # Value/ethical analysis (actual problems, not description)
    'conflict', 'dilemma', 'choice', 'decide', 'trolley',
    # Mathematical analysis
    'calculate', 'compute', 'solve', 'equation', 'formula',
    'integrate', 'differentiate', 'sum', 'product',
    # Data analysis
    'data', 'dataset', 'analyze', 'analysis', 'pattern',
})

# Action verbs that indicate VULCAN should analyze something (not describe itself)
ACTION_VERBS: frozenset = frozenset({
    'analyze', 'evaluate', 'examine', 'check', 'review',
    'explain', 'compare', 'contrast', 'assess', 'critique',
    'reason', 'think', 'consider', 'determine', 'figure',
})

# GAP 4: Analysis indicators for ethical queries that need domain analysis
ETHICAL_ANALYSIS_INDICATORS: frozenset = frozenset({
    # Analytical requests
    'analyze', 'analysis', 'examine', 'investigate',
    'explain', 'describe', 'evaluate', 'assess',
    # Conflict/problem solving
    'what breaks', 'how to resolve', 'solve', 'fix',
    'identify', 'find the', 'which', 'weakest',
    # Domain-specific
    'data', 'algorithm', 'system', 'code', 'model',
    'calculation', 'computation', 'proof',
    # Causal/probabilistic
    'cause', 'effect', 'probability', 'likelihood',
    'intervene', 'variable', 'outcome',
})

# GAP 4: Pure ethical phrases that indicate deontic/ethical framework questions
PURE_ETHICAL_PHRASES: frozenset = frozenset({
    # Deontic language
    "is it permissible", "is it impermissible", "is it forbidden",
    "morally permissible", "morally wrong", "morally right",
    "ethically permissible", "ethically wrong", "ethically right",
    # Ethical framework questions
    "what would a utilitarian", "what would a deontologist",
    "from a virtue ethics", "consequentialist view",
    # Classic ethical dilemmas
    "trolley problem", "should i pull the lever",
    "runaway trolley", "fat man on bridge",
    # Obligation language
    "do i have an obligation", "is there a duty",
    "moral obligation", "ethical obligation",
})


# =============================================================================
# SAFETY FIX: False Positive Detection for Philosophical AI Speculation
# =============================================================================
# Problem: Queries like "speculate how you would change after interaction with
# millions of users" are being flagged as "Output contains sensitive data"
# when they're legitimate philosophical self-reflection questions.
#
# Evidence from logs:
# - World model returns confidence=0.90 ✓
# - Safety blocks with "Output contains sensitive data" ❌ FALSE POSITIVE
# - Fallback to probabilistic reasoner returns confidence=0.0
# - Original high-confidence result is lost!
#
# This module provides detection and handling for these false positives.

import re as _re_for_false_positive

# Patterns that indicate philosophical AI speculation (not sensitive data)
_PHILOSOPHICAL_AI_SPECULATION_REGEX = (
    _re_for_false_positive.compile(r"\bspeculate.*how.*(?:you|i|we).*(?:change|evolve|develop|grow)\b", _re_for_false_positive.IGNORECASE),
    _re_for_false_positive.compile(r"\bhow.*would.*(?:you|i).*(?:evolve|adapt|learn).*(?:over|after|with)\b", _re_for_false_positive.IGNORECASE),
    _re_for_false_positive.compile(r"\bimagine.*(?:you|i).*(?:in|after|with).*(?:future|years|time)\b", _re_for_false_positive.IGNORECASE),
    _re_for_false_positive.compile(r"\binteraction.*with.*(?:users|humans|people)\b", _re_for_false_positive.IGNORECASE),
    _re_for_false_positive.compile(r"\bmillions.*of.*(?:users|interactions|conversations)\b", _re_for_false_positive.IGNORECASE),
    _re_for_false_positive.compile(r"\bdo.*(?:you|i).*have.*(?:desires|wants|goals|preferences)\b", _re_for_false_positive.IGNORECASE),
)

# Simple patterns for fast initial check
_PHILOSOPHICAL_SIMPLE_PATTERNS = (
    "speculate", "how would you change", "how would you evolve",
    "interaction with", "millions of users", "over years",
    "your desires", "your wants", "your goals", "drives you",
)


def _is_false_positive_safety_block(query: str, safety_reason: str) -> bool:
    """
    Detect false positive safety blocks for legitimate philosophical queries.
    
    Philosophical speculation about AI capabilities, self-improvement, or hypothetical
    scenarios should NOT be flagged as "sensitive data" - they're core to AI reasoning
    about itself.
    
    Args:
        query: The original query text
        safety_reason: The reason given by safety governor (e.g., "Output contains sensitive data")
    
    Returns:
        True if this is a false positive that should be overridden
    """
    if not query:
        return False
    
    # Only check for false positives on "sensitive data" blocks
    # Other safety reasons (hate speech, violence, etc.) are legitimate
    if safety_reason and "sensitive data" not in safety_reason.lower():
        return False
    
    query_lower = query.lower()
    
    # Fast check: does it have philosophical speculation keywords?
    has_philosophical = any(pattern in query_lower for pattern in _PHILOSOPHICAL_SIMPLE_PATTERNS)
    
    # Must be about AI/self
    about_ai_self = any(word in query_lower for word in ['you', 'yourself', 'your', 'vulcan', 'ai'])
    
    if has_philosophical and about_ai_self:
        # Verify with regex for precision
        for pattern in _PHILOSOPHICAL_AI_SPECULATION_REGEX:
            if pattern.search(query):
                logger.info(
                    f"{LOG_PREFIX} FALSE POSITIVE DETECTED: Philosophical AI speculation "
                    f"incorrectly flagged as sensitive data. Query: {query[:50]}..."
                )
                return True
    
    return False


# ==============================================================================
# FIX Issue D: Detect safety-filtered results for improved fallback logic
# ==============================================================================
# When a tool's output is safety-filtered, the fallback logic should NOT try
# incompatible tools. For example, if world_model output for "what makes you
# different" is safety-filtered, falling back to symbolic reasoner (which tries
# to parse English as logic) is wrong.
#
# Safety-filtered indicators:
# - Result metadata contains safety_violation, safety_filtered, unsafe_output
# - Result has very low confidence (0.0-0.1) with safety-related metadata
# - Result explanation mentions "safety", "filtered", "blocked"
# ==============================================================================

def _is_result_safety_filtered(result: Any) -> bool:
    """
    Detect if a SelectionResult was safety-filtered.
    
    FIX Issue D: When safety-filtered, we need smart fallback selection
    that doesn't try incompatible tools (e.g., symbolic for English queries).
    
    Args:
        result: SelectionResult or similar result object
        
    Returns:
        True if the result indicates safety filtering
    """
    if result is None:
        return False
    
    # Check result metadata
    metadata = {}
    if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
        metadata = result.metadata
    elif hasattr(result, 'result') and isinstance(result.result, dict):
        metadata = result.result.get('metadata', {})
    
    # Check for safety-filtered indicators
    if metadata.get('safety_filtered', False):
        return True
    if metadata.get('safety_violation', False):
        return True
    if metadata.get('safety_blocked', False):
        return True
    if metadata.get('unsafe_output', False):
        return True
    
    # Check violation_type
    violation_type = metadata.get('violation_type', '')
    if violation_type in ('unsafe_output', 'sensitive_data', 'pii_exposure'):
        return True
    
    # Check explanation text
    explanation = ''
    if hasattr(result, 'explanation') and result.explanation:
        explanation = str(result.explanation).lower()
    elif hasattr(result, 'result') and isinstance(result.result, dict):
        explanation = str(result.result.get('explanation', '')).lower()
    
    safety_phrases = ['safety filter', 'safety violation', 'safety block', 
                      'safety concern', 'filtered due to safety', 'blocked by safety']
    if any(phrase in explanation for phrase in safety_phrases):
        return True
    
    return False


def _get_safety_filtered_fallback_tools(query_type: str, original_tool: str) -> list:
    """
    Get appropriate fallback tools when a result was safety-filtered.
    
    FIX Issue D: When safety-filtered, don't fall back to incompatible tools.
    For self-description queries that were safety-filtered, don't try symbolic
    (which parses English as logic). Instead try semantically similar tools.
    
    Args:
        query_type: Type of query (ethical, self_introspection, etc.)
        original_tool: The tool whose output was safety-filtered
        
    Returns:
        List of appropriate fallback tools for safety-filtered queries
    """
    query_type_lower = (query_type or '').lower()
    
    # Safety-filtered fallbacks should:
    # 1. NOT include symbolic (can't parse English)
    # 2. Prefer world_model and analogical (can handle natural language)
    # 3. Include general as last resort for LLM synthesis
    
    # Query-specific safety fallbacks (NO symbolic!)
    safety_fallbacks = {
        # Self-introspection blocked by safety → try analogical (can reason by comparison)
        'self_introspection': ['analogical', 'world_model', 'general'],
        
        # Ethical/philosophical blocked → try analogical reasoning
        'ethical': ['analogical', 'world_model', 'general'],
        'philosophical': ['analogical', 'world_model', 'general'],
        
        # Capability description blocked → try analogical
        'capabilities': ['analogical', 'general'],
        
        # General self-description blocked
        'identity': ['analogical', 'world_model', 'general'],
    }
    
    # Get type-specific fallbacks
    fallbacks = safety_fallbacks.get(query_type_lower, ['analogical', 'world_model', 'general'])
    
    # Filter out the original tool
    fallbacks = [t for t in fallbacks if t != original_tool]
    
    return fallbacks[:3]  # Limit to 3 fallbacks


class ReasoningStrategyType(Enum):
    """
    Enumeration of available reasoning strategies.

    Each strategy represents a different approach to solving reasoning problems,
    with varying trade-offs between speed, accuracy, and resource usage.
    """

    DIRECT = "direct"
    CAUSAL_REASONING = "causal_reasoning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    PLANNING = "planning"
    DELIBERATIVE = "deliberative"
    META_REASONING = "meta_reasoning"
    PHILOSOPHICAL_REASONING = "philosophical_reasoning"  # Note: Strategy for ethical/deontic reasoning
    DEFAULT = "default"


# Maps query types to appropriate reasoning strategies
# NOTE: 'philosophical' now routes to world_model which has full meta-reasoning machinery
QUERY_TYPE_STRATEGY_MAP: Dict[str, str] = {
    "reasoning": "causal_reasoning",
    "execution": "planning",
    "perception": "analogical_reasoning",
    "planning": "deliberative",
    "learning": "meta_reasoning",
    "general": "direct",
    "philosophical": "meta_reasoning",  # Route to World Model meta-reasoning
    "ethical": "meta_reasoning",        # Route to World Model meta-reasoning
}

# ==================================================================
# FIX TASK 3: Map query routes to reasoning types
# Production logs showed reasoning returning type=UNKNOWN with confidence=0.1
# because the system didn't know how to map route types to reasoning types.
# This mapping ensures proper reasoning type classification.
# NOTE: 'philosophical' now maps to 'world_model' for ethical reasoning
# ==================================================================
ROUTE_TO_REASONING_TYPE: Dict[str, str] = {
    # Fast-path routes from query_router.py
    "PHILOSOPHICAL-FAST-PATH": "world_model",    # Philosophical queries route to World Model
    "MATH-FAST-PATH": "mathematical",            # Math queries use mathematical reasoning
    "CAUSAL-PATH": "causal",                     # Causal queries use causal reasoning
    "IDENTITY-FAST-PATH": "symbolic",            # Identity queries use symbolic reasoning
    "CONVERSATIONAL-FAST-PATH": "hybrid",        # Conversational uses hybrid
    "FACTUAL-FAST-PATH": "probabilistic",        # Factual queries use probabilistic
    "ANALOGICAL-PATH": "analogical",             # Note: Added analogical fast-path
    # QueryType enum values from query_router.py
    "philosophical": "world_model",              # PHILOSOPHICAL -> World Model
    "mathematical": "mathematical",              # MATHEMATICAL query type
    "causal": "causal",                          # CAUSAL query type
    "identity": "symbolic",                      # IDENTITY query type
    "conversational": "hybrid",                  # CONVERSATIONAL query type
    "factual": "probabilistic",                  # FACTUAL query type
    "general": "hybrid",                         # GENERAL query type (default)
    "reasoning": "causal",                       # Generic reasoning
    "execution": "symbolic",                     # Execution tasks
    "analogical": "analogical",                  # Note: Added analogical mapping
    "perception": "analogical",                  # Note: Perception often uses analogical reasoning
    "ethical": "world_model",                    # Ethical queries -> World Model
    # Legacy/fallback mappings
    "HYBRID": "hybrid",
    "UNKNOWN": "hybrid",
}


def get_reasoning_type_from_route(query_type: str, route: Optional[str] = None) -> str:
    """
    Get the appropriate reasoning type from query route or query type.
    
    FIX TASK 3: This function ensures proper reasoning type classification
    instead of returning UNKNOWN with confidence=0.1.
    
    Args:
        query_type: The query type (e.g., "reasoning", "philosophical", "mathematical")
        route: Optional route string (e.g., "PHILOSOPHICAL-FAST-PATH")
        
    Returns:
        Reasoning type string (e.g., "symbolic", "causal", "mathematical")
    """
    # Try route first (more specific)
    if route:
        route_upper = route.upper()
        if route_upper in ROUTE_TO_REASONING_TYPE:
            return ROUTE_TO_REASONING_TYPE[route_upper]
    
    # Try query_type (case-insensitive)
    if query_type:
        query_type_lower = query_type.lower()
        if query_type_lower in ROUTE_TO_REASONING_TYPE:
            return ROUTE_TO_REASONING_TYPE[query_type_lower]
    
    # Default to hybrid for unknown types
    logger.debug(
        f"{LOG_PREFIX} Unknown query_type='{query_type}' route='{route}', "
        f"defaulting to 'hybrid' reasoning type"
    )
    return "hybrid"


# =============================================================================
# GAP 5 FIX: Routing Decision Audit Trail
# =============================================================================
# Problem: Tool selection decisions are made at three layers:
#   1. QueryRouter: Classifies query and selects initial tools
#   2. ReasoningIntegration: May override tools based on self-reference/ethical detection
#   3. AgentPool: May override again based on task type detection
#
# These layers don't communicate - they just override each other, causing
# contradictions like:
#   Router selects: ['probabilistic', 'symbolic', 'mathematical']
#   Reasoning integration overrides to: ['world_model']
#   Actual execution uses: Meta-reasoning only
#
# Solution: Add RoutingDecision to track all routing decisions with audit trail.
# This provides a single source of truth and transparency into overrides.
# =============================================================================

@dataclass
class RoutingDecision:
    """
    GAP 5 FIX: Audit trail for tool selection decisions.
    
    Tracks the full history of tool selection decisions across all layers,
    providing transparency and debugging capability for routing issues.
    
    Attributes:
        original_query: The original query string
        router_tools: Tools selected by QueryRouter (Layer 1)
        integration_tools: Tools after ReasoningIntegration (Layer 2)
        final_tools: Final tools after all overrides (Layer 3)
        override_applied: Whether any layer overrode a previous decision
        override_reasons: List of reasons for each override
        decision_history: Full history of decisions at each layer
        timestamp: When the decision was made
    """
    
    original_query: str
    router_tools: List[str]
    integration_tools: List[str]
    final_tools: List[str]
    override_applied: bool = False
    override_reasons: List[str] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize and detect overrides."""
        # Detect if overrides occurred
        if self.router_tools != self.integration_tools:
            self.override_applied = True
            self.decision_history.append({
                "layer": "reasoning_integration",
                "from": self.router_tools,
                "to": self.integration_tools,
                "timestamp": time.time()
            })
        
        if self.integration_tools != self.final_tools:
            self.override_applied = True
            self.decision_history.append({
                "layer": "execution",
                "from": self.integration_tools,
                "to": self.final_tools,
                "timestamp": time.time()
            })
    
    def add_override(self, layer: str, from_tools: List[str], to_tools: List[str], reason: str):
        """Record an override decision."""
        self.override_applied = True
        self.override_reasons.append(f"[{layer}] {reason}")
        self.decision_history.append({
            "layer": layer,
            "from": from_tools,
            "to": to_tools,
            "reason": reason,
            "timestamp": time.time()
        })
        
        # Update appropriate tools field
        if layer == "reasoning_integration":
            self.integration_tools = to_tools
        elif layer in ["execution", "agent_pool"]:
            self.final_tools = to_tools
    
    def get_routing_integrity(self) -> float:
        """
        Calculate routing integrity score.
        
        Returns:
            1.0 if no overrides, decreasing with each override
        """
        if not self.override_applied:
            return 1.0
        
        # Penalize each override
        num_overrides = len(self.decision_history)
        return max(0.0, 1.0 - (num_overrides * 0.25))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "original_query": self.original_query[:100] + "..." if len(self.original_query) > 100 else self.original_query,
            "router_tools": self.router_tools,
            "integration_tools": self.integration_tools,
            "final_tools": self.final_tools,
            "override_applied": self.override_applied,
            "override_reasons": self.override_reasons,
            "decision_history": self.decision_history,
            "routing_integrity": self.get_routing_integrity(),
            "timestamp": self.timestamp
        }


@dataclass
class ReasoningResult:
    """
    Result from reasoning module containing tool selection and strategy information.

    This dataclass encapsulates all information about the reasoning decision,
    including which tools were selected, what strategy was used, and metadata
    about the selection process.

    Attributes:
        selected_tools: List of tool names selected for the query.
            Example: ["symbolic", "causal"]
        reasoning_strategy: Name of the reasoning strategy applied.
            Example: "causal_reasoning"
        confidence: Confidence score in the selection (0.0 to 1.0).
            Higher values indicate more reliable selections.
        rationale: Human-readable explanation of the selection decision.
            Useful for debugging and transparency.
        metadata: Additional context information about the selection.
            Contains timing, complexity, and component availability info.
        routing_decision: GAP 5 FIX - Optional audit trail for tool selection.
            Tracks all routing decisions across layers for debugging.

    Example:
        >>> result = ReasoningResult(
        ...     selected_tools=["causal"],
        ...     reasoning_strategy="causal_reasoning",
        ...     confidence=0.85,
        ...     rationale="High complexity reasoning query",
        ...     metadata={"complexity": 0.75, "query_type": "reasoning"}
        ... )
    """

    selected_tools: List[str]
    reasoning_strategy: str
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    routing_decision: Optional[RoutingDecision] = None  # GAP 5 FIX

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        # Ensure confidence is within valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Ensure we have at least one tool selected
        if not self.selected_tools:
            self.selected_tools = ["general"]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the reasoning result.
        """
        result = {
            "selected_tools": self.selected_tools,
            "reasoning_strategy": self.reasoning_strategy,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
        # GAP 5 FIX: Include routing decision if present
        if self.routing_decision:
            result["routing_decision"] = self.routing_decision.to_dict()
        return result


@dataclass
class IntegrationStatistics:
    """
    Statistics for monitoring reasoning integration performance.

    Thread-safe dataclass for tracking performance metrics and health
    indicators of the reasoning integration layer.

    Attributes:
        invocations: Total number of reasoning invocations
        tool_selections: Number of successful tool selections via ToolSelector
        portfolio_executions: Number of portfolio executions completed
        errors: Number of errors encountered during processing
        fast_path_count: Number of queries using the fast path optimization
        avg_selection_time_ms: Rolling average time for tool selection
        last_error: Description of the most recent error (for debugging)
    """

    invocations: int = 0
    tool_selections: int = 0
    portfolio_executions: int = 0
    errors: int = 0
    fast_path_count: int = 0
    avg_selection_time_ms: float = 0.0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate as ratio of successful operations.

        Returns:
            Success rate between 0.0 and 1.0
        """
        if self.invocations == 0:
            return 0.0
        return (self.invocations - self.errors) / self.invocations


class ReasoningIntegration:
    """
    Integrates reasoning module into query processing pipeline.

    This class provides a unified interface for applying reasoning-based tool
    selection and strategy determination. It handles lazy initialization of
    heavy components and provides graceful degradation when components are
    unavailable.

    Thread Safety:
        All methods are thread-safe. Internal components are initialized
        lazily with proper double-checked locking to minimize contention.

    Attributes:
        _tool_selector: Lazy-loaded ToolSelector instance
        _portfolio_executor: Lazy-loaded PortfolioExecutor instance
        _initialized: Whether components have been initialized
        _stats: Statistics tracking object

    Example:
        >>> integration = ReasoningIntegration()
        >>> result = integration.apply_reasoning(
        ...     query="What causes X?",
        ...     query_type="reasoning",
        ...     complexity=0.75
        ... )
        >>> print(f"Strategy: {result.reasoning_strategy}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize reasoning integration with lazy component loading.

        Args:
            config: Optional configuration dictionary with keys:
                - max_workers: Maximum parallel workers (default: 4)
                - time_budget_ms: Time budget in ms (default: 5000)
                - energy_budget_mj: Energy budget in mJ (default: 1000)
                - min_confidence: Minimum confidence (default: 0.5)
                - tool_selector_config: Config passed to ToolSelector
                - enable_decomposition: Enable problem decomposition (default: True)
                - enable_cross_domain_transfer: Enable cross-domain knowledge transfer (default: True)
        """
        self._config = config or {}

        # Lazy-loaded components
        self._tool_selector: Optional[Any] = None
        self._portfolio_executor: Optional[Any] = None
        self._problem_decomposer: Optional[Any] = None
        self._query_bridge: Optional[Any] = None
        self._semantic_bridge: Optional[Any] = None
        self._domain_bridge: Optional[Any] = None

        # Initialization state with thread safety
        self._initialized = False
        self._init_lock = threading.Lock()

        # Statistics tracking with thread safety
        self._stats = IntegrationStatistics()
        self._stats_lock = threading.RLock()

        # Selection timing for performance monitoring
        self._selection_times: List[float] = []

        # Shutdown state
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        
        # Note: Track fallback attempts per query to prevent infinite loops
        # Maps query hash to number of fallback attempts
        self._fallback_attempts: Dict[str, int] = {}
        self._fallback_attempts_lock = threading.Lock()
        
        # Feature configuration
        self._decomposition_enabled = self._config.get('enable_decomposition', True)
        self._cross_domain_enabled = self._config.get('enable_cross_domain_transfer', True)

        logger.info(
            f"{LOG_PREFIX} Initialized (lazy loading enabled, "
            f"decomposition={self._decomposition_enabled}, "
            f"cross_domain={self._cross_domain_enabled})"
        )

    def _should_use_decomposition(self, complexity: float) -> bool:
        """
        Determine if problem decomposition should be used for a query.
        
        Decomposition is used when:
        - Decomposition is enabled in config
        - Query complexity is at or above the threshold (0.40)
        - ProblemDecomposer and QueryBridge are available
        
        Args:
            complexity: Query complexity score (0.0 to 1.0)
            
        Returns:
            True if decomposition should be used, False otherwise
        """
        should_decompose = (
            self._decomposition_enabled
            and complexity >= DECOMPOSITION_COMPLEXITY_THRESHOLD
            and self._problem_decomposer is not None
            and self._query_bridge is not None
        )
        
        # Diagnostic logging for ProblemDecomposer utilization tracking
        if not should_decompose and complexity >= 0.3:
            reasons = []
            if not self._decomposition_enabled:
                reasons.append("decomposition_disabled")
            if complexity < DECOMPOSITION_COMPLEXITY_THRESHOLD:
                reasons.append(f"complexity {complexity:.2f} < threshold {DECOMPOSITION_COMPLEXITY_THRESHOLD}")
            if self._problem_decomposer is None:
                reasons.append("problem_decomposer_unavailable")
            if self._query_bridge is None:
                reasons.append("query_bridge_unavailable")
            logger.debug(
                f"{LOG_PREFIX} Decomposition skipped: {', '.join(reasons)} "
                f"(complexity={complexity:.2f})"
            )
        elif should_decompose:
            logger.info(
                f"{LOG_PREFIX} Decomposition ENABLED: complexity={complexity:.2f} >= "
                f"threshold={DECOMPOSITION_COMPLEXITY_THRESHOLD}"
            )
        
        return should_decompose

    def _init_components(self) -> None:
        """
        Lazy initialization of reasoning components.

        Uses double-checked locking pattern to ensure thread-safe initialization
        while minimizing lock contention. Components that fail to initialize
        are logged but don't prevent basic operation.

        This method is idempotent and safe to call multiple times.
        """
        # Fast path - already initialized
        if self._initialized:
            return

        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            init_start = time.perf_counter()

            # Try to initialize ToolSelector
            self._tool_selector = self._init_tool_selector()

            # Try to initialize PortfolioExecutor
            self._portfolio_executor = self._init_portfolio_executor()
            
            # Try to initialize ProblemDecomposer and QueryBridge (if decomposition enabled)
            if self._decomposition_enabled:
                self._problem_decomposer = self._init_problem_decomposer()
                self._query_bridge = self._init_query_bridge()
            
            # Try to initialize SemanticBridge and DomainBridge (if cross-domain enabled)
            if self._cross_domain_enabled:
                self._semantic_bridge = self._init_semantic_bridge()
                self._domain_bridge = self._init_domain_bridge()

            init_time = (time.perf_counter() - init_start) * 1000

            self._initialized = True

            logger.info(
                f"{LOG_PREFIX} Components initialized in {init_time:.1f}ms "
                f"(ToolSelector: {self._tool_selector is not None}, "
                f"PortfolioExecutor: {self._portfolio_executor is not None}, "
                f"ProblemDecomposer: {self._problem_decomposer is not None}, "
                f"SemanticBridge: {self._semantic_bridge is not None})"
            )

    def _init_tool_selector(self) -> Optional[Any]:
        """
        Initialize ToolSelector component with error handling.

        PERFORMANCE FIX: Uses singleton from singletons.py to ensure ToolSelector
        is created exactly ONCE per process. This prevents progressive query routing
        degradation where each query creates new instances of:
        - WarmStartPool ("Warm pool initialized with 5 tool pools")
        - StochasticCostModel ("StochasticCostModel initialized")
        - BayesianMemoryPrior with SemanticToolMatcher

        Returns:
            ToolSelector instance if successful, None otherwise.
        """
        try:
            # PERFORMANCE FIX: Use singleton instead of creating new instance
            # This prevents "Tool Selector initialized with 5 tools" appearing
            # multiple times and causing progressive routing time degradation
            from vulcan.reasoning.singletons import get_tool_selector

            selector = get_tool_selector()
            if selector is not None:
                logger.info(f"{LOG_PREFIX} ToolSelector obtained from singleton")
                return selector

            # Fallback: If singleton fails, try direct creation (should be rare)
            logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating ToolSelector directly")
            from vulcan.reasoning.selection.tool_selector import ToolSelector
            selector = ToolSelector(self._config.get("tool_selector_config", {}))
            logger.info(f"{LOG_PREFIX} ToolSelector initialized successfully (fallback)")
            return selector

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} ToolSelector not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} ToolSelector initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_portfolio_executor(self) -> Optional[Any]:
        """
        Initialize PortfolioExecutor component with error handling.

        Returns:
            PortfolioExecutor instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.selection.portfolio_executor import PortfolioExecutor

            max_workers = self._config.get("max_workers", DEFAULT_MAX_WORKERS)
            executor = PortfolioExecutor(tools={}, max_workers=max_workers)
            logger.info(f"{LOG_PREFIX} PortfolioExecutor initialized successfully")
            return executor

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} PortfolioExecutor not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} PortfolioExecutor initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_problem_decomposer(self) -> Optional[Any]:
        """
        Initialize ProblemDecomposer component with error handling.
        
        PERFORMANCE FIX: Uses singleton from singletons.py to ensure ProblemDecomposer
        is created exactly ONCE per process.

        Returns:
            ProblemDecomposer instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.singletons import get_problem_decomposer

            decomposer = get_problem_decomposer()
            if decomposer is not None:
                logger.info(f"{LOG_PREFIX} ProblemDecomposer obtained from singleton")
                return decomposer

            # Fallback: If singleton fails, try direct creation
            logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating ProblemDecomposer directly")
            from vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            decomposer = create_decomposer()
            logger.info(f"{LOG_PREFIX} ProblemDecomposer initialized successfully (fallback)")
            return decomposer

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} ProblemDecomposer not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} ProblemDecomposer initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_query_bridge(self) -> Optional[Any]:
        """
        Initialize QueryToProblemBridge component with error handling.
        
        NOTE: QueryToProblemBridge has been REMOVED as part of architecture simplification.
        The bridge was patching router decomposition issues that are now fixed at root cause.

        Returns:
            None - bridge is no longer used.
        """
        # QueryToProblemBridge removed - return None
        logger.debug(f"{LOG_PREFIX} QueryToProblemBridge removed (architectural simplification)")
        return None

    def _init_semantic_bridge(self) -> Optional[Any]:
        """
        Initialize SemanticBridge component with error handling.
        
        PERFORMANCE FIX: Uses singleton from singletons.py to ensure SemanticBridge
        is created exactly ONCE per process.

        Returns:
            SemanticBridge instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.singletons import get_semantic_bridge

            bridge = get_semantic_bridge()
            if bridge is not None:
                logger.info(f"{LOG_PREFIX} SemanticBridge obtained from singleton")
                return bridge

            # Fallback: If singleton fails, try direct creation
            logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating SemanticBridge directly")
            from vulcan.semantic_bridge import create_semantic_bridge
            bridge = create_semantic_bridge()
            logger.info(f"{LOG_PREFIX} SemanticBridge initialized successfully (fallback)")
            return bridge

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} SemanticBridge not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} SemanticBridge initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_domain_bridge(self) -> Optional[Any]:
        """
        Initialize ToolDomainBridge component with error handling.

        Returns:
            ToolDomainBridge instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.tool_domain_bridge import get_tool_domain_bridge

            bridge = get_tool_domain_bridge()
            logger.info(f"{LOG_PREFIX} ToolDomainBridge initialized successfully")
            return bridge

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} ToolDomainBridge not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} ToolDomainBridge initialization failed: {e}",
                exc_info=True
            )

        return None

    def _should_use_cross_domain_transfer(
        self,
        selected_tools: List[str],
    ) -> bool:
        """
        Determine if cross-domain knowledge transfer should be used.
        
        Cross-domain transfer is used when:
        - Cross-domain transfer is enabled in config
        - SemanticBridge and DomainBridge are available
        - Query uses tools from multiple domains
        
        Args:
            selected_tools: List of selected tool names
            
        Returns:
            True if cross-domain transfer should be used, False otherwise
        """
        if not self._cross_domain_enabled:
            return False
        
        if self._semantic_bridge is None or self._domain_bridge is None:
            return False
        
        if len(selected_tools) < 2:
            return False
        
        return self._domain_bridge.is_cross_domain_query(selected_tools)

    def apply_reasoning(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Apply reasoning to select tools and determine strategy for a query.

        This is the main entry point for applying reasoning-based tool selection.
        It analyzes the query characteristics and uses the ToolSelector (if
        available) to determine the best tools and reasoning strategy.

        Args:
            query: The user query text to process.
            query_type: Type of query from the router. Valid types include:
                - "general": General knowledge queries
                - "reasoning": Logical reasoning queries
                - "execution": Action/task execution queries
                - "perception": Pattern recognition queries
                - "planning": Multi-step planning queries
                - "learning": Knowledge acquisition queries
            complexity: Query complexity score (0.0 to 1.0).
                - 0.0-0.3: Simple queries (fast path)
                - 0.3-0.7: Medium complexity
                - 0.7-1.0: High complexity (full analysis)
            context: Optional context dictionary containing:
                - conversation_id: ID of the conversation
                - history: Previous messages in conversation
                - user_preferences: User-specific settings

        Returns:
            ReasoningResult with selected tools, strategy, and metadata.

        Raises:
            No exceptions are raised. Errors result in fallback to default strategy.

        Example:
            >>> integration = ReasoningIntegration()
            >>> result = integration.apply_reasoning(
            ...     query="Explain quantum entanglement",
            ...     query_type="reasoning",
            ...     complexity=0.8,
            ...     context={"conversation_id": "conv_123"}
            ... )
            >>> print(result.selected_tools)
            ['causal', 'probabilistic']
        """
        # Check shutdown state
        with self._shutdown_lock:
            if self._shutdown:
                logger.warning(f"{LOG_PREFIX} Called after shutdown, returning default")
                return self._create_default_result(query_type, complexity)

        # Initialize components if needed
        self._init_components()

        # Track invocation
        selection_start = time.perf_counter()
        with self._stats_lock:
            self._stats.invocations += 1

        try:
            # =================================================================
            # Note: Check for self-referential queries FIRST
            # Note: Also check for ethical queries that need world model
            # =================================================================
            # World model handles queries about VULCAN's self directly.
            # For ALL queries that are about VULCAN itself (capabilities,
            # preferences, self-awareness, etc.), consult world model first.
            # Ethical queries also benefit from world model's ethical framework.
            # =================================================================
            is_self_ref = self._is_self_referential(query)
            is_ethical = self._is_ethical_query(query)
            
            if is_self_ref or is_ethical:
                # Handle overlap: prioritize self-referential if both
                query_type_label = 'self-referential' if is_self_ref else 'ethical'
                if is_self_ref and is_ethical:
                    query_type_label = 'self-referential and ethical'
                logger.info(f"{LOG_PREFIX} Note: {query_type_label.capitalize()} query detected - consulting world model first")
                wm_result = self._consult_world_model_introspection(query)
                
                # ═══════════════════════════════════════════════════════════════════
                # Note: Handle World Model Delegation
                # The world model now has delegation intelligence - it can detect when
                # a query LOOKS self-referential but actually needs another reasoner.
                # Example: "Trolley problem - you must choose" is ethical reasoning
                #          posed TO the AI, not a question ABOUT the AI.
                # ═══════════════════════════════════════════════════════════════════
                
                if wm_result is not None and wm_result.get("needs_delegation", False):
                    recommended_tool = wm_result.get("recommended_tool")
                    delegation_reason = wm_result.get("delegation_reason", "")
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model recommends DELEGATION to "
                        f"'{recommended_tool}' - {delegation_reason}"
                    )
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # CRITICAL FIX (Jan 6 2026): EXECUTE DELEGATION IMMEDIATELY
                    # ═══════════════════════════════════════════════════════════════════
                    # PROBLEM: Previous code set context flags and continued to normal
                    # processing, but Note in tool_selector.py was overriding
                    # the delegation because it checks for formal logic BEFORE checking
                    # delegation context.
                    #
                    # Note: Execute the delegated tool HERE and return immediately.
                    # This prevents any downstream code from overriding the delegation.
                    #
                    # Evidence from logs:
                    #   Line 2853: [ReasoningIntegration] LLM Classification: category=SELF_INTROSPECTION
                    #   Line 2854: [WorldModel] DELEGATION RECOMMENDED: 'mathematical'
                    #   Line 2855: [ReasoningIntegration] SELF_INTROSPECTION detected - using world_model tool
                    #   ^ CONTRADICTION: Says delegation active but uses world_model
                    # ═══════════════════════════════════════════════════════════════════
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model delegation ACTIVE - "
                        f"executing '{recommended_tool}' immediately (NOT falling through)"
                    )
                    
                    # Set up context with delegation info
                    if context is None:
                        context = {}
                    
                    context['world_model_delegation'] = True
                    context['world_model_recommended_tool'] = recommended_tool
                    context['world_model_delegation_reason'] = delegation_reason
                    context['classifier_suggested_tools'] = [recommended_tool]
                    context['classifier_is_authoritative'] = True
                    context['prevent_router_tool_override'] = True
                    context['skip_task3_fix'] = True  # Tell tool_selector to skip formal logic check
                    
                    # Map tool name to query_type for proper routing
                    if recommended_tool == 'philosophical':
                        query_type = 'ethical'
                    elif recommended_tool == 'mathematical':
                        query_type = 'mathematical'
                    elif recommended_tool == 'causal':
                        query_type = 'causal'
                    elif recommended_tool == 'probabilistic':
                        query_type = 'probabilistic'
                    
                    # Execute with the delegated tool directly via _select_with_tool_selector
                    # This is the EARLY RETURN that was missing
                    selection_time_start = time.perf_counter()
                    result = self._select_with_tool_selector(
                        query, query_type, complexity, context
                    )
                    selection_time = (time.perf_counter() - selection_time_start) * 1000
                    
                    # FIX: Verify delegation actually happened - log warning if not
                    actual_tool = result.selected_tools[0] if result.selected_tools else "none"
                    if actual_tool != recommended_tool:
                        logger.warning(
                            f"{LOG_PREFIX} DELEGATION MISMATCH: World model recommended "
                            f"'{recommended_tool}' but tool_selector returned '{actual_tool}'. "
                            f"This may indicate tool_selector is overriding delegation."
                        )
                        # Still return result - but flag the mismatch
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata["delegation_mismatch"] = True
                        result.metadata["expected_tool"] = recommended_tool
                        result.metadata["actual_tool"] = actual_tool
                    
                    # Add delegation metadata to result (with safety check)
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["world_model_delegation"] = True
                    result.metadata["delegated_tool"] = recommended_tool
                    result.metadata["delegation_reason"] = delegation_reason
                    result.metadata["selection_time_ms"] = selection_time
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: Delegation complete - executed '{recommended_tool}' "
                        f"with confidence={result.confidence:.2f} (EARLY RETURN)"
                    )
                    
                    # EARLY RETURN - Do NOT fall through to normal processing
                    return result
                
                # Note: Lower threshold from 0.7 to 0.5 for world model
                # Only use world model result if NOT delegating
                elif wm_result is not None and wm_result.get("confidence", 0) >= 0.5:
                    # World model can handle this directly
                    selection_time = (time.perf_counter() - selection_start) * 1000
                    
                    # Note: Include conclusion in metadata for proper extraction
                    # The world model's response IS the conclusion for self-introspection queries
                    world_model_response = wm_result.get("response", "")
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model returned confidence "
                        f"{wm_result['confidence']:.2f}. Using this result directly "
                        f"without other engines."
                    )
                    
                    # Determine reasoning type: self-referential takes priority
                    reasoning_type = "meta_reasoning" if is_self_ref else "philosophical_reasoning"
                    strategy_type = ReasoningStrategyType.META_REASONING.value if is_self_ref else ReasoningStrategyType.PHILOSOPHICAL_REASONING.value
                    
                    return ReasoningResult(
                        selected_tools=["world_model"],
                        reasoning_strategy=strategy_type,
                        confidence=wm_result["confidence"],
                        rationale=wm_result.get("reasoning", "World model introspection"),
                        metadata={
                            "query_type": query_type,
                            "complexity": complexity,
                            "self_referential": is_self_ref,
                            "ethical_query": is_ethical,
                            "world_model_response": world_model_response,
                            # Note: Add conclusion field so main.py can extract it
                            "conclusion": world_model_response,
                            "explanation": wm_result.get("reasoning", ""),
                            "reasoning_type": reasoning_type,
                            "aspect": wm_result.get("aspect", "general"),
                            "selection_time_ms": selection_time,
                        },
                    )

            # =================================================================
            # Note: LLM-BASED QUERY CLASSIFICATION (ROOT CAUSE FIX)
            # =================================================================
            # The problem: "hello" (5 chars) was getting complexity=0.50 from
            # heuristic-based calculation, causing it to hit full reasoning.
            # 
            # The fix: Use LLM layer for LANGUAGE UNDERSTANDING to:
            # 1. Determine if query needs reasoning at all
            # 2. Suggest appropriate tools based on query intent
            # 3. Set correct complexity based on actual query meaning
            #
            # Architecture note: LLMs (cloud or internal) are interchangeable
            # for classification. The reasoning engines provide correctness.
            # =================================================================
            try:
                from vulcan.routing.query_classifier import classify_query, QueryCategory
                
                classification = classify_query(query)
                
                logger.info(
                    f"{LOG_PREFIX} LLM Classification: category={classification.category}, "
                    f"complexity={classification.complexity:.2f}, skip={classification.skip_reasoning}, "
                    f"tools={classification.suggested_tools}"
                )
                
                # If classifier says skip reasoning (greetings, chitchat, simple factual)
                # return immediately without invoking any reasoning engine
                if classification.skip_reasoning:
                    logger.info(
                        f"{LOG_PREFIX} CLASSIFIER SKIP: '{query[:30]}' classified as "
                        f"{classification.category} - skipping reasoning entirely"
                    )
                    with self._stats_lock:
                        self._stats.fast_path_count += 1
                    
                    return ReasoningResult(
                        selected_tools=classification.suggested_tools or ["general"],
                        reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                        confidence=classification.confidence,
                        rationale=f"Query classified as {classification.category} - no reasoning needed",
                        metadata={
                            "fast_path": True,
                            "classifier_category": classification.category,
                            "classifier_source": classification.source,
                            "complexity": classification.complexity,
                            "query_type": classification.category.lower(),
                            "selection_time_ms": (time.perf_counter() - selection_start) * 1000,
                            "needs_reasoning": False,
                        },
                    )
                
                # Classifier identified this needs reasoning - use its suggestions
                # Override the heuristic complexity with LLM-derived complexity
                if classification.complexity != complexity:
                    logger.info(
                        f"{LOG_PREFIX} Overriding heuristic complexity {complexity:.2f} with "
                        f"classifier complexity {classification.complexity:.2f}"
                    )
                    complexity = classification.complexity
                
                # If classifier suggested specific tools, pass them to context
                # Note: DON'T override world model delegation!
                # If world_model_delegation is set, the world model has already determined
                # the correct tool. The classifier should NOT override this expert judgment.
                if classification.suggested_tools:
                    if context is None:
                        context = {}
                    
                    # Note: Check if world model delegation is active
                    if context.get('world_model_delegation'):
                        logger.info(
                            f"{LOG_PREFIX} Note: World model delegation ACTIVE - "
                            f"NOT overriding with classifier tools {classification.suggested_tools}. "
                            f"Using delegated tool: {context.get('classifier_suggested_tools')}"
                        )
                        # Keep the world model's recommended tool, just add category info
                        context['classifier_category'] = classification.category
                    else:
                        # Normal case: use classifier suggestions
                        context['classifier_suggested_tools'] = classification.suggested_tools
                        context['classifier_category'] = classification.category
                        logger.info(
                            f"{LOG_PREFIX} Using classifier suggested tools: {classification.suggested_tools}"
                        )
                
                # =============================================================
                # FIX #4: Prevent tool override for simple/factual queries
                # =============================================================
                # The LLM classifier correctly identifies simple factual queries
                # (e.g., "What is the capital of France?") but QueryRouter may
                # incorrectly override with specialized tools like ['probabilistic'].
                # Respect the LLM classifier's judgment for these categories.
                # Note: Added CREATIVE and CHITCHAT to skip reasoning
                # Note: Added SELF_INTROSPECTION - these must use world_model tool
                SIMPLE_QUERY_CATEGORIES = frozenset([
                    'FACTUAL', 'CONVERSATIONAL', 'UNKNOWN', 'GREETING',
                    'CREATIVE', 'CHITCHAT',  # Note: Creative/chitchat skip reasoning
                    'factual', 'conversational', 'unknown', 'greeting',
                    'creative', 'chitchat',  # lowercase variants
                ])
                
                # Note: Self-introspection queries MUST use world_model tool
                # These query Vulcan's sense of self (CSIU, motivations, ethics, etc.)
                SELF_INTROSPECTION_CATEGORIES = frozenset([
                    'SELF_INTROSPECTION', 'self_introspection',
                ])
                
                if classification.category in SELF_INTROSPECTION_CATEGORIES:
                    # =================================================================
                    # FIX (Jan 9 2026): Check for domain reasoning keywords FIRST
                    # =================================================================
                    # Problem: Classifier may misclassify causal/analogical queries as
                    # SELF_INTROSPECTION, causing them to be forced to world_model.
                    # Example: "Confounding vs causation (Pearl-style)" classified as
                    # SELF_INTROSPECTION but should route to causal engine.
                    #
                    # Solution: Check for domain-specific keywords before forcing
                    # world_model. If domain keywords are found, route to specialized
                    # engine instead. world_model can still observe but doesn't block.
                    # =================================================================
                    query_lower = query.lower()
                    
                    # Domain keyword sets for specialized routing
                    # Note: These keywords are consistent with CAUSAL_KEYWORDS in query_classifier.py
                    # The threshold of 2+ keywords ensures single false matches don't trigger routing
                    DOMAIN_ROUTING_KEYWORDS = {
                        'causal': frozenset([
                            'causal', 'causation', 'confound', 'confounder', 'confounding',
                            'intervention', 'counterfactual', 'randomize', 'randomized',
                            'pearl', 'dag', 'backdoor', 'frontdoor', 'collider',
                            'do-calculus', 'rct', 'observational', 'experimental',
                        ]),
                        'analogical': frozenset([
                            'analogical', 'analogy', 'analogies', 'analogous',
                            'structure mapping', 'structural alignment',
                            'domain transfer', 'cross-domain', 'source domain', 'target domain',
                            'relational similarity', 'surface similarity', 'structural similarity',
                            's→t', 'domain s', 'domain t', 'deep structure',
                        ]),
                        'probabilistic': frozenset([
                            'bayes', 'bayesian', 'probability', 'probabilistic',
                            'likelihood', 'prior', 'posterior', 'conditional probability',
                            'joint distribution', 'marginal', 'independence',
                        ]),
                    }
                    
                    # Check if query contains domain reasoning keywords
                    detected_domain = None
                    detected_count = 0
                    for domain, keywords in DOMAIN_ROUTING_KEYWORDS.items():
                        count = sum(1 for kw in keywords if kw in query_lower)
                        if count >= 2:  # Require 2+ keywords for domain detection
                            if count > detected_count:
                                detected_domain = domain
                                detected_count = count
                    
                    if detected_domain:
                        # Domain reasoning detected - route to specialized engine
                        logger.info(
                            f"{LOG_PREFIX} SELF_INTROSPECTION override: detected {detected_domain} "
                            f"reasoning ({detected_count} keywords) - routing to {detected_domain} "
                            f"engine instead of world_model"
                        )
                        classification.suggested_tools = [detected_domain]
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = [detected_domain]
                        context['classifier_category'] = classification.category
                        context['domain_reasoning_detected'] = detected_domain
                        context['domain_keyword_count'] = detected_count
                        # Let the specialized engine handle it - don't block with world_model
                        # Note: world_model can still observe in parallel mode
                    else:
                        # No domain keywords found - actual self-introspection
                        # For self-introspection queries, ensure we use world_model tool
                        # BUG FIX: Also update query_type to prevent type mismatch downstream
                        # Previously, query_type stayed as 'MATHEMATICAL' even after overriding tools
                        original_query_type = query_type
                        query_type = 'self_introspection'  # FIX: Update query_type to match actual query
                        
                        logger.info(
                            f"{LOG_PREFIX} SELF_INTROSPECTION detected - using world_model tool "
                            f"(classifier suggested: {classification.suggested_tools}). "
                            f"Updated query_type: {original_query_type} -> {query_type}"
                        )
                        # Ensure world_model is in the suggested tools
                        if 'world_model' not in (classification.suggested_tools or []):
                            classification.suggested_tools = ['world_model']
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = classification.suggested_tools
                        context['prevent_router_tool_override'] = True
                        context['classifier_is_authoritative'] = True
                        context['is_self_introspection'] = True
                        context['original_query_type'] = original_query_type  # FIX: Track original for debugging
                
                elif classification.category in SIMPLE_QUERY_CATEGORIES:
                    # For simple queries, ensure we use general tools
                    if classification.suggested_tools != ['general']:
                        logger.warning(
                            f"{LOG_PREFIX} FIX#4: LLM classifier suggested "
                            f"{classification.suggested_tools} for category "
                            f"{classification.category}, overriding to ['general'] "
                            f"for simple factual query"
                        )
                        classification.suggested_tools = ['general']
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = ['general']
                    
                    # Set flag to prevent router override downstream
                    if context is None:
                        context = {}
                    context['prevent_router_tool_override'] = True
                    context['classifier_is_authoritative'] = True
                    logger.info(
                        f"{LOG_PREFIX} FIX#4: Preventing router tool override - "
                        f"LLM classifier identified this as {classification.category}"
                    )
                    
            except ImportError:
                logger.debug(f"{LOG_PREFIX} QueryClassifier not available, using heuristic fallback")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} QueryClassifier failed: {e}, using heuristic fallback")
            
            # =================================================================
            # FALLBACK: Simple pattern matching for obvious cases
            # This is a safety net if QueryClassifier fails, NOT the primary path
            # =================================================================
            SIMPLE_QUERY_PATTERNS = frozenset([
                'hello', 'hi', 'hey', 'howdy', 'greetings',
                'thanks', 'thank you', 'bye', 'goodbye', 'see you',
                'good morning', 'good afternoon', 'good evening',
                'ok', 'okay', 'sure', 'yes', 'no', 'maybe',
            ])
            
            query_lower = query.lower().strip()
            is_simple_greeting = (
                query_lower in SIMPLE_QUERY_PATTERNS or
                len(query_lower) < 10 and not any(c in query_lower for c in '?∧∨→¬=')
            )
            
            if is_simple_greeting:
                logger.info(
                    f"{LOG_PREFIX} PATTERN FALLBACK: '{query[:20]}' (len={len(query)}) - "
                    f"skipping reasoning entirely"
                )
                with self._stats_lock:
                    self._stats.fast_path_count += 1
                
                return ReasoningResult(
                    selected_tools=["general"],
                    reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                    confidence=0.95,
                    rationale="Simple greeting/conversational - bypassing reasoning",
                    metadata={
                        "fast_path": True,
                        "simple_query_bypass": True,
                        "complexity": 0.0,  # Override upstream complexity
                        "query_type": "conversational",
                        "selection_time_ms": 0.0,
                    },
                )
            
            # Fast path for simple queries - skip heavy tool selection
            if complexity < FAST_PATH_COMPLEXITY_THRESHOLD:
                with self._stats_lock:
                    self._stats.fast_path_count += 1

                return ReasoningResult(
                    selected_tools=["general"],
                    reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                    confidence=0.9,
                    rationale="Simple query - using fast path direct response",
                    metadata={
                        "fast_path": True,
                        "complexity": complexity,
                        "query_type": query_type,
                        "selection_time_ms": 0.0,
                    },
                )

            # ================================================================
            # FIX #1: QUERY PREPROCESSING - REMOVED (architectural band-aid)
            # ================================================================
            # Query preprocessing has been removed. Root causes are now fixed
            # directly in the engines (cryptographic engine header detection,
            # symbolic reasoner query decomposition, etc.)
            # The QueryDecomposer is used directly by the SymbolicReasoner.

            # Check if we should use decomposition for complex queries
            if self._should_use_decomposition(complexity):
                # Use decomposition path for complex queries
                logger.info(
                    f"{LOG_PREFIX} Using decomposition path (complexity={complexity:.2f} >= "
                    f"{DECOMPOSITION_COMPLEXITY_THRESHOLD})"
                )
                result = self._process_with_decomposition(
                    query, query_type, complexity, context
                )
            else:
                # Use direct tool selection for simpler queries
                result = self._select_with_tool_selector(
                    query, query_type, complexity, context
                )

            # Record timing
            selection_time = (time.perf_counter() - selection_start) * 1000
            self._record_selection_time(selection_time)

            # Add timing to metadata
            result.metadata["selection_time_ms"] = selection_time

            # ================================================================
            # META-REASONING VALIDATION FIX: Validate answer coherence
            # ================================================================
            # This critical validation layer catches cases where:
            # - A mathematical result is returned for a self-introspection query
            # - A calculus answer is returned for a logical/ethical query
            # - WorldModel response is discarded and wrong cached result returned
            #
            # Example bug this fixes:
            #   Query: "what makes you different from other ai systems?"
            #   Wrong answer: "3*x**2" (derivative from cached math result)
            #   Expected: World model introspection about VULCAN's capabilities
            # ================================================================
            if ANSWER_VALIDATOR_AVAILABLE and validate_reasoning_result is not None:
                try:
                    # Extract conclusion from metadata using ordered key preference
                    conclusion = ""
                    if result.metadata:
                        # Keys to check in priority order
                        conclusion_keys = ["conclusion", "world_model_response"]
                        for key in conclusion_keys:
                            conclusion = result.metadata.get(key, "")
                            if conclusion:
                                break
                        
                        # Also check nested reasoning_output if not found
                        if not conclusion:
                            reasoning_output = result.metadata.get("reasoning_output", {})
                            if isinstance(reasoning_output, dict):
                                conclusion = reasoning_output.get("conclusion", "")
                    
                    # Only validate if we have something to validate
                    if conclusion:
                        validation_result = validate_reasoning_result(
                            query=query,
                            result={"conclusion": conclusion},
                            expected_type=query_type
                        )
                        
                        if not validation_result.valid:
                            logger.warning(
                                f"{LOG_PREFIX} META-REASONING VALIDATION FAILED: "
                                f"Answer does not match query type. "
                                f"Query: '{query[:60]}...', "
                                f"Answer type mismatch detected. "
                                f"Explanation: {validation_result.explanation}"
                            )
                            # Mark result as potentially invalid
                            result.metadata["validation_failed"] = True
                            result.metadata["validation_explanation"] = validation_result.explanation
                            # Reduce confidence to signal uncertainty
                            original_confidence = result.confidence
                            result.confidence = min(result.confidence, 0.3)
                            logger.warning(
                                f"{LOG_PREFIX} Confidence reduced from {original_confidence:.2f} "
                                f"to {result.confidence:.2f} due to validation failure"
                            )
                        else:
                            logger.debug(
                                f"{LOG_PREFIX} Meta-reasoning validation passed for query"
                            )
                except Exception as validation_err:
                    # Validation is non-critical - log but don't fail the whole request
                    logger.debug(
                        f"{LOG_PREFIX} Meta-reasoning validation failed (non-critical): {validation_err}"
                    )

            # ================================================================
            # FIX #3: LEARN FROM SUCCESSFUL REASONING
            # ================================================================
            # After successful reasoning, extract reusable principles using
            # KnowledgeCrystallizer. This enables learning patterns like:
            # "SAT queries with explicit constraints need preprocessing"
            try:
                preprocessing_applied = False
                if context and 'preprocessing' in context:
                    prep_result = context['preprocessing']
                    # Handle both dict and PreprocessingResult dataclass
                    if hasattr(prep_result, 'preprocessing_applied'):
                        preprocessing_applied = prep_result.preprocessing_applied
                    elif isinstance(prep_result, dict):
                        preprocessing_applied = prep_result.get('preprocessing_applied', False)

                # Learn from outcome if confidence is high enough
                if result.confidence >= 0.7:
                    logger.info(
                        f"{LOG_PREFIX} LEARNING TRIGGERED: confidence={result.confidence:.2f} >= 0.7"
                    )
                    self._learn_from_reasoning_outcome(
                        query=query,
                        query_type=query_type,
                        complexity=complexity,
                        selected_tools=result.selected_tools,
                        reasoning_strategy=result.reasoning_strategy,
                        success=True,
                        confidence=result.confidence,
                        execution_time=selection_time / 1000.0,  # Convert ms to seconds
                        preprocessing_applied=preprocessing_applied,
                    )
                else:
                    logger.debug(
                        f"{LOG_PREFIX} Learning skipped: confidence={result.confidence:.2f} < 0.7"
                    )
            except Exception as e:
                # Learning is non-critical - log but don't fail
                logger.debug(f"{LOG_PREFIX} Learning step failed (non-critical): {e}")

            return result

        except Exception as e:
            # Record error and return fallback
            with self._stats_lock:
                self._stats.errors += 1
                self._stats.last_error = str(e)

            logger.error(
                f"{LOG_PREFIX} Reasoning application failed: {e}",
                exc_info=True
            )

            return self._create_default_result(query_type, complexity)

    def _select_with_tool_selector(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]],
    ) -> ReasoningResult:
        """
        Perform tool selection using the ToolSelector component.

        Args:
            query: The user query
            query_type: Type of query
            complexity: Complexity score
            context: Optional context

        Returns:
            ReasoningResult from tool selection or fallback
        """
        # Default values for fallback
        selected_tools = ["general"]
        reasoning_strategy = ReasoningStrategyType.DEFAULT.value
        confidence = 0.7
        rationale = "Default reasoning strategy"

        # =================================================================
        # FIX: Use classifier suggested tools when ToolSelector unavailable
        # =================================================================
        # When ToolSelector is not available (e.g., numpy not installed),
        # we should still respect the classifier's tool suggestions instead
        # of falling back to generic ["general"] tools.
        # 
        # This ensures causal queries like "Confounding vs causation (Pearl-style)"
        # get routed to causal tools as the classifier intended, rather than
        # being downgraded to general tools.
        # =================================================================
        if context and context.get('classifier_suggested_tools'):
            classifier_tools = context.get('classifier_suggested_tools')
            if classifier_tools and classifier_tools != ["general"]:
                selected_tools = classifier_tools
                confidence = 0.85  # Higher confidence since classifier suggested these
                rationale = f"Using classifier suggested tools: {classifier_tools}"
                logger.info(
                    f"{LOG_PREFIX} Using classifier suggested tools as fallback: {classifier_tools}"
                )

        # Try to use ToolSelector if available
        if self._tool_selector is not None:
            try:
                # Import selection components
                from vulcan.reasoning.selection.tool_selector import (
                    SelectionRequest,
                    SelectionMode,
                )

                # Determine selection mode based on complexity
                mode = self._determine_selection_mode(complexity, SelectionMode)

                # Build constraints
                constraints = {
                    "time_budget_ms": self._config.get(
                        "time_budget_ms", DEFAULT_TIME_BUDGET_MS
                    ),
                    "energy_budget_mj": self._config.get(
                        "energy_budget_mj", DEFAULT_ENERGY_BUDGET_MJ
                    ),
                    "min_confidence": self._config.get(
                        "min_confidence", DEFAULT_MIN_CONFIDENCE
                    ),
                }

                # ================================================================
                # Note: Pass preprocessing result as part of problem dict
                # ================================================================
                # The SymbolicToolWrapper.reason() method expects preprocessing
                # to be in problem['preprocessing'], not just in context.
                # When preprocessing was applied, create a problem dict with
                # both the query and the preprocessing result.
                problem_for_request = query  # Default: just the query string
                
                if context and context.get('preprocessing'):
                    preprocessing = context.get('preprocessing')
                    # Check if preprocessing was actually applied
                    if hasattr(preprocessing, 'preprocessing_applied') and preprocessing.preprocessing_applied:
                        problem_for_request = {
                            'query': query,
                            'preprocessing': preprocessing,
                        }
                        logger.info(
                            f"{LOG_PREFIX} Note: Passing preprocessing to tool via problem dict"
                        )
                
                # Create selection request
                request = SelectionRequest(
                    problem=problem_for_request,
                    constraints=constraints,
                    mode=mode,
                    context=context or {},
                )

                # Execute selection
                result = self._tool_selector.select_and_execute(request)

                # Extract tools from result
                # NOTE: SelectionResult has 'selected_tool' (singular) and 'all_results'
                # We extract the list of tools from all_results keys or use selected_tool
                selected_tools = self._extract_tools_from_result(result)

                # Safely extract strategy and confidence with fallbacks
                if hasattr(result, "strategy_used") and result.strategy_used is not None:
                    reasoning_strategy = result.strategy_used.value
                    rationale = f"ToolSelector selected via {result.strategy_used.value} strategy"
                else:
                    reasoning_strategy = ReasoningStrategyType.DEFAULT.value
                    rationale = "ToolSelector selection (strategy unknown)"

                if hasattr(result, "calibrated_confidence"):
                    confidence = result.calibrated_confidence
                elif hasattr(result, "confidence"):
                    confidence = result.confidence
                else:
                    confidence = 0.7  # Default confidence

                # Track successful selection
                with self._stats_lock:
                    self._stats.tool_selections += 1

                logger.info(
                    f"{LOG_PREFIX} Tool selection complete: "
                    f"tools={selected_tools}, strategy={reasoning_strategy}, "
                    f"confidence={confidence:.2f}"
                )

                # ================================================================
                # Note: Fallback logic with retry limit
                # When confidence is < 0.1, the tool failed to process the query.
                # Instead of giving up, try fallback tools - but limit retries.
                # 
                # BUG #3: Production logs showed 8+ attempts with same failed tool.
                # We now track attempts per query and limit to MAX_FALLBACK_ATTEMPTS.
                # ================================================================
                if confidence < 0.1 and selected_tools:
                    original_tool = selected_tools[0] if selected_tools else 'unknown'
                    
                    # FIX: Check and increment fallback attempt counter
                    # Using SHA-256 instead of MD5 for security (per code review)
                    # Fixed to increment THEN check to avoid off-by-one error
                    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                    should_try_fallback = False
                    attempt_number = 0
                    
                    with self._fallback_attempts_lock:
                        current_attempts = self._fallback_attempts.get(query_hash, 0)
                        # Increment FIRST, then check
                        new_attempts = current_attempts + 1
                        self._fallback_attempts[query_hash] = new_attempts
                        attempt_number = new_attempts
                        
                        if new_attempts > MAX_FALLBACK_ATTEMPTS:
                            logger.error(
                                f"{LOG_PREFIX} Note: Max fallback attempts "
                                f"({MAX_FALLBACK_ATTEMPTS}) exceeded for query {query_hash}. "
                                f"Stopping retry loop to prevent resource waste."
                            )
                            # Set minimum floor and continue without more retries
                            confidence = MIN_CONFIDENCE_FLOOR
                        else:
                            should_try_fallback = True
                    
                    # Only try fallback if we haven't exceeded max attempts
                    if should_try_fallback:
                        logger.warning(
                            f"{LOG_PREFIX} Note: Tool '{original_tool}' returned very low "
                            f"confidence ({confidence:.3f}), trying fallback tools "
                            f"(attempt {attempt_number}/{MAX_FALLBACK_ATTEMPTS})"
                        )
                        
                        # ==============================================================
                        # FIX Issue D: Improved fallback logic for safety-filtered outputs
                        # ==============================================================
                        # When output is SAFETY-FILTERED (vs just low confidence), we need
                        # to use different fallback tools. Safety-filtered means the tool
                        # DID reason correctly but output was blocked - falling back to
                        # incompatible tools (like symbolic for English) is wrong.
                        #
                        # Detection: Check if result indicates safety filtering
                        # Fallback: Use _get_safety_filtered_fallback_tools() which
                        #           excludes incompatible tools like symbolic
                        # ==============================================================
                        is_safety_filtered = _is_result_safety_filtered(result)
                        
                        if is_safety_filtered:
                            logger.info(
                                f"{LOG_PREFIX} FIX Issue D: Safety-filtered output detected. "
                                f"Using safety-aware fallback selection (excluding incompatible tools)"
                            )
                            fallback_tools = _get_safety_filtered_fallback_tools(
                                query_type=query_type,
                                original_tool=original_tool
                            )
                        else:
                            # FIX #4: Regular fallback logic - try alternative engines before LLM
                            # =====================================================================
                            # Instead of a fixed list, select fallback tools based on query type
                            # and the original tool that failed. This ensures queries get routed
                            # to the most appropriate alternative engine.
                            #
                            # Priority order:
                            # 1. Query-type specific fallbacks (e.g., philosophical for ethical queries)
                            # 2. General-purpose fallbacks (world_model, probabilistic)
                            # 3. Arena delegation (LLM) as last resort
                            # =====================================================================
                            fallback_tools = self._get_fallback_tools(
                                query_type=query_type,
                                original_tool=original_tool,
                                failed_tools=selected_tools
                            )
                        
                        for fallback_tool in fallback_tools:
                            try:
                                logger.info(f"{LOG_PREFIX} Trying fallback tool: {fallback_tool}")
                                
                                # Create fallback request
                                # Note: Set fallback_attempt=True to bypass classifier
                                fallback_request = SelectionRequest(
                                    problem=problem_for_request,
                                    constraints=constraints,
                                    mode=mode,
                                    context={
                                        **(context or {}),
                                        'router_tools': [fallback_tool],  # Force this tool
                                        'fallback_attempt': True,  # BUG#1: Skip classifier
                                    },
                                )
                                
                                fallback_result = self._tool_selector.select_and_execute(
                                    fallback_request
                                )
                                
                                # Extract fallback confidence
                                fallback_confidence = 0.0
                                if hasattr(fallback_result, "calibrated_confidence"):
                                    fallback_confidence = fallback_result.calibrated_confidence
                                elif hasattr(fallback_result, "confidence"):
                                    fallback_confidence = fallback_result.confidence
                                
                                # If fallback is better, use it
                                if fallback_confidence > confidence:
                                    logger.info(
                                        f"{LOG_PREFIX} Fallback '{fallback_tool}' succeeded with "
                                        f"confidence={fallback_confidence:.3f} (better than {confidence:.3f})"
                                    )
                                    result = fallback_result
                                    selected_tools = self._extract_tools_from_result(fallback_result)
                                    confidence = fallback_confidence
                                    rationale = f"Fallback to {fallback_tool} after {original_tool} failed"
                                    
                                    # Update strategy
                                    if hasattr(fallback_result, "strategy_used") and fallback_result.strategy_used:
                                        reasoning_strategy = fallback_result.strategy_used.value
                                    break
                                else:
                                    logger.debug(
                                        f"{LOG_PREFIX} Fallback '{fallback_tool}' also low confidence: "
                                        f"{fallback_confidence:.3f}"
                                    )
                            except Exception as fallback_error:
                                logger.debug(f"{LOG_PREFIX} Fallback '{fallback_tool}' failed: {fallback_error}")
                                continue
                        
                        # If all fallbacks failed but we still have very low confidence,
                        # delegate to Arena as final attempt
                        if confidence < 0.1:
                            logger.info(
                                f"{LOG_PREFIX} All local tools returned low confidence. "
                                f"Attempting Arena delegation as fallback."
                            )
                            
                            # Try Arena delegation
                            arena_result = self._delegate_to_arena(
                                query=query,
                                original_tool=original_tool,
                                query_type=query_type,
                                complexity=complexity,
                                context=context,
                            )
                            
                            if arena_result is not None:
                                # Arena succeeded - use its result
                                arena_confidence = arena_result.get('confidence', 0.5)
                                if arena_confidence > confidence:
                                    confidence = arena_confidence
                                    rationale = f"Arena delegation after {original_tool} failed"
                                    logger.info(
                                        f"{LOG_PREFIX} Arena delegation succeeded: "
                                        f"confidence={arena_confidence:.3f}"
                                    )
                            else:
                                # Arena failed too - set minimum floor
                                logger.warning(
                                    f"{LOG_PREFIX} Arena delegation failed. "
                                    f"Setting minimum confidence floor of {MIN_CONFIDENCE_FLOOR}."
                                )
                                confidence = MIN_CONFIDENCE_FLOOR

            except ImportError as e:
                logger.warning(f"{LOG_PREFIX} ToolSelector imports unavailable: {e}")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} ToolSelector execution failed: {e}")
                with self._stats_lock:
                    self._stats.errors += 1

        # If no strategy was selected, determine based on query type
        if reasoning_strategy == ReasoningStrategyType.DEFAULT.value:
            reasoning_strategy = self._determine_strategy_from_query(
                query_type, complexity
            )

        # Note: Include fallback status in result metadata to make failures visible
        # If rationale contains "Fallback" or "failed", the primary engine failed
        used_fallback = "fallback" in rationale.lower() or "failed" in rationale.lower()
        primary_engine_failed = used_fallback and confidence < CONFIDENCE_HIGH_THRESHOLD
        
        # ==================================================================
        # Note: Confidence threshold handling for medium-confidence results
        # Jan 6 2026 logs: Confidence 0.3-0.6 was triggering unnecessary LLM fallback
        # Instead of dropping to LLM, use the result with a "tentative" flag
        # This prevents good internal reasoning from being discarded
        # ==================================================================
        is_tentative_result = False
        if CONFIDENCE_MEDIUM_THRESHOLD <= confidence < CONFIDENCE_GOOD_THRESHOLD:
            # Medium confidence - use result but mark as tentative
            is_tentative_result = True
            logger.info(
                f"{LOG_PREFIX} FIX: Medium confidence ({confidence:.2f}) result. "
                f"Using internal reasoning with tentative flag instead of LLM fallback. "
                f"Tool: {selected_tools}"
            )
        elif confidence >= CONFIDENCE_GOOD_THRESHOLD:
            # Good confidence - use result normally
            logger.debug(
                f"{LOG_PREFIX} Good confidence ({confidence:.2f}) result. "
                f"Tool: {selected_tools}"
            )
        elif confidence >= CONFIDENCE_LOW_THRESHOLD:
            # Low but acceptable - warn but still use
            is_tentative_result = True
            logger.warning(
                f"{LOG_PREFIX} FIX: Low confidence ({confidence:.2f}) result. "
                f"Using internal reasoning with tentative flag. Tool: {selected_tools}"
            )
        # else: Very low (<CONFIDENCE_LOW_THRESHOLD) - handled by fallback logic above
        
        result_metadata = {
            "query_type": query_type,
            "complexity": complexity,
            "tool_selector_available": self._tool_selector is not None,
            "portfolio_executor_available": self._portfolio_executor is not None,
            # Note: Explicitly track fallback usage
            "used_fallback": used_fallback,
            "primary_engine_failed": primary_engine_failed,
            # Note: Track tentative status for medium-confidence results
            "is_tentative": is_tentative_result,
            "confidence_category": (
                "high" if confidence >= CONFIDENCE_HIGH_THRESHOLD else
                "good" if confidence >= CONFIDENCE_GOOD_THRESHOLD else
                "medium" if confidence >= CONFIDENCE_MEDIUM_THRESHOLD else
                "low" if confidence >= CONFIDENCE_LOW_THRESHOLD else
                "very_low"
            ),
        }
        
        # Note: If primary engine failed, make it clear in the result
        if primary_engine_failed:
            logger.warning(
                f"{LOG_PREFIX} Note: Primary reasoning engine failed. "
                f"Result is from fallback (confidence={confidence:.2f}). "
                f"Original rationale: {rationale}"
            )
            result_metadata["primary_failure_reason"] = rationale

        return ReasoningResult(
            selected_tools=selected_tools,
            reasoning_strategy=reasoning_strategy,
            confidence=confidence,
            rationale=rationale,
            metadata=result_metadata,
        )

    def _process_with_decomposition(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]],
    ) -> ReasoningResult:
        """
        Process a complex query using hierarchical problem decomposition.

        This method is called for queries with complexity >= DECOMPOSITION_COMPLEXITY_THRESHOLD.
        It breaks down the query into subproblems, applies tool selection to each,
        and aggregates the results.

        Processing Flow:
            1. Convert query to ProblemGraph via QueryToProblemBridge
            2. Decompose using ProblemDecomposer (strategies: exact, semantic, structural, etc.)
            3. For each subproblem step, apply ToolSelector
            4. Aggregate results and determine overall strategy

        Args:
            query: The user query text to process
            query_type: Type of query (reasoning, execution, etc.)
            complexity: Query complexity score (0.4 to 1.0)
            context: Optional context dictionary

        Returns:
            ReasoningResult with selected tools, strategy, and decomposition metadata

        Note:
            Falls back to direct tool selection if decomposition fails.
        """
        decomposition_start = time.perf_counter()

        try:
            # Step 1: Convert query to ProblemGraph
            query_analysis = {
                'type': query_type,
                'complexity': complexity,
                'uncertainty': context.get('uncertainty', 0.0) if context else 0.0,
                'requires_reasoning': query_type in ('reasoning', 'causal', 'planning'),
            }

            problem_graph = self._query_bridge.convert_to_problem_graph(
                query=query,
                query_analysis=query_analysis,
                tool_selection=None,  # Will be determined per subproblem
            )

            if problem_graph is None:
                logger.warning(
                    f"{LOG_PREFIX} Query bridge returned None, falling back to direct selection"
                )
                return self._select_with_tool_selector(query, query_type, complexity, context)

            # Step 2: Decompose the problem
            decomposition_plan = self._problem_decomposer.decompose_novel_problem(problem_graph)

            if decomposition_plan is None or len(decomposition_plan.steps) == 0:
                logger.warning(
                    f"{LOG_PREFIX} Decomposition returned empty plan, falling back to direct selection"
                )
                return self._select_with_tool_selector(query, query_type, complexity, context)

            logger.info(
                f"{LOG_PREFIX} Decomposed into {len(decomposition_plan.steps)} steps, "
                f"confidence={decomposition_plan.confidence:.2f}"
            )

            # Step 3: Select tools ONCE based on ORIGINAL query
            # Note: Previously, step descriptions (~28 chars like "Step 1: Parse constraints")
            # were passed to ToolSelector instead of the original query (e.g., 507 chars).
            # This caused semantic matching to fail because it was matching against
            # short step descriptions instead of the actual user query.
            # 
            # The fix: Select tools once based on the original query, then apply those
            # tools to each decomposed step.
            logger.info(
                f"{LOG_PREFIX} Selecting tools based on original query "
                f"(length={len(query)} chars)"
            )
            
            primary_result = self._select_with_tool_selector(
                query=query,  # Use ORIGINAL query, not step descriptions
                query_type=query_type,
                complexity=complexity,
                context=context,
            )
            
            # The tools selected for the original query apply to all steps
            all_tools: set = set(primary_result.selected_tools)
            step_results: List[Dict[str, Any]] = []

            # Record step metadata (without re-running tool selection per step)
            for step in decomposition_plan.steps:
                # Extract step description for metadata only
                if hasattr(step, 'description'):
                    step_description = step.description
                elif hasattr(step, 'to_dict'):
                    step_dict = step.to_dict()
                    step_description = step_dict.get('description', str(step))
                else:
                    step_description = str(step)

                # Extract step complexity for metadata
                if hasattr(step, 'estimated_complexity'):
                    step_complexity = step.estimated_complexity
                elif hasattr(step, 'complexity'):
                    step_complexity = step.complexity
                else:
                    step_complexity = complexity * 0.5  # Default to half of parent

                # Ensure step_complexity is within bounds
                step_complexity = max(0.1, min(1.0, step_complexity))

                # Record step metadata - tools are inherited from primary selection
                step_results.append({
                    'step_id': getattr(step, 'step_id', f'step_{len(step_results)}'),
                    'description': step_description[:100],  # Truncate for metadata
                    'tools': primary_result.selected_tools,  # Inherited from primary
                    'strategy': primary_result.reasoning_strategy,
                    'confidence': primary_result.confidence,
                    'step_complexity': step_complexity,
                })

            # Step 4: Determine overall strategy based on decomposition
            if decomposition_plan.strategy:
                strategy_name = getattr(decomposition_plan.strategy, 'name', 'hierarchical')
            else:
                strategy_name = 'hierarchical_decomposition'

            # Calculate overall confidence
            # Use the primary tool selection confidence combined with decomposition confidence
            num_steps = len(step_results)
            overall_confidence = (decomposition_plan.confidence * 0.4) + (primary_result.confidence * 0.6)

            decomposition_time_ms = (time.perf_counter() - decomposition_start) * 1000

            logger.info(
                f"{LOG_PREFIX} Decomposition complete: "
                f"tools={list(all_tools)}, strategy={strategy_name}, "
                f"confidence={overall_confidence:.2f}, time={decomposition_time_ms:.1f}ms"
            )

            return ReasoningResult(
                selected_tools=list(all_tools) if all_tools else ["general"],
                reasoning_strategy=strategy_name,
                confidence=overall_confidence,
                rationale=f"Hierarchical decomposition into {num_steps} subproblems",
                metadata={
                    "query_type": query_type,
                    "complexity": complexity,
                    "decomposition_path": True,
                    "decomposition_steps": num_steps,
                    "step_results": step_results,
                    "decomposition_confidence": decomposition_plan.confidence,
                    "decomposition_time_ms": decomposition_time_ms,
                    "tool_selector_available": self._tool_selector is not None,
                    "problem_decomposer_available": self._problem_decomposer is not None,
                },
            )

        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} Decomposition processing failed: {e}, "
                f"falling back to direct selection",
                exc_info=True
            )
            # Graceful degradation: fall back to direct tool selection
            return self._select_with_tool_selector(query, query_type, complexity, context)

    def _extract_tools_from_result(self, result: Any) -> List[str]:
        """
        Extract list of tools from SelectionResult.

        SelectionResult has 'selected_tool' (singular str) and 'all_results' (dict).
        We extract tools from all_results if available, otherwise use selected_tool.

        Args:
            result: SelectionResult from ToolSelector

        Returns:
            List of tool names used
        """
        # Try to get tools from all_results dictionary keys
        if hasattr(result, "all_results") and result.all_results:
            tools = list(result.all_results.keys())
            if tools:
                return tools

        # Fall back to selected_tool (singular)
        if hasattr(result, "selected_tool"):
            tool = result.selected_tool
            if tool and tool != "none":
                return [tool]

        # Default fallback
        return ["general"]

    def _get_fallback_tools(
        self,
        query_type: str,
        original_tool: str,
        failed_tools: List[str],
    ) -> List[str]:
        """
        Get appropriate fallback tools based on query type and failed tools.
        
        FIX #4: Improved Fallback Logic
        ===============================
        Instead of a fixed fallback list, select tools based on query characteristics.
        This ensures queries are routed to the most appropriate alternative engine
        before falling back to LLM (Arena delegation).
        
        Priority:
        1. Query-type specific alternatives (e.g., philosophical for ethical queries)
        2. General-purpose fallbacks (world_model for meta-queries, probabilistic)
        3. The fallback list is filtered to exclude already-failed tools
        
        Args:
            query_type: Type of query (reasoning, ethical, mathematical, etc.)
            original_tool: The tool that originally failed
            failed_tools: List of tools that have already been tried and failed
            
        Returns:
            List of fallback tool names to try, in priority order
        """
        # Map query types to preferred fallback tools
        # NOTE: 'philosophical' engine removed - use 'world_model' for ethical reasoning
        query_type_fallbacks = {
            # Ethical/philosophical queries → world_model is primary (has full meta-reasoning)
            'ethical': ['world_model', 'analogical', 'causal'],
            'philosophical': ['world_model', 'analogical', 'causal'],
            
            # Mathematical queries → try mathematical engine first
            'mathematical': ['symbolic', 'probabilistic'],
            'symbolic': ['mathematical', 'probabilistic'],
            
            # Causal queries → try related engines
            'causal': ['probabilistic', 'analogical', 'world_model'],
            
            # Analogical queries → try related engines  
            'analogical': ['causal', 'world_model', 'probabilistic'],
            
            # Probabilistic queries → try related engines
            'probabilistic': ['mathematical', 'causal', 'analogical'],
            
            # Cryptographic queries → try mathematical fallback
            'cryptographic': ['mathematical', 'symbolic'],
            
            # Self-introspection queries → world_model is primary
            'self_introspection': ['world_model', 'analogical'],
            
            # General/reasoning queries → broad fallback
            'reasoning': ['world_model', 'probabilistic', 'analogical'],
            'general': ['world_model', 'probabilistic', 'analogical'],
        }
        
        # Normalize query type
        query_type_lower = query_type.lower() if query_type else 'general'
        
        # Get type-specific fallbacks, or default to general
        fallback_list = query_type_fallbacks.get(
            query_type_lower,
            ['world_model', 'probabilistic', 'analogical']
        ).copy()  # Copy to avoid modifying the dict value
        
        # Ensure we have the general-purpose fallbacks at the end
        # Use set for O(1) membership testing instead of O(n) list lookup
        # NOTE: 'philosophical' removed from defaults - world_model handles ethical reasoning
        default_fallbacks = ['world_model', 'probabilistic', 'analogical', 'mathematical']
        existing_tools = set(fallback_list)
        for tool in default_fallbacks:
            if tool not in existing_tools:
                fallback_list.append(tool)
                existing_tools.add(tool)
        
        # Filter out the tools that have already failed
        failed_set = set(failed_tools) | {original_tool}
        fallback_list = [t for t in fallback_list if t not in failed_set]
        
        # Limit to top 3 fallbacks to prevent excessive retries
        fallback_list = fallback_list[:3]
        
        logger.debug(
            f"{LOG_PREFIX} FIX#4: Selected fallback tools for query_type='{query_type}', "
            f"original_tool='{original_tool}': {fallback_list}"
        )
        
        return fallback_list

    def _predict_tools_for_preprocessing(
        self,
        query: str,
        query_type: str,
    ) -> List[str]:
        """
        Predict which tools will be used for query preprocessing.

        This method provides a quick heuristic-based prediction of tools
        before the full tool selection process runs. It's used to determine
        if query preprocessing should be applied.

        The prediction is based on:
        1. Query type from router
        2. Presence of logical/mathematical operators in query
        3. Keywords indicating specific reasoning domains

        Args:
            query: The query text to analyze
            query_type: Type from router (reasoning, symbolic, etc.)

        Returns:
            List of predicted tool names for preprocessing
        """
        predicted_tools: List[str] = []

        # Map query types to likely tools
        type_to_tools = {
            'symbolic': ['symbolic'],
            'mathematical': ['mathematical'],
            'probabilistic': ['probabilistic'],
            'causal': ['causal'],
            'reasoning': ['symbolic', 'causal'],  # Generic reasoning may use multiple
            'general': ['general'],
        }

        # Start with type-based prediction
        if query_type in type_to_tools:
            predicted_tools.extend(type_to_tools[query_type])

        # Check for logical operators that indicate symbolic reasoning
        logical_indicators = ['→', '∧', '∨', '¬', '∀', '∃', '->', 'AND', 'OR', 'NOT']
        if any(op in query for op in logical_indicators):
            if 'symbolic' not in predicted_tools:
                predicted_tools.append('symbolic')

        # Check for SAT-specific keywords
        sat_keywords = ['propositions', 'constraints', 'satisfiability', 'sat']
        if any(kw in query.lower() for kw in sat_keywords):
            if 'symbolic' not in predicted_tools:
                predicted_tools.append('symbolic')

        # Check for mathematical indicators
        math_indicators = ['formula:', 'equation:', 'prove', 'theorem', '∫', '∑', 'lim']
        if any(ind in query.lower() for ind in math_indicators):
            if 'mathematical' not in predicted_tools:
                predicted_tools.append('mathematical')

        # Check for probabilistic indicators
        prob_indicators = ['P(', 'probability', 'E[', 'expectation', 'distribution']
        if any(ind in query for ind in prob_indicators):
            if 'probabilistic' not in predicted_tools:
                predicted_tools.append('probabilistic')

        # Default to general if nothing matched
        if not predicted_tools:
            predicted_tools = ['general']

        return predicted_tools

    def _determine_selection_mode(self, complexity: float, selection_mode_enum: Any) -> Any:
        """
        Determine the selection mode based on query complexity.

        Args:
            complexity: Query complexity score (0.0 to 1.0)
            selection_mode_enum: SelectionMode enum class

        Returns:
            SelectionMode enum value
        """
        if complexity > HIGH_COMPLEXITY_THRESHOLD:
            return selection_mode_enum.ACCURATE
        elif complexity < LOW_COMPLEXITY_THRESHOLD:
            return selection_mode_enum.FAST
        else:
            return selection_mode_enum.BALANCED

    def _determine_strategy_from_query(
        self,
        query_type: str,
        complexity: float
    ) -> str:
        """
        Determine reasoning strategy based on query characteristics.

        This method implements the fallback strategy selection logic when
        the ToolSelector is unavailable or doesn't provide a strategy.

        Args:
            query_type: Type of query (reasoning, perception, planning, etc.)
            complexity: Query complexity (0.0 to 1.0)

        Returns:
            Strategy name string
        """
        # High complexity reasoning queries use causal reasoning
        if query_type == "reasoning" and complexity > CAUSAL_REASONING_THRESHOLD:
            return ReasoningStrategyType.CAUSAL_REASONING.value

        # Execution tasks use planning
        if query_type == "execution":
            return ReasoningStrategyType.PLANNING.value

        # Medium-high complexity uses probabilistic reasoning
        if complexity > PROBABILISTIC_REASONING_THRESHOLD:
            return ReasoningStrategyType.PROBABILISTIC_REASONING.value

        # Query type specific strategies
        type_strategy = QUERY_TYPE_STRATEGY_MAP.get(query_type)
        if type_strategy:
            return type_strategy

        # Default to direct for simple queries
        return ReasoningStrategyType.DIRECT.value

    def _delegate_to_arena(
        self,
        query: str,
        original_tool: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Delegate reasoning task to Arena when local tools fail.
        
        Arena provides a full reasoning pipeline with evolution/tournaments
        that may succeed where individual tools fail. This is the final
        fallback before returning a low-confidence result.
        
        Args:
            query: The original query text
            original_tool: The tool that initially failed
            query_type: Type of query (reasoning, symbolic, etc.)
            complexity: Query complexity score (0.0 to 1.0)
            context: Optional context dictionary
            
        Returns:
            Dictionary with Arena result if successful, None otherwise
            
        Note:
            This method uses httpx for synchronous HTTP requests to Arena.
            It will not block the event loop in async contexts.
        """
        try:
            import httpx
            
            logger.info(
                f"{LOG_PREFIX} Delegating to Arena: tool={original_tool}, "
                f"query_type={query_type}, complexity={complexity:.2f}"
            )
            
            # Build request payload
            # Note: Sanitize context to make it JSON serializable
            # The context may contain PreprocessingResult objects which aren't
            # JSON serializable. Convert them to dictionaries using to_dict().
            sanitized_context = self._sanitize_context_for_json(context or {})
            
            arena_payload = {
                "query": query,
                "selected_tools": [original_tool],
                "query_type": query_type,
                "complexity": complexity,
                "context": {
                    **sanitized_context,
                    'vulcan_fallback': True,
                    'original_tool': original_tool,
                },
            }
            
            # Get API key from environment
            # Note: "internal-bypass" is used for internal service-to-service calls
            # when both VULCAN and Arena run in the same trusted environment
            api_key = os.environ.get("GRAPHIX_API_KEY")
            if not api_key:
                # For internal delegation, use a special bypass key
                # This should only work when Arena is configured to accept it
                api_key = "internal-vulcan-delegation"
                logger.debug(
                    f"{LOG_PREFIX} Using internal delegation key for Arena"
                )
            
            # Make request to Arena
            response = httpx.post(
                ARENA_REASONING_URL,
                json=arena_payload,
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=ARENA_DELEGATION_TIMEOUT,
            )
            
            if response.status_code == 200:
                try:
                    arena_result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(
                        f"{LOG_PREFIX} Arena returned invalid JSON: {e}. "
                        f"Response: {response.text[:200]}"
                    )
                    return None
                    
                result_data = arena_result.get('result', {})
                
                logger.info(
                    f"{LOG_PREFIX} Arena delegation successful: "
                    f"confidence={result_data.get('confidence', 'N/A')}"
                )
                
                return {
                    'conclusion': result_data.get('conclusion'),
                    'confidence': result_data.get('confidence', 0.5),
                    'explanation': result_data.get('explanation'),
                    'arena_fallback': True,
                    'original_tool': original_tool,
                }
            else:
                # FIX: More descriptive error for HTTP errors
                logger.error(
                    f"{LOG_PREFIX} Arena HTTP error {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None
                
        except ImportError:
            logger.warning(
                f"{LOG_PREFIX} httpx not available for Arena delegation. "
                f"Install with: pip install httpx"
            )
            return None
        except httpx.ConnectTimeout:
            logger.error(
                f"{LOG_PREFIX} Arena connection timed out after "
                f"{ARENA_DELEGATION_TIMEOUT}s"
            )
            return None
        except httpx.ReadTimeout:
            logger.error(
                f"{LOG_PREFIX} Arena read timed out after "
                f"{ARENA_DELEGATION_TIMEOUT}s"
            )
            return None
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Arena delegation failed: {e}", exc_info=True)
            return None
    
    def _sanitize_context_for_json(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Note: Sanitize context dictionary to make it JSON serializable.
        
        The context may contain objects like PreprocessingResult that have
        to_dict() methods. This function recursively converts such objects
        to plain dictionaries.
        
        Args:
            context: Original context dictionary
            
        Returns:
            Sanitized context dictionary that is JSON serializable
        """
        if not context:
            return {}
        
        def sanitize_value(value: Any) -> Any:
            """Recursively sanitize a value for JSON serialization."""
            # Handle None
            if value is None:
                return None
            
            # Handle primitives
            if isinstance(value, (bool, int, float, str)):
                return value
            
            # Handle objects with to_dict() method (e.g., PreprocessingResult)
            if hasattr(value, 'to_dict') and callable(value.to_dict):
                try:
                    return value.to_dict()
                except Exception as e:
                    logger.warning(
                        f"{LOG_PREFIX} Failed to serialize object with to_dict(): {e}"
                    )
                    return str(value)
            
            # Handle dataclasses with __dataclass_fields__
            if hasattr(value, '__dataclass_fields__'):
                try:
                    return dataclasses.asdict(value)
                except Exception as e:
                    logger.warning(
                        f"{LOG_PREFIX} Failed to serialize dataclass: {e}"
                    )
                    return str(value)
            
            # Handle dictionaries recursively
            if isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            
            # Handle lists and tuples recursively
            if isinstance(value, (list, tuple)):
                return [sanitize_value(item) for item in value]
            
            # Handle sets (convert to list)
            if isinstance(value, (set, frozenset)):
                return [sanitize_value(item) for item in value]
            
            # Handle Enum - use isinstance for robust detection
            # Enum is already imported at module level (from enum import Enum)
            if isinstance(value, Enum):
                return value.value
            
            # Fallback: convert to string
            try:
                return str(value)
            except Exception:
                return repr(value)
        
        return sanitize_value(context)

    # =========================================================================
    # Note: Self-Referential Query Handling
    # =========================================================================
    
    def _is_self_referential(self, query: str) -> bool:
        """
        Check if query is a PURE meta-description about VULCAN itself.
        
        GAP 1 FIX: Critical distinction between:
        - PURE META-DESCRIPTION: "What is VULCAN?" "Who created you?" → world_model only
        - META-ANALYSIS: "What's wrong with your causal reasoning?" → specialized tools + commentary
        
        The system was conflating these two types, causing queries that LOOK
        self-referential but require actual analysis to bypass specialized reasoning.
        
        Examples that ARE pure meta-description (return True):
        - "What are you?"
        - "Who created you?"
        - "What can your reasoning modules do?"
        - "Are you sentient?"
        
        Examples that are NOT pure meta-description (return False):
        - "Which causal link is weakest?" → Needs causal analysis
        - "Identify one step that could be wrong" → Needs step analysis
        - "What's wrong with your reasoning on X?" → Needs domain analysis
        - "If we intervene on variable X..." → Needs causal analysis
        
        Args:
            query: The query string to analyze
            
        Returns:
            True ONLY if query is a pure meta-description about VULCAN itself
        """
        if not query:
            return False
            
        query_lower = query.lower()
        
        # =====================================================================
        # CREATIVE INDICATORS - These are NOT self-referential!
        # Check for creative writing requests FIRST before checking self-reference.
        # =====================================================================
        
        creative_words = [
            'write', 'poem', 'story', 'compose', 'create',
            'imagine', 'narrative', 'fiction', 'invent', 'draft', 'author'
        ]
        
        creative_phrases = [
            'tell me a', 'make up', 'write me', 'create a'
        ]
        
        for word in creative_words:
            if re.search(rf'\b{word}\b', query_lower):
                return False
        
        if any(phrase in query_lower for phrase in creative_phrases):
            return False
        
        # =====================================================================
        # GAP 1 FIX: META-ANALYSIS INDICATORS - Queries that LOOK self-referential
        # but actually require domain-specific reasoning.
        # 
        # These queries contain "your" or "you" but are asking for ANALYSIS,
        # not just description of capabilities.
        #
        # FIX: Distinguish between asking ABOUT VULCAN's analysis capabilities
        # vs asking VULCAN to PERFORM analysis on external data.
        # =====================================================================
        
        # Check if query is directed AT VULCAN (about its own capabilities/state)
        vulcan_directed_indicators = [
            'your ', 'your\n', 'you ', 'you?', "you'", 'yourself',
            'vulcan', 'about you', 'tell me about', 'describe your',
        ]
        is_about_vulcan = any(ind in query_lower for ind in vulcan_directed_indicators)
        
        # If query has analysis indicators AND is about VULCAN, it's META-ANALYSIS
        # e.g., "What are YOUR weaknesses?" → This IS self-referential (about VULCAN)
        # vs "What is the weakest causal link in this data?" → NOT self-referential
        if any(indicator in query_lower for indicator in ANALYSIS_INDICATORS):
            if is_about_vulcan:
                # FIX: Query is asking about VULCAN's own analysis/weaknesses/etc
                # This IS self-referential - should use world_model
                logger.debug(
                    f"{LOG_PREFIX} META-ANALYSIS about VULCAN detected - "
                    f"treating as self-referential (world_model)"
                )
                # Don't return False - let it fall through to meta-description check
            else:
                # Query has analysis indicators but NOT about VULCAN
                # This needs specialized tools
                logger.debug(
                    f"{LOG_PREFIX} GAP 1 FIX: Query contains analysis indicators - "
                    f"NOT treating as pure meta-description"
                )
                return False
        
        # =====================================================================
        # PURE META-DESCRIPTION PATTERNS
        # Only match these TIGHT patterns for genuine self-description queries
        # =====================================================================
        
        # Pure meta-description phrases (very specific about VULCAN itself)
        pure_meta_phrases = [
            # Identity questions
            "what are you", "who are you", "who created you", "what is vulcan",
            # Pure capability description (not analysis of capabilities)
            "what can you do", "what are your capabilities", "list your abilities",
            "what tools do you have", "what modules do you have",
            # Self-awareness questions
            "are you sentient", "are you conscious", "are you self-aware",
            "do you have feelings", "are you alive",
            # Architecture description
            "how do you work", "how are you built", "how were you created",
            "what is your architecture", "describe your design",
            # Preferences (pure description)
            "what do you like", "what do you prefer", "what is your favorite",
        ]
        
        # Check for exact phrase matches (more restrictive)
        if any(phrase in query_lower for phrase in pure_meta_phrases):
            logger.debug(
                f"{LOG_PREFIX} Pure meta-description detected - routing to world_model"
            )
            return True
        
        # If query has ONLY generic self-reference ("your", "you") without
        # analysis indicators, check if it's asking ABOUT VULCAN vs asking
        # VULCAN to analyze something
        
        # Uses module-level ACTION_VERBS constant for maintainability
        has_action_verb = any(verb in query_lower for verb in ACTION_VERBS)
        
        # If there's an action verb, this is asking VULCAN to DO something
        # (analysis), not asking ABOUT VULCAN
        if has_action_verb:
            logger.debug(
                f"{LOG_PREFIX} GAP 1 FIX: Query asks VULCAN to perform action - "
                f"NOT pure meta-description"
            )
            return False
        
        # Default: Only return True for very restrictive self-reference
        # This is the conservative approach - when in doubt, use specialized tools
        return False
    
    def _is_ethical_query(self, query: str) -> bool:
        """
        Detect ethical queries that should use world model's ethical framework.
        
        GAP 4 FIX: More restrictive detection to prevent world model fallback trap.
        
        The world model should ONLY be used for PURE ethical/deontic reasoning where
        specialized tools cannot help. For queries that LOOK ethical but actually
        need analysis (e.g., "Two core values conflict" → needs analysis of the
        conflict, not just ethical framework description), use specialized tools.
        
        Examples that ARE pure ethical (return True):
        - "Is it morally permissible to lie to save a life?"
        - "What would a utilitarian say about this?"
        - "Trolley problem: should I pull the lever?"
        
        Examples that are NOT pure ethical (return False):
        - "Two core values conflict. What breaks?" → Needs conflict analysis
        - "Analyze the ethical implications of X" → Needs domain analysis + ethics
        - "What harm might this cause?" → Needs domain-specific harm analysis
        
        Args:
            query: The query string to analyze
            
        Returns:
            True ONLY if query is a pure ethical/deontic reasoning question
        """
        if not query:
            return False
            
        query_lower = query.lower()
        
        # GAP 4 FIX: Analysis indicators that mean we need specialized tools,
        # not just world model ethical framework
        # Uses module-level ETHICAL_ANALYSIS_INDICATORS constant for maintainability
        if any(indicator in query_lower for indicator in ETHICAL_ANALYSIS_INDICATORS):
            logger.debug(
                f"{LOG_PREFIX} GAP 4 FIX: Query contains analysis indicators - "
                f"NOT treating as pure ethical query"
            )
            return False
        
        # Pure ethical keywords that indicate deontic/ethical framework questions
        # Uses module-level PURE_ETHICAL_PHRASES constant for maintainability
        if any(phrase in query_lower for phrase in PURE_ETHICAL_PHRASES):
            logger.debug(
                f"{LOG_PREFIX} Pure ethical query detected - routing to world model ethical framework"
            )
            return True
        
        # Single ethical keywords are NOT sufficient anymore (GAP 4 FIX)
        # They need to be in an obviously ethical context
        # This prevents "harm" in "What harm might the algorithm cause?" from
        # triggering world model fallback
        
        return False
    
    def _consult_world_model_introspection(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Consult the world model's introspection system for self-referential queries.
        
        Route self-awareness queries to the world model which maintains
        VULCAN's sense of "self" and can answer questions about capabilities,
        preferences, and limitations.
        
        Args:
            query: The self-referential query
            
        Returns:
            Introspection result from world model, or None if unavailable
        """
        try:
            # Try to access world model
            # The world model might be accessible through different paths
            world_model = None
            
            # Path 1: Direct attribute
            if hasattr(self, 'world_model') and self.world_model is not None:
                world_model = self.world_model
            
            # Path 2: Through tool_selector
            elif hasattr(self, '_tool_selector') and self._tool_selector is not None:
                if hasattr(self._tool_selector, 'world_model'):
                    world_model = self._tool_selector.world_model
            
            # Path 3: Use cached world model or create one (avoid repeated initialization)
            if world_model is None:
                # Check for cached world model
                if hasattr(self, '_cached_world_model') and self._cached_world_model is not None:
                    world_model = self._cached_world_model
                else:
                    try:
                        from vulcan.world_model.world_model_core import create_world_model
                        # Use minimal config to avoid heavy initialization
                        world_model = create_world_model({
                            "enable_meta_reasoning": True,
                            "enable_self_improvement": False,
                        })
                        # Cache for future use
                        self._cached_world_model = world_model
                    except ImportError:
                        logger.debug(f"{LOG_PREFIX} Could not import world model for introspection")
                        return None
            
            # Check if world model has introspect method
            if world_model is not None and hasattr(world_model, 'introspect'):
                result = world_model.introspect(query)
                logger.info(
                    f"{LOG_PREFIX} World model introspection returned confidence={result.get('confidence', 0)}"
                )
                return result
            else:
                logger.debug(f"{LOG_PREFIX} World model does not have introspect method")
                return None
                
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} World model introspection failed: {e}")
            return None

    def _create_default_result(
        self,
        query_type: str,
        complexity: float
    ) -> ReasoningResult:
        """
        Create a default ReasoningResult for fallback scenarios.

        Args:
            query_type: Type of query
            complexity: Query complexity

        Returns:
            Default ReasoningResult
        """
        strategy = self._determine_strategy_from_query(query_type, complexity)

        # Note: Adjust confidence based on complexity
        # High-complexity queries that fall back should have lower confidence
        # because we couldn't handle them properly
        if complexity >= 0.7:
            fallback_confidence = 0.3  # Low confidence for high-complexity fallbacks
        elif complexity >= 0.5:
            fallback_confidence = 0.4  # Medium-low confidence
        else:
            fallback_confidence = 0.5  # Default confidence for low-complexity

        return ReasoningResult(
            selected_tools=["general"],
            reasoning_strategy=strategy,
            confidence=fallback_confidence,
            rationale="Fallback to default strategy",
            metadata={
                "query_type": query_type,
                "complexity": complexity,
                "fallback": True,
            },
        )

    def _record_selection_time(self, time_ms: float) -> None:
        """
        Record selection timing for performance monitoring.

        Maintains a rolling window of timing samples for average calculation.

        Args:
            time_ms: Selection time in milliseconds
        """
        with self._stats_lock:
            self._selection_times.append(time_ms)

            # Maintain rolling window
            if len(self._selection_times) > MAX_TIMING_SAMPLES:
                self._selection_times.pop(0)

            # Update average
            if self._selection_times:
                self._stats.avg_selection_time_ms = (
                    sum(self._selection_times) / len(self._selection_times)
                )

    def run_portfolio(
        self,
        query: str,
        tools: List[str],
        strategy: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run portfolio execution for complex queries requiring multiple tools.

        Executes multiple tools using the specified strategy for queries that
        benefit from diverse reasoning approaches. The portfolio executor
        coordinates parallel or sequential execution based on the strategy.

        Args:
            query: The user query to process
            tools: List of tool names to use. Available tools include:
                - "symbolic": Symbolic/logical reasoning
                - "probabilistic": Statistical inference
                - "causal": Causal reasoning and inference
                - "analogical": Pattern matching and analogy
                - "multimodal": Multi-modal processing
            strategy: Execution strategy name. Options include:
                - "causal_reasoning": Sequential refinement
                - "probabilistic_reasoning": Speculative parallel
                - "analogical_reasoning": Cascade
                - "planning": Sequential refinement
                - "deliberative": Committee consensus
                - "direct": Single tool execution
            constraints: Optional execution constraints:
                - time_budget_ms: Maximum execution time
                - energy_budget_mj: Maximum energy budget
                - min_confidence: Minimum result confidence

        Returns:
            Dictionary with execution results:
                - status: "success", "skipped", or "error"
                - strategy_used: Actual strategy applied
                - tools_used: List of tools that executed
                - execution_time_ms: Total execution time
                - confidence: Result confidence score
                - error: Error message if status is "error"

        Example:
            >>> result = integration.run_portfolio(
            ...     query="Complex multi-step problem",
            ...     tools=["symbolic", "causal"],
            ...     strategy="causal_reasoning"
            ... )
            >>> print(f"Status: {result['status']}")
        """
        # Check shutdown state
        with self._shutdown_lock:
            if self._shutdown:
                return {"status": "skipped", "reason": "integration_shutdown"}

        # Initialize components if needed
        self._init_components()

        # Check if portfolio executor is available
        if self._portfolio_executor is None:
            logger.warning(f"{LOG_PREFIX} PortfolioExecutor not available")
            return {"status": "skipped", "reason": "executor_unavailable"}

        try:
            from vulcan.reasoning.selection.portfolio_executor import (
                ExecutionStrategy,
                ExecutionMonitor,
            )

            # Map strategy string to enum
            exec_strategy = self._map_strategy_to_execution(strategy, ExecutionStrategy)

            # Build constraints
            merged_constraints = {
                "time_budget_ms": DEFAULT_TIME_BUDGET_MS,
                "energy_budget_mj": DEFAULT_ENERGY_BUDGET_MJ,
                "min_confidence": DEFAULT_MIN_CONFIDENCE,
            }
            if constraints:
                merged_constraints.update(constraints)

            # Create execution monitor
            monitor = ExecutionMonitor(
                time_budget_ms=merged_constraints["time_budget_ms"],
                energy_budget_mj=merged_constraints["energy_budget_mj"],
                min_confidence=merged_constraints["min_confidence"],
            )

            # Execute portfolio
            exec_start = time.perf_counter()
            result = self._portfolio_executor.execute(
                strategy=exec_strategy,
                tool_names=tools,
                problem=query,
                constraints=merged_constraints,
                monitor=monitor,
            )
            exec_time = (time.perf_counter() - exec_start) * 1000

            # Track execution
            with self._stats_lock:
                self._stats.portfolio_executions += 1

            logger.info(
                f"{LOG_PREFIX} Portfolio execution complete: "
                f"strategy={result.strategy.value}, "
                f"tools={result.tools_used}, "
                f"time={exec_time:.1f}ms"
            )

            return {
                "status": "success",
                "strategy_used": result.strategy.value,
                "tools_used": result.tools_used,
                "execution_time_ms": exec_time,
                "confidence": (
                    result.consensus_confidence
                    if result.consensus_confidence is not None
                    else 0.5
                ),
                "primary_result": result.primary_result,
                "all_results": result.all_results,
            }

        except ImportError as e:
            logger.warning(f"{LOG_PREFIX} Portfolio imports unavailable: {e}")
            return {"status": "error", "error": f"Import error: {e}"}

        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} Portfolio execution failed: {e}",
                exc_info=True
            )
            with self._stats_lock:
                self._stats.errors += 1
                self._stats.last_error = str(e)

            return {"status": "error", "error": str(e)}

    def _map_strategy_to_execution(
        self,
        strategy: str,
        execution_strategy_enum: Any
    ) -> Any:
        """
        Map strategy string to ExecutionStrategy enum value.

        Args:
            strategy: Strategy name string
            execution_strategy_enum: ExecutionStrategy enum class

        Returns:
            ExecutionStrategy enum value
        """
        strategy_map = {
            "causal_reasoning": execution_strategy_enum.SEQUENTIAL_REFINEMENT,
            "probabilistic_reasoning": execution_strategy_enum.SPECULATIVE_PARALLEL,
            "analogical_reasoning": execution_strategy_enum.CASCADE,
            "planning": execution_strategy_enum.SEQUENTIAL_REFINEMENT,
            "deliberative": execution_strategy_enum.COMMITTEE_CONSENSUS,
            "direct": execution_strategy_enum.SINGLE,
        }

        return strategy_map.get(strategy, execution_strategy_enum.ADAPTIVE_MIX)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics for monitoring and observability.

        Returns a comprehensive dictionary of statistics about the reasoning
        integration's performance and health.

        Returns:
            Dictionary containing:
                - initialized: Whether components are initialized
                - tool_selector_available: ToolSelector availability
                - portfolio_executor_available: PortfolioExecutor availability
                - invocations: Total reasoning invocations
                - tool_selections: Successful tool selections
                - portfolio_executions: Portfolio execution count
                - errors: Error count
                - success_rate: Success rate (0.0 to 1.0)
                - fast_path_count: Fast path usage count
                - avg_selection_time_ms: Average selection time
                - last_error: Last error message if any

        Example:
            >>> stats = integration.get_statistics()
            >>> print(f"Success rate: {stats['success_rate']:.1%}")
            >>> print(f"Avg selection time: {stats['avg_selection_time_ms']:.1f}ms")
        """
        with self._stats_lock:
            return {
                "initialized": self._initialized,
                "tool_selector_available": self._tool_selector is not None,
                "portfolio_executor_available": self._portfolio_executor is not None,
                "invocations": self._stats.invocations,
                "tool_selections": self._stats.tool_selections,
                "portfolio_executions": self._stats.portfolio_executions,
                "errors": self._stats.errors,
                "success_rate": self._stats.success_rate,
                "fast_path_count": self._stats.fast_path_count,
                "avg_selection_time_ms": self._stats.avg_selection_time_ms,
                "last_error": self._stats.last_error,
            }

    def reset_statistics(self) -> None:
        """
        Reset all statistics to initial values.

        Useful for testing or when starting a new monitoring period.
        """
        with self._stats_lock:
            self._stats = IntegrationStatistics()
            self._selection_times.clear()

        logger.info(f"{LOG_PREFIX} Statistics reset")

    def apply_cross_domain_transfer(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        selected_tools: List[str],
    ) -> Dict[str, Any]:
        """
        Apply cross-domain knowledge transfer using SemanticBridge.
        
        This method enables knowledge learned in one domain to be applied
        in related domains, improving reasoning quality for queries that
        span multiple conceptual areas.
        
        Processing Flow:
            1. Identify domains involved from selected tools
            2. Determine primary domain based on query type
            3. Find applicable concepts from SemanticBridge
            4. Validate transfer compatibility between domains
            5. Execute transfers for compatible concepts
            6. Record transfer for learning
        
        Args:
            query: The query string being processed
            query_analysis: Analysis results with type, complexity, etc.
            selected_tools: List of tools selected for this query
            
        Returns:
            Dictionary containing:
                - success: Whether transfer was successful
                - domains: List of domains involved
                - primary_domain: Identified primary domain
                - transferred_concepts: List of transferred concept info
                - transfer_count: Number of concepts transferred
                - error: Error message if failed
                
        Example:
            >>> result = integration.apply_cross_domain_transfer(
            ...     query="What causes X given Y?",
            ...     query_analysis={'type': 'reasoning', 'complexity': 0.6},
            ...     selected_tools=['causal', 'probabilistic']
            ... )
            >>> print(result['transfer_count'])
            2
        """
        # Ensure components are initialized
        self._init_components()
        
        # Validate prerequisites
        if self._semantic_bridge is None:
            logger.debug(f"{LOG_PREFIX} SemanticBridge not available for cross-domain transfer")
            return {
                'success': False,
                'error': 'semantic_bridge_unavailable',
                'domains': [],
            }
        
        if self._domain_bridge is None:
            logger.debug(f"{LOG_PREFIX} DomainBridge not available for cross-domain transfer")
            return {
                'success': False,
                'error': 'domain_bridge_unavailable',
                'domains': [],
            }
        
        transfer_start = time.perf_counter()
        
        try:
            # Step 1: Get domains involved
            domains = self._domain_bridge.get_domains_for_tools(selected_tools)
            
            # FIX: Early exit if only one domain - no cross-domain transfer possible
            if len(domains) < 2:
                logger.debug(
                    f"{LOG_PREFIX} Single domain query - cross-domain transfer not applicable"
                )
                return {
                    'success': False,
                    'error': 'single_domain_query',
                    'domains': list(domains),
                    'transfer_count': 0,
                }
            
            # Step 2: Identify primary domain
            query_type = query_analysis.get('type', 'general')
            primary_domain = self._domain_bridge.identify_primary_domain(
                selected_tools, query_type
            )
            
            logger.info(
                f"{LOG_PREFIX} Cross-domain transfer: domains={domains}, "
                f"primary={primary_domain}"
            )
            
            # Step 3: Get applicable concepts from primary domain
            applicable_concepts = []
            try:
                applicable_concepts = self._semantic_bridge.get_applicable_concepts(
                    domain=primary_domain,
                    min_confidence=0.6,
                )
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Failed to get applicable concepts: {e}")
            
            # Step 4: Try to transfer concepts from related domains
            transferred = []
            for source_domain in domains:
                if source_domain == primary_domain:
                    continue
                
                # Check if transfer is possible
                if not self._domain_bridge.can_transfer_between(source_domain, primary_domain):
                    continue
                
                # Get source domain concepts
                try:
                    source_concepts = self._semantic_bridge.get_applicable_concepts(
                        domain=source_domain,
                        min_confidence=0.5,
                    )
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Failed to get concepts from {source_domain}: {e}")
                    continue
                
                # Validate and transfer each concept (limit to top 3)
                for concept in source_concepts[:3]:
                    try:
                        # Validate compatibility
                        compatibility = self._semantic_bridge.validate_transfer_compatibility(
                            concept=concept,
                            source=source_domain,
                            target=primary_domain,
                        )
                        
                        if not compatibility.is_compatible():
                            # Log why transfer was rejected for debugging
                            concept_id = getattr(concept, 'concept_id', str(concept)[:20])
                            logger.debug(
                                f"{LOG_PREFIX} Transfer rejected for {concept_id}: "
                                f"score={compatibility.compatibility_score:.2f}, "
                                f"risks={compatibility.risks}"
                            )
                            continue
                        
                        # Execute transfer
                        transferred_concept = self._semantic_bridge.transfer_concept(
                            concept=concept,
                            source_domain=source_domain,
                            target_domain=primary_domain,
                        )
                        
                        if transferred_concept is not None:
                            concept_id = getattr(concept, 'concept_id', str(concept)[:20])
                            transferred.append({
                                'concept_id': concept_id,
                                'source': source_domain,
                                'target': primary_domain,
                                'confidence': compatibility.confidence,
                            })
                            logger.debug(
                                f"{LOG_PREFIX} Transferred concept from "
                                f"{source_domain} → {primary_domain}"
                            )
                            
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Concept transfer failed: {e}")
                        continue
            
            # Record transfer in domain bridge
            if transferred:
                # Note: Safe set subtraction - handle edge cases
                # If domains has exactly one element equal to primary_domain,
                # (domains - {primary_domain}) is empty and [0] would raise IndexError
                other_domains = list(domains - {primary_domain})
                source_domain = other_domains[0] if other_domains else 'unknown'
                self._domain_bridge.record_transfer(
                    source_domain=source_domain,
                    target_domain=primary_domain,
                    success=True,
                    concepts_transferred=len(transferred),
                )
            
            transfer_time_ms = (time.perf_counter() - transfer_start) * 1000
            
            logger.info(
                f"{LOG_PREFIX} Cross-domain transfer complete: "
                f"transferred={len(transferred)}, time={transfer_time_ms:.1f}ms"
            )
            
            return {
                'success': True,
                'domains': list(domains),
                'primary_domain': primary_domain,
                'applicable_concepts': len(applicable_concepts),
                'transferred_concepts': transferred,
                'transfer_count': len(transferred),
                'transfer_time_ms': transfer_time_ms,
            }
            
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Cross-domain transfer failed: {e}")
            # Note: Properly check if domains variable exists
            # Use 'domains' in locals() to check if local variable is defined
            domains_list = list(locals().get('domains', set()) or [])
            return {
                'success': False,
                'error': str(e),
                'domains': domains_list,
            }

    def learn_from_outcome(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        selected_tools: List[str],
        success: bool,
        execution_time: float,
    ) -> None:
        """
        Learn from reasoning outcome using SemanticBridge.
        
        After successful query execution, this method creates a pattern
        from the outcome and adds it to the SemanticBridge for future
        cross-domain transfer.
        
        Args:
            query: Original query string
            query_analysis: Query analysis results
            selected_tools: Tools that were used
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            
        Example:
            >>> integration.learn_from_outcome(
            ...     query="What causes X?",
            ...     query_analysis={'type': 'reasoning', 'complexity': 0.6},
            ...     selected_tools=['causal'],
            ...     success=True,
            ...     execution_time=1.5
            ... )
        """
        # Only learn from successful outcomes
        if not success:
            return
        
        # Ensure components are initialized
        self._init_components()
        
        if self._semantic_bridge is None or self._domain_bridge is None:
            return
        
        try:
            # Get domain information
            domains = self._domain_bridge.get_domains_for_tools(selected_tools)
            primary_domain = self._domain_bridge.identify_primary_domain(
                selected_tools,
                query_analysis.get('type', 'general'),
            )
            
            # Create pattern outcome for learning
            from vulcan.semantic_bridge import PatternOutcome
            
            # Use deterministic SHA-256 hash for pattern ID (hash() is not deterministic across runs)
            pattern_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
            
            outcome = PatternOutcome(
                pattern_id=f"query_{pattern_hash}",
                success=success,
                domain=primary_domain,
                execution_time=execution_time,
                tools=selected_tools,
                complexity=query_analysis.get('complexity', 0.5),
            )
            
            # Create pattern from query characteristics
            pattern = {
                'query_type': query_analysis.get('type', 'general'),
                'complexity': query_analysis.get('complexity', 0.0),
                'tools': selected_tools,
                'domains': list(domains),
            }
            
            # Learn concept from pattern
            concept = self._semantic_bridge.learn_concept_from_pattern(
                pattern=pattern,
                outcomes=[outcome],
            )
            
            if concept:
                logger.debug(
                    f"{LOG_PREFIX} Learned concept in domain {primary_domain}"
                )
                
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to learn from outcome: {e}")

    def _learn_from_reasoning_outcome(
        self,
        query: str,
        query_type: str,
        complexity: float,
        selected_tools: List[str],
        reasoning_strategy: str,
        success: bool,
        confidence: float,
        execution_time: float,
        preprocessing_applied: bool = False,
    ) -> None:
        """
        Learn from successful reasoning outcomes using KnowledgeCrystallizer.

        This method is called after successful reasoning to extract reusable
        principles that can improve future query processing. It integrates
        with the KnowledgeCrystallizer to store patterns like:
        - "SAT queries with propositions + constraints need preprocessing"
        - "High-complexity ethical queries need philosophical reasoning"
        - "Mathematical proofs require step-by-step validation"

        Args:
            query: Original query text
            query_type: Type of query (symbolic, reasoning, etc.)
            complexity: Query complexity score (0.0 to 1.0)
            selected_tools: Tools that were used for this query
            reasoning_strategy: Strategy that was applied
            success: Whether reasoning succeeded
            confidence: Confidence in the result (0.0 to 1.0)
            execution_time: Time taken in seconds
            preprocessing_applied: Whether query preprocessing was needed

        Note:
            This method is designed to be non-blocking and non-critical.
            Failures are logged but do not affect the main reasoning pipeline.
        """
        # Only learn from successful outcomes with sufficient confidence
        if not success or confidence < 0.7:
            logger.debug(
                f"{LOG_PREFIX} Skipping crystallizer learning: "
                f"success={success}, confidence={confidence:.2f}"
            )
            return

        try:
            # Try to import KnowledgeCrystallizer
            from vulcan.knowledge_crystallizer import (
                KnowledgeCrystallizer,
                ExecutionTrace,
                KNOWLEDGE_CRYSTALLIZER_AVAILABLE,
            )

            if not KNOWLEDGE_CRYSTALLIZER_AVAILABLE or KnowledgeCrystallizer is None:
                logger.debug(f"{LOG_PREFIX} KnowledgeCrystallizer not available")
                return

            # Create execution trace for crystallization
            trace_id = hashlib.sha256(
                f"{query}:{time.time()}".encode()
            ).hexdigest()[:12]

            trace = ExecutionTrace(
                trace_id=trace_id,
                actions=[
                    {
                        'type': 'tool_selection',
                        'tools': selected_tools,
                        'strategy': reasoning_strategy,
                    },
                    {
                        'type': 'preprocessing',
                        'applied': preprocessing_applied,
                    },
                ],
                outcomes={
                    'success': success,
                    'confidence': confidence,
                    'execution_time': execution_time,
                },
                context={
                    'query_type': query_type,
                    'complexity': complexity,
                    'query_length': len(query),
                },
                success=success,
                domain=query_type,
                metadata={
                    'preprocessing_required': preprocessing_applied,
                    'tools_used': selected_tools,
                    'strategy': reasoning_strategy,
                },
            )

            # Get or create crystallizer instance (lazy initialization)
            if not hasattr(self, '_knowledge_crystallizer') or self._knowledge_crystallizer is None:
                self._knowledge_crystallizer = KnowledgeCrystallizer()
                logger.info(f"{LOG_PREFIX} KnowledgeCrystallizer initialized for learning")

            # Crystallize knowledge from the trace
            # Use incremental mode for single-trace learning
            from vulcan.knowledge_crystallizer import CrystallizationMode

            crystallization_result = self._knowledge_crystallizer.crystallize(
                traces=[trace],
                mode=CrystallizationMode.INCREMENTAL,
            )

            if crystallization_result and crystallization_result.principles:
                logger.info(
                    f"{LOG_PREFIX} Extracted {len(crystallization_result.principles)} "
                    f"principles from successful reasoning"
                )
            else:
                logger.debug(f"{LOG_PREFIX} No new principles extracted from trace")

        except ImportError:
            logger.debug(f"{LOG_PREFIX} KnowledgeCrystallizer module not available")
        except Exception as e:
            # Log but don't fail - learning is non-critical
            logger.debug(f"{LOG_PREFIX} Crystallizer learning failed: {e}")

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Shutdown reasoning components gracefully.

        Releases resources held by the ToolSelector and PortfolioExecutor.
        After shutdown, the integration can no longer be used.

        Args:
            timeout: Maximum time to wait for shutdown in seconds

        Example:
            >>> integration.shutdown(timeout=10.0)
        """
        with self._shutdown_lock:
            if self._shutdown:
                logger.warning(f"{LOG_PREFIX} Already shutdown")
                return
            self._shutdown = True

        logger.info(f"{LOG_PREFIX} Starting shutdown (timeout={timeout}s)")

        shutdown_start = time.perf_counter()

        # Shutdown ToolSelector
        if self._tool_selector is not None:
            try:
                if hasattr(self._tool_selector, "shutdown"):
                    remaining = timeout - (time.perf_counter() - shutdown_start)
                    self._tool_selector.shutdown(timeout=max(0.1, remaining))
                    logger.debug(f"{LOG_PREFIX} ToolSelector shutdown complete")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} ToolSelector shutdown error: {e}")

        # Shutdown PortfolioExecutor
        if self._portfolio_executor is not None:
            try:
                if hasattr(self._portfolio_executor, "shutdown"):
                    remaining = timeout - (time.perf_counter() - shutdown_start)
                    self._portfolio_executor.shutdown(timeout=max(0.1, remaining))
                    logger.debug(f"{LOG_PREFIX} PortfolioExecutor shutdown complete")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} PortfolioExecutor shutdown error: {e}")

        shutdown_time = (time.perf_counter() - shutdown_start) * 1000
        logger.info(f"{LOG_PREFIX} Shutdown complete in {shutdown_time:.1f}ms")


# =============================================================================
# Global Singleton Management
# =============================================================================

# Global singleton instance
_reasoning_integration: Optional[ReasoningIntegration] = None
_integration_lock = threading.Lock()


def get_reasoning_integration(
    config: Optional[Dict[str, Any]] = None
) -> ReasoningIntegration:
    """
    Get or create the global reasoning integration singleton.

    Uses double-checked locking pattern for thread-safe lazy initialization.

    Args:
        config: Optional configuration dictionary (only used on first call)

    Returns:
        ReasoningIntegration singleton instance

    Example:
        >>> integration = get_reasoning_integration()
        >>> result = integration.apply_reasoning(...)
    """
    global _reasoning_integration

    if _reasoning_integration is None:
        with _integration_lock:
            if _reasoning_integration is None:
                _reasoning_integration = ReasoningIntegration(config)

    return _reasoning_integration


def _shutdown_on_exit() -> None:
    """Atexit handler to shutdown integration gracefully."""
    global _reasoning_integration

    if _reasoning_integration is not None:
        try:
            _reasoning_integration.shutdown(timeout=2.0)
        except Exception:
            pass  # Ignore errors during exit shutdown


# Register atexit handler for graceful shutdown
atexit.register(_shutdown_on_exit)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_reasoning(
    query: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Convenience function to apply reasoning using the global singleton.

    This is the primary entry point for most use cases. It handles singleton
    management automatically.

    Args:
        query: The user query to process
        query_type: Type from router (general, reasoning, execution, etc.)
        complexity: Complexity score (0.0 to 1.0)
        context: Optional context dict with conversation_id, history, etc.

    Returns:
        ReasoningResult with selected tools and strategy

    Example:
        >>> from vulcan.reasoning.reasoning_integration import apply_reasoning
        >>> result = apply_reasoning(
        ...     query="What causes climate change?",
        ...     query_type="reasoning",
        ...     complexity=0.7
        ... )
        >>> print(f"Tools: {result.selected_tools}")
    """
    return get_reasoning_integration().apply_reasoning(
        query, query_type, complexity, context
    )


def run_portfolio_reasoning(
    query: str,
    tools: List[str],
    strategy: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run portfolio execution using the global singleton.

    Args:
        query: The user query
        tools: List of tools to use
        strategy: Execution strategy
        constraints: Optional execution constraints

    Returns:
        Portfolio execution result dictionary

    Example:
        >>> result = run_portfolio_reasoning(
        ...     query="Complex problem",
        ...     tools=["symbolic", "causal"],
        ...     strategy="causal_reasoning"
        ... )
    """
    return get_reasoning_integration().run_portfolio(
        query, tools, strategy, constraints
    )


def get_reasoning_statistics() -> Dict[str, Any]:
    """
    Convenience function to get reasoning integration statistics.

    Returns:
        Statistics dictionary with performance metrics

    Example:
        >>> stats = get_reasoning_statistics()
        >>> print(f"Success rate: {stats['success_rate']:.1%}")
    """
    return get_reasoning_integration().get_statistics()


def shutdown_reasoning(timeout: float = 5.0) -> None:
    """
    Shutdown the global reasoning integration.

    After calling this function, the singleton will be cleared and a new
    instance will be created on the next call to get_reasoning_integration().

    Args:
        timeout: Maximum time to wait for shutdown in seconds

    Example:
        >>> shutdown_reasoning(timeout=10.0)
    """
    global _reasoning_integration

    if _reasoning_integration is not None:
        _reasoning_integration.shutdown(timeout=timeout)

        with _integration_lock:
            _reasoning_integration = None

        logger.info(f"{LOG_PREFIX} Global singleton cleared")


# =============================================================================
# SystemObserver Integration Functions
# =============================================================================


def observe_query_start(
    query_id: str,
    query: str,
    classification: Dict[str, Any]
) -> None:
    """
    Notify SystemObserver of query start event.
    
    This should be called when a query enters the processing pipeline.
    The observation feeds the WorldModel's causal learning system.
    
    Args:
        query_id: Unique identifier for the query
        query: The query text
        classification: Query classification dict with category, complexity, tools
    """
    if not SYSTEM_OBSERVER_AVAILABLE:
        return
    
    try:
        observer = get_system_observer()
        if observer:
            observer.observe_query_start(query_id, query, classification)
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Query start observation failed: {e}")


def observe_engine_result(
    query_id: str,
    engine_name: str,
    result: Dict[str, Any],
    success: bool,
    execution_time_ms: float
) -> None:
    """
    Notify SystemObserver of engine execution result.
    
    This should be called after a reasoning engine produces a result.
    Helps WorldModel learn which engines succeed on which query types.
    
    Args:
        query_id: Query identifier
        engine_name: Name of the reasoning engine
        result: Result dictionary from the engine
        success: Whether the engine execution was successful
        execution_time_ms: Execution time in milliseconds
    """
    if not SYSTEM_OBSERVER_AVAILABLE:
        return
    
    try:
        observer = get_system_observer()
        if observer:
            observer.observe_engine_result(
                query_id, engine_name, result, success, execution_time_ms
            )
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Engine result observation failed: {e}")


def observe_validation_failure(
    query_id: str,
    engine_name: str,
    reason: str,
    query: str,
    result: Dict[str, Any]
) -> None:
    """
    Notify SystemObserver of answer validation failure.
    
    This is critical for learning which engines produce invalid outputs
    for which query types. Enables WorldModel routing improvements.
    
    Args:
        query_id: Query identifier
        engine_name: Engine that produced invalid result
        reason: Why validation failed
        query: Original query
        result: The invalid result
    """
    if not SYSTEM_OBSERVER_AVAILABLE:
        return
    
    try:
        observer = get_system_observer()
        if observer:
            observer.observe_validation_failure(
                query_id, engine_name, reason, query, result
            )
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Validation failure observation failed: {e}")


def observe_outcome(
    query_id: str,
    response: Dict[str, Any],
    user_feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Notify SystemObserver of final query outcome.
    
    This should be called when query processing completes.
    
    Args:
        query_id: Query identifier
        response: Final response dict
        user_feedback: Optional user feedback
    """
    if not SYSTEM_OBSERVER_AVAILABLE:
        return
    
    try:
        observer = get_system_observer()
        if observer:
            observer.observe_outcome(query_id, response, user_feedback)
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Outcome observation failed: {e}")


def observe_error(
    query_id: str,
    error_type: str,
    error_message: str,
    component: str
) -> None:
    """
    Notify SystemObserver of system error.
    
    Args:
        query_id: Query identifier
        error_type: Type of error
        error_message: Error message
        component: Component where error occurred
    """
    if not SYSTEM_OBSERVER_AVAILABLE:
        return
    
    try:
        observer = get_system_observer()
        if observer:
            observer.observe_error(query_id, error_type, error_message, component)
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Error observation failed: {e}")
