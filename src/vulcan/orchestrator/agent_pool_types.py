# ============================================================
# VULCAN-AGI Orchestrator - Agent Pool Types & Constants
# Extracted from agent_pool.py for modularity
# Contains: constants, fallback TTLCache, conclusion extraction keys,
#           privileged result detection, simple mode configuration
# ============================================================

import logging
import os

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Fallback hardware specification values when psutil is not available
DEFAULT_FALLBACK_MEMORY_GB = 4.0  # Conservative memory estimate
DEFAULT_FALLBACK_STORAGE_GB = 100.0  # Conservative storage estimate

# Note: Import path prefixes for reasoning modules
# Used by both lazy import and fallback reasoning invocation
REASONING_IMPORT_PATHS = ['vulcan', 'src.vulcan']

# Note: Set of reasoning tool names for detecting reasoning tasks
# Used to determine if fallback reasoning should be invoked
REASONING_TOOL_NAMES = frozenset({
    'causal', 'symbolic', 'analogical', 'probabilistic', 'counterfactual',
    'deductive', 'inductive', 'abductive', 'multimodal', 'hybrid', 'ensemble'
})

# BUG FIX #1: Tool Priority Collision (The Hijack)
# INDUSTRY STANDARD: Specific-to-General tool priority
# Most specialized tools first, most general last
# This prevents generic tools (symbolic, probabilistic) from hijacking specialized queries
#
# Tier 1: Highly specialized (check first)
# - causal: Pearl-style causal reasoning
# - analogical: Structure mapping
# - multimodal: Cross-domain constraints
# - mathematical: Proof verification
#
# Tier 2: Domain-specific
# - philosophical: Ethics, thought experiments
# - language: FOL, quantifier scope
#
# Tier 3: General reasoning (check last)
# - symbolic: SAT, propositional logic
# - probabilistic: Bayesian inference
# - general: Fallback
TOOL_SELECTION_PRIORITY_ORDER = [
    # Tier 1: Highly specialized (check first)
    'causal',           # Pearl-style causal reasoning
    'analogical',       # Structure mapping
    'multimodal',       # Cross-domain constraints
    'mathematical',     # Proof verification

    # Tier 2: Domain-specific
    'philosophical',    # Ethics, thought experiments
    'language',         # FOL, quantifier scope
    'cryptographic',    # Cryptographic operations

    # Tier 3: General reasoning (check last)
    'symbolic',         # SAT, propositional logic
    'probabilistic',    # Bayesian inference
    'world_model',      # Meta-reasoning
    'general',          # Fallback
]

# Redis keys for agent pool state persistence
REDIS_KEY_AGENT_POOL_STATS = "vulcan:agent_pool:stats"
REDIS_KEY_PROVENANCE_COUNT = "vulcan:agent_pool:provenance_records_count"

# Tournament-based multi-agent selection configuration
TOURNAMENT_QUERY_TYPES = ('reasoning', 'symbolic', 'analogical', 'causal')
TOURNAMENT_MAX_CANDIDATES = 3  # Maximum agents to run in parallel for tournament
TOURNAMENT_DIVERSITY_PENALTY = 0.3
TOURNAMENT_WINNER_PERCENTAGE = 0.2

# Agent selection timeout configuration
# Note: Optimize agent selection timeout to prevent 50s delays
# This constant controls how long to wait when selecting an agent for a task
AGENT_SELECTION_TIMEOUT_SECONDS: float = 10.0  # 10 seconds max for agent selection

# PERFORMANCE FIX: Dead letter queue and stuck job detection constants
# DLQ stores jobs that fail repeatedly to prevent infinite retry loops
DEFAULT_DLQ_SIZE = 100  # Maximum entries in dead letter queue
# Jobs are considered "slow" at 70% of timeout, "critical" at 90%
STUCK_JOB_WARNING_THRESHOLD = 0.7  # 70% of timeout
STUCK_JOB_CRITICAL_THRESHOLD = 0.9  # 90% of timeout

# FIX TASK 6: Query length thresholds for reasoning validation
# BUG #11 FIX: Reduced threshold from 50 to 15 chars
# Short queries like "write a poem" are valid and should not trigger warnings
# Only warn for extremely short queries that are likely truncation artifacts
MIN_REASONING_QUERY_LENGTH = 15  # Minimum chars for valid reasoning query
# Long queries should force reasoning even with general tools
LONG_QUERY_REASONING_THRESHOLD = 500  # Chars above which reasoning is forced

# Note: Confidence threshold for world model results
# When apply_reasoning() returns a world model result with confidence >= this threshold,
# we skip invoking UnifiedReasoner.reason() to prevent confidence override
WORLD_MODEL_CONFIDENCE_THRESHOLD = 0.5

# Note: General high-confidence threshold for any reasoning engine result
# When apply_reasoning() returns ANY tool result with confidence >= this threshold,
# we use it directly without invoking UnifiedReasoner.reason() to prevent confidence override.
# This applies to all reasoning engines (symbolic, probabilistic, causal, etc.), not just world_model.
HIGH_CONFIDENCE_THRESHOLD = 0.5


# ==============================================================================
# Module-level constants for conclusion extraction (Bug #2 Fix)
# ==============================================================================

# Industry Standard: Define at module level to avoid recreation on every call
# These keys are tried in priority order when extracting conclusions from dicts
CONCLUSION_EXTRACTION_KEYS = (
    'world_model_response',  # World model specific
    'conclusion',            # Standard key
    'response',              # Common alternative
    'output',                # Some engines use this
    'result',                # Mathematical/computational engines
    'answer',                # User-facing key
    'content',               # Generic content key
    'text',                  # Text-based responses
)

# Maximum recursion depth for nested dictionary extraction
# Industry Standard: Prevent stack overflow from circular references
MAX_CONCLUSION_EXTRACTION_DEPTH = 3


# ============================================================
# SIMPLE MODE CONFIGURATION - Performance Optimization
# ============================================================

# Import simple mode configuration with fallback
try:
    from src.vulcan.simple_mode import (
        DEFAULT_MIN_AGENTS as SIMPLE_MODE_MIN_AGENTS,
        DEFAULT_MAX_AGENTS as SIMPLE_MODE_MAX_AGENTS,
        MAX_PROVENANCE_RECORDS as SIMPLE_MODE_MAX_PROVENANCE,
        AGENT_CHECK_INTERVAL as SIMPLE_MODE_CHECK_INTERVAL,
        SIMPLE_MODE,
    )
except ImportError:
    # Fallback if simple_mode not available
    SIMPLE_MODE = os.getenv("VULCAN_SIMPLE_MODE", "false").lower() in ("true", "1", "yes", "on")
    SIMPLE_MODE_MIN_AGENTS = int(os.getenv("MIN_AGENTS", "1" if SIMPLE_MODE else "10"))
    SIMPLE_MODE_MAX_AGENTS = int(os.getenv("MAX_AGENTS", "5" if SIMPLE_MODE else "100"))
    SIMPLE_MODE_MAX_PROVENANCE = int(os.getenv("MAX_PROVENANCE_RECORDS", "50" if SIMPLE_MODE else "1000"))
    SIMPLE_MODE_CHECK_INTERVAL = int(os.getenv("AGENT_CHECK_INTERVAL", "300" if SIMPLE_MODE else "30"))


# ============================================================
# TTLCache FALLBACK
# ============================================================

try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logging.warning("cachetools not available, using dict fallback with manual cleanup")

    class TTLCache(dict):
        """
        Fallback TTLCache implementation when cachetools is not available.
        Provides basic dict functionality with size limit awareness.
        TTL (time-to-live) is handled manually in the calling code.
        """

        def __init__(self, maxsize: int, ttl: float):
            """
            Initialize TTLCache fallback

            Args:
                maxsize: Maximum number of items
                ttl: Time-to-live in seconds (stored but not enforced by this class)
            """
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl

        def __setitem__(self, key, value):
            """Set item with maxsize check"""
            if len(self) >= self.maxsize and key not in self:
                # Remove oldest item (approximate LRU)
                if self:
                    oldest_key = next(iter(self))
                    del self[oldest_key]
            super().__setitem__(key, value)


# ============================================================
# PRIVILEGED RESULT DETECTION (Industry Standard AGI Safety)
# ============================================================
# Privileged results are those from world_model, meta-reasoning, or
# philosophical_reasoning that MUST NOT be overridden by fallback logic,
# consensus voting, or result blending. This ensures architectural
# separation between system/meta reasoning and general task reasoning.
# ============================================================


def is_privileged_result(reasoning_result) -> bool:
    """
    Detect if a ReasoningResult is privileged and must not be overridden.

    INDUSTRY STANDARD IMPLEMENTATION (Fix #3):
    A result is privileged if it's from a meta-reasoning/introspection component
    AND contains substantive content (not just a template).

    A result is privileged if:
    1. selected_tools includes "world_model" AND response is not a template, OR
    2. metadata includes 'is_self_introspection' or 'self_referential', OR
    3. reasoning_strategy is 'meta_reasoning' or 'philosophical_reasoning'

    TEMPLATE DETECTION:
    If world_model returns a boilerplate response like "This is a philosophical
    question requiring reasoned analysis...", it should NOT be privileged.
    This allows fallback to specialized reasoning engines.

    Privileged results bypass ALL fallback, consensus, voting, and blending logic.

    Args:
        reasoning_result: ReasoningResult object (or dict) from apply_reasoning

    Returns:
        True if result is privileged and substantive, False if template/boilerplate
    """
    if reasoning_result is None:
        return False

    # Handle both dict and object formats
    if isinstance(reasoning_result, dict):
        # BUG FIX #1: Ensure tools is always a list, never None
        selected_tools = reasoning_result.get('selected_tools', []) or []
        metadata = reasoning_result.get('metadata', {}) or {}
        strategy = reasoning_result.get('reasoning_strategy', '') or ''
        # Extract response for template detection
        response = str(reasoning_result.get('response', ''))
        if not response:
            # Try alternative keys
            response = str(
                reasoning_result.get('conclusion', '') or
                reasoning_result.get('output', '') or
                reasoning_result.get('result', '')
            )
    else:
        # Object with attributes
        # BUG FIX #1: Ensure tools is always a list, never None
        selected_tools = getattr(reasoning_result, 'selected_tools', []) or []
        metadata = getattr(reasoning_result, 'metadata', {}) or {}
        strategy = getattr(reasoning_result, 'reasoning_strategy', '') or ''
        # Extract response from object
        response = str(getattr(reasoning_result, 'response', ''))
        if not response:
            response = str(
                getattr(reasoning_result, 'conclusion', '') or
                getattr(reasoning_result, 'output', '') or
                getattr(reasoning_result, 'result', '')
            )

    # TEMPLATE DETECTION: Check if response is boilerplate
    # Industry standard: Use multiple indicators to avoid false positives
    template_indicators = [
        "This is a philosophical question requiring reasoned analysis",
        "I'll analyze this using multiple ethical frameworks",
        "Consequentialist: What outcomes matter?",
        "Deontological: What duties or rules apply?",
        "Virtue ethics: What would a person of good character do?",
        # Additional templates from world_model_core.py
        "This presents an ethical dilemma requiring careful consideration",
        "Relevant ethical frameworks:",
        "This philosophical question requires multi-framework analysis",
    ]

    is_template = any(indicator in response for indicator in template_indicators)

    # Check condition 1: world_model tool selected
    if 'world_model' in selected_tools:
        if is_template:
            logger.warning(
                "[AgentPool] Detected boilerplate response from world_model, "
                "NOT marking as privileged to allow fallback"
            )
            return False  # Not privileged - let other engines try
        return True

    # Check condition 2: self-introspection metadata flags
    # These are always privileged regardless of content
    if metadata:
        if metadata.get('is_self_introspection') or metadata.get('self_referential'):
            return True

    # Check condition 3: meta or philosophical reasoning strategy
    # Only privileged if not a template
    if strategy in ('meta_reasoning', 'philosophical_reasoning'):
        if is_template:
            logger.warning(
                f"[AgentPool] Detected boilerplate {strategy} response, "
                f"NOT marking as privileged"
            )
            return False
        return True

    return False


# Backward-compatible alias matching the original underscore-prefixed name
_is_privileged_result = is_privileged_result


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Constants
    "DEFAULT_FALLBACK_MEMORY_GB",
    "DEFAULT_FALLBACK_STORAGE_GB",
    "REASONING_IMPORT_PATHS",
    "REASONING_TOOL_NAMES",
    "TOOL_SELECTION_PRIORITY_ORDER",
    "REDIS_KEY_AGENT_POOL_STATS",
    "REDIS_KEY_PROVENANCE_COUNT",
    "TOURNAMENT_QUERY_TYPES",
    "TOURNAMENT_MAX_CANDIDATES",
    "TOURNAMENT_DIVERSITY_PENALTY",
    "TOURNAMENT_WINNER_PERCENTAGE",
    "AGENT_SELECTION_TIMEOUT_SECONDS",
    "DEFAULT_DLQ_SIZE",
    "STUCK_JOB_WARNING_THRESHOLD",
    "STUCK_JOB_CRITICAL_THRESHOLD",
    "MIN_REASONING_QUERY_LENGTH",
    "LONG_QUERY_REASONING_THRESHOLD",
    "WORLD_MODEL_CONFIDENCE_THRESHOLD",
    "HIGH_CONFIDENCE_THRESHOLD",
    "CONCLUSION_EXTRACTION_KEYS",
    "MAX_CONCLUSION_EXTRACTION_DEPTH",
    # Simple mode
    "SIMPLE_MODE",
    "SIMPLE_MODE_MIN_AGENTS",
    "SIMPLE_MODE_MAX_AGENTS",
    "SIMPLE_MODE_MAX_PROVENANCE",
    "SIMPLE_MODE_CHECK_INTERVAL",
    # TTLCache fallback
    "TTLCache",
    "CACHETOOLS_AVAILABLE",
    # Privileged result detection
    "is_privileged_result",
    "_is_privileged_result",
]
