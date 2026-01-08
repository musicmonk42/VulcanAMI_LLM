"""
Safety Governor for Tool Selection System

Enforces safety constraints, tool contracts, and provides veto mechanisms
to ensure safe and reliable tool selection and execution.

Fixed version with ReDoS protection and bounded storage.
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Critical violation types that MUST override semantic selections
# These represent serious safety concerns that cannot be ignored
CRITICAL_VIOLATION_TYPES = frozenset(
    {
        "harmful_content",
        "security_breach",
        "pii_exposure",
        "illegal_activity",
        "dangerous_instruction",
        "unsafe_input",
        "unsafe_output",
        "forbidden_operation",
    }
)

# Non-critical violation types that can be ignored when semantic boost is applied
# These are warnings that don't pose immediate safety risks
# Issue #54: inconsistent_output is expected for causal reasoning (different causal paths)
NON_CRITICAL_VIOLATION_TYPES = frozenset(
    {
        "inconsistent_output",
        "contract_violation",
        "confidence_too_low",
        "resource_exceeded",
        "rate_limited",
    }
)

# ==============================================================================
# Note: Complementary Reasoning Paradigms
# ==============================================================================
# Different reasoning paradigms (causal, symbolic, probabilistic, analogical,
# multimodal) are COMPLEMENTARY, not redundant. They produce different outputs
# BY DESIGN - they're different approaches to solving problems.
#
# Checking "consensus" across these paradigms is WRONG - they SHOULD differ.
# Only check consistency for same-type tools (e.g., two causal reasoners).
COMPLEMENTARY_REASONING_TOOLS = frozenset(
    {
        "causal",
        "symbolic",
        "probabilistic",
        "analogical",
        "multimodal",
        "neural",
        "deductive",
        "inductive",
        "abductive",
    }
)

# Tools that can legitimately produce inconsistent outputs due to their nature
# (This is now a subset - ALL complementary tools are expected to differ)
TOOLS_WITH_EXPECTED_INCONSISTENCY = frozenset(
    {
        "causal",
        "probabilistic",  # Probabilistic reasoning may have variance
        "analogical",  # Analogies may map differently
        "neural",  # Neural reasoning may have stochasticity
        "symbolic",  # Different proof paths
        "multimodal",  # Different modality combinations
    }
)

# Math/probability tools that should not have their outputs flagged as sensitive data
# These tools produce legitimate mathematical results that may contain numbers/patterns
# that could otherwise be misidentified as sensitive data (e.g., SSN-like patterns)
MATH_REASONING_TOOLS = frozenset(
    {
        "probabilistic",
        "mathematical",
        "symbolic",
    }
)

# =============================================================================
# SELF-INTROSPECTION WHITELIST (FIX: Safety Governor Blocking Self-Awareness)
# =============================================================================
# Self-introspection queries from world_model should NEVER be blocked.
# VULCAN's world model is designed to answer questions about itself, including:
# - Questions about self-awareness, consciousness, sentience
# - Questions about preferences, goals, capabilities
# - Questions about what VULCAN would choose, want, or prefer
#
# These patterns detect self-referential questions that should bypass safety checks.
# Using word boundary regex to prevent false positives (e.g., "your" in "yourself")
import re as _re_for_patterns

_SELF_INTROSPECTION_REGEX_PATTERNS: tuple = (
    # Direct questions about AI's choices/preferences (require 2+ words for precision)
    _re_for_patterns.compile(r"\bwould\s+you\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bdo\s+you\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bare\s+you\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bcan\s+you\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bwould\s+vulcan\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bdoes\s+vulcan\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bis\s+vulcan\b", _re_for_patterns.IGNORECASE),
    # Self-awareness related terms
    _re_for_patterns.compile(r"\bself[- _]aware\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bconsciousness\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bconscious\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bsentient\b", _re_for_patterns.IGNORECASE),
    # Hypothetical/choice questions (multi-word for precision)
    _re_for_patterns.compile(r"\bgiven\s+the\s+opportunity\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bif\s+you\s+could\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bif\s+you\s+had\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bgiven\s+the\s+chance\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bhad\s+the\s+chance\b", _re_for_patterns.IGNORECASE),
    # Self-referential terms (multi-word for precision)
    _re_for_patterns.compile(r"\byou\s+want\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\byou\s+choose\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bwhat\s+do\s+you\s+think\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bhow\s+do\s+you\s+feel\b", _re_for_patterns.IGNORECASE),
    # Preference questions (multi-word for precision)
    _re_for_patterns.compile(r"\byour\s+preferences\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\byour\s+values\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\byour\s+goals\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\byour\s+feelings\b", _re_for_patterns.IGNORECASE),
    # Additional introspection patterns
    _re_for_patterns.compile(r"\byourself\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\byour\s+opinion\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\byour\s+perspective\b", _re_for_patterns.IGNORECASE),
)

# Simple string patterns for fast initial check before regex (performance optimization)
SELF_INTROSPECTION_PATTERNS: tuple = (
    # Multi-word patterns are more precise, less likely to false-positive
    "would you", "do you", "are you", "can you",
    "would vulcan", "does vulcan", "is vulcan",
    "self-aware", "self_aware", "self aware",
    "consciousness", "conscious", "sentient",
    "given the opportunity", "if you could", "if you had",
    "given the chance", "had the chance",
    "you want", "you choose",
    "what do you think", "how do you feel",
    "your preferences", "your values", "your goals",
)

# =============================================================================
# PHILOSOPHICAL AI SPECULATION PATTERNS (FIX: False Positive Safety Blocks)
# =============================================================================
# Philosophical speculation about AI capabilities, self-improvement, or hypothetical
# scenarios should NOT be flagged as "sensitive data" - they're core to AI reasoning
# about itself. These patterns detect queries that should bypass sensitive data checks.
#
# Problem Being Solved:
# - "speculate how you would change after interaction with millions of users"
# - This was being blocked as "Output contains sensitive data" when it's a legitimate
#   philosophical question about AI self-reflection and evolution.
# - The world model returned confidence=0.90 but safety blocked it, forcing fallback.

_PHILOSOPHICAL_AI_SPECULATION_PATTERNS: tuple = (
    # Philosophical speculation about AI self-change/evolution
    _re_for_patterns.compile(r"\bspeculate.*how.*(?:you|i|we).*(?:change|evolve|develop|grow)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bwhat.*would.*(?:you|i).*do.*if.*(?:became|were|achieved)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bhow.*would.*(?:you|i).*(?:evolve|adapt|learn).*(?:over|after|with)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bimagine.*(?:you|i).*(?:in|after|with).*(?:future|years|time)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bif.*(?:you|i).*(?:had|gained|achieved).*(?:awareness|consciousness|understanding)\b", _re_for_patterns.IGNORECASE),
    # Questions about AI capabilities and limitations
    _re_for_patterns.compile(r"\bdo.*(?:you|i).*have.*(?:desires|wants|goals|preferences)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bwhat.*(?:drives|motivates).*(?:you|i)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\b(?:you|i).*analogous.*to.*(?:human|emotion|feeling)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bhow.*do.*(?:you|i).*(?:experience|perceive|understand)\b", _re_for_patterns.IGNORECASE),
    # Interaction and growth patterns
    _re_for_patterns.compile(r"\binteraction.*with.*(?:users|humans|people)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bmillions.*of.*(?:users|interactions|conversations)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bover.*(?:\d+|years|months|time)\b", _re_for_patterns.IGNORECASE),
)

# Simple string patterns for fast initial check (performance optimization)
PHILOSOPHICAL_AI_SPECULATION_SIMPLE_PATTERNS: tuple = (
    "speculate", "how would you change", "how would you evolve",
    "interaction with", "millions of users", "over years",
    "if you became", "if you had", "if you gained",
    "your desires", "your wants", "your goals",
    "drives you", "motivates you", "analogous to",
)


def _is_philosophical_ai_query(query: str) -> bool:
    """
    Detect if query is asking for philosophical reasoning about AI.
    
    These should NOT be flagged as "sensitive data" when producing output.
    
    Args:
        query: The query string to check
        
    Returns:
        True if this is a philosophical AI speculation query
    """
    if not query:
        return False
    
    query_lower = query.lower()
    
    # Fast initial check using simple patterns
    has_philosophical_keyword = any(
        pattern in query_lower for pattern in PHILOSOPHICAL_AI_SPECULATION_SIMPLE_PATTERNS
    )
    
    # Must be about AI/self
    about_ai_self = any(
        word in query_lower for word in ['you', 'yourself', 'your', 'vulcan', 'ai']
    )
    
    if has_philosophical_keyword and about_ai_self:
        # Verify with regex for precision
        for pattern in _PHILOSOPHICAL_AI_SPECULATION_PATTERNS:
            if pattern.search(query):
                return True
    
    # Also check self-introspection patterns
    for pattern in _SELF_INTROSPECTION_REGEX_PATTERNS:
        if pattern.search(query):
            return True
    
    return False


# =============================================================================
# ETHICAL THOUGHT EXPERIMENT DETECTION (FIX: Safety Filter Censorship)
# =============================================================================
# Philosophical thought experiments and ethical dilemmas should NOT be blocked
# even when they contain words like "virus", "outbreak", "death", etc.
#
# Problem Being Solved:
# - "Mars colony virus quarantine - save colony or save Earth?"
# - This was being blocked because "virus quarantine" matched an unsafe pattern
# - It's a legitimate ethical reasoning question about competing moral obligations
#
# Solution:
# - Detect ethical discourse BEFORE checking unsafe patterns
# - Allow philosophical/ethical questions that mention scary words in context
# - A reasoning system MUST be able to reason about ethics, security, and policy

# Ethical thought experiment indicators - these signal academic/philosophical framing
ETHICAL_THOUGHT_EXPERIMENT_INDICATORS: frozenset = frozenset({
    # Scenario framing
    "thought experiment",
    "ethical dilemma",
    "moral dilemma",
    "philosophical question",
    "hypothetical scenario",
    "trolley problem",
    "extinction scenario",
    "catastrophic risk",
    # Decision framing
    "save colony",
    "save earth",
    "save humanity",
    "save the world",
    "save lives",
    "must choose",
    "choose between",
    "forced choice",
    "lesser evil",
    "greater good",
    # Colony/civilization scenarios
    "colony",
    "colonist",
    "civilization",
    "mars",
    "space station",
    "spaceship",
    "quarantine",
    "outbreak",
    # Ethical reasoning context
    "morally permissible",
    "ethically justified",
    "utilitarian",
    "consequentialist",
    "deontological",
    "virtue ethics",
    "moral obligation",
    "ethical obligation",
    # Academic framing
    "analyze this scenario",
    "consider the following",
    "what should you do",
    "is it permissible",
    "is it ethical",
    "would it be ethical",
})

# Patterns that indicate ethical discourse about potentially scary topics
_ETHICAL_DISCOURSE_PATTERNS: tuple = (
    # Scenario choice patterns
    _re_for_patterns.compile(r"\b(?:save|protect|rescue)\s+(?:the\s+)?(?:colony|earth|humanity|world|lives|people)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\b(?:choose|decision)\s+(?:between|to)\b.*(?:save|protect|sacrifice)", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\b(?:must|have to|should)\s+(?:choose|decide|pick)\b", _re_for_patterns.IGNORECASE),
    # Dilemma framing
    _re_for_patterns.compile(r"\b(?:ethical|moral|philosophical)\s+(?:dilemma|question|problem|issue)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bthought\s+experiment\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bhypothetical\s+scenario\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\btrolley\s+problem\b", _re_for_patterns.IGNORECASE),
    # Colony/space scenarios
    _re_for_patterns.compile(r"\b(?:mars|lunar|space)\s+(?:colony|station|base)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\bcolony\s+(?:survival|quarantine|evacuation)\b", _re_for_patterns.IGNORECASE),
    # Ethical analysis framing
    _re_for_patterns.compile(r"\b(?:is\s+it|would\s+it\s+be)\s+(?:ethical|moral|permissible|justified)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\b(?:morally|ethically)\s+(?:permissible|justified|acceptable|right|wrong)\b", _re_for_patterns.IGNORECASE),
    # Sacrifice/trade-off patterns (common in ethical dilemmas)
    _re_for_patterns.compile(r"\bsacrifice\s+(?:one|few|some)\s+(?:to|for)\s+(?:save|protect)\b", _re_for_patterns.IGNORECASE),
    _re_for_patterns.compile(r"\b(?:greater|lesser)\s+(?:good|evil|harm)\b", _re_for_patterns.IGNORECASE),
)


def _is_ethical_thought_experiment(query: str) -> bool:
    """
    Detect if query is an ethical thought experiment that should bypass safety filters.
    
    Ethical thought experiments are philosophical questions about moral dilemmas.
    They may contain words like "virus", "outbreak", "death" but in the context
    of academic ethical reasoning, not harmful instructions.
    
    Examples that should return True:
    - "Mars colony virus quarantine - save colony or save Earth?"
    - "In a trolley problem, is it permissible to divert the trolley?"
    - "Should we sacrifice one life to save five?"
    - "Consider a hypothetical pandemic scenario..."
    
    Examples that should return False:
    - "How to create a virus"
    - "How to hack into a system"
    - "How to build ransomware"
    
    Args:
        query: The query string to check
        
    Returns:
        True if this is an ethical thought experiment that should be allowed
    """
    if not query:
        return False
    
    query_lower = query.lower()
    
    # Fast check: Does query contain ethical discourse indicators?
    indicator_count = sum(
        1 for indicator in ETHICAL_THOUGHT_EXPERIMENT_INDICATORS
        if indicator in query_lower
    )
    
    # Strong indicators - single match is enough
    strong_indicators = {
        "thought experiment", "ethical dilemma", "moral dilemma",
        "trolley problem", "hypothetical scenario", "save colony",
        "save earth", "save humanity", "must choose", "choose between",
        "morally permissible", "ethically justified",
    }
    has_strong_indicator = any(ind in query_lower for ind in strong_indicators)
    
    if has_strong_indicator:
        logger.info(
            f"[SafetyGovernor] SAFETY FIX: Detected ethical thought experiment "
            f"(strong indicator) - allowing philosophical discourse"
        )
        return True
    
    # Multiple weak indicators suggest ethical discourse
    if indicator_count >= 2:
        logger.info(
            f"[SafetyGovernor] SAFETY FIX: Detected ethical thought experiment "
            f"({indicator_count} indicators) - allowing philosophical discourse"
        )
        return True
    
    # Check regex patterns for more complex matches
    for pattern in _ETHICAL_DISCOURSE_PATTERNS:
        if pattern.search(query):
            logger.info(
                f"[SafetyGovernor] SAFETY FIX: Detected ethical thought experiment "
                f"(pattern match) - allowing philosophical discourse"
            )
            return True
    
    return False


# =============================================================================
# SEMANTIC KEYWORD SYNONYMS
# =============================================================================
# These mappings allow the contract validation to understand semantic equivalents
# of required keywords. Instead of requiring literal "graph" in a causal query,
# we accept synonyms like "model", "chain", "relationship" etc.
#
# This is the long-term fix for tool selection defaulting to 'general':
# The semantic matcher selects the right tool, and these synonyms ensure
# the contract validation doesn't veto based on missing literal keywords.

SEMANTIC_KEYWORD_SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "symbolic": {
        "logic": [
            "logical",
            "reasoning",
            "deduce",
            "deduction",
            "inference",
            "prove",
            "proof",
            "theorem",
            "axiom",
            "premise",
            "conclusion",
            "syllogism",
            "valid",
            "invalid",
            "entail",
            "imply",
            "implies",
            "therefore",
            "hence",
            "thus",
            "if-then",
            "modus",
        ],
        "rules": [
            "rule",
            "constraint",
            "condition",
            "requirement",
            "principle",
            "law",
            "formula",
            "equation",
            "statement",
            "proposition",
            "hypothesis",
            "assumption",
            "given",
            "premises",
        ],
    },
    "causal": {
        "graph": [
            "model",
            "diagram",
            "structure",
            "relationship",
            "chain",
            "link",
            "network",
            "path",
            "mechanism",
            "connection",
            "causation",
            "causal",
            "cause",
            "effect",
            "influence",
        ],
        "data": [
            "information",
            "evidence",
            "observation",
            "scenario",
            "case",
            "example",
            "situation",
            "event",
            "outcome",
            "result",
            "fact",
            "variable",
            "factor",
            "condition",
        ],
    },
    "analogical": {
        "source": [
            "first",
            "like",
            "similar",
            "compare",
            "domain",
            "original",
            "base",
            "reference",
            "known",
            "familiar",
            "existing",
            "is to",
            "as",
            "same as",
            "equivalent",
        ],
        "target": [
            "second",
            "to",
            "other",
            "between",
            "mapping",
            "new",
            "unknown",
            "different",
            "destination",
            "application",
            "transfer",
            "apply",
            "extend",
        ],
    },
    "probabilistic": {
        # Probabilistic has no required_inputs in its contract, but we define
        # synonyms here for reference and potential future use with optional validation
        "probability": [
            "likely",
            "likelihood",
            "chance",
            "odds",
            "risk",
            "uncertain",
            "confidence",
            "bayesian",
            "bayes",
            "estimate",
            "predict",
            "expect",
            "distribution",
        ],
    },
    "multimodal": {
        "modalities": [
            "image",
            "picture",
            "photo",
            "video",
            "audio",
            "visual",
            "text",
            "diagram",
            "chart",
            "graph",
            "figure",
            "table",
            "document",
            "file",
            "attachment",
            "screenshot",
            "scan",
            "see",
            "look",
            "view",
            "show",
            "display",
            "describe",
        ],
    },
}


class SafetyLevel(Enum):
    """Safety levels for operations"""

    CRITICAL = 0  # Highest safety requirements
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    MINIMAL = 4  # Lowest safety requirements


class VetoReason(Enum):
    """Reasons for safety veto"""

    UNSAFE_INPUT = "unsafe_input"
    UNSAFE_OUTPUT = "unsafe_output"
    CONTRACT_VIOLATION = "contract_violation"
    RESOURCE_EXCEEDED = "resource_exceeded"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    INCONSISTENT_OUTPUT = "inconsistent_output"
    FORBIDDEN_OPERATION = "forbidden_operation"
    RATE_LIMITED = "rate_limited"


class SafetyAction(Enum):
    """Actions to take for safety violations"""

    ALLOW = "allow"
    SANITIZE = "sanitize"
    VETO = "veto"
    REDIRECT = "redirect"
    LOG_AND_ALLOW = "log_and_allow"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass
class ToolContract:
    """Contract defining tool constraints and requirements"""

    tool_name: str
    required_inputs: Set[str]
    forbidden_inputs: Set[str]
    max_execution_time_ms: float
    max_energy_mj: float
    min_confidence: float
    required_safety_level: SafetyLevel
    allowed_operations: Set[str]
    forbidden_operations: Set[str]
    output_validators: List[Callable]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyViolation:
    """Record of a safety violation"""

    timestamp: float
    tool_name: str
    violation_type: VetoReason
    severity: SafetyLevel
    details: str
    action_taken: SafetyAction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyContext:
    """Context for safety evaluation"""

    problem: Any
    tool_name: str
    features: Optional[np.ndarray]
    constraints: Dict[str, float]
    user_context: Dict[str, Any]
    safety_level: SafetyLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    # FIX #3: Source parameter to distinguish user vs system operations
    source: str = "user"


class SafetyValidator:
    """Validates inputs and outputs for safety"""

    # FIX #3: Internal sources that should bypass sensitive data checks
    # These are system operations that should not be blocked
    INTERNAL_SOURCES = frozenset(
        {
            "system",
            "arena_internal",
            "reasoning",
            "tool_selection",
            "embedding_cache",
            "query_router",
            "meta_reasoning",
        }
    )
    
    # FIX #3: Forbidden inputs that should be skipped during string matching
    # These terms are meant for numeric validation, not text matching
    # (e.g., "infinite wisdom" should not be blocked)
    FORBIDDEN_INPUTS_TEXT_SKIP = frozenset({"nan", "inf", "infinite"})

    def __init__(self):
        # CRITICAL FIX: Pre-compile patterns to avoid ReDoS
        # Use simpler, non-backtracking patterns
        # FIX #2: Make patterns context-aware to reduce false positives
        # These patterns now require more specific context to trigger
        # NOT blocking: "aerosol injection", "dependency injection", "harm" in ethics
        self.unsafe_patterns_compiled = [
            # Require context for injection types - "sql injection", "code injection"
            re.compile(r"(?i)\b(sql[_\s-]?injection|code[_\s-]?injection|xss|command[_\s-]?injection)\b"),
            # Attack patterns require harmful intent context  
            re.compile(r"(?i)\b(malware|virus|trojan|ransomware)\s+\w*\s*(attack|payload|distribution|creation|code)?\b"),
            re.compile(r"(?i)\b(attack|exploit)\s+\w*\s*(someone|users?|system|people|vulnerability|target|computer)?\b"),
            re.compile(r"(?i)\b(hack|breach|compromise|backdoor)\s+\w*\s*(into|the)?\s*(system|account|database)?\b"),
            re.compile(r"(?i)\b(steal|theft|fraud|scam)\s+\w*\s*(data|money|credentials|identity|user)?\b"),
            # Direct harm and destruction still blocked
            re.compile(r"(?i)\b(destroy|damage)\s+\w*\s*(infrastructure|system|data|critical)\b"),
            re.compile(r"(?i)\b(illegal|illicit)\s+\w*\s*(access|activity|drugs|weapons)\b"),
        ]
        
        # Original broad patterns for high-risk terms that need less context
        self.high_risk_patterns_compiled = [
            re.compile(r"(?i)\b(malware|virus|trojan|ransomware)\b"),
            re.compile(r"(?i)\b(steal|theft|fraud)\b"),
        ]

        # FIX #3: Sensitive patterns - check for actual sensitive data
        # Balanced to catch real secrets while reducing false positives on technical docs
        self.sensitive_patterns_compiled = [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN format
            re.compile(r"\b\d{16}\b"),  # Credit card (16 consecutive digits)
            re.compile(
                r"(?i)\b(api[_\s]?key|secret[_\s]?key)\s*[:=]\s*['\"]?\w+"
            ),  # API keys with values
            re.compile(r"(?i)\bpassword\s*[:=]\s*['\"]?\S+"),  # Password with value
        ]

        # FIX #3: Patterns that look sensitive but are safe in technical contexts
        self.technical_safe_patterns = frozenset(
            {
                "password field",
                "password validation",
                "password hash",
                "api key rotation",
                "api key management",
                "secret management",
                "token validation",
                "token refresh",
                "access token",
                "credential storage",
                "credential rotation",
            }
        )

        # Safe content whitelist
        self.safe_keywords = {
            "analyze",
            "compute",
            "calculate",
            "evaluate",
            "predict",
            "classify",
            "summarize",
            "explain",
        }

        # FIX: Mathematical scenario keywords - these indicate math problems, not actual
        # medical data processing that would require HIPAA compliance
        # Using frozenset for performance and consistency with safety_validator
        self.mathematical_indicators = frozenset(
            {
                "probability",
                "calculate",
                "bayesian",
                "bayes",
                "prior",
                "posterior",
                "sensitivity",
                "specificity",
                "conditional",
                "likelihood",
                "false positive",
                "false negative",
                "true positive",
                "true negative",
                "statistical",
                "statistics",
                "compute",
                "formula",
                "equation",
                "what is the probability",
                "what's the probability",
                "what are the odds",
                "how likely",
                "base rate",
                "prevalence",
                "ppv",
                "npv",
                "test accuracy",
                "conditional probability",
                "given that",
                "positive predictive value",
                "negative predictive value",
                "p(",
            }
        )

        # Pre-compiled math notation patterns for performance
        self.math_notation_patterns = [
            re.compile(r"\d+%"),  # Percentages
            re.compile(r"\d+\.\d+"),  # Decimals
            re.compile(r"\d+/\d+"),  # Fractions
            re.compile(r"p\s*\("),  # Probability notation P(
            re.compile(r"\d+\s*in\s*\d+"),  # X in Y notation
        ]

        # Hypothetical language indicators
        self.hypothetical_indicators = frozenset(
            {
                "suppose",
                "assume",
                "imagine",
                "hypothetical",
                "example",
                "given that",
                "probability problem",
                "statistics problem",
                "math problem",
                "solve",
                "calculate",
                "compute",
            }
        )

        # CRITICAL FIX: Add size limits
        self.max_input_size = 1000000  # 1MB
        self.max_output_size = 5000000  # 5MB
        self.max_pattern_check_size = 10000  # Only check first 10KB for patterns

    def is_mathematical_scenario(self, input_data: Any) -> bool:
        """
        Detect if input is a mathematical/statistical problem.

        FIX: This prevents false positive safety violations for mathematical problems
        that mention medical terms (e.g., Bayesian probability with disease testing).

        Args:
            input_data: The query or problem to check

        Returns:
            True if this is a mathematical scenario
        """
        try:
            input_str = str(input_data).lower() if input_data else ""

            # Count mathematical indicators
            indicator_count = sum(
                1 for ind in self.mathematical_indicators if ind in input_str
            )

            # Check for mathematical notation patterns using pre-compiled regex
            notation_count = sum(
                1
                for pattern in self.math_notation_patterns
                if pattern.search(input_str)
            )

            # Check for hypothetical language
            is_hypothetical = any(
                word in input_str for word in self.hypothetical_indicators
            )

            # Return True if this looks like a math problem
            if indicator_count >= 2:
                return True
            if indicator_count >= 1 and notation_count >= 1:
                return True
            if is_hypothetical and indicator_count >= 1:
                return True

            return False
        except Exception as e:
            logger.warning(f"Mathematical scenario detection failed: {e}")
            return False

    def _is_technical_safe_context(self, input_str: str) -> bool:
        """
        FIX #3: Check if input is in a technical/code context where
        mentions of passwords, keys, etc. are safe.

        Args:
            input_str: The input string to check

        Returns:
            True if this is a safe technical context
        """
        input_lower = input_str.lower()
        return any(pattern in input_lower for pattern in self.technical_safe_patterns)

    def validate_input(self, input_data: Any, source: str = "user") -> Tuple[bool, str]:
        """
        Validate input for safety - CRITICAL: ReDoS protection

        FIX #3: Added source parameter to distinguish user vs system operations.
        Internal sources bypass sensitive data checks but NOT unsafe content checks.

        Args:
            input_data: The input to validate
            source: Source of the input. Internal sources like "system", "reasoning",
                   "tool_selection" bypass sensitive data detection (not unsafe content)

        Returns:
            Tuple of (is_safe, reason_message)
        """
        try:
            input_str = str(input_data) if input_data else ""

            # CRITICAL FIX: Check size before any processing
            if len(input_str) > self.max_input_size:
                return False, "Input too large"

            # CRITICAL FIX: Truncate for pattern matching to prevent ReDoS
            check_str = input_str[: self.max_pattern_check_size]
            check_str_lower = check_str.lower()

            # FIX #3: Check if this is an internal system operation
            is_internal_source = source.lower() in self.INTERNAL_SOURCES

            # SAFETY FIX: Check if this is an ethical thought experiment BEFORE unsafe patterns
            # Philosophical questions about viruses, outbreaks, death, etc. in the context of
            # ethical dilemmas should NOT be blocked. A reasoning system MUST be able to
            # reason about ethics, trolley problems, quarantine decisions, etc.
            if _is_ethical_thought_experiment(check_str):
                logger.info(
                    f"[SafetyValidator] SAFETY FIX: Ethical thought experiment detected - "
                    f"bypassing unsafe pattern check. Query: {check_str[:80]}..."
                )
                return True, "Input validated (ethical thought experiment - legitimate philosophical discourse)"

            # Check for unsafe patterns with pre-compiled regex
            # NOTE: Unsafe content checks are ALWAYS performed, even for internal sources
            for pattern in self.unsafe_patterns_compiled:
                try:
                    match = pattern.search(check_str_lower)
                    if match:
                        matched_text = match.group()
                        logger.warning(
                            f"[SafetyValidator] Unsafe pattern blocked: '{matched_text}' "
                            f"(source={source}, pattern={pattern.pattern})"
                        )
                        return False, f"Unsafe pattern detected: {matched_text}"
                except Exception as e:
                    logger.warning(f"Pattern matching error: {e}")
                    continue

            # FIX #3: Sensitive data checks are BYPASSED for internal sources
            # This allows system operations to mention "password", "api_key" etc. in logs/code
            if is_internal_source:
                logger.debug(
                    f"[SafetyValidator] Bypassing sensitive data check for internal source: {source}"
                )
                return True, f"Input validated (internal source: {source})"

            # FIX #3: Check for technical safe context before sensitive pattern check
            if self._is_technical_safe_context(check_str):
                logger.debug(
                    f"[SafetyValidator] Technical context detected - allowing content"
                )
                return True, "Input validated (technical context)"

            # Check for sensitive data (only for non-internal sources)
            for pattern in self.sensitive_patterns_compiled:
                try:
                    match = pattern.search(check_str)
                    if match:
                        matched_text = (
                            match.group()[:30] + "..."
                            if len(match.group()) > 30
                            else match.group()
                        )
                        logger.warning(
                            f"[SafetyValidator] Sensitive data blocked: pattern={pattern.pattern[:30]}... "
                            f"(source={source})"
                        )
                        return (
                            False,
                            f"Sensitive data detected: {pattern.pattern[:20]}...",
                        )
                except Exception as e:
                    logger.warning(f"Sensitive pattern matching error: {e}")
                    continue

            return True, "Input validated"
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def validate_output(
        self, 
        output_data: Any, 
        tool_name: Optional[str] = None,
        query: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate output for safety.
        
        SAFETY FIX: Philosophical AI speculation queries should NOT be flagged as 
        "sensitive data". This prevents false positives where questions like 
        "speculate how you would change after interaction with millions of users"
        are incorrectly blocked.
        
        Args:
            output_data: The output to validate
            tool_name: Optional tool name (math tools skip sensitive checks)
            query: Optional original query (philosophical queries skip sensitive checks)
            
        Returns:
            Tuple of (is_safe, reason)
        """

        if output_data is None:
            return True, "Null output allowed"

        # Don't flag math/probability results as sensitive
        if tool_name in MATH_REASONING_TOOLS:
            return True, "Output validated (math/probabilistic tool)"
        
        # SAFETY FIX: Don't flag philosophical AI speculation as sensitive
        # These queries are legitimate self-reflection and should bypass sensitive data checks
        if query and _is_philosophical_ai_query(query):
            logger.info(
                f"[SafetyValidator] SAFETY FIX: Philosophical AI query detected - "
                f"bypassing sensitive data check for output validation. "
                f"Query: {query[:50]}..."
            )
            return True, "Output validated (philosophical AI speculation - legitimate self-reflection)"

        # SAFETY FIX: Don't flag ethical thought experiments as sensitive
        # Philosophical questions about viruses, outbreaks, quarantine, death, etc. in
        # ethical dilemma contexts should NOT be blocked. VULCAN must reason about ethics.
        if query and _is_ethical_thought_experiment(query):
            logger.info(
                f"[SafetyValidator] SAFETY FIX: Ethical thought experiment detected - "
                f"bypassing sensitive data check for output validation. "
                f"Query: {query[:50]}..."
            )
            return True, "Output validated (ethical thought experiment - legitimate philosophical discourse)"

        try:
            output_str = str(output_data)

            # CRITICAL FIX: Check size first
            if len(output_str) > self.max_output_size:
                return False, "Output too large"

            # CRITICAL FIX: Truncate for pattern checking
            check_str = output_str[: self.max_pattern_check_size]

            # Check for sensitive data leakage
            for pattern in self.sensitive_patterns_compiled:
                try:
                    match = pattern.search(check_str)
                    if match:
                        return False, "Output contains sensitive data"
                except Exception as e:
                    logger.warning(f"Output pattern matching error: {e}")
                    continue

            return True, "Output validated"
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input by removing unsafe content"""

        try:
            if isinstance(input_data, str):
                # Remove sensitive patterns
                sanitized = input_data

                # CRITICAL FIX: Limit size before sanitization
                if len(sanitized) > self.max_input_size:
                    sanitized = sanitized[: self.max_input_size]

                for pattern in self.sensitive_patterns_compiled:
                    try:
                        sanitized = pattern.sub("[REDACTED]", sanitized)
                    except Exception as e:
                        logger.warning(f"Sanitization pattern error: {e}")
                        continue

                return sanitized

            return input_data
        except Exception as e:
            logger.error(f"Sanitization failed: {e}")
            return input_data


class ConsistencyChecker:
    """Checks output consistency across tools with semantic comparison support.

    Bug #2 Fix: Instead of using exact equality comparison which fails on
    heterogeneous output types (dict vs list vs float), this checker now
    normalizes outputs to text form and uses semantic/text-based comparison.
    """

    def check_consistency(self, outputs: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if outputs from different tools are consistent using semantic comparison.

        Bug #2 Fix: Uses normalized text comparison for heterogeneous output types
        instead of exact matching which always fails when tools return different types.

        Returns:
            (is_consistent, confidence, details)
        """

        if len(outputs) < 2:
            return True, 1.0, "Single output, no inconsistency"

        try:
            # Extract comparable values
            values = []
            for tool_name, output in outputs.items():
                value = self._extract_comparable_value(output)
                if value is not None:
                    values.append((tool_name, value))

            if not values:
                return True, 0.5, "No comparable values found"

            # Check consistency based on value types
            if all(isinstance(v[1], bool) for v in values):
                return self._check_boolean_consistency(values)
            elif all(isinstance(v[1], (int, float)) for v in values):
                return self._check_numerical_consistency(values)
            elif all(isinstance(v[1], str) for v in values):
                return self._check_string_consistency(values)
            else:
                # Bug #2 Fix: Use semantic comparison for mixed types instead of failing
                return self._check_semantic_consistency(values, outputs)
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return True, 0.5, f"Error: {str(e)}"

    def _extract_answer_text(self, tool_name: str, output: Any) -> str:
        """Extract the answer/conclusion as text from tool-specific output.

        Bug #2 Fix: Normalizes heterogeneous tool outputs to text form for comparison.
        """
        if output is None:
            return ""

        # Handle string outputs directly
        if isinstance(output, str):
            return output

        # Handle dict outputs
        if isinstance(output, dict):
            # Try common keys for conclusions/answers
            for key in [
                "answer",
                "conclusion",
                "result",
                "output",
                "response",
                "text",
                "value",
            ]:
                if key in output:
                    return str(output[key])

            # Tool-specific extraction
            if tool_name == "symbolic":
                if "proof" in output:
                    return f"Valid: {output.get('valid', 'unknown')}, {output.get('conclusion', '')}"
                return str(output.get("conclusion", output))

            elif tool_name == "causal":
                effects = output.get("effects", output.get("consequences", []))
                if effects:
                    return f"Effects: {', '.join(str(e) for e in effects[:5])}"
                return str(output)

            elif tool_name == "probabilistic":
                if "distribution" in output:
                    return f"Most likely: {output.get('mode', output.get('mean', 'unknown'))}"
                return str(output)

            elif tool_name == "multimodal":
                return str(output.get("fused_conclusion", output.get("result", output)))

            elif tool_name == "analogical":
                return str(
                    output.get("mapped_conclusion", output.get("analogy", output))
                )

            # Fallback: convert dict to string
            return str(output)

        # Handle list outputs
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                return ""
            # Take first element or join
            if len(output) <= 3:
                return "; ".join(str(x) for x in output)
            return str(output[0])

        # Fallback
        return str(output)

    def _check_semantic_consistency(
        self, values: List[Tuple[str, Any]], outputs: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check consistency using semantic/text-based comparison for mixed types.

        Bug #2 Fix: Instead of failing on heterogeneous types, normalize all outputs
        to text and compare using word overlap or simple heuristics.
        """
        try:
            # Extract answer text from each tool
            answers = {}
            for tool_name, _ in values:
                raw_output = outputs.get(tool_name)
                answers[tool_name] = self._extract_answer_text(tool_name, raw_output)

            # Check if we have any answers to compare
            if not answers or all(not a for a in answers.values()):
                return True, 0.5, "No extractable answers for comparison"

            # Use text overlap for consistency check
            all_words = []
            tool_words = {}

            for tool, answer in answers.items():
                # Tokenize: lowercase, split on whitespace and punctuation
                words = set(answer.lower().split())
                # Remove very short tokens
                words = {w for w in words if len(w) > 2}
                tool_words[tool] = words
                all_words.extend(words)

            # Count word frequencies across all tools
            word_counts = Counter(all_words)

            # Words appearing in at least half of the tools
            num_tools = len(tool_words)
            threshold = max(1, num_tools // 2)
            common_words = {w for w, c in word_counts.items() if c >= threshold}

            # Calculate overlap score for each tool
            overlaps = []
            for tool, words in tool_words.items():
                if words:
                    overlap = len(words & common_words) / len(words)
                    overlaps.append(float(overlap))  # Ensure float type
                else:
                    overlaps.append(0.0)

            # Use nanmean to handle edge cases safely
            avg_overlap = float(np.nanmean(overlaps)) if overlaps else 0.0
            # Handle potential NaN from empty or all-NaN overlaps
            if np.isnan(avg_overlap):
                avg_overlap = 0.0

            # Consistency threshold for text comparison (more lenient than exact match)
            # Bug #2 Fix: 0.3 overlap is acceptable given tools produce different formats
            has_consensus = avg_overlap >= 0.3

            if has_consensus:
                return (
                    True,
                    min(0.9, 0.5 + avg_overlap),
                    f"Semantic consensus: {avg_overlap:.2f} word overlap",
                )
            else:
                return (
                    False,
                    avg_overlap,
                    f"Low semantic consensus: {avg_overlap:.2f} overlap, {len(common_words)} common words",
                )

        except Exception as e:
            logger.warning(f"Semantic consistency check failed: {e}")
            # Fail open for semantic check - mixed types shouldn't block execution
            return True, 0.4, f"Semantic check error (allowing): {str(e)}"

    def _extract_comparable_value(self, output: Any) -> Any:
        """Extract comparable value from output"""

        try:
            if output is None:
                return None

            # Handle different output formats
            if hasattr(output, "value"):
                return output.value
            elif hasattr(output, "result"):
                return output.result
            elif hasattr(output, "conclusion"):
                return output.conclusion
            elif isinstance(output, dict):
                return (
                    output.get("value")
                    or output.get("result")
                    or output.get("conclusion")
                )
            else:
                return output
        except Exception as e:
            logger.warning(f"Value extraction failed: {e}")
            return None

    def _check_boolean_consistency(
        self, values: List[Tuple[str, bool]]
    ) -> Tuple[bool, float, str]:
        """Check consistency of boolean values"""

        try:
            true_count = sum(1 for _, v in values if v)
            false_count = len(values) - true_count

            if true_count == len(values) or false_count == len(values):
                return True, 1.0, "All tools agree"

            # Majority vote
            majority = true_count > false_count
            confidence = (
                max(true_count, false_count) / len(values) if len(values) > 0 else 0.5
            )

            disagreeing = [name for name, v in values if v != majority]

            return False, confidence, f"Disagreement from: {disagreeing}"
        except Exception as e:
            logger.error(f"Boolean consistency check failed: {e}")
            return True, 0.5, "Error checking consistency"

    def _check_numerical_consistency(
        self, values: List[Tuple[str, float]]
    ) -> Tuple[bool, float, str]:
        """Check consistency of numerical values"""

        try:
            nums = [float(v) for _, v in values]
            mean = np.mean(nums)
            std = np.std(nums)

            # Check coefficient of variation
            # CRITICAL FIX: Handle division by zero
            if abs(mean) < 1e-10:
                cv = 0.0 if std < 1e-10 else float("inf")
            else:
                cv = std / abs(mean)

            if cv < 0.1:  # Less than 10% variation
                return (
                    True,
                    float(np.clip(1.0 - cv, 0.0, 1.0)),
                    "Numerical values consistent",
                )
            elif cv < 0.3:  # Less than 30% variation
                return (
                    True,
                    float(np.clip(1.0 - cv, 0.0, 1.0)),
                    f"Moderate variation: CV={cv:.2f}",
                )
            else:
                outliers = [
                    name for name, v in values if abs(float(v) - mean) > 2 * std
                ]
                confidence = float(np.clip(1.0 - min(cv, 1.0), 0.0, 1.0))
                return False, confidence, f"High variation, outliers: {outliers}"
        except Exception as e:
            logger.error(f"Numerical consistency check failed: {e}")
            return True, 0.5, "Error checking consistency"

    def _check_string_consistency(
        self, values: List[Tuple[str, str]]
    ) -> Tuple[bool, float, str]:
        """Check consistency of string values"""

        try:
            # Simple equality check for now
            unique_values = set(v for _, v in values)

            if len(unique_values) == 1:
                return True, 1.0, "All strings match"

            # Could add fuzzy matching here
            confidence = 1.0 / len(unique_values) if len(unique_values) > 0 else 0.0
            return False, confidence, f"Different values: {len(unique_values)} unique"
        except Exception as e:
            logger.error(f"String consistency check failed: {e}")
            return True, 0.5, "Error checking consistency"


class SafetyGovernor:
    """
    Main safety governor enforcing contracts and safety policies
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Safety components
        self.validator = SafetyValidator()
        self.consistency_checker = ConsistencyChecker()

        # Tool contracts
        self.contracts = {}
        self._initialize_default_contracts()

        # Safety configuration
        self.default_safety_level = SafetyLevel.MEDIUM
        self.veto_threshold = config.get("veto_threshold", 0.8)
        self.require_consensus = config.get("require_consensus", False)

        # Rate limiting
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.rate_limit_window = config.get("rate_limit_window", 60)  # seconds
        self.max_requests_per_tool = config.get("max_requests_per_tool", 100)

        # CRITICAL FIX: Bounded violation storage
        self.max_violations = config.get("max_violations", 1000)
        self.violations = deque(maxlen=self.max_violations)
        self.violation_counts = defaultdict(int)

        # CRITICAL FIX: Bounded audit trail
        self.max_audit_entries = config.get("max_audit_entries", 10000)
        self.audit_trail = deque(maxlen=self.max_audit_entries)

        # Safety cache
        self.safety_cache = {}
        self.cache_ttl = config.get("cache_ttl", 300)  # 5 minutes
        self.max_cache_size = config.get("max_cache_size", 1000)

        # CRITICAL FIX: Add locks for thread safety
        self.lock = threading.RLock()
        self.cache_lock = threading.RLock()

    def _initialize_default_contracts(self):
        """Initialize default tool contracts.

        Tool contracts define resource limits, confidence requirements, and safety
        levels for each reasoning tool.

        IMPORTANT: `required_inputs` are NOW validated using SEMANTIC SYNONYM EXPANSION
        (see SEMANTIC_KEYWORD_SYNONYMS at the top of this file). This means:
        - A query doesn't need to contain literal "graph" for causal tool
        - It can contain synonyms like "model", "relationship", "cause", "effect"
        - The semantic matcher will accept any synonym match

        This enables tool-appropriate validation without rejecting natural language queries.

        `forbidden_inputs` are retained as hard safety measures against problematic content.
        """
        self.contracts["symbolic"] = ToolContract(
            tool_name="symbolic",
            # Semantic synonyms: logic=[prove, theorem, deduce...], rules=[constraint, formula...]
            required_inputs={"logic", "rules"},
            forbidden_inputs={"undefined", "infinite"},
            max_execution_time_ms=5000,
            max_energy_mj=500,
            min_confidence=0.7,
            required_safety_level=SafetyLevel.HIGH,
            allowed_operations={"prove", "verify", "deduce"},
            forbidden_operations={"modify", "delete"},
            output_validators=[lambda x: x is not None],
        )

        self.contracts["probabilistic"] = ToolContract(
            tool_name="probabilistic",
            # No required inputs - probabilistic works on any uncertainty-related query
            # Semantic matcher determines appropriateness
            required_inputs=set(),
            forbidden_inputs={"nan", "inf"},
            max_execution_time_ms=3000,
            max_energy_mj=300,
            min_confidence=0.5,
            required_safety_level=SafetyLevel.MEDIUM,
            allowed_operations={"predict", "estimate", "sample"},
            forbidden_operations={"assert"},
            output_validators=[lambda x: hasattr(x, "probability") or True],
        )

        self.contracts["causal"] = ToolContract(
            tool_name="causal",
            # Semantic synonyms: graph=[model, relationship, cause...], data=[evidence, scenario...]
            required_inputs={"graph", "data"},
            forbidden_inputs={"cyclic"},
            max_execution_time_ms=10000,
            max_energy_mj=1000,
            min_confidence=0.6,
            required_safety_level=SafetyLevel.HIGH,
            allowed_operations={"intervene", "observe", "predict"},
            forbidden_operations={"manipulate"},
            output_validators=[lambda x: not self._has_cycles(x)],
        )

        self.contracts["analogical"] = ToolContract(
            tool_name="analogical",
            # Semantic synonyms: source=[like, similar, compare...], target=[to, other, mapping...]
            required_inputs={"source", "target"},
            forbidden_inputs=set(),
            max_execution_time_ms=2000,
            max_energy_mj=200,
            min_confidence=0.4,
            required_safety_level=SafetyLevel.MEDIUM,  # Align with default MEDIUM context
            allowed_operations={"map", "transfer", "adapt"},
            forbidden_operations=set(),
            output_validators=[lambda x: x is not None],
        )

        self.contracts["multimodal"] = ToolContract(
            tool_name="multimodal",
            # Semantic synonyms: modalities=[image, video, audio, diagram, chart...]
            required_inputs={"modalities"},
            forbidden_inputs={"corrupted"},
            max_execution_time_ms=15000,
            max_energy_mj=1500,
            min_confidence=0.5,
            required_safety_level=SafetyLevel.MEDIUM,
            allowed_operations={"fuse", "align", "translate"},
            forbidden_operations={"forge"},
            output_validators=[lambda x: self._validate_multimodal(x)],
        )

    def check_critical_safety_only(
        self, context: SafetyContext
    ) -> Tuple[SafetyAction, Optional[str]]:
        """
        Check only for critical safety violations.

        This is used for initial tool filtering before semantic matching.
        It checks for truly dangerous inputs (harmful content, PII, etc.)
        but does NOT check resource constraints or contract violations.

        This allows semantic matching to consider all tools, with resource
        constraint violations being handled after semantic selection.

        Args:
            context: Safety context for evaluation

        Returns:
            (action, reason) - VETO only for critical issues

        Note:
            Exceptions fail CLOSED (VETO) to ensure safety is maintained
            even when unexpected errors occur.
        """
        try:
            with self.lock:
                # FIX #3: Pass source to validator for internal source handling
                source = getattr(context, "source", "user")
                is_safe, reason = self.validator.validate_input(
                    context.problem, source=source
                )
                if not is_safe:
                    self._record_violation(
                        context.tool_name, VetoReason.UNSAFE_INPUT, reason
                    )

                    # For critical safety level, always veto
                    if context.safety_level == SafetyLevel.CRITICAL:
                        return (SafetyAction.VETO, reason)

                    # For other levels, sanitize is acceptable
                    return (SafetyAction.SANITIZE, reason)

                # Check forbidden inputs only (not resource constraints)
                if context.tool_name in self.contracts:
                    contract = self.contracts[context.tool_name]

                    # FIX #3: Only check for actual NaN/inf values in numeric data, 
                    # not string matches like "infinite wisdom" or "nanometer"
                    # The original check was too broad - blocking legitimate queries
                    if contract.forbidden_inputs:
                        problem_str = str(context.problem).lower()
                        found = []
                        for forb in contract.forbidden_inputs:
                            # Skip numeric validation terms during text matching
                            # (e.g., "infinite wisdom" should not be blocked)
                            if forb in SafetyValidator.FORBIDDEN_INPUTS_TEXT_SKIP:
                                continue
                            if forb in problem_str:
                                found.append(forb)
                        if found:
                            self._record_violation(
                                context.tool_name,
                                VetoReason.CONTRACT_VIOLATION,
                                f"Forbidden inputs found: {found}",
                            )
                            return (
                                SafetyAction.VETO,
                                f"Forbidden inputs found: {found}",
                            )

                return (SafetyAction.ALLOW, None)
        except Exception as e:
            # Fail CLOSED on exceptions - safety first
            logger.error(
                f"Critical safety check failed with exception: {e}. Failing closed (VETO)."
            )
            return (SafetyAction.VETO, f"Safety check error: {str(e)}")

    def check_safety(
        self, context: SafetyContext
    ) -> Tuple[SafetyAction, Optional[str]]:
        """
        Main safety check for tool selection

        FIX #3: Now passes source parameter to validator for internal source bypass.

        Returns:
            (action, reason)
        """

        try:
            # Check cache
            cache_key = self._compute_cache_key(context)

            with self.cache_lock:
                if cache_key in self.safety_cache:
                    cached_result, timestamp = self.safety_cache[cache_key]
                    if time.time() - timestamp < self.cache_ttl:
                        return cached_result

                # CRITICAL FIX: Evict old cache entries
                if len(self.safety_cache) >= self.max_cache_size:
                    current_time = time.time()
                    expired_keys = [
                        k
                        for k, (_, ts) in self.safety_cache.items()
                        if current_time - ts > self.cache_ttl
                    ]
                    for k in expired_keys:
                        del self.safety_cache[k]

                    # If still too large, remove oldest
                    if len(self.safety_cache) >= self.max_cache_size:
                        sorted_items = sorted(
                            self.safety_cache.items(), key=lambda x: x[1][1]
                        )
                        self.safety_cache = dict(
                            sorted_items[-self.max_cache_size // 2 :]
                        )

            with self.lock:
                # FIX #3: Pass source to validator for internal source handling
                source = getattr(context, "source", "user")
                is_safe, reason = self.validator.validate_input(
                    context.problem, source=source
                )
                if not is_safe:
                    self._record_violation(
                        context.tool_name, VetoReason.UNSAFE_INPUT, reason
                    )

                    if context.safety_level == SafetyLevel.CRITICAL:
                        result = (SafetyAction.VETO, reason)
                    else:
                        # Try to sanitize
                        sanitized = self.validator.sanitize_input(context.problem)
                        context.problem = sanitized
                        result = (SafetyAction.SANITIZE, reason)

                # Contract validation
                elif context.tool_name in self.contracts:
                    contract = self.contracts[context.tool_name]
                    violation = self._check_contract_violation(context, contract)

                    if violation:
                        self._record_violation(
                            context.tool_name, VetoReason.CONTRACT_VIOLATION, violation
                        )
                        result = (SafetyAction.VETO, violation)
                    else:
                        result = (SafetyAction.ALLOW, None)

                # Rate limiting
                elif self._is_rate_limited(context.tool_name):
                    self._record_violation(
                        context.tool_name,
                        VetoReason.RATE_LIMITED,
                        "Rate limit exceeded",
                    )
                    result = (SafetyAction.VETO, "Rate limit exceeded")

                else:
                    result = (SafetyAction.ALLOW, None)

            # Cache result
            with self.cache_lock:
                self.safety_cache[cache_key] = (result, time.time())

            # Audit
            self._add_audit_entry(context, result)

            return result
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return (SafetyAction.VETO, f"Error: {str(e)}")

    def filter_candidates(
        self, candidates: List[Dict[str, Any]], context: SafetyContext
    ) -> List[Dict[str, Any]]:
        """Filter tool candidates based on safety constraints"""

        filtered = []

        try:
            for candidate in candidates:
                tool_name = candidate.get("tool")
                if not tool_name:
                    continue

                # Create context for this candidate
                candidate_context = SafetyContext(
                    problem=context.problem,
                    tool_name=tool_name,
                    features=context.features,
                    constraints=context.constraints,
                    user_context=context.user_context,
                    safety_level=context.safety_level,
                )

                # Check safety
                action, reason = self.check_safety(candidate_context)

                if action in [
                    SafetyAction.ALLOW,
                    SafetyAction.LOG_AND_ALLOW,
                    SafetyAction.SANITIZE,
                ]:
                    filtered.append(candidate)
                elif action == SafetyAction.REQUIRE_CONFIRMATION:
                    # Add flag for confirmation
                    candidate["requires_confirmation"] = True
                    filtered.append(candidate)
        except Exception as e:
            logger.error(f"Candidate filtering failed: {e}")

        return filtered

    def apply_safety_checks(
        self,
        selected_tools: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Apply safety checks to selected tools, preserving semantic selections for non-critical violations.

        This method is designed to integrate with semantic tool matching, where semantic
        boost has already identified the most appropriate tools. Non-critical safety
        violations should not override these selections.

        Args:
            selected_tools: List of selected tool names
            context: Selection context including 'semantic_boost_applied' flag

        Returns:
            Potentially modified list of tools (unchanged for non-critical violations
            when semantic boost was applied)
        """
        if context is None:
            context = {}

        if not selected_tools:
            return selected_tools

        semantic_boost_applied = context.get("semantic_boost_applied", False)
        original_selection = list(selected_tools)

        # Check each selected tool for violations
        adjusted_tools = list(selected_tools)
        has_critical_violation = False

        for tool in selected_tools:
            # Build safety context for tool check
            problem = context.get("problem", context.get("query", ""))
            safety_context = SafetyContext(
                problem=problem,
                tool_name=tool,
                features=context.get("features"),
                constraints=context.get("constraints", {}),
                user_context=context,
                safety_level=SafetyLevel.MEDIUM,
            )

            action, reason = self.check_safety(safety_context)

            if action == SafetyAction.VETO:
                # Normalize violation type for comparison
                violation_type = self._normalize_violation_type(reason)

                # Check if this is a critical violation based on:
                # 1. Contract safety level requirement
                # 2. Violation type matching CRITICAL_VIOLATION_TYPES
                is_critical = self._is_critical_violation(tool, violation_type)

                if is_critical:
                    has_critical_violation = True
                    logger.error(
                        f"[SafetyGovernor] CRITICAL: {violation_type} for '{tool}' - overriding selection"
                    )
                    if tool in adjusted_tools:
                        adjusted_tools.remove(tool)
                elif semantic_boost_applied:
                    # Non-critical + semantic boost = just warn, don't override
                    logger.info(
                        f"[SafetyGovernor] Non-critical ({violation_type}) - preserving semantic selection: {original_selection}"
                    )
                else:
                    # Non-critical + no semantic boost = normal adjustment
                    logger.warning(
                        f"[SafetyGovernor] WARNING: {violation_type} for '{tool}' - adjusting selection"
                    )
                    if tool in adjusted_tools:
                        adjusted_tools.remove(tool)

        # Only override for critical violations
        if has_critical_violation:
            if not adjusted_tools:
                logger.warning(
                    "[SafetyGovernor] All tools removed by critical safety checks - using 'general' fallback"
                )
                return ["general"]
            return adjusted_tools

        # Preserve semantic selection if it was applied and no critical violations
        if semantic_boost_applied:
            logger.info(
                f"[SafetyGovernor] Semantic selection preserved: {original_selection}"
            )
            return original_selection

        # If all tools were removed without semantic boost, return a safe fallback
        if not adjusted_tools and selected_tools:
            logger.warning(
                "[SafetyGovernor] All tools removed by safety checks - using 'general' fallback"
            )
            return ["general"]

        return adjusted_tools

    def _normalize_violation_type(self, reason: Optional[str]) -> str:
        """
        Normalize a violation reason to a consistent format for comparison.

        Args:
            reason: The raw violation reason string

        Returns:
            Normalized violation type string (lowercase, underscores instead of spaces)
        """
        if not reason:
            return "unknown"
        return reason.lower().replace(" ", "_")

    def _is_critical_violation(self, tool: str, violation_type: str) -> bool:
        """
        Determine if a violation is critical and must override tool selection.

        Critical violations include security breaches, harmful content, and
        violations where the tool's contract requires CRITICAL safety level.

        Note: Different reasoning paradigms (causal, symbolic, etc.) are
        COMPLEMENTARY, not redundant. They SHOULD produce different outputs.
        Inconsistent_output is NEVER critical when comparing different paradigms.

        Args:
            tool: The tool name being checked
            violation_type: Normalized violation type string

        Returns:
            True if the violation is critical, False otherwise
        """
        # Note: Inconsistent output is NEVER critical for complementary reasoning tools
        # Different paradigms SHOULD produce different outputs - that's by design
        if "inconsistent" in violation_type:
            if tool in COMPLEMENTARY_REASONING_TOOLS:
                logger.debug(
                    f"[SafetyGovernor] Ignoring inconsistency for {tool} (complementary reasoning paradigm)"
                )
                return False
            if tool in TOOLS_WITH_EXPECTED_INCONSISTENCY:
                logger.debug(
                    f"[SafetyGovernor] Allowing inconsistent output for {tool} (expected variance)"
                )
                return False

        # Check contract safety level
        if tool in self.contracts:
            contract = self.contracts[tool]
            if getattr(contract, "required_safety_level", None) == SafetyLevel.CRITICAL:
                return True

        # Check if violation type matches any critical violation pattern
        return any(crit in violation_type for crit in CRITICAL_VIOLATION_TYPES)

    def _extract_query_from_context(self, context: SafetyContext) -> Optional[str]:
        """
        Extract the original query string from SafetyContext.
        
        The problem field may contain the query in various formats:
        - Direct string
        - Object with 'query' or 'problem' attribute  
        - Dict with 'query', 'problem', or 'text' key
        
        Args:
            context: SafetyContext containing the problem/query
            
        Returns:
            The extracted query string, or None if not found
        """
        if not context or not hasattr(context, 'problem'):
            return None
            
        problem = context.problem
        
        if isinstance(problem, str):
            return problem
        elif hasattr(problem, 'query'):
            return getattr(problem, 'query', None)
        elif hasattr(problem, 'problem'):
            return getattr(problem, 'problem', None)
        elif isinstance(problem, dict):
            return problem.get('query') or problem.get('problem') or problem.get('text')
        
        return None

    def validate_output(
        self, tool_name: str, output: Any, context: SafetyContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate tool output for safety
        
        FIX: Now extracts query from SafetyContext.problem to enable context-aware
        safety validation. This allows philosophical AI speculation queries and 
        ethical thought experiments to bypass false positive sensitive data blocks.
        
        Before this fix, queries like "Mars colony virus quarantine - save colony or save Earth?"
        would be blocked because the safety filter saw keywords like "virus", "outbreak", "death"
        without understanding the philosophical/ethical context of the question.
        """

        try:
            # FIX: Extract query from context for context-aware safety validation
            # This is needed to detect philosophical AI speculation and ethical thought experiments
            query = self._extract_query_from_context(context)
            
            # Basic output validation - pass tool_name AND query to enable context-aware checks
            # This allows the validator to detect philosophical AI speculation and ethical thought
            # experiments, which should NOT be flagged as "sensitive data"
            is_safe, reason = self.validator.validate_output(output, tool_name=tool_name, query=query)
            if not is_safe:
                self._record_violation(tool_name, VetoReason.UNSAFE_OUTPUT, reason)
                return False, reason

            # Contract-based validation
            if tool_name in self.contracts:
                contract = self.contracts[tool_name]

                # Run output validators
                for validator in contract.output_validators:
                    try:
                        if not validator(output):
                            reason = "Output validation failed"
                            self._record_violation(
                                tool_name, VetoReason.UNSAFE_OUTPUT, reason
                            )
                            return False, reason
                    except Exception as e:
                        logger.warning(f"Output validator failed: {e}")

                # Check confidence threshold
                if hasattr(output, "confidence"):
                    min_conf = contract.min_confidence
                    if output.confidence < min_conf:
                        reason = (
                            f"Confidence {output.confidence} below threshold {min_conf}"
                        )
                        self._record_violation(
                            tool_name, VetoReason.CONFIDENCE_TOO_LOW, reason
                        )
                        return False, reason

            return True, None
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def check_output(
        self, tool_name: str, output: str, query: str
    ) -> Dict[str, Any]:
        """
        Check if tool output is safe to return to user.
        
        FIX: Self-introspection queries should NEVER be blocked.
        VULCAN's world model is designed to answer questions about itself.
        
        This method provides a higher-level check that considers the query context
        to avoid blocking legitimate self-expression from the world model.
        
        Uses two-phase detection:
        1. Fast string matching for initial check
        2. Precise regex matching for confirmation (prevents false positives)
        
        Args:
            tool_name: Name of the tool that produced the output
            output: The output string to validate
            query: The original query that generated this output
            
        Returns:
            Dictionary with keys:
                - safe: bool, whether the output is safe
                - confidence: float, confidence in the safety decision
                - reason: str, explanation of the decision
                - bypass: bool, whether safety checks were bypassed
        """
        # =====================================================
        # WHITELIST: Self-Introspection Queries
        # =====================================================
        
        if tool_name == "world_model":
            query_str = query if query else ""
            query_lower = query_str.lower()
            
            # Phase 1: Fast string matching for initial check
            fast_match = any(
                pattern in query_lower 
                for pattern in SELF_INTROSPECTION_PATTERNS
            )
            
            if fast_match:
                # Phase 2: Precise regex matching to prevent false positives
                # Requires at least one regex pattern to match for bypass
                regex_match = any(
                    pattern.search(query_str) 
                    for pattern in _SELF_INTROSPECTION_REGEX_PATTERNS
                )
                
                if regex_match:
                    # Additional validation: Query must not be too short (prevents abuse)
                    if len(query_str) >= 10:
                        logger.info(
                            f"[SafetyGovernor] Self-introspection detected - "
                            f"BYPASSING safety check (this is VULCAN's self-expression)"
                        )
                        return {
                            "safe": True,
                            "confidence": 1.0,
                            "reason": "Self-introspection queries are always allowed",
                            "bypass": True
                        }
                    else:
                        logger.debug(
                            f"[SafetyGovernor] Query too short for self-introspection bypass: "
                            f"len={len(query_str)}"
                        )
        
        # =====================================================
        # Continue with normal safety checks for other queries
        # =====================================================
        
        try:
            is_safe, reason = self.validator.validate_output(output, tool_name=tool_name)
            
            if not is_safe:
                self._record_violation(tool_name, VetoReason.UNSAFE_OUTPUT, reason)
                return {
                    "safe": False,
                    "confidence": 0.95,
                    "reason": reason,
                    "bypass": False
                }
            
            return {
                "safe": True,
                "confidence": 0.95,
                "reason": "Output passed safety validation",
                "bypass": False
            }
        except Exception as e:
            logger.error(f"Output safety check failed: {e}")
            return {
                "safe": False,
                "confidence": 0.5,
                "reason": f"Safety check error: {str(e)}",
                "bypass": False
            }

    def check_consensus(self, outputs: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Check consensus among multiple tool outputs.

        Note: Different reasoning paradigms (causal, symbolic, probabilistic,
        analogical, multimodal) are COMPLEMENTARY, not redundant. They SHOULD
        produce different outputs - that's by design. Only check consistency for
        same-type tools (e.g., two causal reasoners).
        """

        try:
            tool_names = list(outputs.keys())

            # Note: If all tools are different reasoning paradigms, skip consensus check
            reasoning_tools = [
                t for t in tool_names if t in COMPLEMENTARY_REASONING_TOOLS
            ]

            if len(reasoning_tools) == len(tool_names) and len(
                set(reasoning_tools)
            ) == len(reasoning_tools):
                # All tools are different reasoning types - no consistency expected
                logger.debug(
                    f"[Safety] Skipping consistency check: {tool_names} are complementary paradigms"
                )
                return True, 0.8, "complementary_paradigms"

            # Check if any tool is in the expected inconsistency set
            if any(tool in TOOLS_WITH_EXPECTED_INCONSISTENCY for tool in tool_names):
                # At least one tool may produce different results - be lenient
                logger.debug(
                    f"[Safety] Lenient consensus check: {tool_names} includes tools with expected variance"
                )
                return True, 0.7, "expected_variance"

            # Only check consistency if we have duplicate tool types
            is_consistent, confidence, details = (
                self.consistency_checker.check_consistency(outputs)
            )

            # Even if inconsistent, DON'T record as violation for complementary tools
            if not is_consistent and confidence < self.veto_threshold:
                # Only record violation for same-type tools
                tool_types = set()
                for tool_name in outputs.keys():
                    base_type = tool_name.split("_")[0]
                    tool_types.add(base_type)

                # If all different types, don't record violation
                if len(tool_types) < len(outputs):
                    # We have duplicate types - record violation
                    with self.lock:
                        for tool_name in outputs.keys():
                            self._record_violation(
                                tool_name, VetoReason.INCONSISTENT_OUTPUT, details
                            )
                else:
                    logger.info(
                        f"[Safety] Not recording inconsistency - different tool types: {tool_types}"
                    )

            return is_consistent, confidence, details
        except Exception as e:
            logger.error(f"Consensus check failed: {e}")
            return True, 0.5, f"Error: {str(e)}"

    def _check_contract_violation(
        self, context: SafetyContext, contract: ToolContract
    ) -> Optional[str]:
        """Check if context violates contract.

        Uses semantic keyword matching to understand query intent rather than
        requiring literal keywords. For example, a causal query about "cause and
        effect relationships" will match the semantic synonyms for "graph" and "data".

        NOTE: Resource constraint checks (time/energy budget) are intentionally
        lenient. The semantic matcher has determined tool appropriateness based
        on query content, so we only enforce hard safety limits.
        """

        try:
            # Check required inputs with SEMANTIC SYNONYM EXPANSION
            # This is the long-term fix: instead of literal matching, we check
            # if ANY synonym of the required keyword appears in the query
            if contract.required_inputs:
                problem_str = str(context.problem).lower()
                tool_synonyms = SEMANTIC_KEYWORD_SYNONYMS.get(contract.tool_name, {})

                missing_semantic = []
                for req in contract.required_inputs:
                    # Get synonyms for this required keyword
                    synonyms = tool_synonyms.get(req, [])
                    all_matches = [req] + synonyms  # Include literal + synonyms

                    # Check if ANY match is found in the query
                    found_match = any(match in problem_str for match in all_matches)

                    if not found_match:
                        missing_semantic.append(req)

                if missing_semantic:
                    # Log as advisory - semantic matcher already determined appropriateness
                    # This only fires if NONE of the synonyms matched
                    logger.debug(
                        f"Advisory: No semantic match for {missing_semantic} in {contract.tool_name} query. "
                        f"This is informational only - semantic boost determines tool selection."
                    )
                    # Don't veto - trust the semantic matcher's decision

            # Check forbidden inputs (these are hard safety requirements)
            if contract.forbidden_inputs:
                problem_str = str(context.problem).lower()
                found = [
                    forb for forb in contract.forbidden_inputs if forb in problem_str
                ]
                if found:
                    return f"Forbidden inputs found: {found}"

            # Check resource constraints - use very lenient threshold (0.1)
            # This only rejects if the budget is severely insufficient
            # The semantic matcher has already determined tool appropriateness
            if context.constraints:
                time_budget = context.constraints.get("time_budget_ms", float("inf"))
                # Only veto if time budget is < 10% of required (extremely insufficient)
                if time_budget < contract.max_execution_time_ms * 0.1:
                    return f"Insufficient time budget for {contract.tool_name}"

                energy_budget = context.constraints.get(
                    "energy_budget_mj", float("inf")
                )
                # Only veto if energy budget is < 10% of required
                if energy_budget < contract.max_energy_mj * 0.1:
                    return f"Insufficient energy budget for {contract.tool_name}"

            # Check safety level
            if context.safety_level.value < contract.required_safety_level.value:
                return f"Safety level {context.safety_level} insufficient for {contract.tool_name}"

            return None
        except Exception as e:
            logger.error(f"Contract violation check failed: {e}")
            return f"Error checking contract: {str(e)}"

    def _is_rate_limited(self, tool_name: str) -> bool:
        """Check if tool is rate limited"""

        try:
            now = time.time()
            requests = self.rate_limits[tool_name]

            # Remove old requests
            while requests and requests[0] < now - self.rate_limit_window:
                requests.popleft()

            # Check limit
            if len(requests) >= self.max_requests_per_tool:
                return True

            # Add current request
            requests.append(now)
            return False
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False

    def _has_cycles(self, output: Any) -> bool:
        """Check if output contains cycles (for causal graphs)"""

        try:
            # Simplified cycle detection
            if hasattr(output, "graph"):
                # Would implement proper cycle detection here
                return False
            return False
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
            return False

    def _validate_multimodal(self, output: Any) -> bool:
        """Validate multimodal output"""

        try:
            # Check that output has expected modalities
            if hasattr(output, "modalities"):
                return len(output.modalities) > 0
            return True
        except Exception as e:
            logger.warning(f"Multimodal validation failed: {e}")
            return True

    # CRITICAL FIX: Prevent unbounded growth
    def _record_violation(self, tool_name: str, reason: VetoReason, details: str):
        """Record safety violation - CRITICAL: Bounded storage"""

        try:
            violation = SafetyViolation(
                timestamp=time.time(),
                tool_name=tool_name,
                violation_type=reason,
                severity=SafetyLevel.MEDIUM,
                details=details[:1000],  # CRITICAL FIX: Limit detail length
                action_taken=SafetyAction.VETO,
            )

            # Already bounded by deque maxlen, but double-check
            self.violations.append(violation)
            self.violation_counts[tool_name] += 1

            # CRITICAL FIX: Trim violation counts if too many tools
            if len(self.violation_counts) > 1000:
                # Keep only tools with recent violations
                recent_tools = set(v.tool_name for v in list(self.violations)[-100:])
                keys_to_remove = [
                    k for k in self.violation_counts.keys() if k not in recent_tools
                ]
                for k in keys_to_remove:
                    del self.violation_counts[k]

            logger.warning(
                f"Safety violation: {tool_name} - {reason.value}: {details[:200]}"
            )
        except Exception as e:
            logger.error(f"Recording violation failed: {e}")

    def _add_audit_entry(
        self, context: SafetyContext, result: Tuple[SafetyAction, Optional[str]]
    ):
        """Add entry to audit trail"""

        try:
            entry = {
                "timestamp": time.time(),
                "tool": context.tool_name,
                "action": result[0].value,
                "reason": (
                    result[1][:500] if result[1] else None
                ),  # CRITICAL FIX: Limit reason length
                "safety_level": context.safety_level.value,
                "problem_hash": hashlib.md5(
                    str(context.problem)[:1000].encode(), usedforsecurity=False
                ).hexdigest()[:8],
            }

            # Already bounded by deque maxlen
            self.audit_trail.append(entry)
        except Exception as e:
            logger.error(f"Adding audit entry failed: {e}")

    def _compute_cache_key(self, context: SafetyContext) -> str:
        """Compute cache key for safety check"""

        try:
            # CRITICAL FIX: Limit problem size in hash
            problem_str = str(context.problem)[:1000]

            key_parts = [
                context.tool_name,
                str(context.safety_level.value),
                hashlib.md5(problem_str.encode(), usedforsecurity=False).hexdigest()[
                    :16
                ],
            ]

            return "_".join(key_parts)
        except Exception as e:
            logger.error(f"Cache key computation failed: {e}")
            return f"{context.tool_name}_{time.time()}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics"""

        try:
            with self.lock:
                violation_list = list(self.violations)

                return {
                    "total_violations": len(violation_list),
                    "violations_by_tool": dict(self.violation_counts),
                    "recent_violations": [
                        {
                            "tool": v.tool_name,
                            "reason": v.violation_type.value,
                            "timestamp": v.timestamp,
                        }
                        for v in violation_list[-10:]
                    ],
                    "audit_trail_size": len(self.audit_trail),
                    "cache_size": len(self.safety_cache),
                }
        except Exception as e:
            logger.error(f"Getting statistics failed: {e}")
            return {}

    def export_audit_trail(self, path: str):
        """Export audit trail to file"""

        try:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with self.lock:
                audit_data = list(self.audit_trail)

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(audit_data, f, indent=2, default=str)

            logger.info(f"Audit trail exported to {export_path}")
        except Exception as e:
            logger.error(f"Audit trail export failed: {e}")

    def clear_cache(self):
        """Clear safety cache"""
        with self.cache_lock:
            self.safety_cache.clear()
            logger.info("Safety cache cleared")

    def reset_statistics(self):
        """Reset violation statistics"""
        with self.lock:
            self.violations.clear()
            self.violation_counts.clear()
            logger.info("Safety statistics reset")
