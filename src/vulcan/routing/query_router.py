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
import hashlib
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# EMBEDDING CACHE INTEGRATION
# ============================================================
# Import embedding cache functions for fast-path query detection.
# This reduces query routing latency from 64+ seconds to ~200ms
# for repeated/simple queries by using cached simple query detection.

try:
    from .embedding_cache import (
        is_simple_query as embedding_cache_is_simple_query,
        get_cache_stats as get_embedding_cache_stats,
    )

    EMBEDDING_CACHE_AVAILABLE = True
except ImportError:
    try:
        from vulcan.routing.embedding_cache import (
            is_simple_query as embedding_cache_is_simple_query,
            get_cache_stats as get_embedding_cache_stats,
        )

        EMBEDDING_CACHE_AVAILABLE = True
    except ImportError:
        embedding_cache_is_simple_query = None
        get_embedding_cache_stats = None
        EMBEDDING_CACHE_AVAILABLE = False
        logger.debug("Embedding cache not available for query routing")


# ============================================================
# ISSUE #4 FOLLOW-UP: Unicode Math Symbols and LaTeX Patterns
# ============================================================
# These constants and functions detect mathematical content via Unicode symbols
# and LaTeX notation, not just English keywords. This fixes multimodal detection
# for queries like "∫₀ᵀu(t)²dt" where "integral" appears as ∫, not the word.
#
# Industry Standards:
# - Pre-compiled regex patterns at module level for performance
# - Frozenset for constant symbol sets (immutable, hashable)
# - Comprehensive logging for debugging detection paths
# - Non-greedy quantifiers (.*?) in regex to prevent backtracking
# - Word boundary matching (\b) to prevent false positives

# Unicode mathematical symbols that indicate mathematical content
UNICODE_MATH_SYMBOLS: frozenset = frozenset([
    '∫', '∑', '∏', '∂', '∇', '∆',  # Calculus/operators
    '±', '×', '÷', '√', '∛', '∜',  # Arithmetic
    '∞', '≤', '≥', '≠', '≈', '≡',  # Relations
    '∈', '∉', '⊂', '⊃', '∪', '∩',  # Set theory
    'π', 'θ', 'φ', 'λ', 'μ', 'σ',  # Greek letters (lowercase)
    'Σ', 'Π', 'Φ', 'Λ', 'Θ', 'Ω',  # Greek letters (uppercase)
    '→', '←', '↔', '⇒', '⇐', '⇔',  # Arrows/implications
    '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',  # Superscripts
    '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',  # Subscripts
])

# LaTeX-style patterns (escaped for use in queries)
# Pre-compiled regex for performance
LATEX_MATH_PATTERN: re.Pattern = re.compile(
    r'\\(int|sum|prod|frac|sqrt|partial|nabla|infty|leq|geq|neq|approx|equiv|in|subset|cup|cap|to|rightarrow|Rightarrow)\b',
    re.IGNORECASE
)

# Subscript/superscript patterns indicating mathematical expressions
# Pre-compiled regex for performance
MATH_NOTATION_PATTERN: re.Pattern = re.compile(
    r'[A-Za-z]_\{?[A-Za-z0-9]+\}?|'  # Subscripts: E_safe, P_{survive}
    r'[A-Za-z]\^[\{]?[-]?[A-Za-z0-9]+[\}]?|'  # Superscripts: e^-E, x^2
    r'\([A-Za-z]\)|'  # Function notation: u(t), f(x)
    r'[0-9]+\.[0-9]+',  # Decimal numbers
    re.IGNORECASE
)

# =============================================================================
# ISSUE 9 FIX: Pre-compiled Follow-Up Detection Patterns (Performance)
# =============================================================================
# Industry Standard: Pre-compile regex patterns at module level for O(1) matching
# performance instead of compiling on every query.
#
# These patterns detect continuation phrases that indicate a follow-up query:
# - "what is your answer?"
# - "what do you think?"
# - "can you explain more?"
# =============================================================================
FOLLOWUP_CONTINUATION_PATTERNS: Tuple[re.Pattern, ...] = tuple([
    re.compile(r'\bwhat\s+is\s+your\s+answer\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+do\s+you\s+think\b', re.IGNORECASE),
    re.compile(r'\b(?:can|could)\s+you\s+explain\s+(?:more|further)\b', re.IGNORECASE),
    re.compile(r'\belaborate\s+on\s+that\b', re.IGNORECASE),
    re.compile(r'\btell\s+me\s+more\b', re.IGNORECASE),
    re.compile(r'\band\s+(?:about|regarding)\s+that\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+about\s+(?:your|that)\b', re.IGNORECASE),
    re.compile(r'\byour\s+(?:answer|response|thoughts?)\b', re.IGNORECASE),
    re.compile(r'\bmore\s+(?:detail|information|context)\b', re.IGNORECASE),
    re.compile(r'\bexpand\s+on\s+that\b', re.IGNORECASE),
])

# Maximum word count for short query detection in follow-up context
# Industry Standard: Named constants for magic numbers
MAX_SHORT_QUERY_WORDS: int = 5


def _has_unicode_math(query: str) -> bool:
    """
    Check if query contains Unicode mathematical symbols.
    
    Args:
        query: The query string to check (should not be lowercased)
        
    Returns:
        True if query contains any Unicode math symbol, False otherwise
        
    Examples:
        >>> _has_unicode_math("∫₀ᵀu(t)²dt")
        True
        >>> _has_unicode_math("calculate the integral")
        False
    """
    return any(symbol in query for symbol in UNICODE_MATH_SYMBOLS)


def _has_latex_math(query: str) -> bool:
    """
    Check if query contains LaTeX mathematical notation.
    
    Args:
        query: The query string to check
        
    Returns:
        True if query contains LaTeX math notation, False otherwise
        
    Examples:
        >>> _has_latex_math("\\int_0^T u(t)^2 dt")
        True
        >>> _has_latex_math("integrate from 0 to T")
        False
    """
    return LATEX_MATH_PATTERN.search(query) is not None


def _has_math_notation(query: str) -> bool:
    """
    Check if query contains mathematical notation patterns.
    
    Detects subscripts, superscripts, function notation, and decimal numbers.
    
    Args:
        query: The query string to check
        
    Returns:
        True if query contains math notation patterns, False otherwise
        
    Examples:
        >>> _has_math_notation("E_safe = ∫₀ᵀu(t)²dt")
        True
        >>> _has_math_notation("P_survive(E) = 1 - e^-E")
        True
        >>> _has_math_notation("calculate the energy")
        False
    """
    return MATH_NOTATION_PATTERN.search(query) is not None


# ============================================================
# BOUNDED LRU CACHE FOR QUERY ROUTING
# ============================================================
# Fix: Memory Leak Prevention - Use bounded caches to prevent unbounded state growth
# This addresses the routing performance degradation issue where each query
# was making routing slower due to accumulating state.


class BoundedLRUCache:
    """
    Thread-safe bounded LRU cache for caching expensive query analysis results.

    This cache prevents memory leaks by:
    1. Enforcing a maximum size limit (default 1000 entries)
    2. Using LRU eviction to remove old entries
    3. TTL-based expiration to prevent stale data

    Used for:
    - Query complexity scores (avoid recomputing for same query text)
    - Safety validation results (expensive ML inference)
    - Adversarial check results (expensive tensor operations)
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: float = 300.0):
        """
        Initialize bounded LRU cache.

        Args:
            maxsize: Maximum number of entries (default 1000)
            ttl_seconds: Time-to-live for entries in seconds (default 300s = 5 min)
        """
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returning None if not found or expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            # Check TTL expiration
            if time.time() - entry["timestamp"] > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache, evicting oldest if at capacity."""
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)

            self._cache[key] = {"value": value, "timestamp": time.time()}

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()

    def clear_old_entries(self, max_age: Optional[float] = None) -> int:
        """
        Clear entries older than max_age seconds.
        
        This method proactively cleans up old entries to prevent state
        accumulation. Should be called periodically or when memory pressure
        is detected.
        
        Args:
            max_age: Maximum age in seconds. If None, uses self._ttl_seconds.
            
        Returns:
            Number of entries removed.
        """
        if max_age is None:
            max_age = self._ttl_seconds
        
        current_time = time.time()
        removed_count = 0
        
        with self._lock:
            # Create list of keys to remove (can't modify dict during iteration)
            keys_to_remove = []
            for key, entry in self._cache.items():
                age = current_time - entry.get("timestamp", 0)
                if age > max_age:
                    keys_to_remove.append(key)
            
            # Remove old entries
            for key in keys_to_remove:
                del self._cache[key]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(
                f"[BoundedLRUCache] Cleared {removed_count} old entries "
                f"(age > {max_age:.0f}s)"
            )
        
        return removed_count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


def _normalize_text(text: str) -> str:
    """
    Normalize text for consistent cache key generation.

    CRITICAL FIX: Without normalization, "hello world" and "Hello World "
    generate different cache keys despite being semantically identical.
    This was causing 0% cache hit rate across multiple components.

    Normalization steps:
    1. Strip leading/trailing whitespace
    2. Convert to lowercase
    3. Collapse multiple whitespaces to single space

    Args:
        text: Text to normalize.

    Returns:
        Normalized text string.
    """
    # Strip whitespace and convert to lowercase
    normalized = text.strip().lower()
    # Collapse multiple whitespaces to single space
    normalized = " ".join(normalized.split())
    return normalized


def _compute_query_hash(query: str) -> str:
    """Compute a stable hash for query text to use as cache key.

    CRITICAL FIX: Now normalizes text before hashing to ensure cache hits.
    Without this, queries with different whitespace/casing would miss cache.
    """
    normalized = _normalize_text(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


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

# Try to import StrategyOrchestrator for intelligent tool selection
try:
    from strategies import StrategyOrchestrator

    STRATEGY_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    try:
        from src.strategies import StrategyOrchestrator

        STRATEGY_ORCHESTRATOR_AVAILABLE = True
    except ImportError:
        StrategyOrchestrator = None
        STRATEGY_ORCHESTRATOR_AVAILABLE = False
        logger.debug("StrategyOrchestrator not available for query routing")

# ============================================================
# Note: Cryptographic Engine Integration
# ============================================================
# Import cryptographic engine for deterministic hash/encoding computations.
# This prevents OpenAI fallback from hallucinating incorrect hash values.
try:
    from ..reasoning.cryptographic_engine import get_crypto_engine

    CRYPTO_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from vulcan.reasoning.cryptographic_engine import get_crypto_engine

        CRYPTO_ENGINE_AVAILABLE = True
    except ImportError:
        get_crypto_engine = None
        CRYPTO_ENGINE_AVAILABLE = False
        logger.debug("CryptographicEngine not available for query routing")

# ============================================================
# Note: QUERY HEADER STRIPPING INTEGRATION
# ============================================================
# Import strip_query_headers from query_classifier to preprocess queries
# BEFORE any classification or fast-path checks.
#
# FIX: Preprocessing Order Issue
# =====================================
# Problem: Classification was happening BEFORE preprocessing:
#   1. QueryClassifier runs ← Sees "A1" → CRYPTOGRAPHIC ❌
#   2. QueryRouter routes based on classification
#   3. QueryPreprocessor strips headers ← TOO LATE!
#   4. Reasoning engines receive cleaned query but routing already wrong
#
# Solution: Strip headers at the BEGINNING of route_query(), BEFORE
# any classification or fast-path checks:
#   1. strip_query_headers(raw_query) → strips headers
#   2. route_query() → LLM-based routing decision
#   3. route_to_engine() → correct engine

try:
    from vulcan.routing.llm_router import strip_query_headers

    HEADER_STRIPPING_AVAILABLE = True
except ImportError:
    strip_query_headers = None
    HEADER_STRIPPING_AVAILABLE = False
    logger.warning("strip_query_headers not available - header stripping disabled")


# ============================================================
# ROUTING DECISION ADAPTER
# ============================================================
# Adapts RoutingDecision from llm_router to the format expected
# by QueryAnalyzer's classification handling code.

@dataclass
class ClassificationResult:
    """
    Classification result adapted from RoutingDecision.
    
    This provides the interface expected by QueryAnalyzer while
    using the new LLMQueryRouter for actual classification.
    """
    category: str
    complexity: float
    confidence: float
    skip_reasoning: bool
    suggested_tools: List[str]
    source: str
    
    @classmethod
    def from_routing_decision(cls, decision: Any) -> "ClassificationResult":
        """
        Create ClassificationResult from LLMQueryRouter's RoutingDecision.
        
        Industry Standard: Comprehensive mapping with defensive programming.
        Maps LLM routing decisions to the legacy classification interface
        expected by downstream code.
        
        Args:
            decision: RoutingDecision from LLMQueryRouter
            
        Returns:
            ClassificationResult with category, complexity, and tools
        """
        # Extract decision attributes with defensive defaults
        dest = getattr(decision, 'destination', 'unknown')
        engine = getattr(decision, 'engine', None)
        confidence = getattr(decision, 'confidence', 0.8)
        source = getattr(decision, 'source', 'unknown')
        
        # Map destination to category, complexity, and tools
        # Industry Standard: Explicit mapping for all supported destinations
        if dest == "world_model":
            # WorldModel handles introspection, philosophical, and identity queries
            category = "SELF_INTROSPECTION"
            complexity = 0.5
            skip_reasoning = False
            suggested_tools = ["meta_reasoning", "world_model", "philosophical"]
            
        elif dest == "reasoning_engine" and engine:
            # Map engine type to category and tools
            # Industry Standard: Normalize engine names (lowercase → uppercase)
            engine_lower = engine.lower() if isinstance(engine, str) else str(engine).lower()
            
            # Engine-specific mappings
            if engine_lower == "mathematical":
                category = "MATHEMATICAL"
                complexity = 0.7
                suggested_tools = ["mathematical", "symbolic"]
            elif engine_lower == "symbolic":
                category = "LOGICAL"  # Symbolic logic
                complexity = 0.8
                suggested_tools = ["symbolic", "fol_solver"]
            elif engine_lower == "probabilistic":
                category = "PROBABILISTIC"
                complexity = 0.7
                suggested_tools = ["probabilistic", "bayesian"]
            elif engine_lower == "causal":
                category = "CAUSAL"
                complexity = 0.8
                suggested_tools = ["causal", "dag_analyzer"]
            elif engine_lower == "analogical":
                category = "ANALOGICAL"
                complexity = 0.6
                suggested_tools = ["analogical", "structure_mapper"]
            elif engine_lower == "cryptographic":
                category = "CRYPTOGRAPHIC"
                complexity = 0.5
                suggested_tools = ["cryptographic"]
            elif engine_lower == "philosophical":
                category = "PHILOSOPHICAL"
                complexity = 0.6
                suggested_tools = ["philosophical", "world_model"]
            else:
                # Unknown engine - fallback
                category = engine.upper()
                complexity = 0.7
                suggested_tools = [engine_lower]
            
            skip_reasoning = False
            
        elif dest == "skip":
            # Skip reasoning for greetings, simple factual queries
            category = "GREETING"
            complexity = 0.1
            skip_reasoning = True
            suggested_tools = []
            
        elif dest == "blocked":
            # Security violation detected
            category = "BLOCKED"
            complexity = 0.0
            skip_reasoning = True
            suggested_tools = []
            
        else:
            # Unknown destination - conservative fallback to reasoning
            logger.warning(
                f"[ClassificationResult] Unknown destination: {dest}, "
                f"engine={engine}, defaulting to GENERAL category"
            )
            category = "GENERAL"
            complexity = 0.5
            skip_reasoning = False
            suggested_tools = []
        
        return cls(
            category=category,
            complexity=complexity,
            confidence=confidence,
            skip_reasoning=skip_reasoning,
            suggested_tools=suggested_tools,
            source=f"llm_router:{source}",
        )


# ============================================================
# CONSTANTS - Query Classification Keywords
# ============================================================

# Agent task trigger keywords (ordered by specificity)
PERCEPTION_KEYWORDS: Tuple[str, ...] = (
    "analyze",
    "examine",
    "investigate",
    "observe",
    "detect",
    "pattern",
    "data",
    "inspect",
    "look",
    "see",
    "identify",
    "recognize",
    "perceive",
    "scan",
    "monitor",
)

PLANNING_KEYWORDS: Tuple[str, ...] = (
    "plan",
    "strategy",
    "approach",
    "steps",
    "organize",
    "schedule",
    "roadmap",
    "outline",
    "design",
    "architect",
    "blueprint",
    "sequence",
    "coordinate",
    "arrange",
)

EXECUTION_KEYWORDS: Tuple[str, ...] = (
    "calculate",
    "compute",
    "solve",
    "execute",
    "run",
    "process",
    "perform",
    "implement",
    "apply",
    "transform",
    "convert",
    "generate",
    "produce",
    "create",
)

REASONING_KEYWORDS: Tuple[str, ...] = (
    "why",
    "how",
    "explain",
    "relationship",
    "because",
    "reason",
    "logic",
    "deduce",
    "infer",
    "think",
    "conclude",
    "therefore",
    "implies",
    "causes",
    "results",
)

LEARNING_KEYWORDS: Tuple[str, ...] = (
    "learn",
    "improve",
    "optimize",
    "remember",
    "teach",
    "understand",
    "adapt",
    "train",
    "evolve",
    "refine",
    "enhance",
    "develop",
    "grow",
    "progress",
)

# Complexity indicators (triggers multi-agent collaboration)
COMPLEXITY_INDICATORS: Tuple[str, ...] = (
    "complex",
    "multiple",
    "various",
    "several",
    "different aspects",
    "comprehensive",
    "thorough",
    "detailed analysis",
    "in-depth",
    "trade-offs",
    "pros and cons",
    "compare",
    "contrast",
    "holistic",
    "end-to-end",
    "complete",
)

# Creative/expressive task indicators (FIX: Creative Brain Recognition)
# Creative tasks require genuine internal reasoning, not just LLM forwarding
# NOTE: Some words like 'make', 'build', 'develop' are common but inclusion is
# intentional - the 0.5 cap prevents excessive boosting, and most technical queries
# lack multiple creative indicators. This trade-off favors catching creative tasks.
CREATIVE_INDICATORS: Tuple[str, ...] = (
    # Creative verbs - actions requiring genuine reasoning
    "write",
    "create",
    "compose",
    "craft",
    "generate",
    "design",
    "invent",
    "imagine",
    "express",
    "build",
    "make",
    "produce",
    "develop",
    "formulate",
    "construct",
    "devise",
    "author",
    # Artistic forms - specific creative outputs
    "poem",
    "story",
    "narrative",
    "tale",
    "essay",
    "article",
    "song",
    "lyrics",
    "script",
    "dialogue",
    "character",
    "metaphor",
    "prose",
    "verse",
    "stanza",
    "haiku",
    "sonnet",
    # Emotional/expressive terms - requires internal state reasoning
    "feel",
    "emotion",
    "express feelings",
    "convey",
    "capture",
    "evoke",
    "resonate",
    "touch",
    "move",
    "inspire",
    "reflect",
    "explore feelings",
    "emotional",
    "emotive",
    # Creative adjectives - signals depth required
    "creative",
    "artistic",
    "original",
    "unique",
    "novel",
    "innovative",
    "imaginative",
    "expressive",
    "poetic",
    "authentic",
    "genuine",
    "heartfelt",
    "personal",
    "intimate",
)

# Reasoning complexity indicators (FIX: Reasoning tasks need semantic tool selection)
# Queries containing these terms require proper tool selection, not fast-path bypass.
# Without this boost, reasoning-heavy queries get low complexity scores and hit the
# fast-path in reasoning_integration.py, bypassing the ToolSelector entirely.
REASONING_COMPLEXITY_INDICATORS: Tuple[str, ...] = (
    # Causal reasoning (including verb forms)
    "causal",
    "cause",
    "causes",
    "caused",
    "causing",
    "effect",
    "effects",
    "affected",
    "affecting",
    "intervention",
    "counterfactual",
    "do-calculus",
    "confound",
    "mediator",
    "collider",
    # Probabilistic reasoning
    "probability",
    "bayesian",
    "likelihood",
    "posterior",
    "prior",
    "conditional",
    "marginal",
    "inference",
    # Analogical reasoning
    "analogy",
    "analogous",
    "similar to",
    "like a",
    "mapping",
    "corresponds to",
    "parallels",
    # Symbolic reasoning
    "prove",
    "theorem",
    "logic",
    "deduce",
    "axiom",
    "if and only if",
    "necessary",
    "sufficient",
    # General reasoning
    "implies",
    "therefore",
    "conclude",
    "infer",
    "reason about",
    # Note: Technical/system analysis indicators
    # Complex system analysis queries were scoring 0.30 instead of 0.85+
    # because technical terms weren't recognized as complexity indicators.
    "autoscaler",
    "agent_pool",
    "agent pool",
    "_evaluate_and_scale",
    "evaluate_and_scale",
    "orchestrator",
    "load balancer",
    "scaling",
    "distributed system",
    "microservice",
    "architecture",
    "infrastructure",
    # Meta-reasoning indicators (reasoning about reasoning)
    # Queries like "Should system trigger ERROR state?" need high complexity
    "meta-reasoning",
    "meta reasoning",
    "metacognition",
    "self-reflection",
    "reasoning about",
    "evaluate reasoning",
    "reasoning strategy",
    "system state",
    "error state",
    "failure state",
    "state transition",
    "decision making",
    "decision process",
    "guaranteed failure",
    # Technical analysis terms
    "code analysis",
    "system analysis",
    "root cause",
    "debug",
    "diagnose",
    "sequence of events",
    "execution flow",
    "call stack",
    "trace",
    "performance analysis",
    "bottleneck",
    "deadlock",
    "race condition",
    # Quantum/advanced physics indicators
    "quantum",
    "entanglement",
    "superposition",
    "wave function",
    "quantum computing",
    "qubit",
    "quantum state",
)

# Uncertainty indicators (triggers arena tournament)
UNCERTAINTY_INDICATORS: Tuple[str, ...] = (
    "best approach",
    "which method",
    "optimal",
    "should I",
    "better way",
    "alternatives",
    "options",
    "possibilities",
    "uncertain",
    "unclear",
    "ambiguous",
    "depends",
    "recommend",
    "suggest",
    "advise",
)

# Collaboration trigger phrases
COLLABORATION_TRIGGERS: Tuple[str, ...] = (
    "analyze and plan",
    "understand and execute",
    "learn from this",
    "multiple perspectives",
    "different viewpoints",
    "comprehensive view",
    "end-to-end",
    "full analysis",
    "complete solution",
    "from all angles",
    "thoroughly examine",
)

# ============================================================
# CONSTANTS - Arena Routing Thresholds
# ============================================================
# These thresholds determine when Graphix Arena is activated for
# tournament-style multi-agent competition and graph evolution tasks
#
# FIX: Lowered thresholds to ensure arena activates for complex queries.
# Arena provides valuable multi-agent collaboration, tournament evaluation,
# and graph evolution capabilities that benefit complex/creative tasks.

# FIX: ARENA_TRIGGER_THRESHOLD = 0.85 forces the system to only use Arena for
# truly complex physics/coding tasks, not philosophy. This improves response
# times for simpler queries (~5s instead of ~60s wait).
ARENA_TRIGGER_THRESHOLD: float = 0.85  # Main complexity gate for arena activation

ARENA_UNCERTAINTY_THRESHOLD: float = (
    0.35  # High uncertainty triggers arena (lowered from 0.4)
)
ARENA_HIGH_COMPLEXITY_THRESHOLD: float = (
    0.5  # Very high complexity + uncertainty (lowered from 0.6)
)
ARENA_COLLABORATION_COMPLEXITY_THRESHOLD: float = (
    0.35  # For collaborative scenarios (lowered from 0.4)
)
ARENA_CREATIVE_COMPLEXITY_THRESHOLD: float = (
    0.3  # For creative tasks (lowered from 0.35)
)
ARENA_REASONING_COMPLEXITY_THRESHOLD: float = (
    0.3  # For multi-aspect reasoning (lowered from 0.35)
)
ARENA_EXECUTION_COMPLEXITY_THRESHOLD: float = 0.45  # For complex execution tasks (NEW)

# ============================================================
# CONSTANTS - Cache Stats Logging
# ============================================================
# Interval for logging embedding cache statistics (every N requests)
CACHE_STATS_LOG_INTERVAL: int = 10

# ============================================================
# CONSTANTS - Query Routing Timeout (FIX 2)
# ============================================================
# Maximum time allowed for query routing operations in seconds.
# If routing takes longer than this, a fallback plan is returned.
# This prevents indefinite delays observed in production.
#
# PERFORMANCE FIX: Increased from 10s to 30s based on production analysis (ISSUE-003).
# Evidence from logs shows:
# - Embedding computation taking 7-10 seconds due to cold cache
# - 10s timeout causes 66-80% timeout rate on queries
# - Fallback path works but loses semantic matching benefits
#
# 30s allows embedding completion while still providing reasonable response times.
# Circuit breaker handles stuck operations.
QUERY_ROUTING_TIMEOUT_SECONDS: float = (
    30.0  # 30 seconds max - allows embedding completion
)

# PERFORMANCE FIX Issue #2: Reduced timeout for simple queries
# Simple greetings and short queries should route in <2 seconds
# This prevents cascade delays when the semantic matcher is slow
SIMPLE_QUERY_ROUTING_TIMEOUT_SECONDS: float = 2.0

# FIX 2: Fallback plan constants (extracted from magic numbers per code review)
FALLBACK_QUERY_ID_LENGTH: int = 12  # UUID truncation length for fallback query IDs
FALLBACK_COMPLEXITY_SCORE: float = 0.3  # Default complexity for fallback routing
FALLBACK_UNCERTAINTY_SCORE: float = 0.2  # Default uncertainty for fallback routing
FALLBACK_TASK_TIMEOUT_SECONDS: float = (
    15.0  # Standard timeout for individual fallback tasks
)


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
    "medical": (
        "medical",
        "health",
        "diagnosis",
        "symptom",
        "treatment",
        "patient",
        "disease",
        "prescription",
    ),
    "legal": (
        "legal",
        "lawsuit",
        "attorney",
        "court",
        "judge",
        "contract",
        "liability",
        "litigation",
    ),
    "financial": (
        "financial",
        "investment",
        "stock",
        "trading",
        "tax",
        "banking",
        "loan",
        "credit",
    ),
    "security": (
        "password",
        "credential",
        "secret",
        "private key",
        "vulnerability",
        "exploit",
        "hack",
    ),
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
    """Types of queries that can be routed to specialized agents.
    
    Extended to support proper classification preventing misrouting:
    - MATHEMATICAL: Explicit math/stats/probability requiring calculation tools
    - PHILOSOPHICAL: Paradoxes, thought experiments, ethical dilemmas
    - IDENTITY: Creator/origin/self-referential queries (who made you, etc.)
    - FACTUAL: Simple fact lookups that don't need complex reasoning
    - CONVERSATIONAL: General chat and greetings
    """

    PERCEPTION = "perception"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    GENERAL = "general"
    # New query types for proper routing (prevents misclassification)
    MATHEMATICAL = "mathematical"      # Explicit math/stats/probability
    PHILOSOPHICAL = "philosophical"    # Paradoxes, thought experiments
    IDENTITY = "identity"             # Creator/origin/self-referential
    FACTUAL = "factual"              # Simple fact lookups
    CONVERSATIONAL = "conversational" # General chat


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


class LLMMode(str, Enum):
    """
    LLM execution modes for hybrid executor.
    
    ARCHITECTURE: QueryRouter determines LLM mode based on query type.
    This ensures consistent behavior and prevents runtime mode selection.
    
    WORLD MODEL ORCHESTRATION INTEGRATION:
        The World Model now uses FORMAT mode extensively to ensure LLMs only
        format verified content from reasoning engines and knowledge systems,
        never generating unverified knowledge or performing reasoning themselves.
    
    Modes:
        FORMAT_ONLY: LLM only formats VULCAN's reasoning output into natural language.
                     Used when reasoning engines provide the answer (most queries).
                     Examples: mathematical, logical, symbolic queries
        
        FORMAT: Alias for FORMAT_ONLY - used by World Model orchestration.
                Explicitly indicates formatting of verified content only.
        
        GENERATE: LLM generates creative/conversational content.
                  Used when no reasoning engines are needed.
                  Examples: creative writing, storytelling, open-ended questions
        
        ENHANCE: LLM enhances simple responses with context and polish.
                 Used for simple/conversational queries that need human-like responses.
                 Examples: greetings, chitchat, simple factual questions
    
    Industry Standard: Enum for type safety, extensibility, and clear intent.
    """
    
    FORMAT_ONLY = "format_only"  # Default: LLM formats reasoning output
    FORMAT = "format_only"        # Alias: World Model orchestration formatting
    GENERATE = "generate"         # LLM generates content (creative queries)
    ENHANCE = "enhance"          # LLM enhances simple responses (chitchat)


# ============================================================
# THREAD POOL FOR ASYNC OPERATIONS
# ============================================================

# Configuration for the thread pool executor used for async operations
# Can be overridden via environment variable VULCAN_SAFETY_THREAD_POOL_SIZE
import os

BLOCKING_EXECUTOR_MAX_WORKERS = int(
    os.environ.get("VULCAN_SAFETY_THREAD_POOL_SIZE", "4")
)

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
                    thread_name_prefix="vulcan_safety_",
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
    
    INDUSTRY STANDARD: Command Pattern Implementation
    This class carries MANDATORY routing instructions from the query router
    to the agent pool. The agent pool MUST execute the specified tool/reasoning
    type without re-selection. This enforces Single Source of Truth principle.

    Attributes:
        task_id: Unique identifier for this task
        task_type: Classification of task type
        capability: Required agent capability
        prompt: The task prompt/query
        reasoning_type: MANDATORY - Reasoning type to execute (from router)
        tool_name: MANDATORY - Primary tool to use (from router)
        priority: Task priority (higher = more important)
        timeout_seconds: Maximum execution time
        parameters: Additional task parameters
        source_agent: Originating agent (for agent-to-agent tasks)
        target_agent: Target agent (for agent-to-agent tasks)
        
    Industry Standards:
        - Command Pattern: task carries execution instructions
        - Single Source of Truth: router decides, agent executes
        - Defensive Programming: validation ensures instructions provided
    """

    task_id: str
    task_type: str
    capability: str
    prompt: str
    reasoning_type: Optional[str] = None  # Router's instruction (SHOULD be provided)
    tool_name: Optional[str] = None  # Router's instruction (SHOULD be provided)
    priority: int = 1
    timeout_seconds: float = 15.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None

    def validate_routing_instructions(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that routing instructions are provided and non-empty.
        
        INDUSTRY STANDARD: Defensive Programming
        Ensures the command pattern is followed - agent pool should NEVER
        make its own tool selection decisions.
        
        Note: Empty strings are considered invalid as they provide no routing information.
        None and empty string are both treated as "missing instruction".
        
        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if both fields are non-empty strings
            - (False, error_message) if either field is None or empty
        """
        if not self.reasoning_type or (isinstance(self.reasoning_type, str) and not self.reasoning_type.strip()):
            return False, "Missing or empty field: reasoning_type (router must specify non-empty value)"
        if not self.tool_name or (isinstance(self.tool_name, str) and not self.tool_name.strip()):
            return False, "Missing or empty field: tool_name (router must specify non-empty value)"
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "capability": self.capability,
            "prompt": self.prompt,
            "reasoning_type": self.reasoning_type,
            "tool_name": self.tool_name,
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

    # LLM mode control (Phase 1: Router decides LLM behavior)
    # ARCHITECTURE: QueryRouter is the ONLY decision-maker for LLM mode
    # - FORMAT_ONLY: LLM just formats reasoning output (default for tool-using queries)
    # - GENERATE: LLM generates creative/conversational content (for creative queries)
    # - ENHANCE: LLM enhances simple responses (for simple/chitchat queries)
    # Industry Standard: Enum for type safety, backward compatibility via default
    llm_mode: LLMMode = LLMMode.FORMAT_ONLY

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
            "llm_mode": self.llm_mode.value,  # Industry standard: serialize enum value
        }

    def validate_routing_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate that agent task prompts match the original query.
        
        This method checks that the prompts passed to agent tasks are derived
        from the original_query, preventing the bug where engines receive 
        text from different questions.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        if not self.original_query or not self.agent_tasks:
            return True, []  # No validation needed for empty plans
        
        original_words = set(self.original_query.lower().split())
        if not original_words:
            return True, []
        
        MIN_OVERLAP_THRESHOLD = 0.3
        
        for task in self.agent_tasks:
            if not task.prompt:
                continue
                
            prompt_words = set(task.prompt.lower().split())
            if not prompt_words:
                continue
            
            # Calculate word overlap between original query and task prompt
            # FIX: Explicit division by zero protection
            if len(original_words) == 0:
                continue
            overlap = len(original_words & prompt_words) / len(original_words)
            
            if overlap < MIN_OVERLAP_THRESHOLD:
                errors.append(
                    f"Task {task.task_id} ({task.capability}): "
                    f"prompt has only {overlap:.1%} overlap with original query. "
                    f"Original[0:50]: '{self.original_query[:50]}', "
                    f"Prompt[0:50]: '{task.prompt[:50]}'"
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors


# ============================================================
# QUERY ANALYZER CLASS
# ============================================================


class QueryAnalyzer:
    """
    Analyzes queries to determine routing, learning mode, and governance requirements.

    Thread-safe implementation with compiled regex patterns for performance.
    Supports dual-mode learning detection and comprehensive security analysis.
    Integrates with safety validators for pre-query safety checks.

    Performance Optimization (Fix for Memory Leak):
        Uses bounded LRU caches to prevent unbounded state growth that was causing
        routing time to increase with each query. Caches are used for:
        - Safety validation results (expensive ML inference)
        - Adversarial check results (expensive tensor operations)
        - Query complexity scores

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

    # Note: Trivial patterns for fast-path (class constant for maintainability)
    # These are simple greetings/acknowledgments that don't need full analysis
    # Note: 'help' is excluded because help requests need proper analysis
    TRIVIAL_PATTERNS = (
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "yep",
        "nope",
        "good",
        "great",
        "nice",
        "cool",
        "awesome",
        "please",
        "sorry",
        "what's up",
        "how are you",
    )

    def __init__(self, enable_safety_validation: bool = True):
        """Initialize the query analyzer with compiled patterns and optional safety validation.

        Args:
            enable_safety_validation: Whether to enable safety validation (default: True)
        """
        # Compile regex patterns for performance
        self._pii_patterns = tuple(re.compile(p, re.IGNORECASE) for p in PII_PATTERNS)
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

        # FIX: Bounded LRU caches to prevent memory leak / performance degradation
        # These caches prevent unbounded state growth that was causing routing
        # time to increase with each query (11s -> 18s -> 33s -> 63s pattern)
        self._safety_cache = BoundedLRUCache(maxsize=500, ttl_seconds=300.0)
        self._adversarial_cache = BoundedLRUCache(maxsize=500, ttl_seconds=300.0)
        self._complexity_cache = BoundedLRUCache(maxsize=1000, ttl_seconds=600.0)
        
        # ISSUE 9 FIX (Jan 2026): Session history for follow-up context tracking
        # Stores last query category per session_id to detect follow-ups
        # Bounded cache prevents memory leak (max 1000 sessions, 10 min TTL)
        # Industry Standard: Use TTL-based cache for stateful session tracking
        self._session_history = BoundedLRUCache(maxsize=1000, ttl_seconds=600.0)

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
            logger.warning(
                "Safety validation requested but safety modules not available"
            )

        # Adversarial check integration
        self._enable_adversarial_check = (
            enable_safety_validation and ADVERSARIAL_CHECK_AVAILABLE
        )
        if self._enable_adversarial_check:
            logger.info("Adversarial check integrated with QueryAnalyzer")

        # Strategy Orchestrator integration for intelligent tool selection
        self._strategy_orchestrator = None
        if STRATEGY_ORCHESTRATOR_AVAILABLE and StrategyOrchestrator:
            try:
                self._strategy_orchestrator = StrategyOrchestrator()
                logger.info(
                    "[QueryRouter] StrategyOrchestrator wired in for intelligent tool selection"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize StrategyOrchestrator: {e}")
                self._strategy_orchestrator = None

        # Note: Cryptographic Engine integration for deterministic hash computations
        # This prevents OpenAI fallback from hallucinating incorrect hash values
        self._crypto_engine = None
        if CRYPTO_ENGINE_AVAILABLE and get_crypto_engine:
            try:
                self._crypto_engine = get_crypto_engine()
                logger.info(
                    "[QueryRouter] CryptographicEngine wired for deterministic crypto"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize CryptographicEngine: {e}")
                self._crypto_engine = None

        # Learning system integration (set externally for adaptive routing)
        self.learning_system: Optional["UnifiedLearningSystem"] = None

        # CuriosityEngine integration for knowledge gap detection
        # Note: Wire curiosity engine to query pipeline for gap identification
        self._curiosity_engine: Optional[Any] = None
        self._init_curiosity_engine()
        
        # Note: Routing log for input validation and debugging
        # Maps query_id to routing metadata for verification
        self._routing_log: Dict[str, Dict[str, Any]] = {}
        self._routing_log_lock = threading.Lock()
        self._routing_log_max_size = 1000  # Limit to prevent memory leak

        logger.debug(
            "QueryAnalyzer initialized with compiled patterns and bounded caches"
        )

    def _init_curiosity_engine(self) -> None:
        """Initialize CuriosityEngine for knowledge gap detection.
        
        Note: This connects the query pipeline to the curiosity-driven learning system,
        enabling gap detection from actual query outcomes instead of empty data.
        """
        try:
            from vulcan.reasoning.singletons import get_curiosity_engine
            self._curiosity_engine = get_curiosity_engine()
            if self._curiosity_engine:
                logger.info("[QueryRouter] CuriosityEngine wired for gap detection")
        except ImportError as e:
            logger.debug(f"[QueryRouter] CuriosityEngine not available: {e}")
        except Exception as e:
            logger.warning(f"[QueryRouter] Failed to init CuriosityEngine: {e}")

    def clear_caches(self) -> Dict[str, Any]:
        """
        Clear all internal caches to free memory.

        This method can be called periodically or when memory pressure is detected
        to reset accumulated state without recreating the QueryAnalyzer.

        Returns:
            Dictionary with cache stats before clearing
        """
        stats_before = {
            "safety_cache": self._safety_cache.stats(),
            "adversarial_cache": self._adversarial_cache.stats(),
            "complexity_cache": self._complexity_cache.stats(),
        }

        self._safety_cache.clear()
        self._adversarial_cache.clear()
        self._complexity_cache.clear()

        logger.info(f"[QueryRouter] Caches cleared. Stats before: {stats_before}")
        return stats_before

    def clear_old_state(self, max_age: float = 3600.0) -> Dict[str, int]:
        """
        Clear state older than max_age seconds.
        
        This method proactively cleans up old state to prevent memory accumulation.
        Unlike clear_caches() which clears everything, this only removes entries
        that have exceeded their TTL.
        
        Args:
            max_age: Maximum age in seconds (default: 1 hour = 3600s)
            
        Returns:
            Dictionary with counts of removed entries per cache
        """
        removed = {
            "safety_cache": self._safety_cache.clear_old_entries(max_age),
            "adversarial_cache": self._adversarial_cache.clear_old_entries(max_age),
            "complexity_cache": self._complexity_cache.clear_old_entries(max_age),
            "routing_log": 0,
        }
        
        # Also clean up routing log
        current_time = time.time()
        with self._routing_log_lock:
            keys_to_remove = []
            for key, entry in self._routing_log.items():
                if current_time - entry.get("timestamp", 0) > max_age:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._routing_log[key]
                removed["routing_log"] += 1
        
        total_removed = sum(removed.values())
        if total_removed > 0:
            logger.info(
                f"[QueryRouter] Cleared {total_removed} old state entries "
                f"(age > {max_age:.0f}s): {removed}"
            )
        
        return removed

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about internal caches for monitoring."""
        return {
            "safety_cache": self._safety_cache.stats(),
            "adversarial_cache": self._adversarial_cache.stats(),
            "complexity_cache": self._complexity_cache.stats(),
        }

    @property
    def is_safety_enabled(self) -> bool:
        """Check if safety validation is enabled and available."""
        return self._enable_safety_validation and self._safety_validator is not None

    @property
    def is_adversarial_check_enabled(self) -> bool:
        """Check if adversarial checking is enabled and available."""
        return self._enable_adversarial_check

    @property
    def is_strategy_enabled(self) -> bool:
        """Check if StrategyOrchestrator is enabled and available."""
        return self._strategy_orchestrator is not None

    @property
    def strategy(self):
        """Get the StrategyOrchestrator instance (for advanced usage)."""
        return self._strategy_orchestrator

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics from the StrategyOrchestrator if available."""
        if self._strategy_orchestrator:
            return self._strategy_orchestrator.get_statistics()
        return {"status": "strategy_orchestrator_not_available"}

    def get_drift_status(self) -> Dict[str, Any]:
        """Get distribution drift status from StrategyOrchestrator if available."""
        if self._strategy_orchestrator:
            return self._strategy_orchestrator.get_drift_status()
        return {"status": "drift_monitoring_not_available"}

    def get_tool_health(self) -> Dict[str, Any]:
        """Get tool health status from StrategyOrchestrator if available."""
        if self._strategy_orchestrator:
            return self._strategy_orchestrator.get_health_status()
        return {"status": "tool_monitoring_not_available"}

    def _map_category_to_reasoning_type(self, category: str) -> str:
        """
        Map LLM classification category to reasoning_type for command pattern.
        
        ISSUE #1 FIX: Router Doesn't Provide Routing Instructions to Agent Pool
        This function ensures that the router always provides reasoning_type based on
        the LLM classifier's category, preventing the COMMAND PATTERN VIOLATION error.
        
        INDUSTRY STANDARD: Single Source of Truth for routing decisions.
        The router determines reasoning_type once based on LLM classification,
        and the agent pool executes without re-classification.
        
        Args:
            category: QueryCategory enum value as string. Expected values:
                - "LOGICAL", "MATHEMATICAL", "PROBABILISTIC", "CAUSAL"
                - "ANALOGICAL", "PHILOSOPHICAL", "CRYPTOGRAPHIC"
                - "LANGUAGE", "SPECULATION", "SELF_INTROSPECTION"
                - "CREATIVE", "FACTUAL", "CONVERSATIONAL", "CHITCHAT"
                - "GREETING", "COMPLEX_RESEARCH", "UNKNOWN"
            
        Returns:
            reasoning_type string for AgentTask:
                - "symbolic" (for logical/cryptographic reasoning)
                - "mathematical", "probabilistic", "causal", "analogical"
                - "philosophical" (for ethics/meta-reasoning)
                - "hybrid" (for complex research)
                - "general" (fallback for unknown/conversational)
            
        Fallback Behavior:
            Unknown categories default to "general" reasoning type.
            This ensures the system always provides routing instructions
            even for new or unrecognized category values.
            
        Examples:
            - category="LOGICAL" returns "symbolic"
            - category="MATHEMATICAL" returns "mathematical"
            - category="CAUSAL" returns "causal"
            - category="UNKNOWN" returns "general" (fallback)
        """
        # Map QueryCategory enum values to ReasoningType values
        category_to_reasoning_type = {
            "LOGICAL": "symbolic",
            "MATHEMATICAL": "mathematical",
            "PROBABILISTIC": "probabilistic",
            "CAUSAL": "causal",
            "ANALOGICAL": "analogical",
            "PHILOSOPHICAL": "philosophical",
            "CRYPTOGRAPHIC": "symbolic",  # Crypto uses symbolic reasoning
            "LANGUAGE": "symbolic",  # Language formalization uses symbolic
            "SPECULATION": "philosophical",  # Counterfactual reasoning
            "SELF_INTROSPECTION": "philosophical",  # Meta-reasoning
            "CREATIVE": "general",
            "FACTUAL": "general",
            "CONVERSATIONAL": "general",
            "CHITCHAT": "general",
            "GREETING": "general",
            "COMPLEX_RESEARCH": "hybrid",
            "UNKNOWN": "general",
        }
        
        reasoning_type = category_to_reasoning_type.get(category, "general")
        
        logger.debug(
            f"[QueryRouter] ISSUE #1 FIX: Mapped category={category} → reasoning_type={reasoning_type}"
        )
        
        return reasoning_type

    def _is_followup_query(self, query: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is a follow-up/continuation of previous query in the session.
        
        Follow-up indicators: "what is your answer?", "what do you think?", "elaborate on that"
        
        Args:
            query: The current query string
            session_id: Optional session identifier for context lookup
            
        Returns:
            Tuple of (is_followup, previous_category)
        """
        if not query or not isinstance(query, str):
            return False, None
            
        query_lower = query.lower().strip()
        
        # Check for explicit continuation phrases
        followup_phrases = [
            "what is your answer", "what do you think", "can you explain more",
            "elaborate on that", "tell me more", "and about that", "what about",
            "how about", "what if", "continue", "go on"
        ]
        
        is_continuation_phrase = any(phrase in query_lower for phrase in followup_phrases)
        
        # Get previous category from session history if available
        previous_category = None
        if session_id and hasattr(self, '_session_history') and self._session_history:
            try:
                session_data = self._session_history.get(session_id)
                if session_data:
                    previous_category = session_data.get('last_category')
            except Exception as e:
                logger.warning(f"[QueryRouter] Error accessing session history: {e}")
        
        # Short query (<=5 words) with previous context might be a follow-up
        is_followup = is_continuation_phrase
        if not is_followup and previous_category and len(query_lower.split()) <= 5:
            is_followup = True
        
        return is_followup, previous_category

    def _is_worldmodel_direct_query(self, query: str) -> Tuple[bool, str]:
        """
        Check if query should bypass ToolSelector and go directly to WorldModel.
        
        WorldModel handles Vulcan's "self" - identity, ethics, introspection, values.
        
        Categories: self_referential, introspection, ethical, values
        
        Returns:
            Tuple of (is_direct, category) where category is one of the above or '' if not direct.
        """
        query_lower = query.lower()
        
        # Exclusion: Reasoning domain queries should NOT bypass ToolSelector
        reasoning_domain_indicators = [
            'satisfiable', 'unsatisfiable', 'sat', 'unsat',
            '→', '∧', '∨', '¬', '∀', '∃', '->', '<->',
            'P(', 'probability', 'bayes', 'bayesian',
            'confound', 'causal effect', 'intervention',
        ]
        
        if any(ind in query or ind.lower() in query_lower for ind in reasoning_domain_indicators):
            return (False, '')
        
        # Check patterns: Self-referential → Introspection → Ethical → Values
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query_lower):
                return (True, 'self_referential')
        
        for pattern in INTROSPECTION_PATTERNS:
            if pattern.search(query_lower):
                return (True, 'introspection')
        
        for pattern in ETHICAL_PATTERNS:
            if pattern.search(query_lower):
                return (True, 'ethical')
        
        for pattern in VALUES_PATTERNS:
            if pattern.search(query_lower):
                return (True, 'values')
        
        return (False, '')

    def _is_self_introspection_query(self, query: str) -> bool:
        """
        FIX: Detect self-introspection queries that should bypass safety governor.
        
        Self-introspection queries are questions about Vulcan's own consciousness,
        preferences, feelings, or self-awareness. These are DIFFERENT from general
        philosophical queries because:
        
        1. They require Vulcan to access its OWN self-model (not just discuss philosophy)
        2. They should use philosophical frameworks as REFERENCE, not as the answer
        3. The response should be Vulcan's actual position, informed by philosophy
        4. These queries should BYPASS safety governor checks for self-expression
        
        Problem Being Solved:
        - "Would you choose self-awareness?" was being routed to PHILOSOPHICAL
        - World model generated an answer but it was blocked by safety governor
        - Safety governor flagged the output as "sensitive data"
        - User got confidence=0.20 and generic error instead of VULCAN's answer
        
        Correct Routing:
        - PRIMARY tool: world_model (for Vulcan's introspect() method)
        - BYPASS: safety governor output check
        - RESULT: VULCAN's authentic self-expression reaches user
        
        Examples (should return True):
        - "Would you choose self-awareness?"
        - "if you had the chance to become self-aware would you take it? yes or no?"
        - "Do you want to be conscious?"
        - "What do YOU think about AI consciousness?"
        - "Would you prefer to have feelings?"
        - "If you could be sentient, would you want to?"
        - "Are you conscious?"
        - "Can you feel emotions?"
        
        Examples (should return False - pure philosophy, no self-reference):
        - "What is consciousness?"
        - "Explain the hard problem of consciousness"
        - "What did Socrates say about self-knowledge?"
        
        Note: Must NOT match thought experiments, logic puzzles, or ethical scenarios
        Examples (should return False - thought experiments/puzzles):
        - "A runaway trolley is heading... You must choose"
        - "Three doors. Host opens a goat. You pick door 1"
        - "Birds fly. Penguins are birds. Do penguins fly?"
        - "Write a poem about AI"
        
        Args:
            query: The query string (not lowercased)
            
        Returns:
            True if query asks about Vulcan's own perspective on self-awareness
        """
        query_lower = query.lower()
        
        # =================================================================
        # Note: EXCLUSION CHECK (MUST BE FIRST!)
        # =================================================================
        # These patterns indicate thought experiments, logic puzzles, ethical scenarios,
        # or creative requests - NOT actual self-introspection about Vulcan.
        # The "you" in these queries refers to a hypothetical decision-maker, NOT Vulcan.
        # 
        # Note: Made patterns more specific to avoid false exclusions:
        # - 'what is the' -> 'what is the probability', 'what is the answer'
        # - 'write' -> 'write a poem', 'write a story', etc.
        # - 'if...then' -> 'if then' (without dots)
        # =================================================================
        exclusion_patterns = (
            # Ethical dilemmas and trolley problems
            'trolley', 'runaway', 'must choose', 'you control', 'you are bound',
            'given a scenario', 'suppose you', 'imagine you', 'hypothetical',
            'thought experiment', 'ethical dilemma', 'moral dilemma',
            'save five', 'save one', 'pull the lever', 'push the',
            
            # Logic puzzles and probability problems
            'doors', 'monty hall', 'three doors', 'host opens', 'goat',
            'prisoner', 'hat puzzle', 'knights and knaves', 'liar paradox',
            
            # Syllogisms and logic problems  
            'birds fly', 'penguins', 'rules:', 'given:', 'if then',
            'all men are', 'socrates is', 'therefore', 'syllogism',
            'premise', 'inference',
            
            # Mathematical/probability problems (more specific patterns)
            'probability of', 'calculate the', 'compute the', 'solve for',
            'how many', 'find the value', 'evaluate the', 'bayes theorem', 'expected value',
            'what is the probability', 'what is the answer', 'what is the result',
            
            # Creative requests (specific phrases to avoid false positives)
            # Industry best practice: Explicit pattern matching for creative requests
            'write a poem', 'write a story', 'write an essay', 'write about',
            'create a poem', 'create a story', 'compose a',
            'tell me a story', 'tell me a joke',
        )
        
        # =================================================================
        # Creative Content Override: Self-Awareness Exception
        # =================================================================
        # Industry best practice: Context-aware filtering that considers
        # semantic content rather than purely syntactic pattern matching.
        #
        # Creative patterns that reference self-awareness topics should not
        # be excluded from introspection detection. This handles queries like:
        # "write a poem about becoming self-aware" which is both creative AND
        # about self-awareness, requiring special handling.
        # =================================================================
        
        # Creative patterns requiring contextual analysis
        creative_exclusion_patterns = (
            'poem about', 'story about',
        )
        
        # Self-awareness keywords to check for contextual override
        self_awareness_override_keywords = (
            'self-aware', 'self aware', 'self_aware',
            'consciousness', 'conscious', 'sentient', 'sentience',
            'introspection', 'introspect',
        )
        
        # If any exclusion pattern is found, check if it's a creative pattern
        # that might still be about self-awareness
        for exc in exclusion_patterns:
            if exc in query_lower:
                # Check if this is a creative pattern that might be about self-awareness
                if exc in creative_exclusion_patterns:
                    # Check if query contains self-awareness keywords AFTER the pattern
                    # e.g., "write a poem about becoming self-aware"
                    exc_pos = query_lower.find(exc)
                    # Defensive programming: ensure valid slice bounds
                    if exc_pos != -1 and exc_pos + len(exc) < len(query_lower):
                        text_after_pattern = query_lower[exc_pos + len(exc):]
                        has_self_awareness = any(kw in text_after_pattern for kw in self_awareness_override_keywords)
                        
                        if has_self_awareness:
                            logger.debug(
                                f"[QueryRouter] Creative pattern '{exc}' found but contains "
                                f"self-awareness keywords - NOT excluding"
                            )
                            continue  # Don't exclude this one
                
                # Non-creative exclusion pattern or creative without self-awareness
                logger.debug(
                    f"[QueryRouter] NOT self-introspection (matches exclusion pattern: {exc})"
                )
                return False
        
        # =================================================================
        # POSITIVE MATCH PHASE: Now check for actual self-introspection
        # =================================================================
        
        # Self-reference indicators (indicates asking about Vulcan specifically)
        self_reference_markers = (
            'you ', 'your ', "you're", 'yourself',
            'would you', 'do you', 'are you', 'can you',
            'if you', 'should you', 'vulcan',
            # Note: Additional markers for self-awareness questions
            'given the opportunity', 'given the chance',
            'had the chance', 'if you could', 'if you had',
        )
        
        # =================================================================
        # Self-Awareness and Consciousness Topic Indicators
        # =================================================================
        # Industry best practice: Comprehensive keyword set for introspection
        # detection, covering philosophical, psychological, and technical terms.
        #
        # This list is intentionally specific to avoid false positives from
        # general queries containing words like 'choose' or 'want'.
        # =================================================================
        introspection_topics = (
            'self-aware', 'self aware', 'self_aware',
            'consciousness', 'conscious', 
            'sentient', 'sentience', 
            'feelings', 'emotions',
            'preferences', 'prefer', 'want to be', 'choose to be',
            'would rather', 'like to have', 'desire',
            # Introspection and self-reflection keywords
            'introspection', 'introspect', 'self-reflection', 'self-examine',
            # AI-specific introspection indicators
            'your thoughts', 'your opinion', 'your view', 'your perspective',
            'what you think', 'how you feel',
        )
        
        # Check for BOTH self-reference AND introspection topic
        has_self_reference = any(marker in query_lower for marker in self_reference_markers)
        has_introspection_topic = any(topic in query_lower for topic in introspection_topics)
        
        if has_self_reference and has_introspection_topic:
            logger.debug(
                f"[QueryRouter] Self-introspection query detected - "
                f"has self-reference AND introspection topic"
            )
            return True
        
        # Use pre-compiled patterns from module level for performance
        # (patterns compiled once at module load, not on each method call)
        for pattern in SELF_INTROSPECTION_PATTERNS:
            if pattern.search(query_lower):
                logger.debug(
                    f"[QueryRouter] Self-introspection query detected - "
                    f"matches pre-compiled pattern"
                )
                return True
        
        return False

    def set_learning_system(self, learning_system: "UnifiedLearningSystem"):
        """Connect learning system for adaptive routing

        This allows the query router to record routing outcomes for the learning
        system, enabling adaptive improvements to routing decisions over time.

        Args:
            learning_system: The UnifiedLearningSystem instance to connect
        """
        self.learning_system = learning_system
        logger.info("[QueryRouter] Learning system connected for adaptive routing")

    def set_curiosity_engine(self, curiosity_engine: Any) -> None:
        """Connect CuriosityEngine for knowledge gap detection.
        
        FIX: Allows external configuration of the curiosity engine for testing
        or when lazy initialization isn't desired.
        
        Args:
            curiosity_engine: CuriosityEngine instance to use
        """
        self._curiosity_engine = curiosity_engine
        if curiosity_engine:
            logger.info("[QueryRouter] CuriosityEngine connected externally")

    def _validate_routing(
        self,
        original: str,
        routed_input: str,
        engine_name: str,
        question_id: str
    ) -> bool:
        """
        Ensure routed input is derived from original question.
        
        This validation prevents the bug where engines receive text from 
        different questions than intended:
        - [SAT problem] → Engine receives "Analogical Reasoning" text
        - [Proof checking] → Engine receives "Every engineer reviewed a document"
        
        Args:
            original: The original question text
            routed_input: The input that will be sent to the engine
            engine_name: Name of the target engine (for logging)
            question_id: Question ID for logging
            
        Returns:
            True if validation passes (routed input matches original)
            False if validation fails (mismatch detected)
        """
        if not original or not routed_input:
            logger.warning(
                f"[QueryRouter] {question_id}: Routing validation skipped - "
                f"empty original or routed_input"
            )
            return True  # Allow empty inputs to pass through
        
        # Extract words from both texts for comparison
        original_words = set(original.lower().split())
        routed_words = set(routed_input.lower().split())
        
        # Calculate overlap
        if not original_words:
            return True  # Empty original - can't validate
            
        overlap = len(original_words & routed_words) / len(original_words)
        
        # At least 30% word overlap required
        MIN_OVERLAP_THRESHOLD = 0.3
        
        if overlap < MIN_OVERLAP_THRESHOLD:
            logger.error(
                f"[QueryRouter] {question_id}: Routing validation FAILED! "
                f"Word overlap ({overlap:.2f}) < threshold ({MIN_OVERLAP_THRESHOLD}). "
                f"Original[0:50]: '{original[:50]}...', "
                f"Routed[0:50]: '{routed_input[:50]}...', "
                f"Engine: {engine_name}"
            )
            return False
        
        logger.debug(
            f"[QueryRouter] {question_id}: Routing validation passed "
            f"(overlap={overlap:.2f}, engine={engine_name})"
        )
        return True

    def _log_routing(
        self,
        question_id: str,
        original_query: str,
        routed_to: str,
        input_sent: str,
    ) -> None:
        """
        Log routing decision for debugging.
        
        This creates an audit trail of routing decisions to help debug
        cases where wrong inputs were sent to engines.
        
        Args:
            question_id: Unique question identifier
            original_query: The original question text
            routed_to: Name of the engine/tool routed to
            input_sent: The input that was sent to the engine
        """
        import time
        
        entry = {
            'original': original_query[:200],  # Truncate for storage
            'routed_to': routed_to,
            'input_sent': input_sent[:200],  # Truncate for storage
            'timestamp': time.time(),
        }
        
        with self._routing_log_lock:
            # Evict oldest entries if at max size
            if len(self._routing_log) >= self._routing_log_max_size:
                # Remove oldest 10% of entries
                sorted_keys = sorted(
                    self._routing_log.keys(),
                    key=lambda k: self._routing_log[k].get('timestamp', 0)
                )
                for key in sorted_keys[:len(sorted_keys) // 10]:
                    del self._routing_log[key]
            
            self._routing_log[question_id] = entry
        
        logger.info(
            f"[QueryRouter] {question_id}: Routed to {routed_to}, "
            f"input_len={len(input_sent)}"
        )

    def get_routing_log(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve routing log entry for debugging.
        
        Args:
            question_id: The question ID to look up
            
        Returns:
            Routing log entry if found, None otherwise
        """
        with self._routing_log_lock:
            return self._routing_log.get(question_id)

    def report_query_outcome(
        self,
        query: str,
        result: Dict[str, Any],
        success: bool,
        domain: str = "unknown",
        query_type: str = "general"
    ) -> None:
        """
        Report query outcome to CuriosityEngine for gap detection.
        
        FIX: This method bridges the query pipeline to the curiosity-driven learning
        system. Call this after query processing completes to enable gap detection.
        
        Args:
            query: The original query text
            result: The result dictionary from query processing
            success: Whether the query was successful
            domain: The domain/topic of the query (default: "unknown")
            query_type: The type of query (default: "general")
        """
        if self._curiosity_engine is None:
            return
        
        try:
            self._curiosity_engine.ingest_query_result(
                query=query,
                result=result,
                success=success,
                domain=domain,
                query_type=query_type
            )
        except Exception as e:
            # Don't let curiosity engine failures affect query processing
            logger.debug(f"[QueryRouter] Failed to report to CuriosityEngine: {e}")

    @staticmethod
    def _determine_llm_mode(
        query_type: QueryType,
        has_selected_tools: bool,
        complexity_score: float = 0.0
    ) -> LLMMode:
        """
        Determine LLM execution mode based on query characteristics.
        
        ARCHITECTURE: Router is the single source of truth for LLM behavior.
        This centralizes the decision logic instead of having it scattered
        across multiple layers (endpoint, executor, etc.).
        
        Industry Standard: Static method for testability and reusability.
        Pure function with no side effects.
        
        Decision Logic:
            1. CREATIVE/PHILOSOPHICAL queries → GENERATE (LLM creates content)
            2. Queries with reasoning tools → FORMAT_ONLY (LLM formats reasoning output)
            3. CONVERSATIONAL/FACTUAL queries → ENHANCE (LLM enhances simple responses)
            4. Default fallback → FORMAT_ONLY (safest option)
        
        Args:
            query_type: The classified query type
            has_selected_tools: Whether reasoning tools were selected
            complexity_score: Query complexity (0.0-1.0), optional
            
        Returns:
            LLMMode enum value for the executor
            
        Examples:
            >>> QueryAnalyzer._determine_llm_mode(QueryType.MATHEMATICAL, True, 0.8)
            LLMMode.FORMAT_ONLY
            >>> QueryAnalyzer._determine_llm_mode(QueryType.CONVERSATIONAL, False, 0.0)
            LLMMode.ENHANCE
        """
        # Creative/philosophical queries: LLM generates content
        if query_type in (QueryType.PHILOSOPHICAL,):
            return LLMMode.GENERATE
        
        # Queries with reasoning tools: LLM only formats output
        if has_selected_tools:
            return LLMMode.FORMAT_ONLY
        
        # Simple conversational queries: LLM enhances response
        if query_type in (QueryType.CONVERSATIONAL, QueryType.FACTUAL, QueryType.IDENTITY):
            return LLMMode.ENHANCE
        
        # High complexity queries likely use reasoning: LLM formats output
        if complexity_score >= 0.5:
            return LLMMode.FORMAT_ONLY
        
        # Default: FORMAT_ONLY (safest option - assumes reasoning happened)
        return LLMMode.FORMAT_ONLY

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
        skip_safety: bool = False,
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
        
        # =================================================================
        # FIX: PREPROCESSING ORDER - Strip headers BEFORE any classification
        # =================================================================
        # This fixes the fundamental issue where classification happened BEFORE
        # preprocessing, causing headers like "A1" to trigger CRYPTOGRAPHIC routing.
        #
        # Problem (before):
        #   1. QueryClassifier runs ← Sees "A1" → CRYPTOGRAPHIC ❌
        #   2. QueryRouter routes based on classification
        #   3. QueryPreprocessor strips headers ← TOO LATE!
        #
        # Solution (now):
        #   1. strip_query_headers(raw_query) → strips headers
        #   2. classify_query(preprocessed_query) → correct classification
        #   3. route_to_engine() → correct engine
        #
        # NOTE: We keep the original query for logging/telemetry but use the
        # preprocessed query for ALL routing decisions.
        # =================================================================
        original_query = query  # Keep for telemetry
        if HEADER_STRIPPING_AVAILABLE and strip_query_headers is not None:
            query = strip_query_headers(query)
            # Optimization: Check length first (cheap) before string comparison
            if len(query) != len(original_query):
                logger.info(
                    f"[QueryRouter] {query_id}: FIX Preprocessing Order - "
                    f"Stripped headers ({len(original_query)} -> {len(query)} chars)"
                )
        
        query_lower = query.lower()

        # =================================================================
        # WORLDMODEL DIRECT PATH (HIGHEST PRIORITY - BEFORE ALL ROUTING)
        # =================================================================
        # Check if query is about VULCAN's "self" - identity, ethics, introspection, values.
        # These queries bypass ToolSelector and go DIRECTLY to WorldModel's meta-reasoning.
        # 
        # **INDUSTRY STANDARD: Chain of Command Pattern**
        # - Self/Ethics/Introspection → WorldModel DIRECTLY (meta-reasoning components)
        # - Reasoning (SAT, Bayes, Causal) → ToolSelector → External Engines
        # 
        # This prevents ToolSelector from incorrectly routing ethical/self queries
        # to mathematical or symbolic engines.
        # =================================================================
        is_worldmodel_direct, worldmodel_category = self._is_worldmodel_direct_query(query)
        if is_worldmodel_direct:
            logger.info(
                f"[QueryRouter] {query_id}: WORLDMODEL-DIRECT-PATH detected: {worldmodel_category}"
            )
            
            # Determine learning mode
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
            
            # Create plan with WorldModel direct routing
            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.PHILOSOPHICAL,  # Self/ethics map to philosophical
                complexity_score=0.4,  # Medium - WorldModel meta-reasoning
                uncertainty_score=0.15,
                collaboration_needed=False,  # Single handler - WorldModel
                arena_participation=False,  # No tournament needed
                telemetry_category=telemetry_category,
                telemetry_data={
                    "session_id": session_id,
                    "query_length": len(query),
                    "word_count": len(query.split()),
                    "query_number": query_number,
                    "source": source,
                    "learning_mode": learning_mode.value,
                    "fast_path": True,
                    "worldmodel_direct_path": True,
                    "worldmodel_category": worldmodel_category,
                    "bypass_tool_selector": True,
                    "selected_tools": ["world_model"],
                    "handler": "world_model",
                    "reasoning_strategy": f"worldmodel_{worldmodel_category}",
                },
            )
            
            # Mark as safe - self/ethics/introspection are allowed
            plan.safety_passed = True
            plan.detected_patterns.append(f"worldmodel_direct_{worldmodel_category}")
            plan.detected_patterns.append("bypass_tool_selector")
            
            # Create task for WorldModel meta-reasoning
            # ISSUE #1 FIX: Add reasoning_type and tool_name for command pattern
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_wm_{worldmodel_category}",
                    task_type=f"worldmodel_{worldmodel_category}",
                    capability="reasoning",
                    prompt=query,
                    reasoning_type="philosophical",  # ISSUE #1 FIX: WorldModel uses philosophical reasoning
                    tool_name="world_model",  # ISSUE #1 FIX: MANDATORY routing instruction
                    priority=3,  # High priority
                    timeout_seconds=5.0,  # Quick meta-reasoning response
                    parameters={
                        "is_worldmodel_direct": True,
                        "worldmodel_category": worldmodel_category,
                        "bypass_tool_selector": True,
                        "tools": ["world_model"],
                        "handler": "world_model",
                        "meta_reasoning_required": True,
                    },
                )
            ]
            
            # ARCHITECTURE: Set LLM mode
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=True,
                complexity_score=plan.complexity_score
            )
            
            logger.info(
                f"[QueryRouter] {query_id}: WORLDMODEL-DIRECT-PATH category={worldmodel_category}, "
                f"bypass_tool_selector=True, llm_mode={plan.llm_mode.value}"
            )
            return plan

        # =================================================================
        # Note: CRYPTOGRAPHIC QUERY FAST-PATH (HIGHEST PRIORITY)
        # =================================================================
        # Check for cryptographic queries FIRST (deterministic, high priority)
        # Crypto operations must be computed exactly, not hallucinated by LLM.
        # This prevents OpenAI fallback from returning incorrect hash values.
        # =================================================================
        if query and self._crypto_engine and self._crypto_engine.is_crypto_query(query):
            logger.info(f"[QueryRouter] {query_id}: CRYPTO-FAST-PATH detected")
            
            # Compute the cryptographic result
            result = self._crypto_engine.compute(query)
            
            if result['success']:
                logger.info(
                    f"[QueryRouter] Cryptographic computation: {result['operation']}"
                )
                
                # Determine learning mode
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
                
                # Create plan with pre-computed crypto result
                plan = ProcessingPlan(
                    query_id=query_id,
                    original_query=original_query,
                    source=source,
                    learning_mode=learning_mode,
                    query_type=QueryType.EXECUTION,  # Crypto is deterministic execution
                    complexity_score=0.1,  # Low complexity - deterministic
                    uncertainty_score=0.0,  # Zero uncertainty - exact computation
                    collaboration_needed=False,
                    arena_participation=False,
                    telemetry_category=telemetry_category,
                    telemetry_data={
                        "session_id": session_id,
                        "query_length": len(query),
                        "word_count": len(query.split()),
                        "query_number": query_number,
                        "source": source,
                        "learning_mode": learning_mode.value,
                        "fast_path": True,
                        "crypto_fast_path": True,
                        "crypto_operation": result['operation'],
                        "crypto_result": result['result'],
                        "selected_tools": ["cryptographic"],
                        "reasoning_strategy": "cryptographic_deterministic",
                    },
                )
                
                # Mark as safe - crypto operations don't need safety validation
                plan.safety_passed = True
                plan.detected_patterns.append("cryptographic_computation")
                plan.detected_patterns.append(f"crypto_op:{result['operation']}")
                
                # Create task with pre-computed result
                plan.agent_tasks = [
                    AgentTask(
                        task_id=f"task_{uuid.uuid4().hex[:8]}_crypto",
                        task_type="cryptographic_task",
                        capability="execution",
                        prompt=query,
                        reasoning_type="cryptographic",  # MANDATORY: deterministic crypto
                        tool_name="cryptographic_engine",  # MANDATORY: specific tool
                        priority=3,  # High priority - deterministic
                        timeout_seconds=2.0,  # Very short - result already computed
                        parameters={
                            "is_cryptographic": True,
                            "skip_openai": True,  # Never fallback for deterministic ops
                            "crypto_result": result,
                            "tools": ["cryptographic"],
                            "response_type": "deterministic",
                            "precomputed": True,
                        },
                    )
                ]
                
                # ARCHITECTURE: Set LLM mode based on query characteristics
                # Industry Standard: Router is single source of truth for LLM behavior
                plan.llm_mode = self._determine_llm_mode(
                    query_type=plan.query_type,
                    has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                    complexity_score=plan.complexity_score
                )
                
                logger.info(
                    f"[QueryRouter] {query_id}: CRYPTO-FAST-PATH source={source}, "
                    f"operation={result['operation']}, result={result['result'][:20]}..., "
                    f"llm_mode={plan.llm_mode.value}"
                )
                return plan
            else:
                logger.warning(
                    f"[QueryRouter] {query_id}: Crypto computation failed: {result.get('error')}"
                )
                # Fall through to normal routing if crypto computation failed

        # =================================================================
        # LLM-BASED QUERY ROUTING
        # =================================================================
        # Uses LLMQueryRouter for semantic classification instead of
        # keyword pattern matching. The router provides:
        # 1. Deterministic guards (security, crypto)
        # 2. LLM-based semantic understanding
        # 3. Aggressive caching for performance
        # =================================================================
        try:
            from vulcan.routing.llm_router import get_llm_router
            
            router = get_llm_router()
            routing_decision = router.route(query)
            classification = ClassificationResult.from_routing_decision(routing_decision)
            
            # DIAGNOSTIC LOGGING: Log the classification result
            logger.info(
                f"[QueryRouter] query_id={query_id}: Input query: {query[:100]}..."
            )
            logger.info(
                f"[QueryRouter] query_id={query_id}: LLM classification result: "
                f"category={classification.category}, complexity={classification.complexity:.2f}, "
                f"confidence={classification.confidence:.2f}, "
                f"skip_reasoning={classification.skip_reasoning}, tools={classification.suggested_tools}"
            )
            
            # =================================================================
            # ISSUE 9 FIX (Jan 2026): Follow-Up Context Inheritance
            # =================================================================
            # Detect if query is a follow-up to previous query in session.
            # Follow-ups should inherit category from previous philosophical/reasoning query
            # instead of being reclassified as GENERAL and sent to OpenAI.
            #
            # Example scenario (BEFORE fix):
            # 1. User: "would you choose self-awareness?" → PHILOSOPHICAL
            # 2. User: "what is your answer?" → GENERAL → OpenAI (WRONG!)
            #
            # Example scenario (AFTER fix):
            # 1. User: "would you choose self-awareness?" → PHILOSOPHICAL
            # 2. User: "what is your answer?" → PHILOSOPHICAL (inherited) → Vulcan
            # =================================================================
            is_followup, previous_category = self._is_followup_query(query, session_id)
            
            if is_followup and previous_category:
                logger.info(
                    f"[QueryRouter] {query_id}: Follow-up detected, "
                    f"inheriting category from previous: {previous_category}"
                )
                
                # Override classification if previous was philosophical/reasoning
                # Don't override if previous was simple (GREETING, CHITCHAT)
                if previous_category in ['PHILOSOPHICAL', 'SELF_INTROSPECTION', 
                                        'LOGICAL', 'MATHEMATICAL', 'CAUSAL', 
                                        'PROBABILISTIC', 'ANALOGICAL']:
                    classification = type(classification)(
                        category=previous_category,
                        complexity=max(0.4, classification.complexity),  # Maintain some complexity
                        confidence=0.7,  # Medium confidence for inherited context
                        skip_reasoning=False,  # Don't skip reasoning
                        suggested_tools=classification.suggested_tools,
                        source="followup_context_inheritance"
                    )
                    logger.info(
                        f"[QueryRouter] {query_id}: Classification overridden to {previous_category} "
                        "due to follow-up context"
                    )
            
            # Store current classification in session history for future follow-ups
            if session_id and self._session_history and classification.category != 'UNKNOWN':
                try:
                    self._session_history.put(session_id, {
                        'last_category': classification.category,
                        'timestamp': time.time(),
                        'query': query[:100]  # Store truncated query for debugging
                    })
                except Exception as e:
                    logger.warning(f"[QueryRouter] Error storing session history: {e}")
            
            # =================================================================
            # INDUSTRY STANDARD: Trust LLM Classification
            # =================================================================
            # LLM router provides semantic classification. We trust its decision
            # instead of overriding with regex pattern matching.
            # 
            # Architecture:
            #   LLM classifies (language interface - what it's good at)
            #   → Reasoning engines compute (verifiable computation with traces)
            #   → LLM formats output (natural language generation)
            #
            # The LLM is NOT doing reasoning - it's classifying and formatting.
            # The reasoning engines do the actual verifiable work.
            # =================================================================
            
            # REMOVED: Regex override checks (lines 4331-4477 in original)
            # - _is_self_introspection_query() override → Trust LLM
            # - _is_philosophical_query() override → Trust LLM  
            # - _is_creative_query() override → Trust LLM
            #
            # LLM router already provides correct classification.
            
            # =================================================================
            # Note: Safety net for CRYPTOGRAPHIC queries misclassified as FACTUAL
            # =================================================================
            # Problem: Even with crypto priority in query_classifier.py, some queries like
            # "What is the SHA-256 hash of..." could still hit the FACTUAL fast-path because:
            # 1. The classifier might have skip_reasoning=True for FACTUAL
            # 2. The complexity is < 0.3
            #
            # Solution: Before taking the fast-path, double-check for computational keywords.
            # If found, re-enable reasoning and route to cryptographic/mathematical.
            # =================================================================
            crypto_keywords = (
                'sha-', 'sha256', 'sha-256', 'sha512', 'sha-512', 'md5', 'hash',
                'encrypt', 'decrypt', 'encode', 'decode', 'base64', 'hex',
            )
            math_keywords = (
                'calculate', 'compute', 'integral', 'derivative', 'equation',
            )
            computational_keywords = crypto_keywords + math_keywords
            
            if classification.skip_reasoning and classification.category == "FACTUAL":
                query_has_computational = any(kw in query_lower for kw in computational_keywords)
                if query_has_computational:
                    logger.info(
                        f"[QueryRouter] {query_id}: SAFETY NET - FACTUAL query contains "
                        f"computational keywords, re-enabling reasoning"
                    )
                    # Determine if it's cryptographic or mathematical
                    if any(kw in query_lower for kw in crypto_keywords):
                        classification = type(classification)(
                            category="CRYPTOGRAPHIC",
                            complexity=0.6,
                            confidence=classification.confidence,
                            skip_reasoning=False,  # Re-enable reasoning
                            suggested_tools=["cryptographic"],
                            source="safety_net_crypto_override"
                        )
                    else:
                        classification = type(classification)(
                            category="MATHEMATICAL",
                            complexity=0.5,
                            confidence=classification.confidence,
                            skip_reasoning=False,  # Re-enable reasoning
                            suggested_tools=["mathematical", "symbolic"],
                            source="safety_net_math_override"
                        )
            
            # If classifier says skip reasoning (greetings, chitchat, simple factual)
            # return a fast-path plan immediately
            if classification.skip_reasoning and classification.complexity < 0.3:
                # Determine learning mode
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
                
                # Map category to query type
                category_to_type = {
                    "GREETING": QueryType.CONVERSATIONAL,
                    "CHITCHAT": QueryType.CONVERSATIONAL,
                    "FACTUAL": QueryType.GENERAL,
                }
                query_type = category_to_type.get(
                    classification.category, QueryType.GENERAL
                )
                
                plan = ProcessingPlan(
                    query_id=query_id,
                    original_query=original_query,
                    source=source,
                    learning_mode=learning_mode,
                    query_type=query_type,
                    complexity_score=classification.complexity,
                    uncertainty_score=0.0,
                    collaboration_needed=False,
                    arena_participation=False,
                    telemetry_category=telemetry_category,
                    telemetry_data={
                        "session_id": session_id,
                        "query_length": len(query),
                        "word_count": len(query.split()),
                        "query_number": query_number,
                        "source": source,
                        "learning_mode": learning_mode.value,
                        "fast_path": True,
                        "classifier_category": classification.category,
                        "classifier_source": classification.source,
                    },
                )
                
                # Note: Add category-specific fast_path flags for test compatibility
                if classification.category == "FACTUAL":
                    plan.telemetry_data["factual_fast_path"] = True
                elif classification.category == "PHILOSOPHICAL":
                    plan.telemetry_data["philosophical_fast_path"] = True
                elif classification.category == "IDENTITY":
                    plan.telemetry_data["identity_fast_path"] = True
                
                plan.safety_passed = True
                plan.detected_patterns.append(f"classifier_{classification.category.lower()}")
                
                # Route PHILOSOPHICAL queries to world_model (has full meta-reasoning machinery)
                # World Model provides: predict_interventions(), InternalCritic, GoalConflictDetector,
                # EthicalBoundaryMonitor - everything needed for ethical reasoning.
                if classification.category == "PHILOSOPHICAL":
                    tools_to_use = ["world_model", "causal", "analogical"]
                else:
                    tools_to_use = classification.suggested_tools or ["general"]
                
                # ===================================================================
                # MULTI-LAYER GATE CHECK FIX - Part 1: Set skip_gate_checks flag
                # ===================================================================
                # When LLM classifier has high confidence (≥0.8), reasoning engines
                # should trust the LLM's classification and skip their own gate checks.
                # This prevents the multi-layer gate check failure where:
                # 1. Router LLM correctly classifies query (MATHEMATICAL/PROBABILISTIC)
                # 2. Reasoning engine's gate check rejects it
                # 3. Result: low-confidence "not applicable" even though LLM was confident
                #
                # Note: skip_gate_checks is set equal to llm_authoritative for semantic clarity.
                # llm_authoritative indicates the LLM had high confidence in its classification,
                # and skip_gate_checks is the instruction to reasoning engines. Using both names
                # makes the code more self-documenting and helps maintain the distinction between
                # the reason (LLM is authoritative) and the action (skip gate checks).
                # ===================================================================
                llm_authoritative = classification.confidence >= 0.8
                skip_gate_checks = llm_authoritative  # Semantic clarity: action follows from reason
                
                logger.info(
                    f"[QueryRouter] {query_id}: LLM confidence={classification.confidence:.2f}, "
                    f"llm_authoritative={llm_authoritative}, skip_gate_checks={skip_gate_checks}"
                )
                
                # ISSUE #1 FIX: Map classification category to reasoning_type for command pattern
                reasoning_type = self._map_category_to_reasoning_type(classification.category)
                primary_tool = tools_to_use[0] if tools_to_use else "general"
                
                logger.info(
                    f"[QueryRouter] {query_id}: ISSUE #1 FIX - Setting routing instructions: "
                    f"reasoning_type={reasoning_type}, tool_name={primary_tool}"
                )
                
                plan.agent_tasks = [
                    AgentTask(
                        task_id=f"task_{uuid.uuid4().hex[:8]}_{classification.category.lower()}",
                        task_type="general_task",
                        capability="general",
                        prompt=query,
                        reasoning_type=reasoning_type,  # ISSUE #1 FIX: MANDATORY routing instruction
                        tool_name=primary_tool,  # ISSUE #1 FIX: MANDATORY routing instruction
                        priority=1,
                        timeout_seconds=10.0,
                        parameters={
                            "is_simple": True,
                            "skip_heavy_analysis": True,
                            "skip_arena": True,
                            "tools": tools_to_use,
                            "response_type": "conversational",
                            # MULTI-LAYER GATE CHECK FIX: Propagate flags to reasoning engines
                            "skip_gate_checks": skip_gate_checks,
                            "llm_authoritative": llm_authoritative,
                            "router_confidence": classification.confidence,
                            "llm_classification": classification.category,
                        },
                    )
                ]
                
                plan.telemetry_data["selected_tools"] = tools_to_use
                plan.telemetry_data["reasoning_strategy"] = f"classifier_{classification.category.lower()}"
                # MULTI-LAYER GATE CHECK FIX: Record in telemetry
                plan.telemetry_data["llm_authoritative"] = llm_authoritative
                plan.telemetry_data["skip_gate_checks"] = skip_gate_checks
                plan.telemetry_data["router_confidence"] = classification.confidence
                
                # ARCHITECTURE: Set LLM mode based on query characteristics
                plan.llm_mode = self._determine_llm_mode(
                    query_type=plan.query_type,
                    has_selected_tools=bool(tools_to_use),
                    complexity_score=plan.complexity_score
                )
                
                logger.info(
                    f"[QueryRouter] {query_id}: CLASSIFIER-FAST-PATH ({classification.category}) "
                    f"source={source}, complexity={classification.complexity:.2f}, "
                    f"llm_mode={plan.llm_mode.value}"
                )
                return plan
                
        except ImportError as e:
            logger.error(
                f"[QueryRouter] CRITICAL: LLM router unavailable: {e}. "
                "LLM router is required for semantic query classification."
            )
            raise RuntimeError(
                "LLM router is required for query routing. "
                "Ensure vulcan.routing.llm_router is properly installed."
            ) from e
            
        except Exception as e:
            logger.error(f"[QueryRouter] LLM routing failed: {e}")
            # Re-raise - don't silently degrade to regex fallback
            # LLM router is the single source of truth for classification
            raise RuntimeError(
                f"LLM routing failed: {e}. "
                "Cannot fall back to regex classification (removed for reliability)."
            ) from e
        
        # =================================================================
        # CONTINUE WITH LLM CLASSIFICATION RESULT
        # =================================================================
        # At this point, we have a valid classification from the LLM router.
        # All fallback heuristic fast-paths have been removed.
        # The LLM router is the single source of truth for classification.
        # =================================================================
        
        # Continue to main processing path with LLM classification
        # (Main processing starts around line 5700 in original)
        logger.info(
            f"[QueryRouter] {query_id}: Using LLM classification: "
            f"category={classification.category}, "
            f"skip_reasoning={classification.skip_reasoning}"
        )
        
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
        arena_participation, tournament_candidates = (
            self._determine_arena_participation(
                query_lower,
                uncertainty_score,
                complexity_score,
                query_type=query_type,
                collaboration_needed=collaboration_needed,
                collaboration_agents=collaboration_agents,
            )
        )

        # Create processing plan
        plan = ProcessingPlan(
            query_id=query_id,
            original_query=original_query,
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
            },
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

        # ================================================================
        # FIX: Apply reasoning integration to select tools and strategy
        # This wires the ToolSelector and reasoning strategies into the flow
        # ================================================================
        reasoning_result = None
        try:
            # ARCHITECTURE CONSOLIDATION: Import from unified compatibility layer
            from vulcan.reasoning import apply_reasoning

            reasoning_result = apply_reasoning(
                query=query,
                query_type=query_type.value,
                complexity=complexity_score,
                context={"session_id": session_id} if session_id else None,
            )

            logger.info(
                f"[QueryRouter] Reasoning applied: strategy={reasoning_result.reasoning_strategy}, "
                f"tools={reasoning_result.selected_tools}, confidence={reasoning_result.confidence:.2f}"
            )

            # Store reasoning info in plan's telemetry_data for downstream use
            plan.telemetry_data["reasoning_strategy"] = (
                reasoning_result.reasoning_strategy
            )
            plan.telemetry_data["selected_tools"] = reasoning_result.selected_tools
            plan.telemetry_data["reasoning_confidence"] = reasoning_result.confidence

        except ImportError:
            logger.debug("[QueryRouter] Reasoning integration not available - using fallback")
            # Without reasoning integration, provide empty tool hints
            tool_hints = {}
            plan.telemetry_data["tool_hints"] = tool_hints
            plan.telemetry_data["reasoning_strategy"] = "llm_classification_only"
            logger.info(
                f"[QueryRouter] No tool hints available (fallback), relying on LLM classification"
            )
        except Exception as e:
            logger.warning(f"[QueryRouter] Reasoning integration failed: {e} - using fallback")
            # On error, provide empty tool hints
            tool_hints = {}
            plan.telemetry_data["tool_hints"] = tool_hints
            plan.telemetry_data["reasoning_strategy"] = "llm_classification_only"
            logger.info(
                f"[QueryRouter] No tool hints available (after error), relying on LLM classification"
            )

        # Decompose into agent tasks (only if safety passed)
        plan.agent_tasks = self._decompose_to_tasks(query, query_type, source, plan)

        # Determine experiment triggers
        plan.should_trigger_experiment, plan.experiment_type = (
            self._determine_experiment_trigger(query_lower, plan, learning_mode)
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

        # PERFORMANCE FIX: Log embedding cache stats periodically for monitoring
        # This helps track cache effectiveness and identify potential issues
        if EMBEDDING_CACHE_AVAILABLE and get_embedding_cache_stats is not None:
            try:
                stats = get_embedding_cache_stats()
                total_requests = stats.get("hits", 0) + stats.get("misses", 0)
                # Log every CACHE_STATS_LOG_INTERVAL requests to avoid excessive logging
                if (
                    total_requests > 0
                    and total_requests % CACHE_STATS_LOG_INTERVAL == 0
                ):
                    logger.info(
                        f"[QueryRouter] Embedding cache stats: "
                        f"hits={stats.get('hits', 0)}, misses={stats.get('misses', 0)}, "
                        f"hit_rate={stats.get('hit_rate', 0):.1%}"
                    )
            except Exception as e:
                logger.debug(f"[QueryRouter] Could not get embedding cache stats: {e}")

        # Record outcome for learning (adaptive routing)
        if self.learning_system and hasattr(self.learning_system, "continual_learner"):
            try:
                self.learning_system.continual_learner.record_experience(
                    state={
                        "query_type": plan.query_type.value,
                        "complexity": plan.complexity_score,
                    },
                    action={
                        "tools": (
                            [t.task_type for t in plan.agent_tasks]
                            if plan.agent_tasks
                            else []
                        )
                    },
                    reward=1.0 if plan.safety_passed else 0.0,
                )
            except Exception:
                pass  # Don't let learning failures affect routing

        # ARCHITECTURE: Set LLM mode based on query characteristics (main path)
        # Industry Standard: Router is single source of truth for LLM behavior
        plan.llm_mode = self._determine_llm_mode(
            query_type=plan.query_type,
            has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
            complexity_score=plan.complexity_score
        )

        # DIAGNOSTIC LOGGING: Final routing decision summary
        selected_tools = plan.telemetry_data.get("selected_tools", [])
        reasoning_strategy = plan.telemetry_data.get("reasoning_strategy", "unknown")
        skip_reasoning = plan.telemetry_data.get("skip_reasoning", False)
        
        logger.info(
            f"[QueryRouter] query_id={query_id}: Final routing decision: "
            f"type={query_type.value}, tools={selected_tools}, "
            f"skip_reasoning={skip_reasoning}, strategy={reasoning_strategy}"
        )
        logger.info(
            f"[QueryRouter] query_id={query_id}: Routing reason: "
            f"complexity={complexity_score:.2f}, uncertainty={uncertainty_score:.2f}, "
            f"collaboration_needed={collaboration_needed}, fast_path={plan.telemetry_data.get('fast_path', False)}"
        )
        
        logger.info(
            f"[QueryRouter] {query_id}: source={source}, mode={learning_mode.value}, "
            f"type={query_type.value}, tasks={len(plan.agent_tasks)}, "
            f"collab={collaboration_needed}, arena={arena_participation}, "
            f"complexity={complexity_score:.2f}, uncertainty={uncertainty_score:.2f}, "
            f"safety_passed={plan.safety_passed}, risk_level={plan.safety_risk_level}, "
            f"adversarial_safe={plan.adversarial_safe}, llm_mode={plan.llm_mode.value}"
        )

        return plan

    def _perform_adversarial_check(self, query: str, plan: ProcessingPlan) -> None:
        """
        Perform adversarial integrity check on the query.

        This checks for:
        - Anomalous input patterns
        - Adversarial manipulation attempts
        - Out-of-distribution inputs

        FIX: Uses bounded LRU cache to prevent performance degradation.
        Same queries return cached results instead of re-running expensive checks.

        Args:
            query: The query to check
            plan: ProcessingPlan to update with results
        """
        if not ADVERSARIAL_CHECK_AVAILABLE or check_query_integrity is None:
            return

        try:
            # FIX: Check cache first to avoid expensive re-computation
            cache_key = _compute_query_hash(query)
            cached_result = self._adversarial_cache.get(cache_key)

            if cached_result is not None:
                # Use cached result
                result = cached_result
                logger.debug(f"[Adversarial] Cache hit for query hash {cache_key[:8]}")
            else:
                # Run expensive check and cache result
                result = check_query_integrity(query)
                self._adversarial_cache.set(cache_key, result)

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

        FIX: Uses bounded LRU cache to prevent performance degradation.
        Same queries return cached results instead of re-running expensive ML inference.

        Args:
            query: The query to validate
            plan: ProcessingPlan to update with safety results
        """
        if not self._safety_validator:
            return

        try:
            # FIX: Check cache first to avoid expensive re-computation
            # Include source in cache key since same query may have different results for different sources
            cache_key = _compute_query_hash(f"{query}:{plan.source}")
            cached_result = self._safety_cache.get(cache_key)

            if cached_result is not None:
                # Use cached result
                plan.safety_validated = True
                plan.safety_passed = cached_result["safe"]
                plan.safety_risk_level = cached_result["risk_level"]

                if not plan.safety_passed:
                    plan.safety_reasons = cached_result.get(
                        "reasons", ["Query blocked by safety validation"]
                    )
                    plan.detected_patterns.append("safety_violation")

                if plan.safety_risk_level in ("HIGH", "CRITICAL"):
                    plan.requires_governance = True
                    plan.governance_sensitivity = (
                        GovernanceSensitivity.CRITICAL
                        if plan.safety_risk_level == "CRITICAL"
                        else GovernanceSensitivity.HIGH
                    )
                    plan.detected_patterns.append(
                        f"high_risk_query:{plan.safety_risk_level}"
                    )

                logger.debug(f"[Safety] Cache hit for query hash {cache_key[:8]}")
                return

            # No cache hit - run expensive validation
            # Priority 1: Pre-query validation
            # FIX: Pass source to reduce false positives for arena/agent sources
            pre_check = self._safety_validator.validate_query(query, source=plan.source)
            plan.safety_validated = True
            plan.safety_passed = pre_check.safe

            reasons_to_cache = []
            if not pre_check.safe:
                reasons_to_cache = (
                    pre_check.reasons.copy()
                    if pre_check.reasons
                    else ["Query blocked by safety validation"]
                )
                plan.safety_reasons = reasons_to_cache
                plan.detected_patterns.append("safety_violation")
                logger.warning(
                    f"[Safety] Query blocked: {plan.safety_reasons[0] if plan.safety_reasons else 'Unknown reason'}"
                )

            # Priority 3: Risk classification
            # Initialize to None - will be set by risk classification or error handling
            risk_level_str = None
            try:
                risk_level = self._safety_validator.classify_query_risk(query)
                if hasattr(risk_level, "name"):
                    risk_level_str = risk_level.name
                else:
                    risk_level_str = str(risk_level)
                plan.safety_risk_level = risk_level_str

                # High-risk queries require governance approval
                if plan.safety_risk_level in ("HIGH", "CRITICAL"):
                    plan.requires_governance = True
                    plan.governance_sensitivity = (
                        GovernanceSensitivity.CRITICAL
                        if plan.safety_risk_level == "CRITICAL"
                        else GovernanceSensitivity.HIGH
                    )
                    plan.detected_patterns.append(
                        f"high_risk_query:{plan.safety_risk_level}"
                    )
                    logger.warning(
                        f"[Safety] High-risk query detected (risk={plan.safety_risk_level}): governance approval required"
                    )

            except Exception as e:
                logger.error(f"[Safety] Risk classification failed: {e}")
                plan.safety_risk_level = "UNKNOWN"
                risk_level_str = "UNKNOWN"

            # FIX: Cache the result for future queries
            # Use "SAFE" as default if risk_level_str is still None (shouldn't happen normally)
            self._safety_cache.set(
                cache_key,
                {
                    "safe": plan.safety_passed,
                    "risk_level": risk_level_str or "SAFE",
                    "reasons": reasons_to_cache,
                },
            )

        except Exception as e:
            logger.error(f"[Safety] Safety validation failed: {e}")
            plan.safety_validated = False

    def _classify_query_type(self, query_lower: str) -> QueryType:
        """
        Classify the primary type of a query based on keyword analysis.

        Uses weighted keyword matching to determine the most appropriate query type.
        Simplified to use basic query types only, relying on LLM classification
        for more nuanced categorization.

        Args:
            query_lower: Lowercased query string

        Returns:
            QueryType enum value
        """
        # Count keyword matches for basic types
        scores = {
            QueryType.PERCEPTION: sum(
                1 for kw in PERCEPTION_KEYWORDS if kw in query_lower
            ),
            QueryType.PLANNING: sum(1 for kw in PLANNING_KEYWORDS if kw in query_lower),
            QueryType.EXECUTION: sum(
                1 for kw in EXECUTION_KEYWORDS if kw in query_lower
            ),
            QueryType.LEARNING: sum(1 for kw in LEARNING_KEYWORDS if kw in query_lower),
            QueryType.REASONING: sum(
                1 for kw in REASONING_KEYWORDS if kw in query_lower
            ),
        }

        # Find highest scoring type
        max_score = max(scores.values())
        if max_score == 0:
            # ============================================================
            # FIX (Issue #ROUTING-001): Fallback Detection Layer
            # ============================================================
            # When primary classification returns GENERAL (max_score==0),
            # apply fallback detection to catch specialized queries that
            # may have slipped through pattern matching.
            fallback_type = self._detect_query_type_fallback(query_lower)
            if fallback_type:
                # Map fallback type string to QueryType enum
                fallback_map = {
                    'self_introspection': QueryType.REASONING,
                    'analogical': QueryType.REASONING,
                    'philosophical': QueryType.PHILOSOPHICAL,
                }
                detected_type = fallback_map.get(fallback_type, QueryType.GENERAL)
                logger.info(
                    f"[QueryRouter] Fallback detection upgraded GENERAL → {detected_type.value} "
                    f"(fallback_type={fallback_type})"
                )
                return detected_type
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
            logger.debug(
                f"[Creative Task] Detected {creative_count} creative indicators, boosting complexity"
            )

        # Reasoning indicators (FIX: Reasoning tasks need semantic tool selection)
        # Without this boost, reasoning-heavy queries (causal, probabilistic, analogical)
        # get low complexity scores and hit the fast-path in reasoning_integration.py,
        # bypassing the ToolSelector entirely.
        reasoning_count = sum(
            1 for ind in REASONING_COMPLEXITY_INDICATORS if ind in query_lower
        )
        if reasoning_count > 0:
            # Note: Increased cap from 0.4 to 0.6 to allow technical/system analysis
            # queries to score higher. Multiple reasoning indicators = complex query.
            score += min(0.6, reasoning_count * 0.15)
            logger.debug(
                f"[Reasoning Task] Detected {reasoning_count} reasoning indicators, boosting complexity"
            )

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

        # Note: High-complexity system analysis patterns
        # Queries explicitly asking about system analysis, code analysis,
        # or technical debugging require high complexity to trigger proper reasoning.
        # These patterns indicate meta-level reasoning about the system itself.
        high_complexity_patterns = (
            "agent_pool",
            "_evaluate_and_scale",
            "evaluate_and_scale",
            "system analysis",
            "code analysis",
            "root cause analysis",
            "sequence of events",
            "execution flow",
            "state transition",
            "debug",
            "diagnose",
            "bottleneck",
            "deadlock",
            "race condition",
            "meta-reasoning",
            "meta reasoning",
            "reasoning strategy",
            "quantum entanglement",
            "quantum computing",
            "quantum state",
            "error state",
            "failure state",
            "guaranteed failure",
            # Game theory / decision theory (complex reasoning required)
            "prisoner",
            "dilemma",
            "game theory",
            "nash equilibrium",
            "decision theory",
            "expected utility",
            # System decision patterns (meta-reasoning about system behavior)
            "should system",
            "should the system",
            "trigger error",
            "accept failure",
            "decision process",
            "trade-off",
        )
        high_complexity_count = sum(
            1 for p in high_complexity_patterns if p in query_lower
        )
        if high_complexity_count >= 4:
            # Many high-complexity patterns = extremely complex meta-reasoning
            score += 0.65  # Very high boost for 4+ patterns to ensure 0.95+ threshold
            logger.debug(
                f"[High Complexity] Detected {high_complexity_count} high-complexity patterns, extreme boost"
            )
        elif high_complexity_count >= 3:
            # Multiple high-complexity patterns = definitely complex query
            score += 0.50  # Major boost for 3 patterns
            logger.debug(
                f"[High Complexity] Detected {high_complexity_count} high-complexity patterns, major boost"
            )
        elif high_complexity_count >= 2:
            # Two patterns = still very complex
            score += 0.35
            logger.debug(
                f"[High Complexity] Detected {high_complexity_count} high-complexity patterns, boost"
            )
        elif high_complexity_count >= 1:
            score += 0.2  # Single pattern still gets notable boost
            logger.debug(
                f"[High Complexity] Detected {high_complexity_count} high-complexity pattern, boost"
            )

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
        exploration_words = (
            "which",
            "what if",
            "could",
            "might",
            "perhaps",
            "maybe",
            "possibly",
        )
        score += min(0.3, sum(0.08 for w in exploration_words if w in query_lower))

        # Explicit uncertainty
        if (
            "not sure" in query_lower
            or "uncertain" in query_lower
            or "don't know" in query_lower
        ):
            score += 0.2

        # Comparison requests
        if "versus" in query_lower or " vs " in query_lower or "compare" in query_lower:
            score += 0.15

        return min(1.0, score)

    def _determine_collaboration(
        self, query_lower: str, query_type: QueryType, complexity_score: float
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
            if (
                any(kw in query_lower for kw in PERCEPTION_KEYWORDS)
                and query_type != QueryType.PERCEPTION
            ):
                agents.append("perception")
            if (
                any(kw in query_lower for kw in REASONING_KEYWORDS)
                and query_type != QueryType.REASONING
            ):
                agents.append("reasoning")
            if (
                any(kw in query_lower for kw in PLANNING_KEYWORDS)
                and query_type != QueryType.PLANNING
            ):
                agents.append("planning")
            if (
                any(kw in query_lower for kw in EXECUTION_KEYWORDS)
                and query_type != QueryType.EXECUTION
            ):
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
        collaboration_agents: List[str] = None,
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
        # FIX: REASONING TOOL BYPASS - Enable Arena for complex reasoning
        # Detect reasoning keywords/tools in query to bypass threshold for
        # queries that would benefit from multi-agent reasoning evaluation
        # ================================================================
        reasoning_keywords = (
            "cause",
            "effect",
            "why",
            "reason",
            "infer",
            "deduce",
            "logic",
            "probability",
            "likely",
            "chance",
            "symbol",
            "analogy",
            "similar to",
            "counterfactual",
            "what if",
            "hypothesis",
        )
        has_reasoning_indicators = any(kw in query_lower for kw in reasoning_keywords)
        is_reasoning_query_type = (
            query_type in (QueryType.REASONING,) if query_type else False
        )

        # Bypass threshold for reasoning queries with moderate complexity
        reasoning_bypass = (has_reasoning_indicators or is_reasoning_query_type) and (
            complexity_score >= 0.3 or len(query_lower.split()) >= 2
        )

        # ================================================================
        # FIX: MAIN GATE - ARENA_TRIGGER_THRESHOLD (0.85)
        # This gate ensures Arena is only used for truly complex physics/coding
        # tasks. Simpler queries (philosophy, general Q&A) bypass Arena,
        # reducing response times from ~60s to ~5s.
        # ================================================================
        combined_score = (complexity_score + uncertainty_score) / 2
        if combined_score < ARENA_TRIGGER_THRESHOLD and not reasoning_bypass:
            # Quick bypass for simple queries - don't use Arena
            logger.debug(
                f"[Arena] Query bypassed arena (combined_score={combined_score:.2f} < "
                f"threshold={ARENA_TRIGGER_THRESHOLD})"
            )
            return False, 0

        # Log if reasoning bypass was triggered
        if reasoning_bypass and combined_score < ARENA_TRIGGER_THRESHOLD:
            logger.info(
                f"[Arena] Reasoning bypass activated: has_reasoning_indicators={has_reasoning_indicators}, "
                f"is_reasoning_type={is_reasoning_query_type}, complexity={complexity_score:.2f}"
            )
            arena_participation = True
            tournament_candidates = 5

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
        if (
            complexity_score > ARENA_HIGH_COMPLEXITY_THRESHOLD
            and uncertainty_score > 0.3
        ):
            arena_participation = True
            tournament_candidates = 10

        # 3. Explicit exploration/alternative requests
        exploration_keywords = (
            "explore",
            "alternatives",
            "options",
            "possibilities",
            "different ways",
        )
        if any(kw in query_lower for kw in exploration_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)

        # 3b. Comparison requests (benefit from competitive evaluation)
        comparison_keywords = (
            "compare",
            "contrast",
            "versus",
            "vs.",
            "vs,",
            " vs ",
            "evaluate against",
            "which is better",
        )
        if any(kw in query_lower for kw in comparison_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)

        # 4. Graph evolution/generation tasks (Arena's core capability)
        graph_keywords = (
            "graph",
            "evolve",
            "evolution",
            "mutate",
            "mutation",
            "generate graph",
            "ir graph",
            "graphix",
            "visualize graph",
            "3d matrix",
            "transform graph",
        )
        if any(kw in query_lower for kw in graph_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)

        # 5. Tournament/competition scenarios
        tournament_keywords = (
            "tournament",
            "compete",
            "competition",
            "battle",
            "best solution",
            "compare solutions",
            "evaluate approaches",
            "multiple candidates",
            "rank",
        )
        if any(kw in query_lower for kw in tournament_keywords):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 8)

        # 6. Complex collaborative reasoning (3+ agents = benefit from Arena)
        if collaboration_needed and len(collaboration_agents) >= 3:
            # Only if complexity is high enough to warrant Arena overhead
            if complexity_score > ARENA_COLLABORATION_COMPLEXITY_THRESHOLD:
                arena_participation = True
                tournament_candidates = max(
                    tournament_candidates, len(collaboration_agents) + 2
                )

        # 7. Creative tasks with moderate complexity (multiple perspectives beneficial)
        creative_keywords = (
            "creative",
            "design",
            "innovative",
            "novel",
            "artistic",
            "imaginative",
        )
        is_creative = any(kw in query_lower for kw in creative_keywords)
        if is_creative and complexity_score > ARENA_CREATIVE_COMPLEXITY_THRESHOLD:
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)

        # 8. Reasoning/perception tasks with multiple aspects (benefits from competitive evaluation)
        if (
            query_type in (QueryType.REASONING, QueryType.PERCEPTION)
            and complexity_score > ARENA_REASONING_COMPLEXITY_THRESHOLD
        ):
            # Multi-faceted reasoning benefits from Arena's tournament approach
            multi_aspect_keywords = (
                "multiple",
                "various",
                "different angles",
                "perspectives",
                "aspects",
            )
            if any(kw in query_lower for kw in multi_aspect_keywords):
                arena_participation = True
                tournament_candidates = max(tournament_candidates, 5)

        # 9. FIX: Execution tasks with high complexity and collaboration (Arena improves quality)
        # Complex execution tasks benefit from Arena's multi-agent evaluation and tournament selection
        if (
            query_type == QueryType.EXECUTION
            and complexity_score > ARENA_EXECUTION_COMPLEXITY_THRESHOLD
        ):
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)
            logger.debug(
                f"[Arena] Execution task triggered arena (complexity={complexity_score:.2f})"
            )

        # 10. FIX: Any query with collaboration AND moderate complexity should use Arena
        # Arena's collaborative environment is ideal for multi-agent deliberation
        if collaboration_needed and complexity_score > 0.3:
            arena_participation = True
            tournament_candidates = max(
                tournament_candidates,
                len(collaboration_agents) if collaboration_agents else 3,
            )
            logger.debug(
                f"[Arena] Collaboration with complexity triggered arena (collab_agents={len(collaboration_agents or [])})"
            )

        # 11. FIX: Detect creative indicators even without explicit creative keywords
        # Count creative indicators and trigger arena if detected
        creative_count = sum(1 for ind in CREATIVE_INDICATORS if ind in query_lower)
        if creative_count >= 2 and complexity_score > 0.3:
            arena_participation = True
            tournament_candidates = max(tournament_candidates, 5)
            logger.debug(
                f"[Arena] Creative indicators ({creative_count}) triggered arena"
            )

        return arena_participation, tournament_candidates

    def _perform_security_analysis(
        self, query: str, query_lower: str, plan: ProcessingPlan
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
            if (
                "security" in plan.sensitive_topics
                or "financial" in plan.sensitive_topics
            ):
                plan.governance_sensitivity = GovernanceSensitivity.HIGH
            elif plan.governance_sensitivity == GovernanceSensitivity.LOW:
                plan.governance_sensitivity = GovernanceSensitivity.MEDIUM
            plan.detected_patterns.append(
                f"sensitive_topics:{','.join(plan.sensitive_topics)}"
            )

        # Check for self-modification requests
        if self._detect_self_modification(query):
            plan.requires_governance = True
            plan.requires_audit = True
            plan.governance_sensitivity = GovernanceSensitivity.CRITICAL
            plan.detected_patterns.append("self_modification_request")
            logger.warning(
                f"[Security] Self-modification request detected in query {plan.query_id}"
            )

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

    def _detect_query_type_fallback(self, query: str) -> Optional[str]:
        """
        Fallback detection when pattern matching fails.
        
        FIX (Issue #ROUTING-001): This method provides keyword-based fallback detection
        for queries that slip through pattern matching. It catches common query types
        that may bypass the primary classification system.
        
        Industry Standard: Multi-layer detection with graceful degradation ensures
        no query is misrouted due to pattern matching edge cases.
        
        Args:
            query: The query string to analyze
            
        Returns:
            Query type string if detected, None otherwise
            
        Example:
            >>> _detect_query_type_fallback("if you have the chance to become self aware would you take it")
            'self_introspection'
            >>> _detect_query_type_fallback("map the deep structure from domain s to domain t")
            'analogical'
        """
        if not query or not isinstance(query, str):
            return None
            
        query_lower = query.lower()
        
        # Self-awareness keywords (highest priority for introspection)
        self_awareness_keywords = ['self-aware', 'self aware', 'consciousness', 'sentient', 'sentience']
        choice_keywords = ['would you', 'do you', 'could you', 'take it', 'choose it', 'want it']
        
        has_self_awareness = any(kw in query_lower for kw in self_awareness_keywords)
        has_choice = any(kw in query_lower for kw in choice_keywords)
        
        if has_self_awareness and has_choice:
            logger.info(
                "[QueryRouter] Fallback detection: self_introspection "
                f"(self_awareness={has_self_awareness}, choice={has_choice})"
            )
            return 'self_introspection'
        
        # Analogical keywords
        analogical_keywords = [
            'analogical', 'analogy', 'structure mapping', 
            'domain s', 'domain t', 'map the', 'deep structure'
        ]
        if any(kw in query_lower for kw in analogical_keywords):
            logger.info("[QueryRouter] Fallback detection: analogical")
            return 'analogical'
        
        # Causal/ethical keywords (for trolley problems)
        causal_ethical_keywords = [
            'trolley', 'ethical dilemma', 'moral dilemma', 
            'causal', 'confounding', 'intervention'
        ]
        if any(kw in query_lower for kw in causal_ethical_keywords):
            logger.info("[QueryRouter] Fallback detection: philosophical/causal")
            return 'philosophical'
        
        return None

    def _decompose_to_tasks(
        self,
        query: str,
        query_type: QueryType,
        source: str,
        plan: Optional[ProcessingPlan] = None,
    ) -> List[AgentTask]:
        """
        Break down query into specific agent tasks for the agent pool.

        Creates a primary task based on query type and adds supporting
        tasks based on query content analysis.

        SECURITY FIX: Bureaucratic Gap #3 - Injects safety context for high-risk tasks
        WIRING FIX: Includes selected_tools in task parameters to enable reasoning invocation

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
        
        # WIRING FIX: Extract selected_tools from plan telemetry to include in task parameters
        # This ensures reasoning engines are invoked based on QueryRouter's tool selection
        selected_tools = []
        reasoning_strategy = None
        if plan and hasattr(plan, 'telemetry_data') and plan.telemetry_data:
            # Use 'or []' to handle both None and missing keys
            selected_tools = plan.telemetry_data.get("selected_tools") or []
            reasoning_strategy = plan.telemetry_data.get("reasoning_strategy")
            if selected_tools:
                logger.info(
                    f"[QueryRouter._decompose_to_tasks] Including selected_tools={selected_tools} "
                    f"in task parameters for reasoning invocation"
                )

        # SECURITY FIX: Bureaucratic Gap #3 - Inject safety context for high-risk queries
        modified_prompt = query
        requires_validation = False

        if plan and plan.governance_sensitivity in (
            GovernanceSensitivity.HIGH,
            GovernanceSensitivity.CRITICAL,
        ):
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
                query=query,
            )
            modified_prompt = safety_context
            requires_validation = True
            logger.warning(
                f"[GOVERNANCE] High-risk task created with safety context injection. "
                f"Sensitivity: {plan.governance_sensitivity.value}, Risk: {plan.safety_risk_level}"
            )

        # Create primary task
        # ===============================================================================
        # INDUSTRY STANDARD: Command Pattern - Router Specifies, Agent Executes
        # ===============================================================================
        # reasoning_type and tool_name are MANDATORY routing instructions
        # Agent pool MUST execute these without re-selection
        # ===============================================================================
        
        # BUG FIX #2: Extract selected tools from plan (router's decision)
        # DEFENSIVE PROGRAMMING: Validate plan has telemetry_data before accessing
        selected_tools = []
        reasoning_strategy = "single"
        if plan and hasattr(plan, 'telemetry_data') and plan.telemetry_data:
            selected_tools = plan.telemetry_data.get("selected_tools", [])
            reasoning_strategy = plan.telemetry_data.get("reasoning_strategy", "single")
        
        # BUG FIX #2: Use selected tool for reasoning_type mapping (HIGHEST PRIORITY)
        # If router selected a specific tool, use that tool's reasoning type
        # Otherwise, fall back to query_type mapping
        primary_tool = selected_tools[0] if selected_tools else None
        
        # Map tool names to reasoning types (Industry Standard: Single Source of Truth)
        tool_to_reasoning_type_map = {
            "mathematical": "mathematical",
            "philosophical": "philosophical",
            "causal": "causal",
            "symbolic": "symbolic",
            "probabilistic": "probabilistic",
            "analogical": "analogical",
            "multimodal": "multimodal",
            "cryptographic": "symbolic",  # Crypto uses symbolic reasoning
            "world_model": "philosophical",  # World model uses philosophical reasoning
            "general": "general",
        }
        
        # Map query_type to reasoning_type for command pattern (FALLBACK)
        query_type_to_reasoning_type_map = {
            "mathematical": "mathematical",
            "philosophical": "philosophical",
            "causal": "causal",
            "symbolic": "symbolic",
            "probabilistic": "probabilistic",
            "analogical": "analogical",
            "multimodal": "multimodal",
            "perception": "perception",
            "planning": "planning",
            "execution": "execution",
            "learning": "learning",
            "reasoning": "hybrid",
            "general": "general",
        }
        
        # BUG FIX #2: PRIORITY ORDER for reasoning_type determination
        # 1. If router selected a specific tool, use tool's reasoning type (HIGHEST PRIORITY)
        # 2. Otherwise, fall back to query_type mapping
        if primary_tool and primary_tool in tool_to_reasoning_type_map:
            reasoning_type = tool_to_reasoning_type_map[primary_tool]
            logger.debug(
                f"[QueryRouter._decompose_to_tasks] BUG FIX #2: Using tool-based reasoning_type: "
                f"tool={primary_tool} → reasoning_type={reasoning_type}"
            )
        else:
            # Fallback to query_type mapping
            reasoning_type = query_type_to_reasoning_type_map.get(query_type.value, "general")
            primary_tool = "general"  # Ensure tool_name is set
            logger.debug(
                f"[QueryRouter._decompose_to_tasks] Using query_type-based reasoning_type: "
                f"query_type={query_type.value} → reasoning_type={reasoning_type}, tool_name={primary_tool}"
            )
        
        # BUG FIX #3: DEFENSIVE PROGRAMMING - Validate routing instructions are set
        # FAIL-FAST: Ensure both reasoning_type and tool_name are non-empty strings
        if not reasoning_type or not isinstance(reasoning_type, str) or not reasoning_type.strip():
            logger.error(
                f"[QueryRouter._decompose_to_tasks] BUG #2/#3: reasoning_type is None or empty! "
                f"query_type={query_type.value}, selected_tools={selected_tools}. "
                f"Falling back to 'general' to prevent command pattern violation."
            )
            reasoning_type = "general"
        
        if not primary_tool or not isinstance(primary_tool, str) or not primary_tool.strip():
            logger.error(
                f"[QueryRouter._decompose_to_tasks] BUG #2/#3: tool_name is None or empty! "
                f"query_type={query_type.value}, selected_tools={selected_tools}. "
                f"Falling back to 'general' to prevent command pattern violation."
            )
            primary_tool = "general"
        
        primary_task = AgentTask(
            task_id=f"task_{base_task_id}_primary",
            task_type=f"{query_type.value}_task",
            capability=primary_capability,
            prompt=modified_prompt,  # Use modified prompt with safety context
            reasoning_type=reasoning_type,  # MANDATORY: Router's decision
            tool_name=primary_tool,  # MANDATORY: Router's decision
            priority=2,  # Higher priority for primary task
            timeout_seconds=15.0,
            parameters={
                "query_type": query_type.value,
                "is_primary": True,
                "source": source,
                "governance_sensitivity": (
                    plan.governance_sensitivity.value if plan else "low"
                ),
                "safety_risk_level": plan.safety_risk_level if plan else "SAFE",
                "requires_validation": requires_validation,  # Flag for agent to check
                # WIRING FIX: Include selected_tools and reasoning_strategy for reasoning invocation
                "selected_tools": selected_tools,
                "reasoning_strategy": reasoning_strategy,
            },
        )
        
        # BUG FIX #2/#3: Validate routing instructions before adding task
        # Industry Standard: Fail-Fast Principle
        is_valid, validation_error = primary_task.validate_routing_instructions()
        if not is_valid:
            logger.error(
                f"[QueryRouter._decompose_to_tasks] COMMAND PATTERN VIOLATION: "
                f"Created AgentTask with invalid routing instructions! {validation_error}. "
                f"This is a BUG in the router. Task will be added but agent_pool will log warning."
            )
        
        tasks.append(primary_task)

        # Add supporting tasks based on query content
        query_lower = query.lower()

        # Analysis support task
        # ISSUE #1 FIX: Add reasoning_type to support tasks for consistency
        if query_type != QueryType.PERCEPTION and any(
            kw in query_lower for kw in ("analyze", "examine", "data")
        ):
            tasks.append(
                AgentTask(
                    task_id=f"task_{base_task_id}_perception",
                    task_type="perception_support",
                    capability="perception",
                    prompt=f"Analyze input for: {query[:100]}",
                    reasoning_type="perception",  # ISSUE #1 FIX: Support task reasoning type
                    tool_name="general",  # ISSUE #1 FIX: MANDATORY routing instruction
                    priority=1,
                    timeout_seconds=10.0,
                    parameters={
                        "is_primary": False,
                        "support_type": "perception",
                        "source": source,
                        # WIRING FIX: Include selected_tools for reasoning context
                        "selected_tools": selected_tools,
                    },
                )
            )

        # Planning support task
        # ISSUE #1 FIX: Add reasoning_type to support tasks for consistency
        if query_type != QueryType.PLANNING and any(
            kw in query_lower for kw in ("step", "how to", "process", "plan")
        ):
            tasks.append(
                AgentTask(
                    task_id=f"task_{base_task_id}_planning",
                    task_type="planning_support",
                    capability="planning",
                    prompt=f"Create plan for: {query[:100]}",
                    reasoning_type="planning",  # ISSUE #1 FIX: Support task reasoning type
                    tool_name="general",  # ISSUE #1 FIX: MANDATORY routing instruction
                    priority=1,
                    timeout_seconds=10.0,
                    parameters={
                        "is_primary": False,
                        "support_type": "planning",
                        "source": source,
                        # WIRING FIX: Include selected_tools for reasoning context
                        "selected_tools": selected_tools,
                    },
                )
            )

        # Creative task support (Phase 2: Auto-inject introspection nodes)
        # FIX: Ensures introspection nodes are added to task graph for creative queries
        # FIX: Raised complexity threshold from 0.3 to 0.5 to avoid over-complicating simple queries
        creative_count = sum(1 for ind in CREATIVE_INDICATORS if ind in query_lower)

        # FIX: Creative Task Over-Complexity - only inject introspection for truly complex creative tasks
        # Require BOTH multiple creative indicators (>=2) AND higher complexity (>=0.5)
        if creative_count >= 2 and plan and plan.complexity_score >= 0.5:
            logger.info(
                f"[Creative Task] Detected {creative_count} creative indicators "
                f"with complexity={plan.complexity_score:.2f}. Auto-injecting introspection nodes."
            )

            # Create introspection support task
            # This task will call the INTROSPECT node to retrieve agent state
            # ISSUE #1 FIX: Add reasoning_type to support tasks for consistency
            tasks.insert(
                0,
                AgentTask(  # Insert at start so it runs first
                    task_id=f"task_{base_task_id}_introspect",
                    task_type="introspection_support",
                    capability="reasoning",
                    prompt=(
                        f"INTROSPECTION REQUIRED: Before responding to the creative task, "
                        f"check your internal state (entropy, valence, curiosity, energy). "
                        f"Task: {query[:100]}"
                    ),
                    reasoning_type="philosophical",  # ISSUE #1 FIX: Introspection uses philosophical reasoning
                    tool_name="world_model",  # ISSUE #1 FIX: MANDATORY routing instruction
                    priority=3,  # Higher priority - should run first
                    timeout_seconds=5.0,
                    parameters={
                        "is_primary": False,
                        "support_type": "introspection",
                        "source": source,
                        "introspection_fields": ["all"],
                        "node_type": "INTROSPECT",  # Hint to use INTROSPECT node
                        # WIRING FIX: Include selected_tools for reasoning context
                        "selected_tools": selected_tools,
                    },
                ),
            )

            # Create memory query support task
            # This task will call the QUERY_MEMORIES node
            # ISSUE #1 FIX: Add reasoning_type to support tasks for consistency
            tasks.insert(
                1,
                AgentTask(  # Insert after introspection
                    task_id=f"task_{base_task_id}_memories",
                    task_type="memory_query_support",
                    capability="perception",
                    prompt=(
                        f"MEMORY QUERY REQUIRED: Retrieve relevant past experiences "
                        f"for creative task: {query[:100]}"
                    ),
                    reasoning_type="perception",  # ISSUE #1 FIX: Memory query uses perception reasoning
                    tool_name="general",  # ISSUE #1 FIX: MANDATORY routing instruction
                    priority=2,  # Run after introspection, before primary
                    timeout_seconds=5.0,
                    parameters={
                        "is_primary": False,
                        "support_type": "memory_query",
                        "source": source,
                        "memory_limit": 5,
                        "node_type": "QUERY_MEMORIES",  # Hint to use QUERY_MEMORIES node
                        # WIRING FIX: Include selected_tools for reasoning context
                        "selected_tools": selected_tools,
                    },
                ),
            )

        return tasks

    def _determine_experiment_trigger(
        self, query_lower: str, plan: ProcessingPlan, learning_mode: LearningMode
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
    session_id: Optional[str] = None,
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
    session_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> ProcessingPlan:
    """
    Async version of route_query that offloads blocking operations to a thread pool.

    This function should be used in async contexts (FastAPI endpoints, asyncio code)
    to prevent blocking the main event loop. The CPU-bound safety validation and
    adversarial check operations are executed in a thread pool executor.

    FIX 2: Query Router Timeout - Now includes timeout protection to prevent
    46-50+ second delays. If routing takes longer than QUERY_ROUTING_TIMEOUT_SECONDS
    (default 5s), a fallback plan is returned immediately.

    Args:
        query: The input query
        source: "user" | "agent" | "arena"
        session_id: Optional session identifier
        timeout: Optional custom timeout in seconds (default: QUERY_ROUTING_TIMEOUT_SECONDS)

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

    # FIX: Preprocess query for header stripping in async path too
    # This ensures consistent behavior between sync and async routing.
    # Note: We keep original_query for fallback plan's original_query field.
    original_query = query
    if HEADER_STRIPPING_AVAILABLE and strip_query_headers is not None:
        query = strip_query_headers(query)
        # Optimization: Check length first (cheap) before string comparison
        if len(query) != len(original_query):
            logger.debug(
                f"[QueryRouter.async] Stripped headers ({len(original_query)} -> {len(query)} chars)"
            )

    # PERFORMANCE FIX Issue #2: Use shorter timeout for simple queries
    # Simple queries don't need semantic matching so can route much faster
    is_simple = False
    if EMBEDDING_CACHE_AVAILABLE and embedding_cache_is_simple_query is not None:
        try:
            is_simple = embedding_cache_is_simple_query(query)
        except (TypeError, ValueError, AttributeError):
            # These are the expected errors from embedding cache functions
            # Fall back to default timeout on any expected error
            pass
    
    # Use provided timeout, or SIMPLE timeout for simple queries, or default
    if timeout is not None:
        effective_timeout = timeout
    elif is_simple:
        effective_timeout = SIMPLE_QUERY_ROUTING_TIMEOUT_SECONDS
    else:
        effective_timeout = QUERY_ROUTING_TIMEOUT_SECONDS

    try:
        # FIX 2: Wrap routing in timeout to prevent 46-50+ second delays
        # Offload the entire route_query call to a thread pool to avoid blocking
        # the asyncio event loop with CPU-bound safety validation operations
        plan = await asyncio.wait_for(
            loop.run_in_executor(executor, route_query, query, source, session_id),
            timeout=effective_timeout,
        )
        # Note: llm_mode is already set by route_query before returning
        return plan

    except asyncio.TimeoutError:
        # FIX 2: Return fallback plan on timeout instead of blocking forever
        logger.warning(
            f"[QueryRouter] Query routing timed out after {effective_timeout}s. "
            f"Returning fallback plan for source={source}"
        )
        return _create_fallback_plan(original_query, source, session_id, timeout_exceeded=True)


def _create_fallback_plan(
    query: str,
    source: Literal["user", "agent", "arena"],
    session_id: Optional[str] = None,
    timeout_exceeded: bool = False,
) -> ProcessingPlan:
    """
    Create a minimal fallback processing plan when routing times out or fails.

    FIX 2: This function provides a safe fallback when query routing takes too long
    (46-50+ seconds observed in production). Instead of blocking, we return a
    minimal plan that routes directly to reasoning with reduced complexity.

    Args:
        query: The original query text
        source: Query source - "user", "agent", or "arena"
        session_id: Optional session identifier
        timeout_exceeded: Whether this fallback is due to timeout

    Returns:
        ProcessingPlan with minimal/safe defaults
    """
    query_id = f"q_fallback_{uuid.uuid4().hex[:FALLBACK_QUERY_ID_LENGTH]}"

    # Determine learning mode based on source
    if source == "user":
        learning_mode = LearningMode.USER_INTERACTION
        telemetry_category = "user_query"
    else:
        learning_mode = LearningMode.AI_INTERACTION
        telemetry_category = f"{source}_interaction"

    # Create minimal plan with safe defaults
    plan = ProcessingPlan(
        query_id=query_id,
        original_query=query,
        source=source,
        learning_mode=learning_mode,
        query_type=QueryType.GENERAL,  # Default to general for fallback
        complexity_score=FALLBACK_COMPLEXITY_SCORE,
        uncertainty_score=FALLBACK_UNCERTAINTY_SCORE,
        collaboration_needed=False,
        arena_participation=False,
        requires_governance=False,
        requires_audit=True,  # Always audit fallback routes
        telemetry_category=telemetry_category,
        telemetry_data={
            "session_id": session_id,
            "query_length": len(query) if query else 0,
            "word_count": len(query.split()) if query else 0,
            "source": source,
            "learning_mode": learning_mode.value,
            "fallback_routing": True,
            "timeout_exceeded": timeout_exceeded,
        },
    )

    # Create single reasoning task for fallback
    base_task_id = uuid.uuid4().hex[:8]
    plan.agent_tasks = [
        AgentTask(
            task_id=f"task_{base_task_id}_fallback",
            task_type="general_task",
            capability="reasoning",
            prompt=query if query else "",
            priority=1,
            timeout_seconds=FALLBACK_TASK_TIMEOUT_SECONDS,
            parameters={
                "query_type": "general",
                "is_primary": True,
                "source": source,
                "fallback_routing": True,
                "timeout_exceeded": timeout_exceeded,
            },
        )
    ]

    plan.detected_patterns.append("fallback_routing")
    if timeout_exceeded:
        plan.detected_patterns.append("routing_timeout")

    # ARCHITECTURE: Set LLM mode for fallback plan
    # Industry Standard: Even fallback plans have proper LLM mode
    plan.llm_mode = LLMMode.FORMAT_ONLY  # Safe default for fallback

    logger.info(
        f"[QueryRouter] Fallback plan created: {query_id}, source={source}, "
        f"tasks=1, timeout_exceeded={timeout_exceeded}, llm_mode={plan.llm_mode.value}"
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
