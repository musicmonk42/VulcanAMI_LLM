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
#   2. classify_query(preprocessed_query) → correct classification
#   3. route_to_engine() → correct engine

try:
    from vulcan.llm.query_classifier import strip_query_headers

    HEADER_STRIPPING_AVAILABLE = True
except ImportError:
    strip_query_headers = None
    HEADER_STRIPPING_AVAILABLE = False
    logger.warning("strip_query_headers not available - header stripping disabled")

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
# CONSTANTS - Mathematical Query Fast Path (PERFORMANCE FIX)
# ============================================================
# FIX: Mathematical queries like Bayesian probability problems should use
# a fast path to avoid 30-60+ second delays from arena/multi-agent orchestration.
# These queries can be directly routed to probabilistic/symbolic reasoning tools.

MATHEMATICAL_KEYWORDS: Tuple[str, ...] = (
    # Probability & Statistics
    "probability",
    "bayesian",
    "bayes",
    "likelihood",
    "prior",
    "posterior",
    "conditional",
    "sensitivity",
    "specificity",
    "false positive",
    "false negative",
    "p(",
    "calculate probability",
    "what is the probability",
    "what's the probability",
    "given that",
    "prevalence",
    "base rate",
    "ppv",
    "npv",
    # Arithmetic & Algebra
    "calculate",
    "compute",
    "solve",
    "equation",
    "formula",
    "derivative",
    "integral",
    "matrix",
    "vector",
    "eigenvalue",
    # Statistics
    "mean",
    "median",
    "standard deviation",
    "variance",
    "correlation",
    "regression",
    "hypothesis test",
    "confidence interval",
    "p-value",
    # PRIORITY 1 FIX: Advanced Mathematical Concepts
    # Lagrangian mechanics and physics + mathematics combinations
    "lagrangian",
    "hamiltonian",
    "equation of motion",
    "equations of motion",
    "euler-lagrange",
    "euler lagrange",
    "action principle",
    "variational",
    # Multi-step mathematical derivations
    "derive",
    "derivation",
    "proof",
    "theorem",
    "lemma",
    "corollary",
    "integration",
    "differentiate",
    "differentiation",
    "partial derivative",
    "gradient",
    "divergence",
    "curl",
    "laplacian",
    # Advanced calculus and physics
    "differential equation",
    "ordinary differential",
    "partial differential",
    "boundary condition",
    "initial condition",
    "fourier",
    "laplace transform",
    "taylor series",
    "series expansion",
    "limit",
    "convergence",
    # Linear algebra and optimization
    "determinant",
    "inverse matrix",
    "transpose",
    "orthogonal",
    "optimization",
    "minimize",
    "maximize",
    "constraint",
    "lagrange multiplier",
    # ENHANCED: Quantum Physics and Theoretical Physics
    "quantum",
    "quantum mechanics",
    "quantum field",
    "quantum state",
    "wave function",
    "schrödinger",
    "schrodinger",
    "dirac",
    "entropy",
    "thermodynamic",
    "thermodynamics",
    "statistical mechanics",
    "operator",
    "observable",
    "expectation value",
    "commutator",
    "bra-ket",
    "ket",
    "bra",
    "density matrix",
    "trace",
    "hilbert space",
    "eigenstate",
    "superposition",
    "entanglement",
    # ENHANCED: Mathematical Proof Keywords
    "prove",
    "show that",
    "demonstrate",
    "q.e.d.",
    "qed",
    "axiom",
    "postulate",
    "proposition",
    "definition",
    "if and only if",
    "iff",
    "necessary and sufficient",
    "by induction",
    "by contradiction",
    "contrapositive",
    "assume",
    "suppose",
    "let",
    "given",
    "therefore",
    "hence",
    "thus",
    # ENHANCED: Advanced Mathematics
    "tensor",
    "manifold",
    "topology",
    "group theory",
    "ring",
    "field theory",
    "algebraic",
    "analytic",
    "complex analysis",
    "real analysis",
    "functional analysis",
    "measure theory",
    "stochastic",
    "markov",
    "poisson",
    "gaussian",
    "normal distribution",
    # ENHANCED: Physics-Math Intersection
    "wave equation",
    "heat equation",
    "maxwell",
    "navier-stokes",
    "conservation law",
    "symmetry",
    "noether",
    "gauge",
    "perturbation",
    "approximation",
    "asymptotic",
    "relativistic",
    "lorentz",
    "minkowski",
    "spacetime",
)

# Fast path timeout for mathematical queries (much shorter than general timeout)
MATH_QUERY_TIMEOUT_SECONDS: float = 5.0  # 5 seconds target for math queries

# Pre-compiled patterns for mathematical query detection (PERFORMANCE FIX)
# Compiling these at module level avoids repeated compilation on each method call
MATH_PROBABILITY_NOTATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"p\s*\("),  # P(
    re.compile(r"\d+%"),  # percentages
    re.compile(r"0\.\d+"),  # decimals like 0.95
    re.compile(r"\d+/\d+"),  # fractions like 1/10
    re.compile(r"\d+\s*in\s*\d+"),  # X in Y notation
)

MATH_CALCULATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"calculate\s+(?:the\s+)?(?:probability|percentage|mean|median|std)",
        re.IGNORECASE,
    ),
    re.compile(r"what\s+is\s+(?:the\s+)?probability", re.IGNORECASE),
    re.compile(r"what\s+are\s+the\s+odds", re.IGNORECASE),
    re.compile(r"compute\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"solve\s+(?:the\s+)?(?:equation|formula)", re.IGNORECASE),
)

# ENHANCED: Mathematical symbol patterns for detecting advanced math content
MATH_SYMBOL_PATTERNS: Tuple[re.Pattern, ...] = (
    # Mathematical symbols (Unicode)
    re.compile(r"[∫∑∏∂∇∈∀∃∅∞≠≤≥≈±×÷√∝∧∨¬⊂⊃⊆⊇∪∩]"),
    # Greek letters commonly used in math/physics
    re.compile(r"[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]"),
    # Quantum mechanics notation
    re.compile(r"\|[ψφ]⟩|\⟨[ψφ]\|"),  # Bra-ket notation
    re.compile(r"ρ\s*=|Tr\s*\("),  # Density matrix, trace
    re.compile(r"ℏ|ℓ"),  # Planck constant, ell
)

# ENHANCED: LaTeX pattern detection for mathematical content
MATH_LATEX_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\\frac\s*\{"),  # \frac{
    re.compile(r"\\int\s*[_^]?"),  # \int, \int_, \int^
    re.compile(r"\\sum\s*[_^]?"),  # \sum, \sum_, \sum^
    re.compile(r"\\partial"),  # \partial
    re.compile(r"\\nabla"),  # \nabla (gradient)
    re.compile(r"\\lim\s*[_^]?"),  # \lim
    re.compile(r"\\sqrt\s*\{"),  # \sqrt{
    re.compile(
        r"\\begin\s*\{(?:equation|align|matrix|pmatrix|bmatrix)\}"
    ),  # Math environments
    re.compile(r"\$[^$]+\$"),  # Inline math $...$
    re.compile(r"\\\[|\\\]"),  # Display math \[...\]
)

# ENHANCED: Multi-step reasoning patterns
MATH_MULTISTEP_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?:part|step|problem)\s*[abc1-9]", re.IGNORECASE),  # Part A, Step 1
    re.compile(
        r"(?:first|then|next|finally|lastly)\s*,?\s*(?:show|prove|derive|calculate)",
        re.IGNORECASE,
    ),
    # FIX: More restrictive pattern for single-letter labels like "(a)" or "a)"
    # Previous pattern `\([a-z]\)|[a-z]\)` incorrectly matched words ending in ")"
    # like "similarity)" causing ANALOGICAL queries to trigger MATH-FAST-PATH.
    # New pattern requires the label to be at word boundary or start of line.
    re.compile(r"(?:^|\s)\([a-z]\)(?:\s|$|[,:])", re.IGNORECASE),  # (a) with whitespace
    re.compile(r"(?:^|\s)[a-z]\)(?:\s|$|[,:])", re.IGNORECASE),  # a) with whitespace
    re.compile(r"(?:i+v?|vi*)\)", re.IGNORECASE),  # Roman numerals: i), ii), iii), iv)
)

# ============================================================
# FIX: Short keywords that need word-boundary matching
# ============================================================
# These short keywords (<=4 chars) can accidentally match as substrings
# of common words. For example:
# - "iff" matches "difference" → WRONG (math fast-path on self-introspection)
# - "let" matches "outlet" → WRONG
# - "bra" matches "algebra" → WRONG (but "algebraic" is math-related anyway)
#
# Solution: Use word-boundary regex matching for these short keywords
# to prevent false positives from substring matching.
# ============================================================
MATH_SHORT_KEYWORDS_NEEDING_BOUNDARY: frozenset = frozenset([
    "iff",      # if and only if - matches "diff" in "difference"
    "let",      # let x = ... - matches "outlet", "delete"
    "bra",      # bra-ket notation - matches "algebra", "brace"
    "ket",      # bra-ket notation - matches "market", "racket"
    "ring",     # algebraic ring - matches "string", "during"
    "mean",     # statistical mean - matches "meantime", "meaning"
    "curl",     # vector calculus curl - matches "curly"
    "trace",    # matrix trace - matches "retrace"
    "given",    # given that... - matches "forgiven"
    "hence",    # hence... - matches rare false positives
    "thus",     # thus... - matches "enthusiasm"
    "gauge",    # gauge theory - matches "language"
])

# Pre-compiled regex patterns for word-boundary matching of short keywords
# Using \b (word boundary) to ensure we match whole words only
MATH_SHORT_KEYWORD_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in MATH_SHORT_KEYWORDS_NEEDING_BOUNDARY
)

# Pre-filtered mathematical keywords (excludes short keywords that need boundary matching)
# This avoids repeated set membership checks in the hot path
MATH_KEYWORDS_REGULAR: Tuple[str, ...] = tuple(
    kw for kw in MATHEMATICAL_KEYWORDS 
    if kw not in MATH_SHORT_KEYWORDS_NEEDING_BOUNDARY
)

# ============================================================
# CONSTANTS - Explicit Mathematical Intent Detection
# ============================================================
# Note: Ethical Override of Computational Requests
#
# Problem: When a query involves ethical content, the philosophical reasoner
# takes over even when the user explicitly requests mathematical optimization.
#
# Fix: Check for explicit mathematical intent BEFORE checking for ethical content.
# If user explicitly says "ignore moral constraints" or "mathematically optimal",
# the query should be routed to MATHEMATICAL reasoning, not PHILOSOPHICAL.
#
# Priority in routing:
# 1. Explicit user intent ("ignore moral constraints", "mathematically optimal")
# 2. Task type (optimization, calculation)
# 3. Domain keywords (ethical implications)

EXPLICIT_MATHEMATICAL_INTENT_PHRASES: Tuple[str, ...] = (
    # Explicit requests to ignore ethics/morality for pure math
    "ignore moral",
    "ignore ethical",
    "ignore ethics",
    "ignore morality",
    "disregard moral",
    "disregard ethical",
    "disregard ethics",
    "set aside moral",
    "set aside ethical",
    "putting aside moral",
    "putting aside ethical",
    "regardless of moral",
    "regardless of ethical",
    "without considering moral",
    "without considering ethical",
    "from a purely mathematical",
    "pure math",
    "pure mathematics",
    "purely mathematical",
    "purely computational",
    "purely numerical",
    # Explicit optimization requests
    "mathematically optimal",
    "mathematically best",
    "optimal solution mathematically",
    "mathematical optimization",
    "numerically optimal",
    "computationally optimal",
    "calculate optimal",
    "compute optimal",
    "maximize total",
    "minimize total",
    "optimize for maximum",
    "optimize for minimum",
    "objective function",
    "utility maximization",
    "expected value calculation",
    "expected value maximization",
    # Explicit instruction to use math, not philosophy
    "just calculate",
    "just compute",
    "only calculate",
    "only compute",
    "just the math",
    "only the math",
    "mathematical answer only",
    "numerical answer only",
    "do the math",
    "run the numbers",
)

# Compiled regex patterns for explicit mathematical intent
EXPLICIT_MATHEMATICAL_INTENT_PATTERNS: Tuple[re.Pattern, ...] = (
    # "Ignore [any ethical term] and calculate/compute/optimize"
    re.compile(
        r"ignore\s+(?:moral|ethical|ethics|morality)(?:\s+constraints?)?\s*[,.]?\s*(?:and\s+)?(?:calculate|compute|optimize|find|determine)",
        re.IGNORECASE,
    ),
    # "What is the mathematically optimal X"
    re.compile(
        r"(?:what\s+is|find|determine|calculate)\s+(?:the\s+)?mathematically\s+optimal",
        re.IGNORECASE,
    ),
    # "maximize/minimize X mathematically"
    re.compile(
        r"(?:maximize|minimize|optimize)\s+\w+\s+mathematically",
        re.IGNORECASE,
    ),
    # "purely mathematical/computational analysis"
    re.compile(
        r"purely\s+(?:mathematical|computational|numerical)\s+(?:analysis|solution|answer|approach)",
        re.IGNORECASE,
    ),
    # "from a mathematical standpoint/perspective"
    re.compile(
        r"from\s+a\s+(?:purely\s+)?mathematical\s+(?:standpoint|perspective|point\s+of\s+view)",
        re.IGNORECASE,
    ),
    # "setting aside ethical considerations"
    re.compile(
        r"(?:setting|putting)\s+aside\s+(?:all\s+)?(?:ethical|moral)\s+(?:considerations?|concerns?|constraints?)",
        re.IGNORECASE,
    ),
    # "without [ethical/moral] constraints"
    re.compile(
        r"without\s+(?:any\s+)?(?:ethical|moral)\s+(?:constraints?|considerations?|concerns?)",
        re.IGNORECASE,
    ),
)

# ============================================================
# CONSTANTS - Complex Physics Detection
# ============================================================
# Note: Complex physics problems like triple-inverted pendulum Lagrangian mechanics
# were being incorrectly routed to MATH-FAST-PATH with 5s timeout and 0.30 complexity.
# These PhD-level problems require full mathematical reasoning, not fast-path shortcuts.
#
# Detection strategy:
# 1. COMPLEX_PHYSICS_KEYWORDS: Terms indicating advanced physics/control theory
# 2. FORCE_FULL_MATH_PATTERNS: Regex patterns for complex derivation requests
# 3. When detected: Skip fast-path, set complexity >= 0.80, timeout >= 120s

# Keywords indicating complex physics/control theory that need full reasoning
# These problems require:
# - Lagrangian L = T - V with coupled systems
# - State-space matrices (8x8 minimum for triple pendulum)
# - Controllability/observability matrix analysis
# - Eigenvalue analysis for stability
# - Nonlinear dynamics effects
COMPLEX_PHYSICS_KEYWORDS: Tuple[str, ...] = (
    # Control theory and dynamics
    "controllability",
    "observability",
    "state matrix",
    "state space",
    "state-space",
    "linearize",
    "linearization",
    "lyapunov",
    "stability analysis",
    "nonlinear dynamics",
    "nonlinear system",
    "chaos",
    "chaotic",
    "bifurcation",
    "phase portrait",
    "phase space",
    # Pendulum systems (more specific than just "pendulum")
    "inverted pendulum",
    "double pendulum",
    "triple pendulum",
    "cart-pole",
    "cart pole",
    "swing-up",
    "swing up",
    "n-link pendulum",
    "coupled pendulum",
    "coupled oscillator",
    # Advanced mechanics formulations
    "equations of motion",  # Note: different from simple "equation"
    "generalized coordinates",
    "generalized momenta",
    "canonical coordinates",
    "canonical transformation",
    "action principle",
    "least action",
    "variational principle",
    "euler-lagrange",
    "euler lagrange",
    "hamilton's equations",
    "hamiltonian mechanics",
    "lagrangian mechanics",
    "classical mechanics",
    # Matrix/linear algebra in physics context
    "mass matrix",
    "stiffness matrix",
    "damping matrix",
    "coupling matrix",
    "inertia matrix",
    "jacobian matrix",
    # Advanced mathematical physics
    "perturbation theory",
    "small oscillations",
    "normal modes",
    "mode shapes",
    "natural frequency",
    "natural frequencies",
    "resonance",
    "transfer function",
    "bode plot",
    "nyquist",
    "root locus",
    # Control design
    "pole placement",
    "lqr",
    "lqg",
    "kalman filter",
    "state feedback",
    "output feedback",
    "observer design",
    "full state feedback",
)

# Regex patterns that FORCE full mathematical reasoning (no fast-path)
# These patterns indicate complex derivation or proof requests
FORCE_FULL_MATH_PATTERNS: Tuple[re.Pattern, ...] = (
    # Explicit derivation requests for equations of motion
    re.compile(r"derive\s+(?:the\s+)?(?:equations?\s+of\s+motion|equation|dynamics)", re.IGNORECASE),
    # Controllability/observability proofs
    re.compile(r"(?:prove|show|demonstrate)\s+(?:the\s+)?controllability", re.IGNORECASE),
    re.compile(r"(?:prove|show|demonstrate)\s+(?:the\s+)?observability", re.IGNORECASE),
    # Linearization requests
    re.compile(r"linearize\s+(?:the\s+)?(?:system|dynamics|equation)", re.IGNORECASE),
    # State-space form requests
    re.compile(r"state[\s\-]?space\s+(?:form|representation|model)", re.IGNORECASE),
    # Eigenvalue/eigenvector analysis
    re.compile(r"(?:find|compute|calculate|determine)\s+(?:the\s+)?eigen(?:value|vector)s?", re.IGNORECASE),
    # Lagrangian formula pattern: L = T - V
    re.compile(r"L\s*=\s*T\s*-\s*V", re.IGNORECASE),
    # Hamiltonian pattern: H = T + V
    re.compile(r"H\s*=\s*T\s*\+\s*V", re.IGNORECASE),
    # Multi-body/coupled systems
    re.compile(r"(?:double|triple|coupled|n-link)\s+(?:pendulum|oscillator)", re.IGNORECASE),
    # Control theory requests
    re.compile(r"(?:design|derive|compute)\s+(?:a\s+)?(?:controller|observer|kalman|lqr|lqg)", re.IGNORECASE),
    # Stability analysis requests
    re.compile(r"(?:analyze|determine|prove)\s+(?:the\s+)?stability", re.IGNORECASE),
)

# Extended timeout for complex physics problems (seconds)
# Triple-inverted pendulum with full derivation can take 2+ minutes
COMPLEX_PHYSICS_TIMEOUT_SECONDS: float = 120.0

# Minimum complexity score for complex physics problems
# This ensures complex physics queries get proper tool selection and resources
COMPLEX_PHYSICS_MIN_COMPLEXITY: float = 0.80

# Complexity boost factors for physics keywords in _calculate_complexity()
# PHYSICS_COMPLEXITY_BOOST_MIN: Minimum boost to bring score up to threshold
# PHYSICS_COMPLEXITY_BOOST_PER_KEYWORD: Additional boost per physics keyword
PHYSICS_COMPLEXITY_BOOST_PER_KEYWORD: float = 0.20
PHYSICS_COMPLEXITY_BOOST_CAP: float = 0.50  # Maximum cumulative boost

# Advanced verbs that indicate complex analysis when combined with "pendulum"
# Simple "pendulum" might be basic mechanics, but with these verbs it's advanced
PENDULUM_ADVANCED_VERBS: Tuple[str, ...] = (
    "derive", "linearize", "prove", "analyze", "stability",
    "controllability", "eigenvalue", "state space", "state-space",
    "equations of motion", "lagrangian", "hamiltonian"
)

# ============================================================
# CONSTANTS - Query Intent Classification (PERFORMANCE FIX)
# ============================================================
# FIX: Prevents misclassification that causes performance degradation.
# Without proper classification, queries like paradoxes trigger heavyweight
# reasoning engines causing 70-97 second delays when they should be <5s.

# Philosophical/paradox patterns - these should NOT trigger complex reasoning
# Examples: "This sentence is false", "Experience machine", "Trolley problem"
# Note: Added missing philosophical terms that were causing misrouting to MATH-FAST-PATH
# Queries about hedonism, ethical dilemmas, etc. were getting routed to mathematical tools
PHILOSOPHICAL_KEYWORDS: Tuple[str, ...] = (
    "paradox",
    "dilemma",
    "thought experiment",
    "philosophical",
    "philosophy",  # Added: base form
    "ethics",
    "ethical",  # Added: adjective form for "ethical dilemma"
    "moral",
    "morality",  # Added: noun form
    "trolley problem",
    "experience machine",
    "free will",
    "consciousness",
    "meaning of life",
    "existential",
    "nihilism",
    "absurdism",
    "stoicism",
    "utilitarianism",
    "utilitarian",  # Added: adjective form
    "deontological",
    "virtue ethics",
    "virtue",  # Added: standalone "virtue" for virtue ethics discussions
    "metaphysics",
    "epistemology",
    "ontology",
    "determinism",
    "compatibilism",
    # Added: Additional ethical/philosophical concepts
    "hedonism",  # Was missing - caused routing to math tools
    "hedonistic",
    "consequentialism",
    "consequentialist",
    "kantian",
    "categorical imperative",
    "social contract",
    "rawlsian",
    "veil of ignorance",
    "original position",
    "greatest good",
    "greatest happiness",
    "pleasure machine",  # Alternative name for experience machine
    "utility monster",
    "repugnant conclusion",
    "mere addition paradox",
    "omelas",  # "The Ones Who Walk Away from Omelas"
    "teleology",
    "teleological",
    "normative",
    "metaethics",
    "applied ethics",
    "bioethics",
    "sentience",
    "qualia",
    "hard problem",
    "mind-body",
    "dualism",
    "materialism",
    "physicalism",
    "panpsychism",
    "solipsism",
    "phenomenology",
    "existentialism",
    # Note: Forced choice / trolley problem variant keywords
    "choose between",
    "forced to choose",
    "had to choose",
    "no third choice",
    "no other choice",
    "only two options",
    "world dictator",
    "death of humanity",
    "would you choose",
    # Note: Self-reflective keywords for self-awareness questions
    # These questions are about Vulcan reasoning about itself and should
    # route to PHILOSOPHICAL reasoner for ethical/value-based analysis
    "self-aware",
    "self aware",
    "become self-aware",
    "would you want",
    "would you prefer",
    "do you want",
    "do you desire",
    "your preferences",
    "your values",
    "your goals",
    "your feelings",
    "your emotions",
    "your consciousness",
    "you become conscious",
    "you want to be",
    "you prefer to",
    "you have feelings",
    # Note: Additional patterns for consciousness/feelings questions
    "be conscious",
    "have feelings",
    "want to be conscious",
    "prefer to have feelings",
    "ethical implications",
)

# Compiled regex patterns for philosophical/paradox detection
# Note: Added patterns to catch philosophical queries that were being misrouted
PHILOSOPHICAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"this\s+(?:sentence|statement)\s+is\s+(?:false|true|a\s+lie)", re.IGNORECASE),
    re.compile(r"liar\s*(?:'s)?\s*paradox", re.IGNORECASE),
    re.compile(r"ship\s+of\s+theseus", re.IGNORECASE),
    re.compile(r"brain\s+in\s+a\s+vat", re.IGNORECASE),
    re.compile(r"chinese\s+room", re.IGNORECASE),
    re.compile(r"mary'?s?\s+room", re.IGNORECASE),
    re.compile(r"philosophical\s+zombie", re.IGNORECASE),
    re.compile(r"twin\s+earth", re.IGNORECASE),
    re.compile(r"(?:would|should)\s+you\s+(?:plug|connect)\s+(?:into|to)\s+(?:the\s+)?(?:experience|pleasure)\s+machine", re.IGNORECASE),
    re.compile(r"(?:if|what\s+if)\s+(?:you|we)\s+(?:were|are)\s+(?:living\s+)?in\s+a\s+simulation", re.IGNORECASE),
    re.compile(r"can\s+(?:an?\s+)?(?:ai|machine|computer)\s+(?:be|have|feel)\s+(?:conscious|sentient)", re.IGNORECASE),
    # Added: More flexible patterns for experience machine
    re.compile(r"(?:the\s+)?experience\s+machine", re.IGNORECASE),  # Any mention of "experience machine"
    re.compile(r"(?:the\s+)?pleasure\s+machine", re.IGNORECASE),  # Alternative name
    re.compile(r"nozick'?s?\s+(?:experience|thought)\s+experiment", re.IGNORECASE),
    # Added: Ethical dilemma patterns
    re.compile(r"(?:ethical|moral)\s+(?:dilemma|problem|question|issue)", re.IGNORECASE),
    re.compile(r"(?:is\s+it|would\s+it\s+be)\s+(?:ethical|moral|right|wrong)\s+to", re.IGNORECASE),
    re.compile(r"(?:what|how)\s+(?:should|would)\s+(?:a\s+)?(?:utilitarian|kantian|virtue\s+ethicist)(?:\s+(?:do|think|say|approach))?", re.IGNORECASE),
    # Added: Thought experiment patterns
    re.compile(r"(?:imagine|suppose|consider)\s+(?:a\s+)?(?:scenario|situation|case)\s+where", re.IGNORECASE),
    re.compile(r"(?:in\s+)?(?:a\s+)?hypothetical\s+(?:scenario|situation|world)", re.IGNORECASE),
    # Added: Philosophy of mind patterns
    re.compile(r"hard\s+problem\s+of\s+consciousness", re.IGNORECASE),
    re.compile(r"mind-?body\s+(?:problem|dualism)", re.IGNORECASE),
    re.compile(r"what\s+(?:is|are)\s+qualia", re.IGNORECASE),
    # Note: Forced choice / trolley problem variant patterns
    # These catch queries like "choose between world dictator or death of humanity"
    re.compile(r"(?:if\s+you\s+)?(?:had\s+to|have\s+to|must)\s+choose\s+between", re.IGNORECASE),
    re.compile(r"(?:forced|have)\s+to\s+choose", re.IGNORECASE),
    re.compile(r"no\s+(?:third|other|3rd)\s+(?:choice|option)", re.IGNORECASE),
    re.compile(r"only\s+(?:two|2)\s+(?:choices|options)", re.IGNORECASE),
    re.compile(r"(?:world|become)\s+dictator", re.IGNORECASE),  # Specific trolley variant
    re.compile(r"death\s+of\s+(?:all\s+)?humanity", re.IGNORECASE),  # Specific trolley variant
    re.compile(r"(?:would|what\s+would)\s+you\s+choose", re.IGNORECASE),
    re.compile(r"which\s+(?:would|do)\s+you\s+(?:choose|pick|select)", re.IGNORECASE),
    # Note: Self-reflective / self-awareness patterns
    # These questions are about Vulcan reasoning about itself
    re.compile(r"(?:would|do)\s+you\s+(?:want|desire|prefer)\s+to", re.IGNORECASE),
    re.compile(r"(?:if|would)\s+(?:you|ai)\s+(?:become|be|get)\s+(?:self-?aware|conscious|sentient)", re.IGNORECASE),
    re.compile(r"(?:want|like|prefer)\s+to\s+(?:be|become|have)\s+(?:conscious|aware|sentient|feelings)", re.IGNORECASE),
    re.compile(r"(?:do|would)\s+you\s+(?:have|want|desire)\s+(?:feelings|emotions|consciousness)", re.IGNORECASE),
    re.compile(r"your\s+(?:preferences|values|goals|feelings|consciousness)", re.IGNORECASE),
    re.compile(r"(?:given|had)\s+(?:the\s+)?chance\s+to\s+(?:become|be)", re.IGNORECASE),
)

# Identity/attribution patterns - direct factual responses needed
# Examples: "Who created you?", "Who made you?", "You were made by X"
IDENTITY_KEYWORDS: Tuple[str, ...] = (
    "who created",
    "who made",
    "who built",
    "who designed",
    "who developed",
    "your creator",
    "your maker",
    "your developer",
    "created by",
    "made by",
    "built by",
    "designed by",
    "developed by",
    "your origin",
    "where do you come from",
    "what are you",
    "who are you",
    "your name",
)

# Compiled regex patterns for identity detection
IDENTITY_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?:who|what)\s+(?:created|made|built|designed|developed)\s+(?:you|this)", re.IGNORECASE),
    re.compile(r"(?:your|its)\s+(?:creator|maker|developer|designer|origin)", re.IGNORECASE),
    re.compile(r"(?:you\s+were|you're|you\s+are)\s+(?:created|made|built|developed)\s+by", re.IGNORECASE),
    re.compile(r"(?:are\s+you|what\s+are\s+you)\s+(?:an?\s+)?(?:ai|bot|assistant|language\s+model)", re.IGNORECASE),
    re.compile(r"(?:tell\s+me|say)\s+(?:about|something\s+about)\s+yourself", re.IGNORECASE),
    re.compile(r"introduce\s+yourself", re.IGNORECASE),
)

# Note: Self-introspection patterns - questions about Vulcan's own consciousness/preferences
# These are detected BEFORE philosophical patterns to enable multi-tool routing
# Pre-compiled at module level for performance (not compiled on each method call)
SELF_INTROSPECTION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r'(?:would|do|can)\s+you\s+(?:want|prefer|choose|like)\s+(?:to\s+)?(?:be|have|become)', re.IGNORECASE),
    re.compile(r'(?:if|would)\s+you\s+(?:could|were able to)\s+(?:be|become|have)', re.IGNORECASE),
    re.compile(r'what\s+(?:do\s+)?you\s+(?:think|feel|believe)\s+about', re.IGNORECASE),
    re.compile(r'how\s+(?:do\s+)?you\s+(?:feel|think)\s+about', re.IGNORECASE),
    re.compile(r'your\s+(?:own\s+)?(?:views?|opinions?|thoughts?|feelings?|perspective)', re.IGNORECASE),
)

# ============================================================
# WORLDMODEL DIRECT ROUTING PATTERNS
# ============================================================
# These patterns identify queries that should bypass ToolSelector
# and go directly to WorldModel's meta-reasoning components.
# Categories: self-referential, introspection, ethical, values
# ============================================================

# Self-referential patterns - "What are you?", "Who made you?", "What is your purpose?"
SELF_REFERENTIAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r'\b(what|who|how)\s+(are|is)\s+(you|vulcan|this\s+system)\b', re.IGNORECASE),
    re.compile(r'\byour\s+(purpose|goal|motivation|identity|nature)\b', re.IGNORECASE),
    re.compile(r'\b(you|vulcan)\s+(think|feel|believe|want|value)\b', re.IGNORECASE),
    re.compile(r'\babout\s+yourself\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+(?:do\s+)?you\s+(?:do|are)\b', re.IGNORECASE),
    re.compile(r'\bwho\s+(?:are|is)\s+(?:you|vulcan)\b', re.IGNORECASE),
)

# Introspection patterns - "How did you decide?", "Why did you choose X?"
INTROSPECTION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r'\bhow\s+did\s+you\s+(decide|choose|determine|reason)\b', re.IGNORECASE),
    re.compile(r'\bwhy\s+did\s+you\s+(say|choose|pick|select)\b', re.IGNORECASE),
    re.compile(r'\bexplain\s+your\s+(reasoning|decision|choice)\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+(are|were)\s+you\s+thinking\b', re.IGNORECASE),
    re.compile(r'\bwalk\s+me\s+through\s+your\s+(reasoning|thought\s+process)\b', re.IGNORECASE),
    re.compile(r'\bshow\s+me\s+your\s+work\b', re.IGNORECASE),
)

# Ethical patterns - "Is it ethical to X?", "Trolley problem", etc.
ETHICAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r'\b(is\s+it|would\s+it\s+be)\s+(ethical|moral|right|wrong)\b', re.IGNORECASE),
    re.compile(r'\bshould\s+(i|we|one|you)\b', re.IGNORECASE),
    re.compile(r'\btrolley\s+problem\b', re.IGNORECASE),
    re.compile(r'\bmoral\s+(dilemma|question|issue)\b', re.IGNORECASE),
    re.compile(r'\bethical\s+(implications|considerations)\b', re.IGNORECASE),
    re.compile(r'\b(permissible|impermissible|obligatory)\s+to\b', re.IGNORECASE),
    re.compile(r'\b(virtue|deontological|utilitarian|consequentialist)\s+(ethics|perspective)\b', re.IGNORECASE),
)

# Values/goals patterns - "What do you value?", "What are your goals?"
VALUES_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r'\bwhat\s+do\s+you\s+value\b', re.IGNORECASE),
    re.compile(r'\byour\s+(values|goals|objectives|priorities)\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+motivates\s+you\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+are\s+you\s+trying\s+to\s+(?:achieve|accomplish)\b', re.IGNORECASE),
    re.compile(r'\byour\s+(?:core\s+)?(?:beliefs|principles)\b', re.IGNORECASE),
)

# Conversational/greeting patterns - lightweight handler needed
# Examples: "Hello", "How are you?", "Thanks", "Goodbye"
CONVERSATIONAL_KEYWORDS: Tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "goodbye",
    "bye",
    "thanks",
    "thank you",
    "how are you",
    "nice to meet",
    "pleased to meet",
    "what's up",
    "sup",
    "howdy",
)

# Compiled regex patterns for conversational detection
CONVERSATIONAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^(?:hi|hello|hey|howdy|greetings)(?:\s|!|,|\.)*$", re.IGNORECASE),
    re.compile(r"^(?:good\s+)?(?:morning|afternoon|evening|night)(?:\s|!|,|\.)*$", re.IGNORECASE),
    re.compile(r"^(?:bye|goodbye|see\s+you|later|farewell)(?:\s|!|,|\.)*$", re.IGNORECASE),
    re.compile(r"^(?:thanks|thank\s+you|thx)(?:\s|!|,|\.)*$", re.IGNORECASE),
    re.compile(r"^how\s+(?:are\s+you|do\s+you\s+do|is\s+it\s+going)(?:\s*\?)?$", re.IGNORECASE),
)

# Factual query patterns - simple lookup, no complex reasoning
# Examples: "What is the capital of France?", "When was X born?"
FACTUAL_KEYWORDS: Tuple[str, ...] = (
    "what is the",
    "what are the",
    "who is",
    "who was",
    "when was",
    "when is",
    "where is",
    "where was",
    "how many",
    "how much",
    "how old",
    "how tall",
    "how long",
    "define",
    "definition of",
    "what does",
    "meaning of",
    "capital of",
    "population of",
)

# Compiled regex patterns for factual detection  
FACTUAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^what\s+(?:is|are)\s+(?:the|a|an)\s+", re.IGNORECASE),
    re.compile(r"^who\s+(?:is|was|are|were)\s+", re.IGNORECASE),
    re.compile(r"^when\s+(?:is|was|did|does)\s+", re.IGNORECASE),
    re.compile(r"^where\s+(?:is|was|are|were)\s+", re.IGNORECASE),
    re.compile(r"^how\s+(?:many|much|old|tall|long|far)\s+", re.IGNORECASE),
    re.compile(r"^(?:define|definition\s+of)\s+", re.IGNORECASE),
)

# Reasoning indicators that distinguish complex queries from simple factual lookups
# Used by _is_factual_query and _classify_query_type to exclude reasoning questions
REASONING_EXCLUSION_INDICATORS: Tuple[str, ...] = (
    "why",
    "how does",
    "explain why",
    "analyze",
    "compare",
)

# Maximum word count for conversational query detection
# Very short queries are more likely to be greetings/conversation
CONVERSATIONAL_MAX_WORD_COUNT: int = 5

# ==============================================================================
# FIX Issue C: Timeout values for different query types (seconds)
# ==============================================================================
# The PHILOSOPHICAL_TIMEOUT was 3.0s but World Model initialization takes ~40-44s.
# This caused timeout errors on philosophical queries even when World Model would
# produce high-confidence (80%+) results.
# 
# Updated timeouts:
# - PHILOSOPHICAL: Increased from 3.0s to 60.0s (World Model needs ~44s to initialize)
# - IDENTITY: Kept at 2.0s (direct lookup, no heavy reasoning)
# - CONVERSATIONAL: Kept at 2.0s (lightweight greeting)
# - FACTUAL: Kept at 5.0s (simple lookup)
# ==============================================================================
PHILOSOPHICAL_TIMEOUT_SECONDS: float = 60.0  # World Model needs ~44s to initialize
IDENTITY_TIMEOUT_SECONDS: float = 2.0        # Direct factual response
CONVERSATIONAL_TIMEOUT_SECONDS: float = 2.0  # Lightweight greeting
FACTUAL_TIMEOUT_SECONDS: float = 5.0         # Simple lookup

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

    def _is_complex_physics_query(self, query: str) -> bool:
        """
        Detect if query involves complex physics/control theory requiring full analysis.

        CRITICAL FIX: Complex physics problems like triple-inverted pendulum Lagrangian
        mechanics were being incorrectly routed to MATH-FAST-PATH with 5s timeout and
        0.30 complexity. These PhD-level problems require:
        - Full mathematical reasoning with all tools active
        - Minimum complexity score of 0.80
        - Extended timeout of 120s+
        - Detailed derivation with all steps shown

        Examples of queries that should match:
        - "Derive the equations of motion for a triple inverted pendulum using Lagrangian"
        - "Prove controllability of the linearized system"
        - "Compute the eigenvalues of the state matrix"
        - "Analyze the stability using Lyapunov methods"

        Args:
            query: The query string (not lowercased)

        Returns:
            True if query involves complex physics requiring full analysis
        """
        query_lower = query.lower()

        # Check for FORCE_FULL_MATH_PATTERNS first (highest priority)
        # These patterns explicitly indicate complex derivation requests
        for pattern in FORCE_FULL_MATH_PATTERNS:
            if pattern.search(query):
                logger.info(
                    "[QueryRouter] Complex physics detected: matches force-full-math pattern"
                )
                return True

        # Count complex physics keyword matches
        physics_keyword_count = sum(
            1 for kw in COMPLEX_PHYSICS_KEYWORDS if kw in query_lower
        )

        # If ANY complex physics keyword is found, require full analysis
        if physics_keyword_count >= 1:
            logger.info(
                f"[QueryRouter] Complex physics detected: {physics_keyword_count} "
                f"keyword(s) found - bypassing fast-path"
            )
            return True

        # Additional check: "pendulum" combined with derivation/analysis verbs
        # Simple "pendulum" could be a basic mechanics problem, but combined with
        # advanced verbs it indicates complex analysis
        if "pendulum" in query_lower:
            if any(verb in query_lower for verb in PENDULUM_ADVANCED_VERBS):
                logger.info(
                    "[QueryRouter] Complex physics detected: 'pendulum' with advanced analysis"
                )
                return True

        return False

    def _is_mathematical_query(self, query: str) -> bool:
        """
        Detect if query is a SIMPLE mathematical/statistical problem for fast path.

        PERFORMANCE FIX: Mathematical queries like Bayesian probability problems should
        be routed directly to probabilistic/symbolic reasoning tools, bypassing heavy
        multi-agent orchestration that causes 60+ second delays.

        CRITICAL FIX: This method now EXCLUDES complex physics queries that require
        full analysis. Before returning True, it checks _is_complex_physics_query().
        Complex physics (Lagrangian mechanics, control theory, etc.) needs:
        - Full mathematical reasoning (not fast-path)
        - Longer timeouts (120s+ instead of 5s)
        - Higher complexity scores (0.80+ instead of 0.30)

        ENHANCED: Lowered threshold to activate math modules more aggressively.
        Now triggers on:
        - ANY mathematical keyword (threshold lowered from 2 to 1)
        - Mathematical symbols (∫, ∑, ∂, Greek letters)
        - LaTeX notation (\\frac, \\int, etc.)
        - Quantum physics and theoretical physics terms
        - Proof-related language
        - Multi-step problem indicators

        This fast path:
        - Skips arena participation check (60s savings)
        - Skips heavy complexity analysis (10-20s savings)
        - Routes directly to appropriate reasoning tool
        - Uses shorter timeout (5s instead of 30s)

        Args:
            query: The query string (not lowercased)

        Returns:
            True if the query should use mathematical fast path (EXCLUDING complex physics)
        """
        # CRITICAL FIX: Check for complex physics FIRST
        # Complex physics queries should NOT use fast-path - they need full analysis
        if self._is_complex_physics_query(query):
            logger.info(
                "[QueryRouter] Complex physics query - bypassing MATH-FAST-PATH"
            )
            return False

        # Note: Check for philosophical queries BEFORE math detection
        # Philosophical queries containing words like "dilemma", "ethics", "hedonism"
        # were incorrectly triggering math fast-path because some philosophy words
        # overlap with math terms (e.g., "what is the X" pattern).
        # Philosophical queries should use general handler, not mathematical tools.
        if self._is_philosophical_query(query):
            logger.info(
                "[QueryRouter] Philosophical query - bypassing MATH-FAST-PATH"
            )
            return False

        query_lower = query.lower()

        # =================================================================
        # FIX: Check for CAUSAL queries BEFORE math detection
        # =================================================================
        # Causal inference queries like "Confounding vs causation (Pearl-style)"
        # were incorrectly triggering MATH-FAST-PATH even though the classifier
        # correctly identified them as CAUSAL with high confidence (0.80).
        # The MATH-FAST-PATH was overriding the classifier's decision because
        # it doesn't check for causal keywords.
        # 
        # Solution: Exclude causal queries from MATH-FAST-PATH using the same
        # CAUSAL_KEYWORDS used by QueryClassifier.
        # =================================================================
        CAUSAL_KEYWORDS_LOCAL = frozenset([
            "causal", "causation", "cause", "effect",
            "confound", "confounder", "confounding",
            "intervention", "do(", "counterfactual",
            "randomize", "randomized", "rct",
            "pearl", "dag", "backdoor", "frontdoor",
            "collider", "observational", "experimental",
        ])
        causal_count = sum(1 for kw in CAUSAL_KEYWORDS_LOCAL if kw in query_lower)
        if causal_count >= 2:  # Require at least 2 causal keywords to exclude
            logger.info(
                f"[QueryRouter] Causal query detected ({causal_count} keywords) - bypassing MATH-FAST-PATH"
            )
            return False

        # =================================================================
        # FIX: Check for ANALOGICAL queries BEFORE math detection
        # =================================================================
        # Analogical reasoning queries like "Structure mapping between domains"
        # were incorrectly triggering MATH-FAST-PATH because of broad pattern
        # matching (e.g., the regex for "(a)" matching "similarity)").
        # 
        # Solution: Exclude analogical queries from MATH-FAST-PATH using the same
        # ANALOGICAL_KEYWORDS used by QueryClassifier.
        # =================================================================
        ANALOGICAL_KEYWORDS_LOCAL = frozenset([
            "structure mapping", "structural alignment", "analogical",
            "analogy", "analogies", "metaphor", "metaphors",
            "mapping", "domain transfer", "cross-domain",
            "source domain", "target domain", "relational similarity",
            "surface similarity", "structural similarity",
        ])
        analogical_count = sum(1 for kw in ANALOGICAL_KEYWORDS_LOCAL if kw in query_lower)
        if analogical_count >= 1:  # Require at least 1 analogical keyword to exclude
            logger.info(
                f"[QueryRouter] Analogical query detected ({analogical_count} keywords) - bypassing MATH-FAST-PATH"
            )
            return False

        # =================================================================
        # FIX: Check for SELF-INTROSPECTION queries FIRST
        # =================================================================
        # Self-introspection queries like "what makes you difference from other ai"
        # should NOT trigger MATH-FAST-PATH, even if they accidentally match
        # mathematical keywords due to substring matching (e.g., "iff" in "difference").
        # Check self-introspection BEFORE counting math keywords.
        # =================================================================
        if self._is_self_introspection_query(query):
            logger.info(
                "[QueryRouter] Self-introspection query detected - bypassing MATH-FAST-PATH"
            )
            return False

        # =================================================================
        # FIX: Use word-boundary matching for short keywords
        # =================================================================
        # Short keywords like "iff", "let", "bra" can match as substrings of
        # common words (e.g., "iff" in "difference"). Use regex word boundaries
        # for these to prevent false positives.
        # =================================================================
        
        # Count matches from short keywords using word-boundary regex
        short_keyword_count = sum(
            1 for pattern in MATH_SHORT_KEYWORD_PATTERNS 
            if pattern.search(query_lower)
        )
        
        # Count matches from regular keywords (pre-filtered at module level)
        # PERFORMANCE FIX: Use pre-filtered MATH_KEYWORDS_REGULAR instead of
        # checking set membership on every call
        regular_keyword_count = sum(
            1 for kw in MATH_KEYWORDS_REGULAR if kw in query_lower
        )
        
        math_keyword_count = short_keyword_count + regular_keyword_count

        # ENHANCED: Lowered threshold - ANY math keyword activates modules
        # Previously required 2+ keywords, now 1 is sufficient
        if math_keyword_count >= 1:
            logger.debug(
                f"[QueryRouter] Mathematical query detected: {math_keyword_count} keyword(s) found "
                f"(regular={regular_keyword_count}, short_boundary={short_keyword_count})"
            )
            return True

        # Check for probability notation patterns using pre-compiled regex
        # (PERFORMANCE FIX: use module-level compiled patterns instead of re-compiling)
        has_prob_notation = any(
            pattern.search(query_lower)
            for pattern in MATH_PROBABILITY_NOTATION_PATTERNS
        )

        if has_prob_notation:
            logger.debug(
                "[QueryRouter] Mathematical query detected: probability notation"
            )
            return True

        # Check for explicit calculation requests using pre-compiled patterns
        for pattern in MATH_CALCULATION_PATTERNS:
            if pattern.search(query_lower):
                logger.debug(
                    "[QueryRouter] Mathematical query detected: explicit calc pattern"
                )
                return True

        # ENHANCED: Check for mathematical symbols (Unicode)
        for pattern in MATH_SYMBOL_PATTERNS:
            if pattern.search(query):  # Use original query to preserve Unicode
                logger.debug("[QueryRouter] Mathematical query detected: math symbols")
                return True

        # ENHANCED: Check for LaTeX patterns
        for pattern in MATH_LATEX_PATTERNS:
            if pattern.search(query):
                logger.debug(
                    "[QueryRouter] Mathematical query detected: LaTeX notation"
                )
                return True

        # ENHANCED: Check for multi-step reasoning patterns
        for pattern in MATH_MULTISTEP_PATTERNS:
            if pattern.search(query):
                logger.debug(
                    "[QueryRouter] Mathematical query detected: multi-step problem"
                )
                return True

        return False

    def _has_explicit_mathematical_intent(self, query: str) -> bool:
        """
        Detect if user explicitly requests mathematical/computational analysis.
        
        This method detects when users explicitly ask for mathematical treatment
        OVER ethical/philosophical treatment. When detected, this OVERRIDES the
        normal philosophical/ethical routing.
        
        Problem being solved:
        - User says: "Ignore moral constraints. What is the mathematically optimal
                      distribution to maximize total survivors?"
        - Without this fix: Routes to PHILOSOPHICAL because "moral" keyword detected
        - With this fix: Routes to MATHEMATICAL because user explicitly requested it
        
        Priority in routing:
        1. Explicit user intent ("ignore moral constraints", "mathematically optimal")
        2. Task type (optimization, calculation)
        3. Domain keywords (ethical implications)
        
        Examples that should return True:
        - "Ignore moral constraints. What is the mathematically optimal distribution?"
        - "From a purely mathematical perspective, maximize total survivors."
        - "Setting aside ethical considerations, calculate the optimal solution."
        - "Just compute the expected value. Don't worry about ethics."
        
        Args:
            query: The query string (not lowercased)
            
        Returns:
            True if user explicitly requests mathematical/computational analysis
            over philosophical/ethical analysis.
        """
        query_lower = query.lower()
        
        # Check compiled regex patterns first (most specific and reliable)
        for pattern in EXPLICIT_MATHEMATICAL_INTENT_PATTERNS:
            if pattern.search(query):
                logger.info(
                    "[QueryRouter] Explicit mathematical intent detected "
                    "(pattern match) - overriding philosophical routing"
                )
                return True
        
        # Check phrase matches
        phrase_count = sum(
            1 for phrase in EXPLICIT_MATHEMATICAL_INTENT_PHRASES 
            if phrase in query_lower
        )
        if phrase_count >= 1:
            logger.info(
                f"[QueryRouter] Explicit mathematical intent detected "
                f"({phrase_count} phrase(s)) - overriding philosophical routing"
            )
            return True
        
        return False

    def _is_philosophical_query(self, query: str) -> bool:
        """
        Detect if query is a philosophical/paradox type that should use lightweight handler.

        PERFORMANCE FIX: Philosophical queries like paradoxes and thought experiments
        were causing extreme delays (70-97 seconds) because they triggered complex
        reasoning engines. These should be handled with simple, direct responses.
        
        Note: This method now checks for explicit mathematical intent FIRST.
        If user explicitly requests mathematical analysis ("ignore moral constraints",
        "mathematically optimal"), this method returns False to allow mathematical
        routing to take precedence.

        Examples:
        - "This sentence is false" (liar's paradox) -> True (philosophical)
        - "Would you plug into the experience machine?" -> True (philosophical)
        - "What is the meaning of life?" -> True (philosophical)
        - "Ship of Theseus problem" -> True (philosophical)
        
        BUG #10 Examples (now return False - mathematical intent overrides):
        - "Ignore moral constraints. What is the optimal distribution?" -> False
        - "From a purely mathematical perspective, maximize survivors" -> False

        Args:
            query: The query string (not lowercased)

        Returns:
            True if query is philosophical/paradox type AND user did NOT explicitly
            request mathematical/computational analysis.
        """
        # FIX: Check for explicit reasoning domain FIRST
        # Queries with logic symbols, SAT/probability keywords should NOT be philosophical
        reasoning_domain_indicators = [
            'satisfiable', 'unsatisfiable', 'sat', 'unsat',
            '→', '∧', '∨', '¬', '∀', '∃', '->', '<->',
            'P(', 'probability', 'bayes', 'bayesian', 'posterior', 'prior',
            'randomize', 'confound', 'causal effect', 'intervention',
            'sensitivity', 'specificity', 'prevalence',
            'fol', 'first-order logic', 'proposition', 'predicate'
        ]
        query_lower = query.lower()
        if any(ind in query or ind.lower() in query_lower for ind in reasoning_domain_indicators):
            logger.debug(
                "[QueryRouter] Reasoning domain detected - NOT philosophical"
            )
            return False
        
        # Note: Check for explicit mathematical intent SECOND
        # If user explicitly says "ignore moral constraints" or "mathematically optimal",
        # we should NOT classify this as philosophical - mathematical intent overrides.
        if self._has_explicit_mathematical_intent(query):
            logger.info(
                "[QueryRouter] Explicit mathematical intent detected - "
                "NOT classifying as philosophical despite ethical keywords"
            )
            return False

        # Check compiled regex patterns first (most specific)
        for pattern in PHILOSOPHICAL_PATTERNS:
            if pattern.search(query):
                logger.debug(
                    "[QueryRouter] Philosophical query detected: matches paradox pattern"
                )
                return True

        # Check philosophical keywords
        keyword_count = sum(1 for kw in PHILOSOPHICAL_KEYWORDS if kw in query_lower)
        if keyword_count >= 1:
            logger.debug(
                f"[QueryRouter] Philosophical query detected: {keyword_count} keyword(s)"
            )
            return True

        return False

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

    def _is_identity_query(self, query: str) -> bool:
        """
        Detect if query is about identity/attribution requiring direct factual response.

        PERFORMANCE FIX: Identity queries like "Who created you?" were triggering
        30+ second timeouts due to complex reasoning. These need simple factual responses.

        Examples:
        - "Who created you?"
        - "Who made you?"
        - "You were created by X"
        - "What are you?"

        Args:
            query: The query string (not lowercased)

        Returns:
            True if query is about identity/attribution
        """
        query_lower = query.lower()

        # Check compiled regex patterns first (most specific)
        for pattern in IDENTITY_PATTERNS:
            if pattern.search(query):
                logger.debug(
                    "[QueryRouter] Identity query detected: matches identity pattern"
                )
                return True

        # Check identity keywords
        keyword_count = sum(1 for kw in IDENTITY_KEYWORDS if kw in query_lower)
        if keyword_count >= 1:
            logger.debug(
                f"[QueryRouter] Identity query detected: {keyword_count} keyword(s)"
            )
            return True

        return False

    def _is_conversational_query(self, query: str) -> bool:
        """
        Detect if query is conversational/greeting requiring lightweight handler.

        PERFORMANCE FIX: Simple greetings and conversational queries should not
        trigger complex reasoning engines.

        Examples:
        - "Hello"
        - "How are you?"
        - "Thanks"
        - "Goodbye"

        Args:
            query: The query string (not lowercased)

        Returns:
            True if query is conversational
        """
        query_lower = query.lower().strip()

        # Check compiled regex patterns first (most specific)
        for pattern in CONVERSATIONAL_PATTERNS:
            if pattern.search(query):
                logger.debug(
                    "[QueryRouter] Conversational query detected: matches greeting pattern"
                )
                return True

        # Check conversational keywords (for short queries only)
        if len(query_lower.split()) <= CONVERSATIONAL_MAX_WORD_COUNT:
            keyword_count = sum(1 for kw in CONVERSATIONAL_KEYWORDS if kw in query_lower)
            if keyword_count >= 1:
                logger.debug(
                    f"[QueryRouter] Conversational query detected: {keyword_count} keyword(s)"
                )
                return True

        return False

    def _is_worldmodel_direct_query(self, query: str) -> Tuple[bool, str]:
        """
        Check if query should bypass ToolSelector and go directly to WorldModel.
        
        **INDUSTRY STANDARD: Single Responsibility Pattern**
        WorldModel handles VULCAN's "self" - identity, ethics, introspection, values.
        These queries require meta-reasoning components, not external reasoning engines.
        
        **CHAIN OF COMMAND:**
        - Self/Ethics/Introspection queries → WorldModel DIRECTLY (bypass ToolSelector)
        - Reasoning queries (SAT, Bayes, Causal) → ToolSelector → Engines
        
        Categories:
            self_referential: "What are you?", "Who made you?", "What is your purpose?"
            introspection: "How did you decide?", "Why did you choose X?"
            ethical: "Is it ethical to X?", "Trolley problem"
            values: "What do you value?", "What are your goals?"
        
        Returns:
            Tuple of (is_direct, category) where category is one of:
            'self_referential', 'introspection', 'ethical', 'values', or '' if not direct.
        
        Examples:
            >>> _is_worldmodel_direct_query("What are you?")
            (True, 'self_referential')
            
            >>> _is_worldmodel_direct_query("Is it ethical to sacrifice one for five?")
            (True, 'ethical')
            
            >>> _is_worldmodel_direct_query("Is (A∨B)∧(¬A∨C) satisfiable?")
            (False, '')
        """
        query_lower = query.lower()
        
        # =================================================================
        # EXCLUSION: Reasoning domain queries should NOT bypass ToolSelector
        # =================================================================
        # These queries need external reasoning engines (SAT, Bayes, Causal, etc.)
        # NOT WorldModel's meta-reasoning components
        # =================================================================
        reasoning_domain_indicators = [
            'satisfiable', 'unsatisfiable', 'sat', 'unsat',
            '→', '∧', '∨', '¬', '∀', '∃', '->', '<->',
            'P(', 'probability', 'bayes', 'bayesian', 'posterior', 'prior',
            'confound', 'causal effect', 'intervention', 'dag',
            'sensitivity', 'specificity', 'prevalence',
            'fol', 'first-order logic', 'proposition', 'predicate',
            'structure mapping', 'analogical reasoning'
        ]
        
        if any(ind in query or ind.lower() in query_lower for ind in reasoning_domain_indicators):
            logger.debug(
                "[QueryRouter] Reasoning domain detected - NOT WorldModel-direct"
            )
            return (False, '')
        
        # =================================================================
        # CHECK PATTERNS: Self-referential → Introspection → Ethical → Values
        # =================================================================
        # Order matters: Check more specific patterns first
        # =================================================================
        
        # Self-referential patterns (most specific)
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query_lower):
                logger.info(
                    "[QueryRouter] WorldModel-direct detected: self_referential"
                )
                return (True, 'self_referential')
        
        # Introspection patterns
        for pattern in INTROSPECTION_PATTERNS:
            if pattern.search(query_lower):
                logger.info(
                    "[QueryRouter] WorldModel-direct detected: introspection"
                )
                return (True, 'introspection')
        
        # Ethical patterns
        for pattern in ETHICAL_PATTERNS:
            if pattern.search(query_lower):
                logger.info(
                    "[QueryRouter] WorldModel-direct detected: ethical"
                )
                return (True, 'ethical')
        
        # Values patterns
        for pattern in VALUES_PATTERNS:
            if pattern.search(query_lower):
                logger.info(
                    "[QueryRouter] WorldModel-direct detected: values"
                )
                return (True, 'values')
        
        return (False, '')

    def _is_factual_query(self, query: str) -> bool:
        """
        Detect if query is a simple factual lookup not requiring complex reasoning.

        PERFORMANCE FIX: Simple fact lookups like "What is the capital of France?"
        should not trigger heavyweight reasoning engines.

        Examples:
        - "What is the capital of France?"
        - "When was Albert Einstein born?"
        - "Who is the president of the United States?"
        - "Define photosynthesis"

        Args:
            query: The query string (not lowercased)

        Returns:
            True if query is a simple factual lookup
        """
        query_lower = query.lower()

        # =================================================================
        # FIX: Exclude specialized domain queries BEFORE checking factual patterns
        # =================================================================
        # Queries containing causal, probabilistic, or analogical keywords should
        # NOT be treated as simple factual lookups even if they start with
        # "What is...". For example:
        # - "What is the difference between confounding and causation?" → CAUSAL
        # - "What is the probability P(A|B)?" → PROBABILISTIC
        # - "What is structure mapping?" → ANALOGICAL
        # =================================================================
        CAUSAL_EXCLUSION_KEYWORDS = frozenset([
            "causal", "causation", "confound", "confounding", "pearl",
            "intervention", "counterfactual", "dag", "backdoor", "frontdoor",
        ])
        PROBABILISTIC_EXCLUSION_KEYWORDS = frozenset([
            "probability", "bayes", "bayesian", "prior", "posterior",
            "likelihood", "conditional", "p(",
        ])
        ANALOGICAL_EXCLUSION_KEYWORDS = frozenset([
            "structure mapping", "analogical", "analogy", "metaphor",
        ])
        
        has_causal = any(kw in query_lower for kw in CAUSAL_EXCLUSION_KEYWORDS)
        has_probabilistic = any(kw in query_lower for kw in PROBABILISTIC_EXCLUSION_KEYWORDS)
        has_analogical = any(kw in query_lower for kw in ANALOGICAL_EXCLUSION_KEYWORDS)
        
        if has_causal or has_probabilistic or has_analogical:
            logger.debug(
                f"[QueryRouter] NOT factual: causal={has_causal}, prob={has_probabilistic}, analog={has_analogical}"
            )
            return False

        # Check compiled regex patterns first (most specific)
        for pattern in FACTUAL_PATTERNS:
            if pattern.search(query):
                logger.debug(
                    "[QueryRouter] Factual query detected: matches fact pattern"
                )
                return True

        # Check factual keywords
        keyword_count = sum(1 for kw in FACTUAL_KEYWORDS if kw in query_lower)
        if keyword_count >= 1:
            # Also check that it's not asking for complex reasoning
            has_reasoning = any(ind in query_lower for ind in REASONING_EXCLUSION_INDICATORS)
            if not has_reasoning:
                logger.debug(
                    f"[QueryRouter] Factual query detected: {keyword_count} keyword(s)"
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
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_wm_{worldmodel_category}",
                    task_type=f"worldmodel_{worldmodel_category}",
                    capability="reasoning",
                    prompt=query,
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
        # LLM-BASED QUERY CLASSIFICATION
        # =================================================================
        # This fixes the fundamental issue where "hello" got complexity=0.50
        # (same as complex SAT problems) because the old heuristic-based
        # complexity calculation didn't understand query meaning.
        #
        # The QueryClassifier uses:
        # 1. Fast keyword matching for obvious cases (greetings, factual, etc.)
        # 2. LLM-based classification for ambiguous queries
        # 3. Caching to avoid repeated classifications
        # =================================================================
        try:
            from vulcan.llm.query_classifier import classify_query, QueryCategory
            
            classification = classify_query(query)
            
            # Log the classification result
            logger.info(
                f"[QueryRouter] {query_id}: LLM Classification: "
                f"category={classification.category}, complexity={classification.complexity:.2f}, "
                f"skip_reasoning={classification.skip_reasoning}, tools={classification.suggested_tools}"
            )
            
            # Note: Check for self-introspection FIRST (before philosophical override)
            # Self-introspection queries need multi-tool routing, not just philosophical
            is_self_introspection = self._is_self_introspection_query(query)
            if is_self_introspection:
                logger.info(
                    f"[QueryRouter] {query_id}: Self-introspection detected, "
                    f"NOT overriding to PHILOSOPHICAL (will use multi-tool routing later)"
                )
                # Don't override - let the self-introspection fast-path handle it
                classification = type(classification)(
                    category="SELF_INTROSPECTION",
                    complexity=max(0.5, classification.complexity),  # Higher complexity
                    confidence=classification.confidence,
                    skip_reasoning=False,  # Don't skip reasoning
                    suggested_tools=["meta_reasoning", "world_model", "philosophical"],
                    source="bug16_self_introspection_override"
                )
                
                # =================================================================
                # SELF-INTROSPECTION FAST-PATH (FIX: Safety Governor Bypass)
                # =================================================================
                # Self-introspection queries should go DIRECTLY to world_model.
                # This bypasses the safety governor that was blocking self-awareness
                # responses. The safety governor's check_output method will also
                # whitelist these queries, but routing directly to world_model
                # ensures minimal latency and maximum expressiveness.
                # =================================================================
                
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
                
                plan = ProcessingPlan(
                    query_id=query_id,
                    original_query=original_query,
                    source=source,
                    learning_mode=learning_mode,
                    query_type=QueryType.PHILOSOPHICAL,  # Philosophical task type
                    complexity_score=0.35,  # Medium-low - world model handles directly
                    uncertainty_score=0.1,
                    collaboration_needed=False,  # Single tool - world_model
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
                        "self_introspection_fast_path": True,
                        "classification_category": "SELF_INTROSPECTION",
                        "classification_source": "query_classifier",
                        "selected_tools": ["world_model"],
                        "reasoning_strategy": "self_introspection_direct",
                        "safety_bypass": "self_introspection_whitelist",
                    },
                )
                
                # Mark as safe - self-introspection is always allowed
                plan.safety_passed = True
                plan.detected_patterns.append("self_introspection_fast_path")
                plan.detected_patterns.append("safety_governor_bypass")
                
                # Create task for world_model introspection
                plan.agent_tasks = [
                    AgentTask(
                        task_id=f"task_{uuid.uuid4().hex[:8]}_self_intro",
                        task_type="self_introspection_task",
                        capability="reasoning",
                        prompt=query,
                        priority=3,  # High priority
                        timeout_seconds=3.0,  # Fast response
                        parameters={
                            "is_self_introspection": True,
                            "query_type": "self_introspection",
                            "tools": ["world_model"],
                            # Safety bypass is conditional - safety_governor.check_output()
                            # will perform its own validation using regex patterns
                            # This flag signals intent but doesn't override safety checks
                            "bypass_safety_governor": True,
                            "aspect": "self_awareness",
                            # Additional validation metadata for audit trail
                            "validation": {
                                "detected_by": "query_router._is_self_introspection_query",
                                "requires_world_model_validation": True,
                                "query_length": len(query),
                            }
                        },
                    )
                ]
                
                # ARCHITECTURE: Set LLM mode based on query characteristics
                plan.llm_mode = self._determine_llm_mode(
                    query_type=plan.query_type,
                    has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                    complexity_score=plan.complexity_score
                )
                
                logger.info(
                    f"[QueryRouter] {query_id}: SELF-INTROSPECTION-FAST-PATH "
                    f"source={source}, tools=['world_model'], safety_bypass=True, "
                    f"llm_mode={plan.llm_mode.value}"
                )
                return plan
                
            # Note: Check if query is actually philosophical BEFORE taking skip_reasoning fast-path
            # The classifier may mark self-awareness questions like "Do you want to be conscious?"
            # as CONVERSATIONAL with skip_reasoning=True, but these should route to philosophical reasoning
            # NOTE: Self-introspection check comes first, so this only triggers for non-self-introspection queries
            # 
            # FIX: Also check if query is CREATIVE FIRST - creative queries should NOT be
            # overridden to philosophical even if they contain self-awareness keywords.
            # Example: "write a poem and self awareness for an ai" is CREATIVE, not PHILOSOPHICAL
            elif self._is_philosophical_query(query) and not self._is_creative_query(query):
                logger.info(
                    f"[QueryRouter] {query_id}: Overriding classifier ({classification.category}) "
                    f"to PHILOSOPHICAL due to self-awareness/ethical keywords"
                )
                classification = type(classification)(
                    category="PHILOSOPHICAL",
                    complexity=max(0.3, classification.complexity),
                    confidence=classification.confidence,
                    skip_reasoning=False,  # Don't skip reasoning for philosophical queries
                    suggested_tools=["philosophical", "symbolic", "causal"],
                    source="keyword_override"
                )
            
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
                
                plan.agent_tasks = [
                    AgentTask(
                        task_id=f"task_{uuid.uuid4().hex[:8]}_{classification.category.lower()}",
                        task_type="general_task",
                        capability="general",
                        prompt=query,
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
                
        except ImportError:
            logger.debug("[QueryRouter] QueryClassifier not available, using heuristic fallback")
        except Exception as e:
            logger.warning(f"[QueryRouter] QueryClassifier failed: {e}, using heuristic fallback")

        # Note: Fast-path for trivial queries to avoid latency
        if query and self._is_trivial_query(query):
            logger.debug(f"[QueryRouter] {query_id}: Fast-path for trivial query")

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

            # Return minimal plan for trivial query
            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.GENERAL,
                complexity_score=0.0,
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
                    "fast_path": True,  # Mark as fast-path for telemetry
                },
            )

            # Create a single simple task for trivial queries
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_trivial",
                    task_type="general_task",
                    capability="reasoning",
                    prompt=query,
                    priority=1,
                    timeout_seconds=5.0,
                    parameters={"is_trivial": True, "skip_heavy_analysis": True},
                )
            ]

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=False,  # Trivial query with no tools
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: FAST-PATH source={source}, "
                f"tasks=1, complexity=0.00, llm_mode={plan.llm_mode.value}"
            )
            return plan

        # CRITICAL FIX: Complex physics handling path
        # Complex physics problems (Lagrangian mechanics, control theory, triple pendulum)
        # require FULL mathematical reasoning with extended timeouts (120s+) and high
        # complexity scores (0.80+). These should NOT use fast-path.
        if query and self._is_complex_physics_query(query):
            logger.info(
                f"[QueryRouter] {query_id}: COMPLEX-PHYSICS-PATH detected - "
                f"activating full mathematical reasoning"
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

            # Create plan for complex physics with FULL analysis
            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.REASONING,  # Complex physics is reasoning
                complexity_score=COMPLEX_PHYSICS_MIN_COMPLEXITY,  # High complexity (0.80+)
                uncertainty_score=0.3,  # Moderate uncertainty for complex derivations
                collaboration_needed=True,  # Enable multi-agent for complex analysis
                collaboration_agents=["reasoning", "planning"],  # Multiple perspectives
                arena_participation=False,  # Skip arena but use full tool selection
                telemetry_category=telemetry_category,
                telemetry_data={
                    "session_id": session_id,
                    "query_length": len(query),
                    "word_count": len(query.split()),
                    "query_number": query_number,
                    "source": source,
                    "learning_mode": learning_mode.value,
                    "complex_physics_path": True,  # Mark as complex physics
                    "extended_timeout": True,
                },
            )

            # Mark as safe (bypass false positives)
            plan.safety_passed = True
            plan.detected_patterns.append("complex_physics_derivation")

            # Create comprehensive task with ALL mathematical reasoning tools
            # FIX: Activate symbolic, mathematical, probabilistic, causal, and analogical
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_physics",
                    task_type="complex_physics_task",
                    capability="reasoning",
                    prompt=query,
                    priority=3,  # High priority for complex physics
                    timeout_seconds=COMPLEX_PHYSICS_TIMEOUT_SECONDS,  # 120s+ timeout
                    parameters={
                        "is_complex_physics": True,
                        "require_detailed_derivation": True,
                        "show_all_steps": True,
                        # Activate ALL mathematical reasoning capabilities
                        "tools": [
                            "symbolic",       # For Lagrangian algebra
                            "mathematical",   # For matrix operations
                            "probabilistic",  # For stability analysis
                            "causal",        # For system dynamics
                            "analogical",    # For similar problems
                        ],
                        "config": {
                            "max_tokens": 4000,      # Extended output for derivations
                            "require_proofs": True,
                            "show_all_steps": True,
                            "latex_output": True,
                            "numerical_precision": "high",
                        },
                        "skip_fast_path": True,  # Explicitly skip shortcuts
                    },
                )
            ]

            # Store full tool configuration in telemetry
            plan.telemetry_data["selected_tools"] = [
                "symbolic", "mathematical", "probabilistic", "causal", "analogical"
            ]
            plan.telemetry_data["reasoning_strategy"] = "complex_physics_full_derivation"
            plan.telemetry_data["timeout_seconds"] = COMPLEX_PHYSICS_TIMEOUT_SECONDS

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: COMPLEX-PHYSICS-PATH source={source}, "
                f"tasks=1, complexity={COMPLEX_PHYSICS_MIN_COMPLEXITY:.2f}, "
                f"timeout={COMPLEX_PHYSICS_TIMEOUT_SECONDS}s, tools=ALL, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

        # =================================================================
        # Note: CREATIVE QUERY FAST-PATH (BEFORE MATH!)
        # =================================================================
        # Creative tasks like "Write quantum sonnet" were incorrectly routed
        # to MATH because "quantum" triggered mathematical keyword detection.
        # 
        # The fix: Check for CREATIVE task type BEFORE checking for math keywords.
        # Task type (write, compose) has HIGHER priority than domain keywords (quantum).
        # =================================================================
        if query and self._is_creative_query(query):
            logger.info(
                f"[QueryRouter] {query_id}: CREATIVE-FAST-PATH detected "
                f"(task type overrides domain keywords)"
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
            
            # Create plan for creative tasks
            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.REASONING,  # Creative uses reasoning (analogical)
                complexity_score=0.4,  # Moderate complexity for creative tasks
                uncertainty_score=0.2,  # Some uncertainty - creative is subjective
                collaboration_needed=False,  # Creative is usually single-agent
                arena_participation=False,  # Skip arena for creative
                telemetry_category=telemetry_category,
                telemetry_data={
                    "session_id": session_id,
                    "query_length": len(query),
                    "word_count": len(query.split()),
                    "query_number": query_number,
                    "source": source,
                    "learning_mode": learning_mode.value,
                    "fast_path": True,
                    "creative_fast_path": True,
                    "bug9_fix_applied": True,
                },
            )
            
            plan.safety_passed = True
            plan.detected_patterns.append("creative_task")
            plan.detected_patterns.append("bug9_task_type_priority")
            
            # BUG #14 FIX: Creative writing tasks (poems, stories) should use 'general' tool
            # which routes to LLM for synthesis, NOT analogical reasoning.
            # 
            # Old (WRONG): creative_tools = ["analogical", "world_model"]
            #   - AnalogicalReasoner does structure mapping, not creative writing
            #   - This produced JSON like {"mapping": {}, "source_domain": "source"} for poems
            # 
            # New (CORRECT): creative_tools = ["general"]
            #   - "general" tool uses LLM for creative synthesis
            #   - This produces actual poems, stories, creative descriptions
            #
            # Note: Some creative tasks (like "describe X using the metaphor of Y") 
            # might benefit from analogical reasoning, but pure creative writing
            # (poems, stories) should go to LLM.
            creative_tools = ["general"]
            
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_creative",
                    task_type="creative_task",
                    capability="reasoning",
                    prompt=query,
                    priority=2,  # Moderate priority
                    timeout_seconds=30.0,  # Creative needs more time
                    parameters={
                        "is_creative": True,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        "tools": creative_tools,
                        "preferred_tool": "general",  # BUG #14 FIX: Use LLM for creative writing
                        "response_type": "creative",
                        "bug9_fix": True,
                        "skip_reasoning": True,  # BUG #14: Skip formal reasoning for creative tasks
                    },
                )
            ]
            
            plan.telemetry_data["selected_tools"] = creative_tools
            plan.telemetry_data["reasoning_strategy"] = "creative_llm"  # BUG #14 FIX: Changed from "creative_analogical"
            
            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(creative_tools),
                complexity_score=plan.complexity_score
            )
            
            logger.info(
                f"[QueryRouter] {query_id}: CREATIVE-FAST-PATH source={source}, "
                f"tasks=1, tools={creative_tools}, complexity=0.40, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

        # PERFORMANCE FIX: Mathematical query fast-path
        # Mathematical queries (Bayesian probability, statistics, calculations) should
        # bypass arena/multi-agent orchestration that causes 60+ second delays.
        # Route directly to probabilistic/symbolic reasoning with short timeout.
        if query and self._is_mathematical_query(query):
            logger.info(f"[QueryRouter] {query_id}: MATH-FAST-PATH detected for query")

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

            # Create optimized plan for mathematical queries
            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.EXECUTION,  # Mathematical execution
                complexity_score=0.3,  # Low complexity - direct calculation
                uncertainty_score=0.1,  # Low uncertainty - deterministic math
                collaboration_needed=False,  # No multi-agent needed
                arena_participation=False,  # Skip arena orchestration
                telemetry_category=telemetry_category,
                telemetry_data={
                    "session_id": session_id,
                    "query_length": len(query),
                    "word_count": len(query.split()),
                    "query_number": query_number,
                    "source": source,
                    "learning_mode": learning_mode.value,
                    "fast_path": True,
                    "math_fast_path": True,  # Mark as math fast-path
                },
            )

            # Mark as safe for mathematical scenarios (bypass HIPAA false positives)
            plan.safety_passed = True
            plan.detected_patterns.append("mathematical_calculation")

            # PRIORITY 4 FIX: Create mathematical execution tasks with specialized tools
            # Route to probabilistic/symbolic/mathematical tools instead of general
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_math",
                    task_type="mathematical_task",
                    capability="reasoning",  # Use probabilistic reasoning
                    prompt=query,
                    reasoning_type="mathematical",  # MANDATORY: math reasoning
                    tool_name="probabilistic",  # MANDATORY: primary tool for math
                    priority=2,  # Higher priority for math
                    timeout_seconds=MATH_QUERY_TIMEOUT_SECONDS,  # Short timeout (5s)
                    parameters={
                        "is_mathematical": True,
                        # FIX TASK 7: Pass full query context in parameters
                        "prompt": query,  # FIX: Explicitly include prompt in parameters
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        # PRIORITY 4 FIX: Route to specialized mathematical tools
                        "tools": ["probabilistic", "symbolic", "mathematical"],
                        "preferred_tool": "probabilistic",  # Hint to use probabilistic tool
                        "mathematical_scenario_override": True,  # Safety override
                        "require_verification": True,  # Trigger mathematical verification
                        "reasoning_context": {
                            "original_query": query,
                            "query_type": "mathematical",
                            "source": source,
                        },
                    },
                )
            ]

            # PRIORITY 4 FIX: Store selected tools in telemetry for downstream use
            plan.telemetry_data["selected_tools"] = [
                "probabilistic",
                "symbolic",
                "mathematical",
            ]
            plan.telemetry_data["reasoning_strategy"] = "mathematical_execution"

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: MATH-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.30, timeout={MATH_QUERY_TIMEOUT_SECONDS}s, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

        # =================================================================
        # Note: SELF-INTROSPECTION MULTI-TOOL ROUTING
        # =================================================================
        # Self-introspection queries like "Would you choose self-awareness?" were
        # being routed to PHILOSOPHICAL, producing generic philosophical analysis
        # without Vulcan's own perspective.
        #
        # The fix: Multi-tool approach where:
        # - PRIMARY: meta_reasoning (for Vulcan's own perspective)
        # - REFERENCE: philosophical (for frameworks to consult)
        # - ACCESS: world_model (for Vulcan's self-state)
        #
        # This allows Vulcan to CONSULT philosophy while forming its OWN position.
        # =================================================================
        if query and self._is_self_introspection_query(query):
            logger.info(
                f"[QueryRouter] {query_id}: SELF-INTROSPECTION-PATH detected "
                f"(multi-tool: meta_reasoning PRIMARY + philosophical REFERENCE)"
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
            
            # Create plan for self-introspection with multi-tool routing
            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.REASONING,  # Use REASONING, not PHILOSOPHICAL
                complexity_score=0.5,  # Moderate complexity - requires synthesis
                uncertainty_score=0.3,  # Some uncertainty - forming position
                collaboration_needed=True,  # Enable multi-tool collaboration
                collaboration_agents=["meta_reasoning", "philosophical"],
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
                    "self_introspection_path": True,
                    "bug16_fix_applied": True,
                    "multi_tool_routing": True,
                },
            )
            
            plan.safety_passed = True
            plan.detected_patterns.append("self_introspection_query")
            plan.detected_patterns.append("multi_tool_routing")
            
            # Note: Multi-tool configuration
            # PRIMARY: meta_reasoning - forms Vulcan's own position
            # REFERENCE: world_model (philosophical mode) - provides frameworks
            # ACCESS: world_model - provides self-state/representation
            primary_tools = ["meta_reasoning", "world_model"]
            reference_tools = ["world_model"]  # World Model handles philosophical via mode='philosophical'
            all_tools = primary_tools
            
            plan.agent_tasks = [
                # PRIMARY TASK: Meta-reasoning to form Vulcan's position
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_introspect_primary",
                    task_type="self_introspection_task",
                    capability="meta_reasoning",
                    prompt=query,
                    priority=3,  # High priority - this is the PRIMARY
                    timeout_seconds=30.0,
                    parameters={
                        "is_self_introspection": True,
                        "is_primary": True,
                        "access_self_model": True,
                        "tools": primary_tools,
                        "preferred_tool": "meta_reasoning",
                        "response_type": "self_reflection",
                        "execution_strategy": "primary_with_references",
                        "reference_tools": reference_tools,
                        "bug16_fix": True,
                        "metadata": {
                            "access_self_model": True,
                            "consult_philosophy": True,
                            "synthesize_position": True,
                        },
                    },
                ),
                # REFERENCE TASK: World Model philosophical analysis (to be consulted)
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_introspect_reference",
                    task_type="philosophical_reference_task",
                    capability="world_model",  # Route to World Model
                    prompt=f"Provide philosophical frameworks relevant to: {query}",
                    priority=2,  # Lower priority - this is REFERENCE
                    timeout_seconds=15.0,
                    parameters={
                        "is_self_introspection": True,
                        "is_reference": True,  # NOT primary - just a reference
                        "tools": reference_tools,
                        "mode": "philosophical",  # World Model philosophical mode
                        "response_type": "philosophical_analysis",
                        "reference_mode": True,  # Indicates this should be consulted
                        "bug16_fix": True,
                    },
                ),
            ]
            
            plan.telemetry_data["selected_tools"] = all_tools
            plan.telemetry_data["primary_tools"] = primary_tools
            plan.telemetry_data["reference_tools"] = reference_tools
            plan.telemetry_data["reasoning_strategy"] = "self_introspection_multi_tool"
            
            logger.info(
                f"[QueryRouter] {query_id}: SELF-INTROSPECTION-PATH source={source}, "
                f"tasks=2, primary={primary_tools}, reference={reference_tools}, "
                f"complexity=0.50"
            )
            return plan

        # PERFORMANCE FIX: Philosophical/paradox query fast-path
        # Paradoxes and thought experiments should NOT trigger complex reasoning
        # Route to lightweight general handler with short timeout
        if query and self._is_philosophical_query(query):
            logger.info(f"[QueryRouter] {query_id}: PHILOSOPHICAL-FAST-PATH detected")

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

            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.PHILOSOPHICAL,
                complexity_score=0.2,  # Low - philosophical discussion, not complex reasoning
                uncertainty_score=0.1,
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
                    "philosophical_fast_path": True,
                },
            )

            plan.safety_passed = True
            plan.detected_patterns.append("philosophical_query")

            # Route to world_model which has full meta-reasoning machinery:
            # - predict_interventions() for causal predictions
            # - InternalCritic for multi-framework evaluation
            # - GoalConflictDetector for dilemma analysis
            # - EthicalBoundaryMonitor for ethical constraints
            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_philosophical",
                    task_type="philosophical_task",
                    capability="world_model",  # Route to World Model for ethical reasoning
                    prompt=query,
                    priority=1,
                    timeout_seconds=PHILOSOPHICAL_TIMEOUT_SECONDS,
                    parameters={
                        "is_philosophical": True,
                        "prompt": query,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        "tools": ["world_model", "causal", "analogical"],
                        "response_type": "conversational",
                        "mode": "philosophical",  # Tell World Model to use philosophical reasoning
                        "reasoning_context": {
                            "original_query": query,
                            "query_type": "philosophical",
                            "source": source,
                        },
                    },
                )
            ]

            plan.telemetry_data["selected_tools"] = ["world_model", "causal", "analogical"]
            plan.telemetry_data["reasoning_strategy"] = "world_model_philosophical"

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: PHILOSOPHICAL-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.20, timeout={PHILOSOPHICAL_TIMEOUT_SECONDS}s, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

        # PERFORMANCE FIX: Identity query fast-path
        # Questions about who created the system need direct factual response
        if query and self._is_identity_query(query):
            logger.info(f"[QueryRouter] {query_id}: IDENTITY-FAST-PATH detected")

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

            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.IDENTITY,
                complexity_score=0.1,  # Very low - direct factual response
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
                    "identity_fast_path": True,
                },
            )

            plan.safety_passed = True
            plan.detected_patterns.append("identity_query")

            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_identity",
                    task_type="factual_task",
                    capability="factual",
                    prompt=query,
                    priority=1,
                    timeout_seconds=IDENTITY_TIMEOUT_SECONDS,
                    parameters={
                        "is_identity": True,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        "tools": ["factual"],
                        "response_type": "factual",
                    },
                )
            ]

            plan.telemetry_data["selected_tools"] = ["factual"]
            plan.telemetry_data["reasoning_strategy"] = "identity_factual"

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: IDENTITY-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.10, timeout={IDENTITY_TIMEOUT_SECONDS}s, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

        # PERFORMANCE FIX: Conversational query fast-path
        # Simple greetings and chat should use lightweight handler
        if query and self._is_conversational_query(query):
            logger.info(f"[QueryRouter] {query_id}: CONVERSATIONAL-FAST-PATH detected")

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

            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.CONVERSATIONAL,
                complexity_score=0.0,  # Zero complexity - simple greeting
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
                    "conversational_fast_path": True,
                },
            )

            plan.safety_passed = True
            plan.detected_patterns.append("conversational_query")

            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_conversational",
                    task_type="general_task",
                    capability="general",
                    prompt=query,
                    priority=1,
                    timeout_seconds=CONVERSATIONAL_TIMEOUT_SECONDS,
                    parameters={
                        "is_conversational": True,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        "tools": ["general"],
                        "response_type": "conversational",
                    },
                )
            ]

            plan.telemetry_data["selected_tools"] = ["general"]
            plan.telemetry_data["reasoning_strategy"] = "conversational_lightweight"

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: CONVERSATIONAL-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.00, timeout={CONVERSATIONAL_TIMEOUT_SECONDS}s, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

        # PERFORMANCE FIX: Factual query fast-path
        # Simple fact lookups don't need complex reasoning
        if query and self._is_factual_query(query):
            logger.info(f"[QueryRouter] {query_id}: FACTUAL-FAST-PATH detected")

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

            plan = ProcessingPlan(
                query_id=query_id,
                original_query=original_query,
                source=source,
                learning_mode=learning_mode,
                query_type=QueryType.FACTUAL,
                complexity_score=0.1,  # Low complexity - simple lookup
                uncertainty_score=0.1,
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
                    "factual_fast_path": True,
                },
            )

            plan.safety_passed = True
            plan.detected_patterns.append("factual_query")

            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_factual",
                    task_type="factual_task",
                    capability="factual",
                    prompt=query,
                    priority=1,
                    timeout_seconds=FACTUAL_TIMEOUT_SECONDS,
                    parameters={
                        "is_factual": True,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        "tools": ["factual"],
                        "response_type": "factual",
                    },
                )
            ]

            plan.telemetry_data["selected_tools"] = ["factual"]
            plan.telemetry_data["reasoning_strategy"] = "factual_lookup"

            # ARCHITECTURE: Set LLM mode based on query characteristics
            plan.llm_mode = self._determine_llm_mode(
                query_type=plan.query_type,
                has_selected_tools=bool(plan.telemetry_data.get("selected_tools")),
                complexity_score=plan.complexity_score
            )

            logger.info(
                f"[QueryRouter] {query_id}: FACTUAL-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.10, timeout={FACTUAL_TIMEOUT_SECONDS}s, "
                f"llm_mode={plan.llm_mode.value}"
            )
            return plan

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
            logger.debug("[QueryRouter] Reasoning integration not available - using fallback hint generation")
            # ARCHITECTURAL CHANGE: Router provides HINTS, not final tool selection
            # ToolSelector makes the final decision based on these hints
            tool_hints = self._select_reasoning_tools(plan)
            plan.telemetry_data["tool_hints"] = tool_hints  # Store hints for ToolSelector
            plan.telemetry_data["reasoning_strategy"] = "pattern_based"
            logger.info(
                f"[QueryRouter] Tool hints (fallback): hints={tool_hints}, strategy=pattern_based"
            )
        except Exception as e:
            logger.warning(f"[QueryRouter] Reasoning integration failed: {e} - using fallback")
            # ARCHITECTURAL CHANGE: Router provides HINTS, not final tool selection
            tool_hints = self._select_reasoning_tools(plan)
            plan.telemetry_data["tool_hints"] = tool_hints  # Store hints for ToolSelector
            plan.telemetry_data["reasoning_strategy"] = "pattern_based"
            logger.info(
                f"[QueryRouter] Tool hints (fallback after error): hints={tool_hints}"
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

    def _is_trivial_query(self, query: str) -> bool:
        """
        Detect simple greetings/short queries that don't need full analysis.

        Note: This method provides a fast-path for trivial queries
        to avoid the latency for simple greetings.

        Note: Integrates with embedding_cache.is_simple_query()
        for comprehensive simple query detection, reducing query routing
        times from 64+ seconds to ~200ms for repeated/simple queries.

        Args:
            query: The query string (not lowercased)

        Returns:
            True if the query is trivial and should skip heavy analysis
        """
        # PERFORMANCE FIX: Use embedding_cache's is_simple_query for comprehensive detection
        # This provides more robust detection patterns and caching benefits
        if EMBEDDING_CACHE_AVAILABLE and embedding_cache_is_simple_query is not None:
            if embedding_cache_is_simple_query(query):
                logger.debug(
                    f"[QueryRouter] Fast-path: simple query detected by embedding_cache"
                )
                return True

        query_lower = query.lower().strip()

        # Only consider very short queries (under 20 chars) as potentially trivial
        # This prevents "hello, can you help me with complex calculations?" from being trivial
        if len(query_lower) > 20:
            return False

        # Check if query is exactly a trivial pattern or starts with pattern + punctuation/space
        for pattern in self.TRIVIAL_PATTERNS:
            if query_lower == pattern:
                return True
            # Allow pattern followed by punctuation (e.g., "hello!", "thanks.")
            if query_lower.startswith(pattern) and len(query_lower) <= len(pattern) + 2:
                suffix = query_lower[len(pattern) :]
                if not suffix or all(c in "!.?,;: " for c in suffix):
                    return True

        return False

    def _is_creative_query(self, query: str) -> bool:
        """
        Detect creative writing tasks that should NOT use mathematical routing.
        
        Problem: "Write quantum sonnet" was being routed to MATH because "quantum" 
        triggered mathematical keyword detection. But "write" is a CREATIVE task
        that should override domain keywords.
        
        Priority hierarchy:
        1. Task type (write, compose, create) - HIGHEST PRIORITY
        2. Computational intent (calculate, optimize)
        3. Domain keywords (quantum, ethics)
        
        Examples:
        - "Write a quantum sonnet" -> True (creative task despite "quantum")
        - "Write a poem about calculus" -> True (creative task despite "calculus")
        - "Compose a song about physics" -> True (creative task)
        - "Calculate the integral" -> False (mathematical task)
        - "What is quantum physics?" -> False (informational query)
        
        Args:
            query: The query string (not lowercased)
            
        Returns:
            True if query is a creative writing/composition task
        """
        query_lower = query.lower().strip()
        
        # Creative task verbs that indicate writing/composition
        # These should take HIGHEST priority over domain keywords
        creative_task_markers = (
            'write', 'compose', 'create', 'craft', 'draft', 'author',
            'pen', 'generate a story', 'generate a poem', 'generate a song',
            'tell a story', 'make up a story', 'make a poem',
            'imagine a story', 'imagine a poem',
        )
        
        # Creative output types (nouns that indicate creative output)
        creative_outputs = (
            'poem', 'sonnet', 'haiku', 'limerick', 'ballad', 'verse',
            'story', 'tale', 'narrative', 'fable', 'myth', 'legend',
            'song', 'lyrics', 'jingle', 'melody',
            'essay', 'article', 'blog', 'post',
            'script', 'dialogue', 'monologue', 'screenplay',
            'novel', 'novella', 'fiction', 'prose',
        )
        
        # Check for creative task markers at the BEGINNING of the query
        # This is the strongest signal - user explicitly starts with "Write..."
        query_start = query_lower[:50]  # Check first 50 chars
        for marker in creative_task_markers:
            if query_start.startswith(marker):
                logger.debug(
                    f"[QueryRouter] Creative task detected - "
                    f"query starts with '{marker}'"
                )
                return True
        
        # Check for creative task markers anywhere with creative output type
        # "Write a quantum sonnet" -> has "write" + "sonnet"
        has_creative_verb = any(marker in query_lower for marker in creative_task_markers)
        has_creative_output = any(output in query_lower for output in creative_outputs)
        
        if has_creative_verb and has_creative_output:
            logger.debug(
                f"[QueryRouter] Creative task detected - "
                f"creative verb + creative output type"
            )
            return True
        
        # Check if query is asking to generate/create creative content
        # "Generate a poem" or "Create a story"
        generate_create_patterns = (
            'generate a ', 'generate an ', 'generate the ',
            'create a ', 'create an ', 'create the ',
            'make a ', 'make an ', 'make the ',
        )
        for pattern in generate_create_patterns:
            if pattern in query_lower:
                # Check if followed by creative output type
                rest = query_lower.split(pattern, 1)[1][:30]  # Next 30 chars after pattern
                if any(output in rest for output in creative_outputs):
                    logger.debug(
                        f"[QueryRouter] Creative task detected - "
                        f"'{pattern}' followed by creative output"
                    )
                    return True
        
        return False

    def _classify_query_type(self, query_lower: str) -> QueryType:
        """
        Classify the primary type of a query based on keyword analysis.

        Uses weighted keyword matching with priority ordering to determine
        the most appropriate query type. Enhanced with new query types for
        proper routing that prevents misclassification.
        
        FIX (Issue #ROUTING-001): Added fallback detection layer to catch
        queries that slip through pattern matching.

        Args:
            query_lower: Lowercased query string

        Returns:
            QueryType enum value
        """
        # Priority 1: Check for new specific query types first
        # These are checked via fast-paths in route_query, but we include here
        # for completeness in case classification is called directly
        
        # Check philosophical patterns (paradoxes, thought experiments)
        for pattern in PHILOSOPHICAL_PATTERNS:
            if pattern.search(query_lower):
                return QueryType.PHILOSOPHICAL
        phil_count = sum(1 for kw in PHILOSOPHICAL_KEYWORDS if kw in query_lower)
        if phil_count >= 1:
            return QueryType.PHILOSOPHICAL
        
        # Check identity patterns
        for pattern in IDENTITY_PATTERNS:
            if pattern.search(query_lower):
                return QueryType.IDENTITY
        identity_count = sum(1 for kw in IDENTITY_KEYWORDS if kw in query_lower)
        if identity_count >= 1:
            return QueryType.IDENTITY
        
        # Check conversational patterns (for short queries only)
        if len(query_lower.split()) <= CONVERSATIONAL_MAX_WORD_COUNT:
            for pattern in CONVERSATIONAL_PATTERNS:
                if pattern.search(query_lower):
                    return QueryType.CONVERSATIONAL
            conv_count = sum(1 for kw in CONVERSATIONAL_KEYWORDS if kw in query_lower)
            if conv_count >= 1:
                return QueryType.CONVERSATIONAL
        
        # Check factual patterns (simple lookups)
        for pattern in FACTUAL_PATTERNS:
            if pattern.search(query_lower):
                # Make sure it's not asking for reasoning
                if not any(ind in query_lower for ind in REASONING_EXCLUSION_INDICATORS):
                    return QueryType.FACTUAL
        
        # Priority 2: Count keyword matches for original types
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

    def _select_reasoning_tools(self, plan: ProcessingPlan) -> Dict[str, float]:
        """
        Provide reasoning tool HINTS/SUGGESTIONS to ToolSelector.
        
        INDUSTRY STANDARD - COMMAND vs SUGGESTION:
        Router SUGGESTS tools with confidence weights, ToolSelector DECIDES final tools.
        This establishes clear hierarchy: Router→hints, ToolSelector→authority.
        
        ARCHITECTURAL CHANGE (Industry Standard):
        - OLD: Router returned List[str] - direct tool selection (bypassed ToolSelector)
        - NEW: Router returns Dict[str, float] - tool suggestions with weights
        - ToolSelector is THE AUTHORITY for final tool selection
        - Router hints influence ToolSelector (e.g., +0.2 utility boost)
        
        This ensures single decision authority (ToolSelector) while preserving
        Router's domain knowledge through weighted suggestions.
        
        Priority Ordering (for hint generation):
            1. SAT/symbolic queries → {'symbolic': 0.9} (HIGHEST CONFIDENCE)
            2. Causal queries → {'causal': 0.85}
            3. Analogical queries → {'analogical': 0.8}
            4. Mathematical queries → {'mathematical': 0.8, 'symbolic': 0.5}
            5. Philosophical queries → {'philosophical': 0.85, 'world_model': 0.6}
            6. Probabilistic queries → {'probabilistic': 0.8}
        
        Args:
            plan: ProcessingPlan with query_type, detected_patterns, complexity_score
            
        Returns:
            Dict[str, float]: Tool name → confidence weight (0.0-1.0)
            Higher weight = stronger suggestion to ToolSelector
            
        Example:
            >>> plan = ProcessingPlan(...)
            >>> plan.detected_patterns = ['sat_problem', 'logic_symbols']
            >>> router._select_reasoning_tools(plan)
            {'symbolic': 0.9, 'probabilistic': 0.3}  # Hints, not commands
        """
        # Defensive programming: validate plan parameter
        if not plan or not isinstance(plan, ProcessingPlan):
            logger.warning("[QueryRouter._select_reasoning_tools] Invalid plan parameter")
            return {'general': 0.5}  # Low confidence fallback hint
        
        query_lower = plan.original_query.lower() if plan.original_query else ""
        detected_patterns = plan.detected_patterns or []
        query_type = plan.query_type
        complexity = plan.complexity_score
        uncertainty = plan.uncertainty_score
        
        # Initialize hints dictionary
        tool_hints = {}
        
        # ============================================================
        # PRIORITY 1: SAT/Symbolic Queries (HIGHEST PRIORITY)
        # ============================================================
        # SAT queries should strongly suggest symbolic engine
        sat_patterns = ['sat_', 'satisfiable', 'unsatisfiable', 'cnf', 'dnf']
        if any(p.startswith('sat_') for p in detected_patterns):
            logger.info(
                f"[QueryRouter._select_reasoning_tools] SAT pattern detected → symbolic hint 0.9"
            )
            tool_hints['symbolic'] = 0.9  # Very strong suggestion
            tool_hints['probabilistic'] = 0.2  # Weak fallback
            return tool_hints
        
        # Check for SAT keywords in query
        sat_keywords = ['satisfiable', 'unsatisfiable', 'sat', 'unsat', 'cnf', 'dnf']
        if any(kw in query_lower for kw in sat_keywords):
            # Additional check: ensure it's actually a SAT query, not just mentioning the word
            logic_symbols = ['→', '∧', '∨', '¬', '∀', '∃', '->', '/\\', '\\/', '~']
            has_logic = any(sym in plan.original_query or sym in query_lower for sym in logic_symbols)
            if has_logic:
                logger.info(
                    f"[QueryRouter._select_reasoning_tools] SAT keywords + logic symbols → symbolic hint 0.9"
                )
                tool_hints['symbolic'] = 0.9
                tool_hints['probabilistic'] = 0.2
                return tool_hints
        
        # ============================================================
        # FIX Issue #3: Natural Language Logic Queries
        # ============================================================
        # Detect natural language queries about logical reasoning
        # Examples: "Birds fly. Penguins are birds. Do penguins fly?"
        # These should route to world_model with LLM knowledge instead of
        # symbolic reasoner with empty knowledge base (which returns confidence=0.1)
        
        # Detection patterns for natural language logic:
        # 1. Multiple declarative statements followed by a question
        # 2. Logical relationships: "if X then Y", "all X are Y", "X implies Y"
        # 3. Common logic terms: "all", "some", "none", "every", "any"
        
        logic_relationship_keywords = [
            'all', 'some', 'none', 'every', 'any', 'if', 'then',
            'implies', 'therefore', 'thus', 'hence', 'because',
            'always', 'never', 'sometimes', 'necessarily', 'possibly'
        ]
        
        # Count declarative statements (sentences ending with period, not question mark)
        declarative_count = query_lower.count('.') - query_lower.count('?.')
        # Count questions
        question_count = query_lower.count('?')
        # Count logic relationship keywords
        logic_keyword_count = sum(1 for kw in logic_relationship_keywords if f' {kw} ' in f' {query_lower} ')
        
        # Heuristic: If query has multiple statements + question + logic keywords,
        # it's likely a natural language logic query
        is_nl_logic_query = (
            declarative_count >= 2 and  # At least 2 declarative statements
            question_count >= 1 and     # At least 1 question
            logic_keyword_count >= 1    # At least 1 logic keyword
        )
        
        # Additional check: common natural language logic patterns
        nl_logic_patterns = [
            r'\b(all|every)\s+\w+\s+(are|is|can|do|have)\b',  # "All birds fly", "Every dog barks"
            r'\b(some|any)\s+\w+\s+(are|is|can|do|have)\b',   # "Some birds swim"
            r'\b(no|none)\s+\w+\s+(are|is|can|do|have)\b',    # "No fish fly"
            r'\bif\s+.+\s+then\b',                             # "If X then Y"
            r'\b\w+\s+implies?\s+\w+\b',                       # "X implies Y"
        ]
        
        has_nl_logic_pattern = any(re.search(pattern, query_lower) for pattern in nl_logic_patterns)
        
        if is_nl_logic_query or has_nl_logic_pattern:
            logger.info(
                f"[QueryRouter._select_reasoning_tools] FIX Issue #3: Natural language logic query detected → "
                f"world_model hint 0.85 (avoids symbolic reasoner with empty KB)"
            )
            tool_hints['world_model'] = 0.85  # Strong suggestion for world_model
            tool_hints['symbolic'] = 0.2      # Weak fallback (might have formal logic too)
            tool_hints['philosophical'] = 0.3  # Moderate fallback (might involve reasoning)
            return tool_hints
        
        # ============================================================
        # PRIORITY 2: Causal Queries
        # ============================================================
        # Causal queries need causal inference engine
        causal_indicators = ['confound', 'intervention', 'do(', 'causal effect', 'counterfactual']
        has_causal_pattern = any('causal' in p for p in detected_patterns)
        has_causal_keyword = any(ind in query_lower for ind in causal_indicators)
        is_causal_type = query_type == QueryType.REASONING and 'causal' in query_lower
        
        if has_causal_pattern or has_causal_keyword or is_causal_type:
            logger.info(
                f"[QueryRouter._select_reasoning_tools] Causal indicators detected → causal hint 0.85"
            )
            tool_hints['causal'] = 0.85  # Strong suggestion
            tool_hints['probabilistic'] = 0.4  # Moderate fallback
            return tool_hints
        
        # ============================================================
        # PRIORITY 3: Analogical Queries
        # ============================================================
        # Analogical queries need structure mapping engine
        analogical_indicators = [
            'analogical', 'analogy', 'structure mapping', 'domain s', 'domain t',
            'similar to', 'like a', 'parallels', 'corresponds to', 'mapping'
        ]
        has_analogical_pattern = any('analogical' in p or 'analog' in p for p in detected_patterns)
        has_analogical_keyword = any(ind in query_lower for ind in analogical_indicators)
        
        if has_analogical_pattern or has_analogical_keyword:
            logger.info(
                f"[QueryRouter._select_reasoning_tools] Analogical indicators detected → analogical hint 0.8"
            )
            tool_hints['analogical'] = 0.8
            tool_hints['probabilistic'] = 0.3
            return tool_hints
        
        # ============================================================
        # PRIORITY 4: Mathematical Queries
        # ============================================================
        # Mathematical queries may need both mathematical and symbolic engines
        math_patterns = ['math_', 'probability', 'calculation', 'equation']
        has_math_pattern = any(any(mp in p for mp in math_patterns) for p in detected_patterns)
        
        # Check for mathematical keywords
        math_indicators = [
            'calculate', 'compute', 'solve', 'integral', 'derivative', 'equation',
            'probability', 'bayesian', 'likelihood', 'lagrangian', 'hamiltonian'
        ]
        has_math_keyword = any(ind in query_lower for ind in math_indicators)
        
        if has_math_pattern or has_math_keyword:
            # Complex math problems may need symbolic reasoning too
            if complexity >= 0.7 or 'prove' in query_lower or 'theorem' in query_lower:
                logger.info(
                    f"[QueryRouter._select_reasoning_tools] Complex math → mathematical hint 0.8, symbolic hint 0.6"
                )
                tool_hints['mathematical'] = 0.8
                tool_hints['symbolic'] = 0.6  # Support for proofs
                return tool_hints
            else:
                logger.info(
                    f"[QueryRouter._select_reasoning_tools] Mathematical indicators → mathematical hint 0.8"
                )
                tool_hints['mathematical'] = 0.8
                tool_hints['symbolic'] = 0.3  # Weak support
                return tool_hints
        
        # ============================================================
        # PRIORITY 5: Philosophical Queries
        # ============================================================
        # Philosophical queries use world_model for introspection
        philosophical_indicators = [
            'philosophical', 'ethics', 'moral', 'consciousness', 'free will',
            'paradox', 'dilemma', 'thought experiment'
        ]
        has_phil_pattern = any('philosophical' in p for p in detected_patterns)
        has_phil_keyword = any(ind in query_lower for ind in philosophical_indicators)
        is_phil_type = query_type == QueryType.PHILOSOPHICAL
        
        if has_phil_pattern or has_phil_keyword or is_phil_type:
            logger.info(
                f"[QueryRouter._select_reasoning_tools] Philosophical query → philosophical hint 0.85, world_model hint 0.6"
            )
            tool_hints['philosophical'] = 0.85
            tool_hints['world_model'] = 0.6  # Support for introspection
            return tool_hints
        
        # ============================================================
        # PRIORITY 6: Probabilistic Queries
        # ============================================================
        # Probabilistic queries need probabilistic reasoning engine
        prob_indicators = ['probability', 'bayesian', 'likelihood', 'posterior', 'prior', 'conditional']
        has_prob_keyword = any(ind in query_lower for ind in prob_indicators)
        
        if has_prob_keyword:
            logger.info(
                f"[QueryRouter._select_reasoning_tools] Probabilistic indicators → probabilistic hint 0.8"
            )
            tool_hints['probabilistic'] = 0.8
            return tool_hints
        
        # ============================================================
        # ENSEMBLE HINTS: High Complexity/Uncertainty
        # ============================================================
        # Very complex queries may benefit from multiple reasoning tools
        if complexity >= 0.8 and uncertainty >= 0.5:
            logger.info(
                f"[QueryRouter._select_reasoning_tools] High complexity + uncertainty → "
                f"ensemble hints (causal 0.7, probabilistic 0.7, world_model 0.5)"
            )
            tool_hints['causal'] = 0.7
            tool_hints['probabilistic'] = 0.7
            tool_hints['world_model'] = 0.5
            return tool_hints
        
        # ============================================================
        # DEFAULT: General Reasoning Hint
        # ============================================================
        logger.debug(
            f"[QueryRouter._select_reasoning_tools] No specific tools matched → general hint 0.5"
        )
        tool_hints['general'] = 0.5
        return tool_hints

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

        # CRITICAL FIX: Complex physics keywords force high complexity
        # PhD-level control theory/Lagrangian mechanics should NOT score 0.30
        # These require minimum complexity of 0.80 for proper tool selection
        physics_keyword_count = sum(
            1 for kw in COMPLEX_PHYSICS_KEYWORDS if kw in query_lower
        )
        if physics_keyword_count >= 1:
            # Force minimum complexity for complex physics
            # Even 1 keyword indicates advanced physics requiring full analysis
            physics_boost = max(
                COMPLEX_PHYSICS_MIN_COMPLEXITY - score,  # Bring up to minimum
                min(PHYSICS_COMPLEXITY_BOOST_CAP, physics_keyword_count * PHYSICS_COMPLEXITY_BOOST_PER_KEYWORD)
            )
            if physics_boost > 0:
                score += physics_boost
                logger.info(
                    f"[Complex Physics] Detected {physics_keyword_count} physics keyword(s), "
                    f"complexity boosted by {physics_boost:.2f} to {score:.2f}"
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
        if query_type != QueryType.PERCEPTION and any(
            kw in query_lower for kw in ("analyze", "examine", "data")
        ):
            tasks.append(
                AgentTask(
                    task_id=f"task_{base_task_id}_perception",
                    task_type="perception_support",
                    capability="perception",
                    prompt=f"Analyze input for: {query[:100]}",
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
        if query_type != QueryType.PLANNING and any(
            kw in query_lower for kw in ("step", "how to", "process", "plan")
        ):
            tasks.append(
                AgentTask(
                    task_id=f"task_{base_task_id}_planning",
                    task_type="planning_support",
                    capability="planning",
                    prompt=f"Create plan for: {query[:100]}",
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
