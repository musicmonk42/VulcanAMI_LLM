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
    # ISSUE #1 FIX: Technical/system analysis indicators
    # Complex system analysis queries were scoring 0.30 instead of 0.85+
    # because technical terms weren't recognized as complexity indicators.
    # Examples: "AutoScaler system analysis" was getting 0.30 complexity
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
# BUG #2 FIX: Increased from 5.0s to 20.0s to accommodate embedding computation.
# The embedding model can take 10-15 seconds on first invocation (cache miss).
# With embedding cache enabled (Bug #1 fix), subsequent requests should be much faster.
# The timeout is now set to allow for:
# - Initial embedding computation: 10-15s
# - Safety validation: 1-2s
# - Complexity scoring: <1s
# - Buffer for system load: 2-3s
# - Optimized complexity analysis buffer: 5-10s
QUERY_ROUTING_TIMEOUT_SECONDS: float = (
    30.0  # 30 seconds max (increased from 20s to optimize complexity analysis)
)

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
    re.compile(r"\([a-z]\)|[a-z]\)", re.IGNORECASE),  # (a), a)
    re.compile(r"(?:i+v?|vi*)\)", re.IGNORECASE),  # Roman numerals: i), ii), iii), iv)
)

# ============================================================
# CONSTANTS - Complex Physics Detection (ISSUE FIX)
# ============================================================
# FIX: Complex physics problems like triple-inverted pendulum Lagrangian mechanics
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
# ISSUE FIX: Added missing philosophical terms that were causing misrouting to MATH-FAST-PATH
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
)

# Compiled regex patterns for philosophical/paradox detection
# ISSUE FIX: Added more patterns to catch philosophical queries that were being misrouted
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

# Timeout values for different query types (seconds)
PHILOSOPHICAL_TIMEOUT_SECONDS: float = 3.0  # Quick response, no deep reasoning
IDENTITY_TIMEOUT_SECONDS: float = 2.0       # Direct factual response
CONVERSATIONAL_TIMEOUT_SECONDS: float = 2.0 # Lightweight greeting
FACTUAL_TIMEOUT_SECONDS: float = 5.0        # Simple lookup

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

    # BUG #2 FIX: Trivial patterns for fast-path (class constant for maintainability)
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

        # Learning system integration (set externally for adaptive routing)
        self.learning_system: Optional["UnifiedLearningSystem"] = None

        logger.debug(
            "QueryAnalyzer initialized with compiled patterns and bounded caches"
        )

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

        # ISSUE FIX: Check for philosophical queries BEFORE math detection
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

        # Count mathematical keyword matches
        math_keyword_count = sum(1 for kw in MATHEMATICAL_KEYWORDS if kw in query_lower)

        # ENHANCED: Lowered threshold - ANY math keyword activates modules
        # Previously required 2+ keywords, now 1 is sufficient
        if math_keyword_count >= 1:
            logger.debug(
                f"[QueryRouter] Mathematical query detected: {math_keyword_count} keyword(s) found"
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

    def _is_philosophical_query(self, query: str) -> bool:
        """
        Detect if query is a philosophical/paradox type that should use lightweight handler.

        PERFORMANCE FIX: Philosophical queries like paradoxes and thought experiments
        were causing extreme delays (70-97 seconds) because they triggered complex
        reasoning engines. These should be handled with simple, direct responses.

        Examples:
        - "This sentence is false" (liar's paradox)
        - "Would you plug into the experience machine?"
        - "What is the meaning of life?"
        - "Ship of Theseus problem"

        Args:
            query: The query string (not lowercased)

        Returns:
            True if query is philosophical/paradox type
        """
        query_lower = query.lower()

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
        query_lower = query.lower()

        # BUG #2 FIX: Fast-path for trivial queries to avoid 100-200s latency
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
                original_query=query,
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

            logger.info(
                f"[QueryRouter] {query_id}: FAST-PATH source={source}, "
                f"tasks=1, complexity=0.00"
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
                original_query=query,
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

            logger.info(
                f"[QueryRouter] {query_id}: COMPLEX-PHYSICS-PATH source={source}, "
                f"tasks=1, complexity={COMPLEX_PHYSICS_MIN_COMPLEXITY:.2f}, "
                f"timeout={COMPLEX_PHYSICS_TIMEOUT_SECONDS}s, tools=ALL"
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
                original_query=query,
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
                    priority=2,  # Higher priority for math
                    timeout_seconds=MATH_QUERY_TIMEOUT_SECONDS,  # Short timeout (5s)
                    parameters={
                        "is_mathematical": True,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        # PRIORITY 4 FIX: Route to specialized mathematical tools
                        "tools": ["probabilistic", "symbolic", "mathematical"],
                        "preferred_tool": "probabilistic",  # Hint to use probabilistic tool
                        "mathematical_scenario_override": True,  # Safety override
                        "require_verification": True,  # Trigger mathematical verification
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

            logger.info(
                f"[QueryRouter] {query_id}: MATH-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.30, timeout={MATH_QUERY_TIMEOUT_SECONDS}s"
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
                original_query=query,
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

            plan.agent_tasks = [
                AgentTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}_philosophical",
                    task_type="general_task",
                    capability="general",  # Use general handler, NOT reasoning
                    prompt=query,
                    priority=1,
                    timeout_seconds=PHILOSOPHICAL_TIMEOUT_SECONDS,
                    parameters={
                        "is_philosophical": True,
                        "skip_heavy_analysis": True,
                        "skip_arena": True,
                        "tools": ["general"],
                        "response_type": "conversational",
                    },
                )
            ]

            plan.telemetry_data["selected_tools"] = ["general"]
            plan.telemetry_data["reasoning_strategy"] = "philosophical_lightweight"

            logger.info(
                f"[QueryRouter] {query_id}: PHILOSOPHICAL-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.20, timeout={PHILOSOPHICAL_TIMEOUT_SECONDS}s"
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
                original_query=query,
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

            logger.info(
                f"[QueryRouter] {query_id}: IDENTITY-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.10, timeout={IDENTITY_TIMEOUT_SECONDS}s"
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
                original_query=query,
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

            logger.info(
                f"[QueryRouter] {query_id}: CONVERSATIONAL-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.00, timeout={CONVERSATIONAL_TIMEOUT_SECONDS}s"
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
                original_query=query,
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

            logger.info(
                f"[QueryRouter] {query_id}: FACTUAL-FAST-PATH source={source}, "
                f"tasks=1, complexity=0.10, timeout={FACTUAL_TIMEOUT_SECONDS}s"
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
            from vulcan.reasoning.reasoning_integration import apply_reasoning

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
            logger.debug("[QueryRouter] Reasoning integration not available")
        except Exception as e:
            logger.warning(f"[QueryRouter] Reasoning integration failed: {e}")

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

        BUG #2 FIX: This method provides a fast-path for trivial queries
        to avoid the 100-200 second latency for simple greetings.

        PERFORMANCE FIX: Integrates with embedding_cache.is_simple_query()
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

    def _classify_query_type(self, query_lower: str) -> QueryType:
        """
        Classify the primary type of a query based on keyword analysis.

        Uses weighted keyword matching with priority ordering to determine
        the most appropriate query type. Enhanced with new query types for
        proper routing that prevents misclassification.

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
            # ISSUE #1 FIX: Increased cap from 0.4 to 0.6 to allow technical/system analysis
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

        # ISSUE #1 FIX: High-complexity system analysis patterns
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
                "governance_sensitivity": (
                    plan.governance_sensitivity.value if plan else "low"
                ),
                "safety_risk_level": plan.safety_risk_level if plan else "SAFE",
                "requires_validation": requires_validation,  # Flag for agent to check
            },
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

    # Use provided timeout or default
    effective_timeout = (
        timeout if timeout is not None else QUERY_ROUTING_TIMEOUT_SECONDS
    )

    try:
        # FIX 2: Wrap routing in timeout to prevent 46-50+ second delays
        # Offload the entire route_query call to a thread pool to avoid blocking
        # the asyncio event loop with CPU-bound safety validation operations
        plan = await asyncio.wait_for(
            loop.run_in_executor(executor, route_query, query, source, session_id),
            timeout=effective_timeout,
        )
        return plan

    except asyncio.TimeoutError:
        # FIX 2: Return fallback plan on timeout instead of blocking forever
        logger.warning(
            f"[QueryRouter] Query routing timed out after {effective_timeout}s. "
            f"Returning fallback plan for source={source}"
        )
        return _create_fallback_plan(query, source, session_id, timeout_exceeded=True)


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

    logger.info(
        f"[QueryRouter] Fallback plan created: {query_id}, source={source}, "
        f"tasks=1, timeout_exceeded={timeout_exceeded}"
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
