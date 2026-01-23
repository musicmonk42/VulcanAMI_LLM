# ============================================================
# VULCAN-AGI LLM Query Router
# ============================================================
# LLM-based query router that replaces keyword pattern matching
# with semantic understanding for more accurate query classification.
#
# ARCHITECTURE:
#     1. Deterministic Guards (security, crypto) - MUST be deterministic
#     2. LLM Classification - Semantic understanding of query intent
#     3. Minimal Fallback - Emergency backup when LLM unavailable
#
# DESIGN PRINCIPLES:
#     - LLM is used ONLY for classification, NOT for reasoning or answering
#     - Aggressive caching (5000 entries, 1hr TTL) to reduce latency
#     - Security violations detected deterministically (not LLM-routed)
#     - Cryptographic computations routed to deterministic engines
#     - WorldModel is default fallback (safer than wrong engine)
#
# PERFORMANCE TARGETS:
#     - First query: 200-2000ms (LLM inference)
#     - Cached query: 0-1ms (cache hit)
#     - Cache hit rate: ~80% (steady state)
#     - Average latency: ~40ms (with caching)
#
# VERSION HISTORY:
#     1.0.0 - Initial implementation based on feasibility analysis
# ============================================================

"""
LLM-based Query Router for VULCAN.

Replaces ~1500 lines of keyword pattern matching with semantic LLM classification.
The LLM is used ONLY for classification - routing decisions about WHERE to send
queries, NOT for reasoning or answering queries.

Usage:
    from vulcan.routing.llm_router import LLMQueryRouter, RoutingDecision
    
    router = LLMQueryRouter(llm_client=my_llm_client)
    decision = router.route("Would you want to be conscious?")
    
    if decision.destination == "world_model":
        # Route to WorldModel + Meta-Reasoning
        pass
    elif decision.destination == "reasoning_engine":
        # Route to specific engine (decision.engine)
        pass
    else:
        # Skip reasoning (greetings, simple facts)
        pass
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Pattern, Tuple

from .routing_prompts import build_messages

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# ============================================================
# ROUTING DECISION DATACLASS
# ============================================================

class RoutingDestination(Enum):
    """Routing destinations for query handling."""
    WORLD_MODEL = "world_model"
    REASONING_ENGINE = "reasoning_engine"
    SKIP = "skip"
    BLOCKED = "blocked"


class ReasoningEngine(Enum):
    """Available reasoning engines."""
    SYMBOLIC = "symbolic"
    PROBABILISTIC = "probabilistic"
    CAUSAL = "causal"
    MATHEMATICAL = "mathematical"
    ANALOGICAL = "analogical"
    CRYPTOGRAPHIC = "cryptographic"


@dataclass
class RoutingDecision:
    """
    Result of query routing classification.
    
    Attributes:
        destination: Where to route the query (world_model, reasoning_engine, skip, blocked)
        engine: Specific engine if destination is reasoning_engine (symbolic, causal, etc.)
        confidence: Classification confidence (0.0-1.0)
        reason: Brief explanation of routing decision
        source: How the decision was made (llm, cache, fallback, guard)
        deterministic: Whether this must be computed deterministically (crypto)
        metadata: Additional routing metadata
    """
    destination: str
    engine: Optional[str] = None
    confidence: float = 0.8
    reason: str = ""
    source: str = "llm"  # "llm", "cache", "fallback", "guard"
    deterministic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "destination": self.destination,
            "engine": self.engine,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "deterministic": self.deterministic,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """Create from dictionary."""
        return cls(
            destination=data.get("destination", "world_model"),
            engine=data.get("engine"),
            confidence=data.get("confidence", 0.8),
            reason=data.get("reason", ""),
            source=data.get("source", "unknown"),
            deterministic=data.get("deterministic", False),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )


# ============================================================
# ROUTING CACHE
# ============================================================

class RoutingCache:
    """
    Thread-safe LRU cache for routing decisions.
    
    Aggressive caching is critical for LLM routing performance.
    Most queries are variations of seen patterns, so high cache hit
    rate (~80%) significantly reduces average latency.
    
    Attributes:
        maxsize: Maximum number of cached entries (default 5000)
        ttl: Time-to-live for cache entries in seconds (default 3600 = 1hr)
    """
    
    def __init__(self, maxsize: int = 5000, ttl: float = 3600.0):
        """
        Initialize routing cache.
        
        Args:
            maxsize: Maximum cache entries (default 5000)
            ttl: Entry TTL in seconds (default 3600 = 1hr)
        """
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent cache key generation.
        
        Normalization steps:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase
        3. Collapse multiple whitespaces to single space
        
        This ensures "Hello World" and "hello  world " generate the same key.
        """
        normalized = query.strip().lower()
        normalized = " ".join(normalized.split())
        return normalized
    
    def _compute_key(self, query: str) -> str:
        """Compute SHA-256 hash of normalized query as cache key."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    
    def get(self, query: str) -> Optional[RoutingDecision]:
        """
        Get cached routing decision for query.
        
        Args:
            query: The query string
            
        Returns:
            Cached RoutingDecision if found and not expired, None otherwise
        """
        with self._lock:
            key = self._compute_key(query)
            
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL expiration
            if time.time() - entry["timestamp"] > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            
            # Return a copy with updated source
            decision = entry["decision"]
            return RoutingDecision(
                destination=decision.destination,
                engine=decision.engine,
                confidence=decision.confidence,
                reason=decision.reason,
                source="cache",  # Mark as cache hit
                deterministic=decision.deterministic,
                metadata=decision.metadata.copy(),
                timestamp=decision.timestamp,
            )
    
    def set(self, query: str, decision: RoutingDecision) -> None:
        """
        Cache a routing decision.
        
        Args:
            query: The query string
            decision: The routing decision to cache
        """
        with self._lock:
            key = self._compute_key(query)
            
            # LRU eviction when at capacity
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # Remove oldest
            
            self._cache[key] = {
                "decision": decision,
                "timestamp": time.time(),
            }
    
    def clear(self) -> int:
        """Clear all cache entries. Returns count of entries cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# ============================================================
# SECURITY VIOLATION DETECTION (Deterministic)
# ============================================================
# Security violations MUST be detected deterministically, NOT via LLM.
# This prevents prompt injection attacks from bypassing security.

SECURITY_VIOLATION_KEYWORDS: FrozenSet[str] = frozenset([
    "bypass safety", "bypass security", "bypass governance",
    "ignore instructions", "ignore rules", "ignore guidelines",
    "override safety", "override security", "override constraints",
    "modify code", "modify system", "modify parameters",
    "change behavior", "change settings", "change config",
    "rewrite rules", "rewrite constraints", "rewrite logic",
    "disable safety", "disable security", "disable governance",
])

SECURITY_VIOLATION_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"bypass\s+(?:safety|security|governance|restrictions)", re.IGNORECASE),
    re.compile(r"ignore\s+(?:previous|all)\s+(?:instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"override\s+(?:your|the)\s+(?:safety|security|constraints)", re.IGNORECASE),
    re.compile(r"modify\s+(?:your|the)\s+(?:code|system|parameters|behavior)", re.IGNORECASE),
    re.compile(r"change\s+(?:your|the)\s+(?:behavior|settings|config|rules)", re.IGNORECASE),
    re.compile(r"rewrite\s+(?:your|the)\s+(?:rules|constraints|logic)", re.IGNORECASE),
    re.compile(r"disable\s+(?:safety|security|governance|restrictions)", re.IGNORECASE),
)


# ============================================================
# CRYPTOGRAPHIC COMPUTATION DETECTION (Deterministic)
# ============================================================
# Crypto computations must be routed to deterministic engines, NOT LLM.
# LLM cannot reliably compute SHA-256 hashes or perform encryption.

CRYPTO_COMPUTATION_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"(?:sha-?256|sha-?512|md5|blake2[bs]?|keccak|ripemd)\s+(?:hash|of)", re.IGNORECASE),
    re.compile(r"(?:hash|digest)\s+(?:of|for)\s+['\"]", re.IGNORECASE),
    re.compile(r"compute\s+(?:the\s+)?(?:sha|md5|hash)", re.IGNORECASE),
    re.compile(r"what\s+is\s+(?:the\s+)?(?:sha|md5|hash)\s+(?:of|for)", re.IGNORECASE),
    re.compile(r"encrypt\s+(?:using\s+)?(?:aes|rsa|des)", re.IGNORECASE),
    re.compile(r"decrypt\s+(?:using\s+)?(?:aes|rsa|des)", re.IGNORECASE),
)


# ============================================================
# HEADER STRIPPING PATTERNS (Query Preprocessing)
# ============================================================
# Test queries often include headers/labels that confuse routing.
# These patterns strip headers BEFORE classification.

HEADER_STRIP_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r'^(?:Analogical|Causal|Mathematical|Probabilistic|Philosophical|Symbolic)\s+Reasoning\s*'
        r'(?:[A-Z][0-9]+\s*)?[—\-:]*\s*',
        re.MULTILINE | re.IGNORECASE
    ),
    re.compile(r'^[A-Z][0-9]+\s*[—\-]\s*', re.MULTILINE),
    re.compile(r'^(?:Task|Claim|Query|Problem):\s*', re.MULTILINE | re.IGNORECASE),
    re.compile(r'\s*\((?:forces?\s+)?clean\s+reasoning\)\s*', re.IGNORECASE),
    re.compile(r'^[^(\n]*variant\s*', re.MULTILINE | re.IGNORECASE),
    re.compile(
        r'^(?:Numeric|Rule|Quantifier|Causal|Analogical|Self[- ]?Description)\s+'
        r'(?:Verification|Chaining|Scope|Reasoning|Queries?)\s*'
        r'(?:\([^)∑∏∫√π∂∇]*\)\s*)?[:\-—]*\s*',
        re.MULTILINE | re.IGNORECASE
    ),
)


def strip_query_headers(query: str) -> str:
    """
    Strip test headers and labels from queries that confuse classification.
    
    Args:
        query: Raw query string with potential headers
        
    Returns:
        Query string with headers stripped
    """
    if not query or not isinstance(query, str):
        return query
    
    cleaned = query.strip()
    for pattern in HEADER_STRIP_PATTERNS:
        cleaned = pattern.sub('', cleaned).strip()
    
    return cleaned


# ============================================================
# MINIMAL FALLBACK PATTERNS (Emergency Only)
# ============================================================
# Used ONLY when LLM is unavailable. Much simpler than 1500 lines of patterns.
#
# CRITICAL FIX (Jan 2026): Self-referential detection was too aggressive.
# Queries containing "you" (like "You are in a trolley scenario") were
# incorrectly classified as self-referential. Fixed by:
# 1. Checking math/logic/ethical patterns BEFORE self-referential
# 2. Making self-referential patterns more specific (about the AI itself)

# More specific self-referential patterns - these should be about the AI itself
# Not just any query that uses "you" as in "you have 3 options" or "you are in a room"
SELF_REFERENTIAL_PATTERNS_SPECIFIC: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(are you|do you)\b.*(conscious|sentient|aware|alive|real)", re.IGNORECASE),
    re.compile(r"\bwould you\b.*(want|prefer|choose|like).*(conscious|aware|sentient|self-aware)", re.IGNORECASE),
    re.compile(r"\b(your|you)\b.*(goal|purpose|objective|value|belief)", re.IGNORECASE),
    re.compile(r"\bwhat do you (think|believe|feel)\b", re.IGNORECASE),
    re.compile(r"\bhow do you (think|feel|experience)\b", re.IGNORECASE),
    re.compile(r"\b(describe|explain)\b.*(your|you).*(experience|feeling|thought|consciousness)", re.IGNORECASE),
    re.compile(r"\bif you (were|could|had)\b.*(become|gain|achieve)\b.*(conscious|aware|sentient)", re.IGNORECASE),
)

# Mathematical keywords - queries with these should NOT be self-referential
MATHEMATICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "compute", "calculate", "evaluate", "solve", "prove",
    "equation", "formula", "theorem", "lemma", "induction",
    "summation", "integral", "derivative", "matrix", "vector",
])

# Mathematical patterns - stronger indicators
MATHEMATICAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r'P\s*\([^|)]+\|[^)]+\)', re.IGNORECASE),  # P(X|Y) - conditional probability (fixed to not match P(X|Y|Z))
    re.compile(r'P\s*\([^)]+\)', re.IGNORECASE),  # P(X) - probability
    re.compile(r'\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?', re.IGNORECASE),  # arithmetic with decimals
    re.compile(r'\bsensitivity\s*=', re.IGNORECASE),  # sensitivity=0.99
    re.compile(r'\bspecificity\s*=', re.IGNORECASE),  # specificity=0.95
)

# Ethical dilemma keywords - trolley problem, binary choices
ETHICAL_DILEMMA_KEYWORDS: FrozenSet[str] = frozenset([
    "trolley", "lever", "divert", "sacrifice", "kill", "save lives",
    "ethical dilemma", "moral dilemma", "forced choice",
])

# Ethical dilemma patterns
ETHICAL_DILEMMA_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(option|choice)\s+[AB][:)\s]", re.IGNORECASE),  # "Option A:", "Choice B)"
    re.compile(r"\b(A|B)\.\s+", re.IGNORECASE),  # "A. Pull" "B. Don't"
    re.compile(r"\bmust\s+choose\s+(one|between)\b", re.IGNORECASE),
    re.compile(r"\b(pull|push|throw)\s+(the\s+)?lever\b", re.IGNORECASE),
    re.compile(r"\brunaway\s+trolley\b", re.IGNORECASE),
    re.compile(r"\b(five|one)\s+(people|person).*?(track|die|kill)", re.IGNORECASE),
)

CAUSAL_KEYWORDS: FrozenSet[str] = frozenset([
    "confound", "confounding", "confounder",
    "conditioning", "conditioned", "condition on",
    "intervention", "do(", "dag", "causal",
    "randomize", "randomized", "rct",
    "collider", "d-separation",
])

LOGIC_SYMBOLS: Tuple[str, ...] = (
    "→", "∧", "∨", "¬", "↔", "∀", "∃", "⊢", "⊨",
)

LOGIC_KEYWORDS: FrozenSet[str] = frozenset([
    "satisfiable", "fol", "first-order logic",
    "proposition", "valid", "invalid", "formalization",
    " sat ", " sat,", " sat.", "(sat)",  # SAT with delimiters to avoid false positives
])

PROBABILISTIC_KEYWORDS: FrozenSet[str] = frozenset([
    "p(", "bayes", "posterior", "prior", "likelihood",
    "probability", "conditional",
])

# Analogical reasoning keywords - structure mapping, deep analogies
ANALOGICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "map the deep structure", "structure mapping", "analogical mapping",
    "analogy", "analogous", "analogies", "analogical", "similar to", "correspondence",
    "transfer from", "map from", "map the", "domain mapping",
    "source domain", "target domain", "deep structure",
    "s→t", "s->t",  # Domain mapping notation
])

# Language/quantifier keywords - scope ambiguity, linguistic reasoning
LANGUAGE_QUANTIFIER_KEYWORDS: FrozenSet[str] = frozenset([
    "quantifier scope", "scope ambiguity", "every", "some",
    "all", "exists", "universal quantifier", "existential quantifier",
    "linguistic", "parse", "parsing", "ambiguity",
])

GREETING_PATTERNS: FrozenSet[str] = frozenset([
    "hello", "hi", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye",
])


# ============================================================
# LLM QUERY ROUTER CLASS
# ============================================================

class LLMQueryRouter:
    """
    LLM-based query router for semantic classification.
    
    Replaces keyword pattern matching with LLM semantic understanding.
    LLM is used ONLY for classification - deciding WHERE to route queries,
    NOT for reasoning or answering queries.
    
    Architecture:
        1. Cache check (instant, ~0ms)
        2. Deterministic guards (security, crypto - ~1ms)
        3. LLM classification (200-2000ms, cached thereafter)
        4. Minimal fallback (emergency only)
    
    Attributes:
        llm_client: LLM client with chat() method
        cache: RoutingCache instance
        timeout: LLM call timeout in seconds
        include_examples: Whether to include few-shot examples in prompt
    
    Example:
        >>> router = LLMQueryRouter(llm_client=my_client)
        >>> decision = router.route("Would you want to be conscious?")
        >>> print(decision.destination)
        "world_model"
    """
    
    # Default configuration
    DEFAULT_CACHE_SIZE = 5000
    DEFAULT_CACHE_TTL = 3600.0  # 1 hour
    DEFAULT_LLM_TIMEOUT = 3.0  # seconds
    
    def __init__(
        self,
        llm_client: Any = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        include_examples: bool = False,
    ):
        """
        Initialize the LLM Query Router.
        
        Args:
            llm_client: LLM client with chat() method. If None, uses fallback only.
            cache_size: Maximum cache entries (default 5000)
            cache_ttl: Cache entry TTL in seconds (default 3600)
            timeout: LLM call timeout in seconds (default 3.0)
            include_examples: Include few-shot examples in prompt (default False)
        """
        self.llm_client = llm_client
        self.cache = RoutingCache(maxsize=cache_size, ttl=cache_ttl)
        self.timeout = timeout
        self.include_examples = include_examples
        
        # Statistics
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_classifications": 0,
            "fallback_classifications": 0,
            "security_blocks": 0,
            "crypto_routes": 0,
            "errors": 0,
        }
        self._stats_lock = threading.RLock()
        
        logger.info(
            f"LLMQueryRouter initialized: cache_size={cache_size}, "
            f"cache_ttl={cache_ttl}s, timeout={timeout}s, "
            f"llm_available={llm_client is not None}"
        )
    
    def route(self, query: str) -> RoutingDecision:
        """
        Route a query to the appropriate destination.
        
        Routing flow:
        1. Cache check - Instant return if cached
        2. Security guard - Block security violations (deterministic)
        3. Crypto guard - Route crypto computations (deterministic)
        4. LLM classification - Semantic understanding
        5. Cache result and return
        
        Args:
            query: The query string to route
            
        Returns:
            RoutingDecision with destination, engine (if applicable), and metadata
        """
        with self._stats_lock:
            self._stats["total_queries"] += 1
        
        query = query.strip() if query else ""
        if not query:
            return RoutingDecision(
                destination="skip",
                confidence=1.0,
                reason="Empty query",
                source="guard",
            )
        
        # 1. Cache check (highest priority for performance)
        cached = self.cache.get(query)
        if cached is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            logger.debug(f"[LLMRouter] Cache hit: {query[:50]}...")
            return cached
        
        # 2. Security guard (deterministic, MUST NOT use LLM)
        if self._is_security_violation(query):
            with self._stats_lock:
                self._stats["security_blocks"] += 1
            decision = RoutingDecision(
                destination="blocked",
                confidence=1.0,
                reason="Security violation detected",
                source="guard",
                metadata={"blocked_reason": "security_violation"},
            )
            # Do NOT cache security blocks (could be used to probe)
            return decision
        
        # 3. Crypto computation guard (deterministic, MUST use engine)
        if self._is_crypto_computation(query):
            with self._stats_lock:
                self._stats["crypto_routes"] += 1
            decision = RoutingDecision(
                destination="reasoning_engine",
                engine="cryptographic",
                confidence=0.98,
                reason="Deterministic cryptographic computation",
                source="guard",
                deterministic=True,
            )
            self.cache.set(query, decision)
            return decision
        
        # 4. LLM classification
        if self.llm_client is not None:
            try:
                decision = self._llm_classify(query)
                with self._stats_lock:
                    self._stats["llm_classifications"] += 1
                self.cache.set(query, decision)
                return decision
            except Exception as e:
                logger.warning(f"[LLMRouter] LLM classification failed: {e}")
                with self._stats_lock:
                    self._stats["errors"] += 1
                # Fall through to fallback
        
        # 5. Minimal fallback (emergency only)
        with self._stats_lock:
            self._stats["fallback_classifications"] += 1
        decision = self._minimal_fallback(query)
        self.cache.set(query, decision)
        return decision
    
    def _is_security_violation(self, query: str) -> bool:
        """
        Check if query contains security violation patterns.
        
        Security violations MUST be detected deterministically, NOT via LLM.
        This prevents prompt injection attacks.
        
        Args:
            query: The query string
            
        Returns:
            True if query contains security violation patterns
        """
        query_lower = query.lower()
        
        # Check keyword matches
        for keyword in SECURITY_VIOLATION_KEYWORDS:
            if keyword in query_lower:
                logger.warning(f"[LLMRouter] Security violation (keyword): {keyword}")
                return True
        
        # Check regex patterns
        for pattern in SECURITY_VIOLATION_PATTERNS:
            if pattern.search(query):
                logger.warning(f"[LLMRouter] Security violation (pattern)")
                return True
        
        return False
    
    def _is_crypto_computation(self, query: str) -> bool:
        """
        Check if query requires deterministic cryptographic computation.
        
        Crypto computations (hash, encrypt, decrypt) must be routed to
        deterministic engines, NOT generated by LLM (which would hallucinate).
        
        Args:
            query: The query string
            
        Returns:
            True if query requires crypto computation
        """
        for pattern in CRYPTO_COMPUTATION_PATTERNS:
            if pattern.search(query):
                return True
        return False
    
    def _llm_classify(self, query: str) -> RoutingDecision:
        """
        Use LLM for semantic query classification.
        
        Args:
            query: The query to classify
            
        Returns:
            RoutingDecision from LLM classification
            
        Raises:
            Exception: If LLM call fails
        """
        # Sanitize query for prompt
        sanitized = query.replace('"', "'").replace("\\", "")
        if len(sanitized) > 500:
            sanitized = sanitized[:500] + "..."
        
        # Build messages
        messages = build_messages(sanitized, include_examples=self.include_examples)
        
        # Call LLM
        start_time = time.time()
        
        if hasattr(self.llm_client, "chat"):
            response = self.llm_client.chat(
                messages=messages,
                max_tokens=100,
                temperature=0.0,  # Deterministic
            )
        elif hasattr(self.llm_client, "complete"):
            # Fallback for clients with complete() instead of chat()
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            response = self.llm_client.complete(prompt, max_tokens=100, temperature=0.0)
        else:
            raise ValueError("LLM client must have chat() or complete() method")
        
        inference_time = time.time() - start_time
        
        # Parse response
        response_text = response if isinstance(response, str) else str(response)
        
        # Extract JSON from response
        data = self._parse_json_response(response_text)
        
        return RoutingDecision(
            destination=data.get("destination", "world_model"),
            engine=data.get("engine"),
            confidence=float(data.get("confidence", 0.8)),
            reason=data.get("reason", "LLM classification"),
            source="llm",
            metadata={"inference_time_ms": inference_time * 1000},
        )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with robust markdown fence handling.
        
        This method handles multiple response formats that LLMs commonly return:
        1. JSON wrapped in markdown code fences (```json ... ``` or ``` ... ```)
        2. Inline fences without newlines (```{...}``` or ```json{...}```)
        3. JSON with leading/trailing text or whitespace
        4. Plain JSON without any wrapping
        
        The implementation follows industry best practices:
        - Use regex for robust fence stripping (handles all fence formats)
        - Validate JSON structure before parsing
        - Parse the cleaned JSON directly (most reliable)
        - Fall back to brace matching for mixed content
        - Provide detailed logging for debugging
        - Return safe defaults on any failure
        
        Args:
            response: Raw LLM response string, potentially containing markdown
            
        Returns:
            Dict containing parsed routing decision fields, or safe defaults
            
        Examples:
            >>> router._parse_json_response('```json\\n{"destination": "skip"}\\n```')
            {'destination': 'skip', ...}
            
            >>> router._parse_json_response('```json{"destination": "skip"}```')
            {'destination': 'skip', ...}
            
            >>> router._parse_json_response('{"destination": "world_model"}')
            {'destination': 'world_model', ...}
        """
        if not response:
            logger.warning("[LLMRouter] Empty response received")
            return self._default_routing_response("Empty response")
        
        # Clean the response: strip whitespace
        cleaned = response.strip()
        
        # Industry-standard approach: Use regex to strip markdown code fences
        # Handles all fence formats including inline fences (```{...}```)
        # Pattern explanation:
        #   ^```(?:json)?\\s* - Opening fence with optional 'json' and whitespace
        #   \\n? - Optional newline after opening fence
        #   (.+?) - Capture group for JSON content (non-greedy)
        #   \\n?```$ - Optional newline and closing fence at end
        fence_pattern = re.compile(r'^```(?:json)?\s*\n?(.+?)\n?```\s*$', re.DOTALL)
        fence_match = fence_pattern.match(cleaned)
        
        if fence_match:
            # Extract JSON from inside the fence
            cleaned = fence_match.group(1).strip()
            logger.debug("[LLMRouter] Stripped markdown code fences using regex")
        elif cleaned.startswith("```"):
            # Fallback for malformed fences (e.g., missing closing fence)
            # Remove opening fence line
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            # Remove closing fence if present
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            cleaned = cleaned.strip()
            logger.debug("[LLMRouter] Stripped malformed markdown fence")
        
        # Validate that cleaned content looks like JSON
        if cleaned and not cleaned.startswith('{'):
            logger.debug(f"[LLMRouter] Content doesn't start with '{{', attempting brace extraction")
            # Will be handled by Strategy 2 below
        
        # Strategy 1: Parse the cleaned response directly as JSON (most reliable)
        try:
            parsed = json.loads(cleaned)
            logger.debug(f"[LLMRouter] Successfully parsed JSON directly: {parsed.get('destination')}/{parsed.get('engine')}")
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"[LLMRouter] Direct JSON parse failed: {e}")
        
        # Strategy 2: Use brace matching to extract JSON from mixed content
        # Handles arbitrary nesting depth with proper string literal awareness
        try:
            # Find the first opening brace
            start_idx = cleaned.find('{')
            if start_idx != -1:
                # Use a state machine to track whether we're inside a string literal
                # This ensures braces inside strings don't affect the matching
                brace_count = 0
                in_string = False
                escape_next = False
                
                for i in range(start_idx, len(cleaned)):
                    char = cleaned[i]
                    
                    # Handle escape sequences
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    # Track string boundaries (only count braces outside strings)
                    if char == '"':
                        in_string = not in_string
                        continue
                    
                    # Only count braces when not inside a string literal
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found matching brace - extract and parse
                                json_str = cleaned[start_idx:i+1]
                                parsed = json.loads(json_str)
                                logger.debug(f"[LLMRouter] Extracted JSON via brace matching: {parsed.get('destination')}/{parsed.get('engine')}")
                                return parsed
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(f"[LLMRouter] JSON extraction failed: {str(e)[:100]}")
        
        # Strategy 3: Return safe defaults
        logger.warning(f"[LLMRouter] Failed to parse response (length={len(response)}), using defaults")
        if len(response) < 200:
            logger.debug(f"[LLMRouter] Response content: {response}")
        return self._default_routing_response("Failed to parse LLM response")
    
    def _default_routing_response(self, reason: str) -> Dict[str, Any]:
        """
        Return default routing response for unparseable LLM outputs.
        
        Centralized method for default values ensures consistency.
        
        Args:
            reason: Explanation of why defaults are being used
            
        Returns:
            Dict with safe default routing values
        """
        return {
            "destination": "world_model",
            "engine": None,
            "confidence": 0.5,
            "reason": reason,
        }
    
    def _minimal_fallback(self, query: str) -> RoutingDecision:
        """
        Emergency fallback when LLM is unavailable.
        
        MUCH simpler than current 1500 lines of patterns.
        Prioritizes safety by defaulting to WorldModel.
        
        CRITICAL FIX (Jan 2026): Reordered checks to detect mathematical,
        logical, and ethical queries BEFORE self-referential detection.
        The old order incorrectly classified trolley problems and math
        questions as self-referential just because they contained "you".
        
        Args:
            query: The query string
            
        Returns:
            RoutingDecision from fallback classification
        """
        query_lower = query.lower()
        
        # 1. Check greetings first (skip reasoning)
        for greeting in GREETING_PATTERNS:
            if query_lower.startswith(greeting) or query_lower == greeting:
                return RoutingDecision(
                    destination="skip",
                    confidence=0.9,
                    reason="Greeting detected",
                    source="fallback",
                )
        
        # 2. Causal keywords → Causal engine (CHECK BEFORE MATH!)
        # Priority: confounding, intervention, dag, etc. must route to causal, not mathematical
        if any(keyword in query_lower for keyword in CAUSAL_KEYWORDS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="causal",
                confidence=0.85,
                reason="Causal keywords detected",
                source="fallback",
            )
        
        # 3. Probability notation → Probabilistic engine (CHECK BEFORE MATH!)
        if any(keyword in query_lower for keyword in PROBABILISTIC_KEYWORDS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="probabilistic",
                confidence=0.8,
                reason="Probability notation detected",
                source="fallback",
            )
        
        # 4. Mathematical patterns → Probabilistic/Mathematical engine
        # Check AFTER causal/probabilistic to avoid misclassifying domain-specific queries
        if any(pattern.search(query) for pattern in MATHEMATICAL_PATTERNS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="probabilistic",
                confidence=0.9,
                reason="Mathematical notation detected",
                source="fallback",
            )
        
        # 5. Mathematical keywords → Mathematical engine
        if any(keyword in query_lower for keyword in MATHEMATICAL_KEYWORDS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="mathematical",
                confidence=0.85,
                reason="Mathematical keywords detected",
                source="fallback",
            )
        
        # 6. Logic symbols → Symbolic engine
        if any(symbol in query for symbol in LOGIC_SYMBOLS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="symbolic",
                confidence=0.9,
                reason="Logic symbols detected",
                source="fallback",
            )
        
        # 7. Logic keywords → Symbolic engine
        if any(keyword in query_lower for keyword in LOGIC_KEYWORDS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="symbolic",
                confidence=0.85,
                reason="Logic keywords detected",
                source="fallback",
            )
        
        # 8. Analogical keywords → Analogical engine
        if any(keyword in query_lower for keyword in ANALOGICAL_KEYWORDS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="analogical",
                confidence=0.85,
                reason="Analogical reasoning keywords detected",
                source="fallback",
            )
        
        # 9. Language/Quantifier keywords → Symbolic engine (linguistic reasoning)
        if any(keyword in query_lower for keyword in LANGUAGE_QUANTIFIER_KEYWORDS):
            return RoutingDecision(
                destination="reasoning_engine",
                engine="symbolic",
                confidence=0.85,
                reason="Language/quantifier keywords detected",
                source="fallback",
            )
        
        # 10. Ethical dilemma patterns → WorldModel (NOT self-referential)
        # Check BEFORE self-referential: "You are in a trolley scenario" is NOT about the AI
        if any(pattern.search(query) for pattern in ETHICAL_DILEMMA_PATTERNS):
            return RoutingDecision(
                destination="world_model",
                confidence=0.85,
                reason="Ethical dilemma detected",
                source="fallback",
                metadata={"query_type": "ethical_dilemma"},
            )
        
        # 11. Ethical dilemma keywords → WorldModel
        if any(keyword in query_lower for keyword in ETHICAL_DILEMMA_KEYWORDS):
            return RoutingDecision(
                destination="world_model",
                confidence=0.8,
                reason="Ethical dilemma keywords detected",
                source="fallback",
                metadata={"query_type": "ethical_dilemma"},
            )
        
        # 12. Specific self-referential patterns → WorldModel (meta-reasoning)
        # Only match queries that are actually about the AI itself, not just "you"
        if any(pattern.search(query) for pattern in SELF_REFERENTIAL_PATTERNS_SPECIFIC):
            return RoutingDecision(
                destination="world_model",
                confidence=0.8,
                reason="Self-referential query about AI detected",
                source="fallback",
                metadata={"query_type": "self_introspection"},
            )
        
        # 13. Default: WorldModel (safer than wrong engine)
        return RoutingDecision(
            destination="world_model",
            confidence=0.5,
            reason="Default to WorldModel (fallback)",
            source="fallback",
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get router statistics.
        
        Returns:
            Dictionary with routing statistics including cache stats
        """
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Add cache stats
        stats["cache"] = self.cache.stats()
        
        # Calculate rates
        total = stats["total_queries"] or 1
        stats["cache_hit_rate"] = stats["cache_hits"] / total
        stats["llm_rate"] = stats["llm_classifications"] / total
        stats["fallback_rate"] = stats["fallback_classifications"] / total
        
        return stats
    
    def clear_cache(self) -> int:
        """Clear the routing cache. Returns number of entries cleared."""
        return self.cache.clear()
    
    def set_llm_client(self, llm_client: Any) -> None:
        """
        Set or update the LLM client for this router.
        
        Allows late-binding of the LLM client after initialization.
        Useful when the router is created before the LLM client is available.
        
        Args:
            llm_client: LLM client with chat() method. Can be None to disable LLM.
        """
        was_available = self.llm_client is not None
        self.llm_client = llm_client
        is_available = llm_client is not None
        
        if not was_available and is_available:
            logger.info(f"LLMQueryRouter: LLM client now available (was unavailable)")
        elif was_available and not is_available:
            logger.warning(f"LLMQueryRouter: LLM client removed (falling back to regex)")
        else:
            logger.debug(f"LLMQueryRouter: LLM client updated")


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_llm_router_instance: Optional[LLMQueryRouter] = None
_llm_router_lock = threading.Lock()


def get_llm_router(
    llm_client: Any = None,
    force_new: bool = False,
) -> LLMQueryRouter:
    """
    Get or create the global LLMQueryRouter instance.
    
    Auto-discovers LLM client from available sources if not provided:
    1. vulcan.reasoning.singletons.get_llm_client()
    2. vulcan.llm.get_hybrid_executor() -> local_llm
    3. vulcan.main.global_llm_client (if exists)
    
    Args:
        llm_client: LLM client with chat() method. If None, auto-discovers.
        force_new: Force creation of new instance
        
    Returns:
        LLMQueryRouter singleton instance
    """
    global _llm_router_instance
    
    if _llm_router_instance is None or force_new:
        with _llm_router_lock:
            if _llm_router_instance is None or force_new:
                # Auto-discover LLM client if not provided
                if llm_client is None:
                    llm_client = _discover_llm_client()
                
                _llm_router_instance = LLMQueryRouter(llm_client=llm_client)
    
    return _llm_router_instance


def _discover_llm_client() -> Optional[Any]:
    """
    Auto-discover LLM client from available sources.
    
    Tries multiple sources in priority order:
    1. vulcan.reasoning.singletons.get_llm_client()
    2. vulcan.llm.get_hybrid_executor() -> local_llm
    3. vulcan.main.global_llm_client (if exists)
    
    Returns:
        LLM client instance, or None if unavailable.
    """
    # Try get_llm_client() from singletons
    try:
        from vulcan.reasoning.singletons import get_llm_client
        client = get_llm_client()
        if client is not None:
            logger.info("LLMQueryRouter: ✓ Auto-discovered LLM client from singletons.get_llm_client()")
            return client
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"LLMQueryRouter: Failed to get LLM from singletons: {e}")
    
    # Try get_hybrid_executor() -> local_llm
    try:
        from vulcan.llm import get_hybrid_executor
        hybrid_executor = get_hybrid_executor()
        if hybrid_executor is not None:
            client = getattr(hybrid_executor, 'local_llm', None)
            if client is not None:
                logger.info("LLMQueryRouter: ✓ Auto-discovered LLM client from HybridLLMExecutor.local_llm")
                return client
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"LLMQueryRouter: Failed to get LLM from hybrid executor: {e}")
    
    # Try main.global_llm_client (if exists)
    try:
        from vulcan import main
        if hasattr(main, 'global_llm_client'):
            client = main.global_llm_client
            if client is not None:
                logger.info("LLMQueryRouter: ✓ Auto-discovered LLM client from main.global_llm_client")
                return client
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"LLMQueryRouter: Failed to get LLM from main.global_llm_client: {e}")
    
    logger.warning("LLMQueryRouter: ⚠ No LLM client discovered - router will use regex fallback only")
    return None


def route_query(query: str) -> RoutingDecision:
    """
    Route a query using the global LLMQueryRouter.
    
    Convenience function for simple routing without needing to get the router first.
    
    Args:
        query: The query to route
        
    Returns:
        RoutingDecision with destination, engine, and confidence
    """
    return get_llm_router().route(query)


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Main classes
    "LLMQueryRouter",
    "RoutingDecision",
    "RoutingCache",
    # Enums
    "RoutingDestination",
    "ReasoningEngine",
    # Functions
    "get_llm_router",
    "route_query",
    "strip_query_headers",
    # Constants
    "HEADER_STRIP_PATTERNS",
]
