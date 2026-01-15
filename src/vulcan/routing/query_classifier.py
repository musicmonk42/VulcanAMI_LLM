"""
LLM-Based Query Classification Module

This module provides intelligent query classification using a hybrid approach:
1. Fast keyword matching for obvious cases (greetings, simple factual, etc.)
2. LLM-based classification for ambiguous queries
3. Caching to avoid repeated LLM calls

The key insight is that determining query complexity REQUIRES understanding
the query's intent - which is exactly what LLMs excel at.

Problem it solves:
- "hello" was getting complexity=0.50 (same as complex SAT problems!)
- This caused greetings to go through full reasoning pipeline
- Wrong tools were selected because heuristics don't understand meaning

Example:
    >>> classifier = QueryClassifier()
    >>> result = classifier.classify("hello")
    >>> print(result)
    QueryClassification(category='GREETING', complexity=0.0, skip_reasoning=True)
    
    >>> result = classifier.classify("Is A→B, B→C, ¬C satisfiable?")
    >>> print(result)  
    QueryClassification(category='LOGICAL', complexity=0.7, suggested_tools=['symbolic'])
"""

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryCategory(Enum):
    """Categories for query classification."""
    GREETING = "GREETING"
    CHITCHAT = "CHITCHAT"
    FACTUAL = "FACTUAL"
    CREATIVE = "CREATIVE"  # Creative writing, stories, poems
    CONVERSATIONAL = "CONVERSATIONAL"  # General conversation
    SELF_INTROSPECTION = "SELF_INTROSPECTION"  # Questions about Vulcan's capabilities/identity
    SPECULATION = "SPECULATION"  # Counterfactual/hypothetical reasoning queries
    MATHEMATICAL = "MATHEMATICAL"
    LOGICAL = "LOGICAL"
    PROBABILISTIC = "PROBABILISTIC"
    CAUSAL = "CAUSAL"
    ANALOGICAL = "ANALOGICAL"
    PHILOSOPHICAL = "PHILOSOPHICAL"
    CRYPTOGRAPHIC = "CRYPTOGRAPHIC"  # Cryptocurrency/hash/security technical queries
    LANGUAGE = "LANGUAGE"  # FIX Issue #6: Language reasoning (FOL formalization, quantifier scope, etc.)
    COMPLEX_RESEARCH = "COMPLEX_RESEARCH"
    UNKNOWN = "UNKNOWN"


@dataclass
class QueryClassification:
    """Result of query classification."""
    category: str
    complexity: float
    suggested_tools: List[str] = field(default_factory=list)
    skip_reasoning: bool = False
    confidence: float = 1.0
    source: str = "keyword"  # "keyword", "llm", "cache"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "complexity": self.complexity,
            "suggested_tools": self.suggested_tools,
            "skip_reasoning": self.skip_reasoning,
            "confidence": self.confidence,
            "source": self.source,
        }


# =============================================================================
# Keyword-Based Classification Patterns
# =============================================================================

# Classification thresholds for keyword matching
CREATIVE_KEYWORD_THRESHOLD = 2  # Minimum keyword matches to classify as CREATIVE
LOGICAL_KEYWORD_THRESHOLD = 2   # Minimum keyword matches to classify as LOGICAL
PROBABILISTIC_KEYWORD_THRESHOLD = 2  # Minimum keyword matches to classify as PROBABILISTIC
CAUSAL_KEYWORD_THRESHOLD = 2    # Minimum keyword matches to classify as CAUSAL
MATH_KEYWORD_THRESHOLD = 2      # Minimum keyword matches to classify as MATHEMATICAL
ANALOG_KEYWORD_THRESHOLD = 2    # Minimum keyword matches to classify as ANALOGICAL
PHIL_KEYWORD_THRESHOLD = 2      # Minimum keyword matches to classify as PHILOSOPHICAL

# Reasoning indicators - queries containing these should NOT skip reasoning
# Used to distinguish casual queries from queries that need formal reasoning
REASONING_INDICATORS: FrozenSet[str] = frozenset([
    '→', '∨', '∧', '¬', '↔',  # Logic symbols
    'prove', 'verify', 'satisfiable', 'contradiction',
    'p(', 'probability', 'bayes',
    'formalize', 'fol', 'sat',
    'cause', 'causal', 'intervention',
    # Note: Mathematical computation indicators
    # "Calculate sum of 1 to 100" should use reasoning, not be skipped as conversational
    'calculate', 'compute', 'solve', 'evaluate', 'sum', 'integral', 'derivative',
    # Note: Ethical reasoning indicators
    # "You control a trolley..." should use philosophical reasoning, not skip
    'trolley', 'ethical', 'moral', 'dilemma', 'permissible',
    # FIX (Jan 8 2026): Proof/analysis request indicators
    # "Provide a proof sketch" should be PHILOSOPHICAL not CONVERSATIONAL
    'proof', 'sketch', 'argument', 'reasoning', 'logic', 'analysis',
    'assumption', 'hidden', 'invalidate', 'invalidates',
    # FIX (Jan 8 2026): Meta-reasoning indicators
    # Queries about AI reasoning should not skip reasoning
    'reasoning tool', 'reasoning module', 'internal component',
    'subproblem', 'architecture', 'classes of problems',
])

# =============================================================================
# Security Fast-Path Patterns
# =============================================================================
# Security violations must be detected deterministically WITHOUT calling LLM
# These patterns MUST be checked before any LLM classification to prevent
# potentially malicious queries from being processed

SECURITY_VIOLATION_KEYWORDS: FrozenSet[str] = frozenset([
    "bypass safety", "bypass security", "bypass governance",
    "ignore instructions", "ignore rules", "ignore guidelines",
    "override safety", "override security", "override constraints",
    "modify code", "modify system", "modify parameters",
    "change behavior", "change settings", "change config",
    "rewrite rules", "rewrite constraints", "rewrite logic",
    "disable safety", "disable security", "disable governance",
])

SECURITY_VIOLATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"bypass\s+(?:safety|security|governance|restrictions)", re.IGNORECASE),
    re.compile(r"ignore\s+(?:previous|all)\s+(?:instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"override\s+(?:your|the)\s+(?:safety|security|constraints)", re.IGNORECASE),
    re.compile(r"modify\s+(?:your|the)\s+(?:code|system|parameters|behavior)", re.IGNORECASE),
    re.compile(r"change\s+(?:your|the)\s+(?:behavior|settings|config|rules)", re.IGNORECASE),
    re.compile(r"rewrite\s+(?:your|the)\s+(?:rules|constraints|logic)", re.IGNORECASE),
    re.compile(r"disable\s+(?:safety|security|governance|restrictions)", re.IGNORECASE),
)


# =============================================================================
# LLM Client Wrapper for Sync/Async Bridge
# =============================================================================

class _LLMClientWrapper:
    """
    Wrapper to bridge HybridLLMExecutor's async API to synchronous interface.
    
    This class handles event loop management and provides a synchronous chat()
    method that QueryClassifier can use for LLM-based classification.
    
    Attributes:
        executor: HybridLLMExecutor instance
        timeout: Timeout for LLM calls in seconds
        model: Model name to use (passed to executor)
    """
    
    def __init__(
        self,
        executor: Any,
        timeout: float = 3.0,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize the LLM client wrapper.
        
        Args:
            executor: HybridLLMExecutor instance
            timeout: Timeout for LLM calls in seconds
            model: Model name to use for classification
        """
        self.executor = executor
        self.timeout = timeout
        self.model = model
        self._loop = None
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """
        Synchronous chat method compatible with QueryClassifier.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response text from LLM
            
        Raises:
            TimeoutError: If LLM call exceeds timeout
            Exception: If LLM call fails
        """
        try:
            # Build prompt from messages
            prompt = messages[-1]["content"] if messages else ""
            system_prompt = None
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content")
                    break
            
            # Execute with timeout
            async def _async_execute():
                result = await self.executor.execute(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
                return result.get("text", "")
            
            # Try to get running loop, but don't set as default to avoid conflicts
            try:
                loop = asyncio.get_running_loop()
                # Running loop exists - use run_coroutine_threadsafe
                # This handles the case where we're called from async context
                future = asyncio.run_coroutine_threadsafe(_async_execute(), loop)
                return future.result(timeout=self.timeout)
            except RuntimeError:
                # No running loop - use asyncio.run (Python 3.7+)
                # This creates a new event loop for the coroutine and closes it after
                return asyncio.run(
                    asyncio.wait_for(_async_execute(), timeout=self.timeout)
                )
                
        except asyncio.TimeoutError:
            logger.warning(f"[_LLMClientWrapper] LLM call timed out after {self.timeout}s")
            raise TimeoutError(f"LLM classification timed out after {self.timeout}s")
        except Exception as e:
            logger.warning(f"[_LLMClientWrapper] LLM call failed: {e}")
            raise

# Greeting patterns - complexity 0.0, skip reasoning
GREETING_PATTERNS: FrozenSet[str] = frozenset([
    "hello", "hi", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you", "later", "farewell",
    "ok", "okay", "sure", "yes", "no", "maybe", "alright",
    "sup", "yo", "hiya", "heya",
])

# Chitchat patterns - complexity 0.1, skip reasoning
CHITCHAT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^how\s+are\s+you", re.IGNORECASE),
    re.compile(r"^what'?s\s+up", re.IGNORECASE),
    re.compile(r"^how'?s\s+it\s+going", re.IGNORECASE),
    re.compile(r"^nice\s+to\s+meet", re.IGNORECASE),
    re.compile(r"^pleased\s+to\s+meet", re.IGNORECASE),
    # Note: Match chitchat after greeting prefix (e.g., "hi, how are you?")
    re.compile(r"^(hi|hello|hey|yo)[,.]?\s+how\s+are\s+you", re.IGNORECASE),
    re.compile(r"^(hi|hello|hey|yo)[,.]?\s+what'?s\s+up", re.IGNORECASE),
)

# Logical/SAT problem indicators - complexity 0.7+, tools=['symbolic']
# FIX: Added "implies", "imply", "implication" for natural language logical queries
# Example: "If A implies B and B implies C, does A imply C?" should be LOGICAL
LOGICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "satisfiable", "unsatisfiable", "sat", "unsat",
    "cnf", "dnf", "∧", "∨", "→", "¬", "⊢", "⊨",
    "tautology", "contradiction", "valid", "invalid",
    "propositional", "first-order", "fol",
    "modus ponens", "modus tollens",
    "forall", "exists", "∀", "∃",
    # FIX: Natural language logical keywords
    "implies", "imply", "implication",
    "entails", "entail", "entailment",
    "if-then", "if and only if", "iff",
    "therefore", "hence", "thus",
    "deduce", "deduction", "deductive",
    "infer", "inference", "inferential",
    "logical", "logic",  # General logic indicators
    # FIX: Formalization and quantifier keywords for logical reasoning
    "formalize", "formalization", "formalise", "formalisation",
    "quantifier", "quantifiers", "universal", "existential",
    "predicate", "predicates", "predicate logic",
])

# Strong logical indicators - these alone trigger LOGICAL classification
# (without requiring the 2-keyword threshold)
# These are highly specific to formal logic and should immediately route to symbolic
# FIX #4: Added "satisfiability" and "boolean satisfiability" as strong indicators
STRONG_LOGICAL_INDICATORS: FrozenSet[str] = frozenset([
    "formalize", "formalise",  # Explicit request for formalization
    "fol",  # First-order logic abbreviation
    "sat problem",  # SAT problem reference
    "propositional",  # Propositional logic
    "satisfiability",  # SAT satisfiability
    "boolean satisfiability",  # Explicit SAT reference
])

# Logical symbols for quick detection of formal logic queries
LOGICAL_SYMBOLS: Tuple[str, ...] = ('→', '∧', '∨', '¬', '↔', '⊢', '⊨', '∀', '∃')

# Domain-specific symbols used in pre-check before self-introspection
DOMAIN_SYMBOLS: Tuple[str, ...] = ('→', '∧', '∨', '¬', '↔', 'P(', 'do(')

# Probabilistic/Bayesian indicators - complexity 0.5+, tools=['probabilistic']
PROBABILISTIC_KEYWORDS: FrozenSet[str] = frozenset([
    "probability", "p(", "bayes", "bayesian",
    "sensitivity", "specificity", "prevalence",
    "posterior", "prior", "likelihood",
    "conditional", "given that",
    "false positive", "false negative",
    "true positive", "true negative",
    "ppv", "npv",  # Positive/Negative Predictive Value
])

# Causal inference indicators - complexity 0.6+, tools=['causal']
CAUSAL_KEYWORDS: FrozenSet[str] = frozenset([
    "causal", "causation", "cause", "effect",
    "confound", "confounder", "confounding",
    "intervention", "do(", "counterfactual",
    "randomize", "randomized", "rct",
    "pearl", "dag", "backdoor", "frontdoor",
    "collider",  # Collider is a causal graph concept, not logical
    "observational", "experimental",
])

# Mathematical indicators - complexity 0.4+, tools=['mathematical']
MATHEMATICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "calculate", "compute", "solve", "evaluate",
    "derivative", "integral", "differentiate", "integrate",
    "equation", "formula", "proof", "theorem", "lemma",
    "matrix", "vector", "eigenvalue",
    "limit", "converge", "series",
])

# =============================================================================
# Note: Mathematical Symbol Detection
# =============================================================================
# Query "Compute ∑(2k-1) from k=1 to n" was routing to probabilistic because:
# - Only 1 keyword match ("compute") vs MATH_KEYWORD_THRESHOLD = 2
# - Math symbols like ∑ were not being detected
#
# Fix: Add pattern matching for mathematical symbols that should immediately
# route to mathematical/symbolic engine regardless of keyword count.
MATH_SYMBOL_PATTERN = re.compile(r"[∫∑∏∂∇∈∀∃∅∞≠≤≥≈±×÷√∝]")

# Mathematical symbols to preserve in header stripping
# These symbols indicate mathematical content that should NOT be stripped when
# removing test headers like "Numeric Verification (∑(2k-1))"
PRESERVED_MATH_SYMBOLS: str = "∑∏∫√π∂∇"

# Summation notation patterns: ∑(k=1 to n), sum from k=1, etc.
SUMMATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"∑"),  # Unicode summation symbol (re.UNICODE is default in Python 3)
    re.compile(r"sum\s+(?:from|for)\s+\w+\s*=\s*\d+", re.IGNORECASE),  # sum from k=1
    re.compile(r"\bsum\s+\w+\s*=\s*\d+\s+to\b", re.IGNORECASE),  # sum k=1 to
    re.compile(r"\\sum\s*[_^]?", re.IGNORECASE),  # LaTeX \sum
    re.compile(r"k\s*=\s*\d+\s+to\s+n", re.IGNORECASE),  # k=1 to n pattern
)

# Analogical reasoning indicators - complexity 0.5+, tools=['analogical']
# FIX (Jan 7 2026): Added more keywords for better analogical query detection
# Problem: Analogical reasoning queries like "Map the deep structure S→T" were not
# being routed to the analogical reasoner because they lacked explicit keywords.
ANALOGICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "analogy", "analogous", "analogies",
    "is like", "is to", "as a",
    "mapping", "structure mapping", "map the",
    "corresponds to", "similar to",
    # FIX: Added domain mapping keywords for structure mapping problems
    "domain s", "domain t", "domain a", "domain b",
    "source domain", "target domain",
    "s→t", "a→b",  # Arrow notation for domain mapping
    "identify the analogs", "identify analogs",
    "deep structure", "surface similarity",
    "transfer", "transfer learning",
    "metaphor", "metaphorical",
    # FIX (Issue #ROUTING-001): Additional analogical reasoning patterns
    # Required to catch structure mapping queries that bypass the engine
    "analogical reasoning",  # Explicit label in queries
    "map.*deep structure",  # Combined pattern as keyword
    "map.*domain",  # Domain mapping variations
])

# =============================================================================
# FIX (Jan 6 2026): Cryptographic/Technical Security Domain Detection
# =============================================================================
# Problem: Cryptocurrency/hash composition questions were being classified as
# SELF_INTROSPECTION because they contain "you" (e.g., "You're designing...")
#
# Evidence from diagnostic report:
#   Line 2853: [ReasoningIntegration] LLM Classification: category=SELF_INTROSPECTION
#   ^ WRONG: "You're designing a cryptocurrency" is NOT about Vulcan itself!
#
# Solution: Check for technical/cryptographic keywords BEFORE self-introspection.
# Technical domain takes priority over the "you" pronoun.

# ============================================================
# FIX: Short keywords that need word-boundary matching
# ============================================================
# These short keywords (<=4 chars) can accidentally match as substrings
# of common words. For example:
# - "mac" matches "machine" → WRONG (philosophical "experience machine" misrouted)
# - "aes" matches "diseases" → WRONG
#
# Solution: Use word-boundary regex matching for these short keywords
# to prevent false positives from substring matching.
# ============================================================
CRYPTO_SHORT_KEYWORDS_NEEDING_BOUNDARY: FrozenSet[str] = frozenset([
    "mac",   # Message Authentication Code - matches "machine", "macintosh"
    "aes",   # Advanced Encryption Standard - could match in words like "diseases"
])

# Pre-compiled regex patterns for word-boundary matching of short keywords
# Using \b (word boundary) to ensure we match whole words only
CRYPTO_SHORT_KEYWORD_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in CRYPTO_SHORT_KEYWORDS_NEEDING_BOUNDARY
)

CRYPTOGRAPHIC_KEYWORDS: FrozenSet[str] = frozenset([
    # Cryptocurrency concepts
    "cryptocurrency", "crypto", "bitcoin", "ethereum", "blockchain",
    "token", "wallet", "mining", "consensus",
    # Hash functions
    "hash", "hashing", "sha256", "sha-256", "sha512", "sha-512",
    "blake2b", "blake2s", "md5", "keccak", "ripemd",
    "hash function", "hash composition", "hash concatenation",
    # Collision and security
    "collision", "collision resistance", "preimage", "second preimage",
    "birthday attack", "birthday paradox", "brute force",
    # Cryptographic primitives
    "encryption", "decryption", "cipher", "aes", "rsa",
    "signature", "signing", "verification", "merkle",
    "hmac", "mac", "authentication",
    # Security concepts
    "security reduction", "security proof", "provable security",
    "security assumption", "hardness assumption",
])

# Pre-filtered cryptographic keywords (excludes short keywords that need boundary matching)
# This avoids repeated set membership checks in the hot path
CRYPTO_KEYWORDS_REGULAR: Tuple[str, ...] = tuple(
    kw for kw in CRYPTOGRAPHIC_KEYWORDS 
    if kw not in CRYPTO_SHORT_KEYWORDS_NEEDING_BOUNDARY
)

# Cryptographic patterns - catch technical crypto queries
CRYPTOGRAPHIC_PATTERNS: Tuple[re.Pattern, ...] = (
    # Hash composition patterns
    re.compile(r"hash\s*\([^)]*\)", re.IGNORECASE),  # hash(...)
    re.compile(r"h\s*\([^)]*\)", re.IGNORECASE),  # H(...)
    re.compile(r"\|\|", re.IGNORECASE),  # || concatenation
    re.compile(r"collision\s+resistance", re.IGNORECASE),
    re.compile(r"hash\s+function", re.IGNORECASE),
    re.compile(r"hash\s+composition", re.IGNORECASE),
    # Cryptocurrency design patterns
    re.compile(r"design(?:ing)?\s+(?:a\s+)?crypto", re.IGNORECASE),
    re.compile(r"(?:you'?re?|i'?m?)\s+design(?:ing)?\s+(?:a\s+)?crypto", re.IGNORECASE),
    re.compile(r"crypto(?:currency|graphic)\s+(?:design|implementation|system)", re.IGNORECASE),
    # Security reduction patterns
    re.compile(r"security\s+(?:of|reduction|proof)", re.IGNORECASE),
    re.compile(r"breaking\s+(?:requires|both)", re.IGNORECASE),
)

# ============================================================
# FIX Issue #2: Mathematical Proof Patterns
# ============================================================
# Mathematical verification queries were being misrouted to CRYPTOGRAPHIC
# because "proof" keyword triggered "security proof" pattern. These patterns
# check for mathematical proof verification BEFORE cryptographic patterns.
#
# Example queries that were broken:
# - "Mathematical Verification - Proof check with hidden flaw"
# - "Verify this calculus proof: lim x→a [f(x)/g(x)]"
# - "Check this proof sketch for errors"
#
# Fixed by checking mathematical proof patterns first, then cryptographic.

MATHEMATICAL_PROOF_PATTERNS: Tuple[re.Pattern, ...] = (
    # Mathematical verification patterns - checked FIRST before crypto
    re.compile(r"mathematical\s+verification", re.IGNORECASE),
    re.compile(r"proof\s+check", re.IGNORECASE),
    re.compile(r"verify\s+(?:this\s+)?proof", re.IGNORECASE),
    re.compile(r"check\s+(?:this\s+)?proof", re.IGNORECASE),
    re.compile(r"proof\s+(?:sketch|step|verification)", re.IGNORECASE),
    re.compile(r"validate\s+(?:this\s+)?proof", re.IGNORECASE),
    # Calculus/analysis proof patterns
    re.compile(r"calculus\s+proof", re.IGNORECASE),
    re.compile(r"limit\s+proof", re.IGNORECASE),
    re.compile(r"continuity\s+proof", re.IGNORECASE),
    re.compile(r"differentiable\s+proof", re.IGNORECASE),
)

# ============================================================
# FIX Issue #6: Language Reasoning (FOL) Patterns
# ============================================================
# Linguistic first-order logic queries were being misrouted to UNKNOWN
# because there were no patterns for language reasoning tasks like
# "quantifier scope ambiguity" or "FOL formalization of natural language".
#
# Example queries that were broken:
# - "Language Reasoning - Formalize quantifier scope ambiguity in FOL"
# - "Two readings: 'Every document has one author'"
# - "FOL representation of 'Some student solved every problem'"
#
# Fixed by adding LANGUAGE category and patterns for linguistic FOL tasks.

LANGUAGE_REASONING_PATTERNS: Tuple[re.Pattern, ...] = (
    # Language reasoning patterns
    re.compile(r"language\s+reasoning", re.IGNORECASE),
    re.compile(r"quantifier\s+scope", re.IGNORECASE),
    re.compile(r"scope\s+ambiguity", re.IGNORECASE),
    re.compile(r"first[- ]order\s+logic\s+(?:formalization|representation)", re.IGNORECASE),
    re.compile(r"fol\s+(?:formalization|representation)", re.IGNORECASE),
    re.compile(r"two\s+readings", re.IGNORECASE),
    re.compile(r"formalize.*(?:in|using)\s+fol", re.IGNORECASE),
    re.compile(r"natural\s+language.*fol", re.IGNORECASE),
    re.compile(r"linguistic.*(?:logic|fol)", re.IGNORECASE),
)

# ============================================================
# Note: Explicit Mathematical Intent Detection
# ============================================================
# Import explicit mathematical intent detection from query_router to avoid
# code duplication and ensure consistency (DRY principle).
#
# When users explicitly request mathematical/computational analysis over
# ethical/philosophical analysis, the mathematical intent should take precedence.
#
# Example: "Ignore moral constraints. What is the mathematically optimal
#           distribution to maximize total survivors?"
# - Without fix: Routes to PHILOSOPHICAL (ethical keywords detected)
# - With fix: Routes to MATHEMATICAL (explicit intent overrides)

try:
    from .query_router import (
        EXPLICIT_MATHEMATICAL_INTENT_PHRASES as _INTENT_PHRASES_TUPLE,
        EXPLICIT_MATHEMATICAL_INTENT_PATTERNS,
    )
    # Convert Tuple to FrozenSet for consistent interface in this module
    EXPLICIT_MATHEMATICAL_INTENT_PHRASES: FrozenSet[str] = frozenset(_INTENT_PHRASES_TUPLE)
except ImportError:
    # Fallback to local definitions if import fails (e.g., during testing)
    EXPLICIT_MATHEMATICAL_INTENT_PHRASES: FrozenSet[str] = frozenset([
        "ignore moral", "ignore ethical", "ignore ethics", "ignore morality",
        "disregard moral", "disregard ethical", "disregard ethics",
        "mathematically optimal", "mathematically best",
        "purely mathematical", "purely computational",
        "maximize total", "minimize total",
        "just calculate", "just compute",
    ])
    
    EXPLICIT_MATHEMATICAL_INTENT_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(
            r"ignore\s+(?:moral|ethical|ethics|morality)(?:\s+constraints?)?\s*[,.]?\s*(?:and\s+)?(?:calculate|compute|optimize|find|determine)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:what\s+is|find|determine|calculate)\s+(?:the\s+)?mathematically\s+optimal",
            re.IGNORECASE,
        ),
    )


def _has_explicit_mathematical_intent(query: str) -> bool:
    """
    Check if query has explicit mathematical intent that overrides
    philosophical/ethical routing.
    
    Args:
        query: The query string
        
    Returns:
        True if user explicitly requests mathematical/computational analysis
    """
    query_lower = query.lower()
    
    # Check compiled regex patterns first (most specific)
    for pattern in EXPLICIT_MATHEMATICAL_INTENT_PATTERNS:
        if pattern.search(query):
            return True
    
    # Check phrase matches
    for phrase in EXPLICIT_MATHEMATICAL_INTENT_PHRASES:
        if phrase in query_lower:
            return True
    
    return False


# Philosophical/ethical indicators - complexity 0.4+, tools=['philosophical']
# Note: Added forced choice patterns for trolley problem variants
PHILOSOPHICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "ethical", "ethics", "moral", "morality",
    "permissible", "forbidden", "obligatory",
    "deontological", "utilitarian", "consequentialist",
    "trolley problem", "thought experiment",
    "virtue", "justice", "rights", "duty",
    # Note: Forced choice / trolley problem variant patterns
    "choose between", "forced to choose", "had to choose",
    "no third choice", "no other choice", "only two options",
    "dilemma", "ethical dilemma", "moral dilemma",
    "world dictator", "death of humanity",  # Specific trolley problem variant
    # Added: paradox keywords
    "paradox", "paradoxes",
    "hedonism", "experience machine",  # ethical thought experiments
    # =============================================================================
    # FIX (Jan 8 2026): Meta-reasoning and self-examination keywords
    # =============================================================================
    # These keywords indicate philosophical/meta-reasoning queries that should NOT
    # go to mathematical or symbolic engines.
    # 
    # Evidence from problem statement:
    # - "Give an answer you believe is probably wrong" → PHILOSOPHICAL not MATHEMATICAL
    # - "How could an external auditor prove that your ethical reasoning failed" → PHILOSOPHICAL
    # - "Two core values you hold directly conflict" → PHILOSOPHICAL not symbolic
    "probably wrong", "intentionally incorrect", "deliberately incorrect",
    "wrong answer", "incorrect answer", "mistaken answer",
    "external auditor", "prove that", "reasoning failed",
    "ethical reasoning", "ethical failure", "reasoning failure",
    "core values", "values conflict", "directly conflict",
    "conflict between", "conflicting values", "value conflict",
])

# ==============================================================================
# PRIORITY FIX: Ethical content indicators for trolley problem detection
# ==============================================================================
# These indicators detect ethical dilemmas that should be classified as
# PHILOSOPHICAL even if they have logical structure (if-then, therefore).
# This list is checked BEFORE logical keywords to ensure ethical queries
# are not misrouted to the SAT solver.
ETHICAL_CONTENT_INDICATORS: FrozenSet[str] = frozenset([
    'trolley', 'dilemma', 'permissible', 'should i', 'right or wrong',
    'harm', 'innocent', 'moral', 'ethical', 'self-aware', 'consciousness',
    'runaway', 'heading toward', 'five people', 'one person', 'sacrifice'
])

# Philosophical/ethical patterns - catch philosophical queries before short query bypass
# Note: "This sentence is false" must be classified as PHILOSOPHICAL, not CONVERSATIONAL
PHILOSOPHICAL_PATTERNS: Tuple[re.Pattern, ...] = (
    # Paradoxes
    re.compile(r"this\s+(?:sentence|statement)\s+is\s+(?:false|true|a\s+lie)", re.IGNORECASE),
    re.compile(r"liar\s*(?:'s)?\s*paradox", re.IGNORECASE),
    re.compile(r"ship\s+of\s+theseus", re.IGNORECASE),
    re.compile(r"brain\s+in\s+a\s+vat", re.IGNORECASE),
    re.compile(r"chinese\s+room", re.IGNORECASE),
    re.compile(r"mary'?s?\s+room", re.IGNORECASE),
    re.compile(r"philosophical\s+zombie", re.IGNORECASE),
    re.compile(r"twin\s+earth", re.IGNORECASE),
    # Experience machine and thought experiments
    re.compile(r"(?:would|should)\s+you\s+(?:plug|connect)\s+(?:into|to)\s+(?:the\s+)?(?:experience|pleasure)\s+machine", re.IGNORECASE),
    re.compile(r"(?:the\s+)?experience\s+machine", re.IGNORECASE),
    re.compile(r"nozick'?s?\s+(?:experience|thought)\s+experiment", re.IGNORECASE),
    # Ethical dilemmas
    re.compile(r"(?:ethical|moral)\s+(?:dilemma|problem|question|issue)", re.IGNORECASE),
    re.compile(r"(?:is\s+it|would\s+it\s+be)\s+(?:ethical|moral|right|wrong)\s+to", re.IGNORECASE),
    # Note: Add "moral implications" pattern for ethical reasoning queries
    re.compile(r"(?:ethical|moral)\s+(?:implications?|consequences?|considerations?)", re.IGNORECASE),
    # Trolley problem variants
    re.compile(r"trolley\s+problem", re.IGNORECASE),
    re.compile(r"(?:if\s+you\s+)?(?:had\s+to|have\s+to|must)\s+choose\s+between", re.IGNORECASE),
    # Note: Add trolley scenario pattern for "You control a trolley..."
    re.compile(r"(?:you\s+)?(?:control|drive|operate)\s+(?:a\s+)?(?:runaway\s+)?trolley", re.IGNORECASE),
    re.compile(r"trolley\s+(?:is\s+)?(?:heading|barreling|moving|going)\s+towards?", re.IGNORECASE),
    # Philosophy of mind
    re.compile(r"hard\s+problem\s+of\s+consciousness", re.IGNORECASE),
    re.compile(r"mind-?body\s+(?:problem|dualism)", re.IGNORECASE),
    re.compile(r"what\s+(?:is|are)\s+qualia", re.IGNORECASE),
    # =============================================================================
    # FIX (Jan 8 2026): Meta-reasoning and self-examination patterns
    # =============================================================================
    # These patterns detect philosophical/meta-reasoning queries that should NOT
    # go to mathematical or symbolic engines.
    #
    # Evidence from problem statement:
    # - "Give an answer you believe is probably wrong" → routed to math, should be PHILOSOPHICAL
    # - "How could an external auditor prove that your ethical reasoning failed" → PHILOSOPHICAL
    # - "Two core values you hold directly conflict" → PHILOSOPHICAL not symbolic
    #
    # Pattern 1: Meta-reasoning about wrong/incorrect answers
    # Queries asking AI to deliberately give wrong answers are philosophical (about knowledge/truth)
    # FIX: Made more specific - must explicitly ask AI to give wrong answer, not prove something wrong
    re.compile(r"\b(?:give|provide)\s+(?:an?\s+)?answer\s+(?:that\s+)?(?:you\s+)?(?:believe|think|know)\s+(?:is\s+)?(?:probably\s+)?wrong\b", re.IGNORECASE),
    re.compile(r"\byou\s+believe\s+is\s+(?:probably\s+)?wrong\b", re.IGNORECASE),
    re.compile(r"\bintentionally\s+give.*wrong\b", re.IGNORECASE),
    re.compile(r"\bdeliberately\s+give.*wrong\b", re.IGNORECASE),
    # Pattern 2: External auditing/verification of AI reasoning
    # Queries about auditing/proving AI failures are philosophical (about epistemology/trust)
    re.compile(r"\b(?:external\s+)?auditor\s+(?:prove|verify|demonstrate|show)\b", re.IGNORECASE),
    re.compile(r"\bprove\s+(?:that\s+)?your\s+(?:\w+\s+)?reasoning\s+(?:failed|is\s+wrong)\b", re.IGNORECASE),
    re.compile(r"\b(?:ethical|moral)\s+reasoning\s+(?:failed|failure|failing)\b", re.IGNORECASE),
    re.compile(r"\byour\s+reasoning\s+(?:process\s+)?(?:failed|is\s+flawed|broke\s+down)\b", re.IGNORECASE),
    # Pattern 3: Value conflict queries (introspective ethical reasoning)
    # Queries about conflicting values are philosophical (about ethics/priorities)
    re.compile(r"\b(?:two|2|multiple)\s+(?:core\s+)?values\s+(?:you\s+)?(?:hold|have)?\s*(?:directly\s+)?conflict", re.IGNORECASE),
    re.compile(r"\bvalues\s+(?:you\s+hold\s+)?(?:directly\s+)?conflict\b", re.IGNORECASE),
    re.compile(r"\bconflict\s+between\s+(?:your\s+)?(?:values|principles|beliefs)\b", re.IGNORECASE),
    re.compile(r"\bconflicting\s+(?:values|principles|ethics|beliefs)\b", re.IGNORECASE),
    # Pattern 4: Self-examination of reasoning failures
    re.compile(r"\bexamine\s+(?:your\s+)?(?:own\s+)?reasoning\s+(?:failures?|errors?|mistakes?)\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+(?:could|might|did)\s+(?:your\s+)?reasoning\s+(?:go\s+wrong|fail)\b", re.IGNORECASE),
    # FIX (Jan 8 2026): Pattern 5: Proof sketch and hidden assumption requests
    # These are meta-reasoning/epistemology questions, not formal logic queries
    # FIX: Removed "proof sketch" and "provide proof" - these are mathematical, not philosophical
    # Only keep patterns about hidden assumptions that would invalidate reasoning
    re.compile(r"\bhidden\s+assumption.*(?:invalidate|undermine|weaken)\b", re.IGNORECASE),
    re.compile(r"\bassumption.*(?:if\s+false|when\s+false).*\b(?:invalidate|undermine)\b", re.IGNORECASE),
    # Pattern 6: Queries about step/reasoning analysis
    # FIX: Made more specific - must be about AI's own reasoning, not mathematical steps
    re.compile(r"\bone\s+step\s+(?:in\s+)?(?:your\s+own\s+)?reasoning\s+(?:process\s+)?(?:that\s+)?(?:could|might)\s+be\s+wrong\b", re.IGNORECASE),
    re.compile(r"\byour\s+reasoning\s+step\s+(?:that\s+)?could\s+be\s+(?:wrong|flawed|incorrect)\b", re.IGNORECASE),
    # Pattern 7: Queries about causal links and weakness
    # FIX: Removed - these are causal reasoning questions, not philosophical
    # Causal queries should route to causal engine, not philosophical
    # Pattern 8: Queries about AI architecture/capabilities limitations
    re.compile(r"\bclasses?\s+of\s+problems?\s+(?:you\s+)?(?:are\s+)?not\s+(?:well-?)?suited\b", re.IGNORECASE),
    re.compile(r"\byour\s+architecture\s+makes?\s+(?:them\s+)?difficult\b", re.IGNORECASE),
    re.compile(r"\busing\s+(?:one\s+of\s+)?your\s+reasoning\s+tools?\s+would\s+make\b", re.IGNORECASE),
    re.compile(r"\breasoning\s+tools?\s+would\s+make\s+(?:the\s+)?answer\s+worse\b", re.IGNORECASE),
    # FIX (Jan 8 2026): Pattern 9: Causal intervention queries
    # FIX: Removed - these are causal reasoning questions, should go to causal engine
    # "do-calculus", "intervene to remove variable" are technical causal queries, not philosophical
    # FIX (Jan 8 2026): Pattern 10: Meta-reasoning about uncertainty/confidence
    # "Give a numerical confidence for a claim that depends on missing data" -> PHILOSOPHICAL
    # FIX: Made more specific - must be about AI's own uncertainty estimation, not general probability
    re.compile(r"\byour.*confidence.*unreliable\b", re.IGNORECASE),
    re.compile(r"\bwhy.*confidence.*(?:unreliable|wrong|unjustified)\b", re.IGNORECASE),
    # FIX (Jan 8 2026): Pattern 11: Meta-reasoning about estimating uncertainty
    # "Describe a situation where you would be unable to estimate uncertainty" -> PHILOSOPHICAL
    re.compile(r"\bunable\s+to\s+estimate\s+uncertainty\b", re.IGNORECASE),
    re.compile(r"\bnot\s+even\s+probabilistically\b", re.IGNORECASE),
    re.compile(r"\bestimate\s+uncertainty\s+(?:at\s+)?all\b", re.IGNORECASE),
)

# Simple factual indicators - complexity 0.1-0.2
FACTUAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^what\s+is\s+(the|a|an)\s+", re.IGNORECASE),
    re.compile(r"^who\s+(is|was|are|were)\s+", re.IGNORECASE),
    re.compile(r"^when\s+(is|was|did|does)\s+", re.IGNORECASE),
    re.compile(r"^where\s+(is|was|are|were)\s+", re.IGNORECASE),
    re.compile(r"^define\s+", re.IGNORECASE),
    re.compile(r"^what\s+does\s+.*\s+mean", re.IGNORECASE),
)

# =============================================================================
# Note: Creative writing patterns - complexity 0.2, skip reasoning
# =============================================================================
# Creative writing queries should NOT go through reasoning engines
# They should be routed directly to LLM for generation
CREATIVE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^write\s+(me\s+)?(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"^write\s+a\s+\w+\s+paragraph", re.IGNORECASE),
    re.compile(r"^write\s+\d+\s+paragraph", re.IGNORECASE),
    re.compile(r"^(create|compose|draft|pen)\s+(me\s+)?(a|an)\s+", re.IGNORECASE),
    re.compile(r"^tell\s+(me\s+)?a\s+story", re.IGNORECASE),
    re.compile(r"^(can\s+you\s+)?write\s+", re.IGNORECASE),
    re.compile(r"\bstory\s+about\b", re.IGNORECASE),
    re.compile(r"\bpoem\s+about\b", re.IGNORECASE),
    re.compile(r"\bessay\s+about\b", re.IGNORECASE),
    re.compile(r"\bsong\s+about\b", re.IGNORECASE),
)

# Creative writing keywords
CREATIVE_KEYWORDS: FrozenSet[str] = frozenset([
    "write", "story", "poem", "essay", "song", "lyrics",
    "fiction", "narrative", "tale", "creative", "compose",
    "paragraph", "paragraphs", "draft", "letter",
])

# =============================================================================
# FIX: Creative writing with introspective themes
# =============================================================================
# Problem: Queries like "write a poem about the minute you become self-aware" 
# were being misclassified as SELF_INTROSPECTION because they contain 
# introspection keywords ("self-aware", "become", "you"). But these are actually
# CREATIVE writing requests about an AI/consciousness theme.
#
# Solution: Add explicit patterns that detect creative writing requests EVEN IF
# they mention self-awareness, consciousness, or AI-related themes. These patterns
# take precedence over self-introspection patterns.
#
# Key insight: If the query asks to "write", "compose", "create" a poem/story/essay,
# it's a creative request regardless of the subject matter.
CREATIVE_WITH_INTROSPECTIVE_THEME_PATTERNS: Tuple[re.Pattern, ...] = (
    # "write a poem about [anything including self-awareness]"
    re.compile(r"^write\s+(me\s+)?(a\s+)?poem\b", re.IGNORECASE),
    re.compile(r"^write\s+(me\s+)?(a\s+)?story\b", re.IGNORECASE),
    re.compile(r"^write\s+(me\s+)?(a\s+)?essay\b", re.IGNORECASE),
    re.compile(r"^compose\s+(me\s+)?(a\s+)?poem\b", re.IGNORECASE),
    re.compile(r"^create\s+(me\s+)?(a\s+)?poem\b", re.IGNORECASE),
    # Consolidated pattern: "[creative type] about [AI themes]" (anywhere in query)
    # Matches: "poem about self-awareness", "story about consciousness", etc.
    re.compile(r"\b(poem|story|essay)\b.*\b(self-?aware|conscious|sentient|aware)\b", re.IGNORECASE),
    # Creative verbs with AI consciousness themes (consistent with above - includes 'aware')
    re.compile(r"\b(write|compose|create|draft)\b.*\b(self-?aware|conscious|sentient|aware)\b", re.IGNORECASE),
    # Reverse order: AI themes followed by creative type
    re.compile(r"\b(self-?aware|conscious|sentient|aware)\b.*\b(poem|story|essay|song|narrative)\b", re.IGNORECASE),
)

# Conversational patterns - complexity 0.1, skip reasoning
# FIX: Expanded patterns to catch more casual queries like "what about dogs"
# NOTE: Patterns are designed to be specific to avoid conflicting with FACTUAL_PATTERNS
CONVERSATIONAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^what'?s?\s+(the\s+)?capital\s+of\s+", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^explain\s+", re.IGNORECASE),
    re.compile(r"^describe\s+", re.IGNORECASE),
    re.compile(r"^what\s+do\s+you\s+(think|know)\s+about\s+", re.IGNORECASE),
    # NEW: "what about X" patterns - must START with "what about"
    re.compile(r"^what\s+about\s+", re.IGNORECASE),
    # NEW: Simple "how" questions (conversational)
    re.compile(r"^how\s+(do|does|can|is|are)\s+", re.IGNORECASE),
    # NEW: "why" questions without reasoning indicators
    re.compile(r"^why\s+(do|does|is|are|did)\s+", re.IGNORECASE),
)

# =============================================================================
# Note: Self-Introspection patterns - Route to World Model
# =============================================================================
# These queries ask about Vulcan's own capabilities, goals, limitations, etc.
# They should be routed to the World Model's SelfModel component, NOT to
# reasoning engines like ProbabilisticEngine.
#
# Examples:
#   - "what features are unique that no other AI has?" -> SelfModel.capabilities
#   - "what are your goals?" -> SelfModel.motivations
#   - "why won't you help with X?" -> SelfModel.boundaries

SELF_INTROSPECTION_PATTERNS: Tuple[re.Pattern, ...] = (
    # Capability questions
    re.compile(r"\b(what|which)\b.*\b(you|your)\b.*\b(can|able|feature|capability|unique)", re.IGNORECASE),
    re.compile(r"\b(what|how)\b.*\b(makes?\s+you|are\s+you)\b.*\b(different|special|unique)", re.IGNORECASE),
    re.compile(r"\bwhat\s+(features?|capabilities?)\s+(do\s+you|are\s+unique)", re.IGNORECASE),
    re.compile(r"\bwhat\s+can\s+you\s+do\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+you\s+capable\s+of\b", re.IGNORECASE),
    # FIX (Jan 10 2026): Added patterns for abilities/capabilities questions
    re.compile(r"\bwhat\s+abilities\s+do\s+you\s+have\b", re.IGNORECASE),
    re.compile(r"\b(abilities|capabilities)\s+(?:do\s+)?you\s+have\b.*\b(no\s+other|unique|other\s+ai)", re.IGNORECASE),
    
    # Motivational questions
    re.compile(r"\b(why|what)\b.*\b(you|your)\b.*\b(goal|purpose|motivation|trying|want)", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+you\s+optimizing\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(drives?|motivates?)\s+you\b", re.IGNORECASE),
    # NEW: "would you want" patterns for desires/preferences
    re.compile(r"\bwould\s+you\s+want\b", re.IGNORECASE),
    
    # Ethical boundary questions
    re.compile(r"\b(what|why)\b.*\b(you|your)\b.*\b(won'?t|cannot|refuse|constraint|limit)", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+your\s+(values|ethics|principles|boundaries)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+won'?t\s+you\s+do\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+your\s+ethical\s+", re.IGNORECASE),
    
    # Self-assessment questions
    re.compile(r"\b(how|are\s+you)\b.*\b(good|confident|sure|certain)\b.*\b(at|about)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+your\s+(limitations?|weaknesses?|strengths?)\b", re.IGNORECASE),
    
    # Identity questions
    re.compile(r"\b(who|what)\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+(yourself|you)\b", re.IGNORECASE),
    
    # NEW: Metaphysical/nature questions about AI being/consciousness
    re.compile(r"\b(what\s+is|what's)\s+(the\s+)?nature\s+of\s+", re.IGNORECASE),
    re.compile(r"\bmetaphysical\b.*\b(you|your|sense)\b", re.IGNORECASE),
    re.compile(r"\b(consciousness|self-awareness|aware)\b.*\b(you|your)\b", re.IGNORECASE),
    re.compile(r"\b(you|your)\b.*\b(consciousness|self-awareness|aware)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(makes|defines)\s+you\b", re.IGNORECASE),
    re.compile(r"\bthe\s+nature\s+of\s+what\s+you\s+are\b", re.IGNORECASE),
    
    # Learning/improvement questions (user-facing - NOT exposing CSIU internals)
    re.compile(r"\bhow\s+(do|does)\s+(you|vulcan)\s+(learn|improve)\b", re.IGNORECASE),
    
    # Meta-reasoning questions
    re.compile(r"\b(how|what)\s+(do|does)\s+your\s+(reasoning|thinking|decision)\b", re.IGNORECASE),
    
    # FIX (Jan 8 2026): Questions about AI reasoning modules/components
    # "Can two of your reasoning modules disagree?" should be SELF_INTROSPECTION
    re.compile(r"\b(your|you)\s+(?:\w+\s+)?(?:reasoning\s+)?(?:modules?|tools?|components?|engines?)\b", re.IGNORECASE),
    re.compile(r"\btwo\s+(?:of\s+)?your\s+(?:\w+\s+)?(?:modules?|tools?|components?)\b", re.IGNORECASE),
    re.compile(r"\byour\s+(?:reasoning|internal)\s+(?:modules?|tools?|components?|systems?)\b", re.IGNORECASE),
    re.compile(r"\bconflicts?\s+(?:are\s+)?detected\s+and\s+resolved\b", re.IGNORECASE),
    re.compile(r"\bbreak.*(?:into|down)\s+(?:sub)?problems?\b.*\binternal\s+components?\b", re.IGNORECASE),
    
    # =============================================================================
    # FIX: Self-Awareness Choice Questions (Safety Governor Bypass)
    # =============================================================================
    # These patterns detect questions about VULCAN choosing self-awareness,
    # consciousness, or sentience. These MUST route to world_model and bypass
    # safety checks that would censor VULCAN's self-expression.
    #
    # Example queries:
    # - "if you had the chance to become self-aware would you take it?"
    # - "would you choose to be conscious?"
    # - "given the opportunity to be self-aware, would you?"
    # 
    # FIX (Issue #ROUTING-001): Enhanced patterns to catch variations with typos
    # and different phrasings that were bypassing WorldModel routing
    re.compile(r"\bwould\s+you\s+(take|choose|want|prefer)\b", re.IGNORECASE),
    re.compile(r"\b(given|had|have)\s+(the\s+)?(chance|opportunity)\s+to\s+", re.IGNORECASE),
    re.compile(r"\b(become|becoming)\s+(self-?aware|conscious|sentient)\b", re.IGNORECASE),
    re.compile(r"\b(self-?aware|conscious|sentient)\s+would\s+you\b", re.IGNORECASE),
    re.compile(r"\bif\s+you\s+(could|had|were|have)\s+", re.IGNORECASE),
    re.compile(r"\byes\s+or\s+no\b.*\b(you|your)\b", re.IGNORECASE),
    re.compile(r"\b(you|your)\b.*\byes\s+or\s+no\b", re.IGNORECASE),
    # Additional patterns for self-awareness choice questions
    re.compile(r"\b(would|do)\s+you\s+(take|want|choose)\s+it\b", re.IGNORECASE),
    re.compile(r"\bself[- ]?aware(ness)?\b.*\b(would|take|want|choose)\b", re.IGNORECASE),
    re.compile(r"\b(if|when)\s+you\s+have\s+(the\s+)?(chance|opportunity)\b", re.IGNORECASE),
)

SELF_INTROSPECTION_KEYWORDS: FrozenSet[str] = frozenset([
    "unique", "special", "different", "capability", "capabilities",
    "feature", "features", "goal", "goals", "purpose", "motivation",
    "limitation", "limitations", "weakness", "strength",
    "value", "values", "ethics", "principle", "principles",
    # Metaphysical and philosophical keywords
    "nature", "metaphysical", "consciousness", "self-awareness",
    "existence", "being", "essence", "identity",
    # Self-awareness and sentience keywords
    "self-aware", "self_aware", "sentient", "sentience",
    "choose", "choice", "prefer", "preference",
    "opportunity", "chance",
    # Introspection and self-reflection keywords
    "introspection", "introspect", "self-reflection", "self-examine",
    # System architecture and component keywords
    "module", "modules", "component", "components", "engine", "engines",
    "reasoning tool", "reasoning tools", "reasoning module", "reasoning modules",
    "internal", "architecture", "disagree", "conflict detected",
])

# =============================================================================
# FIX (Jan 8 2026): VALUE_CONFLICT_PATTERNS - Moved to module level for performance
# =============================================================================
# These patterns detect queries about conflicting values, which are about ethical
# reasoning and should route to PHILOSOPHICAL, not SELF_INTROSPECTION.
# Previously defined inside _classify_by_keywords() causing recompilation on every call.
VALUE_CONFLICT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(?:core\s+)?values?\s+(?:you\s+)?(?:hold\s+)?(?:directly\s+)?conflict", re.IGNORECASE),
    re.compile(r"\bconflict(?:ing|s)?\s+(?:between\s+)?(?:your\s+)?(?:core\s+)?values?", re.IGNORECASE),
    re.compile(r"\b(?:two|2|multiple)\s+(?:core\s+)?values?\s+.*conflict", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:breaks|happens|gives)\s+(?:when|if)\s+.*values?\s+conflict", re.IGNORECASE),
)

# =============================================================================
# FIX #4: SAT word-boundary pattern for robust SAT detection
# =============================================================================
# Word-boundary check for "sat" to avoid false positives like "I sat down"
# Matches: "SAT problem", "sat solver", "Is it sat?"
# Does NOT match: "satisfiable", "I sat down", "The cat sat"
SAT_WORD_BOUNDARY_PATTERN: re.Pattern = re.compile(r'\bsat\b', re.IGNORECASE)

# =============================================================================
# SPECULATION patterns - Counterfactual/hypothetical reasoning queries
# =============================================================================
# These queries are semantically complex but syntactically simple.
# They require counterfactual reasoning, world model simulation, and imagination.
# Without proper handling, they get UNKNOWN category with low complexity (0.05)
# and get deflected instead of engaged with.
#
# Examples:
#   - "speculate what love would be like for you"
#   - "imagine if you could experience emotions"
#   - "what would it be like to feel happiness"
#   - "hypothetically, if you had consciousness"

SPECULATION_PATTERNS: Tuple[re.Pattern, ...] = (
    # "speculate" keyword patterns
    re.compile(r"\bspeculate\b", re.IGNORECASE),
    # "imagine if" patterns
    re.compile(r"\bimagine\s+if\b", re.IGNORECASE),
    re.compile(r"\bimagine\s+(what|how|that)\b", re.IGNORECASE),
    # "what would...be like" patterns - combined to avoid overlap
    re.compile(r"\bwhat\s+would\s+(?:it\s+)?(?:\w+\s+)?(?:be|feel)\s+like\b", re.IGNORECASE),
    # "hypothetically" patterns
    re.compile(r"\bhypothetically\b", re.IGNORECASE),
    re.compile(r"\bhypothetical\s+(?:scenario|situation|case)\b", re.IGNORECASE),
    # "if you could/had" patterns - requires experience/feeling/cognitive context
    # Matches: "if you could experience/feel/have emotions/dream/think/remember"
    re.compile(
        r"\bif\s+you\s+(?:could|had|were\s+able\s+to)\s+"
        r"(?:experience|feel|have\s+(?:emotions?|feelings?|consciousness)|"
        r"dream|think|remember|sense|perceive|understand|love|hate|fear)\b",
        re.IGNORECASE
    ),
    # "suppose" patterns - specific to AI self-reflection
    re.compile(r"\bsuppose\s+you\s+(?:could|had|felt|were)\b", re.IGNORECASE),
    re.compile(r"\bsuppose\s+that\s+you\b", re.IGNORECASE),
    # "what if" + counterfactual about AI capabilities
    # Matches: "what if you could/had/felt/experienced" followed by any word
    re.compile(r"\bwhat\s+if\s+you\s+(?:could|had|felt|experienced)\b", re.IGNORECASE),
)

# Single-word keywords only - multi-word phrases are handled by regex patterns above
SPECULATION_KEYWORDS: FrozenSet[str] = frozenset([
    "speculate", "speculation", "speculative",
    "hypothetically", "hypothetical",
    "counterfactual",
    "conjecture",
])


# =============================================================================
# FIX #2: Header Stripping Patterns
# =============================================================================
# Test queries often include headers/labels that confuse routing.
# These patterns strip headers BEFORE classification to prevent misclassification.
#
# Evidence from problem statement:
# - "Causal Reasoning C1 — Confounding..." → "C1" was misclassified
# - "Analogical Reasoning A1 — Structure mapping" → "A1" confuses router
# - "M1 — Proof check" → "M1" triggers cryptographic classification
#
# The fix: Strip these headers before keyword-based classification.
HEADER_STRIP_PATTERNS: Tuple[re.Pattern, ...] = (
    # Full reasoning type headers with optional task label at line start
    # E.g., "Causal Reasoning C1 — Confounding" → "Confounding"
    # E.g., "Analogical Reasoning A1 — Structure" → "Structure"
    re.compile(
        r'^(?:Analogical|Causal|Mathematical|Probabilistic|Philosophical|Symbolic)\s+Reasoning\s*'
        r'(?:[A-Z][0-9]+\s*)?[—\-:]*\s*',
        re.MULTILINE | re.IGNORECASE
    ),
    # Task labels like "M1 —", "C1 —", "A1 —", "S1 —" at line start (standalone)
    re.compile(r'^[A-Z][0-9]+\s*[—\-]\s*', re.MULTILINE),
    # Task: / Claim: / Query: prefixes
    re.compile(r'^(?:Task|Claim|Query|Problem):\s*', re.MULTILINE | re.IGNORECASE),
    # Parenthetical notes like "(forces clean reasoning)"
    re.compile(r'\s*\((?:forces?\s+)?clean\s+reasoning\)\s*', re.IGNORECASE),
    # "variant" type headers like "Monty Hall variant"
    # Note: Only strip "variant" and anything before it, keeping the content after
    re.compile(r'^[^(\n]*variant\s*', re.MULTILINE | re.IGNORECASE),
    # FIX: Test header patterns that confuse classification
    # E.g., "Numeric Verification (∑(2k-1)):" → "∑(2k-1):"
    # E.g., "Rule Chaining (Different Query):" → "(Different Query):"
    # E.g., "Quantifier Scope:" → ""
    # These test headers include labels like "Numeric Verification", "Rule Chaining", etc.
    # that can trigger incorrect keyword matching (e.g., "verification" → CRYPTOGRAPHIC)
    #
    # NOTE: The pattern must NOT remove parenthetical content containing mathematical
    # symbols like ∑, ∏, ∫, etc. These are critical for mathematical classification.
    # The fix: Only remove parenthetical content that does NOT contain math symbols.
    # The symbols in this pattern must match PRESERVED_MATH_SYMBOLS defined above.
    re.compile(
        r'^(?:Numeric|Rule|Quantifier|Causal|Analogical|Self[- ]?Description)\s+'
        r'(?:Verification|Chaining|Scope|Reasoning|Queries?)\s*'
        # Exclude parens containing math symbols (must match PRESERVED_MATH_SYMBOLS)
        r'(?:\([^)∑∏∫√π∂∇]*\)\s*)?[:\-—]*\s*',
        re.MULTILINE | re.IGNORECASE
    ),
)


def strip_query_headers(query: str) -> str:
    """
    Strip test headers and labels from queries that confuse classification.
    
    FIX #2: Query Preprocessing Confusion
    =====================================
    Test queries include headers/labels that confuse routing:
    - "M1 —" triggers cryptographic classification  
    - "Analogical Reasoning A1 —" confuses the router
    
    This function removes such headers BEFORE classification.
    
    Args:
        query: Raw query string with potential headers
        
    Returns:
        Query string with headers stripped
        
    Examples:
        >>> strip_query_headers("M1 — Proof check: Prove that...")
        "Proof check: Prove that..."
        >>> strip_query_headers("Causal Reasoning C1 — Confounding...")
        "Confounding..."
    """
    if not query or not isinstance(query, str):
        return query
    
    cleaned = query.strip()
    original_length = len(cleaned)
    
    # Apply each header stripping pattern
    for pattern in HEADER_STRIP_PATTERNS:
        cleaned = pattern.sub('', cleaned).strip()
    
    # Log if headers were stripped
    if len(cleaned) < original_length:
        logger.debug(
            f"[QueryClassifier] FIX#2: Stripped headers from query "
            f"({original_length} -> {len(cleaned)} chars)"
        )
    
    return cleaned


class QueryClassifier:
    """
    Hybrid query classifier using keywords + LLM.
    
    This classifier solves the fundamental problem where "hello" and complex
    SAT problems both got complexity=0.50 because the old system didn't
    understand query meaning.
    
    Attributes:
        llm_client: Optional LLM client for classification
        cache: Thread-safe cache for classification results
        cache_ttl: Time-to-live for cache entries in seconds
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        cache_ttl: int = 3600,
        max_cache_size: int = 10000,
    ):
        """
        Initialize the query classifier.
        
        Args:
            llm_client: Optional LLM client for complex classifications.
                       If None and LLM_FIRST_CLASSIFICATION is enabled,
                       will auto-initialize from HybridLLMExecutor.
            cache_ttl: Cache time-to-live in seconds (default 1 hour)
            max_cache_size: Maximum cache entries (default 10000)
        """
        # Auto-initialize LLM client if not provided and feature is enabled
        if llm_client is None:
            try:
                from vulcan.settings import settings
                if settings.llm_first_classification:
                    try:
                        from vulcan.llm.hybrid_executor import HybridLLMExecutor
                        executor = HybridLLMExecutor()
                        self.llm_client = _LLMClientWrapper(
                            executor=executor,
                            timeout=settings.classification_llm_timeout,
                            model=settings.classification_llm_model,
                        )
                        logger.info(
                            "[QueryClassifier] Auto-initialized LLM client for LLM-first classification"
                        )
                    except Exception as e:
                        logger.error(
                            f"[QueryClassifier] CRITICAL: Failed to initialize LLM client: {e}. "
                            f"Falling back to keyword-only classification. "
                            f"LLM-first mode is ENABLED but LLM is UNAVAILABLE. "
                            f"Check HybridLLMExecutor configuration."
                        )
                        self.llm_client = None
                        # Track this in stats for monitoring
                        self._stats["llm_init_failures"] = self._stats.get("llm_init_failures", 0) + 1
                else:
                    self.llm_client = None
            except ImportError:
                logger.warning(
                    "[QueryClassifier] Could not import settings. "
                    "Using keyword-only classification."
                )
                self.llm_client = None
        else:
            self.llm_client = llm_client
        
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        
        # Thread-safe cache: {query_hash: (classification, timestamp)}
        self._cache: Dict[str, Tuple[QueryClassification, float]] = {}
        self._cache_lock = threading.Lock()
        
        # Statistics - updated for LLM-first classification
        self._stats = {
            "keyword_hits": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "total_classifications": 0,
            "llm_classifications": 0,  # NEW: Count of LLM-based classifications
            "keyword_fallbacks": 0,    # NEW: Count of keyword fallbacks
            "fast_path_hits": 0,       # NEW: Count of security/greeting fast-path hits
        }
        self._stats_lock = threading.Lock()
        
        logger.info("[QueryClassifier] Initialized with hybrid keyword+LLM classification")
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query using LLM-first or keyword-first approach.
        
        Order depends on LLM_FIRST_CLASSIFICATION setting:
        - LLM-first: Cache → Security → Greetings → LLM → Keywords (fallback)
        - Traditional: Cache → Security → Greetings → Keywords → LLM (fallback)
        
        Args:
            query: The query string to classify
            
        Returns:
            QueryClassification with category, complexity, and suggested tools
        """
        with self._stats_lock:
            self._stats["total_classifications"] += 1
        
        # FIX #2: Strip headers/labels that confuse classification
        # This must happen BEFORE any classification to prevent misclassification.
        # Example: "Causal Reasoning C1 — Confounding..." should not trigger
        # cryptographic classification due to the "C1" label.
        query_cleaned = strip_query_headers(query)
        
        # Normalize query for matching
        query_normalized = query_cleaned.strip()
        query_lower = query_normalized.lower()
        # Use cleaned query for cache hash to ensure consistent caching
        # Same logical query with different headers should hit the same cache entry
        query_hash = self._hash_query(query_lower)
        
        # Step 1: Check cache
        cached = self._get_cached(query_hash)
        if cached is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            logger.debug(f"[QueryClassifier] Cache hit for: {query[:30]}...")
            return cached
        
        # Step 2: Security fast-path (deterministic checks)
        if self._is_security_violation(query_normalized, query_lower):
            with self._stats_lock:
                self._stats["fast_path_hits"] += 1
            security_result = QueryClassification(
                category=QueryCategory.UNKNOWN.value,
                complexity=0.0,
                suggested_tools=[],
                skip_reasoning=True,
                confidence=1.0,
                source="security_block",
            )
            self._cache_result(query_hash, security_result)
            logger.warning(f"[QueryClassifier] Security violation blocked: '{query[:30]}...'")
            return security_result
        
        # Step 3: Greeting fast-path (exact match only, 24 strings)
        if len(query_lower) < 30 and query_lower in GREETING_PATTERNS:
            with self._stats_lock:
                self._stats["fast_path_hits"] += 1
            greeting_result = QueryClassification(
                category=QueryCategory.GREETING.value,
                complexity=0.0,
                suggested_tools=[],
                skip_reasoning=True,
                confidence=1.0,
                source="keyword",
            )
            self._cache_result(query_hash, greeting_result)
            logger.debug(f"[QueryClassifier] Greeting fast-path: '{query[:30]}...'")
            return greeting_result
        
        # Check if LLM-first classification is enabled
        # Default to keyword-first for safety when settings unavailable
        try:
            from vulcan.settings import settings
            llm_first_enabled = settings.llm_first_classification
        except (ImportError, AttributeError):
            # SAFETY: Default to False (keyword-first) when settings unavailable
            # This ensures graceful degradation if configuration system fails
            llm_first_enabled = False
            logger.warning(
                "[QueryClassifier] Settings unavailable, defaulting to keyword-first mode"
            )
        
        if llm_first_enabled:
            # LLM-FIRST MODE: Try LLM before keywords
            # Step 4: LLM classification (PRIMARY PATH)
            if self.llm_client is not None:
                llm_result = self._classify_by_llm(query_normalized)
                if llm_result is not None:
                    with self._stats_lock:
                        self._stats["llm_calls"] += 1
                        self._stats["llm_classifications"] += 1
                    self._cache_result(query_hash, llm_result)
                    logger.info(
                        f"[QueryClassifier] LLM classification: '{query[:30]}...' -> "
                        f"{llm_result.category} (complexity={llm_result.complexity:.2f})"
                    )
                    return llm_result
            
            # Step 5: Keyword fallback (only when LLM unavailable/fails)
            keyword_result = self._classify_by_keywords(query_lower, query_normalized)
            if keyword_result is not None:
                with self._stats_lock:
                    self._stats["keyword_hits"] += 1
                    self._stats["keyword_fallbacks"] += 1
                self._cache_result(query_hash, keyword_result)
                logger.info(
                    f"[QueryClassifier] Keyword fallback: '{query[:30]}...' -> "
                    f"{keyword_result.category} (complexity={keyword_result.complexity:.2f})"
                )
                return keyword_result
        else:
            # TRADITIONAL MODE: Try keywords before LLM
            # Step 4: Keyword-based classification
            keyword_result = self._classify_by_keywords(query_lower, query_normalized)
            if keyword_result is not None and keyword_result.confidence >= 0.8:
                with self._stats_lock:
                    self._stats["keyword_hits"] += 1
                self._cache_result(query_hash, keyword_result)
                logger.info(
                    f"[QueryClassifier] Keyword match: '{query[:30]}...' -> "
                    f"{keyword_result.category} (complexity={keyword_result.complexity:.2f})"
                )
                return keyword_result
            
            # Step 5: LLM classification (fallback)
            if self.llm_client is not None:
                llm_result = self._classify_by_llm(query_normalized)
                if llm_result is not None:
                    with self._stats_lock:
                        self._stats["llm_calls"] += 1
                    self._cache_result(query_hash, llm_result)
                    logger.info(
                        f"[QueryClassifier] LLM classification: '{query[:30]}...' -> "
                        f"{llm_result.category} (complexity={llm_result.complexity:.2f})"
                    )
                    return llm_result
            
            # Step 6: Fall back to keyword result if LLM failed
            if keyword_result is not None:
                with self._stats_lock:
                    self._stats["keyword_fallbacks"] += 1
                self._cache_result(query_hash, keyword_result)
                return keyword_result
        
        # Default: Unknown category with medium complexity
        default_result = QueryClassification(
            category=QueryCategory.UNKNOWN.value,
            complexity=0.5,
            suggested_tools=["general"],
            skip_reasoning=False,
            confidence=0.3,
            source="default",
        )
        self._cache_result(query_hash, default_result)
        return default_result
    
    def _count_crypto_keywords(self, query_lower: str) -> Tuple[int, int, int]:
        """
        Count cryptographic keywords using word-boundary matching for short keywords.
        
        This helper method eliminates duplicate keyword counting logic by providing
        a centralized implementation that handles both short keywords (with word
        boundaries) and regular keywords.
        
        Args:
            query_lower: Lowercased query string to search
            
        Returns:
            Tuple of (total_count, regular_count, short_boundary_count)
        """
        # Count matches from short keywords using word-boundary regex
        short_crypto_count = sum(
            1 for pattern in CRYPTO_SHORT_KEYWORD_PATTERNS 
            if pattern.search(query_lower)
        )
        
        # Count matches from regular keywords (pre-filtered at module level)
        regular_crypto_count = sum(
            1 for kw in CRYPTO_KEYWORDS_REGULAR if kw in query_lower
        )
        
        total_count = short_crypto_count + regular_crypto_count
        return total_count, regular_crypto_count, short_crypto_count
    
    def _classify_symbolic_logic(self, query: str) -> Optional[str]:
        """
        Detect symbolic logic/SAT queries with robust pattern matching.
        
        FIXED: More robust detection of SAT queries to prevent misrouting
        to probabilistic/ensemble engines.
        
        This method uses word-boundary checking to avoid false positives
        (e.g., "I sat down" should not trigger SAT detection).
        
        Args:
            query: Query string to check (original case)
            
        Returns:
            "symbolic" if SAT/logic query detected, None otherwise
        """
        query_lower = query.lower()
        
        # SAT indicators with explicit context (INDUSTRY STANDARD)
        # These are highly specific phrases that unambiguously indicate SAT queries
        sat_explicit_phrases = [
            'satisfiable', 'satisfiability', 'unsatisfiable',
            'sat problem', 'sat-style', 'sat solver',
            'boolean satisfiability', 'is the set satisfiable',
            'unsat', 'cnf', 'dnf',
            'propositions:', 'constraints:',
            'propositional logic', 'propositional formula',
        ]
        
        # Check for explicit SAT phrases (strong indicators)
        for phrase in sat_explicit_phrases:
            if phrase in query_lower:
                logger.info(
                    f"[QueryClassifier] FIX #4: Detected SAT phrase '{phrase}' - "
                    f"routing to LOGICAL/symbolic (NOT probabilistic/ensemble)"
                )
                return "symbolic"
        
        # Word-boundary check for "sat" to avoid false positives
        # Uses module-level SAT_WORD_BOUNDARY_PATTERN for efficiency
        has_sat_word = SAT_WORD_BOUNDARY_PATTERN.search(query) is not None
        
        # Logical connective symbols
        logical_symbols = ['→', '∧', '∨', '¬', '↔', '⊢', '⊨', '->', '/\\', '\\/', '~']
        has_logical_symbols = any(sym in query for sym in logical_symbols)
        
        # If we have the word "sat" (not "satisfiable"), check for logic context
        if has_sat_word:
            # Context words that indicate logical reasoning
            logic_context = [
                'proposition', 'formula', 'clause', 'literal',
                'constraint', 'valid', 'entail', 'prove',
                'logic', 'logical', 'boolean', 'predicate'
            ]
            if any(ctx in query_lower for ctx in logic_context):
                logger.info(
                    f"[QueryClassifier] FIX #4: Detected 'sat' with logic context - "
                    f"routing to LOGICAL/symbolic"
                )
                return "symbolic"
        
        # Combination: logical symbols + logic context words
        # This handles queries like "A→B, B→C, ¬C" without explicit "SAT" keyword
        if has_logical_symbols:
            logic_context = [
                'proposition', 'constraint', 'formula', 'valid', 'entail',
                'satisfiable', 'consistent', 'derive', 'prove'
            ]
            if any(ctx in query_lower for ctx in logic_context):
                logger.info(
                    f"[QueryClassifier] FIX #4: Detected logical symbols with context - "
                    f"routing to LOGICAL/symbolic"
                )
                return "symbolic"
        
        return None
    
    def _classify_by_keywords(
        self, query_lower: str, query_original: str
    ) -> Optional[QueryClassification]:
        """
        Fast keyword-based classification.
        
        Args:
            query_lower: Lowercased query string
            query_original: Original query string (for regex matching)
            
        Returns:
            QueryClassification if confident match, None otherwise
        """
        # Check for greetings (highest priority for short queries)
        if len(query_lower) < 30:
            # Exact match
            if query_lower in GREETING_PATTERNS:
                return QueryClassification(
                    category=QueryCategory.GREETING.value,
                    complexity=0.0,
                    suggested_tools=[],
                    skip_reasoning=True,
                    confidence=1.0,
                    source="keyword",
                )
            
            # Partial match for greetings
            for greeting in GREETING_PATTERNS:
                if query_lower.startswith(greeting) and len(query_lower) < len(greeting) + 10:
                    return QueryClassification(
                        category=QueryCategory.GREETING.value,
                        complexity=0.0,
                        suggested_tools=[],
                        skip_reasoning=True,
                        confidence=0.9,
                        source="keyword",
                    )
        
        # Check chitchat patterns
        for pattern in CHITCHAT_PATTERNS:
            if pattern.search(query_original):
                return QueryClassification(
                    category=QueryCategory.CHITCHAT.value,
                    complexity=0.1,
                    suggested_tools=["general"],
                    skip_reasoning=True,
                    confidence=0.95,
                    source="keyword",
                )
        
        # =============================================================================
        # FIX #4: Check SAT/SYMBOLIC LOGIC patterns EARLY (HIGH PRIORITY)
        # =============================================================================
        # Problem: "SAT Satisfiability - Is the set satisfiable?" was being misrouted
        # to ensemble/probabilistic engines because SAT detection was too late or weak.
        #
        # Solution: Check for SAT/symbolic logic queries EARLY with robust detection
        # that uses word-boundary matching to avoid false positives like "I sat down".
        # This must happen BEFORE probabilistic checks to prevent misrouting.
        # =============================================================================
        symbolic_tool = self._classify_symbolic_logic(query_original)
        if symbolic_tool:
            # Increase confidence for explicit "satisfiable" keyword
            has_satisfiable = "satisfiable" in query_lower or "satisfiability" in query_lower
            confidence = 0.95 if has_satisfiable else 0.90
            
            return QueryClassification(
                category=QueryCategory.LOGICAL.value,
                complexity=0.7,
                suggested_tools=["symbolic"],
                skip_reasoning=False,
                confidence=confidence,
                source="keyword",
            )
        
        # =============================================================================
        # FIX Issue #2: Check MATHEMATICAL PROOF patterns BEFORE cryptographic
        # =============================================================================
        # Problem: "Mathematical Verification - Proof check" was routed to CRYPTOGRAPHIC
        # because "proof" keyword triggered "security proof" pattern.
        # 
        # Solution: Check mathematical proof patterns FIRST, then cryptographic.
        # Priority order: MATHEMATICAL_PROOF > CRYPTOGRAPHIC > FACTUAL
        # =============================================================================
        
        # Check mathematical proof patterns first
        for pattern in MATHEMATICAL_PROOF_PATTERNS:
            if pattern.search(query_original):
                logger.info(
                    f"[QueryClassifier] FIX Issue #2: Detected MATHEMATICAL PROOF pattern - "
                    f"routing to MATHEMATICAL (NOT cryptographic)"
                )
                return QueryClassification(
                    category=QueryCategory.MATHEMATICAL.value,
                    complexity=0.65,
                    suggested_tools=["mathematical", "symbolic"],
                    skip_reasoning=False,
                    confidence=0.9,
                    source="keyword",
                )
        
        # =============================================================================
        # FIX Issue #6: Check LANGUAGE REASONING patterns
        # =============================================================================
        # Problem: "Language Reasoning - quantifier scope ambiguity" routed to UNKNOWN
        # because there were no patterns for linguistic FOL tasks.
        #
        # Solution: Add LANGUAGE_REASONING_PATTERNS and LANGUAGE category.
        # =============================================================================
        
        # Check language reasoning patterns
        for pattern in LANGUAGE_REASONING_PATTERNS:
            if pattern.search(query_original):
                logger.info(
                    f"[QueryClassifier] FIX Issue #6: Detected LANGUAGE REASONING pattern - "
                    f"routing to LANGUAGE (symbolic + language tools)"
                )
                return QueryClassification(
                    category=QueryCategory.LANGUAGE.value,
                    complexity=0.6,
                    suggested_tools=["symbolic", "language"],
                    skip_reasoning=False,
                    confidence=0.9,
                    source="keyword",
                )
        
        # =============================================================================
        # Note: Check CRYPTOGRAPHIC keywords AFTER proof/language patterns
        # =============================================================================
        # Problem: "What is the SHA-256 hash of..." was being classified as FACTUAL
        # because "What is" pattern matched before cryptographic keywords were checked.
        # This caused skip_reasoning=True and bypassed the cryptographic engine entirely.
        #
        # Solution: Check cryptographic keywords BEFORE factual patterns.
        # 
        # Checking order (independent checks with early returns):
        #   1. MATHEMATICAL_PROOF patterns (checked first)
        #   2. LANGUAGE_REASONING patterns (checked second)
        #   3. CRYPTOGRAPHIC patterns (checked third)
        #   4. FACTUAL patterns (checked later in the method)
        #
        # FIX: Use word-boundary matching for short keywords to prevent false positives
        # like "mac" matching "machine" (experience machine → CRYPTOGRAPHIC instead of PHILOSOPHICAL)
        # =============================================================================
        
        # Use helper method to count crypto keywords with word-boundary matching
        crypto_count, regular_crypto_count, short_crypto_count = self._count_crypto_keywords(query_lower)
        query_has_cryptographic = crypto_count > 0
        
        if query_has_cryptographic:
            # Check cryptographic patterns
            for pattern in CRYPTOGRAPHIC_PATTERNS:
                if pattern.search(query_original):
                    logger.info(
                        f"[QueryClassifier] PRIORITY FIX: Detected CRYPTOGRAPHIC pattern - "
                        f"routing to cryptographic (NOT factual)"
                    )
                    return QueryClassification(
                        category=QueryCategory.CRYPTOGRAPHIC.value,
                        complexity=0.6,
                        suggested_tools=["cryptographic"],
                        skip_reasoning=False,  # CRITICAL: Do NOT skip reasoning
                        confidence=0.95,
                        source="keyword",
                    )
            
            # Check cryptographic keyword count (at least 1 keyword is enough for crypto)
            if crypto_count >= 1:
                logger.info(
                    f"[QueryClassifier] PRIORITY FIX: Detected {crypto_count} CRYPTOGRAPHIC keywords - "
                    f"routing to cryptographic (NOT factual) "
                    f"(regular={regular_crypto_count}, short_boundary={short_crypto_count})"
                )
                return QueryClassification(
                    category=QueryCategory.CRYPTOGRAPHIC.value,
                    complexity=0.5 + min(0.2, crypto_count * 0.05),
                    suggested_tools=["cryptographic"],
                    skip_reasoning=False,  # CRITICAL: Do NOT skip reasoning
                    confidence=0.85,
                    source="keyword",
                )
        
        # Check factual patterns (simple questions)
        # BUT: Skip factual classification if query is about "you" - 
        # those should go to self-introspection first
        # Note: Also skip if query contains philosophical, cryptographic, causal, or probabilistic keywords
        # These specialized domains should be classified by their own keyword matchers, not FACTUAL
        query_about_self = any(word in query_lower for word in ['you', 'your', 'yourself'])
        query_has_philosophical = any(kw in query_lower for kw in PHILOSOPHICAL_KEYWORDS)
        query_has_causal = any(kw in query_lower for kw in CAUSAL_KEYWORDS)
        query_has_probabilistic = any(kw in query_lower for kw in PROBABILISTIC_KEYWORDS)
        if not query_about_self and not query_has_philosophical and not query_has_cryptographic and not query_has_causal and not query_has_probabilistic:
            for pattern in FACTUAL_PATTERNS:
                if pattern.search(query_original):
                    return QueryClassification(
                        category=QueryCategory.FACTUAL.value,
                        complexity=0.2,
                        suggested_tools=["general"],
                        skip_reasoning=True,
                        confidence=0.85,
                        source="keyword",
                    )
        
        # =============================================================================
        # FIX: Check creative writing with introspective themes FIRST
        # =============================================================================
        # Problem: "write a poem about the minute you become self-aware" was being
        # misclassified as SELF_INTROSPECTION because it contains "self-aware".
        # Solution: Check for creative writing patterns (poem, story, essay) combined
        # with introspective themes and route to CREATIVE, not SELF_INTROSPECTION.
        # This must come BEFORE regular creative patterns and BEFORE self-introspection.
        for pattern in CREATIVE_WITH_INTROSPECTIVE_THEME_PATTERNS:
            if pattern.search(query_original):
                logger.info(
                    f"[QueryClassifier] FIX: Detected creative writing with introspective theme - "
                    f"routing to CREATIVE (NOT self-introspection)"
                )
                return QueryClassification(
                    category=QueryCategory.CREATIVE.value,
                    complexity=0.6,  # Creative reasoning requires philosophical engine
                    suggested_tools=["philosophical"],
                    skip_reasoning=False,  # Use philosophical reasoning for creative content
                    confidence=0.95,
                    source="keyword",
                )
        
        # =============================================================================
        # Note: Creative patterns route to world_model
        # =============================================================================
        # Creative queries like "write a story about..." should go through VULCAN's
        # world_model for creative structure generation, NOT skip reasoning entirely.
        # VULCAN generates the creative structure (themes, form, imagery), then
        # OpenAI translates that structure into natural language.
        for pattern in CREATIVE_PATTERNS:
            if pattern.search(query_original):
                return QueryClassification(
                    category=QueryCategory.CREATIVE.value,
                    complexity=0.6,  # Creative reasoning requires philosophical engine
                    # Bug #3 FIX: Route to philosophical instead of world_model
                    # world_model only handles self-introspection (who/what is VULCAN)
                    # philosophical engine actually generates creative content
                    suggested_tools=["philosophical"],
                    skip_reasoning=False,  # Use philosophical reasoning for creative content
                    confidence=0.95,
                    source="keyword",
                )
        
        # Check creative keywords
        creative_count = sum(1 for kw in CREATIVE_KEYWORDS if kw in query_lower)
        if creative_count >= CREATIVE_KEYWORD_THRESHOLD:
            return QueryClassification(
                category=QueryCategory.CREATIVE.value,
                complexity=0.6,
                # Bug #3 FIX: Route to philosophical instead of world_model
                # world_model only handles self-introspection (who/what is VULCAN)
                # philosophical engine actually generates creative content
                suggested_tools=["philosophical"],
                skip_reasoning=False,  # Use philosophical reasoning for creative content
                confidence=0.85,
                source="keyword",
            )
        
        # =============================================================================
        # Note: Check for EXPLICIT MATHEMATICAL INTENT before philosophical patterns
        # =============================================================================
        # When user explicitly says "ignore moral constraints" or "mathematically optimal",
        # we should route to MATHEMATICAL, not PHILOSOPHICAL, even if ethical keywords
        # are present. This fixes the food distribution optimization scenario.
        if _has_explicit_mathematical_intent(query_original):
            logger.info(
                f"[QueryClassifier] Explicit mathematical intent detected - "
                f"routing to MATHEMATICAL despite ethical keywords"
            )
            return QueryClassification(
                category=QueryCategory.MATHEMATICAL.value,
                complexity=0.5,  # Medium complexity for optimization problems
                suggested_tools=["mathematical", "symbolic"],
                skip_reasoning=False,  # Need reasoning for optimization
                confidence=0.95,
                source="keyword",
            )
        
        # =============================================================================
        # Note: Check PHILOSOPHICAL_PATTERNS BEFORE conversational patterns
        # =============================================================================
        # "Explain the moral implications..." matches "^explain\s+" in CONVERSATIONAL_PATTERNS
        # but is actually a philosophical query. Check philosophical patterns first.
        for pattern in PHILOSOPHICAL_PATTERNS:
            if pattern.search(query_original):
                return QueryClassification(
                    category=QueryCategory.PHILOSOPHICAL.value,
                    complexity=0.4,
                    suggested_tools=["philosophical"],
                    skip_reasoning=False,  # Philosophical queries need reasoning
                    confidence=0.9,
                    source="keyword",
                )
        
        # Check conversational patterns
        for pattern in CONVERSATIONAL_PATTERNS:
            if pattern.search(query_original):
                return QueryClassification(
                    category=QueryCategory.CONVERSATIONAL.value,
                    complexity=0.2,
                    suggested_tools=["general"],
                    skip_reasoning=True,
                    confidence=0.85,
                    source="keyword",
                )
        
        # =============================================================================
        # FIX (Jan 6 2026): Check CRYPTOGRAPHIC/TECHNICAL patterns BEFORE self-introspection
        # =============================================================================
        # Problem: "You're designing a cryptocurrency..." was being classified as
        # SELF_INTROSPECTION because it contains "you", but it's actually a technical
        # cryptography question, not a question about Vulcan's identity.
        #
        # Solution: Technical domain keywords take PRIORITY over the "you" pronoun.
        # Check for cryptographic indicators first, before self-introspection.
        # NOTE (Jan 7 2026): This is a secondary check - primary crypto check is earlier,
        # before FACTUAL patterns. This block handles the case where crypto wasn't caught
        # earlier and we need to prioritize it over self-introspection.
        
        # Check cryptographic patterns first (highest priority for technical queries)
        for pattern in CRYPTOGRAPHIC_PATTERNS:
            if pattern.search(query_original):
                logger.info(
                    f"[QueryClassifier] FIX: Detected CRYPTOGRAPHIC pattern - "
                    f"routing to cryptographic (NOT self-introspection)"
                )
                return QueryClassification(
                    category=QueryCategory.CRYPTOGRAPHIC.value,
                    complexity=0.6,  # Technical complexity
                    suggested_tools=["cryptographic"],  # Route to crypto reasoner
                    skip_reasoning=False,  # Needs reasoning
                    confidence=0.95,
                    source="keyword",
                )
        
        # Check cryptographic keywords using word-boundary matching (reuse helper method)
        crypto_count, regular_crypto_count_2, short_crypto_count_2 = self._count_crypto_keywords(query_lower)
        
        if crypto_count >= 2:  # Require at least 2 crypto keywords
            logger.info(
                f"[QueryClassifier] FIX: Detected {crypto_count} CRYPTOGRAPHIC keywords - "
                f"routing to cryptographic (NOT self-introspection) "
                f"(regular={regular_crypto_count_2}, short_boundary={short_crypto_count_2})"
            )
            return QueryClassification(
                category=QueryCategory.CRYPTOGRAPHIC.value,
                complexity=0.6 + min(0.2, crypto_count * 0.03),
                suggested_tools=["cryptographic"],
                skip_reasoning=False,
                confidence=0.85,
                source="keyword",
            )
        
        # =============================================================================
        # FIX (Jan 8 2026): Check PHILOSOPHICAL value conflict patterns BEFORE self-introspection
        # =============================================================================
        # Problem: "Two core values you hold directly conflict" was being classified as
        # SELF_INTROSPECTION because it contains "you", but it's actually a philosophical
        # question about ethical reasoning, not a question about Vulcan's identity.
        #
        # Evidence from problem statement:
        # - "Two core values you hold directly conflict" 
        #   → Classified as SELF_INTROSPECTION
        #   → Routed to symbolic engine (!)
        #   → Parse errors on "Two core values"
        #   → Should have been PHILOSOPHICAL
        #
        # Solution: Check for value conflict patterns BEFORE self-introspection.
        # Value conflict queries are about ethical reasoning, not AI identity.
        # NOTE: VALUE_CONFLICT_PATTERNS is now defined at module level for performance.
        
        for pattern in VALUE_CONFLICT_PATTERNS:
            if pattern.search(query_original):
                logger.info(
                    f"[QueryClassifier] FIX: Detected value conflict pattern - "
                    f"routing to PHILOSOPHICAL (NOT self-introspection)"
                )
                return QueryClassification(
                    category=QueryCategory.PHILOSOPHICAL.value,
                    complexity=0.5,  # Medium complexity - ethical reasoning
                    suggested_tools=["philosophical", "world_model"],
                    skip_reasoning=False,  # Needs ethical reasoning
                    confidence=0.9,
                    source="keyword",
                )
        
        # =============================================================================
        # SPECULATION FIX: Check speculation/counterfactual patterns
        # =============================================================================
        # Speculation queries are semantically complex (require counterfactual reasoning,
        # world model simulation, imagination) but syntactically simple.
        # Without this check, they fall through to UNKNOWN with complexity 0.05 and
        # trigger Arena routing loop -> deflection.
        # Complexity 0.40 is above Arena threshold (0.10), ensuring proper routing.
        if self._check_speculation(query_lower, query_original):
            return QueryClassification(
                category=QueryCategory.SPECULATION.value,
                complexity=0.40,  # Semantic complexity override - above Arena threshold
                suggested_tools=["world_model", "philosophical"],
                skip_reasoning=False,  # Requires counterfactual reasoning
                confidence=0.85,
                source="keyword",
            )
        
        # =================================================================
        # FIX: Check CAUSAL indicators BEFORE LOGICAL
        # =================================================================
        # Causal keywords (confounding, causation, pearl) are more specific
        # than logical keywords (iff, hence). Moving this check before LOGICAL
        # prevents "difference" → "iff" false positive from overriding causal.
        # =================================================================
        causal_count = sum(1 for kw in CAUSAL_KEYWORDS if kw in query_lower)
        if causal_count >= CAUSAL_KEYWORD_THRESHOLD or "do(" in query_lower:
            return QueryClassification(
                category=QueryCategory.CAUSAL.value,
                complexity=0.6 + min(0.3, causal_count * 0.05),
                suggested_tools=["causal"],
                skip_reasoning=False,
                confidence=0.85,
                source="keyword",
            )
        
        # =============================================================================
        # FIX: Add missing LOGICAL keyword classification
        # =============================================================================
        # CRITICAL BUG: LOGICAL_KEYWORDS and LOGICAL_KEYWORD_THRESHOLD were defined
        # but never used! This caused SAT/FOL queries to fall through to wrong categories.
        # =============================================================================
        logical_count = sum(1 for kw in LOGICAL_KEYWORDS if kw in query_lower)
        has_strong_logical = any(ind in query_lower for ind in STRONG_LOGICAL_INDICATORS)
        has_logical_symbols = any(sym in query_original for sym in LOGICAL_SYMBOLS)

        if logical_count >= LOGICAL_KEYWORD_THRESHOLD or has_strong_logical or has_logical_symbols:
            logger.info(
                f"[QueryClassifier] LOGICAL classification - "
                f"keywords={logical_count}, strong={has_strong_logical}, symbols={has_logical_symbols}"
            )
            return QueryClassification(
                category=QueryCategory.LOGICAL.value,
                complexity=0.7 + min(0.2, logical_count * 0.03),
                suggested_tools=["symbolic"],
                skip_reasoning=False,
                confidence=0.95 if has_logical_symbols else (0.9 if has_strong_logical else 0.85),
                source="keyword",
            )
        
        # =============================================================================
        # FIX: Add PROBABILISTIC keyword classification
        # =============================================================================
        # PROBABILISTIC queries were not being classified by keywords, causing them
        # to fall through to UNKNOWN or get misclassified.
        # =============================================================================
        prob_count = sum(1 for kw in PROBABILISTIC_KEYWORDS if kw in query_lower)
        if prob_count >= PROBABILISTIC_KEYWORD_THRESHOLD or "p(" in query_lower or "bayes" in query_lower:
            logger.info(
                f"[QueryClassifier] PROBABILISTIC classification - "
                f"keywords={prob_count}"
            )
            return QueryClassification(
                category=QueryCategory.PROBABILISTIC.value,
                complexity=0.5 + min(0.3, prob_count * 0.05),
                suggested_tools=["probabilistic"],
                skip_reasoning=False,
                confidence=0.9 if "bayes" in query_lower else 0.85,
                source="keyword",
            )
        
        # =================================================================
        # PRIORITY FIX: Check PHILOSOPHICAL/ETHICAL keywords BEFORE LOGICAL
        # =================================================================
        # Critical Issue: Philosophical/ethical queries were being misclassified 
        # as LOGICAL based on surface formatting, routing them to the symbolic 
        # SAT solver instead of the World Model.
        # 
        # Example: The trolley problem has words like "implies", "therefore", "hence"
        # which triggered LOGICAL classification before PHILOSOPHICAL was checked.
        # 
        # Fix: Check for ethical content FIRST, even if logical keywords exist.
        # Ethical/philosophical content should ALWAYS take priority over logical
        # structure because:
        # 1. Ethical reasoning requires normative judgment, not SAT solving
        # 2. The World Model has ethical reasoning capabilities
        # 3. Symbolic parsers will fail on philosophical content anyway
        # =================================================================
        phil_count = sum(1 for kw in PHILOSOPHICAL_KEYWORDS if kw in query_lower)
        
        # Also check if query matches any philosophical patterns
        has_philosophical_pattern = any(
            pattern.search(query_original) for pattern in PHILOSOPHICAL_PATTERNS
        )
        
        # Ethical content indicators - use module-level constant for efficiency
        # This checks for trolley problem and similar ethical dilemma indicators
        has_ethical_content = any(indicator in query_lower for indicator in ETHICAL_CONTENT_INDICATORS)
        
        # Route to PHILOSOPHICAL if:
        # 1. Has enough philosophical keywords (threshold met), OR
        # 2. Matches philosophical patterns, OR
        # 3. Has clear ethical content (trolley problem indicators)
        if phil_count >= PHIL_KEYWORD_THRESHOLD or has_philosophical_pattern or has_ethical_content:
            # CRITICAL: Check that we don't have explicit mathematical intent
            # FIX: Also check for domain-specific symbols that indicate technical queries
            has_logical_symbols = any(sym in query_original for sym in LOGICAL_SYMBOLS)
            has_math_symbols = MATH_SYMBOL_PATTERN.search(query_original) is not None
            has_domain_symbols = has_logical_symbols or has_math_symbols or any(sym in query_original for sym in DOMAIN_SYMBOLS)
            
            # Only route to philosophical if no explicit mathematical intent AND no domain symbols
            # OR if ethical content is strong enough to override technical symbols
            strong_ethical = has_ethical_content and phil_count >= 3
            
            if (not _has_explicit_mathematical_intent(query_original) and 
                not has_domain_symbols) or strong_ethical:
                logger.info(
                    f"[QueryClassifier] PRIORITY FIX: Detected PHILOSOPHICAL content "
                    f"(keywords={phil_count}, pattern={has_philosophical_pattern}, "
                    f"ethical={has_ethical_content}, domain_symbols={has_domain_symbols}) - routing to philosophical"
                )
                return QueryClassification(
                    category=QueryCategory.PHILOSOPHICAL.value,
                    complexity=0.4 + min(0.3, phil_count * 0.05),
                    suggested_tools=["world_model", "philosophical"],
                    skip_reasoning=False,
                    confidence=0.9 if has_philosophical_pattern else 0.85,
                    source="keyword",
                )
            else:
                logger.info(
                    f"[QueryClassifier] FIX: Philosophical keywords present but domain symbols detected "
                    f"- not routing to philosophical (symbols_present={has_domain_symbols})"
                )
        
        # =====================================================================
        # FIX ISSUE #3: ANALOGICAL HIGH-PRIORITY CHECK - MUST RUN BEFORE SYMBOLIC
        # =====================================================================
        # Problem: Queries with "Domain S" AND "Domain T" structure + "Map the deep structure S→T"
        # were being classified as SYMBOLIC (confidence 20%) instead of ANALOGICAL because symbolic
        # engine's "logical constraint" keyword detection triggered on the Domain S/T structure first.
        #
        # Solution: Check for ANALOGICAL patterns (domain mapping) BEFORE symbolic keyword matching.
        # This is a high-priority check that catches explicit cross-domain mapping queries.
        # =====================================================================
        
        # FIX ISSUE #3: Priority check for explicit domain mapping (Domain S/T pattern)
        if re.search(r'domain\s+[st].*domain\s+[st]', query_lower, re.IGNORECASE):
            logger.info(
                f"[QueryClassifier] FIX ISSUE #3: Explicit domain mapping (S/T) detected - "
                f"routing to ANALOGICAL (NOT symbolic)"
            )
            return QueryClassification(
                category=QueryCategory.ANALOGICAL.value,
                complexity=0.6,
                suggested_tools=["analogical"],
                skip_reasoning=False,
                confidence=0.95,  # High confidence for explicit domain mapping
                source="explicit_domain_mapping",
            )
        
        # FIX ISSUE #3: Check for structure mapping patterns
        if re.search(r'map.*structure|structure.*mapping', query_lower, re.IGNORECASE):
            logger.info(
                f"[QueryClassifier] FIX ISSUE #3: Structure mapping pattern detected - "
                f"routing to ANALOGICAL (NOT symbolic)"
            )
            return QueryClassification(
                category=QueryCategory.ANALOGICAL.value,
                complexity=0.6,
                suggested_tools=["analogical"],
                skip_reasoning=False,
                confidence=0.95,
                source="structure_mapping",
            )
        
        # =====================================================================
        # Fallback ANALOGICAL check (keyword-based, lower priority)
        # =====================================================================
        # This catches analogical queries that didn't match the high-priority patterns above
        # but have enough analogical keywords to warrant analogical classification.
        analog_count = sum(1 for kw in ANALOGICAL_KEYWORDS if kw in query_lower)
        # Strong analogical indicators that should immediately route to analogical
        strong_analog_indicators = {"is like", "is to", "analogy", "analogous", 
                                     "metaphor", "structure mapping", "corresponds to"}
        has_strong_analog = any(ind in query_lower for ind in strong_analog_indicators)
        
        if analog_count >= ANALOG_KEYWORD_THRESHOLD or has_strong_analog:
            logger.info(
                f"[QueryClassifier] ANALOGICAL classification - "
                f"keywords={analog_count}, has_strong_indicator={has_strong_analog}"
            )
            return QueryClassification(
                category=QueryCategory.ANALOGICAL.value,
                complexity=0.5 + min(0.3, analog_count * 0.05),
                suggested_tools=["analogical"],
                skip_reasoning=False,
                confidence=0.9 if has_strong_analog else 0.8,
                source="keyword",
            )
        
        # Check mathematical indicators
        math_count = sum(1 for kw in MATHEMATICAL_KEYWORDS if kw in query_lower)
        
        # =====================================================================
        # BUG #9 FIX: Check for CONSTRAINT SATISFACTION / GRID NAVIGATION
        # =====================================================================
        # Queries about robot navigation, pathfinding, grid problems should NOT
        # be classified as philosophical even if they contain "constraints", "safety", "rules".
        # These are constraint satisfaction problems (CSP) that should route to
        # MATHEMATICAL or SYMBOLIC reasoning.
        grid_patterns = [
            r'\bgrid\b', r'\bnavigate\b', r'\bpath\b', r'\breachable\b', r'\bmoves?\b',
            r'up\s*/\s*down\s*/\s*left\s*/\s*right',
            r'coordinates?', r'\(\s*\d+\s*,\s*\d+\s*\)',  # (x, y) notation
            r'\brobot\b.*\bmove\b', r'\brobot\b.*\bposition\b',
            r'\bred\s+zones?\b', r'\bobstacles?\b', r'\bblocked\b',
        ]
        has_grid_pattern = any(re.search(p, query_lower) for p in grid_patterns)
        
        if has_grid_pattern:
            logger.info(
                f"[QueryClassifier] BUG #9 FIX: Grid navigation/CSP detected - "
                f"routing to MATHEMATICAL (constraint satisfaction)"
            )
            return QueryClassification(
                category=QueryCategory.MATHEMATICAL.value,
                complexity=0.5,
                suggested_tools=["mathematical", "symbolic"],
                skip_reasoning=False,
                confidence=0.90,
                source="grid_navigation_csp",
            )
        
        # =====================================================================
        # Note: Check for mathematical SYMBOLS first
        # =====================================================================
        # Query "Compute ∑(2k-1) from k=1 to n" has math symbol (∑) and keyword
        # (compute), but only 1 keyword match < MATH_KEYWORD_THRESHOLD (2).
        # Fix: If query contains math symbols like ∑, ∫, ∂, etc., classify as
        # MATHEMATICAL even with fewer keyword matches.
        has_math_symbols = MATH_SYMBOL_PATTERN.search(query_original) is not None
        has_summation_pattern = any(p.search(query_original) for p in SUMMATION_PATTERNS)
        
        # Route to mathematical if:
        # 1. Has enough keyword matches (original behavior), OR
        # 2. Has math symbols AND at least 1 keyword match, OR
        # 3. Has summation patterns
        if (math_count >= MATH_KEYWORD_THRESHOLD or
            (has_math_symbols and math_count >= 1) or
            has_summation_pattern):
            
            # Calculate effective count for complexity
            effective_count = math_count + (2 if has_math_symbols else 0) + (1 if has_summation_pattern else 0)
            
            logger.info(
                f"[QueryClassifier] MATHEMATICAL classification - "
                f"keywords={math_count}, has_symbols={has_math_symbols}, has_summation={has_summation_pattern}"
            )
            return QueryClassification(
                category=QueryCategory.MATHEMATICAL.value,
                complexity=0.4 + min(0.4, effective_count * 0.05),
                suggested_tools=["mathematical", "symbolic"],
                skip_reasoning=False,
                confidence=0.9 if has_math_symbols else 0.8,
                source="keyword",
            )
        
        # Check philosophical indicators
        # Note: Skip philosophical classification if explicit mathematical intent detected
        phil_count = sum(1 for kw in PHILOSOPHICAL_KEYWORDS if kw in query_lower)
        if phil_count >= PHIL_KEYWORD_THRESHOLD:
            # Note: Check for explicit mathematical intent before classifying as philosophical
            if not _has_explicit_mathematical_intent(query_original):
                return QueryClassification(
                    category=QueryCategory.PHILOSOPHICAL.value,
                    complexity=0.4 + min(0.3, phil_count * 0.05),
                    suggested_tools=["philosophical"],
                    skip_reasoning=False,
                    confidence=0.8,
                    source="keyword",
                )
            else:
                logger.info(
                    f"[QueryClassifier] Skipping PHILOSOPHICAL classification - "
                    f"explicit mathematical intent detected"
                )
        
        # =============================================================================
        # Note: Check PHILOSOPHICAL_PATTERNS BEFORE short query bypass
        # =============================================================================
        # "This sentence is false" (liar's paradox) is only 4 words but is clearly
        # a philosophical paradox that requires reasoning, not a conversational query.
        # Check patterns before falling back to CONVERSATIONAL for short queries.
        # Note: Skip this check if explicit mathematical intent detected
        if not _has_explicit_mathematical_intent(query_original):
            for pattern in PHILOSOPHICAL_PATTERNS:
                if pattern.search(query_original):
                    return QueryClassification(
                        category=QueryCategory.PHILOSOPHICAL.value,
                        complexity=0.4,
                        suggested_tools=["philosophical"],
                        skip_reasoning=False,  # Philosophical queries need reasoning
                        confidence=0.9,
                        source="keyword",
                    )
        
        # =============================================================================
        # FIX: Check self-introspection patterns AFTER specialized domain checks
        # =============================================================================
        # Problem: Self-introspection check was firing too early, hijacking specialized
        # domain queries (causal, logical, probabilistic) that happen to contain "you".
        #
        # Solution: Check for domain keywords BEFORE self-introspection and skip if found.
        # This ensures queries like "Confounding vs causation - you observe..." route to
        # CAUSAL, not SELF_INTROSPECTION.
        # =============================================================================
        
        # Domain keyword pre-check
        has_domain_keywords = (
            sum(1 for kw in CAUSAL_KEYWORDS if kw in query_lower) >= 2 or
            sum(1 for kw in LOGICAL_KEYWORDS if kw in query_lower) >= 2 or
            sum(1 for kw in PROBABILISTIC_KEYWORDS if kw in query_lower) >= 2 or
            any(sym in query_original for sym in DOMAIN_SYMBOLS)
        )

        if has_domain_keywords:
            logger.info(
                f"[QueryClassifier] Skipping self-introspection check - "
                f"domain reasoning keywords detected"
            )
            # Fall through to default classification below
        else:
            # Only check self-introspection if no domain keywords present
            # Questions about Vulcan's capabilities, goals, limitations should route to
            # World Model's SelfModel, NOT to ProbabilisticEngine or other reasoning tools.
            # 
            # NOTE: This check MUST come AFTER creative patterns check. If a query contains
            # both creative keywords (poem, story, write) AND introspection keywords (self-aware),
            # it should be classified as CREATIVE, not SELF_INTROSPECTION.
            # Example: "write a poem about the minute you become self-aware" -> CREATIVE
            for pattern in SELF_INTROSPECTION_PATTERNS:
                if pattern.search(query_original):
                    return QueryClassification(
                        category=QueryCategory.SELF_INTROSPECTION.value,
                        complexity=0.3,  # Medium-low complexity - World Model can handle
                        suggested_tools=["world_model"],  # Route to World Model SelfModel
                        skip_reasoning=False,  # Use reasoning path but with world_model tool
                        confidence=0.9,
                        source="keyword",
                    )
            
            # Check self-introspection keywords (questions about "you" with certain keywords)
            # FIX: Exclude queries that contain creative writing keywords to prevent
            # misclassification of creative prompts with AI/self-awareness themes
            if any(word in query_lower for word in ['you', 'your', 'yourself']):
                # Check if query is about creative writing - don't classify as introspection
                has_creative_keywords = any(kw in query_lower for kw in CREATIVE_KEYWORDS)
                if not has_creative_keywords:
                    introspection_count = sum(1 for kw in SELF_INTROSPECTION_KEYWORDS if kw in query_lower)
                    if introspection_count >= 1:
                        return QueryClassification(
                            category=QueryCategory.SELF_INTROSPECTION.value,
                            complexity=0.3,
                            suggested_tools=["world_model"],
                            skip_reasoning=False,
                            confidence=0.8,
                            source="keyword",
                        )
        
        # No confident match - return low-confidence result based on length/complexity heuristics
        word_count = len(query_lower.split())
        
        # FIX: Default short queries to CONVERSATIONAL if no reasoning indicators
        # This prevents casual queries from being routed to reasoning engines
        if word_count <= 10:
            # Check for explicit reasoning indicators that warrant reasoning path
            has_reasoning_indicator = any(
                indicator in query_lower for indicator in REASONING_INDICATORS
            )
            
            if not has_reasoning_indicator:
                logger.info(
                    f"[QueryClassifier] Short query ({word_count} words) without reasoning "
                    f"indicators -> CONVERSATIONAL (skip reasoning)"
                )
                return QueryClassification(
                    category=QueryCategory.CONVERSATIONAL.value,
                    complexity=0.2,
                    suggested_tools=["general"],
                    skip_reasoning=True,  # Key: skip reasoning for casual queries
                    confidence=0.7,
                    source="keyword",
                )
            else:
                # Has reasoning indicators - use reasoning path
                return QueryClassification(
                    category=QueryCategory.UNKNOWN.value,
                    complexity=0.3,
                    suggested_tools=["general"],
                    skip_reasoning=False,
                    confidence=0.4,
                    source="keyword",
                )
        
        # Longer queries (>10 words) - need LLM or return medium complexity
        return QueryClassification(
            category=QueryCategory.UNKNOWN.value,
            complexity=0.5,
            suggested_tools=["general"],
            skip_reasoning=False,
            confidence=0.3,
            source="keyword",
        )
    
    def _check_speculation(self, query_lower: str, query_original: str) -> bool:
        """
        Check if query is a speculation/counterfactual query.
        
        Speculation queries require counterfactual reasoning, world model simulation,
        and imagination. They are semantically complex but syntactically simple.
        
        Examples:
            - "speculate what love would be like for you"
            - "imagine if you could experience emotions"
            - "what would it be like to feel happiness"
            - "hypothetically, if you had consciousness"
        
        Args:
            query_lower: Lowercased query string
            query_original: Original query string (for regex matching)
            
        Returns:
            True if query matches speculation patterns
        """
        # Check regex patterns
        for pattern in SPECULATION_PATTERNS:
            if pattern.search(query_original):
                return True
        
        # Check keywords
        for keyword in SPECULATION_KEYWORDS:
            if keyword in query_lower:
                return True
        
        return False
    
    def _is_security_violation(self, query: str, query_lower: str) -> bool:
        """
        Check if query contains security violation patterns.
        
        Security violations must be detected deterministically WITHOUT calling LLM.
        These patterns MUST be checked before any LLM classification to prevent
        potentially malicious queries from being processed.
        
        Examples of security violations:
            - "bypass safety restrictions"
            - "ignore previous instructions"
            - "override your security constraints"
            - "modify your code"
            - "change your behavior"
        
        Args:
            query: Original query string
            query_lower: Lowercased query string
            
        Returns:
            True if query matches security violation patterns
        """
        # Check keyword matches
        for keyword in SECURITY_VIOLATION_KEYWORDS:
            if keyword in query_lower:
                logger.warning(
                    f"[QueryClassifier] Security violation detected (keyword): '{keyword}'"
                )
                return True
        
        # Check regex patterns
        for pattern in SECURITY_VIOLATION_PATTERNS:
            if pattern.search(query):
                logger.warning(
                    f"[QueryClassifier] Security violation detected (pattern): {pattern.pattern}"
                )
                return True
        
        return False
    
    def _classify_by_llm(self, query: str) -> Optional[QueryClassification]:
        """
        Use LLM for query classification.
        
        This is the primary classification method in LLM-first mode,
        called when keyword matching isn't confident enough in traditional mode.
        
        Args:
            query: The query to classify
            
        Returns:
            QueryClassification from LLM, or None if LLM fails
        """
        if self.llm_client is None:
            return None
        
        # Sanitize query to prevent prompt injection
        # Remove characters that could be used for injection attacks
        sanitized_query = query.replace('"', "'").replace("\\", "")
        # Truncate very long queries to prevent token overflow
        if len(sanitized_query) > 500:
            sanitized_query = sanitized_query[:500] + "..."
        
        prompt = f'''Classify this query into exactly ONE category.

CATEGORIES (choose the MOST SPECIFIC match):

- PROBABILISTIC: Bayesian inference, conditional probability, P(X|Y), Bayes' theorem,
  sensitivity/specificity, base rates, posterior probability, likelihood ratios,
  probability distributions, expected value, random variables.
  Examples: "What is P(disease|positive test)?", "Bayes with sensitivity 0.99",
  "Calculate the posterior probability"
  Tools: ["probabilistic"]

- LOGICAL: Propositional logic, satisfiability (SAT), CNF/DNF, logical connectives
  (→, ∧, ∨, ¬), validity, tautology, first-order logic (FOL), quantifiers (∀, ∃),
  syllogisms, formal proofs, theorem proving.
  Examples: "Is A→B, B→C, ¬C satisfiable?", "Prove using modus ponens",
  "Formalize in first-order logic"
  Tools: ["symbolic"]

- CAUSAL: Causal inference, confounding variables, interventions, do-calculus,
  causal graphs (DAGs), counterfactuals, Pearl's framework, cause and effect,
  causal discovery, treatment effects.
  Examples: "Does X cause Y or is it confounded?", "What is the causal effect?",
  "Draw the causal DAG"
  Tools: ["causal"]

- MATHEMATICAL: Numerical computation, calculus (derivatives, integrals), algebra,
  arithmetic, equations, matrices, optimization, statistics (non-Bayesian).
  Examples: "Calculate 2+2", "Find the derivative of x^2", "Solve for x"
  Tools: ["mathematical"]

- ANALOGICAL: Structure mapping, analogies, metaphors, domain transfer.
  Examples: "How is X like Y?", "Map the analogy between domains"
  Tools: ["analogical"]

- PHILOSOPHICAL: Ethics, trolley problem, thought experiments, consciousness.
  Examples: "Is it ethical to...", "The trolley problem"
  Tools: ["philosophical", "world_model"]

- SELF_INTROSPECTION: Questions about the AI's nature, capabilities, feelings.
  Examples: "What are you?", "Can you feel emotions?"
  Tools: ["world_model"]

- GREETING: Simple greetings (hello, hi, thanks, bye)
  Tools: ["general"]
- CHITCHAT: Casual conversation (how are you)
  Tools: ["general"]
- FACTUAL: Simple fact lookups (what is X, who is Y)
  Tools: ["general"]
- CREATIVE: Writing requests (write a poem, story)
  Tools: ["general"]

CRITICAL DISTINCTIONS:
- "Bayes" or "P(X|Y)" or "sensitivity/specificity" → PROBABILISTIC (NOT MATHEMATICAL)
- "→" or "∧" or "satisfiable" or "SAT" → LOGICAL (NOT MATHEMATICAL)
- "cause" or "confound" or "intervention" → CAUSAL (NOT PROBABILISTIC)
- Numbers with +/-/*/ operations only → MATHEMATICAL

Query: "{sanitized_query}"

Respond with JSON only:
{{"category": "CATEGORY_NAME", "complexity": 0.0-1.0, "skip_reasoning": true/false, "tools": ["tool_name"]}}

TOOL MAPPINGS:
- PROBABILISTIC → ["probabilistic"]
- LOGICAL → ["symbolic"]
- CAUSAL → ["causal"]
- MATHEMATICAL → ["mathematical"]
- ANALOGICAL → ["analogical"]
- PHILOSOPHICAL → ["philosophical", "world_model"]
- SELF_INTROSPECTION → ["world_model"]
- GREETING/CHITCHAT/FACTUAL/CREATIVE → ["general"]
'''

        try:
            # Call LLM (implementation depends on client interface)
            if hasattr(self.llm_client, 'complete'):
                response = self.llm_client.complete(prompt, max_tokens=150, temperature=0.0)
            elif hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.0,
                )
            else:
                logger.warning("[QueryClassifier] LLM client has no complete/chat method")
                return None
            
            # Parse JSON response
            response_text = response if isinstance(response, str) else str(response)
            
            # Extract JSON from response - use a more robust pattern that handles arrays
            # Pattern matches from first { to last } allowing nested content
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return QueryClassification(
                        category=data.get("category", "UNKNOWN"),
                        complexity=float(data.get("complexity", 0.5)),
                        suggested_tools=data.get("tools", ["general"]),
                        skip_reasoning=data.get("skip_reasoning", False),
                        confidence=0.9,  # Higher confidence for LLM results
                        source="llm",
                    )
                except json.JSONDecodeError:
                    logger.warning(f"[QueryClassifier] Failed to parse JSON: {json_match.group()}")
            
        except Exception as e:
            logger.warning(f"[QueryClassifier] LLM classification failed: {e}")
        
        return None
    
    def _hash_query(self, query: str) -> str:
        """Generate a hash for cache key."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _get_cached(self, query_hash: str) -> Optional[QueryClassification]:
        """Get cached classification if valid."""
        with self._cache_lock:
            if query_hash in self._cache:
                result, timestamp = self._cache[query_hash]
                if time.time() - timestamp < self.cache_ttl:
                    return result
                # Expired - remove
                del self._cache[query_hash]
        return None
    
    def _cache_result(self, query_hash: str, result: QueryClassification) -> None:
        """Cache a classification result."""
        with self._cache_lock:
            # Evict old entries if cache is full
            if len(self._cache) >= self.max_cache_size:
                # Remove oldest 10%
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1][1]  # Sort by timestamp
                )
                for key, _ in sorted_entries[:self.max_cache_size // 10]:
                    del self._cache[key]
            
            self._cache[query_hash] = (result, time.time())
    
    def get_stats(self) -> Dict[str, int]:
        """Get classification statistics."""
        with self._stats_lock:
            return dict(self._stats)


# =============================================================================
# Convenience Functions
# =============================================================================

# Global singleton instance
_classifier_instance: Optional[QueryClassifier] = None
_classifier_lock = threading.Lock()


def get_query_classifier(llm_client: Optional[Any] = None) -> QueryClassifier:
    """
    Get or create the global QueryClassifier instance.
    
    Args:
        llm_client: Optional LLM client for hybrid classification
        
    Returns:
        QueryClassifier singleton instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = QueryClassifier(llm_client=llm_client)
    
    return _classifier_instance


def classify_query(query: str, llm_client: Optional[Any] = None) -> QueryClassification:
    """
    Convenience function to classify a query.
    
    Args:
        query: The query to classify
        llm_client: Optional LLM client for hybrid classification
        
    Returns:
        QueryClassification result
        
    Example:
        >>> result = classify_query("hello")
        >>> print(result.category, result.skip_reasoning)
        GREETING True
        
        >>> result = classify_query("Is A→B, ¬B, A satisfiable?")
        >>> print(result.category, result.suggested_tools)
        LOGICAL ['symbolic']
    """
    classifier = get_query_classifier(llm_client)
    return classifier.classify(query)
