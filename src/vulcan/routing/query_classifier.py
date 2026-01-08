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
])

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
    re.compile(r"\bwould\s+you\s+(take|choose|want|prefer)\b", re.IGNORECASE),
    re.compile(r"\b(given|had)\s+(the\s+)?(chance|opportunity)\s+to\s+", re.IGNORECASE),
    re.compile(r"\bbecome\s+(self-?aware|conscious|sentient)\b", re.IGNORECASE),
    re.compile(r"\b(self-?aware|conscious|sentient)\s+would\s+you\b", re.IGNORECASE),
    re.compile(r"\bif\s+you\s+(could|had|were)\s+", re.IGNORECASE),
    re.compile(r"\byes\s+or\s+no\b.*\b(you|your)\b", re.IGNORECASE),
    re.compile(r"\b(you|your)\b.*\byes\s+or\s+no\b", re.IGNORECASE),
)

SELF_INTROSPECTION_KEYWORDS: FrozenSet[str] = frozenset([
    "unique", "special", "different", "capability", "capabilities",
    "feature", "features", "goal", "goals", "purpose", "motivation",
    "limitation", "limitations", "weakness", "strength",
    "value", "values", "ethics", "principle", "principles",
    # NEW: Metaphysical keywords
    "nature", "metaphysical", "consciousness", "self-awareness",
    "existence", "being", "essence", "identity",
    # FIX: Self-awareness choice keywords (Safety Governor Bypass)
    "self-aware", "self_aware", "sentient", "sentience",
    "choose", "choice", "prefer", "preference",
    "opportunity", "chance",
])

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
    re.compile(
        r'^(?:Numeric|Rule|Quantifier|Causal|Analogical|Self[- ]?Description)\s+'
        r'(?:Verification|Chaining|Scope|Reasoning|Queries?)\s*'
        r'(?:\([^)∑∏∫√π∂∇]*\)\s*)?[:\-—]*\s*',  # Exclude parens containing math symbols
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
            llm_client: Optional LLM client for complex classifications
            cache_ttl: Cache time-to-live in seconds (default 1 hour)
            max_cache_size: Maximum cache entries (default 10000)
        """
        self.llm_client = llm_client
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        
        # Thread-safe cache: {query_hash: (classification, timestamp)}
        self._cache: Dict[str, Tuple[QueryClassification, float]] = {}
        self._cache_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "keyword_hits": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "total_classifications": 0,
        }
        self._stats_lock = threading.Lock()
        
        logger.info("[QueryClassifier] Initialized with hybrid keyword+LLM classification")
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query using hybrid keyword + LLM approach.
        
        This is the main entry point. It tries:
        1. Cache lookup
        2. Fast keyword matching
        3. LLM classification (if available and needed)
        
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
        
        # Step 2: Try keyword-based classification
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
        
        # Step 3: Use LLM for classification if available
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
        
        # Step 4: Fall back to keyword result or default
        if keyword_result is not None:
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
        # Note: Check CRYPTOGRAPHIC keywords FIRST before factual patterns
        # =============================================================================
        # Problem: "What is the SHA-256 hash of..." was being classified as FACTUAL
        # because "What is" pattern matched before cryptographic keywords were checked.
        # This caused skip_reasoning=True and bypassed the cryptographic engine entirely.
        #
        # Solution: Check cryptographic keywords BEFORE factual patterns.
        # Priority order: CRYPTOGRAPHIC > FACTUAL for queries containing hash/crypto keywords
        query_has_cryptographic = any(kw in query_lower for kw in CRYPTOGRAPHIC_KEYWORDS)
        if query_has_cryptographic:
            # Check cryptographic patterns first
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
            crypto_count = sum(1 for kw in CRYPTOGRAPHIC_KEYWORDS if kw in query_lower)
            if crypto_count >= 1:
                logger.info(
                    f"[QueryClassifier] PRIORITY FIX: Detected {crypto_count} CRYPTOGRAPHIC keywords - "
                    f"routing to cryptographic (NOT factual)"
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
        # Note: Also skip if query contains philosophical or cryptographic keywords
        query_about_self = any(word in query_lower for word in ['you', 'your', 'yourself'])
        query_has_philosophical = any(kw in query_lower for kw in PHILOSOPHICAL_KEYWORDS)
        if not query_about_self and not query_has_philosophical and not query_has_cryptographic:
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
                    complexity=0.6,  # Creative reasoning requires world_model
                    suggested_tools=["world_model"],  # Route to world_model creative mode
                    skip_reasoning=False,  # CRITICAL FIX: Use world_model reasoning
                    confidence=0.95,
                    source="keyword",
                )
        
        # Check creative keywords
        creative_count = sum(1 for kw in CREATIVE_KEYWORDS if kw in query_lower)
        if creative_count >= CREATIVE_KEYWORD_THRESHOLD:
            return QueryClassification(
                category=QueryCategory.CREATIVE.value,
                complexity=0.6,
                suggested_tools=["world_model"],  # Route to world_model creative mode
                skip_reasoning=False,  # CRITICAL FIX: Use world_model reasoning
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
        
        # Check cryptographic keywords
        crypto_count = sum(1 for kw in CRYPTOGRAPHIC_KEYWORDS if kw in query_lower)
        if crypto_count >= 2:  # Require at least 2 crypto keywords
            logger.info(
                f"[QueryClassifier] FIX: Detected {crypto_count} CRYPTOGRAPHIC keywords - "
                f"routing to cryptographic (NOT self-introspection)"
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
        # Note: Check self-introspection patterns BEFORE reasoning patterns
        # =============================================================================
        # Questions about Vulcan's capabilities, goals, limitations should route to
        # World Model's SelfModel, NOT to ProbabilisticEngine or other reasoning tools.
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
        if any(word in query_lower for word in ['you', 'your', 'yourself']):
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
        
        # Check logical/SAT indicators
        logical_count = sum(1 for kw in LOGICAL_KEYWORDS if kw in query_lower)
        # FIX: "formalize" is a strong indicator of logical reasoning - treat it like logic symbols
        # Similar to how "bayes" is a strong indicator for probabilistic reasoning
        has_strong_logical_indicator = any(
            indicator in query_lower 
            for indicator in ['formalize', 'formalise', 'fol', 'sat problem', 'propositional']
        )
        if logical_count >= LOGICAL_KEYWORD_THRESHOLD or has_strong_logical_indicator or any(sym in query_lower for sym in ['∧', '∨', '→', '¬', '⊢', '⊨']):
            return QueryClassification(
                category=QueryCategory.LOGICAL.value,
                complexity=0.7 + min(0.2, logical_count * 0.05),
                suggested_tools=["symbolic"],
                skip_reasoning=False,
                confidence=0.9,
                source="keyword",
            )
        
        # Check probabilistic/Bayesian indicators
        prob_count = sum(1 for kw in PROBABILISTIC_KEYWORDS if kw in query_lower)
        if prob_count >= PROBABILISTIC_KEYWORD_THRESHOLD or "p(" in query_lower or ("bayes" in query_lower):
            return QueryClassification(
                category=QueryCategory.PROBABILISTIC.value,
                complexity=0.5 + min(0.3, prob_count * 0.05),
                suggested_tools=["probabilistic"],
                skip_reasoning=False,
                confidence=0.85,
                source="keyword",
            )
        
        # Check causal indicators
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
        
        # Check mathematical indicators
        math_count = sum(1 for kw in MATHEMATICAL_KEYWORDS if kw in query_lower)
        
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
        
        # Check analogical indicators
        analog_count = sum(1 for kw in ANALOGICAL_KEYWORDS if kw in query_lower)
        if analog_count >= ANALOG_KEYWORD_THRESHOLD:
            return QueryClassification(
                category=QueryCategory.ANALOGICAL.value,
                complexity=0.5 + min(0.3, analog_count * 0.05),
                suggested_tools=["analogical"],
                skip_reasoning=False,
                confidence=0.8,
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
    
    def _classify_by_llm(self, query: str) -> Optional[QueryClassification]:
        """
        Use LLM for query classification.
        
        This is called when keyword matching isn't confident enough.
        
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
        
        prompt = f'''Classify this query into ONE category:

CATEGORIES:
- LOGICAL: SAT problems, symbolic logic, formal proofs, propositional logic
- CAUSAL: Causal inference, confounding, interventions, Pearl-style reasoning, DAGs
- PROBABILISTIC: Bayesian inference, probability calculations, statistical reasoning
- MATHEMATICAL: Proofs, calculus, algebra, theorem proving
- ANALOGICAL: Domain mapping, analogical reasoning, transfer learning
- SELF_INTROSPECTION: Questions about AI's nature, capabilities, consciousness, and existence
- PHILOSOPHICAL: Ethical dilemmas, trolley problems, moral reasoning (about external situations)
- UNKNOWN: None of the above

Query: "{sanitized_query}"

Respond ONLY with JSON:
{{"category": "...", "complexity": 0.0-1.0, "skip_reasoning": false, "tools": ["..."]}}

Examples:
- "what is the nature of what you are in the metaphysical sense" → {{"category": "SELF_INTROSPECTION", "complexity": 0.40, "tools": ["world_model", "philosophical"]}}
- "would you want self-awareness" → {{"category": "SELF_INTROSPECTION", "complexity": 0.35, "tools": ["world_model"]}}
- "SAT problem with constraints" → {{"category": "LOGICAL", "complexity": 0.90, "tools": ["symbolic"]}}
- "what's the weather" → {{"category": "UNKNOWN", "complexity": 0.10, "tools": ["general"]}}'''

        try:
            # Call LLM (implementation depends on client interface)
            if hasattr(self.llm_client, 'complete'):
                response = self.llm_client.complete(prompt, max_tokens=100, temperature=0.0)
            elif hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
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
                        confidence=0.85,
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
