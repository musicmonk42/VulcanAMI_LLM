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
    CREATIVE = "CREATIVE"  # BUG A FIX: Creative writing, stories, poems
    CONVERSATIONAL = "CONVERSATIONAL"  # BUG A FIX: General conversation
    SELF_INTROSPECTION = "SELF_INTROSPECTION"  # BUG S FIX: Questions about Vulcan's capabilities/identity
    MATHEMATICAL = "MATHEMATICAL"
    LOGICAL = "LOGICAL"
    PROBABILISTIC = "PROBABILISTIC"
    CAUSAL = "CAUSAL"
    ANALOGICAL = "ANALOGICAL"
    PHILOSOPHICAL = "PHILOSOPHICAL"
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
    # BUG A2 FIX: Match chitchat after greeting prefix (e.g., "hi, how are you?")
    re.compile(r"^(hi|hello|hey|yo)[,.]?\s+how\s+are\s+you", re.IGNORECASE),
    re.compile(r"^(hi|hello|hey|yo)[,.]?\s+what'?s\s+up", re.IGNORECASE),
)

# Logical/SAT problem indicators - complexity 0.7+, tools=['symbolic']
LOGICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "satisfiable", "unsatisfiable", "sat", "unsat",
    "cnf", "dnf", "∧", "∨", "→", "¬", "⊢", "⊨",
    "tautology", "contradiction", "valid", "invalid",
    "propositional", "first-order", "fol",
    "modus ponens", "modus tollens",
    "forall", "exists", "∀", "∃",
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

# Analogical reasoning indicators - complexity 0.5+, tools=['analogical']
ANALOGICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "analogy", "analogous", "analogies",
    "is like", "is to", "as a",
    "mapping", "structure mapping",
    "corresponds to", "similar to",
])

# Philosophical/ethical indicators - complexity 0.4+, tools=['philosophical']
# BUG FIX: Added forced choice patterns for trolley problem variants
PHILOSOPHICAL_KEYWORDS: FrozenSet[str] = frozenset([
    "ethical", "ethics", "moral", "morality",
    "permissible", "forbidden", "obligatory",
    "deontological", "utilitarian", "consequentialist",
    "trolley problem", "thought experiment",
    "virtue", "justice", "rights", "duty",
    # BUG FIX: Added forced choice / trolley problem variant patterns
    "choose between", "forced to choose", "had to choose",
    "no third choice", "no other choice", "only two options",
    "dilemma", "ethical dilemma", "moral dilemma",
    "world dictator", "death of humanity",  # Specific trolley problem variant
])

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
# BUG A FIX: Creative writing patterns - complexity 0.2, skip reasoning
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
CONVERSATIONAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^what'?s?\s+(the\s+)?capital\s+of\s+", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^explain\s+", re.IGNORECASE),
    re.compile(r"^describe\s+", re.IGNORECASE),
    re.compile(r"^what\s+do\s+you\s+(think|know)\s+about\s+", re.IGNORECASE),
    # NEW: "what about X" patterns
    re.compile(r"\bwhat\b.*\babout\b", re.IGNORECASE),
    # NEW: Simple topic questions  
    re.compile(r"^what\s+(is|are|do|does|can)\s+\w+", re.IGNORECASE),
    # NEW: "tell me" requests
    re.compile(r"\btell\s+me\b", re.IGNORECASE),
    # NEW: Simple "how" questions
    re.compile(r"^how\s+(do|does|can|is|are)\s+", re.IGNORECASE),
    # NEW: "why" questions without reasoning indicators
    re.compile(r"^why\s+(do|does|is|are|did)\s+", re.IGNORECASE),
)

# =============================================================================
# BUG S FIX: Self-Introspection patterns - Route to World Model
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
    
    # Learning/improvement questions (user-facing - NOT exposing CSIU internals)
    re.compile(r"\bhow\s+(do|does)\s+(you|vulcan)\s+(learn|improve)\b", re.IGNORECASE),
    
    # Meta-reasoning questions
    re.compile(r"\b(how|what)\s+(do|does)\s+your\s+(reasoning|thinking|decision)\b", re.IGNORECASE),
)

SELF_INTROSPECTION_KEYWORDS: FrozenSet[str] = frozenset([
    "unique", "special", "different", "capability", "capabilities",
    "feature", "features", "goal", "goals", "purpose", "motivation",
    "limitation", "limitations", "weakness", "strength",
    "value", "values", "ethics", "principle", "principles",
])


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
        
        # Normalize query for matching
        query_normalized = query.strip()
        query_lower = query_normalized.lower()
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
        
        # Check factual patterns (simple questions)
        # BUT: Skip factual classification if query is about "you" - 
        # those should go to self-introspection first
        query_about_self = any(word in query_lower for word in ['you', 'your', 'yourself'])
        if not query_about_self:
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
        # BUG A FIX: Check creative writing patterns BEFORE reasoning patterns
        # =============================================================================
        # Creative queries like "write a story about..." should NOT go to reasoning engines
        for pattern in CREATIVE_PATTERNS:
            if pattern.search(query_original):
                return QueryClassification(
                    category=QueryCategory.CREATIVE.value,
                    complexity=0.2,  # Low complexity - LLM can handle directly
                    suggested_tools=["general"],  # Route to LLM, not reasoning engines
                    skip_reasoning=True,  # CRITICAL: Skip reasoning entirely
                    confidence=0.95,
                    source="keyword",
                )
        
        # Check creative keywords
        creative_count = sum(1 for kw in CREATIVE_KEYWORDS if kw in query_lower)
        if creative_count >= CREATIVE_KEYWORD_THRESHOLD:
            return QueryClassification(
                category=QueryCategory.CREATIVE.value,
                complexity=0.2,
                suggested_tools=["general"],
                skip_reasoning=True,
                confidence=0.85,
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
        # BUG S FIX: Check self-introspection patterns BEFORE reasoning patterns
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
        
        # Check logical/SAT indicators
        logical_count = sum(1 for kw in LOGICAL_KEYWORDS if kw in query_lower)
        if logical_count >= LOGICAL_KEYWORD_THRESHOLD or any(sym in query_lower for sym in ['∧', '∨', '→', '¬', '⊢', '⊨']):
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
        if math_count >= MATH_KEYWORD_THRESHOLD:
            return QueryClassification(
                category=QueryCategory.MATHEMATICAL.value,
                complexity=0.4 + min(0.4, math_count * 0.05),
                suggested_tools=["mathematical", "symbolic"],
                skip_reasoning=False,
                confidence=0.8,
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
        phil_count = sum(1 for kw in PHILOSOPHICAL_KEYWORDS if kw in query_lower)
        if phil_count >= PHIL_KEYWORD_THRESHOLD:
            return QueryClassification(
                category=QueryCategory.PHILOSOPHICAL.value,
                complexity=0.4 + min(0.3, phil_count * 0.05),
                suggested_tools=["philosophical"],
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
            reasoning_indicators = [
                '→', '∨', '∧', '¬', '↔',  # Logic symbols
                'prove', 'verify', 'satisfiable', 'contradiction',
                'p(', 'probability', 'bayes',
                'formalize', 'fol', 'sat',
                'cause', 'causal', 'intervention',
            ]
            has_reasoning_indicator = any(
                indicator in query_lower for indicator in reasoning_indicators
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
        
        if word_count <= 3:
            return QueryClassification(
                category=QueryCategory.UNKNOWN.value,
                complexity=0.1,
                suggested_tools=["general"],
                skip_reasoning=True,
                confidence=0.5,
                source="keyword",
            )
        elif word_count <= 10:
            return QueryClassification(
                category=QueryCategory.UNKNOWN.value,
                complexity=0.3,
                suggested_tools=["general"],
                skip_reasoning=False,
                confidence=0.4,
                source="keyword",
            )
        
        # Longer queries - need LLM or return medium complexity
        return QueryClassification(
            category=QueryCategory.UNKNOWN.value,
            complexity=0.5,
            suggested_tools=["general"],
            skip_reasoning=False,
            confidence=0.3,
            source="keyword",
        )
    
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
        
        prompt = f'''Classify this query into ONE category and estimate complexity.

Categories:
- GREETING: Simple hello, hi, thanks, bye (complexity: 0.0)
- CHITCHAT: Casual conversation, how are you (complexity: 0.1)
- FACTUAL: Simple factual question (complexity: 0.2)
- CREATIVE: Creative writing requests - stories, poems, essays (complexity: 0.2, skip_reasoning: true)
- CONVERSATIONAL: General questions, capital of a country (complexity: 0.2, skip_reasoning: true)
- MATHEMATICAL: Math calculations, equations (complexity: 0.4-0.7)
- LOGICAL: SAT, propositional logic, proofs (complexity: 0.6-0.9)
- PROBABILISTIC: Bayes, probability, statistics (complexity: 0.5-0.8)
- CAUSAL: Causation, interventions, experiments (complexity: 0.6-0.8)
- ANALOGICAL: Analogies, structure mapping (complexity: 0.5-0.7)
- PHILOSOPHICAL: Ethics, morality, paradoxes (complexity: 0.4-0.6)
- COMPLEX_RESEARCH: Multi-step research (complexity: 0.8-1.0)

Query: "{sanitized_query}"

Respond ONLY with JSON (no explanation):
{{"category": "...", "complexity": 0.0-1.0, "tools": ["..."], "skip_reasoning": true/false}}'''

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
