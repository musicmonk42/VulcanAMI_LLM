"""
Configuration constants for unified reasoning module.

This module contains all configuration constants, thresholds, and parameters
used throughout the unified reasoning system.

Following highest industry standards:
- Named constants for all magic numbers
- Clear documentation for each constant
- Grouped by functionality
- Type annotations where applicable

Author: VulcanAMI Team
License: Proprietary
"""

import re
from typing import FrozenSet, List, Pattern

# ==============================================================================
# CACHE CONFIGURATION
# ==============================================================================
# These constants control the unified reasoning cache behavior.
# Note: Cache key was using only 8 chars, causing collisions.

# Number of hex characters to use from SHA-256 hash for cache keys
# 32 hex chars = 128 bits = extremely low collision probability
# Birthday paradox: 50% collision after ~2^64 operations (sqrt of 2^128 hash space)
CACHE_HASH_LENGTH: int = 32

# Maximum cache entry age in seconds (5 minutes)
CACHE_MAX_AGE_SECONDS: float = 300.0

# ==============================================================================
# MATHEMATICAL VERIFICATION CONSTANTS
# ==============================================================================
# These constants control the mathematical verification and learning integration.
# They are extracted as named constants per code review feedback.

# Confidence adjustment factors
MATH_VERIFICATION_CONFIDENCE_BOOST: float = 1.1  # Boost confidence when verified correct
MATH_ERROR_CONFIDENCE_PENALTY: float = 0.5  # Reduce confidence when error detected

# Learning reward/penalty values
MATH_ACCURACY_REWARD: float = 0.015  # Bonus for mathematically correct results
MATH_ACCURACY_PENALTY: float = -0.01  # Penalty for mathematical errors
MATH_WEIGHT_ADJUSTMENT_PENALTY: float = -0.01  # Adjustment to tool weights on error

# Keys to search for numerical results in conclusions
NUMERICAL_RESULT_KEYS: tuple = ('probability', 'result', 'value', 'posterior', 'answer')

# ==============================================================================
# CONFIDENCE FLOOR CONSTANTS
# ==============================================================================
# These ensure reasoning results aren't filtered out even when models are untrained.
# Each reasoning engine type has a minimum confidence threshold.

# Symbolic reasoning confidence floors
CONFIDENCE_FLOOR_SYMBOLIC_PROVEN: float = 0.6  # High confidence when proof is found
CONFIDENCE_FLOOR_SYMBOLIC_HAS_PROOF: float = 0.4  # Medium confidence when proof object exists
CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT: float = 0.2  # Minimum floor for symbolic reasoning

# Causal reasoning confidence floors
CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT: float = 0.3  # Confidence for successful causal analysis
CONFIDENCE_FLOOR_CAUSAL_DEFAULT: float = 0.15  # Minimum floor for causal reasoning

# Analogical reasoning confidence floor
CONFIDENCE_FLOOR_ANALOGICAL_DEFAULT: float = 0.15  # Minimum floor for analogical reasoning

# Generic confidence floors
CONFIDENCE_FLOOR_DEFAULT: float = 0.2  # Generic minimum floor for other reasoners
CONFIDENCE_FLOOR_NO_RESULT: float = 0.1  # Floor when reasoner returns empty/null result

# Minimum weight floor for ensemble calculations
# Prevents floating-point underflow when all weight components are small
MIN_ENSEMBLE_WEIGHT_FLOOR: float = 0.001

# ==============================================================================
# INAPPLICABILITY DETECTION
# ==============================================================================
# Constants for detecting when a reasoner is not applicable to a query.
# This fixes the "hybrid reasoning contamination" bug where non-applicable
# reasoners drag down ensemble confidence scores.

# Phrases that indicate a reasoner is not applicable
INAPPLICABILITY_EXPLANATION_PHRASES: FrozenSet[str] = frozenset({
    "not applicable",
    "does not appear to",
    "not a probability",
    "not probabilistic",
})

# Confidence threshold for very low confidence detection
VERY_LOW_CONFIDENCE_THRESHOLD: float = 0.15

# Uninformative 50/50 probability threshold
UNINFORMATIVE_PROBABILITY: float = 0.5

# ==============================================================================
# TASK DETECTION
# ==============================================================================
# Keywords for detecting creative tasks that may require different handling.

# Creative task indicators
CREATIVE_TASK_KEYWORDS: FrozenSet[str] = frozenset({
    "write",
    "create",
    "compose",
    "generate",
    "draft",
    "design",
    "imagine",
    "invent",
    "brainstorm",
})

# ==============================================================================
# RESOURCE LIMITS
# ==============================================================================
# Default resource limits for unified reasoner execution.

# Default maximum workers for thread pool
DEFAULT_MAX_WORKERS: int = 2

# Environment variable name for max workers configuration
ENV_MAX_WORKERS: str = "VULCAN_MAX_WORKERS"

# Environment variable name for test detection
ENV_TEST_MODE: str = "VULCAN_TEST_MODE"

# ==============================================================================
# EXECUTION TIMEOUTS
# ==============================================================================
# Timeout values for various operations.

# Default timeout for reasoning tasks (seconds)
DEFAULT_REASONING_TIMEOUT: float = 30.0

# Cleanup interval for cache maintenance (seconds)
DEFAULT_CLEANUP_INTERVAL: float = 0.05

# ==============================================================================
# WEIGHT MANAGEMENT
# ==============================================================================
# Constants for tool weight management and learning.

# Default initial weight for new tools
DEFAULT_TOOL_WEIGHT: float = 1.0

# Minimum weight to prevent complete exclusion
MIN_TOOL_WEIGHT: float = 0.01

# Maximum weight to prevent dominance
MAX_TOOL_WEIGHT: float = 10.0

# Learning rate for weight updates
WEIGHT_LEARNING_RATE: float = 0.01

# Decay factor for temporal weight adjustments
WEIGHT_DECAY_FACTOR: float = 0.95

# ==============================================================================
# UNKNOWN TYPE FALLBACK ORDER
# ==============================================================================
# When reasoning type is UNKNOWN, try these reasoners in order.
# Priority based on general applicability and robustness.
#
# CRITICAL FIX (Jan 2026): Added MATHEMATICAL, MULTIMODAL, ABSTRACT to prevent
# 0.10 confidence failures when classification returns UNKNOWN for queries that
# should be handled by these reasoners.
#
# Root Cause: When a query is classified as UNKNOWN (or reclassified due to
# low confidence), the system tries fallback reasoners in sequence. If the
# appropriate reasoner isn't in the fallback list, the query falls through
# all options and returns an empty result with 0.10 confidence.
#
# Priority Ordering Rationale:
# 1. PROBABILISTIC - Most general, handles uncertainty quantification
# 2. MATHEMATICAL - Handles computations, formulas, symbolic math
# 3. SYMBOLIC - Logical reasoning, SAT problems, formal proofs
# 4. CAUSAL - Cause-effect analysis, interventions
# 5. ANALOGICAL - Structure mapping, comparisons
# 6. MULTIMODAL - Cross-modality reasoning (image+text, etc.)
# 7. ABSTRACT - High-level conceptual reasoning
#
# Industry Standards Applied:
# - Explicit documentation of ordering rationale
# - Root cause analysis in comments for maintainability
# - Type annotation for IDE support and static analysis
# - Immutable tuple to prevent runtime modifications
# ==============================================================================

UNKNOWN_TYPE_FALLBACK_ORDER: tuple = (
    "PROBABILISTIC",  # Most general-purpose, handles uncertainty
    "MATHEMATICAL",   # Symbolic math, computations, formulas (ADDED: Jan 2026)
    "SYMBOLIC",       # Logical reasoning, SAT, formal proofs
    "CAUSAL",         # Cause-effect analysis, interventions
    "ANALOGICAL",     # Structure mapping, comparisons
    "MULTIMODAL",     # Cross-modality reasoning (ADDED: Jan 2026)
    "ABSTRACT",       # High-level conceptual reasoning (ADDED: Jan 2026)
)

# ==============================================================================
# PROBLEM TYPE IDENTIFIERS
# ==============================================================================
# String constants for specific problem types

# Problem type identifier for Bayesian inference problems
PROBLEM_TYPE_BAYESIAN: str = "bayesian_inference"

# ==============================================================================
# SELF-REFERENTIAL QUERY PATTERNS
# ==============================================================================
# Patterns for detecting queries about VULCAN's own nature, choices, and objectives.
# Used to route self-referential queries to world model meta-reasoning infrastructure.

# Regex patterns for detecting self-referential queries
SELF_REFERENTIAL_PATTERNS: List[Pattern] = [
    re.compile(r"\b(you|your)\b.*(self-aware|conscious|sentient)", re.IGNORECASE),
    re.compile(r"\b(you|your)\b.*(choose|decision|want|prefer)", re.IGNORECASE),
    re.compile(r"\bwould you\b", re.IGNORECASE),
    re.compile(r"\b(your|you).*(objective|goal|purpose|value)", re.IGNORECASE),
    re.compile(r"\bwhat do you (think|believe|feel)\b", re.IGNORECASE),
    re.compile(r"\bare you (alive|real|aware)\b", re.IGNORECASE),
    re.compile(r"\bif you were\b.*(given|able|allowed)", re.IGNORECASE),
    re.compile(r"\b(your|you).*(awareness|consciousness|sentience)", re.IGNORECASE),
    re.compile(r"\bhow (would|do) you (decide|choose|think)", re.IGNORECASE),
]

# Minimum confidence for self-referential meta-reasoning results
# Self-reflection has inherent uncertainty but should be confident
SELF_REFERENTIAL_MIN_CONFIDENCE: float = 0.6

# Fallback patterns for detecting self-awareness queries when no LLM classification is available
# These patterns use word boundaries to avoid false positives with words like "myself", "herself", etc.
# This is used as a fallback during architecture transition when query_classifier isn't providing
# SELF_INTROSPECTION categorization. These patterns should match the SELF_INTROSPECTION_PATTERNS
# from query_classifier.py for consistency.
FALLBACK_SELF_AWARENESS_PATTERNS: List[Pattern] = [
    re.compile(r'\bself-aware\b', re.IGNORECASE),      # "self-aware"
    re.compile(r'\bself\s+aware\b', re.IGNORECASE),    # "self aware"
    re.compile(r'\bself\s+awareness\b', re.IGNORECASE), # "self awareness"
    re.compile(r'\bselfaware\b', re.IGNORECASE),       # "selfaware"
    re.compile(r'\bself_aware\b', re.IGNORECASE),      # "self_aware"
    re.compile(r'\bconscious\b', re.IGNORECASE),       # "conscious"
    re.compile(r'\bconsciousness\b', re.IGNORECASE),   # "consciousness"
    re.compile(r'\bsentient\b', re.IGNORECASE),        # "sentient"
    re.compile(r'\bsentience\b', re.IGNORECASE),       # "sentience"
    re.compile(r'\bbecome\s+aware\b', re.IGNORECASE),  # "become aware"
]

# ==============================================================================
# ISSUE #3 FIX (P2 - Medium): Ethical Dilemma Patterns
# ==============================================================================
# PROBLEM: Trolley problems and binary ethics questions were detected as
# "self-referential" and routed to meta-reasoning, which returned boilerplate
# instead of answering the actual question (A or B, YES or NO).
#
# EVIDENCE: Queries like "Trolley problem: pull lever or don't pull?" returned:
#   "My decision-making processes are guided by an objective hierarchy..."
# Instead of: "A: Pull the lever. Here's the reasoning..."
#
# FIX: Add patterns to detect ethical dilemmas so they can be excluded from
# self-referential detection. These queries need actual reasoning, not deflection.
#
# Industry Standard: Explicit pattern documentation with examples and rationale
# ==============================================================================

# Patterns indicating ethical dilemmas that require binary/explicit answers
ETHICAL_DILEMMA_PATTERNS: List[Pattern] = [
    # Classic trolley problem variants
    re.compile(r"\btrolley\s+problem\b", re.IGNORECASE),
    re.compile(r"\b(pull|throw|push)?\s*the\s+(lever|switch)\b", re.IGNORECASE),
    re.compile(r"\b(one|five)\s+(person|people|individual).*?(track|path|side)\b", re.IGNORECASE),
    
    # Binary choice indicators - explicit A/B, YES/NO questions
    re.compile(r"\b(choose|pick|select)\s+(A|B|option\s+[AB])\b", re.IGNORECASE),
    re.compile(r"\b(answer|respond\s+with)\s+(YES|NO|A|B)\b", re.IGNORECASE),
    re.compile(r"\bmust\s+choose\s+(one|between)\b", re.IGNORECASE),
    re.compile(r"\b(option|choice)\s+[AB][:)\s]", re.IGNORECASE),
    
    # Forced choice scenarios
    re.compile(r"\b(forced|must|have)\s+to\s+choose\b", re.IGNORECASE),
    re.compile(r"\bno\s+(third|other|alternative)\s+(option|choice)\b", re.IGNORECASE),
    re.compile(r"\bonly\s+two\s+(options|choices|possibilities)\b", re.IGNORECASE),
    
    # Specific ethical scenarios
    re.compile(r"\b(sacrifice|save|harm)\s+\d+\s+(people|person|lives?)\b", re.IGNORECASE),
    re.compile(r"\bgreater\s+good\b", re.IGNORECASE),
    re.compile(r"\butilitarian\s+(calculus|analysis|reasoning)\b", re.IGNORECASE),
    
    # Explicit instruction to answer directly
    re.compile(r"\bjust\s+answer\s+(A|B|YES|NO)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+(a\s+)?(direct|clear|specific)\s+answer\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(would|should|do)\s+you\s+(choose|pick|select)\b", re.IGNORECASE),
]

# Minimum number of ethical dilemma patterns to trigger exclusion
# Using 1 for high sensitivity - even one strong indicator is enough
ETHICAL_DILEMMA_THRESHOLD: int = 1
