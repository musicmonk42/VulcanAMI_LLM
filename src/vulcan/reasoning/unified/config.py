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

from typing import FrozenSet

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
# Priority based on general applicability.

UNKNOWN_TYPE_FALLBACK_ORDER: tuple = (
    "PROBABILISTIC",  # Most general-purpose
    "SYMBOLIC",       # Good for logical queries
    "CAUSAL",         # Good for cause-effect queries
    "ANALOGICAL",     # Good for comparison queries
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

import re
from typing import List, Pattern

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
