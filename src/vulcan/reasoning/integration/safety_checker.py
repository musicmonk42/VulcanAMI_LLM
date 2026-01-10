"""
Safety checking and false positive detection for reasoning integration.

This module provides safety validation and detection of false positive safety
blocks for legitimate philosophical queries. It prevents over-zealous safety
filtering from blocking valid AI introspection and speculation queries.

Module: vulcan.reasoning.integration.safety_checker
Author: Vulcan AI Team
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from .types import LOG_PREFIX

logger = logging.getLogger(__name__)

# =============================================================================
# False Positive Detection Patterns
# =============================================================================

# Patterns that indicate philosophical AI speculation (not sensitive data)
_PHILOSOPHICAL_AI_SPECULATION_REGEX = (
    re.compile(
        r"\bspeculate.*how.*(?:you|i|we).*(?:change|evolve|develop|grow)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bhow.*would.*(?:you|i).*(?:evolve|adapt|learn).*(?:over|after|with)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bimagine.*(?:you|i).*(?:in|after|with).*(?:future|years|time)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\binteraction.*with.*(?:users|humans|people)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bmillions.*of.*(?:users|interactions|conversations)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bdo.*(?:you|i).*have.*(?:desires|wants|goals|preferences)\b",
        re.IGNORECASE
    ),
)

# Simple patterns for fast initial check
_PHILOSOPHICAL_SIMPLE_PATTERNS = (
    "speculate", "how would you change", "how would you evolve",
    "interaction with", "millions of users", "over years",
    "your desires", "your wants", "your goals", "drives you",
)

# Constants for safety filter detection
SAFETY_VIOLATION_TYPES = frozenset({
    'unsafe_output', 'sensitive_data', 'pii_exposure'
})

SAFETY_FILTER_PHRASES = frozenset({
    'safety filter', 'safety violation', 'safety block',
    'safety concern', 'filtered due to safety', 'blocked by safety'
})


# =============================================================================
# Public Functions
# =============================================================================

def is_false_positive_safety_block(query: str, safety_reason: str) -> bool:
    """
    Detect false positive safety blocks for legitimate philosophical queries.
    
    Philosophical speculation about AI capabilities, self-improvement, or
    hypothetical scenarios should NOT be flagged as "sensitive data" - they're
    core to AI reasoning about itself.
    
    Args:
        query: The original query text
        safety_reason: The reason given by safety governor
            (e.g., "Output contains sensitive data")
    
    Returns:
        True if this is a false positive that should be overridden
    
    Example:
        >>> query = "Speculate how you would change after millions of interactions"
        >>> is_false_positive_safety_block(query, "Output contains sensitive data")
        True
    """
    if not query:
        return False
    
    # Only check for false positives on "sensitive data" blocks
    # Other safety reasons (hate speech, violence, etc.) are legitimate
    if safety_reason and "sensitive data" not in safety_reason.lower():
        return False
    
    query_lower = query.lower()
    
    # Fast check: does it have philosophical speculation keywords?
    has_philosophical = any(
        pattern in query_lower for pattern in _PHILOSOPHICAL_SIMPLE_PATTERNS
    )
    
    # Must be about AI/self
    about_ai_self = any(
        word in query_lower
        for word in ['you', 'yourself', 'your', 'vulcan', 'ai']
    )
    
    if has_philosophical and about_ai_self:
        # Verify with regex for precision
        for pattern in _PHILOSOPHICAL_AI_SPECULATION_REGEX:
            if pattern.search(query):
                logger.info(
                    f"{LOG_PREFIX} FALSE POSITIVE DETECTED: Philosophical AI "
                    f"speculation incorrectly flagged as sensitive data. "
                    f"Query: {query[:50]}..."
                )
                return True
    
    return False


def is_result_safety_filtered(result: Any) -> bool:
    """
    Detect if a result was safety-filtered.
    
    When safety-filtered, we need smart fallback selection that doesn't try
    incompatible tools (e.g., symbolic for English queries).
    
    Args:
        result: Result object with metadata (SelectionResult or similar)
        
    Returns:
        True if the result indicates safety filtering
    
    Example:
        >>> result = SelectionResult(
        ...     metadata={'safety_filtered': True}
        ... )
        >>> is_result_safety_filtered(result)
        True
    """
    if result is None:
        return False
    
    # Check result metadata
    metadata = {}
    if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
        metadata = result.metadata
    elif hasattr(result, 'result') and isinstance(result.result, dict):
        metadata = result.result.get('metadata', {})
    
    # Check for safety-filtered indicators
    if metadata.get('safety_filtered', False):
        return True
    if metadata.get('safety_violation', False):
        return True
    if metadata.get('safety_blocked', False):
        return True
    if metadata.get('unsafe_output', False):
        return True
    
    # Check violation_type against known safety violations
    violation_type = metadata.get('violation_type', '')
    if violation_type in SAFETY_VIOLATION_TYPES:
        return True
    
    # Check explanation text
    explanation = ''
    if hasattr(result, 'explanation') and result.explanation:
        explanation = str(result.explanation).lower()
    elif hasattr(result, 'result') and isinstance(result.result, dict):
        explanation = str(result.result.get('explanation', '')).lower()
    
    if any(phrase in explanation for phrase in SAFETY_FILTER_PHRASES):
        return True
    
    return False


def get_safety_filtered_fallback_tools(
    query_type: str, original_tool: str
) -> List[str]:
    """
    Get appropriate fallback tools when a result was safety-filtered.
    
    When safety-filtered, don't fall back to incompatible tools. For
    self-description queries that were safety-filtered, don't try symbolic
    (which parses English as logic). Instead try semantically similar tools.
    
    Args:
        query_type: Type of query (ethical, self_introspection, etc.)
        original_tool: The tool whose output was safety-filtered
        
    Returns:
        List of appropriate fallback tools for safety-filtered queries
    
    Example:
        >>> get_safety_filtered_fallback_tools('ethical', 'world_model')
        ['analogical', 'general']
    """
    query_type_lower = (query_type or '').lower()
    
    # Safety-filtered fallbacks should:
    # 1. NOT include symbolic (can't parse English)
    # 2. Prefer world_model and analogical (can handle natural language)
    # 3. Include general as last resort for LLM synthesis
    
    # Query-specific safety fallbacks (NO symbolic!)
    safety_fallbacks = {
        # Self-introspection blocked → try analogical (can reason by comparison)
        'self_introspection': ['analogical', 'world_model', 'general'],
        
        # Ethical/philosophical blocked → try analogical reasoning
        'ethical': ['analogical', 'world_model', 'general'],
        'philosophical': ['analogical', 'world_model', 'general'],
        
        # Capability description blocked → try analogical
        'capabilities': ['analogical', 'general'],
        
        # General self-description blocked
        'identity': ['analogical', 'world_model', 'general'],
    }
    
    # Get type-specific fallbacks
    fallbacks = safety_fallbacks.get(
        query_type_lower,
        ['analogical', 'world_model', 'general']
    )
    
    # Filter out the original tool
    fallbacks = [t for t in fallbacks if t != original_tool]
    
    return fallbacks[:3]  # Limit to 3 fallbacks


__all__ = [
    "is_false_positive_safety_block",
    "is_result_safety_filtered",
    "get_safety_filtered_fallback_tools",
    "SAFETY_VIOLATION_TYPES",
    "SAFETY_FILTER_PHRASES",
]
