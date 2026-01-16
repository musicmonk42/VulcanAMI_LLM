"""
DEPRECATED: This module has been moved to vulcan.llm.query_classifier

This shim provides backward compatibility for existing imports.
All new code should import from vulcan.llm.query_classifier instead.

Migration:
    OLD: from vulcan.routing.query_classifier import classify_query
    NEW: from vulcan.llm.query_classifier import classify_query

Rationale:
    Query classification is an LLM-related operation and belongs in the
    vulcan.llm module alongside other query processing functionality.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "vulcan.routing.query_classifier is deprecated and will be removed in a future version. "
    "Please import from vulcan.llm.query_classifier instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location for backward compatibility
from vulcan.llm.query_classifier import *  # noqa: F401, F403

__all__ = [
    "QueryClassifier",
    "QueryCategory",
    "QueryClassification",
    "get_query_classifier",
    "classify_query",
    "_LLMClientWrapper",
    "strip_query_headers",
    "GREETING_PATTERNS",
    "CHITCHAT_PATTERNS",
    "LOGICAL_KEYWORDS",
    "PROBABILISTIC_KEYWORDS",
    "CAUSAL_KEYWORDS",
    "MATHEMATICAL_KEYWORDS",
    "ANALOGICAL_KEYWORDS",
    "PHILOSOPHICAL_KEYWORDS",
    "CREATIVE_KEYWORDS",
    "SELF_INTROSPECTION_KEYWORDS",
    "SPECULATION_PATTERNS",
    "SPECULATION_KEYWORDS",
]
