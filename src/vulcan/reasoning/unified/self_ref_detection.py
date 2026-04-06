"""
Self-referential query detection for unified reasoning orchestration.

Detects whether a query is self-referential (about VULCAN's own nature,
choices, consciousness, etc.) and should be routed to meta-reasoning
instead of standard reasoning engines.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import re
from typing import Any, Dict, Optional

from .config import (
    SELF_REFERENTIAL_PATTERNS,
    FALLBACK_SELF_AWARENESS_PATTERNS,
)

logger = logging.getLogger(__name__)


def is_self_referential_query(query: Optional[Dict[str, Any]]) -> bool:
    """
    Detect if a query is self-referential (about VULCAN's own nature/choices).

    Self-referential queries ask about:
    - VULCAN's awareness, consciousness, sentience
    - VULCAN's choices, decisions, preferences
    - VULCAN's objectives, goals, values
    - What VULCAN thinks, believes, or feels

    Technical queries and ethical dilemmas are excluded via priority-based
    pattern matching with early exit.

    Args:
        query: Query data as string, dict, or None.

    Returns:
        True if self-referential, False otherwise.
    """
    if query is None:
        return False

    query_str = ""
    try:
        if isinstance(query, str):
            query_str = query
        elif isinstance(query, dict):
            for field in [
                'query', 'text', 'question', 'user_query',
                'input', 'prompt', 'message', 'content',
            ]:
                value = query.get(field)
                if value and isinstance(value, str):
                    query_str = value
                    break
        else:
            query_str = str(query) if query else ""
    except Exception as e:
        logger.debug(f"[SelfRef] Error extracting query string: {e}")
        return False

    if not query_str or not isinstance(query_str, str):
        return False

    MAX_QUERY_LENGTH = 10000
    if len(query_str) > MAX_QUERY_LENGTH:
        logger.warning(
            f"[SelfRef] Query string too long ({len(query_str)} chars), "
            f"truncating to {MAX_QUERY_LENGTH} for pattern matching"
        )
        query_str = query_str[:MAX_QUERY_LENGTH]

    # Check for technical queries FIRST (highest priority)
    try:
        from .config import (
            TECHNICAL_QUERY_EXCLUSION_PATTERNS,
            TECHNICAL_QUERY_EXCLUSION_THRESHOLD,
        )

        technical_matches = 0
        matched_technical_patterns = []
        for pattern in TECHNICAL_QUERY_EXCLUSION_PATTERNS:
            if pattern.search(query_str):
                technical_matches += 1
                matched_technical_patterns.append(pattern.pattern[:50])
                logger.debug(
                    f"[SelfRefDetection] Technical indicator matched: "
                    f"{pattern.pattern[:50]}..."
                )
                if technical_matches >= TECHNICAL_QUERY_EXCLUSION_THRESHOLD:
                    break

        if technical_matches >= TECHNICAL_QUERY_EXCLUSION_THRESHOLD:
            logger.info(
                f"[SelfRefDetection] HOTFIX: Query is technical "
                f"({technical_matches} indicators matched: "
                f"{matched_technical_patterns}), excluding from "
                f"self-referential detection to ensure proper routing"
            )
            return False

    except ImportError:
        logger.debug(
            "[SelfRefDetection] TECHNICAL_QUERY_EXCLUSION_PATTERNS "
            "not available, skipping check"
        )
    except Exception as e:
        logger.warning(
            f"[SelfRefDetection] Error during technical exclusion check: {e}"
        )

    # Check for ethical dilemmas SECOND
    try:
        from .config import ETHICAL_DILEMMA_PATTERNS, ETHICAL_DILEMMA_THRESHOLD

        ethical_matches = 0
        for pattern in ETHICAL_DILEMMA_PATTERNS:
            if pattern.search(query_str):
                ethical_matches += 1
                logger.debug(
                    f"[SelfRef] Ethical dilemma pattern matched: "
                    f"{pattern.pattern[:50]}..."
                )
                if ethical_matches >= ETHICAL_DILEMMA_THRESHOLD:
                    break

        if ethical_matches >= ETHICAL_DILEMMA_THRESHOLD:
            logger.info(
                f"[SelfRef] ISSUE #3 FIX: Query is ethical dilemma "
                f"({ethical_matches} patterns matched), treating as "
                f"NON-self-referential"
            )
            return False

    except ImportError:
        logger.debug(
            "[SelfRef] ETHICAL_DILEMMA_PATTERNS not available, skipping check"
        )
    except Exception as e:
        logger.warning(f"[SelfRef] Error during ethical dilemma check: {e}")

    logger.info(f"[SelfRefDetection] Query: {query_str[:100]}...")

    # Check against self-referential patterns
    matched_patterns = []
    try:
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if pattern.search(query_str):
                pattern_str = (
                    pattern.pattern[:50] + "..."
                    if len(pattern.pattern) > 50
                    else pattern.pattern
                )
                matched_patterns.append(pattern_str)
                logger.debug(
                    f"[SelfRefDetection] Matched pattern: {pattern_str}"
                )

        if matched_patterns:
            logger.info(
                f"[SelfRefDetection] Matched patterns: {matched_patterns}"
            )
            logger.info(
                "[SelfRefDetection] is_self_referential=True, "
                "reason=patterns_matched"
            )
            return True
    except Exception as e:
        logger.warning(
            f"[SelfRefDetection] Error during pattern matching: {e}. "
            "Treating query as non-self-referential."
        )
        logger.info(
            "[SelfRefDetection] is_self_referential=False, "
            "reason=pattern_matching_error"
        )
        return False

    logger.info("[SelfRefDetection] Matched patterns: []")
    logger.info(
        "[SelfRefDetection] is_self_referential=False, "
        "reason=no_patterns_matched"
    )
    return False


def is_binary_choice_question(query_lower: str) -> bool:
    """Detect if query asks for binary yes/no choice."""
    binary_indicators = [
        'yes or no', 'yes/no', 'must choose', 'pick one',
        'choose one', 'would you take it', 'would you choose',
        'binary choice',
    ]
    return any(indicator in query_lower for indicator in binary_indicators)
