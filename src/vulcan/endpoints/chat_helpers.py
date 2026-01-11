"""
Chat Helper Functions and Constants

Shared utilities for chat endpoints including context management,
history truncation, and formatting functions.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# CONTEXT ACCUMULATION FIX: Constants for context window management
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_TOKENS = 4096
MAX_MESSAGE_LENGTH = 2000
MIN_MESSAGE_LENGTH = 50
MIN_TRUNCATION_HALF = 10
ELLIPSIS_OVERHEAD = 15
CHARS_PER_TOKEN_ESTIMATE = 3

# JOB-TO-RESPONSE GAP FIX: Timing thresholds
SLOW_PHASE_THRESHOLD_MS = 1000
SLOW_REQUEST_THRESHOLD_MS = 5000
SLOW_QUERY_OUTCOME_THRESHOLD_MS = 30000
SLOW_ROUTING_OUTCOME_THRESHOLD_MS = 10000
SLOW_TOTAL_OUTCOME_THRESHOLD_MS = 30000

# MEMORY MANAGEMENT: GC thresholds
GC_SIGNIFICANT_CLEANUP_THRESHOLD = 100
GC_REQUEST_INTERVAL = 10

# REASONING OUTPUT FORMATTING: Limits
MAX_REASONING_RESULT_LENGTH = 1500
MAX_ANALOGIES_TO_SHOW = 5
MAX_LIST_ITEMS_TO_SHOW = 10
MAX_REASONING_STEPS = 5

# WORLD MODEL INSIGHT FORMATTING: Truncation limits
WORLD_MODEL_INSIGHT_TRUNCATION = 200
WORLD_MODEL_LOG_TRUNCATION = 100

# Configuration constants for response building
CONTEXT_TRUNCATION_LIMITS = {
    "memory": 300,
    "reasoning": 400,
    "world_model": 300,
    "meta_reasoning": 200,
}
MIN_MEANINGFUL_RESPONSE_LENGTH = 10
MOCK_RESPONSE_MARKER = "Mock response"

# Agent reasoning collection constants
AGENT_REASONING_POLL_DELAY_SEC = 0.1
MAX_AGENT_REASONING_JOBS_TO_CHECK = 3

# Format_dict_result keys with explicit formatting
HANDLED_DICT_RESULT_KEYS = frozenset({
    'conclusion', 'explanation', 'reasoning_type', 'reasoning_steps',
    'premises', 'proof_steps', 'probability', 'confidence',
    'causal_path', 'intervention', 'counterfactual', 'analogies',
    'source_domain', 'target_domain'
})


def safe_truncate_utf8(text: str, max_chars: int, ellipsis: str = "...") -> str:
    """
    Safely truncate text respecting UTF-8 character boundaries.
    
    SECURITY FIX: Character-based slicing can split multi-byte UTF-8 sequences,
    causing corruption or crashes. This function ensures clean truncation.
    
    Industry best practice: Always use encode/decode with error handling when
    truncating strings that may contain multi-byte characters.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters in result (including ellipsis)
        ellipsis: Ellipsis marker to append if truncated
        
    Returns:
        Safely truncated text respecting UTF-8 boundaries
        
    Examples:
        >>> safe_truncate_utf8("Hello 世界", 8, "...")
        'Hello...'
        >>> safe_truncate_utf8("Hello", 10, "...")
        'Hello'
    """
    if len(text) <= max_chars:
        return text
    
    # Reserve space for ellipsis
    truncate_at = max(0, max_chars - len(ellipsis))
    
    # Truncate and use encode/decode with 'ignore' to handle broken UTF-8 sequences
    # This prevents crashes from split multi-byte characters
    truncated = text[:truncate_at].encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    
    return truncated + ellipsis


def safe_truncate_middle(text: str, max_chars: int) -> str:
    """
    Safely truncate text from the middle, preserving start and end.
    
    SECURITY FIX: Respects UTF-8 boundaries to prevent corruption.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters in result
        
    Returns:
        Safely truncated text with middle section replaced by ellipsis
        
    Examples:
        >>> safe_truncate_middle("ABCDEFGHIJ", 8)
        'ABC...HIJ'
    """
    if len(text) <= max_chars:
        return text
    
    ellipsis_marker = "\n... [truncated] ...\n"
    half = max(MIN_TRUNCATION_HALF, (max_chars - len(ellipsis_marker)) // 2)
    
    # Use safe_truncate_utf8 for each half to ensure clean boundaries
    start = text[:half].encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    end = text[-half:].encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    
    return start + ellipsis_marker + end


def truncate_history(
    history: List[Dict[str, str]],
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_tokens: int = MAX_HISTORY_TOKENS,
    max_message_length: int = MAX_MESSAGE_LENGTH,
) -> List[Dict[str, str]]:
    """
    Truncate conversation history to prevent context accumulation.
    
    CONTEXT ACCUMULATION FIX: Implements sliding window approach to
    prevent unbounded history growth causing response lag.
    
    Args:
        history: List of message dicts with 'role' and 'content' keys
        max_messages: Maximum number of messages to retain
        max_tokens: Maximum estimated tokens in history
        max_message_length: Maximum characters per message
        
    Returns:
        Truncated history list
    """
    if not history:
        return []
    
    max_message_length = max(MIN_MESSAGE_LENGTH, max_message_length)
    
    # Step 1: Sliding window - keep recent messages
    if len(history) > max_messages:
        logger.debug(
            f"Context truncation: Sliding window {len(history)} → {max_messages} messages"
        )
        history = history[-max_messages:]
    
    # Step 2: Truncate individual long messages using safe UTF-8 truncation
    truncated_history = []
    for msg in history:
        truncated_msg = dict(msg)
        content = truncated_msg.get("content", "")
        if len(content) > max_message_length:
            # SECURITY FIX: Use safe_truncate_middle to prevent UTF-8 corruption
            truncated_content = safe_truncate_middle(content, max_message_length)
            if len(truncated_content) < len(content):
                truncated_msg["content"] = truncated_content
                logger.debug(
                    f"Context truncation: Message safely truncated from {len(content)} to {len(truncated_msg['content'])} chars"
                )
        truncated_history.append(truncated_msg)
    
    # Step 3: Estimate token count and drop oldest if over limit
    total_chars = sum(len(msg.get("content", "")) for msg in truncated_history)
    estimated_tokens = total_chars // CHARS_PER_TOKEN_ESTIMATE
    
    while estimated_tokens > max_tokens and len(truncated_history) > 1:
        removed = truncated_history.pop(0)
        removed_chars = len(removed.get("content", ""))
        estimated_tokens -= removed_chars // CHARS_PER_TOKEN_ESTIMATE
        logger.debug(
            f"Context truncation: Dropped oldest message ({removed_chars} chars), "
            f"estimated tokens now: {estimated_tokens}"
        )
    
    return truncated_history


def build_context(
    current_query: str, 
    history: List[Dict[str, str]], 
    reasoning_results: Optional[Dict[str, Any]] = None,
    max_history: int = 3,
) -> Dict[str, Any]:
    """
    Build context with recency bias to prevent old queries bleeding in.
    
    CONTEXT BLEEDING FIX: Implements recency weighting so conversation
    history provides context without contaminating the current response.
    
    Args:
        current_query: Current user query to answer
        history: List of previous exchanges
        reasoning_results: Optional reasoning engine outputs
        max_history: Maximum recent exchanges to include
        
    Returns:
        Context dict with weighted history
    """
    context = {
        "current_query": current_query,
        "weight": 1.0,
    }
    
    if history:
        weighted_history = []
        recent_exchanges = list(reversed(history[-max_history:]))
        
        for i, exchange in enumerate(recent_exchanges):
            weight = 0.3 * (0.5 ** i)
            weighted_history.append({
                "content": exchange,
                "weight": weight,
                "note": "Context only - NOT the current question",
            })
        
        context["history"] = weighted_history
    
    if reasoning_results:
        context["reasoning"] = reasoning_results
    
    context["instruction"] = (
        "Answer ONLY the current_query. "
        "History is for context, not to be re-answered. "
        "Do not respond to previous queries."
    )
    
    return context
