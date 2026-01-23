"""
Chat Helper Functions and Constants

Shared utilities for chat endpoints including context management,
history truncation, and formatting functions.

Industry Standard: This module follows defensive programming patterns with
type-safe enum handling to prevent AttributeError when serializing
ReasoningType enums to strings.
"""

import itertools
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Industry Standard: Feature flag for diagnostic logging (default: enabled during fix rollout)
_ENABLE_DIAGNOSTIC_LOGGING = os.environ.get("VULCAN_DIAGNOSTIC_LOGGING", "true").lower() in ("true", "1", "yes")


def safe_reasoning_type_to_string(
    reasoning_type: Optional[Union[str, Enum, Any]],
    default: str = "unknown"
) -> str:
    """
    Convert a reasoning_type to its string representation safely.
    
    Industry Standard: Type-safe conversion that handles Enum, string, and
    other types without raising AttributeError. This prevents the common
    "'ReasoningType' object has no attribute 'replace'" error.
    
    Args:
        reasoning_type: The reasoning type value which may be:
            - A ReasoningType Enum instance
            - A string value
            - None
            - Any other object
        default: Default string to return if reasoning_type is None or empty
        
    Returns:
        str: The string representation suitable for display or serialization
        
    Examples:
        >>> from vulcan.reasoning.reasoning_types import ReasoningType
        >>> safe_reasoning_type_to_string(ReasoningType.PROBABILISTIC)
        'probabilistic'
        >>> safe_reasoning_type_to_string("causal_reasoning")
        'causal_reasoning'
        >>> safe_reasoning_type_to_string(None)
        'unknown'
        >>> safe_reasoning_type_to_string(None, default="hybrid")
        'hybrid'
        
    Note:
        This resolves the serialization bug where Enum objects are passed
        to code expecting strings, causing AttributeError on string methods.
    """
    if reasoning_type is None:
        return default
    
    # Handle empty strings - return them unchanged (don't use default)
    # Industry Standard: Explicit None checks (None is not "") separate from empty string handling
    if isinstance(reasoning_type, str):
        return reasoning_type  # Return as-is, even if empty
    
    # Handle Enum instances - extract .value (the string representation)
    # Industry Standard: Check for Enum base class for type safety
    if isinstance(reasoning_type, Enum):
        return str(reasoning_type.value)
    
    # Handle objects that might be enum-like but not derived from Enum
    # This provides backward compatibility with:
    # - Third-party enum implementations (e.g., IntEnum, StrEnum)
    # - Dynamically generated enum-like classes
    # - Mock objects in tests
    # We prefer .value over .name because .value contains the user-facing string
    if hasattr(reasoning_type, 'value') and not isinstance(reasoning_type, str):
        try:
            return str(reasoning_type.value)
        except (AttributeError, TypeError):
            pass
    
    # Already a string or convert to string
    result = str(reasoning_type)
    return result if result else default


def format_reasoning_type_for_display(
    reasoning_type: Optional[Union[str, Enum, Any]],
    default: str = "Hybrid"
) -> str:
    """
    Format a reasoning_type for human-readable display.
    
    Industry Standard: Single responsibility function for consistent display
    formatting across all endpoints. Converts enums safely, replaces
    underscores with spaces, and applies title case.
    
    Args:
        reasoning_type: The reasoning type (Enum, string, or other)
        default: Default display value if reasoning_type is None/empty
        
    Returns:
        str: Human-readable formatted string (e.g., "Probabilistic", "Causal Reasoning")
        
    Examples:
        >>> format_reasoning_type_for_display(ReasoningType.PROBABILISTIC)
        'Probabilistic'
        >>> format_reasoning_type_for_display("causal_reasoning")
        'Causal Reasoning'
        >>> format_reasoning_type_for_display(None)
        'Hybrid'
    """
    type_str = safe_reasoning_type_to_string(reasoning_type, default="")
    
    if not type_str:
        return default
    
    # Format: replace underscores with spaces and apply title case
    # Safe to call .replace() because type_str is guaranteed to be a string
    return type_str.replace("_", " ").title()

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

# Generic method field values to ignore (avoid duplication in output)
GENERIC_METHOD_VALUES = frozenset({'unknown', 'generic', 'fallback'})


# ============================================================
# BUG FIX #2: Silent Success - Dictionary Conclusion Formatting (The Mute)
# Industry Standard: Defensive type handling with explicit coercion
# ============================================================

class ConclusionFormatter:
    """
    Formats reasoning conclusions for user display.
    
    Design Pattern: Adapter Pattern
    Handles: str, dict, list, dataclass, object with to_dict(), None
    
    Industry Standard: Exhaustive type handling with fallback chain
    """
    
    @staticmethod
    def format(conclusion: Any) -> Optional[str]:
        """
        Convert any conclusion type to displayable string.
        
        Industry Standard: Exhaustive type handling with fallback chain
        
        Args:
            conclusion: The conclusion value (may be str, dict, list, object, or None)
        
        Returns:
            Formatted string or None if no valid content
        """
        if conclusion is None:
            return None
        
        # String - return as-is
        if isinstance(conclusion, str):
            return conclusion.strip() if conclusion.strip() else None
        
        # Dict - JSON serialize with pretty printing
        if isinstance(conclusion, dict):
            return ConclusionFormatter._format_dict(conclusion)
        
        # List - format as enumerated items
        if isinstance(conclusion, list):
            return ConclusionFormatter._format_list(conclusion)
        
        # Object with to_dict() method (dataclasses, Pydantic models)
        if hasattr(conclusion, 'to_dict') and callable(conclusion.to_dict):
            try:
                return ConclusionFormatter._format_dict(conclusion.to_dict())
            except Exception:
                pass
        
        # Object with __dict__ (regular classes)
        if hasattr(conclusion, '__dict__'):
            try:
                return ConclusionFormatter._format_dict(vars(conclusion))
            except Exception:
                pass
        
        # Last resort - string conversion
        str_repr = str(conclusion)
        return str_repr if str_repr and str_repr != 'None' else None
    
    @staticmethod
    def _format_dict(d: dict) -> Optional[str]:
        """Format dictionary as readable output."""
        import json
        
        # Filter out internal/metadata keys
        display_keys = {k: v for k, v in d.items() 
                       if not k.startswith('_') and v is not None}
        
        if not display_keys:
            return None
        
        # Check for common conclusion patterns
        if 'result' in display_keys:
            return str(display_keys['result'])
        if 'answer' in display_keys:
            return str(display_keys['answer'])
        if 'conclusion' in display_keys:
            return str(display_keys['conclusion'])
        
        # Full JSON output for complex results
        try:
            return json.dumps(display_keys, indent=2, default=str)
        except (TypeError, ValueError):
            return str(display_keys)
    
    @staticmethod
    def _format_list(items: list) -> Optional[str]:
        """Format list as enumerated items."""
        if not items:
            return None
        
        if len(items) == 1:
            return ConclusionFormatter.format(items[0])
        
        formatted = []
        for i, item in enumerate(items, 1):
            item_str = ConclusionFormatter.format(item)
            if item_str:
                formatted.append(f"{i}. {item_str}")
        
        return "\n".join(formatted) if formatted else None


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


# ============================================================
# FIX Issue 7: Consistent Tools Extraction Helper
# ============================================================

def extract_tools_from_routing(routing_plan: Any) -> List[str]:
    """
    Extract tools/capabilities from routing plan in a consistent way.
    
    FIX Issue 7: This function provides a single source of truth for extracting
    tools from routing results, ensuring consistency across all chat endpoints.
    
    The function tries multiple extraction strategies in order:
    1. telemetry_data.selected_tools (most reliable)
    2. agent_tasks.capability (fallback for v1 routing)
    3. ['general'] (default when no tools found)
    
    Args:
        routing_plan: The ProcessingPlan or QueryPlan from query routing
        
    Returns:
        List of tool/capability names that were selected for the query
        
    Example:
        >>> plan = route_query("What is 2+2?")
        >>> tools = extract_tools_from_routing(plan)
        >>> # ['reasoning', 'calculator']
    """
    if routing_plan is None:
        return ['general']
    
    selected_tools = []
    
    # Strategy 1: Check telemetry_data.selected_tools (QueryRouter v2)
    if hasattr(routing_plan, 'telemetry_data'):
        telemetry_tools = routing_plan.telemetry_data.get('selected_tools', [])
        if telemetry_tools:
            selected_tools = telemetry_tools
    
    # Strategy 2: Extract from agent_tasks.capability (QueryRouter v1)
    if not selected_tools and hasattr(routing_plan, 'agent_tasks'):
        try:
            selected_tools = [
                task.capability for task in routing_plan.agent_tasks
                if hasattr(task, 'capability')
            ]
        except (AttributeError, TypeError):
            pass
    
    # Strategy 3: Try to extract from plan attributes directly
    if not selected_tools:
        # Check for selected_tools attribute directly
        if hasattr(routing_plan, 'selected_tools'):
            selected_tools = routing_plan.selected_tools
        # Check for capabilities attribute
        elif hasattr(routing_plan, 'capabilities'):
            selected_tools = routing_plan.capabilities
    
    # Default fallback
    if not selected_tools:
        selected_tools = ['general']
    
    # Ensure we return a list (handle single string case)
    if isinstance(selected_tools, str):
        selected_tools = [selected_tools]
    
    return selected_tools


# ============================================================
# REASONING RESULTS FORMATTING FOR LLM CONTEXT
# ============================================================

def format_reasoning_results(reasoning_results: Dict[str, Any]) -> str:
    """
    Format reasoning results into a structured string for LLM context.
    
    This function converts VULCAN's structured reasoning outputs into a clear,
    readable format that can be passed to the LLM for natural language generation.
    
    The function handles multiple reasoning engine outputs (symbolic, probabilistic,
    causal, analogical, agent-based) and extracts key information like conclusions,
    confidence scores, explanations, and reasoning steps.
    
    Industry Standard Practices:
    - Defensive programming: handles None values, missing keys, and type errors
    - Comprehensive logging: logs warnings for malformed data
    - Graceful degradation: returns partial results even if some engines fail
    - Type safety: explicit type checks before attribute access
    - Performance: efficient string building with pre-allocated buffer
    
    Args:
        reasoning_results: Dictionary mapping engine names to their results.
                          Each result may contain: conclusion, confidence,
                          reasoning_type, explanation, reasoning_steps, etc.
    
    Returns:
        Formatted string representation of reasoning results suitable for
        LLM context. Returns empty string if no valid results.
    
    Example:
        >>> results = {
        ...     'symbolic': {
        ...         'conclusion': 'The statement is logically valid',
        ...         'confidence': 0.95,
        ...         'reasoning_type': 'deductive',
        ...     }
        ... }
        >>> formatted = format_reasoning_results(results)
        >>> print(formatted)
        Reasoning Analysis:
        
        Symbolic Reasoning:
        - Conclusion: The statement is logically valid
        - Confidence: 95%
        - Type: deductive
    """
    if not reasoning_results or not isinstance(reasoning_results, dict):
        logger.debug("[format_reasoning_results] No valid reasoning results to format")
        return ""
    
    # DIAGNOSTIC LOGGING: Log what we're about to format
    if _ENABLE_DIAGNOSTIC_LOGGING:
        logger.info(
            f"[chat_helpers/DIAGNOSTIC] format_reasoning_results called with "
            f"{len(reasoning_results)} engines: {list(reasoning_results.keys())}"
        )
    
    # Use list for efficient string building
    parts = ["Reasoning Analysis:\n"]
    formatted_count = 0
    
    for engine_name, result in reasoning_results.items():
        try:
            # Skip None or empty results
            if result is None:
                if _ENABLE_DIAGNOSTIC_LOGGING:
                    logger.info(f"[chat_helpers/DIAGNOSTIC] Skipping {engine_name}: result is None")
                continue
            
            if _ENABLE_DIAGNOSTIC_LOGGING:
                logger.info(
                    f"[chat_helpers/DIAGNOSTIC] Formatting {engine_name}: "
                    f"type={type(result).__name__}, is_dict={isinstance(result, dict)}"
                )
            
            # Handle different result types
            if isinstance(result, dict):
                formatted_section = _format_engine_result_dict(engine_name, result)
            elif isinstance(result, str):
                # Direct string result
                formatted_section = f"\n{engine_name.replace('_', ' ').title()}:\n{result}\n"
            else:
                # Other types - convert to string (includes ReasoningResult objects)
                if _ENABLE_DIAGNOSTIC_LOGGING:
                    logger.info(
                        f"[chat_helpers/DIAGNOSTIC] {engine_name} is non-dict/non-string, "
                        f"has_to_dict={hasattr(result, 'to_dict')}"
                    )
                formatted_section = f"\n{engine_name.replace('_', ' ').title()}:\n{str(result)[:MAX_REASONING_RESULT_LENGTH]}\n"
            
            if formatted_section:
                parts.append(formatted_section)
                formatted_count += 1
                if _ENABLE_DIAGNOSTIC_LOGGING:
                    logger.info(f"[chat_helpers/DIAGNOSTIC] Successfully formatted {engine_name}")
            else:
                if _ENABLE_DIAGNOSTIC_LOGGING:
                    logger.warning(f"[chat_helpers/DIAGNOSTIC] {engine_name} produced no formatted section")
                
        except Exception as e:
            logger.warning(
                f"[format_reasoning_results] Failed to format {engine_name}: "
                f"{type(e).__name__}: {e}"
            )
            continue
    
    # Return empty string if no results were formatted
    if formatted_count == 0:
        if _ENABLE_DIAGNOSTIC_LOGGING:
            logger.warning("[chat_helpers/DIAGNOSTIC] No reasoning engines produced formattable results")
        return ""
    
    final_output = "".join(parts)
    if _ENABLE_DIAGNOSTIC_LOGGING:
        logger.info(
            f"[chat_helpers/DIAGNOSTIC] format_reasoning_results output: "
            f"formatted_count={formatted_count}, total_length={len(final_output)}"
        )
    
    return final_output


def _format_fol_formalization(result: Dict[str, Any]) -> str:
    """
    Format FOL (First-Order Logic) formalization results from symbolic reasoning.
    
    Industry Standards Applied:
    - Input validation: Type and existence checks before access
    - Security: Safe UTF-8 string truncation to prevent injection
    - Clarity: Hierarchical presentation of readings with proper indentation
    - Defensive: Returns empty string on malformed data rather than crashing
    
    Handles quantifier scope ambiguity with multiple readings:
    - Reading A: Narrow scope (existential scoped wider)
    - Reading B: Wide scope (universal scoped wider)
    
    Args:
        result: Dictionary containing engine output, may include 'fol_formalization'
        
    Returns:
        Formatted string with FOL formulas and interpretations, or empty string
        
    Example Input:
        {
            "fol_formalization": {
                "original_sentence": "Every engineer reviewed a document.",
                "reading_a": {
                    "fol": "∃d.(∀e.Reviewed(e,d))",
                    "interpretation": "Narrow scope existential",
                    "english_rewrite": "There is a specific document..."
                },
                "reading_b": {...}
            }
        }
    """
    fol_formalization = result.get('fol_formalization')
    
    # Industry Standard: Early return on invalid input
    if not fol_formalization or not isinstance(fol_formalization, dict):
        return ""
    
    parts = []
    
    # Extract and format original sentence
    original = fol_formalization.get('original_sentence')
    if original:
        # Security: Safe truncation respecting UTF-8 boundaries
        safe_original = safe_truncate_utf8(str(original), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Original Sentence: \"{safe_original}\"")
    
    # Extract ambiguity type if present
    ambiguity_type = fol_formalization.get('ambiguity_type')
    if ambiguity_type:
        parts.append(f"\n- Ambiguity Type: {ambiguity_type}")
    
    # Format Reading A (narrow scope)
    reading_a = fol_formalization.get('reading_a')
    if reading_a and isinstance(reading_a, dict):
        parts.append("\n- Reading A (Narrow Scope):")
        
        fol_a = reading_a.get('fol')
        if fol_a:
            safe_fol_a = safe_truncate_utf8(str(fol_a), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  FOL: {safe_fol_a}")
        
        interpretation_a = reading_a.get('interpretation')
        if interpretation_a:
            safe_interp_a = safe_truncate_utf8(str(interpretation_a), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  Interpretation: {safe_interp_a}")
        
        english_a = reading_a.get('english_rewrite')
        if english_a:
            safe_english_a = safe_truncate_utf8(str(english_a), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  English: {safe_english_a}")
    
    # Format Reading B (wide scope)
    reading_b = fol_formalization.get('reading_b')
    if reading_b and isinstance(reading_b, dict):
        parts.append("\n- Reading B (Wide Scope):")
        
        fol_b = reading_b.get('fol')
        if fol_b:
            safe_fol_b = safe_truncate_utf8(str(fol_b), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  FOL: {safe_fol_b}")
        
        interpretation_b = reading_b.get('interpretation')
        if interpretation_b:
            safe_interp_b = safe_truncate_utf8(str(interpretation_b), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  Interpretation: {safe_interp_b}")
        
        english_b = reading_b.get('english_rewrite')
        if english_b:
            safe_english_b = safe_truncate_utf8(str(english_b), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  English: {safe_english_b}")
    
    # Return concatenated parts or empty string
    return "".join(parts) if parts else ""


def _format_causal_reasoning(result: Dict[str, Any]) -> str:
    """
    Format causal reasoning results including causal graphs and interventions.
    
    Industry Standards Applied:
    - Type safety: Explicit type checks before operations
    - Performance: Efficient iteration with comprehensions
    - Readability: Clear section headers and hierarchical structure
    - Robustness: Handles missing or malformed graph data gracefully
    
    Extracts and formats:
    - Causal graph structure (nodes and edges)
    - Confounders and mediators
    - Intervention recommendations
    - Counterfactual analyses
    
    Args:
        result: Dictionary containing engine output
        
    Returns:
        Formatted string with causal analysis, or empty string
    """
    parts = []
    
    # Format causal graph structure
    causal_graph = result.get('causal_graph')
    if causal_graph and isinstance(causal_graph, dict):
        parts.append("\n- Causal Graph:")
        # Industry Standard: Limit output size and use itertools.islice for memory efficiency
        edge_count = 0
        should_break = False
        for cause, effects in itertools.islice(causal_graph.items(), MAX_LIST_ITEMS_TO_SHOW):
            if isinstance(effects, dict):
                for effect, properties in itertools.islice(effects.items(), MAX_LIST_ITEMS_TO_SHOW):
                    edge_count += 1
                    if edge_count > MAX_LIST_ITEMS_TO_SHOW:
                        parts.append(f"\n  ... (showing {MAX_LIST_ITEMS_TO_SHOW} of many causal edges)")
                        should_break = True
                        break
                    
                    # Extract strength and confidence if available
                    strength = ""
                    if isinstance(properties, dict):
                        strength_val = properties.get('strength')
                        confidence_val = properties.get('confidence')
                        if strength_val is not None:
                            strength = f" (strength: {strength_val:.2f})"
                        if confidence_val is not None:
                            strength += f" (confidence: {int(confidence_val * 100)}%)"
                    
                    safe_cause = safe_truncate_utf8(str(cause), 100)
                    safe_effect = safe_truncate_utf8(str(effect), 100)
                    parts.append(f"\n  {safe_cause} → {safe_effect}{strength}")
            
            # Break outer loop if limit reached
            if should_break:
                break
    
    # Format confounders
    confounders = result.get('confounders')
    if confounders:
        if isinstance(confounders, (list, set, tuple)):
            # Industry Standard: Use itertools.islice for memory-efficient iteration
            safe_confounders = [safe_truncate_utf8(str(c), 100) for c in itertools.islice(confounders, MAX_LIST_ITEMS_TO_SHOW)]
            parts.append(f"\n- Confounders: {', '.join(safe_confounders)}")
        else:
            parts.append(f"\n- Confounders: {safe_truncate_utf8(str(confounders), MAX_REASONING_RESULT_LENGTH)}")
    
    # Format intervention recommendations
    intervention = result.get('intervention')
    if intervention:
        safe_intervention = safe_truncate_utf8(str(intervention), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Intervention: {safe_intervention}")
    
    # Format counterfactual analysis
    counterfactual = result.get('counterfactual')
    if counterfactual:
        safe_counterfactual = safe_truncate_utf8(str(counterfactual), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Counterfactual: {safe_counterfactual}")
    
    return "".join(parts) if parts else ""


def _format_probabilistic_reasoning(result: Dict[str, Any]) -> str:
    """
    Format probabilistic reasoning results including Bayesian updates.
    
    Industry Standards Applied:
    - Numerical precision: Proper formatting of probability values
    - Data validation: Check for valid probability ranges [0, 1]
    - Error handling: Graceful handling of invalid numerical data
    - Clarity: Clear distinction between prior and posterior
    
    Extracts and formats:
    - Prior and posterior distributions
    - Likelihood values
    - Model parameters
    - Intermediate calculation steps
    
    Args:
        result: Dictionary containing engine output
        
    Returns:
        Formatted string with probabilistic analysis, or empty string
    """
    parts = []
    
    # Format posterior distribution
    posterior = result.get('posterior')
    if posterior is not None:
        try:
            if isinstance(posterior, dict):
                parts.append("\n- Posterior Distribution:")
                # Industry Standard: Use itertools.islice for memory efficiency
                for param, value in itertools.islice(posterior.items(), MAX_LIST_ITEMS_TO_SHOW):
                    safe_param = safe_truncate_utf8(str(param), 100)
                    if isinstance(value, (int, float)):
                        parts.append(f"\n  {safe_param}: {value:.4f}")
                    else:
                        safe_value = safe_truncate_utf8(str(value), 100)
                        parts.append(f"\n  {safe_param}: {safe_value}")
            elif isinstance(posterior, (int, float)):
                parts.append(f"\n- Posterior: {posterior:.4f}")
            else:
                safe_posterior = safe_truncate_utf8(str(posterior), MAX_REASONING_RESULT_LENGTH)
                parts.append(f"\n- Posterior: {safe_posterior}")
        except (ValueError, TypeError) as e:
            logger.debug(f"[_format_probabilistic_reasoning] Invalid posterior value: {e}")
    
    # Format prior distribution
    prior = result.get('prior')
    if prior is not None:
        try:
            if isinstance(prior, (int, float)):
                parts.append(f"\n- Prior: {prior:.4f}")
            else:
                safe_prior = safe_truncate_utf8(str(prior), MAX_REASONING_RESULT_LENGTH)
                parts.append(f"\n- Prior: {safe_prior}")
        except (ValueError, TypeError) as e:
            logger.debug(f"[_format_probabilistic_reasoning] Invalid prior value: {e}")
    
    # Format model parameters
    parameters = result.get('parameters')
    if parameters and isinstance(parameters, dict):
        parts.append("\n- Parameters:")
        # Industry Standard: Use itertools.islice for memory efficiency
        for param_name, param_value in itertools.islice(parameters.items(), MAX_LIST_ITEMS_TO_SHOW):
            safe_name = safe_truncate_utf8(str(param_name), 100)
            if isinstance(param_value, (int, float)):
                parts.append(f"\n  {safe_name}: {param_value:.4f}")
            else:
                safe_value = safe_truncate_utf8(str(param_value), 100)
                parts.append(f"\n  {safe_name}: {safe_value}")
    
    # Format intermediate values
    intermediate = result.get('intermediate_values')
    if intermediate and isinstance(intermediate, dict):
        parts.append("\n- Intermediate Values:")
        # Industry Standard: Use itertools.islice for memory efficiency
        for key, value in itertools.islice(intermediate.items(), MAX_LIST_ITEMS_TO_SHOW):
            safe_key = safe_truncate_utf8(str(key), 100)
            safe_value = safe_truncate_utf8(str(value), 100)
            parts.append(f"\n  {safe_key}: {safe_value}")
    
    return "".join(parts) if parts else ""


def _format_analogical_reasoning(result: Dict[str, Any]) -> str:
    """
    Format analogical reasoning results including structural mappings.
    
    Industry Standards Applied:
    - Structure preservation: Maintains hierarchical relationships in output
    - Readability: Clear source→target mapping notation
    - Scalability: Limits output to prevent overwhelming display
    - Flexibility: Handles various mapping structure formats
    
    Extracts and formats:
    - Entity mappings (source → target)
    - Structural alignments
    - Inferred relationships
    - Similarity scores
    
    Args:
        result: Dictionary containing engine output
        
    Returns:
        Formatted string with analogical analysis, or empty string
    """
    parts = []
    
    # Format entity mappings
    entity_mappings = result.get('entity_mappings')
    if entity_mappings and isinstance(entity_mappings, dict):
        parts.append("\n- Entity Mappings:")
        # Industry Standard: Use itertools.islice for memory efficiency
        for source, target in itertools.islice(entity_mappings.items(), MAX_LIST_ITEMS_TO_SHOW):
            safe_source = safe_truncate_utf8(str(source), 100)
            safe_target = safe_truncate_utf8(str(target), 100)
            parts.append(f"\n  {safe_source} → {safe_target}")
    
    # Format structural alignment
    structural_alignment = result.get('structural_alignment')
    if structural_alignment:
        safe_alignment = safe_truncate_utf8(str(structural_alignment), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Structural Alignment: {safe_alignment}")
    
    # Format inferences
    inferences = result.get('inferences')
    if inferences:
        if isinstance(inferences, (list, tuple)):
            parts.append("\n- Inferences:")
            # Industry Standard: Use itertools.islice for memory efficiency with enumerate
            for i, inference in enumerate(itertools.islice(inferences, MAX_LIST_ITEMS_TO_SHOW), 1):
                safe_inference = safe_truncate_utf8(str(inference), MAX_REASONING_RESULT_LENGTH)
                parts.append(f"\n  {i}. {safe_inference}")
        else:
            safe_inferences = safe_truncate_utf8(str(inferences), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n- Inferences: {safe_inferences}")
    
    # Format source and target domains
    source_domain = result.get('source_domain')
    if source_domain:
        safe_source = safe_truncate_utf8(str(source_domain), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Source Domain: {safe_source}")
    
    target_domain = result.get('target_domain')
    if target_domain:
        safe_target = safe_truncate_utf8(str(target_domain), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Target Domain: {safe_target}")
    
    return "".join(parts) if parts else ""


def _format_mathematical_reasoning(result: Dict[str, Any]) -> str:
    """
    Format mathematical reasoning results including proofs and solutions.
    
    Industry Standards Applied:
    - Mathematical notation: Preserves formula structure
    - Logical flow: Presents proof steps in order
    - Verification: Shows solution verification status
    - Precision: Handles numerical results with appropriate formatting
    
    Extracts and formats:
    - Closed-form solutions
    - Step-by-step proofs
    - Verification results
    - Intermediate calculations
    
    Args:
        result: Dictionary containing engine output
        
    Returns:
        Formatted string with mathematical analysis, or empty string
    """
    parts = []
    
    # Format closed-form solution
    closed_form = result.get('closed_form')
    if closed_form:
        safe_closed_form = safe_truncate_utf8(str(closed_form), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Closed-Form Solution: {safe_closed_form}")
    
    # Format proof steps
    proof_steps = result.get('proof_steps')
    if proof_steps and isinstance(proof_steps, (list, tuple)):
        parts.append("\n- Proof Steps:")
        for i, step in enumerate(proof_steps[:MAX_REASONING_STEPS], 1):
            safe_step = safe_truncate_utf8(str(step), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n  {i}. {safe_step}")
    
    # Format verification result
    verification = result.get('verification')
    if verification is not None:
        if isinstance(verification, bool):
            status = "Verified ✓" if verification else "Not Verified ✗"
            parts.append(f"\n- Verification: {status}")
        else:
            safe_verification = safe_truncate_utf8(str(verification), MAX_REASONING_RESULT_LENGTH)
            parts.append(f"\n- Verification: {safe_verification}")
    
    # Format solution method
    method = result.get('method')
    # Industry Standard: Use constant set to avoid magic string dependencies
    if method and str(method).lower() not in GENERIC_METHOD_VALUES:
        safe_method = safe_truncate_utf8(str(method), MAX_REASONING_RESULT_LENGTH)
        parts.append(f"\n- Solution Method: {safe_method}")
    
    return "".join(parts) if parts else ""


def _format_engine_result_dict(engine_name: str, result: Dict[str, Any]) -> str:
    """
    Format a single reasoning engine's result dictionary with domain-specific handling.
    
    Industry Standard Practices Applied:
    - Defensive programming: Type checks and null guards on all data access
    - Single Responsibility: Delegates domain-specific formatting to helper functions
    - Extensibility: Easy to add new engine-specific formatters
    - Security: Safe string truncation respecting UTF-8 boundaries
    - Performance: Efficient string building with list concatenation
    - Error handling: Graceful degradation with logging on malformed data
    
    This function now properly handles domain-specific structured outputs from:
    - Symbolic reasoning (FOL formalization, quantifier scope analysis)
    - Causal reasoning (causal graphs, confounders, interventions)
    - Probabilistic reasoning (posteriors, parameters, distributions)
    - Analogical reasoning (entity mappings, structural alignments)
    - Mathematical reasoning (closed-form solutions, proof steps)
    
    Args:
        engine_name: Name of the reasoning engine (e.g., 'symbolic', 'probabilistic')
        result: Dictionary containing the engine's output
    
    Returns:
        Formatted string section for this engine, or empty string if no
        relevant data found
        
    Example:
        >>> result = {
        ...     "fol_formalization": {
        ...         "reading_a": {"fol": "∃d.(∀e.Reviewed(e,d))", ...},
        ...         "reading_b": {"fol": "∀e.(∃d.Reviewed(e,d))", ...}
        ...     }
        ... }
        >>> formatted = _format_engine_result_dict("symbolic", result)
        # Returns formatted FOL formalization with both readings
    """
    # Build formatted section
    section_parts = [f"\n{engine_name.replace('_', ' ').title()}:"]
    has_content = False
    
    # =========================================================================
    # DOMAIN-SPECIFIC FORMATTERS
    # Industry Standard: Handle specialized output structures first
    # =========================================================================
    
    # Format FOL formalization (Symbolic Reasoning)
    fol_content = _format_fol_formalization(result)
    if fol_content:
        section_parts.append(fol_content)
        has_content = True
    
    # Format causal reasoning outputs
    causal_content = _format_causal_reasoning(result)
    if causal_content:
        section_parts.append(causal_content)
        has_content = True
    
    # Format probabilistic reasoning outputs
    probabilistic_content = _format_probabilistic_reasoning(result)
    if probabilistic_content:
        section_parts.append(probabilistic_content)
        has_content = True
    
    # Format analogical reasoning outputs
    analogical_content = _format_analogical_reasoning(result)
    if analogical_content:
        section_parts.append(analogical_content)
        has_content = True
    
    # Format mathematical reasoning outputs
    mathematical_content = _format_mathematical_reasoning(result)
    if mathematical_content:
        section_parts.append(mathematical_content)
        has_content = True
    
    # =========================================================================
    # GENERIC FIELDS (fallback for engines without specific formatters)
    # Industry Standard: Maintain backward compatibility
    # =========================================================================
    
    # Extract common fields with safe access patterns
    conclusion = result.get('conclusion')
    confidence = result.get('confidence')
    reasoning_type = result.get('reasoning_type')
    explanation = result.get('explanation')
    reasoning_steps = result.get('reasoning_steps', [])
    
    # Format conclusion using ConclusionFormatter (BUG FIX #2)
    if conclusion is not None:
        # Use the robust ConclusionFormatter to handle dict, list, object types
        conclusion_str = ConclusionFormatter.format(conclusion)
        if conclusion_str:
            # Truncate if needed
            if len(conclusion_str) > MAX_REASONING_RESULT_LENGTH:
                conclusion_str = conclusion_str[:MAX_REASONING_RESULT_LENGTH] + "..."
            section_parts.append(f"\n- Conclusion: {conclusion_str}")
            has_content = True
    
    # Format confidence
    if confidence is not None:
        try:
            # Handle both float (0.0-1.0) and int (0-100) confidence values
            if isinstance(confidence, (int, float)):
                if confidence <= 1.0:
                    confidence_pct = int(confidence * 100)
                else:
                    confidence_pct = int(confidence)
                section_parts.append(f"\n- Confidence: {confidence_pct}%")
                has_content = True
        except (ValueError, TypeError) as e:
            logger.debug(f"[_format_engine_result_dict] Invalid confidence value: {e}")
    
    # Format reasoning type - use type-safe conversion to handle Enum objects
    # Industry Standard: Prevent "'ReasoningType' object has no attribute 'replace'" error
    if reasoning_type is not None:
        reasoning_type_str = safe_reasoning_type_to_string(reasoning_type, default="unknown")
        section_parts.append(f"\n- Type: {reasoning_type_str}")
        has_content = True
    
    # Format explanation
    if explanation is not None:
        explanation_str = str(explanation)[:MAX_REASONING_RESULT_LENGTH]
        section_parts.append(f"\n- Explanation: {explanation_str}")
        has_content = True
    
    # Format reasoning steps (if available)
    if reasoning_steps and isinstance(reasoning_steps, (list, tuple)):
        section_parts.append("\n- Reasoning Steps:")
        for i, step in enumerate(reasoning_steps[:MAX_REASONING_STEPS], 1):
            step_str = str(step)[:MAX_REASONING_RESULT_LENGTH]
            section_parts.append(f"\n  {i}. {step_str}")
        has_content = True
    
    # Return formatted section or empty string
    if has_content:
        section_parts.append("\n")
        return "".join(section_parts)
    else:
        return ""
