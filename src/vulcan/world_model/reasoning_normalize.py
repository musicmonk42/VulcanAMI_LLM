"""
reasoning_normalize.py - Normalize reasoning results to standard format.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def normalize_reasoning_result(wm, result: Any) -> Dict[str, Any]:
    """
    Normalize UnifiedReasoner result to World Model's standard format.

    Args:
        wm: WorldModel instance
        result: Result from UnifiedReasoner (may be ReasoningResult or dict)

    Returns:
        Dict with World Model's standard keys: response, confidence, reasoning_trace, etc.
    """
    # Handle ReasoningResult object
    if hasattr(result, 'content'):
        return {
            'response': result.content,
            'confidence': getattr(result, 'confidence', 0.8),
            'reasoning_trace': getattr(result, 'metadata', {}),
            'mode': 'delegated',
            'engine_used': getattr(result, 'selected_tools', ['unified'])[0] if hasattr(result, 'selected_tools') else 'unified',
            'tool_selector_decision': True,  # Indicates ToolSelector made the decision
        }

    # Handle dict result
    elif isinstance(result, dict):
        return {
            'response': result.get('content', result.get('response', str(result))),
            'confidence': result.get('confidence', 0.8),
            'reasoning_trace': result.get('metadata', result.get('reasoning_trace', {})),
            'mode': 'delegated',
            'engine_used': result.get('selected_tools', ['unified'])[0] if 'selected_tools' in result else 'unified',
            'tool_selector_decision': True,
        }

    # Fallback for unknown result types
    else:
        return {
            'response': str(result),
            'confidence': 0.7,
            'reasoning_trace': {},
            'mode': 'delegated',
            'engine_used': 'unified',
            'tool_selector_decision': True,
        }


def normalize_engine_result(
    wm, result: Any, engine_used: str, query: str
) -> Dict[str, Any]:
    """
    Normalize reasoning engine results to WorldModel standard format.

    Handles diverse result formats from different engines with
    defensive programming and sensible defaults.

    Args:
        wm: WorldModel instance
        result: The result from the reasoning engine (format varies)
        engine_used: Name of the engine that produced the result
        query: Original query (for context in trace)

    Returns:
        Dict[str, Any]: Normalized result in WorldModel format
    """
    try:
        # If result is already in correct format, validate and return
        if isinstance(result, dict) and 'response' in result:
            return {
                'response': str(result.get('response', '')),
                'confidence': float(result.get('confidence', 0.7)),
                'reasoning_trace': result.get('reasoning_trace', {}),
                'mode': result.get('mode', engine_used),
                'engine_used': engine_used
            }

        # Handle string results (direct answers)
        if isinstance(result, str):
            return {
                'response': result,
                'confidence': 0.75,
                'reasoning_trace': {
                    'engine': engine_used,
                    'query': query,
                    'result_type': 'string'
                },
                'mode': engine_used,
                'engine_used': engine_used
            }

        # Handle complex result objects (extract relevant fields)
        response_text = str(result)
        if hasattr(result, 'result'):
            response_text = str(result.result)
        elif hasattr(result, 'answer'):
            response_text = str(result.answer)
        elif hasattr(result, 'output'):
            response_text = str(result.output)

        confidence = 0.70  # Default confidence
        if hasattr(result, 'confidence'):
            confidence = float(result.confidence)
        elif hasattr(result, 'certainty'):
            confidence = float(result.certainty)

        reasoning_trace = {'engine': engine_used, 'query': query}
        if hasattr(result, 'trace'):
            reasoning_trace.update(result.trace)
        elif hasattr(result, 'steps'):
            reasoning_trace['steps'] = result.steps

        return {
            'response': response_text,
            'confidence': confidence,
            'reasoning_trace': reasoning_trace,
            'mode': engine_used,
            'engine_used': engine_used
        }

    except Exception as e:
        logger.error(f"[WorldModel] Result normalization failed: {e}")
        # Return minimal valid result on error
        return {
            'response': f"Reasoning engine {engine_used} completed but result normalization failed: {e}",
            'confidence': 0.5,
            'reasoning_trace': {
                'engine': engine_used,
                'error': str(e),
                'raw_result': str(result)[:500]  # Truncate to prevent overflow
            },
            'mode': engine_used,
            'engine_used': engine_used
        }
