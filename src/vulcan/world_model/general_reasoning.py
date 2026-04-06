"""
General reasoning functions extracted from WorldModel.

Handles general reasoning queries via introspection.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _general_reasoning(wm, query: str, **kwargs) -> Dict[str, Any]:
    """Handle general reasoning queries via introspection."""
    # Use existing introspection method for general queries
    introspection_result = wm.introspect(query)

    # Convert to standard reason() output format
    return {
        'response': introspection_result.get('response', ''),
        'confidence': introspection_result.get('confidence', 0.7),
        'reasoning_trace': {
            'aspect': introspection_result.get('aspect', 'general'),
            'reasoning': introspection_result.get('reasoning', '')
        },
        'mode': 'general',
        # Pass through delegation info if present
        'needs_delegation': introspection_result.get('needs_delegation', False),
        'recommended_tool': introspection_result.get('recommended_tool'),
        'delegation_reason': introspection_result.get('delegation_reason')
    }
