"""
request_dispatch.py - Overflow from request_handling.py for 250-line limit.

Contains self-referential, introspection, and conversational request handlers.
Phase 1 of WorldModel decomposition.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _handle_self_referential_request(
    wm, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
) -> Dict[str, Any]:
    """Handle self-referential queries - 'What are you?', 'What is your purpose?'"""
    logger.info(f"[WorldModel] Handling self-referential request")
    try:
        introspection = None
        motivation = None

        if hasattr(wm, 'motivational_introspection') and wm.motivational_introspection:
            try:
                introspection = wm.motivational_introspection.introspect_current_objective()
                motivation = wm.motivational_introspection.explain_motivation_structure()
            except Exception as e:
                logger.warning(f"[WorldModel] Motivational introspection failed: {e}")

        response = _synthesize_self_response(wm, query, introspection, motivation, context)

        return {
            'response': response.get('response', 'I am VULCAN, an AI reasoning system.'),
            'confidence': response.get('confidence', 0.75),
            'source': 'meta_reasoning',
            'category': 'self_referential',
            'metadata': {
                'introspection': introspection if introspection else {},
                'motivation': motivation if motivation else {},
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Self-referential request failed: {e}", exc_info=True)
        return {
            'response': "I am VULCAN, an AI reasoning system designed to help with complex reasoning tasks.",
            'confidence': 0.60,
            'source': 'fallback',
            'metadata': {'error': str(e)},
        }


def _handle_introspection_request(
    wm, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
) -> Dict[str, Any]:
    """Handle introspection queries - 'How did you decide?', 'Why did you choose X?'"""
    logger.info(f"[WorldModel] Handling introspection request")
    try:
        explanation = None

        if hasattr(wm, 'transparency_interface') and wm.transparency_interface:
            try:
                explanation = wm.transparency_interface.explain_decision(
                    decision=context.get('last_decision') if context else None,
                    factors=context.get('decision_factors') if context else None,
                    reasoning_steps=context.get('reasoning_steps') if context else None,
                )
            except Exception as e:
                logger.warning(f"[WorldModel] Transparency interface failed: {e}")

        response = _synthesize_introspection_response(wm, query, explanation, context)

        return {
            'response': response.get('response', 'I can explain my reasoning process.'),
            'confidence': response.get('confidence', 0.70),
            'source': 'meta_reasoning',
            'category': 'introspection',
            'metadata': {
                'explanation': explanation if explanation else {},
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Introspection request failed: {e}", exc_info=True)
        return {
            'response': "I make decisions based on the available information and reasoning methods.",
            'confidence': 0.60,
            'source': 'fallback',
            'metadata': {'error': str(e)},
        }


def _synthesize_self_response(
    wm, query: str, introspection: Any, motivation: Any, context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Synthesize self-referential response from introspection data."""
    response = "I am VULCAN, an AI reasoning system designed to assist with complex reasoning, causal analysis, and decision support."
    confidence = 0.75

    if introspection and isinstance(introspection, dict):
        if 'purpose' in introspection:
            response += f" My purpose is: {introspection['purpose']}"
            confidence = 0.85

    return {'response': response, 'confidence': confidence}


def _synthesize_introspection_response(
    wm, query: str, explanation: Any, context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Synthesize introspection response from explanation data."""
    response = "I make decisions by analyzing the available information, considering multiple perspectives, and applying appropriate reasoning methods."
    confidence = 0.70

    if explanation and isinstance(explanation, dict):
        if 'reasoning_steps' in explanation:
            steps = explanation['reasoning_steps']
            if steps:
                response += f" The key steps were: {', '.join(str(s) for s in steps[:3])}"
                confidence = 0.80

    return {'response': response, 'confidence': confidence}


def _handle_conversational_request(
    wm, query: str, classification, **kwargs
) -> Dict[str, Any]:
    """Handle simple conversational requests."""
    try:
        from vulcan.world_model.llm_guidance import LLMGuidance
        from .request_formatting import _format_with_llm

        guidance = LLMGuidance(
            task="Respond conversationally to the user",
            verified_content={'query': query},
            structure={'format': 'conversational'},
            constraints=[
                "Keep response brief and friendly",
                "Do not make factual claims you cannot verify",
            ],
            permissions=[
                "You may be conversational and friendly",
                "You may ask clarifying questions",
            ],
            tone="friendly and helpful",
            format="conversation",
            max_length=200,
            metadata={'source': 'conversational'},
        )

        response = _format_with_llm(wm, guidance)

        return {
            'response': response,
            'confidence': 0.90,
            'source': 'conversational',
            'metadata': {
                'classification': classification.to_dict(),
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Conversational request failed: {e}", exc_info=True)
        return {
            'response': "Hello! How can I help you today?",
            'confidence': 0.50,
            'source': 'fallback',
            'metadata': {'error': str(e)},
        }
