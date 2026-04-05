"""
request_handling.py - Extracted request handling functions from WorldModel.

Phase 1 of WorldModel decomposition: functions take `wm` (WorldModel instance)
as first parameter instead of `self`.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def process_request(wm, query: str, **kwargs) -> Dict[str, Any]:
    """Main entry point - World Model coordinates ALL requests."""
    try:
        classification = wm.request_classifier.classify(query, context=kwargs.get('context'))

        logger.info(
            f"[WorldModel.process_request] Request classified: "
            f"type={classification.request_type.value}, domain={classification.domain}, "
            f"confidence={classification.confidence:.2f}"
        )

        from vulcan.vulcan_types import RequestType

        if classification.request_type == RequestType.REASONING:
            return _handle_reasoning_request(wm, query, classification, **kwargs)
        elif classification.request_type == RequestType.KNOWLEDGE_SYNTHESIS:
            return _handle_knowledge_request(wm, query, classification, **kwargs)
        elif classification.request_type == RequestType.CREATIVE:
            return _handle_creative_request(wm, query, classification, **kwargs)
        elif classification.request_type == RequestType.ETHICAL:
            return _handle_ethical_request(wm, query, classification, **kwargs)
        else:
            from .request_dispatch import _handle_conversational_request
            return _handle_conversational_request(wm, query, classification, **kwargs)

    except Exception as e:
        logger.error(f"[WorldModel.process_request] Error processing request: {e}", exc_info=True)
        return {
            'response': f"I encountered an error processing your request: {str(e)}",
            'confidence': 0.0,
            'source': 'error',
            'metadata': {'error': str(e), 'error_type': type(e).__name__},
        }


def _handle_reasoning_request(wm, query: str, classification, **kwargs) -> Dict[str, Any]:
    """Handle requests that require reasoning."""
    try:
        engine = classification.reasoning_engine
        logger.info(f"[WorldModel] Routing to reasoning engine: {engine}")

        from .request_formatting import _invoke_reasoning_engine, _format_with_llm

        reasoning_result = _invoke_reasoning_engine(wm, query, engine)

        guidance = wm.llm_guidance_builder.build_for_reasoning(reasoning_result, query)
        formatted_response = _format_with_llm(wm, guidance)

        return {
            'response': formatted_response,
            'confidence': reasoning_result.get('confidence', 0.0),
            'source': 'reasoning_engine',
            'engine': engine,
            'reasoning_result': reasoning_result,
            'metadata': {
                'classification': classification.to_dict(),
                'reasoning_type': engine,
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Reasoning request failed: {e}", exc_info=True)
        return {
            'response': f"Reasoning failed: {str(e)}",
            'confidence': 0.0,
            'source': 'error',
            'metadata': {'error': str(e)},
        }


def _handle_knowledge_request(wm, query: str, classification, **kwargs) -> Dict[str, Any]:
    """Handle requests that require knowledge retrieval and synthesis."""
    try:
        knowledge = wm.knowledge_handler.retrieve_knowledge(
            domain=classification.domain,
            topic=classification.subdomain or classification.domain,
            query=query,
        )
        verified = wm.knowledge_handler.verify_knowledge(knowledge)

        logger.info(
            f"[WorldModel] Knowledge retrieved: {len(verified.verified_facts)} facts, "
            f"confidence={verified.confidence:.2f}"
        )

        from .request_formatting import _determine_synthesis_format, _format_with_llm

        format_type = _determine_synthesis_format(wm, query)

        guidance = wm.llm_guidance_builder.build_for_knowledge_synthesis(
            verified, classification.subdomain or classification.domain, format_type
        )
        formatted_response = _format_with_llm(wm, guidance)

        return {
            'response': formatted_response,
            'confidence': verified.confidence,
            'source': 'knowledge_retrieval',
            'verified_facts': len(verified.verified_facts),
            'sources': knowledge.sources,
            'metadata': {
                'classification': classification.to_dict(),
                'retrieval': knowledge.metadata,
                'verification': verified.metadata,
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Knowledge request failed: {e}", exc_info=True)
        return {
            'response': f"Knowledge retrieval failed: {str(e)}",
            'confidence': 0.0,
            'source': 'error',
            'metadata': {'error': str(e)},
        }


def _handle_creative_request(wm, query: str, classification, **kwargs) -> Dict[str, Any]:
    """Handle creative requests with knowledge grounding."""
    try:
        creative_guidance = wm.creative_handler.prepare_creative_guidance(
            format_type=classification.creative_format,
            subject=classification.subdomain,
            domain=classification.domain,
            query=query,
        )

        logger.info(
            f"[WorldModel] Creative guidance prepared: format={creative_guidance.format}, "
            f"subject={creative_guidance.subject_knowledge.topic}"
        )

        from .request_formatting import _format_with_llm

        guidance = wm.llm_guidance_builder.build_for_creative(creative_guidance, query)
        creative_output = _format_with_llm(wm, guidance)

        verification = wm.creative_handler.verify_creative_output(
            creative_output, creative_guidance
        )

        if not verification['passed']:
            logger.warning(
                f"[WorldModel] Creative output failed verification: "
                f"{verification['violations']}"
            )

        return {
            'response': creative_output,
            'confidence': 0.85 if verification['passed'] else 0.60,
            'source': 'creative_handler',
            'format': classification.creative_format,
            'grounded_in': creative_guidance.subject_knowledge.facts[:3],
            'verification': verification,
            'metadata': {
                'classification': classification.to_dict(),
                'knowledge': creative_guidance.subject_knowledge.metadata,
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Creative request failed: {e}", exc_info=True)
        return {
            'response': f"Creative generation failed: {str(e)}",
            'confidence': 0.0,
            'source': 'error',
            'metadata': {'error': str(e)},
        }


def _handle_ethical_request(wm, query: str, classification, **kwargs) -> Dict[str, Any]:
    """Handle ethical/philosophical requests using meta-reasoning."""
    try:
        from .request_formatting import _run_ethical_analysis, _format_with_llm

        ethical_analysis = _run_ethical_analysis(wm, query)

        guidance = wm.llm_guidance_builder.build_for_ethical(ethical_analysis, query)
        formatted_response = _format_with_llm(wm, guidance)

        return {
            'response': formatted_response,
            'confidence': ethical_analysis.get('confidence', 0.70),
            'source': 'meta_reasoning',
            'analysis': ethical_analysis,
            'metadata': {
                'classification': classification.to_dict() if hasattr(classification, 'to_dict') else {},
            },
        }
    except Exception as e:
        logger.error(f"[WorldModel] Ethical request failed: {e}", exc_info=True)
        return {
            'response': f"Ethical analysis failed: {str(e)}",
            'confidence': 0.0,
            'source': 'error',
            'metadata': {'error': str(e)},
        }


def _handle_self_referential_request(
    wm, query: str, context: Optional[Dict[str, Any]] = None, **kwargs
) -> Dict[str, Any]:
    """Handle self-referential queries. Delegates to request_dispatch."""
    from .request_dispatch import _handle_self_referential_request as _impl
    return _impl(wm, query, context, **kwargs)


def _synthesize_self_response(
    wm, query: str, introspection: Any, motivation: Any, context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Delegates to request_dispatch."""
    from .request_dispatch import _synthesize_self_response as _impl
    return _impl(wm, query, introspection, motivation, context)
