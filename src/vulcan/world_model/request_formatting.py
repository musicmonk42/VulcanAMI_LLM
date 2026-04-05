"""
request_formatting.py - Extracted formatting/engine functions from WorldModel.

Contains reasoning engine invocation, ethical analysis, LLM formatting,
and synthesis format determination.
Phase 1 of WorldModel decomposition.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _invoke_reasoning_engine(wm, query: str, engine: str) -> Dict[str, Any]:
    """Invoke appropriate reasoning engine. Lazy loading with graceful fallback."""
    try:
        if engine == 'symbolic':
            from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
            reasoner = SymbolicReasoner()
            result = reasoner.query(query, timeout=10)
            return result

        elif engine == 'probabilistic':
            from vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
            reasoner = ProbabilisticReasoner()
            result = reasoner.predict_with_uncertainty(query)
            return result

        elif engine == 'causal':
            from vulcan.reasoning.causal_reasoning import CausalReasoner
            reasoner = CausalReasoner()
            result = reasoner.analyze(query)
            return result

        elif engine == 'mathematical':
            from vulcan.reasoning.mathematical_computation import MathematicalReasoner
            reasoner = MathematicalReasoner()
            result = reasoner.compute(query)
            return result

        else:
            logger.warning(f"[WorldModel] Unknown engine '{engine}', using unified reasoner")
            from vulcan.reasoning.unified import UnifiedReasoner
            reasoner = UnifiedReasoner()
            result = reasoner.reason(query)
            return result

    except Exception as e:
        logger.error(f"[WorldModel] Reasoning engine '{engine}' failed: {e}")
        return {
            'conclusion': f"Reasoning failed: {str(e)}",
            'confidence': 0.0,
            'status': 'error',
            'error': str(e),
        }


def _run_ethical_analysis(wm, query: str) -> Dict[str, Any]:
    """Run ethical analysis using meta-reasoning components."""
    try:
        if hasattr(wm, 'ethical_boundary_monitor') and wm.ethical_boundary_monitor:
            analysis = wm.ethical_boundary_monitor.analyze_query(query)
            return analysis

        if hasattr(wm, 'motivational_introspection') and wm.motivational_introspection:
            analysis = wm.motivational_introspection.analyze_query(query)
            return analysis

        return {
            'analysis': 'Ethical analysis not available',
            'confidence': 0.50,
        }
    except Exception as e:
        logger.error(f"[WorldModel] Ethical analysis failed: {e}")
        return {
            'analysis': f"Analysis failed: {str(e)}",
            'confidence': 0.0,
        }


def _format_with_llm(wm, guidance) -> str:
    """Use LLM to format content according to guidance."""
    try:
        prompt = _build_formatting_prompt(wm, guidance)

        from vulcan.llm.hybrid_executor import get_hybrid_executor

        executor = get_hybrid_executor()
        if executor:
            result = executor.execute_sync(
                prompt=prompt,
                max_tokens=guidance.max_length or 1024,
            )
            return result.get('text', '')
    except Exception as e:
        logger.error(f"[WorldModel] LLM formatting failed: {e}")

    return _fallback_format(wm, guidance)


def _build_formatting_prompt(wm, guidance) -> str:
    """Build the prompt for LLM formatting."""
    prompt_parts = [
        f"TASK: {guidance.task}",
        "",
        "CONTENT TO FORMAT:",
        json.dumps(guidance.verified_content, indent=2, default=str),
        "",
        "STRUCTURE:",
        json.dumps(guidance.structure, indent=2),
        "",
        "CONSTRAINTS (you MUST follow these):",
    ]
    for constraint in guidance.constraints:
        prompt_parts.append(f"- {constraint}")

    prompt_parts.append("")
    prompt_parts.append("PERMISSIONS (you MAY do these):")
    for permission in guidance.permissions:
        prompt_parts.append(f"- {permission}")

    prompt_parts.append("")
    prompt_parts.append(f"TONE: {guidance.tone}")
    prompt_parts.append(f"FORMAT: {guidance.format}")

    return "\n".join(prompt_parts)


def _fallback_format(wm, guidance) -> str:
    """Fallback formatting when LLM is unavailable."""
    content = guidance.verified_content

    if 'conclusion' in content:
        return f"Result: {content['conclusion']}"
    elif 'facts' in content:
        return "\n".join([f"- {fact}" for fact in content['facts'][:10]])
    else:
        return str(content)


def _determine_synthesis_format(wm, query: str) -> str:
    """Determine synthesis format from query."""
    query_lower = query.lower()

    if 'paper' in query_lower or 'essay' in query_lower:
        return 'paper'
    elif 'summary' in query_lower or 'summarize' in query_lower:
        return 'summary'
    else:
        return 'explanation'
