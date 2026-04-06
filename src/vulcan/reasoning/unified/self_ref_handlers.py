"""
Self-referential query handling for unified reasoning orchestration.

Handles self-referential queries using world model meta-reasoning
infrastructure (ObjectiveHierarchy, GoalConflictDetector,
EthicalBoundaryMonitor, CounterfactualObjectiveReasoner,
TransparencyInterface).

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .config import SELF_REFERENTIAL_MIN_CONFIDENCE
from .types import ReasoningTask
from ..reasoning_types import (
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningType,
)

logger = logging.getLogger(__name__)


def handle_self_referential_query(
    reasoner: Any,
    task: ReasoningTask,
    reasoning_chain: ReasoningChain,
    _recursion_depth: int = 0,
) -> ReasoningResult:
    """
    Handle self-referential queries using world model meta-reasoning.

    Args:
        reasoner: UnifiedReasoner instance (for accessing helper methods).
        task: ReasoningTask containing the self-referential query.
        reasoning_chain: ReasoningChain to accumulate reasoning steps.
        _recursion_depth: Current recursion depth (internal).

    Returns:
        ReasoningResult with PHILOSOPHICAL type and substantive analysis.
    """
    logger.info(
        f"[SelfRef] Handling self-referential query via meta-reasoning "
        f"(recursion_depth={_recursion_depth})"
    )

    try:
        from vulcan.world_model.meta_reasoning import (
            ObjectiveHierarchy,
            GoalConflictDetector,
            EthicalBoundaryMonitor,
            CounterfactualObjectiveReasoner,
            TransparencyInterface,
        )

        hierarchy = ObjectiveHierarchy()
        conflict_detector = GoalConflictDetector(hierarchy)
        boundary_monitor = EthicalBoundaryMonitor()
        counterfactual = CounterfactualObjectiveReasoner(hierarchy)
        transparency = TransparencyInterface()

        query_str = reasoner._extract_query_string(task.query)
        if not query_str:
            query_str = (
                str(task.input_data) if task.input_data
                else "self-referential query"
            )

        analysis = {
            'query': query_str,
            'objectives': [],
            'conflicts': [],
            'ethical_check': None,
            'counterfactual': None,
            'transparency_explanation': None,
        }

        # Get relevant objectives
        try:
            relevant_objective_names = hierarchy.get_top_objectives(limit=5)
            analysis['objectives'] = []
            for name in relevant_objective_names:
                obj = hierarchy.objectives.get(name)
                if obj and hasattr(obj, 'name') and hasattr(obj, 'priority'):
                    analysis['objectives'].append(
                        {'name': obj.name, 'priority': obj.priority}
                    )
                else:
                    analysis['objectives'].append(
                        {'name': name, 'priority': 0}
                    )
        except Exception as e:
            logger.warning(f"[SelfRef] Failed to get objectives: {e}")

        # Check for goal conflicts
        try:
            conflicts = conflict_detector.detect_conflicts_in_query(query_str)
            analysis['conflicts'] = conflicts if conflicts else []
        except Exception as e:
            logger.warning(f"[SelfRef] Failed to detect conflicts: {e}")

        # Validate against ethical boundaries
        try:
            ethical_result = boundary_monitor.check_action(query_str)
            if isinstance(ethical_result, tuple):
                is_allowed, violation = ethical_result
                reason = (
                    getattr(violation, 'reason', 'Ethical boundary violated')
                    if violation is not None
                    else 'No ethical concerns'
                )
                analysis['ethical_check'] = {
                    'allowed': is_allowed,
                    'reason': reason,
                }
            elif isinstance(ethical_result, dict):
                analysis['ethical_check'] = {
                    'allowed': ethical_result.get('allowed', True),
                    'reason': ethical_result.get('reason', 'No ethical concerns'),
                }
            else:
                analysis['ethical_check'] = {
                    'allowed': True,
                    'reason': 'Check completed',
                }
        except Exception as e:
            logger.warning(f"[SelfRef] Failed ethical check: {e}")
            analysis['ethical_check'] = {
                'allowed': True,
                'reason': 'Check unavailable',
            }

        # Perform counterfactual analysis if applicable
        if 'if you were' in query_str.lower() or 'would you' in query_str.lower():
            try:
                counterfactual_result = counterfactual.analyze_scenario(query_str)
                analysis['counterfactual'] = counterfactual_result
            except Exception as e:
                logger.warning(
                    f"[SelfRef] Failed counterfactual analysis: {e}"
                )

        # Generate transparent explanation
        try:
            transparency_result = transparency.explain_decision(
                decision=query_str,
                factors=analysis,
                reasoning_steps=[
                    'meta-reasoning analysis',
                    'objective alignment',
                    'ethical validation',
                ],
            )
            analysis['transparency_explanation'] = transparency_result
        except Exception as e:
            logger.warning(
                f"[SelfRef] Failed to generate transparency explanation: {e}"
            )

        # Build substantive conclusion
        from .self_ref_conclusion import build_self_referential_conclusion
        conclusion = build_self_referential_conclusion(
            reasoner, query_str, analysis
        )

        step = ReasoningStep(
            step_id=f"self_ref_{uuid.uuid4().hex[:8]}",
            step_type=ReasoningType.PHILOSOPHICAL,
            input_data=task.input_data,
            output_data=conclusion,
            confidence=SELF_REFERENTIAL_MIN_CONFIDENCE,
            explanation=(
                "Self-referential query analyzed through "
                "meta-reasoning infrastructure"
            ),
        )
        reasoning_chain.steps.append(step)

        result = ReasoningResult(
            conclusion=conclusion,
            confidence=SELF_REFERENTIAL_MIN_CONFIDENCE,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            explanation=(
                "This self-referential query was analyzed through VULCAN's "
                "meta-reasoning infrastructure, considering objective "
                "hierarchy, goal conflicts, ethical boundaries, and "
                "transparency requirements."
            ),
            metadata={
                'self_referential': True,
                'meta_reasoning_applied': True,
                'analysis': analysis,
            },
            reasoning_chain=reasoning_chain,
        )

        logger.info(
            f"[SelfRef] Meta-reasoning complete: "
            f"confidence={result.confidence:.2f}"
        )
        return result

    except ImportError as e:
        logger.error(
            f"[SelfRef] Failed to import meta-reasoning components: {e}"
        )
        return create_self_referential_fallback(reasoner, task, reasoning_chain)
    except Exception as e:
        logger.error(f"[SelfRef] Meta-reasoning failed: {e}")
        return create_self_referential_fallback(reasoner, task, reasoning_chain)


def create_self_referential_fallback(
    reasoner: Any,
    task: ReasoningTask,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Create fallback result when meta-reasoning components are unavailable.

    Args:
        reasoner: UnifiedReasoner instance.
        task: Original ReasoningTask.
        reasoning_chain: ReasoningChain to attach.

    Returns:
        Simple ReasoningResult with PHILOSOPHICAL type.
    """
    query_str = reasoner._extract_query_string(task.query)

    conclusion = (
        "This appears to be a self-referential query about my nature or "
        "capabilities. As an AI system, I operate through computational "
        "processes guided by predefined objectives and ethical constraints. "
        "I aim to be transparent about my limitations while providing "
        "helpful, accurate responses."
    )

    return ReasoningResult(
        conclusion=conclusion,
        confidence=0.5,
        reasoning_type=ReasoningType.PHILOSOPHICAL,
        explanation="Self-referential query handled with basic introspection",
        metadata={'self_referential': True, 'fallback_mode': True},
        reasoning_chain=reasoning_chain,
    )
