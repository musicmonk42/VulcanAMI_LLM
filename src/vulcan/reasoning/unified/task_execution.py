"""
Task execution for unified reasoning orchestration.

Executes individual reasoning tasks, handling COMMAND PATTERN where
task.task_type was determined during planning by ToolSelector and
is executed without re-decision at runtime.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
from typing import Any

from .config import UNKNOWN_TYPE_FALLBACK_ORDER
from .types import ReasoningTask
from ..reasoning_types import ReasoningResult, ReasoningType

logger = logging.getLogger(__name__)


def execute_task(reasoner: Any, task: ReasoningTask) -> ReasoningResult:
    """
    Execute a single reasoning task using the COMMAND PATTERN.

    task.task_type was determined during planning by ToolSelector.
    Execution uses that decision without re-deciding.

    Args:
        reasoner: UnifiedReasoner instance.
        task: ReasoningTask to execute.

    Returns:
        ReasoningResult from the selected reasoner.
    """
    try:
        if task.task_type in reasoner.reasoners:
            r = reasoner.reasoners[task.task_type]
            return reasoner._execute_reasoner(r, task)

        elif task.task_type == ReasoningType.HYBRID:
            if ReasoningType.PROBABILISTIC in reasoner.reasoners:
                fallback_task = ReasoningTask(
                    task_id=task.task_id,
                    task_type=ReasoningType.PROBABILISTIC,
                    input_data=task.input_data,
                    query=task.query,
                    constraints=task.constraints,
                    utility_context=task.utility_context,
                )
                result = reasoner._execute_reasoner(
                    reasoner.reasoners[ReasoningType.PROBABILISTIC],
                    fallback_task,
                )
                result.reasoning_type = ReasoningType.HYBRID
                return result
            else:
                logger.warning(
                    f"No reasoner for type {task.task_type} and "
                    "no PROBABILISTIC fallback available"
                )
                return reasoner._create_empty_result()

        elif task.task_type == ReasoningType.UNKNOWN:
            return _handle_unknown_type(reasoner, task)

        elif task.task_type == ReasoningType.PHILOSOPHICAL:
            return _handle_philosophical_type(reasoner, task)

        else:
            logger.warning(f"No reasoner for type {task.task_type}")
            return reasoner._create_empty_result()

    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return reasoner._create_error_result(str(e))


def _handle_unknown_type(
    reasoner: Any, task: ReasoningTask
) -> ReasoningResult:
    """Handle UNKNOWN reasoning type with fallback chain."""
    logger.info(
        f"[UnifiedReasoner] Task {task.task_id} has UNKNOWN type, "
        "attempting fallback to available reasoners"
    )

    for fallback_name in UNKNOWN_TYPE_FALLBACK_ORDER:
        try:
            fallback_type = ReasoningType[fallback_name]
        except KeyError:
            logger.warning(
                f"[UnifiedReasoner] Invalid fallback type: {fallback_name}"
            )
            continue

        if fallback_type in reasoner.reasoners:
            logger.info(
                f"[UnifiedReasoner] Using {fallback_type.value} "
                "as fallback for UNKNOWN type"
            )
            fallback_task = ReasoningTask(
                task_id=task.task_id,
                task_type=fallback_type,
                input_data=task.input_data,
                query=task.query,
                constraints=task.constraints,
                utility_context=task.utility_context,
            )
            return reasoner._execute_reasoner(
                reasoner.reasoners[fallback_type], fallback_task
            )

    logger.warning(
        "[UnifiedReasoner] No fallback reasoner available for UNKNOWN type."
    )
    return reasoner._create_empty_result()


def _handle_philosophical_type(
    reasoner: Any, task: ReasoningTask
) -> ReasoningResult:
    """Handle PHILOSOPHICAL reasoning type via World Model."""
    logger.info(
        f"[UnifiedReasoner] FIX Issue B: PHILOSOPHICAL type detected "
        f"for task {task.task_id}, routing to World Model"
    )

    world_model = None
    try:
        from vulcan.reasoning.singletons import get_world_model
        world_model = get_world_model()
    except (ImportError, Exception) as e:
        logger.debug(f"World Model singleton not available: {e}")

    if world_model is None:
        try:
            from vulcan.world_model.world_model_core import WorldModel
            world_model = WorldModel()
        except (ImportError, Exception) as e:
            logger.debug(f"Failed to instantiate WorldModel: {e}")

    if world_model is not None and hasattr(world_model, 'reason'):
        try:
            if isinstance(task.input_data, str):
                query_str = task.input_data
            elif isinstance(task.input_data, dict):
                query_str = (
                    task.input_data.get('query')
                    or task.input_data.get('text')
                    or str(task.input_data)
                )
            elif isinstance(task.query, dict):
                query_str = (
                    task.query.get('query')
                    or task.query.get('text')
                    or str(task.query)
                )
            else:
                query_str = (
                    str(task.query) if task.query else str(task.input_data)
                )

            wm_result = world_model.reason(query_str, mode='philosophical')

            if isinstance(wm_result, dict):
                return ReasoningResult(
                    conclusion=wm_result.get('response', wm_result),
                    confidence=max(
                        0.35, wm_result.get('confidence', 0.80)
                    ),
                    reasoning_type=ReasoningType.PHILOSOPHICAL,
                    explanation="Philosophical reasoning via World Model",
                    metadata={
                        'reasoning_trace': wm_result.get(
                            'reasoning_trace', {}
                        ),
                        'mode': 'philosophical',
                        'source': 'world_model',
                    },
                )
            else:
                return ReasoningResult(
                    conclusion=wm_result,
                    confidence=0.80,
                    reasoning_type=ReasoningType.PHILOSOPHICAL,
                    explanation="Philosophical reasoning via World Model",
                )
        except Exception as e:
            logger.warning(
                f"World Model philosophical reasoning failed: {e}"
            )

    # Fallback to PROBABILISTIC
    if ReasoningType.PROBABILISTIC in reasoner.reasoners:
        logger.warning(
            "[UnifiedReasoner] World Model not available for "
            "PHILOSOPHICAL, falling back to PROBABILISTIC"
        )
        fallback_task = ReasoningTask(
            task_id=task.task_id,
            task_type=ReasoningType.PROBABILISTIC,
            input_data=task.input_data,
            query=task.query,
            constraints=task.constraints,
            utility_context=task.utility_context,
        )
        result = reasoner._execute_reasoner(
            reasoner.reasoners[ReasoningType.PROBABILISTIC], fallback_task
        )
        result.reasoning_type = ReasoningType.PHILOSOPHICAL
        return result

    logger.warning(
        f"No reasoner available for PHILOSOPHICAL type "
        f"(task {task.task_id})"
    )
    return reasoner._create_empty_result()
