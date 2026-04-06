"""
Reasoner execution for unified reasoning orchestration.

Dispatches to specific reasoner types (probabilistic, symbolic, causal,
philosophical, mathematical, analogical) and handles result formatting
and reasoning chain creation.

Extracted from orchestrator.py for modularity.
Sub-modules: task_reasoner_symbolic, task_reasoner_ml.

Author: VulcanAMI Team
"""

import logging
import time
import uuid
from typing import Any

from .types import ReasoningTask
from ..reasoning_types import (
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningType,
)

logger = logging.getLogger(__name__)


def execute_reasoner(
    reasoner: Any, engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """
    Execute specific reasoner with task and measure execution time.

    Args:
        reasoner: UnifiedReasoner instance.
        engine: The reasoning engine to execute.
        task: ReasoningTask to execute.

    Returns:
        ReasoningResult with execution metadata.
    """
    result = None
    start_time = time.time()
    try:
        result = _dispatch_reasoner(reasoner, engine, task)
    except Exception as e:
        logger.error(f"Reasoner execution failed: {e}")
        result = _create_error_result(str(e))
    finally:
        elapsed_time_ms = (time.time() - start_time) * 1000
        if result:
            if not hasattr(result, "metadata") or result.metadata is None:
                result.metadata = {}
            result.metadata["execution_time_ms"] = elapsed_time_ms

            if not result.reasoning_chain or not result.reasoning_chain.steps:
                _attach_reasoning_chain(result, task)

    if result is None:
        logger.warning(
            f"[UnifiedReasoner] No result from _execute_reasoner "
            f"for task_type={task.task_type}"
        )
        result = _create_empty_result()

    return result


def _dispatch_reasoner(
    reasoner: Any, engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Dispatch to the appropriate reasoner based on task type."""
    from .task_reasoner_symbolic import (
        execute_symbolic,
        execute_causal,
        execute_philosophical,
        execute_analogical,
    )
    from .task_reasoner_ml import (
        execute_probabilistic,
        execute_mathematical,
        execute_default,
    )

    if task.task_type == ReasoningType.PROBABILISTIC:
        return execute_probabilistic(engine, task)
    elif task.task_type == ReasoningType.SYMBOLIC:
        return execute_symbolic(reasoner, engine, task)
    elif task.task_type == ReasoningType.CAUSAL:
        return execute_causal(engine, task)
    elif task.task_type == ReasoningType.PHILOSOPHICAL:
        return execute_philosophical(engine, task)
    elif task.task_type == ReasoningType.MATHEMATICAL:
        return execute_mathematical(engine, task)
    elif task.task_type == ReasoningType.ANALOGICAL:
        return execute_analogical(engine, task)
    else:
        return execute_default(engine, task)


def _attach_reasoning_chain(
    result: ReasoningResult, task: ReasoningTask
) -> None:
    """Attach a reasoning chain to a result that lacks one."""
    try:
        is_error = (
            isinstance(result.conclusion, dict)
            and "error" in result.conclusion
        )
        if not is_error:
            step = ReasoningStep(
                step_id=(
                    f"{task.task_type.value}_"
                    f"{uuid.uuid4().hex[:8]}"
                ),
                step_type=task.task_type,
                input_data=task.input_data,
                output_data=result.conclusion,
                confidence=result.confidence,
                explanation=(
                    result.explanation
                    or f"Executed {task.task_type.value} reasoner."
                ),
            )
            result.reasoning_chain = ReasoningChain(
                chain_id=str(uuid.uuid4()),
                steps=[step],
                initial_query=task.query,
                final_conclusion=result.conclusion,
                total_confidence=result.confidence,
                reasoning_types_used={task.task_type},
                modalities_involved=set(),
                safety_checks=[],
                audit_trail=[],
            )
    except Exception as e:
        logger.warning(
            f"Failed to create reasoning chain for "
            f"task {task.task_id}: {e}"
        )


def _create_empty_result() -> ReasoningResult:
    """Create empty result with minimal confidence."""
    return ReasoningResult(
        conclusion=None,
        confidence=0.1,
        reasoning_type=ReasoningType.UNKNOWN,
        explanation="No reasoning performed - using minimal fallback confidence",
    )


def _create_error_result(error: str) -> ReasoningResult:
    """Create error result with minimal confidence."""
    return ReasoningResult(
        conclusion={"error": error},
        confidence=0.1,
        reasoning_type=ReasoningType.UNKNOWN,
        explanation=f"Reasoning error: {error}",
    )
