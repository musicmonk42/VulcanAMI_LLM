"""
Probabilistic, mathematical, and default reasoner execution.

Handles execution of reasoning engines that work with numerical,
statistical, or general-purpose computation.

Extracted from task_reasoner.py for modularity.

Author: VulcanAMI Team
"""

import logging
from typing import Any

from .config import (
    CONFIDENCE_FLOOR_DEFAULT,
    CONFIDENCE_FLOOR_NO_RESULT,
)
from .types import ReasoningTask
from ..reasoning_types import ReasoningResult, ReasoningType

logger = logging.getLogger(__name__)


def execute_probabilistic(
    engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Execute probabilistic reasoner."""
    query_dict = task.query if isinstance(task.query, dict) else {}
    threshold = query_dict.get("threshold", 0.5)

    reasoning_kwargs = {"threshold": threshold}
    if "skip_gate_check" in query_dict:
        reasoning_kwargs["skip_gate_check"] = query_dict["skip_gate_check"]
        reasoning_kwargs["router_confidence"] = query_dict.get(
            "router_confidence", 0.0
        )
        reasoning_kwargs["llm_classification"] = query_dict.get(
            "llm_classification", "unknown"
        )

    raw_result = engine.reason_with_uncertainty(
        input_data=task.input_data, **reasoning_kwargs
    )

    if isinstance(raw_result, ReasoningResult):
        result = raw_result
        if isinstance(result.conclusion, dict):
            conclusion_dict = result.conclusion
            if 'details' in conclusion_dict:
                result.conclusion = (
                    f"Analysis result: {conclusion_dict['details']}"
                )
            elif 'is_above_threshold' in conclusion_dict:
                threshold_met = conclusion_dict.get(
                    'is_above_threshold', False
                )
                result.conclusion = (
                    "Probabilistic analysis indicates: "
                    f"{'positive' if threshold_met else 'negative'} "
                    f"outcome (confidence: {result.confidence:.2f})"
                )
        return result
    else:
        raw_conclusion = raw_result.get("conclusion")
        if (
            isinstance(raw_conclusion, dict)
            and 'details' in raw_conclusion
        ):
            formatted = f"Analysis result: {raw_conclusion['details']}"
        else:
            formatted = raw_conclusion

        return ReasoningResult(
            conclusion=formatted,
            confidence=raw_result.get("confidence", 0.5),
            reasoning_type=task.task_type,
            explanation=raw_result.get("explanation", str(raw_result)),
        )


def execute_mathematical(
    engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Execute mathematical reasoner."""
    from .task_reasoner import _create_empty_result

    if not hasattr(engine, "reason"):
        logger.warning("Mathematical reasoner missing 'reason' method")
        return _create_empty_result()

    if isinstance(task.input_data, str):
        math_query = task.input_data
    elif isinstance(task.input_data, dict):
        math_query = (
            task.input_data.get('query')
            or task.input_data.get('problem')
            or str(task.input_data)
        )
    else:
        math_query = (
            str(task.query.get('query', ''))
            if isinstance(task.query, dict)
            else str(task.query)
        )

    raw_result = engine.reason(math_query, task.query)

    if isinstance(raw_result, ReasoningResult):
        return raw_result
    elif isinstance(raw_result, dict):
        return _format_math_result(raw_result)
    else:
        return ReasoningResult(
            conclusion=raw_result,
            confidence=0.9,
            reasoning_type=ReasoningType.MATHEMATICAL,
            explanation="Mathematical computation performed",
        )


def _format_math_result(raw_result: dict) -> ReasoningResult:
    """Format mathematical computation result."""
    conclusion = raw_result.get('conclusion', {})
    computed_result = None

    if isinstance(conclusion, dict):
        computed_result = conclusion.get('result')
        if not computed_result and conclusion.get('success'):
            computed_result = (
                conclusion.get('value') or conclusion.get('answer')
            )
    elif conclusion:
        computed_result = conclusion

    formatted_output = raw_result.get('formatted_output', '')
    stripped = (
        formatted_output.strip()
        if isinstance(formatted_output, str)
        else ""
    )

    if stripped:
        user_conclusion = formatted_output
    elif computed_result is not None:
        if isinstance(computed_result, dict):
            user_conclusion = (
                computed_result.get('value')
                or computed_result.get('answer')
                or str(computed_result)
            )
        else:
            user_conclusion = f"**Result:** {computed_result}"
    else:
        user_conclusion = raw_result

    return ReasoningResult(
        conclusion=user_conclusion,
        confidence=raw_result.get('confidence', 0.9),
        reasoning_type=ReasoningType.MATHEMATICAL,
        explanation=raw_result.get(
            'explanation', 'Mathematical computation performed'
        ),
        metadata=raw_result.get('metadata', {}),
    )


def execute_default(engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute default reasoner with standard interface."""
    from .task_reasoner import _create_empty_result

    if hasattr(engine, "reason"):
        raw_result = engine.reason(task.input_data, task.query)
        if isinstance(raw_result, ReasoningResult):
            result = raw_result
            if result.confidence == 0.0 and result.conclusion is not None:
                result.confidence = CONFIDENCE_FLOOR_DEFAULT
            return result
        else:
            raw_conf = (
                raw_result.get("confidence", CONFIDENCE_FLOOR_DEFAULT)
                if isinstance(raw_result, dict)
                else CONFIDENCE_FLOOR_DEFAULT
            )
            confidence = (
                max(CONFIDENCE_FLOOR_DEFAULT, raw_conf)
                if raw_result
                else CONFIDENCE_FLOOR_NO_RESULT
            )
            return ReasoningResult(
                conclusion=(
                    raw_result.get("conclusion")
                    if isinstance(raw_result, dict)
                    else raw_result
                ),
                confidence=confidence,
                reasoning_type=task.task_type,
                explanation=str(raw_result),
            )
    return _create_empty_result()
