"""
Reasoner execution for unified reasoning orchestration.

Executes specific reasoner types (probabilistic, symbolic, causal,
philosophical, mathematical, analogical) with proper result formatting
and reasoning chain creation.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import time
import uuid
from typing import Any

from .config import (
    CONFIDENCE_FLOOR_ANALOGICAL_DEFAULT,
    CONFIDENCE_FLOOR_CAUSAL_DEFAULT,
    CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT,
    CONFIDENCE_FLOOR_DEFAULT,
    CONFIDENCE_FLOOR_NO_RESULT,
    CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
    CONFIDENCE_FLOOR_SYMBOLIC_HAS_PROOF,
    CONFIDENCE_FLOOR_SYMBOLIC_PROVEN,
)
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
        if task.task_type == ReasoningType.PROBABILISTIC:
            result = _execute_probabilistic(engine, task)
        elif task.task_type == ReasoningType.SYMBOLIC:
            result = _execute_symbolic(reasoner, engine, task)
        elif task.task_type == ReasoningType.CAUSAL:
            result = _execute_causal(engine, task)
        elif task.task_type == ReasoningType.PHILOSOPHICAL:
            result = _execute_philosophical(engine, task)
        elif task.task_type == ReasoningType.MATHEMATICAL:
            result = _execute_mathematical(engine, task)
        elif task.task_type == ReasoningType.ANALOGICAL:
            result = _execute_analogical(engine, task)
        else:
            result = _execute_default(engine, task)
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

    if result is None:
        logger.warning(
            f"[UnifiedReasoner] No result from _execute_reasoner "
            f"for task_type={task.task_type}"
        )
        result = _create_empty_result()

    return result


def _execute_probabilistic(
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


def _execute_symbolic(
    reasoner: Any, engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Execute symbolic reasoner."""
    if isinstance(task.input_data, str):
        extracted = reasoner._extract_symbolic_constraints(task.input_data)

        if extracted["constraints"]:
            for constraint in extracted["constraints"]:
                try:
                    engine.add_rule(constraint)
                except Exception as e:
                    logger.debug(
                        f"Failed to add constraint '{constraint}': {e}"
                    )

            if extracted["is_sat_query"]:
                try:
                    return reasoner._check_sat_satisfiability(
                        engine, extracted
                    )
                except Exception as e:
                    logger.debug(f"SAT check failed: {e}")
                    return ReasoningResult(
                        conclusion={
                            "satisfiable": "unknown",
                            "reason": str(e),
                        },
                        confidence=CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
                        reasoning_type=task.task_type,
                        explanation=(
                            f"Could not determine satisfiability: {e}"
                        ),
                    )
            else:
                hypothesis = extracted.get(
                    "hypothesis", task.query.get("goal", "")
                )
                if hypothesis:
                    query_result = engine.query(hypothesis)
                    raw_conf = (
                        query_result.get("confidence", 0.0)
                        if isinstance(query_result, dict)
                        else 0.0
                    )
                    if (
                        isinstance(query_result, dict)
                        and query_result.get("proven")
                    ):
                        confidence = max(
                            CONFIDENCE_FLOOR_SYMBOLIC_PROVEN, raw_conf
                        )
                    else:
                        confidence = max(
                            CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT, raw_conf
                        )
                    return ReasoningResult(
                        conclusion=query_result,
                        confidence=confidence,
                        reasoning_type=task.task_type,
                        explanation=str(
                            query_result.get("proof", "No proof found")
                        ),
                    )
                else:
                    count = len(extracted["constraints"])
                    return ReasoningResult(
                        conclusion=(
                            f"Extracted {count} logical constraint(s) "
                            "from the query, but no specific hypothesis "
                            "was provided to evaluate."
                        ),
                        confidence=CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
                        reasoning_type=task.task_type,
                        explanation=(
                            "The symbolic reasoner successfully parsed "
                            "the logical structure, but needs a specific "
                            "question or hypothesis to prove."
                        ),
                        metadata={
                            "constraints_added": count,
                            "extracted_constraints": extracted.get(
                                "constraints", []
                            ),
                            "parsed_successfully": True,
                        },
                    )
        else:
            query_result = engine.query(task.input_data)
            raw_conf = (
                query_result.get("confidence", 0.0)
                if isinstance(query_result, dict)
                else 0.0
            )
            return ReasoningResult(
                conclusion=query_result,
                confidence=max(
                    CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT, raw_conf
                ),
                reasoning_type=task.task_type,
                explanation=str(
                    query_result.get("proof", "Direct query attempted")
                ),
            )
    else:
        hypothesis = task.query.get("goal", "")
        kb_data = (
            task.input_data.get("kb", [])
            if isinstance(task.input_data, dict)
            else []
        )
        for fact in kb_data:
            engine.add_rule(fact)

        query_result = engine.query(hypothesis)
        raw_conf = (
            query_result.get("confidence", 0.0)
            if isinstance(query_result, dict)
            else 0.0
        )
        if isinstance(query_result, dict) and query_result.get("proven"):
            confidence = max(CONFIDENCE_FLOOR_SYMBOLIC_PROVEN, raw_conf)
        elif (
            isinstance(query_result, dict)
            and query_result.get("proof") is not None
        ):
            confidence = max(CONFIDENCE_FLOOR_SYMBOLIC_HAS_PROOF, raw_conf)
        else:
            confidence = max(CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT, raw_conf)

        return ReasoningResult(
            conclusion=query_result,
            confidence=confidence,
            reasoning_type=task.task_type,
            explanation=(
                str(query_result.get("proof"))
                if isinstance(query_result, dict)
                else str(query_result)
            ),
        )


def _execute_causal(engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute causal reasoner."""
    if hasattr(engine, "reason"):
        result_dict = engine.reason(task.input_data, task.query)
        raw_conf = (
            result_dict.get("confidence", CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT)
            if isinstance(result_dict, dict)
            else CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT
        )
        if isinstance(result_dict, dict) and not result_dict.get("error"):
            confidence = max(CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT, raw_conf)
        else:
            confidence = max(CONFIDENCE_FLOOR_CAUSAL_DEFAULT, raw_conf)

        return ReasoningResult(
            conclusion=result_dict,
            confidence=confidence,
            reasoning_type=task.task_type,
            explanation="Causal analysis performed",
        )
    return _create_empty_result()


def _execute_philosophical(
    engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Execute philosophical reasoner."""
    if hasattr(engine, "reason"):
        problem = (
            task.query if isinstance(task.query, dict)
            else {'query': str(task.query)}
        )
        if task.input_data:
            if isinstance(task.input_data, dict):
                problem.update(task.input_data)
            else:
                problem['input'] = task.input_data

        raw_result = engine.reason(problem, mode='philosophical')

        if isinstance(raw_result, ReasoningResult):
            result = raw_result
            if result.confidence < 0.2:
                result.confidence = max(0.35, result.confidence)
            return result
        else:
            raw_conf = (
                raw_result.get("confidence", 0.55)
                if isinstance(raw_result, dict)
                else 0.55
            )
            return ReasoningResult(
                conclusion=raw_result,
                confidence=max(0.35, raw_conf),
                reasoning_type=ReasoningType.PHILOSOPHICAL,
                explanation="Philosophical/ethical analysis performed",
            )

    logger.warning("Philosophical reasoner missing 'reason' method")
    return _create_empty_result()


def _execute_mathematical(
    engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Execute mathematical reasoner."""
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


def _execute_analogical(
    engine: Any, task: ReasoningTask
) -> ReasoningResult:
    """Execute analogical reasoner."""
    if hasattr(engine, "reason"):
        raw_result = engine.reason(task.input_data, task.query)
        if isinstance(raw_result, ReasoningResult):
            return raw_result
        elif isinstance(raw_result, dict):
            return ReasoningResult(
                conclusion=raw_result.get("conclusion") or raw_result,
                confidence=max(
                    CONFIDENCE_FLOOR_ANALOGICAL_DEFAULT,
                    raw_result.get("confidence", 0.5),
                ),
                reasoning_type=ReasoningType.ANALOGICAL,
                explanation=raw_result.get(
                    "explanation", "Analogical reasoning performed"
                ),
            )
        else:
            return ReasoningResult(
                conclusion=raw_result,
                confidence=0.5,
                reasoning_type=ReasoningType.ANALOGICAL,
                explanation="Analogical reasoning performed",
            )

    logger.warning("Analogical reasoner missing 'reason' method")
    return _create_empty_result()


def _execute_default(engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute default reasoner with standard interface."""
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
