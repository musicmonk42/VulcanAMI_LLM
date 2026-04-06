"""
Symbolic, causal, analogical, and philosophical reasoner execution.

Handles execution of reasoning engines that work with structured,
logical, or relational knowledge representations.

Extracted from task_reasoner.py for modularity.

Author: VulcanAMI Team
"""

import logging
from typing import Any

from .config import (
    CONFIDENCE_FLOOR_ANALOGICAL_DEFAULT,
    CONFIDENCE_FLOOR_CAUSAL_DEFAULT,
    CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT,
    CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
    CONFIDENCE_FLOOR_SYMBOLIC_HAS_PROOF,
    CONFIDENCE_FLOOR_SYMBOLIC_PROVEN,
)
from .types import ReasoningTask
from ..reasoning_types import ReasoningResult, ReasoningType

logger = logging.getLogger(__name__)


def _symbolic_confidence(query_result: Any) -> float:
    """Compute confidence for a symbolic query result."""
    raw = query_result.get("confidence", 0.0) if isinstance(query_result, dict) else 0.0
    if isinstance(query_result, dict) and query_result.get("proven"):
        return max(CONFIDENCE_FLOOR_SYMBOLIC_PROVEN, raw)
    if isinstance(query_result, dict) and query_result.get("proof") is not None:
        return max(CONFIDENCE_FLOOR_SYMBOLIC_HAS_PROOF, raw)
    return max(CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT, raw)


def _symbolic_result(query_result: Any, task: ReasoningTask, fallback_expl: str = "No proof found") -> ReasoningResult:
    """Build a ReasoningResult from a symbolic query result."""
    explanation = (
        str(query_result.get("proof", fallback_expl))
        if isinstance(query_result, dict) else str(query_result)
    )
    return ReasoningResult(
        conclusion=query_result,
        confidence=_symbolic_confidence(query_result),
        reasoning_type=task.task_type,
        explanation=explanation,
    )


def execute_symbolic(reasoner: Any, engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute symbolic reasoner."""
    if isinstance(task.input_data, str):
        extracted = reasoner._extract_symbolic_constraints(task.input_data)

        if extracted["constraints"]:
            for constraint in extracted["constraints"]:
                try:
                    engine.add_rule(constraint)
                except Exception as e:
                    logger.debug(f"Failed to add constraint '{constraint}': {e}")

            if extracted["is_sat_query"]:
                try:
                    return reasoner._check_sat_satisfiability(engine, extracted)
                except Exception as e:
                    logger.debug(f"SAT check failed: {e}")
                    return ReasoningResult(
                        conclusion={"satisfiable": "unknown", "reason": str(e)},
                        confidence=CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
                        reasoning_type=task.task_type,
                        explanation=f"Could not determine satisfiability: {e}",
                    )
            else:
                hypothesis = extracted.get("hypothesis", task.query.get("goal", ""))
                if hypothesis:
                    return _symbolic_result(engine.query(hypothesis), task)
                count = len(extracted["constraints"])
                return ReasoningResult(
                    conclusion=(
                        f"Extracted {count} logical constraint(s) from the query, "
                        "but no specific hypothesis was provided to evaluate."
                    ),
                    confidence=CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
                    reasoning_type=task.task_type,
                    explanation=(
                        "The symbolic reasoner successfully parsed the logical "
                        "structure, but needs a specific question or hypothesis to prove."
                    ),
                    metadata={
                        "constraints_added": count,
                        "extracted_constraints": extracted.get("constraints", []),
                        "parsed_successfully": True,
                    },
                )
        else:
            return _symbolic_result(
                engine.query(task.input_data), task, "Direct query attempted"
            )
    else:
        hypothesis = task.query.get("goal", "")
        kb_data = (
            task.input_data.get("kb", [])
            if isinstance(task.input_data, dict) else []
        )
        for fact in kb_data:
            engine.add_rule(fact)
        return _symbolic_result(engine.query(hypothesis), task)


def execute_causal(engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute causal reasoner."""
    from .task_reasoner import _create_empty_result

    if not hasattr(engine, "reason"):
        return _create_empty_result()

    result_dict = engine.reason(task.input_data, task.query)
    raw_conf = (
        result_dict.get("confidence", CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT)
        if isinstance(result_dict, dict) else CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT
    )
    if isinstance(result_dict, dict) and not result_dict.get("error"):
        confidence = max(CONFIDENCE_FLOOR_CAUSAL_WITH_RESULT, raw_conf)
    else:
        confidence = max(CONFIDENCE_FLOOR_CAUSAL_DEFAULT, raw_conf)

    return ReasoningResult(
        conclusion=result_dict, confidence=confidence,
        reasoning_type=task.task_type, explanation="Causal analysis performed",
    )


def execute_philosophical(engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute philosophical reasoner."""
    from .task_reasoner import _create_empty_result

    if not hasattr(engine, "reason"):
        logger.warning("Philosophical reasoner missing 'reason' method")
        return _create_empty_result()

    problem = task.query if isinstance(task.query, dict) else {'query': str(task.query)}
    if task.input_data:
        if isinstance(task.input_data, dict):
            problem.update(task.input_data)
        else:
            problem['input'] = task.input_data

    raw_result = engine.reason(problem, mode='philosophical')

    if isinstance(raw_result, ReasoningResult):
        if raw_result.confidence < 0.2:
            raw_result.confidence = max(0.35, raw_result.confidence)
        return raw_result

    raw_conf = raw_result.get("confidence", 0.55) if isinstance(raw_result, dict) else 0.55
    return ReasoningResult(
        conclusion=raw_result, confidence=max(0.35, raw_conf),
        reasoning_type=ReasoningType.PHILOSOPHICAL,
        explanation="Philosophical/ethical analysis performed",
    )


def execute_analogical(engine: Any, task: ReasoningTask) -> ReasoningResult:
    """Execute analogical reasoner."""
    from .task_reasoner import _create_empty_result

    if not hasattr(engine, "reason"):
        logger.warning("Analogical reasoner missing 'reason' method")
        return _create_empty_result()

    raw_result = engine.reason(task.input_data, task.query)
    if isinstance(raw_result, ReasoningResult):
        return raw_result
    if isinstance(raw_result, dict):
        return ReasoningResult(
            conclusion=raw_result.get("conclusion") or raw_result,
            confidence=max(
                CONFIDENCE_FLOOR_ANALOGICAL_DEFAULT,
                raw_result.get("confidence", 0.5),
            ),
            reasoning_type=ReasoningType.ANALOGICAL,
            explanation=raw_result.get("explanation", "Analogical reasoning performed"),
        )
    return ReasoningResult(
        conclusion=raw_result, confidence=0.5,
        reasoning_type=ReasoningType.ANALOGICAL,
        explanation="Analogical reasoning performed",
    )
