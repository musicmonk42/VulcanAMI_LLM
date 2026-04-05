"""
Mathematical Verification Utilities

Standalone verification functions extracted from orchestrator.py.
These perform mathematical result verification and apply corrections,
using the MathematicalVerificationEngine and learning integration.

All functions accept their dependencies as explicit parameters
rather than relying on UnifiedReasoner instance state.

Author: VulcanAMI Team
"""

import logging
from typing import Any, Dict, Optional

from .config import (
    MATH_ERROR_CONFIDENCE_PENALTY,
    MATH_VERIFICATION_CONFIDENCE_BOOST,
    NUMERICAL_RESULT_KEYS,
    PROBLEM_TYPE_BAYESIAN,
)
from .types import ReasoningTask
from ..reasoning_types import ReasoningResult

logger = logging.getLogger(__name__)


def verify_mathematical_result(
    result: ReasoningResult,
    task: ReasoningTask,
    math_verification_engine: Any,
    optional_components: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Verify mathematical calculation results using MathematicalVerificationEngine.

    Integrates the MathematicalVerificationEngine into the calculation
    validation workflow to detect and correct mathematical errors.

    Args:
        result: The reasoning result to verify.
        task: The original task with context.
        math_verification_engine: A MathematicalVerificationEngine instance.
        optional_components: Dict of optional loaded components (must contain
            ``"BayesianProblem"`` for Bayesian verification).

    Returns:
        VerificationResult if verification was performed, None otherwise.
    """
    if not math_verification_engine:
        return None

    if optional_components is None:
        optional_components = {}

    try:
        conclusion = result.conclusion
        if conclusion is None:
            return None

        # Extract numerical result if present
        if isinstance(conclusion, dict):
            numerical_value = None
            for key in NUMERICAL_RESULT_KEYS:
                if key in conclusion and isinstance(conclusion[key], (int, float)):
                    numerical_value = conclusion[key]
                    break

            if numerical_value is None:
                return None

            # Check for Bayesian calculation context
            if "prior" in conclusion or (
                task.query and task.query.get("problem_type") == PROBLEM_TYPE_BAYESIAN
            ):
                BayesianProblem = optional_components.get("BayesianProblem")
                if BayesianProblem:
                    query = task.query or {}
                    problem = BayesianProblem(
                        prior=conclusion.get("prior", query.get("prior", 0.01)),
                        sensitivity=conclusion.get(
                            "sensitivity", query.get("sensitivity")
                        ),
                        specificity=conclusion.get(
                            "specificity", query.get("specificity")
                        ),
                    )

                    verification = (
                        math_verification_engine.verify_bayesian_calculation(
                            problem, numerical_value
                        )
                    )
                    logger.info(
                        f"[MathVerification] Bayesian verification: "
                        f"status={verification.status.value}, "
                        f"confidence={verification.confidence:.2f}"
                    )
                    return verification

            # For general arithmetic, verify the expression if available
            query = task.query or {}
            expression = conclusion.get("expression", query.get("expression"))
            if expression and isinstance(expression, str):
                variables = conclusion.get("variables", query.get("variables", {}))
                verification = math_verification_engine.verify_arithmetic(
                    expression, numerical_value, variables
                )
                logger.info(
                    f"[MathVerification] Arithmetic verification: "
                    f"status={verification.status.value}, "
                    f"confidence={verification.confidence:.2f}"
                )
                return verification

        elif isinstance(conclusion, (int, float)):
            # Direct numerical result - check for expression in query
            expression = task.query.get("expression") if task.query else None
            if expression:
                variables = task.query.get("variables", {})
                verification = math_verification_engine.verify_arithmetic(
                    expression, conclusion, variables
                )
                return verification

    except Exception as e:
        logger.warning(f"Mathematical verification failed: {e}")

    return None


def apply_verification_to_result(
    result: ReasoningResult,
    verification: Any,
    task: ReasoningTask,
    math_verification_status_class: Optional[Any] = None,
    math_accuracy_integration: Optional[Any] = None,
    learner: Optional[Any] = None,
) -> ReasoningResult:
    """
    Apply verification results to a ReasoningResult and update learning system.

    If verification detects errors, applies corrections and triggers
    learning system penalties/rewards based on mathematical accuracy.

    Args:
        result: Original reasoning result.
        verification: VerificationResult from math engine.
        task: Original task for context.
        math_verification_status_class: The MathVerificationStatus enum class
            (needed for status comparison).
        math_accuracy_integration: Optional math accuracy integration object
            with ``reward_tool`` and ``penalize_tool`` methods.
        learner: Optional learner instance passed to accuracy integration.

    Returns:
        Updated ReasoningResult with verification applied.
    """
    if not math_verification_status_class:
        return result

    try:
        # Add verification metadata to result
        if not hasattr(result, "metadata") or result.metadata is None:
            result.metadata = {}
        result.metadata["math_verification"] = {
            "status": verification.status.value,
            "confidence": verification.confidence,
            "errors": (
                [e.value for e in verification.errors] if verification.errors else []
            ),
        }

        if verification.status == math_verification_status_class.VERIFIED:
            # Correct result - boost confidence and trigger reward
            result.confidence = min(
                1.0, result.confidence * MATH_VERIFICATION_CONFIDENCE_BOOST
            )
            logger.info("[MathVerification] Calculation verified as correct")

            # Reward tool through learning integration
            if math_accuracy_integration and learner:
                tool_name = (
                    task.task_type.value if task.task_type else "unknown"
                )
                math_accuracy_integration.reward_tool(tool_name, learner)

        elif verification.status == math_verification_status_class.ERROR_DETECTED:
            # Error detected - apply corrections and trigger penalty
            logger.warning(
                f"[MathVerification] Mathematical error detected: "
                f"{verification.errors}"
            )

            # Apply corrections to result (handle non-dict conclusions safely)
            if verification.corrections:
                if isinstance(result.conclusion, dict):
                    corrected_conclusion = result.conclusion.copy()
                else:
                    corrected_conclusion = {"original_value": result.conclusion}
                corrected_conclusion["math_correction"] = {
                    "original": result.conclusion,
                    "corrected": (
                        verification.corrections.get("correct_posterior")
                        or verification.corrections.get("correct_result")
                    ),
                    "errors": [e.value for e in verification.errors],
                    "explanation": verification.explanation,
                }
                result.conclusion = corrected_conclusion

            # Reduce confidence due to detected error
            result.confidence = max(
                0.0, result.confidence * MATH_ERROR_CONFIDENCE_PENALTY
            )
            result.explanation = (
                (result.explanation or "")
                + f"\n[Math Error: {verification.explanation}]"
            )

            # Penalize tool through learning integration
            if math_accuracy_integration and learner and verification.errors:
                tool_name = (
                    task.task_type.value if task.task_type else "unknown"
                )
                for error in verification.errors:
                    math_accuracy_integration.penalize_tool(
                        tool_name, error, learner
                    )

    except Exception as e:
        logger.warning(f"Failed to apply verification to result: {e}")

    return result


def postprocess_result(
    result: ReasoningResult,
    task: ReasoningTask,
    confidence_threshold: float,
    explainer: Optional[Any] = None,
    math_verification_engine: Optional[Any] = None,
    optional_components: Optional[Dict[str, Any]] = None,
    math_verification_status_class: Optional[Any] = None,
    math_accuracy_integration: Optional[Any] = None,
    learner: Optional[Any] = None,
) -> ReasoningResult:
    """
    Post-process a reasoning result with explanation generation and math verification.

    This is the top-level post-processing function that:
    1. Generates an explanation from the reasoning chain if missing.
    2. Annotates results below the confidence threshold (without altering conclusion).
    3. Runs mathematical verification when applicable.

    Args:
        result: The reasoning result to post-process.
        task: The original reasoning task.
        confidence_threshold: Minimum confidence threshold.
        explainer: Optional ReasoningExplainer for chain explanation.
        math_verification_engine: Optional MathematicalVerificationEngine.
        optional_components: Optional dict of loaded components.
        math_verification_status_class: Optional MathVerificationStatus enum.
        math_accuracy_integration: Optional math accuracy integration object.
        learner: Optional learner instance.

    Returns:
        Post-processed ReasoningResult.
    """
    try:
        if not result.explanation and result.reasoning_chain and explainer:
            result.explanation = explainer.explain_chain(result.reasoning_chain)

        threshold = task.constraints.get(
            "confidence_threshold", confidence_threshold
        )
        if result.confidence < threshold:
            result.metadata["below_confidence_threshold"] = True
            result.metadata["filter_reason"] = (
                f"Confidence {result.confidence:.2f} below threshold {threshold}"
            )
            result.metadata["threshold"] = threshold

            if not result.explanation or result.explanation.strip() == "":
                result.explanation = (
                    "Analysis completed with moderate confidence. "
                    "Results may benefit from additional context or verification."
                )

        # Apply mathematical verification to calculation results
        is_mathematical = (
            task.query.get("is_mathematical", False) if task.query else False
        )
        require_verification = (
            task.constraints.get("require_verification", False)
            if task.constraints
            else False
        )

        if (is_mathematical or require_verification) and math_verification_engine:
            verification_result = verify_mathematical_result(
                result,
                task,
                math_verification_engine,
                optional_components=optional_components,
            )
            if verification_result:
                result = apply_verification_to_result(
                    result,
                    verification_result,
                    task,
                    math_verification_status_class=math_verification_status_class,
                    math_accuracy_integration=math_accuracy_integration,
                    learner=learner,
                )

    except Exception as e:
        logger.warning(f"Post-processing failed: {e}")

    return result
