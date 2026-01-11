"""
Reasoning execution strategies for unified reasoning orchestration.

This module contains all strategy implementations for executing reasoning tasks:
- Sequential reasoning (tasks executed in order)
- Parallel reasoning (concurrent execution with ThreadPoolExecutor)
- Ensemble reasoning (weighted voting with non-applicable filtering)
- Adaptive reasoning (characteristic-based strategy selection)
- Hybrid reasoning (probabilistic → symbolic → causal cascade)
- Hierarchical reasoning (dependency-aware topological execution)
- Portfolio reasoning (using PortfolioExecutor)
- Utility-based reasoning (maximizing utility instead of confidence)

All strategies properly manage reasoning chains, handle errors gracefully,
and support learning/adaptation through weight management.

Author: VulcanAMI Team
Version: 2.0 (Post-refactoring)
"""

import logging
import uuid
from collections import defaultdict, deque
from concurrent.futures import TimeoutError
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from .config import MIN_ENSEMBLE_WEIGHT_FLOOR
from .types import ReasoningPlan, ReasoningTask
from ..reasoning_types import (
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningType,
)

# Use existing planning module for plan optimization
from vulcan.planning import Plan, PlanStep

if TYPE_CHECKING:
    from .orchestrator import UnifiedReasoner

logger = logging.getLogger(__name__)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _is_result_not_applicable(result: Any) -> bool:
    """
    Check if a reasoning result indicates the reasoner was not applicable.
    
    This prevents non-applicable reasoners from contaminating ensemble
    confidence scores. When a reasoner returns "not applicable" (e.g.,
    probabilistic reasoner on a philosophical query), it should be excluded
    from the ensemble calculation rather than dragging down confidence.
    
    Enhanced to check for:
    - "uninformative" reason in metadata/conclusion
    - 50/50 probability (0.5 confidence with "uninformative" indicator)
    - Very low confidence (< 0.15) with non-UNKNOWN type
    
    Args:
        result: ReasoningResult to check (using Any type to avoid circular imports)
        
    Returns:
        True if the result indicates the reasoner was not applicable.
        
    Examples:
        >>> result = ReasoningResult(conclusion={"not_applicable": True}, ...)
        >>> _is_result_not_applicable(result)
        True
        
        >>> result = ReasoningResult(confidence=0.05, reasoning_type=ReasoningType.PROBABILISTIC, ...)
        >>> _is_result_not_applicable(result)
        True
    """
    if result is None:
        return True
    
    # Check conclusion for not_applicable flag
    if isinstance(result.conclusion, dict):
        if result.conclusion.get("not_applicable", False):
            return True
        if result.conclusion.get("applicable") is False:
            return True
        # Check for "uninformative" reason in conclusion
        if result.conclusion.get("reason") == "uninformative":
            return True
        if result.conclusion.get("uninformative", False):
            return True
    
    # Check metadata for uninformative/failed indicators
    if hasattr(result, "metadata") and isinstance(result.metadata, dict):
        if result.metadata.get("uninformative_result", False):
            return True
        if result.metadata.get("gate_check") == "failed":
            return True
        if result.metadata.get("methodology_check") == "failed":
            return True
        if result.metadata.get("reason") == "uninformative":
            return True
    
    # Check for very low confidence (<0.15) which usually indicates non-applicability
    # Note: UNKNOWN type is allowed low confidence for initialization
    if (
        hasattr(result, "confidence")
        and hasattr(result, "reasoning_type")
        and result.confidence < 0.15
        and result.reasoning_type != ReasoningType.UNKNOWN
    ):
        return True
    
    return False


# ==============================================================================
# SEQUENTIAL REASONING
# ==============================================================================


def execute_sequential_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Execute reasoning tasks sequentially with proper chain handling.
    
    Executes tasks in order, collecting results and merging reasoning chains.
    Returns the **best** result (highest confidence) instead of the last result,
    fixing the "last tool wins" bug.
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult with best conclusion and complete chain.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[task1, task2], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_sequential_reasoning(reasoner, plan, chain)
        >>> result.confidence
        0.85  # Best result, not necessarily from last task
    
    Note:
        Uses max() to select result with highest confidence, not results[-1].
        Properly merges reasoning chains from all executed tasks.
    """
    results = []

    for task in plan.tasks:
        try:
            if task.task_type in reasoner.reasoners:
                reasoner_instance = reasoner.reasoners[task.task_type]
                result = reasoner._execute_reasoner(reasoner_instance, task)
                results.append(result)

                # Properly merge reasoning chains - add ALL steps from result
                if (
                    hasattr(result, "reasoning_chain")
                    and result.reasoning_chain
                    and result.reasoning_chain.steps
                ):
                    # Skip the initial "unknown" step if it exists
                    for step in result.reasoning_chain.steps:
                        # Don't duplicate the initial UNKNOWN step
                        if (
                            step.step_type != ReasoningType.UNKNOWN
                            or step.explanation != "Reasoning process initialized"
                        ):
                            reasoning_chain.steps.append(step)
        except Exception as e:
            logger.error(f"Sequential task execution failed: {e}")

    if results:
        # Select BEST result (highest confidence), NOT last result
        # Using getattr with default 0 is intentional - results may come from
        # different reasoning engines with varying result structures
        final_result = max(results, key=lambda r: getattr(r, 'confidence', 0))
        
        # Log what we selected vs what would have been selected before
        last_result = results[-1]
        if final_result != last_result:
            logger.info(
                f"[Sequential] Selected BEST result "
                f"(confidence={final_result.confidence:.2f}) instead of LAST result "
                f"(confidence={last_result.confidence:.2f})"
            )

        # Update the provided reasoning chain with aggregated info
        reasoning_chain.final_conclusion = final_result.conclusion
        reasoning_chain.total_confidence = np.mean([r.confidence for r in results])
        reasoning_chain.reasoning_types_used.update(
            {r.reasoning_type for r in results if r.reasoning_type}
        )

        # Create a new result with the complete chain
        return ReasoningResult(
            conclusion=final_result.conclusion,
            confidence=final_result.confidence,
            reasoning_type=final_result.reasoning_type,
            reasoning_chain=reasoning_chain,
            explanation=final_result.explanation,
        )

    return reasoner._create_empty_result()


# ==============================================================================
# PARALLEL REASONING
# ==============================================================================


def execute_parallel_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Execute reasoning tasks in parallel with proper resource management.
    
    Submits tasks to ThreadPoolExecutor, waits for completion with timeout,
    and combines results. Properly cancels timed-out futures to prevent
    resource leaks.
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult with combined conclusion and complete chain.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[task1, task2, task3], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_parallel_reasoning(reasoner, plan, chain)
        >>> result.reasoning_type
        ReasoningType.HYBRID
    
    Note:
        Uses executor.submit() with future.result(timeout) for parallel execution.
        Cancels futures that timeout or fail to prevent hanging threads.
        Combines results using combine_parallel_results() helper.
    """
    futures = []

    for task in plan.tasks:
        if task.task_type in reasoner.reasoners:
            try:
                future = reasoner.executor.submit(reasoner._execute_task, task)
                futures.append((task, future))
            except Exception as e:
                logger.error(f"Failed to submit parallel task: {e}")

    results = []

    for task, future in futures:
        try:
            result = future.result(timeout=reasoner.max_reasoning_time)
            results.append(result)

            # Add steps from result to main chain
            if (
                hasattr(result, "reasoning_chain")
                and result.reasoning_chain
                and result.reasoning_chain.steps
            ):
                for step in result.reasoning_chain.steps:
                    if (
                        step.step_type != ReasoningType.UNKNOWN
                        or step.explanation != "Reasoning process initialized"
                    ):
                        reasoning_chain.steps.append(step)
        except TimeoutError:
            logger.warning(f"Parallel task {task.task_id} timed out")
            future.cancel()
        except Exception as e:
            logger.warning(f"Parallel task {task.task_id} failed: {e}")
            if not future.done():
                future.cancel()

    if results:
        conclusion = combine_parallel_results(results)
        confidence = np.mean([r.confidence for r in results])

        # Update the provided reasoning chain
        reasoning_chain.final_conclusion = conclusion
        reasoning_chain.total_confidence = confidence
        reasoning_chain.reasoning_types_used.update(
            {r.reasoning_type for r in results if r.reasoning_type}
        )

        return ReasoningResult(
            conclusion=conclusion,
            confidence=confidence,
            reasoning_type=ReasoningType.HYBRID,
            reasoning_chain=reasoning_chain,
            explanation=f"Parallel reasoning with {len(results)} tasks",
        )

    return reasoner._create_empty_result()


def combine_parallel_results(results: List[ReasoningResult]) -> Any:
    """
    Combine results from parallel execution.
    
    Strategy depends on conclusion types:
    - All dicts: Merge all key-value pairs
    - All numbers: Return mean
    - Otherwise: Return conclusion from highest-confidence result
    
    Args:
        results: List of ReasoningResult objects from parallel execution
        
    Returns:
        Combined conclusion (dict, float, or Any).
        
    Examples:
        >>> results = [ReasoningResult(conclusion={"a": 1}, confidence=0.8, ...), ...]
        >>> combine_parallel_results(results)
        {"a": 1, "b": 2}  # Merged dicts
        
        >>> results = [ReasoningResult(conclusion=5.0, confidence=0.7, ...), ...]
        >>> combine_parallel_results(results)
        4.5  # Mean of numbers
    """
    if not results:
        return None

    conclusions = [r.conclusion for r in results if r]

    if all(isinstance(c, dict) for c in conclusions):
        merged = {}
        for c in conclusions:
            if c:
                merged.update(c)
        return merged
    elif all(isinstance(c, (int, float)) for c in conclusions if c is not None):
        return np.mean([c for c in conclusions if c is not None])
    else:
        valid_results = [r for r in results if r]
        if not valid_results:
            return None
        max_idx = np.argmax([r.confidence for r in valid_results])
        return valid_results[max_idx].conclusion


# ==============================================================================
# ENSEMBLE REASONING
# ==============================================================================


def execute_ensemble_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Ensemble reasoning with weighted voting and non-applicable filtering.
    
    Executes all tasks, filters out non-applicable results (e.g., probabilistic
    on philosophical queries), computes weighted average of applicable results.
    
    Key fixes:
    - Non-applicable reasoners excluded from confidence calculations
    - Individual weights floored to prevent underflow
    - Defensive handling for zero weights (uniform fallback)
    - ToolWeightManager for learned weights
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult with ensemble conclusion and weighted confidence.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[symbolic_task, causal_task], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_ensemble_reasoning(reasoner, plan, chain)
        >>> result.reasoning_type
        ReasoningType.ENSEMBLE
    
    Note:
        Uses _is_result_not_applicable() to filter non-applicable results.
        Computes weights from confidence * type_weight * utility_weight.
        Falls back to uniform weights if all weights are zero.
    """
    results = []

    for task in plan.tasks:
        try:
            if task.task_type in reasoner.reasoners:
                result = reasoner._execute_task(task)
                results.append((task.task_type, result))

                # Add steps from result to main chain
                if (
                    hasattr(result, "reasoning_chain")
                    and result.reasoning_chain
                    and result.reasoning_chain.steps
                ):
                    for step in result.reasoning_chain.steps:
                        if (
                            step.step_type != ReasoningType.UNKNOWN
                            or step.explanation != "Reasoning process initialized"
                        ):
                            reasoning_chain.steps.append(step)
        except Exception as e:
            logger.warning(f"Ensemble task failed: {e}")

    if not results:
        return reasoner._create_empty_result()

    # Filter out non-applicable results before ensemble calculation
    applicable_results = []
    skipped_results = []
    
    for reasoning_type, result in results:
        if _is_result_not_applicable(result):
            skipped_results.append((reasoning_type, result))
            logger.info(
                f"[Ensemble] Skipping non-applicable result from "
                f"{reasoning_type.value} (confidence={result.confidence:.2f})"
            )
        else:
            applicable_results.append((reasoning_type, result))
    
    # If all results were non-applicable, fall back to the original results
    if not applicable_results:
        logger.warning(
            f"[Ensemble] All {len(results)} results were non-applicable. "
            f"Using all results as fallback."
        )
        applicable_results = results
    elif skipped_results:
        logger.info(
            f"[Ensemble] Using {len(applicable_results)} applicable results, "
            f"skipped {len(skipped_results)} non-applicable"
        )
    
    conclusions = []
    weights = []

    for reasoning_type, result in applicable_results:
        conclusions.append(result.conclusion)

        base_weight = result.confidence
        type_weight = reasoner._get_reasoning_type_weight(reasoning_type)

        if plan.tasks and plan.tasks[0].utility_context:
            execution_time_ms = getattr(result, "metadata", {}).get(
                "execution_time_ms", 100
            )
            utility_weight = reasoner._calculate_result_utility(
                result, plan.tasks[0].utility_context, execution_time_ms
            )
            raw_weight = base_weight * type_weight * utility_weight
        else:
            raw_weight = base_weight * type_weight
        
        # Floor individual weights to prevent floating-point underflow
        weights.append(max(MIN_ENSEMBLE_WEIGHT_FLOOR, raw_weight))

    # Defensive handling for zero weights
    total_weight = sum(weights)
    if total_weight <= 0:
        logger.warning("[Ensemble] All weights are zero - using uniform weights")
        logger.warning(f"[Ensemble] Weight breakdown: {list(zip([r[0].value for r in applicable_results], weights))}")
        
        # Try to log ToolWeightManager state for debugging
        try:
            from .cache import get_weight_manager
            wm = get_weight_manager()
            raw_weights = wm.get_raw_weights()
            logger.warning(f"[Ensemble] ToolWeightManager raw weights: {raw_weights}")
            
            # Log individual weight components
            for reasoning_type, result in applicable_results:
                tool_name = reasoning_type.value if reasoning_type else "unknown"
                shared = wm.get_weight(tool_name, default=1.0)
                conf = result.confidence
                logger.warning(
                    f"[Ensemble] {tool_name}: confidence={conf:.4f}, shared_weight={shared:.4f}, "
                    f"product={conf * shared:.4f}"
                )
        except Exception as e:
            logger.warning(f"[Ensemble] Could not log weight debug info: {e}")
        
        weights = [1.0 / len(applicable_results)] * len(applicable_results) if applicable_results else [1.0]
    else:
        logger.info(f"[Ensemble] Using learned weights: {dict(zip([r[0].value for r in applicable_results], weights))}")
    
    ensemble_conclusion = weighted_voting(conclusions, weights)
    ensemble_confidence = (
        np.average([r[1].confidence for r in applicable_results], weights=list(weights))
        if weights and sum(weights) > 0 and len(weights) == len(applicable_results)
        else 0.5
    )

    # Add ensemble step to the provided reasoning chain
    ensemble_step = ReasoningStep(
        "ensemble_step",
        ReasoningType.ENSEMBLE,
        plan.tasks[0].query if plan.tasks else {},
        ensemble_conclusion,
        ensemble_confidence,
        "Ensemble reasoning",
    )
    reasoning_chain.steps.append(ensemble_step)
    reasoning_chain.final_conclusion = ensemble_conclusion
    reasoning_chain.total_confidence = ensemble_confidence
    # Include all results (including skipped) in types used for tracking
    reasoning_chain.reasoning_types_used.update({r[0] for r in results})

    return ReasoningResult(
        conclusion=ensemble_conclusion,
        confidence=ensemble_confidence,
        reasoning_type=ReasoningType.ENSEMBLE,
        reasoning_chain=reasoning_chain,
        explanation=f"Ensemble of {len(applicable_results)} applicable reasoners (skipped {len(skipped_results)} non-applicable) with weighted voting",
    )


def weighted_voting(conclusions: List[Any], weights: List[float]) -> Any:
    """
    Weighted voting for ensemble conclusions.
    
    Strategy depends on conclusion types:
    - All bools: Return True if weighted sum > 0.5
    - All strings: Return string with highest total weight
    - All numbers: Return weighted average
    - Otherwise: Return conclusion with highest weight
    
    Args:
        conclusions: List of conclusions from different reasoners
        weights: List of weights (same length as conclusions)
        
    Returns:
        Combined conclusion using weighted voting.
        
    Examples:
        >>> weighted_voting([True, False, True], [0.8, 0.1, 0.1])
        True  # Weighted sum (0.9) > 0.5
        
        >>> weighted_voting(["A", "B", "A"], [0.5, 0.3, 0.2])
        "A"  # Total weight 0.7 > 0.3
        
        >>> weighted_voting([10, 20, 30], [0.5, 0.3, 0.2])
        17.0  # Weighted average
    """
    if not conclusions:
        return None

    try:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        if all(isinstance(c, bool) for c in conclusions):
            true_weight = sum(w for c, w in zip(conclusions, weights) if c)
            return true_weight > 0.5

        if all(isinstance(c, str) for c in conclusions):
            vote_weights = defaultdict(float)
            for c, w in zip(conclusions, weights):
                vote_weights[c] += w
            return max(vote_weights.items(), key=lambda x: x[1])[0]

        if all(isinstance(c, (int, float)) for c in conclusions):
            return sum(c * w for c, w in zip(conclusions, weights))

        max_idx = np.argmax(weights)
        return conclusions[max_idx]
    except Exception as e:
        logger.error(f"Weighted voting failed: {e}")
        return conclusions[0] if conclusions else None


# ==============================================================================
# ADAPTIVE REASONING
# ==============================================================================


def execute_adaptive_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Adaptive strategy selection based on input characteristics.
    
    Analyzes input characteristics (complexity, uncertainty, multimodal) and
    selects appropriate strategy:
    - High complexity → Portfolio or Ensemble
    - High uncertainty → Ensemble with Probabilistic + Causal
    - Multimodal → Multimodal reasoning
    - Otherwise → Utility-based reasoning
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult from selected strategy.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[complex_task], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_adaptive_reasoning(reasoner, plan, chain)
        # Automatically selects portfolio strategy for complex task
    
    Note:
        Adds analysis step to reasoning chain documenting characteristics.
        Falls back through strategies based on availability and context.
    """
    try:
        from .component_loader import _load_selection_components
        
        characteristics = analyze_input_characteristics(reasoner, plan.tasks[0])

        # Add analysis step
        reasoning_chain.steps.append(
            ReasoningStep(
                step_id=f"adaptive_analysis_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningType.UNKNOWN,
                input_data=plan.tasks[0].input_data,
                output_data=characteristics,
                confidence=1.0,
                explanation=f"Analyzed input characteristics: {characteristics}",
            )
        )

        if characteristics["complexity"] > 0.8:
            if plan.tasks[0].utility_context and hasattr(
                plan.tasks[0].utility_context, "mode"
            ):
                selection_components = _load_selection_components()
                ContextMode = selection_components.get("ContextMode")
                if (
                    ContextMode
                    and plan.tasks[0].utility_context.mode == ContextMode.ACCURATE
                ):
                    return execute_ensemble_reasoning(reasoner, plan, reasoning_chain)
            return execute_portfolio_reasoning(reasoner, plan, reasoning_chain)
        elif characteristics["uncertainty"] > 0.7:
            adaptive_plan = create_adaptive_plan(
                reasoner, plan.tasks[0], [ReasoningType.PROBABILISTIC, ReasoningType.CAUSAL]
            )
            return execute_ensemble_reasoning(reasoner, adaptive_plan, reasoning_chain)
        elif characteristics["multimodal"]:
            if ReasoningType.MULTIMODAL in reasoner.reasoners:
                multimodal_result = reasoner.reason_multimodal(
                    plan.tasks[0].input_data, plan.tasks[0].query
                )
                # Merge chains
                if (
                    multimodal_result.reasoning_chain
                    and multimodal_result.reasoning_chain.steps
                ):
                    for step in multimodal_result.reasoning_chain.steps:
                        if (
                            step.step_type != ReasoningType.UNKNOWN
                            or step.explanation != "Reasoning process initialized"
                        ):
                            reasoning_chain.steps.append(step)

                reasoning_chain.final_conclusion = multimodal_result.conclusion
                reasoning_chain.total_confidence = multimodal_result.confidence

                return ReasoningResult(
                    conclusion=multimodal_result.conclusion,
                    confidence=multimodal_result.confidence,
                    reasoning_type=ReasoningType.MULTIMODAL,
                    reasoning_chain=reasoning_chain,
                    explanation=multimodal_result.explanation,
                )
        else:
            return execute_utility_based_reasoning(reasoner, plan, reasoning_chain)
    except Exception as e:
        logger.error(f"Adaptive reasoning failed: {e}")
        return reasoner._create_error_result(str(e))


def analyze_input_characteristics(
    reasoner: 'UnifiedReasoner', task: ReasoningTask
) -> Dict[str, Any]:
    """
    Analyze characteristics of input data for adaptive strategy selection.
    
    Returns dict with:
    - complexity: 0.0-1.0 (based on size)
    - uncertainty: 0.0-1.0 (default 0.5, can be enhanced)
    - multimodal: bool (multiple modality types)
    - size: "small" or "large"
    - structure: "graph", "text", or "unstructured"
    
    Args:
        reasoner: UnifiedReasoner instance
        task: ReasoningTask to analyze
        
    Returns:
        Dict with characteristic scores.
        
    Examples:
        >>> task = ReasoningTask(input_data=[1, 2, ..., 2000], ...)
        >>> chars = analyze_input_characteristics(reasoner, task)
        >>> chars["size"]
        "large"
        >>> chars["complexity"]
        1.0  # 2000 items > 1000 threshold
    """
    from .component_loader import _load_reasoning_components
    
    characteristics = {
        "complexity": 0.5,
        "uncertainty": 0.5,
        "multimodal": False,
        "size": "small",
        "structure": "unstructured",
    }

    try:
        reasoning_components = _load_reasoning_components()
        ModalityType = reasoning_components.get("ModalityType")

        if isinstance(task.input_data, dict) and ModalityType:
            modality_count = sum(
                1 for k in task.input_data.keys() if isinstance(k, ModalityType)
            )
            characteristics["multimodal"] = modality_count > 1

        if isinstance(task.input_data, (list, np.ndarray)):
            characteristics["size"] = (
                "large" if len(task.input_data) > 1000 else "small"
            )
            characteristics["complexity"] = min(1.0, len(task.input_data) / 1000)

        if isinstance(task.input_data, dict) and "graph" in task.input_data:
            characteristics["structure"] = "graph"
        elif isinstance(task.input_data, str):
            characteristics["structure"] = "text"
    except Exception as e:
        logger.warning(f"Characteristic analysis failed: {e}")

    return characteristics


def create_adaptive_plan(
    reasoner: 'UnifiedReasoner',
    task: ReasoningTask,
    reasoning_types: List[ReasoningType],
) -> ReasoningPlan:
    """
    Create adaptive plan with specified reasoning types.
    
    Generates subtasks for each reasoning type, combining them into
    an ensemble plan.
    
    Args:
        reasoner: UnifiedReasoner instance
        task: Original ReasoningTask
        reasoning_types: List of ReasoningType to include
        
    Returns:
        ReasoningPlan with ENSEMBLE strategy.
        
    Examples:
        >>> task = ReasoningTask(...)
        >>> plan = create_adaptive_plan(reasoner, task, [ReasoningType.PROBABILISTIC, ReasoningType.CAUSAL])
        >>> plan.strategy
        ReasoningStrategy.ENSEMBLE
        >>> len(plan.tasks)
        2
    """
    from ..reasoning_types import ReasoningStrategy
    
    tasks = []
    for reasoning_type in reasoning_types:
        if reasoning_type in reasoner.reasoners:
            sub_task = ReasoningTask(
                task_id=f"{task.task_id}_{reasoning_type.value}",
                task_type=reasoning_type,
                input_data=task.input_data,
                query=task.query,
                constraints=task.constraints,
                utility_context=task.utility_context,
            )
            tasks.append(sub_task)

    return ReasoningPlan(
        plan_id=str(uuid.uuid4()),
        tasks=tasks,
        strategy=ReasoningStrategy.ENSEMBLE,
        dependencies={},
        estimated_time=len(tasks) * 1.0,
        estimated_cost=len(tasks) * 100,
        confidence_threshold=task.constraints.get("confidence_threshold", 0.5),
    )


# ==============================================================================
# HYBRID REASONING
# ==============================================================================


def execute_hybrid_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Custom hybrid reasoning approach with cascading strategies.
    
    Execution flow:
    1. Try Probabilistic first
    2. If confidence < 0.7, try Symbolic
    3. If "cause" in query, try Causal
    4. Use utility-based selection when utility_context provided
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult from selected approach.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[task], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_hybrid_reasoning(reasoner, plan, chain)
        >>> result.reasoning_type
        ReasoningType.HYBRID
    
    Note:
        Cascades through strategies based on confidence and query content.
        Properly merges reasoning chains from all attempted strategies.
    """
    try:
        if ReasoningType.PROBABILISTIC in reasoner.reasoners:
            prob_task = ReasoningTask(
                task_id=f"{plan.tasks[0].task_id}_prob",
                task_type=ReasoningType.PROBABILISTIC,
                input_data=plan.tasks[0].input_data,
                query=plan.tasks[0].query,
                constraints=plan.tasks[0].constraints,
                utility_context=plan.tasks[0].utility_context,
            )
            prob_result = reasoner._execute_task(prob_task)

            # Add to reasoning chain
            if (
                hasattr(prob_result, "reasoning_chain")
                and prob_result.reasoning_chain
                and prob_result.reasoning_chain.steps
            ):
                for step in prob_result.reasoning_chain.steps:
                    if (
                        step.step_type != ReasoningType.UNKNOWN
                        or step.explanation != "Reasoning process initialized"
                    ):
                        reasoning_chain.steps.append(step)

            if (
                prob_result.confidence < 0.7
                and ReasoningType.SYMBOLIC in reasoner.reasoners
            ):
                symb_task = ReasoningTask(
                    task_id=f"{plan.tasks[0].task_id}_symb",
                    task_type=ReasoningType.SYMBOLIC,
                    input_data=plan.tasks[0].input_data,
                    query=plan.tasks[0].query,
                    constraints=plan.tasks[0].constraints,
                    utility_context=plan.tasks[0].utility_context,
                )
                symb_result = reasoner._execute_task(symb_task)

                if (
                    hasattr(symb_result, "reasoning_chain")
                    and symb_result.reasoning_chain
                    and symb_result.reasoning_chain.steps
                ):
                    for step in symb_result.reasoning_chain.steps:
                        if (
                            step.step_type != ReasoningType.UNKNOWN
                            or step.explanation != "Reasoning process initialized"
                        ):
                            reasoning_chain.steps.append(step)

                if plan.tasks[0].utility_context:
                    prob_time = getattr(prob_result, "metadata", {}).get(
                        "execution_time_ms", 100
                    )
                    symb_time = getattr(symb_result, "metadata", {}).get(
                        "execution_time_ms", 100
                    )

                    prob_utility = reasoner._calculate_result_utility(
                        prob_result, plan.tasks[0].utility_context, prob_time
                    )
                    symb_utility = reasoner._calculate_result_utility(
                        symb_result, plan.tasks[0].utility_context, symb_time
                    )

                    if symb_utility > prob_utility:
                        reasoning_chain.final_conclusion = symb_result.conclusion
                        reasoning_chain.total_confidence = symb_result.confidence
                        return ReasoningResult(
                            conclusion=symb_result.conclusion,
                            confidence=symb_result.confidence,
                            reasoning_type=ReasoningType.HYBRID,
                            reasoning_chain=reasoning_chain,
                            explanation=symb_result.explanation,
                        )

            if (
                "cause" in str(plan.tasks[0].query).lower()
                and ReasoningType.CAUSAL in reasoner.reasoners
            ):
                causal_task = ReasoningTask(
                    task_id=f"{plan.tasks[0].task_id}_causal",
                    task_type=ReasoningType.CAUSAL,
                    input_data=plan.tasks[0].input_data,
                    query=plan.tasks[0].query,
                    constraints=plan.tasks[0].constraints,
                    utility_context=plan.tasks[0].utility_context,
                )
                causal_result = reasoner._execute_task(causal_task)

                if (
                    hasattr(causal_result, "reasoning_chain")
                    and causal_result.reasoning_chain
                    and causal_result.reasoning_chain.steps
                ):
                    for step in causal_result.reasoning_chain.steps:
                        if (
                            step.step_type != ReasoningType.UNKNOWN
                            or step.explanation != "Reasoning process initialized"
                        ):
                            reasoning_chain.steps.append(step)

                reasoning_chain.final_conclusion = causal_result.conclusion
                reasoning_chain.total_confidence = causal_result.confidence
                return ReasoningResult(
                    conclusion=causal_result.conclusion,
                    confidence=causal_result.confidence,
                    reasoning_type=ReasoningType.HYBRID,
                    reasoning_chain=reasoning_chain,
                    explanation=causal_result.explanation,
                )

            reasoning_chain.final_conclusion = prob_result.conclusion
            reasoning_chain.total_confidence = prob_result.confidence
            return ReasoningResult(
                conclusion=prob_result.conclusion,
                confidence=prob_result.confidence,
                reasoning_type=ReasoningType.HYBRID,
                reasoning_chain=reasoning_chain,
                explanation=prob_result.explanation,
            )
    except Exception as e:
        logger.error(f"Hybrid reasoning failed: {e}")

    return reasoner._create_empty_result()


# ==============================================================================
# HIERARCHICAL REASONING
# ==============================================================================


def execute_hierarchical_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Hierarchical reasoning with dependency-aware topological execution.
    
    Uses topological sort to order tasks by dependencies, executes in order,
    and merges dependency results into subsequent task inputs.
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks and dependencies
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult from final task in sorted order.
        
    Examples:
        >>> plan = ReasoningPlan(
        ...     tasks=[task1, task2, task3],
        ...     dependencies={"task2": ["task1"], "task3": ["task2"]},
        ...     ...
        ... )
        >>> chain = ReasoningChain()
        >>> result = execute_hierarchical_reasoning(reasoner, plan, chain)
        # Executes task1 → task2 (with task1 result) → task3 (with task2 result)
    
    Note:
        Uses topological_sort() to handle dependencies.
        Merges dependency results using merge_dependency_results().
        Returns result from final task in sorted order.
    """
    completed = {}

    try:
        sorted_tasks = topological_sort(plan.tasks, plan.dependencies)

        for task in sorted_tasks:
            deps = plan.dependencies.get(task.task_id, [])
            dep_results = [
                completed[dep_id] for dep_id in deps if dep_id in completed
            ]

            if dep_results:
                task.input_data = merge_dependency_results(
                    task.input_data, dep_results
                )

            result = reasoner._execute_task(task)
            completed[task.task_id] = result

            # Add step to reasoning chain
            if (
                hasattr(result, "reasoning_chain")
                and result.reasoning_chain
                and result.reasoning_chain.steps
            ):
                for step in result.reasoning_chain.steps:
                    if (
                        step.step_type != ReasoningType.UNKNOWN
                        or step.explanation != "Reasoning process initialized"
                    ):
                        reasoning_chain.steps.append(step)

        if completed and sorted_tasks:
            final_task_id = sorted_tasks[-1].task_id
            final_result = completed[final_task_id]

            reasoning_chain.final_conclusion = final_result.conclusion
            reasoning_chain.total_confidence = final_result.confidence

            return ReasoningResult(
                conclusion=final_result.conclusion,
                confidence=final_result.confidence,
                reasoning_type=final_result.reasoning_type,
                reasoning_chain=reasoning_chain,
                explanation=final_result.explanation,
            )
    except Exception as e:
        logger.error(f"Hierarchical reasoning failed: {e}")

    return reasoner._create_empty_result()


def topological_sort(
    tasks: List[ReasoningTask], dependencies: Dict[str, List[str]]
) -> List[ReasoningTask]:
    """
    Topological sort of tasks based on dependencies using Kahn's algorithm.
    
    This function sorts reasoning tasks in an order such that for every
    dependency from task A to task B, task A comes before task B in the
    sorted output. Uses Kahn's algorithm with O(V+E) time complexity.
    
    Args:
        tasks: List of ReasoningTask objects to sort. Each task must have
            a unique task_id attribute.
        dependencies: Dict mapping task_id → list of prerequisite task_ids.
            For example, {"t2": ["t1"]} means task t2 depends on task t1.
            Empty dict {} means no dependencies (all tasks are independent).
        
    Returns:
        List of tasks in topological order. If a cycle is detected,
        returns the original order with an error logged.
        
    Raises:
        No exceptions are raised; errors are logged and original order returned.
        
    Examples:
        >>> from vulcan.reasoning.unified.types import ReasoningTask
        >>> from vulcan.reasoning.reasoning_types import ReasoningType
        >>> tasks = [
        ...     ReasoningTask(task_id="t1", task_type=ReasoningType.PROBABILISTIC, input_data="", query={}),
        ...     ReasoningTask(task_id="t2", task_type=ReasoningType.SYMBOLIC, input_data="", query={}),
        ...     ReasoningTask(task_id="t3", task_type=ReasoningType.CAUSAL, input_data="", query={}),
        ... ]
        >>> deps = {"t2": ["t1"], "t3": ["t2"]}  # t1 → t2 → t3
        >>> sorted_tasks = topological_sort(tasks, deps)
        >>> [t.task_id for t in sorted_tasks]
        ['t1', 't2', 't3']
        
        >>> # No dependencies - order preserved
        >>> sorted_tasks = topological_sort(tasks, {})
        >>> len(sorted_tasks) == 3
        True
    
    Algorithm:
        Uses Kahn's algorithm (BFS-based topological sort):
        1. Build adjacency list and compute in-degrees
        2. Initialize queue with nodes having in-degree 0
        3. Process queue: for each node, add to result and decrement
           in-degrees of neighbors
        4. If result size != input size, a cycle exists
    
    Note:
        - Returns original order if cycle detected (with error logged).
        - Uses deque for efficient O(1) queue operations.
        - For Plan-based ordering with PlanStep objects, consider using
          Plan.optimize() from vulcan.planning instead.
        
    See Also:
        - :class:`vulcan.planning.Plan`: Plan.optimize() for PlanStep ordering
        - :func:`execute_hierarchical_reasoning`: Uses this for task ordering
    """
    try:
        # Build lookup table for O(1) task retrieval
        task_lookup: Dict[str, ReasoningTask] = {t.task_id: t for t in tasks}
        
        # Build adjacency list (parent -> children)
        adj: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        
        # Track in-degree (number of prerequisites) for each task
        in_degree: Dict[str, int] = {t.task_id: 0 for t in tasks}

        # Build the graph from dependencies
        for child, parents in dependencies.items():
            for parent in parents:
                if parent in adj and child in adj:
                    adj[parent].append(child)
                    in_degree[child] += 1

        # Initialize queue with tasks that have no prerequisites (in-degree 0)
        queue: deque = deque([t_id for t_id, deg in in_degree.items() if deg == 0])
        sorted_order: List[ReasoningTask] = []

        # Process queue using Kahn's algorithm
        while queue:
            current_id = queue.popleft()
            if current_id in task_lookup:
                sorted_order.append(task_lookup[current_id])

            # Decrement in-degree for all children
            if current_id in adj:
                for child_id in adj[current_id]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        queue.append(child_id)

        # Check for cycles
        if len(sorted_order) == len(tasks):
            return sorted_order
        else:
            logger.error(
                f"Cycle detected in task dependencies, cannot sort. "
                f"Got {len(sorted_order)} sorted tasks from {len(tasks)} total."
            )
            return tasks
            
    except Exception as e:
        logger.error(f"Topological sort failed: {e}")
        return tasks


def topological_sort_using_plan(
    tasks: List[ReasoningTask],
    dependencies: Dict[str, List[str]],
) -> List[ReasoningTask]:
    """
    Topological sort using Plan.optimize() from the planning module.
    
    This is an alternative implementation that converts ReasoningTasks to
    PlanSteps, uses Plan.optimize() for sorting, and converts back.
    
    Args:
        tasks: List of ReasoningTask objects to sort.
        dependencies: Dict mapping task_id → list of prerequisite task_ids.
        
    Returns:
        List of tasks in topological order.
        
    Note:
        This function is provided for compatibility with the planning module.
        For most use cases, the standard topological_sort() function is
        more efficient as it avoids the conversion overhead.
        
    See Also:
        - :func:`topological_sort`: Direct implementation without conversion
    """
    try:
        # Create Plan object
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal="topological_sort",
            context={},
        )
        
        # Convert tasks to PlanSteps
        task_lookup: Dict[str, ReasoningTask] = {t.task_id: t for t in tasks}
        
        for task in tasks:
            step = PlanStep(
                step_id=task.task_id,
                action=str(task.task_type.value) if task.task_type else "unknown",
                resources={"compute": 1.0},
                duration=1.0,
                probability=1.0,
                dependencies=dependencies.get(task.task_id, []),
            )
            plan.add_step(step)
        
        # Use Plan.optimize() for topological sort
        plan.optimize()
        
        # Convert back to ReasoningTasks in sorted order
        sorted_tasks: List[ReasoningTask] = []
        for step in plan.steps:
            if step.step_id in task_lookup:
                sorted_tasks.append(task_lookup[step.step_id])
        
        return sorted_tasks
        
    except Exception as e:
        logger.error(f"Plan-based topological sort failed: {e}")
        # Fallback to direct implementation
        return topological_sort(tasks, dependencies)


def merge_dependency_results(
    original_input: Any, dep_results: List[ReasoningResult]
) -> Any:
    """
    Merge results from dependencies into input.
    
    Creates a dict with:
    - "original": Original input data
    - "dependencies": List of conclusions from dependency results
    - "dep_confidence": Mean confidence of dependencies
    
    Args:
        original_input: Original task input data
        dep_results: List of ReasoningResult from dependency tasks
        
    Returns:
        Merged input dict.
        
    Examples:
        >>> original = {"query": "What is X?"}
        >>> deps = [ReasoningResult(conclusion="X is Y", confidence=0.9, ...)]
        >>> merged = merge_dependency_results(original, deps)
        >>> merged["dependencies"]
        ["X is Y"]
        >>> merged["dep_confidence"]
        0.9
    """
    if not dep_results:
        return original_input

    merged = {
        "original": original_input,
        "dependencies": [r.conclusion for r in dep_results],
        "dep_confidence": np.mean([r.confidence for r in dep_results]),
    }

    return merged


# ==============================================================================
# PORTFOLIO REASONING
# ==============================================================================


def execute_portfolio_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Execute reasoning using portfolio strategy with PortfolioExecutor.
    
    Falls back to ensemble if PortfolioExecutor not available.
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult from portfolio execution.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[task1, task2], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_portfolio_reasoning(reasoner, plan, chain)
        # Uses PortfolioExecutor with ExecutionMonitor
    
    Note:
        Converts ExecutionResult to ReasoningResult.
        Falls back to ensemble if portfolio executor unavailable.
    """
    from .component_loader import _load_selection_components
    
    if not reasoner.portfolio_executor:
        logger.warning("Portfolio executor not available, falling back to ensemble")
        return execute_ensemble_reasoning(reasoner, plan, reasoning_chain)

    try:
        if not plan.selected_tools:
            plan.selected_tools = [task.task_type.value for task in plan.tasks]

        ExecutionStrategy = reasoner._selection_components.get("ExecutionStrategy")
        if ExecutionStrategy:
            exec_strategy = (
                plan.execution_strategy or ExecutionStrategy.SEQUENTIAL_REFINEMENT
            )
        else:
            exec_strategy = None

        ExecutionMonitor = reasoner._selection_components.get("ExecutionMonitor")
        if ExecutionMonitor:
            monitor = ExecutionMonitor(
                time_budget_ms=plan.tasks[0].constraints.get(
                    "time_budget_ms", 5000
                ),
                energy_budget_mj=plan.tasks[0].constraints.get(
                    "energy_budget_mj", 1000
                ),
                min_confidence=plan.confidence_threshold,
            )
        else:
            monitor = None

        exec_result = reasoner.portfolio_executor.execute(
            strategy=exec_strategy,
            tool_names=plan.selected_tools,
            problem=plan.tasks[0].input_data,
            constraints=plan.tasks[0].constraints,
            monitor=monitor,
        )

        if (
            exec_result
            and hasattr(exec_result, "primary_result")
            and exec_result.primary_result
        ):
            result = reasoner._convert_execution_to_reasoning_result(exec_result)
            if result:
                result.reasoning_chain = reasoning_chain
                return result
    except Exception as e:
        logger.error(f"Portfolio reasoning failed: {e}")

    return reasoner._create_empty_result()


# ==============================================================================
# UTILITY-BASED REASONING
# ==============================================================================


def execute_utility_based_reasoning(
    reasoner: 'UnifiedReasoner',
    plan: ReasoningPlan,
    reasoning_chain: ReasoningChain,
) -> ReasoningResult:
    """
    Execute reasoning optimized for utility instead of confidence.
    
    For single task, executes directly. For multiple tasks, delegates to
    ensemble reasoning.
    
    Args:
        reasoner: UnifiedReasoner instance for task execution
        plan: ReasoningPlan with tasks to execute
        reasoning_chain: ReasoningChain to accumulate steps
        
    Returns:
        ReasoningResult from utility-optimized execution.
        
    Examples:
        >>> plan = ReasoningPlan(tasks=[single_task], ...)
        >>> chain = ReasoningChain()
        >>> result = execute_utility_based_reasoning(reasoner, plan, chain)
        # Executes single task directly
    
    Note:
        Falls back to ensemble for multiple tasks.
        Utility calculations happen in ensemble weight computation.
    """
    try:
        if len(plan.tasks) == 1:
            result = reasoner._execute_task(plan.tasks[0])
            result.reasoning_chain = reasoning_chain
            return result
        else:
            return execute_ensemble_reasoning(reasoner, plan, reasoning_chain)
    except Exception as e:
        logger.error(f"Utility-based reasoning failed: {e}")
        return reasoner._create_error_result(str(e))
