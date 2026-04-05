"""
Plan Estimation Utilities

Standalone estimation functions extracted from orchestrator.py.
These compute time and cost estimates for reasoning task plans,
either via the Plan class or via legacy per-task cost model lookups.

All functions accept their dependencies as explicit parameters
rather than relying on UnifiedReasoner instance state.

Author: VulcanAMI Team
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .types import ReasoningTask

from vulcan.planning import Plan, PlanStep

logger = logging.getLogger(__name__)


def reasoning_task_to_plan_step(
    task: ReasoningTask,
    step_index: int,
    cost_model: Optional[Any] = None,
) -> PlanStep:
    """
    Convert a ReasoningTask to a PlanStep for use with the Plan class.

    This adapter enables using the existing Plan class's optimize(), total_cost,
    and expected_duration properties with ReasoningTask objects.

    Args:
        task: ReasoningTask to convert. Must have valid task_id and task_type.
        step_index: Index for generating step_id (unused, task_id preferred).
        cost_model: Optional StochasticCostModel for duration prediction.

    Returns:
        PlanStep representation of the task with estimated resources and duration.

    Examples:
        >>> step = reasoning_task_to_plan_step(task, 0)
        >>> step.step_id == task.task_id
        True
    """
    # Estimate resources from task constraints or use defaults
    resources: Dict[str, float] = {}
    if task.constraints:
        if "cpu" in task.constraints:
            resources["cpu"] = float(task.constraints["cpu"])
        if "memory" in task.constraints:
            resources["memory"] = float(task.constraints["memory"])
        if "energy_budget_mj" in task.constraints:
            resources["energy"] = float(task.constraints["energy_budget_mj"])
    if not resources:
        resources = {"compute": 1.0}

    # Estimate duration from cost model if available
    duration = 1.0  # Default 1 second
    if cost_model is not None and task.features is not None:
        try:
            prediction = cost_model.predict_cost(
                str(task.task_type), task.features
            )
            if "time_ms" in prediction and "mean" in prediction["time_ms"]:
                duration = prediction["time_ms"]["mean"] / 1000
        except Exception as e:
            logger.debug(f"Cost model prediction failed for task {task.task_id}: {e}")

    return PlanStep(
        step_id=task.task_id,
        action=(
            getattr(task.task_type, "value", str(task.task_type))
            if task.task_type
            else "unknown"
        ),
        resources=resources,
        duration=duration,
        probability=0.8,
        dependencies=[],
    )


def estimate_plan_time_legacy(
    tasks: List[ReasoningTask],
    cost_model: Optional[Any] = None,
) -> float:
    """
    Legacy time estimation for plan execution using a cost model.

    .. deprecated::
        Use :func:`compute_plan_estimates_using_plan_class` instead.
        Retained for backward compatibility.

    Args:
        tasks: List of ReasoningTask objects to estimate time for.
        cost_model: Optional StochasticCostModel for time prediction.

    Returns:
        Total estimated time in seconds (defaults to 1 second per task).

    Examples:
        >>> time = estimate_plan_time_legacy([task1, task2, task3])
        >>> time >= 3.0
        True
    """
    total_time = 0

    for task in tasks:
        try:
            if cost_model is not None and task.features is not None:
                prediction = cost_model.predict_cost(
                    str(task.task_type), task.features
                )
                total_time += prediction["time_ms"]["mean"]
            else:
                total_time += 1000  # Default 1000ms per task
        except Exception as e:
            logger.warning(f"Time estimation failed for task {task.task_id}: {e}")
            total_time += 1000

    return total_time / 1000  # Convert to seconds


def estimate_plan_cost_legacy(
    tasks: List[ReasoningTask],
    cost_model: Optional[Any] = None,
) -> float:
    """
    Legacy cost estimation for plan execution.

    .. deprecated::
        Use :func:`compute_plan_estimates_using_plan_class` instead.
        Retained for backward compatibility.

    Args:
        tasks: List of ReasoningTask objects to estimate cost for.
        cost_model: Optional StochasticCostModel for cost prediction.

    Returns:
        Total estimated cost in arbitrary units (defaults to 100 per task).

    Examples:
        >>> cost = estimate_plan_cost_legacy([task1, task2, task3])
        >>> cost >= 300
        True
    """
    total_cost = 0

    for task in tasks:
        try:
            if cost_model is not None and task.features is not None:
                cost_estimate = cost_model.estimate_total_cost(
                    str(task.task_type), task.features
                )
                total_cost += cost_estimate
            else:
                total_cost += 100  # Default 100 units per task
        except Exception as e:
            logger.warning(f"Cost estimation failed for task {task.task_id}: {e}")
            total_cost += 100

    return total_cost


def compute_plan_estimates_using_plan_class(
    tasks: List[ReasoningTask],
    dependencies: Dict[str, List[str]],
    original_task: ReasoningTask,
    cost_model: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Use Plan class to compute optimized cost and duration estimates.

    Creates a Plan object from ReasoningTasks, uses its optimize() method
    for topological ordering, and extracts total_cost and expected_duration
    properties.

    Args:
        tasks: List of ReasoningTask objects to estimate.
        dependencies: Task dependency graph (task_id -> list of prerequisite task_ids).
        original_task: Original task for context (goal extraction).
        cost_model: Optional StochasticCostModel for duration/cost prediction.

    Returns:
        Tuple of (estimated_time, estimated_cost) where:
            - estimated_time: Total expected duration in seconds
            - estimated_cost: Total resource cost (arbitrary units)

    Note:
        Falls back to legacy estimation if Plan class fails.
    """
    try:
        goal = ""
        if original_task.query:
            goal = str(original_task.query.get("question", ""))

        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            context=original_task.query or {},
        )

        for i, task in enumerate(tasks):
            step = reasoning_task_to_plan_step(task, i, cost_model=cost_model)
            step.dependencies = dependencies.get(task.task_id, [])
            plan.add_step(step)

        plan.optimize()

        return (plan.expected_duration, plan.total_cost)

    except Exception as e:
        logger.warning(f"Plan class estimation failed, falling back to legacy: {e}")
        estimated_time = estimate_plan_time_legacy(tasks, cost_model=cost_model)
        estimated_cost = estimate_plan_cost_legacy(tasks, cost_model=cost_model)
        return (estimated_time, estimated_cost)


def estimate_energy(time_ms: float) -> float:
    """
    Simple model to estimate energy cost from execution time.

    Args:
        time_ms: Execution time in milliseconds.

    Returns:
        Estimated energy cost in millijoules.
    """
    return time_ms * 0.01
