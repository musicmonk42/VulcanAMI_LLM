"""
Strategy planning and plan creation for unified reasoning orchestration.

Creates optimized execution plans for reasoning tasks, honoring the
Single Authority Pattern where ToolSelector is the authority for
tool selection and Router provides hints (not commands).

Extracted from orchestrator.py for modularity.
Sub-module: strategy_planning_helpers.

Author: VulcanAMI Team
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .types import ReasoningPlan, ReasoningTask
from ..reasoning_types import ReasoningStrategy

logger = logging.getLogger(__name__)


def create_optimized_plan(
    reasoner: Any,
    task: ReasoningTask,
    strategy: ReasoningStrategy,
    router_hints: Optional[Dict[str, float]] = None,
    pre_selected_tools: Optional[List[str]] = None,
    skip_tool_selection: bool = False,
) -> ReasoningPlan:
    """
    Create execution plan optimized for utility.

    Implements the Single Authority Pattern: ToolSelector is THE AUTHORITY
    for tool selection. Router provides HINTS (suggestions with weights).

    Args:
        reasoner: UnifiedReasoner instance.
        task: The reasoning task to plan.
        strategy: Execution strategy.
        router_hints: Optional hints from Router.
        pre_selected_tools: Tools pre-selected by ToolSelector.
        skip_tool_selection: If True, use pre_selected_tools directly.

    Returns:
        ReasoningPlan with tasks selected by ToolSelector.
    """
    cache_key = f"{task.task_type}_{strategy}"
    if cache_key in reasoner.plan_cache:
        cached_plan = reasoner.plan_cache[cache_key]
        cached_plan.tasks = [task]
        if router_hints:
            cached_plan.metadata = cached_plan.metadata or {}
            cached_plan.metadata['router_hints'] = router_hints
        if skip_tool_selection and pre_selected_tools:
            cached_plan.selected_tools = pre_selected_tools
            cached_plan.metadata = cached_plan.metadata or {}
            cached_plan.metadata['skip_tool_selection'] = True
        return cached_plan

    # Single Authority: honor pre-selected tools
    if skip_tool_selection and pre_selected_tools:
        return _create_plan_from_preselected(
            reasoner, task, strategy, pre_selected_tools, cache_key
        )

    tasks, dependencies = _build_strategy_tasks(
        reasoner, task, strategy, router_hints
    )

    from .estimation import compute_plan_estimates_using_plan_class
    estimated_time, estimated_cost = compute_plan_estimates_using_plan_class(
        tasks, dependencies, task, reasoner.cost_model
    )

    plan = ReasoningPlan(
        plan_id=str(uuid.uuid4()),
        tasks=tasks,
        strategy=strategy,
        dependencies=dependencies,
        estimated_time=estimated_time,
        estimated_cost=estimated_cost,
        confidence_threshold=task.constraints.get(
            "confidence_threshold", 0.5
        ),
        metadata=(
            {'router_hints': router_hints} if router_hints else {}
        ),
    )

    reasoner.plan_cache[cache_key] = plan
    return plan


def _build_strategy_tasks(
    reasoner: Any,
    task: ReasoningTask,
    strategy: ReasoningStrategy,
    router_hints: Optional[Dict[str, float]],
) -> tuple:
    """Build tasks and dependencies for the given strategy."""
    from .strategy_planning_helpers import (
        plan_utility_based,
        plan_portfolio,
        plan_ensemble,
        plan_hierarchical,
    )

    tasks = []
    dependencies = {}

    try:
        if (
            strategy == ReasoningStrategy.UTILITY_BASED
            and reasoner.utility_model
            and reasoner.cost_model
        ):
            tasks = plan_utility_based(reasoner, task, router_hints)

        elif strategy == ReasoningStrategy.PORTFOLIO:
            tasks = plan_portfolio(reasoner, task)

        elif strategy == ReasoningStrategy.ENSEMBLE:
            tasks = plan_ensemble(reasoner, task, router_hints)

        elif strategy == ReasoningStrategy.HIERARCHICAL:
            tasks, dependencies = plan_hierarchical(reasoner, task)

        else:
            tasks = [task]
    except Exception as e:
        logger.error(f"Plan creation failed: {e}")
        tasks = [task]

    return tasks, dependencies


def _create_plan_from_preselected(
    reasoner: Any,
    task: ReasoningTask,
    strategy: ReasoningStrategy,
    pre_selected_tools: List[str],
    cache_key: str,
) -> ReasoningPlan:
    """Create plan from pre-selected tools (Single Authority)."""
    logger.info(
        f"[SingleAuthority] Using pre-selected tools in plan: "
        f"{pre_selected_tools}"
    )
    tasks = []
    for tool_name in pre_selected_tools:
        reasoning_type = reasoner._map_tool_name_to_reasoning_type(tool_name)
        if reasoning_type and reasoning_type in reasoner.reasoners:
            sub_task = ReasoningTask(
                task_id=f"{task.task_id}_{reasoning_type.value}",
                task_type=reasoning_type,
                input_data=task.input_data,
                query=task.query,
                constraints=task.constraints,
                utility_context=task.utility_context,
            )
            tasks.append(sub_task)
        else:
            logger.warning(
                f"[SingleAuthority] Pre-selected tool '{tool_name}' "
                "not available, skipping"
            )

    if not tasks:
        logger.error(
            f"[SingleAuthority] No valid tools from pre-selection: "
            f"{pre_selected_tools}"
        )
        tasks = [task]

    dependencies = {}
    from .estimation import compute_plan_estimates_using_plan_class
    estimated_time, estimated_cost = compute_plan_estimates_using_plan_class(
        tasks, dependencies, task, reasoner.cost_model
    )

    plan = ReasoningPlan(
        plan_id=str(uuid.uuid4()),
        tasks=tasks,
        strategy=strategy,
        dependencies=dependencies,
        estimated_time=estimated_time,
        estimated_cost=estimated_cost,
        confidence_threshold=task.constraints.get(
            "confidence_threshold", 0.5
        ),
        selected_tools=pre_selected_tools,
        metadata={
            'skip_tool_selection': True,
            'pre_selected_tools': pre_selected_tools,
            'authority': 'ToolSelector',
        },
    )

    reasoner.plan_cache[cache_key] = plan
    return plan
