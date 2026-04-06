"""
Strategy planning and plan creation for unified reasoning orchestration.

Creates optimized execution plans for reasoning tasks, honoring the
Single Authority Pattern where ToolSelector is the authority for
tool selection and Router provides hints (not commands).

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .types import ReasoningPlan, ReasoningTask
from ..reasoning_types import ReasoningStrategy, ReasoningType

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

    tasks = []
    dependencies = {}

    try:
        if (
            strategy == ReasoningStrategy.UTILITY_BASED
            and reasoner.utility_model
            and reasoner.cost_model
        ):
            tasks = _plan_utility_based(reasoner, task, router_hints)

        elif strategy == ReasoningStrategy.PORTFOLIO:
            tasks = _plan_portfolio(reasoner, task)

        elif strategy == ReasoningStrategy.ENSEMBLE:
            tasks = _plan_ensemble(reasoner, task, router_hints)

        elif strategy == ReasoningStrategy.HIERARCHICAL:
            tasks, dependencies = _plan_hierarchical(reasoner, task)

        else:
            tasks = [task]
    except Exception as e:
        logger.error(f"Plan creation failed: {e}")
        tasks = [task]

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


def _plan_utility_based(
    reasoner: Any,
    task: ReasoningTask,
    router_hints: Optional[Dict[str, float]],
) -> List[ReasoningTask]:
    """Create tasks for utility-based strategy."""
    import numpy as np

    available_reasoners = list(reasoner.reasoners.keys())
    best_utility = -float("inf")
    best_tasks = []

    for reasoner_type in available_reasoners:
        estimated_quality = 0.7
        features = (
            task.features if task.features is not None else np.zeros(10)
        )
        cost_pred = reasoner.cost_model.predict_cost(
            str(reasoner_type), features
        )
        estimated_time = cost_pred["time_ms"]["mean"]
        estimated_energy = cost_pred["energy_mj"]["mean"]

        utility = reasoner.utility_model.compute_utility(
            quality=estimated_quality,
            time=estimated_time,
            energy=estimated_energy,
            risk=0.2,
            context=task.utility_context,
        )

        if router_hints and str(reasoner_type.value).lower() in router_hints:
            hint_weight = router_hints[str(reasoner_type.value).lower()]
            utility_boost = hint_weight * 0.2
            utility += utility_boost

        if utility > best_utility:
            best_utility = utility
            best_tasks = [
                ReasoningTask(
                    task_id=f"{task.task_id}_{reasoner_type.value}",
                    task_type=reasoner_type,
                    input_data=task.input_data,
                    query=task.query,
                    constraints=task.constraints,
                    utility_context=task.utility_context,
                )
            ]

    return best_tasks


def _plan_portfolio(
    reasoner: Any, task: ReasoningTask
) -> List[ReasoningTask]:
    """Create tasks for portfolio strategy."""
    from .strategy_classification import select_portfolio_reasoners
    portfolio_types = select_portfolio_reasoners(reasoner, task)

    tasks = []
    for reasoning_type in portfolio_types:
        sub_task = ReasoningTask(
            task_id=f"{task.task_id}_{reasoning_type.value}",
            task_type=reasoning_type,
            input_data=task.input_data,
            query=task.query,
            constraints=task.constraints,
            utility_context=task.utility_context,
        )
        tasks.append(sub_task)
    return tasks


def _plan_ensemble(
    reasoner: Any,
    task: ReasoningTask,
    router_hints: Optional[Dict[str, float]],
) -> List[ReasoningTask]:
    """Create tasks for ensemble strategy."""
    tools_to_use = []

    # OPTION A: Use ToolSelector
    if reasoner.tool_selector:
        try:
            logger.info(
                "[Ensemble] Using ToolSelector for ensemble tool selection"
            )
            SelectionRequest = reasoner._selection_components.get(
                "SelectionRequest"
            )
            SelectionMode = reasoner._selection_components.get(
                "SelectionMode"
            )

            if SelectionRequest and SelectionMode:
                selection_request = SelectionRequest(
                    problem=task.input_data,
                    features=task.features,
                    constraints={
                        "time_budget_ms": task.constraints.get(
                            "time_budget_ms", 5000
                        ),
                        "energy_budget_mj": task.constraints.get(
                            "energy_budget_mj", 1000
                        ),
                        "min_confidence": task.constraints.get(
                            "confidence_threshold", 0.5
                        ),
                        "router_hints": router_hints,
                    },
                    mode=SelectionMode.ACCURATE,
                    context=task.query,
                )

                selection_result = reasoner.tool_selector.select_and_execute(
                    selection_request
                )

                if hasattr(selection_result, 'selected_tool'):
                    primary_tool = selection_result.selected_tool
                    from .strategy_classification import (
                        map_tool_name_to_reasoning_type,
                    )
                    reasoning_type = map_tool_name_to_reasoning_type(
                        primary_tool
                    )
                    if (
                        reasoning_type
                        and reasoning_type in reasoner.reasoners
                    ):
                        tools_to_use.append(reasoning_type)

                if hasattr(selection_result, 'alternative_tools'):
                    for alt_tool in selection_result.alternative_tools[:2]:
                        from .strategy_classification import (
                            map_tool_name_to_reasoning_type,
                        )
                        reasoning_type = map_tool_name_to_reasoning_type(
                            alt_tool
                        )
                        if (
                            reasoning_type
                            and reasoning_type in reasoner.reasoners
                            and reasoning_type not in tools_to_use
                        ):
                            tools_to_use.append(reasoning_type)

        except Exception as e:
            logger.warning(
                f"[Ensemble] ToolSelector invocation failed: {e}"
            )

    # OPTION B: Router hints fallback
    if not tools_to_use and router_hints:
        logger.info(
            "[Ensemble] ToolSelector unavailable, using router hints"
        )
        sorted_hints = sorted(
            router_hints.items(), key=lambda x: x[1], reverse=True
        )
        for tool_name, confidence in sorted_hints[:3]:
            if confidence >= 0.3:
                try:
                    from .strategy_classification import (
                        map_tool_name_to_reasoning_type,
                    )
                    reasoning_type = map_tool_name_to_reasoning_type(
                        tool_name
                    )
                    if (
                        reasoning_type
                        and reasoning_type in reasoner.reasoners
                    ):
                        tools_to_use.append(reasoning_type)
                except Exception as e:
                    logger.warning(
                        f"[Ensemble] Failed to map tool '{tool_name}': {e}"
                    )

    # OPTION C: Default fallback
    if not tools_to_use:
        logger.info("[Ensemble] No tools selected, using defaults")
        from .orchestrator_types import DEFAULT_ENSEMBLE_TOOLS
        tools_to_use = [
            rt for rt in DEFAULT_ENSEMBLE_TOOLS
            if rt in reasoner.reasoners
        ]

    tasks = []
    for reasoning_type in tools_to_use:
        sub_task = ReasoningTask(
            task_id=f"{task.task_id}_{reasoning_type.value}",
            task_type=reasoning_type,
            input_data=task.input_data,
            query=task.query,
            constraints=task.constraints,
            utility_context=task.utility_context,
        )
        tasks.append(sub_task)

    logger.info(
        f"[Ensemble] Created {len(tasks)} tasks for reasoning types: "
        f"{[t.task_type.value for t in tasks]}"
    )
    return tasks


def _plan_hierarchical(
    reasoner: Any, task: ReasoningTask
) -> tuple:
    """Create tasks for hierarchical strategy. Returns (tasks, deps)."""
    tasks = []
    dependencies = {}

    if ReasoningType.PROBABILISTIC in reasoner.reasoners:
        basic_task = ReasoningTask(
            task_id=f"{task.task_id}_basic",
            task_type=ReasoningType.PROBABILISTIC,
            input_data=task.input_data,
            query=task.query,
            constraints=task.constraints,
            utility_context=task.utility_context,
        )
        tasks.append(basic_task)

    advanced_task = ReasoningTask(
        task_id=f"{task.task_id}_advanced",
        task_type=task.task_type,
        input_data=task.input_data,
        query=task.query,
        constraints=task.constraints,
        utility_context=task.utility_context,
    )
    tasks.append(advanced_task)

    if len(tasks) > 1:
        dependencies[advanced_task.task_id] = [tasks[0].task_id]

    return tasks, dependencies
