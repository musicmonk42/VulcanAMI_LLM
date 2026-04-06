"""
Helper functions for strategy planning.

Contains plan creation helpers for utility-based, portfolio, ensemble,
and hierarchical strategies.

Extracted from strategy_planning.py for modularity.

Author: VulcanAMI Team
"""

import logging
from typing import Any, Dict, List, Optional

from .types import ReasoningTask
from ..reasoning_types import ReasoningType

logger = logging.getLogger(__name__)


def plan_utility_based(
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
        features = (
            task.features if task.features is not None else np.zeros(10)
        )
        cost_pred = reasoner.cost_model.predict_cost(
            str(reasoner_type), features
        )
        utility = reasoner.utility_model.compute_utility(
            quality=0.7,
            time=cost_pred["time_ms"]["mean"],
            energy=cost_pred["energy_mj"]["mean"],
            risk=0.2,
            context=task.utility_context,
        )
        if router_hints and str(reasoner_type.value).lower() in router_hints:
            utility += router_hints[str(reasoner_type.value).lower()] * 0.2

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


def plan_portfolio(
    reasoner: Any, task: ReasoningTask
) -> List[ReasoningTask]:
    """Create tasks for portfolio strategy."""
    from .strategy_classification import select_portfolio_reasoners
    portfolio_types = select_portfolio_reasoners(reasoner, task)
    return [
        ReasoningTask(
            task_id=f"{task.task_id}_{rt.value}",
            task_type=rt,
            input_data=task.input_data,
            query=task.query,
            constraints=task.constraints,
            utility_context=task.utility_context,
        )
        for rt in portfolio_types
    ]


def plan_ensemble(
    reasoner: Any,
    task: ReasoningTask,
    router_hints: Optional[Dict[str, float]],
) -> List[ReasoningTask]:
    """Create tasks for ensemble strategy."""
    from .strategy_classification import map_tool_name_to_reasoning_type

    tools_to_use: List[ReasoningType] = []

    # OPTION A: Use ToolSelector
    if reasoner.tool_selector:
        try:
            logger.info("[Ensemble] Using ToolSelector for tool selection")
            SelReq = reasoner._selection_components.get("SelectionRequest")
            SelMode = reasoner._selection_components.get("SelectionMode")
            if SelReq and SelMode:
                sel_result = reasoner.tool_selector.select_and_execute(
                    SelReq(
                        problem=task.input_data, features=task.features,
                        constraints={
                            "time_budget_ms": task.constraints.get("time_budget_ms", 5000),
                            "energy_budget_mj": task.constraints.get("energy_budget_mj", 1000),
                            "min_confidence": task.constraints.get("confidence_threshold", 0.5),
                            "router_hints": router_hints,
                        },
                        mode=SelMode.ACCURATE, context=task.query,
                    )
                )
                # Collect primary + alternative tools
                candidate_names = []
                if hasattr(sel_result, 'selected_tool'):
                    candidate_names.append(sel_result.selected_tool)
                if hasattr(sel_result, 'alternative_tools'):
                    candidate_names.extend(sel_result.alternative_tools[:2])
                for name in candidate_names:
                    rt = map_tool_name_to_reasoning_type(name)
                    if rt and rt in reasoner.reasoners and rt not in tools_to_use:
                        tools_to_use.append(rt)
        except Exception as e:
            logger.warning(f"[Ensemble] ToolSelector invocation failed: {e}")

    # OPTION B: Router hints fallback
    if not tools_to_use and router_hints:
        logger.info("[Ensemble] ToolSelector unavailable, using router hints")
        sorted_hints = sorted(router_hints.items(), key=lambda x: x[1], reverse=True)
        for tool_name, confidence in sorted_hints[:3]:
            if confidence >= 0.3:
                try:
                    rt = map_tool_name_to_reasoning_type(tool_name)
                    if rt and rt in reasoner.reasoners:
                        tools_to_use.append(rt)
                except Exception as e:
                    logger.warning(f"[Ensemble] Failed to map tool '{tool_name}': {e}")

    # OPTION C: Default fallback
    if not tools_to_use:
        logger.info("[Ensemble] No tools selected, using defaults")
        from .orchestrator_types import DEFAULT_ENSEMBLE_TOOLS
        tools_to_use = [rt for rt in DEFAULT_ENSEMBLE_TOOLS if rt in reasoner.reasoners]

    tasks = [
        ReasoningTask(
            task_id=f"{task.task_id}_{rt.value}", task_type=rt,
            input_data=task.input_data, query=task.query,
            constraints=task.constraints, utility_context=task.utility_context,
        )
        for rt in tools_to_use
    ]
    logger.info(
        f"[Ensemble] Created {len(tasks)} tasks: "
        f"{[t.task_type.value for t in tasks]}"
    )
    return tasks


def plan_hierarchical(
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
