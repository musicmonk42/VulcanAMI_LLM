"""
Type definitions for unified reasoning module.

This module contains all dataclasses and type definitions used in the unified
reasoning system, including ReasoningTask and ReasoningPlan.

Following highest industry standards:
- Complete type annotations
- Google-style docstrings with examples
- Immutable where possible
- Clear separation of concerns

Author: VulcanAMI Team
License: Proprietary
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from ..reasoning_types import ReasoningStrategy, ReasoningType


@dataclass
class ReasoningTask:
    """
    Represents a reasoning task to be executed by the unified reasoner.
    
    A reasoning task encapsulates all information needed to execute a single
    reasoning operation, including the query, constraints, priority, and
    optional deadline.
    
    Attributes:
        task_id: Unique identifier for this task
        task_type: Type of reasoning to perform (e.g., SYMBOLIC, PROBABILISTIC)
        input_data: Raw input data for the reasoning task
        query: Structured query dictionary with keys like 'question', 'context'
        constraints: Additional constraints or requirements for reasoning
        priority: Execution priority (higher = more urgent), default 0
        deadline: Optional deadline (seconds since epoch) for task completion
        metadata: Additional metadata for tracking and debugging
        features: Optional feature vector for ML-based tool selection
        utility_context: Optional context for utility-based decision making
        
    Examples:
        >>> task = ReasoningTask(
        ...     task_id="task_001",
        ...     task_type=ReasoningType.SYMBOLIC,
        ...     input_data="Prove: If P then Q",
        ...     query={"question": "Is the theorem valid?"},
        ...     priority=5
        ... )
        >>> print(task.task_id)
        'task_001'
        
        >>> # With deadline
        >>> import time
        >>> task_with_deadline = ReasoningTask(
        ...     task_id="urgent_task",
        ...     task_type=ReasoningType.CAUSAL,
        ...     input_data="What causes X?",
        ...     query={"question": "Causal analysis"},
        ...     deadline=time.time() + 60  # 60 seconds from now
        ... )
        
    Note:
        The features attribute is used for ML-based tool selection when available.
        If not provided, the system falls back to heuristic-based selection.
    """
    
    task_id: str
    task_type: ReasoningType
    input_data: Any
    query: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Optional[npt.NDArray[np.float64]] = None
    utility_context: Optional[Any] = None


@dataclass
class ReasoningPlan:
    """
    Execution plan for a set of reasoning tasks.
    
    A reasoning plan describes how multiple reasoning tasks should be executed,
    including their dependencies, execution strategy, and resource constraints.
    
    Attributes:
        plan_id: Unique identifier for this plan
        tasks: List of reasoning tasks to execute
        strategy: Overall reasoning strategy (SEQUENTIAL, PARALLEL, ENSEMBLE)
        dependencies: DAG of task dependencies (task_id -> list of prerequisite task_ids)
        estimated_time: Estimated total execution time in seconds
        estimated_cost: Estimated computational cost (arbitrary units)
        confidence_threshold: Minimum confidence for accepting results (0.0-1.0)
        execution_strategy: Optional advanced execution strategy object
        selected_tools: Optional list of specific tools to use
        
    Examples:
        >>> plan = ReasoningPlan(
        ...     plan_id="plan_001",
        ...     tasks=[task1, task2, task3],
        ...     strategy=ReasoningStrategy.PARALLEL,
        ...     dependencies={"task2": ["task1"], "task3": ["task1", "task2"]},
        ...     estimated_time=5.0,
        ...     estimated_cost=100.0,
        ...     confidence_threshold=0.7
        ... )
        >>> print(len(plan.tasks))
        3
        
        >>> # Sequential plan with no dependencies
        >>> sequential_plan = ReasoningPlan(
        ...     plan_id="seq_plan",
        ...     tasks=[task1, task2],
        ...     strategy=ReasoningStrategy.SEQUENTIAL,
        ...     dependencies={},
        ...     estimated_time=10.0,
        ...     estimated_cost=50.0
        ... )
        
    Note:
        Dependencies form a directed acyclic graph (DAG). Cycles will cause
        execution errors. Use topological sort for proper execution order.
    """
    
    plan_id: str
    tasks: List[ReasoningTask]
    strategy: ReasoningStrategy
    dependencies: Dict[str, List[str]]
    estimated_time: float
    estimated_cost: float
    confidence_threshold: float = 0.5
    execution_strategy: Optional[Any] = None
    selected_tools: Optional[List[str]] = None
