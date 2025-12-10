"""
fallback_chain.py - Fallback chain management for problem decomposition
Part of the VULCAN-AGI system
"""

import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Status of strategy execution"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class FailureType(Enum):
    """Types of strategy failures"""

    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    INCOMPLETE = "incomplete"
    RESOURCE_EXCEEDED = "resource_exceeded"
    UNSUPPORTED = "unsupported"


class ComponentType(Enum):
    """Types of decomposition components"""

    ATOMIC = "atomic"
    COMPOSITE = "composite"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class DecompositionComponent:
    """Single component in decomposition"""

    component_id: str
    component_type: ComponentType
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 1.0
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "description": self.description,
            "dependencies": self.dependencies,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DecompositionFailure:
    """Detailed failure information"""

    problem_signature: str
    missing_component: str
    attempted_strategies: List[str] = field(default_factory=list)
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recovery_attempted: bool = False
    recoverable: bool = True
    suggested_fallbacks: List[str] = field(default_factory=list)

    def add_failure(self, strategy_name: str, reason: str):
        """Add strategy failure information"""
        self.attempted_strategies.append(strategy_name)
        self.failure_reasons[strategy_name] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "problem_signature": self.problem_signature,
            "missing_component": self.missing_component,
            "attempted_strategies": self.attempted_strategies,
            "failure_reasons": self.failure_reasons,
            "timestamp": self.timestamp,
            "recovery_attempted": self.recovery_attempted,
            "recoverable": self.recoverable,
            "suggested_fallbacks": self.suggested_fallbacks,
        }


class ExecutionPlan:
    """Decomposition execution plan"""

    def __init__(self, plan_id: Optional[str] = None):
        """
        Initialize execution plan

        Args:
            plan_id: Optional plan identifier
        """
        self.plan_id = plan_id or f"plan_{int(time.time())}_{np.random.randint(1000)}"
        self.components = []  # List of DecompositionComponent
        self.component_map = {}  # component_id -> DecompositionComponent
        self.execution_order = []  # Ordered list of component_ids
        self.confidence_scores = {}  # component_id -> confidence
        self.metadata = {}
        self.total_cost = 0.0
        self.creation_time = time.time()

        # Execution tracking
        self.execution_status = {}  # component_id -> StrategyStatus
        self.execution_results = {}  # component_id -> result

        logger.debug("ExecutionPlan %s created", self.plan_id)

    @property
    def steps(self) -> List[DecompositionComponent]:
        """
        Get components as steps for API compatibility with DecompositionPlan

        Returns:
            List of components (aliased as steps)
        """
        return self.components

    @property
    def confidence(self) -> float:
        """
        Get overall confidence for API compatibility with DecompositionPlan

        Returns:
            Overall confidence score [0, 1]
        """
        return self.overall_confidence()

    def add_components(
        self, components: List[DecompositionComponent], confidence: float = None
    ):
        """
        Add components to the plan

        Args:
            components: List of components to add
            confidence: Optional override confidence for all components
        """
        for component in components:
            # Set confidence if provided
            if confidence is not None:
                component.confidence = confidence

            # Add to structures
            self.components.append(component)
            self.component_map[component.component_id] = component
            self.confidence_scores[component.component_id] = component.confidence

            # Update total cost
            self.total_cost += component.estimated_cost

            # Initialize execution status
            self.execution_status[component.component_id] = StrategyStatus.PENDING

            logger.debug(
                "Added component %s to plan %s", component.component_id, self.plan_id
            )

    def overall_confidence(self) -> float:
        """
        Calculate overall plan confidence

        Returns:
            Overall confidence score [0, 1]
        """
        if not self.confidence_scores:
            return 0.0

        # Weighted average by component cost
        weighted_sum = 0.0
        weight_total = 0.0

        for component_id, confidence in self.confidence_scores.items():
            component = self.component_map.get(component_id)
            if component:
                weight = component.estimated_cost
                weighted_sum += confidence * weight
                weight_total += weight

        if weight_total > 0:
            return weighted_sum / weight_total
        else:
            return np.mean(list(self.confidence_scores.values()))

    def get_execution_order(self) -> List[str]:
        """
        Get execution order considering dependencies

        Returns:
            Ordered list of component IDs
        """
        if self.execution_order:
            return self.execution_order

        # Build dependency graph
        dependency_graph = defaultdict(set)
        in_degree = defaultdict(int)

        for component in self.components:
            for dep in component.dependencies:
                dependency_graph[dep].add(component.component_id)
                in_degree[component.component_id] += 1

        # Topological sort using Kahn's algorithm
        queue = deque()
        for component in self.components:
            if in_degree[component.component_id] == 0:
                queue.append(component.component_id)

        ordered = []
        while queue:
            current = queue.popleft()
            ordered.append(current)

            for neighbor in dependency_graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(ordered) != len(self.components):
            logger.warning("Dependency cycle detected in plan %s", self.plan_id)
            # Return components without dependencies first, then others
            no_deps = [c.component_id for c in self.components if not c.dependencies]
            with_deps = [c.component_id for c in self.components if c.dependencies]
            ordered = no_deps + with_deps

        self.execution_order = ordered
        return ordered

    def validate_completeness(self) -> Tuple[bool, List[str]]:
        """
        Validate plan completeness

        Returns:
            Tuple of (is_complete, list_of_issues)
        """
        issues = list(]

        # Check for empty plan
        if not self.components:
            issues.append("Plan has no components")

        # Check for missing dependencies
        all_component_ids = set(self.component_map.keys())
        for component in self.components:
            for dep in component.dependencies:
                if dep not in all_component_ids:
                    issues.append(f"Missing dependency: {dep}")

        # Check for isolated components (no dependencies and nothing depends on them)
        dependencies = set()
        dependents = set()

        for component in self.components:
            dependencies.update(component.dependencies)
            if component.dependencies:
                dependents.add(component.component_id)

        for component in self.components:
            comp_id = component.component_id
            if comp_id not in dependencies and comp_id not in dependents:
                if len(self.components) > 1:  # Only issue if multiple components
                    issues.append(f"Isolated component: {comp_id}")

        # Check for very low confidence components
        for component in self.components:
            if component.confidence < 0.2:
                issues.append(
                    f"Very low confidence component: {component.component_id}"
                )

        is_complete = len(issues) == 0

        return is_complete, issues

    def update_component_status(
        self, component_id: str, status: StrategyStatus, result: Any = None
    ):
        """Update component execution status"""
        self.execution_status[component_id] = status
        if result is not None:
            self.execution_results[component_id] = result

        logger.debug("Updated component %s status to %s", component_id, status.value)

    def get_next_executable_component(self) -> Optional[str]:
        """Get next component ready for execution"""
        execution_order = self.get_execution_order()

        for component_id in execution_order:
            if self.execution_status[component_id] != StrategyStatus.PENDING:
                continue

            component = self.component_map[component_id]

            # Check if dependencies are satisfied
            deps_satisfied = True
            for dep in component.dependencies:
                if dep not in self.execution_status:
                    deps_satisfied = False
                    break
                if self.execution_status[dep] != StrategyStatus.SUCCESS:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                return component_id

        return None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        status_counts = defaultdict(int)
        for status in self.execution_status.values():
            status_counts[status.value] += 1

        return {
            "plan_id": self.plan_id,
            "total_components": len(self.components),
            "total_cost": self.total_cost,
            "overall_confidence": self.overall_confidence(),
            "status_counts": dict(status_counts),
            "completion_rate": status_counts[StrategyStatus.SUCCESS.value]
            / max(1, len(self.components)),
            "creation_time": self.creation_time,
            "execution_time": time.time() - self.creation_time,
        }


class FallbackChain:
    """Manages strategy execution order and fallbacks"""

    def __init__(self, strategies: List[Any] = None):
        """
        Initialize fallback chain

        Args:
            strategies: Initial list of strategies
        """
        self.strategies = strategies or []
        self.strategy_costs = {}  # strategy_name -> cost estimate
        self.strategy_success_rates = defaultdict(
            lambda: 0.5
        )  # strategy_name -> success rate
        self.execution_history = deque(maxlen=1000)

        # Failure handling
        self.failure_counts = defaultdict(int)
        self.recovery_strategies = {}

        # Thread safety and size limits
        self._lock = threading.RLock()
        self.max_recovery_strategies = 100

        # Configuration
        self.max_retries = 3
        self.timeout_seconds = 60
        self.cost_weight = 0.3
        self.success_weight = 0.7

        logger.info(
            "FallbackChain initialized with %d strategies", len(self.strategies)
        )

    def execute_with_fallbacks(
        self, problem_graph
    ) -> Tuple[Optional[ExecutionPlan], DecompositionFailure]:
        """
        Execute strategies with fallback chain

        Args:
            problem_graph: Problem to decompose

        Returns:
            Tuple of (execution_plan, failure_info)
        """
        problem_signature = (
            problem_graph.get_signature()
            if hasattr(problem_graph, "get_signature")
            else str(problem_graph)
        )

        failure = DecompositionFailure(
            problem_signature=problem_signature,
            missing_component="complete_decomposition",
        )

        # Try each strategy in order
        with self._lock:
            strategies_copy = list(self.strategies)

        for strategy in strategies_copy:
            strategy_name = getattr(strategy, "name", str(strategy))

            try:
                logger.debug("Attempting strategy: %s", strategy_name)

                # Execute strategy
                result = self._execute_strategy(strategy, problem_graph)

                if result and self._is_valid_result(result):
                    # Convert to execution plan
                    plan = self._create_execution_plan(result, strategy_name)

                    # Update success rate
                    with self._lock:
                        self.strategy_success_rates[strategy_name] = (
                            0.9 * self.strategy_success_rates[strategy_name] + 0.1
                        )

                    # Record execution
                    self._record_execution(strategy_name, True)

                    return plan, None
                else:
                    failure.add_failure(strategy_name, "Invalid result")

            except Exception as e:
                # Handle strategy failure
                failure_reason = self.handle_strategy_failure(strategy, e)
                failure.add_failure(strategy_name, failure_reason)

                # Update success rate
                with self._lock:
                    self.strategy_success_rates[strategy_name] *= 0.9

                # Record execution
                self._record_execution(strategy_name, False)

        # All strategies failed
        failure.suggested_fallbacks = self._suggest_fallback_strategies(problem_graph)

        return None, failure

    def reorder_by_cost(self, problem_class: str = None):
        """
        Reorder strategies by cost-effectiveness

        Args:
            problem_class: Optional problem class for specific ordering
        """
        with self._lock:
            # Calculate cost-effectiveness for each strategy
            scores = []

            for strategy in self.strategies:
                strategy_name = getattr(strategy, "name", str(strategy))

                # Get cost and success rate
                cost = self.strategy_costs.get(strategy_name, 1.0)
                success_rate = self.strategy_success_rates[strategy_name]

                # Adjust for problem class if specified
                if problem_class:
                    success_rate = self._adjust_for_problem_class(
                        strategy_name, problem_class, success_rate
                    )

                # Calculate score (higher is better)
                score = (success_rate * self.success_weight) / (
                    cost * self.cost_weight + 0.01
                )
                scores.append((strategy, score))

            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)

            # Update strategy order
            self.strategies = [s for s, _ in scores]

            logger.info("Reordered strategies by cost-effectiveness")

    def handle_strategy_failure(self, strategy, exception: Exception) -> str:
        """
        Handle strategy execution failure

        Args:
            strategy: Failed strategy
            exception: Exception that occurred

        Returns:
            Failure reason string
        """
        strategy_name = getattr(strategy, "name", str(strategy))

        # Determine failure type
        failure_reason = self._classify_failure(exception)

        with self._lock:
            # Increment failure count
            self.failure_counts[strategy_name] += 1

            # Limit recovery strategies size
            if len(self.recovery_strategies) >= self.max_recovery_strategies:
                # Remove oldest (first inserted)
                oldest = next(iter(self.recovery_strategies))
                del self.recovery_strategies[oldest]

            # Suggest recovery strategy
            if strategy_name not in self.recovery_strategies:
                self.recovery_strategies[strategy_name] = (
                    self._determine_recovery_strategy(failure_reason)
                )

            # Check if we should remove strategy from chain
            if self.failure_counts[strategy_name] > 5:
                logger.warning(
                    "Strategy %s has failed too many times, considering removal",
                    strategy_name,
                )

                # Move to end of chain instead of removing
                if strategy in self.strategies:
                    self.strategies.remove(strategy)
                    self.strategies.append(strategy)

        # Log failure details
        logger.error("Strategy %s failed: %s", strategy_name, failure_reason)
        logger.debug("Exception details: %s", traceback.format_exc())

        return failure_reason

    def add_strategy(self, strategy, cost: float = 1.0):
        """Add strategy to chain"""
        with self._lock:
            self.strategies.append(strategy)
            strategy_name = getattr(strategy, "name", str(strategy))
            self.strategy_costs[strategy_name] = cost

            logger.debug("Added strategy %s to chain", strategy_name)

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove strategy from chain"""
        with self._lock:
            for i, strategy in enumerate(self.strategies):
                if getattr(strategy, "name", str(strategy)) == strategy_name:
                    del self.strategies[i]
                    logger.debug("Removed strategy %s from chain", strategy_name)
                    return True
            return False

    def generate_fallback_plans(self, problem_graph) -> List[ExecutionPlan]:
        """
        Generate multiple fallback plans

        Args:
            problem_graph: Problem to decompose

        Returns:
            List of fallback execution plans
        """
        plans = []

        with self._lock:
            strategies_copy = list(self.strategies)

        # Try different strategy combinations
        for i in range(min(3, len(strategies_copy))):
            # Rotate strategies
            rotated_strategies = strategies_copy[i:] + strategies_copy[:i]

            # Try with rotated order
            temp_chain = FallbackChain(rotated_strategies)
            plan, failure = temp_chain.execute_with_fallbacks(problem_graph)

            if plan:
                plans.append(plan)

        # Try simplified decomposition
        simplified_plan = self._create_simplified_plan(problem_graph)
        if simplified_plan:
            plans.append(simplified_plan)

        # Try hierarchical decomposition
        hierarchical_plan = self._create_hierarchical_plan(problem_graph)
        if hierarchical_plan:
            plans.append(hierarchical_plan)

        return plans

    def _execute_strategy(self, strategy, problem_graph) -> Any:
        """Execute single strategy with timeout"""
        # Simple execution - in production would use actual timeout mechanism
        if hasattr(strategy, "decompose"):
            return strategy.decompose(problem_graph)
        elif callable(strategy):
            return strategy(problem_graph)
        else:
            raise ValueError(f"Strategy {strategy} is not executable")

    def _is_valid_result(self, result: Any) -> bool:
        """Check if strategy result is valid"""
        if result is None:
            return False

        if isinstance(result, list):
            return len(result) > 0
        elif isinstance(result, dict):
            return len(result) > 0
        else:
            return True

    def _create_execution_plan(self, result: Any, strategy_name: str) -> ExecutionPlan:
        """Create execution plan from strategy result"""
        plan = ExecutionPlan()

        # Convert result to components
        if isinstance(result, list):
            components = []
            for i, item in enumerate(result):
                component = DecompositionComponent(
                    component_id=f"{strategy_name}_{i}",
                    component_type=ComponentType.ATOMIC,
                    description=str(item),
                    confidence=0.7,
                )
                components.append(component)
            plan.add_components(components)

        elif isinstance(result, dict):
            components = []
            for key, value in result.items():
                component = DecompositionComponent(
                    component_id=f"{strategy_name}_{key}",
                    component_type=ComponentType.COMPOSITE,
                    description=str(value),
                    confidence=0.7,
                )
                components.append(component)
            plan.add_components(components)

        plan.metadata["strategy"] = strategy_name

        return plan

    def _classify_failure(self, exception: Exception) -> str:
        """Classify type of failure from exception"""
        exception_str = str(exception).lower()

        if "timeout" in exception_str:
            return FailureType.TIMEOUT.value
        elif "memory" in exception_str:
            return FailureType.RESOURCE_EXCEEDED.value
        elif "not supported" in exception_str or "not implemented" in exception_str:
            return FailureType.UNSUPPORTED.value
        elif "invalid" in exception_str:
            return FailureType.INVALID_OUTPUT.value
        else:
            return FailureType.EXCEPTION.value

    def _determine_recovery_strategy(self, failure_reason: str) -> str:
        """Determine recovery strategy based on failure reason"""
        if failure_reason == FailureType.TIMEOUT.value:
            return "increase_timeout_or_simplify"
        elif failure_reason == FailureType.RESOURCE_EXCEEDED.value:
            return "reduce_problem_size"
        elif failure_reason == FailureType.UNSUPPORTED.value:
            return "use_alternative_strategy"
        elif failure_reason == FailureType.INVALID_OUTPUT.value:
            return "validate_and_retry"
        else:
            return "fallback_to_simple"

    def _suggest_fallback_strategies(self, problem_graph) -> List[str]:
        """Suggest alternative strategies"""
        suggestions = []

        # Based on problem characteristics
        if hasattr(problem_graph, "complexity_score"):
            if problem_graph.complexity_score > 3:
                suggestions.append("hierarchical_decomposition")
            else:
                suggestions.append("simple_subdivision")

        # Based on past failures
        with self._lock:
            failed_types = set()
            for strategy_name in self.failure_counts.keys():
                if "recursive" in strategy_name.lower():
                    failed_types.add("recursive")
                elif "iterative" in strategy_name.lower():
                    failed_types.add("iterative")

        if "recursive" not in failed_types:
            suggestions.append("recursive_decomposition")
        if "iterative" not in failed_types:
            suggestions.append("iterative_refinement")

        return suggestions

    def _record_execution(self, strategy_name: str, success: bool):
        """Record execution for history"""
        with self._lock:
            self.execution_history.append(
                {
                    "strategy": strategy_name,
                    "success": success,
                    "timestamp": time.time(),
                }
            )

    def _adjust_for_problem_class(
        self, strategy_name: str, problem_class: str, base_rate: float
    ) -> float:
        """Adjust success rate for specific problem class"""
        # Simple heuristic adjustments
        adjustments = {
            "optimization": {"gradient": 1.2, "genetic": 1.1, "brute_force": 0.8},
            "planning": {"hierarchical": 1.3, "forward": 1.1, "backward": 1.0},
            "classification": {"tree": 1.2, "linear": 0.9, "ensemble": 1.3},
        }

        if problem_class in adjustments:
            for keyword, multiplier in adjustments[problem_class].items():
                if keyword in strategy_name.lower():
                    return min(1.0, base_rate * multiplier)

        return base_rate

    def _create_simplified_plan(self, problem_graph) -> Optional[ExecutionPlan]:
        """Create simplified fallback plan"""
        signature = (
            problem_graph.get_signature()[:8]
            if hasattr(problem_graph, "get_signature")
            else "unknown"
        )
        plan = ExecutionPlan(plan_id=f"simplified_{signature}")

        # Create single monolithic component
        component = DecompositionComponent(
            component_id="simplified_solution",
            component_type=ComponentType.ATOMIC,
            description="Direct solution attempt",
            confidence=0.4,
            estimated_cost=1.0,
        )

        plan.add_components([component])
        plan.metadata["type"] = "simplified"

        return plan

    def _create_hierarchical_plan(self, problem_graph) -> Optional[ExecutionPlan]:
        """Create hierarchical fallback plan"""
        signature = (
            problem_graph.get_signature()[:8]
            if hasattr(problem_graph, "get_signature")
            else "unknown"
        )
        plan = ExecutionPlan(plan_id=f"hierarchical_{signature}")

        # Create two-level hierarchy
        components = []

        # Top level
        top_component = DecompositionComponent(
            component_id="top_level",
            component_type=ComponentType.COMPOSITE,
            description="High-level decomposition",
            confidence=0.5,
            estimated_cost=0.5,
        )
        components.append(top_component)

        # Sub-components
        for i in range(3):
            sub_component = DecompositionComponent(
                component_id=f"sub_level_{i}",
                component_type=ComponentType.ATOMIC,
                description=f"Sub-task {i}",
                dependencies=["top_level"],
                confidence=0.5,
                estimated_cost=0.3,
            )
            components.append(sub_component)

        plan.add_components(components)
        plan.metadata["type"] = "hierarchical"

        return plan

    def get_statistics(self) -> Dict[str, Any]:
        """Get chain execution statistics"""
        with self._lock:
            total_executions = len(self.execution_history)
            successful_executions = sum(
                1 for e in self.execution_history if e["success"]
            )

            strategy_stats = {}
            for strategy in self.strategies:
                strategy_name = getattr(strategy, "name", str(strategy))
                strategy_stats[strategy_name] = {
                    "success_rate": self.strategy_success_rates[strategy_name],
                    "failure_count": self.failure_counts[strategy_name],
                    "cost": self.strategy_costs.get(strategy_name, 1.0),
                }

            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "overall_success_rate": successful_executions
                / max(1, total_executions),
                "strategy_count": len(self.strategies),
                "strategy_stats": strategy_stats,
            }
