"""
adaptive_thresholds.py - Adaptive threshold management for problem decomposer
Part of the VULCAN-AGI system
"""

import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ThresholdType(Enum):
    """Types of thresholds"""

    CONFIDENCE = "confidence"
    COMPLEXITY = "complexity"
    PERFORMANCE = "performance"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


class StrategyStatus(Enum):
    """Status of strategy execution"""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


@dataclass
class ThresholdConfig:
    """Configuration for a threshold"""

    name: str
    value: float
    min_value: float
    max_value: float
    adjustment_rate: float = 0.05
    auto_adjust: bool = True

    def adjust(self, factor: float, direction: str = "up"):
        """Adjust threshold value"""
        if direction == "up":
            new_value = self.value * (1 + factor)
        else:
            new_value = self.value * (1 - factor)

        self.value = max(self.min_value, min(self.max_value, new_value))


@dataclass
class PerformanceRecord:
    """Single performance record"""

    problem_signature: str
    timestamp: float
    success: bool
    strategy_used: Optional[str] = None
    execution_time: float = 0.0
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyProfile:
    """Profile for a decomposition strategy"""

    name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    success_rate: float = 0.5
    cost_estimates: Dict[str, float] = field(default_factory=dict)

    def update(self, latency: float, success: bool):
        """Update profile with new execution"""
        self.total_attempts += 1
        self.total_latency += latency

        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1

        # Update averages
        self.avg_latency = self.total_latency / self.total_attempts

        if self.total_attempts > 0:
            self.success_rate = self.successful_attempts / self.total_attempts


class AdaptiveThresholds:
    """Self-adjusting confidence thresholds"""

    def __init__(self, initial_values: Dict[str, float] = None):
        """
        Initialize adaptive thresholds

        Args:
            initial_values: Initial threshold values
        """
        # Thread safety
        self._lock = threading.RLock()

        # Default initial values
        defaults = {
            ThresholdType.CONFIDENCE.value: 0.7,
            ThresholdType.COMPLEXITY.value: 3.0,
            ThresholdType.PERFORMANCE.value: 0.6,
            ThresholdType.TIMEOUT.value: 60.0,
            ThresholdType.RESOURCE.value: 0.8,
        }

        # Merge with provided values
        if initial_values:
            defaults.update(initial_values)

        # Create threshold configs
        self.thresholds = {}
        for name, value in defaults.items():
            self.thresholds[name] = ThresholdConfig(
                name=name,
                value=value,
                min_value=self._get_min_value(name),
                max_value=self._get_max_value(name),
                adjustment_rate=0.05,
            )

        # Adjustment history
        self.adjustment_history = deque(maxlen=100)

        # Performance window for auto-calibration
        self.performance_window = deque(maxlen=50)

        # Statistics
        self.total_adjustments = 0
        self.auto_calibrations = 0

        logger.info(
            "AdaptiveThresholds initialized with %d thresholds", len(self.thresholds)
        )

    def get_current(self, threshold_type: str = None) -> Any:
        """
        Get current threshold value(s)

        Args:
            threshold_type: Specific threshold type or None for all

        Returns:
            Threshold value or dict of all values
        """
        with self._lock:
            if threshold_type:
                if threshold_type in self.thresholds:
                    return self.thresholds[threshold_type].value
                else:
                    logger.warning("Unknown threshold type: %s", threshold_type)
                    return None
            else:
                return {name: config.value for name, config in self.thresholds.items()}

    def adjust_up(self, factor: float = 0.05, threshold_type: str = None):
        """
        Adjust threshold(s) up

        Args:
            factor: Adjustment factor
            threshold_type: Specific threshold or None for all
        """
        with self._lock:
            if threshold_type:
                if threshold_type in self.thresholds:
                    self._adjust_threshold(threshold_type, factor, "up")
            else:
                for threshold_name in self.thresholds:
                    self._adjust_threshold(threshold_name, factor, "up")

    def adjust_down(self, factor: float = 0.05, threshold_type: str = None):
        """
        Adjust threshold(s) down

        Args:
            factor: Adjustment factor
            threshold_type: Specific threshold or None for all
        """
        with self._lock:
            if threshold_type:
                if threshold_type in self.thresholds:
                    self._adjust_threshold(threshold_type, factor, "down")
            else:
                for threshold_name in self.thresholds:
                    self._adjust_threshold(threshold_name, factor, "down")

    def auto_calibrate(self, recent_performance: List[Dict[str, Any]]):
        """
        Auto-calibrate thresholds based on recent performance

        Args:
            recent_performance: List of recent performance records
        """
        with self._lock:
            if not recent_performance:
                return

            # Calculate performance metrics
            success_count = sum(
                1 for p in recent_performance if p.get("success", False)
            )
            total_count = len(recent_performance)
            success_rate = success_count / total_count

            # Calculate average execution time
            exec_times = [p.get("execution_time", 0) for p in recent_performance]
            avg_exec_time = np.mean(exec_times) if exec_times else 0

            # Adjust confidence threshold based on success rate
            if success_rate < 0.4:
                # Too many failures - lower confidence threshold
                self.adjust_down(0.1, ThresholdType.CONFIDENCE.value)
                logger.debug("Lowered confidence threshold due to low success rate")
            elif success_rate > 0.9:
                # Very high success - can increase confidence threshold
                self.adjust_up(0.05, ThresholdType.CONFIDENCE.value)
                logger.debug("Raised confidence threshold due to high success rate")

            # Adjust timeout based on execution times
            if avg_exec_time > 0:
                current_timeout = self.get_current(ThresholdType.TIMEOUT.value)
                if avg_exec_time > current_timeout * 0.8:
                    # Close to timeout - increase it
                    self.adjust_up(0.2, ThresholdType.TIMEOUT.value)
                    logger.debug("Increased timeout threshold")
                elif avg_exec_time < current_timeout * 0.3:
                    # Much faster than timeout - can decrease
                    self.adjust_down(0.1, ThresholdType.TIMEOUT.value)
                    logger.debug("Decreased timeout threshold")

            # Adjust complexity threshold based on problem characteristics
            complexities = [
                p.get("complexity", 0) for p in recent_performance if "complexity" in p
            ]
            if complexities:
                avg_complexity = np.mean(complexities)
                current_complexity = self.get_current(ThresholdType.COMPLEXITY.value)

                if avg_complexity > current_complexity:
                    # Problems are more complex - adjust threshold
                    self.adjust_up(0.1, ThresholdType.COMPLEXITY.value)
                elif avg_complexity < current_complexity * 0.7:
                    # Problems are simpler - can lower threshold
                    self.adjust_down(0.05, ThresholdType.COMPLEXITY.value)

            self.auto_calibrations += 1

            # Track calibration
            self.adjustment_history.append(
                {
                    "type": "auto_calibration",
                    "timestamp": time.time(),
                    "success_rate": success_rate,
                    "avg_exec_time": avg_exec_time,
                    "thresholds": self.get_current(),
                }
            )

            logger.info(
                "Auto-calibrated thresholds based on %d performance records",
                total_count,
            )

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold (convenience method)"""
        return self.get_current(ThresholdType.CONFIDENCE.value)

    def update_from_outcome(
        self, complexity: float, success: bool, execution_time: float
    ):
        """
        Update thresholds based on execution outcome

        Args:
            complexity: Problem complexity
            success: Whether execution was successful
            execution_time: Execution time
        """
        with self._lock:
            # Add to performance window
            self.performance_window.append(
                {
                    "complexity": complexity,
                    "success": success,
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                }
            )

            # Auto-calibrate if enough data
            if len(self.performance_window) >= 20:
                self.auto_calibrate(list(self.performance_window))

    def _adjust_threshold(self, threshold_name: str, factor: float, direction: str):
        """Adjust specific threshold"""
        # Should be called within lock
        if threshold_name not in self.thresholds:
            return

        config = self.thresholds[threshold_name]
        old_value = config.value

        config.adjust(factor, direction)

        self.total_adjustments += 1

        # Track adjustment
        self.adjustment_history.append(
            {
                "type": "manual_adjustment",
                "threshold": threshold_name,
                "direction": direction,
                "factor": factor,
                "old_value": old_value,
                "new_value": config.value,
                "timestamp": time.time(),
            }
        )

        logger.debug(
            "Adjusted %s threshold: %.3f -> %.3f",
            threshold_name,
            old_value,
            config.value,
        )

    def _get_min_value(self, threshold_name: str) -> float:
        """Get minimum value for threshold"""
        min_values = {
            ThresholdType.CONFIDENCE.value: 0.1,
            ThresholdType.COMPLEXITY.value: 1.0,
            ThresholdType.PERFORMANCE.value: 0.1,
            ThresholdType.TIMEOUT.value: 1.0,
            ThresholdType.RESOURCE.value: 0.1,
        }
        return min_values.get(threshold_name, 0.0)

    def _get_max_value(self, threshold_name: str) -> float:
        """Get maximum value for threshold"""
        max_values = {
            ThresholdType.CONFIDENCE.value: 0.99,
            ThresholdType.COMPLEXITY.value: 10.0,
            ThresholdType.PERFORMANCE.value: 1.0,
            ThresholdType.TIMEOUT.value: 300.0,
            ThresholdType.RESOURCE.value: 1.0,
        }
        return max_values.get(threshold_name, 1.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get threshold statistics"""
        with self._lock:
            return {
                "current_thresholds": self.get_current(),
                "total_adjustments": self.total_adjustments,
                "auto_calibrations": self.auto_calibrations,
                "recent_adjustments": len(self.adjustment_history),
            }


class PerformanceTracker:
    """Tracks decomposition performance"""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance tracker

        Args:
            window_size: Size of performance window
        """
        self.window_size = window_size

        # Performance records
        self.records = deque(maxlen=window_size)

        # Bound unbounded structures
        self.problem_history = defaultdict(
            lambda: deque(maxlen=50)
        )  # Was unbounded list

        # Strategy tracking - use Counter instead of defaultdict(int)
        self.strategy_successes = Counter()
        self.strategy_failures = Counter()

        # Execution time tracking by strategy
        self.strategy_execution_times = defaultdict(lambda: deque(maxlen=100))

        # Failure reasons
        self.failure_reasons = Counter()  # Use Counter

        # Statistics
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0

        # Thread safety
        self._lock = threading.RLock()

        logger.info("PerformanceTracker initialized with window size %d", window_size)

    def record_attempt(self, problem_signature: str) -> str:
        """
        Record decomposition attempt

        Args:
            problem_signature: Problem signature

        Returns:
            Attempt ID
        """
        with self._lock:
            attempt_id = f"{problem_signature}_{int(time.time())}_{self.total_attempts}"
            self.total_attempts += 1

            logger.debug("Recorded attempt for problem %s", problem_signature[:8])

            return attempt_id

    def record_execution(self, problem, plan, outcome):
        """
        Record execution of a plan for a problem

        This is the main entry point for recording execution results.
        It delegates to record_success or record_failure based on the outcome.

        Args:
            problem: ProblemGraph object
            plan: DecompositionPlan or ExecutionPlan object
            outcome: ExecutionOutcome object
        """
        with self._lock:
            # Extract problem signature
            if hasattr(problem, "get_signature"):
                problem_signature = problem.get_signature()
            else:
                problem_signature = str(hash(str(problem)))

            # Extract strategy name safely from either DecompositionPlan or ExecutionPlan
            strategy_used = "unknown"
            if hasattr(plan, "strategy") and plan.strategy:
                if hasattr(plan.strategy, "name"):
                    strategy_used = plan.strategy.name
                else:
                    strategy_used = str(plan.strategy)
            elif hasattr(plan, "metadata") and isinstance(plan.metadata, dict):
                strategy_used = plan.metadata.get("strategy", "unknown")

            # Extract execution time
            execution_time = getattr(outcome, "execution_time", 0.0)

            # Extract metadata
            metadata = getattr(outcome, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Determine success
            success = getattr(outcome, "success", False)

            if success:
                # Record success
                self.record_success(
                    problem_signature=problem_signature,
                    strategy_used=strategy_used,
                    execution_time=execution_time,
                    metadata=metadata,
                )
            else:
                # Extract failure reason
                errors = getattr(outcome, "errors", [])
                if errors:
                    failure_reason = (
                        errors[0] if isinstance(errors, list) else str(errors)
                    )
                else:
                    failure_reason = metadata.get("error", "unknown_error")

                # Record failure
                self.record_failure(
                    problem_signature=problem_signature,
                    reason=failure_reason,
                    strategy_used=strategy_used,
                    execution_time=execution_time,
                )

            logger.debug(
                "Recorded execution for problem %s: success=%s, strategy=%s, time=%.2f",
                problem_signature[:8],
                success,
                strategy_used,
                execution_time,
            )

    def record_success(
        self,
        problem_signature: str,
        strategy_used: str,
        execution_time: float = 0.0,
        metadata: Dict[str, Any] = None,
    ):
        """
        Record successful decomposition

        Args:
            problem_signature: Problem signature
            strategy_used: Strategy that succeeded
            execution_time: Execution time
            metadata: Additional metadata
        """
        with self._lock:
            # FIX: Increment total_attempts counter
            self.total_attempts += 1

            record = PerformanceRecord(
                problem_signature=problem_signature,
                timestamp=time.time(),
                success=True,
                strategy_used=strategy_used,
                execution_time=execution_time,
                metadata=metadata or {},
            )

            self.records.append(record)
            self.problem_history[problem_signature].append(record)
            self.strategy_successes[strategy_used] += 1
            self.strategy_execution_times[strategy_used].append(execution_time)
            self.total_successes += 1

            logger.debug(
                "Recorded success for problem %s using strategy %s",
                problem_signature[:8],
                strategy_used,
            )

    def record_failure(
        self,
        problem_signature: str,
        reason: str,
        strategy_used: str = None,
        execution_time: float = 0.0,
    ):
        """
        Record failed decomposition

        Args:
            problem_signature: Problem signature
            reason: Failure reason
            strategy_used: Strategy that failed
            execution_time: Execution time
        """
        with self._lock:
            # FIX: Increment total_attempts counter
            self.total_attempts += 1

            record = PerformanceRecord(
                problem_signature=problem_signature,
                timestamp=time.time(),
                success=False,
                strategy_used=strategy_used,
                execution_time=execution_time,
                failure_reason=reason,
            )

            self.records.append(record)
            self.problem_history[problem_signature].append(record)

            if strategy_used:
                self.strategy_failures[strategy_used] += 1
                self.strategy_execution_times[strategy_used].append(execution_time)

            self.failure_reasons[reason] += 1
            self.total_failures += 1

            logger.debug(
                "Recorded failure for problem %s: %s", problem_signature[:8], reason
            )

    def get_success_rate(self, window: int = 50) -> float:
        """
        Get success rate over recent window

        Args:
            window: Window size

        Returns:
            Success rate [0, 1]
        """
        with self._lock:
            recent_records = list(self.records)[-window:]

            if not recent_records:
                return 0.5  # Default

            successes = sum(1 for r in recent_records if r.success)
            return successes / len(recent_records)

    def get_strategy_success_rate(self, strategy_name: str) -> float:
        """
        Get success rate for a specific strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Success rate [0, 1]
        """
        with self._lock:
            total = (
                self.strategy_successes[strategy_name]
                + self.strategy_failures[strategy_name]
            )

            if total == 0:
                return 0.5  # Default when no data

            return self.strategy_successes[strategy_name] / total

    def get_average_execution_time(self, strategy_name: str) -> float:
        """
        Get average execution time for a specific strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Average execution time in seconds
        """
        with self._lock:
            times = self.strategy_execution_times.get(strategy_name, [])

            if not times:
                return 30.0  # Default when no data

            return float(np.mean(times))

    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance metrics for specific strategy"""
        with self._lock:
            total = (
                self.strategy_successes[strategy_name]
                + self.strategy_failures[strategy_name]
            )

            if total == 0:
                return {
                    "success_rate": 0.5,
                    "total_attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_execution_time": 30.0,
                }

            return {
                "success_rate": self.strategy_successes[strategy_name] / total,
                "total_attempts": total,
                "successes": self.strategy_successes[strategy_name],
                "failures": self.strategy_failures[strategy_name],
                "avg_execution_time": self.get_average_execution_time(strategy_name),
            }

    def get_problem_history(self, problem_signature: str) -> List[PerformanceRecord]:
        """Get history for specific problem"""
        with self._lock:
            return list(self.problem_history.get(problem_signature, []))

    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get analysis of failure reasons"""
        with self._lock:
            total_failures = sum(self.failure_reasons.values())

            if total_failures == 0:
                return {"total_failures": 0, "reasons": {}}

            reason_percentages = {
                reason: count / total_failures
                for reason, count in self.failure_reasons.items()
            }

            return {
                "total_failures": total_failures,
                "reasons": dict(self.failure_reasons),
                "percentages": reason_percentages,
                "top_reason": max(self.failure_reasons, key=self.failure_reasons.get)
                if self.failure_reasons
                else None,
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        with self._lock:
            return {
                "total_attempts": self.total_attempts,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "overall_success_rate": self.total_successes
                / max(1, self.total_attempts),
                "recent_success_rate": self.get_success_rate(),
                "unique_problems": len(self.problem_history),
                "strategies_used": len(
                    set(
                        list(self.strategy_successes.keys())
                        + list(self.strategy_failures.keys())
                    )
                ),
            }


class StrategyProfiler:
    """Profiles strategy performance and costs"""

    def __init__(self):
        """Initialize strategy profiler"""
        self.profiles = {}  # strategy_name -> StrategyProfile

        # Problem class specific costs
        self.class_costs = defaultdict(lambda: defaultdict(float))

        # Optimal orderings cache
        self.ordering_cache = {}
        self.max_cache_size = 100

        # Statistics
        self.total_updates = 0

        # Thread safety
        self._lock = threading.RLock()

        logger.info("StrategyProfiler initialized")

    def update(
        self,
        strategy_name: str,
        latency: float,
        success: bool,
        problem_class: str = None,
    ):
        """
        Update strategy profile

        Args:
            strategy_name: Name of strategy
            latency: Execution latency
            success: Whether execution succeeded
            problem_class: Optional problem class
        """
        with self._lock:
            # Get or create profile
            if strategy_name not in self.profiles:
                self.profiles[strategy_name] = StrategyProfile(name=strategy_name)

            profile = self.profiles[strategy_name]
            profile.update(latency, success)

            # Update class-specific costs
            if problem_class:
                current_cost = self.class_costs[problem_class][strategy_name]
                # Exponential moving average
                self.class_costs[problem_class][strategy_name] = (
                    0.7 * current_cost + 0.3 * latency
                )

            # Limit ordering cache size
            if len(self.ordering_cache) > self.max_cache_size:
                self.ordering_cache.clear()

            self.total_updates += 1

            logger.debug(
                "Updated profile for strategy %s: latency=%.2f, success=%s",
                strategy_name,
                latency,
                success,
            )

    def get_cost(self, strategy_name: str, problem_class: str = None) -> float:
        """
        Get cost estimate for strategy

        Args:
            strategy_name: Name of strategy
            problem_class: Optional problem class

        Returns:
            Cost estimate (latency)
        """
        with self._lock:
            # Check class-specific cost
            if problem_class and problem_class in self.class_costs:
                if strategy_name in self.class_costs[problem_class]:
                    return self.class_costs[problem_class][strategy_name]

            # Fall back to average latency
            if strategy_name in self.profiles:
                return self.profiles[strategy_name].avg_latency

            # Default cost
            return 10.0

    def get_optimal_ordering(self, problem_class: str = None) -> List[str]:
        """
        Get optimal strategy ordering

        Args:
            problem_class: Optional problem class

        Returns:
            Ordered list of strategy names
        """
        with self._lock:
            # Check cache
            cache_key = problem_class or "general"
            if cache_key in self.ordering_cache:
                return self.ordering_cache[cache_key]

            # FIX: Enforce cache size limit BEFORE adding new entry
            if len(self.ordering_cache) >= self.max_cache_size:
                # Remove oldest 10% of entries (FIFO approximation)
                items = list(self.ordering_cache.items())
                num_to_remove = max(1, len(items) // 10)
                self.ordering_cache = dict(items[num_to_remove:])

            # Calculate cost-effectiveness for each strategy
            strategy_scores = []

            for strategy_name, profile in self.profiles.items():
                # Get cost
                cost = self.get_cost(strategy_name, problem_class)

                # Calculate score (success_rate / cost)
                if cost > 0:
                    score = profile.success_rate / cost
                else:
                    score = profile.success_rate

                strategy_scores.append((strategy_name, score))

            # Sort by score (highest first)
            strategy_scores.sort(key=lambda x: x[1], reverse=True)

            # Extract ordered names
            ordering = [name for name, _ in strategy_scores]

            # Cache result
            self.ordering_cache[cache_key] = ordering

            return ordering

    def get_strategy_profile(self, strategy_name: str) -> Optional[StrategyProfile]:
        """Get profile for specific strategy"""
        with self._lock:
            return self.profiles.get(strategy_name)

    def get_all_profiles(self) -> Dict[str, StrategyProfile]:
        """Get all strategy profiles"""
        with self._lock:
            return self.profiles.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get profiler statistics"""
        with self._lock:
            stats = {
                "total_strategies": len(self.profiles),
                "total_updates": self.total_updates,
                "problem_classes": len(self.class_costs),
                "strategy_summaries": {},
            }

            for name, profile in self.profiles.items():
                stats["strategy_summaries"][name] = {
                    "attempts": profile.total_attempts,
                    "success_rate": profile.success_rate,
                    "avg_latency": profile.avg_latency,
                }

            return stats

    def recommend_strategy(
        self, problem_class: str = None, max_latency: float = None
    ) -> Optional[str]:
        """
        Recommend best strategy for constraints

        Args:
            problem_class: Problem class
            max_latency: Maximum allowed latency

        Returns:
            Recommended strategy name or None
        """
        with self._lock:
            ordering = self.get_optimal_ordering(problem_class)

            for strategy_name in ordering:
                # Check latency constraint
                if max_latency:
                    cost = self.get_cost(strategy_name, problem_class)
                    if cost > max_latency:
                        continue

                # Check if strategy has reasonable success rate
                if strategy_name in self.profiles:
                    if self.profiles[strategy_name].success_rate >= 0.3:
                        return strategy_name

            # Return first strategy if no good match
            return ordering[0] if ordering else None
