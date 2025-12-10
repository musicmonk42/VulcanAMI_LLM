from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================


class OptimizationStrategy(Enum):
    """Optimization strategies."""

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    COST_FOCUSED = "cost_focused"
    PERFORMANCE_FOCUSED = "performance_focused"


class OptimizationPhase(Enum):
    """Phases of optimization."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETED = "completed"


class ResourceTier(Enum):
    """Storage resource tiers."""

    HOT = "hot"  # In-memory, fastest
    WARM = "warm"  # SSD, fast
    COLD = "cold"  # HDD/S3, slow
    ARCHIVE = "archive"  # Glacier, slowest


class CostMetric(Enum):
    """Cost metrics to track."""

    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    COMPUTE_HOURS = "compute_hours"
    API_CALLS = "api_calls"
    CDN_REQUESTS = "cdn_requests"


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class CostBreakdown:
    """Breakdown of costs."""

    storage_cost: float = 0.0
    bandwidth_cost: float = 0.0
    compute_cost: float = 0.0
    api_cost: float = 0.0
    cdn_cost: float = 0.0
    total_cost: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_cost": self.storage_cost,
            "bandwidth_cost": self.bandwidth_cost,
            "compute_cost": self.compute_cost,
            "api_cost": self.api_cost,
            "cdn_cost": self.cdn_cost,
            "total_cost": self.total_cost,
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizationReport:
    """Report of optimization results."""

    optimization_id: str
    strategy: OptimizationStrategy
    phase: OptimizationPhase
    started_at: float
    completed_at: Optional[float] = None
    cost_before: Optional[CostBreakdown] = None
    cost_after: Optional[CostBreakdown] = None
    savings: float = 0.0
    savings_percentage: float = 0.0
    actions_taken: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration(self) -> Optional[float]:
        """Get optimization duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimization_id": self.optimization_id,
            "strategy": self.strategy.value,
            "phase": self.phase.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cost_before": self.cost_before.to_dict() if self.cost_before else None,
            "cost_after": self.cost_after.to_dict() if self.cost_after else None,
            "savings": self.savings,
            "savings_percentage": self.savings_percentage,
            "actions_taken": self.actions_taken,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "duration": self.get_duration(),
            "metadata": self.metadata,
        }


@dataclass
class OptimizationMetrics:
    """Metrics for optimization operations."""

    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    total_savings: float = 0.0
    total_gb_saved: float = 0.0
    total_api_calls_reduced: int = 0
    average_savings_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "failed_optimizations": self.failed_optimizations,
            "total_savings": self.total_savings,
            "total_gb_saved": self.total_gb_saved,
            "total_api_calls_reduced": self.total_api_calls_reduced,
            "average_savings_percentage": self.average_savings_percentage,
            "success_rate": self.get_success_rate(),
        }

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations


@dataclass
class BudgetConfig:
    """Budget configuration and limits."""

    monthly_budget: float = 1000.0
    storage_limit_gb: float = 1000.0
    bandwidth_limit_gb: float = 5000.0
    alert_threshold: float = 0.8  # Alert at 80% of budget
    hard_limit_threshold: float = 0.95  # Hard stop at 95%

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "monthly_budget": self.monthly_budget,
            "storage_limit_gb": self.storage_limit_gb,
            "bandwidth_limit_gb": self.bandwidth_limit_gb,
            "alert_threshold": self.alert_threshold,
            "hard_limit_threshold": self.hard_limit_threshold,
        }


# ============================================================
# COST ANALYZER
# ============================================================


class CostAnalyzer:
    """Analyzes costs and identifies optimization opportunities."""

    def __init__(self, memory_system):
        """Initialize cost analyzer."""
        self.memory = memory_system
        self.cost_history: deque = deque(maxlen=1000)

        # Pricing (example rates per GB/hour/call)
        self.pricing = {
            "storage_hot_gb_month": 0.23,
            "storage_warm_gb_month": 0.10,
            "storage_cold_gb_month": 0.023,
            "bandwidth_gb": 0.09,
            "compute_hour": 0.05,
            "api_call": 0.0001,
            "cdn_request": 0.000001,
        }

    def analyze_current_costs(self) -> CostBreakdown:
        """
        Analyze current cost structure with defensive programming.

        FIXED: Now provides fallback logic when memory system methods don't exist.
        Calculates costs from alternative methods if primary methods are unavailable.
        """
        breakdown = CostBreakdown()

        try:
            # Storage costs - FIXED: Multiple fallback methods
            if hasattr(self.memory, "get_storage_stats"):
                stats = self.memory.get_storage_stats()
                if isinstance(stats, dict):
                    try:
                        breakdown.storage_cost = (
                            float(stats.get("hot_gb", 0))
                            * self.pricing["storage_hot_gb_month"]
                            + float(stats.get("warm_gb", 0))
                            * self.pricing["storage_warm_gb_month"]
                            + float(stats.get("cold_gb", 0))
                            * self.pricing["storage_cold_gb_month"]
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid storage stats data: {e}")
                        breakdown.storage_cost = 0.0
            # FIXED: Fallback method using tier-based storage
            elif hasattr(self.memory, "get_tier_storage_gb"):
                try:
                    hot_gb = (
                        self.memory.get_tier_storage_gb(ResourceTier.HOT)
                        if hasattr(self.memory, "get_tier_storage_gb")
                        else 0
                    )
                    warm_gb = (
                        self.memory.get_tier_storage_gb(ResourceTier.WARM)
                        if hasattr(self.memory, "get_tier_storage_gb")
                        else 0
                    )
                    cold_gb = (
                        self.memory.get_tier_storage_gb(ResourceTier.COLD)
                        if hasattr(self.memory, "get_tier_storage_gb")
                        else 0
                    )

                    breakdown.storage_cost = (
                        float(hot_gb) * self.pricing["storage_hot_gb_month"]
                        + float(warm_gb) * self.pricing["storage_warm_gb_month"]
                        + float(cold_gb) * self.pricing["storage_cold_gb_month"]
                    )
                except Exception as e:
                    logger.warning(f"Fallback storage calculation failed: {e}")
            # FIXED: Second fallback using total storage
            elif hasattr(self.memory, "get_total_storage_gb"):
                try:
                    total_gb = float(self.memory.get_total_storage_gb())
                    # Assume mixed tier pricing (average)
                    avg_price = (
                        self.pricing["storage_hot_gb_month"]
                        + self.pricing["storage_warm_gb_month"]
                        + self.pricing["storage_cold_gb_month"]
                    ) / 3
                    breakdown.storage_cost = total_gb * avg_price
                except Exception as e:
                    logger.warning(f"Total storage calculation failed: {e}")

            # Bandwidth costs - FIXED: Multiple fallback methods
            if hasattr(self.memory, "get_bandwidth_stats"):
                stats = self.memory.get_bandwidth_stats()
                if isinstance(stats, dict):
                    try:
                        breakdown.bandwidth_cost = (
                            float(stats.get("total_gb", 0))
                            * self.pricing["bandwidth_gb"]
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid bandwidth stats data: {e}")
                        breakdown.bandwidth_cost = 0.0
            # FIXED: Fallback method
            elif hasattr(self.memory, "get_bandwidth_usage_gb"):
                try:
                    bandwidth_gb = float(self.memory.get_bandwidth_usage_gb())
                    breakdown.bandwidth_cost = (
                        bandwidth_gb * self.pricing["bandwidth_gb"]
                    )
                except Exception as e:
                    logger.warning(f"Bandwidth calculation failed: {e}")

            # Compute costs - FIXED: Add type checking
            if hasattr(self.memory, "get_compute_stats"):
                stats = self.memory.get_compute_stats()
                if isinstance(stats, dict):
                    try:
                        breakdown.compute_cost = (
                            float(stats.get("hours", 0)) * self.pricing["compute_hour"]
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid compute stats data: {e}")
                        breakdown.compute_cost = 0.0

            # API costs - FIXED: Multiple fallback methods
            if hasattr(self.memory, "get_api_stats"):
                stats = self.memory.get_api_stats()
                if isinstance(stats, dict):
                    try:
                        breakdown.api_cost = (
                            float(stats.get("total_calls", 0))
                            * self.pricing["api_call"]
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid API stats data: {e}")
                        breakdown.api_cost = 0.0
            # FIXED: Fallback method
            elif hasattr(self.memory, "get_api_call_count"):
                try:
                    api_calls = float(self.memory.get_api_call_count())
                    breakdown.api_cost = api_calls * self.pricing["api_call"]
                except Exception as e:
                    logger.warning(f"API cost calculation failed: {e}")

            # CDN costs - FIXED: Multiple fallback methods
            if hasattr(self.memory, "get_cdn_stats"):
                stats = self.memory.get_cdn_stats()
                if isinstance(stats, dict):
                    try:
                        breakdown.cdn_cost = (
                            float(stats.get("requests", 0))
                            * self.pricing["cdn_request"]
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid CDN stats data: {e}")
                        breakdown.cdn_cost = 0.0
            # FIXED: Fallback method
            elif hasattr(self.memory, "get_cdn_request_count"):
                try:
                    cdn_requests = float(self.memory.get_cdn_request_count())
                    breakdown.cdn_cost = cdn_requests * self.pricing["cdn_request"]
                except Exception as e:
                    logger.warning(f"CDN cost calculation failed: {e}")

            # Calculate total
            breakdown.total_cost = (
                breakdown.storage_cost
                + breakdown.bandwidth_cost
                + breakdown.compute_cost
                + breakdown.api_cost
                + breakdown.cdn_cost
            )

        except Exception as e:
            logger.error(f"Cost analysis failed: {e}")

        self.cost_history.append(breakdown)
        return breakdown

    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify potential cost optimization opportunities with defensive programming.

        FIXED: Now provides fallback estimates when specific methods don't exist.
        """
        opportunities = []

        try:
            # Check for deduplication opportunities - FIXED: Better error handling
            if hasattr(self.memory, "dedup_engine"):
                try:
                    # Try to get potential savings
                    if hasattr(self.memory.dedup_engine, "get_potential_savings"):
                        savings = self.memory.dedup_engine.get_potential_savings()
                    # FIXED: Fallback - estimate from duplication rate
                    elif hasattr(self.memory, "get_stats"):
                        stats = self.memory.get_stats()
                        if isinstance(stats, dict) and "duplication_rate" in stats:
                            total_storage = (
                                self.memory.get_total_storage_gb()
                                if hasattr(self.memory, "get_total_storage_gb")
                                else 0
                            )
                            savings = total_storage * float(
                                stats.get("duplication_rate", 0)
                            )
                        else:
                            savings = 0
                    else:
                        # FIXED: Conservative estimate if no data available
                        total_storage = (
                            self.memory.get_total_storage_gb()
                            if hasattr(self.memory, "get_total_storage_gb")
                            else 0
                        )
                        savings = total_storage * 0.1  # Assume 10% duplication

                    if isinstance(savings, (int, float)) and savings > 0:
                        opportunities.append(
                            {
                                "type": "deduplication",
                                "potential_savings_gb": float(savings),
                                "priority": "high" if savings > 10 else "medium",
                                "action": "fold_ir_atoms",
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get deduplication savings: {e}")

            # Check for quantization opportunities - FIXED: Validate return values
            if hasattr(self.memory, "quantizer"):
                try:
                    # Try to get estimate
                    if hasattr(self.memory.quantizer, "estimate_savings"):
                        quant_savings = self.memory.quantizer.estimate_savings()
                    else:
                        # FIXED: Estimate based on storage size (quantization typically saves 50-75%)
                        total_storage = (
                            self.memory.get_total_storage_gb()
                            if hasattr(self.memory, "get_total_storage_gb")
                            else 0
                        )
                        quant_savings = total_storage * 0.5  # Conservative 50% estimate

                    if isinstance(quant_savings, (int, float)) and quant_savings > 0.05:
                        opportunities.append(
                            {
                                "type": "quantization",
                                "potential_savings_gb": float(quant_savings),
                                "priority": "medium",
                                "action": "apply_rotational_quantization",
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to estimate quantization savings: {e}")

            # Check for tier migration opportunities - FIXED: Null check before len()
            if hasattr(self.memory, "tier_c"):
                try:
                    # Try to get candidates
                    if hasattr(self.memory.tier_c, "get_cold_candidates"):
                        migration_candidates = self.memory.tier_c.get_cold_candidates()
                    else:
                        # FIXED: Estimate based on utilization
                        if hasattr(self.memory, "get_stats"):
                            stats = self.memory.get_stats()
                            if isinstance(stats, dict):
                                util_rate = float(stats.get("utilization_rate", 1.0))
                                if (
                                    util_rate < 0.5
                                ):  # Low utilization suggests migration opportunity
                                    # Estimate number of candidates
                                    migration_candidates = ["estimated"] * int(
                                        100 * (1 - util_rate)
                                    )
                                else:
                                    migration_candidates = []
                            else:
                                migration_candidates = []
                        else:
                            migration_candidates = []

                    if migration_candidates and hasattr(
                        migration_candidates, "__len__"
                    ):
                        candidate_count = len(migration_candidates)
                        if candidate_count > 0:
                            opportunities.append(
                                {
                                    "type": "tier_migration",
                                    "items": candidate_count,
                                    "priority": "medium"
                                    if candidate_count > 50
                                    else "low",
                                    "action": "migrate_to_cold_storage",
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to check migration candidates: {e}")

            # Check for pruning opportunities - FIXED: Validate numeric return
            if hasattr(self.memory, "tier_c"):
                try:
                    # Try to count low utility items
                    if hasattr(self.memory.tier_c, "count_low_utility"):
                        low_utility = self.memory.tier_c.count_low_utility(
                            threshold=0.1
                        )
                    else:
                        # FIXED: Estimate based on stats
                        if hasattr(self.memory, "get_stats"):
                            stats = self.memory.get_stats()
                            if isinstance(stats, dict):
                                # Estimate: assume 10% of items are low utility if utilization is low
                                util_rate = float(stats.get("utilization_rate", 1.0))
                                if util_rate < 0.7:
                                    low_utility = int(
                                        1000 * (1 - util_rate)
                                    )  # Rough estimate
                                else:
                                    low_utility = 0
                            else:
                                low_utility = 0
                        else:
                            low_utility = 0

                    if isinstance(low_utility, (int, float)) and low_utility > 100:
                        opportunities.append(
                            {
                                "type": "pruning",
                                "items": int(low_utility),
                                "priority": "low",
                                "action": "prune_low_utility",
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to count low utility items: {e}")

            # Check CDN efficiency - FIXED: Better validation
            if hasattr(self.memory, "cdn"):
                try:
                    if hasattr(self.memory.cdn, "get_hit_rate"):
                        hit_rate = self.memory.cdn.get_hit_rate()
                    else:
                        # FIXED: Default to 0.5 if unavailable
                        hit_rate = 0.5

                    if isinstance(hit_rate, (int, float)) and hit_rate < 0.7:
                        opportunities.append(
                            {
                                "type": "cdn_optimization",
                                "hit_rate": float(hit_rate),
                                "priority": "high" if hit_rate < 0.5 else "medium",
                                "action": "optimize_cache_policy",
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get CDN hit rate: {e}")

        except Exception as e:
            logger.error(f"Opportunity identification failed: {e}")

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        opportunities.sort(
            key=lambda x: priority_order.get(x.get("priority", "low"), 2)
        )

        return opportunities

    def estimate_savings(self, opportunities: List[Dict[str, Any]]) -> float:
        """Estimate total potential savings from opportunities."""
        total_savings = 0.0

        for opp in opportunities:
            try:
                if "potential_savings_gb" in opp:
                    # Convert GB savings to cost savings
                    savings_gb = float(opp["potential_savings_gb"])
                    total_savings += savings_gb * self.pricing["storage_cold_gb_month"]
                elif opp["type"] == "cdn_optimization":
                    # Estimate bandwidth savings from improved cache hit rate
                    if hasattr(self.memory, "get_bandwidth_stats"):
                        try:
                            stats = self.memory.get_bandwidth_stats()
                            if isinstance(stats, dict):
                                current_bandwidth = float(stats.get("total_gb", 0))
                                # Estimate 20% bandwidth reduction
                                total_savings += (
                                    current_bandwidth
                                    * 0.2
                                    * self.pricing["bandwidth_gb"]
                                )
                        except (ValueError, TypeError):
                            pass  # Skip if can't calculate
                    # FIXED: Fallback bandwidth calculation
                    elif hasattr(self.memory, "get_bandwidth_usage_gb"):
                        try:
                            current_bandwidth = float(
                                self.memory.get_bandwidth_usage_gb()
                            )
                            total_savings += (
                                current_bandwidth * 0.2 * self.pricing["bandwidth_gb"]
                            )
                        except Exception:
                            pass
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to estimate savings for opportunity: {e}")
                continue

        return total_savings

    def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trends over time."""
        if len(self.cost_history) < 2:
            return {
                "trend": "insufficient_data",
                "average_cost": 0,
                "change_percentage": 0,
            }

        # FIXED: Prevent division by zero
        if days <= 0:
            days = 30  # Default to 30 days

        recent_costs = list(self.cost_history)[-days:]

        if len(recent_costs) == 0:
            return {
                "trend": "insufficient_data",
                "average_cost": 0,
                "change_percentage": 0,
            }

        # Calculate trend
        costs = [c.total_cost for c in recent_costs]
        avg_cost = sum(costs) / len(costs) if len(costs) > 0 else 0

        # Compare first half vs second half
        mid = len(costs) // 2
        if mid == 0:
            return {
                "trend": "insufficient_data",
                "average_cost": avg_cost,
                "change_percentage": 0,
            }

        first_half_avg = sum(costs[:mid]) / mid if mid > 0 else 0
        second_half_avg = (
            sum(costs[mid:]) / (len(costs) - mid) if mid < len(costs) else 0
        )

        trend = "stable"
        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"

        return {
            "trend": trend,
            "average_cost": avg_cost,
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
            "change_percentage": (
                (second_half_avg - first_half_avg) / first_half_avg * 100
            )
            if first_half_avg > 0
            else 0,
        }


# ============================================================
# COST OPTIMIZER
# ============================================================


class CostOptimizer:
    """
    Comprehensive cost optimization system for memory storage.

    Features:
    - Multi-strategy optimization
    - Cost analysis and forecasting
    - Budget management
    - Automated scheduling
    - What-if analysis
    - Comprehensive metrics
    - Rollback support
    """

    def __init__(
        self,
        persistent_memory,
        budget_config: Optional[BudgetConfig] = None,
        default_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        auto_optimize: bool = False,
        optimization_interval: int = 3600,
    ):
        """
        Initialize cost optimizer.

        Args:
            persistent_memory: Persistent memory system
            budget_config: Budget configuration
            default_strategy: Default optimization strategy
            auto_optimize: Enable automatic optimization
            optimization_interval: Interval for automatic optimization (seconds)
        """
        self.memory = persistent_memory
        self.budget_config = budget_config or BudgetConfig()
        self.default_strategy = default_strategy

        # Components
        self.analyzer = CostAnalyzer(persistent_memory)

        # State
        self.optimization_history: deque = deque(maxlen=100)
        self.current_optimization: Optional[OptimizationReport] = None
        self.metrics = OptimizationMetrics()

        # Callbacks
        self.on_optimization_complete: List[Callable] = []
        self.on_budget_alert: List[Callable] = []

        # Auto-optimization
        self.auto_optimize = auto_optimize
        self.optimization_interval = optimization_interval
        self._shutdown = False
        self._lock = threading.Lock()

        if auto_optimize:
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True
            )
            self._scheduler_thread.start()

        logger.info("CostOptimizer initialized")

    def optimize_storage(
        self, strategy: Optional[OptimizationStrategy] = None, dry_run: bool = False
    ) -> OptimizationReport:
        """
        Optimize storage costs using specified strategy.

        Args:
            strategy: Optimization strategy to use
            dry_run: If True, only analyze without making changes

        Returns:
            Optimization report
        """
        strategy = strategy or self.default_strategy

        with self._lock:
            report = OptimizationReport(
                optimization_id=self._generate_optimization_id(),
                strategy=strategy,
                phase=OptimizationPhase.ANALYSIS,
                started_at=time.time(),
            )

            self.current_optimization = report
            self.metrics.total_optimizations += 1

        try:
            logger.info(f"Starting storage optimization with {strategy.value} strategy")

            # Analysis phase
            report.cost_before = self.analyzer.analyze_current_costs()
            opportunities = self.analyzer.identify_optimization_opportunities()

            if dry_run:
                # Just provide recommendations
                for opp in opportunities:
                    report.recommendations.append(
                        f"Execute {opp['action']} ({opp['priority']} priority)"
                    )
                # FIXED: Ensure phase is set to COMPLETED
                report.phase = OptimizationPhase.COMPLETED
                report.completed_at = time.time()
                return report

            # Planning phase
            report.phase = OptimizationPhase.PLANNING
            plan = self._create_optimization_plan(opportunities, strategy)

            # Execution phase
            report.phase = OptimizationPhase.EXECUTION
            self._execute_storage_optimizations(plan, report, strategy)

            # Verification phase
            report.phase = OptimizationPhase.VERIFICATION
            report.cost_after = self.analyzer.analyze_current_costs()

            # Calculate savings
            if report.cost_before and report.cost_after:
                report.savings = (
                    report.cost_before.total_cost - report.cost_after.total_cost
                )
                if report.cost_before.total_cost > 0:
                    report.savings_percentage = (
                        report.savings / report.cost_before.total_cost
                    ) * 100

            # Update metrics
            with self._lock:
                if report.savings > 0:
                    self.metrics.successful_optimizations += 1
                    self.metrics.total_savings += report.savings
                    # FIXED: Update average savings percentage
                    if self.metrics.successful_optimizations > 0:
                        self.metrics.average_savings_percentage = (
                            self.metrics.total_savings
                            / self.metrics.successful_optimizations
                        )
                else:
                    self.metrics.failed_optimizations += 1

            # FIXED: Ensure phase is always set to COMPLETED
            report.phase = OptimizationPhase.COMPLETED
            report.completed_at = time.time()

            # Trigger callbacks
            for callback in self.on_optimization_complete:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"Optimization callback failed: {e}")

            logger.info(
                f"Storage optimization completed. Savings: ${report.savings:.2f}"
            )

        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            report.warnings.append(f"Optimization failed: {str(e)}")
            # FIXED: Still mark as completed even on error
            report.phase = OptimizationPhase.COMPLETED
            report.completed_at = time.time()
            with self._lock:
                self.metrics.failed_optimizations += 1

        finally:
            with self._lock:
                self.optimization_history.append(report)
                self.current_optimization = None

        return report

    def optimize_retrieval(
        self, strategy: Optional[OptimizationStrategy] = None, dry_run: bool = False
    ) -> OptimizationReport:
        """
        Optimize retrieval performance and costs.

        Args:
            strategy: Optimization strategy to use
            dry_run: If True, only analyze without making changes

        Returns:
            Optimization report
        """
        strategy = strategy or self.default_strategy

        report = OptimizationReport(
            optimization_id=self._generate_optimization_id(),
            strategy=strategy,
            phase=OptimizationPhase.ANALYSIS,
            started_at=time.time(),
        )

        try:
            logger.info(
                f"Starting retrieval optimization with {strategy.value} strategy"
            )
            report.cost_before = self.analyzer.analyze_current_costs()

            if dry_run:
                report.recommendations.append("Enable adaptive range queries")
                report.recommendations.append("Optimize CDN cache policy")
                report.recommendations.append("Enable intelligent prefetching")
                # FIXED: Ensure phase is set to COMPLETED
                report.phase = OptimizationPhase.COMPLETED
                report.completed_at = time.time()
                return report

            report.phase = OptimizationPhase.EXECUTION

            # Enable adaptive range queries - FIXED: Add error handling
            if hasattr(self.memory, "persistent_store") and hasattr(
                self.memory.persistent_store, "enable_adaptive_range"
            ):
                try:
                    self.memory.persistent_store.enable_adaptive_range()
                    report.actions_taken.append("Enabled adaptive range queries")
                except Exception as e:
                    report.warnings.append(f"Adaptive range failed: {str(e)}")

            # Optimize CDN - FIXED: Add error handling
            if hasattr(self.memory, "cdn"):
                try:
                    if hasattr(self.memory.cdn, "batch_purges"):
                        self.memory.cdn.batch_purges(interval_hours=6)
                        report.actions_taken.append("Configured CDN batch purges")
                    elif hasattr(self.memory.cdn, "optimize_cache_policy"):
                        self.memory.cdn.optimize_cache_policy()
                        report.actions_taken.append("Optimized CDN cache policy")
                except Exception as e:
                    report.warnings.append(f"CDN optimization failed: {str(e)}")

            # Enable prefetching - FIXED: Add error handling
            if hasattr(self.memory, "prefetcher") and hasattr(
                self.memory.prefetcher, "enable_intelligent_prefetch"
            ):
                try:
                    self.memory.prefetcher.enable_intelligent_prefetch()
                    report.actions_taken.append("Enabled intelligent prefetching")
                except Exception as e:
                    report.warnings.append(f"Prefetcher setup failed: {str(e)}")

            report.phase = OptimizationPhase.VERIFICATION
            report.cost_after = self.analyzer.analyze_current_costs()

            # Calculate savings
            if report.cost_before and report.cost_after:
                report.savings = (
                    report.cost_before.total_cost - report.cost_after.total_cost
                )
                if report.cost_before.total_cost > 0:
                    report.savings_percentage = (
                        report.savings / report.cost_before.total_cost
                    ) * 100

            # FIXED: Ensure phase is always set to COMPLETED
            report.phase = OptimizationPhase.COMPLETED
            report.completed_at = time.time()

        except Exception as e:
            logger.error(f"Retrieval optimization failed: {e}")
            report.warnings.append(f"Retrieval optimization failed: {str(e)}")
            # FIXED: Still mark as completed even on error
            report.phase = OptimizationPhase.COMPLETED
            report.completed_at = time.time()

        return report

    def optimize_full(
        self, strategy: Optional[OptimizationStrategy] = None, dry_run: bool = False
    ) -> OptimizationReport:
        """
        Run full optimization (both storage and retrieval).

        FIXED: Now returns a single consolidated OptimizationReport instead of a list.
        Combines both storage and retrieval optimization into one comprehensive report.

        Args:
            strategy: Optimization strategy to use
            dry_run: If True, only analyze without making changes

        Returns:
            Single consolidated optimization report
        """
        strategy = strategy or self.default_strategy

        # Create consolidated report
        consolidated_report = OptimizationReport(
            optimization_id=self._generate_optimization_id(),
            strategy=strategy,
            phase=OptimizationPhase.ANALYSIS,
            started_at=time.time(),
        )

        try:
            logger.info(f"Starting full optimization with {strategy.value} strategy")

            # Get initial costs
            consolidated_report.cost_before = self.analyzer.analyze_current_costs()

            # Run storage optimization
            storage_report = self.optimize_storage(strategy, dry_run)

            # Merge storage results
            consolidated_report.actions_taken.extend(storage_report.actions_taken)
            consolidated_report.recommendations.extend(storage_report.recommendations)
            consolidated_report.warnings.extend(storage_report.warnings)

            # Run retrieval optimization
            retrieval_report = self.optimize_retrieval(strategy, dry_run)

            # Merge retrieval results
            consolidated_report.actions_taken.extend(retrieval_report.actions_taken)
            consolidated_report.recommendations.extend(retrieval_report.recommendations)
            consolidated_report.warnings.extend(retrieval_report.warnings)

            # Get final costs
            consolidated_report.cost_after = self.analyzer.analyze_current_costs()

            # Calculate total savings
            if consolidated_report.cost_before and consolidated_report.cost_after:
                consolidated_report.savings = (
                    consolidated_report.cost_before.total_cost
                    - consolidated_report.cost_after.total_cost
                )
                if consolidated_report.cost_before.total_cost > 0:
                    consolidated_report.savings_percentage = (
                        consolidated_report.savings
                        / consolidated_report.cost_before.total_cost
                    ) * 100

            # Add metadata
            consolidated_report.metadata["storage_report_id"] = (
                storage_report.optimization_id
            )
            consolidated_report.metadata["retrieval_report_id"] = (
                retrieval_report.optimization_id
            )

            # FIXED: Ensure phase is COMPLETED
            consolidated_report.phase = OptimizationPhase.COMPLETED
            consolidated_report.completed_at = time.time()

            logger.info(
                f"Full optimization completed. Total savings: ${consolidated_report.savings:.2f}"
            )

        except Exception as e:
            logger.error(f"Full optimization failed: {e}")
            consolidated_report.warnings.append(f"Full optimization failed: {str(e)}")
            # FIXED: Still mark as completed even on error
            consolidated_report.phase = OptimizationPhase.COMPLETED
            consolidated_report.completed_at = time.time()

        finally:
            with self._lock:
                self.optimization_history.append(consolidated_report)

        return consolidated_report

    def check_budget(self) -> Dict[str, Any]:
        """Check current budget status and trigger alerts if needed."""
        try:
            current_cost = self.analyzer.analyze_current_costs()
            monthly_cost = current_cost.total_cost * 30  # Estimate monthly from current

            usage_percentage = (
                monthly_cost / self.budget_config.monthly_budget
                if self.budget_config.monthly_budget > 0
                else 0
            )

            status = {
                "current_monthly_cost": monthly_cost,
                "projected_monthly_cost": monthly_cost,  # Could be more sophisticated
                "budget_limit": self.budget_config.monthly_budget,
                "usage_percentage": usage_percentage,
                "remaining_budget": max(
                    0, self.budget_config.monthly_budget - monthly_cost
                ),
                "status": "ok",
            }

            # Check thresholds
            if usage_percentage >= self.budget_config.hard_limit_threshold:
                status["status"] = "critical"
                self._trigger_budget_alert("critical", status)
            elif usage_percentage >= self.budget_config.alert_threshold:
                status["status"] = "warning"
                self._trigger_budget_alert("warning", status)

            return status

        except Exception as e:
            logger.error(f"Budget check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "current_monthly_cost": 0.0,
                "budget_limit": self.budget_config.monthly_budget,
                "usage_percentage": 0,
            }

    def forecast_costs(self, days: int = 30) -> Dict[str, Any]:
        """Forecast costs for the next N days."""
        try:
            trends = self.analyzer.get_cost_trends(days)
            current_daily = trends.get("average_cost", 0)

            # Simple linear projection based on trend
            if trends["trend"] == "increasing":
                growth_rate = 1.05  # 5% growth
            elif trends["trend"] == "decreasing":
                growth_rate = 0.95  # 5% decrease
            else:
                growth_rate = 1.0  # Stable

            projected_daily = current_daily * growth_rate
            projected_total = projected_daily * days

            return {
                "current_daily_average": current_daily,
                "projected_daily_average": projected_daily,
                "projected_total": projected_total,
                "trend": trends["trend"],
                "confidence": "medium",  # Could be more sophisticated
                "forecast_period_days": days,
            }

        except Exception as e:
            logger.error(f"Cost forecasting failed: {e}")
            return {
                "error": str(e),
                "projected_total": 0.0,
                "forecast_period_days": days,
            }

    def what_if_analysis(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential impact of changes.

        FIXED: Now handles all change scenarios including percentage-based changes.

        Supported change parameters:
        - reduce_storage_by_gb: Absolute GB reduction
        - reduce_storage_by_percent: Percentage reduction (0-100)
        - reduce_bandwidth_by_gb: Absolute GB reduction
        - reduce_bandwidth_by_percent: Percentage reduction (0-100)
        - increase_cdn_hit_rate: Hit rate increase (0-1)
        - reduce_api_calls_by_percent: Percentage reduction (0-100)
        - migrate_to_cold_tier_gb: GB to migrate to cold storage

        Args:
            changes: Dictionary of potential changes to analyze

        Returns:
            Analysis of potential impact
        """
        try:
            current_costs = self.analyzer.analyze_current_costs()
            projected_costs = CostBreakdown()

            # Copy current costs
            projected_costs.storage_cost = current_costs.storage_cost
            projected_costs.bandwidth_cost = current_costs.bandwidth_cost
            projected_costs.compute_cost = current_costs.compute_cost
            projected_costs.api_cost = current_costs.api_cost
            projected_costs.cdn_cost = current_costs.cdn_cost

            # FIXED: Handle all storage reduction scenarios
            if "reduce_storage_by_gb" in changes:
                try:
                    reduction_gb = float(changes["reduce_storage_by_gb"])
                    storage_savings = (
                        reduction_gb * self.analyzer.pricing["storage_cold_gb_month"]
                    )
                    projected_costs.storage_cost = max(
                        0, projected_costs.storage_cost - storage_savings
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid storage reduction value: {e}")

            # FIXED: Handle percentage-based storage reduction
            if "reduce_storage_by_percent" in changes:
                try:
                    reduction_percent = (
                        float(changes["reduce_storage_by_percent"]) / 100.0
                    )
                    if 0 <= reduction_percent <= 1:
                        projected_costs.storage_cost = current_costs.storage_cost * (
                            1 - reduction_percent
                        )
                    else:
                        logger.warning(
                            f"Invalid storage reduction percentage: {reduction_percent * 100}%"
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid storage reduction percentage: {e}")

            # FIXED: Handle absolute bandwidth reduction
            if "reduce_bandwidth_by_gb" in changes:
                try:
                    reduction_gb = float(changes["reduce_bandwidth_by_gb"])
                    bandwidth_savings = (
                        reduction_gb * self.analyzer.pricing["bandwidth_gb"]
                    )
                    projected_costs.bandwidth_cost = max(
                        0, projected_costs.bandwidth_cost - bandwidth_savings
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid bandwidth reduction value: {e}")

            # FIXED: Handle percentage-based bandwidth reduction
            if "reduce_bandwidth_by_percent" in changes:
                try:
                    reduction_percent = (
                        float(changes["reduce_bandwidth_by_percent"]) / 100.0
                    )
                    if 0 <= reduction_percent <= 1:
                        projected_costs.bandwidth_cost = (
                            current_costs.bandwidth_cost * (1 - reduction_percent)
                        )
                    else:
                        logger.warning(
                            f"Invalid bandwidth reduction percentage: {reduction_percent * 100}%"
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid bandwidth reduction percentage: {e}")

            # FIXED: Handle CDN hit rate improvement
            if "increase_cdn_hit_rate" in changes:
                try:
                    hit_rate_increase = float(changes["increase_cdn_hit_rate"])
                    if 0 <= hit_rate_increase <= 1:
                        # Better hit rate reduces bandwidth usage
                        bandwidth_savings = (
                            current_costs.bandwidth_cost * hit_rate_increase
                        )
                        projected_costs.bandwidth_cost = max(
                            0, projected_costs.bandwidth_cost - bandwidth_savings
                        )
                    else:
                        logger.warning(
                            f"Invalid CDN hit rate value: {hit_rate_increase}"
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid CDN hit rate value: {e}")

            # FIXED: Handle API call reduction
            if "reduce_api_calls_by_percent" in changes:
                try:
                    reduction_percent = (
                        float(changes["reduce_api_calls_by_percent"]) / 100.0
                    )
                    if 0 <= reduction_percent <= 1:
                        projected_costs.api_cost = current_costs.api_cost * (
                            1 - reduction_percent
                        )
                    else:
                        logger.warning(
                            f"Invalid API reduction percentage: {reduction_percent * 100}%"
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid API reduction percentage: {e}")

            # FIXED: Handle tier migration
            if "migrate_to_cold_tier_gb" in changes:
                try:
                    migrate_gb = float(changes["migrate_to_cold_tier_gb"])
                    # Savings from moving hot/warm storage to cold
                    hot_price = self.analyzer.pricing["storage_hot_gb_month"]
                    cold_price = self.analyzer.pricing["storage_cold_gb_month"]
                    savings = migrate_gb * (hot_price - cold_price)
                    projected_costs.storage_cost = max(
                        0, projected_costs.storage_cost - savings
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid migration value: {e}")

            # Calculate total projected cost
            projected_costs.total_cost = (
                projected_costs.storage_cost
                + projected_costs.bandwidth_cost
                + projected_costs.compute_cost
                + projected_costs.api_cost
                + projected_costs.cdn_cost
            )

            potential_savings = current_costs.total_cost - projected_costs.total_cost
            savings_percentage = (
                (potential_savings / current_costs.total_cost * 100)
                if current_costs.total_cost > 0
                else 0
            )

            return {
                "current_cost": current_costs.total_cost,
                "projected_cost": projected_costs.total_cost,
                "potential_savings": potential_savings,
                "savings_percentage": savings_percentage,
                "changes_applied": changes,
                "cost_breakdown": {
                    "current": current_costs.to_dict(),
                    "projected": projected_costs.to_dict(),
                },
            }

        except Exception as e:
            logger.error(f"What-if analysis failed: {e}")
            return {
                "error": str(e),
                "current_cost": 0.0,
                "projected_cost": 0.0,
                "potential_savings": 0.0,
                "savings_percentage": 0.0,
                "changes_applied": changes,
            }

    def get_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        with self._lock:
            # Return a copy to prevent external modification
            return OptimizationMetrics(
                total_optimizations=self.metrics.total_optimizations,
                successful_optimizations=self.metrics.successful_optimizations,
                failed_optimizations=self.metrics.failed_optimizations,
                total_savings=self.metrics.total_savings,
                total_gb_saved=self.metrics.total_gb_saved,
                total_api_calls_reduced=self.metrics.total_api_calls_reduced,
                average_savings_percentage=self.metrics.average_savings_percentage,
            )

    def get_optimization_history(self, n: int = 10) -> List[OptimizationReport]:
        """Get recent optimization history."""
        with self._lock:
            return list(self.optimization_history)[-n:]

    def _create_optimization_plan(
        self, opportunities: List[Dict[str, Any]], strategy: OptimizationStrategy
    ) -> List[Dict[str, Any]]:
        """Create optimization plan based on strategy."""
        plan = []

        for opp in opportunities:
            # Determine if we should execute this opportunity based on strategy
            should_execute = False

            if strategy == OptimizationStrategy.AGGRESSIVE:
                should_execute = True
            elif strategy == OptimizationStrategy.COST_FOCUSED:
                should_execute = opp.get("priority") in ["high", "medium"]
            elif strategy == OptimizationStrategy.BALANCED:
                should_execute = opp.get("priority") == "high"
            elif strategy == OptimizationStrategy.CONSERVATIVE:
                should_execute = (
                    opp.get("priority") == "high" and opp.get("type") != "pruning"
                )
            elif strategy == OptimizationStrategy.PERFORMANCE_FOCUSED:
                # Only do optimizations that improve performance
                should_execute = opp.get("type") in [
                    "cdn_optimization",
                    "tier_migration",
                ]

            if should_execute:
                plan.append(
                    {
                        "type": opp["type"],
                        "action": opp["action"],
                        "priority": opp.get("priority", "medium"),
                        "description": self._get_action_description(opp),
                    }
                )

        return plan

    def _execute_storage_optimizations(
        self,
        plan: List[Dict[str, Any]],
        report: OptimizationReport,
        strategy: OptimizationStrategy,
    ) -> None:
        """Execute storage optimization plan with better error handling."""
        for step in plan:
            try:
                action_type = step["type"]

                if action_type == "deduplication":
                    if hasattr(self.memory, "dedup_engine") and hasattr(
                        self.memory.dedup_engine, "fold_ir_atoms"
                    ):
                        try:
                            self.memory.dedup_engine.fold_ir_atoms()
                            report.actions_taken.append("Deduplication completed")
                            # FIXED: Handle case where get_savings might not exist
                            if hasattr(self.memory.dedup_engine, "get_savings"):
                                gb_saved = self.memory.dedup_engine.get_savings()
                                if isinstance(gb_saved, (int, float)):
                                    self.metrics.total_gb_saved += float(gb_saved)
                        except Exception as e:
                            report.warnings.append(f"Deduplication failed: {str(e)}")

                elif action_type == "quantization":
                    if hasattr(self.memory, "quantizer") and hasattr(
                        self.memory.quantizer, "apply_rotational_quantization"
                    ):
                        try:
                            self.memory.quantizer.apply_rotational_quantization(
                                error_correction=True
                            )
                            report.actions_taken.append("Quantization applied")
                        except Exception as e:
                            report.warnings.append(f"Quantization failed: {str(e)}")

                elif action_type == "tier_migration":
                    if hasattr(self.memory, "tier_c") and hasattr(
                        self.memory.tier_c, "convert_to_binary_embeddings"
                    ):
                        try:
                            self.memory.tier_c.convert_to_binary_embeddings()
                            report.actions_taken.append(
                                "Converted to binary embeddings"
                            )
                        except Exception as e:
                            report.warnings.append(f"Tier migration failed: {str(e)}")

                elif action_type == "pruning":
                    if hasattr(self.memory, "tier_c") and hasattr(
                        self.memory.tier_c, "prune_low_utility"
                    ):
                        try:
                            threshold = (
                                0.1
                                if strategy == OptimizationStrategy.AGGRESSIVE
                                else 0.05
                            )
                            self.memory.tier_c.prune_low_utility(
                                threshold=threshold, disk_based=True
                            )
                            report.actions_taken.append(
                                f"Pruned low utility items (threshold={threshold})"
                            )
                        except Exception as e:
                            report.warnings.append(f"Pruning failed: {str(e)}")

                elif action_type == "cdn_optimization":
                    if hasattr(self.memory, "cdn") and hasattr(
                        self.memory.cdn, "optimize_cache_policy"
                    ):
                        try:
                            self.memory.cdn.optimize_cache_policy()
                            report.actions_taken.append("CDN cache policy optimized")
                        except Exception as e:
                            report.warnings.append(f"CDN optimization failed: {str(e)}")

            except Exception as e:
                error_msg = f"Action '{step['type']}' failed: {str(e)}"
                logger.error(error_msg)
                report.warnings.append(error_msg)

    def _get_action_description(self, opportunity: Dict[str, Any]) -> str:
        """Get human-readable description of action."""
        descriptions = {
            "deduplication": f"Remove duplicate data ({opportunity.get('potential_savings_gb', 0):.1f} GB potential savings)",
            "quantization": f"Apply quantization ({opportunity.get('potential_savings_gb', 0):.1f} GB potential savings)",
            "tier_migration": f"Migrate {opportunity.get('items', 0)} items to cold storage",
            "pruning": f"Prune {opportunity.get('items', 0)} low-utility items",
            "cdn_optimization": f"Optimize CDN (current hit rate: {opportunity.get('hit_rate', 0):.1%})",
        }
        return descriptions.get(opportunity["type"], f"Execute {opportunity['type']}")

    def _generate_optimization_id(self) -> str:
        """Generate unique optimization ID."""
        import hashlib

        content = f"opt_{time.time()}_{threading.current_thread().ident}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]

    def _trigger_budget_alert(self, level: str, status: Dict[str, Any]) -> None:
        """Trigger budget alert with proper error handling."""
        logger.warning(f"Budget alert ({level}): {status.get('status', 'unknown')}")

        for callback in self.on_budget_alert:
            try:
                callback(level, status)
            except Exception as e:
                logger.error(f"Budget alert callback failed: {e}")

    def _scheduler_loop(self) -> None:
        """Scheduler loop for automatic optimization."""
        while not self._shutdown:
            try:
                time.sleep(self.optimization_interval)

                if not self._shutdown:
                    logger.info("Running scheduled optimization")
                    self.optimize_full(strategy=self.default_strategy)

            except Exception as e:
                logger.error(f"Scheduled optimization failed: {e}")

    def shutdown(self) -> None:
        """Shutdown cost optimizer."""
        logger.info("Shutting down CostOptimizer...")
        self._shutdown = True

        if hasattr(self, "_scheduler_thread") and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)
            # Check if thread stopped
            if self._scheduler_thread.is_alive():
                logger.warning("Scheduler thread did not stop within timeout")

        logger.info("CostOptimizer shutdown complete")

    def __del__(self):
        """Destructor."""
        try:
            self.shutdown()
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )
