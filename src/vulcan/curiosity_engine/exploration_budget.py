"""
exploration_budget.py - Budget and resource management for Curiosity Engine
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources to monitor"""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    TIME = "time"
    GPU = "gpu"


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_bandwidth: float
    active_processes: int
    gpu_percent: float = 0.0
    gpu_memory: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_usage": self.disk_usage,
            "network_bandwidth": self.network_bandwidth,
            "active_processes": self.active_processes,
            "gpu_percent": self.gpu_percent,
            "gpu_memory": self.gpu_memory,
        }

    def get_bottleneck(self) -> Optional[ResourceType]:
        """Identify resource bottleneck"""
        try:
            resources = {
                ResourceType.CPU: self.cpu_percent,
                ResourceType.MEMORY: self.memory_percent,
                ResourceType.DISK: self.disk_usage,
            }

            # FIX: Handle empty resources
            if not resources:
                return None

            # Find highest usage
            max_resource = max(resources.items(), key=lambda x: x[1])

            if max_resource[1] > 0.8:
                return max_resource[0]

            return None
        except Exception as e:
            logger.warning("Error getting bottleneck: %s", e)
            return None


@dataclass
class CostHistory:
    """Historical cost data for calibration"""

    experiment_type: str
    predicted_cost: float
    actual_cost: float
    domain: str
    timestamp: float
    accuracy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def error(self) -> float:
        """Calculate prediction error"""
        try:
            if self.actual_cost > 0:
                return abs(self.predicted_cost - self.actual_cost) / self.actual_cost
            return 0.0
        except Exception as e:
            logger.warning("Error calculating error: %s", e)
            return 0.0

    def relative_error(self) -> float:
        """Calculate relative error"""
        try:
            if self.predicted_cost > 0:
                return (self.actual_cost - self.predicted_cost) / self.predicted_cost
            return 0.0
        except Exception as e:
            logger.warning("Error calculating relative error: %s", e)
            return 0.0


class BudgetTracker:
    """Tracks budget consumption and reservations - SEPARATED CONCERN"""

    def __init__(self, initial_budget: float, max_history: int = 1000):
        self.current_budget = initial_budget
        self.reserved_budget = 0.0
        self.total_consumed = 0.0
        self.total_reserved = 0.0

        # History tracking
        self.consumption_history = deque(maxlen=max_history)
        self.reservation_history = deque(maxlen=100)

        # Thread safety
        self.lock = threading.RLock()

    def get_available(self) -> float:
        """Get available budget"""
        with self.lock:
            return max(0.0, self.current_budget - self.reserved_budget)

    def consume(self, amount: float) -> bool:
        """Consume budget amount"""
        with self.lock:
            try:
                available = self.get_available()

                if available < amount:
                    return False

                self.current_budget -= amount
                self.total_consumed += amount

                # Track consumption
                self.consumption_history.append(
                    {
                        "timestamp": time.time(),
                        "cost": amount,
                        "remaining": self.current_budget,
                        "type": "consume",
                    }
                )

                return True
            except Exception as e:
                logger.error("Error consuming budget: %s", e)
                return False

    def reserve(self, amount: float, reservation_id: str) -> bool:
        """Reserve budget amount"""
        with self.lock:
            try:
                if self.get_available() < amount:
                    return False

                self.reserved_budget += amount
                self.total_reserved += amount

                # Track reservation
                self.reservation_history.append(
                    {"id": reservation_id, "amount": amount, "timestamp": time.time()}
                )

                return True
            except Exception as e:
                logger.error("Error reserving budget: %s", e)
                return False

    def release_reservation(self, amount: float, reservation_id: str):
        """Release reserved budget"""
        with self.lock:
            try:
                self.reserved_budget = max(0, self.reserved_budget - amount)

                # Track release
                self.reservation_history.append(
                    {
                        "id": reservation_id,
                        "amount": -amount,
                        "timestamp": time.time(),
                        "type": "release",
                    }
                )
            except Exception as e:
                logger.error("Error releasing reservation: %s", e)

    def add_budget(self, amount: float):
        """Add budget (for recovery)"""
        with self.lock:
            self.current_budget += amount

    def set_budget(self, amount: float):
        """Set budget to specific amount"""
        with self.lock:
            self.current_budget = amount

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        with self.lock:
            try:
                recent_consumption = []
                if len(self.consumption_history) > 10:
                    recent_consumption = [
                        c["cost"] for c in list(self.consumption_history)[-10:]
                    ]

                return {
                    "current_budget": self.current_budget,
                    "available": self.get_available(),
                    "reserved": self.reserved_budget,
                    "total_consumed": self.total_consumed,
                    "total_reserved": self.total_reserved,
                    "consumption_rate": np.mean(recent_consumption)
                    if recent_consumption
                    else 0,
                    "budget_utilization": 1.0
                    - (self.current_budget / max(self.current_budget, 1)),
                }
            except Exception as e:
                logger.error("Error getting statistics: %s", e)
                return {}


class BudgetRecovery:
    """Handles budget recovery over time - SEPARATED CONCERN"""

    def __init__(self, recovery_rate: float = 0.1, recovery_interval: float = 60.0):
        self.recovery_rate = recovery_rate
        self.recovery_interval = recovery_interval
        self.last_recovery_time = time.time()
        self.enabled = True
        self.lock = threading.RLock()

    def calculate_recovery(
        self, current_budget: float, base_allocation: float
    ) -> float:
        """Calculate budget recovery amount"""
        with self.lock:
            try:
                if not self.enabled:
                    return 0.0

                current_time = time.time()
                time_elapsed = current_time - self.last_recovery_time

                # Recover at specified interval
                if time_elapsed >= self.recovery_interval:
                    recovery_cycles = int(time_elapsed / self.recovery_interval)
                    recovery_amount = min(
                        base_allocation - current_budget,
                        base_allocation * self.recovery_rate * recovery_cycles,
                    )

                    if recovery_amount > 0:
                        self.last_recovery_time = current_time
                        return recovery_amount

                return 0.0
            except Exception as e:
                logger.error("Error calculating recovery: %s", e)
                return 0.0

    def set_enabled(self, enabled: bool):
        """Enable or disable recovery"""
        with self.lock:
            self.enabled = enabled

    def reset_timer(self):
        """Reset recovery timer"""
        with self.lock:
            self.last_recovery_time = time.time()


class LoadAdjuster:
    """Adjusts budget based on system load - SEPARATED CONCERN"""

    def __init__(self, adjustment_rate: float = 0.1):
        self.adjustment_rate = adjustment_rate
        self.load_threshold_low = 0.3
        self.load_threshold_high = 0.8
        self.allocation_history = deque(maxlen=1000)
        self.lock = threading.RLock()

    def calculate_adjustment(
        self,
        system_load: float,
        current_budget: float,
        min_budget: float,
        max_budget: float,
    ) -> float:
        """Calculate budget adjustment for system load"""
        with self.lock:
            try:
                # FIX: Clamp system load to valid range
                system_load = max(0.0, min(1.0, system_load))

                # EXAMINE: Evaluate system load
                if system_load < self.load_threshold_low:
                    # Low load - increase budget
                    adjustment = 1.0 + self.adjustment_rate
                elif system_load > self.load_threshold_high:
                    # High load - decrease budget
                    adjustment = 1.0 - self.adjustment_rate
                else:
                    # Normal load - slight adjustment based on exact value
                    range_size = self.load_threshold_high - self.load_threshold_low
                    normalized = (system_load - self.load_threshold_low) / range_size
                    adjustment = 1.0 + self.adjustment_rate * (0.5 - normalized)

                # SELECT & APPLY: Calculate new budget
                new_budget = current_budget * adjustment

                # Clamp to limits
                new_budget = max(min_budget, min(max_budget, new_budget))

                # REMEMBER: Track if significant change
                if abs(new_budget - current_budget) > 0.01:
                    self.allocation_history.append(
                        {
                            "timestamp": time.time(),
                            "old_budget": current_budget,
                            "new_budget": new_budget,
                            "system_load": system_load,
                            "adjustment": adjustment,
                        }
                    )

                return new_budget
            except Exception as e:
                logger.error("Error calculating adjustment: %s", e)
                return current_budget


class EfficiencyTracker:
    """Tracks efficiency and adjusts base allocation - SEPARATED CONCERN"""

    def __init__(self):
        self.efficiency_scores = deque(maxlen=100)
        self.lock = threading.RLock()

    def update(self, experiments_run: int, successes: int):
        """Update efficiency tracking"""
        with self.lock:
            try:
                if experiments_run > 0:
                    efficiency = successes / experiments_run
                    self.efficiency_scores.append(efficiency)
            except Exception as e:
                logger.error("Error updating efficiency: %s", e)

    def calculate_base_adjustment(
        self, current_base: float, min_base: float, max_base: float
    ) -> float:
        """Calculate base allocation adjustment based on efficiency"""
        with self.lock:
            try:
                if len(self.efficiency_scores) < 10:
                    return current_base

                avg_efficiency = np.mean(list(self.efficiency_scores)[-20:])

                if avg_efficiency > 0.7:
                    # High efficiency - can increase base
                    new_base = min(max_base, current_base * 1.05)
                elif avg_efficiency < 0.3:
                    # Low efficiency - reduce base
                    new_base = max(min_base, current_base * 0.95)
                else:
                    new_base = current_base

                return new_base
            except Exception as e:
                logger.error("Error calculating base adjustment: %s", e)
                return current_base

    def get_average_efficiency(self) -> float:
        """Get average efficiency"""
        with self.lock:
            try:
                if self.efficiency_scores:
                    return np.mean(self.efficiency_scores)
                return 0.5
            except Exception as e:
                logger.error("Error getting efficiency: %s", e)
                return 0.5


class DynamicBudget:
    """Adaptive exploration budget management - REFACTORED"""

    def __init__(self, base_allocation: float = 100.0, enable_recovery: bool = True):
        """
        Initialize dynamic budget

        Args:
            base_allocation: Base budget allocation
            enable_recovery: Enable automatic budget recovery
        """
        self.base_allocation = base_allocation
        self.enable_recovery = enable_recovery

        # Budget limits
        self.min_budget = max(1.0, base_allocation * 0.1)
        self.max_budget = base_allocation * 3.0

        # Components
        self.tracker = BudgetTracker(base_allocation)
        self.recovery = BudgetRecovery()
        self.load_adjuster = LoadAdjuster()
        self.efficiency_tracker = EfficiencyTracker()

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            "DynamicBudget initialized (refactored) with base allocation: %.2f",
            base_allocation,
        )

    def get_available(self) -> float:
        """Get available budget - REFACTORED"""
        with self._lock:
            try:
                # EXAMINE: Check for recovery
                if self.enable_recovery:
                    recovery_amount = self.recovery.calculate_recovery(
                        self.tracker.current_budget, self.base_allocation
                    )

                    # APPLY: Add recovered budget
                    if recovery_amount > 0:
                        self.tracker.add_budget(recovery_amount)
                        logger.debug("Recovered budget: %.2f", recovery_amount)

                # REMEMBER: Return available
                return self.tracker.get_available()
            except Exception as e:
                logger.error("Error getting available budget: %s", e)
                return 0.0

    def can_afford(self, cost: float) -> bool:
        """Check if can afford cost"""
        try:
            return self.get_available() >= cost
        except Exception as e:
            logger.error("Error checking affordability: %s", e)
            return False

    def consume(self, cost: float) -> bool:
        """Consume budget - DELEGATED"""
        with self._lock:
            try:
                success = self.tracker.consume(cost)
                if success:
                    logger.debug("Consumed budget: %.2f", cost)
                else:
                    logger.warning("Cannot afford cost %.2f", cost)
                return success
            except Exception as e:
                logger.error("Error consuming budget: %s", e)
                return False

    def reserve(
        self, cost: float, reservation_id: Optional[str] = None
    ) -> Optional[str]:
        """Reserve budget for future use - DELEGATED"""
        with self._lock:
            try:
                if reservation_id is None:
                    reservation_id = f"res_{time.time()}_{cost}"

                if self.tracker.reserve(cost, reservation_id):
                    logger.debug("Reserved budget: %.2f (ID: %s)", cost, reservation_id)
                    return reservation_id

                return None
            except Exception as e:
                logger.error("Error reserving budget: %s", e)
                return None

    def release_reservation(self, amount: float, reservation_id: Optional[str] = None):
        """Release reserved budget - DELEGATED"""
        with self._lock:
            try:
                if reservation_id is None:
                    reservation_id = "unknown"

                self.tracker.release_reservation(amount, reservation_id)
                logger.debug(
                    "Released reservation: %.2f (ID: %s)", amount, reservation_id
                )
            except Exception as e:
                logger.error("Error releasing reservation: %s", e)

    def adjust_for_load(self, system_load: float):
        """Adjust budget based on system load - DELEGATED"""
        with self._lock:
            try:
                new_budget = self.load_adjuster.calculate_adjustment(
                    system_load,
                    self.tracker.current_budget,
                    self.min_budget,
                    self.max_budget,
                )

                if abs(new_budget - self.tracker.current_budget) > 0.01:
                    logger.debug(
                        "Adjusted budget for load %.2f: %.2f -> %.2f",
                        system_load,
                        self.tracker.current_budget,
                        new_budget,
                    )
                    self.tracker.set_budget(new_budget)
            except Exception as e:
                logger.error("Error adjusting for load: %s", e)

    def update_efficiency(self, experiments_run: int, successes: int):
        """Update efficiency tracking - DELEGATED"""
        with self._lock:
            try:
                # Update tracker
                self.efficiency_tracker.update(experiments_run, successes)

                # Adjust base allocation
                new_base = self.efficiency_tracker.calculate_base_adjustment(
                    self.base_allocation, self.min_budget, self.max_budget
                )

                if abs(new_base - self.base_allocation) > 0.01:
                    logger.debug(
                        "Adjusted base allocation: %.2f -> %.2f",
                        self.base_allocation,
                        new_base,
                    )
                    self.base_allocation = new_base
            except Exception as e:
                logger.error("Error updating efficiency: %s", e)

    def get_statistics(self) -> Dict[str, Any]:
        """Get budget statistics - DELEGATED"""
        with self._lock:
            try:
                tracker_stats = self.tracker.get_statistics()

                return {
                    **tracker_stats,
                    "base_allocation": self.base_allocation,
                    "efficiency": self.efficiency_tracker.get_average_efficiency(),
                    "recovery_enabled": self.enable_recovery,
                }
            except Exception as e:
                logger.error("Error getting statistics: %s", e)
                return {}

    def reset(self):
        """Reset budget to base allocation"""
        with self._lock:
            try:
                self.tracker.set_budget(self.base_allocation)
                self.tracker.reserved_budget = 0.0
                self.recovery.reset_timer()
                logger.info(
                    "Budget reset to base allocation: %.2f", self.base_allocation
                )
            except Exception as e:
                logger.error("Error resetting budget: %s", e)


class ResourceSampler:
    """Samples system resources - SEPARATED CONCERN"""

    def __init__(self, enable_gpu: bool = False):
        self.enable_gpu = enable_gpu
        self.gpu_available = False
        self.gputil = None

        if enable_gpu:
            self._setup_gpu_monitoring()

    def sample(self) -> ResourceSnapshot:
        """Sample current resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0

            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0

            # Disk usage
            disk_usage = psutil.disk_usage("/").percent / 100.0

            # Network bandwidth (simplified)
            net_io = psutil.net_io_counters()
            network_bandwidth = (net_io.bytes_sent + net_io.bytes_recv) / (
                1024 * 1024
            )  # MB

            # Active processes
            active_processes = len(psutil.pids())

            # GPU usage if available
            gpu_percent = 0.0
            gpu_memory = 0.0
            if self.gpu_available:
                gpu_percent, gpu_memory = self._get_gpu_usage()

        except Exception as e:
            logger.warning("Error getting resource snapshot: %s", e)
            # Return default values
            cpu_percent = 0.5
            memory_percent = 0.5
            disk_usage = 0.5
            network_bandwidth = 0.0
            active_processes = 0
            gpu_percent = 0.0
            gpu_memory = 0.0

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage=disk_usage,
            network_bandwidth=network_bandwidth,
            active_processes=active_processes,
            gpu_percent=gpu_percent,
            gpu_memory=gpu_memory,
        )

    def _setup_gpu_monitoring(self):
        """Setup GPU monitoring if available"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            self.gpu_available = len(gpus) > 0
            self.gputil = GPUtil
            if self.gpu_available:
                logger.info("GPU monitoring enabled")
        except ImportError:
            self.gpu_available = False
            self.gputil = None
            logger.debug("GPU monitoring not available (GPUtil not installed)")
        except Exception as e:
            self.gpu_available = False
            self.gputil = None
            logger.warning("Error setting up GPU monitoring: %s", e)

    def _get_gpu_usage(self) -> Tuple[float, float]:
        """Get GPU usage if available"""
        # FIX: Better exception handling
        if not self.gpu_available or not self.gputil:
            return 0.0, 0.0

        try:
            gpus = self.gputil.getGPUs()
            if gpus and len(gpus) > 0:
                gpu = gpus[0]  # Use first GPU
                load = getattr(gpu, "load", 0.0)
                memory = getattr(gpu, "memoryUtil", 0.0)
                return float(load), float(memory)
        except Exception as e:
            logger.debug("Error getting GPU usage: %s", e)

        return 0.0, 0.0


class ResourcePredictor:
    """Predicts future resource usage - SEPARATED CONCERN"""

    def __init__(self, history_size: int = 100):
        self.history = deque(maxlen=history_size)
        self.trend_window = min(10, history_size // 2)
        self.lock = threading.RLock()

    def add_snapshot(self, snapshot: ResourceSnapshot):
        """Add snapshot to history"""
        with self.lock:
            self.history.append(snapshot)

    def predict_load(
        self, horizon_minutes: float = 10, sampling_interval: float = 1.0
    ) -> Tuple[float, float]:
        """Predict future system load"""
        with self.lock:
            try:
                if len(self.history) < self.trend_window:
                    # Not enough history
                    if self.history:
                        latest = self.history[-1]
                        current_load = (
                            latest.cpu_percent * 0.5 + latest.memory_percent * 0.5
                        )
                        return current_load, 0.3
                    return 0.5, 0.3

                # EXAMINE: Extract recent trends
                recent = list(self.history)[-self.trend_window :]

                cpu_values = np.array([s.cpu_percent for s in recent])
                memory_values = np.array([s.memory_percent for s in recent])

                # SELECT: Calculate trends
                cpu_trend = self._calculate_trend(cpu_values)
                memory_trend = self._calculate_trend(memory_values)

                # APPLY: Project forward
                steps = int(horizon_minutes * 60 / sampling_interval)

                projected_cpu = cpu_values[-1] + cpu_trend * steps
                projected_memory = memory_values[-1] + memory_trend * steps

                # Clamp to valid range
                projected_cpu = max(0.0, min(1.0, projected_cpu))
                projected_memory = max(0.0, min(1.0, projected_memory))

                # Combine predictions
                predicted_load = projected_cpu * 0.5 + projected_memory * 0.5

                # REMEMBER: Calculate confidence
                cpu_variance = np.var(cpu_values)
                memory_variance = np.var(memory_values)
                avg_variance = (cpu_variance + memory_variance) / 2

                confidence = max(0.2, 1.0 - avg_variance)
                confidence *= min(1.0, len(self.history) / 20)

                return min(1.0, predicted_load), confidence
            except Exception as e:
                logger.error("Error predicting load: %s", e)
                return 0.5, 0.3

    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend"""
        try:
            if len(values) < 2:
                return 0.0

            x = np.arange(len(values))
            len(x)
            x_mean = np.mean(x)
            y_mean = np.mean(values)

            numerator = np.sum((x - x_mean) * (values - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            # FIX: Handle division by zero
            if abs(denominator) < 1e-10:
                return 0.0

            slope = numerator / denominator

            return slope
        except Exception as e:
            logger.warning("Error calculating trend: %s", e)
            return 0.0


class ResourceAdvisor:
    """Provides resource-based recommendations - SEPARATED CONCERN"""

    def __init__(self):
        self.critical_thresholds = {
            ResourceType.CPU: 0.9,
            ResourceType.MEMORY: 0.85,
            ResourceType.DISK: 0.95,
            ResourceType.GPU: 0.9,
        }

        self.warning_thresholds = {
            ResourceType.CPU: 0.7,
            ResourceType.MEMORY: 0.7,
            ResourceType.DISK: 0.8,
            ResourceType.GPU: 0.75,
        }

    def recommend_adjustment(
        self, snapshot: ResourceSnapshot, predicted_load: float, confidence: float
    ) -> Dict[str, Any]:
        """Generate recommendation based on resources"""

        try:
            # EXAMINE: Analyze current state
            current_load = snapshot.cpu_percent * 0.5 + snapshot.memory_percent * 0.5
            bottleneck = snapshot.get_bottleneck()

            recommendation = {
                "current_load": current_load,
                "predicted_load": predicted_load,
                "confidence": confidence,
                "action": "maintain",
                "adjustment_factor": 1.0,
                "bottleneck": bottleneck.value if bottleneck else None,
            }

            # SELECT: Determine action based on thresholds
            if snapshot.cpu_percent > self.critical_thresholds[ResourceType.CPU]:
                recommendation["action"] = "reduce"
                recommendation["adjustment_factor"] = 0.5
                recommendation["reason"] = "Critical CPU usage"

            elif (
                snapshot.memory_percent > self.critical_thresholds[ResourceType.MEMORY]
            ):
                recommendation["action"] = "reduce"
                recommendation["adjustment_factor"] = 0.6
                recommendation["reason"] = "Critical memory usage"

            elif snapshot.disk_usage > self.critical_thresholds[ResourceType.DISK]:
                recommendation["action"] = "reduce"
                recommendation["adjustment_factor"] = 0.7
                recommendation["reason"] = "Critical disk usage"

            elif snapshot.cpu_percent > self.warning_thresholds[ResourceType.CPU]:
                recommendation["action"] = "reduce"
                recommendation["adjustment_factor"] = 0.8
                recommendation["reason"] = "High CPU usage"

            elif snapshot.memory_percent > self.warning_thresholds[ResourceType.MEMORY]:
                recommendation["action"] = "reduce"
                recommendation["adjustment_factor"] = 0.85
                recommendation["reason"] = "High memory usage"

            elif current_load < 0.3 and predicted_load < 0.4 and confidence > 0.6:
                recommendation["action"] = "increase"
                recommendation["adjustment_factor"] = 1.2
                recommendation["reason"] = "Low resource utilization"

            return recommendation
        except Exception as e:
            logger.error("Error generating recommendation: %s", e)
            return {
                "current_load": 0.5,
                "predicted_load": 0.5,
                "confidence": 0.5,
                "action": "maintain",
                "adjustment_factor": 1.0,
            }


class ResourceMonitor:
    """Monitors system resources - REFACTORED"""

    def __init__(
        self,
        sampling_interval: float = 1.0,
        history_size: int = 100,
        enable_gpu: bool = False,
    ):
        """
        Initialize resource monitor

        Args:
            sampling_interval: Interval between samples (seconds)
            history_size: Size of history buffer
            enable_gpu: Enable GPU monitoring if available
        """
        self.sampling_interval = sampling_interval

        # Components
        self.sampler = ResourceSampler(enable_gpu)
        self.predictor = ResourcePredictor(history_size)
        self.advisor = ResourceAdvisor()

        # Caching
        self.last_sample_time = 0
        self._current_load_cache = None
        self._cache_time = 0
        self._cache_ttl = 0.5

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ResourceMonitor initialized (refactored)")

    def get_current_load(self) -> float:
        """Get current system load - REFACTORED"""
        with self.lock:
            try:
                # FIX: Atomic cache check
                current_time = time.time()
                if (
                    self._current_load_cache is not None
                    and current_time - self._cache_time < self._cache_ttl
                ):
                    return self._current_load_cache

                # EXAMINE: Sample if needed
                if current_time - self.last_sample_time >= self.sampling_interval:
                    snapshot = self.sampler.sample()
                    self.predictor.add_snapshot(snapshot)
                    self.last_sample_time = current_time

                # Get latest snapshot
                if self.predictor.history:
                    latest = self.predictor.history[-1]

                    # SELECT & APPLY: Calculate load
                    load = (
                        latest.cpu_percent * 0.35
                        + latest.memory_percent * 0.35
                        + latest.disk_usage * 0.15
                        + min(1.0, latest.network_bandwidth / 100) * 0.05
                    )

                    if self.sampler.gpu_available and latest.gpu_percent > 0:
                        load = load * 0.9 + latest.gpu_percent * 0.1

                    # FIX: Ensure load is in valid range
                    result = max(0.0, min(1.0, load))
                else:
                    result = 0.5

                # REMEMBER: Cache result atomically
                self._current_load_cache = result
                self._cache_time = current_time

                return result
            except Exception as e:
                logger.error("Error getting current load: %s", e)
                return 0.5

    def predict_future_load(self, horizon_minutes: float = 10) -> Tuple[float, float]:
        """Predict future system load - DELEGATED"""
        return self.predictor.predict_load(horizon_minutes, self.sampling_interval)

    def recommend_budget_adjustment(self) -> Dict[str, Any]:
        """Recommend budget adjustments - DELEGATED"""
        with self.lock:
            try:
                # Get current snapshot
                snapshot = self.sampler.sample()
                self.predictor.add_snapshot(snapshot)

                # Get prediction
                predicted_load, confidence = self.predictor.predict_load()

                # Get recommendation
                return self.advisor.recommend_adjustment(
                    snapshot, predicted_load, confidence
                )
            except Exception as e:
                logger.error("Error recommending adjustment: %s", e)
                return {}

    def get_resource_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot - DELEGATED"""
        return self.sampler.sample()

    def get_resource_trends(self) -> Dict[str, Dict[str, float]]:
        """Get resource usage trends"""
        with self.lock:
            try:
                if len(self.predictor.history) < 2:
                    return {}

                recent = list(self.predictor.history)[
                    -min(20, len(self.predictor.history)) :
                ]

                trends = {}

                # CPU trend
                cpu_values = [s.cpu_percent for s in recent]
                trends["cpu"] = {
                    "current": cpu_values[-1],
                    "average": np.mean(cpu_values),
                    "trend": self.predictor._calculate_trend(np.array(cpu_values)),
                    "variance": np.var(cpu_values),
                }

                # Memory trend
                memory_values = [s.memory_percent for s in recent]
                trends["memory"] = {
                    "current": memory_values[-1],
                    "average": np.mean(memory_values),
                    "trend": self.predictor._calculate_trend(np.array(memory_values)),
                    "variance": np.var(memory_values),
                }

                # Disk trend
                disk_values = [s.disk_usage for s in recent]
                trends["disk"] = {
                    "current": disk_values[-1],
                    "average": np.mean(disk_values),
                    "trend": self.predictor._calculate_trend(np.array(disk_values)),
                    "variance": np.var(disk_values),
                }

                return trends
            except Exception as e:
                logger.error("Error getting trends: %s", e)
                return {}


class CostCalibrator:
    """Calibrates cost estimates from history - SEPARATED CONCERN"""

    def __init__(self, learning_rate: float = 0.1, adaptation_threshold: float = 0.2):
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.calibration_factor = 1.0
        self.confidence_intervals = {}  # Use regular dict with default handling
        self.lock = threading.RLock()

    def get_confidence_interval(self, key: str) -> Tuple[float, float]:
        """Get confidence interval with default"""
        with self.lock:
            return self.confidence_intervals.get(key, (0.8, 1.2))

    def calibrate(
        self,
        historical_costs: List[CostHistory],
        base_costs: Dict[str, float],
        domain_costs: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calibrate costs from history"""
        with self.lock:
            try:
                # FIX: Validate input
                if not historical_costs or not isinstance(historical_costs, list):
                    return base_costs, domain_costs

                # EXAMINE: Group by type and domain
                type_groups = defaultdict(list)
                domain_groups = defaultdict(list)

                for record in historical_costs:
                    type_groups[record.experiment_type].append(record)
                    domain_groups[record.domain].append(record)

                # SELECT & APPLY: Calibrate each experiment type
                updated_base = base_costs.copy()
                for exp_type, records in type_groups.items():
                    if len(records) < 3:
                        continue

                    errors = [r.error() for r in records]
                    relative_errors = [r.relative_error() for r in records]

                    avg_error = np.mean(errors)
                    avg_relative = np.mean(relative_errors)

                    if exp_type in updated_base:
                        if abs(avg_relative) > self.adaptation_threshold:
                            adjustment = 1.0 + avg_relative * self.learning_rate
                            updated_base[exp_type] *= adjustment
                            updated_base[exp_type] = max(1.0, updated_base[exp_type])

                    # Update confidence intervals
                    std_error = np.std(errors)
                    self.confidence_intervals[exp_type] = (
                        max(0.5, 1 - 2 * std_error),
                        min(2.0, 1 + 2 * std_error),
                    )

                # Calibrate domain costs
                updated_domain = domain_costs.copy()
                for domain, records in domain_groups.items():
                    if len(records) < 2:
                        continue

                    relative_errors = [r.relative_error() for r in records]
                    avg_relative = np.mean(relative_errors)

                    if abs(avg_relative) > self.adaptation_threshold:
                        adjustment = 1.0 + avg_relative * self.learning_rate
                        updated_domain[domain] = (
                            updated_domain.get(domain, 1.0) * adjustment
                        )
                        updated_domain[domain] = max(
                            0.5, min(2.0, updated_domain[domain])
                        )

                # REMEMBER: Update overall calibration
                all_errors = [r.error() for r in historical_costs]
                if all_errors:
                    avg_error = np.mean(all_errors)

                    if avg_error > self.adaptation_threshold:
                        self.calibration_factor *= 1 - min(
                            0.2, avg_error * self.learning_rate
                        )
                    elif avg_error < self.adaptation_threshold / 2:
                        self.calibration_factor *= 1 + min(0.1, self.learning_rate)

                    self.calibration_factor = max(
                        0.5, min(2.0, self.calibration_factor)
                    )

                return updated_base, updated_domain
            except Exception as e:
                logger.error("Error calibrating: %s", e)
                return base_costs, domain_costs


class CostEstimator:
    """Estimates experiment costs with learning - REFACTORED"""

    def __init__(self):
        """Initialize cost estimator"""
        # Base costs by experiment type
        self.base_costs = {
            "decomposition": 10.0,
            "causal": 20.0,
            "transfer": 15.0,
            "synthetic": 5.0,
            "exploratory": 8.0,
            "validation": 12.0,
            "iterative": 10.0,
            "ablation": 7.0,
        }

        # FIX: Use regular dict for domain costs with thread safety
        self.domain_costs = {}
        self.domain_novelty = {}

        # Components
        self.calibrator = CostCalibrator()

        # Historical data
        self.cost_history = deque(maxlen=1000)

        # Statistics
        self.prediction_errors = deque(maxlen=100)
        self.domain_statistics = {}

        # Caching
        self._estimate_cache = {}
        self._cache_ttl = 60

        # Thread safety
        self.lock = threading.RLock()

        logger.info("CostEstimator initialized (refactored)")

    def _get_domain_cost(self, domain: str) -> float:
        """Thread-safe domain cost retrieval"""
        with self.lock:
            return self.domain_costs.get(domain, 1.0)

    def _get_domain_novelty(self, domain: str) -> float:
        """Thread-safe domain novelty retrieval"""
        with self.lock:
            return self.domain_novelty.get(domain, 1.0)

    def estimate_learning_cost(
        self, gap_type: str, complexity: float, priority: float, domain: str
    ) -> float:
        """
        Estimate cost of learning from gap - REFACTORED

        NOTE: Removed @lru_cache decorator to avoid hashability issues with domain parameter
        """
        with self.lock:
            try:
                # EXAMINE: Get base cost
                base_cost = self.base_costs.get(gap_type, 10.0)

                # SELECT: Apply adjustments
                base_cost *= 1 + complexity  # Complexity adjustment
                base_cost *= 0.5 + priority  # Priority adjustment

                # Domain adjustments
                domain_multiplier = self._get_domain_cost(domain)
                novelty_factor = self._get_domain_novelty(domain)
                base_cost *= domain_multiplier * novelty_factor

                # APPLY: Apply calibration
                calibrated_cost = base_cost * self.calibrator.calibration_factor

                # Add uncertainty
                lower, upper = self.calibrator.get_confidence_interval(gap_type)
                final_cost = calibrated_cost * np.random.uniform(lower, upper)

                # REMEMBER: Ensure minimum cost
                return max(1.0, final_cost)
            except Exception as e:
                logger.error("Error estimating learning cost: %s", e)
                return 10.0

    def estimate_experiment_cost(self, experiment) -> float:
        """Estimate cost of running experiment - REFACTORED"""
        with self.lock:
            try:
                # Check cache
                cache_key = getattr(experiment, "experiment_id", None)
                if cache_key and cache_key in self._estimate_cache:
                    cached_time, cached_cost = self._estimate_cache[cache_key]
                    if time.time() - cached_time < self._cache_ttl:
                        return cached_cost

                # EXAMINE: Extract experiment properties
                complexity = getattr(experiment, "complexity", 0.5)
                base_cost = complexity * 10.0

                # SELECT: Apply adjustments
                if hasattr(experiment, "timeout"):
                    timeout_factor = 1 + min(2.0, experiment.timeout / 30.0)
                    base_cost *= timeout_factor

                if hasattr(experiment, "experiment_type"):
                    type_str = str(experiment.experiment_type).split(".")[-1].lower()
                    type_cost = self.base_costs.get(type_str, 10.0)
                    type_multiplier = type_cost / 10.0
                    base_cost *= type_multiplier

                if hasattr(experiment, "iteration"):
                    iteration_factor = 1 + min(1.0, experiment.iteration * 0.2)
                    base_cost *= iteration_factor

                if hasattr(experiment, "gap") and hasattr(experiment.gap, "domain"):
                    domain = experiment.gap.domain
                    domain_multiplier = self._get_domain_cost(domain)
                    novelty_factor = self._get_domain_novelty(domain)
                    base_cost *= domain_multiplier * novelty_factor

                # APPLY: Apply calibration
                calibrated_cost = base_cost * self.calibrator.calibration_factor

                if hasattr(experiment, "parameters"):
                    params = experiment.parameters
                    if "sample_size" in params:
                        sample_factor = 1 + min(1.0, params["sample_size"] / 100 * 0.1)
                        calibrated_cost *= sample_factor
                    if "iterations" in params:
                        iter_factor = 1 + min(1.0, params["iterations"] / 100 * 0.1)
                        calibrated_cost *= iter_factor

                final_cost = max(1.0, calibrated_cost)

                # REMEMBER: Cache result
                if cache_key:
                    self._estimate_cache[cache_key] = (time.time(), final_cost)

                    # FIX: Cleanup cache safely
                    if len(self._estimate_cache) > 200:
                        # Copy keys to avoid modification during iteration
                        all_keys = list(self._estimate_cache.keys())
                        sorted_keys = sorted(
                            all_keys, key=lambda k: self._estimate_cache[k][0]
                        )
                        for key in sorted_keys[:50]:
                            self._estimate_cache.pop(key, None)

                return final_cost
            except Exception as e:
                logger.error("Error estimating experiment cost: %s", e)
                return 10.0

    def calibrate_from_history(
        self, historical_costs: Optional[List[CostHistory]] = None
    ):
        """Calibrate estimator from historical data - DELEGATED"""
        with self.lock:
            try:
                if historical_costs is None:
                    historical_costs = list(self.cost_history)

                if not historical_costs:
                    return

                # Use calibrator
                self.base_costs, updated_domain_costs = self.calibrator.calibrate(
                    historical_costs, self.base_costs, self.domain_costs.copy()
                )

                # Update domain costs
                self.domain_costs.update(updated_domain_costs)

                # Track prediction errors
                all_errors = [r.error() for r in historical_costs]
                self.prediction_errors.extend(all_errors)

                # Clear estimate cache after calibration
                self._estimate_cache.clear()

                logger.info(
                    "Calibrated from %d historical records (avg error: %.2f%%)",
                    len(historical_costs),
                    np.mean(all_errors) * 100 if all_errors else 0,
                )
            except Exception as e:
                logger.error("Error calibrating from history: %s", e)

    def adjust_for_domain_novelty(self, domain: str, base_cost: float) -> float:
        """Adjust cost for domain novelty"""
        with self.lock:
            try:
                current_novelty = self._get_domain_novelty(domain)

                # Decay novelty over time
                decay_rate = 0.95
                self.domain_novelty[domain] = max(0.5, current_novelty * decay_rate)

                # Apply novelty adjustment
                adjusted_cost = base_cost * current_novelty

                return adjusted_cost
            except Exception as e:
                logger.error("Error adjusting for novelty: %s", e)
                return base_cost

    def update_from_actual(
        self,
        experiment_type: str,
        predicted: float,
        actual: float,
        domain: Optional[str] = None,
    ):
        """Update estimator with actual cost"""
        with self.lock:
            try:
                # Create history record
                record = CostHistory(
                    experiment_type=experiment_type,
                    predicted_cost=predicted,
                    actual_cost=actual,
                    domain=domain or "unknown",
                    timestamp=time.time(),
                    accuracy=1.0 - abs(predicted - actual) / max(actual, 1.0),
                )

                self.cost_history.append(record)

                # Update domain statistics
                if domain:
                    if domain not in self.domain_statistics:
                        self.domain_statistics[domain] = {"count": 0, "total_error": 0}

                    stats = self.domain_statistics[domain]
                    stats["count"] += 1
                    stats["total_error"] += record.error()

                    # Update domain novelty if error is large
                    if record.error() > self.calibrator.adaptation_threshold:
                        current_novelty = self._get_domain_novelty(domain)
                        if actual > predicted:
                            self.domain_novelty[domain] = min(
                                2.0, current_novelty * 1.1
                            )
                        else:
                            self.domain_novelty[domain] = max(
                                0.5, current_novelty * 0.9
                            )

                # Trigger recalibration periodically
                if len(self.cost_history) % 20 == 0:
                    recent_history = list(self.cost_history)[-50:]
                    self.calibrate_from_history(recent_history)
            except Exception as e:
                logger.error("Error updating from actual: %s", e)

    def get_accuracy_stats(self) -> Dict[str, float]:
        """Get accuracy statistics"""
        with self.lock:
            try:
                if not self.prediction_errors:
                    return {
                        "mean_error": 0,
                        "std_error": 0,
                        "median_error": 0,
                        "max_error": 0,
                        "confidence": 0.5,
                        "calibration_factor": self.calibrator.calibration_factor,
                    }

                errors = list(self.prediction_errors)

                return {
                    "mean_error": np.mean(errors),
                    "std_error": np.std(errors),
                    "median_error": np.median(errors),
                    "max_error": np.max(errors),
                    "min_error": np.min(errors),
                    "confidence": max(0, 1 - np.mean(errors)),
                    "calibration_factor": self.calibrator.calibration_factor,
                    "total_predictions": len(self.cost_history),
                }
            except Exception as e:
                logger.error("Error getting accuracy stats: %s", e)
                return {}

    def get_domain_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics by domain"""
        with self.lock:
            try:
                stats = {}

                for domain, domain_stats in self.domain_statistics.items():
                    if domain_stats["count"] > 0:
                        stats[domain] = {
                            "count": domain_stats["count"],
                            "avg_error": domain_stats["total_error"]
                            / domain_stats["count"],
                            "cost_multiplier": self._get_domain_cost(domain),
                            "novelty": self._get_domain_novelty(domain),
                        }

                return stats
            except Exception as e:
                logger.error("Error getting domain statistics: %s", e)
                return {}
