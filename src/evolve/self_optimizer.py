# self_optimizer.py
"""
Self Optimizer for Graphix IR
Autonomous performance optimization and resource management
"""

import asyncio
import logging
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from .security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    latency_ms: float
    throughput_ops: float
    memory_mb: float
    cpu_percent: float
    gpu_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: float = field(default_factory=time.time)

    def score(self) -> float:
        """Calculate overall performance score."""
        # Lower latency is better
        latency_score = 100 / (1 + self.latency_ms)
        # Higher throughput is better
        throughput_score = min(100, self.throughput_ops)
        # Lower memory usage is better
        memory_score = 100 / (1 + self.memory_mb / 100)
        # Lower CPU usage is better
        cpu_score = 100 - self.cpu_percent

        # Weighted average
        return (
            latency_score * 0.3
            + throughput_score * 0.3
            + memory_score * 0.2
            + cpu_score * 0.2
        )


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""

    name: str
    enabled: bool = True
    priority: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0

    def effectiveness(self) -> float:
        """Calculate strategy effectiveness."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown effectiveness
        return self.success_count / total


class SelfOptimizer:
    """
    Autonomous self-optimization system for continuous performance improvement.
    """

    # Security: Define limits
    MAX_MEMORY_MB = 16000
    MIN_MEMORY_MB = 100
    MAX_CPU_PERCENT = 95
    MAX_WORKERS = 32
    MIN_WORKERS = 1
    MAX_CACHE_SIZE = 100000
    MIN_CACHE_SIZE = 10
    MAX_BATCH_SIZE = 512
    MIN_BATCH_SIZE = 1

    def __init__(
        self,
        target_latency_ms: float = 100,
        target_memory_mb: float = 1000,
        optimization_interval_s: float = 60,
        enable_auto_tune: bool = True,
    ):
        """Initialize self optimizer with validated parameters."""
        # Validate and bound parameters
        self.target_latency_ms = max(1, min(10000, target_latency_ms))
        self.target_memory_mb = max(
            self.MIN_MEMORY_MB, min(self.MAX_MEMORY_MB, target_memory_mb)
        )
        self.optimization_interval_s = max(1, min(3600, optimization_interval_s))
        self.enable_auto_tune = bool(enable_auto_tune)

        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = None
        self.baseline_metrics = None

        # Optimization strategies with safe defaults
        self.strategies = {
            "caching": OptimizationStrategy(
                "caching", parameters={"cache_size": 1000, "ttl_seconds": 300}
            ),
            "batching": OptimizationStrategy(
                "batching", parameters={"batch_size": 32, "timeout_ms": 10}
            ),
            "parallelization": OptimizationStrategy(
                "parallelization", parameters={"num_workers": 4, "chunk_size": 100}
            ),
            "pruning": OptimizationStrategy(
                "pruning", parameters={"threshold": 0.01, "sparsity": 0.1}
            ),
            "quantization": OptimizationStrategy(
                "quantization", parameters={"bits": 8, "symmetric": True}
            ),
            "compilation": OptimizationStrategy(
                "compilation", parameters={"backend": "jit", "optimize": True}
            ),
        }

        # Resource limits
        self.resource_limits = {
            "max_memory_mb": self.MAX_MEMORY_MB,
            "max_cpu_percent": self.MAX_CPU_PERCENT,
            "max_gpu_percent": 90,
            "min_cache_size": self.MIN_CACHE_SIZE,
            "max_cache_size": self.MAX_CACHE_SIZE,
        }

        # Optimization state
        self.is_optimizing = False
        self.optimization_thread = None
        self.stop_event = threading.Event()

        # Cache management with bounded size
        self.cache = {}
        self.cache_stats = defaultdict(int)
        self.cache_access_order = deque(maxlen=self.MAX_CACHE_SIZE)

        # Model parameters for auto-tuning with validation
        self.tunable_parameters = {
            "batch_size": {
                "min": self.MIN_BATCH_SIZE,
                "max": self.MAX_BATCH_SIZE,
                "current": 32,
                "step": 1,
                "type": int,
            },
            "num_workers": {
                "min": self.MIN_WORKERS,
                "max": self.MAX_WORKERS,
                "current": 4,
                "step": 1,
                "type": int,
            },
            "cache_size": {
                "min": self.MIN_CACHE_SIZE,
                "max": self.MAX_CACHE_SIZE,
                "current": 1000,
                "step": 100,
                "type": int,
            },
            "learning_rate": {
                "min": 0.0001,
                "max": 0.1,
                "current": 0.001,
                "step": 0.0001,
                "type": float,
            },
        }

        # Predefined safe evaluation functions (no eval!)
        self.parameter_evaluators = {
            "batch_size": self._evaluate_batch_size,
            "num_workers": self._evaluate_num_workers,
            "cache_size": self._evaluate_cache_size,
            "learning_rate": self._evaluate_learning_rate,
        }

        # Thread pool for async operations
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)

        logger.info(f"SelfOptimizer initialized with auto-tune={enable_auto_tune}")

    def start(self):
        """Start continuous optimization loop."""
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            self.stop_event.clear()
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop, daemon=True
            )
            self.optimization_thread.start()
            logger.info("SelfOptimizer started")

    def stop(self):
        """Stop optimization loop."""
        self.stop_event.set()
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("SelfOptimizer stopped")

    async def start_async(self):
        """Async version of start."""
        if not hasattr(self, "_async_task") or self._async_task.done():
            self._async_task = asyncio.create_task(self._async_optimization_loop())
            logger.info("SelfOptimizer async started")

    async def stop_async(self):
        """Async version of stop."""
        self.stop_event.set()
        if hasattr(self, "_async_task"):
            await self._async_task
        logger.info("SelfOptimizer async stopped")

    def _optimization_loop(self):
        """Main optimization loop (synchronous)."""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.current_metrics = metrics

                # Initialize baseline if needed
                if self.baseline_metrics is None:
                    self.baseline_metrics = metrics

                # Check if optimization needed
                if self._should_optimize(metrics):
                    self._run_optimization_cycle()

                # Auto-tune parameters
                if self.enable_auto_tune:
                    self._auto_tune_parameters()

                # Clean up resources
                self._cleanup_resources()

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

            # Wait for next cycle
            self.stop_event.wait(self.optimization_interval_s)

    async def _async_optimization_loop(self):
        """Async optimization loop for better concurrency."""
        while not self.stop_event.is_set():
            try:
                # Collect metrics asynchronously
                metrics = await asyncio.to_thread(self._collect_metrics)
                self.metrics_history.append(metrics)
                self.current_metrics = metrics

                # Initialize baseline if needed
                if self.baseline_metrics is None:
                    self.baseline_metrics = metrics

                # Check if optimization needed
                if self._should_optimize(metrics):
                    await asyncio.to_thread(self._run_optimization_cycle)

                # Auto-tune parameters
                if self.enable_auto_tune:
                    await asyncio.to_thread(self._auto_tune_parameters)

                # Clean up resources
                await asyncio.to_thread(self._cleanup_resources)

            except Exception as e:
                logger.error(f"Async optimization loop error: {e}")

            # Wait for next cycle
            await asyncio.sleep(self.optimization_interval_s)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            process = psutil.Process()

            return PerformanceMetrics(
                latency_ms=self._measure_latency(),
                throughput_ops=self._measure_throughput(),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                cpu_percent=process.cpu_percent(interval=0.1),
                gpu_percent=self._get_gpu_usage(),
                cache_hits=self.cache_stats["hits"],
                cache_misses=self.cache_stats["misses"],
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return safe defaults
            return PerformanceMetrics(
                latency_ms=100.0,
                throughput_ops=100.0,
                memory_mb=1000.0,
                cpu_percent=50.0,
            )

    def _measure_latency(self) -> float:
        """Measure current operation latency."""
        if len(self.metrics_history) > 0:
            recent = list(self.metrics_history)[-min(10, len(self.metrics_history)) :]
            latencies = [m.latency_ms for m in recent if m.latency_ms > 0]
            if latencies:
                return np.mean(latencies)
        return 100.0

    def _measure_throughput(self) -> float:
        """Measure current throughput."""
        if len(self.metrics_history) > 0:
            recent = list(self.metrics_history)[-min(10, len(self.metrics_history)) :]
            throughputs = [m.throughput_ops for m in recent if m.throughput_ops > 0]
            if throughputs:
                return np.mean(throughputs)
        return 100.0

    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Determine if optimization is needed."""
        if not self.baseline_metrics:
            return False

        # Check against targets
        needs_optimization = (
            metrics.latency_ms > self.target_latency_ms * 1.1
            or metrics.memory_mb > self.target_memory_mb * 1.1
            or metrics.cpu_percent > self.resource_limits["max_cpu_percent"]
            or metrics.score() < self.baseline_metrics.score() * 0.9
        )

        return needs_optimization

    def _run_optimization_cycle(self):
        """Run one optimization cycle."""
        self.is_optimizing = True
        logger.info("Starting optimization cycle")

        try:
            # Try each strategy
            improvements = []

            for name, strategy in self.strategies.items():
                if not strategy.enabled:
                    continue

                # Apply strategy
                before_score = (
                    self.current_metrics.score() if self.current_metrics else 0
                )
                success = self._apply_strategy(strategy)

                if success:
                    # Measure improvement
                    after_metrics = self._collect_metrics()
                    after_score = after_metrics.score()
                    improvement = after_score - before_score

                    improvements.append((strategy, improvement))

                    if improvement > 0:
                        strategy.success_count += 1
                        logger.info(
                            f"Strategy {name} improved performance by {improvement:.2f}"
                        )
                    else:
                        strategy.failure_count += 1
                else:
                    strategy.failure_count += 1

            # Rank strategies by effectiveness
            self._update_strategy_priorities(improvements)

        finally:
            self.is_optimizing = False

    def _apply_strategy(self, strategy: OptimizationStrategy) -> bool:
        """Apply optimization strategy."""
        try:
            if strategy.name == "caching":
                return self._optimize_caching(strategy.parameters)
            elif strategy.name == "batching":
                return self._optimize_batching(strategy.parameters)
            elif strategy.name == "parallelization":
                return self._optimize_parallelization(strategy.parameters)
            elif strategy.name == "pruning":
                return self._optimize_pruning(strategy.parameters)
            elif strategy.name == "quantization":
                return self._optimize_quantization(strategy.parameters)
            elif strategy.name == "compilation":
                return self._optimize_compilation(strategy.parameters)
            else:
                return False
        except Exception as e:
            logger.error(f"Strategy {strategy.name} failed: {e}")
            return False

    def _optimize_caching(self, params: Dict) -> bool:
        """Optimize caching strategy."""
        cache_size = params.get("cache_size", 1000)
        cache_size = max(self.MIN_CACHE_SIZE, min(self.MAX_CACHE_SIZE, cache_size))

        # Adjust cache size based on hit rate
        hit_rate = self._get_cache_hit_rate()

        if hit_rate < 0.5 and cache_size < self.resource_limits["max_cache_size"]:
            # Increase cache size
            new_size = min(
                int(cache_size * 1.5), self.resource_limits["max_cache_size"]
            )
            params["cache_size"] = new_size
            logger.info(f"Increased cache size to {new_size}")
        elif hit_rate > 0.9 and cache_size > self.resource_limits["min_cache_size"]:
            # Decrease cache size to save memory
            new_size = max(
                int(cache_size * 0.8), self.resource_limits["min_cache_size"]
            )
            params["cache_size"] = new_size
            logger.info(f"Decreased cache size to {new_size}")

        # Implement LRU eviction
        self._evict_cache_lru(int(params["cache_size"]))

        return True

    def _optimize_batching(self, params: Dict) -> bool:
        """Optimize batching parameters."""
        current_batch_size = params.get("batch_size", 32)
        current_batch_size = max(
            self.MIN_BATCH_SIZE, min(self.MAX_BATCH_SIZE, current_batch_size)
        )

        # Adjust based on memory usage
        if (
            self.current_metrics
            and self.current_metrics.memory_mb > self.target_memory_mb
        ):
            new_size = max(self.MIN_BATCH_SIZE, current_batch_size // 2)
        else:
            new_size = min(self.MAX_BATCH_SIZE, int(current_batch_size * 1.5))

        params["batch_size"] = new_size
        logger.info(f"Adjusted batch size to {new_size}")
        return True

    def _optimize_parallelization(self, params: Dict) -> bool:
        """Optimize parallelization settings."""
        cpu_count = min(psutil.cpu_count(), self.MAX_WORKERS)
        current_workers = params.get("num_workers", 4)
        current_workers = max(self.MIN_WORKERS, min(self.MAX_WORKERS, current_workers))

        # Adjust based on CPU usage
        if self.current_metrics:
            if self.current_metrics.cpu_percent < 50:
                new_workers = min(cpu_count, current_workers + 1)
            elif self.current_metrics.cpu_percent > 80:
                new_workers = max(self.MIN_WORKERS, current_workers - 1)
            else:
                new_workers = current_workers

            params["num_workers"] = new_workers
            logger.info(f"Adjusted workers to {new_workers}")

        return True

    def _optimize_pruning(self, params: Dict) -> bool:
        """Optimize model pruning."""
        current_sparsity = params.get("sparsity", 0.1)
        # Clamp sparsity to safe range
        new_sparsity = min(0.9, max(0.0, current_sparsity * 1.1))
        params["sparsity"] = new_sparsity
        logger.info(f"Adjusted pruning sparsity to {new_sparsity:.2f}")
        return True

    def _optimize_quantization(self, params: Dict) -> bool:
        """Optimize quantization settings."""
        current_bits = params.get("bits", 8)
        # Clamp bits to safe range [4, 32]
        current_bits = max(4, min(32, current_bits))

        if (
            self.current_metrics
            and self.current_metrics.latency_ms > self.target_latency_ms
        ):
            new_bits = max(4, current_bits - 1)
        else:
            new_bits = current_bits

        params["bits"] = new_bits
        logger.info(f"Adjusted quantization to {new_bits} bits")
        return True

    def _optimize_compilation(self, params: Dict) -> bool:
        """Optimize compilation settings."""
        params["optimize"] = True
        params["backend"] = "jit"  # Safe default
        logger.info("Enabled compilation optimization")
        return True

    def _auto_tune_parameters(self):
        """Auto-tune hyperparameters using safe evaluation."""
        # FIXED: Always clamp parameters to valid ranges first, even if no metrics history
        for param_name, param_info in self.tunable_parameters.items():
            current_value = param_info["current"]
            current_value = max(
                param_info["min"], min(param_info["max"], current_value)
            )
            param_info["current"] = current_value

        # Early return if no metrics history for tuning
        if not self.metrics_history:
            return

        # Simple hill climbing for each parameter
        for param_name, param_info in self.tunable_parameters.items():
            current_value = param_info["current"]
            param_type = param_info.get("type", float)

            # Calculate safe perturbation
            step_size = param_info.get(
                "step", (param_info["max"] - param_info["min"]) * 0.1
            )

            test_values = []
            # Try decreasing
            decreased = current_value - step_size
            if decreased >= param_info["min"]:
                test_values.append(decreased)

            # Try increasing
            increased = current_value + step_size
            if increased <= param_info["max"]:
                test_values.append(increased)

            best_value = current_value
            best_score = self.current_metrics.score() if self.current_metrics else 0

            for test_value in test_values:
                # Cast to appropriate type
                if param_type == int:
                    test_value = int(test_value)
                else:
                    test_value = float(test_value)

                # Temporarily set parameter
                param_info["current"] = test_value

                # Evaluate using safe predefined function
                score = self._evaluate_parameter_safe(param_name, test_value)

                if score > best_score:
                    best_score = score
                    best_value = test_value

            # Update to best value
            if param_type == int:
                best_value = int(best_value)
            param_info["current"] = best_value

            if best_value != current_value:
                logger.info(
                    f"Auto-tuned {param_name} from {current_value} to {best_value}"
                )

    def _evaluate_parameter_safe(self, param_name: str, value: float) -> float:
        """SAFELY evaluate a parameter setting without eval."""
        if param_name not in self.tunable_parameters:
            logger.warning(f"Unknown parameter: {param_name}")
            return 0.0

        # Validate value range
        param_info = self.tunable_parameters[param_name]
        value = max(param_info["min"], min(param_info["max"], value))

        # Use predefined evaluation function
        evaluator = self.parameter_evaluators.get(param_name)
        if evaluator:
            return evaluator(value)

        # Default evaluation
        return 50.0

    def _evaluate_batch_size(self, value: float) -> float:
        """Evaluate batch size parameter."""
        # Optimal around 32, decreases as we move away
        optimal = 32
        distance = abs(value - optimal)
        score = max(0, 100 - distance * 2)

        # Penalize extremes
        if value < 4:
            score *= 0.5
        elif value > 256:
            score *= 0.7

        return score

    def _evaluate_num_workers(self, value: float) -> float:
        """Evaluate number of workers parameter."""
        cpu_count = psutil.cpu_count()
        # Optimal at cpu_count/2
        optimal = cpu_count / 2
        distance = abs(value - optimal)
        score = max(0, 100 - distance * 10)

        # Penalize too many workers
        if value > cpu_count:
            score *= 0.5

        return score

    def _evaluate_cache_size(self, value: float) -> float:
        """Evaluate cache size parameter."""
        # Consider hit rate if available
        hit_rate = self._get_cache_hit_rate()

        # Base score on current hit rate
        base_score = hit_rate * 100

        # Adjust based on size
        if value < 100:
            score = base_score * 0.5
        elif value > 50000:
            score = base_score * 0.8  # Diminishing returns
        else:
            # Linear in reasonable range
            score = base_score * (0.5 + value / 100000)

        return min(100, score)

    def _evaluate_learning_rate(self, value: float) -> float:
        """Evaluate learning rate parameter."""
        # Optimal around 0.001-0.01
        if value < 0.0001:
            return 20.0
        elif value > 0.1:
            return 10.0
        else:
            # Peak around 0.001-0.01
            log_value = np.log10(value)
            # Best at log_value = -3 to -2
            if -3 <= log_value <= -2:
                return 100.0
            else:
                distance = min(abs(log_value + 3), abs(log_value + 2))
                return max(0, 100 - distance * 30)

    def _update_strategy_priorities(
        self, improvements: List[Tuple[OptimizationStrategy, float]]
    ):
        """Update strategy priorities based on effectiveness."""
        # Sort by improvement
        improvements.sort(key=lambda x: x[1], reverse=True)

        # Update priorities
        for i, (strategy, _) in enumerate(improvements):
            strategy.priority = len(improvements) - i

    def _cleanup_resources(self):
        """Clean up unused resources."""
        # Clear old metrics
        if len(self.metrics_history) > 900:
            # Keep only recent metrics
            self.metrics_history = deque(list(self.metrics_history)[-500:], maxlen=1000)

        # Clear cache stats periodically
        total_accesses = self.cache_stats["hits"] + self.cache_stats["misses"]
        # FIXED: Changed from > to >= to handle exact boundary
        if total_accesses >= 100000:
            # Reset counters but keep ratio
            hit_rate = self._get_cache_hit_rate()
            self.cache_stats["hits"] = int(hit_rate * 1000)
            self.cache_stats["misses"] = int((1 - hit_rate) * 1000)

        # FIXED: Clean up thread pool if too many threads - with safe attribute access
        if hasattr(self, "thread_executor") and hasattr(
            self.thread_executor, "_threads"
        ):
            self.thread_executor._threads.clear()

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return self.cache_stats["hits"] / total

    def _evict_cache_lru(self, max_size: int):
        """Evict cache entries using LRU."""
        max_size = max(self.MIN_CACHE_SIZE, min(self.MAX_CACHE_SIZE, max_size))

        if len(self.cache) <= max_size:
            return

        # Remove least recently used
        to_remove = len(self.cache) - max_size

        # Use access order if available
        if self.cache_access_order:
            removed_keys = set()
            while to_remove > 0 and self.cache_access_order:
                key = self.cache_access_order.popleft()
                if key in self.cache and key not in removed_keys:
                    del self.cache[key]
                    removed_keys.add(key)
                    to_remove -= 1
        else:
            # FIXED: Fallback to removing first entries with bounds checking
            keys_to_remove = list(self.cache.keys())[: min(to_remove, len(self.cache))]
            for key in keys_to_remove:
                del self.cache[key]

    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not isinstance(key, str):
            key = str(key)

        if key in self.cache:
            self.cache_stats["hits"] += 1
            # Update access order (Note: deque.remove() is O(n) - acceptable for bounded cache)
            if key in self.cache_access_order:
                self.cache_access_order.remove(key)
            self.cache_access_order.append(key)
            return self.cache[key]
        else:
            self.cache_stats["misses"] += 1
            return None

    def cache_set(self, key: str, value: Any):
        """Set value in cache."""
        if not isinstance(key, str):
            key = str(key)

        # Check cache size limit
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            self._evict_cache_lru(self.MAX_CACHE_SIZE - 1)

        self.cache[key] = value
        self.cache_access_order.append(key)

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization status report."""
        report = {
            "is_optimizing": self.is_optimizing,
            "current_metrics": None,
            "strategies": {},
            "tunable_parameters": {},
            "cache_hit_rate": self._get_cache_hit_rate(),
            "cache_size": len(self.cache),
            "metrics_history_size": len(self.metrics_history),
        }

        if self.current_metrics:
            report["current_metrics"] = {
                "latency_ms": self.current_metrics.latency_ms,
                "throughput_ops": self.current_metrics.throughput_ops,
                "memory_mb": self.current_metrics.memory_mb,
                "cpu_percent": self.current_metrics.cpu_percent,
                "gpu_percent": self.current_metrics.gpu_percent,
                "score": self.current_metrics.score(),
            }

        for name, strategy in self.strategies.items():
            report["strategies"][name] = {
                "enabled": strategy.enabled,
                "effectiveness": strategy.effectiveness(),
                "priority": strategy.priority,
                "parameters": strategy.parameters,
                "success_count": strategy.success_count,
                "failure_count": strategy.failure_count,
            }

        for name, param in self.tunable_parameters.items():
            report["tunable_parameters"][name] = {
                "current": param["current"],
                "min": param["min"],
                "max": param["max"],
                "type": (
                    param["type"].__name__
                    if hasattr(param["type"], "__name__")
                    else str(param["type"])
                ),
            }

        return report

    def save_state(self, filepath: str):
        """Save optimizer state to disk."""
        try:
            state = {
                "strategies": {},
                "tunable_parameters": self.tunable_parameters,
                "metrics_history": list(self.metrics_history),
                "baseline_metrics": self.baseline_metrics,
                "cache_stats": dict(self.cache_stats),
                "optimization_interval_s": self.optimization_interval_s,
                "target_latency_ms": self.target_latency_ms,
                "target_memory_mb": self.target_memory_mb,
            }

            # Convert strategies to serializable format
            for name, strategy in self.strategies.items():
                state["strategies"][name] = {
                    "enabled": strategy.enabled,
                    "priority": strategy.priority,
                    "parameters": strategy.parameters,
                    "success_count": strategy.success_count,
                    "failure_count": strategy.failure_count,
                }

            # Ensure directory exists
            os.makedirs(
                os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
                exist_ok=True,
            )

            with open(filepath, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved optimizer state to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")

    def load_state(self, filepath: str):
        """Load optimizer state from disk."""
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} not found")
            return

        try:
            with open(filepath, "rb") as f:
                state = safe_pickle_load(f)

            # Restore strategies
            if "strategies" in state:
                for name, strategy_data in state["strategies"].items():
                    if name in self.strategies:
                        self.strategies[name].enabled = strategy_data["enabled"]
                        self.strategies[name].priority = strategy_data["priority"]
                        self.strategies[name].parameters = strategy_data["parameters"]
                        self.strategies[name].success_count = strategy_data[
                            "success_count"
                        ]
                        self.strategies[name].failure_count = strategy_data[
                            "failure_count"
                        ]

            # Restore tunable parameters with validation
            if "tunable_parameters" in state:
                for name, param_data in state["tunable_parameters"].items():
                    if name in self.tunable_parameters:
                        # Validate loaded values
                        current = param_data.get(
                            "current", self.tunable_parameters[name]["current"]
                        )
                        current = max(
                            param_data["min"], min(param_data["max"], current)
                        )
                        self.tunable_parameters[name]["current"] = current

            # Restore metrics history
            if "metrics_history" in state:
                self.metrics_history = deque(state["metrics_history"], maxlen=1000)

            # Restore other state
            self.baseline_metrics = state.get("baseline_metrics")

            if "cache_stats" in state:
                self.cache_stats.update(state["cache_stats"])

            # Restore configuration
            if "optimization_interval_s" in state:
                self.optimization_interval_s = max(
                    1, min(3600, state["optimization_interval_s"])
                )
            if "target_latency_ms" in state:
                self.target_latency_ms = max(1, min(10000, state["target_latency_ms"]))
            if "target_memory_mb" in state:
                self.target_memory_mb = max(
                    self.MIN_MEMORY_MB,
                    min(self.MAX_MEMORY_MB, state["target_memory_mb"]),
                )

            logger.info(f"Loaded optimizer state from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")
            # Continue with default state on error

    def reset(self):
        """Reset optimizer to initial state."""
        self.metrics_history.clear()
        self.current_metrics = None
        self.baseline_metrics = None
        self.cache.clear()
        # FIXED: Reset cache_stats to 0 instead of clearing defaultdict
        self.cache_stats["hits"] = 0
        self.cache_stats["misses"] = 0
        self.cache_access_order.clear()

        # Reset strategies
        for strategy in self.strategies.values():
            strategy.success_count = 0
            strategy.failure_count = 0
            strategy.priority = 0

        # Reset tunable parameters to defaults
        for param in self.tunable_parameters.values():
            if param["type"] == int:
                param["current"] = int((param["min"] + param["max"]) / 2)
            else:
                param["current"] = (param["min"] + param["max"]) / 2

        logger.info("Optimizer reset to initial state")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop()
            if hasattr(self, "thread_executor"):
                self.thread_executor.shutdown(wait=False)
            if hasattr(self, "process_executor"):
                self.process_executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )
