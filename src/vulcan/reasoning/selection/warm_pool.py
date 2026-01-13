"""
Warm Start Pool for Tool Selection System

Manages pre-warmed tool instances to eliminate cold start penalties
and ensure fast tool execution when needed.

Fixed version with proper error handling, thread safety, and interruptible threads.
"""

import inspect
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


def _class_accepts_name_parameter(tool_class) -> bool:
    """
    Check if a class's __init__ method accepts 'name' as a constructor parameter.
    
    This is used to differentiate between classes that:
    1. Take 'name' as a constructor argument (e.g., MockTool(name, warm_time))
    2. Set 'name' internally (e.g., MathematicalComputationTool sets self.name internally)
    
    Returns:
        True if 'name' is a parameter in __init__, False otherwise
    """
    try:
        sig = inspect.signature(tool_class.__init__)
        return 'name' in sig.parameters
    except (ValueError, TypeError):
        # Return False as the safe default to avoid passing unexpected 'name' arguments
        # to classes that don't expect them. ValueError can occur if the callable has
        # no signature, TypeError if the object is not callable.
        return False


class PoolState(Enum):
    """State of a pool instance"""

    COLD = "cold"
    WARMING = "warming"
    READY = "ready"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


class ScalingPolicy(Enum):
    """Pool scaling policies"""

    FIXED = "fixed"
    DYNAMIC = "dynamic"
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"


@dataclass
class PoolInstance:
    """Single instance in the warm pool"""

    instance_id: str
    tool_name: str
    tool_instance: Any
    state: PoolState
    created_time: float
    last_used_time: float
    usage_count: int = 0
    health_check_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if instance is available for use"""
        return self.state == PoolState.READY

    def is_healthy(self) -> bool:
        """Check if instance is healthy"""
        return self.health_check_failures < 3 and self.state != PoolState.UNHEALTHY


@dataclass
class PoolStatistics:
    """Statistics for a tool pool"""

    total_instances: int = 0
    ready_instances: int = 0
    busy_instances: int = 0
    unhealthy_instances: int = 0
    total_requests: int = 0
    cache_hits: int = 0
    cold_starts: int = 0
    avg_wait_time_ms: float = 0.0
    avg_warm_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class ToolPool:
    """Pool for a single tool type"""

    def __init__(
        self,
        tool_name: str,
        tool_factory: Callable,
        min_instances: int = 1,
        max_instances: int = 5,
        warm_up_func: Optional[Callable] = None,
    ):
        """
        Initialize tool pool

        Args:
            tool_name: Name of the tool
            tool_factory: Function to create tool instances
            min_instances: Minimum instances to maintain
            max_instances: Maximum instances allowed
            warm_up_func: Optional function to warm up instance
        """
        self.tool_name = tool_name
        self.tool_factory = tool_factory
        self.warm_up_func = warm_up_func
        self.min_instances = min_instances
        self.max_instances = max_instances

        # Instance management
        self.instances = {}  # instance_id -> PoolInstance
        self.available_queue = queue.Queue()
        self.busy_instances = set()

        # Statistics
        self.stats = PoolStatistics()
        self.request_history = deque(maxlen=100)

        # CRITICAL FIX: Use RLock for thread safety
        self.lock = threading.RLock()

        # CRITICAL FIX: Add warm-up timeout
        self.max_warm_time = 60.0  # seconds

        # Initialize minimum instances
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize minimum number of instances"""
        try:
            for i in range(self.min_instances):
                self._create_instance()
        except Exception as e:
            logger.error(f"Pool initialization failed: {e}")

    def _create_instance(self) -> Optional[str]:
        """Create new pool instance"""
        with self.lock:
            try:
                if len(self.instances) >= self.max_instances:
                    return None

                instance_id = f"{self.tool_name}_{time.time()}_{len(self.instances)}"

                # Create tool instance
                tool_instance = self.tool_factory()

                # Create pool instance
                pool_instance = PoolInstance(
                    instance_id=instance_id,
                    tool_name=self.tool_name,
                    tool_instance=tool_instance,
                    state=PoolState.COLD,
                    created_time=time.time(),
                    last_used_time=time.time(),
                )

                self.instances[instance_id] = pool_instance
                self.stats.total_instances += 1

                # Start warming
                threading.Thread(
                    target=self._warm_instance, args=(instance_id,), daemon=True
                ).start()

                return instance_id

            except Exception as e:
                logger.error(f"Failed to create instance for {self.tool_name}: {e}")
                return None

    # CRITICAL FIX: Handle warm-up failures properly
    def _warm_instance(self, instance_id: str):
        """Warm up an instance - CRITICAL: With proper error handling and cleanup"""

        with self.lock:
            if instance_id not in self.instances:
                return

            instance = self.instances[instance_id]
            instance.state = PoolState.WARMING

        start_time = time.time()

        try:
            if self.warm_up_func:
                # Custom warm-up function
                self.warm_up_func(instance.tool_instance)
            else:
                # Default warm-up: call with dummy input
                if hasattr(instance.tool_instance, "warm_up"):
                    instance.tool_instance.warm_up()
                elif hasattr(instance.tool_instance, "reason"):
                    # Try a simple reasoning call
                    instance.tool_instance.reason("test")

            warm_time = (time.time() - start_time) * 1000

            # CRITICAL FIX: Check if warm-up took too long
            if warm_time > self.max_warm_time * 1000:
                raise TimeoutError(f"Warm-up exceeded {self.max_warm_time}s")

            with self.lock:
                # Double-check instance still exists
                if instance_id not in self.instances:
                    return

                instance.state = PoolState.READY
                self.available_queue.put(instance_id)
                self.stats.ready_instances += 1

                # Update average warm time
                alpha = 0.1
                self.stats.avg_warm_time_ms = (
                    1 - alpha
                ) * self.stats.avg_warm_time_ms + alpha * warm_time

            logger.debug(f"Instance {instance_id} warmed in {warm_time:.0f}ms")

        except Exception as e:
            logger.error(f"Failed to warm instance {instance_id}: {e}")

            # CRITICAL FIX: Proper cleanup of failed instances
            with self.lock:
                if instance_id not in self.instances:
                    return

                instance.state = PoolState.UNHEALTHY
                instance.health_check_failures = 999  # Mark as permanently failed
                self.stats.unhealthy_instances += 1

                # CRITICAL FIX: Remove failed instance to prevent resource leak
                if instance.health_check_failures > 5:
                    self._remove_instance(instance_id)
                    logger.info(
                        f"Removed failed instance {instance_id} after warm-up failure"
                    )

    def acquire(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Any]]:
        """
        Acquire a warm instance

        Returns:
            (instance_id, tool_instance) or None
        """

        try:
            start_time = time.time()

            with self.lock:
                self.stats.total_requests += 1

            deadline = start_time + timeout if timeout else None

            # Try to get available instance
            while True:
                try:
                    remaining_timeout = None
                    if deadline:
                        remaining_timeout = max(0, deadline - time.time())
                        if remaining_timeout <= 0:
                            break

                    instance_id = self.available_queue.get(
                        timeout=remaining_timeout or 0.1
                    )

                    with self.lock:
                        if instance_id in self.instances:
                            instance = self.instances[instance_id]

                            if instance.is_available() and instance.is_healthy():
                                instance.state = PoolState.BUSY
                                instance.last_used_time = time.time()
                                instance.usage_count += 1
                                self.busy_instances.add(instance_id)

                                self.stats.ready_instances -= 1
                                self.stats.busy_instances += 1
                                self.stats.cache_hits += 1

                                wait_time = (time.time() - start_time) * 1000
                                self._update_wait_time(wait_time)

                                return instance_id, instance.tool_instance
                            else:
                                # Instance not ready, try again
                                continue

                except queue.Empty:
                    if deadline and time.time() >= deadline:
                        break
                    continue

            # No warm instance available, create cold start
            with self.lock:
                self.stats.cold_starts += 1

                # Try to create new instance if under limit
                if len(self.instances) < self.max_instances:
                    instance_id = self._create_instance()
                    if instance_id:
                        # Wait for warm-up or timeout
                        while deadline is None or time.time() < deadline:
                            with self.lock:
                                if instance_id not in self.instances:
                                    break

                                instance = self.instances[instance_id]
                                if instance.state == PoolState.READY:
                                    instance.state = PoolState.BUSY
                                    instance.last_used_time = time.time()
                                    instance.usage_count += 1
                                    self.busy_instances.add(instance_id)

                                    self.stats.busy_instances += 1

                                    wait_time = (time.time() - start_time) * 1000
                                    self._update_wait_time(wait_time)

                                    return instance_id, instance.tool_instance
                                elif instance.state == PoolState.UNHEALTHY:
                                    break

                            time.sleep(0.01)

            # Failed to acquire
            logger.warning(f"Failed to acquire instance for {self.tool_name}")
            return None
        except Exception as e:
            logger.error(f"Acquire failed: {e}")
            return None

    def release(self, instance_id: str):
        """Release instance back to pool"""
        try:
            with self.lock:
                if instance_id not in self.instances:
                    return

                instance = self.instances[instance_id]

                if instance_id in self.busy_instances:
                    self.busy_instances.remove(instance_id)
                    self.stats.busy_instances -= 1

                # Check health before returning to pool
                if instance.is_healthy():
                    instance.state = PoolState.READY
                    self.available_queue.put(instance_id)
                    self.stats.ready_instances += 1
                else:
                    instance.state = PoolState.UNHEALTHY
                    self.stats.unhealthy_instances += 1
        except Exception as e:
            logger.error(f"Release failed: {e}")

    def health_check(self, instance_id: str) -> bool:
        """Perform health check on instance"""
        try:
            with self.lock:
                if instance_id not in self.instances:
                    return False

                instance = self.instances[instance_id]

                # Simple health check
                if hasattr(instance.tool_instance, "health_check"):
                    healthy = instance.tool_instance.health_check()
                else:
                    # Default: check if callable
                    healthy = callable(instance.tool_instance)

                if healthy:
                    instance.health_check_failures = 0
                else:
                    instance.health_check_failures += 1

                if instance.health_check_failures >= 3:
                    instance.state = PoolState.UNHEALTHY
                    self.stats.unhealthy_instances += 1

                return healthy

        except Exception as e:
            logger.error(f"Health check failed for {instance_id}: {e}")
            with self.lock:
                if instance_id in self.instances:
                    self.instances[instance_id].health_check_failures += 1
            return False

    # CRITICAL FIX: Thread-safe scaling with proper error handling
    def scale(self, target_size: int):
        """Scale pool to target size - CRITICAL: Thread-safe implementation"""

        with self.lock:
            try:
                current_size = len(self.instances)
                target_size = max(
                    self.min_instances, min(target_size, self.max_instances)
                )

                if target_size > current_size:
                    # CRITICAL FIX: Scale up with error handling
                    to_add = target_size - current_size
                    added = 0

                    for _ in range(to_add):
                        new_id = self._create_instance()
                        if new_id is None:
                            logger.warning(
                                f"Failed to scale up {self.tool_name}, may have hit resource limits"
                            )
                            break
                        added += 1

                    if added > 0:
                        logger.info(
                            f"Scaled up {self.tool_name} pool by {added} instances"
                        )

                elif target_size < current_size:
                    # CRITICAL FIX: Scale down - remove unhealthy/idle instances
                    to_remove = current_size - target_size

                    # Get removable instances (not busy, prefer unhealthy/old)
                    removable = [
                        inst
                        for inst in self.instances.values()
                        if inst.state != PoolState.BUSY
                    ]

                    # Sort by health (unhealthy first) and age (oldest first)
                    removable.sort(
                        key=lambda x: (
                            x.is_healthy(),  # False (unhealthy) sorts before True (healthy)
                            -x.last_used_time,  # Negative for oldest first
                        )
                    )

                    removed = 0
                    for instance in removable[:to_remove]:
                        self._remove_instance(instance.instance_id)
                        removed += 1

                    if removed > 0:
                        logger.info(
                            f"Scaled down {self.tool_name} pool by {removed} instances"
                        )

                    if removed < to_remove:
                        logger.warning(
                            f"Could only remove {removed}/{to_remove} instances from {self.tool_name} pool"
                        )
            except Exception as e:
                logger.error(f"Scaling failed for {self.tool_name}: {e}")

    def _remove_instance(self, instance_id: str):
        """Remove instance from pool"""
        try:
            with self.lock:
                if instance_id not in self.instances:
                    return

                instance = self.instances[instance_id]
                instance.state = PoolState.SHUTDOWN

                # Clean up
                if hasattr(instance.tool_instance, "shutdown"):
                    try:
                        instance.tool_instance.shutdown()
                    except Exception as e:
                        logger.warning(f"Instance shutdown failed: {e}")

                del self.instances[instance_id]
                self.stats.total_instances -= 1
        except Exception as e:
            logger.error(f"Remove instance failed: {e}")

    def _update_wait_time(self, wait_time_ms: float):
        """Update average wait time"""
        try:
            alpha = 0.1
            self.stats.avg_wait_time_ms = (
                1 - alpha
            ) * self.stats.avg_wait_time_ms + alpha * wait_time_ms

            # Track history for prediction
            self.request_history.append(
                {"timestamp": time.time(), "wait_time": wait_time_ms}
            )
        except Exception as e:
            logger.error(f"Update wait time failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            try:
                return {
                    "tool": self.tool_name,
                    "total_instances": self.stats.total_instances,
                    "ready": self.stats.ready_instances,
                    "busy": self.stats.busy_instances,
                    "unhealthy": self.stats.unhealthy_instances,
                    "requests": self.stats.total_requests,
                    "hit_rate": self.stats.hit_rate,
                    "cold_starts": self.stats.cold_starts,
                    "avg_wait_ms": self.stats.avg_wait_time_ms,
                    "avg_warm_ms": self.stats.avg_warm_time_ms,
                }
            except Exception as e:
                logger.error(f"Get statistics failed: {e}")
                return {}

    def shutdown(self):
        """Shutdown all instances"""
        try:
            with self.lock:
                for instance_id in list(self.instances.keys()):
                    self._remove_instance(instance_id)
        except Exception as e:
            logger.error(f"Pool shutdown failed: {e}")


class WarmStartPool:
    """
    Main warm start pool managing all tool pools
    """

    def __init__(self, tools: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize warm start pool

        Args:
            tools: Dictionary of tool_name -> tool_instance/factory
            config: Configuration
        """
        config = config or {}

        self.tools = tools
        self.config = config

        # Scaling configuration
        try:
            policy_name = config.get("scaling_policy", "dynamic").upper()
            self.scaling_policy = (
                ScalingPolicy[policy_name]
                if policy_name in ScalingPolicy.__members__
                else ScalingPolicy.DYNAMIC
            )
        except Exception as e:
            logger.error(f"Invalid scaling policy: {e}, using DYNAMIC")
            self.scaling_policy = ScalingPolicy.DYNAMIC

        self.min_pool_size = config.get("min_pool_size", 1)
        self.max_pool_size = config.get("max_pool_size", 5)

        # Tool pools
        self.pools = {}
        self._initialize_pools()

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()

        # Demand predictor
        self.demand_predictor = DemandPredictor()

        # CRITICAL FIX: Use Event for interruptible thread management
        self._shutdown_event = threading.Event()
        self.shutdown_lock = threading.Lock()

        # CRITICAL FIX: Add monitoring flag for test compatibility
        self.monitoring = True

        # Background threads
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()

        self.health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_thread.start()

        logger.info(f"Warm pool initialized with {len(self.pools)} tool pools")

    def _initialize_pools(self):
        """Initialize pools for each tool - CRITICAL FIX: Proper factory creation"""

        try:
            for tool_name, tool_instance in self.tools.items():
                # CRITICAL FIX: Determine if it's a factory or instance and create appropriate factory
                if callable(tool_instance) and not hasattr(tool_instance, "__self__"):
                    # It's a factory function or unbound class
                    factory = tool_instance
                else:
                    # It's an instance - create a factory that produces new instances
                    tool_class = tool_instance.__class__
                    
                    # Note: Check if the class actually accepts 'name' as a constructor parameter
                    # Some classes (like MathematicalComputationTool) have self.name but don't
                    # accept it as a constructor argument - they set it internally
                    class_accepts_name = _class_accepts_name_parameter(tool_class)

                    # CRITICAL FIX: Introspect the instance to determine how to clone it
                    if hasattr(tool_instance, "name") and class_accepts_name:
                        # Has a name attribute AND the class accepts it as a constructor param
                        tool_name_arg = getattr(tool_instance, "name", tool_name)

                        if hasattr(tool_instance, "warm_time"):
                            # MockTool specifically - has name and warm_time
                            warm_time = getattr(tool_instance, "warm_time", 0.01)
                            factory = lambda n=tool_name_arg, w=warm_time, cls=tool_class: cls(
                                n, w
                            )
                        elif hasattr(tool_instance, "config"):
                            # Has config attribute
                            config = getattr(tool_instance, "config", {})
                            factory = (
                                lambda n=tool_name_arg, c=config, cls=tool_class: cls(
                                    n, c
                                )
                            )
                        else:
                            # Just name
                            def factory(n=tool_name_arg, cls=tool_class):
                                return cls(n)

                    elif hasattr(tool_instance, "config"):
                        # Config-based tool without name attribute
                        # Check if class accepts config parameter
                        config = getattr(tool_instance, "config", {})
                        
                        # Try to determine if class takes (name, config) or just (config)
                        sig = inspect.signature(tool_class.__init__)
                        params = list(sig.parameters.keys())
                        
                        # Check if 'config' is a parameter and there are no other required params
                        # A param is required if it has no default value and isn't self/cls
                        required_params = [
                            p for p, v in sig.parameters.items()
                            if p not in ('self', 'cls') 
                            and v.default == inspect.Parameter.empty
                        ]
                        
                        if 'config' in params and required_params == ['config']:
                            def factory(c=config, cls=tool_class):
                                return cls(config=c)
                        else:
                            # Fallback to (name, config) pattern
                            def factory(n=tool_name, c=config, cls=tool_class):
                                return cls(n, c)

                    else:
                        # Default: try no-arg constructor or use instance as singleton
                        # This handles classes like MathematicalComputationTool that have
                        # self.name but don't take name as a constructor argument
                        try:
                            # First try: no-arg constructor
                            tool_class()

                            def factory(cls=tool_class):
                                return cls()

                        except TypeError:
                            # TypeError often means missing required args
                            # Try with empty config dict (common pattern for reasoning tools)
                            try:
                                tool_class(config={})
                                
                                def factory(cls=tool_class):
                                    return cls(config={})
                                    
                                logger.debug(f"Factory for {tool_name} created with empty config")
                            except Exception as e:
                                # Can't instantiate - use as singleton (not ideal but safe)
                                logger.warning(
                                    f"Using {tool_name} as singleton - factory creation failed: {type(e).__name__}: {e}"
                                )

                                def factory(inst=tool_instance):
                                    return inst
                        except Exception as e:
                            # Can't instantiate - use as singleton (not ideal but safe)
                            logger.warning(
                                f"Using {tool_name} as singleton - factory creation failed: {type(e).__name__}: {e}"
                            )

                            def factory(inst=tool_instance):
                                return inst

                # Create pool
                pool = ToolPool(
                    tool_name=tool_name,
                    tool_factory=factory,
                    min_instances=self.min_pool_size,
                    max_instances=self.max_pool_size,
                    warm_up_func=self._get_warm_up_func(tool_name),
                )

                self.pools[tool_name] = pool
        except Exception as e:
            logger.error(f"Pool initialization failed: {e}")

    def _get_warm_up_func(self, tool_name: str) -> Optional[Callable]:
        """Get warm-up function for tool"""

        # Tool-specific warm-up functions
        warm_up_funcs = {
            "symbolic": lambda tool: (
                tool.reason("A → B, A ⊢ B") if hasattr(tool, "reason") else None
            ),
            "probabilistic": lambda tool: (
                tool.reason({"data": [1, 2, 3]}) if hasattr(tool, "reason") else None
            ),
            "causal": lambda tool: (
                tool.reason({"graph": {"A": ["B"]}})
                if hasattr(tool, "reason")
                else None
            ),
            "analogical": lambda tool: (
                tool.reason({"source": "A", "target": "B"})
                if hasattr(tool, "reason")
                else None
            ),
            "multimodal": lambda tool: (
                tool.reason({"modalities": ["text", "image"]})
                if hasattr(tool, "reason")
                else None
            ),
        }

        return warm_up_funcs.get(tool_name)

    def acquire_tool(
        self, tool_name: str, timeout: float = 5.0
    ) -> Optional[Tuple[str, Any]]:
        """
        Acquire warm tool instance

        Returns:
            (instance_id, tool_instance) or None
        """

        try:
            if tool_name not in self.pools:
                logger.error(f"Unknown tool: {tool_name}")
                return None

            pool = self.pools[tool_name]

            # Update demand tracking
            self.demand_predictor.record_request(tool_name)

            return pool.acquire(timeout)
        except Exception as e:
            logger.error(f"Acquire tool failed: {e}")
            return None

    def release_tool(self, tool_name: str, instance_id: str):
        """Release tool instance back to pool"""

        try:
            if tool_name not in self.pools:
                return

            pool = self.pools[tool_name]
            pool.release(instance_id)
        except Exception as e:
            logger.error(f"Release tool failed: {e}")

    # CRITICAL FIX: Interruptible background threads
    def _monitor_loop(self):
        """Background monitoring - CRITICAL: With error handling and interruptible sleep"""

        consecutive_errors = 0
        max_errors = 5

        while not self._shutdown_event.is_set():
            try:
                # Collect statistics
                stats = self.get_statistics()

                # Log periodic summary
                if int(time.time()) % 60 == 0:
                    logger.debug(f"Warm pool stats: {stats.get('summary', {})}")

                # CRITICAL FIX: Interruptible sleep - can be interrupted by shutdown
                if self._shutdown_event.wait(timeout=1):
                    break
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Monitoring error ({consecutive_errors}/{max_errors}): {e}"
                )

                if consecutive_errors >= max_errors:
                    logger.critical("Monitoring failed repeatedly, stopping")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=5):
                    break

    # CRITICAL FIX: Interruptible scaling thread
    def _scaling_loop(self):
        """Background scaling - CRITICAL: With error handling and interruptible sleep"""

        consecutive_errors = 0
        max_errors = 5

        while not self._shutdown_event.is_set():
            try:
                if self.scaling_policy == ScalingPolicy.DYNAMIC:
                    self._dynamic_scaling()
                elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
                    self._predictive_scaling()
                elif self.scaling_policy == ScalingPolicy.REACTIVE:
                    self._reactive_scaling()

                # CRITICAL FIX: Interruptible scaling interval - 10 seconds can be interrupted
                if self._shutdown_event.wait(timeout=10):
                    break
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Scaling error ({consecutive_errors}/{max_errors}): {e}")

                if consecutive_errors >= max_errors:
                    logger.critical("Scaling failed repeatedly, stopping")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=30):
                    break

    def _dynamic_scaling(self):
        """Dynamic scaling based on current load"""

        try:
            for tool_name, pool in self.pools.items():
                stats = pool.get_statistics()

                # Scale up if high utilization
                total = stats.get("total_instances", 1)
                busy = stats.get("busy", 0)
                utilization = busy / max(1, total)

                if utilization > 0.8 and total < self.max_pool_size:
                    # Scale up
                    new_size = min(total + 1, self.max_pool_size)
                    pool.scale(new_size)
                    logger.info(f"Scaling up {tool_name} pool to {new_size}")

                elif utilization < 0.2 and total > self.min_pool_size:
                    # Scale down
                    new_size = max(total - 1, self.min_pool_size)
                    pool.scale(new_size)
                    logger.info(f"Scaling down {tool_name} pool to {new_size}")
        except Exception as e:
            logger.error(f"Dynamic scaling failed: {e}")

    def _predictive_scaling(self):
        """Predictive scaling based on demand patterns"""

        try:
            for tool_name, pool in self.pools.items():
                # Predict demand for next window
                predicted_demand = self.demand_predictor.predict(tool_name)

                # Calculate required instances
                avg_service_time = 100  # ms, simplified
                required_instances = max(
                    self.min_pool_size,
                    min(
                        int(predicted_demand * avg_service_time / 1000) + 1,
                        self.max_pool_size,
                    ),
                )

                current_size = pool.stats.total_instances
                if required_instances != current_size:
                    pool.scale(required_instances)
                    logger.info(
                        f"Predictive scaling {tool_name}: {current_size} -> {required_instances}"
                    )
        except Exception as e:
            logger.error(f"Predictive scaling failed: {e}")

    def _reactive_scaling(self):
        """Reactive scaling based on recent metrics"""

        try:
            for tool_name, pool in self.pools.items():
                stats = pool.get_statistics()

                # React to cold starts
                cold_starts = stats.get("cold_starts", 0)
                cache_hits = stats.get("cache_hits", 1)

                if cold_starts > cache_hits * 0.1:
                    # Too many cold starts, scale up
                    new_size = min(pool.stats.total_instances + 1, self.max_pool_size)
                    pool.scale(new_size)

                # React to wait times
                elif stats.get("avg_wait_ms", 0) > 100:
                    # Long wait times, scale up
                    new_size = min(pool.stats.total_instances + 1, self.max_pool_size)
                    pool.scale(new_size)
        except Exception as e:
            logger.error(f"Reactive scaling failed: {e}")

    # CRITICAL FIX: Interruptible health check thread
    def _health_check_loop(self):
        """Background health checking - CRITICAL: With error handling and interruptible sleep"""

        consecutive_errors = 0
        max_errors = 5

        while not self._shutdown_event.is_set():
            try:
                for pool in self.pools.values():
                    for instance_id in list(pool.instances.keys()):
                        pool.health_check(instance_id)

                # CRITICAL FIX: Interruptible health check interval - 30 seconds can be interrupted
                if self._shutdown_event.wait(timeout=30):
                    break
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Health check error ({consecutive_errors}/{max_errors}): {e}"
                )

                if consecutive_errors >= max_errors:
                    logger.critical("Health checking failed repeatedly, stopping")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=60):
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""

        try:
            pool_stats = {}
            total_instances = 0
            total_ready = 0
            total_busy = 0

            for tool_name, pool in self.pools.items():
                stats = pool.get_statistics()
                pool_stats[tool_name] = stats
                total_instances += stats.get("total_instances", 0)
                total_ready += stats.get("ready", 0)
                total_busy += stats.get("busy", 0)

            return {
                "pools": pool_stats,
                "summary": {
                    "total_instances": total_instances,
                    "total_ready": total_ready,
                    "total_busy": total_busy,
                    "utilization": total_busy / max(1, total_instances),
                },
                "resource_usage": self.resource_monitor.get_usage(),
            }
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}

    def shutdown(self, timeout: float = 2.0):
        """Shutdown warm pool - CRITICAL: Fast shutdown with interruptible threads"""

        with self.shutdown_lock:
            if self._shutdown_event.is_set():
                return

            logger.info("Shutting down warm pool")

            # CRITICAL FIX: Signal all threads to stop immediately
            self._shutdown_event.set()

            # CRITICAL FIX: Set monitoring flag to False
            self.monitoring = False

        # CRITICAL FIX: Give threads minimal time to notice shutdown signal
        deadline = time.time() + timeout

        # Wait for threads with short timeout
        threads = [
            ("monitor", self.monitor_thread),
            ("scaling", self.scaling_thread),
            ("health", self.health_thread),
        ]

        for name, thread in threads:
            if thread.is_alive():
                remaining = max(0.01, deadline - time.time())
                thread.join(timeout=remaining)
                if thread.is_alive():
                    logger.warning(f"{name} thread still alive after shutdown signal")

        # Shutdown all pools
        for pool in self.pools.values():
            try:
                pool.shutdown()
            except Exception as e:
                logger.error(f"Pool shutdown error: {e}")

        logger.info("Warm pool shutdown complete")


class ResourceMonitor:
    """Monitor system resources for pool management"""

    def __init__(self):
        self.cpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.lock = threading.RLock()

    def get_usage(self) -> Dict[str, float]:
        """Get current resource usage"""

        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent

            with self.lock:
                self.cpu_history.append(cpu)
                self.memory_history.append(memory)

                cpu_list = list(self.cpu_history)
                memory_list = list(self.memory_history)

            return {
                "cpu_percent": cpu,
                "memory_percent": memory,
                "avg_cpu": np.mean(cpu_list) if cpu_list else 0.0,
                "avg_memory": np.mean(memory_list) if memory_list else 0.0,
            }
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "avg_cpu": 0.0,
                "avg_memory": 0.0,
            }


class DemandPredictor:
    """Predict tool demand for proactive scaling"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.demand_history = defaultdict(lambda: deque(maxlen=window_size))
        self.time_buckets = defaultdict(lambda: defaultdict(list))
        self.lock = threading.RLock()

    def record_request(self, tool_name: str):
        """Record tool request"""

        try:
            current_time = time.time()

            with self.lock:
                self.demand_history[tool_name].append(current_time)

                # Track by hour of day for patterns
                hour = int(current_time % 86400 / 3600)
                self.time_buckets[tool_name][hour].append(current_time)
        except Exception as e:
            logger.error(f"Record request failed: {e}")

    def predict(self, tool_name: str, horizon_seconds: int = 60) -> float:
        """
        Predict demand for next time horizon

        Returns:
            Predicted requests per second
        """

        try:
            with self.lock:
                history = list(self.demand_history[tool_name])

            if len(history) < 2:
                return 0.1  # Default low demand

            # Simple moving average using the specified horizon
            recent_window = horizon_seconds  # Use parameter instead of hardcoded value
            current_time = time.time()
            recent_requests = sum(
                1 for t in history if current_time - t < recent_window
            )

            return recent_requests / recent_window if recent_window > 0 else 0.1
        except Exception as e:
            logger.error(f"Demand prediction failed: {e}")
            return 0.1
