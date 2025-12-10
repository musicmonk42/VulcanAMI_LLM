"""
Admission Control for Tool Selection System

Manages request admission, rate limiting, backpressure, and overload protection
to ensure system stability and predictable performance.

Fixed version with proper thread safety, error recovery, and interruptible threads.
"""

import heapq
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""

    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4  # Lowest priority


class AdmissionDecision(Enum):
    """Admission control decisions"""

    ADMIT = "admit"
    REJECT = "reject"
    DEFER = "defer"
    REDIRECT = "redirect"


class SystemHealth(Enum):
    """System health states"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    CRITICAL = "critical"


@dataclass
class Request:
    """Request for admission"""

    request_id: str
    priority: RequestPriority
    estimated_cost: Dict[str, float]  # time_ms, energy_mj, memory_mb
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdmissionMetrics:
    """Metrics for admission control"""

    total_requests: int = 0
    admitted_requests: int = 0
    rejected_requests: int = 0
    deferred_requests: int = 0
    current_queue_depth: int = 0
    avg_wait_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    rejection_rate: float = 0.0
    throughput_rps: float = 0.0


class TokenBucketRateLimiter:
    """Token bucket algorithm for rate limiting"""

    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

        # CRITICAL FIX: Use RLock for better thread safety
        self.lock = threading.RLock()

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens"""
        with self.lock:
            try:
                now = time.time()
                elapsed = now - self.last_update

                # Add tokens based on elapsed time
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                # Check if we can consume
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                return False
            except Exception as e:
                logger.error(f"Token consumption failed: {e}")
                return False

    def available_tokens(self) -> float:
        """Get current available tokens"""
        with self.lock:
            try:
                now = time.time()
                elapsed = now - self.last_update
                return min(self.capacity, self.tokens + elapsed * self.rate)
            except Exception as e:
                logger.error(f"Available tokens check failed: {e}")
                return 0.0


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate rate limiting"""

    def __init__(self, window_size_seconds: int, max_requests: int):
        self.window_size = window_size_seconds
        self.max_requests = max_requests
        self.requests = deque()

        # CRITICAL FIX: Use RLock
        self.lock = threading.RLock()

    def allow_request(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            try:
                now = time.time()

                # Remove old requests outside window
                while self.requests and self.requests[0] < now - self.window_size:
                    self.requests.popleft()

                # Check if we can allow new request
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True
                return False
            except Exception as e:
                logger.error(f"Rate limit check failed: {e}")
                return False

    def current_rate(self) -> float:
        """Get current request rate"""
        with self.lock:
            try:
                now = time.time()

                # Clean old requests
                while self.requests and self.requests[0] < now - self.window_size:
                    self.requests.popleft()

                return (
                    len(self.requests) / self.window_size
                    if self.window_size > 0
                    else 0.0
                )
            except Exception as e:
                logger.error(f"Rate calculation failed: {e}")
                return 0.0


class CircuitBreaker:
    """Circuit breaker for handling failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = "closed"  # closed, open, half-open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

        # CRITICAL FIX: Use RLock
        self.lock = threading.RLock()

    def record_success(self):
        """Record successful operation"""
        with self.lock:
            try:
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                        logger.info("Circuit breaker closed after recovery")
                elif self.state == "closed":
                    self.failure_count = max(0, self.failure_count - 1)
            except Exception as e:
                logger.error(f"Record success failed: {e}")

    def record_failure(self):
        """Record failed operation"""
        with self.lock:
            try:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if (
                    self.state == "closed"
                    and self.failure_count >= self.failure_threshold
                ):
                    self.state = "open"
                    logger.warning(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )
                elif self.state == "half-open":
                    self.state = "open"
                    self.success_count = 0
                    logger.warning("Circuit breaker reopened during recovery")
            except Exception as e:
                logger.error(f"Record failure failed: {e}")

    def is_available(self) -> bool:
        """Check if circuit allows operations"""
        with self.lock:
            try:
                if self.state == "closed":
                    return True
                elif self.state == "open":
                    # Check if we should try half-open
                    if (
                        self.last_failure_time
                        and time.time() - self.last_failure_time > self.recovery_timeout
                    ):
                        self.state = "half-open"
                        self.success_count = 0
                        logger.info("Circuit breaker entering half-open state")
                        return True
                    return False
                else:  # half-open
                    return True
            except Exception as e:
                logger.error(f"Availability check failed: {e}")
                return False


class ResourceMonitor:
    """Monitor system resources"""

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        queue_threshold: int = 1000,
    ):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.queue_threshold = queue_threshold

        self.cpu_history = deque(maxlen=60)  # 1 minute of seconds
        self.memory_history = deque(maxlen=60)
        self.queue_depths = defaultdict(lambda: deque(maxlen=60))

        # CRITICAL FIX: Add locks and shutdown event
        self._history_lock = threading.RLock()
        self._shutdown_event = threading.Event()

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    # CRITICAL FIX: Interruptible monitoring loop
    def _monitor_loop(self):
        """Background monitoring loop - CRITICAL: With error recovery and interruptible sleep"""

        consecutive_errors = 0
        max_consecutive_errors = 5

        while not self._shutdown_event.is_set():
            try:
                # Monitor CPU
                cpu_percent = psutil.cpu_percent(interval=0.1)

                with self._history_lock:
                    self.cpu_history.append(cpu_percent)

                # Monitor memory
                memory = psutil.virtual_memory()

                with self._history_lock:
                    self.memory_history.append(memory.percent)

                # CRITICAL FIX: Interruptible sleep
                if self._shutdown_event.wait(timeout=1):
                    break

                # CRITICAL FIX: Reset error counter on success
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Resource monitoring error ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                # CRITICAL FIX: Stop monitoring if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Resource monitoring failed repeatedly, stopping")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=5):
                    break

    def get_current_load(self) -> Dict[str, float]:
        """Get current system load"""
        with self._history_lock:
            try:
                cpu_list = list(self.cpu_history)
                memory_list = list(self.memory_history)

                return {
                    "cpu_percent": np.mean(cpu_list) if cpu_list else 0.0,
                    "memory_percent": np.mean(memory_list) if memory_list else 0.0,
                    "cpu_peak": max(cpu_list) if cpu_list else 0.0,
                    "memory_peak": max(memory_list) if memory_list else 0.0,
                }
            except Exception as e:
                logger.error(f"Load calculation failed: {e}")
                return {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "cpu_peak": 0.0,
                    "memory_peak": 0.0,
                }

    def is_overloaded(self) -> Tuple[bool, str]:
        """Check if system is overloaded"""
        try:
            load = self.get_current_load()

            if load["cpu_percent"] > self.cpu_threshold:
                return True, f"CPU overload: {load['cpu_percent']:.1f}%"

            if load["memory_percent"] > self.memory_threshold:
                return True, f"Memory overload: {load['memory_percent']:.1f}%"

            # Check queue depths
            with self._history_lock:
                for queue_name, depths in self.queue_depths.items():
                    depth_list = list(depths)
                    if depth_list and np.mean(depth_list) > self.queue_threshold:
                        return True, f"Queue {queue_name} overloaded"

            return False, "System healthy"
        except Exception as e:
            logger.error(f"Overload check failed: {e}")
            return False, "Unknown"

    def update_queue_depth(self, queue_name: str, depth: int):
        """Update queue depth metric"""
        try:
            with self._history_lock:
                self.queue_depths[queue_name].append(depth)
        except Exception as e:
            logger.error(f"Queue depth update failed: {e}")

    def stop(self, timeout: float = 2.0):
        """Stop monitoring - CRITICAL: Fast shutdown with event"""
        # Signal shutdown
        self._shutdown_event.set()

        # Wait for thread to finish with timeout
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=timeout)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread still alive after shutdown signal")


class PriorityQueue:
    """Priority queue with timeout handling - CRITICAL: Deadlock-free"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = []
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.entry_finder = {}  # Map request_id to entry
        self.counter = 0  # Unique sequence for tie-breaking

    def put(self, request: Request) -> bool:
        """Add request to queue"""
        with self.lock:
            try:
                if len(self.queue) >= self.max_size:
                    return False

                # Priority queue entry: (priority, counter, request)
                entry = [request.priority.value, self.counter, request]
                self.counter += 1

                if request.request_id in self.entry_finder:
                    # Mark old entry as removed
                    old_entry = self.entry_finder[request.request_id]
                    old_entry[-1] = None

                self.entry_finder[request.request_id] = entry
                heapq.heappush(self.queue, entry)
                self.not_empty.notify()
                return True
            except Exception as e:
                logger.error(f"Queue put failed: {e}")
                return False

    # CRITICAL FIX: Prevent deadlock by not holding lock during wait
    def get(self, timeout: Optional[float] = None) -> Optional[Request]:
        """Get highest priority request - CRITICAL: Deadlock-free"""

        end_time = time.time() + timeout if timeout else None

        while True:
            # CRITICAL FIX: Acquire lock only for queue operations
            with self.lock:
                # Clean removed entries
                while self.queue and self.queue[0][-1] is None:
                    heapq.heappop(self.queue)

                if self.queue:
                    priority, count, request = heapq.heappop(self.queue)
                    if request:
                        del self.entry_finder[request.request_id]
                        return request

            # CRITICAL FIX: Check timeout outside lock
            if timeout:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                wait_time = min(0.1, remaining)
            else:
                wait_time = 0.1

            # CRITICAL FIX: Wait with timeout to periodically check queue
            try:
                with self.not_empty:
                    if not self.not_empty.wait(timeout=wait_time):
                        if timeout and time.time() >= end_time:
                            return None
            except Exception as e:
                logger.error(f"Queue wait failed: {e}")
                if timeout and time.time() >= end_time:
                    return None
                time.sleep(0.01)

    def remove(self, request_id: str) -> bool:
        """Remove request from queue"""
        with self.lock:
            try:
                if request_id in self.entry_finder:
                    entry = self.entry_finder[request_id]
                    entry[-1] = None  # Mark as removed
                    del self.entry_finder[request_id]
                    return True
                return False
            except Exception as e:
                logger.error(f"Queue remove failed: {e}")
                return False

    def size(self) -> int:
        """Get queue size"""
        with self.lock:
            try:
                return sum(1 for entry in self.queue if entry[-1] is not None)
            except Exception as e:
                logger.error(f"Queue size check failed: {e}")
                return 0

    def clear_expired(self) -> int:
        """Remove expired requests"""
        now = time.time()
        expired_count = 0

        with self.lock:
            try:
                for request_id, entry in list(self.entry_finder.items()):
                    request = entry[-1]
                    if request and request.deadline and request.deadline < now:
                        entry[-1] = None
                        del self.entry_finder[request_id]
                        expired_count += 1
            except Exception as e:
                logger.error(f"Clear expired failed: {e}")

        return expired_count


class AdaptiveAdmissionController:
    """Adaptive admission control with multiple strategies"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # CRITICAL FIX: Add locks for thread safety and shutdown event
        self._admission_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._is_shutdown = False

        # Rate limiting
        self.global_rate_limiter = TokenBucketRateLimiter(
            rate=config.get("global_rate", 100),  # 100 requests/second
            capacity=config.get("burst_capacity", 200),
        )

        # Per-priority rate limiters
        self.priority_limiters = {
            RequestPriority.CRITICAL: SlidingWindowRateLimiter(1, 50),
            RequestPriority.HIGH: SlidingWindowRateLimiter(1, 30),
            RequestPriority.NORMAL: SlidingWindowRateLimiter(1, 15),
            RequestPriority.LOW: SlidingWindowRateLimiter(1, 5),
            RequestPriority.BATCH: SlidingWindowRateLimiter(1, 2),
        }

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(
            cpu_threshold=config.get("cpu_threshold", 80),
            memory_threshold=config.get("memory_threshold", 80),
        )

        # Request queue
        self.queue = PriorityQueue(max_size=config.get("max_queue_size", 1000))

        # Metrics tracking
        self.metrics = AdmissionMetrics()
        self.metrics_window = deque(maxlen=3600)  # 1 hour of seconds

        # Adaptive parameters
        self.load_factor = 1.0  # Multiplier for admission decisions
        self.rejection_threshold = config.get("rejection_threshold", 0.3)

        # Backpressure settings
        self.backpressure_enabled = config.get("backpressure_enabled", True)
        self.backpressure_threshold = config.get("backpressure_threshold", 0.8)

        # Deferred requests
        self.deferred_queue = deque(maxlen=100)

        # Background threads - CRITICAL: Now interruptible
        self.processor_thread = threading.Thread(
            target=self._process_deferred, daemon=True
        )
        self.processor_thread.start()

        # Statistics update thread
        self.stats_thread = threading.Thread(
            target=self._update_statistics, daemon=True
        )
        self.stats_thread.start()

    def admit(self, request: Request) -> Tuple[AdmissionDecision, Dict[str, Any]]:
        """Main admission control decision"""

        # CRITICAL FIX: Check shutdown
        if self._shutdown_event.is_set():
            return AdmissionDecision.REJECT, {"reason": "system_shutdown"}

        with self._metrics_lock:
            self.metrics.total_requests += 1

        try:
            # Step 1: Circuit breaker check
            if not self.circuit_breaker.is_available():
                with self._metrics_lock:
                    self.metrics.rejected_requests += 1
                return AdmissionDecision.REJECT, {"reason": "circuit_breaker_open"}

            # Step 2: System health check
            health = self._check_system_health()
            if health == SystemHealth.CRITICAL:
                with self._metrics_lock:
                    self.metrics.rejected_requests += 1
                self.circuit_breaker.record_failure()
                return AdmissionDecision.REJECT, {"reason": "system_critical"}

            # Step 3: Rate limiting check
            if not self._check_rate_limits(request):
                with self._metrics_lock:
                    self.metrics.rejected_requests += 1
                return AdmissionDecision.REJECT, {"reason": "rate_limit_exceeded"}

            # Step 4: Resource availability check
            if not self._check_resource_availability(request):
                if request.priority == RequestPriority.CRITICAL:
                    # Try to make room for critical requests
                    self._shed_load()
                else:
                    # Defer or reject based on priority
                    if request.priority.value <= RequestPriority.HIGH.value:
                        return self._defer_request(request)
                    else:
                        with self._metrics_lock:
                            self.metrics.rejected_requests += 1
                        return AdmissionDecision.REJECT, {
                            "reason": "insufficient_resources"
                        }

            # Step 5: Backpressure check
            if self.backpressure_enabled and self._check_backpressure():
                if request.priority.value >= RequestPriority.NORMAL.value:
                    return self._defer_request(request)

            # Step 6: Queue check
            queue_size = self.queue.size()
            if queue_size > self.queue.max_size * 0.9:
                if request.priority.value >= RequestPriority.LOW.value:
                    with self._metrics_lock:
                        self.metrics.rejected_requests += 1
                    return AdmissionDecision.REJECT, {"reason": "queue_full"}

            # Admit the request
            with self._metrics_lock:
                self.metrics.admitted_requests += 1
            self.circuit_breaker.record_success()

            return AdmissionDecision.ADMIT, {
                "queue_position": queue_size,
                "estimated_wait_ms": self._estimate_wait_time(request),
                "system_load": self.resource_monitor.get_current_load(),
            }
        except Exception as e:
            logger.error(f"Admission failed: {e}")
            with self._metrics_lock:
                self.metrics.rejected_requests += 1
            return AdmissionDecision.REJECT, {"reason": f"error: {str(e)}"}

    def _check_system_health(self) -> SystemHealth:
        """Check overall system health"""
        try:
            load = self.resource_monitor.get_current_load()

            if load["cpu_percent"] > 95 or load["memory_percent"] > 95:
                return SystemHealth.CRITICAL
            elif load["cpu_percent"] > 80 or load["memory_percent"] > 80:
                return SystemHealth.OVERLOADED
            elif load["cpu_percent"] > 60 or load["memory_percent"] > 60:
                return SystemHealth.DEGRADED
            else:
                return SystemHealth.HEALTHY
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth.DEGRADED

    def _check_rate_limits(self, request: Request) -> bool:
        """Check if request passes rate limits"""

        try:
            # Global rate limit
            if not self.global_rate_limiter.consume():
                return False

            # Per-priority rate limit
            if request.priority in self.priority_limiters:
                if not self.priority_limiters[request.priority].allow_request():
                    return False

            return True
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False

    def _check_resource_availability(self, request: Request) -> bool:
        """Check if resources are available for request"""
        try:
            load = self.resource_monitor.get_current_load()

            # Estimate resource usage
            estimated_cpu = request.estimated_cost.get("cpu_percent", 5)
            estimated_memory = request.estimated_cost.get("memory_mb", 100)

            # Check if we have headroom
            if load["cpu_percent"] + estimated_cpu > 90:
                return False

            # Memory check (convert MB to percentage)
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            estimated_memory_percent = (
                (estimated_memory / total_memory_mb) * 100 if total_memory_mb > 0 else 0
            )
            if load["memory_percent"] + estimated_memory_percent > 90:
                return False

            return True
        except Exception as e:
            logger.error(f"Resource availability check failed: {e}")
            return False

    def _check_backpressure(self) -> bool:
        """Check if backpressure should be applied"""
        try:
            queue_size = self.queue.size()
            queue_utilization = (
                queue_size / self.queue.max_size if self.queue.max_size > 0 else 0
            )

            return queue_utilization > self.backpressure_threshold
        except Exception as e:
            logger.error(f"Backpressure check failed: {e}")
            return False

    def _defer_request(
        self, request: Request
    ) -> Tuple[AdmissionDecision, Dict[str, Any]]:
        """Defer request for later processing"""

        try:
            # Check if request has deadline
            if request.deadline:
                time_until_deadline = request.deadline - time.time()
                estimated_wait = self._estimate_wait_time(request)

                if estimated_wait > time_until_deadline:
                    with self._metrics_lock:
                        self.metrics.rejected_requests += 1
                    return AdmissionDecision.REJECT, {"reason": "deadline_miss"}

            # Add to deferred queue
            self.deferred_queue.append(request)
            with self._metrics_lock:
                self.metrics.deferred_requests += 1

            return AdmissionDecision.DEFER, {
                "retry_after_ms": 1000,
                "queue_position": len(self.deferred_queue),
            }
        except Exception as e:
            logger.error(f"Defer request failed: {e}")
            with self._metrics_lock:
                self.metrics.rejected_requests += 1
            return AdmissionDecision.REJECT, {"reason": f"defer_error: {str(e)}"}

    def _shed_load(self):
        """Shed load by removing low priority requests"""
        try:
            removed = 0

            # Remove batch requests first
            for request_id in list(self.queue.entry_finder.keys()):
                entry = self.queue.entry_finder.get(request_id)
                if entry and entry[-1] and entry[-1].priority == RequestPriority.BATCH:
                    self.queue.remove(request_id)
                    removed += 1
                    if removed >= 5:
                        break

            logger.info(f"Shed {removed} low priority requests")
        except Exception as e:
            logger.error(f"Load shedding failed: {e}")

    def _estimate_wait_time(self, request: Request) -> float:
        """Estimate wait time for request"""
        try:
            queue_size = self.queue.size()

            # Simple estimation based on queue size and recent throughput
            with self._metrics_lock:
                throughput = self.metrics.throughput_rps

            if throughput > 0:
                base_wait = (queue_size / throughput) * 1000  # ms
            else:
                base_wait = queue_size * 10  # 10ms per request default

            # Adjust for priority
            priority_factor = 1.0 + (request.priority.value * 0.2)

            return base_wait * priority_factor
        except Exception as e:
            logger.error(f"Wait time estimation failed: {e}")
            return 1000.0

    # CRITICAL FIX: Interruptible deferred processing
    def _process_deferred(self):
        """Background processing of deferred requests - CRITICAL: Interruptible"""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while not self._shutdown_event.is_set():
            try:
                if self.deferred_queue:
                    # Check if we can process deferred requests
                    health = self._check_system_health()
                    if health == SystemHealth.HEALTHY:
                        request = self.deferred_queue.popleft()
                        # Re-evaluate admission
                        decision, info = self.admit(request)
                        if decision == AdmissionDecision.ADMIT:
                            logger.info(
                                f"Admitted deferred request {request.request_id}"
                            )

                # CRITICAL FIX: Interruptible sleep
                if self._shutdown_event.wait(timeout=0.1):
                    break
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Error processing deferred requests ({consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Deferred processing failed repeatedly")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=1):
                    break

    # CRITICAL FIX: Interruptible statistics update
    def _update_statistics(self):
        """Update statistics periodically - CRITICAL: Interruptible"""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while not self._shutdown_event.is_set():
            try:
                # Calculate metrics
                queue_depth = self.queue.size()

                with self._metrics_lock:
                    self.metrics.current_queue_depth = queue_depth

                self.resource_monitor.update_queue_depth("main", queue_depth)

                # Update rejection rate
                with self._metrics_lock:
                    if self.metrics.total_requests > 0:
                        self.metrics.rejection_rate = (
                            self.metrics.rejected_requests / self.metrics.total_requests
                        )

                    # Update CPU and memory
                    load = self.resource_monitor.get_current_load()
                    self.metrics.cpu_usage_percent = load["cpu_percent"]
                    self.metrics.memory_usage_percent = load["memory_percent"]

                    # Calculate throughput
                    current_time = time.time()
                    self.metrics_window.append(
                        {
                            "time": current_time,
                            "admitted": self.metrics.admitted_requests,
                        }
                    )

                    if len(self.metrics_window) > 1:
                        time_diff = (
                            self.metrics_window[-1]["time"]
                            - self.metrics_window[0]["time"]
                        )
                        request_diff = (
                            self.metrics_window[-1]["admitted"]
                            - self.metrics_window[0]["admitted"]
                        )
                        if time_diff > 0:
                            self.metrics.throughput_rps = request_diff / time_diff

                # Adaptive load factor
                self._update_load_factor()

                # CRITICAL FIX: Interruptible sleep
                if self._shutdown_event.wait(timeout=1):
                    break
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error updating statistics ({consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Statistics update failed repeatedly")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=5):
                    break

    def _update_load_factor(self):
        """Update adaptive load factor based on metrics"""

        try:
            with self._metrics_lock:
                cpu_usage = self.metrics.cpu_usage_percent
                rejection_rate = self.metrics.rejection_rate

            # Increase load factor if system is underutilized
            if cpu_usage < 50 and rejection_rate < 0.05:
                self.load_factor = min(1.5, self.load_factor * 1.1)

            # Decrease load factor if system is overloaded
            elif cpu_usage > 80 or rejection_rate > 0.2:
                self.load_factor = max(0.5, self.load_factor * 0.9)
        except Exception as e:
            logger.error(f"Load factor update failed: {e}")

    def get_metrics(self) -> AdmissionMetrics:
        """Get current metrics"""
        with self._metrics_lock:
            # Return a copy
            return AdmissionMetrics(
                total_requests=self.metrics.total_requests,
                admitted_requests=self.metrics.admitted_requests,
                rejected_requests=self.metrics.rejected_requests,
                deferred_requests=self.metrics.deferred_requests,
                current_queue_depth=self.metrics.current_queue_depth,
                avg_wait_time_ms=self.metrics.avg_wait_time_ms,
                cpu_usage_percent=self.metrics.cpu_usage_percent,
                memory_usage_percent=self.metrics.memory_usage_percent,
                rejection_rate=self.metrics.rejection_rate,
                throughput_rps=self.metrics.throughput_rps,
            )

    def reset_metrics(self):
        """Reset metrics"""
        with self._metrics_lock:
            self.metrics = AdmissionMetrics()
            self.metrics_window.clear()

    def shutdown(self, timeout: float = 5.0):
        """Shutdown admission controller - CRITICAL: Fast shutdown with events"""
        if self._shutdown_event.is_set():
            return

        logger.info("Shutting down admission controller")

        # Signal shutdown
        self._shutdown_event.set()
        self._is_shutdown = True

        # Stop resource monitor
        self.resource_monitor.stop(timeout=timeout / 3)

        # Wait for threads with timeout split
        deadline = time.time() + timeout

        if self.processor_thread.is_alive():
            remaining = max(0.01, deadline - time.time())
            self.processor_thread.join(timeout=remaining)
            if self.processor_thread.is_alive():
                logger.warning("Processor thread still alive after shutdown signal")

        if self.stats_thread.is_alive():
            remaining = max(0.01, deadline - time.time())
            self.stats_thread.join(timeout=remaining)
            if self.stats_thread.is_alive():
                logger.warning("Stats thread still alive after shutdown signal")

        logger.info("Admission controller shutdown complete")


class AdmissionControlIntegration:
    """Integration with tool selection system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.controller = AdaptiveAdmissionController(config)
        self.request_map = {}  # Map request_id to callback
        self.lock = threading.RLock()

    def check_admission(
        self,
        problem: Any,
        constraints: Dict[str, float],
        priority: RequestPriority = RequestPriority.NORMAL,
        callback: Optional[Callable] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be admitted

        Returns:
            (admitted, info_dict)
        """

        try:
            # Create request
            request = Request(
                request_id=f"req_{time.time()}_{id(problem)}",
                priority=priority,
                estimated_cost={
                    "time_ms": constraints.get("time_budget", 1000),
                    "energy_mj": constraints.get("energy_budget", 1000),
                    "memory_mb": 100,  # Default estimate
                },
                context={"problem": str(problem)[:100], "constraints": constraints},
                deadline=time.time() + constraints.get("timeout", 60),
            )

            # Check admission
            decision, info = self.controller.admit(request)

            if decision == AdmissionDecision.ADMIT:
                # Add to queue for processing
                if self.controller.queue.put(request):
                    with self.lock:
                        if callback:
                            self.request_map[request.request_id] = callback
                    return True, info
                else:
                    return False, {"reason": "queue_full"}

            elif decision == AdmissionDecision.DEFER:
                # Schedule retry
                if callback:
                    retry_delay = info.get("retry_after_ms", 1000) / 1000
                    timer = threading.Timer(
                        retry_delay,
                        lambda: self.check_admission(
                            problem, constraints, priority, callback
                        ),
                    )
                    timer.daemon = True
                    timer.start()
                return False, info

            else:  # REJECT
                return False, info
        except Exception as e:
            logger.error(f"Check admission failed: {e}")
            return False, {"reason": f"error: {str(e)}"}

    def process_next(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Process next admitted request"""

        try:
            request = self.controller.queue.get(timeout=timeout)
            if request:
                with self.lock:
                    callback = self.request_map.pop(request.request_id, None)

                return {"request": request, "callback": callback}
            return None
        except Exception as e:
            logger.error(f"Process next failed: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            metrics = self.controller.get_metrics()
            load = self.controller.resource_monitor.get_current_load()

            return {
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "admitted": metrics.admitted_requests,
                    "rejected": metrics.rejected_requests,
                    "deferred": metrics.deferred_requests,
                    "queue_depth": metrics.current_queue_depth,
                    "rejection_rate": metrics.rejection_rate,
                    "throughput_rps": metrics.throughput_rps,
                },
                "resources": load,
                "circuit_breaker": self.controller.circuit_breaker.state,
                "load_factor": self.controller.load_factor,
            }
        except Exception as e:
            logger.error(f"Get system status failed: {e}")
            return {}

    def shutdown(self, timeout: float = 5.0):
        """Shutdown integration"""
        self.controller.shutdown(timeout=timeout)
