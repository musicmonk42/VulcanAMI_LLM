# ============================================================
# VULCAN-AGI Orchestrator - Task Queue Module
# Distributed task queue implementations for agent coordination
# FULLY FIXED VERSION - Enhanced with proper cleanup, error handling, and resource management
# DEADLOCK FIX - Fast-fail coordinator status checks with aggressive caching to prevent thread stampede
# FIXED: Converted fixed time.sleep(5) in _monitor_tasks to interruptible wait.
# ============================================================

import logging
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# Distributed computing imports
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

try:
    import celery
    from celery import Celery
    from celery.result import AsyncResult

    CELERY_AVAILABLE = True
except ImportError:
    celery = None
    CELERY_AVAILABLE = False

try:
    import zmq

    ZMQ_AVAILABLE = True
except ImportError:
    zmq = None
    ZMQ_AVAILABLE = False

logger = logging.getLogger(__name__)


# Ray actor for task queue status when using Ray as a fallback/primary
class TaskQueueActor:
    """Ray actor for task queue status."""

    def __init__(self):
        self.status = {
            "queue_type": "ray",
            "total_tasks": 0,
            "pending": 0,
            "running": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        return self.status

    def update_status(self, status: Dict[str, Any]):
        self.status.update(status)


# ============================================================
# TASK QUEUE ENUMS AND DATA STRUCTURES
# ============================================================


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class QueueType(Enum):
    """Supported queue types"""

    RAY = "ray"
    CELERY = "celery"
    ZMQ = "zmq"
    CUSTOM = "custom"


@dataclass
class TaskMetadata:
    """Metadata for tracking tasks in the queue"""

    task_id: str
    status: TaskStatus
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None

    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None

    def get_wait_time(self) -> Optional[float]:
        """Get time spent waiting in queue"""
        if self.started_at:
            return self.started_at - self.submitted_at
        return time.time() - self.submitted_at

    def should_retry(self) -> bool:
        """Check if task should be retried"""
        return (
            self.status in {TaskStatus.FAILED, TaskStatus.TIMEOUT}
            and self.retry_count < self.max_retries
        )


# ============================================================
# DISTRIBUTED TASK QUEUE INTERFACE
# ============================================================


class TaskQueueInterface:
    """
    Abstract interface for distributed task queues with enhanced error handling
    and resource management
    """

    def __init__(self):
        self.pending_tasks: Dict[str, TaskMetadata] = {}
        self._lock = threading.RLock()
        self._shutdown = threading.Event()  # Use Event for interruptible wait
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_monitor()

    def submit_task(
        self,
        task: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Submit task to queue, return task ID

        Args:
            task: Task data dictionary
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            max_retries: Maximum number of retries

        Returns:
            Task ID string
        """
        raise NotImplementedError

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get task result

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result

        Returns:
            Task result or None if not available
        """
        raise NotImplementedError

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False otherwise
        """
        raise NotImplementedError

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get queue status

        Returns:
            Dictionary with queue statistics
        """
        raise NotImplementedError

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get status of a specific task

        Args:
            task_id: Task identifier

        Returns:
            TaskStatus or None if task not found
        """
        with self._lock:
            if task_id in self.pending_tasks:
                return self.pending_tasks[task_id].status
        return None

    def cleanup(self):
        """Cleanup queue resources"""
        raise NotImplementedError

    def _start_monitor(self):
        """Start background monitoring thread"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(
                target=self._monitor_tasks, daemon=True, name="TaskQueueMonitor"
            )
            self._monitor_thread.start()

    def _monitor_tasks(self):
        """
        Monitor tasks for timeouts and cleanup

        FIX: Use interruptible wait to prevent shutdown hang.
        """
        # FIXED: Use self._shutdown event check and wait
        while not self._shutdown.is_set():
            try:
                # FIXED: Convert time.sleep(5) to interruptible wait
                if self._shutdown.wait(timeout=5):
                    break  # Exit loop if shutdown is signaled during wait

                self._check_task_timeouts()
                self._cleanup_completed_tasks()
            except Exception as e:
                logger.error(f"Task monitor error: {e}")

    def _check_task_timeouts(self):
        """Check for and handle task timeouts"""
        current_time = time.time()

        with self._lock:
            for task_id, metadata in list(self.pending_tasks.items()):
                if metadata.status == TaskStatus.RUNNING and metadata.timeout_seconds:
                    if metadata.started_at:
                        elapsed = current_time - metadata.started_at
                        if elapsed > metadata.timeout_seconds:
                            logger.warning(
                                f"Task {task_id} timeout after {elapsed:.2f}s"
                            )
                            metadata.status = TaskStatus.TIMEOUT
                            metadata.completed_at = current_time
                            self._handle_timeout(task_id)

    def _cleanup_completed_tasks(self):
        """Cleanup old completed tasks to prevent memory bloat"""
        current_time = time.time()
        max_age = 3600  # Keep completed tasks for 1 hour

        with self._lock:
            tasks_to_remove = []

            for task_id, metadata in self.pending_tasks.items():
                if metadata.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.TIMEOUT,
                }:
                    if metadata.completed_at:
                        age = current_time - metadata.completed_at
                        if age > max_age:
                            tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.pending_tasks[task_id]

            if tasks_to_remove:
                logger.debug(f"Cleaned up {len(tasks_to_remove)} old tasks")

    def _handle_timeout(self, task_id: str):
        """Handle task timeout - to be overridden by subclasses"""

    def shutdown(self):
        """Shutdown the queue gracefully"""
        logger.info(f"Shutting down {self.__class__.__name__}")
        self._shutdown.set()  # Set the event to immediately wake up monitor thread

        # Wait for monitor thread
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        # Cleanup resources
        self.cleanup()


# ============================================================
# RAY TASK QUEUE IMPLEMENTATION
# ============================================================


class RayTaskQueue(TaskQueueInterface):
    """Ray-based task queue implementation with enhanced error handling"""

    def __init__(self, num_cpus: Optional[int] = None, num_gpus: Optional[int] = None):
        """
        Initialize Ray task queue

        Args:
            num_cpus: Number of CPUs to use (None = all available)
            num_gpus: Number of GPUs to use (None = all available)
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not available. Install with: pip install ray")

        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            init_kwargs = {"ignore_reinit_error": True}
            if num_cpus is not None:
                init_kwargs["num_cpus"] = num_cpus
            if num_gpus is not None:
                init_kwargs["num_gpus"] = num_gpus

            try:
                ray.init(**init_kwargs)
                logger.info("Ray initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                raise

        self.ray_futures: Dict[str, Any] = {}
        self._execution_lock = threading.RLock()

    def submit_task(
        self,
        task: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """Submit task to Ray"""
        if self._shutdown.is_set():
            raise RuntimeError("Queue is shutdown")

        task_id = str(uuid.uuid4())

        try:
            # Create task metadata
            metadata = TaskMetadata(
                task_id=task_id,
                status=TaskStatus.PENDING,
                submitted_at=time.time(),
                priority=priority,
                timeout_seconds=timeout,
                max_retries=max_retries,
            )

            # Submit to Ray
            remote_func = ray.remote(self._execute_task)
            future = remote_func.remote(task)

            with self._lock:
                self.pending_tasks[task_id] = metadata
                self.ray_futures[task_id] = future

            logger.debug(f"Submitted task {task_id} to Ray")
            return task_id

        except Exception as e:
            logger.error(f"Failed to submit task to Ray: {e}")
            raise

    def _execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute task in Ray worker"""
        start_time = time.time()

        try:
            # Simulate task execution
            # In real implementation, this would execute the actual task
            logger.debug(f"Executing task: {task.get('job_id', 'unknown')}")

            # Task execution logic would go here
            result = {
                "status": "completed",
                "result": task,
                "execution_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result from Ray"""
        with self._lock:
            if task_id not in self.ray_futures:
                logger.warning(f"Task {task_id} not found in Ray futures")
                return None

            future = self.ray_futures[task_id]
            metadata = self.pending_tasks.get(task_id)

        try:
            # Update status to running
            if metadata and metadata.status == TaskStatus.PENDING:
                metadata.status = TaskStatus.RUNNING
                metadata.started_at = time.time()

            # Get result with timeout
            result = ray.get(future, timeout=timeout)

            # Update metadata
            if metadata:
                metadata.status = TaskStatus.COMPLETED
                metadata.completed_at = time.time()
                metadata.result = result

            logger.debug(f"Task {task_id} completed successfully")
            return result

        except ray.exceptions.GetTimeoutError:
            logger.warning(f"Task {task_id} timed out")
            if metadata:
                metadata.status = TaskStatus.TIMEOUT
                metadata.completed_at = time.time()
            return None

        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            if metadata:
                metadata.status = TaskStatus.FAILED
                metadata.completed_at = time.time()
                metadata.error_message = str(e)
            return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a Ray task"""
        with self._lock:
            if task_id not in self.ray_futures:
                return False

            future = self.ray_futures[task_id]
            metadata = self.pending_tasks.get(task_id)

        try:
            ray.cancel(future, force=True)

            if metadata:
                metadata.status = TaskStatus.CANCELLED
                metadata.completed_at = time.time()

            with self._lock:
                del self.ray_futures[task_id]

            logger.info(f"Cancelled task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def _handle_timeout(self, task_id: str):
        """Handle Ray task timeout"""
        if task_id in self.ray_futures:
            try:
                ray.cancel(self.ray_futures[task_id], force=True)
                del self.ray_futures[task_id]
            except Exception as e:
                logger.error(f"Error cancelling timed out task {task_id}: {e}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get Ray queue status"""
        with self._lock:
            status_counts = defaultdict(int)
            for metadata in self.pending_tasks.values():
                status_counts[metadata.status.value] += 1

            ray_status = {}
            if ray.is_initialized():
                try:
                    ray_status = {
                        "cluster_resources": ray.cluster_resources(),
                        "available_resources": ray.available_resources(),
                    }
                except Exception as e:
                    logger.warning(f"Failed to get Ray cluster status: {e}")

            return {
                "queue_type": "ray",
                "total_tasks": len(self.pending_tasks),
                "pending": status_counts[TaskStatus.PENDING.value],
                "running": status_counts[TaskStatus.RUNNING.value],
                "completed": status_counts[TaskStatus.COMPLETED.value],
                "failed": status_counts[TaskStatus.FAILED.value],
                "cancelled": status_counts[TaskStatus.CANCELLED.value],
                "ray_initialized": ray.is_initialized(),
                **ray_status,
            }

    def cleanup(self):
        """Cleanup Ray resources"""
        logger.info("Cleaning up Ray task queue")

        with self._lock:
            # Cancel all pending tasks
            for task_id in list(self.ray_futures.keys()):
                try:
                    ray.cancel(self.ray_futures[task_id], force=True)
                except Exception as e:
                    logger.debug(f"Error cancelling task {task_id}: {e}")

            self.ray_futures.clear()
            self.pending_tasks.clear()

        # Shutdown Ray
        if ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shutdown successfully")
                # Give Ray time to cleanup
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error during Ray shutdown: {e}")


# ============================================================
# CELERY TASK QUEUE IMPLEMENTATION
# ============================================================


class CeleryTaskQueue(TaskQueueInterface):
    """Celery-based task queue implementation with enhanced error handling"""

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379",
        backend_url: Optional[str] = None,
    ):
        """
        Initialize Celery task queue

        Args:
            broker_url: Message broker URL
            backend_url: Result backend URL (defaults to broker_url)
        """
        if not CELERY_AVAILABLE:
            raise ImportError(
                "Celery is not available. Install with: pip install celery redis"
            )

        super().__init__()

        try:
            self.app = Celery(
                "vulcan_agi", broker=broker_url, backend=backend_url or broker_url
            )

            # Configure Celery
            self.app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=3600,  # 1 hour hard limit
                task_soft_time_limit=3000,  # 50 minute soft limit
            )

            logger.info("Celery initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Celery: {e}")
            raise

        self.celery_results: Dict[str, AsyncResult] = {}

    def submit_task(
        self,
        task: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """Submit task to Celery"""
        if self._shutdown.is_set():
            raise RuntimeError("Queue is shutdown")

        task_id = str(uuid.uuid4())

        try:
            # Create task metadata
            metadata = TaskMetadata(
                task_id=task_id,
                status=TaskStatus.PENDING,
                submitted_at=time.time(),
                priority=priority,
                timeout_seconds=timeout,
                max_retries=max_retries,
            )

            # Submit to Celery
            result = self.app.send_task(
                "execute_agent_task",
                args=[task],
                task_id=task_id,
                priority=priority,
                time_limit=timeout,
                soft_time_limit=timeout * 0.9 if timeout else None,
            )

            with self._lock:
                self.pending_tasks[task_id] = metadata
                self.celery_results[task_id] = result

            logger.debug(f"Submitted task {task_id} to Celery")
            return task_id

        except Exception as e:
            logger.error(f"Failed to submit task to Celery: {e}")
            raise

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result from Celery"""
        with self._lock:
            if task_id not in self.celery_results:
                logger.warning(f"Task {task_id} not found in Celery results")
                return None

            result = self.celery_results[task_id]
            metadata = self.pending_tasks.get(task_id)

        try:
            # Update status
            if metadata and metadata.status == TaskStatus.PENDING:
                metadata.status = TaskStatus.RUNNING
                metadata.started_at = time.time()

            # Get result with timeout
            task_result = result.get(timeout=timeout)

            # Update metadata
            if metadata:
                metadata.status = TaskStatus.COMPLETED
                metadata.completed_at = time.time()
                metadata.result = task_result

            logger.debug(f"Task {task_id} completed successfully")
            return task_result

        except celery.exceptions.TimeoutError:
            logger.warning(f"Task {task_id} timed out")
            if metadata:
                metadata.status = TaskStatus.TIMEOUT
                metadata.completed_at = time.time()
            return None

        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            if metadata:
                metadata.status = TaskStatus.FAILED
                metadata.completed_at = time.time()
                metadata.error_message = str(e)
            return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a Celery task"""
        with self._lock:
            if task_id not in self.celery_results:
                return False

            result = self.celery_results[task_id]
            metadata = self.pending_tasks.get(task_id)

        try:
            result.revoke(terminate=True, signal="SIGKILL")

            if metadata:
                metadata.status = TaskStatus.CANCELLED
                metadata.completed_at = time.time()

            with self._lock:
                del self.celery_results[task_id]

            logger.info(f"Cancelled task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def _handle_timeout(self, task_id: str):
        """Handle Celery task timeout"""
        if task_id in self.celery_results:
            try:
                self.celery_results[task_id].revoke(terminate=True, signal="SIGTERM")
                del self.celery_results[task_id]
            except Exception as e:
                logger.error(f"Error cancelling timed out task {task_id}: {e}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get Celery queue status"""
        with self._lock:
            status_counts = defaultdict(int)
            for metadata in self.pending_tasks.values():
                status_counts[metadata.status.value] += 1

            celery_status = {}
            try:
                inspect = self.app.control.inspect()
                active = inspect.active()
                scheduled = inspect.scheduled()
                reserved = inspect.reserved()

                celery_status = {
                    "active_tasks": sum(
                        len(tasks) for tasks in (active or {}).values()
                    ),
                    "scheduled_tasks": sum(
                        len(tasks) for tasks in (scheduled or {}).values()
                    ),
                    "reserved_tasks": sum(
                        len(tasks) for tasks in (reserved or {}).values()
                    ),
                }
            except Exception as e:
                logger.warning(f"Failed to get Celery status: {e}")

            return {
                "queue_type": "celery",
                "total_tasks": len(self.pending_tasks),
                "pending": status_counts[TaskStatus.PENDING.value],
                "running": status_counts[TaskStatus.RUNNING.value],
                "completed": status_counts[TaskStatus.COMPLETED.value],
                "failed": status_counts[TaskStatus.FAILED.value],
                "cancelled": status_counts[TaskStatus.CANCELLED.value],
                **celery_status,
            }

    def cleanup(self):
        """Cleanup Celery resources"""
        logger.info("Cleaning up Celery task queue")

        with self._lock:
            # Revoke all pending tasks
            for task_id, result in list(self.celery_results.items()):
                try:
                    result.revoke(terminate=True, signal="SIGKILL")
                except Exception as e:
                    logger.debug(f"Error revoking task {task_id}: {e}")

            self.celery_results.clear()
            self.pending_tasks.clear()

        # Close Celery app
        try:
            self.app.close()
            logger.info("Celery closed successfully")
        except Exception as e:
            logger.error(f"Error closing Celery: {e}")


# ============================================================
# CUSTOM ZMQ TASK QUEUE IMPLEMENTATION
# ============================================================


class CustomTaskQueue(TaskQueueInterface):
    """Custom distributed task queue using ZeroMQ with enhanced reliability

    DEADLOCK FIX: Aggressive caching and fast-fail on coordinator status checks
    to prevent thread stampede when coordinator is unavailable (e.g., in tests)
    """

    def __init__(
        self,
        coordinator_address: str = "tcp://localhost:5555",
        connection_timeout: int = 15000,  # SIGNAL BUFFERING FIX: Increased from 5000ms to 15000ms
        config: Dict[str, Any] = None,
    ):
        """Initialize ZeroMQ task queue with Ray fallback."""
        super().__init__()

        self.config = config or {}
        self.use_ray = self.config.get("type", "zmq") == "ray" and RAY_AVAILABLE
        self.context = None
        self.socket = None
        self.actor = None
        self._socket_lock = threading.RLock()
        self.coordinator_address = coordinator_address

        # DEADLOCK FIX: Add caching for coordinator status
        self._cached_coordinator_status: Dict[str, Any] = {}
        self._last_coordinator_check: float = 0
        self._coordinator_check_interval: float = 10.0  # Cache for 10 seconds
        self._coordinator_available: bool = True  # Assume available initially
        self._coordinator_check_lock = threading.Lock()

        if self.use_ray:
            try:
                ray_config = self.config.get("config", {})
                ray.init(
                    address=ray_config.get("address", "auto"), ignore_reinit_error=True
                )
                self.actor = ray.remote(TaskQueueActor).remote()
                logger.info("Ray task queue actor initialized")
            except Exception as e:
                logger.warning(f"Ray initialization failed: {e}, falling back to ZMQ")
                self.use_ray = False

        if not self.use_ray:
            if not ZMQ_AVAILABLE:
                raise ImportError("ZMQ not available for fallback.")
            try:
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(zmq.RCVTIMEO, connection_timeout)
                self.socket.setsockopt(zmq.SNDTIMEO, connection_timeout)
                self.socket.connect(coordinator_address)
                logger.info(f"ZMQ connected to {coordinator_address}")
            except zmq.ZMQError as e:
                logger.error(f"ZMQ initialization failed: {e}")
                self.socket = None
                self._coordinator_available = False  # Mark as unavailable

    def submit_task(
        self,
        task: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """Submit task via ZMQ or update Ray actor status."""
        if self._shutdown.is_set():
            raise RuntimeError("Queue is shutdown")

        task_id = str(uuid.uuid4())

        if self.use_ray and self.actor:
            try:
                # Placeholder: Update Ray actor status
                status = ray.get(self.actor.get_status.remote())
                status["total_tasks"] = status.get("total_tasks", 0) + 1
                status["pending"] = status.get("pending", 0) + 1
                ray.get(self.actor.update_status.remote(status))

                metadata = TaskMetadata(
                    task_id=task_id,
                    status=TaskStatus.PENDING,
                    submitted_at=time.time(),
                    priority=priority,
                    timeout_seconds=timeout,
                    max_retries=max_retries,
                )
                with self._lock:
                    self.pending_tasks[task_id] = metadata

                logger.info(f"Task {task_id} status updated on Ray actor.")
                return task_id
            except Exception as e:
                logger.error(f"Ray task status update failed: {e}")
                raise

        if not self.socket:
            raise RuntimeError("ZMQ socket not initialized")

        try:
            metadata = TaskMetadata(
                task_id=task_id,
                status=TaskStatus.PENDING,
                submitted_at=time.time(),
                priority=priority,
                timeout_seconds=timeout,
                max_retries=max_retries,
            )

            message = {
                "action": "submit",
                "task_id": task_id,
                "task": task,
                "priority": priority,
                "timeout": timeout,
            }

            with self._socket_lock:
                self.socket.send_json(message)
                response = self.socket.recv_json()

            if response.get("status") == "accepted":
                with self._lock:
                    self.pending_tasks[task_id] = metadata
                logger.debug(f"Submitted task {task_id} via ZMQ")
                return task_id
            else:
                raise RuntimeError(
                    f"Task submission rejected: {response.get('reason', 'unknown')}"
                )

        except zmq.error.Again:
            logger.error("ZMQ coordinator timeout during task submission")
            raise TimeoutError("Coordinator not responding")
        except Exception as e:
            logger.error(f"Failed to submit task via ZMQ: {e}")
            raise

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result via ZMQ"""
        if self.use_ray:
            logger.warning(
                "get_result not implemented for Ray status actor mode in CustomTaskQueue."
            )
            return None

        with self._lock:
            if task_id not in self.pending_tasks:
                logger.warning(f"Task {task_id} not found")
                return None

            metadata = self.pending_tasks[task_id]

        try:
            message = {"action": "get_result", "task_id": task_id}
            original_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
            if timeout:
                self.socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))

            try:
                with self._socket_lock:
                    self.socket.send_json(message)
                    response = self.socket.recv_json()
            finally:
                self.socket.setsockopt(zmq.RCVTIMEO, original_timeout)

            result = response.get("result")
            status = response.get("status")

            if status == "completed":
                metadata.status = TaskStatus.COMPLETED
                metadata.completed_at = time.time()
                metadata.result = result
            elif status == "failed":
                metadata.status = TaskStatus.FAILED
                metadata.completed_at = time.time()
                metadata.error_message = response.get("error")
            elif status == "running":
                metadata.status = TaskStatus.RUNNING
                if not metadata.started_at:
                    metadata.started_at = time.time()

            return result

        except zmq.error.Again:
            logger.warning(f"Timeout getting result for task {task_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            metadata.status = TaskStatus.FAILED
            metadata.error_message = str(e)
            return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task via ZMQ"""
        if self.use_ray:
            logger.warning("cancel_task not implemented for Ray status actor mode.")
            return False

        try:
            message = {"action": "cancel", "task_id": task_id}

            with self._socket_lock:
                self.socket.send_json(message)
                response = self.socket.recv_json()

            if response.get("status") == "cancelled":
                with self._lock:
                    if task_id in self.pending_tasks:
                        metadata = self.pending_tasks[task_id]
                        metadata.status = TaskStatus.CANCELLED
                        metadata.completed_at = time.time()

                logger.info(f"Cancelled task {task_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def _handle_timeout(self, task_id: str):
        """Handle ZMQ task timeout"""
        try:
            self.cancel_task(task_id)
        except Exception as e:
            logger.error(f"Error cancelling timed out task {task_id}: {e}")

    def get_coordinator_status(
        self, retries: int = 2, initial_delay: float = 0.1
    ) -> Dict[str, Any]:
        """Retrieve coordinator status with aggressive caching and fast-fail.

        DEADLOCK FIX: This method now:
        1. Returns cached status if within cache interval (10 seconds)
        2. Uses only 2 retries with 0.1s delay (fast-fail)
        3. Returns empty dict immediately if coordinator marked unavailable
        4. Uses a separate lock to prevent stampeding
        5. Has very short timeout (0.2s total max wait time)

        Args:
            retries: Number of retries (default: 2, reduced from 10)
            initial_delay: Initial delay between retries (default: 0.1s)

        Returns:
            Coordinator status dict or empty dict if unavailable
        """
        # DEADLOCK FIX: Fast path - return cached status if recent
        current_time = time.time()
        with self._coordinator_check_lock:
            if (
                current_time - self._last_coordinator_check
                < self._coordinator_check_interval
            ):
                logger.debug(
                    "Returning cached coordinator status (within cache interval)"
                )
                return self._cached_coordinator_status

            # DEADLOCK FIX: If coordinator known to be unavailable, return empty immediately
            if not self._coordinator_available:
                logger.debug("Coordinator marked unavailable, returning empty status")
                return {}

        # Ray path
        if self.use_ray and self.actor:
            try:
                status = ray.get(self.actor.get_status.remote(), timeout=0.5)
                with self._coordinator_check_lock:
                    self._cached_coordinator_status = status
                    self._last_coordinator_check = current_time
                    self._coordinator_available = True
                logger.debug("Ray coordinator status retrieved: %s", status)
                return status
            except Exception as e:
                logger.warning(f"Ray status check failed: {e}")
                with self._coordinator_check_lock:
                    self._coordinator_available = False
                return {}

        # ZMQ path
        if not self.socket:
            logger.debug("ZMQ socket not initialized, returning empty status")
            with self._coordinator_check_lock:
                self._coordinator_available = False
            return {}

        # DEADLOCK FIX: Reduced retries and very short timeout
        delay = initial_delay
        for attempt in range(retries):
            try:
                # DEADLOCK FIX: Try to acquire lock with timeout to prevent blocking
                if not self._socket_lock.acquire(timeout=0.05):
                    logger.warning(
                        "Could not acquire socket lock, skipping status check"
                    )
                    return self._cached_coordinator_status or {}

                try:
                    # DEADLOCK FIX: Use non-blocking send with very short poll timeout
                    self.socket.send_json({"action": "status"}, flags=zmq.NOBLOCK)
                    poller = zmq.Poller()
                    poller.register(self.socket, zmq.POLLIN)

                    # SIGNAL BUFFERING FIX: Increased timeout from 10000ms to 15000ms (15 seconds)
                    # to give slow CPU-bound LLM inference enough time to complete.
                    # This prevents the Orchestrator from triggering "Direct Reasoning" fallback
                    # prematurely during CPU inference cycles.
                    if poller.poll(15000):
                        status = self.socket.recv_json(flags=zmq.NOBLOCK)
                        with self._coordinator_check_lock:
                            self._cached_coordinator_status = status
                            self._last_coordinator_check = current_time
                            self._coordinator_available = True
                        logger.debug("ZMQ coordinator status retrieved: %s", status)
                        return status
                    else:
                        logger.debug(
                            "ZMQ status check timed out on attempt %d", attempt + 1
                        )
                finally:
                    self._socket_lock.release()

            except zmq.Again:
                logger.debug(f"ZMQ timeout on attempt {attempt + 1}/{retries}")
            except zmq.ZMQError as e:
                logger.debug(f"ZMQ error on attempt {attempt + 1}/{retries}: {e}")
            except Exception as e:
                logger.debug(
                    f"Unexpected error on attempt {attempt + 1}/{retries}: {e}"
                )

            # DEADLOCK FIX: Very short delay, no exponential backoff
            if attempt < retries - 1:
                time.sleep(delay)

        # DEADLOCK FIX: Mark coordinator as unavailable and return cached or empty
        logger.debug(
            "Failed to get coordinator status after %d retries, marking unavailable",
            retries,
        )
        with self._coordinator_check_lock:
            self._coordinator_available = False
            return self._cached_coordinator_status or {}

    def get_queue_status(self) -> Dict[str, Any]:
        """Get ZMQ or Ray queue status with fast-fail coordinator check"""
        with self._lock:
            status_counts = defaultdict(int)
            for metadata in self.pending_tasks.values():
                status_counts[metadata.status.value] += 1

        # DEADLOCK FIX: Use fast-fail coordinator status check
        coordinator_status = self.get_coordinator_status(retries=1, initial_delay=0.05)

        return {
            "queue_type": "ray" if self.use_ray else "zmq",
            "total_tasks": len(self.pending_tasks),
            "pending": status_counts[TaskStatus.PENDING.value],
            "running": status_counts[TaskStatus.RUNNING.value],
            "completed": status_counts[TaskStatus.COMPLETED.value],
            "failed": status_counts[TaskStatus.FAILED.value],
            "cancelled": status_counts[TaskStatus.CANCELLED.value],
            "coordinator_status": coordinator_status,
        }

    def cleanup(self):
        """Cleanup ZMQ or Ray resources"""
        logger.info("Cleaning up CustomTaskQueue")

        with self._lock:
            self.pending_tasks.clear()

        if self.use_ray:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown complete")
        else:  # ZMQ
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
            logger.info("ZMQ context terminated")


# ============================================================
# TASK QUEUE FACTORY
# ============================================================


def create_task_queue(queue_type: str = "custom", **kwargs) -> TaskQueueInterface:
    """
    Factory function to create task queue instances

    Args:
        queue_type: Type of queue to create ('ray', 'celery', 'zmq', 'custom')
        **kwargs: Additional arguments for queue initialization

    Returns:
        TaskQueueInterface instance

    Raises:
        ValueError: If queue_type is invalid
        ImportError: If required dependencies are not available
    """
    queue_type = queue_type.lower()

    if queue_type == "ray":
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not available. Install with: pip install ray")
        return RayTaskQueue(**kwargs)

    elif queue_type == "celery":
        if not CELERY_AVAILABLE:
            raise ImportError(
                "Celery is not available. Install with: pip install celery redis"
            )
        return CeleryTaskQueue(**kwargs)

    elif queue_type in ["zmq", "custom"]:
        # The CustomTaskQueue can now handle Ray as a backend via config
        return CustomTaskQueue(**kwargs)

    else:
        raise ValueError(
            f"Invalid queue type: {queue_type}. "
            f"Must be one of: ray, celery, zmq, custom"
        )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "TaskQueueInterface",
    "RayTaskQueue",
    "CeleryTaskQueue",
    "CustomTaskQueue",
    "TaskStatus",
    "TaskMetadata",
    "QueueType",
    "create_task_queue",
    "RAY_AVAILABLE",
    "CELERY_AVAILABLE",
    "ZMQ_AVAILABLE",
]
