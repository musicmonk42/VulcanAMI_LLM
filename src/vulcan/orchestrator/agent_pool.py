# ============================================================
# VULCAN-AGI Orchestrator - Agent Pool Module
# Agent pool management with lifecycle control, auto-scaling, and recovery
# FULLY FIXED VERSION - Enhanced with proper resource management, state validation, and comprehensive error handling
# TTLCache fallback class added for Python environments without cachetools
# TIMEOUT FIXES - Prevents hanging in tests and production
# WINDOWS MULTIPROCESSING FIX - Worker process doesn't access parent's unpicklable objects
# FIXED: Converted long time.sleep calls to interruptible self._shutdown_event.wait().
# ============================================================

import hashlib
import json
import logging
import multiprocessing
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import psutil with fallback for missing or broken installations
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    # Note: Logger not yet configured at module level, so using logging directly here
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "psutil not available, system resource monitoring will be disabled"
    )

from .agent_lifecycle import (AgentCapability, AgentMetadata, AgentState,
                              JobProvenance, StateTransitionRules,
                              create_agent_metadata, create_job_provenance)
from .task_queues import (CeleryTaskQueue, CustomTaskQueue, RayTaskQueue,
                          TaskQueueInterface, TaskStatus, create_task_queue)

# ============================================================
# CONSTANTS
# ============================================================

# Fallback hardware specification values when psutil is not available
DEFAULT_FALLBACK_MEMORY_GB = 4.0  # Conservative memory estimate
DEFAULT_FALLBACK_STORAGE_GB = 100.0  # Conservative storage estimate

# FIXED: Add cachetools import for LRU cache with TTL
try:
    from cachetools import TTLCache

    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logging.warning("cachetools not available, using dict fallback with manual cleanup")

    # FIXED: Define fallback TTLCache class
    class TTLCache(dict):
        """
        Fallback TTLCache implementation when cachetools is not available.
        Provides basic dict functionality with size limit awareness.
        TTL (time-to-live) is handled manually in the calling code.
        """

        def __init__(self, maxsize: int, ttl: float):
            """
            Initialize TTLCache fallback

            Args:
                maxsize: Maximum number of items
                ttl: Time-to-live in seconds (stored but not enforced by this class)
            """
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl

        def __setitem__(self, key, value):
            """Set item with maxsize check"""
            if len(self) >= self.maxsize and key not in self:
                # Remove oldest item (approximate LRU)
                if self:
                    oldest_key = next(iter(self))
                    del self[oldest_key]
            super().__setitem__(key, value)


logger = logging.getLogger(__name__)


# ============================================================
# STANDALONE WORKER FUNCTION (MUST BE AT MODULE LEVEL FOR PICKLING)
# ============================================================


def _standalone_agent_worker(agent_id: str):
    """
    Standalone agent worker function - runs in separate process
    FIXED: Must be at module level to be picklable on Windows

    This is a minimal stub that just runs without accessing parent state.
    In a production system, this would communicate via IPC (queues, pipes, etc.)

    Args:
        agent_id: Agent identifier
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Agent {agent_id} worker started (standalone)")

    try:
        # This is intentionally minimal to avoid Windows pickling issues
        # In production, implement proper IPC here (multiprocessing.Queue, etc.)
        while True:
            # Short, non-blocking sleep is fine here since it's just a placeholder loop
            time.sleep(0.1)
            # TODO: Poll for tasks via IPC mechanism
            # TODO: Execute tasks
            # TODO: Report results back via IPC
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Agent {agent_id} worker error: {e}")

    logger.info(f"Agent {agent_id} worker stopped")


# ============================================================
# AGENT POOL MANAGER (FULLY FIXED)
# ============================================================


class AgentPoolManager:
    """
    Manages pools of agents with lifecycle control and proper resource management

    Key Features:
    - Automatic agent spawning and retirement
    - State machine validation for all state transitions
    - Memory-bounded provenance tracking with TTL
    - Stale task cleanup to prevent memory leaks
    - Comprehensive error handling and recovery
    - Thread-safe operations throughout
    - FIXED: Proper timeouts to prevent hanging
    - FIXED: Windows multiprocessing compatibility (uses standalone worker function)
    """

    def __init__(
        self,
        max_agents: int = 1000,
        min_agents: int = 10,
        task_queue_type: str = "custom",
        provenance_ttl: int = 3600,
        task_timeout_seconds: int = 300,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize Agent Pool Manager

        Args:
            max_agents: Maximum number of agents in pool
            min_agents: Minimum number of agents to maintain
            task_queue_type: Type of task queue ('ray', 'celery', 'custom')
            provenance_ttl: Time-to-live for provenance records in seconds
            task_timeout_seconds: Default timeout for task assignments
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.max_agents = max_agents
        self.min_agents = min_agents
        self.task_timeout_seconds = task_timeout_seconds

        # Agent tracking
        self.agents: Dict[str, AgentMetadata] = {}
        self.agent_processes: Dict[str, multiprocessing.Process] = {}

        # FIXED: Use TTLCache for provenance to prevent unbounded memory growth
        self.provenance_records = TTLCache(maxsize=10000, ttl=provenance_ttl)
        if not CACHETOOLS_AVAILABLE:
            # Manual TTL tracking when using fallback
            self.provenance_creation_times: Dict[str, float] = {}
            self.provenance_ttl = provenance_ttl

        # Task assignment tracking
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_assignment_times: Dict[str, float] = {}  # task_id -> timestamp

        # Main lock for thread-safe operations
        self.lock = threading.RLock()

        # Task queue initialization
        queue_config = self.config.get("queue_config", {})
        self.task_queue: Optional[TaskQueueInterface] = create_task_queue(
            task_queue_type, **queue_config
        )
        self.task_queue_type = task_queue_type

        # Monitoring and lifecycle management
        self.monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Auto-scaling and recovery
        self.auto_scaler: Optional["AutoScaler"] = None
        self.recovery_manager: Optional["RecoveryManager"] = None

        # Provenance archiving
        self.archive_dir = Path("provenance_archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._last_archive_time = time.time()
        self._archive_lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_jobs_submitted": 0,
            "total_jobs_completed": 0,
            "total_jobs_failed": 0,
            "total_agents_spawned": 0,
            "total_agents_retired": 0,
            "total_recoveries_attempted": 0,
            "total_recoveries_successful": 0,
        }
        self.stats_lock = threading.Lock()

        # Status check throttling
        self.last_status_check = 0
        self.status_check_interval = 5.0  # Seconds

        # Start monitoring
        self._start_monitor()

        # Initialize auto-scaling and recovery managers
        self.auto_scaler = AutoScaler(self)
        self.recovery_manager = RecoveryManager(self)

        # Initialize minimum agents
        self._initialize_agent_pool()

        logger.info(
            f"AgentPoolManager initialized: "
            f"min_agents={min_agents}, max_agents={max_agents}, "
            f"queue_type={task_queue_type}, "
            f"cachetools_available={CACHETOOLS_AVAILABLE}"
        )

    def _init_task_queue(self, task_queue_type: str):
        """Initialize task queue with error handling and fallback"""
        try:
            queue_config = self.config.get("queue_config", {})
            self.task_queue = create_task_queue(task_queue_type, **queue_config)
            logger.info(f"Task queue initialized: {task_queue_type}")
        except ImportError as e:
            logger.warning(f"Failed to initialize {task_queue_type} queue: {e}")
            logger.info("Attempting fallback to custom queue...")
            try:
                if task_queue_type != "custom":
                    self.task_queue = create_task_queue(
                        "custom", **self.config.get("queue_config", {})
                    )
                    logger.info("Fallback to custom queue successful")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback queue: {fallback_error}")
                self.task_queue = None
        except Exception as e:
            logger.error(f"Failed to initialize task queue: {e}")
            self.task_queue = None

    def _initialize_agent_pool(self):
        """Initialize minimum number of agents with diverse capabilities"""
        logger.info(f"Initializing agent pool with {self.min_agents} agents")

        capabilities = list(AgentCapability)
        num_capabilities = len(capabilities)

        for i in range(self.min_agents):
            # Distribute capabilities evenly
            if i < self.min_agents // 2:
                capability = capabilities[i % num_capabilities]
            else:
                capability = AgentCapability.GENERAL

            try:
                agent_id = self.spawn_agent(capability)
                if agent_id:
                    logger.debug(
                        f"Initialized agent {agent_id} with capability {capability.value}"
                    )
            except Exception as e:
                logger.error(f"Failed to spawn agent during initialization: {e}")

        logger.info(f"Agent pool initialized with {len(self.agents)} agents")

    def _start_monitor(self):
        """Start background monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitor_agents, daemon=True, name="AgentPoolMonitor"
            )
            self.monitor_thread.start()
            logger.info("Agent pool monitor started")

    def spawn_agent(
        self,
        capability: AgentCapability = AgentCapability.GENERAL,
        location: str = "local",
        hardware_spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Spawn a new agent

        Args:
            capability: Agent capability
            location: Agent location ('local', 'remote', 'cloud')
            hardware_spec: Hardware specification dictionary

        Returns:
            Agent ID if successful, None otherwise
        """
        with self.lock:
            # Check capacity
            if len(self.agents) >= self.max_agents:
                logger.warning(f"Agent pool at maximum capacity ({self.max_agents})")
                return None

            # Generate unique agent ID
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"

            try:
                # Create agent metadata using factory function
                metadata = create_agent_metadata(
                    agent_id=agent_id,
                    capability=capability,
                    location=location,
                    hardware_spec=hardware_spec or self._get_default_hardware_spec(),
                )

                # Register agent
                self.agents[agent_id] = metadata

                # Spawn agent process/thread based on location
                if location == "local":
                    self._spawn_local_agent(agent_id, metadata)
                elif location == "remote":
                    self._spawn_remote_agent(agent_id, metadata)
                elif location == "cloud":
                    self._spawn_cloud_agent(agent_id, metadata)
                else:
                    logger.warning(
                        f"Unknown location '{location}', defaulting to local"
                    )
                    self._spawn_local_agent(agent_id, metadata)

                # Update statistics
                with self.stats_lock:
                    self.stats["total_agents_spawned"] += 1

                logger.info(
                    f"Spawned agent {agent_id} with capability {capability.value}"
                )
                return agent_id

            except Exception as e:
                logger.error(f"Failed to spawn agent: {e}", exc_info=True)
                # Cleanup on failure
                if agent_id in self.agents:
                    del self.agents[agent_id]
                return None

    def _spawn_local_agent(self, agent_id: str, metadata: AgentMetadata):
        """
        Spawn local agent process
        FIXED: Uses standalone worker function to avoid pickling issues
        """
        try:
            # FIXED: Use standalone function that doesn't reference self
            process = multiprocessing.Process(
                target=_standalone_agent_worker,
                args=(agent_id,),
                daemon=True,
                name=f"Agent-{agent_id}",
            )
            process.start()
            self.agent_processes[agent_id] = process

            # Transition to IDLE state using validated state machine
            metadata.transition_state(AgentState.IDLE, "Local agent process started")

            logger.debug(f"Local agent {agent_id} process started (PID: {process.pid})")

        except Exception as e:
            logger.error(f"Failed to spawn local agent {agent_id}: {e}")
            metadata.transition_state(AgentState.ERROR, f"Spawn failed: {e}")
            metadata.record_error(e, {"phase": "spawn_local"})

    def _spawn_remote_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn remote agent (via SSH, RPC, etc.)"""
        logger.info(f"Spawning remote agent {agent_id}")
        # TODO: Implement remote agent spawning via SSH/RPC
        # For now, just mark as IDLE
        metadata.transition_state(AgentState.IDLE, "Remote agent spawned (stub)")

    def _spawn_cloud_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn cloud agent (AWS, GCP, Azure, etc.)"""
        logger.info(f"Spawning cloud agent {agent_id}")
        # TODO: Implement cloud agent spawning
        # For now, just mark as IDLE
        metadata.transition_state(AgentState.IDLE, "Cloud agent spawned (stub)")

    def retire_agent(self, agent_id: str, force: bool = False) -> bool:
        """
        Retire an agent gracefully

        Args:
            agent_id: Agent identifier
            force: If True, force immediate termination

        Returns:
            True if agent was retired, False otherwise
        """
        with self.lock:
            if agent_id not in self.agents:
                logger.warning(f"Cannot retire agent {agent_id}: not found")
                return False

            metadata = self.agents[agent_id]

            # Cancel any assigned tasks
            tasks_to_cancel = [
                tid for tid, aid in self.task_assignments.items() if aid == agent_id
            ]

            for task_id in tasks_to_cancel:
                logger.warning(f"Cancelling task {task_id} due to agent retirement")
                self._cancel_task(task_id)

            if metadata.state == AgentState.WORKING and not force:
                # Mark for retirement after current task
                metadata.transition_state(
                    AgentState.RETIRING, "Marked for retirement after current task"
                )
                logger.info(
                    f"Agent {agent_id} marked for retirement after current task"
                )
            else:
                # Immediate termination
                metadata.transition_state(
                    AgentState.TERMINATED,
                    "Forced retirement" if force else "Retirement",
                )

                # Cleanup process
                if agent_id in self.agent_processes:
                    process = self.agent_processes[agent_id]

                    if process.is_alive():
                        if not force:
                            # Graceful shutdown
                            process.terminate()
                            process.join(timeout=5)

                        # Force kill if still alive
                        if process.is_alive():
                            logger.warning(f"Force killing agent {agent_id} process")
                            process.kill()
                            process.join(timeout=2)

                        # Close process handle
                        try:
                            process.close()
                        except Exception as e:
                            logger.debug(f"Error closing process handle: {e}")

                    del self.agent_processes[agent_id]

                # Update statistics
                with self.stats_lock:
                    self.stats["total_agents_retired"] += 1

                logger.info(f"Agent {agent_id} terminated")

        return True

    def recover_agent(self, agent_id: str) -> bool:
        """
        Recover a failed agent

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent was recovered, False otherwise
        """
        with self.lock:
            if agent_id not in self.agents:
                logger.warning(f"Cannot recover agent {agent_id}: not found")
                return False

            metadata = self.agents[agent_id]

            # Check if agent can be recovered
            if not metadata.should_recover():
                logger.info(
                    f"Agent {agent_id} should not be recovered (too many errors)"
                )
                return False

            # Validate state transition
            if not metadata.transition_state(
                AgentState.RECOVERING, "Recovery initiated"
            ):
                logger.error(f"Cannot transition agent {agent_id} to RECOVERING state")
                return False

            logger.info(f"Recovering agent {agent_id}")

            # Clean up old process if exists
            if agent_id in self.agent_processes:
                process = self.agent_processes[agent_id]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                try:
                    process.close()
                except Exception:
                    pass
                del self.agent_processes[agent_id]

            # Respawn agent based on location
            success = False
            try:
                if metadata.location == "local":
                    self._spawn_local_agent(agent_id, metadata)
                    success = True
                elif metadata.location == "remote":
                    self._spawn_remote_agent(agent_id, metadata)
                    success = True
                elif metadata.location == "cloud":
                    self._spawn_cloud_agent(agent_id, metadata)
                    success = True

                if success:
                    # Reset error counters
                    metadata.consecutive_errors = 0
                    metadata.transition_state(AgentState.IDLE, "Recovery successful")

                    # Update statistics
                    with self.stats_lock:
                        self.stats["total_recoveries_successful"] += 1

                    logger.info(f"Agent {agent_id} recovered successfully")

            except Exception as e:
                logger.error(f"Failed to recover agent {agent_id}: {e}")
                metadata.transition_state(AgentState.ERROR, f"Recovery failed: {e}")
                success = False

            # Update statistics
            with self.stats_lock:
                self.stats["total_recoveries_attempted"] += 1

            return success

    def submit_job(
        self,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        capability_required: AgentCapability = AgentCapability.GENERAL,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Submit a job to the agent pool
        FIXED: Uses reasonable timeout defaults and proper error handling

        Args:
            graph: Computation graph
            parameters: Job parameters
            priority: Job priority (higher = more important)
            capability_required: Required agent capability
            timeout_seconds: Job timeout in seconds

        Returns:
            Job ID

        Raises:
            RuntimeError: If job queue is full or pool is shutting down
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        # FIXED: Use shorter default timeout (5 seconds instead of 300)
        timeout_seconds = timeout_seconds if timeout_seconds is not None else 5.0

        with self.lock:
            # FIXED: Check shutdown first
            if self._shutdown_event.is_set():
                raise RuntimeError("Agent pool is shutting down")

            # FIXED: Check queue capacity BEFORE accepting job
            if len(self.task_assignments) >= 1000:
                logger.error(f"Job queue full, rejecting job {job_id}")
                raise RuntimeError("Job queue at maximum capacity (1000 tasks)")

            # Create provenance record using factory function
            provenance = create_job_provenance(
                job_id=job_id,
                graph_id=graph.get("id", "unknown"),
                parameters=parameters,
                priority=priority,
                timeout_seconds=timeout_seconds,
            )

            # Store provenance
            self.provenance_records[job_id] = provenance
            if not CACHETOOLS_AVAILABLE:
                self.provenance_creation_times[job_id] = time.time()

            # Update statistics
            with self.stats_lock:
                self.stats["total_jobs_submitted"] += 1

            # FIXED: Archive old provenance if needed
            if time.time() - self._last_archive_time > 3600:
                self._archive_old_provenance()

            # FIXED: Find suitable agent with timeout using proper locking
            agent_id = self._assign_agent_with_timeout(
                capability_required, timeout_seconds
            )

            if agent_id:
                # Direct assignment to available agent
                provenance.agent_id = agent_id
                provenance.hardware_used = self.agents[agent_id].hardware_spec
                self._assign_job_to_agent(job_id, agent_id, graph, parameters)
                logger.info(f"Job {job_id} assigned to agent {agent_id}")
            else:
                # FIXED: If no agent available, mark as failed immediately
                # Don't try to queue to task_queue as that can hang
                logger.warning(
                    f"No agent available for job {job_id} within {timeout_seconds}s"
                )
                provenance.agent_id = "no_agent_available"
                provenance.complete("failed", error="No agent available within timeout")

                # Update statistics
                with self.stats_lock:
                    self.stats["total_jobs_failed"] += 1

        return job_id

    def _assign_agent_with_timeout(
        self, capability: AgentCapability, timeout_seconds: float
    ) -> Optional[str]:
        """
        Assign agent with timeout and proper locking to prevent race conditions
        FIXED: Won't hang if no agents available

        Args:
            capability: Required capability
            timeout_seconds: Timeout in seconds

        Returns:
            Agent ID if assigned, None otherwise
        """
        start_time = time.time()
        retry_delay = 0.05  # Start with 50ms delay
        max_retry_delay = 0.2  # FIXED: Reduced from 1.0 to 0.2 seconds
        max_retries = 10  # FIXED: Maximum number of retries to prevent infinite loops
        retry_count = 0

        while time.time() - start_time < timeout_seconds and retry_count < max_retries:
            # FIXED: Check shutdown event
            if self._shutdown_event.is_set():
                logger.debug("Shutdown requested, aborting agent assignment")
                return None

            # FIXED: Hold lock for entire check-and-spawn operation
            with self.lock:
                agent_id = self._assign_agent(capability)
                if agent_id:
                    return agent_id

                # Try to spawn if under capacity
                if len(self.agents) < self.max_agents:
                    new_agent = self.spawn_agent(capability)
                    if new_agent:
                        # Give agent a moment to initialize
                        time.sleep(0.05)
                        # Try to assign the newly spawned agent
                        agent_id = self._assign_agent(capability)
                        if agent_id:
                            return agent_id
                else:
                    # FIXED: At max capacity and no agents available - fail fast
                    logger.warning(
                        f"At max capacity ({self.max_agents}) with no available agents "
                        f"for capability {capability.value}"
                    )
                    return None

            # FIXED: Increment retry counter
            retry_count += 1

            # Brief wait before retry (outside the lock)
            time.sleep(retry_delay)

            # Exponential backoff up to max delay
            retry_delay = min(retry_delay * 1.5, max_retry_delay)

        logger.warning(
            f"Failed to assign agent with capability {capability.value} "
            f"within {timeout_seconds}s after {retry_count} retries"
        )
        return None

    def _assign_agent(self, capability: AgentCapability) -> Optional[str]:
        """
        Assign an available agent with required capability

        Must be called with lock held.

        Args:
            capability: Required capability

        Returns:
            Agent ID if available, None otherwise
        """
        available_agents = [
            agent_id
            for agent_id, metadata in self.agents.items()
            if metadata.state.can_accept_work()
            and metadata.capability.can_handle_capability(capability)
        ]

        if not available_agents:
            return None

        # Select agent with best performance (lowest failure rate)
        best_agent = min(
            available_agents,
            key=lambda aid: self.agents[aid].tasks_failed
            / max(1, self.agents[aid].tasks_completed),
        )

        return best_agent

    def _assign_job_to_agent(
        self,
        job_id: str,
        agent_id: str,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]],
    ):
        """
        Assign job to specific agent

        Must be called with lock held.

        Args:
            job_id: Job identifier
            agent_id: Agent identifier
            graph: Computation graph
            parameters: Job parameters
        """
        task = {"task_id": job_id, "graph": graph, "parameters": parameters or {}}

        # Queue task for agent
        self.task_assignments[job_id] = agent_id
        self.task_assignment_times[job_id] = time.time()

        # Transition agent to WORKING state
        if agent_id in self.agents:
            self.agents[agent_id].transition_state(
                AgentState.WORKING, f"Assigned job {job_id}"
            )

    def _get_agent_task(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get next task for agent

        Args:
            agent_id: Agent identifier

        Returns:
            Task dictionary if available, None otherwise
        """
        with self.lock:
            for task_id, assigned_agent in self.task_assignments.items():
                if assigned_agent == agent_id:
                    provenance = self.provenance_records.get(task_id)
                    if provenance:
                        provenance.start_execution()

                    return {"task_id": task_id, "provenance": provenance}

        return None

    def _execute_agent_task(
        self, agent_id: str, task: Dict[str, Any], metadata: AgentMetadata
    ) -> Any:
        """
        Execute task on agent

        Args:
            agent_id: Agent identifier
            task: Task dictionary
            metadata: Agent metadata

        Returns:
            Task result

        Raises:
            Exception: If task execution fails
        """
        start_time = time.time()
        task_id = task.get("task_id")
        provenance = task.get("provenance")

        try:
            # TODO: Implement actual task execution
            # For now, simulate task execution
            logger.debug(f"Agent {agent_id} executing task {task_id}")

            # Simulate work
            time.sleep(0.1)

            # Create result
            duration = time.time() - start_time
            result = {
                "status": "completed",
                "agent_id": agent_id,
                "execution_time": duration,
                "timestamp": time.time(),
            }

            # Update agent metadata
            metadata.record_task_completion(success=True, duration_s=duration)

            # Update provenance
            if provenance:
                provenance.complete("success", result=result)
                resource_consumption = {"cpu_seconds": duration}
                if PSUTIL_AVAILABLE:
                    try:
                        resource_consumption["memory_mb"] = (
                            psutil.Process().memory_info().rss / 1024 / 1024
                        )
                    except Exception:
                        pass
                provenance.update_resource_consumption(resource_consumption)

            # Update statistics
            with self.stats_lock:
                self.stats["total_jobs_completed"] += 1

            logger.debug(
                f"Agent {agent_id} completed task {task_id} in {duration:.3f}s"
            )
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Agent {agent_id} task {task_id} failed: {e}")

            # Update agent metadata
            metadata.record_task_completion(success=False, duration_s=duration)
            metadata.record_error(e, {"task_id": task_id, "phase": "execution"})

            # Update provenance
            if provenance:
                provenance.complete("failed", error=str(e))
                provenance.update_resource_consumption({"cpu_seconds": duration})

            # Update statistics
            with self.stats_lock:
                self.stats["total_jobs_failed"] += 1

            raise

    def _complete_agent_task(self, agent_id: str, task_id: str, result: Any):
        """
        Mark task as completed and cleanup

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            result: Task result
        """
        with self.lock:
            # Remove from assignments
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]

            if task_id in self.task_assignment_times:
                del self.task_assignment_times[task_id]

            # Transition agent back to IDLE
            if agent_id in self.agents:
                metadata = self.agents[agent_id]
                metadata.transition_state(AgentState.IDLE, f"Completed task {task_id}")
                metadata.last_active = time.time()

    def _handle_task_failure(self, agent_id: str, task_id: str, error: Exception):
        """
        Handle task failure

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            error: Error that caused failure
        """
        with self.lock:
            # Update provenance
            if task_id in self.provenance_records:
                provenance = self.provenance_records[task_id]
                if not provenance.is_complete():
                    provenance.complete("failed", error=str(error))

            # Remove from assignments
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            if task_id in self.task_assignment_times:
                del self.task_assignment_times[task_id]

            # Return agent to idle
            if agent_id in self.agents:
                metadata = self.agents[agent_id]
                metadata.transition_state(AgentState.IDLE, f"Task {task_id} failed")

    def _cancel_task(self, task_id: str):
        """
        Cancel a task

        Args:
            task_id: Task identifier
        """
        with self.lock:
            # Update provenance
            if task_id in self.provenance_records:
                provenance = self.provenance_records[task_id]
                if not provenance.is_complete():
                    provenance.complete("cancelled")

            # Get assigned agent
            agent_id = self.task_assignments.get(task_id)

            # Remove from assignments
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            if task_id in self.task_assignment_times:
                del self.task_assignment_times[task_id]

            # Return agent to idle if it was working on this task
            if agent_id and agent_id in self.agents:
                metadata = self.agents[agent_id]
                if metadata.state == AgentState.WORKING:
                    metadata.transition_state(
                        AgentState.IDLE, f"Task {task_id} cancelled"
                    )

    def _archive_old_provenance(self):
        """Archive old provenance records to disk"""
        with self._archive_lock:
            try:
                # Manual cleanup for non-TTLCache
                if not CACHETOOLS_AVAILABLE:
                    current_time = time.time()
                    expired_jobs = [
                        job_id
                        for job_id, create_time in self.provenance_creation_times.items()
                        if current_time - create_time > self.provenance_ttl
                    ]

                    if expired_jobs:
                        for job_id in expired_jobs:
                            if job_id in self.provenance_records:
                                del self.provenance_records[job_id]
                            del self.provenance_creation_times[job_id]

                        logger.debug(
                            f"Cleaned up {len(expired_jobs)} expired provenance records"
                        )

                # Archive if too many records
                if len(self.provenance_records) > 9000:
                    timestamp = int(time.time())
                    archive_file = self.archive_dir / f"provenance_{timestamp}.jsonl"

                    # Archive oldest 1000 records
                    records_to_archive = list(self.provenance_records.items())[:1000]

                    with open(archive_file, "w") as f:
                        for job_id, prov in records_to_archive:
                            try:
                                f.write(json.dumps(prov.to_dict(), default=str) + "\n")
                            except Exception as e:
                                logger.error(
                                    f"Failed to serialize provenance {job_id}: {e}"
                                )

                    # Remove archived records from cache
                    with self.lock:
                        for job_id, _ in records_to_archive:
                            if job_id in self.provenance_records:
                                del self.provenance_records[job_id]
                            if (
                                not CACHETOOLS_AVAILABLE
                                and job_id in self.provenance_creation_times
                            ):
                                del self.provenance_creation_times[job_id]

                    self._last_archive_time = time.time()
                    logger.info(
                        f"Archived {len(records_to_archive)} provenance records to {archive_file}"
                    )

            except Exception as e:
                logger.error(f"Failed to archive provenance: {e}", exc_info=True)

    def _monitor_agents(self):
        """
        Monitor agent health and performance with comprehensive cleanup

        FIXED: Converted long time.sleep(10) to interruptible self._shutdown_event.wait(timeout=10).
        """
        logger.info("Agent monitor started")

        # FIXED: Use interruptible wait
        while not self._shutdown_event.is_set():
            try:
                # If shutdown is signaled, break immediately
                if self._shutdown_event.wait(timeout=10):
                    break

                current_time = time.time()

                with self.lock:
                    # FIXED: Clean up stale task assignments
                    stale_tasks = [
                        task_id
                        for task_id, assign_time in self.task_assignment_times.items()
                        if current_time - assign_time > self.task_timeout_seconds
                    ]

                    for task_id in stale_tasks:
                        agent_id = self.task_assignments.get(task_id)
                        logger.warning(
                            f"Cleaning up stale task {task_id} "
                            f"(assigned to {agent_id}, age: {current_time - self.task_assignment_times[task_id]:.1f}s)"
                        )
                        self._cancel_task(task_id)

                    # Check provenance archiving
                    if len(self.provenance_records) > 9000:
                        self._archive_old_provenance()

                    # Monitor each agent
                    agents_to_recover = []
                    agents_to_retire = []

                    for agent_id, metadata in list(self.agents.items()):
                        # Check for stale idle agents
                        if metadata.state == AgentState.IDLE:
                            idle_time = current_time - metadata.last_active
                            if idle_time > 300 and len(self.agents) > self.min_agents:
                                agents_to_retire.append(agent_id)

                        # Check for error agents
                        elif metadata.state == AgentState.ERROR:
                            if metadata.should_recover():
                                agents_to_recover.append(agent_id)
                            else:
                                agents_to_retire.append(agent_id)

                        # Update resource usage for local agents
                        if PSUTIL_AVAILABLE and agent_id in self.agent_processes:
                            process = self.agent_processes[agent_id]
                            if process.is_alive():
                                try:
                                    p = psutil.Process(process.pid)
                                    metadata.resource_usage = {
                                        "cpu_percent": p.cpu_percent(interval=0.1),
                                        "memory_mb": p.memory_info().rss / 1024 / 1024,
                                        "num_threads": p.num_threads(),
                                    }
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    logger.debug(
                                        f"Cannot access process info for agent {agent_id}"
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Error accessing process info for agent {agent_id}: {e}"
                                    )

                # Perform recovery and retirement outside the lock
                for agent_id in agents_to_recover:
                    logger.info(f"Attempting to recover agent {agent_id}")
                    self.recover_agent(agent_id)

                for agent_id in agents_to_retire:
                    logger.info(f"Retiring stale/error agent {agent_id}")
                    self.retire_agent(agent_id)

            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)

        logger.info("Agent monitor stopped")

    def _get_default_hardware_spec(self) -> Dict[str, Any]:
        """Get default hardware specification"""
        try:
            if PSUTIL_AVAILABLE:
                return {
                    "cpu_cores": psutil.cpu_count(logical=True),
                    "cpu_cores_physical": psutil.cpu_count(logical=False),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "gpu_available": self._check_gpu_available(),
                    "storage_gb": psutil.disk_usage("/").total / (1024**3),
                }
            else:
                # Fallback when psutil is not available
                return {
                    "cpu_cores": multiprocessing.cpu_count(),
                    "cpu_cores_physical": multiprocessing.cpu_count(),
                    "memory_gb": DEFAULT_FALLBACK_MEMORY_GB,
                    "gpu_available": self._check_gpu_available(),
                    "storage_gb": DEFAULT_FALLBACK_STORAGE_GB,
                }
        except Exception as e:
            logger.warning(f"Failed to get hardware spec: {e}")
            return {
                "cpu_cores": 1,
                "memory_gb": 1,
                "gpu_available": False,
                "storage_gb": 10,
            }

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status with throttled status checks."""
        current_time = time.time()
        if current_time - self.last_status_check < self.status_check_interval:
            logger.debug(
                "Skipping queue status check due to throttling; returning cached agent data."
            )
            return self._cached_status()

        self.last_status_check = current_time

        with self.lock:
            state_counts = defaultdict(int)
            for metadata in self.agents.values():
                state_counts[metadata.state.value] += 1

            capability_counts = defaultdict(int)
            for metadata in self.agents.values():
                capability_counts[metadata.capability.value] += 1

            health_scores = [m.get_health_score() for m in self.agents.values()]
            avg_health = (
                sum(health_scores) / len(health_scores) if health_scores else 0.0
            )

            queue_status = {}
            if self.task_queue and hasattr(self.task_queue, "get_coordinator_status"):
                try:
                    queue_status = self.task_queue.get_coordinator_status()
                except Exception as e:
                    logger.warning(f"Failed to get queue status: {e}")
            elif self.task_queue:
                try:
                    queue_status = self.task_queue.get_queue_status()
                except Exception as e:
                    logger.warning(
                        f"Failed to get queue status from get_queue_status: {e}"
                    )

            with self.stats_lock:
                stats = dict(self.stats)

            status = {
                "total_agents": len(self.agents),
                "state_distribution": dict(state_counts),
                "capability_distribution": dict(capability_counts),
                "pending_tasks": len(self.task_assignments),
                "average_health_score": avg_health,
                "queue_status": queue_status,
                "statistics": stats,
                "provenance_records_count": len(self.provenance_records),
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
            }
            logger.info("Agent pool status: %s", status)
            return status

    def _cached_status(self) -> Dict[str, Any]:
        """Return cached status to avoid frequent checks."""
        with self.lock:
            state_counts = defaultdict(int)
            for metadata in self.agents.values():
                state_counts[metadata.state.value] += 1

            capability_counts = defaultdict(int)
            for metadata in self.agents.values():
                capability_counts[metadata.capability.value] += 1

            return {
                "total_agents": len(self.agents),
                "state_distribution": dict(state_counts),
                "capability_distribution": dict(capability_counts),
                "queue_status": {},
            }

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of specific agent

        Args:
            agent_id: Agent identifier

        Returns:
            Agent status dictionary or None if not found
        """
        with self.lock:
            if agent_id not in self.agents:
                return None

            metadata = self.agents[agent_id]
            return metadata.get_summary()

    def get_job_provenance(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete provenance for a job

        Args:
            job_id: Job identifier

        Returns:
            Job provenance dictionary or None if not found
        """
        with self.lock:
            if job_id not in self.provenance_records:
                return None

            provenance = self.provenance_records[job_id]
            return provenance.get_summary()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pool statistics

        Returns:
            Statistics dictionary
        """
        with self.stats_lock:
            return dict(self.stats)

    def shutdown(self):
        """Gracefully shutdown agent pool"""
        logger.info("Shutting down agent pool")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop auto-scaler
        if self.auto_scaler:
            try:
                self.auto_scaler.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down auto-scaler: {e}")

        # Stop accepting new jobs and retire all agents
        with self.lock:
            for agent_id in list(self.agents.keys()):
                self.retire_agent(agent_id, force=False)

        # Wait for agents to complete current tasks
        timeout = time.time() + 30
        while time.time() < timeout:
            with self.lock:
                working = any(
                    m.state == AgentState.WORKING for m in self.agents.values()
                )

            if not working:
                break

            time.sleep(0.5)

        # Force terminate remaining agents
        with self.lock:
            for agent_id in list(self.agents.keys()):
                self.retire_agent(agent_id, force=True)

        # Cleanup task queue
        if self.task_queue:
            try:
                self.task_queue.shutdown()
                logger.info("Task queue shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down task queue: {e}")

        # Wait for monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # Final cleanup
        with self.lock:
            self.agents.clear()
            self.agent_processes.clear()
            self.task_assignments.clear()
            self.task_assignment_times.clear()

        logger.info("Agent pool shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logger.debug(f"Error in destructor: {e}")


# ============================================================
# AUTO SCALER
# ============================================================


class AutoScaler:
    """Automatically scale agent pool based on load with proper locking"""

    def __init__(self, pool_manager: AgentPoolManager):
        """
        Initialize auto-scaler

        Args:
            pool_manager: Agent pool manager instance
        """
        self.pool = pool_manager
        self._shutdown_event = threading.Event()
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop, daemon=True, name="AutoScaler"
        )
        self.scaling_thread.start()
        logger.info("Auto-scaler started")

    def _scaling_loop(self):
        """
        Auto-scaling control loop

        FIXED: Converted hardcoded time.sleep(30) to interruptible self._shutdown_event.wait(timeout=30).
        """
        # FIXED: Use interruptible wait
        while not self._shutdown_event.is_set():
            try:
                # If shutdown is signaled, break immediately
                if self._shutdown_event.wait(timeout=30):
                    break

                self._evaluate_and_scale()
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}", exc_info=True)

        logger.info("Auto-scaler stopped")

    def _evaluate_and_scale(self):
        """Evaluate load and scale accordingly with proper locking"""
        with self.pool.lock:
            status = self.pool.get_pool_status()

            total_agents = status["total_agents"]
            idle_agents = status["state_distribution"].get(AgentState.IDLE.value, 0)
            working_agents = status["state_distribution"].get(
                AgentState.WORKING.value, 0
            )
            # FIXED: Use .get() with default to avoid KeyError during shutdown
            pending_tasks = status.get("pending_tasks", 0)

            # Calculate utilization
            if total_agents > 0:
                utilization = working_agents / total_agents
            else:
                utilization = 0.0

            logger.debug(
                f"Auto-scaler evaluation: "
                f"utilization={utilization:.2f}, "
                f"total={total_agents}, "
                f"idle={idle_agents}, "
                f"working={working_agents}, "
                f"pending={pending_tasks}"
            )

            # Scale up if high utilization or pending tasks
            if utilization > 0.8 or pending_tasks > idle_agents:
                agents_to_spawn = min(
                    max(1, pending_tasks - idle_agents),
                    self.pool.max_agents - total_agents,
                )

                if agents_to_spawn > 0:
                    logger.info(f"Scaling up by {agents_to_spawn} agents")
                    for _ in range(agents_to_spawn):
                        self.pool.spawn_agent()

            # Scale down if low utilization
            elif utilization < 0.2 and total_agents > self.pool.min_agents:
                agents_to_retire = min(
                    idle_agents // 2, total_agents - self.pool.min_agents
                )

                if agents_to_retire > 0:
                    idle_agent_ids = [
                        agent_id
                        for agent_id, metadata in self.pool.agents.items()
                        if metadata.state == AgentState.IDLE
                    ][:agents_to_retire]

                    logger.info(f"Scaling down by {agents_to_retire} agents")
                    for agent_id in idle_agent_ids:
                        self.pool.retire_agent(agent_id)

    def shutdown(self):
        """Shutdown auto-scaler"""
        logger.info("Shutting down auto-scaler")
        self._shutdown_event.set()
        if self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5)
        logger.info("Auto-scaler shutdown complete")


# ============================================================
# RECOVERY MANAGER
# ============================================================


class RecoveryManager:
    """Manages agent recovery and fault tolerance"""

    def __init__(self, pool_manager: AgentPoolManager):
        """
        Initialize recovery manager

        Args:
            pool_manager: Agent pool manager instance
        """
        self.pool = pool_manager
        self.recovery_strategies = {
            AgentState.ERROR: self._recover_error_agent,
            AgentState.TERMINATED: self._recover_terminated_agent,
            AgentState.SUSPENDED: self._recover_suspended_agent,
        }
        logger.info("Recovery manager initialized")

    def recover_agent(self, agent_id: str) -> bool:
        """
        Attempt to recover an agent

        Args:
            agent_id: Agent identifier

        Returns:
            True if recovery successful, False otherwise
        """
        if agent_id not in self.pool.agents:
            logger.warning(f"Cannot recover agent {agent_id}: not found")
            return False

        metadata = self.pool.agents[agent_id]

        if metadata.state in self.recovery_strategies:
            strategy = self.recovery_strategies[metadata.state]
            return strategy(agent_id, metadata)

        logger.warning(
            f"No recovery strategy for agent {agent_id} in state {metadata.state}"
        )
        return False

    def _recover_error_agent(self, agent_id: str, metadata: AgentMetadata) -> bool:
        """Recover agent in error state"""
        error_count = len(metadata.error_history)
        consecutive_errors = metadata.consecutive_errors

        logger.info(
            f"Attempting to recover error agent {agent_id}: "
            f"errors={error_count}, consecutive={consecutive_errors}"
        )

        if consecutive_errors < 3:
            # Try recovery
            return self.pool.recover_agent(agent_id)
        elif consecutive_errors < 5:
            # Reset error history and try recovery
            logger.info(f"Resetting error history for agent {agent_id}")
            metadata.error_history = []
            metadata.consecutive_errors = 0
            return self.pool.recover_agent(agent_id)
        else:
            # Too many errors, retire agent
            logger.warning(f"Agent {agent_id} has too many errors, retiring")
            self.pool.retire_agent(agent_id, force=True)
            return False

    def _recover_terminated_agent(self, agent_id: str, metadata: AgentMetadata) -> bool:
        """Recover terminated agent by spawning replacement"""
        if self.pool.get_pool_status()["total_agents"] < self.pool.min_agents:
            logger.info(
                f"Pool below minimum, spawning replacement for terminated agent {agent_id}"
            )
            new_agent_id = self.pool.spawn_agent(
                capability=metadata.capability, location=metadata.location
            )
            return new_agent_id is not None
        return False

    def _recover_suspended_agent(self, agent_id: str, metadata: AgentMetadata) -> bool:
        """Recover suspended agent"""
        logger.info(f"Recovering suspended agent {agent_id}")
        return metadata.transition_state(AgentState.IDLE, "Recovered from suspension")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "AgentPoolManager",
    "AutoScaler",
    "RecoveryManager",
    "CACHETOOLS_AVAILABLE",
    "TTLCache",
]
