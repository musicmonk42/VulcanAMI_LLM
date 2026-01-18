# ============================================================
# VULCAN-AGI Orchestrator - Variants Module
# Specialized orchestrator variants: Parallel, Fault-Tolerant, Adaptive
# FULLY FIXED VERSION - Enhanced with proper cleanup, error handling, and timeouts
# Python 3.8+ compatible
#
# Non-Blocking Job Submission:
# This module implements both blocking and non-blocking job submission strategies.
# 
# BLOCKING MODE (default):
# - Simple and predictable execution model
# - Executor threads block until job completes
# - Cannot manage other agents during job execution
# - Best for single-agent systems or when simplicity is preferred
#
# POLLING MODE:
# - Submits jobs without blocking executor threads
# - Returns job handle immediately for status tracking
# - Orchestrator can manage multiple agents concurrently
# - Requires manual polling via get_job_status() or wait_for_job()
# - Best for multi-agent systems or when orchestrator needs to multiplex
#
# ASYNC MODE (reserved for future):
# - Fully asynchronous callback-based execution
# - Not yet implemented
#
# Example usage:
#     # Blocking mode (default, backward compatible):
#     orchestrator = ParallelOrchestrator(config, sys, deps)
#     result = await orchestrator.step_parallel(history, context)
#     
#     # Non-blocking polling mode:
#     orchestrator = ParallelOrchestrator(
#         config, sys, deps, 
#         timeout_strategy=TimeoutStrategy.POLLING
#     )
#     job_id = orchestrator.step_parallel_nonblocking(history, context)
#     # Do other work...
#     status = orchestrator.get_job_status(job_id)
#     result = orchestrator.wait_for_job(job_id, timeout_ms=10000)
# ============================================================

import asyncio
import enum
import logging
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, CancelledError
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .collective import ModalityType, VULCANAGICollective
from .dependencies import EnhancedCollectiveDeps

logger = logging.getLogger(__name__)


# ============================================================
# TIMEOUT CONSTANTS AND ENUMS
# ============================================================

# Default time budget for parallel orchestrator operations (in milliseconds)
# TASK 3 FIX: Increased from 5000ms to 300000ms (5 minutes) to account for
# slow local LLM inference (~1s per token on CPU) and observed system latencies
# of 99-180 seconds. This prevents the orchestrator from timing out and
# orphaning valid long-running reasoning tasks that would have succeeded.
# Configurable via agent_config.time_budget_ms if needed.
DEFAULT_TIME_BUDGET_MS = 300000


class TimeoutStrategy(enum.Enum):
    """
    Strategy for handling long-running operations.
    
    BLOCKING: Traditional blocking execution (default, backward compatible)
              - Simple execution model
              - Blocks until job completes or times out
              - Cannot manage other agents during execution
              
    POLLING:  Non-blocking execution with manual polling
              - Returns job handle immediately
              - Requires manual status checks via get_job_status()
              - Allows orchestrator to manage multiple agents
              
    ASYNC:    Fully asynchronous callback-based execution
              - Reserved for future implementation
              - Will support event-driven workflows
    """
    BLOCKING = "blocking"
    POLLING = "polling"
    ASYNC = "async"


class JobStatus(enum.Enum):
    """Status of a submitted job"""
    PENDING = "pending"      # Job submitted but not yet started
    RUNNING = "running"      # Job is currently executing
    COMPLETE = "complete"    # Job completed successfully
    FAILED = "failed"        # Job failed with error
    CANCELLED = "cancelled"  # Job was cancelled
    TIMEOUT = "timeout"      # Job exceeded timeout


# ============================================================
# PYTHON VERSION COMPATIBILITY
# ============================================================

PYTHON_VERSION = sys.version_info
SUPPORTS_EXECUTOR_TIMEOUT = PYTHON_VERSION >= (3, 9)

if not SUPPORTS_EXECUTOR_TIMEOUT:
    logger.info("Python < 3.9 detected, using manual timeout for executor shutdown")


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================


class PerceptionError(Exception):
    """Raised when perception phase fails"""


class ReasoningError(Exception):
    """Raised when reasoning phase fails"""


class ExecutionError(Exception):
    """Raised when execution phase fails"""


# ============================================================
# EXECUTOR SHUTDOWN HELPER
# ============================================================


def shutdown_executor_with_timeout(executor, executor_name: str, timeout: float = 5.0):
    """
    Shutdown executor with timeout support for Python 3.8+

    Args:
        executor: ProcessPoolExecutor or ThreadPoolExecutor instance
        executor_name: Name for logging
        timeout: Timeout in seconds
    """
    if executor is None:
        return

    try:
        # Check if already shutdown by trying to submit a dummy task
        try:
            future = executor.submit(lambda: None)
            future.cancel()
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e).lower():
                logger.debug(f"{executor_name} already shutdown")
                return

        # Try shutdown with timeout if supported
        if SUPPORTS_EXECUTOR_TIMEOUT:
            try:
                executor.shutdown(wait=True, timeout=timeout)
                logger.info(f"{executor_name} shutdown complete")
                return
            except TypeError:
                # timeout parameter not supported, fall through to manual method
                logger.debug(
                    f"{executor_name}: timeout parameter not supported, using manual shutdown"
                )

        # Manual shutdown for Python < 3.9 or if timeout parameter failed
        executor.shutdown(wait=False)

        # Wait with manual timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if executor is done
            try:
                # For ThreadPoolExecutor, check _threads attribute
                if hasattr(executor, "_threads"):
                    if not any(t.is_alive() for t in executor._threads):
                        logger.info(f"{executor_name} shutdown complete")
                        return
                # For ProcessPoolExecutor, check _processes attribute
                elif hasattr(executor, "_processes"):
                    if not executor._processes or not any(
                        p.is_alive() for p in executor._processes.values()
                    ):
                        logger.info(f"{executor_name} shutdown complete")
                        return
            except (AttributeError, RuntimeError):
                # Executor already shut down or attributes not accessible
                logger.info(f"{executor_name} shutdown complete")
                return

            time.sleep(0.1)

        logger.warning(f"{executor_name} shutdown timeout after {timeout}s")

    except Exception as e:
        logger.error(f"Error shutting down {executor_name}: {e}")
        # Try force shutdown as last resort
        try:
            executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Operation failed: {e}")


# ============================================================
# PARALLEL ORCHESTRATOR
# ============================================================


class ParallelOrchestrator(VULCANAGICollective):
    """
    TRUE parallel execution with proper process/thread separation

    Features:
    - Concurrent perception and memory operations
    - Process-based reasoning for CPU-intensive tasks
    - Thread-based execution for I/O-bound tasks
    - Parallel learning and reflection
    - Proper resource cleanup
    - Python 3.8+ compatible
    - Non-blocking job submission for multi-agent orchestration
    
    Timeout Strategies:
    - BLOCKING (default): Traditional blocking execution, backward compatible
    - POLLING: Non-blocking with manual status checks, enables multi-agent management
    - ASYNC: Reserved for future callback-based implementation
    """

    def __init__(
        self, 
        config: Any, 
        sys: Any, 
        deps: EnhancedCollectiveDeps, 
        redis_client: Optional[Any] = None,
        timeout_strategy: TimeoutStrategy = TimeoutStrategy.BLOCKING
    ):
        """
        Initialize Parallel Orchestrator

        Args:
            config: Configuration object
            sys: System state object
            deps: Dependencies container
            redis_client: Optional Redis client for state persistence across workers/restarts
            timeout_strategy: Strategy for handling long-running operations (default: BLOCKING)
        """
        super().__init__(config, sys, deps, redis_client=redis_client)

        # Initialize executors with reasonable limits
        max_workers_process = getattr(config, "max_parallel_processes", 4)
        max_workers_thread = getattr(config, "max_parallel_threads", 8)

        self.process_executor = ProcessPoolExecutor(max_workers=max_workers_process)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers_thread)
        
        # Timeout strategy configuration
        self.timeout_strategy = timeout_strategy
        
        # Job tracking for non-blocking execution
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.RLock()

        logger.info(
            f"ParallelOrchestrator initialized: "
            f"processes={max_workers_process}, threads={max_workers_thread}, "
            f"timeout_strategy={timeout_strategy.value}, "
            f"python_version={PYTHON_VERSION.major}.{PYTHON_VERSION.minor}"
        )

    async def step_parallel(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute cognitive cycle with TRUE parallel phases
        
        Behavior depends on timeout_strategy:
        - BLOCKING: Blocks until job completes (default, backward compatible)
        - POLLING: Delegates to step_parallel_nonblocking() and immediately waits
        - ASYNC: Reserved for future implementation

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result dictionary
        """
        # Check timeout strategy
        if self.timeout_strategy == TimeoutStrategy.POLLING:
            # In POLLING mode, step_parallel behaves as submit + immediate wait
            job_id = self.step_parallel_nonblocking(history, context)
            timeout_ms = context.get("time_budget_ms", DEFAULT_TIME_BUDGET_MS)
            return self.wait_for_job(job_id, timeout_ms)
        elif self.timeout_strategy == TimeoutStrategy.ASYNC:
            raise NotImplementedError("ASYNC timeout strategy not yet implemented")
        
        # BLOCKING mode (default): original blocking implementation
        if self._shutdown_event.is_set():
            return self._create_fallback_result("System is shutting down")

        start_time = time.time()
        # Use the named constant for default time budget
        timeout = context.get("time_budget_ms", DEFAULT_TIME_BUDGET_MS) / 1000

        try:
            # Phase 1: Parallel perception and memory operations
            perception_task = asyncio.create_task(
                asyncio.to_thread(self._perceive_and_understand, history, context)
            )
            memory_task = asyncio.create_task(
                asyncio.to_thread(self._update_memory_async, history)
            )

            # Wait for perception with timeout
            try:
                perception_result = await asyncio.wait_for(
                    perception_task, timeout=timeout * 0.3
                )
            except asyncio.TimeoutError:
                logger.error("Perception phase timeout")
                return self._create_fallback_result("Perception timeout")

            # Phase 2: Reasoning in process executor (CPU-intensive)
            loop = asyncio.get_event_loop()
            try:
                plan = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.process_executor,
                        self._reason_and_plan,
                        perception_result,
                        context,
                    ),
                    timeout=timeout * 0.4,
                )
            except asyncio.TimeoutError:
                logger.error("Reasoning phase timeout")
                return self._create_fallback_result("Reasoning timeout")

            # Phase 3: Validation in thread (I/O-bound)
            try:
                validated_plan = await asyncio.wait_for(
                    asyncio.to_thread(self._validate_and_ensure_safety, plan, context),
                    timeout=timeout * 0.15,
                )
            except asyncio.TimeoutError:
                logger.error("Validation phase timeout")
                validated_plan = self._create_safe_fallback("Validation timeout", plan)

            # Phase 4: Execution in thread
            try:
                execution_result = await asyncio.wait_for(
                    asyncio.to_thread(self._execute_action, validated_plan),
                    timeout=timeout * 0.15,
                )
            except asyncio.TimeoutError:
                logger.error("Execution phase timeout")
                execution_result = self._create_fallback_result("Execution timeout")

            # Phase 5 & 6: Parallel learning, reflection, and memory completion
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        asyncio.to_thread(
                            self._learn_and_adapt, execution_result, perception_result
                        ),
                        asyncio.to_thread(self._reflect_and_improve),
                        memory_task,
                        return_exceptions=True,
                    ),
                    timeout=max(1.0, timeout - (time.time() - start_time)),
                )
            except asyncio.TimeoutError:
                logger.warning("Learning/reflection phase timeout (non-critical)")
            except Exception as e:
                logger.error(f"Error in parallel learning/reflection: {e}")

            # Update metrics and state
            duration = time.time() - start_time
            self.deps.metrics.record_step(duration, execution_result)

            self._update_system_state(execution_result, duration)
            self._add_provenance(execution_result)

            return execution_result

        except Exception as e:
            logger.error(f"Error in parallel cognitive cycle: {e}", exc_info=True)
            self.deps.metrics.increment_counter("errors_total")
            return self._create_fallback_result(str(e))

    def step_parallel_nonblocking(
        self, history: List[Any], context: Dict[str, Any]
    ) -> str:
        """
        Submit cognitive cycle job without blocking, return job ID immediately.
        
        This method enables the orchestrator to manage multiple agents concurrently
        by submitting the cognitive cycle to the executor and returning immediately.
        The caller can then poll job status or wait for completion.
        
        Use this when:
        - Managing multiple agents that need concurrent processing
        - Orchestrator needs to perform other work while jobs execute
        - Need fine-grained control over job lifecycle (cancel, timeout, etc.)
        
        Use blocking step_parallel() when:
        - Single agent system where blocking is acceptable
        - Simpler execution model is preferred
        - Don't need to multiplex between agents
        
        Args:
            history: Historical observations
            context: Context dictionary
            
        Returns:
            job_id: Unique identifier for tracking this job
            
        Raises:
            RuntimeError: If executor is shutdown or unavailable
            
        Examples:
            >>> # Submit job and continue with other work
            >>> job_id = orchestrator.step_parallel_nonblocking(history, context)
            >>> # Manage other agents...
            >>> # Later, check status
            >>> status = orchestrator.get_job_status(job_id)
            >>> if status == JobStatus.COMPLETE:
            >>>     result = orchestrator.get_job_result(job_id)
            >>>
            >>> # Or wait with timeout
            >>> result = orchestrator.wait_for_job(job_id, timeout_ms=10000)
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Cannot submit job: System is shutting down")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create wrapper function that executes the full cognitive cycle
        def execute_job():
            """Execute the full cognitive cycle and return result"""
            try:
                # Create event loop for this thread if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the cognitive cycle
                result = loop.run_until_complete(
                    self._step_parallel_blocking_impl(history, context)
                )
                return result
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}", exc_info=True)
                return self._create_fallback_result(str(e))
        
        # Submit to thread executor (handles both sync and async work)
        try:
            future = self.thread_executor.submit(execute_job)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to submit job: {e}")
        
        # Track job with metadata
        with self._jobs_lock:
            self._jobs[job_id] = {
                "future": future,
                "status": JobStatus.PENDING,
                "submitted_at": time.time(),
                "history": history,
                "context": context,
                "result": None,
                "error": None
            }
        
        logger.info(f"Job {job_id} submitted for non-blocking execution")
        return job_id

    async def _step_parallel_blocking_impl(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal blocking implementation of parallel cognitive cycle.
        
        This is the core logic extracted from step_parallel() to enable
        both blocking and non-blocking execution modes.
        
        Args:
            history: Historical observations
            context: Context dictionary
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        timeout = context.get("time_budget_ms", DEFAULT_TIME_BUDGET_MS) / 1000

        try:
            # Phase 1: Parallel perception and memory operations
            perception_task = asyncio.create_task(
                asyncio.to_thread(self._perceive_and_understand, history, context)
            )
            memory_task = asyncio.create_task(
                asyncio.to_thread(self._update_memory_async, history)
            )

            # Wait for perception with timeout
            try:
                perception_result = await asyncio.wait_for(
                    perception_task, timeout=timeout * 0.3
                )
            except asyncio.TimeoutError:
                logger.error("Perception phase timeout")
                return self._create_fallback_result("Perception timeout")

            # Phase 2: Reasoning in process executor (CPU-intensive)
            loop = asyncio.get_event_loop()
            try:
                plan = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.process_executor,
                        self._reason_and_plan,
                        perception_result,
                        context,
                    ),
                    timeout=timeout * 0.4,
                )
            except asyncio.TimeoutError:
                logger.error("Reasoning phase timeout")
                return self._create_fallback_result("Reasoning timeout")

            # Phase 3: Validation in thread (I/O-bound)
            try:
                validated_plan = await asyncio.wait_for(
                    asyncio.to_thread(self._validate_and_ensure_safety, plan, context),
                    timeout=timeout * 0.15,
                )
            except asyncio.TimeoutError:
                logger.error("Validation phase timeout")
                validated_plan = self._create_safe_fallback("Validation timeout", plan)

            # Phase 4: Execution in thread
            try:
                execution_result = await asyncio.wait_for(
                    asyncio.to_thread(self._execute_action, validated_plan),
                    timeout=timeout * 0.15,
                )
            except asyncio.TimeoutError:
                logger.error("Execution phase timeout")
                execution_result = self._create_fallback_result("Execution timeout")

            # Phase 5 & 6: Parallel learning, reflection, and memory completion
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        asyncio.to_thread(
                            self._learn_and_adapt, execution_result, perception_result
                        ),
                        asyncio.to_thread(self._reflect_and_improve),
                        memory_task,
                        return_exceptions=True,
                    ),
                    timeout=max(1.0, timeout - (time.time() - start_time)),
                )
            except asyncio.TimeoutError:
                logger.warning("Learning/reflection phase timeout (non-critical)")
            except Exception as e:
                logger.error(f"Error in parallel learning/reflection: {e}")

            # Update metrics and state
            duration = time.time() - start_time
            self.deps.metrics.record_step(duration, execution_result)

            self._update_system_state(execution_result, duration)
            self._add_provenance(execution_result)

            return execution_result

        except Exception as e:
            logger.error(f"Error in parallel cognitive cycle: {e}", exc_info=True)
            self.deps.metrics.increment_counter("errors_total")
            return self._create_fallback_result(str(e))

    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get current status of a submitted job.
        
        Thread-safe status check that doesn't block. Call this periodically
        to monitor job progress when using non-blocking submission.
        
        Args:
            job_id: Job identifier returned by step_parallel_nonblocking()
            
        Returns:
            Current job status (PENDING/RUNNING/COMPLETE/FAILED/CANCELLED)
            
        Raises:
            KeyError: If job_id is not found (invalid or already cleaned up)
            
        Examples:
            >>> status = orchestrator.get_job_status(job_id)
            >>> if status == JobStatus.COMPLETE:
            >>>     result = orchestrator.get_job_result(job_id)
            >>> elif status == JobStatus.FAILED:
            >>>     logger.error(f"Job failed: {orchestrator.get_job_error(job_id)}")
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            future = job["future"]
            
            # Update status based on future state
            if future.cancelled():
                job["status"] = JobStatus.CANCELLED
            elif future.done():
                try:
                    # Check if completed successfully or with error
                    result = future.result(timeout=0)
                    job["status"] = JobStatus.COMPLETE
                    job["result"] = result
                except CancelledError:
                    job["status"] = JobStatus.CANCELLED
                except Exception as e:
                    job["status"] = JobStatus.FAILED
                    job["error"] = str(e)
            elif job["status"] == JobStatus.PENDING:
                # Check if job has started (future is running)
                if future.running():
                    job["status"] = JobStatus.RUNNING
            
            return job["status"]

    def wait_for_job(
        self, job_id: str, timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wait for job completion and return result.
        
        Blocks until job completes, fails, or timeout expires. This provides
        a convenient way to wait for specific jobs while still allowing the
        orchestrator to manage other work in the meantime.
        
        Args:
            job_id: Job identifier returned by step_parallel_nonblocking()
            timeout_ms: Maximum time to wait in milliseconds (None = wait forever)
            
        Returns:
            Job execution result dictionary
            
        Raises:
            KeyError: If job_id is not found
            TimeoutError: If timeout expires before job completes
            RuntimeError: If job was cancelled or failed
            
        Examples:
            >>> # Wait with timeout
            >>> try:
            >>>     result = orchestrator.wait_for_job(job_id, timeout_ms=30000)
            >>>     print(f"Job completed: {result}")
            >>> except TimeoutError:
            >>>     orchestrator.cancel_job(job_id)
            >>>     print("Job cancelled due to timeout")
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} not found")
            job = self._jobs[job_id]
            future = job["future"]
        
        timeout_sec = timeout_ms / 1000.0 if timeout_ms is not None else None
        start_time = time.time()
        
        logger.info(f"Waiting for job {job_id} (timeout={timeout_ms}ms)")
        
        try:
            # Poll with timeout
            while True:
                status = self.get_job_status(job_id)
                
                if status == JobStatus.COMPLETE:
                    with self._jobs_lock:
                        result = self._jobs[job_id]["result"]
                    logger.info(f"Job {job_id} completed successfully")
                    return result
                
                elif status == JobStatus.FAILED:
                    with self._jobs_lock:
                        error = self._jobs[job_id].get("error", "Unknown error")
                    logger.error(f"Job {job_id} failed: {error}")
                    raise RuntimeError(f"Job failed: {error}")
                
                elif status == JobStatus.CANCELLED:
                    logger.warning(f"Job {job_id} was cancelled")
                    raise RuntimeError("Job was cancelled")
                
                # Check timeout
                if timeout_sec is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout_sec:
                        logger.error(f"Job {job_id} timeout after {elapsed:.1f}s")
                        # Mark as timeout in job metadata
                        with self._jobs_lock:
                            self._jobs[job_id]["status"] = JobStatus.TIMEOUT
                        raise TimeoutError(f"Job timeout after {elapsed:.1f}s")
                
                # Sleep briefly before next poll
                time.sleep(0.1)
                
        except (TimeoutError, RuntimeError):
            raise
        except Exception as e:
            logger.error(f"Error waiting for job {job_id}: {e}", exc_info=True)
            raise RuntimeError(f"Error waiting for job: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Attempts to cancel the job. If the job hasn't started yet, cancellation
        will succeed. If it's already running, cancellation may or may not succeed
        depending on what the job is doing.
        
        Args:
            job_id: Job identifier returned by step_parallel_nonblocking()
            
        Returns:
            True if job was successfully cancelled, False otherwise
            
        Raises:
            KeyError: If job_id is not found
            
        Examples:
            >>> if orchestrator.cancel_job(job_id):
            >>>     print("Job cancelled successfully")
            >>> else:
            >>>     print("Job already completed or couldn't be cancelled")
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            future = job["future"]
            
            # Attempt to cancel
            cancelled = future.cancel()
            
            if cancelled:
                job["status"] = JobStatus.CANCELLED
                logger.info(f"Job {job_id} cancelled successfully")
            else:
                logger.warning(
                    f"Job {job_id} could not be cancelled (likely already running or complete)"
                )
            
            return cancelled

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """
        Get result of a completed job without waiting.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job result dictionary
            
        Raises:
            KeyError: If job_id is not found
            RuntimeError: If job is not complete
        """
        status = self.get_job_status(job_id)
        
        if status != JobStatus.COMPLETE:
            raise RuntimeError(f"Job {job_id} is not complete (status: {status.value})")
        
        with self._jobs_lock:
            return self._jobs[job_id]["result"]

    def get_job_error(self, job_id: str) -> Optional[str]:
        """
        Get error message from a failed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Error message or None if job didn't fail
            
        Raises:
            KeyError: If job_id is not found
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} not found")
            return self._jobs[job_id].get("error")

    def cleanup_job(self, job_id: str) -> None:
        """
        Remove job from tracking (cleanup completed/failed/cancelled jobs).
        
        Call this after retrieving results to free memory. Jobs are NOT
        automatically cleaned up to allow result retrieval.
        
        Args:
            job_id: Job identifier
            
        Raises:
            KeyError: If job_id is not found
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} not found")
            del self._jobs[job_id]
        logger.debug(f"Job {job_id} cleaned up")

    def get_active_jobs(self) -> List[str]:
        """
        Get list of all active (non-complete) job IDs.
        
        Returns:
            List of job IDs that are PENDING, RUNNING, or in other active states
        """
        with self._jobs_lock:
            active = []
            for job_id, job in self._jobs.items():
                status = job["status"]
                if status in (JobStatus.PENDING, JobStatus.RUNNING):
                    active.append(job_id)
            return active

    def _update_memory_async(self, history: List[Any]):
        """
        Async memory consolidation and cleanup

        Args:
            history: Historical observations

        Returns:
            True on success
        """
        logger.debug("Starting async memory update")

        try:
            # Compress old memories if we have enough
            if len(self.execution_history) > 10:
                if (
                    hasattr(self.deps, "compressed_memory")
                    and self.deps.compressed_memory
                ):
                    older_memories = list(self.execution_history)[:-10]
                    if older_memories:
                        try:
                            self.deps.compressed_memory.compress_batch(older_memories)
                            logger.debug(f"Compressed {len(older_memories)} memories")
                        except Exception as e:
                            logger.debug(f"Failed to compress memories: {e}")

            # Clear multimodal cache periodically
            if hasattr(self.deps.multimodal, "clear_cache"):
                if len(self.execution_history) % 100 == 0:
                    try:
                        self.deps.multimodal.clear_cache()
                        logger.debug("Cleared multimodal cache")
                    except Exception as e:
                        logger.debug(f"Failed to clear cache: {e}")

            logger.debug("Async memory update completed")
            return True

        except Exception as e:
            logger.error(f"Error in async memory update: {e}")
            return False

    def shutdown(self):
        """Gracefully shutdown parallel orchestrator with Python 3.8+ compatibility"""
        logger.info("Shutting down ParallelOrchestrator")
        
        # Cancel all pending jobs
        with self._jobs_lock:
            job_ids = list(self._jobs.keys())
        
        if job_ids:
            logger.info(f"Cancelling {len(job_ids)} pending jobs")
            for job_id in job_ids:
                try:
                    self.cancel_job(job_id)
                except Exception as e:
                    logger.debug(f"Error cancelling job {job_id}: {e}")

        # Shutdown executors with timeout handling
        shutdown_executor_with_timeout(
            self.process_executor, "Process executor", timeout=5.0
        )

        shutdown_executor_with_timeout(
            self.thread_executor, "Thread executor", timeout=5.0
        )

        # Call parent shutdown
        super().shutdown()

    def __del__(self):
        """Cleanup executors on deletion"""
        try:
            if hasattr(self, "process_executor"):
                self.process_executor.shutdown(wait=False)
            if hasattr(self, "thread_executor"):
                self.thread_executor.shutdown(wait=False)
            if hasattr(self, "_shutdown_event") and not self._shutdown_event.is_set():
                super().shutdown()
        except Exception as e:
            logger.debug(f"Error in destructor: {e}")


# ============================================================
# FAULT TOLERANT ORCHESTRATOR
# ============================================================


class FaultTolerantOrchestrator(VULCANAGICollective):
    """
    Fault-tolerant orchestrator with automatic recovery

    Features:
    - Multiple retry attempts with exponential backoff
    - Fallback strategies for each phase
    - Error history tracking
    - Graceful degradation
    """

    def __init__(self, config: Any, sys: Any, deps: EnhancedCollectiveDeps, redis_client: Optional[Any] = None):
        """
        Initialize Fault Tolerant Orchestrator

        Args:
            config: Configuration object
            sys: System state object
            deps: Dependencies container
            redis_client: Optional Redis client for state persistence across workers/restarts
        """
        super().__init__(config, sys, deps, redis_client=redis_client)

        # Fallback strategies for different error types
        self.fallback_strategies = {
            "perception_error": self._perception_fallback,
            "reasoning_error": self._reasoning_fallback,
            "execution_error": self._execution_fallback,
        }

        # Error tracking
        self.error_history = deque(maxlen=100)
        self._error_lock = threading.RLock()

        logger.info("FaultTolerantOrchestrator initialized")

    def step_with_recovery(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute with automatic recovery from failures

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result dictionary
        """
        max_retries = getattr(self.config, "max_retries", 3)

        for attempt in range(max_retries):
            try:
                result = self.step(history, context)

                # Record success
                with self._error_lock:
                    self.error_history.append(
                        {"timestamp": time.time(), "attempt": attempt, "success": True}
                    )

                return result

            except PerceptionError as e:
                logger.warning(
                    f"Perception error on attempt {attempt + 1}/{max_retries}: {e}"
                )

                with self._error_lock:
                    self.error_history.append(
                        {
                            "timestamp": time.time(),
                            "attempt": attempt,
                            "error_type": "perception",
                            "error": str(e),
                        }
                    )

                if attempt < max_retries - 1:
                    recovery_result = self.fallback_strategies["perception_error"](e)
                    if recovery_result:
                        return recovery_result
                    # Exponential backoff before retry
                    time.sleep(0.1 * (2**attempt))

            except ReasoningError as e:
                logger.warning(
                    f"Reasoning error on attempt {attempt + 1}/{max_retries}: {e}"
                )

                with self._error_lock:
                    self.error_history.append(
                        {
                            "timestamp": time.time(),
                            "attempt": attempt,
                            "error_type": "reasoning",
                            "error": str(e),
                        }
                    )

                if attempt < max_retries - 1:
                    recovery_result = self.fallback_strategies["reasoning_error"](e)
                    if recovery_result:
                        return recovery_result
                    time.sleep(0.1 * (2**attempt))

            except ExecutionError as e:
                logger.warning(
                    f"Execution error on attempt {attempt + 1}/{max_retries}: {e}"
                )

                with self._error_lock:
                    self.error_history.append(
                        {
                            "timestamp": time.time(),
                            "attempt": attempt,
                            "error_type": "execution",
                            "error": str(e),
                        }
                    )

                if attempt < max_retries - 1:
                    recovery_result = self.fallback_strategies["execution_error"](e)
                    if recovery_result:
                        return recovery_result
                    time.sleep(0.1 * (2**attempt))

            except Exception as e:
                logger.error(
                    f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}"
                )

                with self._error_lock:
                    self.error_history.append(
                        {
                            "timestamp": time.time(),
                            "attempt": attempt,
                            "error_type": "unexpected",
                            "error": str(e),
                        }
                    )

                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2**attempt))
                    continue

        logger.error("Max retries exceeded, returning fallback result")
        return self._create_fallback_result("Max retries exceeded")

    def _perception_fallback(self, error: Exception) -> Optional[Dict[str, Any]]:
        """
        Fallback for perception errors

        Args:
            error: The exception that occurred

        Returns:
            Fallback result or None
        """
        logger.info("Attempting perception fallback")

        try:
            # Try to use recent memory as fallback
            if self.deps.ltm and hasattr(self.deps.ltm, "search"):
                recent = self.deps.ltm.search(np.zeros(384), k=1)
                if recent and len(recent) > 0:
                    # Extract embedding from search result
                    if len(recent[0]) > 2:
                        embedding = recent[0][2].get("embedding", np.zeros(384))
                    else:
                        embedding = np.zeros(384)

                    perception_result = {
                        "modality": ModalityType.UNKNOWN,
                        "embedding": embedding,
                        "uncertainty": 1.0,
                        "fallback": True,
                    }

                    plan = self._reason_and_plan(perception_result, {})
                    validated_plan = self._validate_and_ensure_safety(plan, {})
                    return self._execute_action(validated_plan)
        except Exception as e:
            logger.error(f"Perception fallback failed: {e}")

        return None

    def _reasoning_fallback(self, error: Exception) -> Optional[Dict[str, Any]]:
        """
        Fallback for reasoning errors

        Args:
            error: The exception that occurred

        Returns:
            Fallback result
        """
        logger.info("Attempting reasoning fallback")

        try:
            wait_plan = self._create_wait_plan(f"Reasoning error: {error}")
            validated_plan = self._validate_and_ensure_safety(wait_plan, {})
            return self._execute_action(validated_plan)
        except Exception as e:
            logger.error(f"Reasoning fallback failed: {e}")
            return None

    def _execution_fallback(self, error: Exception) -> Dict[str, Any]:
        """
        Fallback for execution errors

        Args:
            error: The exception that occurred

        Returns:
            Fallback result
        """
        logger.info("Creating execution fallback result")
        return self._create_fallback_result(f"Execution failed: {error}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics

        Returns:
            Dictionary with error statistics
        """
        with self._error_lock:
            errors = list(self.error_history)

            if not errors:
                return {
                    "total_attempts": 0,
                    "total_errors": 0,
                    "success_rate": 0.0,
                    "error_types": {},
                }

            total_attempts = len(errors)
            total_errors = sum(1 for e in errors if not e.get("success", False))
            success_rate = (total_attempts - total_errors) / total_attempts

            error_types = {}
            for error in errors:
                if not error.get("success", False):
                    error_type = error.get("error_type", "unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1

            return {
                "total_attempts": total_attempts,
                "total_errors": total_errors,
                "success_rate": success_rate,
                "error_types": error_types,
            }


# ============================================================
# ADAPTIVE ORCHESTRATOR
# ============================================================


class PerformanceMonitor:
    """
    Monitors performance metrics for adaptive strategy selection
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize Performance Monitor

        Args:
            window_size: Size of the metrics history window
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self._lock = threading.RLock()

    def record(self, metrics: Dict[str, float]):
        """
        Record metrics

        Args:
            metrics: Dictionary of metric values
        """
        with self._lock:
            self.metrics_history.append({**metrics, "timestamp": time.time()})

    def get_recent_metrics(self, n: int = 20) -> Dict[str, float]:
        """
        Get recent metrics summary

        Args:
            n: Number of recent entries to consider

        Returns:
            Dictionary with aggregated metrics
        """
        with self._lock:
            if not self.metrics_history:
                return {
                    "avg_latency": 0.0,
                    "error_rate": 0.0,
                    "avg_reward": 0.0,
                    "uncertainty": 0.5,
                }

            recent = list(self.metrics_history)[-n:]

            return {
                "avg_latency": np.mean([m.get("latency", 0) for m in recent]),
                "error_rate": sum(1 for m in recent if m.get("error", False))
                / len(recent),
                "avg_reward": np.mean([m.get("reward", 0) for m in recent]),
                "uncertainty": np.mean([m.get("uncertainty", 0.5) for m in recent]),
            }


class StrategySelector:
    """
    Selects execution strategy based on performance metrics
    """

    def select_strategy(self, metrics: Dict[str, float]) -> str:
        """
        Select strategy based on performance metrics

        Args:
            metrics: Current performance metrics

        Returns:
            Strategy name ('fast', 'careful', 'exploratory', 'balanced')
        """
        if not metrics:
            return "balanced"

        # High error rate -> careful
        if metrics.get("error_rate", 0) > 0.1:
            return "careful"

        # High latency -> fast
        if metrics.get("avg_latency", 0) > 1000:
            return "fast"

        # Low reward -> exploratory
        if metrics.get("avg_reward", 0) < 0.3:
            return "exploratory"

        # Default to balanced
        return "balanced"


class AdaptiveOrchestrator(VULCANAGICollective):
    """
    Adaptive orchestrator that adjusts strategy based on performance

    Features:
    - Performance monitoring
    - Dynamic strategy selection
    - Adaptation history tracking
    - Multiple execution modes (fast, careful, exploratory, balanced)
    """

    def __init__(self, config: Any, sys: Any, deps: EnhancedCollectiveDeps, redis_client: Optional[Any] = None):
        """
        Initialize Adaptive Orchestrator

        Args:
            config: Configuration object
            sys: System state object
            deps: Dependencies container
            redis_client: Optional Redis client for state persistence across workers/restarts
        """
        super().__init__(config, sys, deps, redis_client=redis_client)

        self.performance_monitor = PerformanceMonitor(
            window_size=getattr(config, "performance_window_size", 100)
        )
        self.strategy_selector = StrategySelector()
        self.adaptation_history = deque(maxlen=100)

        logger.info("AdaptiveOrchestrator initialized")

    def adaptive_step(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute with adaptive strategy selection

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result dictionary
        """
        # Get recent performance metrics
        metrics = self.performance_monitor.get_recent_metrics()

        # Select strategy based on metrics
        strategy = self.strategy_selector.select_strategy(metrics)

        # Record adaptation decision
        self.adaptation_history.append(
            {"strategy": strategy, "metrics": metrics, "timestamp": time.time()}
        )

        logger.info(
            f"Selected strategy: {strategy} (latency={metrics.get('avg_latency', 0):.1f}ms, "
            f"error_rate={metrics.get('error_rate', 0):.3f}, "
            f"reward={metrics.get('avg_reward', 0):.3f})"
        )

        # Execute with selected strategy
        try:
            if strategy == "fast":
                result = self._fast_step(history, context)
            elif strategy == "careful":
                result = self._careful_step(history, context)
            elif strategy == "exploratory":
                result = self._exploratory_step(history, context)
            else:
                result = self.step(history, context)

            # Record performance
            if self.sys.provenance_chain:
                latency = (time.time() - self.sys.provenance_chain[-1].t) * 1000
            else:
                latency = 0

            self.performance_monitor.record(
                {
                    "latency": latency,
                    "error": not result.get("success", True),
                    "reward": result.get("reward", 0),
                    "uncertainty": result.get("uncertainty", 0.5),
                    "strategy": strategy,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error in adaptive step: {e}", exc_info=True)

            # Record error
            self.performance_monitor.record(
                {
                    "latency": 0,
                    "error": True,
                    "reward": -1.0,
                    "uncertainty": 1.0,
                    "strategy": strategy,
                }
            )

            return self._create_fallback_result(str(e))

    def _fast_step(self, history: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast execution with minimal processing

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result
        """
        # Modify context for fast execution
        fast_context = {
            **context,
            "time_budget_ms": 100,
            "quality": "fast",
            "skip_expensive_operations": True,
        }

        return self.step(history, fast_context)

    def _careful_step(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Careful execution with thorough validation

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result
        """
        # Modify context for careful execution
        careful_context = {
            **context,
            "time_budget_ms": 2000,
            "quality": "high",
            "safety_level": "strict",
            "thorough_validation": True,
        }

        return self.step(history, careful_context)

    def _exploratory_step(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Exploratory execution to gather information

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result
        """
        # Modify context for exploratory execution
        exploratory_context = {
            **context,
            "high_level_goal": "explore",
            "exploration_bonus": 0.5,
            "risk_tolerance": "high",
        }

        return self.step(history, exploratory_context)

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """
        Get adaptation statistics

        Returns:
            Dictionary with adaptation statistics
        """
        adaptations = list(self.adaptation_history)

        if not adaptations:
            return {
                "total_adaptations": 0,
                "strategy_distribution": {},
                "current_strategy": "balanced",
            }

        strategy_counts = {}
        for adaptation in adaptations:
            strategy = adaptation["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "total_adaptations": len(adaptations),
            "strategy_distribution": strategy_counts,
            "current_strategy": (
                adaptations[-1]["strategy"] if adaptations else "balanced"
            ),
            "recent_metrics": self.performance_monitor.get_recent_metrics(),
        }


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "ParallelOrchestrator",
    "FaultTolerantOrchestrator",
    "AdaptiveOrchestrator",
    "PerformanceMonitor",
    "StrategySelector",
    "PerceptionError",
    "ReasoningError",
    "ExecutionError",
    "TimeoutStrategy",
    "JobStatus",
    "shutdown_executor_with_timeout",
    "SUPPORTS_EXECUTOR_TIMEOUT",
    "PYTHON_VERSION",
    "DEFAULT_TIME_BUDGET_MS",
]
