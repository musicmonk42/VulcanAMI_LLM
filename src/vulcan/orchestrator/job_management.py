# ============================================================
# VULCAN-AGI Orchestrator - Job Management Module
# Extracted from agent_pool.py for modularity
# Job submission, execution, dead letter queue, stuck job handling
# ============================================================

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_lifecycle import AgentCapability, AgentState, create_job_provenance
from .agent_pool_types import AGENT_SELECTION_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def submit_job(
    manager: "AgentPoolManager",
    graph: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    capability_required: AgentCapability = AgentCapability.GENERAL,
    timeout_seconds: Optional[float] = None,
) -> str:
    """
    Submit a job to the agent pool.

    THREAD POOL FIX: This method is now NON-BLOCKING.

    Args:
        manager: AgentPoolManager instance
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
    timeout_seconds = timeout_seconds if timeout_seconds is not None else AGENT_SELECTION_TIMEOUT_SECONDS

    with manager.lock:
        if manager._shutdown_event.is_set():
            raise RuntimeError("Agent pool is shutting down")

        if len(manager.task_assignments) >= 1000:
            logger.error(f"Job queue full, rejecting job {job_id}")
            raise RuntimeError("Job queue at maximum capacity (1000 tasks)")

        # Create provenance record
        provenance = create_job_provenance(
            job_id=job_id,
            graph_id=graph.get("id", "unknown"),
            parameters=parameters,
            priority=priority,
            timeout_seconds=timeout_seconds,
        )

        manager._set_provenance_by_job_id(job_id, provenance)

        with manager.stats_lock:
            manager.stats["total_jobs_submitted"] += 1

        manager._persist_state_to_redis()

        if time.time() - manager._last_archive_time > 3600:
            manager._archive_old_provenance()

        agent_id = manager._assign_agent_with_timeout(
            capability_required, timeout_seconds
        )

        if agent_id:
            provenance.agent_id = agent_id
            provenance.hardware_used = manager.agents[agent_id].hardware_spec
            manager.task_assignments[job_id] = agent_id
            manager.task_assignment_times[job_id] = time.time()
            manager.agents[agent_id].transition_state(
                AgentState.WORKING, f"Assigned job {job_id}"
            )
            metadata = manager.agents[agent_id]
            logger.info(f"Job {job_id} assigned to agent {agent_id}")
        else:
            logger.warning(
                f"No agent available for job {job_id} within {timeout_seconds}s"
            )
            provenance.agent_id = "no_agent_available"
            provenance.complete("failed", error="No agent available within timeout")

            with manager.stats_lock:
                manager.stats["total_jobs_failed"] += 1

            manager._persist_state_to_redis()
            return job_id

    # THREAD POOL FIX: Queue for background execution instead of blocking
    if agent_id:
        with manager._pending_executions_lock:
            manager._pending_executions[job_id] = {
                "agent_id": agent_id,
                "graph": graph,
                "parameters": parameters,
                "metadata": metadata,
                "queued_at": time.time(),
            }
        logger.debug(f"Job {job_id} queued for background execution")

    return job_id


def move_to_dead_letter_queue(
    manager: "AgentPoolManager",
    task_id: str,
    reason: str,
    error: Optional[Exception] = None
) -> None:
    """Move a job to the dead letter queue after repeated failures."""
    with manager._dead_letter_lock:
        dlq_entry = {
            "task_id": task_id,
            "reason": reason,
            "error": str(error) if error else None,
            "retry_count": manager._job_retry_counts.get(task_id, 0),
            "timestamp": time.time(),
            "provenance": None,
        }

        try:
            prov = manager._get_provenance_by_job_id(task_id)
            if prov:
                dlq_entry["provenance"] = prov.to_dict() if hasattr(prov, 'to_dict') else str(prov)
        except Exception:
            pass

        manager._dead_letter_queue.append(dlq_entry)
        manager._job_retry_counts.pop(task_id, None)

        logger.warning(
            f"[DLQ] Job {task_id} moved to dead letter queue: {reason} "
            f"(retries={dlq_entry['retry_count']})"
        )


def get_dead_letter_queue(manager: "AgentPoolManager") -> List[Dict[str, Any]]:
    """Get all jobs in the dead letter queue."""
    with manager._dead_letter_lock:
        return list(manager._dead_letter_queue)


def clear_dead_letter_queue(manager: "AgentPoolManager") -> int:
    """Clear the dead letter queue."""
    with manager._dead_letter_lock:
        count = len(manager._dead_letter_queue)
        manager._dead_letter_queue.clear()
        logger.info(f"[DLQ] Cleared {count} entries from dead letter queue")
        return count


def retry_dead_letter_job(manager: "AgentPoolManager", task_id: str) -> bool:
    """Retry a job from the dead letter queue."""
    with manager._dead_letter_lock:
        for i, entry in enumerate(manager._dead_letter_queue):
            if entry["task_id"] == task_id:
                del manager._dead_letter_queue[i]
                logger.info(f"[DLQ] Job {task_id} removed from DLQ for retry")
                manager._job_retry_counts[task_id] = 0
                return True
    return False


def get_stuck_jobs(manager: "AgentPoolManager") -> List[Dict[str, Any]]:
    """Get list of jobs that appear to be stuck."""
    from .agent_pool_types import STUCK_JOB_WARNING_THRESHOLD, STUCK_JOB_CRITICAL_THRESHOLD

    current_time = time.time()
    stuck_jobs = []

    with manager.lock:
        for task_id, assign_time in manager.task_assignment_times.items():
            elapsed = current_time - assign_time
            warning_threshold = manager._stuck_job_threshold_seconds * STUCK_JOB_WARNING_THRESHOLD
            critical_threshold = manager._stuck_job_threshold_seconds * STUCK_JOB_CRITICAL_THRESHOLD

            provenance = manager._get_provenance_by_job_id(task_id)
            is_heartbeat_stale = False
            time_since_heartbeat = None

            if provenance and hasattr(provenance, 'is_stale'):
                is_heartbeat_stale = provenance.is_stale()
                if hasattr(provenance, 'get_time_since_heartbeat'):
                    time_since_heartbeat = provenance.get_time_since_heartbeat()

            if is_heartbeat_stale or elapsed > warning_threshold:
                agent_id = manager.task_assignments.get(task_id)
                stuck_jobs.append({
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": manager._stuck_job_threshold_seconds,
                    "is_critical": elapsed > critical_threshold or is_heartbeat_stale,
                    "heartbeat_stale": is_heartbeat_stale,
                    "time_since_heartbeat": time_since_heartbeat,
                    "detection_method": "heartbeat" if is_heartbeat_stale else "timeout",
                })

    return stuck_jobs


def process_stuck_jobs(manager: "AgentPoolManager") -> Dict[str, Any]:
    """Process jobs that are stuck in processing state."""
    stuck_jobs = get_stuck_jobs(manager)
    results = {
        "total_stuck": len(stuck_jobs),
        "warned": 0,
        "recovered": 0,
        "moved_to_dlq": 0,
    }

    for job in stuck_jobs:
        task_id = job["task_id"]

        if job["is_critical"]:
            retry_count = manager._job_retry_counts.get(task_id, 0)

            if retry_count >= manager._max_job_retries:
                manager._cancel_task(task_id)
                move_to_dead_letter_queue(manager, task_id, "stuck_max_retries")
                results["moved_to_dlq"] += 1
            else:
                manager._job_retry_counts[task_id] = retry_count + 1
                logger.warning(
                    f"[StuckJobs] Task {task_id} is stuck "
                    f"(elapsed={job['elapsed_seconds']:.0f}s), "
                    f"retry {retry_count + 1}/{manager._max_job_retries}"
                )
                manager._cancel_task(task_id)
                results["recovered"] += 1
        else:
            logger.debug(
                f"[StuckJobs] Task {task_id} is slow "
                f"(elapsed={job['elapsed_seconds']:.0f}s, "
                f"threshold={job['timeout_seconds']:.0f}s)"
            )
            results["warned"] += 1

    if results["total_stuck"] > 0:
        logger.info(
            f"[StuckJobs] Processed {results['total_stuck']} stuck jobs: "
            f"warned={results['warned']}, recovered={results['recovered']}, "
            f"moved_to_dlq={results['moved_to_dlq']}"
        )

    return results


def reassign_job(manager: "AgentPoolManager", task_id: str, force: bool = False) -> Optional[str]:
    """Reassign a stuck or failed job to a different agent."""
    with manager.lock:
        if task_id not in manager.task_assignments:
            logger.warning(f"[Reassign] Task {task_id} not found in assignments")
            return None

        old_agent_id = manager.task_assignments.get(task_id)
        old_agent = manager.agents.get(old_agent_id) if old_agent_id else None

        provenance = manager._get_provenance_by_job_id(task_id)
        if provenance is None:
            logger.warning(f"[Reassign] Task {task_id} has no provenance record")
            return None

        capability = AgentCapability.GENERAL
        if hasattr(provenance, 'capability_required') and provenance.capability_required:
            capability = provenance.capability_required
        elif old_agent:
            capability = old_agent.capability

        if old_agent_id and old_agent_id in manager.agents:
            if old_agent.state == AgentState.WORKING or force:
                old_agent.transition_state(
                    AgentState.IDLE,
                    f"Task {task_id} reassigned"
                )
                logger.info(
                    f"[Reassign] Released agent {old_agent_id} from task {task_id}"
                )

        del manager.task_assignments[task_id]
        if task_id in manager.task_assignment_times:
            del manager.task_assignment_times[task_id]

    try:
        new_agent_id = manager._assign_agent_with_timeout(
            capability=capability,
            timeout_seconds=AGENT_SELECTION_TIMEOUT_SECONDS
        )

        if new_agent_id:
            with manager.lock:
                manager.task_assignments[task_id] = new_agent_id
                manager.task_assignment_times[task_id] = time.time()

                new_agent = manager.agents.get(new_agent_id)
                if new_agent:
                    new_agent.transition_state(
                        AgentState.WORKING,
                        f"Reassigned task {task_id}"
                    )

            logger.info(
                f"[Reassign] Task {task_id} reassigned from {old_agent_id} to {new_agent_id}"
            )
            return new_agent_id
        else:
            logger.warning(f"[Reassign] No available agent for task {task_id}")
            return None

    except Exception as e:
        logger.error(f"[Reassign] Failed to reassign task {task_id}: {e}")
        return None


def recover_stuck_job(manager: "AgentPoolManager", task_id: str) -> bool:
    """Attempt to recover a stuck job."""
    retry_count = manager._job_retry_counts.get(task_id, 0)

    if retry_count >= manager._max_job_retries:
        logger.warning(
            f"[RecoverStuck] Job {task_id} exceeded max retries ({retry_count}), "
            f"moving to dead letter queue"
        )
        manager._cancel_task(task_id)
        move_to_dead_letter_queue(manager, task_id, "max_recovery_attempts")
        return False

    manager._job_retry_counts[task_id] = retry_count + 1

    new_agent_id = reassign_job(manager, task_id, force=True)

    if new_agent_id:
        logger.info(
            f"[RecoverStuck] Job {task_id} recovered (retry {retry_count + 1}), "
            f"reassigned to {new_agent_id}"
        )
        return True
    else:
        logger.warning(
            f"[RecoverStuck] Failed to recover job {task_id}, "
            f"no available agents"
        )
        return False
