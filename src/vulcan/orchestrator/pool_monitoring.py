# ============================================================
# VULCAN-AGI Orchestrator - Pool Monitoring Module
# Extracted from agent_pool.py for modularity
# Status reporting, statistics, and performance monitoring
# ============================================================

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_lifecycle import AgentState

logger = logging.getLogger(__name__)


def get_pool_status(manager: "AgentPoolManager") -> Dict[str, Any]:
    """Get pool status with throttled status checks.

    Also reports live_agents count to distinguish from terminated.

    Args:
        manager: AgentPoolManager instance

    Returns:
        Pool status dictionary
    """
    current_time = time.time()
    if current_time - manager.last_status_check < manager.status_check_interval:
        logger.debug(
            "Skipping queue status check due to throttling; returning cached agent data."
        )
        return get_cached_status(manager)

    manager.last_status_check = current_time

    # Trigger cleanup when reporting status
    manager.cleanup_terminated_agents()

    with manager.lock:
        # PERFORMANCE FIX: Combine into single loop to avoid double iteration
        state_counts = defaultdict(int)
        capability_counts = defaultdict(int)
        health_scores = []
        live_count = 0
        for metadata in manager.agents.values():
            state_counts[metadata.state.value] += 1
            capability_counts[metadata.capability.value] += 1
            health_scores.append(metadata.get_health_score())
            if metadata.state not in (AgentState.TERMINATED, AgentState.ERROR):
                live_count += 1

        avg_health = (
            sum(health_scores) / len(health_scores) if health_scores else 0.0
        )

        queue_status = {}
        if manager.task_queue and hasattr(manager.task_queue, "get_coordinator_status"):
            try:
                queue_status = manager.task_queue.get_coordinator_status()
            except Exception as e:
                logger.warning(f"Failed to get queue status: {e}")
        elif manager.task_queue:
            try:
                queue_status = manager.task_queue.get_queue_status()
            except Exception as e:
                logger.warning(
                    f"Failed to get queue status from get_queue_status: {e}"
                )

        with manager.stats_lock:
            stats = dict(manager.stats)

        # THREAD POOL FIX: Include pending executions count
        with manager._pending_executions_lock:
            pending_executions = len(manager._pending_executions)

        status = {
            "total_agents": len(manager.agents),
            "live_agents": live_count,
            "state_distribution": dict(state_counts),
            "capability_distribution": dict(capability_counts),
            "pending_tasks": len(manager.task_assignments),
            "pending_executions": pending_executions,
            "average_health_score": avg_health,
            "queue_status": queue_status,
            "statistics": stats,
            "provenance_records_count": len(manager.provenance_records),
            "min_agents": manager.min_agents,
            "max_agents": manager.max_agents,
        }
        logger.debug("[AutoScaler] - Agent pool status: %s", status)
        return status


def get_cached_status(manager: "AgentPoolManager") -> Dict[str, Any]:
    """Return cached status to avoid frequent checks."""
    with manager.lock:
        state_counts = defaultdict(int)
        capability_counts = defaultdict(int)
        for metadata in manager.agents.values():
            state_counts[metadata.state.value] += 1
            capability_counts[metadata.capability.value] += 1

        return {
            "total_agents": len(manager.agents),
            "state_distribution": dict(state_counts),
            "capability_distribution": dict(capability_counts),
            "queue_status": {},
        }


def get_agent_status(manager: "AgentPoolManager", agent_id: str) -> Optional[Dict[str, Any]]:
    """Get status of specific agent."""
    with manager.lock:
        if agent_id not in manager.agents:
            return None
        metadata = manager.agents[agent_id]
        return metadata.get_summary()


def get_job_provenance(manager: "AgentPoolManager", job_id: str) -> Optional[Dict[str, Any]]:
    """Get complete provenance for a job."""
    with manager.lock:
        provenance = manager._get_provenance_by_job_id(job_id)
        if provenance is None:
            return None
        return provenance.get_summary()


def get_statistics(manager: "AgentPoolManager") -> Dict[str, Any]:
    """Get pool statistics including performance metrics."""
    with manager.stats_lock:
        base_stats = dict(manager.stats)

    # Add performance metrics
    base_stats["response_times"] = manager.response_time_tracker.get_stats()
    base_stats["priority_queue"] = manager.priority_queue.get_stats()
    base_stats["perf_thresholds"] = manager.perf_thresholds

    # THREAD POOL FIX: Include pending executions count
    with manager._pending_executions_lock:
        base_stats["pending_executions"] = len(manager._pending_executions)

    # PERFORMANCE FIX: Include dead letter queue and stuck job stats
    with manager._dead_letter_lock:
        base_stats["dead_letter_queue_size"] = len(manager._dead_letter_queue)
    base_stats["stuck_jobs"] = len(manager.get_stuck_jobs())

    return base_stats


def reset_statistics(manager: "AgentPoolManager", preserve_totals: bool = True) -> None:
    """Reset pool statistics to prevent unbounded memory growth."""
    with manager.stats_lock:
        if not preserve_totals:
            manager.stats = {
                "total_jobs_submitted": 0,
                "total_jobs_completed": 0,
                "total_jobs_failed": 0,
                "total_agents_spawned": 0,
                "total_agents_retired": 0,
                "total_recoveries_attempted": 0,
                "total_recoveries_successful": 0,
            }

    # Reset response time tracker's sliding window
    if hasattr(manager, 'response_time_tracker'):
        manager.response_time_tracker.trim_to_window_size()

    # Reset priority queue statistics
    if hasattr(manager, 'priority_queue'):
        manager.priority_queue.reset_priority_distribution()

    # Trigger pool recovery after stats reset
    cleaned = manager.cleanup_terminated_agents()
    live_count = manager.get_live_agent_count()

    logger.info(
        f"Statistics reset completed (preserve_totals={preserve_totals}). "
        f"Pool recovered: {live_count} live agents, {cleaned} agents cleaned up"
    )


def get_performance_stats(manager: "AgentPoolManager") -> Dict[str, Any]:
    """Get detailed performance statistics for monitoring."""
    response_stats = manager.response_time_tracker.get_stats()
    queue_stats = manager.priority_queue.get_stats()

    with manager.lock:
        agent_utilization = {
            "total": len(manager.agents),
            "working": sum(1 for m in manager.agents.values() if m.state == AgentState.WORKING),
            "idle": sum(1 for m in manager.agents.values() if m.state == AgentState.IDLE),
            "error": sum(1 for m in manager.agents.values() if m.state == AgentState.ERROR),
        }
        pending_tasks = len(manager.task_assignments)

    # THREAD POOL FIX: Include pending executions
    with manager._pending_executions_lock:
        pending_executions = len(manager._pending_executions)

    utilization_pct = (
        agent_utilization["working"] / agent_utilization["total"] * 100
        if agent_utilization["total"] > 0 else 0.0
    )

    return {
        "response_times": response_stats,
        "queue_stats": queue_stats,
        "agent_utilization": agent_utilization,
        "utilization_percent": utilization_pct,
        "pending_tasks": pending_tasks,
        "pending_executions": pending_executions,
        "thresholds": manager.perf_thresholds,
        "trend": manager.response_time_tracker.get_recent_trend(),
    }
