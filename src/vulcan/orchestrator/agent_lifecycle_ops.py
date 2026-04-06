# ============================================================
# VULCAN-AGI Orchestrator - Agent Lifecycle Operations Module
# Extracted from agent_pool.py for modularity
# spawn_agent, retire_agent, recover_agent, cleanup_terminated_agents
# ============================================================

import gc
import logging
import multiprocessing
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_lifecycle import AgentCapability, AgentMetadata, AgentState, create_agent_metadata
from .agent_pool_proxy import _standalone_agent_worker

logger = logging.getLogger(__name__)


def cleanup_terminated_agents(manager: "AgentPoolManager") -> int:
    """
    Remove terminated agents from the pool and respawn to minimum.

    Returns:
        Number of agents cleaned up
    """
    with manager.lock:
        before_count = len(manager.agents)

        # Find terminated agents
        terminated_ids = [
            agent_id for agent_id, metadata in manager.agents.items()
            if metadata.state == AgentState.TERMINATED
        ]

        # Remove terminated agents
        for agent_id in terminated_ids:
            # Clean up process reference if exists
            if agent_id in manager.agent_processes:
                process = manager.agent_processes[agent_id]
                if process.is_alive():
                    try:
                        process.terminate()
                        process.join(timeout=1)
                    except Exception as e:
                        logger.debug(f"Error terminating process for {agent_id}: {e}")
                try:
                    process.close()
                except Exception as e:
                    logger.debug(f"Error closing process for {agent_id}: {e}")
                del manager.agent_processes[agent_id]

            # Clean up specialized agents tracking
            for spec_list in manager.specialized_agents.values():
                if agent_id in spec_list:
                    spec_list.remove(agent_id)

            # Remove from agents dictionary
            del manager.agents[agent_id]

        removed = before_count - len(manager.agents)

        if removed > 0:
            logger.info(f"Cleaned up {removed} terminated agents")

    # Ensure minimum agents outside the lock to avoid deadlock
    manager._ensure_minimum_agents()

    return removed


def spawn_agent(
    manager: "AgentPoolManager",
    capability: AgentCapability = AgentCapability.GENERAL,
    location: str = "local",
    hardware_spec: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Spawn a new agent.

    Args:
        manager: AgentPoolManager instance
        capability: Agent capability
        location: Agent location ('local', 'remote', 'cloud')
        hardware_spec: Hardware specification dictionary

    Returns:
        Agent ID if successful, None otherwise
    """
    with manager.lock:
        # Check capacity using LIVE agent count
        live_count = manager._get_live_agent_count_unsafe()
        if live_count >= manager.max_agents:
            logger.warning(f"Agent pool at maximum capacity ({manager.max_agents} live agents)")
            return None

        # Generate unique agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"

        try:
            # Create agent metadata using factory function
            metadata = create_agent_metadata(
                agent_id=agent_id,
                capability=capability,
                location=location,
                hardware_spec=hardware_spec or manager._get_default_hardware_spec(),
            )

            # Register agent
            manager.agents[agent_id] = metadata

            # Spawn agent process/thread based on location
            if location == "local":
                _spawn_local_agent(manager, agent_id, metadata)
            elif location == "remote":
                _spawn_remote_agent(manager, agent_id, metadata)
            elif location == "cloud":
                _spawn_cloud_agent(manager, agent_id, metadata)
            else:
                logger.warning(
                    f"Unknown location '{location}', defaulting to local"
                )
                _spawn_local_agent(manager, agent_id, metadata)

            # Update statistics
            with manager.stats_lock:
                manager.stats["total_agents_spawned"] += 1

            # Persist state to Redis
            manager._persist_state_to_redis()

            logger.info(
                f"Spawned agent {agent_id} with capability {capability.value}"
            )
            return agent_id

        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}", exc_info=True)
            # Cleanup on failure
            if agent_id in manager.agents:
                del manager.agents[agent_id]
            return None


def _spawn_local_agent(manager: "AgentPoolManager", agent_id: str, metadata: AgentMetadata):
    """Spawn local agent process."""
    try:
        process = multiprocessing.Process(
            target=_standalone_agent_worker,
            args=(agent_id,),
            daemon=True,
            name=f"Agent-{agent_id}",
        )
        process.start()
        manager.agent_processes[agent_id] = process

        metadata.transition_state(AgentState.IDLE, "Local agent process started")
        logger.debug(f"Local agent {agent_id} process started (PID: {process.pid})")

    except Exception as e:
        logger.error(f"Failed to spawn local agent {agent_id}: {e}")
        metadata.transition_state(AgentState.ERROR, f"Spawn failed: {e}")
        metadata.record_error(e, {"phase": "spawn_local"})


def _spawn_remote_agent(manager: "AgentPoolManager", agent_id: str, metadata: AgentMetadata):
    """Spawn remote agent (via SSH, RPC, etc.)"""
    logger.info(f"Spawning remote agent {agent_id}")
    metadata.transition_state(AgentState.IDLE, "Remote agent spawned (stub)")


def _spawn_cloud_agent(manager: "AgentPoolManager", agent_id: str, metadata: AgentMetadata):
    """Spawn cloud agent (AWS, GCP, Azure, etc.)"""
    logger.info(f"Spawning cloud agent {agent_id}")
    metadata.transition_state(AgentState.IDLE, "Cloud agent spawned (stub)")


def retire_agent(manager: "AgentPoolManager", agent_id: str, force: bool = False) -> bool:
    """
    Retire an agent gracefully.

    Args:
        manager: AgentPoolManager instance
        agent_id: Agent identifier
        force: If True, force immediate termination

    Returns:
        True if agent was retired, False otherwise
    """
    with manager.lock:
        if agent_id not in manager.agents:
            logger.warning(f"Cannot retire agent {agent_id}: not found")
            return False

        metadata = manager.agents[agent_id]

        # Cancel any assigned tasks
        tasks_to_cancel = [
            tid for tid, aid in manager.task_assignments.items() if aid == agent_id
        ]

        for task_id in tasks_to_cancel:
            logger.warning(f"Cancelling task {task_id} due to agent retirement")
            manager._cancel_task(task_id)

        if metadata.state == AgentState.WORKING and not force:
            metadata.transition_state(
                AgentState.RETIRING, "Marked for retirement after current task"
            )
            logger.info(
                f"Agent {agent_id} marked for retirement after current task"
            )
        else:
            metadata.transition_state(
                AgentState.TERMINATED,
                "Forced retirement" if force else "Retirement",
            )

            # Cleanup process
            if agent_id in manager.agent_processes:
                process = manager.agent_processes[agent_id]

                if process.is_alive():
                    if not force:
                        process.terminate()
                        process.join(timeout=5)

                    if process.is_alive():
                        logger.warning(f"Force killing agent {agent_id} process")
                        process.kill()
                        process.join(timeout=2)

                    try:
                        process.close()
                    except Exception as e:
                        logger.debug(f"Error closing process handle: {e}")

                del manager.agent_processes[agent_id]

            # PERFORMANCE FIX: Clean up specialized_agents tracking
            for spec_list in manager.specialized_agents.values():
                if agent_id in spec_list:
                    spec_list.remove(agent_id)

            # Update statistics
            with manager.stats_lock:
                manager.stats["total_agents_retired"] += 1

            # Persist state to Redis
            manager._persist_state_to_redis()

            logger.info(f"Agent {agent_id} terminated")

    # Immediately ensure minimum agents after retirement
    manager._ensure_minimum_agents()

    # ISSUE 8 FIX: Rate-limited GC after retiring agent
    manager._maybe_gc()

    return True


def recover_agent(manager: "AgentPoolManager", agent_id: str) -> bool:
    """
    Recover a failed agent.

    Args:
        manager: AgentPoolManager instance
        agent_id: Agent identifier

    Returns:
        True if agent was recovered, False otherwise
    """
    with manager.lock:
        if agent_id not in manager.agents:
            logger.warning(f"Cannot recover agent {agent_id}: not found")
            return False

        metadata = manager.agents[agent_id]

        if not metadata.should_recover():
            logger.info(
                f"Agent {agent_id} should not be recovered (too many errors)"
            )
            return False

        if not metadata.transition_state(
            AgentState.RECOVERING, "Recovery initiated"
        ):
            logger.error(f"Cannot transition agent {agent_id} to RECOVERING state")
            return False

        logger.info(f"Recovering agent {agent_id}")

        # Clean up old process if exists
        if agent_id in manager.agent_processes:
            process = manager.agent_processes[agent_id]
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            try:
                process.close()
            except Exception as e:
                logger.debug(f"Failed to cleanup agent: {e}")
            del manager.agent_processes[agent_id]

        # Respawn agent based on location
        success = False
        try:
            if metadata.location == "local":
                _spawn_local_agent(manager, agent_id, metadata)
                success = True
            elif metadata.location == "remote":
                _spawn_remote_agent(manager, agent_id, metadata)
                success = True
            elif metadata.location == "cloud":
                _spawn_cloud_agent(manager, agent_id, metadata)
                success = True

            if success:
                metadata.consecutive_errors = 0
                metadata.transition_state(AgentState.IDLE, "Recovery successful")

                with manager.stats_lock:
                    manager.stats["total_recoveries_successful"] += 1

                logger.info(f"Agent {agent_id} recovered successfully")

        except Exception as e:
            logger.error(f"Failed to recover agent {agent_id}: {e}")
            metadata.transition_state(AgentState.ERROR, f"Recovery failed: {e}")
            success = False

        # Update statistics
        with manager.stats_lock:
            manager.stats["total_recoveries_attempted"] += 1

        # Persist state to Redis
        manager._persist_state_to_redis()

        return success
