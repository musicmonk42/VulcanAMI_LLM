# ============================================================
# VULCAN-AGI Orchestrator - Agent Scoring Module
# Extracted from agent_pool.py for modularity
# Agent scoring, assignment, and capability matching
# ============================================================

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_lifecycle import AgentCapability, AgentState
from .agent_pool_types import TOURNAMENT_MAX_CANDIDATES

logger = logging.getLogger(__name__)


def assign_agent(manager: "AgentPoolManager", capability: AgentCapability) -> Optional[str]:
    """
    Assign an available agent with required capability.

    Must be called with manager.lock held.

    Args:
        manager: AgentPoolManager instance
        capability: Required capability

    Returns:
        Agent ID if available, None otherwise

    AGENT POOL FIX: Enhanced logging to help diagnose routing failures.
    """
    available_agents = [
        agent_id
        for agent_id, metadata in manager.agents.items()
        if metadata.state.can_accept_work()
        and metadata.capability.can_handle_capability(capability)
    ]

    if not available_agents:
        # AGENT POOL FIX: Enhanced logging for debugging capability mismatches
        all_caps = [m.capability.value for m in manager.agents.values()]
        idle_agents = [
            (aid, m.capability.value)
            for aid, m in manager.agents.items()
            if m.state.can_accept_work()
        ]

        # Check if any agent exists with the requested capability
        has_capability = any(
            m.capability == capability or m.capability.can_handle_capability(capability)
            for m in manager.agents.values()
        )

        if not has_capability:
            logger.warning(
                f"[AgentPool] CAPABILITY MISMATCH: No agent has capability "
                f"'{capability.value}'. Pool capabilities: {set(all_caps)}. "
                f"Consider updating agent pool configuration to include this capability."
            )
        elif idle_agents:
            logger.debug(
                f"[AgentPool] No available agent for capability '{capability.value}'. "
                f"Idle agents: {idle_agents}"
            )
        else:
            logger.debug(
                f"[AgentPool] All agents busy. No agent available for capability "
                f"'{capability.value}'."
            )
        return None

    # RESOURCE-AWARE JOB DISTRIBUTION
    # Select agent using weighted scoring based on health, load, and success rate
    agent_scores = []
    for agent_id in available_agents:
        score = calculate_agent_score(manager, agent_id)
        agent_scores.append((agent_id, score))

    # Pick best agent (highest score)
    best_agent = max(agent_scores, key=lambda x: x[1])[0]

    return best_agent


def calculate_agent_score(manager: "AgentPoolManager", agent_id: str) -> float:
    """
    Calculate a composite score for agent selection based on multiple factors.

    RESOURCE-AWARE JOB DISTRIBUTION: This enables smarter job assignment
    by considering agent health, load, and historical performance.

    Factors considered:
    - Health score (40%): Agent's overall health (0.0-1.0)
    - Current load (30%): Inverse of current workload
    - Success rate (20%): Historical success rate
    - Capability match (10%): How well capability matches job

    Args:
        manager: AgentPoolManager instance
        agent_id: The agent to score

    Returns:
        Composite score between 0.0 and 1.0
    """
    if agent_id not in manager.agents:
        return 0.0

    metadata = manager.agents[agent_id]

    try:
        score = 0.0

        # Factor 1: Health score (40% weight)
        health_score = metadata.get_health_score()
        score += health_score * 0.4

        # Factor 2: Current load - inverse (30% weight)
        is_working = 1.0 if metadata.state == AgentState.WORKING else 0.0
        load_factor = 1.0 - is_working
        score += load_factor * 0.3

        # Factor 3: Success rate (20% weight)
        total_tasks = metadata.tasks_completed + metadata.tasks_failed
        if total_tasks > 0:
            success_rate = metadata.tasks_completed / total_tasks
        else:
            success_rate = 0.5  # Default for new agents
        score += success_rate * 0.2

        # Factor 4: Capability match bonus (10% weight)
        capability_bonus = 1.0 if metadata.capability != AgentCapability.GENERAL else 0.8
        score += capability_bonus * 0.1

        return min(1.0, max(0.0, score))

    except Exception as e:
        logger.warning(f"Error calculating agent score for {agent_id}: {e}")
        return 0.5  # Default mid-range score


def get_agents_by_capability(
    manager: "AgentPoolManager",
    capabilities: List[str],
    max_agents: int = TOURNAMENT_MAX_CANDIDATES
) -> List[str]:
    """
    Get available agents that can handle the specified capabilities.

    Args:
        manager: AgentPoolManager instance
        capabilities: List of capability names to filter by
        max_agents: Maximum number of agents to return

    Returns:
        List of agent IDs that can handle the capabilities
    """
    with manager.lock:
        available_agents = []
        for agent_id, metadata in manager.agents.items():
            if metadata.state.can_accept_work():
                if metadata.capability.value in capabilities:
                    available_agents.append(agent_id)

        # Sort by performance (lowest failure rate first)
        available_agents.sort(
            key=lambda aid: manager.agents[aid].tasks_failed
            / max(1, manager.agents[aid].tasks_completed)
        )

        return available_agents[:max_agents]


def get_capability_distribution(manager: "AgentPoolManager") -> Dict[str, int]:
    """
    Get the current capability distribution in the agent pool.

    AGENT POOL CONFIGURATION FIX: This method provides observability into
    which capabilities are available in the pool.

    Args:
        manager: AgentPoolManager instance

    Returns:
        Dictionary mapping capability names to agent counts.
    """
    with manager.lock:
        capability_counts: Dict[str, int] = {}
        for metadata in manager.agents.values():
            cap_name = metadata.capability.value
            capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
        return capability_counts
