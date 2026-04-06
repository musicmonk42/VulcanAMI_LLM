# ============================================================
# VULCAN-AGI Orchestrator - Pool Persistence Module
# Extracted from agent_pool.py for modularity
# Redis state hydration and persistence for AgentPoolManager
# ============================================================

import json
import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_pool_types import (
    REDIS_KEY_AGENT_POOL_STATS,
    REDIS_KEY_PROVENANCE_COUNT,
)

logger = logging.getLogger(__name__)


def hydrate_state_from_redis(manager: "AgentPoolManager") -> None:
    """
    Hydrate Agent Pool state from Redis on startup.

    This function loads persisted statistics and provenance counts from Redis
    to restore state after container restarts. If Redis is not available or
    empty, defaults to 0 values (as currently implemented).

    Args:
        manager: AgentPoolManager instance
    """
    if manager.redis_client is None:
        logger.debug("Redis client not available, skipping state hydration")
        return

    loaded_stats = {}
    try:
        # Load statistics from Redis
        stats_json = manager.redis_client.get(REDIS_KEY_AGENT_POOL_STATS)
        if stats_json:
            try:
                # Handle both string and bytes responses
                if isinstance(stats_json, bytes):
                    stats_json = stats_json.decode('utf-8')
                loaded_stats = json.loads(stats_json)

                # Update stats with loaded values, preserving default structure
                with manager.stats_lock:
                    for key, value in loaded_stats.items():
                        if key in manager.stats:
                            manager.stats[key] = value
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse stats JSON from Redis: {e}")

        # Load provenance records count from Redis
        provenance_count = manager.redis_client.get(REDIS_KEY_PROVENANCE_COUNT)
        if provenance_count:
            try:
                # Handle both string and bytes responses
                if isinstance(provenance_count, bytes):
                    provenance_count = provenance_count.decode('utf-8')
                manager._provenance_records_count = int(provenance_count)
                loaded_stats["provenance_records_count"] = manager._provenance_records_count
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse provenance count from Redis: {e}")

        if loaded_stats:
            logger.info(f"Hydrated Agent Pool state from Redis: {loaded_stats}")
        else:
            logger.debug("No persisted state found in Redis, using defaults")

    except Exception as e:
        logger.warning(f"Failed to hydrate state from Redis: {e}. Using default values.")


def persist_state_to_redis(manager: "AgentPoolManager") -> None:
    """
    Persist Agent Pool state to Redis for recovery after restarts.

    This function saves statistics and provenance counts to Redis so they
    can be restored after container restarts.

    PERFORMANCE FIX: Throttle to max once per second to avoid excessive
    Redis round-trips under high throughput.

    Args:
        manager: AgentPoolManager instance
    """
    if manager.redis_client is None:
        return

    # Throttle Redis persistence to max once per second
    now = time.time()
    if now - getattr(manager, '_last_redis_persist', 0) < 1.0:
        return
    manager._last_redis_persist = now

    try:
        # Persist statistics
        with manager.stats_lock:
            stats_json = json.dumps(manager.stats)
        manager.redis_client.set(REDIS_KEY_AGENT_POOL_STATS, stats_json)

        # Persist provenance records count
        provenance_count = len(manager.provenance_records)
        manager.redis_client.set(REDIS_KEY_PROVENANCE_COUNT, str(provenance_count))

        logger.debug(f"Persisted Agent Pool state to Redis: stats and provenance_count={provenance_count}")
    except Exception as e:
        logger.warning(f"Failed to persist state to Redis: {e}")
