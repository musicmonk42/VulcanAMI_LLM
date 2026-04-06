"""
state_resources.py - Resource usage monitoring for World Model.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import time

logger = logging.getLogger(__name__)


def get_cpu_usage(wm) -> float:
    """Get current CPU usage"""
    try:
        import psutil

        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return 50.0


def get_memory_usage(wm) -> float:
    """Get current memory usage in MB"""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 1024.0


def get_low_activity_duration(wm) -> float:
    """Get duration of low activity in minutes.

    Calculates how long since the last observation was processed.
    This is used for self-improvement triggers that activate during idle periods.

    Returns:
        Duration in minutes since last observation, or 0.0 if no observations yet
    """
    if wm.last_observation_time is None:
        return 0.0

    current_time = time.time()
    elapsed_seconds = current_time - wm.last_observation_time
    return elapsed_seconds / 60.0  # Convert to minutes
