"""Memory monitoring and resource management utilities."""

from vulcan.monitoring.memory_guard import (
    MemoryGuard,
    get_memory_guard,
    set_aggressive_gc_callback,
    start_memory_guard,
    stop_memory_guard,
    trigger_gc,
)

__all__ = [
    "MemoryGuard",
    "start_memory_guard",
    "stop_memory_guard",
    "get_memory_guard",
    "trigger_gc",
    "set_aggressive_gc_callback",
]
