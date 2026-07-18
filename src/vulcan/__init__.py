"""Lightweight package root for VULCAN.

Canonical runtime and security tests must be importable without optional neural,
research, web, or graph dependencies.  Heavy historical subsystems remain
available through their explicit submodule paths instead of being imported as a
side effect of ``import vulcan``.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Vulcan AI Team"

MEMORY_AVAILABLE = False
REASONING_AVAILABLE = False
LEARNING_AVAILABLE = False
ROUTING_AVAILABLE = False
CURIOSITY_AVAILABLE = False
DECOMPOSER_AVAILABLE = False
CRYSTALLIZER_AVAILABLE = False
SAFETY_AVAILABLE = False
CONFIG_AVAILABLE = False
ORCHESTRATOR_AVAILABLE = False


def get_vulcan_status() -> dict:
    """Get availability status without importing optional subsystems."""
    return {
        "version": __version__,
        "memory": MEMORY_AVAILABLE,
        "reasoning": REASONING_AVAILABLE,
        "learning": LEARNING_AVAILABLE,
        "routing": ROUTING_AVAILABLE,
        "curiosity": CURIOSITY_AVAILABLE,
        "decomposer": DECOMPOSER_AVAILABLE,
        "crystallizer": CRYSTALLIZER_AVAILABLE,
        "safety": SAFETY_AVAILABLE,
        "config": CONFIG_AVAILABLE,
        "orchestrator": ORCHESTRATOR_AVAILABLE,
    }


def print_vulcan_status() -> None:
    status = get_vulcan_status()
    print(status)


__all__ = [
    "__version__", "get_vulcan_status", "print_vulcan_status",
    "MEMORY_AVAILABLE", "REASONING_AVAILABLE", "LEARNING_AVAILABLE",
    "ROUTING_AVAILABLE", "CURIOSITY_AVAILABLE", "DECOMPOSER_AVAILABLE",
    "CRYSTALLIZER_AVAILABLE", "SAFETY_AVAILABLE", "CONFIG_AVAILABLE",
    "ORCHESTRATOR_AVAILABLE",
]
