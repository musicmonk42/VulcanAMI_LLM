"""Canonical governed memory exports.

Historical memory implementations remain importable by their explicit module
paths for research and migration tooling, but are intentionally not initialized
when the package is imported by the serving runtime.
"""

from .governed import (
    DisabledMemoryService,
    DefaultMemoryPolicy,
    DeletionReceipt,
    DeletionState,
    GovernedMemoryService,
    MemoryActor,
    MemoryCommitResult,
    MemoryKind,
    MemoryReadRequest,
    MemoryPolicyPort,
    MemoryReason,
    MemoryWriteProposal,
    SQLiteMemoryRepository,
    compose_governed_memory,
)

__all__ = [
    "DisabledMemoryService", "DefaultMemoryPolicy", "DeletionReceipt", "DeletionState", "GovernedMemoryService", "MemoryActor",
    "MemoryCommitResult", "MemoryKind", "MemoryPolicyPort", "MemoryReadRequest", "MemoryReason",
    "MemoryWriteProposal", "SQLiteMemoryRepository", "compose_governed_memory",
]

# Compatibility is deliberately lazy: importing the canonical authority must
# not construct or import any legacy store, while explicit research callers
# keep their historical symbols when optional dependencies are installed.
_LEGACY_MODULES = {
    "Memory": ".base", "MemoryConfig": ".base", "MemoryException": ".base",
    "MemoryQuery": ".base", "MemoryStats": ".base", "MemoryType": ".base",
    "MemoryUsageMonitor": ".base", "HierarchicalMemory": ".hierarchical",
    "MemoryLevel": ".hierarchical", "MemoryPersistence": ".persistence",
    "MemoryIndex": ".retrieval", "EpisodicMemory": ".specialized",
    "SemanticMemory": ".specialized", "ProceduralMemory": ".specialized",
    "WorkingMemory": ".specialized", "LearningStatePersistence": ".learning_persistence",
}


def __getattr__(name: str):
    module_name = _LEGACY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(module_name, __name__), name)
