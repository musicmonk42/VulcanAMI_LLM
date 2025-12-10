"""
VULCAN-AGI Memory Module
Hierarchical, distributed memory with multiple specialized types
"""

from .base import (
    MemoryType,
    Memory,
    MemoryQuery,
    MemoryConfig,
    MemoryStats,
    MemoryException,
)

from .hierarchical import HierarchicalMemory, MemoryLevel

from .distributed import (
    DistributedMemory,
    MemoryFederation,
    MemoryNode,
    ConsistencyLevel,
)

from .persistence import MemoryPersistence, MemoryVersionControl, CompressionType

from .retrieval import MemoryIndex, MemorySearch, AttentionMechanism, RetrievalResult

from .consolidation import MemoryConsolidator, ConsolidationStrategy, MemoryOptimizer

from .specialized import (
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    WorkingMemory,
    Episode,
    Concept,
    Skill,
    WorkingMemoryBuffer,
)

__all__ = [
    # Base
    "MemoryType",
    "Memory",
    "MemoryQuery",
    "MemoryConfig",
    "MemoryStats",
    "MemoryException",
    # Hierarchical
    "HierarchicalMemory",
    "MemoryLevel",
    # Distributed
    "DistributedMemory",
    "MemoryFederation",
    "MemoryNode",
    "ConsistencyLevel",
    # Persistence
    "MemoryPersistence",
    "MemoryVersionControl",
    "CompressionType",
    # Retrieval
    "MemoryIndex",
    "MemorySearch",
    "AttentionMechanism",
    "RetrievalResult",
    # Consolidation
    "MemoryConsolidator",
    "ConsolidationStrategy",
    "MemoryOptimizer",
    # Specialized
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "WorkingMemory",
    "Episode",
    "Concept",
    "Skill",
    "WorkingMemoryBuffer",
]

# Version info
__version__ = "1.0.0"
