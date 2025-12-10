"""
VULCAN-AGI Memory Module
Hierarchical, distributed memory with multiple specialized types
"""

from .base import (Memory, MemoryConfig, MemoryException, MemoryQuery,
                   MemoryStats, MemoryType)
from .consolidation import (ConsolidationStrategy, MemoryConsolidator,
                            MemoryOptimizer)
from .distributed import (ConsistencyLevel, DistributedMemory,
                          MemoryFederation, MemoryNode)
from .hierarchical import HierarchicalMemory, MemoryLevel
from .persistence import (CompressionType, MemoryPersistence,
                          MemoryVersionControl)
from .retrieval import (AttentionMechanism, MemoryIndex, MemorySearch,
                        RetrievalResult)
from .specialized import (Concept, Episode, EpisodicMemory, ProceduralMemory,
                          SemanticMemory, Skill, WorkingMemory,
                          WorkingMemoryBuffer)

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
