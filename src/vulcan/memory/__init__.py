"""
VULCAN-AGI Memory Module
Hierarchical, distributed memory with multiple specialized types
"""

from .base import (
    Memory,
    MemoryConfig,
    MemoryException,
    MemoryQuery,
    MemoryStats,
    MemoryType,
    MemoryUsageMonitor,
)
from .consolidation import ConsolidationStrategy, MemoryConsolidator, MemoryOptimizer
from .distributed import (
    ConnectionPool,
    ConsistencyLevel,
    DistributedCheckpoint,
    DistributedMemory,
    MemoryFederation,
    MemoryNode,
)
from .hierarchical import HierarchicalMemory, MemoryLevel
from .learning_persistence import LearningStatePersistence
from .persistence import (
    CompressionStats,
    CompressionType,
    MemoryPersistence,
    MemoryVersionControl,
)
from .retrieval import (
    AttentionMechanism,
    MemoryIndex,
    MemorySearch,
    RetrievalResult,
    ShardedMemoryIndex,
)
from .specialized import (
    Concept,
    Episode,
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    Skill,
    WorkingMemory,
    WorkingMemoryBuffer,
)

# Alias for API Gateway compatibility
VectorMemoryStore = MemoryIndex

__all__ = [
    # Base
    "MemoryType",
    "Memory",
    "MemoryQuery",
    "MemoryConfig",
    "MemoryStats",
    "MemoryException",
    "MemoryUsageMonitor",
    # Hierarchical
    "HierarchicalMemory",
    "MemoryLevel",
    # Distributed
    "DistributedMemory",
    "MemoryFederation",
    "MemoryNode",
    "ConsistencyLevel",
    "ConnectionPool",
    "DistributedCheckpoint",
    # Persistence
    "MemoryPersistence",
    "MemoryVersionControl",
    "CompressionType",
    "CompressionStats",
    "LearningStatePersistence",
    # Retrieval
    "MemoryIndex",
    "MemorySearch",
    "AttentionMechanism",
    "RetrievalResult",
    "VectorMemoryStore",  # Added - Alias for MemoryIndex
    "ShardedMemoryIndex",
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
