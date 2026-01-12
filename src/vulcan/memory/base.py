"""Base classes and core types for memory system"""

import logging
import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# MEMORY TYPES
# ============================================================


class MemoryType(Enum):
    """Types of memory in the system."""

    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    LONG_TERM = "long_term"
    CACHE = "cache"


class CompressionType(Enum):
    """Memory compression types."""

    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    NEURAL = "neural"
    SEMANTIC = "semantic"


class ConsistencyLevel(Enum):
    """Consistency levels for distributed memory."""

    EVENTUAL = "eventual"
    STRONG = "strong"
    LINEARIZABLE = "linearizable"


# ============================================================
# BASE MEMORY CLASS
# ============================================================


@dataclass
class Memory:
    """Base memory unit."""

    id: str
    type: MemoryType
    content: Any
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5
    decay_rate: float = 0.01
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    compression_type: Optional[CompressionType] = None

    def __post_init__(self):
        """Validate memory parameters after initialization."""
        # Validate importance is in valid range
        if not 0 <= self.importance <= 1:
            logger.warning(f"Importance {self.importance} out of range [0,1], clamping")
            self.importance = float(np.clip(self.importance, 0, 1))

        # Validate decay_rate is non-negative
        if self.decay_rate < 0:
            logger.warning(f"Negative decay_rate {self.decay_rate}, setting to 0")
            self.decay_rate = 0.0

        # Validate access_count is non-negative
        if self.access_count < 0:
            logger.warning(f"Negative access_count {self.access_count}, setting to 0")
            self.access_count = 0

        # Validate timestamp is reasonable
        current_time = time.time()
        if self.timestamp > current_time + 60:  # Allow 60s future for clock skew
            logger.warning(
                f"Timestamp {self.timestamp} is in the future, using current time"
            )
            self.timestamp = current_time

    def compute_salience(self, current_time: Optional[float] = None) -> float:
        """Compute current salience of memory with validation."""
        if current_time is None:
            current_time = time.time()

        # Validate time
        if current_time < self.timestamp:
            logger.warning(
                f"Current time {current_time} is before timestamp {self.timestamp}"
            )
            current_time = self.timestamp

        age = current_time - self.timestamp

        # Ensure age is non-negative
        age = max(0, age)

        recency = np.exp(-self.decay_rate * age)
        frequency = np.log(1 + self.access_count)

        salience = self.importance * 0.4 + recency * 0.3 + frequency * 0.3

        # Return salience value - can exceed 1.0 for high importance/frequency memories
        return float(salience)

    def access(self):
        """Record memory access."""
        self.access_count += 1
        self.metadata["last_access"] = time.time()

    def decay(self, time_delta: float):
        """Apply time-based decay with validation."""
        if time_delta < 0:
            logger.warning(f"Negative time delta {time_delta}, using 0")
            time_delta = 0

        self.importance *= np.exp(-self.decay_rate * time_delta)
        self.importance = max(0.0, min(1.0, self.importance))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "metadata": self.metadata,
            "compressed": self.compressed,
        }

        # Include compression type if compressed
        if self.compressed and self.compression_type:
            result["compression_type"] = self.compression_type.value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create Memory from dictionary."""
        # Convert type string back to enum
        memory_type = MemoryType(data["type"])

        # Convert compression type if present
        compression_type = None
        if "compression_type" in data:
            compression_type = CompressionType(data["compression_type"])

        return cls(
            id=data["id"],
            type=memory_type,
            content=data.get("content"),
            timestamp=data["timestamp"],
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            decay_rate=data.get("decay_rate", 0.01),
            metadata=data.get("metadata", {}),
            compressed=data.get("compressed", False),
            compression_type=compression_type,
        )


# ============================================================
# MEMORY CONFIGURATION
# ============================================================


@dataclass
class MemoryConfig:
    """Configuration for memory system."""

    # Capacity limits
    max_working_memory: int = 7
    max_short_term: int = 100
    max_long_term: int = 1000000

    # Performance settings
    enable_compression: bool = True
    compression_threshold: int = 1024
    compression_type: CompressionType = CompressionType.LZ4

    # Consolidation settings
    consolidation_interval: float = 60.0
    consolidation_threshold: float = 0.3

    # Distributed settings
    enable_distributed: bool = False
    replication_factor: int = 3
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL

    # Persistence settings
    enable_persistence: bool = True
    persistence_path: str = "./memory_store"
    checkpoint_interval: float = 300.0

    # Search settings
    enable_indexing: bool = True
    index_update_batch: int = 100
    similarity_threshold: float = 0.7
    
    # New configuration options for improvements
    enable_connection_pooling: bool = True
    max_connections_per_node: int = 5
    shard_size: int = 100000
    enable_memory_monitoring: bool = True
    memory_warning_threshold_mb: float = 1000

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate capacity limits
        if self.max_working_memory < 1:
            logger.warning(
                f"max_working_memory {self.max_working_memory} too low, setting to 1"
            )
            self.max_working_memory = 1

        if self.max_short_term < 1:
            logger.warning(
                f"max_short_term {self.max_short_term} too low, setting to 1"
            )
            self.max_short_term = 1

        if self.max_long_term < 1:
            logger.warning(f"max_long_term {self.max_long_term} too low, setting to 1")
            self.max_long_term = 1

        # Validate thresholds
        if not 0 <= self.consolidation_threshold <= 1:
            logger.warning(
                f"consolidation_threshold {self.consolidation_threshold} out of range, clamping"
            )
            self.consolidation_threshold = float(
                np.clip(self.consolidation_threshold, 0, 1)
            )

        if not 0 <= self.similarity_threshold <= 1:
            logger.warning(
                f"similarity_threshold {self.similarity_threshold} out of range, clamping"
            )
            self.similarity_threshold = float(np.clip(self.similarity_threshold, 0, 1))

        # Validate intervals
        if self.consolidation_interval < 0:
            logger.warning(
                f"consolidation_interval {self.consolidation_interval} negative, setting to 0"
            )
            self.consolidation_interval = 0

        if self.checkpoint_interval < 0:
            logger.warning(
                f"checkpoint_interval {self.checkpoint_interval} negative, setting to 0"
            )
            self.checkpoint_interval = 0

        # Validate replication factor
        if self.replication_factor < 1:
            logger.warning(
                f"replication_factor {self.replication_factor} too low, setting to 1"
            )
            self.replication_factor = 1
        
        # Validate new configuration options
        if self.max_connections_per_node < 1:
            logger.warning(
                f"max_connections_per_node {self.max_connections_per_node} too low, setting to 1"
            )
            self.max_connections_per_node = 1
        
        if self.shard_size < 1000:
            logger.warning(
                f"shard_size {self.shard_size} too low, setting to 1000"
            )
            self.shard_size = 1000
        
        if self.memory_warning_threshold_mb < 0:
            logger.warning(
                f"memory_warning_threshold_mb {self.memory_warning_threshold_mb} negative, setting to 0"
            )
            self.memory_warning_threshold_mb = 0


# ============================================================
# MEMORY QUERY
# ============================================================


@dataclass
class MemoryQuery:
    """Query for memory retrieval."""

    query_type: str  # similarity, exact, temporal, causal
    content: Optional[Any] = None
    embedding: Optional[np.ndarray] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    time_range: Optional[Tuple[float, float]] = None
    limit: int = 10
    threshold: float = 0.5
    include_metadata: bool = False

    def __post_init__(self):
        """Validate query parameters."""
        # Validate limit
        if self.limit < 1:
            logger.warning(f"Query limit {self.limit} too low, setting to 1")
            self.limit = 1

        # Validate threshold
        if not 0 <= self.threshold <= 1:
            logger.warning(f"Query threshold {self.threshold} out of range, clamping")
            self.threshold = float(np.clip(self.threshold, 0, 1))

        # Validate time_range if provided
        if self.time_range is not None:
            start, end = self.time_range
            if start > end:
                logger.warning(f"time_range start {start} > end {end}, swapping")
                self.time_range = (end, start)


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""

    memories: List[Memory]
    scores: List[float]
    query_time_ms: float
    total_matches: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result data."""
        # Ensure lengths match
        if len(self.memories) != len(self.scores):
            logger.warning(
                f"Memories count {len(self.memories)} != scores count {len(self.scores)}"
            )
            # Truncate to shorter length
            min_len = min(len(self.memories), len(self.scores))
            self.memories = self.memories[:min_len]
            self.scores = self.scores[:min_len]

        # Validate query_time_ms
        if self.query_time_ms < 0:
            logger.warning(f"Negative query_time_ms {self.query_time_ms}, setting to 0")
            self.query_time_ms = 0

        # Validate total_matches
        if self.total_matches < 0:
            logger.warning(f"Negative total_matches {self.total_matches}, setting to 0")
            self.total_matches = 0


# ============================================================
# MEMORY STATISTICS
# ============================================================


@dataclass
class MemoryStats:
    """Statistics for memory system."""

    total_memories: int = 0
    by_type: Dict[MemoryType, int] = field(default_factory=dict)

    # Performance metrics
    avg_retrieval_time_ms: float = 0
    cache_hit_rate: float = 0
    compression_ratio: float = 0

    # Resource usage
    memory_bytes: int = 0
    index_bytes: int = 0

    # Activity metrics
    total_queries: int = 0
    total_stores: int = 0
    total_consolidations: int = 0

    def update(self, other: "MemoryStats"):
        """Update stats with another stats object."""
        self.total_memories += other.total_memories
        for mem_type, count in other.by_type.items():
            self.by_type[mem_type] = self.by_type.get(mem_type, 0) + count

        # Update averages
        if other.total_queries > 0:
            total = self.total_queries + other.total_queries
            if total > 0:
                self.avg_retrieval_time_ms = (
                    self.avg_retrieval_time_ms * self.total_queries
                    + other.avg_retrieval_time_ms * other.total_queries
                ) / total

        self.total_queries += other.total_queries
        self.total_stores += other.total_stores
        self.total_consolidations += other.total_consolidations

    def reset(self):
        """Reset all statistics to initial values."""
        self.total_memories = 0
        self.by_type.clear()
        self.avg_retrieval_time_ms = 0
        self.cache_hit_rate = 0
        self.compression_ratio = 0
        self.memory_bytes = 0
        self.index_bytes = 0
        self.total_queries = 0
        self.total_stores = 0
        self.total_consolidations = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_memories": self.total_memories,
            "by_type": {k.value: v for k, v in self.by_type.items()},
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "compression_ratio": self.compression_ratio,
            "memory_bytes": self.memory_bytes,
            "index_bytes": self.index_bytes,
            "total_queries": self.total_queries,
            "total_stores": self.total_stores,
            "total_consolidations": self.total_consolidations,
        }


# ============================================================
# MEMORY EXCEPTION
# ============================================================


class MemoryException(Exception):
    """Base exception for memory operations."""


class MemoryCapacityException(MemoryException):
    """Raised when memory capacity is exceeded."""


class MemoryRetrievalException(MemoryException):
    """Raised when memory retrieval fails."""


class MemoryCorruptionException(MemoryException):
    """Raised when memory data is corrupted."""


class MemoryLockException(MemoryException):
    """Raised when memory lock cannot be acquired."""


# ============================================================
# ABSTRACT BASE CLASSES
# ============================================================


class BaseMemorySystem(ABC):
    """Abstract base class for memory systems."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.stats = MemoryStats()
        self._lock = threading.RLock()
        self._shutdown = False

    @abstractmethod
    def store(self, content: Any, **kwargs) -> Memory:
        """Store content in memory."""

    @abstractmethod
    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve memories matching query."""

    @abstractmethod
    def forget(self, memory_id: str) -> bool:
        """Remove memory by ID."""

    @abstractmethod
    def consolidate(self) -> int:
        """Consolidate memories."""

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        with self._lock:
            return self.stats

    def is_shutdown(self) -> bool:
        """Check if system is shutdown."""
        return self._shutdown

    def shutdown(self):
        """Shutdown memory system and cleanup resources."""
        with self._lock:
            if self._shutdown:
                logger.warning("Memory system already shutdown")
                return

            logger.info("Shutting down memory system")
            self._shutdown = True

            # Subclasses should override to add specific cleanup
            logger.info("Memory system shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
        return False


# ============================================================
# MEMORY USAGE MONITORING
# ============================================================


class MemoryUsageMonitor:
    """Monitor actual memory consumption using tracemalloc."""
    
    def __init__(
        self,
        warning_threshold_mb: float = 1000,
        critical_threshold_mb: float = 2000,
    ):
        """Initialize memory usage monitor.
        
        Args:
            warning_threshold_mb: Warning threshold in megabytes
            critical_threshold_mb: Critical threshold in megabytes
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self._lock = threading.RLock()
        self._monitoring = False
        self._usage_by_type: Dict[MemoryType, int] = {}
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Started memory tracing with tracemalloc")
    
    def track_memory(self, memory: Memory):
        """Track memory consumption for a memory object."""
        with self._lock:
            # Estimate memory size
            size = self._estimate_memory_size(memory)
            
            # Update usage by type
            if memory.type not in self._usage_by_type:
                self._usage_by_type[memory.type] = 0
            self._usage_by_type[memory.type] += size
            
            # Check thresholds
            total_mb = sum(self._usage_by_type.values()) / (1024 * 1024)
            
            if total_mb >= self.critical_threshold_mb:
                logger.critical(
                    f"Memory usage critical: {total_mb:.2f} MB "
                    f"(threshold: {self.critical_threshold_mb} MB)"
                )
            elif total_mb >= self.warning_threshold_mb:
                logger.warning(
                    f"Memory usage warning: {total_mb:.2f} MB "
                    f"(threshold: {self.warning_threshold_mb} MB)"
                )
    
    def untrack_memory(self, memory: Memory):
        """Remove memory from tracking."""
        with self._lock:
            size = self._estimate_memory_size(memory)
            
            if memory.type in self._usage_by_type:
                self._usage_by_type[memory.type] = max(
                    0, self._usage_by_type[memory.type] - size
                )
    
    def _estimate_memory_size(self, memory: Memory) -> int:
        """Estimate memory size in bytes."""
        size = 0
        
        # Content size
        if hasattr(memory.content, '__sizeof__'):
            size += memory.content.__sizeof__()
        
        # Embedding size
        if memory.embedding is not None:
            size += memory.embedding.nbytes
        
        # Metadata size (rough estimate)
        size += len(str(memory.metadata))
        
        return size
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        with self._lock:
            current, peak = tracemalloc.get_traced_memory()
            
            return {
                'current_mb': current / (1024 * 1024),
                'peak_mb': peak / (1024 * 1024),
                'by_type': {
                    k.value: v / (1024 * 1024) 
                    for k, v in self._usage_by_type.items()
                },
                'warning_threshold_mb': self.warning_threshold_mb,
                'critical_threshold_mb': self.critical_threshold_mb,
            }
    
    def get_adaptive_capacity(self, memory_type: MemoryType) -> int:
        """Calculate adaptive capacity based on current usage."""
        with self._lock:
            stats = self.get_usage_stats()
            current_mb = stats['current_mb']
            
            # If we're above warning threshold, reduce capacity
            if current_mb >= self.warning_threshold_mb:
                reduction_factor = min(
                    0.5,
                    (current_mb - self.warning_threshold_mb) / self.warning_threshold_mb
                )
                return int(10000 * (1 - reduction_factor))
            
            # Normal capacity
            return 100000
    
    def shutdown(self):
        """Clean up monitoring resources."""
        with self._lock:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
                logger.info("Stopped memory tracing")
