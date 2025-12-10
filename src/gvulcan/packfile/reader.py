"""
Adaptive Packfile Reader

This module provides intelligent packfile reading with adaptive range selection,
prefetching strategies, and optimization for different artifact types and access patterns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ArtifactClass(Enum):
    """Classification of artifacts for reading optimization"""

    CONFLICT = "conflict"  # Merge conflict artifacts
    CITATION = "citation"  # Citation/reference artifacts
    INLINE = "inline"  # Small inline artifacts
    DOCUMENT = "document"  # Full documents
    MEDIA = "media"  # Images, audio, video
    CODE = "code"  # Source code artifacts


class ChunkType(Enum):
    """Type of chunk in packfile"""

    PARENT = "parent"  # Complete artifact
    CHILD = "child"  # Child/delta of parent
    STANDALONE = "standalone"  # Independent chunk


class ReadStrategy(Enum):
    """Strategy for reading from packfile"""

    MINIMAL = "minimal"  # Read exact bytes only
    CONTEXT = "context"  # Read with surrounding context
    PREFETCH = "prefetch"  # Aggressively prefetch related chunks
    STREAMING = "streaming"  # Stream large chunks


@dataclass
class ArtifactMeta:
    """
    Metadata for an artifact in a packfile.

    Attributes:
        artifact_class: Classification of artifact
        chunk_type: Type of chunk
        parent_size: Size of parent artifact
        child_offset: Offset of child within parent
        child_length: Length of child data
        compression_ratio: Compression ratio estimate
        access_frequency: Historical access frequency
    """

    artifact_class: str
    chunk_type: str
    parent_size: int
    child_offset: int
    child_length: int
    compression_ratio: float = 1.0
    access_frequency: int = 0

    def get_artifact_class(self) -> ArtifactClass:
        """Get artifact class as enum"""
        try:
            return ArtifactClass(self.artifact_class)
        except ValueError:
            return ArtifactClass.DOCUMENT

    def get_chunk_type(self) -> ChunkType:
        """Get chunk type as enum"""
        try:
            return ChunkType(self.chunk_type)
        except ValueError:
            return ChunkType.STANDALONE


@dataclass
class ReadRange:
    """Byte range for reading"""

    start: int
    end: int
    reason: str = ""

    @property
    def size(self) -> int:
        """Size of range in bytes"""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"ReadRange({self.start}-{self.end}, {self.size} bytes, {self.reason})"


@dataclass
class ReadResult:
    """Result of a read operation"""

    data: bytes
    range: ReadRange
    strategy: ReadStrategy
    cache_hit: bool = False
    latency_ms: float = 0.0


def compute_adaptive_range(
    meta: ArtifactMeta, strategy: Optional[ReadStrategy] = None
) -> Optional[Tuple[int, int]]:
    """
    Compute adaptive byte range for reading based on artifact metadata.

    This function implements intelligent range selection that:
    1. Considers artifact type and access patterns
    2. Includes context for certain artifact types
    3. Optimizes for cache line and alignment
    4. Balances read size vs. number of requests

    Args:
        meta: Artifact metadata
        strategy: Optional reading strategy override

    Returns:
        Tuple of (start, end) byte offsets, or None for full read

    Example:
        >>> meta = ArtifactMeta(
        ...     artifact_class="citation",
        ...     chunk_type="child",
        ...     parent_size=50000,
        ...     child_offset=10000,
        ...     child_length=500
        ... )
        >>> start, end = compute_adaptive_range(meta)
        >>> print(f"Range: {start}-{end} ({end-start} bytes)")
    """
    artifact_class = meta.get_artifact_class()
    chunk_type = meta.get_chunk_type()

    # Determine strategy if not provided
    if strategy is None:
        strategy = _determine_strategy(meta)

    # Small artifacts - read in full
    if meta.child_length < 4096 and chunk_type == ChunkType.CHILD:
        return (meta.child_offset, meta.child_offset + meta.child_length)

    # Conflict and citation artifacts need surrounding context
    if artifact_class in [ArtifactClass.CONFLICT, ArtifactClass.CITATION]:
        if chunk_type == ChunkType.CHILD:
            # Include 4 KB context on each side
            context_size = 4096
            start = max(0, meta.child_offset - context_size)
            end = min(
                meta.parent_size, meta.child_offset + meta.child_length + context_size
            )

            logger.debug(
                f"Computed context range for {artifact_class.value}: "
                f"{start}-{end} ({end - start} bytes)"
            )
            return (start, end)

    # Large parents - use streaming strategy
    if chunk_type == ChunkType.CHILD and meta.parent_size > 12 * 1024:
        # Read child with minimal context
        context_size = 2048
        start = max(0, meta.child_offset - context_size)
        end = min(
            meta.parent_size, meta.child_offset + meta.child_length + context_size
        )

        return (start, end)

    # Media artifacts - align to block boundaries
    if artifact_class == ArtifactClass.MEDIA:
        block_size = 65536  # 64 KB blocks
        start = (meta.child_offset // block_size) * block_size
        end = (
            (meta.child_offset + meta.child_length + block_size - 1) // block_size
        ) * block_size
        end = min(end, meta.parent_size)

        return (start, end)

    # Code artifacts - read with line context
    if artifact_class == ArtifactClass.CODE and chunk_type == ChunkType.CHILD:
        # Estimate ~80 bytes per line, read ±50 lines
        line_context = 50 * 80
        start = max(0, meta.child_offset - line_context)
        end = min(
            meta.parent_size, meta.child_offset + meta.child_length + line_context
        )

        return (start, end)

    # Frequently accessed - prefetch more
    if meta.access_frequency > 10 and strategy == ReadStrategy.PREFETCH:
        prefetch_size = min(8192, meta.parent_size // 4)
        start = max(0, meta.child_offset - prefetch_size)
        end = min(
            meta.parent_size, meta.child_offset + meta.child_length + prefetch_size
        )

        return (start, end)

    # Default: exact range
    if chunk_type == ChunkType.CHILD:
        return (meta.child_offset, meta.child_offset + meta.child_length)

    # Parent or standalone: full read
    return None


def _determine_strategy(meta: ArtifactMeta) -> ReadStrategy:
    """Determine optimal reading strategy based on metadata"""
    artifact_class = meta.get_artifact_class()

    # Conflict/citation artifacts benefit from context
    if artifact_class in [ArtifactClass.CONFLICT, ArtifactClass.CITATION]:
        return ReadStrategy.CONTEXT

    # Media artifacts use streaming
    if artifact_class == ArtifactClass.MEDIA:
        if meta.child_length > 1024 * 1024:  # > 1 MB
            return ReadStrategy.STREAMING
        return ReadStrategy.MINIMAL

    # Frequently accessed artifacts get prefetched
    if meta.access_frequency > 10:
        return ReadStrategy.PREFETCH

    # Small artifacts - minimal read
    if meta.child_length < 4096:
        return ReadStrategy.MINIMAL

    # Default: context-aware reading
    return ReadStrategy.CONTEXT


def compute_multi_chunk_ranges(
    metas: List[ArtifactMeta], max_gap: int = 8192
) -> List[ReadRange]:
    """
    Compute optimized ranges for reading multiple chunks.

    Combines nearby chunks into single ranges to reduce read operations.

    Args:
        metas: List of artifact metadata
        max_gap: Maximum gap to bridge between chunks

    Returns:
        List of optimized read ranges
    """
    if not metas:
        return []

    # Sort by offset
    sorted_metas = sorted(metas, key=lambda m: m.child_offset)

    # Compute individual ranges
    ranges = []
    for meta in sorted_metas:
        range_tuple = compute_adaptive_range(meta)
        if range_tuple:
            start, end = range_tuple
            ranges.append(
                ReadRange(
                    start=start,
                    end=end,
                    reason=f"{meta.artifact_class}:{meta.chunk_type}",
                )
            )

    if not ranges:
        return []

    # Merge overlapping or nearby ranges
    merged = [ranges[0]]

    for current in ranges[1:]:
        last = merged[-1]

        # Check if ranges overlap or are within max_gap
        if current.start <= last.end + max_gap:
            # Merge ranges
            last.end = max(last.end, current.end)
            last.reason += f", {current.reason}"
        else:
            # Start new range
            merged.append(current)

    logger.debug(
        f"Merged {len(ranges)} ranges into {len(merged)} "
        f"(total size: {sum(r.size for r in merged)} bytes)"
    )

    return merged


def estimate_read_cost(
    meta: ArtifactMeta, read_latency_ms: float = 10.0, bytes_per_ms: float = 100_000.0
) -> float:
    """
    Estimate cost of reading an artifact.

    Args:
        meta: Artifact metadata
        read_latency_ms: Base read latency in milliseconds
        bytes_per_ms: Throughput in bytes per millisecond

    Returns:
        Estimated cost in milliseconds
    """
    range_tuple = compute_adaptive_range(meta)

    if range_tuple:
        start, end = range_tuple
        read_size = end - start
    else:
        read_size = meta.parent_size

    # Account for compression
    actual_size = read_size * meta.compression_ratio

    # Cost = latency + transfer time
    transfer_time = actual_size / bytes_per_ms
    total_cost = read_latency_ms + transfer_time

    return total_cost


def should_prefetch(
    meta: ArtifactMeta, cache_hit_rate: float = 0.5, cost_threshold: float = 50.0
) -> bool:
    """
    Determine if artifact should be prefetched.

    Args:
        meta: Artifact metadata
        cache_hit_rate: Current cache hit rate
        cost_threshold: Cost threshold for prefetching (ms)

    Returns:
        True if should prefetch
    """
    # Don't prefetch if cache is working well
    if cache_hit_rate > 0.8:
        return False

    # Prefetch frequently accessed items
    if meta.access_frequency > 5:
        return True

    # Prefetch if read cost is high
    cost = estimate_read_cost(meta)
    if cost > cost_threshold:
        return True

    # Prefetch small items speculatively
    if meta.child_length < 8192:
        return True

    return False


def optimize_batch_reads(
    metas: List[ArtifactMeta],
    max_total_size: int = 10 * 1024 * 1024,  # 10 MB
) -> List[List[ArtifactMeta]]:
    """
    Optimize a batch of reads into efficient groups.

    Args:
        metas: List of artifacts to read
        max_total_size: Maximum total size per batch

    Returns:
        List of batches
    """
    # Sort by offset for sequential access
    sorted_metas = sorted(metas, key=lambda m: m.child_offset)

    batches = []
    current_batch = []
    current_size = 0

    for meta in sorted_metas:
        estimated_size = meta.child_length

        if current_size + estimated_size > max_total_size and current_batch:
            # Start new batch
            batches.append(current_batch)
            current_batch = [meta]
            current_size = estimated_size
        else:
            current_batch.append(meta)
            current_size += estimated_size

    if current_batch:
        batches.append(current_batch)

    logger.debug(f"Optimized {len(metas)} reads into {len(batches)} batches")

    return batches


class AdaptiveReader:
    """
    Adaptive packfile reader with intelligent range selection and caching.

    Example:
        reader = AdaptiveReader()

        meta = ArtifactMeta(
            artifact_class="citation",
            chunk_type="child",
            parent_size=100000,
            child_offset=50000,
            child_length=1000
        )

        range_tuple = reader.compute_range(meta)
        if range_tuple:
            data = read_from_pack(range_tuple)
    """

    def __init__(self, enable_caching: bool = True, enable_prefetch: bool = True):
        self.enable_caching = enable_caching
        self.enable_prefetch = enable_prefetch
        self.access_stats: Dict[str, int] = {}

    def compute_range(
        self, meta: ArtifactMeta, strategy: Optional[ReadStrategy] = None
    ) -> Optional[Tuple[int, int]]:
        """Compute range with access tracking"""
        # Track access
        key = f"{meta.artifact_class}:{meta.chunk_type}:{meta.child_offset}"
        self.access_stats[key] = self.access_stats.get(key, 0) + 1

        # Update access frequency in metadata
        meta.access_frequency = self.access_stats[key]

        return compute_adaptive_range(meta, strategy)

    def get_stats(self) -> Dict[str, Any]:
        """Get reader statistics"""
        return {
            "total_accesses": sum(self.access_stats.values()),
            "unique_artifacts": len(self.access_stats),
            "hot_artifacts": len([k for k, v in self.access_stats.items() if v > 5]),
        }
