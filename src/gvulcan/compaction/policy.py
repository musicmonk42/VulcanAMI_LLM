"""
Compaction Policy System

This module provides comprehensive compaction strategies for LSM-tree based storage
with support for Time-Windowed, Leveled, and Hybrid compaction policies.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CompactionStrategy(Enum):
    """Available compaction strategies"""

    TIME_WINDOWED = "time_windowed"
    LEVELED = "leveled"
    TIERED = "tiered"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class CompactionPriority(Enum):
    """Priority levels for compaction tasks"""

    CRITICAL = "critical"  # Immediate compaction needed
    HIGH = "high"  # Should compact soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Can be deferred
    IDLE = "idle"  # Only when idle


@dataclass
class PackStats:
    """
    Statistics for a pack file

    Attributes:
        pack_id: Unique pack identifier
        read_qps: Read queries per second
        write_qps: Write queries per second
        domain: Data domain/category
        live_bytes: Number of live (non-deleted) bytes
        total_bytes: Total bytes in pack
        fragmentation_ratio: Ratio of dead to total bytes
        creation_time: When pack was created
        last_access_time: Last access timestamp
        level: Level in LSM tree (if applicable)
        compaction_count: Number of times compacted
        size_amplification: Size amplification factor
        tombstone_count: Number of tombstones
    """

    pack_id: str
    read_qps: float
    write_qps: float
    domain: str
    live_bytes: int
    total_bytes: int
    fragmentation_ratio: float
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    level: int = 0
    compaction_count: int = 0
    size_amplification: float = 1.0
    tombstone_count: int = 0

    def __post_init__(self):
        """Validate pack statistics"""
        if self.live_bytes > self.total_bytes:
            raise ValueError(
                f"live_bytes ({self.live_bytes}) cannot exceed total_bytes ({self.total_bytes})"
            )

        if not 0 <= self.fragmentation_ratio <= 1:
            raise ValueError(
                f"fragmentation_ratio must be in [0, 1]: {self.fragmentation_ratio}"
            )

        if self.read_qps < 0 or self.write_qps < 0:
            raise ValueError("QPS values cannot be negative")

    @property
    def dead_bytes(self) -> int:
        """Calculate dead (garbage) bytes"""
        return self.total_bytes - self.live_bytes

    @property
    def utilization(self) -> float:
        """Calculate space utilization (0-1)"""
        if self.total_bytes == 0:
            return 0.0
        return self.live_bytes / self.total_bytes

    @property
    def age_seconds(self) -> float:
        """Get age of pack in seconds"""
        return (datetime.now() - self.creation_time).total_seconds()

    @property
    def access_recency_seconds(self) -> float:
        """Get time since last access in seconds"""
        return (datetime.now() - self.last_access_time).total_seconds()

    @property
    def is_cold(self) -> bool:
        """Check if pack is cold (not recently accessed)"""
        return self.access_recency_seconds > 3600  # 1 hour

    @property
    def write_amplification_estimate(self) -> float:
        """Estimate write amplification for this pack"""
        if self.compaction_count == 0:
            return 1.0
        return 1.0 + (self.compaction_count * 0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d["creation_time"] = self.creation_time.isoformat()
        d["last_access_time"] = self.last_access_time.isoformat()
        return d


@dataclass
class CompactionTask:
    """
    Represents a compaction task

    Attributes:
        task_id: Unique task identifier
        strategy: Compaction strategy to use
        input_packs: List of pack IDs to compact
        output_level: Target level for output
        priority: Task priority
        estimated_cost: Estimated I/O cost
        estimated_duration: Estimated duration in seconds
        created_at: Task creation time
        metadata: Additional metadata
    """

    task_id: str
    strategy: CompactionStrategy
    input_packs: List[str]
    output_level: int
    priority: CompactionPriority = CompactionPriority.MEDIUM
    estimated_cost: float = 0.0
    estimated_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "strategy": self.strategy.value,
            "input_packs": self.input_packs,
            "output_level": self.output_level,
            "priority": self.priority.value,
            "estimated_cost": self.estimated_cost,
            "estimated_duration": self.estimated_duration,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CompactionResult:
    """
    Result of a compaction operation

    Attributes:
        task_id: Task identifier
        input_packs: Input pack IDs
        output_packs: Output pack IDs
        bytes_read: Bytes read during compaction
        bytes_written: Bytes written during compaction
        duration_seconds: Duration in seconds
        live_bytes_recovered: Live bytes recovered
        dead_bytes_removed: Dead bytes removed
        success: Whether compaction succeeded
        error: Error message if failed
    """

    task_id: str
    input_packs: List[str]
    output_packs: List[str]
    bytes_read: int
    bytes_written: int
    duration_seconds: float
    live_bytes_recovered: int
    dead_bytes_removed: int
    success: bool = True
    error: Optional[str] = None

    @property
    def write_amplification(self) -> float:
        """Calculate write amplification for this compaction"""
        if self.live_bytes_recovered == 0:
            return 0.0
        return self.bytes_written / self.live_bytes_recovered

    @property
    def space_reclamation_ratio(self) -> float:
        """Calculate space reclamation ratio"""
        if self.bytes_read == 0:
            return 0.0
        return self.dead_bytes_removed / self.bytes_read


class TimeWindowedCompaction:
    """
    Time-Windowed Compaction Strategy (TWCS)

    Optimized for time-series data where recent data is frequently accessed
    and old data is rarely or never accessed. Compacts data within time windows.

    Features:
    - Organizes data into time windows
    - Minimal compaction of recent windows
    - Aggressive compaction of old windows
    - Efficient for append-only workloads
    - Excellent for time-series data
    """

    def __init__(
        self,
        window_size_hours: int = 24,
        max_windows_per_tier: int = 4,
        compaction_lag_hours: int = 1,
    ):
        """
        Initialize Time-Windowed Compaction strategy.

        Args:
            window_size_hours: Size of each time window in hours
            max_windows_per_tier: Maximum windows before compacting
            compaction_lag_hours: Delay before compacting recent data
        """
        self.window_size_hours = window_size_hours
        self.max_windows_per_tier = max_windows_per_tier
        self.compaction_lag_hours = compaction_lag_hours
        self.strategy = CompactionStrategy.TIME_WINDOWED

        logger.info(
            f"Initialized TimeWindowedCompaction: "
            f"window_size={window_size_hours}h, "
            f"max_windows={max_windows_per_tier}"
        )

    def select_packs(self, packs: List[PackStats]) -> List[CompactionTask]:
        """
        Select packs for time-windowed compaction.

        Args:
            packs: List of pack statistics

        Returns:
            List of compaction tasks
        """
        tasks = []

        # Group packs by time window
        windows = self._group_by_time_window(packs)

        # Check each window for compaction
        for window_start, window_packs in windows.items():
            # Skip recent windows (within compaction lag)
            window_age_hours = (datetime.now() - window_start).total_seconds() / 3600
            if window_age_hours < self.compaction_lag_hours:
                continue

            # Compact if too many packs in window
            if len(window_packs) >= self.max_windows_per_tier:
                task = CompactionTask(
                    task_id=f"twcs_{window_start.isoformat()}",
                    strategy=self.strategy,
                    input_packs=[p.pack_id for p in window_packs],
                    output_level=0,
                    priority=self._determine_priority(window_packs),
                    estimated_cost=self._estimate_cost(window_packs),
                    metadata={
                        "window_start": window_start.isoformat(),
                        "window_age_hours": window_age_hours,
                    },
                )
                tasks.append(task)

        return tasks

    def _group_by_time_window(
        self, packs: List[PackStats]
    ) -> Dict[datetime, List[PackStats]]:
        """Group packs by time window"""
        windows = defaultdict(list)

        for pack in packs:
            # Calculate window start
            window_start = self._get_window_start(pack.creation_time)
            windows[window_start].append(pack)

        return dict(windows)

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Get the start of the time window for a timestamp"""
        hours_since_epoch = timestamp.timestamp() / 3600
        window_number = int(hours_since_epoch / self.window_size_hours)
        window_start_hours = window_number * self.window_size_hours
        return datetime.fromtimestamp(window_start_hours * 3600)

    def _determine_priority(self, packs: List[PackStats]) -> CompactionPriority:
        """Determine priority for compacting these packs"""
        if len(packs) >= self.max_windows_per_tier * 2:
            return CompactionPriority.HIGH
        elif any(p.fragmentation_ratio > 0.5 for p in packs):
            return CompactionPriority.MEDIUM
        else:
            return CompactionPriority.LOW

    def _estimate_cost(self, packs: List[PackStats]) -> float:
        """Estimate I/O cost of compacting these packs"""
        total_bytes = sum(p.total_bytes for p in packs)
        # Cost = read all + write live data
        total_live = sum(p.live_bytes for p in packs)
        return (total_bytes + total_live) / (1024 * 1024)  # MB


class LeveledCompaction:
    """
    Leveled Compaction Strategy (LCS)

    Maintains data in sorted levels where each level is 10x larger than previous.
    Optimized for read-heavy workloads with good space efficiency.

    Features:
    - Sorted runs at each level
    - Predictable space amplification
    - Excellent read performance
    - Higher write amplification
    - Good for read-heavy workloads
    """

    def __init__(
        self,
        aggressive: bool = False,
        level_size_multiplier: int = 10,
        max_levels: int = 7,
        level0_trigger: int = 4,
    ):
        """
        Initialize Leveled Compaction strategy.

        Args:
            aggressive: Use aggressive compaction (lower thresholds)
            level_size_multiplier: Size multiplier between levels
            max_levels: Maximum number of levels
            level0_trigger: Number of L0 files to trigger compaction
        """
        self.aggressive = aggressive
        self.level_size_multiplier = level_size_multiplier
        self.max_levels = max_levels
        self.level0_trigger = level0_trigger
        self.strategy = CompactionStrategy.LEVELED

        # Adjust thresholds for aggressive mode
        if aggressive:
            self.level0_trigger = max(2, level0_trigger // 2)
            self.compaction_threshold = 0.7
        else:
            self.compaction_threshold = 0.8

        logger.info(
            f"Initialized LeveledCompaction: "
            f"aggressive={aggressive}, "
            f"multiplier={level_size_multiplier}, "
            f"l0_trigger={self.level0_trigger}"
        )

    def select_packs(self, packs: List[PackStats]) -> List[CompactionTask]:
        """
        Select packs for leveled compaction.

        Args:
            packs: List of pack statistics

        Returns:
            List of compaction tasks
        """
        tasks = []

        # Group packs by level
        levels = self._group_by_level(packs)

        # Check L0 compaction (always highest priority)
        if 0 in levels and len(levels[0]) >= self.level0_trigger:
            task = self._create_l0_compaction_task(levels[0], levels.get(1, []))
            tasks.append(task)

        # Check other levels
        for level in range(1, self.max_levels):
            if level not in levels:
                continue

            level_packs = levels[level]
            level_size = sum(p.total_bytes for p in level_packs)

            # Calculate target size for this level
            target_size = self._calculate_target_size(level)

            # Compact if level exceeds target
            if level_size > target_size * self.compaction_threshold:
                task = self._create_level_compaction_task(
                    level, level_packs, levels.get(level + 1, [])
                )
                if task:
                    tasks.append(task)

        return tasks

    def _group_by_level(self, packs: List[PackStats]) -> Dict[int, List[PackStats]]:
        """Group packs by level"""
        levels = defaultdict(list)
        for pack in packs:
            levels[pack.level].append(pack)
        return dict(levels)

    def _calculate_target_size(self, level: int) -> int:
        """Calculate target size for a level"""
        if level == 0:
            return 64 * 1024 * 1024  # 64MB for L0

        # Each level is multiplier times larger
        base_size = 64 * 1024 * 1024
        return base_size * (self.level_size_multiplier**level)

    def _create_l0_compaction_task(
        self, l0_packs: List[PackStats], l1_packs: List[PackStats]
    ) -> CompactionTask:
        """Create compaction task for L0 -> L1"""
        # Select all L0 packs and overlapping L1 packs
        input_packs = [p.pack_id for p in l0_packs]

        # Find overlapping L1 packs (simplified - in production would check key ranges)
        if l1_packs:
            input_packs.extend([p.pack_id for p in l1_packs[:2]])

        return CompactionTask(
            task_id=f"leveled_l0_to_l1_{datetime.now().timestamp()}",
            strategy=self.strategy,
            input_packs=input_packs,
            output_level=1,
            priority=CompactionPriority.HIGH,
            estimated_cost=self._estimate_cost(l0_packs + l1_packs[:2]),
            metadata={"source_level": 0, "target_level": 1},
        )

    def _create_level_compaction_task(
        self,
        level: int,
        level_packs: List[PackStats],
        next_level_packs: List[PackStats],
    ) -> Optional[CompactionTask]:
        """Create compaction task for level N -> level N+1"""
        if level >= self.max_levels - 1:
            return None

        # Select packs to compact (oldest first)
        sorted_packs = sorted(level_packs, key=lambda p: p.creation_time)
        input_packs = [p.pack_id for p in sorted_packs[:4]]

        # Add overlapping next level packs (simplified)
        if next_level_packs:
            input_packs.extend([p.pack_id for p in next_level_packs[:2]])

        return CompactionTask(
            task_id=f"leveled_l{level}_to_l{level + 1}_{datetime.now().timestamp()}",
            strategy=self.strategy,
            input_packs=input_packs,
            output_level=level + 1,
            priority=CompactionPriority.MEDIUM,
            estimated_cost=self._estimate_cost(sorted_packs[:4] + next_level_packs[:2]),
            metadata={"source_level": level, "target_level": level + 1},
        )

    def _estimate_cost(self, packs: List[PackStats]) -> float:
        """Estimate I/O cost"""
        total_bytes = sum(p.total_bytes for p in packs)
        total_live = sum(p.live_bytes for p in packs)
        return (total_bytes + total_live) / (1024 * 1024)


class TieredCompaction:
    """
    Tiered Compaction Strategy (STCS - Size-Tiered Compaction)

    Groups SSTables into tiers based on size. Compacts tables of similar size.
    Optimized for write-heavy workloads with good write throughput.

    Features:
    - Low write amplification
    - Higher space amplification
    - Good for write-heavy workloads
    - Simpler than leveled compaction
    """

    def __init__(
        self,
        min_threshold: int = 4,
        max_threshold: int = 32,
        bucket_low: float = 0.5,
        bucket_high: float = 1.5,
    ):
        """
        Initialize Tiered Compaction strategy.

        Args:
            min_threshold: Minimum tables to compact
            max_threshold: Maximum tables in one compaction
            bucket_low: Lower bound for size bucket (as ratio)
            bucket_high: Upper bound for size bucket (as ratio)
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.bucket_low = bucket_low
        self.bucket_high = bucket_high
        self.strategy = CompactionStrategy.TIERED

        logger.info(
            f"Initialized TieredCompaction: min={min_threshold}, max={max_threshold}"
        )

    def select_packs(self, packs: List[PackStats]) -> List[CompactionTask]:
        """Select packs for tiered compaction"""
        tasks = []

        # Sort packs by size
        sorted_packs = sorted(packs, key=lambda p: p.total_bytes)

        # Group into size buckets
        buckets = self._group_into_buckets(sorted_packs)

        # Find buckets ready for compaction
        for bucket_id, bucket_packs in buckets.items():
            if len(bucket_packs) >= self.min_threshold:
                # Take up to max_threshold packs
                compaction_packs = bucket_packs[: self.max_threshold]

                task = CompactionTask(
                    task_id=f"tiered_bucket_{bucket_id}_{datetime.now().timestamp()}",
                    strategy=self.strategy,
                    input_packs=[p.pack_id for p in compaction_packs],
                    output_level=0,
                    priority=self._determine_priority(compaction_packs),
                    estimated_cost=self._estimate_cost(compaction_packs),
                    metadata={"bucket_id": bucket_id},
                )
                tasks.append(task)

        return tasks

    def _group_into_buckets(self, packs: List[PackStats]) -> Dict[int, List[PackStats]]:
        """Group packs into size buckets"""
        if not packs:
            return {}

        buckets = defaultdict(list)

        for pack in packs:
            # Find appropriate bucket
            bucket_id = self._find_bucket(pack.total_bytes, packs)
            buckets[bucket_id].append(pack)

        return dict(buckets)

    def _find_bucket(self, size: int, all_packs: List[PackStats]) -> int:
        """Find bucket ID for a given size"""
        # Simple bucketing: log2 of size
        if size == 0:
            return 0
        return int(math.log2(size))

    def _determine_priority(self, packs: List[PackStats]) -> CompactionPriority:
        """Determine compaction priority"""
        avg_fragmentation = sum(p.fragmentation_ratio for p in packs) / len(packs)

        if avg_fragmentation > 0.6:
            return CompactionPriority.HIGH
        elif avg_fragmentation > 0.4:
            return CompactionPriority.MEDIUM
        else:
            return CompactionPriority.LOW

    def _estimate_cost(self, packs: List[PackStats]) -> float:
        """Estimate I/O cost"""
        total_bytes = sum(p.total_bytes for p in packs)
        total_live = sum(p.live_bytes for p in packs)
        return (total_bytes + total_live) / (1024 * 1024)


class HybridCompaction:
    """
    Hybrid Compaction Strategy

    Combines tiered compaction for lower levels and leveled for upper levels.
    Balances write and read performance.

    Features:
    - Tiered compaction for write-heavy lower levels
    - Leveled compaction for read-optimized upper levels
    - Balanced performance characteristics
    - Configurable transition point
    """

    def __init__(
        self,
        tier_levels: List[int] = None,
        leveled_levels: List[int] = None,
        transition_size_mb: int = 1024,
    ):
        """
        Initialize Hybrid Compaction strategy.

        Args:
            tier_levels: Levels to use tiered compaction (default: [0, 1])
            leveled_levels: Levels to use leveled compaction (default: [2+])
            transition_size_mb: Size threshold for transition
        """
        self.tier_levels = tier_levels or [0, 1]
        self.leveled_levels = leveled_levels or list(range(2, 7))
        self.transition_size_mb = transition_size_mb
        self.strategy = CompactionStrategy.HYBRID

        # Initialize sub-strategies
        self.tiered = TieredCompaction(min_threshold=4, max_threshold=32)
        self.leveled = LeveledCompaction(aggressive=False)

        logger.info(
            f"Initialized HybridCompaction: "
            f"tier_levels={tier_levels}, leveled_levels={leveled_levels}"
        )

    def select_packs(self, packs: List[PackStats]) -> List[CompactionTask]:
        """Select packs using hybrid strategy"""
        tasks = []

        # Separate packs by strategy
        tier_packs = [p for p in packs if p.level in self.tier_levels]
        leveled_packs = [p for p in packs if p.level in self.leveled_levels]

        # Apply tiered compaction to lower levels
        if tier_packs:
            tier_tasks = self.tiered.select_packs(tier_packs)
            tasks.extend(tier_tasks)

        # Apply leveled compaction to upper levels
        if leveled_packs:
            leveled_tasks = self.leveled.select_packs(leveled_packs)
            tasks.extend(leveled_tasks)

        return tasks


class AdaptiveCompaction:
    """
    Adaptive Compaction Strategy

    Dynamically switches between strategies based on workload characteristics.

    Features:
    - Monitors workload patterns
    - Switches strategies dynamically
    - Optimizes for current access patterns
    - Self-tuning parameters
    """

    def __init__(self):
        """Initialize Adaptive Compaction"""
        self.strategy = CompactionStrategy.ADAPTIVE
        self.current_strategy = None
        self.workload_history = []

        # Initialize all strategies
        self.strategies = {
            CompactionStrategy.TIME_WINDOWED: TimeWindowedCompaction(),
            CompactionStrategy.LEVELED: LeveledCompaction(),
            CompactionStrategy.TIERED: TieredCompaction(),
            CompactionStrategy.HYBRID: HybridCompaction(),
        }

        logger.info("Initialized AdaptiveCompaction with dynamic strategy selection")

    def select_packs(self, packs: List[PackStats]) -> List[CompactionTask]:
        """Select packs using adaptive strategy"""
        # Analyze workload
        workload = self._analyze_workload(packs)

        # Select best strategy
        best_strategy = self._select_best_strategy(workload)

        # Use selected strategy
        strategy = self.strategies[best_strategy]
        tasks = strategy.select_packs(packs)

        # Track decision
        self.current_strategy = best_strategy
        self.workload_history.append(
            {
                "timestamp": datetime.now(),
                "strategy": best_strategy.value,
                "pack_count": len(packs),
                "workload": workload,
            }
        )

        return tasks

    def _analyze_workload(self, packs: List[PackStats]) -> Dict[str, float]:
        """Analyze workload characteristics"""
        if not packs:
            return {
                "read_write_ratio": 1.0,
                "avg_fragmentation": 0.0,
                "time_series_ratio": 0.0,
                "hot_data_ratio": 0.0,
            }

        total_reads = sum(p.read_qps for p in packs)
        total_writes = sum(p.write_qps for p in packs)

        return {
            "read_write_ratio": total_reads / (total_writes + 1e-6),
            "avg_fragmentation": sum(p.fragmentation_ratio for p in packs) / len(packs),
            "time_series_ratio": sum(
                1 for p in packs if p.domain in ["metrics", "audit"]
            )
            / len(packs),
            "hot_data_ratio": sum(1 for p in packs if not p.is_cold) / len(packs),
        }

    def _select_best_strategy(self, workload: Dict[str, float]) -> CompactionStrategy:
        """Select best strategy for workload"""
        # Time-series workload
        if workload["time_series_ratio"] > 0.7:
            return CompactionStrategy.TIME_WINDOWED

        # Read-heavy workload
        elif workload["read_write_ratio"] > 3.0:
            return CompactionStrategy.LEVELED

        # Write-heavy workload
        elif workload["read_write_ratio"] < 0.5:
            return CompactionStrategy.TIERED

        # Balanced workload
        else:
            return CompactionStrategy.HYBRID


def select_compaction_policy(pack_stats: PackStats) -> object:
    """
    Select appropriate compaction policy based on pack statistics.

    This function analyzes pack characteristics and workload patterns to
    select the optimal compaction strategy.

    Args:
        pack_stats: Statistics for the pack

    Returns:
        Compaction policy instance
    """
    # Calculate read/write ratio
    read_heavy = pack_stats.read_qps / (pack_stats.write_qps + 1e-6) > 3.0
    write_heavy = pack_stats.write_qps / (pack_stats.read_qps + 1e-6) > 3.0

    # Check for time-series data
    time_series = pack_stats.domain in ["metrics", "audit", "logs", "events"]

    # Check for high fragmentation
    high_fragmentation = pack_stats.fragmentation_ratio > 0.4

    # Select strategy
    if time_series:
        logger.info(f"Selected TimeWindowedCompaction for domain: {pack_stats.domain}")
        return TimeWindowedCompaction()

    elif read_heavy or high_fragmentation:
        aggressive = high_fragmentation
        logger.info(f"Selected LeveledCompaction (aggressive={aggressive})")
        return LeveledCompaction(aggressive=aggressive)

    elif write_heavy:
        logger.info("Selected TieredCompaction for write-heavy workload")
        return TieredCompaction()

    else:
        logger.info("Selected HybridCompaction for balanced workload")
        return HybridCompaction(tier_levels=[0, 1], leveled=[2])


class CompactionCostModel:
    """
    Cost model for estimating compaction costs.

    Estimates I/O cost, CPU cost, and total cost for compaction operations.
    """

    def __init__(self, io_cost_per_mb: float = 1.0, cpu_cost_per_mb: float = 0.1):
        """
        Initialize cost model.

        Args:
            io_cost_per_mb: Cost of I/O per MB
            cpu_cost_per_mb: Cost of CPU per MB
        """
        self.io_cost_per_mb = io_cost_per_mb
        self.cpu_cost_per_mb = cpu_cost_per_mb

    def estimate_task_cost(
        self, task: CompactionTask, packs: Dict[str, PackStats]
    ) -> float:
        """
        Estimate total cost of a compaction task.

        Args:
            task: Compaction task
            packs: Map of pack ID to pack stats

        Returns:
            Estimated cost
        """
        # Get pack statistics
        task_packs = [packs[pid] for pid in task.input_packs if pid in packs]

        if not task_packs:
            return 0.0

        # Calculate I/O cost
        total_bytes = sum(p.total_bytes for p in task_packs)
        total_live = sum(p.live_bytes for p in task_packs)

        io_mb = (total_bytes + total_live) / (1024 * 1024)
        io_cost = io_mb * self.io_cost_per_mb

        # Calculate CPU cost (for merge/sort)
        cpu_mb = total_live / (1024 * 1024)
        cpu_cost = cpu_mb * self.cpu_cost_per_mb

        return io_cost + cpu_cost

    def estimate_write_amplification(
        self, packs: List[PackStats], strategy: CompactionStrategy
    ) -> float:
        """
        Estimate write amplification for a strategy.

        Args:
            packs: List of pack statistics
            strategy: Compaction strategy

        Returns:
            Estimated write amplification factor
        """
        # Strategy-specific amplification factors
        amplification_map = {
            CompactionStrategy.TIME_WINDOWED: 2.0,
            CompactionStrategy.LEVELED: 10.0,
            CompactionStrategy.TIERED: 5.0,
            CompactionStrategy.HYBRID: 7.0,
            CompactionStrategy.ADAPTIVE: 6.0,
        }

        return amplification_map.get(strategy, 5.0)


class CompactionPlanner:
    """
    Compaction Planner for LSM-tree Storage.

    High-level planner that coordinates compaction across multiple strategies,
    selects compaction candidates based on various criteria, and manages
    compaction scheduling with resource constraints.

    This class acts as the orchestration layer above individual compaction
    strategies, making decisions about:
    - Which packs need compaction most urgently
    - Resource allocation for compaction tasks
    - Prioritization when multiple compactions are possible
    - Throttling based on fragmentation and amplification limits

    Example:
        planner = CompactionPlanner(
            strategy='leveled',
            fragmentation_threshold=0.3,
            size_amplification_limit=10.0
        )

        candidates = planner.select_compaction_candidates(packs)
        for pack_id in candidates:
            schedule_compaction(pack_id)
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        fragmentation_threshold: float = 0.25,
        size_amplification_limit: float = 10.0,
        max_concurrent_compactions: int = 2,
        enable_throttling: bool = True,
    ):
        """
        Initialize Compaction Planner.

        Args:
            strategy: Compaction strategy name ('time_windowed', 'leveled',
                     'tiered', 'hybrid', or 'adaptive')
            fragmentation_threshold: Trigger compaction when fragmentation
                                    exceeds this ratio (0-1)
            size_amplification_limit: Maximum allowed size amplification factor
            max_concurrent_compactions: Maximum number of concurrent compactions
            enable_throttling: Whether to throttle compaction based on I/O load
        """
        self.fragmentation_threshold = fragmentation_threshold
        self.size_amplification_limit = size_amplification_limit
        self.max_concurrent_compactions = max_concurrent_compactions
        self.enable_throttling = enable_throttling

        # Initialize strategy
        strategy_map = {
            "time_windowed": TimeWindowedCompaction,
            "leveled": LeveledCompaction,
            "tiered": TieredCompaction,
            "hybrid": HybridCompaction,
            "adaptive": AdaptiveCompaction,
        }

        strategy_lower = strategy.lower()
        if strategy_lower not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Must be one of {list(strategy_map.keys())}"
            )

        strategy_class = strategy_map[strategy_lower]
        self.strategy_name = strategy_lower
        self.strategy = strategy_class()

        # Cost model for estimation
        self.cost_model = CompactionCostModel()

        # Tracking
        self.active_compactions: Set[str] = set()
        self.compaction_history: List[CompactionTask] = []
        self.total_compactions_planned = 0

        logger.info(
            f"Initialized CompactionPlanner: "
            f"strategy={strategy}, "
            f"frag_threshold={fragmentation_threshold:.2f}, "
            f"size_amp_limit={size_amplification_limit:.1f}"
        )

    def select_compaction_candidates(self, packs: List[PackStats]) -> List[str]:
        """
        Select pack IDs that are candidates for compaction.

        This method analyzes pack statistics and selects packs that should
        be compacted based on fragmentation, size amplification, and
        strategy-specific criteria.

        Args:
            packs: List of pack statistics

        Returns:
            List of pack IDs to compact (in priority order)

        Example:
            >>> packs = [
            ...     PackStats(pack_id="p1", fragmentation_ratio=0.4, ...),
            ...     PackStats(pack_id="p2", fragmentation_ratio=0.1, ...),
            ... ]
            >>> planner = CompactionPlanner()
            >>> candidates = planner.select_compaction_candidates(packs)
            >>> print(f"Should compact: {candidates}")
        """
        if not packs:
            logger.debug("No packs to evaluate for compaction")
            return []

        logger.debug(f"Evaluating {len(packs)} packs for compaction")

        # Filter packs that definitely need compaction (high fragmentation)
        urgent_packs = []
        for pack in packs:
            if pack.fragmentation_ratio >= self.fragmentation_threshold:
                urgent_packs.append(pack.pack_id)
                logger.info(
                    f"Urgent compaction needed for {pack.pack_id}: "
                    f"fragmentation={pack.fragmentation_ratio:.2%}"
                )

        # Check size amplification
        total_live = sum(p.live_bytes for p in packs)
        total_size = sum(p.total_bytes for p in packs)
        if total_live > 0:
            current_amplification = total_size / total_live

            if current_amplification > self.size_amplification_limit:
                logger.warning(
                    f"Size amplification {current_amplification:.2f}x "
                    f"exceeds limit {self.size_amplification_limit:.2f}x"
                )
                # Add all packs with high dead bytes ratio
                for pack in packs:
                    if pack.pack_id not in urgent_packs:
                        dead_ratio = (
                            pack.dead_bytes / pack.total_bytes
                            if pack.total_bytes > 0
                            else 0
                        )
                        if dead_ratio > 0.2:  # More than 20% dead space
                            urgent_packs.append(pack.pack_id)

        # Get strategy-specific compaction tasks
        tasks = self.strategy.select_packs(packs)

        # Extract pack IDs from tasks, prioritizing by task priority
        strategy_candidates = []

        # Sort tasks by priority
        priority_order = {
            CompactionPriority.CRITICAL: 0,
            CompactionPriority.HIGH: 1,
            CompactionPriority.MEDIUM: 2,
            CompactionPriority.LOW: 3,
            CompactionPriority.IDLE: 4,
        }

        sorted_tasks = sorted(tasks, key=lambda t: priority_order.get(t.priority, 99))

        for task in sorted_tasks:
            strategy_candidates.extend(task.input_packs)

            # Track task for history
            self.compaction_history.append(task)
            self.total_compactions_planned += 1

        # Combine urgent and strategy candidates, removing duplicates
        all_candidates = []
        seen = set()

        # Urgent first
        for pack_id in urgent_packs:
            if pack_id not in seen:
                all_candidates.append(pack_id)
                seen.add(pack_id)

        # Then strategy candidates
        for pack_id in strategy_candidates:
            if pack_id not in seen:
                all_candidates.append(pack_id)
                seen.add(pack_id)

        # Apply throttling if enabled
        if self.enable_throttling:
            # Limit based on concurrent compactions
            available_slots = self.max_concurrent_compactions - len(
                self.active_compactions
            )
            if available_slots <= 0:
                logger.info(
                    f"Throttling compaction: {len(self.active_compactions)} "
                    f"active (max: {self.max_concurrent_compactions})"
                )
                all_candidates = []
            elif len(all_candidates) > available_slots:
                logger.info(
                    f"Limiting candidates to {available_slots} due to throttling"
                )
                all_candidates = all_candidates[:available_slots]

        logger.info(
            f"Selected {len(all_candidates)} compaction candidates "
            f"(urgent: {len(urgent_packs)}, strategy: {len(strategy_candidates)})"
        )

        return all_candidates

    def mark_compaction_started(self, pack_id: str) -> None:
        """Mark a pack as being compacted"""
        self.active_compactions.add(pack_id)
        logger.debug(
            f"Marked {pack_id} as compacting (active: {len(self.active_compactions)})"
        )

    def mark_compaction_finished(self, pack_id: str) -> None:
        """Mark a pack compaction as finished"""
        if pack_id in self.active_compactions:
            self.active_compactions.discard(pack_id)
            logger.debug(
                f"Marked {pack_id} as finished (active: {len(self.active_compactions)})"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            "strategy": self.strategy_name,
            "fragmentation_threshold": self.fragmentation_threshold,
            "size_amplification_limit": self.size_amplification_limit,
            "max_concurrent_compactions": self.max_concurrent_compactions,
            "active_compactions": len(self.active_compactions),
            "total_planned": self.total_compactions_planned,
            "history_size": len(self.compaction_history),
        }

    def estimate_compaction_benefit(self, packs: List[PackStats]) -> Dict[str, float]:
        """
        Estimate the benefit of compacting given packs.

        Args:
            packs: List of pack statistics

        Returns:
            Dictionary with benefit metrics:
                - space_reclaimed_mb: Expected space to reclaim
                - fragmentation_improvement: Expected fragmentation reduction
                - estimated_cost: Estimated I/O cost
        """
        if not packs:
            return {
                "space_reclaimed_mb": 0.0,
                "fragmentation_improvement": 0.0,
                "estimated_cost": 0.0,
            }

        # Calculate space to reclaim
        total_dead = sum(p.dead_bytes for p in packs)
        space_reclaimed_mb = total_dead / (1024 * 1024)

        # Calculate fragmentation improvement
        total_bytes = sum(p.total_bytes for p in packs)
        current_frag = (
            sum(p.fragmentation_ratio * p.total_bytes for p in packs) / total_bytes
            if total_bytes > 0
            else 0
        )
        expected_frag_after = 0.05  # Assume ~5% fragmentation after compaction
        fragmentation_improvement = current_frag - expected_frag_after

        # Estimate I/O cost
        total_live = sum(p.live_bytes for p in packs)
        # Cost = read all data + write live data
        io_mb = (total_bytes + total_live) / (1024 * 1024)

        return {
            "space_reclaimed_mb": space_reclaimed_mb,
            "fragmentation_improvement": max(0, fragmentation_improvement),
            "estimated_cost": io_mb,
        }


if __name__ == "__main__":
    # Example usage and testing

    print("=" * 80)
    print(" " * 20 + "COMPACTION POLICY SYSTEM TEST")
    print("=" * 80)
    print()

    # Create sample pack statistics
    packs = [
        PackStats(
            pack_id="pack001",
            read_qps=100.0,
            write_qps=10.0,
            domain="user_data",
            live_bytes=80 * 1024 * 1024,
            total_bytes=100 * 1024 * 1024,
            fragmentation_ratio=0.20,
            level=0,
        ),
        PackStats(
            pack_id="pack002",
            read_qps=50.0,
            write_qps=5.0,
            domain="metrics",
            live_bytes=60 * 1024 * 1024,
            total_bytes=100 * 1024 * 1024,
            fragmentation_ratio=0.40,
            level=0,
        ),
        PackStats(
            pack_id="pack003",
            read_qps=200.0,
            write_qps=50.0,
            domain="user_data",
            live_bytes=90 * 1024 * 1024,
            total_bytes=100 * 1024 * 1024,
            fragmentation_ratio=0.10,
            level=1,
        ),
    ]

    print("Sample Packs:")
    for pack in packs:
        print(
            f"  {pack.pack_id}: {pack.domain}, "
            f"level={pack.level}, "
            f"frag={pack.fragmentation_ratio:.2f}, "
            f"util={pack.utilization:.2f}"
        )
    print()

    # Test strategy selection
    print("Testing Strategy Selection:")
    for pack in packs:
        policy = select_compaction_policy(pack)
        print(f"  {pack.pack_id} -> {policy.__class__.__name__}")
    print()

    # Test each strategy
    print("Testing Compaction Strategies:")

    strategies = [
        ("Time-Windowed", TimeWindowedCompaction()),
        ("Leveled", LeveledCompaction()),
        ("Tiered", TieredCompaction()),
        ("Hybrid", HybridCompaction()),
        ("Adaptive", AdaptiveCompaction()),
    ]

    for name, strategy in strategies:
        print(f"\n{name} Compaction:")
        tasks = strategy.select_packs(packs)
        print(f"  Generated {len(tasks)} tasks")

        for task in tasks:
            print(f"    Task: {task.task_id}")
            print(f"      Input packs: {len(task.input_packs)}")
            print(f"      Priority: {task.priority.value}")
            print(f"      Estimated cost: {task.estimated_cost:.2f} MB")
