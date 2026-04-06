"""
Adaptive Repacking System

This module provides intelligent pack consolidation and repacking capabilities
with support for fragmentation reduction, artifact management, and cost optimization.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RepackReason(Enum):
    """Reasons for triggering a repack operation"""

    FRAGMENTATION = "fragmentation"
    LOW_UTILIZATION = "low_utilization"
    ARTIFACT_CONSOLIDATION = "artifact_consolidation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class RepackPriority(Enum):
    """Priority levels for repack operations"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    IDLE = 5


@dataclass
class PackMetadata:
    """
    Metadata for a pack file

    Attributes:
        pack_id: Unique pack identifier
        artifact_id: Parent artifact identifier
        live_bytes: Number of live bytes
        total_bytes: Total size in bytes
        fragmentation_ratio: Ratio of dead to total bytes
        creation_time: When pack was created
        last_modified: Last modification time
        access_count: Number of accesses
        chunk_count: Number of chunks in pack
        compression_ratio: Compression ratio achieved
    """

    pack_id: str
    artifact_id: str
    live_bytes: int
    total_bytes: int
    fragmentation_ratio: float
    creation_time: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    chunk_count: int = 0
    compression_ratio: float = 1.0

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
    def age_hours(self) -> float:
        """Get pack age in hours"""
        return (datetime.now() - self.creation_time).total_seconds() / 3600

    @property
    def is_hot(self) -> bool:
        """Check if pack is hot (frequently accessed)"""
        return self.access_count > 100 and self.age_hours < 24

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pack_id": self.pack_id,
            "artifact_id": self.artifact_id,
            "live_bytes": self.live_bytes,
            "total_bytes": self.total_bytes,
            "fragmentation_ratio": self.fragmentation_ratio,
            "utilization": self.utilization,
            "dead_bytes": self.dead_bytes,
            "creation_time": self.creation_time.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "access_count": self.access_count,
            "chunk_count": self.chunk_count,
            "compression_ratio": self.compression_ratio,
            "age_hours": self.age_hours,
        }


@dataclass
class ArtifactInfo:
    """
    Information about an artifact (collection of packs)

    Attributes:
        artifact_id: Unique artifact identifier
        pack_ids: List of pack IDs in this artifact
        total_live_bytes: Total live bytes across all packs
        total_bytes: Total bytes across all packs
        pack_count: Number of packs
        avg_fragmentation: Average fragmentation ratio
    """

    artifact_id: str
    pack_ids: List[str]
    total_live_bytes: int
    total_bytes: int

    @property
    def pack_count(self) -> int:
        """Get number of packs"""
        return len(self.pack_ids)

    @property
    def avg_fragmentation(self) -> float:
        """Calculate average fragmentation"""
        if self.total_bytes == 0:
            return 0.0
        return 1.0 - (self.total_live_bytes / self.total_bytes)

    @property
    def total_dead_bytes(self) -> int:
        """Calculate total dead bytes"""
        return self.total_bytes - self.total_live_bytes

    @property
    def utilization(self) -> float:
        """Calculate overall utilization"""
        if self.total_bytes == 0:
            return 0.0
        return self.total_live_bytes / self.total_bytes


@dataclass
class RepackTask:
    """
    Represents a repack operation

    Attributes:
        task_id: Unique task identifier
        artifact_id: Artifact being repacked
        input_packs: List of input pack IDs
        reason: Reason for repacking
        priority: Task priority
        estimated_savings_bytes: Estimated space savings
        estimated_io_bytes: Estimated I/O required
        estimated_duration_seconds: Estimated duration
        created_at: Task creation time
        metadata: Additional metadata
    """

    task_id: str
    artifact_id: str
    input_packs: List[str]
    reason: RepackReason
    priority: RepackPriority = RepackPriority.MEDIUM
    estimated_savings_bytes: int = 0
    estimated_io_bytes: int = 0
    estimated_duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def savings_ratio(self) -> float:
        """Calculate savings to I/O ratio"""
        if self.estimated_io_bytes == 0:
            return 0.0
        return self.estimated_savings_bytes / self.estimated_io_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "input_packs": self.input_packs,
            "reason": self.reason.value,
            "priority": self.priority.value,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "estimated_savings_mb": self.estimated_savings_bytes / (1024 * 1024),
            "estimated_io_bytes": self.estimated_io_bytes,
            "estimated_io_mb": self.estimated_io_bytes / (1024 * 1024),
            "savings_ratio": self.savings_ratio,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RepackResult:
    """
    Result of a repack operation

    Attributes:
        task_id: Task identifier
        artifact_id: Artifact that was repacked
        input_packs: Input pack IDs
        output_packs: Output pack IDs
        bytes_read: Bytes read during repack
        bytes_written: Bytes written during repack
        bytes_saved: Space saved
        duration_seconds: Duration in seconds
        success: Whether repack succeeded
        error: Error message if failed
    """

    task_id: str
    artifact_id: str
    input_packs: List[str]
    output_packs: List[str]
    bytes_read: int
    bytes_written: int
    bytes_saved: int
    duration_seconds: float
    success: bool = True
    error: Optional[str] = None

    @property
    def space_amplification(self) -> float:
        """Calculate space amplification"""
        if self.bytes_saved == 0:
            return 1.0
        return self.bytes_read / (self.bytes_read - self.bytes_saved)

    @property
    def throughput_mbps(self) -> float:
        """Calculate I/O throughput in MB/s"""
        if self.duration_seconds == 0:
            return 0.0
        total_io = (self.bytes_read + self.bytes_written) / (1024 * 1024)
        return total_io / self.duration_seconds


@dataclass
class RepackConfig:
    """
    Configuration for repack operations

    Attributes:
        fragmentation_threshold: Trigger repack above this fragmentation
        utilization_threshold: Trigger repack below this utilization
        max_packs_per_artifact: Maximum packs per artifact
        min_pack_size_mb: Minimum pack size in MB
        max_pack_size_mb: Maximum pack size in MB
        target_pack_count: Target number of packs after repack
        repack_window_hours: Time window for repack scheduling
        cost_benefit_threshold: Minimum benefit/cost ratio
        enable_emergency_repack: Enable emergency repacking
        parallel_repacks: Number of parallel repack operations
    """

    fragmentation_threshold: float = 0.40
    utilization_threshold: float = 0.65
    max_packs_per_artifact: int = 8
    min_pack_size_mb: int = 64
    max_pack_size_mb: int = 1024
    target_pack_count: int = 4
    repack_window_hours: int = 24
    cost_benefit_threshold: float = 0.3
    enable_emergency_repack: bool = True
    parallel_repacks: int = 2

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []

        if not 0 < self.fragmentation_threshold <= 1:
            errors.append("fragmentation_threshold must be in (0, 1]")

        if not 0 < self.utilization_threshold <= 1:
            errors.append("utilization_threshold must be in (0, 1]")

        if self.max_packs_per_artifact < 1:
            errors.append("max_packs_per_artifact must be positive")

        if self.min_pack_size_mb >= self.max_pack_size_mb:
            errors.append("min_pack_size_mb must be less than max_pack_size_mb")

        return errors


class RepackScheduler:
    """
    Scheduler for repack operations with priority queue and throttling.

    Features:
    - Priority-based scheduling
    - Throttling to limit concurrent operations
    - Cost-benefit analysis
    - Resource management
    """

    def __init__(self, config: RepackConfig):
        """
        Initialize repack scheduler.

        Args:
            config: Repack configuration
        """
        self.config = config
        self.pending_tasks: List[Tuple[int, RepackTask]] = []  # Priority queue
        self.active_tasks: Dict[str, RepackTask] = {}
        self.completed_tasks: List[RepackResult] = []
        self.task_counter = 0

        logger.info(
            f"Initialized RepackScheduler with max_packs={config.max_packs_per_artifact}"
        )

    def schedule_task(self, task: RepackTask) -> None:
        """
        Schedule a repack task.

        Args:
            task: Repack task to schedule
        """
        # Add to priority queue (lower priority value = higher priority)
        heapq.heappush(
            self.pending_tasks, (task.priority.value, self.task_counter, task)
        )
        self.task_counter += 1

        logger.info(
            f"Scheduled repack task: {task.task_id} "
            f"(priority={task.priority.value}, "
            f"savings={task.estimated_savings_bytes / (1024 * 1024):.2f} MB)"
        )

    def get_next_task(self) -> Optional[RepackTask]:
        """
        Get next task to execute.

        Returns:
            Next repack task or None if no tasks available
        """
        # Check if we have capacity
        if len(self.active_tasks) >= self.config.parallel_repacks:
            return None

        # Get highest priority task
        if self.pending_tasks:
            _, _, task = heapq.heappop(self.pending_tasks)
            self.active_tasks[task.task_id] = task
            return task

        return None

    def complete_task(self, result: RepackResult) -> None:
        """
        Mark a task as completed.

        Args:
            result: Repack result
        """
        if result.task_id in self.active_tasks:
            del self.active_tasks[result.task_id]

        self.completed_tasks.append(result)

        logger.info(
            f"Completed repack task: {result.task_id} "
            f"(saved={result.bytes_saved / (1024 * 1024):.2f} MB, "
            f"duration={result.duration_seconds:.2f}s)"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        total_saved = sum(r.bytes_saved for r in self.completed_tasks if r.success)
        total_io = sum(
            r.bytes_read + r.bytes_written for r in self.completed_tasks if r.success
        )

        return {
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_bytes_saved": total_saved,
            "total_bytes_saved_mb": total_saved / (1024 * 1024),
            "total_io_bytes": total_io,
            "total_io_mb": total_io / (1024 * 1024),
            "success_count": sum(1 for r in self.completed_tasks if r.success),
            "failure_count": sum(1 for r in self.completed_tasks if not r.success),
        }


class AdaptiveRepacker:
    """
    Adaptive repacking engine with intelligent pack selection and consolidation.

    Features:
    - Automatic fragmentation detection
    - Artifact-aware pack consolidation
    - Cost-benefit analysis
    - Adaptive thresholds
    - Priority-based scheduling
    """

    def __init__(self, config: Optional[RepackConfig] = None):
        """
        Initialize adaptive repacker.

        Args:
            config: Optional repack configuration
        """
        self.config = config or RepackConfig()
        self.scheduler = RepackScheduler(self.config)
        self.pack_cache: Dict[str, PackMetadata] = {}
        self.artifact_cache: Dict[str, ArtifactInfo] = {}
        self.repack_history: List[RepackResult] = []

        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")

        logger.info("Initialized AdaptiveRepacker")

    def analyze_pack(self, pack: PackMetadata) -> Tuple[bool, Optional[RepackReason]]:
        """
        Analyze if a pack needs repacking.

        Args:
            pack: Pack metadata to analyze

        Returns:
            Tuple of (needs_repack, reason)
        """
        # Check fragmentation
        if pack.fragmentation_ratio > self.config.fragmentation_threshold:
            return True, RepackReason.FRAGMENTATION

        # Check utilization
        if pack.utilization < self.config.utilization_threshold:
            return True, RepackReason.LOW_UTILIZATION

        return False, None

    def analyze_artifact(
        self, artifact: ArtifactInfo, packs: Dict[str, PackMetadata]
    ) -> Tuple[bool, Optional[RepackReason]]:
        """
        Analyze if an artifact needs repacking.

        Args:
            artifact: Artifact information
            packs: Map of pack ID to metadata

        Returns:
            Tuple of (needs_repack, reason)
        """
        # Check pack count
        if artifact.pack_count > self.config.max_packs_per_artifact:
            return True, RepackReason.ARTIFACT_CONSOLIDATION

        # Check overall fragmentation
        if artifact.avg_fragmentation > self.config.fragmentation_threshold:
            return True, RepackReason.FRAGMENTATION

        # Check overall utilization
        if artifact.utilization < self.config.utilization_threshold:
            return True, RepackReason.LOW_UTILIZATION

        return False, None

    def create_repack_task(
        self,
        artifact_id: str,
        pack_ids: List[str],
        reason: RepackReason,
        packs: Dict[str, PackMetadata],
    ) -> RepackTask:
        """
        Create a repack task for the given packs.

        Args:
            artifact_id: Artifact identifier
            pack_ids: List of pack IDs to repack
            reason: Reason for repacking
            packs: Map of pack ID to metadata

        Returns:
            RepackTask
        """
        # Calculate estimates
        task_packs = [packs[pid] for pid in pack_ids if pid in packs]

        total_bytes = sum(p.total_bytes for p in task_packs)
        total_live = sum(p.live_bytes for p in task_packs)
        total_dead = total_bytes - total_live

        # Estimate I/O (read all, write live)
        estimated_io = total_bytes + total_live

        # Estimate duration (10 MB/s throughput)
        estimated_duration = estimated_io / (10 * 1024 * 1024)

        # Determine priority
        priority = self._determine_priority(reason, task_packs)

        task_id = f"repack_{artifact_id}_{datetime.now().timestamp()}"

        task = RepackTask(
            task_id=task_id,
            artifact_id=artifact_id,
            input_packs=pack_ids,
            reason=reason,
            priority=priority,
            estimated_savings_bytes=total_dead,
            estimated_io_bytes=estimated_io,
            estimated_duration_seconds=estimated_duration,
            metadata={
                "pack_count": len(pack_ids),
                "total_bytes": total_bytes,
                "total_live": total_live,
                "avg_fragmentation": sum(p.fragmentation_ratio for p in task_packs)
                / len(task_packs),
            },
        )

        return task

    def _determine_priority(
        self, reason: RepackReason, packs: List[PackMetadata]
    ) -> RepackPriority:
        """Determine priority for repack task"""
        avg_frag = (
            sum(p.fragmentation_ratio for p in packs) / len(packs) if packs else 0.0
        )

        if reason == RepackReason.EMERGENCY:
            return RepackPriority.CRITICAL

        elif avg_frag > 0.7 or len(packs) > self.config.max_packs_per_artifact * 1.5:
            return RepackPriority.HIGH

        elif avg_frag > 0.5 or len(packs) > self.config.max_packs_per_artifact:
            return RepackPriority.MEDIUM

        else:
            return RepackPriority.LOW

    def select_packs_for_repack(
        self, artifact_id: str, packs: Dict[str, PackMetadata]
    ) -> List[str]:
        """
        Select packs to include in repack operation.

        Uses intelligent selection to minimize I/O while maximizing benefit.

        Args:
            artifact_id: Artifact identifier
            packs: Map of pack ID to metadata

        Returns:
            List of pack IDs to repack
        """
        # Get artifact packs
        artifact_packs = {
            pid: p for pid, p in packs.items() if p.artifact_id == artifact_id
        }

        if not artifact_packs:
            return []

        # Sort by fragmentation (worst first)
        sorted_packs = sorted(
            artifact_packs.items(), key=lambda x: x[1].fragmentation_ratio, reverse=True
        )

        # Select packs based on strategy
        selected = []
        total_bytes = 0
        max_bytes = self.config.max_pack_size_mb * 1024 * 1024

        for pack_id, pack in sorted_packs:
            # Include high-fragmentation packs
            if pack.fragmentation_ratio > self.config.fragmentation_threshold:
                selected.append(pack_id)
                total_bytes += pack.total_bytes

            # Stop if we have enough
            if len(selected) >= self.config.max_packs_per_artifact:
                break

            # Stop if size limit reached
            if total_bytes >= max_bytes:
                break

        return selected

    def calculate_cost_benefit(self, task: RepackTask) -> float:
        """
        Calculate cost-benefit ratio for a repack task.

        Higher values indicate better cost-benefit.

        Args:
            task: Repack task

        Returns:
            Cost-benefit ratio
        """
        if task.estimated_io_bytes == 0:
            return 0.0

        # Benefit is space saved
        benefit = task.estimated_savings_bytes

        # Cost is I/O bytes
        cost = task.estimated_io_bytes

        return benefit / cost

    def should_repack(self, task: RepackTask) -> bool:
        """
        Determine if repack should proceed based on cost-benefit.

        Args:
            task: Repack task

        Returns:
            True if should repack
        """
        cost_benefit = self.calculate_cost_benefit(task)

        # Emergency repacks always proceed
        if task.reason == RepackReason.EMERGENCY:
            return True

        # Manual repacks always proceed
        if task.reason == RepackReason.MANUAL:
            return True

        # Check cost-benefit threshold
        return cost_benefit >= self.config.cost_benefit_threshold


def adaptive_repack(
    pack_id: str,
    packs: Optional[Dict[str, PackMetadata]] = None,
    config: Optional[RepackConfig] = None,
) -> Dict[str, Any]:
    """
    Trigger adaptive repack for a pack or artifact.

    This function analyzes pack statistics and triggers repacking when:
    - live_bytes/total_bytes < 0.65 (low utilization)
    - fragmentation_ratio > 0.40 (high fragmentation)
    - Enforces max packs_per_artifact = 8

    Args:
        pack_id: Pack or artifact identifier
        packs: Optional map of pack metadata (for testing)
        config: Optional repack configuration

    Returns:
        Dictionary with repack result
    """
    # Initialize repacker
    repacker = AdaptiveRepacker(config)

    # In production, would load pack metadata from storage
    if packs is None:
        # Simulate loading pack metadata
        logger.warning("No pack metadata provided, using simulation")
        packs = _simulate_pack_metadata(pack_id)

    # Get pack metadata
    if pack_id not in packs:
        logger.error(f"Pack not found: {pack_id}")
        return {
            "repacked": False,
            "reason": "pack_not_found",
            "error": f"Pack {pack_id} not found",
        }

    pack = packs[pack_id]

    # Analyze pack
    needs_repack, reason = repacker.analyze_pack(pack)

    if not needs_repack:
        logger.info(f"Pack {pack_id} does not need repacking")
        return {
            "repacked": False,
            "reason": "no_repack_needed",
            "pack_id": pack_id,
            "fragmentation_ratio": pack.fragmentation_ratio,
            "utilization": pack.utilization,
        }

    # Get artifact info
    artifact_id = pack.artifact_id
    artifact_packs = {
        pid: p for pid, p in packs.items() if p.artifact_id == artifact_id
    }

    # Check artifact constraints
    artifact = ArtifactInfo(
        artifact_id=artifact_id,
        pack_ids=list(artifact_packs.keys()),
        total_live_bytes=sum(p.live_bytes for p in artifact_packs.values()),
        total_bytes=sum(p.total_bytes for p in artifact_packs.values()),
    )

    # Analyze artifact
    artifact_needs_repack, artifact_reason = repacker.analyze_artifact(artifact, packs)

    if artifact_needs_repack and artifact_reason == RepackReason.ARTIFACT_CONSOLIDATION:
        reason = artifact_reason

    # Select packs for repack
    selected_packs = repacker.select_packs_for_repack(artifact_id, packs)

    if not selected_packs:
        logger.warning(f"No packs selected for repack in artifact {artifact_id}")
        return {
            "repacked": False,
            "reason": "no_packs_selected",
            "artifact_id": artifact_id,
        }

    # Create repack task
    task = repacker.create_repack_task(artifact_id, selected_packs, reason, packs)

    # Check cost-benefit
    if not repacker.should_repack(task):
        logger.info(f"Repack not cost-effective for {artifact_id}")
        return {
            "repacked": False,
            "reason": "cost_benefit_too_low",
            "task": task.to_dict(),
            "cost_benefit": repacker.calculate_cost_benefit(task),
        }

    # Schedule repack
    repacker.scheduler.schedule_task(task)

    logger.info(f"Scheduled repack for artifact {artifact_id}: {reason.value}")

    return {
        "repacked": True,
        "task_id": task.task_id,
        "artifact_id": artifact_id,
        "reason": reason.value,
        "input_packs": selected_packs,
        "pack_count": len(selected_packs),
        "estimated_savings_mb": task.estimated_savings_bytes / (1024 * 1024),
        "estimated_io_mb": task.estimated_io_bytes / (1024 * 1024),
        "priority": task.priority.value,
        "cost_benefit": repacker.calculate_cost_benefit(task),
        "task": task.to_dict(),
    }


def _simulate_pack_metadata(pack_id: str) -> Dict[str, PackMetadata]:
    """Simulate pack metadata for testing"""
    import random

    artifact_id = f"artifact_{pack_id.split('_')[0]}"

    packs = {}
    for i in range(5):
        pid = f"{artifact_id}_pack{i}"
        total_bytes = random.randint(50, 150) * 1024 * 1024
        live_bytes = int(total_bytes * random.uniform(0.5, 0.9))

        packs[pid] = PackMetadata(
            pack_id=pid,
            artifact_id=artifact_id,
            live_bytes=live_bytes,
            total_bytes=total_bytes,
            fragmentation_ratio=(total_bytes - live_bytes) / total_bytes,
            chunk_count=random.randint(100, 1000),
            access_count=random.randint(0, 200),
        )

    return packs


class RepackMonitor:
    """
    Monitor and report on repack operations.

    Tracks statistics, generates reports, and provides insights.
    """

    def __init__(self):
        """Initialize repack monitor"""
        self.repacks: List[RepackResult] = []
        self.active_tasks: Dict[str, RepackTask] = {}

    def record_repack(self, result: RepackResult) -> None:
        """Record a repack result"""
        self.repacks.append(result)

        if result.task_id in self.active_tasks:
            del self.active_tasks[result.task_id]

    def get_statistics(self, window_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Get repack statistics.

        Args:
            window_hours: Optional time window in hours

        Returns:
            Statistics dictionary
        """
        # Filter by time window if specified
        repacks = self.repacks
        if window_hours:
            cutoff = datetime.now() - timedelta(hours=window_hours)
            # In production, would filter by timestamp
            # repacks = list(repacks if r.timestamp > cutoff)

        if not repacks:
            return {
                "count": 0,
                "total_saved_mb": 0.0,
                "total_io_mb": 0.0,
                "success_rate": 0.0,
            }

        total_saved = sum(r.bytes_saved for r in repacks)
        total_io = sum(r.bytes_read + r.bytes_written for r in repacks)
        success_count = sum(1 for r in repacks if r.success)

        return {
            "count": len(repacks),
            "success_count": success_count,
            "failure_count": len(repacks) - success_count,
            "success_rate": success_count / len(repacks),
            "total_saved_bytes": total_saved,
            "total_saved_mb": total_saved / (1024 * 1024),
            "total_saved_gb": total_saved / (1024 * 1024 * 1024),
            "total_io_bytes": total_io,
            "total_io_mb": total_io / (1024 * 1024),
            "avg_throughput_mbps": sum(r.throughput_mbps for r in repacks)
            / len(repacks),
            "total_duration_hours": sum(r.duration_seconds for r in repacks) / 3600,
        }

    def generate_report(self) -> str:
        """Generate comprehensive repack report"""
        stats = self.get_statistics()

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        REPACK OPERATIONS REPORT                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

SUMMARY
=======
Total Repacks: {stats["count"]}
Successful: {stats["success_count"]}
Failed: {stats["failure_count"]}
Success Rate: {stats["success_rate"]:.2%}

SPACE RECLAIMED
===============
Total Saved: {stats["total_saved_gb"]:.2f} GB
Total I/O: {stats["total_io_mb"]:.2f} MB

PERFORMANCE
===========
Average Throughput: {stats["avg_throughput_mbps"]:.2f} MB/s
Total Duration: {stats["total_duration_hours"]:.2f} hours

ACTIVE TASKS
============
Currently Active: {len(self.active_tasks)}
"""

        if self.active_tasks:
            report += "\nActive Tasks:\n"
            for task_id, task in self.active_tasks.items():
                report += f"  - {task_id}: {task.artifact_id} ({task.priority.name})\n"

        return report


if __name__ == "__main__":
    # Example usage and testing

    print("=" * 80)
    print(" " * 20 + "ADAPTIVE REPACK SYSTEM TEST")
    print("=" * 80)
    print()

    # Test adaptive repack
    print("Testing adaptive_repack():")
    result = adaptive_repack("artifact_test_pack0")

    print("\nRepack Result:")
    import json

    print(json.dumps(result, indent=2, default=str))

    # Test with custom config
    print("\n" + "=" * 80)
    print("Testing with custom configuration:")

    config = RepackConfig(
        fragmentation_threshold=0.30,
        utilization_threshold=0.70,
        max_packs_per_artifact=6,
        cost_benefit_threshold=0.2,
    )

    result = adaptive_repack("artifact_test2_pack0", config=config)
    print(json.dumps(result, indent=2, default=str))

    # Test repack monitor
    print("\n" + "=" * 80)
    print("Testing RepackMonitor:")

    monitor = RepackMonitor()

    # Simulate some repack results
    for i in range(5):
        result = RepackResult(
            task_id=f"task{i}",
            artifact_id=f"artifact{i}",
            input_packs=[f"pack{j}" for j in range(3)],
            output_packs=[f"new_pack{i}"],
            bytes_read=100 * 1024 * 1024,
            bytes_written=80 * 1024 * 1024,
            bytes_saved=20 * 1024 * 1024,
            duration_seconds=10.0,
        )
        monitor.record_repack(result)

    print(monitor.generate_report())
