"""
Service Level Indicators (SLI) and Service Level Objectives (SLO) System

This module provides a comprehensive system for tracking, monitoring, and managing
service level indicators with support for SLOs, alerting, trending, and reporting.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SLICategory(Enum):
    """Categories for grouping SLIs"""

    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    AVAILABILITY = "availability"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"


class SLOStatus(Enum):
    """Status of SLO compliance"""

    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class AggregationMethod(Enum):
    """Methods for aggregating SLI values"""

    MEAN = "mean"
    MEDIAN = "median"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    RATE = "rate"


@dataclass
class SLIMetadata:
    """
    Metadata describing an SLI

    Attributes:
        name: SLI name
        description: Human-readable description
        unit: Unit of measurement (ms, sec, ratio, etc.)
        category: SLI category
        higher_is_better: Whether higher values are better
        aggregation_method: Default aggregation method
    """

    name: str
    description: str
    unit: str
    category: SLICategory
    higher_is_better: bool = False
    aggregation_method: AggregationMethod = AggregationMethod.MEAN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "category": self.category.value,
            "higher_is_better": self.higher_is_better,
            "aggregation_method": self.aggregation_method.value,
        }


@dataclass
class SLO:
    """
    Service Level Objective

    Defines the target/threshold for an SLI along with warning levels.

    Attributes:
        target: Target value for the SLI
        warning_threshold: Value at which to warn (between current and target)
        breach_threshold: Value at which SLO is considered breached
        evaluation_window: Window for evaluation (in seconds)
    """

    target: float
    warning_threshold: Optional[float] = None
    breach_threshold: Optional[float] = None
    evaluation_window: int = 3600  # 1 hour default

    def evaluate(self, value: float, higher_is_better: bool = False) -> SLOStatus:
        """
        Evaluate current value against SLO.

        Args:
            value: Current SLI value
            higher_is_better: Whether higher values are better

        Returns:
            SLOStatus indicating compliance
        """
        if higher_is_better:
            # For metrics where higher is better (e.g., success rate)
            if self.breach_threshold and value < self.breach_threshold:
                return SLOStatus.BREACHED
            elif self.warning_threshold and value < self.warning_threshold:
                return SLOStatus.WARNING
            elif value >= self.target:
                return SLOStatus.HEALTHY
            else:
                return SLOStatus.WARNING
        else:
            # For metrics where lower is better (e.g., latency)
            if self.breach_threshold and value > self.breach_threshold:
                return SLOStatus.BREACHED
            elif self.warning_threshold and value > self.warning_threshold:
                return SLOStatus.WARNING
            elif value <= self.target:
                return SLOStatus.HEALTHY
            else:
                return SLOStatus.WARNING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "target": self.target,
            "warning_threshold": self.warning_threshold,
            "breach_threshold": self.breach_threshold,
            "evaluation_window": self.evaluation_window,
        }


@dataclass
class SLIs:
    """
    Service Level Indicators for the Vulcan system

    This dataclass holds current SLI values with validation and metadata support.
    """

    # Performance SLIs (latency in ms/sec)
    get_by_hash_p95_ms: float = 0.0
    hybrid_search_p95_ms: float = 0.0
    cold_restore_p95_sec: float = 0.0
    unlearning_duration_seconds: float = 0.0
    fast_lane_duration_seconds: float = 0.0
    tombstone_visible_p95_sec: float = 0.0

    # Reliability SLIs (rates and ratios)
    proof_verify_success_rate: float = 1.0
    vector_recall_at_50: float = 1.0
    range_read_hit_ratio: float = 1.0

    # Efficiency SLIs
    fragmentation_ratio: float = 0.0
    compaction_wa_ratio: float = 0.0  # Write amplification

    # Quality SLIs
    dqs_reject_rate: float = 0.0
    unlearning_retain_recall_degradation: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate SLI values"""
        self.validate()

    def validate(self) -> List[str]:
        """
        Validate SLI values are within reasonable ranges.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check latency metrics are non-negative
        latency_fields = [
            "get_by_hash_p95_ms",
            "hybrid_search_p95_ms",
            "cold_restore_p95_sec",
            "unlearning_duration_seconds",
            "fast_lane_duration_seconds",
            "tombstone_visible_p95_sec",
        ]

        for field_name in latency_fields:
            value = getattr(self, field_name)
            if value < 0:
                errors.append(f"{field_name} cannot be negative: {value}")

        # Check rate/ratio metrics are in [0, 1]
        ratio_fields = [
            "proof_verify_success_rate",
            "vector_recall_at_50",
            "range_read_hit_ratio",
            "fragmentation_ratio",
            "compaction_wa_ratio",
            "dqs_reject_rate",
            "unlearning_retain_recall_degradation",
        ]

        for field_name in ratio_fields:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                errors.append(f"{field_name} must be in [0, 1]: {value}")

        if errors:
            logger.warning(f"SLI validation errors: {errors}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "get_by_hash_p95_ms": self.get_by_hash_p95_ms,
            "hybrid_search_p95_ms": self.hybrid_search_p95_ms,
            "cold_restore_p95_sec": self.cold_restore_p95_sec,
            "unlearning_duration_seconds": self.unlearning_duration_seconds,
            "fast_lane_duration_seconds": self.fast_lane_duration_seconds,
            "tombstone_visible_p95_sec": self.tombstone_visible_p95_sec,
            "proof_verify_success_rate": self.proof_verify_success_rate,
            "vector_recall_at_50": self.vector_recall_at_50,
            "range_read_hit_ratio": self.range_read_hit_ratio,
            "fragmentation_ratio": self.fragmentation_ratio,
            "compaction_wa_ratio": self.compaction_wa_ratio,
            "dqs_reject_rate": self.dqs_reject_rate,
            "unlearning_retain_recall_degradation": self.unlearning_retain_recall_degradation,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SLIs:
        """Create from dictionary"""
        timestamp = data.pop("timestamp", None)
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(**data, timestamp=timestamp or datetime.now())

    def get_category_values(self, category: SLICategory) -> Dict[str, float]:
        """
        Get all SLI values for a specific category.

        Args:
            category: SLI category to filter by

        Returns:
            Dictionary of SLI name to value
        """
        metadata_map = get_sli_metadata_map()
        result = {}

        for field_name, meta in metadata_map.items():
            if meta.category == category:
                result[field_name] = getattr(self, field_name)

        return result


@dataclass
class SLISnapshot:
    """
    Snapshot of SLIs at a point in time with metadata

    Attributes:
        slis: SLI values
        metadata: Additional metadata
        labels: Custom labels for filtering
    """

    slis: SLIs
    metadata: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "slis": self.slis.to_dict(),
            "metadata": self.metadata,
            "labels": self.labels,
        }


def get_sli_metadata_map() -> Dict[str, SLIMetadata]:
    """
    Get metadata for all SLIs.

    Returns:
        Dictionary mapping SLI field names to metadata
    """
    return {
        "get_by_hash_p95_ms": SLIMetadata(
            name="Get by Hash P95 Latency",
            description="95th percentile latency for hash lookups",
            unit="ms",
            category=SLICategory.PERFORMANCE,
            higher_is_better=False,
            aggregation_method=AggregationMethod.P95,
        ),
        "hybrid_search_p95_ms": SLIMetadata(
            name="Hybrid Search P95 Latency",
            description="95th percentile latency for hybrid search queries",
            unit="ms",
            category=SLICategory.PERFORMANCE,
            higher_is_better=False,
            aggregation_method=AggregationMethod.P95,
        ),
        "cold_restore_p95_sec": SLIMetadata(
            name="Cold Restore P95 Time",
            description="95th percentile time for cold restore operations",
            unit="sec",
            category=SLICategory.PERFORMANCE,
            higher_is_better=False,
            aggregation_method=AggregationMethod.P95,
        ),
        "unlearning_duration_seconds": SLIMetadata(
            name="Unlearning Duration",
            description="Time taken for unlearning operations",
            unit="sec",
            category=SLICategory.PERFORMANCE,
            higher_is_better=False,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "fast_lane_duration_seconds": SLIMetadata(
            name="Fast Lane Duration",
            description="Time taken for fast lane processing",
            unit="sec",
            category=SLICategory.PERFORMANCE,
            higher_is_better=False,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "tombstone_visible_p95_sec": SLIMetadata(
            name="Tombstone Visibility P95",
            description="95th percentile time until tombstones are visible",
            unit="sec",
            category=SLICategory.PERFORMANCE,
            higher_is_better=False,
            aggregation_method=AggregationMethod.P95,
        ),
        "proof_verify_success_rate": SLIMetadata(
            name="Proof Verification Success Rate",
            description="Rate of successful proof verifications",
            unit="ratio",
            category=SLICategory.RELIABILITY,
            higher_is_better=True,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "vector_recall_at_50": SLIMetadata(
            name="Vector Recall@50",
            description="Recall at 50 for vector search",
            unit="ratio",
            category=SLICategory.QUALITY,
            higher_is_better=True,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "range_read_hit_ratio": SLIMetadata(
            name="Range Read Hit Ratio",
            description="Cache hit ratio for range reads",
            unit="ratio",
            category=SLICategory.EFFICIENCY,
            higher_is_better=True,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "fragmentation_ratio": SLIMetadata(
            name="Fragmentation Ratio",
            description="Storage fragmentation ratio",
            unit="ratio",
            category=SLICategory.EFFICIENCY,
            higher_is_better=False,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "compaction_wa_ratio": SLIMetadata(
            name="Compaction Write Amplification",
            description="Write amplification ratio during compaction",
            unit="ratio",
            category=SLICategory.EFFICIENCY,
            higher_is_better=False,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "dqs_reject_rate": SLIMetadata(
            name="DQS Reject Rate",
            description="Rate of data rejected due to quality score",
            unit="ratio",
            category=SLICategory.QUALITY,
            higher_is_better=False,
            aggregation_method=AggregationMethod.MEAN,
        ),
        "unlearning_retain_recall_degradation": SLIMetadata(
            name="Unlearning Recall Degradation",
            description="Degradation in recall after unlearning",
            unit="ratio",
            category=SLICategory.QUALITY,
            higher_is_better=False,
            aggregation_method=AggregationMethod.MEAN,
        ),
    }


def get_default_slos() -> Dict[str, SLO]:
    """
    Get default SLOs for all SLIs.

    Returns:
        Dictionary mapping SLI field names to SLOs
    """
    return {
        # Performance SLOs (latency targets)
        "get_by_hash_p95_ms": SLO(
            target=10.0,
            warning_threshold=15.0,
            breach_threshold=25.0,
            evaluation_window=3600,
        ),
        "hybrid_search_p95_ms": SLO(
            target=100.0,
            warning_threshold=200.0,
            breach_threshold=500.0,
            evaluation_window=3600,
        ),
        "cold_restore_p95_sec": SLO(
            target=30.0,
            warning_threshold=60.0,
            breach_threshold=120.0,
            evaluation_window=3600,
        ),
        "unlearning_duration_seconds": SLO(
            target=300.0,
            warning_threshold=600.0,
            breach_threshold=1800.0,
            evaluation_window=3600,
        ),
        "fast_lane_duration_seconds": SLO(
            target=5.0,
            warning_threshold=10.0,
            breach_threshold=30.0,
            evaluation_window=3600,
        ),
        "tombstone_visible_p95_sec": SLO(
            target=1.0,
            warning_threshold=5.0,
            breach_threshold=10.0,
            evaluation_window=3600,
        ),
        # Reliability SLOs (rate targets)
        "proof_verify_success_rate": SLO(
            target=0.9999,
            warning_threshold=0.999,
            breach_threshold=0.99,
            evaluation_window=3600,
        ),
        "vector_recall_at_50": SLO(
            target=0.95,
            warning_threshold=0.90,
            breach_threshold=0.85,
            evaluation_window=3600,
        ),
        "range_read_hit_ratio": SLO(
            target=0.90,
            warning_threshold=0.80,
            breach_threshold=0.70,
            evaluation_window=3600,
        ),
        # Efficiency SLOs
        "fragmentation_ratio": SLO(
            target=0.10,
            warning_threshold=0.20,
            breach_threshold=0.30,
            evaluation_window=3600,
        ),
        "compaction_wa_ratio": SLO(
            target=2.0,
            warning_threshold=5.0,
            breach_threshold=10.0,
            evaluation_window=3600,
        ),
        # Quality SLOs
        "dqs_reject_rate": SLO(
            target=0.01,
            warning_threshold=0.05,
            breach_threshold=0.10,
            evaluation_window=3600,
        ),
        "unlearning_retain_recall_degradation": SLO(
            target=0.01,
            warning_threshold=0.05,
            breach_threshold=0.10,
            evaluation_window=3600,
        ),
    }


class SLITracker:
    """
    Track SLI values over time with aggregation and analysis.

    This class maintains a rolling window of SLI measurements and provides
    statistical analysis, trend detection, and SLO monitoring.
    """

    def __init__(self, max_history: int = 10000, slos: Optional[Dict[str, SLO]] = None):
        """
        Initialize SLI tracker.

        Args:
            max_history: Maximum number of snapshots to keep
            slos: Optional SLO definitions (uses defaults if not provided)
        """
        self.max_history = max_history
        self.history: deque[SLISnapshot] = deque(maxlen=max_history)
        self.slos = slos or get_default_slos()
        self.metadata_map = get_sli_metadata_map()

        # Statistics
        self.total_recorded = 0
        self.slo_breaches: Dict[str, int] = defaultdict(int)
        self.slo_warnings: Dict[str, int] = defaultdict(int)

        logger.info(f"Initialized SLI Tracker with max_history={max_history}")

    def record(
        self,
        slis: SLIs,
        metadata: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record an SLI snapshot.

        Args:
            slis: SLI values to record
            metadata: Optional metadata
            labels: Optional labels for filtering
        """
        # Validate SLIs
        errors = slis.validate()
        if errors:
            logger.warning(f"Recording SLIs with validation errors: {errors}")

        snapshot = SLISnapshot(slis=slis, metadata=metadata or {}, labels=labels or {})

        self.history.append(snapshot)
        self.total_recorded += 1

        # Check SLOs
        self._check_slos(slis)

        logger.debug(f"Recorded SLI snapshot (total: {self.total_recorded})")

    def _check_slos(self, slis: SLIs) -> None:
        """Check current SLIs against SLOs"""
        for field_name, slo in self.slos.items():
            value = getattr(slis, field_name)
            metadata = self.metadata_map.get(field_name)

            if metadata:
                status = slo.evaluate(value, metadata.higher_is_better)

                if status == SLOStatus.BREACHED:
                    self.slo_breaches[field_name] += 1
                    logger.error(
                        f"SLO BREACH: {field_name}={value} (target={slo.target})"
                    )
                elif status == SLOStatus.WARNING:
                    self.slo_warnings[field_name] += 1
                    logger.warning(
                        f"SLO WARNING: {field_name}={value} (target={slo.target})"
                    )

    def get_current(self) -> Optional[SLIs]:
        """Get most recent SLI values"""
        if self.history:
            return self.history[-1].slis
        return None

    def get_statistics(
        self, field_name: str, window_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get statistics for a specific SLI field.

        Args:
            field_name: Name of SLI field
            window_seconds: Optional time window (all data if None)

        Returns:
            Dictionary with statistics (mean, median, p95, etc.)
        """
        values = self._get_field_values(field_name, window_seconds)

        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "stddev": 0.0,
            }

        sorted_values = sorted(values)

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p50": sorted_values[int(len(values) * 0.50)],
            "p95": sorted_values[int(len(values) * 0.95)]
            if len(values) > 1
            else sorted_values[0],
            "p99": sorted_values[int(len(values) * 0.99)]
            if len(values) > 1
            else sorted_values[0],
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def _get_field_values(
        self, field_name: str, window_seconds: Optional[int] = None
    ) -> List[float]:
        """Get values for a field within optional time window"""
        cutoff = None
        if window_seconds:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)

        values = []
        for snapshot in self.history:
            if cutoff and snapshot.slis.timestamp < cutoff:
                continue
            values.append(getattr(snapshot.slis, field_name))

        return values

    def get_trend(self, field_name: str, window_size: int = 100) -> str:
        """
        Detect trend for an SLI field.

        Args:
            field_name: Name of SLI field
            window_size: Number of recent samples to analyze

        Returns:
            "improving", "stable", or "degrading"
        """
        if len(self.history) < window_size:
            return "insufficient_data"

        recent = list(self.history)[-window_size:]
        first_half = [getattr(s.slis, field_name) for s in recent[: window_size // 2]]
        second_half = [getattr(s.slis, field_name) for s in recent[window_size // 2 :]]

        mean_first = statistics.mean(first_half)
        mean_second = statistics.mean(second_half)

        metadata = self.metadata_map.get(field_name)
        if not metadata:
            return "unknown"

        diff = mean_second - mean_first

        # Adjust interpretation based on whether higher is better
        if metadata.higher_is_better:
            if diff > 0.05 * mean_first:
                return "improving"
            elif diff < -0.05 * mean_first:
                return "degrading"
        else:
            if diff < -0.05 * mean_first:
                return "improving"
            elif diff > 0.05 * mean_first:
                return "degrading"

        return "stable"

    def get_slo_status(self, field_name: str) -> Tuple[SLOStatus, float]:
        """
        Get current SLO status for a field.

        Args:
            field_name: Name of SLI field

        Returns:
            Tuple of (status, current_value)
        """
        current = self.get_current()
        if not current:
            return SLOStatus.UNKNOWN, 0.0

        value = getattr(current, field_name)
        slo = self.slos.get(field_name)
        metadata = self.metadata_map.get(field_name)

        if not slo or not metadata:
            return SLOStatus.UNKNOWN, value

        status = slo.evaluate(value, metadata.higher_is_better)
        return status, value

    def get_all_slo_statuses(self) -> Dict[str, Tuple[SLOStatus, float]]:
        """Get SLO status for all fields"""
        return {
            field_name: self.get_slo_status(field_name)
            for field_name in self.slos.keys()
        }

    def get_slo_compliance_rate(
        self, field_name: str, window_seconds: Optional[int] = None
    ) -> float:
        """
        Calculate SLO compliance rate for a field.

        Args:
            field_name: Name of SLI field
            window_seconds: Optional time window

        Returns:
            Compliance rate (0-1)
        """
        values = self._get_field_values(field_name, window_seconds)
        if not values:
            return 1.0

        slo = self.slos.get(field_name)
        metadata = self.metadata_map.get(field_name)

        if not slo or not metadata:
            return 1.0

        compliant = 0
        for value in values:
            status = slo.evaluate(value, metadata.higher_is_better)
            if status == SLOStatus.HEALTHY:
                compliant += 1

        return compliant / len(values)

    def generate_report(self, window_seconds: Optional[int] = 3600) -> str:
        """
        Generate comprehensive SLI/SLO report.

        Args:
            window_seconds: Time window for report (default 1 hour)

        Returns:
            Formatted report string
        """
        window_str = f"last {window_seconds}s" if window_seconds else "all time"

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SLI/SLO MONITORING REPORT                             ║
║                        Window: {window_str:^42}                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

SUMMARY
=======
Total Snapshots: {self.total_recorded}
History Size: {len(self.history)}
SLO Breaches: {sum(self.slo_breaches.values())}
SLO Warnings: {sum(self.slo_warnings.values())}

"""

        # Group by category
        by_category = defaultdict(list)
        for field_name, metadata in self.metadata_map.items():
            by_category[metadata.category].append((field_name, metadata))

        for category in SLICategory:
            if category not in by_category:
                continue

            report += f"\n{category.value.upper()} METRICS\n"
            report += "=" * 80 + "\n"

            for field_name, metadata in by_category[category]:
                stats = self.get_statistics(field_name, window_seconds)
                status, current_value = self.get_slo_status(field_name)
                compliance = self.get_slo_compliance_rate(field_name, window_seconds)
                trend = self.get_trend(field_name)
                slo = self.slos.get(field_name)

                status_icon = {
                    SLOStatus.HEALTHY: "✓",
                    SLOStatus.WARNING: "⚠",
                    SLOStatus.BREACHED: "✗",
                    SLOStatus.UNKNOWN: "?",
                }.get(status, "?")

                report += f"\n{status_icon} {metadata.name}\n"
                report += f"  Current: {current_value:.4f} {metadata.unit}"

                if slo:
                    report += f" (target: {slo.target:.4f})\n"
                else:
                    report += "\n"

                report += f"  Statistics: mean={stats['mean']:.4f}, "
                report += f"p95={stats['p95']:.4f}, "
                report += f"max={stats['max']:.4f}\n"
                report += f"  Compliance: {compliance:.2%}\n"
                report += f"  Trend: {trend.upper()}\n"

                if (
                    field_name in self.slo_breaches
                    and self.slo_breaches[field_name] > 0
                ):
                    report += f"  Breaches: {self.slo_breaches[field_name]}\n"
                if (
                    field_name in self.slo_warnings
                    and self.slo_warnings[field_name] > 0
                ):
                    report += f"  Warnings: {self.slo_warnings[field_name]}\n"

        return report

    def export_metrics(self, format: str = "prometheus") -> str:
        """
        Export metrics in various formats.

        Args:
            format: Export format ("prometheus", "json", "influx")

        Returns:
            Formatted metrics string
        """
        current = self.get_current()
        if not current:
            return ""

        if format == "prometheus":
            return self._export_prometheus(current)
        elif format == "json":
            return json.dumps(current.to_dict(), indent=2)
        elif format == "influx":
            return self._export_influx(current)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def _export_prometheus(self, slis: SLIs) -> str:
        """Export in Prometheus format"""
        lines = []

        for field_name, metadata in self.metadata_map.items():
            value = getattr(slis, field_name)
            metric_name = f"vulcan_sli_{field_name}"

            # Add help text
            lines.append(f"# HELP {metric_name} {metadata.description}")
            lines.append(f"# TYPE {metric_name} gauge")

            # Add metric value
            lines.append(f"{metric_name} {value}")

            # Add SLO target as separate metric
            slo = self.slos.get(field_name)
            if slo:
                target_metric = f"{metric_name}_slo_target"
                lines.append(f"# HELP {target_metric} SLO target for {metadata.name}")
                lines.append(f"# TYPE {target_metric} gauge")
                lines.append(f"{target_metric} {slo.target}")

        return "\n".join(lines) + "\n"

    def _export_influx(self, slis: SLIs) -> str:
        """Export in InfluxDB line protocol format"""
        lines = []
        timestamp_ns = int(slis.timestamp.timestamp() * 1e9)

        for field_name, metadata in self.metadata_map.items():
            value = getattr(slis, field_name)

            # Format: measurement,tag=value field=value timestamp
            line = f"vulcan_sli,metric={field_name},category={metadata.category.value} "
            line += f"value={value} {timestamp_ns}"
            lines.append(line)

        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        """
        Save SLI history to file.

        Args:
            path: Path to save to
        """
        data = {
            "total_recorded": self.total_recorded,
            "slo_breaches": dict(self.slo_breaches),
            "slo_warnings": dict(self.slo_warnings),
            "history": [snapshot.to_dict() for snapshot in self.history],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved SLI history to {path}")

    @classmethod
    def load(cls, path: Path) -> SLITracker:
        """
        Load SLI history from file.

        Args:
            path: Path to load from

        Returns:
            Restored SLITracker
        """
        with open(path, "r") as f:
            data = json.load(f)

        tracker = cls()
        tracker.total_recorded = data["total_recorded"]
        tracker.slo_breaches = defaultdict(int, data["slo_breaches"])
        tracker.slo_warnings = defaultdict(int, data["slo_warnings"])

        for snapshot_dict in data["history"]:
            slis = SLIs.from_dict(snapshot_dict["slis"])
            snapshot = SLISnapshot(
                slis=slis,
                metadata=snapshot_dict.get("metadata", {}),
                labels=snapshot_dict.get("labels", {}),
            )
            tracker.history.append(snapshot)

        logger.info(f"Loaded SLI history from {path}")
        return tracker


class SLIAggregator:
    """
    Aggregate SLI values across multiple time windows.

    Provides time-based aggregation (e.g., per minute, per hour) for dashboards.
    """

    def __init__(self):
        """Initialize SLI aggregator"""
        self.aggregated_data: Dict[str, Dict[datetime, Dict[str, float]]] = defaultdict(
            dict
        )

    def aggregate(
        self, tracker: SLITracker, window_minutes: int = 5
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Aggregate SLI values into time buckets.

        Args:
            tracker: SLI tracker with history
            window_minutes: Size of time buckets in minutes

        Returns:
            Dictionary mapping field names to time series
        """
        result = defaultdict(list)

        # Group snapshots by time bucket
        buckets = defaultdict(lambda: defaultdict(list))

        for snapshot in tracker.history:
            # Round timestamp to window
            bucket_time = snapshot.slis.timestamp.replace(second=0, microsecond=0)
            bucket_time = bucket_time.replace(
                minute=(bucket_time.minute // window_minutes) * window_minutes
            )

            # Add all field values to bucket
            for field_name in tracker.metadata_map.keys():
                value = getattr(snapshot.slis, field_name)
                buckets[field_name][bucket_time].append(value)

        # Aggregate each bucket
        for field_name, time_buckets in buckets.items():
            metadata = tracker.metadata_map.get(field_name)
            if not metadata:
                continue

            for bucket_time, values in sorted(time_buckets.items()):
                # Apply aggregation method
                agg_value = self._apply_aggregation(values, metadata.aggregation_method)
                result[field_name].append((bucket_time, agg_value))

        return dict(result)

    def _apply_aggregation(
        self, values: List[float], method: AggregationMethod
    ) -> float:
        """Apply aggregation method to values"""
        if not values:
            return 0.0

        if method == AggregationMethod.MEAN:
            return statistics.mean(values)
        elif method == AggregationMethod.MEDIAN or method == AggregationMethod.P50:
            return statistics.median(values)
        elif method == AggregationMethod.P95:
            sorted_vals = sorted(values)
            return sorted_vals[int(len(values) * 0.95)]
        elif method == AggregationMethod.P99:
            sorted_vals = sorted(values)
            return sorted_vals[int(len(values) * 0.99)]
        elif method == AggregationMethod.MAX:
            return max(values)
        elif method == AggregationMethod.MIN:
            return min(values)
        elif method == AggregationMethod.SUM:
            return sum(values)
        else:
            return statistics.mean(values)


class SLIAlertManager:
    """
    Manage alerts based on SLI/SLO thresholds.

    Provides alert generation, suppression, and notification callbacks.
    """

    def __init__(
        self,
        tracker: SLITracker,
        alert_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize alert manager.

        Args:
            tracker: SLI tracker to monitor
            alert_callback: Optional callback for alerts (level, message, context)
        """
        self.tracker = tracker
        self.alert_callback = alert_callback
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.suppression_window = 300  # 5 minutes

        logger.info("Initialized SLI Alert Manager")

    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.

        Returns:
            List of active alerts
        """
        alerts = []
        current_time = datetime.now()

        statuses = self.tracker.get_all_slo_statuses()

        for field_name, (status, value) in statuses.items():
            if status in [SLOStatus.BREACHED, SLOStatus.WARNING]:
                # Check if already alerted recently
                if field_name in self.active_alerts:
                    last_alert = self.active_alerts[field_name]
                    if (
                        current_time - last_alert
                    ).total_seconds() < self.suppression_window:
                        continue  # Suppress duplicate alert

                # Generate alert
                metadata = self.tracker.metadata_map.get(field_name)
                slo = self.tracker.slos.get(field_name)

                alert = {
                    "field": field_name,
                    "level": "critical" if status == SLOStatus.BREACHED else "warning",
                    "status": status.value,
                    "current_value": value,
                    "target": slo.target if slo else None,
                    "timestamp": current_time,
                    "message": f"{metadata.name} {status.value}: {value:.4f} {metadata.unit}",
                }

                alerts.append(alert)
                self.active_alerts[field_name] = current_time
                self.alert_history.append(alert)

                # Trigger callback
                if self.alert_callback:
                    self.alert_callback(alert["level"], alert["message"], alert)

                logger.warning(f"ALERT: {alert['message']}")

        return alerts

    def clear_alert(self, field_name: str) -> None:
        """Clear active alert for a field"""
        if field_name in self.active_alerts:
            del self.active_alerts[field_name]
            logger.info(f"Cleared alert for {field_name}")

    def get_active_alerts(self) -> Dict[str, datetime]:
        """Get currently active alerts"""
        return self.active_alerts.copy()

    def get_alert_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        history = self.alert_history
        if limit:
            history = history[-limit:]
        return history


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("=== Testing SLI System ===\n")

    # Create SLIs
    slis = SLIs(
        get_by_hash_p95_ms=8.5,
        hybrid_search_p95_ms=95.0,
        cold_restore_p95_sec=28.0,
        proof_verify_success_rate=0.9999,
        vector_recall_at_50=0.96,
        fragmentation_ratio=0.08,
        dqs_reject_rate=0.02,
    )

    print("Current SLIs:")
    print(json.dumps(slis.to_dict(), indent=2))

    # Validate
    errors = slis.validate()
    print(f"\nValidation: {'PASS' if not errors else f'FAIL - {errors}'}")

    # Create tracker
    print("\n=== Testing SLI Tracker ===\n")
    tracker = SLITracker()

    # Record multiple snapshots
    import random

    for i in range(100):
        test_slis = SLIs(
            get_by_hash_p95_ms=random.uniform(5, 20),
            hybrid_search_p95_ms=random.uniform(80, 150),
            cold_restore_p95_sec=random.uniform(20, 50),
            proof_verify_success_rate=random.uniform(0.995, 1.0),
            vector_recall_at_50=random.uniform(0.90, 0.98),
            fragmentation_ratio=random.uniform(0.05, 0.15),
            dqs_reject_rate=random.uniform(0.01, 0.05),
        )
        tracker.record(test_slis)

    # Get statistics
    print("Get by Hash P95 Statistics:")
    stats = tracker.get_statistics("get_by_hash_p95_ms")
    print(json.dumps(stats, indent=2))

    # Check SLO status
    print("\nSLO Statuses:")
    statuses = tracker.get_all_slo_statuses()
    for field_name, (status, value) in statuses.items():
        print(f"  {field_name}: {status.value} (value={value:.4f})")

    # Generate report
    print("\n" + "=" * 80)
    print(tracker.generate_report(window_seconds=None))

    # Export metrics
    print("\n=== Prometheus Export ===\n")
    prometheus_output = tracker.export_metrics("prometheus")
    print(prometheus_output[:500] + "...\n")

    # Test alerts
    print("=== Testing Alert Manager ===\n")

    def alert_handler(level, message, context):
        print(f"[{level.upper()}] {message}")

    alert_mgr = SLIAlertManager(tracker, alert_callback=alert_handler)
    alerts = alert_mgr.check_alerts()
    print(f"Active alerts: {len(alerts)}")
