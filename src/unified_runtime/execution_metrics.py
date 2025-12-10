"""
Execution Metrics Module for Graphix IR
Tracks, aggregates, and summarizes runtime behavior at multiple levels:
- Per-node execution
- Per-graph execution
- Rolling runtime health (through MetricsAggregator)

This module is intentionally self-contained and thread-safe so it can be
used safely both in the core execution engine and at UnifiedRuntime scope.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

# psutil is optional. We degrade gracefully if not installed.
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    _PSUTIL_AVAILABLE = False


# ============================================================================
# LOW-LEVEL HELPERS
# ============================================================================


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that won't throw on denominator == 0."""
    if denominator == 0:
        return default
    return numerator / denominator


def _timestamp_ms() -> float:
    """Return current time in milliseconds."""
    return time.time() * 1000.0


def _collect_resource_snapshot() -> Dict[str, Any]:
    """
    Capture lightweight system/process resource usage.
    Returns empty fields if psutil is not available.

    Returns:
        {
            "rss_mb": float,
            "cpu_percent": float,
            "timestamp_ms": float
        }
    """
    snapshot = {
        "rss_mb": None,
        "cpu_percent": None,
        "timestamp_ms": _timestamp_ms(),
    }

    if not _PSUTIL_AVAILABLE:
        return snapshot

    try:
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=None)

        snapshot["rss_mb"] = float(rss_mb)
        snapshot["cpu_percent"] = float(cpu_percent)
    except Exception:
        # If psutil explodes for some reason, just return partial.
        pass

    return snapshot


# ============================================================================
# PER-NODE METRICS
# ============================================================================


@dataclass
class NodeExecutionStats:
    """
    Metrics for a single node execution.
    This is useful if you want fine-grained profiling or audit trails.

    We track:
    - node_id
    - node_type / op_type (so we can aggregate by type later)
    - status ("success", "failed", "timeout", etc.)
    - start_ms / end_ms / duration_ms
    - cache_hit
    - error_message (if any)
    """

    node_id: str
    node_type: Optional[str] = None
    status: str = "unknown"
    cache_hit: bool = False
    error_message: Optional[str] = None
    start_ms: float = field(default_factory=_timestamp_ms)
    end_ms: Optional[float] = None
    duration_ms: Optional[float] = None

    def finalize(self):
        """Mark end time, compute duration."""
        if self.end_ms is None:
            self.end_ms = _timestamp_ms()
        if self.duration_ms is None:
            self.duration_ms = max(0.0, self.end_ms - self.start_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "status": self.status,
            "cache_hit": self.cache_hit,
            "error_message": self.error_message,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


# ============================================================================
# PER-GRAPH EXECUTION METRICS (ONE RUN)
# ============================================================================


@dataclass
class ExecutionMetrics:
    """
    Metrics for a single graph execution run.

    This is accumulated during execution _and then frozen_ at the end
    (and passed into MetricsAggregator).

    Fields:
        cache_hits           : how many node results were served from cache
        cache_misses         : how many node results required fresh compute
        nodes_executed       : total count of nodes that actually ran
        nodes_succeeded      : how many nodes ended in a success state
        nodes_failed         : how many nodes errored
        total_latency_ms     : wall clock duration for the entire graph run
        execution_count      : how many graph executions contributed to this metric
                               (usually 1 for a single run snapshot)

        graph_start_ms       : timestamp when the graph run began (ms)
        graph_end_ms         : timestamp when the graph run ended (ms)

        resource_start       : resource snapshot (cpu/mem) at start
        resource_end         : resource snapshot (cpu/mem) at end

        node_details         : list[NodeExecutionStats] for detailed profiling
    """

    cache_hits: int = 0
    cache_misses: int = 0
    nodes_executed: int = 0
    nodes_succeeded: int = 0
    nodes_failed: int = 0

    total_latency_ms: float = 0.0
    execution_count: int = 1

    graph_start_ms: float = field(default_factory=_timestamp_ms)
    graph_end_ms: Optional[float] = None

    resource_start: Dict[str, Any] = field(default_factory=_collect_resource_snapshot)
    resource_end: Dict[str, Any] = field(default_factory=dict)

    # Per-node breakdowns
    node_details: List[NodeExecutionStats] = field(default_factory=list)

    # Arbitrary metadata hooks (graph_id, execution_id, mode, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------
    # MUTATION METHODS
    # ------------------------

    def record_node_start(
        self, node_id: str, node_type: Optional[str] = None, cache_hit: bool = False
    ) -> NodeExecutionStats:
        """
        Create and register a NodeExecutionStats object for a node that
        is *beginning* execution. The caller will later finalize() it
        and update success/failure.
        """
        nes = NodeExecutionStats(
            node_id=node_id,
            node_type=node_type,
            cache_hit=cache_hit,
            status="running",
        )
        self.node_details.append(nes)

        # Bookkeeping for hits/misses as soon as we know
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # We'll increment nodes_executed once we know it actually completed.
        return nes

    def record_node_end(
        self,
        node_stats: NodeExecutionStats,
        status: str,
        error_message: Optional[str] = None,
    ):
        """
        Finalize a node's execution record and update rollups.
        """
        node_stats.status = status
        node_stats.error_message = error_message
        node_stats.finalize()

        self.nodes_executed += 1

        if status.lower() in ("success", "ok", "completed"):
            self.nodes_succeeded += 1
        else:
            # treat anything else as failure-ish
            self.nodes_failed += 1

    def finalize_graph(
        self,
        execution_id: Optional[str] = None,
        mode: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Mark the graph execution as complete:
        - stamps end timestamp
        - computes total_latency_ms
        - captures end resource snapshot
        - attaches metadata like execution_id, mode, etc.
        """
        if self.graph_end_ms is None:
            self.graph_end_ms = _timestamp_ms()

        self.total_latency_ms = max(0.0, self.graph_end_ms - self.graph_start_ms)

        # record end resources
        self.resource_end = _collect_resource_snapshot()

        # allow callers to stash runtime-level info
        if execution_id:
            self.metadata["execution_id"] = execution_id
        if mode:
            self.metadata["execution_mode"] = mode
        if extra_metadata:
            # merge shallowly
            for k, v in extra_metadata.items():
                self.metadata[k] = v

    # ------------------------
    # DERIVED METRICS / SUMMARIES
    # ------------------------

    @property
    def cache_hit_rate(self) -> float:
        total_requests = self.cache_hits + self.cache_misses
        return _safe_div(self.cache_hits, float(total_requests), default=0.0)

    @property
    def avg_latency_per_node_ms(self) -> float:
        return _safe_div(
            self.total_latency_ms, float(max(1, self.nodes_executed)), default=0.0
        )

    @property
    def throughput_nodes_per_sec(self) -> float:
        # nodes / (ms/1000) → nodes/sec
        return _safe_div(
            self.nodes_executed,
            (self.total_latency_ms / 1000.0),
            default=0.0,
        )

    @property
    def success_rate(self) -> float:
        # fraction of executed nodes that didn't fail
        return _safe_div(
            self.nodes_succeeded,
            float(max(1, self.nodes_executed)),
            default=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize all raw + derived metrics.
        NOTE: node_details is included in summarized form.
        """
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "nodes_executed": self.nodes_executed,
            "nodes_succeeded": self.nodes_succeeded,
            "nodes_failed": self.nodes_failed,
            "success_rate": self.success_rate,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_per_node_ms": self.avg_latency_per_node_ms,
            "throughput_nodes_per_sec": self.throughput_nodes_per_sec,
            "execution_count": self.execution_count,
            "graph_start_ms": self.graph_start_ms,
            "graph_end_ms": self.graph_end_ms,
            "resource_start": self.resource_start,
            "resource_end": self.resource_end,
            "metadata": dict(self.metadata),
            "node_details": [n.to_dict() for n in self.node_details],
        }

    def to_prometheus_lines(self, prefix: str = "graphix") -> str:
        """
        Export a Prometheus-style text payload.
        We intentionally keep this tiny and inline — no external exporter.
        """
        lines = [
            f"{prefix}_cache_hits {self.cache_hits}",
            f"{prefix}_cache_misses {self.cache_misses}",
            f"{prefix}_cache_hit_rate {self.cache_hit_rate}",
            f"{prefix}_nodes_executed {self.nodes_executed}",
            f"{prefix}_nodes_succeeded {self.nodes_succeeded}",
            f"{prefix}_nodes_failed {self.nodes_failed}",
            f"{prefix}_success_rate {self.success_rate}",
            f"{prefix}_total_latency_ms {self.total_latency_ms}",
            f"{prefix}_avg_latency_per_node_ms {self.avg_latency_per_node_ms}",
            f"{prefix}_throughput_nodes_per_sec {self.throughput_nodes_per_sec}",
            f"{prefix}_execution_count {self.execution_count}",
        ]

        # Add memory/cpu if we have them
        if self.resource_end.get("rss_mb") is not None:
            lines.append(f"{prefix}_rss_mb {self.resource_end['rss_mb']}")
        if self.resource_end.get("cpu_percent") is not None:
            lines.append(f"{prefix}_cpu_percent {self.resource_end['cpu_percent']}")

        return "\n".join(lines)


# ============================================================================
# ROLLING AGGREGATOR / RUNTIME HEALTH
# ============================================================================


class MetricsAggregator:
    """
    Aggregates ExecutionMetrics objects across many runs.
    Thread-safe.

    The aggregator acts like a rolling window:
    - metrics_history holds up to maxlen entries (default 1000)
    - get_summary() returns global rollup & derived KPIs
    - get_recent(n) returns recent runs' dicts for dashboards
    """

    def __init__(self, max_history: int = 1000):
        self.metrics_history: Deque[ExecutionMetrics] = deque(maxlen=max_history)
        self._lock = threading.Lock()

        # We also maintain a couple of rolling counters for quick checks.
        self._totals = defaultdict(float)  # float so we can safely sum latency_ms, etc.
        self._total_runs = 0

    # ------------------------
    # INGESTION
    # ------------------------

    def record_metrics(self, metrics: ExecutionMetrics):
        """
        Add a completed ExecutionMetrics snapshot.
        Updates internal rollups under a lock.
        (This is the 'record_run' method)
        """
        if not isinstance(metrics, ExecutionMetrics):
            raise TypeError("record_metrics() expects an ExecutionMetrics instance")

        with self._lock:
            self.metrics_history.append(metrics)

            self._totals["cache_hits"] += metrics.cache_hits
            self._totals["cache_misses"] += metrics.cache_misses
            self._totals["nodes_executed"] += metrics.nodes_executed
            self._totals["nodes_succeeded"] += metrics.nodes_succeeded
            self._totals["nodes_failed"] += metrics.nodes_failed
            self._totals["total_latency_ms"] += metrics.total_latency_ms
            self._totals["execution_count"] += metrics.execution_count

            # track memory, cpu if available
            if metrics.resource_end.get("rss_mb") is not None:
                # store most recent in totals for quick peek
                self._totals["last_rss_mb"] = metrics.resource_end.get("rss_mb")
            if metrics.resource_end.get("cpu_percent") is not None:
                self._totals["last_cpu_percent"] = metrics.resource_end.get(
                    "cpu_percent"
                )

            self._total_runs += 1

    # ------------------------
    # SUMMARY / HEALTH
    # ------------------------

    def _compute_rollup(self) -> Dict[str, Any]:
        """
        Internal helper. Assumes lock is already held.
        Produces a summary for all recorded runs in history.
        """
        total_exec_count = int(self._totals.get("execution_count", 0))
        total_nodes = int(self._totals.get("nodes_executed", 0))
        total_hits = int(self._totals.get("cache_hits", 0))
        total_misses = int(self._totals.get("cache_misses", 0))
        total_succeeded = int(self._totals.get("nodes_succeeded", 0))
        total_failed = int(self._totals.get("nodes_failed", 0))
        total_latency_ms = float(self._totals.get("total_latency_ms", 0.0))

        cache_hit_rate = _safe_div(
            total_hits,
            float(total_hits + total_misses),
            default=0.0,
        )

        success_rate = _safe_div(
            total_succeeded,
            float(max(1, total_nodes)),
            default=0.0,
        )

        avg_latency_ms = _safe_div(
            total_latency_ms,
            float(max(1, total_exec_count)),
            default=0.0,
        )

        avg_latency_per_node_ms = _safe_div(
            total_latency_ms,
            float(max(1, total_nodes)),
            default=0.0,
        )

        throughput_nodes_per_sec = _safe_div(
            total_nodes,
            (total_latency_ms / 1000.0),
            default=0.0,
        )

        # We'll surface "last seen" resource stats, not an average.
        last_rss_mb = self._totals.get("last_rss_mb", None)
        last_cpu_percent = self._totals.get("last_cpu_percent", None)

        return {
            "total_runs_recorded": self._total_runs,
            "execution_count_total": total_exec_count,
            "nodes_executed_total": total_nodes,
            "nodes_succeeded_total": total_succeeded,
            "nodes_failed_total": total_failed,
            "cache_hits_total": total_hits,
            "cache_misses_total": total_misses,
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate,
            "avg_latency_ms_per_execution": avg_latency_ms,
            "avg_latency_ms_per_node": avg_latency_per_node_ms,
            "throughput_nodes_per_sec": throughput_nodes_per_sec,
            "last_rss_mb": last_rss_mb,
            "last_cpu_percent": last_cpu_percent,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Public summary API.
        Returns global rollup KPIs across the lifetime of this aggregator.
        This provides the "rolled-up totals" for dashboards.
        Thread-safe.
        """
        with self._lock:
            if self._total_runs == 0:
                # No data yet
                return {
                    "total_runs_recorded": 0,
                    "execution_count_total": 0,
                    "nodes_executed_total": 0,
                    "nodes_succeeded_total": 0,
                    "nodes_failed_total": 0,
                    "cache_hits_total": 0,
                    "cache_misses_total": 0,
                    "cache_hit_rate": 0.0,
                    "success_rate": 0.0,
                    "avg_latency_ms_per_execution": 0.0,
                    "avg_latency_ms_per_node": 0.0,
                    "throughput_nodes_per_sec": 0.0,
                    "last_rss_mb": None,
                    "last_cpu_percent": None,
                }

            return self._compute_rollup()

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return up to the N most recent execution snapshots as dicts.
        Good for dashboards / debugging.
        """
        with self._lock:
            recent = list(self.metrics_history)[-n:]
            return [m.to_dict() for m in recent]

    def to_prometheus_lines(self, prefix: str = "graphix_agg") -> str:
        """
        Export a Prometheus-style flat metric dump of the rollup.
        Not meant to be scraped at nanosecond frequency — it's a snapshot.
        """
        with self._lock:
            roll = self._compute_rollup() if self._total_runs > 0 else None

        if roll is None:
            # No data
            return (
                f"{prefix}_total_runs_recorded 0\n"
                f"{prefix}_cache_hit_rate 0.0\n"
                f"{prefix}_success_rate 0.0\n"
                f"{prefix}_throughput_nodes_per_sec 0.0\n"
            )

        lines = [
            f"{prefix}_total_runs_recorded {roll['total_runs_recorded']}",
            f"{prefix}_execution_count_total {roll['execution_count_total']}",
            f"{prefix}_nodes_executed_total {roll['nodes_executed_total']}",
            f"{prefix}_nodes_succeeded_total {roll['nodes_succeeded_total']}",
            f"{prefix}_nodes_failed_total {roll['nodes_failed_total']}",
            f"{prefix}_cache_hits_total {roll['cache_hits_total']}",
            f"{prefix}_cache_misses_total {roll['cache_misses_total']}",
            f"{prefix}_cache_hit_rate {roll['cache_hit_rate']}",
            f"{prefix}_success_rate {roll['success_rate']}",
            f"{prefix}_avg_latency_ms_per_execution {roll['avg_latency_ms_per_execution']}",
            f"{prefix}_avg_latency_ms_per_node {roll['avg_latency_ms_per_node']}",
            f"{prefix}_throughput_nodes_per_sec {roll['throughput_nodes_per_sec']}",
        ]

        if roll.get("last_rss_mb") is not None:
            lines.append(f"{prefix}_last_rss_mb {roll['last_rss_mb']}")
        if roll.get("last_cpu_percent") is not None:
            lines.append(f"{prefix}_last_cpu_percent {roll['last_cpu_percent']}")

        return "\n".join(lines)
