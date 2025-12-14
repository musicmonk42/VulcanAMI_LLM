"""
Test suite for the updated execution_metrics.py
Covers:
- NodeExecutionStats
- ExecutionMetrics
- MetricsAggregator
- Derived metrics
- Graceful psutil handling
- Prometheus export helpers
"""

import math
import time
from unittest.mock import MagicMock, patch

import pytest

# Adjust this import if needed depending on how you run tests, e.g.:
# import execution_metrics as em
import unified_runtime.execution_metrics as em

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    """Tiny helper for float comparisons that shouldn't require pytest.approx everywhere."""
    return abs(a - b) <= tol


# -----------------------------------------------------------------------------
# NodeExecutionStats tests
# -----------------------------------------------------------------------------


class TestNodeExecutionStats:
    def test_creation_defaults_and_finalize(self):
        node = em.NodeExecutionStats(
            node_id="node-123",
            node_type="AddNode",
            cache_hit=True,
        )

        # start_ms should be automatically populated as a float timestamp in ms
        assert isinstance(node.start_ms, float)

        # These should not be populated until finalize() is called
        assert node.end_ms is None
        assert node.duration_ms is None

        # In the implementation, NodeExecutionStats defaults status to "unknown"
        # and not "running". We assert that here.
        assert node.status == "unknown"

        # Provided / captured fields
        assert node.cache_hit is True
        assert node.error_message is None
        assert node.node_id == "node-123"
        assert node.node_type == "AddNode"

        # After a very small sleep to ensure time advances, finalize should stamp
        # end_ms and duration_ms and guarantee non-negative duration.
        pre_finalize_start = node.start_ms
        time.sleep(0.001)
        node.finalize()

        assert node.end_ms is not None
        assert node.duration_ms is not None
        assert node.end_ms >= pre_finalize_start
        assert node.duration_ms >= 0.0

    def test_to_dict(self):
        node = em.NodeExecutionStats(
            node_id="nA",
            node_type="MultiplyNode",
            cache_hit=False,
        )
        node.status = "success"
        node.error_message = None
        node.finalize()

        data = node.to_dict()
        assert data["node_id"] == "nA"
        assert data["node_type"] == "MultiplyNode"
        assert data["status"] == "success"
        assert data["cache_hit"] is False
        assert "duration_ms" in data
        assert "start_ms" in data
        assert "end_ms" in data


# -----------------------------------------------------------------------------
# ExecutionMetrics tests
# -----------------------------------------------------------------------------


class TestExecutionMetrics:
    def test_initial_state(self):
        m = em.ExecutionMetrics()

        # Baseline counters should start at zero / neutral values
        assert m.cache_hits == 0
        assert m.cache_misses == 0
        assert m.nodes_executed == 0
        assert m.nodes_succeeded == 0
        assert m.nodes_failed == 0
        assert m.total_latency_ms == 0.0
        assert m.execution_count == 1

        # Timestamps and resource snapshots
        assert isinstance(m.graph_start_ms, float)
        assert m.graph_end_ms is None
        assert isinstance(m.resource_start, dict)
        assert m.resource_end == {}

        # Per-node detail list and arbitrary metadata
        assert isinstance(m.node_details, list)
        assert m.metadata == {}

    def test_record_node_start_and_end_success(self):
        m = em.ExecutionMetrics()

        # Call record_node_start() to begin tracking a node.
        # We mark cache_hit=True to ensure it increments cache_hits.
        node_stats = m.record_node_start(
            node_id="node1",
            node_type="TransformNode",
            cache_hit=True,
        )

        # The returned node_stats must be a NodeExecutionStats
        assert isinstance(node_stats, em.NodeExecutionStats)
        assert node_stats.node_id == "node1"
        assert node_stats.node_type == "TransformNode"
        assert node_stats.cache_hit is True

        # cache_hits should increment because cache_hit=True
        assert m.cache_hits == 1
        assert m.cache_misses == 0

        # node_details should now contain this in-flight node
        assert len(m.node_details) == 1

        # Simulate some execution time and then finish successfully
        time.sleep(0.001)
        m.record_node_end(
            node_stats=node_stats,
            status="success",
            error_message=None,
        )

        # Rollups should have been updated
        assert m.nodes_executed == 1
        assert m.nodes_succeeded == 1
        assert m.nodes_failed == 0

        # Node stats should have been finalized
        assert node_stats.status == "success"
        assert node_stats.error_message is None
        assert node_stats.duration_ms is not None
        assert node_stats.end_ms is not None

    def test_record_node_start_and_end_failure_and_cache_miss(self):
        m = em.ExecutionMetrics()

        # Start node where cache_hit=False so cache_misses should increment
        node_stats = m.record_node_start(
            node_id="node2",
            node_type="CheckNode",
            cache_hit=False,
        )

        # cache_misses incremented and cache_hits not touched
        assert m.cache_hits == 0
        assert m.cache_misses == 1

        # Now end this node with a failing status
        m.record_node_end(
            node_stats=node_stats,
            status="FAILED",
            error_message="boom",
        )

        # nodes_executed should now be 1
        assert m.nodes_executed == 1
        assert m.nodes_succeeded == 0
        assert m.nodes_failed == 1

        # Node stats should be finalized and contain error message
        assert node_stats.status == "FAILED"
        assert node_stats.error_message == "boom"
        assert node_stats.duration_ms is not None

    def test_finalize_graph_sets_end_and_latency_and_metadata(self):
        m = em.ExecutionMetrics()

        # Add two nodes so we have something non-trivial in metrics
        n1 = m.record_node_start("a", "TypeA", cache_hit=True)
        time.sleep(0.001)
        m.record_node_end(n1, "success")

        n2 = m.record_node_start("b", "TypeB", cache_hit=False)
        time.sleep(0.001)
        m.record_node_end(n2, "FAILED", error_message="x")

        # Before finalize_graph(), resource_end should still be {}, graph_end_ms None,
        # and total_latency_ms should still be 0.0
        assert m.resource_end == {}
        assert m.graph_end_ms is None
        assert m.total_latency_ms == 0.0

        # Finalize the graph and attach some metadata
        m.finalize_graph(
            execution_id="exec-777",
            mode="parallel",
            extra_metadata={"graph_node_count": 2, "custom_flag": True},
        )

        # After finalize:
        # - graph_end_ms populated
        # - total_latency_ms populated
        # - resource_end snapshot captured
        # - metadata merged in
        assert m.graph_end_ms is not None
        assert m.total_latency_ms >= 0.0
        assert isinstance(m.resource_end, dict)
        assert m.metadata["execution_id"] == "exec-777"
        assert m.metadata["execution_mode"] == "parallel"
        assert m.metadata["graph_node_count"] == 2
        assert m.metadata["custom_flag"] is True

    def test_derived_properties_cache_hit_rate_success_rate_throughput(self):
        m = em.ExecutionMetrics()

        # Create two nodes, one succeeds and one fails.
        # First node: cache_hit=True
        n1 = m.record_node_start("n1", "Foo", cache_hit=True)
        m.record_node_end(n1, "success")

        # Second node: cache_hit=False
        n2 = m.record_node_start("n2", "Foo", cache_hit=False)
        m.record_node_end(n2, "FAILED")

        # At this point:
        # cache_hits = 1
        # cache_misses = 1
        # nodes_executed = 2
        # nodes_succeeded = 1
        # nodes_failed = 1

        # Pretend this run took 100ms total so we can test throughput, etc.
        # We'll force graph_start_ms and graph_end_ms and total_latency_ms so that
        # throughput_nodes_per_sec and avg_latency_per_node_ms are deterministic.
        m.graph_start_ms = m.graph_start_ms - 100.0
        m.graph_end_ms = m.graph_start_ms + 100.0
        m.total_latency_ms = 100.0  # pretend 100ms wall time

        # cache_hit_rate: hits / (hits + misses) = 1 / 2 = 0.5
        assert m.cache_hit_rate == pytest.approx(0.5)

        # success_rate: succeeded / nodes_executed = 1 / 2 = 0.5
        assert m.success_rate == pytest.approx(0.5)

        # throughput_nodes_per_sec:
        #   nodes_executed / (total_latency_ms / 1000)
        #   = 2 / (100 / 1000) = 2 / 0.1 = 20.0
        assert m.throughput_nodes_per_sec == pytest.approx(20.0)

        # avg_latency_per_node_ms:
        #   total_latency_ms / nodes_executed
        #   = 100 / 2 = 50.0
        assert m.avg_latency_per_node_ms == pytest.approx(50.0)

    def test_to_dict_structure_and_values(self):
        m = em.ExecutionMetrics()

        # Add one node that fails so that we have non-trivial values
        n = m.record_node_start("nodeX", "WeirdNode", cache_hit=False)
        m.record_node_end(n, "FAILED", error_message="nope")

        # finalize_graph() should populate timestamps, latency, resource_end,
        # and inject metadata we pass in
        m.finalize_graph(execution_id="abc123", mode="sequential")

        d = m.to_dict()

        # The dict must contain all of these top-level keys
        for key in [
            "cache_hits",
            "cache_misses",
            "cache_hit_rate",
            "nodes_executed",
            "nodes_succeeded",
            "nodes_failed",
            "success_rate",
            "total_latency_ms",
            "avg_latency_per_node_ms",
            "throughput_nodes_per_sec",
            "execution_count",
            "graph_start_ms",
            "graph_end_ms",
            "resource_start",
            "resource_end",
            "metadata",
            "node_details",
        ]:
            assert key in d

        # node_details should be a list of dicts describing per-node execution
        assert isinstance(d["node_details"], list)

        # metadata should include execution_id / execution_mode from finalize_graph()
        assert d["metadata"]["execution_id"] == "abc123"
        assert d["metadata"]["execution_mode"] == "sequential"

        # Sanity check numeric rolled-up values
        # We executed 1 node, which failed
        assert d["nodes_executed"] == 1
        assert d["nodes_failed"] == 1
        assert d["nodes_succeeded"] == 0

    def test_to_prometheus_lines_includes_core_metrics(self):
        m = em.ExecutionMetrics()

        # One successful node so success_rate and hit rate are non-zero-ish.
        n = m.record_node_start("X", "Thing", cache_hit=True)
        m.record_node_end(n, "success")

        # finalize_graph() to make sure resource_end and total_latency_ms are populated
        m.finalize_graph()

        # Export Prometheus-style metrics with a custom prefix to make assertions easier
        out = m.to_prometheus_lines(prefix="graphix_test")

        # Check that the output contains critical metrics we expect to publish
        assert "graphix_test_cache_hits " in out
        assert "graphix_test_cache_misses " in out
        assert "graphix_test_cache_hit_rate " in out
        assert "graphix_test_nodes_executed " in out
        assert "graphix_test_nodes_succeeded " in out
        assert "graphix_test_nodes_failed " in out
        assert "graphix_test_success_rate " in out
        assert "graphix_test_total_latency_ms " in out
        assert "graphix_test_avg_latency_per_node_ms " in out
        assert "graphix_test_throughput_nodes_per_sec " in out
        assert "graphix_test_execution_count " in out

        # Now simulate that resource_end captured memory and CPU usage.
        # We manually patch it so we can assert those lines appear.
        m.resource_end["rss_mb"] = 123.0
        m.resource_end["cpu_percent"] = 88.0

        out2 = m.to_prometheus_lines(prefix="graphix_test")
        assert "graphix_test_rss_mb 123.0" in out2
        assert "graphix_test_cpu_percent 88.0" in out2


# -----------------------------------------------------------------------------
# MetricsAggregator tests
# -----------------------------------------------------------------------------


class TestMetricsAggregator:
    def test_initial_state(self):
        agg = em.MetricsAggregator(max_history=5)

        # metrics_history should be a deque with maxlen, initially empty.
        # We don't assert exact type equality, we assert deque-like behavior:
        # - it has append()
        # - it's length-checkable
        assert hasattr(agg.metrics_history, "append")
        assert len(agg.metrics_history) == 0

        # Internal bookkeeping counters should start clean.
        assert agg._total_runs == 0
        assert isinstance(agg._totals, dict)

    def test_record_metrics_and_summary_basic(self):
        agg = em.MetricsAggregator(max_history=10)

        # Build and finalize two ExecutionMetrics runs
        m1 = em.ExecutionMetrics()
        n1 = m1.record_node_start("a", "TypeA", cache_hit=True)
        m1.record_node_end(n1, "success")
        m1.finalize_graph()

        m2 = em.ExecutionMetrics()
        n2 = m2.record_node_start("b", "TypeB", cache_hit=False)
        m2.record_node_end(n2, "FAILED", error_message="x")
        m2.finalize_graph()

        # Ingest these into the aggregator
        agg.record_metrics(m1)
        agg.record_metrics(m2)

        # After recording two runs, the summary should reflect totals across both
        summary = agg.get_summary()

        # total_runs_recorded should equal 2
        assert summary["total_runs_recorded"] == 2

        # cache_hits_total should count from both runs. In m1 we had cache_hit=True once,
        # in m2 we had cache_hit=False once. That means:
        # - hits_total should be 1
        # - misses_total should be 1
        assert summary["cache_hits_total"] == 1
        assert summary["cache_misses_total"] == 1

        # Nodes executed in total: one in each run, so 2 total
        assert summary["nodes_executed_total"] == 2

        # nodes_succeeded_total: m1 succeeded, m2 failed
        assert summary["nodes_succeeded_total"] == 1
        assert summary["nodes_failed_total"] == 1

        # Derived success_rate: 1 success / 2 executed = 0.5
        assert summary["success_rate"] == pytest.approx(0.5)

        # Derived cache_hit_rate: 1 hit / (1 hit + 1 miss) = 0.5
        assert summary["cache_hit_rate"] == pytest.approx(0.5)

        # Timing-dependent fields are computed internally; we only assert presence.
        assert "avg_latency_ms_per_execution" in summary
        assert "avg_latency_ms_per_node" in summary
        assert "throughput_nodes_per_sec" in summary

        # Resource snapshots "last_rss_mb" / "last_cpu_percent" may be None or floats.
        # We just assert the keys exist.
        assert "last_rss_mb" in summary
        assert "last_cpu_percent" in summary

    def test_record_metrics_limits_history(self):
        agg = em.MetricsAggregator(max_history=3)

        # Add 5 finalized runs into an aggregator that is only supposed to keep
        # a history of max_history=3 elements. We assert that metrics_history
        # respects its maxlen, while _total_runs keeps counting all runs ever seen.
        for i in range(5):
            m = em.ExecutionMetrics()
            n = m.record_node_start(f"id{i}", "Foo", cache_hit=(i % 2 == 0))
            m.record_node_end(n, "success")
            m.finalize_graph()

            agg.record_metrics(m)

        # deque should have at most 3
        assert len(agg.metrics_history) == 3

        # but we've ingested 5 total runs
        assert agg._total_runs == 5

    def test_get_recent_returns_dicts(self):
        agg = em.MetricsAggregator(max_history=10)

        runs = []
        for i in range(4):
            m = em.ExecutionMetrics()
            n = m.record_node_start(f"op{i}", "Foo", cache_hit=False)
            m.record_node_end(n, "success")
            m.finalize_graph(execution_id=f"exec-{i}")
            agg.record_metrics(m)
            runs.append(m)

        # Ask aggregator for the 2 most recent runs
        recent_2 = agg.get_recent(2)

        # Should be a list of dicts
        assert isinstance(recent_2, list)
        assert len(recent_2) == 2

        # They should correspond to the last 2 executions we recorded:
        # exec-2 and exec-3
        assert recent_2[0]["metadata"]["execution_id"] == "exec-2"
        assert recent_2[-1]["metadata"]["execution_id"] == "exec-3"

    def test_prometheus_export_from_aggregator(self):
        agg = em.MetricsAggregator(max_history=10)

        # Create a single run with one successful node, then finalize it.
        m = em.ExecutionMetrics()
        n = m.record_node_start("foo", "Bar", cache_hit=True)
        m.record_node_end(n, "success")
        m.finalize_graph()

        # Feed into aggregator
        agg.record_metrics(m)

        # Export a Prometheus-style dump
        prom = agg.to_prometheus_lines(prefix="graphix_agg_test")

        # Confirm presence of key metrics in the dump
        assert "graphix_agg_test_total_runs_recorded " in prom
        assert "graphix_agg_test_execution_count_total " in prom
        assert "graphix_agg_test_nodes_executed_total " in prom
        assert "graphix_agg_test_nodes_succeeded_total " in prom
        assert "graphix_agg_test_nodes_failed_total " in prom
        assert "graphix_agg_test_cache_hits_total " in prom
        assert "graphix_agg_test_cache_misses_total " in prom
        assert "graphix_agg_test_cache_hit_rate " in prom
        assert "graphix_agg_test_success_rate " in prom
        assert "graphix_agg_test_avg_latency_ms_per_execution " in prom
        assert "graphix_agg_test_avg_latency_ms_per_node " in prom
        assert "graphix_agg_test_throughput_nodes_per_sec " in prom


# -----------------------------------------------------------------------------
# psutil / resource snapshot behavior
# -----------------------------------------------------------------------------


class TestResourceSnapshotBehavior:
    def test_collect_resource_snapshot_without_psutil(self, monkeypatch):
        """
        Force _PSUTIL_AVAILABLE = False and make sure we still return the right shape
        and don't throw.
        """
        # Force the module-level _PSUTIL_AVAILABLE to False so code
        # goes down the "psutil not available" path.
        monkeypatch.setattr(em, "_PSUTIL_AVAILABLE", False, raising=True)

        snap = em._collect_resource_snapshot()

        # The shape of the snapshot dict should always include rss_mb,
        # cpu_percent, and timestamp_ms. When psutil is not available,
        # rss_mb and cpu_percent should be None.
        assert "rss_mb" in snap
        assert "cpu_percent" in snap
        assert "timestamp_ms" in snap

        assert snap["rss_mb"] is None
        assert snap["cpu_percent"] is None
        assert isinstance(snap["timestamp_ms"], float)

    def test_collect_resource_snapshot_with_psutil(self, monkeypatch):
        """
        Mock psutil to verify that we pick up memory/cpu values correctly.
        """
        mock_psutil = MagicMock()

        mock_process = MagicMock()
        mock_meminfo = MagicMock()

        # Pretend the process is using 123 MB of RSS.
        mock_meminfo.rss = 123 * 1024 * 1024
        mock_process.memory_info.return_value = mock_meminfo
        mock_process.memory_info.return_value.rss = 123 * 1024 * 1024
        mock_process.memory_info.return_value.vms = 0  # not used currently

        # cpu_percent() is queried at module level
        mock_process.memory_percent.return_value = 7.5
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_percent.return_value = 42.0

        # Force the code path that thinks psutil is available,
        # and swap in our mock psutil object.
        monkeypatch.setattr(em, "_PSUTIL_AVAILABLE", True, raising=True)
        monkeypatch.setattr(em, "psutil", mock_psutil, raising=True)

        snap = em._collect_resource_snapshot()

        # After mocking, these should be real numbers instead of None.
        assert snap["rss_mb"] == pytest.approx(123.0)
        assert snap["cpu_percent"] == pytest.approx(42.0)
        assert isinstance(snap["timestamp_ms"], float)


# -----------------------------------------------------------------------------
# _safe_div tests
# -----------------------------------------------------------------------------


class TestSafeDiv:
    def test_safe_div_normal(self):
        # Normal division should behave correctly
        assert em._safe_div(10, 2) == 5.0
        assert approx_equal(em._safe_div(3.0, 0.5), 6.0)

    def test_safe_div_zero_denominator(self):
        # When denominator is zero, we should get the provided default
        assert em._safe_div(10, 0) == 0.0
        assert em._safe_div(10, 0, default=-1.0) == -1.0
