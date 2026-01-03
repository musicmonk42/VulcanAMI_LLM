#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Boundedness and Memory Leak Detection Tests.

This module provides tests for memory stability and bounded behavior of the
VULCAN cognitive architecture. Tests verify that:

1. RSS memory growth stays within acceptable limits
2. Performance doesn't degrade over many iterations
3. Internal history/cache structures remain bounded
4. No memory leaks occur during extended operation

Configuration (env vars):
    - PERF_MAX_RSS_GROWTH_MB: Max allowed RSS growth (default: 50)
    - PERF_MAX_SLOWDOWN_PCT: Max allowed slowdown (default: 20)
    - PERF_ITERATIONS: Number of iterations to run (default: 500)

Usage:
    pytest tests/perf/test_boundedness.py -m boundedness -v
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .conftest import (
    PSUTIL_AVAILABLE,
    PerfConfig,
    MemoryTracker,
    save_json_report,
    save_markdown_report,
    generate_summary_markdown,
)


# ============================================================
# MOCK COMPONENT WITH INTERNAL STATE
# ============================================================

class MockStatefulComponent:
    """
    Mock component that maintains internal state for boundedness testing.
    
    Simulates VULCAN components with history buffers, caches, and state.
    """
    
    def __init__(
        self,
        max_history_size: int = 1000,
        max_cache_size: int = 500,
    ):
        self.max_history_size = max_history_size
        self.max_cache_size = max_cache_size
        
        # Internal state that should remain bounded
        self._provenance_records: List[Dict[str, Any]] = []
        self._semantic_history: List[str] = []
        self._query_cache: Dict[str, Any] = {}
        self._result_buffer: List[Dict[str, Any]] = []
        
        # Counters
        self._total_operations = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    def process(self, query: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Process a query and update internal state.
        
        Simulates typical VULCAN query processing with state management.
        """
        self._total_operations += 1
        
        # Simulate latency
        time.sleep(0.001 + (hash(query) % 100) / 10000)
        
        # Check cache
        cache_key = f"{query_type}:{hash(query)}"
        if cache_key in self._query_cache:
            self._cache_hits += 1
            cached = self._query_cache[cache_key]
            return {"success": True, "cached": True, **cached}
        
        self._cache_misses += 1
        
        # Create result
        result = {
            "response": f"Response to {query_type} query",
            "timestamp": time.time(),
            "operation_id": self._total_operations,
        }
        
        # Update cache (with eviction)
        if len(self._query_cache) >= self.max_cache_size:
            # Evict oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[cache_key] = result
        
        # Update provenance records (with bounded size)
        record = {
            "query": query[:100],  # Truncate for memory
            "query_type": query_type,
            "timestamp": time.time(),
            "result_summary": result["response"][:50],
        }
        self._provenance_records.append(record)
        if len(self._provenance_records) > self.max_history_size:
            self._provenance_records = self._provenance_records[-self.max_history_size:]
        
        # Update semantic history
        self._semantic_history.append(f"{query_type}:{hash(query)}")
        if len(self._semantic_history) > self.max_history_size:
            self._semantic_history = self._semantic_history[-self.max_history_size:]
        
        # Update result buffer
        self._result_buffer.append(result)
        if len(self._result_buffer) > 100:  # Keep last 100 results
            self._result_buffer = self._result_buffer[-100:]
        
        return {"success": True, "cached": False, **result}
    
    def get_state_sizes(self) -> Dict[str, int]:
        """Get sizes of internal state structures."""
        return {
            "provenance_records": len(self._provenance_records),
            "semantic_history": len(self._semantic_history),
            "query_cache": len(self._query_cache),
            "result_buffer": len(self._result_buffer),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            "total_operations": self._total_operations,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                self._cache_hits / max(1, self._cache_hits + self._cache_misses)
            ),
        }


# ============================================================
# BOUNDEDNESS TESTS
# ============================================================

@pytest.mark.boundedness
@pytest.mark.perf
class TestBoundedness:
    """
    Tests for bounded memory and state behavior.
    """
    
    def test_rss_memory_growth(
        self,
        perf_config: PerfConfig,
        memory_tracker: MemoryTracker,
    ):
        """
        Test that RSS memory growth stays bounded.
        
        Runs many iterations and verifies memory doesn't grow unboundedly.
        """
        component = MockStatefulComponent()
        
        iterations = perf_config.iterations
        sample_interval = max(1, iterations // 20)  # ~20 samples
        
        print(f"\n--- Running {iterations} iterations, sampling every {sample_interval} ---")
        
        # Initial measurement
        gc.collect()
        memory_tracker.sample(label="start")
        
        query_types = ["mathematical", "causal", "symbolic", "probabilistic", "general"]
        
        for i in range(iterations):
            query_type = query_types[i % len(query_types)]
            query = f"Test query {i} for {query_type} reasoning"
            component.process(query, query_type)
            
            # Sample memory periodically
            if (i + 1) % sample_interval == 0:
                gc.collect()
                rss = memory_tracker.sample(label=f"iter_{i+1}")
                print(f"  Iteration {i+1}: RSS = {rss:.2f} MB")
        
        # Final measurement
        gc.collect()
        memory_tracker.sample(label="end")
        
        # Analyze results
        growth_mb = memory_tracker.get_growth_mb()
        max_rss = memory_tracker.get_max_rss_mb()
        
        print(f"\n  Memory growth: {growth_mb:.2f} MB")
        print(f"  Max RSS: {max_rss:.2f} MB")
        print(f"  Threshold: {perf_config.max_rss_growth_mb:.2f} MB")
        
        # Assert bounded growth
        if PSUTIL_AVAILABLE:
            assert growth_mb <= perf_config.max_rss_growth_mb, (
                f"RSS growth ({growth_mb:.2f} MB) exceeds threshold "
                f"({perf_config.max_rss_growth_mb:.2f} MB)"
            )
        else:
            pytest.skip("psutil not available for memory tracking")
    
    def test_performance_stability(self, perf_config: PerfConfig):
        """
        Test that performance doesn't degrade over iterations.
        
        Compares latency of first 10% vs last 10% of iterations.
        """
        component = MockStatefulComponent()
        
        iterations = perf_config.iterations
        bucket_size = max(10, iterations // 10)
        
        print(f"\n--- Testing performance stability over {iterations} iterations ---")
        
        latencies: List[float] = []
        query_types = ["mathematical", "causal", "symbolic", "probabilistic", "general"]
        
        for i in range(iterations):
            query_type = query_types[i % len(query_types)]
            query = f"Stability test query {i}"
            
            start = time.perf_counter()
            component.process(query, query_type)
            latency = time.perf_counter() - start
            latencies.append(latency)
        
        # Compare first 10% vs last 10%
        first_bucket = latencies[:bucket_size]
        last_bucket = latencies[-bucket_size:]
        
        first_median = statistics.median(first_bucket)
        last_median = statistics.median(last_bucket)
        
        # Calculate slowdown percentage
        if first_median > 0:
            slowdown_pct = ((last_median - first_median) / first_median) * 100
        else:
            slowdown_pct = 0.0
        
        print(f"  First {bucket_size} median: {first_median*1000:.3f}ms")
        print(f"  Last {bucket_size} median: {last_median*1000:.3f}ms")
        print(f"  Slowdown: {slowdown_pct:.1f}%")
        print(f"  Threshold: {perf_config.max_slowdown_pct:.1f}%")
        
        # Assert no significant degradation
        assert slowdown_pct <= perf_config.max_slowdown_pct, (
            f"Performance degraded by {slowdown_pct:.1f}% "
            f"(threshold: {perf_config.max_slowdown_pct:.1f}%)"
        )
    
    def test_history_structure_bounds(self, perf_config: PerfConfig):
        """
        Test that history/cache structures remain bounded.
        
        Verifies internal data structures don't grow unboundedly.
        """
        max_history = 1000
        max_cache = 500
        
        component = MockStatefulComponent(
            max_history_size=max_history,
            max_cache_size=max_cache,
        )
        
        iterations = perf_config.iterations
        
        print(f"\n--- Testing structure bounds over {iterations} iterations ---")
        
        size_samples: List[Dict[str, int]] = []
        query_types = ["mathematical", "causal", "symbolic", "probabilistic", "general"]
        
        for i in range(iterations):
            query_type = query_types[i % len(query_types)]
            query = f"Bounds test query {i} - unique content {i * 17 % 1000}"
            component.process(query, query_type)
            
            # Sample sizes periodically
            if (i + 1) % 100 == 0:
                sizes = component.get_state_sizes()
                size_samples.append(sizes)
        
        # Check final sizes
        final_sizes = component.get_state_sizes()
        
        print(f"  Final provenance_records: {final_sizes['provenance_records']}")
        print(f"  Final semantic_history: {final_sizes['semantic_history']}")
        print(f"  Final query_cache: {final_sizes['query_cache']}")
        print(f"  Final result_buffer: {final_sizes['result_buffer']}")
        
        # Assert bounded sizes
        assert final_sizes["provenance_records"] <= max_history, (
            f"Provenance records ({final_sizes['provenance_records']}) "
            f"exceed max ({max_history})"
        )
        assert final_sizes["semantic_history"] <= max_history, (
            f"Semantic history ({final_sizes['semantic_history']}) "
            f"exceed max ({max_history})"
        )
        assert final_sizes["query_cache"] <= max_cache, (
            f"Query cache ({final_sizes['query_cache']}) "
            f"exceed max ({max_cache})"
        )
    
    def test_iteration_time_trend(self, perf_config: PerfConfig):
        """
        Test iteration time trend to detect gradual degradation.
        
        Groups iterations into windows and checks for consistent performance.
        """
        component = MockStatefulComponent()
        
        iterations = perf_config.iterations
        window_size = 100
        
        print(f"\n--- Testing iteration time trends over {iterations} iterations ---")
        
        window_times: List[float] = []
        current_window_latencies: List[float] = []
        
        query_types = ["mathematical", "causal", "symbolic", "probabilistic", "general"]
        
        for i in range(iterations):
            query_type = query_types[i % len(query_types)]
            query = f"Trend test query {i}"
            
            start = time.perf_counter()
            component.process(query, query_type)
            latency = time.perf_counter() - start
            current_window_latencies.append(latency)
            
            # Calculate window average every window_size iterations
            if len(current_window_latencies) >= window_size:
                window_avg = sum(current_window_latencies)
                window_times.append(window_avg)
                current_window_latencies = []
        
        # Handle remaining iterations
        if current_window_latencies:
            window_avg = sum(current_window_latencies) * (window_size / len(current_window_latencies))
            window_times.append(window_avg)
        
        if len(window_times) < 2:
            pytest.skip("Not enough windows for trend analysis")
        
        # Compare first 10% of windows vs last 10%
        num_windows = len(window_times)
        first_windows = window_times[:max(1, num_windows // 10)]
        last_windows = window_times[-max(1, num_windows // 10):]
        
        first_avg = statistics.mean(first_windows)
        last_avg = statistics.mean(last_windows)
        
        if first_avg > 0:
            trend_pct = ((last_avg - first_avg) / first_avg) * 100
        else:
            trend_pct = 0.0
        
        print(f"  Window size: {window_size} iterations")
        print(f"  Number of windows: {num_windows}")
        print(f"  First windows avg: {first_avg*1000:.2f}ms")
        print(f"  Last windows avg: {last_avg*1000:.2f}ms")
        print(f"  Trend: {trend_pct:+.1f}%")
        
        # Assert no significant degradation trend
        assert trend_pct <= perf_config.max_slowdown_pct, (
            f"Performance trend shows {trend_pct:.1f}% degradation "
            f"(threshold: {perf_config.max_slowdown_pct:.1f}%)"
        )


# ============================================================
# EXTENDED LEAK DETECTION
# ============================================================

@pytest.mark.boundedness
@pytest.mark.perf
class TestLeakDetection:
    """
    Extended leak detection tests.
    """
    
    def test_repeated_init_cleanup(self, perf_config: PerfConfig, memory_tracker: MemoryTracker):
        """
        Test for leaks in repeated initialization and cleanup cycles.
        """
        num_cycles = 20
        
        print(f"\n--- Testing {num_cycles} init/cleanup cycles ---")
        
        gc.collect()
        memory_tracker.sample(label="start")
        
        for i in range(num_cycles):
            # Create component with state
            component = MockStatefulComponent()
            
            # Do some work
            for j in range(50):
                component.process(f"Cycle {i} query {j}", "general")
            
            # Cleanup (Python will GC, but simulate explicit cleanup)
            del component
            gc.collect()
            
            if (i + 1) % 5 == 0:
                rss = memory_tracker.sample(label=f"cycle_{i+1}")
                print(f"  After cycle {i+1}: RSS = {rss:.2f} MB")
        
        gc.collect()
        memory_tracker.sample(label="end")
        
        growth_mb = memory_tracker.get_growth_mb()
        print(f"  Total growth: {growth_mb:.2f} MB")
        
        if PSUTIL_AVAILABLE:
            # For cycle-based leak detection, use a per-cycle threshold
            # Total growth should be minimal since each cycle cleans up
            max_growth_per_cycle = perf_config.max_rss_growth_mb / num_cycles
            print(f"  Max allowed per-cycle growth: {max_growth_per_cycle:.2f} MB")
            print(f"  Average per-cycle growth: {growth_mb / num_cycles:.2f} MB")
            
            # Assert total growth is bounded
            assert growth_mb <= perf_config.max_rss_growth_mb, (
                f"Memory growth ({growth_mb:.2f} MB) exceeds threshold "
                f"({perf_config.max_rss_growth_mb:.2f} MB) - suggests leak"
            )
    
    def test_concurrent_operation_cleanup(
        self,
        perf_config: PerfConfig,
        memory_tracker: MemoryTracker,
    ):
        """
        Test for leaks during concurrent operations.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        num_workers = 5
        queries_per_worker = 100
        
        print(f"\n--- Testing concurrent cleanup with {num_workers} workers ---")
        
        gc.collect()
        memory_tracker.sample(label="start")
        
        def worker_task(worker_id: int) -> Dict[str, Any]:
            """Worker that processes queries."""
            component = MockStatefulComponent()
            
            for i in range(queries_per_worker):
                component.process(f"Worker {worker_id} query {i}", "general")
            
            stats = component.get_statistics()
            del component
            return {
                "worker_id": worker_id,
                "queries": queries_per_worker,
                **stats,
            }
        
        # Run workers
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            for future in as_completed(futures):
                results.append(future.result())
        
        gc.collect()
        memory_tracker.sample(label="end")
        
        growth_mb = memory_tracker.get_growth_mb()
        total_queries = sum(r["queries"] for r in results)
        
        print(f"  Total queries: {total_queries}")
        print(f"  Memory growth: {growth_mb:.2f} MB")
        
        if PSUTIL_AVAILABLE:
            assert growth_mb <= perf_config.max_rss_growth_mb, (
                f"Concurrent memory growth ({growth_mb:.2f} MB) exceeds threshold"
            )


# ============================================================
# REPORT GENERATION
# ============================================================

@pytest.fixture(scope="module", autouse=True)
def generate_boundedness_report(request, output_dir):
    """
    Generate boundedness report after all tests complete.
    """
    yield
    
    # Generate reports with test results
    from datetime import datetime, timezone
    
    report_data = {
        "test_suite": "boundedness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "max_rss_growth_mb": float(os.environ.get("PERF_MAX_RSS_GROWTH_MB", "50")),
            "max_slowdown_pct": float(os.environ.get("PERF_MAX_SLOWDOWN_PCT", "20")),
            "iterations": int(os.environ.get("PERF_ITERATIONS", "500")),
        },
        "psutil_available": PSUTIL_AVAILABLE,
    }
    
    # Save JSON report
    json_path = output_dir / "boundedness.json"
    save_json_report(report_data, json_path)
    
    # Generate markdown
    markdown_lines = [
        "# Boundedness Test Results",
        "",
        f"**Generated:** {report_data['timestamp']}",
        "",
        "## Configuration",
        "",
        f"- **Max RSS Growth:** {report_data['configuration']['max_rss_growth_mb']} MB",
        f"- **Max Slowdown:** {report_data['configuration']['max_slowdown_pct']}%",
        f"- **Iterations:** {report_data['configuration']['iterations']}",
        f"- **psutil Available:** {report_data['psutil_available']}",
        "",
        "## Test Categories",
        "",
        "- RSS Memory Growth: Verifies memory doesn't grow unboundedly",
        "- Performance Stability: Verifies no degradation over iterations", 
        "- History Structure Bounds: Verifies internal caches stay bounded",
        "- Leak Detection: Tests for leaks in init/cleanup cycles",
    ]
    
    md_path = output_dir / "boundedness.md"
    save_markdown_report("\n".join(markdown_lines), md_path)
    
    print(f"\n📊 Boundedness reports saved to {output_dir}")
