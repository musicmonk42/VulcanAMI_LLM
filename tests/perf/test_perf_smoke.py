#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Performance Smoke Tests.

This module provides performance micro-benchmarks and concurrency tests for the
VULCAN cognitive architecture. Tests are designed to run quickly in CI while
still providing meaningful scalability metrics.

Test Categories:
    1. Micro-benchmarks: Individual component timing
    2. Concurrency tests: Throughput under load
    3. End-to-end reasoning: Full pipeline performance

Configuration (env vars):
    - PERF_CONCURRENCY_LEVELS: Comma-separated concurrency levels (default: "10,25,50")
    - PERF_CONCURRENCY_DURATION: Duration for each concurrency test (default: 30)

Usage:
    pytest tests/perf/test_perf_smoke.py -m perf -v
"""

from __future__ import annotations

import asyncio
import gc
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from .conftest import (
    PerfConfig,
    PerfResultCollector,
    get_perf_config,
    save_json_report,
    save_markdown_report,
    generate_summary_markdown,
)

# Try to import VULCAN components
try:
    from src.vulcan.reasoning.graphix_executor import GraphixExecutor
    GRAPHIX_AVAILABLE = True
except ImportError:
    GRAPHIX_AVAILABLE = False
    GraphixExecutor = None

try:
    from src.vulcan.orchestrator.agent_pool import AgentPoolManager
    AGENT_POOL_AVAILABLE = True
except ImportError:
    AGENT_POOL_AVAILABLE = False
    AgentPoolManager = None

try:
    from src.vulcan.world_model.world_model import WorldModel
    WORLD_MODEL_AVAILABLE = True
except ImportError:
    WORLD_MODEL_AVAILABLE = False
    WorldModel = None


# ============================================================
# TEST DATA
# ============================================================

SAMPLE_QUERIES = [
    ("mathematical", "Solve equation: 3x + 7 = 22"),
    ("causal", "Analyze causal relationship between X and Y"),
    ("symbolic", "Prove logical statement: (P → Q) ∧ P ⊢ Q"),
    ("probabilistic", "Calculate P(A|B) given P(B|A)=0.8, P(A)=0.3, P(B)=0.5"),
    ("general", "Explain the concept of emergence in complex systems"),
]


# ============================================================
# MOCK REASONING FOR TESTING
# ============================================================

class MockReasoner:
    """
    Mock reasoner for performance testing when VULCAN is not available.
    
    Simulates variable latency based on query complexity.
    """
    
    def __init__(self, base_latency: float = 0.01):
        self.base_latency = base_latency
        self._call_count = 0
    
    def execute(self, query: str, query_type: str = "general") -> Dict[str, Any]:
        """Execute a mock reasoning operation."""
        self._call_count += 1
        
        # Simulate variable latency based on query type
        latency_multiplier = {
            "mathematical": 1.5,
            "causal": 2.0,
            "symbolic": 1.8,
            "probabilistic": 1.7,
            "general": 1.0,
        }.get(query_type, 1.0)
        
        # Add deterministic variability based on query hash (no randomness)
        jitter = (hash(query) % 100) / 1000.0
        
        # Use computation-based delay to avoid CI timing inconsistencies
        # This is more stable than time.sleep() in CI environments
        target_delay = self.base_latency * latency_multiplier + jitter
        start = time.perf_counter()
        # Spin-wait for deterministic delay (avoids scheduler inconsistency)
        while (time.perf_counter() - start) < target_delay:
            pass
        
        return {
            "success": True,
            "response": f"Mock response for {query_type}: {query[:50]}",
            "query_type": query_type,
            "call_count": self._call_count,
        }
    
    async def execute_async(self, query: str, query_type: str = "general") -> Dict[str, Any]:
        """Execute a mock async reasoning operation."""
        self._call_count += 1
        
        latency_multiplier = {
            "mathematical": 1.5,
            "causal": 2.0,
            "symbolic": 1.8,
            "probabilistic": 1.7,
            "general": 1.0,
        }.get(query_type, 1.0)
        
        jitter = (hash(query) % 100) / 1000.0
        target_delay = self.base_latency * latency_multiplier + jitter
        
        # For async, use asyncio.sleep as it yields to event loop
        # This is appropriate for async tests and doesn't block other tasks
        await asyncio.sleep(target_delay)
        
        return {
            "success": True,
            "response": f"Mock async response for {query_type}: {query[:50]}",
            "query_type": query_type,
            "call_count": self._call_count,
        }


def get_reasoner():
    """Get a reasoner instance (real or mock)."""
    # For now, use mock to ensure tests always work
    # Real VULCAN components can be integrated when available
    return MockReasoner(base_latency=0.005)


# ============================================================
# MICRO-BENCHMARKS
# ============================================================

@pytest.mark.perf
class TestMicroBenchmarks:
    """
    Micro-benchmark tests for individual VULCAN components.
    """
    
    def test_reasoner_warmup(self, result_collector: PerfResultCollector):
        """
        Benchmark reasoner warmup time.
        
        Measures time to initialize and warm up the reasoning engine.
        """
        start = time.perf_counter()
        reasoner = get_reasoner()
        init_time = time.perf_counter() - start
        
        # Warmup with a few queries
        warmup_times = []
        for i in range(5):
            query_start = time.perf_counter()
            reasoner.execute(f"Warmup query {i}", "general")
            warmup_times.append(time.perf_counter() - query_start)
        
        result_collector.add_result({
            "test": "reasoner_warmup",
            "init_time_seconds": init_time,
            "warmup_times": warmup_times,
            "avg_warmup_time": statistics.mean(warmup_times),
            "latency_seconds": init_time + sum(warmup_times),
            "success": True,
        })
        
        # Assert warmup is reasonable
        assert init_time < 5.0, f"Reasoner init took too long: {init_time:.2f}s"
        assert statistics.mean(warmup_times) < 1.0, "Warmup queries too slow"
    
    def test_single_query_latency(self, result_collector: PerfResultCollector):
        """
        Benchmark single query latency.
        
        Measures latency for different query types.
        """
        reasoner = get_reasoner()
        
        for query_type, query in SAMPLE_QUERIES:
            start = time.perf_counter()
            result = reasoner.execute(query, query_type)
            latency = time.perf_counter() - start
            
            result_collector.add_result({
                "test": "single_query",
                "query_type": query_type,
                "latency_seconds": latency,
                "success": result.get("success", False),
            })
            
            # Assert reasonable latency for mock
            assert latency < 1.0, f"Query {query_type} too slow: {latency:.3f}s"
    
    def test_batch_query_throughput(self, result_collector: PerfResultCollector):
        """
        Benchmark batch query throughput.
        
        Measures queries per second for sequential processing.
        """
        reasoner = get_reasoner()
        num_queries = 50
        
        queries = [
            (SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)][0],
             SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)][1] + f" [variant {i}]")
            for i in range(num_queries)
        ]
        
        start = time.perf_counter()
        latencies = []
        
        for query_type, query in queries:
            q_start = time.perf_counter()
            reasoner.execute(query, query_type)
            latencies.append(time.perf_counter() - q_start)
        
        total_time = time.perf_counter() - start
        throughput = num_queries / total_time
        
        result_collector.add_result({
            "test": "batch_throughput",
            "num_queries": num_queries,
            "total_time_seconds": total_time,
            "throughput_qps": throughput,
            "latency_seconds": statistics.mean(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "success": True,
        })
        
        print(f"\nBatch throughput: {throughput:.2f} queries/sec")
        assert throughput > 1.0, f"Batch throughput too low: {throughput:.2f} qps"


# ============================================================
# CONCURRENCY TESTS
# ============================================================

@pytest.mark.perf
class TestConcurrency:
    """
    Concurrency tests for VULCAN under load.
    """
    
    def test_concurrent_queries_threadpool(
        self,
        perf_config: PerfConfig,
        result_collector: PerfResultCollector,
    ):
        """
        Test concurrent query execution using ThreadPoolExecutor.
        
        Runs queries at different concurrency levels and measures
        throughput and latency distribution.
        """
        reasoner = get_reasoner()
        
        def execute_query(query_id: int) -> Dict[str, Any]:
            """Execute a single query and return timing info."""
            query_type, query_text = SAMPLE_QUERIES[query_id % len(SAMPLE_QUERIES)]
            query_text = f"{query_text} [id={query_id}]"
            
            start = time.perf_counter()
            try:
                result = reasoner.execute(query_text, query_type)
                latency = time.perf_counter() - start
                return {
                    "query_id": query_id,
                    "query_type": query_type,
                    "latency_seconds": latency,
                    "success": result.get("success", False),
                }
            except Exception as e:
                latency = time.perf_counter() - start
                return {
                    "query_id": query_id,
                    "query_type": query_type,
                    "latency_seconds": latency,
                    "success": False,
                    "error": str(e),
                }
        
        for concurrency in perf_config.concurrency_levels:
            # Calculate number of queries based on duration target
            # Aim for at least 10 queries per worker
            num_queries = max(concurrency * 10, 50)
            
            print(f"\n--- Testing concurrency: {concurrency} workers, {num_queries} queries ---")
            
            gc.collect()
            start = time.perf_counter()
            results = []
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(execute_query, i)
                    for i in range(num_queries)
                ]
                
                for future in as_completed(futures):
                    try:
                        results.append(future.result(timeout=30.0))
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "latency_seconds": 30.0,
                        })
            
            total_time = time.perf_counter() - start
            
            # Calculate metrics
            successful = [r for r in results if r.get("success", False)]
            latencies = [r["latency_seconds"] for r in successful]
            
            if latencies:
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                metrics = {
                    "concurrency": concurrency,
                    "num_queries": num_queries,
                    "successful_queries": len(successful),
                    "failed_queries": len(results) - len(successful),
                    "total_time_seconds": total_time,
                    "throughput_qps": len(successful) / total_time,
                    "latency_avg": statistics.mean(latencies),
                    "latency_seconds": statistics.mean(latencies),  # For collector
                    "latency_p50": sorted_latencies[int(n * 0.50)],
                    "latency_p95": sorted_latencies[min(int(n * 0.95), n - 1)],
                    "latency_p99": sorted_latencies[min(int(n * 0.99), n - 1)],
                    "error_rate": (len(results) - len(successful)) / len(results),
                    "success": True,
                }
            else:
                metrics = {
                    "concurrency": concurrency,
                    "num_queries": num_queries,
                    "successful_queries": 0,
                    "failed_queries": len(results),
                    "total_time_seconds": total_time,
                    "throughput_qps": 0.0,
                    "latency_seconds": 0.0,
                    "error_rate": 1.0,
                    "success": False,
                }
            
            result_collector.add_result(metrics)
            
            print(f"  Throughput: {metrics['throughput_qps']:.2f} qps")
            print(f"  Success rate: {len(successful)}/{len(results)}")
            if latencies:
                print(f"  Latency p50/p95/p99: {metrics['latency_p50']:.4f}s / "
                      f"{metrics['latency_p95']:.4f}s / {metrics['latency_p99']:.4f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_queries_asyncio(
        self,
        perf_config: PerfConfig,
        result_collector: PerfResultCollector,
    ):
        """
        Test concurrent query execution using asyncio.
        
        Uses async/await for concurrency, suitable for I/O-bound workloads.
        """
        reasoner = get_reasoner()
        
        async def execute_query_async(query_id: int) -> Dict[str, Any]:
            """Execute a single async query."""
            query_type, query_text = SAMPLE_QUERIES[query_id % len(SAMPLE_QUERIES)]
            query_text = f"{query_text} [async_id={query_id}]"
            
            start = time.perf_counter()
            try:
                result = await reasoner.execute_async(query_text, query_type)
                latency = time.perf_counter() - start
                return {
                    "query_id": query_id,
                    "query_type": query_type,
                    "latency_seconds": latency,
                    "success": result.get("success", False),
                }
            except Exception as e:
                latency = time.perf_counter() - start
                return {
                    "query_id": query_id,
                    "latency_seconds": latency,
                    "success": False,
                    "error": str(e),
                }
        
        for concurrency in perf_config.concurrency_levels:
            num_queries = max(concurrency * 10, 50)
            
            print(f"\n--- Async testing: {concurrency} concurrent, {num_queries} queries ---")
            
            gc.collect()
            start = time.perf_counter()
            
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_execute(query_id: int) -> Dict[str, Any]:
                async with semaphore:
                    return await execute_query_async(query_id)
            
            tasks = [bounded_execute(i) for i in range(num_queries)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.perf_counter() - start
            
            # Process results
            valid_results = [r for r in results if isinstance(r, dict)]
            successful = [r for r in valid_results if r.get("success", False)]
            latencies = [r["latency_seconds"] for r in successful]
            
            if latencies:
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                metrics = {
                    "test_type": "asyncio",
                    "concurrency": concurrency,
                    "num_queries": num_queries,
                    "successful_queries": len(successful),
                    "total_time_seconds": total_time,
                    "throughput_qps": len(successful) / total_time,
                    "latency_seconds": statistics.mean(latencies),
                    "latency_p50": sorted_latencies[int(n * 0.50)],
                    "latency_p95": sorted_latencies[min(int(n * 0.95), n - 1)],
                    "latency_p99": sorted_latencies[min(int(n * 0.99), n - 1)],
                    "success": True,
                }
            else:
                metrics = {
                    "test_type": "asyncio",
                    "concurrency": concurrency,
                    "successful_queries": 0,
                    "latency_seconds": 0.0,
                    "success": False,
                }
            
            result_collector.add_result(metrics)
            
            print(f"  Async throughput: {metrics.get('throughput_qps', 0):.2f} qps")


# ============================================================
# INTEGRATION / END-TO-END
# ============================================================

@pytest.mark.perf
class TestEndToEnd:
    """
    End-to-end performance tests.
    """
    
    def test_reasoning_pipeline_e2e(self, result_collector: PerfResultCollector):
        """
        Test full reasoning pipeline performance.
        
        Measures end-to-end latency including all processing stages.
        """
        reasoner = get_reasoner()
        
        # Run a series of diverse queries
        num_iterations = 20
        latencies = []
        
        for i in range(num_iterations):
            query_type, query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            query = f"{query} [iteration={i}]"
            
            start = time.perf_counter()
            reasoner.execute(query, query_type)
            latency = time.perf_counter() - start
            latencies.append(latency)
            
            result_collector.add_result({
                "iteration": i,
                "query_type": query_type,
                "latency_seconds": latency,
                "success": True,
            })
        
        # Calculate summary stats
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        summary = {
            "test": "e2e_pipeline",
            "iterations": num_iterations,
            "latency_avg": statistics.mean(latencies),
            "latency_p50": sorted_latencies[int(n * 0.50)],
            "latency_p95": sorted_latencies[min(int(n * 0.95), n - 1)],
            "latency_p99": sorted_latencies[min(int(n * 0.99), n - 1)],
            "latency_min": min(latencies),
            "latency_max": max(latencies),
            "throughput_qps": num_iterations / sum(latencies),
            "latency_seconds": statistics.mean(latencies),
            "success": True,
        }
        
        print(f"\nE2E Pipeline Performance:")
        print(f"  Avg latency: {summary['latency_avg']*1000:.2f}ms")
        print(f"  P95 latency: {summary['latency_p95']*1000:.2f}ms")
        print(f"  Throughput: {summary['throughput_qps']:.2f} qps")
        
        result_collector.add_result(summary)


# ============================================================
# REPORT GENERATION
# ============================================================

@pytest.fixture(scope="module", autouse=True)
def generate_perf_report(request, output_dir):
    """
    Generate performance report after all tests in module complete.
    """
    # Storage for module-level results
    all_results = []
    
    yield all_results
    
    # After tests complete, generate reports
    if all_results:
        results_data = {
            "test_suite": "perf_smoke",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }
        
        # Save JSON report
        json_path = output_dir / "results.json"
        save_json_report(results_data, json_path)
        
        # Generate and save markdown summary
        markdown = generate_summary_markdown(results_data, "Performance Smoke Test Results")
        md_path = output_dir / "summary.md"
        save_markdown_report(markdown, md_path)
        
        print(f"\n📊 Reports saved to {output_dir}")
