# tests/perf/__init__.py
"""
VULCAN Performance Test Suite.

This package contains performance and boundedness tests for the VULCAN cognitive
architecture. Tests are designed to run in CI environments with configurable
thresholds via environment variables.

Test Categories:
    - perf_smoke_test.py: Micro-benchmarks and concurrency tests
    - boundedness_test.py: Memory leak and history growth detection
    - conftest.py: Shared fixtures and configuration

Configuration:
    Tests can be configured via environment variables:
    - PERF_MAX_RSS_GROWTH_MB: Max allowed RSS growth (default: 50)
    - PERF_MAX_SLOWDOWN_PCT: Max allowed slowdown over iterations (default: 20)
    - PERF_MAX_P95_REGRESSION_PCT: Max p95 latency regression (default: 25)
    - PERF_MAX_RPS_REGRESSION_PCT: Max throughput regression (default: 25)
    - PERF_ITERATIONS: Number of iterations for boundedness tests (default: 500)
    - PERF_CONCURRENCY_LEVELS: Concurrency levels to test (default: "10,25,50")

Usage:
    # Run all perf tests
    pytest tests/perf/ -m perf

    # Run only boundedness tests
    pytest tests/perf/ -m boundedness

    # Run with custom thresholds
    PERF_MAX_RSS_GROWTH_MB=100 pytest tests/perf/ -m perf
"""

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"
