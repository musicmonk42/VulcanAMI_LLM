# VULCAN Performance Testing Guide

This guide covers the performance and boundedness testing infrastructure for
the VULCAN cognitive architecture. These tests ensure system scalability,
memory efficiency, and regression-free releases.

## Overview

The performance test suite provides:

1. **Performance Smoke Tests**: Quick micro-benchmarks and concurrency tests
2. **Boundedness Tests**: Memory leak detection and state growth monitoring
3. **Regression Detection**: Automatic comparison against baseline metrics

## Quick Start

### Run Performance Tests Locally

```bash
# Install perf-lite dependencies (fast, no heavy ML deps)
pip install -e ".[perf-lite]"

# Run all performance tests
pytest tests/perf/ -m perf -v

# Run only boundedness tests
pytest tests/perf/ -m boundedness -v

# Run with custom configuration
PERF_ITERATIONS=1000 PERF_MAX_RSS_GROWTH_MB=100 pytest tests/perf/ -m perf -v
```

### Run with Full Dependencies (Nightly)

```bash
# Install full dependencies including torch, faiss-cpu, sentence-transformers
pip install -e ".[perf-full]"

# Run extended performance tests
pytest tests/perf/ -m perf -v --timeout=600
```

## Configuration

All thresholds and parameters are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_MAX_RSS_GROWTH_MB` | 50 | Maximum allowed RSS memory growth (MB) |
| `PERF_MAX_SLOWDOWN_PCT` | 20 | Maximum allowed performance slowdown (%) |
| `PERF_MAX_P95_REGRESSION_PCT` | 25 | Maximum p95 latency regression vs baseline (%) |
| `PERF_MAX_RPS_REGRESSION_PCT` | 25 | Maximum throughput regression vs baseline (%) |
| `PERF_ITERATIONS` | 500 | Number of iterations for boundedness tests |
| `PERF_CONCURRENCY_LEVELS` | "10,25,50" | Concurrency levels to test |
| `PERF_CONCURRENCY_DURATION` | 30 | Duration for each concurrency test (seconds) |
| `PERF_OUTPUT_DIR` | "perf" | Output directory for reports |

## Test Categories

### Performance Smoke Tests (`test_perf_smoke.py`)

These tests measure throughput and latency under various conditions:

#### Micro-Benchmarks
- **Reasoner Warmup**: Measures initialization and warmup time
- **Single Query Latency**: Tests latency for different query types
- **Batch Throughput**: Measures queries per second for sequential processing

#### Concurrency Tests
- **ThreadPool Concurrency**: Tests parallel execution using ThreadPoolExecutor
- **AsyncIO Concurrency**: Tests async/await based concurrency

#### End-to-End Tests
- **Full Pipeline**: Measures complete reasoning pipeline performance

### Boundedness Tests (`test_boundedness.py`)

These tests ensure the system doesn't leak memory or degrade over time:

#### Memory Tracking
- **RSS Growth Test**: Verifies memory stays bounded over many iterations
- **Leak Detection**: Tests for leaks in init/cleanup cycles

#### Performance Stability
- **Iteration Time Trend**: Detects gradual performance degradation
- **History Structure Bounds**: Verifies caches and buffers stay bounded

### Dependency Tests (`test_deps.py`)

Verifies expected dependencies are available:
- **psutil**: Required for memory tracking
- **perf-lite deps**: Core performance test dependencies
- **perf-full deps**: Extended ML dependencies (optional)

## Reports and Artifacts

### Generated Files

After running tests, the following files are generated in the `perf/` directory:

| File | Description |
|------|-------------|
| `results.json` | Detailed performance test results |
| `boundedness.json` | Memory and stability test results |
| `summary.md` | Markdown summary of performance results |
| `boundedness.md` | Markdown summary of boundedness results |
| `comparison.md` | Regression comparison report |
| `comparison.json` | Structured regression data |

### CI Artifacts

In GitHub Actions, artifacts are uploaded as `perf-smoke-results-{run_number}`:
- All JSON reports
- All Markdown summaries
- JUnit XML test results
- Test output logs

## Baseline Management

### Understanding Baselines

The baseline file (`perf/baseline.json`) defines expected performance:

```json
{
  "benchmarks": {
    "end_to_end": {
      "pipeline": {
        "avg_latency_max_seconds": 0.5,
        "p95_latency_max_seconds": 1.0,
        "min_throughput_qps": 2.0
      }
    },
    "concurrency": {
      "threadpool": {
        "concurrency_10": {
          "min_throughput_qps": 5.0,
          "max_latency_p95_seconds": 2.0,
          "min_success_rate": 0.95
        }
      }
    }
  }
}
```

### Updating Baselines

To update baselines after verified performance improvements:

1. Run performance tests with current code:
   ```bash
   pytest tests/perf/ -m perf -v
   ```

2. Review results in `perf/results.json`

3. Update `perf/baseline.json` with new values

4. Commit the updated baseline

### Comparing Against Baseline

```bash
# Compare current results against baseline
python scripts/compare_perf.py \
  --results perf/results.json \
  --baseline perf/baseline.json \
  --output perf/comparison.md

# Informational run (don't fail on regression)
python scripts/compare_perf.py --no-fail
```

## Interpreting Results

### Latency Metrics

- **p50 (Median)**: Half of requests complete faster than this
- **p95**: 95% of requests complete faster (important for SLAs)
- **p99**: 99% of requests complete faster (worst case)

**Good Results:**
- p95 latency < baseline threshold
- Low variance between p50 and p99
- Consistent latency across concurrency levels

### Throughput Metrics

- **QPS**: Queries per second
- **Success Rate**: Percentage of successful queries

**Good Results:**
- Throughput ≥ baseline minimum
- Success rate ≥ 95%
- Linear scaling with concurrency (up to resource limits)

### Memory Metrics

- **RSS Growth**: Increase in resident set size over test duration
- **Max RSS**: Peak memory usage

**Good Results:**
- RSS growth < threshold (default: 50 MB)
- No continuous growth pattern (indicates leak)

### Performance Stability

- **Slowdown %**: Change in latency from early to late iterations

**Good Results:**
- Slowdown < threshold (default: 20%)
- No increasing trend in iteration times

## CI Integration

### Automatic Testing

Performance tests run automatically on:
- Push to main/develop branches
- Pull requests to main/develop
- Manual workflow dispatch

### Configuring CI Thresholds

Override thresholds in CI via workflow inputs or environment variables:

```yaml
env:
  PERF_ITERATIONS: '100'  # Fewer iterations for faster CI
  PERF_CONCURRENCY_LEVELS: '10,25'  # Skip highest concurrency
  PERF_MAX_RSS_GROWTH_MB: '75'  # More lenient for CI
```

### Handling Flaky Tests

If tests are flaky in CI:

1. Increase thresholds slightly for CI
2. Use environment variables to configure
3. Add `continue-on-error: true` during rollout
4. Review artifact logs for patterns

## Troubleshooting

### "psutil not available"

```bash
pip install psutil
```

Memory tracking requires psutil. Tests will skip memory assertions without it.

### "No baseline for comparison"

The comparison script needs both a baseline and results file:

1. Run tests first: `pytest tests/perf/ -m perf`
2. Ensure baseline exists: `perf/baseline.json`

### High Memory Growth

If RSS growth exceeds threshold:

1. Check for leaked references in test fixtures
2. Look for unbounded caches or buffers
3. Verify proper cleanup in teardown
4. Run with `PERF_ITERATIONS=5000` to confirm

### Performance Degradation

If tests show degradation over iterations:

1. Check for memory pressure (GC pauses)
2. Look for cache invalidation issues
3. Verify no accumulating state
4. Profile with `py-spy` or `cProfile`

## Best Practices

### Writing New Performance Tests

1. Use `@pytest.mark.perf` for performance tests
2. Use `@pytest.mark.boundedness` for memory tests
3. Accept `perf_config` fixture for configuration
4. Report results via `result_collector` fixture
5. Use `memory_tracker` for RSS monitoring

### Example Test

```python
@pytest.mark.perf
def test_my_component_throughput(
    perf_config: PerfConfig,
    result_collector: PerfResultCollector,
):
    """Test throughput of my component."""
    num_iterations = perf_config.iterations

    for i in range(num_iterations):
        start = time.perf_counter()
        result = my_component.process(input_data)
        latency = time.perf_counter() - start

        result_collector.add_result({
            "iteration": i,
            "latency_seconds": latency,
            "success": result.success,
        })

    # Assertions
    latencies = result_collector.get_latencies()
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 1.0, f"Average latency too high: {avg_latency}"
```

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md): General testing practices
- [CI/CD Guide](CI_CD.md): Continuous integration setup
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md): System architecture
