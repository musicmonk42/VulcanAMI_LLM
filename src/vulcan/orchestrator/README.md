# VULCAN-AMI Orchestrator Module

## Overview

The Orchestrator Module in VULCAN-AMI serves as the central coordination system for AGI agents, managing lifecycle, task distribution, scaling, metrics, and deployment. It implements a full cognitive cycle (perception → reasoning → validation → execution → learning → reflection → self-improvement) with distributed execution via agent pools and task queues. The module supports variants for parallel, fault-tolerant, and adaptive orchestration, ensuring robustness, scalability, and autonomous improvement through experiment generation.

Designed for production readiness, it features no circular dependencies, bounded memory usage, thread-safe operations with **sharded locks for reduced contention**, comprehensive error handling, and Unicode-safe utilities. Self-improvement is integrated via knowledge gap analysis and experiment execution.

## Recent Improvements (2026)

### Performance & Concurrency
- **Sharded Locks in Metrics**: Separate locks for counters, gauges, histograms, and timeseries reduce lock contention in high-throughput scenarios (50+ agents)
- **GC Rate Limiting**: Garbage collection limited to once per minute (generation 0 only) prevents performance degradation
- **Non-Blocking Job Submission**: New polling mode allows orchestrator to manage multiple long-running tasks concurrently

### Reliability & Observability
- **Heartbeat Mechanism**: Jobs signal liveness to distinguish "thinking hard" from "frozen" states
- **Queue Depth Tracking**: Health scores now include job backlog (10% weight) for better system health visibility
- **Main Process Validation**: Prevents Windows multiprocessing issues with singleton pattern

### Production Hardening
- **Atomic Writes with Jitter**: Exponential backoff with ±20% jitter handles antivirus/cloud sync file locking
- **Strict Dependency Mode**: `VULCAN_STRICT_DEPS=1` raises errors instead of degrading silently in production

## Key Features

**Agent Lifecycle Management**: Strict state machine for agent states (e.g., IDLE, WORKING) with validation, provenance tracking, and **heartbeat-based stuck job detection**.

**Agent Pool**: Dynamic spawning, retirement, recovery, and auto-scaling based on load and health. **Main process validation** prevents worker process misuse on Windows. **GC rate limiting** prevents performance spikes.

**Task Queues**: Distributed implementations (Ray, Celery, ZMQ, Custom) with retries, timeouts, and status tracking.

**Collective Orchestration**: Core cognitive loop with modality handling, action selection, and self-improvement drive.

**Metrics Collection**: Counters, gauges, histograms, timeseries with aggregations, health scores (including **queue depth**), and bounded storage. **Sharded locks** enable high-concurrency metric collection.

**Dependencies Management**: Centralized container with validation, factories, and status reporting. **Strict mode** available for production environments.

**Variants**: Specialized orchestrators for parallel processing, fault tolerance (redundancy/retries), and adaptive strategies. **Non-blocking job submission** for managing multiple long-running tasks.

**Deployment**: Production setup with monitoring, checkpointing, **robust atomic saves** (Windows-compatible with jitter), and graceful shutdown.

**Self-Improvement**: Generates and executes experiments to fill knowledge gaps, integrated into the cognitive cycle.

## Architecture and Components

The module is modular, with each file handling a specific aspect, exported via `__init__.py`:

**agent_lifecycle.py**: Defines `AgentState`, `AgentCapability`, `AgentMetadata`, `JobProvenance` (with **heartbeat fields**), and state transition rules.

**agent_pool.py**: Implements `AgentPoolManager` (with **main process validation**) for pool operations, `AutoScaler` for dynamic scaling, `RecoveryManager` for error recovery, and **`AgentPoolProxy`** for worker processes. Includes **GC rate limiting** and **heartbeat-based stuck job detection**.

**collective.py**: Core `VULCANAGICollective` for orchestration, with cognitive steps, agent assignment, and self-improvement logic.

**dependencies.py**: `EnhancedCollectiveDeps` container with factories (`create_minimal_deps`, etc.), validation utilities, and **strict mode** support.

**deployment.py**: `ProductionDeployment` for running orchestrators with monitoring, checkpointing, and **atomic file operations with exponential backoff and jitter**.

**metrics.py**: `EnhancedMetricsCollector` for tracking metrics with **sharded locks**, types (`MetricType`), aggregations (`AggregationType`), and **queue depth tracking**.

**task_queues.py**: `TaskQueueInterface` with implementations (`RayTaskQueue`, etc.) and factory (`create_task_queue`).

**variants.py**: Specialized classes like `ParallelOrchestrator` (with **non-blocking job submission**), `FaultTolerantOrchestrator`, `AdaptiveOrchestrator`.

**`__init__.py`**: Exports all components, provides convenience functions (e.g., `create_orchestrator`), and module documentation.

The system uses **sharded locks** for thread safety, deques for bounded histories, TTL caches for provenance, and fallbacks for optional libraries.

## Installation and Dependencies

This module is part of the VULCAN-AMI project. To use it:

1. Clone the repository (or integrate into your project).

2. Install required dependencies:
   ```bash
   pip install numpy psutil cachetools
   ```

**Core**: logging, threading, multiprocessing, time, uuid, json, hashlib, pathlib, collections, dataclasses, psutil, enum, typing, traceback, asyncio, sys, os, pickle, datetime, concurrent.futures.

**Optional**: ray (distributed), celery (queues), zmq (messaging), numpy (metrics/arrays).

**Fallbacks**: Custom implementations for missing optionals (e.g., dict-based TTL cache).

3. Import the module:
   ```python
   from vulcan.orchestrator import VULCANAGICollective, create_orchestrator
   ```

## Usage Examples

### Basic Usage
```python
import logging
from vulcan.orchestrator import create_orchestrator, create_agent_pool, ModalityType

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create dependencies and config (minimal example)
from vulcan.orchestrator import create_minimal_deps

deps = create_minimal_deps()
config = type('Config', (), {'enable_self_improvement': True})()  # Mock config

# Create agent pool (only in main process!)
pool = create_agent_pool(min_agents=2, max_agents=10)

# Create orchestrator
collective = create_orchestrator('collective', config, deps=deps, agent_pool=pool)

# Process an input
input_data = {"modality": ModalityType.TEXT, "content": "Sample task"}
result = collective.process_input(input_data)

print("Result:", result)

# Get status
status = collective.get_status()
print("System Status:", status)

# Shutdown
collective.shutdown()
```

### Non-Blocking Job Submission (NEW)
```python
from vulcan.orchestrator.variants import ParallelOrchestrator, TimeoutStrategy

# Create orchestrator with polling strategy
orchestrator = ParallelOrchestrator(
    config, deps, 
    timeout_strategy=TimeoutStrategy.POLLING
)

# Submit job without blocking
job_id = orchestrator.step_parallel_nonblocking(input_data)

# Check status
status = orchestrator.get_job_status(job_id)
print(f"Job {job_id}: {status}")

# Wait with timeout
try:
    result = orchestrator.wait_for_job(job_id, timeout_ms=30000)
    print(f"Result: {result}")
except TimeoutError:
    print("Job timed out, cancelling...")
    orchestrator.cancel_job(job_id)
```

### Strict Dependency Mode (NEW)
```python
# Production: Fail fast on missing dependencies
import os
os.environ['VULCAN_STRICT_DEPS'] = '1'

# Or via parameter
deps = create_minimal_deps(
    enable_learning=True,
    strict=True  # Raises ImportError if unavailable
)
```

### Heartbeat Usage (NEW)
```python
# In long-running jobs
from vulcan.orchestrator.agent_lifecycle import create_job_provenance

job = create_job_provenance(
    job_id="job_123",
    agent_id="agent_1",
    # ... other params
)

# Update heartbeat during processing
while processing:
    do_work()
    job.update_heartbeat()  # Signal: still alive

# Check if job is stale
if job.is_stale():
    print(f"Job may be frozen: {job.get_time_since_heartbeat()}s since heartbeat")
```

## Configuration

**Pool Parameters**: Set `min_agents`, `max_agents`, `scale_threshold` in `AgentPoolManager`. **Must be instantiated in main process only**.

**Queue Type**: Choose via `create_task_queue` (e.g., "ray", "celery").

**Metrics**: Tune `max_histogram_size`, `max_timeseries_size`, `max_healthy_queue_depth` in `EnhancedMetricsCollector`.

**Deployment**: Configure `checkpoint_interval`, `checkpoint_dir` in `ProductionDeployment`.

**Self-Improvement**: Enable via `config.enable_self_improvement`; adjust experiment limits.

**Variants**: Select orchestrator type in `create_orchestrator` (e.g., "parallel", "adaptive"). Configure `timeout_strategy` for blocking vs. non-blocking behavior.

**Strict Mode**: Set `VULCAN_STRICT_DEPS=1` environment variable or pass `strict=True` to dependency factories.

## Performance Tuning

### High-Concurrency Scenarios (50+ Agents)
- **Sharded locks** in metrics automatically reduce contention
- Adjust `max_healthy_queue_depth` to match your workload
- Use `TimeoutStrategy.POLLING` for long-running jobs

### Memory Management
- **GC rate limiting** prevents frequent collection spikes
- Bounded deques prevent unbounded growth
- Call `metrics.record_queue_depth()` for backlog tracking

### Job Reliability
- **Heartbeat mechanism** detects frozen jobs early
- Stuck jobs detected via `agent_pool.get_stuck_jobs()`
- Use `JobProvenance.is_stale()` for custom logic

## Notes

**Thread Safety**: All critical sections use appropriate locks (**sharded** where beneficial); supports concurrent operations and async (asyncio).

**Error Handling**: Custom exceptions (PerceptionError, etc.), retries, fallbacks, and **atomic operations with jitter**.

**Performance**: Bounded data structures prevent OOM; **sharded locks** reduce contention; monitors rates, percentiles, and health scores (including **queue depth**).

**Extensibility**: Extend enums (e.g., `AgentCapability`), factories, or subclass orchestrators.

**Limitations**: Optionals required for advanced features (e.g., Ray for distributed); Windows file handling for checkpoints.

**Platform Compatibility**: 
- **Windows**: Main process validation prevents multiprocessing issues
- **All Platforms**: Atomic writes handle file locking from antivirus/cloud sync

For contributions or issues, refer to the VULCAN-AMI project repository.

