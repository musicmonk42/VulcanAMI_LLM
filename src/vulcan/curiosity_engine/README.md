Curiosity Engine

Overview

The Curiosity Engine is a core module of the VULCAN-AMI system, designed to drive curiosity-based learning and exploration. It identifies knowledge gaps in the system's understanding, analyzes dependencies between these gaps, generates targeted experiments to address them, and manages resource budgets for efficient exploration. The engine follows a structured EXAMINE → SELECT → APPLY → REMEMBER (ESA-R) pattern to ensure systematic and adaptive learning.

This module enables AI systems to proactively seek new knowledge, detect anomalies, and iteratively improve through simulated or real-world experiments, all while respecting resource constraints.

Key Features

- Knowledge Gap Detection: Analyzes failures, predictions, and patterns to identify gaps (e.g., causal, latent, decomposition).
- Dependency Management: Builds and analyzes graphs of gap dependencies to prioritize learning paths and avoid cycles.
- Experiment Generation: Designs iterative, adaptive experiments tailored to specific gaps, with safety constraints and pivoting on failures.
- Resource Budgeting: Dynamically allocates and recovers budgets based on system load, historical costs, and efficiency metrics.
- Orchestration: Coordinates the full curiosity loop, from gap prioritization to knowledge integration.
- Cross-Process Persistence: SQLite-based bridges preserve resolution state and query outcomes across subprocess invocations, preventing phantom resolutions and enabling pattern detection.

Architecture and Components

The module is composed of several interconnected Python files, each handling a separated concern:

- **curiosity_engine_core.py**: The main orchestrator. Manages the learning cycle, integrates components, and runs experiments in sandboxed environments. Key classes: CuriosityEngine, RegionManager, ExplorationValueEstimator.

- **gap_analyzer.py**: Identifies and analyzes knowledge gaps from failures and patterns. Supports types like decomposition, causal, latent, and transfer. Key classes: GapAnalyzer, KnowledgeGap, LatentGap, AnomalyAnalyzer.

- **dependency_graph.py**: Models dependencies between gaps as a directed graph, detects cycles, and calculates adjusted ROI (Return on Investment) for prioritization. Key classes: CycleAwareDependencyGraph, DependencyAnalyzer, ROICalculator.

- **experiment_generator.py**: Generates and iterates on experiments to fill gaps, with failure analysis and adaptive pivoting. Supports types like causal, transfer, and exploratory. Key classes: ExperimentGenerator, Experiment, IterativeExperimentDesigner.

- **exploration_budget.py**: Manages dynamic budgets, monitors system resources (CPU, memory, etc.), and calibrates cost estimates from historical data. Key classes: DynamicBudget, ResourceMonitor, CostEstimator.

- **resolution_bridge.py**: SQLite-based persistence layer for gap resolution state across subprocess invocations. Solves the "phantom resolution" problem where gaps were resolved 40-90 times/hour due to subprocess memory loss. Features:
 - Atomic phantom detection in single transactions (prevents race conditions)
 - TTL-based resolution expiration (30-min default)
 - Phantom resolution detection (>3 resolutions/hour triggers 1-hour cooldown)
 - Experiment counters to prevent false cold-start detection
 - Key functions: `mark_gap_resolved()`, `is_gap_resolved()`, `is_phantom_resolution()`

- **outcome_bridge.py**: SQLite-based persistence layer for query execution outcomes across subprocesses. Enables pattern detection and gap analysis from historical query performance data. Features:
 - Deduplicate-on-insert via `INSERT OR REPLACE` (prevents database bloat)
 - Error rate aggregation by domain and time window
 - Slow query detection with configurable thresholds
 - Comprehensive statistics for monitoring
 - Key functions: `record_query_outcome()`, `get_error_rate()`, `get_outcome_statistics()`

- **__init__.py**: Empty module initializer (for package structure).

### SQLite Bridge Architecture

The SQLite bridges solve critical cross-process memory loss issues:

**Problem**: The CuriosityEngine runs learning cycles in subprocesses for CPU isolation. Each subprocess creates a fresh engine instance with no memory of previous resolutions, causing:
- Phantom resolutions (gaps "resolved" 40-90 times/hour without actual fixes)
- False cold starts (thinking 0/5 experiments ran each cycle)
- Wasted computation on already-addressed gaps

**Solution**: Persistent SQLite databases with atomic operations:
- `resolution_bridge.py` tracks which gaps have been resolved and when
- `outcome_bridge.py` tracks query execution outcomes for pattern detection
- Both use WAL mode for concurrent access and atomic transactions to prevent race conditions

**Recent Fixes** (2025-01-06):
1. **Atomic phantom detection**: Combined phantom check + cooldown check + insert in single transaction (fixes race condition in `mark_gap_resolved()`)
2. **Fixed threshold logic**: Changed `count >= threshold` to `count > threshold` (fixes off-by-one error allowing exactly 3 resolutions before phantom flag)
3. **Enhanced gap key normalization**: Added validation to prevent key mismatches
4. **Deduplicate-on-insert**: Changed from post-analysis deduplication to `INSERT OR REPLACE` at insert time (prevents database bloat and ensures consistent counts)

The system uses thread-safe locks for concurrency, caching for performance, and optional libraries like NumPy, SciPy, and scikit-learn for advanced analytics (with fallbacks if unavailable).

Installation and Dependencies

This module is part of the larger VULCAN-AMI project. To use it:

1. Clone the repository (or integrate into your project).

2. Install required dependencies:
 ```
 pip install numpy scipy scikit-learn networkx psutil
 ```

 - Core: numpy, logging, typing, dataclasses, collections, queue, threading, sqlite3
 - Optional: scipy (stats), sklearn (anomaly detection), networkx (graph algorithms), psutil (resource monitoring)
 - Fallbacks are provided for missing optional libraries.

3. Import the module:
 ```python
 from vulcan.curiosity_engine import CuriosityEngine
 ```

Usage Example

```python
import logging
from vulcan.curiosity_engine import CuriosityEngine

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the engine with custom parameters
engine = CuriosityEngine(
 base_exploration_budget=100.0,
 max_experiments_per_cycle=5,
 anomaly_threshold=0.2
)

# Run a learning cycle
cycle_summary = engine.run_learning_cycle(max_experiments=3)
print("Learning Cycle Summary:", cycle_summary)

# Manually record a failure for analysis
engine.record_failure(
 failure_type="prediction",
 failure_data={"domain": "physics", "error": 0.45}
)

# Get identified gaps
gaps = engine.get_all_gaps()
print("Identified Gaps:", [gap.type for gap in gaps])
```

### Using SQLite Bridges Directly

```python
from vulcan.curiosity_engine.resolution_bridge import (
 mark_gap_resolved,
 is_gap_resolved,
 is_phantom_resolution,
 get_resolution_statistics
)

# Check if gap was already resolved (persists across subprocesses)
if is_gap_resolved("high_error_rate:query_processing"):
 print("Gap already addressed, skipping")
else:
 # Run experiment...
 # Mark as resolved after success
 mark_gap_resolved("high_error_rate:query_processing", success=True)

# Check for phantom resolution pattern
if is_phantom_resolution("high_error_rate:query_processing"):
 print("Warning: Gap shows phantom resolution - underlying issue not fixed")

# Get statistics
stats = get_resolution_statistics()
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Phantom resolutions: {stats.phantom_count}")
```

```python
from vulcan.curiosity_engine.outcome_bridge import (
 record_query_outcome,
 get_error_rate,
 get_outcome_statistics
)

# Record query outcome (automatically deduplicated)
record_query_outcome(
 query_id="q_12345",
 success=True,
 execution_time_ms=45.2,
 domain="query_processing"
)

# Get error rate for domain in last hour
error_rate = get_error_rate(domain="query_processing", window_seconds=3600)
if error_rate > 0.15:
 print(f"High error rate detected: {error_rate:.1%}")

# Get comprehensive statistics
stats = get_outcome_statistics(domain="query_processing")
print(f"Median query time: {stats.median_execution_time_ms:.1f}ms")
print(f"P95 query time: {stats.p95_execution_time_ms:.1f}ms")
```

Configuration

- **Budget Parameters**: Adjust `base_allocation`, `recovery_rate`, and `adjustment_rate` in DynamicBudget.
- **Thresholds**: Set `anomaly_threshold` in GapAnalyzer or `min_frequency` for pattern detection.
- **Graph Limits**: Configure `max_nodes` and `max_edges` in GraphStorage for large-scale graphs.
- **Caching**: TTLs and sizes are tunable in classes like CostEstimator and ROICalculator.
- **SQLite Bridge Configuration**: Environment variables:
 - `VULCAN_RESOLUTION_DB_PATH`: Custom database path for resolution bridge
 - `VULCAN_RESOLUTION_TTL`: Resolution TTL in seconds (default: 1800 = 30 min)
 - `VULCAN_PHANTOM_THRESHOLD`: Phantom resolution threshold (default: 3)
 - `VULCAN_PHANTOM_WINDOW`: Phantom detection window in seconds (default: 3600 = 1 hour)
 - `VULCAN_OUTCOME_DB_PATH`: Custom database path for outcome bridge
 - `VULCAN_OUTCOME_TTL`: Outcome TTL in seconds (default: 86400 = 24 hours)

Notes

- **Thread Safety**: All major operations are locked for multi-threaded environments. SQLite bridges use WAL mode for concurrent access.
- **Error Handling**: Robust fallbacks for missing libraries and edge cases (e.g., empty sets, division by zero). SQLite operations return sensible defaults on error.
- **Performance**: Uses caching (lru_cache, deques with maxlen) and efficient algorithms (e.g., BFS for paths, NetworkX for graph ops). SQLite bridges use indexes and bounded queries.
- **Extensibility**: Add custom gap types, experiment strategies, or resource types by extending enums and classes.
- **Subprocess Isolation**: The engine runs experiments in subprocesses for CPU isolation. SQLite bridges ensure state persists across process boundaries.
- **Phantom Resolution Prevention**: The system automatically detects when gaps are being "resolved" repeatedly without actual fixes and applies cooldown periods.
- **Database Cleanup**: Both bridges support automatic cleanup of old data via `cleanup_old_data()` functions.

Troubleshooting

**Phantom Resolutions Detected**:
- Check logs for gaps being resolved >3 times/hour
- Verify experiments are actually fixing underlying issues
- Review `resolution_history` table to see resolution patterns
- Consider increasing `VULCAN_PHANTOM_THRESHOLD` if legitimate rapid fixes occur

**Database Lock Errors**:
- Ensure WAL mode is enabled (automatic on initialization)
- Check for long-running transactions blocking writes
- Verify proper connection cleanup (use context managers)

**Missing Resolution State After Restart**:
- Verify database file permissions (should be readable/writable)
- Check database path (defaults to `~/.local/share/vulcan/`)
- Ensure `_init_db()` completes successfully at module import

**High Error Rates**:
- Use `get_outcome_statistics()` to analyze error patterns
- Check `error_types` dictionary for specific failure modes
- Use `get_recent_outcomes(error_only=True)` to examine failed queries

For contributions or issues, refer to the VULCAN-AMI project repository.
