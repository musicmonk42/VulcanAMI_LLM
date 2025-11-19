# Execution Engine Mechanics

## 1. Goals
Deterministic concurrency, failure containment, adaptive resource usage, streaming insights.

## 2. Scheduling (Parallel)
- Build dependency map (node → prerequisites)
- Ready set = nodes whose dependencies succeeded
- Launch tasks (bounded by max_parallel - currently executing)
- FIRST_COMPLETED wait, process results, update executed/failed sets

## 3. Streaming
Periodic yield on interval or node completion, includes intermediate outputs and partial errors.

## 4. Node Execution Steps
1. Cache lookup (deterministic candidates only)
2. Gather inputs (edge port mapping)
3. Handler resolution (runtime.get_node_executor)
4. Execution (async or thread pool for sync functions)
5. Output serialization sanity check
6. Cache store (deterministic only)
7. Metrics & audit emission

## 5. Failure Containment
- Critical node fail → halt remaining execution (status CANCELLED or FAILED).
- Soft failure nodes flagged allow downstream optional degradation (future shaping).

## 6. Timeouts
Global graph timeout via asyncio.wait_for wrapper; node-level durations collected for adaptive analysis (historical percentiles future optimization).

## 7. Caching Strategy
Key: md5(node_type + params + sorted hashed dependency outputs).
Negative caching (future) to prevent repeated expensive failures.

## 8. Metrics Integration
record_node_start / record_node_end update ExecutionMetrics with per-node granular stats and aggregated throughput, success_rate, cache_hit_rate.

## 9. Pseudocode (Condensed)
```python
while remaining:
  ready = scheduler.get_ready()
  start_tasks(ready)
  done = await wait_first_completed(tasks)
  for task in done:
     result = process(task)
     update scheduler & context
  if critical failure: break
finalize result object
```

## 10. Optimization Targets
- Larger layer batching for uniform compute handlers
- Work stealing (future)
- Node reuse detection (common subgraph caching)
- Speculative execution for optional branches (future design)

## 11. Future Enhancements
Distributed actor scatter/gather, energy-aware concurrency scaling, predictive timeout ML layer.
