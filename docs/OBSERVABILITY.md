# Observability & Telemetry

**Version:** 2.2.0  
**Last Updated:** December 23, 2024

> See also: [OPERATIONS.md](OPERATIONS.md) for comprehensive operations guide

## 1. Principles
High signal, bounded cardinality, multi-layer correlation (node → graph → rolling aggregator → alignment overlay).

## 2. Metrics Taxonomy
| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| graphix_nodes_executed | Counter | graph | Node execution count |
| graphix_nodes_failed | Counter | graph | Failure count |
| graphix_total_latency_ms | Gauge | graph | Wall-clock latency |
| graphix_cache_hit_rate | Gauge | graph | Deterministic node cache efficiency |
| graphix_success_rate | Gauge | graph | Per-graph success fraction |
| graphix_throughput_nodes_per_sec | Gauge | graph | Nodes/sec |
| graphix_rss_mb | Gauge | graph | Resident memory |
| graphix_cpu_percent | Gauge | graph | CPU usage |
| graphix_safety_blocks_total | Counter | graph | Safety validation rejections |
| graphix_alignment_conflict_total | Counter | graph | Alignment-based conflict events |
| graphix_consensus_failure_total | Counter | graph | Unreached consensus occurrences |
| graphix_energy_nj_total | Counter | backend | Accumulated energy estimate |
| graphix_photonic_emulation_runs_total | Counter | backend | Photonic emulated dispatches |
| graphix_memristor_emulation_runs_total | Counter | backend | Memristor emulated dispatches |

## 3. Cardinality Control
- Hash long IDs (proposal_id) if long.
- Avoid dynamic free-form labels (user input).
- Limit node_type expansion; experimental gating.

## 4. Tracing
Optional spans: node execution (attributes: node_type, latency_ms, cache_hit); parent graph span linking causal chain.

## 5. Audit & Provenance
Hash-linked audit events; provenance graph indexing output consumption edges for lineage queries.

## 6. Dashboards (Suggested Panels)
- Node latency p95/p99 trend
- Success vs failure counts
- Cache hit rate over time
- Safety block spikes
- Consensus approval latency
- Energy usage trends (hardware dispatch)

## 7. Anomaly Heuristics
- Latency EWMA drift > factor threshold
- Error burst ratio > baseline × multiplier
- Alignment conflict frequency > rolling mean × multiplier
- Cache hit rate deterioration slope negative > threshold

## 8. Performance Optimization via Telemetry
Identify hotspots (node_type latency tail), adjust concurrency, apply batching for micro-transform nodes.

## 9. Data Integrity
Validate non-negative durations; enforce audit chain hash verification; deduplicate events.

## 10. Future Extensions
Distributed tracing across sharded graph workers; live provenance visualization; retention-tiering; ML anomaly classification.
