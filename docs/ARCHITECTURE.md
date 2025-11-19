# Architecture (Comprehensive Deep Dive)

## 1. Conceptual Stratification
Layers:
1. Governance & Evolution Substrate
2. Validation Pipeline (Structure → Ontology → Semantics → Resource → Security → Alignment/Safety)
3. Execution Engine (Mode-aware scheduler)
4. Extension & Integration (AI Runtime, Hardware Dispatcher, Runtime Extensions)
5. Observability & Audit & Provenance
6. Autonomous Optimization & Pattern Learning
7. VULCAN World Model Bridge (Motivational & Safety Semantics)

Each layer publishes discrete metrics, exposes standardized error/value envelopes, and supports cryptographically verifiable transitions.

## 2. Graph IR Canonical Shape
```json
{
  "nodes": [
    { "id": "string", "type": "string", "params": {}, "critical": false }
  ],
  "edges": [
    { "from": "nodeOrObj", "to": "nodeOrObj", "metadata": {} }
  ],
  "metadata": {
    "version": "semver",
    "authors": ["agent_id"],
    "risk_profile": {},
    "lineage": [],
    "_semantic_transfers": []
  }
}
```

### Node Behavioral Classes
- Pure (ADD, MULTIPLY)
- I/O (INPUT, OUTPUT)
- AI External (EMBED, GenerativeAINode)
- Hardware Accelerated (PHOTONIC_MVM, MEMRISTOR_MVM, SPARSE_MVM)
- Governance (ProposalNode, ConsensusNode, ValidationNode, ContractNode)
- Recursive / Composite (MetaGraphNode, COMPOSITE)
- Search / Optimization (SearchNode, HyperParamNode)
- Scheduler / Temporal (SchedulerNode)
- Utility / Data (NormalizeNode, GET_PROPERTY)
- Disabled Dangerous (ExecuteNode placeholder)

## 3. Validation Pipeline Stages
| Stage | Algorithm | Key Checks | Failure Classification |
|-------|-----------|------------|------------------------|
| Structure | JSON shape, list typing | nodes/edges presence | STRUCTURE_INVALID |
| Identity | Duplicate elimination | unique node IDs | NODE_INVALID |
| Edges | Endpoint resolution | source/target existence | EDGE_INVALID |
| Ontology | Enum or concept map membership | type validity | SEMANTIC_INVALID (warning vs error) |
| Semantics | Param/property constraints vs ontology | unexpected properties | SEMANTIC_INVALID (warning) |
| Cycles | DAG enforcement (dependency edges) | cycle detection | CYCLE_DETECTED (warning if soft edges) |
| Resources | Heuristic memory/time/size | param size & counts | RESOURCE_EXCEEDED |
| Security | Pattern regex scan | eval/exec/path traversal | SECURITY_VIOLATION |
| Alignment (VULCAN) | Motivational introspection | proposal alignment score | SECURITY_VIOLATION / SEMANTIC_INVALID |
| Safety | world_model.safety_validator | violation list | SECURITY_VIOLATION |

TTL Caching keyed by md5(normalized_graph) + ontology_version + validator_version.

## 4. Execution Engine Modalities
Modes: SEQUENTIAL, PARALLEL, STREAMING, BATCH.
- Parallel: Layer derivation via dependency sets; concurrency bound by semaphore & adaptive backoff.
- Streaming: Interleaved partial snapshots (RUNNING result frames) with yield_interval gating.
- Batch: Partition graphs into batch_size; handle oversize by recursive split.

Timeout Envelope: global graph timeout_seconds (config), node-level measured durations with dynamic caching.

## 5. Node Execution Data Flow
- Inputs aggregated from upstream node outputs with port mapping (edge `to.port`).
- Handler receives: (node, context_dict, inputs_dict).
- Output stored in context.outputs under node_id, optionally caching deterministic results.

Determinism Filter: Node types flagged non-deterministic excluded from execution cache.

## 6. Metrics Fabric
Per-node: start_ms, end_ms, duration_ms, status, cache_hit.
Per-graph: nodes_executed, nodes_succeeded, nodes_failed, latency, throughput_nodes_per_sec, cache_hit_rate.
Aggregated: Rolling sums, success_rate trends, resource_end snapshot (rss_mb, cpu_percent).

## 7. Hardware Dispatcher Integration
Backend Profiles = latency_ms, throughput_tops, energy_per_op_nj, accuracy, max_tensor_size_mb, health_score.
Strategy Adapters: FASTEST, LOWEST_ENERGY, BEST_ACCURACY, MOST_EFFICIENT, BALANCED (weighted average scoring).
Fallback Sequence: Real Hardware → Emulator → CPU pure computation.

## 8. AI Runtime Layer
Provider Abstraction: execute(task, contract) async; rate limiting & keyed caching; latency/cost SLA evaluation sets warnings or failure codes.
Operations polymorphism: EMBED, GENERATE, CLASSIFY; classification layered over generation mock path.

## 9. Runtime Extensions
SubgraphLearner: pattern id generation (md5(type+graph_json)).
AutonomousOptimizer: fitness_score calculation & triggers for evolution proposals.
ExecutionExplainer: ExplanationType-driven summarization with interpretability augmentation.

## 10. Governance & Evolution Coupling
Proposal gating order: structural validation precedes trust-weighted voting; alignment & safety pre-apply; similarity dampening prevents spam duplicates.
Consensus threshold dynamic within configured min/max bounds; critical proposals require elevated confidence.

## 11. VULCAN Bridge
Stages:
1. Structural prelim
2. Proposal extraction
3. Parallel motivational validation
4. Consensus aggregation
5. Safety validator
6. Semantic transfers
7. Delegated execution
8. Post-run world_model observation

## 12. Observability & Provenance
Audit chain: prev_hash + serialized_event.
Provenance edges track node output consumption lineage; graph-level analytic queries reconstruct causal chains.

## 13. Anti-Patterns & Guardrails
| Anti-Pattern | Risk | Mitigation |
|--------------|------|------------|
| Untyped params ingestion | Runtime errors & silent misuse | Strict schema & semantic property filtering |
| Hidden side effects | Corrupted reproducibility | Mandatory declaration & audit wrapping |
| Resource over-subscription | Memory exhaustion | Pre-run heuristic + continuous resource snapshots |
| Unbounded retries | Thundering herd / cost blow-up | Retry caps + backoff + circuit breakers |
| Critical node silent failures | Partial invalid outputs | Critical flag halts, audit error & fail-fast |

## 14. Scaling Trajectory
Single-process intended targets: 10k nodes / 100k edges (assuming moderate param complexity).
Distributed evolution (future): Sharded execution graphs, remote trace correlation, multi-instance governance consensus.

## 15. Future Research Vectors
- Formal invariant spec (temporal logic).
- ML-based dynamic timeout predictors.
- Energy-aware scheduling objective multi-optimization.
- Semantic graph embedding anomaly detection.
- Policy DSL integration (declarative safety signatures).
