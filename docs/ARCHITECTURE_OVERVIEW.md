# VulcanAMI Platform Architecture Overview

**Version:** 2.2.0  
**Last Updated:** December 23, 2024

This document provides a comprehensive architectural deep dive into the VulcanAMI/GraphixVulcan platform, covering all layers from conceptual stratification to implementation details.

---

## Table of Contents

1. [Conceptual Stratification](#1-conceptual-stratification)
2. [Graph IR Canonical Shape](#2-graph-ir-canonical-shape)
3. [Validation Pipeline Stages](#3-validation-pipeline-stages)
4. [Execution Engine Modalities](#4-execution-engine-modalities)
5. [Node Execution Data Flow](#5-node-execution-data-flow)
6. [Metrics Fabric](#6-metrics-fabric)
7. [Hardware Dispatcher Integration](#7-hardware-dispatcher-integration)
8. [AI Runtime Layer](#8-ai-runtime-layer)
9. [Runtime Extensions](#9-runtime-extensions)
10. [Governance & Evolution Coupling](#10-governance--evolution-coupling)
11. [VULCAN Bridge](#11-vulcan-bridge)
12. [Observability & Provenance](#12-observability--provenance)
13. [Anti-Patterns & Guardrails](#13-anti-patterns--guardrails)
14. [Scaling Trajectory](#14-scaling-trajectory)
15. [Future Research Vectors](#15-future-research-vectors)
16. [Platform Components](#16-platform-components)
17. [Component Integration](#17-component-integration)

---

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

### 14.1 Demo-stage horizontal scaling story
- A single container image can be replicated behind a load balancer; horizontal scale = **multiple replicas of the same image** with externalized state (DB/object store/vector index).
- Current demo topology: one container (API + workers) as a single instance. This supports Railway-style deployments and quick iteration.
- Phase 2 decomposition: split into API gateway, worker/agent pool, memory/vector service, and queue/object store so pools can scale independently.
- Portability evidence: the platform is deployed on Railway in two regions today; this is deployment portability, not global HA.

## 15. Future Research Vectors
- Formal invariant spec (temporal logic).
- ML-based dynamic timeout predictors.
- Energy-aware scheduling objective multi-optimization.
- Semantic graph embedding anomaly detection.
- Policy DSL integration (declarative safety signatures).

---

## 16. Platform Components

The VulcanAMI platform integrates multiple sophisticated components:

### 16.1 GraphixIR Compiler Layer

**Location:** `src/compiler/graph_compiler.py`  
**Purpose:** Compiles JSON graph representations to optimized native machine code via LLVM

**Key Features:**
- **10-100x performance** vs interpreted execution
- **Heterogeneous hardware support** (CPU, GPU, photonic, memristor)
- **Graph-level optimizations**: Operation fusion, dead code elimination, CSE

**Optimization Pipeline:**
1. **Operation Fusion**: Conv2D → BatchNorm → ReLU ⟹ Fused_Conv_BN_ReLU
2. **Dead Code Elimination**: Removes unused nodes and edges
3. **Constant Folding**: Compile-time computation
4. **Common Subexpression Elimination (CSE)**: Reuse computation

### 16.2 LLM Core

**Location:** `src/llm_core/`  
**Purpose:** Custom transformer with graph execution integration

**Components:**
- `graphix_transformer.py` - Main transformer (913 LOC)
- `graphix_executor.py` - Execution engine (1,166 LOC)
- `persistant_context.py` - Context management (857 LOC)

**Features:**
- LoRA (Low-Rank Adaptation) fine-tuning
- Gradient checkpointing for memory efficiency
- Top-P sampling for generation
- IR caching for performance

### 16.3 Persistent Memory v46

**Location:** `src/persistant_memory_v46/`  
**Purpose:** Advanced storage with privacy-preserving unlearning

**Components:**
- **Graph RAG**: Multi-strategy retrieval (dense, sparse, graph-based)
- **Merkle LSM Tree**: High-performance key-value store with versioning
- **Machine Unlearning**: 4 methods (Gradient Surgery, SISA, Influence Functions, Amnesiac)
- **Zero-Knowledge Proofs**: Groth16 zk-SNARKs for unlearning verification

### 16.4 VULCAN-AGI Cognitive Architecture

**Location:** `src/vulcan/`  
**Scale:** 256 files, 13,304 functions, 285,000+ LOC

**Subsystems:**
- **World Model**: Causal reasoning, state prediction, intervention management
  - **SystemObserver**: Converts system events (queries, engine results, validation failures) into observations that feed the causal learning system
  - **Routing Recommendations**: Provides learned routing suggestions based on historical patterns
  - **Performance Introspection**: Self-awareness of system capabilities and known issues
- **Meta-Reasoning**: Self-improvement, motivational introspection, CSIU framework
- **Reasoning Systems**: Symbolic, causal, analogical, multimodal, probabilistic
- **Memory Systems**: Hierarchical, distributed, episodic, semantic, working

---

## 17. Component Integration

### 17.1 Cognitive Cycle (EXAMINE → SELECT → APPLY → REMEMBER)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GRAPHIX-VULCAN BRIDGE (Orchestration)                    │
└───┬─────────────────────────────────────────────────────────┘
    ↓
┌───▼─────────────────────────────────────────────────────────┐
│ 2. EXAMINE PHASE                                             │
│    - Graph RAG retrieves relevant context                    │
│    - World Model updates with observations                   │
│    - Confidence and uncertainty quantification               │
└───┬─────────────────────────────────────────────────────────┘
    ↓
┌───▼─────────────────────────────────────────────────────────┐
│ 3. SELECT PHASE                                              │
│    - VULCAN generates candidate actions                      │
│    - Predict outcomes (causal prediction engine)             │
│    - Meta-reasoning filters (CSIU, safety)                   │
│    - Select optimal action                                   │
└───┬─────────────────────────────────────────────────────────┘
    ↓
┌───▼─────────────────────────────────────────────────────────┐
│ 4. APPLY PHASE                                               │
│    - Safety validation                                       │
│    - Graph compilation (if needed)                           │
│    - LLM generation or graph execution                       │
│    - Monitor with rollback capability                        │
└───┬─────────────────────────────────────────────────────────┘
    ↓
┌───▼─────────────────────────────────────────────────────────┐
│ 5. REMEMBER PHASE                                            │
│    - Update World Model                                      │
│    - Store in Persistent Memory with unlearning metadata     │
│    - Extract principles via Knowledge Crystallizer           │
│    - Trigger self-improvement if needed                      │
└─────────────────────────────────────────────────────────────┘
```

### 17.2 Component Interactions

| From | To | Data Flow | Purpose |
|------|-----|-----------|---------|
| Bridge → Graph RAG | Query + embeddings | Context retrieval |
| Bridge → World Model | Observations | State update and causal inference |
| SystemObserver → World Model | System events | Convert queries, results, errors to observations |
| World Model → Routing | Recommendations | Suggest tool selection based on learned patterns |
| World Model → Reasoning | Current state | Generate candidate actions |
| Reasoning → Meta-Reasoning | Candidates + predictions | Filter and select |
| Meta-Reasoning → Safety | Selected action | Validate before execution |
| Bridge → Compiler | Graph definition | Optimize and compile |
| Bridge → LLM Core | Context + constraints | Generate text response |
| Bridge → Persistent Memory | Action + result | Store for future retrieval |

### 17.2.1 SystemObserver Event Flow

The **SystemObserver** (located in `vulcan/world_model/system_observer.py`) creates a "nervous system" that connects the query processing pipeline to the World Model's causal learning system:

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Processing Pipeline                 │
│  QueryRouter → ReasoningIntegration → Engines → Response    │
└───┬─────────────────────────────────────────────────────────┘
    │ Events: query_start, engine_result, validation_failure,
    │         outcome, error
    ↓
┌───▼─────────────────────────────────────────────────────────┐
│                      SystemObserver                          │
│  - Converts system events to Observation objects            │
│  - Tracks query, engine, and outcome history                │
│  - Detects patterns (formal logic, probability, etc.)       │
└───┬─────────────────────────────────────────────────────────┘
    │ Observations
    ↓
┌───▼─────────────────────────────────────────────────────────┐
│                      World Model                             │
│  - Updates causal graph with observations                   │
│  - Learns which engines succeed on which query types        │
│  - Provides routing recommendations                         │
│  - Enables performance introspection                        │
└─────────────────────────────────────────────────────────────┘
```

### 17.3 Platform Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  API Layer (Flask/FastAPI)                   │
│         Registry API | Arena API | Gateway | Health          │
├─────────────────────────────────────────────────────────────┤
│                    Governance Layer                          │
│    Trust-Weighted Consensus | Policy Enforcement | Voting   │
├─────────────────────────────────────────────────────────────┤
│               VULCAN-AGI Core (285,000+ LOC)                 │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │  Reasoning   │ World Model  │   Meta-Reasoning (Self-  │ │
│  │   Systems    │   (Causal)   │   Improvement/Awareness) │ │
│  ├──────────────┼──────────────┼──────────────────────────┤ │
│  │   Memory     │   Planning   │   Safety & Ethics        │ │
│  │  Hierarchy   │   Engine     │   Boundaries (CSIU)      │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│         Graph Execution & Compilation Layer                  │
│  GraphixIR Compiler | Unified Runtime | LLM Core (3.2K LOC) │
├─────────────────────────────────────────────────────────────┤
│      Persistent Memory v46 (5.3K LOC) - Storage Layer       │
│  Graph RAG | LSM Tree | Unlearning | ZK Proofs | S3/CDN    │
├─────────────────────────────────────────────────────────────┤
│            Observability & Security Layer                    │
│  Prometheus | Grafana | Audit Logs | Security Scanning      │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure & Deployment                   │
│   Docker/K8s | Helm Charts | Redis | SQLite/PostgreSQL     │
└─────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [COMPLETE_SERVICE_CATALOG.md](COMPLETE_SERVICE_CATALOG.md) - Function-level service documentation
- [OBSERVABILITY.md](OBSERVABILITY.md) - Monitoring and metrics details
- [GOVERNANCE.md](GOVERNANCE.md) - Governance and consensus documentation
- [SECURITY.md](SECURITY.md) - Security architecture and threat modeling

---

**Document Version:** 2.2.0  
**Last Updated:** December 23, 2024
