# Architecture Plan

## Risk Grade: L3

### Risk Assessment
- [x] Contains security/auth logic -> L3 (JWT/API key auth, rate limiting, trust-weighted consensus, safety validators)
- [x] Modifies existing APIs -> L2 (graph execution engine, governance proposals, metaprogramming handlers)
- [ ] UI-only changes -> L1

**Justification**: L3 — Platform includes authentication layers (JWT/API key), cryptographic audit trails, trust-weighted consensus with reputation scoring, privacy-preserving unlearning (GDPR), and autonomous self-modification capabilities. Security is load-bearing at every layer.

## File Tree (The Contract)

```
src/
├── vulcan/                    # Core cognitive architecture
│   ├── orchestrator/          # World model orchestration
│   ├── reasoning/             # Causal inference, meta-cognition
│   ├── memory/                # Persistent knowledge systems
│   ├── safety/                # Safety validators, alignment checks
│   ├── learning/              # Autonomous self-improvement
│   ├── semantic_bridge/       # Cross-domain inference
│   ├── curiosity_engine/      # Exploration-driven learning
│   ├── knowledge_crystallizer/# Knowledge verification
│   └── world_model/           # World state representation
├── execution/                 # Graph execution engine (DAG scheduler)
├── governance/                # Trust-weighted consensus, proposals
├── compiler/                  # LLVM-based graph compilation
├── safety/                    # Multi-layer safety framework
├── gvulcan/                   # Graph Vulcan core (IR, validation)
│   └── zk/                    # Zero-knowledge proof integration
├── memory/                    # Persistent memory subsystem
├── persistant_memory_v46/     # S3-backed storage with CDN
├── unified_runtime/           # Runtime unification layer
├── integration/               # External system bridges
├── llm_core/                  # LLM provider abstraction
├── tools/                     # Utility tooling
├── tests/                     # Test suite (89 files)
├── api_server.py              # REST API gateway
├── consensus_engine.py        # Core consensus logic
├── evolution_engine.py        # Autonomous graph evolution
├── graph_aware_evolution.py   # Metaprogramming evolution engine
├── hardware_dispatcher.py     # Hardware backend selection
├── audit_log.py               # Immutable audit trail (SQLite/WAL)
└── security_audit_engine.py   # Security pattern scanning
```

## Interface Contracts

### Graph Execution Engine
- **Input**: Graph IR (JSON — nodes, edges, metadata), execution mode (SEQUENTIAL|PARALLEL|STREAMING|BATCH)
- **Output**: Execution result envelope (outputs, metrics, provenance)
- **Side Effects**: Audit log entries, Prometheus metrics, cache mutations

### Governance Consensus
- **Input**: Proposal (draft), voter identities, trust weights
- **Output**: Decision (approved|rejected|expired), confidence score
- **Side Effects**: Proposal lifecycle transitions, audit trail, alignment checks

### Validation Pipeline
- **Input**: Graph IR, ontology version, validator config
- **Output**: Validation result (pass|fail), violation list by stage
- **Side Effects**: Cache population (keyed by graph hash + versions)

### VULCAN World Model Bridge
- **Input**: Proposal context, motivational state, safety constraints
- **Output**: Assessment score, safety violations, semantic transfers
- **Side Effects**: World model observation updates, delegation decisions

## Data Flow

```
Graph IR Submission
  -> Validation Pipeline (Structure -> Identity -> Edges -> Ontology -> Semantics -> Cycles -> Resources -> Security -> Alignment -> Safety)
  -> Governance Gate (if modification proposal)
    -> Trust-Weighted Consensus
    -> VULCAN Bridge Assessment
  -> Execution Engine (mode-aware DAG scheduler)
    -> Hardware Dispatcher (CPU|GPU|Photonic|Memristor fallback chain)
    -> Node Handler Execution (with per-node timeout, caching, audit)
  -> Observability (Prometheus metrics, Grafana dashboards, audit log)
  -> Provenance Chain (causal attribution, lineage tracking)
```

## Dependencies

| Package | Justification | Vanilla Alternative |
|---------|---------------|---------------------|
| Flask/FastAPI | REST API gateway | No — standard for Python APIs |
| SQLite (WAL) | Immutable audit trail | No — embedded, zero-config |
| Redis | Rate limiting, caching | In-memory dict (non-distributed) |
| Prometheus client | Metrics export | Custom metrics (not standard) |
| PyJWT | Authentication tokens | No — industry standard |
| LLVM (via ctypes) | Graph compilation to native | Interpreter-only (10-100x slower) |
| Docker/K8s | Production deployment | Bare-metal (no orchestration) |

## Section 4 Razor Pre-Check
- [ ] All planned functions <= 40 lines — **Existing codebase; needs audit**
- [ ] All planned files <= 250 lines — **Existing codebase; needs audit**
- [ ] No planned nesting > 3 levels — **Existing codebase; needs audit**

**Note**: This is a genesis bootstrap for an existing 285K+ LoC platform. Section 4 compliance requires a dedicated `/qor-audit` pass.

---
*Blueprint sealed. Awaiting GATE tribunal.*
