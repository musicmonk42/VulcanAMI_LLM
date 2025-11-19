# Graphix Vulcan Documentation Index

Welcome to the root documentation corpus for the Graphix Vulcan Platform (Proprietary & Confidential — Novatrax Labs LLC).  
This folder aggregates deep technical, governance, safety, execution, optimization, and integration references.

## Contents Overview
| Category | File | Purpose |
|----------|------|---------|
| Platform Overview | ARCHITECTURE.md | Layered structural & conceptual deep dive |
| API Surfaces | API_REFERENCE.md | Registry/Arena endpoints & schemas |
| Configuration | CONFIGURATION.md | Environment variables & resolution order |
| Manifests & Semantics | CONFIG_FILES.md | Reference for core config & grammar files |
| Governance | GOVERNANCE.md | Proposal lifecycle, trust weighting, consensus |
| Safety & Security | SECURITY.md | Threat model, cryptographic & validation controls |
| Execution | EXECUTION_ENGINE.md | Scheduling, concurrency & optimization |
| Validation & Ontology | ONTOLOGY.md / TYPE_SYSTEM.md / VALIDATION.md | Semantic/structural contracts & enforcement |
| Intrinsic Improvement | INTRINSIC_DRIVES.md / AUTONOMY.md | Self-improvement policy & autonomous cycles |
| AI Runtime | AI_RUNTIME.md | Provider abstraction & SLA contracts |
| Hardware | HARDWARE.md | Backend profiles & dispatch heuristics |
| Observability | OBSERVABILITY.md | Metrics, tracing, audit, provenance & anomaly detection |
| Operations | OPERATIONS.md | Deployment, HA, DR runbook |
| Development | DEVELOPMENT.md | Local workflow, testing & performance profiling |
| Explanations & Interpretability | EXPLANATIONS.md | Execution explanation system |
| VULCAN Integration | VULCAN_BRIDGE.md | World model alignment, consensus & semantic transfer |
| Legal & Compliance | LEGAL_NOTICES.md | Proprietary rights & restrictions |
| Change Tracking | CHANGELOG.md | Versioned evolution record |

## High-Level Layering

```
Governance & Evolution
    → Validation (Structure/Ontology/Semantic/Safety/Motivational)
        → Execution Engine (Parallel/Sequential/Streaming/Batch)
            → AI Runtime / Hardware Dispatch / Node Handlers
                → Metrics + Audit + Provenance
                    → Autonomous Optimization / Pattern Learning
                        → World Model Alignment (VULCAN Bridge)
```

## Cross-Cutting Invariants
- Deterministic replay of validated graph IR (except bounded non-deterministic node types explicitly declared).
- Cryptographic hash chaining for proposals & audit events.
- Strict resource envelopes (memory, node/edge counts, recursion depth).
- Safety-first gating (Exec-like or external side-effect nodes quarantined unless approved).
- Alignment + consensus required for high-impact ontology & grammar changes.

## New Integrations (Deepened)
- VULCAN motivational introspection pre/post execution.
- Semantic concept transfer attaching `_semantic_transfers` metadata inside graph artifacts.
- Autonomous cycles generating evolutionary proposals with risk-weighted gating.

Refer to each file for exhaustive detail. Update or extend docs only through governed change processes.
