# Graphix Vulcan Documentation Index

**Version:** 2.2.0  
**Last Updated:** December 23, 2024

Welcome to the root documentation corpus for the Graphix Vulcan Platform (Proprietary & Confidential — Novatrax Labs LTD).  
This folder aggregates deep technical, governance, safety, execution, optimization, and integration references.

> **📚 Complete documentation index:** See [INDEX.md](INDEX.md) for full navigation

## Contents Overview
| Category | File | Purpose |
|----------|------|---------|
| Platform Overview | [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | Layered structural & conceptual deep dive |
| Documentation Index | [INDEX.md](INDEX.md) | Complete navigation guide |
| API Surfaces | [API_DOCUMENTATION.md](API_DOCUMENTATION.md), [api_reference.md](api_reference.md) | Registry/Arena endpoints & schemas |
| Configuration | [CONFIGURATION.md](CONFIGURATION.md) | Environment variables & resolution order |
| Manifests & Semantics | [CONFIG_FILES.md](CONFIG_FILES.md) | Reference for core config & grammar files |
| Governance | [GOVERNANCE.md](GOVERNANCE.md) | Proposal lifecycle, trust weighting, consensus |
| Safety & Security | [SECURITY.md](SECURITY.md) | Threat model, cryptographic & validation controls |
| Execution | [EXECUTION_ENGINE.md](EXECUTION_ENGINE.md) | Scheduling, concurrency & optimization |
| Validation & Ontology | [ONTOLOGY.md](ONTOLOGY.md) | Semantic/structural contracts & enforcement |
| Intrinsic Improvement | [INTRINSIC_DRIVES.md](INTRINSIC_DRIVES.md) | Self-improvement policy & autonomous cycles |
| Observability | [OBSERVABILITY.md](OBSERVABILITY.md) | Metrics, tracing, audit, provenance & anomaly detection |
| Operations | [OPERATIONS.md](OPERATIONS.md) | Comprehensive operations guide |
| Troubleshooting | [troubleshooting.md](troubleshooting.md) | Common issues and solutions |
| Development | [TESTING_GUIDE.md](TESTING_GUIDE.md) | Local workflow, testing & performance profiling |
| Dependency Management | [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) | pip-tools, hashed requirements, security practices |
| Code Quality | [CODE_QUALITY_REQUIREMENTS.md](CODE_QUALITY_REQUIREMENTS.md) | Development tools, linting, testing standards |

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

---

**Document Version:** 2.2.0  
**Last Updated:** December 23, 2024
