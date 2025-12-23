# 🧬 Evolution Roadmap (Graphix & VULCAN-AI)

**Version:** 2.2.0  
**Goal:** Autonomous, safe, recursively improving cognitive substrate.

---

## Phase Summary

| Phase | Focus | Status | Key Files | Milestones |
|-------|-------|--------|-----------|------------|
| 0 | Foundations | Complete | `graphix_arena.py`, `agent_registry.py` | Agent orchestration, initial mutator |
| 1 | Mutation & Rollback | Complete | `nso_aligner.py`, `persistence.py` | Risky code removal, integrity checks |
| 2 | Synthetic Data | Complete | `data_augmentor.py`, `feedback_protocol.py` | Counterfactual/adversarial proposal generation |
| 3 | Ethical AI | Complete | `nso_aligner.py`, `security_nodes.py` | Multi-LLM consensus safety audit |
| 4 | Curriculum & Optima Escape | Complete | `tournament_manager.py`, augmentor modules | Adaptive tournament & diversity penalty |
| 5 | Human Feedback Loop | Complete | `feedback_protocol.py`, `self_optimizer.py` | RLHF endpoint integration |
| 6 | Recursive Meta-Graphs | Partial | `unified_runtime.py`, `evolution_engine.py` | Depth-limited recursion, RL proposals |
| 7 | Hardware & Sustainability | Partial | `hardware_dispatcher.py`, `analog_photonic_emulator.py` | Energy-aware backend selection |
| 8 | Full Cognitive Cycle | Complete | `vulcan/orchestrator.py`, `vulcan/memory.py` | Perceive → reason → act loop |
| 9 | Formal Safety Layer | Pending | Planned specs | Invariant definition & verification |
| 10 | Federated Evolution | Pending | (future modules) | Cross-instance proposal exchange |

---

## Detailed Notes

- **Phases 0–5:** Rapid stabilization of proposal life cycle and safe mutation pipeline.  
- **Phase 6:** Recursion safeguards (depth & resource) to prevent runaway meta-graphs.  
- **Phase 7:** RL-driven hardware strategy exploring energy vs latency trade-offs.  
- **Phase 8:** Cognitive orchestration bridging memory, reasoning, semantic transfer.  
- **Future (9–10):** Formalization & multi-instance resilience.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Mutation Validity | > 95% non-dead mutations |
| Ethical Safety | 100% flagged risky patterns |
| LTM Recall | > 95% reintroduction prevention |
| Energy Efficiency | < 0.4 nJ/op (emulated target) |
| Audit Coverage | 100% I/O events |
| Consensus Stability | < 1% deadlock rate |

---

## Risk Map

| Risk | Mitigation |
|------|------------|
| Over-specialization | Diversity penalties in tournaments |
| Safety bypass attempts | Multi-layer NSO + pattern scanning |
| Resource exhaustion | Pre-exec memory/latency estimation |
| Replay spam | Hash + embedding similarity gating |
| Hardware skew | Balanced strategy weighting adjustments |

---

## Current Priorities (Q4 2025)

1. Formal invariant specification (safety, determinism bounds).  
2. Federated proposal exchange with trust normalization.  
3. Enhanced sustainability metrics (temperature, utilization).  
4. Semantic pattern clustering for macro optimization (learned subgraphs).  
5. Specialized photonic noise calibration for more realistic variance modeling.

---

## Files of Interest

| Domain | File |
|--------|------|
| Governance & Ethics | `nso_aligner.py`, `security_nodes.py` |
| Evolution Logic | `evolution_engine.py`, `tournament_manager.py` |
| Memory | `vulcan/memory.py` |
| Hardware | `hardware_dispatcher.py`, `analog_photonic_emulator.py` |
| Validation | `run_validation_test.py` |

---

## Roadmap Extensions

- Multi-agent negotiation across divergent optimization proposals  
- Adaptive ontology evolution guided by usage heatmaps  
- Policy DSL for editable safety constraints  
- Confidence-calibrated alignment scoring adjustments

---

## Design Principles

> *Consolidated from UNFLATTENABLE_ROADMAP.md*

### Hierarchical Preservation
Avoid oversimplification of multi-stage reasoning. Preserve:
- Multi-level graph layering
- Distributed execution semantics
- Analog/photonic variability
- Semantic traceability & provenance

### Core Design Principles

| Principle | Rationale |
|-----------|-----------|
| Hierarchical Preservation | Avoid oversimplification of multi-stage reasoning |
| Analog Fidelity | Capture non-digital performance variance |
| Semantic Layering | Track conceptual transformations, not just raw ops |
| Deterministic Auditing | Immutable execution & proposal lineage |
| Energy Awareness | Sustainable scaling under constrained resources |

### Unflattenable Concepts
"Unflattenable" design resists collapsing multi-level, heterogeneous computation into a purely uniform digital abstraction. The objective is to minimize information loss and optimize sustainability & cognitive fidelity.

---

> Evolution remains subservient to safety constraints; any failure to meet invariants triggers rollback + audit escalation.
