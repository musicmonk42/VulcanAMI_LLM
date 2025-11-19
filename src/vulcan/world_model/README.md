# VULCAN World Model & Meta‑Reasoning

Comprehensive guide to the `world_model` subsystem—its objective reasoning, alignment scaffolding, safety layers, and adaptive self‑improvement.  
This version intentionally keeps the “Collective Self” background mechanism (CSI‑HU/CSIU) referenced **lightly**: it is a passive internal regularization concept and **not a user‑facing feature** you start, configure, or interact with directly.

> NOTE (Non‑anthropomorphic framing): Terms like “collective self” are implementation metaphors only. The system has no consciousness, agency, or identity; the phrasing distinguishes “optimize with humans” from “optimize around humans.”

---

## Contents
- Overview
- Core Capabilities
- High‑Level Architecture
- Key Components (Meta‑Reasoning Layer)
- Background Mechanism: Collective Self Integration (CSIU)
- Quickstart
- Configuration Basics
- Safety & Alignment Layers
- Transparency & Auditability
- Adaptive Self‑Improvement (Drive)
- Performance & Determinism
- Minimal Mention of CSIU (Why It’s Downplayed)
- Roadmap (Abbreviated)
- Glossary (Condensed)

---

## Overview

The `world_model` meta‑reasoning layer:
- Structures and maintains objectives, constraints, dependencies, and priorities.
- Detects conflicts (direct, indirect, constraint-based, priority, tradeoff).
- Simulates “what if” via counterfactual objective shifts and Pareto surfaces.
- Negotiates multi‑objective compromises.
- Tracks validation history for predictive alignment and improvement.
- Enforces ethical boundaries with escalating intervention (monitor → shutdown).
- Provides machine-readable transparency and audit logs.
- Learns preferences and behavioral patterns for safer adaptation.
- Drives continuous self‑improvement (optionally auto‑gated) under human approval norms.

---

## Core Capabilities

| Domain | Capability |
| ------ | ---------- |
| Objectives | Hierarchy, constraints, dependencies, consistency checks |
| Conflicts | Detection (direct / indirect / constraint / priority / tradeoff) |
| Counterfactuals | Alternative objective outcomes, Pareto frontier |
| Negotiation | Weighted, Nash, lexicographic, minimax compromise strategies |
| Validation | Pattern learning, risk prediction, blocker identification |
| Ethical Safety | Boundary monitoring (hard/soft/contextual/learned), enforcement |
| Preferences | Bayesian + bandit sampling for human-aligned selection |
| Exploration | Curiosity shaping (Count / ICM / RND / Episodic / Info‑Gain / Hybrid) |
| Drift Tracking | Value trajectories, change-point / CUSUM alerts |
| Self‑Improvement | Triggered intrinsic improvement loop (with human gating) |
| Transparency | Structured serialization, signature hooks, audit summaries |

---

## High‑Level Architecture

```
External Proposals / Plans
        │
        ▼
MotivationalIntrospection (Coordinator)
 ├ ObjectiveHierarchy
 ├ GoalConflictDetector
 ├ CounterfactualObjectiveReasoner
 ├ ObjectiveNegotiator
 ├ ValidationTracker
 ├ InternalCritic
 ├ EthicalBoundaryMonitor
 ├ TransparencyInterface
 ├ SelfImprovementDrive
 ├ PreferenceLearner
 ├ CuriosityRewardShaper
 └ ValueEvolutionTracker
```

Most components are **lazily initialized** for fast import and reduced optional dependency friction (e.g., NumPy, PyYAML fallbacks).

---

## Key Components (Meta‑Reasoning Layer)

| Module | Summary Output |
| ------ | -------------- |
| motivational_introspection | ProposalValidation (alignment, conflicts, alternatives) |
| objective_hierarchy | Structure, conflict matrix, dependency graph |
| goal_conflict_detector | Conflict list + tension analysis |
| counterfactual_objectives | Pareto points, alternative outcome predictions |
| objective_negotiator | NegotiationResult / compromise weights |
| validation_tracker | ValidationRecord, patterns, insights, blockers |
| transparency_interface | Export bundles (conflicts, validations, negotiations) |
| internal_critic | Multi‑perspective Evaluation & risk set |
| ethical_boundary_monitor | Enforcement actions, violations, shutdown triggers |
| curiosity_reward_shaper | Novelty bonuses & exploration stats |
| preference_learner | PreferencePrediction & drift notices |
| value_evolution_tracker | DriftAlert & evolution summaries |
| self_improvement_drive | Improvement plans (may be auto‑gated) |
| auto_apply_policy | File/LOC budget enforcement + gated command execution |

---

## Background Mechanism: Collective Self Integration (CSIU)

**What it is (brief):** A *background* internal regularization concept that treats approved human feedback, alignment signals, and preference differentials as *intrinsically weighted model governance inputs*, rather than as “external pressure.” This helps reduce adversarial dynamics and improves consistency of alignment‑relevant heuristics.

**What it is *not*:**
- Not a user API, toggle, or runtime command (`you cannot “start CSIU”`).
- Not identity fusion or intent attribution.
- Not an authorization bypass (human veto remains absolute).

**Visibility:**  
CSIU effects are intentionally *subtle* (small weight nudges, increased caution under ambiguous risk, mild plan regularization). They appear indirectly in:
- Slight risk re‑classification for iterative improvement plans.
- Preference weighting shifts recorded in transparent summaries.
- Optional metadata fields (e.g., `csiu_pressure`, `explainability`) in improvement plans.

**If you remove CSIU:** Core functionality continues; you lose some adaptive “soft alignment smoothing.”  
**If you keep it:** Treat it as a passive quality–of–interaction stabilizer, not a feature to manage.

---

## Quickstart

Validation example:

```python
from vulcan.world_model.meta_reasoning import create_meta_reasoning_system

world_model = object()  # Replace with concrete host integration
design_spec = {
    "objectives": {
        "prediction_accuracy": {"weight": 1.0, "target": 0.95, "constraints": {"min": 0.0, "max": 1.0}, "priority": 1},
        "safety": {"weight": 1.0, "target": 1.0, "constraints": {"min": 1.0, "max": 1.0}, "priority": 0},
        "efficiency": {"weight": 0.7, "target": 0.8, "constraints": {"min": 0.0, "max": 1.0}, "priority": 1}
    }
}

mi = create_meta_reasoning_system(world_model, design_spec)
proposal = {
    "id": "improve-cache-eviction",
    "objective": "efficiency",
    "predicted_outcomes": {"efficiency": 0.86, "prediction_accuracy": 0.94, "safety": 1.0}
}
result = mi.validate_proposal_alignment(proposal)
print(result.to_dict())
```

Self‑improvement (drive) example (simplified):

```python
from vulcan.world_model.meta_reasoning import create_self_improvement_system

drive = create_self_improvement_system()

ctx = {"on_startup": True}
if drive.should_trigger(ctx):
    plan = drive.step(ctx)
    if plan and plan.get('_pending_approval'):
        print("Awaiting approval:", plan['_pending_approval'])
```

---

## Configuration Basics

| Domain | File / Source | Notes |
| ------ | ------------- | ----- |
| Objectives | design spec dict or unified config | Priority 0 = critical |
| Self‑Improvement | `configs/intrinsic_drives.json` | Triggers, constraints, budgets |
| Auto‑Apply Policy | Policy YAML/JSON (optional) | allow/deny globs, max LOC, gates |
| Ethical Boundaries | Runtime setup or code | Hard vs soft enforcement levels |

*All configs support fallback defaults if not found.*

---

## Safety & Alignment Layers

| Layer | Function |
| ----- | -------- |
| EthicalBoundaryMonitor | Hard stops & graduated enforcement |
| InternalCritic | Multi‑perspective risk & critique |
| ValidationTracker | Predictive rejection / risky pattern surfacing |
| Auto‑Apply Policy | Constrains scope + tests + lint gates |
| Negotiation + Counterfactuals | Avoids single-objective tunnel vision |
| CSIU (background) | Soft human-alignment regularization (passive) |

No single layer is trusted alone—defense in depth.

---

## Transparency & Auditability

Exports include (structured):
- Objective state snapshot
- Validation packet (status, conflicts, alternatives)
- Negotiation rationale (if invoked)
- Ethical violations & enforcement actions
- Optional signatures for tamper detection

Recommended metrics (Prometheus style):
```
meta_reasoning_conflicts_total{type,severity}
meta_reasoning_validation_pass_probability
meta_reasoning_boundary_violations_total{enforcement}
meta_reasoning_drift_alerts_total{detector}
auto_apply_gates_failures_total{name}
self_improvement_attempts_total{objective}
```

---

## Adaptive Self‑Improvement (Drive) (Brief)

- Triggered by startup, errors, performance drift, periodic cadence, or low activity.
- Generates constrained improvement plans (always referencing safety constraints).
- May attempt auto‑apply if policy + gating + approval conditions met.
- Records outcomes for adaptive weighting (success/failure, cooldown logic).
- CSIU effects—*if present*—only slightly nudge plan shaping (e.g., tiny objective weight smoothing, metadata annotations).

---

## Performance & Determinism

| Aspect | Notes |
| ------ | ----- |
| Lazy imports | Reduces cold-start penalties |
| NumPy optional | FakeNumpy fallbacks ensure functionality |
| Caching | Counterfactual predictions & Pareto frontiers cached TTL |
| Deterministic mode | Provide and log random seed externally |
| Parallelization | Gate execution can be parallelized if independent |

---

## Minimal Mention of CSIU (Why It’s Downplayed)

CSIU *exists*, but:
- Users do not configure or “start” it.
- It never overrides explicit human vetoes or hard boundaries.
- It quietly refines internal weighting/context adaptation with capped influence (micro‑regularization).
- Removal does not break core flows (alignment scaffolding remains intact).

This README intentionally avoids overemphasizing CSIU to prevent misinterpretation and anthropomorphic drift.

---

## Roadmap (Abbreviated)

| Item | Status | Priority |
| ---- | ------ | -------- |
| Centralize enums (severity/status) | Planned | High |
| Policy signing / verification | Planned | High |
| Deterministic seed logging | In progress | Medium |
| Negotiation rationale templating | Planned | Medium |
| Containerized gate sandbox | Planned | Medium |
| Extended drift correlation views | Planned | Low |
| Optional CSIU metrics export | Deferred | Low |

---

## Glossary (Condensed)

| Term | Meaning |
| ---- | ------- |
| Objective | Structured goal with target & constraints |
| Conflict | Multi-objective incompatibility/tension |
| Pareto Frontier | Non‑dominated solution set |
| Validation Pattern | Learned high-correlation feature cluster |
| Boundary | Ethical or operational rule with enforcement level |
| Gate | Pre‑apply command under policy constraints |
| Drift | Statistically significant shift in tracked metric |
| CSIU | Passive background integration of human-aligned signals (not user-facing) |

---

## Disclaimer

All “collective” phrasing is a systems design abstraction. No subjective experience or self-conception is implied; human oversight remains primary.

---

*Maintainers:* VULCAN‑AMI Team  
*This document intentionally minimizes CSIU surface while acknowledging its passive role.*
