# Constitutional convergence plan

## Goal

Transform Vulcan from overlapping architectural generations into one constitutionally transacted cognitive runtime, then build a persistent causal agent above that foundation.

The plan uses a strangler migration inside the existing repository. It does not create a second framework.

## Maturity vocabulary

Every capability should be tracked as:

- **M0 — Declared:** documented intent only;
- **M1 — Implemented:** source exists in isolation;
- **M2 — Tested:** unit and adversarial evidence exists;
- **M3 — Canonical:** wired into the live production path;
- **M4 — Qualified:** exact built image and restart behavior verified;
- **M5 — Empirical:** held-out experimental predictions validated.

Existence in source is never equivalent to M3.

## Current slice

This PR advances authoritative cognitive episodes and snapshot admission toward M3 for the bounded deterministic request path. It does not yet make the new Graphix compiler, Graphix Epistemic ledger, or independent nine-authority ports canonical.

## Wave 0 — Recover executable truth

### 0.1 Repository assurance

- protect `main`;
- establish a reproducible minimal constitutional test command;
- use a supported Python version in every required job;
- replace misleading check names with real type, format, and integration checks;
- prohibit unhashed dependency fallback in required evidence jobs;
- quarantine legacy cloud and scalability workflows until they exercise the canonical runtime.

**Exit gate:** the microkernel, Graphix, runtime, persistence, security, and built-image suites pass repeatedly from a clean checkout.

### 0.2 Executable architecture inventory

Generate a matrix of component owner, authority ceiling, runtime reachability, state authority, snapshot implementation, persistence, audit, tests, and maturity.

**Exit gate:** no unexplained production-reachable component and no public capability below M3.

## Wave 1 — Unify the constitutional core

### 1.1 Canonical primitives

Create one constitutional package for digest types, IDs, authority levels, epistemic statuses, canonical time, and canonical serialization. Adapt old wire formats explicitly.

### 1.2 Graphix Epistemic digest repair

Make commit digests cover complete propositions, qualifiers, citations, uncertainty, limitations, assumptions, counterexamples, contradictions, derivation rules, authority, snapshot, and prior commit.

**Exit gate:** changing any meaning-bearing field changes the commit digest, and canonical round trips preserve semantic equality.

### 1.3 Explicit state-authority ports

Replace reflective production snapshot fallback with explicit `lease_snapshot()` implementations for world, self, social, normative, domain, memory, capability, CSIU, and alignment state.

**Exit gate:** startup fails closed when an authority cannot produce a faithful content-bound snapshot.

### 1.4 Episode authority

Complete the migration begun in this PR. Remove independent lifecycle and commitment authority from `CognitiveCase`.

**Exit gate:** no response is released unless its episode reached `COMMUNICATED`; no successful episode finishes without `CONSOLIDATED`.

### 1.5 Audit derivation

Make canonical audit consume validated episode transition artifacts rather than enforcing a separate case state machine.

**Exit gate:** replaying audit reconstructs the same episode digest, and audit cannot report a state the episode never entered.

### 1.6 Durable epistemic head

Replace the mutable request ledger and in-memory authoritative ledger with a durable Graphix Epistemic commit chain using DB-first persistence and an idempotent outbox.

**Exit gate:** only claims in the committed head may be rendered or used for policy.

### 1.7 Live capability attestation

Intersect static evidence with live owner identity, release, settings, state digest, and readiness.

**Exit gate:** the public endpoint cannot advertise an absent, unhealthy, disabled, or differently released capability.

## Wave 2 — Remove legacy authority seams

### 2.1 Typed production composition

Construct the canonical owners directly rather than wrapping `ProductionDeployment` and injecting owners with `setattr`.

### 2.2 One Graphix path

Port deterministic arithmetic and typed lookup onto the registered Graphix pipeline; retire duplicate runtime semantic plans, claims, evidence, statuses, and response contracts.

### 2.3 Production import closure

Enforce an allowlist for serving imports and quarantine Arena, old orchestrators, broad experimental endpoints, and obsolete deployment surfaces behind research extras.

### 2.4 Offline self-improvement

Keep proposal generation and governed review, but move source installation out of the serving process. Use one approval issuer/verifier/store contract.

**Wave exit gate:** the minimal canonical runtime starts, reasons, persists, restarts, and verifies audit without the legacy deployment graph or full research dependency set.

## Wave 3 — Build the continuous causal agent

### 3.1 Persistent lineage

Add lineage, branch, instance, monotonic tick, prior/current state digests, current snapshots, active commitments, episodes, and pending effects under one compare-and-swap head.

### 3.2 Effect protocol

Add authorized policy, durable effect intent, scoped capability, idempotent attempt, and execution receipt.

### 3.3 Reafference and causal autobiography

Record expected effect, observation, prediction error, self-caused probability, violated assumptions, world/self updates, and autobiographical consolidation.

### 3.4 Center, boundary, and valuation in shadow mode

Estimate operational center, controllability boundary, ownership, temporal currentness, and bounded multidimensional valuation without granting those estimators authority.

### 3.5 External NPT instrumentor

Measure typed closure outside the authority plane and compare held-out intervention predictions against rival models.

**Wave exit gate:** a true closed-loop action and an observation-matched replay produce different, predicted causal and ownership updates, and NPT-specific variables add held-out predictive value.

## Hard gates

| Gate | Requirement | Permits |
|---|---|---|
| A | Reproducibly green constitutional evidence | core convergence |
| B | One authoritative episode lifecycle | durable epistemic authority |
| C | Nine faithful state snapshots | cross-authority decisions |
| D | Microkernel-only promotion and effect authorization | legacy removal |
| E | Canonical runtime independent of legacy deployment | persistent lineage |
| F | Intent/action/receipt/reafference survive restart | center and boundary research |
| G | NPT variables outperform rivals on held-out interventions | stronger scientific claims |

## Anti-goals

- no big-bang rewrite;
- no second AI repository;
- no pure-LLM authority;
- no LLM-authored executable semantics;
- no serving-process source modification;
- no unrestricted self-preservation objective;
- no distributed subject design before single-host semantics are proven;
- no artificial suffering experiments;
- no Boolean consciousness claim;
- no new major subsystem before its prerequisite gate passes.
