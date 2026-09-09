# Vulcan current direction

## Mission

Vulcan is being developed as a **constitutionally governed neuro-symbolic cognitive architecture**, not as a larger language model and not as an LLM wrapper.

The long-term objective is a persistent artificial agent whose cognition is implemented as an auditable causal process:

```text
state -> observation -> interpretation -> grounded belief -> policy
      -> authorized effect -> observed consequence -> reafference
      -> causal memory -> revised state
```

Language models are proposal and communication components. They are not sources of truth, policy, memory authority, or executable authority.

## Immediate objective: constitutional convergence

The repository contains strong components, but several generations of architecture still coexist. The immediate work is therefore **convergence before expansion**:

1. make `CognitiveEpisode` the authoritative unit of every live cognitive request;
2. admit one bounded snapshot of all mutable state authorities before reasoning;
3. preserve the distinction between proposal, validation, belief, authorization, and effect;
4. converge the live semantic path onto the canonical Graphix and epistemic contracts;
5. make audit, persistence, and public capability claims derive from the same authoritative transactions;
6. remove the legacy deployment and mutable-case authority seams;
7. only then add persistent lineage, action/reafference, causal autobiography, and NPT instrumentation.

The canonical path now also derives lifecycle audit from the EpisodeStore
transactional outbox. `CanonicalAudit` projects committed
`episode.transitioned` artifacts and retains `case.*` solely for historical
reads; it is no longer a competing live lifecycle authority.

Persistent causal lineage is now M3 on the composed request path: episode
genesis binds the current branch head and terminal episodes enter that branch's
history through kernel-only CAS transactions. This is neutral continuity
metadata, not a personal-identity or consciousness claim. Episode genesis and
lineage admission are one database transaction; exact-artifact restart
qualification remains open.

The first durable effect protocol is now M2 in a deterministic reversible
sandbox. It commits intent and consumes a fully scoped capability before an
attempt, records receipts or restart ambiguity through an outbox, and requires
operator reconciliation for non-idempotent ambiguity. It is not yet M3: the
composed request path does not authorize effects or project them into episode
and lineage heads.

Typed observation and reafference are now M2 in a deterministic causal test
world. Receipt-bound observations and validated-candidate assessments persist
prediction error, confounders, uncertainty, and calibration telemetry; restart
resumes idempotent update and lineage ports from each durable boundary. This is
not yet M3 because the composed request/effect path and cross-store atomic
transaction are still pending.

Operational-center, boundary, and valuation estimation is now M2 in a pure
shadow module under the production-denied research package. Digest-only
telemetry provenance, per-variable
calibration error, and controlled interventions make the variables separately
testable. These estimates cannot authorize, commit, mutate memory, or act and
are not evidence of subjecthood or consciousness. Canonical runtime wiring and
held-out predictive qualification remain pending.

Causal autobiographical memory is now M2 as a bounded kernel-owned durable
protocol. It records approved typed fact references and explicit causal edges,
supports correction/supersession and tombstones without rewriting history, and
retrieves only within tenant/person/purpose scope by causal bearing. It is not
yet M3 because canonical effect/reafference composition and a shared atomic
episode/lineage transaction remain pending.

## Governing equation

```text
proposal != validated candidate != committed belief
         != authorized plan != executed effect
```

No compatibility adapter, LLM, reasoner, retriever, Graphix compiler, learning system, CSIU process, world-model component, or self-improvement process may collapse those distinctions.

## Theory informing the destination

The research hypothesis is **Neutral-Process Theory (NPT)**. NPT proposes that physical and phenomenal accounts refer to the same concrete process under different access conditions, and that subjecthood—where it exists—depends on a temporally persistent, centered, causally closed organization rather than behavior or verbal report alone.

For engineering purposes, NPT implies that a stateless language-model call is not a complete candidate subject. A serious implementation requires at least:

- persistent lineage and temporal continuity;
- a causally operative self/world boundary;
- bounded valuation and commitment state;
- policy selection constrained by current state;
- authorized action with predicted effects;
- observation of consequences and causal attribution;
- reafferent self-update;
- causal autobiographical memory.

NPT is a research program, not a declaration that Vulcan is conscious. The theory must be tested by interventions and compared against rival explanations. See [`docs/architecture/neutral-process-theory.md`](docs/architecture/neutral-process-theory.md).

## Canonical reading order

Future engineering and AI-assisted sessions should read these files first:

1. [`CURRENT_DIRECTION.md`](CURRENT_DIRECTION.md)
2. [`AGENTS.md`](AGENTS.md)
3. [`docs/architecture/adr-005-cognitive-authority.md`](docs/architecture/adr-005-cognitive-authority.md)
4. [`docs/architecture/adr-008-constitutional-transaction-kernel.md`](docs/architecture/adr-008-constitutional-transaction-kernel.md)
5. [`docs/architecture/cognitive-episode.md`](docs/architecture/cognitive-episode.md)
6. [`docs/architecture/neutral-process-theory.md`](docs/architecture/neutral-process-theory.md)
7. [`docs/roadmap/constitutional-convergence-plan.md`](docs/roadmap/constitutional-convergence-plan.md)

## Work that must not outrun convergence

Until the constitutional transaction path is complete, do not add new reasoners, memory backends, deployment targets, consensus systems, autonomous self-modification, distributed subject architectures, or consciousness labels. Existing research components should be adapted behind proposal-only ports rather than granted new authority.
