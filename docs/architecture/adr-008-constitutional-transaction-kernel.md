# ADR 008: Constitutional Transaction Kernel

## Status

Accepted as the target architecture. This PR implements the first bounded migration slice.

## Context

Vulcan currently contains several overlapping descriptions of a cognitive event: the mutable `CognitiveCase`, the immutable `CognitiveEpisode`, the runtime semantic ledger, Graphix Epistemic, the audit lifecycle, and static capability evidence. Each contains useful work, but more than one can presently appear authoritative.

That ambiguity blocks trustworthy expansion. Persistent agency, tools, learning, self-improvement, and NPT-directed research cannot safely be added while the system lacks one answer to:

- which state was admitted;
- which interpretation was accepted;
- which facts were committed;
- which policy authorized publication or action;
- which effect occurred;
- which lifecycle state was reached.

## Decision

Vulcan will converge on a **Constitutional Transaction Kernel (CTK)**. CTK is the completed form of the existing microkernel architecture, not a replacement repository.

`CognitiveEpisode` is the root aggregate for one bounded cognitive transaction. Every live request must:

1. establish one canonical `case-*`/episode identity;
2. bind one bounded `SnapshotBundle` before semantic interpretation;
3. advance through explicit typed episode transitions;
4. bind interpretation, plans, claims, evidence, derivations, authorization, response, effects, and consolidation by digest;
5. terminate only in an authoritative terminal episode state;
6. release all state leases after completion.

The target live path is:

```text
authenticated ingress
  -> episode and snapshot admission
  -> untrusted language/semantic proposal
  -> validation and grounding
  -> epistemic commitment
  -> normative authorization
  -> response or effect execution
  -> observation
  -> communication
  -> consolidation
```

## Authority model

The authority lattice remains:

```text
UNTRUSTED_PROPOSAL
  -> VALIDATED_CANDIDATE
  -> COMMITTED_BELIEF
  -> AUTHORIZED_PLAN
  -> EXECUTED_EFFECT
```

Only the microkernel may perform promotion. Cognitive organs propose; the kernel commits.

## This PR

The atomic-admission migration slice changes the transaction boundary. Formerly, the runtime created a snapshot-free episode/case projection and the wrapper later mutated its digest by binding a bundle. Now `EpisodeAdmissionService` owns ID generation, bundle admission and validation, snapshot-bound genesis construction, and only then compatibility projection. A failed admission has no visible case and releases acquired leases exactly once.

This migration slice:

- gives the case and episode one canonical `case-*` identity;
- binds the composed runtime's nine-authority snapshot admission function to the kernel;
- preserves direct-kernel tests as an explicitly uncomposed compatibility path while production composition uses the real snapshot admission function;
- advances successful requests through the complete episode lifecycle to `CONSOLIDATED`;
- records abstentions and failures as authoritative terminal episode states;
- binds compatibility-ledger artifacts into the episode by digest;
- releases snapshot leases on every completed handling path;
- composes the legacy semantic kernel as a delegate without copying its resource ownership, and rejects direct production bypass with an unadmitted case;
- requires typed, evidence-bound snapshot rebase commands rather than magic reason strings.

## Deliberate compatibility boundary

The current `runtime.semantic` ledger and Graphix-like plan remain migration projections. This PR binds validated ledger and response artifacts, but it does not yet bind the compiled compatibility plan into the episode or rename the compatibility ledger as the final Graphix Epistemic authority. The next convergence waves will repair full semantic digest coverage, bind canonical plan artifacts, establish one durable epistemic head, derive audit from episode transitions, and remove the duplicate runtime-semantic contracts.

## Consequences

Positive consequences:

- successful public responses can no longer coexist with an episode left at `PERCEIVED`;
- audit retrieval and returned case identity use the same canonical format;
- future persistence work has an immutable episode digest chain to store;
- state admission becomes visible in the live cognitive path;
- later lineage and reafference work can compose episodes rather than bypass them.

Costs and risks:

- compatibility ledger collections remain temporarily mutable;
- production snapshot providers still require explicit per-authority implementations in a later PR;
- audit retains a duplicate lifecycle map until its planned convergence;
- the current Graphix compiler and Graphix Epistemic store are not yet the live path.

## Durable episode authority slice

The old episode transaction boundary ended at assignment of an immutable Python
object to the mutable `CognitiveCase`; process loss erased the authoritative
head, and legacy audit finalization could precede that assignment. The new sole
authority boundary is the microkernel `EpisodeStore`: snapshot-bound genesis and
every subsequent episode transition use a per-episode SQLite compare-and-swap
head, and the immutable transition, new head, and audit-outbox item commit in one
database transaction. The composed handler verifies the required durable head
after an explicit cancellation point and before returning a transportable
result. Terminal compatibility audit events are appended only after that
durable terminal commit.

`CognitiveCase` remains the named working-state adapter, while
`ConstitutionalCognitiveKernel` remains the production admission/transport
adapter. Remove both when the semantic kernel consumes durable episodes
directly. The legacy `CanonicalAudit` lifecycle remains a compatibility
projection until Wave 1.5 makes it consume episode outbox records; undelivered
rows are retained and startup reconciliation delivers them when that sink is
configured. This slice does not make the compatibility semantic ledger or
Graphix Epistemic durable authority.

## Graphix semantic-integrity repair

The old Graphix representation boundary accepted an extension digest without
checking it against its declared schema context, and Epistemic commit digests
omitted multiple meaning-bearing fields. The repaired boundary validates Core
extension digests at object construction and decode, and defines one complete,
canonical Epistemic commit document whose supplied digest is verified on load.
This is an M2 representation and validation change only: the microkernel remains
the sole authority promoter, and the canonical runtime and durable authority
boundaries do not change.

`project_semantic_claim` remains the named legacy claim adapter until Wave 2.2
retires the runtime semantic claim contract. `AuthoritativeClaimLedger` remains
the in-memory compatibility ledger until Wave 1.6 replaces it with a durable
Graphix Epistemic head.

## Faithful state-authority admission

The old state-admission boundary reflectively inspected arbitrary legacy objects,
guessed fields, synthesized fallback digests from `repr`, and reused legacy world,
learning, and self-improvement objects as unrelated authorities. The new boundary
is the explicit nine-member `StateAuthoritySet`. Each member serializes a declared
state document together with its owner, independent revision, release, schema,
and validity window; composition probes and releases all nine before admitting
traffic. Missing or failing readers abort startup.

The legacy world, self, social, and normative projections are explicitly disabled
until their owners expose complete constitutional state. Domain Registry,
Governed Memory, Cognitive Kernel capability truth, the dedicated proposal-only
CSIU policy configuration, and Alignment Registry have named owners. A disabled
authority is itself versioned content-bound state, not a digest of `None`.
`vulcan.testing.snapshots.AttributeSnapshotProvider` is the sole compatibility
adapter and is forbidden in production composition; remove it when migration and
direct-kernel tests construct `StateAuthoritySet` instances.

## Rejected alternatives

### Start a new framework

Rejected because it would duplicate authority, memory, audit, Graphix, alignment, and governance while leaving Vulcan unfinished.

### Add NPT modules directly to the legacy runtime

Rejected because naming a module `center`, `self`, or `consciousness` does not establish the required causal organization and would make the theory circular.

### Keep `CognitiveCase` and `CognitiveEpisode` as co-equal records

Rejected because two lifecycle authorities cannot provide deterministic recovery, audit, or identity.
