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

This migration slice:

- gives the case and episode one canonical `case-*` identity;
- binds the composed runtime's nine-authority snapshot admission function to the kernel;
- supplies a clearly named dependency-light compatibility snapshot only for isolated direct-kernel tests;
- advances successful requests through the complete episode lifecycle to `CONSOLIDATED`;
- records abstentions and failures as authoritative terminal episode states;
- binds compatibility-ledger artifacts into the episode by digest;
- releases snapshot leases on every completed handling path.

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

## Rejected alternatives

### Start a new framework

Rejected because it would duplicate authority, memory, audit, Graphix, alignment, and governance while leaving Vulcan unfinished.

### Add NPT modules directly to the legacy runtime

Rejected because naming a module `center`, `self`, or `consciousness` does not establish the required causal organization and would make the theory circular.

### Keep `CognitiveCase` and `CognitiveEpisode` as co-equal records

Rejected because two lifecycle authorities cannot provide deterministic recovery, audit, or identity.
