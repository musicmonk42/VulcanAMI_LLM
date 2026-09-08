# Authoritative CognitiveEpisode contract

`vulcan.microkernel.episode.CognitiveEpisode` is the immutable request-scoped aggregate for chat, reasoning, tools, learning, CSIU, and improvement. `CognitiveCase` remains a temporary mutable working projection for the current semantic runtime; it is not a second lifecycle authority.

## Identity and admission

The public case identity and episode identity are one canonical `case-*` identifier. `EpisodeAdmissionService` creates that identity, admits and validates a bounded `SnapshotBundle`, and constructs the episode with the bundle reference already present in its genesis `PERCEIVED` event. The compatibility case is projected only after that construction succeeds; production never performs a later bind.

Production composition delegates to `CognitiveKernel` through an explicit kernel protocol and `RuntimeContainer.admit_snapshot_bundle`; it does not copy the delegate attributes or own its resources. Direct use of the constitutional kernel with a preconstructed, unadmitted case fails closed.

## Lifecycle

Episodes move only through typed transitions in `vulcan.microkernel.state_machine`:

```text
PERCEIVED
  -> INTERPRETED
  -> GROUNDED
  -> DELIBERATING
  -> EPISTEMICALLY_COMMITTED
  -> NORMATIVELY_AUTHORIZED
  -> EXECUTED
  -> OBSERVED
  -> COMMUNICATED
  -> CONSOLIDATED
```

Terminal non-success outcomes are `ABSTAINED`, `BLOCKED`, `FAILED`, and `CANCELLED`.

A successful `CognitiveCase` may close only after the episode reaches `CONSOLIDATED`. An abstention, block, failure, or cancellation must transition the episode to its corresponding terminal state before the compatibility projection closes.

Every transition records the prior episode digest, reason, kernel principal digest, admitted snapshot identity, and relevant artifact references. Claims, evidence, derivations, candidate plan, response authorization, exact response publication, and consolidation are bound into the immutable episode by digest.

## Current compatibility ledger

The lists held by `CognitiveCase` are a projection used by `runtime.semantic`. They validate request-local compatibility objects but cannot transition or persist an episode. The kernel projects those objects into immutable references submitted to the transaction service. They are not the final durable epistemic authority.

The planned replacement is one durable Graphix Epistemic commit head. Until that lands, changes must not describe the compatibility ledger as final or allow it to bypass episode transitions.

## Durable retention policy

## Command and publication semantics

New authoritative transitions are expressed as typed
`ConstitutionalTransactionService` commands. Each command binds the kernel
principal and release, its current authority grant, validation/evidence and
policy digests, the admitted snapshot digest, and the expected prior episode
digest. The durable store compare-and-swap is the commit point.

Communication is not a real-world effect. Response publication requires a
`response-publication-authorization.v1` artifact binding the exact rendered text
and governing decisions, then advances directly to `COMMUNICATED`. `EXECUTED`
and `OBSERVED` are reserved for an independently authorized external effect with
an `execution-receipt.v1` artifact.

Raw request bytes may exist only in working memory long enough to compute a digest and approved projection digest. Durable episode and audit stores persist `input_digest`, optional approved `projection_digest`, artifact references, and canonical digests. They must not persist raw prompts, raw provider text, secrets, hidden prompts, or private reasoning traces.

`vulcan.microkernel.episode_store.EpisodeStore` is the durable lifecycle
authority. It stores canonical episode documents, immutable transition rows,
one compare-and-swap head per episode, and an audit outbox row in the same
SQLite transaction. Startup verifies every document, transition chain, replayed
head, and pending outbox delivery. Delivery is at-least-once: projection sinks
must deduplicate by transition digest. Database failpoints bracket writes, head
CAS, commit, and outbox delivery for restart qualification.

Retention is digest-only and artifact-reference-only. Episode rows are retained
for the deployment's governance retention period; deletion/compaction is not
implemented in this slice and must not occur independently of the future
audit-retention policy.

## Lease ownership

The handler releases the admitted snapshot bundle after terminal handling, including failures and cancellations. The episode retains the immutable bundle reference and digest, not live leases.

## Migration and rollback

`vulcan.runtime.case.episode_from_case` is the explicit adapter for callers that still hold a `CognitiveCase`. Rollback may retain the adapter, but no durable schema or runtime route may introduce a separate cognitive lifecycle authority.

`CognitiveEpisode.bind_snapshot_bundle_for_migration` and `CognitiveCase.bind_snapshot_bundle` are migration-only adapters. Remove both when all test and external callers use `EpisodeAdmissionService`; neither is on the production route.

See [`adr-008-constitutional-transaction-kernel.md`](adr-008-constitutional-transaction-kernel.md) and [`../roadmap/constitutional-convergence-plan.md`](../roadmap/constitutional-convergence-plan.md).
