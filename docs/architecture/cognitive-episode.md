# Authoritative CognitiveEpisode contract

`vulcan.microkernel.episode.CognitiveEpisode` is the immutable request-scoped aggregate for chat, reasoning, tools, learning, CSIU, and improvement. `CognitiveCase` remains a temporary mutable working projection for the current semantic runtime; it is not a second lifecycle authority.

## Identity and admission

The public case identity and episode identity are one canonical `case-*` identifier. The runtime creates that identity before reasoning, admits a bounded `SnapshotBundle`, and binds the bundle to the episode while it is still at its genesis `PERCEIVED` state.

Production composition wraps `CognitiveKernel` with `RuntimeContainer.admit_snapshot_bundle`. Direct-kernel tests remain an explicitly uncomposed compatibility path; they do not establish or claim production snapshot admission. Constitutional-path tests supply explicit bounded providers while production state-authority ports are completed.

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

Every transition records the prior episode digest, reason, authority, admitted snapshot identity, and relevant artifact references. In this migration slice, claims, evidence, derivations, response authorization, response publication, and consolidation are bound into the immutable episode by digest. The compiled compatibility plan is not yet retained on `CognitiveCase`; binding a canonical plan artifact is an explicit next-step requirement.

## Current compatibility ledger

The lists held by `CognitiveCase` are a migration projection used by `runtime.semantic`. The kernel validates them, creates immutable artifact references, and immediately binds those references to the episode. They are not the final durable epistemic authority.

The planned replacement is one durable Graphix Epistemic commit head. Until that lands, changes must not describe the compatibility ledger as final or allow it to bypass episode transitions.

## Durable retention policy

Raw request bytes may exist only in working memory long enough to compute a digest and approved projection digest. Durable episode and audit stores persist `input_digest`, optional approved `projection_digest`, artifact references, and canonical digests. They must not persist raw prompts, raw provider text, secrets, hidden prompts, or private reasoning traces.

## Lease ownership

The handler releases the admitted snapshot bundle after terminal handling, including failures and cancellations. The episode retains the immutable bundle reference and digest, not live leases.

## Migration and rollback

`vulcan.runtime.case.episode_from_case` is the explicit adapter for callers that still hold a `CognitiveCase`. Rollback may retain the adapter, but no durable schema or runtime route may introduce a separate cognitive lifecycle authority.

See [`adr-008-constitutional-transaction-kernel.md`](adr-008-constitutional-transaction-kernel.md) and [`../roadmap/constitutional-convergence-plan.md`](../roadmap/constitutional-convergence-plan.md).
