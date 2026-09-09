# Durable effect protocol

## Authority and transaction boundary

The old boundary ended at `AUTHORIZED_PLAN`; `CognitiveEpisode.effects` was a
contract slot and no production-safe executor owned durable intent, capability
consumption, attempts, or receipts. Legacy tools and research executors are not
effect authorities.

The new boundary is `EffectTransactionService` and `EffectStore`. Only a
`SYSTEM_KERNEL` principal can submit an effect. Before the sandbox is called,
the kernel first validates the complete proposal/policy/expectation chain and
the exact live episode and lineage heads, then durably issues the capability.
One SQLite transaction commits the canonical `EffectIntent` and atomically
consumes that issued capability's nonce and idempotency key. It binds episode,
lineage, branch,
policy, expected effect, resource, operation, release, principal, expiry, and
budget, and appends an audit-outbox record. A second transaction records the
attempt. A final transaction records its receipt and outbox event. These are
deliberately at-least-once audit deliveries and not a claim of exactly-once
external execution.

Constructing a token value is not authority: execution accepts only the exact
canonical token document previously issued by the kernel transaction service.
Authorization evidence is persisted and digest-verified on restart. Before the
sandbox mutation, its current resource digest must match `ExpectedEffect`; an
unexpected consequence is rolled back and recorded as failed.

The canonical contracts preserve the authority separation:

```text
PolicyProposal (UNTRUSTED_PROPOSAL)
  != AuthorizedPolicy
  != ExpectedEffect / durable EffectIntent
  != single-use CapabilityToken
  != EffectAttempt
  != EffectReceipt
```

The first target is a code-owned key/value sandbox with only `put` and `delete`.
It cannot access the host filesystem, processes, shell, network, finance, or
provider APIs. It is reversible and supports target-side idempotency. An
attempt lacking a receipt after restart becomes `ambiguous`; it is never
automatically executed again. A non-idempotent target remains ambiguous until
an `OPERATOR` principal reconciles it. Successful reversible operations may be
compensated, and compensation is separately audited.

## Episode and lineage binding

Intent and receipt records contain content-bound episode and lineage identities,
including the exact episode and lineage head digests used for authorization.
Their transactional outbox projections carry those references into audit. The
existing episode `execution-receipt.v1` slot and lineage `pending_effect_refs`
remain the aggregate projections; canonical runtime orchestration must advance
those projections from the durable effect artifacts rather than inventing an
independent effect state.

## Compatibility and removal

`ReversibleSandbox` is the named `deterministic-effect-sandbox` compatibility
target. Remove it only when a governed effect-port registry supplies equivalent
allowlisting, idempotency/reconciliation declarations, compensation semantics,
and restart tests. No legacy executor is adapted or authorized by this slice.

This is M2 local evidence. It is not M3 until a supported live request can
authorize the protocol through the composed cognitive kernel and atomically
project its intent/receipt into the episode and lineage heads. It is not M4
until the exact built artifact passes crash/restart qualification.
