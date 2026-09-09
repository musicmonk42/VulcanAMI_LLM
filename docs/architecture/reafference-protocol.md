# Observation and reafference protocol

## Authority and transaction boundary

Previously the durable effect boundary ended at an execution receipt. Existing
observation and learning modules could describe changes, but none constituted a
receipt-bound causal chain and none was authorized to update the constitutional
lineage. The new bounded authority is `ReafferenceTransactionService`: a typed
environment observation and deterministic assessment are persisted as a
validated candidate, the microkernel commits proposed updates, and only then
may it advance lineage and durably mark the chain reconciled.

The authority equation remains:

```text
effect receipt != observation != assessment candidate != committed update
```

`Observation` binds the expected effect, exact receipt, typed changes, timing,
intervention identity, factual action-channel status, and measurement
uncertainty; it does not assert that the action caused the change.
`ReafferenceAssessment` binds prediction error, candidate causes, calibrated
self-caused probability, violated assumptions, proposed world/self/policy
updates, and uncertainty. It is always a `validated_candidate`; only the
microkernel callback may commit its proposed updates.

The transaction service requires explicit observation- and assessment-validator
ports. Production composition must bind those ports to code-registered typed
adapters and assessors; accepting provider or model-authored artifacts is not a
valid implementation. Receipt admission uses `EffectStore`'s read-only
`validate_reafference_evidence` adapter, which verifies the complete persisted
receipt, intent, and expected-effect digest chain and admits successful effects
only.

## Causal test adapter

`DeterministicTestWorld` is the named compatibility adapter for local causal
qualification. In closed-loop mode the current action counterfactually changes
the next state. In yoked mode the same observation is replayed without current
control. It is removed when a governed production environment port supplies
typed observations, intervention identity, delay semantics, receipt binding,
and restart qualification. It is not a language-model or production-world
authority.

Matched observations therefore need not imply matched attribution. Control
evidence, confounders, external actors, prediction mismatch, and delay reduce a
bounded probability rather than yielding a certainty. Brier scores are stored
as calibration telemetry; ownership-related telemetry is not a belief or a
claim of consciousness.

## Failure and recovery boundary

Missing or mismatched adapter, assessor, receipt, or effect references fail
before a candidate is stored. Exact replays are idempotent while conflicting
replays fail closed. Startup verifies the SQLite integrity result, strict
schemas, canonical documents, every digest link, state invariants, and
calibration rows. The update and lineage ports are explicitly idempotent by
assessment digest, allowing restart to resume from every durable boundary. A
chain is marked reconciled only after the update commit and lineage head
references exist. This slice has local restart
evidence, but is **M2**: effect execution and reafference are not yet wired into
the composed request path, and the update/lineage callbacks do not yet share one
database transaction with the reafference store. Gate F and M3 remain open
until that canonical atomic wiring and exact-artifact restart qualification are
complete.
