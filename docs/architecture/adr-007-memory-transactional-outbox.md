# ADR-007: Governed memory DB-first transactional outbox

Status: accepted

## Context

Governed preference memory must have one durable authority for memory effects.
Audit publication is an effect description, not the authority that advances a
memory head.  A process can crash after SQLite commits and before audit delivery,
or after audit delivery and before the outbox row is marked delivered.

## Decision

SQLite is the authoritative owner for governed memory publication.  A memory
write transaction commits the immutable revision, authoritative head, idempotency
record, journal entry, and deterministic outbox event in one `BEGIN IMMEDIATE`
transaction.  Audit append occurs only after that database commit.  The outbox
`operation_id` is a deterministic event ID derived from the request digest,
record ID, revision, and event type; retry delivery treats canonical duplicate
transaction responses as an already-delivered audit effect and then marks the
row delivered in SQLite.

## Recovery contract

On startup/readiness, pending outbox rows are flushed.  If audit is unavailable,
the committed memory head remains authoritative and the outbox row stays pending.
If a process terminates after audit append but before `delivered_at`, restart
replays the same deterministic transaction ID and canonical audit deduplication
prevents a duplicate audit effect.  Stale base revisions and duplicate
idempotency keys are terminal domain outcomes, not retryable corruption.

## Rollback

Rollback is schema-aware: keep the SQLite database backup taken before migration
or run an operator-reviewed export/import into the prior schema.  Do not delete
pending outbox rows during rollback; first restore audit availability and flush
or preserve them with the database backup.
