# Legacy migration and journal audit projection

## Authority and threat boundary

The constitutional journal is the sole Phase-A persistence authority. The legacy
episode, epistemic, lineage, and segmented-audit files are untrusted migration
inputs or disposable projections; none may advance a live cognitive head.

## Offline migration transaction boundary

Migration is exclusive, offline, and one-way. Operators approve canonical logical
digests for both legacy databases and supply an authenticated migration
`ActorBinding` plus a non-secret credential-provenance digest. The migrator:

1. rejects symlinks and locks both source files exclusively;
2. opens SQLite read-only and validates integrity, schemas, row counts, episode,
   epistemic, and lineage replay chains and cross-store episode references;
3. emits a reconciliation report and installs nothing on disagreement;
4. imports deterministically ordered source evidence and records through the
   caller-owned journal API into a same-directory temporary database;
5. classifies historical identity labels as `LEGACY_UNVERIFIED`, distinct from the
   authenticated migration workload actor, and rejects credential material;
6. verifies, checkpoints, fsyncs, and atomically installs the journal; and
7. makes the source databases and extant WAL/SHM files read-only rollback evidence.

There is no reverse or in-place migration and no production dual write.

## Audit projection boundary

`JournalAuditProjector` orders exclusively by `(commit_seq, event_ordinal)` and
retains the journal receipt hashes. Its JSONL bytes are canonical and reproducible.
Rebuild uses fsync plus atomic replacement. Missing, stale, deleted, or unavailable
JSONL never changes readiness or cognition; only journal integrity and recoverable
outbox state are authoritative.

## Recovery qualification

The parent/child harness requires a durable marker proving the child reached the
named boundary before sending `SIGKILL`. A fresh process then opens and verifies the
same volume as the complete prior or successor state. Unreached boundaries are
`NOT_EXECUTED`. Qualification covers artifacts, heads, terminal rows, outbox
insertion, commit acknowledgement, lock contention, WAL truncation, deliberate
corruption, and normal/optimized Python.

## Compatibility removal

`CanonicalAudit` and the legacy stores remain for non-Phase-A readers and rollback
evidence. Remove them after all operator migration windows close and historical
consumers use journal-derived projections. The next change may introduce live
transition permits; it must not grant mutation authority to these compatibility
types.
