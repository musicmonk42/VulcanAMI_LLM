# ADR 0005: Governed durable memory

## Decision

The canonical runtime owns exactly one `GovernedMemoryService`.  When disabled
(the default), it rejects mutation rather than falling back to process memory.
When enabled, the only supported backend is a one-replica SQLite database at an
explicit configured path.  Historical graph, vector, pickle, hierarchical, and
unlearning packages are research or migration-only code and are not composed by
`vulcan.runtime`.

The supported record is a minimized explicit user preference.  It is tenant,
subject, actor, purpose, policy-version, retention, revision, lifecycle, and
digest bound.  Raw transcripts, answers, hidden reasoning, arbitrary metadata,
callables, embeddings, and procedures are not accepted.

## Lifecycle and recovery

Writes are idempotent SQLite transactions with journal and outbox rows.
Corrections append immutable revisions; reads choose the latest active revision.
Forget appends a tombstone revision before returning, so restarts cannot serve
the preceding revision.  The tombstone and journal are logical denial controls,
not a claim of physical media sanitization or model-weight unlearning.

## Residual limitations

This decision establishes neither HA/multi-writer support, backup media purge,
key management, semantic/vector retrieval, legacy-data migration, nor legal or
privacy compliance.  Enabling durable memory requires a maintenance-owned path
and a single runtime replica; other topology values fail composition.

## Current operational boundary

The canonical configuration has no derived projection/outbox and no supported
legacy migration or arbitrary backup restore. The inventory at
`docs/memory/legacy-inventory.json` freezes legacy writers as research-only or
read-only migration sources. The operational semantics and limits are recorded
in `docs/memory/governed-memory-operations.md`.
