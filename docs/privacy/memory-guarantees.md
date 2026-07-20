# Governed memory guarantees

The production memory authority is the narrow SQLite preference store in `src/vulcan/memory/governed.py`. It only stores allowlisted `profile` preferences and does not accept arbitrary transcript, implicit signal, vector, or graph memory as durable relational preference.

## Semantics

This implementation intentionally uses a **mutable redaction store**. Correction appends a new revision and marks the prior revision `SUPERSEDED`; deletion/revocation appends a tombstone and redacts retained payload values from supported application read paths. The store must not be described as immutable history because prior rows can be updated for supersession and redaction.

Deletion receipts return `LOGICALLY_REDACTED`. They do not claim storage-level erasure across SQLite WAL files, free pages, backups, snapshots, replicas, or operator copies unless a separate independent erasure proof exists.

## Envelope

Every memory revision carries tenant, subject, actor/owner, purpose, namespace/key, typed value, consent reference, lawful basis, retention rule, source provenance, access classification, expiry, lifecycle, supersession pointer, deletion epoch, policy/schema version, and canonical digest.

## Subject rights

The repository exposes scoped subject access/export, correction, consent revocation, and logical deletion APIs. All are tenant/subject/purpose scoped by a server-owned `MemoryActorContext`; caller-supplied raw model text is never accepted as authority.
