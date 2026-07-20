# Segmented canonical audit v2

`vulcan-audit/2` is the authoritative runtime audit format. `CanonicalAudit` keeps exclusive writer ownership with an advisory lock next to a descriptor-safe audit directory and rejects symlinks or path replacement for the root, manifest, lock, and segment files.

The manifest (`manifest.json`) records the active numbered segment, next event sequence, last event hash, closed segment records, and an optional `vulcan-audit/1` source digest. Segment files are immutable after rotation. Each event contains a global sequence, per-segment sequence, previous event hash, canonical JSON payload, and SHA-256 event hash. Segment rotation writes a segment close record containing the previous segment digest and the closed segment digest, then atomically replaces the manifest.

Append durability is separate from anchoring/checkpoint policy through `AuditDurabilityProfile`. Safe profiles may disable per-event fsync only for callers that can tolerate replay from the last durable manifest; manifest replacement remains explicit and fail-closed.

When a legacy `vulcan-audit/1` JSONL file exists, v2 verifies it without mutation and writes an `audit.migration_boundary` event in the first v2 segment containing the legacy digest and event count. Rollback is therefore to keep the legacy JSONL and remove the `.d` audit directory before restarting the old binary.

Bounded export uses `export_archive(max_bytes=...)`; verification is run before export and closed segment chain digests are preserved.
