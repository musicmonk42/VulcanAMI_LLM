# Graphix Core v1

Graphix Core v1 is the immutable, content-addressed envelope shared by Graphix cognitive dialects. The cognitive microkernel remains the authoritative owner; Graphix Core records authority context but does not grant or execute authority.

## Contract

The envelope carries closed enum fields for authority level, epistemic status, privacy class, and source kind. It separates display identifiers (`node_artifact_id`, `episode_id`) from authority-bearing SHA-256 digests (`content_digest`, `snapshot_bundle_digest`, source digests, and extension schema digests).

JSON decoding is strict: duplicate keys, unknown envelope fields, non-finite or oversized numbers, control characters, executable-looking keys, and digest mismatches fail closed. Canonical serialization uses UTF-8 JSON with sorted keys and compact separators.

Extensions are bounded, reverse-DNS namespaced declarations. They may contain display metadata only and must not carry policy, command, code, callable, import, or authority semantics.

## Dialect registry

Dialects register at startup against the serving release and the registry is then frozen. Unknown dialects or versions are rejected unless an explicit compatibility migration function is registered.

## Migration and rollback

Graphix Core is additive. Rollback is to stop producing v1 envelopes and keep rejecting unsupported dialect/version pairs; persisted v1 records remain verifiable by digest and schema.
