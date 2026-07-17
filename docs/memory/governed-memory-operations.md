# Governed durable-memory operations

The only supported durable record is a typed explicit preference: `locale`,
`response_style`, `unit_system`, or the transitional `color` test surface. Each
has a closed set of values enforced before persistence. Raw prompts, answers,
reasoning, metadata, embeddings, procedures, and secrets are unsupported.

`remember` creates a logical preference; `correct` requires the current
revision; and `forget` advances the authoritative head to a content-free
tombstone and erases payload values in the canonical SQLite store. A tombstone
is logical read denial and not a claim of offline-media sanitization, model
unlearning, encryption, cryptographic audit, HA, or legal compliance.

There is no derived projection or outbox in the supported configuration.
Recovery and migration are unsupported until a validated lifecycle-replay
bundle exists; operators must not restore arbitrary old SQLite files. Legacy
stores are frozen/quarantined according to `legacy-inventory.json`; rollback
must leave them disabled and cannot restore a pre-tombstone head.

SQLite is single-writer only. Enablement requires a configured durable root,
an absolute database beneath that root, one replica, and exclusive ownership.
