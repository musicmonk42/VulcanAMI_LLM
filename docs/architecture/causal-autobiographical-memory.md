# Causal autobiographical memory

## Status

M2: durable bounded protocol with local evidence; canonical composition and
exact-artifact restart qualification are pending.

## Authority and transaction boundary

Previously, the causal loop ended with a reconciled reafference and an advanced
lineage head. Existing episodic and governed-memory implementations could store
content, but none was authorized to constitute the continuing lineage's causal
history. The new boundary is the kernel-only
`AutobiographicalMemoryStore` transaction. It commits an approved,
content-addressed `AutobiographicalEpisode` and its audit-outbox event in one
SQLite transaction. It does not replace the immutable episode, lineage,
reafference, effect, or audit authorities.

The store requires a `BiographyChainAuthority` and invokes it before every
write, authoritative read, retrieval, and restart verification. Composition
therefore fails closed when the durable episode/effect/reafference/lineage chain
is absent or contradictory; the memory store cannot validate itself into
authority. The validator is a required authority port, not a compatibility
adapter.

Each autobiographical episode binds prior and next lineage states, the bounded
cognitive episode, interpretation, committed beliefs, policy, expected effect,
intent, receipt, observation, reafference, corrections, and four distinct fact
kinds: observation, inference, prediction, and outcome. Facts contain artifact
digests and explicit causal/model/policy edges, never raw prompts, provider
text, hidden prompts, secrets, or private reasoning. A restart verifies the
canonical documents and revision chain rather than reconstructing identity from
a prose summary.

Corrections append a new revision and supersede the active projection without
rewriting the old document. Authorized tombstones erase all retrievable memory
documents while retaining scope/index metadata, content digests, and the
minimum revision chain needed to prove governed deletion. This is logical
application-level deletion, not a claim of forensic erasure from storage media
or backups.
Retrieval is tenant, person, purpose, and active-state scoped and ranks only
explicit causal, policy, or model edges; textual similarity has no authority.

## Compatibility and removal

`autobiography-outbox-callback` is the single retained adapter. It projects
digest-only events to the existing immutable audit append interface and is
removed when autobiography and canonical audit share a typed outbox dispatcher.
The existing generic memory packages are not adapters to this authority and
remain outside this causal-history path.

## Remaining boundary

This slice is not M3. The canonical request/effect/reafference path does not yet
invoke the store, and the episode, lineage, reafference, and autobiography heads
do not yet advance in one cross-store transaction. Canonical composition must
also validate consent and retention against admitted policy authority rather
than accepting the already-authorized scope fields supplied here. Until that
composition supplies a production `BiographyChainAuthority`, startup must fail
closed rather than substituting a permissive validator.
