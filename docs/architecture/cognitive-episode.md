# Authoritative CognitiveEpisode contract

`vulcan.microkernel.episode.CognitiveEpisode` is the immutable request-scoped aggregate for chat, reasoning, tools, learning, CSIU, and improvement. Legacy `CognitiveCase` remains as a compatibility adapter and must not become a second authority.

## Lifecycle

Episodes move only through typed transitions in `vulcan.microkernel.state_machine`: `PERCEIVED`, `INTERPRETED`, `GROUNDED`, `DELIBERATING`, `EPISTEMICALLY_COMMITTED`, `NORMATIVELY_AUTHORIZED`, `EXECUTED`, `OBSERVED`, `COMMUNICATED`, and `CONSOLIDATED`. Terminal failure outcomes are `ABSTAINED`, `BLOCKED`, `FAILED`, and `CANCELLED`.

Every transition records an event with the prior episode digest, transition reason, authority, snapshot identifiers, and evidence references. Canonical JSON uses sorted keys and compact separators; the episode digest is the SHA-256 digest of the canonical payload excluding the digest field.

## Durable retention policy

Raw request bytes may exist only in working memory long enough to compute a digest and approved projection digest. Durable episode and audit stores persist `input_digest`, optional approved `projection_digest`, references, and canonical digests; they must not persist raw prompts, raw provider text, secrets, hidden prompts, or private reasoning traces.

## Migration and rollback

`vulcan.runtime.case.episode_from_case` is the single explicit adapter from legacy `CognitiveCase` to `CognitiveEpisode`. Rollback keeps the adapter while callers continue using `CognitiveCase`; no durable schema may introduce a separate cognitive authority.
