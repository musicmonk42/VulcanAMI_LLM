# Graphix Epistemic v1

Graphix Epistemic is the typed authority boundary for claims. The authoritative owner is `EpistemicCommit`: claims become citable only after a commit has validated its episode, case, snapshot, evidence integrity, temporal validity, provenance identifiers, and authority principal.

## Status invariants

Statuses are semantic categories, not a scalar confidence field: `PROVEN`, `DISPROVEN`, `COMPUTED`, `OBSERVED`, `RETRIEVED`, `ESTIMATED`, `HYPOTHESIS`, `CONTESTED`, `UNKNOWN`, and `ERROR` remain distinct. Estimated claims use `UncertaintyDescriptor` with a distribution digest, interval, calibration identity, or unknown marker.

## Evidence and reuse

Evidence is bound to an episode and snapshot. Cross-episode reuse must be represented by an explicit `EvidenceArtifact.source_episode_id`; claim objects are not shared across episodes. `PROVEN` requires proof evidence. `RETRIEVED` requires cited retrieval evidence. Expired evidence fails closed at commit time.

## Ledger boundary

`AuthoritativeClaimLedger` is append-only and idempotent by commit id and digest. `require_committed_claim` rejects uncommitted claims so response generation or learning-positive outcomes cannot cite proposal-only claims. Append failpoints surround the externally observable mutation, and `reconcile()` rebuilds claim indexes after restart.

## Compatibility

`project_semantic_claim` is the single compatibility adapter for legacy semantic claim shapes. It projects legacy values into typed `Claim` instances without granting committed authority.
