# Graphix Epistemic v1

Graphix Epistemic is the typed commit format for claims. The cognitive
microkernel is the sole authority that may create a `COMMITTED_BELIEF`;
`EpistemicCommit` validates and represents that decision but cannot promote its
own authority. Claims become citable only after the kernel commits a document
that has validated its episode, case, snapshot, evidence integrity, temporal
validity, provenance identifiers, and authority principal.

## Semantic integrity boundary

Previously, the commit digest covered claim and proposition identifiers but
omitted proposition content and qualifiers, structured citations, uncertainty,
limitations, assumptions, counterexamples, contradictions, and the canonical
authority level. The new digest transaction covers every one of those fields,
the snapshot and prior commit, and verifies a supplied digest while decoding.
Canonical dump/load round trips preserve the complete semantic document.

## Status invariants

Statuses are semantic categories, not a scalar confidence field: `PROVEN`, `DISPROVEN`, `COMPUTED`, `OBSERVED`, `RETRIEVED`, `ESTIMATED`, `HYPOTHESIS`, `CONTESTED`, `UNKNOWN`, and `ERROR` remain distinct. Estimated claims use `UncertaintyDescriptor` with a distribution digest, interval, calibration identity, or unknown marker.

## Evidence and reuse

Evidence is bound to an episode and snapshot. Cross-episode reuse must be represented by an explicit `EvidenceArtifact.source_episode_id`; claim objects are not shared across episodes. `PROVEN` requires proof evidence. `RETRIEVED` requires cited retrieval evidence. Expired evidence fails closed at commit time.

## Ledger boundary

`AuthoritativeClaimLedger` is append-only and idempotent by commit id and digest. `require_committed_claim` rejects uncommitted claims so response generation or learning-positive outcomes cannot cite proposal-only claims. Append failpoints surround the externally observable mutation, and `reconcile()` rebuilds claim indexes after restart.

## Compatibility

`project_semantic_claim` is the single compatibility adapter for legacy semantic claim shapes. It projects legacy values into typed `Claim` instances without granting committed authority.

Remove this adapter when Wave 2.2 retires the duplicate runtime semantic claim
contract. `AuthoritativeClaimLedger` remains an in-memory compatibility ledger;
remove it when Wave 1.6 installs the durable Graphix Epistemic head. This repair
does not wire Graphix into the canonical request path and therefore does not
advance it to M3.
