# Governance, Consensus & Evolution

## 1. Goals
- Safe, accountable grammar & graph artifact evolution
- Weighted trust participation
- Alignment & safety gate before apply
- Replay & duplication dampening

## 2. Proposal Structure
```json
{
 "proposal_id": "prop_2025_0042",
 "type": "ontology_update",
 "changeset": {
 "add_nodes": [{ "type_uri": "https://graphix.ai/ontology/AuditNode", "lifecycle": "active" }],
 "deprecate_nodes": ["https://graphix.ai/ontology/LegacyNode"]
 },
 "metadata": {
 "risk_estimate": 0.19,
 "description": "Add structured audit node",
 "tags": ["observability"]
 },
 "status": "open",
 "signatures": { "author_sig": "sig_abcd1234" },
 "critical": false
}
```

## 3. Voting & Trust Weight
approval_ratio = approve_weight / (approve_weight + reject_weight) 
Abstain excluded. Trust concentration warnings if > percentile threshold.

## 4. Thresholds & Quorum
- Quorum: votes / active_agents ≥ 0.51 (adjustable)
- Approval threshold baseline ~0.66 (bounded by min/max in registry)
- Critical proposals may push threshold upward (e.g., >0.75)

## 5. Lifecycle
draft → open → approved/rejected/expired → applied → completed/failed

Expired: TTL lapse without quorum decision. Failed Application: rollback & audit entry.

## 6. Replay & Similarity
- Replay window rejects identical hash inside REPLAY_WINDOW_SECONDS.
- Similarity (embedding cosine) > threshold triggers duplicate damping suggestion.

## 7. Alignment & Safety Overlay (VULCAN)
Motivational misalignment or safety violations preempt approval; surfaced as conflict reason list.

## 8. Metrics
- governance_votes_total{proposal_id, outcome}
- proposal_time_to_quorum_seconds
- duplicate_proposal_rejections_total
- replay_rejections_total
- alignment_conflict_total (future)

## 9. Failure Modes
| Mode | Cause | Mitigation |
|------|-------|------------|
| Stagnation | Insufficient quorum | Prompt nudge / dynamic quorum adjustment |
| Veto Concentration | Single high-trust dominate | Diversity weighting / cap |
| Similarity Flood | Many near-duplicates | Merge suggestion pipelines |
| Safety Rejection Spike | Frequent unsafe proposals | Strengthen risk scoring & training |

## 10. Best Practices
- Atomic changesets
- Clear rationale referencing risk & impact analysis
- Dry-run semantics prior to submission
- Lifecycle marking for experimental types

## 11. Roadmap
Multi-tier federation, predictive approval guidance, formal invariant registry, cross-instance semantic sync, external anchoring (on-chain hash root).
