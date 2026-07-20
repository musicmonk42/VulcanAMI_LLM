from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from vulcan.graphix.epistemic import *
from vulcan.microkernel.ledger import AuthoritativeClaimLedger, ClaimNotCommittedError, LedgerError

NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)
D = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64

def evidence(eid="evidence:one", kind=EvidenceKind.PROOF, *, valid_until=None, citations=()):
    return EvidenceArtifact(eid, kind, "episode:one", D, D2, "prov:kernel", NOW, valid_until, citations)

def claim(cid="claim:one", status=ClaimStatus.PROVEN, ev=("evidence:one",), deriv=(), contested=()):
    return Claim(cid, Proposition("prop:" + cid.split(":")[-1], "s", "p", "o"), status, "episode:one", D, ev, deriv, contested_by=contested)

def commit(**kw):
    base = dict(commit_id="commit:one", episode_id="episode:one", case_id="case:one", snapshot_digest=D, authority_principal_id="principal:kernel", committed_at=NOW, claims=(claim(),), evidence=(evidence(),))
    base.update(kw)
    return EpistemicCommit(**base)

def test_valid_commit_digest_and_ledger_authority_boundary():
    c = commit()
    ledger = AuthoritativeClaimLedger()
    with pytest.raises(ClaimNotCommittedError):
        ledger.require_committed_claim("claim:one")
    assert ledger.append(c) == c.commit_digest
    assert ledger.require_committed_claim("claim:one").commit_id == "commit:one"

def test_dangling_and_cross_episode_references_fail_closed():
    with pytest.raises(ReferenceValidationError):
        commit(claims=(claim(ev=("missing:evidence",)),))
    cross = replace(evidence(), episode_id="episode:two")
    with pytest.raises(ReferenceValidationError):
        commit(evidence=(cross,))
    explicit = replace(cross, source_episode_id="episode:two-original")
    assert commit(evidence=(explicit,)).evidence[0].source_episode_id == "episode:two-original"

def test_circular_derivation_fails_closed():
    c1 = claim("claim:one", ClaimStatus.COMPUTED, deriv=("deriv:one",), ev=())
    c2 = claim("claim:two", ClaimStatus.COMPUTED, deriv=("deriv:two",), ev=())
    d1 = Derivation("deriv:one", ("claim:two",), (), "rule:modus", "claim:one")
    d2 = Derivation("deriv:two", ("claim:one",), (), "rule:modus", "claim:two")
    with pytest.raises(CircularDerivationError):
        commit(claims=(c1,c2), evidence=(), derivations=(d1,d2))

def test_status_specific_evidence_requirements():
    with pytest.raises(EvidenceIntegrityError):
        commit(claims=(claim(status=ClaimStatus.PROVEN),), evidence=(evidence(kind=EvidenceKind.OBSERVATION),))
    retrieved = claim(status=ClaimStatus.RETRIEVED)
    with pytest.raises(EvidenceIntegrityError):
        commit(claims=(retrieved,), evidence=(evidence(kind=EvidenceKind.RETRIEVAL),))
    cited = evidence(kind=EvidenceKind.RETRIEVAL, citations=(Citation("citation:one", uri="https://example.test/source"),))
    assert commit(claims=(retrieved,), evidence=(cited,)).claims[0].status is ClaimStatus.RETRIEVED

def test_expired_evidence_contested_claims_and_digest_tamper():
    with pytest.raises(TemporalValidityError):
        commit(evidence=(evidence(valid_until=NOW - timedelta(seconds=1)),))
    with pytest.raises(ReferenceValidationError):
        commit(claims=(claim(status=ClaimStatus.CONTESTED, ev=()),), evidence=())
    contested = claim(status=ClaimStatus.CONTESTED, ev=(), contested=("claim:other",))
    assert commit(claims=(contested,), evidence=()).claims[0].status is ClaimStatus.CONTESTED
    good = commit()
    with pytest.raises(EvidenceIntegrityError):
        commit(commit_digest="sha256:" + "f" * 64)
    assert digest_commit(good, include_digest=False) == good.commit_digest

def test_uncertainty_is_typed_not_scalar_confidence():
    with pytest.raises(EpistemicContractError):
        UncertaintyDescriptor(UncertaintyKind.PROBABILITY_DISTRIBUTION)
    u = UncertaintyDescriptor(UncertaintyKind.INTERVAL, interval_low="1/3", interval_high="2/3")
    assert u.kind is UncertaintyKind.INTERVAL

def test_failpoints_and_restart_reconciliation():
    c = commit()
    ledger = AuthoritativeClaimLedger(failpoint=lambda name, _c: (_ for _ in ()).throw(LedgerError(name)) if name == "after_append" else None)
    with pytest.raises(LedgerError, match="after_append"):
        ledger.append(c)
    ledger.failpoint = None
    ledger.reconcile()
    assert ledger.require_committed_claim("claim:one").commit_digest == c.commit_digest

def test_malformed_provider_cannot_escalate_claim_authority():
    malicious = {"claim_id":"claim:evil", "status":"PROVEN", "authority_level":"COMMITTED_BELIEF", "evidence_ids":[]}
    with pytest.raises(TypeError):
        Claim(**malicious)

def test_compatibility_projection_and_schema():
    projected = project_semantic_claim(claim_id="claim:legacy", episode_id="episode:one", snapshot_digest=D, subject="legacy", predicate="says", object_value="value")
    assert projected.status is ClaimStatus.HYPOTHESIS
    schema = json.loads(Path("schemas/graphix/epistemic-v1.json").read_text())
    assert schema["additionalProperties"] is False
    assert "confidence" not in json.dumps(schema).lower()
