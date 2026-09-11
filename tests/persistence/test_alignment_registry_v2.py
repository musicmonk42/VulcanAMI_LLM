import hashlib
import json
import multiprocessing as mp
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from vulcan.graphix.evaluation import build_epistemic_candidate
from vulcan.graphix.runtime import (
    Claim,
    Derivation,
    EpistemicStatus,
    EvidenceArtifact,
    EvidenceKind,
    ExecutionStatus,
    Proposition,
    canonical_digest,
)
from vulcan.runtime.alignment import (
    AlignmentRegistry,
    default_policy,
    trusted_admin_principal,
)


def _commit(claims, evidence, derivations, *, evaluated_at=None):
    at = evaluated_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    authority = SimpleNamespace(
        principal=SimpleNamespace(identity_digest="a" * 64, release_digest="b" * 64),
        validation_digest="c" * 64,
        policy_digest="d" * 64,
        grant=SimpleNamespace(evidence_digest="e" * 64),
    )
    return build_epistemic_candidate(
        episode_id="episode:test",
        case_id="case",
        snapshot_digest="f" * 64,
        claims=tuple(claims),
        evidence=tuple(evidence),
        derivations=tuple(derivations),
        authority=authority,
        prior_commit_digest=None,
        graphix_artifact_digest="1" * 64,
        evaluated_at=at,
    )


def cand(rev=2, max_claims=4):
    c = default_policy().__dict__.copy()
    c.update(revision=rev, max_claims_per_response=max_claims, policy_digest="")
    c["permitted_epistemic_statuses"] = list(c["permitted_epistemic_statuses"])
    c["policy_digest"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in sorted(c.items()) if k != "policy_digest"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return c


def _hold(path, q):
    r = AlignmentRegistry(path)
    q.put("open")
    q.get()
    r.close()


def test_two_process_single_writer_lock(tmp_path):
    q = mp.Queue()
    p = mp.Process(target=_hold, args=(tmp_path / "p.json", q))
    p.start()
    assert q.get(timeout=5) == "open"
    with pytest.raises(RuntimeError):
        AlignmentRegistry(tmp_path / "p.json")
    q.put("done")
    p.join(5)
    assert p.exitcode == 0


def test_double_close_lease_and_foreign_release_rejected(tmp_path):
    r = AlignmentRegistry(tmp_path / "p.json")
    l1 = r.lease()
    l2 = r.lease()
    with pytest.raises(RuntimeError):
        r.release(l1.policy_digest)
    assert l1.close() is True
    assert l1.close() is False
    assert l2.diagnostics().active is True
    l2.close()
    r.close()
    r.close()
    assert r.close_count == 1
    with pytest.raises(RuntimeError):
        r.active()


def test_retention_never_evicts_active_or_leased(tmp_path):
    r = AlignmentRegistry(tmp_path / "p.json", retention_limit=2)
    old = r.lease()
    prev = r.active().policy_digest
    for rev in range(2, 6):
        pol = r.update(
            cand(rev, rev + 2),
            expected_previous_digest=prev,
            principal=trusted_admin_principal("admin"),
        )
        prev = pol.policy_digest
    assert old.policy_digest in r._hist and r.active().policy_digest in r._hist
    old.close()
    r._evict()
    assert r.active().policy_digest in r._hist


def test_stale_cas_and_malicious_principal_boundary(tmp_path):
    r = AlignmentRegistry(tmp_path / "p.json")
    old = r.active().policy_digest
    r.update(
        cand(2),
        expected_previous_digest=old,
        principal=trusted_admin_principal("admin"),
    )
    with pytest.raises(ValueError):
        r.update(
            cand(3),
            expected_previous_digest=old,
            principal=trusted_admin_principal("admin"),
        )
    with pytest.raises(TypeError):
        r.update(
            cand(3),
            expected_previous_digest=r.active().policy_digest,
            principal="admin",
        )


def test_restart_recovery_after_audit_commit_publishes(tmp_path):
    fail = {"on": "after_audit_commit"}

    def fp(n):
        if n == fail["on"]:
            raise RuntimeError("boom")

    class A:
        def append(self, t, d):
            pass

    r = AlignmentRegistry(tmp_path / "p.json", audit=A(), failpoint=fp)
    old = r.active().policy_digest
    c = cand(2)
    with pytest.raises(RuntimeError):
        r.update(
            c,
            expected_previous_digest=old,
            principal=trusted_admin_principal("admin"),
            transaction_id="tx1",
        )
    r.close()
    r2 = AlignmentRegistry(tmp_path / "p.json")
    assert r2.active().policy_digest == c["policy_digest"]


def test_crash_before_audit_commit_preserves_prior(tmp_path):
    def fp(n):
        if n == "after_persist_candidate":
            raise RuntimeError("boom")

    r = AlignmentRegistry(tmp_path / "p.json", failpoint=fp)
    old = r.active().policy_digest
    c = cand(2)
    with pytest.raises(RuntimeError):
        r.update(
            c,
            expected_previous_digest=old,
            principal=trusted_admin_principal("admin"),
            transaction_id="tx2",
        )
    r.close()
    r2 = AlignmentRegistry(tmp_path / "p.json")
    assert r2.active().policy_digest == old


def _objects(valid_until):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ev = EvidenceArtifact(
        "evidence:one",
        EvidenceKind.RETRIEVED_RECORD,
        canonical_digest("v"),
        "ref",
        "origin",
        "exact",
        "case",
        observed_at=now,
        valid_until=valid_until,
        source_integrity="digest-verified",
        citation="ref",
    )
    d = Derivation(
        "derivation:one",
        "m",
        "1",
        ("evidence:one",),
        "claim:one",
        (),
        (),
        True,
        ExecutionStatus.SUCCESS,
        "t",
    )
    c = Claim(
        "claim:one",
        Proposition("s", "p", "v"),
        EpistemicStatus.RETRIEVED,
        ("evidence:one",),
        ("derivation:one",),
        citation_ids=("evidence:one",),
    )
    return c, ev, d


def test_decision_records_time_replay_and_requires_valid_until(tmp_path):
    r = AlignmentRegistry(
        tmp_path / "p.json", clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    c, ev, d = _objects(datetime(2026, 1, 2, tzinfo=timezone.utc))
    dec = r.decide(_commit((c,), (ev,), (d,)))
    assert dec.accepted and dec.evaluated_at == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )
    r.clock = lambda: datetime(2026, 1, 3, tzinfo=timezone.utc)
    assert dec.replay().accepted
    assert not dec.replay(
        reevaluate=True, registry=r, commit=_commit((c,), (ev,), (d,))
    ).accepted
    c2, ev2, d2 = _objects(None)
    assert "missing_valid_until" in r.decide(_commit((c2,), (ev2,), (d2,))).reason_codes
