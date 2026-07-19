from datetime import datetime, timezone
import dataclasses
import os
from pathlib import Path
import sqlite3
import pytest

from vulcan.learning_observation import ObservationContext, ProvenanceType, TerminalStatus, construct_observation, digest_json
from vulcan.learning_outbox import LearningDeliveryStatus, LearningObservationOutbox, LearningOutboxError, IdempotencyConflictError, ManualRecoveryRequiredError
from vulcan.runtime.audit import CanonicalAudit, AuditError


def h(x): return digest_json({"x": x})

def obs(label="a", *, owner="learning-owner"):
    now = datetime.now(timezone.utc).replace(microsecond=123456)
    ctx = ObservationContext(
        case_id=f"case-{label}", case_digest=h("case"+label), request_digest=h("req"+label), tenant_digest=h("tenant"),
        alignment_revision=1, alignment_digest=h("align"), csiu_policy_digest=h("csiu-p"), csiu_snapshot_digest=h("csiu-s"),
        domain_snapshot_digest=h("domain"), runtime_owner_id=owner, acquisition_time=now)
    ob, elig = construct_observation(context=ctx, selected_plan_digest=h("plan"+label), selected_tool_id="graphix_arithmetic", selection_distribution_digest=h("dist"+label), action_propensity=0.5, terminal_status=TerminalStatus.VALIDATED_SUCCESS, ledger_digest=h("ledger"+label), evidence_digest=h("evidence"+label), provenance_type=ProvenanceType.DERIVATION, terminal_case_validated=True, ledger_validated=True, evidence_integrity_validated=True, bindings_match=True, alignment_matches_lease=True, csiu_bindings_valid=True, clock=lambda: now)
    assert elig.status.value == "eligible_positive"
    return ob


def audit(tmp_path): return CanonicalAudit(tmp_path / "audit.jsonl")


def outbox(tmp_path, **kw): return LearningObservationOutbox(tmp_path / "learning.db", audit=audit(tmp_path), **kw)


def test_deliver_publishes_once_and_replay_returns_original(tmp_path):
    ob = obs()
    o = outbox(tmp_path)
    r1 = o.deliver(ob, expected_revision=0)
    r2 = o.deliver(ob, expected_revision=0)
    assert r1.status is LearningDeliveryStatus.PUBLISHED
    assert r2.status is LearningDeliveryStatus.REPLAYED
    assert r1.transaction_id == r2.transaction_id
    assert o.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 1


def test_reuse_observation_id_with_different_content_conflicts(tmp_path):
    o = outbox(tmp_path)
    first = obs("a")
    second = dataclasses.replace(obs("b"), observation_id=first.observation_id)
    from vulcan.learning_observation import digest_json
    second = dataclasses.replace(second, canonical_observation_digest=digest_json(second.canonical_payload(include_digest=False)))
    o.deliver(first, expected_revision=0)
    with pytest.raises(IdempotencyConflictError):
        o.deliver(second, expected_revision=1)


@pytest.mark.parametrize("failpoint", ["before_prepared_audit", "after_prepared_audit", "during_candidate_persistence"])
def test_early_failures_publish_and_persist_nothing(tmp_path, failpoint):
    o = outbox(tmp_path, failpoint=failpoint)
    with pytest.raises(LearningOutboxError):
        o.deliver(obs(), expected_revision=0)
    assert o.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    assert o.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 0


@pytest.mark.parametrize("failpoint", ["after_candidate_persistence", "during_committed_audit"])
def test_candidate_failure_does_not_publish(tmp_path, failpoint):
    o = outbox(tmp_path, failpoint=failpoint)
    with pytest.raises(LearningOutboxError):
        o.deliver(obs(), expected_revision=0)
    assert o.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 0


def test_after_commit_failure_recovers_on_reopen(tmp_path):
    ob = obs()
    o = outbox(tmp_path, failpoint="after_committed_audit")
    with pytest.raises(LearningOutboxError):
        o.deliver(ob, expected_revision=0)
    o.close()
    recovered = outbox(tmp_path)
    recovered.readiness()
    assert recovered.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 1


def test_after_publication_before_ack_is_retried_without_second_effect(tmp_path):
    ob = obs()
    o = outbox(tmp_path, failpoint="after_publication_before_ack")
    with pytest.raises(LearningOutboxError):
        o.deliver(ob, expected_revision=0)
    assert o.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 1
    o.close()
    recovered = outbox(tmp_path)
    assert recovered.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 1
    r = recovered.deliver(ob, expected_revision=0)
    assert r.status is LearningDeliveryStatus.REPLAYED


def test_competing_expected_revision_returns_stale(tmp_path):
    o = outbox(tmp_path)
    r1 = o.deliver(obs("a"), expected_revision=0)
    r2 = o.deliver(obs("b"), expected_revision=0)
    assert r1.status is LearningDeliveryStatus.PUBLISHED
    assert r2.status is LearningDeliveryStatus.STALE_REVISION


def test_audit_unavailable_prepared_failure_writes_nothing(tmp_path):
    class BadAudit:
        def append(self, *a, **k): raise AuditError("down")
    o = LearningObservationOutbox(tmp_path / "learning.db", audit=BadAudit())
    with pytest.raises(AuditError): o.deliver(obs(), expected_revision=0)
    assert o.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0


def test_symlink_database_and_second_writer_fail(tmp_path):
    target = tmp_path / "target.db"; target.write_text("")
    link = tmp_path / "link.db"; link.symlink_to(target)
    bad_audit = audit(tmp_path / "bad")
    with pytest.raises(LearningOutboxError): LearningObservationOutbox(link, audit=bad_audit)
    bad_audit.close()
    o = outbox(tmp_path)
    with pytest.raises(LearningOutboxError): LearningObservationOutbox(tmp_path / "learning.db", audit=audit(tmp_path / "a2"))
    o.close()


def test_direct_row_tampering_fails_readiness_on_reopen(tmp_path):
    ob = obs()
    o = outbox(tmp_path); o.deliver(ob, expected_revision=0); o.close()
    con = sqlite3.connect(tmp_path / "learning.db")
    con.execute("UPDATE transactions SET candidate_digest='bad'"); con.commit(); con.close()
    with pytest.raises(ManualRecoveryRequiredError): outbox(tmp_path)


def test_close_reopen_safe(tmp_path):
    o = outbox(tmp_path); o.close(); o.close()
    o2 = outbox(tmp_path); assert o2.readiness() is True; o2.close()
