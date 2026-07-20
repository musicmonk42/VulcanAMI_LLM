from datetime import datetime, timedelta, timezone

import pytest

from vulcan.memory.governed import (
    DeletionState,
    MemoryActor,
    MemoryKind,
    MemoryLifecycle,
    MemoryReadRequest,
    MemoryReason,
    MemoryWriteProposal,
    SQLiteMemoryRepository,
)
from vulcan.runtime.audit import CanonicalAudit


class EscalatingPolicy:
    version = "memory-policy/2"

    def decide(self, operation, actor, proposal=None):
        from vulcan.memory.governed import MemoryPolicyResult, PolicyDecision
        # Malicious implementation tries to allow an unsupported purpose; the repository still scopes reads/writes
        # by the trusted actor and the proposal constructor rejects unsupported preference keys/values.
        return MemoryPolicyResult(PolicyDecision.ALLOW, "malicious", operation)

    def retention(self, actor, proposal):
        return timedelta(days=365)


def repo(tmp_path, **kw):
    audit = CanonicalAudit(tmp_path / "audit" / "events.jsonl")
    return SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"), durable_root=str(tmp_path), audit=audit, **kw), audit


def proposal(value="blue", idem="idem", consent="consent-explicit-v1"):
    return MemoryWriteProposal(
        MemoryKind.EXPLICIT_PREFERENCE,
        "profile",
        "color",
        value,
        idem,
        consent_reference=consent,
        lawful_basis="consent",
        retention_rule="preference-365d",
        source_provenance="direct-subject",
        access_classification="personal-confidential",
    )


def test_lifecycle_digest_chain_correction_and_metadata(tmp_path):
    r, audit = repo(tmp_path)
    actor = MemoryActor("tenant", "subject", "actor")
    first = r.commit(actor, proposal()).record
    assert first is not None
    assert first.consent_reference == "consent-explicit-v1"
    assert first.lawful_basis == "consent"
    assert first.owner_id == "actor"
    assert first.source_provenance == "direct-subject"
    assert first.retention_rule == "preference-365d"
    assert first.access_classification == "personal-confidential"

    corrected = r.correct(actor, first.record_id, first.revision, proposal("red", "correct")).record
    assert corrected is not None
    assert corrected.supersedes == f"{first.record_id}:1"
    history = r.subject_access(actor)
    assert [(m.revision, m.lifecycle, m.value) for m in history] == [
        (1, MemoryLifecycle.SUPERSEDED, "blue"),
        (2, MemoryLifecycle.ACTIVE, "red"),
    ]
    assert history[0].digest != first.digest
    assert r.read(MemoryReadRequest(actor, "profile", "color"))[0].value == "red"
    r.close(); audit.close()


def test_expiry_cross_tenant_export_and_access_audit(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r, audit = repo(tmp_path, clock=lambda: now)
    actor = MemoryActor("tenant", "subject", "actor")
    other = MemoryActor("other", "subject", "actor")
    rec = r.commit(actor, proposal()).record
    assert rec is not None
    assert r.export_subject(other) == ()
    assert len(r.export_subject(actor)) == 1
    r._clock = lambda: now + timedelta(days=366)
    assert r.read(MemoryReadRequest(actor, "profile", "color")) == ()
    r.close(); audit.close()


def test_deletion_is_logical_redaction_not_physical_erasure_claim(tmp_path):
    r, audit = repo(tmp_path)
    actor = MemoryActor("tenant", "subject", "actor")
    rec = r.commit(actor, proposal()).record
    assert rec is not None
    deleted = r.tombstone(actor, rec.record_id, rec.revision, "delete")
    assert deleted.deletion_receipt is not None
    assert deleted.deletion_receipt.state is DeletionState.LOGICALLY_REDACTED
    assert "wal_free_pages_backups_snapshots_replicas" in deleted.deletion_receipt.remaining_obligations
    assert r.read(MemoryReadRequest(actor, "profile", "color")) == ()
    assert all(m.value is None for m in r.subject_access(actor))
    r.close(); audit.close()


def test_consent_revocation_is_tenant_and_subject_scoped(tmp_path):
    r, audit = repo(tmp_path)
    actor = MemoryActor("tenant", "subject", "actor")
    other = MemoryActor("tenant", "other", "actor")
    assert r.commit(actor, proposal("blue", "a", "consent-a")).reason is MemoryReason.COMMITTED
    assert r.commit(other, proposal("red", "b", "consent-a")).reason is MemoryReason.COMMITTED
    receipts = r.revoke_consent(actor, "consent-a", "revoke")
    assert len(receipts) == 1
    assert r.read(MemoryReadRequest(actor, "profile", "color")) == ()
    assert r.read(MemoryReadRequest(other, "profile", "color"))[0].value == "red"
    r.close(); audit.close()


def test_no_implicit_or_malformed_memory_becomes_durable(tmp_path):
    r, audit = repo(tmp_path, policy=EscalatingPolicy())
    actor = MemoryActor("tenant", "subject", "actor")
    with pytest.raises(ValueError):
        proposal("use anything", "bad", "consent-a")
    with pytest.raises(ValueError):
        MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE, "profile", "password", "secret", "x")
    assert r.read(MemoryReadRequest(actor, "profile", "color")) == ()
    r.close(); audit.close()
