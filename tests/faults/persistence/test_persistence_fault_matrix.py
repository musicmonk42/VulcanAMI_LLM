from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.qualification.persistence_fault_matrix import FAULT_POINTS, SUBSYSTEMS, report
from vulcan.memory.governed import MemoryActorContext, MemoryKind, MemoryReadRequest, MemoryReason, MemoryWriteProposal, SQLiteMemoryRepository
from vulcan.persistence.audit.reconcile import reconcile_transactions
from vulcan.runtime.alignment import AlignmentRegistry, default_policy, trusted_admin_principal
from vulcan.runtime.domain_registry import PersistentDomainRegistry


def _alignment_candidate(revision: int) -> dict[str, object]:
    candidate = default_policy().__dict__.copy()
    candidate.update(revision=revision, max_claims_per_response=revision + 2, policy_digest="")
    candidate["permitted_epistemic_statuses"] = list(candidate["permitted_epistemic_statuses"])
    payload = {k: v for k, v in sorted(candidate.items()) if k != "policy_digest"}
    candidate["policy_digest"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return candidate


def _domain_bundle(revision: int = 1, value: str = "paris") -> str:
    content = json.dumps({"subject": "france", "predicate": "capital", "object": value})
    doc = {
        "schema_version": "vulcan-domain/1",
        "domain": "geo",
        "version": f"v{revision}",
        "revision": revision,
        "evidence": [{
            "evidence_id": "e1",
            "uri": "https://example.test/geo/1",
            "content": content,
            "content_digest": hashlib.sha256(content.encode()).hexdigest(),
            "acquired_at": "2026-01-01T00:00:00Z",
            "acquisition_method": "reviewed-jsonl",
            "license": "CC0",
            "provenance": {"reviewer": "fault-suite"},
        }],
        "facts": [{"fact_id": "f1", "subject": "france", "predicate": "capital", "object": value, "evidence_ids": ["e1"]}],
    }
    doc["digest"] = hashlib.sha256(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return json.dumps(doc, separators=(",", ":"))


def test_generated_matrix_is_machine_verifiable_and_covers_required_transitions(tmp_path: Path) -> None:
    output = tmp_path / "qualification.json"
    subprocess.check_call([sys.executable, "scripts/qualification/persistence_fault_matrix.py", "--output", str(output)])
    data = json.loads(output.read_text())
    assert data == report()
    scenarios = data["scenarios"]
    assert data["w2_gate"]["machine_verifiable"] is True
    assert data["w2_gate"]["ambiguous_silent_success_allowed"] is False
    assert {s["subsystem"] for s in scenarios} >= set(SUBSYSTEMS)
    assert {s["fault_point"] for s in scenarios} >= set(FAULT_POINTS)
    assert any(s["execution"] == "subprocess_terminate" for s in scenarios)
    for critical in ("domain.rollback", "memory.audit_ordering", "alignment.lease_double_close", "audit.correlation"):
        assert any(s["scenario_id"] == critical and s["fault_class"] == "negative_control" for s in scenarios)


def test_alignment_restart_resolves_after_audit_commit_and_rejects_lease_double_close_control(tmp_path: Path) -> None:
    def fail_after_audit(point: str) -> None:
        if point == "after_audit_commit":
            raise RuntimeError("crash after audit commit")

    registry = AlignmentRegistry(tmp_path / "policy.json", failpoint=fail_after_audit)
    previous = registry.active().policy_digest
    candidate = _alignment_candidate(2)
    with pytest.raises(RuntimeError, match="crash after audit commit"):
        registry.update(candidate, expected_previous_digest=previous, principal=trusted_admin_principal("admin"), transaction_id="align-crash")
    registry.close()

    restarted = AlignmentRegistry(tmp_path / "policy.json")
    assert restarted.active().policy_digest == candidate["policy_digest"]
    lease = restarted.lease()
    assert lease.close() is True
    assert lease.close() is False
    with pytest.raises(RuntimeError):
        restarted.release(lease.policy_digest)
    restarted.close()


def test_domain_rollback_negative_control_detects_truncated_durable_state(tmp_path: Path) -> None:
    registry = PersistentDomainRegistry(tmp_path / "domains")
    registry.load_bundle(_domain_bundle(1, "paris"))
    prior = registry.domain_snapshot_id
    old_digest = registry._active.domains["geo"].digest
    registry.load_bundle(_domain_bundle(2, "lyon"), expected_previous_digest=old_digest)
    active_file = tmp_path / "domains" / "geo-0000000002.json"
    active_file.write_text(active_file.read_text()[:20], encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        PersistentDomainRegistry(tmp_path / "domains")
    assert prior != registry.domain_snapshot_id


class _MemoryAudit:
    owner_id = "audit:fault-suite"
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.terminals: set[str] = set()
    def readiness(self) -> bool:
        return True
    def append(self, event_type: str, data: dict[str, object]) -> None:
        tx = str(data["transaction_id"])
        if event_type == "memory.write_committed":
            if tx in self.terminals:
                raise RuntimeError("duplicate transaction terminal")
            self.terminals.add(tx)
        self.events.append((event_type, dict(data)))


class _TripAfterAudit:
    def hit(self, point: str) -> None:
        if point == "after_audit_append":
            raise RuntimeError("crash after audit append")


def test_memory_audit_ordering_restart_reconciles_outbox_idempotently(tmp_path: Path) -> None:
    audit = _MemoryAudit()
    db = tmp_path / "memory.sqlite"
    actor = MemoryActorContext("tenant", "subject", "actor", request_id="req")
    proposal = MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE, "profile", "response_style", "concise", "idem")
    repo = SQLiteMemoryRepository(str(db), durable_root=str(tmp_path), audit=audit, failpoint=_TripAfterAudit())
    with pytest.raises(RuntimeError, match="crash after audit append"):
        repo.commit(actor, proposal)
    repo.close()
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT delivered_at FROM memory_audit_outbox").fetchone()[0] is None
    repo = SQLiteMemoryRepository(str(db), durable_root=str(tmp_path), audit=audit)
    assert repo.read(MemoryReadRequest(actor, "profile", "response_style", 1))[0].value == "concise"
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT delivered_at FROM memory_audit_outbox").fetchone()[0] is not None
    assert len([e for e in audit.events if e[0] == "memory.write_committed"]) == 1
    assert repo.commit(actor, proposal).reason is MemoryReason.COMMITTED
    repo.close()


def test_audit_correlation_negative_control_rejects_terminal_without_prepare() -> None:
    class Event:
        def __init__(self, event_type: str, transaction_id: str) -> None:
            self.event_type = event_type
            self.data = {"transaction_id": transaction_id}

    with pytest.raises(ValueError, match="terminal without prepare"):
        reconcile_transactions((Event("memory.write_committed", "tx-missing-prepare"),))
