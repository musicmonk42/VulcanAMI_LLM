from __future__ import annotations

import sqlite3

import pytest

from vulcan.memory.governed import MemoryActorContext, MemoryKind, MemoryReadRequest, MemoryReason, MemoryWriteProposal, SQLiteMemoryRepository


class IdempotentAudit:
    owner_id = "audit:test"
    def __init__(self, fail_after_commit_once: bool = False):
        self.events = []
        self._terminal = set()
        self._prepared = set()
        self.fail_after_commit_once = fail_after_commit_once
    def readiness(self): return True
    def append(self, event_type, data):
        tx = data["transaction_id"]
        if event_type == "memory.write_prepared":
            if tx in self._prepared: raise RuntimeError("duplicate transaction prepare")
            self._prepared.add(tx)
        if event_type == "memory.write_committed":
            if tx in self._terminal: raise RuntimeError("duplicate transaction terminal")
            self._terminal.add(tx)
        self.events.append((event_type, dict(data)))
        if event_type == "memory.write_committed" and self.fail_after_commit_once:
            self.fail_after_commit_once = False
            raise RuntimeError("duplicate transaction terminal")


def actor(): return MemoryActorContext("tenant", "subject", "actor", request_id="req")
def proposal(key="idem", value="concise"): return MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE, "profile", "response_style", value, key)


def rows(path, table):
    con = sqlite3.connect(path)
    try: return con.execute(f"SELECT * FROM {table}").fetchall()
    finally: con.close()


def test_commit_writes_head_and_outbox_in_one_transaction_before_audit_delivery(tmp_path):
    audit = IdempotentAudit()
    repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=audit)
    res = repo.commit(actor(), proposal())
    assert res.reason is MemoryReason.COMMITTED
    assert len(rows(tmp_path / "m.sqlite", "memory_heads")) == 1
    outbox = rows(tmp_path / "m.sqlite", "memory_audit_outbox")
    assert outbox[0][-1] is not None
    assert [e[0] for e in audit.events] == ["memory.write_prepared", "memory.write_committed"]
    assert audit.events[1][1]["record_id"] == res.record.record_id
    repo.close()


def test_audit_unavailable_leaves_committed_outbox_for_restart_recovery(tmp_path):
    class Down(IdempotentAudit):
        def append(self, event_type, data): raise RuntimeError("audit down")
    repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=Down())
    with pytest.raises(RuntimeError, match="delivery failed"):
        repo.commit(actor(), proposal())
    repo.close()
    assert len(rows(tmp_path / "m.sqlite", "memory_heads")) == 1
    assert rows(tmp_path / "m.sqlite", "memory_audit_outbox")[0][-1] is None
    audit = IdempotentAudit()
    repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=audit)
    assert rows(tmp_path / "m.sqlite", "memory_audit_outbox")[0][-1] is not None
    assert len(audit.events) == 2
    repo.close()


def test_duplicate_idempotency_replays_exact_committed_record_without_duplicate_audit(tmp_path):
    audit = IdempotentAudit()
    repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=audit)
    first = repo.commit(actor(), proposal())
    second = repo.commit(actor(), proposal())
    assert second.reason is MemoryReason.COMMITTED
    assert second.record == first.record
    assert len([e for e in audit.events if e[0] == "memory.write_committed"]) == 1
    assert repo.commit(actor(), proposal("idem", "detailed")).reason is MemoryReason.IDEMPOTENCY_CONFLICT
    repo.close()


def test_stale_base_revision_is_terminal_conflict(tmp_path):
    audit = IdempotentAudit(); repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=audit)
    rec = repo.commit(actor(), proposal()).record
    assert repo.correct(actor(), rec.record_id, rec.revision, proposal("c1", "detailed")).reason is MemoryReason.COMMITTED
    assert repo.correct(actor(), rec.record_id, rec.revision, proposal("c2", "balanced")).reason is MemoryReason.CONFLICT
    assert repo.read(MemoryReadRequest(actor(), "profile", "response_style", 1))[0].value == "detailed"
    repo.close()

class TripOnce:
    def __init__(self, name):
        self.name = name
        self.tripped = False
    def hit(self, name):
        if name == self.name and not self.tripped:
            self.tripped = True
            raise RuntimeError(name)


def test_restart_after_audit_append_before_delivery_mark_is_idempotent(tmp_path):
    audit = IdempotentAudit()
    repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=audit, failpoint=TripOnce("after_audit_append"))
    with pytest.raises(RuntimeError, match="after_audit_append"):
        repo.commit(actor(), proposal())
    repo.close()
    assert rows(tmp_path / "m.sqlite", "memory_audit_outbox")[0][-1] is None
    repo = SQLiteMemoryRepository(str(tmp_path / "m.sqlite"), durable_root=str(tmp_path), audit=audit)
    assert rows(tmp_path / "m.sqlite", "memory_audit_outbox")[0][-1] is not None
    assert len([e for e in audit.events if e[0] == "memory.write_committed"]) == 1
    repo.close()
