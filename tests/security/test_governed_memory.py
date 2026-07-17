"""Hard invariants for the small supported governed-memory surface."""
from datetime import datetime, timedelta, timezone

from vulcan.memory.governed import (
    MemoryActor, MemoryKind, MemoryReadRequest, MemoryReason,
    MemoryWriteProposal, SQLiteMemoryRepository,
)


def _proposal(value: str, key: str = "idem") -> MemoryWriteProposal:
    return MemoryWriteProposal(MemoryKind.EXPLICIT_PREFERENCE, "profile", "color", value, key)


def test_correction_is_immutable_and_stale_revision_conflicts(tmp_path):
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    actor = MemoryActor("tenant", "subject", "actor")
    first = repo.commit(actor, _proposal("blue")).record
    assert first is not None
    corrected = repo.correct(actor, first.record_id, first.revision, _proposal("red", "correct"))
    assert corrected.record is not None and corrected.record.revision == 2
    assert repo.correct(actor, first.record_id, 1, _proposal("green", "stale")).reason is MemoryReason.CONFLICT
    assert [record.value for record in repo.read(MemoryReadRequest(actor, "profile", "color"))] == ["red"]
    repo.close()


def test_tombstone_is_a_new_revision_and_denies_reads(tmp_path):
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    actor = MemoryActor("tenant", "subject", "actor")
    first = repo.commit(actor, _proposal("blue")).record
    assert first is not None
    deleted = repo.tombstone(actor, first.record_id, first.revision)
    assert deleted.record is not None and deleted.record.revision == 2
    assert repo.read(MemoryReadRequest(actor, "profile", "color")) == ()
    assert repo.tombstone(actor, first.record_id, first.revision).reason is MemoryReason.CONFLICT
    repo.close()


def test_scope_and_expiry_are_enforced(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = lambda: now
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"), clock=clock)
    owner = MemoryActor("tenant", "subject", "actor")
    other = MemoryActor("tenant", "other", "actor")
    assert repo.commit(owner, _proposal("blue")).reason is MemoryReason.COMMITTED
    assert repo.read(MemoryReadRequest(other, "profile", "color")) == ()
    repo._clock = lambda: now + timedelta(days=366)  # controlled clock boundary
    assert repo.read(MemoryReadRequest(owner, "profile", "color")) == ()
    repo.close()
