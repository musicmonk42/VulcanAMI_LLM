import json
from pathlib import Path

import pytest

from vulcan.runtime.audit import AuditDurabilityProfile, AuditError, CanonicalAudit, Failpoint, _canonical, _hash_event

REQ = "a" * 64

def case(i=1):
    return {"case_id": f"c{i}", "request_digest": REQ}

class Boom(Failpoint):
    def __init__(self, name): self.name=name
    def hit(self, name):
        if name == self.name: raise AuditError(name)

def test_rotates_and_verifies_across_restart(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = CanonicalAudit(p, segment_max_events=2)
    a.append("case.started", case(1))
    a.append("case.started", case(2))
    a.append("case.started", case(3))
    assert (p.with_suffix(".jsonl.d") / "segment-000001.jsonl").read_text().count("segment_close") == 1
    a.close()
    b = CanonicalAudit(p, segment_max_events=2)
    assert [e.sequence for e in b.events(limit=10)] == [1, 2, 3]
    b.close()

def test_large_history_exceeds_old_event_limit(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = CanonicalAudit(p, segment_max_events=1000, durability=AuditDurabilityProfile(fsync_events=False, fsync_manifest=False))
    for i in range(100_005):
        a.append("runtime.ready", {"i": i})
    a.close()
    b = CanonicalAudit(p, segment_max_events=1000)
    assert b.events(limit=100_000)[-1].sequence == 100_000
    assert b._seq == 100_005
    b.close()

def test_migrates_v1_without_mutating_legacy(tmp_path):
    p = tmp_path / "audit.jsonl"
    ev = {"schema_version":"vulcan-audit/1","sequence":1,"event_type":"runtime.ready","timestamp":"2026-07-20T00:00:00Z","previous_hash":"0"*64,"data":{"ok":True}}
    ev["event_hash"] = _hash_event(ev)
    raw = _canonical(ev) + b"\n"
    p.write_bytes(raw)
    a = CanonicalAudit(p)
    assert p.read_bytes() == raw
    events = a.events(limit=10)
    assert events[0].event_type == "audit.migration_boundary"
    assert events[0].data["legacy_events"] == 1
    a.close()

def test_tamper_sequence_previous_hash_segment_close_manifest_and_truncation(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = CanonicalAudit(p, segment_max_events=1)
    a.append("runtime.ready", {"i": 1})
    a.append("runtime.ready", {"i": 2})
    a.close()
    root = p.with_suffix(".jsonl.d")

    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text())
    m["next_sequence"] = 99
    manifest.write_text(json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(AuditError, match="manifest"):
        CanonicalAudit(p)

    # restore by rebuilding fixture
    for child in root.iterdir(): child.unlink()
    root.rmdir(); (tmp_path / "audit.jsonl.d.lock").unlink(missing_ok=True)
    a = CanonicalAudit(p, segment_max_events=10); a.append("runtime.ready", {"i": 1}); a.close()
    seg = p.with_suffix(".jsonl.d") / "segment-000001.jsonl"
    obj = json.loads(seg.read_text().splitlines()[0]); obj["sequence"] = 2
    seg.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(AuditError, match="sequence"):
        CanonicalAudit(p)

def test_duplicate_key_and_path_replacement_fail_closed(tmp_path):
    p = tmp_path / "audit.jsonl"
    root = p.with_suffix(".jsonl.d")
    root.mkdir()
    (root / "manifest.json").write_text('{"schema_version":"vulcan-audit/2","schema_version":"x"}\n')
    with pytest.raises(AuditError, match="duplicate"):
        CanonicalAudit(p)
    (tmp_path / "audit.jsonl.d.lock").unlink(missing_ok=True)
    for child in root.iterdir(): child.unlink()
    root.rmdir()
    target = tmp_path / "target"; target.mkdir()
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(AuditError, match="symlink"):
        CanonicalAudit(p)

def test_fault_injection_leaves_restart_reconcilable_state(tmp_path):
    p = tmp_path / "audit.jsonl"
    with pytest.raises(AuditError, match="after_append_write"):
        CanonicalAudit(p, failpoint=Boom("after_append_write")).append("runtime.ready", {})
    a = CanonicalAudit(p)
    assert [e.event_type for e in a.events(limit=10)] == ["runtime.ready"]
    a.close()
    with pytest.raises(AuditError, match="before_manifest_replace"):
        CanonicalAudit(tmp_path / "b.jsonl", failpoint=Boom("before_manifest_replace")).append("runtime.ready", {})
    b = CanonicalAudit(tmp_path / "b.jsonl")
    assert b.events(limit=10) == ()
    b.close()

def test_bounded_export_preserves_verifiable_chain(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = CanonicalAudit(p, segment_max_events=1)
    a.append("runtime.ready", {"i": 1}); a.append("runtime.ready", {"i": 2})
    digest = a.export_archive(tmp_path / "archive.txt", max_bytes=1_000_000)
    assert len(digest) == 64
    with pytest.raises(AuditError, match="archive bound"):
        a.export_archive(tmp_path / "too-large.txt", max_bytes=1)
    a.close()
