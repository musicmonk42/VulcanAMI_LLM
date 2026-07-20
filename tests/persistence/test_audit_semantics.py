import json
import time

import pytest

from vulcan.runtime.audit import AuditDurabilityProfile, AuditError, CanonicalAudit

D = "a" * 64
ACTOR = "b" * 64

def test_concurrent_same_operation_transactions_cannot_satisfy_lifecycle(tmp_path):
    audit = CanonicalAudit(tmp_path / "audit.jsonl")
    audit.append("domain.activation_prepared", {"transaction_id": "tx-a", "domain": "math", "actor_digest": ACTOR})
    audit.append("domain.activation_prepared", {"transaction_id": "tx-b", "domain": "math", "actor_digest": ACTOR})
    audit.append("domain.activation_committed", {"transaction_id": "tx-a", "domain": "math", "actor_digest": ACTOR})
    with pytest.raises(AuditError, match="terminal without prepare"):
        audit.append("domain.activation_committed", {"transaction_id": "tx-c", "domain": "math", "actor_digest": ACTOR})


def test_index_rebuild_equals_original_corrupt_index_rebuilt_source_fails_closed(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = CanonicalAudit(path)
    audit.append("case.started", {"case_id": "case-1", "request_digest": D})
    audit.append("case.failed", {"case_id": "case-1", "request_digest": D})
    audit.append("memory.write_prepared", {"transaction_id": "tx-1", "record_id": "mem-1", "actor_digest": ACTOR})
    audit.append("memory.write_committed", {"transaction_id": "tx-1", "record_id": "mem-1", "actor_digest": ACTOR})
    original = json.loads((path.with_suffix(".jsonl.d") / "index.json").read_text())
    audit.close()

    (path.with_suffix(".jsonl.d") / "index.json").write_text('{"schema_version":"bad"}\n')
    reopened = CanonicalAudit(path)
    rebuilt = json.loads((path.with_suffix(".jsonl.d") / "index.json").read_text())
    assert rebuilt == original
    assert [e.event_type for e in reopened.events_for_memory_record("mem-1")] == ["memory.write_prepared", "memory.write_committed"]
    reopened.close()

    segment = path.with_suffix(".jsonl.d") / "segment-000001.jsonl"
    segment.write_text(segment.read_text().replace('"sequence":1', '"sequence":9', 1))
    with pytest.raises(AuditError):
        CanonicalAudit(path)


def test_index_pagination_is_bounded_as_history_grows(tmp_path):
    audit = CanonicalAudit(tmp_path / "audit.jsonl", durability=AuditDurabilityProfile(fsync_events=False, fsync_manifest=False))
    for i in range(1500):
        audit.append("runtime.ready", {"actor_digest": ACTOR})
    started = time.perf_counter()
    page = audit.page_by_index("actor_digest", ACTOR, limit=100)
    elapsed = time.perf_counter() - started
    assert len(page.events) == 100
    assert page.next_cursor == "100"
    assert elapsed < 1.0
    with pytest.raises(AuditError, match="page limit bound"):
        audit.page_by_index("actor_digest", ACTOR, limit=1001)


def test_response_digest_field_names_are_exact(tmp_path):
    audit = CanonicalAudit(tmp_path / "audit.jsonl")
    audit.append("case.started", {"case_id": "case-1", "request_digest": D})
    audit.append("case.failed", {"case_id": "case-1", "request_digest": D, "response_ir_digest": D, "rendered_text_digest": ACTOR})
    with pytest.raises(AuditError, match="response_ir_digest"):
        audit.append("case.started", {"case_id": "case-2", "request_digest": D, "response_digest": D})


def test_deep_verify_requires_exactly_one_terminal_case_event(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = CanonicalAudit(path, segment_max_events=1)
    audit.append("case.started", {"case_id": "case-1", "request_digest": D})
    audit.append("runtime.ready", {})
    with pytest.raises(AuditError, match="terminal"):
        audit.deep_verify()
