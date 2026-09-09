from __future__ import annotations

import pytest

from vulcan.runtime.audit import AuditError, CanonicalAudit


def payload() -> dict[str, object]:
    return {
        "event_id": "epistemic.commit:commit:one",
        "commit_id": "commit:one",
        "commit_digest": "1" * 64,
        "episode_id": "episode:one",
        "case_id": "episode:one",
        "snapshot_digest": "2" * 64,
        "policy_digest": "3" * 64,
    }


def test_epistemic_outbox_projection_is_idempotent_across_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = CanonicalAudit(path)
    first = audit.append_epistemic_commit(str(payload()["event_id"]), payload())
    assert audit.append_epistemic_commit(str(payload()["event_id"]), payload()) == first
    audit.close()

    restarted = CanonicalAudit(path)
    replayed = restarted.append_epistemic_commit(str(payload()["event_id"]), payload())
    assert replayed.sequence == first.sequence
    assert (
        len(
            [
                event
                for event in restarted.events()
                if event.event_type == "epistemic.committed"
            ]
        )
        == 1
    )
    changed = payload()
    changed["policy_digest"] = "4" * 64
    with pytest.raises(AuditError, match="different data"):
        restarted.append_epistemic_commit(str(changed["event_id"]), changed)
    restarted.close()
