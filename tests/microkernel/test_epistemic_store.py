from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from vulcan.graphix.epistemic import (
    Claim,
    ClaimStatus,
    EpistemicCommit,
    EvidenceArtifact,
    EvidenceKind,
    Proposition,
)
from vulcan.microkernel.epistemic_store import (
    EpistemicConflict,
    EpistemicIntegrityError,
    EpistemicStore,
)

D = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
NOW = datetime(2026, 9, 8, tzinfo=timezone.utc)


def commit(
    episode: str,
    number: int,
    *,
    prior: str | None = None,
    claim: Claim | None = None,
    evidence: tuple[EvidenceArtifact, ...] = (),
) -> EpistemicCommit:
    item = claim or Claim(
        f"claim:{episode}:{number}",
        Proposition(f"prop:{episode}:{number}", "subject", "predicate", str(number)),
        ClaimStatus.HYPOTHESIS,
        episode,
        D,
    )
    return EpistemicCommit(
        commit_id=f"commit:{episode}:{number}",
        episode_id=episode,
        case_id=episode,
        snapshot_digest=D,
        authority_principal_id="principal:kernel",
        committed_at=NOW,
        claims=(item,),
        evidence=evidence,
        prior_commit_digest=prior,
        authority_release_digest=D,
        validation_digest=D,
        policy_digest=D,
        authority_evidence_digest=D,
    )


def test_concurrent_one_episode_has_one_cas_winner(tmp_path):
    store = EpistemicStore(tmp_path / "epistemic.sqlite3")
    candidates = [commit("episode:one", number) for number in range(8)]

    def append(item):
        try:
            return store.append(item, None).commit_digest
        except EpistemicConflict:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(append, candidates))
    assert len([item for item in results if item]) == 1
    assert store.head("episode:one") is not None


def test_concurrent_episodes_do_not_share_a_global_head(tmp_path):
    store = EpistemicStore(tmp_path / "epistemic.sqlite3")
    candidates = [commit(f"episode:{number}", 1) for number in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda item: store.append(item, None), candidates))
    assert len(results) == 8
    assert {item.episode_id for item in results} == {
        f"episode:{number}" for number in range(8)
    }


def test_crash_windows_restart_and_idempotent_retry(tmp_path):
    path = tmp_path / "epistemic.sqlite3"
    candidate = commit("episode:one", 1)

    def before(name):
        if name == "before_commit":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        EpistemicStore(path, failpoint=before).append(candidate, None)
    assert EpistemicStore(path).head("episode:one") is None

    def after(name):
        if name == "after_db_commit":
            raise RuntimeError("lost acknowledgement")

    with pytest.raises(RuntimeError, match="lost acknowledgement"):
        EpistemicStore(path, failpoint=after).append(candidate, None)
    restarted = EpistemicStore(path)
    assert restarted.append(candidate, None).commit_digest == candidate.commit_digest
    collision = replace(candidate, policy_digest=D2, commit_digest="")
    with pytest.raises(EpistemicConflict, match="collision"):
        restarted.append(collision, None)


def test_rebuilds_derived_indexes_but_rejects_semantic_corruption(tmp_path):
    path = tmp_path / "epistemic.sqlite3"
    candidate = commit("episode:one", 1)
    EpistemicStore(path).append(candidate, None)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM epistemic_claim_index")
    restarted = EpistemicStore(path)
    assert restarted.require_committed_claim(
        "episode:one", candidate.claims[0].claim_id
    )
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE epistemic_commits SET snapshot_digest=?", (D2,))
    with pytest.raises(EpistemicIntegrityError):
        EpistemicStore(path)


def test_cross_episode_reuse_preserves_snapshot_and_provenance(tmp_path):
    store = EpistemicStore(tmp_path / "epistemic.sqlite3")
    source_evidence = EvidenceArtifact(
        "evidence:source",
        EvidenceKind.OBSERVATION,
        "episode:source",
        D,
        D2,
        "provenance:source",
        NOW,
    )
    source_claim = Claim(
        "claim:source",
        Proposition("prop:source", "s", "p", "o"),
        ClaimStatus.OBSERVED,
        "episode:source",
        D,
        ("evidence:source",),
    )
    store.append(
        commit("episode:source", 1, claim=source_claim, evidence=(source_evidence,)),
        None,
    )
    reused = replace(
        source_evidence,
        episode_id="episode:target",
        source_episode_id="episode:source",
    )
    target_claim = Claim(
        "claim:target",
        Proposition("prop:target", "s", "p", "o"),
        ClaimStatus.OBSERVED,
        "episode:target",
        D,
        ("evidence:source",),
    )
    store.append(
        commit("episode:target", 1, claim=target_claim, evidence=(reused,)), None
    )
    with pytest.raises(EpistemicIntegrityError, match="preserve"):
        store.append(
            commit(
                "episode:bad",
                1,
                claim=replace(
                    target_claim, claim_id="claim:bad", episode_id="episode:bad"
                ),
                evidence=(replace(reused, episode_id="episode:bad", content_digest=D),),
            ),
            None,
        )


def test_contestation_and_supersession_are_immutable_append_only(tmp_path):
    store = EpistemicStore(tmp_path / "epistemic.sqlite3")
    first = commit("episode:one", 1)
    store.append(first, None)
    correction = Claim(
        "claim:correction",
        Proposition("prop:correction", "subject", "predicate", "corrected"),
        ClaimStatus.CONTESTED,
        "episode:one",
        D,
        contested_by=(first.claims[0].claim_id,),
        supersedes=(first.claims[0].claim_id,),
    )
    second = commit("episode:one", 2, prior=first.commit_digest, claim=correction)
    store.append(second, first.commit_digest)
    assert store.head("episode:one").claims[0].supersedes == (first.claims[0].claim_id,)
    with pytest.raises(EpistemicConflict, match="immutable"):
        store.append(
            commit(
                "episode:one",
                3,
                prior=second.commit_digest,
                claim=replace(first.claims[0], proposition=correction.proposition),
            ),
            second.commit_digest,
        )


def test_outbox_delivery_retries_without_duplicate_commit(tmp_path):
    path = tmp_path / "epistemic.sqlite3"
    calls: list[str] = []

    def flaky(event_id, payload):
        calls.append(event_id)
        if len(calls) == 1:
            raise RuntimeError("audit unavailable")

    store = EpistemicStore(path, outbox_sink=flaky)
    candidate = commit("episode:one", 1)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        store.append(candidate, None)
    assert store.append(candidate, None).commit_digest == candidate.commit_digest
    restarted = EpistemicStore(path, outbox_sink=flaky)
    assert calls == ["epistemic.commit:commit:episode:one:1"] * 2
    assert restarted.head("episode:one") is not None


def test_restart_rejects_tampered_or_missing_outbox(tmp_path):
    path = tmp_path / "epistemic.sqlite3"
    EpistemicStore(path).append(commit("episode:one", 1), None)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE epistemic_audit_outbox SET payload=?", ('{"tampered":true}',)
        )
    with pytest.raises(EpistemicIntegrityError, match="outbox"):
        EpistemicStore(path)
