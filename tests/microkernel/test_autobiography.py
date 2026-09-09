from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from vulcan.constitution.primitives import Digest
from vulcan.microkernel.autobiography import (
    AutobiographicalEpisode,
    AutobiographicalMemoryStore,
    AutobiographyConflict,
    AutobiographyCrash,
    AutobiographyError,
    CausalFact,
    FactKind,
)
from vulcan.microkernel.principals import Principal, PrincipalKind


def d(label: str) -> str:
    return Digest.of_bytes(label.encode()).hex


class ChainAuthority:
    def __init__(self) -> None:
        self.rejected: set[str] = set()

    def verify(self, episode: AutobiographicalEpisode) -> None:
        if episode.episode_ref in self.rejected:
            raise AutobiographyError("durable chain mismatch")


def memory_store(path, **kwargs) -> AutobiographicalMemoryStore:
    return AutobiographicalMemoryStore(
        path, chain_authority=kwargs.pop("chain_authority", ChainAuthority()), **kwargs
    )


@pytest.fixture
def kernel() -> Principal:
    return Principal(PrincipalKind.SYSTEM_KERNEL, "kernel-1", d("release"))


def biography(
    memory_id: str = "memory-1",
    *,
    tenant: str = "tenant-a",
    person: str = "person-a",
    policy: str | None = None,
) -> AutobiographicalEpisode:
    episode = d(f"episode:{memory_id}")
    policy = policy or d("policy-a")
    observation = CausalFact(
        "observation-1", FactKind.OBSERVATION, d("observation"), episode
    )
    inference = CausalFact(
        "inference-1",
        FactKind.INFERENCE,
        d("interpretation"),
        episode,
        (observation.digest,),
        (policy,),
        (d("model-a"),),
    )
    prediction = CausalFact(
        "prediction-1",
        FactKind.PREDICTION,
        d("expectation"),
        episode,
        (inference.digest,),
        (policy,),
    )
    outcome = CausalFact(
        "outcome-1",
        FactKind.OUTCOME,
        d("reafference"),
        episode,
        (prediction.digest, observation.digest),
    )
    return AutobiographicalEpisode(
        memory_id=memory_id,
        tenant_id=tenant,
        person_id=person,
        purpose="policy-improvement",
        consent_ref=d("consent"),
        privacy_policy_ref=d("privacy-policy"),
        retention_until="2030-01-01T00:00:00Z",
        prior_lineage_state=d("lineage-before"),
        episode_ref=episode,
        interpretation_ref=d("interpretation"),
        committed_belief_refs=(d("belief"),),
        policy_ref=policy,
        expected_effect_ref=d("expectation"),
        intent_ref=d("intent"),
        receipt_ref=d("receipt"),
        observation_ref=d("observation"),
        reafference_ref=d("reafference"),
        correction_refs=(),
        next_lineage_state=d("lineage-after"),
        facts=(observation, inference, prediction, outcome),
    )


def test_db_first_outbox_restart_and_causal_retrieval(tmp_path, kernel):
    path = tmp_path / "autobiography.db"
    store = memory_store(path)
    record = biography()
    store.commit(kernel, record)

    restarted = memory_store(path)
    assert (
        restarted.load(
            record.memory_id, tenant_id="tenant-a", person_id="person-a"
        ).digest
        == record.digest
    )
    assert restarted.retrieve_causal(
        tenant_id="tenant-a",
        person_id="person-a",
        purpose="policy-improvement",
        policy_refs=(record.policy_ref,),
    ) == (record,)

    events: dict[str, object] = {}
    assert (
        restarted.drain_outbox(
            lambda event, payload: events.setdefault(
                payload["transaction_id"], (event, payload)
            )
        )
        == 1
    )
    assert (
        restarted.drain_outbox(lambda *_: pytest.fail("delivered outbox replayed")) == 0
    )


def test_correction_propagates_and_preserves_history(tmp_path, kernel):
    store = memory_store(tmp_path / "db")
    original = biography()
    store.commit(kernel, original)
    corrected_policy = d("policy-corrected")
    observation = original.facts[0]
    inference = replace(
        original.facts[1],
        policy_refs=(corrected_policy,),
        causal_parents=(observation.digest,),
    )
    prediction = replace(
        original.facts[2],
        policy_refs=(corrected_policy,),
        causal_parents=(inference.digest,),
    )
    outcome = replace(
        original.facts[3], causal_parents=(prediction.digest, observation.digest)
    )
    corrected_facts = (observation, inference, prediction, outcome)
    corrected = replace(
        original,
        revision=2,
        supersedes=original.digest,
        policy_ref=corrected_policy,
        correction_refs=(d("correction-evidence"),),
        facts=corrected_facts,
    )
    store.correct(kernel, original.digest, corrected)

    assert (
        store.load(
            original.memory_id, tenant_id="tenant-a", person_id="person-a"
        ).digest
        == corrected.digest
    )
    assert (
        store.load_digest(
            original.digest, tenant_id="tenant-a", person_id="person-a"
        ).digest
        == original.digest
    )
    assert (
        store.retrieve_causal(
            tenant_id="tenant-a",
            person_id="person-a",
            purpose="policy-improvement",
            policy_refs=(original.policy_ref,),
        )
        == ()
    )
    assert store.retrieve_causal(
        tenant_id="tenant-a",
        person_id="person-a",
        purpose="policy-improvement",
        policy_refs=(corrected_policy,),
    ) == (corrected,)


def test_irrelevant_similarity_and_poisoned_fact_are_rejected(tmp_path, kernel):
    store = memory_store(tmp_path / "db")
    record = biography()
    store.commit(kernel, record)
    # There is no prose or embedding selector: a text-like digest cannot retrieve.
    assert (
        store.retrieve_causal(
            tenant_id="tenant-a",
            person_id="person-a",
            purpose="policy-improvement",
            model_refs=(d("same words, unrelated cause"),),
        )
        == ()
    )
    with pytest.raises(ValueError, match="absent, cyclic, or out of order"):
        replace(
            record,
            memory_id="poison",
            facts=(
                replace(record.facts[0], causal_parents=(d("outside-chain"),)),
                *record.facts[1:],
            ),
        )


def test_tombstone_and_cross_person_isolation(tmp_path, kernel):
    store = memory_store(tmp_path / "db")
    record = biography()
    store.commit(kernel, record)
    with pytest.raises(AutobiographyError):
        store.load(record.memory_id, tenant_id="tenant-a", person_id="person-b")
    store.tombstone(
        kernel,
        tenant_id="tenant-a",
        person_id="person-a",
        memory_id=record.memory_id,
        deletion_authorization_ref=d("deletion-authorization"),
    )
    with pytest.raises(AutobiographyError):
        store.load(record.memory_id, tenant_id="tenant-a", person_id="person-a")
    assert (
        store.retrieve_causal(
            tenant_id="tenant-a",
            person_id="person-a",
            purpose="policy-improvement",
            policy_refs=(record.policy_ref,),
        )
        == ()
    )


def test_replayed_biography_and_non_kernel_mutation_fail_closed(tmp_path, kernel):
    store = memory_store(tmp_path / "db")
    record = biography()
    outsider = Principal(
        PrincipalKind.LANGUAGE_PROVIDER, "provider", d("provider-release")
    )
    with pytest.raises(AutobiographyError, match="kernel"):
        store.commit(outsider, record)
    store.commit(kernel, record)
    with pytest.raises(AutobiographyConflict):
        store.commit(kernel, record)
    forged = replace(
        record,
        revision=2,
        supersedes=d("invented history"),
        correction_refs=(d("correction-evidence"),),
    )
    with pytest.raises(AutobiographyConflict):
        store.correct(kernel, record.digest, forged)


def test_concurrent_corrections_have_one_winner(tmp_path, kernel):
    store = memory_store(tmp_path / "db")
    original = biography()
    store.commit(kernel, original)
    candidates = [
        replace(
            biography(policy=d(f"policy-{i}")),
            revision=2,
            supersedes=original.digest,
            correction_refs=(d(f"evidence-{i}"),),
        )
        for i in range(2)
    ]

    def apply(candidate):
        try:
            store.correct(kernel, original.digest, candidate)
            return True
        except AutobiographyConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(apply, candidates)) == 1


def test_restart_detects_document_tampering(tmp_path, kernel):
    path = tmp_path / "db"
    store = memory_store(path)
    store.commit(kernel, biography())
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE autobiography SET document=replace(document, 'policy-improvement', 'poisoned-purpose')"
        )
    with pytest.raises(AutobiographyError):
        memory_store(path)


def test_missing_or_changed_chain_authority_fails_closed(tmp_path, kernel):
    path = tmp_path / "db"
    authority = ChainAuthority()
    record = biography()
    memory_store(path, chain_authority=authority).commit(kernel, record)
    authority.rejected.add(record.episode_ref)
    with pytest.raises(AutobiographyError, match="chain mismatch"):
        memory_store(path, chain_authority=authority)
    with pytest.raises(AutobiographyError, match="authority"):
        AutobiographicalMemoryStore(path, chain_authority=None)  # type: ignore[arg-type]


def test_same_memory_id_isolated_across_tenants_and_digest_reads_are_scoped(
    tmp_path, kernel
):
    store = memory_store(tmp_path / "db")
    first = biography(tenant="tenant-a", person="person-a")
    second = biography(tenant="tenant-b", person="person-a")
    store.commit(kernel, first)
    store.commit(kernel, second)
    assert (
        store.load(first.memory_id, tenant_id="tenant-a", person_id="person-a") == first
    )
    assert (
        store.load(second.memory_id, tenant_id="tenant-b", person_id="person-a")
        == second
    )
    with pytest.raises(AutobiographyError):
        store.load_digest(first.digest, tenant_id="tenant-b", person_id="person-a")


def test_retention_is_enforced_at_read_and_retrieval(tmp_path, kernel):
    clock = lambda: datetime(2031, 1, 1, tzinfo=timezone.utc)
    store = memory_store(tmp_path / "db", clock=clock)
    record = biography()
    store.commit(kernel, record)
    with pytest.raises(AutobiographyError, match="expired"):
        store.load(record.memory_id, tenant_id="tenant-a", person_id="person-a")
    assert (
        store.retrieve_causal(
            tenant_id="tenant-a",
            person_id="person-a",
            purpose=record.purpose,
            policy_refs=(record.policy_ref,),
        )
        == ()
    )


class CrashAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary

    def hit(self, name: str) -> None:
        if name == self.boundary:
            raise AutobiographyCrash(name)


def test_cancellation_before_commit_rolls_back_and_after_commit_recovers(
    tmp_path, kernel
):
    before_path = tmp_path / "before.db"
    record = biography()
    store = memory_store(before_path, failpoint=CrashAt("before_commit"))
    with pytest.raises(AutobiographyCrash):
        store.commit(kernel, record)
    with pytest.raises(AutobiographyError):
        memory_store(before_path).load(
            record.memory_id, tenant_id="tenant-a", person_id="person-a"
        )

    after_path = tmp_path / "after.db"
    store = memory_store(after_path, failpoint=CrashAt("after_commit"))
    with pytest.raises(AutobiographyCrash):
        store.commit(kernel, record)
    assert (
        memory_store(after_path)
        .load(record.memory_id, tenant_id="tenant-a", person_id="person-a")
        .digest
        == record.digest
    )


def test_deletion_erases_memory_documents_but_preserves_tombstone_chain(
    tmp_path, kernel
):
    path = tmp_path / "db"
    store = memory_store(path)
    record = biography()
    store.commit(kernel, record)
    tombstone = store.tombstone(
        kernel,
        tenant_id=record.tenant_id,
        person_id=record.person_id,
        memory_id=record.memory_id,
        deletion_authorization_ref=d("delete-approved"),
    )
    with sqlite3.connect(path) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM autobiography WHERE memory_id=? AND document IS NOT NULL",
                (record.memory_id,),
            ).fetchone()[0]
            == 0
        )
        assert db.execute(
            "SELECT state,supersedes FROM autobiography WHERE digest=?", (tombstone,)
        ).fetchone() == ("tombstoned", record.digest)
    with pytest.raises(AutobiographyError):
        store.load_digest(
            record.digest, tenant_id=record.tenant_id, person_id=record.person_id
        )
    memory_store(path)  # restart verifies metadata-only deletion history


def test_outbox_crash_window_accepts_only_duplicate_transaction_proof(tmp_path, kernel):
    store = memory_store(tmp_path / "db")
    store.commit(kernel, biography())

    def already_appended(_event, _payload):
        raise RuntimeError("duplicate transaction")

    assert store.drain_outbox(already_appended) == 1
    assert store.drain_outbox(lambda *_: pytest.fail("outbox replayed")) == 0


def test_restart_rejects_outbox_tampering_and_legacy_unscoped_schema(tmp_path, kernel):
    path = tmp_path / "tampered.db"
    store = memory_store(path)
    store.commit(kernel, biography())
    with sqlite3.connect(path) as db:
        db.execute("UPDATE autobiography_outbox SET event_type='forged.event'")
    with pytest.raises(AutobiographyError, match="outbox"):
        memory_store(path)

    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as db:
        db.executescript("""
        CREATE TABLE autobiography(
          digest TEXT PRIMARY KEY, memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
          tenant_id TEXT NOT NULL, person_id TEXT NOT NULL, purpose TEXT NOT NULL,
          prior_lineage TEXT NOT NULL, next_lineage TEXT NOT NULL,
          state TEXT NOT NULL, supersedes TEXT, document TEXT NOT NULL,
          UNIQUE(memory_id, revision));
        CREATE UNIQUE INDEX autobiography_active ON autobiography(memory_id)
          WHERE state='active';
        """)
    with pytest.raises(AutobiographyError, match="governed migration"):
        memory_store(legacy)
