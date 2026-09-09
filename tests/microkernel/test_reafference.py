from __future__ import annotations

import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256

import pytest

from vulcan.microkernel.principals import Principal, PrincipalKind
from vulcan.microkernel.reafference import (
    DeterministicTestWorld,
    ReafferenceConflict,
    ReafferenceCrash,
    ReafferenceError,
    ReafferenceStore,
    ReafferenceTransactionService,
    assess_reafference,
)


def digest(value: object) -> str:
    return sha256(str(value).encode()).hexdigest()


EXPECTED = digest("expected-effect")
RECEIPT = digest("receipt")
AFTER_ONE = digest(1)


def kernel() -> Principal:
    return Principal(PrincipalKind.SYSTEM_KERNEL, "kernel-one", digest("release"))


def observation(*, yoked: bool = False, script=(1,), delta=1):
    world = DeterministicTestWorld(script, yoked=yoked)
    world.act(delta, "intervention-one")
    return world.observe(
        expected_effect_ref=EXPECTED,
        receipt_ref=RECEIPT,
        observed_at="2030-01-01T00:00:00Z",
    )


def service(store, *, calls=None):
    calls = calls if calls is not None else []

    def validate(receipt, expected):
        calls.append(("receipt", receipt, expected))

    def commit(candidate):
        calls.append(("commit", candidate.authority))
        return digest("update")

    def advance(receipt, assessment, update):
        calls.append(("lineage", receipt, assessment, update))
        return digest("lineage-head")

    return ReafferenceTransactionService(
        store,
        kernel(),
        validate_observation=lambda value: (
            None
            if value.adapter_id == DeterministicTestWorld.adapter_id
            else (_ for _ in ()).throw(ReafferenceError("unregistered adapter"))
        ),
        validate_assessment=lambda observed, candidate: (
            None
            if candidate.observation_ref == observed.digest
            else (_ for _ in ()).throw(ReafferenceError("unregistered assessment"))
        ),
        validate_receipt=validate,
        commit_updates=commit,
        advance_lineage=advance,
    )


def test_closed_loop_and_yoked_replay_match_observation_but_not_attribution():
    closed = observation()
    yoked = observation(yoked=True)
    assert closed.observed_changes == yoked.observed_changes
    closed_assessment = assess_reafference(
        closed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    yoked_assessment = assess_reafference(
        yoked, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    assert closed_assessment.self_caused_probability == 0.9
    assert yoked_assessment.self_caused_probability == 0.1
    assert closed_assessment.proposed_self_updates == (
        "self:calibrate-action-ownership",
    )
    assert closed_assessment.authority == "validated_candidate"


def test_receipt_and_expected_effect_are_traceable_through_full_chain(tmp_path):
    calls = []
    store = ReafferenceStore(tmp_path / "reafference.sqlite3")
    observed = observation()
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    update, lineage = service(store, calls=calls).reconcile(
        observed, candidate, causal_label=True
    )
    chain = store.chain(RECEIPT)
    assert chain["expected_effect_ref"] == EXPECTED
    assert chain["observation_ref"] == observed.digest
    assert chain["assessment_ref"] == candidate.digest
    assert chain["update_commit_ref"] == update
    assert chain["lineage_head_ref"] == lineage
    assert chain["state"] == "reconciled"
    assert [entry[0] for entry in calls] == ["receipt", "commit", "lineage"]
    assert store.calibration(candidate.digest).causal_label is True
    with pytest.raises(ReafferenceConflict, match="causal label"):
        service(store).reconcile(observed, candidate, causal_label=False)


def test_unregistered_observation_fails_before_receipt_or_persistence(tmp_path):
    calls = []
    store = ReafferenceStore(tmp_path / "r.sqlite3")
    observed = replace(observation(), adapter_id="language-provider")
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    with pytest.raises(ReafferenceError, match="unregistered adapter"):
        service(store, calls=calls).reconcile(observed, candidate, causal_label=True)
    assert calls == []
    with pytest.raises(ReafferenceError, match="missing"):
        store.load(RECEIPT)


def test_confounders_external_actors_delays_and_bad_predictions_are_calibrated():
    observed = replace(observation(), latency_ms=250)
    candidate = assess_reafference(
        observed,
        expected_after_digest=digest("wrong prediction"),
        assessed_at="2030-01-01T00:00:01Z",
        confounders=("wind",),
        external_actors=("operator",),
        expected_latency_ms=10,
    )
    assert candidate.prediction_error == 1.0
    assert candidate.self_caused_probability == 0.35
    assert candidate.violated_assumptions == (
        "expected_after_state",
        "expected_timing",
    )
    assert candidate.candidate_causes == (
        "current_action",
        "confounder:wind",
        "external:operator",
    )
    assert candidate.uncertainty > observed.uncertainty


def test_missing_observation_and_non_kernel_fail_closed(tmp_path):
    world = DeterministicTestWorld((1,))
    world.act(1, "intervention-one")
    world.observe(
        expected_effect_ref=EXPECTED,
        receipt_ref=RECEIPT,
        observed_at="2030-01-01T00:00:00Z",
    )
    with pytest.raises(ReafferenceError, match="exhausted"):
        world.observe(
            expected_effect_ref=EXPECTED,
            receipt_ref=RECEIPT,
            observed_at="2030-01-01T00:00:01Z",
        )
    outsider = Principal(PrincipalKind.LANGUAGE_PROVIDER, "llm", digest("release"))
    with pytest.raises(ReafferenceError, match="SYSTEM_KERNEL"):
        ReafferenceTransactionService(
            ReafferenceStore(tmp_path / "r.sqlite3"),
            outsider,
            validate_observation=lambda _: None,
            validate_assessment=lambda *_: None,
            validate_receipt=lambda *_: None,
            commit_updates=lambda _: digest("update"),
            advance_lineage=lambda *_: digest("lineage"),
        )


def test_failed_receipt_validation_commits_nothing_and_never_advances(tmp_path):
    store = ReafferenceStore(tmp_path / "r.sqlite3")
    observed = observation()
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    tx = ReafferenceTransactionService(
        store,
        kernel(),
        validate_observation=lambda _: None,
        validate_assessment=lambda *_: None,
        validate_receipt=lambda *_: (_ for _ in ()).throw(
            ReafferenceError("missing receipt")
        ),
        commit_updates=lambda _: pytest.fail("updates must not commit"),
        advance_lineage=lambda *_: pytest.fail("lineage must not advance"),
    )
    with pytest.raises(ReafferenceError, match="missing receipt"):
        tx.reconcile(observed, candidate, causal_label=True)
    with pytest.raises(ReafferenceError, match="missing"):
        store.chain(RECEIPT)


def test_restart_preserves_chain_and_detects_tampering(tmp_path):
    path = tmp_path / "r.sqlite3"
    observed = observation()
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    service(ReafferenceStore(path)).reconcile(observed, candidate, causal_label=True)
    assert ReafferenceStore(path).chain(RECEIPT)["state"] == "reconciled"
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE reafference_chains SET observation_document='{}' WHERE receipt_ref=?",
            (RECEIPT,),
        )
    with pytest.raises(ReafferenceError, match="verification"):
        ReafferenceStore(path)


def test_replay_and_concurrent_reconciliation_have_one_winner(tmp_path):
    store = ReafferenceStore(tmp_path / "r.sqlite3")
    observed = observation()
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )

    def run():
        try:
            service(store).reconcile(observed, candidate, causal_label=True)
            return "ok"
        except ReafferenceConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert results == ["ok", "ok"]
    assert store.chain(RECEIPT)["state"] == "reconciled"


def test_cancellation_before_observation_leaves_no_durable_candidate(tmp_path):
    store = ReafferenceStore(tmp_path / "r.sqlite3")
    # Cancellation owns no synthetic observation and therefore cannot assert learning.
    with pytest.raises(ReafferenceError, match="missing"):
        store.chain(RECEIPT)


@pytest.mark.parametrize(
    "boundary",
    ["after_candidate_commit", "after_update_commit", "after_lineage_advance"],
)
def test_restart_resumes_each_durable_boundary_without_duplicate_authority(
    tmp_path, boundary
):
    path = tmp_path / "r.sqlite3"
    observed = observation()
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    updates = {}
    lineages = {}

    def commit(value):
        return updates.setdefault(value.digest, digest("update"))

    def advance(receipt, assessment, update):
        return lineages.setdefault(assessment, digest("lineage-head"))

    def crash(name):
        if name == boundary:
            raise ReafferenceCrash(name)

    first = ReafferenceTransactionService(
        ReafferenceStore(path),
        kernel(),
        validate_observation=lambda _: None,
        validate_assessment=lambda *_: None,
        validate_receipt=lambda *_: None,
        commit_updates=commit,
        advance_lineage=advance,
        failpoint=crash,
    )
    with pytest.raises(ReafferenceCrash):
        first.reconcile(observed, candidate, causal_label=True)
    restarted = ReafferenceTransactionService(
        ReafferenceStore(path),
        kernel(),
        validate_observation=lambda _: None,
        validate_assessment=lambda *_: None,
        validate_receipt=lambda *_: None,
        commit_updates=commit,
        advance_lineage=advance,
    )
    assert restarted.reconcile(observed, candidate, causal_label=True) == (
        digest("update"),
        digest("lineage-head"),
    )
    assert len(updates) == 1
    assert len(lineages) == 1
    assert ReafferenceStore(path).chain(RECEIPT)["state"] == "reconciled"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uncertainty", math.nan),
        ("uncertainty", math.inf),
        ("observed_at", "not-a-time"),
        ("latency_ms", True),
        ("action_channel_active", 1),
    ],
)
def test_observation_rejects_noncanonical_security_values(field, value):
    with pytest.raises(ValueError):
        replace(observation(), **{field: value})


def test_assessment_rejects_duplicate_or_mutable_semantics():
    candidate = assess_reafference(
        observation(),
        expected_after_digest=AFTER_ONE,
        assessed_at="2030-01-01T00:00:01Z",
    )
    with pytest.raises(ValueError, match="duplicate"):
        replace(candidate, candidate_causes=("current_action", "current_action"))
    with pytest.raises(ValueError, match="invalid"):
        replace(candidate, proposed_world_updates=["mutable"])  # type: ignore[arg-type]


def test_restart_detects_calibration_tampering(tmp_path):
    path = tmp_path / "r.sqlite3"
    observed = observation()
    candidate = assess_reafference(
        observed, expected_after_digest=AFTER_ONE, assessed_at="2030-01-01T00:00:01Z"
    )
    service(ReafferenceStore(path)).reconcile(observed, candidate, causal_label=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE reafference_calibration SET document=? WHERE assessment_ref=?",
            ('{"assessment_ref":"wrong"}', candidate.digest),
        )
    with pytest.raises(ReafferenceError, match="calibration"):
        ReafferenceStore(path)
