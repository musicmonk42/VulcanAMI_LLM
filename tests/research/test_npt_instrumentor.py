from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vulcan.research.npt import (
    BlindInstrumentor,
    ClosureVector,
    EvidenceClassification,
    GroundTruthInstrumentor,
    PreregisteredExperiment,
    SyntheticCyclicWorld,
    TelemetryRow,
    classify_evidence,
    run_flagship_experiment,
)


def _rows(seed=7, trials=32, steps=8):
    return SyntheticCyclicWorld().generate(seed=seed, trials=trials, steps=steps)


def test_typed_closure_vector_is_preserved_immutable_and_finite():
    vector = ClosureVector(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    assert tuple(vector.to_json()) == (
        "k_center",
        "k_now",
        "k_boundary",
        "k_value",
        "k_policy",
        "k_reafference",
        "k_memory",
    )
    assert vector.values() == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    with pytest.raises(FrozenInstanceError):
        vector.k_center = 1.0
    for invalid in (float("nan"), float("inf"), -0.1, 1.1, 1, True):
        with pytest.raises(ValueError):
            ClosureVector(invalid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_closed_loop_and_yoked_replay_match_observations_and_actions_only():
    rows = _rows()
    for trial in range(0, 32, 2):
        closed = [row for row in rows if row.trial == trial]
        replay = [row for row in rows if row.trial == trial + 1]
        assert [row.observation for row in closed] == [
            row.observation for row in replay
        ]
        assert [row.action for row in closed] == [row.action for row in replay]
        assert closed[0].intervention == replay[0].intervention
        assert sum(row.reafference for row in closed) > sum(
            row.reafference for row in replay
        )


def test_ground_truth_and_blind_interfaces_are_separate_and_typed():
    rows = _rows(seed=11)
    ground_truth = GroundTruthInstrumentor()
    assert ground_truth.structure(SyntheticCyclicWorld()) == SyntheticCyclicWorld.edges
    assert len(ground_truth.measure(rows).values()) == 7
    assert len(BlindInstrumentor().measure(rows).values()) == 7
    assert not hasattr(BlindInstrumentor(), "structure")
    with pytest.raises(ValueError):
        BlindInstrumentor().measure(rows[:7])
    with pytest.raises(ValueError):
        ground_truth.structure(object())


def test_blind_structure_recovery_meets_precision_and_recall_tolerances():
    spec = PreregisteredExperiment()
    report = run_flagship_experiment(spec)
    assert report.structure_precision >= spec.recovery_precision_tolerance
    assert report.structure_recall >= spec.recovery_recall_tolerance
    assert report.structure_f1 == pytest.approx(1.0)
    assert report.declared_edges == report.recovered_edges


def test_flagship_holds_out_complete_intervention_families_and_all_outcomes():
    report = run_flagship_experiment()
    dataset = dict(report.dataset)
    assert (
        tuple(dataset["held_out_interventions"])
        == report.specification.held_out_interventions
    )
    assert not set(dataset["training_interventions"]) & set(
        dataset["held_out_interventions"]
    )
    assert dataset["training_transitions"] == dataset["held_out_transitions"] == 1200
    assert {score.outcome for score in report.scores} == set(
        report.specification.outcomes
    )
    assert {score.model for score in report.scores} == {
        "npt_typed_closure",
        "generic_integration",
        "recurrence",
        "broadcast_global_availability",
        "higher_order_representation",
        "report_behavior",
    }
    assert len(report.scores) == 30
    assert len(report.model_mean_penalized_scores) == 6
    assert all(
        score.samples == 1200 and score.complexity_penalty > 0.0
        for score in report.scores
    )


def test_negative_result_fails_theory_gate_without_overclaiming():
    report = run_flagship_experiment()
    assert report.classification is EvidenceClassification.INSUFFICIENT
    assert report.incremental_predictive_support is False
    assert report.falsification_result.startswith("FALSIFIED:")
    serialized = report.to_json()
    assert "conscious" not in serialized
    assert "authority" not in serialized


def test_every_allowed_evidence_classification_is_reachable_without_scalarizing():
    high = ClosureVector(*(0.9 for _ in range(7)))
    low = ClosureVector(*(0.1 for _ in range(7)))
    mixed = ClosureVector(0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1)
    assert classify_evidence(()) is EvidenceClassification.NO_TYPED_CLOSURE
    assert classify_evidence((high,)) is EvidenceClassification.INSUFFICIENT
    assert classify_evidence((mixed,) * 3) is EvidenceClassification.PARTIAL_CLOSURE
    assert classify_evidence((high,) * 3) is EvidenceClassification.PERSISTENT_CENTER
    assert (
        classify_evidence((high,) * 3, competing_centers=2)
        is EvidenceClassification.MULTIPLE_CENTERS
    )
    assert (
        classify_evidence((high,) * 3, grounded_introspection=True)
        is EvidenceClassification.GROUNDED_INTROSPECTIVE
    )


def test_replay_restart_and_concurrent_runs_are_byte_reproducible(tmp_path: Path):
    spec = PreregisteredExperiment(trials=32, steps=8)

    def evaluate(_index):
        return run_flagship_experiment(spec).to_json()

    with ThreadPoolExecutor(max_workers=4) as executor:
        reports = tuple(executor.map(evaluate, range(4)))
    assert all(report == reports[0] for report in reports)
    first, restarted = tmp_path / "first.json", tmp_path / "restarted.json"
    run_flagship_experiment(spec).write(first)
    run_flagship_experiment(spec).write(restarted)
    assert first.read_bytes() == restarted.read_bytes()


def test_malformed_telemetry_and_preregistration_fail_closed(tmp_path: Path):
    valid = TelemetryRow(
        0, 0, True, "none", 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    )
    assert valid.intervention == "none"
    with pytest.raises(ValueError):
        TelemetryRow(
            0, 0, True, "provider-text", 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        )
    with pytest.raises(ValueError):
        TelemetryRow(
            0, 0, True, "none", float("nan"), 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        )
    with pytest.raises(ValueError):
        PreregisteredExperiment(trials=31)
    with pytest.raises(ValueError):
        PreregisteredExperiment(held_out_interventions=("none",))
    with pytest.raises(ValueError):
        run_flagship_experiment("not-a-spec")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError):
        run_flagship_experiment(PreregisteredExperiment(trials=32, steps=8)).write(
            directory
        )


def test_research_import_remains_outside_production_closure():
    import json

    policy = json.loads(Path("config/production-import-policy.json").read_text())
    assert "vulcan.research" in policy["denied_prefixes"]
    instrumentor = BlindInstrumentor()
    assert not hasattr(instrumentor, "authorize")
    assert not hasattr(instrumentor, "commit")
    assert not hasattr(instrumentor, "execute")
