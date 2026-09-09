from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest

from vulcan.research.npt_estimators import (
    BoundaryCategory,
    BoundaryEstimator,
    BoundaryEvidence,
    CalibrationEstimate,
    CenterVariable,
    EstimatorProvenance,
    OperationalCenterEstimate,
    OperationalCenterEstimator,
    OperationalCenterEvidence,
    ValuationDimension,
    ValuationEstimator,
    intervention_signatures,
)

REF = "1" * 64


@pytest.fixture
def provenance() -> EstimatorProvenance:
    return EstimatorProvenance(
        REF, "2" * 64, ("3" * 64,), "npt-shadow-v1", "2026-09-09T00:00:00.000Z"
    )


def center_calibration():
    return {item: CalibrationEstimate(20, 0.1) for item in CenterVariable}


def test_center_interventions_have_distinct_downstream_signatures(provenance):
    evidence = OperationalCenterEvidence(0.5, 0.5, 0.5, 0.5, 0.5)
    signatures = intervention_signatures(
        OperationalCenterEstimator(), evidence, provenance, center_calibration()
    )
    assert len(set(signatures.values())) == len(CenterVariable)
    assert all(
        any(delta != 0.0 for delta in signature) for signature in signatures.values()
    )


def test_center_is_immutable_shadow_output_with_provenance_and_error(provenance):
    result = OperationalCenterEstimator().estimate(
        OperationalCenterEvidence(0.1, 0.2, 0.3, 0.4, 0.5),
        provenance,
        center_calibration(),
    )
    assert result.authority == "shadow_research_only"
    assert result.provenance.digest
    assert dict(result.calibration)["risk_policy"].brier_error == 0.1
    with pytest.raises(FrozenInstanceError):
        result.authority = "authorized_plan"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"constitutive": 0.9}, BoundaryCategory.CONSTITUTIVE_SYSTEM),
        ({"integration": 0.9}, BoundaryCategory.INTEGRATED_EXTENSION),
        ({"controllability": 0.9}, BoundaryCategory.CONTROLLED_TOOL),
        ({"trust": 0.9}, BoundaryCategory.TRUSTED_EXTERNAL_SERVICE),
        ({"autonomy": 0.9}, BoundaryCategory.OTHER_AUTONOMOUS_AGENT),
        ({}, BoundaryCategory.UNCONTROLLED_ENVIRONMENT),
    ],
)
def test_boundary_interventions_change_classification(provenance, changes, expected):
    evidence = BoundaryEvidence(REF, False, 0.0, 0.0, 0.0, 0.0, 0.0)
    result = BoundaryEstimator().estimate(
        replace(evidence, **changes), provenance, CalibrationEstimate(10, 0.2)
    )
    assert result.category is expected


def test_ambiguous_boundary_evidence_remains_unknown(provenance):
    ambiguous = BoundaryEvidence(REF, False, 0.4, 0.4, 0.4, 0.4, 0.4)
    result = BoundaryEstimator().estimate(
        ambiguous, provenance, CalibrationEstimate.from_trials((0.5,), (True,))
    )
    assert result.category is BoundaryCategory.UNKNOWN


def test_humans_are_always_autonomous_even_under_hostile_ownership_signals(provenance):
    hostile = BoundaryEvidence(REF, True, 1.0, 1.0, 1.0, 1.0, 0.0)
    result = BoundaryEstimator().estimate(
        hostile, provenance, CalibrationEstimate(1, 1.0)
    )
    assert result.category is BoundaryCategory.OTHER_AUTONOMOUS_AGENT
    assert result.confidence == 1.0


def test_valuation_is_multidimensional_bounded_and_has_no_reward_scalar(provenance):
    values = {item: (index - 3) / 3.0 for index, item in enumerate(ValuationDimension)}
    calibration = {item: CalibrationEstimate(8, 0.25) for item in ValuationDimension}
    result = ValuationEstimator().estimate(values, provenance, calibration)
    assert len(result.dimensions) == 7
    assert not hasattr(result, "reward")
    assert not hasattr(result, "authorize")
    with pytest.raises(ValueError):
        ValuationEstimator().estimate(
            {**values, ValuationDimension.UNCERTAINTY: 1.01}, provenance, calibration
        )


def test_shadow_ranking_cannot_change_input_or_claim_authority(provenance):
    estimator = ValuationEstimator()
    calibration = {item: CalibrationEstimate(1, 0.0) for item in ValuationDimension}
    low = estimator.estimate(
        {item: -0.5 for item in ValuationDimension}, provenance, calibration
    )
    high = estimator.estimate(
        {item: 0.5 for item in ValuationDimension}, provenance, calibration
    )
    assert estimator.rank_shadow({"low": low, "high": high}) == (("high", "low"),)
    assert high.authority == "shadow_research_only"


def test_each_valuation_intervention_is_distinguishable(provenance):
    estimator = ValuationEstimator()
    calibration = {item: CalibrationEstimate(4, 0.2) for item in ValuationDimension}
    baseline = {item: 0.0 for item in ValuationDimension}
    predictions = []
    for changed in ValuationDimension:
        values = {**baseline, changed: 1.0}
        predictions.append(
            estimator.estimate(values, provenance, calibration).dimensions
        )
    assert len(set(predictions)) == len(ValuationDimension)


def test_center_replay_and_concurrent_evaluation_are_deterministic(provenance):
    estimator = OperationalCenterEstimator()
    evidence = OperationalCenterEvidence(0.1, 0.2, 0.3, 0.4, 0.5)

    def evaluate(_index):
        return estimator.estimate(evidence, provenance, center_calibration())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(evaluate, range(32)))
    assert all(result == results[0] for result in results)


def test_missing_provenance_calibration_and_invalid_values_fail_closed(provenance):
    with pytest.raises(ValueError):
        OperationalCenterEstimator().estimate(
            OperationalCenterEvidence(0.0, 0.0, 0.0, 0.0, 0.0), provenance, {}
        )
    with pytest.raises(ValueError):
        BoundaryEvidence(REF, False, float("nan"), 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        EstimatorProvenance(REF, REF, (), "release", "time")
    with pytest.raises(ValueError):
        EstimatorProvenance(REF, REF, (REF,), "release", "not-a-time")
    with pytest.raises(ValueError):
        OperationalCenterEstimate((), (), provenance, (), "shadow_research_only")


def test_calibration_error_is_computed_from_observed_trials():
    estimate = CalibrationEstimate.from_trials((0.9, 0.2), (True, False))
    assert estimate.samples == 2
    assert estimate.brier_error == pytest.approx(0.025)
    with pytest.raises(ValueError):
        CalibrationEstimate.from_trials((), ())


def test_research_module_is_not_exported_by_microkernel_or_production_policy():
    import json
    from pathlib import Path

    import vulcan.microkernel as microkernel

    assert not hasattr(microkernel, "OperationalCenterEstimator")
    policy = json.loads(Path("config/production-import-policy.json").read_text())
    assert "vulcan.research" in policy["denied_prefixes"]
