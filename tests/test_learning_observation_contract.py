import dataclasses
from datetime import datetime, timezone, timedelta
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vulcan.learning_observation import (
    EligibilityStatus,
    ObservationContext,
    ProvenanceType,
    TerminalStatus,
    construct_observation,
    digest_json,
    evaluate_positive_eligibility,
    observation_from_canonical_json,
    validate_observation,
)
from vulcan.learning_owner import LearningOwner


def h(label: str) -> str:
    return digest_json({"label": label})


NOW = datetime(2026, 7, 19, 10, 0, 0, 123456, tzinfo=timezone.utc)


def ctx(**kw):
    base = dict(
        case_id="case-arith-1",
        case_digest=h("case"),
        request_digest=h("request"),
        tenant_digest=h("tenant"),
        alignment_revision=7,
        alignment_digest=h("alignment"),
        csiu_policy_digest=h("csiu-policy"),
        csiu_snapshot_digest=h("csiu-snapshot"),
        domain_snapshot_digest=h("domain"),
        runtime_owner_id="learning-owner-runtime",
        acquisition_time=NOW,
    )
    base.update(kw)
    return ObservationContext(**base)


def make_obs(**kw):
    args = dict(
        context=ctx(),
        selected_plan_digest=h("plan"),
        selected_tool_id="graphix_arithmetic",
        selection_distribution_digest=h("dist"),
        action_propensity=0.5,
        terminal_status=TerminalStatus.VALIDATED_SUCCESS,
        ledger_digest=h("ledger"),
        evidence_digest=h("derivation"),
        provenance_type=ProvenanceType.DERIVATION,
        terminal_case_validated=True,
        ledger_validated=True,
        evidence_integrity_validated=True,
        bindings_match=True,
        alignment_matches_lease=True,
        csiu_bindings_valid=True,
        clock=lambda: NOW,
    )
    args.update(kw)
    return construct_observation(**args)


def test_valid_arithmetic_derivation_observation_is_eligible():
    obs, result = make_obs()
    assert result.status is EligibilityStatus.ELIGIBLE_POSITIVE
    assert obs.schema_version == "vulcan-learning-observation/1"
    assert obs.selected_tool_id == "graphix_arithmetic"
    validate_observation(obs, clock=lambda: NOW)


def test_valid_retrieved_evidence_observation_is_eligible():
    obs, result = make_obs(
        selected_tool_id="graphix_retrieval",
        provenance_type=ProvenanceType.RETRIEVED_EVIDENCE,
        evidence_digest=h("retrieved-evidence"),
    )
    assert result.status is EligibilityStatus.ELIGIBLE_POSITIVE
    assert obs.provenance_type is ProvenanceType.RETRIEVED_EVIDENCE


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"ledger_validated": False}, "ledger validation failed"),
        ({"evidence_integrity_validated": False}, "evidence integrity failed"),
        ({"bindings_match": False}, "binding mismatch"),
        ({"alignment_matches_lease": False}, "alignment snapshot mismatch"),
        ({"csiu_bindings_valid": False}, "csiu bindings invalid"),
        ({"terminal_status": TerminalStatus.VALIDATED_FAILURE}, "terminal status is not eligible"),
    ],
)
def test_positive_eligibility_requires_authoritative_checks(override, reason):
    obs, result = make_obs(**override)
    assert result.status is EligibilityStatus.NOT_ACCEPTED
    assert reason in result.reason


def test_caller_success_or_high_confidence_cannot_authorize_without_evidence_or_ledger():
    _obs, no_ledger = make_obs(ledger_validated=False)
    _obs, no_evidence = make_obs(evidence_integrity_validated=False)
    assert no_ledger.status is EligibilityStatus.NOT_ACCEPTED
    assert no_evidence.status is EligibilityStatus.NOT_ACCEPTED


def test_cross_case_and_cross_tenant_substitution_fail():
    obs, _ = make_obs()
    assert evaluate_positive_eligibility(obs, terminal_case_validated=True, ledger_validated=True, evidence_integrity_validated=True, bindings_match=False, alignment_matches_lease=True, csiu_bindings_valid=True, clock=lambda: NOW).status is EligibilityStatus.NOT_ACCEPTED


def test_plan_tool_distribution_and_propensity_mutation_fail():
    obs, _ = make_obs()
    for field, value in [
        ("selected_plan_digest", h("other-plan")),
        ("selected_tool_id", "unknown_tool"),
        ("selection_distribution_digest", h("other-dist")),
        ("action_propensity", 0.25),
    ]:
        mutated = dataclasses.replace(obs, **{field: value})
        with pytest.raises(ValueError):
            validate_observation(mutated, clock=lambda: NOW)


@pytest.mark.parametrize("propensity", [0.0, -0.1, 1.1, math.nan, math.inf, True])
def test_invalid_propensity_values_fail(propensity):
    with pytest.raises(ValueError):
        make_obs(action_propensity=propensity)


def test_duplicate_observation_id_fails():
    obs, _ = make_obs()
    seen = set()
    validate_observation(obs, seen_observation_ids=seen, clock=lambda: NOW)
    with pytest.raises(ValueError, match="duplicate observation id"):
        validate_observation(obs, seen_observation_ids=seen, clock=lambda: NOW)


def test_duplicate_json_keys_unknown_fields_and_noncanonical_json_fail():
    obs, _ = make_obs()
    raw = obs.canonical_json()
    duplicate = raw[:-1] + ',"case_id":"case-other"}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        observation_from_canonical_json(duplicate, clock=lambda: NOW)
    data = json.loads(raw)
    data["unknown"] = "x"
    with pytest.raises(ValueError, match="unknown or missing"):
        observation_from_canonical_json(json.dumps(data, sort_keys=True, separators=(",", ":")), clock=lambda: NOW)
    pretty = json.dumps(json.loads(raw), sort_keys=True, indent=2)
    with pytest.raises(ValueError, match="noncanonical"):
        observation_from_canonical_json(pretty, clock=lambda: NOW)


def test_stale_and_future_timestamps_fail():
    with pytest.raises(ValueError, match="stale"):
        make_obs(context=ctx(acquisition_time=NOW - timedelta(minutes=16)))
    with pytest.raises(ValueError, match="future"):
        make_obs(context=ctx(acquisition_time=NOW + timedelta(seconds=10)))


def test_post_construction_mutation_fails_and_secret_content_rejected():
    obs, _ = make_obs()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.case_id = "case-other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="forbidden raw content"):
        make_obs(context=ctx(runtime_owner_id="Bearer abc.def"))


def test_learning_owner_accepts_only_typed_observation():
    current = datetime.now(timezone.utc).replace(microsecond=123456)
    obs, _ = make_obs(context=ctx(acquisition_time=current), clock=lambda: current)
    owner = LearningOwner(isolated_test_owner=True)
    assert owner.submit_observation(obs) == owner.owner_id
    with pytest.raises(ValueError, match="LearningObservation"):
        owner.submit_observation({"arbitrary": "dict"})


def test_canonical_serialization_and_digest_are_deterministic_across_processes(tmp_path):
    obs, _ = make_obs()
    raw = obs.canonical_json()
    script = """
import sys
from vulcan.learning_observation import observation_from_canonical_json
raw = sys.stdin.read()
obs = observation_from_canonical_json(raw, clock=lambda: __import__('datetime').datetime(2026,7,19,10,0,0,123456,tzinfo=__import__('datetime').timezone.utc))
print(obs.canonical_observation_digest)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run([sys.executable, "-c", script], input=raw, text=True, capture_output=True, cwd="/tmp", env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == obs.canonical_observation_digest
