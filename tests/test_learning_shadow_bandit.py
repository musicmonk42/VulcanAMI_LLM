from datetime import datetime, timezone
import dataclasses
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from vulcan.learning_bandit import (
    ACTIONS,
    BanditUpdateStatus,
    ShadowLinUCBToolBandit,
)
from vulcan.learning_observation import (
    ObservationContext,
    ProvenanceType,
    TerminalStatus,
    construct_observation,
    digest_json,
)
from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner


def h(value):
    return digest_json({"x": value})


def make_observation(bandit, label, tool="graphix_arithmetic", status=TerminalStatus.VALIDATED_SUCCESS, *, propensity=None, dist_digest=None):
    now = datetime.now(timezone.utc).replace(microsecond=123456)
    actions = ACTIONS
    prop = propensity if propensity is not None else round(1.0 / len(actions), 12)
    dist = {a: round(1.0 / len(actions), 12) for a in actions}
    ctx = ObservationContext(
        case_id=f"case-{label}",
        case_digest=h("case" + str(label)),
        request_digest=h("request" + str(label)),
        tenant_digest=h("tenant"),
        alignment_revision=1,
        alignment_digest=h("alignment"),
        csiu_policy_digest=h("csiu-policy"),
        csiu_snapshot_digest=h("csiu-snapshot"),
        domain_snapshot_digest=h("domain"),
        runtime_owner_id="learning-owner",
        acquisition_time=now,
    )
    observation, eligibility = construct_observation(
        context=ctx,
        selected_plan_digest=h("plan" + str(label)),
        selected_tool_id=tool,
        selection_distribution_digest=dist_digest or bandit.distribution_digest(dist),
        action_propensity=prop,
        terminal_status=status,
        ledger_digest=h("ledger" + str(label)),
        evidence_digest=h("evidence" + str(label)),
        provenance_type=ProvenanceType.DERIVATION,
        terminal_case_validated=True,
        ledger_validated=True,
        evidence_integrity_validated=True,
        bindings_match=True,
        alignment_matches_lease=True,
        csiu_bindings_valid=True,
        clock=lambda: now,
    )
    assert eligibility.status.value == "eligible_positive" or status is not TerminalStatus.VALIDATED_SUCCESS
    return observation


def log_and_update(bandit, observation):
    record = bandit.select_shadow(observation)
    assert record.active_choice == ACTIONS[0]
    assert record.active_distribution[observation.selected_tool_id] == observation.action_propensity
    return bandit.update_from_observation(observation)


def test_shadow_candidate_learns_better_tool_without_changing_active_choice():
    bandit = ShadowLinUCBToolBandit(alpha=0.1)
    for i in range(30):
        good = make_observation(bandit, f"good-{i}", "graphix_retrieval", TerminalStatus.VALIDATED_SUCCESS)
        bad = make_observation(bandit, f"bad-{i}", "graphix_arithmetic", TerminalStatus.VALIDATED_FAILURE)
        assert log_and_update(bandit, good).status is BanditUpdateStatus.APPLIED
        assert log_and_update(bandit, bad).reward == -1.0
    probe = make_observation(bandit, "probe", "graphix_arithmetic")
    record = bandit.select_shadow(probe)
    assert record.active_choice == ACTIONS[0]
    assert record.candidate_choice == "graphix_retrieval"
    assert math.isclose(sum(record.candidate_distribution.values()), 1.0, abs_tol=1e-9)
    assert all(math.isfinite(v) and v > 0.0 for v in record.candidate_distribution.values())
    assert bandit.evaluation_metrics()["ready"] is True


def test_unverified_or_neutral_status_produces_no_positive_update():
    bandit = ShadowLinUCBToolBandit()
    unsupported = make_observation(bandit, "unsupported", "graphix_arithmetic", TerminalStatus.SAFETY_FILTERED)
    bandit.select_shadow(unsupported)
    result = bandit.update_from_observation(unsupported)
    assert result.status is BanditUpdateStatus.NOT_ACCEPTED
    assert result.reward == 0.0
    assert bandit.candidate_policy_revision == 0


def test_verified_incorrect_output_is_negative_not_positive():
    bandit = ShadowLinUCBToolBandit()
    observation = make_observation(bandit, "incorrect", "graphix_symbolic", TerminalStatus.VALIDATED_FAILURE)
    result = log_and_update(bandit, observation)
    assert result.status is BanditUpdateStatus.APPLIED
    assert result.reward == -1.0


def test_duplicate_observation_has_one_effect_and_conflict_is_rejected():
    bandit = ShadowLinUCBToolBandit()
    observation = make_observation(bandit, "dup")
    bandit.select_shadow(observation)
    assert bandit.update_from_observation(observation).status is BanditUpdateStatus.APPLIED
    rev = bandit.candidate_policy_revision
    assert bandit.update_from_observation(observation).status is BanditUpdateStatus.REPLAYED
    assert bandit.candidate_policy_revision == rev
    changed = dataclasses.replace(make_observation(bandit, "dup2"), observation_id=observation.observation_id)
    changed = dataclasses.replace(changed, canonical_observation_digest=digest_json(changed.canonical_payload(include_digest=False)))
    bandit.select_shadow(changed)
    assert bandit.update_from_observation(changed).status is BanditUpdateStatus.CONFLICT


@pytest.mark.parametrize("propensity", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_propensity_fails(propensity):
    bandit = ShadowLinUCBToolBandit()
    with pytest.raises(ValueError):
        make_observation(bandit, f"bad-prop-{propensity}", propensity=propensity)


def test_policy_and_distribution_mismatch_fail():
    bandit = ShadowLinUCBToolBandit()
    observation = make_observation(bandit, "mismatch")
    bandit.select_shadow(observation)
    bandit._active_policy_digest = h("changed-policy")
    with pytest.raises(ValueError, match="policy digest mismatch"):
        bandit.update_from_observation(observation)

    bandit = ShadowLinUCBToolBandit()
    observation = make_observation(bandit, "dist-mismatch", dist_digest=h("wrong-dist"))
    bandit.select_shadow(observation)
    with pytest.raises(ValueError, match="distribution digest mismatch"):
        bandit.update_from_observation(observation)


def test_importance_correction_is_bounded():
    bandit = ShadowLinUCBToolBandit(importance_clip=2.0)
    dist = {"graphix_arithmetic": 0.01, "graphix_retrieval": 0.495, "graphix_symbolic": 0.495}
    observation = make_observation(bandit, "clip", propensity=0.01, dist_digest=bandit.distribution_digest(dist))
    bandit.select_shadow(observation, active_distribution=dist)
    result = bandit.update_from_observation(observation)
    assert result.importance_weight == 2.0
    assert result.clipping_reason == "clipped"


def test_unknown_tool_fails():
    bandit = ShadowLinUCBToolBandit()
    observation = make_observation(bandit, "unknown")
    with pytest.raises(ValueError):
        bandit.select_shadow(observation, candidate_set=["graphix_arithmetic", "removed_tool"])


def test_same_seed_sequence_save_reopen_identical_state_and_next_decision(tmp_path):
    fixed_clock = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    b1 = ShadowLinUCBToolBandit(alpha=0.2, clock=fixed_clock)
    b2 = ShadowLinUCBToolBandit(alpha=0.2, clock=fixed_clock)
    for i in range(10):
        o1 = make_observation(b1, f"seq-{i}", "graphix_retrieval")
        o2 = dataclasses.replace(o1)
        b1.select_shadow(o1); b2.select_shadow(o2)
        b1.update_from_observation(o1); b2.update_from_observation(o2)
    assert b1.state_bytes() == b2.state_bytes()
    path = tmp_path / "bandit.json"
    b1.save(path)
    reopened = ShadowLinUCBToolBandit.load(path, clock=fixed_clock)
    assert reopened.state_bytes() == b1.state_bytes()
    probe = make_observation(b1, "next")
    assert reopened.select_shadow(probe).candidate_distribution == b1.select_shadow(probe).candidate_distribution


def test_concurrent_observations_preserve_revision_accounting():
    bandit = ShadowLinUCBToolBandit()
    observations = [make_observation(bandit, f"concurrent-{i}", "graphix_retrieval") for i in range(20)]
    for observation in observations:
        bandit.select_shadow(observation)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(bandit.update_from_observation, observations))
    assert sum(1 for r in results if r.status is BanditUpdateStatus.APPLIED) == 20
    assert bandit.candidate_policy_revision == 20


def test_legacy_private_weight_mutation_cannot_alter_policy_and_owner_is_shadow():
    owner = LearningOwner(shadow_bandit=ShadowLinUCBToolBandit(), capability=LearningCapabilityStatus.SHADOW, isolated_test_owner=True)
    assert owner.capability is LearningCapabilityStatus.SHADOW
    observation = make_observation(owner._shadow_bandit, "owner")
    owner.record_shadow_tool_selection(observation)
    owner._shadow_bandit.tool_weight_adjustments = {"graphix_retrieval": 999.0}
    before = owner._shadow_bandit.candidate_policy_digest
    result = owner.apply_committed_observation(observation)
    assert result.status is BanditUpdateStatus.APPLIED
    assert owner._shadow_bandit.tool_weight_adjustments["graphix_retrieval"] == 999.0
    assert owner._shadow_bandit.candidate_policy_digest != before


def test_outbox_failure_leaves_candidate_head_unchanged(tmp_path):
    from vulcan.learning_outbox import LearningObservationOutbox, LearningOutboxError
    from vulcan.runtime.audit import CanonicalAudit

    bandit = ShadowLinUCBToolBandit()
    observation = make_observation(bandit, "outbox-failure")
    bandit.select_shadow(observation)
    outbox = LearningObservationOutbox(tmp_path / "learning.db", audit=CanonicalAudit(tmp_path / "audit.jsonl"), failpoint="after_prepared_audit")
    with pytest.raises(LearningOutboxError):
        outbox.deliver(observation, expected_revision=0)
    assert bandit.candidate_policy_revision == 0
    assert outbox.conn.execute("SELECT revision FROM active_head").fetchone()[0] == 0
