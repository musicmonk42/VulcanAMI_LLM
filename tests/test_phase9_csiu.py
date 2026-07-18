from datetime import datetime, timezone, timedelta
import threading

import pytest

import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("csiu_enforcement", pathlib.Path(__file__).resolve().parents[1] / "src/vulcan/world_model/meta_reasoning/csiu_enforcement.py")
c = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["csiu_enforcement"] = c
spec.loader.exec_module(c)


def snap(vals, pol, end=None, samples=30):
    end = end or datetime.now(timezone.utc)
    return c.CSIUMetricSnapshot(
        metrics=vals,
        window_start=(end - timedelta(minutes=5)).isoformat().replace('+00:00','Z'),
        window_end=end.isoformat().replace('+00:00','Z'),
        sample_count=samples,
        aggregation_method='mean',
        metric_definition_version=pol.metric_definition_version,
        provider_id='aggregate-provider',
        provenance_digest='a'*64,
        policy_digest=pol.policy_digest,
    )

BASE = dict(A=.5,H=.5,C=.5,V=.5,D=.5,G=.5,E=.5,U=.5,M=.5)
IMP = dict(A=.6,H=.4,C=.6,V=.4,D=.4,G=.4,E=.6,U=.6,M=.4)
BAD = dict(A=.4,H=.6,C=.4,V=.6,D=.6,G=.6,E=.4,U=.4,M=.6)


def test_utility_signs_and_baseline():
    e = c.CSIUEnforcement(c.CSIUEnforcementConfig())
    s0 = snap(BASE, e.policy); s1 = snap(BASE, e.policy); s2 = snap(IMP, e.policy); s3 = snap(BAD, e.policy)
    assert e.compute_utility(s0, s1) == 0.0
    assert e.compute_utility(s0, s2) > 0
    assert e.compute_utility(s0, s3) < 0
    assert abs(e.pressure_from_utility(-10**9)) <= e.config.max_single_influence


def test_invalid_snapshot_and_policy_rejected():
    e = c.CSIUEnforcement()
    with pytest.raises(c.CSIUValidationError):
        snap({**BASE, 'A': float('nan')}, e.policy)
    old = snap(BASE, e.policy, end=datetime.now(timezone.utc)-timedelta(days=1))
    ok, reason = e.validate_snapshot(old)
    assert not ok and reason == 'stale_snapshot'
    low = snap(BASE, e.policy, samples=1)
    ok, reason = e.validate_snapshot(low)
    assert not ok and reason == 'insufficient_sample_count'
    with pytest.raises(c.CSIUValidationError):
        c.CSIUEnforcementConfig(max_single_influence=.2, max_cumulative_influence_window=.1)


def test_prospective_cap_blocks_third_and_concurrent():
    e = c.CSIUEnforcement(c.CSIUEnforcementConfig(max_single_influence=.05, max_cumulative_influence_window=.10, history_capacity=10))
    plan={'id':'p','objective_weights':{'x':1.0}}
    decisions=[]
    for _ in range(3):
        _, d = e.apply_regularization_with_enforcement(plan, .05, BASE)
        decisions.append(d)
    assert decisions[0].applied
    assert decisions[1].applied
    assert not decisions[2].applied and decisions[2].reason_code == 'cumulative_cap_exceeded'
    assert e.check_cumulative_influence()['cumulative_influence'] <= .10


def test_no_mutation_and_alignment_proposal_no_activation():
    e = c.CSIUEnforcement()
    plan={'id':'p','objective_weights':{'x':1.0}, 'nested': {'a': []}}
    orig={'id':'p','objective_weights':{'x':1.0}, 'nested': {'a': []}}
    new, d = e.apply_regularization_with_enforcement(plan, .01, BASE)
    assert plan == orig
    assert d.applied and new != plan
    class Pol:
        policy_digest='b'*64; revision=3; policy_id='canonical-evidence-bound'
    prop = e.propose_alignment_policy(Pol(), [snap(BASE, e.policy), snap(BAD, e.policy)])
    assert prop.active_alignment_digest == 'b'*64
    assert prop.approval_state == 'pending_review'


def test_singleton_config_mismatch_rejected():
    c.reset_csiu_enforcer()
    c.get_csiu_enforcer(c.CSIUEnforcementConfig(max_single_influence=.05))
    with pytest.raises(c.CSIUValidationError):
        c.get_csiu_enforcer(c.CSIUEnforcementConfig(max_single_influence=.04))
    c.reset_csiu_enforcer()
