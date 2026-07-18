import json, os, time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from vulcan.world_model.meta_reasoning.csiu_enforcement import (
    CSIUEnforcement, CSIUEnforcementConfig, CSIUPolicy, CSIUMetricSnapshot, METRIC_ORDER, CSIUValidationError
)
from vulcan.world_model.meta_reasoning.self_improvement_drive import SelfImprovementDrive
from vulcan.world_model.meta_reasoning.governed_transaction import ClosedApprovalAuthority, ApprovalRecord, ApprovalStore, TransactionError


def _snap(enf, end, vals=None, prov="a"):
    return CSIUMetricSnapshot(metrics=vals or {k: .5 for k in METRIC_ORDER}, window_start=(end-timedelta(minutes=5)).isoformat().replace('+00:00','Z'), window_end=end.isoformat().replace('+00:00','Z'), sample_count=30, aggregation_method='mean', metric_definition_version=enf.policy.metric_definition_version, provider_id='phase9g', provenance_digest=prov*64, policy_digest=enf.policy.policy_digest, privacy_cohort={'kind':'aggregate'})


def test_explicit_empty_objectives_stays_empty(tmp_path, monkeypatch):
    monkeypatch.setenv('INTRINSIC_CSIU_OFF','1')
    cfg={"drives":{"self_improvement":{"enabled":True,"priority":1,"objectives":[],"constraints":{"require_human_approval":True},"triggers":[],"resource_limits":{}}}}
    d=SelfImprovementDrive(config_path=cfg, state_path=str(tmp_path/'s.json'))
    assert d.objectives == []
    assert d.select_objective() is None


def test_custom_policy_reopens_without_policy_and_preserves_header(tmp_path):
    store=tmp_path/'csiu.jsonl'
    policy=CSIUPolicy(policy_id='custom', weights={k:(i+1)/10 for i,k in enumerate(METRIC_ORDER)}, metric_ranges={k:(0,2) for k in METRIC_ORDER})
    e=CSIUEnforcement(CSIUEnforcementConfig(durable_store_path=str(store)), policy=policy)
    created=policy.created_at; digest=policy.policy_digest
    e.close()
    e2=CSIUEnforcement(CSIUEnforcementConfig(durable_store_path=str(store)))
    assert e2.policy.created_at == created
    assert e2.policy.policy_digest == digest
    e2.close()
    bad=replace(policy, weights={**policy.weights, 'A': policy.weights['A']+.1}, policy_digest='')
    with pytest.raises(CSIUValidationError):
        CSIUEnforcement(CSIUEnforcementConfig(durable_store_path=str(store)), policy=bad)


def test_baseline_persistence_failure_does_not_publish_cursor(tmp_path, monkeypatch):
    e=CSIUEnforcement(CSIUEnforcementConfig(durable_store_path=str(tmp_path/'c.jsonl')))
    s1=_snap(e, datetime.now(timezone.utc), prov='b')
    seen=set(e._seen_snapshot_digests); ew=e._last_ewma
    monkeypatch.setattr(e, '_persist', lambda r: (_ for _ in ()).throw(CSIUValidationError('boom')))
    with pytest.raises(CSIUValidationError):
        e.apply_regularization_from_snapshots({}, None, s1)
    assert e._last_snapshot is None and e._last_snapshot_digest == ''
    assert e._seen_snapshot_digests == seen and e._last_ewma == ew
    monkeypatch.undo()
    _, dec=e.apply_regularization_from_snapshots({}, None, s1)
    assert dec.reason_code == 'baseline_established'
    _, replay=e.apply_regularization_from_snapshots({}, None, s1)
    assert replay.reason_code in {'replayed_snapshot','overlapping_or_reordered_window'}


def test_public_csiu_observation_accepts_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv('INTRINSIC_CSIU_OFF','0')
    from vulcan.world_model.meta_reasoning.csiu_enforcement import reset_csiu_enforcer
    reset_csiu_enforcer()
    d=SelfImprovementDrive(config_path={"drives":{"self_improvement":{"enabled":True,"priority":1,"objectives":[],"constraints":{"require_human_approval":True}}}}, state_path=str(tmp_path/'s.json'))
    now=datetime.now(timezone.utc)
    vals={k:.5 for k in METRIC_ORDER}
    data={**{v: vals[k] for k,v in d._csiu_metric_keys().items()}, 'metrics.window_start':(now-timedelta(minutes=5)).isoformat().replace('+00:00','Z'), 'metrics.window_end':now.isoformat().replace('+00:00','Z'), 'metrics.sample_count':30, 'metrics.aggregation_method':'mean', 'metrics.metric_definition_version':d._csiu_enforcer.policy.metric_definition_version, 'metrics.provider_id':'phase9g', 'metrics.provenance_digest':'c'*64}
    d.set_metrics_provider(lambda k: data.get(k))
    dec=d.observe_csiu_telemetry()
    assert dec.reason_code == 'baseline_established'
    assert d._csiu_previous_snapshot.snapshot_digest == dec.snapshot_digest


def test_closed_approval_authority_accepts_only_trusted_principal(tmp_path):
    store=ApprovalStore(tmp_path/'a.json')
    auth=ClosedApprovalAuthority()
    p=auth.issue_principal('reviewer', ['self_improvement.approve'], 60)
    b={'approval_id':'a','proposal_digest':'1'*64,'policy_digest':'2'*64,'original_source_digest':'3'*64,'required_scope':'self_improvement.approve'}
    assert not auth.is_authorized('reviewer', b)
    assert auth.is_authorized(p, b)
    store.save(ApprovalRecord('a','1'*64,'2'*64,'3'*64,'reviewer',time.time(),time.time()+60))
    with pytest.raises(TransactionError):
        store.save(ApprovalRecord('a','1'*64,'2'*64,'3'*64,'reviewer',time.time(),time.time()+60))
