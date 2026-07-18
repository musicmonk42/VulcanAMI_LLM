from datetime import datetime, timezone, timedelta
import json, math
import pytest
from vulcan.world_model.meta_reasoning.csiu_enforcement import *

BASE=dict(A=.5,H=.5,C=.5,V=.5,D=.5,G=.5,E=.5,U=.5,M=.5)
IMP=dict(A=.6,H=.4,C=.6,V=.4,D=.4,G=.4,E=.6,U=.6,M=.4)

def snap(vals, pol, start, end, prov):
    return CSIUMetricSnapshot(metrics=vals, window_start=start.isoformat().replace('+00:00','Z'), window_end=end.isoformat().replace('+00:00','Z'), sample_count=30, aggregation_method='mean', metric_definition_version=pol.metric_definition_version, provider_id='p', provenance_digest=prov*64, policy_digest=pol.policy_digest, privacy_cohort={'c':'a'})

def test_baseline_then_s2_authoritative_and_ewma_restart(tmp_path):
    t=datetime.now(timezone.utc)-timedelta(minutes=30)
    cfg=CSIUEnforcementConfig(durable_store_path=str(tmp_path/'csiu.jsonl'))
    e=CSIUEnforcement(cfg)
    s1=snap(BASE,e.policy,t,t+timedelta(minutes=5),'a')
    plan={'objective_weights':{'x':1.0}}
    out,d=e.apply_regularization_from_snapshots(plan,None,s1)
    assert d.reason_code=='baseline_established' and not d.applied and out==plan
    s2=snap(IMP,e.policy,t+timedelta(minutes=5),t+timedelta(minutes=10),'b')
    out,d=e.apply_regularization_from_snapshots(plan,s1,s2)
    assert d.applied and d.utility>0 and d.ewma_utility == pytest.approx(e.policy.ewma_alpha*d.utility)
    assert e.get_statistics()['last_decision_digest']==d.decision_digest
    ew=d.ewma_utility
    e.close(); e2=CSIUEnforcement(CSIUEnforcementConfig(durable_store_path=str(tmp_path/'csiu.jsonl')))
    assert s2.snapshot_digest in e2._seen_snapshot_digests
    s3=snap(IMP,e2.policy,t+timedelta(minutes=10),t+timedelta(minutes=15),'c')
    _,d3=e2.apply_regularization_from_snapshots(plan,s2,s3)
    assert d3.ewma_utility == pytest.approx((1-e2.policy.ewma_alpha)*ew)
    _,replay=e2.apply_regularization_from_snapshots(plan,s2,s3)
    assert replay.reason_code in {'replayed_snapshot','previous_snapshot_digest_mismatch'}

def test_public_arbitrary_pressure_bypass_removed():
    e=CSIUEnforcement()
    with pytest.raises(CSIUValidationError):
        e.apply_regularization_with_enforcement({'objective_weights':{'x':1.0}}, .05, {})
