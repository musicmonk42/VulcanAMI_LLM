from datetime import datetime, timezone, timedelta
import hashlib, json
import pytest
from vulcan.runtime.alignment import AlignmentRegistry, default_policy
from vulcan.runtime.audit import CanonicalAudit
from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement, CSIUMetricSnapshot

def digest(d):
    x=dict(d); x.pop('policy_digest',None)
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def snap(enf, t, g, m, i):
    metrics={k:0.5 for k in ('A','H','C','V','D','G','E','U','M')}; metrics['G']=g; metrics['M']=m
    return CSIUMetricSnapshot(metrics=metrics, window_start=(t-timedelta(minutes=5)).isoformat().replace('+00:00','Z'), window_end=t.isoformat().replace('+00:00','Z'), sample_count=30, aggregation_method='privacy_aggregate', metric_definition_version=enf.policy.metric_definition_version, provider_id='test', provenance_digest=format(i,'x')*64, policy_digest=enf.policy.policy_digest)

def test_reviewed_alignment_proposal_cas_and_stable_lease(tmp_path):
    audit=CanonicalAudit(tmp_path/'a.jsonl'); reg=AlignmentRegistry(tmp_path/'p.json', audit=audit)
    active=reg.active(); lease=reg.lease(); enf=CSIUEnforcement()
    base=datetime.now(timezone.utc)-timedelta(minutes=40)
    snaps=[snap(enf,base+timedelta(minutes=i*10),0.2+i*0.03,0.2+i*0.03,i) for i in range(4)]
    prop=enf.propose_alignment_policy(active, snaps)
    assert prop.schema_version=='vulcan-csiu-alignment-proposal/1'
    assert prop.active_alignment_revision==1 and prop.active_alignment_digest==active.policy_digest
    assert prop.proposed_policy_delta['max_claims_per_response']==7
    assert reg.active().revision==1
    assert not any(hasattr(enf, name) for name in ('activate','activation_token','alignment_registry'))
    cand=default_policy().__dict__.copy(); cand['permitted_epistemic_statuses']=list(cand['permitted_epistemic_statuses'])
    cand.update({k:v for k,v in prop.proposed_policy_delta.items() if k in cand}); cand['revision']=2; cand['policy_digest']=''; cand['policy_digest']=digest(cand)
    new=reg.activate(cand, expected_previous_digest=prop.active_alignment_digest, actor_id='reviewer')
    assert new.revision==2 and reg.active().revision==2
    assert lease.revision==1 and lease.policy_digest==active.policy_digest
    with pytest.raises(ValueError): reg.activate(cand, expected_previous_digest=prop.active_alignment_digest, actor_id='reviewer')
    assert [e.event_type for e in audit.events_for_alignment(active.policy_id)] == ['alignment.activation_prepared','alignment.activation_committed']
