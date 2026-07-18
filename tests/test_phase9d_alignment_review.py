from datetime import datetime, timezone, timedelta
from vulcan.world_model.meta_reasoning.csiu_enforcement import *
from vulcan.runtime.alignment import default_policy, validate_policy
import pytest, hashlib, json
BASE=dict(A=.5,H=.5,C=.5,V=.5,D=.5,G=.5,E=.5,U=.5,M=.5)
def snap(vals, pol, i):
    t=datetime.now(timezone.utc)-timedelta(minutes=60-i*5)
    return CSIUMetricSnapshot(metrics=vals, window_start=t.isoformat().replace('+00:00','Z'), window_end=(t+timedelta(minutes=5)).isoformat().replace('+00:00','Z'), sample_count=30, aggregation_method='mean', metric_definition_version=pol.metric_definition_version, provider_id='p', provenance_digest=hex(i)[2:]*64, policy_digest=pol.policy_digest)
def digest(d):
    x=dict(d); x.pop("policy_digest",None)
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
class Pol: policy_digest='b'*64; revision=7; policy_id='canonical-evidence-bound'; max_claims_per_response=8
def test_alignment_proposal_review_only_valid_delta():
    e=CSIUEnforcement(); snaps=[snap({**BASE,'G':.5+.03*i,'M':.5+.03*i},e.policy,i) for i in range(4)]
    prop=e.propose_alignment_policy(Pol(),snaps)
    assert prop.active_alignment_revision==7 and prop.active_alignment_digest=='b'*64
    assert prop.approval_state=='pending_review'
    assert 'mapping' not in prop.proposed_policy_delta
    assert 'G' in prop.reason_codes and 'calibration' not in prop.expected_effect.lower()
    candidate=default_policy().__dict__.copy()
    candidate["permitted_epistemic_statuses"]=list(candidate["permitted_epistemic_statuses"])
    candidate.update(prop.proposed_policy_delta); candidate["revision"]=2; candidate["policy_digest"]=""; candidate["policy_digest"]=digest(candidate)
    assert validate_policy(candidate).revision==2
    with pytest.raises(CSIUValidationError):
        e.propose_alignment_policy(Pol(),snaps)
