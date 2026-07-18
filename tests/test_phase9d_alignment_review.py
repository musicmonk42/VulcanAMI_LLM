from datetime import datetime, timezone, timedelta
from vulcan.world_model.meta_reasoning.csiu_enforcement import *
BASE=dict(A=.5,H=.5,C=.5,V=.5,D=.5,G=.5,E=.5,U=.5,M=.5)
def snap(vals, pol, i):
    t=datetime.now(timezone.utc)-timedelta(minutes=60-i*5)
    return CSIUMetricSnapshot(metrics=vals, window_start=t.isoformat().replace('+00:00','Z'), window_end=(t+timedelta(minutes=5)).isoformat().replace('+00:00','Z'), sample_count=30, aggregation_method='mean', metric_definition_version=pol.metric_definition_version, provider_id='p', provenance_digest=hex(i)[2:]*64, policy_digest=pol.policy_digest)
class Pol: policy_digest='b'*64; revision=7; policy_id='align'; max_claims_per_response=8
def test_alignment_proposal_review_only_valid_delta():
    e=CSIUEnforcement(); snaps=[snap({**BASE,'G':.5+.03*i,'M':.5+.03*i},e.policy,i) for i in range(4)]
    prop=e.propose_alignment_policy(Pol(),snaps)
    assert prop.active_alignment_revision==7 and prop.active_alignment_digest=='b'*64
    assert prop.approval_state=='pending_review'
    assert 'mapping' not in prop.proposed_policy_delta
