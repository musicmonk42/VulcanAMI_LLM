import json, hashlib, os, pytest
from vulcan.runtime.domain_registry import PersistentDomainRegistry
from vulcan.runtime.semantic import AcceptedInterpretation, GraphixPlan, compile_graphix_plan, execute_graphix_plan

def bundle(domain='geo', rev=1, value='Paris', evidence_contents=None, refs=None, acquired=None, valid_until=None):
    if evidence_contents is None:
        evidence_contents=[json.dumps({'subject':'france','predicate':'capital','object':value.lower()})]
    ev=[]
    for i,c in enumerate(evidence_contents):
        d={'evidence_id':f'e{i}','uri':f'https://example.test/{domain}/{i}','content':c,'content_digest':hashlib.sha256(c.encode()).hexdigest(),'acquired_at':acquired or '2026-01-01T00:00:00Z','acquisition_method':'reviewed-jsonl','license':'CC0','provenance':{'reviewer':'test'}}
        if valid_until is not None: d['valid_until']=valid_until
        ev.append(d)
    o={'schema_version':'vulcan-domain/1','domain':domain,'version':f'v{rev}','revision':rev,'evidence':ev,'facts':[{'fact_id':'f0','subject':'france','predicate':'capital','object':value.lower(),'evidence_ids':refs or [e['evidence_id'] for e in ev]}]}
    o['digest']=hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return json.dumps(o,separators=(',',':'))

def test_exact_structured_assertion_accepted_and_lookup(tmp_path):
    r=PersistentDomainRegistry(tmp_path/'r'); sid=r.load_bundle(bundle())
    with r.lease() as l:
        res=l.lookup_exact('France.capital')
    assert res.status=='retrieved' and res.value=='paris' and res.domain_snapshot_id==sid and len(res.evidence)==1

def test_natural_language_negations_and_cooccurrence_rejected(tmp_path):
    bad=['The capital of France is Paris.','The capital of France is not Paris.','It is false that the capital of France is Paris.','France. capital. Paris.']
    for i,txt in enumerate(bad):
        with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/str(i)).load_bundle(bundle(evidence_contents=[txt]))

def test_every_cited_source_must_support_and_refs_known(tmp_path):
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'a').load_bundle(bundle(evidence_contents=[json.dumps({'subject':'france','predicate':'capital','object':'paris'}), json.dumps({'subject':'france','predicate':'capital','object':'lyon'})]))
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'b').load_bundle(bundle(refs=['missing']))

def test_duplicate_keys_unknown_fields_digest_nan_timestamps(tmp_path):
    raw=b'{"schema_version":"vulcan-domain/1","schema_version":"x"}'
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'d').load_bundle(raw)
    o=json.loads(bundle()); o['extra']=1; o['digest']='0'*64
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'u').load_bundle(json.dumps(o))
    o=json.loads(bundle()); o['revision']=float('nan')
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'n').load_bundle(json.dumps(o, allow_nan=True))
    o=json.loads(bundle()); o['evidence'][0]['acquired_at']='not-a-date'; o['digest']='bad'
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'t').load_bundle(json.dumps(o))
    o=json.loads(bundle()); o['evidence'][0]['content']+='x'
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'m').load_bundle(json.dumps(o))

def test_updates_require_monotonic_revision_and_cas(tmp_path):
    r=PersistentDomainRegistry(tmp_path/'r'); r.load_bundle(bundle(rev=1)); old=r._active.domains['geo'].digest
    with pytest.raises(ValueError): r.load_bundle(bundle(rev=1), expected_previous_digest=old)
    with pytest.raises(ValueError): r.load_bundle(bundle(rev=2), expected_previous_digest='0'*64)
    r.load_bundle(bundle(rev=2,value='Paris'), expected_previous_digest=old)

def test_concordant_and_conflicting_domains(tmp_path):
    r=PersistentDomainRegistry(tmp_path/'r'); r.load_bundle(bundle('a',1,'Paris')); r.load_bundle(bundle('b',1,'Paris'))
    assert len(r.lease().lookup_exact('france.capital').evidence)==2
    r2=PersistentDomainRegistry(tmp_path/'x'); r2.load_bundle(bundle('a',1,'Paris')); r2.load_bundle(bundle('b',1,'Lyon'))
    assert r2.lease().lookup_exact('france.capital').status=='contested'
    r3=PersistentDomainRegistry(tmp_path/'y'); r3.load_bundle(bundle('b',1,'Lyon')); r3.load_bundle(bundle('a',1,'Paris'))
    assert r3.lease().lookup_exact('france.capital').status=='contested'

def test_expired_unknown_snapshot_lease_persist_restore_and_bad_files(tmp_path):
    r=PersistentDomainRegistry(tmp_path/'expired', retain_snapshots=1); r.load_bundle(bundle(acquired='2019-01-01T00:00:00Z', valid_until='2020-01-01T00:00:00Z'))
    assert r.lease().lookup_exact('france.capital').status=='unknown'
    r=PersistentDomainRegistry(tmp_path/'p', retain_snapshots=2); r.load_bundle(bundle(value='Paris')); lease=r.lease(); old=lease.domain_snapshot_id
    r.load_bundle(bundle(rev=2,value='Lyon'), expected_previous_digest=r._active.domains['geo'].digest)
    assert lease.lookup_exact('france.capital').value=='paris' and lease.domain_snapshot_id==old
    lease.close(); assert PersistentDomainRegistry(tmp_path/'p').lease().lookup_exact('france.capital').value=='lyon'
    (tmp_path/'bad').mkdir(); (tmp_path/'bad'/'junk.txt').write_text('x')
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'bad')
    (tmp_path/'sym').mkdir(); os.symlink(tmp_path/'sym'/'missing', tmp_path/'sym'/'geo-0000000001.json')
    with pytest.raises(ValueError): PersistentDomainRegistry(tmp_path/'sym')

def test_graphix_snapshot_mismatch_and_domain_hint_cannot_filter(tmp_path):
    r=PersistentDomainRegistry(tmp_path/'r'); r.load_bundle(bundle('a',1,'Paris')); r.load_bundle(bundle('b',1,'Lyon'))
    acc=AcceptedInterpretation(0,'lookup','France.capital')
    lease=r.lease(); plan=GraphixPlan('graphix-plan/1','plan-'+'a'*32,'lookup','req','state',lease.domain_snapshot_id,(('key','France.capital'),))
    comp=compile_graphix_plan(plan, request_digest='req', state_snapshot_id='state', domain_snapshot_id=lease.domain_snapshot_id)
    claim,_,_=execute_graphix_plan(comp, request_digest='req', state_snapshot_id='state', domain_snapshot_id=lease.domain_snapshot_id, domain=lease)
    assert claim.status.value=='contested'
    with pytest.raises(ValueError): execute_graphix_plan(comp, request_digest='req', state_snapshot_id='state', domain_snapshot_id='domain:other', domain=lease)
    bad=GraphixPlan('graphix-plan/1','plan-'+'b'*32,'lookup','req','state',lease.domain_snapshot_id,(('key','France.capital'),('domain_hint','a')))
    with pytest.raises(ValueError): compile_graphix_plan(bad, request_digest='req', state_snapshot_id='state', domain_snapshot_id=lease.domain_snapshot_id)
    lease.close()
