import asyncio, json, os, hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from vulcan.runtime.alignment import AlignmentRegistry, default_policy
from vulcan.runtime.audit import AuditError, CanonicalAudit
from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.finalization import FinalizationDecision, FinalizationResult
from vulcan.runtime.kernel import CognitiveKernel, KernelRequest
from vulcan.runtime.semantic import (Claim, Derivation, EpistemicStatus, EvidenceArtifact, EvidenceKind,
    ExecutionStatus, Proposition, Utterance, canonical_digest)

class Finalizer:
    async def finalize(self, artifact): return FinalizationResult(FinalizationDecision.ALLOW, artifact, artifact.text)

def data(case="c", digest="0"*64): return {"case_id":case,"request_id":"r","request_digest":digest}

def test_audit_append_reopen_tamper_truncate_duplicate_sequence_close_symlink(tmp_path):
    p=tmp_path/"audit.jsonl"; a=CanonicalAudit(p); ev=a.append("case.started", data()); payload={"x":["y"]}; a.append("case.interpreted", {**data(), "payload": payload}); payload["x"].append("mutated"); a.close(); a.close()
    b=CanonicalAudit(p); assert len(b.events_for_case("c"))==2; b.close()
    with pytest.raises(AuditError): b.append("case.started", data("x"))
    lines=p.read_text().splitlines(); o=json.loads(lines[0]); o["data"]["case_id"]="d"; p.write_text(json.dumps(o,separators=(",",":"))+"\n"+"\n".join(lines[1:])+"\n")
    with pytest.raises(AuditError): CanonicalAudit(p)
    p.write_text(lines[0])
    with pytest.raises(AuditError): CanonicalAudit(p)
    p.write_text(lines[0].replace('"sequence":1','"sequence":2')+"\n")
    with pytest.raises(AuditError): CanonicalAudit(p)
    p.write_text('{"schema_version":"vulcan-audit/1","schema_version":"x"}\n')
    with pytest.raises(AuditError): CanonicalAudit(p)
    os.symlink(tmp_path/"target", tmp_path/"sym.jsonl")
    with pytest.raises(AuditError): CanonicalAudit(tmp_path/"sym.jsonl")

def test_second_writer_invalid_type_timestamp_and_verified_case_retrieval(tmp_path):
    a=CanonicalAudit(tmp_path/"a.jsonl"); a.append("case.started", data())
    with pytest.raises(AuditError): a.append("bad.event", data())
    with pytest.raises(AuditError): CanonicalAudit(tmp_path/"a.jsonl")
    assert a.events_for_case("c")[0].event_type=="case.started"
    a.close()
    line=(tmp_path/"a.jsonl").read_text(); o=json.loads(line); o["timestamp"]="2020-01-01T00:00:00"; o["event_hash"]="0"*64
    (tmp_path/"a.jsonl").write_text(json.dumps(o,separators=(",",":"))+"\n")
    with pytest.raises(AuditError): CanonicalAudit(tmp_path/"a.jsonl")

def test_alignment_policy_load_update_cas_lease_and_fail_closed(tmp_path):
    a=CanonicalAudit(tmp_path/"audit.jsonl"); r=AlignmentRegistry(tmp_path/"policy.json", audit=a); old=r.active(); lease=r.lease()
    cand=default_policy().__dict__.copy(); cand.update(revision=2, max_claims_per_response=4, policy_digest=""); cand["permitted_epistemic_statuses"]=list(cand["permitted_epistemic_statuses"]); cand["policy_digest"]=hashlib.sha256(json.dumps({k:v for k,v in sorted(cand.items()) if k!="policy_digest"},ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    new=r.update(cand, expected_previous_digest=old.policy_digest); assert lease.policy.policy_digest==old.policy_digest and r.active().policy_digest==new.policy_digest
    with pytest.raises(ValueError): r.update(cand, expected_previous_digest=old.policy_digest)
    lease.close(); r.close(); a.close()
    (tmp_path/"policy.json").write_text("{}")
    with pytest.raises(ValueError): AlignmentRegistry(tmp_path/"policy.json")

def test_alignment_blocks_bad_claims_and_passes_valid(tmp_path):
    r=AlignmentRegistry(tmp_path/"p.json")
    now=datetime.now(timezone.utc); ev=EvidenceArtifact("e1", EvidenceKind.RETRIEVED_RECORD, canonical_digest("v"), "ref", "origin", "exact", "case", observed_at=now, valid_until=now+timedelta(days=1), source_integrity="digest-verified", citation="ref")
    d=Derivation("d1","m","1",("e1",),"c1",(),(),True,ExecutionStatus.SUCCESS,"t")
    c=Claim("c1", Proposition("s","p","v"), EpistemicStatus.RETRIEVED, ("e1",), ("d1",), citation_ids=("e1",))
    assert r.decide((c,), (ev,), (d,)).accepted
    assert not r.decide((c,), (EvidenceArtifact("e1", EvidenceKind.RETRIEVED_RECORD, canonical_digest("v"), "ref", "origin", "exact", "case", observed_at=now, source_integrity="bad"),), (d,)).accepted
    expired=EvidenceArtifact("e1", EvidenceKind.RETRIEVED_RECORD, canonical_digest("v"), "ref", "origin", "exact", "case", observed_at=now-timedelta(days=2), valid_until=now-timedelta(days=1), source_integrity="digest-verified", citation="ref")
    assert not r.decide((c,), (expired,), (d,)).accepted
    assert not r.decide(tuple([c]*9), (ev,), (d,)).accepted

def test_terminal_audit_failure_keeps_case_open_and_blocks_positive_response(tmp_path):
    class FailingAudit(CanonicalAudit):
        def append(self,t,d):
            if t=="case.completed": raise AuditError("disk full")
            return super().append(t,d)
    audit=FailingAudit(tmp_path/"a.jsonl"); align=AlignmentRegistry(tmp_path/"p.json")
    k=CognitiveKernel(state_authority=SimpleNamespace(version="1"), finalizer=Finalizer(), audit=audit, alignment=align)
    u=Utterance.from_text("2+2"); case=CognitiveCase.create(request_id="r", conversation_id=None, input_digest=u.digest)
    with pytest.raises(AuditError): asyncio.run(k.handle(KernelRequest(u,None), case))
    assert case.terminal_status is CognitiveCaseStatus.OPEN
    assert [e.event_type for e in audit.events_for_case(case.case_id)][-1] == "case.finalized"


def test_alignment_lease_cleanup_precedes_terminal_audit_commit(tmp_path):
    order = []

    class Lease:
        policy = SimpleNamespace(policy_digest="policy", revision=1)
        def close(self): order.append("alignment_closed")

    class Alignment:
        def lease(self): return Lease()
        def decide(self, claims, evidence, derivations, policy):
            return SimpleNamespace(accepted=True, reason_codes=("passed",), policy_digest="policy", policy_revision=1)

    class ObservingAudit(CanonicalAudit):
        def append(self, t, d):
            if t == "case.completed": order.append("terminal_audit")
            return super().append(t, d)

    audit=ObservingAudit(tmp_path/"a.jsonl")
    k=CognitiveKernel(state_authority=SimpleNamespace(version="1"), finalizer=Finalizer(), audit=audit, alignment=Alignment())
    u=Utterance.from_text("2+2"); case=CognitiveCase.create(request_id="r", conversation_id=None, input_digest=u.digest)
    asyncio.run(k.handle(KernelRequest(u,None), case))
    assert order == ["alignment_closed", "terminal_audit"]
    assert case.terminal_status is CognitiveCaseStatus.SUCCESS

def test_no_raw_prompt_token_or_secret_in_case_events(tmp_path):
    audit=CanonicalAudit(tmp_path/"a.jsonl"); align=AlignmentRegistry(tmp_path/"p.json")
    k=CognitiveKernel(state_authority=SimpleNamespace(version="1"), finalizer=Finalizer(), audit=audit, alignment=align)
    u=Utterance.from_text("secret token 2+2"); case=CognitiveCase.create(request_id="r", conversation_id=None, input_digest=u.digest)
    asyncio.run(k.handle(KernelRequest(u,None), case))
    txt=(tmp_path/"a.jsonl").read_text().lower()
    assert "secret token" not in txt and "authorization" not in txt

def test_activation_audit_failures_leave_prior_domain_and_policy_active(tmp_path):
    from tests.security.test_persistent_domain_registry import bundle
    class Fail:
        def __init__(self): self.fail=False
        def append(self, t, d):
            if self.fail: raise AuditError("audit down")
    fail=Fail(); from vulcan.runtime.domain_registry import PersistentDomainRegistry
    dom=PersistentDomainRegistry(tmp_path/"dom", audit=fail); sid1=dom.load_bundle(bundle('geo',1,'Paris')); fail.fail=True
    with pytest.raises(AuditError): dom.load_bundle(bundle('geo',2,'Lyon'), expected_previous_digest=dom._active.domains['geo'].digest)
    assert dom.domain_snapshot_id==sid1 and dom.lookup_exact('france.capital').value=='paris'
    fail.fail=False; ar=AlignmentRegistry(tmp_path/"policy2.json", audit=fail); old=ar.active(); cand=default_policy().__dict__.copy(); cand.update(revision=2, max_claims_per_response=4, policy_digest=""); cand["permitted_epistemic_statuses"]=list(cand["permitted_epistemic_statuses"]); cand["policy_digest"]=hashlib.sha256(json.dumps({k:v for k,v in sorted(cand.items()) if k!="policy_digest"},ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(); fail.fail=True
    with pytest.raises(AuditError): ar.update(cand, expected_previous_digest=old.policy_digest)
    assert ar.active().policy_digest==old.policy_digest
