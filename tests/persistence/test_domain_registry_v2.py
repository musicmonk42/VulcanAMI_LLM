import hashlib
import json
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from vulcan.runtime.domain_registry import PersistentDomainRegistry

NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def bundle(domain="geo", rev=1, value="Paris", *, acquired="2026-07-20T00:00:00Z", valid_until=None, valid_from=None, content=None):
    assertion = content if content is not None else json.dumps({"subject":"france","predicate":"capital","object":value})
    ev={"evidence_id":"e0","uri":f"https://example.test/{domain}","content":assertion,"content_digest":hashlib.sha256(assertion.encode()).hexdigest(),"acquired_at":acquired,"acquisition_method":"reviewed-jsonl","license":"CC0","provenance":{"reviewer":"alice","case_id":"case-1"}}
    if valid_until: ev["valid_until"]=valid_until
    fact={"fact_id":"f0","subject":"france","predicate":"capital","object":value,"evidence_ids":["e0"]}
    if valid_from: fact["valid_from"]=valid_from
    if valid_until: fact["valid_until"]=valid_until
    o={"schema_version":"vulcan-domain/1","domain":domain,"version":f"v{rev}","revision":rev,"evidence":[ev],"facts":[fact]}
    o["digest"]=hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return json.dumps(o,separators=(",",":"))


def registry(path, **kw):
    return PersistentDomainRegistry(path, clock=lambda: NOW, **kw)


def test_case_sensitive_objects_and_full_snapshot_digest(tmp_path):
    r=registry(tmp_path/"r")
    sid=r.load_bundle(bundle(value="Paris"), actor_id="operator:alice", approval_provenance={"review":"R1"})
    assert len(sid) == len("domain:") + 64
    res=r.lookup_exact("france.capital")
    assert res.status == "retrieved"
    assert res.value == "Paris"


def test_duplicate_embedded_key_future_evidence_and_conflict_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate JSON key"):
        registry(tmp_path/"dup").load_bundle(bundle(content='{"subject":"france","predicate":"capital","object":"Paris","object":"Lyon"}'))
    with pytest.raises(ValueError, match="future evidence"):
        registry(tmp_path/"future").load_bundle(bundle(acquired="2026-07-21T00:00:00Z"))
    with pytest.raises(ValueError, match="evidence does not assert"):
        registry(tmp_path/"conflict").load_bundle(bundle(value="Paris", content=json.dumps({"subject":"france","predicate":"capital","object":"Lyon"})))


def test_lease_close_is_idempotent_and_close_reports_leaks(tmp_path):
    r=registry(tmp_path/"r", retain_snapshots=2)
    r.load_bundle(bundle())
    lease=r.lease()
    lease.close(); lease.close()
    leaked=r.lease()
    with pytest.raises(RuntimeError, match="leases leaked"):
        r.close()
    leaked.close()


def test_audit_failure_does_not_publish_or_serve_missing_audit(tmp_path):
    events=[]
    class Audit:
        fail_commit=False
        def append(self, event_type, data):
            if event_type == "domain.activation_committed" and self.fail_commit:
                raise RuntimeError("audit down")
            events.append((event_type, data))
    audit=Audit(); r=registry(tmp_path/"r", audit=audit)
    sid1=r.load_bundle(bundle(value="Paris"), actor_id="operator:alice", approval_provenance={"ticket":"T1"})
    old=r._active.domains["geo"].digest
    audit.fail_commit=True
    with pytest.raises(RuntimeError, match="audit down"):
        r.load_bundle(bundle(rev=2,value="Lyon"), expected_previous_digest=old, actor_id="operator:bob")
    assert r.domain_snapshot_id == sid1
    assert r.lookup_exact("france.capital").value == "Paris"
    assert not (tmp_path/"r"/"geo-0000000002.json").exists()
    assert events[0][1]["actor_id"] == "operator:alice"
    assert events[0][1]["approval_provenance"] == {"ticket":"T1"}


def test_concurrent_writers_allow_one_cas_winner(tmp_path):
    r=registry(tmp_path/"r")
    r.load_bundle(bundle())
    old=r._active.domains["geo"].digest
    def write(value):
        try:
            return r.load_bundle(bundle(rev=2,value=value), expected_previous_digest=old)
        except ValueError as e:
            return str(e)
    with ThreadPoolExecutor(max_workers=2) as ex:
        results=list(ex.map(write, ["Paris", "Lyon"]))
    assert sum(str(x).startswith("domain:") for x in results) == 1
    assert any("stale expected_previous_digest" in str(x) or "non-monotonic" in str(x) for x in results)
