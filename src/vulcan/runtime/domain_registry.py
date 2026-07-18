"""Persistent evidence-bound factual domain registry for Graphix lookup.

Domain bundles are governed data. Raw documents must pass a separate governed
extraction/review step that emits exact line-oriented JSON assertions before
Vulcan may use them as factual support. Digest integrity proves the reviewed
bytes used by Vulcan; it does not prove those bytes are true.
"""
from __future__ import annotations

import copy, hashlib, json, os, re, tempfile, threading, unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable
from urllib.parse import urlparse

SCHEMA_VERSION = "vulcan-domain/1"
MAX_FILE_BYTES = 1_000_000
MAX_ENTRIES = 1024
MAX_CONTENT = 65536
MAX_RETURNED_EVIDENCE = 16
_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_NAME = re.compile(r"([a-z0-9][a-z0-9_.-]{0,63})-(\d{10}).json")
_ALLOWED_TOP = {"schema_version","domain","version","revision","evidence","facts","digest"}
_ALLOWED_EVID = {"evidence_id","uri","content","content_digest","acquired_at","valid_until","acquisition_method","license","provenance"}
_ALLOWED_FACT = {"fact_id","subject","predicate","object","evidence_ids","valid_from","valid_until"}
_ALLOWED_PROV = {"reviewer","source_system","notes","case_id"}

@dataclass(frozen=True)
class DomainEvidenceSupport:
    domain: str; revision: int; fact_id: str; evidence_id: str; uri: str; content_digest: str; acquired_at: datetime; valid_until: datetime | None; acquisition_method: str; license: str; provenance: tuple[tuple[str,str], ...]

@dataclass(frozen=True)
class DomainLookupResult:
    status: str; key: str; value: str | None; domain_snapshot_id: str; evidence: tuple[DomainEvidenceSupport, ...] = (); contested_values: tuple[str, ...] = (); truncated: bool = False; total_evidence: int = 0; truncation_limit: int = MAX_RETURNED_EVIDENCE

@dataclass(frozen=True)
class _Evidence:
    evidence_id: str; uri: str; content: str; content_digest: str; acquired_at: datetime; valid_until: datetime | None; acquisition_method: str; license: str; provenance: tuple[tuple[str,str], ...]
@dataclass(frozen=True)
class _Fact:
    fact_id: str; subject: str; predicate: str; object: str; evidence_ids: tuple[str, ...]; valid_from: datetime | None; valid_until: datetime | None
@dataclass(frozen=True)
class _Bundle:
    domain: str; version: str; revision: int; digest: str; evidence: MappingProxyType; facts: tuple[_Fact, ...]
@dataclass(frozen=True)
class _Snapshot:
    snapshot_id: str; domains: MappingProxyType; index: MappingProxyType

class _Lease:
    def __init__(self, reg: 'PersistentDomainRegistry', snap: _Snapshot): self._r=reg; self._s=snap; self.domain_snapshot_id=snap.snapshot_id; self._closed=False
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def close(self):
        if not self._closed: self._closed=True; self._r._release(self._s.snapshot_id)
    def lookup_exact(self, key: str) -> DomainLookupResult: return self._r._lookup(self._s, key)

class PersistentDomainRegistry:
    def __init__(self, root: str | os.PathLike[str], *, audit: Callable[[dict[str,Any]],None] | None=None, retain_snapshots: int=4):
        self.root=Path(root); self.audit=audit; self.retain=max(1, retain_snapshots); self._lock=threading.RLock(); self._leases: dict[str,int]={}
        self.root.mkdir(parents=True, exist_ok=True); self._active=self._restore(); self._snapshots={self._active.snapshot_id:self._active}
    @property
    def domain_snapshot_id(self)->str: return self._active.snapshot_id
    def lease(self)->_Lease:
        with self._lock:
            self._leases[self._active.snapshot_id]=self._leases.get(self._active.snapshot_id,0)+1
            return _Lease(self,self._active)
    def close(self): pass
    def lookup_exact(self, key: str): return self.lease().lookup_exact(key)
    def load_bundle(self, data: bytes | str, *, expected_previous_digest: str | None=None)->str:
        bundle=_parse_bundle(data)
        with self._lock:
            old=self._active.domains.get(bundle.domain)
            if old:
                if bundle.revision <= old.revision: raise ValueError("non-monotonic revision")
                if not expected_previous_digest or expected_previous_digest != old.digest: raise ValueError("stale expected_previous_digest")
            elif expected_previous_digest is not None: raise ValueError("unexpected previous digest")
            domains=dict(self._active.domains); domains[bundle.domain]=bundle
            snap=_build_snapshot(domains)
            if self.audit:
                append = getattr(self.audit, "append", None)
                event_data={"domain":bundle.domain,"revision":bundle.revision,"version":bundle.version,"prior_bundle_digest":old.digest if old else None,"new_bundle_digest":bundle.digest,"snapshot_id":snap.snapshot_id,"fact_count":len(bundle.facts),"evidence_count":len(bundle.evidence),"actor_id":"system"}
                if append: append("domain.activated", event_data)
                else: self.audit(event_data)
            self._persist(bundle)
            self._active=snap; self._snapshots[snap.snapshot_id]=snap
            self._evict()
            return snap.snapshot_id
    def _release(self, sid):
        with self._lock:
            n=self._leases.get(sid,0)-1
            if n>0: self._leases[sid]=n
            else: self._leases.pop(sid,None)
    def _evict(self):
        extra=len(self._snapshots)-self.retain
        if extra<=0: return
        victims=[sid for sid in self._snapshots if sid != self._active.snapshot_id][:extra]
        if any(self._leases.get(sid) for sid in victims): raise RuntimeError("snapshot retention exhausted by live lease")
        for sid in victims: self._snapshots.pop(sid, None)
    def _persist(self,b:_Bundle):
        name=f"{b.domain}-{b.revision:010d}.json"; final=self.root/name; raw=_canonical_bytes(_bundle_to_public(b))
        fd,tmp=tempfile.mkstemp(prefix=f".{name}.",suffix=".tmp",dir=self.root)
        with os.fdopen(fd,"wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, final)
        try:
            d=os.open(self.root, os.O_DIRECTORY); os.fsync(d); os.close(d)
        except OSError: pass
    def _restore(self)->_Snapshot:
        domains={}; files=[]
        for p in self.root.iterdir():
            if p.is_symlink() or not p.is_file(): raise ValueError("unexpected persisted file")
            m=_NAME.fullmatch(p.name)
            if not m: raise ValueError("malformed persisted filename")
            files.append((m.group(1), int(m.group(2)), p))
        for dom,rev,p in sorted(files):
            if p.stat().st_size>MAX_FILE_BYTES: raise ValueError("oversized persisted file")
            b=_parse_bundle(p.read_bytes())
            if b.domain!=dom or b.revision!=rev: raise ValueError("filename/bundle mismatch")
            old=domains.get(dom)
            if old and b.revision<=old.revision: raise ValueError("conflicting history")
            domains[dom]=b
        return _build_snapshot(domains)
    def _lookup(self,snap:_Snapshot,key:str)->DomainLookupResult:
        sub,pred=_split_key(key); now=datetime.now(timezone.utc); supports=[]; vals={}
        for fact, bundle, evs in snap.index.get((sub,pred),()):
            if fact.valid_until and fact.valid_until < now: continue
            good=[]
            for ev in evs:
                if ev.valid_until and ev.valid_until < now: good=[]; break
                good.append(DomainEvidenceSupport(bundle.domain,bundle.revision,fact.fact_id,ev.evidence_id,ev.uri,ev.content_digest,ev.acquired_at,ev.valid_until,ev.acquisition_method,ev.license,ev.provenance))
            if good: vals.setdefault(fact.object,[]).extend(good)
        if not vals: return DomainLookupResult("unknown", key, None, snap.snapshot_id)
        if len(vals)>1: return DomainLookupResult("contested", key, None, snap.snapshot_id, contested_values=tuple(sorted(vals)))
        val, ev=list(vals.items())[0]; trunc=len(ev)>MAX_RETURNED_EVIDENCE
        return DomainLookupResult("retrieved", key, val, snap.snapshot_id, tuple(ev[:MAX_RETURNED_EVIDENCE]), truncated=trunc, total_evidence=len(ev))

def _split_key(k):
    if not isinstance(k,str) or "." not in k: raise ValueError("lookup key must be subject.predicate")
    a,b=k.rsplit('.',1); return _norm_term(a), _norm_term(b)
def _norm_term(s): return _text(s,256).casefold().strip()
def _text(v,maxlen):
    if not isinstance(v,str) or not v or len(v)>maxlen: raise ValueError("invalid text")
    n=unicodedata.normalize("NFC",v)
    if any((ord(c)<32 and c not in "\n\t\r") or 0xD800<=ord(c)<=0xDFFF for c in n): raise ValueError("invalid unicode/control")
    return n
def _dt(v, optional=False):
    if v is None and optional: return None
    s=_text(v,64).replace('Z','+00:00')
    try: d=datetime.fromisoformat(s)
    except ValueError: raise ValueError("invalid timestamp") from None
    if d.tzinfo is None: raise ValueError("invalid timestamp")
    return d.astimezone(timezone.utc)
def _loads(data):
    raw=data.encode('utf-8') if isinstance(data,str) else data
    if len(raw)>MAX_FILE_BYTES: raise ValueError("oversized bundle")
    def pairs(p):
        seen={}
        for k,v in p:
            if k in seen: raise ValueError("duplicate JSON key")
            seen[k]=v
        return seen
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in()).throw(ValueError("NaN/Infinity rejected")))
def _parse_bundle(data):
    o=_loads(data)
    if set(o)!=_ALLOWED_TOP or o.get('schema_version')!=SCHEMA_VERSION: raise ValueError("invalid bundle schema")
    dom=_text(o['domain'],64).casefold();
    if not _ID.fullmatch(dom): raise ValueError("invalid domain")
    rev=o['revision'];
    if type(rev) is not int or rev<0: raise ValueError("invalid revision")
    if not isinstance(o['evidence'],list) or len(o['evidence'])>MAX_ENTRIES or not isinstance(o['facts'],list) or len(o['facts'])>MAX_ENTRIES: raise ValueError("entry bound")
    evmap={}
    for e in o['evidence']:
        if set(e)-_ALLOWED_EVID or not _ALLOWED_EVID-{"valid_until","provenance"} <= set(e): raise ValueError("invalid evidence fields")
        eid=_text(e['evidence_id'],96); uri=_text(e['uri'],512); scheme=urlparse(uri).scheme
        if scheme not in {'https','urn','file'}: raise ValueError("unsupported URI")
        content=_text(e['content'],MAX_CONTENT); cd=e['content_digest']
        if cd!=hashlib.sha256(content.encode()).hexdigest(): raise ValueError("source digest mismatch")
        acq=_dt(e['acquired_at']); vu=_dt(e.get('valid_until'), True)
        if vu and vu<acq: raise ValueError("reversed validity")
        prov=e.get('provenance') or {}
        if not isinstance(prov,dict) or set(prov)-_ALLOWED_PROV: raise ValueError("invalid provenance")
        ev=_Evidence(eid,uri,content,cd,acq,vu,_text(e['acquisition_method'],64),_text(e['license'],64),tuple(sorted((str(k),_text(str(v),256)) for k,v in prov.items())))
        if eid in evmap: raise ValueError("duplicate evidence id")
        evmap[eid]=ev
    facts=[]; fids=set()
    for f in o['facts']:
        if set(f)-_ALLOWED_FACT or not _ALLOWED_FACT-{"valid_from","valid_until"} <= set(f): raise ValueError("invalid fact fields")
        fid=_text(f['fact_id'],96)
        if fid in fids: raise ValueError("duplicate fact id")
        refs=f['evidence_ids']
        if not isinstance(refs,list) or not refs: raise ValueError("fact requires evidence")
        refs=tuple(_text(r,96) for r in refs)
        if not set(refs)<=set(evmap): raise ValueError("unknown evidence reference")
        fact=_Fact(fid,_norm_term(f['subject']),_norm_term(f['predicate']),_norm_term(f['object']),refs,_dt(f.get('valid_from'),True),_dt(f.get('valid_until'),True))
        if fact.valid_from and fact.valid_until and fact.valid_until<fact.valid_from: raise ValueError("reversed validity")
        for r in refs:
            assertions=[json.loads(line, object_pairs_hook=lambda p: dict(p)) for line in evmap[r].content.splitlines() if line.strip().startswith('{')]
            if {"subject":fact.subject,"predicate":fact.predicate,"object":fact.object} not in [{k:_norm_term(v) for k,v in a.items()} for a in assertions if set(a)=={"subject","predicate","object"}]: raise ValueError("evidence does not assert fact")
        fids.add(fid); facts.append(fact)
    pub=copy.deepcopy(o); pub.pop('digest')
    digest=hashlib.sha256(_canonical_bytes(pub)).hexdigest()
    if o['digest']!=digest: raise ValueError("bundle digest mismatch")
    return _Bundle(dom,_text(o['version'],64),rev,digest,MappingProxyType(evmap),tuple(facts))
def _canonical_bytes(o): return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def _bundle_to_public(b):
    evs=[]
    for e in b.evidence.values():
        d={"evidence_id":e.evidence_id,"uri":e.uri,"content":e.content,"content_digest":e.content_digest,"acquired_at":e.acquired_at.isoformat().replace('+00:00','Z'),"acquisition_method":e.acquisition_method,"license":e.license}
        if e.valid_until is not None: d["valid_until"]=e.valid_until.isoformat().replace('+00:00','Z')
        if e.provenance: d["provenance"]=dict(e.provenance)
        evs.append(d)
    facts=[]
    for f in b.facts:
        d={"fact_id":f.fact_id,"subject":f.subject,"predicate":f.predicate,"object":f.object,"evidence_ids":list(f.evidence_ids)}
        if f.valid_from is not None: d["valid_from"]=f.valid_from.isoformat().replace('+00:00','Z')
        if f.valid_until is not None: d["valid_until"]=f.valid_until.isoformat().replace('+00:00','Z')
        facts.append(d)
    return {"schema_version":SCHEMA_VERSION,"domain":b.domain,"version":b.version,"revision":b.revision,"evidence":evs,"facts":facts,"digest":b.digest}
def _build_snapshot(domains):
    temp={}
    for b in domains.values():
        for f in b.facts: temp.setdefault((f.subject,f.predicate),[]).append((f,b,tuple(b.evidence[e] for e in f.evidence_ids)))
    frozen={k:tuple(v) for k,v in sorted(temp.items())}
    sid='domain:'+hashlib.sha256(_canonical_bytes({d:(b.revision,b.digest) for d,b in sorted(domains.items())})).hexdigest()[:16]
    return _Snapshot(sid, MappingProxyType(dict(domains)), MappingProxyType(frozen))
