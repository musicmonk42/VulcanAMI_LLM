from __future__ import annotations
"""Inspectable CSIU enforcement and typed contracts.

CSIU is visible to operators/auditors. It can regularize only a closed set of
plan metadata fields and can propose, but never activate, alignment policy
changes.
"""
import copy, fcntl, hashlib, json, math, os, threading, time, uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

try:
    from vulcan.world_model.meta_reasoning.serialization_mixin import SerializationMixin
except Exception:
    class SerializationMixin:
        pass

SCHEMA_SNAPSHOT="vulcan-csiu-metric-snapshot/1"; SCHEMA_POLICY="vulcan-csiu-policy/1"; SCHEMA_DECISION="vulcan-csiu-decision/1"; SCHEMA_RECORD="vulcan-csiu-influence-record/2"; SCHEMA_PROPOSAL="vulcan-csiu-alignment-proposal/1"; SCHEMA_STORE_HEADER="vulcan-csiu-durable-store/1"
UTC=timezone.utc
METRIC_ORDER=("A","H","C","V","D","G","E","U","M")
DIRECTIONS={"A":1,"H":-1,"C":1,"V":-1,"D":-1,"G":-1,"E":1,"U":1,"M":-1}
DEFAULT_RANGES={k:(0.0,1.0) for k in METRIC_ORDER}
DEFAULT_WEIGHTS={"A":0.6,"H":0.6,"C":0.6,"V":0.6,"D":0.6,"G":0.6,"E":0.5,"U":0.5,"M":0.5}
MAX_ITEMS=64; MAX_STR=512
def re_full_hash(s): return isinstance(s,str) and len(s)==64 and all(c in "0123456789abcdef" for c in s)

class CSIUValidationError(ValueError): pass

def _now(): return datetime.now(UTC)
def _ts(dt: datetime)->str:
    if dt.tzinfo is None: raise CSIUValidationError("timestamp must be timezone-aware UTC")
    return dt.astimezone(UTC).isoformat().replace("+00:00","Z")
def _parse_ts(s: str)->datetime:
    if not isinstance(s,str) or not s.endswith("Z"): raise CSIUValidationError("invalid timestamp")
    try: return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(UTC)
    except Exception as e: raise CSIUValidationError("invalid timestamp") from e

def _num(x: Any, name="number")->float:
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(float(x)): raise CSIUValidationError(f"invalid {name}")
    return float(x)
def _clean(v: Any, depth=0)->Any:
    if depth>8: raise CSIUValidationError("nested object too deep")
    if v is None or isinstance(v,bool): return v
    if isinstance(v,(int,float)): return _num(v)
    if isinstance(v,str):
        if len(v)>MAX_STR or any((ord(c)<32 and c not in "\n\t\r") for c in v): raise CSIUValidationError("invalid string")
        return v
    if isinstance(v,Mapping):
        if len(v)>MAX_ITEMS: raise CSIUValidationError("object too large")
        return {str(k):_clean(val,depth+1) for k,val in sorted(v.items())}
    if isinstance(v,(list,tuple)):
        if len(v)>MAX_ITEMS: raise CSIUValidationError("array too large")
        return [_clean(x,depth+1) for x in v]
    raise CSIUValidationError("unsupported value")
def canonical_digest(o: Mapping[str,Any])->str:
    return hashlib.sha256(json.dumps(_clean(dict(o)),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _id(prefix: str, body: Mapping[str,Any])->str: return f"{prefix}-{canonical_digest(body)[:24]}"

@dataclass(frozen=True)
class CSIUPolicy:
    schema_version: str = SCHEMA_POLICY
    policy_version: str = "csiu-weight-policy/1"
    metric_definition_version: str = "csiu-metrics/1"
    weights: Mapping[str,float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    metric_ranges: Mapping[str,Tuple[float,float]] = field(default_factory=lambda: dict(DEFAULT_RANGES))
    min_sample_count: int = 30
    max_snapshot_age_seconds: float = 3600.0
    ewma_alpha: float = 0.3
    max_single_influence: float = 0.05
    max_cumulative_influence_window: float = 0.10
    cumulative_window_seconds: float = 3600.0
    policy_id: str = "csiu-policy-default"
    created_at: str = "1970-01-01T00:00:00Z"
    policy_digest: str = ""
    def __post_init__(self):
        w={k:_num(self.weights.get(k),f"weight {k}") for k in METRIC_ORDER}
        if set(self.weights)!=set(METRIC_ORDER): raise CSIUValidationError("policy weight keys mismatch")
        if any(v<0 or v>10 for v in w.values()): raise CSIUValidationError("weight out of bounds")
        ranges={}
        for k in METRIC_ORDER:
            lo,hi=self.metric_ranges.get(k,(None,None)); lo=_num(lo); hi=_num(hi)
            if hi<=lo: raise CSIUValidationError("bad metric range")
            ranges[k]=(lo,hi)
        if type(self.min_sample_count) is not int or self.min_sample_count<1: raise CSIUValidationError("bad sample count")
        for n in (self.max_snapshot_age_seconds,self.ewma_alpha,self.max_single_influence,self.max_cumulative_influence_window,self.cumulative_window_seconds): _num(n)
        if not (0<self.ewma_alpha<=1) or self.max_single_influence<=0 or self.max_cumulative_influence_window<self.max_single_influence or self.cumulative_window_seconds<=0: raise CSIUValidationError("bad policy caps")
        _parse_ts(self.created_at)
        body=self.to_dict(include_digest=False); digest=canonical_digest(body)
        object.__setattr__(self,"weights",w); object.__setattr__(self,"metric_ranges",ranges)
        if self.policy_digest and self.policy_digest!=digest: raise CSIUValidationError("policy digest mismatch")
        object.__setattr__(self,"policy_digest",digest)
    def to_dict(self, include_digest=True):
        d={"schema_version":self.schema_version,"policy_id":self.policy_id,"policy_version":self.policy_version,"metric_definition_version":self.metric_definition_version,"weights":dict(self.weights),"metric_ranges":{k:list(v) for k,v in self.metric_ranges.items()},"min_sample_count":self.min_sample_count,"max_snapshot_age_seconds":self.max_snapshot_age_seconds,"ewma_alpha":self.ewma_alpha,"max_single_influence":self.max_single_influence,"max_cumulative_influence_window":self.max_cumulative_influence_window,"cumulative_window_seconds":self.cumulative_window_seconds,"created_at":self.created_at}
        if include_digest: d["policy_digest"]=self.policy_digest
        return d

@dataclass(frozen=True)
class CSIUMetricSnapshot:
    metrics: Mapping[str,float]; window_start: str; window_end: str; sample_count: int; aggregation_method: str; metric_definition_version: str; provider_id: str; provenance_digest: str; policy_digest: str; schema_version: str=SCHEMA_SNAPSHOT; snapshot_id: str=""; created_at: str=field(default_factory=lambda:_ts(_now())); privacy_cohort: Optional[Mapping[str,Any]]=None; snapshot_digest: str=""
    def __post_init__(self):
        if set(self.metrics)!=set(METRIC_ORDER): raise CSIUValidationError("missing metrics")
        m={k:_num(self.metrics[k],k) for k in METRIC_ORDER}
        for k,v in m.items():
            lo,hi=DEFAULT_RANGES[k]
            if not (lo<=v<=hi): raise CSIUValidationError(f"metric {k} out of range")
        ws=_parse_ts(self.window_start); we=_parse_ts(self.window_end); _parse_ts(self.created_at)
        if we<=ws: raise CSIUValidationError("bad window")
        if type(self.sample_count) is not int or self.sample_count<1: raise CSIUValidationError("bad sample count")
        if not self.provider_id or not self.provenance_digest: raise CSIUValidationError("missing provenance")
        object.__setattr__(self,"metrics",m)
        body=self.to_dict(include_digest=False); sid=self.snapshot_id or _id("csiu-snapshot",body); object.__setattr__(self,"snapshot_id",sid)
        body["snapshot_id"]=sid; dg=canonical_digest(body)
        if self.snapshot_digest and self.snapshot_digest!=dg: raise CSIUValidationError("snapshot digest mismatch")
        object.__setattr__(self,"snapshot_digest",dg)
    def to_dict(self, include_digest=True):
        d={"schema_version":self.schema_version,"snapshot_id":self.snapshot_id,"created_at":self.created_at,"window_start":self.window_start,"window_end":self.window_end,"sample_count":self.sample_count,"aggregation_method":self.aggregation_method,"metric_definition_version":self.metric_definition_version,"provider_id":self.provider_id,"provenance_digest":self.provenance_digest,"policy_digest":self.policy_digest,"metrics":dict(self.metrics),"privacy_cohort":dict(self.privacy_cohort or {})}
        if include_digest: d["snapshot_digest"]=self.snapshot_digest
        return d

@dataclass(frozen=True)
class CSIUDecision:
    decision_id: str; policy_digest: str; plan_digest: str; reason_code: str; utility: float=0.0; ewma_utility: float=0.0; pressure: float=0.0; actual_effect: float=0.0; applied: bool=False; blocked: bool=True; snapshot_id: str=""; snapshot_digest: str=""; schema_version: str=SCHEMA_DECISION; created_at: str=field(default_factory=lambda:_ts(_now())); decision_digest: str=""
    def __post_init__(self):
        for n in (self.utility,self.ewma_utility,self.pressure,self.actual_effect): _num(n)
        _parse_ts(self.created_at); body=self.to_dict(False); dg=canonical_digest(body)
        if self.decision_digest and self.decision_digest!=dg: raise CSIUValidationError("decision digest mismatch")
        object.__setattr__(self,"decision_digest",dg)
    def to_dict(self, include_digest=True):
        d=self.__dict__.copy(); d.pop("decision_digest",None)
        if include_digest: d["decision_digest"]=self.decision_digest
        return dict(sorted(d.items()))

@dataclass(frozen=True)
class CSIUInfluenceRecord:
    decision_id: str; policy_digest: str; plan_digest: str; reason_code: str; previous_snapshot_digest: str; current_snapshot_digest: str; decision_digest: str; utility: float; previous_ewma_utility: float; ewma_utility: float; reserved_influence: float; measured_actual_effect: float; charged_influence: float; applied: bool; blocked: bool; state: str; enforcer_id: str; timestamp: str=field(default_factory=lambda:_ts(_now())); schema_version: str=SCHEMA_RECORD; previous_record_digest: str="0"*64; record_id: str=""; record_digest: str=""
    def __post_init__(self):
        for n in ("utility","previous_ewma_utility","ewma_utility","reserved_influence","measured_actual_effect","charged_influence"):
            _num(getattr(self,n), n)
        _parse_ts(self.timestamp)
        if self.state not in {"prepared","committed","aborted","blocked","telemetry"}: raise CSIUValidationError("bad state")
        if bool(self.applied) == bool(self.blocked): raise CSIUValidationError("applied/blocked must be exclusive")
        if not isinstance(self.enforcer_id,str) or not self.enforcer_id: raise CSIUValidationError("missing enforcer id")
        if not isinstance(self.reason_code,str) or not self.reason_code or len(self.reason_code)>128: raise CSIUValidationError("bad reason code")
        for name in ("policy_digest","plan_digest","decision_digest","previous_record_digest"):
            if not re_full_hash(getattr(self,name)): raise CSIUValidationError(f"bad {name}")
        for name in ("previous_snapshot_digest","current_snapshot_digest"):
            v=getattr(self,name)
            if v and not re_full_hash(v): raise CSIUValidationError(f"bad {name}")
        body=self.to_dict(False); rid=self.record_id or _id("csiu-record",body); object.__setattr__(self,"record_id",rid); body["record_id"]=rid; dg=canonical_digest(body)
        if self.record_digest and self.record_digest!=dg: raise CSIUValidationError("record digest mismatch")
        object.__setattr__(self,"record_digest",dg)
    @property
    def pressure(self): return self.reserved_influence
    @property
    def actual_effect(self): return self.measured_actual_effect
    @property
    def actual_influence(self): return self.measured_actual_effect
    def to_dict(self, include_digest=True):
        d={"schema_version":self.schema_version,"enforcer_id":self.enforcer_id,"policy_digest":self.policy_digest,"plan_digest":self.plan_digest,"decision_id":self.decision_id,"decision_digest":self.decision_digest,"reason_code":self.reason_code,"previous_snapshot_digest":self.previous_snapshot_digest,"current_snapshot_digest":self.current_snapshot_digest,"utility":self.utility,"previous_ewma_utility":self.previous_ewma_utility,"ewma_utility":self.ewma_utility,"reserved_influence":self.reserved_influence,"measured_actual_effect":self.measured_actual_effect,"charged_influence":self.charged_influence,"applied":self.applied,"blocked":self.blocked,"state":self.state,"timestamp":self.timestamp,"previous_record_digest":self.previous_record_digest,"record_id":self.record_id}
        if include_digest: d["record_digest"]=self.record_digest
        return dict(sorted(d.items()))

@dataclass(frozen=True)
class CSIUAlignmentProposal:
    active_alignment_policy_id: str; active_alignment_revision: int; active_alignment_digest: str; csiu_policy_digest: str; supporting_snapshot_digests: Tuple[str,...]; trend: Mapping[str,float]; reason_codes: Tuple[str,...]; expected_effect: str; required_evaluation: str; expires_at: str; proposed_policy_delta: Mapping[str,Any]=field(default_factory=dict); approval_state: str="pending_review"; schema_version: str=SCHEMA_PROPOSAL; proposal_id: str=""; created_at: str=field(default_factory=lambda:_ts(_now())); proposal_digest: str=""
    def __post_init__(self):
        _parse_ts(self.created_at); _parse_ts(self.expires_at)
        if self.approval_state not in {"pending_review","accepted","rejected","expired"}: raise CSIUValidationError("bad approval")
        body=self.to_dict(False); pid=self.proposal_id or _id("csiu-align",body); object.__setattr__(self,"proposal_id",pid); body["proposal_id"]=pid; dg=canonical_digest(body)
        if self.proposal_digest and self.proposal_digest!=dg: raise CSIUValidationError("proposal digest mismatch")
        object.__setattr__(self,"proposal_digest",dg)
    def to_dict(self, include_digest=True):
        d=self.__dict__.copy(); d["supporting_snapshot_digests"]=list(self.supporting_snapshot_digests); d["reason_codes"]=list(self.reason_codes); d.pop("proposal_digest",None)
        if include_digest: d["proposal_digest"]=self.proposal_digest
        return dict(sorted(d.items()))

@dataclass(frozen=True)
class CSIUEnforcementConfig:
    max_single_influence: float=0.05; max_cumulative_influence_window: float=0.10; cumulative_window_seconds: float=3600.0; history_capacity: int=1000; global_enabled: bool=True; calculation_enabled: bool=True; regularization_enabled: bool=True; proposal_generation_enabled: bool=True; history_tracking_enabled: bool=True; emergency_stop: bool=False; audit_trail_enabled: bool=True; audit_trail_max_entries: int=10000; durable_accounting_required: bool=False; durable_store_path: Optional[str]=None; clock: Optional[Callable[[],datetime]]=None
    def __post_init__(self):
        for name in ("max_single_influence","max_cumulative_influence_window","cumulative_window_seconds"):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)): raise CSIUValidationError(f"bad {name}")
        if self.max_single_influence<=0 or self.max_cumulative_influence_window<self.max_single_influence or self.cumulative_window_seconds<=0 or self.cumulative_window_seconds>86400*30: raise CSIUValidationError("bad cap relationship")
        if type(self.history_capacity) is not int or not 1<=self.history_capacity<=100000: raise CSIUValidationError("bad history capacity")
        if type(self.audit_trail_max_entries) is not int or self.audit_trail_max_entries<0: raise CSIUValidationError("bad audit capacity")

class CSIUEnforcement(SerializationMixin):
    _unpickleable_attrs=["_lock","_clock"]
    def __init__(self, config: Optional[CSIUEnforcementConfig]=None, policy: Optional[CSIUPolicy]=None):
        self.config=config or CSIUEnforcementConfig(); self.policy=policy or CSIUPolicy(max_single_influence=self.config.max_single_influence,max_cumulative_influence_window=self.config.max_cumulative_influence_window,cumulative_window_seconds=self.config.cumulative_window_seconds)
        self._lock=threading.RLock(); self._clock=self.config.clock or _now; self.enforcer_id=f"csiu:{self.policy.policy_id}:{self.policy.policy_digest}"; self._durable_ready=False; self._durable_prev="0"*64; self._durable_fd=None; self._influence_history=[]; self._audit_trail=[]; self._last_decision=None; self._last_snapshot=None; self._last_ewma=0.0; self._closed=False; self._seen_snapshot_digests=set(); self._total_applications=0; self._total_blocked=0; self._total_capped=0; self._max_influence_seen=0.0; self._last_snapshot_digest=""
        try: self._load_durable()
        except Exception:
            if self._durable_fd is not None:
                try: os.close(self._durable_fd)
                finally: self._durable_fd=None
            raise
    def _restore_unpickleable_attrs(self): self._lock=threading.RLock(); self._clock=_now
    def _emit(self,event, data):
        rec={"event":event,"timestamp":_ts(self._clock()),"data":_clean(data)}
        if self.config.audit_trail_enabled and len(self._audit_trail)<self.config.audit_trail_max_entries: self._audit_trail.append(rec)
    def _load_json_line(self,line: bytes)->dict:
        def pairs(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise CSIUValidationError("duplicate durable key")
                d[k]=v
            return d
        return json.loads(line.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in()).throw(CSIUValidationError("non-finite durable number")))
    def _load_durable(self):
        p=self.config.durable_store_path
        if not p:
            self._durable_ready=not self.config.durable_accounting_required
            if self.config.durable_accounting_required: raise CSIUValidationError("durable accounting store required")
            return
        path=Path(p); lock_path=path.with_suffix(path.suffix+".lock")
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_symlink() or lock_path.is_symlink(): raise CSIUValidationError("symlinked durable store or lock")
        self._durable_fd=os.open(lock_path, os.O_CREAT|os.O_RDWR,0o600)
        try: fcntl.flock(self._durable_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise CSIUValidationError("second durable writer") from e
        if not path.exists():
            if self.config.durable_accounting_required: raise CSIUValidationError("missing durable store")
            path.touch(mode=0o600)
        if path.stat().st_size == 0:
            self._write_header(path)
        raw=path.read_bytes()
        if raw and not raw.endswith(b"\n"): raise CSIUValidationError("truncated durable store")
        lines=raw.splitlines()
        if not lines: raise CSIUValidationError("missing durable policy header")
        header=self._load_json_line(lines[0])
        if set(header)!={"schema_version","policy","config","header_digest"} or header.get("schema_version")!=SCHEMA_STORE_HEADER: raise CSIUValidationError("missing durable policy header")
        hd=header.pop("header_digest")
        if canonical_digest(header)!=hd: raise CSIUValidationError("durable policy header digest mismatch")
        stored_policy=CSIUPolicy(**header["policy"])
        if self.policy.policy_digest!=stored_policy.policy_digest: raise CSIUValidationError("durable policy/config mismatch")
        cfgdoc=header["config"]
        for k in ("max_single_influence","max_cumulative_influence_window","cumulative_window_seconds"):
            if float(cfgdoc.get(k)) != float(getattr(self.config,k)): raise CSIUValidationError("durable policy/config mismatch")
        prev="0"*64; seen=set(); retained=[]; cutoff=self._clock()-timedelta(seconds=self.config.cumulative_window_seconds)
        required=set(CSIUInfluenceRecord("0"*64,self.policy.policy_digest,"0"*64,"bootstrap","","","0"*64,0.0,0.0,0.0,0.0,0.0,0.0,False,True,"telemetry",self.enforcer_id).to_dict().keys())
        for line in lines[1:]:
            d=self._load_json_line(line)
            if set(d)!=required: raise CSIUValidationError("unknown durable fields")
            r=CSIUInfluenceRecord(**d)
            if r.record_id in seen: raise CSIUValidationError("duplicate durable record")
            seen.add(r.record_id)
            if r.previous_record_digest!=prev: raise CSIUValidationError("durable chain mutation")
            if r.policy_digest!=self.policy.policy_digest or r.enforcer_id!=self.enforcer_id: raise CSIUValidationError("durable policy/config mismatch")
            if r.state=="prepared": raise CSIUValidationError("unresolved prepared influence")
            if r.current_snapshot_digest:
                self._seen_snapshot_digests.add(r.current_snapshot_digest); self._last_ewma=float(r.ewma_utility); self._last_snapshot_digest=r.current_snapshot_digest
            if _parse_ts(r.timestamp)>=cutoff and r.state=="committed": retained.append(r)
            prev=r.record_digest
        self._influence_history=retained; self._durable_prev=prev; self._durable_ready=True
    def readiness(self):
        if self._closed: raise CSIUValidationError("csiu enforcer closed")
        if self.config.durable_accounting_required and not self._durable_ready: raise CSIUValidationError("durable accounting unavailable")
        return True
    def close(self):
        if self._durable_fd is not None:
            try: fcntl.flock(self._durable_fd, fcntl.LOCK_UN); os.close(self._durable_fd)
            finally: self._durable_fd=None
        self._durable_ready=False; self._closed=True
    def _write_header(self, path: Path):
        doc={"schema_version":SCHEMA_STORE_HEADER,"policy":self.policy.to_dict(),"config":{"max_single_influence":self.config.max_single_influence,"max_cumulative_influence_window":self.config.max_cumulative_influence_window,"cumulative_window_seconds":self.config.cumulative_window_seconds}}
        doc["header_digest"]=canonical_digest(doc)
        with open(path,"ab",buffering=0) as f:
            f.write(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"); f.flush(); os.fsync(f.fileno())
    def _persist(self,r):
        p=self.config.durable_store_path
        if p:
            path=Path(p)
            if path.is_symlink(): raise CSIUValidationError("symlinked durable store")
            with open(path,"ab",buffering=0) as f:
                doc=r.to_dict()
                line=json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
                f.write(line); f.flush(); os.fsync(f.fileno())
            self._durable_prev=r.record_digest
    def is_enabled(self): return self.config.global_enabled and not self.config.emergency_stop
    def validate_snapshot(self, snapshot: CSIUMetricSnapshot)->Tuple[bool,str]:
        if self._closed: return False,"enforcer_closed"
        if snapshot.policy_digest!=self.policy.policy_digest: return False,"policy_digest_mismatch"
        if snapshot.metric_definition_version!=self.policy.metric_definition_version: return False,"metric_definition_mismatch"
        if snapshot.sample_count<self.policy.min_sample_count: return False,"insufficient_sample_count"
        ws=_parse_ts(snapshot.window_start); we=_parse_ts(snapshot.window_end)
        age=(self._clock()-we).total_seconds()
        if age<0 or age>self.policy.max_snapshot_age_seconds: return False,"stale_snapshot"
        if self._last_snapshot is not None:
            last_end=_parse_ts(self._last_snapshot.window_end)
            if ws < last_end: return False,"overlapping_or_reordered_window"
            if snapshot.provider_id!=self._last_snapshot.provider_id or snapshot.provenance_digest==self._last_snapshot.provenance_digest or snapshot.privacy_cohort!=self._last_snapshot.privacy_cohort:
                return False,"provider_or_provenance_incompatible"
        if snapshot.snapshot_digest in self._seen_snapshot_digests: return False,"replayed_snapshot"
        return True,"ok"
    def compute_utility(self, prev: CSIUMetricSnapshot, cur: CSIUMetricSnapshot)->float:
        total=sum(abs(self.policy.weights[k]) for k in METRIC_ORDER)
        if total<=0: raise CSIUValidationError("zero weights")
        acc=0.0
        for k in METRIC_ORDER:
            lo,hi=self.policy.metric_ranges[k]; raw=(cur.metrics[k]-prev.metrics[k])*DIRECTIONS[k]; acc+=self.policy.weights[k]*(raw/(hi-lo))
        u=acc/total
        if not math.isfinite(u): raise CSIUValidationError("non-finite utility")
        return max(-1.0,min(1.0,u))
    def pressure_from_utility(self, u: float)->float:
        u=_num(u,"utility"); p=self.config.max_single_influence*math.tanh(u)
        return max(-self.config.max_single_influence,min(self.config.max_single_influence,p))
    def check_cumulative_influence(self):
        with self._lock:
            self._prune_locked(); cum=sum(abs(r.charged_influence) for r in self._influence_history if r.state=="committed")
            return {"cumulative_influence":cum,"count":len(self._influence_history),"window_seconds":self.config.cumulative_window_seconds,"max_allowed":self.config.max_cumulative_influence_window,"remaining":max(0.0,self.config.max_cumulative_influence_window-cum),"exceeds_cap":cum>self.config.max_cumulative_influence_window}
    def _prune_locked(self):
        cutoff=self._clock()-timedelta(seconds=self.config.cumulative_window_seconds)
        self._influence_history=[r for r in self._influence_history if _parse_ts(r.timestamp)>=cutoff]
    def enforce_pressure_cap(self,pressure):
        p=_num(pressure,"pressure"); c=max(-self.config.max_single_influence,min(self.config.max_single_influence,p))
        if abs(c)<abs(p): self._total_capped+=1
        self._max_influence_seen=max(self._max_influence_seen,abs(c)); return c
    def _reserve_locked(self, decision: CSIUDecision)->Tuple[bool,str]:
        self._prune_locked()
        if len(self._influence_history)>=self.config.history_capacity: return False,"history_capacity_exhausted"
        cur=sum(abs(r.charged_influence) for r in self._influence_history if r.state=="committed")
        need=max(abs(decision.actual_effect),abs(decision.pressure))
        if need>self.config.max_single_influence+1e-12: return False,"single_cap_exceeded"
        if cur+need>self.config.max_cumulative_influence_window+1e-12: return False,"cumulative_cap_exceeded"
        return True,"ok"
    def apply_regularization_with_enforcement(self, *args, **kwargs):
        raise CSIUValidationError("public arbitrary pressure route removed; use apply_regularization_from_snapshots")

    def _record_for_decision(self, dec: CSIUDecision, state: str, previous_snapshot_digest: str, effect: float, charged: float) -> CSIUInfluenceRecord:
        return CSIUInfluenceRecord(dec.decision_id,self.policy.policy_digest,dec.plan_digest,dec.reason_code,previous_snapshot_digest or "",dec.snapshot_digest or "",dec.decision_digest,dec.utility,self._last_ewma,dec.ewma_utility,dec.pressure,effect,charged,dec.applied,dec.blocked,state,self.enforcer_id,timestamp=_ts(self._clock()),previous_record_digest=self._durable_prev)

    def _persist_terminal_decision(self, dec: CSIUDecision, previous_snapshot_digest: str, state: str = "telemetry") -> None:
        if self.config.durable_store_path:
            r=self._record_for_decision(dec,state,previous_snapshot_digest,dec.actual_effect,0.0 if not dec.applied else max(abs(dec.actual_effect),abs(dec.pressure)))
            self._persist(r)

    def _apply_regularization_with_enforcement(self, plan: Dict[str,Any], pressure: float, metrics: Mapping[str,float], plan_id="unknown", action_type="improvement", snapshot: Optional[CSIUMetricSnapshot]=None, *, _utility: float=0.0, _ewma: float=0.0, _previous_snapshot_digest: str="")->Tuple[Dict[str,Any],CSIUDecision]:
        if snapshot is None:
            raise CSIUValidationError("typed snapshot decision required")
        if self._closed: return copy.deepcopy(plan or {}), self._decision(canonical_digest({"plan":copy.deepcopy(plan or {})}),"enforcer_closed",0,0,0,True,False,snapshot)
        original=copy.deepcopy(plan or {}); input_digest=canonical_digest({"plan":plan or {}}); pd=canonical_digest({"plan":original})
        if not self.is_enabled() or not self.config.regularization_enabled:
            return original,self._decision(pd,"disabled",0,0,0,True,False,snapshot)
        try: pressure=self.enforce_pressure_cap(pressure)
        except Exception: return original,self._decision(pd,"invalid_pressure",0,0,0,True,False,snapshot)
        if pressure==0: return original,self._decision(pd,"zero_pressure",0,0,0,True,False,snapshot)
        if input_digest != pd: return original,self._decision(pd,"original_mutated_before_regularization",0,0,0,True,False,snapshot)
        proposed=copy.deepcopy(original); effect=0.0
        ow=proposed.get("objective_weights")
        if isinstance(ow,dict):
            if any(isinstance(v,bool) for v in ow.values()): return original,self._decision(pd,"invalid_objective_weight",0,0,0,True,False,snapshot)
            proposed["objective_weights"]={str(k):_num(v)*(1.0-0.03*pressure) for k,v in ow.items()}; effect=max(effect,abs(0.03*pressure))
            if set(proposed["objective_weights"]) != set(ow): return original,self._decision(pd,"objective_weight_removed",0,0,0,True,False,snapshot)
        proposed.setdefault("csiu_regularization",{})["route_penalty"]={"kind":"entropy","value":0.03*pressure}; effect=max(effect,abs(0.03*pressure))
        proposed.setdefault("csiu_regularization",{})["explainability_preference"]={"value":0.02*pressure}; effect=max(effect,abs(0.02*pressure))
        effect=self._measure_plan_effect(original, proposed)
        if effect>self.config.max_single_influence+1e-12: return original,self._decision(pd,"single_effect_exceeded",0,0,pressure,True,False,snapshot,effect)
        if not self._diff_allowed(original, proposed): return original,self._decision(pd,"closed_path_violation",0,0,pressure,True,False,snapshot,effect)
        dec=self._decision(pd,"applied",_utility,_ewma,pressure,False,True,snapshot,effect)
        with self._lock:
            ok,reason=self._reserve_locked(dec)
            if not ok:
                self._total_blocked+=1; b=CSIUDecision(dec.decision_id,self.policy.policy_digest,pd,reason,dec.utility,dec.ewma_utility,0.0,0.0,False,True,dec.snapshot_id,dec.snapshot_digest); self._persist_terminal_decision(b,_previous_snapshot_digest,"blocked"); self._emit("csiu.influence_blocked",b.to_dict()); return original,b
            r=self._record_for_decision(dec,"committed",_previous_snapshot_digest,effect,max(abs(effect),abs(pressure))); self._persist(r); self._influence_history.append(r); self._total_applications+=1; self._last_decision=dec; self._last_ewma=_ewma; self._emit("csiu.influence_applied",{"decision":dec.to_dict(),"record":r.to_dict(),"previous_snapshot_digest":_previous_snapshot_digest,"current_snapshot_digest":getattr(snapshot,"snapshot_digest",""),"decision_digest":dec.decision_digest,"utility":dec.utility,"ewma_utility":dec.ewma_utility,"budget":self.check_cumulative_influence()})
        if snapshot is not None:
            self._last_snapshot=snapshot; self._seen_snapshot_digests.add(snapshot.snapshot_digest)
        return proposed,dec
    def apply_regularization_from_snapshots(self, plan: Dict[str,Any], previous: Optional[CSIUMetricSnapshot], current: Optional[CSIUMetricSnapshot], plan_id="unknown", action_type="improvement"):
        original=copy.deepcopy(plan or {})
        if self._closed: raise CSIUValidationError("csiu enforcer closed")
        if previous is None or current is None:
            if current is not None:
                ok,reason=self.validate_snapshot(current)
                if ok:
                    self._last_snapshot=current; self._last_snapshot_digest=current.snapshot_digest; self._seen_snapshot_digests.add(current.snapshot_digest)
                    reason="baseline_established"
                    dec=self._decision(canonical_digest({"plan":original}),reason,0,0,0,True,False,current); self._persist_terminal_decision(dec,"","telemetry"); return original,dec
                return original,self._decision(canonical_digest({"plan":original}),reason,0,0,0,True,False,current)
            return original,self._decision(canonical_digest({"plan":original}),"missing_snapshot",0,0,0,True,False,current)
        expected=getattr(self,"_last_snapshot_digest",getattr(self._last_snapshot,"snapshot_digest",None))
        if expected and previous.snapshot_digest!=expected: return original,self._decision(canonical_digest({"plan":original}),"previous_snapshot_digest_mismatch",0,0,0,True,False,current)
        if previous.policy_digest!=self.policy.policy_digest or current.policy_digest!=self.policy.policy_digest: return original,self._decision(canonical_digest({"plan":original}),"policy_digest_mismatch",0,0,0,True,False,current)
        if current.snapshot_digest in self._seen_snapshot_digests: return original,self._decision(canonical_digest({"plan":original}),"replayed_snapshot",0,0,0,True,False,current)
        ok,reason=self.validate_snapshot(current)
        if not ok: return original,self._decision(canonical_digest({"plan":original}),reason,0,0,0,True,False,current)
        if _parse_ts(previous.window_end)>_parse_ts(current.window_start): return original,self._decision(canonical_digest({"plan":original}),"overlapping_or_reordered_window",0,0,0,True,False,current)
        if previous.provider_id!=current.provider_id or previous.privacy_cohort!=current.privacy_cohort: return original,self._decision(canonical_digest({"plan":original}),"provider_or_provenance_incompatible",0,0,0,True,False,current)
        u=self.compute_utility(previous,current); ew=self.policy.ewma_alpha*u + (1-self.policy.ewma_alpha)*self._last_ewma; pressure=self.pressure_from_utility(ew)
        proposed,dec=self._apply_regularization_evaluated(original, pressure, current.metrics, plan_id, action_type, current, u, ew, previous.snapshot_digest)
        return proposed,dec
    def _apply_regularization_evaluated(self, plan, pressure, metrics, plan_id, action_type, snapshot, utility, ewma, previous_snapshot_digest):
        proposed,dec=self._apply_regularization_with_enforcement(plan, pressure, metrics, plan_id, action_type, snapshot, _utility=utility, _ewma=ewma, _previous_snapshot_digest=previous_snapshot_digest)
        if dec.reason_code not in {"policy_digest_mismatch","metric_definition_mismatch","insufficient_sample_count","stale_snapshot","overlapping_or_reordered_window","provider_or_provenance_incompatible","replayed_snapshot","previous_snapshot_digest_mismatch"}:
            if not dec.applied and dec.reason_code != "cumulative_cap_exceeded":
                self._persist_terminal_decision(dec, previous_snapshot_digest, "telemetry")
            self._last_snapshot=snapshot; self._last_snapshot_digest=snapshot.snapshot_digest; self._seen_snapshot_digests.add(snapshot.snapshot_digest); self._last_ewma=ewma
        return proposed,dec
    def _measure_plan_effect(self, before, after):
        vals=[]
        ow1=before.get("objective_weights") if isinstance(before,dict) else None; ow2=after.get("objective_weights") if isinstance(after,dict) else None
        if isinstance(ow1,dict) and isinstance(ow2,dict):
            for k,v in ow1.items():
                if k not in ow2: raise CSIUValidationError("objective weight removed")
                vals.append(abs(_num(ow2[k])- _num(v))/max(1.0,abs(_num(v))))
        reg=after.get("csiu_regularization",{}) if isinstance(after,dict) else {}
        if isinstance(reg,dict):
            for entry in reg.values():
                if isinstance(entry,dict) and "value" in entry: vals.append(abs(_num(entry["value"])))
        return max(vals or [0.0])
    def _diff_allowed(self,before,after):
        allowed={('objective_weights',),('csiu_regularization',)}
        return all((path[:1] in allowed) for path in self._changed_paths(before,after))
    def _changed_paths(self,a,b,p=()):
        if type(a)!=type(b): return [p]
        if isinstance(a,dict):
            out=[]
            for k in set(a)|set(b): out += self._changed_paths(a.get(k), b.get(k), p+(k,))
            return out
        return [] if a==b else [p]
    def _decision(self,pd,reason,u,ew,p,blocked,applied,snapshot=None,effect=0.0):
        body={"policy_digest":self.policy.policy_digest,"plan_digest":pd,"reason":reason,"snapshot":getattr(snapshot,"snapshot_digest","")}; did=_id("csiu-decision",body)
        return CSIUDecision(did,self.policy.policy_digest,pd,reason,u,ew,p,effect,applied,blocked,getattr(snapshot,"snapshot_id",''),getattr(snapshot,"snapshot_digest",''))
    def propose_weight_revision(self, snapshots: List[CSIUMetricSnapshot])->Dict[str,Any]:
        return {"schema_version":"vulcan-csiu-weight-proposal/1","active_policy_digest":self.policy.policy_digest,"proposed_weights":dict(self.policy.weights),"supporting_snapshot_digests":[s.snapshot_digest for s in snapshots],"approval_state":"pending_review","proposal_digest":canonical_digest({"active_policy_digest":self.policy.policy_digest,"weights":dict(self.policy.weights)})}
    def propose_alignment_policy(self, active_policy: Any, snapshots: List[CSIUMetricSnapshot])->CSIUAlignmentProposal:
        dig=getattr(active_policy,"policy_digest",""); rev=int(getattr(active_policy,"revision",0)); pid=getattr(active_policy,"policy_id","vulcan-alignment/1")
        trend={k:0.0 for k in METRIC_ORDER}
        if len(snapshots)>=2:
            a,b=snapshots[-2],snapshots[-1]; trend={k:(b.metrics[k]-a.metrics[k])*DIRECTIONS[k] for k in METRIC_ORDER}
        reasons=tuple(k for k,v in trend.items() if v<0) or ("review_recommended",)
        delta={}
        # Conservative deterministic bridge mapping (review-only): when at least
        # four validated windows show sustained worsening in calibration (C) or
        # miscommunication (M), propose tightening the evidence-bound alignment
        # surface by lowering max_claims_per_response by one within [1,64] and
        # retaining/strengthening abstention/citation/integrity requirements.
        # CSIU cannot activate this; reviewers must evaluate and submit through
        # AlignmentRegistry.activate/update with exact CAS against this digest.
        if len(snapshots)>=4:
            sustained=[]
            for k in ("C","M"):
                vals=[(snapshots[i].metrics[k]-snapshots[i-1].metrics[k])*DIRECTIONS[k] for i in range(1,len(snapshots))]
                if len(vals)>=3 and all(v<=-0.02 for v in vals[-3:]): sustained.append(k)
            if sustained:
                current=int(getattr(active_policy,"max_claims_per_response",8))
                delta={"max_claims_per_response":max(1,current-1),"explicit_unknown_behavior":"abstain","require_citations":True,"require_verified_integrity":True,"require_temporal_validity":True}
                reasons=tuple(sorted(set(reasons+tuple(sustained)+("conservative_alignment_tightening",))))
        return CSIUAlignmentProposal(pid,rev,dig,self.policy.policy_digest,tuple(s.snapshot_digest for s in snapshots[-4:]),trend,reasons,"review human-understanding and safety thresholds; proposed deltas only tighten controls","authorized governance review plus ordinary alignment CAS activation",_ts(self._clock()+timedelta(hours=24)),delta)
    def get_statistics(self):
        c=self.check_cumulative_influence(); return {"enabled":self.is_enabled(),"policy_digest":self.policy.policy_digest,"metric_definition_version":self.policy.metric_definition_version,"last_valid_snapshot_digest":getattr(self._last_snapshot,"snapshot_digest",None),"last_decision_digest":getattr(self._last_decision,"decision_digest",None),"total_applications":self._total_applications,"total_blocked":self._total_blocked,"total_capped":self._total_capped,"max_influence_seen":self._max_influence_seen,"cumulative_stats":c,"kill_switches":{"global_enabled":self.config.global_enabled,"calculation_enabled":self.config.calculation_enabled,"regularization_enabled":self.config.regularization_enabled,"proposal_generation_enabled":self.config.proposal_generation_enabled,"history_display_enabled":self.config.history_tracking_enabled,"emergency_stop":self.config.emergency_stop}}
    def export_audit_trail(self,path: Optional[Path]=None):
        records=list(self._audit_trail) if self.config.history_tracking_enabled else []
        if path:
            Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(records,indent=2,sort_keys=True),encoding="utf-8")
        return records
    def reset_statistics(self, test_only: bool=False):
        if not test_only and self._influence_history: raise CSIUValidationError("reset would erase live production budget")
        with self._lock: self._total_applications=self._total_blocked=self._total_capped=0; self._max_influence_seen=0.0; self._audit_trail.clear(); self._emit("csiu.kill_switch_changed",{"reset":"statistics"})

_csiu_enforcer=None; _enforcer_lock=threading.Lock()
def get_csiu_enforcer(config: Optional[CSIUEnforcementConfig]=None)->CSIUEnforcement:
    global _csiu_enforcer
    with _enforcer_lock:
        if _csiu_enforcer is None: _csiu_enforcer=CSIUEnforcement(config)
        elif config is not None and config!=_csiu_enforcer.config: raise CSIUValidationError("global CSIU enforcer configuration mismatch")
        return _csiu_enforcer
def reset_csiu_enforcer(test_only: bool=True):
    global _csiu_enforcer
    with _enforcer_lock:
        if _csiu_enforcer and not test_only and _csiu_enforcer._influence_history: raise CSIUValidationError("cannot reset live CSIU singleton")
        if _csiu_enforcer: _csiu_enforcer.close()
        _csiu_enforcer=None
