"""Canonical durable authority for a deliberately small preference surface.

This module has no vector, graph, transcript, or model-memory fallback.  It is
an exact-key SQLite store whose authoritative ``memory_heads`` rows prevent a
missing historical revision from becoming effective after a restore or purge.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol
from vulcan.memory.outbox import MemoryOutboxFailpoint, NoopMemoryOutboxFailpoint, deliver_idempotently, deterministic_event_id, payload_digest
from uuid import uuid4

SCHEMA_VERSION = "governed-memory/4"
POLICY_VERSION = "memory-policy/2"
MAX_RESULTS = 20

@dataclass(frozen=True)
class MemoryRuntimeConfig:
    """Validated deployment input; environment parsing is isolated here."""
    enabled: bool
    path: Path | None = None
    durable_root: Path | None = None
    replicas: int = 1
    backend: str = "sqlite"
    policy_version: str = POLICY_VERSION

    @classmethod
    def from_environment(cls) -> "MemoryRuntimeConfig":
        enabled = os.getenv("VULCAN_MEMORY_ENABLED", "0") == "1"
        if not enabled:
            return cls(False)
        path, root = os.getenv("VULCAN_MEMORY_SQLITE_PATH"), os.getenv("VULCAN_MEMORY_DURABLE_ROOT")
        if not path or not root:
            raise RuntimeError("durable memory requires VULCAN_MEMORY_SQLITE_PATH and VULCAN_MEMORY_DURABLE_ROOT")
        try:
            replicas = int(os.getenv("VULCAN_RUNTIME_REPLICAS", "1"))
        except ValueError as exc:
            raise RuntimeError("invalid durable-memory replica count") from exc
        return cls(True, Path(path), Path(root), replicas, os.getenv("VULCAN_MEMORY_BACKEND", "sqlite"), os.getenv("VULCAN_MEMORY_POLICY_VERSION", POLICY_VERSION)).validated()

    def validated(self) -> "MemoryRuntimeConfig":
        if not self.enabled: return self
        if self.backend != "sqlite" or self.replicas != 1 or self.policy_version != POLICY_VERSION:
            raise RuntimeError("unsupported governed-memory topology or policy")
        if self.path is None or self.durable_root is None or not self.path.is_absolute() or not self.durable_root.is_absolute():
            raise RuntimeError("governed-memory paths must be absolute")
        root = self.durable_root.resolve(strict=True)
        if root in {Path("/tmp"), Path.home(), Path.cwd()} or root.is_symlink() or not root.is_dir():
            raise RuntimeError("unsafe governed-memory durable root")
        candidate = self.path.resolve(strict=False)
        if root not in (candidate, *candidate.parents) or (candidate.exists() and not candidate.is_file()):
            raise RuntimeError("governed-memory database escapes durable root")
        return self


class AuditPort(Protocol):
    owner_id: str
    def append(self, event_type: str, data: dict[str, object]): ...
    def readiness(self) -> object: ...

@dataclass(frozen=True)
class BorrowedAudit:
    owner_id: str
    port: AuditPort

    @classmethod
    def bind(cls, audit: AuditPort) -> "BorrowedAudit":
        owner_id = getattr(audit, "owner_id", None)
        if not isinstance(owner_id, str) or not owner_id.startswith("audit:"):
            raise RuntimeError("canonical memory requires a canonical audit owner")
        if not callable(getattr(audit, "append", None)) or not callable(getattr(audit, "readiness", None)):
            raise RuntimeError("canonical memory requires an appendable audit owner")
        return cls(owner_id, audit)

    def readiness(self) -> None:
        if getattr(self.port, "_closed", False):
            raise RuntimeError("canonical memory audit owner is closed")
        self.port.readiness()

    def append(self, event_type: str, data: dict[str, object]):
        return self.port.append(event_type, data)

class MemoryKind(str, Enum): EXPLICIT_PREFERENCE = "explicit_preference"
class MemoryLifecycle(str, Enum): ACTIVE="active"; SUPERSEDED="superseded"; TOMBSTONED="tombstoned"; PURGED="purged"
class MemoryOperation(str, Enum): CREATE="create"; READ="read"; CORRECT="correct"; FORGET="forget"; LIST="list"; EXPORT="export"; MIGRATE="migrate"; REVOKE="revoke"
class MemoryReason(str, Enum):
    COMMITTED="committed"; CONFLICT="conflict"; NOT_FOUND="not_found"; UNAUTHORIZED="unauthorized"; MEMORY_DISABLED="memory_disabled"; POLICY_REJECTED="policy_rejected"; EMPTY="empty"; IDEMPOTENCY_CONFLICT="idempotency_conflict"
class PolicyDecision(str, Enum): ALLOW="allow"; REJECT="reject"
class DeletionState(str, Enum): REQUESTED="requested"; TOMBSTONED="tombstoned"; LOGICALLY_REDACTED="logically_redacted"; REJECTED="rejected"

# Closed, typed, non-free-form product values.  ``color`` remains only for the
# pre-existing supported test/product surface; it is not arbitrary text.
_PREFERENCES: dict[str, frozenset[str]] = {
    "locale": frozenset({"en", "en-us", "en-gb", "fr", "de", "es"}),
    "response_style": frozenset({"concise", "balanced", "detailed"}),
    "unit_system": frozenset({"metric", "imperial"}),
    "compatibility_color": frozenset({"blue", "red", "green"}),
    "color": frozenset({"blue", "red", "green"}),  # legacy alias for compatibility_color
}

def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.replace("-", "").replace("_", "").isalnum()
def _time(clock: Callable[[], datetime]) -> str:
    value=clock()
    if value.tzinfo is None or value.utcoffset() is None: raise ValueError("memory clock must return UTC-aware time")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def _digest(values: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
def _normal(key: str, value: str) -> tuple[str, str]:
    if not isinstance(key, str) or not isinstance(value, str) or any(ord(c)<32 for c in key+value): raise ValueError("invalid preference text")
    key, value = key.strip().lower(), value.strip().lower()
    forbidden=("password","passwd","secret","token","api_key","apikey","credential","jwt","bearer")
    if any(x in key or x in value for x in forbidden): raise ValueError("credential-like memory rejected")
    if key=="color": key="compatibility_color"
    if key not in _PREFERENCES or value not in _PREFERENCES[key]: raise ValueError("unsupported typed preference")
    return key, value

@dataclass(frozen=True)
class MemoryActorContext:
    """Server-owned trusted principal binding; never decoded from wire data."""
    tenant_id: str; subject_id: str; actor_id: str; purpose: str = "personalization"; request_id: str = "system"
    def __post_init__(self) -> None:
        if not all(_identifier(x) for x in (self.tenant_id,self.subject_id,self.actor_id,self.purpose,self.request_id)): raise ValueError("invalid trusted memory actor")
MemoryActor = MemoryActorContext # compatibility name; construction remains domain-only.

@dataclass(frozen=True)
class MemoryWriteProposal:
    kind: MemoryKind; namespace: str; key: str; value: str; idempotency_key: str; case_id: str|None=None; consent_reference: str="consent-explicit-v1"; lawful_basis: str="consent"; retention_rule: str="preference-365d"; source_provenance: str="direct-subject"; access_classification: str="personal-confidential"
    def __post_init__(self) -> None:
        if self.kind is not MemoryKind.EXPLICIT_PREFERENCE or self.namespace != "profile" or not _identifier(self.idempotency_key): raise ValueError("invalid memory proposal")
        key,value=_normal(self.key,self.value); object.__setattr__(self,"key",key); object.__setattr__(self,"value",value)
        for field in ("consent_reference","lawful_basis","retention_rule","source_provenance","access_classification"):
            if not _identifier(getattr(self, field)): raise ValueError("invalid memory governance metadata")
        if self.lawful_basis != "consent" or self.retention_rule != "preference-365d" or self.source_provenance != "direct-subject" or self.access_classification not in {"personal-confidential","personal-restricted"}: raise ValueError("unsupported memory governance metadata")
        if self.case_id is not None and not _identifier(self.case_id): raise ValueError("invalid case reference")
@dataclass(frozen=True)
class MemoryReadRequest:
    actor: MemoryActorContext; namespace: str; query: str; maximum_results: int=5
    def __post_init__(self) -> None:
        if self.namespace != "profile" or (self.query if self.query != "color" else "compatibility_color") not in _PREFERENCES or not 1<=self.maximum_results<=MAX_RESULTS: raise ValueError("memory reads require an allowlisted exact key")
        if self.query == "color": object.__setattr__(self, "query", "compatibility_color")
@dataclass(frozen=True)
class MemoryRecord:
    record_id:str; revision:int; tenant_id:str; subject_id:str; actor_id:str; purpose:str; namespace:str; key:str; value:str|None; kind:MemoryKind; lifecycle:MemoryLifecycle; policy_version:str; created_at:str; expires_at:str; deletion_epoch:int; digest:str; consent_reference:str; lawful_basis:str; owner_id:str; source_provenance:str; retention_rule:str; access_classification:str; supersedes:str|None=None
@dataclass(frozen=True)
class DeletionReceipt:
    operation_id:str; state:DeletionState; record_id:str; revision:int; deletion_epoch:int; policy_version:str; required_locations:tuple[str,...]; completed_locations:tuple[str,...]; remaining_obligations:tuple[str,...]=()
@dataclass(frozen=True)
class MemoryCommitResult:
    reason:MemoryReason; record:MemoryRecord|None=None; reconciliation_pending:bool=False; deletion_receipt:DeletionReceipt|None=None
@dataclass(frozen=True)
class MemoryPolicyResult:
    decision:PolicyDecision; decision_id:str; operation:MemoryOperation; version:str=POLICY_VERSION

class MemoryPolicyPort(Protocol):
    version:str
    def decide(self, operation:MemoryOperation, actor:MemoryActorContext, proposal:MemoryWriteProposal|None=None)->MemoryPolicyResult: ...
    def retention(self, actor:MemoryActorContext, proposal:MemoryWriteProposal)->timedelta: ...
class DefaultMemoryPolicy:
    version=POLICY_VERSION
    def decide(self, operation, actor, proposal=None):
        allowed=actor.purpose=="personalization" and operation in {MemoryOperation.CREATE,MemoryOperation.READ,MemoryOperation.CORRECT,MemoryOperation.FORGET,MemoryOperation.LIST,MemoryOperation.EXPORT,MemoryOperation.REVOKE}
        if proposal is not None:
            try: _normal(proposal.key,proposal.value)
            except ValueError: allowed=False
        return MemoryPolicyResult(PolicyDecision.ALLOW if allowed else PolicyDecision.REJECT, f"{self.version}:{operation.value}", operation)
    def retention(self, actor, proposal): return timedelta(days=365)

class GovernedMemoryPort(Protocol):
    def remember(self, actor:MemoryActorContext, proposal:MemoryWriteProposal)->MemoryCommitResult: ...
    def retrieve(self, request:MemoryReadRequest)->tuple[MemoryRecord,...]: ...
    def correct(self, actor:MemoryActorContext, record_id:str, base_revision:int, proposal:MemoryWriteProposal)->MemoryCommitResult: ...
    def forget(self, actor:MemoryActorContext, record_id:str, revision:int, idempotency_key:str|None=None)->MemoryCommitResult: ...
    def subject_access(self, actor:MemoryActorContext)->tuple[MemoryRecord,...]: ...
    def export_subject(self, actor:MemoryActorContext)->tuple[MemoryRecord,...]: ...
    def revoke_consent(self, actor:MemoryActorContext, consent_reference:str, idempotency_key:str)->tuple[DeletionReceipt,...]: ...
    def readiness(self)->None: ...
    def close(self)->None: ...
    def capabilities(self)->tuple[str,...]: ...

class SQLiteMemoryRepository:
    """Serialized SQLite repository with immutable revisions and authoritative heads."""
    def __init__(self,path:str,*,policy:MemoryPolicyPort|None=None,clock:Callable[[],datetime]|None=None, durable_root: str | None = None, audit: BorrowedAudit | AuditPort | None = None, failpoint: MemoryOutboxFailpoint | None = None)->None:
        if not path or path==":memory:": raise ValueError("durable memory requires an explicit filesystem path")
        raw_path=Path(path)
        if raw_path.is_symlink() or Path(str(raw_path)+".lock").is_symlink(): raise RuntimeError("governed-memory symlink path rejected")
        self._path=raw_path.resolve(); self._failpoint=failpoint or NoopMemoryOutboxFailpoint();
        if durable_root is not None:
            root=Path(durable_root).resolve(strict=True)
            if root not in (self._path, *self._path.parents): raise RuntimeError("memory database is outside durable root")
        self._path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);
        if self._path.exists() and not self._path.is_file(): raise RuntimeError("memory database is not a regular file")
        self._ownership=open(str(self._path)+".lock", "a+", encoding="utf-8")
        try: fcntl.flock(self._ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: self._ownership.close(); raise RuntimeError("governed-memory writer is already owned") from exc
        self._lock=threading.RLock(); self._closed=False; self._policy=policy or DefaultMemoryPolicy(); self._clock=clock or (lambda:datetime.now(timezone.utc)); self._owner_id=f"memory:{self._path}"; self._audit=BorrowedAudit.bind(audit) if audit is not None and not isinstance(audit, BorrowedAudit) else audit
        try:
            self._db=sqlite3.connect(str(self._path),check_same_thread=False,isolation_level=None); self._db.execute("PRAGMA foreign_keys=ON"); self._db.execute("PRAGMA busy_timeout=5000"); self._db.execute("PRAGMA journal_mode=WAL"); self._migrate(); self.readiness()
        except Exception:
            if hasattr(self,"_db"):
                try: self._db.close()
                except Exception: pass
            fcntl.flock(self._ownership.fileno(), fcntl.LOCK_UN); self._ownership.close()
            raise
    def _migrate(self)->None:
        with self._lock:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_schema(version TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS memory_revisions(record_id TEXT NOT NULL,revision INTEGER NOT NULL,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,actor_id TEXT NOT NULL,purpose TEXT NOT NULL,namespace TEXT NOT NULL,key_name TEXT NOT NULL,value TEXT,kind TEXT NOT NULL CHECK(kind='explicit_preference'),lifecycle TEXT NOT NULL CHECK(lifecycle IN ('active','superseded','tombstoned','purged')),policy_version TEXT NOT NULL,created_at TEXT NOT NULL,expires_at TEXT NOT NULL,deletion_epoch INTEGER NOT NULL CHECK(deletion_epoch>=0),digest TEXT NOT NULL,consent_reference TEXT NOT NULL,lawful_basis TEXT NOT NULL,owner_id TEXT NOT NULL,source_provenance TEXT NOT NULL,retention_rule TEXT NOT NULL,access_classification TEXT NOT NULL,supersedes TEXT,PRIMARY KEY(record_id,revision));
            CREATE TABLE IF NOT EXISTS memory_heads(record_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,purpose TEXT NOT NULL,namespace TEXT NOT NULL,key_name TEXT NOT NULL,current_revision INTEGER NOT NULL,current_lifecycle TEXT NOT NULL CHECK(current_lifecycle IN ('active','tombstoned','purged')),deletion_epoch INTEGER NOT NULL,UNIQUE(tenant_id,subject_id,purpose,namespace,key_name),FOREIGN KEY(record_id,current_revision) REFERENCES memory_revisions(record_id,revision));
            CREATE TABLE IF NOT EXISTS memory_idempotency(tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,idempotency_key TEXT NOT NULL,operation TEXT NOT NULL,request_digest TEXT NOT NULL,record_id TEXT,PRIMARY KEY(tenant_id,subject_id,idempotency_key));
            CREATE TABLE IF NOT EXISTS memory_journal(sequence INTEGER PRIMARY KEY AUTOINCREMENT,operation TEXT NOT NULL,record_id TEXT NOT NULL,revision INTEGER NOT NULL,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,request_digest TEXT NOT NULL,committed_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memory_audit_outbox(operation_id TEXT PRIMARY KEY,event_type TEXT NOT NULL,operation TEXT NOT NULL,record_id TEXT NOT NULL,revision INTEGER NOT NULL,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,purpose TEXT NOT NULL,namespace TEXT NOT NULL,key_name TEXT NOT NULL,prior_record_digest TEXT,new_record_digest TEXT NOT NULL,deletion_epoch INTEGER NOT NULL,request_digest TEXT NOT NULL,payload_digest TEXT NOT NULL,delivered_at TEXT);
            """)
            versions=self._db.execute("SELECT version FROM memory_schema").fetchall()
            if versions == [("governed-memory/3",)]:
                cols={r[1] for r in self._db.execute("PRAGMA table_info(memory_revisions)")}
                additions=(
                    ("consent_reference", "TEXT NOT NULL DEFAULT 'consent-explicit-v1'"),
                    ("lawful_basis", "TEXT NOT NULL DEFAULT 'consent'"),
                    ("owner_id", "TEXT NOT NULL DEFAULT 'owner-migrated'"),
                    ("source_provenance", "TEXT NOT NULL DEFAULT 'direct-subject'"),
                    ("retention_rule", "TEXT NOT NULL DEFAULT 'preference-365d'"),
                    ("access_classification", "TEXT NOT NULL DEFAULT 'personal-confidential'"),
                )
                for name, ddl in additions:
                    if name not in cols:
                        self._db.execute(f"ALTER TABLE memory_revisions ADD COLUMN {name} {ddl}")
                self._db.execute("UPDATE memory_schema SET version=?",(SCHEMA_VERSION,))
            if not versions: self._db.execute("INSERT INTO memory_schema VALUES(?)",(SCHEMA_VERSION,))
            elif versions == [("governed-memory/2",)]:
                cols={r[1] for r in self._db.execute("PRAGMA table_info(memory_audit_outbox)")}
                if "audit_complete" not in cols: raise RuntimeError("unsupported governed-memory schema version")
                self._db.execute("ALTER TABLE memory_audit_outbox RENAME TO memory_audit_outbox_v2")
                self._db.execute("CREATE TABLE memory_audit_outbox(operation_id TEXT PRIMARY KEY,event_type TEXT NOT NULL,operation TEXT NOT NULL,record_id TEXT NOT NULL,revision INTEGER NOT NULL,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,purpose TEXT NOT NULL,namespace TEXT NOT NULL,key_name TEXT NOT NULL,prior_record_digest TEXT,new_record_digest TEXT NOT NULL,deletion_epoch INTEGER NOT NULL,request_digest TEXT NOT NULL,payload_digest TEXT NOT NULL,delivered_at TEXT)")
                old_rows=self._db.execute("SELECT operation_id,event_type,operation,record_id,revision,tenant_id,subject_id,purpose,namespace,key_name,prior_record_digest,new_record_digest,deletion_epoch,request_digest,audit_complete FROM memory_audit_outbox_v2").fetchall()
                for old in old_rows:
                    payload=self._audit_payload(old[1], old[2], old)
                    event_id=deterministic_event_id(request_digest=old[13], record_id=old[3], revision=old[4], event_type=old[1])
                    self._db.execute("INSERT INTO memory_audit_outbox VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(event_id,*old[1:14],payload_digest(payload),_time(self._clock) if old[14] else None))
                self._db.execute("DROP TABLE memory_audit_outbox_v2")
                self._db.execute("UPDATE memory_schema SET version=?",(SCHEMA_VERSION,))
            elif versions != [(SCHEMA_VERSION,)]: raise RuntimeError("unsupported governed-memory schema version")
    def readiness(self)->None:
        with self._lock:
            if self._closed: raise RuntimeError("memory repository is closed")
            if self._audit is None: raise RuntimeError("canonical memory requires canonical audit")
            self._audit.readiness()
            if self._db.execute("PRAGMA integrity_check").fetchone() != ("ok",): raise RuntimeError("memory repository integrity check failed")
            expected={
                "memory_schema":{"version"},
                "memory_revisions":{"record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key_name","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","consent_reference","lawful_basis","owner_id","source_provenance","retention_rule","access_classification","supersedes"},
                "memory_heads":{"record_id","tenant_id","subject_id","purpose","namespace","key_name","current_revision","current_lifecycle","deletion_epoch"},
                "memory_idempotency":{"tenant_id","subject_id","idempotency_key","operation","request_digest","record_id"},
                "memory_journal":{"sequence","operation","record_id","revision","tenant_id","subject_id","request_digest","committed_at"},
                "memory_audit_outbox":{"operation_id","event_type","operation","record_id","revision","tenant_id","subject_id","purpose","namespace","key_name","prior_record_digest","new_record_digest","deletion_epoch","request_digest","payload_digest","delivered_at"},
            }
            tables={r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables != set(expected): raise RuntimeError("memory schema table verification failed")
            for table, cols in expected.items():
                got={r[1] for r in self._db.execute(f"PRAGMA table_info({table})")}
                if got != cols: raise RuntimeError("memory schema column verification failed")
            if self._db.execute("SELECT version FROM memory_schema").fetchone() != (SCHEMA_VERSION,): raise RuntimeError("memory schema verification failed")
            rows=self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,lifecycle,policy_version,created_at,expires_at,deletion_epoch,digest,consent_reference,lawful_basis,owner_id,source_provenance,retention_rule,access_classification,supersedes FROM memory_revisions ORDER BY record_id,revision").fetchall()
            groups={}
            for r in rows:
                rid,rev,tenant,subject,actor,purpose,namespace,key,value,kind,lifecycle,policy,created,expires,epoch,digest,consent,lawful,owner,source,retention_rule,access,sup=r
                if not all(_identifier(x) for x in (rid,tenant,subject,actor,purpose,namespace,key,consent,lawful,owner,source,retention_rule,access)) or policy != self._policy.version: raise RuntimeError("memory identity corruption fails closed")
                if namespace!="profile" or key not in _PREFERENCES or kind!=MemoryKind.EXPLICIT_PREFERENCE.value: raise RuntimeError("memory policy surface corruption fails closed")
                if type(rev) is not int or type(epoch) is not int or rev<1 or epoch<0: raise RuntimeError("memory revision corruption fails closed")
                c=datetime.fromisoformat(created.replace("Z","+00:00")); e=datetime.fromisoformat(expires.replace("Z","+00:00"))
                if c.tzinfo is None or e.tzinfo is None or e<c: raise RuntimeError("memory timestamp corruption fails closed")
                if lifecycle in (MemoryLifecycle.TOMBSTONED.value, MemoryLifecycle.PURGED.value) and value is not None: raise RuntimeError("memory tombstone plaintext fails closed")
                data={"record_id":rid,"revision":rev,"tenant_id":tenant,"subject_id":subject,"actor_id":actor,"purpose":purpose,"namespace":namespace,"key":key,"value":value,"kind":kind,"lifecycle":lifecycle,"policy_version":policy,"created_at":created,"expires_at":expires,"deletion_epoch":epoch,"consent_reference":consent,"lawful_basis":lawful,"owner_id":owner,"source_provenance":source,"retention_rule":retention_rule,"access_classification":access,"supersedes":sup}
                if _digest(data)!=digest: raise RuntimeError("memory digest corruption fails closed")
                groups.setdefault(rid,[]).append(r)
            for rid, rs in groups.items():
                ident=None; prev_epoch=0
                for i,r in enumerate(rs, start=1):
                    if r[1]!=i: raise RuntimeError("memory revision gap fails closed")
                    cur_ident=(r[2],r[3],r[5],r[6],r[7],r[9],r[18])
                    if ident is None: ident=cur_ident
                    elif cur_ident!=ident: raise RuntimeError("memory identity mutation fails closed")
                    expected_sup=None if i==1 else f"{rid}:{i-1}"
                    if r[22]!=expected_sup: raise RuntimeError("memory supersedes chain fails closed")
                    if r[14] not in (prev_epoch, prev_epoch+1): raise RuntimeError("memory deletion epoch corruption fails closed")
                    if r[14]==prev_epoch+1 and r[10] not in (MemoryLifecycle.TOMBSTONED.value, MemoryLifecycle.PURGED.value): raise RuntimeError("memory deletion epoch changed outside deletion")
                    prev_epoch=r[14]
            heads=self._db.execute("SELECT record_id,tenant_id,subject_id,purpose,namespace,key_name,current_revision,current_lifecycle,deletion_epoch FROM memory_heads").fetchall()
            if len(heads)>len(groups): raise RuntimeError("memory head/history mismatch fails closed")
            for h in heads:
                rs=groups.get(h[0]);
                if not rs: raise RuntimeError("memory head orphan fails closed")
                last=rs[-1]
                head_key_ok = h[5] == last[7] or (last[10] in (MemoryLifecycle.TOMBSTONED.value, MemoryLifecycle.PURGED.value) and str(h[5]).startswith(last[7]+"#deleted"))
                if (h[1],h[2],h[3],h[4],h[6],h[7],h[8]) != (last[2],last[3],last[5],last[6],last[1],last[10],last[14]) or not head_key_ok: raise RuntimeError("memory head not latest fails closed")
            bad=self._db.execute("SELECT 1 FROM memory_idempotency i LEFT JOIN memory_revisions r ON i.record_id=r.record_id WHERE i.record_id IS NOT NULL AND r.record_id IS NULL LIMIT 1").fetchone()
            if bad: raise RuntimeError("memory idempotency orphan fails closed")
            pending=self._db.execute("SELECT COUNT(*) FROM memory_audit_outbox WHERE delivered_at IS NULL").fetchone()[0]
            if pending and self._audit is None: raise RuntimeError("memory audit outbox pending without audit owner")
            if pending: self.flush_outbox()
    def _record(self,row):
        data=list(row); data[9]=MemoryKind(data[9]); data[10]=MemoryLifecycle(data[10]); return MemoryRecord(*data)
    def _decision(self,op,actor,proposal=None): return self._policy.decide(op,actor,proposal)
    def _envelope(self,op,actor,target,proposal=None,base=None): return _digest({"operation":op.value,"tenant":actor.tenant_id,"subject":actor.subject_id,"actor":actor.actor_id,"purpose":actor.purpose,"target":target,"base":base,"proposal":None if proposal is None else [proposal.namespace,proposal.key,proposal.value],"policy":self._policy.version,"schema":SCHEMA_VERSION})
    def _idempotent(self,actor,key,op,digest):
        row=self._db.execute("SELECT operation,request_digest,record_id FROM memory_idempotency WHERE tenant_id=? AND subject_id=? AND idempotency_key=?",(actor.tenant_id,actor.subject_id,key)).fetchone()
        if not row:return None
        if row[0]!=op.value or row[1]!=digest:return MemoryCommitResult(MemoryReason.IDEMPOTENCY_CONFLICT)
        pending=self._db.execute("SELECT 1 FROM memory_audit_outbox WHERE operation_id=? AND delivered_at IS NULL",(digest,)).fetchone()
        if pending: self._failpoint.hit("before_db_commit");self._db.execute("COMMIT");self._failpoint.hit("after_db_commit"); self.flush_outbox(); self._db.execute("BEGIN IMMEDIATE")
        return self._current(actor,row[2]) if row[2] else MemoryCommitResult(MemoryReason.COMMITTED)
    def _current(self,actor,record_id):
        row=self._db.execute("SELECT r.record_id,r.revision,r.tenant_id,r.subject_id,r.actor_id,r.purpose,r.namespace,r.key_name,r.value,r.kind,r.lifecycle,r.policy_version,r.created_at,r.expires_at,r.deletion_epoch,r.digest,r.consent_reference,r.lawful_basis,r.owner_id,r.source_provenance,r.retention_rule,r.access_classification,r.supersedes FROM memory_heads h JOIN memory_revisions r ON r.record_id=h.record_id AND r.revision=h.current_revision WHERE h.record_id=? AND h.tenant_id=? AND h.subject_id=? AND h.purpose=?",(record_id,actor.tenant_id,actor.subject_id,actor.purpose)).fetchone()
        return MemoryCommitResult(MemoryReason.COMMITTED,self._record(row)) if row else MemoryCommitResult(MemoryReason.NOT_FOUND)
    def _insert_revision(self,data):
        self._failpoint.hit("before_revision_write")
        self._db.execute("INSERT INTO memory_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","consent_reference","lawful_basis","owner_id","source_provenance","retention_rule","access_classification","supersedes")))
        self._failpoint.hit("after_revision_write")

    def _mark_superseded(self, rid: str, revision: int) -> None:
        row=self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,policy_version,created_at,expires_at,deletion_epoch,consent_reference,lawful_basis,owner_id,source_provenance,retention_rule,access_classification,supersedes FROM memory_revisions WHERE record_id=? AND revision=?",(rid,revision)).fetchone()
        if row is None: raise RuntimeError("memory supersession target missing")
        data={"record_id":row[0],"revision":row[1],"tenant_id":row[2],"subject_id":row[3],"actor_id":row[4],"purpose":row[5],"namespace":row[6],"key":row[7],"value":row[8],"kind":row[9],"lifecycle":"superseded","policy_version":row[10],"created_at":row[11],"expires_at":row[12],"deletion_epoch":row[13],"consent_reference":row[14],"lawful_basis":row[15],"owner_id":row[16],"source_provenance":row[17],"retention_rule":row[18],"access_classification":row[19],"supersedes":row[20]}
        self._db.execute("UPDATE memory_revisions SET lifecycle='superseded', digest=? WHERE record_id=? AND revision=?",(_digest(data),rid,revision))
    def _journal(self,op,data,digest): self._failpoint.hit("before_journal_write"); self._db.execute("INSERT INTO memory_journal(operation,record_id,revision,tenant_id,subject_id,request_digest,committed_at) VALUES(?,?,?,?,?,?,?)",(op.value,data["record_id"],data["revision"],data["tenant_id"],data["subject_id"],digest,_time(self._clock))); self._failpoint.hit("after_journal_write")
    def _audit_payload(self,event_type,op,row):
        h=lambda x: hashlib.sha256(str(x).encode()).hexdigest()
        return {"transaction_id":row[0],"operation_id":row[0],"actor_digest":h(row[5]+":"+row[6]),"operation_type":op,"tenant_digest":h(row[5]),"subject_digest":h(row[6]),"purpose":row[7],"namespace":row[8],"key":row[9],"record_id":row[3],"revision":row[4],"prior_record_digest":row[10],"new_record_digest":row[11],"record_digest":row[11],"policy_identity":self._policy.version,"policy_revision":self._policy.version,"deletion_epoch":row[12],"result_category":event_type.rsplit("_",1)[-1]}
    def _audit_event(self,*a,**k): return None
    def _outbox(self,event_type,op,data,digest,prior=None):
        self._failpoint.hit("before_outbox_write")
        event_id=deterministic_event_id(request_digest=digest, record_id=data["record_id"], revision=data["revision"], event_type=event_type)
        preview=(event_id,event_type,op.value,data["record_id"],data["revision"],data["tenant_id"],data["subject_id"],data["purpose"],data["namespace"],data["key"],prior,data["digest"],data["deletion_epoch"],digest)
        payload=self._audit_payload(event_type, op.value, preview)
        self._db.execute("INSERT OR IGNORE INTO memory_audit_outbox VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",(*preview,payload_digest(payload)))
        self._failpoint.hit("after_outbox_write")
    def flush_outbox(self):
        with self._lock:
            rows=self._db.execute("SELECT operation_id,event_type,operation,record_id,revision,tenant_id,subject_id,purpose,namespace,key_name,prior_record_digest,new_record_digest,deletion_epoch,request_digest,payload_digest,delivered_at FROM memory_audit_outbox WHERE delivered_at IS NULL ORDER BY rowid").fetchall()
            for row in rows:
                if self._audit is None:
                    raise RuntimeError("canonical memory audit owner is absent")
                payload=self._audit_payload(row[1], row[2], row)
                self._failpoint.hit("before_audit_append")
                deliver_idempotently(append=self._audit.append, event_type="memory.write_prepared", payload={**payload,"result_category":"prepared"})
                deliver_idempotently(append=self._audit.append, event_type=row[1], payload=payload)
                self._failpoint.hit("after_audit_append")
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    self._failpoint.hit("before_delivery_mark")
                    self._db.execute("UPDATE memory_audit_outbox SET delivered_at=? WHERE operation_id=? AND delivered_at IS NULL",(_time(self._clock),row[0])); self._failpoint.hit("after_delivery_mark"); self._db.execute("COMMIT")
                except Exception:
                    try: self._db.execute("ROLLBACK")
                    except Exception: pass
                    raise
    def commit(self,actor,proposal):
        if self._decision(MemoryOperation.CREATE,actor,proposal).decision is PolicyDecision.REJECT:return MemoryCommitResult(MemoryReason.POLICY_REJECTED)
        digest=self._envelope(MemoryOperation.CREATE,actor,f"{proposal.namespace}:{proposal.key}",proposal)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prior=self._idempotent(actor,proposal.idempotency_key,MemoryOperation.CREATE,digest)
                if prior is not None:self._db.execute("COMMIT");return prior
                exists=self._db.execute("SELECT record_id,current_lifecycle,deletion_epoch FROM memory_heads WHERE tenant_id=? AND subject_id=? AND purpose=? AND namespace=? AND key_name=?",(actor.tenant_id,actor.subject_id,actor.purpose,proposal.namespace,proposal.key)).fetchone()
                if exists and exists[1]=="active":self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.CONFLICT)
                if exists and exists[1] in ("tombstoned","purged"):
                    self._db.execute("UPDATE memory_heads SET key_name=? WHERE record_id=?",(proposal.key+"#deleted"+str(exists[2]), exists[0]))
                now=_time(self._clock); rid="mem-"+uuid4().hex; data={"record_id":rid,"revision":1,"tenant_id":actor.tenant_id,"subject_id":actor.subject_id,"actor_id":actor.actor_id,"purpose":actor.purpose,"namespace":proposal.namespace,"key":proposal.key,"value":proposal.value,"kind":proposal.kind.value,"lifecycle":"active","policy_version":self._policy.version,"created_at":now,"expires_at":(datetime.fromisoformat(now.replace("Z","+00:00"))+self._policy.retention(actor,proposal)).isoformat().replace("+00:00","Z"),"deletion_epoch":0,"consent_reference":proposal.consent_reference,"lawful_basis":proposal.lawful_basis,"owner_id":actor.actor_id,"source_provenance":proposal.source_provenance,"retention_rule":proposal.retention_rule,"access_classification":proposal.access_classification,"supersedes":None};data["digest"]=_digest(data);self._insert_revision(data);self._failpoint.hit("before_head_write");self._db.execute("INSERT INTO memory_heads SELECT record_id,tenant_id,subject_id,purpose,namespace,key_name,revision,lifecycle,deletion_epoch FROM memory_revisions WHERE record_id=? AND revision=?",(rid,1));self._failpoint.hit("after_head_write");self._outbox("memory.write_committed", MemoryOperation.CREATE, data, digest);self._failpoint.hit("before_idempotency_write");self._db.execute("INSERT INTO memory_idempotency VALUES(?,?,?,?,?,?)",(actor.tenant_id,actor.subject_id,proposal.idempotency_key,"create",digest,rid));self._failpoint.hit("after_idempotency_write");self._journal(MemoryOperation.CREATE,data,digest);self._failpoint.hit("before_db_commit");self._db.execute("COMMIT");self._failpoint.hit("after_db_commit");self.flush_outbox();rec=self._record(tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","consent_reference","lawful_basis","owner_id","source_provenance","retention_rule","access_classification","supersedes")));return MemoryCommitResult(MemoryReason.COMMITTED,rec)
            except Exception:
                try: self._db.execute("ROLLBACK")
                except Exception: pass
                raise
    def read(self,request):
        if self._decision(MemoryOperation.READ,request.actor).decision is PolicyDecision.REJECT:return ()
        with self._lock:
            now=_time(self._clock); rows=self._db.execute("SELECT r.record_id,r.revision,r.tenant_id,r.subject_id,r.actor_id,r.purpose,r.namespace,r.key_name,r.value,r.kind,r.lifecycle,r.policy_version,r.created_at,r.expires_at,r.deletion_epoch,r.digest,r.consent_reference,r.lawful_basis,r.owner_id,r.source_provenance,r.retention_rule,r.access_classification,r.supersedes FROM memory_heads h JOIN memory_revisions r ON r.record_id=h.record_id AND r.revision=h.current_revision WHERE h.tenant_id=? AND h.subject_id=? AND h.purpose=? AND h.namespace=? AND h.key_name=? AND h.current_lifecycle='active' AND r.expires_at>? LIMIT ?",(request.actor.tenant_id,request.actor.subject_id,request.actor.purpose,request.namespace,request.query,now,request.maximum_results)).fetchall();return tuple(self._record(r) for r in rows)
    def correct(self,actor,record_id,base_revision,proposal): return self._advance(MemoryOperation.CORRECT,actor,record_id,base_revision,proposal,proposal.idempotency_key)
    def tombstone(self,actor,record_id,revision,idempotency_key=None): return self._advance(MemoryOperation.FORGET,actor,record_id,revision,None,idempotency_key or f"forget-{record_id}-{revision}-{uuid4().hex}")
    def _advance(self,op,actor,rid,base,proposal,key):
        if not _identifier(rid) or base<1:raise ValueError("invalid memory revision")
        if self._decision(op,actor,proposal).decision is PolicyDecision.REJECT:return MemoryCommitResult(MemoryReason.POLICY_REJECTED)
        digest=self._envelope(op,actor,rid,proposal,base)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prior=self._idempotent(actor,key,op,digest)
                if prior is not None:self._db.execute("COMMIT");return prior
                cur=self._current(actor,rid)
                if cur.reason is not MemoryReason.COMMITTED:self._db.execute("COMMIT");return cur
                old=cur.record
                if old is None or old.revision!=base or old.lifecycle is not MemoryLifecycle.ACTIVE:self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.CONFLICT)
                if proposal and (proposal.namespace!=old.namespace or proposal.key!=old.key):self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.CONFLICT)
                now=_time(self._clock); deletion=old.deletion_epoch+(op is MemoryOperation.FORGET); lifecycle="tombstoned" if op is MemoryOperation.FORGET else "active"; data={"record_id":rid,"revision":base+1,"tenant_id":old.tenant_id,"subject_id":old.subject_id,"actor_id":actor.actor_id,"purpose":old.purpose,"namespace":old.namespace,"key":old.key,"value":None if op is MemoryOperation.FORGET else proposal.value,"kind":old.kind.value,"lifecycle":lifecycle,"policy_version":self._policy.version,"created_at":now,"expires_at":old.expires_at,"deletion_epoch":deletion,"consent_reference":old.consent_reference if proposal is None else proposal.consent_reference,"lawful_basis":old.lawful_basis if proposal is None else proposal.lawful_basis,"owner_id":old.owner_id,"source_provenance":old.source_provenance if proposal is None else proposal.source_provenance,"retention_rule":old.retention_rule if proposal is None else proposal.retention_rule,"access_classification":old.access_classification if proposal is None else proposal.access_classification,"supersedes":f"{rid}:{base}"};data["digest"]=_digest(data);self._insert_revision(data)
                if op is MemoryOperation.CORRECT:
                    self._mark_superseded(rid, base)
                # Erase every retained payload revision in the same transaction; the head/tombstone remains.
                if op is MemoryOperation.FORGET:
                    self._db.execute("UPDATE memory_revisions SET value=NULL WHERE record_id=?",(rid,))
                    redacted=self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,lifecycle,policy_version,created_at,expires_at,deletion_epoch,consent_reference,lawful_basis,owner_id,source_provenance,retention_rule,access_classification,supersedes FROM memory_revisions WHERE record_id=?",(rid,)).fetchall()
                    for rr in redacted:
                        rd={"record_id":rr[0],"revision":rr[1],"tenant_id":rr[2],"subject_id":rr[3],"actor_id":rr[4],"purpose":rr[5],"namespace":rr[6],"key":rr[7],"value":rr[8],"kind":rr[9],"lifecycle":rr[10],"policy_version":rr[11],"created_at":rr[12],"expires_at":rr[13],"deletion_epoch":rr[14],"consent_reference":rr[15],"lawful_basis":rr[16],"owner_id":rr[17],"source_provenance":rr[18],"retention_rule":rr[19],"access_classification":rr[20],"supersedes":rr[21]}
                        self._db.execute("UPDATE memory_revisions SET digest=? WHERE record_id=? AND revision=?",(_digest(rd),rid,rr[1]))
                    data["value"]=None; data["digest"]=_digest(data)
                self._failpoint.hit("before_head_write");self._db.execute("UPDATE memory_heads SET current_revision=?,current_lifecycle=?,deletion_epoch=? WHERE record_id=?",(data["revision"],data["lifecycle"],data["deletion_epoch"],rid));self._failpoint.hit("after_head_write");self._outbox("memory.write_committed", op, data, digest, old.digest);self._failpoint.hit("before_idempotency_write");self._db.execute("INSERT INTO memory_idempotency VALUES(?,?,?,?,?,?)",(actor.tenant_id,actor.subject_id,key,op.value,digest,rid));self._failpoint.hit("after_idempotency_write");self._journal(op,data,digest);self._failpoint.hit("before_db_commit");self._db.execute("COMMIT");self._failpoint.hit("after_db_commit"); self.flush_outbox()
                rec=self._record(tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","consent_reference","lawful_basis","owner_id","source_provenance","retention_rule","access_classification","supersedes")))
                receipt=None if op is not MemoryOperation.FORGET else DeletionReceipt("del-"+uuid4().hex,DeletionState.LOGICALLY_REDACTED,rid,base+1,deletion,self._policy.version,("sqlite_payloads","canonical_head","wal_free_pages_backups_snapshots_replicas"),("sqlite_payloads","canonical_head"),("wal_free_pages_backups_snapshots_replicas",))
                return MemoryCommitResult(MemoryReason.COMMITTED,rec,False,receipt)
            except Exception:
                try: self._db.execute("ROLLBACK")
                except Exception: pass
                raise

    def subject_access(self, actor):
        if self._decision(MemoryOperation.LIST,actor).decision is PolicyDecision.REJECT:return ()
        with self._lock:
            rows=self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,lifecycle,policy_version,created_at,expires_at,deletion_epoch,digest,consent_reference,lawful_basis,owner_id,source_provenance,retention_rule,access_classification,supersedes FROM memory_revisions WHERE tenant_id=? AND subject_id=? AND purpose=? ORDER BY record_id,revision",(actor.tenant_id,actor.subject_id,actor.purpose)).fetchall()
            return tuple(self._record(r) for r in rows)
    def export_subject(self, actor):
        if self._decision(MemoryOperation.EXPORT,actor).decision is PolicyDecision.REJECT:return ()
        return self.subject_access(actor)
    def revoke_consent(self, actor, consent_reference, idempotency_key):
        if not _identifier(consent_reference) or not _identifier(idempotency_key): raise ValueError("invalid consent revocation")
        if self._decision(MemoryOperation.REVOKE,actor).decision is PolicyDecision.REJECT:return ()
        receipts=[]
        with self._lock:
            rows=self._db.execute("SELECT h.record_id,h.current_revision FROM memory_heads h JOIN memory_revisions r ON r.record_id=h.record_id AND r.revision=h.current_revision WHERE h.tenant_id=? AND h.subject_id=? AND h.purpose=? AND h.current_lifecycle='active' AND r.consent_reference=? ORDER BY h.record_id",(actor.tenant_id,actor.subject_id,actor.purpose,consent_reference)).fetchall()
        for rid, rev in rows:
            result=self.tombstone(actor,rid,rev,f"{idempotency_key}-{rid}")
            if result.deletion_receipt is not None: receipts.append(result.deletion_receipt)
        return tuple(receipts)
    def close(self):
        with self._lock:
            if not self._closed:self._db.close();fcntl.flock(self._ownership.fileno(), fcntl.LOCK_UN);self._ownership.close();self._closed=True

class GovernedMemoryService:
    def __init__(self,repository:SQLiteMemoryRepository)->None:self._repository=repository
    def remember(self,actor,proposal):return self._repository.commit(actor,proposal)
    def retrieve(self,request):return self._repository.read(request)
    def correct(self,actor,record_id,base_revision,proposal):return self._repository.correct(actor,record_id,base_revision,proposal)
    def forget(self,actor,record_id,revision,idempotency_key=None):return self._repository.tombstone(actor,record_id,revision,idempotency_key)
    def subject_access(self,actor):return self._repository.subject_access(actor)
    def export_subject(self,actor):return self._repository.export_subject(actor)
    def revoke_consent(self,actor,consent_reference,idempotency_key):return self._repository.revoke_consent(actor,consent_reference,idempotency_key)
    def readiness(self):self._repository.readiness()
    def close(self):self._repository.close()
    def capabilities(self):
        self.readiness(); return ("governed-preference-memory",)
    def diagnostics(self):
        return {"memory_owner_id": self._repository._owner_id, "audit_owner_id": self._repository._audit.owner_id if self._repository._audit else None}
class DisabledMemoryService:
    def remember(self,*args,**kwargs):return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def retrieve(self,*args,**kwargs):return ()
    def correct(self,*args,**kwargs):return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def forget(self,*args,**kwargs):return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def subject_access(self,*args,**kwargs):return ()
    def export_subject(self,*args,**kwargs):return ()
    def revoke_consent(self,*args,**kwargs):return ()
    def readiness(self):return None
    def capabilities(self):return ()
    def close(self):return None

def compose_governed_memory(config: MemoryRuntimeConfig, *, audit: AuditPort | BorrowedAudit | None = None)->GovernedMemoryPort:
    config = config.validated()
    if not config.enabled:return DisabledMemoryService()
    if config.path is None or config.durable_root is None:
        raise RuntimeError("governed-memory paths must be configured")
    if audit is None:
        raise RuntimeError("canonical memory cannot be enabled without canonical audit")
    return GovernedMemoryService(SQLiteMemoryRepository(str(config.path), durable_root=str(config.durable_root), audit=audit))
