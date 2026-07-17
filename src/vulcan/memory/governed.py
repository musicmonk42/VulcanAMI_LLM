"""The canonical, deliberately small durable-memory authority.

This module is intentionally independent of the historical memory packages.  It
stores only bounded, typed user preferences in the supported runtime surface;
the old graph/vector stores are not a fallback or a peer authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4


SCHEMA_VERSION = "governed-memory/1"
POLICY_VERSION = "memory-policy/1"
MAX_VALUE_CHARS = 512
MAX_KEY_CHARS = 64
MAX_RESULTS = 20


class MemoryKind(str, Enum):
    EXPLICIT_PREFERENCE = "explicit_preference"


class MemoryLifecycle(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    TOMBSTONED = "tombstoned"
    PURGED = "purged"
    QUARANTINED = "quarantined"


class MemoryReason(str, Enum):
    COMMITTED = "committed"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    MEMORY_DISABLED = "memory_disabled"
    POLICY_REJECTED = "policy_rejected"
    EMPTY = "empty"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"


class DeletionState(str, Enum):
    COMPLETED = "completed"
    PENDING = "pending"
    REJECTED = "rejected"


@dataclass(frozen=True)
class MemoryActor:
    tenant_id: str
    subject_id: str
    actor_id: str
    purpose: str = "personalization"

    def __post_init__(self) -> None:
        for value in (self.tenant_id, self.subject_id, self.actor_id, self.purpose):
            if not _identifier(value):
                raise ValueError("memory actor contains an invalid identifier")


@dataclass(frozen=True)
class MemoryWriteProposal:
    """Untrusted intent.  It cannot choose IDs, retention, or lifecycle."""
    kind: MemoryKind
    namespace: str
    key: str
    value: str
    idempotency_key: str
    case_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind is not MemoryKind.EXPLICIT_PREFERENCE:
            raise ValueError("unsupported memory kind")
        if not _identifier(self.namespace) or not _plain(self.key, MAX_KEY_CHARS) or not _plain(self.value, MAX_VALUE_CHARS):
            raise ValueError("invalid bounded memory proposal")
        if not _identifier(self.idempotency_key):
            raise ValueError("invalid idempotency key")
        if self.case_id is not None and not _identifier(self.case_id):
            raise ValueError("invalid case reference")


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    revision: int
    tenant_id: str
    subject_id: str
    actor_id: str
    purpose: str
    namespace: str
    key: str
    value: str
    kind: MemoryKind
    lifecycle: MemoryLifecycle
    policy_version: str
    created_at: str
    expires_at: str
    deletion_epoch: int
    digest: str
    supersedes: str | None = None


@dataclass(frozen=True)
class MemoryCommitResult:
    reason: MemoryReason
    record: MemoryRecord | None = None
    reconciliation_pending: bool = False


@dataclass(frozen=True)
class DeletionReceipt:
    operation_id: str
    state: DeletionState
    record_id: str
    revision: int
    deletion_epoch: int
    policy_version: str
    required_locations: tuple[str, ...]
    completed_locations: tuple[str, ...]
    remaining_obligations: tuple[str, ...] = ()


class MemoryPolicyPort(Protocol):
    """Server-owned policy; callers cannot set a retention or lifecycle value."""
    version: str
    def allow_write(self, actor: "MemoryActor", proposal: "MemoryWriteProposal") -> PolicyDecision: ...
    def retention(self, actor: "MemoryActor", proposal: "MemoryWriteProposal") -> timedelta: ...


class DefaultMemoryPolicy:
    """Conservative product policy: only minimized explicit preferences."""
    version = POLICY_VERSION
    def allow_write(self, actor: "MemoryActor", proposal: "MemoryWriteProposal") -> PolicyDecision:
        if actor.purpose != "personalization" or proposal.kind is not MemoryKind.EXPLICIT_PREFERENCE:
            return PolicyDecision.REJECT
        forbidden = ("password", "secret", "token", "authorization", "credential")
        return PolicyDecision.REJECT if any(word in proposal.key.lower() for word in forbidden) else PolicyDecision.ALLOW
    def retention(self, actor: "MemoryActor", proposal: "MemoryWriteProposal") -> timedelta:
        return timedelta(days=365)


@dataclass(frozen=True)
class MemoryReadRequest:
    actor: MemoryActor
    namespace: str
    query: str
    maximum_results: int = 5

    def __post_init__(self) -> None:
        if not _identifier(self.namespace) or not _plain(self.query, MAX_VALUE_CHARS):
            raise ValueError("invalid memory read request")
        if not 1 <= self.maximum_results <= MAX_RESULTS:
            raise ValueError("memory result limit out of bounds")


class MemoryRepositoryPort(Protocol):
    def commit(self, actor: MemoryActor, proposal: MemoryWriteProposal) -> MemoryCommitResult: ...
    def read(self, request: MemoryReadRequest) -> tuple[MemoryRecord, ...]: ...
    def tombstone(self, actor: MemoryActor, record_id: str, revision: int) -> MemoryCommitResult: ...
    def correct(self, actor: MemoryActor, record_id: str, base_revision: int, proposal: MemoryWriteProposal) -> MemoryCommitResult: ...
    def readiness(self) -> None: ...
    def close(self) -> None: ...


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.replace("-", "").replace("_", "").isalnum()


def _plain(value: object, limit: int) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= limit and "\x00" not in value


def _canonical_digest(values: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


class SQLiteMemoryRepository:
    """Single-writer SQLite source of truth with journal and transactional outbox."""
    def __init__(self, path: str, *, policy: MemoryPolicyPort | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        if not path or path == ":memory:":
            raise ValueError("durable memory requires an explicit filesystem path")
        self._path = Path(path).resolve()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._policy = policy or DefaultMemoryPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._db = sqlite3.connect(str(self._path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._migrate()
        self.readiness()

    def _migrate(self) -> None:
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS memory_records (
          record_id TEXT NOT NULL, revision INTEGER NOT NULL, tenant_id TEXT NOT NULL,
          subject_id TEXT NOT NULL, actor_id TEXT NOT NULL, purpose TEXT NOT NULL,
          namespace TEXT NOT NULL, key_name TEXT NOT NULL, value TEXT NOT NULL,
          kind TEXT NOT NULL, lifecycle TEXT NOT NULL, policy_version TEXT NOT NULL,
          created_at TEXT NOT NULL, expires_at TEXT NOT NULL, deletion_epoch INTEGER NOT NULL,
          digest TEXT NOT NULL, supersedes TEXT, PRIMARY KEY(record_id, revision),
          CHECK(revision > 0), CHECK(deletion_epoch >= 0)
        );
        CREATE INDEX IF NOT EXISTS active_scope ON memory_records(tenant_id, subject_id, namespace, lifecycle, expires_at);
        CREATE TABLE IF NOT EXISTS memory_idempotency (tenant_id TEXT NOT NULL, subject_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, record_id TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(tenant_id, subject_id, idempotency_key));
        CREATE TABLE IF NOT EXISTS memory_journal (sequence INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, record_id TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, committed_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS memory_outbox (sequence INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, revision INTEGER NOT NULL, deletion_epoch INTEGER NOT NULL, operation TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
        """)

    def readiness(self) -> None:
        if self._closed:
            raise RuntimeError("memory repository is closed")
        integrity = self._db.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError("memory repository integrity check failed")

    def _row(self, row: tuple[object, ...]) -> MemoryRecord:
        normalized = list(row)
        normalized[9] = MemoryKind(normalized[9])
        normalized[10] = MemoryLifecycle(normalized[10])
        record = MemoryRecord(*normalized)
        expected = _canonical_digest({k: v.value if isinstance(v, Enum) else v for k, v in asdict(record).items() if k != "digest"})
        if record.digest != expected:
            raise RuntimeError("memory record digest mismatch")
        return record

    def commit(self, actor: MemoryActor, proposal: MemoryWriteProposal) -> MemoryCommitResult:
        with self._lock:
            if self._closed: raise RuntimeError("memory repository is closed")
            if self._policy.allow_write(actor, proposal) is not PolicyDecision.ALLOW:
                return MemoryCommitResult(MemoryReason.POLICY_REJECTED)
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prior = self._db.execute("SELECT record_id, revision FROM memory_idempotency WHERE tenant_id=? AND subject_id=? AND idempotency_key=?", (actor.tenant_id, actor.subject_id, proposal.idempotency_key)).fetchone()
                if prior:
                    row = self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,lifecycle,policy_version,created_at,expires_at,deletion_epoch,digest,supersedes FROM memory_records WHERE record_id=? AND revision=?", prior).fetchone()
                    self._db.execute("COMMIT"); return MemoryCommitResult(MemoryReason.COMMITTED, self._row(row), True)
                now = self._clock(); record_id = f"mem-{uuid4().hex}"; created = now.isoformat().replace("+00:00", "Z"); expires = (now + self._policy.retention(actor, proposal)).isoformat().replace("+00:00", "Z")
                values: dict[str, object] = {"record_id":record_id,"revision":1,"tenant_id":actor.tenant_id,"subject_id":actor.subject_id,"actor_id":actor.actor_id,"purpose":actor.purpose,"namespace":proposal.namespace,"key":proposal.key,"value":proposal.value,"kind":proposal.kind.value,"lifecycle":MemoryLifecycle.ACTIVE.value,"policy_version":self._policy.version,"created_at":created,"expires_at":expires,"deletion_epoch":0,"supersedes":None}
                digest = _canonical_digest(values); values["digest"] = digest
                self._db.execute("INSERT INTO memory_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (values["record_id"],values["revision"],values["tenant_id"],values["subject_id"],values["actor_id"],values["purpose"],values["namespace"],values["key"],values["value"],values["kind"],values["lifecycle"],values["policy_version"],values["created_at"],values["expires_at"],values["deletion_epoch"],values["digest"],None))
                self._db.execute("INSERT INTO memory_idempotency VALUES (?,?,?,?,?)", (actor.tenant_id,actor.subject_id,proposal.idempotency_key,record_id,1))
                self._event("create", record_id, 1, digest, 0); self._db.execute("COMMIT")
                return MemoryCommitResult(MemoryReason.COMMITTED, self._row(tuple(values[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes"))), True)
            except Exception:
                self._db.execute("ROLLBACK"); raise

    def _event(self, operation: str, record_id: str, revision: int, digest: str, epoch: int) -> None:
        now = self._clock().isoformat().replace("+00:00", "Z")
        self._db.execute("INSERT INTO memory_journal(operation,record_id,revision,digest,committed_at) VALUES (?,?,?,?,?)", (operation,record_id,revision,digest,now))
        self._db.execute("INSERT INTO memory_outbox(record_id,revision,deletion_epoch,operation) VALUES (?,?,?,?)", (record_id,revision,epoch,operation))

    def read(self, request: MemoryReadRequest) -> tuple[MemoryRecord, ...]:
        if self._closed: raise RuntimeError("memory repository is closed")
        now = self._clock().isoformat().replace("+00:00", "Z")
        # Exact, scoped lexical candidate generation; canonical rows are rehydrated.
        rows = self._db.execute("SELECT r.record_id,r.revision,r.tenant_id,r.subject_id,r.actor_id,r.purpose,r.namespace,r.key_name,r.value,r.kind,r.lifecycle,r.policy_version,r.created_at,r.expires_at,r.deletion_epoch,r.digest,r.supersedes FROM memory_records r WHERE r.tenant_id=? AND r.subject_id=? AND r.purpose=? AND r.namespace=? AND r.lifecycle=? AND r.expires_at>? AND r.revision=(SELECT MAX(newer.revision) FROM memory_records newer WHERE newer.record_id=r.record_id) AND (r.key_name LIKE ? OR r.value LIKE ?) ORDER BY r.created_at DESC LIMIT ?", (request.actor.tenant_id,request.actor.subject_id,request.actor.purpose,request.namespace,MemoryLifecycle.ACTIVE.value,now,f"%{request.query}%",f"%{request.query}%",request.maximum_results)).fetchall()
        return tuple(self._row(row) for row in rows)

    def correct(self, actor: MemoryActor, record_id: str, base_revision: int,
                proposal: MemoryWriteProposal) -> MemoryCommitResult:
        """Append an immutable successor; stale revisions never overwrite data."""
        if not _identifier(record_id) or base_revision < 1:
            raise ValueError("invalid correction reference")
        if self._policy.allow_write(actor, proposal) is not PolicyDecision.ALLOW:
            return MemoryCommitResult(MemoryReason.POLICY_REJECTED)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,lifecycle,policy_version,created_at,expires_at,deletion_epoch,digest,supersedes FROM memory_records WHERE record_id=? AND revision=? AND tenant_id=? AND subject_id=?", (record_id,base_revision,actor.tenant_id,actor.subject_id)).fetchone()
                if row is None:
                    self._db.execute("COMMIT"); return MemoryCommitResult(MemoryReason.NOT_FOUND)
                old = self._row(row)
                latest = self._db.execute("SELECT MAX(revision) FROM memory_records WHERE record_id=?", (record_id,)).fetchone()[0]
                if latest != base_revision or old.lifecycle is not MemoryLifecycle.ACTIVE:
                    self._db.execute("COMMIT"); return MemoryCommitResult(MemoryReason.CONFLICT)
                now = self._clock(); created = now.isoformat().replace("+00:00", "Z")
                data: dict[str, object] = {"record_id":record_id,"revision":base_revision + 1,"tenant_id":actor.tenant_id,"subject_id":actor.subject_id,"actor_id":actor.actor_id,"purpose":actor.purpose,"namespace":proposal.namespace,"key":proposal.key,"value":proposal.value,"kind":proposal.kind.value,"lifecycle":MemoryLifecycle.ACTIVE.value,"policy_version":self._policy.version,"created_at":created,"expires_at":(now + self._policy.retention(actor, proposal)).isoformat().replace("+00:00", "Z"),"deletion_epoch":old.deletion_epoch,"supersedes":f"{record_id}:{base_revision}"}
                data["digest"] = _canonical_digest(data)
                self._db.execute("INSERT INTO memory_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes")))
                self._event("correct", record_id, base_revision + 1, data["digest"], old.deletion_epoch)
                self._db.execute("COMMIT")
                return MemoryCommitResult(MemoryReason.COMMITTED, self._row(tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes"))), True)
            except Exception:
                self._db.execute("ROLLBACK"); raise

    def tombstone(self, actor: MemoryActor, record_id: str, revision: int) -> MemoryCommitResult:
        if not _identifier(record_id) or revision < 1: raise ValueError("invalid memory revision")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT record_id,revision,tenant_id,subject_id,actor_id,purpose,namespace,key_name,value,kind,lifecycle,policy_version,created_at,expires_at,deletion_epoch,digest,supersedes FROM memory_records WHERE record_id=? AND revision=? AND tenant_id=? AND subject_id=?", (record_id,revision,actor.tenant_id,actor.subject_id)).fetchone()
                if row is None: self._db.execute("COMMIT"); return MemoryCommitResult(MemoryReason.NOT_FOUND)
                old = self._row(row)
                latest = self._db.execute("SELECT MAX(revision) FROM memory_records WHERE record_id=?", (record_id,)).fetchone()[0]
                if latest != revision or old.lifecycle is not MemoryLifecycle.ACTIVE: self._db.execute("COMMIT"); return MemoryCommitResult(MemoryReason.CONFLICT)
                data = asdict(old); data.update(revision=revision + 1, lifecycle=MemoryLifecycle.TOMBSTONED, deletion_epoch=old.deletion_epoch+1, supersedes=f"{record_id}:{revision}", created_at=self._clock().isoformat().replace("+00:00", "Z")); data["digest"] = _canonical_digest({k:v.value if isinstance(v,Enum) else v for k,v in data.items() if k != "digest"})
                self._db.execute("INSERT INTO memory_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(data[k].value if isinstance(data[k], Enum) else data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes"))); self._event("tombstone",record_id,revision + 1,data["digest"],data["deletion_epoch"]); self._db.execute("COMMIT")
                return MemoryCommitResult(MemoryReason.COMMITTED, self._row(tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes"))), True)
            except Exception: self._db.execute("ROLLBACK"); raise

    def close(self) -> None:
        with self._lock:
            if not self._closed: self._db.close(); self._closed = True


class DisabledMemoryService:
    def commit(self, actor: MemoryActor, proposal: MemoryWriteProposal) -> MemoryCommitResult: return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def read(self, request: MemoryReadRequest) -> tuple[MemoryRecord, ...]: return ()
    def tombstone(self, actor: MemoryActor, record_id: str, revision: int) -> MemoryCommitResult: return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def correct(self, actor: MemoryActor, record_id: str, base_revision: int, proposal: MemoryWriteProposal) -> MemoryCommitResult: return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def readiness(self) -> None: return None
    def close(self) -> None: return None


class GovernedMemoryService:
    """Policy boundary: only explicit, authenticated preferences are supported."""
    def __init__(self, repository: MemoryRepositoryPort) -> None: self._repository = repository
    def remember(self, actor: MemoryActor, proposal: MemoryWriteProposal) -> MemoryCommitResult: return self._repository.commit(actor, proposal)
    def retrieve(self, request: MemoryReadRequest) -> tuple[MemoryRecord, ...]: return self._repository.read(request)
    def forget(self, actor: MemoryActor, record_id: str, revision: int) -> MemoryCommitResult: return self._repository.tombstone(actor, record_id, revision)
    def correct(self, actor: MemoryActor, record_id: str, base_revision: int, proposal: MemoryWriteProposal) -> MemoryCommitResult: return self._repository.correct(actor, record_id, base_revision, proposal)
    def readiness(self) -> None: self._repository.readiness()
    def close(self) -> None: self._repository.close()


def compose_governed_memory() -> GovernedMemoryService | DisabledMemoryService:
    if os.getenv("VULCAN_MEMORY_ENABLED", "0") != "1": return DisabledMemoryService()
    path = os.getenv("VULCAN_MEMORY_SQLITE_PATH")
    if not path: raise RuntimeError("VULCAN_MEMORY_SQLITE_PATH is required when durable memory is enabled")
    # SQLite local storage has one writer process. Replica counts must be made
    # explicit rather than hoping WAL makes an unsupported topology safe.
    if os.getenv("VULCAN_RUNTIME_REPLICAS", "1") != "1": raise RuntimeError("SQLite governed memory supports exactly one runtime replica")
    return GovernedMemoryService(SQLiteMemoryRepository(path))
