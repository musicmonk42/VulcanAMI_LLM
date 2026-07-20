"""Immutable episode snapshot bundle contracts and lease ownership."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
from typing import Callable, Protocol, Sequence
from uuid import uuid4

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_VERSION = "snapshot-bundle.v1"
MAX_EPISODE_LIFETIME = timedelta(hours=6)

class SnapshotLease(Protocol):
    def close(self) -> object: ...

Clock = Callable[[], datetime]

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def _digest(value: object) -> str:
    return sha256(_canon(value)).hexdigest()

def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)

def _text(value: str, name: str, max_len: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid {name}")
    return value

@dataclass(frozen=True)
class SnapshotRef:
    kind: str
    digest: str
    schema_version: str
    owner: str
    revision: str
    acquired_at: datetime
    valid_from: datetime
    valid_until: datetime
    release_id: str

    def __post_init__(self) -> None:
        _text(self.kind, "kind", 64)
        _text(self.schema_version, "schema_version", 64)
        _text(self.owner, "owner", 128)
        _text(self.revision, "revision", 128)
        _text(self.release_id, "release_id", 128)
        if not _HEX64.fullmatch(self.digest):
            raise ValueError("snapshot digest must be a full sha256 hex digest")
        acquired = _utc(self.acquired_at, "acquired_at")
        start = _utc(self.valid_from, "valid_from")
        end = _utc(self.valid_until, "valid_until")
        if end <= start:
            raise ValueError("snapshot validity window is empty")
        if not (start <= acquired <= end):
            raise ValueError("snapshot acquisition time outside validity window")
        object.__setattr__(self, "acquired_at", acquired)
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_until", end)

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "digest": self.digest, "schema_version": self.schema_version, "owner": self.owner, "revision": self.revision, "acquired_at": self.acquired_at.isoformat(), "valid_from": self.valid_from.isoformat(), "valid_until": self.valid_until.isoformat(), "release_id": self.release_id}

@dataclass(frozen=True)
class SnapshotBundle:
    episode_id: str
    bundle_id: str
    acquired_at: datetime
    expires_at: datetime
    world: SnapshotRef
    self_state: SnapshotRef
    social: SnapshotRef
    normative: SnapshotRef
    domain: SnapshotRef
    memory: SnapshotRef
    capability: SnapshotRef
    csiu: SnapshotRef
    alignment: SnapshotRef
    leases: tuple[SnapshotLease, ...] = field(default_factory=tuple, repr=False, compare=False)
    released: bool = field(default=False, init=False, compare=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        acquired = _utc(self.acquired_at, "acquired_at")
        expires = _utc(self.expires_at, "expires_at")
        if expires <= acquired or expires - acquired > MAX_EPISODE_LIFETIME:
            raise ValueError("episode snapshot lease exceeds bounded lifetime")
        refs = self.refs()
        if tuple(ref.kind for ref in refs) != ("world", "self", "social", "normative", "domain", "memory", "capability", "csiu", "alignment"):
            raise ValueError("snapshot refs are bound to the wrong authority slot")
        for ref in refs:
            if not (ref.valid_from <= acquired <= ref.valid_until) or expires > ref.valid_until:
                raise ValueError(f"{ref.kind} snapshot expires before episode lease")
        object.__setattr__(self, "acquired_at", acquired)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "leases", tuple(self.leases))
        object.__setattr__(self, "digest", _digest(self.to_json(include_digest=False)))

    def refs(self) -> tuple[SnapshotRef, ...]:
        return (self.world, self.self_state, self.social, self.normative, self.domain, self.memory, self.capability, self.csiu, self.alignment)

    def ref_digests(self) -> tuple[str, ...]:
        return tuple(ref.digest for ref in self.refs())

    def validate_active(self, now: datetime) -> None:
        if self.released:
            raise RuntimeError("snapshot bundle already released")
        if _utc(now, "now") >= self.expires_at:
            raise RuntimeError("snapshot bundle expired")

    def close(self) -> None:
        if self.released:
            return
        object.__setattr__(self, "released", True)
        first: BaseException | None = None
        for lease in reversed(self.leases):
            try:
                lease.close()
            except BaseException as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first

    def to_json(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {"schema_version": SCHEMA_VERSION, "episode_id": self.episode_id, "bundle_id": self.bundle_id, "acquired_at": self.acquired_at.isoformat(), "expires_at": self.expires_at.isoformat(), "refs": [r.to_json() for r in self.refs()]}
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def bundle_ref(self):
        from vulcan.microkernel.episode import SnapshotBundleRef
        return SnapshotBundleRef(self.bundle_id, self.digest)

class SnapshotProvider(Protocol):
    def lease_snapshot(self, *, kind: str, episode_id: str, acquired_at: datetime, expires_at: datetime) -> tuple[SnapshotRef, SnapshotLease | None]: ...

def default_snapshot_ref(kind: str, owner: str, revision: object, payload: object, *, acquired_at: datetime, expires_at: datetime, release_id: str) -> SnapshotRef:
    digest = _digest({"kind": kind, "owner": owner, "revision": str(revision), "payload": payload})
    return SnapshotRef(kind, digest, "opaque-state.v1", owner, str(revision), acquired_at, acquired_at, expires_at, release_id)

class AttributeSnapshotProvider:
    def __init__(self, owner: object, *, owner_name: str):
        self.owner = owner
        self.owner_name = owner_name
    def lease_snapshot(self, *, kind: str, episode_id: str, acquired_at: datetime, expires_at: datetime):
        lease_fn = getattr(self.owner, "lease", None)
        lease = lease_fn() if callable(lease_fn) else None
        target = lease if lease is not None else self.owner
        digest = getattr(target, f"{kind}_snapshot_id", None) or getattr(target, "policy_digest", None) or getattr(target, "digest", None) or getattr(target, "snapshot_id", None)
        revision = getattr(target, "revision", None) or getattr(target, "version", None) or "0"
        if isinstance(digest, str) and _HEX64.fullmatch(digest):
            ref = SnapshotRef(kind, digest, "opaque-state.v1", self.owner_name, str(revision), acquired_at, acquired_at, expires_at, f"lease:{episode_id}")
        else:
            ref = default_snapshot_ref(kind, self.owner_name, revision, repr(digest), acquired_at=acquired_at, expires_at=expires_at, release_id=f"lease:{episode_id}")
        return ref, lease

def construct_snapshot_bundle(*, episode_id: str, providers: Sequence[SnapshotProvider], clock: Clock = utc_now, lifetime: timedelta = MAX_EPISODE_LIFETIME) -> SnapshotBundle:
    if len(providers) != 9:
        raise ValueError("exactly nine state authority providers are required")
    acquired = _utc(clock(), "acquired_at")
    expires = acquired + lifetime
    refs: list[SnapshotRef] = []
    leases: list[SnapshotLease] = []
    kinds = ("world", "self", "social", "normative", "domain", "memory", "capability", "csiu", "alignment")
    try:
        for kind, provider in zip(kinds, providers, strict=True):
            ref, lease = provider.lease_snapshot(kind=kind, episode_id=episode_id, acquired_at=acquired, expires_at=expires)
            refs.append(ref)
            if lease is not None:
                leases.append(lease)
        return SnapshotBundle(episode_id, str(uuid4()), acquired, expires, *refs, leases=tuple(leases))
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise

def require_bundle_snapshot(bundle: SnapshotBundle, snapshot_digest: str, *, now: datetime, transition_event: bool = False) -> None:
    bundle.validate_active(now)
    if snapshot_digest != bundle.digest and snapshot_digest not in bundle.ref_digests():
        if not transition_event:
            raise RuntimeError("mixed snapshot versions require explicit transition/rebase event")
