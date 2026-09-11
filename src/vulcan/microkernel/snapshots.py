"""Replayable immutable episode admission context and lease ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, Protocol, Sequence

from vulcan.constitution.primitives import Digest, canonical_json, canonical_json_loads

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_VERSION = "admitted-context.v1"
MAX_EPISODE_LIFETIME = timedelta(hours=6)


class SnapshotLease(Protocol):
    def close(self) -> object: ...


Clock = Callable[[], datetime]


def utc_now():
    return datetime.now(timezone.utc)


def _digest(v):
    return Digest.of_bytes(canonical_json(v)).hex


def _utc(v, n):
    if not isinstance(v, datetime) or v.tzinfo is None:
        raise ValueError(f"{n} must be timezone-aware")
    return v.astimezone(timezone.utc)


def _text(v, n):
    if not isinstance(v, str) or not v or len(v) > 128 or any(ord(c) < 32 for c in v):
        raise ValueError(f"invalid {n}")
    return v


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

    def __post_init__(self):
        for v, n in (
            (self.kind, "kind"),
            (self.schema_version, "schema_version"),
            (self.owner, "owner"),
            (self.revision, "revision"),
            (self.release_id, "release_id"),
        ):
            _text(v, n)
        if not _HEX64.fullmatch(self.digest):
            raise ValueError("snapshot digest must be a full sha256 hex digest")
        a, s, e = (
            _utc(self.acquired_at, "acquired_at"),
            _utc(self.valid_from, "valid_from"),
            _utc(self.valid_until, "valid_until"),
        )
        if e <= s or not s <= a <= e:
            raise ValueError("invalid snapshot validity window")
        object.__setattr__(self, "acquired_at", a)
        object.__setattr__(self, "valid_from", s)
        object.__setattr__(self, "valid_until", e)

    def to_json(self):
        return {
            "kind": self.kind,
            "digest": self.digest,
            "schema_version": self.schema_version,
            "owner": self.owner,
            "revision": self.revision,
            "acquired_at": self.acquired_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "release_id": self.release_id,
        }


@dataclass(frozen=True)
class SnapshotMaterialRef(SnapshotRef):
    canonical_material: bytes
    live_view: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self.canonical_material, bytes)
            or not self.canonical_material
        ):
            raise ValueError("snapshot canonical material is required")
        value = canonical_json_loads(self.canonical_material)
        if (
            canonical_json(value) != self.canonical_material
            or _digest(value) != self.digest
        ):
            raise ValueError("snapshot material digest mismatch")

    def to_json(self):
        return {
            **super().to_json(),
            "material": canonical_json_loads(self.canonical_material),
        }

    @classmethod
    def from_json(cls, v):
        fields = {
            "kind",
            "digest",
            "schema_version",
            "owner",
            "revision",
            "acquired_at",
            "valid_from",
            "valid_until",
            "release_id",
            "material",
        }
        if not isinstance(v, Mapping) or set(v) != fields:
            raise ValueError("invalid snapshot material fields")
        if any(not isinstance(v[name], str) for name in fields - {"material"}):
            raise ValueError("snapshot material metadata must be text")
        return cls(
            v["kind"],
            v["digest"],
            v["schema_version"],
            v["owner"],
            v["revision"],
            datetime.fromisoformat(v["acquired_at"]),
            datetime.fromisoformat(v["valid_from"]),
            datetime.fromisoformat(v["valid_until"]),
            v["release_id"],
            canonical_json(v["material"]),
        )


@dataclass(frozen=True)
class PinnedDomainView:
    snapshot_id: str
    evaluated_at: datetime
    _resolver: object = field(repr=False, compare=False)

    @property
    def domain_snapshot_id(self) -> str:
        return self.snapshot_id

    def lookup_exact(self, key: str) -> object:
        lookup = getattr(self._resolver, "lookup_at", None)
        if not callable(lookup):
            raise RuntimeError("admitted domain material is not locally resolvable")
        return lookup(key, self.evaluated_at)


@dataclass(frozen=True)
class PinnedAlignmentView:
    policy: object
    evaluated_at: datetime


@dataclass(frozen=True)
class GraphixEvaluationContext:
    evaluated_at: datetime
    context_digest: str
    domain: PinnedDomainView
    alignment: PinnedAlignmentView


@dataclass(frozen=True)
class AdmittedContext:
    episode_id: str
    bundle_id: str
    acquired_at: datetime
    expires_at: datetime
    world: SnapshotMaterialRef
    self_state: SnapshotMaterialRef
    social: SnapshotMaterialRef
    normative: SnapshotMaterialRef
    domain: SnapshotMaterialRef
    memory: SnapshotMaterialRef
    capability: SnapshotMaterialRef
    csiu: SnapshotMaterialRef
    alignment: SnapshotMaterialRef
    leases: tuple[SnapshotLease, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    released: bool = field(default=False, init=False, compare=False)
    digest: str = field(init=False)

    def __post_init__(self):
        a, e = _utc(self.acquired_at, "acquired_at"), _utc(
            self.expires_at, "expires_at"
        )
        if e <= a or e - a > MAX_EPISODE_LIFETIME:
            raise ValueError("episode snapshot lease exceeds bounded lifetime")
        if any(not isinstance(r, SnapshotMaterialRef) for r in self.refs()):
            raise ValueError("all authorities require material-bearing references")
        if tuple(r.kind for r in self.refs()) != (
            "world",
            "self",
            "social",
            "normative",
            "domain",
            "memory",
            "capability",
            "csiu",
            "alignment",
        ):
            raise ValueError("snapshot refs are bound to wrong authority slot")
        if any(
            e > r.valid_until or not r.valid_from <= a <= r.valid_until
            for r in self.refs()
        ):
            raise ValueError("authority validity does not cover context")
        object.__setattr__(self, "acquired_at", a)
        object.__setattr__(self, "expires_at", e)
        object.__setattr__(self, "leases", tuple(self.leases))
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def refs(self):
        return (
            self.world,
            self.self_state,
            self.social,
            self.normative,
            self.domain,
            self.memory,
            self.capability,
            self.csiu,
            self.alignment,
        )

    def ref_digests(self):
        return tuple(r.digest for r in self.refs())

    def validate_active(self, now):
        if self.released:
            raise RuntimeError("snapshot bundle already released")
        if _utc(now, "now") >= self.expires_at:
            raise RuntimeError("snapshot bundle expired")

    def close(self):
        if self.released:
            return
        object.__setattr__(self, "released", True)
        first = None
        for lease in reversed(self.leases):
            try:
                lease.close()
            except BaseException as exc:
                first = first or exc
        if first:
            raise first

    def to_json(self, include_digest=True):
        value = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "bundle_id": self.bundle_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "refs": [r.to_json() for r in self.refs()],
        }
        if include_digest:
            value["digest"] = self.digest
        return value

    @classmethod
    def from_bytes(cls, raw):
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("admitted context must be non-empty bytes")
        value = canonical_json_loads(raw)
        if (
            not isinstance(value, dict)
            or canonical_json(value) != raw
            or set(value)
            != {
                "schema_version",
                "episode_id",
                "bundle_id",
                "acquired_at",
                "expires_at",
                "refs",
                "digest",
            }
            or value["schema_version"] != SCHEMA_VERSION
            or not isinstance(value["refs"], list)
            or len(value["refs"]) != 9
        ):
            raise ValueError("invalid canonical admitted context")
        context = cls(
            value["episode_id"],
            value["bundle_id"],
            datetime.fromisoformat(value["acquired_at"]),
            datetime.fromisoformat(value["expires_at"]),
            *(SnapshotMaterialRef.from_json(r) for r in value["refs"]),
        )
        if context.digest != value["digest"]:
            raise ValueError("admitted context digest mismatch")
        return context

    def bundle_ref(self):
        from vulcan.microkernel.episode import SnapshotBundleRef

        return SnapshotBundleRef(self.bundle_id, self.digest)

    def evaluation_context(self) -> GraphixEvaluationContext:
        domain = self.domain.live_view
        alignment = self.alignment.live_view
        if domain is None or alignment is None:
            raise RuntimeError("admitted live views are unavailable after restart")
        return GraphixEvaluationContext(
            self.acquired_at,
            self.digest,
            PinnedDomainView(self.domain.revision, self.acquired_at, domain),
            PinnedAlignmentView(getattr(alignment, "policy", None), self.acquired_at),
        )


SnapshotBundle = AdmittedContext


class SnapshotProvider(Protocol):
    def lease_snapshot(
        self, *, kind: str, episode_id: str, acquired_at: datetime, expires_at: datetime
    ) -> tuple[SnapshotMaterialRef, SnapshotLease | None]: ...
def default_snapshot_ref(
    kind, owner, revision, payload, *, acquired_at, expires_at, release_id
):
    material = canonical_json(
        {"kind": kind, "owner": owner, "revision": str(revision), "payload": payload}
    )
    return SnapshotMaterialRef(
        kind,
        Digest.of_bytes(material).hex,
        "opaque-state.v1",
        owner,
        str(revision),
        acquired_at,
        acquired_at,
        expires_at,
        release_id,
        material,
    )


def construct_snapshot_bundle(
    *,
    episode_id,
    providers: Sequence[SnapshotProvider],
    clock: Clock = utc_now,
    lifetime=MAX_EPISODE_LIFETIME,
):
    _text(episode_id, "episode_id")
    if lifetime <= timedelta(0) or lifetime > MAX_EPISODE_LIFETIME:
        raise ValueError("episode snapshot lease exceeds bounded lifetime")
    if len(providers) != 9:
        raise ValueError("exactly nine state authority providers are required")
    acquired = _utc(clock(), "acquired_at")
    expires = acquired + lifetime
    refs = []
    leases = []
    kinds = (
        "world",
        "self",
        "social",
        "normative",
        "domain",
        "memory",
        "capability",
        "csiu",
        "alignment",
    )
    try:
        for kind, provider in zip(kinds, providers, strict=True):
            ref, lease = provider.lease_snapshot(
                kind=kind,
                episode_id=episode_id,
                acquired_at=acquired,
                expires_at=expires,
            )
            if not isinstance(ref, SnapshotMaterialRef):
                raise ValueError("authority returned opaque snapshot reference")
            refs.append(ref)
            if lease is not None:
                leases.append(lease)
        identity = _digest(
            {
                "episode_id": episode_id,
                "evaluated_at": acquired.isoformat(),
                "refs": [r.to_json() for r in refs],
            }
        )
        return AdmittedContext(
            episode_id,
            f"context:{identity}",
            acquired,
            expires,
            *refs,
            leases=tuple(leases),
        )
    except BaseException as exc:
        for lease in reversed(leases):
            try:
                lease.close()
            except BaseException as close_exc:
                exc.add_note(f"lease cleanup failed: {close_exc}")
        raise


def require_bundle_snapshot(bundle, snapshot_digest, *, now, transition_event=False):
    bundle.validate_active(now)
    if snapshot_digest != bundle.digest and snapshot_digest not in bundle.ref_digests():
        if not transition_event:
            raise RuntimeError(
                "mixed snapshot versions require explicit transition/rebase event"
            )
        raise RuntimeError(
            "snapshot transition requires a typed authorized rebase event"
        )
