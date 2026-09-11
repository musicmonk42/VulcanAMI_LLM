"""Explicit, independently versioned production state-snapshot authorities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Final

from vulcan.constitution.primitives import Digest, canonical_json
from vulcan.microkernel.snapshots import SnapshotLease, SnapshotMaterialRef

StateReader = Callable[[], tuple[str, object, SnapshotLease | None]]
AUTHORITY_KINDS: Final = (
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


class ContentBoundStateAuthority:
    """Snapshot an explicitly supplied canonical state document without reflection."""

    def __init__(
        self, *, kind: str, owner: str, schema: str, release: str, read: StateReader
    ):
        if kind not in AUTHORITY_KINDS:
            raise ValueError("unknown state authority kind")
        for value, name in ((owner, "owner"), (schema, "schema"), (release, "release")):
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError(f"invalid authority {name}")
        if not callable(read):
            raise TypeError("state reader must be callable")
        self.kind, self.owner, self.schema, self.release, self._read = (
            kind,
            owner,
            schema,
            release,
            read,
        )

    def lease_snapshot(
        self, *, kind: str, episode_id: str, acquired_at: datetime, expires_at: datetime
    ):
        if kind != self.kind:
            raise RuntimeError(f"{self.kind} authority cannot serve {kind}")
        revision, state, lease = self._read()
        try:
            if not isinstance(revision, str) or not revision or len(revision) > 128:
                raise ValueError("invalid authority revision")
            document = {
                "kind": kind,
                "owner": self.owner,
                "revision": revision,
                "release": self.release,
                "schema": self.schema,
                "valid_from": acquired_at.isoformat(),
                "valid_until": expires_at.isoformat(),
                "state": state,
            }
            digest = Digest.of_bytes(canonical_json(document)).hex
            ref = SnapshotMaterialRef(
                kind,
                digest,
                self.schema,
                self.owner,
                revision,
                acquired_at,
                acquired_at,
                expires_at,
                f"{self.owner}:{episode_id}:{revision}",
                canonical_json(document),
                lease,
            )
            return ref, lease
        except BaseException as exc:
            if lease is not None:
                try:
                    lease.close()
                except BaseException as close_exc:
                    exc.add_note(f"snapshot lease cleanup failed: {close_exc}")
            raise


def disabled_authority(kind: str, *, reason: str) -> ContentBoundStateAuthority:
    """A real, content-bound declaration that an authority has no enabled state."""
    return ContentBoundStateAuthority(
        kind=kind,
        owner=f"constitutional:{kind}",
        schema=f"vulcan-{kind}-disabled.v1",
        release="constitutional-v1",
        read=lambda: ("0", {"enabled": False, "reason": reason}, None),
    )


class DisabledCSIUPolicyAuthority(ContentBoundStateAuthority):
    """Explicit CSIU policy owner when no live CSIU policy is admitted.

    This is intentionally not backed by a learning or improvement runtime.
    """

    def __init__(self, *, reason: str = "CSIU policy authority is disabled"):
        super().__init__(
            kind="csiu",
            owner="constitutional:disabled-csiu-policy",
            schema="vulcan-csiu-policy-disabled.v1",
            release="constitutional-v1",
            read=lambda: (
                "0",
                {"enabled": False, "mode": "proposal-only", "reason": reason},
                None,
            ),
        )


@dataclass(frozen=True)
class StateAuthoritySet:
    world: ContentBoundStateAuthority
    self_state: ContentBoundStateAuthority
    social: ContentBoundStateAuthority
    normative: ContentBoundStateAuthority
    domain: ContentBoundStateAuthority
    memory: ContentBoundStateAuthority
    capability: ContentBoundStateAuthority
    csiu: ContentBoundStateAuthority
    alignment: ContentBoundStateAuthority

    def __post_init__(self) -> None:
        self.validate()

    def providers(self) -> tuple[ContentBoundStateAuthority, ...]:
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

    def validate(self) -> None:
        if len({id(authority) for authority in self.providers()}) != len(
            AUTHORITY_KINDS
        ):
            raise RuntimeError(
                "each state authority slot requires an independent provider"
            )
        kinds = tuple(authority.kind for authority in self.providers())
        if kinds != AUTHORITY_KINDS:
            raise RuntimeError("state authority set is incomplete or mis-bound")
