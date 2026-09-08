"""Reflective snapshot adapter retained only for legacy tests and migrations."""

from __future__ import annotations

import re

from vulcan.microkernel.snapshots import SnapshotRef, default_snapshot_ref

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AttributeSnapshotProvider:
    """Compatibility adapter; remove when legacy tests construct explicit authorities."""

    def __init__(self, owner: object, *, owner_name: str):
        self.owner = owner
        self.owner_name = owner_name

    def lease_snapshot(self, *, kind, episode_id, acquired_at, expires_at):
        lease_fn = getattr(self.owner, "lease", None)
        lease = lease_fn() if callable(lease_fn) else None
        target = lease if lease is not None else self.owner
        digest = (
            getattr(target, f"{kind}_snapshot_id", None)
            or getattr(target, "policy_digest", None)
            or getattr(target, "digest", None)
            or getattr(target, "snapshot_id", None)
        )
        revision = (
            getattr(target, "revision", None) or getattr(target, "version", None) or "0"
        )
        if isinstance(digest, str) and _HEX64.fullmatch(digest):
            return (
                SnapshotRef(
                    kind,
                    digest,
                    "legacy-reflective.v1",
                    self.owner_name,
                    str(revision),
                    acquired_at,
                    acquired_at,
                    expires_at,
                    f"migration:{episode_id}",
                ),
                lease,
            )
        return (
            default_snapshot_ref(
                kind,
                self.owner_name,
                revision,
                repr(digest),
                acquired_at=acquired_at,
                expires_at=expires_at,
                release_id=f"migration:{episode_id}",
            ),
            lease,
        )
