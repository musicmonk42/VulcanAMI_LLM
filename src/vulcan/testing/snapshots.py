"""Reflective snapshot adapter retained only for legacy tests and migrations."""

from __future__ import annotations

from vulcan.microkernel.snapshots import default_snapshot_ref


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
        return (
            default_snapshot_ref(
                kind,
                self.owner_name,
                revision,
                {"legacy_digest": repr(digest)},
                acquired_at=acquired_at,
                expires_at=expires_at,
                release_id=f"migration:{episode_id}",
            ),
            lease,
        )
