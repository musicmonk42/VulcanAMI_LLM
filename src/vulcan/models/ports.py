"""Read-only state authority ports for episode snapshot admission."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from vulcan.microkernel.snapshots import SnapshotLease, SnapshotRef

@runtime_checkable
class ReadOnlyStatePort(Protocol):
    def lease_snapshot(self, *, kind: str, episode_id: str, acquired_at: datetime, expires_at: datetime) -> tuple[SnapshotRef, SnapshotLease | None]: ...

class WorldStatePort(ReadOnlyStatePort, Protocol):
    pass
class SelfStatePort(ReadOnlyStatePort, Protocol):
    pass
class SocialStatePort(ReadOnlyStatePort, Protocol):
    pass
class NormativeStatePort(ReadOnlyStatePort, Protocol):
    pass
class DomainStatePort(ReadOnlyStatePort, Protocol):
    pass
class MemoryViewPort(ReadOnlyStatePort, Protocol):
    pass
class CapabilityManifestPort(ReadOnlyStatePort, Protocol):
    pass
class AlignmentPolicyPort(ReadOnlyStatePort, Protocol):
    pass
class CSIUPolicyPort(ReadOnlyStatePort, Protocol):
    pass

__all__ = ["ReadOnlyStatePort", "WorldStatePort", "SelfStatePort", "SocialStatePort", "NormativeStatePort", "DomainStatePort", "MemoryViewPort", "CapabilityManifestPort", "AlignmentPolicyPort", "CSIUPolicyPort"]
