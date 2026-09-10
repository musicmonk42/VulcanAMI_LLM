"""Acyclic value contracts shared by audit clients and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AuditError(RuntimeError):
    """Raised when the canonical audit cannot preserve its contract."""


@dataclass(frozen=True)
class AuditEvent:
    schema_version: str
    sequence: int
    event_type: str
    timestamp: str
    previous_hash: str
    data: dict[str, Any]
    event_hash: str
    segment: int = 0
    segment_sequence: int = 0


@dataclass(frozen=True)
class AuditDurabilityProfile:
    fsync_events: bool = True
    fsync_manifest: bool = True


class Failpoint:
    def hit(self, name: str) -> None:
        return None
