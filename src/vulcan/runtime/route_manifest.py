"""Dependency-light static route manifest for diagnostics and capability inventory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouteMethod = Literal["GET", "POST", "PATCH", "DELETE"]
RouteClassification = Literal["public", "protected"]

@dataclass(frozen=True)
class RouteManifestEntry:
    path: str
    method: RouteMethod
    classification: RouteClassification

    def public_dict(self) -> dict[str, str]:
        return {"path": self.path, "method": self.method, "classification": self.classification}

_ROUTES: tuple[tuple[str, RouteMethod], ...] = (
    ("/health/live", "GET"), ("/health/ready", "GET"), ("/v1/capabilities", "GET"),
    ("/v1/chat", "POST"), ("/v1/admin/domains", "POST"), ("/v1/admin/alignment", "POST"),
    ("/v1/audit/cases/{case_id}", "GET"), ("/v1/admin/improvements", "GET"),
    ("/v1/admin/improvements/{proposal_id}", "GET"), ("/v1/admin/improvements/{proposal_id}/approve", "POST"),
    ("/v1/admin/improvements/{proposal_id}/reject", "POST"), ("/v1/admin/improvements/{proposal_id}/resume", "POST"),
    ("/v1/admin/improvements/{proposal_id}/status", "GET"), ("/v1/audit/improvements/{proposal_digest}", "GET"),
    ("/v1/memory/preferences", "POST"), ("/v1/memory/preferences/{key}", "GET"),
    ("/v1/memory/preferences/{record_id}", "PATCH"), ("/v1/memory/preferences/{record_id}", "DELETE"),
)

def route_manifest() -> tuple[RouteManifestEntry, ...]:
    return tuple(RouteManifestEntry(path, method, "public" if path.startswith("/health/") else "protected") for path, method in _ROUTES)

def generate_route_manifest() -> tuple[dict[str, str], ...]:
    return tuple(entry.public_dict() for entry in route_manifest())
