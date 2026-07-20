"""Route inventory generated from the composed ASGI route table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

RouteMethod = Literal["GET", "POST", "PATCH", "DELETE"]
RouteExposure = Literal["public", "operator", "tenant"]

_PUBLIC = {("GET", "/health/live"), ("GET", "/health/ready"), ("GET", "/v1/capabilities")}
_SCOPE_BY_ROUTE: dict[tuple[str, str], str] = {
    ("GET", "/health/integrity"): "operator:read",
    ("POST", "/v1/chat"): "reason:write",
    ("POST", "/v1/chat/orchestrated"): "reason:write",
    ("POST", "/vulcan/v1/chat"): "reason:write",
    ("POST", "/v1/admin/domains"): "domains:write",
    ("POST", "/v1/admin/alignment"): "alignment:write",
    ("GET", "/v1/audit/cases/{case_id}"): "audit:read",
    ("GET", "/v1/admin/improvements"): "self_improvement:read",
    ("GET", "/v1/admin/improvements/{proposal_id}"): "self_improvement:read",
    ("POST", "/v1/admin/improvements/{proposal_id}/approve"): "self_improvement:approve",
    ("POST", "/v1/admin/improvements/{proposal_id}/reject"): "self_improvement:approve",
    ("POST", "/v1/admin/improvements/{proposal_id}/resume"): "self_improvement:approve",
    ("GET", "/v1/admin/improvements/{proposal_id}/status"): "self_improvement:read",
    ("GET", "/v1/audit/improvements/{proposal_digest}"): "audit:read",
    ("POST", "/v1/memory/preferences"): "memory:write",
    ("GET", "/v1/memory/preferences/{key}"): "memory:read",
    ("PATCH", "/v1/memory/preferences/{record_id}"): "memory:write",
    ("DELETE", "/v1/memory/preferences/{record_id}"): "memory:forget",
}

@dataclass(frozen=True)
class RouteManifestEntry:
    path: str
    method: RouteMethod
    classification: str
    auth_scope: str | None
    exposure: RouteExposure

    def public_dict(self) -> dict[str, str | bool | None]:
        return {
            "path": self.path,
            "method": self.method,
            "classification": self.classification,
            "authentication_required": self.auth_scope is not None,
            "auth_scope": self.auth_scope,
            "exposure": self.exposure,
            "authorization": "public" if self.auth_scope is None else self.auth_scope,
        }

def _exposure(path: str, scope: str | None) -> RouteExposure:
    if scope is None:
        return "public"
    if path.startswith(("/v1/admin/", "/v1/audit/", "/health/integrity")):
        return "operator"
    return "tenant"

def route_manifest(app=None) -> tuple[RouteManifestEntry, ...]:
    if app is None:
        routes = tuple((path, method) for (method, path) in sorted(_PUBLIC | set(_SCOPE_BY_ROUTE)))
    else:
        rows: list[tuple[str, str]] = []
        for route in app.routes:
            path = getattr(route, "path", None)
            methods: Iterable[str] = getattr(route, "methods", ()) or ()
            if not isinstance(path, str) or path in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}:
                continue
            for method in methods:
                if method in {"GET", "POST", "PATCH", "DELETE"}:
                    rows.append((path, method))
        routes = tuple(sorted(set(rows), key=lambda x: (x[0], x[1])))
    entries=[]
    for path, method in routes:
        scope = _SCOPE_BY_ROUTE.get((method, path))
        if (method, path) not in _PUBLIC and scope is None:
            raise RuntimeError(f"unclassified route {method} {path}")
        entries.append(RouteManifestEntry(path, method, "public" if scope is None else "protected", scope, _exposure(path, scope)))
    return tuple(entries)

def generate_route_manifest(app=None) -> tuple[dict[str, str | bool | None], ...]:
    return tuple(entry.public_dict() for entry in route_manifest(app))
