"""Compatibility exports for canonical runtime route inventory."""
from __future__ import annotations

from .route_manifest import RouteManifestEntry, generate_route_manifest, route_manifest

__all__ = ["RouteManifestEntry", "generate_route_manifest", "route_manifest"]
