"""Runtime adapter for the evidence-driven capability registry."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vulcan.assurance.capabilities import CapabilityRegistry
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "capabilities.yaml"


def composed_runtime_ports() -> set[str]:
    from vulcan.runtime.app import generate_route_manifest
    return {f"{item['method']} {item['path']}" for item in generate_route_manifest()}


def load_capability_registry(now: datetime | None = None) -> CapabilityRegistry:
    observed_now = now or datetime.now(timezone.utc)
    return CapabilityRegistry.from_json_text(
        CONFIG_PATH.read_text(encoding="utf-8"),
        root=ROOT,
        now=observed_now,
        composed_ports=composed_runtime_ports(),
    )


def public_capability_response(now: datetime | None = None) -> dict[str, object]:
    registry = load_capability_registry(now)
    return {"capabilities": [dict(item) for item in registry.public_capabilities()]}
