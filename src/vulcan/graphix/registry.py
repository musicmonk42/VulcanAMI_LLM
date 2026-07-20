"""Startup-only release-bound dialect registry for Graphix Core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from vulcan.graphix.core import GraphixCoreError, GraphixEnvelope, UnsupportedDialectError

MigrationFunction = Callable[[GraphixEnvelope], GraphixEnvelope]

@dataclass(frozen=True, slots=True)
class DialectRegistration:
    dialect: str
    schema_version: int
    release_id: str
    compatible_versions: frozenset[int] = frozenset()
    migrations: Mapping[int, MigrationFunction] = field(default_factory=dict)

class DialectRegistry:
    """Explicit registry; call freeze() after startup wiring and before use."""
    def __init__(self, *, release_id: str) -> None:
        if not release_id:
            raise GraphixCoreError("registry release_id is required")
        self._release_id = release_id
        self._frozen = False
        self._items: dict[tuple[str, int], DialectRegistration] = {}
    def register(self, registration: DialectRegistration) -> None:
        if self._frozen:
            raise GraphixCoreError("dialect registration is startup-only")
        if registration.release_id != self._release_id:
            raise GraphixCoreError("dialect registration release mismatch")
        key = (registration.dialect, registration.schema_version)
        if key in self._items:
            raise GraphixCoreError("duplicate dialect registration")
        self._items[key] = registration
    def freeze(self) -> None:
        self._frozen = True
    @property
    def frozen(self) -> bool:
        return self._frozen
    def require_supported(self, envelope: GraphixEnvelope) -> DialectRegistration:
        reg = self._items.get((envelope.dialect, envelope.schema_version))
        if reg is None:
            raise UnsupportedDialectError("unknown Graphix dialect/schema version")
        return reg
    def migrate_to(self, envelope: GraphixEnvelope, target_version: int) -> GraphixEnvelope:
        reg = self.require_supported(envelope)
        if target_version == envelope.schema_version:
            return envelope
        if target_version not in reg.compatible_versions or target_version not in reg.migrations:
            raise UnsupportedDialectError("no explicit compatible migration for target version")
        migrated = reg.migrations[target_version](envelope)
        if migrated.dialect != envelope.dialect or migrated.schema_version != target_version:
            raise GraphixCoreError("migration returned wrong dialect or schema version")
        return migrated
