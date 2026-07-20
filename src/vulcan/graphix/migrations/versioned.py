from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
from typing import Callable, Mapping
from vulcan.graphix.codec import canonical_json
from vulcan.graphix.core import GraphixEnvelope

@dataclass(frozen=True, slots=True)
class MigrationRecord:
    source_digest: str; target_digest: str; from_version: int; to_version: int; artifact: GraphixEnvelope

Migration = Callable[[GraphixEnvelope], GraphixEnvelope]

class MigrationError(ValueError): pass

class MigrationPlan:
    def __init__(self, migrations: Mapping[tuple[str,int,int], Migration]) -> None:
        self._migrations = MappingProxyType(dict(migrations))
    def migrate(self, artifact: GraphixEnvelope, *, to_version: int) -> MigrationRecord:
        source = _digest(artifact)
        if artifact.schema_version == to_version:
            return MigrationRecord(source, source, artifact.schema_version, to_version, artifact)
        fn = self._migrations.get((artifact.dialect, artifact.schema_version, to_version))
        if fn is None: raise MigrationError("unsupported or ambiguous Graphix migration")
        migrated = fn(artifact)
        if migrated.dialect != artifact.dialect or migrated.schema_version != to_version: raise MigrationError("migration target mismatch")
        if migrated.authority_level != artifact.authority_level: raise MigrationError("migration cannot silently change authority level")
        return MigrationRecord(source, _digest(migrated), artifact.schema_version, to_version, migrated)

def identity_v1(envelope: GraphixEnvelope) -> GraphixEnvelope: return envelope

def _digest(envelope: GraphixEnvelope) -> str:
    from vulcan.graphix.codec import envelope_to_dict
    return "sha256:"+hashlib.sha256(canonical_json(envelope_to_dict(envelope))).hexdigest()
