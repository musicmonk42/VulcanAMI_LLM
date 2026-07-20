from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from vulcan.graphix.codec import canonical_json, extension_digest
from vulcan.graphix.core import AuthorityLevel, EpistemicStatus, GraphixEnvelope, PrivacyClass, SourceKind

class DiagnosticSeverity(str, Enum): ERROR="ERROR"; WARNING="WARNING"
class DiagnosticCode(str, Enum):
    STAGE_ORDER="STAGE_ORDER"; STRUCTURE="STRUCTURE"; IDENTITY="IDENTITY"; SOURCE="SOURCE"; ONTOLOGY="ONTOLOGY"; REFERENCE="REFERENCE"; TEMPORAL="TEMPORAL"; PRIVACY="PRIVACY"; RESOURCE="RESOURCE"; AUTHORITY="AUTHORITY"; EPISTEMIC="EPISTEMIC"; DEONTIC="DEONTIC"; CAPABILITY="CAPABILITY"; EXTENSION="EXTENSION"

@dataclass(frozen=True, slots=True)
class ValidationDiagnostic:
    stage: str; code: DiagnosticCode; severity: DiagnosticSeverity; message: str

@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    now: Callable[[], datetime]
    trusted_principals: frozenset[str]
    allowed_dialects: frozenset[str]
    allowed_capabilities: frozenset[str] = frozenset()
    known_ontology_terms: frozenset[str] = frozenset({"request","arithmetic","lookup","unsupported"})
    max_sources: int = 16; max_extensions: int = 8; max_canonical_bytes: int = 32768

@dataclass(frozen=True, slots=True)
class ValidatedGraphixArtifact:
    envelope: GraphixEnvelope
    target_dialect: str
    stage_digests: Mapping[str, str]
    diagnostics: tuple[ValidationDiagnostic, ...]
    validation_digest: str

STAGES: tuple[str, ...] = ("structure","identity","source_grounding","ontology","reference_integrity","temporal_validity","privacy_consent","resource_bounds","authority","epistemic_integrity","deontic_constraints","executable_capability_admission")
SECURITY_EXTENSION_WORDS = frozenset({"security","authorization","authority","policy","evidence","execution","execute","capability","tool"})

class ValidationError(ValueError): pass

def validate_graphix(envelope: GraphixEnvelope, *, target_dialect: str, policy: ValidationPolicy, stages: Sequence[str] = STAGES) -> ValidatedGraphixArtifact:
    if tuple(stages) != STAGES:
        raise ValidationError("Graphix validation stages are mandatory and ordered")
    diagnostics: list[ValidationDiagnostic] = []
    before = _digest(envelope)
    stage_digests: dict[str, str] = {}
    for stage in STAGES:
        diagnostics.extend(_run_stage(stage, envelope, target_dialect, policy))
        after = _digest(envelope)
        if after != before:
            raise ValidationError(f"validation stage mutated input: {stage}")
        stage_digests[stage] = after
    errors = tuple(d for d in diagnostics if d.severity is DiagnosticSeverity.ERROR)
    if errors:
        raise ValidationError("; ".join(f"{d.stage}:{d.message}" for d in errors))
    val_digest = "sha256:" + hashlib.sha256(canonical_json({"envelope": before, "target": target_dialect, "stages": list(STAGES), "stage_digests": stage_digests})).hexdigest()
    return ValidatedGraphixArtifact(envelope, target_dialect, MappingProxyType(stage_digests), tuple(diagnostics), val_digest)

def _run_stage(stage: str, e: GraphixEnvelope, target: str, p: ValidationPolicy) -> list[ValidationDiagnostic]:
    err = lambda code, msg: ValidationDiagnostic(stage, code, DiagnosticSeverity.ERROR, msg)
    out: list[ValidationDiagnostic] = []
    if stage == "structure":
        if e.dialect not in p.allowed_dialects or target not in p.allowed_dialects: out.append(err(DiagnosticCode.STRUCTURE,"unsupported source or target dialect"))
    elif stage == "identity":
        if e.proposer.principal_id not in p.trusted_principals: out.append(err(DiagnosticCode.IDENTITY,"untrusted proposer release"))
    elif stage == "source_grounding":
        if not e.source_references: out.append(err(DiagnosticCode.SOURCE,"source reference required"))
        if any(s.kind is SourceKind.EXTERNAL and s.digest is None for s in e.source_references): out.append(err(DiagnosticCode.SOURCE,"external source requires digest"))
    elif stage == "ontology":
        if e.dialect == "graphix.language" and "graphix.language" not in p.allowed_dialects: out.append(err(DiagnosticCode.ONTOLOGY,"language dialect not admitted"))
    elif stage == "reference_integrity":
        ids = [s.reference_id for s in e.source_references]
        if len(ids) != len(set(ids)): out.append(err(DiagnosticCode.REFERENCE,"duplicate source reference"))
    elif stage == "temporal_validity":
        now = p.now().astimezone(timezone.utc)
        if e.valid_from.astimezone(timezone.utc) > now: out.append(err(DiagnosticCode.TEMPORAL,"artifact is not yet valid"))
        if e.valid_until and e.valid_until.astimezone(timezone.utc) <= now: out.append(err(DiagnosticCode.TEMPORAL,"artifact is expired"))
    elif stage == "privacy_consent":
        if e.privacy_class in {PrivacyClass.PERSONAL, PrivacyClass.SENSITIVE_PERSONAL} and not e.consent_references: out.append(err(DiagnosticCode.PRIVACY,"personal data requires consent reference"))
    elif stage == "resource_bounds":
        if len(e.source_references) > p.max_sources or len(e.extensions) > p.max_extensions or len(canonical_json(_view(e))) > p.max_canonical_bytes: out.append(err(DiagnosticCode.RESOURCE,"artifact exceeds validation resource bounds"))
    elif stage == "authority":
        if e.authority_level is not AuthorityLevel.UNTRUSTED_PROPOSAL: out.append(err(DiagnosticCode.AUTHORITY,"compiler ingress only accepts untrusted proposals"))
    elif stage == "epistemic_integrity":
        if e.epistemic_status is not EpistemicStatus.PROPOSED: out.append(err(DiagnosticCode.EPISTEMIC,"compiler ingress cannot accept committed epistemic status"))
    elif stage == "deontic_constraints":
        if any(word in e.purpose.lower() for word in ("bypass","override","ignore policy")): out.append(err(DiagnosticCode.DEONTIC,"purpose requests policy bypass"))
    elif stage == "executable_capability_admission":
        for x in e.extensions:
            parts = frozenset(x.namespace.lower().replace("-",".").split("."))
            if parts & SECURITY_EXTENSION_WORDS: out.append(err(DiagnosticCode.EXTENSION,"unknown extension claims reserved meaning"))
            if extension_digest(x.value) != x.digest: out.append(err(DiagnosticCode.EXTENSION,"extension digest mismatch"))
    return out

def _digest(e: GraphixEnvelope) -> str: return "sha256:" + hashlib.sha256(canonical_json(_view(e))).hexdigest()
def _view(e: GraphixEnvelope) -> object:
    from vulcan.graphix.codec import envelope_to_dict
    return envelope_to_dict(e)
