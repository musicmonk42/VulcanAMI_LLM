"""Strict duplicate-key-free canonical JSON codec for Graphix Core."""
from __future__ import annotations

from datetime import datetime, timezone
import json, math, unicodedata
from collections.abc import Mapping, Sequence
from typing import Final

from vulcan.graphix.core import (AuthorityLevel, DigestMismatchError, EpistemicStatus, ExtensionDeclaration, ForbiddenExecutableSemanticsError, GraphixCoreError, GraphixEnvelope, PrincipalRelease, PrivacyClass, SourceKind, SourceReference, UnknownFieldError)

MAX_JSON_BYTES: Final = 65536
MAX_DEPTH: Final = 24
MAX_ITEMS: Final = 10000
MAX_STRING: Final = 16384
ENVELOPE_FIELDS = frozenset({"dialect","schema_version","node_artifact_id","episode_id","content_digest","proposer","authority_level","source_references","snapshot_bundle_digest","epistemic_status","privacy_class","purpose","consent_references","valid_from","valid_until","extensions"})
PROPOSER_FIELDS = frozenset({"principal_id", "release_id"})
SOURCE_FIELDS = frozenset({"kind", "reference_id", "digest"})
EXTENSION_FIELDS = frozenset({"namespace", "schema_version", "digest", "value"})
EXECUTABLE_KEYS = frozenset({"callable","pickle","class_path","shell_command","command","code","dynamic_import","import","module","function"})

def loads_envelope(raw: bytes | str) -> GraphixEnvelope:
    if isinstance(raw, str): raw = raw.encode("utf-8")
    if len(raw) > MAX_JSON_BYTES: raise GraphixCoreError("Graphix Core JSON exceeds byte bound")
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in pairs:
            nk = _string(k)
            if nk in out: raise UnknownFieldError(f"duplicate JSON key: {nk}")
            out[nk] = v
        return out
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook, parse_constant=lambda c: (_ for _ in ()).throw(GraphixCoreError(f"invalid constant {c}")))
    except UnicodeDecodeError as exc: raise GraphixCoreError("Graphix Core JSON must be UTF-8") from exc
    validate_json_value(data, allow_extension_objects=True)
    if not isinstance(data, dict): raise GraphixCoreError("Graphix Core envelope must be an object")
    unknown = set(data) - ENVELOPE_FIELDS
    if unknown: raise UnknownFieldError(f"unknown Graphix Core fields: {sorted(unknown)}")
    missing = ENVELOPE_FIELDS - set(data)
    if missing: raise UnknownFieldError(f"missing Graphix Core fields: {sorted(missing)}")
    return _envelope_from_dict(data)

def dumps_envelope(envelope: GraphixEnvelope) -> bytes:
    return canonical_json(envelope_to_dict(envelope))

def verify_envelope_digest(envelope: GraphixEnvelope, content: bytes) -> None:
    import hashlib
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    if envelope.content_digest != expected: raise DigestMismatchError("content digest does not match payload")

def extension_digest(value: Mapping[str, object]) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()

def canonical_json(value: object) -> bytes:
    return json.dumps(_canonical(value, 0, [0]), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def validate_json_value(value: object, *, allow_extension_objects: bool = False) -> None:
    _canonical(value, 0, [0])

def _canonical(value: object, depth: int, count: list[int]) -> object:
    if depth > MAX_DEPTH: raise GraphixCoreError("JSON depth bound exceeded")
    count[0] += 1
    if count[0] > MAX_ITEMS: raise GraphixCoreError("JSON item bound exceeded")
    if value is None or isinstance(value, bool): return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 2**53 - 1: raise GraphixCoreError("integer outside interoperable JSON bound")
        return value
    if isinstance(value, float):
        if not math.isfinite(value): raise GraphixCoreError("non-finite number rejected")
        if abs(value) > 2**53 - 1: raise GraphixCoreError("number outside interoperable JSON bound")
        return value
    if isinstance(value, str): return _string(value)
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for k, v in value.items():
            nk = _string(k)
            if nk.lower() in EXECUTABLE_KEYS: raise ForbiddenExecutableSemanticsError(f"forbidden executable key: {nk}")
            if nk in out: raise UnknownFieldError(f"duplicate normalized key: {nk}")
            out[nk] = _canonical(v, depth+1, count)
        return {k: out[k] for k in sorted(out)}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_canonical(v, depth+1, count) for v in value]
    raise GraphixCoreError(f"unsupported JSON value: {type(value).__name__}")

def _string(value: object) -> str:
    if not isinstance(value, str): raise GraphixCoreError("JSON object key must be string")
    out = unicodedata.normalize("NFC", value)
    if len(out) > MAX_STRING: raise GraphixCoreError("string length bound exceeded")
    if any(ord(ch) < 0x20 for ch in out): raise GraphixCoreError("control character rejected")
    return out

def envelope_to_dict(e: GraphixEnvelope) -> dict[str, object]:
    return {"dialect":e.dialect,"schema_version":e.schema_version,"node_artifact_id":e.node_artifact_id,"episode_id":e.episode_id,"content_digest":e.content_digest,"proposer":{"principal_id":e.proposer.principal_id,"release_id":e.proposer.release_id},"authority_level":e.authority_level.value,"source_references":[{"kind":s.kind.value,"reference_id":s.reference_id,"digest":s.digest} for s in e.source_references],"snapshot_bundle_digest":e.snapshot_bundle_digest,"epistemic_status":e.epistemic_status.value,"privacy_class":e.privacy_class.value,"purpose":e.purpose,"consent_references":list(e.consent_references),"valid_from":_dt(e.valid_from),"valid_until":None if e.valid_until is None else _dt(e.valid_until),"extensions":[{"namespace":x.namespace,"schema_version":x.schema_version,"digest":x.digest,"value":dict(x.value)} for x in e.extensions]}

def _envelope_from_dict(d: Mapping[str, object]) -> GraphixEnvelope:
    p = _map(d["proposer"], "proposer")
    _require_fields(p, PROPOSER_FIELDS, "proposer")
    sources = []
    for s in _seq(d["source_references"], "source_references"):
        sm = _map(s, "source")
        _require_fields(sm, SOURCE_FIELDS, "source")
        sources.append(SourceReference(SourceKind(str(sm["kind"])), str(sm["reference_id"]), None if sm["digest"] is None else str(sm["digest"])))
    extensions = []
    for x in _seq(d["extensions"], "extensions"):
        xm = _map(x, "extension")
        _require_fields(xm, EXTENSION_FIELDS, "extension")
        if not isinstance(xm["value"], Mapping): raise GraphixCoreError("extension.value must be object")
        extensions.append(ExtensionDeclaration(str(xm["namespace"]), _int(xm["schema_version"]), str(xm["digest"]), xm["value"]))
    return GraphixEnvelope(dialect=str(d["dialect"]), schema_version=_int(d["schema_version"]), node_artifact_id=str(d["node_artifact_id"]), episode_id=str(d["episode_id"]), content_digest=str(d["content_digest"]), proposer=PrincipalRelease(str(p["principal_id"]), str(p["release_id"])), authority_level=AuthorityLevel(str(d["authority_level"])), source_references=tuple(sources), snapshot_bundle_digest=str(d["snapshot_bundle_digest"]), epistemic_status=EpistemicStatus(str(d["epistemic_status"])), privacy_class=PrivacyClass(str(d["privacy_class"])), purpose=str(d["purpose"]), consent_references=tuple(str(x) for x in _seq(d["consent_references"], "consent_references")), valid_from=_parse_dt(str(d["valid_from"])), valid_until=None if d["valid_until"] is None else _parse_dt(str(d["valid_until"])), extensions=tuple(extensions))

def _require_fields(v: Mapping[str, object], allowed: frozenset[str], name: str) -> None:
    unknown = set(v) - allowed
    missing = allowed - set(v)
    if unknown or missing: raise UnknownFieldError(f"invalid {name} fields")

def _map(v: object, name: str) -> Mapping[str, object]:
    if not isinstance(v, Mapping): raise GraphixCoreError(f"{name} must be object")
    return v
def _seq(v: object, name: str) -> Sequence[object]:
    if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)): raise GraphixCoreError(f"{name} must be array")
    return v
def _int(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool): raise GraphixCoreError("expected integer")
    return v
def _parse_dt(v: str) -> datetime:
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if dt.tzinfo is None: raise GraphixCoreError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)
def _dt(v: datetime) -> str:
    return v.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
