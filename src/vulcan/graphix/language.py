"""Graphix Language v1: grounded language ingress and realization contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import re
import unicodedata
from types import MappingProxyType
from typing import Mapping, Sequence

from vulcan.graphix.codec import canonical_json
from vulcan.graphix.core import GraphixCoreError, SourceKind, SourceReference

LANGUAGE_SCHEMA_VERSION = "graphix.language/1"
MAX_TEXT_CHARS = 10_000
MAX_SPANS = 32
MAX_MENTIONS = 32
MAX_FRAMES = 16
MAX_CANDIDATES = 8
MAX_STYLE_RULES = 16
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_LOCALE_RE = re.compile(r"^[A-Za-z0-9_-]{2,35}$")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
FORBIDDEN_PROVIDER_FIELDS = frozenset({
    "answer", "belief", "claim", "fact", "facts", "evidence", "citation", "citations",
    "tool", "tools", "tool_call", "tool_calls", "function_call", "policy", "authorization",
    "authority", "memory", "memory_mutation", "memory_write", "code", "command", "executable",
    "plan", "graphix_plan", "sql", "path", "secret", "token", "raw_prompt",
})

class LanguageContractError(GraphixCoreError): pass
class ForbiddenProviderFieldError(LanguageContractError): pass
class UngroundedSemanticValueError(LanguageContractError): pass

class PrivacyLabel(str, Enum):
    PUBLIC = "PUBLIC"; INTERNAL = "INTERNAL"; PERSONAL = "PERSONAL"; SENSITIVE_PERSONAL = "SENSITIVE_PERSONAL"; SECRET = "SECRET"
class DialogueActKind(str, Enum):
    ASK = "ASK"; REQUEST = "REQUEST"; INFORM = "INFORM"; CLARIFY = "CLARIFY"; CORRECT = "CORRECT"; REFUSE = "REFUSE"; OTHER = "OTHER"
class Modality(str, Enum):
    ASSERTIVE = "ASSERTIVE"; QUESTION = "QUESTION"; REQUEST = "REQUEST"; POSSIBLE = "POSSIBLE"; REQUIRED = "REQUIRED"; PROHIBITED = "PROHIBITED"; PREFERRED = "PREFERRED"

@dataclass(frozen=True, slots=True)
class UtteranceRef:
    utterance_id: str
    normalized_text: str
    digest: str = ""
    locale: str = "und"
    normalization: str = "NFC"
    privacy_label: PrivacyLabel = PrivacyLabel.INTERNAL
    def __post_init__(self) -> None:
        _id("utterance_id", self.utterance_id)
        text = _norm_text(self.normalized_text)
        if not text or len(text) > MAX_TEXT_CHARS: raise LanguageContractError("utterance text outside bounds")
        if self.normalization != "NFC": raise LanguageContractError("only NFC normalization is supported")
        if not _LOCALE_RE.fullmatch(self.locale): raise LanguageContractError("invalid locale")
        object.__setattr__(self, "normalized_text", text)
        object.__setattr__(self, "privacy_label", PrivacyLabel(self.privacy_label))
        expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        object.__setattr__(self, "digest", expected if not self.digest else self.digest)
        if self.digest != expected: raise LanguageContractError("utterance digest mismatch")

@dataclass(frozen=True, slots=True)
class SourceSpan:
    utterance_id: str
    start: int
    end: int
    normalized_text: str
    privacy_label: PrivacyLabel = PrivacyLabel.INTERNAL
    unit: str = "unicode-codepoint"
    def __post_init__(self) -> None:
        _id("span.utterance_id", self.utterance_id)
        if self.unit != "unicode-codepoint" or type(self.start) is not int or type(self.end) is not int or self.start < 0 or self.end <= self.start:
            raise LanguageContractError("invalid source span")
        object.__setattr__(self, "normalized_text", _norm_text(self.normalized_text))
        object.__setattr__(self, "privacy_label", PrivacyLabel(self.privacy_label))
    def resolve(self, utterance: UtteranceRef) -> str:
        if utterance.utterance_id != self.utterance_id or self.end > len(utterance.normalized_text): raise LanguageContractError("span utterance mismatch")
        found = utterance.normalized_text[self.start:self.end]
        if found != self.normalized_text: raise LanguageContractError("span reconstruction mismatch")
        return found
    def redacted(self) -> "SourceSpan":
        return SourceSpan(self.utterance_id, self.start, self.end, "█" * (self.end - self.start), self.privacy_label, self.unit)

@dataclass(frozen=True, slots=True)
class GroundedValue:
    value: str
    source_spans: tuple[SourceSpan, ...] = ()
    external_sources: tuple[SourceReference, ...] = ()
    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _norm_text(self.value))
        object.__setattr__(self, "source_spans", tuple(self.source_spans))
        object.__setattr__(self, "external_sources", tuple(self.external_sources))
        if not self.value or (not self.source_spans and not self.external_sources): raise UngroundedSemanticValueError("semantic value requires source span or external source")
        if len(self.source_spans) > MAX_SPANS: raise LanguageContractError("too many source spans")

@dataclass(frozen=True, slots=True)
class DialogueAct: kind: DialogueActKind; span: SourceSpan; polarity: str = "affirmed"
@dataclass(frozen=True, slots=True)
class EntityMention: entity_type: str; surface: GroundedValue; canonical_hint: GroundedValue | None = None
@dataclass(frozen=True, slots=True)
class SemanticFrame:
    frame_type: str; predicate: GroundedValue; participants: Mapping[str, GroundedValue] = field(default_factory=dict); negated: bool = False; modality: Modality = Modality.ASSERTIVE; quantities: tuple[GroundedValue, ...] = (); temporal_expressions: tuple[GroundedValue, ...] = ()
    def __post_init__(self) -> None:
        _token("frame_type", self.frame_type); object.__setattr__(self, "modality", Modality(self.modality)); object.__setattr__(self, "participants", MappingProxyType(dict(self.participants)))
@dataclass(frozen=True, slots=True)
class InterpretationCandidate:
    candidate_id: str; dialogue_acts: tuple[DialogueAct, ...]; entity_mentions: tuple[EntityMention, ...]; frames: tuple[SemanticFrame, ...]; confidence: float | None = None; notes: tuple[GroundedValue, ...] = ()
    def __post_init__(self) -> None:
        _id("candidate_id", self.candidate_id)
        if self.confidence is not None and (not isinstance(self.confidence, float) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1): raise LanguageContractError("invalid confidence")
        if not self.dialogue_acts or len(self.entity_mentions) > MAX_MENTIONS or len(self.frames) > MAX_FRAMES: raise LanguageContractError("candidate outside bounds")
@dataclass(frozen=True, slots=True)
class AmbiguityReport: candidates: tuple[InterpretationCandidate, ...]; ambiguity_spans: tuple[SourceSpan, ...] = (); unresolved_dimensions: tuple[str, ...] = ()
@dataclass(frozen=True, slots=True)
class ClarificationRequest: request_id: str; question: GroundedValue; target_spans: tuple[SourceSpan, ...]
@dataclass(frozen=True, slots=True)
class StyleContract: style_id: str; locale: str = "und"; tone: str = "plain"; max_chars: int = 4000; required_citations: bool = True
@dataclass(frozen=True, slots=True)
class RealizationDraft: draft_id: str; utterance: UtteranceRef; interpretation_refs: tuple[str, ...]; style: StyleContract; realization_spans: tuple[SourceSpan, ...] = ()

def validate_provider_proposal(proposal: Mapping[str, object]) -> None:
    _scan_forbidden(proposal)

def from_runtime_utterance(utterance: object, utterance_id: str) -> UtteranceRef:
    text = getattr(utterance, "text")
    locale = getattr(utterance, "locale", "und")
    return UtteranceRef(utterance_id, text, locale=locale)

def from_runtime_proposal(utterance: UtteranceRef, proposal: object) -> AmbiguityReport:
    candidates = []
    for idx, candidate in enumerate(getattr(proposal, "candidates")):
        op = getattr(candidate, "operation")
        old_span = getattr(candidate, "span")
        span = SourceSpan(utterance.utterance_id, old_span.start, old_span.end, utterance.normalized_text[old_span.start:old_span.end])
        span.resolve(utterance)
        gv = GroundedValue(span.normalized_text.strip(), (span,))
        frame = SemanticFrame(op if op in {"arithmetic", "lookup", "unsupported"} else "request", gv, {"request_span": gv}, modality=Modality.REQUEST)
        act = DialogueAct(DialogueActKind.REQUEST, span)
        conf = getattr(candidate, "diagnostic_confidence", None)
        candidates.append(InterpretationCandidate(f"candidate:{idx}", (act,), (), (frame,), None if conf is None else float(conf)))
    return AmbiguityReport(tuple(candidates))

def external_provider_projection(utterance: UtteranceRef, spans: Sequence[SourceSpan]) -> Mapping[str, object]:
    safe = []
    for span in spans:
        span.resolve(utterance)
        text = span.normalized_text if span.privacy_label in {PrivacyLabel.PUBLIC, PrivacyLabel.INTERNAL} else "█" * (span.end - span.start)
        safe.append({"start": span.start, "end": span.end, "text": text, "privacy_label": span.privacy_label.value})
    return MappingProxyType({"schema_version": LANGUAGE_SCHEMA_VERSION, "utterance_digest": utterance.digest, "spans": tuple(safe)})

def content_digest(value: object) -> str: return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()

def _scan_forbidden(value: object) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str): raise LanguageContractError("provider keys must be strings")
            if unicodedata.normalize("NFC", k).lower() in FORBIDDEN_PROVIDER_FIELDS: raise ForbiddenProviderFieldError(f"forbidden provider field: {k}")
            _scan_forbidden(v)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value: _scan_forbidden(item)

def _norm_text(value: str) -> str:
    if not isinstance(value, str): raise LanguageContractError("expected text")
    out = unicodedata.normalize("NFC", value)
    if any(ord(ch) < 0x20 and ch not in "\n\t" for ch in out): raise LanguageContractError("control character rejected")
    return out

def _id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value): raise LanguageContractError(f"invalid {name}")
def _token(name: str, value: str) -> None:
    if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value): raise LanguageContractError(f"invalid {name}")
