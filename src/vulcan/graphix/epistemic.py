"""Graphix Epistemic v1: typed claims, evidence, derivations, and commits."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from vulcan.graphix.codec import canonical_json, validate_json_value
from vulcan.graphix.core import GraphixCoreError

EPISTEMIC_SCHEMA_VERSION = "graphix.epistemic/1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_REFS = 32

class EpistemicContractError(GraphixCoreError): pass
class ReferenceValidationError(EpistemicContractError): pass
class AuthorityValidationError(EpistemicContractError): pass
class EvidenceIntegrityError(EpistemicContractError): pass
class TemporalValidityError(EpistemicContractError): pass
class CircularDerivationError(EpistemicContractError): pass

class ClaimStatus(str, Enum):
    PROVEN="PROVEN"; DISPROVEN="DISPROVEN"; COMPUTED="COMPUTED"; OBSERVED="OBSERVED"; RETRIEVED="RETRIEVED"; ESTIMATED="ESTIMATED"; HYPOTHESIS="HYPOTHESIS"; CONTESTED="CONTESTED"; UNKNOWN="UNKNOWN"; ERROR="ERROR"
class EvidenceKind(str, Enum): PROOF="PROOF"; OBSERVATION="OBSERVATION"; RETRIEVAL="RETRIEVAL"; COMPUTATION="COMPUTATION"; CITATION="CITATION"; COUNTEREXAMPLE="COUNTEREXAMPLE"
class UncertaintyKind(str, Enum): UNKNOWN="UNKNOWN"; PROBABILITY_DISTRIBUTION="PROBABILITY_DISTRIBUTION"; INTERVAL="INTERVAL"; CALIBRATION_IDENTITY="CALIBRATION_IDENTITY"

@dataclass(frozen=True, slots=True)
class Proposition:
    proposition_id: str; subject: str; predicate: str; object_value: str; qualifiers: Mapping[str, object] = field(default_factory=dict)
    def __post_init__(self) -> None:
        _id("proposition_id", self.proposition_id)
        for n in ("subject","predicate","object_value"):
            if not (1 <= len(getattr(self,n)) <= 512): raise EpistemicContractError(f"invalid {n}")
        validate_json_value(self.qualifiers); object.__setattr__(self,"qualifiers",MappingProxyType(dict(self.qualifiers)))

@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str; uri: str | None = None; title: str | None = None; artifact_id: str | None = None; artifact_digest: str | None = None
    def __post_init__(self) -> None:
        _id("citation_id", self.citation_id)
        if self.uri is None and self.artifact_id is None: raise EpistemicContractError("citation requires uri or artifact_id")
        if self.artifact_id is not None: _id("citation.artifact_id", self.artifact_id)
        if self.artifact_digest is not None: _digest("citation.artifact_digest", self.artifact_digest)

@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    evidence_id: str; kind: EvidenceKind; episode_id: str; snapshot_digest: str; content_digest: str; provenance_id: str; observed_at: datetime; valid_until: datetime | None = None; citations: tuple[Citation,...] = (); source_episode_id: str | None = None
    def __post_init__(self) -> None:
        _id("evidence_id", self.evidence_id); _id("episode_id", self.episode_id); _digest("snapshot_digest", self.snapshot_digest); _digest("content_digest", self.content_digest); _id("provenance_id", self.provenance_id)
        object.__setattr__(self,"kind",EvidenceKind(self.kind)); _aware(self.observed_at,"observed_at")
        if self.valid_until is not None: _aware(self.valid_until,"valid_until")
        if self.source_episode_id is not None and self.source_episode_id == self.episode_id: raise ReferenceValidationError("source_episode_id is only for cross-episode evidence reuse")
        object.__setattr__(self,"citations",tuple(self.citations))

@dataclass(frozen=True, slots=True)
class UncertaintyDescriptor:
    kind: UncertaintyKind; distribution_digest: str | None = None; interval_low: str | None = None; interval_high: str | None = None; calibration_id: str | None = None
    def __post_init__(self) -> None:
        object.__setattr__(self,"kind",UncertaintyKind(self.kind))
        if self.distribution_digest: _digest("distribution_digest", self.distribution_digest)
        if self.kind is UncertaintyKind.PROBABILITY_DISTRIBUTION and not self.distribution_digest: raise EpistemicContractError("distribution requires digest")
        if self.kind is UncertaintyKind.INTERVAL and (self.interval_low is None or self.interval_high is None): raise EpistemicContractError("interval requires bounds")
        if self.kind is UncertaintyKind.CALIBRATION_IDENTITY and not self.calibration_id: raise EpistemicContractError("calibration requires identity")

@dataclass(frozen=True, slots=True)
class Assumption: assumption_id: str; proposition_id: str
@dataclass(frozen=True, slots=True)
class Counterexample: counterexample_id: str; evidence_id: str; target_claim_id: str
@dataclass(frozen=True, slots=True)
class Contradiction: contradiction_id: str; claim_ids: tuple[str, ...]
@dataclass(frozen=True, slots=True)
class Limitation: limitation_id: str; description: str; affected_claim_ids: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class Derivation:
    derivation_id: str; input_claim_ids: tuple[str,...]; evidence_ids: tuple[str,...]; rule_id: str; output_claim_id: str
    def __post_init__(self) -> None:
        _id("derivation_id", self.derivation_id); _id("rule_id", self.rule_id); _id("output_claim_id", self.output_claim_id)
        object.__setattr__(self,"input_claim_ids",tuple(self.input_claim_ids)); object.__setattr__(self,"evidence_ids",tuple(self.evidence_ids))
        if self.output_claim_id in self.input_claim_ids: raise CircularDerivationError("derivation cannot directly depend on its output")

@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str; proposition: Proposition; status: ClaimStatus; episode_id: str; snapshot_digest: str; evidence_ids: tuple[str,...] = (); derivation_ids: tuple[str,...] = (); uncertainty: UncertaintyDescriptor = field(default_factory=lambda: UncertaintyDescriptor(UncertaintyKind.UNKNOWN)); contested_by: tuple[str,...] = (); limitations: tuple[Limitation,...] = ()
    def __post_init__(self) -> None:
        _id("claim_id", self.claim_id); _id("episode_id", self.episode_id); _digest("snapshot_digest", self.snapshot_digest); object.__setattr__(self,"status",ClaimStatus(self.status))
        object.__setattr__(self,"evidence_ids",tuple(self.evidence_ids)); object.__setattr__(self,"derivation_ids",tuple(self.derivation_ids)); object.__setattr__(self,"contested_by",tuple(self.contested_by))

@dataclass(frozen=True, slots=True)
class EpistemicCommit:
    commit_id: str; episode_id: str; case_id: str; snapshot_digest: str; authority_principal_id: str; committed_at: datetime; claims: tuple[Claim,...]; evidence: tuple[EvidenceArtifact,...] = (); derivations: tuple[Derivation,...] = (); assumptions: tuple[Assumption,...] = (); counterexamples: tuple[Counterexample,...] = (); contradictions: tuple[Contradiction,...] = (); prior_commit_digest: str | None = None; commit_digest: str = ""
    def __post_init__(self) -> None:
        _id("commit_id", self.commit_id); _id("episode_id", self.episode_id); _id("case_id", self.case_id); _digest("snapshot_digest", self.snapshot_digest); _id("authority_principal_id", self.authority_principal_id); _aware(self.committed_at,"committed_at")
        if self.prior_commit_digest: _digest("prior_commit_digest", self.prior_commit_digest)
        for name in ("claims","evidence","derivations","assumptions","counterexamples","contradictions"): object.__setattr__(self,name,tuple(getattr(self,name)))
        _validate_commit(self)
        expected = digest_commit(self, include_digest=False)
        object.__setattr__(self,"commit_digest", self.commit_digest or expected)
        if self.commit_digest != expected: raise EvidenceIntegrityError("commit digest mismatch")

def _validate_commit(c: EpistemicCommit) -> None:
    if not c.claims: raise EpistemicContractError("commit requires at least one claim")
    ev = {e.evidence_id:e for e in c.evidence}; claims={cl.claim_id:cl for cl in c.claims}; deriv={d.derivation_id:d for d in c.derivations}
    if len(ev)!=len(c.evidence) or len(claims)!=len(c.claims) or len(deriv)!=len(c.derivations): raise ReferenceValidationError("duplicate ids")
    for e in c.evidence:
        if e.episode_id != c.episode_id and e.source_episode_id is None: raise ReferenceValidationError("cross-episode evidence requires explicit source_episode_id")
        if e.snapshot_digest != c.snapshot_digest: raise ReferenceValidationError("evidence snapshot mismatch")
        if e.valid_until is not None and e.valid_until <= c.committed_at: raise TemporalValidityError("expired evidence")
    for cl in c.claims:
        if cl.episode_id != c.episode_id or cl.snapshot_digest != c.snapshot_digest: raise ReferenceValidationError("claim binding mismatch")
        for i in cl.evidence_ids:
            if i not in ev: raise ReferenceValidationError("dangling evidence reference")
        for i in cl.derivation_ids:
            if i not in deriv: raise ReferenceValidationError("dangling derivation reference")
        if cl.status is ClaimStatus.PROVEN and not any(ev[i].kind is EvidenceKind.PROOF for i in cl.evidence_ids): raise EvidenceIntegrityError("PROVEN claim requires proof artifact")
        if cl.status is ClaimStatus.RETRIEVED and (not cl.evidence_ids or not any(ev[i].citations for i in cl.evidence_ids)): raise EvidenceIntegrityError("RETRIEVED claim requires evidence citation")
        if cl.status is ClaimStatus.CONTESTED and not cl.contested_by: raise ReferenceValidationError("CONTESTED claim requires contested_by")
    for d in c.derivations:
        if d.output_claim_id not in claims: raise ReferenceValidationError("dangling derivation output")
        for i in d.input_claim_ids:
            if i not in claims: raise ReferenceValidationError("dangling derivation input")
        for i in d.evidence_ids:
            if i not in ev: raise ReferenceValidationError("dangling derivation evidence")
    _detect_cycles(c.derivations)

def _detect_cycles(derivations: Sequence[Derivation]) -> None:
    graph={d.output_claim_id:set(d.input_claim_ids) for d in derivations}; visiting=set(); seen=set()
    def visit(n: str) -> None:
        if n in visiting: raise CircularDerivationError("circular derivation")
        if n in seen: return
        visiting.add(n)
        for m in graph.get(n, ()): visit(m)
        visiting.remove(n); seen.add(n)
    for n in graph: visit(n)

def commit_to_dict(c: EpistemicCommit, *, include_digest: bool=True) -> dict[str, object]:
    def dt(x): return x.astimezone(timezone.utc).isoformat().replace("+00:00","Z")
    out={"schema_version":EPISTEMIC_SCHEMA_VERSION,"commit_id":c.commit_id,"episode_id":c.episode_id,"case_id":c.case_id,"snapshot_digest":c.snapshot_digest,"authority_principal_id":c.authority_principal_id,"committed_at":dt(c.committed_at),"prior_commit_digest":c.prior_commit_digest,"claims":[{"claim_id":cl.claim_id,"proposition_id":cl.proposition.proposition_id,"status":cl.status.value,"episode_id":cl.episode_id,"snapshot_digest":cl.snapshot_digest,"evidence_ids":list(cl.evidence_ids),"derivation_ids":list(cl.derivation_ids),"contested_by":list(cl.contested_by)} for cl in c.claims],"evidence":[{"evidence_id":e.evidence_id,"kind":e.kind.value,"episode_id":e.episode_id,"snapshot_digest":e.snapshot_digest,"content_digest":e.content_digest,"provenance_id":e.provenance_id,"observed_at":dt(e.observed_at),"valid_until":None if e.valid_until is None else dt(e.valid_until),"citations":[ci.citation_id for ci in e.citations],"source_episode_id":e.source_episode_id} for e in c.evidence],"derivations":[{"derivation_id":d.derivation_id,"input_claim_ids":list(d.input_claim_ids),"evidence_ids":list(d.evidence_ids),"rule_id":d.rule_id,"output_claim_id":d.output_claim_id} for d in c.derivations]}
    if include_digest: out["commit_digest"]=c.commit_digest
    return out

def digest_commit(c: EpistemicCommit, *, include_digest: bool=True) -> str: return "sha256:"+hashlib.sha256(canonical_json(commit_to_dict(c, include_digest=include_digest))).hexdigest()
def content_digest(value: object) -> str: return "sha256:"+hashlib.sha256(canonical_json(value)).hexdigest()
def project_semantic_claim(*, claim_id: str, episode_id: str, snapshot_digest: str, subject: str, predicate: str, object_value: str, evidence_id: str | None = None) -> Claim:
    return Claim(claim_id, Proposition("prop:"+claim_id.split(":")[-1], subject, predicate, object_value), ClaimStatus.OBSERVED if evidence_id else ClaimStatus.HYPOTHESIS, episode_id, snapshot_digest, () if evidence_id is None else (evidence_id,))
def _id(name: str, value: str) -> None:
    if not isinstance(value,str) or _ID_RE.fullmatch(value) is None: raise EpistemicContractError(f"invalid {name}")
def _digest(name: str, value: str) -> None:
    if not isinstance(value,str) or _DIGEST_RE.fullmatch(value) is None: raise EvidenceIntegrityError(f"invalid {name}")
def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None: raise TemporalValidityError(f"{name} must be timezone-aware")
