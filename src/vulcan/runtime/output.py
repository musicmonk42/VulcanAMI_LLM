"""Capability-minimized fluent rendering adapter and conservative firewall.

Fluent rendering is disabled by default.  If enabled by a caller, this module
accepts only structured references to already validated IR/ledger content;
unreferenced prose is not an allowed wire feature.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .semantic import Claim, EvidenceArtifact, ResponseIR

@dataclass(frozen=True)
class ProjectedClaim:
    claim_id: str
    text: str
    status: str
    caveat: str | None
    citation_ids: tuple[str, ...]
@dataclass(frozen=True)
class ResponseIRProjection:
    response_id: str
    locale: str
    max_chars: int
    required_claim_ids: tuple[str, ...]
    claims: tuple[ProjectedClaim, ...]
    # No utterance, request, history, case, world-state, tool, memory, or finalizer capability is present.
@dataclass(frozen=True)
class DraftSegment:
    kind: str  # only claim or citation
    reference_id: str
@dataclass(frozen=True)
class UntrustedRenderDraft:
    schema_version: str
    segments: tuple[DraftSegment, ...]
class LanguageOutputPort(Protocol):
    async def render(self, projection: ResponseIRProjection) -> UntrustedRenderDraft: ...
@dataclass(frozen=True)
class FirewallResult:
    accepted: bool
    findings: tuple[str, ...]

def project(ir: ResponseIR, claims: tuple[Claim, ...]) -> ResponseIRProjection:
    indexed = {claim.claim_id: claim for claim in claims}
    if not set(ir.required_claim_ids) <= set(indexed):
        raise ValueError("projection references unknown claim")
    selected = tuple(indexed[identifier] for identifier in ir.required_claim_ids)
    return ResponseIRProjection(ir.response_id, ir.locale, ir.max_chars, ir.required_claim_ids, tuple(ProjectedClaim(c.claim_id, c.proposition.object, c.status.value, c.caveat, c.citation_ids) for c in selected))

class SemanticFirewall:
    """Accept only an exact, ordered structural realization of the projection."""
    def validate(self, projection: ResponseIRProjection, draft: UntrustedRenderDraft) -> FirewallResult:
        if draft.schema_version != "untrusted-render/1":
            return FirewallResult(False, ("unsupported draft schema",))
        findings: list[str] = []
        claim_ids = tuple(segment.reference_id for segment in draft.segments if segment.kind == "claim")
        if any(segment.kind not in {"claim", "citation"} for segment in draft.segments):
            findings.append("unrecognized segment kind")
        if claim_ids != projection.required_claim_ids:
            findings.append("required claim coverage or order changed")
        known_claims = {claim.claim_id for claim in projection.claims}
        known_citations = {citation for claim in projection.claims for citation in claim.citation_ids}
        for segment in draft.segments:
            if segment.kind == "claim" and segment.reference_id not in known_claims:
                findings.append("unknown claim reference")
            if segment.kind == "citation" and segment.reference_id not in known_citations:
                findings.append("unknown citation reference")
        if len(draft.segments) > len(projection.required_claim_ids) + len(known_citations):
            findings.append("draft exceeds structural bounds")
        return FirewallResult(not findings, tuple(findings))
