"""The one canonical, capability-minimized output-language contract.

The supported surface is ``und`` strict rendering of bounded arithmetic,
unknown, and clarification results.  Drafts are references only: arbitrary
model prose is deliberately not representable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .semantic import Claim, EpistemicStatus, ResponseIR, ResponseMode

OUTPUT_DRAFT_SCHEMA = "untrusted-render/1"
SUPPORTED_LOCALES = frozenset({"und"})


@dataclass(frozen=True)
class ProjectedClaim:
    claim_id: str
    variant: str
    value: str | None
    status: EpistemicStatus
    caveat: str | None
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResponseIRProjection:
    response_id: str
    locale: str
    max_chars: int
    mode: ResponseMode
    required_claim_ids: tuple[str, ...]
    claims: tuple[ProjectedClaim, ...]


@dataclass(frozen=True)
class DraftSegment:
    kind: str  # claim, caveat, or citation
    reference_id: str


@dataclass(frozen=True)
class UntrustedRenderDraft:
    schema_version: str
    adapter_identity: str
    segments: tuple[DraftSegment, ...]


class LanguageOutputPort(Protocol):
    async def render(self, projection: ResponseIRProjection) -> UntrustedRenderDraft: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FirewallResult:
    accepted: bool
    findings: tuple[str, ...]


def project(ir: ResponseIR, claims: tuple[Claim, ...]) -> ResponseIRProjection:
    """Minimize a validated ledger to the closed strict-rendering surface."""
    if ir.locale not in SUPPORTED_LOCALES or ir.mode not in {
        ResponseMode.STRICT, ResponseMode.UNKNOWN, ResponseMode.CLARIFICATION, ResponseMode.ERROR,
    }:
        raise ValueError("unsupported output locale or mode")
    indexed = {claim.claim_id: claim for claim in claims}
    if len(indexed) != len(claims) or not set(ir.required_claim_ids) <= set(indexed):
        raise ValueError("projection references unknown claim")
    selected: list[ProjectedClaim] = []
    for claim_id in ir.required_claim_ids:
        claim = indexed[claim_id]
        if claim.status is EpistemicStatus.COMPUTED:
            selected.append(ProjectedClaim(claim_id, "computed", claim.proposition.object, claim.status, claim.caveat, claim.citation_ids))
        elif claim.status in {EpistemicStatus.UNKNOWN, EpistemicStatus.ERROR}:
            selected.append(ProjectedClaim(claim_id, claim.status.value, None, claim.status, claim.caveat, claim.citation_ids))
        else:
            raise ValueError("claim is outside strict rendering surface")
    return ResponseIRProjection(ir.response_id, ir.locale, ir.max_chars, ir.mode, ir.required_claim_ids, tuple(selected))


class DeterministicLanguageOutput:
    """Reference-only adapter used by deterministic-only deployments."""
    identity = "deterministic-strict-output/1"

    async def render(self, projection: ResponseIRProjection) -> UntrustedRenderDraft:
        segments: list[DraftSegment] = []
        for claim in projection.claims:
            segments.append(DraftSegment("claim", claim.claim_id))
            if claim.caveat:
                segments.append(DraftSegment("caveat", claim.claim_id))
            segments.extend(DraftSegment("citation", citation) for citation in claim.citation_ids)
        return UntrustedRenderDraft(OUTPUT_DRAFT_SCHEMA, self.identity, tuple(segments))

    def close(self) -> None:
        return None


class SemanticFirewall:
    """Prove a draft realizes precisely the server-owned projection references."""

    def validate(self, projection: ResponseIRProjection, draft: UntrustedRenderDraft) -> FirewallResult:
        findings: list[str] = []
        if draft.schema_version != OUTPUT_DRAFT_SCHEMA:
            findings.append("unsupported draft schema")
        if not draft.adapter_identity or len(draft.adapter_identity) > 128:
            findings.append("invalid adapter identity")
        if projection.locale not in SUPPORTED_LOCALES or projection.max_chars <= 0:
            findings.append("unsupported projection bounds")
        expected: list[DraftSegment] = []
        for claim in projection.claims:
            expected.append(DraftSegment("claim", claim.claim_id))
            if claim.caveat:
                expected.append(DraftSegment("caveat", claim.claim_id))
            expected.extend(DraftSegment("citation", citation) for citation in claim.citation_ids)
        if draft.segments != tuple(expected):
            findings.append("claim, caveat, or citation coverage changed")
        if len(draft.segments) > len(expected):
            findings.append("draft exceeds structural bounds")
        return FirewallResult(not findings, tuple(findings))
