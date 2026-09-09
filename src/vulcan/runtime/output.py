"""The one canonical, capability-minimized output-language contract.

The supported surface is ``und`` strict rendering of bounded arithmetic,
unknown, and clarification results.  Drafts are references only: arbitrary
model prose is deliberately not representable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vulcan.graphix.epistemic import ClaimStatus, EpistemicCommit
from vulcan.microkernel.episode import CognitiveEpisode
from vulcan.microkernel.state_machine import EpisodeState

from .semantic import (
    Claim,
    EpistemicStatus,
    RenderArtifact,
    ResponseIR,
    ResponseMode,
    canonical_digest,
)

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
    episode_head_digest: str = ""
    epistemic_head_digest: str = ""
    snapshot_digest: str = ""
    accepted_interpretation_id: str | None = None
    accepted_plan_ids: tuple[str, ...] = ()


# Removal condition: delete this alias when output adapters have migrated from
# the response-ir.v3 name to the durable response-projection.v1 contract.
ResponseProjection = ResponseIRProjection


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
    async def render(
        self, projection: ResponseIRProjection
    ) -> UntrustedRenderDraft: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FirewallResult:
    accepted: bool
    findings: tuple[str, ...]


def project(ir: ResponseIR, claims: tuple[Claim, ...]) -> ResponseIRProjection:
    """Minimize a validated ledger to the closed strict-rendering surface."""
    if ir.locale not in SUPPORTED_LOCALES or ir.mode not in {
        ResponseMode.STRICT,
        ResponseMode.UNKNOWN,
        ResponseMode.CLARIFICATION,
        ResponseMode.ERROR,
    }:
        raise ValueError("unsupported output locale or mode")
    indexed = {claim.claim_id: claim for claim in claims}
    if len(indexed) != len(claims) or not set(ir.required_claim_ids) <= set(indexed):
        raise ValueError("projection references unknown claim")
    selected: list[ProjectedClaim] = []
    for claim_id in ir.required_claim_ids:
        claim = indexed[claim_id]
        if claim.status is EpistemicStatus.COMPUTED:
            selected.append(
                ProjectedClaim(
                    claim_id,
                    "computed",
                    claim.proposition.object,
                    claim.status,
                    claim.caveat,
                    claim.citation_ids,
                )
            )
        elif claim.status is EpistemicStatus.RETRIEVED:
            selected.append(
                ProjectedClaim(
                    claim_id,
                    "retrieved",
                    claim.proposition.object,
                    claim.status,
                    claim.caveat,
                    claim.citation_ids,
                )
            )
        elif claim.status in {EpistemicStatus.UNKNOWN, EpistemicStatus.ERROR}:
            selected.append(
                ProjectedClaim(
                    claim_id,
                    claim.status.value,
                    None,
                    claim.status,
                    claim.caveat,
                    claim.citation_ids,
                )
            )
        else:
            raise ValueError("claim is outside strict rendering surface")
    return ResponseIRProjection(
        ir.response_id,
        ir.locale,
        ir.max_chars,
        ir.mode,
        ir.required_claim_ids,
        tuple(selected),
    )


def project_committed(
    ir: ResponseIR, episode: CognitiveEpisode, epistemic: EpistemicCommit
) -> ResponseProjection:
    """Build the only renderable projection from matching durable heads."""
    if episode.state is not EpisodeState.EPISTEMICALLY_COMMITTED:
        raise ValueError("projection requires the durable epistemic episode head")
    if (
        episode.snapshot_bundle is None
        or ir.state_snapshot_id != episode.snapshot_bundle.state_digest
    ):
        raise ValueError("response projection snapshot mismatch")
    if epistemic.episode_id != episode.episode_id or epistemic.case_id != ir.case_id:
        raise ValueError("response projection episode mismatch")
    if (
        epistemic.snapshot_digest.removeprefix("sha256:")
        != episode.snapshot_bundle.state_digest
    ):
        raise ValueError("epistemic projection snapshot mismatch")
    if {item.artifact_id for item in episode.claims} != {
        item.claim_id for item in epistemic.claims
    }:
        raise ValueError("episode and epistemic claim heads diverged")
    indexed = {item.claim_id: item for item in epistemic.claims}
    if not set(ir.required_claim_ids) <= set(indexed):
        raise ValueError("projection references an uncommitted claim")
    status_map = {
        ClaimStatus.COMPUTED: EpistemicStatus.COMPUTED,
        ClaimStatus.RETRIEVED: EpistemicStatus.RETRIEVED,
        ClaimStatus.UNKNOWN: EpistemicStatus.UNKNOWN,
        ClaimStatus.ERROR: EpistemicStatus.ERROR,
        ClaimStatus.PROVEN: EpistemicStatus.PROVEN,
    }
    projected = []
    for claim_id in ir.required_claim_ids:
        claim = indexed[claim_id]
        if claim.status not in status_map:
            raise ValueError("committed claim is outside the rendering surface")
        citations = tuple(
            citation.citation_id
            for evidence in epistemic.evidence
            if evidence.evidence_id in claim.evidence_ids
            for citation in evidence.citations
        )
        projected.append(
            ProjectedClaim(
                claim_id,
                claim.status.value.lower(),
                (
                    None
                    if claim.status in {ClaimStatus.UNKNOWN, ClaimStatus.ERROR}
                    else claim.proposition.object_value
                ),
                status_map[claim.status],
                None,
                citations,
            )
        )
    return ResponseProjection(
        ir.response_id,
        ir.locale,
        ir.max_chars,
        ir.mode,
        ir.required_claim_ids,
        tuple(projected),
        episode.digest,
        epistemic.commit_digest,
        episode.snapshot_bundle.state_digest,
        ir.accepted_interpretation_id,
        tuple(item.artifact_id for item in episode.candidate_plans),
    )


def render_projection(projection: ResponseProjection) -> RenderArtifact:
    """Deterministically realize only values carried by the committed projection."""
    import html

    if not projection.episode_head_digest or not projection.epistemic_head_digest:
        raise ValueError("durable response projection is required")
    parts = []
    citations = []
    for claim in projection.claims:
        if claim.status is EpistemicStatus.COMPUTED:
            value = f"The computed result is {claim.value}."
        elif claim.status is EpistemicStatus.UNKNOWN:
            value = "This request is not supported by the deterministic interpreter."
        elif claim.status is EpistemicStatus.ERROR:
            value = "The deterministic interpreter could not complete this request."
        else:
            value = f"{claim.status.value.capitalize()}: {claim.value}."
        if claim.caveat:
            value += f" Caveat: {claim.caveat}"
        parts.append(html.escape(value, quote=False))
        citations.extend(claim.citation_ids)
    text = "\n".join(parts)
    if len(text) > projection.max_chars:
        raise ValueError("render bound")
    projection_digest = canonical_digest(projection)
    return RenderArtifact(
        text,
        "committed-strict-template",
        "1",
        projection_digest,
        projection.required_claim_ids,
        tuple(citations),
        projection.locale,
    )


class DeterministicLanguageOutput:
    """Reference-only adapter used by deterministic-only deployments."""

    identity = "deterministic-strict-output/1"

    async def render(self, projection: ResponseIRProjection) -> UntrustedRenderDraft:
        segments: list[DraftSegment] = []
        for claim in projection.claims:
            segments.append(DraftSegment("claim", claim.claim_id))
            if claim.caveat:
                segments.append(DraftSegment("caveat", claim.claim_id))
            segments.extend(
                DraftSegment("citation", citation) for citation in claim.citation_ids
            )
        return UntrustedRenderDraft(OUTPUT_DRAFT_SCHEMA, self.identity, tuple(segments))

    def close(self) -> None:
        return None


class SemanticFirewall:
    """Prove a draft realizes precisely the server-owned projection references."""

    def validate(
        self, projection: ResponseIRProjection, draft: UntrustedRenderDraft
    ) -> FirewallResult:
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
            expected.extend(
                DraftSegment("citation", citation) for citation in claim.citation_ids
            )
        if draft.segments != tuple(expected):
            findings.append("claim, caveat, or citation coverage changed")
        if len(draft.segments) > len(expected):
            findings.append("draft exceeds structural bounds")
        return FirewallResult(not findings, tuple(findings))
