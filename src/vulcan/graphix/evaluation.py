"""Canonical Graphix evaluation-to-epistemic candidate construction."""

from __future__ import annotations

from datetime import datetime

from vulcan.constitution.primitives import Digest
from vulcan.graphix import runtime as semantic
from vulcan.graphix.epistemic import (
    Citation,
    Claim,
    ClaimStatus,
    Derivation,
    EpistemicCommit,
    EvidenceArtifact,
    EvidenceKind,
    Proposition,
)
from vulcan.graphix.verifier import VerificationRequest
from vulcan.microkernel.transactions import CommandAuthority


def _digest(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


_STATUS = {
    semantic.EpistemicStatus.COMPUTED: ClaimStatus.COMPUTED,
    semantic.EpistemicStatus.RETRIEVED: ClaimStatus.RETRIEVED,
    semantic.EpistemicStatus.CONTESTED: ClaimStatus.CONTESTED,
    semantic.EpistemicStatus.UNKNOWN: ClaimStatus.UNKNOWN,
}

_EVIDENCE_KIND = {
    semantic.EvidenceKind.OBSERVATION: EvidenceKind.OBSERVATION,
    semantic.EvidenceKind.RETRIEVED_RECORD: EvidenceKind.RETRIEVAL,
    semantic.EvidenceKind.FORMAL_PREMISE: EvidenceKind.PROOF,
    semantic.EvidenceKind.SOURCE_DOCUMENT: EvidenceKind.CITATION,
    semantic.EvidenceKind.SOURCE_EXCERPT: EvidenceKind.CITATION,
}


def verify_evaluation_projection(
    commit: EpistemicCommit | None,
    *,
    claims: tuple[semantic.Claim, ...],
    evidence: tuple[semantic.EvidenceArtifact, ...],
    derivations: tuple[semantic.Derivation, ...],
) -> None:
    """Fail closed unless every compatibility object is digest-bound in ``commit``."""
    if commit is None:
        raise ValueError("durable epistemic head is unavailable")
    claim_bindings = {
        item.claim_id: item.proposition.qualifiers.get("semantic_claim_digest")
        for item in commit.claims
    }
    evidence_bindings = {
        item.evidence_id: item.provenance_id for item in commit.evidence
    }
    derivation_bindings = {
        item.derivation_id: item.rule_id for item in commit.derivations
    }
    if claim_bindings != {
        item.claim_id: _digest(semantic.canonical_digest(item)) for item in claims
    }:
        raise ValueError("runtime claim projection is not durably committed")
    if evidence_bindings != {
        item.artifact_id: f"provenance:{semantic.canonical_digest(item)}"
        for item in evidence
    }:
        raise ValueError("runtime evidence projection is not durably committed")
    if derivation_bindings != {
        item.derivation_id: f"rule:{semantic.canonical_digest(item)}"
        for item in derivations
    }:
        raise ValueError("runtime derivation projection is not durably committed")


def build_epistemic_candidate(
    *,
    episode_id: str,
    case_id: str,
    snapshot_digest: str,
    claims: tuple[semantic.Claim, ...],
    evidence: tuple[semantic.EvidenceArtifact, ...],
    derivations: tuple[semantic.Derivation, ...],
    authority: CommandAuthority,
    prior_commit_digest: str | None,
    graphix_artifact_digest: str,
    evaluated_at: datetime,
) -> EpistemicCommit:
    """Translate a validated request ledger without granting it authority."""
    semantic.validate_ledger(evidence, derivations, claims, case_id=case_id)
    unsupported = tuple(item.status for item in claims if item.status not in _STATUS)
    if unsupported:
        raise ValueError(
            "epistemic status has no registered Phase-B verifier: "
            + ",".join(item.value for item in unsupported)
        )
    snapshot = _digest(snapshot_digest)
    evidence_ids = {item.artifact_id for item in evidence}
    claim_ids = {item.claim_id for item in claims}
    graph_evidence = tuple(
        EvidenceArtifact(
            item.artifact_id,
            _EVIDENCE_KIND.get(item.kind, EvidenceKind.COMPUTATION),
            episode_id,
            snapshot,
            _digest(item.content_digest),
            f"provenance:{semantic.canonical_digest(item)}",
            item.observed_at or evaluated_at,
            item.valid_until,
            (
                (
                    Citation(
                        f"citation:{Digest.of_json({'evidence': item.artifact_id}).hex}",
                        artifact_id=item.artifact_id,
                        artifact_digest=_digest(item.content_digest),
                    ),
                )
                if item.kind is semantic.EvidenceKind.RETRIEVED_RECORD
                else ()
            ),
        )
        for item in evidence
    )
    graph_derivations = tuple(
        Derivation(
            item.derivation_id,
            tuple(ref for ref in item.inputs if ref in claim_ids),
            tuple(ref for ref in item.inputs if ref in evidence_ids),
            f"rule:{semantic.canonical_digest(item)}",
            item.output_claim_id,
        )
        for item in derivations
    )
    graph_claims = tuple(
        Claim(
            item.claim_id,
            Proposition(
                f"proposition:{Digest.of_json({'claim': item.claim_id}).hex}",
                item.proposition.subject,
                item.proposition.predicate,
                item.proposition.object,
                {
                    "semantic_claim_digest": _digest(semantic.canonical_digest(item)),
                    "graphix_artifact_digest": _digest(graphix_artifact_digest),
                    "expression_digest": (
                        None
                        if item.proposition.expression is None
                        else str(Digest.of_json(item.proposition.expression))
                    ),
                    "expression": item.proposition.expression,
                    "modality": item.proposition.modality,
                    "negated": item.proposition.negated,
                    "quantifier": item.proposition.quantifier,
                    "units": item.proposition.units,
                },
            ),
            _STATUS[item.status],
            episode_id,
            snapshot,
            item.evidence_ids,
            item.derivation_ids,
            contested_by=item.contradictions,
        )
        for item in claims
    )
    content = {
        "case": case_id,
        "claims": [semantic.canonical_digest(item) for item in claims],
        "derivations": [semantic.canonical_digest(item) for item in derivations],
        "evidence": [semantic.canonical_digest(item) for item in evidence],
        "prior": prior_commit_digest,
        "graphix_artifact_digest": _digest(graphix_artifact_digest),
    }
    return EpistemicCommit(
        commit_id=f"commit:{Digest.of_json(content).hex[:48]}",
        episode_id=episode_id,
        case_id=case_id,
        snapshot_digest=snapshot,
        authority_principal_id=f"principal:{authority.principal.identity_digest[:32]}",
        authority_release_digest=_digest(authority.principal.release_digest),
        validation_digest=_digest(authority.validation_digest),
        policy_digest=_digest(authority.policy_digest),
        authority_evidence_digest=_digest(authority.grant.evidence_digest),
        committed_at=evaluated_at,
        claims=graph_claims,
        evidence=graph_evidence,
        derivations=graph_derivations,
        prior_commit_digest=prior_commit_digest,
    )


def verification_requests(commit: EpistemicCommit) -> tuple[VerificationRequest, ...]:
    """Derive closed verifier inputs from the canonical candidate itself."""
    requests = []
    for claim in commit.claims:
        if claim.status is ClaimStatus.COMPUTED:
            expression = claim.proposition.qualifiers.get("expression")
            if not isinstance(expression, str) or not expression:
                raise ValueError("computed candidate lacks its exact expression")
            requests.append(
                VerificationRequest(
                    claim.claim_id,
                    claim.status,
                    claim.proposition.object_value,
                    operation_id="bounded-rational-expression/1",
                    operands=(expression,),
                )
            )
        elif claim.status is ClaimStatus.UNKNOWN:
            requests.append(VerificationRequest(claim.claim_id, claim.status, None))
        else:
            raise ValueError(
                f"{claim.status.value} requires admitted material verification"
            )
    return tuple(requests)
