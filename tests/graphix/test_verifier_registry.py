from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vulcan.constitution.primitives import canonical_json
from vulcan.graphix.epistemic import (
    Citation,
    Claim,
    ClaimStatus,
    EpistemicCommit,
    EvidenceArtifact,
    EvidenceKind,
    Proposition,
)
from vulcan.graphix.verifier import (
    DeonticState,
    EffectState,
    VerificationError,
    VerificationRequest,
    VerifiedEpistemicCandidate,
    VerifierRegistry,
    WarrantReceipt,
    phase_b_registry,
)

D = "sha256:" + "1" * 64
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _commit(status: ClaimStatus, value: str) -> EpistemicCommit:
    evidence = ()
    evidence_ids = ()
    if status in {ClaimStatus.RETRIEVED, ClaimStatus.PROVEN}:
        evidence = (
            EvidenceArtifact(
                "evidence:one",
                (
                    EvidenceKind.RETRIEVAL
                    if status is ClaimStatus.RETRIEVED
                    else EvidenceKind.PROOF
                ),
                "episode:one",
                D,
                D,
                "provenance:test",
                NOW,
                citations=(Citation("citation:one", uri="urn:test"),),
            ),
        )
        evidence_ids = ("evidence:one",)
    claim = Claim(
        "claim:one",
        Proposition("proposition:one", "calculation", "equals", value),
        status,
        "episode:one",
        D,
        evidence_ids=evidence_ids,
        contested_by=("claim:other",) if status is ClaimStatus.CONTESTED else (),
    )
    return EpistemicCommit(
        "commit:one",
        "episode:one",
        "episode:one",
        D,
        "principal:kernel",
        D,
        D,
        D,
        D,
        NOW,
        (claim,),
        evidence=evidence,
    )


def _registry() -> VerifierRegistry:
    return phase_b_registry(qualified_core_release=D)


def test_computation_is_reexecuted_and_cannot_be_invented() -> None:
    registry = _registry()
    commit = _commit(ClaimStatus.COMPUTED, "4")
    request = VerificationRequest(
        "claim:one",
        ClaimStatus.COMPUTED,
        "4",
        "integer.add/1",
        ("2", "2"),
        ("load:2", "load:2", "add:4"),
    )
    candidate = registry.issue(
        commit=commit, requests=(request,), admitted_at=NOW, context_digest=D
    )
    assert registry.validate(candidate) is commit
    with pytest.raises(VerificationError, match="reproduce"):
        registry.issue(
            commit=_commit(ClaimStatus.COMPUTED, "5"),
            requests=(replace(request, value="5"),),
            admitted_at=NOW,
            context_digest=D,
        )


def test_bounded_expression_is_independently_reexecuted() -> None:
    registry = _registry()
    commit = _commit(ClaimStatus.COMPUTED, "14")
    candidate = registry.issue(
        commit=commit,
        requests=(
            VerificationRequest(
                "claim:one",
                ClaimStatus.COMPUTED,
                "14",
                "bounded-rational-expression/1",
                ("2 + 3 * 4",),
            ),
        ),
        admitted_at=NOW,
        context_digest=D,
    )
    assert registry.validate(candidate) is commit
    raw = candidate.receipts[0].canonical_bytes()
    restarted = phase_b_registry(qualified_core_release=D)
    assert restarted.reverify_persisted(raw, commit=commit).receipt_digest
    with pytest.raises(VerificationError):
        restarted.reverify_persisted(raw.replace(b'"14"', b'"15"', 1), commit=commit)


def test_retrieval_resolves_exact_admitted_material() -> None:
    registry = _registry()
    commit = _commit(ClaimStatus.RETRIEVED, "Paris")
    request = VerificationRequest(
        "claim:one",
        ClaimStatus.RETRIEVED,
        "Paris",
        material=b'{"facts":{"capital":"Paris"}}',
        extraction_path=("facts", "capital"),
    )
    candidate = registry.issue(
        commit=commit, requests=(request,), admitted_at=NOW, context_digest=D
    )
    assert registry.validate(candidate) is commit
    assert (
        _registry()
        .reverify_persisted(candidate.receipts[0].canonical_bytes(), commit=commit)
        .status
        is ClaimStatus.RETRIEVED
    )
    with pytest.raises(VerificationError, match="differs"):
        registry.issue(
            commit=commit,
            requests=(replace(request, material=b'{"facts":{"capital":"Lyon"}}'),),
            admitted_at=NOW,
            context_digest=D,
        )
    with pytest.raises(VerificationError, match="canonical JSON"):
        registry.issue(
            commit=commit,
            requests=(replace(request, material=b'{"facts": {"capital":"Paris"}}'),),
            admitted_at=NOW,
            context_digest=D,
        )


@pytest.mark.parametrize(
    "status",
    [
        ClaimStatus.PROVEN,
        ClaimStatus.DISPROVEN,
        ClaimStatus.OBSERVED,
        ClaimStatus.ESTIMATED,
    ],
)
def test_unimplemented_statuses_fail_closed(status: ClaimStatus) -> None:
    with pytest.raises(VerificationError):
        _registry().issue(
            commit=_commit(ClaimStatus.UNKNOWN, "unknown"),
            requests=(VerificationRequest("claim:one", status, "value"),),
            admitted_at=NOW,
            context_digest=D,
        )


def test_unknown_and_contested_choose_no_value() -> None:
    registry = _registry()
    unknown = _commit(ClaimStatus.UNKNOWN, "unknown")
    unknown_candidate = registry.issue(
        commit=unknown,
        requests=(VerificationRequest("claim:one", ClaimStatus.UNKNOWN, None),),
        admitted_at=NOW,
        context_digest=D,
    )
    registry.validate(unknown_candidate)
    _registry().reverify_persisted(
        unknown_candidate.receipts[0].canonical_bytes(), commit=unknown
    )
    contested = _commit(ClaimStatus.CONTESTED, "unavailable")
    contested_candidate = registry.issue(
        commit=contested,
        requests=(
            VerificationRequest(
                "claim:one",
                ClaimStatus.CONTESTED,
                None,
                supported_values=(("Paris", D), ("Lyon", "sha256:" + "2" * 64)),
            ),
        ),
        admitted_at=NOW,
        context_digest=D,
    )
    registry.validate(contested_candidate)
    _registry().reverify_persisted(
        contested_candidate.receipts[0].canonical_bytes(), commit=contested
    )
    with pytest.raises(VerificationError, match="independent"):
        registry.issue(
            commit=contested,
            requests=(
                VerificationRequest(
                    "claim:one",
                    ClaimStatus.CONTESTED,
                    None,
                    supported_values=(("Paris", D), ("Lyon", D)),
                ),
            ),
            admitted_at=NOW,
            context_digest=D,
        )


def test_duplicate_warrant_requests_fail_closed() -> None:
    registry = _registry()
    commit = _commit(ClaimStatus.UNKNOWN, "unknown")
    request = VerificationRequest("claim:one", ClaimStatus.UNKNOWN, None)
    with pytest.raises(VerificationError, match="duplicate"):
        registry.issue(
            commit=commit,
            requests=(request, request),
            admitted_at=NOW,
            context_digest=D,
        )


def test_structural_lookalike_and_coordinate_substitution_are_rejected() -> None:
    registry = _registry()
    commit = _commit(ClaimStatus.UNKNOWN, "unknown")
    candidate = registry.issue(
        commit=commit,
        requests=(VerificationRequest("claim:one", ClaimStatus.UNKNOWN, None),),
        admitted_at=NOW,
        context_digest=D,
    )
    receipt = candidate.receipts[0]
    lookalike = replace(receipt, _issuer=object())
    with pytest.raises(VerificationError, match="not issued"):
        registry.validate(replace(candidate, receipts=(lookalike,)))
    with pytest.raises(VerificationError, match="publication"):
        registry.validate(replace(candidate, publication=DeonticState.AUTHORIZED))
    with pytest.raises(VerificationError, match="effect"):
        registry.validate(replace(candidate, effect=EffectState.AUTHORIZED))
    with pytest.raises(TypeError, match="privately verified"):
        registry.validate(commit)  # type: ignore[arg-type]
    assert not hasattr(registry, "register")
    with pytest.raises(TypeError, match="qualified core"):
        VerifierRegistry(qualified_core_release=D, evaluators={})


def test_receipt_from_another_release_or_registry_is_not_interchangeable() -> None:
    first, second = _registry(), _registry()
    commit = _commit(ClaimStatus.UNKNOWN, "unknown")
    candidate = first.issue(
        commit=commit,
        requests=(VerificationRequest("claim:one", ClaimStatus.UNKNOWN, None),),
        admitted_at=NOW,
        context_digest=D,
    )
    with pytest.raises(VerificationError, match="not issued"):
        second.validate(candidate)
    receipt = candidate.receipts[0]
    forged = WarrantReceipt(
        *[getattr(receipt, field) for field in receipt.__dataclass_fields__]
    )
    with pytest.raises(VerificationError, match="not issued"):
        first.validate(VerifiedEpistemicCandidate(commit, (forged,)))
    substituted = _commit(ClaimStatus.UNKNOWN, "unavailable")
    with pytest.raises(VerificationError, match="not issued"):
        first.validate(replace(candidate, commit=substituted))


def test_every_durable_receipt_binding_is_mutation_sensitive() -> None:
    registry = _registry()
    commit = _commit(ClaimStatus.UNKNOWN, "unknown")
    candidate = registry.issue(
        commit=commit,
        requests=(VerificationRequest("claim:one", ClaimStatus.UNKNOWN, None),),
        admitted_at=NOW,
        context_digest=D,
    )
    receipt = candidate.receipts[0]
    mutations = {
        "schema_version": "graphix.warrant-receipt/999",
        "registry_digest": "sha256:" + "2" * 64,
        "qualified_core_release": "sha256:" + "2" * 64,
        "verifier_id": "unknown:caller",
        "verifier_release": "phase-b/999",
        "episode_id": "episode:two",
        "snapshot_digest": "sha256:" + "2" * 64,
        "admitted_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "candidate_digest": "sha256:" + "2" * 64,
        "claim_id": "claim:two",
        "status": ClaimStatus.CONTESTED,
        "value_digest": D,
        "evidence_digest": "sha256:" + "2" * 64,
        "trace_digest": "sha256:" + "2" * 64,
        "receipt_digest": "sha256:" + "2" * 64,
        "_issuer": object(),
    }
    for field, value in mutations.items():
        with pytest.raises(VerificationError):
            registry.validate(
                replace(candidate, receipts=(replace(receipt, **{field: value}),))
            )
    document = json.loads(receipt.canonical_bytes())
    for field, value in (
        ("evidence_material", {"value": "leak"}),
        ("trace_material", ["invented"]),
    ):
        mutated = canonical_json({**document, field: value})
        with pytest.raises(VerificationError):
            _registry().reverify_persisted(mutated, commit=commit)


def test_emitted_registry_and_mutation_matrices_are_closed() -> None:
    declaration = json.loads(Path("config/phase-b-verifier-registry.json").read_text())
    assert declaration["registry_schema"] == "graphix.verifier-registry/1"
    assert declaration["runtime_registration"] is False
    assert declaration["registered_operations"] == [
        "bounded-rational-expression/1",
        "integer.add/1",
    ]
    assert tuple(sorted(declaration["supported"])) == (
        "COMPUTED",
        "CONTESTED",
        "RETRIEVED",
        "UNKNOWN",
    )
    assert declaration["unsupported"] == [
        "PROVEN",
        "DISPROVEN",
        "OBSERVED",
        "ESTIMATED",
    ]
    matrices = json.loads(Path("config/phase-b-verifier-matrices.json").read_text())
    assert matrices["schema"] == "graphix.verifier-matrices/1"
    assert set(matrices["mutation_matrix"]) == {
        "registry_digest",
        "qualified_core_release",
        "verifier_id",
        "episode_id",
        "snapshot_digest",
        "admitted_at",
        "candidate_digest",
        "claim_id",
        "status",
        "value_digest",
        "evidence_digest",
        "evidence_material",
        "trace_digest",
        "trace_material",
        "receipt_digest",
        "issuer_identity",
        "schema_version",
        "verifier_release",
    }
