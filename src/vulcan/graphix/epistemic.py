"""Graphix Epistemic v1: typed claims, evidence, derivations, and commits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from vulcan.constitution.primitives import (
    ArtifactId,
    AuthorityLevel,
    CommitId,
    Digest,
    EpisodeId,
    PrincipalId,
    canonical_json_loads,
    canonical_timestamp,
    parse_timestamp,
    require_utc,
)
from vulcan.graphix.codec import canonical_json, validate_json_value
from vulcan.graphix.core import GraphixCoreError

EPISTEMIC_SCHEMA_VERSION = "graphix.epistemic/1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
MAX_REFS = 32


class EpistemicContractError(GraphixCoreError):
    pass


class ReferenceValidationError(EpistemicContractError):
    pass


class AuthorityValidationError(EpistemicContractError):
    pass


class EvidenceIntegrityError(EpistemicContractError):
    pass


class TemporalValidityError(EpistemicContractError):
    pass


class CircularDerivationError(EpistemicContractError):
    pass


class ClaimStatus(str, Enum):
    PROVEN = "PROVEN"
    DISPROVEN = "DISPROVEN"
    COMPUTED = "COMPUTED"
    OBSERVED = "OBSERVED"
    RETRIEVED = "RETRIEVED"
    ESTIMATED = "ESTIMATED"
    HYPOTHESIS = "HYPOTHESIS"
    CONTESTED = "CONTESTED"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class EvidenceKind(str, Enum):
    PROOF = "PROOF"
    OBSERVATION = "OBSERVATION"
    RETRIEVAL = "RETRIEVAL"
    COMPUTATION = "COMPUTATION"
    CITATION = "CITATION"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"


class UncertaintyKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    PROBABILITY_DISTRIBUTION = "PROBABILITY_DISTRIBUTION"
    INTERVAL = "INTERVAL"
    CALIBRATION_IDENTITY = "CALIBRATION_IDENTITY"


@dataclass(frozen=True, slots=True)
class Proposition:
    proposition_id: str
    subject: str
    predicate: str
    object_value: str
    qualifiers: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id("proposition_id", self.proposition_id)
        for n in ("subject", "predicate", "object_value"):
            if not (1 <= len(getattr(self, n)) <= 512):
                raise EpistemicContractError(f"invalid {n}")
        validate_json_value(self.qualifiers)
        object.__setattr__(
            self,
            "qualifiers",
            MappingProxyType(
                {key: _freeze_json(item) for key, item in self.qualifiers.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    uri: str | None = None
    title: str | None = None
    artifact_id: str | None = None
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        _id("citation_id", self.citation_id)
        if self.uri is None and self.artifact_id is None:
            raise EpistemicContractError("citation requires uri or artifact_id")
        if self.artifact_id is not None:
            _id("citation.artifact_id", self.artifact_id)
        if self.artifact_digest is not None:
            _digest("citation.artifact_digest", self.artifact_digest)


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    evidence_id: str
    kind: EvidenceKind
    episode_id: str
    snapshot_digest: str
    content_digest: str
    provenance_id: str
    observed_at: datetime
    valid_until: datetime | None = None
    citations: tuple[Citation, ...] = ()
    source_episode_id: str | None = None

    def __post_init__(self) -> None:
        _id("evidence_id", self.evidence_id)
        _id("episode_id", self.episode_id)
        _digest("snapshot_digest", self.snapshot_digest)
        _digest("content_digest", self.content_digest)
        _id("provenance_id", self.provenance_id)
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        _aware(self.observed_at, "observed_at")
        if self.valid_until is not None:
            _aware(self.valid_until, "valid_until")
        if (
            self.source_episode_id is not None
            and self.source_episode_id == self.episode_id
        ):
            raise ReferenceValidationError(
                "source_episode_id is only for cross-episode evidence reuse"
            )
        object.__setattr__(self, "citations", tuple(self.citations))
        _bounded("citations", self.citations)
        _unique("citation ids", (item.citation_id for item in self.citations))


@dataclass(frozen=True, slots=True)
class UncertaintyDescriptor:
    kind: UncertaintyKind
    distribution_digest: str | None = None
    interval_low: str | None = None
    interval_high: str | None = None
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", UncertaintyKind(self.kind))
        if self.distribution_digest:
            _digest("distribution_digest", self.distribution_digest)
        if (
            self.kind is UncertaintyKind.PROBABILITY_DISTRIBUTION
            and not self.distribution_digest
        ):
            raise EpistemicContractError("distribution requires digest")
        if self.kind is UncertaintyKind.INTERVAL and (
            self.interval_low is None or self.interval_high is None
        ):
            raise EpistemicContractError("interval requires bounds")
        if self.kind is UncertaintyKind.INTERVAL:
            try:
                low, high = Fraction(self.interval_low), Fraction(self.interval_high)
            except (ValueError, ZeroDivisionError, TypeError) as exc:
                raise EpistemicContractError(
                    "interval bounds must be finite rational numbers"
                ) from exc
            if low > high:
                raise EpistemicContractError("invalid or reversed interval")
        if (
            self.kind is UncertaintyKind.CALIBRATION_IDENTITY
            and not self.calibration_id
        ):
            raise EpistemicContractError("calibration requires identity")
        fields = (
            self.distribution_digest,
            self.interval_low,
            self.interval_high,
            self.calibration_id,
        )
        expected = {
            UncertaintyKind.UNKNOWN: (None, None, None, None),
            UncertaintyKind.PROBABILITY_DISTRIBUTION: (
                self.distribution_digest,
                None,
                None,
                None,
            ),
            UncertaintyKind.INTERVAL: (
                None,
                self.interval_low,
                self.interval_high,
                None,
            ),
            UncertaintyKind.CALIBRATION_IDENTITY: (
                None,
                None,
                None,
                self.calibration_id,
            ),
        }[self.kind]
        if fields != expected:
            raise EpistemicContractError("uncertainty fields do not match kind")


@dataclass(frozen=True, slots=True)
class Assumption:
    assumption_id: str
    proposition_id: str

    def __post_init__(self) -> None:
        _artifact_id("assumption_id", self.assumption_id)
        _artifact_id("assumption.proposition_id", self.proposition_id)


@dataclass(frozen=True, slots=True)
class Counterexample:
    counterexample_id: str
    evidence_id: str
    target_claim_id: str

    def __post_init__(self) -> None:
        _artifact_id("counterexample_id", self.counterexample_id)
        _artifact_id("counterexample.evidence_id", self.evidence_id)
        _artifact_id("counterexample.target_claim_id", self.target_claim_id)


@dataclass(frozen=True, slots=True)
class Contradiction:
    contradiction_id: str
    claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _artifact_id("contradiction_id", self.contradiction_id)
        object.__setattr__(self, "claim_ids", tuple(self.claim_ids))
        if len(self.claim_ids) < 2:
            raise ReferenceValidationError("contradiction requires at least two claims")
        for claim_id in self.claim_ids:
            _artifact_id("contradiction.claim_id", claim_id)


@dataclass(frozen=True, slots=True)
class Limitation:
    limitation_id: str
    description: str
    affected_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _artifact_id("limitation_id", self.limitation_id)
        if (
            not isinstance(self.description, str)
            or not 1 <= len(self.description) <= 1024
        ):
            raise EpistemicContractError("invalid limitation description")
        object.__setattr__(self, "affected_claim_ids", tuple(self.affected_claim_ids))
        for claim_id in self.affected_claim_ids:
            _artifact_id("limitation.affected_claim_id", claim_id)


@dataclass(frozen=True, slots=True)
class Derivation:
    derivation_id: str
    input_claim_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rule_id: str
    output_claim_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("derivation_id", self.derivation_id),
            ("rule_id", self.rule_id),
            ("output_claim_id", self.output_claim_id),
        ):
            _artifact_id(name, value)
        object.__setattr__(self, "input_claim_ids", tuple(self.input_claim_ids))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        _bounded("derivation input claims", self.input_claim_ids)
        _bounded("derivation evidence", self.evidence_ids)
        if self.output_claim_id in self.input_claim_ids:
            raise CircularDerivationError(
                "derivation cannot directly depend on its output"
            )


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    proposition: Proposition
    status: ClaimStatus
    episode_id: str
    snapshot_digest: str
    evidence_ids: tuple[str, ...] = ()
    derivation_ids: tuple[str, ...] = ()
    uncertainty: UncertaintyDescriptor = field(
        default_factory=lambda: UncertaintyDescriptor(UncertaintyKind.UNKNOWN)
    )
    contested_by: tuple[str, ...] = ()
    limitations: tuple[Limitation, ...] = ()
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _artifact_id("claim_id", self.claim_id)
        _episode_id("episode_id", self.episode_id)
        _digest("snapshot_digest", self.snapshot_digest)
        object.__setattr__(self, "status", ClaimStatus(self.status))
        for name in (
            "evidence_ids",
            "derivation_ids",
            "contested_by",
            "limitations",
            "supersedes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            _bounded(name, getattr(self, name))
        _unique(
            "claim limitation ids", (item.limitation_id for item in self.limitations)
        )


@dataclass(frozen=True, slots=True)
class EpistemicCommit:
    commit_id: str
    episode_id: str
    case_id: str
    snapshot_digest: str
    authority_principal_id: str
    authority_release_digest: str
    validation_digest: str
    policy_digest: str
    authority_evidence_digest: str
    committed_at: datetime
    claims: tuple[Claim, ...]
    evidence: tuple[EvidenceArtifact, ...] = ()
    derivations: tuple[Derivation, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    counterexamples: tuple[Counterexample, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    prior_commit_digest: str | None = None
    authority_level: AuthorityLevel = AuthorityLevel.COMMITTED_BELIEF
    commit_digest: str = ""

    def __post_init__(self) -> None:
        _commit_id("commit_id", self.commit_id)
        _episode_id("episode_id", self.episode_id)
        _episode_id("case_id", self.case_id)
        _digest("snapshot_digest", self.snapshot_digest)
        _principal_id("authority_principal_id", self.authority_principal_id)
        for name in (
            "authority_release_digest",
            "validation_digest",
            "policy_digest",
            "authority_evidence_digest",
        ):
            _digest(name, getattr(self, name))
        _aware(self.committed_at, "committed_at")
        if self.prior_commit_digest is not None:
            _digest("prior_commit_digest", self.prior_commit_digest)
        try:
            authority = AuthorityLevel(self.authority_level)
        except (TypeError, ValueError) as exc:
            raise AuthorityValidationError("invalid commit authority") from exc
        if authority is not AuthorityLevel.COMMITTED_BELIEF:
            raise AuthorityValidationError(
                "epistemic commit requires COMMITTED_BELIEF authority"
            )
        object.__setattr__(self, "authority_level", authority)
        for name in (
            "claims",
            "evidence",
            "derivations",
            "assumptions",
            "counterexamples",
            "contradictions",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _validate_commit(self)
        expected = digest_commit(self, include_digest=False)
        object.__setattr__(self, "commit_digest", self.commit_digest or expected)
        if self.commit_digest != expected:
            raise EvidenceIntegrityError("commit digest mismatch")


def _validate_commit(c: EpistemicCommit) -> None:
    if not c.claims:
        raise EpistemicContractError("commit requires at least one claim")
    for name in (
        "claims",
        "evidence",
        "derivations",
        "assumptions",
        "counterexamples",
        "contradictions",
    ):
        if len(getattr(c, name)) > MAX_REFS:
            raise ReferenceValidationError(f"too many {name}")
    ev = {e.evidence_id: e for e in c.evidence}
    claims = {claim.claim_id: claim for claim in c.claims}
    propositions = {claim.proposition.proposition_id for claim in c.claims}
    derivations = {d.derivation_id: d for d in c.derivations}
    if (
        len(ev) != len(c.evidence)
        or len(claims) != len(c.claims)
        or len(derivations) != len(c.derivations)
    ):
        raise ReferenceValidationError("duplicate ids")
    for evidence in c.evidence:
        if evidence.episode_id != c.episode_id:
            raise ReferenceValidationError("evidence target episode mismatch")
        if (
            evidence.source_episode_id is None
            and evidence.snapshot_digest != c.snapshot_digest
        ):
            raise ReferenceValidationError("evidence snapshot mismatch")
        if evidence.valid_until is not None and evidence.valid_until <= c.committed_at:
            raise TemporalValidityError("expired evidence")
        if evidence.observed_at > c.committed_at:
            raise TemporalValidityError("future observation")
        if (
            evidence.valid_until is not None
            and evidence.valid_until <= evidence.observed_at
        ):
            raise TemporalValidityError("invalid evidence time window")
    for claim in c.claims:
        if (
            claim.episode_id != c.episode_id
            or claim.snapshot_digest != c.snapshot_digest
        ):
            raise ReferenceValidationError("claim binding mismatch")
        _references_exist(claim.evidence_ids, ev, "evidence")
        _references_exist(claim.derivation_ids, derivations, "derivation")
        for limitation in claim.limitations:
            _references_exist(limitation.affected_claim_ids, claims, "limited claim")
        if claim.status is ClaimStatus.PROVEN and not any(
            ev[item].kind is EvidenceKind.PROOF for item in claim.evidence_ids
        ):
            raise EvidenceIntegrityError("PROVEN claim requires proof artifact")
        if claim.status is ClaimStatus.RETRIEVED and (
            not claim.evidence_ids
            or not any(ev[item].citations for item in claim.evidence_ids)
        ):
            raise EvidenceIntegrityError("RETRIEVED claim requires evidence citation")
        if claim.status is ClaimStatus.CONTESTED and not claim.contested_by:
            raise ReferenceValidationError("CONTESTED claim requires contested_by")
    for derivation in c.derivations:
        _references_exist(
            (derivation.output_claim_id, *derivation.input_claim_ids),
            claims,
            "derivation claim",
        )
        _references_exist(derivation.evidence_ids, ev, "derivation evidence")
    producers: dict[str, str] = {}
    for derivation in c.derivations:
        if derivation.output_claim_id in producers:
            raise ReferenceValidationError("claim has multiple derivation producers")
        producers[derivation.output_claim_id] = derivation.derivation_id
    for assumption in c.assumptions:
        if assumption.proposition_id not in propositions:
            raise ReferenceValidationError("dangling assumption proposition")
    for counterexample in c.counterexamples:
        _references_exist((counterexample.evidence_id,), ev, "counterexample evidence")
        _references_exist(
            (counterexample.target_claim_id,), claims, "counterexample claim"
        )
    for contradiction in c.contradictions:
        _references_exist(contradiction.claim_ids, claims, "contradiction claim")
    _detect_cycles(c.derivations)
    _detect_supersession_cycles(c.claims)


def _references_exist(
    references: Sequence[str], available: Mapping[str, object] | set[str], name: str
) -> None:
    if any(reference not in available for reference in references):
        raise ReferenceValidationError(f"dangling {name} reference")


def _bounded(name: str, values: Sequence[object]) -> None:
    if len(values) > MAX_REFS:
        raise ReferenceValidationError(f"too many {name}")


def _unique(name: str, values: Iterable[str]) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ReferenceValidationError(f"duplicate {name}")


def _detect_cycles(derivations: Sequence[Derivation]) -> None:
    graph: dict[str, set[str]] = {}
    for derivation in derivations:
        graph.setdefault(derivation.output_claim_id, set()).update(
            derivation.input_claim_ids
        )
    visiting: set[str] = set()
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise CircularDerivationError("circular derivation")
        if node in seen:
            return
        visiting.add(node)
        for dependency in graph.get(node, ()):
            visit(dependency)
        visiting.remove(node)
        seen.add(node)

    for node in graph:
        visit(node)


def _detect_supersession_cycles(claims: Sequence[Claim]) -> None:
    claim_ids = {claim.claim_id for claim in claims}
    graph = {claim.claim_id: set(claim.supersedes) for claim in claims}
    for dependencies in graph.values():
        _references_exist(tuple(dependencies), claim_ids, "superseded claim")
    visiting: set[str] = set()
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ReferenceValidationError("cyclic supersession")
        if node in seen:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        seen.add(node)

    for node in graph:
        visit(node)


def _citation_to_dict(citation: Citation) -> dict[str, object]:
    return {
        "citation_id": citation.citation_id,
        "uri": citation.uri,
        "title": citation.title,
        "artifact_id": citation.artifact_id,
        "artifact_digest": citation.artifact_digest,
    }


def _uncertainty_to_dict(uncertainty: UncertaintyDescriptor) -> dict[str, object]:
    return {
        "kind": uncertainty.kind.value,
        "distribution_digest": uncertainty.distribution_digest,
        "interval_low": uncertainty.interval_low,
        "interval_high": uncertainty.interval_high,
        "calibration_id": uncertainty.calibration_id,
    }


def commit_to_dict(
    c: EpistemicCommit, *, include_digest: bool = True
) -> dict[str, object]:
    out: dict[str, object] = {
        "schema_version": EPISTEMIC_SCHEMA_VERSION,
        "commit_id": c.commit_id,
        "episode_id": c.episode_id,
        "case_id": c.case_id,
        "snapshot_digest": c.snapshot_digest,
        "authority_principal_id": c.authority_principal_id,
        "authority_level": c.authority_level.value,
        "authority_release_digest": c.authority_release_digest,
        "validation_digest": c.validation_digest,
        "policy_digest": c.policy_digest,
        "authority_evidence_digest": c.authority_evidence_digest,
        "committed_at": canonical_timestamp(c.committed_at),
        "prior_commit_digest": c.prior_commit_digest,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "proposition": {
                    "proposition_id": claim.proposition.proposition_id,
                    "subject": claim.proposition.subject,
                    "predicate": claim.proposition.predicate,
                    "object_value": claim.proposition.object_value,
                    "qualifiers": dict(claim.proposition.qualifiers),
                },
                "status": claim.status.value,
                "episode_id": claim.episode_id,
                "snapshot_digest": claim.snapshot_digest,
                "evidence_ids": list(claim.evidence_ids),
                "derivation_ids": list(claim.derivation_ids),
                "uncertainty": _uncertainty_to_dict(claim.uncertainty),
                "contested_by": list(claim.contested_by),
                "supersedes": list(claim.supersedes),
                "limitations": [
                    {
                        "limitation_id": item.limitation_id,
                        "description": item.description,
                        "affected_claim_ids": list(item.affected_claim_ids),
                    }
                    for item in claim.limitations
                ],
            }
            for claim in c.claims
        ],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind.value,
                "episode_id": item.episode_id,
                "snapshot_digest": item.snapshot_digest,
                "content_digest": item.content_digest,
                "provenance_id": item.provenance_id,
                "observed_at": canonical_timestamp(item.observed_at),
                "valid_until": (
                    None
                    if item.valid_until is None
                    else canonical_timestamp(item.valid_until)
                ),
                "citations": [
                    _citation_to_dict(citation) for citation in item.citations
                ],
                "source_episode_id": item.source_episode_id,
            }
            for item in c.evidence
        ],
        "derivations": [
            {
                "derivation_id": item.derivation_id,
                "input_claim_ids": list(item.input_claim_ids),
                "evidence_ids": list(item.evidence_ids),
                "rule_id": item.rule_id,
                "output_claim_id": item.output_claim_id,
            }
            for item in c.derivations
        ],
        "assumptions": [
            {"assumption_id": item.assumption_id, "proposition_id": item.proposition_id}
            for item in c.assumptions
        ],
        "counterexamples": [
            {
                "counterexample_id": item.counterexample_id,
                "evidence_id": item.evidence_id,
                "target_claim_id": item.target_claim_id,
            }
            for item in c.counterexamples
        ],
        "contradictions": [
            {
                "contradiction_id": item.contradiction_id,
                "claim_ids": list(item.claim_ids),
            }
            for item in c.contradictions
        ],
    }
    if include_digest:
        out["commit_digest"] = c.commit_digest
    return out


def digest_commit(c: EpistemicCommit, *, include_digest: bool = True) -> str:
    return str(
        Digest.of_bytes(
            canonical_json(commit_to_dict(c, include_digest=include_digest))
        )
    )


def dumps_commit(c: EpistemicCommit) -> bytes:
    return canonical_json(canonical_commit_document(c))


def loads_commit(raw: bytes | str) -> EpistemicCommit:
    try:
        value = canonical_json_loads(raw)
    except (TypeError, ValueError) as exc:
        raise EpistemicContractError("invalid canonical commit JSON") from exc
    if not isinstance(value, dict):
        raise EpistemicContractError("epistemic commit must be an object")
    return commit_from_dict(value)


def canonical_commit_document(c: EpistemicCommit) -> dict[str, object]:
    return commit_to_dict(c)


def commit_from_dict(value: Mapping[str, object]) -> EpistemicCommit:
    required = {
        "schema_version",
        "commit_id",
        "episode_id",
        "case_id",
        "snapshot_digest",
        "authority_principal_id",
        "authority_level",
        "authority_release_digest",
        "validation_digest",
        "policy_digest",
        "authority_evidence_digest",
        "committed_at",
        "prior_commit_digest",
        "claims",
        "evidence",
        "derivations",
        "assumptions",
        "counterexamples",
        "contradictions",
        "commit_digest",
    }
    _exact_fields(value, required, "commit")
    if value["schema_version"] != EPISTEMIC_SCHEMA_VERSION:
        raise EpistemicContractError("unsupported epistemic schema version")
    claims = tuple(
        _claim_from_dict(_mapping(item, "claim"))
        for item in _sequence(value["claims"], "claims")
    )
    evidence = tuple(
        _evidence_from_dict(_mapping(item, "evidence"))
        for item in _sequence(value["evidence"], "evidence")
    )
    derivations = tuple(
        Derivation(**_mapping(item, "derivation"))
        for item in _sequence(value["derivations"], "derivations")
    )
    assumptions = tuple(
        Assumption(**_mapping(item, "assumption"))
        for item in _sequence(value["assumptions"], "assumptions")
    )
    counterexamples = tuple(
        Counterexample(**_mapping(item, "counterexample"))
        for item in _sequence(value["counterexamples"], "counterexamples")
    )
    contradictions = tuple(
        Contradiction(item["contradiction_id"], tuple(item["claim_ids"]))
        for raw in _sequence(value["contradictions"], "contradictions")
        for item in (_mapping(raw, "contradiction"),)
    )
    return EpistemicCommit(
        commit_id=value["commit_id"],
        episode_id=value["episode_id"],
        case_id=value["case_id"],
        snapshot_digest=value["snapshot_digest"],
        authority_principal_id=value["authority_principal_id"],
        committed_at=parse_timestamp(value["committed_at"]),
        claims=claims,
        evidence=evidence,
        derivations=derivations,
        assumptions=assumptions,
        counterexamples=counterexamples,
        contradictions=contradictions,
        prior_commit_digest=value["prior_commit_digest"],
        authority_level=AuthorityLevel(value["authority_level"]),
        authority_release_digest=value["authority_release_digest"],
        validation_digest=value["validation_digest"],
        policy_digest=value["policy_digest"],
        authority_evidence_digest=value["authority_evidence_digest"],
        commit_digest=value["commit_digest"],
    )


def _claim_from_dict(value: Mapping[str, object]) -> Claim:
    _exact_fields(
        value,
        {
            "claim_id",
            "proposition",
            "status",
            "episode_id",
            "snapshot_digest",
            "evidence_ids",
            "derivation_ids",
            "uncertainty",
            "contested_by",
            "limitations",
            "supersedes",
        },
        "claim",
    )
    proposition = _mapping(value["proposition"], "proposition")
    _exact_fields(
        proposition,
        {"proposition_id", "subject", "predicate", "object_value", "qualifiers"},
        "proposition",
    )
    uncertainty = _mapping(value["uncertainty"], "uncertainty")
    _exact_fields(
        uncertainty,
        {
            "kind",
            "distribution_digest",
            "interval_low",
            "interval_high",
            "calibration_id",
        },
        "uncertainty",
    )
    limitations = tuple(
        Limitation(
            item["limitation_id"],
            item["description"],
            tuple(item["affected_claim_ids"]),
        )
        for raw in _sequence(value["limitations"], "limitations")
        for item in (_mapping(raw, "limitation"),)
    )
    return Claim(
        value["claim_id"],
        Proposition(
            proposition["proposition_id"],
            proposition["subject"],
            proposition["predicate"],
            proposition["object_value"],
            proposition["qualifiers"],
        ),
        ClaimStatus(value["status"]),
        value["episode_id"],
        value["snapshot_digest"],
        tuple(value["evidence_ids"]),
        tuple(value["derivation_ids"]),
        UncertaintyDescriptor(
            UncertaintyKind(uncertainty["kind"]),
            uncertainty["distribution_digest"],
            uncertainty["interval_low"],
            uncertainty["interval_high"],
            uncertainty["calibration_id"],
        ),
        tuple(value["contested_by"]),
        limitations,
        tuple(value["supersedes"]),
    )


def _evidence_from_dict(value: Mapping[str, object]) -> EvidenceArtifact:
    _exact_fields(
        value,
        {
            "evidence_id",
            "kind",
            "episode_id",
            "snapshot_digest",
            "content_digest",
            "provenance_id",
            "observed_at",
            "valid_until",
            "citations",
            "source_episode_id",
        },
        "evidence",
    )
    citations = tuple(
        Citation(**_mapping(item, "citation"))
        for item in _sequence(value["citations"], "citations")
    )
    return EvidenceArtifact(
        value["evidence_id"],
        EvidenceKind(value["kind"]),
        value["episode_id"],
        value["snapshot_digest"],
        value["content_digest"],
        value["provenance_id"],
        parse_timestamp(value["observed_at"]),
        None if value["valid_until"] is None else parse_timestamp(value["valid_until"]),
        citations,
        value["source_episode_id"],
    )


def _exact_fields(value: Mapping[str, object], required: set[str], name: str) -> None:
    if set(value) != required:
        raise EpistemicContractError(f"invalid {name} fields")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EpistemicContractError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EpistemicContractError(f"{name} must be an array")
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def content_digest(value: object) -> str:
    return str(Digest.of_json(value))


def project_semantic_claim(
    *,
    claim_id: str,
    episode_id: str,
    snapshot_digest: str,
    subject: str,
    predicate: str,
    object_value: str,
    evidence_id: str | None = None,
) -> Claim:
    return Claim(
        claim_id,
        Proposition(
            "prop:" + claim_id.split(":")[-1], subject, predicate, object_value
        ),
        ClaimStatus.OBSERVED if evidence_id else ClaimStatus.HYPOTHESIS,
        episode_id,
        snapshot_digest,
        () if evidence_id is None else (evidence_id,),
    )


def _canonical_id(kind: type[str], name: str, value: str) -> None:
    try:
        kind(value)
    except (TypeError, ValueError) as exc:
        raise EpistemicContractError(f"invalid {name}") from exc


def _id(name: str, value: str) -> None:
    _artifact_id(name, value)


def _artifact_id(name: str, value: str) -> None:
    _canonical_id(ArtifactId, name, value)


def _episode_id(name: str, value: str) -> None:
    _canonical_id(EpisodeId, name, value)


def _commit_id(name: str, value: str) -> None:
    _canonical_id(CommitId, name, value)


def _principal_id(name: str, value: str) -> None:
    _canonical_id(PrincipalId, name, value)


def _digest(name: str, value: str) -> None:
    try:
        Digest(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceIntegrityError(f"invalid {name}") from exc


def _aware(value: datetime, name: str) -> None:
    try:
        require_utc(value, name=name)
    except ValueError as exc:
        raise TemporalValidityError(str(exc)) from exc
