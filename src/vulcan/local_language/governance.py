"""Offline, fail-closed data and release-governance value objects.

Nothing here is imported by the runtime.  These records intentionally contain
digests rather than prompts, answers, claims, or production data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class GovernanceError(ValueError):
    pass


class ExampleRole(str, Enum):
    INPUT_PROPOSAL = "input_proposal"
    OUTPUT_DRAFT = "output_draft"


class ReleaseState(str, Enum):
    EXPERIMENTAL = "experimental"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    STAGED = "staged"
    ACTIVE = "active"
    REJECTED = "rejected"
    REVOKED = "revoked"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    license_identifier: str
    consent_basis: str
    purpose: str
    privacy_class: str
    retention: str
    allowed_uses: tuple[str, ...]
    revoked: bool = False

    def eligible(self) -> bool:
        return (not self.revoked and all(bool(value) for value in (
            self.source_id, self.license_identifier, self.consent_basis, self.purpose,
            self.privacy_class, self.retention)) and bool(self.allowed_uses))


@dataclass(frozen=True)
class LanguageExample:
    example_id: str
    role: ExampleRole
    locale: str
    domain: str
    schema_version: str
    ontology_version: str
    input_digest: str
    target_digest: str
    target_kind: str
    source_id: str
    generator_version: str
    group_id: str
    split: str
    excluded: bool = False
    exclusion_reason: str | None = None

    def validate(self, source: DatasetSource) -> None:
        if not all(_ID.fullmatch(value) for value in (self.example_id, self.source_id, self.generator_version, self.group_id)):
            raise GovernanceError("invalid dataset identifier")
        if self.locale != "und" or self.domain != "bounded-arithmetic":
            raise GovernanceError("unsupported dataset locale or domain")
        if self.schema_version != "semantic-ingress/2" or self.ontology_version != "formal-arithmetic/2":
            raise GovernanceError("unsupported semantic contract")
        if not _DIGEST.fullmatch(self.input_digest) or not _DIGEST.fullmatch(self.target_digest):
            raise GovernanceError("invalid example digest")
        expected_target = "interpretation_proposal" if self.role is ExampleRole.INPUT_PROPOSAL else "untrusted_render_draft"
        if self.target_kind != expected_target:
            raise GovernanceError("role target is not a narrow language contract")
        if self.split not in {"development", "validation", "promotion_test", "red_team", "canary"}:
            raise GovernanceError("unknown locked split")
        if self.excluded or not source.eligible() or source.source_id != self.source_id:
            raise GovernanceError("dataset example is not eligible")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    role: ExampleRole
    examples_digest: str
    source_digest: str
    split_digest: str
    code_version: str
    non_promotable_fixture: bool = False

    def validate(self) -> None:
        if not _ID.fullmatch(self.dataset_id) or not _ID.fullmatch(self.code_version):
            raise GovernanceError("invalid dataset manifest identifier")
        if not all(_DIGEST.fullmatch(value) for value in (self.examples_digest, self.source_digest, self.split_digest)):
            raise GovernanceError("invalid dataset manifest digest")


def validate_grouped_split(examples: tuple[LanguageExample, ...], source: DatasetSource) -> None:
    """Reject leakage: a source/template group may occur in exactly one split."""
    assigned: dict[str, str] = {}
    for example in examples:
        example.validate(source)
        previous = assigned.setdefault(example.group_id, example.split)
        if previous != example.split:
            raise GovernanceError("dataset group leaks across locked splits")


_TRANSITIONS = {
    ReleaseState.EXPERIMENTAL: {ReleaseState.EVALUATED, ReleaseState.REJECTED},
    ReleaseState.EVALUATED: {ReleaseState.APPROVED, ReleaseState.REJECTED},
    ReleaseState.APPROVED: {ReleaseState.STAGED, ReleaseState.REVOKED},
    ReleaseState.STAGED: {ReleaseState.ACTIVE, ReleaseState.ROLLED_BACK, ReleaseState.REVOKED},
    ReleaseState.ACTIVE: {ReleaseState.ROLLED_BACK, ReleaseState.REVOKED},
}


def transition_release(current: ReleaseState, target: ReleaseState, *, authority: str) -> ReleaseState:
    """Enforce that only the external release authority can approve/activate."""
    if target not in _TRANSITIONS.get(current, set()):
        raise GovernanceError("invalid release transition")
    required = "evaluator" if target is ReleaseState.EVALUATED else "release_authority"
    if authority != required:
        raise GovernanceError("unauthorized release transition")
    return target
