"""Proposal-only improvement boundary safe to import in a serving image.

This module deliberately has no process execution, source-write, git, package,
approval, or installation primitive.  Serving code can validate and durably
emit a candidate, but cannot promote it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "vulcan-improvement-proposal/1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_FIELDS = frozenset({
    "schema_version", "proposal_id", "objective_type", "target_path",
    "expected_original_sha256", "original_content", "candidate_content",
    "candidate_sha256", "inspected_source_digest", "generator_identity",
    "provider_release_digest", "rationale", "expected_policy_digest",
    "approval_id",
})


class ProposalError(ValueError):
    """An untrusted proposal failed closed validation."""


@dataclass(frozen=True)
class ImprovementProposal:
    schema_version: str
    proposal_id: str
    objective_type: str
    target_path: str
    expected_original_sha256: str
    original_content: str
    candidate_content: str
    candidate_sha256: str
    inspected_source_digest: str
    generator_identity: str
    provider_release_digest: str
    rationale: str = ""
    expected_policy_digest: str = ""
    approval_id: str = ""

    @classmethod
    def from_json(cls, text: str) -> "ImprovementProposal":
        def pairs(pairs_: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs_:
                if key in result:
                    raise ProposalError(f"duplicate proposal key: {key}")
                result[key] = value
            return result
        try:
            value = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ProposalError("non-finite proposal value")))
        except ProposalError:
            raise
        except Exception as exc:
            raise ProposalError("invalid proposal JSON") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImprovementProposal":
        if not isinstance(value, Mapping):
            raise ProposalError("proposal must be a mapping")
        unknown = set(value) - _FIELDS
        if unknown:
            raise ProposalError(f"unknown proposal fields: {sorted(unknown)}")
        missing = sorted((_FIELDS - {"rationale", "approval_id"}) - set(value))
        if missing:
            raise ProposalError(f"missing proposal fields: {missing}")
        data = {key: value.get(key, "") for key in _FIELDS}
        if not all(isinstance(item, str) for item in data.values()):
            raise ProposalError("proposal fields must be strings")
        proposal = cls(**{key: data[key] for key in cls.__dataclass_fields__})
        proposal.validate_envelope()
        return proposal

    def validate_envelope(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProposalError("unsupported proposal schema")
        if not _IDENTIFIER.fullmatch(self.proposal_id):
            raise ProposalError("invalid proposal id")
        target = Path(self.target_path)
        if (not self.target_path or target.is_absolute() or ".." in target.parts
                or "\\" in self.target_path or any(part in {"", "."} for part in target.parts)):
            raise ProposalError("proposal target escapes repository")
        for value, label in ((self.objective_type, "objective"), (self.generator_identity, "generator identity"),
                             (self.provider_release_digest, "provider release")):
            if not value or len(value) > 128 or any(ord(character) < 32 for character in value):
                raise ProposalError(f"invalid {label}")
        for value in (self.expected_original_sha256, self.candidate_sha256, self.inspected_source_digest, self.expected_policy_digest):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ProposalError("invalid proposal digest")
        if hashlib.sha256(self.original_content.encode()).hexdigest() != self.expected_original_sha256:
            raise ProposalError("original source digest mismatch")
        if hashlib.sha256(self.candidate_content.encode()).hexdigest() != self.candidate_sha256:
            raise ProposalError("candidate source digest mismatch")
        if not self.candidate_content or len(self.candidate_content.encode()) > 50_000:
            raise ProposalError("candidate source size invalid")

    def digest(self) -> str:
        body = {key: getattr(self, key) for key in sorted(self.__dataclass_fields__) if key != "approval_id"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def public_record(self) -> dict[str, str]:
        """Return audit metadata without source/provider text."""
        return {"schema_version": self.schema_version, "proposal_id": self.proposal_id,
                "proposal_digest": self.digest(), "objective_type": self.objective_type,
                "target_path": self.target_path, "original_digest": self.expected_original_sha256,
                "candidate_digest": self.candidate_sha256, "generator_identity": self.generator_identity,
                "provider_release_digest": self.provider_release_digest,
                "policy_digest": self.expected_policy_digest}


class ImprovementProposalStore:
    """Append-only proposal outbox; existing identifiers are immutable."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def emit(self, proposal: ImprovementProposal) -> str:
        proposal.validate_envelope()
        if proposal.approval_id:
            raise ProposalError("serving proposals cannot carry approval authority")
        if proposal.rationale:
            raise ProposalError("serving proposals cannot persist free-form rationale")
        target = self.root / f"{proposal.proposal_id}.json"
        if target.exists() or target.is_symlink():
            raise ProposalError("proposal id already emitted")
        document = asdict(proposal)
        document.pop("rationale", None)
        document.pop("approval_id", None)
        payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return proposal.digest()

    def load(self, proposal_id: str) -> ImprovementProposal:
        if not proposal_id or "/" in proposal_id or "\\" in proposal_id:
            raise ProposalError("invalid proposal id")
        path = self.root / f"{proposal_id}.json"
        if path.is_symlink():
            raise ProposalError("symlinked proposal")
        return ImprovementProposal.from_json(path.read_text(encoding="utf-8"))
