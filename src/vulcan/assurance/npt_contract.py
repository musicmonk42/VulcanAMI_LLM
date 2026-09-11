"""Fail-closed reader for the governed NPT research contract.

This module is deliberately dependency-light and contains no estimator.  Production
may inspect the contract, but the values returned here confer no cognitive authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

SCHEMA = "vulcan-npt-engineering-contract/v1"
THEORY_STATUS = "research_hypothesis"
DIMENSIONS = (
    "center_indexical",
    "temporal_now",
    "boundary",
    "valuation",
    "policy",
    "reafferent",
    "memory",
)
MATURITY = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5}
SOURCE_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = Path(__file__).resolve().parents[1] / "_release_evidence"
ROOT = SOURCE_ROOT if (SOURCE_ROOT / "config").is_dir() else RELEASE_ROOT
DEFAULT_CONTRACT = ROOT / "config" / "npt-engineering-contract.json"
DEFAULT_DIGEST = ROOT / "config" / "npt-engineering-contract.sha256"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

TOP_FIELDS = {
    "schema",
    "theory_status",
    "representation_instantiation",
    "authority",
    "replay",
    "forbidden_claims",
    "dimensions",
    "evidence",
}
DIMENSION_FIELDS = {
    "id",
    "observable_definition",
    "intervention_family",
    "rival_explanation",
    "evidence_schema",
    "evidence_references",
    "current_gate",
    "maturity",
}
EVIDENCE_FIELDS = {"source_git_sha", "commands", "artifacts"}
ARTIFACT_FIELDS = {"path", "sha256"}
AUTHORITY_FIELDS = {
    "authority_ceiling",
    "promotion_owner",
    "estimator_role",
    "production_access",
    "production_import_rule",
}
REPLAY_FIELDS = {"mode", "canonical_encoding", "missing_dimension_rule"}
CLAIMS_FIELDS = {"scientific", "product"}
REQUIRED_COMMANDS = (
    "python scripts/ci/validate_npt_contract.py",
    "python -m pytest -q tests/assurance/test_npt_contract.py",
    "python -O -m pytest -q tests/assurance/test_npt_contract.py",
    "python scripts/ci/check_production_imports.py",
)


class NPTContractError(ValueError):
    """The contract or one of its bound artifacts is invalid."""


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NPTContractError(f"duplicate field: {key}")
        result[key] = value
    return result


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NPTContractError(f"{label} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown or missing:
        raise NPTContractError(
            f"{label} fields invalid; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return value


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 4096
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
    ):
        raise NPTContractError(f"{label} must be non-empty text")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise NPTContractError(f"{label} must be a non-empty list")
    result = [_text(item, label) for item in value]
    if len(set(result)) != len(result):
        raise NPTContractError(f"{label} contains duplicates")
    return result


def _canonical(data: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NPTContractError("contract contains a noncanonical value") from exc


def _invariant_ids(path: Path) -> set[str]:
    return {
        line.split(":", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- id:")
    }


def _safe_file(root: Path, relative: str, label: str) -> Path:
    """Resolve a repository-relative regular file without accepting symlinks."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise NPTContractError(f"unsafe {label}: {relative}")
    candidate = root / relative_path
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise NPTContractError(f"nonexistent {label}: {relative}") from exc
    chain = [root]
    for part in relative_path.parts:
        chain.append(chain[-1] / part)
    if (
        any(path.is_symlink() for path in chain)
        or not candidate.is_file()
        or not resolved.is_relative_to(resolved_root)
    ):
        raise NPTContractError(f"unsafe {label}: {relative}")
    return candidate


def validate_contract(
    data: object, *, root: Path = ROOT, verify_artifacts: bool = True
) -> Mapping[str, object]:
    def reject_consciousness_fields(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if "conscious" in key.lower() and not isinstance(child, (dict, list)):
                    raise NPTContractError(
                        "scalar or Boolean consciousness fields are forbidden"
                    )
                reject_consciousness_fields(child)
        elif isinstance(value, list):
            for child in value:
                reject_consciousness_fields(child)

    reject_consciousness_fields(data)
    contract = _object(data, TOP_FIELDS, "contract")
    if contract["schema"] != SCHEMA or contract["theory_status"] != THEORY_STATUS:
        raise NPTContractError("schema and theory_status must use their exact values")
    if contract["representation_instantiation"] != "Rep(process) != Inst(process)":
        raise NPTContractError("representation must remain distinct from instantiation")

    authority = _object(contract["authority"], AUTHORITY_FIELDS, "authority")
    required_authority = {
        "authority_ceiling": "NONE",
        "promotion_owner": "cognitive_microkernel",
        "estimator_role": "research_measurement_only",
        "production_access": "inspect_contract_only",
        "production_import_rule": "research_import_denied",
    }
    if authority != required_authority:
        raise NPTContractError("NPT contract may not grant authority")
    replay = _object(contract["replay"], REPLAY_FIELDS, "replay")
    if replay != {
        "mode": "exact",
        "canonical_encoding": "RFC8785_style_sorted_utf8_json",
        "missing_dimension_rule": "reject_not_average",
    }:
        raise NPTContractError(
            "exact replay and reject-not-average semantics are required"
        )

    claims = _object(contract["forbidden_claims"], CLAIMS_FIELDS, "forbidden_claims")
    scientific = _string_list(claims["scientific"], "scientific claims")
    product = _string_list(claims["product"], "product claims")
    required_claims = {
        "scientific": {"consciousness_declaration", "boolean_consciousness_conclusion"},
        "product": {
            "consciousness_feature",
            "subjecthood_claim",
            "estimator_as_truth_or_policy",
        },
    }
    if (
        set(scientific) != required_claims["scientific"]
        or set(product) != required_claims["product"]
    ):
        raise NPTContractError("forbidden scientific/product claims must be exact")

    dimensions = contract["dimensions"]
    if not isinstance(dimensions, list) or len(dimensions) != len(DIMENSIONS):
        raise NPTContractError("exactly seven closure dimensions are required")
    invariant_path = _safe_file(
        root, "docs/architecture/ami-invariants.yaml", "invariant catalog"
    )
    known_invariants = _invariant_ids(invariant_path)
    seen: list[str] = []
    evidence_paths: set[str] = set()
    for index, raw in enumerate(dimensions):
        dimension = _object(raw, DIMENSION_FIELDS, f"dimension[{index}]")
        identifier = _text(dimension["id"], "dimension id")
        seen.append(identifier)
        for field in (
            "observable_definition",
            "intervention_family",
            "rival_explanation",
        ):
            _text(dimension[field], f"{identifier}.{field}")
        schema = _object(
            dimension["evidence_schema"],
            {"schema", "required_fields", "invariant_references"},
            f"{identifier}.evidence_schema",
        )
        _text(schema["schema"], f"{identifier}.evidence_schema.schema")
        required_evidence_fields = _string_list(
            schema["required_fields"], f"{identifier}.required_fields"
        )
        if schema["schema"] != "npt-closure-evidence/v1" or set(
            required_evidence_fields
        ) != {
            "contract_sha256",
            "validator_sha256",
            "invariant_sha256",
            "import_policy_sha256",
            "artifact_sha256",
            "command",
            "source_git_sha",
            "intervention_id",
            "rival_id",
            "observable",
            "uncertainty",
        }:
            raise NPTContractError(f"{identifier} evidence schema must be exact")
        refs = _string_list(schema["invariant_references"], f"{identifier}.invariants")
        if not set(refs).issubset(known_invariants):
            raise NPTContractError(f"{identifier} references nonexistent invariant")
        for reference in _string_list(
            dimension["evidence_references"], f"{identifier}.evidence_references"
        ):
            if reference.startswith("/") or ".." in Path(reference).parts:
                raise NPTContractError(f"unsafe evidence reference: {reference}")
            if verify_artifacts:
                _safe_file(root, reference, "evidence reference")
            evidence_paths.add(reference)
        gate = _object(
            dimension["current_gate"],
            {"gate_f", "gate_g", "evidence_disposition"},
            f"{identifier}.gate",
        )
        if gate != {
            "gate_f": "open",
            "gate_g": "open",
            "evidence_disposition": "insufficient",
        }:
            raise NPTContractError(f"{identifier} may not close Gate F or G")
        maturity = _text(dimension["maturity"], f"{identifier}.maturity")
        if maturity not in MATURITY or MATURITY[maturity] > MATURITY["M2"]:
            raise NPTContractError("maturity exceeds available exact-artifact proof")
    if tuple(seen) != DIMENSIONS or len(set(seen)) != len(DIMENSIONS):
        raise NPTContractError(
            "dimensions must be the seven canonical dimensions in order"
        )

    evidence = _object(contract["evidence"], EVIDENCE_FIELDS, "evidence")
    if not isinstance(evidence["source_git_sha"], str) or not _GIT_SHA.fullmatch(
        evidence["source_git_sha"]
    ):
        raise NPTContractError("source_git_sha must be an exact commit SHA")
    commands = _string_list(evidence["commands"], "evidence.commands")
    if tuple(commands) != REQUIRED_COMMANDS:
        raise NPTContractError("evidence commands must be the exact governed set")
    artifacts = evidence["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise NPTContractError("evidence.artifacts must be non-empty")
    bound: set[str] = set()
    for index, raw in enumerate(artifacts):
        artifact = _object(raw, ARTIFACT_FIELDS, f"artifact[{index}]")
        path = _text(artifact["path"], "artifact.path")
        digest = _text(artifact["sha256"], "artifact.sha256")
        if path in bound or path.startswith("/") or ".." in Path(path).parts:
            raise NPTContractError(f"duplicate or unsafe artifact path: {path}")
        if not _SHA256.fullmatch(digest):
            raise NPTContractError(f"invalid artifact binding: {path}")
        if verify_artifacts:
            target = _safe_file(root, path, "artifact binding")
            try:
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError as exc:
                raise NPTContractError(f"unreadable artifact binding: {path}") from exc
            if actual != digest:
                raise NPTContractError(f"artifact digest mismatch: {path}")
        bound.add(path)
    required_bound = {
        "src/vulcan/assurance/npt_contract.py",
        "scripts/ci/validate_npt_contract.py",
        "scripts/ci/run_constitutional_gate.py",
        "docs/architecture/ami-invariants.yaml",
        "config/production-import-policy.json",
        "tests/assurance/test_npt_contract.py",
    }
    if not required_bound.issubset(bound) or not evidence_paths.issubset(bound):
        raise NPTContractError(
            "validator, invariants, import policy, and all evidence must be digest-bound"
        )
    return MappingProxyType(contract)


def load_contract(
    path: Path = DEFAULT_CONTRACT,
    digest_path: Path = DEFAULT_DIGEST,
    *,
    root: Path = ROOT,
    verify_artifacts: bool | None = None,
) -> Mapping[str, object]:
    try:
        if path.is_symlink() or digest_path.is_symlink():
            raise NPTContractError("contract and digest must be regular files")
        raw = path.read_bytes()
        expected = digest_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise NPTContractError("contract or digest is unreadable") from exc
    if not _SHA256.fullmatch(expected) or hashlib.sha256(raw).hexdigest() != expected:
        raise NPTContractError("contract digest mismatch")
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                NPTContractError(f"non-finite JSON value: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise NPTContractError("contract is not valid UTF-8 JSON") from exc
    if verify_artifacts is None:
        verify_artifacts = root.resolve() != RELEASE_ROOT.resolve()
    validated = validate_contract(data, root=root, verify_artifacts=verify_artifacts)
    if raw != _canonical(data) + b"\n":
        raise NPTContractError("contract is not in canonical encoding")
    return validated
