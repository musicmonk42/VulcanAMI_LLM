"""Fail-closed, offline verification for a local language adapter release.

A manifest identifies a complete, role-specific artifact set.  This module does
*not* load models or select a release for serving; it only gives offline release
processes a small, deterministic verifier.  It intentionally makes no signing,
provenance, safety, or quality claim beyond the fields and bytes it verifies.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_MANIFEST_SCHEMA = "local-language-release/1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_REQUIRED_ARTIFACTS = frozenset({"weights", "tokenizer", "config", "evaluation_report"})


class ReleaseVerificationError(ValueError):
    """The release is incomplete, unapproved, or differs from its manifest."""


class ReleaseRole(str, Enum):
    INPUT = "input-language-adapter"
    OUTPUT = "output-language-adapter"


@dataclass(frozen=True)
class Artifact:
    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class Evaluation:
    report_sha256: str
    deterministic_baseline: str
    baseline_score: float
    candidate_score: float
    zero_tolerance_passed: bool


@dataclass(frozen=True)
class LocalLanguageRelease:
    release_id: str
    role: ReleaseRole
    runtime_abi: str
    license_identifier: str
    promotion_state: str
    approval_id: str | None
    artifacts: tuple[Artifact, ...]
    evaluation: Evaluation


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: object, label: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ReleaseVerificationError(f"invalid {label} schema")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseVerificationError(f"invalid {label}")
    return value


def _score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseVerificationError(f"invalid {label}")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ReleaseVerificationError(f"invalid {label}")
    return result


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not relative:
        raise ReleaseVerificationError("artifact path escapes release root")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / candidate).resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ReleaseVerificationError("artifact path escapes release root")
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseVerificationError("artifact is not a regular release file")
    return resolved


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_manifest(raw: object) -> LocalLanguageRelease:
    manifest = _mapping(raw, "release manifest", {
        "schema_version", "release_id", "role", "runtime_abi", "license_identifier",
        "promotion", "artifacts", "evaluation",
    })
    release_id = _string(manifest["release_id"], "release_id")
    if not _IDENTIFIER.fullmatch(release_id):
        raise ReleaseVerificationError("invalid release_id")
    if manifest["schema_version"] != _MANIFEST_SCHEMA:
        raise ReleaseVerificationError("unsupported manifest schema")
    try:
        role = ReleaseRole(manifest["role"])
    except (TypeError, ValueError) as exc:
        raise ReleaseVerificationError("unsupported language-adapter role") from exc
    promotion = _mapping(manifest["promotion"], "promotion", {"state", "approval_id"})
    state = _string(promotion["state"], "promotion state")
    approval = promotion["approval_id"]
    if state != "approved" or not isinstance(approval, str) or not _IDENTIFIER.fullmatch(approval):
        raise ReleaseVerificationError("release is not explicitly approved")
    items = manifest["artifacts"]
    if not isinstance(items, list) or not items:
        raise ReleaseVerificationError("invalid artifacts")
    artifacts: list[Artifact] = []
    for item in items:
        artifact = _mapping(item, "artifact", {"name", "path", "sha256"})
        name, path, digest = (_string(artifact[key], key) for key in ("name", "path", "sha256"))
        if not _IDENTIFIER.fullmatch(name) or not _SHA256.fullmatch(digest):
            raise ReleaseVerificationError("invalid artifact identity")
        artifacts.append(Artifact(name, path, digest))
    if {item.name for item in artifacts} != _REQUIRED_ARTIFACTS or len({item.path for item in artifacts}) != len(artifacts):
        raise ReleaseVerificationError("release must bind weights, tokenizer, config, and evaluation report exactly once")
    evaluation_raw = _mapping(manifest["evaluation"], "evaluation", {
        "report_sha256", "deterministic_baseline", "baseline_score", "candidate_score", "zero_tolerance_passed",
    })
    report = _string(evaluation_raw["report_sha256"], "evaluation report digest")
    if not _SHA256.fullmatch(report):
        raise ReleaseVerificationError("invalid evaluation report digest")
    evaluation = Evaluation(report, _string(evaluation_raw["deterministic_baseline"], "deterministic baseline"),
                            _score(evaluation_raw["baseline_score"], "baseline score"),
                            _score(evaluation_raw["candidate_score"], "candidate score"),
                            evaluation_raw["zero_tolerance_passed"] is True)
    if next(item.sha256 for item in artifacts if item.name == "evaluation_report") != evaluation.report_sha256:
        raise ReleaseVerificationError("evaluation report is not bound to its artifact")
    if not evaluation.zero_tolerance_passed or evaluation.candidate_score <= evaluation.baseline_score:
        raise ReleaseVerificationError("evaluation does not justify neural activation")
    return LocalLanguageRelease(release_id, role, _string(manifest["runtime_abi"], "runtime ABI"),
                                _string(manifest["license_identifier"], "license identifier"), state, approval,
                                tuple(artifacts), evaluation)


def verify_release(release_root: str | Path) -> LocalLanguageRelease:
    """Verify one local release directory without network or model execution."""
    root = Path(release_root)
    if not root.is_dir() or root.is_symlink():
        raise ReleaseVerificationError("release root is not a directory")
    try:
        manifest_path = _safe_path(root, "manifest.json")
    except OSError as exc:
        raise ReleaseVerificationError("unreadable release root") from exc
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("unreadable release manifest") from exc
    release = _parse_manifest(raw)
    for artifact in release.artifacts:
        if _digest(_safe_path(root, artifact.path)) != artifact.sha256:
            raise ReleaseVerificationError(f"artifact digest mismatch: {artifact.name}")
    return release
