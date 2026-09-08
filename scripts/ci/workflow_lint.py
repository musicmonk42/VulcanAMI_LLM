#!/usr/bin/env python3
"""Fail closed lint for required GitHub workflow gates."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    ROOT / ".github/workflows/ci.yml",
    ROOT / ".github/workflows/security.yml",
    ROOT / ".github/workflows/docker.yml",
    ROOT / ".github/workflows/runtime-e2e.yml",
]
FORBIDDEN = [
    "|| true",
    "--exit-zero",
    "continue-on-error: true",
    "empty SARIF",
    "touch trivy-results.sarif",
]
USES_RE = re.compile(r"uses:\s*([^\s#]+)")
PIN_RE = re.compile(r"@[0-9a-f]{40}$")
SETUP_PYTHON = "actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c"
REQUIRED_JOBS = {
    "ci.yml": [
        "dependency-light-unit-contract",
        "full-integration",
        "architecture-fitness",
        "static-typing",
        "lint-format",
        "optimized-python",
    ],
    "security.yml": ["secret-scan", "sast", "dependency-vulnerability-policy"],
    "docker.yml": ["image-e2e", "supply-chain-evidence"],
    "runtime-e2e.yml": ["built-image-runtime-qualification", "helm-conformance"],
}
HASHED_INSTALL = (
    "python -m pip install --require-hashes -r requirements-constitutional.txt"
)
THIRD_PARTY_COMMAND_RE = re.compile(r"\b(?:pytest|mypy|black|isort)\b")
PYTHON_COMMAND_RE = re.compile(
    r"(?:run:|\n\s+)(?:[^\n]*\bpython(?:3)?\b|[^\n]*\bpip\b)"
)
JOB_RE = re.compile(
    r"^  ([a-zA-Z0-9_-]+):\n(.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)", re.M | re.S
)


def lint_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for token in FORBIDDEN:
        if token in text:
            errors.append(f"{path}: forbidden non-gating token {token!r}")
    for match in USES_RE.finditer(text):
        target = match.group(1).strip("\"'")
        if not target.startswith("./") and PIN_RE.search(target) is None:
            errors.append(f"{path}: action is not pinned by immutable SHA: {target}")
    for job in REQUIRED_JOBS[path.name]:
        if f"  {job}:" not in text:
            errors.append(f"{path}: missing required job {job}")
    for job_name, body in JOB_RE.findall(text):
        if PYTHON_COMMAND_RE.search(body) and SETUP_PYTHON not in body:
            errors.append(
                f"{path}: Python job {job_name!r} does not set up Python 3.11"
            )
        if THIRD_PARTY_COMMAND_RE.search(body) and HASHED_INSTALL not in body:
            errors.append(
                f"{path}: Python job {job_name!r} does not install the hash-locked toolchain"
            )
    if path.name != "runtime-e2e.yml" and "scripts/ci/write_evidence.py" not in text:
        errors.append(f"{path}: missing machine-readable evidence writer")
    return errors


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv] if argv else REQUIRED
    errors = [error for path in paths for error in lint_file(path)]
    if errors:
        print(*errors, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
