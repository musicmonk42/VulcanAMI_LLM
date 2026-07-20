#!/usr/bin/env python3
"""Fail closed lint for required GitHub workflow gates."""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    ROOT / ".github/workflows/ci.yml",
    ROOT / ".github/workflows/security.yml",
    ROOT / ".github/workflows/docker.yml",
]
FORBIDDEN = ["|| true", "--exit-zero", "continue-on-error: true", "empty SARIF", "touch trivy-results.sarif"]
USES_RE = re.compile(r"uses:\s*([^\s#]+)")
PIN_RE = re.compile(r"@[0-9a-f]{40}$")
REQUIRED_JOBS = {
    "ci.yml": ["dependency-light-unit-contract", "full-integration", "architecture-fitness", "static-typing", "lint-format", "optimized-python"],
    "security.yml": ["secret-scan", "sast", "dependency-vulnerability-policy"],
    "docker.yml": ["image-e2e", "supply-chain-evidence"],
}


def lint_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for token in FORBIDDEN:
        if token in text:
            errors.append(f"{path}: forbidden non-gating token {token!r}")
    for match in USES_RE.finditer(text):
        target = match.group(1).strip('"\'')
        if target.startswith("./"):
            continue
        if PIN_RE.search(target) is None:
            errors.append(f"{path}: action is not pinned by immutable SHA: {target}")
    for job in REQUIRED_JOBS[path.name]:
        if f"  {job}:" not in text:
            errors.append(f"{path}: missing required job {job}")
    if "scripts/ci/write_evidence.py" not in text:
        errors.append(f"{path}: missing machine-readable evidence writer")
    return errors


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv] if argv else REQUIRED
    errors: list[str] = []
    for path in paths:
        errors.extend(lint_file(path))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
