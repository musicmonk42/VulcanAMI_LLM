#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    ROOT / "requirements.txt",
    ROOT / "requirements-constitutional.txt",
    ROOT / "config/capabilities.yaml",
    ROOT / "docs/governance/controls.yaml",
]
PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s]+$")
HASH_RE = re.compile(r"^--hash=sha256:[0-9a-f]{64}$")


def validate_hashed_requirements(text: str) -> list[str]:
    """Validate the deliberately small, fully pinned constitutional lock format."""
    errors: list[str] = []
    blocks: list[list[str]] = []
    current: list[str] | None = None
    only_binary = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "--only-binary=:all:":
            only_binary = True
            continue
        if raw[0].isspace():
            if current is None:
                errors.append(f"line {number}: orphan lock continuation")
            elif line.startswith("--hash="):
                current.append(line)
            continue
        if current is not None:
            blocks.append(current)
        current = [line]
    if current is not None:
        blocks.append(current)
    if not only_binary:
        errors.append("lock must require binary distributions")
    if not blocks:
        errors.append("lock contains no requirements")
    for block in blocks:
        requirement = block[0]
        if (
            not requirement.endswith("\\")
            or PIN_RE.fullmatch(requirement[:-1].strip()) is None
        ):
            errors.append(
                f"requirement is not exactly pinned with continuations: {block[0]}"
            )
        hashes = [
            item[:-1].strip() if item.endswith("\\") else item for item in block[1:]
        ]
        if not hashes or any(HASH_RE.fullmatch(item) is None for item in hashes):
            errors.append(
                f"requirement has an invalid or missing SHA-256 hash: {block[0]}"
            )
    return errors


def main() -> int:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in REQUIRED
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        print(
            "Missing required dependency/evidence inputs:",
            *missing,
            sep="\n",
            file=sys.stderr,
        )
        return 1
    production = (ROOT / "requirements.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    if "==" not in production and "-r" not in production:
        print(
            "requirements.txt must contain exact pins or locked includes",
            file=sys.stderr,
        )
        return 1
    errors = validate_hashed_requirements(
        (ROOT / "requirements-constitutional.txt").read_text(encoding="utf-8")
    )
    if errors:
        print(
            "Invalid constitutional dependency lock:",
            *errors,
            sep="\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
