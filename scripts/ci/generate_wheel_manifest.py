#!/usr/bin/env python3
"""Generate or verify the exact positive source inclusion for the runtime wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from check_production_imports import ROOT, _path, check

MANIFEST = ROOT / "config/wheel-inclusion-manifest.json"
RESOURCES = (
    "config/capabilities.yaml",
    "config/architecture-status.json",
    "docs/architecture/ami-invariants.yaml",
    "docs/architecture/adr-006-local-language-interface.md",
    "docs/governance/controls.yaml",
    "docs/governance/impact-assessment.yaml",
    "evidence/qualification/language-contracts.json",
)


def document():
    modules, errors = check()
    if errors:
        raise RuntimeError("\n".join(errors))
    rows = []
    for module in sorted(modules):
        path = _path(module)
        if path is None:
            continue
        rel = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        rows.append(
            {
                "module": module,
                "path": rel,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "loc": len(raw.splitlines()),
            }
        )
    resources = [
        {
            "path": path,
            "wheel_path": f"vulcan/_release_evidence/{path}",
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
        for path in RESOURCES
    ]
    body = {
        "schema": "vulcan-wheel-inclusion/v1",
        "entrypoint": "vulcan.runtime.app",
        "files": rows,
        "resources": resources,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["manifest_digest"] = hashlib.sha256(canonical).hexdigest()
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    expected = json.dumps(document(), indent=2, sort_keys=True) + "\n"
    if a.check:
        if not MANIFEST.exists() or MANIFEST.read_text() != expected:
            print(f"{MANIFEST.relative_to(ROOT)} is not current")
            return 1
    else:
        MANIFEST.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
