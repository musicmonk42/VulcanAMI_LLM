#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SECRET_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{20,}['\"]")
EXCLUDED = {".git", "docs/generated"}
ALLOWLISTED_EXISTING = {"docs/API_DOCUMENTATION.md", "configs/helm_chart.yaml", "src/vulcan/tests/test_distillation.py", "k8s/base/secret.yaml"}


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or any(rel.startswith(item) for item in EXCLUDED) or rel in ALLOWLISTED_EXISTING:
            continue
        if path.stat().st_size > 1_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_RE.search(text.replace("VULCAN_TEST_FAULTS_ONLY", "TEST_FAULTS")):
            findings.append(rel)
    if findings:
        print("Potential secret patterns found:", *findings, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
