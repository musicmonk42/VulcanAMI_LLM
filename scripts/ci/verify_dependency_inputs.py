#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [ROOT / "requirements.txt", ROOT / "config/capabilities.yaml", ROOT / "docs/governance/controls.yaml"]


def main() -> int:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print("Missing required dependency/evidence inputs:", *missing, sep="\n", file=sys.stderr)
        return 1
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8", errors="ignore")
    if "==" not in req and "-r" not in req:
        print("requirements.txt must contain exact pins or locked includes for production dependency policy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
