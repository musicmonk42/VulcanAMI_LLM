#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [ROOT / "docs/governance/controls.yaml", ROOT / "docs/governance/standards-crosswalk.md", ROOT / "docs/generated/architecture-inventory.json"]


def main() -> int:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print("Missing supply-chain evidence inputs:", *missing, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
