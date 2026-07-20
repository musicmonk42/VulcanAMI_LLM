#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [ROOT / "docker-compose.prod.yml", ROOT / "docker/api/Dockerfile", ROOT / "helm/vulcanami/values.yaml"]


def main() -> int:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED if not path.is_file()]
    if missing:
        print("Missing supported image inputs:", *missing, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
