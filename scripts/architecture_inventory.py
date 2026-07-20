#!/usr/bin/env python3
"""Generate the repository architecture inventory without importing targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vulcan.assurance.inventory import InventoryConfig, build_inventory, canonical_json, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--json", default="docs/generated/architecture-inventory.json")
    parser.add_argument("--markdown", default="docs/generated/architecture-inventory.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    inventory = build_inventory(InventoryConfig(root=root))
    json_bytes = canonical_json(inventory) + b"\n"
    markdown = render_markdown(inventory)
    json_path = root / args.json
    md_path = root / args.markdown
    if args.check:
        if json_path.read_bytes() != json_bytes:
            raise SystemExit(f"{json_path.relative_to(root)} is not current")
        if md_path.read_text(encoding="utf-8") != markdown:
            raise SystemExit(f"{md_path.relative_to(root)} is not current")
        return 0
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(json_bytes)
    md_path.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
