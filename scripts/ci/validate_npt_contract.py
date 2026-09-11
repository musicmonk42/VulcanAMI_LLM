#!/usr/bin/env python3
"""Validate the canonical, digest-bound NPT engineering contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vulcan.assurance.npt_contract import DEFAULT_CONTRACT, load_contract

if __name__ == "__main__":
    contract = load_contract()
    print(
        f"NPT engineering contract verified: {DEFAULT_CONTRACT}; "
        f"dimensions={len(contract['dimensions'])}; theory_status={contract['theory_status']}"
    )
