"""Quarantine-only legacy migration entrypoint.

Legacy memory formats are deliberately unsupported: this program never loads
pickle/object payloads and reports a deterministic dry-run refusal instead.
"""
from __future__ import annotations
import argparse, hashlib, json

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("source"); parser.add_argument("--dry-run", action="store_true", required=True); args=parser.parse_args()
    print(json.dumps({"source_digest":hashlib.sha256(args.source.encode()).hexdigest(),"imported":0,"quarantined":1,"reason":"legacy_migration_unsupported","raw_data_logged":False},sort_keys=True))
    return 2
if __name__ == "__main__": raise SystemExit(main())
