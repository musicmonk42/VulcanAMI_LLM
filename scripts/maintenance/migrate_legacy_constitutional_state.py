#!/usr/bin/env python3
"""Operator entry point for the exclusive one-way A07 migration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vulcan.microkernel.episode import ActorBinding  # noqa: E402
from vulcan.persistence.legacy_migration import migrate_legacy_stores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--epistemic", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--actor-binding", required=True)
    parser.add_argument("--credential-provenance-digest", required=True)
    parser.add_argument("--episode-source-digest", required=True)
    parser.add_argument("--epistemic-source-digest", required=True)
    parser.add_argument("--migration-time", required=True)
    arguments = parser.parse_args()
    actor = ActorBinding(**json.loads(Path(arguments.actor_binding).read_text()))
    migrate_legacy_stores(
        episode_database=arguments.episodes,
        epistemic_database=arguments.epistemic,
        target_database=arguments.target,
        report_path=arguments.report,
        migration_actor=actor,
        credential_provenance_digest=arguments.credential_provenance_digest,
        migration_time=datetime.fromisoformat(
            arguments.migration_time.replace("Z", "+00:00")
        ),
        expected_source_digests={
            "episode-lineage": arguments.episode_source_digest,
            "epistemic": arguments.epistemic_source_digest,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
