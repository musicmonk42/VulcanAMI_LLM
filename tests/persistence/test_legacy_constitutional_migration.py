from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from vulcan.microkernel.constitutional_journal import ConstitutionalDatabase
from vulcan.microkernel.episode import ActorBinding
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.lineage import LineageStore
from vulcan.persistence.legacy_migration import (
    MigrationReconciliationError,
    legacy_source_digest,
    migrate_legacy_stores,
)

ACTOR = ActorBinding._from_verified_identity(
    tenant="migration", issuer="operator-issuer", subject="migration-workload"
)


def legacy_sources(tmp_path):
    episodes = tmp_path / "legacy-episodes.sqlite3"
    epistemic = tmp_path / "legacy-epistemic.sqlite3"
    EpisodeStore(episodes).close()
    LineageStore(episodes)
    EpistemicStore(epistemic).close()
    return episodes, epistemic


def migrate(tmp_path, episodes, epistemic):
    return migrate_legacy_stores(
        episode_database=episodes,
        epistemic_database=epistemic,
        target_database=tmp_path / "constitutional.sqlite3",
        report_path=tmp_path / "migration-report.json",
        migration_actor=ACTOR,
        credential_provenance_digest="c" * 64,
        migration_time=datetime(2026, 9, 10, tzinfo=timezone.utc),
        expected_source_digests={
            "episode-lineage": legacy_source_digest(episodes),
            "epistemic": legacy_source_digest(epistemic),
        },
    )


def test_migration_is_exclusive_verified_one_way_and_preserves_sources(tmp_path):
    episodes, epistemic = legacy_sources(tmp_path)
    report = migrate(tmp_path, episodes, epistemic)
    assert report["status"] == "INSTALLED"
    assert report["classification_rule"].startswith("historical labels")
    assert os.stat(episodes).st_mode & 0o777 == 0o440
    assert os.stat(epistemic).st_mode & 0o777 == 0o440
    database = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    database.verify()
    rows = database.read(
        "SELECT payload FROM transactional_outbox ORDER BY commit_seq,event_ordinal"
    )
    assert rows
    assert all(
        json.loads(row["payload"])["legacy_attribution"] == "LEGACY_UNVERIFIED"
        for row in rows
    )
    database.close()
    with pytest.raises(Exception, match="target already exists"):
        migrate(tmp_path, episodes, epistemic)


def test_reconciliation_failure_reports_and_does_not_install(tmp_path):
    episodes, epistemic = legacy_sources(tmp_path)
    connection = sqlite3.connect(epistemic)
    connection.execute("DROP TABLE epistemic_heads")
    connection.commit()
    connection.close()
    with pytest.raises(MigrationReconciliationError, match="schema is incomplete"):
        migrate(tmp_path, episodes, epistemic)
    assert not (tmp_path / "constitutional.sqlite3").exists()
    report = json.loads((tmp_path / "migration-report.json").read_text())
    assert report["status"] == "RECONCILIATION_REQUIRED"
    assert "error" in report


def test_source_digest_mismatch_stops_before_temporary_import(tmp_path):
    episodes, epistemic = legacy_sources(tmp_path)
    with pytest.raises(MigrationReconciliationError, match="digest mismatch"):
        migrate_legacy_stores(
            episode_database=episodes,
            epistemic_database=epistemic,
            target_database=tmp_path / "constitutional.sqlite3",
            report_path=tmp_path / "migration-report.json",
            migration_actor=ACTOR,
            credential_provenance_digest="c" * 64,
            migration_time=datetime(2026, 9, 10, tzinfo=timezone.utc),
            expected_source_digests={
                "episode-lineage": "0" * 64,
                "epistemic": legacy_source_digest(epistemic),
            },
        )
    assert not (tmp_path / "constitutional.sqlite3").exists()
