from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vulcan.microkernel.constitutional_journal import ConstitutionalDatabase


@pytest.mark.parametrize(
    "boundary",
    (
        "after_artifact",
        "after_head",
        "after_terminal",
        "after_outbox",
        "before_commit",
        "after_commit",
    ),
)
def test_parent_kills_proven_child_boundary_and_reopens(tmp_path, boundary):
    command = [
        sys.executable,
        "scripts/qualification/journal_recovery_harness.py",
        "parent",
        str(tmp_path / "constitutional.sqlite3"),
        str(tmp_path / "reached.json"),
        boundary,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["boundary"] == boundary


def test_optimized_child_recovery(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/journal_recovery_harness.py",
            "parent",
            str(tmp_path / "constitutional.sqlite3"),
            str(tmp_path / "reached.json"),
            "after_commit",
            "--optimized",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout)["status"] == "PASS"


def test_unreached_failpoint_is_not_executed(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/journal_recovery_harness.py",
            "parent",
            str(tmp_path / "constitutional.sqlite3"),
            str(tmp_path / "reached.json"),
            "unknown-boundary",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "NOT_EXECUTED"


def test_live_writer_lock_contention_fails_closed_then_recovers(tmp_path):
    database_path = tmp_path / "constitutional.sqlite3"
    marker = tmp_path / "reached.json"
    child = subprocess.Popen(
        [
            sys.executable,
            "scripts/qualification/journal_recovery_harness.py",
            "child",
            str(database_path),
            str(marker),
            "after_artifact",
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "child did not prove its lock-holding failpoint"
        with pytest.raises(Exception, match="locked"):
            contender = ConstitutionalDatabase(database_path, busy_timeout_ms=1)
            try:
                with contender.transaction():
                    pass
            finally:
                contender.close()
    finally:
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    reopened = ConstitutionalDatabase(database_path)
    reopened.verify()
    reopened.close()


def test_truncated_wal_is_never_reported_as_pass(tmp_path):
    database_path = tmp_path / "constitutional.sqlite3"
    marker = tmp_path / "reached.json"
    child = subprocess.Popen(
        [
            sys.executable,
            "scripts/qualification/journal_recovery_harness.py",
            "child",
            str(database_path),
            str(marker),
            "after_commit",
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "child did not prove post-commit failpoint"
        wal = Path(f"{database_path}-wal")
        assert wal.exists() and wal.stat().st_size > 32
        with wal.open("r+b") as stream:
            stream.truncate(wal.stat().st_size // 2)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    reopened = ConstitutionalDatabase(database_path)
    reopened.verify()
    commits = reopened.read("SELECT count(*) AS n FROM journal_commits")[0]["n"]
    assert commits in (0, 1), "truncated WAL produced a partial successor"
    reopened.close()
