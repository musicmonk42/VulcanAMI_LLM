#!/usr/bin/env python3
"""Parent/child crash harness for named constitutional journal boundaries."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vulcan.microkernel.constitutional_journal import (  # noqa: E402
    ConstitutionalDatabase,
    ConstitutionalJournal,
    JournalEvent,
)
from vulcan.microkernel.episode import ActorBinding  # noqa: E402

ACTOR = ActorBinding._from_verified_identity(
    tenant="qualification", issuer="vulcan", subject="recovery-child"
)
NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)
CREDENTIAL = "c" * 64


def reached(marker: Path, name: str) -> None:
    marker.write_text(json.dumps({"pid": os.getpid(), "reached": name}) + "\n")
    descriptor = os.open(marker, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    while True:
        time.sleep(60)


def child(database_path: Path, marker: Path, boundary: str) -> None:
    def failpoint(name: str) -> None:
        if name == boundary:
            reached(marker, boundary)

    database = ConstitutionalDatabase(database_path, failpoint=failpoint)
    journal = ConstitutionalJournal()
    with database.transaction() as uow:
        actor = journal.bind_actor(uow, ACTOR)
        artifact = journal.put_artifact(
            uow, kind="qualification-artifact.v1", content=b"candidate"
        )
        if boundary == "after_artifact":
            reached(marker, boundary)
        journal.bind_command(
            uow,
            command_id="command-recovery",
            actor_digest=actor,
            credential_provenance_digest=CREDENTIAL,
            request_id="request-recovery",
            request_digest="d" * 64,
            operation="qualification",
            idempotency_key="recovery-key",
        )
        context = journal.put_artifact(
            uow, kind="admitted-context.v1", content=b"context"
        )
        journal.create_episode(
            uow,
            episode_id="episode-recovery",
            command_id="command-recovery",
            actor_digest=actor,
            context_digest=context,
        )
        if boundary == "after_head":
            reached(marker, boundary)
        journal.append_transition(
            uow,
            episode_id="episode-recovery",
            from_state="perceived",
            to_state="failed",
            transition_digest="e" * 64,
            actor_digest=actor,
            credential_provenance_digest=CREDENTIAL,
        )
        journal.record_terminal(
            uow,
            episode_id="episode-recovery",
            result_digest=artifact,
            actor_digest=actor,
            credential_provenance_digest=CREDENTIAL,
            terminal_state="failed",
        )
        if boundary == "after_terminal":
            reached(marker, boundary)
        uow.emit(
            JournalEvent(
                "qualification.committed",
                actor,
                CREDENTIAL,
                {"episode_id": "episode-recovery"},
                NOW,
            )
        )
        if boundary == "after_outbox":
            reached(marker, boundary)
    if boundary == "after_commit":
        reached(marker, boundary)
    raise RuntimeError(f"failpoint was not reached: {boundary}")


def parent(database_path: Path, marker: Path, boundary: str, optimized: bool) -> dict:
    marker.unlink(missing_ok=True)
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend([__file__, "child", str(database_path), str(marker), boundary])
    process = subprocess.Popen(command, cwd=ROOT)
    deadline = time.monotonic() + 15
    while (
        time.monotonic() < deadline and not marker.exists() and process.poll() is None
    ):
        time.sleep(0.02)
    if not marker.exists():
        process.kill()
        process.wait()
        return {"boundary": boundary, "status": "NOT_EXECUTED"}
    proof = json.loads(marker.read_text())
    if proof.get("reached") != boundary:
        process.kill()
        process.wait()
        return {"boundary": boundary, "status": "NOT_EXECUTED"}
    os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=5)
    database = ConstitutionalDatabase(database_path)
    database.verify()
    committed = database.read("SELECT count(*) AS n FROM journal_commits")[0]["n"]
    expected = 1 if boundary == "after_commit" else 0
    if committed != expected:
        database.close()
        raise AssertionError(
            f"{boundary}: expected {expected} commits, got {committed}"
        )
    database.close()
    return {
        "boundary": boundary,
        "committed": committed,
        "optimized": optimized,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("child", "parent"))
    parser.add_argument("database")
    parser.add_argument("marker")
    parser.add_argument("boundary")
    parser.add_argument("--optimized", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "child":
        child(Path(arguments.database), Path(arguments.marker), arguments.boundary)
        return 1
    result = parent(
        Path(arguments.database),
        Path(arguments.marker),
        arguments.boundary,
        arguments.optimized,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
