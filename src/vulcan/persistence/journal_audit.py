"""Disposable deterministic projection of constitutional outbox receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from vulcan.constitution.primitives import canonical_json
from vulcan.microkernel.constitutional_journal import ConstitutionalDatabase

from .audit_contracts import AuditEvent


class JournalAuditProjector:
    """Rebuildable JSONL view; the journal remains the only authority."""

    def __init__(self, database: ConstitutionalDatabase, path: str | Path):
        self._database = database
        self.path = Path(path)
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("audit projection path cannot be symlinked")
        self._closed = False

    def _rows(self):
        self._require_open()
        self._database.verify()
        return self._database.read(
            "SELECT commit_seq,event_ordinal,event_type,actor_digest,"
            "credential_provenance_digest,payload,occurred_at,"
            "previous_receipt_digest,receipt_digest FROM transactional_outbox "
            "ORDER BY commit_seq,event_ordinal"
        )

    @staticmethod
    def _document(row) -> dict[str, object]:
        payload = json.loads(row["payload"])
        return {
            "actor_digest": row["actor_digest"],
            "commit_seq": row["commit_seq"],
            "credential_provenance_digest": row["credential_provenance_digest"],
            "event_ordinal": row["event_ordinal"],
            "event_type": row["event_type"],
            "occurred_at": row["occurred_at"],
            "payload": payload,
            "previous_receipt_digest": row["previous_receipt_digest"],
            "receipt_digest": row["receipt_digest"],
            "schema_version": "vulcan-audit-projection/1",
        }

    def bytes(self) -> bytes:
        return b"".join(
            canonical_json(self._document(row)) + b"\n" for row in self._rows()
        )

    def rebuild(self) -> bytes:
        """Atomically replace the disposable projection with canonical bytes."""
        raw = self.bytes()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.unlink(missing_ok=True)
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("audit projection write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return raw

    def readiness(self) -> None:
        """Gate only on authoritative/recoverable journal state, never JSONL."""
        self._require_open()
        self._database.verify()

    def deep_verify(self) -> None:
        """Projection absence or staleness is harmless; verify authority only."""
        self.readiness()

    def events_for_episode(self, episode_id: str) -> tuple[AuditEvent, ...]:
        events = []
        for sequence, row in enumerate(self._rows(), 1):
            document = self._document(row)
            payload = document["payload"]
            if not isinstance(payload, dict) or payload.get("episode_id") != episode_id:
                continue
            events.append(
                AuditEvent(
                    "vulcan-audit-projection/1",
                    sequence,
                    str(row["event_type"]),
                    str(row["occurred_at"]),
                    str(row["previous_receipt_digest"]),
                    dict(document),
                    str(row["receipt_digest"]),
                )
            )
        return tuple(events)

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("audit projector is closed")
