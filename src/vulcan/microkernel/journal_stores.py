"""Journal-backed serving repositories with caller-owned mutation boundaries.

These repositories never open connections and expose no commit operation.  Reads
are delegated to the single database owner; every mutation requires the caller's
active :class:`UnitOfWork`.
"""

from __future__ import annotations

from dataclasses import dataclass

from vulcan.graphix.epistemic import EpistemicCommit, dumps_commit, loads_commit

from .constitutional_journal import (
    ConstitutionalDatabase,
    JournalEvent,
    SuccessorError,
    UnitOfWork,
)
from .episode import CognitiveEpisode
from .episode_store import EpisodeConflict, EpisodeIntegrityError, episode_from_document


class JournalEpisodeStore:
    """Exact CognitiveEpisode documents in the constitutional journal."""

    def __init__(self, database: ConstitutionalDatabase):
        self.database = database
        self.path = str(database.path)

    def load(self, episode_id: str) -> CognitiveEpisode:
        rows = self.database.read(
            "SELECT head_digest,document FROM episode_documents WHERE episode_id=?",
            (episode_id,),
        )
        if not rows:
            raise EpisodeIntegrityError(f"unknown episode: {episode_id}")
        episode = episode_from_document(rows[0]["document"])
        if episode.digest != rows[0]["head_digest"]:
            raise EpisodeIntegrityError("journal episode head digest mismatch")
        return episode

    def write_genesis(self, uow: UnitOfWork, episode: CognitiveEpisode) -> None:
        document = episode.canonical_json()
        episode_from_document(document)
        uow._execute(
            "INSERT INTO episode_documents VALUES (?,?,?)",
            (episode.episode_id, episode.digest, document),
        )

    def advance(
        self, uow: UnitOfWork, expected: str, episode: CognitiveEpisode
    ) -> CognitiveEpisode:
        document = episode.canonical_json()
        episode_from_document(document)
        changed = uow._execute(
            "UPDATE episode_documents SET head_digest=?,document=? "
            "WHERE episode_id=? AND head_digest=?",
            (episode.digest, document, episode.episode_id, expected),
        ).rowcount
        if changed != 1:
            raise EpisodeConflict("journal episode head compare-and-swap failed")
        return episode

    def verify_all(self) -> None:
        for row in self.database.read(
            "SELECT episode_id,head_digest,document FROM episode_documents"
        ):
            episode = episode_from_document(row["document"])
            if (
                episode.episode_id != row["episode_id"]
                or episode.digest != row["head_digest"]
            ):
                raise EpisodeIntegrityError("journal episode document diverged")

    def close(self) -> None:
        """Lifecycle belongs to ConstitutionalDatabase, not this view."""


class JournalEpistemicStore:
    """Exact Graphix commits in the constitutional journal."""

    def __init__(self, database: ConstitutionalDatabase):
        self.database = database
        self.path = str(database.path)

    def head(self, episode_id: str) -> EpistemicCommit | None:
        rows = self.database.read(
            "SELECT d.document FROM epistemic_heads h JOIN epistemic_documents d "
            "ON d.epistemic_digest=h.epistemic_digest WHERE h.episode_id=?",
            (episode_id,),
        )
        return None if not rows else loads_commit(bytes(rows[0]["document"]))

    def write_document(
        self, uow: UnitOfWork, digest: str, commit: EpistemicCommit
    ) -> None:
        document = dumps_commit(commit)
        decoded = loads_commit(document)
        if decoded.commit_digest.removeprefix("sha256:") != digest:
            raise SuccessorError("epistemic document digest mismatch")
        uow._execute("INSERT INTO epistemic_documents VALUES (?,?)", (digest, document))

    def reconcile(self) -> None:
        for row in self.database.read(
            "SELECT epistemic_digest,document FROM epistemic_documents"
        ):
            commit = loads_commit(bytes(row["document"]))
            if commit.commit_digest.removeprefix("sha256:") != row["epistemic_digest"]:
                raise SuccessorError("journal epistemic document diverged")

    def close(self) -> None:
        """Lifecycle belongs to ConstitutionalDatabase, not this view."""


@dataclass(frozen=True, slots=True)
class JournalLineageHead:
    branch_id: str
    head_episode_id: str
    head_digest: str
    active_episode_ids: tuple[str, ...]
    past_episode_ids: tuple[str, ...]


class JournalLineageStore:
    """Read-only view of the journal's canonical lineage rows."""

    def __init__(self, database: ConstitutionalDatabase):
        self.database = database
        self.path = str(database.path)

    def load(self, branch_id: str) -> JournalLineageHead:
        rows = self.database.read(
            "SELECT head_episode_id,head_digest FROM lineage_branches WHERE branch_id=?",
            (branch_id,),
        )
        if not rows:
            raise KeyError(f"unknown lineage branch: {branch_id}")
        memberships = self.database.read(
            "SELECT episode_id,status FROM lineage_membership "
            "WHERE branch_id=? ORDER BY admitted_commit_seq,episode_id",
            (branch_id,),
        )
        return JournalLineageHead(
            branch_id,
            rows[0]["head_episode_id"],
            rows[0]["head_digest"],
            tuple(
                row["episode_id"] for row in memberships if row["status"] == "active"
            ),
            tuple(row["episode_id"] for row in memberships if row["status"] == "past"),
        )

    def verify_all(self) -> None:
        self.database.verify()

    def close(self) -> None:
        """Lifecycle belongs to ConstitutionalDatabase, not this view."""


def emit_transition(
    uow: UnitOfWork,
    *,
    event_type: str,
    actor_digest: str,
    credential_provenance_digest: str,
    episode: CognitiveEpisode,
) -> None:
    """Emit the canonical episode event within its mutation transaction."""
    from datetime import datetime, timezone

    uow.emit(
        JournalEvent(
            event_type,
            actor_digest,
            credential_provenance_digest,
            {"episode_digest": episode.digest, "episode_id": episode.episode_id},
            datetime.now(timezone.utc),
        )
    )
