from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from vulcan.graphix.epistemic import Claim, ClaimStatus, EpistemicCommit, Proposition
from vulcan.graphix.verifier import VerifiedEpistemicCandidate, phase_b_registry
from vulcan.microkernel.authority import AuthorityError
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.journal_transactions import (
    JournalConstitutionalTransactionService,
)
from vulcan.microkernel.transactions import ConstitutionalTransactionService

D = "sha256:" + "1" * 64


def _bare_commit() -> EpistemicCommit:
    claim = Claim(
        "claim:one",
        Proposition("proposition:one", "request", "support", "unsupported"),
        ClaimStatus.UNKNOWN,
        "episode:one",
        D,
    )
    return EpistemicCommit(
        "commit:one",
        "episode:one",
        "episode:one",
        D,
        "principal:kernel",
        D,
        D,
        D,
        D,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        (claim,),
    )


def test_bare_epistemic_commit_is_rejected_before_store_access(tmp_path) -> None:
    registry = phase_b_registry(qualified_core_release=D)
    service = ConstitutionalTransactionService(
        EpisodeStore(tmp_path / "episode.sqlite3"),
        EpistemicStore(tmp_path / "epistemic.sqlite3"),
        registry,
    )
    with pytest.raises(AuthorityError, match="verifier-backed"):
        service.commit_epistemic_candidate(
            "episode:one", object(), _bare_commit()  # type: ignore[arg-type]
        )


def test_both_commit_ports_require_the_typed_verified_candidate() -> None:
    for owner in (
        ConstitutionalTransactionService,
        JournalConstitutionalTransactionService,
    ):
        annotation = (
            inspect.signature(owner.commit_epistemic_candidate)
            .parameters["candidate"]
            .annotation
        )
        assert annotation in {"VerifiedEpistemicCandidate", VerifiedEpistemicCandidate}
