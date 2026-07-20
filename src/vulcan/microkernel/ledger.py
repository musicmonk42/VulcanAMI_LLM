"""Authoritative in-memory claim ledger for Graphix Epistemic commits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from vulcan.graphix.epistemic import EpistemicCommit, EpistemicContractError

class LedgerError(EpistemicContractError): pass
class ClaimNotCommittedError(LedgerError): pass
class FailpointTriggered(LedgerError): pass

@dataclass
class AuthoritativeClaimLedger:
    """Single-authority append-only commit ledger.

    Failpoints bracket the externally observable append transition so tests can
    prove restart reconciliation without simulating success.
    """
    _commits: dict[str, EpistemicCommit] = field(default_factory=dict)
    _claims: dict[str, str] = field(default_factory=dict)
    failpoint: Callable[[str, EpistemicCommit], None] | None = None

    def append(self, commit: EpistemicCommit) -> str:
        if commit.commit_id in self._commits:
            if self._commits[commit.commit_id].commit_digest != commit.commit_digest:
                raise LedgerError("commit id collision")
            return commit.commit_digest
        if commit.prior_commit_digest is not None and commit.prior_commit_digest not in {c.commit_digest for c in self._commits.values()}:
            raise LedgerError("unknown prior commit")
        self._trip("before_append", commit)
        self._commits[commit.commit_id] = commit
        for claim in commit.claims:
            self._claims[claim.claim_id] = commit.commit_id
        self._trip("after_append", commit)
        return commit.commit_digest

    def require_committed_claim(self, claim_id: str) -> EpistemicCommit:
        commit_id = self._claims.get(claim_id)
        if commit_id is None:
            raise ClaimNotCommittedError("claim is not committed")
        return self._commits[commit_id]

    def reconcile(self) -> None:
        rebuilt: dict[str, str] = {}
        for commit_id, commit in self._commits.items():
            for claim in commit.claims:
                rebuilt[claim.claim_id] = commit_id
        self._claims = rebuilt

    def _trip(self, name: str, commit: EpistemicCommit) -> None:
        if self.failpoint is not None:
            self.failpoint(name, commit)
