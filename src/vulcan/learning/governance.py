"""Governed activation for shadow learning routing policies.

A candidate routing policy can become active only through this dependency-light
transaction boundary: deterministic evaluation, server-recomputed TVD influence,
CSIU budget reservation, verifier-only alignment approval, audit prepare/commit,
CAS activation, durable influence commit, and idempotent publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib, json, math, os, tempfile, threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from vulcan.learning_bandit import SelectionRecord, ShadowLinUCBToolBandit

SCHEMA_VERSION = "vulcan-learning-policy-activation/1"
APPROVAL_SCHEMA = "vulcan-learning-alignment-approval/1"
MAX_SINGLE_INFLUENCE = 0.05
MAX_CUMULATIVE_INFLUENCE = 0.10


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("invalid approval timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class ActivationStatus(Enum):
    ACTIVATED = "activated"
    REPLAYED = "replayed"
    BLOCKED = "blocked"
    STALE = "stale"
    NOT_ACCEPTED = "not_accepted"


@dataclass(frozen=True)
class InfluenceMeasurement:
    evaluation_cohort_digest: str
    weighted_mean_tvd: float
    max_tvd: float
    charged_influence: float
    single_update_cap: float
    cumulative_window_cap: float
    sample_count: int


@dataclass(frozen=True)
class LearningPolicyActivationProposal:
    schema_version: str
    candidate_policy_digest: str
    candidate_policy_revision: int
    active_policy_digest: str
    active_policy_revision: int
    evaluation_cohort_digest: str
    observation_set_digest: str
    weighted_mean_tvd: float
    max_tvd: float
    charged_influence: float
    sample_count: int
    alignment_revision: int
    alignment_digest: str
    csiu_policy_digest: str
    csiu_snapshot_digest: str
    requested_activation_time_utc: str
    cooldown_id: str
    proposal_digest: str = ""
    def __post_init__(self):
        body = self.to_dict(False); dg = _digest(body)
        if self.proposal_digest and self.proposal_digest != dg: raise ValueError("proposal digest mismatch")
        object.__setattr__(self, "proposal_digest", dg)
    def to_dict(self, include_digest=True):
        d = dict(self.__dict__); d.pop("proposal_digest", None)
        if include_digest: d["proposal_digest"] = self.proposal_digest
        return d


@dataclass(frozen=True)
class AlignmentApproval:
    schema_version: str
    approval_id: str
    issuer_id: str
    proposal_digest: str
    candidate_policy_digest: str
    active_policy_digest: str
    active_policy_revision: int
    evaluation_cohort_digest: str
    charged_influence: float
    alignment_digest: str
    expires_at_utc: str
    approval_digest: str = ""
    used: bool = False
    def __post_init__(self):
        if self.issuer_id.startswith("learning-owner"):
            raise ValueError("learning owner cannot issue approval")
        _parse_utc(self.expires_at_utc)
        body = self.to_dict(False); dg = _digest(body)
        if self.approval_digest and self.approval_digest != dg: raise ValueError("approval digest mismatch")
        object.__setattr__(self, "approval_digest", dg)
    def to_dict(self, include_digest=True):
        d = dict(self.__dict__); d.pop("approval_digest", None)
        if include_digest: d["approval_digest"] = self.approval_digest
        return d


@dataclass(frozen=True)
class ActivationResult:
    status: ActivationStatus
    proposal_digest: str
    active_policy_digest: str
    charged_influence: float
    reason: str = ""


class AlignmentApprovalVerifier:
    def __init__(self): self._used: set[str] = set()
    def verify(self, approval: AlignmentApproval, proposal: LearningPolicyActivationProposal) -> None:
        if approval.approval_id in self._used or approval.used: raise ValueError("approval reused")
        if _parse_utc(approval.expires_at_utc) <= datetime.now(timezone.utc): raise ValueError("approval expired")
        checks = {
            "proposal_digest": proposal.proposal_digest,
            "candidate_policy_digest": proposal.candidate_policy_digest,
            "active_policy_digest": proposal.active_policy_digest,
            "active_policy_revision": proposal.active_policy_revision,
            "evaluation_cohort_digest": proposal.evaluation_cohort_digest,
            "charged_influence": proposal.charged_influence,
            "alignment_digest": proposal.alignment_digest,
        }
        for name, expected in checks.items():
            if getattr(approval, name) != expected: raise ValueError("approval binding mismatch")
    def consume(self, approval: AlignmentApproval) -> None:
        if approval.approval_id in self._used: raise ValueError("approval reused")
        self._used.add(approval.approval_id)


class DurableInfluenceLedger:
    def __init__(self, path: str | os.PathLike[str], *, single_cap: float = MAX_SINGLE_INFLUENCE, cumulative_cap: float = MAX_CUMULATIVE_INFLUENCE):
        self.path = Path(path); self.single_cap = float(single_cap); self.cumulative_cap = float(cumulative_cap); self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink(): raise ValueError("symlinked influence ledger")
        if not self.path.exists(): self._write({"schema_version": SCHEMA_VERSION, "records": []})
        self._read()
    def _read(self):
        doc = json.loads(self.path.read_text(), object_pairs_hook=_reject_dups, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite ledger")))
        if doc.get("schema_version") != SCHEMA_VERSION or not isinstance(doc.get("records"), list): raise ValueError("ledger schema mismatch")
        return doc
    def _write(self, doc):
        raw = _canonical(doc).encode(); fd, tmp = tempfile.mkstemp(prefix=".learning-ledger.", dir=self.path.parent)
        with os.fdopen(fd, "wb") as fh: fh.write(raw); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, self.path)
    def consumed(self) -> float:
        return sum(float(r["charged_influence"]) for r in self._read()["records"])
    def reserve(self, proposal: LearningPolicyActivationProposal) -> None:
        if proposal.charged_influence > self.single_cap: raise ValueError("single influence cap exceeded")
        if self.consumed() + proposal.charged_influence > self.cumulative_cap: raise ValueError("cumulative influence cap exceeded")
    def commit(self, proposal: LearningPolicyActivationProposal) -> None:
        with self._lock:
            doc = self._read()
            if any(r["proposal_digest"] == proposal.proposal_digest for r in doc["records"]): return
            if sum(float(r["charged_influence"]) for r in doc["records"]) + proposal.charged_influence > self.cumulative_cap: raise ValueError("cumulative influence cap exceeded")
            doc["records"].append({"proposal_digest": proposal.proposal_digest, "charged_influence": proposal.charged_influence, "timestamp_utc": _utc()})
            self._write(doc)


class GovernedLearningActivator:
    def __init__(self, *, bandit: ShadowLinUCBToolBandit, audit: Any, ledger: DurableInfluenceLedger, verifier: AlignmentApprovalVerifier):
        self.bandit = bandit; self.audit = audit; self.ledger = ledger; self.verifier = verifier; self._published: dict[str, ActivationResult] = {}; self._lock = threading.RLock()
    def measure_influence(self, records: Sequence[SelectionRecord]) -> InfluenceMeasurement:
        if not records: raise ValueError("empty evaluation cohort")
        tvds=[]; cohort=[]
        for r in records:
            actions=sorted(set(r.active_distribution)|set(r.candidate_distribution))
            tvd=0.5*sum(abs(float(r.active_distribution.get(a,0.0))-float(r.candidate_distribution.get(a,0.0))) for a in actions)
            if not math.isfinite(tvd) or tvd<0 or tvd>1: raise ValueError("invalid tvd")
            tvds.append(tvd); cohort.append({"context_digest":r.context_digest,"active":dict(r.active_distribution),"candidate":dict(r.candidate_distribution),"tvd":tvd})
        mean=sum(tvds)/len(tvds); mx=max(tvds); charged=max(mean,mx)
        return InfluenceMeasurement(_digest({"schema":"vulcan-learning-eval-cohort/1","contexts":cohort}), mean, mx, charged, self.ledger.single_cap, self.ledger.cumulative_cap, len(records))
    def propose(self, *, alignment_revision:int, alignment_digest:str, csiu_policy_digest:str, csiu_snapshot_digest:str, observation_digests:Sequence[str]) -> LearningPolicyActivationProposal:
        records=self.bandit.selection_records(); m=self.measure_influence(records)
        return LearningPolicyActivationProposal(SCHEMA_VERSION,self.bandit.candidate_policy_digest,self.bandit.candidate_policy_revision,self.bandit.active_policy_digest,self.bandit.active_policy_revision,m.evaluation_cohort_digest,_digest({"observations":sorted(observation_digests)}),m.weighted_mean_tvd,m.max_tvd,m.charged_influence,m.sample_count,alignment_revision,alignment_digest,csiu_policy_digest,csiu_snapshot_digest,_utc(),f"learning-cooldown-{self.bandit.candidate_policy_digest[:16]}")
    def activate(self, proposal: LearningPolicyActivationProposal, approval: AlignmentApproval, *, failpoint: str|None=None) -> ActivationResult:
        with self._lock:
            if proposal.proposal_digest in self._published: return self._published[proposal.proposal_digest]
            current = self.bandit.active_policy_digest
            if current != proposal.active_policy_digest: return ActivationResult(ActivationStatus.STALE, proposal.proposal_digest, current, 0.0, "stale active revision")
            measured = self.measure_influence(self.bandit.selection_records())
            if measured.charged_influence != proposal.charged_influence or measured.evaluation_cohort_digest != proposal.evaluation_cohort_digest:
                return ActivationResult(ActivationStatus.BLOCKED, proposal.proposal_digest, current, 0.0, "candidate influence underreporting detected")
            self.verifier.verify(approval, proposal)
            self.ledger.reserve(proposal)
            if failpoint == "before_prepared_audit": raise RuntimeError("before_prepared_audit")
            self.audit.append("learning.policy_activation_prepared", {"proposal_digest": proposal.proposal_digest, "candidate_policy_digest": proposal.candidate_policy_digest, "active_policy_digest": proposal.active_policy_digest, "charged_influence": proposal.charged_influence})
            if failpoint == "after_prepared_audit": raise RuntimeError("after_prepared_audit")
            try:
                active = self.bandit.activate_candidate(expected_active_digest=proposal.active_policy_digest, expected_candidate_digest=proposal.candidate_policy_digest)
                if failpoint == "after_cas_before_commit": raise RuntimeError("after_cas_before_commit")
                self.ledger.commit(proposal)
                if failpoint == "after_commit_before_audit": raise RuntimeError("after_commit_before_audit")
                self.audit.append("learning.policy_activation_committed", {"proposal_digest": proposal.proposal_digest, "active_policy_digest": active, "charged_influence": proposal.charged_influence})
                self.verifier.consume(approval)
                result=ActivationResult(ActivationStatus.ACTIVATED, proposal.proposal_digest, active, proposal.charged_influence)
                self._published[proposal.proposal_digest]=result
                return result
            except Exception:
                try: self.audit.append("learning.policy_activation_aborted", {"proposal_digest": proposal.proposal_digest, "result_category":"aborted"})
                except Exception: pass
                raise


def issue_alignment_approval(*, proposal: LearningPolicyActivationProposal, issuer_id: str, expires_at_utc: str) -> AlignmentApproval:
    return AlignmentApproval(APPROVAL_SCHEMA, f"learning-approval-{proposal.proposal_digest[:24]}", issuer_id, proposal.proposal_digest, proposal.candidate_policy_digest, proposal.active_policy_digest, proposal.active_policy_revision, proposal.evaluation_cohort_digest, proposal.charged_influence, proposal.alignment_digest, expires_at_utc)


def _reject_dups(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise ValueError("duplicate JSON key")
        d[k]=v
    return d
