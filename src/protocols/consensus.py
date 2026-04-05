"""Canonical consensus protocol and implementation.

Merged from: consensus_engine.py (governance proposals, trust-weighted voting)
and consensus_manager.py (Raft-inspired leader election).
"""
from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class ProposalStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class VoteDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Agent:
    agent_id: str
    trust_weight: float = 1.0


@dataclass
class Vote:
    agent_id: str
    decision: VoteDecision
    timestamp: float = field(default_factory=time.time)


@dataclass
class Proposal:
    proposal_id: str
    content: dict
    status: ProposalStatus = ProposalStatus.DRAFT
    votes: list[Vote] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    quorum_ratio: float = 0.5


@runtime_checkable
class ConsensusProtocol(Protocol):
    """Interface for consensus implementations."""

    def register_agent(self, agent_id: str, trust_weight: float) -> None: ...

    def propose(self, proposal: dict) -> str: ...

    def vote(self, proposal_id: str, agent_id: str, decision: str) -> None: ...

    def evaluate(self, proposal_id: str) -> dict: ...

    def elect_leader(self) -> str | None: ...

    def get_proposal(self, proposal_id: str) -> Proposal | None: ...

    def shutdown(self) -> None: ...


class ConsensusEngine:
    """Canonical consensus implementation with trust-weighted voting."""

    def __init__(self, quorum_ratio: float = 0.5):
        self._agents: dict[str, Agent] = {}
        self._proposals: dict[str, Proposal] = {}
        self._leader: str | None = None
        self._lock = threading.RLock()
        self._quorum_ratio = quorum_ratio

    def register_agent(self, agent_id: str, trust_weight: float) -> None:
        with self._lock:
            self._agents[agent_id] = Agent(agent_id, trust_weight)

    def propose(self, proposal: dict) -> str:
        proposal_id = hashlib.md5(
            json.dumps(proposal, sort_keys=True).encode()
        ).hexdigest()[:12]
        with self._lock:
            self._proposals[proposal_id] = Proposal(
                proposal_id=proposal_id,
                content=proposal,
                status=ProposalStatus.OPEN,
                quorum_ratio=self._quorum_ratio,
            )
        return proposal_id

    def vote(self, proposal_id: str, agent_id: str, decision: str) -> None:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if not prop or prop.status != ProposalStatus.OPEN:
                return
            if agent_id not in self._agents:
                return
            prop.votes.append(Vote(agent_id, VoteDecision(decision)))

    def evaluate(self, proposal_id: str) -> dict:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if not prop:
                return {"verdict": "not_found"}
            total_weight = sum(
                self._agents[v.agent_id].trust_weight
                for v in prop.votes
                if v.agent_id in self._agents
            )
            approve_weight = sum(
                self._agents[v.agent_id].trust_weight
                for v in prop.votes
                if v.decision == VoteDecision.APPROVE
                and v.agent_id in self._agents
            )
            max_weight = sum(a.trust_weight for a in self._agents.values())
            if max_weight == 0:
                return {"verdict": "no_agents", "confidence": 0.0}
            quorum_met = (total_weight / max_weight) >= prop.quorum_ratio
            if not quorum_met:
                return {"verdict": "pending", "confidence": 0.0}
            confidence = approve_weight / total_weight if total_weight else 0
            if confidence > 0.5:
                prop.status = ProposalStatus.APPROVED
                return {"verdict": "approved", "confidence": confidence}
            prop.status = ProposalStatus.REJECTED
            return {"verdict": "rejected", "confidence": confidence}

    def elect_leader(self) -> str | None:
        with self._lock:
            if not self._agents:
                return None
            self._leader = max(
                self._agents.values(), key=lambda a: a.trust_weight
            ).agent_id
            return self._leader

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        return self._proposals.get(proposal_id)

    def shutdown(self) -> None:
        with self._lock:
            self._agents.clear()
            self._proposals.clear()
            self._leader = None
