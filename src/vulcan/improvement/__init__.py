"""Governed improvement contracts shared by serving and offline operators."""

from .proposal import ImprovementProposal, ImprovementProposalStore, ProposalError

__all__ = ["ImprovementProposal", "ImprovementProposalStore", "ProposalError"]
