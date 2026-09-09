"""Retired serving self-improvement compatibility module.

The old runtime owner, approval authority, CSIU alias, transaction, and install
surface were removed by Wave 2.4.  Proposal contracts live in
``vulcan.improvement.proposal``; privileged operator capabilities live only in
``vulcan.improvement.offline``.  Remove this import tombstone when downstream
imports no longer reference the historical module path.
"""
from vulcan.improvement.proposal import ImprovementProposal, ImprovementProposalStore, ProposalError

__all__ = ["ImprovementProposal", "ImprovementProposalStore", "ProposalError"]
