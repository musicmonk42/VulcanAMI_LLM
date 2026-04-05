"""Canonical protocols for Vulcan AMI cross-cutting concerns.

This package sits at the bottom of the dependency graph.
All dependency arrows point inward -- no imports from src/vulcan/ or callers.
"""
from src.protocols.consensus import ConsensusProtocol
from src.protocols.audit import AuditProtocol, AuditEvent
from src.protocols.safety import SafetyProtocol, SafetyResult

__all__ = [
    "ConsensusProtocol",
    "AuditProtocol",
    "AuditEvent",
    "SafetyProtocol",
    "SafetyResult",
]
