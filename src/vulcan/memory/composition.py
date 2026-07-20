"""Composition adapter for governed memory.

This module is the single backward-compatible composition authority.  It keeps
environment parsing outside the repository and requires callers that enable
memory to provide the already-created canonical audit owner as a borrowed
resource.
"""
from __future__ import annotations

from .governed import AuditPort, BorrowedAudit, GovernedMemoryPort, MemoryRuntimeConfig, compose_governed_memory

__all__ = ["AuditPort", "BorrowedAudit", "GovernedMemoryPort", "MemoryRuntimeConfig", "compose_governed_memory"]
