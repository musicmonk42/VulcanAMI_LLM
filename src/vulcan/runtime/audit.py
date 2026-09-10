"""Compatibility imports for the persistence-owned canonical audit.

Remove this module once callers (including tests using the legacy private hash
helpers) import :mod:`vulcan.persistence.audit` directly.  The runtime owns no
audit persistence implementation.
"""
from vulcan.persistence.audit import (
    AuditDurabilityProfile,
    AuditError,
    AuditEvent,
    CanonicalAudit,
    Failpoint,
)
from vulcan.persistence.audit.store import _canonical, _hash_event

__all__ = [
    "AuditDurabilityProfile",
    "AuditError",
    "AuditEvent",
    "CanonicalAudit",
    "Failpoint",
]
