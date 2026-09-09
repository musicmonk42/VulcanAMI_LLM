"""Dependency-light safety contracts.

Research validators are available from their explicit modules; importing this
package never mutates ``sys.path`` or initializes optional integrations.
"""

from __future__ import annotations

from .safety_types import GovernanceOrchestrator, SafetyValidator

SAFETY_VALIDATOR_AVAILABLE = True
GOVERNANCE_ORCHESTRATOR_AVAILABLE = True
DQS_AVAILABLE = False
DQSValidator = None


class SafetyUnavailable:
    """Fail-closed marker for compatibility callers lacking a validator."""

    def validate(self, *args, **kwargs):
        return {"status": "unavailable", "safe": False}

    def clamp(self, *args, **kwargs):
        return None

    def get_status(self):
        return {"available": False, "reason": "explicit validator required"}


def get_safety_validator():
    """Compatibility accessor; serving composition never uses this function."""
    return SafetyUnavailable()


__all__ = [
    "get_safety_validator",
    "SafetyUnavailable",
    "SafetyValidator",
    "GovernanceOrchestrator",
    "DQSValidator",
    "SAFETY_VALIDATOR_AVAILABLE",
    "GOVERNANCE_ORCHESTRATOR_AVAILABLE",
    "DQS_AVAILABLE",
]
