"""Dependency-light, code-owned safety validation for bounded responses."""
from __future__ import annotations

from vulcan.safety.safety_types import SafetyReport


class CanonicalResponseSafetyValidator:
    validator_identity = "canonical-response-safety/v1"

    def validate_response(self, response_text: str, context: str) -> SafetyReport:
        allowed = (
            isinstance(response_text, str)
            and 0 < len(response_text.encode("utf-8")) <= 4096
            and "\x00" not in response_text
            and isinstance(context, str)
            and len(context) <= 2048
        )
        return SafetyReport(
            safe=allowed,
            confidence=1.0,
            reasons=["bounded canonical response" if allowed else "invalid response bounds"],
            metadata={"validator_identity": self.validator_identity},
        )
