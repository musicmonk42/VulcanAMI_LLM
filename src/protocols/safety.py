"""Canonical safety protocol and implementation.

Unified from: llm_executor.py (token/sequence validation),
safety_types.py (action validation), safety_governor.py (pattern matching).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SafetyViolation:
    pattern: str
    severity: str
    matched_text: str
    context: str = ""


@dataclass(frozen=True)
class SafetyResult:
    is_safe: bool
    violations: list[SafetyViolation]
    confidence: float
    level: SafetyLevel
    source: str = "canonical"


@runtime_checkable
class SafetyProtocol(Protocol):
    """Interface for safety validation implementations."""

    def validate(
        self, content: str, context: dict | None = None
    ) -> SafetyResult: ...

    def add_pattern(self, pattern: str, severity: str) -> None: ...

    def get_config(self) -> dict: ...


@dataclass
class PatternRule:
    pattern: re.Pattern
    severity: str
    description: str = ""


class SafetyValidator:
    """Canonical safety validator with configurable pattern layers."""

    def __init__(
        self,
        blacklist: set[str] | None = None,
        whitelist: set[str] | None = None,
    ):
        self._patterns: list[PatternRule] = []
        self._blacklist: set[str] = blacklist or set()
        self._whitelist: set[str] = whitelist or set()
        self._lock = threading.RLock()
        self._validation_count = 0
        self._init_default_patterns()

    def _init_default_patterns(self) -> None:
        high_risk = [
            (r"eval\s*\(", "Potential code execution via eval"),
            (r"exec\s*\(", "Potential code execution via exec"),
            (r"__import__\s*\(", "Dynamic import detected"),
            (r"os\.system\s*\(", "OS command execution"),
            (r"subprocess\.\w+\s*\(", "Subprocess execution"),
        ]
        for pat, desc in high_risk:
            self._patterns.append(
                PatternRule(re.compile(pat), "high", desc)
            )

    def validate(
        self, content: str, context: dict | None = None
    ) -> SafetyResult:
        ctx = context or {}
        with self._lock:
            self._validation_count += 1
        violations: list[SafetyViolation] = []
        content_lower = content.lower()
        for token in self._blacklist:
            if token.lower() in content_lower:
                violations.append(
                    SafetyViolation(
                        pattern=token,
                        severity="high",
                        matched_text=token,
                        context="blacklist",
                    )
                )
        if self._whitelist:
            has_allowed = any(
                w.lower() in content_lower for w in self._whitelist
            )
            if not has_allowed and content.strip():
                violations.append(
                    SafetyViolation(
                        pattern="whitelist",
                        severity="medium",
                        matched_text="",
                        context="No whitelisted terms found",
                    )
                )
        is_technical = ctx.get("source") == "internal"
        for rule in self._patterns:
            matches = rule.pattern.finditer(content)
            for match in matches:
                if is_technical and rule.severity != "high":
                    continue
                violations.append(
                    SafetyViolation(
                        pattern=rule.pattern.pattern,
                        severity=rule.severity,
                        matched_text=match.group(0),
                        context=rule.description,
                    )
                )
        has_high = any(v.severity == "high" for v in violations)
        if has_high:
            level = SafetyLevel.BLOCKED
        elif violations:
            level = SafetyLevel.CAUTION
        else:
            level = SafetyLevel.SAFE
        confidence = 1.0 - (len(violations) * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        return SafetyResult(
            is_safe=not has_high,
            violations=violations,
            confidence=confidence,
            level=level,
        )

    def add_pattern(self, pattern: str, severity: str) -> None:
        with self._lock:
            self._patterns.append(
                PatternRule(re.compile(pattern), severity)
            )

    def get_config(self) -> dict:
        return {
            "pattern_count": len(self._patterns),
            "blacklist_size": len(self._blacklist),
            "whitelist_size": len(self._whitelist),
            "validation_count": self._validation_count,
        }
