"""Named test-only fault-injection points."""
from __future__ import annotations

from dataclasses import dataclass, field
import os

_TEST_TOKEN = "VULCAN_TEST_FAULTS_ONLY"


class FaultInjected(RuntimeError):
    pass


@dataclass(slots=True)
class FaultInjector:
    enabled: bool = False
    armed: set[str] = field(default_factory=set)

    @classmethod
    def disabled(cls) -> "FaultInjector":
        return cls(False, set())

    @classmethod
    def for_tests(cls, token: str) -> "FaultInjector":
        if token != _TEST_TOKEN or os.getenv("VULCAN_ENABLE_TEST_FAULTS") != _TEST_TOKEN:
            raise RuntimeError("fault injection is test-only and explicitly gated")
        return cls(True, set())

    def arm(self, name: str) -> None:
        if not self.enabled:
            raise RuntimeError("cannot arm disabled fault injector")
        _validate_name(name)
        self.armed.add(name)

    def check(self, name: str) -> None:
        _validate_name(name)
        if self.enabled and name in self.armed:
            self.armed.remove(name)
            raise FaultInjected(name)


def _validate_name(name: str) -> None:
    if not name or not all(ch.islower() or ch.isdigit() or ch in "._-" for ch in name):
        raise ValueError("fault point name must be lowercase dotted token")
