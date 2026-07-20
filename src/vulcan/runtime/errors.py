"""Typed startup failures for the canonical runtime."""
from __future__ import annotations

from enum import Enum


class StartupErrorCategory(str, Enum):
    SETTINGS_INVALID = "settings_invalid"
    SERVER_DEPENDENCY_MISSING = "server_dependency_missing"
    DEPLOYMENT_IMPORT_FAILED = "deployment_import_failed"
    DEPLOYMENT_CONSTRUCTION_FAILED = "deployment_construction_failed"
    WORLD_MISSING = "world_missing"
    SAFETY_MISSING = "safety_missing"
    REASONER_MISSING = "reasoner_missing"
    FILESYSTEM_UNAVAILABLE = "filesystem_unavailable"
    RUNTIME_UNHEALTHY = "runtime_unhealthy"


class StartupFailure(RuntimeError):
    """Operator-visible startup failure with safe public readiness code."""

    def __init__(self, category: StartupErrorCategory, operator_message: str, cause: BaseException | None = None) -> None:
        super().__init__(operator_message)
        self.category = category
        self.operator_message = operator_message
        self.cause = cause

    @property
    def public_code(self) -> str:
        return self.category.value
