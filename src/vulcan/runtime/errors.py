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


class ApiErrorCategory(str, Enum):
    SCHEMA_INVALID = "schema_invalid"
    MALFORMED_JSON = "malformed_json"
    BODY_TOO_LARGE = "body_too_large"
    CONTENT_TYPE_UNSUPPORTED = "content_type_unsupported"
    ETAG_REQUIRED = "etag_required"
    ETAG_MALFORMED = "etag_malformed"
    CAS_STALE = "cas_stale"
    NOT_FOUND = "not_found"
    POLICY_REJECTED = "policy_rejected"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CONFLICT = "conflict"
    AUTHENTICATION_REQUIRED = "authentication_required"
    FORBIDDEN = "forbidden"
    RUNTIME_NOT_READY = "runtime_not_ready"


class ApiContractError(RuntimeError):
    def __init__(self, status_code: int, category: ApiErrorCategory, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.message = message
