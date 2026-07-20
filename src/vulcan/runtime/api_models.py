"""Typed immutable API contracts for the canonical runtime."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


class ApiContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ReasonRequest(ApiContract):
    message: StrictStr = Field(..., min_length=1, max_length=2048)
    conversation_id: StrictStr | None = Field(default=None, max_length=128)


class MemoryWriteBody(ApiContract):
    key: StrictStr = Field(..., min_length=1, max_length=64)
    value: StrictStr = Field(..., min_length=1, max_length=64)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=128)


class MemoryCorrectBody(MemoryWriteBody):
    base_revision: StrictInt = Field(..., ge=1, le=1_000_000)


class BundleBody(ApiContract):
    bundle: dict[str, object]


class ProposalBody(ApiContract):
    proposal: dict[str, object]


class ApprovalRejectBody(ApiContract):
    approval_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)
