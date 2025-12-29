# ============================================================
# VULCAN-AGI API Models Module
# Pydantic models for API requests and responses
# ============================================================
#
# This module defines all request and response models for the
# VULCAN-AGI API endpoints.
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# ============================================================
# REQUEST MODELS
# ============================================================


class StepRequest(BaseModel):
    """Request model for executing a cognitive step."""
    history: List[Any] = []
    context: Dict[str, Any]
    timeout: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "history": [],
                "context": {
                    "high_level_goal": "explore",
                    "raw_observation": "Test observation",
                },
            }
        }
    )


class PlanRequest(BaseModel):
    """Request model for planning endpoints."""
    goal: str
    context: Dict[str, Any] = {}
    method: str = "hierarchical"


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    query: str
    k: int = 10
    filters: Optional[Dict[str, Any]] = None


class ErrorReportRequest(BaseModel):
    """Request model for error reporting."""
    error_type: str
    error_message: str
    context: Optional[Dict[str, Any]] = None
    severity: str = "medium"


class ApprovalRequest(BaseModel):
    """Request model for approval operations."""
    approval_id: str
    approved: bool
    notes: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    prompt: str
    max_tokens: int = 2000  # Increased for diagnostic purposes


class ReasonRequest(BaseModel):
    """Request model for reasoning endpoint."""
    query: str
    context: Dict[str, Any] = {}


class ExplainRequest(BaseModel):
    """Request model for explanation endpoint."""
    concept: str
    context: Dict[str, Any] = {}


# ============================================================
# RESPONSE MODELS
# ============================================================


class StepResponse(BaseModel):
    """Response model for step execution."""
    action: Optional[str] = None
    reasoning: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Model for a single chat message."""
    role: str
    content: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    systems_used: List[str] = []
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    """Response model for status endpoints."""
    status: str
    health: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    version: Optional[str] = None


class ConfigResponse(BaseModel):
    """Response model for configuration endpoints."""
    deployment_mode: str
    settings: Dict[str, Any]


class ImprovementApproval(BaseModel):
    """Model for self-improvement approval."""
    approval_id: str
    objective_type: str
    description: str
    cost_estimate: float
    risk_level: str
    timestamp: float
    status: str = "pending"


class HealthStatus(str, Enum):
    """Enumeration of possible health statuses."""
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class ErrorType(str, Enum):
    """Enumeration of common error types."""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""
    status: HealthStatus = Field(default=HealthStatus.OK, description="Current health status")
    healthy: bool = Field(default=True, description="Overall health indicator")
    components: Optional[Dict[str, Any]] = Field(default=None, description="Component health details")
    version: Optional[str] = Field(default=None, description="Application version")
    uptime_seconds: Optional[float] = Field(default=None, description="Uptime in seconds")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoints."""
    prometheus_data: Optional[str] = Field(default=None, description="Prometheus-format metrics")
    internal_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Internal metrics data")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error description")
    error_type: Optional[ErrorType] = Field(default=None, description="Type of error")
    timestamp: Optional[float] = Field(default=None, description="Error timestamp")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Enums
    "HealthStatus",
    "ErrorType",
    # Request models
    "StepRequest",
    "PlanRequest",
    "MemorySearchRequest",
    "ErrorReportRequest",
    "ApprovalRequest",
    "ChatRequest",
    "ReasonRequest",
    "ExplainRequest",
    # Response models
    "StepResponse",
    "ChatMessage",
    "ChatResponse",
    "StatusResponse",
    "ConfigResponse",
    "ImprovementApproval",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
]


# Module initialization logging
logger.debug(f"API models module v{__version__} loaded with {len(__all__)} models")
