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


class FeedbackRequest(BaseModel):
    """
    Request model for submitting RLHF (Reinforcement Learning from Human Feedback).
    
    This model captures structured feedback on AI responses to enable continuous
    improvement through human preference learning.
    
    Attributes:
        feedback_type: Type of feedback being submitted. Valid values:
            - "rating": Numeric rating feedback
            - "correction": Correction to the AI's response
            - "preference": Preference between multiple responses
            - "thumbs": Simple thumbs up/down feedback
        query_id: Unique identifier of the original query that generated the response
        response_id: Unique identifier of the response being rated
        reward_signal: Normalized reward signal in range [-1.0, 1.0] where:
            - 1.0 = Maximum positive feedback
            - 0.0 = Neutral feedback
            - -1.0 = Maximum negative feedback
        content: Optional structured feedback content (corrections, comments, etc.)
        context: Optional contextual information about the feedback submission
        
    Example:
        >>> feedback = FeedbackRequest(
        ...     feedback_type="rating",
        ...     query_id="q_12345",
        ...     response_id="r_67890",
        ...     reward_signal=0.8
        ... )
    """
    feedback_type: str = Field(
        default="rating",
        description="Type of feedback: rating, correction, preference, thumbs",
        pattern="^(rating|correction|preference|thumbs)$"
    )
    query_id: str = Field(
        ...,
        description="ID of the original query",
        min_length=1,
        max_length=256
    )
    response_id: str = Field(
        ...,
        description="ID of the response being rated",
        min_length=1,
        max_length=256
    )
    reward_signal: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized reward signal from -1.0 (worst) to 1.0 (best)"
    )
    content: Optional[Any] = Field(
        default=None,
        description="Optional feedback content (corrections, comments, etc.)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional contextual metadata about the feedback"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feedback_type": "rating",
                "query_id": "q_12345",
                "response_id": "r_67890",
                "reward_signal": 0.8,
                "content": {"comment": "Great response!"},
                "context": {"user_id": "user_001"}
            }
        }
    )


class ThumbsFeedbackRequest(BaseModel):
    """
    Simplified request model for binary thumbs up/down feedback.
    
    This model provides a simple interface for collecting binary user satisfaction
    feedback, typically triggered by UI thumb buttons. It's a simplified alternative
    to the full FeedbackRequest for quick user interactions.
    
    Attributes:
        query_id: Unique identifier of the original query
        response_id: Unique identifier of the response being rated
        is_positive: True for thumbs up (positive), False for thumbs down (negative)
        
    Example:
        >>> feedback = ThumbsFeedbackRequest(
        ...     query_id="q_12345",
        ...     response_id="r_67890",
        ...     is_positive=True
        ... )
    """
    query_id: str = Field(
        ...,
        description="ID of the original query",
        min_length=1,
        max_length=256
    )
    response_id: str = Field(
        ...,
        description="ID of the response being rated",
        min_length=1,
        max_length=256
    )
    is_positive: bool = Field(
        default=True,
        description="True for thumbs up (positive), False for thumbs down (negative)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_id": "q_12345",
                "response_id": "r_67890",
                "is_positive": True
            }
        }
    )


class UnifiedChatRequest(BaseModel):
    """
    Request model for the unified chat endpoint with full platform integration.
    
    This endpoint orchestrates the entire VulcanAMI platform, providing access to:
    - Multi-modal processing and understanding
    - Long-term and episodic memory systems
    - Safety validation and governance
    - Multiple reasoning engines (symbolic, probabilistic, causal, analogical)
    - Planning and goal management systems
    - World model predictions and simulations
    
    Attributes:
        message: The user's chat message/query (required)
        max_tokens: Maximum tokens in the response (default: 2000)
        history: Conversation history as list of {"role": "...", "content": "..."} dicts
        conversation_id: Optional conversation identifier for context continuity.
            If None, a new conversation ID will be auto-generated.
        enable_reasoning: Enable advanced reasoning engines (default: True)
        enable_memory: Enable long-term memory search and retrieval (default: True)
        enable_safety: Enable safety validation and compliance checking (default: True)
        enable_planning: Enable hierarchical planning systems (default: True)
        enable_causal: Enable causal reasoning and inference (default: True)
        
    Example:
        >>> request = UnifiedChatRequest(
        ...     message="Explain quantum entanglement",
        ...     max_tokens=1500,
        ...     history=[{"role": "user", "content": "Hi"}],
        ...     conversation_id="conv_12345"
        ... )
        
    Note:
        Feature toggles (enable_*) allow fine-grained control over platform
        capabilities for performance tuning or specialized use cases.
    """
    message: str = Field(
        ...,
        description="User's chat message or query",
        min_length=1,
        max_length=10000
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=8000,
        description="Maximum tokens in the response"
    )
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history with role/content dictionaries"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for context continuity (auto-generated if None)",
        max_length=256
    )
    # Feature toggles for fine-grained control
    enable_reasoning: bool = Field(
        default=True,
        description="Enable advanced reasoning engines"
    )
    enable_memory: bool = Field(
        default=True,
        description="Enable long-term memory search and retrieval"
    )
    enable_safety: bool = Field(
        default=True,
        description="Enable safety validation and compliance checking"
    )
    enable_planning: bool = Field(
        default=True,
        description="Enable hierarchical planning systems"
    )
    enable_causal: bool = Field(
        default=True,
        description="Enable causal reasoning and inference"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Explain quantum entanglement",
                "max_tokens": 1500,
                "history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ],
                "conversation_id": "conv_12345",
                "enable_reasoning": True,
                "enable_memory": True,
                "enable_safety": True,
                "enable_planning": True,
                "enable_causal": True
            }
        }
    )


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


class VulcanResponse(BaseModel):
    """Response model for VULCAN reasoning system direct responses.
    
    Used when reasoning engines provide high-confidence results
    that can be returned directly without LLM synthesis.
    """
    response: str
    systems_used: List[str] = []
    confidence: float = 0.0
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
    "FeedbackRequest",
    "ThumbsFeedbackRequest",
    "UnifiedChatRequest",
    # Response models
    "StepResponse",
    "ChatMessage",
    "ChatResponse",
    "VulcanResponse",
    "StatusResponse",
    "ConfigResponse",
    "ImprovementApproval",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
]


# Module initialization logging
logger.debug(f"API models module v{__version__} loaded with {len(__all__)} models")
