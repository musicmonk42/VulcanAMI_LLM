"""
Feedback Endpoints

RLHF (Reinforcement Learning from Human Feedback) endpoints for collecting
user feedback on AI responses to improve future performance.

This module provides endpoints for:
- Detailed feedback submission with reward signals
- Simplified thumbs up/down feedback for UI buttons  
- Feedback statistics retrieval

All endpoints use the `require_deployment()` or `get_deployment()` utility
functions to ensure consistent deployment access across standalone and
sub-app mounting scenarios.

Security considerations:
- All endpoints validate input parameters via Pydantic models
- Error messages are sanitized to prevent information leakage
- Feedback IDs use cryptographically secure random generation
- Rate limiting should be applied at the infrastructure level
"""

import logging
import secrets
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment, get_deployment

logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI documentation
router = APIRouter(tags=["feedback"])


@router.post("/v1/feedback", response_model=None)
async def submit_feedback(request: Request) -> Dict[str, Any]:
    """
    Submit human feedback for RLHF learning.
    
    This endpoint accepts feedback on AI responses to improve future performance
    through Reinforcement Learning from Human Feedback (RLHF).
    
    Args:
        request: FastAPI Request object containing JSON body with:
            - query_id: ID of the original query
            - response_id: ID of the AI response being rated
            - feedback_type: Type of feedback (e.g., "rating", "preference")
            - content: Feedback content/description
            - context: Optional context dictionary
            - reward_signal: Numeric reward signal (-1.0 to 1.0)
        
    Returns:
        Dict containing:
            - status: "accepted" on success
            - feedback_id: Unique cryptographically secure feedback identifier
            - message: Human-readable status message
        
    Raises:
        HTTPException: 400 if request body is invalid
        HTTPException: 503 if deployment or learning system not initialized
        HTTPException: 500 if feedback submission fails
    """
    # Parse and validate request body
    try:
        from vulcan.api.models import FeedbackRequest
        body = await request.json()
        feedback_data = FeedbackRequest(**body)
    except ValueError as e:
        logger.warning(f"Invalid feedback request body: {e}")
        raise HTTPException(status_code=400, detail="Invalid request body format")
    except Exception as e:
        logger.warning(f"Failed to parse feedback request: {type(e).__name__}")
        raise HTTPException(status_code=400, detail="Invalid request body")
    
    # Get deployment with consistent fallback behavior
    deployment = require_deployment(request)

    try:
        # Access learning system via deps.continual
        learning_system = None
        if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
            learning_system = deployment.collective.deps.continual
        
        if learning_system is None:
            raise HTTPException(status_code=503, detail="Learning system not available")

        # Submit feedback if the learning system supports it
        if hasattr(learning_system, "receive_feedback"):
            from vulcan.learning.learning_types import FeedbackData
            
            # Extract context safely
            context = getattr(feedback_data, 'context', None) or {}
            human_preference = context.get("preferred_response") if isinstance(context, dict) else None
            
            # Create feedback with cryptographically secure ID
            feedback = FeedbackData(
                feedback_id=f"fb_{secrets.token_urlsafe(16)}",
                timestamp=time.time(),
                feedback_type=str(feedback_data.feedback_type),
                content=str(feedback_data.content),
                context=context,
                agent_response=str(feedback_data.response_id),
                human_preference=human_preference,
                reward_signal=float(feedback_data.reward_signal),
                metadata={
                    "query_id": str(feedback_data.query_id),
                    "response_id": str(feedback_data.response_id),
                    "source": "api",
                },
            )
            
            learning_system.receive_feedback(feedback)
            
            # ALIGNMENT FIX: Also propagate detailed feedback to World Model's self-improvement drive
            # This ensures all feedback types impact CSIU (Continuous Self-Improvement Unit) and meta-reasoning
            world_model = None
            if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "world_model"):
                world_model = deployment.collective.deps.world_model
            
            if world_model and hasattr(world_model, "self_improvement_drive"):
                try:
                    # Notify self-improvement drive with detailed feedback context
                    feedback_context = {
                        "query_id": str(feedback_data.query_id),
                        "response_id": str(feedback_data.response_id),
                        "feedback_type": str(feedback_data.feedback_type),
                        "feedback_source": "api_detailed",
                        "reward_signal": float(feedback_data.reward_signal),
                        "content": str(feedback_data.content)
                    }
                    
                    # Record outcome based on reward signal (positive if reward > 0)
                    world_model.self_improvement_drive.record_outcome(
                        success=(feedback_data.reward_signal > 0),
                        metrics={"user_satisfaction": float(feedback_data.reward_signal)},
                        context=feedback_context
                    )
                    logger.info(f"[ALIGNMENT] Detailed feedback propagated to self-improvement drive: {feedback_data.reward_signal}")
                except Exception as e:
                    # Log but don't fail the request if self-improvement integration fails
                    logger.warning(f"[ALIGNMENT] Failed to propagate detailed feedback to self-improvement drive: {e}")
            
            return {
                "status": "accepted",
                "feedback_id": feedback.feedback_id,
                "message": "Feedback submitted successfully"
            }

        raise HTTPException(status_code=503, detail="RLHF feedback not available on this system")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.post("/v1/feedback/thumbs", response_model=None)
async def submit_thumbs_feedback(request: Request) -> Dict[str, Any]:
    """
    Submit thumbs up/down feedback (simplified endpoint for UI buttons).
    
    This is a simplified endpoint for submitting binary feedback (thumbs up/down)
    on AI responses, typically triggered by UI buttons. It provides a quick way
    for users to indicate satisfaction without detailed feedback.
    
    Args:
        request: FastAPI Request object containing JSON body with:
            - query_id: ID of the original query
            - response_id: ID of the AI response being rated
            - is_positive: Boolean indicating thumbs up (True) or down (False)
        
    Returns:
        Dict containing:
            - status: "accepted" on success
            - feedback_type: "thumbs_up" or "thumbs_down"
            - message: Human-readable confirmation message
        
    Raises:
        HTTPException: 400 if request body is invalid
        HTTPException: 503 if deployment or learning system not initialized
        HTTPException: 500 if feedback submission fails
    """
    # Parse and validate request body
    try:
        from vulcan.api.models import ThumbsFeedbackRequest
        body = await request.json()
        thumbs_data = ThumbsFeedbackRequest(**body)
    except ValueError as e:
        logger.warning(f"Invalid thumbs feedback request body: {e}")
        raise HTTPException(status_code=400, detail="Invalid request body format")
    except Exception as e:
        logger.warning(f"Failed to parse thumbs feedback request: {type(e).__name__}")
        raise HTTPException(status_code=400, detail="Invalid request body")
    
    # Get deployment with consistent fallback behavior
    deployment = require_deployment(request)

    try:
        # Access learning system via deps.continual
        learning_system = None
        if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
            learning_system = deployment.collective.deps.continual
        
        if learning_system is None:
            raise HTTPException(status_code=503, detail="Learning system not available")

        # Submit thumbs feedback if the learning system supports it
        if hasattr(learning_system, "submit_thumbs_feedback"):
            learning_system.submit_thumbs_feedback(
                query_id=str(thumbs_data.query_id),
                response_id=str(thumbs_data.response_id),
                is_positive=bool(thumbs_data.is_positive),
            )
            
            # ALIGNMENT FIX: Also propagate feedback to World Model and self-improvement drive
            # This ensures feedback impacts CSIU (Continuous Self-Improvement Unit) and meta-reasoning
            world_model = None
            if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "world_model"):
                world_model = deployment.collective.deps.world_model
            
            if world_model and hasattr(world_model, "self_improvement_drive"):
                try:
                    # Notify self-improvement drive of user feedback for alignment
                    reward_signal = 1.0 if thumbs_data.is_positive else -1.0
                    feedback_context = {
                        "query_id": str(thumbs_data.query_id),
                        "response_id": str(thumbs_data.response_id),
                        "feedback_source": "ui_thumbs",
                        "reward_signal": reward_signal,
                        "is_positive": bool(thumbs_data.is_positive)
                    }
                    
                    # Record outcome for self-improvement drive
                    world_model.self_improvement_drive.record_outcome(
                        success=(thumbs_data.is_positive),
                        metrics={"user_satisfaction": reward_signal},
                        context=feedback_context
                    )
                    logger.info(f"[ALIGNMENT] Thumbs feedback propagated to self-improvement drive: {reward_signal}")
                except Exception as e:
                    # Log but don't fail the request if self-improvement integration fails
                    logger.warning(f"[ALIGNMENT] Failed to propagate feedback to self-improvement drive: {e}")
            
            feedback_type = "thumbs_up" if thumbs_data.is_positive else "thumbs_down"
            return {
                "status": "accepted",
                "feedback_type": feedback_type,
                "message": f"Thumbs {'up' if thumbs_data.is_positive else 'down'} recorded"
            }

        raise HTTPException(status_code=503, detail="Thumbs feedback not available on this system")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbs feedback failed: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit thumbs feedback")


@router.get("/v1/feedback/stats", response_model=None)
async def get_feedback_stats(request: Request) -> Dict[str, Any]:
    """
    Get RLHF feedback statistics.
    
    Returns current feedback stats including total feedback received,
    positive/negative counts, and RLHF learning status. This endpoint
    is useful for monitoring the feedback collection process and
    understanding user satisfaction trends.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Dict containing:
            - status: "active", "unavailable", or error status
            - message: Human-readable status message (if unavailable)
            - total_feedback: Total count of feedback items received
            - positive_feedback: Count of positive feedback items
            - negative_feedback: Count of negative feedback items
            - learning_enabled: Boolean indicating if RLHF learning is active
        
    Raises:
        HTTPException: 500 if stats retrieval fails unexpectedly
        
    Note:
        Returns graceful degradation response (status="unavailable") if
        deployment or learning system is not initialized, rather than
        raising an exception. This allows monitoring dashboards to
        display status even during initialization.
    """
    # Get deployment with consistent fallback behavior
    deployment = get_deployment(request)
    
    if deployment is None:
        return {
            "status": "unavailable",
            "message": "Deployment not initialized",
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
        }

    try:
        # Access learning system via deps.continual
        learning_system = None
        if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
            learning_system = deployment.collective.deps.continual
        
        if learning_system is None:
            return {
                "status": "unavailable",
                "message": "Learning system not available",
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
            }

        # Build stats response with safe defaults
        stats: Dict[str, Any] = {
            "status": "active",
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "learning_enabled": True,
        }
        
        # Get stats from learning system if available
        if hasattr(learning_system, "get_feedback_stats"):
            retrieved_stats = learning_system.get_feedback_stats()
            if isinstance(retrieved_stats, dict):
                # Safely update with type coercion
                stats["total_feedback"] = int(retrieved_stats.get("total_feedback", 0))
                stats["positive_feedback"] = int(retrieved_stats.get("positive_feedback", 0))
                stats["negative_feedback"] = int(retrieved_stats.get("negative_feedback", 0))
                if "learning_enabled" in retrieved_stats:
                    stats["learning_enabled"] = bool(retrieved_stats["learning_enabled"])
        
        return stats

    except Exception as e:
        logger.error(f"Failed to get feedback stats: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback statistics")
