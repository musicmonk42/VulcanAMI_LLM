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

These endpoints support two calling patterns:
1. Via FastAPI router (Request object injected by FastAPI)
2. Direct calls from proxy functions (e.g., full_platform.py) with optional Request
"""

import logging
import secrets
import time
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment, get_deployment

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/v1/feedback")
async def submit_feedback(request: Request) -> Dict[str, Any]:
    """
    Submit human feedback for RLHF learning.
    
    This endpoint accepts feedback on AI responses to improve future performance
    through Reinforcement Learning from Human Feedback (RLHF).
    
    Args:
        request: FastAPI request object containing FeedbackRequest body with:
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
        HTTPException: 503 if deployment or learning system not initialized
        HTTPException: 500 if feedback submission fails
    """
    # Handle both Request object and direct model call patterns
    actual_request: Optional[Request] = None
    feedback_data = None
    
    if isinstance(request, Request):
        actual_request = request
        # Parse body from Request
        try:
            from vulcan.api.models import FeedbackRequest
            body = await request.json()
            feedback_data = FeedbackRequest(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
    else:
        # Direct call with model object (from proxy functions)
        feedback_data = request
    
    deployment = require_deployment(actual_request)

    try:
        
        # FIX MAJOR-4: Use deps.continual instead of deps.learning
        learning_system = None
        if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
            learning_system = deployment.collective.deps.continual
        
        if learning_system is None:
            raise HTTPException(status_code=503, detail="Learning system not available")

        # FIX MAJOR-4: learning_system IS the ContinualLearner, check its methods directly
        if hasattr(learning_system, "receive_feedback"):
            learner = learning_system
            from vulcan.learning.learning_types import FeedbackData
            
            # human_preference is the user's preferred response (None if not a preference comparison)
            context = getattr(feedback_data, 'context', None) or {}
            human_preference = context.get("preferred_response") if context else None
            
            feedback = FeedbackData(
                # SECURITY FIX: Use full cryptographic randomness instead of predictable time prefix
                # Old: f"fb_{int(time.time())}_{secrets.token_hex(4)}"
                # This prevents timing attacks and ID enumeration
                feedback_id=f"fb_{secrets.token_urlsafe(16)}",
                timestamp=time.time(),
                feedback_type=feedback_data.feedback_type,
                content=feedback_data.content,
                context=context,
                agent_response=feedback_data.response_id,
                human_preference=human_preference,
                reward_signal=float(feedback_data.reward_signal),
                metadata={
                    "query_id": feedback_data.query_id,
                    "response_id": feedback_data.response_id,
                    "source": "api",
                },
            )
            
            learner.receive_feedback(feedback)
            
            return {
                "status": "accepted",
                "feedback_id": feedback.feedback_id,
                "message": "Feedback submitted successfully"
            }

        raise HTTPException(status_code=503, detail="RLHF feedback not available on this system")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/feedback/thumbs")
async def submit_thumbs_feedback(request: Union[Request, Any] = None) -> Dict[str, Any]:
    """
    Submit thumbs up/down feedback (simplified endpoint for UI buttons).
    
    This is a simplified endpoint for submitting binary feedback (thumbs up/down)
    on AI responses, typically triggered by UI buttons. It provides a quick way
    for users to indicate satisfaction without detailed feedback.
    
    Args:
        request: FastAPI request object or ThumbsFeedbackRequest model with:
            - query_id: ID of the original query
            - response_id: ID of the AI response being rated
            - is_positive: Boolean indicating thumbs up (True) or down (False)
        
    Returns:
        Dict containing:
            - status: "accepted" on success
            - feedback_type: "thumbs_up" or "thumbs_down"
            - message: Human-readable confirmation message
        
    Raises:
        HTTPException: 503 if deployment or learning system not initialized
        HTTPException: 500 if feedback submission fails
    """
    # Handle both Request object and direct model call patterns
    actual_request: Optional[Request] = None
    thumbs_data = None
    
    if isinstance(request, Request):
        actual_request = request
        # Parse body from Request
        try:
            from vulcan.api.models import ThumbsFeedbackRequest
            body = await request.json()
            thumbs_data = ThumbsFeedbackRequest(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
    else:
        # Direct call with model object (from proxy functions)
        thumbs_data = request
    
    # Use require_deployment for consistent fallback behavior with sub-app mounting
    deployment = require_deployment(actual_request)

    try:
        
        # FIX MAJOR-4: Use deps.continual instead of deps.learning
        learning_system = None
        if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
            learning_system = deployment.collective.deps.continual
        
        if learning_system is None:
            raise HTTPException(status_code=503, detail="Learning system not available")

        # FIX MAJOR-4: learning_system IS the ContinualLearner, check its methods directly
        if hasattr(learning_system, "submit_thumbs_feedback"):
            learning_system.submit_thumbs_feedback(
                query_id=thumbs_data.query_id,
                response_id=thumbs_data.response_id,
                is_positive=thumbs_data.is_positive,
            )
            
            return {
                "status": "accepted",
                "feedback_type": "thumbs_up" if thumbs_data.is_positive else "thumbs_down",
                "message": f"Thumbs {'up' if thumbs_data.is_positive else 'down'} recorded"
            }

        raise HTTPException(status_code=503, detail="Thumbs feedback not available on this system")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbs feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/feedback/stats")
async def get_feedback_stats(request: Optional[Request] = None) -> Dict[str, Any]:
    """
    Get RLHF feedback statistics.
    
    Returns current feedback stats including total feedback received,
    positive/negative counts, and RLHF learning status. This endpoint
    is useful for monitoring the feedback collection process and
    understanding user satisfaction trends.
    
    Args:
        request: Optional FastAPI request object. Can be None when called
                 directly from proxy functions in full_platform.py.
        
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
    # Use get_deployment for consistent fallback behavior with sub-app mounting
    # request can be None when called directly from proxy functions
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
        
        # FIX MAJOR-4: Use deps.continual instead of deps.learning
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

        # Get stats from learning system
        stats = {
            "status": "active",
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "learning_enabled": True,
        }
        
        if hasattr(learning_system, "get_feedback_stats"):
            stats.update(learning_system.get_feedback_stats())
        
        return stats

    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
