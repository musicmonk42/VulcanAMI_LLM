"""
Feedback Endpoints

RLHF (Reinforcement Learning from Human Feedback) endpoints for collecting
user feedback on AI responses to improve future performance.
"""

import logging
import secrets
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/v1/feedback")
async def submit_feedback(request: Request, app):
    """
    Submit human feedback for RLHF learning.
    
    This endpoint accepts feedback on AI responses to improve future performance
    through Reinforcement Learning from Human Feedback (RLHF).
    
    Args:
        request: FeedbackRequest with feedback details
        app: FastAPI app instance for accessing state
        
    Returns:
        Dict containing feedback_id and acceptance status
        
    Raises:
        HTTPException: If system not initialized or feedback submission fails
        
    Example:
        >>> response = await submit_feedback(feedback_request, app)
        >>> print(response["feedback_id"])
    """
    deployment = require_deployment(request)

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
            human_preference = request.context.get("preferred_response") if request.context else None
            
            feedback = FeedbackData(
                # SECURITY FIX: Use full cryptographic randomness instead of predictable time prefix
                # Old: f"fb_{int(time.time())}_{secrets.token_hex(4)}"
                # This prevents timing attacks and ID enumeration
                feedback_id=f"fb_{secrets.token_urlsafe(16)}",
                timestamp=time.time(),
                feedback_type=request.feedback_type,
                content=request.content,
                context=request.context or {},
                agent_response=request.response_id,
                human_preference=human_preference,
                reward_signal=float(request.reward_signal),
                metadata={
                    "query_id": request.query_id,
                    "response_id": request.response_id,
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
async def submit_thumbs_feedback(request, app):
    """
    Submit thumbs up/down feedback (simplified endpoint for UI buttons).
    
    This is a simplified endpoint for submitting binary feedback (thumbs up/down)
    on AI responses, typically triggered by UI buttons.
    
    Args:
        request: ThumbsFeedbackRequest with query_id, response_id, is_positive
        app: FastAPI app instance for accessing state
        
    Returns:
        Dict containing feedback acceptance status
        
    Raises:
        HTTPException: If system not initialized or feedback submission fails
        
    Example:
        >>> response = await submit_thumbs_feedback(thumbs_request, app)
        >>> print(response["feedback_type"])  # "thumbs_up" or "thumbs_down"
    """
    if not hasattr(app.state, "deployment"):
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        deployment = app.state.deployment
        
        # FIX MAJOR-4: Use deps.continual instead of deps.learning
        learning_system = None
        if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
            learning_system = deployment.collective.deps.continual
        
        if learning_system is None:
            raise HTTPException(status_code=503, detail="Learning system not available")

        # FIX MAJOR-4: learning_system IS the ContinualLearner, check its methods directly
        if hasattr(learning_system, "submit_thumbs_feedback"):
            learning_system.submit_thumbs_feedback(
                query_id=request.query_id,
                response_id=request.response_id,
                is_positive=request.is_positive,
            )
            
            return {
                "status": "accepted",
                "feedback_type": "thumbs_up" if request.is_positive else "thumbs_down",
                "message": f"Thumbs {'up' if request.is_positive else 'down'} recorded"
            }

        raise HTTPException(status_code=503, detail="Thumbs feedback not available on this system")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbs feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/feedback/stats")
async def get_feedback_stats(app):
    """
    Get RLHF feedback statistics.
    
    Returns current feedback stats including total feedback received,
    positive/negative counts, and RLHF learning status.
    
    Args:
        app: FastAPI app instance for accessing state
        
    Returns:
        Dict containing feedback statistics
        
    Raises:
        HTTPException: If system not initialized or stats retrieval fails
        
    Example:
        >>> stats = await get_feedback_stats(app)
        >>> print(stats["total_feedback"])
    """
    if not hasattr(app.state, "deployment"):
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        deployment = app.state.deployment
        
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
