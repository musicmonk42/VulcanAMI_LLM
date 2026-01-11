"""
Planning Endpoint

This module provides the endpoint for creating execution plans using
VULCAN's goal system and hierarchical planner.

Endpoints:
    POST /v1/plan - Create execution plan with validation
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment

logger = logging.getLogger(__name__)

router = APIRouter(tags=["planning"])


@router.post("/v1/plan")
async def create_plan(request: Request) -> dict:
    """
    Create execution plan with validation.
    
    Uses VULCAN's hierarchical goal system to generate a structured
    plan for achieving a high-level goal. The planner breaks down
    complex goals into executable sub-goals with dependencies.
    
    Args:
        request: FastAPI request with PlanRequest body containing:
            - goal: High-level goal to plan for (string)
            - context: Optional context dictionary for planning
    
    Returns:
        Dict containing:
            - plan: The generated plan (dict or string representation)
            - status: "created" on success
    
    Raises:
        HTTPException: 503 if system not initialized or planner unavailable
        HTTPException: 503 if planning service encounters error
    
    Note:
        The planner may use different signatures depending on version.
        This endpoint handles both:
        - generate_plan({"high_level_goal": goal}, context)
        - generate_plan(goal, context)
    """
    app = request.app
    
    deployment = require_deployment(request)

    # Try to get error counter
    error_counter = None
    try:
        from prometheus_client import Counter
        error_counter = Counter("errors_total", "Total errors", ["error_type"])
    except ImportError:
        pass

    try:
        planner = deployment.collective.deps.goal_system
        if planner is None:
            raise HTTPException(status_code=503, detail="Planner not available")

        # Get request body
        from vulcan.api.models import PlanRequest
        body = await request.json()
        plan_request = PlanRequest(**body)

        loop = asyncio.get_running_loop()

        # Try primary signature first
        try:
            plan = await loop.run_in_executor(
                None,
                planner.generate_plan,
                {"high_level_goal": plan_request.goal},
                plan_request.context,
            )
        except TypeError:
            # Fallback to alternative signature
            try:
                plan = await loop.run_in_executor(
                    None,
                    planner.generate_plan,
                    plan_request.goal,
                    plan_request.context
                )
            except Exception as e:
                logger.error(f"Planning failed with alternative signature: {e}")
                raise HTTPException(
                    status_code=503, detail=f"Planning service error: {str(e)}"
                )

        return {
            "plan": plan.to_dict() if hasattr(plan, "to_dict") else str(plan),
            "status": "created",
        }

    except HTTPException:
        raise
    except Exception as e:
        if error_counter:
            error_counter.labels(error_type="planning").inc()
        logger.error(f"Planning failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Planning service error: {str(e)}")
