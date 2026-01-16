"""
Planning Endpoint

This module provides the endpoint for creating execution plans using
VULCAN's ProblemDecomposer system.

The endpoint integrates the fully-featured ProblemDecomposer which provides:
- 6 decomposition strategies (Exact, Semantic, Structural, Analogical, SyntheticBridging, BruteForce)
- Predictive strategy selection
- Caching and learning integration
- Safety validation

Endpoints:
    POST /v1/plan - Create execution plan with validation
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment
from vulcan.metrics import error_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["planning"])


@router.post("/v1/plan", response_model=None)
async def create_plan(request: Request) -> dict:
    """
    Create execution plan using ProblemDecomposer.
    
    This endpoint uses VULCAN's advanced ProblemDecomposer to break down
    complex goals into actionable steps with confidence scoring and
    strategy selection.
    
    The ProblemDecomposer provides:
    - Multiple decomposition strategies (Exact, Semantic, Structural, Analogical, etc.)
    - Intelligent strategy selection based on problem characteristics
    - Confidence scoring and validation
    - Safety checks and compliance
    - Learning from previous decompositions
    
    Args:
        request: FastAPI request with PlanRequest body containing:
            - goal: High-level goal to plan for (string, required)
            - context: Optional context dictionary for planning (dict)
            - method: Planning method, currently only "hierarchical" supported (string)
    
    Returns:
        Dict containing:
            - plan: The generated decomposition plan with steps and metadata
            - status: "created" on success
            - strategy_used: The decomposition strategy that was selected (str)
            - confidence: Confidence score for the plan (float, 0.0-1.0)
            - steps_count: Number of steps in the decomposition (int)
    
    Raises:
        HTTPException: 400 if request validation fails
        HTTPException: 503 if planner not available or encounters error
    
    Example Request:
        {
            "goal": "Build a machine learning pipeline",
            "context": {"domain": "data_science", "constraints": ["budget"]},
            "method": "hierarchical"
        }
    
    Example Response:
        {
            "plan": {...},
            "status": "created",
            "strategy_used": "SemanticDecomposition",
            "confidence": 0.85,
            "steps_count": 7
        }
    
    Note:
        Falls back to legacy goal_system if ProblemDecomposer is unavailable,
        ensuring graceful degradation.
    """
    deployment = require_deployment(request)
    
    try:
        # Parse and validate request body
        from vulcan.api.models import PlanRequest
        try:
            body = await request.json()
            plan_request = PlanRequest(**body)
        except Exception as e:
            logger.error(f"Invalid request body: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request format: {str(e)}"
            )
        
        # Try to get the ProblemDecomposer singleton
        from vulcan.reasoning.singletons import get_problem_decomposer
        decomposer = get_problem_decomposer()
        
        if decomposer is not None:
            # Use the real ProblemDecomposer
            logger.info(f"Using ProblemDecomposer for goal: {plan_request.goal[:50]}...")
            
            # Convert goal to ProblemGraph format
            try:
                from vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
                
                # Safely handle context which may be None
                context = plan_request.context if plan_request.context is not None else {}
                
                problem = ProblemGraph(
                    nodes={
                        'goal': {
                            'description': plan_request.goal,
                            'type': 'objective'
                        }
                    },
                    edges=[],
                    metadata={
                        'context': context,
                        'domain': context.get('domain', 'general'),
                        'goal_text': plan_request.goal,
                        'method': plan_request.method
                    }
                )
            except ImportError as e:
                logger.error(f"Failed to import ProblemGraph: {e}")
                # Fall through to legacy system
                decomposer = None
            except Exception as e:
                logger.error(f"Failed to create ProblemGraph: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process goal: {str(e)}"
                )
            
            if decomposer is not None:
                # Execute decomposition in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                try:
                    plan = await loop.run_in_executor(
                        None,
                        decomposer.decompose_novel_problem,
                        problem
                    )
                    
                    # Extract metadata from the decomposition result
                    strategy_used = getattr(plan, 'strategy_used', 'unknown')
                    confidence = getattr(plan, 'confidence', 0.0)
                    steps = getattr(plan, 'steps', [])
                    steps_count = len(steps)
                    
                    # Convert plan to serializable format
                    if hasattr(plan, 'to_dict'):
                        plan_data = plan.to_dict()
                    elif isinstance(plan, dict):
                        plan_data = plan
                    else:
                        plan_data = {'description': str(plan)}
                    
                    logger.info(
                        f"Decomposition complete: strategy={strategy_used}, "
                        f"confidence={confidence:.2f}, steps={steps_count}"
                    )
                    
                    return {
                        "plan": plan_data,
                        "status": "created",
                        "strategy_used": strategy_used,
                        "confidence": confidence,
                        "steps_count": steps_count
                    }
                    
                except Exception as e:
                    logger.error(f"Decomposition failed: {e}", exc_info=True)
                    # Fall through to legacy system
                    decomposer = None
        
        # Fallback to legacy goal_system if decomposer unavailable
        if decomposer is None:
            logger.warning(
                "ProblemDecomposer not available, falling back to legacy goal_system"
            )
            planner = deployment.collective.deps.goal_system
            if planner is None:
                raise HTTPException(
                    status_code=503,
                    detail="Planning service not available"
                )
            
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
                        status_code=503,
                        detail=f"Planning service error: {str(e)}"
                    )
            
            # Format legacy response
            plan_data = plan.to_dict() if hasattr(plan, "to_dict") else (
                plan if isinstance(plan, dict) else {"description": str(plan)}
            )
            
            return {
                "plan": plan_data,
                "status": "created",
                "strategy_used": "legacy_goal_system",
                "confidence": 0.0,
                "steps_count": len(plan_data.get("actions", []))
            }
    
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="planning").inc()
        logger.error(f"Planning failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Planning service error: {str(e)}"
        )
