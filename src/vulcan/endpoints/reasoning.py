"""
Reasoning Endpoints

This module provides endpoints for direct reasoning and explanation generation
using VULCAN's unified reasoning bridge and world model.

Endpoints:
    POST /llm/reason  - LLM-enhanced reasoning using unified reasoning bridge
    POST /llm/explain - Natural language explanations using world model bridge
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reasoning"])


@router.post("/llm/reason")
async def reason(request: Request) -> dict:
    """
    LLM-enhanced reasoning using VULCAN's unified reasoning bridge.
    
    Performs sophisticated reasoning on queries using VULCAN's hybrid approach
    that combines symbolic reasoning, probabilistic inference, and LLM capabilities.
    The unified reasoning bridge orchestrates multiple reasoning engines.
    
    Args:
        request: FastAPI request with ReasonRequest body containing:
            - query: Query or problem to reason about
            - context: Optional context dict providing background information
    
    Returns:
        Dict containing:
            - reasoning: The reasoning result from the unified bridge
    
    Raises:
        HTTPException: 503 if LLM not initialized
        HTTPException: 500 if reasoning fails
    
    Note:
        This endpoint uses the "hybrid" reasoning mode which combines:
        - Symbolic reasoning for formal logic
        - Probabilistic reasoning for uncertainty
        - LLM reasoning for natural language understanding
        - Causal reasoning for cause-effect analysis
    
    Example:
        Request:
        {
            "query": "If all birds can fly, and penguins are birds, can penguins fly?",
            "context": {"domain": "logic_puzzle"}
        }
        
        Response:
        {
            "reasoning": {
                "conclusion": "No, this is a false syllogism...",
                "reasoning_type": "symbolic",
                "confidence": 0.95,
                "explanation": "The premise 'all birds can fly' is false..."
            }
        }
    """
    app = request.app
    
    if not hasattr(app.state, "llm"):
        raise HTTPException(status_code=503, detail="LLM not initialized")

    llm = app.state.llm

    try:
        # Get request body
        from vulcan.api.models import ReasonRequest
        body = await request.json()
        reason_request = ReasonRequest(**body)
        
        loop = asyncio.get_running_loop()
        # Use VULCAN's unified reasoning with LLM
        result = await loop.run_in_executor(
            None,
            llm.bridge.reasoning.reason,
            reason_request.query,
            reason_request.context,
            "hybrid"
        )
        return {"reasoning": result}
    except Exception as e:
        logger.error(f"LLM reasoning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/explain")
async def explain(request: Request) -> dict:
    """
    Natural language explanations using the LLM's world model bridge.
    
    Generates clear, natural language explanations for concepts using VULCAN's
    world model. The world model provides context-aware explanations that draw
    on learned knowledge and causal understanding.
    
    Args:
        request: FastAPI request with ExplainRequest body containing:
            - concept: Concept or phenomenon to explain
            - context: Optional context dict for tailoring the explanation
    
    Returns:
        Dict containing:
            - explanation: Natural language explanation of the concept
    
    Raises:
        HTTPException: 503 if LLM not initialized
        HTTPException: 500 if explanation generation fails
    
    Note:
        The world model bridge provides explanations that:
        - Use appropriate level of detail for the context
        - Include relevant examples and analogies
        - Connect to related concepts
        - Incorporate causal relationships
    
    Example:
        Request:
        {
            "concept": "photosynthesis",
            "context": {"audience": "middle_school", "detail_level": "moderate"}
        }
        
        Response:
        {
            "explanation": "Photosynthesis is how plants make their own food using sunlight..."
        }
    """
    app = request.app
    
    if not hasattr(app.state, "llm"):
        raise HTTPException(status_code=503, detail="LLM not initialized")

    llm = app.state.llm

    try:
        # Get request body
        from vulcan.api.models import ExplainRequest
        body = await request.json()
        explain_request = ExplainRequest(**body)
        
        loop = asyncio.get_running_loop()
        # Use the LLM's world model bridge for explanation
        explanation = await loop.run_in_executor(
            None,
            llm.bridge.world_model.explain,
            explain_request.concept,
            explain_request.context
        )
        return {"explanation": explanation}
    except Exception as e:
        logger.error(f"LLM explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
