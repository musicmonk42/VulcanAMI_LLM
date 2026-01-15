"""
Configuration Endpoints

This module provides dynamic configuration management endpoints for LLM execution
and knowledge distillation settings.

Endpoints:
    GET  /v1/llm/config           - Get current LLM execution configuration
    POST /v1/llm/config           - Update LLM execution configuration
    GET  /v1/distillation/status  - Get distillation system status
    POST /v1/distillation/train   - Trigger distillation data flush
    DELETE /v1/distillation/buffer - Clear distillation buffer
    POST /v1/distillation/config  - Update distillation configuration
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["configuration"])


class LLMConfigUpdate(BaseModel):
    """
    Request model for updating LLM configuration.
    
    Attributes:
        execution_mode: LLM execution strategy (local_first, openai_first, parallel, ensemble)
        parallel_timeout: Timeout for parallel execution in seconds (1-120)
        ensemble_min_confidence: Minimum confidence threshold for ensemble selection (0-1)
    """

    execution_mode: Optional[str] = Field(
        None,
        description="LLM execution mode: local_first, openai_first, parallel, ensemble"
    )
    parallel_timeout: Optional[float] = Field(
        None,
        ge=1.0,
        le=120.0,
        description="Timeout for parallel execution (1-120 seconds)"
    )
    ensemble_min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for ensemble selection (0-1)"
    )


class DistillationConfigUpdate(BaseModel):
    """
    Request model for updating distillation capture configuration.
    
    Attributes:
        require_opt_in: Whether to require per-session opt-in for capture
        enable_pii_redaction: Whether to enable PII redaction
        enable_governance_check: Whether to enable governance sensitivity checks
        max_buffer_size: Maximum buffer size before auto-flush (1-10000)
    """

    require_opt_in: Optional[bool] = Field(
        None,
        description="Whether to require per-session opt-in for capture"
    )
    enable_pii_redaction: Optional[bool] = Field(
        None,
        description="Whether to enable PII redaction"
    )
    enable_governance_check: Optional[bool] = Field(
        None,
        description="Whether to enable governance sensitivity checks"
    )
    max_buffer_size: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Maximum buffer size before auto-flush"
    )


@router.get("/v1/llm/config", response_model=None)
async def get_llm_config(request: Request) -> Dict[str, Any]:
    """
    Get current LLM execution configuration.
    
    Returns the hybrid LLM execution settings that control how OpenAI
    and Vulcan's local LLM work together. Includes provider availability
    and supported execution modes.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - execution_mode: Current execution mode
            - parallel_timeout: Timeout for parallel operations
            - ensemble_min_confidence: Ensemble confidence threshold
            - available_modes: List of supported execution modes
            - mode_descriptions: Description of each mode
            - providers: Status of OpenAI and local LLM providers
            - timestamp: Current Unix timestamp
    """
    try:
        from vulcan.llm import get_openai_client, get_openai_init_error
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="LLM module not available"
        )
    
    app = request.app
    settings = getattr(app.state, "settings", None)
    
    if not settings:
        raise HTTPException(
            status_code=500,
            detail="Settings not initialized"
        )
    
    openai_client = get_openai_client()
    openai_init_error = get_openai_init_error()

    return {
        "execution_mode": settings.llm_execution_mode,
        "parallel_timeout": settings.llm_parallel_timeout,
        "ensemble_min_confidence": settings.llm_ensemble_min_confidence,
        "available_modes": ["local_first", "openai_first", "parallel", "ensemble"],
        "mode_descriptions": {
            "local_first": "Try Vulcan's local LLM first, fallback to OpenAI if needed",
            "openai_first": "Try OpenAI first, fallback to local LLM if needed",
            "parallel": "Run both simultaneously, use first successful response",
            "ensemble": "Run both, combine/select best response based on quality",
        },
        "providers": {
            "openai": {
                "available": openai_client is not None,
                "error": openai_init_error if openai_client is None else None,
            },
            "local_llm": {
                "available": hasattr(app.state, "llm") and app.state.llm is not None,
            },
        },
        "timestamp": time.time(),
    }


@router.post("/v1/llm/config", response_model=None)
async def update_llm_config(
    config: LLMConfigUpdate,
    request: Request
) -> Dict[str, Any]:
    """
    Update LLM execution configuration at runtime.
    
    Allows dynamic switching between execution modes without restarting the server.
    Only provided fields will be updated; others remain unchanged.
    
    Args:
        config: LLM configuration update request
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "success" if update completed
            - updated: Dict of fields that were updated
            - current_config: Complete current configuration
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 400 if invalid execution_mode provided
        HTTPException: 500 if settings not initialized
    """
    app = request.app
    settings = getattr(app.state, "settings", None)
    
    if not settings:
        raise HTTPException(
            status_code=500,
            detail="Settings not initialized"
        )
    
    valid_modes = ["local_first", "openai_first", "parallel", "ensemble"]
    updated = {}

    if config.execution_mode is not None:
        mode = config.execution_mode.lower()
        if mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid execution_mode. Must be one of: {valid_modes}",
            )
        settings.llm_execution_mode = mode
        updated["execution_mode"] = mode
        logger.info(f"LLM execution mode updated to: {mode}")

    if config.parallel_timeout is not None:
        settings.llm_parallel_timeout = config.parallel_timeout
        updated["parallel_timeout"] = config.parallel_timeout
        logger.info(f"LLM parallel timeout updated to: {config.parallel_timeout}s")

    if config.ensemble_min_confidence is not None:
        settings.llm_ensemble_min_confidence = config.ensemble_min_confidence
        updated["ensemble_min_confidence"] = config.ensemble_min_confidence
        logger.info(f"LLM ensemble min confidence updated to: {config.ensemble_min_confidence}")

    return {
        "status": "success",
        "updated": updated,
        "current_config": {
            "execution_mode": settings.llm_execution_mode,
            "parallel_timeout": settings.llm_parallel_timeout,
            "ensemble_min_confidence": settings.llm_ensemble_min_confidence,
        },
        "timestamp": time.time(),
    }


@router.get("/v1/distillation/status", response_model=None)
async def get_distillation_status(request: Request) -> Dict[str, Any]:
    """
    Get the current status of the OpenAI Knowledge Distiller.
    
    Returns comprehensive information about the distillation system including:
    - Whether distillation is enabled
    - Number of examples captured
    - Training statistics
    - Buffer size and configuration
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing distillation status, config, and statistics
    """
    try:
        from vulcan.distillation import get_knowledge_distiller
    except ImportError:
        return {
            "enabled": False,
            "message": "Distillation module not available",
        }
    
    app = request.app
    settings = getattr(app.state, "settings", None)
    
    distiller = get_knowledge_distiller()
    if distiller is None:
        return {
            "enabled": False,
            "message": "Knowledge distillation is not enabled",
            "config": {
                "enable_knowledge_distillation": getattr(settings, "enable_knowledge_distillation", False) if settings else False,
            },
        }

    status = distiller.get_status()
    if settings:
        status["config"] = {
            "enable_knowledge_distillation": settings.enable_knowledge_distillation,
            "storage_path": settings.distillation_storage_path,
            "batch_size": settings.distillation_batch_size,
            "training_interval_s": settings.distillation_training_interval_s,
            "learning_rate": settings.distillation_learning_rate,
            "auto_train": settings.distillation_auto_train,
        }
    return status


@router.post("/v1/distillation/train", response_model=None)
async def trigger_distillation_flush(request: Request) -> Dict[str, Any]:
    """
    Flush captured examples to storage for training system consumption.
    
    NOTE: Training is handled by Vulcan's GovernedTrainer and SelfImprovingTraining
    systems, NOT by main.py. This endpoint flushes captured examples to JSONL storage
    where they can be consumed by the training pipeline.
    
    The correct training flow is:
    1. main.py captures OpenAI responses → JSONL storage
    2. GovernedTrainer reads from storage → proposes weight updates
    3. ConsensusEngine approves/rejects updates
    4. SelfImprovingTraining evaluates and promotes/rollbacks
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "flushed"
            - examples_flushed: Number of examples written to storage
            - storage_path: Path where examples were saved
            - note: Reminder about training flow
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if distillation not enabled
    """
    try:
        from vulcan.distillation import get_knowledge_distiller
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Distillation module not available"
        )
    
    distiller = get_knowledge_distiller()
    if distiller is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge distillation is not enabled",
        )

    flushed_count = distiller.flush()
    logger.info(f"Distillation examples flushed: {flushed_count}")
    
    return {
        "status": "flushed",
        "examples_flushed": flushed_count,
        "storage_path": str(distiller.storage_backend.storage_path),
        "note": "Training is handled by GovernedTrainer/SelfImprovingTraining",
        "timestamp": time.time(),
    }


@router.delete("/v1/distillation/buffer", response_model=None)
async def clear_distillation_buffer(request: Request) -> Dict[str, Any]:
    """
    Clear the distillation training buffer without training.
    
    Use this to discard captured examples if needed. This permanently
    removes examples from the buffer without saving them to storage.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "success"
            - examples_cleared: Number of examples removed
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if distillation not enabled
    """
    try:
        from vulcan.distillation import get_knowledge_distiller
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Distillation module not available"
        )
    
    distiller = get_knowledge_distiller()
    if distiller is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge distillation is not enabled",
        )

    count = distiller.clear_buffer()
    logger.info(f"Distillation buffer cleared: {count} examples removed")
    
    return {
        "status": "success",
        "examples_cleared": count,
        "timestamp": time.time(),
    }


@router.post("/v1/distillation/config", response_model=None)
async def update_distillation_config(
    config: DistillationConfigUpdate,
    request: Request
) -> Dict[str, Any]:
    """
    Update knowledge distillation capture configuration at runtime.
    
    NOTE: This updates capture settings only. Training parameters are
    managed by GovernedTrainer and SelfImprovingTraining systems.
    
    Args:
        config: Distillation configuration update request
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "success"
            - updated: Dict of fields that were updated
            - current_config: Complete current configuration
            - note: Reminder about training config location
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if distillation not enabled
    """
    try:
        from vulcan.distillation import get_knowledge_distiller
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Distillation module not available"
        )
    
    distiller = get_knowledge_distiller()
    if distiller is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge distillation is not enabled",
        )

    updated = {}

    if config.require_opt_in is not None:
        distiller.require_opt_in = config.require_opt_in
        updated["require_opt_in"] = config.require_opt_in
        logger.info(f"Distillation require_opt_in updated to: {config.require_opt_in}")

    if config.enable_pii_redaction is not None:
        distiller.enable_pii_redaction = config.enable_pii_redaction
        updated["enable_pii_redaction"] = config.enable_pii_redaction
        logger.info(f"Distillation enable_pii_redaction updated to: {config.enable_pii_redaction}")

    if config.enable_governance_check is not None:
        distiller.enable_governance_check = config.enable_governance_check
        updated["enable_governance_check"] = config.enable_governance_check
        logger.info(f"Distillation enable_governance_check updated to: {config.enable_governance_check}")

    if config.max_buffer_size is not None:
        distiller.max_buffer_size = config.max_buffer_size
        updated["max_buffer_size"] = config.max_buffer_size
        logger.info(f"Distillation max_buffer_size updated to: {config.max_buffer_size}")

    return {
        "status": "success",
        "updated": updated,
        "current_config": distiller.get_status()["config"],
        "note": "Training config managed by GovernedTrainer/SelfImprovingTraining",
        "timestamp": time.time(),
    }
