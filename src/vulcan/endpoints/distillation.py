"""
Distillation Endpoints

Provides endpoints for knowledge distillation management including buffer status,
flush triggers, buffer clearing, and configuration updates.

This module implements the distillation API that allows monitoring and controlling
the knowledge distillation process from teacher to student models.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/distillation")


def _get_distiller(request: Optional[Request] = None):
    """
    Get the knowledge distiller instance from app state or global singleton.
    
    Args:
        request: Optional FastAPI request for accessing app state
        
    Returns:
        OpenAIKnowledgeDistiller instance or None
    """
    # Try app state first (set by startup manager)
    if request is not None:
        distiller = getattr(request.app.state, "knowledge_distiller", None)
        if distiller is not None:
            return distiller
    
    # Fall back to global singleton
    try:
        from vulcan.distillation import get_knowledge_distiller
        return get_knowledge_distiller()
    except ImportError:
        return None


class DistillationStatusResponse(BaseModel):
    """Response model for distillation status."""
    buffer_size: int = Field(description="Number of examples in distillation buffer")
    model_info: Dict[str, Any] = Field(description="Teacher and student model information")
    last_flush: Optional[str] = Field(description="Timestamp of last buffer flush")
    config: Dict[str, Any] = Field(description="Current distillation configuration")


class DistillationFlushRequest(BaseModel):
    """Request model for triggering distillation flush."""
    force: bool = Field(default=False, description="Force flush even if buffer not full")


class DistillationConfigUpdateRequest(BaseModel):
    """Request model for updating distillation configuration."""
    buffer_size: Optional[int] = Field(None, description="Maximum buffer size")
    flush_threshold: Optional[float] = Field(None, description="Buffer fill threshold for auto-flush")
    temperature: Optional[float] = Field(None, description="Distillation temperature")


@router.get("/status", response_model=DistillationStatusResponse)
async def get_distillation_status(request: Request):
    """
    Get current distillation buffer status and configuration.
    
    Returns information about the distillation buffer, models, and configuration.
    
    Returns:
        DistillationStatusResponse: Current distillation status
        
    Raises:
        HTTPException: If distillation system not available
        
    Example:
        ```python
        response = await client.get("/v1/distillation/status")
        print(f"Buffer has {response.buffer_size} examples")
        ```
    """
    try:
        distiller = _get_distiller(request)
        
        if distiller is None:
            # Return sensible defaults when distiller is not available
            return DistillationStatusResponse(
                buffer_size=0,
                model_info={
                    "teacher": "not_configured",
                    "student": "not_configured",
                    "status": "distillation_disabled"
                },
                last_flush=None,
                config={
                    "buffer_size": 0,
                    "flush_threshold": 0.0,
                    "temperature": 0.0,
                    "enabled": False
                }
            )
        
        # Get real status from the distiller
        status = distiller.get_status()
        
        # Extract buffer size from state
        buffer_size = status.get("state", {}).get("buffer_size", 0)
        
        # Build model info
        model_info = {
            "teacher": "gpt-4",  # Default teacher model
            "student": "vulcan-local-llm",
            "status": "active" if status.get("enabled", False) else "inactive"
        }
        
        # Get stats for last flush time
        stats = status.get("stats", {})
        last_flush_time = stats.get("last_training_trigger_time")
        last_flush = None
        if last_flush_time:
            from datetime import datetime
            last_flush = datetime.fromtimestamp(last_flush_time).isoformat() + "Z"
        
        # Build config from distiller config
        config_data = status.get("config", {})
        config = {
            "buffer_size": config_data.get("max_buffer_size", 1000),
            "flush_threshold": config_data.get("flush_threshold", 0.8) if "flush_threshold" in config_data else 0.8,
            "temperature": 2.0,  # Default distillation temperature
            "require_opt_in": config_data.get("require_opt_in", True),
            "pii_redaction": config_data.get("pii_redaction", True),
            "governance_check": config_data.get("governance_check", True),
            "retention_days": config_data.get("retention_days", 30),
            "enabled": status.get("enabled", False)
        }
        
        return DistillationStatusResponse(
            buffer_size=buffer_size,
            model_info=model_info,
            last_flush=last_flush,
            config=config
        )
    except Exception as e:
        logger.error(f"Error getting distillation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get distillation status: {str(e)}"
        )


@router.post("/flush", response_model=None)
async def trigger_distillation_flush(request: Request, flush_request: DistillationFlushRequest):
    """
    Trigger knowledge distillation flush.
    
    Forces the distillation system to train the student model on the current buffer.
    
    Args:
        request: FastAPI request for accessing app state
        flush_request: Flush request with force flag
        
    Returns:
        Dict with flush status and metrics
        
    Raises:
        HTTPException: If flush fails
        
    Example:
        ```python
        response = await client.post("/v1/distillation/flush", json={"force": True})
        print(f"Flushed {response['examples_processed']} examples")
        ```
    """
    try:
        logger.info(f"Distillation flush triggered (force={flush_request.force})")
        
        distiller = _get_distiller(request)
        
        if distiller is None:
            return {
                "status": "unavailable",
                "message": "Distillation system not initialized",
                "examples_processed": 0,
                "training_time": 0.0,
                "metrics": {}
            }
        
        # Flush the buffer to storage
        examples_flushed = distiller.flush()
        
        # Optionally trigger training if forced
        trigger_result = None
        if flush_request.force:
            trigger_result = distiller.trigger_training(reason="api_flush", force=True)
        
        return {
            "status": "success",
            "examples_processed": examples_flushed,
            "training_time": 0.0,  # Training is async, no immediate time available
            "metrics": {
                "examples_flushed": examples_flushed,
                "training_triggered": trigger_result.get("triggered", False) if trigger_result else False,
            },
            "training_result": trigger_result
        }
    except Exception as e:
        logger.error(f"Error triggering distillation flush: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger distillation flush: {str(e)}"
        )


@router.post("/clear", response_model=None)
async def clear_distillation_buffer(request: Request):
    """
    Clear the distillation buffer without training.
    
    Removes all examples from the buffer without performing distillation.
    Useful for resetting the buffer or removing low-quality examples.
    
    Returns:
        Dict with clear status
        
    Raises:
        HTTPException: If clear operation fails
        
    Example:
        ```python
        response = await client.post("/v1/distillation/clear")
        print(f"Cleared {response['examples_cleared']} examples")
        ```
    """
    try:
        distiller = _get_distiller(request)
        
        if distiller is None:
            return {
                "status": "unavailable",
                "message": "Distillation system not initialized",
                "examples_cleared": 0
            }
        
        # Clear the buffer
        examples_cleared = distiller.clear_buffer()
        
        logger.info(f"Distillation buffer cleared: {examples_cleared} examples removed")
        
        return {
            "status": "success",
            "examples_cleared": examples_cleared
        }
    except Exception as e:
        logger.error(f"Error clearing distillation buffer: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear distillation buffer: {str(e)}"
        )


@router.post("/config", response_model=None)
async def update_distillation_config(request: Request, config_request: DistillationConfigUpdateRequest):
    """
    Update distillation configuration.
    
    Allows runtime updates to distillation parameters without restart.
    
    Args:
        request: FastAPI request for accessing app state
        config_request: Configuration update request
        
    Returns:
        Dict with updated configuration
        
    Raises:
        HTTPException: If configuration update fails
        
    Example:
        ```python
        response = await client.post(
            "/v1/distillation/config",
            json={"buffer_size": 2000, "temperature": 1.5}
        )
        print(f"Updated config: {response['config']}")
        ```
    """
    try:
        updates = {}
        if config_request.buffer_size is not None:
            updates["buffer_size"] = config_request.buffer_size
        if config_request.flush_threshold is not None:
            updates["flush_threshold"] = config_request.flush_threshold
        if config_request.temperature is not None:
            updates["temperature"] = config_request.temperature
            
        logger.info(f"Distillation config update requested: {updates}")
        
        distiller = _get_distiller(request)
        
        if distiller is None:
            return {
                "status": "unavailable",
                "message": "Distillation system not initialized",
                "config": updates
            }
        
        # Apply configuration updates
        if config_request.buffer_size is not None:
            distiller.max_buffer_size = config_request.buffer_size
        
        # Get current status to return updated config
        status = distiller.get_status()
        config_data = status.get("config", {})
        
        return {
            "status": "success",
            "config": {
                "buffer_size": distiller.max_buffer_size,
                "flush_threshold": config_request.flush_threshold or config_data.get("flush_threshold", 0.8),
                "temperature": config_request.temperature or 2.0,
                "require_opt_in": config_data.get("require_opt_in", True),
                "pii_redaction": config_data.get("pii_redaction", True),
            }
        }
    except Exception as e:
        logger.error(f"Error updating distillation config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update distillation config: {str(e)}"
        )
