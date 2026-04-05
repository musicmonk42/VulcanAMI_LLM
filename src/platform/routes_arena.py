"""
Arena API route handlers extracted from full_platform.py.

Provides:
- get_arena_instance (lazy singleton factory)
- arena_run_agent (POST /api/arena/run/{agent_id})
- arena_feedback (POST /api/arena/feedback)
- arena_tournament (POST /api/arena/tournament)
- arena_feedback_dispatch (POST /api/arena/feedback_dispatch)
"""

import json
import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from src.platform.auth import verify_authentication

router = APIRouter()
logger = logging.getLogger(__name__)

_arena_instance_cache = None
_arena_instance_initialized = False


def get_arena_instance():
    """
    Get or create Arena instance for proxy endpoints.
    """
    global _arena_instance_cache, _arena_instance_initialized

    if _arena_instance_cache is not None:
        return _arena_instance_cache

    if _arena_instance_initialized:
        return None

    _arena_instance_initialized = True

    try:
        from src.graphix_arena import _ARENA_INSTANCE, GraphixArena, register_routes

        if _ARENA_INSTANCE is not None:
            logger.info("Using existing Arena instance from src.graphix_arena")
            _arena_instance_cache = _ARENA_INSTANCE
            return _arena_instance_cache

        logger.info("Creating new Arena instance for platform proxy endpoints")
        arena = GraphixArena()

        try:
            register_routes(arena)
        except Exception as e:
            logger.warning(f"Could not register Arena routes: {e}")

        _arena_instance_cache = arena
        logger.info("Arena instance initialized successfully")
        return _arena_instance_cache

    except ImportError as e:
        logger.error(f"Could not import Arena components: {e}")
        logger.error("   Arena API endpoints will return 503 Service Unavailable")
        return None
    except Exception as e:
        logger.error(f"Could not initialize Arena instance: {e}")
        return None


@router.post("/api/arena/run/{agent_id}", response_model=None)
async def arena_run_agent(
    agent_id: str, request: Request, auth: Dict = Depends(verify_authentication)
):
    """Run agent task via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        result = await arena.run_agent_task(request)
        return result
    except Exception as e:
        logger.error(f"Arena agent task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/arena/feedback", response_model=None)
async def arena_feedback(request: Request, auth: Dict = Depends(verify_authentication)):
    """Submit feedback via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        result = await arena.feedback_ingestion(request)
        return result
    except Exception as e:
        logger.error(f"Arena feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/arena/tournament", response_model=None)
async def arena_tournament(
    request: Request, auth: Dict = Depends(verify_authentication)
):
    """Run tournament via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        result = await arena.run_tournament_task(request)
        return result
    except Exception as e:
        logger.error(f"Arena tournament failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/arena/feedback_dispatch", response_model=None)
async def arena_feedback_dispatch(
    request: Request, auth: Dict = Depends(verify_authentication)
):
    """Dispatch feedback protocol via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        from src.graphix_arena import MAX_PAYLOAD_SIZE, dispatch_feedback_protocol

        data = await request.json()

        payload_size = len(json.dumps(data))
        if payload_size > MAX_PAYLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large: {payload_size} > {MAX_PAYLOAD_SIZE}",
            )

        context = {"audit_log": []}
        result = dispatch_feedback_protocol(data, context)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Arena feedback dispatch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
