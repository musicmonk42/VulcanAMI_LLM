"""
FastAPI Application Setup

Creates and configures the FastAPI application with lifespan management.

REFACTORED VERSION:
- Reduced from 780+ lines to ~100 lines
- Delegates to StartupManager for all initialization
- Properly declares all module-level globals
- Fixes ThreadPoolExecutor shutdown leak
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, Any
from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vulcan.server.startup import StartupManager
from vulcan.server import state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager with phased startup.
    
    Delegates all initialization to StartupManager for modular,
    testable, and maintainable startup sequence.
    
    Raises:
        RuntimeError: If critical initialization fails or settings cannot be loaded
    """
    worker_id = os.getpid()
    
    # ============================================================
    # TEST MODE BYPASS
    # ============================================================
    # If deployment exists in app.state, we're in test mode with mocks.
    # Skip all initialization and provide clean exception handling.
    if hasattr(app.state, "deployment") and app.state.deployment is not None:
        logger.info("Test mode detected - using existing mock deployment")
        try:
            yield
        except Exception as e:
            logger.error(f"Test mode error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Test mode shutdown complete")
        return

    # ============================================================
    # SETTINGS INITIALIZATION (P0 Fix: Issue #1)
    # ============================================================
    # Load settings once with proper error handling. Do NOT retry on failure
    # as the same error will likely occur again.
    from vulcan.settings import Settings
    
    try:
        settings = Settings()
    except Exception as e:
        logger.critical(f"Failed to initialize settings: {e}", exc_info=True)
        raise RuntimeError("Startup aborted: Could not initialize Settings.") from e

    # ============================================================
    # REDIS INITIALIZATION (P0 Fix: Issue #4)
    # ============================================================
    # Initialize Redis client from settings BEFORE checking availability.
    # This ensures the client is properly configured before use.
    if settings.redis_url:
        try:
            import redis
            state.redis_client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=False
            )
            # Verify connection with a ping
            state.redis_client.ping()
            logger.info(f"Redis connected: {settings.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            state.redis_client = None
    else:
        logger.info("Redis URL not configured - running in standalone mode")
        state.redis_client = None

    # ============================================================
    # SPLIT-BRAIN PREVENTION (P0 Fix: Issue #5)
    # ============================================================
    # Acquire process lock when Redis is unavailable to ensure only
    # one orchestrator instance runs, preventing split-brain conditions.
    # The lock now includes a heartbeat mechanism for self-healing in
    # distributed systems - stale locks from crashed processes can be
    # detected and safely acquired.
    if state.redis_client is None:
        logger.warning(
            "Redis unavailable - acquiring process lock to prevent split-brain"
        )
        from vulcan.utils_main.process_lock import ProcessLock
        from vulcan.server.startup.constants import (
            PROCESS_LOCK_HEARTBEAT_INTERVAL_SECONDS,
            PROCESS_LOCK_TTL_SECONDS,
        )
        
        state.process_lock = ProcessLock(
            heartbeat_interval=PROCESS_LOCK_HEARTBEAT_INTERVAL_SECONDS,
            lock_ttl=PROCESS_LOCK_TTL_SECONDS,
            enable_heartbeat=True,  # Enable heartbeat for self-healing
        )
        if not state.process_lock.acquire():
            logger.critical(
                "FATAL: Cannot acquire process lock. Another vulcan.orchestrator "
                "instance is already running. Without Redis for state synchronization, "
                "running multiple instances would cause a split-brain condition. "
                "Either start Redis or stop the other instance."
            )
            raise RuntimeError(
                "Split-brain prevention: Another orchestrator instance is running. "
                "Start Redis for multi-instance support or stop the other process."
            )
        logger.info(
            f"Process lock acquired - singleton mode active "
            f"(heartbeat interval: {PROCESS_LOCK_HEARTBEAT_INTERVAL_SECONDS}s, "
            f"TTL: {PROCESS_LOCK_TTL_SECONDS}s)"
        )
    else:
        logger.info("Redis available - multi-instance mode supported")
    
    # ============================================================
    # STARTUP EXECUTION
    # ============================================================
    # Create startup manager with all required dependencies
    startup_manager = StartupManager(
        app=app,
        settings=settings,
        redis_client=state.redis_client,
        process_lock=state.process_lock,
    )
    
    try:
        # Execute phased startup
        await startup_manager.run_startup()
        
        # Yield control to application
        yield
        
    except asyncio.CancelledError:
        logger.info(f"VULCAN-AGI worker {worker_id} received cancellation signal")
        raise
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        # ============================================================
        # SHUTDOWN EXECUTION
        # ============================================================
        await startup_manager.run_shutdown()


# ============================================================
# FastAPI Application with Enhanced Security
# ============================================================



def create_app(settings) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Application settings
        
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.api_title,
        description="Advanced Multimodal Collective Intelligence System with Autonomous Self-Improvement",
        version=settings.api_version,
        docs_url="/docs" if settings.deployment_mode != "production" else None,
        redoc_url="/redoc" if settings.deployment_mode != "production" else None,
        lifespan=lifespan,
    )
    
    @app.get("/", response_class=JSONResponse)
    async def root():
        return {"status": "ok", "message": "VULCAN-AGI API is alive"}
    
    # CORS middleware
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=settings.cors_methods,
            allow_headers=["*"],
        )
    
    # Mount safety router
    try:
        from vulcan.safety.safety_status_endpoint import router as safety_router
        app.include_router(safety_router, prefix="/safety", tags=["safety"])
        logger.info("Safety status endpoint mounted at /safety")
    except Exception as e:
        logger.error(f"Failed to mount safety status endpoint: {e}")
    
    return app
