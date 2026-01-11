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

logger = logging.getLogger(__name__)


# ============================================================
# Module-Level Globals (P0 Fix: Issue #3)
# ============================================================
# These must be declared at module level with proper types to avoid
# UnboundLocalError when referenced in lifespan() function.

_process_lock: Optional[Any] = None
"""Process lock for split-brain prevention when Redis is unavailable."""

rate_limit_cleanup_thread: Optional[Thread] = None
"""Background thread for rate limit cleanup."""

redis_client: Optional[Any] = None
"""Redis client instance (may be None if Redis is not configured)."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager with phased startup.
    
    Delegates all initialization to StartupManager for modular,
    testable, and maintainable startup sequence.
    """
    global rate_limit_cleanup_thread, _process_lock, redis_client
    
    worker_id = os.getpid()
    
    # ============================================================
    # TEST MODE BYPASS (P1 Fix: Issue #8)
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
    # SPLIT-BRAIN PREVENTION (P0 Fix: Issue #3)
    # ============================================================
    # Acquire process lock when Redis is unavailable to ensure only
    # one orchestrator instance runs, preventing split-brain conditions.
    if redis_client is None:
        logger.warning(
            "Redis unavailable - acquiring process lock to prevent split-brain"
        )
        from vulcan.utils_main.process_lock import ProcessLock
        
        _process_lock = ProcessLock()
        if not _process_lock.acquire():
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
        logger.info("Process lock acquired - singleton mode active")
    else:
        logger.info("Redis available - multi-instance mode supported")
    
    # ============================================================
    # STARTUP EXECUTION
    # ============================================================
    # Import settings and create startup manager
    from vulcan.settings import Settings
    
    try:
        settings = Settings()
    except Exception as e:
        logger.error(f"Failed to initialize settings: {e}")
        settings = Settings()
    
    # Create startup manager with all required dependencies
    startup_manager = StartupManager(
        app=app,
        settings=settings,
        redis_client=redis_client,
        process_lock=_process_lock,
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
