#!/usr/bin/env python3
"""
VULCAN-AGI Refactored Main Entry Point

This is the streamlined entry point (222 lines) that uses 24 extracted modules.
All functionality has been modularized while maintaining full compatibility
with src/full_platform.py mounting.

Original 11,316-line monolith archived → 222-line entry point + 24 focused modules
"""

# ====================================================================
# PRIORITY 1: RESTRICT CPU THREADS - MUST BE ABSOLUTE FIRST
# Prevents PyTorch/NumPy/OpenBLAS from spawning 40+ threads causing
# "Thread Thrashing" which locks the CPU for 60-100+ seconds.
# These environment variables MUST be set before ANY torch/numpy imports.
#
# FORENSIC AUDIT FIX (Issue #2):
# These variables are ALSO set in:
# - entrypoint.sh (for Docker/container deployments)
# - Dockerfile (as ENV defaults)
# This ensures they are set BEFORE Python even starts, which is the only
# guaranteed way to control thread counts for compiled libraries.
#
# Setting them here in Python provides defense-in-depth for:
# - Direct Python execution (python main.py)
# - Development environments without Docker
# - Cases where entrypoint.sh is bypassed
#
# The pattern ${VAR:-default} in shell and setdefault here means
# user overrides via environment are respected.
# ====================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TORCH_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Prevent tokenizer deadlocks

# Disable TQDM progress bars (can enable with VULCAN_DEBUG=1)
if not os.environ.get("VULCAN_DEBUG"):
    os.environ["TQDM_DISABLE"] = "1"
else:
    os.environ.pop("TQDM_DISABLE", None)

# ====================================================================
# IMPORTS FROM EXTRACTED MODULES
# ====================================================================
import sys
import uvicorn
import logging

# Import server components
from vulcan.server import create_app
from vulcan.settings import Settings

# ====================================================================
# BACKWARD COMPATIBILITY EXPORTS
# ====================================================================
# This section maintains backward compatibility with code that imports
# directly from src.vulcan.main after the modular refactoring (PR #704).
#
# CONTEXT:
# - Original main.py: 11,316 lines (monolithic)
# - Refactored main.py: 222 lines (modular)
# - Components extracted to 24 focused modules
#
# DEPENDENCIES:
# The following imports are required by:
# 1. src/full_platform.py - Proxy endpoints for feedback and chat
# 2. tests/test_*.py - Test suites that import from main
# 3. External integrations - Any code using the previous API surface
#
# MAINTENANCE:
# When adding new exports, document the dependent code and reason.
# Consider deprecation warnings for future major version updates.
# ====================================================================

# Feedback models and handlers (used by full_platform.py proxy endpoints)
# Location: Lines 3602, 3660, 3700 in src/full_platform.py
from vulcan.api.models import (
    FeedbackRequest,           # RLHF feedback submission model
    ThumbsFeedbackRequest,     # Simplified thumbs up/down model
    UnifiedChatRequest,        # Full platform chat request model
)
from vulcan.endpoints.feedback import (
    submit_feedback,           # POST /v1/feedback handler
    submit_thumbs_feedback,    # POST /v1/feedback/thumbs handler
    get_feedback_stats,        # GET /v1/feedback/stats handler
)

# Unified chat endpoint handler (used by full_platform.py proxy)
# Location: Line 3454 in src/full_platform.py
from vulcan.endpoints.unified_chat import (
    unified_chat,              # POST /v1/chat handler
)

# Import all endpoint routers
from vulcan.endpoints import (
    agents_router,
    config_router,
    distillation_router,
    execution_router,
    feedback_router,
    health_router,
    memory_router,
    monitoring_router,
    planning_router,
    reasoning_router,
    safety_router,
    self_improvement_router,
    status_router,
    unified_chat_router,
    world_model_router,
)

# ====================================================================
# SETTINGS AND CONFIGURATION
# ====================================================================
logger = logging.getLogger(__name__)

# Initialize settings
try:
    settings = Settings()
except Exception as e:
    logger.error(f"Failed to initialize settings: {e}")
    # Create minimal settings for fallback
    settings = Settings()

# ====================================================================
# CREATE FASTAPI APP - EXPORTED FOR full_platform.py
# ====================================================================
# The 'app' variable is exported so that src/full_platform.py can:
# vulcan_module = importlib.import_module("src.vulcan.main")
# app.mount("/vulcan", vulcan_module.app)
app = create_app(settings)

# ====================================================================
# MOUNT ALL ENDPOINT ROUTERS
# ====================================================================
# Health & Monitoring
app.include_router(health_router, tags=["health"])
app.include_router(monitoring_router, tags=["monitoring"])

# Configuration
app.include_router(config_router, tags=["configuration"])

# Status & Diagnostics
app.include_router(status_router, tags=["status"])

# Core Functionality
app.include_router(execution_router, tags=["execution"])
app.include_router(planning_router, tags=["planning"])
app.include_router(memory_router, tags=["memory"])

# Advanced Features
app.include_router(self_improvement_router, tags=["self-improvement"])
app.include_router(reasoning_router, tags=["reasoning"])
app.include_router(feedback_router, tags=["feedback"])
app.include_router(distillation_router, tags=["distillation"])

# Agent & World Model
app.include_router(agents_router, tags=["agents"])
app.include_router(world_model_router, tags=["world-model"])

# Safety
app.include_router(safety_router, tags=["safety"])

# Chat Endpoint (Unified)
app.include_router(unified_chat_router, tags=["chat"])

logger.info("VULCAN-AGI: All endpoint routers mounted successfully")
logger.info(f"VULCAN-AGI: Application initialized with {len(app.routes)} routes")

# ====================================================================
# CLI ENTRY POINT
# ====================================================================
def main():
    """
    Main entry point with CLI argument handling.
    
    Supports:
    - python main.py --interactive : Start interactive REPL mode
    - python main.py --test : Run functional test suite
    - python main.py --benchmark : Run performance benchmarks
    - python main.py : Start production server (default)
    """
    if "--interactive" in sys.argv:
        # Interactive REPL mode
        logger.info("Starting VULCAN interactive mode...")
        try:
            from vulcan.cli.interactive import run_interactive_mode
            run_interactive_mode()
        except Exception as e:
            logger.error(f"Failed to start interactive mode: {e}")
            sys.exit(1)
    
    elif "--test" in sys.argv:
        # Run functional tests
        logger.info("Running VULCAN functional test suite...")
        try:
            from vulcan.tests.test_functional import run_all_tests
            success = run_all_tests()
            sys.exit(0 if success else 1)
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            sys.exit(1)
    
    elif "--benchmark" in sys.argv:
        # Run performance benchmarks
        logger.info("Running VULCAN performance benchmarks...")
        try:
            from vulcan.tests.test_benchmarks import run_all_benchmarks
            run_all_benchmarks()
        except Exception as e:
            logger.error(f"Failed to run benchmarks: {e}")
            sys.exit(1)
    
    elif "--subsystem-tests" in sys.argv:
        # Run subsystem integration tests
        logger.info("Running VULCAN subsystem integration tests...")
        try:
            from vulcan.tests.test_subsystems import run_subsystem_tests
            success = run_subsystem_tests()
            sys.exit(0 if success else 1)
        except Exception as e:
            logger.error(f"Failed to run subsystem tests: {e}")
            sys.exit(1)
    
    else:
        # Start production server (default)
        logger.info("Starting VULCAN-AGI production server...")
        logger.info(f"Server will be available at http://0.0.0.0:8000")
        logger.info(f"Health check: http://0.0.0.0:8000/health")
        logger.info(f"API docs: http://0.0.0.0:8000/docs")
        
        # Get server configuration from settings
        host = getattr(settings, 'host', '0.0.0.0')
        port = getattr(settings, 'port', 8000)
        workers = getattr(settings, 'workers', 1)
        log_level = getattr(settings, 'log_level', 'info')
        
        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                workers=workers,
                log_level=log_level,
                access_log=True,
                use_colors=True,
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            sys.exit(1)

# ====================================================================
# ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log startup information
    logger.info("=" * 70)
    logger.info("VULCAN-AGI: Advanced General Intelligence System")
    logger.info("Refactored Architecture: 24 Focused Modules")
    logger.info("=" * 70)
    
    # Run main entry point
    main()
