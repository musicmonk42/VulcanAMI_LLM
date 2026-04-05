#!/usr/bin/env python3
# =============================================================================
# Background Task Management
# =============================================================================
# Extracted from full_platform.py:
# - acquire_background_task_lock (ghost process prevention)
# - release_background_task_lock (lock cleanup)
# - _background_model_loading (async ML model loading)
# =============================================================================

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Platform-specific file locking
# fcntl is Unix-only, msvcrt is Windows-only
_HAVE_FCNTL = False
_HAVE_MSVCRT = False

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:
    pass

if not _HAVE_FCNTL:
    try:
        import msvcrt
        _HAVE_MSVCRT = True
    except ImportError:
        pass

# Global lock file path and file descriptor for background task singleton
# Include port number in filename to allow multiple instances on different ports
_PLATFORM_PORT = os.environ.get("PORT", os.environ.get("UNIFIED_PORT", "8080"))
_BACKGROUND_TASK_LOCK_FILE = Path(tempfile.gettempdir()) / f"vulcan_platform_background_tasks_{_PLATFORM_PORT}.lock"
_background_task_lock_fd = None


def acquire_background_task_lock() -> bool:
    """
    Acquire an exclusive lock to prevent duplicate background task initialization.

    Uses file locking (flock on Unix, msvcrt on Windows) to ensure only ONE process
    initializes background tasks, regardless of how many processes uvicorn spawns.

    The lock file includes the port number, allowing multiple instances of the
    application to run on different ports without interfering with each other.

    Returns:
        True if lock was acquired (this process should initialize tasks)
        False if lock is held by another process (skip initialization)
    """
    global _background_task_lock_fd

    # If no file locking is available, return True (allow initialization)
    # This means ghost process prevention won't work on unsupported platforms
    if not _HAVE_FCNTL and not _HAVE_MSVCRT:
        print("⚠️  File locking not available - ghost process prevention disabled")
        return True

    try:
        # Open/create the lock file
        _background_task_lock_fd = open(_BACKGROUND_TASK_LOCK_FILE, 'w')

        if _HAVE_FCNTL:
            # Unix: Use fcntl.flock for exclusive non-blocking lock
            fcntl.flock(_background_task_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif _HAVE_MSVCRT:
            # Windows: Use msvcrt.locking for exclusive non-blocking lock
            msvcrt.locking(_background_task_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)

        # Write our PID to the lock file for debugging
        _background_task_lock_fd.write(f"PID: {os.getpid()}\n")
        _background_task_lock_fd.write(f"Time: {datetime.now(timezone.utc).isoformat()}\n")
        _background_task_lock_fd.write(f"Port: {_PLATFORM_PORT}\n")
        _background_task_lock_fd.flush()

        return True

    except (IOError, OSError) as e:
        # Lock is held by another process - this is expected in multi-process scenarios
        if _background_task_lock_fd:
            try:
                _background_task_lock_fd.close()
            except Exception:
                pass
            _background_task_lock_fd = None
        return False
    except Exception as e:
        # Unexpected error - log and proceed cautiously
        print(f"⚠️  Warning: Failed to acquire background task lock: {e}")
        return False


def release_background_task_lock():
    """
    Release the background task lock during shutdown.
    """
    global _background_task_lock_fd

    if _background_task_lock_fd:
        try:
            if _HAVE_FCNTL:
                fcntl.flock(_background_task_lock_fd.fileno(), fcntl.LOCK_UN)
            elif _HAVE_MSVCRT:
                msvcrt.locking(_background_task_lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            _background_task_lock_fd.close()
        except Exception as e:
            print(f"⚠️  Warning: Failed to release background task lock: {e}")
        finally:
            _background_task_lock_fd = None

        # Optionally clean up the lock file
        try:
            if _BACKGROUND_TASK_LOCK_FILE.exists():
                _BACKGROUND_TASK_LOCK_FILE.unlink()
        except Exception:
            pass  # Lock file cleanup is optional


async def _background_model_loading(app_state: Any, components_status: dict, logger) -> None:
    """
    Background task to load heavy ML models AFTER server starts accepting connections.

    This function is called after the lifespan yields, allowing the /health/live endpoint
    to respond immediately while models load in the background.

    CRITICAL FIX: This prevents Railway healthcheck failures by ensuring the server
    accepts connections before heavy model loading completes.
    """
    from platform.startup import _configure_ml_threading

    # Import the global flag from full_platform
    import full_platform

    try:
        logger.info("=" * 70)
        logger.info("Background Model Loading Started (non-blocking)")
        logger.info("=" * 70)

        # Configure thread limits for ML libraries NOW (after server is accepting connections)
        _configure_ml_threading(logger)

        # Give the event loop time to process incoming requests
        await asyncio.sleep(0.1)

        try:
            # 1. Pre-load BERT model (GraphixTransformer singleton)
            logger.info("Pre-loading BERT model...")
            from vulcan.processing import GraphixTransformer
            _ = GraphixTransformer.get_instance()  # Force singleton initialization
            logger.info("  ✅ BERT model pre-loaded")
            components_status["BERT Model"] = True
        except Exception as e:
            logger.warning(f"  ⚠ BERT model pre-load failed: {e}")
            components_status["BERT Model"] = False

        # Give the event loop time to process incoming requests
        await asyncio.sleep(0.1)

        try:
            # 2. Pre-load QueryAnalyzer singleton (includes safety validators)
            logger.info("Pre-loading QueryAnalyzer...")
            from vulcan.routing.query_router import get_query_analyzer
            _ = get_query_analyzer()  # Force singleton initialization
            logger.info("  ✅ QueryAnalyzer pre-loaded")
            components_status["QueryAnalyzer"] = True
        except Exception as e:
            logger.warning(f"  ⚠ QueryAnalyzer pre-load failed: {e}")
            components_status["QueryAnalyzer"] = False

        # Give the event loop time to process incoming requests
        await asyncio.sleep(0.1)

        try:
            # 3. Pre-load DynamicModelManager and essential models
            logger.info("Pre-loading text embedding model...")
            from vulcan.processing import DynamicModelManager
            model_manager = DynamicModelManager()
            model_manager.preload_essential_models()
            logger.info("  ✅ Text embedding model pre-loaded")
            components_status["Text Embedding Model"] = True
        except Exception as e:
            logger.warning(f"  ⚠ Text embedding model pre-load failed: {e}")
            components_status["Text Embedding Model"] = False

        # Mark as complete
        full_platform._background_init_complete = True
        app_state.models_loaded = True

        logger.info("=" * 70)
        logger.info("Background Model Loading Complete!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Background model loading failed: {e}", exc_info=True)
        full_platform._background_init_complete = True  # Still mark complete so we don't block forever
        app_state.models_loaded = False
