#!/usr/bin/env python3
# =============================================================================
# Platform Startup Utilities
# =============================================================================
# Extracted from full_platform.py:
# - _get_startup_elapsed (timing diagnostics)
# - setup_unified_logging (unified logging configuration)
# - setup_python_path (Python path setup for imports)
# - _build_subprocess_env (subprocess environment builder)
# - _configure_ml_threading (ML thread limit configuration)
# =============================================================================

import logging
import os
import sys
import time as _startup_time_module
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# STARTUP TIMING (for Railway/production diagnostics)
# =============================================================================
# Timestamp at very start of module loading to track cold start time.
_MODULE_LOAD_START = _startup_time_module.time()


def _get_startup_elapsed() -> float:
    """Return elapsed time since module load started (in seconds)."""
    return _startup_time_module.time() - _MODULE_LOAD_START


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_unified_logging():
    """Configure unified logging for all services with UTF-8 support."""
    # Import settings lazily to avoid circular imports
    from full_platform import settings

    # Clear existing handlers to avoid duplicates from the watcher process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create handlers with explicit UTF-8 encoding
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Ensure stdout is UTF-8 on Windows
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            # Silently ignore - logger not available yet at this point
            pass

    # File handler with explicit UTF-8 encoding
    file_handler = logging.FileHandler("unified_platform.log", encoding="utf-8")

    # Configure logging with UTF-8 handlers
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[stdout_handler, file_handler],
        force=True,  # Force re-configuration
    )

    # Set log levels for sub-apps
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("vulcan").setLevel(logging.INFO)
    logging.getLogger("arena").setLevel(logging.INFO)
    logging.getLogger("registry").setLevel(logging.INFO)


# =============================================================================
# PATH SETUP WITH ABSOLUTE IMPORT SUPPORT
# =============================================================================


def setup_python_path():
    """
    Ensure proper Python path for imports.
    Supports both flat and src/ directory structures.
    """
    script_dir = Path(__file__).resolve().parent

    # Add current directory
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Add src directory if it exists
    src_dir = script_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        # Use print here as logger isn't configured yet
        print(f"✓ Added src directory to path: {src_dir}")

    # Add parent directory
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    print(f"Debug: Python path configured: {sys.path[:3]}...")


def _build_subprocess_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build environment for subprocess with proper PYTHONPATH.

    PERFORMANCE FIX: This ensures subprocesses can find llm_client and other
    project modules, preventing ModuleNotFoundError and 30-40s timeout per query.

    Args:
        extra_env: Optional additional environment variables to include

    Returns:
        Environment dictionary with PYTHONPATH set correctly
    """
    # Get project root and src directory
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    # Start with copy of current environment
    env = os.environ.copy()

    # Build PYTHONPATH: /app:/app/src (or equivalent for local paths)
    # Use str() for cross-platform compatibility (Windows vs Unix)
    existing_pythonpath = env.get("PYTHONPATH", "")
    new_pythonpath = f"{str(project_root)}{os.pathsep}{str(src_dir)}"
    if existing_pythonpath:
        new_pythonpath = f"{new_pythonpath}{os.pathsep}{existing_pythonpath}"
    env["PYTHONPATH"] = new_pythonpath

    # Add any extra environment variables
    if extra_env:
        env.update(extra_env)

    return env


# =============================================================================
# ML THREADING CONFIGURATION
# =============================================================================


def _configure_ml_threading(logger) -> None:
    """
    Configure thread limits for ML libraries (PyTorch, BLAS, OpenMP).

    This function should be called AFTER the server starts accepting connections
    to avoid blocking healthchecks during module import.

    Environment variables are already set at module level, but this function
    applies runtime configuration for libraries that need it.
    """
    # Configure threadpoolctl for BLAS/OpenMP
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=4, user_api='blas')
        threadpool_limits(limits=4, user_api='openmp')
        logger.info("[THREAD_LIMIT] threadpoolctl limits applied: blas=4, openmp=4")
    except ImportError:
        logger.info("[THREAD_LIMIT] threadpoolctl not available, using env vars only")

    # Import and configure PyTorch
    try:
        import torch
        current_threads = torch.get_num_threads()
        logger.info(f"[THREAD_LIMIT] torch.get_num_threads() = {current_threads}")
        if current_threads > 4:
            torch.set_num_threads(4)
            logger.info(f"[THREAD_LIMIT] Reduced torch threads to {torch.get_num_threads()}")
    except ImportError:
        logger.info("[THREAD_LIMIT] torch not available, skipping torch thread limits")
    except RuntimeError as e:
        logger.warning(f"[THREAD_LIMIT] Could not set torch threads: {e}")
