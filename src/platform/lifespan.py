#!/usr/bin/env python3
# =============================================================================
# Platform Lifespan Management
# =============================================================================
# Extracted from full_platform.py:
# - lifespan async context manager (startup/shutdown lifecycle)
# =============================================================================

import asyncio
import importlib
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from platform.background import release_background_task_lock
from platform.startup import _get_startup_elapsed, setup_unified_logging

logger = logging.getLogger("unified_platform")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the unified platform.
    Handles startup and graceful shutdown.

    CRITICAL FIX for Railway Healthcheck:
    This lifespan function yields IMMEDIATELY after basic setup to allow
    the server to start accepting HTTP connections. All heavy initialization
    (service mounting, ML model loading, etc.) is done in background tasks
    AFTER the yield. This ensures /health/live responds immediately.
    """
    # Import from full_platform to access shared state
    import full_platform
    from full_platform import _background_services_initialization

    worker_id = os.getpid()

    # ====================================================================
    # MINIMAL BLOCKING SETUP (must complete before yield)
    # ====================================================================

    # Fix Windows console encoding issues
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
            for handler in logging.root.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    try:
                        if hasattr(handler.stream, "reconfigure"):
                            handler.stream.reconfigure(encoding="utf-8")
                    except (AttributeError, OSError):
                        pass
            print("✅ Reconfigured stdout/stderr to UTF-8")
        except Exception as e:
            print(f"⚠️  Could not reconfigure encoding: {e}")

    # Load environment variables
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"✅ Loaded environment from: {env_path}")
    except ImportError:
        pass
    except Exception as e:
        print(f"❌ Error loading .env: {e}")

    # Setup logging
    setup_unified_logging()

    # Initialize state flags
    app.state.models_loaded = False
    app.state.services_initialized = False
    app.state.background_processes = []
    app.state.components_status = {}
    app.state.background_tasks_initialized = False

    # ====================================================================
    # YIELD IMMEDIATELY - Server starts accepting connections NOW
    # ====================================================================
    # /health/live will respond as soon as we yield, before any services
    # are mounted. This is critical for Railway/K8s healthcheck success.
    #
    # NOTE: The _services_init_started flag is checked here to prevent
    # duplicate background tasks. This is safe because:
    # 1. lifespan() runs exactly once per worker process
    # 2. For multi-worker deployments, acquire_background_task_lock()
    #    ensures only one worker initializes background tasks
    # ====================================================================

    logger.info("=" * 70)
    logger.info(f"Starting Unified Platform (Worker {worker_id})")
    logger.info("=" * 70)
    logger.info(f"🚀 Lifespan started at {_get_startup_elapsed():.2f}s since module load")
    logger.info("🚀 Server accepting connections - scheduling background initialization...")

    # Schedule background services initialization (NON-BLOCKING)
    # This flag check is safe: lifespan runs once per process, not concurrently
    if not full_platform._services_init_started:
        full_platform._services_init_started = True
        asyncio.create_task(
            _background_services_initialization(app, worker_id, logger),
            name="services_initialization"
        )

    try:
        logger.info(f"✅ YIELDING NOW - /health/live will respond (total startup: {_get_startup_elapsed():.2f}s)")
        yield  # Server is now accepting connections!
    except asyncio.CancelledError:
        logger.info(f"Unified Platform (Worker {worker_id}) received cancellation signal")
    finally:
        # SHUTDOWN
        logger.info("=" * 70)
        logger.info(f"Shutting down Unified Platform (Worker {worker_id})...")
        logger.info("=" * 70)

        # Shutdown adversarial tester (only if we initialized it)
        if getattr(app.state, 'background_tasks_initialized', False):
            try:
                from vulcan.safety.adversarial_integration import (
                    shutdown_adversarial_tester,
                )

                shutdown_adversarial_tester()
                logger.info("✓ AdversarialTester shutdown complete")
            except ImportError:
                pass
            except Exception as adv_err:
                logger.error(f"Error during adversarial tester shutdown: {adv_err}")

            # Shutdown CuriosityDriver (only if we initialized it)
            # FIX Issue 1: Robust shutdown with multiple fallback locations
            try:
                curiosity_driver = None

                # Try to get driver from vulcan module app state first
                try:
                    vulcan_module = importlib.import_module("src.vulcan.main")
                    curiosity_driver = getattr(
                        vulcan_module.app.state, "curiosity_driver", None
                    )
                except (ImportError, AttributeError):
                    pass

                # Fallback: Check main app.state directly
                if curiosity_driver is None:
                    curiosity_driver = getattr(app.state, "curiosity_driver", None)

                if curiosity_driver is not None:
                    logger.info("Stopping CuriosityDriver...")
                    await curiosity_driver.stop()
                    logger.info("✓ CuriosityDriver shutdown complete")
                else:
                    logger.debug("CuriosityDriver not found in app state (not initialized or already stopped)")
            except Exception as cd_err:
                logger.error(f"Error during CuriosityDriver shutdown: {cd_err}")

            # Release the background task lock
            release_background_task_lock()
            logger.info("✓ Background task lock released")

        # Cleanup background processes
        if hasattr(app.state, "background_processes"):
            logger.info("Terminating background services...")
            for service_name, process in app.state.background_processes:
                try:
                    if process.poll() is None:  # Process is still running
                        logger.info(
                            f"Terminating {service_name} (PID: {process.pid})..."
                        )
                        process.terminate()
                        try:
                            # Wait up to 5 seconds for graceful shutdown
                            process.wait(timeout=5.0)
                            logger.info(f"✓ {service_name} terminated gracefully")
                        except subprocess.TimeoutExpired:
                            # Force kill if graceful shutdown fails
                            logger.warning(
                                f"Force killing {service_name} (PID: {process.pid})..."
                            )
                            process.kill()
                            process.wait()
                            logger.info(f"✓ {service_name} killed")
                except Exception as e:
                    logger.error(f"Error terminating {service_name}: {e}")

        # Cleanup tasks
        logger.info("Shutdown complete")
