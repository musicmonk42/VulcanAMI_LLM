#!/usr/bin/env python3
# =============================================================================
# Platform CLI & Main Entry Point
# =============================================================================
# Extracted from full_platform.py:
# - parse_args (command-line argument parsing)
# - __main__ block (server launch)
# =============================================================================

import argparse
import logging

from platform.startup import setup_unified_logging

logger = logging.getLogger("unified_platform")


def parse_args():
    """Parse command-line arguments."""
    from full_platform import settings

    parser = argparse.ArgumentParser(description="Graphix Vulcan Unified Platform v2.1")

    parser.add_argument("--host", default=settings.host, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind")
    parser.add_argument(
        "--workers", type=int, default=settings.workers, help="Number of workers"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument("--mount-vulcan", default=settings.vulcan_mount)
    parser.add_argument("--mount-arena", default=settings.arena_mount)
    parser.add_argument("--mount-registry", default=settings.registry_mount)

    parser.add_argument(
        "--vulcan-module",
        default=settings.vulcan_module,
        help="VULCAN import module (e.g., 'main' or 'src.vulcan.main')",
    )
    parser.add_argument(
        "--arena-module", default=settings.arena_module, help="Arena import module"
    )
    parser.add_argument(
        "--registry-module",
        default=settings.registry_module,
        help="Registry import module",
    )

    parser.add_argument(
        "--auth-method",
        choices=["none", "api_key", "jwt", "oauth2"],
        default=settings.auth_method.value,
        help="Authentication method",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--jwt-secret", help="JWT secret key")

    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-health-checks", action="store_true")
    parser.add_argument(
        "--no-auto-detect-src",
        action="store_true",
        help="Disable automatic src/ directory detection",
    )

    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    from full_platform import AuthMethod, settings

    args = parse_args()

    settings.host = args.host
    settings.port = args.port
    settings.workers = args.workers
    settings.reload = args.reload
    settings.vulcan_mount = args.mount_vulcan
    settings.arena_mount = args.mount_arena
    settings.registry_mount = args.mount_registry
    settings.vulcan_module = args.vulcan_module
    settings.arena_module = args.arena_module
    settings.registry_module = args.registry_module
    settings.auth_method = AuthMethod(args.auth_method)
    settings.log_level = args.log_level
    settings.auto_detect_src = not args.no_auto_detect_src

    if args.api_key:
        settings.api_key = args.api_key
    if args.jwt_secret:
        settings.jwt_secret = args.jwt_secret
    if args.disable_metrics:
        settings.enable_metrics = False
    if args.disable_health_checks:
        settings.enable_health_checks = False

    setup_unified_logging()

    import uvicorn

    logger.info("=" * 70)
    logger.info("Launching Unified Platform Server v2.1")
    logger.info("=" * 70)
    logger.info(f"URL: http://{settings.host}:{settings.port}")
    logger.info(f"Docs: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Workers: {settings.workers}")
    logger.info(f"Auth: {settings.auth_method.value}")
    logger.info("=" * 70)

    uvicorn.run(
        "full_platform:app",
        host=settings.host,
        port=settings.port,
        workers=1 if settings.reload else settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
