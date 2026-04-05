"""Canonical logging configuration for Vulcan AMI.

Call configure() once from entry points (api_server.py, full_platform.py).
All other modules: logger = logging.getLogger(__name__)
"""
import logging
import logging.handlers
import sys

_CONFIGURED = False


def configure(
    level: str = "INFO",
    fmt: str = "%(asctime)s %(name)s %(levelname)s %(message)s",
    log_file: str | None = None,
) -> None:
    """Configure logging once. Subsequent calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_val = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10_000_000, backupCount=5
            )
        )
    logging.basicConfig(level=level_val, format=fmt, handlers=handlers)
    _CONFIGURED = True
