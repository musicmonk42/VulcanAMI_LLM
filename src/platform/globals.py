"""Lazy accessor functions for Vulcan platform singletons.

These functions defer construction to first call, avoiding import-time
side effects (e.g., Settings.__init__ raising ValueError in production
when auth is not configured). Route modules call these inside request
handlers, not at module level.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from src.platform.settings import UnifiedPlatformSettings
    from src.platform.services import AsyncServiceManager
    from src.platform.session import FlashMessageManager

_lock = threading.Lock()
_app: "FastAPI | None" = None
_settings: "UnifiedPlatformSettings | None" = None
_service_manager: "AsyncServiceManager | None" = None
_flash_manager: "FlashMessageManager | None" = None


def get_app() -> "FastAPI":
    """Return the FastAPI app instance. Raises if not initialized."""
    if _app is None:
        raise RuntimeError("App not initialized. Call init_app() first.")
    return _app


def get_settings() -> "UnifiedPlatformSettings":
    """Return settings, delegating to settings.py singleton (thread-safe)."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                from src.platform.settings import get_settings as _get
                _settings = _get()
    return _settings


def get_service_manager() -> "AsyncServiceManager":
    """Return service manager. Raises if not initialized."""
    if _service_manager is None:
        raise RuntimeError("ServiceManager not initialized.")
    return _service_manager


def get_flash_manager() -> "FlashMessageManager":
    """Return flash manager, constructing on first call."""
    global _flash_manager
    if _flash_manager is None:
        with _lock:
            if _flash_manager is None:
                from src.platform.session import FlashMessageManager
                _flash_manager = FlashMessageManager()
    return _flash_manager


def init_app(app, settings, service_manager):
    """Called once by full_platform.py during startup."""
    global _app, _settings, _service_manager
    _app = app
    _settings = settings
    _service_manager = service_manager


# --- Background init flags (IRV-7) ---
_services_init_complete = False
_services_init_failed = False


def is_services_init_complete() -> bool:
    return _services_init_complete


def is_services_init_failed() -> bool:
    return _services_init_failed


def set_services_init_complete(value: bool) -> None:
    global _services_init_complete
    _services_init_complete = value


def set_services_init_failed(value: bool) -> None:
    global _services_init_failed
    _services_init_failed = value
