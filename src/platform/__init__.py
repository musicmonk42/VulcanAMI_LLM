"""Platform package for Vulcan AMI unified platform.

This package name collides with Python's stdlib ``platform`` module when the
repository ``src`` directory is on ``PYTHONPATH``.  Expose the stdlib surface
needed by tooling and keep Vulcan-specific exports lazy so dependency-light test
collection does not import web/research stacks.
"""
from __future__ import annotations

import importlib.util as _util
import sysconfig as _sysconfig
from pathlib import Path as _Path

_stdlib_platform = _Path(_sysconfig.get_path("stdlib")) / "platform.py"
_spec = _util.spec_from_file_location("_stdlib_platform", _stdlib_platform)
if _spec and _spec.loader:
    _module = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    for _name in dir(_module):
        if not _name.startswith("_"):
            globals().setdefault(_name, getattr(_module, _name))

_LAZY = {
    "AuthMethod": ".auth", "JWTAuth": ".auth", "AuthenticationError": ".auth",
    "SecretsManager": ".secrets", "UnifiedPlatformSettings": ".settings",
    "FlashMessage": ".session", "FlashMessageManager": ".session",
    "ServiceImportResult": ".service_imports", "import_service_async": ".service_imports",
    "check_service_health_async": ".service_imports", "stop_service": ".service_lifecycle",
    "start_service": ".service_lifecycle",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(module_name, __name__), name)
