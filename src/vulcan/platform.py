"""Canonical serving platform policy.

Linux containers are the only qualified serving platform.  This check is kept
at the composition boundary so unsupported hosts fail before authority stores
or request state are created.  Portable research utilities use the explicit
``ProcessLock`` adapter; they are not part of canonical serving.
"""
from __future__ import annotations

import sys


class UnsupportedServingPlatform(RuntimeError):
    """Raised before composition on a platform without qualification evidence."""


def require_canonical_serving_platform(platform: str | None = None) -> None:
    selected = sys.platform if platform is None else platform
    if selected != "linux":
        raise UnsupportedServingPlatform(
            "canonical serving is qualified only for Linux/Docker"
        )
