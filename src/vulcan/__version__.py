#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Version Information.

This module provides version information for the VULCAN-AGI package.
It follows PEP 440 versioning conventions and provides both string
and tuple representations of the version.

Versioning Scheme:
    MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

    - MAJOR: Incompatible API changes
    - MINOR: New functionality (backward compatible)
    - PATCH: Notees (backward compatible)
    - PRERELEASE: Optional (alpha, beta, rc)
    - BUILD: Optional build metadata

Example:
    >>> from src.vulcan.__version__ import __version__, __version_info__
    >>> print(__version__)
    2.0.0
    >>> print(__version_info__)
    (2, 0, 0)

Author: VULCAN-AGI Team
License: MIT
"""

from __future__ import annotations

from typing import Tuple

# Version string (PEP 440 compliant)
__version__: str = "2.0.0"

# Version tuple for programmatic comparison
__version_info__: Tuple[int, int, int] = (2, 0, 0)

# Package metadata
__author__: str = "VULCAN-AGI Team"
__author_email__: str = "team@vulcan-agi.dev"
__license__: str = "MIT"
__copyright__: str = "Copyright 2024-2026 VULCAN-AGI Team"

# Build information (can be set by CI/CD)
__build__: str = ""
__build_date__: str = ""

# All public exports
__all__ = [
    "__version__",
    "__version_info__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "__build__",
    "__build_date__",
]
