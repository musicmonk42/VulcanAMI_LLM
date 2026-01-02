#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Scripts Package.

This package contains utility scripts for VULCAN installation verification,
system health checks, and maintenance operations.

Available Scripts:
    - verify_installation: Verify VULCAN package installation
    - health_smoke: Quick health check for running VULCAN instances

Usage:
    Command line:
        python -m scripts.verify_installation

    After installation:
        vulcan-verify

Author: VULCAN-AGI Team
License: MIT
"""

from __future__ import annotations

__all__: list[str] = [
    "verify_installation",
]
