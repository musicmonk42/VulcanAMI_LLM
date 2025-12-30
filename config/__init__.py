"""
VULCAN Configuration Package.

This package provides centralized configuration for VULCAN components.

Modules:
    distillation_config: Configuration for knowledge distillation system
"""

from .distillation_config import (
    DISTILLATION_CONFIG,
    PRODUCTION_CONFIG,
    DEVELOPMENT_CONFIG,
    ENSEMBLE_CONFIG,
    get_config,
    validate_config,
)

__all__ = [
    "DISTILLATION_CONFIG",
    "PRODUCTION_CONFIG",
    "DEVELOPMENT_CONFIG",
    "ENSEMBLE_CONFIG",
    "get_config",
    "validate_config",
]
