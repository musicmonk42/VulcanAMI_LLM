"""
VULCAN Configuration Package.

This package provides centralized, security-first configuration for VULCAN components.

Security Principles:
    - Encryption enabled by default in production
    - Opt-in required by default (privacy-first)
    - PII redaction always enabled
    - Configurable retention for compliance (GDPR/CCPA)

Modules:
    distillation_config: Configuration for knowledge distillation system

Usage:
    from config import get_config, validate_config
    
    # Get production configuration (secure defaults)
    config = get_config("production")
    
    # Validate configuration
    validate_config(config)

Environment Variables:
    DISTILLATION_ENCRYPTION_KEY: Fernet key for encryption
    DISTILLATION_ENV: Environment name (production, development, staging)

Copyright (c) 2024 VULCAN-AGI Team
License: MIT
"""

from .distillation_config import (
    DISTILLATION_CONFIG,
    PRODUCTION_CONFIG,
    STAGING_CONFIG,
    DEVELOPMENT_CONFIG,
    ENSEMBLE_CONFIG,
    get_config,
    validate_config,
    get_retention_config,
)

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

__all__ = [
    "DISTILLATION_CONFIG",
    "PRODUCTION_CONFIG",
    "STAGING_CONFIG",
    "DEVELOPMENT_CONFIG",
    "ENSEMBLE_CONFIG",
    "get_config",
    "validate_config",
    "get_retention_config",
]
