"""
VULCAN Distillation Configuration
==================================

Centralized, security-first configuration for the knowledge distillation system.

This module provides environment-specific configurations with sensible security
defaults that comply with industry best practices for data protection.

Security Principles:
    - Encryption enabled by default in production
    - Opt-in required by default (privacy-first)
    - PII redaction always enabled
    - Configurable retention for compliance (GDPR/CCPA)

Usage:
    from config.distillation_config import get_config, validate_config
    
    # Get production configuration (secure defaults)
    config = get_config("production")
    
    # Validate before use
    validate_config(config)
    
    # Override specific settings
    config["retention_days"] = 7  # GDPR-compliant shorter retention

Environment Variables:
    DISTILLATION_ENCRYPTION_KEY: Fernet key for encryption (required in production)
    DISTILLATION_ENV: Environment name (production, development, staging)

Copyright (c) 2024 VULCAN-AGI Team
License: MIT
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# ============================================================
# SECURITY CONSTANTS
# ============================================================

# Minimum retention days (for audit trail requirements)
MIN_RETENTION_DAYS = 1

# Maximum retention days (for data minimization)
MAX_RETENTION_DAYS = 365

# Default retention aligned with common compliance requirements
DEFAULT_RETENTION_DAYS = 30

# Valid executor modes
VALID_EXECUTOR_MODES = frozenset({"ensemble", "local_first", "openai_first", "parallel"})

# ============================================================
# BASE CONFIGURATION
# ============================================================

DISTILLATION_CONFIG: Dict[str, Any] = {
    # Storage Settings
    "storage_path": "data/distillation_examples.jsonl",
    "storage_dir": "data/distillation",
    "max_buffer_size": 100,           # Flush to disk every N examples
    "retention_days": DEFAULT_RETENTION_DAYS,  # Configurable for compliance

    # Privacy & Security (SECURE DEFAULTS)
    # These defaults comply with privacy-by-design principles
    "require_opt_in": True,           # REQUIRED: User consent is mandatory
    "enable_pii_redaction": True,     # REQUIRED: Always redact PII
    "enable_governance_check": True,  # REQUIRED: Check sensitive content
    "use_encryption": True,           # SECURE DEFAULT: Encryption enabled
    "encryption_key": None,           # Set via DISTILLATION_ENCRYPTION_KEY env var

    # Quality Thresholds
    "min_quality_score": 0.65,
    "max_boilerplate_ratio": 0.4,
    "min_response_length": 50,
    "max_response_length": 4000,

    # Hybrid Executor Settings
    "executor_mode": "ensemble",      # ensemble, local_first, openai_first, parallel
    "executor_timeout": 30.0,         # Seconds to wait for both LLMs
    "openai_max_tokens": 2000,        # Max tokens for OpenAI

    # Training Integration
    "training_batch_size": 32,
    "training_check_interval": 10,    # Check storage every N steps
    
    # Compliance Settings
    "log_capture_events": True,       # Audit trail for captures
    "anonymize_logs": True,           # Don't log PII in capture logs
}

# ============================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ============================================================

PRODUCTION_CONFIG: Dict[str, Any] = {
    **DISTILLATION_CONFIG,
    # Security hardened for production
    "require_opt_in": True,           # MANDATORY in production
    "use_encryption": True,           # MANDATORY in production
    "enable_pii_redaction": True,     # MANDATORY in production
    "enable_governance_check": True,  # MANDATORY in production
    "min_quality_score": 0.70,        # Higher quality bar in production
    "log_capture_events": True,       # Required for audit compliance
    "anonymize_logs": True,           # GDPR/CCPA compliance
}

STAGING_CONFIG: Dict[str, Any] = {
    **DISTILLATION_CONFIG,
    # Staging mirrors production security but with observability
    "require_opt_in": True,
    "use_encryption": True,
    "enable_pii_redaction": True,
    "enable_governance_check": True,
    "min_quality_score": 0.65,
    "retention_days": 7,              # Shorter retention for staging
}

DEVELOPMENT_CONFIG: Dict[str, Any] = {
    **DISTILLATION_CONFIG,
    # Development allows relaxed settings for testing
    # WARNING: Never use these settings in production
    "require_opt_in": False,          # FOR TESTING ONLY - enables capture without consent
    "use_encryption": False,          # FOR TESTING ONLY - disables encryption
    "min_quality_score": 0.50,        # Lower bar for testing
    "max_buffer_size": 20,            # Smaller buffer for faster testing
    "retention_days": 1,              # Minimal retention for dev
    "log_capture_events": True,       # Keep logging for debugging
}

# Ensemble mode specific settings (recommended for knowledge distillation)
ENSEMBLE_CONFIG: Dict[str, Any] = {
    **PRODUCTION_CONFIG,              # Start with production security
    "executor_mode": "ensemble",
    "combine_responses": True,        # Append local insights to OpenAI response
    "insight_threshold": 0.3,         # Minimum uniqueness for local insights
}


# ============================================================
# CONFIGURATION ACCESSOR
# ============================================================

def get_config(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for specified environment.
    
    Security Note:
        Production configuration has encryption enabled by default.
        If DISTILLATION_ENCRYPTION_KEY is not set, a warning is logged
        and a temporary key is generated (NOT recommended for production).
    
    Args:
        env: Environment name. If None, reads from DISTILLATION_ENV 
             environment variable, defaulting to "production".
        
    Returns:
        Dictionary with configuration settings for the environment.
        
    Example:
        >>> config = get_config("production")
        >>> print(config["use_encryption"])
        True
        
        >>> config = get_config()  # Uses DISTILLATION_ENV or defaults to production
    """
    # Determine environment
    if env is None:
        env = os.getenv("DISTILLATION_ENV", "production")
    
    configs = {
        "production": PRODUCTION_CONFIG,
        "staging": STAGING_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "ensemble": ENSEMBLE_CONFIG,
    }
    
    config = configs.get(env, PRODUCTION_CONFIG).copy()
    
    # Handle encryption key from environment
    if config.get("use_encryption") and not config.get("encryption_key"):
        env_key = os.getenv("DISTILLATION_ENCRYPTION_KEY")
        if env_key:
            config["encryption_key"] = env_key
        elif env == "production":
            logger.warning(
                "SECURITY WARNING: use_encryption=True but DISTILLATION_ENCRYPTION_KEY "
                "not set. Set this environment variable in production. "
                "Generating temporary key (data will be unrecoverable after restart)."
            )
            # Generate a temporary key - NOT for production use
            config["encryption_key"] = _generate_temporary_key()
    
    return config


def _generate_temporary_key() -> str:
    """
    Generate a temporary Fernet-compatible key.
    
    WARNING: This key is not persisted and data encrypted with it
    will be unrecoverable after restart. For production, always
    set DISTILLATION_ENCRYPTION_KEY environment variable.
    """
    import base64
    key_bytes = secrets.token_bytes(32)
    return base64.urlsafe_b64encode(key_bytes).decode('ascii')


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config(config: Dict[str, Any], strict: bool = True) -> bool:
    """
    Validate configuration settings for security and correctness.
    
    Args:
        config: Configuration dictionary to validate
        strict: If True, enforces production security requirements
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
        SecurityWarning: If security settings are inadequate (strict mode)
        
    Example:
        >>> config = get_config("production")
        >>> validate_config(config)
        True
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # Required keys
    required_keys = [
        "storage_dir",
        "require_opt_in",
        "enable_pii_redaction",
        "executor_mode",
    ]
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")
    
    # Validate executor mode
    if config.get("executor_mode") not in VALID_EXECUTOR_MODES:
        errors.append(
            f"Invalid executor_mode: {config.get('executor_mode')}. "
            f"Must be one of: {sorted(VALID_EXECUTOR_MODES)}"
        )
    
    # Validate quality thresholds
    quality_score = config.get("min_quality_score", 0.65)
    if not isinstance(quality_score, (int, float)) or not 0.0 <= quality_score <= 1.0:
        errors.append("min_quality_score must be a number between 0.0 and 1.0")
    
    # Validate retention days
    retention = config.get("retention_days", DEFAULT_RETENTION_DAYS)
    if not isinstance(retention, int) or not MIN_RETENTION_DAYS <= retention <= MAX_RETENTION_DAYS:
        errors.append(
            f"retention_days must be an integer between "
            f"{MIN_RETENTION_DAYS} and {MAX_RETENTION_DAYS}"
        )
    
    # Security validations (strict mode)
    if strict:
        if not config.get("require_opt_in"):
            warnings.append(
                "SECURITY: require_opt_in is False. User consent should be "
                "required for data collection in production."
            )
        
        if not config.get("enable_pii_redaction"):
            errors.append(
                "SECURITY: enable_pii_redaction must be True. "
                "PII redaction is required for compliance."
            )
        
        if not config.get("use_encryption"):
            warnings.append(
                "SECURITY: use_encryption is False. Encryption is strongly "
                "recommended for protecting training data at rest."
            )
        
        if config.get("use_encryption") and not config.get("encryption_key"):
            warnings.append(
                "SECURITY: use_encryption is True but encryption_key is not set. "
                "Set DISTILLATION_ENCRYPTION_KEY environment variable."
            )
    
    # Log warnings
    for warning in warnings:
        logger.warning(warning)
    
    # Raise on errors
    if errors:
        raise ValueError(
            f"Configuration validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    
    return True


def get_retention_config(
    compliance_framework: str = "default"
) -> Dict[str, Any]:
    """
    Get retention configuration for specific compliance frameworks.
    
    Args:
        compliance_framework: One of "default", "gdpr", "ccpa", "hipaa"
        
    Returns:
        Dictionary with retention-related settings
        
    Example:
        >>> retention = get_retention_config("gdpr")
        >>> print(retention["retention_days"])
        30
    """
    frameworks = {
        "default": {
            "retention_days": 30,
            "require_opt_in": True,
            "enable_deletion_requests": True,
        },
        "gdpr": {
            "retention_days": 30,  # Data minimization principle
            "require_opt_in": True,  # Explicit consent required
            "enable_deletion_requests": True,  # Right to erasure
            "anonymize_logs": True,
        },
        "ccpa": {
            "retention_days": 45,  # California requirements
            "require_opt_in": True,
            "enable_deletion_requests": True,
            "enable_opt_out": True,  # Right to opt out of sale
        },
        "hipaa": {
            "retention_days": 180,  # Healthcare data retention
            "require_opt_in": True,
            "use_encryption": True,  # Required for PHI
            "enable_audit_trail": True,
        },
    }
    return frameworks.get(compliance_framework, frameworks["default"])
