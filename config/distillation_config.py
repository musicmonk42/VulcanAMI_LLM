"""
VULCAN Distillation Configuration
Centralized settings for knowledge distillation system.

This module provides environment-specific configurations for the distillation
system, enabling consistent settings across initialization, execution, and
training components.

Usage:
    from config.distillation_config import get_config
    
    # Get production configuration
    config = get_config("production")
    
    # Get specific settings
    storage_path = config["storage_path"]
"""

from typing import Any, Dict

# ============================================================
# BASE CONFIGURATION
# ============================================================

DISTILLATION_CONFIG: Dict[str, Any] = {
    # Storage Settings
    "storage_path": "data/distillation_examples.jsonl",
    "storage_dir": "data/distillation",
    "max_buffer_size": 100,           # Flush to disk every N examples
    "retention_days": 30,             # Keep examples for N days

    # Privacy & Security (CRITICAL - Do not disable in production)
    "require_opt_in": True,           # Require user consent
    "enable_pii_redaction": True,     # Redact PII automatically
    "enable_governance_check": True,  # Check sensitive content
    "use_encryption": False,          # Enable in production
    "encryption_key": None,           # Set your Fernet key in production

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
}

# ============================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ============================================================

PRODUCTION_CONFIG: Dict[str, Any] = {
    **DISTILLATION_CONFIG,
    "require_opt_in": True,
    "use_encryption": True,
    "min_quality_score": 0.70,
    "enable_pii_redaction": True,
    "enable_governance_check": True,
}

DEVELOPMENT_CONFIG: Dict[str, Any] = {
    **DISTILLATION_CONFIG,
    "require_opt_in": False,          # For testing only
    "min_quality_score": 0.50,
    "max_buffer_size": 20,            # Smaller buffer for faster testing
    "use_encryption": False,
}

# Ensemble mode specific settings (recommended for knowledge distillation)
ENSEMBLE_CONFIG: Dict[str, Any] = {
    **DISTILLATION_CONFIG,
    "executor_mode": "ensemble",
    "combine_responses": True,        # Append local insights to OpenAI response
    "insight_threshold": 0.3,         # Minimum uniqueness for local insights
}


# ============================================================
# CONFIGURATION ACCESSOR
# ============================================================

def get_config(env: str = "production") -> Dict[str, Any]:
    """
    Get configuration for specified environment.
    
    Args:
        env: Environment name (production, development, ensemble)
        
    Returns:
        Dictionary with configuration settings
        
    Example:
        >>> config = get_config("production")
        >>> print(config["executor_mode"])
        'ensemble'
    """
    configs = {
        "production": PRODUCTION_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "ensemble": ENSEMBLE_CONFIG,
    }
    return configs.get(env, DISTILLATION_CONFIG)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = [
        "storage_dir",
        "require_opt_in",
        "enable_pii_redaction",
        "executor_mode",
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate executor mode
    valid_modes = ("ensemble", "local_first", "openai_first", "parallel")
    if config["executor_mode"] not in valid_modes:
        raise ValueError(
            f"Invalid executor_mode: {config['executor_mode']}. "
            f"Must be one of: {valid_modes}"
        )
    
    # Validate quality thresholds
    if not 0.0 <= config.get("min_quality_score", 0.65) <= 1.0:
        raise ValueError("min_quality_score must be between 0.0 and 1.0")
    
    return True
