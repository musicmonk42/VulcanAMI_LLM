"""Environment classification for Vulcan AMI.

Single source of truth for environment-based security gates.
Uses allowlist — only explicitly listed values are treated as dev.
"""
import os

_DEV_ENVS = frozenset({"development", "test"})


def is_dev_env() -> bool:
    """Return True only for explicitly allowed dev environments."""
    return os.getenv("VULCAN_ENV") in _DEV_ENVS


def get_env() -> str:
    """Return the current environment name, defaulting to production."""
    return os.getenv("VULCAN_ENV", "production")
