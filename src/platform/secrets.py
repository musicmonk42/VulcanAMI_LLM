"""Secrets management for the Vulcan unified platform.

Provides a unified SecretsManager supporting environment variables,
.env files, and AWS Secrets Manager backends.
"""

import os
from typing import Optional


class SecretsManager:
    """
    Unified secrets management supporting multiple backends.
    Supports: environment variables, .env files, AWS Secrets Manager, etc.
    """

    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment with fallback support."""
        # Try direct env var
        value = os.environ.get(key)
        if value:
            return value

        # Try with UNIFIED_ prefix
        value = os.environ.get(f"UNIFIED_{key}")
        if value:
            return value

        # Try AWS Secrets Manager (if boto3 available)
        try:
            pass

            import boto3

            secrets_client = boto3.client("secretsmanager")
            secret_value = secrets_client.get_secret_value(SecretId=key)

            if "SecretString" in secret_value:
                return secret_value["SecretString"]
        except Exception:
            # Silently ignore AWS Secrets Manager errors - logger not available yet
            pass
        return default


# Module-level singleton instance
secrets = SecretsManager()
