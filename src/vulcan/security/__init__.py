"""
VULCAN Security Module

Provides security components for the VULCAN system including:
- Zero-knowledge proof integration
- Security auditing
- Access control
"""

import logging

logger = logging.getLogger(__name__)

# Try to import ZK integration
try:
    from .zk_integration import ZKVerifier, ZK_AVAILABLE, create_verifier
except ImportError as e:
    logger.warning(f"ZK integration not available: {e}")
    ZK_AVAILABLE = False
    ZKVerifier = None
    create_verifier = None

__all__ = [
    "ZKVerifier",
    "ZK_AVAILABLE",
    "create_verifier",
]

logger.info("VULCAN security module initialized")
