"""
VULCAN Security Module

Provides security components for the VULCAN system including:
- Pre-LLM security filtering (prompt injection, jailbreak detection)
- Zero-knowledge proof integration
- Security auditing
- Access control
"""

import logging

logger = logging.getLogger(__name__)

# Import security filter
try:
    from .filter import SecurityFilter, SecurityResult, RiskLevel, ThreatCategory
    SECURITY_FILTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Security filter not available: {e}")
    SECURITY_FILTER_AVAILABLE = False
    SecurityFilter = None
    SecurityResult = None
    RiskLevel = None
    ThreatCategory = None

# Try to import ZK integration
try:
    from .zk_integration import ZKVerifier, ZK_AVAILABLE, create_verifier
except ImportError as e:
    logger.warning(f"ZK integration not available: {e}")
    ZK_AVAILABLE = False
    ZKVerifier = None
    create_verifier = None

__all__ = [
    # Security filter
    "SecurityFilter",
    "SecurityResult",
    "RiskLevel",
    "ThreatCategory",
    "SECURITY_FILTER_AVAILABLE",
    # ZK integration
    "ZKVerifier",
    "ZK_AVAILABLE",
    "create_verifier",
]

logger.info("VULCAN security module initialized")
