"""
GVulcan Package - Data quality and policy utilities.

This package provides specialized utilities for data quality scoring and policy
enforcement. For storage components (Merkle trees, Bloom filters, LSM), use
persistant_memory_v46 instead.

Unique Components:
    - DQS (Data Quality Score): Multi-dimensional quality scoring
    - OPA (Open Policy Agent): Policy-as-code enforcement
    - Configuration: Centralized configuration management

Deprecated Components:
    - BloomFilter: Use src.persistant_memory_v46.lsm.BloomFilter
    - MerkleLSMDAG: Use src.persistant_memory_v46.lsm.MerkleLSM

For backwards compatibility, deprecated components are re-exported from
persistant_memory_v46 with deprecation warnings.

Example:
    Using DQS for data quality validation:
    
    >>> from src.gvulcan import DQSScorer, DQSComponents
    >>> scorer = DQSScorer(model="v2", reject_below=0.3)
    >>> components = DQSComponents(
    ...     pii_confidence=0.05,
    ...     graph_completeness=0.95,
    ...     syntactic_completeness=0.98
    ... )
    >>> result = scorer.score(components)
    >>> print(f"Score: {result.score}, Decision: {result.gate_decision}")

Author: VULCAN-AGI Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# VERSION MANAGEMENT
# ============================================================================


def _read_semver() -> str:
    """Read semantic version from config file."""
    try:
        p = Path(__file__).resolve().parents[2] / "configs" / "packer" / "semver.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug(f"Could not read semver: {e}")
    return "0.1.0"


__version__ = _read_semver()


# ============================================================================
# STORAGE COMPONENTS (DEPRECATED - USE persistant_memory_v46)
# ============================================================================

_PERSISTANT_MEMORY_AVAILABLE = False

try:
    from src.persistant_memory_v46.lsm import BloomFilter as _BloomFilter
    from src.persistant_memory_v46.lsm import MerkleLSM as _MerkleLSM
    _PERSISTANT_MEMORY_AVAILABLE = True
    logger.debug("persistant_memory_v46 available for backwards compatibility")
    
    # Wrapper classes that issue deprecation warnings
    class BloomFilter(_BloomFilter):
        """
        Deprecated: BloomFilter from gvulcan is deprecated.
        
        Use: from src.persistant_memory_v46.lsm import BloomFilter
        
        This wrapper provides backwards compatibility but will be removed
        in a future version.
        """
        
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "Importing BloomFilter from gvulcan is deprecated. "
                "Use: from src.persistant_memory_v46.lsm import BloomFilter",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)
    
    class MerkleLSMDAG:
        """
        Deprecated: MerkleLSMDAG from gvulcan is deprecated.
        
        Use: from src.persistant_memory_v46.lsm import MerkleLSM
        
        This wrapper provides backwards compatibility but will be removed
        in a future version.
        """
        
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "MerkleLSMDAG from gvulcan is deprecated. "
                "Use: from src.persistant_memory_v46.lsm import MerkleLSM",
                DeprecationWarning,
                stacklevel=2
            )
            self._lsm = _MerkleLSM(*args, **kwargs)
        
        def __getattr__(self, name: str):
            """Delegate all attribute access to wrapped MerkleLSM."""
            return getattr(self._lsm, name)
    
    logger.info("Deprecated storage components available (BloomFilter, MerkleLSMDAG)")
            
except ImportError as e:
    # Fall back to local implementations if persistant_memory_v46 not available
    logger.debug(f"persistant_memory_v46 not available: {e}")
    try:
        from .merkle import MerkleLSMDAG, MerkleTree
        from .bloom import BloomFilter, CountingBloomFilter, ScalableBloomFilter
        logger.info("Using local implementations of storage components")
    except ImportError as e:
        logger.warning(f"Local storage implementations not available: {e}")
        MerkleLSMDAG = None
        MerkleTree = None
        BloomFilter = None


# ============================================================================
# UNIQUE GVULCAN COMPONENTS
# ============================================================================

# Data Quality Scoring (unique to gvulcan)
try:
    from .dqs import DQSScorer, DQSComponents, DQSResult, compute_dqs, DQSTracker
    DQS_AVAILABLE = True
    logger.info("DQS (Data Quality Score) available")
except ImportError as e:
    logger.debug(f"DQS not available: {e}")
    DQS_AVAILABLE = False
    DQSScorer = None
    DQSComponents = None
    DQSResult = None
    compute_dqs = None
    DQSTracker = None

# Open Policy Agent integration (unique to gvulcan)
try:
    from .opa import OPAClient, WriteBarrierInput, WriteBarrierResult, PolicyRegistry
    OPA_AVAILABLE = True
    logger.info("OPA (Open Policy Agent) available")
except ImportError as e:
    logger.debug(f"OPA not available: {e}")
    OPA_AVAILABLE = False
    OPAClient = None
    WriteBarrierInput = None
    WriteBarrierResult = None
    PolicyRegistry = None

# Configuration management (unique to gvulcan)
try:
    from .config import GVulcanConfig, ConfigurationManager, ConfigurationFactory
    CONFIG_AVAILABLE = True
    logger.info("GVulcan configuration management available")
except ImportError as e:
    logger.debug(f"Config not available: {e}")
    CONFIG_AVAILABLE = False
    GVulcanConfig = None
    ConfigurationManager = None
    ConfigurationFactory = None

# Additional modules (if they exist)
try:
    from . import merkle, zk
except ImportError:
    merkle = None
    zk = None


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "__version__",
    # Modules
    "merkle",
    "zk",
    # Data Quality (unique to gvulcan)
    "DQSScorer",
    "DQSComponents",
    "DQSResult",
    "compute_dqs",
    "DQSTracker",
    "DQS_AVAILABLE",
    # Policy (unique to gvulcan)
    "OPAClient",
    "WriteBarrierInput",
    "WriteBarrierResult",
    "PolicyRegistry",
    "OPA_AVAILABLE",
    # Config (unique to gvulcan)
    "GVulcanConfig",
    "ConfigurationManager",
    "ConfigurationFactory",
    "CONFIG_AVAILABLE",
    # Deprecated (use persistant_memory_v46 instead)
    "BloomFilter",
    "MerkleLSMDAG",
]

logger.debug(f"GVulcan package v{__version__} loaded")

