# ============================================================
# VULCAN Context Package
# Context management for LLM generation
# ============================================================
#
# MODULES:
#     hierarchical_context - Multi-tier memory context (episodic/semantic/procedural)
#     causal_context       - Causal reasoning context selection
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Hierarchical Context
try:
    from src.context.hierarchical_context import HierarchicalContext
    HIERARCHICAL_CONTEXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HierarchicalContext not available: {e}")
    HIERARCHICAL_CONTEXT_AVAILABLE = False
    HierarchicalContext = None

# Causal Context
try:
    from src.context.causal_context import CausalContext
    CAUSAL_CONTEXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CausalContext not available: {e}")
    CAUSAL_CONTEXT_AVAILABLE = False
    CausalContext = None

__all__ = [
    "__version__",
    "__author__",
    # Hierarchical
    "HierarchicalContext",
    "HIERARCHICAL_CONTEXT_AVAILABLE",
    # Causal
    "CausalContext",
    "CAUSAL_CONTEXT_AVAILABLE",
]

logger.debug(f"VULCAN Context package v{__version__} loaded")
