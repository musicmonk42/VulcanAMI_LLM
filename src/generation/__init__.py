# ============================================================
# VULCAN Generation Package
# Text generation components with safety and explainability
# ============================================================
#
# MODULES:
#     unified_generation     - Unified generation interface
#     safe_generation        - Safety-filtered generation
#     explainable_generation - Explainable generation with reasoning traces
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Unified Generation
try:
    from src.generation.unified_generation import UnifiedGeneration
    UNIFIED_GENERATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"UnifiedGeneration not available: {e}")
    UNIFIED_GENERATION_AVAILABLE = False
    UnifiedGeneration = None

# Safe Generation
try:
    from src.generation.safe_generation import SafeGeneration
    SAFE_GENERATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SafeGeneration not available: {e}")
    SAFE_GENERATION_AVAILABLE = False
    SafeGeneration = None

# Explainable Generation
try:
    from src.generation.explainable_generation import ExplainableGeneration
    EXPLAINABLE_GENERATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ExplainableGeneration not available: {e}")
    EXPLAINABLE_GENERATION_AVAILABLE = False
    ExplainableGeneration = None

__all__ = [
    "__version__",
    "__author__",
    # Unified
    "UnifiedGeneration",
    "UNIFIED_GENERATION_AVAILABLE",
    # Safe
    "SafeGeneration",
    "SAFE_GENERATION_AVAILABLE",
    # Explainable
    "ExplainableGeneration",
    "EXPLAINABLE_GENERATION_AVAILABLE",
]

logger.debug(f"VULCAN Generation package v{__version__} loaded")
