# ============================================================
# VULCAN Execution Package
# LLM execution and dynamic architecture components
# ============================================================
#
# MODULES:
#     llm_executor        - Main LLM execution engine
#     dynamic_architecture - Dynamic model architecture adaptation
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# LLM Executor
try:
    from src.execution.llm_executor import LLMExecutor
    LLM_EXECUTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLMExecutor not available: {e}")
    LLM_EXECUTOR_AVAILABLE = False
    LLMExecutor = None

# Dynamic Architecture
try:
    from src.execution.dynamic_architecture import DynamicArchitecture
    DYNAMIC_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DynamicArchitecture not available: {e}")
    DYNAMIC_ARCHITECTURE_AVAILABLE = False
    DynamicArchitecture = None

__all__ = [
    "__version__",
    "__author__",
    # Executor
    "LLMExecutor",
    "LLM_EXECUTOR_AVAILABLE",
    # Dynamic
    "DynamicArchitecture",
    "DYNAMIC_ARCHITECTURE_AVAILABLE",
]

logger.debug(f"VULCAN Execution package v{__version__} loaded")
