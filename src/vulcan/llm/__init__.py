# ============================================================
# VULCAN-AGI LLM Package
# LLM integration components for VULCAN-AGI
# ============================================================
#
# MODULES:
#     mock_llm        - Mock LLM implementation for testing/fallback
#     hybrid_executor - Hybrid execution across multiple LLM backends
#     openai_client   - OpenAI API client management
#
# USAGE:
#     from vulcan.llm import HybridLLMExecutor, MockGraphixVulcanLLM
#     from vulcan.llm import get_openai_client, OPENAI_AVAILABLE
#
# VERSION HISTORY:
#     1.0.0 - Initial extraction from main.py
# ============================================================

import logging
import sys
from typing import Any, Dict, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CORE IMPORTS WITH AVAILABILITY TRACKING
# ============================================================

_imports_successful = True

try:
    from vulcan.llm.mock_llm import (
        MockGraphixVulcanLLM,
        GraphixVulcanLLM,
        GRAPHIX_LLM_AVAILABLE,
    )
except ImportError as e:
    logger.warning(f"mock_llm module not available: {e}")
    _imports_successful = False
    MockGraphixVulcanLLM = None
    GraphixVulcanLLM = None
    GRAPHIX_LLM_AVAILABLE = False

try:
    from vulcan.llm.hybrid_executor import (
        HybridLLMExecutor,
        VulcanReasoningOutput,
        get_or_create_hybrid_executor,
        get_hybrid_executor,
        set_hybrid_executor,
        verify_hybrid_executor_setup,
        OPENAI_LANGUAGE_FORMATTING,
        OPENAI_LANGUAGE_POLISH,
    )
except ImportError as e:
    logger.warning(f"hybrid_executor module not available: {e}")
    _imports_successful = False
    HybridLLMExecutor = None
    VulcanReasoningOutput = None
    get_or_create_hybrid_executor = None
    get_hybrid_executor = None
    set_hybrid_executor = None
    verify_hybrid_executor_setup = None
    OPENAI_LANGUAGE_FORMATTING = False
    OPENAI_LANGUAGE_POLISH = False

try:
    from vulcan.llm.openai_client import (
        get_openai_client,
        get_openai_init_error,
        OPENAI_AVAILABLE,
        initialize_openai_client,
        verify_openai_configuration,
        log_openai_status,
    )
except ImportError as e:
    logger.warning(f"openai_client module not available: {e}")
    _imports_successful = False
    get_openai_client = None
    get_openai_init_error = None
    OPENAI_AVAILABLE = False
    initialize_openai_client = None
    verify_openai_configuration = None
    log_openai_status = None


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Mock LLM
    "MockGraphixVulcanLLM",
    "GraphixVulcanLLM",
    "GRAPHIX_LLM_AVAILABLE",
    # Hybrid Executor
    "HybridLLMExecutor",
    "VulcanReasoningOutput",
    "get_or_create_hybrid_executor",
    "get_hybrid_executor",
    "set_hybrid_executor",
    "verify_hybrid_executor_setup",
    # Configuration flags for OpenAI language formatting
    "OPENAI_LANGUAGE_FORMATTING",
    "OPENAI_LANGUAGE_POLISH",
    # OpenAI Client
    "get_openai_client",
    "get_openai_init_error",
    "OPENAI_AVAILABLE",
    "initialize_openai_client",
    "verify_openai_configuration",
    "log_openai_status",
    # Module utilities
    "get_module_info",
    "validate_llm_module",
]


# ============================================================
# MODULE UTILITIES
# ============================================================


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the LLM module.
    
    Returns:
        Dictionary with module information and availability status
    """
    return {
        "version": __version__,
        "author": __author__,
        "imports_successful": _imports_successful,
        "python_version": sys.version,
        "components": {
            "mock_llm": MockGraphixVulcanLLM is not None,
            "hybrid_executor": HybridLLMExecutor is not None,
            "vulcan_reasoning_output": VulcanReasoningOutput is not None,
            "openai_client": get_openai_client is not None,
        },
        "backends": {
            "openai_available": OPENAI_AVAILABLE if OPENAI_AVAILABLE is not None else False,
            "graphix_available": GRAPHIX_LLM_AVAILABLE if GRAPHIX_LLM_AVAILABLE is not None else False,
        },
    }


def validate_llm_module() -> bool:
    """
    Validate that the LLM module is properly loaded.
    
    Returns:
        True if module is functional, False otherwise
    """
    try:
        info = get_module_info()
        
        # At minimum, we need mock_llm and hybrid_executor
        essential = info["components"]["mock_llm"] and info["components"]["hybrid_executor"]
        
        if essential:
            logger.info("LLM module validated successfully")
        else:
            logger.warning("LLM module validation failed - essential components missing")
        
        return essential
        
    except Exception as e:
        logger.error(f"LLM module validation failed: {e}")
        return False


# ============================================================
# MODULE INITIALIZATION
# ============================================================

if _imports_successful:
    logger.info(f"VULCAN-AGI LLM module v{__version__} loaded successfully")
    if OPENAI_AVAILABLE:
        logger.info("  ✓ OpenAI backend available")
    else:
        logger.info("  ✗ OpenAI backend not available")
    if GRAPHIX_LLM_AVAILABLE:
        logger.info("  ✓ Graphix LLM backend available")
    else:
        logger.info("  ✗ Graphix LLM backend not available (using mock)")
else:
    logger.warning(f"VULCAN-AGI LLM module v{__version__} loaded with some components unavailable")
