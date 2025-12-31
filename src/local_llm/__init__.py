# ============================================================
# VULCAN Local LLM Package
# Local LLM provider and tokenizer components
# ============================================================
#
# SUBPACKAGES:
#     provider   - Local LLM providers (LocalGPTProvider)
#     tokenizer  - Tokenization utilities
#     scripts    - Utility scripts for local LLM management
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Local GPT Provider
try:
    from src.local_llm.provider.local_gpt_provider import LocalGPTProvider
    LOCAL_GPT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LocalGPTProvider not available: {e}")
    LOCAL_GPT_AVAILABLE = False
    LocalGPTProvider = None

__all__ = [
    "__version__",
    "__author__",
    # Provider
    "LocalGPTProvider",
    "LOCAL_GPT_AVAILABLE",
]

logger.debug(f"VULCAN Local LLM package v{__version__} loaded")
