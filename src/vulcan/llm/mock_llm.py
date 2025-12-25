# ============================================================
# VULCAN-AGI Mock LLM Module
# Mock implementation of GraphixVulcanLLM for safe execution
# ============================================================
#
# This module provides a mock LLM implementation that can be used
# when the real GraphixVulcanLLM package is not available.
#
# USAGE:
#     from vulcan.llm.mock_llm import MockGraphixVulcanLLM, GraphixVulcanLLM
#     
#     # GraphixVulcanLLM will be the mock if real package unavailable
#     llm = GraphixVulcanLLM("configs/llm_config.yaml")
#     response = llm.generate("Hello, world!", max_tokens=100)
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import logging
from typing import Optional
from unittest.mock import MagicMock

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# MOCK LLM IMPLEMENTATION
# ============================================================


class MockGraphixVulcanLLM:
    """
    Mock implementation of GraphixVulcanLLM for safe execution.
    
    This mock provides a fully functional interface that mimics
    the real GraphixVulcanLLM, allowing the system to run even
    when the real LLM package is not available.
    
    Attributes:
        config_path: Path to the configuration file
        bridge: Mock bridge object supporting reasoning and world_model calls
        logger: Logger instance for this mock
        
    Example:
        >>> llm = MockGraphixVulcanLLM("configs/llm_config.yaml")
        >>> response = llm.generate("What is 2+2?", max_tokens=100)
        >>> print(response)
        "Mock response to: What is 2+2?"
    """

    def __init__(self, config_path: str):
        """
        Initialize the mock LLM.
        
        Args:
            config_path: Path to the configuration file (used for compatibility)
        """
        self.config_path = config_path
        self.logger = logging.getLogger("MockLLM")
        self.logger.info(f"Initialized mock LLM with config: {config_path}")

        # Mock bridge structure to support reasoning and world_model calls
        self.bridge = MagicMock()
        self.bridge.reasoning.reason.return_value = "Mocked LLM Reasoning Result"
        self.bridge.world_model.explain.return_value = "Mocked LLM Explanation"
        
        # Track generation statistics
        self._generation_count = 0
        self._total_tokens_requested = 0

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Simulate text generation.
        
        Args:
            prompt: The input prompt to respond to
            max_tokens: Maximum number of tokens to generate (used for compatibility)
            
        Returns:
            A mock response string
        """
        self._generation_count += 1
        self._total_tokens_requested += max_tokens
        
        # Log with truncated prompt for privacy
        prompt_preview = prompt[:30] + "..." if len(prompt) > 30 else prompt
        self.logger.info(
            f"Generating response for prompt: '{prompt_preview}' (max_tokens: {max_tokens})"
        )
        
        # Return a mock response
        response_preview = prompt[:50] if len(prompt) <= 50 else prompt[:50] + "..."
        return f"Mock response to: {response_preview}"
    
    def reason(self, query: str, context: Optional[dict] = None) -> str:
        """
        Mock reasoning operation.
        
        Args:
            query: The query to reason about
            context: Optional context dictionary
            
        Returns:
            Mock reasoning result
        """
        return self.bridge.reasoning.reason(query, context)
    
    def explain(self, concept: str) -> str:
        """
        Mock explanation operation.
        
        Args:
            concept: The concept to explain
            
        Returns:
            Mock explanation
        """
        return self.bridge.world_model.explain(concept)
    
    def get_stats(self) -> dict:
        """
        Get generation statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        return {
            "generation_count": self._generation_count,
            "total_tokens_requested": self._total_tokens_requested,
            "is_mock": True,
        }
    
    def __repr__(self) -> str:
        return f"MockGraphixVulcanLLM(config_path={self.config_path!r})"


# ============================================================
# GRAPHIX LLM IMPORT WITH FALLBACK
# ============================================================

GRAPHIX_LLM_AVAILABLE = False
GraphixVulcanLLM = MockGraphixVulcanLLM  # Default to mock

try:
    from graphix_vulcan_llm import GraphixVulcanLLM as _RealGraphixVulcanLLM

    # Guard against partial or bad installation by testing if it's usable
    try:
        # Quick instantiation test
        _test_instance = _RealGraphixVulcanLLM("configs/llm_config.yaml")
        # If we get here, the real implementation works
        GraphixVulcanLLM = _RealGraphixVulcanLLM
        GRAPHIX_LLM_AVAILABLE = True
        logger.info("Real GraphixVulcanLLM loaded successfully")
    except Exception as e:
        logger.warning(f"GraphixVulcanLLM available but not usable: {e}")
        logger.info("Falling back to MockGraphixVulcanLLM")
        GraphixVulcanLLM = MockGraphixVulcanLLM
        GRAPHIX_LLM_AVAILABLE = False
        
except ImportError:
    logger.info("GraphixVulcanLLM package not installed - using MockGraphixVulcanLLM")
    GraphixVulcanLLM = MockGraphixVulcanLLM
    GRAPHIX_LLM_AVAILABLE = False


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "MockGraphixVulcanLLM",
    "GraphixVulcanLLM",
    "GRAPHIX_LLM_AVAILABLE",
]


# Log module status
if GRAPHIX_LLM_AVAILABLE:
    logger.debug(f"Mock LLM module v{__version__} loaded (real GraphixVulcanLLM available)")
else:
    logger.debug(f"Mock LLM module v{__version__} loaded (using mock implementation)")
