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
from typing import Optional, Any
from unittest.mock import MagicMock

# Module metadata
__version__ = "1.1.0"  # P1 FIX: Transparent mock mode with echo/logging
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# ============================================================
# MOCK DETECTION UTILITIES
# ============================================================

def is_mock_llm(llm: Any) -> bool:
    """
    Check if the given LLM instance is a mock.
    
    P1 FIX: This utility function allows callers to detect mock mode
    and handle it appropriately (e.g., use rule-based formatting instead
    of expecting real language generation).
    
    Args:
        llm: The LLM instance to check
        
    Returns:
        True if the LLM is a mock instance, False otherwise
        
    Example:
        >>> from vulcan.llm.mock_llm import is_mock_llm
        >>> if is_mock_llm(executor.local_llm):
        ...     logger.warning("Using mock LLM - responses will be template-based")
    """
    if llm is None:
        return False
    
    # Check by class name (works across module boundaries)
    class_name = type(llm).__name__
    if class_name == "MockGraphixVulcanLLM":
        return True
    
    # Check for mock marker attribute
    if hasattr(llm, '_is_mock') and llm._is_mock:
        return True
    
    # Check get_stats for mock indicator
    if hasattr(llm, 'get_stats'):
        try:
            stats = llm.get_stats()
            if isinstance(stats, dict) and stats.get('is_mock', False):
                return True
        except Exception:
            pass
    
    return False


# ============================================================
# MOCK RESPONSE DETECTION
# ============================================================

MOCK_RESPONSE_PREFIX = "MOCK_LLM:"


def is_mock_response(response: str) -> bool:
    """
    Check if a response string is from the mock LLM.
    
    P1 FIX: Allows callers to detect mock responses and handle them
    appropriately in the output pipeline.
    
    Args:
        response: The response string to check
        
    Returns:
        True if the response is from mock LLM, False otherwise
    """
    if not response:
        return False
    return response.strip().startswith(MOCK_RESPONSE_PREFIX)


# ============================================================
# MOCK LLM IMPLEMENTATION
# ============================================================


class MockGraphixVulcanLLM:
    """
    Mock implementation of GraphixVulcanLLM for safe execution.
    
    This mock provides a fully functional interface that mimics
    the real GraphixVulcanLLM, allowing the system to run even
    when the real LLM package is not available.
    
    NOTE: The internal LLM (including this mock) is for LANGUAGE GENERATION only,
    not reasoning. VULCAN's reasoning systems (symbolic, probabilistic, causal,
    mathematical) do ALL the thinking. This LLM converts structured reasoning
    outputs to natural language prose.
    
    Attributes:
        config_path: Path to the configuration file
        bridge: Mock bridge object supporting reasoning and world_model calls
        logger: Logger instance for this mock
        
    Example:
        >>> llm = MockGraphixVulcanLLM("configs/llm_config.yaml")
        >>> response = llm.generate("What is machine learning?", max_tokens=100)
        >>> print(response)  # Will produce a meaningful explanation, not "Mock response"
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the mock LLM.
        
        NOTE: The internal LLM is for LANGUAGE GENERATION only, not reasoning.
        VULCAN's reasoning systems (symbolic, probabilistic, causal, mathematical)
        do ALL the thinking. This mock provides template-based language output
        when the real GraphixVulcanLLM is not available.
        
        Args:
            config_path: Path to the configuration file (used for compatibility)
        """
        self.config_path = config_path or "configs/llm_config.yaml"
        self.logger = logging.getLogger("MockLLM")
        self.logger.info(f"Initialized mock LLM with config: {self.config_path}")
        self.logger.info("NOTE: Using fallback mock LLM for language generation (not reasoning)")

        # Mock bridge structure to support reasoning and world_model calls
        # These return meaningful template responses, not "Mocked" strings
        self.bridge = MagicMock()
        self.bridge.reasoning.reason.return_value = (
            "Analysis complete: The reasoning system has processed the query "
            "and identified the key logical relationships and implications."
        )
        self.bridge.world_model.explain.return_value = (
            "The concept has been analyzed through the world model, revealing "
            "its structural properties and relationships to other elements."
        )
        
        # Track generation statistics
        self._generation_count = 0
        self._total_tokens_requested = 0

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Simulate text generation for language output formatting.
        
        P1 FIX: Mock LLM is now TRANSPARENT and NON-BLOCKING.
        
        The mock implementation now:
        1. Clearly indicates it's a mock response with "MOCK_LLM:" prefix
        2. Echoes back the input prompt for debugging/testing visibility
        3. Returns structured JSON-like output for easier testing
        4. Logs a WARNING whenever used (for production alerting)
        
        NOTE: The internal LLM is for LANGUAGE GENERATION only, not reasoning.
        VULCAN's reasoning systems (symbolic, probabilistic, causal, mathematical)
        do ALL the thinking BEFORE this LLM is called.
        
        Args:
            prompt: The input prompt to respond to (may include reasoning context)
            max_tokens: Maximum number of tokens to generate (used for compatibility)
            
        Returns:
            A mock response string clearly marked as mock output
        """
        self._generation_count += 1
        self._total_tokens_requested += max_tokens
        
        # P1 FIX: Log WARNING whenever mock is used for total transparency
        # This ensures developers/operators know when mock is being used in production
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        self.logger.warning(
            f"[MOCK_LLM] ⚠️ Mock LLM generating response - real GraphixVulcanLLM not available. "
            f"Prompt: '{prompt_preview}' (max_tokens: {max_tokens})"
        )
        
        # P1 FIX: Return transparent mock response that:
        # 1. Is clearly marked as mock (MOCK_LLM prefix)
        # 2. Echoes the input for debugging
        # 3. Provides structured output for testing
        response = self._generate_transparent_mock_response(prompt, max_tokens)
        
        return response
    
    def _generate_transparent_mock_response(self, prompt: str, max_tokens: int) -> str:
        """
        Generate a transparent mock response that clearly indicates mock status.
        
        P1 FIX: This response is designed to:
        1. Be instantly recognizable as mock output (not mistaken for real LLM)
        2. Echo back input for debugging/testing
        3. Provide structured data for test assertions
        4. Never mislead users or developers
        
        Args:
            prompt: The input prompt
            max_tokens: Token limit (included in response for visibility)
            
        Returns:
            Transparent mock response string
        """
        import json
        
        # Create structured mock response
        mock_data = {
            "mock_llm": True,
            "warning": "This is a mock response - real GraphixVulcanLLM not available",
            "input_echo": {
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "prompt_length": len(prompt),
                "max_tokens_requested": max_tokens,
            },
            "generation_stats": {
                "generation_count": self._generation_count,
                "total_tokens_requested": self._total_tokens_requested,
            },
            "message": (
                "VULCAN's reasoning systems completed their work. "
                "This mock LLM formatted the output. "
                "Install the real GraphixVulcanLLM for production use."
            ),
        }
        
        # Return clearly-marked mock response
        return f"MOCK_LLM: {json.dumps(mock_data, indent=2)}"
    
    def generate_legacy(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Legacy generation method that produces template-based responses.
        
        DEPRECATED: This method is kept for backwards compatibility but should
        not be used. Use generate() instead which provides transparent mock output.
        
        Args:
            prompt: The input prompt to respond to
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            A template-based response string
        """
        import warnings
        warnings.warn(
            "generate_legacy() is deprecated. Use generate() for transparent mock output.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Log with truncated prompt for privacy
        prompt_preview = prompt[:30] + "..." if len(prompt) > 30 else prompt
        self.logger.info(
            f"[LEGACY] Generating response for prompt: '{prompt_preview}' (max_tokens: {max_tokens})"
        )
        
        # Generate a meaningful response based on the prompt content
        prompt_lower = prompt.lower().strip()
        
        # Handle common query types with appropriate template responses
        if any(q in prompt_lower for q in ["what is", "what are", "define", "explain"]):
            response = self._generate_explanation_response(prompt)
        elif any(q in prompt_lower for q in ["how to", "how do", "how can"]):
            response = self._generate_howto_response(prompt)
        elif any(q in prompt_lower for q in ["why", "reason", "cause"]):
            response = self._generate_reasoning_response(prompt)
        elif any(op in prompt_lower for op in ["+", "-", "*", "/", "calculate", "compute", "math", "sum", "multiply"]):
            response = self._generate_math_response(prompt)
        elif "?" in prompt:
            response = self._generate_question_response(prompt)
        else:
            response = self._generate_general_response(prompt)
        
        return response
    
    def _generate_explanation_response(self, prompt: str) -> str:
        """Generate an explanation-style response."""
        topic = self._extract_topic(prompt)
        return (
            f"Based on my analysis, {topic} can be understood as follows:\n\n"
            f"This concept relates to the fundamental aspects of the subject matter. "
            f"The key points to consider are the underlying principles and their applications. "
            f"For a more detailed explanation, additional context about the specific aspects "
            f"you're interested in would be helpful."
        )
    
    def _generate_howto_response(self, prompt: str) -> str:
        """Generate a how-to style response."""
        topic = self._extract_topic(prompt)
        return (
            f"Here's a general approach for {topic}:\n\n"
            f"1. First, identify your specific requirements and constraints\n"
            f"2. Consider the available resources and tools\n"
            f"3. Plan your approach step by step\n"
            f"4. Execute the plan while monitoring progress\n"
            f"5. Review and adjust as needed\n\n"
            f"The specific steps may vary depending on your exact situation."
        )
    
    def _generate_reasoning_response(self, prompt: str) -> str:
        """Generate a reasoning-style response."""
        topic = self._extract_topic(prompt)
        return (
            f"The reasoning behind {topic} involves several factors:\n\n"
            f"From a logical perspective, we can analyze this by considering "
            f"the underlying causes and their effects. The relationship between "
            f"these elements helps us understand the broader context and implications."
        )
    
    def _generate_math_response(self, prompt: str) -> str:
        """Generate a math-related response."""
        return (
            "I've processed your mathematical query. The computation involves "
            "applying the relevant mathematical operations to the given values. "
            "For precise calculations, please ensure all numerical inputs are correctly specified."
        )
    
    def _generate_question_response(self, prompt: str) -> str:
        """Generate a response to a question."""
        topic = self._extract_topic(prompt)
        return (
            f"Regarding your question about {topic}:\n\n"
            f"This is a thoughtful inquiry that touches on important aspects of the subject. "
            f"The answer depends on the specific context and requirements involved. "
            f"I'd be happy to provide more detailed information if you can clarify "
            f"which aspects you're most interested in."
        )
    
    def _generate_general_response(self, prompt: str) -> str:
        """Generate a general-purpose response."""
        return (
            "I've processed your request and analyzed the input provided. "
            "Based on the information available, I can offer the following insights:\n\n"
            "The topic you've raised involves considerations that span multiple areas. "
            "A comprehensive understanding requires examining both the immediate context "
            "and the broader implications. Let me know if you'd like me to focus on "
            "any specific aspect in more detail."
        )
    
    def _extract_topic(self, prompt: str) -> str:
        """Extract the main topic from a prompt for response generation."""
        # Remove common question words and extract the core topic
        prompt_clean = prompt.strip()
        
        # Remove leading question words
        for prefix in ["what is ", "what are ", "how to ", "how do ", "how can ", 
                       "why ", "define ", "explain ", "tell me about "]:
            if prompt_clean.lower().startswith(prefix):
                prompt_clean = prompt_clean[len(prefix):]
                break
        
        # Remove trailing punctuation
        prompt_clean = prompt_clean.rstrip("?!.")
        
        # Truncate if too long
        if len(prompt_clean) > 50:
            prompt_clean = prompt_clean[:50] + "..."
        
        return prompt_clean if prompt_clean else "this topic"
    
    def reason(self, query: str, context: Optional[dict] = None) -> str:
        """
        Mock reasoning operation.
        
        NOTE: The internal LLM does NOT do reasoning - VULCAN's reasoning systems
        (symbolic, probabilistic, causal, mathematical) do ALL the thinking.
        This method is for language generation only.
        
        Args:
            query: The query to reason about
            context: Optional context dictionary
            
        Returns:
            A structured reasoning result as text
        """
        topic = self._extract_topic(query)
        return (
            f"Reasoning analysis for: {topic}\n\n"
            f"Based on systematic analysis, the key considerations are:\n"
            f"1. The logical structure of the problem\n"
            f"2. The relationships between components\n"
            f"3. The implications of the given constraints\n\n"
            f"Conclusion: The analysis indicates that a comprehensive approach "
            f"considering all relevant factors is recommended."
        )
    
    def explain(self, concept: str) -> str:
        """
        Mock explanation operation.
        
        NOTE: The internal LLM does NOT do reasoning - it generates language output.
        VULCAN's reasoning systems do the actual thinking.
        
        Args:
            concept: The concept to explain
            
        Returns:
            An explanation of the concept
        """
        return (
            f"Explanation of {concept}:\n\n"
            f"This concept represents an important element in understanding the broader context. "
            f"The key aspects to consider include its fundamental properties, "
            f"how it relates to other elements, and its practical applications. "
            f"A thorough understanding requires examining both theoretical foundations "
            f"and real-world implications."
        )
    
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
    # P1 FIX: Mock detection utilities
    "is_mock_llm",
    "is_mock_response",
    "MOCK_RESPONSE_PREFIX",
]


# Log module status
if GRAPHIX_LLM_AVAILABLE:
    logger.debug(f"Mock LLM module v{__version__} loaded (real GraphixVulcanLLM available)")
else:
    logger.warning(
        f"Mock LLM module v{__version__} loaded - USING MOCK IMPLEMENTATION. "
        f"Install graphix_vulcan_llm package for production use."
    )
