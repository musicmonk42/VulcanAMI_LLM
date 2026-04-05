"""
self_improvement.py - Disabled external LLM client for the World Model.

Extracted from world_model_core.py to reduce file size and improve modularity.

Contains:
- CodeLLMClient: DISABLED external LLM code generation client.
  OpenAI is NOT permitted for code generation or reasoning.
  The self-improvement pipeline must use VULCAN's internal capabilities only.
"""

import logging

logger = logging.getLogger(__name__)


class CodeLLMClient:
    """
    DISABLED: External LLM code generation is prohibited.

    OpenAI and other external LLMs are NOT permitted for:
    - Code generation (this is reasoning)
    - Code improvement (this is reasoning)
    - Any form of independent analysis

    OpenAI is ONLY permitted for interpreting VULCAN's reasoning into natural language.
    The self-improvement system must use VULCAN's internal capabilities.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.last_tokens_used = 0
        self.model_name = "DISABLED"
        self.client = None

        # Log that external LLM code generation is disabled
        logger.warning(
            "[CodeLLMClient] DISABLED - External LLM code generation is prohibited. "
            "Self-improvement must use VULCAN's internal reasoning capabilities. "
            "OpenAI is ONLY permitted for language interpretation, not code generation."
        )

    def generate_code(self, prompt: str) -> str:
        """
        DISABLED: External LLM code generation is prohibited.

        Raises:
            RuntimeError: Always - external LLM code generation is not permitted.
        """
        raise RuntimeError(
            "[VULCAN Policy] External LLM code generation is DISABLED. "
            "OpenAI and other external LLMs are NOT permitted to generate code. "
            "The self-improvement system must use VULCAN's internal reasoning capabilities. "
            "OpenAI is ONLY permitted for interpreting VULCAN's reasoning into natural language."
        )
