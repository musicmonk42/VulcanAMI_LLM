# ============================================================
# VULCAN-AGI Hybrid LLM Executor Module
# Enables simultaneous use of OpenAI and Vulcan's internal LLM
# ============================================================
#
# This module provides:
#     - Multi-backend LLM execution
#     - Fallback strategies
#     - Parallel and ensemble execution modes
#     - Response quality evaluation
#
# EXECUTION MODES:
#     - local_first: Try Vulcan's local LLM first, fallback to OpenAI
#     - openai_first: Try OpenAI first, fallback to local LLM
#     - parallel: Run both simultaneously, use first successful response
#     - ensemble: Run both, combine/select best response based on quality
#
# USAGE:
#     from vulcan.llm.hybrid_executor import HybridLLMExecutor
#     
#     executor = HybridLLMExecutor(local_llm=my_llm, mode="parallel")
#     result = await executor.execute("What is 2+2?")
#     print(result["text"])
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation and type hints
# ============================================================

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS
# ============================================================

# Import OpenAI client getter (with fallback)
try:
    from vulcan.llm.openai_client import get_openai_client
except ImportError:
    # Fallback definition
    def get_openai_client():
        logger.warning("OpenAI client not available")
        return None

# Import knowledge distiller getter (with fallback)
try:
    from vulcan.distillation import get_knowledge_distiller
except ImportError:
    # Fallback definition
    def get_knowledge_distiller():
        return None


# ============================================================
# HYBRID LLM EXECUTOR CLASS
# ============================================================


class HybridLLMExecutor:
    """
    Executes LLM requests using both OpenAI and Vulcan's local LLM.

    Supports multiple execution modes:
    - local_first: Try Vulcan's local LLM first, fallback to OpenAI
    - openai_first: Try OpenAI first, fallback to local LLM
    - parallel: Run both simultaneously, use first successful response
    - ensemble: Run both, combine/select best response based on quality

    This allows VulcanAMI_LLM to leverage both its native reasoning capabilities
    AND OpenAI's language generation without conflicts.
    
    Attributes:
        local_llm: The local LLM instance
        mode: Execution mode (local_first, openai_first, parallel, ensemble)
        timeout: Timeout for parallel/ensemble execution in seconds
        ensemble_min_confidence: Minimum confidence for ensemble selection
        openai_max_tokens: Maximum tokens for OpenAI API calls
        
    Example:
        >>> executor = HybridLLMExecutor(
        ...     local_llm=my_llm,
        ...     mode="parallel",
        ...     timeout=30.0
        ... )
        >>> result = await executor.execute("Explain quantum computing")
        >>> print(result["text"])
        >>> print(result["source"])  # "local", "openai", "parallel_both", or "ensemble"
    """

    # ============================================================
    # CLASS CONSTANTS
    # ============================================================
    
    # Constants for response quality evaluation
    MIN_MEANINGFUL_LENGTH = 10
    MOCK_RESPONSE_MARKER = "Mock response"
    # Maximum length for local response in ensemble mode
    ENSEMBLE_LOCAL_RESPONSE_MAX_LENGTH = 500
    # Valid execution modes
    VALID_MODES = ("local_first", "openai_first", "parallel", "ensemble")

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        local_llm: Optional[Any] = None,
        openai_client_getter: Optional[Callable] = None,
        mode: str = "parallel",
        timeout: float = 30.0,
        ensemble_min_confidence: float = 0.7,
        openai_max_tokens: int = 2000,  # Increased for diagnostic purposes
    ):
        """
        Initialize the hybrid executor.

        Args:
            local_llm: Vulcan's local LLM instance
            openai_client_getter: Function to get OpenAI client (lazy loading)
            mode: Execution mode (local_first, openai_first, parallel, ensemble)
            timeout: Timeout for parallel/ensemble execution in seconds
            ensemble_min_confidence: Minimum confidence for ensemble selection
            openai_max_tokens: Maximum tokens for OpenAI API calls
        """
        self.local_llm = local_llm
        self.openai_client_getter = openai_client_getter or get_openai_client
        
        # Validate and set mode
        mode_lower = mode.lower()
        if mode_lower not in self.VALID_MODES:
            self.logger = logging.getLogger("HybridLLMExecutor")
            self.logger.warning(
                f"Invalid mode '{mode}', defaulting to 'parallel'. Valid modes: {self.VALID_MODES}"
            )
            mode_lower = "parallel"
        self.mode = mode_lower
        
        self.timeout = timeout
        self.ensemble_min_confidence = ensemble_min_confidence
        self.openai_max_tokens = openai_max_tokens
        self.logger = logging.getLogger("HybridLLMExecutor")
        
        # Statistics tracking
        self._execution_count = 0
        self._local_successes = 0
        self._openai_successes = 0
        self._failures = 0

    # ============================================================
    # MAIN EXECUTION METHOD
    # ============================================================

    # Default system prompt that explicitly allows conversation memory
    # MEMORY FIX: The default prompt now tells the model to remember information from conversation
    DEFAULT_SYSTEM_PROMPT = (
        "You are VULCAN, an advanced AI assistant. "
        "You SHOULD remember and reference information shared earlier in this conversation. "
        "When a user shares their name, location, preferences, or any personal details during this session, "
        "you may recall and use that information naturally in your responses."
    )

    async def execute(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: str = None,
        enable_distillation: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute LLM request using configured mode.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            system_prompt: System prompt for OpenAI (defaults to DEFAULT_SYSTEM_PROMPT if None)
            enable_distillation: Whether to capture responses for knowledge distillation
            conversation_history: Optional list of previous messages in the conversation.
                                 Each message should be a dict with 'role' and 'content' keys.
                                 This enables multi-turn conversation context.

        Returns:
            Dict with 'text', 'source', 'systems_used', and optional 'metadata'
        """
        self._execution_count += 1
        loop = asyncio.get_running_loop()
        
        # Use default system prompt if none provided
        # MEMORY FIX: Default prompt now allows conversation memory
        effective_system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

        if self.mode == "local_first":
            result = await self._execute_local_first(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif self.mode == "openai_first":
            result = await self._execute_openai_first(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif self.mode == "parallel":
            result = await self._execute_parallel(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif self.mode == "ensemble":
            result = await self._execute_ensemble(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        else:
            self.logger.warning(f"Unknown mode '{self.mode}', defaulting to parallel")
            result = await self._execute_parallel(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )

        # Update statistics
        self._update_stats(result)

        # Capture OpenAI responses for knowledge distillation
        if enable_distillation and result.get("source") in ("openai", "parallel_both", "ensemble"):
            self._capture_for_distillation(prompt, result)

        return result

    # ============================================================
    # EXECUTION MODE IMPLEMENTATIONS
    # ============================================================

    async def _execute_local_first(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Try local LLM first, fallback to OpenAI."""
        systems_used = []

        # Try local LLM
        local_result = await self._call_local_llm(loop, prompt, max_tokens)
        if self._is_valid_response(local_result):
            systems_used.append("vulcan_local_llm")
            return {
                "text": local_result,
                "source": "local",
                "systems_used": systems_used,
            }

        # Fallback to OpenAI
        openai_result = await self._call_openai(
            loop, prompt, max_tokens, temperature, system_prompt, conversation_history
        )
        if openai_result:
            systems_used.append("openai_fallback")
            return {
                "text": openai_result,
                "source": "openai",
                "systems_used": systems_used,
            }

        return {"text": "", "source": "none", "systems_used": systems_used}

    async def _execute_openai_first(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Try OpenAI first, fallback to local LLM."""
        systems_used = []

        # Try OpenAI
        openai_result = await self._call_openai(
            loop, prompt, max_tokens, temperature, system_prompt, conversation_history
        )
        if openai_result:
            systems_used.append("openai_llm")
            return {
                "text": openai_result,
                "source": "openai",
                "systems_used": systems_used,
            }

        # Fallback to local
        local_result = await self._call_local_llm(loop, prompt, max_tokens)
        if self._is_valid_response(local_result):
            systems_used.append("vulcan_local_llm_fallback")
            return {
                "text": local_result,
                "source": "local",
                "systems_used": systems_used,
            }

        return {"text": "", "source": "none", "systems_used": systems_used}

    async def _execute_parallel(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Run both LLMs simultaneously, use first successful response."""
        systems_used = []

        async def local_task():
            return await self._call_local_llm(loop, prompt, max_tokens)

        async def openai_task():
            return await self._call_openai(
                loop, prompt, max_tokens, temperature, system_prompt, conversation_history
            )

        # Run both tasks concurrently with timeout
        try:
            tasks = [
                asyncio.create_task(local_task()),
                asyncio.create_task(openai_task()),
            ]

            # Wait for first successful result or all to complete
            done, pending = await asyncio.wait(
                tasks, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED
            )

            results = {"local": None, "openai": None}

            for task in done:
                try:
                    result = task.result()
                    # Determine which task completed
                    if task == tasks[0]:
                        results["local"] = result
                    else:
                        results["openai"] = result
                except Exception as e:
                    self.logger.debug(f"Task failed: {e}")

            # Cancel pending tasks and clean up
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # Ignore any other exceptions from cancelled tasks
                    pass

            # Select the best available result
            local_valid = self._is_valid_response(results["local"])
            openai_valid = results["openai"] is not None and len(
                str(results["openai"]).strip()
            ) > self.MIN_MEANINGFUL_LENGTH

            if local_valid and openai_valid:
                # Both succeeded - prefer OpenAI for language quality
                systems_used.extend(["vulcan_local_llm", "openai_llm"])
                return {
                    "text": results["openai"],
                    "source": "parallel_both",
                    "systems_used": systems_used,
                    "metadata": {
                        "local_response_available": True,
                        "openai_response_available": True,
                        "local_response_preview": str(results["local"])[:100],
                    },
                }
            elif openai_valid:
                systems_used.append("openai_llm")
                return {
                    "text": results["openai"],
                    "source": "openai",
                    "systems_used": systems_used,
                }
            elif local_valid:
                systems_used.append("vulcan_local_llm")
                return {
                    "text": results["local"],
                    "source": "local",
                    "systems_used": systems_used,
                }

        except asyncio.TimeoutError:
            self.logger.warning("Parallel execution timed out")

        return {"text": "", "source": "none", "systems_used": systems_used}

    async def _execute_ensemble(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Run both LLMs, combine/select best response based on quality."""
        systems_used = []

        async def local_task():
            return await self._call_local_llm(loop, prompt, max_tokens)

        async def openai_task():
            return await self._call_openai(
                loop, prompt, max_tokens, temperature, system_prompt, conversation_history
            )

        # Run both tasks concurrently
        try:
            local_result, openai_result = await asyncio.wait_for(
                asyncio.gather(local_task(), openai_task(), return_exceptions=True),
                timeout=self.timeout,
            )

            # Handle exceptions from gather
            if isinstance(local_result, Exception):
                self.logger.debug(f"Local LLM failed: {local_result}")
                local_result = None
            if isinstance(openai_result, Exception):
                self.logger.debug(f"OpenAI failed: {openai_result}")
                openai_result = None

        except asyncio.TimeoutError:
            self.logger.warning("Ensemble execution timed out")
            return {"text": "", "source": "none", "systems_used": systems_used}

        local_valid = self._is_valid_response(local_result)
        openai_valid = openai_result is not None and len(
            str(openai_result).strip()
        ) > self.MIN_MEANINGFUL_LENGTH

        if local_valid and openai_valid:
            # Both succeeded - evaluate and combine
            systems_used.extend(["vulcan_local_llm", "openai_llm"])

            # Ensemble strategy: Use OpenAI for final language quality,
            # but enrich with local LLM insights if available
            local_str = str(local_result)
            openai_str = str(openai_result)

            # Simple ensemble: If local response contains unique insights not in OpenAI,
            # append them. Otherwise, use OpenAI response (better language quality).
            combined_response = openai_str

            # Check if local has meaningful additional content
            if (
                len(local_str) > 50
                and self.MOCK_RESPONSE_MARKER not in local_str
                and local_str.strip() != openai_str.strip()
            ):
                # Local has different content - could be valuable reasoning
                truncated_local = local_str[: self.ENSEMBLE_LOCAL_RESPONSE_MAX_LENGTH]
                combined_response = f"{openai_str}\n\n[Additional Analysis from VULCAN Local LLM]:\n{truncated_local}"

            return {
                "text": combined_response,
                "source": "ensemble",
                "systems_used": systems_used,
                "metadata": {
                    "ensemble_mode": True,
                    "local_length": len(local_str),
                    "openai_length": len(openai_str),
                },
            }
        elif openai_valid:
            systems_used.append("openai_llm")
            return {
                "text": openai_result,
                "source": "openai",
                "systems_used": systems_used,
            }
        elif local_valid:
            systems_used.append("vulcan_local_llm")
            return {
                "text": local_result,
                "source": "local",
                "systems_used": systems_used,
            }

        return {"text": "", "source": "none", "systems_used": systems_used}

    # ============================================================
    # LLM CALL METHODS
    # ============================================================

    async def _call_local_llm(
        self, loop, prompt: str, max_tokens: int
    ) -> Optional[str]:
        """Call Vulcan's local LLM."""
        if not self.local_llm:
            return None

        try:
            result = await loop.run_in_executor(
                None, self.local_llm.generate, prompt, max_tokens
            )

            # Handle None result (returned when event loop conflict is detected)
            if result is None:
                self.logger.debug("Local LLM returned None - triggering fallback")
                return None

            if hasattr(result, "text"):
                return result.text
            elif isinstance(result, str):
                return result
            elif isinstance(result, dict) and "text" in result:
                return result["text"]
            else:
                return str(result)
        except Exception as e:
            self.logger.debug(f"Local LLM call failed: {e}")
            return None

    async def _call_openai(
        self,
        loop,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Call OpenAI API with conversation history support.
        
        Args:
            loop: The asyncio event loop
            prompt: The current user prompt
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            system_prompt: System prompt for OpenAI
            conversation_history: Optional list of previous messages for multi-turn context.
                                 Each message should have 'role' and 'content' keys.
        
        Returns:
            The generated response text, or None if the call fails.
        """
        openai_client = self.openai_client_getter()
        if not openai_client:
            return None

        try:
            # Use configurable max_tokens limit
            effective_max_tokens = min(max_tokens, self.openai_max_tokens)

            def call_openai():
                # Build messages array with conversation history
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add conversation history if provided
                # This enables multi-turn conversation context for the LLM
                if conversation_history:
                    for msg in conversation_history:
                        # Validate message structure
                        role = msg.get("role", "").lower()
                        content = msg.get("content", "")
                        
                        # Skip messages with empty or whitespace-only content
                        # to avoid issues with OpenAI API
                        if not content or not content.strip():
                            continue
                        
                        # Map roles to OpenAI-compatible roles
                        if role in ("user", "human"):
                            messages.append({"role": "user", "content": content})
                        elif role in ("assistant", "ai", "bot"):
                            messages.append({"role": "assistant", "content": content})
                        # Skip messages with invalid/unknown roles
                    
                    self.logger.debug(
                        f"OpenAI call with conversation history: {len(conversation_history)} messages"
                    )
                
                # Add current prompt as the final user message
                messages.append({"role": "user", "content": prompt})
                
                completion = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content

            return await loop.run_in_executor(None, call_openai)
        except Exception as e:
            self.logger.debug(f"OpenAI call failed: {e}")
            return None

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _is_valid_response(self, response: Optional[str]) -> bool:
        """Check if response is valid and meaningful."""
        if not response:
            return False
        response_str = str(response).strip()
        return (
            len(response_str) > self.MIN_MEANINGFUL_LENGTH
            and self.MOCK_RESPONSE_MARKER not in response_str
        )

    def _capture_for_distillation(self, prompt: str, result: Dict[str, Any]):
        """Capture response for knowledge distillation training."""
        try:
            distiller = get_knowledge_distiller()
            if distiller is None:
                return

            openai_response = result.get("text", "")
            local_response = result.get("metadata", {}).get("local_response_preview")

            # Capture the response for training
            distiller.capture_response(
                prompt=prompt,
                openai_response=openai_response,
                local_response=local_response,
                metadata={
                    "source": result.get("source"),
                    "systems_used": result.get("systems_used", []),
                    "mode": self.mode,
                },
            )
        except Exception as e:
            self.logger.debug(f"Failed to capture response for distillation: {e}")

    def _update_stats(self, result: Dict[str, Any]):
        """Update execution statistics."""
        source = result.get("source", "none")
        if source in ("local", "parallel_both", "ensemble"):
            self._local_successes += 1
        if source in ("openai", "parallel_both", "ensemble"):
            self._openai_successes += 1
        if source == "none":
            self._failures += 1

    # ============================================================
    # PUBLIC UTILITY METHODS
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_executions": self._execution_count,
            "local_successes": self._local_successes,
            "openai_successes": self._openai_successes,
            "failures": self._failures,
            "mode": self.mode,
            "has_local_llm": self.local_llm is not None,
        }

    def set_mode(self, mode: str) -> bool:
        """
        Change the execution mode.
        
        Args:
            mode: New execution mode
            
        Returns:
            True if mode was valid and set, False otherwise
        """
        mode_lower = mode.lower()
        if mode_lower not in self.VALID_MODES:
            self.logger.warning(f"Invalid mode: {mode}. Valid modes: {self.VALID_MODES}")
            return False
        self.mode = mode_lower
        self.logger.info(f"Execution mode changed to: {mode_lower}")
        return True

    def __repr__(self) -> str:
        return (
            f"HybridLLMExecutor(mode={self.mode!r}, "
            f"has_local_llm={self.local_llm is not None}, "
            f"timeout={self.timeout})"
        )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "HybridLLMExecutor",
]


# Log module initialization
logger.debug(f"Hybrid LLM executor module v{__version__} loaded")
