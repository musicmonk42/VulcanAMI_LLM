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
#     - OpenAI response caching (LRU with TTL)
#
# EXECUTION MODES:
#     - local_first: Try Vulcan's local LLM first, fallback to OpenAI
#     - openai_first: Try OpenAI first, fallback to local LLM
#     - parallel: Run both simultaneously, use first successful response
#     - ensemble: Run both, combine/select best response based on quality
#
# CACHING:
#     - OpenAI responses are cached with configurable TTL (default: 1 hour)
#     - Cache key includes prompt, max_tokens, temperature for uniqueness
#     - LRU eviction when cache exceeds max size
#     - Cache can be disabled per-request or globally
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
#     1.1.0 - Added OpenAI response caching with LRU and TTL
# ============================================================

import asyncio
import hashlib
import logging
import os
import threading
import time
import traceback
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

# Module metadata
__version__ = "1.2.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
# Default path for GraphixVulcanLLM config, can be overridden via environment variable
LLM_CONFIG_PATH = os.environ.get("VULCAN_LLM_CONFIG_PATH", "configs/llm_config.yaml")

# ============================================================
# COMPONENT REGISTRY INTEGRATION
# ============================================================
# Import component registry getter for auto-fetching internal LLM
# This import is at module level to avoid repeated import overhead
try:
    from vulcan.utils_main.components import get_component as _get_component_from_registry
except ImportError:
    _get_component_from_registry = None


# ============================================================
# LRU CACHE WITH TTL FOR OPENAI RESPONSES
# ============================================================


class OpenAIResponseCache:
    """
    Thread-safe LRU cache for OpenAI API responses with TTL support.
    
    Features:
    - LRU eviction when cache exceeds max size
    - TTL-based expiration (default: 1 hour)
    - Thread-safe operations with RLock
    - Cache key includes prompt hash, max_tokens, temperature
    
    This significantly reduces API costs and latency for repeated queries.
    
    Attributes:
        max_size: Maximum number of entries in cache
        ttl_seconds: Time-to-live for cache entries in seconds
        
    Example:
        >>> cache = OpenAIResponseCache(max_size=1000, ttl_seconds=3600)
        >>> cache.put("What is AI?", 1000, 0.7, "AI is...", {"tokens": 50})
        >>> result = cache.get("What is AI?", 1000, 0.7)
        >>> if result:
        ...     print(result["response"])  # "AI is..."
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    def _make_key(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Create a unique cache key for the request parameters.
        
        Uses SHA256 hash of combined parameters for efficient lookup.
        """
        # Create deterministic string representation
        # Use 4 decimal precision for temperature to avoid unintended cache collisions
        key_parts = [
            str(prompt),
            str(max_tokens),
            f"{temperature:.4f}",  # 4 decimal precision for temperature
            str(system_prompt or ""),
        ]
        key_str = "|".join(key_parts)
        
        # Use SHA256 hash for efficient fixed-size key
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def get(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available and not expired.
        
        Args:
            prompt: The user prompt
            max_tokens: Max tokens parameter
            temperature: Temperature parameter
            system_prompt: Optional system prompt
            
        Returns:
            Cached entry dict with 'response' and 'metadata' keys, or None if miss
        """
        key = self._make_key(prompt, max_tokens, temperature, system_prompt)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL expiration
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                # Entry expired - remove it
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            
            return {
                "response": entry["response"],
                "metadata": entry["metadata"],
                "cached_at": entry["timestamp"],
                "cache_age_seconds": time.time() - entry["timestamp"],
            }
    
    def put(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Store a response in the cache.
        
        Args:
            prompt: The user prompt
            max_tokens: Max tokens parameter
            temperature: Temperature parameter
            response: The OpenAI response text
            metadata: Optional metadata (tokens used, model, etc.)
            system_prompt: Optional system prompt
        """
        key = self._make_key(prompt, max_tokens, temperature, system_prompt)
        
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Check if we need to evict
                while len(self._cache) >= self.max_size:
                    # Remove oldest (first) item
                    self._cache.popitem(last=False)
                    self._evictions += 1
            
            # Store new entry
            self._cache[key] = {
                "response": response,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
    
    def invalidate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> bool:
        """
        Invalidate (remove) a specific cache entry.
        
        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._make_key(prompt, max_tokens, temperature, system_prompt)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        removed = 0
        
        with self._lock:
            # Create list of expired keys (can't modify dict during iteration)
            expired_keys = [
                key for key, entry in self._cache.items()
                if now - entry["timestamp"] > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed += 1
                self._expirations += 1
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
            }

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
    
    # Default system prompt that explicitly allows conversation memory
    # MEMORY FIX: The default prompt tells the model to remember information from conversation
    DEFAULT_SYSTEM_PROMPT = (
        "You are VULCAN, an advanced AI assistant. "
        "You SHOULD remember and reference information shared earlier in this conversation. "
        "When a user shares their name, location, preferences, or any personal details during this session, "
        "you may recall and use that information naturally in your responses."
    )

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
        enable_openai_cache: bool = True,
        openai_cache_max_size: int = 1000,
        openai_cache_ttl_seconds: int = 3600,  # 1 hour default
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
            enable_openai_cache: Enable caching of OpenAI responses (default: True)
            openai_cache_max_size: Maximum cache entries (default: 1000)
            openai_cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
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
        
        # OpenAI response cache for reducing API costs and latency
        self._enable_openai_cache = enable_openai_cache
        self._openai_cache: Optional[OpenAIResponseCache] = None
        if enable_openai_cache:
            self._openai_cache = OpenAIResponseCache(
                max_size=openai_cache_max_size,
                ttl_seconds=openai_cache_ttl_seconds,
            )
            self.logger.info(
                f"OpenAI response cache enabled (max_size={openai_cache_max_size}, "
                f"ttl={openai_cache_ttl_seconds}s)"
            )
        
        # Statistics tracking
        self._execution_count = 0
        self._local_successes = 0
        self._openai_successes = 0
        self._failures = 0

    # ============================================================
    # MAIN EXECUTION METHOD
    # ============================================================

    async def execute(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
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
                # Both succeeded - PREFER LOCAL LLM to reduce OpenAI API costs
                # OpenAI response is still captured in metadata for distillation
                systems_used.extend(["vulcan_local_llm", "openai_llm"])
                self.logger.info(
                    "[HybridExecutor] Both LLMs succeeded - using LOCAL response (OpenAI captured for distillation)"
                )
                return {
                    "text": results["local"],  # Use local LLM response
                    "source": "parallel_both",
                    "systems_used": systems_used,
                    "metadata": {
                        "local_response_available": True,
                        "openai_response_available": True,
                        # Store OpenAI response in metadata for knowledge distillation
                        "openai_response_for_distillation": str(results["openai"])[:2000],
                        "used_source": "local",
                    },
                }
            elif local_valid:
                # Local succeeded, OpenAI failed or not ready
                systems_used.append("vulcan_local_llm")
                self.logger.info("[HybridExecutor] Using LOCAL LLM response (OpenAI unavailable)")
                return {
                    "text": results["local"],
                    "source": "local",
                    "systems_used": systems_used,
                }
            elif openai_valid:
                # Only OpenAI succeeded - fallback
                systems_used.append("openai_llm")
                self.logger.info("[HybridExecutor] Using OpenAI FALLBACK (local LLM unavailable)")
                return {
                    "text": results["openai"],
                    "source": "openai",
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
        """Call Vulcan's local LLM.
        
        BUG #1 FIX: Added detailed error logging to expose why local model
        generation fails silently. Previously, exceptions were caught and
        logged at debug level, hiding the real cause of 100% OpenAI fallback.
        """
        import traceback
        import time as time_module
        
        # BUG #1 FIX: Log entry point with model state
        self.logger.info("=" * 60)
        self.logger.info("[HybridExecutor] ATTEMPTING LOCAL LLM GENERATION")
        self.logger.info(f"[HybridExecutor] local_llm is None: {self.local_llm is None}")
        
        if not self.local_llm:
            self.logger.error("[HybridExecutor] CRITICAL: local_llm is None - cannot generate!")
            self.logger.info("=" * 60)
            return None
        
        # BUG #1 FIX: Log detailed model state
        self.logger.info(f"[HybridExecutor] local_llm type: {type(self.local_llm).__name__}")
        self.logger.info(f"[HybridExecutor] local_llm has generate: {hasattr(self.local_llm, 'generate')}")
        
        # Check for common model state issues
        if hasattr(self.local_llm, 'model'):
            self.logger.info(f"[HybridExecutor] internal model: {type(self.local_llm.model).__name__ if self.local_llm.model else 'None'}")
        if hasattr(self.local_llm, 'tokenizer'):
            self.logger.info(f"[HybridExecutor] tokenizer: {type(self.local_llm.tokenizer).__name__ if self.local_llm.tokenizer else 'None'}")
        if hasattr(self.local_llm, 'device'):
            self.logger.info(f"[HybridExecutor] device: {self.local_llm.device}")

        try:
            self.logger.info(f"[HybridExecutor] Calling generate(prompt_len={len(prompt)}, max_tokens={max_tokens})...")
            start_time = time_module.perf_counter()
            
            result = await loop.run_in_executor(
                None, self.local_llm.generate, prompt, max_tokens
            )
            
            elapsed = time_module.perf_counter() - start_time
            self.logger.info(f"[HybridExecutor] generate() returned in {elapsed:.2f}s")

            # Handle None result (returned when event loop conflict is detected)
            if result is None:
                self.logger.warning("[HybridExecutor] Local LLM returned None - triggering fallback")
                self.logger.info("=" * 60)
                return None

            # BUG #1 FIX: Log successful generation
            if hasattr(result, "text"):
                self.logger.info(f"[HybridExecutor] ✓ LOCAL GENERATION SUCCEEDED ({len(result.text)} chars)")
                self.logger.info("=" * 60)
                return result.text
            elif isinstance(result, str):
                self.logger.info(f"[HybridExecutor] ✓ LOCAL GENERATION SUCCEEDED ({len(result)} chars)")
                self.logger.info("=" * 60)
                return result
            elif isinstance(result, dict) and "text" in result:
                self.logger.info(f"[HybridExecutor] ✓ LOCAL GENERATION SUCCEEDED ({len(result['text'])} chars)")
                self.logger.info("=" * 60)
                return result["text"]
            else:
                result_str = str(result)
                self.logger.info(f"[HybridExecutor] ✓ LOCAL GENERATION SUCCEEDED ({len(result_str)} chars, converted)")
                self.logger.info("=" * 60)
                return result_str
                
        except Exception as e:
            # BUG #1 FIX: Log FULL error details - this is critical for debugging
            self.logger.error("=" * 60)
            self.logger.error("[HybridExecutor] LOCAL MODEL GENERATION FAILED!")
            self.logger.error(f"[HybridExecutor] Exception type: {type(e).__name__}")
            self.logger.error(f"[HybridExecutor] Exception message: {str(e)}")
            self.logger.error(f"[HybridExecutor] Full traceback:\n{traceback.format_exc()}")
            self.logger.error("=" * 60)
            return None

    async def _call_openai(
        self,
        loop,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Call OpenAI API with conversation history support and caching.
        
        Args:
            loop: The asyncio event loop
            prompt: The current user prompt
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            system_prompt: System prompt for OpenAI
            conversation_history: Optional list of previous messages for multi-turn context.
                                 Each message should have 'role' and 'content' keys.
            use_cache: Whether to use the response cache (default: True)
        
        Returns:
            The generated response text, or None if the call fails.
        
        Caching:
            - Responses are cached based on prompt, max_tokens, temperature, system_prompt
            - Conversation history is NOT included in cache key (each unique prompt is cached)
            - Cache reduces API costs by ~95% for repeated queries
            - Cache entries expire after TTL (default: 1 hour)
        """
        # Check cache first (only for single-turn requests without history)
        # Note: We don't cache conversation history queries as context changes results
        if (
            use_cache 
            and self._openai_cache 
            and not conversation_history
        ):
            cached = self._openai_cache.get(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            if cached:
                # PERF FIX Issue #7: Log cache hit at INFO level for visibility
                self.logger.info(
                    f"[CACHE HIT] OpenAI response from cache "
                    f"(age={cached['cache_age_seconds']:.1f}s, saved ~2-5s API call)"
                )
                return cached["response"]
            else:
                # Log cache miss to help debug cache effectiveness
                cache_stats = self._openai_cache.get_stats()
                self.logger.debug(
                    f"[CACHE MISS] OpenAI cache miss "
                    f"(cache_size={cache_stats['size']}, hit_rate={cache_stats['hit_rate']:.1%})"
                )
        
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

            result = await loop.run_in_executor(None, call_openai)
            
            # Cache the result (only for single-turn requests without history)
            if (
                result 
                and use_cache 
                and self._openai_cache 
                and not conversation_history
            ):
                self._openai_cache.put(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response=result,
                    metadata={"model": "gpt-3.5-turbo"},
                    system_prompt=system_prompt,
                )
                # PERF FIX Issue #7: Log cache storage at INFO level
                cache_stats = self._openai_cache.get_stats()
                self.logger.info(
                    f"[CACHE STORED] OpenAI response cached "
                    f"(cache_size={cache_stats['size']}, hit_rate={cache_stats['hit_rate']:.1%})"
                )
            
            return result
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
        Get execution statistics including cache statistics.
        
        Returns:
            Dictionary with execution and cache statistics
        """
        stats = {
            "total_executions": self._execution_count,
            "local_successes": self._local_successes,
            "openai_successes": self._openai_successes,
            "failures": self._failures,
            "mode": self.mode,
            "has_local_llm": self.local_llm is not None,
            "openai_cache_enabled": self._enable_openai_cache,
        }
        
        # Add cache statistics if cache is enabled
        if self._openai_cache:
            stats["openai_cache"] = self._openai_cache.get_stats()
        
        return stats
    
    def clear_openai_cache(self) -> int:
        """
        Clear the OpenAI response cache.
        
        Returns:
            Number of entries cleared
        """
        if self._openai_cache:
            count = self._openai_cache.clear()
            self.logger.info(f"OpenAI cache cleared ({count} entries)")
            return count
        return 0
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired entries from the OpenAI response cache.
        
        Returns:
            Number of entries removed
        """
        if self._openai_cache:
            count = self._openai_cache.cleanup_expired()
            if count > 0:
                self.logger.info(f"Cleaned up {count} expired cache entries")
            return count
        return 0

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

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None, max_tokens: int = 500) -> str:
        """
        Synchronous generation using the local LLM.
        
        BUG #1 FIX: This method provides explicit error handling and logging
        instead of silently falling back to OpenAI. Errors are raised, not hidden.
        
        Args:
            prompt: The input prompt for generation
            context: Optional context dictionary (currently unused, for API compatibility)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text from the local LLM
            
        Raises:
            RuntimeError: If the local model is not initialized
            Exception: Propagates any exception from the local model generation
        """
        if self.local_llm is None:
            self.logger.error("[HybridExecutor] local_llm is None!")
            raise RuntimeError("Local model not initialized")
        
        self.logger.info(f"[HybridExecutor] Starting local generation...")
        
        try:
            response = self.local_llm.generate(
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            # Handle different response types
            if hasattr(response, 'text'):
                result = response.text
            elif isinstance(response, dict) and 'text' in response:
                result = response['text']
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)
            
            self.logger.info(f"[HybridExecutor] ✓ Local generation succeeded: {len(result)} chars")
            return result
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"[HybridExecutor] GENERATION FAILED: {type(e).__name__}")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.logger.error("=" * 80)
            raise  # Don't hide the error

    def __repr__(self) -> str:
        return (
            f"HybridLLMExecutor(mode={self.mode!r}, "
            f"has_local_llm={self.local_llm is not None}, "
            f"timeout={self.timeout})"
        )


# ============================================================
# SINGLETON MANAGEMENT FOR HYBRID LLM EXECUTOR
# ============================================================
# This ensures the HybridLLMExecutor is only created once per process,
# preventing repeated initialization overhead and ensuring consistent state.

_hybrid_executor_instance: Optional["HybridLLMExecutor"] = None
_hybrid_executor_lock = threading.Lock()


def get_or_create_hybrid_executor(
    local_llm: Optional[Any] = None,
    openai_client_getter: Optional[Callable] = None,
    mode: str = "parallel",
    timeout: float = 30.0,
    ensemble_min_confidence: float = 0.7,
    openai_max_tokens: int = 2000,
    enable_openai_cache: bool = True,
    force_new: bool = False,
) -> "HybridLLMExecutor":
    """
    Get or create a singleton HybridLLMExecutor instance.
    
    This ensures only one executor exists per process, preventing:
    - Repeated initialization overhead (~0.5s per request)
    - Lost cache state between requests
    - Inconsistent configuration
    
    IMPORTANT: If local_llm is not provided, this function will automatically
    attempt to fetch it from the global component registry. This ensures the
    HybridExecutor always has access to the internal LLM when available.
    
    Args:
        local_llm: Vulcan's local LLM instance (only used on first creation).
                   If None, will attempt to fetch from global component registry.
        openai_client_getter: Function to get OpenAI client (only used on first creation)
        mode: Execution mode (only used on first creation unless force_new)
        timeout: Timeout for parallel/ensemble execution (only used on first creation)
        ensemble_min_confidence: Minimum confidence for ensemble selection
        openai_max_tokens: Maximum tokens for OpenAI API calls
        enable_openai_cache: Enable caching of OpenAI responses
        force_new: If True, create a new instance even if one exists (for testing)
        
    Returns:
        The singleton HybridLLMExecutor instance
        
    Example:
        # First call creates the instance
        executor = get_or_create_hybrid_executor(local_llm=my_llm, mode="parallel")
        
        # Subsequent calls return the same instance
        executor2 = get_or_create_hybrid_executor()
        assert executor is executor2
    """
    global _hybrid_executor_instance
    
    with _hybrid_executor_lock:
        if _hybrid_executor_instance is not None and not force_new:
            # Return existing instance
            logger.debug("[HybridExecutor] Returning cached singleton instance")
            return _hybrid_executor_instance
        
        # Auto-fetch local LLM from component registry if not provided
        # This ensures HybridExecutor can access the internal LLM even when called
        # without explicit parameters (e.g., during singleton creation on first request)
        effective_local_llm = local_llm
        if effective_local_llm is None and _get_component_from_registry is not None:
            try:
                effective_local_llm = _get_component_from_registry("llm")
                if effective_local_llm is not None:
                    logger.info("[HybridExecutor] ✓ Auto-fetched internal LLM from global registry")
                else:
                    logger.warning(
                        "[HybridExecutor] No internal LLM found in global registry - "
                        "will try direct GraphixVulcanLLM import"
                    )
            except Exception as e:
                logger.warning(f"[HybridExecutor] Failed to fetch internal LLM from registry: {e}")
        elif effective_local_llm is None and _get_component_from_registry is None:
            logger.debug("[HybridExecutor] Component registry not available for auto-fetch")
        
        # FIX #1: If still no LLM, try direct import of GraphixVulcanLLM as fallback
        # This handles cases where the component registry hasn't been initialized yet
        if effective_local_llm is None:
            try:
                # Try importing GraphixVulcanLLM directly
                from graphix_vulcan_llm import GraphixVulcanLLM
                logger.info(f"[HybridExecutor] Attempting direct GraphixVulcanLLM instantiation (config={LLM_CONFIG_PATH})...")
                effective_local_llm = GraphixVulcanLLM(config_path=LLM_CONFIG_PATH)
                logger.info("[HybridExecutor] ✓ Direct GraphixVulcanLLM instantiation successful")
                
                # Register in component registry for future use
                if _get_component_from_registry is not None:
                    try:
                        from vulcan.utils_main.components import set_component
                        set_component("llm", effective_local_llm)
                        logger.info("[HybridExecutor] ✓ Registered GraphixVulcanLLM in component registry")
                    except Exception as reg_e:
                        logger.debug(f"[HybridExecutor] Could not register LLM in registry: {reg_e}")
            except ImportError as ie:
                logger.warning(f"[HybridExecutor] GraphixVulcanLLM not available for import: {ie}")
            except Exception as e:
                logger.warning(f"[HybridExecutor] Failed to create GraphixVulcanLLM directly: {e}")
        
        # Create new instance
        has_local = effective_local_llm is not None
        logger.info(
            f"[HybridExecutor] Creating singleton instance "
            f"(mode={mode}, has_local_llm={has_local})"
        )
        _hybrid_executor_instance = HybridLLMExecutor(
            local_llm=effective_local_llm,
            openai_client_getter=openai_client_getter,
            mode=mode,
            timeout=timeout,
            ensemble_min_confidence=ensemble_min_confidence,
            openai_max_tokens=openai_max_tokens,
            enable_openai_cache=enable_openai_cache,
        )
        logger.info(
            f"[HybridExecutor] ✓ Singleton instance created successfully "
            f"(internal_llm_available={has_local})"
        )
        return _hybrid_executor_instance


def get_hybrid_executor() -> Optional["HybridLLMExecutor"]:
    """
    Get the existing HybridLLMExecutor singleton without creating a new one.
    
    Returns:
        The singleton instance if it exists, None otherwise
    """
    return _hybrid_executor_instance


def set_hybrid_executor(executor: "HybridLLMExecutor") -> None:
    """
    Set the HybridLLMExecutor singleton instance.
    
    This is useful when the executor is created elsewhere (e.g., app startup)
    and needs to be registered with the singleton.
    
    Args:
        executor: The HybridLLMExecutor instance to set as singleton
    """
    global _hybrid_executor_instance
    with _hybrid_executor_lock:
        _hybrid_executor_instance = executor
        logger.info("[HybridExecutor] Singleton instance registered externally")


def verify_hybrid_executor_setup() -> dict:
    """
    Verify that HybridExecutor has access to internal LLM.
    
    FIX #1 VERIFICATION: This function can be called after startup to verify
    that the internal LLM is properly connected to the HybridExecutor.
    
    Returns:
        Dictionary with verification results:
        - has_internal_llm: bool - Whether internal LLM is available
        - internal_llm_type: str - Type name of internal LLM (or None)
        - internal_llm_vocab_size: int - Vocab size if available
        - status: str - "PASS" or "FAIL"
        - message: str - Human-readable status message
    """
    result = {
        "has_internal_llm": False,
        "internal_llm_type": None,
        "internal_llm_vocab_size": None,
        "status": "FAIL",
        "message": "HybridExecutor not initialized"
    }
    
    executor = get_hybrid_executor()
    if executor is None:
        result["message"] = "HybridExecutor singleton not created yet"
        return result
    
    # Check internal model
    has_internal = executor.local_llm is not None
    result["has_internal_llm"] = has_internal
    
    if has_internal:
        result["internal_llm_type"] = type(executor.local_llm).__name__
        
        # Try to get vocab size
        vocab_size = getattr(executor.local_llm, 'vocab_size', None)
        if vocab_size is None and hasattr(executor.local_llm, 'config'):
            vocab_size = getattr(executor.local_llm.config, 'vocab_size', None)
        result["internal_llm_vocab_size"] = vocab_size
        
        result["status"] = "PASS"
        result["message"] = f"✓ Internal LLM connected: {result['internal_llm_type']}"
        if vocab_size:
            result["message"] += f" (vocab_size={vocab_size})"
        logger.info(f"[HybridExecutor] VERIFICATION PASSED: {result['message']}")
    else:
        result["status"] = "FAIL"
        result["message"] = "❌ Internal LLM is None - queries will fall back to OpenAI"
        logger.warning(f"[HybridExecutor] VERIFICATION FAILED: {result['message']}")
    
    return result


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "HybridLLMExecutor",
    "OpenAIResponseCache",
    "get_or_create_hybrid_executor",
    "get_hybrid_executor",
    "set_hybrid_executor",
    "verify_hybrid_executor_setup",
]


# Log module initialization
logger.debug(f"Hybrid LLM executor module v{__version__} loaded")
