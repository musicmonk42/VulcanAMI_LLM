# ============================================================
# VULCAN-AGI Hybrid LLM Executor Module
# VULCAN's reasoning systems do ALL thinking - LLMs are for language only
# ============================================================
#
# ARCHITECTURE:
#     - VULCAN's reasoning systems (symbolic, probabilistic, causal, 
#       mathematical) handle ALL reasoning/thinking
#     - The internal LLM (GraphixVulcanLLM) is for LANGUAGE GENERATION,
#       converting structured reasoning outputs to natural language prose
#     - OpenAI is ALSO for language generation - same role as internal LLM
#     - Neither LLM (internal nor OpenAI) does "thinking" - they are
#       language output formatters for VULCAN's reasoning results
#
# KEY INSIGHT:
#     Neither GraphixVulcanLLM nor OpenAI is "the mind." They're output
#     formatters. VULCAN's reasoning systems already did the thinking
#     before any LLM is invoked.
#
# PERMITTED OPENAI USAGE:
#     - When OPENAI_LANGUAGE_POLISH=true:
#       OpenAI can polish the language output into clearer prose
#     - OpenAI must NOT reason, analyze, or generate independent responses
#
# INTERNAL LLM ROLE:
#     - Primary language generation from VULCAN's reasoning outputs
#     - Same conceptual role as OpenAI - converting structured results
#       to natural language
#     - Not for reasoning - reasoning is done by VULCAN's reasoning systems
#
# CONFIGURATION:
#     - OPENAI_LANGUAGE_POLISH=false (default) - Use internal LLM for output
#     - OPENAI_LANGUAGE_POLISH=true - Polish internal LLM output with OpenAI
#
# USAGE:
#     from vulcan.llm.hybrid_executor import HybridLLMExecutor
#     
#     executor = HybridLLMExecutor(local_llm=my_llm)
#     result = await executor.execute("What is 2+2?")
#     print(result["text"])  # Language output from VULCAN's reasoning
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.1.0 - Added OpenAI response caching with LRU and TTL
#     1.2.0 - Removed OpenAI reasoning fallback - VULCAN only for reasoning
#     1.3.0 - Clarified that internal LLM is for language, not reasoning
# ============================================================

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Module metadata
__version__ = "1.5.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
# Default path for GraphixVulcanLLM config, can be overridden via environment variable
LLM_CONFIG_PATH = os.environ.get("VULCAN_LLM_CONFIG_PATH", "configs/llm_config.yaml")

# NOTE: OPENAI_FALLBACK_CONFIDENCE is kept for backwards compatibility but
# OpenAI reasoning fallback is now disabled. This constant is only used
# if language polish mode is enabled.
OPENAI_FALLBACK_CONFIDENCE = 0.6

# ARCHITECTURE: VULCAN does ALL reasoning. OpenAI is for language polish ONLY.
# 
# The correct flow is:
# 1. VULCAN reasoning engines analyze query (symbolic, probabilistic, causal, math)
# 2. GraphixVulcanLLM generates response using VULCAN's internal model
# 3. If VULCAN succeeds -> return VULCAN response (optionally polish with OpenAI)
# 4. If VULCAN fails -> return error (NO OpenAI reasoning fallback)
#
# SKIP_LOCAL_LLM is deprecated - VULCAN must always run for reasoning.
_skip_local_llm_env = os.environ.get("SKIP_LOCAL_LLM", "false").lower()
SKIP_LOCAL_LLM = _skip_local_llm_env in ("true", "1", "yes")

# ============================================================
# TIMEOUT CONFIGURATION - INCREASED FOR CPU EXECUTION
# ============================================================
# FIX: Increased timeouts to prevent premature timeouts during CPU-intensive
# language generation. The internal LLM can take 3+ seconds per token on CPU.
VULCAN_HARD_TIMEOUT = float(os.environ.get("VULCAN_LLM_HARD_TIMEOUT", "120.0"))  # 2 minutes (was 30s)
PER_TOKEN_TIMEOUT = float(os.environ.get("VULCAN_LLM_PER_TOKEN_TIMEOUT", "30.0"))  # 30s per token (was 10s)

# Fast mode timeout for output formatting (when reasoning is already done)
# This can be shorter since no reasoning hooks run per-token
FAST_MODE_MAX_TIMEOUT_SECONDS = float(os.environ.get("VULCAN_LLM_FAST_TIMEOUT", "60.0"))  # 60 seconds

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
# VULCAN REASONING OUTPUT - STRUCTURED OUTPUT FORMAT
# ============================================================
# This dataclass defines the structured format VULCAN's mind outputs
# BEFORE any LLM is called for prose generation.


@dataclass
class VulcanReasoningOutput:
    """
    Output from VULCAN's reasoning systems (the mind).
    
    This represents the structured result from VULCAN's internal reasoning
    systems (symbolic, probabilistic, causal, mathematical) BEFORE any
    language model converts it to natural language prose.
    
    The key insight is that neither GraphixVulcanLLM nor OpenAI is "the mind."
    They're output formatters. VULCAN's mind already did its reasoning work
    before any LLM is invoked.
    
    Attributes:
        query_id: Unique identifier for this query
        success: Whether reasoning succeeded
        result: The actual answer/computation (can be any type)
        result_type: Category of result (mathematical, symbolic, factual, etc.)
        method_used: Which reasoning system solved it
        confidence: Confidence score 0.0 - 1.0
        reasoning_trace: Steps taken during reasoning (for transparency)
        error: Error message if reasoning failed
        metadata: Additional context about the reasoning
    """
    query_id: str
    success: bool
    result: Any
    result_type: str = "unknown"  # "mathematical", "symbolic", "factual", "causal", etc.
    method_used: str = "unknown"  # "symbolic_integration", "probabilistic", "agent_pool", etc.
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if this is a valid, successful reasoning output."""
        return self.success and self.result is not None
    
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"VulcanReasoningOutput({status} query_id={self.query_id!r}, "
            f"result_type={self.result_type!r}, confidence={self.confidence:.2f})"
        )


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

# Import settings for skip_local_llm configuration (with fallback)
try:
    from vulcan.settings import settings as _settings
except ImportError:
    _settings = None


def _should_skip_local_llm() -> bool:
    """
    Check if local LLM should be skipped based on configuration.
    
    ARCHITECTURE: VULCAN is the primary brain, OpenAI is language fallback only.
    By default, this returns False so VULCAN reasoning runs first.
    
    Set SKIP_LOCAL_LLM=true environment variable ONLY if you want to bypass
    VULCAN entirely for testing purposes.
    
    Returns:
        True if local LLM should be skipped (default: False)
    """
    # Environment variable takes precedence (set at module load time)
    if SKIP_LOCAL_LLM:
        return True
    
    # Check settings module if available
    if _settings is not None:
        return getattr(_settings, 'skip_local_llm', False)
    
    return False


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
    
    # Default system prompt - OpenAI is ONLY for language generation, NOT reasoning
    # ARCHITECTURE: VULCAN does ALL reasoning. OpenAI only expresses VULCAN's reasoning in fluent prose.
    # OpenAI should NEVER reason independently - it is purely a language generation layer.
    DEFAULT_SYSTEM_PROMPT = (
        "You are a language generation assistant for VULCAN. "
        "Your ONLY role is to express VULCAN's reasoning results in clear, natural language. "
        "You must NOT perform any independent reasoning, analysis, or problem-solving. "
        "Simply take the reasoning provided and express it in fluent, conversational prose. "
        "If no reasoning context is provided, acknowledge that VULCAN's reasoning system is processing. "
        "NEVER answer questions using your own knowledge - only express what VULCAN's reasoning provides."
    )
    
    # Prompt template when VULCAN reasoning succeeds and needs language polish
    LANGUAGE_ONLY_PROMPT_TEMPLATE = (
        "You are a language polisher. Your ONLY job is to improve the clarity and grammar "
        "of the text below.\n\n"
        "RULES:\n"
        "- Do NOT add new information or reasoning\n"
        "- Do NOT change the meaning\n"
        "- Do NOT expand on ideas\n"
        "- Do NOT answer questions or add explanations\n"
        "- ONLY fix grammar, punctuation, and clarity\n"
        "- Keep approximately the same length\n\n"
        "Text to polish:\n{reasoning_result}\n\n"
        "Polished version:"
    )
    
    # NOTE: FULL_REASONING_FALLBACK_PROMPT has been REMOVED
    # OpenAI is NOT permitted to do reasoning. If VULCAN fails, we return an error.
    # OpenAI can ONLY interpret/polish what VULCAN produces - nothing else.

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
        
        # HARD TIMEOUT FIX: ThreadPoolExecutor for VULCAN calls
        # asyncio.wait_for() only checks timeouts between await points
        # ThreadPoolExecutor.submit().result(timeout=X) provides TRUE hard timeout
        self._timeout_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="hybrid_timeout_"
        )
        # Parse VULCAN_LLM_TIMEOUT with error handling for invalid values
        # FIX: Use VULCAN_HARD_TIMEOUT constant (default 120s) for CPU-intensive reasoning
        # Previous 30s was causing timeouts before internal LLM could complete complex reasoning
        try:
            env_timeout = os.environ.get("VULCAN_LLM_TIMEOUT")
            if env_timeout:
                self.vulcan_timeout = float(env_timeout)
            else:
                # Use module-level constant as default
                self.vulcan_timeout = VULCAN_HARD_TIMEOUT
        except (ValueError, TypeError):
            self.logger.warning(
                f"[HybridExecutor] Invalid VULCAN_LLM_TIMEOUT value, using default {VULCAN_HARD_TIMEOUT}s"
            )
            self.vulcan_timeout = VULCAN_HARD_TIMEOUT
        self.logger.info(f"[HybridExecutor] VULCAN hard timeout set to {self.vulcan_timeout}s")
        
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
        
        # Distillation queue for capturing polish training examples
        # When OpenAI polishes Internal LLM output, we capture the pair for training
        self._distillation_queue: List[Dict[str, Any]] = []
        self._distillation_enabled = os.environ.get("ENABLE_DISTILLATION", "true").lower() in ("true", "1", "yes")
        if self._distillation_enabled:
            self.logger.info("[HybridExecutor] Distillation capture enabled")

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

    async def execute_with_structured_output(
        self,
        prompt: str,
        reasoning_output: Optional["VulcanReasoningOutput"] = None,
        context: Optional[Dict[str, Any]] = None,
        use_openai_formatting: Optional[bool] = None,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute with support for structured reasoning output.
        
        This implements the VULCAN Hybrid Output pattern where:
        1. VULCAN's reasoning systems (the "mind") complete their work first
        2. The result is captured in a VulcanReasoningOutput structure
        3. OpenAI (or internal LLM) is used ONLY for prose formatting
        
        This approach solves the timeout problem because:
        - VULCAN's reasoning is already complete (passed in reasoning_output)
        - OpenAI is fast for simple prose generation (~2-5 seconds)
        - The slow internal LLM is bypassed for the output formatting step
        
        Args:
            prompt: The original user query
            reasoning_output: Pre-computed structured output from VULCAN's reasoning systems.
                            If None, falls back to legacy execute() behavior.
            context: Optional context dict (may contain reasoning_output if not provided directly)
            use_openai_formatting: Whether to use OpenAI for formatting.
                                 None = auto-detect from OPENAI_LANGUAGE_POLISH env var.
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with 'text', 'source', 'systems_used', 'metadata', and optional 'reasoning_output'
        """
        loop = asyncio.get_running_loop()
        
        # Check if reasoning_output is in context
        if reasoning_output is None and context:
            reasoning_output = context.get("reasoning_output")
        
        # If no structured output, fall back to legacy execution
        if reasoning_output is None:
            self.logger.info("[HybridExecutor] No structured output provided, using legacy execution")
            return await self.execute(prompt, max_tokens=max_tokens)
        
        # Validate reasoning output
        if not isinstance(reasoning_output, VulcanReasoningOutput):
            self.logger.warning(
                f"[HybridExecutor] reasoning_output is not VulcanReasoningOutput (got {type(reasoning_output).__name__}), "
                "using legacy execution"
            )
            return await self.execute(prompt, max_tokens=max_tokens)
        
        # Determine if we should use OpenAI for formatting
        if use_openai_formatting is None:
            use_openai_formatting = os.environ.get("OPENAI_LANGUAGE_POLISH", "false").lower() in ("true", "1", "yes")
        
        systems_used = ["vulcan_reasoning"]
        
        # Check if reasoning succeeded
        if not reasoning_output.success:
            # Return error in a user-friendly way
            error_text = self._format_reasoning_error(reasoning_output)
            return {
                "text": error_text,
                "source": "vulcan_reasoning_error",
                "systems_used": systems_used,
                "error": True,
                "metadata": {
                    "reasoning_output": reasoning_output.to_dict(),
                    "query": prompt,
                },
            }
        
        # Format the successful reasoning output
        if use_openai_formatting:
            try:
                formatted = await self._format_with_openai(reasoning_output, prompt, loop)
                if formatted:
                    systems_used.append("openai_formatting")
                    self.logger.info("[HybridExecutor] ✓ Used OpenAI for output formatting (fast path)")
                    
                    # Capture for distillation if enabled
                    if self._distillation_enabled:
                        self._capture_polish_for_distillation(
                            prompt=prompt,
                            internal_output=self._format_structured_output_sync(reasoning_output),
                            teacher_output=formatted,
                        )
                    
                    return {
                        "text": formatted,
                        "source": "vulcan_with_openai_formatting",
                        "systems_used": systems_used,
                        "metadata": {
                            "reasoning_output": reasoning_output.to_dict(),
                            "query": prompt,
                            "openai_role": "formatting_only",
                        },
                    }
            except Exception as e:
                self.logger.warning(f"[HybridExecutor] OpenAI formatting failed: {e}, using fallback")
        
        # Fallback to simple formatting (no external API)
        formatted = self._format_structured_output_sync(reasoning_output)
        systems_used.append("internal_formatting")
        
        return {
            "text": formatted,
            "source": "vulcan_internal_formatting",
            "systems_used": systems_used,
            "metadata": {
                "reasoning_output": reasoning_output.to_dict(),
                "query": prompt,
            },
        }

    def _format_reasoning_error(self, reasoning_output: "VulcanReasoningOutput") -> str:
        """Format a reasoning error for user display."""
        # Generate error reference using module-level imports (hashlib, time)
        error_ref = hashlib.sha256(
            f"{time.time()}:{reasoning_output.query_id}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an issue while processing your request.\n\n"
        )
        
        if reasoning_output.error:
            # Provide specific error context without exposing internal details
            if "timeout" in reasoning_output.error.lower():
                error_text += (
                    "**Issue:** The computation took longer than expected.\n\n"
                    "**Suggestions:**\n"
                    "• Try breaking your question into smaller parts\n"
                    "• Simplify complex calculations\n"
                    "• Try again in a moment\n"
                )
            elif "memory" in reasoning_output.error.lower():
                error_text += (
                    "**Issue:** The system ran into resource constraints.\n\n"
                    "**Suggestions:**\n"
                    "• Simplify your query\n"
                    "• Try again shortly\n"
                )
            else:
                error_text += (
                    "**Issue:** An internal processing error occurred.\n\n"
                    "**Suggestions:**\n"
                    "• Rephrase your question\n"
                    "• Try a different approach\n"
                )
        else:
            error_text += (
                "**Issue:** Could not complete the reasoning process.\n\n"
                "**Suggestions:**\n"
                "• Try rephrasing your question\n"
                "• Break it into simpler parts\n"
            )
        
        error_text += f"\nIf this persists, reference: **{error_ref}**"
        
        return error_text

    # ============================================================
    # EXECUTION MODE IMPLEMENTATIONS
    # ============================================================

    async def _execute_local_first(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """VULCAN's reasoning systems do ALL thinking. LLMs are for language output only.
        
        ARCHITECTURE:
        1. VULCAN's reasoning systems (symbolic, probabilistic, causal, math) do thinking
        2. Internal LLM generates language output from the reasoning results
        3. If enabled, OpenAI can polish the language (same role as internal LLM)
        4. Neither internal LLM nor OpenAI does reasoning - they format output
        
        OpenAI is permitted to polish language output but must NOT reason independently.
        The internal LLM has the same conceptual role - language generation, not reasoning.
        """
        systems_used = []

        # Step 1: Internal LLM generates language output (reasoning already done by VULCAN systems)
        local_result = await self._call_local_llm(loop, prompt, max_tokens)
        
        if self._is_valid_response(local_result):
            systems_used.append("vulcan_local_llm")
            self.logger.info("[HybridExecutor] ✓ Internal LLM language generation succeeded")
            
            # Step 2: Optionally use OpenAI to polish the language
            # Both internal LLM and OpenAI serve the same role - language generation
            use_language_polish = os.environ.get("OPENAI_LANGUAGE_POLISH", "false").lower() in ("true", "1", "yes")
            
            if use_language_polish:
                try:
                    # Use OpenAI to express the result in better language
                    polish_prompt = self.LANGUAGE_ONLY_PROMPT_TEMPLATE.format(reasoning_result=local_result)
                    polished = await self._call_openai(
                        loop, polish_prompt, max_tokens, temperature, 
                        self.DEFAULT_SYSTEM_PROMPT, conversation_history
                    )
                    if polished and len(polished.strip()) > self.MIN_MEANINGFUL_LENGTH:
                        systems_used.append("openai_language_polish")
                        self.logger.info("[HybridExecutor] ✓ OpenAI polished language output")
                        
                        # DISTILLATION: Capture training example for Internal LLM to learn
                        # Student learns to produce polished output directly
                        if self._distillation_enabled:
                            self._capture_polish_for_distillation(
                                prompt=prompt,
                                internal_output=local_result,
                                teacher_output=polished,
                            )
                        
                        return {
                            "text": polished,
                            "source": "vulcan_with_openai_polish",
                            "systems_used": systems_used,
                            "metadata": {
                                "vulcan_raw_result": local_result[:1000],  # Store raw result
                                "openai_role": "language_polish_only",
                            },
                        }
                except Exception as e:
                    self.logger.warning(f"[HybridExecutor] Language polish failed, using raw output: {e}")
            
            # Return internal LLM's result directly (no polish)
            return {
                "text": local_result,
                "source": "local",
                "systems_used": systems_used,
            }

        # Internal LLM language generation failed
        # Note: This doesn't mean reasoning failed - reasoning is done by VULCAN's reasoning systems
        # The internal LLM is only for language output, same role as OpenAI
        self.logger.error(
            "[HybridExecutor] ❌ Internal LLM language generation failed. "
            "Note: The internal LLM is for language output, not reasoning. "
            "Reasoning is done by VULCAN's reasoning systems (symbolic, causal, etc.)."
        )
        systems_used.append("vulcan_local_llm_failed")
        
        # GRACEFUL DEGRADATION FIX: Provide a user-friendly error message
        # Generate a unique error reference for tracking
        import hashlib
        import time as time_module
        error_ref = hashlib.sha256(
            f"{time_module.time()}:{prompt[:50]}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an internal processing issue while reasoning about your request.\n\n"
            "This could be due to:\n"
            "• High system load causing a timeout\n"
            "• A complex query requiring more processing time\n"
            "• A temporary issue with the internal reasoning system\n\n"
            "**Suggestions:**\n"
            "• Please try rephrasing your question\n"
            "• Try breaking down complex questions into simpler parts\n"
            "• Wait a moment and try again\n\n"
            f"If this issue persists, please contact support with error reference: **{error_ref}**"
        )
        
        return {
            "text": error_text,
            "source": "error_graceful_degradation",
            "systems_used": systems_used,
            "error": True,
            "metadata": {
                "reason": "VULCAN internal LLM returned None or raised exception - graceful degradation active",
                "vulcan_llm_failed": True,
                "suggestion": "Rephrase question or try again later",
                "timeout_seconds": self.vulcan_timeout,
                "can_retry": True,
                "error_reference": error_ref,
            },
        }

    async def _execute_openai_first(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        DISABLED MODE: OpenAI-first mode is no longer available.
        This mode has been disabled because VULCAN should handle ALL reasoning internally.
        External AI usage is prohibited.
        """
        self.logger.error(
            "[HybridExecutor] ❌ openai_first mode is DISABLED! "
            "External AI usage is prohibited. VULCAN handles ALL reasoning internally. "
            "Falling back to local_first mode."
        )
        # Redirect to local_first mode instead
        return await self._execute_local_first(
            loop, prompt, max_tokens, temperature, system_prompt, conversation_history
        )

    async def _execute_parallel(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Execute using VULCAN internal LLM only - NO EXTERNAL AI.
        
        ARCHITECTURE: External AI has been removed. Only VULCAN internal LLM is used.
        The 'parallel' mode now simply delegates to local_first since there's nothing
        to run in parallel.
        """
        self.logger.info("[HybridExecutor] Parallel mode redirecting to local_first (external AI disabled)")
        return await self._execute_local_first(
            loop, prompt, max_tokens, temperature, system_prompt, conversation_history
        )

    async def _execute_ensemble(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Ensemble mode now delegates to local_first.
        
        ARCHITECTURE: OpenAI is NOT permitted for reasoning.
        Ensemble mode previously ran both LLMs, but now only VULCAN is allowed to reason.
        OpenAI can only be used for language polish (if enabled), not as a reasoning participant.
        """
        self.logger.info("[HybridExecutor] Ensemble mode redirecting to local_first (OpenAI reasoning prohibited)")
        return await self._execute_local_first(
            loop, prompt, max_tokens, temperature, system_prompt, conversation_history
        )

    # ============================================================
    # LLM CALL METHODS
    # ============================================================

    async def _call_local_llm(
        self, loop, prompt: str, max_tokens: int
    ) -> Optional[str]:
        """Call Vulcan's local LLM.
        
        ARCHITECTURE: VULCAN is primary brain, OpenAI is language fallback only.
        This method attempts to use VULCAN's internal LLM first.
        
        BUG #1 FIX: Added detailed error logging to expose why local model
        generation fails silently. Previously, exceptions were caught and
        logged at debug level, hiding the real cause of 100% OpenAI fallback.
        """
        import traceback
        import time as time_module
        
        # Check if local LLM should be skipped (default: False, VULCAN runs first)
        if _should_skip_local_llm():
            self.logger.warning(
                "[HybridExecutor] SKIP_LOCAL_LLM=true - VULCAN brain bypassed! "
                "Set SKIP_LOCAL_LLM=false to enable VULCAN reasoning."
            )
            return None
        
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
            self.logger.info(f"[HybridExecutor] Using HARD timeout: {self.vulcan_timeout}s")
            start_time = time_module.perf_counter()
            
            # HARD TIMEOUT FIX: Use ThreadPoolExecutor for TRUE hard timeout
            # asyncio.wait_for() only checks between await points
            # ThreadPoolExecutor.submit().result(timeout=X) fires even on sync blocks
            def generate_sync():
                return self.local_llm.generate(prompt, max_tokens)
            
            future = self._timeout_executor.submit(generate_sync)
            try:
                result = future.result(timeout=self.vulcan_timeout)
            except FuturesTimeoutError:
                self.logger.error(f"[HybridExecutor] ❌ HARD TIMEOUT after {self.vulcan_timeout}s!")
                future.cancel()
                return None
            
            elapsed = time_module.perf_counter() - start_time
            self.logger.info(f"[HybridExecutor] generate() returned in {elapsed:.2f}s")

            # Handle None result (returned when event loop conflict is detected)
            if result is None:
                self.logger.warning("[HybridExecutor] Local LLM returned None - triggering fallback")
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

    async def _call_local_llm_fast(
        self, loop, prompt: str, max_tokens: int
    ) -> Optional[str]:
        """Call Vulcan's local LLM in FAST OUTPUT FORMATTING MODE.
        
        This method uses generate_fast() which bypasses reasoning hooks for faster
        token generation. Use this when VULCAN's reasoning has already completed
        and the LLM is only needed to format output as prose.
        
        ARCHITECTURE:
        - VULCAN reasoning systems (the "mind") complete their work first
        - This LLM call is ONLY for formatting the result as natural language
        - No independent reasoning occurs - just prose generation
        
        PERFORMANCE:
        - Standard _call_local_llm(): ~2400ms first token (with reasoning hooks)
        - _call_local_llm_fast(): ~500ms first token (no reasoning hooks)
        - 30-token response: ~15s instead of TIMEOUT
        """
        import traceback
        import time as time_module
        
        if _should_skip_local_llm():
            self.logger.warning(
                "[HybridExecutor] SKIP_LOCAL_LLM=true in fast mode - bypassed"
            )
            return None
        
        if not self.local_llm:
            self.logger.error("[HybridExecutor] CRITICAL: local_llm is None in fast mode")
            return None
        
        # Check if generate_fast is available
        if not hasattr(self.local_llm, 'generate_fast'):
            self.logger.info(
                "[HybridExecutor] generate_fast() not available - falling back to standard generate()"
            )
            return await self._call_local_llm(loop, prompt, max_tokens)
        
        self.logger.info("=" * 60)
        self.logger.info("[HybridExecutor] FAST OUTPUT FORMATTING MODE")
        self.logger.info(f"[HybridExecutor] local_llm type: {type(self.local_llm).__name__}")
        
        try:
            self.logger.info(f"[HybridExecutor] Calling generate_fast(prompt_len={len(prompt)}, max_tokens={max_tokens})...")
            start_time = time_module.perf_counter()
            
            # Use generate_fast which skips reasoning hooks
            def generate_fast_sync():
                return self.local_llm.generate_fast(prompt, max_tokens)
            
            # Use hard timeout (should be faster than standard generate)
            # Fast mode uses FAST_MODE_MAX_TIMEOUT_SECONDS since no reasoning hooks run
            fast_timeout = min(self.vulcan_timeout, FAST_MODE_MAX_TIMEOUT_SECONDS)
            future = self._timeout_executor.submit(generate_fast_sync)
            
            try:
                result = future.result(timeout=fast_timeout)
            except FuturesTimeoutError:
                self.logger.error(f"[HybridExecutor] ❌ FAST MODE TIMEOUT after {fast_timeout}s!")
                future.cancel()
                return None
            
            elapsed = time_module.perf_counter() - start_time
            self.logger.info(f"[HybridExecutor] generate_fast() completed in {elapsed:.2f}s")
            
            if result is None:
                self.logger.warning("[HybridExecutor] Fast generation returned None")
                return None
            
            # Extract text from result
            if hasattr(result, "text"):
                self.logger.info(f"[HybridExecutor] ✓ FAST GENERATION SUCCEEDED ({len(result.text)} chars)")
                self.logger.info("=" * 60)
                return result.text
            elif isinstance(result, str):
                self.logger.info(f"[HybridExecutor] ✓ FAST GENERATION SUCCEEDED ({len(result)} chars)")
                self.logger.info("=" * 60)
                return result
            elif isinstance(result, dict) and "text" in result:
                self.logger.info(f"[HybridExecutor] ✓ FAST GENERATION SUCCEEDED ({len(result['text'])} chars)")
                self.logger.info("=" * 60)
                return result["text"]
            else:
                result_str = str(result)
                self.logger.info(f"[HybridExecutor] ✓ FAST GENERATION SUCCEEDED ({len(result_str)} chars, converted)")
                self.logger.info("=" * 60)
                return result_str
                
        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error("[HybridExecutor] FAST GENERATION FAILED!")
            self.logger.error(f"[HybridExecutor] Exception: {type(e).__name__}: {e}")
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

    async def _format_with_openai(
        self, 
        reasoning_output: "VulcanReasoningOutput",
        original_query: str,
        loop,
    ) -> Optional[str]:
        """
        Use OpenAI to format VULCAN's reasoning output as natural language.
        
        POLICY COMPLIANCE:
        - OpenAI receives VULCAN's completed reasoning (not the original query)
        - OpenAI does NOT reason independently
        - OpenAI ONLY converts structured data to prose
        
        This method implements the "hybrid output" pattern where:
        1. VULCAN's reasoning systems (the actual intelligence) complete their work
        2. OpenAI is used ONLY as a language formatter for the final prose
        
        Args:
            reasoning_output: The structured output from VULCAN's reasoning systems
            original_query: The user's original question (for context)
            loop: The asyncio event loop
            
        Returns:
            Formatted natural language response, or None if formatting fails
        """
        system_prompt = """You are a language formatter for VULCAN AI.

ROLE: Convert VULCAN's structured reasoning output into clear, natural language.

RULES:
- DO NOT perform independent reasoning or analysis
- DO NOT add information beyond what VULCAN provided
- DO NOT generate code unless VULCAN's result contains code
- ONLY format VULCAN's results as readable prose
- If VULCAN's result is mathematical, present it clearly with the answer
- If VULCAN's output indicates an error, explain it helpfully

FORMAT:
- Start directly with the answer or explanation
- Be concise but complete
- Use markdown formatting where appropriate (e.g., for code blocks, math)
"""

        # Build the user prompt with VULCAN's structured output
        try:
            output_dict = reasoning_output.to_dict()
            output_json = json.dumps(output_dict, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to serialize reasoning output: {e}")
            output_json = str(reasoning_output)

        user_prompt = f"""Format this VULCAN reasoning output for the user.

Original question: {original_query}

VULCAN's output:
{output_json}

Write a natural, helpful response based on VULCAN's results."""

        try:
            # Use OpenAI for fast and cheap formatting (currently gpt-3.5-turbo, configured in _call_openai)
            response = await self._call_openai(
                loop=loop,
                prompt=user_prompt,
                max_tokens=500,
                temperature=0.3,  # Low temp for consistent formatting
                system_prompt=system_prompt,
                use_cache=True,
            )
            
            if response and len(response.strip()) > self.MIN_MEANINGFUL_LENGTH:
                self.logger.info("[HybridExecutor] ✓ OpenAI formatted VULCAN's structured output")
                return response
            else:
                self.logger.warning("[HybridExecutor] OpenAI returned empty/short response for formatting")
                return None
                
        except Exception as e:
            self.logger.warning(f"[HybridExecutor] OpenAI formatting failed: {e}")
            return None

    def _format_structured_output_sync(
        self,
        reasoning_output: "VulcanReasoningOutput",
    ) -> str:
        """
        Format VULCAN's structured output as plain text (fallback when OpenAI unavailable).
        
        This provides a simple, reliable fallback that doesn't require any external API.
        
        Args:
            reasoning_output: The structured output from VULCAN's reasoning systems
            
        Returns:
            Plain text formatted response
        """
        if not reasoning_output.success:
            return f"I encountered an issue: {reasoning_output.error or 'Unknown error'}"
        
        result = reasoning_output.result
        result_type = reasoning_output.result_type
        
        # Format based on result type
        if result_type == "mathematical":
            if reasoning_output.confidence >= 0.9:
                return f"The answer is: **{result}**"
            else:
                return f"The calculated result is: **{result}** (confidence: {reasoning_output.confidence:.0%})"
        
        elif result_type == "symbolic":
            return f"Based on symbolic reasoning: {result}"
        
        elif result_type == "factual":
            return str(result)
        
        elif result_type == "causal":
            return f"Based on causal analysis: {result}"
        
        else:
            # Generic formatting
            if isinstance(result, dict):
                try:
                    return json.dumps(result, indent=2)
                except Exception:
                    return str(result)
            return str(result)

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

    def _capture_polish_for_distillation(
        self,
        prompt: str,
        internal_output: str,
        teacher_output: str,
    ) -> bool:
        """
        Capture training example for Internal LLM to learn from OpenAI polish.
        
        When OpenAI polishes Internal LLM output, we capture the pair:
        - Student input: The prompt + Internal LLM's raw output
        - Teacher output: OpenAI's polished version
        
        Over time, Internal LLM learns to produce polished output directly,
        reducing OpenAI dependency.
        
        Args:
            prompt: The original user prompt
            internal_output: What Internal LLM generated (student)
            teacher_output: What OpenAI polished it to (teacher)
            
        Returns:
            True if captured, False if skipped/failed
        """
        # Skip if outputs are too similar (nothing to learn)
        if internal_output.strip() == teacher_output.strip():
            self.logger.debug("[Distillation] Skipping - outputs identical")
            return False
        
        # Skip very short outputs
        if len(internal_output.strip()) < 20 or len(teacher_output.strip()) < 20:
            self.logger.debug("[Distillation] Skipping - outputs too short")
            return False
        
        example = {
            "prompt": prompt,
            "internal_output": internal_output,
            "teacher_output": teacher_output,
            "timestamp": time.time(),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "capture_type": "polish_learning",
        }
        
        # Try real distillation system first
        try:
            distiller = get_knowledge_distiller()
            if distiller is not None:
                # Use existing distillation pipeline
                captured = distiller.capture_response(
                    prompt=prompt,
                    openai_response=teacher_output,
                    local_response=internal_output,
                    metadata={
                        "capture_type": "polish_learning",
                        "mode": self.mode,
                    },
                )
                if captured:
                    self.logger.info(f"[Distillation] ✓ Captured polish example: {example['prompt_hash']}")
                    return True
                else:
                    self.logger.debug("[Distillation] Example rejected by quality filters")
                    return False
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"[Distillation] Real distiller failed: {e}")
        
        # Fallback: queue locally for batch processing
        self._distillation_queue.append(example)
        self.logger.info(
            f"[Distillation] Queued locally: {len(self._distillation_queue)} examples "
            f"(hash={example['prompt_hash']})"
        )
        return True

    def get_distillation_queue(self) -> List[Dict[str, Any]]:
        """
        Get queued distillation examples for batch training.
        
        Retrieves and clears the local queue of polish training examples.
        These can be used to train Internal LLM to produce polished outputs.
        
        Returns:
            List of distillation examples, each containing:
            - prompt: Original user prompt
            - internal_output: What Internal LLM generated
            - teacher_output: What OpenAI polished it to
            - timestamp: When captured
            - prompt_hash: Hash for deduplication
        """
        queue = self._distillation_queue.copy()
        self._distillation_queue.clear()
        if queue:
            self.logger.info(f"[Distillation] Retrieved {len(queue)} examples from queue")
        return queue

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
        Get execution statistics including cache and distillation statistics.
        
        Returns:
            Dictionary with execution, cache, and distillation statistics
        """
        stats = {
            "total_executions": self._execution_count,
            "local_successes": self._local_successes,
            "openai_successes": self._openai_successes,
            "failures": self._failures,
            "mode": self.mode,
            "has_local_llm": self.local_llm is not None,
            "openai_cache_enabled": self._enable_openai_cache,
            "distillation_enabled": self._distillation_enabled,
            "distillation_queue_size": len(self._distillation_queue),
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
        
        Args:
            prompt: The input prompt for generation
            context: Optional context dictionary (currently unused, for API compatibility)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text from the local LLM
            
        Raises:
            RuntimeError: If the local model is not initialized
        """
        if self.local_llm is None:
            raise RuntimeError("No local model")
        
        self.logger.info(f"[HybridExecutor] Calling local model...")
        response = self.local_llm.generate(prompt, max_tokens)
        self.logger.info(f"[HybridExecutor] ✓ Success")
        return response

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
