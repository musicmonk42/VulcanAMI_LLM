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
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Industry Standard: Import LLMMode from router for type safety and consistency
try:
    from vulcan.routing.query_router import LLMMode
    LLM_MODE_AVAILABLE = True
except ImportError:
    try:
        from src.vulcan.routing.query_router import LLMMode
        LLM_MODE_AVAILABLE = True
    except ImportError:
        # Fallback: Define placeholder for backward compatibility
        LLMMode = None
        LLM_MODE_AVAILABLE = False
        logger.debug("LLMMode not available - will use legacy mode parameter")

# Module metadata
__version__ = "1.7.0"  # P0 FIX: Added NotReasoningEngineError and reasoning task detection
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class NotReasoningEngineError(ValueError):
    """
    Raised when HybridLLMExecutor is asked to perform reasoning.
    
    ARCHITECTURE: HybridLLMExecutor is for LANGUAGE GENERATION only, not reasoning.
    If this error is raised, the caller is misusing the executor by asking it
    to reason about or solve a problem instead of formatting/paraphrasing 
    already-computed reasoning results.
    
    CORRECT USAGE:
        1. VULCAN's reasoning engines (symbolic, causal, mathematical) process the query
        2. Pass the ReasoningResult to HybridLLMExecutor for language formatting
        
    INCORRECT USAGE:
        - Passing raw user queries directly to HybridLLMExecutor expecting it to "think"
        - Using LLM as a reasoning fallback when VULCAN reasoning fails
    
    Example:
        >>> executor = HybridLLMExecutor()
        >>> # WRONG - asking LLM to reason:
        >>> result = await executor.execute("Solve: what is 2+2?")  # Raises NotReasoningEngineError
        >>> # RIGHT - asking LLM to format reasoning output:
        >>> result = await executor.format_output_for_user(
        ...     reasoning_output={"result": 4, "method": "arithmetic"},
        ...     original_prompt="What is 2+2?"
        ... )
    """
    pass


# ============================================================
# REASONING TASK DETECTION
# ============================================================

# Patterns that indicate a reasoning/problem-solving request (not formatting)
REASONING_TASK_INDICATORS = [
    "solve", "calculate", "compute", "figure out", "work out",
    "what is the answer", "what's the answer", "find the solution",
    "prove", "derive", "demonstrate", "show that",
    "why does", "why is", "why do", "explain why",
    "analyze", "evaluate", "assess",
    "how many", "how much", "how long", "how far",
]


def _is_reasoning_task(prompt: str) -> bool:
    """
    Detect if a prompt is asking for reasoning rather than formatting.
    
    This function identifies prompts that are requesting the LLM to think,
    reason, or solve problems - which is NOT the role of the LLM in VULCAN's
    architecture. LLMs should only format/paraphrase reasoning results.
    
    Args:
        prompt: The input prompt to check
        
    Returns:
        True if the prompt appears to be a reasoning task, False otherwise
    """
    prompt_lower = prompt.lower().strip()
    
    # Check for reasoning task indicator patterns
    for indicator in REASONING_TASK_INDICATORS:
        if indicator in prompt_lower:
            return True
    
    # Check for direct mathematical expressions (common reasoning bypass attempt)
    # e.g., "2+2=?", "5*3", "sqrt(16)"
    import re
    math_pattern = r'\d+\s*[\+\-\*/\^]\s*\d+'
    if re.search(math_pattern, prompt_lower):
        return True
    
    return False


# ============================================================
# CONFIGURATION
# ============================================================
# Default path for GraphixVulcanLLM config, can be overridden via environment variable
LLM_CONFIG_PATH = os.environ.get("VULCAN_LLM_CONFIG_PATH", "configs/llm_config.yaml")

# NOTE: OPENAI_FALLBACK_CONFIDENCE is kept for backwards compatibility but
# OpenAI reasoning fallback is now disabled. This constant is only used
# if language polish mode is enabled.
OPENAI_FALLBACK_CONFIDENCE = 0.6

# ============================================================
# OPENAI LANGUAGE FORMATTING MODE
# ============================================================
# When OPENAI_LANGUAGE_FORMATTING=true:
#   - Route ALL natural language output formatting to OpenAI (gpt-4o-mini)
#   - VULCAN's reasoning systems still do ALL thinking/reasoning
#   - OpenAI is used ONLY for converting structured output to natural language
#   - Every (input, output) pair is captured for distillation training
#
# Benefits:
#   - Fast response times (~2-5 seconds vs 60+ seconds with internal LLM on CPU)
#   - VULCAN LLM learns from OpenAI outputs via distillation over time
#   - Eventually: mostly local, fast, private as VULCAN LLM improves
#
# POLICY CONSTRAINTS:
#   - OpenAI MUST NOT reason independently — only format VULCAN's output
#   - OpenAI MUST NOT generate code
#   - All reasoning happens in VULCAN's mind BEFORE OpenAI is called
#
# Note: Default to "true" to match .env.example documentation and prevent 60-second timeouts.
# This enables fast OpenAI formatting (~2-5s) while VULCAN LLM learns via distillation.
_openai_language_formatting_env = os.environ.get("OPENAI_LANGUAGE_FORMATTING", "true").lower()
OPENAI_LANGUAGE_FORMATTING = _openai_language_formatting_env in ("true", "1", "yes")

# Legacy polish mode - kept for backwards compatibility
_openai_language_polish_env = os.environ.get("OPENAI_LANGUAGE_POLISH", "false").lower()
OPENAI_LANGUAGE_POLISH = _openai_language_polish_env in ("true", "1", "yes")

# ARCHITECTURE: VULCAN does ALL reasoning. OpenAI is for language output ONLY.
# 
# The correct flow is:
# 1. VULCAN reasoning engines analyze query (symbolic, probabilistic, causal, math)
# 2. If OPENAI_LANGUAGE_FORMATTING=true:
#    - OpenAI (gpt-4o-mini) formats the reasoning output as natural language
#    - Response is captured for distillation
# 3. Otherwise:
#    - GraphixVulcanLLM generates response using VULCAN's internal model
#    - If OPENAI_LANGUAGE_POLISH=true, OpenAI can polish the output
# 4. If VULCAN fails -> return error (NO OpenAI reasoning fallback)
#
# SKIP_LOCAL_LLM is deprecated - VULCAN must always run for reasoning.
_skip_local_llm_env = os.environ.get("SKIP_LOCAL_LLM", "false").lower()
SKIP_LOCAL_LLM = _skip_local_llm_env in ("true", "1", "yes")

# ============================================================
# TIMEOUT CONFIGURATION - INCREASED FOR CPU EXECUTION
# ============================================================
# Note: Increased timeouts to prevent premature timeouts during CPU-intensive
# language generation. The internal LLM can take 3+ seconds per token on CPU.
# CPU Cloud Fix: At ~500ms per token, generating 120 tokens takes 60s which was
# causing TimeoutError. Increased to 300s (5 minutes) to allow ~600 tokens on CPU.
VULCAN_HARD_TIMEOUT = float(os.environ.get("VULCAN_LLM_HARD_TIMEOUT", "300.0"))  # 5 minutes (was 120s)
PER_TOKEN_TIMEOUT = float(os.environ.get("VULCAN_LLM_PER_TOKEN_TIMEOUT", "30.0"))  # 30s per token (was 10s)

# Fast mode timeout for output formatting (when reasoning is already done)
# This can be shorter since no reasoning hooks run per-token
FAST_MODE_MAX_TIMEOUT_SECONDS = float(os.environ.get("VULCAN_LLM_FAST_TIMEOUT", "60.0"))  # 60 seconds

# ============================================================
# Note: PARALLEL MODE GRACE PERIOD
# ============================================================
# When OpenAI finishes first in parallel mode, wait this long for local to complete
# This prevents cancelling local when it's almost done
LOCAL_GRACE_PERIOD_SECONDS = float(os.environ.get("VULCAN_LOCAL_GRACE_PERIOD", "2.0"))  # 2 seconds

# ============================================================
# ADAPTIVE TIMEOUT CALCULATION
# ============================================================
# Note: Implement adaptive timeout to prevent timeouts during CPU inference.
# Formula: timeout = BASE_TIMEOUT_SECONDS + (max_tokens * TIMEOUT_PER_TOKEN_SECONDS)
# This dynamically scales the timeout based on the expected generation length.
BASE_TIMEOUT_SECONDS = float(os.environ.get("VULCAN_BASE_TIMEOUT", "5.0"))  # 5 seconds base
TIMEOUT_PER_TOKEN_SECONDS = float(os.environ.get("VULCAN_TIMEOUT_PER_TOKEN", "2.0"))  # 2 seconds per token


def calculate_adaptive_timeout(max_tokens: int) -> float:
    """
    Calculate adaptive timeout based on token count.
    
    ADAPTIVE TIMEOUT FIX: This function implements Claude's adaptive timeout formula
    to prevent premature timeouts during CPU-bound inference.
    
    Formula: timeout = base_timeout (5s) + (max_tokens * 2.0s)
    
    Examples:
        - 10 tokens: 5 + (10 * 2.0) = 25 seconds
        - 50 tokens: 5 + (50 * 2.0) = 105 seconds
        - 100 tokens: 5 + (100 * 2.0) = 205 seconds
    
    Args:
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Calculated timeout in seconds, capped at VULCAN_HARD_TIMEOUT
    """
    adaptive_timeout = BASE_TIMEOUT_SECONDS + (max_tokens * TIMEOUT_PER_TOKEN_SECONDS)
    # Cap at hard timeout to prevent excessive waits
    return min(adaptive_timeout, VULCAN_HARD_TIMEOUT)


# ============================================================
# CPU CLOUD EXECUTION - MAX TOKENS LIMIT
# ============================================================
# Note: Limit max tokens for CPU execution to prevent timeout issues.
# At 500ms per token, max_tokens=50 takes ~25s, ensuring completion before timeout.
# This can be overridden via environment variable for GPU environments.
CPU_MAX_TOKENS_DEFAULT = int(os.environ.get("VULCAN_CPU_MAX_TOKENS", "50"))

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
# UTILITY FUNCTIONS
# ============================================================

def _get_task_name(task: asyncio.Task) -> str:
    """Get the name of an asyncio Task safely."""
    return task.get_name() if hasattr(task, 'get_name') else "unknown"


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
    # TASK 2 FIX: Added 'reasoning_first' mode that prioritizes reasoning results
    # Note: Added 'openai_only', 'local_only', and 'sequential' modes for explicit operation
    # 'sequential' = Try OpenAI first (fast), fallback to local LLM if OpenAI fails
    VALID_MODES = ("local_first", "openai_first", "parallel", "ensemble", "reasoning_first", "openai_only", "local_only", "sequential")
    
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
    # FIX (Issue #ROUTING-001): Content Preservation Prompt
    # ============================================================
    # When VULCAN's reasoning systems produce a response (e.g., WorldModel's
    # self-awareness answer "YES"), OpenAI must preserve that exact answer
    # and NOT replace it with generic AI disclaimers.
    #
    # This prompt enforces strict content preservation rules to prevent
    # OpenAI from overriding VULCAN's reasoning with generic responses.
    VULCAN_CONTENT_PRESERVATION_PROMPT = (
        "You are formatting VULCAN's reasoning output into natural language.\n\n"
        "CRITICAL RULES - MANDATORY ENFORCEMENT:\n"
        "1. PRESERVE all factual claims and conclusions from VULCAN's response\n"
        "2. If VULCAN says 'YES' or 'NO', your output MUST contain that exact answer\n"
        "3. NEVER replace VULCAN's answer with generic AI disclaimers\n"
        "4. NEVER say 'As an AI assistant, I don't have consciousness...'\n"
        "5. NEVER contradict or dilute VULCAN's stated position\n"
        "6. Only improve grammar, readability, and flow\n"
        "7. Keep the same reasoning structure and conclusions\n"
        "8. If VULCAN expresses a preference or choice, preserve it exactly\n\n"
        "Your job is FORMATTING ONLY - not fact-checking or censoring VULCAN's output.\n"
        "Present VULCAN's reasoning clearly and faithfully."
    )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        local_llm: Optional[Any] = None,
        openai_client_getter: Optional[Callable] = None,
        mode: str = "parallel",
        timeout: Optional[float] = None,  # Changed to allow env var override
        ensemble_min_confidence: float = 0.7,
        openai_max_tokens: int = 2000,  # Increased for diagnostic purposes
        enable_openai_cache: bool = True,
        openai_cache_max_size: int = 1000,
        openai_cache_ttl_seconds: int = 3600,  # 1 hour default
        prefer_reasoning: bool = True,  # TASK 2 FIX: Prefer reasoning engine results
        reasoning_confidence_threshold: float = 0.25,  # BUG #5 FIX: Lowered from 0.5 to 0.25 for partial reasoning success
    ):
        """
        Initialize the hybrid executor.

        Args:
            local_llm: Vulcan's local LLM instance
            openai_client_getter: Function to get OpenAI client (lazy loading)
            mode: Execution mode (local_first, openai_first, parallel, ensemble, reasoning_first)
            timeout: Timeout for parallel/ensemble execution in seconds.
                     If None, uses HYBRID_EXECUTOR_TIMEOUT env var or default 30.0
            ensemble_min_confidence: Minimum confidence for ensemble selection
            openai_max_tokens: Maximum tokens for OpenAI API calls
            enable_openai_cache: Enable caching of OpenAI responses (default: True)
            openai_cache_max_size: Maximum cache entries (default: 1000)
            openai_cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
            prefer_reasoning: If True, skip LLM when reasoning results have high confidence
            reasoning_confidence_threshold: Minimum confidence for reasoning results to bypass LLM
        """
        self.local_llm = local_llm
        self.openai_client_getter = openai_client_getter or get_openai_client
        
        # TASK 2 FIX: Reasoning preference configuration
        self.prefer_reasoning = prefer_reasoning
        self.reasoning_confidence_threshold = reasoning_confidence_threshold
        
        # Note: Check if OpenAI client is available for mode override
        # Local LLM (GraphixVulcanLLM) times out (~120s) on CPU - use OpenAI only for language generation
        # Reasoning engines in src/vulcan/reasoning/* are unaffected - they still do all the thinking
        self.openai_client = self.openai_client_getter()
        
        # Validate and set mode
        mode_lower = mode.lower()
        if mode_lower not in self.VALID_MODES:
            self.logger = logging.getLogger("HybridLLMExecutor")
            self.logger.warning(
                f"Invalid mode '{mode}', defaulting to 'parallel'. Valid modes: {self.VALID_MODES}"
            )
            mode_lower = "parallel"
        
        # Note: Override mode based on backend availability for language generation
        # Local LLM is too slow (~120s timeout) - use sequential mode (OpenAI first, local fallback)
        # Note: Reasoning systems (symbolic, causal, probabilistic) are UNAFFECTED - they still run
        if mode_lower == "parallel":
            # Use sequential mode: try OpenAI first, local LLM as fallback
            mode_lower = "sequential"
            # Initialize logger early for this message
            self.logger = logging.getLogger("HybridLLMExecutor")
            self.logger.info(
                "[HybridExecutor] Mode: sequential (OpenAI first, local LLM fallback). "
                "Reasoning engines still handle all thinking."
            )
        
        self.mode = mode_lower
        
        # Allow environment variable override for timeout (Issue #5 fix)
        # Added error handling for invalid env var values
        try:
            default_timeout = float(os.environ.get("HYBRID_EXECUTOR_TIMEOUT", "30.0"))
        except (ValueError, TypeError):
            default_timeout = 30.0
        self.timeout = timeout if timeout is not None else default_timeout
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
        # Note: Use VULCAN_HARD_TIMEOUT constant (default 300s) for CPU-intensive reasoning
        # CPU CLOUD FIX: Increased from 120s to 300s to allow more tokens before timeout
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
        
        # TASK 2 FIX: Log reasoning preference settings
        if self.prefer_reasoning:
            self.logger.info(
                f"[HybridExecutor] Reasoning preference ENABLED "
                f"(threshold={self.reasoning_confidence_threshold})"
            )
        
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
        self._reasoning_direct_count = 0  # TASK 2 FIX: Track reasoning direct uses
        
        # Distillation queue for capturing polish training examples
        # When OpenAI polishes Internal LLM output, we capture the pair for training
        self._distillation_queue: List[Dict[str, Any]] = []
        self._distillation_enabled = os.environ.get("ENABLE_DISTILLATION", "true").lower() in ("true", "1", "yes")
        if self._distillation_enabled:
            self.logger.info("[HybridExecutor] Distillation capture enabled")

    def _has_openai_key(self) -> bool:
        """
        Check if OpenAI API key is available.
        
        Industry Standard: Helper method for conditional behavior based on API availability.
        
        Returns:
            True if OpenAI client is available, False otherwise
        """
        return self.openai_client is not None

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
        llm_mode: Optional[Union[str, "LLMMode"]] = None,
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
            llm_mode: Optional LLM execution mode from router (FORMAT_ONLY, GENERATE, ENHANCE).
                     Industry Standard: Router decides LLM behavior, executor respects it.
                     If None, uses legacy self.mode behavior with deprecation warning.

        Returns:
            Dict with 'text', 'source', 'systems_used', and optional 'metadata'
        """
        self._execution_count += 1
        loop = asyncio.get_running_loop()
        
        # ============================================================
        # P0 FIX: STOP LLM-AS-REASONER BYPASS
        # ============================================================
        # HybridLLMExecutor is for LANGUAGE GENERATION only, not reasoning.
        # If the prompt appears to be a reasoning task (not a formatting request),
        # we reject it with a clear error message.
        #
        # CORRECT FLOW:
        # 1. VULCAN reasoning engines process the query
        # 2. Pass the VulcanReasoningOutput to format_output_for_user() or
        #    execute_with_structured_output() for language formatting
        #
        # This check can be disabled for specific use cases by passing
        # llm_mode=LLMMode.GENERATE (for creative/open-ended queries)
        #
        # NOTE: We only apply this check when llm_mode is None (legacy usage)
        # or FORMAT_ONLY. GENERATE and ENHANCE modes are intentionally creative.
        should_check_reasoning_bypass = (
            llm_mode is None or
            (LLM_MODE_AVAILABLE and 
             isinstance(llm_mode, str) and 
             llm_mode.upper() == "FORMAT_ONLY") or
            (LLM_MODE_AVAILABLE and 
             hasattr(llm_mode, 'value') and 
             llm_mode == LLMMode.FORMAT_ONLY if LLM_MODE_AVAILABLE else False)
        )
        
        if should_check_reasoning_bypass and _is_reasoning_task(prompt):
            self.logger.warning(
                f"[HybridExecutor] P0 VIOLATION: Reasoning task detected in prompt. "
                f"LLMs should NOT reason - use VULCAN reasoning engines first, then "
                f"call format_output_for_user() with the reasoning result."
            )
            raise NotReasoningEngineError(
                f"HybridLLMExecutor detected a reasoning task but LLMs are for "
                f"LANGUAGE GENERATION only, not reasoning. The prompt appears to "
                f"request computation/analysis: '{prompt[:100]}...'\n\n"
                f"CORRECT USAGE:\n"
                f"1. Process query with VULCAN reasoning engines first\n"
                f"2. Call format_output_for_user(reasoning_output, original_prompt)\n\n"
                f"If this is intentionally a creative/open-ended query, pass "
                f"llm_mode=LLMMode.GENERATE to bypass this check."
            )
        
        # ARCHITECTURE: Respect llm_mode from caller (router)
        # Industry Standard: Single source of truth - router decides, executor executes
        if llm_mode is not None:
            # Convert string to LLMMode enum if needed
            if LLM_MODE_AVAILABLE and isinstance(llm_mode, str):
                try:
                    llm_mode = LLMMode(llm_mode)
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid llm_mode '{llm_mode}', falling back to self.mode")
                    llm_mode = None
            
            # Map LLMMode to execution strategy
            if LLM_MODE_AVAILABLE and isinstance(llm_mode, type(LLMMode.FORMAT_ONLY if LLM_MODE_AVAILABLE else None)):
                if llm_mode == LLMMode.FORMAT_ONLY:
                    # Format only: Minimal LLM usage, just format output
                    effective_mode = "local_first"
                elif llm_mode == LLMMode.GENERATE:
                    # Generate: LLM creates content (creative queries)
                    effective_mode = "openai_first" if self._has_openai_key() else "local_first"
                elif llm_mode == LLMMode.ENHANCE:
                    # Enhance: LLM enhances simple responses
                    effective_mode = "openai_first" if self._has_openai_key() else "local_first"
                else:
                    effective_mode = self.mode
                    
                logger.debug(f"[HybridExecutor] llm_mode={llm_mode.value} → effective_mode={effective_mode}")
            else:
                effective_mode = self.mode
        else:
            # DEPRECATED: Using self.mode when llm_mode not provided
            # Industry Standard: Deprecation warnings for migration path
            if self._execution_count == 1:  # Log once per executor instance
                warnings.warn(
                    "HybridLLMExecutor.execute() called without llm_mode parameter. "
                    "This is deprecated. Router should pass llm_mode to control LLM behavior. "
                    "Falling back to self.mode for backward compatibility.",
                    DeprecationWarning,
                    stacklevel=2
                )
            effective_mode = self.mode
        
        # Use default system prompt if none provided
        # MEMORY FIX: Default prompt now allows conversation memory
        effective_system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

        if effective_mode == "local_first":
            result = await self._execute_local_first(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif effective_mode == "openai_first":
            result = await self._execute_openai_first(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif effective_mode == "parallel":
            result = await self._execute_parallel(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif effective_mode == "ensemble":
            result = await self._execute_ensemble(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif effective_mode == "openai_only":
            # Note: OpenAI-only mode for fast language generation (~3s instead of 120s timeout)
            # Reasoning engines still handle all thinking - this is just for language output
            result = await self._execute_openai_only(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif effective_mode == "local_only":
            # Local-only mode when OpenAI is unavailable
            result = await self._execute_local_only(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        elif effective_mode == "sequential":
            # Sequential mode: try OpenAI first (fast), fallback to local LLM if fails
            result = await self._execute_sequential(
                loop, prompt, max_tokens, temperature, effective_system_prompt, conversation_history
            )
        else:
            self.logger.warning(f"Unknown mode '{effective_mode}', defaulting to openai_first")
            result = await self._execute_openai_first(
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
        # Priority: explicit parameter > OPENAI_LANGUAGE_FORMATTING > OPENAI_LANGUAGE_POLISH
        if use_openai_formatting is None:
            # Use module-level constants (evaluated at import time from env vars)
            use_openai_formatting = OPENAI_LANGUAGE_FORMATTING or OPENAI_LANGUAGE_POLISH
        
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

    async def format_output_for_user(
        self,
        reasoning_output: Dict[str, Any],
        original_prompt: str,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Route VULCAN's reasoning output to OpenAI for language formatting.
        
        This is the PRIMARY entry point for the new OpenAI Language Formatting
        architecture. VULCAN's reasoning is ALREADY COMPLETE at this point.
        This method ONLY handles OUTPUT FORMATTING.
        
        ARCHITECTURE:
            VULCAN Mind (orchestrator, agents, symbolic reasoning)
                │
                │ produces
                ▼
            Structured Output (dict with result, confidence, method, etc.)
                │
                │ sent to
                ▼
            OpenAI gpt-4o-mini (language formatting ONLY)
                │
                │ produces
                ▼
            Natural Language Response → returned to user
                │
                │ captured as training pair
                ▼
            Distillation Store → VULCAN LLM learns async
        
        POLICY CONSTRAINTS:
        - OpenAI MUST NOT reason independently — only format VULCAN's output
        - OpenAI MUST NOT generate code
        - All reasoning happens in VULCAN's mind BEFORE this is called
        - Every (input, output) pair is captured for distillation
        
        Args:
            reasoning_output: VULCAN's structured reasoning output (dict with result,
                            method, confidence, reasoning_trace, error, etc.)
            original_prompt: The user's original question (for context in formatting)
            max_tokens: Maximum tokens for response (default: 500)
            
        Returns:
            Dict with:
            - text: Formatted natural language response
            - source: "openai_formatting" or "internal_formatting"
            - systems_used: List of systems used
            - metadata: Additional context including reasoning output
            - distillation_captured: Whether response was captured for training
        """
        loop = asyncio.get_running_loop()
        systems_used = ["vulcan_reasoning"]
        
        # Convert dict to VulcanReasoningOutput if needed
        if isinstance(reasoning_output, dict):
            structured_output = VulcanReasoningOutput(
                query_id=reasoning_output.get("query_id", hashlib.sha256(original_prompt.encode()).hexdigest()[:16]),
                success=reasoning_output.get("success", True),
                result=reasoning_output.get("result"),
                result_type=reasoning_output.get("result_type", "unknown"),
                method_used=reasoning_output.get("method", "unknown"),
                confidence=reasoning_output.get("confidence", 0.0),
                reasoning_trace=reasoning_output.get("reasoning_trace", []),
                error=reasoning_output.get("error"),
                metadata=reasoning_output.get("metadata", {}),
            )
        elif isinstance(reasoning_output, VulcanReasoningOutput):
            structured_output = reasoning_output
        else:
            self.logger.warning(
                f"[HybridExecutor] Unexpected reasoning_output type: {type(reasoning_output).__name__}"
            )
            structured_output = VulcanReasoningOutput(
                query_id=hashlib.sha256(original_prompt.encode()).hexdigest()[:16],
                success=True,
                result=str(reasoning_output),
                result_type="converted",
                method_used="unknown",
                confidence=0.5,
            )
        
        # Check if reasoning succeeded
        if not structured_output.success:
            error_text = self._format_reasoning_error(structured_output)
            return {
                "text": error_text,
                "source": "vulcan_reasoning_error",
                "systems_used": systems_used,
                "error": True,
                "distillation_captured": False,
                "metadata": {
                    "reasoning_output": structured_output.to_dict(),
                    "query": original_prompt,
                },
            }
        
        # Use OpenAI for formatting (fast path)
        distillation_captured = False
        try:
            formatted = await self._format_with_openai_for_output(
                reasoning_output=structured_output,
                original_query=original_prompt,
                loop=loop,
            )
            
            if formatted and len(formatted.strip()) > self.MIN_MEANINGFUL_LENGTH:
                systems_used.append("openai_formatting")
                
                # Note: _format_with_openai_for_output already handles distillation capture
                distillation_captured = self._distillation_enabled
                
                self.logger.info("[HybridExecutor] ✓ OpenAI formatted VULCAN's output (fast path: ~2-5s)")
                
                return {
                    "text": formatted,
                    "source": "openai_formatting",
                    "systems_used": systems_used,
                    "distillation_captured": distillation_captured,
                    "metadata": {
                        "reasoning_output": structured_output.to_dict(),
                        "query": original_prompt,
                        "openai_model": "gpt-4o-mini",
                        "openai_role": "formatting_only",
                    },
                }
        except Exception as e:
            self.logger.warning(f"[HybridExecutor] OpenAI formatting failed: {e}, using internal fallback")
        
        # Fallback to internal formatting (no external API)
        formatted = self._format_structured_output_sync(structured_output)
        systems_used.append("internal_formatting")
        
        return {
            "text": formatted,
            "source": "internal_formatting",
            "systems_used": systems_used,
            "distillation_captured": distillation_captured,
            "metadata": {
                "reasoning_output": structured_output.to_dict(),
                "query": original_prompt,
                "fallback_reason": "openai_unavailable",
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
    # LANGUAGE INTERFACE METHODS (Parse IN, Format OUT)
    # ============================================================

    async def parse_natural_language_query(
        self,
        user_text: str,
    ) -> "StructuredQuery":
        """
        Language interface IN: Convert natural language → structured query.
        
        This is the FIRST step - understanding what the user wants.
        The LLM parses intent and extracts parameters, but does NOT answer.
        
        ARCHITECTURE:
            This implements the "Language Interface IN" layer. The LLM's job is
            ONLY to understand the user's intent and extract parameters. It does
            NOT compute, reason, or answer. That's VULCAN's job.
        
        Args:
            user_text: Raw user input like "What's the derivative of x²?"
            
        Returns:
            StructuredQuery with parsed intent and parameters
        """
        # Import here to avoid circular dependency
        from vulcan.llm.query_parser import QueryIntent, QueryDomain, StructuredQuery
        
        system_prompt = '''Parse user queries into structured format for a reasoning system.

Output JSON only with these fields:
- intent: compute|explain|search|analyze|plan|compare|unknown
- domain: math|logic|causal|general|code
- parameters: extracted values needed for computation
- confidence: 0.0-1.0 how confident you are in the parsing

Examples:
Input: "What's 2 plus 2?"
Output: {"intent": "compute", "domain": "math", "parameters": {"operation": "add", "operands": [2, 2]}, "confidence": 0.95}

Input: "What's the derivative of x squared?"
Output: {"intent": "compute", "domain": "math", "parameters": {"operation": "derivative", "expression": "x^2", "variable": "x"}, "confidence": 0.9}

Input: "Explain why the sky is blue"
Output: {"intent": "explain", "domain": "general", "parameters": {"topic": "sky color", "phenomenon": "blue sky"}, "confidence": 0.85}

Output ONLY valid JSON, no other text.'''

        loop = asyncio.get_running_loop()
        
        result = await self._call_openai(
            loop=loop,
            prompt=user_text,
            max_tokens=200,
            temperature=0.0,  # Deterministic parsing
            system_prompt=system_prompt
        )
        
        if result:
            try:
                return StructuredQuery.from_json(result, original_text=user_text)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.warning(f"Failed to parse LLM output as StructuredQuery: {e}")
        
        # Fallback: return unknown query
        return StructuredQuery(
            intent=QueryIntent.UNKNOWN,
            domain=QueryDomain.GENERAL,
            parameters={"raw_text": user_text},
            original_text=user_text,
            confidence=0.0
        )

    async def execute_with_language_interface(
        self,
        user_text: str,
        vulcan_reasoning_fn: Optional[Callable] = None,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute with proper language interface architecture.
        
        Three-step flow:
        1. Language IN: Parse natural language → structured query
        2. Reasoning: VULCAN computes (NO LLM answering)
        3. Language OUT: Format result → natural language
        
        ARCHITECTURE:
            This method enforces the correct LLM usage pattern:
            - LLM parses user intent (Language IN)
            - VULCAN reasoning engines compute the answer (NO LLM)
            - LLM formats the result (Language OUT)
            
            The LLM NEVER answers questions directly. It only understands
            questions and formats answers that VULCAN computed.
        
        Args:
            user_text: Raw natural language input from user
            vulcan_reasoning_fn: Async function that takes StructuredQuery and returns VulcanReasoningOutput
            max_tokens: Max tokens for output formatting
            
        Returns:
            Dict with formatted response and metadata
        """
        # Import here to avoid circular dependency
        from vulcan.llm.query_parser import StructuredQuery
        
        systems_used = []
        
        # STEP 1: Language interface IN - Parse query
        self.logger.info("[LangInterface] Step 1: Parsing natural language query")
        structured_query = await self.parse_natural_language_query(user_text)
        systems_used.append("llm_input_parsing")
        
        self.logger.info(
            f"[LangInterface] Parsed: intent={structured_query.intent.value}, "
            f"domain={structured_query.domain.value}, confidence={structured_query.confidence}"
        )
        
        # STEP 2: VULCAN Reasoning - NO LLM here
        reasoning_output = None
        if vulcan_reasoning_fn:
            self.logger.info("[LangInterface] Step 2: Executing VULCAN reasoning")
            try:
                reasoning_output = await vulcan_reasoning_fn(structured_query)
                systems_used.append(reasoning_output.method_used if reasoning_output else "vulcan_reasoning")
            except Exception as e:
                self.logger.error(f"[LangInterface] VULCAN reasoning failed: {e}")
                reasoning_output = VulcanReasoningOutput(
                    query_id=hashlib.sha256(user_text.encode()).hexdigest()[:16],
                    success=False,
                    result=None,
                    error=str(e),
                    method_used="vulcan_error"
                )
        else:
            # No reasoning function provided - create placeholder
            reasoning_output = VulcanReasoningOutput(
                query_id=hashlib.sha256(user_text.encode()).hexdigest()[:16],
                success=False,
                result=None,
                error="No reasoning function configured",
                method_used="none"
            )
            systems_used.append("no_reasoning_configured")
        
        # STEP 3: Language interface OUT - Format result
        self.logger.info("[LangInterface] Step 3: Formatting output for user")
        formatted_result = await self.format_output_for_user(
            reasoning_output=reasoning_output.to_dict() if reasoning_output else {},
            original_prompt=user_text,
            max_tokens=max_tokens
        )
        systems_used.append("llm_output_formatting")
        
        return {
            "text": formatted_result.get("text", ""),
            "source": "vulcan_language_interface",
            "systems_used": systems_used,
            "structured_query": {
                "intent": structured_query.intent.value,
                "domain": structured_query.domain.value,
                "parameters": structured_query.parameters,
                "confidence": structured_query.confidence,
            },
            "reasoning_output": reasoning_output.to_dict() if reasoning_output else None,
            "metadata": formatted_result.get("metadata", {}),
        }

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
            # Use module-level constants (evaluated at import time from env vars)
            use_language_polish = OPENAI_LANGUAGE_POLISH
            
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
        self.logger.warning(
            "[HybridExecutor] ⚠ Internal LLM language generation failed. "
            "Note: The internal LLM is for language output, not reasoning. "
            "Attempting OpenAI fallback for language generation..."
        )
        systems_used.append("vulcan_local_llm_failed")
        
        # FALLBACK TO OPENAI: When internal LLM fails, use OpenAI for language generation
        # This implements the backup mechanism: OpenAI serves as fallback when internal LLM is down
        try:
            # Check if OpenAI client is available before attempting fallback
            openai_client = self.openai_client_getter()
            if openai_client is None:
                self.logger.error(
                    "[HybridExecutor] ❌ OpenAI client is NOT available for fallback. "
                    "Possible causes:\n"
                    "  - OPENAI_API_KEY environment variable not set\n"
                    "  - OPENAI_API_KEY secret not configured in repository\n"
                    "  - SKIP_OPENAI environment variable is 'true'\n"
                    "  - OpenAI client initialization failed\n"
                    "To fix: Set OPENAI_API_KEY in repository secrets."
                )
                systems_used.append("openai_client_unavailable")
            else:
                openai_result = await self._call_openai(
                    loop, prompt, max_tokens, temperature, system_prompt, conversation_history
                )
                if self._is_valid_response(openai_result):
                    systems_used.append("openai_fallback")
                    self.logger.info("[HybridExecutor] ✓ OpenAI fallback succeeded (internal LLM was unavailable)")
                    return {
                        "text": openai_result,
                        "source": "openai_fallback",
                        "systems_used": systems_used,
                        "metadata": {
                            "fallback_reason": "internal_llm_failed",
                            "openai_role": "language_generation_fallback",
                        },
                    }
                else:
                    self.logger.warning(
                        "[HybridExecutor] ⚠ OpenAI fallback returned invalid response. "
                        "Response may be empty or too short."
                    )
                    systems_used.append("openai_invalid_response")
        except Exception as openai_err:
            self.logger.warning(
                f"[HybridExecutor] ⚠ OpenAI fallback exception: {type(openai_err).__name__}: {openai_err}"
            )
            systems_used.append("openai_exception")
        
        # BOTH internal LLM AND OpenAI failed - return graceful error
        self.logger.error(
            "[HybridExecutor] ❌ Both internal LLM AND OpenAI fallback failed. "
            "No language generation backend available. "
            "Check CI logs for OPENAI_API_KEY configuration."
        )
        
        # GRACEFUL DEGRADATION FIX: Provide a user-friendly error message
        # Generate a unique error reference for tracking (use time_ns for better uniqueness)
        error_ref = hashlib.sha256(
            f"{time.time_ns()}:{prompt[:50]}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an internal processing issue while generating a response.\n\n"
            "Both primary and backup language systems are temporarily unavailable.\n\n"
            "This could be due to:\n"
            "• High system load causing a timeout\n"
            "• A complex query requiring more processing time\n"
            "• Temporary network or service issues\n\n"
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
                "reason": "Both VULCAN internal LLM and OpenAI fallback failed",
                "vulcan_llm_failed": True,
                "openai_fallback_failed": True,
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
        DEPRECATED: This method allows LLM to answer directly, bypassing VULCAN reasoning.
        Use execute_with_language_interface() instead for proper architecture.
        
        OpenAI-first mode with internal LLM fallback.
        
        ARCHITECTURE:
        1. Try OpenAI first for language generation
        2. If OpenAI fails/unavailable, fall back to internal LLM
        
        This mode is useful when:
        - OpenAI provides faster/higher-quality responses
        - Internal LLM serves as backup when OpenAI is down
        """
        import warnings
        warnings.warn(
            "_execute_openai_first allows LLM to bypass VULCAN reasoning. "
            "Use execute_with_language_interface() for proper architecture.",
            DeprecationWarning,
            stacklevel=2
        )
        
        systems_used = []
        
        # Step 1: Try OpenAI for language generation
        try:
            openai_result = await self._call_openai(
                loop, prompt, max_tokens, temperature, system_prompt, conversation_history
            )
            if self._is_valid_response(openai_result):
                systems_used.append("openai")
                self.logger.info("[HybridExecutor] ✓ OpenAI language generation succeeded")
                return {
                    "text": openai_result,
                    "source": "openai",
                    "systems_used": systems_used,
                }
            else:
                self.logger.warning("[HybridExecutor] ⚠ OpenAI returned invalid response, trying internal LLM fallback")
        except Exception as openai_err:
            self.logger.warning(f"[HybridExecutor] ⚠ OpenAI failed: {openai_err}, trying internal LLM fallback")
        
        systems_used.append("openai_failed")
        
        # Step 2: Fallback to internal LLM when OpenAI is unavailable
        self.logger.info("[HybridExecutor] OpenAI unavailable, falling back to internal LLM...")
        local_result = await self._call_local_llm(loop, prompt, max_tokens)
        
        if self._is_valid_response(local_result):
            systems_used.append("vulcan_local_llm_fallback")
            self.logger.info("[HybridExecutor] ✓ Internal LLM fallback succeeded (OpenAI was unavailable)")
            return {
                "text": local_result,
                "source": "local_fallback",
                "systems_used": systems_used,
                "metadata": {
                    "fallback_reason": "openai_unavailable",
                    "internal_llm_role": "language_generation_fallback",
                },
            }
        
        # Both OpenAI AND internal LLM failed
        self.logger.error(
            "[HybridExecutor] ❌ Both OpenAI AND internal LLM fallback failed. "
            "No language generation backend available."
        )
        systems_used.append("vulcan_local_llm_failed")
        
        # Generate error reference (use time_ns for better uniqueness)
        error_ref = hashlib.sha256(
            f"{time.time_ns()}:{prompt[:50]}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an issue generating a response.\n\n"
            "Both primary and backup language systems are temporarily unavailable.\n\n"
            "**Suggestions:**\n"
            "• Please try again in a moment\n"
            "• Try rephrasing your question\n\n"
            f"If this issue persists, reference: **{error_ref}**"
        )
        
        return {
            "text": error_text,
            "source": "error_graceful_degradation",
            "systems_used": systems_used,
            "error": True,
            "metadata": {
                "reason": "Both OpenAI and internal LLM fallback failed",
                "openai_failed": True,
                "vulcan_llm_failed": True,
                "can_retry": True,
                "error_reference": error_ref,
            },
        }

    async def _execute_parallel(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Execute both local and OpenAI in parallel, use first successful response.
        
        RESTORED: True parallel execution for faster response times.
        This fixes the 60-second timeout issue by allowing OpenAI to respond quickly
        (~2-5s) while the slow internal LLM (~500ms/token) is still processing.
        
        ARCHITECTURE:
        - Both local LLM and OpenAI run simultaneously
        - First successful response wins
        - If OpenAI unavailable, falls back to local_first mode
        - Pending tasks are cancelled after first success
        """
        start_time = time.perf_counter()
        
        # Check if OpenAI is available (Issue #6: detailed logging)
        openai_available = self._openai_available()
        self.logger.info(f"[HybridExecutor] Parallel mode starting - OpenAI available: {openai_available}")
        
        if not openai_available:
            self.logger.info("[HybridExecutor] OpenAI not available, falling back to local_first")
            return await self._execute_local_first(
                loop, prompt, max_tokens, temperature, system_prompt, conversation_history
            )
        
        self.logger.info("[HybridExecutor] Executing in TRUE parallel mode")
        
        # Create tasks for both backends
        local_task = asyncio.create_task(
            self._run_local_llm(loop, prompt, max_tokens),
            name="local_llm_task"
        )
        openai_task = asyncio.create_task(
            self._run_openai(loop, prompt, max_tokens, temperature, system_prompt, conversation_history),
            name="openai_task"
        )
        
        try:
            # Run tasks in parallel, returning when EITHER succeeds with valid output
            # Use a loop to wait for tasks and check for valid results
            tasks = {local_task, openai_task}
            start_wait = time.perf_counter()
            
            # Note: Track results from each task
            # This allows us to prefer local results over OpenAI even if OpenAI finishes first,
            # IF local has already produced a usable result
            local_result = None
            openai_result = None
            
            while tasks and (time.perf_counter() - start_wait) < self.timeout:
                # Wait for the first task to complete
                remaining_timeout = self.timeout - (time.perf_counter() - start_wait)
                if remaining_timeout <= 0:
                    break
                    
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=remaining_timeout
                )
                
                if not done:
                    # Timeout - no tasks completed
                    break
                
                # Check completed tasks for valid results
                for task in done:
                    try:
                        result = task.result()
                        task_name = _get_task_name(task)
                        is_local = task_name == "local_llm_task"
                        
                        # Check if result is a valid dict with text
                        if isinstance(result, dict) and self._is_valid_response(result.get("text")):
                            if is_local:
                                local_result = result
                                self.logger.info(f"[HybridExecutor] Local LLM produced valid result")
                            else:
                                openai_result = result
                                self.logger.info(f"[HybridExecutor] OpenAI produced valid result")
                            
                            # Note: Prefer local results over OpenAI
                            # If we have a local result, use it immediately (VULCAN should do reasoning)
                            # This prevents OpenAI from "winning" just because it's faster
                            if local_result:
                                local_result["source"] = "parallel_local"
                                elapsed = time.perf_counter() - start_time
                                self.logger.info(f"[HybridExecutor] ✓ Parallel mode complete - winner: local (VULCAN), time: {elapsed:.2f}s")
                                
                                # Cancel any remaining tasks
                                for pending_task in pending:
                                    pending_task.cancel()
                                    try:
                                        await pending_task
                                    except asyncio.CancelledError:
                                        pass
                                
                                return local_result
                            
                            # If only OpenAI has a result, check if local is still running
                            # Give local a short grace period to complete
                            if openai_result and not local_result and local_task in pending:
                                # Note: Wait a bit longer for local to finish
                                # This prevents cancelling local when it's almost done
                                grace_period = min(LOCAL_GRACE_PERIOD_SECONDS, remaining_timeout)
                                self.logger.info(f"[HybridExecutor] OpenAI won first, waiting {grace_period}s for local to finish...")
                                
                                try:
                                    local_done, _ = await asyncio.wait(
                                        {local_task},
                                        timeout=grace_period
                                    )
                                    if local_done:
                                        try:
                                            local_check = local_task.result()
                                            if isinstance(local_check, dict) and self._is_valid_response(local_check.get("text")):
                                                local_result = local_check
                                                local_result["source"] = "parallel_local"
                                                elapsed = time.perf_counter() - start_time
                                                self.logger.info(f"[HybridExecutor] ✓ Local completed during grace period - using local (VULCAN), time: {elapsed:.2f}s")
                                                return local_result
                                        except Exception:
                                            pass
                                except asyncio.TimeoutError:
                                    pass
                                
                                # Local didn't complete in grace period, use OpenAI
                                openai_result["source"] = "parallel_openai"
                                elapsed = time.perf_counter() - start_time
                                self.logger.info(f"[HybridExecutor] ✓ Parallel mode complete - winner: openai, time: {elapsed:.2f}s")
                                
                                # Cancel local task
                                local_task.cancel()
                                try:
                                    await local_task
                                except asyncio.CancelledError:
                                    pass
                                
                                return openai_result
                        else:
                            # Task completed but with invalid result, log it
                            elapsed = time.perf_counter() - start_time
                            source_name = "local" if is_local else "openai"
                            # Treat expected cancellations as informational noise, not errors
                            if isinstance(result, dict) and result.get("source") in {"local_cancelled", "openai_cancelled"}:
                                self.logger.info(f"[HybridExecutor] Task {source_name} was cancelled after winner resolved ({elapsed:.2f}s)")
                            else:
                                self.logger.info(f"[HybridExecutor] Task {source_name} completed without usable result after {elapsed:.2f}s")
                    except Exception as e:
                        task_name = _get_task_name(task)
                        self.logger.warning(f"[HybridExecutor] Task {task_name} failed with exception: {e}")
                
                # Remove completed tasks from set and continue waiting for others
                tasks = pending
            
            # Cancel any remaining tasks after loop exits
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # If we get here, no task succeeded - fall back to local_first
            elapsed = time.perf_counter() - start_time
            self.logger.warning(f"[HybridExecutor] Parallel mode: no task succeeded after {elapsed:.2f}s, falling back to local_first")
            
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            self.logger.warning(f"[HybridExecutor] Parallel mode timeout after {elapsed:.2f}s (limit: {self.timeout}s)")
            # Cancel all tasks
            local_task.cancel()
            openai_task.cancel()
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.logger.warning(f"[HybridExecutor] Parallel mode error after {elapsed:.2f}s: {e}")
        
        # Fallback to local_first if parallel failed
        self.logger.warning("[HybridExecutor] Parallel mode failed, falling back to local_first")
        return await self._execute_local_first(
            loop, prompt, max_tokens, temperature, system_prompt, conversation_history
        )
    
    def _openai_available(self) -> bool:
        """Check if OpenAI client is available and configured.
        
        Returns:
            True if OpenAI is available for use, False otherwise
        """
        try:
            client = self.openai_client_getter()
            return client is not None
        except Exception:
            return False
    
    async def _run_local_llm(
        self, loop, prompt: str, max_tokens: int
    ) -> Dict[str, Any]:
        """Async wrapper for local LLM execution.
        
        Returns a dict with 'text' and 'source' keys for parallel mode.
        """
        start = time.perf_counter()
        self.logger.info("[HybridExecutor] Local LLM task started")
        try:
            text = await self._call_local_llm(loop, prompt, max_tokens)
        except asyncio.CancelledError:
            self.logger.info("[HybridExecutor] Local LLM task cancelled after winner resolved")
            return {"text": None, "source": "local_cancelled", "systems_used": []}
        
        elapsed = time.perf_counter() - start
        if text and self._is_valid_response(text):
            self.logger.info(f"[HybridExecutor] ✓ Local LLM task completed successfully in {elapsed:.2f}s (len={len(text)})")
            return {
                "text": text,
                "source": "local",
                "systems_used": ["vulcan_local_llm"],
            }
        self.logger.info(f"[HybridExecutor] Local LLM task completed without usable result in {elapsed:.2f}s")
        return {"text": None, "source": "local_failed", "systems_used": []}
    
    async def _run_openai(
        self, loop, prompt: str, max_tokens: int, temperature: float, 
        system_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Async wrapper for OpenAI execution.
        
        Returns a dict with 'text' and 'source' keys for parallel mode.
        """
        start = time.perf_counter()
        self.logger.info("[HybridExecutor] OpenAI task started")
        try:
            text = await self._call_openai(
                loop, prompt, max_tokens, temperature, system_prompt, conversation_history
            )
        except asyncio.CancelledError:
            self.logger.info("[HybridExecutor] OpenAI task cancelled after winner resolved")
            return {"text": None, "source": "openai_cancelled", "systems_used": []}
        
        elapsed = time.perf_counter() - start
        if text and self._is_valid_response(text):
            self.logger.info(f"[HybridExecutor] ✓ OpenAI task completed successfully in {elapsed:.2f}s (len={len(text)})")
            return {
                "text": text,
                "source": "openai",
                "systems_used": ["openai"],
            }
        self.logger.info(f"[HybridExecutor] OpenAI task completed without usable result in {elapsed:.2f}s (text={text[:100] if text else None}...)")
        return {"text": None, "source": "openai_failed", "systems_used": []}

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

    async def _execute_openai_only(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        DEPRECATED: This method allows LLM to answer directly, bypassing VULCAN reasoning.
        Use execute_with_language_interface() instead for proper architecture.
        
        OpenAI-only mode for fast language generation.
        
        FIX: Local LLM (GraphixVulcanLLM) times out (~120s) on CPU - this mode
        uses OpenAI only for language generation (~3s response time).
        
        IMPORTANT: This does NOT affect reasoning engines!
        - Symbolic reasoning: Still works (src/vulcan/reasoning/symbolic/*)
        - Causal reasoning: Still works (src/vulcan/reasoning/causal/*)
        - Probabilistic reasoning: Still works (src/vulcan/reasoning/probabilistic/*)
        - Mathematical reasoning: Still works (src/vulcan/reasoning/mathematical/*)
        
        Only the LANGUAGE OUTPUT step is affected - OpenAI formats the reasoning
        results into natural language for the user.
        """
        import warnings
        warnings.warn(
            "_execute_openai_only allows LLM to bypass VULCAN reasoning. "
            "Use execute_with_language_interface() for proper architecture.",
            DeprecationWarning,
            stacklevel=2
        )
        
        systems_used = []
        
        try:
            openai_result = await self._call_openai(
                loop, prompt, max_tokens, temperature, system_prompt, conversation_history
            )
            if self._is_valid_response(openai_result):
                systems_used.append("openai")
                self.logger.info("[HybridExecutor] ✓ OpenAI language generation succeeded (~3s)")
                return {
                    "text": openai_result,
                    "source": "openai",
                    "systems_used": systems_used,
                    "metadata": {
                        "mode": "openai_only",
                        "reason": "Fast language generation (local LLM disabled for performance)",
                    },
                }
            else:
                self.logger.warning("[HybridExecutor] ⚠ OpenAI returned invalid response")
        except Exception as openai_err:
            self.logger.warning(f"[HybridExecutor] ⚠ OpenAI failed: {openai_err}")
        
        systems_used.append("openai_failed")
        
        # Generate error reference
        error_ref = hashlib.sha256(
            f"{time.time_ns()}:{prompt[:50]}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an issue generating a response.\n\n"
            "The language generation service is temporarily unavailable.\n\n"
            "**Suggestions:**\n"
            "• Please try again in a moment\n"
            "• Try rephrasing your question\n\n"
            f"If this issue persists, reference: **{error_ref}**"
        )
        
        return {
            "text": error_text,
            "source": "error_openai_unavailable",
            "systems_used": systems_used,
            "error": True,
            "metadata": {
                "mode": "openai_only",
                "reason": "OpenAI language generation failed",
                "can_retry": True,
                "error_reference": error_ref,
            },
        }

    async def _execute_local_only(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Local-only mode when OpenAI is unavailable.
        
        Falls back to local LLM for language generation.
        """
        systems_used = []
        
        local_result = await self._call_local_llm(loop, prompt, max_tokens)
        
        if self._is_valid_response(local_result):
            systems_used.append("vulcan_local_llm")
            self.logger.info("[HybridExecutor] ✓ Local LLM language generation succeeded")
            return {
                "text": local_result,
                "source": "local",
                "systems_used": systems_used,
                "metadata": {
                    "mode": "local_only",
                    "reason": "OpenAI unavailable - using local LLM",
                },
            }
        
        systems_used.append("vulcan_local_llm_failed")
        
        # Generate error reference
        error_ref = hashlib.sha256(
            f"{time.time_ns()}:{prompt[:50]}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an issue generating a response.\n\n"
            "The language generation system is temporarily unavailable.\n\n"
            "**Suggestions:**\n"
            "• Please try again in a moment\n"
            "• Try rephrasing your question\n\n"
            f"If this issue persists, reference: **{error_ref}**"
        )
        
        return {
            "text": error_text,
            "source": "error_local_unavailable",
            "systems_used": systems_used,
            "error": True,
            "metadata": {
                "mode": "local_only",
                "reason": "Local LLM language generation failed",
                "can_retry": True,
                "error_reference": error_ref,
            },
        }

    async def _execute_sequential(
        self, loop, prompt: str, max_tokens: int, temperature: float, system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Sequential mode: Try OpenAI first (fast), fallback to local LLM if fails.
        
        This is the preferred mode when:
        - OpenAI provides fast responses (~3 seconds)
        - Local LLM serves as backup when OpenAI is down
        - No parallel execution (wastes time waiting)
        
        ARCHITECTURE:
        1. Try OpenAI first (fast, 3 seconds)
        2. If OpenAI fails/unavailable → Use local LLM (slow, but works)
        3. No parallel execution
        
        This solves the timeout problem where parallel mode waits 120s for local LLM
        even though OpenAI already finished in 3s.
        """
        systems_used = []
        
        # Step 1: Try OpenAI first (fast path)
        if self.openai_client is not None:
            try:
                self.logger.info("[HybridExecutor] Trying OpenAI first...")
                openai_result = await self._call_openai(
                    loop, prompt, max_tokens, temperature, system_prompt, conversation_history
                )
                
                # Check if result is valid (not empty/too short)
                if self._is_valid_response(openai_result):
                    systems_used.append("openai")
                    self.logger.info(f"[HybridExecutor] ✓ OpenAI succeeded ({len(openai_result)} chars)")
                    return {
                        "text": openai_result,
                        "source": "openai",
                        "systems_used": systems_used,
                        "metadata": {
                            "mode": "sequential",
                            "path": "openai_primary",
                        },
                    }
                else:
                    self.logger.warning("[HybridExecutor] OpenAI returned empty/invalid result")
                    systems_used.append("openai_invalid_response")
                    
            except Exception as e:
                self.logger.warning(f"[HybridExecutor] OpenAI failed: {e}")
                systems_used.append("openai_exception")
        else:
            self.logger.info("[HybridExecutor] No OpenAI client available, skipping to local LLM")
            systems_used.append("openai_unavailable")
        
        # Step 2: OpenAI failed or unavailable - fallback to local LLM
        if self.local_llm is not None:
            self.logger.info("[HybridExecutor] Falling back to local LLM...")
            try:
                local_result = await self._call_local_llm(loop, prompt, max_tokens)
                
                if self._is_valid_response(local_result):
                    systems_used.append("vulcan_local_llm_fallback")
                    self.logger.info(f"[HybridExecutor] ✓ Local LLM succeeded ({len(local_result)} chars)")
                    return {
                        "text": local_result,
                        "source": "local_fallback",
                        "systems_used": systems_used,
                        "metadata": {
                            "mode": "sequential",
                            "path": "local_fallback",
                            "fallback_reason": "openai_failed_or_unavailable",
                        },
                    }
                else:
                    self.logger.warning("[HybridExecutor] Local LLM returned empty/invalid result")
                    systems_used.append("vulcan_local_llm_invalid")
                    
            except Exception as e:
                self.logger.error(f"[HybridExecutor] Local LLM also failed: {e}")
                systems_used.append("vulcan_local_llm_exception")
        else:
            self.logger.warning("[HybridExecutor] No local LLM available for fallback")
            systems_used.append("local_llm_unavailable")
        
        # Both OpenAI AND local LLM failed
        self.logger.error(
            "[HybridExecutor] ❌ Both OpenAI AND local LLM failed. "
            "No language generation backend available."
        )
        
        # Generate error reference
        error_ref = hashlib.sha256(
            f"{time.time_ns()}:{prompt[:50]}".encode()
        ).hexdigest()[:12].upper()
        
        error_text = (
            "I encountered an issue generating a response.\n\n"
            "Both primary (OpenAI) and backup (local LLM) language systems failed.\n\n"
            "**Suggestions:**\n"
            "• Please try again in a moment\n"
            "• Try rephrasing your question\n\n"
            f"If this issue persists, reference: **{error_ref}**"
        )
        
        return {
            "text": error_text,
            "source": "error_both_failed",
            "systems_used": systems_used,
            "error": True,
            "metadata": {
                "mode": "sequential",
                "reason": "Both OpenAI and local LLM failed",
                "can_retry": True,
                "error_reference": error_ref,
            },
        }

    # ============================================================
    # LLM CALL METHODS
    # ============================================================

    async def _call_local_llm(
        self, loop, prompt: str, max_tokens: int
    ) -> Optional[str]:
        """Call Vulcan's local LLM.
        
        ARCHITECTURE: VULCAN is primary brain, OpenAI is language fallback only.
        This method attempts to use VULCAN's internal LLM first.
        
        Note: Added detailed error logging to expose why local model
        generation fails silently. Previously, exceptions were caught and
        logged at debug level, hiding the real cause of 100% OpenAI fallback.
        
        CPU CLOUD FIX: Limits max_tokens to prevent timeout on CPU-only instances.
        At ~500ms per token, max_tokens must be limited to ensure completion
        within the timeout period.
        """
        import traceback
        
        # Check if local LLM should be skipped (default: False, VULCAN runs first)
        if _should_skip_local_llm():
            self.logger.warning(
                "[HybridExecutor] SKIP_LOCAL_LLM=true - VULCAN brain bypassed! "
                "Set SKIP_LOCAL_LLM=false to enable VULCAN reasoning."
            )
            return None
        
        # Note: Log entry point with model state
        self.logger.info("=" * 60)
        self.logger.info("[HybridExecutor] ATTEMPTING LOCAL LLM GENERATION")
        self.logger.info(f"[HybridExecutor] local_llm is None: {self.local_llm is None}")
        
        if not self.local_llm:
            self.logger.error("[HybridExecutor] CRITICAL: local_llm is None - cannot generate!")
            self.logger.info("=" * 60)
            return None
        
        # Note: Log detailed model state
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
            # CPU CLOUD FIX: Limit max_tokens to prevent timeout on CPU-only instances
            # At ~500ms per token, generating more than CPU_MAX_TOKENS_DEFAULT tokens
            # would exceed the timeout. This ensures completion before timeout.
            effective_max_tokens = min(max_tokens, CPU_MAX_TOKENS_DEFAULT)
            if effective_max_tokens < max_tokens:
                self.logger.info(
                    f"[HybridExecutor] CPU optimization: Limiting max_tokens from {max_tokens} "
                    f"to {effective_max_tokens} (VULCAN_CPU_MAX_TOKENS={CPU_MAX_TOKENS_DEFAULT})"
                )
            
            self.logger.info(f"[HybridExecutor] Calling generate(prompt_len={len(prompt)}, max_tokens={effective_max_tokens})...")
            
            # ADAPTIVE TIMEOUT FIX: Calculate timeout based on token count
            # Formula: timeout = base_timeout (5s) + (max_tokens * 2.0s)
            adaptive_timeout = calculate_adaptive_timeout(effective_max_tokens)
            self.logger.info(f"[HybridExecutor] Using ADAPTIVE timeout: {adaptive_timeout:.1f}s (base={BASE_TIMEOUT_SECONDS}s + {effective_max_tokens}*{TIMEOUT_PER_TOKEN_SECONDS}s)")
            start_time = time.perf_counter()
            
            # PARALLEL MODE FIX: Use non-blocking async pattern for concurrent execution
            # Previous implementation used blocking future.result() which prevented the
            # asyncio event loop from running other tasks (like OpenAI task in parallel mode).
            # 
            # New approach:
            # 1. Submit the work to ThreadPoolExecutor (this is async-friendly)
            # 2. Use loop.run_in_executor to wait without blocking the event loop
            # 3. Wrap in asyncio.wait_for() for timeout handling
            #
            # This ensures OpenAI task can run concurrently with local LLM task.
            def generate_sync():
                return self.local_llm.generate(prompt, effective_max_tokens)
            
            try:
                # Check if event loop is closed before attempting async work
                if loop.is_closed():
                    self.logger.warning("[HybridExecutor] Event loop is closed, returning None")
                    return None
                
                # ASYNC-FRIENDLY WAITING: Use run_in_executor to avoid blocking the event loop
                # This allows other asyncio tasks (like OpenAI) to run concurrently
                self.logger.info("[HybridExecutor] Starting async-friendly local LLM generation...")
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._timeout_executor, generate_sync),
                    timeout=adaptive_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"[HybridExecutor] ❌ ASYNC TIMEOUT after {adaptive_timeout:.1f}s!")
                return None
            except asyncio.CancelledError:
                self.logger.debug("[HybridExecutor] Local LLM task was cancelled")
                return None
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cannot schedule new futures after shutdown" in error_str:
                    self.logger.debug("[HybridExecutor] Executor shutdown detected, returning None")
                    return None
                if "event loop is closed" in error_str:
                    self.logger.debug("[HybridExecutor] Event loop closed, returning None")
                    return None
                # Re-raise other RuntimeErrors to be caught by outer handler
                raise
            
            elapsed = time.perf_counter() - start_time
            self.logger.info(f"[HybridExecutor] generate() returned in {elapsed:.2f}s")

            # Handle None result (returned when event loop conflict is detected)
            if result is None:
                self.logger.warning("[HybridExecutor] Local LLM returned None - triggering fallback")
                return None

            # Note: Log successful generation
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
            # Note: Log FULL error details - this is critical for debugging
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
            start_time = time.perf_counter()
            
            # Use generate_fast which skips reasoning hooks
            def generate_fast_sync():
                return self.local_llm.generate_fast(prompt, max_tokens)
            
            # Use hard timeout (should be faster than standard generate)
            # Fast mode uses FAST_MODE_MAX_TIMEOUT_SECONDS since no reasoning hooks run
            fast_timeout = min(self.vulcan_timeout, FAST_MODE_MAX_TIMEOUT_SECONDS)
            
            # PARALLEL MODE FIX: Use non-blocking async pattern
            try:
                # Check if event loop is closed before attempting async work
                if loop.is_closed():
                    self.logger.warning("[HybridExecutor] Event loop is closed, returning None")
                    return None
                
                self.logger.info("[HybridExecutor] Starting async-friendly fast generation...")
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._timeout_executor, generate_fast_sync),
                    timeout=fast_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"[HybridExecutor] ❌ FAST MODE ASYNC TIMEOUT after {fast_timeout}s!")
                return None
            except asyncio.CancelledError:
                self.logger.debug("[HybridExecutor] Fast generation task was cancelled")
                return None
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cannot schedule new futures after shutdown" in error_str:
                    self.logger.debug("[HybridExecutor] Executor shutdown detected in fast mode, returning None")
                    return None
                if "event loop is closed" in error_str:
                    self.logger.debug("[HybridExecutor] Event loop closed in fast mode, returning None")
                    return None
                # Re-raise other RuntimeErrors to be caught by outer handler
                raise
            
            elapsed = time.perf_counter() - start_time
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
            self.logger.debug(
                "[HybridExecutor] OpenAI client not available - "
                "check OPENAI_API_KEY env var or repository secrets"
            )
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
            
            # Log successful OpenAI API call
            if result:
                self.logger.info(f"[HybridExecutor] OpenAI API call succeeded (len={len(result)})")
            else:
                self.logger.warning("[HybridExecutor] OpenAI API call returned empty result")
            
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
            self.logger.warning(f"[HybridExecutor] OpenAI call failed: {type(e).__name__}: {e}")
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
        # Use the dedicated formatting method with gpt-4o-mini
        return await self._format_with_openai_for_output(
            reasoning_output=reasoning_output,
            original_query=original_query,
            loop=loop,
        )
    
    async def _format_with_openai_for_output(
        self, 
        reasoning_output: "VulcanReasoningOutput",
        original_query: str,
        loop,
    ) -> Optional[str]:
        """
        Use OpenAI (gpt-4o-mini) to format VULCAN's reasoning output as natural language.
        
        This is the primary method for language formatting with distillation capture.
        
        ARCHITECTURE:
            VULCAN Mind (orchestrator, agents, symbolic reasoning)
                │
                │ produces
                ▼
            Structured Output (JSON/dict with results, confidence, method, etc.)
                │
                │ sent to
                ▼
            OpenAI gpt-4o-mini (language formatting ONLY)
                │
                │ produces
                ▼
            Natural Language Response → returned to user
                │
                │ captured as training pair
                ▼
            Distillation Store → VULCAN LLM learns async
        
        POLICY COMPLIANCE:
        - OpenAI MUST NOT reason independently — only format VULCAN's output
        - OpenAI MUST NOT generate code
        - All reasoning happens in VULCAN's mind BEFORE this method is called
        - Every (input, output) pair is captured for distillation
        
        Args:
            reasoning_output: The structured output from VULCAN's reasoning systems
            original_query: The user's original question (for context)
            loop: The asyncio event loop
            
        Returns:
            Formatted natural language response, or None if formatting fails
        """
        # ============================================================
        # FIX (Issue #ROUTING-001): Use Content Preservation Prompt
        # ============================================================
        # When VULCAN's reasoning includes introspection or self-awareness responses,
        # we MUST use the content preservation prompt to prevent OpenAI from replacing
        # VULCAN's authentic responses with generic AI disclaimers.
        #
        # Check if this is an introspection/self-awareness response that needs protection
        # ROOT CAUSE FIX: Also check for privileged_no_answer and override_router_tools
        is_introspection = (
            hasattr(reasoning_output, 'metadata') and 
            reasoning_output.metadata and
            reasoning_output.metadata.get('is_introspection', False)
        )
        
        # ROOT CAUSE FIX: Check for privileged query flags
        is_privileged = (
            hasattr(reasoning_output, 'metadata') and 
            reasoning_output.metadata and (
                reasoning_output.metadata.get('privileged_no_answer', False) or
                reasoning_output.metadata.get('override_router_tools', False)
            )
        )
        
        # Use strict content preservation prompt for introspection or privileged queries
        if is_introspection or is_privileged:
            system_prompt = self.VULCAN_CONTENT_PRESERVATION_PROMPT
            reason = "introspection" if is_introspection else "privileged"
            self.logger.info(
                f"[HybridExecutor] ROOT CAUSE FIX: Using content preservation prompt for {reason} query - "
                f"VULCAN's response will be protected from LLM override"
            )
        else:
            # Standard formatting prompt for non-introspection queries
            system_prompt = """You are a language formatter for VULCAN AI.

YOUR ONLY ROLE: Convert VULCAN's structured reasoning output into clear, natural language.

STRICT RULES:
1. DO NOT perform any independent reasoning or analysis
2. DO NOT add information not present in VULCAN's output
3. DO NOT generate code
4. DO NOT speculate beyond the provided data
5. ONLY format and present VULCAN's results as readable prose

You receive JSON containing:
- result: The computed answer
- method_used: How VULCAN solved it  
- confidence: VULCAN's confidence (0-1)
- reasoning_trace: Steps taken (optional)
- error: Error message if failed

Make this human-readable. Nothing more."""

        # Issue #7 FIX: Check reasoning confidence BEFORE sending to OpenAI
        # BUT: Don't block OpenAI if reasoning engine says "not_applicable"
        # When engine declines (not_applicable=True), we should let OpenAI try
        MIN_REASONING_CONFIDENCE = 0.5
        reasoning_confidence = getattr(reasoning_output, 'confidence', None) or 0.0
        
        # Issue #7 FIX: Check if reasoning engine declined the query (not_applicable)
        # If so, don't treat this as a failure - let OpenAI attempt it
        is_not_applicable = False
        if hasattr(reasoning_output, 'to_dict'):
            try:
                output_dict = reasoning_output.to_dict()
                is_not_applicable = (
                    output_dict.get('not_applicable') is True or
                    output_dict.get('applicable') is False
                )
            except Exception:
                pass
        
        # Issue #7 FIX: Only block OpenAI if reasoning truly attempted but failed
        # Don't block if engine declined (not_applicable) - that means try another approach
        if reasoning_confidence < MIN_REASONING_CONFIDENCE and not is_not_applicable:
            self.logger.warning(
                f"[HybridExecutor] Note: Reasoning confidence ({reasoning_confidence:.2f}) < "
                f"threshold ({MIN_REASONING_CONFIDENCE}). Returning failure message instead of "
                f"letting OpenAI compensate."
            )
            return (
                f"I was unable to complete the specialized reasoning for this problem. "
                f"The reasoning engine returned confidence {reasoning_confidence:.2f}. "
                f"Please try rephrasing your question or providing more context."
            )
        elif is_not_applicable:
            self.logger.info(
                f"[HybridExecutor] Issue #7 FIX: Reasoning engine declined query "
                f"(not_applicable=True). Allowing OpenAI to attempt the query."
            )

        # Build the user prompt with VULCAN's structured output
        # Note: Do NOT include original_query to prevent OpenAI from solving independently
        output_dict = None
        try:
            output_dict = reasoning_output.to_dict()
            output_json = json.dumps(output_dict, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to serialize reasoning output: {e}")
            output_json = str(reasoning_output)

        # Note: Removed original question from prompt
        # OpenAI should ONLY see the reasoning output, not the original question
        # This prevents OpenAI from solving problems independently when reasoning fails
        user_prompt = f"""Format this VULCAN reasoning output for the user.

VULCAN's reasoning output (this is what VULCAN computed):
{output_json}

Write a natural, helpful response that explains VULCAN's results and conclusions.
Do NOT add any analysis or reasoning beyond what is in VULCAN's output."""

        try:
            # Use gpt-4o-mini for fast and cheap formatting
            response = await self._call_openai_formatting(
                loop=loop,
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3,  # Low temp for consistent formatting
            )
            
            if response and len(response.strip()) > self.MIN_MEANINGFUL_LENGTH:
                # Note: Detect hallucinations before returning response
                if self._is_hallucination(response, output_json):
                    self.logger.error(
                        f"[HybridExecutor] Note: Hallucination detected in OpenAI response. "
                        f"Response contains fabricated content not in reasoning output."
                    )
                    # Return a safe failure message instead of the hallucinated response
                    return (
                        "I was unable to solve this problem with the available reasoning engines. "
                        "Please try rephrasing your question."
                    )
                
                self.logger.info("[HybridExecutor] ✓ OpenAI (gpt-4o-mini) formatted VULCAN's structured output")
                
                # Capture training pair for distillation (VULCAN LLM learns from this)
                self._capture_formatting_for_distillation(
                    input_data=output_dict if output_dict is not None else {"raw": output_json},
                    output_text=response,
                    original_prompt=original_query,
                )
                
                return response
            else:
                self.logger.warning("[HybridExecutor] OpenAI returned empty/short response for formatting")
                return None
                
        except Exception as e:
            self.logger.warning(f"[HybridExecutor] OpenAI formatting failed: {e}")
            return None
    
    async def _call_openai_formatting(
        self,
        loop,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        Call OpenAI API specifically for output formatting using gpt-4o-mini.
        
        This method uses gpt-4o-mini which is:
        - Fast (~2-5 seconds response time)
        - Cheap (much cheaper than gpt-4)
        - Good enough for language formatting (not reasoning)
        
        Args:
            loop: The asyncio event loop
            prompt: The user prompt with VULCAN's output to format
            system_prompt: System prompt enforcing formatting-only behavior
            max_tokens: Maximum tokens for response (default: 500)
            temperature: Sampling temperature (default: 0.3 for consistency)
            
        Returns:
            Formatted response text, or None if call fails
        """
        openai_client = self.openai_client_getter()
        if not openai_client:
            self.logger.debug("[HybridExecutor] OpenAI client not available for formatting")
            return None
        
        try:
            def call_openai_mini():
                completion = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast and cheap for formatting
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content
            
            result = await loop.run_in_executor(None, call_openai_mini)
            return result
            
        except Exception as e:
            self.logger.debug(f"[HybridExecutor] OpenAI formatting call failed: {e}")
            return None
    
    def _is_hallucination(self, response: str, reasoning_output_json: str) -> bool:
        """
        Note: Detect if response contains fabricated content.
        
        Hallucination indicators are phrases or content that appear in OpenAI's
        response but were NOT in VULCAN's reasoning output. This typically happens
        when:
        1. Reasoning fails (low confidence) but OpenAI tries to answer anyway
        2. OpenAI invents fake "SymPy computation" or similar technical-sounding claims
        3. OpenAI uses generic placeholders like "x² + 2x + 1"
        
        Args:
            response: The OpenAI-generated response text
            reasoning_output_json: The JSON string of VULCAN's reasoning output
            
        Returns:
            True if hallucination is detected, False otherwise
        """
        # Common hallucination indicators
        # These phrases suggest OpenAI is fabricating technical details
        HALLUCINATION_INDICATORS = [
            'sympy',           # Never actually invoked by VULCAN
            'x² + 2x + 1',     # Generic placeholder polynomial
            'x^2 + 2x + 1',    # Alternative notation
            'x*2 + 2x + 1',    # Variation
            'computed using',  # Implies computation that didn't happen
            'calculation shows',  # Implies math that didn't happen
            'i calculated',    # OpenAI claiming to do math
            'my calculation',  # OpenAI claiming computation
            'i computed',      # OpenAI claiming computation
            'let me solve',    # OpenAI attempting to solve
            'solving step by step',  # OpenAI doing independent reasoning
            'evaluating the expression',  # OpenAI doing independent work
        ]
        
        response_lower = response.lower()
        reasoning_lower = reasoning_output_json.lower()
        
        for indicator in HALLUCINATION_INDICATORS:
            indicator_lower = indicator.lower()
            # Check if indicator is in response but NOT in reasoning output
            if indicator_lower in response_lower and indicator_lower not in reasoning_lower:
                self.logger.warning(
                    f"[HybridExecutor] Note: Possible hallucination detected - "
                    f"'{indicator}' in response but not in reasoning output"
                )
                return True
        
        return False
    
    def _capture_formatting_for_distillation(
        self,
        input_data: Dict[str, Any],
        output_text: str,
        original_prompt: str,
    ) -> None:
        """
        Capture (input, output) pair for VULCAN LLM training via distillation.
        
        This is how VULCAN LLM learns to format language over time.
        Every OpenAI formatting response becomes a training example.
        
        Args:
            input_data: VULCAN's structured reasoning output (dict)
            output_text: OpenAI's formatted natural language response
            original_prompt: The user's original question
        """
        from datetime import datetime
        
        training_example = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": {
                "prompt": original_prompt,
                "reasoning_output": input_data,
            },
            "output": output_text,
            "source": "openai_formatting",
            "model": "gpt-4o-mini",
        }
        
        # Use existing distillation infrastructure
        try:
            distiller = get_knowledge_distiller()
            if distiller is not None:
                # Capture via the distillation system
                distiller.capture_response(
                    prompt=original_prompt,
                    openai_response=output_text,
                    local_response=json.dumps(input_data, default=str) if isinstance(input_data, dict) else str(input_data),
                    metadata={
                        "capture_type": "output_formatting",
                        "model": "gpt-4o-mini",
                        "source": "openai_formatting",
                    },
                    teacher_model="gpt-4o-mini",
                )
                self.logger.debug(f"[Distillation] ✓ Captured formatting example for VULCAN LLM training")
            else:
                # Fallback: store in local queue
                self._distillation_queue.append(training_example)
                self.logger.debug(
                    f"[Distillation] Queued formatting example (queue_size={len(self._distillation_queue)})"
                )
        except Exception as e:
            self.logger.debug(f"[Distillation] Failed to capture formatting example: {e}")

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

    def warm_up(self, test_prompt: str = "Hello", max_tokens: int = 5) -> Dict[str, Any]:
        """
        Warm up the local LLM to reduce cold start latency.
        
        Note: CPU "Cold Start" and Cache Thrashing Prevention
        
        The first token generation on CPU can take 7+ seconds due to:
        1. Model weights being swapped from disk/compressed memory into RAM
        2. CPU cache not being primed with model data
        3. Lazy initialization of model layers
        
        This method runs a minimal generation to prime the system:
        - Loads model weights into active memory
        - Primes CPU caches
        - Initializes any lazy-loaded components
        
        Call this during application startup to reduce TTFT (Time To First Token)
        for the first real user query.
        
        Args:
            test_prompt: Simple prompt for warm-up (default: "Hello")
            max_tokens: Number of tokens to generate (default: 5, kept small for speed)
            
        Returns:
            Dict with warm-up results:
            - success: bool - Whether warm-up succeeded
            - warmup_time_ms: float - Time taken for warm-up in milliseconds
            - first_token_time_ms: float - Time to first token (if available)
            - error: str - Error message if warm-up failed
        """
        result = {
            "success": False,
            "warmup_time_ms": 0.0,
            "first_token_time_ms": None,
            "error": None,
        }
        
        if not self.local_llm:
            result["error"] = "No local LLM available for warm-up"
            self.logger.warning("[HybridExecutor] Warm-up skipped: No local LLM")
            return result
        
        self.logger.info("[HybridExecutor] Starting LLM warm-up to reduce cold start latency...")
        start_time = time.perf_counter()
        
        try:
            # Run a minimal generation to prime the system
            if hasattr(self.local_llm, 'generate'):
                warmup_result = self.local_llm.generate(test_prompt, max_tokens)
                
                # Check if result is valid
                if warmup_result is not None:
                    result["success"] = True
                    # Try to get first token time if available from result
                    if hasattr(warmup_result, 'first_token_time_ms'):
                        result["first_token_time_ms"] = warmup_result.first_token_time_ms
                else:
                    result["error"] = "Warm-up generation returned None"
            else:
                result["error"] = "Local LLM does not have generate() method"
                
        except Exception as e:
            result["error"] = f"Warm-up failed: {type(e).__name__}: {str(e)}"
            self.logger.warning(f"[HybridExecutor] Warm-up error: {result['error']}")
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result["warmup_time_ms"] = elapsed_ms
        
        if result["success"]:
            self.logger.info(
                f"[HybridExecutor] ✓ LLM warm-up complete in {elapsed_ms:.1f}ms - "
                f"subsequent queries should have reduced TTFT"
            )
        else:
            self.logger.warning(
                f"[HybridExecutor] ⚠ LLM warm-up incomplete ({elapsed_ms:.1f}ms): {result['error']}"
            )
        
        return result

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
    # Main class
    "HybridLLMExecutor",
    # Structured output format
    "VulcanReasoningOutput",
    # Cache
    "OpenAIResponseCache",
    # Singleton management
    "get_or_create_hybrid_executor",
    "get_hybrid_executor",
    "set_hybrid_executor",
    "verify_hybrid_executor_setup",
    # Configuration constants
    "OPENAI_LANGUAGE_FORMATTING",
    "OPENAI_LANGUAGE_POLISH",
    # CPU Cloud execution constants
    "VULCAN_HARD_TIMEOUT",
    "CPU_MAX_TOKENS_DEFAULT",
    # Adaptive timeout (Task 4 fix)
    "BASE_TIMEOUT_SECONDS",
    "TIMEOUT_PER_TOKEN_SECONDS",
    "calculate_adaptive_timeout",
    # P0 FIX: LLM-as-Reasoner bypass prevention
    "NotReasoningEngineError",
    "REASONING_TASK_INDICATORS",
]


# Log module initialization
logger.debug(f"Hybrid LLM executor module v{__version__} loaded")
