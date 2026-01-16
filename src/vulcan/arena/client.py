# ============================================================
# VULCAN-AGI Arena Client Module
# Client functions for Graphix Arena API integration
# ============================================================
#
# Arena is the training and execution environment for:
#     1. Agent training - Agents compete in tournaments, winners improve
#     2. Graph language runtime - Graphix IR graphs are executed and evolved
#     3. Language evolution - Successful patterns stored in Registry
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.1.0 - FIX #2: Added circuit breaker, timeout handling, and progress monitoring
# ============================================================

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

# Module metadata
__version__ = "1.2.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# Override default timeouts via environment variables
_env_timeout = os.getenv("VULCAN_ARENA_TIMEOUT")
if _env_timeout:
    try:
        GENERATOR_TIMEOUT = float(_env_timeout)
        logger.info(f"[ARENA] Using custom timeout from env: {GENERATOR_TIMEOUT}s")
    except ValueError:
        logger.warning(f"[ARENA] Invalid VULCAN_ARENA_TIMEOUT value: {_env_timeout}, using default")

# Redis URL for distributed circuit breaker
REDIS_URL = os.getenv("VULCAN_ARENA_REDIS_URL", os.getenv("REDIS_URL"))

# Max retry attempts for feedback submission
MAX_FEEDBACK_RETRIES = int(os.getenv("VULCAN_ARENA_FEEDBACK_RETRIES", "3"))

# ============================================================
# FIX #2: TIMEOUT AND CIRCUIT BREAKER CONSTANTS
# ============================================================

# PERFORMANCE FIX: Generator timeout increased from 45s to 90s to allow Arena
# operations to complete. Evidence from logs shows Arena completing in 47-53s.
# The 45s timeout was causing circuit breaker trips (2/3 consecutive timeouts)
# and wasting work that would have completed in 15-23 more seconds.
# Increased to 90s to accommodate complex tasks like tournaments, graph evolution,
# and agent training which can take 50-80s to complete.
GENERATOR_TIMEOUT = 90.0

# Simple task timeout for quick operations (feedback submission, health checks)
SIMPLE_TASK_TIMEOUT = 30.0

# Progress monitoring - abort if no progress after this time
PROGRESS_TIMEOUT_SECONDS = 15.0

# Maximum concurrent Arena requests to prevent overload
MAX_CONCURRENT_ARENA = 2

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 3  # Skip Arena after 3 consecutive timeouts
CIRCUIT_BREAKER_RESET_TIME = 60.0  # Reset circuit after 60s of no failures

# Issue #52: Hard timeout buffer for asyncio.wait_for wrapper
# Adds small buffer beyond aiohttp timeout to catch edge cases
TIMEOUT_BUFFER_SECONDS = 5.0


# ============================================================
# FIX #2: CIRCUIT BREAKER AND CONCURRENCY CONTROL
# ============================================================


@dataclass
class ArenaCircuitBreaker:
    """
    FIX #2: Circuit breaker to skip Arena when experiencing consecutive timeouts.

    The circuit breaker prevents wasted time waiting for Arena when it's
    experiencing issues. After CIRCUIT_BREAKER_THRESHOLD consecutive timeouts,
    the circuit "opens" and Arena is bypassed until CIRCUIT_BREAKER_RESET_TIME
    passes without any new failures.

    States:
        CLOSED: Normal operation, Arena calls proceed
        OPEN: Arena is bypassed, all calls return fallback immediately
    """

    consecutive_timeouts: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
    total_timeouts: int = 0
    total_successes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_timeout(self) -> None:
        """Record a timeout and potentially open the circuit."""
        with self._lock:
            self.consecutive_timeouts += 1
            self.total_timeouts += 1
            self.last_failure_time = time.time()

            if self.consecutive_timeouts >= CIRCUIT_BREAKER_THRESHOLD:
                if not self.is_open:
                    logger.warning(
                        f"[ARENA] Circuit breaker OPEN: {self.consecutive_timeouts} consecutive timeouts. "
                        f"Arena bypassed for {CIRCUIT_BREAKER_RESET_TIME}s"
                    )
                self.is_open = True

    def record_success(self) -> None:
        """Record a success and reset consecutive timeout counter."""
        with self._lock:
            self.consecutive_timeouts = 0
            self.total_successes += 1
            # Don't immediately close - let reset time handle it

    def should_bypass(self) -> bool:
        """
        Check if Arena should be bypassed.

        Returns:
            True if circuit is open and should bypass Arena
        """
        with self._lock:
            if not self.is_open:
                return False

            # Check if enough time has passed to try again
            elapsed = time.time() - self.last_failure_time
            if elapsed >= CIRCUIT_BREAKER_RESET_TIME:
                logger.info(
                    f"[ARENA] Circuit breaker RESET: {elapsed:.1f}s since last failure. "
                    f"Attempting Arena call."
                )
                self.is_open = False
                self.consecutive_timeouts = 0
                return False

            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "is_open": self.is_open,
                "consecutive_timeouts": self.consecutive_timeouts,
                "total_timeouts": self.total_timeouts,
                "total_successes": self.total_successes,
                "last_failure_time": self.last_failure_time,
                "time_since_failure": (
                    time.time() - self.last_failure_time
                    if self.last_failure_time > 0
                    else None
                ),
            }


@dataclass
class DistributedCircuitBreaker:
    """
    Redis-backed circuit breaker for multi-worker deployments.
    
    In production environments with multiple Uvicorn/Gunicorn workers, each worker
    has its own process memory. A process-local circuit breaker causes inconsistent
    behavior where one worker trips the breaker while others continue hammering
    the failing service (thundering herd problem).
    
    This distributed circuit breaker uses Redis to share state across all workers,
    ensuring consistent behavior. Falls back to local circuit breaker if Redis
    is unavailable (graceful degradation).
    
    Redis Schema:
        vulcan:arena:circuit_breaker:consecutive_timeouts - INT with TTL
        vulcan:arena:circuit_breaker:total_timeouts - INT (no TTL)
        vulcan:arena:circuit_breaker:total_successes - INT (no TTL)
        vulcan:arena:circuit_breaker:last_failure_time - FLOAT (no TTL)
        vulcan:arena:circuit_breaker:is_open - BOOL with TTL
    """
    
    REDIS_KEY_PREFIX = "vulcan:arena:circuit_breaker:"
    
    _redis: Optional[Any] = None
    _local_fallback: ArenaCircuitBreaker = field(default_factory=ArenaCircuitBreaker)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _redis_available: bool = False
    
    def __post_init__(self):
        """Initialize Redis connection if available."""
        if REDIS_URL and not self._redis:
            try:
                import redis
                self._redis = redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self._redis.ping()
                self._redis_available = True
                logger.info("[ARENA] Distributed circuit breaker initialized with Redis")
            except ImportError:
                logger.warning("[ARENA] redis package not available, using local circuit breaker")
                self._redis_available = False
            except Exception as e:
                logger.warning(f"[ARENA] Redis connection failed: {e}, using local circuit breaker")
                self._redis_available = False
    
    def _get_redis_key(self, suffix: str) -> str:
        """Generate Redis key for circuit breaker state."""
        return f"{self.REDIS_KEY_PREFIX}{suffix}"
    
    def record_timeout(self) -> None:
        """Record a timeout and potentially open the circuit."""
        if self._redis_available and self._redis:
            try:
                with self._lock:
                    pipe = self._redis.pipeline()
                    
                    # Increment consecutive timeouts
                    consecutive_key = self._get_redis_key("consecutive_timeouts")
                    pipe.incr(consecutive_key)
                    pipe.expire(consecutive_key, int(CIRCUIT_BREAKER_RESET_TIME))
                    
                    # Increment total timeouts
                    total_key = self._get_redis_key("total_timeouts")
                    pipe.incr(total_key)
                    
                    # Update last failure time
                    failure_time_key = self._get_redis_key("last_failure_time")
                    current_time = time.time()
                    pipe.set(failure_time_key, current_time)
                    
                    results = pipe.execute()
                    consecutive_count = results[0]  # Result of incr
                    
                    # Open circuit if threshold reached
                    if consecutive_count >= CIRCUIT_BREAKER_THRESHOLD:
                        is_open_key = self._get_redis_key("is_open")
                        self._redis.set(is_open_key, "1", ex=int(CIRCUIT_BREAKER_RESET_TIME))
                        
                        # Log only on transition to open
                        if consecutive_count == CIRCUIT_BREAKER_THRESHOLD:
                            logger.warning(
                                f"[ARENA] Distributed circuit breaker OPEN: {consecutive_count} consecutive timeouts. "
                                f"Arena bypassed for {CIRCUIT_BREAKER_RESET_TIME}s across all workers"
                            )
            except Exception as e:
                logger.warning(f"[ARENA] Redis operation failed in record_timeout: {e}, using local fallback")
                self._local_fallback.record_timeout()
        else:
            self._local_fallback.record_timeout()
    
    def record_success(self) -> None:
        """Record a success and reset consecutive timeout counter."""
        if self._redis_available and self._redis:
            try:
                with self._lock:
                    pipe = self._redis.pipeline()
                    
                    # Reset consecutive timeouts
                    consecutive_key = self._get_redis_key("consecutive_timeouts")
                    pipe.delete(consecutive_key)
                    
                    # Increment total successes
                    success_key = self._get_redis_key("total_successes")
                    pipe.incr(success_key)
                    
                    pipe.execute()
            except Exception as e:
                logger.warning(f"[ARENA] Redis operation failed in record_success: {e}, using local fallback")
                self._local_fallback.record_success()
        else:
            self._local_fallback.record_success()
    
    def should_bypass(self) -> bool:
        """
        Check if Arena should be bypassed.
        
        Returns:
            True if circuit is open and should bypass Arena
        """
        if self._redis_available and self._redis:
            try:
                with self._lock:
                    is_open_key = self._get_redis_key("is_open")
                    is_open = self._redis.get(is_open_key)
                    
                    if is_open == "1":
                        # Circuit is open - check if enough time has passed
                        failure_time_key = self._get_redis_key("last_failure_time")
                        last_failure = self._redis.get(failure_time_key)
                        
                        if last_failure:
                            elapsed = time.time() - float(last_failure)
                            if elapsed >= CIRCUIT_BREAKER_RESET_TIME:
                                # Reset circuit
                                pipe = self._redis.pipeline()
                                pipe.delete(is_open_key)
                                pipe.delete(self._get_redis_key("consecutive_timeouts"))
                                pipe.execute()
                                
                                logger.info(
                                    f"[ARENA] Distributed circuit breaker RESET: {elapsed:.1f}s since last failure. "
                                    f"Attempting Arena call."
                                )
                                return False
                        
                        return True
                    
                    return False
            except Exception as e:
                logger.warning(f"[ARENA] Redis operation failed in should_bypass: {e}, using local fallback")
                return self._local_fallback.should_bypass()
        else:
            return self._local_fallback.should_bypass()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        if self._redis_available and self._redis:
            try:
                with self._lock:
                    pipe = self._redis.pipeline()
                    pipe.get(self._get_redis_key("consecutive_timeouts"))
                    pipe.get(self._get_redis_key("total_timeouts"))
                    pipe.get(self._get_redis_key("total_successes"))
                    pipe.get(self._get_redis_key("last_failure_time"))
                    pipe.get(self._get_redis_key("is_open"))
                    
                    results = pipe.execute()
                    
                    consecutive_timeouts = int(results[0] or 0)
                    total_timeouts = int(results[1] or 0)
                    total_successes = int(results[2] or 0)
                    last_failure_time = float(results[3] or 0)
                    is_open = results[4] == "1"
                    
                    return {
                        "is_open": is_open,
                        "consecutive_timeouts": consecutive_timeouts,
                        "total_timeouts": total_timeouts,
                        "total_successes": total_successes,
                        "last_failure_time": last_failure_time,
                        "time_since_failure": (
                            time.time() - last_failure_time
                            if last_failure_time > 0
                            else None
                        ),
                        "distributed": True,
                        "redis_available": True,
                    }
            except Exception as e:
                logger.warning(f"[ARENA] Redis operation failed in get_stats: {e}, using local fallback")
                stats = self._local_fallback.get_stats()
                stats["distributed"] = False
                stats["redis_available"] = False
                return stats
        else:
            stats = self._local_fallback.get_stats()
            stats["distributed"] = False
            stats["redis_available"] = False
            return stats


class FeedbackRetryQueue:
    """
    Async queue for retrying failed feedback submissions with exponential backoff.
    
    When feedback submission fails due to transient errors (network blips, temporary
    service unavailability), the RLHF training data would be permanently lost with
    fire-and-forget submission. This queue ensures feedback is eventually delivered
    by retrying with exponential backoff.
    
    Features:
        - Exponential backoff: 1s → 2s → 4s → 8s → ... → 30s max
        - Jitter: Randomized delays to prevent thundering herd
        - Max retries: Configurable (default 3 attempts)
        - Background worker: Async task that processes queue without blocking
        - Thread-safe: Can be called from multiple async contexts
    
    Usage:
        queue = FeedbackRetryQueue()
        await queue.enqueue(feedback_data)
        # Background worker automatically retries failed submissions
    """
    
    MAX_RETRIES = MAX_FEEDBACK_RETRIES
    INITIAL_DELAY = 1.0  # seconds
    MAX_DELAY = 30.0  # seconds
    
    def __init__(self):
        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of queue and worker (thread-safe)."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    self._queue = asyncio.Queue()
                    self._worker_task = asyncio.create_task(self._retry_worker())
                    self._initialized = True
                    logger.info("[ARENA] Feedback retry queue initialized")
    
    async def enqueue(self, feedback_data: dict) -> None:
        """
        Add failed feedback to retry queue.
        
        Args:
            feedback_data: The feedback payload that failed to submit
        """
        await self._ensure_initialized()
        
        retry_item = {
            "data": feedback_data,
            "attempts": 0,
            "next_retry": time.time()
        }
        
        await self._queue.put(retry_item)
        logger.info(f"[ARENA] Enqueued feedback for retry: {feedback_data.get('graph_id', 'unknown')}")
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        import random
        
        # Exponential backoff: delay = initial * (2 ^ attempt)
        delay = self.INITIAL_DELAY * (2 ** attempt)
        delay = min(delay, self.MAX_DELAY)
        
        # Add jitter: ±25% randomization to prevent thundering herd
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay += jitter
        
        return max(0, delay)
    
    async def _retry_worker(self) -> None:
        """Background worker that processes retry queue."""
        logger.info("[ARENA] Feedback retry worker started")
        
        while True:
            try:
                # Get next item from queue
                retry_item = await self._queue.get()
                
                # Wait until it's time to retry
                now = time.time()
                if retry_item["next_retry"] > now:
                    delay = retry_item["next_retry"] - now
                    await asyncio.sleep(delay)
                
                # Attempt to submit feedback
                feedback_data = retry_item["data"]
                attempt = retry_item["attempts"]
                
                logger.info(
                    f"[ARENA] Retrying feedback submission "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES}): "
                    f"{feedback_data.get('graph_id', 'unknown')}"
                )
                
                # Import here to avoid circular dependency
                from vulcan.arena.client import submit_arena_feedback
                
                result = await submit_arena_feedback(
                    proposal_id=feedback_data["graph_id"],
                    score=feedback_data["score"],
                    rationale=feedback_data["rationale"],
                    arena_base_url=feedback_data.get("arena_base_url")
                )
                
                if result.get("status") == "success":
                    logger.info(
                        f"[ARENA] Feedback retry succeeded "
                        f"(attempt {attempt + 1}): "
                        f"{feedback_data.get('graph_id', 'unknown')}"
                    )
                elif attempt + 1 < self.MAX_RETRIES:
                    # Schedule next retry with exponential backoff
                    backoff = self._calculate_backoff(attempt + 1)
                    retry_item["attempts"] = attempt + 1
                    retry_item["next_retry"] = time.time() + backoff
                    
                    await self._queue.put(retry_item)
                    logger.warning(
                        f"[ARENA] Feedback retry failed, will retry in {backoff:.1f}s "
                        f"(attempt {attempt + 2}/{self.MAX_RETRIES}): "
                        f"{feedback_data.get('graph_id', 'unknown')}"
                    )
                else:
                    # Max retries exceeded
                    logger.error(
                        f"[ARENA] Feedback permanently failed after {self.MAX_RETRIES} attempts: "
                        f"{feedback_data.get('graph_id', 'unknown')}"
                    )
                
                self._queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("[ARENA] Feedback retry worker cancelled")
                break
            except Exception as e:
                logger.error(f"[ARENA] Feedback retry worker error: {e}")
                # Continue processing queue despite errors
                await asyncio.sleep(1)
    
    async def shutdown(self):
        """Gracefully shutdown the retry queue and worker."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[ARENA] Feedback retry queue shutdown")


# Global circuit breaker instance - use distributed version if Redis available
_circuit_breaker = DistributedCircuitBreaker()

# Global feedback retry queue (lazy initialization)
_feedback_retry_queue: Optional[FeedbackRetryQueue] = None

# Semaphore for limiting concurrent Arena requests
_arena_semaphore: Optional[asyncio.Semaphore] = None
_arena_semaphore_lock = threading.Lock()


def _get_arena_semaphore() -> asyncio.Semaphore:
    """
    Get or create the Arena concurrency semaphore with thread-safe initialization.
    
    Uses double-check locking pattern to prevent race conditions during
    semaphore initialization when multiple threads access this function concurrently.
    
    Returns:
        asyncio.Semaphore: The global Arena semaphore
    """
    global _arena_semaphore
    if _arena_semaphore is None:
        with _arena_semaphore_lock:
            if _arena_semaphore is None:  # Double-check locking
                _arena_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ARENA)
    return _arena_semaphore


def get_circuit_breaker_stats() -> Dict[str, Any]:
    """
    FIX #2: Get Arena circuit breaker statistics for monitoring.

    Returns:
        Dictionary with circuit breaker state and metrics
    """
    return _circuit_breaker.get_stats()


def reset_circuit_breaker() -> None:
    """
    FIX #2: Manually reset the circuit breaker.

    Use this to force Arena to be tried again after manual intervention.
    """
    global _circuit_breaker
    _circuit_breaker = DistributedCircuitBreaker()
    logger.info("[ARENA] Circuit breaker manually reset")


# ============================================================
# IMPORTS
# ============================================================

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False
    logger.info("aiohttp not available - Arena API calls disabled")

try:
    from vulcan.settings import settings
except ImportError:
    settings = None
    logger.warning("Settings module not available - using defaults")

try:
    from vulcan.arena.http_session import get_http_session
except ImportError:
    get_http_session = None
    logger.warning("HTTP session module not available")

try:
    from vulcan.utils_main.sanitize import deep_sanitize_for_json, sanitize_payload
except ImportError:
    # Fallback implementations
    def sanitize_payload(data):
        if isinstance(data, dict):
            return {
                str(k): sanitize_payload(v) for k, v in data.items() if k is not None
            }
        elif isinstance(data, list):
            return [sanitize_payload(item) for item in data]
        return data

    def deep_sanitize_for_json(data, _depth=0):
        return sanitize_payload(data)


# ============================================================
# AGENT SELECTION
# ============================================================


def select_arena_agent(routing_plan) -> str:
    """
    Map query type to appropriate Arena agent.

    Arena agents:
    - generator: Creates new graphs from specs (creative/generative tasks)
    - evolver: Mutates existing graphs (optimization/evolution tasks)
    - visualizer: Renders graphs (explanation/visualization tasks)
    - photonic_optimizer: Hardware optimization (photonic/analog tasks)
    - automl_optimizer: Model tuning (hyperparameter/automl tasks)

    Args:
        routing_plan: The routing plan from QueryRouter

    Returns:
        Agent ID string
    """
    query_type = (
        routing_plan.query_type.value
        if hasattr(routing_plan.query_type, "value")
        else str(routing_plan.query_type)
    )

    agent_mapping = {
        # Creative/generative → generator
        "GENERATIVE": "generator",
        "generative": "generator",
        "generation": "generator",
        "creative": "generator",
        "design": "generator",
        # Optimization → evolver
        "OPTIMIZATION": "evolver",
        "optimization": "evolver",
        "evolution": "evolver",
        "improve": "evolver",
        # Explanation → visualizer
        "PERCEPTION": "visualizer",
        "perception": "visualizer",
        "visualization": "visualizer",
        "explain": "visualizer",
        # Reasoning can use generator for graph-based reasoning
        "REASONING": "generator",
        "reasoning": "generator",
        # Planning uses generator for planning graphs
        "PLANNING": "generator",
        "planning": "generator",
        # Hardware → photonic_optimizer
        "photonic": "photonic_optimizer",
        "hardware": "photonic_optimizer",
        "analog": "photonic_optimizer",
        # Model tuning → automl_optimizer
        "automl": "automl_optimizer",
        "hyperparameter": "automl_optimizer",
        "tune": "automl_optimizer",
    }

    agent_id = agent_mapping.get(query_type)
    if agent_id is None:
        logger.warning(
            f"[ARENA] Unknown query type '{query_type}', defaulting to generator"
        )
        return "generator"
    return agent_id


def build_arena_payload(query: str, routing_plan, agent_id: str) -> dict:
    """
    Build payload for Arena API based on agent type.

    Generator expects GraphSpec format, others expect GraphixIRGraph format.

    Args:
        query: The user query
        routing_plan: The routing plan from QueryRouter
        agent_id: The target agent ID

    Returns:
        Payload dictionary for Arena API
    """
    query_id = (
        routing_plan.query_id
        if hasattr(routing_plan, "query_id")
        else f"q_{int(time.time() * 1000)}"
    )
    query_type = (
        routing_plan.query_type.value
        if hasattr(routing_plan.query_type, "value")
        else str(routing_plan.query_type)
    )
    complexity = (
        routing_plan.complexity_score
        if hasattr(routing_plan, "complexity_score")
        else 0.5
    )

    # CRITICAL FIX: Extract selected_tools from routing_plan for reasoning invocation
    # This enables GraphixArena to invoke reasoning engines when selected_tools are present
    selected_tools = getattr(routing_plan, "selected_tools", []) or []

    if agent_id == "generator":
        # Generator expects GraphSpec format
        return {
            "spec_id": f"query_{query_id}",
            "parameters": {
                "goal": query,
                "query_type": query_type,
                "complexity": complexity,
                "source": "vulcan",
                "timestamp": time.time(),
                "selected_tools": selected_tools,  # Pass reasoning tools to Arena
            },
        }
    else:
        # Evolver, visualizer, etc. expect GraphixIRGraph format
        return {
            "graph_id": f"g_{query_id}",
            "nodes": [
                {
                    "id": "root",
                    "label": "query_input",
                    "properties": {
                        "text": query,
                        "query_type": query_type,
                    },
                }
            ],
            "edges": [],
            "properties": {
                "source": "vulcan",
                "query_id": query_id,
                "query_type": query_type,
                "complexity": complexity,
                "timestamp": time.time(),
                "selected_tools": selected_tools,  # Pass reasoning tools to Arena
            },
        }


# ============================================================
# ARENA API FUNCTIONS
# ============================================================


async def execute_via_arena(
    query: str, routing_plan, arena_base_url: str = None
) -> dict:
    """
    Execute query through Graphix Arena for training + graph execution.

    FIX #2: Now includes:
    - Circuit breaker to skip Arena after consecutive timeouts
    - Concurrency limiting (MAX_CONCURRENT_ARENA simultaneous requests)
    - Proper timeout (GENERATOR_TIMEOUT = 45s to allow Arena completion)
    - Logging when Arena is bypassed due to performance issues

    Arena handles:
    - Agent execution (generator/evolver/visualizer/etc)
    - Tournament selection among proposals
    - Feedback integration (RLHF)
    - Governance enforcement

    Args:
        query: The user query to process
        routing_plan: The routing plan from QueryRouter
        arena_base_url: Base URL for Arena API (defaults to settings)

    Returns:
        dict with Arena execution result including:
        - result: The agent's output
        - agent_id: Which agent was used
        - execution_time: How long it took
        - metrics: Any performance metrics from Arena
    """
    # FIX #2: Check circuit breaker first - skip Arena if experiencing issues
    if _circuit_breaker.should_bypass():
        cb_stats = _circuit_breaker.get_stats()
        logger.warning(
            f"[ARENA] Circuit breaker OPEN - bypassing Arena. "
            f"Stats: {cb_stats['consecutive_timeouts']} consecutive timeouts, "
            f"{cb_stats['time_since_failure']:.1f}s since last failure"
        )
        return {
            "status": "circuit_breaker_open",
            "reason": f"Arena bypassed due to {cb_stats['consecutive_timeouts']} consecutive timeouts",
            "result": None,
            "execution_time": 0,
            "circuit_breaker_stats": cb_stats,
        }

    if not AIOHTTP_AVAILABLE:
        logger.warning(
            "[ARENA] aiohttp not available, falling back to VULCAN-only processing"
        )
        return {
            "status": "fallback",
            "reason": "aiohttp not available for Arena API calls",
            "result": None,
        }

    if get_http_session is None:
        logger.warning("[ARENA] HTTP session not available")
        return {
            "status": "error",
            "reason": "HTTP session module not available",
            "result": None,
        }

    # Get Arena configuration
    base_url = arena_base_url
    api_key = None
    # FIX #2: Use GENERATOR_TIMEOUT (30s) instead of 120s
    timeout = GENERATOR_TIMEOUT
    complexity_threshold = 0.3  # PERFORMANCE FIX: Default fast-path threshold

    if settings is not None:
        base_url = base_url or settings.arena_base_url
        api_key = settings.arena_api_key
        # FIX #2: Cap timeout at GENERATOR_TIMEOUT even if settings has higher value
        settings_timeout = settings.arena_timeout
        timeout = (
            min(settings_timeout, GENERATOR_TIMEOUT)
            if settings_timeout
            else GENERATOR_TIMEOUT
        )
        complexity_threshold = getattr(settings, "arena_complexity_threshold", 0.3)
    else:
        base_url = base_url or "http://localhost:8080/arena"

    # Note: Improved Arena threshold logic
    # Previously defaulted complexity to 0.0 which always skipped Arena
    # Now: If complexity_score is explicitly set AND below threshold, skip
    #      If complexity_score is not set, proceed with Arena (let it decide)
    complexity = getattr(routing_plan, "complexity_score", None)

    if complexity is not None:
        if complexity < complexity_threshold:
            logger.info(
                f"[ARENA] Fast-path skip: complexity {complexity:.2f} < threshold {complexity_threshold:.2f}"
            )
            return {
                "status": "skipped",
                "reason": f"Query complexity ({complexity:.2f}) below threshold ({complexity_threshold:.2f})",
                "result": None,
                "execution_time": 0,
            }
        else:
            logger.info(
                f"[ARENA] Proceeding: complexity {complexity:.2f} >= threshold {complexity_threshold:.2f}"
            )
    else:
        # No complexity score provided - check if arena_participation flag is set
        arena_flag = getattr(routing_plan, "arena_participation", False)
        if not arena_flag:
            logger.info(
                "[ARENA] Skipping: no complexity_score and arena_participation=False"
            )
            return {
                "status": "skipped",
                "reason": "No complexity_score provided and arena_participation not enabled",
                "result": None,
                "execution_time": 0,
            }
        logger.info(
            "[ARENA] Proceeding: arena_participation=True (no complexity_score)"
        )

    # Select appropriate Arena agent
    agent_id = select_arena_agent(routing_plan)

    # Build payload for Arena
    payload = build_arena_payload(query, routing_plan, agent_id)

    # CRITICAL FIX: Sanitize payload to remove None keys that cause serialization failures
    payload = sanitize_payload(payload)
    
    # Validate that required fields are present after sanitization
    required_fields = ['spec_id', 'parameters'] if agent_id == 'generator' else ['graph_id', 'nodes']
    if not all(key in payload for key in required_fields):
        logger.error(f"[ARENA] Sanitization removed required fields: {required_fields}")
        return {
            "status": "error",
            "error": "Payload sanitization removed required data",
        }

    # CRITICAL FIX: Pre-serialize to JSON to catch serialization errors early
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError) as json_err:
        logger.error(f"[ARENA] JSON serialization failed: {json_err}")
        # Attempt deep sanitization as fallback
        payload = deep_sanitize_for_json(payload)
        try:
            payload_json = json.dumps(payload)
            logger.info("[ARENA] Deep sanitization succeeded, retrying serialization")
        except Exception as retry_err:
            logger.error(f"[ARENA] Deep sanitization also failed: {retry_err}")
            return {
                "status": "error",
                "agent_id": agent_id,
                "execution_time": 0,
                "error": f"JSON serialization failed (original: {json_err}, after deep sanitize: {retry_err})",
            }

    # Construct Arena API URL
    url = f"{base_url}/api/run/{agent_id}"

    # Warn if API key is not configured
    if not api_key:
        logger.warning(
            "[ARENA] API key not configured - Arena request may fail authentication"
        )

    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    logger.info(f"[ARENA] Executing via {agent_id}: {url} (timeout={timeout}s)")
    t0 = time.perf_counter()

    # FIX #2: Use semaphore to limit concurrent Arena requests
    semaphore = _get_arena_semaphore()

    # Note: Two-layer timeout strategy:
    # 1. Inner: aiohttp.ClientTimeout handles HTTP transport-level timeout
    # 2. Outer: asyncio.wait_for provides hard cutoff for entire async operation
    # The outer timeout is slightly longer to allow clean aiohttp timeout handling
    async def _execute_request():
        async with semaphore:
            session = await get_http_session()
            async with session.post(
                url,
                data=payload_json,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {"status": 200, "result": result}
                else:
                    error_text = await resp.text()
                    return {"status": resp.status, "error": error_text}

    try:
        # Outer timeout: hard limit = configured timeout + buffer for clean handling
        response = await asyncio.wait_for(
            _execute_request(), timeout=timeout + TIMEOUT_BUFFER_SECONDS
        )
        elapsed = time.perf_counter() - t0

        if response["status"] == 200:
            # FIX #2: Record success for circuit breaker
            _circuit_breaker.record_success()
            logger.info(f"[ARENA] {agent_id} completed in {elapsed:.2f}s")
            return {
                "status": "success",
                "agent_id": agent_id,
                "execution_time": elapsed,
                "result": response["result"],
                "arena_url": url,
            }
        else:
            logger.error(
                f"[ARENA] {agent_id} failed: {response['status']} - {response.get('error', 'Unknown')}"
            )
            return {
                "status": "error",
                "agent_id": agent_id,
                "execution_time": elapsed,
                "error": response.get("error", "Unknown error"),
                "status_code": response["status"],
            }

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        # FIX #2: Record timeout for circuit breaker
        _circuit_breaker.record_timeout()
        cb_stats = _circuit_breaker.get_stats()
        logger.error(
            f"[ARENA] {agent_id} timeout after {elapsed:.2f}s. "
            f"Circuit breaker: {cb_stats['consecutive_timeouts']}/{CIRCUIT_BREAKER_THRESHOLD} consecutive timeouts"
        )
        return {
            "status": "timeout",
            "agent_id": agent_id,
            "execution_time": elapsed,
            "error": f"Arena request timed out after {timeout}s",
            "circuit_breaker_stats": cb_stats,
        }
    except aiohttp.ClientError as e:
        elapsed = time.perf_counter() - t0
        # FIX #2: Record as timeout for circuit breaker (connection issues count as failures)
        _circuit_breaker.record_timeout()
        logger.error(f"[ARENA] {agent_id} connection error: {e}")
        return {
            "status": "connection_error",
            "agent_id": agent_id,
            "execution_time": elapsed,
            "error": str(e),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"[ARENA] {agent_id} unexpected error: {e}")
        return {
            "status": "error",
            "agent_id": agent_id,
            "execution_time": elapsed,
            "error": str(e),
        }


async def submit_arena_feedback(
    proposal_id: str, score: float, rationale: str, arena_base_url: str = None
) -> dict:
    """
    Submit feedback to Arena for RLHF (Reinforcement Learning from Human Feedback).
    
    Now includes automatic retry queue for failed submissions. If submission fails,
    feedback is automatically enqueued for retry with exponential backoff to ensure
    RLHF training data is not lost due to transient failures.

    This enables the evolution loop where:
    - User feedback influences agent training
    - Successful patterns are reinforced
    - Losers get diversity penalty applied

    Args:
        proposal_id: ID of the proposal/graph to provide feedback on
        score: Feedback score (typically -1.0 to 1.0)
        rationale: Human-readable explanation of the feedback
        arena_base_url: Base URL for Arena API

    Returns:
        dict with feedback submission result
    """
    global _feedback_retry_queue
    
    if not AIOHTTP_AVAILABLE:
        logger.warning("[ARENA] aiohttp not available, cannot submit feedback")
        return {"status": "error", "reason": "aiohttp not available"}

    if get_http_session is None:
        return {"status": "error", "reason": "HTTP session module not available"}

    # Get configuration
    base_url = arena_base_url
    api_key = None

    if settings is not None:
        base_url = base_url or settings.arena_base_url
        api_key = settings.arena_api_key
    else:
        base_url = base_url or "http://localhost:8080/arena"

    url = f"{base_url}/api/feedback"

    payload = {
        "graph_id": proposal_id,
        "agent_id": "vulcan",
        "score": score,
        "rationale": rationale,
    }

    # Sanitize and serialize
    payload = sanitize_payload(payload)
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError) as json_err:
        logger.error(f"[ARENA] Feedback JSON serialization failed: {json_err}")
        payload = deep_sanitize_for_json(payload)
        try:
            payload_json = json.dumps(payload)
        except Exception as retry_err:
            return {"status": "error", "error": f"JSON serialization failed"}

    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        session = await get_http_session()
        # Use SIMPLE_TASK_TIMEOUT for feedback submission (quick operation)
        async with session.post(
            url,
            data=payload_json,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=SIMPLE_TASK_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                logger.info(
                    f"[ARENA] Feedback submitted for {proposal_id}: score={score}"
                )
                return {"status": "success", "result": result}
            else:
                error_text = await resp.text()
                logger.error(f"[ARENA] Feedback submission failed: {resp.status}")
                
                # Enqueue for retry
                if _feedback_retry_queue is None:
                    _feedback_retry_queue = FeedbackRetryQueue()
                
                feedback_data = {
                    "graph_id": proposal_id,
                    "score": score,
                    "rationale": rationale,
                    "arena_base_url": base_url
                }
                await _feedback_retry_queue.enqueue(feedback_data)
                
                return {"status": "error", "error": error_text, "retry_queued": True}

    except Exception as e:
        logger.error(f"[ARENA] Feedback submission error: {e}")
        
        # Enqueue for retry on any exception
        if _feedback_retry_queue is None:
            _feedback_retry_queue = FeedbackRetryQueue()
        
        feedback_data = {
            "graph_id": proposal_id,
            "score": score,
            "rationale": rationale,
            "arena_base_url": base_url
        }
        await _feedback_retry_queue.enqueue(feedback_data)
        
        return {"status": "error", "error": str(e), "retry_queued": True}


# Backward compatibility aliases
_select_arena_agent = select_arena_agent
_build_arena_payload = build_arena_payload
_execute_via_arena = execute_via_arena
_submit_arena_feedback = submit_arena_feedback


__all__ = [
    "execute_via_arena",
    "submit_arena_feedback",
    "select_arena_agent",
    "build_arena_payload",
    "AIOHTTP_AVAILABLE",
    # FIX #2: Circuit breaker and timeout constants
    "GENERATOR_TIMEOUT",
    "SIMPLE_TASK_TIMEOUT",
    "MAX_CONCURRENT_ARENA",
    "CIRCUIT_BREAKER_THRESHOLD",
    "get_circuit_breaker_stats",
    "reset_circuit_breaker",
    # New resilience classes
    "ArenaCircuitBreaker",
    "DistributedCircuitBreaker",
    "FeedbackRetryQueue",
    # Backward compatibility
    "_select_arena_agent",
    "_build_arena_payload",
    "_execute_via_arena",
    "_submit_arena_feedback",
]
