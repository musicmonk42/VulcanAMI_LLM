"""
Embedding Circuit Breaker for VulcanAMI

This module implements a circuit breaker pattern specifically for embedding operations
to handle the latency problem where embedding times grow exponentially under load.

The circuit breaker tracks embedding latency and can automatically skip embedding
operations when performance degrades, falling back to keyword-based selection.

Key Features:
- Tracks embedding latency with exponential moving average
- Opens circuit when latency exceeds threshold (default: 2 seconds)
- Half-open state allows testing if embeddings have recovered
- Automatic reset after recovery timeout
- Thread-safe implementation

Usage:
    from vulcan.reasoning.selection.embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
        get_embedding_circuit_breaker,
    )
    
    circuit_breaker = get_embedding_circuit_breaker()
    
    if circuit_breaker.should_skip_embedding():
        # Use keyword-based fallback
        pass
    else:
        # Try embedding
        start = time.perf_counter()
        try:
            embedding = model.encode(text)
            latency_ms = (time.perf_counter() - start) * 1000
            circuit_breaker.record_latency(latency_ms)
        except Exception:
            circuit_breaker.record_failure()
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, embeddings allowed
    OPEN = "open"          # Circuit tripped, skip embeddings
    HALF_OPEN = "half_open"  # Testing if embeddings have recovered


# Configuration constants
# PERFORMANCE FIX: Aggressive thresholds to prevent 6-30 second embedding delays
# Evidence from logs: Batches: 100%|██████████| 1/1 [00:20<00:00, 20.23s/it]
# Issue: SemanticToolMatcher taking 6-30 seconds per query
DEFAULT_LATENCY_THRESHOLD_MS = 1000.0  # 1 second - aggressive fail-fast on slow embeddings
DEFAULT_FAILURE_THRESHOLD = 2  # Only 2 slow operations before opening circuit (was 3)
DEFAULT_RESET_TIMEOUT_S = 60.0  # Wait longer before retrying (was 30s)
DEFAULT_SUCCESS_THRESHOLD = 3  # More successes needed to confirm recovery (was 2)
DEFAULT_EMA_ALPHA = 0.3  # Exponential moving average smoothing factor

# Log prefix for consistent output
LOG_PREFIX = "[EmbeddingCircuitBreaker]"


@dataclass
class CircuitBreakerStats:
    """Statistics for monitoring circuit breaker performance"""
    state: str
    failure_count: int
    success_count: int
    latency_ema_ms: float
    total_skipped: int
    total_allowed: int
    last_state_change: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "latency_ema_ms": self.latency_ema_ms,
            "total_skipped": self.total_skipped,
            "total_allowed": self.total_allowed,
            "last_state_change": self.last_state_change,
        }


class EmbeddingCircuitBreaker:
    """
    Circuit breaker for embedding operations to prevent runaway latency.
    
    This implements the circuit breaker pattern with latency-based triggering:
    
    1. CLOSED (normal): Embeddings are allowed. Track latency with EMA.
       - If latency exceeds threshold N times -> OPEN
    
    2. OPEN (tripped): Skip embeddings entirely, use keyword fallback.
       - After reset_timeout -> HALF_OPEN
    
    3. HALF_OPEN (testing): Allow limited embeddings to test recovery.
       - If fast enough N times -> CLOSED
       - If slow again -> OPEN
    
    Thread Safety:
        All public methods are thread-safe using RLock.
    
    Attributes:
        latency_threshold_ms: Latency (ms) considered "slow" that may trip circuit
        failure_threshold: Number of slow operations before opening circuit
        reset_timeout_s: Seconds before transitioning from OPEN to HALF_OPEN
        success_threshold: Successful operations in HALF_OPEN to close circuit
    """
    
    def __init__(
        self,
        latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
        success_threshold: int = DEFAULT_SUCCESS_THRESHOLD,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
    ):
        """
        Initialize embedding circuit breaker.
        
        Args:
            latency_threshold_ms: Latency threshold to consider operation slow.
            failure_threshold: Number of slow operations before opening circuit.
            reset_timeout_s: Seconds before testing recovery from open state.
            success_threshold: Fast operations needed in half-open to close.
            ema_alpha: Smoothing factor for exponential moving average (0-1).
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.success_threshold = success_threshold
        self.ema_alpha = ema_alpha
        
        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: Optional[float] = None
        
        # Latency tracking with exponential moving average
        self._latency_ema_ms = 0.0
        self._has_latency_data = False
        
        # Statistics
        self._total_skipped = 0
        self._total_allowed = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"{LOG_PREFIX} Initialized with thresholds: "
            f"latency={latency_threshold_ms}ms, "
            f"failures={failure_threshold}, "
            f"reset={reset_timeout_s}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            return self._state
    
    def should_skip_embedding(self) -> bool:
        """
        Check if embedding should be skipped due to circuit breaker.
        
        Returns:
            True if embeddings should be skipped (use fallback instead).
        
        Example:
            >>> if circuit_breaker.should_skip_embedding():
            ...     return keyword_based_features(query)
            >>> else:
            ...     return compute_embedding(query)
        """
        with self._lock:
            # Check for state transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_try_half_open():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._total_allowed += 1
                    return False
                
                self._total_skipped += 1
                return True
            
            # CLOSED or HALF_OPEN: allow embedding
            self._total_allowed += 1
            return False
    
    def _should_try_half_open(self) -> bool:
        """Check if we should transition from OPEN to HALF_OPEN"""
        if self._last_failure_time is None:
            return True
        
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.reset_timeout_s
    
    def record_latency(self, latency_ms: float) -> None:
        """
        Record an embedding operation's latency.
        
        Call this after each successful embedding operation.
        The circuit breaker uses this to determine if operations are slow.
        
        Args:
            latency_ms: Time taken for embedding operation in milliseconds.
        
        Example:
            >>> start = time.perf_counter()
            >>> embedding = model.encode(text)
            >>> latency_ms = (time.perf_counter() - start) * 1000
            >>> circuit_breaker.record_latency(latency_ms)
        """
        with self._lock:
            # Update exponential moving average
            if not self._has_latency_data:
                self._latency_ema_ms = latency_ms
                self._has_latency_data = True
            else:
                self._latency_ema_ms = (
                    self.ema_alpha * latency_ms +
                    (1 - self.ema_alpha) * self._latency_ema_ms
                )
            
            is_slow = latency_ms > self.latency_threshold_ms
            
            if self._state == CircuitState.CLOSED:
                self._handle_closed_state(latency_ms, is_slow)
            elif self._state == CircuitState.HALF_OPEN:
                self._handle_half_open_state(latency_ms, is_slow)
            # In OPEN state, we shouldn't be recording latency
            # (embeddings should be skipped)
    
    def _handle_closed_state(self, latency_ms: float, is_slow: bool) -> None:
        """Handle latency recording in CLOSED state"""
        if is_slow:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.warning(
                f"{LOG_PREFIX} Slow embedding: {latency_ms:.0f}ms "
                f"(threshold={self.latency_threshold_ms}ms, "
                f"failures={self._failure_count}/{self.failure_threshold})"
            )
            
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        else:
            # Successful operation: decay failure count
            self._failure_count = max(0, self._failure_count - 1)
    
    def _handle_half_open_state(self, latency_ms: float, is_slow: bool) -> None:
        """Handle latency recording in HALF_OPEN state"""
        if is_slow:
            # Still slow - go back to OPEN
            self._last_failure_time = time.time()
            self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"{LOG_PREFIX} Recovery failed: {latency_ms:.0f}ms - reopening circuit"
            )
        else:
            self._success_count += 1
            logger.info(
                f"{LOG_PREFIX} Recovery progress: {self._success_count}/{self.success_threshold}"
            )
            
            if self._success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
    
    def record_failure(self) -> None:
        """
        Record an embedding operation failure (exception, timeout, etc).
        
        Call this when an embedding operation fails for any reason.
        This is more severe than slow latency and immediately counts as a failure.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.warning(
                f"{LOG_PREFIX} Embedding failure recorded "
                f"(failures={self._failure_count}/{self.failure_threshold})"
            )
            
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new circuit state"""
        self._state = new_state
        self._last_state_change = time.time()
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"{LOG_PREFIX} Circuit CLOSED - embeddings fully enabled")
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            logger.warning(
                f"{LOG_PREFIX} Circuit OPEN - skipping embeddings for "
                f"{self.reset_timeout_s}s (latency_ema={self._latency_ema_ms:.0f}ms)"
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            logger.info(f"{LOG_PREFIX} Circuit HALF_OPEN - testing recovery")
    
    def force_reset(self) -> None:
        """
        Force reset the circuit breaker to CLOSED state.
        
        Use this for testing or when external conditions have changed
        (e.g., system resources freed up).
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_state_change = time.time()
            logger.info(f"{LOG_PREFIX} Circuit force reset to CLOSED")
    
    def get_stats(self) -> CircuitBreakerStats:
        """
        Get circuit breaker statistics for monitoring.
        
        Returns:
            CircuitBreakerStats with current metrics.
        """
        with self._lock:
            return CircuitBreakerStats(
                state=self._state.value,
                failure_count=self._failure_count,
                success_count=self._success_count,
                latency_ema_ms=self._latency_ema_ms,
                total_skipped=self._total_skipped,
                total_allowed=self._total_allowed,
                last_state_change=self._last_state_change,
            )


# =============================================================================
# Global Singleton Management
# =============================================================================

_embedding_circuit_breaker: Optional[EmbeddingCircuitBreaker] = None
_circuit_breaker_lock = threading.Lock()


def get_embedding_circuit_breaker(
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
) -> EmbeddingCircuitBreaker:
    """
    Get or create the global embedding circuit breaker singleton.
    
    Uses double-checked locking for thread-safe lazy initialization.
    
    Args:
        latency_threshold_ms: Latency threshold (only used on first call).
        failure_threshold: Failure threshold (only used on first call).
        reset_timeout_s: Reset timeout (only used on first call).
    
    Returns:
        Global EmbeddingCircuitBreaker instance.
    """
    global _embedding_circuit_breaker
    
    if _embedding_circuit_breaker is None:
        with _circuit_breaker_lock:
            if _embedding_circuit_breaker is None:
                _embedding_circuit_breaker = EmbeddingCircuitBreaker(
                    latency_threshold_ms=latency_threshold_ms,
                    failure_threshold=failure_threshold,
                    reset_timeout_s=reset_timeout_s,
                )
    
    return _embedding_circuit_breaker


def reset_embedding_circuit_breaker() -> None:
    """
    Reset the global embedding circuit breaker.
    
    This clears the singleton and allows a fresh instance to be created.
    Useful for testing or when configuration needs to change.
    """
    global _embedding_circuit_breaker
    
    with _circuit_breaker_lock:
        if _embedding_circuit_breaker is not None:
            _embedding_circuit_breaker.force_reset()
        _embedding_circuit_breaker = None


def get_circuit_breaker_stats() -> Dict[str, Any]:
    """
    Get circuit breaker statistics for monitoring.
    
    Returns:
        Dictionary with circuit breaker metrics, or empty dict if not initialized.
    """
    global _embedding_circuit_breaker
    
    if _embedding_circuit_breaker is None:
        return {"status": "not_initialized"}
    
    return _embedding_circuit_breaker.get_stats().to_dict()
