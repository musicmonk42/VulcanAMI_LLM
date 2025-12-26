"""
outcome_queue.py - Thread-safe queue for sharing query outcomes
Part of the VULCAN-AGI system

This module provides a shared buffer for query outcomes that enables data flow
from the main request handling pipeline to the curiosity engine's learning loop.

BUG #3 FIX: This solves the problem of the curiosity engine finding 0 knowledge
gaps because it had no data. The OutcomeBuffer collects query outcomes from the
main process and makes them available for the curiosity engine to analyze.

Architecture:
    MAIN PROCESS                              SUBPROCESS (Curiosity Engine)
    ─────────────────────────────────────────────────────────────────
    Query Router ──┐                          ┌── GapAnalyzer
    Agent Pool ────┼── QueryOutcome ──────▶   ├── DependencyGraph  
    LLM Response ──┘    (OutcomeBuffer)       ├── ExperimentGenerator
                                              └── CuriosityEngine

Thread Safety:
    All public methods are thread-safe. The OutcomeBuffer uses a reentrant
    lock (RLock) to allow nested locking from the same thread while preventing
    concurrent modification from different threads.

Memory Management:
    The buffer uses a bounded deque to prevent unbounded memory growth.
    When the buffer reaches max_size, oldest entries are automatically evicted.

Usage:
    # In main request handler (producer):
    from vulcan.curiosity_engine import record_outcome, QueryOutcome, OutcomeStatus
    
    outcome = QueryOutcome(
        query_id="q_abc123",
        query_type="reasoning",
        status=OutcomeStatus.SUCCESS,
        execution_time_ms=4500.0,
    )
    outcome.compute_features()
    record_outcome(outcome)
    
    # In curiosity engine (consumer):
    from vulcan.curiosity_engine import get_outcome_buffer
    
    buffer = get_outcome_buffer()
    outcomes = buffer.get_batch(max_items=50)
    for outcome in outcomes:
        gap_analyzer.ingest_query_result(...)
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from .query_outcome import QueryOutcome, OutcomeStatus

logger = logging.getLogger(__name__)


# =============================================================================
# OUTCOME BUFFER CLASS
# =============================================================================


class OutcomeBuffer:
    """
    Thread-safe buffer for query outcomes.
    
    Implements a producer-consumer pattern where main request handlers
    (producers) add outcomes and the curiosity engine (consumer) retrieves
    batches for analysis.
    
    EXAMINE: Accepts QueryOutcome objects with full processing metrics
    SELECT: Uses bounded deque for automatic LRU eviction
    APPLY: Thread-safe add/get operations with RLock
    REMEMBER: Tracks statistics for monitoring and debugging
    
    Thread Safety:
        All public methods acquire the internal RLock before accessing
        the buffer. RLock allows recursive locking from the same thread,
        which is useful for compound operations.
    
    Memory Management:
        Uses collections.deque with maxlen to automatically evict oldest
        entries when capacity is reached. This provides O(1) append and
        popleft operations with bounded memory usage.
    
    Attributes:
        max_size: Maximum number of outcomes to store
        _buffer: Internal deque storage
        _lock: Reentrant lock for thread safety
        _total_added: Counter for total outcomes added (for statistics)
        _total_consumed: Counter for total outcomes consumed
        
    Example:
        >>> buffer = OutcomeBuffer(max_size=1000)
        >>> buffer.add(outcome1)
        >>> buffer.add(outcome2)
        >>> batch = buffer.get_batch(max_items=50)
        >>> len(batch)
        2
    """
    
    # Configuration constants
    DEFAULT_MAX_SIZE = 1000
    LOG_INTERVAL = 100  # Log statistics every N additions
    
    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        """
        Initialize outcome buffer.
        
        Args:
            max_size: Maximum number of outcomes to store.
                     When exceeded, oldest outcomes are automatically evicted.
                     Default is 1000 which balances memory usage with
                     sufficient history for gap analysis.
        """
        if max_size <= 0:
            logger.warning(f"Invalid max_size {max_size}, using default {self.DEFAULT_MAX_SIZE}")
            max_size = self.DEFAULT_MAX_SIZE
        
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        
        # Statistics tracking
        self._total_added = 0
        self._total_consumed = 0
        self._last_add_time: Optional[float] = None
        self._last_consume_time: Optional[float] = None
        
        logger.debug(f"[OutcomeBuffer] Initialized with max_size={max_size}")
    
    # =============================================================================
    # PRODUCER METHODS (Main Process Side)
    # =============================================================================
    
    def add(self, outcome: QueryOutcome) -> bool:
        """
        Add an outcome to the buffer (called from main process).
        
        Thread-safe method that appends an outcome to the buffer.
        Oldest outcomes are automatically evicted when buffer is full.
        
        EXAMINE: Validates outcome is not None
        SELECT: Appends to deque (O(1) operation)
        APPLY: Updates statistics
        REMEMBER: Logs periodically for monitoring
        
        Args:
            outcome: QueryOutcome to add to the buffer
            
        Returns:
            True if successfully added, False if outcome was None
        """
        if outcome is None:
            logger.warning("[OutcomeBuffer] Attempted to add None outcome")
            return False
        
        with self._lock:
            try:
                self._buffer.append(outcome)
                self._total_added += 1
                self._last_add_time = time.time()
                
                # Log periodically for monitoring
                if self._total_added % self.LOG_INTERVAL == 0:
                    logger.info(
                        f"[OutcomeBuffer] Total outcomes recorded: {self._total_added}, "
                        f"buffer_size={len(self._buffer)}"
                    )
                
                return True
                
            except Exception as e:
                logger.error(f"[OutcomeBuffer] Error adding outcome: {e}")
                return False
    
    def add_batch(self, outcomes: List[QueryOutcome]) -> int:
        """
        Add multiple outcomes to the buffer atomically.
        
        More efficient than calling add() multiple times as it
        acquires the lock only once.
        
        Args:
            outcomes: List of QueryOutcome objects to add
            
        Returns:
            Number of outcomes successfully added
        """
        if not outcomes:
            return 0
        
        with self._lock:
            added_count = 0
            for outcome in outcomes:
                if outcome is not None:
                    try:
                        self._buffer.append(outcome)
                        self._total_added += 1
                        added_count += 1
                    except Exception as e:
                        logger.warning(f"[OutcomeBuffer] Error in batch add: {e}")
            
            self._last_add_time = time.time()
            
            if added_count > 0:
                logger.debug(f"[OutcomeBuffer] Batch added {added_count} outcomes")
            
            return added_count
    
    # =============================================================================
    # CONSUMER METHODS (Curiosity Engine Side)
    # =============================================================================
    
    def get_batch(self, max_items: int = 50) -> List[QueryOutcome]:
        """
        Get a batch of outcomes for processing (called from curiosity engine).
        
        Removes and returns up to max_items outcomes from the buffer.
        This is the consumer side of the producer-consumer pattern.
        
        EXAMINE: Checks buffer has items
        SELECT: Retrieves up to max_items (oldest first - FIFO)
        APPLY: Removes items from buffer atomically
        REMEMBER: Updates consumption statistics
        
        Args:
            max_items: Maximum number of outcomes to return.
                      Use smaller batches for more responsive processing,
                      larger batches for efficiency.
            
        Returns:
            List of QueryOutcome objects (may be empty if buffer is empty)
        """
        if max_items <= 0:
            return []
        
        with self._lock:
            batch: List[QueryOutcome] = []
            
            try:
                while self._buffer and len(batch) < max_items:
                    batch.append(self._buffer.popleft())
                
                self._total_consumed += len(batch)
                
                if batch:
                    self._last_consume_time = time.time()
                    logger.debug(
                        f"[OutcomeBuffer] Consumed {len(batch)} outcomes, "
                        f"remaining={len(self._buffer)}"
                    )
                
            except Exception as e:
                logger.error(f"[OutcomeBuffer] Error getting batch: {e}")
            
            return batch
    
    def peek_recent(self, n: int = 10) -> List[QueryOutcome]:
        """
        Peek at recent outcomes without removing them.
        
        Useful for monitoring and debugging without affecting the buffer.
        Returns copies to prevent external modification.
        
        Args:
            n: Number of recent outcomes to return
            
        Returns:
            List of most recent QueryOutcome objects (newest last)
        """
        if n <= 0:
            return []
        
        with self._lock:
            try:
                items = list(self._buffer)
                return items[-n:] if len(items) >= n else items.copy()
            except Exception as e:
                logger.error(f"[OutcomeBuffer] Error peeking: {e}")
                return []
    
    def peek_oldest(self, n: int = 10) -> List[QueryOutcome]:
        """
        Peek at oldest outcomes without removing them.
        
        Useful for examining what will be consumed next.
        
        Args:
            n: Number of oldest outcomes to return
            
        Returns:
            List of oldest QueryOutcome objects (oldest first)
        """
        if n <= 0:
            return []
        
        with self._lock:
            try:
                items = list(self._buffer)
                return items[:n] if len(items) >= n else items.copy()
            except Exception as e:
                logger.error(f"[OutcomeBuffer] Error peeking oldest: {e}")
                return []
    
    # =============================================================================
    # STATISTICS AND MONITORING
    # =============================================================================
    
    def size(self) -> int:
        """Get current number of outcomes in buffer."""
        with self._lock:
            return len(self._buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return len(self._buffer) >= self.max_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive buffer statistics for monitoring.
        
        Returns:
            Dictionary with:
            - buffer_size: Current number of items
            - max_size: Maximum capacity
            - total_added: Total items ever added
            - total_consumed: Total items ever consumed
            - pending: Current items waiting to be consumed
            - utilization: Buffer fill percentage (0.0 to 1.0)
            - last_add_time: Timestamp of last add (or None)
            - last_consume_time: Timestamp of last consume (or None)
            - throughput_estimate: Items per second (if sufficient data)
        """
        with self._lock:
            current_size = len(self._buffer)
            utilization = current_size / self.max_size if self.max_size > 0 else 0.0
            
            # Calculate throughput estimate
            throughput = None
            if self._last_add_time and self._last_consume_time and self._total_consumed > 0:
                time_span = self._last_consume_time - (self._last_add_time - 60)  # Approx last minute
                if time_span > 0:
                    throughput = self._total_consumed / max(1, time_span)
            
            return {
                "buffer_size": current_size,
                "max_size": self.max_size,
                "total_added": self._total_added,
                "total_consumed": self._total_consumed,
                "pending": current_size,
                "utilization": utilization,
                "last_add_time": self._last_add_time,
                "last_consume_time": self._last_consume_time,
                "throughput_estimate": throughput,
            }
    
    def get_status_summary(self) -> str:
        """
        Get a human-readable status summary.
        
        Returns:
            String suitable for logging or display
        """
        stats = self.get_statistics()
        return (
            f"OutcomeBuffer: {stats['buffer_size']}/{stats['max_size']} items "
            f"({stats['utilization']:.1%} full), "
            f"added={stats['total_added']}, consumed={stats['total_consumed']}"
        )
    
    # =============================================================================
    # MANAGEMENT METHODS
    # =============================================================================
    
    def clear(self) -> int:
        """
        Clear all outcomes from buffer.
        
        Use with caution as this discards unprocessed data.
        
        Returns:
            Number of outcomes that were cleared
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            logger.info(f"[OutcomeBuffer] Cleared {count} outcomes")
            return count
    
    def reset_statistics(self) -> None:
        """
        Reset statistics counters without clearing the buffer.
        
        Useful for starting fresh measurement periods.
        """
        with self._lock:
            self._total_added = len(self._buffer)  # Keep current count accurate
            self._total_consumed = 0
            self._last_add_time = None
            self._last_consume_time = None
            logger.info("[OutcomeBuffer] Statistics reset")


# =============================================================================
# GLOBAL SINGLETON MANAGEMENT
# =============================================================================

# Global singleton instance (shared across modules)
_outcome_buffer: Optional[OutcomeBuffer] = None
_buffer_lock = threading.Lock()


def get_outcome_buffer(max_size: int = OutcomeBuffer.DEFAULT_MAX_SIZE) -> OutcomeBuffer:
    """
    Get or create the global outcome buffer.
    
    Thread-safe singleton access to the shared outcome buffer.
    The buffer is created on first access and reused thereafter.
    
    Args:
        max_size: Maximum buffer size (only used on first creation)
        
    Returns:
        The global OutcomeBuffer instance
    """
    global _outcome_buffer
    
    with _buffer_lock:
        if _outcome_buffer is None:
            _outcome_buffer = OutcomeBuffer(max_size=max_size)
            logger.info(f"[OutcomeBuffer] Initialized global outcome buffer (max_size={max_size})")
        return _outcome_buffer


def reset_outcome_buffer(max_size: int = OutcomeBuffer.DEFAULT_MAX_SIZE) -> OutcomeBuffer:
    """
    Reset the global outcome buffer (for testing).
    
    Creates a new buffer instance, discarding any existing data.
    
    Args:
        max_size: Maximum size for new buffer
        
    Returns:
        The new OutcomeBuffer instance
    """
    global _outcome_buffer
    
    with _buffer_lock:
        if _outcome_buffer is not None:
            old_stats = _outcome_buffer.get_statistics()
            logger.info(f"[OutcomeBuffer] Resetting buffer (was: {old_stats})")
        
        _outcome_buffer = OutcomeBuffer(max_size=max_size)
        return _outcome_buffer


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def record_outcome(outcome: QueryOutcome) -> bool:
    """
    Convenience function to record an outcome.
    
    This is the primary interface for the main request handler to
    feed query outcomes to the curiosity engine.
    
    Args:
        outcome: QueryOutcome to record
        
    Returns:
        True if successfully recorded, False otherwise
        
    Example:
        outcome = QueryOutcome(
            query_id="q_abc123",
            query_type="reasoning",
            status=OutcomeStatus.SUCCESS,
            execution_time_ms=4500.0,
        )
        outcome.compute_features()
        record_outcome(outcome)
    """
    return get_outcome_buffer().add(outcome)


def get_pending_outcome_count() -> int:
    """Get the number of pending (unconsumed) outcomes."""
    return get_outcome_buffer().size()


def get_outcome_statistics() -> Dict[str, Any]:
    """Get statistics about the outcome buffer."""
    return get_outcome_buffer().get_statistics()


def consume_outcomes(max_items: int = 50) -> List[QueryOutcome]:
    """
    Convenience function to consume outcomes from the buffer.
    
    Intended for use by the curiosity engine's learning loop.
    
    Args:
        max_items: Maximum number of outcomes to retrieve
        
    Returns:
        List of QueryOutcome objects
    """
    return get_outcome_buffer().get_batch(max_items=max_items)
