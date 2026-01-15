"""
Memory pressure monitoring and automatic GC trigger.

This module provides a background thread that monitors memory usage
and triggers garbage collection when memory pressure exceeds a threshold.

This helps prevent the progressive memory degradation seen in production logs,
where query routing times increased from 469ms to 152,048ms due to:
- Repeated SentenceTransformer model loading (~300MB per instance)
- Memory accumulation without garbage collection
- Python's lazy GC not keeping up with allocation rate

DEATH SPIRAL PREVENTION (Forensic Audit Fix):
When memory stays stuck above the critical threshold, repeated aggressive GC
calls can freeze the process for 1-5 seconds each time, creating a feedback loop
where requests pile up during freezes, consuming more memory. This module
implements exponential backoff for consecutive critical events:
- After each critical event, the check interval increases by 1x (up to 5x max)
- This gives the system time to process requests and memory to recover naturally
- The counter resets when memory drops below critical threshold

CALLBACK SAFETY (Forensic Audit Fix):
The aggressive_gc_callback is wrapped in try/except to prevent callback exceptions
from killing the monitoring thread silently.
"""

import atexit
import gc
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger('vulcan.monitoring.memory_guard')

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logger.warning("[MemoryGuard] psutil not available - memory monitoring disabled")


class MemoryGuard:
    """
    Background thread that monitors memory usage and triggers GC.
    
    Memory Guard Logic with Graduated Thresholds:
    1. Every `check_interval` seconds, check system memory usage
    2. Warning threshold (70%): Log warning only
    3. GC threshold (75%): Trigger garbage collection
    4. Critical threshold (85%): Full GC + aggressive cleanup callback
    
    Death Spiral Prevention:
    When memory stays above critical threshold, consecutive GC triggers
    use exponential backoff (up to max_backoff_multiplier) to prevent
    CPU thrashing from repeated aggressive GC operations.
    
    This prevents progressive degradation where:
    - Query routing degrades from 469ms to 152,048ms
    - Embedding batch times increase from 0.15s to 16s+
    - Memory usage grows unbounded
    
    Attributes:
        max_backoff_multiplier: Maximum multiplier for sleep interval (default: 5)
        _consecutive_criticals: Counter for consecutive critical events
    
    Args:
        warning_threshold: Memory percentage for warnings (default: 70.0%)
        gc_threshold: Memory percentage that triggers GC (default: 75.0%)
        critical_threshold: Memory percentage for aggressive cleanup (default: 85.0%)
        check_interval: Seconds between checks (default: 5.0)
        aggressive_gc_callback: Optional callback for critical memory pressure
        threshold_percent: Deprecated - use gc_threshold instead
    """
    
    def __init__(
        self,
        warning_threshold: float = 70.0,
        gc_threshold: float = 75.0,
        critical_threshold: float = 85.0,
        check_interval: float = 5.0,
        aggressive_gc_callback: Optional[Callable[[], None]] = None,
        threshold_percent: Optional[float] = None  # Backward compatibility
    ):
        # Backward compatibility: threshold_percent maps to gc_threshold
        if threshold_percent is not None:
            gc_threshold = threshold_percent
            logger.debug(
                f"[MemoryGuard] threshold_percent is deprecated, use gc_threshold instead"
            )
        
        self.warning_threshold = warning_threshold
        self.gc_threshold = gc_threshold
        self.critical_threshold = critical_threshold
        self.interval = check_interval
        self.aggressive_gc_callback = aggressive_gc_callback
        
        # Thread synchronization
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Thread-safe statistics
        self._stats_lock = threading.Lock()
        self._gc_triggers = 0
        self._last_gc_time: Optional[float] = None
        self._peak_memory_percent: float = 0.0
        
        # Death spiral prevention: backoff after consecutive critical events
        self._consecutive_criticals = 0
        self.max_backoff_multiplier = 5  # Cap backoff at 5x normal interval
    
    def _calculate_backoff_multiplier(self) -> int:
        """Calculate current backoff multiplier based on consecutive criticals."""
        return min(self._consecutive_criticals + 1, self.max_backoff_multiplier)
    
    def _increment_gc_trigger(self):
        """Thread-safe GC trigger increment."""
        with self._stats_lock:
            self._gc_triggers += 1
            self._last_gc_time = time.time()
    
    def _update_peak_memory(self, percent: float):
        """Thread-safe peak memory update."""
        with self._stats_lock:
            if percent > self._peak_memory_percent:
                self._peak_memory_percent = percent
    
    @property
    def gc_triggers(self) -> int:
        """Get GC trigger count (thread-safe)."""
        with self._stats_lock:
            return self._gc_triggers
    
    @property
    def last_gc_time(self) -> Optional[float]:
        """Get last GC timestamp (thread-safe)."""
        with self._stats_lock:
            return self._last_gc_time
    
    @property
    def peak_memory_percent(self) -> float:
        """Get peak memory percentage (thread-safe)."""
        with self._stats_lock:
            return self._peak_memory_percent
    
    @property
    def threshold(self) -> float:
        """Backward compatibility property for gc_threshold."""
        return self.gc_threshold
    
    def start(self):
        """Start the memory monitoring thread."""
        if not PSUTIL_AVAILABLE:
            logger.warning("[MemoryGuard] Cannot start - psutil not available")
            return
        
        if self._running:
            logger.warning("[MemoryGuard] Already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=False,  # NOT daemon - must clean up properly
            name="MemoryGuard"
        )
        self._thread.start()
        
        # Register cleanup on process exit
        atexit.register(self.stop)
        
        logger.info(
            f"[MemoryGuard] Started (warning={self.warning_threshold}%, "
            f"gc={self.gc_threshold}%, critical={self.critical_threshold}%)"
        )
    
    def stop(self):
        """Stop the memory monitoring thread."""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        
        logger.info(
            f"[MemoryGuard] Stopped (triggered GC {self.gc_triggers} times, "
            f"peak memory: {self.peak_memory_percent:.1f}%)"
        )
    
    def _monitor_loop(self):
        """Background monitoring loop with graduated response and death spiral prevention."""
        while self._running and not self._shutdown_event.is_set():
            # Calculate sleep interval with backoff for consecutive critical events
            backoff_multiplier = self._calculate_backoff_multiplier()
            sleep_interval = self.interval * backoff_multiplier
            
            try:
                memory = psutil.virtual_memory()
                percent = memory.percent
                
                self._update_peak_memory(percent)
                
                if percent > self.critical_threshold:
                    # CRITICAL: Aggressive cleanup
                    self._consecutive_criticals += 1
                    logger.critical(
                        f"[MemoryGuard] CRITICAL memory: {percent:.1f}% "
                        f"(available: {memory.available / (1024**3):.1f}GB) - aggressive cleanup "
                        f"(consecutive criticals: {self._consecutive_criticals}, "
                        f"backoff: {backoff_multiplier}x)"
                    )
                    gc.collect(generation=2)  # Full collection
                    self._increment_gc_trigger()
                    
                    if self.aggressive_gc_callback:
                        try:
                            self.aggressive_gc_callback()
                        except Exception as e:
                            logger.error(f"[MemoryGuard] Aggressive GC callback failed: {e}")
                    
                    # Check memory after cleanup
                    memory_after = psutil.virtual_memory()
                    freed_mb = max(0, (memory.used - memory_after.used) / (1024**2))
                    logger.info(
                        f"[MemoryGuard] Aggressive cleanup completed, "
                        f"freed ~{freed_mb:.1f}MB, memory now: {memory_after.percent:.1f}%"
                    )
                
                elif percent > self.gc_threshold:
                    # ACTION: Trigger GC - reset consecutive criticals
                    self._consecutive_criticals = 0
                    logger.warning(
                        f"[MemoryGuard] High memory usage: {percent:.1f}% "
                        f"(available: {memory.available / (1024**3):.1f}GB) - triggering GC"
                    )
                    collected = gc.collect()
                    self._increment_gc_trigger()
                    
                    # Check memory after GC
                    memory_after = psutil.virtual_memory()
                    freed_mb = max(0, (memory.used - memory_after.used) / (1024**2))
                    logger.info(
                        f"[MemoryGuard] GC collected {collected} objects, "
                        f"freed ~{freed_mb:.1f}MB, memory now: {memory_after.percent:.1f}%"
                    )
                
                elif percent > self.warning_threshold:
                    # WARNING: Just log - reset consecutive criticals
                    self._consecutive_criticals = 0
                    logger.warning(
                        f"[MemoryGuard] Memory warning: {percent:.1f}% "
                        f"(available: {memory.available / (1024**3):.1f}GB)"
                    )
                
                else:
                    # NORMAL: Reset consecutive criticals
                    self._consecutive_criticals = 0
                
            except Exception as e:
                logger.error(f"[MemoryGuard] Monitor error: {e}")
            
            # Interruptible sleep with backoff
            self._shutdown_event.wait(timeout=sleep_interval)
    
    def get_status(self) -> dict:
        """Get current memory guard status."""
        status = {
            "running": self._running,
            "warning_threshold": self.warning_threshold,
            "gc_threshold": self.gc_threshold,
            "critical_threshold": self.critical_threshold,
            "check_interval": self.interval,
            "gc_triggers": self.gc_triggers,
            "last_gc_time": self.last_gc_time,
            "peak_memory_percent": self.peak_memory_percent,
            "has_aggressive_callback": self.aggressive_gc_callback is not None,
            "consecutive_criticals": self._consecutive_criticals,
            "current_backoff_multiplier": self._calculate_backoff_multiplier(),
            "max_backoff_multiplier": self.max_backoff_multiplier,
        }
        
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                status["current_memory_percent"] = memory.percent
                status["available_gb"] = memory.available / (1024**3)
                
                # Determine current action level
                if memory.percent > self.critical_threshold:
                    status["action_level"] = "critical"
                elif memory.percent > self.gc_threshold:
                    status["action_level"] = "gc"
                elif memory.percent > self.warning_threshold:
                    status["action_level"] = "warning"
                else:
                    status["action_level"] = "normal"
            except Exception:
                pass
        
        return status
    
    def force_gc(self) -> int:
        """
        Force a garbage collection and return count of collected objects.
        
        This can be called manually after heavy operations like
        model loading or after processing complex queries.
        """
        collected = gc.collect()
        self._increment_gc_trigger()
        logger.info(f"[MemoryGuard] Manual GC collected {collected} objects")
        return collected


# Global instance and lock
_guard: Optional[MemoryGuard] = None
_guard_lock = threading.Lock()


def start_memory_guard(
    threshold_percent: Optional[float] = None,  # Deprecated
    check_interval: float = 5.0,
    warning_threshold: float = 70.0,
    gc_threshold: Optional[float] = None,
    critical_threshold: float = 85.0,
    aggressive_gc_callback: Optional[Callable[[], None]] = None
) -> Optional[MemoryGuard]:
    """
    Start the global memory guard instance with double-checked locking.
    
    Args:
        warning_threshold: Memory percentage for warnings (default: 70.0%)
        gc_threshold: Memory percentage that triggers GC (default: 75.0%)
        critical_threshold: Memory percentage for aggressive cleanup (default: 85.0%)
        check_interval: Seconds between checks (default: 5.0)
        aggressive_gc_callback: Optional callback for critical memory pressure
        threshold_percent: Deprecated - use gc_threshold instead
        
    Returns:
        MemoryGuard instance or None if psutil not available
    """
    global _guard
    
    if not PSUTIL_AVAILABLE:
        logger.warning("[MemoryGuard] Cannot start - psutil not available")
        return None
    
    # Backward compatibility: threshold_percent maps to gc_threshold
    if gc_threshold is None:
        gc_threshold = threshold_percent if threshold_percent is not None else 75.0
    
    # Fast path: already started
    if _guard is not None and _guard._running:
        logger.debug("[MemoryGuard] Already running")
        return _guard
    
    # Slow path: need to create/start
    with _guard_lock:
        # Double-check after acquiring lock
        if _guard is not None and _guard._running:
            return _guard
        
        if _guard is None:
            _guard = MemoryGuard(
                warning_threshold=warning_threshold,
                gc_threshold=gc_threshold,
                critical_threshold=critical_threshold,
                check_interval=check_interval,
                aggressive_gc_callback=aggressive_gc_callback
            )
        
        _guard.start()
        return _guard


def stop_memory_guard():
    """Stop the global memory guard instance."""
    global _guard
    
    if _guard:
        _guard.stop()
        _guard = None


def get_memory_guard() -> Optional[MemoryGuard]:
    """Get the global memory guard instance (if started)."""
    return _guard


def trigger_gc() -> int:
    """
    Trigger garbage collection via the guard or directly.
    
    Returns:
        Number of objects collected
    """
    if _guard:
        return _guard.force_gc()
    else:
        collected = gc.collect()
        logger.debug(f"[MemoryGuard] Direct GC collected {collected} objects")
        return collected


def set_aggressive_gc_callback(callback: Callable[[], None]) -> None:
    """
    Set callback for aggressive cleanup during critical memory pressure.
    
    Args:
        callback: Function to call when memory exceeds critical threshold
    """
    global _guard
    if _guard:
        _guard.aggressive_gc_callback = callback
        logger.info("[MemoryGuard] Aggressive GC callback registered")
    else:
        logger.warning("[MemoryGuard] Cannot set callback - guard not started")
