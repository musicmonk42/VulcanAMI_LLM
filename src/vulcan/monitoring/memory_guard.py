"""
Memory pressure monitoring and automatic GC trigger.

This module provides a background thread that monitors memory usage
and triggers garbage collection when memory pressure exceeds a threshold.

This helps prevent the progressive memory degradation seen in production logs,
where query routing times increased from 469ms to 152,048ms due to:
- Repeated SentenceTransformer model loading (~300MB per instance)
- Memory accumulation without garbage collection
- Python's lazy GC not keeping up with allocation rate
"""

import gc
import logging
import threading
import time
from typing import Optional

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
    
    Memory Guard Logic:
    1. Every `check_interval` seconds, check system memory usage
    2. If memory usage exceeds `threshold_percent`, trigger gc.collect()
    3. Log warnings when high memory is detected
    
    This prevents progressive degradation where:
    - Query routing degrades from 469ms to 152,048ms
    - Embedding batch times increase from 0.15s to 16s+
    - Memory usage grows unbounded
    
    Args:
        threshold_percent: Memory usage percentage that triggers GC (default: 85%)
        check_interval: Seconds between checks (default: 5.0)
    """
    
    def __init__(
        self,
        threshold_percent: float = 85.0,
        check_interval: float = 5.0
    ):
        self.threshold = threshold_percent
        self.interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.gc_triggers = 0
        self.last_gc_time: Optional[float] = None
        self.peak_memory_percent: float = 0.0
    
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
            daemon=True,
            name="MemoryGuard"
        )
        self._thread.start()
        logger.info(f"[MemoryGuard] Started (threshold={self.threshold}%)")
    
    def stop(self):
        """Stop the memory monitoring thread."""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        logger.info(
            f"[MemoryGuard] Stopped (triggered GC {self.gc_triggers} times, "
            f"peak memory: {self.peak_memory_percent:.1f}%)"
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                memory = psutil.virtual_memory()
                
                # Track peak
                if memory.percent > self.peak_memory_percent:
                    self.peak_memory_percent = memory.percent
                
                # Check threshold
                if memory.percent > self.threshold:
                    logger.warning(
                        f"[MemoryGuard] High memory usage: {memory.percent:.1f}% "
                        f"(available: {memory.available / (1024**3):.1f}GB) - triggering GC"
                    )
                    collected = gc.collect()
                    self.gc_triggers += 1
                    self.last_gc_time = time.time()
                    
                    # Check memory after GC
                    memory_after = psutil.virtual_memory()
                    freed_mb = max(0, (memory.used - memory_after.used) / (1024**2))
                    logger.info(
                        f"[MemoryGuard] GC collected {collected} objects, "
                        f"freed ~{freed_mb:.1f}MB, "
                        f"memory now: {memory_after.percent:.1f}%"
                    )
                
            except Exception as e:
                logger.error(f"[MemoryGuard] Monitor error: {e}")
            
            # Interruptible sleep
            self._shutdown_event.wait(timeout=self.interval)
    
    def get_status(self) -> dict:
        """Get current memory guard status."""
        status = {
            "running": self._running,
            "threshold_percent": self.threshold,
            "check_interval": self.interval,
            "gc_triggers": self.gc_triggers,
            "last_gc_time": self.last_gc_time,
            "peak_memory_percent": self.peak_memory_percent,
        }
        
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                status["current_memory_percent"] = memory.percent
                status["available_gb"] = memory.available / (1024**3)
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
        self.gc_triggers += 1
        self.last_gc_time = time.time()
        logger.info(f"[MemoryGuard] Manual GC collected {collected} objects")
        return collected


# Global instance
_guard: Optional[MemoryGuard] = None


def start_memory_guard(
    threshold_percent: float = 85.0,
    check_interval: float = 5.0
) -> Optional[MemoryGuard]:
    """
    Start the global memory guard instance.
    
    Args:
        threshold_percent: Memory threshold for GC trigger (default: 85%)
        check_interval: Seconds between checks (default: 5.0)
        
    Returns:
        MemoryGuard instance or None if psutil not available
    """
    global _guard
    
    if not PSUTIL_AVAILABLE:
        logger.warning("[MemoryGuard] Cannot start - psutil not available")
        return None
    
    if _guard is None:
        _guard = MemoryGuard(
            threshold_percent=threshold_percent,
            check_interval=check_interval
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
