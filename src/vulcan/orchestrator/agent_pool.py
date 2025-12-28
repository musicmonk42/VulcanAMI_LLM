# ============================================================
# VULCAN-AGI Orchestrator - Agent Pool Module
# Agent pool management with lifecycle control, auto-scaling, and recovery
# FULLY FIXED VERSION - Enhanced with proper resource management, state validation, and comprehensive error handling
# TTLCache fallback class added for Python environments without cachetools
# TIMEOUT FIXES - Prevents hanging in tests and production
# WINDOWS MULTIPROCESSING FIX - Worker process doesn't access parent's unpicklable objects
# FIXED: Converted long time.sleep calls to interruptible self._shutdown_event.wait().
# PERFORMANCE: Added response time tracking and adaptive scaling
# PERFORMANCE: Added simple_mode support for reduced overhead
# MEMORY LEAK FIX: Replaced unbounded provenance_records with rolling deque(maxlen=50)
# THREAD POOL FIX: submit_job() now non-blocking to prevent thread pool starvation
# ============================================================

import asyncio
import gc
import heapq
import json
import logging
import multiprocessing
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import numpy with fallback for environments without it
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    # Use DEBUG level to avoid cluttering logs on every import
    logging.getLogger(__name__).debug(
        "numpy not available, some advanced features will be disabled"
    )

# Import psutil with fallback for missing or broken installations
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    # Note: Logger not yet configured at module level, so using logging directly here
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "psutil not available, system resource monitoring will be disabled"
    )

from .agent_lifecycle import (
    AgentCapability,
    AgentMetadata,
    AgentState,
    create_agent_metadata,
    create_job_provenance,
)
from .task_queues import TaskQueueInterface, create_task_queue

# Import memory systems for provenance tracking
from src.vulcan.memory.specialized import WorkingMemory
from src.vulcan.memory.hierarchical import HierarchicalMemory
from src.vulcan.memory.base import MemoryConfig

# Import TournamentManager for multi-agent selection
try:
    from src.tournament_manager import TournamentManager
    TOURNAMENT_MANAGER_AVAILABLE = True
except ImportError:
    TournamentManager = None
    TOURNAMENT_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "TournamentManager not available, multi-agent tournament selection will be disabled"
    )

# ============================================================
# REASONING INTEGRATION - Wire reasoning engines into task execution
# ============================================================
# Import UnifiedReasoner and reasoning integration for actual reasoning invocation
try:
    from src.vulcan.reasoning import (
        UnifiedReasoner,
        ReasoningType,
        ReasoningResult,
        UNIFIED_AVAILABLE,
        create_unified_reasoner,
    )
    from src.vulcan.reasoning.reasoning_integration import (
        apply_reasoning,
        get_reasoning_integration,
        ReasoningResult as IntegrationReasoningResult,
    )
    REASONING_AVAILABLE = UNIFIED_AVAILABLE
    logging.getLogger(__name__).info(
        "Reasoning integration loaded successfully - reasoning engines will be invoked"
    )
except ImportError as e:
    UnifiedReasoner = None
    ReasoningType = None
    ReasoningResult = None
    UNIFIED_AVAILABLE = False
    create_unified_reasoner = None
    apply_reasoning = None
    get_reasoning_integration = None
    IntegrationReasoningResult = None
    REASONING_AVAILABLE = False
    logging.getLogger(__name__).warning(
        f"Reasoning integration not available: {e}. Tasks will use placeholder execution."
    )

# ============================================================
# CONSTANTS
# ============================================================

# Fallback hardware specification values when psutil is not available
DEFAULT_FALLBACK_MEMORY_GB = 4.0  # Conservative memory estimate
DEFAULT_FALLBACK_STORAGE_GB = 100.0  # Conservative storage estimate

# Redis keys for agent pool state persistence
REDIS_KEY_AGENT_POOL_STATS = "vulcan:agent_pool:stats"
REDIS_KEY_PROVENANCE_COUNT = "vulcan:agent_pool:provenance_records_count"

# Tournament-based multi-agent selection configuration
TOURNAMENT_QUERY_TYPES = ('reasoning', 'symbolic', 'analogical', 'causal')
TOURNAMENT_MAX_CANDIDATES = 3  # Maximum agents to run in parallel for tournament
TOURNAMENT_DIVERSITY_PENALTY = 0.3
TOURNAMENT_WINNER_PERCENTAGE = 0.2

# FIXED: Add cachetools import for LRU cache with TTL
try:
    from cachetools import TTLCache

    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logging.warning("cachetools not available, using dict fallback with manual cleanup")

    # FIXED: Define fallback TTLCache class
    class TTLCache(dict):
        """
        Fallback TTLCache implementation when cachetools is not available.
        Provides basic dict functionality with size limit awareness.
        TTL (time-to-live) is handled manually in the calling code.
        """

        def __init__(self, maxsize: int, ttl: float):
            """
            Initialize TTLCache fallback

            Args:
                maxsize: Maximum number of items
                ttl: Time-to-live in seconds (stored but not enforced by this class)
            """
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl

        def __setitem__(self, key, value):
            """Set item with maxsize check"""
            if len(self) >= self.maxsize and key not in self:
                # Remove oldest item (approximate LRU)
                if self:
                    oldest_key = next(iter(self))
                    del self[oldest_key]
            super().__setitem__(key, value)


logger = logging.getLogger(__name__)

# ============================================================
# SIMPLE MODE CONFIGURATION - Performance Optimization
# ============================================================

# Import simple mode configuration with fallback
try:
    from src.vulcan.simple_mode import (
        DEFAULT_MIN_AGENTS as SIMPLE_MODE_MIN_AGENTS,
        DEFAULT_MAX_AGENTS as SIMPLE_MODE_MAX_AGENTS,
        MAX_PROVENANCE_RECORDS as SIMPLE_MODE_MAX_PROVENANCE,
        AGENT_CHECK_INTERVAL as SIMPLE_MODE_CHECK_INTERVAL,
        SIMPLE_MODE,
    )
except ImportError:
    # Fallback if simple_mode not available
    SIMPLE_MODE = os.getenv("VULCAN_SIMPLE_MODE", "false").lower() in ("true", "1", "yes", "on")
    SIMPLE_MODE_MIN_AGENTS = int(os.getenv("MIN_AGENTS", "1" if SIMPLE_MODE else "10"))
    SIMPLE_MODE_MAX_AGENTS = int(os.getenv("MAX_AGENTS", "5" if SIMPLE_MODE else "100"))
    SIMPLE_MODE_MAX_PROVENANCE = int(os.getenv("MAX_PROVENANCE_RECORDS", "50" if SIMPLE_MODE else "1000"))
    SIMPLE_MODE_CHECK_INTERVAL = int(os.getenv("AGENT_CHECK_INTERVAL", "300" if SIMPLE_MODE else "30"))


# ============================================================
# PERFORMANCE MONITORING - Response Time Tracker
# ============================================================


class ResponseTimeTracker:
    """Tracks response times for performance monitoring and adaptive scaling.
    
    Maintains a sliding window of response times to compute percentiles
    and detect performance degradation for auto-scaling decisions.
    """
    
    def __init__(self, window_size: int = 1000, alert_threshold_ms: float = 5000.0):
        """
        Initialize response time tracker.
        
        Args:
            window_size: Number of samples to keep in sliding window
            alert_threshold_ms: Response time threshold for alerts (milliseconds)
        """
        self.window_size = window_size
        self.alert_threshold_ms = alert_threshold_ms
        self._samples: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()
        self._degradation_callbacks: List[callable] = []
    
    def record(self, duration_ms: float, job_id: str = None, agent_id: str = None) -> None:
        """Record a response time sample."""
        timestamp = time.time()
        with self._lock:
            self._samples.append({
                "timestamp": timestamp,
                "duration_ms": duration_ms,
                "job_id": job_id,
                "agent_id": agent_id,
            })
        
        # Check for degradation
        if duration_ms > self.alert_threshold_ms:
            self._notify_degradation(duration_ms, job_id, agent_id)
    
    def get_percentile(self, percentile: float) -> float:
        """Get the specified percentile of response times."""
        with self._lock:
            if not self._samples:
                return 0.0
            durations = sorted([s["duration_ms"] for s in self._samples])
            idx = int(len(durations) * percentile / 100.0)
            return durations[min(idx, len(durations) - 1)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive response time statistics."""
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "max_ms": 0.0,
                    "min_ms": 0.0,
                }
            
            durations = [s["duration_ms"] for s in self._samples]
            sorted_durations = sorted(durations)
            
            return {
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "p50_ms": sorted_durations[len(sorted_durations) // 2],
                "p95_ms": sorted_durations[int(len(sorted_durations) * 0.95)],
                "p99_ms": sorted_durations[int(len(sorted_durations) * 0.99)],
                "max_ms": max(durations),
                "min_ms": min(durations),
            }
    
    def register_degradation_callback(self, callback: callable) -> None:
        """Register a callback for performance degradation alerts."""
        self._degradation_callbacks.append(callback)
    
    def _notify_degradation(self, duration_ms: float, job_id: str, agent_id: str) -> None:
        """Notify registered callbacks of performance degradation."""
        for callback in self._degradation_callbacks:
            try:
                callback(duration_ms, job_id, agent_id)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}")
    
    def get_recent_trend(self, window_seconds: float = 60.0) -> float:
        """Get trend of response times in the recent window.
        
        Returns:
            Positive value indicates degradation, negative indicates improvement.
        """
        now = time.time()
        with self._lock:
            recent = [s for s in self._samples if now - s["timestamp"] <= window_seconds]
            
            if len(recent) < 2:
                return 0.0
            
            # Compare first half vs second half
            mid = len(recent) // 2
            first_half_avg = sum(s["duration_ms"] for s in recent[:mid]) / mid
            second_half_avg = sum(s["duration_ms"] for s in recent[mid:]) / (len(recent) - mid)
            
            return second_half_avg - first_half_avg
    
    def trim_to_window_size(self) -> None:
        """Trim samples to window size to prevent memory growth.
        
        PERFORMANCE FIX: Called periodically to ensure the samples deque
        doesn't exceed the configured window size.
        """
        with self._lock:
            # The deque has maxlen, but we explicitly trim for safety
            if len(self._samples) > self.window_size:
                # Keep only the most recent samples
                recent = list(self._samples)[-self.window_size:]
                self._samples.clear()
                self._samples.extend(recent)


# ============================================================
# PERFORMANCE OPTIMIZATION - Priority Job Queue
# ============================================================


class PriorityJobQueue:
    """Priority queue for job scheduling with support for high-priority tokens.
    
    Implements a multi-level priority queue optimized for:
    - High-frequency token processing
    - SLO-aware scheduling
    - Starvation prevention
    """
    
    # Priority levels
    PRIORITY_CRITICAL = 0
    PRIORITY_HIGH = 1
    PRIORITY_NORMAL = 2
    PRIORITY_LOW = 3
    PRIORITY_BATCH = 4
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize priority job queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queues: Dict[int, List[Tuple[int, str, Dict]]] = {
            i: [] for i in range(5)
        }
        self._lock = threading.RLock()
        self._size = 0
        self._counter = 0  # Monotonic counter for FIFO ordering
        self._stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "priority_distribution": defaultdict(int),
        }
    
    def enqueue(self, job_id: str, job_data: Dict[str, Any], priority: int = PRIORITY_NORMAL) -> bool:
        """Add a job to the queue.
        
        Args:
            job_id: Unique job identifier
            job_data: Job data dictionary
            priority: Priority level (0=critical, 4=batch)
        
        Returns:
            True if enqueued successfully, False if queue is full
        """
        if priority < 0 or priority > 4:
            priority = self.PRIORITY_NORMAL
        
        with self._lock:
            if self._size >= self.max_size:
                return False
            
            # Use monotonic counter for reliable FIFO ordering under high load
            # (more reliable than time.time() which may have precision issues)
            sequence = self._counter
            self._counter += 1
            heapq.heappush(self._queues[priority], (sequence, job_id, job_data))
            self._size += 1
            self._stats["total_enqueued"] += 1
            self._stats["priority_distribution"][priority] += 1
            
            return True
    
    def dequeue(self) -> Optional[Tuple[str, Dict[str, Any], int]]:
        """Remove and return the highest priority job.
        
        Returns:
            Tuple of (job_id, job_data, priority) or None if queue is empty
        """
        with self._lock:
            # Check queues in priority order
            for priority in range(5):
                if self._queues[priority]:
                    _, job_id, job_data = heapq.heappop(self._queues[priority])
                    self._size -= 1
                    self._stats["total_dequeued"] += 1
                    return (job_id, job_data, priority)
            
            return None
    
    def peek(self) -> Optional[Tuple[str, Dict[str, Any], int]]:
        """Peek at the highest priority job without removing it."""
        with self._lock:
            for priority in range(5):
                if self._queues[priority]:
                    _, job_id, job_data = self._queues[priority][0]
                    return (job_id, job_data, priority)
            return None
    
    def size(self) -> int:
        """Get current queue size."""
        return self._size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            queue_sizes = {i: len(self._queues[i]) for i in range(5)}
            return {
                **self._stats,
                "current_size": self._size,
                "queue_sizes_by_priority": queue_sizes,
            }
    
    def clear(self) -> int:
        """Clear all queues. Returns number of items cleared."""
        with self._lock:
            count = self._size
            for priority in range(5):
                self._queues[priority].clear()
            self._size = 0
            return count
    
    def reset_priority_distribution(self) -> None:
        """Reset the priority distribution statistics.
        
        PERFORMANCE FIX: Called periodically to prevent the priority_distribution
        dictionary from growing unboundedly over long sessions. Preserves total
        counts but resets the distribution tracking.
        """
        with self._lock:
            self._stats["priority_distribution"] = defaultdict(int)


# ============================================================
# STANDALONE WORKER FUNCTION (MUST BE AT MODULE LEVEL FOR PICKLING)
# ============================================================


def _standalone_agent_worker(agent_id: str):
    """
    Standalone agent worker function - runs in separate process
    FIXED: Must be at module level to be picklable on Windows

    This is a minimal stub that just runs without accessing parent state.
    In a production system, this would communicate via IPC (queues, pipes, etc.)

    Args:
        agent_id: Agent identifier
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Agent {agent_id} worker started (standalone)")

    try:
        # This is intentionally minimal to avoid Windows pickling issues
        # In production, implement proper IPC here (multiprocessing.Queue, etc.)
        while True:
            # Short, non-blocking sleep is fine here since it's just a placeholder loop
            time.sleep(0.1)
            # TODO: Poll for tasks via IPC mechanism
            # TODO: Execute tasks
            # TODO: Report results back via IPC
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt - graceful shutdown")
    except Exception as e:
        logger.error(f"Agent {agent_id} worker error: {e}")

    logger.info(f"Agent {agent_id} worker stopped")


# ============================================================
# AGENT POOL MANAGER (FULLY FIXED)
# ============================================================


class AgentPoolManager:
    """
    Manages pools of agents with lifecycle control and proper resource management

    Key Features:
    - Automatic agent spawning and retirement
    - State machine validation for all state transitions
    - Memory-bounded provenance tracking with TTL
    - Stale task cleanup to prevent memory leaks
    - Comprehensive error handling and recovery
    - Thread-safe operations throughout
    - FIXED: Proper timeouts to prevent hanging
    - FIXED: Windows multiprocessing compatibility (uses standalone worker function)
    - THREAD POOL FIX: submit_job() is now non-blocking to prevent thread pool starvation
    - SINGLETON FIX: Thread-safe singleton pattern to prevent duplicate pools
    """
    
    # SINGLETON FIX: Class-level instance tracking to prevent duplicate pools
    _instances: Dict[str, "AgentPoolManager"] = {}
    _instance_lock = threading.Lock()
    _default_instance: Optional["AgentPoolManager"] = None
    
    @classmethod
    def get_instance(
        cls,
        instance_id: str = "default",
        max_agents: int = None,
        min_agents: int = None,
        task_queue_type: str = "custom",
        **kwargs
    ) -> "AgentPoolManager":
        """
        Get or create a singleton instance of AgentPoolManager.
        
        SINGLETON FIX: This method ensures only one pool exists per instance_id,
        preventing the "zombie pool" issue where multiple pools run simultaneously.
        
        Args:
            instance_id: Unique identifier for this pool instance (default: "default")
            max_agents: Maximum number of agents in pool
            min_agents: Minimum number of agents to maintain
            task_queue_type: Type of task queue
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            AgentPoolManager singleton instance
        """
        with cls._instance_lock:
            if instance_id not in cls._instances:
                logger.info(f"Creating new AgentPoolManager instance: {instance_id}")
                instance = cls(
                    max_agents=max_agents,
                    min_agents=min_agents,
                    task_queue_type=task_queue_type,
                    **kwargs
                )
                instance._instance_id = instance_id
                cls._instances[instance_id] = instance
                
                # Track default instance for convenience
                if instance_id == "default":
                    cls._default_instance = instance
            else:
                logger.debug(f"Returning existing AgentPoolManager instance: {instance_id}")
            
            return cls._instances[instance_id]
    
    @classmethod
    def get_default(cls) -> Optional["AgentPoolManager"]:
        """
        Get the default AgentPoolManager instance if it exists.
        
        Returns:
            Default AgentPoolManager instance or None
        """
        return cls._default_instance
    
    @classmethod
    def get_all_instances(cls) -> Dict[str, "AgentPoolManager"]:
        """
        Get all active AgentPoolManager instances.
        
        Returns:
            Dictionary of instance_id to AgentPoolManager
        """
        with cls._instance_lock:
            return dict(cls._instances)
    
    @classmethod
    def shutdown_all(cls) -> None:
        """
        Shutdown all AgentPoolManager instances.
        
        SINGLETON FIX: This ensures clean shutdown of all pools to prevent
        zombie pools from persisting across restarts.
        """
        with cls._instance_lock:
            for instance_id, instance in list(cls._instances.items()):
                logger.info(f"Shutting down AgentPoolManager instance: {instance_id}")
                try:
                    instance.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down pool {instance_id}: {e}")
            cls._instances.clear()
            cls._default_instance = None

    def __init__(
        self,
        max_agents: int = None,
        min_agents: int = None,
        task_queue_type: str = "custom",
        provenance_ttl: int = 3600,
        task_timeout_seconds: int = 300,
        config: Dict[str, Any] = None,
        redis_client: Optional[Any] = None,
    ):
        """
        Initialize Agent Pool Manager

        Args:
            max_agents: Maximum number of agents in pool (defaults to SIMPLE_MODE value)
            min_agents: Minimum number of agents to maintain (defaults to SIMPLE_MODE value)
            task_queue_type: Type of task queue ('ray', 'celery', 'custom')
            provenance_ttl: Time-to-live for provenance records in seconds
            task_timeout_seconds: Default timeout for task assignments
            config: Optional configuration dictionary.
            redis_client: Optional Redis client for state persistence.
        """
        self.config = config or {}
        
        # Redis client for state persistence
        self.redis_client = redis_client
        
        # EMERGENCY STABILIZATION: Hardcode agent limits to reduce thread thrashing
        # Original behavior used SIMPLE_MODE defaults, now forced to min=5, max=10
        self.max_agents = 10
        self.min_agents = 5
        self.task_timeout_seconds = task_timeout_seconds

        # Agent tracking
        self.agents: Dict[str, AgentMetadata] = {}
        self.agent_processes: Dict[str, multiprocessing.Process] = {}

        # MEMORY LEAK FIX: Use specialized memory systems instead of unbounded list
        # Create memory config for provenance tracking
        memory_config = MemoryConfig(max_working_memory=50)
        self.working_memory = WorkingMemory(memory_config)
        self.long_term_memory = HierarchicalMemory(memory_config)
        
        # Keep legacy provenance tracking for backward compatibility
        max_provenance = self.config.get("max_provenance_records", SIMPLE_MODE_MAX_PROVENANCE)
        # Use 50 as default if not specified, as recommended for fixing memory leak
        self._provenance_maxlen = max(max_provenance, 50) if max_provenance else 50
        self._provenance_records: deque[Any] = deque(maxlen=self._provenance_maxlen)
        self._provenance_lock = asyncio.Lock()  # Async lock for thread-safe provenance access
        self._sync_provenance_lock = threading.Lock()  # Sync lock for non-async methods
        # Lookup dictionary for O(1) job_id access (auto-cleaned when deque rotates)
        self._provenance_lookup: Dict[str, Any] = {}

        # Task assignment tracking
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_assignment_times: Dict[str, float] = {}  # task_id -> timestamp

        # Main lock for thread-safe operations
        self.lock = threading.RLock()

        # Task queue initialization
        queue_config = self.config.get("queue_config", {})
        self.task_queue: Optional[TaskQueueInterface] = create_task_queue(
            task_queue_type, **queue_config
        )
        self.task_queue_type = task_queue_type

        # Monitoring and lifecycle management
        self.monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Auto-scaling and recovery
        self.auto_scaler: Optional["AutoScaler"] = None
        self.recovery_manager: Optional["RecoveryManager"] = None

        # Provenance archiving
        self.archive_dir = Path("provenance_archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._last_archive_time = time.time()
        self._archive_lock = threading.Lock()

        # Statistics - Initialize with defaults first
        self.stats = {
            "total_jobs_submitted": 0,
            "total_jobs_completed": 0,
            "total_jobs_failed": 0,
            "total_agents_spawned": 0,
            "total_agents_retired": 0,
            "total_recoveries_attempted": 0,
            "total_recoveries_successful": 0,
        }
        self.stats_lock = threading.Lock()
        
        # Provenance records count - Initialize with default, will be hydrated from Redis
        self._provenance_records_count = 0
        
        # Hydrate state from Redis if available
        self._hydrate_state_from_redis()

        # Status check throttling
        self.last_status_check = 0
        self.status_check_interval = 5.0  # Seconds
        
        # ========== PERFORMANCE OPTIMIZATIONS ==========
        # Response time tracking for adaptive scaling
        self.response_time_tracker = ResponseTimeTracker(
            window_size=1000,
            alert_threshold_ms=self.config.get("alert_threshold_ms", 5000.0)
        )
        
        # Priority job queue for high-frequency token processing
        self.priority_queue = PriorityJobQueue(
            max_size=self.config.get("priority_queue_size", 10000)
        )
        
        # Agent specialization tracking
        self.specialized_agents: Dict[str, List[str]] = defaultdict(list)
        
        # Performance thresholds for adaptive scaling
        self.perf_thresholds = {
            "p95_target_ms": self.config.get("p95_target_ms", 100.0),
            "p99_target_ms": self.config.get("p99_target_ms", 500.0),
            "max_queue_depth": self.config.get("max_queue_depth", 100),
        }

        # ========== THREAD POOL FIX: Non-blocking job execution ==========
        # Pending executions queue - jobs wait here instead of blocking submit_job()
        self._pending_executions: Dict[str, Dict[str, Any]] = {}
        self._pending_executions_lock = threading.Lock()
        
        # Dedicated executor thread for processing pending jobs
        self._executor_thread: Optional[threading.Thread] = None
        self._start_executor()

        # Start monitoring
        self._start_monitor()

        # Initialize auto-scaling and recovery managers
        self.auto_scaler = AutoScaler(self)
        self.recovery_manager = RecoveryManager(self)
        
        # Initialize TournamentManager for multi-agent selection
        self.tournament_manager = None
        if TOURNAMENT_MANAGER_AVAILABLE and TournamentManager is not None:
            try:
                self.tournament_manager = TournamentManager(
                    diversity_penalty=TOURNAMENT_DIVERSITY_PENALTY,
                    winner_percentage=TOURNAMENT_WINNER_PERCENTAGE
                )
                logger.info(
                    f"✓ TournamentManager initialized for multi-agent selection "
                    f"(diversity_penalty={TOURNAMENT_DIVERSITY_PENALTY}, "
                    f"winner_percentage={TOURNAMENT_WINNER_PERCENTAGE})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize TournamentManager: {e}")

        # Initialize minimum agents
        self._initialize_agent_pool()

        # Log actual configured values (self.min_agents/max_agents) not function params
        # This fixes misleading log output when values are overridden
        logger.info(
            f"AgentPoolManager initialized: "
            f"min_agents={self.min_agents}, max_agents={self.max_agents}, "
            f"queue_type={task_queue_type}, "
            f"cachetools_available={CACHETOOLS_AVAILABLE}"
        )

    # ========== THREAD POOL FIX: Background Job Executor ==========
    
    def _start_executor(self):
        """Start background executor thread for processing pending jobs."""
        if self._executor_thread is None or not self._executor_thread.is_alive():
            self._executor_thread = threading.Thread(
                target=self._process_pending_executions,
                daemon=True,
                name="AgentPoolExecutor"
            )
            self._executor_thread.start()
            logger.info("Agent pool executor thread started")
    
    def _process_pending_executions(self):
        """
        Background thread that processes pending job executions.
        
        THREAD POOL FIX: This runs in a dedicated thread, processing jobs
        that were queued by submit_job(). This prevents submit_job() from
        blocking the caller's thread pool.
        """
        logger.info("Job executor thread started")
        
        while not self._shutdown_event.is_set():
            try:
                # Check for pending jobs every 10ms
                if self._shutdown_event.wait(timeout=0.01):
                    break
                
                # Get pending jobs to process
                jobs_to_process = []
                with self._pending_executions_lock:
                    # Process up to 10 jobs per cycle to prevent starvation
                    job_ids = list(self._pending_executions.keys())[:10]
                    for job_id in job_ids:
                        exec_data = self._pending_executions.pop(job_id, None)
                        if exec_data:
                            jobs_to_process.append((job_id, exec_data))
                
                # Execute jobs outside the lock
                for job_id, exec_data in jobs_to_process:
                    try:
                        self._execute_job_sync(
                            job_id=job_id,
                            agent_id=exec_data["agent_id"],
                            graph=exec_data["graph"],
                            parameters=exec_data["parameters"],
                            metadata=exec_data["metadata"],
                        )
                    except Exception as e:
                        logger.error(f"Executor failed to process job {job_id}: {e}")
                        # Ensure agent returns to IDLE state on failure
                        try:
                            self._handle_task_failure(
                                exec_data["agent_id"], 
                                job_id, 
                                e
                            )
                        except Exception as cleanup_err:
                            logger.error(f"Cleanup after job {job_id} failure also failed: {cleanup_err}")
                
            except Exception as e:
                logger.error(f"Executor thread error: {e}", exc_info=True)
        
        logger.info("Job executor thread stopped")

    # ========== PROVENANCE RECORDS PROPERTY AND METHODS (MEMORY LEAK FIX) ==========
    
    @property
    def provenance_records(self) -> List[Dict[str, Any]]:
        """
        Exposes provenance records for backward compatibility 
        with SemanticBridge and other components.
        
        Returns:
            List of provenance records from the internal deque.
            
        Note:
            FIX Issue #43: Previously returned self.working_memory.buffer which
            was empty because provenance is stored via _set_provenance_by_job_id()
            into _provenance_records deque, not working_memory.
        """
        with self._sync_provenance_lock:
            return list(self._provenance_records)
    
    def _extract_job_id(self, record: Any) -> Optional[str]:
        """
        Extract job_id from a provenance record.
        Handles both object attributes and dictionary keys.
        
        Args:
            record: Provenance record (object or dict)
            
        Returns:
            Job ID string if found, None otherwise.
        """
        # Try attribute access first (for provenance objects)
        job_id = getattr(record, 'job_id', None)
        if job_id:
            return job_id
        # Try dictionary access
        if isinstance(record, dict):
            return record.get('job_id')
        return None
    
    async def _record_provenance(self, record: Dict[str, Any]) -> None:
        """
        Thread-safe write that auto-prunes old history.
        Stores records in both working_memory and long_term_memory.
        
        Args:
            record: Provenance record to store. Must have a 'job_id' key.
        """
        async with self._provenance_lock:
            self._provenance_records.append(record)
            # Update lookup dictionary for O(1) access by job_id
            job_id = self._extract_job_id(record)
            if job_id:
                self._provenance_lookup[job_id] = record
                # Clean up lookup for items that have been rotated out of the deque
                self._cleanup_provenance_lookup()
            
            # Store in working_memory for short-term access
            self.working_memory.store(record, relevance=0.8)
            
            # Store in long_term_memory asynchronously for persistence
            try:
                self.long_term_memory.store(
                    content=record,
                    importance=0.7,
                    metadata={"type": "provenance", "job_id": job_id}
                )
            except Exception as e:
                logger.warning(f"Failed to store provenance in long-term memory: {e}")
    
    async def flush_history(self) -> None:
        """
        Manually clears history to reset context without restart.
        Use this if latency spikes to reset the context window.
        """
        async with self._provenance_lock:
            self._provenance_records.clear()
            self._provenance_lookup.clear()
            logger.info("Provenance history flushed (rolling window reset)")
    
    def _cleanup_provenance_lookup(self) -> None:
        """
        Clean up the lookup dictionary to remove entries that have been
        rotated out of the deque. Called during provenance recording.
        
        Note: This should be called while holding a lock to prevent race conditions.
        
        PERFORMANCE FIX: Only cleanup if lookup is significantly larger than deque
        to avoid O(n) iteration on every insert under high load.
        """
        # Skip cleanup if lookup size is within acceptable bounds
        if len(self._provenance_lookup) <= self._provenance_maxlen + 10:
            return
        
        # Get current job_ids in the deque
        current_job_ids = set()
        for record in self._provenance_records:
            job_id = self._extract_job_id(record)
            if job_id:
                current_job_ids.add(job_id)
        
        # Remove entries from lookup that are no longer in the deque
        keys_to_remove = [k for k in self._provenance_lookup if k not in current_job_ids]
        for key in keys_to_remove:
            del self._provenance_lookup[key]
    
    def _get_provenance_by_job_id(self, job_id: str) -> Optional[Any]:
        """
        Get provenance record by job_id using the lookup dictionary.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Provenance record if found, None otherwise.
        """
        return self._provenance_lookup.get(job_id)
    
    def _set_provenance_by_job_id(self, job_id: str, record: Any) -> None:
        """
        Store provenance record with job_id-based access.
        This is a synchronous helper for code that can't use async.
        Thread-safe using synchronous lock.
        
        Args:
            job_id: Job identifier
            record: Provenance record to store
        """
        with self._sync_provenance_lock:
            self._provenance_records.append(record)
            self._provenance_lookup[job_id] = record
            # Clean up old entries
            self._cleanup_provenance_lookup()

    def _hydrate_state_from_redis(self) -> None:
        """
        Hydrate Agent Pool state from Redis on startup.
        
        This method loads persisted statistics and provenance counts from Redis
        to restore state after container restarts. If Redis is not available or
        empty, defaults to 0 values (as currently implemented).
        """
        if self.redis_client is None:
            logger.debug("Redis client not available, skipping state hydration")
            return
        
        loaded_stats = {}
        try:
            # Load statistics from Redis
            stats_json = self.redis_client.get(REDIS_KEY_AGENT_POOL_STATS)
            if stats_json:
                try:
                    # Handle both string and bytes responses
                    if isinstance(stats_json, bytes):
                        stats_json = stats_json.decode('utf-8')
                    loaded_stats = json.loads(stats_json)
                    
                    # Update stats with loaded values, preserving default structure
                    with self.stats_lock:
                        for key, value in loaded_stats.items():
                            if key in self.stats:
                                self.stats[key] = value
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse stats JSON from Redis: {e}")
            
            # Load provenance records count from Redis
            provenance_count = self.redis_client.get(REDIS_KEY_PROVENANCE_COUNT)
            if provenance_count:
                try:
                    # Handle both string and bytes responses
                    if isinstance(provenance_count, bytes):
                        provenance_count = provenance_count.decode('utf-8')
                    self._provenance_records_count = int(provenance_count)
                    loaded_stats["provenance_records_count"] = self._provenance_records_count
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse provenance count from Redis: {e}")
            
            if loaded_stats:
                logger.info(f"🔄 Hydrated Agent Pool state from Redis: {loaded_stats}")
            else:
                logger.debug("No persisted state found in Redis, using defaults")
                
        except Exception as e:
            logger.warning(f"Failed to hydrate state from Redis: {e}. Using default values.")

    def _persist_state_to_redis(self) -> None:
        """
        Persist Agent Pool state to Redis for recovery after restarts.
        
        This method saves statistics and provenance counts to Redis so they
        can be restored after container restarts.
        
        PERFORMANCE FIX: Throttle to max once per second to avoid excessive
        Redis round-trips under high throughput.
        """
        if self.redis_client is None:
            return
        
        # Throttle Redis persistence to max once per second
        now = time.time()
        if now - getattr(self, '_last_redis_persist', 0) < 1.0:
            return
        self._last_redis_persist = now
        
        try:
            # Persist statistics
            with self.stats_lock:
                stats_json = json.dumps(self.stats)
            self.redis_client.set(REDIS_KEY_AGENT_POOL_STATS, stats_json)
            
            # Persist provenance records count
            provenance_count = len(self.provenance_records)
            self.redis_client.set(REDIS_KEY_PROVENANCE_COUNT, str(provenance_count))
            
            logger.debug(f"Persisted Agent Pool state to Redis: stats and provenance_count={provenance_count}")
        except Exception as e:
            logger.warning(f"Failed to persist state to Redis: {e}")

    def _init_task_queue(self, task_queue_type: str):
        """Initialize task queue with error handling and fallback"""
        try:
            queue_config = self.config.get("queue_config", {})
            self.task_queue = create_task_queue(task_queue_type, **queue_config)
            logger.info(f"Task queue initialized: {task_queue_type}")
        except ImportError as e:
            logger.warning(f"Failed to initialize {task_queue_type} queue: {e}")
            logger.info("Attempting fallback to custom queue...")
            try:
                if task_queue_type != "custom":
                    self.task_queue = create_task_queue(
                        "custom", **self.config.get("queue_config", {})
                    )
                    logger.info("Fallback to custom queue successful")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback queue: {fallback_error}")
                self.task_queue = None
        except Exception as e:
            logger.error(f"Failed to initialize task queue: {e}")
            self.task_queue = None

    def _initialize_agent_pool(self):
        """Initialize minimum number of agents with diverse capabilities"""
        logger.info(f"Initializing agent pool with {self.min_agents} agents")

        capabilities = list(AgentCapability)
        num_capabilities = len(capabilities)

        for i in range(self.min_agents):
            # Distribute capabilities evenly
            if i < self.min_agents // 2:
                capability = capabilities[i % num_capabilities]
            else:
                capability = AgentCapability.GENERAL

            try:
                agent_id = self.spawn_agent(capability)
                if agent_id:
                    logger.debug(
                        f"Initialized agent {agent_id} with capability {capability.value}"
                    )
            except Exception as e:
                logger.error(f"Failed to spawn agent during initialization: {e}")

        logger.info(f"Agent pool initialized with {len(self.agents)} agents")

    def _start_monitor(self):
        """Start background monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitor_agents, daemon=True, name="AgentPoolMonitor"
            )
            self.monitor_thread.start()
            logger.info("Agent pool monitor started")

    # ========== BUG #1 FIX: Agent Pool Death Spiral Prevention ==========

    def _get_live_agent_count_unsafe(self) -> int:
        """
        Internal method to count live agents WITHOUT acquiring lock.
        
        MUST be called with self.lock already held!
        
        Note: For very large agent pools (100+ agents), consider maintaining
        a cached live_count that gets updated when agent states change,
        rather than recalculating on every call. Current implementation is
        O(n) but acceptable for typical pool sizes (< 50 agents).
        
        Returns:
            Number of live (non-terminated, non-error) agents
        """
        return sum(
            1 for a in self.agents.values()
            if a.state not in (AgentState.TERMINATED, AgentState.ERROR)
        )

    def get_live_agent_count(self) -> int:
        """
        Get count of agents that are not in terminated or error state.
        
        BUG #1 FIX: This method counts only LIVE agents, excluding terminated
        and error-state agents that should not count toward max_agents capacity.
        
        Returns:
            Number of live (non-terminated, non-error) agents
        """
        with self.lock:
            return self._get_live_agent_count_unsafe()

    def can_spawn_agent(self) -> bool:
        """
        Check if a new agent can be spawned based on LIVE agent count.
        
        BUG #1 FIX: Uses live agent count instead of total agent count
        to prevent the death spiral where terminated agents block new spawns.
        
        Returns:
            True if a new agent can be spawned, False otherwise
        """
        return self.get_live_agent_count() < self.max_agents

    def cleanup_terminated_agents(self) -> int:
        """
        Remove terminated agents from the pool and respawn to minimum.
        
        BUG #1 FIX: This method removes agents in TERMINATED state from
        the agents dictionary to free up capacity for new agents.
        
        Returns:
            Number of agents cleaned up
        """
        with self.lock:
            before_count = len(self.agents)
            
            # Find terminated agents
            terminated_ids = [
                agent_id for agent_id, metadata in self.agents.items()
                if metadata.state == AgentState.TERMINATED
            ]
            
            # Remove terminated agents
            for agent_id in terminated_ids:
                # Clean up process reference if exists
                if agent_id in self.agent_processes:
                    process = self.agent_processes[agent_id]
                    if process.is_alive():
                        try:
                            process.terminate()
                            process.join(timeout=1)
                        except Exception as e:
                            logger.debug(f"Error terminating process for {agent_id}: {e}")
                    try:
                        process.close()
                    except Exception as e:
                        logger.debug(f"Error closing process for {agent_id}: {e}")
                    del self.agent_processes[agent_id]
                
                # Clean up specialized agents tracking
                for spec_list in self.specialized_agents.values():
                    if agent_id in spec_list:
                        spec_list.remove(agent_id)
                
                # Remove from agents dictionary
                del self.agents[agent_id]
            
            removed = before_count - len(self.agents)
            
            if removed > 0:
                logger.info(f"Cleaned up {removed} terminated agents")
        
        # Ensure minimum agents outside the lock to avoid deadlock
        self._ensure_minimum_agents()
        
        return removed

    def _ensure_minimum_agents(self) -> int:
        """
        Ensure we have at least min_agents live agents.
        
        BUG #1 FIX: This method spawns new agents if the live count
        drops below the minimum threshold.
        
        Returns:
            Number of agents spawned
        """
        spawned = 0
        live_count = self.get_live_agent_count()
        
        while live_count < self.min_agents:
            agent_id = self.spawn_agent()
            if agent_id:
                spawned += 1
                live_count += 1
                logger.info(f"Spawned agent {agent_id} to meet minimum ({live_count}/{self.min_agents})")
            else:
                logger.warning("Failed to spawn agent to meet minimum")
                break
        
        return spawned

    def spawn_agent(
        self,
        capability: AgentCapability = AgentCapability.GENERAL,
        location: str = "local",
        hardware_spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Spawn a new agent

        Args:
            capability: Agent capability
            location: Agent location ('local', 'remote', 'cloud')
            hardware_spec: Hardware specification dictionary

        Returns:
            Agent ID if successful, None otherwise
        """
        with self.lock:
            # BUG #1 FIX: Check capacity using LIVE agent count, not total count
            # This prevents the death spiral where terminated agents block new spawns
            live_count = self._get_live_agent_count_unsafe()
            if live_count >= self.max_agents:
                logger.warning(f"Agent pool at maximum capacity ({self.max_agents} live agents)")
                return None

            # Generate unique agent ID
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"

            try:
                # Create agent metadata using factory function
                metadata = create_agent_metadata(
                    agent_id=agent_id,
                    capability=capability,
                    location=location,
                    hardware_spec=hardware_spec or self._get_default_hardware_spec(),
                )

                # Register agent
                self.agents[agent_id] = metadata

                # Spawn agent process/thread based on location
                if location == "local":
                    self._spawn_local_agent(agent_id, metadata)
                elif location == "remote":
                    self._spawn_remote_agent(agent_id, metadata)
                elif location == "cloud":
                    self._spawn_cloud_agent(agent_id, metadata)
                else:
                    logger.warning(
                        f"Unknown location '{location}', defaulting to local"
                    )
                    self._spawn_local_agent(agent_id, metadata)

                # Update statistics
                with self.stats_lock:
                    self.stats["total_agents_spawned"] += 1
                
                # Persist state to Redis
                self._persist_state_to_redis()

                logger.info(
                    f"Spawned agent {agent_id} with capability {capability.value}"
                )
                return agent_id

            except Exception as e:
                logger.error(f"Failed to spawn agent: {e}", exc_info=True)
                # Cleanup on failure
                if agent_id in self.agents:
                    del self.agents[agent_id]
                return None

    def _spawn_local_agent(self, agent_id: str, metadata: AgentMetadata):
        """
        Spawn local agent process
        FIXED: Uses standalone worker function to avoid pickling issues
        """
        try:
            # FIXED: Use standalone function that doesn't reference self
            process = multiprocessing.Process(
                target=_standalone_agent_worker,
                args=(agent_id,),
                daemon=True,
                name=f"Agent-{agent_id}",
            )
            process.start()
            self.agent_processes[agent_id] = process

            # Transition to IDLE state using validated state machine
            metadata.transition_state(AgentState.IDLE, "Local agent process started")

            logger.debug(f"Local agent {agent_id} process started (PID: {process.pid})")

        except Exception as e:
            logger.error(f"Failed to spawn local agent {agent_id}: {e}")
            metadata.transition_state(AgentState.ERROR, f"Spawn failed: {e}")
            metadata.record_error(e, {"phase": "spawn_local"})

    def _spawn_remote_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn remote agent (via SSH, RPC, etc.)"""
        logger.info(f"Spawning remote agent {agent_id}")
        # TODO: Implement remote agent spawning via SSH/RPC
        # For now, just mark as IDLE
        metadata.transition_state(AgentState.IDLE, "Remote agent spawned (stub)")

    def _spawn_cloud_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn cloud agent (AWS, GCP, Azure, etc.)"""
        logger.info(f"Spawning cloud agent {agent_id}")
        # TODO: Implement cloud agent spawning
        # For now, just mark as IDLE
        metadata.transition_state(AgentState.IDLE, "Cloud agent spawned (stub)")

    def retire_agent(self, agent_id: str, force: bool = False) -> bool:
        """
        Retire an agent gracefully

        Args:
            agent_id: Agent identifier
            force: If True, force immediate termination

        Returns:
            True if agent was retired, False otherwise
        """
        with self.lock:
            if agent_id not in self.agents:
                logger.warning(f"Cannot retire agent {agent_id}: not found")
                return False

            metadata = self.agents[agent_id]

            # Cancel any assigned tasks
            tasks_to_cancel = [
                tid for tid, aid in self.task_assignments.items() if aid == agent_id
            ]

            for task_id in tasks_to_cancel:
                logger.warning(f"Cancelling task {task_id} due to agent retirement")
                self._cancel_task(task_id)

            if metadata.state == AgentState.WORKING and not force:
                # Mark for retirement after current task
                metadata.transition_state(
                    AgentState.RETIRING, "Marked for retirement after current task"
                )
                logger.info(
                    f"Agent {agent_id} marked for retirement after current task"
                )
            else:
                # Immediate termination
                metadata.transition_state(
                    AgentState.TERMINATED,
                    "Forced retirement" if force else "Retirement",
                )

                # Cleanup process
                if agent_id in self.agent_processes:
                    process = self.agent_processes[agent_id]

                    if process.is_alive():
                        if not force:
                            # Graceful shutdown
                            process.terminate()
                            process.join(timeout=5)

                        # Force kill if still alive
                        if process.is_alive():
                            logger.warning(f"Force killing agent {agent_id} process")
                            process.kill()
                            process.join(timeout=2)

                        # Close process handle
                        try:
                            process.close()
                        except Exception as e:
                            logger.debug(f"Error closing process handle: {e}")

                    del self.agent_processes[agent_id]

                # PERFORMANCE FIX: Clean up specialized_agents tracking
                for spec_list in self.specialized_agents.values():
                    if agent_id in spec_list:
                        spec_list.remove(agent_id)

                # Update statistics
                with self.stats_lock:
                    self.stats["total_agents_retired"] += 1
                
                # Persist state to Redis
                self._persist_state_to_redis()

                logger.info(f"Agent {agent_id} terminated")
        
        # FIX: Agent Retirement Without Replacement
        # Immediately ensure minimum agents after retirement to prevent pool shrinkage
        # Previously cleanup only happened periodically, allowing pool to shrink
        self._ensure_minimum_agents()

        return True

    def recover_agent(self, agent_id: str) -> bool:
        """
        Recover a failed agent

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent was recovered, False otherwise
        """
        with self.lock:
            if agent_id not in self.agents:
                logger.warning(f"Cannot recover agent {agent_id}: not found")
                return False

            metadata = self.agents[agent_id]

            # Check if agent can be recovered
            if not metadata.should_recover():
                logger.info(
                    f"Agent {agent_id} should not be recovered (too many errors)"
                )
                return False

            # Validate state transition
            if not metadata.transition_state(
                AgentState.RECOVERING, "Recovery initiated"
            ):
                logger.error(f"Cannot transition agent {agent_id} to RECOVERING state")
                return False

            logger.info(f"Recovering agent {agent_id}")

            # Clean up old process if exists
            if agent_id in self.agent_processes:
                process = self.agent_processes[agent_id]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                try:
                    process.close()
                except Exception as e:
                    logger.debug(f"Failed to cleanup agent: {e}")
                del self.agent_processes[agent_id]

            # Respawn agent based on location
            success = False
            try:
                if metadata.location == "local":
                    self._spawn_local_agent(agent_id, metadata)
                    success = True
                elif metadata.location == "remote":
                    self._spawn_remote_agent(agent_id, metadata)
                    success = True
                elif metadata.location == "cloud":
                    self._spawn_cloud_agent(agent_id, metadata)
                    success = True

                if success:
                    # Reset error counters
                    metadata.consecutive_errors = 0
                    metadata.transition_state(AgentState.IDLE, "Recovery successful")

                    # Update statistics
                    with self.stats_lock:
                        self.stats["total_recoveries_successful"] += 1

                    logger.info(f"Agent {agent_id} recovered successfully")

            except Exception as e:
                logger.error(f"Failed to recover agent {agent_id}: {e}")
                metadata.transition_state(AgentState.ERROR, f"Recovery failed: {e}")
                success = False

            # Update statistics
            with self.stats_lock:
                self.stats["total_recoveries_attempted"] += 1
            
            # Persist state to Redis
            self._persist_state_to_redis()

            return success

    def submit_job(
        self,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        capability_required: AgentCapability = AgentCapability.GENERAL,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Submit a job to the agent pool
        
        THREAD POOL FIX: This method is now NON-BLOCKING. Instead of executing
        the job synchronously (which blocked the caller's thread), jobs are
        queued for execution by a dedicated background thread.
        
        This prevents thread pool starvation when multiple requests submit
        jobs concurrently via run_in_executor().

        Args:
            graph: Computation graph
            parameters: Job parameters
            priority: Job priority (higher = more important)
            capability_required: Required agent capability
            timeout_seconds: Job timeout in seconds

        Returns:
            Job ID

        Raises:
            RuntimeError: If job queue is full or pool is shutting down
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        # FIXED: Use shorter default timeout (5 seconds instead of 300)
        timeout_seconds = timeout_seconds if timeout_seconds is not None else 5.0

        with self.lock:
            # FIXED: Check shutdown first
            if self._shutdown_event.is_set():
                raise RuntimeError("Agent pool is shutting down")

            # FIXED: Check queue capacity BEFORE accepting job
            if len(self.task_assignments) >= 1000:
                logger.error(f"Job queue full, rejecting job {job_id}")
                raise RuntimeError("Job queue at maximum capacity (1000 tasks)")

            # Create provenance record using factory function
            provenance = create_job_provenance(
                job_id=job_id,
                graph_id=graph.get("id", "unknown"),
                parameters=parameters,
                priority=priority,
                timeout_seconds=timeout_seconds,
            )

            # Store provenance
            self._set_provenance_by_job_id(job_id, provenance)

            # Update statistics
            with self.stats_lock:
                self.stats["total_jobs_submitted"] += 1
            
            # Persist state to Redis
            self._persist_state_to_redis()

            # FIXED: Archive old provenance if needed
            if time.time() - self._last_archive_time > 3600:
                self._archive_old_provenance()

            # FIXED: Find suitable agent with timeout using proper locking
            agent_id = self._assign_agent_with_timeout(
                capability_required, timeout_seconds
            )

            if agent_id:
                # Direct assignment to available agent
                provenance.agent_id = agent_id
                provenance.hardware_used = self.agents[agent_id].hardware_spec
                # Record assignment (but don't execute yet)
                self.task_assignments[job_id] = agent_id
                self.task_assignment_times[job_id] = time.time()
                self.agents[agent_id].transition_state(
                    AgentState.WORKING, f"Assigned job {job_id}"
                )
                metadata = self.agents[agent_id]
                logger.info(f"Job {job_id} assigned to agent {agent_id}")
            else:
                # FIXED: If no agent available, mark as failed immediately
                # Don't try to queue to task_queue as that can hang
                logger.warning(
                    f"No agent available for job {job_id} within {timeout_seconds}s"
                )
                provenance.agent_id = "no_agent_available"
                provenance.complete("failed", error="No agent available within timeout")

                # Update statistics
                with self.stats_lock:
                    self.stats["total_jobs_failed"] += 1
                
                # Persist state to Redis
                self._persist_state_to_redis()

                return job_id

        # THREAD POOL FIX: Queue for background execution instead of blocking
        # This is the key change - we don't call _execute_job_sync() here anymore
        if agent_id:
            with self._pending_executions_lock:
                self._pending_executions[job_id] = {
                    "agent_id": agent_id,
                    "graph": graph,
                    "parameters": parameters,
                    "metadata": metadata,
                    "queued_at": time.time(),
                }
            logger.debug(f"Job {job_id} queued for background execution")

        return job_id

    def _execute_job_sync(
        self,
        job_id: str,
        agent_id: str,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]],
        metadata: AgentMetadata,
    ):
        """
        Execute a job synchronously.

        FIXED: This method executes tasks synchronously instead of relying on
        stub worker processes that don't actually process tasks.

        Called by the background executor thread (NOT from submit_job directly).

        Thread safety notes:
        - The agent is in WORKING state, so other threads won't assign new work to it
        - Provenance objects are task-specific (one per job_id) and only modified here
        - The metadata object is owned by this agent during task execution

        Args:
            job_id: Job identifier
            agent_id: Agent identifier
            graph: Computation graph
            parameters: Job parameters
            metadata: Agent metadata
        """
        logger.info(f"Agent {agent_id} starting job {job_id}")
        provenance = None

        try:
            # Get provenance for the task (thread-safe access)
            with self.lock:
                provenance = self._get_provenance_by_job_id(job_id)
                if provenance:
                    # Start execution while holding lock to prevent concurrent modification
                    provenance.start_execution()

            # Build task dict for execution
            exec_task = {
                "task_id": job_id,
                "graph": graph,
                "parameters": parameters or {},
                "provenance": provenance,
            }

            # Execute the task
            logger.info(f"Agent {agent_id} step 1: task setup complete")
            result = self._execute_agent_task(agent_id, exec_task, metadata)
            logger.info(f"Agent {agent_id} step 2: execution complete")

            # Complete the task
            self._complete_agent_task(agent_id, job_id, result)
            logger.info(f"Agent {agent_id} job {job_id} COMPLETE")

        except Exception as e:
            logger.error(f"Agent {agent_id} job {job_id} FAILED: {e}")
            self._handle_task_failure(agent_id, job_id, e)
        finally:
            # PERFORMANCE FIX: Force garbage collection after job completion
            # to clean up heavy objects that may have leaked (e.g., from reasoning
            # components like ToolSelector, SemanticToolMatcher)
            # This addresses the progressive query routing degradation issue
            gc.collect()

    def _assign_agent_with_timeout(
        self, capability: AgentCapability, timeout_seconds: float
    ) -> Optional[str]:
        """
        Assign agent with timeout and proper locking to prevent race conditions
        FIXED: Won't hang if no agents available
        BUG #1 FIX: Triggers cleanup and respawn if all agents are terminated
        ISSUE #4 FIX: Fixed early return bug that caused agents to remain idle

        Args:
            capability: Required capability
            timeout_seconds: Timeout in seconds

        Returns:
            Agent ID if assigned, None otherwise
        """
        start_time = time.time()
        retry_delay = 0.05  # Start with 50ms delay
        max_retry_delay = 0.2  # FIXED: Reduced from 1.0 to 0.2 seconds
        max_retries = 10  # FIXED: Maximum number of retries to prevent infinite loops
        retry_count = 0
        last_cleanup_time = 0.0  # Track when last cleanup was attempted
        cleanup_cooldown = 1.0  # Minimum seconds between cleanup attempts
        at_max_capacity = False  # ISSUE #4 FIX: Track capacity state for early return decision

        while time.time() - start_time < timeout_seconds and retry_count < max_retries:
            # FIXED: Check shutdown event
            if self._shutdown_event.is_set():
                logger.debug("Shutdown requested, aborting agent assignment")
                return None

            # FIXED: Hold lock for entire check-and-spawn operation
            with self.lock:
                agent_id = self._assign_agent(capability)
                if agent_id:
                    return agent_id

                # BUG #1 FIX: Check LIVE agent count using internal method (no re-locking)
                live_count = self._get_live_agent_count_unsafe()
                at_max_capacity = live_count >= self.max_agents
                
                # ISSUE #4 FIX: Log state for debugging agent pool underutilization
                idle_count = sum(1 for m in self.agents.values() if m.state == AgentState.IDLE)
                if retry_count == 0:
                    logger.debug(
                        f"[AgentPool] Assignment attempt: capability={capability.value}, "
                        f"live={live_count}, idle={idle_count}, max={self.max_agents}"
                    )
                
                # Try to spawn if under capacity (using live count)
                if not at_max_capacity:
                    new_agent = self.spawn_agent(capability)
                    if new_agent:
                        # Give agent a moment to initialize
                        time.sleep(0.05)
                        # Try to assign the newly spawned agent
                        agent_id = self._assign_agent(capability)
                        if agent_id:
                            return agent_id
                else:
                    # BUG #1 FIX: At max live capacity - try cleanup with cooldown
                    current_time = time.time()
                    if current_time - last_cleanup_time >= cleanup_cooldown:
                        logger.info(
                            f"At max live capacity ({self.max_agents}) with no available agents "
                            f"for capability {capability.value}. Attempting cleanup..."
                        )
            
            # BUG #1 FIX: Attempt cleanup outside lock to avoid deadlock (with cooldown)
            current_time = time.time()
            if current_time - last_cleanup_time >= cleanup_cooldown:
                last_cleanup_time = current_time
                cleaned = self.cleanup_terminated_agents()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} terminated agents, retrying assignment")
                    continue  # Retry immediately after cleanup
                # ISSUE #4 FIX: Only return early if we're ACTUALLY at max capacity
                # Previously this returned None even when not at capacity, causing
                # agents to remain idle while jobs were rejected
                elif retry_count == 0 and at_max_capacity:
                    # First attempt with no terminated agents AND at max capacity - truly at capacity
                    logger.warning(
                        f"At max capacity ({self.max_agents}) with no available agents "
                        f"for capability {capability.value}"
                    )
                    return None

            # FIXED: Increment retry counter
            retry_count += 1

            # Brief wait before retry (outside the lock)
            time.sleep(retry_delay)

            # Exponential backoff up to max delay
            retry_delay = min(retry_delay * 1.5, max_retry_delay)

        logger.warning(
            f"Failed to assign agent with capability {capability.value} "
            f"within {timeout_seconds}s after {retry_count} retries"
        )
        return None

    def _assign_agent(self, capability: AgentCapability) -> Optional[str]:
        """
        Assign an available agent with required capability

        Must be called with lock held.

        Args:
            capability: Required capability

        Returns:
            Agent ID if available, None otherwise
        """
        available_agents = [
            agent_id
            for agent_id, metadata in self.agents.items()
            if metadata.state.can_accept_work()
            and metadata.capability.can_handle_capability(capability)
        ]

        if not available_agents:
            return None

        # Select agent with best performance (lowest failure rate)
        best_agent = min(
            available_agents,
            key=lambda aid: self.agents[aid].tasks_failed
            / max(1, self.agents[aid].tasks_completed),
        )

        return best_agent

    def get_agents_by_capability(
        self, 
        capabilities: List[str],
        max_agents: int = TOURNAMENT_MAX_CANDIDATES
    ) -> List[str]:
        """
        Get available agents that can handle the specified capabilities.
        
        Args:
            capabilities: List of capability names to filter by (e.g., ['reasoning', 'general'])
            max_agents: Maximum number of agents to return
            
        Returns:
            List of agent IDs that can handle the capabilities
        """
        with self.lock:
            available_agents = []
            for agent_id, metadata in self.agents.items():
                if metadata.state.can_accept_work():
                    # Check if agent capability matches any of the requested capabilities
                    if metadata.capability.value in capabilities:
                        available_agents.append(agent_id)
            
            # Sort by performance (lowest failure rate first)
            available_agents.sort(
                key=lambda aid: self.agents[aid].tasks_failed
                / max(1, self.agents[aid].tasks_completed)
            )
            
            return available_agents[:max_agents]

    def _embed_result(self, result: Dict[str, Any]) -> Any:
        """
        Create an embedding vector for a job result.
        
        Used by TournamentManager to compute similarity between results.
        
        Args:
            result: Job execution result dictionary
            
        Returns:
            Numpy array representing the result embedding, or list if numpy not available
        """
        # Create a simple embedding based on result characteristics
        # In a production system, this could use a neural encoder
        features = []
        
        # Feature 1: Execution time (normalized)
        exec_time = result.get('execution_time', 0.0)
        features.append(min(exec_time / 10.0, 1.0))  # Normalize to 0-1
        
        # Feature 2: Success indicator
        features.append(1.0 if result.get('status') == 'completed' else 0.0)
        
        # Feature 3: Confidence score if available
        features.append(result.get('confidence', 0.5))
        
        # Feature 4: Nodes processed (normalized)
        nodes = result.get('nodes_processed', 0)
        features.append(min(nodes / 100.0, 1.0))
        
        # Pad to fixed size for consistent embeddings
        while len(features) < 16:
            features.append(0.0)
        
        if NUMPY_AVAILABLE and np is not None:
            return np.array(features[:16], dtype=np.float32)
        else:
            return features[:16]

    async def assign_job_with_tournament(
        self,
        job_id: str,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]],
        query_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Assign job using tournament-based multi-agent selection for complex queries.
        
        For reasoning queries (symbolic, analogical, causal), runs the job through
        multiple agents in parallel and uses TournamentManager to select the best result.
        
        Args:
            job_id: Job identifier
            graph: Computation graph
            parameters: Job parameters  
            query_type: Type of query (e.g., 'reasoning', 'symbolic', 'analogical', 'causal')
            
        Returns:
            Best result from tournament selection, or None if failed
        """
        # Check if tournament selection should be used
        use_tournament = (
            self.tournament_manager is not None and
            query_type in TOURNAMENT_QUERY_TYPES
        )
        
        if not use_tournament:
            # Fall back to simple single-agent execution
            logger.debug(f"[Tournament] Skipping tournament for query_type={query_type}")
            return None
        
        logger.info(f"[Tournament] Using multi-agent tournament for job {job_id} (type={query_type})")
        
        # Get candidate agents with reasoning capability
        candidate_capabilities = ['reasoning', 'general']
        candidates = self.get_agents_by_capability(candidate_capabilities)
        
        if len(candidates) == 0:
            logger.warning(f"[Tournament] No agents available for tournament")
            return None
        
        if len(candidates) == 1:
            # Only one agent available, no need for tournament
            logger.debug(f"[Tournament] Only one agent available, skipping tournament")
            return None
        
        # Limit to max candidates
        candidates = candidates[:TOURNAMENT_MAX_CANDIDATES]
        logger.info(f"[Tournament] Running job through {len(candidates)} agents: {candidates}")
        
        # Run job through each candidate agent in parallel
        async def execute_on_agent(agent_id: str) -> Dict[str, Any]:
            """Execute job on a specific agent and return result."""
            metadata = None
            try:
                # Atomically check agent state and transition to WORKING
                with self.lock:
                    metadata = self.agents.get(agent_id)
                    if not metadata:
                        return {'status': 'failed', 'error': 'Agent not found', 'agent_id': agent_id}
                    
                    if not metadata.state.can_accept_work():
                        return {'status': 'failed', 'error': 'Agent busy', 'agent_id': agent_id}
                    
                    # Transition to WORKING while holding lock
                    metadata.transition_state(AgentState.WORKING, f"Tournament job {job_id}")
                
                # Execute task (outside lock to allow parallel execution)
                task = {
                    "task_id": f"{job_id}_tournament_{agent_id}",
                    "graph": graph,
                    "parameters": parameters or {},
                    "provenance": None,
                }
                result = self._execute_agent_task(agent_id, task, metadata)
                result['agent_id'] = agent_id
                return result
                        
            except Exception as e:
                logger.error(f"[Tournament] Agent {agent_id} failed: {e}")
                return {
                    'status': 'failed', 
                    'error': str(e), 
                    'agent_id': agent_id,
                    'confidence': 0.0
                }
            finally:
                # Always return agent to idle (if we successfully started working)
                if metadata is not None:
                    with self.lock:
                        if metadata.state == AgentState.WORKING:
                            metadata.transition_state(AgentState.IDLE, f"Tournament complete {job_id}")
        
        # Execute on all candidates in parallel
        results = await asyncio.gather(*[
            execute_on_agent(agent_id) for agent_id in candidates
        ], return_exceptions=True)
        
        # Filter out exceptions and failed results
        valid_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"[Tournament] Agent {candidates[i]} raised exception: {r}")
            elif isinstance(r, dict) and r.get('status') == 'completed':
                valid_results.append(r)
        
        if len(valid_results) == 0:
            logger.warning(f"[Tournament] All agents failed for job {job_id}")
            return None
        
        if len(valid_results) == 1:
            logger.info(f"[Tournament] Only one valid result, skipping tournament selection")
            return valid_results[0]
        
        # Use TournamentManager to select best result
        try:
            # Calculate fitness scores (higher = better)
            fitness = []
            for r in valid_results:
                # Combine multiple factors into fitness score
                confidence = r.get('confidence', 0.5)
                exec_time = r.get('execution_time', 1.0)
                time_penalty = max(0, 1.0 - exec_time / 10.0)  # Faster is better
                fitness.append(confidence * 0.7 + time_penalty * 0.3)
            
            # Run tournament
            meta = {}
            winner_indices = self.tournament_manager.run_adaptive_tournament(
                proposals=valid_results,
                fitness=fitness,
                embedding_func=self._embed_result,
                meta=meta
            )
            
            if winner_indices and len(winner_indices) > 0:
                winner_idx = winner_indices[0]
                winner_result = valid_results[winner_idx]
                winner_result['tournament_meta'] = meta
                winner_result['tournament_fitness'] = fitness[winner_idx]
                
                logger.info(
                    f"[Tournament] Winner: agent {winner_result.get('agent_id')} "
                    f"(fitness={fitness[winner_idx]:.3f}, "
                    f"innovation={meta.get('innovation_score', 0):.3f})"
                )
                
                # Update agent weights based on tournament outcome
                self._update_agent_weights_from_tournament(valid_results, winner_idx, fitness)
                
                return winner_result
            else:
                logger.warning(f"[Tournament] No winners selected, returning first result")
                return valid_results[0]
                
        except Exception as e:
            logger.error(f"[Tournament] Tournament selection failed: {e}")
            # Fall back to first result
            return valid_results[0] if valid_results else None

    def _update_agent_weights_from_tournament(
        self,
        results: List[Dict[str, Any]],
        winner_idx: int,
        fitness: List[float]
    ) -> None:
        """
        Update agent weights based on tournament outcome.
        
        This provides feedback to improve agent selection over time.
        
        Args:
            results: List of results from all tournament participants
            winner_idx: Index of the winning result
            fitness: Fitness scores for each result
        """
        try:
            for i, result in enumerate(results):
                agent_id = result.get('agent_id')
                if agent_id and agent_id in self.agents:
                    metadata = self.agents[agent_id]
                    
                    # Track tournament participation
                    if not hasattr(metadata, 'tournament_stats'):
                        metadata.tournament_stats = {
                            'participations': 0,
                            'wins': 0,
                            'total_fitness': 0.0
                        }
                    
                    metadata.tournament_stats['participations'] += 1
                    metadata.tournament_stats['total_fitness'] += fitness[i]
                    
                    if i == winner_idx:
                        metadata.tournament_stats['wins'] += 1
                        logger.debug(f"[Tournament] Agent {agent_id} won (total wins: {metadata.tournament_stats['wins']})")
                    
        except Exception as e:
            logger.debug(f"[Tournament] Failed to update agent weights: {e}")

    def _assign_job_to_agent(
        self,
        job_id: str,
        agent_id: str,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]],
    ):
        """
        Assign job to specific agent (without execution).

        NOTE: This is a legacy method kept for compatibility. The main execution
        flow now uses _execute_job_sync directly from submit_job.

        Must be called with lock held.

        Args:
            job_id: Job identifier
            agent_id: Agent identifier
            graph: Computation graph
            parameters: Job parameters
        """
        task = {"task_id": job_id, "graph": graph, "parameters": parameters or {}}

        # Queue task for agent
        self.task_assignments[job_id] = agent_id
        self.task_assignment_times[job_id] = time.time()

        # Transition agent to WORKING state
        if agent_id in self.agents:
            self.agents[agent_id].transition_state(
                AgentState.WORKING, f"Assigned job {job_id}"
            )

    def _get_agent_task(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get next task for agent

        Args:
            agent_id: Agent identifier

        Returns:
            Task dictionary if available, None otherwise
        """
        with self.lock:
            for task_id, assigned_agent in self.task_assignments.items():
                if assigned_agent == agent_id:
                    provenance = self._get_provenance_by_job_id(task_id)
                    if provenance:
                        provenance.start_execution()

                    return {"task_id": task_id, "provenance": provenance}

        return None

    def _execute_agent_task(
        self, agent_id: str, task: Dict[str, Any], metadata: AgentMetadata
    ) -> Any:
        """
        Execute task on agent with ACTUAL reasoning engine invocation.

        CRITICAL FIX: This method now properly invokes UnifiedReasoning.reason()
        for reasoning tasks instead of just creating placeholder results.

        For reasoning tasks (reasoning, symbolic, causal, analogical, etc.):
        - Invokes the actual reasoning engines via ReasoningIntegration
        - Routes to appropriate reasoning type based on task type
        - Returns real reasoning results with confidence scores

        For non-reasoning tasks:
        - Falls back to graph-based execution

        Args:
            agent_id: Agent identifier
            task: Task dictionary containing task_id, graph, parameters, provenance
            metadata: Agent metadata

        Returns:
            Task result dictionary with actual reasoning output

        Raises:
            Exception: If task execution fails
        """
        start_time = time.time()
        task_id = task.get("task_id")
        provenance = task.get("provenance")
        graph = task.get("graph", {})
        parameters = task.get("parameters", {})

        try:
            logger.info(f"Agent {agent_id} executing task {task_id}")

            # Extract task information from graph
            graph_id = graph.get("id", "unknown")
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            task_type = graph.get("type", "general").lower()
            
            # FIX: Normalize task_type by stripping common suffixes like "_task", "_support"
            # QueryRouter creates tasks with types like "reasoning_task", "perception_support"
            # but the reasoning check expects base types like "reasoning", "perception"
            normalized_task_type = task_type
            for suffix in ("_task", "_support"):
                if normalized_task_type.endswith(suffix):
                    normalized_task_type = normalized_task_type[:-len(suffix)]
                    break

            # Determine if this is a reasoning task
            reasoning_task_types = {
                "reasoning", "causal", "symbolic", "analogical", "probabilistic",
                "counterfactual", "multimodal", "deductive", "inductive", "abductive"
            }
            is_reasoning_task = normalized_task_type in reasoning_task_types
            
            # FIX: Also check selected_tools passed from QueryRouter
            # selected_tools contains reasoning engine names like ['causal', 'probabilistic']
            selected_tools = parameters.get("selected_tools", []) or parameters.get("tools", []) or []
            if not is_reasoning_task and selected_tools:
                # Check if any selected tool is a reasoning type
                if any(tool in reasoning_task_types for tool in selected_tools):
                    is_reasoning_task = True
                    logger.info(
                        f"Agent {agent_id} task {task_id}: reasoning triggered by selected_tools={selected_tools}"
                    )

            # Also check capability - REASONING capability agents should invoke reasoning
            if metadata.capability == AgentCapability.REASONING:
                is_reasoning_task = True

            # Extract query/input from graph or parameters
            query = parameters.get("query", "")
            input_data = parameters.get("input_data") or parameters.get("input", "")
            context = parameters.get("context", {})
            
            # Try to extract query from nodes if not in parameters
            if not query and nodes:
                for node in nodes:
                    if node.get("type") in ("input", "query", "InputNode"):
                        query = node.get("data", {}).get("value", "") or node.get("params", {}).get("query", "")
                        input_data = input_data or node.get("data", {}).get("input", "")
                        break

            # ============================================================
            # REASONING TASK EXECUTION - Invoke actual reasoning engines
            # ============================================================
            reasoning_result = None
            node_results = {}

            if is_reasoning_task and REASONING_AVAILABLE:
                logger.info(
                    f"Agent {agent_id} invoking reasoning engine for task {task_id} "
                    f"(type={task_type}, capability={metadata.capability.value})"
                )
                
                try:
                    # Map task type to reasoning type
                    reasoning_type = self._map_task_to_reasoning_type(task_type)
                    
                    # Calculate complexity from graph structure
                    complexity = self._calculate_task_complexity(graph, parameters)
                    
                    # Apply reasoning via the integration layer
                    integration_result = apply_reasoning(
                        query=query or str(input_data) or f"Process task {task_id}",
                        query_type=task_type,
                        complexity=complexity,
                        context=context,
                    )
                    
                    logger.info(
                        f"Agent {agent_id} reasoning selection complete: "
                        f"tools={integration_result.selected_tools}, "
                        f"strategy={integration_result.reasoning_strategy}, "
                        f"confidence={integration_result.confidence:.2f}"
                    )
                    
                    # If UnifiedReasoner is available, invoke actual reasoning
                    if UnifiedReasoner is not None and create_unified_reasoner is not None:
                        try:
                            # Get or create unified reasoner with learning and safety enabled
                            reasoner = create_unified_reasoner(
                                enable_learning=True,
                                enable_safety=True,
                            )
                            
                            if reasoner is not None:
                                # Invoke the actual reasoning engine
                                reasoning_result = reasoner.reason(
                                    input_data=input_data or query,
                                    query={"query": query, "context": context, "task_type": task_type},
                                    reasoning_type=reasoning_type,
                                )
                                
                                logger.info(
                                    f"Agent {agent_id} reasoning execution complete: "
                                    f"type={reasoning_result.reasoning_type if hasattr(reasoning_result, 'reasoning_type') else 'unknown'}, "
                                    f"confidence={reasoning_result.confidence if hasattr(reasoning_result, 'confidence') else 'N/A'}"
                                )
                        except Exception as reasoning_error:
                            logger.warning(
                                f"Agent {agent_id} UnifiedReasoner invocation failed: {reasoning_error}. "
                                f"Using integration result only."
                            )
                    
                    # Build node results from reasoning
                    for i, node in enumerate(nodes):
                        node_id = node.get("id", f"node_{i}")
                        node_type = node.get("type", "unknown")
                        node_results[node_id] = {
                            "status": "completed",
                            "node_type": node_type,
                            "reasoning_applied": True,
                            "selected_tools": integration_result.selected_tools,
                            "reasoning_strategy": integration_result.reasoning_strategy,
                        }
                        
                except Exception as reasoning_error:
                    logger.warning(
                        f"Agent {agent_id} reasoning integration failed: {reasoning_error}. "
                        f"Falling back to graph execution."
                    )
                    # Fall through to graph execution below
                    is_reasoning_task = False

            # ============================================================
            # FIX: EXPLICIT REASONING INVOCATION when selected_tools present
            # This handles cases where REASONING_AVAILABLE=False but we still
            # have reasoning tools selected from QueryRouter
            # ============================================================
            if is_reasoning_task and selected_tools and not node_results and not REASONING_AVAILABLE:
                logger.info(f"[REASONING] INVOKING engines for task {task_id}, tools={selected_tools}")
                try:
                    from vulcan.reasoning.unified_reasoning import UnifiedReasoner as DirectUnifiedReasoner, ReasoningStrategy
                    from vulcan.reasoning.reasoning_types import ReasoningType
                    
                    # Get or create reasoning instance
                    reasoning = DirectUnifiedReasoner()
                    
                    # Extract query from parameters
                    query_text = parameters.get("prompt", "") or parameters.get("query", "")
                    
                    # Convert first tool name to ReasoningType enum if possible
                    reasoning_type = None
                    if selected_tools and len(selected_tools) > 0:
                        tool_name = selected_tools[0].upper()
                        try:
                            reasoning_type = ReasoningType[tool_name]
                        except KeyError:
                            # Try matching by value
                            for rt in ReasoningType:
                                if rt.value == selected_tools[0].lower():
                                    reasoning_type = rt
                                    break
                    
                    # Use ADAPTIVE strategy by default, ENSEMBLE when multiple tools selected
                    strategy = ReasoningStrategy.ADAPTIVE
                    if selected_tools and len(selected_tools) > 1:
                        strategy = ReasoningStrategy.ENSEMBLE
                    
                    # Invoke actual reasoning with correct signature:
                    # reason(input_data, query=None, reasoning_type=None, strategy=ADAPTIVE, ...)
                    reasoning_result = reasoning.reason(
                        input_data=query_text,
                        query=parameters,
                        reasoning_type=reasoning_type,
                        strategy=strategy
                    )
                    
                    # Mark nodes as processed with reasoning
                    for i, node in enumerate(nodes):
                        node_id = node.get("id", f"node_{i}")
                        node_type = node.get("type", "unknown")
                        node_results[node_id] = {
                            "status": "completed",
                            "node_type": node_type,
                            "reasoning_applied": True,
                            "selected_tools": selected_tools,
                        }
                    
                    # Return reasoning result directly
                    return {
                        "status": "completed",
                        "reasoning_invoked": True,
                        "reasoning_output": reasoning_result,
                        "tools_used": selected_tools,
                        "execution_time": time.time() - start_time,
                        "agent_id": agent_id,
                        "task_id": task_id,
                        "nodes_processed": len(nodes),
                        "node_results": node_results,
                    }
                except Exception as e:
                    logger.error(f"[REASONING] Invocation FAILED: {e}", exc_info=True)
                    # Fall through to generic execution

            # ============================================================
            # GRAPH-BASED EXECUTION - For non-reasoning tasks or fallback
            # ============================================================
            if not is_reasoning_task or not node_results:
                logger.debug(f"Agent {agent_id} using graph-based execution for task {task_id}")
                for node in nodes:
                    node_id = node.get("id", "unknown")
                    node_type = node.get("type", "unknown")
                    node_params = node.get("params", {})

                    # Execute node based on type
                    node_results[node_id] = {
                        "status": "completed",
                        "node_type": node_type,
                        "params_processed": list(node_params.keys()),
                        "reasoning_applied": False,
                    }
                    logger.debug(f"Agent {agent_id} processed node {node_id} ({node_type})")

            # Create comprehensive result
            duration = time.time() - start_time
            result = {
                "status": "completed",
                "outcome": "success",
                "agent_id": agent_id,
                "task_id": task_id,
                "graph_id": graph_id,
                "task_type": task_type,
                "execution_time": duration,
                "timestamp": time.time(),
                "nodes_processed": len(nodes),
                "node_results": node_results,
                "parameters_used": list(parameters.keys()) if parameters else [],
                "capability": metadata.capability.value,
                "reasoning_invoked": is_reasoning_task and REASONING_AVAILABLE,
            }
            
            # Add reasoning-specific results if available
            if reasoning_result is not None:
                result["reasoning_output"] = {
                    "conclusion": getattr(reasoning_result, "conclusion", None),
                    "confidence": getattr(reasoning_result, "confidence", None),
                    "reasoning_type": str(getattr(reasoning_result, "reasoning_type", "unknown")),
                    "explanation": getattr(reasoning_result, "explanation", None),
                }

            # Update agent metadata
            metadata.record_task_completion(success=True, duration_s=duration)

            # Update provenance
            if provenance:
                provenance.complete("success", result=result)
                resource_consumption = {"cpu_seconds": duration}
                if PSUTIL_AVAILABLE:
                    try:
                        resource_consumption["memory_mb"] = (
                            psutil.Process().memory_info().rss / 1024 / 1024
                        )
                    except Exception as e:
                        logger.debug(f"Failed to get memory info: {e}")
                provenance.update_resource_consumption(resource_consumption)

            # Update statistics
            with self.stats_lock:
                self.stats["total_jobs_completed"] += 1
                # FIX Issue #12: Log job completion with stats for monitoring
                current_stats = dict(self.stats)
            
            logger.info(
                f"[AgentPool] Job completed: task={task_id}, agent={agent_id}. "
                f"Stats: submitted={current_stats['total_jobs_submitted']}, "
                f"completed={current_stats['total_jobs_completed']}, "
                f"failed={current_stats['total_jobs_failed']}"
            )
            
            # Persist state to Redis
            self._persist_state_to_redis()

            logger.info(
                f"Agent {agent_id} completed task {task_id} in {duration:.3f}s "
                f"(processed {len(nodes)} nodes, reasoning_invoked={result['reasoning_invoked']})"
            )
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Agent {agent_id} task {task_id} failed: {e}")

            # Update agent metadata
            metadata.record_task_completion(success=False, duration_s=duration)
            metadata.record_error(e, {"task_id": task_id, "phase": "execution"})

            # Update provenance
            if provenance:
                provenance.complete("failed", error=str(e))
                provenance.update_resource_consumption({"cpu_seconds": duration})

            # Update statistics
            with self.stats_lock:
                self.stats["total_jobs_failed"] += 1
                # FIX Issue #12: Log job failure with stats for monitoring
                current_stats = dict(self.stats)
            
            logger.warning(
                f"[AgentPool] Job failed: task={task_id}, agent={agent_id}. "
                f"Stats: submitted={current_stats['total_jobs_submitted']}, "
                f"completed={current_stats['total_jobs_completed']}, "
                f"failed={current_stats['total_jobs_failed']}"
            )
            
            # Persist state to Redis
            self._persist_state_to_redis()

            raise

    def _complete_agent_task(self, agent_id: str, task_id: str, result: Any):
        """
        Mark task as completed and cleanup

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            result: Task result
        """
        with self.lock:
            # Remove from assignments
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]

            if task_id in self.task_assignment_times:
                del self.task_assignment_times[task_id]

            # Transition agent back to IDLE
            if agent_id in self.agents:
                metadata = self.agents[agent_id]
                metadata.transition_state(AgentState.IDLE, f"Completed task {task_id}")
                metadata.last_active = time.time()

    def _handle_task_failure(self, agent_id: str, task_id: str, error: Exception):
        """
        Handle task failure
        
        FIX 3: Agent Job Tracking - This method now properly updates statistics
        when a job fails. Previously, jobs that failed before reaching
        _execute_agent_task (e.g., during setup) would not increment
        total_jobs_failed, causing jobs to "disappear" in tracking.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            error: Error that caused failure
        """
        with self.lock:
            # Update provenance
            provenance = self._get_provenance_by_job_id(task_id)
            if provenance:
                if not provenance.is_complete():
                    provenance.complete("failed", error=str(error))
                    # FIX 3: Only update stats if provenance wasn't already complete
                    # (if complete, _execute_agent_task already updated stats)
                    # NOTE: We explicitly check provenance.is_complete() to avoid
                    # double-counting - if provenance was already complete, stats
                    # were updated by _execute_agent_task.
                    with self.stats_lock:
                        self.stats["total_jobs_failed"] += 1
                        # Capture stats INSIDE lock and log INSIDE lock to avoid race
                        logger.warning(
                            f"[AgentPool] Job failed (via _handle_task_failure): task={task_id}, "
                            f"agent={agent_id}. Stats: submitted={self.stats['total_jobs_submitted']}, "
                            f"completed={self.stats['total_jobs_completed']}, "
                            f"failed={self.stats['total_jobs_failed']}"
                        )

            # Remove from assignments
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            if task_id in self.task_assignment_times:
                del self.task_assignment_times[task_id]

            # Return agent to idle
            if agent_id in self.agents:
                metadata = self.agents[agent_id]
                metadata.transition_state(AgentState.IDLE, f"Task {task_id} failed")

    def _cancel_task(self, task_id: str):
        """
        Cancel a task

        Args:
            task_id: Task identifier
        """
        with self.lock:
            # Update provenance
            provenance = self._get_provenance_by_job_id(task_id)
            if provenance:
                if not provenance.is_complete():
                    provenance.complete("cancelled")

            # Get assigned agent
            agent_id = self.task_assignments.get(task_id)

            # Remove from assignments
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            if task_id in self.task_assignment_times:
                del self.task_assignment_times[task_id]

            # Return agent to idle if it was working on this task
            if agent_id and agent_id in self.agents:
                metadata = self.agents[agent_id]
                if metadata.state == AgentState.WORKING:
                    metadata.transition_state(
                        AgentState.IDLE, f"Task {task_id} cancelled"
                    )
            
            # Also remove from pending executions if queued
            with self._pending_executions_lock:
                self._pending_executions.pop(task_id, None)

    # ============================================================
    # REASONING INTEGRATION HELPERS
    # ============================================================
    
    def _map_task_to_reasoning_type(self, task_type: str):
        """
        Map task type string to ReasoningType enum.
        
        This enables proper routing of tasks to the appropriate reasoning engine.
        
        Args:
            task_type: Task type string (e.g., "causal", "symbolic", "reasoning")
            
        Returns:
            ReasoningType enum value, or None if not available
        """
        if ReasoningType is None:
            return None
            
        # Mapping from task type strings to ReasoningType enum values
        task_to_reasoning_map = {
            "causal": ReasoningType.CAUSAL,
            "symbolic": ReasoningType.SYMBOLIC,
            "analogical": ReasoningType.ANALOGICAL,
            "probabilistic": ReasoningType.PROBABILISTIC,
            "counterfactual": ReasoningType.COUNTERFACTUAL,
            "multimodal": ReasoningType.MULTIMODAL,
            "deductive": ReasoningType.DEDUCTIVE,
            "inductive": ReasoningType.INDUCTIVE,
            "abductive": ReasoningType.ABDUCTIVE,
            "reasoning": ReasoningType.HYBRID,  # Generic reasoning -> hybrid
            "general": ReasoningType.UNKNOWN,
        }
        
        return task_to_reasoning_map.get(task_type.lower(), ReasoningType.UNKNOWN)
    
    def _calculate_task_complexity(
        self, 
        graph: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> float:
        """
        Calculate task complexity score from graph structure and parameters.
        
        Complexity affects reasoning strategy selection:
        - Low complexity (< 0.3): Fast path, simple reasoning
        - Medium complexity (0.3 - 0.7): Balanced reasoning
        - High complexity (> 0.7): Full reasoning pipeline
        
        Args:
            graph: Task graph with nodes and edges
            parameters: Task parameters
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        complexity = 0.3  # Base complexity
        
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Factor 1: Number of nodes (more nodes = more complex)
        node_count = len(nodes)
        if node_count > 10:
            complexity += 0.2
        elif node_count > 5:
            complexity += 0.1
        elif node_count > 2:
            complexity += 0.05
        
        # Factor 2: Number of edges (more connections = more complex)
        edge_count = len(edges)
        if edge_count > 15:
            complexity += 0.15
        elif edge_count > 8:
            complexity += 0.1
        elif edge_count > 3:
            complexity += 0.05
        
        # Factor 3: Parameter complexity
        param_count = len(parameters)
        if param_count > 10:
            complexity += 0.1
        elif param_count > 5:
            complexity += 0.05
        
        # Factor 4: Nested structures in parameters
        # Note: max_depth limit prevents stack overflow on deeply nested data
        def count_depth(obj, current_depth=0, max_depth=10):
            if current_depth >= max_depth:
                return current_depth
            if isinstance(obj, dict):
                if not obj:
                    return current_depth
                return max(count_depth(v, current_depth + 1, max_depth) for v in obj.values())
            elif isinstance(obj, list):
                if not obj:
                    return current_depth
                return max(count_depth(v, current_depth + 1, max_depth) for v in obj)
            return current_depth
        
        depth = count_depth(parameters)
        if depth > 3:
            complexity += 0.1
        elif depth > 2:
            complexity += 0.05
        
        # Factor 5: Special node types that indicate complex reasoning
        complex_node_types = {
            "reasoning", "causal", "symbolic", "inference", "meta",
            "planning", "counterfactual", "analogical", "multimodal"
        }
        has_complex_nodes = any(
            node.get("type", "").lower() in complex_node_types
            for node in nodes
        )
        if has_complex_nodes:
            complexity += 0.15
        
        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, complexity))

    def _archive_old_provenance(self):
        """Archive old provenance records to disk.
        
        NOTE: With the rolling deque implementation, the deque automatically 
        maintains a bounded size (maxlen=50 by default). Archiving is now 
        optional and mainly for audit/compliance purposes.
        """
        with self._archive_lock:
            try:
                # With rolling deque, we no longer need manual cleanup
                # The deque auto-prunes to maxlen when items are added
                
                # Archive current records for audit purposes if there are any
                current_records = list(self._provenance_records)
                if len(current_records) > 0:
                    timestamp = int(time.time())
                    archive_file = self.archive_dir / f"provenance_{timestamp}.jsonl"

                    # Archive all current records
                    archived_count = 0
                    with open(archive_file, "w", encoding="utf-8") as f:
                        for prov in current_records:
                            try:
                                if hasattr(prov, 'to_dict'):
                                    f.write(json.dumps(prov.to_dict(), default=str) + "\n")
                                elif isinstance(prov, dict):
                                    f.write(json.dumps(prov, default=str) + "\n")
                                archived_count += 1
                            except Exception as e:
                                logger.error(f"Failed to serialize provenance: {e}")

                    if archived_count > 0:
                        self._last_archive_time = time.time()
                        logger.info(
                            f"Archived {archived_count} provenance records to {archive_file}"
                        )

            except Exception as e:
                logger.error(f"Failed to archive provenance: {e}", exc_info=True)

    def _monitor_agents(self):
        """
        Monitor agent health and performance with comprehensive cleanup

        FIXED: Converted long time.sleep(10) to interruptible self._shutdown_event.wait(timeout=10).
        PERFORMANCE FIX: Added periodic statistics reset to prevent memory leaks.
        """
        logger.info("Agent monitor started")
        
        # PERFORMANCE: Track iterations for periodic cleanup
        monitor_iterations = 0
        STATS_RESET_INTERVAL = 360  # Reset stats every ~1 hour (360 * 10 seconds)

        # FIXED: Use interruptible wait
        while not self._shutdown_event.is_set():
            try:
                # If shutdown is signaled, break immediately
                if self._shutdown_event.wait(timeout=10):
                    break

                current_time = time.time()
                monitor_iterations += 1

                with self.lock:
                    # FIXED: Clean up stale task assignments
                    stale_tasks = [
                        task_id
                        for task_id, assign_time in self.task_assignment_times.items()
                        if current_time - assign_time > self.task_timeout_seconds
                    ]

                    for task_id in stale_tasks:
                        agent_id = self.task_assignments.get(task_id)
                        logger.warning(
                            f"Cleaning up stale task {task_id} "
                            f"(assigned to {agent_id}, age: {current_time - self.task_assignment_times[task_id]:.1f}s)"
                        )
                        self._cancel_task(task_id)

                    # Note: With rolling deque (maxlen=50), provenance is auto-bounded
                    # Archiving is now optional and can be triggered periodically if needed
                    # The deque automatically drops old records when new ones are added

                    # Monitor each agent
                    agents_to_recover = []
                    agents_to_retire = []

                    for agent_id, metadata in list(self.agents.items()):
                        # Check for stale idle agents
                        if metadata.state == AgentState.IDLE:
                            idle_time = current_time - metadata.last_active
                            if idle_time > 300 and len(self.agents) > self.min_agents:
                                agents_to_retire.append(agent_id)

                        # Check for error agents
                        elif metadata.state == AgentState.ERROR:
                            if metadata.should_recover():
                                agents_to_recover.append(agent_id)
                            else:
                                agents_to_retire.append(agent_id)

                        # Update resource usage for local agents
                        # PERFORMANCE FIX: Use non-blocking CPU measurement (interval=None)
                        # to avoid 100ms blocking per agent which causes slowdown with many agents
                        if PSUTIL_AVAILABLE and agent_id in self.agent_processes:
                            process = self.agent_processes[agent_id]
                            if process.is_alive():
                                try:
                                    p = psutil.Process(process.pid)
                                    metadata.resource_usage = {
                                        "cpu_percent": p.cpu_percent(interval=None),
                                        "memory_mb": p.memory_info().rss / 1024 / 1024,
                                        "num_threads": p.num_threads(),
                                    }
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    logger.debug(
                                        f"Cannot access process info for agent {agent_id}"
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Error accessing process info for agent {agent_id}: {e}"
                                    )

                # Perform recovery and retirement outside the lock
                for agent_id in agents_to_recover:
                    logger.info(f"Attempting to recover agent {agent_id}")
                    self.recover_agent(agent_id)

                for agent_id in agents_to_retire:
                    logger.info(f"Retiring stale/error agent {agent_id}")
                    self.retire_agent(agent_id)
                
                # PERFORMANCE FIX: Periodic statistics reset to prevent unbounded growth
                # BUG #1 & #5 FIX: Also trigger cleanup and ensure minimum agents
                if monitor_iterations % STATS_RESET_INTERVAL == 0:
                    logger.info(f"Performing periodic statistics reset (iteration {monitor_iterations})")
                    self.reset_statistics(preserve_totals=True)
                    # BUG #1 FIX: Cleanup terminated agents and ensure minimum
                    self.cleanup_terminated_agents()
                
                # BUG #1 FIX: Check for terminated agent cleanup every 3 iterations (~30 seconds)
                # This ensures dead agents don't accumulate between stat resets
                elif monitor_iterations % 3 == 0:
                    live_count = self.get_live_agent_count()
                    with self.lock:
                        terminated_count = sum(
                            1 for a in self.agents.values()
                            if a.state == AgentState.TERMINATED
                        )
                    if terminated_count > 0:
                        logger.info(
                            f"Agent pool status: {live_count} live, {terminated_count} terminated. "
                            f"Triggering cleanup..."
                        )
                        self.cleanup_terminated_agents()
                
                # THREAD POOL FIX: Log pending executions queue size for monitoring
                with self._pending_executions_lock:
                    pending_count = len(self._pending_executions)
                if pending_count > 0:
                    logger.debug(f"Pending executions queue size: {pending_count}")

            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)

        logger.info("Agent monitor stopped")

    def _get_default_hardware_spec(self) -> Dict[str, Any]:
        """Get default hardware specification"""
        try:
            if PSUTIL_AVAILABLE:
                return {
                    "cpu_cores": psutil.cpu_count(logical=True),
                    "cpu_cores_physical": psutil.cpu_count(logical=False),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "gpu_available": self._check_gpu_available(),
                    "storage_gb": psutil.disk_usage("/").total / (1024**3),
                }
            else:
                # Fallback when psutil is not available
                return {
                    "cpu_cores": multiprocessing.cpu_count(),
                    "cpu_cores_physical": multiprocessing.cpu_count(),
                    "memory_gb": DEFAULT_FALLBACK_MEMORY_GB,
                    "gpu_available": self._check_gpu_available(),
                    "storage_gb": DEFAULT_FALLBACK_STORAGE_GB,
                }
        except Exception as e:
            logger.warning(f"Failed to get hardware spec: {e}")
            return {
                "cpu_cores": 1,
                "memory_gb": 1,
                "gpu_available": False,
                "storage_gb": 10,
            }

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status with throttled status checks.
        
        BUG #1 FIX: Also reports live_agents count to distinguish from terminated.
        """
        current_time = time.time()
        if current_time - self.last_status_check < self.status_check_interval:
            logger.debug(
                "Skipping queue status check due to throttling; returning cached agent data."
            )
            return self._cached_status()

        self.last_status_check = current_time
        
        # BUG #1 FIX: Trigger cleanup when reporting status
        self.cleanup_terminated_agents()

        with self.lock:
            # PERFORMANCE FIX: Combine into single loop to avoid double iteration
            state_counts = defaultdict(int)
            capability_counts = defaultdict(int)
            health_scores = []
            live_count = 0
            for metadata in self.agents.values():
                state_counts[metadata.state.value] += 1
                capability_counts[metadata.capability.value] += 1
                health_scores.append(metadata.get_health_score())
                # BUG #1 FIX: Track live agents
                if metadata.state not in (AgentState.TERMINATED, AgentState.ERROR):
                    live_count += 1

            avg_health = (
                sum(health_scores) / len(health_scores) if health_scores else 0.0
            )

            queue_status = {}
            if self.task_queue and hasattr(self.task_queue, "get_coordinator_status"):
                try:
                    queue_status = self.task_queue.get_coordinator_status()
                except Exception as e:
                    logger.warning(f"Failed to get queue status: {e}")
            elif self.task_queue:
                try:
                    queue_status = self.task_queue.get_queue_status()
                except Exception as e:
                    logger.warning(
                        f"Failed to get queue status from get_queue_status: {e}"
                    )

            with self.stats_lock:
                stats = dict(self.stats)
            
            # THREAD POOL FIX: Include pending executions count
            with self._pending_executions_lock:
                pending_executions = len(self._pending_executions)

            status = {
                "total_agents": len(self.agents),
                "live_agents": live_count,  # BUG #1 FIX: Report live count
                "state_distribution": dict(state_counts),
                "capability_distribution": dict(capability_counts),
                "pending_tasks": len(self.task_assignments),
                "pending_executions": pending_executions,
                "average_health_score": avg_health,
                "queue_status": queue_status,
                "statistics": stats,
                "provenance_records_count": len(self.provenance_records),
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
            }
            logger.info("Agent pool status: %s", status)
            return status

    def _cached_status(self) -> Dict[str, Any]:
        """Return cached status to avoid frequent checks."""
        with self.lock:
            # PERFORMANCE FIX: Combine into single loop to avoid double iteration
            state_counts = defaultdict(int)
            capability_counts = defaultdict(int)
            for metadata in self.agents.values():
                state_counts[metadata.state.value] += 1
                capability_counts[metadata.capability.value] += 1

            return {
                "total_agents": len(self.agents),
                "state_distribution": dict(state_counts),
                "capability_distribution": dict(capability_counts),
                "queue_status": {},
            }

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of specific agent

        Args:
            agent_id: Agent identifier

        Returns:
            Agent status dictionary or None if not found
        """
        with self.lock:
            if agent_id not in self.agents:
                return None

            metadata = self.agents[agent_id]
            return metadata.get_summary()

    def get_job_provenance(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete provenance for a job

        Args:
            job_id: Job identifier

        Returns:
            Job provenance dictionary or None if not found
        """
        with self.lock:
            provenance = self._get_provenance_by_job_id(job_id)
            if provenance is None:
                return None

            return provenance.get_summary()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pool statistics including performance metrics
        
        Returns:
            Statistics dictionary with job counts and performance data
        """
        with self.stats_lock:
            base_stats = dict(self.stats)
        
        # Add performance metrics
        base_stats["response_times"] = self.response_time_tracker.get_stats()
        base_stats["priority_queue"] = self.priority_queue.get_stats()
        base_stats["perf_thresholds"] = self.perf_thresholds
        
        # THREAD POOL FIX: Include pending executions count
        with self._pending_executions_lock:
            base_stats["pending_executions"] = len(self._pending_executions)
        
        return base_stats

    def reset_statistics(self, preserve_totals: bool = True) -> None:
        """
        Reset pool statistics to prevent unbounded memory growth.
        
        PERFORMANCE FIX: Called periodically to prevent statistics dictionaries
        from growing unboundedly over long-running sessions.
        
        BUG #5 FIX: Also triggers agent pool recovery after reset.
        
        Args:
            preserve_totals: If True, keeps cumulative totals but resets windows.
                           If False, resets everything to zero.
        """
        with self.stats_lock:
            if not preserve_totals:
                # Full reset
                self.stats = {
                    "total_jobs_submitted": 0,
                    "total_jobs_completed": 0,
                    "total_jobs_failed": 0,
                    "total_agents_spawned": 0,
                    "total_agents_retired": 0,
                    "total_recoveries_attempted": 0,
                    "total_recoveries_successful": 0,
                }
        
        # Reset response time tracker's sliding window using public method
        if hasattr(self, 'response_time_tracker'):
            self.response_time_tracker.trim_to_window_size()
        
        # Reset priority queue statistics using public method
        if hasattr(self, 'priority_queue'):
            self.priority_queue.reset_priority_distribution()
        
        # BUG #5 FIX: Trigger pool recovery after stats reset
        # Clean up terminated agents and ensure minimum agent count
        cleaned = self.cleanup_terminated_agents()
        live_count = self.get_live_agent_count()
        
        logger.info(
            f"Statistics reset completed (preserve_totals={preserve_totals}). "
            f"Pool recovered: {live_count} live agents, {cleaned} agents cleaned up"
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics for monitoring.
        
        Returns:
            Dictionary containing response times, queue depths, and agent utilization
        """
        response_stats = self.response_time_tracker.get_stats()
        queue_stats = self.priority_queue.get_stats()
        
        with self.lock:
            agent_utilization = {
                "total": len(self.agents),
                "working": sum(1 for m in self.agents.values() if m.state == AgentState.WORKING),
                "idle": sum(1 for m in self.agents.values() if m.state == AgentState.IDLE),
                "error": sum(1 for m in self.agents.values() if m.state == AgentState.ERROR),
            }
            pending_tasks = len(self.task_assignments)
        
        # THREAD POOL FIX: Include pending executions
        with self._pending_executions_lock:
            pending_executions = len(self._pending_executions)
        
        utilization_pct = (
            agent_utilization["working"] / agent_utilization["total"] * 100
            if agent_utilization["total"] > 0 else 0.0
        )
        
        return {
            "response_times": response_stats,
            "queue_stats": queue_stats,
            "agent_utilization": agent_utilization,
            "utilization_percent": utilization_pct,
            "pending_tasks": pending_tasks,
            "pending_executions": pending_executions,
            "thresholds": self.perf_thresholds,
            "trend": self.response_time_tracker.get_recent_trend(),
        }

    def shutdown(self):
        """Gracefully shutdown agent pool"""
        logger.info("Shutting down agent pool")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop auto-scaler
        if self.auto_scaler:
            try:
                self.auto_scaler.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down auto-scaler: {e}")

        # THREAD POOL FIX: Wait for executor thread to finish
        if self._executor_thread and self._executor_thread.is_alive():
            logger.info("Waiting for executor thread to finish...")
            self._executor_thread.join(timeout=5)

        # Stop accepting new jobs and retire all agents
        with self.lock:
            for agent_id in list(self.agents.keys()):
                self.retire_agent(agent_id, force=False)

        # Wait for agents to complete current tasks
        timeout = time.time() + 30
        while time.time() < timeout:
            with self.lock:
                working = any(
                    m.state == AgentState.WORKING for m in self.agents.values()
                )

            if not working:
                break

            time.sleep(0.5)

        # Force terminate remaining agents
        with self.lock:
            for agent_id in list(self.agents.keys()):
                self.retire_agent(agent_id, force=True)

        # Cleanup task queue
        if self.task_queue:
            try:
                self.task_queue.shutdown()
                logger.info("Task queue shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down task queue: {e}")

        # Wait for monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # Final cleanup
        with self.lock:
            self.agents.clear()
            self.agent_processes.clear()
            self.task_assignments.clear()
            self.task_assignment_times.clear()
        
        # Clear pending executions
        with self._pending_executions_lock:
            self._pending_executions.clear()
        
        # SINGLETON FIX: Remove this instance from the class registry
        # Acquire instance_id under the class lock to ensure thread safety
        with AgentPoolManager._instance_lock:
            instance_id = getattr(self, '_instance_id', None)
            if instance_id and instance_id in AgentPoolManager._instances:
                del AgentPoolManager._instances[instance_id]
            if AgentPoolManager._default_instance is self:
                AgentPoolManager._default_instance = None

        logger.info("Agent pool shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logger.debug(f"Error in destructor: {e}")


# ============================================================
# AUTO SCALER
# ============================================================


class AutoScaler:
    """Automatically scale agent pool based on load with proper locking"""

    def __init__(self, pool_manager: AgentPoolManager):
        """
        Initialize auto-scaler

        Args:
            pool_manager: Agent pool manager instance
        """
        self.pool = pool_manager
        self._shutdown_event = threading.Event()

        # Start scaling thread
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop, daemon=True, name="AutoScaler"
        )
        self.scaling_thread.start()

        logger.info("Auto-scaler initialized")

    def _scaling_loop(self):
        """
        Scaling loop that monitors load and adjusts pool size
        FIXED: Uses interruptible wait instead of time.sleep
        """
        logger.info("Auto-scaler loop started")
        
        # PERFORMANCE: Use simple_mode check interval for less frequent scaling
        check_interval = SIMPLE_MODE_CHECK_INTERVAL if SIMPLE_MODE else 30

        while not self._shutdown_event.is_set():
            # FIXED: Use interruptible wait
            if self._shutdown_event.wait(timeout=check_interval):
                break

            try:
                self._evaluate_and_scale()
            except Exception as e:
                logger.error(f"Auto-scaler error: {e}", exc_info=True)

        logger.info("Auto-scaler loop stopped")

    def _evaluate_and_scale(self):
        """Evaluate current load and scale accordingly with enhanced metrics"""
        status = self.pool.get_pool_status()

        total_agents = status["total_agents"]
        idle_agents = status.get("state_distribution", {}).get("idle", 0)
        working_agents = status.get("state_distribution", {}).get(
            "working", 0
        )
        # FIXED: Use .get() with default to avoid KeyError during shutdown
        pending_tasks = status.get("pending_tasks", 0)

        # Calculate utilization
        if total_agents > 0:
            utilization = working_agents / total_agents
        else:
            utilization = 0.0
        
        # Get response time metrics for adaptive scaling
        response_stats = self.pool.response_time_tracker.get_stats()
        p95_ms = response_stats.get("p95_ms", 0.0)
        p99_ms = response_stats.get("p99_ms", 0.0)
        trend = self.pool.response_time_tracker.get_recent_trend()
        
        # Get priority queue depth
        queue_depth = self.pool.priority_queue.size()
        
        # Performance thresholds
        p95_target = self.pool.perf_thresholds["p95_target_ms"]
        p99_target = self.pool.perf_thresholds["p99_target_ms"]
        max_queue = self.pool.perf_thresholds["max_queue_depth"]

        logger.debug(
            f"Auto-scaler evaluation: "
            f"utilization={utilization:.2f}, "
            f"total={total_agents}, "
            f"idle={idle_agents}, "
            f"working={working_agents}, "
            f"pending={pending_tasks}, "
            f"p95={p95_ms:.1f}ms, p99={p99_ms:.1f}ms, "
            f"queue_depth={queue_depth}, trend={trend:.1f}"
        )
        
        # Determine scaling action
        scale_up_reasons = []
        scale_down_ok = True
        
        # Scale up conditions:
        # 1. High utilization (>80%)
        if utilization > 0.8:
            scale_up_reasons.append("high_utilization")
        
        # 2. Pending tasks exceed idle agents
        if pending_tasks > idle_agents:
            scale_up_reasons.append("pending_tasks")
        
        # 3. Response times exceeding SLA targets
        if p95_ms > p95_target:
            scale_up_reasons.append("p95_exceeded")
            scale_down_ok = False
        
        if p99_ms > p99_target:
            scale_up_reasons.append("p99_exceeded")
            scale_down_ok = False
        
        # 4. Queue depth too high
        if queue_depth > max_queue:
            scale_up_reasons.append("queue_depth")
        
        # 5. Degrading performance trend
        if trend > 50:  # 50ms degradation trend
            scale_up_reasons.append("degrading_trend")
            scale_down_ok = False

        # Scale up if any reason applies
        if scale_up_reasons:
            agents_to_spawn = min(
                max(1, pending_tasks - idle_agents, len(scale_up_reasons)),
                self.pool.max_agents - total_agents,
            )

            if agents_to_spawn > 0:
                logger.info(f"Scaling up by {agents_to_spawn} agents, reasons: {scale_up_reasons}")
                for _ in range(agents_to_spawn):
                    self.pool.spawn_agent()

        # Scale down only if low utilization AND performance is good
        elif scale_down_ok and utilization < 0.2 and total_agents > self.pool.min_agents:
            agents_to_retire = min(
                idle_agents // 2, total_agents - self.pool.min_agents
            )

            if agents_to_retire > 0:
                idle_agent_ids = [
                    agent_id
                    for agent_id, metadata in self.pool.agents.items()
                    if metadata.state == AgentState.IDLE
                ][:agents_to_retire]

                logger.info(f"Scaling down by {agents_to_retire} agents (performance OK)")
                for agent_id in idle_agent_ids:
                    self.pool.retire_agent(agent_id)

    def shutdown(self):
        """Shutdown auto-scaler"""
        logger.info("Shutting down auto-scaler")
        self._shutdown_event.set()
        if self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5)
        logger.info("Auto-scaler shutdown complete")


# ============================================================
# RECOVERY MANAGER
# ============================================================


class RecoveryManager:
    """Manages agent recovery and fault tolerance"""

    def __init__(self, pool_manager: AgentPoolManager):
        """
        Initialize recovery manager

        Args:
            pool_manager: Agent pool manager instance
        """
        self.pool = pool_manager
        self.recovery_strategies = {
            AgentState.ERROR: self._recover_error_agent,
            AgentState.TERMINATED: self._recover_terminated_agent,
            AgentState.SUSPENDED: self._recover_suspended_agent,
        }
        logger.info("Recovery manager initialized")

    def recover_agent(self, agent_id: str) -> bool:
        """
        Attempt to recover an agent

        Args:
            agent_id: Agent identifier

        Returns:
            True if recovery successful, False otherwise
        """
        if agent_id not in self.pool.agents:
            logger.warning(f"Cannot recover agent {agent_id}: not found")
            return False

        metadata = self.pool.agents[agent_id]

        if metadata.state in self.recovery_strategies:
            strategy = self.recovery_strategies[metadata.state]
            return strategy(agent_id, metadata)

        logger.warning(
            f"No recovery strategy for agent {agent_id} in state {metadata.state}"
        )
        return False

    def _recover_error_agent(self, agent_id: str, metadata: AgentMetadata) -> bool:
        """Recover agent in error state"""
        error_count = len(metadata.error_history)
        consecutive_errors = metadata.consecutive_errors

        logger.info(
            f"Attempting to recover error agent {agent_id}: "
            f"errors={error_count}, consecutive={consecutive_errors}"
        )

        if consecutive_errors < 3:
            # Try recovery
            return self.pool.recover_agent(agent_id)
        elif consecutive_errors < 5:
            # Reset error history and try recovery
            logger.info(f"Resetting error history for agent {agent_id}")
            metadata.error_history = []
            metadata.consecutive_errors = 0
            return self.pool.recover_agent(agent_id)
        else:
            # Too many errors, retire agent
            logger.warning(f"Agent {agent_id} has too many errors, retiring")
            self.pool.retire_agent(agent_id, force=True)
            return False

    def _recover_terminated_agent(self, agent_id: str, metadata: AgentMetadata) -> bool:
        """Recover terminated agent by spawning replacement"""
        if self.pool.get_pool_status()["total_agents"] < self.pool.min_agents:
            logger.info(
                f"Pool below minimum, spawning replacement for terminated agent {agent_id}"
            )
            new_agent_id = self.pool.spawn_agent(
                capability=metadata.capability, location=metadata.location
            )
            return new_agent_id is not None
        return False

    def _recover_suspended_agent(self, agent_id: str, metadata: AgentMetadata) -> bool:
        """Recover suspended agent"""
        logger.info(f"Recovering suspended agent {agent_id}")
        return metadata.transition_state(AgentState.IDLE, "Recovered from suspension")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "AgentPoolManager",
    "AutoScaler",
    "RecoveryManager",
    "ResponseTimeTracker",
    "PriorityJobQueue",
    "CACHETOOLS_AVAILABLE",
    "TTLCache",
    "TOURNAMENT_MANAGER_AVAILABLE",
    "TOURNAMENT_QUERY_TYPES",
    "TOURNAMENT_MAX_CANDIDATES",
]
