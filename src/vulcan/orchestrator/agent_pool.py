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
    DEFAULT_HEARTBEAT_INTERVAL_S,
    HEARTBEAT_STALENESS_THRESHOLD_S,
)
from .task_queues import TaskQueueInterface, create_task_queue, PriorityJobQueue
from .metrics import ResponseTimeTracker, SystemMetrics

# ============================================================
# EXTRACTED MODULE IMPORTS
# Constants, types, proxy, and worker functions extracted for modularity
# ============================================================
from .agent_pool_types import (
    DEFAULT_FALLBACK_MEMORY_GB,
    DEFAULT_FALLBACK_STORAGE_GB,
    REASONING_IMPORT_PATHS,
    REASONING_TOOL_NAMES,
    TOOL_SELECTION_PRIORITY_ORDER,
    REDIS_KEY_AGENT_POOL_STATS,
    REDIS_KEY_PROVENANCE_COUNT,
    TOURNAMENT_QUERY_TYPES,
    TOURNAMENT_MAX_CANDIDATES,
    TOURNAMENT_DIVERSITY_PENALTY,
    TOURNAMENT_WINNER_PERCENTAGE,
    AGENT_SELECTION_TIMEOUT_SECONDS,
    DEFAULT_DLQ_SIZE,
    STUCK_JOB_WARNING_THRESHOLD,
    STUCK_JOB_CRITICAL_THRESHOLD,
    MIN_REASONING_QUERY_LENGTH,
    LONG_QUERY_REASONING_THRESHOLD,
    WORLD_MODEL_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    CONCLUSION_EXTRACTION_KEYS as _CONCLUSION_EXTRACTION_KEYS,
    MAX_CONCLUSION_EXTRACTION_DEPTH as _MAX_CONCLUSION_EXTRACTION_DEPTH,
    SIMPLE_MODE,
    SIMPLE_MODE_MIN_AGENTS,
    SIMPLE_MODE_MAX_AGENTS,
    SIMPLE_MODE_MAX_PROVENANCE,
    SIMPLE_MODE_CHECK_INTERVAL,
    TTLCache,
    CACHETOOLS_AVAILABLE,
    is_privileged_result as _is_privileged_result,
)
from .agent_pool_proxy import (
    is_main_process,
    _standalone_agent_worker,
    AgentPoolProxy,
)

# ============================================================
# EXTRACTED MODULE IMPORTS (decomposed from AgentPoolManager)
# ============================================================
from . import pool_persistence as _persistence
from . import pool_monitoring as _monitoring
from . import agent_scoring as _scoring
from . import agent_lifecycle_ops as _lifecycle_ops
from . import job_management as _job_mgmt
from . import task_execution_core as _task_exec


# ============================================================
# CIRCULAR IMPORT FIX: Memory imports are now lazy-loaded
# ============================================================
# Import memory systems lazily to avoid circular import with hierarchical.py
# The circular import occurs because:
#   1. src/vulcan/__init__.py imports from .memory (HierarchicalMemory)
#   2. src/vulcan/memory/__init__.py imports from .hierarchical
#   3. hierarchical.py starts loading but HierarchicalMemory class not defined yet
#   4. src/vulcan/__init__.py then imports from .orchestrator
#   5. orchestrator/__init__.py imports from .agent_pool
#   6. agent_pool.py (here) tries to import HierarchicalMemory - but it's not ready!
#
# Solution: Use lazy imports that defer loading until first use.
# ============================================================
WorkingMemory = None  # Lazy-loaded
HierarchicalMemory = None  # Lazy-loaded
MemoryConfig = None  # Lazy-loaded
_memory_import_attempted = False  # Track if we've tried to import


def _lazy_import_memory():
    """
    Lazily import memory components to avoid circular import issues.
    
    CIRCULAR IMPORT FIX: This function is called when memory components are
    actually needed, not at module load time. This prevents the circular import
    that occurs when agent_pool.py imports from src.vulcan.memory.hierarchical
    which in turn depends on modules that import from agent_pool.py.
    
    Returns:
        bool: True if imports succeeded, False otherwise
    """
    global WorkingMemory, HierarchicalMemory, MemoryConfig, _memory_import_attempted
    
    # Only attempt import once
    if _memory_import_attempted:
        return WorkingMemory is not None and HierarchicalMemory is not None and MemoryConfig is not None
    
    _memory_import_attempted = True
    
    # Try multiple import paths for robustness
    import_paths = [
        ('vulcan.memory.specialized', 'vulcan.memory.hierarchical', 'vulcan.memory.base'),
        ('src.vulcan.memory.specialized', 'src.vulcan.memory.hierarchical', 'src.vulcan.memory.base'),
    ]
    
    for specialized_path, hierarchical_path, base_path in import_paths:
        try:
            # Dynamic import using __import__
            specialized_module = __import__(specialized_path, fromlist=['WorkingMemory'])
            hierarchical_module = __import__(hierarchical_path, fromlist=['HierarchicalMemory'])
            base_module = __import__(base_path, fromlist=['MemoryConfig'])
            
            # Update global references
            WorkingMemory = getattr(specialized_module, 'WorkingMemory', None)
            HierarchicalMemory = getattr(hierarchical_module, 'HierarchicalMemory', None)
            MemoryConfig = getattr(base_module, 'MemoryConfig', None)
            
            if WorkingMemory and HierarchicalMemory and MemoryConfig:
                logging.getLogger(__name__).debug(
                    f"Memory components (WorkingMemory, HierarchicalMemory, MemoryConfig) "
                    f"loaded successfully via {specialized_path.rsplit('.', 1)[0]} (lazy import)"
                )
                return True
            
        except ImportError as e:
            logging.getLogger(__name__).debug(
                f"Memory component import failed for path prefix "
                f"'{specialized_path.rsplit('.', 1)[0]}': {e}. Trying next path..."
            )
            continue
    
    # All paths failed
    logging.getLogger(__name__).warning(
        "Memory components not available (all import paths failed). "
        "Provenance tracking will use fallback implementations."
    )
    return False

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
# GRAPHIX PLATFORM DEEP INTEGRATION - ConsensusManager
# ============================================================
# Import ConsensusManager for distributed voting on conflicting agent results
try:
    from src.consensus_manager import ConsensusManager
    CONSENSUS_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from consensus_manager import ConsensusManager
        CONSENSUS_MANAGER_AVAILABLE = True
    except ImportError:
        ConsensusManager = None
        CONSENSUS_MANAGER_AVAILABLE = False
        logging.getLogger(__name__).warning(
            "ConsensusManager not available, distributed voting will be disabled"
        )

# ============================================================
# REASONING INTEGRATION - Wire reasoning engines into task execution
# ============================================================
# CIRCULAR IMPORT FIX: Do NOT import UnifiedReasoner at module level.
# These imports are now done lazily inside methods that need them.
# This prevents the "cannot import name 'UnifiedReasoner' from partially
# initialized module" error that forces placeholder execution.
#
# The lazy import pattern is used in:
# - _get_unified_reasoner() helper method
# - _execute_agent_task() when reasoning is needed
#
# Module-level flags for availability check (these don't cause circular imports)
UnifiedReasoner = None  # Lazy-loaded
ReasoningType = None  # Lazy-loaded
ReasoningResult = None  # Lazy-loaded
UNIFIED_AVAILABLE = False  # Updated by lazy import
create_unified_reasoner = None  # Lazy-loaded
apply_reasoning = None  # Lazy-loaded
get_reasoning_integration = None  # Lazy-loaded
IntegrationReasoningResult = None  # Lazy-loaded
REASONING_AVAILABLE = False  # Updated by lazy import
_reasoning_import_attempted = False  # Track if we've tried to import

# ═══════════════════════════════════════════════════════════════════
# BUG FIX: Import enum conversion helper for reasoning_type safety
# ═══════════════════════════════════════════════════════════════════
# Import at module level to avoid repeated import attempts in hot path.
# This is safe because it doesn't cause circular imports - utils.py
# only depends on reasoning_types.py which is a leaf module.
# ═══════════════════════════════════════════════════════════════════
try:
    # ARCHITECTURE CONSOLIDATION: Import from unified compatibility layer via reasoning
    from vulcan.reasoning import ensure_reasoning_type_enum
    TYPE_CONVERSION_AVAILABLE = True
except ImportError:
    ensure_reasoning_type_enum = None
    TYPE_CONVERSION_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "[AgentPool] Type conversion utility not available - may drop philosophical results"
    )


def _lazy_import_reasoning():
    """
    Lazily import reasoning components to avoid circular import issues.
    
    CIRCULAR IMPORT FIX: This function is called when reasoning is actually
    needed, not at module load time. This prevents the circular import
    that occurs when agent_pool.py imports from src.vulcan.reasoning which
    in turn imports from agent_pool.py.
    
    FIX: Tries multiple import paths to handle different execution contexts:
    - 'vulcan.reasoning' - when running from src/ directory
    - 'src.vulcan.reasoning' - when running from project root
    
    Returns:
        bool: True if imports succeeded, False otherwise
    """
    global UnifiedReasoner, ReasoningType, ReasoningResult, UNIFIED_AVAILABLE
    global create_unified_reasoner, apply_reasoning, get_reasoning_integration
    global IntegrationReasoningResult, REASONING_AVAILABLE, _reasoning_import_attempted
    
    # Only attempt import once
    if _reasoning_import_attempted:
        return REASONING_AVAILABLE
    
    _reasoning_import_attempted = True
    
    # Try multiple import paths for robustness
    # ARCHITECTURE CONSOLIDATION: Integration package has been consolidated into unified
    # All functions now available through vulcan.reasoning via compatibility layer
    import_paths = [
        ('vulcan.reasoning', 'vulcan.reasoning'),  # Both from same place now
        ('src.vulcan.reasoning', 'src.vulcan.reasoning'),  # Both from same place now
    ]
    
    for reasoning_path, integration_path in import_paths:
        try:
            # Dynamic import using __import__
            reasoning_module = __import__(reasoning_path, fromlist=[
                'UnifiedReasoner', 'ReasoningType', 'ReasoningResult',
                'UNIFIED_AVAILABLE', 'create_unified_reasoner'
            ])
            # ARCHITECTURE CONSOLIDATION: Import from same module (compatibility layer)
            integration_module = __import__(integration_path, fromlist=[
                'apply_reasoning', 'get_reasoning_integration', 'ReasoningResult'
            ])
            
            # Update global references
            UnifiedReasoner = getattr(reasoning_module, 'UnifiedReasoner', None)
            ReasoningType = getattr(reasoning_module, 'ReasoningType', None)
            ReasoningResult = getattr(reasoning_module, 'ReasoningResult', None)
            UNIFIED_AVAILABLE = getattr(reasoning_module, 'UNIFIED_AVAILABLE', False)
            create_unified_reasoner = getattr(reasoning_module, 'create_unified_reasoner', None)
            apply_reasoning = getattr(integration_module, 'apply_reasoning', None)
            get_reasoning_integration = getattr(integration_module, 'get_reasoning_integration', None)
            IntegrationReasoningResult = getattr(integration_module, 'ReasoningResult', None)
            REASONING_AVAILABLE = UNIFIED_AVAILABLE
            
            logging.getLogger(__name__).info(
                f"Reasoning integration loaded successfully via {reasoning_path} (lazy import) - reasoning engines will be invoked"
            )
            return True
            
        except ImportError as e:
            logging.getLogger(__name__).debug(
                f"Import path {reasoning_path} failed: {e}. Trying next path..."
            )
            continue
    
    # All paths failed
    logging.getLogger(__name__).warning(
        f"Reasoning integration not available (all import paths failed). Tasks will use placeholder execution."
    )
    REASONING_AVAILABLE = False
    return False

logger = logging.getLogger(__name__)


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
        
        ISSUE 2 FIX: Validates that this is called from the main process only.
        On Windows with 'spawn' multiprocessing, worker processes get fresh module
        copies with empty _instances dict, causing orphaned pool managers.
        
        Args:
            instance_id: Unique identifier for this pool instance (default: "default")
            max_agents: Maximum number of agents in pool
            min_agents: Minimum number of agents to maintain
            task_queue_type: Type of task queue
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            AgentPoolManager singleton instance
            
        Raises:
            RuntimeError: If called from a worker process (not main process)
        """
        # ISSUE 2 FIX: Process validation
        if not is_main_process():
            error_msg = (
                "AgentPoolManager.get_instance() must only be called from the main process. "
                "Worker processes should use AgentPoolProxy for read-only access to pool status. "
                f"Current process: {multiprocessing.current_process().name}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
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
        
        # AGENT POOL CONFIGURATION FIX: Updated min_agents to support reasoning capabilities
        # Previously: min_agents=2 which only allowed 2 agent types (perception, general)
        # Now: min_agents=8 to ensure priority reasoning capabilities get dedicated agents
        # This is critical because ~45% of queries were failing due to capability mismatches
        #
        # Priority reasoning capabilities (from _initialize_agent_pool):
        # 1. PROBABILISTIC - ProbabilisticReasoner
        # 2. SYMBOLIC - SymbolicReasoner
        # 3. PHILOSOPHICAL - World Model (mode='philosophical')
        # 4. MATHEMATICAL - MathematicalComputationTool
        # 5. CAUSAL - CausalReasoner
        # 6. ANALOGICAL - AnalogicalReasoningEngine
        # 7. WORLD_MODEL - WorldModel
        # + 1 GENERAL for fallback
        #
        # Note: Actual reasoning execution uses singletons from reasoning_integration.py
        # so there's no memory overhead from having more agents - each just has capability metadata
        self.max_agents = 15  # Increased from 10 to accommodate more capabilities
        self.min_agents = 8   # Increased from 2 to cover priority reasoning capabilities
        self.task_timeout_seconds = task_timeout_seconds

        # Agent tracking
        self.agents: Dict[str, AgentMetadata] = {}
        self.agent_processes: Dict[str, multiprocessing.Process] = {}

        # MEMORY LEAK FIX: Use specialized memory systems instead of unbounded list
        # PERF FIX Issue #2: Use singleton HierarchicalMemory to avoid re-initialization
        # CIRCULAR IMPORT FIX: Lazy-load memory components to avoid import cycle
        _lazy_import_memory()
        
        # Initialize memory systems (with fallback if lazy import failed)
        if MemoryConfig is not None:
            memory_config = MemoryConfig(max_working_memory=50)
        else:
            # Fallback: use a simple object with the required attribute
            memory_config = type('MemoryConfig', (), {'max_working_memory': 50})()
            logger.warning("[AgentPool] Using fallback MemoryConfig - memory components not available")
        
        if WorkingMemory is not None:
            self.working_memory = WorkingMemory(memory_config)
        else:
            # Fallback: use None and handle gracefully elsewhere
            self.working_memory = None
            logger.warning("[AgentPool] WorkingMemory not available - using fallback")
        
        # Try to use singleton HierarchicalMemory first
        self.long_term_memory = None
        try:
            from vulcan.reasoning.singletons import get_hierarchical_memory
            self.long_term_memory = get_hierarchical_memory(memory_config)
            if self.long_term_memory:
                logger.info("[AgentPool] Using singleton HierarchicalMemory")
        except ImportError:
            pass
        
        # Fallback to direct instantiation if singleton not available
        if self.long_term_memory is None and HierarchicalMemory is not None:
            self.long_term_memory = HierarchicalMemory(memory_config)
            logger.info("[AgentPool] HierarchicalMemory initialized (direct)")
        elif self.long_term_memory is None:
            logger.warning("[AgentPool] HierarchicalMemory not available - using fallback")
        
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
        
        # ISSUE 8 FIX: GC rate limiting to prevent performance degradation
        self._last_gc_time = 0.0
        self._gc_interval_s = 60.0  # Rate limit GC to once per minute
        
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
        
        # ========== PERFORMANCE FIX: Dead Letter Queue for Failed Jobs ==========
        # Jobs that fail repeatedly are moved here instead of being retried infinitely
        self._dead_letter_queue: deque = deque(
            maxlen=self.config.get("dlq_size", DEFAULT_DLQ_SIZE)
        )
        self._dead_letter_lock = threading.Lock()
        # Track retry counts per job
        self._job_retry_counts: Dict[str, int] = {}
        self._max_job_retries = self.config.get("max_job_retries", 3)
        
        # Track stuck jobs (jobs taking too long to complete)
        self._stuck_job_threshold_seconds = self.config.get(
            "stuck_job_threshold", 
            task_timeout_seconds
        )

        # Start monitoring
        self._start_monitor()

        # Initialize auto-scaling and recovery managers (lazy import to avoid circular dependency)
        from .deployment import AutoScaler, RecoveryManager
        self.auto_scaler = AutoScaler(self)
        self.recovery_manager = RecoveryManager(self)
        
        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - ConsensusManager
        # ============================================================
        # Initialize ConsensusManager for distributed voting on conflicting agent results
        # and leader election for task assignment
        self.consensus_manager = None
        if CONSENSUS_MANAGER_AVAILABLE and ConsensusManager is not None:
            try:
                # Use conservative chaos parameters for production reliability
                self.consensus_manager = ConsensusManager(
                    chaos_params={
                        "failure_rate": 0.05,  # 5% simulated failure rate
                        "max_delay": 0.01,      # 10ms max delay
                        "drop_rate": 0.0        # No dropped votes in production
                    },
                    timeout=0.1,  # 100ms voting timeout
                    deadlock_threshold=3,
                    max_retries=5,
                    backend="thread"  # Use thread backend for Windows compatibility
                )
                logger.info(
                    f"✅ ConsensusManager initialized for distributed voting "
                    f"(quorum=0.51, timeout=100ms, backend=thread)"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ConsensusManager: {e}")
                self.consensus_manager = None
        else:
            logger.warning("⚠️ ConsensusManager not available - distributed voting disabled")
        
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
            f"cachetools_available={CACHETOOLS_AVAILABLE}, "
            f"consensus_manager_available={self.consensus_manager is not None}, "
            f"gc_interval_s={self._gc_interval_s}"
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
            
            # Store in working_memory for short-term access (if available)
            if self.working_memory is not None:
                self.working_memory.store(record, relevance=0.8)
            
            # Store in long_term_memory asynchronously for persistence (if available)
            if self.long_term_memory is not None:
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
        """Hydrate Agent Pool state from Redis on startup. Delegates to pool_persistence module."""
        _persistence.hydrate_state_from_redis(self)

    def _persist_state_to_redis(self) -> None:
        """Persist Agent Pool state to Redis. Delegates to pool_persistence module."""
        _persistence.persist_state_to_redis(self)

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
        """Initialize minimum number of agents with diverse capabilities
        
        AGENT POOL CONFIGURATION FIX: Updated to ensure specialized agents are
        spawned for existing reasoning engines. This ensures proper routing of
        queries to the correct reasoning capabilities.
        
        Priority Order for Agent Spawning:
        1. Core reasoning capabilities (probabilistic, symbolic, philosophical, etc.)
        2. General capability for fallback
        3. Basic capabilities (perception, learning, etc.)
        
        This order ensures that reasoning queries are properly routed to specialized
        agents instead of falling back to general agents that cannot handle them.
        """
        logger.info(f"Initializing agent pool with {self.min_agents} agents")
        
        # AGENT POOL FIX: Define priority capabilities for reasoning engines
        # These capabilities map to reasoning engines stored in _AVAILABLE_ENGINES
        # in portfolio_executor.py
        priority_reasoning_capabilities = [
            AgentCapability.PROBABILISTIC,   # ProbabilisticReasoner - WORKING
            AgentCapability.SYMBOLIC,         # SymbolicReasoner - WORKING
            AgentCapability.PHILOSOPHICAL,    # World Model (mode='philosophical') - WORKING
            AgentCapability.MATHEMATICAL,     # MathematicalComputationTool
            AgentCapability.CAUSAL,           # CausalReasoner
            AgentCapability.ANALOGICAL,       # AnalogicalReasoningEngine
            AgentCapability.WORLD_MODEL,      # WorldModel - WORKING
        ]
        
        # Track which capabilities we've spawned
        spawned_capabilities = set()
        agents_spawned = 0
        
        # STEP 1: Spawn agents for priority reasoning capabilities first
        # This ensures at least one agent exists for each working reasoning engine
        for capability in priority_reasoning_capabilities:
            if agents_spawned >= self.min_agents:
                break
            try:
                agent_id = self.spawn_agent(capability)
                if agent_id:
                    spawned_capabilities.add(capability)
                    agents_spawned += 1
                    logger.info(
                        f"[AgentPool] Spawned reasoning agent {agent_id} with "
                        f"capability {capability.value}"
                    )
            except Exception as e:
                logger.error(
                    f"[AgentPool] Failed to spawn {capability.value} agent: {e}"
                )
        
        # STEP 2: Fill remaining slots with general agents
        # General agents serve as fallback for capabilities not yet spawned
        while agents_spawned < self.min_agents:
            try:
                agent_id = self.spawn_agent(AgentCapability.GENERAL)
                if agent_id:
                    spawned_capabilities.add(AgentCapability.GENERAL)
                    agents_spawned += 1
                    logger.debug(
                        f"[AgentPool] Spawned general agent {agent_id}"
                    )
            except Exception as e:
                logger.error(f"[AgentPool] Failed to spawn general agent: {e}")
                break  # Prevent infinite loop on persistent errors
        
        # Log capability distribution for observability
        capability_distribution = {}
        for agent_metadata in self.agents.values():
            cap_name = agent_metadata.capability.value
            capability_distribution[cap_name] = capability_distribution.get(cap_name, 0) + 1
        
        logger.info(
            f"[AgentPool] Agent pool initialized with {len(self.agents)} agents. "
            f"Capability distribution: {capability_distribution}"
        )


    def _start_monitor(self):
        """Start background monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitor_agents, daemon=True, name="AgentPoolMonitor"
            )
            self.monitor_thread.start()
            logger.info("Agent pool monitor started")

    # ========== Note: Agent Pool Death Spiral Prevention ==========

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
        
        Note: This method counts only LIVE agents, excluding terminated
        and error-state agents that should not count toward max_agents capacity.
        
        Returns:
            Number of live (non-terminated, non-error) agents
        """
        with self.lock:
            return self._get_live_agent_count_unsafe()

    def can_spawn_agent(self) -> bool:
        """
        Check if a new agent can be spawned based on LIVE agent count.
        
        Note: Uses live agent count instead of total agent count
        to prevent the death spiral where terminated agents block new spawns.
        
        Returns:
            True if a new agent can be spawned, False otherwise
        """
        return self.get_live_agent_count() < self.max_agents

    def cleanup_terminated_agents(self) -> int:
        """Remove terminated agents from the pool. Delegates to agent_lifecycle_ops module."""
        return _lifecycle_ops.cleanup_terminated_agents(self)

    def _maybe_gc(self) -> None:
        """
        Rate-limited garbage collection to prevent performance degradation.
        
        ISSUE 8 FIX: GC is expensive (10-100ms per call). This method rate-limits
        GC to at most once per minute and only collects youngest generation for
        minimal performance impact.
        
        Industry Standard: Only collect generation 0 (youngest objects) to minimize
        pause time while still preventing memory bloat from short-lived objects.
        Full GC (generations 1-2) should only be triggered by Python's automatic
        thresholds, not on every agent termination.
        """
        current_time = time.time()
        time_since_last_gc = current_time - self._last_gc_time
        
        if time_since_last_gc < self._gc_interval_s:
            logger.debug(
                f"[GC] Skipping GC (last run {time_since_last_gc:.1f}s ago, "
                f"interval={self._gc_interval_s}s)"
            )
            return
        
        # Rate limit passed - perform GC
        logger.debug(
            f"[GC] Triggering generation 0 GC (last run {time_since_last_gc:.1f}s ago)"
        )
        
        try:
            # Only collect generation 0 (youngest) for minimal performance impact
            # Full collections of older generations happen automatically via Python's thresholds
            collected = gc.collect(generation=0)
            self._last_gc_time = current_time
            logger.debug(
                f"[GC] Generation 0 collection completed: {collected} objects collected"
            )
        except Exception as e:
            logger.warning(f"[GC] Garbage collection failed: {e}")
    
    def _ensure_minimum_agents(self) -> int:
        """
        Ensure we have at least min_agents live agents.
        
        Note: This method spawns new agents if the live count
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
        """Spawn a new agent. Delegates to agent_lifecycle_ops module."""
        return _lifecycle_ops.spawn_agent(self, capability, location, hardware_spec)

    def _spawn_local_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn local agent process. Delegates to agent_lifecycle_ops module."""
        _lifecycle_ops._spawn_local_agent(self, agent_id, metadata)

    def _spawn_remote_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn remote agent. Delegates to agent_lifecycle_ops module."""
        _lifecycle_ops._spawn_remote_agent(self, agent_id, metadata)

    def _spawn_cloud_agent(self, agent_id: str, metadata: AgentMetadata):
        """Spawn cloud agent. Delegates to agent_lifecycle_ops module."""
        _lifecycle_ops._spawn_cloud_agent(self, agent_id, metadata)

    def retire_agent(self, agent_id: str, force: bool = False) -> bool:
        """Retire an agent gracefully. Delegates to agent_lifecycle_ops module."""
        return _lifecycle_ops.retire_agent(self, agent_id, force)

    def recover_agent(self, agent_id: str) -> bool:
        """Recover a failed agent. Delegates to agent_lifecycle_ops module."""
        return _lifecycle_ops.recover_agent(self, agent_id)

    def submit_job(
        self,
        graph: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        capability_required: AgentCapability = AgentCapability.GENERAL,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """Submit a job to the agent pool. Delegates to job_management module."""
        return _job_mgmt.submit_job(self, graph, parameters, priority, capability_required, timeout_seconds)

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
        Note: Triggers cleanup and respawn if all agents are terminated
        Note: Fixed early return bug that caused agents to remain idle

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
        at_max_capacity = False  # Note: Track capacity state for early return decision

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

                # Note: Check LIVE agent count using internal method (no re-locking)
                live_count = self._get_live_agent_count_unsafe()
                at_max_capacity = live_count >= self.max_agents
                
                # Note: Log state for debugging agent pool underutilization
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
                    # Note: At max live capacity - try cleanup with cooldown
                    current_time = time.time()
                    if current_time - last_cleanup_time >= cleanup_cooldown:
                        logger.info(
                            f"At max live capacity ({self.max_agents}) with no available agents "
                            f"for capability {capability.value}. Attempting cleanup..."
                        )
            
            # Note: Attempt cleanup outside lock to avoid deadlock (with cooldown)
            current_time = time.time()
            if current_time - last_cleanup_time >= cleanup_cooldown:
                last_cleanup_time = current_time
                cleaned = self.cleanup_terminated_agents()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} terminated agents, retrying assignment")
                    continue  # Retry immediately after cleanup
                # Note: Only return early if we're ACTUALLY at max capacity
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
        """Assign an available agent. Must be called with lock held. Delegates to agent_scoring module."""
        return _scoring.assign_agent(self, capability)

    def calculate_agent_score(self, agent_id: str) -> float:
        """Calculate composite score for agent selection. Delegates to agent_scoring module."""
        return _scoring.calculate_agent_score(self, agent_id)

    def get_agents_by_capability(
        self,
        capabilities: List[str],
        max_agents: int = TOURNAMENT_MAX_CANDIDATES
    ) -> List[str]:
        """Get available agents matching capabilities. Delegates to agent_scoring module."""
        return _scoring.get_agents_by_capability(self, capabilities, max_agents)

    def get_capability_distribution(self) -> Dict[str, int]:
        """Get capability distribution in the agent pool. Delegates to agent_scoring module."""
        return _scoring.get_capability_distribution(self)

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
        """Execute task on agent. Delegates to task_execution_core module."""
        return _task_exec.execute_agent_task(self, agent_id, task, metadata)

    # Keep original docstring as reference comment for the delegation stub above:
    # Executes task with ACTUAL reasoning engine invocation via task_execution_core.
    # For reasoning tasks: invokes engines via ReasoningIntegration
    # For non-reasoning tasks: falls back to graph-based execution
    _execute_agent_task_REPLACED = True  # Marker for refactoring audit
    def _extract_conclusion_from_dict(
        self,
        data_dict: Dict[str, Any],
        _depth: int = 0
    ) -> Optional[Any]:
        """Extract conclusion from dict. Delegates to task_execution_core module."""
        return _task_exec.extract_conclusion_from_dict(data_dict, _depth)

    def _is_valid_conclusion(self, conclusion: Any) -> bool:
        """Check if conclusion is valid. Delegates to task_execution_core module."""
        return _task_exec.is_valid_conclusion(conclusion)

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
    # Dead Letter Queue and Stuck Job Handling (delegated to job_management module)
    # ============================================================

    def _move_to_dead_letter_queue(self, task_id: str, reason: str, error: Optional[Exception] = None) -> None:
        """Move a job to the dead letter queue. Delegates to job_management module."""
        _job_mgmt.move_to_dead_letter_queue(self, task_id, reason, error)

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Get all jobs in the dead letter queue. Delegates to job_management module."""
        return _job_mgmt.get_dead_letter_queue(self)

    def clear_dead_letter_queue(self) -> int:
        """Clear the dead letter queue. Delegates to job_management module."""
        return _job_mgmt.clear_dead_letter_queue(self)

    def retry_dead_letter_job(self, task_id: str) -> bool:
        """Retry a job from the dead letter queue. Delegates to job_management module."""
        return _job_mgmt.retry_dead_letter_job(self, task_id)

    def get_stuck_jobs(self) -> List[Dict[str, Any]]:
        """Get list of stuck jobs. Delegates to job_management module."""
        return _job_mgmt.get_stuck_jobs(self)

    def process_stuck_jobs(self) -> Dict[str, Any]:
        """Process stuck jobs. Delegates to job_management module."""
        return _job_mgmt.process_stuck_jobs(self)

    def reassign_job(self, task_id: str, force: bool = False) -> Optional[str]:
        """Reassign a stuck/failed job. Delegates to job_management module."""
        return _job_mgmt.reassign_job(self, task_id, force)

    def recover_stuck_job(self, task_id: str) -> bool:
        """Recover a stuck job. Delegates to job_management module."""
        return _job_mgmt.recover_stuck_job(self, task_id)

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
        # Note: Map "general" to SYMBOLIC instead of UNKNOWN to leverage the LanguageReasoner
        # for general language/text queries. This prevents the 10% confidence issue.
        #
        # Note: Added "_task" suffix variants for task types coming from query_router.py
        # The router creates tasks with types like "mathematical_task", "philosophical_task", etc.
        # Without these mappings, the system falls back to SYMBOLIC for all math/philosophical queries.
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
            "general": ReasoningType.SYMBOLIC,  # Note: General queries -> SYMBOLIC
            "text": ReasoningType.SYMBOLIC,  # Text tasks -> SYMBOLIC
            "mathematical": ReasoningType.MATHEMATICAL,  # Mathematical tasks
            "math": ReasoningType.MATHEMATICAL,  # Math shorthand
            "philosophical": ReasoningType.PHILOSOPHICAL,  # Note: Philosophical/ethical tasks
            "ethical": ReasoningType.PHILOSOPHICAL,  # Note: Ethical queries
            "deontic": ReasoningType.PHILOSOPHICAL,  # Note: Deontic logic queries
            # Note: Add "_task" suffix variants for task types from query_router.py
            # The router systematically generates task types using `f'{query_type.value}_task'` pattern.
            # Explicit task types (mathematical_task, philosophical_task) are created in fast-path handlers.
            # These mappings prevent "Unrecognized task type" warnings and incorrect SYMBOLIC fallback.
            "mathematical_task": ReasoningType.MATHEMATICAL,
            "philosophical_task": ReasoningType.PHILOSOPHICAL,
            "probabilistic_task": ReasoningType.PROBABILISTIC,
            "causal_task": ReasoningType.CAUSAL,
            "analogical_task": ReasoningType.ANALOGICAL,
            "symbolic_task": ReasoningType.SYMBOLIC,
            "reasoning_task": ReasoningType.HYBRID,
            "general_task": ReasoningType.SYMBOLIC,
            # Note: execution_task uses HYBRID because execution often involves multi-step
            # planning and may include mathematical operations. HYBRID reasoning combines
            # multiple reasoning types adaptively (see _execute_task in unified_reasoning.py).
            "execution_task": ReasoningType.HYBRID,
            "perception_task": ReasoningType.ANALOGICAL,  # Perception uses pattern matching
            "planning_task": ReasoningType.HYBRID,  # Planning uses hybrid reasoning
            "learning_task": ReasoningType.HYBRID,  # Learning uses hybrid
            # Note: Add self-introspection and meta-reasoning task types
            # These task types are generated for queries about the system's own state/objectives
            # and should route to HYBRID reasoning which can access world_model/meta-reasoning
            "self_introspection_task": ReasoningType.HYBRID,  # Self-introspection -> world_model/meta-reasoning
            "meta_reasoning_task": ReasoningType.HYBRID,  # Meta-reasoning about objectives -> HYBRID
            "introspection_task": ReasoningType.HYBRID,  # Introspection shorthand -> HYBRID
            # Pattern 9 FIX: Add cryptographic task types
            # The cryptographic tool exists in available_tools but 'cryptographic_task' was not mapped
            # Cryptographic operations use deterministic SYMBOLIC reasoning (not probabilistic)
            "cryptographic": ReasoningType.SYMBOLIC,
            "cryptographic_task": ReasoningType.SYMBOLIC,
            "crypto": ReasoningType.SYMBOLIC,  # Shorthand
            # Bug #3 FIX: Add creative task type mappings
            # Creative queries like "write a poem" should route to PHILOSOPHICAL reasoning
            # (which can handle creative/imaginative content) instead of falling back to SYMBOLIC
            # which produces literal/technical responses inappropriate for creative requests.
            "creative": ReasoningType.PHILOSOPHICAL,  # Creative -> philosophical reasoning
            "creative_task": ReasoningType.PHILOSOPHICAL,  # Creative tasks use philosophical
            "poetry": ReasoningType.PHILOSOPHICAL,  # Poetry requests
            "poetry_task": ReasoningType.PHILOSOPHICAL,  # Poetry tasks
            "writing": ReasoningType.PHILOSOPHICAL,  # Writing tasks
            "writing_task": ReasoningType.PHILOSOPHICAL,  # Writing tasks
            "artistic": ReasoningType.PHILOSOPHICAL,  # Artistic content
            "artistic_task": ReasoningType.PHILOSOPHICAL,  # Artistic tasks
            "imaginative": ReasoningType.PHILOSOPHICAL,  # Imaginative content
            "imaginative_task": ReasoningType.PHILOSOPHICAL,  # Imaginative tasks
        }
        
        # Note: Default to SYMBOLIC instead of UNKNOWN for unrecognized task types
        # SYMBOLIC reasoning can handle most general queries
        result = task_to_reasoning_map.get(task_type.lower())
        if result is None:
            # Log warning for unrecognized task types to help identify missing mappings
            logger.warning(
                f"[AgentPool] Unrecognized task type '{task_type}' - "
                f"falling back to SYMBOLIC. Consider adding explicit mapping."
            )
            return ReasoningType.SYMBOLIC
        return result
    
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
                # Note: Also trigger cleanup and ensure minimum agents
                if monitor_iterations % STATS_RESET_INTERVAL == 0:
                    logger.info(f"Performing periodic statistics reset (iteration {monitor_iterations})")
                    self.reset_statistics(preserve_totals=True)
                    # Note: Cleanup terminated agents and ensure minimum
                    self.cleanup_terminated_agents()
                
                # Note: Check for terminated agent cleanup every 3 iterations (~30 seconds)
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
                
                # PERFORMANCE FIX: Process stuck jobs every 6 iterations (~60 seconds)
                # This catches jobs that are taking too long before they fully timeout
                if monitor_iterations % 6 == 0:
                    self.process_stuck_jobs()

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
        """Get pool status. Delegates to pool_monitoring module."""
        return _monitoring.get_pool_status(self)

    def _cached_status(self) -> Dict[str, Any]:
        """Return cached status. Delegates to pool_monitoring module."""
        return _monitoring.get_cached_status(self)

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific agent. Delegates to pool_monitoring module."""
        return _monitoring.get_agent_status(self, agent_id)

    def get_job_provenance(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get complete provenance for a job. Delegates to pool_monitoring module."""
        return _monitoring.get_job_provenance(self, job_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics. Delegates to pool_monitoring module."""
        return _monitoring.get_statistics(self)

    def reset_statistics(self, preserve_totals: bool = True) -> None:
        """Reset pool statistics. Delegates to pool_monitoring module."""
        _monitoring.reset_statistics(self, preserve_totals)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics. Delegates to pool_monitoring module."""
        return _monitoring.get_performance_stats(self)

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


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "AgentPoolManager",
    "AgentPoolProxy",
    "is_main_process",
    "CACHETOOLS_AVAILABLE",
    "TTLCache",
    "TOURNAMENT_MANAGER_AVAILABLE",
    "TOURNAMENT_QUERY_TYPES",
    "TOURNAMENT_MAX_CANDIDATES",
    "DEFAULT_HEARTBEAT_INTERVAL_S",
    "HEARTBEAT_STALENESS_THRESHOLD_S",
]
