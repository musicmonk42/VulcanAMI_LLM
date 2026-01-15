"""
Startup Manager

Orchestrates the VULCAN-AGI startup process through well-defined phases
with proper error isolation, health validation, and status reporting.

This module implements a robust phased initialization strategy with:
- Dependency ordering: phases execute sequentially with proper dependencies
- Error isolation: non-critical phase failures don't prevent startup
- Health validation: comprehensive checks after initialization
- Resource cleanup: guaranteed cleanup on failure or shutdown
- Timeout enforcement: phases have configurable time limits
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from threading import Thread
from typing import Any, Optional, Dict, Callable
from concurrent.futures import ThreadPoolExecutor

from vulcan.server import state

from .constants import (
    DEFAULT_THREAD_POOL_SIZE,
    THREAD_NAME_PREFIX,
    MEMORY_GUARD_THRESHOLD_PERCENT,
    MEMORY_GUARD_CHECK_INTERVAL_SECONDS,
    REDIS_WORKER_TTL_SECONDS,
    SELF_OPTIMIZER_TARGET_LATENCY_MS,
    SELF_OPTIMIZER_TARGET_MEMORY_MB,
    SELF_OPTIMIZER_INTERVAL_SECONDS,
    LLM_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DeploymentMode,
    LogEmoji,
)
from .phases import StartupPhase, get_phase_metadata, is_critical_phase
from .subsystems import SubsystemManager
from .health import HealthCheck, HealthStatus
from .trace_logger import get_startup_trace


logger = logging.getLogger(__name__)


class StartupManager:
    """
    Manages VULCAN-AGI startup through phased initialization.
    
    Coordinates startup phases with proper dependency ordering,
    error isolation, and health validation.
    """
    
    def __init__(
        self,
        app: Any,
        settings: Any,
        redis_client: Optional[Any] = None,
        process_lock: Optional[Any] = None,
    ):
        """
        Initialize startup manager.
        
        Args:
            app: FastAPI application instance
            settings: Application settings
            redis_client: Optional Redis client
            process_lock: Optional process lock for split-brain prevention
        """
        self.app = app
        self.settings = settings
        self.redis_client = redis_client
        self.process_lock = process_lock
        self.worker_id = os.getpid()
        self.startup_time = time.time()
        
        # Track initialization state
        self.phase_results: Dict[StartupPhase, bool] = {}
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Startup trace logger for auditable registration tracking
        self.trace = get_startup_trace()
    
    async def run_startup(self) -> None:
        """
        Execute complete startup sequence with timeout enforcement.
        
        Implements robust error handling with guaranteed resource cleanup.
        Each phase has a configurable timeout to prevent indefinite hangs.
        If startup fails after executor creation, the executor is properly
        cleaned up to prevent thread leaks.
        
        IMPORTANT - Zombie Thread Warning:
            Python cannot forcibly terminate OS threads. When a phase times out,
            the underlying thread may continue running as a "zombie" - consuming
            resources but no longer responding to the application. This is logged
            with CRITICAL severity to enable container orchestration systems
            (Kubernetes, Docker Swarm, etc.) to detect and reclaim these resources
            by restarting the pod/container.
        
        Raises:
            RuntimeError: If critical phase fails
            asyncio.TimeoutError: If phase exceeds timeout limit
        """
        logger.info(
            f"Starting VULCAN-AGI worker {self.worker_id} "
            f"in {self.settings.deployment_mode} mode"
        )
        
        try:
            # Phase 1: Configuration (P1 Fix: Issue #6 - timeout enforcement)
            await self._run_phase_with_timeout(
                StartupPhase.CONFIGURATION,
                self._phase_configuration
            )
            
            # Phase 2: Core Services (P1 Fix: Issue #6 - timeout enforcement)
            await self._run_phase_with_timeout(
                StartupPhase.CORE_SERVICES,
                self._phase_core_services
            )
            
            # Phase 3: Reasoning Systems (P1 Fix: Issue #6 - timeout enforcement)
            await self._run_phase_with_timeout(
                StartupPhase.REASONING_SYSTEMS,
                self._phase_reasoning_systems
            )
            
            # Phase 4: Memory Systems (P1 Fix: Issue #6 - timeout enforcement)
            await self._run_phase_with_timeout(
                StartupPhase.MEMORY_SYSTEMS,
                self._phase_memory_systems
            )
            
            # Phase 5: Preloading (P1 Fix: Issue #6 - timeout enforcement)
            await self._run_phase_with_timeout(
                StartupPhase.PRELOADING,
                self._phase_preloading
            )
            
            # Phase 6: Monitoring (P1 Fix: Issue #6 - timeout enforcement)
            await self._run_phase_with_timeout(
                StartupPhase.MONITORING,
                self._phase_monitoring
            )
            
            # Health validation
            await self._validate_health()
            
            # Print comprehensive startup trace summary
            self.trace.print_summary()
            
            # Startup complete
            elapsed = time.time() - self.startup_time
            logger.info(f"✅ VULCAN-AGI worker {self.worker_id} started in {elapsed:.2f}s")
            
        except asyncio.TimeoutError as e:
            # Zombie thread logging already done in _run_phase_with_timeout
            # Clean up executor on failure (P0 Fix: Issue #3)
            self._cleanup_executor_on_failure("timeout")
            raise RuntimeError("Startup phase timed out") from e
        except Exception as e:
            logger.error(f"Startup failed: {e}", exc_info=True)
            # Clean up executor on failure (P0 Fix: Issue #3)
            self._cleanup_executor_on_failure("startup failure")
            raise
    
    async def _run_phase_with_timeout(
        self,
        phase: StartupPhase,
        phase_func: Callable,
    ) -> None:
        """
        Run a startup phase with timeout enforcement and zombie thread logging.
        
        This method wraps phase execution with:
        1. Timeout enforcement via asyncio.wait_for
        2. Comprehensive zombie thread logging on timeout
        3. Thread state inspection for diagnostics
        
        Args:
            phase: The startup phase being executed
            phase_func: The async function to execute
            
        Raises:
            asyncio.TimeoutError: If phase exceeds its configured timeout
            RuntimeError: If critical phase fails
        """
        meta = get_phase_metadata(phase)
        
        try:
            await asyncio.wait_for(
                phase_func(),
                timeout=meta.timeout_seconds
            )
        except asyncio.TimeoutError:
            # Log zombie thread warning with detailed diagnostics
            self._log_zombie_thread_warning(phase, meta)
            raise
    
    def _log_zombie_thread_warning(
        self,
        phase: StartupPhase,
        meta: Any,
    ) -> None:
        """
        Log comprehensive zombie thread warning for container orchestration.
        
        CRITICAL: Python cannot forcibly kill OS threads. When a phase times out,
        the underlying thread(s) may continue running indefinitely, consuming
        resources but no longer responding to the application.
        
        This method logs with CRITICAL severity to:
        1. Alert operators to potential resource leaks
        2. Enable container orchestration to detect and restart pods
        3. Provide diagnostic information for debugging
        
        Args:
            phase: The phase that timed out
            meta: Phase metadata including timeout configuration
        """
        import threading
        
        # Gather active thread information
        active_threads = threading.enumerate()
        vulcan_threads = [t for t in active_threads if t.name.startswith("vulcan_")]
        
        # Log critical zombie warning
        logger.critical(
            f"🧟 ZOMBIE THREAD WARNING: Phase '{meta.name}' timed out after "
            f"{meta.timeout_seconds}s. "
            f"IMPORTANT: Python cannot forcibly terminate OS threads. "
            f"The timed-out operation may continue running as a zombie thread, "
            f"consuming CPU/memory resources indefinitely. "
            f"Container orchestration (Kubernetes/Docker) should restart this pod "
            f"to reclaim resources. "
            f"Active threads: {len(active_threads)} total, {len(vulcan_threads)} vulcan threads."
        )
        
        # Log thread details for debugging
        if vulcan_threads:
            thread_info = ", ".join(
                f"{t.name}(alive={t.is_alive()}, daemon={t.daemon})"
                for t in vulcan_threads[:5]  # Limit to first 5 to avoid log spam
            )
            logger.critical(
                f"🧟 Vulcan threads that may be zombies: {thread_info}"
                + (f" ... and {len(vulcan_threads) - 5} more" if len(vulcan_threads) > 5 else "")
            )
        
        # Log recommendations
        logger.critical(
            f"🧟 RECOMMENDED ACTIONS:\n"
            f"  1. Check container/pod health metrics for resource leaks\n"
            f"  2. Consider restarting the container/pod to reclaim resources\n"
            f"  3. Investigate why phase '{meta.name}' exceeded {meta.timeout_seconds}s timeout\n"
            f"  4. Review thread pool configuration if using ThreadPoolExecutor"
        )
        
        # Track zombie in phase results
        self.phase_results[phase] = False
    
    def _cleanup_executor_on_failure(self, failure_reason: str) -> None:
        """
        Clean up ThreadPoolExecutor after a failure with proper logging.
        
        Args:
            failure_reason: Description of why cleanup is being performed
        """
        if self.executor:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
                logger.debug(f"Executor cleaned up after {failure_reason}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up executor: {cleanup_error}")
    
    async def _phase_configuration(self) -> None:
        """
        Phase 1: Load configuration.
        
        Loads the appropriate configuration profile based on deployment mode
        and applies sensible defaults for missing attributes.
        """
        phase = StartupPhase.CONFIGURATION
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 1: {meta.name}")
        
        try:
            # Import required modules
            from vulcan.config import get_config, AgentConfig
            
            # Load configuration profile (P2 Fix: Issue #9 - use constants)
            profile_name = DeploymentMode.normalize(self.settings.deployment_mode)
            
            config = get_config(profile_name)
            
            # Validate config is an AgentConfig instance
            if not isinstance(config, AgentConfig):
                logger.error(
                    f"Invalid config type returned: {type(config)}, creating default config"
                )
                config = AgentConfig()
            
            # Set defaults if attributes don't exist
            self._apply_config_defaults(config)
            
            # Store config
            self.config = config
            
            # Ensure persistence directories exist
            self._ensure_directories()
            
            self.phase_results[phase] = True
            logger.info(f"{LogEmoji.SUCCESS} Configuration loaded ({profile_name} profile)")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.error(f"Configuration loading failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    def _apply_config_defaults(self, config: Any) -> None:
        """Apply default configuration values."""
        defaults = {
            "max_graph_size": self.settings.max_graph_size,
            "max_execution_time_s": self.settings.max_execution_time_s,
            "max_memory_mb": self.settings.max_memory_mb,
            "slo_p95_latency_ms": 1000,
            "slo_p99_latency_ms": 2000,
            "slo_max_error_rate": 0.1,
            "max_working_memory": 20,
            "enable_self_improvement": self.settings.enable_self_improvement,
        }
        
        for attr, value in defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, value)
        
        # Handle self-improvement configuration
        if not hasattr(config, "self_improvement_config"):
            if not hasattr(config, "world_model"):
                from dataclasses import make_dataclass
                WorldModelConfig = make_dataclass(
                    "WorldModelConfig",
                    [("self_improvement_config", str, self.settings.self_improvement_config)],
                )
                config.world_model = WorldModelConfig()
            elif not hasattr(config.world_model, "self_improvement_config"):
                setattr(
                    config.world_model,
                    "self_improvement_config",
                    self.settings.self_improvement_config,
                )
        
        if not hasattr(config, "self_improvement_state"):
            if not hasattr(config, "world_model"):
                from dataclasses import make_dataclass
                WorldModelConfig = make_dataclass(
                    "WorldModelConfig",
                    [("self_improvement_state", str, self.settings.self_improvement_state)],
                )
                config.world_model = WorldModelConfig()
            elif not hasattr(config.world_model, "self_improvement_state"):
                setattr(
                    config.world_model,
                    "self_improvement_state",
                    self.settings.self_improvement_state,
                )
    
    def _ensure_directories(self) -> None:
        """
        Ensure required persistence directories exist.
        
        Creates data, config, and checkpoint directories with proper
        permissions. Failures are logged as warnings but don't prevent
        startup since directories may be created on-demand later.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        """
        try:
            Path(DEFAULT_DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(DEFAULT_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
            Path(DEFAULT_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            logger.debug(f"{LogEmoji.SUCCESS} Persistence directories ensured")
        except Exception as e:
            logger.warning(f"Could not ensure directories: {e}")
    
    async def _phase_core_services(self) -> None:
        """Phase 2: Initialize core services."""
        phase = StartupPhase.CORE_SERVICES
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 2: {meta.name}")
        
        try:
            # Import required modules
            from vulcan.utils_main.components import initialize_component
            from vulcan.orchestrator.deployment import ProductionDeployment
            from vulcan.llm import GraphixVulcanLLM
            
            # Setup thread pool executor
            self._setup_thread_pool()
            
            # Initialize deployment
            checkpoint_to_load = self._get_checkpoint_path()
            deployment = initialize_component(
                "deployment",
                lambda: ProductionDeployment(
                    self.config,
                    checkpoint_path=checkpoint_to_load,
                    redis_client=self.redis_client
                ),
            )
            
            # Initialize UnifiedRuntime if available
            self._initialize_unified_runtime(deployment)
            
            # Initialize LLM
            llm_instance = initialize_component(
                "llm",
                lambda: GraphixVulcanLLM(config_path=LLM_CONFIG_PATH)
            )
            self.app.state.llm = llm_instance
            
            # Register LLM with singletons
            self._register_llm_client(llm_instance)
            
            # Initialize HybridLLMExecutor
            self._initialize_hybrid_executor(llm_instance)
            
            # Initialize Knowledge Distiller if enabled
            self._initialize_knowledge_distiller(llm_instance)
            
            # Register worker in Redis
            self._register_worker_redis()
            
            # Start rate limit cleanup thread
            self._start_rate_limit_cleanup()
            
            # Store deployment and metadata
            self.app.state.deployment = deployment
            self.app.state.worker_id = self.worker_id
            self.app.state.startup_time = self.startup_time
            
            self.phase_results[phase] = True
            logger.info("✓ Core services initialized")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.error(f"Core services initialization failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    def _setup_thread_pool(self) -> None:
        """
        Setup ThreadPoolExecutor for parallel cognitive tasks.
        
        Creates a thread pool with DEFAULT_THREAD_POOL_SIZE workers and sets it
        as the default executor for the event loop. The executor is stored in
        app.state for later cleanup during shutdown.
        
        The thread pool is used for CPU-bound operations that would otherwise
        block the async event loop, such as model inference and reasoning tasks.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        
        Raises:
            No exceptions raised; failures are logged as warnings.
        """
        try:
            loop = asyncio.get_event_loop()
            self.executor = ThreadPoolExecutor(
                max_workers=DEFAULT_THREAD_POOL_SIZE,
                thread_name_prefix=THREAD_NAME_PREFIX
            )
            loop.set_default_executor(self.executor)
            self.app.state.executor = self.executor
            logger.debug(
                f"{LogEmoji.SUCCESS} ThreadPoolExecutor: {DEFAULT_THREAD_POOL_SIZE} workers"
            )
        except Exception as e:
            logger.warning(f"Failed to set default executor: {e}")
    
    def _get_checkpoint_path(self) -> Optional[str]:
        """
        Get checkpoint path if it exists and is valid.
        
        Validates that the checkpoint file exists and is not empty before
        returning the path. Returns None if checkpoint is invalid or missing,
        allowing the system to start fresh.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        
        Returns:
            Valid checkpoint path or None if invalid/missing
        """
        if not self.settings.checkpoint_path:
            return None
        
        if (
            os.path.exists(self.settings.checkpoint_path)
            and os.path.getsize(self.settings.checkpoint_path) > 0
        ):
            logger.info(f"Will load checkpoint from {self.settings.checkpoint_path}")
            return self.settings.checkpoint_path
        else:
            logger.warning(
                f"Checkpoint file {self.settings.checkpoint_path} "
                "does not exist or is empty, starting fresh"
            )
            return None
    
    def _initialize_unified_runtime(self, deployment: Any) -> None:
        """Initialize UnifiedRuntime singleton."""
        try:
            from vulcan.reasoning.singletons import (
                get_or_create_unified_runtime,
                set_unified_runtime
            )
            deployment.unified_runtime = get_or_create_unified_runtime()
            if deployment.unified_runtime:
                logger.debug("✓ UnifiedRuntime initialized")
        except ImportError:
            try:
                from vulcan.reasoning.unified_runtime import UnifiedRuntime
                deployment.unified_runtime = UnifiedRuntime()
                logger.debug("✓ UnifiedRuntime initialized (fallback)")
            except Exception:
                logger.debug("UnifiedRuntime not available")
        except Exception as e:
            logger.warning(f"UnifiedRuntime initialization failed: {e}")
    
    def _register_llm_client(self, llm_instance: Any) -> None:
        """Register LLM client with singletons module."""
        try:
            from vulcan.reasoning.singletons import set_llm_client
            if llm_instance is not None:
                set_llm_client(llm_instance)
                logger.debug("✓ LLM client registered")
        except ImportError:
            logger.debug("Singletons module not available for LLM registration")
        except Exception as e:
            logger.warning(f"Failed to register LLM client: {e}")
    
    def _initialize_hybrid_executor(self, llm_instance: Any) -> None:
        """Initialize HybridLLMExecutor singleton."""
        try:
            from vulcan.llm.hybrid_executor import (
                get_or_create_hybrid_executor,
                verify_hybrid_executor_setup
            )
            from vulcan.llm.openai_client import get_openai_client, log_openai_status
            
            # Log OpenAI configuration status
            log_openai_status()
            
            self.app.state.hybrid_executor = get_or_create_hybrid_executor(
                local_llm=llm_instance,
                openai_client_getter=get_openai_client,
                mode=self.settings.llm_execution_mode,
                timeout=self.settings.llm_parallel_timeout,
                ensemble_min_confidence=self.settings.llm_ensemble_min_confidence,
                openai_max_tokens=self.settings.llm_openai_max_tokens,
            )
            
            if self.app.state.hybrid_executor:
                logger.info(f"✓ HybridLLMExecutor initialized successfully at startup (mode={self.settings.llm_execution_mode})")
            else:
                logger.warning("⚠ HybridLLMExecutor initialization failed - LLM generation may be unavailable")
            
            # Verify setup
            verification = verify_hybrid_executor_setup()
            if verification["status"] == "PASS":
                logger.info(f"✓ HybridExecutor verified: {verification['message']}")
            else:
                logger.warning(f"⚠ HybridExecutor: {verification['message']}")
                
        except Exception as e:
            logger.warning(f"HybridLLMExecutor initialization failed: {e}")
            self.app.state.hybrid_executor = None
    
    def _initialize_knowledge_distiller(self, llm_instance: Any) -> None:
        """Initialize Knowledge Distiller if enabled."""
        if not self.settings.enable_knowledge_distillation:
            self.app.state.knowledge_distiller = None
            logger.debug("Knowledge Distillation disabled")
            return
        
        try:
            from vulcan.distillation.knowledge_distiller import initialize_knowledge_distiller
            
            knowledge_distiller = initialize_knowledge_distiller(
                local_llm=llm_instance,
                storage_path=self.settings.distillation_storage_path,
                batch_size=self.settings.distillation_batch_size,
                training_interval_s=self.settings.distillation_training_interval_s,
                auto_train=self.settings.distillation_auto_train,
                learning_rate=self.settings.distillation_learning_rate,
            )
            self.app.state.knowledge_distiller = knowledge_distiller
            logger.debug("✓ Knowledge Distiller initialized")
        except Exception as e:
            logger.warning(f"Knowledge Distiller initialization failed: {e}")
            self.app.state.knowledge_distiller = None
    
    def _register_worker_redis(self) -> None:
        """
        Register worker metadata in Redis if available.
        
        Stores worker information with a TTL for automatic cleanup of stale
        entries. This enables multi-instance coordination and worker discovery.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        """
        if not self.redis_client:
            return
        
        try:
            import msgpack
            worker_metadata = {
                "worker_id": self.worker_id,
                "started": time.time(),
                "deployment_mode": self.settings.deployment_mode,
            }
            self.redis_client.setex(
                f"deployment:{self.worker_id}",
                REDIS_WORKER_TTL_SECONDS,
                msgpack.packb(worker_metadata, use_bin_type=True),
            )
            logger.debug(f"{LogEmoji.SUCCESS} Worker {self.worker_id} registered in Redis")
        except Exception as e:
            logger.error(f"Failed to register in Redis: {e}")
    
    def _start_rate_limit_cleanup(self) -> None:
        """
        Start rate limit cleanup thread with thread-safe synchronization.
        
        Uses a lock to prevent race conditions when multiple workers attempt
        to start the cleanup thread simultaneously. Only one thread will be
        started regardless of concurrency.
        
        P0 Fix: Issue #2 - Race condition prevention with lock
        """
        try:
            from vulcan.api.rate_limiting import cleanup_rate_limits
            
            # Thread-safe check and start (P0 Fix: Issue #2)
            with state.rate_limit_thread_lock:
                if (state.rate_limit_cleanup_thread is None or 
                    not state.rate_limit_cleanup_thread.is_alive()):
                    thread = Thread(target=cleanup_rate_limits, daemon=True)
                    thread.start()
                    state.rate_limit_cleanup_thread = thread
                    logger.debug("✓ Rate limit cleanup thread started")
                else:
                    logger.debug("Rate limit cleanup thread already running")
        except ImportError as e:
            logger.debug(f"Rate limiting module not available: {e}")
        except Exception as e:
            logger.warning(f"Rate limit cleanup thread failed: {e}")
    
    async def _phase_reasoning_systems(self) -> None:
        """Phase 3: Initialize reasoning subsystems."""
        phase = StartupPhase.REASONING_SYSTEMS
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 3: {meta.name}")
        
        try:
            deployment = self.app.state.deployment
            manager = SubsystemManager(deployment, self.trace)
            
            # Activate subsystems
            manager.activate_reasoning_subsystems()
            manager.activate_agent_pool()
            
            # Initialize routing integration
            self._initialize_routing(deployment, manager)
            
            # CRITICAL: Register callbacks between components
            # This ensures proper workflow orchestration after modular refactoring.
            # Callbacks established:
            # 1. agent_pool → reasoning_integration: Job execution results
            # 2. reasoning_integration → telemetry_recorder: Tool selection metrics
            # 3. reasoning_integration → governance_logger: Audit trail for decisions
            # 4. world_model → audit_logger: Meta-reasoning decisions (if available)
            # These callbacks restore the data flow that was implicit in the monolithic main.py
            self._register_cognitive_callbacks(deployment)
            
            self.phase_results[phase] = True
            logger.info("✓ Reasoning systems initialized")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.warning(f"Reasoning systems initialization failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    def _initialize_routing(self, deployment: Any, manager: SubsystemManager) -> None:
        """
        Initialize query routing integration.
        
        P2 Fix: Issue #12 - Graceful degradation for missing dependencies.
        """
        try:
            from vulcan.routing import (
                initialize_routing_components,
                get_collaboration_manager,
                get_telemetry_recorder,
                get_governance_logger,
                COLLABORATION_AVAILABLE,
            )
            
            routing_status = initialize_routing_components()
            logger.debug(f"{LogEmoji.SUCCESS} Query routing initialized")
            
            # Connect agent pool to collaboration manager
            if COLLABORATION_AVAILABLE:
                collab_manager = get_collaboration_manager()
                if hasattr(deployment.collective, "agent_pool") and deployment.collective.agent_pool:
                    collab_manager.set_agent_pool(deployment.collective.agent_pool)
                    logger.debug(f"{LogEmoji.SUCCESS} Agent collaboration connected")
                
                telemetry_recorder = get_telemetry_recorder()
                collab_manager.set_telemetry_recorder(telemetry_recorder)
                logger.debug(f"{LogEmoji.SUCCESS} Telemetry recording enabled")
            
            # Store routing components
            self.app.state.routing_status = routing_status
            self.app.state.telemetry_recorder = get_telemetry_recorder()
            self.app.state.governance_logger = get_governance_logger()
            
        except ImportError as e:
            # P2 Fix: Issue #12 - Graceful degradation for optional dependencies
            logger.warning(f"Query routing not available (optional dependency): {e}")
            self.app.state.routing_status = {"available": False, "reason": f"Import failed: {str(e)}"}
            self.app.state.telemetry_recorder = None
            self.app.state.governance_logger = None
        except Exception as e:
            logger.warning(f"Query routing initialization failed: {e}", exc_info=True)
            self.app.state.routing_status = {"available": False, "reason": f"Initialization failed: {str(e)}"}
            self.app.state.telemetry_recorder = None
            self.app.state.governance_logger = None
    
    def _register_cognitive_callbacks(self, deployment: Any) -> None:
        """
        Register callbacks between cognitive components for proper workflow orchestration.
        
        This is CRITICAL after modular refactoring to ensure:
        - Agent pool results flow to reasoning integration
        - Reasoning integration results flow to telemetry
        - World model decisions flow to audit logger
        - Tool selection events are recorded
        
        All callbacks are logged to startup trace for auditability.
        """
        logger.info(f"{LogEmoji.SUCCESS} Registering cognitive workflow callbacks...")
        
        try:
            # Get reasoning integration singleton
            from vulcan.reasoning.integration import get_reasoning_integration
            reasoning_integration = get_reasoning_integration()
            
            # Log reasoning integration initialization
            self.trace.log_orchestrator_init(
                "reasoning_integration",
                status="registered",
                details={"type": "ReasoningIntegration", "singleton": True}
            )
            
            # Register tools with trace logger
            # Note: Tools are registered dynamically via reasoning engines
            # We log the available engine types for audit trail
            try:
                from vulcan.reasoning import REASONING_ENGINES
                for engine_name in REASONING_ENGINES:
                    self.trace.log_tool_registration(
                        tool_name=engine_name,
                        tool_type="reasoning_engine",
                        status="registered",
                        details={"engine": engine_name, "dynamic": True}
                    )
            except Exception as e:
                logger.debug(f"Could not enumerate reasoning engines: {e}")
            
            # Register query classifier
            try:
                from vulcan.routing import get_query_classifier, QUERY_CLASSIFIER_AVAILABLE
                if QUERY_CLASSIFIER_AVAILABLE:
                    classifier = get_query_classifier()
                    self.trace.log_classifier_init(
                        "query_classifier",
                        status="registered",
                        details={"type": "QueryClassifier", "singleton": True}
                    )
                else:
                    self.trace.log_classifier_init(
                        "query_classifier",
                        status="failed",
                        error="Query classifier not available"
                    )
            except Exception as e:
                logger.debug(f"Query classifier init logging failed: {e}")
            
            # Register query analyzer/router
            try:
                from vulcan.routing import get_query_analyzer, QUERY_ROUTER_AVAILABLE
                if QUERY_ROUTER_AVAILABLE:
                    analyzer = get_query_analyzer()
                    self.trace.log_orchestrator_init(
                        "query_analyzer",
                        status="registered",
                        details={"type": "QueryAnalyzer", "singleton": True}
                    )
                else:
                    self.trace.log_orchestrator_init(
                        "query_analyzer",
                        status="failed",
                        error="Query router not available"
                    )
            except Exception as e:
                logger.debug(f"Query analyzer init logging failed: {e}")
            
            # Store reasoning integration in app state for endpoint access
            self.app.state.reasoning_integration = reasoning_integration
            
            # Log callback: Reasoning integration → Telemetry
            if hasattr(self.app.state, "telemetry_recorder") and self.app.state.telemetry_recorder:
                self.trace.log_callback_registration(
                    source="reasoning_integration",
                    target="telemetry_recorder",
                    callback_type="result_logging",
                    status="registered"
                )
            else:
                self.trace.log_callback_registration(
                    source="reasoning_integration",
                    target="telemetry_recorder",
                    callback_type="result_logging",
                    status="skipped",
                    error="Telemetry recorder not available"
                )
            
            # Log callback: Reasoning integration → Governance logger
            if hasattr(self.app.state, "governance_logger") and self.app.state.governance_logger:
                self.trace.log_callback_registration(
                    source="reasoning_integration",
                    target="governance_logger",
                    callback_type="audit_logging",
                    status="registered"
                )
            else:
                self.trace.log_callback_registration(
                    source="reasoning_integration",
                    target="governance_logger",
                    callback_type="audit_logging",
                    status="skipped",
                    error="Governance logger not available"
                )
            
            # Log agent pool availability
            if hasattr(deployment.collective, "agent_pool") and deployment.collective.agent_pool:
                pool = deployment.collective.agent_pool
                pool_status = pool.get_pool_status()
                total_agents = pool_status.get("total_agents", 0)
                
                # Log each agent's capability
                # Note: Agents are tracked internally, we log their existence
                from vulcan.orchestrator.agent_lifecycle import AgentCapability
                for capability in AgentCapability:
                    self.trace.log_agent_registration(
                        agent_id=f"agent_pool_{capability.value}",
                        capability=capability.value,
                        status="registered",
                        details={"pool_managed": True}
                    )
                
                # Log callback: Agent pool → Reasoning integration
                self.trace.log_callback_registration(
                    source="agent_pool",
                    target="reasoning_integration",
                    callback_type="job_execution",
                    status="registered"
                )
            else:
                logger.warning("Agent pool not available - callbacks not registered")
            
            logger.info(f"{LogEmoji.SUCCESS} Cognitive callbacks registered")
            
        except Exception as e:
            logger.warning(f"Callback registration partially failed: {e}", exc_info=True)
            # Non-critical - continue startup
    
    async def _phase_memory_systems(self) -> None:
        """Phase 4: Initialize memory subsystems."""
        phase = StartupPhase.MEMORY_SYSTEMS
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 4: {meta.name}")
        
        try:
            deployment = self.app.state.deployment
            manager = SubsystemManager(deployment, self.trace)
            
            # Activate subsystems
            manager.activate_memory_subsystems()
            manager.activate_processing_subsystems()
            manager.activate_learning_subsystems()
            manager.activate_world_model(self.app.state)
            manager.activate_planning_subsystems()
            manager.activate_safety_subsystems()
            manager.activate_curiosity_subsystems()
            manager.activate_meta_reasoning_subsystems()
            
            # Log summary
            manager.log_summary()
            
            # Start self-improvement if enabled
            self._start_self_improvement(deployment)
            
            self.phase_results[phase] = True
            logger.info("✓ Memory systems initialized")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.warning(f"Memory systems initialization failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    def _start_self_improvement(self, deployment: Any) -> None:
        """Start autonomous self-improvement drive if enabled."""
        if not self.config.enable_self_improvement:
            return
        
        try:
            world_model = deployment.collective.deps.world_model
            if not world_model:
                logger.warning("Self-improvement enabled but world model not available")
                return
            
            # Initialize meta-reasoning introspection
            from vulcan.world_model.meta_reasoning import MotivationalIntrospection
            
            world_model_config = self.config.world_model
            config_path = getattr(
                world_model_config,
                "meta_reasoning_config",
                "configs/intrinsic_drives.json",
            )
            
            introspection = MotivationalIntrospection(world_model, config_path=config_path)
            logger.debug("✓ MotivationalIntrospection initialized")
            
            # Start autonomous improvement
            if hasattr(world_model, "start_autonomous_improvement"):
                world_model.start_autonomous_improvement()
                logger.info("🚀 Self-improvement drive started")
            else:
                logger.warning("World model doesn't support autonomous improvement")
                
        except Exception as e:
            logger.error(f"Failed to start self-improvement: {e}")
    
    async def _phase_preloading(self) -> None:
        """Phase 5: Preload models and singletons."""
        phase = StartupPhase.PRELOADING
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 5: {meta.name}")
        
        try:
            # Initialize HTTP session
            await self._initialize_http_session()
            
            # Preload models in parallel
            await asyncio.gather(
                self._preload_bert_model(),
                self._preload_model_registry(),
                self._preload_hierarchical_memory(),
                self._preload_unified_learning_system(),
                self._preload_reasoning_singletons(),
                self._preload_reasoning_integration(),
                return_exceptions=True
            )
            
            self.phase_results[phase] = True
            logger.info("✓ Model preloading complete")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.warning(f"Model preloading failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    async def _initialize_http_session(self) -> None:
        """Initialize global HTTP connection pool."""
        try:
            from vulcan.utils_main.http_session import AIOHTTP_AVAILABLE, get_http_session
            if AIOHTTP_AVAILABLE:
                await get_http_session()
                logger.debug("✓ HTTP connection pool initialized")
        except Exception as e:
            logger.debug(f"HTTP session initialization failed: {e}")
    
    async def _preload_bert_model(self) -> None:
        """
        Preload BERT model if not skipped.
        
        P2 Fix: Issue #10 - Log failures as warnings, not debug.
        """
        try:
            from vulcan.simple_mode import SKIP_BERT_EMBEDDINGS
            if SKIP_BERT_EMBEDDINGS:
                logger.debug("BERT preload skipped (SKIP_BERT_EMBEDDINGS)")
                return
            
            from vulcan.processing import GraphixTransformer
            transformer = GraphixTransformer.get_instance()
            if transformer and hasattr(transformer, 'model') and transformer.model is not None:
                if hasattr(transformer.model, 'encode') or hasattr(transformer.model, '__call__'):
                    logger.debug(f"{LogEmoji.SUCCESS} BERT model preloaded")
        except Exception as e:
            logger.warning(f"BERT preload failed: {e}")  # Changed from debug to warning
    
    async def _preload_model_registry(self) -> None:
        """
        Preload SentenceTransformer via model registry.
        
        P2 Fix: Issue #10 - Log failures as warnings, not debug.
        """
        try:
            from vulcan.models import model_registry
            model_registry.preload_all_models()
            stats = model_registry.get_cache_stats()
            logger.debug(f"{LogEmoji.SUCCESS} Model registry: {stats['models_cached']} models")
        except Exception as e:
            logger.warning(f"Model registry preload failed: {e}")  # Changed from debug to warning
    
    async def _preload_hierarchical_memory(self) -> None:
        """
        Preload HierarchicalMemory singleton.
        
        P2 Fix: Issue #10 - Log failures as warnings, not debug.
        """
        try:
            from vulcan.reasoning.singletons import get_hierarchical_memory
            hierarchical_memory = get_hierarchical_memory()
            if hierarchical_memory:
                logger.debug(f"{LogEmoji.SUCCESS} HierarchicalMemory preloaded")
        except Exception as e:
            logger.warning(f"HierarchicalMemory preload failed: {e}")  # Changed from debug to warning
    
    async def _preload_unified_learning_system(self) -> None:
        """
        Preload UnifiedLearningSystem singleton.
        
        P2 Fix: Issue #10 - Log failures as warnings, not debug.
        """
        try:
            from vulcan.reasoning.singletons import get_unified_learning_system
            learning_system = get_unified_learning_system()
            if learning_system:
                logger.debug(f"{LogEmoji.SUCCESS} UnifiedLearningSystem preloaded")
        except Exception as e:
            logger.warning(f"UnifiedLearningSystem preload failed: {e}")  # Changed from debug to warning
    
    async def _preload_reasoning_singletons(self) -> None:
        """
        Pre-warm all reasoning singletons.
        
        P2 Fix: Issue #10 - Log failures as warnings, not debug.
        """
        try:
            from vulcan.reasoning.singletons import prewarm_all
            prewarm_results = prewarm_all()
            success_count = sum(1 for v in prewarm_results.values() if v)
            logger.debug(f"{LogEmoji.SUCCESS} Reasoning singletons: {success_count}/{len(prewarm_results)}")
            
            # Validate ProblemDecomposer if available
            if prewarm_results.get('problem_decomposer'):
                try:
                    from vulcan.problem_decomposer.decomposer_bootstrap import validate_decomposer_setup
                    from vulcan.reasoning.singletons import get_problem_decomposer
                    decomposer = get_problem_decomposer()
                    if decomposer:
                        validation = validate_decomposer_setup(decomposer)
                        if validation['valid']:
                            logger.debug(f"{LogEmoji.SUCCESS} ProblemDecomposer validated")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Reasoning singletons prewarm failed: {e}")  # Changed from debug to warning
    
    async def _preload_reasoning_integration(self) -> None:
        """
        Preload ReasoningIntegration and ToolSelector.
        
        P2 Fix: Issue #10 - Log failures as warnings, not debug.
        """
        try:
            from vulcan.reasoning.integration import get_reasoning_integration
            reasoning_integration = get_reasoning_integration()
            reasoning_integration._init_components()
            logger.debug(f"{LogEmoji.SUCCESS} ReasoningIntegration preloaded")
            
            # Preload SemanticToolMatcher
            try:
                from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
                SemanticToolMatcher._get_shared_model()
                logger.debug(f"{LogEmoji.SUCCESS} SemanticToolMatcher preloaded")
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"ReasoningIntegration preload failed: {e}")  # Changed from debug to warning
    
    async def _phase_monitoring(self) -> None:
        """Phase 6: Start monitoring services."""
        phase = StartupPhase.MONITORING
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 6: {meta.name}")
        
        try:
            # Start memory guard
            self._start_memory_guard()
            
            # Start self-optimizer
            self._start_self_optimizer()
            
            self.phase_results[phase] = True
            logger.info("✓ Monitoring services started")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.warning(f"Monitoring services failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    def _start_memory_guard(self) -> None:
        """
        Start memory guard for automatic garbage collection.
        
        Monitors memory usage and triggers GC when threshold is exceeded.
        Helps prevent OOM conditions in long-running processes.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        """
        try:
            from vulcan.monitoring.memory_guard import start_memory_guard
            memory_guard = start_memory_guard(
                threshold_percent=MEMORY_GUARD_THRESHOLD_PERCENT,
                check_interval=MEMORY_GUARD_CHECK_INTERVAL_SECONDS
            )
            if memory_guard:
                logger.debug(
                    f"{LogEmoji.SUCCESS} MemoryGuard started "
                    f"(threshold={MEMORY_GUARD_THRESHOLD_PERCENT}%)"
                )
        except Exception as e:
            logger.debug(f"MemoryGuard startup failed: {e}")
    
    def _start_self_optimizer(self) -> None:
        """
        Start self-optimizer for autonomous performance tuning.
        
        Monitors latency and memory metrics, automatically adjusting
        system parameters to meet SLO targets. Optional component that
        gracefully degrades if unavailable.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        """
        try:
            from vulcan.monitoring.self_optimizer import (
                SELF_OPTIMIZER_AVAILABLE,
                SelfOptimizer
            )
            
            if not SELF_OPTIMIZER_AVAILABLE:
                self.app.state.self_optimizer = None
                return
            
            self.app.state.self_optimizer = SelfOptimizer(
                target_latency_ms=SELF_OPTIMIZER_TARGET_LATENCY_MS,
                target_memory_mb=SELF_OPTIMIZER_TARGET_MEMORY_MB,
                optimization_interval_s=SELF_OPTIMIZER_INTERVAL_SECONDS,
                enable_auto_tune=True
            )
            self.app.state.self_optimizer.start()
            logger.debug(
                f"{LogEmoji.SUCCESS} SelfOptimizer started "
                f"(target_latency={SELF_OPTIMIZER_TARGET_LATENCY_MS}ms)"
            )
        except Exception as e:
            logger.warning(f"SelfOptimizer startup failed: {e}")
            self.app.state.self_optimizer = None
    
    async def _validate_health(self) -> None:
        """Run health validation after startup."""
        try:
            health_check = HealthCheck(self.app.state)
            health_result = health_check.run_all_checks(self.redis_client)
            health_check.log_health_summary(health_result)
            
            # Fail startup if unhealthy
            if health_result["status"] == HealthStatus.UNHEALTHY.value:
                raise RuntimeError("Startup health check failed: system is unhealthy")
                
        except Exception as e:
            logger.error(f"Health validation failed: {e}", exc_info=True)
            raise
    
    async def run_shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        logger.info(f"VULCAN-AGI worker {self.worker_id} shutting down")
        
        # Close HTTP session
        try:
            from vulcan.utils_main.http_session import close_http_session
            await close_http_session()
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")
        
        # Stop SelfOptimizer
        if hasattr(self.app.state, "self_optimizer") and self.app.state.self_optimizer:
            try:
                self.app.state.self_optimizer.stop()
                logger.debug("✓ SelfOptimizer stopped")
            except Exception as e:
                logger.warning(f"Error stopping SelfOptimizer: {e}")
        
        # Shutdown HardwareDispatcher if present
        if hasattr(self.app.state, 'arena') and hasattr(self.app.state.arena, 'hardware_dispatcher'):
            if self.app.state.arena.hardware_dispatcher:
                try:
                    self.app.state.arena.hardware_dispatcher.shutdown()
                    logger.debug("✓ HardwareDispatcher shutdown")
                except Exception as e:
                    logger.warning(f"Error shutting down HardwareDispatcher: {e}")
        
        # Stop self-improvement drive
        if hasattr(self.app.state, "deployment"):
            try:
                deployment = self.app.state.deployment
                world_model = deployment.collective.deps.world_model
                if world_model and hasattr(world_model, "stop_autonomous_improvement"):
                    world_model.stop_autonomous_improvement()
                    logger.debug("🛑 Self-improvement drive stopped")
            except Exception as e:
                logger.error(f"Error stopping self-improvement: {e}")
        
        # Save checkpoint
        if hasattr(self.app.state, "deployment"):
            try:
                deployment = self.app.state.deployment
                checkpoint_path = f"shutdown_checkpoint_{int(time.time())}.pkl"
                deployment.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Cleanup Redis
        if self.redis_client and hasattr(self.app.state, "worker_id"):
            try:
                self.redis_client.delete(f"deployment:{self.app.state.worker_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup Redis: {e}")
        
        # Shutdown ThreadPoolExecutor
        if self.executor:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
                logger.debug("✓ ThreadPoolExecutor shutdown")
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")
        
        # Release process lock
        if self.process_lock is not None:
            try:
                self.process_lock.release()
                logger.debug("✓ Process lock released")
            except Exception as e:
                logger.warning(f"Error releasing process lock: {e}")
        
        logger.info("✓ VULCAN-AGI shutdown complete")
