"""
Startup Manager

Orchestrates the VULCAN-AGI startup process through well-defined phases
with proper error isolation, health validation, and status reporting.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from threading import Thread
from typing import Any, Optional, Dict, Callable
from concurrent.futures import ThreadPoolExecutor

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
)
from .phases import StartupPhase, get_phase_metadata, is_critical_phase
from .subsystems import SubsystemManager
from .health import HealthCheck, HealthStatus


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
    
    async def run_startup(self) -> None:
        """
        Execute complete startup sequence.
        
        Raises:
            RuntimeError: If critical phase fails
        """
        logger.info(
            f"Starting VULCAN-AGI worker {self.worker_id} "
            f"in {self.settings.deployment_mode} mode"
        )
        
        try:
            # Phase 1: Configuration
            await self._phase_configuration()
            
            # Phase 2: Core Services
            await self._phase_core_services()
            
            # Phase 3: Reasoning Systems
            await self._phase_reasoning_systems()
            
            # Phase 4: Memory Systems
            await self._phase_memory_systems()
            
            # Phase 5: Preloading
            await self._phase_preloading()
            
            # Phase 6: Monitoring
            await self._phase_monitoring()
            
            # Health validation
            await self._validate_health()
            
            # Startup complete
            elapsed = time.time() - self.startup_time
            logger.info(f"✅ VULCAN-AGI worker {self.worker_id} started in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}", exc_info=True)
            raise
    
    async def _phase_configuration(self) -> None:
        """Phase 1: Load configuration."""
        phase = StartupPhase.CONFIGURATION
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 1: {meta.name}")
        
        try:
            # Import required modules
            from vulcan.config import get_config, AgentConfig
            
            # Load configuration profile
            profile_name = self.settings.deployment_mode
            if profile_name not in ["production", "testing", "development"]:
                profile_name = "development"
            
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
            logger.info(f"✓ Configuration loaded ({profile_name} profile)")
            
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
        """Ensure required directories exist."""
        try:
            Path(DEFAULT_DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(DEFAULT_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
            Path(DEFAULT_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            logger.debug("✓ Persistence directories ensured")
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
            from vulcan.orchestrator.variants import ProductionDeployment
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
        """Setup ThreadPoolExecutor for parallel tasks."""
        try:
            loop = asyncio.get_event_loop()
            self.executor = ThreadPoolExecutor(
                max_workers=DEFAULT_THREAD_POOL_SIZE,
                thread_name_prefix=THREAD_NAME_PREFIX
            )
            loop.set_default_executor(self.executor)
            self.app.state.executor = self.executor
            logger.debug(
                f"✓ ThreadPoolExecutor: {DEFAULT_THREAD_POOL_SIZE} workers"
            )
        except Exception as e:
            logger.warning(f"Failed to set default executor: {e}")
    
    def _get_checkpoint_path(self) -> Optional[str]:
        """Get checkpoint path if it exists and is valid."""
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
            from vulcan.distillation.hybrid_executor import (
                get_or_create_hybrid_executor,
                verify_hybrid_executor_setup
            )
            from vulcan.utils_main.openai_client import get_openai_client
            
            self.app.state.hybrid_executor = get_or_create_hybrid_executor(
                local_llm=llm_instance,
                openai_client_getter=get_openai_client,
                mode=self.settings.llm_execution_mode,
                timeout=self.settings.llm_parallel_timeout,
                ensemble_min_confidence=self.settings.llm_ensemble_min_confidence,
                openai_max_tokens=self.settings.llm_openai_max_tokens,
            )
            logger.debug(f"✓ HybridLLMExecutor ({self.settings.llm_execution_mode})")
            
            # Verify setup
            verification = verify_hybrid_executor_setup()
            if verification["status"] == "PASS":
                logger.debug(f"✓ HybridExecutor verified")
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
        """Register worker in Redis if available."""
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
            logger.debug(f"✓ Worker {self.worker_id} registered in Redis")
        except Exception as e:
            logger.error(f"Failed to register in Redis: {e}")
    
    def _start_rate_limit_cleanup(self) -> None:
        """Start rate limit cleanup thread."""
        try:
            from vulcan.api.rate_limiting import cleanup_rate_limits
            
            # Get global thread reference
            import vulcan.server.app as app_module
            if not hasattr(app_module, 'rate_limit_cleanup_thread') or \
               app_module.rate_limit_cleanup_thread is None or \
               not app_module.rate_limit_cleanup_thread.is_alive():
                thread = Thread(target=cleanup_rate_limits, daemon=True)
                thread.start()
                app_module.rate_limit_cleanup_thread = thread
                logger.debug("✓ Rate limit cleanup thread started")
        except Exception as e:
            logger.warning(f"Rate limit cleanup thread failed: {e}")
    
    async def _phase_reasoning_systems(self) -> None:
        """Phase 3: Initialize reasoning subsystems."""
        phase = StartupPhase.REASONING_SYSTEMS
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 3: {meta.name}")
        
        try:
            deployment = self.app.state.deployment
            manager = SubsystemManager(deployment)
            
            # Activate subsystems
            manager.activate_reasoning_subsystems()
            manager.activate_agent_pool()
            
            # Initialize routing integration
            self._initialize_routing(deployment, manager)
            
            self.phase_results[phase] = True
            logger.info("✓ Reasoning systems initialized")
            
        except Exception as e:
            self.phase_results[phase] = False
            logger.warning(f"Reasoning systems initialization failed: {e}", exc_info=True)
            if is_critical_phase(phase):
                raise RuntimeError(f"Critical phase {meta.name} failed") from e
    
    def _initialize_routing(self, deployment: Any, manager: SubsystemManager) -> None:
        """Initialize query routing integration."""
        try:
            from vulcan.routing import (
                initialize_routing_components,
                get_collaboration_manager,
                get_telemetry_recorder,
                get_governance_logger,
                COLLABORATION_AVAILABLE,
            )
            
            routing_status = initialize_routing_components()
            logger.debug("✓ Query routing initialized")
            
            # Connect agent pool to collaboration manager
            if COLLABORATION_AVAILABLE:
                collab_manager = get_collaboration_manager()
                if hasattr(deployment.collective, "agent_pool") and deployment.collective.agent_pool:
                    collab_manager.set_agent_pool(deployment.collective.agent_pool)
                    logger.debug("✓ Agent collaboration connected")
                
                telemetry_recorder = get_telemetry_recorder()
                collab_manager.set_telemetry_recorder(telemetry_recorder)
                logger.debug("✓ Telemetry recording enabled")
            
            # Store routing components
            self.app.state.routing_status = routing_status
            self.app.state.telemetry_recorder = get_telemetry_recorder()
            self.app.state.governance_logger = get_governance_logger()
            
        except ImportError as e:
            logger.debug(f"Query routing not available: {e}")
        except Exception as e:
            logger.warning(f"Query routing initialization failed: {e}", exc_info=True)
    
    async def _phase_memory_systems(self) -> None:
        """Phase 4: Initialize memory subsystems."""
        phase = StartupPhase.MEMORY_SYSTEMS
        meta = get_phase_metadata(phase)
        
        logger.info(f"Phase 4: {meta.name}")
        
        try:
            deployment = self.app.state.deployment
            manager = SubsystemManager(deployment)
            
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
        """Preload BERT model if not skipped."""
        try:
            from vulcan.simple_mode import SKIP_BERT_EMBEDDINGS
            if SKIP_BERT_EMBEDDINGS:
                logger.debug("BERT preload skipped (SKIP_BERT_EMBEDDINGS)")
                return
            
            from vulcan.processing import GraphixTransformer
            transformer = GraphixTransformer.get_instance()
            if transformer and hasattr(transformer, 'model') and transformer.model is not None:
                if hasattr(transformer.model, 'encode') or hasattr(transformer.model, '__call__'):
                    logger.debug("✓ BERT model preloaded")
        except Exception as e:
            logger.debug(f"BERT preload failed: {e}")
    
    async def _preload_model_registry(self) -> None:
        """Preload SentenceTransformer via model registry."""
        try:
            from vulcan.models import model_registry
            model_registry.preload_all_models()
            stats = model_registry.get_cache_stats()
            logger.debug(f"✓ Model registry: {stats['models_cached']} models")
        except Exception as e:
            logger.debug(f"Model registry preload failed: {e}")
    
    async def _preload_hierarchical_memory(self) -> None:
        """Preload HierarchicalMemory singleton."""
        try:
            from vulcan.reasoning.singletons import get_hierarchical_memory
            hierarchical_memory = get_hierarchical_memory()
            if hierarchical_memory:
                logger.debug("✓ HierarchicalMemory preloaded")
        except Exception as e:
            logger.debug(f"HierarchicalMemory preload failed: {e}")
    
    async def _preload_unified_learning_system(self) -> None:
        """Preload UnifiedLearningSystem singleton."""
        try:
            from vulcan.reasoning.singletons import get_unified_learning_system
            learning_system = get_unified_learning_system()
            if learning_system:
                logger.debug("✓ UnifiedLearningSystem preloaded")
        except Exception as e:
            logger.debug(f"UnifiedLearningSystem preload failed: {e}")
    
    async def _preload_reasoning_singletons(self) -> None:
        """Pre-warm all reasoning singletons."""
        try:
            from vulcan.reasoning.singletons import prewarm_all
            prewarm_results = prewarm_all()
            success_count = sum(1 for v in prewarm_results.values() if v)
            logger.debug(f"✓ Reasoning singletons: {success_count}/{len(prewarm_results)}")
            
            # Validate ProblemDecomposer if available
            if prewarm_results.get('problem_decomposer'):
                try:
                    from vulcan.problem_decomposer.decomposer_bootstrap import validate_decomposer_setup
                    from vulcan.reasoning.singletons import get_problem_decomposer
                    decomposer = get_problem_decomposer()
                    if decomposer:
                        validation = validate_decomposer_setup(decomposer)
                        if validation['valid']:
                            logger.debug(f"✓ ProblemDecomposer validated")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Reasoning singletons prewarm failed: {e}")
    
    async def _preload_reasoning_integration(self) -> None:
        """Preload ReasoningIntegration and ToolSelector."""
        try:
            from vulcan.reasoning.integration import get_reasoning_integration
            reasoning_integration = get_reasoning_integration()
            reasoning_integration._init_components()
            logger.debug("✓ ReasoningIntegration preloaded")
            
            # Preload SemanticToolMatcher
            try:
                from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
                SemanticToolMatcher._get_shared_model()
                logger.debug("✓ SemanticToolMatcher preloaded")
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"ReasoningIntegration preload failed: {e}")
    
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
        """Start memory guard for automatic GC."""
        try:
            from vulcan.monitoring.memory_guard import start_memory_guard
            memory_guard = start_memory_guard(
                threshold_percent=MEMORY_GUARD_THRESHOLD_PERCENT,
                check_interval=MEMORY_GUARD_CHECK_INTERVAL_SECONDS
            )
            if memory_guard:
                logger.debug(
                    f"✓ MemoryGuard started "
                    f"(threshold={MEMORY_GUARD_THRESHOLD_PERCENT}%)"
                )
        except Exception as e:
            logger.debug(f"MemoryGuard startup failed: {e}")
    
    def _start_self_optimizer(self) -> None:
        """Start self-optimizer if available."""
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
                f"✓ SelfOptimizer started "
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
