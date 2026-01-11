"""
FastAPI Application Setup

Creates and configures the FastAPI application with lifespan management.
"""

import asyncio
import concurrent.futures
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# This will be populated from main.py imports
# Placeholder for now - actual implementation uses imports from main
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP LOGIC
    global rate_limit_cleanup_thread, _process_lock

    worker_id = os.getpid()
    startup_complete = False

    # CRITICAL: Check if we're in test mode (deployment already set)
    # If deployment exists in app.state, skip initialization (tests use mock)
    if hasattr(app.state, "deployment") and app.state.deployment is not None:
        logger.info(f"Test mode detected - using existing mock deployment")
        try:
            yield
        finally:
            logger.info("Test mode shutdown - skipping cleanup")
        return

    # ============================================================
    # SPLIT-BRAIN PREVENTION: Acquire process lock when Redis unavailable
    # This ensures only one orchestrator instance can run at a time,
    # preventing the race condition where multiple processes have
    # isolated state and cause oscillating job counts.
    # ============================================================
    if redis_client is None:
        logger.warning(
            "Redis unavailable - acquiring file-based process lock to prevent split-brain"
        )
        _process_lock = ProcessLock()
        if not _process_lock.acquire():
            logger.critical(
                "FATAL: Cannot acquire process lock. Another vulcan.orchestrator "
                "instance is already running. Without Redis for state synchronization, "
                "running multiple instances would cause a split-brain condition. "
                "Either start Redis or stop the other instance."
            )
            # Raise error to prevent this instance from starting
            raise RuntimeError(
                "Split-brain prevention: Another orchestrator instance is running. "
                "Start Redis for multi-instance support or stop the other process."
            )
        logger.info("Process lock acquired - singleton mode active (no Redis)")
    else:
        logger.info("Redis available - multi-instance mode supported")

    logger.info(
        f"Starting VULCAN-AGI worker {worker_id} in {settings.deployment_mode} mode"
    )

    # PERFORMANCE FIX: Increase default ThreadPoolExecutor size to prevent
    # thread pool exhaustion when parallel cognitive tasks all use run_in_executor(None, ...)
    # The default pool size is min(32, os.cpu_count() + 4) which may be too small
    # when multiple parallel tasks (memory, reasoning, planning, world_model) all compete.
    # This was causing intermittent 25-30 second delays when threads were exhausted.
    try:
        import concurrent.futures
        loop = asyncio.get_event_loop()
        # Use 32 workers to handle parallel cognitive tasks without exhaustion
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="vulcan_")
        loop.set_default_executor(executor)
        logger.info("[PERFORMANCE] Set default ThreadPoolExecutor to 32 workers to prevent thread pool exhaustion")
    except Exception as e:
        logger.warning(f"Failed to set default executor: {e}")

    try:
        # Load configuration profile
        profile_name = settings.deployment_mode
        if profile_name not in ["production", "testing", "development"]:
            profile_name = "development"

        config = get_config(profile_name)

        # Validate config is an AgentConfig instance
        if not isinstance(config, AgentConfig):
            logger.error(
                f"Invalid config type returned: {type(config)}, creating default config"
            )
            config = AgentConfig()

    except Exception as e:
        logger.error(f"Failed to load configuration profile: {e}")
        logger.info("Creating default AgentConfig")
        config = AgentConfig()

    # Set defaults if attributes don't exist
    if not hasattr(config, "max_graph_size"):
        config.max_graph_size = settings.max_graph_size
    if not hasattr(config, "max_execution_time_s"):
        config.max_execution_time_s = settings.max_execution_time_s
    if not hasattr(config, "max_memory_mb"):
        config.max_memory_mb = settings.max_memory_mb
    if not hasattr(config, "slo_p95_latency_ms"):
        config.slo_p95_latency_ms = 1000
    if not hasattr(config, "slo_p99_latency_ms"):
        config.slo_p99_latency_ms = 2000
    if not hasattr(config, "slo_max_error_rate"):
        config.slo_max_error_rate = 0.1
    if not hasattr(config, "max_working_memory"):
        config.max_working_memory = 20

    # Add self-improvement configuration
    if not hasattr(config, "enable_self_improvement"):
        config.enable_self_improvement = settings.enable_self_improvement
    if not hasattr(config, "self_improvement_config"):
        # Ensure world_model config section exists if needed
        if not hasattr(config, "world_model"):
            from dataclasses import make_dataclass

            WorldModelConfig = make_dataclass(
                "WorldModelConfig",
                [("self_improvement_config", str, settings.self_improvement_config)],
            )
            config.world_model = WorldModelConfig()
        elif not hasattr(config.world_model, "self_improvement_config"):
            setattr(
                config.world_model,
                "self_improvement_config",
                settings.self_improvement_config,
            )

    if not hasattr(config, "self_improvement_state"):
        # Ensure world_model config section exists if needed
        if not hasattr(config, "world_model"):
            from dataclasses import make_dataclass

            WorldModelConfig = make_dataclass(
                "WorldModelConfig",
                [("self_improvement_state", str, settings.self_improvement_state)],
            )
            config.world_model = WorldModelConfig()
        elif not hasattr(config.world_model, "self_improvement_state"):
            setattr(
                config.world_model,
                "self_improvement_state",
                settings.self_improvement_state,
            )

    try:
        # Check if checkpoint file exists and is valid before loading
        checkpoint_to_load = None
        if settings.checkpoint_path:
            if (
                os.path.exists(settings.checkpoint_path)
                and os.path.getsize(settings.checkpoint_path) > 0
            ):
                checkpoint_to_load = settings.checkpoint_path
                logger.info(f"Will load checkpoint from {checkpoint_to_load}")
            else:
                logger.warning(
                    f"Checkpoint file {settings.checkpoint_path} does not exist or is empty, starting fresh"
                )

        deployment = initialize_component(
            "deployment",
            lambda: ProductionDeployment(config, checkpoint_path=checkpoint_to_load, redis_client=redis_client),
        )

        # Note: Use singleton UnifiedRuntime to prevent manifest reload per-query
        if UNIFIED_RUNTIME_AVAILABLE:
            # Note: Use get_or_create_unified_runtime to prevent repeated init/shutdown
            set_runtime_func = None
            try:
                from vulcan.reasoning.singletons import get_or_create_unified_runtime, set_unified_runtime
                set_runtime_func = set_unified_runtime
                deployment.unified_runtime = get_or_create_unified_runtime()
                if deployment.unified_runtime:
                    logger.info("✓ UnifiedRuntime initialized via singleton")
                else:
                    logger.warning("UnifiedRuntime not available")
            except ImportError:
                deployment.unified_runtime = UnifiedRuntime()
                # Note Issue #1: Register fallback instance with singleton
                if set_runtime_func is not None:
                    try:
                        set_runtime_func(deployment.unified_runtime)
                    except Exception:
                        pass
                logger.info("✓ UnifiedRuntime initialized directly (registered with singleton)")

        # Initialize LLM component
        llm_instance = initialize_component(
            "llm", lambda: GraphixVulcanLLM(config_path="configs/llm_config.yaml")
        )
        app.state.llm = llm_instance
        
        # Note: Register LLM client with singletons module
        # This ensures MathematicalComputationTool and other reasoning components
        # receive the actual LLM client object instead of None or a string.
        # Without this, tools fall back to template-only mode.
        try:
            from vulcan.reasoning.singletons import set_llm_client
            if llm_instance is not None:
                set_llm_client(llm_instance)
                logger.info("✓ LLM client registered with singletons module")
        except ImportError:
            logger.debug("Singletons module not available for LLM client registration")
        except Exception as e:
            logger.warning(f"Failed to register LLM client with singletons: {e}")

        # PERFORMANCE FIX (Issue #1): Initialize HybridLLMExecutor ONCE at startup
        # Previously this was instantiated per-request, adding ~0.5s overhead each time
        # Now we use a module-level singleton that persists across all requests
        try:
            app.state.hybrid_executor = get_or_create_hybrid_executor(
                local_llm=llm_instance,
                openai_client_getter=get_openai_client,
                mode=settings.llm_execution_mode,
                timeout=settings.llm_parallel_timeout,
                ensemble_min_confidence=settings.llm_ensemble_min_confidence,
                openai_max_tokens=settings.llm_openai_max_tokens,
            )
            logger.info(f"✓ HybridLLMExecutor initialized at startup (mode={settings.llm_execution_mode})")
            
            # FIX #1 VERIFICATION: Verify internal LLM connection
            verification = verify_hybrid_executor_setup()
            if verification["status"] == "PASS":
                logger.info(f"✓ HybridExecutor verification: {verification['message']}")
            else:
                logger.warning(f"⚠ HybridExecutor verification: {verification['message']}")
        except Exception as e:
            logger.warning(f"Failed to initialize HybridLLMExecutor at startup: {e}")
            app.state.hybrid_executor = None

        # Initialize Knowledge Distiller for learning from OpenAI responses
        if settings.enable_knowledge_distillation:
            try:
                knowledge_distiller = initialize_knowledge_distiller(
                    local_llm=llm_instance,
                    storage_path=settings.distillation_storage_path,
                    batch_size=settings.distillation_batch_size,
                    training_interval_s=settings.distillation_training_interval_s,
                    auto_train=settings.distillation_auto_train,
                    learning_rate=settings.distillation_learning_rate,
                )
                app.state.knowledge_distiller = knowledge_distiller
                logger.info("✓ OpenAI Knowledge Distiller initialized - Vulcan will learn from OpenAI responses")
            except Exception as e:
                logger.warning(f"Failed to initialize Knowledge Distiller: {e}")
                app.state.knowledge_distiller = None
        else:
            app.state.knowledge_distiller = None
            logger.info("Knowledge Distillation disabled by configuration")

        if redis_client:
            try:
                worker_metadata = {
                    "worker_id": worker_id,
                    "started": time.time(),
                    "deployment_mode": settings.deployment_mode,
                }
                redis_client.setex(
                    f"deployment:{worker_id}",
                    3600,
                    msgpack.packb(worker_metadata, use_bin_type=True),
                )
                logger.info(f"Worker {worker_id} registered in Redis")
            except Exception as e:
                logger.error(f"Failed to register in Redis: {e}")

        app.state.deployment = deployment
        app.state.worker_id = worker_id
        app.state.startup_time = time.time()

        # CRITICAL: Ensure persistence directories exist IMMEDIATELY after setting app.state
        try:
            Path("data").mkdir(parents=True, exist_ok=True)
            Path("configs").mkdir(parents=True, exist_ok=True)
            Path("checkpoints").mkdir(parents=True, exist_ok=True)
            logger.info("✓ Data, Configs, and Checkpoints directories ensured")

        except Exception as e:
            logger.warning(
                f"Could not ensure data/configs/checkpoints directories: {e}"
            )

        if not rate_limit_cleanup_thread or not rate_limit_cleanup_thread.is_alive():
            rate_limit_cleanup_thread = Thread(target=cleanup_rate_limits, daemon=True)
            rate_limit_cleanup_thread.start()

        logger.info(f"VULCAN-AGI worker {worker_id} started successfully")

        # ADDED: Initialize all Vulcan subsystem modules for complete activation
        def _activate_subsystem(
            deps, attr_name: str, display_name: str, needs_init: bool = False
        ):
            """Helper to activate a subsystem with optional initialization."""
            if hasattr(deps, attr_name) and getattr(deps, attr_name):
                subsystem = getattr(deps, attr_name)
                if needs_init and hasattr(subsystem, "initialize"):
                    subsystem.initialize()
                logger.info(f"✓ {display_name} activated")
                return True
            return False

        try:
            logger.info("Activating all Vulcan subsystem modules...")

            # ================================================================
            # AGENT POOL ACTIVATION - Core distributed processing
            # ================================================================
            if (
                hasattr(deployment.collective, "agent_pool")
                and deployment.collective.agent_pool
            ):
                pool = deployment.collective.agent_pool
                pool_status = pool.get_pool_status()
                total_agents = pool_status.get("total_agents", 0)
                idle_agents = pool_status.get("state_distribution", {}).get("idle", 0)

                logger.info(
                    f"✓ Agent Pool activated: {total_agents} agents ({idle_agents} idle)"
                )

                # Ensure minimum agents are available
                if total_agents < pool.min_agents:
                    logger.warning(
                        f"Agent pool below minimum ({total_agents} < {pool.min_agents}), spawning more..."
                    )
                    from vulcan.orchestrator.agent_lifecycle import AgentCapability

                    for _ in range(pool.min_agents - total_agents):
                        pool.spawn_agent(AgentCapability.GENERAL)
                    logger.info(
                        f"✓ Agent Pool scaled to {pool.get_pool_status()['total_agents']} agents"
                    )
            else:
                logger.warning(
                    "Agent Pool not available - distributed processing disabled"
                )

            # ================================================================
            # REASONING SUBSYSTEMS - Core cognitive processing
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps, "symbolic", "Symbolic Reasoning"
            )
            _activate_subsystem(
                deployment.collective.deps, "probabilistic", "Probabilistic Reasoning"
            )
            _activate_subsystem(
                deployment.collective.deps, "causal", "Causal Reasoning"
            )
            _activate_subsystem(
                deployment.collective.deps, "abstract", "Analogical/Abstract Reasoning"
            )
            _activate_subsystem(
                deployment.collective.deps, "cross_modal", "Cross-Modal Reasoning"
            )

            # ================================================================
            # MEMORY SUBSYSTEMS - Knowledge persistence
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps, "ltm", "Long-term Memory (Vector Index)"
            )
            _activate_subsystem(
                deployment.collective.deps, "am", "Episodic/Autobiographical Memory"
            )
            _activate_subsystem(
                deployment.collective.deps,
                "compressed_memory",
                "Compressed Memory Persistence",
            )

            # ================================================================
            # PROCESSING SUBSYSTEMS - Input/Output handling
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps, "multimodal", "Multimodal Processor"
            )

            # ================================================================
            # LEARNING SUBSYSTEMS - Adaptation and growth
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps, "continual", "Continual Learning"
            )
            _activate_subsystem(
                deployment.collective.deps, "meta_cognitive", "Meta-Cognitive Monitor"
            )
            _activate_subsystem(
                deployment.collective.deps,
                "compositional",
                "Compositional Understanding",
            )

            # ================================================================
            # WORLD MODEL - Predictive modeling
            # ================================================================
            if (
                hasattr(deployment.collective.deps, "world_model")
                and deployment.collective.deps.world_model
            ):
                world_model = deployment.collective.deps.world_model
                logger.info("✓ World Model activated")

                # Check for meta-reasoning components
                if (
                    hasattr(world_model, "motivational_introspection")
                    and world_model.motivational_introspection
                ):
                    logger.info("  ✓ Motivational Introspection sub-system active")
                if (
                    hasattr(world_model, "self_improvement_drive")
                    and world_model.self_improvement_drive
                ):
                    logger.info("  ✓ Self-Improvement Drive sub-system active")
                
                # Initialize SystemObserver for event tracking
                # This connects the query processing pipeline to the world model
                try:
                    from vulcan.world_model.system_observer import initialize_system_observer
                    system_observer = initialize_system_observer(world_model)
                    app.state.system_observer = system_observer
                    logger.info("  ✓ SystemObserver initialized - World Model now receives system events")
                except ImportError as e:
                    logger.debug(f"SystemObserver not available: {e}")
                    app.state.system_observer = None
                except Exception as e:
                    logger.warning(f"SystemObserver initialization failed: {e}")
                    app.state.system_observer = None

            # ================================================================
            # PLANNING SUBSYSTEMS - Goal management
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps, "goal_system", "Hierarchical Goal System"
            )
            _activate_subsystem(
                deployment.collective.deps, "resource_compute", "Resource-Aware Compute"
            )

            # ================================================================
            # SAFETY SUBSYSTEMS - Safety constraints
            # ================================================================
            if (
                hasattr(deployment.collective.deps, "safety_validator")
                and deployment.collective.deps.safety_validator
            ):
                safety_validator = deployment.collective.deps.safety_validator
                if hasattr(safety_validator, "activate_all_constraints"):
                    try:
                        safety_validator.activate_all_constraints()
                        logger.info("✓ Safety Validator with all constraints activated")
                    except Exception as e:
                        logger.warning(f"Failed to activate all constraints: {e}")
                        logger.info(
                            "✓ Safety Validator activated (without all constraints)"
                        )
                else:
                    logger.info("✓ Safety Validator activated")

            _activate_subsystem(
                deployment.collective.deps, "governance", "Governance Orchestrator"
            )
            _activate_subsystem(
                deployment.collective.deps, "nso_aligner", "NSO Aligner"
            )
            _activate_subsystem(
                deployment.collective.deps, "explainer", "Explainability Node"
            )

            # ================================================================
            # CURIOSITY & EXPLORATION SUBSYSTEMS
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps,
                "experiment_generator",
                "Experiment Generator",
            )
            _activate_subsystem(
                deployment.collective.deps, "problem_executor", "Problem Executor"
            )

            # ================================================================
            # META-REASONING SUBSYSTEMS (from world model)
            # ================================================================
            _activate_subsystem(
                deployment.collective.deps,
                "self_improvement_drive",
                "Self-Improvement Drive",
            )
            _activate_subsystem(
                deployment.collective.deps,
                "motivational_introspection",
                "Motivational Introspection",
            )
            _activate_subsystem(
                deployment.collective.deps, "objective_hierarchy", "Objective Hierarchy"
            )
            _activate_subsystem(
                deployment.collective.deps,
                "objective_negotiator",
                "Objective Negotiator",
            )
            _activate_subsystem(
                deployment.collective.deps,
                "goal_conflict_detector",
                "Goal Conflict Detector",
            )

            logger.info("✅ All Vulcan subsystem modules activation complete")

            # Log summary of active systems
            deps_status = deployment.collective.deps.get_status()
            logger.info(
                f"📊 System Status: {deps_status['available_count']}/{deps_status['total_dependencies']} subsystems active"
            )

            # ================================================================
            # QUERY ROUTING INTEGRATION - Dual-Mode Learning Support
            # ================================================================
            try:
                from vulcan.routing import (
                    initialize_routing_components,
                    get_collaboration_manager,
                    get_telemetry_recorder,
                    get_governance_logger,
                    COLLABORATION_AVAILABLE,
                )

                routing_status = initialize_routing_components()
                logger.info("✓ Query Routing Layer initialized")

                # Connect agent pool to collaboration manager
                if COLLABORATION_AVAILABLE:
                    collab_manager = get_collaboration_manager()
                    if (
                        hasattr(deployment.collective, "agent_pool")
                        and deployment.collective.agent_pool
                    ):
                        collab_manager.set_agent_pool(deployment.collective.agent_pool)
                        logger.info("  ✓ Agent Collaboration connected to Agent Pool")

                    telemetry_recorder = get_telemetry_recorder()
                    collab_manager.set_telemetry_recorder(telemetry_recorder)
                    logger.info("  ✓ AI Interaction Telemetry recording enabled")

                # Store routing components in app.state for endpoint access
                app.state.routing_status = routing_status
                app.state.telemetry_recorder = get_telemetry_recorder()
                app.state.governance_logger = get_governance_logger()

                logger.info("✓ Dual-Mode Learning System activated")
                logger.info("  ✓ MODE 1: User Interaction Telemetry → utility_memory")
                logger.info(
                    "  ✓ MODE 2: AI-to-AI Interaction Telemetry → success/risk_memory"
                )

            except ImportError as e:
                logger.warning(f"Query Routing Layer not available: {e}")
            except Exception as e:
                logger.warning(
                    f"Query Routing Layer initialization failed: {e}", exc_info=True
                )

        except Exception as e:
            logger.error(f"Error during subsystem activation: {e}", exc_info=True)
            logger.warning("Continuing with partial subsystem activation")

        # Start self-improvement drive if enabled
        if config.enable_self_improvement:
            try:
                # Access world model from deployment
                world_model = deployment.collective.deps.world_model

                # ADDED: Initialize meta-reasoning introspection (MODERN MODE - FIXED)
                if world_model:
                    from vulcan.world_model.meta_reasoning import (
                        MotivationalIntrospection,
                    )

                    # Modern approach: get config path from AgentConfig
                    world_model_config = (
                        config.world_model
                    )  # This returns WorldModelConfig instance
                    config_path = getattr(
                        world_model_config,
                        "meta_reasoning_config",
                        "configs/intrinsic_drives.json",
                    )

                    introspection = MotivationalIntrospection(
                        world_model, config_path=config_path
                    )
                    logger.info("✓ MotivationalIntrospection initialized (modern mode)")

                if world_model and hasattr(world_model, "start_autonomous_improvement"):
                    world_model.start_autonomous_improvement()
                    logger.info("🚀 Autonomous self-improvement drive started")
                else:
                    logger.warning(
                        "Self-improvement enabled but world model doesn't support it"
                    )
            except Exception as e:
                logger.error(f"Failed to start self-improvement drive: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize deployment: {e}", exc_info=True)
        raise
    except asyncio.CancelledError:
        logger.warning(f"VULCAN-AGI worker {worker_id} startup cancelled")
        raise

    startup_complete = True
    
    # HTTP CONNECTION POOL FIX: Initialize global HTTP session
    if AIOHTTP_AVAILABLE:
        await get_http_session()
        logger.info("✓ Global HTTP connection pool initialized")

    # PERFORMANCE FIX (Issue #30): Preload BERT model at startup to avoid 3.5s+ load during first request
    # Only load if SKIP_BERT_EMBEDDINGS is not enabled (i.e., not in simple mode)
    # Import outside try block to distinguish import failures from loading failures
    from vulcan.simple_mode import SKIP_BERT_EMBEDDINGS
    if not SKIP_BERT_EMBEDDINGS:
        try:
            from vulcan.processing import GraphixTransformer
            # Use singleton pattern - this ensures BERT is loaded once at startup
            transformer = GraphixTransformer.get_instance()
            # Check if model is properly loaded (not just non-None but has expected attributes)
            if transformer and hasattr(transformer, 'model') and transformer.model is not None:
                # Verify it's a proper model by checking for encode method (BERT models have this)
                if hasattr(transformer.model, 'encode') or hasattr(transformer.model, '__call__'):
                    logger.info("✓ BERT model preloaded at startup (singleton pattern)")
                else:
                    logger.info("✓ BERT model instance created but may not be fully initialized")
            else:
                logger.info("✓ BERT model loading deferred (fallback mode or SKIP_BERT_EMBEDDINGS active)")
        except ImportError as e:
            logger.warning(f"GraphixTransformer import failed: {e}")
        except Exception as e:
            logger.warning(f"BERT model preload failed (will load on first request): {e}")
    else:
        logger.info("BERT model loading skipped (SKIP_BERT_EMBEDDINGS=true)")

    # PERFORMANCE FIX (Phase 3): Preload SentenceTransformer via global model registry
    # This ensures the model is loaded exactly ONCE per process and shared across all components
    # Must be done BEFORE ReasoningIntegration preload to avoid duplicate loads
    try:
        from vulcan.models import model_registry
        model_registry.preload_all_models()
        stats = model_registry.get_cache_stats()
        logger.info(f"✓ Model Registry preloaded: {stats['models_cached']} models cached ({stats['model_keys']})")
    except ImportError as e:
        logger.debug(f"Model registry module not available for preload: {e}")
    except Exception as e:
        logger.warning(f"Model registry preload failed (models will load lazily): {e}")

    # PERF FIX Issue #2: Preload HierarchicalMemory singleton at startup
    # This ensures the memory system (and its embedding model) is initialized once
    try:
        from vulcan.reasoning.singletons import get_hierarchical_memory
        hierarchical_memory = get_hierarchical_memory()
        if hierarchical_memory:
            logger.info("✓ HierarchicalMemory singleton preloaded at startup")
        else:
            logger.debug("HierarchicalMemory singleton not available (will load lazily)")
    except ImportError as e:
        logger.debug(f"HierarchicalMemory singleton not available for preload: {e}")
    except Exception as e:
        logger.warning(f"HierarchicalMemory preload failed (will load lazily): {e}")

    # PERF FIX Issue #5: Preload UnifiedLearningSystem singleton at startup
    # This ensures ensemble weights persist across requests (no more "All weights are zero")
    try:
        from vulcan.reasoning.singletons import get_unified_learning_system
        learning_system = get_unified_learning_system()
        if learning_system:
            logger.info("✓ UnifiedLearningSystem singleton preloaded at startup")
            logger.info("✓ Ensemble weights will persist across requests")
        else:
            logger.debug("UnifiedLearningSystem singleton not available (will load lazily)")
    except ImportError as e:
        logger.debug(f"UnifiedLearningSystem singleton not available for preload: {e}")
    except Exception as e:
        logger.warning(f"UnifiedLearningSystem preload failed (will load lazily): {e}")

    # ================================================================
    # PERFORMANCE FIX (Progressive Degradation): Pre-warm all reasoning singletons
    # This prevents query routing from degrading 469ms → 152,048ms over time
    # by ensuring all components (ToolSelector, BayesianMemoryPrior, WarmPool,
    # StochasticCostModel, SemanticToolMatcher, ProblemDecomposer) are created
    # exactly ONCE at startup.
    # ================================================================
    try:
        from vulcan.reasoning.singletons import prewarm_all
        prewarm_results = prewarm_all()
        success_count = sum(1 for v in prewarm_results.values() if v)
        logger.info(f"✓ Reasoning singletons pre-warmed: {success_count}/{len(prewarm_results)} components")
        
        # Validate ProblemDecomposer setup if available
        if prewarm_results.get('problem_decomposer'):
            try:
                from vulcan.problem_decomposer.decomposer_bootstrap import validate_decomposer_setup
                from vulcan.reasoning.singletons import get_problem_decomposer
                decomposer = get_problem_decomposer()
                if decomposer:
                    validation = validate_decomposer_setup(decomposer)
                    if validation['valid']:
                        logger.info(
                            f"  ✅ ProblemDecomposer validated: "
                            f"strategies={validation['checks'].get('strategy_count', 0)}, "
                            f"fallback_chain={validation['checks'].get('fallback_chain_count', 0)}"
                        )
                    else:
                        logger.warning(f"  ⚠️ ProblemDecomposer validation warnings: {validation.get('errors', [])}")
            except Exception as ve:
                logger.debug(f"ProblemDecomposer validation skipped: {ve}")
    except ImportError as e:
        logger.debug(f"Reasoning singletons module not available for prewarm: {e}")
    except Exception as e:
        logger.warning(f"Reasoning singletons prewarm failed: {e}")

    # PERFORMANCE FIX (Issue #55/#56): Preload ReasoningIntegration and embedding models at startup
    # This prevents the first query from taking 60+ seconds to load all models
    try:
        from vulcan.reasoning.integration import get_reasoning_integration
        reasoning_integration = get_reasoning_integration()
        # Force initialization of components (ToolSelector, PortfolioExecutor)
        reasoning_integration._init_components()
        logger.info("✓ ReasoningIntegration and ToolSelector preloaded at startup")
        
        # Also preload the SemanticToolMatcher's embedding model
        try:
            from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
            # Force model loading via the singleton pattern
            SemanticToolMatcher._get_shared_model()
            logger.info("✓ SemanticToolMatcher embedding model preloaded at startup")
        except Exception as e:
            logger.debug(f"SemanticToolMatcher preload skipped: {e}")
            
    except ImportError as e:
        logger.debug(f"ReasoningIntegration not available for preload: {e}")
    except Exception as e:
        logger.warning(f"ReasoningIntegration preload failed (will load on first request): {e}")

    # ================================================================
    # MEMORY MANAGEMENT: Start memory guard for automatic GC
    # This monitors memory pressure and triggers garbage collection
    # when usage exceeds threshold, preventing memory accumulation
    # from repeated model loading.
    # ================================================================
    try:
        from vulcan.monitoring.memory_guard import start_memory_guard
        memory_guard = start_memory_guard(threshold_percent=85.0, check_interval=5.0)
        if memory_guard:
            logger.info("✓ MemoryGuard started (threshold=85%, interval=5s)")
        else:
            logger.debug("MemoryGuard not started (psutil may not be available)")
    except ImportError as e:
        logger.debug(f"MemoryGuard module not available: {e}")
    except Exception as e:
        logger.warning(f"MemoryGuard startup failed: {e}")

    # Initialize SelfOptimizer for autonomous performance tuning
    if SELF_OPTIMIZER_AVAILABLE:
        try:
            app.state.self_optimizer = SelfOptimizer(
                target_latency_ms=100,
                target_memory_mb=2000,
                optimization_interval_s=60,
                enable_auto_tune=True
            )
            app.state.self_optimizer.start()
            logger.info("[VULCAN] ✓ SelfOptimizer started (target_latency=100ms, interval=60s)")
        except Exception as e:
            logger.warning(f"[VULCAN] Failed to initialize SelfOptimizer: {e}")
            app.state.self_optimizer = None
    else:
        app.state.self_optimizer = None

    try:
        yield
    except asyncio.CancelledError:
        logger.info(f"VULCAN-AGI worker {worker_id} received cancellation signal")
    finally:
        # SHUTDOWN LOGIC
        
        # HTTP CONNECTION POOL FIX: Close HTTP session first
        try:
            await close_http_session()
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")
        
        # Stop SelfOptimizer if running
        if hasattr(app.state, "self_optimizer") and app.state.self_optimizer:
            try:
                app.state.self_optimizer.stop()
                logger.info("[VULCAN] ✓ SelfOptimizer stopped")
            except Exception as e:
                logger.warning(f"[VULCAN] Error stopping SelfOptimizer: {e}")
        
        # Shutdown HardwareDispatcher if present in Arena
        if hasattr(app.state, 'arena') and hasattr(app.state.arena, 'hardware_dispatcher'):
            if app.state.arena.hardware_dispatcher:
                try:
                    app.state.arena.hardware_dispatcher.shutdown()
                    logger.info("[VULCAN] ✓ HardwareDispatcher shutdown complete")
                except Exception as e:
                    logger.warning(f"[VULCAN] Error shutting down HardwareDispatcher: {e}")

        if startup_complete and hasattr(app.state, "deployment"):
            deployment = app.state.deployment

            # Stop self-improvement drive if running
            try:
                world_model = deployment.collective.deps.world_model
                if world_model and hasattr(world_model, "stop_autonomous_improvement"):
                    world_model.stop_autonomous_improvement()
                    logger.info("🛑 Autonomous self-improvement drive stopped")
            except Exception as e:
                logger.error(f"Error stopping self-improvement: {e}")

            try:
                checkpoint_path = f"shutdown_checkpoint_{int(time.time())}.pkl"
                deployment.save_checkpoint(checkpoint_path)
                logger.info(f"Saved shutdown checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save shutdown checkpoint: {e}")

            if redis_client and hasattr(app.state, "worker_id"):
                try:
                    redis_client.delete(f"deployment:{app.state.worker_id}")
                except Exception as e:
                    logger.error(f"Failed to cleanup Redis: {e}")

        # Release process lock if held (split-brain prevention cleanup)
        # Always attempt release even if is_locked() returns False (state may be inconsistent)
        if _process_lock is not None:
            try:
                _process_lock.release()
                logger.info("Process lock released during shutdown")
            except Exception as e:
                logger.warning(f"Error releasing process lock during shutdown: {e}")

        logger.info("VULCAN-AGI API shutdown complete")


# ============================================================
# FastAPI Application with Enhanced Security
# ============================================================



def create_app(settings) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Application settings
        
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.api_title,
        description="Advanced Multimodal Collective Intelligence System with Autonomous Self-Improvement",
        version=settings.api_version,
        docs_url="/docs" if settings.deployment_mode != "production" else None,
        redoc_url="/redoc" if settings.deployment_mode != "production" else None,
        lifespan=lifespan,
    )
    
    @app.get("/", response_class=JSONResponse)
    async def root():
        return {"status": "ok", "message": "VULCAN-AGI API is alive"}
    
    # CORS middleware
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=settings.cors_methods,
            allow_headers=["*"],
        )
    
    # Mount safety router
    try:
        from vulcan.safety.safety_status_endpoint import router as safety_router
        app.include_router(safety_router, prefix="/safety", tags=["safety"])
        logger.info("Safety status endpoint mounted at /safety")
    except Exception as e:
        logger.error(f"Failed to mount safety status endpoint: {e}")
    
    return app
