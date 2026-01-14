# ============================================================
# VULCAN-AGI Orchestrator Module
# Main collective orchestrator, dependencies, metrics, and deployment
# Enhanced with agent pool management, lifecycle controls, and distributed scaling
# FULLY DEBUGGED AND FIXED VERSION - All critical issues resolved
# REFACTORED INTO MODULAR STRUCTURE WITH NO CIRCULAR DEPENDENCIES
# INTEGRATED: Experiment generation and problem execution for self-improvement
# ============================================================

# CRITICAL: Fix Windows console encoding FIRST (before any imports that might log)
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        # Fallback for older Python or if reconfigure fails
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

"""
VULCAN-AGI Orchestrator

A modular orchestration system for AGI with:
- Agent pool management with lifecycle control
- Distributed task queues (Ray, Celery, ZMQ)
- Auto-scaling and recovery
- Comprehensive metrics collection
- Multiple orchestrator variants (Parallel, Fault-Tolerant, Adaptive)
- Production deployment with monitoring and checkpointing
- Autonomous self-improvement via experiment generation and execution

ARCHITECTURE:
    agent_lifecycle.py    - Agent states, capabilities, and job provenance
    task_queues.py        - Distributed task queue implementations
    agent_pool.py         - Agent pool management with auto-scaling
    metrics.py            - Comprehensive metrics collection
    dependencies.py       - Dependencies container with validation
    collective.py         - Main orchestrator with cognitive cycle
    variants.py           - Specialized orchestrator variants
    deployment.py         - Production deployment with monitoring

FIXED ISSUES (Comprehensive):
1. Circular Dependencies:
   - Removed all imports from config.py
   - Defined ModalityType and ActionType locally in collective.py
   - All modules now self-contained

2. Memory Management:
   - All histograms bounded with maxlen
   - All timeseries bounded with maxlen
   - TTL cache for provenance records
   - Automatic cleanup of old data
   - Bounded reasoning traces and execution history

3. Race Conditions:
   - Atomic agent assignment operations
   - Proper locking in all critical sections
   - State machine validation for agent transitions
   - Thread-safe operations throughout

4. Resource Leaks:
   - Proper process/thread cleanup with timeouts
   - Executor shutdown with wait
   - File handle management
   - Garbage collection on shutdown

5. Error Handling:
   - Comprehensive try-except blocks
   - Phase isolation in cognitive cycle
   - Fallback mechanisms at all levels
   - Graceful degradation

6. API Fixes:
   - MemoryIndex uses add() method (not upsert)
   - EpisodicMemory uses add_episode() (not append)
   - EnhancedCausalReasoning (was CausalReasoningEngine)
   - AnalogicalReasoner (was AbstractReasoner)

VERSION HISTORY:
    1.0.0 - Initial production release with all fixes
    1.0.1 - Fixed Unicode emoji encoding for cross-platform compatibility
    1.0.2 - Integrated experiment generation and problem execution for self-improvement

REQUIREMENTS:
    - Python 3.8+
    - numpy
    - Optional: ray, celery, pyzmq, cachetools, psutil, torch
"""

import logging
import sys

# Setup logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.2"
__author__ = "VULCAN-AGI Team"
__status__ = "Production"
__license__ = "MIT"

# ============================================================
# UNICODE SAFE PRINTING
# ============================================================


def safe_print(text: str):
    """
    Print text with fallback for Unicode characters

    Args:
        text: Text to print
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace Unicode characters with ASCII equivalents
        safe_text = text.encode("ascii", "replace").decode("ascii")
        print(safe_text)


def get_status_symbol(status: bool) -> str:
    """
    Get status symbol (Unicode with ASCII fallback)

    Args:
        status: True for success, False for failure

    Returns:
        Status symbol string
    """
    try:
        # Try Unicode symbols
        return "✓" if status else "✗"
    except UnicodeEncodeError:
        # Fallback to ASCII
        return "[OK]" if status else "[FAIL]"


def get_bullet_symbol() -> str:
    """
    Get bullet point symbol (Unicode with ASCII fallback)

    Returns:
        Bullet symbol string
    """
    try:
        # Try Unicode bullet
        return "•"
    except UnicodeEncodeError:
        # Fallback to ASCII
        return "*"


# ============================================================
# CORE IMPORTS
# ============================================================

try:
    # Agent lifecycle components
    from .agent_lifecycle import (
        AgentCapability,
        AgentMetadata,
        AgentState,
        JobProvenance,
        StateTransitionRules,
        create_agent_metadata,
        create_job_provenance,
        validate_state_machine,
    )

    # Agent pool management
    from .agent_pool import (
        AgentPoolManager,
        AGENT_SELECTION_TIMEOUT_SECONDS,
    )
    
    # Auto-scaling and recovery (now in deployment module)
    from .deployment import (
        AutoScaler,
        RecoveryManager,
    )

    # Main orchestrator
    from .collective import ActionType, ModalityType, VULCANAGICollective

    # Dependencies container
    from .dependencies import (
        DependencyCategory,
        EnhancedCollectiveDeps,
        create_full_deps,
        create_minimal_deps,
        print_dependency_report,
        validate_dependencies,
    )

    # Production deployment
    from .deployment import ProductionDeployment

    # Metrics collection
    from .metrics import (
        AggregationType,
        EnhancedMetricsCollector,
        MetricType,
        compute_moving_average,
        compute_percentile,
        compute_rate,
        create_metrics_collector,
        ResponseTimeTracker,
        SystemMetrics,
    )

    # Task queue implementations
    from .task_queues import (
        CELERY_AVAILABLE,
        RAY_AVAILABLE,
        ZMQ_AVAILABLE,
        CeleryTaskQueue,
        CustomTaskQueue,
        QueueType,
        RayTaskQueue,
        TaskMetadata,
        TaskQueueInterface,
        TaskStatus,
        create_task_queue,
        PriorityJobQueue,
    )

    # Orchestrator variants
    from .variants import (
        AdaptiveOrchestrator,
        ExecutionError,
        FaultTolerantOrchestrator,
        ParallelOrchestrator,
        PerceptionError,
        PerformanceMonitor,
        ReasoningError,
        StrategySelector,
    )

    _imports_successful = True

except ImportError as e:
    logger.error(f"Failed to import orchestrator components: {e}")
    _imports_successful = False
    raise


# ============================================================
# EXPERIMENT GENERATION AND PROBLEM EXECUTION IMPORTS
# ============================================================

# Track availability of experiment/execution components
EXPERIMENT_GENERATOR_AVAILABLE = False
PROBLEM_EXECUTOR_AVAILABLE = False

try:
    # Import experiment generation components
    from ..curiosity_engine.experiment_generator import (
        Experiment,
        ExperimentGenerator,
        ExperimentType,
        KnowledgeGap,
    )

    EXPERIMENT_GENERATOR_AVAILABLE = True
    logger.info("ExperimentGenerator components loaded successfully")

except ImportError as e:
    logger.warning(f"Failed to import ExperimentGenerator components: {e}")
    logger.warning("Self-improvement experiments will not be available")
    # Define stub classes to prevent import errors
    ExperimentGenerator = None
    ExperimentType = None
    Experiment = None
    KnowledgeGap = None

try:
    # Import problem execution components
    from ..problem_decomposer.problem_executor import (
        ExecutionStrategy,
        ProblemExecutor,
        SolutionType,
    )

    PROBLEM_EXECUTOR_AVAILABLE = True
    logger.info("ProblemExecutor components loaded successfully")

except ImportError as e:
    logger.warning(f"Failed to import ProblemExecutor components: {e}")
    logger.warning("Self-improvement execution will not be available")
    # Define stub classes to prevent import errors
    ProblemExecutor = None
    SolutionType = None
    ExecutionStrategy = None

# Log combined status
if EXPERIMENT_GENERATOR_AVAILABLE and PROBLEM_EXECUTOR_AVAILABLE:
    logger.info("Self-improvement system fully available")
elif EXPERIMENT_GENERATOR_AVAILABLE or PROBLEM_EXECUTOR_AVAILABLE:
    logger.warning("Self-improvement system partially available")
else:
    logger.warning("Self-improvement system not available")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__status__",
    "__license__",
    # Agent Lifecycle
    "AgentState",
    "AgentCapability",
    "AgentMetadata",
    "JobProvenance",
    "StateTransitionRules",
    "create_agent_metadata",
    "create_job_provenance",
    "validate_state_machine",
    # Task Queues
    "TaskQueueInterface",
    "RayTaskQueue",
    "CeleryTaskQueue",
    "CustomTaskQueue",
    "TaskStatus",
    "TaskMetadata",
    "QueueType",
    "create_task_queue",
    "RAY_AVAILABLE",
    "CELERY_AVAILABLE",
    "ZMQ_AVAILABLE",
    "PriorityJobQueue",
    # Agent Pool Management
    "AgentPoolManager",
    "AutoScaler",
    "RecoveryManager",
    "AGENT_SELECTION_TIMEOUT_SECONDS",  # Note: Agent selection timeout constant
    # Metrics
    "EnhancedMetricsCollector",
    "MetricType",
    "AggregationType",
    "create_metrics_collector",
    "compute_percentile",
    "compute_moving_average",
    "compute_rate",
    "ResponseTimeTracker",
    "SystemMetrics",
    # Dependencies
    "EnhancedCollectiveDeps",
    "DependencyCategory",
    "create_minimal_deps",
    "create_full_deps",
    "validate_dependencies",
    "print_dependency_report",
    # Main Orchestrator
    "VULCANAGICollective",
    "ModalityType",
    "ActionType",
    # Variants
    "ParallelOrchestrator",
    "FaultTolerantOrchestrator",
    "AdaptiveOrchestrator",
    "PerformanceMonitor",
    "StrategySelector",
    "PerceptionError",
    "ReasoningError",
    "ExecutionError",
    # Production
    "ProductionDeployment",
    # Experiment Generation & Problem Execution
    "ExperimentGenerator",
    "ExperimentType",
    "Experiment",
    "KnowledgeGap",
    "ProblemExecutor",
    "SolutionType",
    "ExecutionStrategy",
    "EXPERIMENT_GENERATOR_AVAILABLE",
    "PROBLEM_EXECUTOR_AVAILABLE",
    # Utility functions
    "safe_print",
    "get_status_symbol",
    "get_bullet_symbol",
]


# ============================================================
# MODULE INITIALIZATION
# ============================================================


def get_module_info() -> dict:
    """
    Get information about the orchestrator module

    Returns:
        Dictionary with module information
    """
    return {
        "version": __version__,
        "author": __author__,
        "status": __status__,
        "license": __license__,
        "imports_successful": _imports_successful,
        "python_version": sys.version,
        "optional_dependencies": {
            "ray": RAY_AVAILABLE,
            "celery": CELERY_AVAILABLE,
            "zmq": ZMQ_AVAILABLE,
        },
        "self_improvement": {
            "experiment_generator": EXPERIMENT_GENERATOR_AVAILABLE,
            "problem_executor": PROBLEM_EXECUTOR_AVAILABLE,
            "fully_available": EXPERIMENT_GENERATOR_AVAILABLE
            and PROBLEM_EXECUTOR_AVAILABLE,
        },
    }


def print_module_info():
    """Print module information to stdout with Unicode-safe formatting"""
    info = get_module_info()

    safe_print("\n" + "=" * 70)
    safe_print("VULCAN-AGI ORCHESTRATOR MODULE")
    safe_print("=" * 70)
    safe_print(f"Version:        {info['version']}")
    safe_print(f"Status:         {info['status']}")
    safe_print(f"Author:         {info['author']}")
    safe_print(f"License:        {info['license']}")
    safe_print(f"Python:         {info['python_version']}")

    success_symbol = get_status_symbol(info["imports_successful"])
    status_text = "Success" if info["imports_successful"] else "Failed"
    safe_print(f"\nImports:        {success_symbol} {status_text}")

    safe_print("\nOptional Dependencies:")
    for dep, available in info["optional_dependencies"].items():
        status_symbol = get_status_symbol(available)
        status_text = "Available" if available else "Not Available"
        safe_print(f"  {dep:12s} {status_symbol} {status_text}")

    safe_print("\nSelf-Improvement Components:")
    si = info["self_improvement"]
    exp_symbol = get_status_symbol(si["experiment_generator"])
    exp_status = "Available" if si["experiment_generator"] else "Not Available"
    safe_print(f"  ExperimentGen {exp_symbol} {exp_status}")

    prob_symbol = get_status_symbol(si["problem_executor"])
    prob_status = "Available" if si["problem_executor"] else "Not Available"
    safe_print(f"  ProblemExec   {prob_symbol} {prob_status}")

    si_symbol = get_status_symbol(si["fully_available"])
    si_status = (
        "Fully Available"
        if si["fully_available"]
        else (
            "Partially Available"
            if (si["experiment_generator"] or si["problem_executor"])
            else "Not Available"
        )
    )
    safe_print(f"  Self-Improve  {si_symbol} {si_status}")

    bullet = get_bullet_symbol()
    safe_print("\nKey Features:")
    safe_print(f"  {bullet} Agent pool management with lifecycle control")
    safe_print(f"  {bullet} Distributed task queues (Ray, Celery, ZMQ)")
    safe_print(f"  {bullet} Auto-scaling and recovery")
    safe_print(f"  {bullet} Comprehensive metrics collection")
    safe_print(f"  {bullet} Multiple orchestrator variants")
    safe_print(f"  {bullet} Production deployment with monitoring")
    safe_print(f"  {bullet} Autonomous self-improvement via experiments")

    safe_print("\nFixed Issues:")
    safe_print(f"  {bullet} No circular dependencies")
    safe_print(f"  {bullet} Bounded memory usage")
    safe_print(f"  {bullet} Race condition free")
    safe_print(f"  {bullet} Proper resource cleanup")
    safe_print(f"  {bullet} Comprehensive error handling")

    safe_print("=" * 70 + "\n")


def validate_installation() -> bool:
    """
    Validate that the orchestrator module is properly installed

    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check core imports
        assert _imports_successful, "Core imports failed"

        # Validate state machine
        validate_state_machine()

        # Test factory functions
        metrics = create_metrics_collector()
        assert metrics is not None, "Failed to create metrics collector"

        deps = create_minimal_deps()
        assert deps is not None, "Failed to create minimal deps"

        # Test task queue factory
        if ZMQ_AVAILABLE:
            queue = create_task_queue("custom")
            assert queue is not None, "Failed to create custom task queue"

        # Check self-improvement components (optional, just log warnings)
        if not EXPERIMENT_GENERATOR_AVAILABLE:
            logger.warning(
                "ExperimentGenerator not available - self-improvement experiments disabled"
            )

        if not PROBLEM_EXECUTOR_AVAILABLE:
            logger.warning(
                "ProblemExecutor not available - self-improvement execution disabled"
            )

        logger.info("Module validation passed")
        return True

    except Exception as e:
        logger.error(f"Module validation failed: {e}")
        return False


# ============================================================
# MODULE-LEVEL INITIALIZATION
# ============================================================

# Log successful import with Unicode-safe symbols
if _imports_successful:
    logger.info("VULCAN-AGI Orchestrator module loaded successfully")
    logger.info(f"Version: {__version__}")
    logger.info("All critical fixes applied:")

    # Use ASCII-safe logging
    try:
        check = "✓"
        logger.info(f"  {check} No circular dependencies")
        logger.info(f"  {check} Bounded memory usage")
        logger.info(f"  {check} Thread-safe operations")
        logger.info(f"  {check} Proper resource cleanup")
        logger.info(f"  {check} Comprehensive error handling")
        logger.info(f"  {check} State machine validation")
    except UnicodeEncodeError:
        # Fallback to ASCII
        logger.info("  [OK] No circular dependencies")
        logger.info("  [OK] Bounded memory usage")
        logger.info("  [OK] Thread-safe operations")
        logger.info("  [OK] Proper resource cleanup")
        logger.info("  [OK] Comprehensive error handling")
        logger.info("  [OK] State machine validation")

    # Log optional dependencies
    try:
        check = "✓"
        if RAY_AVAILABLE:
            logger.info(f"  {check} Ray support enabled")
        if CELERY_AVAILABLE:
            logger.info(f"  {check} Celery support enabled")
        if ZMQ_AVAILABLE:
            logger.info(f"  {check} ZeroMQ support enabled")
        if EXPERIMENT_GENERATOR_AVAILABLE and PROBLEM_EXECUTOR_AVAILABLE:
            logger.info(f"  {check} Self-improvement system enabled")
    except UnicodeEncodeError:
        # Fallback to ASCII
        if RAY_AVAILABLE:
            logger.info("  [OK] Ray support enabled")
        if CELERY_AVAILABLE:
            logger.info("  [OK] Celery support enabled")
        if ZMQ_AVAILABLE:
            logger.info("  [OK] ZeroMQ support enabled")
        if EXPERIMENT_GENERATOR_AVAILABLE and PROBLEM_EXECUTOR_AVAILABLE:
            logger.info("  [OK] Self-improvement system enabled")
else:
    logger.error("VULCAN-AGI Orchestrator module failed to load")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def create_production_deployment(
    config, checkpoint_path=None, orchestrator_type="parallel", redis_client=None
):
    """
    Convenience function to create a production deployment

    Args:
        config: Configuration object
        checkpoint_path: Optional checkpoint path
        orchestrator_type: Type of orchestrator ('parallel', 'adaptive', 'fault_tolerant', 'basic')
        redis_client: Optional Redis client for state persistence across workers/restarts

    Returns:
        ProductionDeployment instance
    """
    return ProductionDeployment(config, checkpoint_path, orchestrator_type, redis_client=redis_client)


def create_orchestrator(config, sys, deps, variant="basic", redis_client=None):
    """
    Convenience function to create an orchestrator

    Args:
        config: Configuration object
        sys: System state object
        deps: Dependencies container
        variant: Orchestrator variant ('basic', 'parallel', 'adaptive', 'fault_tolerant')
        redis_client: Optional Redis client for state persistence across workers/restarts

    Returns:
        Orchestrator instance
    """
    if variant == "parallel":
        return ParallelOrchestrator(config, sys, deps, redis_client=redis_client)
    elif variant == "adaptive":
        return AdaptiveOrchestrator(config, sys, deps, redis_client=redis_client)
    elif variant == "fault_tolerant":
        return FaultTolerantOrchestrator(config, sys, deps, redis_client=redis_client)
    else:
        return VULCANAGICollective(config, sys, deps, redis_client=redis_client)


def create_agent_pool(max_agents=10, min_agents=2, task_queue_type="custom", redis_client=None):
    """
    Convenience function to create an agent pool

    Args:
        max_agents: Maximum number of agents (default: 10, reduced from 100 for CPU optimization)
        min_agents: Minimum number of agents (default: 2, reduced from 5 for CPU cloud optimization)
        task_queue_type: Type of task queue ('ray', 'celery', 'custom')
        redis_client: Optional Redis client for state persistence across workers/restarts

    Returns:
        AgentPoolManager instance
    
    Note:
        CPU CLOUD FIX: min_agents reduced from 5 to 2 to reduce context-switching
        overhead on CPU-only cloud instances.
    """
    return AgentPoolManager(max_agents, min_agents, task_queue_type, redis_client=redis_client)


def create_experiment_generator(config=None):
    """
    Convenience function to create an experiment generator

    Args:
        config: Optional configuration

    Returns:
        ExperimentGenerator instance or None if not available
    """
    if not EXPERIMENT_GENERATOR_AVAILABLE:
        logger.warning("ExperimentGenerator not available")
        return None

    try:
        return ExperimentGenerator(config)
    except Exception as e:
        logger.error(f"Failed to create ExperimentGenerator: {e}")
        return None


def create_problem_executor(config=None):
    """
    Convenience function to create a problem executor

    Args:
        config: Optional configuration

    Returns:
        ProblemExecutor instance or None if not available
    """
    if not PROBLEM_EXECUTOR_AVAILABLE:
        logger.warning("ProblemExecutor not available")
        return None

    try:
        return ProblemExecutor(config)
    except Exception as e:
        logger.error(f"Failed to create ProblemExecutor: {e}")
        return None


# ============================================================
# MODULE DOCUMENTATION
# ============================================================

__doc__ = """
VULCAN-AGI Orchestrator Module

This module provides a comprehensive orchestration system for AGI with the following components:

Components:
    agent_lifecycle    - Agent lifecycle management with state machine
    task_queues        - Distributed task queue implementations
    agent_pool         - Agent pool with auto-scaling and recovery
    metrics            - Comprehensive metrics collection
    dependencies       - Dependency injection container
    collective         - Main orchestrator with cognitive cycle
    variants           - Specialized orchestrator variants
    deployment         - Production deployment utilities

Self-Improvement:
    experiment_generator - Generate experiments to fill knowledge gaps
    problem_executor     - Execute problem-solving plans

Quick Start:
    >>> from orchestrator import ProductionDeployment, create_minimal_deps
    >>>
    >>> # Create configuration
    >>> config = Config()
    >>>
    >>> # Create production deployment
    >>> deployment = ProductionDeployment(config, orchestrator_type="parallel")
    >>>
    >>> # Execute steps
    >>> result = deployment.step_with_monitoring(history=[], context={})
    >>>
    >>> # Get status
    >>> status = deployment.get_status()
    >>>
    >>> # Shutdown
    >>> deployment.shutdown()

Usage Examples:

    1. Basic Orchestrator:
        >>> from orchestrator import create_orchestrator, create_minimal_deps
        >>> deps = create_minimal_deps()
        >>> orchestrator = create_orchestrator(config, sys, deps, variant="basic")
        >>> result = orchestrator.step(history, context)

    2. Parallel Orchestrator:
        >>> orchestrator = create_orchestrator(config, sys, deps, variant="parallel")
        >>> result = await orchestrator.step_parallel(history, context)

    3. Adaptive Orchestrator:
        >>> orchestrator = create_orchestrator(config, sys, deps, variant="adaptive")
        >>> result = orchestrator.adaptive_step(history, context)

    4. Agent Pool:
        >>> from orchestrator import create_agent_pool
        >>> pool = create_agent_pool(max_agents=10, min_agents=5)
        >>> job_id = pool.submit_job(graph, parameters, priority=0)
        >>> result = pool.get_job_provenance(job_id)

    5. Self-Improvement (if available):
        >>> from orchestrator import create_experiment_generator, create_problem_executor
        >>> exp_gen = create_experiment_generator()
        >>> executor = create_problem_executor()
        >>> if exp_gen and executor:
        >>>     gap = KnowledgeGap(type='optimization', domain='system', priority=1.0)
        >>>     experiments = exp_gen.generate_for_gap(gap)
        >>>     for exp in experiments:
        >>>         result = executor.execute_plan(exp.graph, exp.plan)

For more information, see the documentation for individual modules.
"""


# Add convenience functions to __all__
__all__.extend(
    [
        "get_module_info",
        "print_module_info",
        "validate_installation",
        "create_production_deployment",
        "create_orchestrator",
        "create_agent_pool",
        "create_experiment_generator",
        "create_problem_executor",
    ]
)


# ============================================================
# SUBMODULE DOCUMENTATION
# ============================================================

# This allows users to access module documentation easily
_module_docs = {
    "agent_lifecycle": "Agent states, capabilities, metadata, and job provenance tracking",
    "task_queues": "Distributed task queue implementations (Ray, Celery, ZMQ)",
    "agent_pool": "Agent pool management with auto-scaling and recovery",
    "metrics": "Comprehensive metrics collection and monitoring",
    "dependencies": "Dependency injection container with validation",
    "collective": "Main orchestrator with full cognitive cycle",
    "variants": "Specialized orchestrator variants (Parallel, Adaptive, FaultTolerant)",
    "deployment": "Production deployment with monitoring and checkpointing",
    "experiment_generator": "Generate experiments to fill knowledge gaps (self-improvement)",
    "problem_executor": "Execute problem-solving plans (self-improvement)",
}


def get_submodule_docs() -> dict:
    """Get documentation for all submodules"""
    return _module_docs


def print_submodule_docs():
    """Print documentation for all submodules with Unicode-safe formatting"""
    safe_print("\n" + "=" * 70)
    safe_print("SUBMODULE DOCUMENTATION")
    safe_print("=" * 70)
    for module, doc in _module_docs.items():
        safe_print(f"\n{module}:")
        safe_print(f"  {doc}")
    safe_print("\n" + "=" * 70 + "\n")


# Add to __all__
__all__.extend(["get_submodule_docs", "print_submodule_docs"])
