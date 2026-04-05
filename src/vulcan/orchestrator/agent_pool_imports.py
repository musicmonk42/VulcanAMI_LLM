# ============================================================
# VULCAN-AGI Orchestrator - Agent Pool Lazy Imports
# Extracted from agent_pool.py for modularity
# Contains: lazy import functions for memory and reasoning modules
#           that must be deferred to avoid circular imports
# ============================================================

import logging

logger = logging.getLogger(__name__)


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
                logger.debug(
                    f"Memory components (WorkingMemory, HierarchicalMemory, MemoryConfig) "
                    f"loaded successfully via {specialized_path.rsplit('.', 1)[0]} (lazy import)"
                )
                return True

        except ImportError as e:
            logger.debug(
                f"Memory component import failed for path prefix "
                f"'{specialized_path.rsplit('.', 1)[0]}': {e}. Trying next path..."
            )
            continue

    # All paths failed
    logger.warning(
        "Memory components not available (all import paths failed). "
        "Provenance tracking will use fallback implementations."
    )
    return False


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

            logger.info(
                f"Reasoning integration loaded successfully via {reasoning_path} (lazy import) - reasoning engines will be invoked"
            )
            return True

        except ImportError as e:
            logger.debug(
                f"Import path {reasoning_path} failed: {e}. Trying next path..."
            )
            continue

    # All paths failed
    logger.warning(
        f"Reasoning integration not available (all import paths failed). Tasks will use placeholder execution."
    )
    REASONING_AVAILABLE = False
    return False


# ============================================================
# EXTERNAL OPTIONAL IMPORTS
# ============================================================

# Import TournamentManager for multi-agent selection
try:
    from src.tournament_manager import TournamentManager
    TOURNAMENT_MANAGER_AVAILABLE = True
except ImportError:
    TournamentManager = None
    TOURNAMENT_MANAGER_AVAILABLE = False
    logger.warning(
        "TournamentManager not available, multi-agent tournament selection will be disabled"
    )

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
        logger.warning(
            "ConsensusManager not available, distributed voting will be disabled"
        )

# Import enum conversion helper for reasoning_type safety
try:
    # ARCHITECTURE CONSOLIDATION: Import from unified compatibility layer via reasoning
    from vulcan.reasoning import ensure_reasoning_type_enum
    TYPE_CONVERSION_AVAILABLE = True
except ImportError:
    ensure_reasoning_type_enum = None
    TYPE_CONVERSION_AVAILABLE = False
    logger.warning(
        "[AgentPool] Type conversion utility not available - may drop philosophical results"
    )

# Import numpy with fallback for environments without it
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    logger.debug(
        "numpy not available, some advanced features will be disabled"
    )

# Import psutil with fallback for missing or broken installations
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logger.warning(
        "psutil not available, system resource monitoring will be disabled"
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Lazy import functions
    "_lazy_import_memory",
    "_lazy_import_reasoning",
    # Memory globals (populated by _lazy_import_memory)
    "WorkingMemory",
    "HierarchicalMemory",
    "MemoryConfig",
    # Reasoning globals (populated by _lazy_import_reasoning)
    "UnifiedReasoner",
    "ReasoningType",
    "ReasoningResult",
    "UNIFIED_AVAILABLE",
    "create_unified_reasoner",
    "apply_reasoning",
    "get_reasoning_integration",
    "IntegrationReasoningResult",
    "REASONING_AVAILABLE",
    # External optional imports
    "TournamentManager",
    "TOURNAMENT_MANAGER_AVAILABLE",
    "ConsensusManager",
    "CONSENSUS_MANAGER_AVAILABLE",
    "ensure_reasoning_type_enum",
    "TYPE_CONVERSION_AVAILABLE",
    "np",
    "NUMPY_AVAILABLE",
    "psutil",
    "PSUTIL_AVAILABLE",
]
