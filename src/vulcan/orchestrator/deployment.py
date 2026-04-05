# ============================================================
# VULCAN-AGI Orchestrator - Deployment Module
# Production-ready deployment with monitoring, persistence, and agent pool
# FULLY FIXED VERSION - Enhanced with proper error handling, validation, and cleanup
# Added orchestrator type validation for better error messages
# FIXED: Memory component initialization with proper config handling
# FIXED: SafetyPolicies object handling in _create_system_state
# FIXED: Import paths for safety, reasoning, and learning components to avoid circular imports
# FIXED: Reasoning/Learning imports to use package-level imports with graceful fallbacks
# FIXED: Safety validator config handling to avoid AgentConfig.to_dict() error
# FIXED: step_with_monitoring to check for missing governance methods before calling
# FIXED: Windows checkpoint file locking with atomic write pattern and retry logic
# FIXED: Path handling in atomic_write_with_retry for proper directory creation and string conversion
# FIXED: Windows checkpoint race during parallel tests with locking and step tracking
# FIXED: Analogical reasoning now imports from modular analogical subpackage
# FIXED: MultimodalProcessor import path (vulcan.processing not processing)
# FIXED: Planning imports path (vulcan.planning not planning)
# FIXED: Multimodal processor key alignment in _load_reasoners
# FIXED: UnifiedRuntime import (unified_runtime not src.unified_runtime)
# FIXED: Refactored checkpoint locking to avoid nested RLock acquisition.
# SECURITY: Replaced pickle.load with safe_pickle_load to prevent deserialization attacks
# ============================================================

import asyncio
import atexit
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from .agent_lifecycle import AgentState
from .collective import VULCANAGICollective
from .dependencies import (
    EnhancedCollectiveDeps,
    print_dependency_report,
    validate_dependencies,
)
from .metrics import EnhancedMetricsCollector
from .variants import (
    AdaptiveOrchestrator,
    FaultTolerantOrchestrator,
    ParallelOrchestrator,
)
from ..security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)


# ============================================================
# WINDOWS FILE HANDLING UTILITIES
# ============================================================


def atomic_write_with_retry(
    data: bytes, target_path: str, max_retries: int = 5, retry_delay: float = 0.1
) -> bool:
    """
    Atomic file write with Windows-compatible retry logic and exponential backoff with jitter.
    
    ISSUE 4 FIX: Enhanced atomic write pattern for Windows reliability:
    - Writes to temp file in same directory (same filesystem)
    - Uses os.fsync() to ensure data hits disk
    - Handles Windows-specific file locking (antivirus, cloud sync)
    - Uses exponential backoff with jitter to avoid thundering herd
    - Uses os.replace() for atomic rename on Windows

    Handles Windows file locking issues by:
    - Writing to temporary file first
    - Properly closing file handles
    - Implementing exponential backoff with jitter
    - Cleaning up on failure
    - Using os.replace for Windows-safe atomic operations

    Args:
        data: Binary data to write
        target_path: Target file path
        max_retries: Maximum retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 0.1)

    Returns:
        True if successful, False otherwise
        
    Raises:
        PermissionError: If file cannot be written after max retries
        OSError: For other file system errors
    """
    import random
    import tempfile

    # FIXED: Convert to absolute path immediately and ensure parent exists
    target_path_obj = Path(target_path).resolve()
    target_path_obj.parent.mkdir(parents=True, exist_ok=True)

    temp_fd = None
    temp_path = None

    try:
        # Create temporary file in same directory for atomic operation
        # ISSUE 4 FIX: Same directory = same filesystem = atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(target_path_obj.parent),
            prefix=".tmp_checkpoint_",
            suffix=".pkl",
        )

        # Write data to temporary file
        try:
            os.write(temp_fd, data)
            # ISSUE 4 FIX: Ensure data hits disk before rename
            os.fsync(temp_fd)
        finally:
            # Always close the file descriptor
            os.close(temp_fd)
            temp_fd = None

        # Attempt atomic rename with exponential backoff and jitter
        for attempt in range(max_retries):
            try:
                # ISSUE 4 FIX: os.replace works atomically on both Unix and Windows
                # On Windows, it handles the "remove before rename" requirement internally
                os.replace(temp_path, str(target_path_obj))

                # Success!
                logger.debug(
                    f"Atomic write successful on attempt {attempt + 1}: {target_path_obj}"
                )
                return True

            except PermissionError as e:
                if attempt < max_retries - 1:
                    # ISSUE 4 FIX: Exponential backoff with jitter
                    # Base delay doubles each retry: 0.1, 0.2, 0.4, 0.8, 1.6
                    # Jitter adds randomness (±20%) to prevent thundering herd
                    base_delay = retry_delay * (2 ** attempt)
                    jitter = base_delay * 0.2 * (2 * random.random() - 1)  # ±20%
                    actual_delay = base_delay + jitter
                    
                    logger.debug(
                        f"File locked on attempt {attempt + 1}/{max_retries}, "
                        f"retrying in {actual_delay:.3f}s (base={base_delay:.3f}s, jitter={jitter:+.3f}s): {e}"
                    )
                    time.sleep(actual_delay)
                else:
                    logger.error(
                        f"Failed to write file after {max_retries} attempts: {e}"
                    )
                    # Re-raise the exception after max retries
                    raise PermissionError(
                        f"Failed to replace file {target_path_obj} after {max_retries} attempts. "
                        f"File may be locked by antivirus or cloud sync. Error: {e}"
                    ) from e
            except OSError as e:
                # Other OS errors (disk full, invalid path, etc.)
                logger.error(f"OS error during atomic write: {e}")
                raise OSError(
                    f"Failed to write file {target_path_obj}: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error during atomic write: {e}")
                raise  # Re-raise other unexpected exceptions

        # Should not be reached if PermissionError is raised properly after retries
        return False

    except Exception as e:
        logger.error(f"Failed to perform atomic write to {target_path_obj}: {e}", exc_info=True)
        return False

    finally:
        # Cleanup: Close file descriptor if still open
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except Exception as e:
                logger.debug(f"Operation failed: {e}")

        # Cleanup: Remove temporary file if it still exists
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception as e:
                logger.debug(f"Failed to cleanup temporary file {temp_path}: {e}")


# ============================================================
# PRODUCTION DEPLOYMENT
# ============================================================


class ProductionDeployment:
    """
    Production-ready deployment with monitoring, persistence, and agent pool

    Features:
    - Multiple orchestrator variants (Parallel, Adaptive, FaultTolerant)
    - Comprehensive health checks
    - Automatic checkpointing with Windows-compatible file handling
    - Thread-safe checkpoint operations with race condition prevention
    - Metrics collection and monitoring
    - Graceful shutdown
    - Dependency validation
    - Orchestrator type validation
    """

    def __init__(
        self,
        config: Any,
        checkpoint_path: Optional[str] = None,
        orchestrator_type: str = "parallel",
        redis_client: Optional[Any] = None,
    ):
        """
        Initialize Production Deployment

        Args:
            config: Configuration object
            checkpoint_path: Path to load checkpoint from
            orchestrator_type: Type of orchestrator ('parallel', 'adaptive', 'fault_tolerant', 'basic')
            redis_client: Optional Redis client for state persistence across workers/restarts
        """
        self.config = config
        self.collective: Optional[VULCANAGICollective] = None
        self.metrics_collector = EnhancedMetricsCollector()
        self.checkpoint_path = checkpoint_path
        self.unified_runtime = None
        self.startup_time = time.time()
        self.orchestrator_type = orchestrator_type
        self.redis_client = redis_client
        self._shutdown_requested = False
        
        # SAFETY FIX: Register cleanup with atexit instead of __del__
        # This ensures deterministic cleanup before interpreter shutdown,
        # avoiding NameError/ImportError when global variables are destroyed.
        atexit.register(self._atexit_cleanup)

        # FIXED: Add checkpoint locking and step tracking to prevent race conditions
        self._checkpoint_lock = RLock()
        self._last_checkpointed_step = -1

        # Checkpoint directory
        self.checkpoint_dir = Path(getattr(config, "checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Try to initialize UnifiedRuntime if available
        # FIXED: Import from unified_runtime not src.unified_runtime
        try:
            from unified_runtime import UnifiedRuntime

            # ADD THIS CHECK
            if getattr(self.config, "disable_unified_runtime", False):
                self.unified_runtime = None
                logger.info("UnifiedRuntime explicitly disabled by config")
            else:
                # Note Issue #27: Use singleton to prevent per-query manifest reloading
                # Note: Use get_or_create_unified_runtime to prevent repeated init/shutdown
                try:
                    from vulcan.reasoning.singletons import get_or_create_unified_runtime
                    self.unified_runtime = get_or_create_unified_runtime()
                    if self.unified_runtime:
                        logger.info("UnifiedRuntime obtained via singleton")
                    else:
                        logger.warning("UnifiedRuntime not available via singleton")
                except ImportError:
                    self.unified_runtime = UnifiedRuntime()
                    logger.info("UnifiedRuntime initialized directly")
        except ImportError:
            logger.info("UnifiedRuntime not available, using internal orchestrator")
            self.unified_runtime = None
        except Exception as e:
            logger.warning(f"Failed to initialize UnifiedRuntime: {e}")
            self.unified_runtime = None

        # Initialize the system
        self.initialize(checkpoint_path)

        logger.info(
            f"ProductionDeployment initialized (orchestrator: {orchestrator_type})"
        )

    def initialize(self, checkpoint_path: Optional[str]):
        """
        Initialize all components

        Args:
            checkpoint_path: Optional path to checkpoint file
        """
        logger.info("Initializing VULCAN-AGI Production Deployment")

        try:
            # Import modules with error handling
            components = self._import_components()
            
            # Store components for external access (e.g., CuriosityEngine)
            # This enables full_platform.py to access curiosity_engine directly
            self._components = components

            # Create dependencies
            deps = self._create_dependencies(components)

            # Validate dependencies
            validation_passed = validate_dependencies(deps)
            if not validation_passed:
                logger.warning("Some dependencies are missing, printing report:")
                print_dependency_report(deps)

            # Initialize system state
            system_state = self._create_system_state()

            # Create collective with specified orchestrator type
            self.collective = self._create_orchestrator(
                self.orchestrator_type, system_state, deps
            )

            # Load checkpoint if provided
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)

            logger.info(f"System initialized with CID: {system_state.CID}")
            logger.info(
                f"Agent pool status: {self.collective.agent_pool.get_pool_status()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize system: {e}", exc_info=True)
            raise
    
    @property
    def curiosity_engine(self):
        """
        Get the CuriosityEngine instance.
        
        This property provides access to the CuriosityEngine for the 
        CuriosityDriver initialization in full_platform.py.
        
        Returns:
            CuriosityEngine instance or None if not available
        """
        if hasattr(self, '_components') and self._components:
            return self._components.get("curiosity_engine")
        return None
    
    @property
    def problem_decomposer(self):
        """
        Get the ProblemDecomposer instance.
        
        Returns:
            ProblemDecomposer instance or None if not available
        """
        if hasattr(self, '_components') and self._components:
            return self._components.get("problem_decomposer")
        return None
    
    @property
    def semantic_bridge(self):
        """
        Get the SemanticBridge instance.
        
        Returns:
            SemanticBridge instance or None if not available
        """
        if hasattr(self, '_components') and self._components:
            return self._components.get("semantic_bridge")
        return None

    def _load_reasoners(self, components: Dict[str, Any]):
        """Loads all available reasoning components with lazy imports and error handling."""
        logger.info("Loading reasoning components...")

        # --- Load Symbolic Reasoner ---
        try:
            from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner

            components["symbolic"] = SymbolicReasoner()
            logger.info("SymbolicReasoner loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load SymbolicReasoner: {e}")

        # --- Load Multimodal Reasoner ---
        try:
            from vulcan.reasoning.multimodal_reasoning import MultimodalReasoner

            components["multimodal_reasoner"] = MultimodalReasoner()
            logger.info("MultimodalReasoner loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load MultimodalReasoner: {e}")

        # --- Load Probabilistic Reasoner ---
        try:
            from vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner

            components["probabilistic"] = ProbabilisticReasoner()
            logger.info("ProbabilisticReasoner loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load ProbabilisticReasoner: {e}")

        # --- Load Causal Reasoner ---
        try:
            # Assuming path based on project structure
            from vulcan.reasoning.causal_reasoning import EnhancedCausalReasoning

            components["causal"] = EnhancedCausalReasoning()
            logger.info("EnhancedCausalReasoning loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load EnhancedCausalReasoning: {e}")

        # --- Load Analogical (Abstract) Reasoner ---
        try:
            from vulcan.reasoning.analogical import AnalogicalReasoner

            components["abstract"] = AnalogicalReasoner()
            logger.info("AnalogicalReasoner loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load AnalogicalReasoner: {e}")

        # --- Load Cross-Modal Reasoner ---
        try:
            from vulcan.reasoning.multimodal_reasoning import CrossModalReasoner

            # FIXED: Use correct key 'multimodal' not 'multimodal_processor'
            multimodal_processor = components.get("multimodal")
            components["cross_modal"] = CrossModalReasoner(multimodal_processor)
            logger.info("CrossModalReasoner loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load CrossModalReasoner: {e}")

        # --- INTEGRATION FIX: Load UnifiedReasoner with ToolSelector ---
        # This is the main reasoning orchestrator that intelligently selects the best
        # reasoning tool for each query, rather than running all in parallel blindly.
        try:
            from vulcan.reasoning import create_unified_reasoner, UNIFIED_AVAILABLE
            
            if UNIFIED_AVAILABLE:
                # Create UnifiedReasoner with tool selector config
                unified_config = {
                    "tool_selector_config": {
                        "available_tools": ["symbolic", "probabilistic", "causal", "analogical", "multimodal"],
                        "default_strategy": "adaptive_mix",
                        "min_confidence": 0.5,
                        "time_budget_ms": 10000,
                        "energy_budget_mj": 500,
                        "exploration_strategy": "EPSILON_GREEDY",
                        "cache_ttl": 300,
                    },
                    "enable_learning": True,
                    "enable_safety": True,
                }
                
                # Pass existing reasoners to UnifiedReasoner
                components["unified_reasoner"] = create_unified_reasoner(
                    config=unified_config,
                    enable_learning=True,
                    enable_safety=True,
                )
                
                if components["unified_reasoner"]:
                    logger.info("UnifiedReasoner with ToolSelector loaded successfully")
                else:
                    logger.warning("UnifiedReasoner creation returned None")
            else:
                logger.warning("UnifiedReasoner not available - reasoning will use direct calls")
                components["unified_reasoner"] = None
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load UnifiedReasoner: {e}")
            components["unified_reasoner"] = None

    def _import_components(self) -> Dict[str, Any]:
        """
        Import all required components with error handling

        FIXED: Import reasoning/learning from package level with graceful fallbacks
        FIXED: Pass None to safety validator to avoid AgentConfig.to_dict() error
        FIXED: Prioritize causal WorldModel with meta-reasoning + self-improvement
        FIXED: Import MultimodalProcessor from vulcan.processing
        FIXED: Import planning from vulcan.planning

        Returns:
            Dictionary of imported components
        """
        components = {}

        # Add env (config) to components dict
        components["env"] = self.config

        # Processing - FIXED: Import from vulcan.processing
        try:
            from vulcan.processing import MultimodalProcessor

            components["multimodal"] = MultimodalProcessor()
            logger.info("MultimodalProcessor loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import MultimodalProcessor: {e}")
            components["multimodal"] = None
        except Exception as e:
            logger.error(f"Failed to initialize MultimodalProcessor: {e}")
            components["multimodal"] = None

        # Reasoning - Refactored to use lazy, component-wise imports
        self._load_reasoners(components)

        # PREFER: Causal WorldModel (meta-reasoning + self-improvement)
        # Guard with try/except and fall back to learning.UnifiedWorldModel if needed
        # FIX: Mock Object Poisoning - Use None instead of MagicMock to prevent
        # downstream systems from receiving truthy mock objects that produce gibberish
        # output (e.g., "<MagicMock id='...'>") when serialized.
        components["world_model"] = None
        CausalWorldModel = None  # Define first
        try:
            from vulcan.world_model import WorldModel as CausalWorldModel
        except ImportError:
            # FIX: Set to None instead of MagicMock to prevent mock object poisoning
            CausalWorldModel = None
            logger.warning(
                "WorldModel not available - skipping initialization (no mock fallback)"
            )

        try:
            # FIX: Check both that CausalWorldModel is not None AND that it's callable
            # This ensures the import succeeded with a valid class, not just any truthy value
            if CausalWorldModel is not None and callable(CausalWorldModel):
                # Build configuration for CausalWorldModel from AgentConfig
                enable_si = bool(getattr(self.config, "enable_self_improvement", False))
                cfg_file = getattr(
                    self.config,
                    "intrinsic_drives_config_file",
                    "configs/intrinsic_drives.json",
                )
                state_file = getattr(
                    self.config, "intrinsic_drives_state_file", "data/agent_state.json"
                )
                
                # Log warning if self-improvement is disabled
                if not enable_si:
                    logger.warning(
                        "⚠ Self-improvement DISABLED - set enable_self_improvement=True in config "
                        "or VULCAN_ENABLE_SELF_IMPROVEMENT=true in environment to enable CSIU governance"
                    )

                wm_config = {
                    "enable_meta_reasoning": True,
                    "meta_reasoning_config": cfg_file,
                    "enable_self_improvement": enable_si,
                    "self_improvement_config": cfg_file,
                    "self_improvement_state": state_file,
                }

                # Note Issues #1-4: Use singleton WorldModel to prevent per-query reinitialization
                try:
                    from vulcan.reasoning.singletons import get_world_model
                    components["world_model"] = get_world_model(config=wm_config)
                    if components["world_model"]:
                        logger.info(
                            "WorldModel (causal) initialized via singleton with meta-reasoning=%s, self-improvement=%s",
                            True,
                            enable_si,
                        )
                    else:
                        # Fallback to direct instantiation if singleton fails
                        components["world_model"] = CausalWorldModel(config=wm_config)
                        logger.info(
                            "WorldModel (causal) initialized directly (singleton fallback) with meta-reasoning=%s, self-improvement=%s",
                            True,
                            enable_si,
                        )
                except ImportError:
                    components["world_model"] = CausalWorldModel(config=wm_config)
                    logger.info(
                        "WorldModel (causal) initialized directly with meta-reasoning=%s, self-improvement=%s",
                        True,
                        enable_si,
                    )
            else:
                # CausalWorldModel import failed, components["world_model"] stays None
                logger.warning(
                    "CausalWorldModel not available, skipping initialization."
                )
                components["world_model"] = None

        except Exception as e:
            logger.warning(
                f"Causal WorldModel unavailable or failed to initialize: {e}"
            )
            components["world_model"] = None

        # Learning - graceful fallbacks
        try:
            import vulcan.learning as learning_module

            components["continual"] = None
            components["meta_cognitive"] = None
            components["compositional"] = None

            # ONLY set a learning.UnifiedWorldModel if causal WM isn't available
            if components["world_model"] is None and hasattr(
                learning_module, "UnifiedWorldModel"
            ):
                try:
                    components["world_model"] = learning_module.UnifiedWorldModel(
                        components.get("multimodal")
                    )
                    logger.info(
                        "UnifiedWorldModel (learning) loaded (no meta-reasoning/self-improvement)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize UnifiedWorldModel: {e}")

            if hasattr(learning_module, "ContinualLearner"):
                try:
                    components["continual"] = learning_module.ContinualLearner()
                    logger.info("ContinualLearner loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize ContinualLearner: {e}")

            if hasattr(learning_module, "MetaCognitiveMonitor"):
                try:
                    components["meta_cognitive"] = (
                        learning_module.MetaCognitiveMonitor()
                    )
                    logger.info("MetaCognitiveMonitor loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize MetaCognitiveMonitor: {e}")

            if hasattr(learning_module, "CompositionalUnderstanding"):
                try:
                    components["compositional"] = (
                        learning_module.CompositionalUnderstanding()
                    )
                    logger.info("CompositionalUnderstanding loaded")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize CompositionalUnderstanding: {e}"
                    )

            # Log summary
            available_learners = sum(
                1
                for v in [
                    components["continual"],
                    components["meta_cognitive"],
                    components["compositional"],
                    components[
                        "world_model"
                    ],  # Count any world_model (CausalWorldModel or UnifiedWorldModel)
                ]
                if v is not None
            )
            logger.info(f"Learning components: {available_learners}/4 available")

        except ImportError as e:
            logger.error(f"Failed to import learning module: {e}")
            logger.info(
                "Learning components will be unavailable - system will operate in degraded mode"
            )
            components["continual"] = None
            components["meta_cognitive"] = None
            components["compositional"] = None
            # keep components['world_model'] as is (causal or None)
        except Exception as e:
            logger.error(f"Failed to initialize learning components: {e}")
            components["continual"] = None
            components["meta_cognitive"] = None
            components["compositional"] = None
            # keep components['world_model'] as is (causal or None)

        # Planning - FIXED: Import from vulcan.planning
        try:
            from vulcan.planning import (
                DistributedCoordinator,
                HierarchicalGoalSystem,
                ResourceAwareCompute,
            )

            components["goal_system"] = HierarchicalGoalSystem()
            components["resource_compute"] = ResourceAwareCompute()

            if getattr(self.config, "enable_distributed", False):
                components["distributed"] = DistributedCoordinator()
                logger.info("DistributedCoordinator enabled")
            else:
                components["distributed"] = None

            logger.info("Planning components loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import planning components: {e}")
            components["goal_system"] = None
            components["resource_compute"] = None
            components["distributed"] = None
        except Exception as e:
            logger.error(f"Failed to initialize planning components: {e}")
            components["goal_system"] = None
            components["resource_compute"] = None
            components["distributed"] = None

        # Safety - FIXED: Pass None instead of self.config to avoid AgentConfig.to_dict() error
        try:
            from vulcan.safety.safety_types import (
                ExplainabilityNode,
                GovernanceOrchestrator,
                NSOAligner,
                get_nso_aligner,
            )
            from vulcan.safety.safety_validator import EnhancedSafetyValidator

            # Pass None to use default SafetyConfig - avoids AgentConfig.to_dict() error
            # EnhancedSafetyValidator will create its own SafetyConfig with defaults
            components["safety_validator"] = EnhancedSafetyValidator(config=None)
            components["governance"] = GovernanceOrchestrator()
            # Note: Use singleton pattern to prevent model reloading on every request
            components["nso_aligner"] = get_nso_aligner() if get_nso_aligner is not None else NSOAligner()
            components["explainer"] = ExplainabilityNode()

            logger.info("Safety components loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import safety components: {e}")
            logger.warning(
                "Safety components will be unavailable - system will operate WITHOUT safety validation"
            )
            components["safety_validator"] = None
            components["governance"] = None
            components["nso_aligner"] = None
            components["explainer"] = None
        except Exception as e:
            logger.error(f"Failed to initialize safety components: {e}")
            logger.warning(
                "Safety components will be unavailable - system will operate WITHOUT safety validation"
            )
            components["safety_validator"] = None
            components["governance"] = None
            components["nso_aligner"] = None
            components["explainer"] = None

        # Memory - Import from parent directory (vulcan/) with proper config handling
        try:
            from ..memory import EpisodicMemory, MemoryIndex, MemoryPersistence

            # Create memory config with safe attribute access
            try:
                from ..memory.base import MemoryConfig

                # Extract memory config values with safe defaults
                memory_config = MemoryConfig(
                    max_working_memory=getattr(self.config, "max_working_memory", 20),
                    max_short_term=getattr(self.config, "short_term_capacity", 1000),
                    max_long_term=getattr(self.config, "long_term_capacity", 100000),
                    consolidation_interval=getattr(
                        self.config, "consolidation_interval", 1000
                    ),
                )
            except ImportError:
                # Fallback: create minimal config object
                class MinimalMemoryConfig:
                    def __init__(self):
                        self.max_working_memory = getattr(
                            self.config, "max_working_memory", 20
                        )
                        self.max_short_term = getattr(
                            self.config, "short_term_capacity", 1000
                        )
                        self.max_long_term = getattr(
                            self.config, "long_term_capacity", 100000
                        )
                        self.consolidation_interval = getattr(
                            self.config, "consolidation_interval", 1000
                        )

                memory_config = MinimalMemoryConfig()

            # Initialize memory components with proper config
            components["ltm"] = MemoryIndex()
            components["am"] = EpisodicMemory(memory_config)
            components["compressed"] = MemoryPersistence()

            logger.info("Memory components loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import memory components: {e}")
            components["ltm"] = None
            components["am"] = None
            components["compressed"] = None
        except Exception as e:
            logger.error(f"Failed to initialize memory components: {e}")
            components["ltm"] = None
            components["am"] = None
            components["compressed"] = None

        # --- ADDED MISSING IMPORTS ---

        # Curiosity Engine (Experiment Generator)
        try:
            from vulcan.curiosity_engine.experiment_generator import ExperimentGenerator

            components["experiment_generator"] = ExperimentGenerator()
            logger.info("ExperimentGenerator loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import ExperimentGenerator: {e}")
            components["experiment_generator"] = None
        except Exception as e:
            logger.error(f"Failed to initialize ExperimentGenerator: {e}")
            components["experiment_generator"] = None

        # Problem Decomposer (Problem Executor)
        try:
            from vulcan.problem_decomposer.problem_executor import ProblemExecutor

            components["problem_executor"] = ProblemExecutor()
            logger.info("ProblemExecutor loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import ProblemExecutor: {e}")
            components["problem_executor"] = None
        except Exception as e:
            logger.error(f"Failed to initialize ProblemExecutor: {e}")
            components["problem_executor"] = None

        # Semantic Bridge (Core) - Initialize before components that need it
        try:
            # Note Issue #48: Use singleton SemanticBridge to prevent per-query reinitialization.
            try:
                from vulcan.reasoning.singletons import get_semantic_bridge
                components["semantic_bridge"] = get_semantic_bridge()
                if components["semantic_bridge"] is not None:
                    logger.info("SemanticBridge obtained from singleton")
                else:
                    # Fallback to direct instantiation
                    from vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
                    components["semantic_bridge"] = SemanticBridge(
                        world_model=components.get("world_model"),
                        vulcan_memory=components.get("am"),
                        safety_config=None,
                    )
                    logger.info("SemanticBridge initialized directly (singleton unavailable)")
            except ImportError:
                from vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
                # Pass world model, memory, and safety config
                components["semantic_bridge"] = SemanticBridge(
                    world_model=components.get("world_model"),
                    vulcan_memory=components.get("am"),  # EpisodicMemory
                    safety_config=None,  # Will use singleton safety validator
                )
                logger.info("SemanticBridge initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import SemanticBridge: {e}")
            components["semantic_bridge"] = None
        except Exception as e:
            logger.error(f"Failed to initialize SemanticBridge: {e}")
            components["semantic_bridge"] = None

        # Knowledge Crystallizer (Core)
        try:
            from vulcan.knowledge_crystallizer.knowledge_crystallizer_core import (
                KnowledgeCrystallizer,
            )

            # Pass memory and semantic bridge
            components["knowledge_crystallizer"] = KnowledgeCrystallizer(
                vulcan_memory=components.get("am"),  # EpisodicMemory
                semantic_bridge=components.get("semantic_bridge"),  # Now available
            )
            logger.info("KnowledgeCrystallizer initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import KnowledgeCrystallizer: {e}")
            components["knowledge_crystallizer"] = None
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeCrystallizer: {e}")
            components["knowledge_crystallizer"] = None

        # Curiosity Engine (Core)
        try:
            from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine

            # Pass knowledge crystallizer, decomposer, and world model if available
            components["curiosity_engine"] = CuriosityEngine(
                knowledge=components.get("knowledge_crystallizer"),
                decomposer=None,  # Will be set after ProblemDecomposer is created
                world_model=components.get("world_model"),
            )
            logger.info("CuriosityEngine initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import CuriosityEngine: {e}")
            components["curiosity_engine"] = None
        except Exception as e:
            logger.error(f"Failed to initialize CuriosityEngine: {e}")
            components["curiosity_engine"] = None

        # Problem Decomposer (Core) - Use bootstrap for proper initialization
        try:
            from vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer

            # Use factory function to create fully initialized decomposer with:
            # - All strategies registered
            # - Fallback chain populated
            # - Base principles initialized
            components["problem_decomposer"] = create_decomposer(
                semantic_bridge=components.get("semantic_bridge"),  # Now available
                vulcan_memory=components.get("am"),  # EpisodicMemory
                validator=None,
                storage_path=None,  # Use default in-memory for now
                config=None,  # No special config needed
                safety_validator=components.get(
                    "safety_validator"
                ),  # Pass as separate parameter
            )
            logger.info("ProblemDecomposer initialized successfully with bootstrap")

            # Now update CuriosityEngine with the decomposer
            if components.get("curiosity_engine"):
                components["curiosity_engine"].decomposer = components[
                    "problem_decomposer"
                ]
                logger.info("CuriosityEngine linked to ProblemDecomposer")

        except ImportError as e:
            logger.error(f"Failed to import ProblemDecomposer bootstrap: {e}")
            components["problem_decomposer"] = None
        except Exception as e:
            logger.error(f"Failed to initialize ProblemDecomposer: {e}")
            components["problem_decomposer"] = None

        # --- END ADDED IMPORTS ---

        # Log component summary
        total_components = (
            27  # Total expected components (was 26, added SemanticBridge)
        )
        available_components = sum(1 for v in components.values() if v is not None)
        logger.info(
            f"Component loading complete: {available_components}/{total_components} components available"
        )

        return components

    def _create_dependencies(
        self, components: Dict[str, Any]
    ) -> EnhancedCollectiveDeps:
        """
        Create dependencies container from components

        Args:
            components: Dictionary of imported components

        Returns:
            EnhancedCollectiveDeps instance
        """

        # Unpack the world_model to get its sub-components if it exists
        world_model = components.get("world_model")
        if world_model and not isinstance(world_model, MagicMock):
            # Get motivational_introspection which contains most meta-reasoning components as properties
            motivational_intro = getattr(
                world_model, "motivational_introspection", None
            )
            self_improvement = getattr(world_model, "self_improvement_drive", None)

            meta_components = {
                # Top-level world_model attributes
                "self_improvement_drive": self_improvement,
                "motivational_introspection": motivational_intro,
                "validation_tracker": getattr(world_model, "validation_tracker", None),
                "transparency_interface": getattr(
                    world_model, "transparency_interface", None
                ),
                "value_evolution_tracker": getattr(
                    world_model, "value_evolution_tracker", None
                ),
            }

            # Extract components from motivational_introspection properties
            if motivational_intro and not isinstance(motivational_intro, MagicMock):
                meta_components.update(
                    {
                        "objective_hierarchy": getattr(
                            motivational_intro, "objective_hierarchy", None
                        ),
                        "objective_negotiator": getattr(
                            motivational_intro, "objective_negotiator", None
                        ),
                        "goal_conflict_detector": getattr(
                            motivational_intro, "conflict_detector", None
                        ),
                        "counterfactual_objectives": getattr(
                            motivational_intro, "counterfactual_reasoner", None
                        ),
                    }
                )
            else:
                # If motivational_introspection not available, set these to None
                meta_components.update(
                    {
                        "objective_hierarchy": None,
                        "objective_negotiator": None,
                        "goal_conflict_detector": None,
                        "counterfactual_objectives": None,
                    }
                )

            # Components that may not be implemented yet (setting to None to avoid missing keys)
            meta_components.update(
                {
                    "preference_learner": None,
                    "ethical_boundary_monitor": None,
                    "curiosity_reward_shaper": None,
                    "internal_critic": None,
                    # Note: Accessing private attribute _auto_apply_policy because SelfImprovementDrive
                    # doesn't expose a public property for this. Consider adding public accessor if needed.
                    "auto_apply_policy": (
                        getattr(self_improvement, "_auto_apply_policy", None)
                        if self_improvement
                        else None
                    ),
                }
            )
        else:
            meta_components = {}

        deps = EnhancedCollectiveDeps(
            env=components.get(
                "env"
            ),  # Pass 'env' if you load it in _import_components
            metrics=self.metrics_collector,
            safety_validator=components.get("safety_validator"),
            governance=components.get("governance"),
            nso_aligner=components.get("nso_aligner"),
            explainer=components.get("explainer"),
            ltm=components.get("ltm"),
            am=components.get("am"),
            compressed_memory=components.get("compressed"),
            multimodal=components.get("multimodal"),
            probabilistic=components.get("probabilistic"),
            symbolic=components.get("symbolic"),
            causal=components.get("causal"),
            abstract=components.get("abstract"),
            cross_modal=components.get("cross_modal"),
            continual=components.get("continual"),
            compositional=components.get("compositional"),
            meta_cognitive=components.get("meta_cognitive"),
            world_model=world_model,
            # --- ADDED MISSING COMPONENTS ---
            experiment_generator=components.get("experiment_generator"),
            problem_executor=components.get("problem_executor"),
            goal_system=components.get("goal_system"),
            resource_compute=components.get("resource_compute"),
            distributed=components.get("distributed"),
            # --- ADDED META_REASONING COMPONENTS ---
            **meta_components,
        )

        return deps

    def _create_system_state(self):
        """
        Create system state object

        Returns:
            SystemState instance
        """
        try:
            # Import from parent directory (vulcan/)
            from ..vulcan_types import SystemState

            # FIXED: Handle SafetyPolicies object correctly
            safety_policies = getattr(self.config, "safety_policies", None)
            if safety_policies and hasattr(safety_policies, "names_to_versions"):
                policies = safety_policies.names_to_versions
            elif (
                isinstance(safety_policies, dict)
                and "names_to_versions" in safety_policies
            ):
                policies = safety_policies["names_to_versions"]
            else:
                policies = {}

            system_state = SystemState(
                CID=f"vulcan_agi_{int(time.time())}", policies=policies
            )

            return system_state

        except ImportError:
            logger.error("Failed to import SystemState, creating minimal state")

            # Create minimal state object
            class MinimalSystemState:
                def __init__(self):
                    self.CID = f"vulcan_agi_{int(time.time())}"
                    self.step = 0

                    # FIXED: Handle SafetyPolicies object correctly
                    safety_policies = getattr(self.config, "safety_policies", None)
                    if safety_policies and hasattr(
                        safety_policies, "names_to_versions"
                    ):
                        self.policies = safety_policies.names_to_versions
                    elif (
                        isinstance(safety_policies, dict)
                        and "names_to_versions" in safety_policies
                    ):
                        self.policies = safety_policies["names_to_versions"]
                    else:
                        self.policies = {}

                    class Health:
                        energy_budget_left_nJ = 1e9
                        memory_usage_mb = 0
                        latency_ms = 0
                        error_rate = 0.0

                    class SelfAwareness:
                        learning_efficiency = 1.0
                        uncertainty = 0.5
                        identity_drift = 0.0

                    self.health = Health()
                    self.SA = SelfAwareness()
                    self.active_modalities = set()
                    self.uncertainty_estimates = {}
                    self.provenance_chain = []
                    self.last_obs = None
                    self.last_reward = None

            return MinimalSystemState()

    def _create_orchestrator(
        self, orchestrator_type: str, system_state: Any, deps: EnhancedCollectiveDeps
    ) -> VULCANAGICollective:
        """
        Create orchestrator of specified type with validation

        Args:
            orchestrator_type: Type of orchestrator
            system_state: System state
            deps: Dependencies

        Returns:
            Orchestrator instance
        """
        # Validate orchestrator type
        valid_types = ["parallel", "adaptive", "fault_tolerant", "basic"]

        if orchestrator_type not in valid_types:
            logger.warning(
                f"Unknown orchestrator type '{orchestrator_type}', "
                f"valid types are: {', '.join(valid_types)}. Using 'basic'."
            )
            orchestrator_type = "basic"

        # Create appropriate orchestrator
        # Note: Pass redis_client to all orchestrators for state persistence
        if orchestrator_type == "parallel":
            logger.info("Creating ParallelOrchestrator")
            return ParallelOrchestrator(self.config, system_state, deps, redis_client=self.redis_client)
        elif orchestrator_type == "adaptive":
            logger.info("Creating AdaptiveOrchestrator")
            return AdaptiveOrchestrator(self.config, system_state, deps, redis_client=self.redis_client)
        elif orchestrator_type == "fault_tolerant":
            logger.info("Creating FaultTolerantOrchestrator")
            return FaultTolerantOrchestrator(self.config, system_state, deps, redis_client=self.redis_client)
        else:
            logger.info("Creating basic VULCANAGICollective")
            return VULCANAGICollective(self.config, system_state, deps, redis_client=self.redis_client)

    def step_with_monitoring(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute step with comprehensive monitoring

        FIXED: Check for goal_system and governance methods before calling
        FIXED: Pass governed_actions to orchestrator instead of context

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result dictionary
        """
        if self._shutdown_requested:
            return {
                "action": {"type": "SYSTEM_SHUTDOWN"},
                "error": "System shutdown requested",
            }

        try:
            # Health check
            if not self._health_check():
                return {
                    "action": {"type": "SYSTEM_UNHEALTHY"},
                    "error": "Health check failed",
                }

            # Generate and enforce policies - FIXED: Check for methods before calling
            governed_actions = context
            try:
                # Try to generate plan
                if self.collective.deps.goal_system and hasattr(
                    self.collective.deps.goal_system, "generate_plan"
                ):
                    planned_actions = self.collective.deps.goal_system.generate_plan(
                        context
                    )
                else:
                    planned_actions = context

                # Try to enforce policies
                if self.collective.deps.governance and hasattr(
                    self.collective.deps.governance, "enforce_policies"
                ):
                    governed_actions = self.collective.deps.governance.enforce_policies(
                        planned_actions
                    )
                else:
                    governed_actions = planned_actions

            except Exception as e:
                logger.debug(f"Policy enforcement not available: {e}")
                governed_actions = context

            # Execute with unified runtime or orchestrator - FIXED: Use governed_actions
            if self.unified_runtime is not None:
                try:
                    result = asyncio.run(
                        self.unified_runtime.execute_graph(governed_actions)
                    )
                except Exception as e:
                    logger.warning(
                        f"UnifiedRuntime execution failed: {e}, falling back to orchestrator"
                    )
                    result = self._execute_with_orchestrator(history, governed_actions)
            else:
                result = self._execute_with_orchestrator(history, governed_actions)

            # Update monitoring
            self._update_monitoring(result)

            # Auto-checkpoint periodically (thread-safe with race condition prevention)
            checkpoint_interval = getattr(self.config, "checkpoint_interval", 100)
            current_step = self.collective.sys.step

            # FIXED: Guard against step 0 and duplicate checkpoints
            if (
                current_step > 0
                and current_step % checkpoint_interval == 0
                and current_step != self._last_checkpointed_step
            ):
                self._auto_checkpoint()

            return result

        except Exception as e:
            logger.error(f"Error in step_with_monitoring: {e}", exc_info=True)
            self.metrics_collector.increment_counter("errors_total")
            return {
                "action": {"type": "ERROR_FALLBACK"},
                "error": str(e),
                "success": False,
            }

    def _execute_with_orchestrator(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute with appropriate orchestrator type

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Execution result
        """
        if isinstance(self.collective, ParallelOrchestrator):
            return asyncio.run(self.collective.step_parallel(history, context))
        elif isinstance(self.collective, AdaptiveOrchestrator):
            return self.collective.adaptive_step(history, context)
        elif isinstance(self.collective, FaultTolerantOrchestrator):
            return self.collective.step_with_recovery(history, context)
        else:
            return self.collective.step(history, context)

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including agent pool

        Returns:
            Status dictionary
        """
        uptime = time.time() - self.startup_time

        try:
            # Get modalities as strings for JSON serialization
            active_modalities = [
                m.value if hasattr(m, "value") else str(m)
                for m in self.collective.sys.active_modalities
            ]
        except Exception as e:
            logger.debug(f"Failed to get active modalities: {e}")
            active_modalities = []

        # Get goal status
        try:
            goal_status = (
                self.collective.deps.goal_system.get_goal_status()
                if self.collective.deps.goal_system
                else {}
            )
        except Exception as e:
            logger.debug(f"Failed to get goal status: {e}")
            goal_status = {}

        # Get safety report
        try:
            safety_report = (
                self.collective.deps.safety_validator.get_safety_report()
                if self.collective.deps.safety_validator
                else {}
            )
        except Exception as e:
            logger.debug(f"Failed to get safety report: {e}")
            safety_report = {}

        return {
            "cid": self.collective.sys.CID,
            "step": self.collective.sys.step,
            "uptime_seconds": uptime,
            "orchestrator_type": self.orchestrator_type,
            "health": {
                "energy_budget_left_nJ": self.collective.sys.health.energy_budget_left_nJ,
                "memory_usage_mb": self.collective.sys.health.memory_usage_mb,
                "latency_ms": self.collective.sys.health.latency_ms,
                "error_rate": self.collective.sys.health.error_rate,
            },
            "self_awareness": {
                "learning_efficiency": self.collective.sys.SA.learning_efficiency,
                "uncertainty": self.collective.sys.SA.uncertainty,
                "identity_drift": self.collective.sys.SA.identity_drift,
            },
            "active_modalities": active_modalities,
            "metrics": self.metrics_collector.get_summary(),
            "goal_status": goal_status,
            "safety_report": safety_report,
            "agent_pool": self.collective.agent_pool.get_pool_status(),
            "config": {
                "multimodal": getattr(self.config, "enable_multimodal", False),
                "symbolic": getattr(self.config, "enable_symbolic", False),
                "distributed": getattr(self.config, "enable_distributed", False),
            },
            "shutdown_requested": self._shutdown_requested,
        }

    def _save_checkpoint_internal(self, checkpoint_path: str) -> bool:
        """
        Internal, non-locking method to perform the checkpoint save.

        Assumes the caller holds _checkpoint_lock.

        Args:
            checkpoint_path: The exact path to save the checkpoint to.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare checkpoint data
            checkpoint = {
                "system_state": self.collective.sys,
                "metrics": self.metrics_collector.export_metrics(),
                "agent_pool_status": self.collective.agent_pool.get_pool_status(),
                "timestamp": time.time(),
                "step": self.collective.sys.step,
                "cid": self.collective.sys.CID,
                "orchestrator_type": self.orchestrator_type,
            }

            # Serialize checkpoint to bytes
            checkpoint_data = pickle.dumps(checkpoint)

            # Write using atomic operation with retry logic
            success = atomic_write_with_retry(
                data=checkpoint_data,
                target_path=checkpoint_path,
                max_retries=getattr(self.config, "checkpoint_retry_attempts", 5),
                retry_delay=getattr(self.config, "checkpoint_retry_delay", 0.1),
            )

            if not success:
                logger.error(f"Failed to save checkpoint to {checkpoint_path}")
                return False

            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Also save metadata as JSON for easier inspection
            metadata_path = checkpoint_path.replace(".pkl", "_metadata.json")
            try:
                metadata = {
                    "timestamp": checkpoint["timestamp"],
                    "step": checkpoint["step"],
                    "cid": checkpoint["cid"],
                    "orchestrator_type": checkpoint["orchestrator_type"],
                    "agent_pool_status": checkpoint["agent_pool_status"],
                }

                # Write metadata using atomic operation
                metadata_data = json.dumps(metadata, indent=2).encode("utf-8")
                atomic_write_with_retry(
                    data=metadata_data, target_path=metadata_path, max_retries=3
                )

            except Exception as e:
                logger.warning(f"Failed to save checkpoint metadata: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            return False

    def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Save system checkpoint with Windows-compatible atomic file operations

        FIXED: Uses atomic_write_with_retry with os.replace for Windows safety
        FIXED: Implements proper file handle management and retry logic
        FIXED: Handles PermissionError [WinError 32] gracefully
        FIXED: Thread-safe with locking to prevent race conditions
        FIXED: Refactored to use non-locking _save_checkpoint_internal to avoid nested locks.

        Args:
            checkpoint_path: Path to save checkpoint (auto-generated if None)

        Returns:
            True if successful, False otherwise
        """
        # FIXED: Use lock to serialize checkpoint operations
        with self._checkpoint_lock:
            if checkpoint_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = str(
                    self.checkpoint_dir
                    / f"checkpoint_{timestamp}_step{self.collective.sys.step}.pkl"
                )

            # Call the internal, non-locking method
            return self._save_checkpoint_internal(checkpoint_path)

    def _health_check(self) -> bool:
        """
        Check system health

        Returns:
            True if healthy, False otherwise
        """
        try:
            health = self.collective.sys.health

            # Energy budget check
            energy_threshold = getattr(self.config, "min_energy_budget_nJ", 1000)
            if health.energy_budget_left_nJ < energy_threshold:
                logger.warning(f"Low energy budget: {health.energy_budget_left_nJ} nJ")
                return False

            # Memory usage check
            memory_threshold = getattr(self.config, "max_memory_usage_mb", 7000)
            if health.memory_usage_mb > memory_threshold:
                logger.warning(f"High memory usage: {health.memory_usage_mb} MB")
                return False

            # Error rate check
            error_rate_threshold = getattr(self.config, "slo_max_error_rate", 0.1)
            if health.error_rate > error_rate_threshold:
                logger.warning(
                    f"Error rate {health.error_rate:.3f} exceeds SLO {error_rate_threshold}"
                )
                return False

            # Agent pool capacity check
            pool_status = self.collective.agent_pool.get_pool_status()
            if pool_status["total_agents"] < self.collective.agent_pool.min_agents:
                logger.warning(
                    f"Agent pool below minimum capacity: {pool_status['total_agents']}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Health check failed with error: {e}")
            return False

    def _update_monitoring(self, result: Dict[str, Any]):
        """
        Update monitoring metrics

        Args:
            result: Execution result
        """
        try:
            # Update system metrics
            self.metrics_collector.update_gauge(
                "energy_remaining_nJ", self.collective.sys.health.energy_budget_left_nJ
            )
            self.metrics_collector.update_gauge(
                "identity_drift", self.collective.sys.SA.identity_drift
            )
            self.metrics_collector.update_gauge(
                "uncertainty", self.collective.sys.SA.uncertainty
            )
            self.metrics_collector.update_gauge(
                "learning_efficiency", self.collective.sys.SA.learning_efficiency
            )

            # Update agent pool metrics
            pool_status = self.collective.agent_pool.get_pool_status()
            self.metrics_collector.update_gauge(
                "agent_pool_size", pool_status["total_agents"]
            )
            self.metrics_collector.update_gauge(
                "agent_pool_idle",
                pool_status["state_distribution"].get(AgentState.IDLE.value, 0),
            )
            self.metrics_collector.update_gauge(
                "agent_pool_working",
                pool_status["state_distribution"].get(AgentState.WORKING.value, 0),
            )

            # Update result-based metrics
            if result.get("success"):
                self.metrics_collector.increment_counter("successful_steps")
            else:
                self.metrics_collector.increment_counter("failed_steps")

        except Exception as e:
            logger.debug(f"Failed to update monitoring: {e}")

    # *** MODIFIED METHOD START ***
    def _auto_checkpoint(self):
        """
        Automatic checkpointing with thread-safety and duplicate prevention

        FIXED: Uses lock with timeout to prevent concurrent checkpoint operations
        FIXED: Tracks last checkpointed step to prevent duplicates
        FIXED: Skips step 0 to avoid race conditions during initialization
        FIXED: Calls _save_checkpoint_internal directly to avoid nested RLock acquisition.
        """
        current_step = self.collective.sys.step

        # FIXED: Guard against step 0 and duplicate checkpoints
        if current_step <= 0:
            logger.debug(
                f"Skipping auto-checkpoint at step {current_step} (step must be > 0)"
            )
            return

        # FIXED: Check if we already checkpointed this step
        # Acquire lock with timeout
        if not self._checkpoint_lock.acquire(
            timeout=5.0
        ):  # Add timeout to prevent test hang
            logger.warning(
                "Timeout acquiring checkpoint lock, skipping auto-checkpoint"
            )
            return

        try:
            if current_step == self._last_checkpointed_step:
                logger.debug(
                    f"Skipping duplicate auto-checkpoint at step {current_step}"
                )
                return

            checkpoint_path = str(
                self.checkpoint_dir / f"checkpoint_auto_{current_step}.pkl"
            )

            # Check if this checkpoint already exists (another thread may have created it)
            if Path(checkpoint_path).exists():
                logger.debug(f"Auto-checkpoint at step {current_step} already exists")
                self._last_checkpointed_step = current_step
                return

            # Perform the checkpoint
            # FIXED: Call the internal method directly since we already hold the lock
            success = self._save_checkpoint_internal(checkpoint_path)

            if success:
                # Update last checkpointed step to prevent duplicates
                self._last_checkpointed_step = current_step
                logger.info(f"Auto-checkpoint completed at step {current_step}")

                # Cleanup old auto checkpoints
                self._cleanup_old_checkpoints()
            else:
                logger.warning(f"Auto-checkpoint failed at step {current_step}")

        except Exception as e:
            logger.error(f"Auto-checkpoint failed: {e}", exc_info=True)
        finally:
            self._checkpoint_lock.release()  # Release the lock

    # *** MODIFIED METHOD END ***

    # *** MODIFIED METHOD START ***
    def _cleanup_old_checkpoints(self):
        """
        Cleanup old auto-generated checkpoints

        Thread-safe cleanup of old checkpoints to prevent disk space issues
        FIXED: Use missing_ok=True for unlink to handle potential locking/missing files gracefully.
        NOTE: This method assumes the caller (_auto_checkpoint) already holds the _checkpoint_lock.
        """
        try:
            max_checkpoints = getattr(self.config, "max_auto_checkpoints", 10)

            # Find all auto checkpoints
            auto_checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_auto_*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )

            # Remove oldest checkpoints if over limit
            if len(auto_checkpoints) > max_checkpoints:
                checkpoints_to_remove = auto_checkpoints[:-max_checkpoints]
                logger.debug(
                    f"Cleaning up {len(checkpoints_to_remove)} old checkpoints."
                )
                for checkpoint in checkpoints_to_remove:
                    try:
                        checkpoint.unlink(
                            missing_ok=True
                        )  # Non-blocking, skip if not exists/locked
                        # Also remove metadata file
                        metadata_file = checkpoint.with_name(
                            checkpoint.stem + "_metadata.json"
                        )
                        metadata_file.unlink(missing_ok=True)  # Use missing_ok here too
                        logger.debug(f"Removed old checkpoint: {checkpoint}")
                    except (
                        OSError
                    ) as e:  # Catch OS errors like permission denied more specifically
                        logger.warning(
                            f"Failed to remove old checkpoint {checkpoint} due to OS error: {e}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old checkpoint {checkpoint}: {e}"
                        )
            else:
                logger.debug(
                    f"No old checkpoints to clean up ({len(auto_checkpoints)} <= {max_checkpoints})."
                )

        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")

    # *** MODIED METHOD END ***

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load system from checkpoint
        SECURITY: Use safe_pickle_load to prevent deserialization attacks

        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            # SECURITY FIX: Use safe_pickle_load instead of pickle.load
            checkpoint = safe_pickle_load(checkpoint_path)

            # Restore system state
            self.collective.sys = checkpoint["system_state"]

            # FIXED: Update last checkpointed step to prevent immediate re-checkpoint
            self._last_checkpointed_step = self.collective.sys.step

            # Restore metrics if available
            if "metrics" in checkpoint:
                try:
                    self.metrics_collector.import_metrics(checkpoint["metrics"])
                except Exception as e:
                    logger.warning(f"Failed to restore metrics: {e}")

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"Restored to step {self.collective.sys.step}")

        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []

        try:
            for checkpoint_file in sorted(
                self.checkpoint_dir.glob("checkpoint_*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                metadata_file = checkpoint_file.with_name(
                    checkpoint_file.stem + "_metadata.json"
                )

                info = {
                    "path": str(checkpoint_file),
                    "size_mb": checkpoint_file.stat().st_size / (1024 * 1024),
                    "created": checkpoint_file.stat().st_mtime,
                }

                # Load metadata if available
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            info.update(metadata)
                    except Exception as e:
                        logger.debug(
                            f"Failed to load metadata for {checkpoint_file}: {e}"
                        )

                checkpoints.append(info)

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")

        return checkpoints

    def request_shutdown(self):
        """Request graceful shutdown"""
        logger.info("Shutdown requested")
        self._shutdown_requested = True

    def shutdown(self):
        """Gracefully shutdown the deployment"""
        if self._shutdown_requested:
            logger.info("Shutdown already in progress")
            return

        logger.info("Shutting down Production Deployment")
        self._shutdown_requested = True

        # Save final checkpoint (thread-safe)
        try:
            final_checkpoint = str(
                self.checkpoint_dir / f"checkpoint_final_{self.collective.sys.step}.pkl"
            )
            self.save_checkpoint(final_checkpoint)
            logger.info("Final checkpoint saved")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")

        # Shutdown collective
        if self.collective:
            try:
                self.collective.shutdown()
                logger.info("Collective shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down collective: {e}")

        # --- ADD THIS BLOCK ---
        # Shutdown all other dependencies (safety, memory, processing, etc.)
        # This will signal all their background threads to stop.
        if self.collective and self.collective.deps:
            try:
                self.collective.deps.shutdown_all()
                logger.info("All dependencies shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down dependencies: {e}")
        # --- END ADD BLOCK ---

        # Shutdown unified runtime
        if self.unified_runtime:
            try:
                # Use async_cleanup if available, otherwise cleanup
                if hasattr(self.unified_runtime, "async_cleanup"):
                    asyncio.run(self.unified_runtime.async_cleanup())
                elif hasattr(self.unified_runtime, "cleanup"):
                    self.unified_runtime.cleanup()
                elif hasattr(self.unified_runtime, "shutdown"):  # Added fallback check
                    if asyncio.iscoroutinefunction(self.unified_runtime.shutdown):
                        asyncio.run(self.unified_runtime.shutdown())
                    else:
                        self.unified_runtime.shutdown()

                logger.info("UnifiedRuntime shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down UnifiedRuntime: {e}")

        # Shutdown metrics collector
        try:
            self.metrics_collector.shutdown()
            logger.info("Metrics collector shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down metrics: {e}")

        # Force garbage collection
        import gc

        gc.collect()

        logger.info("Shutdown complete")

    def _atexit_cleanup(self):
        """
        Cleanup method registered with atexit for deterministic shutdown.
        
        SAFETY FIX: Replaces __del__ to avoid NameError/ImportError when
        Python destroys global variables during interpreter shutdown.
        This ensures logging and other globals are still available.
        """
        try:
            if not self._shutdown_requested:
                logger.info("ProductionDeployment atexit cleanup triggered")
                self.shutdown()
        except Exception as e:
            # Use print as fallback since logging might be unavailable
            try:
                logger.error(f"Error during atexit cleanup: {e}", exc_info=True)
            except:
                print(f"ProductionDeployment: Error in atexit cleanup: {e}", file=sys.stderr)

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup on context exit."""
        self.shutdown()
        return False  # Don't suppress exceptions


# ============================================================
# AUTO-SCALER - Adaptive Resource Management
# ============================================================


class AutoScaler:
    """Automatically scale agent pool based on load with proper locking"""

    def __init__(self, pool_manager):
        """
        Initialize auto-scaler

        Args:
            pool_manager: Agent pool manager instance
        """
        import threading
        
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
        
        # Import SIMPLE_MODE constants if available
        try:
            from .agent_pool_types import SIMPLE_MODE, SIMPLE_MODE_CHECK_INTERVAL
        except ImportError:
            SIMPLE_MODE = False
            SIMPLE_MODE_CHECK_INTERVAL = 300
        
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
# RECOVERY MANAGER - Fault Tolerance
# ============================================================


class RecoveryManager:
    """Manages agent recovery and fault tolerance"""

    def __init__(self, pool_manager):
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

    def _recover_error_agent(self, agent_id: str, metadata) -> bool:
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

    def _recover_terminated_agent(self, agent_id: str, metadata) -> bool:
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

    def _recover_suspended_agent(self, agent_id: str, metadata) -> bool:
        """Recover suspended agent"""
        logger.info(f"Recovering suspended agent {agent_id}")
        return metadata.transition_state(AgentState.IDLE, "Recovered from suspension")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = ["ProductionDeployment", "atomic_write_with_retry", "AutoScaler", "RecoveryManager"]
