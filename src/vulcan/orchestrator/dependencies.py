# ============================================================
# VULCAN-AGI Orchestrator - Dependencies Module
# Enhanced dependencies container for all system components
# FULLY FIXED VERSION - Enhanced with validation, type hints, and factory functions
# Unicode-safe printing for cross-platform compatibility
# FIXED: Corrected Unicode checkmark symbols
# UPDATED: Added experiment_generator, problem_executor, and self_improvement_drive
# INTEGRATED: Self-improvement factory with proper initialization
# META-REASONING INTEGRATION: Added imports from meta_reasoning directory
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set

from .metrics import EnhancedMetricsCollector

logger = logging.getLogger(__name__)


# ============================================================
# DEPENDENCY TRACKING
# ============================================================

# Learning dependencies tracking dictionary (keep general learning deps here)
learning_deps = {}

# Distributed dependencies tracking dictionary
distributed_deps = {}

# Meta-Reasoning dependencies tracking dictionary
meta_reasoning_deps = {}

# Learning Component Imports
try:
    from vulcan.learning.continual_learning import ContinualLearner

    learning_deps["continual"] = True
except ImportError as e:
    logger.debug(f"Failed to import ContinualLearner: {e}")
    learning_deps["continual"] = False

# Distributed Component Imports
try:
    from vulcan.planning import DistributedCoordinator

    distributed_deps["distributed"] = True
except ImportError as e:
    logger.debug(f"Failed to import DistributedCoordinator: {e}")
    distributed_deps["distributed"] = False

# Meta-Reasoning Component Imports
try:
    from vulcan.world_model.meta_reasoning.self_improvement_drive import (
        SelfImprovementDrive,
    )

    meta_reasoning_deps["self_improvement_drive"] = True
except ImportError as e:
    logger.debug(
        f"Failed to import SelfImprovementDrive: {e}"
    )  # Debug level for optional
    meta_reasoning_deps["self_improvement_drive"] = False

try:
    from vulcan.world_model.meta_reasoning.motivational_introspection import (
        MotivationalIntrospection,
    )

    meta_reasoning_deps["motivational_introspection"] = True
except ImportError as e:
    logger.debug(f"Failed to import MotivationalIntrospection: {e}")
    meta_reasoning_deps["motivational_introspection"] = False

try:
    from vulcan.world_model.meta_reasoning.objective_hierarchy import ObjectiveHierarchy

    meta_reasoning_deps["objective_hierarchy"] = True
except ImportError as e:
    logger.debug(f"Failed to import ObjectiveHierarchy: {e}")
    meta_reasoning_deps["objective_hierarchy"] = False

try:
    from vulcan.world_model.meta_reasoning.objective_negotiator import (
        ObjectiveNegotiator,
    )

    meta_reasoning_deps["objective_negotiator"] = True
except ImportError as e:
    logger.debug(f"Failed to import ObjectiveNegotiator: {e}")
    meta_reasoning_deps["objective_negotiator"] = False

try:
    from vulcan.world_model.meta_reasoning.goal_conflict_detector import (
        GoalConflictDetector,
    )

    meta_reasoning_deps["goal_conflict_detector"] = True
except ImportError as e:
    logger.debug(f"Failed to import GoalConflictDetector: {e}")
    meta_reasoning_deps["goal_conflict_detector"] = False

try:
    from vulcan.world_model.meta_reasoning.preference_learner import PreferenceLearner

    meta_reasoning_deps["preference_learner"] = True
except ImportError as e:
    logger.debug(f"Failed to import PreferenceLearner: {e}")
    meta_reasoning_deps["preference_learner"] = False

try:
    from vulcan.world_model.meta_reasoning.value_evolution_tracker import (
        ValueEvolutionTracker,
    )

    meta_reasoning_deps["value_evolution_tracker"] = True
except ImportError as e:
    logger.debug(f"Failed to import ValueEvolutionTracker: {e}")
    meta_reasoning_deps["value_evolution_tracker"] = False

try:
    from vulcan.world_model.meta_reasoning.ethical_boundary_monitor import (
        EthicalBoundaryMonitor,
    )

    meta_reasoning_deps["ethical_boundary_monitor"] = True
except ImportError as e:
    logger.debug(f"Failed to import EthicalBoundaryMonitor: {e}")
    meta_reasoning_deps["ethical_boundary_monitor"] = False

try:
    from vulcan.world_model.meta_reasoning.curiosity_reward_shaper import (
        CuriosityRewardShaper,
    )

    meta_reasoning_deps["curiosity_reward_shaper"] = True
except ImportError as e:
    logger.debug(f"Failed to import CuriosityRewardShaper: {e}")
    meta_reasoning_deps["curiosity_reward_shaper"] = False

try:
    from vulcan.world_model.meta_reasoning.internal_critic import InternalCritic

    meta_reasoning_deps["internal_critic"] = True
except ImportError as e:
    logger.debug(f"Failed to import InternalCritic: {e}")
    meta_reasoning_deps["internal_critic"] = False

# Assuming these are classes based on user request - adjust if they are functions/modules
try:
    # Import the Policy class and load_policy function from auto_apply_policy module
    from vulcan.world_model.meta_reasoning.auto_apply_policy import Policy, load_policy

    meta_reasoning_deps["auto_apply_policy"] = True
except ImportError as e:
    logger.debug(f"Failed to import auto_apply_policy module: {e}")
    meta_reasoning_deps["auto_apply_policy"] = False

try:
    # Assuming 'validation_tracker' refers to a component
    # Adjust the import path and class name as needed
    from vulcan.world_model.meta_reasoning.validation_tracker import (
        ValidationTracker,
    )  # Placeholder

    meta_reasoning_deps["validation_tracker"] = True
except ImportError as e:
    logger.debug(f"Failed to import ValidationTracker: {e}")
    meta_reasoning_deps["validation_tracker"] = False

try:
    # Assuming 'transparency_interface' refers to a component
    # Adjust the import path and class name as needed
    from vulcan.world_model.meta_reasoning.transparency_interface import (
        TransparencyInterface,
    )  # Placeholder

    meta_reasoning_deps["transparency_interface"] = True
except ImportError as e:
    logger.debug(f"Failed to import TransparencyInterface: {e}")
    meta_reasoning_deps["transparency_interface"] = False

try:
    # Import CounterfactualObjectiveReasoner from counterfactual_objectives
    from vulcan.world_model.meta_reasoning.counterfactual_objectives import (
        CounterfactualObjectiveReasoner,
    )

    meta_reasoning_deps["counterfactual_objectives"] = True
except ImportError as e:
    logger.debug(f"Failed to import CounterfactualObjectiveReasoner: {e}")
    meta_reasoning_deps["counterfactual_objectives"] = False

# ============================================================
# UNICODE SAFE PRINTING UTILITIES
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
        status: True for success/available, False for failure/missing

    Returns:
        Status symbol string
    """
    try:
        # Try Unicode symbols
        return "✓" if status else "✗"
    except UnicodeEncodeError:
        # Fallback to ASCII
        return "[OK]" if status else "[MISSING]"


# ============================================================
# DEPENDENCY CATEGORIES
# ============================================================


class DependencyCategory:
    """Categories of dependencies for validation and initialization"""

    CORE = "core"
    SAFETY = "safety"
    MEMORY = "memory"
    PROCESSING = "processing"
    REASONING = "reasoning"
    LEARNING = "learning"
    PLANNING = "planning"
    DISTRIBUTED = "distributed"
    META_REASONING = "meta_reasoning"  # Added Category


# ============================================================
# ENHANCED DEPENDENCIES CONTAINER
# ============================================================


@dataclass
class EnhancedCollectiveDeps:
    """
    Enhanced dependencies container for all system components.

    Provides centralized management of all system dependencies with:
    - Automatic metrics initialization
    - Dependency validation
    - Graceful shutdown management
    - Component health tracking
    - Self-improvement components
    """

    # ========================================
    # CORE INFRASTRUCTURE
    # ========================================

    env: Any = None
    """Environment interface"""

    metrics: EnhancedMetricsCollector = field(default_factory=EnhancedMetricsCollector)
    """Metrics collection and monitoring system"""

    # ========================================
    # SAFETY & GOVERNANCE
    # ========================================

    safety_validator: Any = None
    """SafetyValidator - Validates actions for safety compliance"""

    governance: Any = field(default=None)
    """GovernanceOrchestrator - Policy enforcement and compliance"""

    nso_aligner: Any = field(default=None)
    """NSOAligner - Neural-Symbolic-Objective alignment"""

    explainer: Any = field(default=None)
    """ExplainabilityNode - Provides explanations for decisions"""

    # ========================================
    # MEMORY SYSTEMS
    # ========================================

    ltm: Any = None
    """MemoryIndex - Long-term memory with vector storage"""

    am: Any = None
    """EpisodicMemory - Autobiographical/episodic memory"""

    compressed_memory: Any = None
    """MemoryPersistence - Compressed memory storage"""

    # ========================================
    # PROCESSING SYSTEMS
    # ========================================

    multimodal: Any = None
    """MultimodalProcessor - Handles multi-modal inputs (text, image, audio, etc.)"""

    # ========================================
    # REASONING SYSTEMS
    # ========================================

    probabilistic: Any = None
    """ProbabilisticReasoner - Probabilistic inference and reasoning"""

    symbolic: Any = None
    """SymbolicReasoner - Symbolic logic and rule-based reasoning"""

    causal: Any = None
    """EnhancedCausalReasoning - Causal inference and modeling"""

    abstract: Any = None
    """AnalogicalReasoner - Analogical and abstract reasoning"""

    cross_modal: Any = None
    """CrossModalReasoner - Cross-modal pattern recognition"""
    
    # INTEGRATION FIX: Add UnifiedReasoner as the main reasoning orchestrator
    unified_reasoner: Any = None
    """UnifiedReasoner - Main reasoning orchestrator with tool selection"""

    # ========================================
    # LEARNING SYSTEMS
    # ========================================

    continual: Any = None
    """ContinualLearner - Continual learning without catastrophic forgetting"""

    compositional: Any = None
    """CompositionalUnderstanding - Compositional concept learning"""

    meta_cognitive: Any = (
        None  # Keep this? Meta-reasoning covers much of it. Decide based on implementation.
    )
    """MetaCognitiveMonitor - Meta-cognitive monitoring and self-reflection"""

    world_model: Any = None
    """UnifiedWorldModel - World model for prediction and planning"""

    experiment_generator: Any = None
    """ExperimentGenerator - Generates experiments for knowledge gaps"""

    problem_executor: Any = None
    """ProblemExecutor - Executes problem decomposition plans"""

    learning_system: Any = None
    """UnifiedLearningSystem - Unified learning system for outcome recording, feedback, and integration with reasoning engines (e.g., cryptographic fast-path)"""

    # ========================================
    # META-REASONING SYSTEMS (Specific learning/reasoning sub-category)
    # ========================================

    self_improvement_drive: Any = None  # Moved here for grouping
    """SelfImprovementDrive - Autonomous improvement loop"""

    motivational_introspection: Any = field(default=None)
    """MotivationalIntrospection - Introspection on agent motivations"""

    objective_hierarchy: Any = field(default=None)
    """ObjectiveHierarchy - Hierarchical objective management"""

    objective_negotiator: Any = field(default=None)
    """ObjectiveNegotiator - Negotiates between competing objectives"""

    goal_conflict_detector: Any = field(default=None)
    """GoalConflictDetector - Detects conflicts between goals"""

    preference_learner: Any = field(default=None)
    """PreferenceLearner - Learns preferences from interactions"""

    value_evolution_tracker: Any = field(default=None)
    """ValueEvolutionTracker - Tracks evolution of agent values"""

    ethical_boundary_monitor: Any = field(default=None)
    """EthicalBoundaryMonitor - Monitors ethical boundaries"""

    curiosity_reward_shaper: Any = field(default=None)
    """CuriosityRewardShaper - Shapes rewards based on curiosity"""

    internal_critic: Any = field(default=None)
    """InternalCritic - Internal critique mechanism"""

    auto_apply_policy: Any = field(default=None)  # Added
    """Component for automatically applying learned policies"""

    validation_tracker: Any = field(default=None)  # Added
    """Component for tracking validation results over time"""

    transparency_interface: Any = field(default=None)  # Added
    """Interface for providing transparency into meta-reasoning"""

    counterfactual_objectives: Any = field(default=None)  # Added
    """Component for exploring counterfactual objectives"""

    # ========================================
    # PLANNING & GOALS
    # ========================================

    goal_system: Any = None
    """HierarchicalGoalSystem - Hierarchical goal management"""

    resource_compute: Any = None
    """ResourceAwareCompute - Resource-aware computation and planning"""

    # ========================================
    # DISTRIBUTED PROCESSING
    # ========================================

    distributed: Optional[Any] = None
    """DistributedCoordinator - Distributed computation coordination"""

    # ========================================
    # INTERNAL STATE
    # ========================================

    _initialized: bool = field(default=False, init=False, repr=False)
    """Whether dependencies have been initialized"""

    _shutdown: bool = field(default=False, init=False, repr=False)
    """Whether shutdown has been called"""

    # Class-level mapping of field names to categories
    _field_categories: ClassVar[Dict[str, str]] = {
        "env": DependencyCategory.CORE,
        "metrics": DependencyCategory.CORE,
        "safety_validator": DependencyCategory.SAFETY,
        "governance": DependencyCategory.SAFETY,
        "nso_aligner": DependencyCategory.SAFETY,
        "explainer": DependencyCategory.SAFETY,
        "ltm": DependencyCategory.MEMORY,
        "am": DependencyCategory.MEMORY,
        "compressed_memory": DependencyCategory.MEMORY,
        "multimodal": DependencyCategory.PROCESSING,
        "probabilistic": DependencyCategory.REASONING,
        "symbolic": DependencyCategory.REASONING,
        "causal": DependencyCategory.REASONING,
        "abstract": DependencyCategory.REASONING,
        "cross_modal": DependencyCategory.REASONING,
        "continual": DependencyCategory.LEARNING,
        "compositional": DependencyCategory.LEARNING,
        "meta_cognitive": DependencyCategory.LEARNING,
        "world_model": DependencyCategory.LEARNING,
        "experiment_generator": DependencyCategory.LEARNING,
        "problem_executor": DependencyCategory.LEARNING,
        "learning_system": DependencyCategory.LEARNING,
        "self_improvement_drive": DependencyCategory.META_REASONING,
        "motivational_introspection": DependencyCategory.META_REASONING,
        "objective_hierarchy": DependencyCategory.META_REASONING,
        "objective_negotiator": DependencyCategory.META_REASONING,
        "goal_conflict_detector": DependencyCategory.META_REASONING,
        "preference_learner": DependencyCategory.META_REASONING,
        "value_evolution_tracker": DependencyCategory.META_REASONING,
        "ethical_boundary_monitor": DependencyCategory.META_REASONING,
        "curiosity_reward_shaper": DependencyCategory.META_REASONING,
        "internal_critic": DependencyCategory.META_REASONING,
        "auto_apply_policy": DependencyCategory.META_REASONING,
        "validation_tracker": DependencyCategory.META_REASONING,
        "transparency_interface": DependencyCategory.META_REASONING,
        "counterfactual_objectives": DependencyCategory.META_REASONING,
        "goal_system": DependencyCategory.PLANNING,
        "resource_compute": DependencyCategory.PLANNING,
        "distributed": DependencyCategory.DISTRIBUTED,
    }

    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._initialized = True
        logger.info("EnhancedCollectiveDeps initialized")

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate dependencies and return validation report

        Returns:
            Dictionary mapping categories to lists of missing/invalid dependencies
        """
        validation_report = {
            DependencyCategory.CORE: [],
            DependencyCategory.SAFETY: [],
            DependencyCategory.MEMORY: [],
            DependencyCategory.PROCESSING: [],
            DependencyCategory.REASONING: [],
            DependencyCategory.LEARNING: [],
            DependencyCategory.PLANNING: [],
            DependencyCategory.DISTRIBUTED: [],
            DependencyCategory.META_REASONING: [],  # Added
        }

        # FIXED: Iterate over dataclass fields instead of hardcoded lists
        for field_name, category in self._field_categories.items():
            if getattr(self, field_name, None) is None:
                validation_report[category].append(field_name)

        # Handle 'metrics' specially as it's auto-initialized
        if self.metrics is None:
            validation_report[DependencyCategory.CORE].append("metrics")
        elif "metrics" in validation_report[DependencyCategory.CORE]:
            validation_report[DependencyCategory.CORE].remove(
                "metrics"
            )  # Remove if it was added

        return validation_report

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status report

        Returns:
            Dictionary with status information
        """
        # FIXED: Dynamically build dependency lists from dataclass fields
        # FIXED: Iterate keys and check isupper() to filter class attributes
        all_deps = {
            value: []
            for key, value in DependencyCategory.__dict__.items()
            if key.isupper() and isinstance(value, str)
        }
        available = {category: [] for category in all_deps}
        missing = {category: [] for category in all_deps}

        total_deps = 0
        available_count = 0

        for field_name, category in self._field_categories.items():
            if category not in all_deps:
                all_deps[category] = []  # Ensure category exists
            all_deps[category].append(field_name)
            total_deps += 1

            if hasattr(self, field_name) and getattr(self, field_name) is not None:
                if category not in available:
                    available[category] = []  # Ensure category exists
                available[category].append(field_name)
                available_count += 1
            else:
                if category not in missing:
                    missing[category] = []  # Ensure category exists
                # Check if it's a component that might not be imported
                if (
                    category == DependencyCategory.META_REASONING
                    and not meta_reasoning_deps.get(field_name, True)
                ):
                    missing[category].append(f"{field_name} (import failed)")
                elif category == DependencyCategory.LEARNING and not learning_deps.get(
                    field_name, True
                ):
                    missing[category].append(f"{field_name} (import failed)")
                elif (
                    category == DependencyCategory.DISTRIBUTED
                    and not distributed_deps.get(field_name, True)
                ):
                    missing[category].append(f"{field_name} (import failed)")
                else:
                    missing[category].append(field_name)

        # Handle 'metrics' specially
        if "metrics" not in all_deps[DependencyCategory.CORE]:
            all_deps[DependencyCategory.CORE].append("metrics")
            total_deps += 1
        if self.metrics is not None:
            if "metrics" not in available[DependencyCategory.CORE]:
                available[DependencyCategory.CORE].append("metrics")
                available_count += 1
        else:
            if "metrics" not in missing[DependencyCategory.CORE]:
                missing[DependencyCategory.CORE].append("metrics")

        missing_count = total_deps - available_count

        # Check if self-improvement core drive is enabled
        self_improvement_enabled = self.self_improvement_drive is not None

        return {
            "initialized": self._initialized,
            "shutdown": self._shutdown,
            "complete": missing_count == 0,
            "total_dependencies": total_deps,
            "available_count": available_count,
            "missing_count": missing_count,
            "distributed_enabled": self.distributed is not None,
            "self_improvement_enabled": self_improvement_enabled,
            "available_by_category": available,
            "missing_by_category": missing,
            "learning_import_status": learning_deps,  # Include learning import status
            "distributed_import_status": distributed_deps,  # Include distributed import status
            "meta_reasoning_import_status": meta_reasoning_deps,  # Include meta-reasoning import status
        }

    def get_available_components(self) -> Set[str]:
        """
        Get set of available component names

        Returns:
            Set of component names that are initialized
        """
        components = set()

        # FIXED: Iterate over dataclass fields
        for field_name in self.__dataclass_fields__:
            attr_value = getattr(self, field_name)
            if attr_value is not None:
                components.add(field_name)

        return components

    def is_complete(self) -> bool:
        """
        Check if all dependencies are loaded and complete.

        Returns:
            True if all components are loaded, False otherwise
        """
        return self.get_status().get("complete", False)

    def shutdown_all(self):
        """
        Shutdown all components gracefully
        """
        if self._shutdown:
            logger.warning("Already shutdown")
            return

        logger.info("Starting shutdown of all dependencies...")

        # Shutdown order (reverse of initialization)
        shutdown_order = [
            # Distributed first
            ("distributed", self.distributed),
            # Planning
            ("goal_system", self.goal_system),
            ("resource_compute", self.resource_compute),
            # Meta-Reasoning
            ("self_improvement_drive", self.self_improvement_drive),
            ("motivational_introspection", self.motivational_introspection),
            ("objective_hierarchy", self.objective_hierarchy),
            ("objective_negotiator", self.objective_negotiator),
            ("goal_conflict_detector", self.goal_conflict_detector),
            ("preference_learner", self.preference_learner),
            ("value_evolution_tracker", self.value_evolution_tracker),
            ("ethical_boundary_monitor", self.ethical_boundary_monitor),
            ("curiosity_reward_shaper", self.curiosity_reward_shaper),
            ("internal_critic", self.internal_critic),
            ("auto_apply_policy", self.auto_apply_policy),
            ("validation_tracker", self.validation_tracker),
            ("transparency_interface", self.transparency_interface),
            ("counterfactual_objectives", self.counterfactual_objectives),
            # General Learning
            ("experiment_generator", self.experiment_generator),
            ("problem_executor", self.problem_executor),
            ("world_model", self.world_model),
            ("continual", self.continual),
            ("compositional", self.compositional),
            ("meta_cognitive", self.meta_cognitive),
            # Reasoning
            ("probabilistic", self.probabilistic),
            ("symbolic", self.symbolic),
            ("causal", self.causal),
            ("abstract", self.abstract),
            ("cross_modal", self.cross_modal),
            # Processing
            ("multimodal", self.multimodal),
            # Memory
            ("ltm", self.ltm),
            ("am", self.am),
            ("compressed_memory", self.compressed_memory),
            # Safety
            ("safety_validator", self.safety_validator),
            ("governance", self.governance),
            ("nso_aligner", self.nso_aligner),
            ("explainer", self.explainer),
            # Core
            ("metrics", self.metrics),
            # Env usually doesn't need shutdown
        ]

        for name, component in shutdown_order:
            if component is not None:
                try:
                    if hasattr(component, "shutdown"):
                        component.shutdown()
                        logger.info(f"Shutdown {name}")
                    elif hasattr(component, "close"):
                        component.close()
                        logger.info(f"Closed {name}")
                    # else: component might not require explicit shutdown
                except Exception as e:
                    logger.error(f"Error shutting down/closing {name}: {e}")

        self._shutdown = True
        logger.info("All dependencies shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if not self._shutdown:
                self.shutdown_all()
        except Exception as e:
            try:
                logger.error(f"Error during shutdown in destructor: {e}", exc_info=True)
            except (
                Exception
            ):  # nosec B110 - Logger may be unavailable during interpreter shutdown
                pass


# ============================================================
# FACTORY FUNCTIONS
# ============================================================


# Default configuration constants for minimal initialization
DEFAULT_MAX_AGENTS = 8


def create_minimal_deps(
    enable_learning: bool = False,
    enable_distributed: bool = False,
    enable_meta_reasoning: bool = False,
    max_agents: int = DEFAULT_MAX_AGENTS,
) -> EnhancedCollectiveDeps:
    """
    Create dependencies with core components and optionally learning/meta-reasoning

    Uses minimal/default configuration for all components. For more extensive
    configuration options, use create_full_deps() instead.

    Args:
        enable_learning: Whether to initialize learning components (continual learner)
        enable_distributed: Whether to initialize distributed coordinator
        enable_meta_reasoning: Whether to initialize meta-reasoning components
        max_agents: Maximum number of distributed agents (default: 8)

    Returns:
        EnhancedCollectiveDeps with minimal configuration plus optional components
    """
    deps = EnhancedCollectiveDeps()

    # If any optional components requested, initialize them with minimal config
    if enable_learning or enable_distributed or enable_meta_reasoning:
        # Initialize learning components - minimal config (no parameters required)
        if enable_learning and learning_deps.get("continual", False):
            try:
                from vulcan.learning.continual_learning import ContinualLearner

                deps.continual = ContinualLearner()
                logger.info("✓ ContinualLearner initialized (minimal config)")
            except Exception as e:
                logger.debug(f"Could not initialize ContinualLearner: {e}")

        # Initialize distributed coordinator - minimal config (configurable max_agents)
        if enable_distributed and distributed_deps.get("distributed", False):
            try:
                from vulcan.planning import DistributedCoordinator

                deps.distributed = DistributedCoordinator(max_agents=max_agents)
                logger.info(
                    f"✓ DistributedCoordinator initialized (minimal config, max_agents={max_agents})"
                )
            except Exception as e:
                logger.debug(f"Could not initialize DistributedCoordinator: {e}")

        # Initialize meta-reasoning components - minimal config via helper
        if enable_meta_reasoning:
            _initialize_meta_reasoning_components(deps)

    return deps


def _initialize_meta_reasoning_components(deps: EnhancedCollectiveDeps) -> None:
    """
    Helper function to initialize meta-reasoning components with minimal configuration

    Initializes components with default parameters suitable for minimal setups.
    For more extensive configuration, use create_full_deps() instead.

    Args:
        deps: EnhancedCollectiveDeps instance to populate
    """
    # PreferenceLearner - minimal config
    if meta_reasoning_deps.get("preference_learner", False):
        try:
            from vulcan.world_model.meta_reasoning.preference_learner import (
                PreferenceLearner,
            )

            deps.preference_learner = PreferenceLearner()
            logger.info("✓ PreferenceLearner initialized (minimal config)")
        except Exception as e:
            logger.debug(f"Could not initialize PreferenceLearner: {e}")

    # ValueEvolutionTracker - minimal config
    if meta_reasoning_deps.get("value_evolution_tracker", False):
        try:
            from vulcan.world_model.meta_reasoning.value_evolution_tracker import (
                ValueEvolutionTracker,
            )

            deps.value_evolution_tracker = ValueEvolutionTracker()
            logger.info("✓ ValueEvolutionTracker initialized (minimal config)")
        except Exception as e:
            logger.debug(f"Could not initialize ValueEvolutionTracker: {e}")

    # EthicalBoundaryMonitor - minimal config
    # Note: load_defaults=False to avoid automatic boundary loading in minimal setups
    if meta_reasoning_deps.get("ethical_boundary_monitor", False):
        try:
            from vulcan.world_model.meta_reasoning.ethical_boundary_monitor import (
                EthicalBoundaryMonitor,
            )

            deps.ethical_boundary_monitor = EthicalBoundaryMonitor(load_defaults=False)
            logger.info("✓ EthicalBoundaryMonitor initialized (minimal config)")
        except Exception as e:
            logger.debug(f"Could not initialize EthicalBoundaryMonitor: {e}")

    # CuriosityRewardShaper - minimal config
    if meta_reasoning_deps.get("curiosity_reward_shaper", False):
        try:
            from vulcan.world_model.meta_reasoning.curiosity_reward_shaper import (
                CuriosityRewardShaper,
            )

            deps.curiosity_reward_shaper = CuriosityRewardShaper()
            logger.info("✓ CuriosityRewardShaper initialized (minimal config)")
        except Exception as e:
            logger.debug(f"Could not initialize CuriosityRewardShaper: {e}")

    # InternalCritic - minimal config
    # Note: ethical_boundary_monitor reference is optional and will be None if not initialized
    if meta_reasoning_deps.get("internal_critic", False):
        try:
            from vulcan.world_model.meta_reasoning.internal_critic import InternalCritic

            # Only pass ethical_boundary_monitor if it was successfully initialized
            ethical_monitor = getattr(deps, "ethical_boundary_monitor", None)
            deps.internal_critic = InternalCritic(
                ethical_boundary_monitor=ethical_monitor
            )
            logger.info("✓ InternalCritic initialized (minimal config)")
        except Exception as e:
            logger.debug(f"Could not initialize InternalCritic: {e}")


def create_full_deps(
    config: Any = None, env: Any = None, enable_distributed: bool = False, **kwargs
) -> EnhancedCollectiveDeps:
    """
    Create fully initialized dependencies container with extensive configuration

    Initializes components with full configuration options including cross-dependencies
    and integration with validation, transparency, and other subsystems. For simpler
    initialization with defaults, use create_minimal_deps() instead.

    Args:
        config: Configuration object (AgentConfig) with component-specific settings
        env: Environment interface
        enable_distributed: Whether to enable distributed processing
        **kwargs: Additional component instances (will override auto-initialization)

    Returns:
        EnhancedCollectiveDeps with components initialized based on availability

    Note:
        - Components are only initialized if their imports succeed (checked via *_deps dicts)
        - Provided kwargs override auto-initialization for specific components
        - Meta-reasoning components get full configuration with cross-dependencies
        - Use create_minimal_deps() for simpler initialization with default parameters
    """
    deps = EnhancedCollectiveDeps(env=env)
    logger.info("Initializing full dependencies container...")

    # Set any explicitly provided components first
    for key, value in kwargs.items():
        if hasattr(deps, key):
            setattr(deps, key, value)
        else:
            logger.warning(f"Unknown dependency provided via kwargs: {key}")

    # ========================================
    # LEARNING: Initialize continual learner
    # ========================================
    if deps.continual is None and learning_deps.get("continual", False):
        try:
            from vulcan.learning.continual_learning import ContinualLearner

            deps.continual = ContinualLearner()
            logger.info("✓ ContinualLearner initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ContinualLearner: {e}")

    # ========================================
    # DISTRIBUTED: Initialize distributed coordinator
    # ========================================
    if (
        enable_distributed
        and deps.distributed is None
        and distributed_deps.get("distributed", False)
    ):
        try:
            from vulcan.planning import DistributedCoordinator

            max_agents = getattr(config, "max_distributed_agents", 8) if config else 8
            deps.distributed = DistributedCoordinator(max_agents=max_agents)
            logger.info(
                f"✓ DistributedCoordinator initialized with max_agents={max_agents}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize DistributedCoordinator: {e}")

    # ========================================
    # META_REASONING: Initialize meta-reasoning components
    # ========================================

    # PreferenceLearner
    if deps.preference_learner is None and meta_reasoning_deps.get(
        "preference_learner", False
    ):
        try:
            from vulcan.world_model.meta_reasoning.preference_learner import (
                PreferenceLearner,
            )

            deps.preference_learner = PreferenceLearner(
                decay_rate=0.99,
                exploration_bonus=0.1,
                validation_tracker=deps.validation_tracker,
                transparency_interface=deps.transparency_interface,
            )
            logger.info("✓ PreferenceLearner initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PreferenceLearner: {e}")

    # ValueEvolutionTracker
    if deps.value_evolution_tracker is None and meta_reasoning_deps.get(
        "value_evolution_tracker", False
    ):
        try:
            from vulcan.world_model.meta_reasoning.value_evolution_tracker import (
                ValueEvolutionTracker,
            )

            deps.value_evolution_tracker = ValueEvolutionTracker(
                max_history=10000,
                drift_threshold=0.15,
                self_improvement_drive=deps.self_improvement_drive,
                validation_tracker=deps.validation_tracker,
                transparency_interface=deps.transparency_interface,
            )
            logger.info("✓ ValueEvolutionTracker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ValueEvolutionTracker: {e}")

    # EthicalBoundaryMonitor
    if deps.ethical_boundary_monitor is None and meta_reasoning_deps.get(
        "ethical_boundary_monitor", False
    ):
        try:
            from vulcan.world_model.meta_reasoning.ethical_boundary_monitor import (
                EthicalBoundaryMonitor,
            )

            deps.ethical_boundary_monitor = EthicalBoundaryMonitor(
                strict_mode=False,
                validation_tracker=deps.validation_tracker,
                self_improvement_drive=deps.self_improvement_drive,
                transparency_interface=deps.transparency_interface,
                load_defaults=False,  # Don't load defaults to avoid side effects
            )
            logger.info("✓ EthicalBoundaryMonitor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize EthicalBoundaryMonitor: {e}")

    # CuriosityRewardShaper
    if deps.curiosity_reward_shaper is None and meta_reasoning_deps.get(
        "curiosity_reward_shaper", False
    ):
        try:
            from vulcan.world_model.meta_reasoning.curiosity_reward_shaper import (
                CuriosityRewardShaper,
            )

            deps.curiosity_reward_shaper = CuriosityRewardShaper(
                curiosity_weight=0.1,
                decay_rate=0.99,
                max_bonus=1.0,
                world_model=deps.world_model,
                validation_tracker=deps.validation_tracker,
                transparency_interface=deps.transparency_interface,
            )
            logger.info("✓ CuriosityRewardShaper initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CuriosityRewardShaper: {e}")

    # InternalCritic
    if deps.internal_critic is None and meta_reasoning_deps.get(
        "internal_critic", False
    ):
        try:
            from vulcan.world_model.meta_reasoning.internal_critic import InternalCritic

            # Only pass ethical_boundary_monitor if it exists (may be None)
            ethical_monitor = getattr(deps, "ethical_boundary_monitor", None)
            deps.internal_critic = InternalCritic(
                strict_mode=False,
                max_history=10000,
                validation_tracker=deps.validation_tracker,
                ethical_boundary_monitor=ethical_monitor,  # Safe to pass None
                transparency_interface=deps.transparency_interface,
            )
            logger.info("✓ InternalCritic initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize InternalCritic: {e}")

    logger.info("✓ Full dependencies initialization complete")
    return deps


def create_self_improving_deps(
    config: Any, env: Any = None, enable_distributed: bool = False, **kwargs
) -> EnhancedCollectiveDeps:
    """
    Create dependencies with self-improvement components initialized.

    This factory function creates a fully-equipped dependency container with
    experiment generation and problem execution capabilities for autonomous
    self-improvement.

    Args:
        config: Configuration object (AgentConfig) with enable_self_improvement flag
        env: Environment interface
        enable_distributed: Whether to enable distributed processing
        **kwargs: Additional component instances (safety_validator, semantic_bridge, etc.)

    Returns:
        EnhancedCollectiveDeps with self-improvement components initialized

    Example:
        >>> config = AgentConfig(enable_self_improvement=True)
        >>> deps = create_self_improving_deps(config)
        >>> # deps.experiment_generator and deps.problem_executor are now available
    """
    # Create base dependencies using the placeholder factory
    deps = create_full_deps(
        config=config, env=env, enable_distributed=enable_distributed, **kwargs
    )

    # Initialize self-improvement components if enabled in config
    if hasattr(config, "enable_self_improvement") and config.enable_self_improvement:
        logger.info("Attempting to initialize self-improvement components...")

        # --- Initialize ExperimentGenerator ---
        if deps.experiment_generator is None:
            try:
                # Import here to potentially avoid import loops if called early
                from ..curiosity_engine.experiment_generator import ExperimentGenerator

                deps.experiment_generator = ExperimentGenerator(
                    default_timeout=getattr(config, "exp_gen_timeout", 30.0),
                    max_complexity=getattr(config, "exp_gen_max_complexity", 1.0),
                    # Add other necessary args based on ExperimentGenerator.__init__
                )
                logger.info("✓ ExperimentGenerator initialized")
            except ImportError:
                logger.error(
                    "Failed to import ExperimentGenerator. Self-improvement may be limited."
                )
            except Exception as e:
                logger.error(f"Failed to initialize ExperimentGenerator: {e}")

        # --- Initialize ProblemExecutor ---
        if deps.problem_executor is None:
            try:
                # Import here
                from ..problem_decomposer.problem_executor import ProblemExecutor

                # ProblemExecutor likely needs other dependencies like validator, bridge
                semantic_bridge = kwargs.get(
                    "semantic_bridge", getattr(deps, "semantic_bridge", None)
                )
                if deps.safety_validator and semantic_bridge:
                    deps.problem_executor = ProblemExecutor(
                        validator=deps.safety_validator,
                        semantic_bridge=semantic_bridge,
                        # Add other necessary args based on ProblemExecutor.__init__
                    )
                    logger.info("✓ ProblemExecutor initialized")
                else:
                    logger.warning(
                        "Cannot initialize ProblemExecutor: missing safety_validator or semantic_bridge."
                    )
            except ImportError:
                logger.error(
                    "Failed to import ProblemExecutor. Self-improvement may be limited."
                )
            except Exception as e:
                logger.error(f"Failed to initialize ProblemExecutor: {e}")

        # --- Initialize SelfImprovementDrive ---
        if deps.self_improvement_drive is None and meta_reasoning_deps.get(
            "self_improvement_drive", False
        ):
            try:
                # SelfImprovementDrive initialization is complex and depends on many other components.
                # It's often better handled within the WorldModel or Orchestrator initialization
                # once all other dependencies are ready. Here, we just log its availability.
                # If direct initialization is needed, ensure all its required deps (world_model, etc.)
                # are passed or available in 'deps'.
                # Example (conceptual):
                # if deps.world_model:
                #     deps.self_improvement_drive = SelfImprovementDrive(
                #         world_model=deps.world_model,
                #         config_path=config.self_improvement_config,
                #         state_path=config.self_improvement_state
                #     )
                #     logger.info("✓ SelfImprovementDrive initialized")
                # else:
                #      logger.warning("Cannot initialize SelfImprovementDrive: world_model missing.")
                logger.info(
                    "✓ SelfImprovementDrive module available (initialization deferred)"
                )
            except Exception as e:
                logger.error(f"Failed during SelfImprovementDrive preparation: {e}")

        # --- Initialize other Meta-Reasoning Components ---
        # Similar pattern: check availability, import, initialize if deps are met.
        # Example for MotivationalIntrospection:
        if deps.motivational_introspection is None and meta_reasoning_deps.get(
            "motivational_introspection", False
        ):
            try:
                from vulcan.world_model.meta_reasoning.motivational_introspection import (
                    MotivationalIntrospection,
                )

                if deps.world_model:  # Assuming it needs world_model
                    deps.motivational_introspection = MotivationalIntrospection(
                        world_model=deps.world_model,
                        design_spec=config,  # Assuming it needs config
                    )
                    logger.info("✓ MotivationalIntrospection initialized")
                else:
                    logger.warning(
                        "Cannot initialize MotivationalIntrospection: world_model missing."
                    )
            except ImportError:
                logger.error("Failed to import MotivationalIntrospection.")
            except Exception as e:
                logger.error(f"Failed to initialize MotivationalIntrospection: {e}")

        # ... Add similar blocks for other meta-reasoning components as needed ...
        # (ObjectiveHierarchy, ObjectiveNegotiator, GoalConflictDetector, etc.)
        # Ensure their dependencies are available in 'deps' or passed via kwargs.

        logger.info("✓ Self-improvement component initialization attempted.")

    else:
        logger.info(
            "Self-improvement disabled in configuration, skipping related component initialization."
        )

    return deps


def wire_world_model_components(
    deps: EnhancedCollectiveDeps, world_model: Any
) -> EnhancedCollectiveDeps:
    """
    Wire WorldModel's meta-reasoning components to the orchestrator's deps.
    
    This function solves the architectural mismatch where:
    - Meta-reasoning components are initialized in WorldModel (world_model_core.py)
    - But the dependency validator checks deps.internal_critic, deps.preference_learner, etc.
    - Without this wiring, components appear as "Missing/Not Initialized" even though
      they are fully functional in WorldModel
    
    Args:
        deps: The EnhancedCollectiveDeps instance to populate
        world_model: The WorldModel instance containing initialized meta-reasoning components
            (type is Any to avoid circular imports with world_model_core)
        
    Returns:
        The deps object with meta-reasoning components wired from WorldModel
        
    Example:
        >>> from vulcan.orchestrator.dependencies import wire_world_model_components
        >>> wire_world_model_components(deps, world_model)
        >>> # Now deps.internal_critic, deps.preference_learner, etc. are populated
    """
    if world_model is None:
        logger.debug("No WorldModel provided, skipping meta-reasoning component wiring")
        return deps
    
    # Map of deps field names to WorldModel attribute names
    # Note: Some names differ between deps and WorldModel (e.g., counterfactual_objectives
    # vs counterfactual_objective_reasoner) due to historical naming conventions
    component_mapping = {
        # Meta-reasoning components
        "internal_critic": "internal_critic",
        "preference_learner": "preference_learner",
        "ethical_boundary_monitor": "ethical_boundary_monitor",
        "curiosity_reward_shaper": "curiosity_reward_shaper",
        "value_evolution_tracker": "value_evolution_tracker",
        "validation_tracker": "validation_tracker",
        "transparency_interface": "transparency_interface",
        # Note: deps uses "counterfactual_objectives", WorldModel uses "counterfactual_objective_reasoner"
        "counterfactual_objectives": "counterfactual_objective_reasoner",
        "goal_conflict_detector": "goal_conflict_detector",
        "objective_negotiator": "objective_negotiator",
        "objective_hierarchy": "objective_hierarchy",
        "motivational_introspection": "motivational_introspection",
        "self_improvement_drive": "self_improvement_drive",
        # Also wire the world_model itself
        "world_model": None,  # Special case - wire the world_model directly
    }
    
    wired_count = 0
    
    for deps_field, wm_attr in component_mapping.items():
        try:
            # Special case: wire world_model directly
            if deps_field == "world_model":
                if getattr(deps, deps_field, None) is None:
                    setattr(deps, deps_field, world_model)
                    wired_count += 1
                    logger.debug(f"✓ Wired world_model to deps")
                continue
            
            # Get component from WorldModel
            component = getattr(world_model, wm_attr, None)
            
            # Only wire if component exists in WorldModel and deps field is empty
            if component is not None and getattr(deps, deps_field, None) is None:
                setattr(deps, deps_field, component)
                wired_count += 1
                logger.debug(f"✓ Wired {deps_field} from WorldModel")
                
        except Exception as e:
            logger.debug(f"Could not wire {deps_field}: {e}")
    
    if wired_count > 0:
        logger.info(
            f"✓ Wired {wired_count} meta-reasoning components from WorldModel to deps"
        )
    
    return deps


# ============================================================
# DEPENDENCY ALIAS CONFIGURATION
# ============================================================

# Mapping of legacy/expected dependency names to their actual implementation names
# This allows for backward compatibility when dependency names change during refactoring
DEPENDENCY_ALIASES = {
    DependencyCategory.LEARNING: {
        "learning_system": "continual",  # learning_system is implemented as 'continual' (ContinualLearner)
    }
    # Add more aliases here as needed:
    # DependencyCategory.REASONING: {
    #     "old_name": "new_name",
    # }
}


def validate_dependencies(
    deps: EnhancedCollectiveDeps, required_categories: Optional[List[str]] = None
) -> bool:
    """
    Validate that dependencies meet requirements with support for aliased names.
    
    This function checks that all required dependencies are initialized, with support
    for legacy/aliased dependency names. If a dependency is missing but an alias is
    available, the validation will pass.

    Args:
        deps: Dependencies container to validate
        required_categories: List of required categories (None = all core categories)

    Returns:
        True if validation passes, False otherwise
        
    Note:
        See DEPENDENCY_ALIASES for configured name mappings.
    """
    if required_categories is None:
        # Define core functional categories (excluding optional like DISTRIBUTED, META_REASONING)
        required_categories = [
            DependencyCategory.CORE,
            DependencyCategory.SAFETY,
            DependencyCategory.MEMORY,
            DependencyCategory.PROCESSING,
            DependencyCategory.REASONING,
            DependencyCategory.LEARNING,  # Basic learning
            DependencyCategory.PLANNING,
        ]

    validation_report = deps.validate()
    all_valid = True

    for category in required_categories:
        if category in validation_report and validation_report[category]:
            # Filter out components that failed to import (logged elsewhere)
            missing_initialized = [
                dep
                for dep in validation_report[category]
                if "(import failed)" not in dep
            ]
            
            # ================================================================
            # DEPENDENCY ALIAS RESOLUTION
            # ================================================================
            # Check if any missing dependencies have available aliases/substitutes.
            # This handles cases where dependency names changed during refactoring
            # but the validation still expects the old names.
            #
            # Example: 'learning_system' was refactored to 'continual' but validation
            # still checks for 'learning_system'. If 'continual' is available,
            # consider 'learning_system' satisfied.
            # ================================================================
            if category in DEPENDENCY_ALIASES:
                aliases = DEPENDENCY_ALIASES[category]
                for missing_dep in list(missing_initialized):  # Create copy to allow removal
                    if missing_dep in aliases:
                        actual_dep_name = aliases[missing_dep]
                        # Check if the actual implementation is available
                        if hasattr(deps, actual_dep_name) and getattr(deps, actual_dep_name) is not None:
                            missing_initialized.remove(missing_dep)
                            logger.debug(
                                f"Dependency '{missing_dep}' satisfied by alias '{actual_dep_name}' "
                                f"in category '{category}'"
                            )
            
            if missing_initialized:
                logger.error(
                    f"Missing critical dependencies in category '{category}': "
                    f"{', '.join(missing_initialized)}"
                )
                all_valid = False

    if all_valid:
        logger.info("Core dependency validation passed")
    else:
        logger.error("Core dependency validation failed")

    return all_valid


def print_dependency_report(deps: EnhancedCollectiveDeps):
    """
    Print human-readable dependency report with Unicode-safe formatting

    Args:
        deps: Dependencies container
    """
    status = deps.get_status()

    safe_print("\n" + "=" * 60)
    safe_print("VULCAN-AGI DEPENDENCIES REPORT")
    safe_print("=" * 60)
    safe_print(f"Initialized: {status['initialized']}")
    safe_print(f"Complete (all components loaded): {status['complete']}")
    safe_print(f"Shutdown: {status['shutdown']}")
    safe_print(f"Total Dependencies Tracked: {status['total_dependencies']}")
    safe_print(f"Available (Initialized): {status['available_count']}")
    safe_print(f"Missing/Not Initialized: {status['missing_count']}")
    safe_print(f"Distributed Enabled: {status['distributed_enabled']}")
    safe_print(
        f"Self-Improvement Enabled: {status.get('self_improvement_enabled', False)}"
    )

    safe_print("\n" + "-" * 60)
    safe_print("IMPORT STATUS BY CATEGORY:")
    safe_print("-" * 60)

    # Learning import status
    learning_import_status = status.get("learning_import_status", {})
    if learning_import_status:
        safe_print("\nLEARNING:")
        for component, is_imported in learning_import_status.items():
            symbol = get_status_symbol(is_imported)
            safe_print(
                f"  {symbol} {component}: {'Import OK' if is_imported else 'Import FAILED'}"
            )

    # Distributed import status
    distributed_import_status = status.get("distributed_import_status", {})
    if distributed_import_status:
        safe_print("\nDISTRIBUTED:")
        for component, is_imported in distributed_import_status.items():
            symbol = get_status_symbol(is_imported)
            safe_print(
                f"  {symbol} {component}: {'Import OK' if is_imported else 'Import FAILED'}"
            )

    # Meta-reasoning import status
    mr_import_status = status.get("meta_reasoning_import_status", {})
    if mr_import_status:
        safe_print("\nMETA_REASONING:")
        for component, is_imported in mr_import_status.items():
            symbol = get_status_symbol(is_imported)
            safe_print(
                f"  {symbol} {component}: {'Import OK' if is_imported else 'Import FAILED'}"
            )

    safe_print("\n" + "-" * 60)
    safe_print("AVAILABLE DEPENDENCIES BY CATEGORY:")
    safe_print("-" * 60)
    for category, deps_list in status["available_by_category"].items():
        if deps_list:
            safe_print(f"\n{category.upper()}:")
            for dep in deps_list:
                check_symbol = get_status_symbol(True)
                safe_print(f"  {check_symbol} {dep}")

    safe_print("\n" + "-" * 60)
    safe_print("MISSING/NOT INITIALIZED DEPENDENCIES BY CATEGORY:")
    safe_print("-" * 60)
    has_missing = False
    for category, deps_list in status["missing_by_category"].items():
        if deps_list:
            has_missing = True
            safe_print(f"\n{category.upper()}:")
            for dep in deps_list:
                missing_symbol = get_status_symbol(False)
                safe_print(f"  {missing_symbol} {dep}")
    if not has_missing:
        safe_print("  None")

    safe_print("\n" + "=" * 60 + "\n")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "EnhancedCollectiveDeps",
    "DependencyCategory",
    "create_minimal_deps",
    "create_full_deps",
    "create_self_improving_deps",
    "wire_world_model_components",  # Added: Wire WorldModel components to deps
    "validate_dependencies",
    "print_dependency_report",
    "safe_print",
    "get_status_symbol",
    "learning_deps",  # Export learning dependencies status
    "distributed_deps",  # Export distributed dependencies status
    "meta_reasoning_deps",  # Export meta-reasoning dependencies status
]
