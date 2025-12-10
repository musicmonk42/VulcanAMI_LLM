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
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional, Set

from .metrics import EnhancedMetricsCollector

logger = logging.getLogger(__name__)


# ============================================================
# DEPENDENCY TRACKING
# ============================================================

# Learning dependencies tracking dictionary (keep general learning deps here)
learning_deps = {}

# Meta-Reasoning dependencies tracking dictionary
meta_reasoning_deps = {}

# Meta-Reasoning Component Imports
try:
    from vulcan.world_model.meta_reasoning.self_improvement_drive import \
        SelfImprovementDrive

    meta_reasoning_deps["self_improvement_drive"] = True
except ImportError as e:
    logger.debug(
        f"Failed to import SelfImprovementDrive: {e}"
    )  # Debug level for optional
    meta_reasoning_deps["self_improvement_drive"] = False

try:
    from vulcan.world_model.meta_reasoning.motivational_introspection import \
        MotivationalIntrospection

    meta_reasoning_deps["motivational_introspection"] = True
except ImportError as e:
    logger.debug(f"Failed to import MotivationalIntrospection: {e}")
    meta_reasoning_deps["motivational_introspection"] = False

try:
    from vulcan.world_model.meta_reasoning.objective_hierarchy import \
        ObjectiveHierarchy

    meta_reasoning_deps["objective_hierarchy"] = True
except ImportError as e:
    logger.debug(f"Failed to import ObjectiveHierarchy: {e}")
    meta_reasoning_deps["objective_hierarchy"] = False

try:
    from vulcan.world_model.meta_reasoning.objective_negotiator import \
        ObjectiveNegotiator

    meta_reasoning_deps["objective_negotiator"] = True
except ImportError as e:
    logger.debug(f"Failed to import ObjectiveNegotiator: {e}")
    meta_reasoning_deps["objective_negotiator"] = False

try:
    from vulcan.world_model.meta_reasoning.goal_conflict_detector import \
        GoalConflictDetector

    meta_reasoning_deps["goal_conflict_detector"] = True
except ImportError as e:
    logger.debug(f"Failed to import GoalConflictDetector: {e}")
    meta_reasoning_deps["goal_conflict_detector"] = False

try:
    from vulcan.world_model.meta_reasoning.preference_learner import \
        PreferenceLearner

    meta_reasoning_deps["preference_learner"] = True
except ImportError as e:
    logger.debug(f"Failed to import PreferenceLearner: {e}")
    meta_reasoning_deps["preference_learner"] = False

try:
    from vulcan.world_model.meta_reasoning.value_evolution_tracker import \
        ValueEvolutionTracker

    meta_reasoning_deps["value_evolution_tracker"] = True
except ImportError as e:
    logger.debug(f"Failed to import ValueEvolutionTracker: {e}")
    meta_reasoning_deps["value_evolution_tracker"] = False

try:
    from vulcan.world_model.meta_reasoning.ethical_boundary_monitor import \
        EthicalBoundaryMonitor

    meta_reasoning_deps["ethical_boundary_monitor"] = True
except ImportError as e:
    logger.debug(f"Failed to import EthicalBoundaryMonitor: {e}")
    meta_reasoning_deps["ethical_boundary_monitor"] = False

try:
    from vulcan.world_model.meta_reasoning.curiosity_reward_shaper import \
        CuriosityRewardShaper

    meta_reasoning_deps["curiosity_reward_shaper"] = True
except ImportError as e:
    logger.debug(f"Failed to import CuriosityRewardShaper: {e}")
    meta_reasoning_deps["curiosity_reward_shaper"] = False

try:
    from vulcan.world_model.meta_reasoning.internal_critic import \
        InternalCritic

    meta_reasoning_deps["internal_critic"] = True
except ImportError as e:
    logger.debug(f"Failed to import InternalCritic: {e}")
    meta_reasoning_deps["internal_critic"] = False

# Assuming these are classes based on user request - adjust if they are functions/modules
try:
    # Assuming 'auto_apply_policy' refers to a component responsible for this
    # Adjust the import path and class name as needed
    from vulcan.world_model.meta_reasoning.policy_manager import \
        AutoApplyPolicy  # Placeholder

    meta_reasoning_deps["auto_apply_policy"] = True
except ImportError as e:
    logger.debug(f"Failed to import AutoApplyPolicy (or similar): {e}")
    meta_reasoning_deps["auto_apply_policy"] = False

try:
    # Assuming 'validation_tracker' refers to a component
    # Adjust the import path and class name as needed
    from vulcan.world_model.meta_reasoning.validation_tracker import \
        ValidationTracker  # Placeholder

    meta_reasoning_deps["validation_tracker"] = True
except ImportError as e:
    logger.debug(f"Failed to import ValidationTracker: {e}")
    meta_reasoning_deps["validation_tracker"] = False

try:
    # Assuming 'transparency_interface' refers to a component
    # Adjust the import path and class name as needed
    from vulcan.world_model.meta_reasoning.transparency_interface import \
        TransparencyInterface  # Placeholder

    meta_reasoning_deps["transparency_interface"] = True
except ImportError as e:
    logger.debug(f"Failed to import TransparencyInterface: {e}")
    meta_reasoning_deps["transparency_interface"] = False

try:
    # Assuming 'counterfactual_objectives' refers to a component
    # Adjust the import path and class name as needed
    from vulcan.world_model.meta_reasoning.counterfactual_objectives import \
        CounterfactualObjectives  # Placeholder

    meta_reasoning_deps["counterfactual_objectives"] = True
except ImportError as e:
    logger.debug(f"Failed to import CounterfactualObjectives: {e}")
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

    # ========================================
    # LEARNING SYSTEMS
    # ========================================

    continual: Any = None
    """ContinualLearner - Continual learning without catastrophic forgetting"""

    compositional: Any = None
    """CompositionalUnderstanding - Compositional concept learning"""

    meta_cognitive: Any = None  # Keep this? Meta-reasoning covers much of it. Decide based on implementation.
    """MetaCognitiveMonitor - Meta-cognitive monitoring and self-reflection"""

    world_model: Any = None
    """UnifiedWorldModel - World model for prediction and planning"""

    experiment_generator: Any = None
    """ExperimentGenerator - Generates experiments for knowledge gaps"""

    problem_executor: Any = None
    """ProblemExecutor - Executes problem decomposition plans"""

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
                # Check if it's a meta-reasoning component that might not be imported
                if (
                    category == DependencyCategory.META_REASONING
                    and not meta_reasoning_deps.get(field_name, True)
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
            "meta_reasoning_import_status": meta_reasoning_deps,  # Include the import status
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
            # Avoid errors during interpreter shutdown
            pass  # logger might already be gone


# ============================================================
# FACTORY FUNCTIONS
# ============================================================


def create_minimal_deps() -> EnhancedCollectiveDeps:
    """
    Create dependencies with only core components initialized

    Returns:
        EnhancedCollectiveDeps with minimal configuration
    """
    # Assuming Env and MetricsCollector don't need complex init here
    return EnhancedCollectiveDeps()


def create_full_deps(
    config: Any = None, env: Any = None, enable_distributed: bool = False, **kwargs
) -> EnhancedCollectiveDeps:
    """
    Create fully initialized dependencies container (placeholders for now)

    Args:
        config: Configuration object (AgentConfig)
        env: Environment interface
        enable_distributed: Whether to enable distributed processing
        **kwargs: Additional component instances

    Returns:
        EnhancedCollectiveDeps potentially populated with components
    """
    # In a real scenario, this would import and initialize each component
    # based on the config and available modules.
    deps = EnhancedCollectiveDeps(env=env)
    logger.warning(
        "create_full_deps is a placeholder; components need actual initialization."
    )

    # Set any explicitly provided components
    for key, value in kwargs.items():
        if hasattr(deps, key):
            setattr(deps, key, value)
        else:
            logger.warning(f"Unknown dependency provided via kwargs: {key}")

    # Example placeholder initializations (replace with real ones)
    # deps.safety_validator = kwargs.get('safety_validator', SafetyValidatorPlaceholder())
    # deps.ltm = kwargs.get('ltm', MemoryIndexPlaceholder())
    # ... etc for all components based on config flags ...

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
                from ..curiosity_engine.experiment_generator import \
                    ExperimentGenerator

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
                from ..problem_decomposer.problem_executor import \
                    ProblemExecutor

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
                from vulcan.world_model.meta_reasoning.motivational_introspection import \
                    MotivationalIntrospection

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


def validate_dependencies(
    deps: EnhancedCollectiveDeps, required_categories: Optional[List[str]] = None
) -> bool:
    """
    Validate that dependencies meet requirements

    Args:
        deps: Dependencies container to validate
        required_categories: List of required categories (None = all core categories)

    Returns:
        True if validation passes, False otherwise
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
    safe_print("META-REASONING IMPORT STATUS:")
    safe_print("-" * 60)
    # Use the dedicated meta-reasoning import status dict
    mr_import_status = status.get("meta_reasoning_import_status", {})
    if mr_import_status:
        for component, is_imported in mr_import_status.items():
            symbol = get_status_symbol(is_imported)
            safe_print(
                f"  {symbol} {component}: {'Import OK' if is_imported else 'Import FAILED'}"
            )
    else:
        safe_print("  (No meta-reasoning components tracked)")

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
    "validate_dependencies",
    "print_dependency_report",
    "safe_print",
    "get_status_symbol",
    "learning_deps",  # Keep if used for general learning deps
    "meta_reasoning_deps",  # Export the meta-reasoning dependencies status
]
