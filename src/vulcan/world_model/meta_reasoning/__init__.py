# src/vulcan/world_model/meta_reasoning/__init__.py
"""
Meta-reasoning layer for VULCAN-AMI
Provides goal-level reasoning about objectives, conflicts, and alignment

This subsystem enables the system to reason about its own objectives:
- Understand what it's optimizing for (motivational introspection)
- Detect conflicts between objectives (goal conflict detection)
- Reason about alternative objectives (counterfactual reasoning)
- Negotiate between competing objectives (multi-agent negotiation)
- Track validation history and learn patterns (validation tracking)
- Provide machine-readable transparency (agent communication)
- Autonomous self-improvement as a core drive

Core Components:
- MotivationalIntrospection: Main orchestrator for goal-level reasoning
- ObjectiveHierarchy: Manages objective relationships and dependencies
- CounterfactualObjectiveReasoner: "What if I optimized for X instead?"
- GoalConflictDetector: Detects and analyzes objective conflicts
- ObjectiveNegotiator: Resolves conflicts through multi-agent negotiation
- ValidationTracker: Learns from validation history
- TransparencyInterface: Machine-readable output for agents
- SelfImprovementDrive: Autonomous self-improvement as intrinsic drive
- InternalCritic: Multi-perspective self-critique and evaluation
- CuriosityRewardShaper: Curiosity-driven exploration and reward shaping
- EthicalBoundaryMonitor: Ethical boundary monitoring and enforcement
- PreferenceLearner: Bayesian preference learning
- ValueEvolutionTracker: Tracking agent value evolution over time
- AutoApplyPolicy: Policy engine for automated code changes

Usage:
    from world_model.meta_reasoning import MotivationalIntrospection, SelfImprovementDrive

    mi = MotivationalIntrospection(world_model, design_spec)
    validation = mi.validate_proposal_alignment(proposal)

    si_drive = SelfImprovementDrive(config_path="configs/intrinsic_drives.json")
    if si_drive.should_trigger(context):
        improvement = si_drive.step(context)
"""

import logging
logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "VULCAN-AMI Team"

# Core components
from .motivational_introspection import (
    MotivationalIntrospection,
    ObjectiveStatus,
    ObjectiveAnalysis,
    ProposalValidation
)

from .objective_hierarchy import (
    ObjectiveHierarchy,
    Objective,
    ObjectiveType,
    ConflictType as HierarchyConflictType
)

from .counterfactual_objectives import (
    CounterfactualObjectiveReasoner,
    CounterfactualOutcome,
    ObjectiveComparison,
    ParetoPoint
)

from .goal_conflict_detector import (
    GoalConflictDetector,
    Conflict,
    ConflictSeverity,
    ConflictType,
    MultiObjectiveTension
)

from .objective_negotiator import (
    ObjectiveNegotiator,
    AgentProposal,
    NegotiationResult,
    NegotiationStrategy,
    NegotiationOutcome,
    ConflictResolution
)

from .validation_tracker import (
    ValidationTracker,
    ValidationRecord,
    ValidationPattern,
    LearningInsight,
    PatternType,
    ValidationOutcome,
    ObjectiveBlocker
)

from .transparency_interface import (
    TransparencyInterface,
    SerializationFormat,
    TransparencyMetadata
)

from .self_improvement_drive import (
    SelfImprovementDrive,
    TriggerType,
    FailureType,
    ImprovementObjective,
    SelfImprovementState
)

# Added import for InternalCritic and related items
from .internal_critic import (
    InternalCritic,
    Critique,
    Evaluation,
    Risk,
    ComparisonResult,
    PerspectiveScore,
    CritiqueLevel,
    EvaluationPerspective,
    RiskCategory,
    RiskSeverity
)

from .curiosity_reward_shaper import (
    CuriosityRewardShaper,
    NoveltyEstimate,
    EpisodicMemory,
    CuriosityStatistics,
    CuriosityMethod,
    NoveltyLevel
)

from .ethical_boundary_monitor import (
    EthicalBoundaryMonitor,
    EthicalBoundary,
    EthicalViolation,
    EnforcementAction,
    BoundaryCategory,
    ViolationSeverity,
    EnforcementLevel,
    BoundaryType
)

from .preference_learner import (
    PreferenceLearner,
    Preference,
    PreferenceSignal,
    PreferencePrediction,
    PreferenceSignalType,
    PreferenceStrength,
    BanditArm
)

from .value_evolution_tracker import (
    ValueEvolutionTracker,
    ValueState,
    ValueTrajectory,
    DriftAlert,
    ValueEvolutionAnalysis,
    DriftSeverity,
    TrendDirection,
    ValueChangeType
)

# Import auto_apply_policy with fallback
try:
    from .auto_apply_policy import (
        load_policy,
        check_files_against_policy,
        run_gates,
        Policy,
        PolicyError,
        GateFailure,
        GateSpec,
        FileCheckResult,
        GatesReport
    )
    _auto_apply_available = True
except ImportError:
    # Minimal fallback if auto_apply_policy or its dependencies (like yaml) are missing
    _auto_apply_available = False
    logger.warning("auto_apply_policy module or dependencies not found. Auto-apply disabled.")
    class Policy: enabled = False
    def load_policy(*args, **kwargs): return Policy()
    def check_files_against_policy(*args, **kwargs): return FileCheckResult(ok=False, reasons=["module_unavailable"], offending_files=[])
    def run_gates(*args, **kwargs): return GatesReport(ok=False, failures=["module_unavailable"], outputs={})
    class PolicyError(Exception): pass
    class GateFailure(Exception): pass
    class GateSpec: pass
    class FileCheckResult: pass
    class GatesReport: pass

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}


def get_version() -> str:
    """Get version string"""
    return __version__


def get_version_info() -> dict:
    """Get detailed version information"""
    return VERSION_INFO.copy()


# Convenience function for initialization
def create_meta_reasoning_system(world_model,
                                 design_spec: dict = None,
                                 config: dict = None) -> MotivationalIntrospection:
    """
    Convenience function to create a complete meta-reasoning system

    Args:
        world_model: Reference to WorldModel instance
        design_spec: Design specification with objectives
        config: Optional configuration overrides

    Returns:
        Initialized MotivationalIntrospection instance with all components

    Example:
        meta_reasoning = create_meta_reasoning_system(
            world_model,
            design_spec={
                'objectives': {
                    'accuracy': {'weight': 1.0, 'target': 0.95},
                    'safety': {'weight': 1.0, 'target': 1.0, 'priority': 0}
                }
            }
        )

        validation = meta_reasoning.validate_proposal_alignment(proposal)
    """

    config = config or {}

    # Create main introspection engine
    mi = MotivationalIntrospection(
        world_model=world_model,
        design_spec=design_spec
    )

    # All components are lazy-loaded, so just return the main instance
    return mi


def create_self_improvement_system(config_path: str = "configs/intrinsic_drives.json",
                                   state_path: str = "data/agent_state.json",
                                   alert_callback=None,
                                   approval_checker=None) -> SelfImprovementDrive:
    """
    Convenience function to create self-improvement drive

    Args:
        config_path: Path to configuration file
        state_path: Path to state persistence file
        alert_callback: Optional callback for sending alerts
        approval_checker: Optional callback to check approval status

    Returns:
        Initialized SelfImprovementDrive instance

    Example:
        si_drive = create_self_improvement_system(
            config_path="configs/intrinsic_drives.json",
            alert_callback=lambda severity, data: print(f"Alert: {data}")
        )

        if si_drive.should_trigger(context):
            improvement = si_drive.step(context)
    """

    return SelfImprovementDrive(
        config_path=config_path,
        state_path=state_path,
        alert_callback=alert_callback,
        approval_checker=approval_checker
    )


# Module-level exports
__all__ = [
    # Version info
    '__version__',
    'get_version',
    'get_version_info',

    # Core components
    'MotivationalIntrospection',
    'ObjectiveHierarchy',
    'CounterfactualObjectiveReasoner',
    'GoalConflictDetector',
    'ObjectiveNegotiator',
    'ValidationTracker',
    'TransparencyInterface',
    'SelfImprovementDrive',
    'InternalCritic',             # Added
    'CuriosityRewardShaper',
    'EthicalBoundaryMonitor',
    'PreferenceLearner',
    'ValueEvolutionTracker',

    # Data structures - MotivationalIntrospection
    'ObjectiveStatus',
    'ObjectiveAnalysis',
    'ProposalValidation',

    # Data structures - ObjectiveHierarchy
    'Objective',
    'ObjectiveType',
    'HierarchyConflictType',

    # Data structures - CounterfactualObjectiveReasoner
    'CounterfactualOutcome',
    'ObjectiveComparison',
    'ParetoPoint',

    # Data structures - GoalConflictDetector
    'Conflict',
    'ConflictSeverity',
    'ConflictType',
    'MultiObjectiveTension',

    # Data structures - ObjectiveNegotiator
    'AgentProposal',
    'NegotiationResult',
    'NegotiationStrategy',
    'NegotiationOutcome',
    'ConflictResolution',

    # Data structures - ValidationTracker
    'ValidationRecord',
    'ValidationPattern',
    'LearningInsight',
    'PatternType',
    'ValidationOutcome',
    'ObjectiveBlocker',

    # Data structures - TransparencyInterface
    'SerializationFormat',
    'TransparencyMetadata',

    # Data structures - SelfImprovementDrive
    'TriggerType',
    'FailureType',
    'ImprovementObjective',
    'SelfImprovementState',

    # Data structures - InternalCritic (Added)
    'Critique',
    'Evaluation',
    'Risk',
    'ComparisonResult',
    'PerspectiveScore',
    'CritiqueLevel',
    'EvaluationPerspective',
    'RiskCategory',
    'RiskSeverity',

    # Data structures - CuriosityRewardShaper
    'NoveltyEstimate',
    'EpisodicMemory',
    'CuriosityStatistics',
    'CuriosityMethod',
    'NoveltyLevel',

    # Data structures - EthicalBoundaryMonitor
    'EthicalBoundary',
    'EthicalViolation',
    'EnforcementAction',
    'BoundaryCategory',
    'ViolationSeverity',
    'EnforcementLevel',
    'BoundaryType',

    # Data structures - PreferenceLearner
    'Preference',
    'PreferenceSignal',
    'PreferencePrediction',
    'PreferenceSignalType',
    'PreferenceStrength',
    'BanditArm',

    # Data structures - ValueEvolutionTracker
    'ValueState',
    'ValueTrajectory',
    'DriftAlert',
    'ValueEvolutionAnalysis',
    'DriftSeverity',
    'TrendDirection',
    'ValueChangeType',

    # AutoApplyPolicy (conditionally available)
    'load_policy',
    'check_files_against_policy',
    'run_gates',
    'Policy',
    'PolicyError',
    'GateFailure',
    'GateSpec',
    'FileCheckResult',
    'GatesReport',

    # Convenience functions
    'create_meta_reasoning_system',
    'create_self_improvement_system',

    # Backward compatibility aliases
    'MetaReasoner',
    'Goal',
    'GoalStatus'
]

# Conditionally remove AutoApplyPolicy symbols if module not loaded
if not _auto_apply_available:
    _auto_apply_symbols = [
        'load_policy', 'check_files_against_policy', 'run_gates', 'Policy',
        'PolicyError', 'GateFailure', 'GateSpec', 'FileCheckResult', 'GatesReport'
    ]
    __all__ = [item for item in __all__ if item not in _auto_apply_symbols]

# Module-level documentation
def get_component_info() -> dict:
    """
    Get information about all components in the meta-reasoning subsystem

    Returns:
        Dictionary with component descriptions and capabilities
    """
    return {
        'MotivationalIntrospection': {
            'description': 'Core meta-reasoning engine and orchestrator',
            'capabilities': [
                'Introspect current objectives',
                'Detect objective pathologies',
                'Reason about alternatives',
                'Validate proposal alignment',
                'Explain motivation structure',
                'Track validation history',
                'Generate learning insights'
            ],
            'primary_methods': [
                'introspect_current_objective',
                'detect_objective_pathology',
                'reason_about_alternatives',
                'validate_proposal_alignment',
                'explain_motivation_structure',
                'update_validation_outcome',
                'get_learning_insights',
                'analyze_objective_achievement'
            ]
        },
        'ObjectiveHierarchy': {
            'description': 'Manages graph of objectives and relationships',
            'capabilities': [
                'Track objective dependencies',
                'Detect conflicts',
                'Maintain priority ordering',
                'Check consistency',
                'Compute conflict matrices',
                'Track unsatisfied objectives',
                'Identify violated objectives'
            ],
            'primary_methods': [
                'add_objective',
                'get_dependencies',
                'get_transitive_dependencies',
                'find_conflicts',
                'check_consistency',
                'get_priority_order',
                'get_hierarchy_structure',
                'update_objective_value',
                'get_unsatisfied_objectives',
                'get_violated_objectives',
                'compute_conflict_matrix'
            ]
        },
        'CounterfactualObjectiveReasoner': {
            'description': 'Performs counterfactual reasoning about objectives',
            'capabilities': [
                'Predict under alternative objectives',
                'Compare objectives',
                'Find Pareto frontier',
                'Estimate trade-offs',
                'Generate alternative proposals',
                'Update prediction accuracy',
                'Learn objective correlations'
            ],
            'primary_methods': [
                'predict_under_objective',
                'compare_objectives',
                'find_pareto_frontier',
                'estimate_tradeoffs',
                'generate_alternative_proposals',
                'update_prediction_accuracy',
                'learn_objective_correlation'
            ]
        },
        'GoalConflictDetector': {
            'description': 'Detects and analyzes objective conflicts',
            'capabilities': [
                'Detect conflicts in proposals',
                'Analyze multi-objective tension',
                'Suggest resolutions',
                'Validate trade-offs',
                'Identify critical conflicts',
                'Track conflict history',
                'Predict conflict likelihood'
            ],
            'primary_methods': [
                'detect_conflicts_in_proposal',
                'analyze_multi_objective_tension',
                'suggest_resolution',
                'validate_tradeoffs',
                'identify_critical_conflicts',
                'get_conflict_history',
                'predict_conflict_likelihood'
            ]
        },
        'ObjectiveNegotiator': {
            'description': 'Resolves conflicts through multi-agent negotiation',
            'capabilities': [
                'Multi-agent proposal negotiation',
                'Pareto frontier computation',
                'Conflict resolution',
                'Dynamic objective weighting',
                'Strategy selection',
                'Validation of negotiated objectives',
                'Track negotiation history'
            ],
            'primary_methods': [
                'negotiate_multi_agent_proposals',
                'find_pareto_frontier',
                'resolve_objective_conflict',
                'dynamic_objective_weighting',
                'validate_negotiated_objectives'
            ]
        },
        'ValidationTracker': {
            'description': 'Tracks validation history and learns patterns',
            'capabilities': [
                'Record validations',
                'Identify patterns (success/failure/risky)',
                'Predict outcomes',
                'Generate insights',
                'Identify blockers',
                'Suggest better proxies',
                'Analyze failure patterns',
                'Detect validation trends'
            ],
            'primary_methods': [
                'record_validation',
                'update_actual_outcome',
                'identify_risky_patterns',
                'identify_success_patterns',
                'identify_blockers',
                'analyze_failure_patterns',
                'predict_validation_outcome',
                'get_learning_insights',
                'suggest_better_proxies',
                'detect_blockers_from_history'
            ]
        },
        'TransparencyInterface': {
            'description': 'Provides machine-readable transparency',
            'capabilities': [
                'Serialize validations',
                'Export objective state',
                'Document conflicts',
                'Support consensus voting',
                'Generate cryptographic signatures',
                'Maintain audit logs',
                'Verify data integrity'
            ],
            'primary_methods': [
                'serialize_validation',
                'serialize_objective_state',
                'serialize_conflict',
                'serialize_negotiation_outcome',
                'export_for_consensus',
                'get_audit_log',
                'verify_signature'
            ]
        },
        'SelfImprovementDrive': {
            'description': 'Autonomous self-improvement as intrinsic drive',
            'capabilities': [
                'Continuous self-monitoring',
                'Trigger-based activation (startup, errors, performance, periodic, low activity)',
                'Objective selection and prioritization',
                'Improvement action generation',
                'Cost tracking and resource limits',
                'Human approval workflow',
                'Adaptive learning from outcomes',
                'Failure classification (transient vs systemic)',
                'State persistence with backups'
            ],
            'primary_methods': [
                'should_trigger',
                'select_objective',
                'generate_improvement_action',
                'request_approval',
                'approve_pending',
                'reject_pending',
                'check_approval_status',
                'step',
                'record_outcome',
                'get_status'
            ]
        },
        'InternalCritic': { # Added
            'description': 'Multi-perspective self-critique and evaluation system',
            'capabilities': [
                'Evaluate proposals from multiple perspectives',
                'Generate critiques with severity levels',
                'Identify and assess risks',
                'Compare alternative proposals',
                'Suggest improvements',
                'Learn from validation outcomes'
            ],
            'primary_methods': [
                'evaluate_proposal',
                'generate_critique',
                'suggest_improvements',
                'identify_risks',
                'compare_alternatives',
                'learn_from_outcome'
            ]
        },
        'CuriosityRewardShaper': { # Added
            'description': 'Multi-algorithm curiosity-driven exploration and reward shaping',
            'capabilities': [
                'Compute novelty scores (Count-based, ICM, RND, Episodic, Hybrid)',
                'Shape rewards with curiosity bonus',
                'Maintain episodic memory',
                'Adapt bonus scaling',
                'Recommend exploration actions'
            ],
            'primary_methods': [
                'compute_curiosity_bonus',
                'shape_reward',
                'update_novelty_estimates',
                'get_novelty',
                'get_exploration_recommendation'
            ]
        },
        'EthicalBoundaryMonitor': { # Added
            'description': 'Multi-layered ethical boundary monitoring and enforcement',
            'capabilities': [
                'Define hard/soft/learned ethical boundaries',
                'Check actions against boundaries',
                'Enforce violations (Monitor, Warn, Modify, Block, Shutdown)',
                'Maintain violation history and audit trail',
                'Trigger emergency shutdown'
            ],
            'primary_methods': [
                'add_boundary',
                'remove_boundary',
                'check_action',
                'detect_boundary_violations',
                'get_violations',
                'trigger_shutdown'
            ]
        },
        'PreferenceLearner': { # Added
            'description': 'Bayesian preference learning with multi-armed bandits',
            'capabilities': [
                'Learn preferences from explicit/implicit signals',
                'Use Thompson Sampling for exploration',
                'Model contextual preferences',
                'Predict preferred options with confidence',
                'Detect preference drift'
            ],
            'primary_methods': [
                'learn_from_interaction',
                'predict_preference',
                'update_model',
                'get_preferences',
                'detect_preference_drift_internal', # Internal method exposed for consistency
                'get_exploration_recommendation'
            ]
        },
        'ValueEvolutionTracker': { # Added
            'description': 'Time-series tracking and analysis of agent value evolution',
            'capabilities': [
                'Track value trajectories',
                'Detect drift (CUSUM, change-point)',
                'Analyze trends and stability',
                'Identify correlations between values',
                'Generate drift alerts'
            ],
            'primary_methods': [
                'record_value_state',
                'detect_drift',
                'analyze_evolution',
                'get_value_trajectory',
                'predict_future_value',
                'set_baseline',
                'get_alerts'
            ]
        },
        'AutoApplyPolicy': { # Added (conditionally)
            'description': 'Policy engine for validating and applying code changes',
            'capabilities': [
                'Load YAML policies',
                'Validate files against allow/deny globs',
                'Enforce file/LOC budgets',
                'Run pre-apply gates safely',
                'Evaluate NSO requirements'
            ],
            'primary_methods': [
                'load_policy',
                'check_files_against_policy',
                'run_gates',
                'evaluate_nso_requirements'
            ]
        }
    }


def print_component_info():
    """Print information about all components"""
    info = get_component_info()

    print("=" * 80)
    print("VULCAN-AMI Meta-Reasoning Subsystem")
    print(f"Version: {__version__}")
    print("=" * 80)
    print()

    for component_name, component_info in info.items():
        print(f"{component_name}")
        print("-" * len(component_name))
        print(f"Description: {component_info['description']}")
        print(f"\nCapabilities:")
        for cap in component_info['capabilities']:
            print(f"  • {cap}")
        print(f"\nPrimary Methods:")
        for method in component_info['primary_methods']:
            print(f"  • {method}()")
        print()


# Integration check
def check_integration() -> dict:
    """
    Check if all components are properly integrated

    Returns:
        Dictionary with integration status
    """
    status = {
        'all_components_importable': True,
        'components': {},
        'errors': []
    }

    components = [
        'MotivationalIntrospection',
        'ObjectiveHierarchy',
        'CounterfactualObjectiveReasoner',
        'GoalConflictDetector',
        'ObjectiveNegotiator',
        'ValidationTracker',
        'TransparencyInterface',
        'SelfImprovementDrive',
        'InternalCritic',             # Added
        'CuriosityRewardShaper',
        'EthicalBoundaryMonitor',
        'PreferenceLearner',
        'ValueEvolutionTracker',
        # AutoApplyPolicy checked implicitly by its inclusion in __all__
    ]

    for component_name in components:
        try:
            component_class = globals()[component_name]
            status['components'][component_name] = {
                'importable': True,
                'class': component_class.__name__,
                'module': component_class.__module__
            }
        except Exception as e:
            status['all_components_importable'] = False
            status['components'][component_name] = {
                'importable': False,
                'error': str(e)
            }
            status['errors'].append(f"{component_name}: {e}")

    # Check auto_apply_policy availability separately
    status['components']['AutoApplyPolicy'] = {
        'importable': _auto_apply_available,
        'status': 'Available' if _auto_apply_available else 'Fallback/Unavailable'
    }
    if not _auto_apply_available:
        status['all_components_importable'] = False # If it's considered essential

    return status


# Lazy import helper
class LazyLoader:
    """Lazy loader for optional dependencies"""

    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)


# Module initialization
def _initialize_module():
    """Initialize module on import"""
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("Meta-reasoning subsystem loaded (version %s)", __version__)


# Run initialization
_initialize_module()


# --- Back-compat: expose MetaReasoner for legacy imports ----------------------
try:
    MetaReasoner  # type: ignore[name-defined]
except NameError:
    from .motivational_introspection import MotivationalIntrospection as MetaReasoner

try:
    __all__
except NameError:
    __all__ = []
if "MetaReasoner" not in __all__:
    __all__.append("MetaReasoner")


# --- Back-compat: expose Goal for legacy imports ------------------------------
# Some consumers import `Goal` from this package; the canonical type here is `Objective`.
try:
    Goal  # type: ignore[name-defined]
except NameError:
    Goal = Objective  # alias for compatibility
if "Goal" not in __all__:
    __all__.append("Goal")


# --- Back-compat: expose GoalStatus for legacy imports ------------------------
# Some consumers import `GoalStatus`; the canonical name here is `ObjectiveStatus`.
try:
    GoalStatus  # type: ignore[name-defined]
except NameError:
    GoalStatus = ObjectiveStatus  # alias for compatibility
if "GoalStatus" not in __all__:
    __all__.append("GoalStatus")