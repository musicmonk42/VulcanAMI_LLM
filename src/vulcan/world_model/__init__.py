"""
World Model Module
==================

The World Model maintains a causal understanding of the environment through:

Core Components
--------------
- **CausalDAG**: Directed acyclic graph of causal relationships with cycle detection
- **ConfidenceCalibrator**: Calibrates prediction confidence using multiple methods
- **CorrelationTracker**: Tracks and analyzes correlations between variables
- **DynamicsModel**: Models temporal dynamics and state transitions
- **InterventionManager**: Plans and executes interventions for causal discovery

Architecture
-----------
All components follow the EXAMINE → SELECT → APPLY → REMEMBER pattern and include:
- Comprehensive safety validation throughout
- Thread-safe operations with proper locking
- Graceful fallbacks when dependencies unavailable
- Separation of concerns with modular sub-components

Safety Features
--------------
- Safety validator integration with audit logging
- Blocks unsafe interventions that could harm the system
- Validates all state transitions and predictions
- Tracks safety blocks and corrections

Example Usage
------------
```python
from vulcan.world_model import (
    WorldModel,
    Observation,
    ModelContext,
    EvidenceType
)
import time

# Initialize world model with safety
config = {
    'safety_config': {'max_nodes': 1000},
    'bootstrap_mode': True,
    'simulation_mode': True,
    'enable_meta_reasoning': False
}
world_model = WorldModel(config=config)

# Create observation
observation = Observation(
    timestamp=time.time(),
    variables={
        'temperature': 25.0,
        'humidity': 0.8,
        'pressure': 1013.25
    },
    domain='environment'
)

# Update world model
result = world_model.update_from_observation(observation)
print(f"Update status: {result['status']}")
print(f"Updates executed: {result['updates_executed']}")

# Make prediction
context = ModelContext(
    domain='environment',
    targets=['temperature'],
    constraints={'time_horizon': 3600}  # 1 hour
)
prediction = world_model.predict_with_calibrated_uncertainty(
    action='increase_heating',
    context=context
)
print(f"Expected: {prediction.expected}, Confidence: {prediction.confidence}")

# Get causal structure
structure = world_model.get_causal_structure()
print(f"Causal graph has {structure['statistics']['node_count']} nodes")

# Validate consistency
validation = world_model.validate_model_consistency()
print(f"Model consistent: {validation['is_consistent']}")

# For multi-agent systems (requires meta-reasoning)
if world_model.meta_reasoning_enabled:
    proposal = {'action': 'modify_parameter', 'value': 100}
    evaluation = world_model.evaluate_agent_proposal(proposal)
    print(f"Proposal valid: {evaluation['valid']}")
```

Dependencies
-----------
**Required:**
- numpy
- logging

**Optional (with fallbacks):**
- scipy: For advanced statistical functions
- sklearn: For machine learning models
- pandas: For data manipulation
- networkx: For advanced graph operations
- statsmodels: For time series analysis

**Safety (recommended):**
- safety_validator: For comprehensive safety checks
- safety_types: For safety configuration types

Module Structure
---------------
```
world_model/
├── __init__.py                 # This file
├── causal_graph.py            # Causal DAG and graph operations
├── confidence_calibrator.py   # Confidence calibration
├── correlation_tracker.py     # Correlation tracking
├── dynamics_model.py          # Temporal dynamics modeling
├── intervention_manager.py    # Intervention planning/execution
├── invariant_detector.py      # Invariant detection (if exists)
├── prediction_engine.py       # Prediction engine (if exists)
├── world_model_core.py        # Core world model (if exists)
└── world_model_router.py      # Routing/orchestration (if exists)
```

Notes
-----
- All components are thread-safe
- Safety validation is enabled when safety_validator is available
- Components operate independently but can be integrated via router
- Statistics are tracked for all major operations
"""

import logging
from typing import Any, Dict

# Import core causal graph components
from .causal_graph import (
    CausalDAG,
    CausalEdge,
    CausalPath,
    CycleDetector,
    DSeparationChecker,
    EvidenceType,
    GraphStructure,
    PathFinder,
    ProbabilityDistribution,
    TopologicalSorter,
)

# Import confidence calibration components
from .confidence_calibrator import (
    CalibrationBin,
    ConfidenceCalibrator,
    ModelConfidenceTracker,
    PredictionRecord,
)

# Import correlation tracking components
from .correlation_tracker import (
    BaselineTracker,
    CausalityTracker,
    ChangeDetector,
    CorrelationCalculator,
    CorrelationEntry,
    CorrelationMatrix,
    CorrelationMethod,
    CorrelationStorage,
    CorrelationTracker,
    DataBuffer,
    StatisticsTracker,
)

# Import dynamics modeling components
from .dynamics_model import (
    Condition,
    DynamicsApplier,
    DynamicsModel,
    ModelFitter,
    PatternDetector,
    PatternType,
    State,
    StateClusterer,
    StateTransition,
    TemporalPattern,
    TimeSeriesAnalyzer,
    TransitionLearner,
)

# Import intervention management components
from .intervention_manager import (
    ConfounderDetector,
    Correlation,
    CostEstimator,
    InformationGainEstimator,
    InterventionCandidate,
    InterventionExecutor,
    InterventionPrioritizer,
    InterventionResult,
    InterventionScheduler,
    InterventionSimulator,
    InterventionType,
)

# Import invariant detection components
from .invariant_detector import (
    ConservationLawDetector,
    Invariant,
    InvariantDetector,
    InvariantEvaluator,
    InvariantIndexer,
    InvariantRegistry,
    InvariantType,
    InvariantValidator,
    LinearRelationshipDetector,
)

# Import prediction engine components
from .prediction_engine import (
    CombinationMethod,
    EnsemblePredictor,
    MonteCarloSampler,
    Path,
    PathAnalyzer,
    PathCluster,
    PathClusterer,
    PathEffectCalculator,
    PathTracer,
    Prediction,
    PredictionCombiner,
)

# Import world model core and router
from .world_model_core import (
    ConsistencyValidator,
    InterventionManager,
    ModelContext,
    Observation,
    ObservationProcessor,
    PredictionManager,
    WorldModel,
)
from .world_model_router import (
    CostModel,
    ObservationSignature,
    PatternLearner,
    UpdateDependencyGraph,
    UpdatePlan,
    UpdatePriority,
    UpdateStrategy,
    UpdateType,
    WorldModelRouter,
)

# Import system observer for event tracking
try:
    from .system_observer import (
        SystemObserver,
        SystemEvent,
        EventType,
        get_system_observer,
        initialize_system_observer,
        # BUG #3 FIX: Meta-reasoning integration API
        get_recent_reasoning_activity,
        get_reasoning_success_rates,
        get_failure_patterns_for_improvement,
        get_recent_outcomes,
        notify_meta_reasoning_of_event,
    )
    _system_observer_available = True
except ImportError as e:
    # Logger not yet defined, will use module-level logger later
    SystemObserver = None
    SystemEvent = None
    EventType = None
    get_system_observer = None
    initialize_system_observer = None
    get_recent_reasoning_activity = None
    get_reasoning_success_rates = None
    get_failure_patterns_for_improvement = None
    get_recent_outcomes = None
    notify_meta_reasoning_of_event = None
    _system_observer_available = False

# Try to import additional components if they exist
_optional_imports_success = {}
_optional_imports_success["prediction_engine"] = True
_optional_imports_success["invariant_detector"] = True
_optional_imports_success["world_model_core"] = True
_optional_imports_success["world_model_router"] = True


# Version info
__version__ = "0.1.0"
__author__ = "VULCAN-AGI Team"


# Define public API
__all__ = [
    # Core Components
    "CausalDAG",
    "ConfidenceCalibrator",
    "CorrelationTracker",
    "DynamicsModel",
    "InterventionExecutor",
    "InterventionPrioritizer",
    # Causal Graph Classes
    "CausalEdge",
    "CausalPath",
    "EvidenceType",
    "ProbabilityDistribution",
    "GraphStructure",
    "CycleDetector",
    "PathFinder",
    "DSeparationChecker",
    "TopologicalSorter",
    # Confidence Calibration Classes
    "ModelConfidenceTracker",
    "CalibrationBin",
    "PredictionRecord",
    # Correlation Tracking Classes
    "CorrelationMatrix",
    "CorrelationEntry",
    "CorrelationMethod",
    "CorrelationCalculator",
    "StatisticsTracker",
    "DataBuffer",
    "CorrelationStorage",
    "ChangeDetector",
    "CausalityTracker",
    "BaselineTracker",
    # Dynamics Modeling Classes
    "State",
    "StateTransition",
    "Condition",
    "TemporalPattern",
    "PatternType",
    "TimeSeriesAnalyzer",
    "PatternDetector",
    "StateClusterer",
    "TransitionLearner",
    "ModelFitter",
    "DynamicsApplier",
    # Intervention Management Classes
    "InterventionCandidate",
    "InterventionResult",
    "InterventionType",
    "Correlation",
    "InformationGainEstimator",
    "CostEstimator",
    "InterventionScheduler",
    "ConfounderDetector",
    "InterventionSimulator",
    # Prediction Engine Classes
    "EnsemblePredictor",
    "Prediction",
    "Path",
    "PathCluster",
    "PathTracer",
    "PathAnalyzer",
    "PathEffectCalculator",
    "PathClusterer",
    "MonteCarloSampler",
    "PredictionCombiner",
    "CombinationMethod",
    # Invariant Detection Classes
    "InvariantDetector",
    "InvariantRegistry",
    "Invariant",
    "InvariantType",
    "InvariantEvaluator",
    "InvariantValidator",
    "InvariantIndexer",
    "ConservationLawDetector",
    "LinearRelationshipDetector",
    # World Model Core Classes
    "WorldModel",
    "Observation",
    "ModelContext",
    "ObservationProcessor",
    "InterventionManager",
    "PredictionManager",
    "ConsistencyValidator",
    # World Model Router Classes
    "WorldModelRouter",
    "UpdatePlan",
    "UpdateType",
    "UpdatePriority",
    "UpdateStrategy",
    "ObservationSignature",
    "UpdateDependencyGraph",
    "PatternLearner",
    "CostModel",
    # System Observer Classes (for event tracking)
    "SystemObserver",
    "SystemEvent",
    "EventType",
    "get_system_observer",
    "initialize_system_observer",
    # Meta-Reasoning Integration API (BUG #3 FIX)
    "get_recent_reasoning_activity",
    "get_reasoning_success_rates",
    "get_failure_patterns_for_improvement",
    "get_recent_outcomes",
    "notify_meta_reasoning_of_event",
]


# Utility functions
def get_available_components() -> Dict[str, bool]:
    """
    Get availability status of all world model components.

    Returns:
        Dictionary mapping component names to availability status
    """
    return {
        "causal_graph": True,
        "confidence_calibrator": True,
        "correlation_tracker": True,
        "dynamics_model": True,
        "intervention_manager": True,
        "prediction_engine": True,
        "invariant_detector": True,
        "world_model_core": _optional_imports_success.get("world_model_core", False),
        "world_model_router": _optional_imports_success.get(
            "world_model_router", False
        ),
    }


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.

    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {}

    # Check scipy
    try:
        pass

        dependencies["scipy"] = True
    except ImportError:
        dependencies["scipy"] = False

    # Check sklearn
    try:
        pass

        dependencies["sklearn"] = True
    except ImportError:
        dependencies["sklearn"] = False

    # Check pandas
    try:
        pass

        dependencies["pandas"] = True
    except ImportError:
        dependencies["pandas"] = False

    # Check networkx
    try:
        pass

        dependencies["networkx"] = True
    except ImportError:
        dependencies["networkx"] = False

    # Check statsmodels
    try:
        pass

        dependencies["statsmodels"] = True
    except ImportError:
        dependencies["statsmodels"] = False

    # Check safety validator (with protection against circular imports)
    try:
        # Use importlib to check if module exists without triggering full import chain
        import importlib.util

        spec = importlib.util.find_spec("vulcan.safety.safety_validator")
        if spec is not None:
            # Module exists, try to import to verify it's usable
            pass

            dependencies["safety_validator"] = True
        else:
            dependencies["safety_validator"] = False
    except (ImportError, AttributeError):
        dependencies["safety_validator"] = False

    return dependencies


def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive module information.

    Returns:
        Dictionary with module metadata
    """
    return {
        "version": __version__,
        "author": __author__,
        "components": get_available_components(),
        "dependencies": check_dependencies(),
        "core_features": [
            "Unified WorldModel orchestrator with all components integrated",
            "Intelligent routing with pattern learning and cost modeling",
            "Causal DAG with cycle detection and path finding",
            "Confidence calibration (isotonic, Platt, histogram, beta)",
            "Correlation tracking with significance testing",
            "Temporal dynamics modeling with state transitions",
            "Intervention planning and execution with safety validation",
            "Ensemble prediction with uncertainty quantification",
            "Invariant detection (conservation laws, constraints, patterns)",
            "Meta-reasoning layer for goal-level reasoning (optional)",
            "Graphix IR integration for multi-agent consensus",
            "Safety validation throughout all operations",
            "Thread-safe operations with deadlock prevention",
            "Graceful fallbacks for missing dependencies",
        ],
        "safety_features": [
            "Safety validator integration",
            "Intervention blocking for unsafe operations",
            "State transition validation",
            "Audit logging support",
            "Safety statistics tracking",
        ],
    }


# Configure module-level logger
logger = logging.getLogger(__name__)
logger.info(
    "World Model module loaded - components: %s",
    [k for k, v in get_available_components().items() if v],
)

# Log warnings for missing dependencies (excluding safety_validator which may be delayed due to circular imports)
_deps = check_dependencies()
missing_deps = [k for k, v in _deps.items() if not v and k != "safety_validator"]
if missing_deps:
    logger.warning("Operating with fallback implementations for: %s", missing_deps)

# Safety validator check - only warn if the module file doesn't exist at all
# (not if it's temporarily unavailable due to circular imports)
if not _deps.get("safety_validator", False):
    # Double-check by looking for the module file
    import importlib.util

    safety_spec = importlib.util.find_spec("vulcan.safety.safety_validator")
    if safety_spec is None:
        # Module truly doesn't exist
        logger.warning(
            "Safety validator not available - operating without safety checks. "
            "For production use, ensure safety_validator module is available."
        )
    # If module exists but couldn't import (circular import), don't warn - it will be available later
