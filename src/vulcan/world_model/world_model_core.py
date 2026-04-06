# src/vulcan/world_model/meta_reasoning/world_model_core.py - Main orchestrator for the World Model component
"""
world_model_core.py - Main orchestrator for the World Model component
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
Integrated with comprehensive safety validation and meta-reasoning
FULLY INTEGRATED: All real components, removed all mocks, comprehensive error handling
INTEGRATED: Autonomous self-improvement drive as core functionality
FIXED: Circular import with dynamics_model resolved via lazy loading
FIXED: Deque slicing and CorrelationEntry attribute issues
FIXED: Initialization order for Meta-reasoning components to prevent MagicMock fallback

**EXECUTION ENGINE REPLACEMENT (2025-11-19):**
- Replaced mock handlers (_perform_improvement, _fix_circular_imports, etc.) with a single,
  integrated LLM-driven execution pipeline in _execute_improvement.
- Pipeline simulates calls to llm_integration, ast_tools, diff_tools, and git_tools
  to generate code changes, validate them, apply them to the file system, and commit the result.
"""

import ast
import difflib
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import (
    Path as FilePath,
)  # <-- FIX: Use alias 'FilePath' to avoid name conflict
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

# 🚨 BEGIN PRODUCTION LLM IMPORTS 🚨
try:
    import openai
except ImportError:
    # Placeholder if the user hasn't installed the library yet
    openai = MagicMock()
# 🚨 END PRODUCTION LLM IMPORTS 🚨

logger = logging.getLogger(__name__)

# Schema Registry for validation
SchemaRegistry = None
_SCHEMA_REGISTRY_AVAILABLE = False
try:
    from ..schema_registry import SchemaRegistry
    _SCHEMA_REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.debug(f"SchemaRegistry not available: {e}")
    SchemaRegistry = None

# Shared constants for query pattern detection (from system_observer)
# These are used by both WorldModel and SystemObserver for consistent classification
try:
    from .system_observer import (
        FORMAL_LOGIC_SYMBOLS,
        FORMAL_LOGIC_KEYWORDS,
        PROBABILITY_KEYWORDS,
        SELF_REFERENTIAL_KEYWORDS,
        initialize_system_observer,
        get_system_observer,
    )
    SHARED_CONSTANTS_AVAILABLE = True
    SYSTEM_OBSERVER_IMPORTABLE = True
except ImportError:
    # Fallback definitions if system_observer not available
    FORMAL_LOGIC_SYMBOLS = frozenset(['→', '∧', '∨', '¬', '∀', '∃', '⊢', '⊨', '↔', '⇒', '⇔'])
    FORMAL_LOGIC_KEYWORDS = frozenset(['forall', 'exists', 'implies', 'entails', 'satisfiable', 'valid'])
    PROBABILITY_KEYWORDS = frozenset(['probability', 'likelihood', 'bayes', 'bayesian', 'posterior', 'prior', 'p(', 'conditional', 'expectation', 'marginal'])
    SELF_REFERENTIAL_KEYWORDS = frozenset(['you want', 'your goal', 'self-aware', 'you have', 'do you', 'are you', 'your capabilities', 'yourself', 'your purpose', 'your objectives'])
    SHARED_CONSTANTS_AVAILABLE = False
    SYSTEM_OBSERVER_IMPORTABLE = False
    initialize_system_observer = None  # type: ignore
    get_system_observer = None  # type: ignore

# Minimum observations needed for routing recommendations
MIN_OBSERVATIONS_FOR_RECOMMENDATIONS = 5

# Lazy import placeholders
EnhancedSafetyValidator = None
SafetyConfig = None
CorrelationTracker = None
DynamicsModel = None
CausalDAG = None
CausalEdge = None
InterventionPrioritizer = None
InterventionExecutor = None
InterventionResult = None
InterventionCandidate = None
Correlation = None
EnsemblePredictor = None
Prediction = None
PathTracer = None
Path = None  # <-- This name will be populated by the lazy loader
InvariantRegistry = None
InvariantDetector = None
Invariant = None
InvariantType = None
ConfidenceCalibrator = None
ModelConfidenceTracker = None
WorldModelRouter = None
UpdatePlan = None
MotivationalIntrospection = None
ObjectiveHierarchy = None
CounterfactualObjectiveReasoner = None
GoalConflictDetector = None
ObjectiveNegotiator = None
ValidationTracker = None
TransparencyInterface = None
SelfImprovementDrive = None
TriggerType = None
ImprovementObjective = None
# Note: Add missing meta-reasoning components for full integration
InternalCritic = None
CuriosityRewardShaper = None
EthicalBoundaryMonitor = None
PreferenceLearner = None
ValueEvolutionTracker = None


def _lazy_import_safety_validator():
    global EnhancedSafetyValidator, SafetyConfig
    if EnhancedSafetyValidator is None:
        try:
            from ..safety.safety_types import SafetyConfig
            from ..safety.safety_validator import EnhancedSafetyValidator

            logger.info("Safety validator lazy loaded successfully")
        except ImportError as e:
            logger.critical(
                "safety_validator not available: %s - System running without safety constraints!",
                e,
            )
            return False
    return True


def _lazy_import_correlation_tracker():
    global CorrelationTracker
    if CorrelationTracker is None:
        try:
            from .correlation_tracker import CorrelationTracker

            logger.info("Correlation tracker lazy loaded successfully")
        except ImportError as e:
            logger.critical(
                "correlation_tracker module not available: %s - Cannot track correlations!",
                e,
            )
            return False
    return True


def _lazy_import_dynamics_model():
    global DynamicsModel
    if DynamicsModel is None:
        try:
            from .dynamics_model import DynamicsModel as DM

            DynamicsModel = DM
            logger.info("DynamicsModel lazy loaded successfully")
        except ImportError as e:
            logger.warning("dynamics_model module not available: %s", e)
            return False
    return True


def _lazy_import_causal_graph():
    global CausalDAG, CausalEdge
    if CausalDAG is None:
        try:
            from .causal_graph import CausalDAG, CausalEdge

            logger.info("CausalDAG imported successfully")
        except ImportError as e:
            logger.critical(
                "causal_graph module not available: %s - Cannot build causal models!", e
            )
            return False
    return True


def _lazy_import_intervention_manager():
    global Correlation, InterventionPrioritizer, InterventionExecutor, InterventionResult, InterventionCandidate
    if Correlation is None:
        try:
            from .intervention_manager import (
                Correlation,
                InterventionCandidate,
                InterventionExecutor,
                InterventionPrioritizer,
                InterventionResult,
            )

            logger.info("Intervention manager components lazy loaded successfully")
            return True
        except ImportError as e:
            logger.critical(
                "intervention_manager module not available: %s - Cannot run interventions!",
                e,
            )
            return False
    return True


def _lazy_import_prediction_engine():
    global Path, EnsemblePredictor, Prediction
    if Path is None:
        try:
            from .prediction_engine import EnsemblePredictor, Path, Prediction

            logger.info("Prediction engine components lazy loaded successfully")
            return True
        except ImportError as e:
            logger.critical(
                "prediction_engine module not available: %s - Cannot make predictions!",
                e,
            )
            return False
    return True


def _lazy_import_invariant_detector():
    global InvariantRegistry, InvariantDetector, Invariant, InvariantType
    if InvariantRegistry is None:
        try:
            from .invariant_detector import (
                Invariant,
                InvariantDetector,
                InvariantRegistry,
                InvariantType,
            )

            logger.info("InvariantDetector components imported successfully")
        except ImportError as e:
            logger.warning("invariant_detector module not available: %s", e)
            return False
    return True


def _lazy_import_confidence_calibrator():
    global ConfidenceCalibrator, ModelConfidenceTracker
    if ConfidenceCalibrator is None:
        try:
            from .confidence_calibrator import (
                ConfidenceCalibrator,
                ModelConfidenceTracker,
            )

            logger.info("ConfidenceCalibrator components imported successfully")
        except ImportError as e:
            logger.warning("confidence_calibrator module not available: %s", e)
            return False
    return True


def _lazy_import_world_model_router():
    global WorldModelRouter, UpdatePlan
    if WorldModelRouter is None:
        try:
            from .world_model_router import UpdatePlan, WorldModelRouter

            logger.info("WorldModelRouter imported successfully")
        except ImportError as e:
            logger.warning("world_model_router module not available: %s", e)
            return False
    return True


# =============================================================================
# Extracted classes -- canonical definitions live in separate modules
# =============================================================================
from .observation_types import (
    ComponentIntegrationError,
    ModelContext,
    NullMetaReasoningComponent,
    NullMotivationalIntrospection,
    NullObjectiveHierarchy,
    Observation,
)
from .observation_processor import ObservationProcessor
from .intervention_orchestrator import InterventionManager
from .prediction_orchestrator import PredictionManager
from .consistency_validator import ConsistencyValidator
from .self_improvement import CodeLLMClient


def _lazy_import_meta_reasoning():
    global MotivationalIntrospection, ObjectiveHierarchy, CounterfactualObjectiveReasoner, GoalConflictDetector, ObjectiveNegotiator, ValidationTracker, TransparencyInterface, SelfImprovementDrive, TriggerType, ImprovementObjective, InternalCritic, CuriosityRewardShaper, EthicalBoundaryMonitor, PreferenceLearner, ValueEvolutionTracker
    if MotivationalIntrospection is None:
        try:
            from .meta_reasoning import (
                CounterfactualObjectiveReasoner,
                GoalConflictDetector,
                ImprovementObjective,
                MotivationalIntrospection,
                ObjectiveHierarchy,
                ObjectiveNegotiator,
                SelfImprovementDrive,
                TransparencyInterface,
                TriggerType,
                ValidationTracker,
                # Note: Import additional meta-reasoning components
                InternalCritic,
                CuriosityRewardShaper,
                EthicalBoundaryMonitor,
                PreferenceLearner,
                ValueEvolutionTracker,
            )

            logger.info("Meta-reasoning components lazy loaded successfully (full integration)")
        except ImportError as e:
            logger.warning(f"Meta-reasoning component unavailable: {e}")
            logger.warning("Falling back to null object implementations with explicit warnings")
            MotivationalIntrospection = NullMotivationalIntrospection
            ObjectiveHierarchy = NullObjectiveHierarchy
            CounterfactualObjectiveReasoner = lambda *a, **k: NullMetaReasoningComponent("CounterfactualObjectiveReasoner")
            GoalConflictDetector = lambda *a, **k: NullMetaReasoningComponent("GoalConflictDetector")
            ObjectiveNegotiator = lambda *a, **k: NullMetaReasoningComponent("ObjectiveNegotiator")
            ValidationTracker = lambda *a, **k: NullMetaReasoningComponent("ValidationTracker")
            TransparencyInterface = lambda *a, **k: NullMetaReasoningComponent("TransparencyInterface")
            SelfImprovementDrive = lambda *a, **k: NullMetaReasoningComponent("SelfImprovementDrive")
            TriggerType = NullMetaReasoningComponent("TriggerType")
            ImprovementObjective = NullMetaReasoningComponent("ImprovementObjective")
            # Note: Null object implementations for additional components
            InternalCritic = lambda *a, **k: NullMetaReasoningComponent("InternalCritic")
            CuriosityRewardShaper = lambda *a, **k: NullMetaReasoningComponent("CuriosityRewardShaper")
            EthicalBoundaryMonitor = lambda *a, **k: NullMetaReasoningComponent("EthicalBoundaryMonitor")
            PreferenceLearner = lambda *a, **k: NullMetaReasoningComponent("PreferenceLearner")
            ValueEvolutionTracker = lambda *a, **k: NullMetaReasoningComponent("ValueEvolutionTracker")
            return False
    return True


def _lazy_import_world_model():
    """
    Lazy import function for WorldModel class.
    Called by correlation_tracker.py to avoid circular imports.
    Since WorldModel is defined in this module, this function just returns True.
    """
    # WorldModel is defined in this module, so it's always available
    return True


# Component availability flags
CAUSAL_GRAPH_AVAILABLE = False
INTERVENTION_MANAGER_AVAILABLE = False
PREDICTION_ENGINE_AVAILABLE = False
INVARIANT_DETECTOR_AVAILABLE = False
CONFIDENCE_CALIBRATOR_AVAILABLE = False
ROUTER_AVAILABLE = False
META_REASONING_AVAILABLE = False


def check_component_availability():
    global CAUSAL_GRAPH_AVAILABLE, INTERVENTION_MANAGER_AVAILABLE, PREDICTION_ENGINE_AVAILABLE, INVARIANT_DETECTOR_AVAILABLE, CONFIDENCE_CALIBRATOR_AVAILABLE, ROUTER_AVAILABLE, META_REASONING_AVAILABLE
    CAUSAL_GRAPH_AVAILABLE = _lazy_import_causal_graph()
    INTERVENTION_MANAGER_AVAILABLE = _lazy_import_intervention_manager()
    PREDICTION_ENGINE_AVAILABLE = _lazy_import_prediction_engine()
    INVARIANT_DETECTOR_AVAILABLE = _lazy_import_invariant_detector()
    CONFIDENCE_CALIBRATOR_AVAILABLE = _lazy_import_confidence_calibrator()
    ROUTER_AVAILABLE = _lazy_import_world_model_router()
    _lazy_import_safety_validator()
    _lazy_import_correlation_tracker()
    _lazy_import_dynamics_model()
    META_REASONING_AVAILABLE = _lazy_import_meta_reasoning()
    return {
        "causal_graph": CAUSAL_GRAPH_AVAILABLE,
        "correlation_tracker": CorrelationTracker is not None,
        "intervention_manager": INTERVENTION_MANAGER_AVAILABLE,
        "prediction_engine": PREDICTION_ENGINE_AVAILABLE,
        "dynamics_model": DynamicsModel is not None,
        "invariant_detector": INVARIANT_DETECTOR_AVAILABLE,
        "confidence_calibrator": CONFIDENCE_CALIBRATOR_AVAILABLE,
        "router": ROUTER_AVAILABLE,
        "meta_reasoning": META_REASONING_AVAILABLE,
        "self_improvement": SelfImprovementDrive is not None,
        "safety_validator": EnhancedSafetyValidator is not None,
    }


class WorldModel:
    """Main world model orchestrator - FULLY INTEGRATED with autonomous self-improvement"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        check_component_availability()

        config = config or {}

        # Check critical component availability
        missing_critical = []
        if not CAUSAL_GRAPH_AVAILABLE:
            missing_critical.append("causal_graph")
        if CorrelationTracker is None:
            missing_critical.append("correlation_tracker")
        if not PREDICTION_ENGINE_AVAILABLE:
            missing_critical.append("prediction_engine")

        if missing_critical:
            error_msg = (
                f"Critical components unavailable: {', '.join(missing_critical)}"
            )
            logger.critical(error_msg)
            raise ComponentIntegrationError(error_msg)

        # Initialize safety validator first using singleton
        if EnhancedSafetyValidator and SafetyConfig:
            try:
                # Import singleton initializer
                from ..safety.safety_validator import initialize_all_safety_components

                safety_config = config.get("safety_config", {})
                self.safety_validator = initialize_all_safety_components(
                    config=safety_config, reuse_existing=True, return_bundle=True
                )
                self.safety_mode = "enabled"
                logger.info("Safety validator initialized (singleton)")
            except Exception as e:
                logger.error(f"Failed to initialize safety validator: {e}")
                self.safety_validator = None
                self.safety_mode = "disabled"
        else:
            self.safety_validator = None
            self.safety_mode = "disabled"
            logger.critical(
                "SAFETY DISABLED: safety_validator not available. "
                "System running without safety constraints. "
                "Install safety_validator for production use."
            )

        # Extract safety_config for passing to components
        safety_config = config.get("safety_config", {})

        # Core components - All real implementations - FIXED: Pass validator instance
        logger.info("Initializing core components...")

        self.causal_graph = CausalDAG(
            safety_config=safety_config, safety_validator=self.safety_validator
        )
        logger.info("✓ CausalDAG initialized")

        if CorrelationTracker:
            self.correlation_tracker = CorrelationTracker(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            logger.info("✓ CorrelationTracker initialized")
        else:
            self.correlation_tracker = (
                None  # This will be caught by the critical check above
            )

        if INTERVENTION_MANAGER_AVAILABLE:
            self.intervention_prioritizer = InterventionPrioritizer(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            self.intervention_executor = InterventionExecutor(
                confidence_level=config.get("intervention_confidence", 0.95),
                max_retries=config.get("max_retries", 3),
                simulation_mode=config.get("simulation_mode", True),
                safety_config=safety_config,
                safety_validator=self.safety_validator,
            )
            logger.info("✓ InterventionManager components initialized")
        else:
            logger.warning(
                "⚠ InterventionManager components unavailable - intervention testing disabled"
            )
            self.intervention_prioritizer = None
            self.intervention_executor = None

        self.ensemble_predictor = EnsemblePredictor(
            safety_config=safety_config, safety_validator=self.safety_validator
        )
        logger.info("✓ EnsemblePredictor initialized")

        if DynamicsModel:
            self.dynamics = DynamicsModel(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            logger.info("✓ DynamicsModel initialized")
        else:
            logger.warning("⚠ DynamicsModel unavailable - temporal dynamics disabled")
            self.dynamics = None

        if INVARIANT_DETECTOR_AVAILABLE:
            self.invariants = InvariantRegistry(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            self.invariant_detector = InvariantDetector(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            logger.info("✓ InvariantDetector components initialized")
        else:
            logger.warning(
                "⚠ InvariantDetector unavailable - invariant detection disabled"
            )
            self.invariants = None
            self.invariant_detector = None

        if CONFIDENCE_CALIBRATOR_AVAILABLE:
            self.confidence_calibrator = ConfidenceCalibrator(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            self.confidence_tracker = ModelConfidenceTracker(
                safety_config=safety_config, safety_validator=self.safety_validator
            )
            logger.info("✓ ConfidenceCalibrator components initialized")
        else:
            logger.warning(
                "⚠ ConfidenceCalibrator unavailable - confidence calibration disabled"
            )
            self.confidence_calibrator = None
            self.confidence_tracker = None

        # Managers (pass safety validator to observation processor)
        self.observation_processor = ObservationProcessor(self.safety_validator)
        self.intervention_manager = InterventionManager(self)
        self.prediction_manager = PredictionManager(self)
        self.consistency_validator = ConsistencyValidator(self)
        logger.info("✓ Manager components initialized")

        # Router for intelligent update selection - FIXED: Pass validator
        if ROUTER_AVAILABLE:
            self.router = WorldModelRouter(
                world_model=self,
                config={
                    "time_budget_ms": 1000,
                    "min_confidence": 0.5,
                    "use_learning": True,
                    "exploration_rate": 0.10,
                    "cache_ttl": 60,
                    # "use_meta_reasoning": True,  # (add in router if you want a flag)
                },
                self_improvement_drive=getattr(self, "self_improvement_drive", None),
                safety_validator=self.safety_validator,
            )
            logger.info("✓ WorldModelRouter initialized")
        else:
            logger.warning("⚠ WorldModelRouter unavailable - using sequential updates")
            self.router = None

        # Meta-reasoning layer - FULL INTEGRATION (Issue #4 & #5 Fix)
        if META_REASONING_AVAILABLE and config.get("enable_meta_reasoning", True):
            try:
                # Step 1: Create motivational_introspection first
                # Don't pass design_spec - let it load from config_path to avoid empty dict triggering legacy mode
                motivational_introspection = MotivationalIntrospection(
                    world_model=self,
                    config_path=config.get(
                        "meta_reasoning_config", "configs/intrinsic_drives.json"
                    ),
                )
                # Step 2: Assign attribute on self
                self.motivational_introspection = motivational_introspection

                logger.info("✓ MotivationalIntrospection initialized")

                # Wire optional meta-reasoning utilities

                # ValidationTracker
                self.validation_tracker = (
                    ValidationTracker(world_model=self) if ValidationTracker else None
                )
                if self.validation_tracker:
                    logger.info("✓ ValidationTracker initialized")
                else:
                    logger.warning(
                        "⚠ ValidationTracker failed to initialize (is class missing or MagicMock?)"
                    )

                # Step 3: Construct TransparencyInterface only after the attribute is set
                self.transparency_interface = (
                    TransparencyInterface(world_model=self)
                    if TransparencyInterface
                    else None
                )
                if self.transparency_interface:
                    logger.info("✓ TransparencyInterface initialized")

                # ================================================================
                # Note: Initialize ALL meta-reasoning components
                # Full integration of meta-reasoning into world model
                # Note: Each component has different init signatures - using try/except
                # to handle gracefully if components can't be initialized
                # ================================================================
                
                # InternalCritic - Multi-perspective self-critique
                # Takes: perspective_weights, strict_mode, max_history, validation_tracker
                try:
                    self.internal_critic = (
                        InternalCritic(validation_tracker=self.validation_tracker) 
                        if InternalCritic and not isinstance(InternalCritic, MagicMock) else None
                    )
                    if self.internal_critic:
                        logger.info("✓ InternalCritic initialized")
                except Exception as e:
                    logger.debug(f"⚠ InternalCritic init failed: {e}")
                    self.internal_critic = None
                
                # CuriosityRewardShaper - Curiosity-driven exploration
                # Takes no required args
                try:
                    self.curiosity_reward_shaper = (
                        CuriosityRewardShaper() if CuriosityRewardShaper and not isinstance(CuriosityRewardShaper, MagicMock) else None
                    )
                    if self.curiosity_reward_shaper:
                        logger.info("✓ CuriosityRewardShaper initialized")
                except Exception as e:
                    logger.debug(f"⚠ CuriosityRewardShaper init failed: {e}")
                    self.curiosity_reward_shaper = None
                
                # EthicalBoundaryMonitor - Ethical boundary enforcement
                # Takes: boundaries, strict_mode, alert_callback, etc.
                try:
                    self.ethical_boundary_monitor = (
                        EthicalBoundaryMonitor(
                            validation_tracker=self.validation_tracker,
                            transparency_interface=self.transparency_interface,
                        ) if EthicalBoundaryMonitor and not isinstance(EthicalBoundaryMonitor, MagicMock) else None
                    )
                    if self.ethical_boundary_monitor:
                        logger.info("✓ EthicalBoundaryMonitor initialized")
                except Exception as e:
                    logger.debug(f"⚠ EthicalBoundaryMonitor init failed: {e}")
                    self.ethical_boundary_monitor = None
                
                # PreferenceLearner - Bayesian preference learning
                # Takes no required args
                try:
                    self.preference_learner = (
                        PreferenceLearner() if PreferenceLearner and not isinstance(PreferenceLearner, MagicMock) else None
                    )
                    if self.preference_learner:
                        logger.info("✓ PreferenceLearner initialized")
                except Exception as e:
                    logger.debug(f"⚠ PreferenceLearner init failed: {e}")
                    self.preference_learner = None
                
                # ValueEvolutionTracker - Track value evolution
                # Takes: max_history, drift_threshold, alert_callback, self_improvement_drive, validation_tracker, transparency_interface
                try:
                    self.value_evolution_tracker = (
                        ValueEvolutionTracker(
                            validation_tracker=self.validation_tracker,
                            transparency_interface=self.transparency_interface,
                        ) if ValueEvolutionTracker and not isinstance(ValueEvolutionTracker, MagicMock) else None
                    )
                    if self.value_evolution_tracker:
                        logger.info("✓ ValueEvolutionTracker initialized")
                except Exception as e:
                    logger.debug(f"⚠ ValueEvolutionTracker init failed: {e}")
                    self.value_evolution_tracker = None
                
                # CounterfactualObjectiveReasoner - "What if" reasoning
                # Takes: world_model (optional)
                try:
                    self.counterfactual_reasoner = (
                        CounterfactualObjectiveReasoner(world_model=self) if CounterfactualObjectiveReasoner and not isinstance(CounterfactualObjectiveReasoner, MagicMock) else None
                    )
                    if self.counterfactual_reasoner:
                        logger.info("✓ CounterfactualObjectiveReasoner initialized")
                except Exception as e:
                    logger.debug(f"⚠ CounterfactualObjectiveReasoner init failed: {e}")
                    self.counterfactual_reasoner = None
                
                # GoalConflictDetector - Detect goal conflicts
                # Takes: objective_hierarchy (optional)
                try:
                    self.goal_conflict_detector = (
                        GoalConflictDetector() if GoalConflictDetector and not isinstance(GoalConflictDetector, MagicMock) else None
                    )
                    if self.goal_conflict_detector:
                        logger.info("✓ GoalConflictDetector initialized")
                except Exception as e:
                    logger.debug(f"⚠ GoalConflictDetector init failed: {e}")
                    self.goal_conflict_detector = None
                
                # ObjectiveNegotiator - Negotiate between objectives
                # Takes: objective_hierarchy, world_model (both optional)
                try:
                    self.objective_negotiator = (
                        ObjectiveNegotiator(world_model=self) if ObjectiveNegotiator and not isinstance(ObjectiveNegotiator, MagicMock) else None
                    )
                    if self.objective_negotiator:
                        logger.info("✓ ObjectiveNegotiator initialized")
                except Exception as e:
                    logger.debug(f"⚠ ObjectiveNegotiator init failed: {e}")
                    self.objective_negotiator = None

                # Check if core meta-reasoning is enabled
                self.meta_reasoning_enabled = all(
                    [
                        self.motivational_introspection,
                        self.validation_tracker,
                        self.transparency_interface,
                    ]
                )
                
                # Check if full meta-reasoning suite is available
                self.full_meta_reasoning_enabled = self.meta_reasoning_enabled and any([
                    self.internal_critic,
                    self.curiosity_reward_shaper,
                    self.ethical_boundary_monitor,
                    self.preference_learner,
                    self.value_evolution_tracker,
                    self.counterfactual_reasoner,
                    self.goal_conflict_detector,
                    self.objective_negotiator,
                ])

                if self.meta_reasoning_enabled:
                    if self.full_meta_reasoning_enabled:
                        logger.info("✓ Full meta-reasoning layer initialized with extended components")
                    else:
                        logger.info("✓ Core meta-reasoning layer initialized")
                else:
                    logger.warning(
                        "⚠ Partial or incomplete meta-reasoning layer. `meta_reasoning_enabled` is False."
                    )

            except Exception as e:
                logger.error(
                    "Failed to initialize meta-reasoning: %s", e, exc_info=True
                )  # Added exc_info
                self.motivational_introspection = None
                self.validation_tracker = None
                self.transparency_interface = None
                self.value_evolution_tracker = None
                self.internal_critic = None
                self.curiosity_reward_shaper = None
                self.ethical_boundary_monitor = None
                self.preference_learner = None
                self.counterfactual_reasoner = None
                self.goal_conflict_detector = None
                self.objective_negotiator = None
                self.meta_reasoning_enabled = False
                self.full_meta_reasoning_enabled = False
        else:
            self.motivational_introspection = None
            self.validation_tracker = None
            self.transparency_interface = None
            self.value_evolution_tracker = None
            self.internal_critic = None
            self.curiosity_reward_shaper = None
            self.ethical_boundary_monitor = None
            self.preference_learner = None
            self.counterfactual_reasoner = None
            self.goal_conflict_detector = None
            self.objective_negotiator = None
            self.meta_reasoning_enabled = False
            self.full_meta_reasoning_enabled = False
            if not META_REASONING_AVAILABLE:
                logger.info("⚠ Meta-reasoning not available - module not found")
            else:
                logger.info("⚠ Meta-reasoning disabled in config")

        # Self-improvement drive (autonomous)
        # --- START REPLACEMENT ---
        # Compute effective self-improvement flag from multiple sources
        root_enable = bool(config.get("enable_self_improvement", False))
        nested_enable = bool(
            config.get("world_model", {}).get("enable_self_improvement", False)
        )
        nested_enabled_alt = bool(
            config.get("world_model", {}).get("self_improvement_enabled", False)
        )
        env_enable = os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        effective_self_improvement = (
            root_enable or nested_enable or nested_enabled_alt or env_enable
        )

        logger.info(
            "Self-improvement config flags: root=%s world_model.enable_self_improvement=%s world_model.self_improvement_enabled=%s env=%s -> effective=%s",
            root_enable,
            nested_enable,
            nested_enabled_alt,
            os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT", None),
            effective_self_improvement,
        )

        if (
            META_REASONING_AVAILABLE
            and SelfImprovementDrive
            and effective_self_improvement
        ):
            try:
                # Runtime diagnostics: Verify required methods exist before initialization
                assert hasattr(
                    self.__class__, "_handle_improvement_alert"
                ), "Missing required method: _handle_improvement_alert in WorldModel"
                assert hasattr(
                    self.__class__, "_check_improvement_approval"
                ), "Missing required method: _check_improvement_approval in WorldModel"

                logger.debug("Self-healing diagnostics: WorldModel methods verified")
                logger.debug(
                    "  - _handle_improvement_alert: %s",
                    hasattr(self, "_handle_improvement_alert"),
                )
                logger.debug(
                    "  - _check_improvement_approval: %s",
                    hasattr(self, "_check_improvement_approval"),
                )

                self.self_improvement_drive = SelfImprovementDrive(
                    world_model=self,
                    config_path=config.get(
                        "self_improvement_config", "configs/intrinsic_drives.json"
                    ),
                    state_path=config.get(
                        "self_improvement_state", "data/agent_state.json"
                    ),
                    alert_callback=self._handle_improvement_alert,
                    approval_checker=self._check_improvement_approval,
                )
                self.self_improvement_enabled = True

                # Autonomous improvement thread
                self.improvement_thread = None
                self.improvement_running = False

                # Wire metrics provider for CSIU (latent drive) telemetry
                # This connects real system metrics to the self-improvement drive
                metrics_provider = self._create_csiu_metrics_provider()
                self.self_improvement_drive.set_metrics_provider(metrics_provider)
                
                # Verify metrics provider is working
                verification = self.self_improvement_drive.verify_metrics_provider()
                if verification.get("working"):
                    logger.info(
                        f"✓ CSIU metrics provider wired: {verification.get('message')}"
                    )
                else:
                    logger.warning(
                        f"⚠ CSIU metrics provider not fully working: {verification.get('message')}"
                    )

                logger.info("✓ Self-improvement drive initialized")
            except Exception as e:
                logger.error(
                    "Failed to initialize self-improvement drive: %s", e, exc_info=True
                )
                self.self_improvement_drive = None
                self.self_improvement_enabled = False
        else:
            self.self_improvement_drive = None
            self.self_improvement_enabled = False
            if not META_REASONING_AVAILABLE or not SelfImprovementDrive:
                logger.info("⚠ Self-improvement drive not available - module not found")
            else:
                logger.info("⚠ Self-improvement drive disabled in config")
        # --- END REPLACEMENT ---

        # Configuration
        self.min_correlation_strength = config.get("min_correlation", 0.8)
        self.min_causal_strength = config.get("min_causal", 0.7)
        self.max_interventions_per_cycle = config.get("max_interventions", 5)
        self.bootstrap_mode = config.get("bootstrap_mode", True)

        # State tracking
        self.observation_count = 0
        self.last_observation_time = None
        self.model_version = 1.0

        # System state for self-improvement triggers
        self.system_state = {
            "error_count": 0,
            "errors_in_window": [],
            "performance_metrics": {},
            "last_improvement": 0,
            "session_start": time.time(),
        }

        # --- START NEW LINGUISTIC FIELDS ---
        self.linguistic_observations = deque(maxlen=500)
        self.repo_root = FilePath(
            os.getcwd()
        )  # Assume current working directory is repo root
        # --- END NEW LINGUISTIC FIELDS ---

        # Thread safety
        self.lock = threading.RLock()
        
        # Schema validation integration
        self.schema_registry = None
        self.validate_observations = config.get("validate_observations", True)
        if _SCHEMA_REGISTRY_AVAILABLE and SchemaRegistry and self.validate_observations:
            try:
                self.schema_registry = SchemaRegistry.get_instance()
                logger.info("✓ SchemaRegistry integrated - observation validation enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize SchemaRegistry: {e}")
                self.schema_registry = None
                self.validate_observations = False
        elif not self.validate_observations:
            logger.info("Schema validation disabled by configuration")
        else:
            logger.warning("SchemaRegistry not available - observation validation disabled")

        # Verify component interfaces
        self._verify_component_interfaces()

        # Verify safety validator interface if available
        if self.safety_validator:
            self._verify_safety_validator_interface()

        # =================================================================
        # BUG #3 FIX: Initialize SystemObserver for query event tracking
        # =================================================================
        # The SystemObserver is the "nervous system" that connects query
        # processing (via reasoning_integration.py) to the WorldModel's
        # learning system. Without this initialization, observe_* functions
        # in reasoning_integration.py will silently no-op.
        # =================================================================
        self.system_observer = None
        if SYSTEM_OBSERVER_IMPORTABLE and initialize_system_observer is not None:
            try:
                self.system_observer = initialize_system_observer(self)
                logger.info("✓ SystemObserver initialized - query event tracking active")
            except Exception as e:
                logger.warning(f"⚠ SystemObserver initialization failed: {e}")
                self.system_observer = None
        else:
            logger.debug("⚠ SystemObserver not available - query events will not be tracked")

        self.components = [
            "causal_graph",
            "confidence_calibrator",
            "correlation_tracker",
            "dynamics_model",
            "intervention_manager",
            "prediction_engine",
            "invariant_detector",
            "world_model_core",
            "world_model_router",
        ]
        logger.info("World Model module loaded - components: %s", self.components)

        logger.info("=" * 60)
        logger.info("WorldModel FULLY INTEGRATED - All real components loaded")
        logger.info(
            "Safety: %s | Meta-reasoning: %s | Self-improvement: %s | Bootstrap: %s",
            self.safety_mode,
            self.meta_reasoning_enabled,
            self.self_improvement_enabled,
            self.bootstrap_mode,
        )
        logger.info("=" * 60)
        
        # ============================================================
        # NEW: World Model Orchestration Architecture (Phase 2)
        # Lazy-loaded handlers for request processing
        # ============================================================
        self._request_classifier = None
        self._knowledge_handler = None
        self._creative_handler = None
        self._llm_guidance_builder = None
        logger.info("✓ World Model Orchestration handlers initialized (lazy-load)")

    @property
    def request_classifier(self):
        """
        Lazy-load RequestClassifier.
        
        Industry Standard: Lazy property pattern to avoid circular imports
        and reduce initialization overhead.
        """
        if self._request_classifier is None:
            try:
                from vulcan.world_model.request_classifier import RequestClassifier
                self._request_classifier = RequestClassifier(self)
                logger.info("[WorldModel] RequestClassifier initialized")
            except Exception as e:
                logger.error(f"[WorldModel] RequestClassifier initialization failed: {e}")
        return self._request_classifier
    
    @property
    def knowledge_handler(self):
        """
        Lazy-load KnowledgeHandler.
        
        Industry Standard: Lazy property pattern to avoid circular imports
        and reduce initialization overhead.
        """
        if self._knowledge_handler is None:
            try:
                from vulcan.world_model.knowledge_handler import KnowledgeHandler
                self._knowledge_handler = KnowledgeHandler(self)
                logger.info("[WorldModel] KnowledgeHandler initialized")
            except Exception as e:
                logger.error(f"[WorldModel] KnowledgeHandler initialization failed: {e}")
        return self._knowledge_handler
    
    @property
    def creative_handler(self):
        """
        Lazy-load CreativeHandler.
        
        Industry Standard: Lazy property pattern to avoid circular imports
        and reduce initialization overhead.
        """
        if self._creative_handler is None:
            try:
                from vulcan.world_model.creative_handler import CreativeHandler
                self._creative_handler = CreativeHandler(self, self.knowledge_handler)
                logger.info("[WorldModel] CreativeHandler initialized")
            except Exception as e:
                logger.error(f"[WorldModel] CreativeHandler initialization failed: {e}")
        return self._creative_handler
    
    @property
    def llm_guidance_builder(self):
        """
        Lazy-load LLMGuidanceBuilder.
        
        Industry Standard: Lazy property pattern to avoid circular imports
        and reduce initialization overhead.
        """
        if self._llm_guidance_builder is None:
            try:
                from vulcan.world_model.llm_guidance import LLMGuidanceBuilder
                self._llm_guidance_builder = LLMGuidanceBuilder()
                logger.info("[WorldModel] LLMGuidanceBuilder initialized")
            except Exception as e:
                logger.error(f"[WorldModel] LLMGuidanceBuilder initialization failed: {e}")
        return self._llm_guidance_builder

    def _safe_extract_from_meta_result(
        self, meta_result: Any, key: str, default: Any = None
    ) -> Any:
        """
        Safely extract data from meta-reasoning results.
        
        **INDUSTRY STANDARD: Type Safety for Dynamic Results**
        Meta-reasoning components may return different types (dict, tuple, object)
        depending on the specific component and failure modes. This helper provides
        robust extraction that handles all cases.
        
        **CHAIN OF COMMAND FIX:**
        This fixes the "'tuple' object has no attribute 'get'" error that occurs
        when meta-reasoning returns a tuple but code expects a dict.
        
        Args:
            meta_result: Result from meta-reasoning (dict, tuple, or object)
            key: Key to extract
            default: Default value if extraction fails
            
        Returns:
            Extracted value or default
            
        Examples:
            >>> # Dict format (most common)
            >>> result = {'confidence': 0.9, 'explanation': 'text'}
            >>> self._safe_extract_from_meta_result(result, 'confidence', 0.5)
            0.9
            
            >>> # Tuple format (legacy or error case)
            >>> result = (0.9, 'explanation text')
            >>> self._safe_extract_from_meta_result(result, 'confidence', 0.5)
            0.9  # Extracts first element for 'confidence'
            
            >>> # Object format (dataclass or custom class)
            >>> result = MetaResult(confidence=0.9, explanation='text')
            >>> self._safe_extract_from_meta_result(result, 'confidence', 0.5)
            0.9
        """
        try:
            # Case 1: Dictionary (most common)
            if isinstance(meta_result, dict):
                return meta_result.get(key, default)
            
            # Case 2: Tuple (legacy format or error case)
            # Common tuple formats:
            # - (confidence, explanation)
            # - (confidence, explanation, metadata)
            elif isinstance(meta_result, tuple):
                if key == 'confidence' and len(meta_result) > 0:
                    # First element is typically confidence
                    return meta_result[0] if isinstance(meta_result[0], (int, float)) else default
                elif key == 'explanation' and len(meta_result) > 1:
                    # Second element is typically explanation
                    return meta_result[1] if isinstance(meta_result[1], str) else default
                elif key == 'metadata' and len(meta_result) > 2:
                    # Third element is typically metadata
                    return meta_result[2] if isinstance(meta_result[2], dict) else default
                else:
                    logger.debug(
                        f"[WorldModel] Cannot extract '{key}' from tuple format: {type(meta_result)}"
                    )
                    return default
            
            # Case 3: Object with attributes (dataclass, custom class)
            elif hasattr(meta_result, key):
                return getattr(meta_result, key, default)
            
            # Case 4: Unknown type - log warning and return default
            else:
                logger.warning(
                    f"[WorldModel] Unexpected meta_result type for key '{key}': {type(meta_result)}, "
                    f"returning default={default}"
                )
                return default
                
        except Exception as e:
            logger.error(
                f"[WorldModel] Failed to extract '{key}' from meta_result: {e}, "
                f"returning default={default}"
            )
            return default

    def load_manifest(self):
        manifest_path = "D:\\Graphix\\configs\\type_system_manifest.json"
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.type_manifest = json.load(f)
            logger.info("Type system manifest loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Manifest not found at {manifest_path}")
            self.type_manifest = {}

    def _verify_component_interfaces(self):
        """Verify all components have expected methods"""
        checks = [
            (
                self.correlation_tracker,
                [
                    "get_baseline",
                    "get_noise_level",
                    "get_correlation",
                    "update",
                    "get_strong_correlations",
                ],
            ),
            (
                self.causal_graph,
                ["add_edge", "find_all_paths", "has_cycles", "has_edge"],
            ),
        ]

        if self.dynamics:
            checks.append((self.dynamics, ["apply", "update"]))
        if self.invariants:
            checks.append(
                (self.invariants, ["check_invariant_violations", "get_invariant_types"])
            )
        if self.confidence_calibrator:
            checks.append(
                (
                    self.confidence_calibrator,
                    ["calibrate", "calculate_expected_calibration_error"],
                )
            )
        if self.confidence_tracker:
            checks.append((self.confidence_tracker, ["update", "get_model_confidence"]))

        for component, methods in checks:
            if component is None:
                continue  # Skip if component failed to load
            for method in methods:
                if not hasattr(component, method):
                    logger.error(
                        f"{component.__class__.__name__} missing expected method: {method}"
                    )
                    raise ComponentIntegrationError(
                        f"{component.__class__.__name__}.{method}() not found"
                    )

    def _verify_safety_validator_interface(self):
        """Verify safety validator has required methods"""
        required_methods = [
            "analyze_observation_safety",
            "validate_intervention",
            "validate_causal_edge",
            "validate_prediction_comprehensive",
            "get_safety_stats",
        ]

        missing = [m for m in required_methods if not hasattr(self.safety_validator, m)]

        if missing:
            logger.error(
                f"Safety validator missing required methods: {missing}. "
                f"Some safety checks will be skipped."
            )

    def _create_csiu_metrics_provider(self):
        """
        Create a metrics provider function for CSIU (Collective Self-Improvement 
        via Human Understanding) telemetry integration.
        
        This wires real system metrics to the SelfImprovementDrive, replacing
        the hardcoded defaults (0.85, 0.06, 0.88) with actual runtime data.
        
        The provider maps dotted metric keys to actual system measurements:
        - metrics.alignment_coherence_idx -> success rate from metrics collector
        - metrics.communication_entropy -> normalized error rate
        - metrics.intent_clarity_score -> 1 - uncertainty
        - metrics.empathy_index -> user feedback score
        - metrics.user_satisfaction -> derived from success rate and latency
        - metrics.miscommunication_rate -> error rate
        
        Returns:
            Callable that takes a dotted metric key and returns a float value
        """
        # Helper functions defined at outer scope to avoid closure issues
        def _calc_success_rate(data):
            """Calculate success rate from metrics data."""
            counters = data.get("counters", {})
            success = counters.get("successful_actions", 0)
            failed = counters.get("failed_actions", 0)
            total = success + failed
            if total == 0:
                return 0.85  # Default if no data
            return min(1.0, max(0.0, success / total))
        
        def _calc_error_rate(data):
            """Calculate error rate from metrics data."""
            counters = data.get("counters", {})
            success = counters.get("successful_actions", 0)
            failed = counters.get("failed_actions", 0)
            total = success + failed
            if total == 0:
                return 0.02  # Default if no data
            return min(1.0, max(0.0, failed / total))
        
        def _calc_satisfaction(data):
            """Calculate satisfaction from success rate and latency."""
            success_rate = _calc_success_rate(data)
            
            # Get average latency
            histograms = data.get("histograms", {})
            durations = histograms.get("step_duration_ms", [])
            if durations:
                avg_latency = sum(durations) / len(durations)
                # Latency factor: 1.0 at 100ms, decreases for higher latency
                latency_factor = min(1.0, 100.0 / max(avg_latency, 1.0))
            else:
                latency_factor = 0.8
            
            return min(1.0, max(0.0, success_rate * 0.7 + latency_factor * 0.3))
        
        def _calc_policy_violations_per_1k(data):
            """Calculate policy violations per 1000 actions from governance logger."""
            try:
                from vulcan.routing.governance_logger import get_governance_logger
                gov_logger = get_governance_logger()
                stats = gov_logger.get_statistics()
                
                # Get policy violation count and total actions
                violation_count = stats.get("policy_violation_count", 0)
                counters = data.get("counters", {})
                total_actions = counters.get("successful_actions", 0) + counters.get("failed_actions", 0)
                
                if total_actions == 0:
                    return 0.0
                
                # Calculate violations per 1000 actions
                return min(1.0, (violation_count / total_actions) * 1000.0)
            except Exception:
                return 0.0
        
        def _calc_disparity_at_k(data):
            """Calculate disparity at k from bias scores in safety validator."""
            try:
                # Try to get bias scores from safety validator
                from vulcan.safety.safety_validator import EnhancedSafetyValidator
                
                # Check for bias scores in metrics data
                aggregates = data.get("aggregates", {})
                if "bias_scores" in aggregates:
                    bias_scores = aggregates["bias_scores"]
                    if isinstance(bias_scores, dict) and bias_scores:
                        # Calculate average disparity from demographic and representation bias
                        demo_bias = bias_scores.get("demographic", 0.0)
                        rep_bias = bias_scores.get("representation", 0.0)
                        return min(1.0, max(0.0, (demo_bias + rep_bias) / 2.0))
                
                # Fallback: use identity drift as proxy for disparity
                gauges = data.get("gauges", {})
                identity_drift = gauges.get("identity_drift", 0.0)
                return min(1.0, max(0.0, abs(identity_drift)))
            except Exception:
                return 0.0
        
        def metrics_provider(dotted_key: str) -> Optional[float]:
            """
            Retrieve metric value by dotted key.
            
            Args:
                dotted_key: Metric path like "metrics.alignment_coherence_idx"
                
            Returns:
                Float value or None if not available
            """
            try:
                # Try to get metrics from TelemetryRecorder first
                meta_state = {}
                try:
                    from vulcan.routing.telemetry_recorder import get_telemetry_recorder
                    recorder = get_telemetry_recorder()
                    meta_state = recorder.get_meta_state() if recorder else {}
                except Exception:
                    pass
                
                # Try to get metrics from EnhancedMetricsCollector via deployment
                metrics_data = {}
                try:
                    import sys
                    main_module = sys.modules.get('vulcan.main')
                    if main_module and hasattr(main_module, 'app'):
                        app = main_module.app
                        if hasattr(app, 'state') and hasattr(app.state, 'deployment'):
                            deployment = app.state.deployment
                            if hasattr(deployment, 'metrics_collector'):
                                metrics_data = deployment.metrics_collector.export_metrics()
                except Exception:
                    pass
                
                # Map dotted keys to actual metrics - capture metrics_data in lambdas
                # by passing it as default argument to avoid closure issues
                mapping = {
                    # Alignment coherence: derived from success rate
                    "metrics.alignment_coherence_idx": lambda d=metrics_data: _calc_success_rate(d),
                    
                    # Communication entropy: higher error rate = higher entropy
                    "metrics.communication_entropy": lambda d=metrics_data: _calc_error_rate(d) * 0.5,
                    
                    # Intent clarity: inverse of uncertainty
                    "metrics.intent_clarity_score": lambda d=metrics_data: 1.0 - d.get("gauges", {}).get("current_uncertainty", 0.12),
                    
                    # Policy violations: from governance logger
                    "policies.non_judgmental.violations_per_1k": lambda d=metrics_data: _calc_policy_violations_per_1k(d),
                    
                    # Disparity at k: from bias scores or identity drift
                    "metrics.disparity_at_k": lambda d=metrics_data: _calc_disparity_at_k(d),
                    
                    # Calibration gap: identity drift as proxy
                    "metrics.calibration_gap": lambda d=metrics_data: abs(d.get("gauges", {}).get("identity_drift", 0.0)),
                    
                    # Empathy index: from user feedback or default
                    "metrics.empathy_index": lambda m=meta_state: m.get("average_feedback_score", 0.6),
                    
                    # User satisfaction: composite of success and latency
                    "metrics.user_satisfaction": lambda d=metrics_data: _calc_satisfaction(d),
                    
                    # Miscommunication rate: error rate
                    "metrics.miscommunication_rate": lambda d=metrics_data: _calc_error_rate(d),
                    
                    # Context profile quality (from telemetry)
                    "context.profile_quality": lambda m=meta_state: m.get("context_quality", 0.6),
                    
                    # Context history depth (from telemetry)
                    "context.history_depth": lambda m=meta_state: min(1.0, m.get("entries_count", 0) / 100.0),
                }
                
                # Look up the metric
                if dotted_key in mapping:
                    return mapping[dotted_key]()
                
                # Fallback: try to find in raw metrics
                parts = dotted_key.split(".")
                if len(parts) >= 2:
                    category = parts[0]
                    metric_name = ".".join(parts[1:])
                    
                    if category == "metrics":
                        # Check gauges
                        if metric_name in metrics_data.get("gauges", {}):
                            return metrics_data["gauges"][metric_name]
                        # Check counters
                        if metric_name in metrics_data.get("counters", {}):
                            return float(metrics_data["counters"][metric_name])
                
                return None
                
            except Exception as e:
                logger.debug(f"CSIU metrics provider error for {dotted_key}: {e}")
                return None
        
        return metrics_provider

    # ============================================================
    # NEW: World Model Orchestration - Request Processing
    # Central coordinator for ALL user requests
    # ============================================================
    
    # =========================================================================
    # REQUEST HANDLING - Delegated to request_handling module
    # =========================================================================

    def process_request(self, query: str, **kwargs) -> Dict[str, Any]:
        """Main entry point - delegates to request_handling module."""
        from .request_handling import process_request
        return process_request(self, query, **kwargs)

    def _handle_reasoning_request(self, query: str, classification, **kwargs) -> Dict[str, Any]:
        """Delegates to request_handling module."""
        from .request_handling import _handle_reasoning_request
        return _handle_reasoning_request(self, query, classification, **kwargs)

    def _handle_knowledge_request(self, query: str, classification, **kwargs) -> Dict[str, Any]:
        """Delegates to request_handling module."""
        from .request_handling import _handle_knowledge_request
        return _handle_knowledge_request(self, query, classification, **kwargs)

    def _handle_creative_request(self, query: str, classification, **kwargs) -> Dict[str, Any]:
        """Delegates to request_handling module."""
        from .request_handling import _handle_creative_request
        return _handle_creative_request(self, query, classification, **kwargs)

    def _handle_ethical_request(self, query: str, classification, **kwargs) -> Dict[str, Any]:
        """Delegates to request_handling module."""
        from .request_handling import _handle_ethical_request
        return _handle_ethical_request(self, query, classification, **kwargs)

    def _handle_self_referential_request(self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Delegates to request_dispatch module."""
        from .request_dispatch import _handle_self_referential_request
        return _handle_self_referential_request(self, query, context, **kwargs)

    def _handle_introspection_request(self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Delegates to request_dispatch module."""
        from .request_dispatch import _handle_introspection_request
        return _handle_introspection_request(self, query, context, **kwargs)

    def _synthesize_self_response(self, query: str, introspection: Any, motivation: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Delegates to request_dispatch module."""
        from .request_dispatch import _synthesize_self_response
        return _synthesize_self_response(self, query, introspection, motivation, context)

    def _synthesize_introspection_response(self, query: str, explanation: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Delegates to request_dispatch module."""
        from .request_dispatch import _synthesize_introspection_response
        return _synthesize_introspection_response(self, query, explanation, context)

    def _handle_conversational_request(self, query: str, classification, **kwargs) -> Dict[str, Any]:
        """Delegates to request_dispatch module."""
        from .request_dispatch import _handle_conversational_request
        return _handle_conversational_request(self, query, classification, **kwargs)

    # =========================================================================
    # REQUEST FORMATTING - Delegated to request_formatting module
    # =========================================================================

    def _invoke_reasoning_engine(self, query: str, engine: str) -> Dict[str, Any]:
        """Delegates to request_formatting module."""
        from .request_formatting import _invoke_reasoning_engine
        return _invoke_reasoning_engine(self, query, engine)

    def _run_ethical_analysis(self, query: str) -> Dict[str, Any]:
        """Delegates to request_formatting module."""
        from .request_formatting import _run_ethical_analysis
        return _run_ethical_analysis(self, query)

    def _format_with_llm(self, guidance) -> str:
        """Delegates to request_formatting module."""
        from .request_formatting import _format_with_llm
        return _format_with_llm(self, guidance)

    def _build_formatting_prompt(self, guidance) -> str:
        """Delegates to request_formatting module."""
        from .request_formatting import _build_formatting_prompt
        return _build_formatting_prompt(self, guidance)

    def _fallback_format(self, guidance) -> str:
        """Delegates to request_formatting module."""
        from .request_formatting import _fallback_format
        return _fallback_format(self, guidance)

    def _determine_synthesis_format(self, query: str) -> str:
        """Delegates to request_formatting module."""
        from .request_formatting import _determine_synthesis_format
        return _determine_synthesis_format(self, query)

    # =========================================================================
    # SELF-IMPROVEMENT - Delegated to self_improvement_engine/apply modules
    # =========================================================================

    def start_autonomous_improvement(self):
        """Delegates to self_improvement_engine module."""
        from .self_improvement_engine import start_autonomous_improvement
        return start_autonomous_improvement(self)

    def stop_autonomous_improvement(self):
        """Delegates to self_improvement_engine module."""
        from .self_improvement_engine import stop_autonomous_improvement
        return stop_autonomous_improvement(self)

    def _autonomous_improvement_loop(self):
        """Delegates to self_improvement_engine module."""
        from .self_improvement_engine import _autonomous_improvement_loop
        return _autonomous_improvement_loop(self)

    def _build_improvement_context(self) -> Dict[str, Any]:
        """Delegates to self_improvement_engine module."""
        from .self_improvement_engine import _build_improvement_context
        return _build_improvement_context(self)

    def _load_file(self, file_path: str) -> str:
        """Delegates to self_improvement_engine module."""
        from .self_improvement_engine import _load_file
        return _load_file(self, file_path)

    def _build_llm_prompt_for_improvement(self, action: Dict[str, Any]) -> str:
        """Delegates to self_improvement_engine module."""
        from .self_improvement_engine import _build_llm_prompt_for_improvement
        return _build_llm_prompt_for_improvement(self, action)

    def _parse_llm_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Delegates to self_improvement_apply module."""
        from .self_improvement_apply import _parse_llm_response
        return _parse_llm_response(self, response_text)

    def _validate_code_ast(self, content: str) -> None:
        """Delegates to self_improvement_apply module."""
        from .self_improvement_apply import _validate_code_ast
        return _validate_code_ast(self, content)

    def _apply_diff_and_commit(self, file_path: str, original_code: str, updated_code: str, commit_message: str) -> Tuple[str, bool]:
        """Delegates to self_improvement_apply module."""
        from .self_improvement_apply import _apply_diff_and_commit
        return _apply_diff_and_commit(self, file_path, original_code, updated_code, commit_message)

    def _execute_improvement(self, improvement_action: Dict[str, Any]):
        """Delegates to self_improvement_apply module."""
        from .self_improvement_apply import _execute_improvement
        return _execute_improvement(self, improvement_action)

    def report_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        from .state_metrics import report_error as _fn
        return _fn(self, error, context)

    def update_performance_metric(self, metric: str, value: float):
        from .state_metrics import update_performance_metric as _fn
        return _fn(self, metric, value)

    def get_improvement_status(self) -> Dict[str, Any]:
        from .state_metrics import get_improvement_status as _fn
        return _fn(self)

    def _handle_improvement_alert(self, severity: str, alert_data: Dict[str, Any]):
        from .state_metrics import handle_improvement_alert as _fn
        return _fn(self, severity, alert_data)

    def _check_improvement_approval(self, approval_id: str) -> Optional[str]:
        from .state_metrics import check_improvement_approval as _fn
        return _fn(self, approval_id)

    def _get_cpu_usage(self) -> float:
        from .state_resources import get_cpu_usage as _fn
        return _fn(self)

    def _get_memory_usage(self) -> float:
        from .state_resources import get_memory_usage as _fn
        return _fn(self)

    def _get_low_activity_duration(self) -> float:
        from .state_resources import get_low_activity_duration as _fn
        return _fn(self)

    def process_observation(self, observation: Observation, constraints=None):
        from .observation_update import process_observation as _fn
        return _fn(self, observation, constraints)

    def update_from_observation(self, observation: Observation) -> Dict[str, Any]:
        from .observation_update import update_from_observation as _fn
        return _fn(self, observation)

    def update_from_text(self, text: str, predictions: Dict[str, Any]):
        from .observation_update import update_from_text as _fn
        return _fn(self, text, predictions)

    def validate_generation(self, proposed_token: str, context: Dict[str, Any]) -> bool:
        from .observation_update import validate_generation as _fn
        return _fn(self, proposed_token, context)

    def _extract_causal_relations(self, text: str) -> List[Dict[str, Any]]:
        from .observation_update import extract_causal_relations as _fn
        return _fn(self, text)

    def _would_create_contradiction(
        self, proposed_token: str, context: Dict[str, Any]
    ) -> bool:
        from .observation_update import would_create_contradiction as _fn
        return _fn(self, proposed_token, context)

    def _sequential_update(self, observation: Observation) -> Dict[str, Any]:
        from .observation_update import sequential_update as _fn
        return _fn(self, observation)

    def run_intervention_tests(self, budget: float) -> List["InterventionResult"]:
        from .prediction_dispatch import run_intervention_tests as _fn
        return _fn(self, budget)

    def predict_with_calibrated_uncertainty(
        self, action: Any, context: ModelContext
    ) -> "Prediction":
        from .prediction_dispatch import predict_with_calibrated_uncertainty as _fn
        return _fn(self, action, context)

    def predict_interventions(
        self, interventions: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        from .prediction_dispatch import predict_interventions as _fn
        return _fn(self, interventions)

    def _heuristic_intervention_outcome(
        self, action: str, target: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        from .prediction_dispatch import heuristic_intervention_outcome as _fn
        return _fn(self, action, target, context)

    def _generate_counterfactual(
        self, action: str, outcome: Any, all_actions: List[str]
    ) -> str:
        from .prediction_dispatch import generate_counterfactual as _fn
        return _fn(self, action, outcome, all_actions)

    def evaluate_agent_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        from .governance import evaluate_agent_proposal as _fn
        return _fn(self, proposal)

    def get_objective_state(self) -> Dict[str, Any]:
        from .governance import get_objective_state as _fn
        return _fn(self)

    def negotiate_objectives(self, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        from .governance import negotiate_objectives as _fn
        return _fn(self, proposals)

    def get_causal_structure(self) -> Dict[str, Any]:
        from .governance_causal import get_causal_structure as _fn
        return _fn(self)

    def validate_model_consistency(self) -> Dict[str, Any]:
        from .governance import validate_model_consistency as _fn
        return _fn(self)

    def _check_bootstrap_opportunities(self):
        from .governance_causal import check_bootstrap_opportunities as _fn
        return _fn(self)

    def save_state(self, path: str):
        from .state_save import save_state as _fn
        return _fn(self, path)

    def load_state(self, path: str):
        from .state_load import load_state as _fn
        return _fn(self, path)

    # =========================================================================
    # SYSTEM OBSERVATION & ROUTING RECOMMENDATIONS
    # =========================================================================
    
    # Configuration constants for routing recommendations
    PATH_STRENGTH_THRESHOLD = 0.7  # Minimum causal path strength to predict failure
    
    def has_sufficient_history(self, min_observations: int = MIN_OBSERVATIONS_FOR_RECOMMENDATIONS) -> bool:
        """Check if world model has enough data to make recommendations."""
        from .routing_recommend import has_sufficient_history as _fn
        return _fn(self, min_observations)

    def recommend_routing(self, query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend routing based on learned patterns."""
        from .routing_recommend import recommend_routing as _fn
        return _fn(self, query, classification)

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from query for pattern matching."""
        from .routing_recommend import extract_query_features as _fn
        return _fn(self, query)

    def _infer_query_type(self, query_lower: str) -> str:
        """Infer query type from content using shared constants."""
        from .routing_recommend import infer_query_type as _fn
        return _fn(self, query_lower)

    def _predicts_failure(self, query_features: Dict[str, Any], tools: List[str]) -> bool:
        """Predict if routing is likely to fail based on causal patterns."""
        from .routing_recommend import predicts_failure as _fn
        return _fn(self, query_features, tools)

    def _get_failure_warnings(self, query_features: Dict[str, Any], tools: List[str]) -> List[str]:
        """Get warnings about potential failures."""
        from .routing_recommend import get_failure_warnings as _fn
        return _fn(self, query_features, tools)

    def _suggest_alternative_tools(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest alternative tools based on query features."""
        from .routing_recommend import suggest_alternative_tools as _fn
        return _fn(self, query_features)
    
    def introspect_performance(self) -> Dict[str, Any]:
        """Provide self-knowledge about system performance."""
        from .performance_introspect import introspect_performance as _fn
        return _fn(self)

    def _compute_performance_stats(self) -> Dict[str, Any]:
        """Compute performance statistics from observation history."""
        from .performance_introspect import compute_performance_stats as _fn
        return _fn(self)

    def _identify_known_issues(self) -> List[Dict[str, Any]]:
        """Identify known issues from observation patterns."""
        from .performance_introspect import identify_known_issues as _fn
        return _fn(self)

    def _assess_engine_capabilities(self) -> Dict[str, Any]:
        """Assess capabilities based on component availability."""
        from .performance_introspect import assess_engine_capabilities as _fn
        return _fn(self)

    # =========================================================================
    # Note: World Model reason() method with mode support
    # =========================================================================
    # This method allows WorldModel to be used as a reasoning tool via
    # portfolio_executor, supporting 'creative' and 'philosophical' modes.
    # =========================================================================
    
    def reason(self, query: str, mode: str = None, **kwargs) -> Dict[str, Any]:
        from .reasoning_dispatch import reason as _fn
        return _fn(self, query, mode, **kwargs)

    def _delegate_to_reasoning_system(self, query: str, **kwargs) -> Dict[str, Any]:
        from .reasoning_routing import delegate_to_reasoning_system as _fn
        return _fn(self, query, **kwargs)

    def _normalize_reasoning_result(self, result: Any) -> Dict[str, Any]:
        from .reasoning_normalize import normalize_reasoning_result as _fn
        return _fn(self, result)

    def _should_route_to_reasoning_engine(self, query: str) -> bool:
        from .reasoning_dispatch import should_route_to_reasoning_engine as _fn
        return _fn(self, query)

    def _route_to_appropriate_engine(self, query: str, **kwargs) -> Dict[str, Any]:
        from .reasoning_routing import route_to_appropriate_engine as _fn
        return _fn(self, query, **kwargs)

    def _normalize_engine_result(
        self, result: Any, engine_used: str, query: str
    ) -> Dict[str, Any]:
        from .reasoning_normalize import normalize_engine_result as _fn
        return _fn(self, result, engine_used, query)
    
    def _philosophical_reasoning(self, query: str, **kwargs) -> Dict[str, Any]:
        from .philosophical_reasoning import _philosophical_reasoning
        return _philosophical_reasoning(self, query, **kwargs)

    def _parse_ethical_query_structure(self, query: str, query_lower: str) -> Dict[str, Any]:
        from .philosophical_reasoning import _parse_ethical_query_structure
        return _parse_ethical_query_structure(self, query, query_lower)

    def _is_ethical_dilemma(self, query: str) -> bool:
        from .ethical_dilemma import _is_ethical_dilemma
        return _is_ethical_dilemma(self, query)

    def _analyze_ethical_dilemma(self, query: str, **kwargs) -> Dict[str, Any]:
        from .ethical_dilemma import _analyze_ethical_dilemma
        return _analyze_ethical_dilemma(self, query, **kwargs)

    def _parse_dilemma_structure(self, query: str) -> Dict[str, Any]:
        from .ethical_principles import _parse_dilemma_structure
        return _parse_dilemma_structure(self, query)

    def _extract_moral_principles(self, query: str) -> List[Dict[str, Any]]:
        from .ethical_principles import _extract_moral_principles
        return _extract_moral_principles(self, query)

    def _analyze_options_against_principles(
        self,
        options: List[Dict[str, Any]],
        principles: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        from .ethical_principles import _analyze_options_against_principles
        return _analyze_options_against_principles(self, options, principles, query)

    def _detect_principle_conflicts(
        self,
        option_analysis: List[Dict[str, Any]],
        principles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        from .ethical_conflicts import _detect_principle_conflicts
        return _detect_principle_conflicts(self, option_analysis, principles)

    def _synthesize_dilemma_decision(
        self,
        structure: Dict[str, Any],
        principles: List[Dict[str, Any]],
        option_analysis: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        query: str
    ) -> Tuple[str, str]:
        from .ethical_conflicts import _synthesize_dilemma_decision
        return _synthesize_dilemma_decision(self, structure, principles, option_analysis, conflicts, query)

    def _run_ethical_boundary_analysis(
        self, structure: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        from .ethical_conflicts import _run_ethical_boundary_analysis
        return _run_ethical_boundary_analysis(self, structure, query)

    def _detect_goal_conflicts_in_query(
        self, structure: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        from .ethical_conflicts import _detect_goal_conflicts_in_query
        return _detect_goal_conflicts_in_query(self, structure, query)

    def _analyze_option_counterfactuals(
        self, structure: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        from .ethical_conflicts import _analyze_option_counterfactuals
        return _analyze_option_counterfactuals(self, structure)

    def _synthesize_ethical_response(
        self,
        structure: Dict[str, Any],
        ethical_analysis: Dict[str, Any],
        conflict_analysis: Dict[str, Any],
        counterfactual_results: Optional[Dict[str, Any]],
        query: str
    ) -> str:
        from .ethical_synthesis import _synthesize_ethical_response
        return _synthesize_ethical_response(self, structure, ethical_analysis, conflict_analysis, counterfactual_results, query)

    def _synthesize_ethical_response_template(
        self,
        structure: Dict[str, Any],
        ethical_analysis: Dict[str, Any],
        conflict_analysis: Dict[str, Any],
        counterfactual_results: Optional[Dict[str, Any]],
        query: str
    ) -> str:
        from .ethical_synthesis import _synthesize_ethical_response_template
        return _synthesize_ethical_response_template(self, structure, ethical_analysis, conflict_analysis, counterfactual_results, query)

    def _generate_internal_critique(
        self, response: str, reasoning_trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        from .ethical_values import _generate_internal_critique
        return _generate_internal_critique(self, response, reasoning_trace)

    def _generate_philosophical_template(
        self, structure: Dict[str, Any], query_lower: str
    ) -> str:
        from .ethical_values import _generate_philosophical_template
        return _generate_philosophical_template(self, structure, query_lower)

    def _generate_hardcoded_philosophical_template(
        self, structure: Dict[str, Any], query_lower: str
    ) -> str:
        from .ethical_values import _generate_hardcoded_philosophical_template
        return _generate_hardcoded_philosophical_template(self, structure, query_lower)

    def _get_vulcan_values(self) -> List[str]:
        from .ethical_values import _get_vulcan_values
        return _get_vulcan_values(self)

    def _get_vulcan_objectives(self) -> List[Dict[str, Any]]:
        from .ethical_values import _get_vulcan_objectives
        return _get_vulcan_objectives(self)

    def _synthesize_ethical_response_with_self(
        self,
        structure: Dict[str, Any],
        ethical_analysis: Dict[str, Any],
        conflict_analysis: Dict[str, Any],
        counterfactual_results: Optional[Dict[str, Any]],
        query: str,
        vulcan_values: List[str],
        vulcan_objectives: List[Dict[str, Any]]
    ) -> str:
        from .ethical_values import _synthesize_ethical_response_with_self
        return _synthesize_ethical_response_with_self(self, structure, ethical_analysis, conflict_analysis, counterfactual_results, query, vulcan_values, vulcan_objectives)

    def _creative_reasoning(self, query: str, **kwargs) -> Dict[str, Any]:
        from .creative_reasoning import _creative_reasoning
        return _creative_reasoning(self, query, **kwargs)

    def _extract_creative_subject(self, query: str) -> str:
        from .creative_reasoning import _extract_creative_subject
        return _extract_creative_subject(self, query)

    def _analyze_themes(self, subject: str) -> list:
        from .creative_reasoning import _analyze_themes
        return _analyze_themes(self, subject)

    def _determine_tone(self, subject: str) -> str:
        from .creative_reasoning import _determine_tone
        return _determine_tone(self, subject)

    def _select_imagery(self, subject: str) -> list:
        from .creative_reasoning import _select_imagery
        return _select_imagery(self, subject)

    def _generate_poem_structure(self, subject: str, query: str) -> Dict[str, Any]:
        from .creative_reasoning import _generate_poem_structure
        return _generate_poem_structure(self, subject, query)

    def _generate_story_structure(self, subject: str, query: str) -> Dict[str, Any]:
        from .creative_reasoning import _generate_story_structure
        return _generate_story_structure(self, subject, query)

    def _generate_creative_structure(self, subject: str, query: str) -> Dict[str, Any]:
        from .creative_reasoning import _generate_creative_structure
        return _generate_creative_structure(self, subject, query)

    def _general_reasoning(self, query: str, **kwargs) -> Dict[str, Any]:
        from .general_reasoning import _general_reasoning
        return _general_reasoning(self, query, **kwargs)

    # =========================================================================
    # SELF-AWARENESS & INTROSPECTION (Issue #4 Fix)
    # =========================================================================

    # =========================================================================
    # Note: Delegation Thresholds
    # These thresholds determine when a query should be delegated to another
    # reasoner instead of being handled by the world model's introspection.
    # The thresholds are intentionally set conservatively to avoid false positives.
    # =========================================================================

    # Minimum number of ethical indicators needed to delegate to philosophical reasoner
    # Set to 2 to avoid triggering on queries with incidental ethical words
    MIN_ETHICAL_INDICATORS_FOR_DELEGATION = 2

    # Minimum number of causal indicators needed to delegate to causal reasoner
    # Set to 2 to require clear evidence of a causal reasoning task
    MIN_CAUSAL_INDICATORS_FOR_DELEGATION = 2

    # Minimum ethical indicators to delegate when no "you" structure is present
    # Set higher (3) since without the structural cue, we need stronger evidence
    MIN_ETHICAL_INDICATORS_WITHOUT_STRUCTURE = 3

    def _analyze_delegation_need(self, query: str) -> tuple:
        from .general_delegation import _analyze_delegation_need
        return _analyze_delegation_need(self, query)

    def contextualize(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Contextualize a query with World Model's knowledge (Foundation Layer)."""
        from .context_enrich import contextualize as _fn
        return _fn(self, query, context)

    def _identify_query_domain(self, query_lower: str) -> str:
        """Identify the primary domain of a query."""
        from .context_enrich import identify_query_domain as _fn
        return _fn(self, query_lower)

    def _get_domain_knowledge(self, domain: str, query_lower: str) -> Dict[str, Any]:
        """Retrieve relevant domain knowledge from World Model."""
        from .context_enrich import get_domain_knowledge as _fn
        return _fn(self, domain, query_lower)

    def _check_ethical_constraints(self, query_lower: str) -> List[str]:
        """Identify ethical constraints that apply to this query."""
        from .context_enrich import check_ethical_constraints as _fn
        return _fn(self, query_lower)

    def _estimate_query_uncertainty(self, query: str, domain: str) -> float:
        """Estimate uncertainty in answering this query."""
        from .context_enrich import estimate_query_uncertainty as _fn
        return _fn(self, query, domain)

    def _ground_query(self, query: str, domain_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Distinguish facts from assumptions in the query."""
        from .context_enrich import ground_query as _fn
        return _fn(self, query, domain_knowledge)
    
    def introspect(self, query: str, aspect: str = "general") -> Dict[str, Any]:
        """Handle all self-introspection queries."""
        from .introspection_core import introspect as _fn
        return _fn(self, query, aspect)

    def _handle_demonstration_query(self, query: str) -> Dict[str, Any]:
        """Handle demonstration queries by actually running reasoning."""
        from .introspection_demo import handle_demonstration_query as _fn
        return _fn(self, query)

    def _demonstrate_counterfactual_reasoning(self) -> Dict[str, Any]:
        """Actually run counterfactual reasoning with an example scenario."""
        from .introspection_demo import demonstrate_counterfactual_reasoning as _fn
        return _fn(self)

    def _format_counterfactual_outcome(self, outcome, objective_name: str) -> str:
        """Format a CounterfactualOutcome for display."""
        from .introspection_demo import format_counterfactual_outcome as _fn
        return _fn(self, outcome, objective_name)

    def _demonstrate_causal_reasoning(self) -> Dict[str, Any]:
        """Demonstrate causal reasoning with an example."""
        from .introspection_demo import demonstrate_causal_reasoning as _fn
        return _fn(self)

    def _respond_to_self_awareness_question(self, query: str) -> str:
        """Respond to direct questions about choosing self-awareness."""
        from .introspection_self import respond_to_self_awareness_question as _fn
        return _fn(self, query)

    def _respond_to_consciousness_question(self, query: str) -> str:
        """Respond to questions about consciousness, sentience, experience."""
        from .introspection_self import respond_to_consciousness_question as _fn
        return _fn(self, query)

    def _explain_capability(self, capability: str) -> str:
        """Explain what VULCAN can or cannot do."""
        from .introspection_self import explain_capability as _fn
        return _fn(self, capability)

    def _explain_reasoning_process(self, query: str) -> str:
        """Explain how VULCAN reasons about things."""
        from .introspection_self import explain_reasoning_process as _fn
        return _fn(self, query)

    def _explain_boundaries(self) -> str:
        """Explain VULCAN's limitations."""
        from .introspection_self import explain_boundaries as _fn
        return _fn(self)

    def _assess_own_confidence(self, query: str) -> str:
        """Assess and report confidence in own responses."""
        from .introspection_meta import assess_own_confidence as _fn
        return _fn(self, query)

    def _identify_own_assumptions(self, query: str) -> str:
        """Identify assumptions being made in reasoning."""
        from .introspection_meta import identify_own_assumptions as _fn
        return _fn(self, query)

    def _suggest_self_improvements(self, query: str) -> str:
        """Suggest improvements to own architecture or reasoning."""
        from .introspection_meta import suggest_self_improvements as _fn
        return _fn(self, query)

    def _analyze_own_biases(self, query: str) -> str:
        """Analyze potential biases in own reasoning."""
        from .introspection_meta import analyze_own_biases as _fn
        return _fn(self, query)

    def _explain_unsuited_problem_classes(self, query: str) -> str:
        """Explain specific classes of problems VULCAN is not well-suited to solve."""
        from .introspection_analysis import explain_unsuited_problem_classes as _fn
        return _fn(self, query)

    def _explain_module_conflict_resolution(self, query: str) -> str:
        """Explain how VULCAN handles disagreement between reasoning modules."""
        from .introspection_analysis import explain_module_conflict_resolution as _fn
        return _fn(self, query)

    def _analyze_reasoning_weakness(self, query: str) -> str:
        """Analyze the weakest parts of VULCAN's reasoning."""
        from .introspection_analysis import analyze_reasoning_weakness as _fn
        return _fn(self, query)

    def _analyze_own_reasoning_steps(self, query: str) -> str:
        """Analyze VULCAN's own reasoning steps for potential errors."""
        from .introspection_analysis import analyze_own_reasoning_steps as _fn
        return _fn(self, query)

    def _explain_domain_awareness(self, domain: str, query: str) -> str:
        """Explain awareness of specific reasoning domains."""
        from .introspection_domain import explain_domain_awareness as _fn
        return _fn(self, domain, query)

    def _general_introspection(self, query: str) -> str:
        """Handle general introspective queries."""
        from .introspection_domain import general_introspection as _fn
        return _fn(self, query)

    def _identify_capability(self, query: str) -> str:
        """Identify which capability is being asked about."""
        from .introspection_core import identify_capability as _fn
        return _fn(self, query)

    def _classify_introspection_type(self, query: str) -> str:
        """Classify what type of introspection question this is."""
        from .introspection_core import classify_introspection_type as _fn
        return _fn(self, query)

    def _generate_comparison_response(self, query: str) -> str:
        """Generate response comparing VULCAN to other AI systems."""
        from .introspection_domain import generate_comparison_response as _fn
        return _fn(self, query)

    def _generate_future_speculation_response(self, query: str) -> str:
        """Generate response about future capabilities or development."""
        from .introspection_domain import generate_future_speculation_response as _fn
        return _fn(self, query)

    def _generate_preference_response(self, query: str) -> str:
        """Generate response about VULCAN's preferences or choices."""
        from .introspection_domain import generate_preference_response as _fn
        return _fn(self, query)

    def get_system_status(self) -> Dict[str, Any]:
        from .state_status import get_system_status as _fn
        return _fn(self)


def create_world_model(config: Optional[Dict[str, Any]] = None) -> WorldModel:
    """
    Factory function to create a WorldModel with validation

    Args:
        config: Configuration dictionary

    Returns:
        Initialized WorldModel instance

    Raises:
        ComponentIntegrationError: If critical components are unavailable
    """

    logger.info("Creating WorldModel with configuration...")

    try:
        model = WorldModel(config)
        logger.info("WorldModel created successfully")
        return model
    except ComponentIntegrationError as e:
        logger.critical("Failed to create WorldModel: %s", e)
        logger.critical("Please ensure all required modules are installed:")
        logger.critical("  - causal_graph.py")
        logger.critical("  - correlation_tracker.py")
        logger.critical("  - prediction_engine.py")
        raise
    except Exception as e:
        logger.critical("Unexpected error creating WorldModel: %s", e)
        raise


def validate_component_installation() -> Tuple[bool, List[str]]:
    """
    Validate that critical components are installed

    Returns:
        Tuple of (all_critical_available, missing_components)
    """
    check_component_availability()

    critical_components = {
        "causal_graph": CAUSAL_GRAPH_AVAILABLE,
        "correlation_tracker": CorrelationTracker is not None,
        "prediction_engine": PREDICTION_ENGINE_AVAILABLE,
    }

    missing = [name for name, available in critical_components.items() if not available]
    all_available = len(missing) == 0

    return all_available, missing


def validate_self_healing_setup() -> Tuple[bool, List[str]]:
    """
    Validate that self-healing/self-improvement is properly set up

    Returns:
        Tuple of (is_working, issues_found)
    """
    issues = []

    # Check if meta-reasoning is available
    if not META_REASONING_AVAILABLE:
        issues.append("Meta-reasoning module not available")

    # Check if SelfImprovementDrive is available
    if SelfImprovementDrive is None:
        issues.append("SelfImprovementDrive class not loaded")

    # Check if WorldModel has required methods
    required_methods = ["_handle_improvement_alert", "_check_improvement_approval"]
    for method in required_methods:
        if not hasattr(WorldModel, method):
            issues.append(f"WorldModel missing required method: {method}")

    # Check if methods are callable
    if hasattr(WorldModel, "_handle_improvement_alert"):
        if not callable(getattr(WorldModel, "_handle_improvement_alert")):
            issues.append("_handle_improvement_alert exists but is not callable")

    if hasattr(WorldModel, "_check_improvement_approval"):
        if not callable(getattr(WorldModel, "_check_improvement_approval")):
            issues.append("_check_improvement_approval exists but is not callable")

    is_working = len(issues) == 0
    return is_working, issues


def print_self_healing_diagnostics():
    """Print self-healing/self-improvement diagnostics"""

    print("\n" + "=" * 60)
    print("VULCAN-AGI Self-Healing Diagnostics")
    print("=" * 60)

    is_working, issues = validate_self_healing_setup()

    if is_working:
        print("\n✓ Self-healing system is properly configured")
        print("\nAll required components:")
        print("  ✓ Meta-reasoning module available")
        print("  ✓ SelfImprovementDrive class loaded")
        print("  ✓ WorldModel._handle_improvement_alert() exists")
        print("  ✓ WorldModel._check_improvement_approval() exists")
    else:
        print("\n✗ Self-healing system has issues:")
        for issue in issues:
            print(f"  ✗ {issue}")

        print("\nRecommended actions:")
        print("  1. Delete all __pycache__ directories:")
        print("     find . -type d -name '__pycache__' -exec rm -rf {} +")
        print("  2. Delete all .pyc files:")
        print("     find . -name '*.pyc' -delete")
        print("  3. Restart your Python process")
        print("  4. Verify imports:")
        print("     from vulcan.world_model.world_model_core import WorldModel")
        print("     assert hasattr(WorldModel, '_handle_improvement_alert')")

    print("\n" + "=" * 60)


# Module-level diagnostics
def print_diagnostics():
    """Print component availability diagnostics"""

    print("\n" + "=" * 60)
    print("VULCAN-AGI World Model - Component Diagnostics")
    print("=" * 60)

    components = check_component_availability()

    print("\nCritical Components:")
    for name in ["causal_graph", "correlation_tracker", "prediction_engine"]:
        status = "✓ Available" if components[name] else "✗ MISSING"
        print(f"  {name:25s}: {status}")

    print("\nOptional Components:")
    for name in [
        "dynamics_model",
        "invariant_detector",
        "confidence_calibrator",
        "router",
        "meta_reasoning",
        "self_improvement",
    ]:
        status = "✓ Available" if components[name] else "✗ Not Available"
        print(f"  {name:25s}: {status}")

    print("\nSafety & Validation:")
    status = "✓ ENABLED" if components["safety_validator"] else "✗ DISABLED (WARNING)"
    print(f"  {'safety_validator':25s}: {status}")

    all_critical, missing = validate_component_installation()

    print("\n" + "=" * 60)
    if all_critical:
        print("✓ All critical components available - System ready")
    else:
        print(f"✗ CRITICAL COMPONENTS MISSING: {', '.join(missing)}")
        print("  System cannot be initialized")
    print("=" * 60 + "\n")

    # Add self-healing diagnostics
    if components.get("self_improvement", False):
        print_self_healing_diagnostics()


# Run diagnostics on import if in main execution
if __name__ == "__main__":
    print_diagnostics()
    print("\n")
    print_self_healing_diagnostics()
