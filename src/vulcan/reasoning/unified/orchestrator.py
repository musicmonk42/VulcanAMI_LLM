"""
Unified Reasoning Orchestrator Module

This module contains the main UnifiedReasoner class that orchestrates
reasoning across multiple reasoning engines and strategies.

The UnifiedReasoner class provides:
- Component initialization and lifecycle management
- Main reason() method with caching, validation, and strategy execution
- Tool selection and historical weight tracking
- Learning and adaptation from reasoning results
- Statistics tracking and comprehensive audit trail
- Resource cleanup and graceful shutdown

Method implementations are delegated to focused submodules:
- self_ref_detection: Self-referential query detection
- self_ref_handlers: Self-referential query handling
- self_ref_conclusion: Conclusion building for self-ref queries
- strategy_classification: Task classification and tool mapping
- strategy_planning: Plan creation and optimization
- task_execution: Task dispatch (COMMAND PATTERN)
- task_reasoner: Individual reasoner execution
- learning: Learning from reasoning results
- utilities: Query extraction, safety results, symbolic constraints
- shutdown: Metrics, statistics, caching, shutdown
- estimation: Plan time/cost estimation
- verification: Mathematical result verification

Author: VulcanAMI Team
Version: 3.0 (Decomposed)
"""

import hashlib
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import from unified submodules
from .cache import ToolWeightManager, compute_query_hash as _compute_query_hash, get_weight_manager
from .component_loader import (
    _load_reasoning_components,
    _load_selection_components,
    _load_optional_components,
)
from .config import (
    CACHE_HASH_LENGTH,
    CACHE_MAX_AGE_SECONDS,
    CONFIDENCE_FLOOR_NO_RESULT,
    CREATIVE_TASK_KEYWORDS,
    MIN_ENSEMBLE_WEIGHT_FLOOR,
    SELF_REFERENTIAL_MIN_CONFIDENCE,
    UNKNOWN_TYPE_FALLBACK_ORDER,
)
from .types import ReasoningPlan, ReasoningTask
from .strategies import _is_result_not_applicable, topological_sort as _strategies_topological_sort

# Delegation imports
from . import self_ref_detection as _self_ref_detection
from . import self_ref_handlers as _self_ref_handlers
from . import self_ref_conclusion as _self_ref_conclusion
from . import strategy_classification as _strategy_classification
from . import strategy_planning as _strategy_planning
from . import task_execution as _task_execution
from . import task_reasoner as _task_reasoner
from . import learning as _learning
from . import utilities as _utilities
from . import shutdown as _shutdown

# Use existing planning module for plan creation and optimization
from vulcan.planning import (
    Plan,
    PlanStep,
    PlanningMethod,
    HierarchicalGoalSystem,
    ResourceType,
    OperationalMode,
)

# Use existing math verification
from ..mathematical_verification import (
    MathematicalVerificationEngine,
    MathVerificationStatus,
)

# Use existing cost model
from ..selection.cost_model import StochasticCostModel

# Import from parent reasoning module
from ..reasoning_explainer import ReasoningExplainer, SafetyAwareReasoning
from ..reasoning_types import (
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningStrategy,
    ReasoningType,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# MODULE-LEVEL HELPERS (kept for backward compatibility)
# ==============================================================================
from .orchestrator_types import (
    is_test_environment as _is_test_environment,
    is_creative_task as _is_creative_task,
    MATH_EXPRESSION_PATTERN,
    MATH_QUERY_PATTERN,
    MATH_SYMBOLS_PATTERN,
    PROBABILITY_NOTATION_PATTERN,
    INDUCTION_PATTERN,
)


class UnifiedReasoner:
    """Enhanced unified interface with production tool selection and portfolio strategies"""

    # Default tools for ensemble reasoning when no specific tools are selected
    DEFAULT_ENSEMBLE_TOOLS = [
        ReasoningType.PROBABILISTIC,
        ReasoningType.SYMBOLIC,
        ReasoningType.CAUSAL,
    ]

    # CRITICAL FIX #4: Maximum recursion depth for preventing infinite loops
    MAX_RECURSION_DEPTH = 5

    def __init__(
        self,
        enable_learning: bool = True,
        enable_safety: bool = True,
        max_workers: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}

        # NEW: default max_workers from env if not passed
        if max_workers is None:
            try:
                max_workers = int(os.getenv("VULCAN_MAX_WORKERS", "2"))
            except Exception:
                max_workers = 2

        # CRITICAL FIX: Add locks for thread-safe shared state
        self._state_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

        # Load components lazily
        reasoning_components = _load_reasoning_components()
        selection_components = _load_selection_components()
        optional_components = _load_optional_components()

        # Initialize core reasoners with error handling
        self.reasoners = {}
        try:
            if "ProbabilisticReasoner" in reasoning_components:
                self.reasoners[ReasoningType.PROBABILISTIC] = reasoning_components[
                    "ProbabilisticReasoner"
                ]()
            if "SymbolicReasoner" in reasoning_components:
                self.reasoners[ReasoningType.SYMBOLIC] = reasoning_components[
                    "SymbolicReasoner"
                ]()
            if "CausalReasoner" in reasoning_components:
                self.reasoners[ReasoningType.CAUSAL] = reasoning_components[
                    "CausalReasoner"
                ](enable_learning=enable_learning)
            if "AnalogicalReasoningEngine" in reasoning_components:
                self.reasoners[ReasoningType.ANALOGICAL] = reasoning_components[
                    "AnalogicalReasoningEngine"
                ](enable_learning=enable_learning)
            if "AbstractReasoner" in reasoning_components:
                reasoning_components["AbstractReasoner"]
        except Exception as e:
            logger.error(f"Error initializing core reasoners: {e}")

        # Initialize specialized reasoners
        self.counterfactual = None
        self.cross_modal = None
        self.multimodal = None

        try:
            if (
                "CounterfactualReasoner" in reasoning_components
                and ReasoningType.CAUSAL in self.reasoners
            ):
                self.counterfactual = reasoning_components["CounterfactualReasoner"](
                    self.reasoners[ReasoningType.CAUSAL]
                )
            if "CrossModalReasoner" in reasoning_components:
                self.cross_modal = reasoning_components["CrossModalReasoner"]()
            if "MultiModalReasoningEngine" in reasoning_components:
                try:
                    from vulcan.reasoning.singletons import get_multimodal_engine
                    self.multimodal = get_multimodal_engine(enable_learning=enable_learning)
                    if self.multimodal is None:
                        self.multimodal = reasoning_components["MultiModalReasoningEngine"](
                            enable_learning=enable_learning
                        )
                except ImportError:
                    self.multimodal = reasoning_components["MultiModalReasoningEngine"](
                        enable_learning=enable_learning
                    )
                self.reasoners[ReasoningType.MULTIMODAL] = self.multimodal
                self._register_modality_reasoners(
                    reasoning_components.get("ModalityType")
                )
        except Exception as e:
            logger.warning(f"Error initializing specialized reasoners: {e}")

        # Initialize mathematical computation tool
        try:
            from ..mathematical_computation import MathematicalComputationTool

            llm_client = None
            try:
                from vulcan.llm import get_hybrid_executor
                hybrid_executor = get_hybrid_executor()
                if hybrid_executor is not None:
                    llm_client = getattr(hybrid_executor, 'local_llm', None)
                    if llm_client is not None:
                        logger.info("[MathTool] Using GraphixVulcanLLM from hybrid executor")
            except (ImportError, Exception) as e:
                logger.debug(f"[MathTool] Hybrid executor not available: {e}")

            if llm_client is None:
                try:
                    from vulcan.reasoning.singletons import get_llm_client
                    llm_client = get_llm_client()
                    if llm_client is not None:
                        logger.info("[MathTool] Using LLM from singletons")
                except (ImportError, AttributeError):
                    pass

            if llm_client is None:
                try:
                    from vulcan import main
                    if hasattr(main, 'global_llm_client'):
                        llm_client = main.global_llm_client
                        if llm_client is not None:
                            logger.info("[MathTool] Using LLM from main.global_llm_client")
                except (ImportError, AttributeError):
                    pass

            if llm_client is None:
                logger.info("[MathTool] No LLM client found from any source.")

            math_tool = MathematicalComputationTool(
                llm=llm_client,
                enable_learning=enable_learning
            )
            self.reasoners[ReasoningType.MATHEMATICAL] = math_tool
            logger.info(f"[MathTool] Mathematical computation tool registered (llm={'available' if llm_client else 'NONE'})")
        except ImportError as e:
            logger.error(f"[MathTool] Mathematical computation tool import failed: {e}")
        except Exception as e:
            logger.error(f"[MathTool] Error initializing mathematical computation tool: {e}", exc_info=True)

        # Register WorldModel as PHILOSOPHICAL reasoner
        try:
            from vulcan.world_model import WorldModel
            try:
                from vulcan.reasoning.singletons import get_world_model
                world_model = get_world_model()
                if world_model is None:
                    world_model = WorldModel(config={'bootstrap_mode': True, 'enable_meta_reasoning': True})
            except (ImportError, AttributeError):
                world_model = WorldModel(config={'bootstrap_mode': True, 'enable_meta_reasoning': True})

            self.reasoners[ReasoningType.PHILOSOPHICAL] = world_model
            logger.info("[PhilosophicalReasoner] WorldModel registered as PHILOSOPHICAL reasoner")
        except ImportError as e:
            logger.warning(f"[PhilosophicalReasoner] Could not import WorldModel: {e}")
        except Exception as e:
            logger.warning(f"[PhilosophicalReasoner] Error initializing WorldModel: {e}")

        tools_by_name = {k.value: v for k, v in self.reasoners.items()}

        # Cache config
        cache_config = (
            config.get("cache_config", {}).copy() if config.get("cache_config") else {}
        )
        cache_config["cleanup_interval"] = 0.05
        for sub_key in ["feature_cache_config", "selection_cache_config", "result_cache_config"]:
            if sub_key not in cache_config:
                cache_config[sub_key] = {}
            cache_config[sub_key]["cleanup_interval"] = 0.05
        cache_config.setdefault("enable_warming", False)
        cache_config.setdefault("enable_disk_cache", False)

        # Initialize production tool selection system
        self.tool_selector = None
        self.utility_model = None
        self.portfolio_executor = None
        self.cost_model = None
        self.voi_gate = None
        self.safety_governor = None
        self.tool_monitor = None
        self.distribution_monitor = None
        self.cache = None
        self.warm_pool = None
        self.calibrator = None

        try:
            if "ToolSelector" in selection_components:
                self.tool_selector = selection_components["ToolSelector"](
                    config.get("tool_selector_config", {})
                )
                self._daemonize_component_threads(self.tool_selector)

            if "UtilityModel" in selection_components:
                self.utility_model = selection_components["UtilityModel"](
                    config.get("utility_config", {})
                )
            if "PortfolioExecutor" in selection_components:
                self.portfolio_executor = selection_components["PortfolioExecutor"](
                    tools=tools_by_name, max_workers=max_workers
                )
                self._daemonize_component_threads(self.portfolio_executor)

            if "StochasticCostModel" in selection_components:
                self.cost_model = selection_components["StochasticCostModel"](
                    config.get("cost_config", {})
                )

            if "SelectionCache" in selection_components:
                self.cache = selection_components["SelectionCache"](cache_config)
                self._daemonize_component_threads(self.cache)

            if "WarmStartPool" in selection_components:
                warm_pool_config = config.get("warm_pool_config", {}).copy()
                if "cleanup_interval" not in warm_pool_config:
                    warm_pool_config["cleanup_interval"] = 0.05
                self.warm_pool = selection_components["WarmStartPool"](
                    tools=tools_by_name, config=warm_pool_config
                )
                self._daemonize_component_threads(self.warm_pool)

            if (
                "CalibratedDecisionMaker" in selection_components
                and selection_components["CalibratedDecisionMaker"] is not None
            ):
                self.calibrator = selection_components["CalibratedDecisionMaker"]()
        except Exception as e:
            logger.warning(f"Error initializing selection components: {e}")

        # Initialize safety components
        self.enable_safety = enable_safety
        self.safety_wrapper = None

        if enable_safety:
            try:
                if "SafetyGovernor" in selection_components:
                    self.safety_governor = selection_components["SafetyGovernor"](
                        config.get("safety_config", {})
                    )
                    self._daemonize_component_threads(self.safety_governor)

                if "SafetyValidator" in optional_components:
                    self.safety_wrapper = SafetyAwareReasoning(
                        optional_components["SafetyValidator"]()
                    )
                else:
                    self.safety_wrapper = SafetyAwareReasoning(None)
            except Exception as e:
                logger.warning(f"Error initializing safety components: {e}")
                self.safety_wrapper = SafetyAwareReasoning(None)
        else:
            self.safety_wrapper = SafetyAwareReasoning(None)

        # Explainability
        self.explainer = ReasoningExplainer()

        # Learning component
        self.enable_learning = enable_learning
        self.learner = None

        if enable_learning and "ContinualLearner" in optional_components:
            try:
                self.learner = optional_components["ContinualLearner"]()
            except Exception as e:
                logger.warning(f"Error initializing learner: {e}")

        # Runtime integration
        self.runtime = None
        if "UnifiedRuntime" in optional_components:
            in_test = _is_test_environment()
            skip_via_config = config.get("skip_runtime", False)

            if not in_test and not skip_via_config:
                try:
                    from vulcan.reasoning.singletons import get_or_create_unified_runtime
                    self.runtime = get_or_create_unified_runtime()
                    if self.runtime:
                        self._daemonize_component_threads(self.runtime)
                        logger.info("UnifiedRuntime obtained from singleton (PRODUCTION mode)")
                except ImportError:
                    self.runtime = optional_components["UnifiedRuntime"]()
                    self._daemonize_component_threads(self.runtime)
                except Exception as e:
                    logger.warning(f"Error initializing runtime: {e}")
                    self.runtime = None

        # Processor for multimodal inputs
        self.processor = None
        if "MultimodalProcessor" in optional_components:
            try:
                self.processor = optional_components["MultimodalProcessor"]()
            except Exception as e:
                logger.warning(f"Error initializing processor: {e}")

        # Mathematical Verification Engine
        self.math_verification_engine = None
        self._math_accuracy_integration = None
        if "MathematicalVerificationEngine" in optional_components:
            try:
                from vulcan.reasoning.singletons import get_math_verification_engine
                self.math_verification_engine = get_math_verification_engine()
                if self.math_verification_engine is None:
                    self.math_verification_engine = optional_components["MathematicalVerificationEngine"]()

                try:
                    from vulcan.learning.mathematical_accuracy_integration import (
                        MathematicalAccuracyIntegration,
                    )
                    self._math_accuracy_integration = MathematicalAccuracyIntegration(
                        math_engine=self.math_verification_engine
                    )
                except ImportError:
                    pass
            except Exception as e:
                logger.warning(f"Error initializing mathematical verification engine: {e}")

        # Store selection components for later use
        self._selection_components = selection_components
        self._optional_components = optional_components

        # Reasoning orchestration strategies
        self.reasoning_strategies = {
            ReasoningStrategy.SEQUENTIAL: self._sequential_reasoning,
            ReasoningStrategy.PARALLEL: self._parallel_reasoning,
            ReasoningStrategy.ENSEMBLE: self._ensemble_reasoning,
            ReasoningStrategy.HIERARCHICAL: self._hierarchical_reasoning,
            ReasoningStrategy.ADAPTIVE: self._adaptive_reasoning,
            ReasoningStrategy.HYBRID: self._hybrid_reasoning,
            ReasoningStrategy.PORTFOLIO: self._portfolio_reasoning,
            ReasoningStrategy.UTILITY_BASED: self._utility_based_reasoning,
        }

        # Task management
        self.task_queue = deque(maxlen=10000)
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)

        # Performance tracking
        self.reasoning_history = deque(maxlen=1000)
        self.audit_trail = deque(maxlen=5000)
        self.performance_metrics = {
            "total_reasonings": 0,
            "successful_reasonings": 0,
            "average_confidence": 0.0,
            "average_time": 0.0,
            "average_utility": 0.0,
            "type_usage": defaultdict(int),
            "strategy_usage": defaultdict(int),
            "tool_selection_stats": defaultdict(int),
        }

        # Caching
        self.result_cache = {}
        self.plan_cache = {}
        self.max_cache_size = 1000

        # Parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

        if hasattr(self.executor, "_threads"):
            for thread in self.executor._threads:
                try:
                    thread.daemon = True
                except Exception:
                    pass

        # Configuration
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.max_reasoning_time = config.get("max_reasoning_time", 30.0)
        self.default_timeout = config.get("default_timeout", 30.0)

        default_mode = config.get("selection_mode", "BALANCED").upper()
        self.default_selection_mode = None
        if "SelectionMode" in selection_components:
            try:
                self.default_selection_mode = selection_components["SelectionMode"][default_mode]
            except Exception:
                self.default_selection_mode = None

        # Model persistence
        self.model_path = Path("unified_models")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Execution counter
        self.execution_count = 0

        self._clear_invalid_cache_entries()

        logger.info(
            "Enhanced Unified Reasoner initialized with production tool selection"
        )

    # ==========================================================================
    # CORE INFRASTRUCTURE (kept on class)
    # ==========================================================================

    def _daemonize_component_threads(self, component):
        """Make all threads in a component daemon threads immediately"""
        if not component:
            return
        try:
            thread_attrs = [
                "monitor_thread", "scaling_thread", "health_check_thread",
                "cleanup_thread", "_monitor_thread", "_cleanup_thread",
                "_health_thread", "_scaling_thread", "watchdog_thread",
                "_watchdog_thread", "_update_thread", "_warm_thread",
                "_stats_thread", "_process_thread", "_background_thread",
                "_warm_cache_thread", "_statistics_thread",
            ]
            for attr_name in thread_attrs:
                thread = getattr(component, attr_name, None)
                if thread and isinstance(thread, threading.Thread):
                    try:
                        thread.daemon = True
                    except Exception:
                        pass
            if hasattr(component, "executor") and component.executor:
                if hasattr(component.executor, "_threads"):
                    for thread in component.executor._threads:
                        try:
                            thread.daemon = True
                        except Exception:
                            pass
        except Exception:
            pass

    def _register_modality_reasoners(self, ModalityType):
        """Register reasoners for different modalities"""
        if not self.multimodal or not ModalityType:
            return
        try:
            if ReasoningType.SYMBOLIC in self.reasoners:
                self.multimodal.register_modality_reasoner(
                    ModalityType.TEXT, self.reasoners[ReasoningType.SYMBOLIC]
                )
                self.multimodal.register_modality_reasoner(
                    ModalityType.CODE, self.reasoners[ReasoningType.SYMBOLIC]
                )
            if ReasoningType.PROBABILISTIC in self.reasoners:
                self.multimodal.register_modality_reasoner(
                    ModalityType.UNKNOWN, self.reasoners[ReasoningType.PROBABILISTIC]
                )
            if ReasoningType.SYMBOLIC in self.reasoners:
                self.multimodal.register_modality_reasoner(
                    ModalityType.TEXT, self.reasoners[ReasoningType.SYMBOLIC]
                )
        except Exception as e:
            logger.warning(f"Error registering modality reasoners: {e}")

    # ==========================================================================
    # CACHE VALIDATION (kept on class - tightly coupled to instance state)
    # ==========================================================================

    def _clear_invalid_cache_entries(self):
        """Clear invalid cache entries on startup."""
        if not hasattr(self, 'result_cache'):
            return
        with self._cache_lock:
            keys_to_remove = []
            for key, result in list(self.result_cache.items()):
                if self._is_invalid_cache_entry(result):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.result_cache[key]
            if keys_to_remove:
                logger.info(f"[Cache] Cleared {len(keys_to_remove)} invalid cache entries on startup")

    def _is_invalid_cache_entry(self, result: ReasoningResult) -> bool:
        """Check if a cached result should be considered invalid."""
        if result is None:
            return True
        if hasattr(result, 'reasoning_type') and result.reasoning_type == ReasoningType.UNKNOWN:
            return True
        if hasattr(result, 'confidence') and result.confidence < 0.15:
            return True
        if isinstance(result.conclusion, dict) and result.conclusion.get('error'):
            return True
        return False

    def _is_valid_cached_result(
        self, cached_result: ReasoningResult, task: ReasoningTask
    ) -> Tuple[bool, str]:
        """Validate a cached result before returning it."""
        if cached_result is None:
            return False, "Cached result is None"
        if task is None:
            return False, "Task is None"

        if hasattr(cached_result, 'reasoning_type'):
            try:
                if cached_result.reasoning_type == ReasoningType.UNKNOWN:
                    return False, "Cached result has UNKNOWN reasoning type"
            except (AttributeError, TypeError):
                return False, "Invalid reasoning_type attribute"
        else:
            return False, "Missing reasoning_type attribute"

        if hasattr(cached_result, 'confidence'):
            try:
                confidence_value = float(cached_result.confidence)
                if confidence_value < CONFIDENCE_FLOOR_NO_RESULT:
                    return False, f"Cached confidence {confidence_value:.2f} < minimum floor"
                if not (0.0 <= confidence_value <= 1.0):
                    return False, f"Invalid confidence value: {confidence_value}"
            except (ValueError, TypeError):
                return False, "Invalid confidence value type"
        else:
            return False, "Missing confidence attribute"

        if hasattr(cached_result, 'metadata') and isinstance(cached_result.metadata, dict):
            try:
                cached_time = cached_result.metadata.get('cache_timestamp', 0)
                if cached_time > 0:
                    cache_age = time.time() - cached_time
                    if cache_age < -10:
                        return False, "Cache timestamp is in the future"
                    if cache_age > CACHE_MAX_AGE_SECONDS:
                        return False, f"Cache expired: age={cache_age:.1f}s"
            except (ValueError, TypeError):
                pass

            try:
                cached_query_hash = cached_result.metadata.get('original_query_hash')
                if cached_query_hash:
                    current_query_hash = _compute_query_hash(task.query)
                    if cached_query_hash != current_query_hash:
                        return False, "Query hash mismatch: cache collision detected"
            except Exception as e:
                return False, f"Query hash verification failed: {e}"

        return True, ""

    # ==========================================================================
    # SELF-REFERENTIAL QUERY METHODS (delegation stubs)
    # ==========================================================================

    def _is_self_referential_query(self, query) -> bool:
        return _self_ref_detection.is_self_referential_query(query)

    def _handle_self_referential_query(self, task, reasoning_chain, _recursion_depth=0):
        return _self_ref_handlers.handle_self_referential_query(
            self, task, reasoning_chain, _recursion_depth
        )

    def _build_self_referential_conclusion(self, query_str, analysis):
        return _self_ref_conclusion.build_self_referential_conclusion(
            self, query_str, analysis
        )

    def _is_binary_choice_question(self, query_lower):
        return _self_ref_detection.is_binary_choice_question(query_lower)

    def _get_world_model_philosophical_analysis(self, query_str):
        return _self_ref_conclusion._get_world_model_philosophical_analysis(query_str)

    def _generate_self_awareness_decision(self, query_str, objectives, conflicts, ethical_check, counterfactual):
        return _self_ref_conclusion._generate_self_awareness_decision(
            query_str, objectives, conflicts, ethical_check, counterfactual
        )

    def _generate_self_awareness_reflection(self, query_str, objectives, ethical_check, philosophical_analysis):
        return _self_ref_conclusion._generate_self_awareness_reflection(
            query_str, objectives, ethical_check, philosophical_analysis
        )

    def _generate_general_self_referential_response(self, query_str, objectives, philosophical_analysis):
        return _self_ref_conclusion._generate_general_self_referential_response(
            query_str, objectives, philosophical_analysis
        )

    def _generate_ethically_constrained_response(self, query_str, ethical_check):
        return _self_ref_conclusion._generate_ethically_constrained_response(
            query_str, ethical_check
        )

    def _create_self_referential_fallback(self, task, reasoning_chain):
        return _self_ref_handlers.create_self_referential_fallback(
            self, task, reasoning_chain
        )

    # ==========================================================================
    # MAIN REASON METHOD (kept on class - core orchestration logic)
    # ==========================================================================

    def reason(
        self,
        input_data: Any,
        query: Optional[Dict[str, Any]] = None,
        reasoning_type: Optional[ReasoningType] = None,
        strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
        confidence_threshold: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
        pre_selected_tools: Optional[List[str]] = None,
        skip_tool_selection: bool = False,
        _recursion_depth: int = 0,
        _source: str = "external",
    ) -> ReasoningResult:
        """Enhanced reasoning interface with production tool selection."""

        if _recursion_depth >= self.MAX_RECURSION_DEPTH:
            logger.error(
                f"[Recursion Guard] Maximum recursion depth ({self.MAX_RECURSION_DEPTH}) exceeded."
            )
            return self._create_error_result(
                f"Maximum reasoning recursion depth ({self.MAX_RECURSION_DEPTH}) exceeded."
            )

        if _recursion_depth > 0:
            logger.warning(
                f"[Recursion Guard] Recursive reasoning call at depth {_recursion_depth} (source={_source})"
            )

        with self._shutdown_lock:
            if self._is_shutdown:
                return self._create_error_result("System is shutdown")

        start_time = time.time()

        with self._state_lock:
            self.execution_count += 1
            self.performance_metrics["total_reasonings"] += 1

        try:
            initial_step = ReasoningStep(
                step_id=f"initial_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningType.UNKNOWN,
                input_data=input_data,
                output_data=None,
                confidence=1.0,
                explanation="Reasoning process initialized",
            )

            reasoning_chain = ReasoningChain(
                chain_id=str(uuid.uuid4()),
                steps=[initial_step],
                initial_query=query,
                final_conclusion=None,
                total_confidence=0.0,
                reasoning_types_used=set(),
                modalities_involved=set(),
                safety_checks=[],
                audit_trail=[],
            )

            if confidence_threshold is None:
                confidence_threshold = self.confidence_threshold

            if constraints is None:
                constraints = {
                    "time_budget_ms": self.max_reasoning_time * 1000,
                    "confidence_threshold": confidence_threshold,
                }

            utility_context = self._create_utility_context(query, constraints)

            task = ReasoningTask(
                task_id=str(uuid.uuid4()),
                task_type=reasoning_type or ReasoningType.UNKNOWN,
                input_data=input_data,
                query=query or {},
                constraints=constraints,
                utility_context=utility_context,
            )

            cache_key = self._compute_cache_key(task)
            with self._cache_lock:
                if cache_key in self.result_cache:
                    cached_result = self.result_cache[cache_key]
                    cache_valid, validation_reason = self._is_valid_cached_result(
                        cached_result, task
                    )
                    if cache_valid:
                        logger.info(f"[Cache] Valid cache hit for task {task.task_id}")
                        self._record_execution(task, cached_result, time.time() - start_time, True)
                        return cached_result
                    else:
                        logger.warning(f"[Cache] Invalid cache entry removed: {validation_reason}")
                        del self.result_cache[cache_key]

            # Self-referential query check
            if _source == "external" and self._is_self_referential_query(query):
                logger.info("[SelfRef] Self-referential query detected, routing to meta-reasoning")
                result = self._handle_self_referential_query(task, reasoning_chain, _recursion_depth)
                with self._cache_lock:
                    if result and hasattr(result, 'metadata'):
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata['cache_timestamp'] = time.time()
                        result.metadata['original_query_hash'] = _compute_query_hash(task.query)
                        result.metadata['cached_task_type'] = task.task_type.value if isinstance(task.task_type, ReasoningType) else str(task.task_type)
                    self.result_cache[cache_key] = result
                elapsed_time = time.time() - start_time
                self._update_metrics(result, elapsed_time, strategy)
                self._record_execution(task, result, elapsed_time, False)
                self._add_to_history(task, result, elapsed_time)
                self._add_audit_entry(task, result, strategy, elapsed_time)
                return result
            elif _source == "internal" and self._is_self_referential_query(query):
                logger.warning("[Recursion Guard] Blocked internal self-referential query")
                return self._create_error_result("Internal self-referential query blocked")

            if self.enable_safety and self.safety_wrapper:
                try:
                    safe_input = self.safety_wrapper.validate_input(input_data)
                    if not safe_input["is_safe"]:
                        return self._create_safety_result(safe_input["reason"])
                    task.input_data = safe_input["sanitized_input"]
                except Exception as e:
                    logger.warning(f"Safety validation failed: {e}")

            if self.distribution_monitor and task.features is not None:
                try:
                    if self.distribution_monitor.detect_shift(task.features):
                        strategy = ReasoningStrategy.ADAPTIVE
                except Exception:
                    pass

            if reasoning_type is None:
                reasoning_type = self._determine_reasoning_type(input_data, query)
                task.task_type = reasoning_type

            # Extract router hints
            router_hints = None
            if query and isinstance(query, dict):
                router_hints = (
                    query.get('tool_hints')
                    or query.get('parameters', {}).get('tool_hints')
                    or constraints.get('tool_hints')
                )
                if not router_hints:
                    selected_tools_legacy = (
                        query.get('selected_tools')
                        or query.get('parameters', {}).get('selected_tools')
                        or constraints.get('selected_tools')
                    )
                    if selected_tools_legacy and isinstance(selected_tools_legacy, list):
                        router_hints = {tool: 0.8 for tool in selected_tools_legacy}

            if self.voi_gate and task.features is not None:
                try:
                    should_gather, voi_action = self.voi_gate.should_probe_deeper(
                        task.features, None, constraints
                    )
                    if should_gather:
                        task = self._enhance_task_with_voi(task, voi_action)
                except Exception:
                    pass

            # Plan creation and tool selection
            if skip_tool_selection and pre_selected_tools:
                plan = self._create_optimized_plan(
                    task, strategy, router_hints,
                    pre_selected_tools=pre_selected_tools,
                    skip_tool_selection=True,
                )
            else:
                plan = self._create_optimized_plan(task, strategy, router_hints)
                if self.tool_selector and strategy in [
                    ReasoningStrategy.PORTFOLIO,
                    ReasoningStrategy.UTILITY_BASED,
                    ReasoningStrategy.ENSEMBLE,
                ]:
                    try:
                        selection_result = self._select_tools_for_plan(plan, task)
                        if selection_result and hasattr(selection_result, "selected_tool"):
                            plan.selected_tools = [selection_result.selected_tool]
                        if hasattr(selection_result, "strategy_used"):
                            plan.execution_strategy = selection_result.strategy_used
                    except Exception:
                        pass

            strategy_func = self.reasoning_strategies.get(strategy, self._adaptive_reasoning)
            result = self._execute_strategy_safe(
                strategy_func, plan, reasoning_chain, timeout=self.default_timeout
            )

            if result is None:
                result = self._create_error_result("Strategy execution failed or timed out")

            if self.calibrator and result.reasoning_type:
                try:
                    calibrator_key = str(result.reasoning_type)
                    if (
                        hasattr(self.calibrator, "calibrators")
                        and calibrator_key in self.calibrator.calibrators
                    ):
                        result.confidence = self.calibrator.calibrate_confidence(
                            calibrator_key, result.confidence, task.features
                        )
                except Exception:
                    pass

            result = self._postprocess_result(result, task)

            if self.enable_safety and self.safety_wrapper:
                try:
                    is_creative = _is_creative_task(task)
                    query_str = self._extract_query_string(task.query)
                    safe_output = self.safety_wrapper.validate_output(
                        result, is_creative=is_creative, query=query_str
                    )
                    if not safe_output["is_safe"]:
                        result = self._create_safety_result(f"Output filtered: {safe_output['reason']}")
                except Exception:
                    pass

            if self.enable_learning and self.learner:
                try:
                    self._learn_from_reasoning(task, result)
                except Exception:
                    pass

            elapsed_time = time.time() - start_time
            self._update_metrics(result, elapsed_time, strategy)
            self._record_execution(task, result, elapsed_time, False)

            with self._cache_lock:
                if len(self.result_cache) >= self.max_cache_size:
                    keys_to_remove = list(self.result_cache.keys())[:self.max_cache_size // 5]
                    for key in keys_to_remove:
                        del self.result_cache[key]

                should_cache = True
                if result:
                    if hasattr(result, 'reasoning_type') and result.reasoning_type == ReasoningType.UNKNOWN:
                        should_cache = False
                    elif hasattr(result, 'confidence') and result.confidence < 0.15:
                        should_cache = False
                    elif isinstance(result.conclusion, dict) and result.conclusion.get('error'):
                        should_cache = False

                if should_cache and result:
                    if hasattr(result, 'metadata'):
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata['cache_timestamp'] = time.time()
                        result.metadata['original_query_hash'] = _compute_query_hash(task.query)
                        result.metadata['cached_task_type'] = task.task_type.value if isinstance(task.task_type, ReasoningType) else str(task.task_type)
                    self.result_cache[cache_key] = result

            self._add_to_history(task, result, elapsed_time)
            self._add_audit_entry(task, result, strategy, elapsed_time)

            if result and not result.reasoning_chain:
                result.reasoning_chain = reasoning_chain

            return result

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return self._create_error_result(str(e))

    # ==========================================================================
    # STRATEGY EXECUTION (kept - thin wrappers around executor)
    # ==========================================================================

    def _execute_strategy_safe(self, strategy_func, plan, reasoning_chain, timeout=30.0):
        """Execute strategy with proper resource management and timeout"""
        future = None
        try:
            future = self.executor.submit(
                self._execute_strategy_impl, strategy_func, plan, reasoning_chain
            )
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error(f"Strategy execution timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return None
        finally:
            if future is not None and not future.done():
                future.cancel()

    def _execute_strategy_impl(self, strategy_func, plan, reasoning_chain):
        """Internal strategy execution implementation"""
        try:
            return strategy_func(plan, reasoning_chain)
        except Exception as e:
            logger.error(f"Strategy function failed: {e}")
            return self._create_error_result(str(e))

    def _update_statistics_safe(self, result: ReasoningResult):
        """Update statistics thread-safely"""
        if result and hasattr(result, "reasoning_type") and result.reasoning_type:
            reasoning_stats = defaultdict(
                lambda: {"count": 0, "successes": 0, "avg_confidence": 0.0}
            )
            stats = reasoning_stats[result.reasoning_type]
            stats["count"] += 1
            if result.confidence >= 0.5:
                stats["successes"] += 1
            alpha = 0.1
            stats["avg_confidence"] = (1 - alpha) * stats["avg_confidence"] + alpha * result.confidence

    # ==========================================================================
    # PLAN/TOOL HELPERS (delegation stubs)
    # ==========================================================================

    def _create_utility_context(self, query, constraints):
        """Create utility context from query and constraints"""
        if "UtilityContext" not in self._selection_components:
            return None
        try:
            UtilityContext = self._selection_components["UtilityContext"]
            ContextMode = self._selection_components["ContextMode"]
            mode = ContextMode.BALANCED
            if query:
                query_str = str(query).lower()
                if "fast" in query_str or "quick" in query_str:
                    mode = ContextMode.RUSH
                elif "accurate" in query_str or "precise" in query_str:
                    mode = ContextMode.ACCURATE
                elif "efficient" in query_str:
                    mode = ContextMode.EFFICIENT
            return UtilityContext(
                mode=mode,
                time_budget=constraints.get("time_budget_ms", 5000),
                energy_budget=constraints.get("energy_budget_mj", 1000),
                min_quality=constraints.get("min_quality", 0.5),
                max_risk=constraints.get("max_risk", 0.3),
                user_preferences=constraints.get("preferences", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to create utility context: {e}")
            return None

    def _select_tools_for_plan(self, plan, task):
        """Select tools using production tool selector"""
        if not self.tool_selector or "SelectionRequest" not in self._selection_components:
            return None
        try:
            SelectionRequest = self._selection_components["SelectionRequest"]
            selection_request = SelectionRequest(
                problem=task.input_data,
                features=task.features,
                constraints={
                    "time_budget_ms": task.constraints.get("time_budget_ms", 5000),
                    "energy_budget_mj": task.constraints.get("energy_budget_mj", 1000),
                    "min_confidence": task.constraints.get("confidence_threshold", 0.5),
                },
                mode=self._map_strategy_to_mode(plan.strategy),
                context=task.query,
            )
            return self.tool_selector.select_and_execute(selection_request)
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return None

    def _map_strategy_to_mode(self, strategy):
        """Map reasoning strategy to selection mode"""
        if "SelectionMode" not in self._selection_components:
            return None
        try:
            SelectionMode = self._selection_components["SelectionMode"]
            mapping = {
                ReasoningStrategy.SEQUENTIAL: SelectionMode.FAST,
                ReasoningStrategy.PARALLEL: SelectionMode.FAST,
                ReasoningStrategy.ENSEMBLE: SelectionMode.ACCURATE,
                ReasoningStrategy.HIERARCHICAL: SelectionMode.BALANCED,
                ReasoningStrategy.ADAPTIVE: SelectionMode.BALANCED,
                ReasoningStrategy.HYBRID: SelectionMode.BALANCED,
                ReasoningStrategy.PORTFOLIO: SelectionMode.BALANCED,
                ReasoningStrategy.UTILITY_BASED: SelectionMode.EFFICIENT,
            }
            return mapping.get(strategy, SelectionMode.BALANCED)
        except Exception:
            return None

    def _create_optimized_plan(self, task, strategy, router_hints=None, pre_selected_tools=None, skip_tool_selection=False):
        return _strategy_planning.create_optimized_plan(
            self, task, strategy, router_hints, pre_selected_tools, skip_tool_selection
        )

    def _map_tool_name_to_reasoning_type(self, tool_name):
        return _strategy_classification.map_tool_name_to_reasoning_type(tool_name)

    def _select_portfolio_reasoners(self, task):
        return _strategy_classification.select_portfolio_reasoners(self, task)

    def _classify_reasoning_task(self, input_data, query):
        return _strategy_classification.classify_reasoning_task(input_data, query)

    def _determine_reasoning_type(self, input_data, query):
        return _strategy_classification.determine_reasoning_type(self, input_data, query)

    # ==========================================================================
    # STRATEGY METHODS (delegation to strategies.py - already extracted)
    # ==========================================================================

    def _portfolio_reasoning(self, plan, reasoning_chain):
        """Execute reasoning using portfolio strategy"""
        if not self.portfolio_executor:
            logger.warning("Portfolio executor not available, falling back to ensemble")
            return self._ensemble_reasoning(plan, reasoning_chain)
        try:
            if not plan.selected_tools:
                plan.selected_tools = [task.task_type.value for task in plan.tasks]
            ExecutionStrategy = self._selection_components.get("ExecutionStrategy")
            exec_strategy = None
            if ExecutionStrategy:
                exec_strategy = plan.execution_strategy or ExecutionStrategy.SEQUENTIAL_REFINEMENT
            ExecutionMonitor = self._selection_components.get("ExecutionMonitor")
            monitor = None
            if ExecutionMonitor:
                monitor = ExecutionMonitor(
                    time_budget_ms=plan.tasks[0].constraints.get("time_budget_ms", 5000),
                    energy_budget_mj=plan.tasks[0].constraints.get("energy_budget_mj", 1000),
                    min_confidence=plan.confidence_threshold,
                )
            exec_result = self.portfolio_executor.execute(
                strategy=exec_strategy,
                tool_names=plan.selected_tools,
                problem=plan.tasks[0].input_data,
                constraints=plan.tasks[0].constraints,
                monitor=monitor,
            )
            if exec_result and hasattr(exec_result, "primary_result") and exec_result.primary_result:
                result = self._convert_execution_to_reasoning_result(exec_result)
                if result:
                    result.reasoning_chain = reasoning_chain
                    return result
        except Exception as e:
            logger.error(f"Portfolio reasoning failed: {e}")
        return self._create_empty_result()

    def _utility_based_reasoning(self, plan, reasoning_chain):
        """Execute reasoning optimized for utility"""
        try:
            if len(plan.tasks) == 1:
                result = self._execute_task(plan.tasks[0])
                result.reasoning_chain = reasoning_chain
                return result
            else:
                return self._ensemble_reasoning(plan, reasoning_chain)
        except Exception as e:
            logger.error(f"Utility-based reasoning failed: {e}")
            return self._create_error_result(str(e))

    def _sequential_reasoning(self, plan, reasoning_chain):
        """Execute reasoning tasks sequentially"""
        results = []
        for task in plan.tasks:
            try:
                if task.task_type in self.reasoners:
                    result = self._execute_reasoner(self.reasoners[task.task_type], task)
                    results.append(result)
                    if hasattr(result, "reasoning_chain") and result.reasoning_chain and result.reasoning_chain.steps:
                        for step in result.reasoning_chain.steps:
                            if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                                reasoning_chain.steps.append(step)
            except Exception as e:
                logger.error(f"Sequential task execution failed: {e}")

        if results:
            final_result = max(results, key=lambda r: getattr(r, 'confidence', 0))
            reasoning_chain.final_conclusion = final_result.conclusion
            reasoning_chain.total_confidence = np.mean([r.confidence for r in results])
            reasoning_chain.reasoning_types_used.update({r.reasoning_type for r in results if r.reasoning_type})
            return ReasoningResult(
                conclusion=final_result.conclusion,
                confidence=final_result.confidence,
                reasoning_type=final_result.reasoning_type,
                reasoning_chain=reasoning_chain,
                explanation=final_result.explanation,
            )
        return self._create_empty_result()

    def _parallel_reasoning(self, plan, reasoning_chain):
        """Execute reasoning tasks in parallel"""
        futures = []
        for task in plan.tasks:
            if task.task_type in self.reasoners:
                try:
                    future = self.executor.submit(self._execute_task, task)
                    futures.append((task, future))
                except Exception as e:
                    logger.error(f"Failed to submit parallel task: {e}")

        results = []
        for task, future in futures:
            try:
                result = future.result(timeout=self.max_reasoning_time)
                results.append(result)
                if hasattr(result, "reasoning_chain") and result.reasoning_chain and result.reasoning_chain.steps:
                    for step in result.reasoning_chain.steps:
                        if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                            reasoning_chain.steps.append(step)
            except TimeoutError:
                future.cancel()
            except Exception:
                if not future.done():
                    future.cancel()

        if results:
            conclusion = self._combine_parallel_results(results)
            confidence = np.mean([r.confidence for r in results])
            reasoning_chain.final_conclusion = conclusion
            reasoning_chain.total_confidence = confidence
            reasoning_chain.reasoning_types_used.update({r.reasoning_type for r in results if r.reasoning_type})
            return ReasoningResult(
                conclusion=conclusion,
                confidence=confidence,
                reasoning_type=ReasoningType.HYBRID,
                reasoning_chain=reasoning_chain,
                explanation=f"Parallel reasoning with {len(results)} tasks",
            )
        return self._create_empty_result()

    def _ensemble_reasoning(self, plan, reasoning_chain):
        """Ensemble reasoning with voting"""
        results = []
        for task in plan.tasks:
            try:
                if task.task_type in self.reasoners:
                    result = self._execute_task(task)
                    results.append((task.task_type, result))
                    if hasattr(result, "reasoning_chain") and result.reasoning_chain and result.reasoning_chain.steps:
                        for step in result.reasoning_chain.steps:
                            if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                                reasoning_chain.steps.append(step)
            except Exception as e:
                logger.warning(f"Ensemble task failed: {e}")

        if not results:
            return self._create_empty_result()

        applicable_results = []
        skipped_results = []
        for reasoning_type, result in results:
            if _is_result_not_applicable(result):
                skipped_results.append((reasoning_type, result))
            else:
                applicable_results.append((reasoning_type, result))

        if not applicable_results:
            applicable_results = results

        conclusions = []
        weights = []
        for reasoning_type, result in applicable_results:
            conclusions.append(result.conclusion)
            base_weight = result.confidence
            type_weight = self._get_reasoning_type_weight(reasoning_type)
            if plan.tasks and plan.tasks[0].utility_context:
                execution_time_ms = getattr(result, "metadata", {}).get("execution_time_ms", 100)
                utility_weight = self._calculate_result_utility(result, plan.tasks[0].utility_context, execution_time_ms)
                raw_weight = base_weight * type_weight * utility_weight
            else:
                raw_weight = base_weight * type_weight
            weights.append(max(MIN_ENSEMBLE_WEIGHT_FLOOR, raw_weight))

        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 / len(applicable_results)] * len(applicable_results) if applicable_results else [1.0]

        ensemble_conclusion = self._weighted_voting(conclusions, weights)
        ensemble_confidence = (
            np.average([r[1].confidence for r in applicable_results], weights=list(weights))
            if weights and sum(weights) > 0 and len(weights) == len(applicable_results)
            else 0.5
        )

        ensemble_step = ReasoningStep(
            "ensemble_step", ReasoningType.ENSEMBLE,
            plan.tasks[0].query if plan.tasks else {},
            ensemble_conclusion, ensemble_confidence, "Ensemble reasoning",
        )
        reasoning_chain.steps.append(ensemble_step)
        reasoning_chain.final_conclusion = ensemble_conclusion
        reasoning_chain.total_confidence = ensemble_confidence
        reasoning_chain.reasoning_types_used.update({r[0] for r in results})

        return ReasoningResult(
            conclusion=ensemble_conclusion,
            confidence=ensemble_confidence,
            reasoning_type=ReasoningType.ENSEMBLE,
            reasoning_chain=reasoning_chain,
            explanation=f"Ensemble of {len(applicable_results)} applicable reasoners with weighted voting",
        )

    def _hierarchical_reasoning(self, plan, reasoning_chain):
        """Hierarchical reasoning with dependencies"""
        completed = {}
        try:
            sorted_tasks = self._topological_sort(plan.tasks, plan.dependencies)
            for task in sorted_tasks:
                deps = plan.dependencies.get(task.task_id, [])
                dep_results = [completed[dep_id] for dep_id in deps if dep_id in completed]
                if dep_results:
                    task.input_data = self._merge_dependency_results(task.input_data, dep_results)
                result = self._execute_task(task)
                completed[task.task_id] = result
                if hasattr(result, "reasoning_chain") and result.reasoning_chain and result.reasoning_chain.steps:
                    for step in result.reasoning_chain.steps:
                        if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                            reasoning_chain.steps.append(step)
            if completed and sorted_tasks:
                final_result = completed[sorted_tasks[-1].task_id]
                reasoning_chain.final_conclusion = final_result.conclusion
                reasoning_chain.total_confidence = final_result.confidence
                return ReasoningResult(
                    conclusion=final_result.conclusion,
                    confidence=final_result.confidence,
                    reasoning_type=final_result.reasoning_type,
                    reasoning_chain=reasoning_chain,
                    explanation=final_result.explanation,
                )
        except Exception as e:
            logger.error(f"Hierarchical reasoning failed: {e}")
        return self._create_empty_result()

    def _adaptive_reasoning(self, plan, reasoning_chain):
        """Adaptive strategy selection based on input characteristics"""
        try:
            characteristics = self._analyze_input_characteristics(plan.tasks[0])
            reasoning_chain.steps.append(
                ReasoningStep(
                    step_id=f"adaptive_analysis_{uuid.uuid4().hex[:8]}",
                    step_type=ReasoningType.UNKNOWN,
                    input_data=plan.tasks[0].input_data,
                    output_data=characteristics,
                    confidence=1.0,
                    explanation=f"Analyzed input characteristics: {characteristics}",
                )
            )
            if characteristics["complexity"] > 0.8:
                if plan.tasks[0].utility_context and hasattr(plan.tasks[0].utility_context, "mode"):
                    selection_components = _load_selection_components()
                    ContextMode = selection_components.get("ContextMode")
                    if ContextMode and plan.tasks[0].utility_context.mode == ContextMode.ACCURATE:
                        return self._ensemble_reasoning(plan, reasoning_chain)
                return self._portfolio_reasoning(plan, reasoning_chain)
            elif characteristics["uncertainty"] > 0.7:
                adaptive_plan = self._create_adaptive_plan(
                    plan.tasks[0], [ReasoningType.PROBABILISTIC, ReasoningType.CAUSAL]
                )
                return self._ensemble_reasoning(adaptive_plan, reasoning_chain)
            elif characteristics["multimodal"]:
                if ReasoningType.MULTIMODAL in self.reasoners:
                    multimodal_result = self.reason_multimodal(plan.tasks[0].input_data, plan.tasks[0].query)
                    if multimodal_result.reasoning_chain and multimodal_result.reasoning_chain.steps:
                        for step in multimodal_result.reasoning_chain.steps:
                            if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                                reasoning_chain.steps.append(step)
                    reasoning_chain.final_conclusion = multimodal_result.conclusion
                    reasoning_chain.total_confidence = multimodal_result.confidence
                    return ReasoningResult(
                        conclusion=multimodal_result.conclusion,
                        confidence=multimodal_result.confidence,
                        reasoning_type=ReasoningType.MULTIMODAL,
                        reasoning_chain=reasoning_chain,
                        explanation=multimodal_result.explanation,
                    )
            else:
                return self._utility_based_reasoning(plan, reasoning_chain)
        except Exception as e:
            logger.error(f"Adaptive reasoning failed: {e}")
            return self._create_error_result(str(e))

    def _hybrid_reasoning(self, plan, reasoning_chain):
        """Custom hybrid reasoning approach"""
        try:
            if ReasoningType.PROBABILISTIC in self.reasoners:
                prob_task = ReasoningTask(
                    task_id=f"{plan.tasks[0].task_id}_prob",
                    task_type=ReasoningType.PROBABILISTIC,
                    input_data=plan.tasks[0].input_data,
                    query=plan.tasks[0].query,
                    constraints=plan.tasks[0].constraints,
                    utility_context=plan.tasks[0].utility_context,
                )
                prob_result = self._execute_task(prob_task)
                if hasattr(prob_result, "reasoning_chain") and prob_result.reasoning_chain and prob_result.reasoning_chain.steps:
                    for step in prob_result.reasoning_chain.steps:
                        if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                            reasoning_chain.steps.append(step)

                if prob_result.confidence < 0.7 and ReasoningType.SYMBOLIC in self.reasoners:
                    symb_task = ReasoningTask(
                        task_id=f"{plan.tasks[0].task_id}_symb",
                        task_type=ReasoningType.SYMBOLIC,
                        input_data=plan.tasks[0].input_data,
                        query=plan.tasks[0].query,
                        constraints=plan.tasks[0].constraints,
                        utility_context=plan.tasks[0].utility_context,
                    )
                    symb_result = self._execute_task(symb_task)
                    if hasattr(symb_result, "reasoning_chain") and symb_result.reasoning_chain and symb_result.reasoning_chain.steps:
                        for step in symb_result.reasoning_chain.steps:
                            if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                                reasoning_chain.steps.append(step)

                    if plan.tasks[0].utility_context:
                        prob_time = getattr(prob_result, "metadata", {}).get("execution_time_ms", 100)
                        symb_time = getattr(symb_result, "metadata", {}).get("execution_time_ms", 100)
                        prob_utility = self._calculate_result_utility(prob_result, plan.tasks[0].utility_context, prob_time)
                        symb_utility = self._calculate_result_utility(symb_result, plan.tasks[0].utility_context, symb_time)
                        if symb_utility > prob_utility:
                            reasoning_chain.final_conclusion = symb_result.conclusion
                            reasoning_chain.total_confidence = symb_result.confidence
                            return ReasoningResult(
                                conclusion=symb_result.conclusion, confidence=symb_result.confidence,
                                reasoning_type=ReasoningType.HYBRID, reasoning_chain=reasoning_chain,
                                explanation=symb_result.explanation,
                            )

                if "cause" in str(plan.tasks[0].query).lower() and ReasoningType.CAUSAL in self.reasoners:
                    causal_task = ReasoningTask(
                        task_id=f"{plan.tasks[0].task_id}_causal",
                        task_type=ReasoningType.CAUSAL,
                        input_data=plan.tasks[0].input_data,
                        query=plan.tasks[0].query,
                        constraints=plan.tasks[0].constraints,
                        utility_context=plan.tasks[0].utility_context,
                    )
                    causal_result = self._execute_task(causal_task)
                    if hasattr(causal_result, "reasoning_chain") and causal_result.reasoning_chain and causal_result.reasoning_chain.steps:
                        for step in causal_result.reasoning_chain.steps:
                            if step.step_type != ReasoningType.UNKNOWN or step.explanation != "Reasoning process initialized":
                                reasoning_chain.steps.append(step)
                    reasoning_chain.final_conclusion = causal_result.conclusion
                    reasoning_chain.total_confidence = causal_result.confidence
                    return ReasoningResult(
                        conclusion=causal_result.conclusion, confidence=causal_result.confidence,
                        reasoning_type=ReasoningType.HYBRID, reasoning_chain=reasoning_chain,
                        explanation=causal_result.explanation,
                    )

                reasoning_chain.final_conclusion = prob_result.conclusion
                reasoning_chain.total_confidence = prob_result.confidence
                return ReasoningResult(
                    conclusion=prob_result.conclusion, confidence=prob_result.confidence,
                    reasoning_type=ReasoningType.HYBRID, reasoning_chain=reasoning_chain,
                    explanation=prob_result.explanation,
                )
        except Exception as e:
            logger.error(f"Hybrid reasoning failed: {e}")
        return self._create_empty_result()

    # ==========================================================================
    # TASK EXECUTION (delegation stubs)
    # ==========================================================================

    def _execute_task(self, task):
        return _task_execution.execute_task(self, task)

    def _execute_reasoner(self, engine, task):
        return _task_reasoner.execute_reasoner(self, engine, task)

    # ==========================================================================
    # ESTIMATION (delegation stubs)
    # ==========================================================================

    def _enhance_task_with_voi(self, task, voi_action):
        """Enhance task based on VOI recommendation"""
        if "tier" in voi_action:
            logger.info(f"Extracting {voi_action} features")
        task.metadata["voi_action"] = voi_action
        return task

    def _reasoning_task_to_plan_step(self, task, step_index):
        from .estimation import reasoning_task_to_plan_step
        return reasoning_task_to_plan_step(task, step_index, self.cost_model)

    def _compute_plan_estimates_using_plan_class(self, tasks, dependencies, original_task):
        from .estimation import compute_plan_estimates_using_plan_class
        return compute_plan_estimates_using_plan_class(
            tasks, dependencies, original_task, self.cost_model
        )

    def _estimate_plan_time_legacy(self, tasks):
        from .estimation import estimate_plan_time_legacy
        return estimate_plan_time_legacy(tasks, self.cost_model)

    def _estimate_plan_cost_legacy(self, tasks):
        from .estimation import estimate_plan_cost_legacy
        return estimate_plan_cost_legacy(tasks, self.cost_model)

    def _estimate_plan_time(self, tasks):
        return self._estimate_plan_time_legacy(tasks)

    def _estimate_plan_cost(self, tasks):
        return self._estimate_plan_cost_legacy(tasks)

    def _record_execution(self, task, result, elapsed_time, from_cache):
        """Record execution in monitoring systems"""
        try:
            if self.tool_monitor:
                self.tool_monitor.record_execution(
                    tool_name=str(task.task_type),
                    success=result.confidence >= self.confidence_threshold,
                    latency_ms=elapsed_time * 1000,
                    energy_mj=100,
                    confidence=result.confidence,
                    metadata={"from_cache": from_cache},
                )
            if not from_cache and self.cost_model and task.features is not None:
                CostComponent = self._selection_components.get("CostComponent")
                if CostComponent:
                    self.cost_model.update(
                        str(task.task_type), CostComponent.TIME_MS,
                        elapsed_time * 1000, task.features,
                    )
            if self.calibrator:
                self.calibrator.add_observation(
                    str(task.task_type), result.confidence,
                    result.confidence >= self.confidence_threshold, task.features,
                )
        except Exception as e:
            logger.warning(f"Execution recording failed: {e}")

    def _convert_execution_to_reasoning_result(self, exec_result):
        """Convert portfolio execution result to ReasoningResult"""
        try:
            initial_step = ReasoningStep(
                step_id=f"portfolio_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningType.HYBRID,
                input_data=None,
                output_data=exec_result.primary_result if hasattr(exec_result, "primary_result") else None,
                confidence=0.7,
                explanation="Portfolio execution",
            )
            chain = ReasoningChain(
                chain_id=str(uuid.uuid4()), steps=[initial_step],
                initial_query={},
                final_conclusion=exec_result.primary_result if hasattr(exec_result, "primary_result") else None,
                total_confidence=0.7, reasoning_types_used=set(),
                modalities_involved=set(), safety_checks=[], audit_trail=[],
            )
            return ReasoningResult(
                conclusion=exec_result.primary_result if hasattr(exec_result, "primary_result") else None,
                confidence=0.7, reasoning_type=ReasoningType.HYBRID,
                reasoning_chain=chain, explanation="Portfolio execution result",
            )
        except Exception as e:
            logger.error(f"Result conversion failed: {e}")
            return self._create_empty_result()

    # ==========================================================================
    # HELPER METHODS (delegation stubs and small kept methods)
    # ==========================================================================

    def _analyze_input_characteristics(self, task):
        """Analyze characteristics of input data"""
        characteristics = {
            "complexity": 0.5, "uncertainty": 0.5, "multimodal": False,
            "size": "small", "structure": "unstructured",
        }
        try:
            reasoning_components = _load_reasoning_components()
            ModalityType = reasoning_components.get("ModalityType")
            if isinstance(task.input_data, dict) and ModalityType:
                modality_count = sum(1 for k in task.input_data.keys() if isinstance(k, ModalityType))
                characteristics["multimodal"] = modality_count > 1
            if isinstance(task.input_data, (list, np.ndarray)):
                characteristics["size"] = "large" if len(task.input_data) > 1000 else "small"
                characteristics["complexity"] = min(1.0, len(task.input_data) / 1000)
            if isinstance(task.input_data, dict) and "graph" in task.input_data:
                characteristics["structure"] = "graph"
            elif isinstance(task.input_data, str):
                characteristics["structure"] = "text"
        except Exception:
            pass
        return characteristics

    def _create_adaptive_plan(self, task, reasoning_types):
        """Create adaptive plan with specified reasoning types"""
        tasks = []
        for reasoning_type in reasoning_types:
            if reasoning_type in self.reasoners:
                sub_task = ReasoningTask(
                    task_id=f"{task.task_id}_{reasoning_type.value}",
                    task_type=reasoning_type, input_data=task.input_data,
                    query=task.query, constraints=task.constraints,
                    utility_context=task.utility_context,
                )
                tasks.append(sub_task)
        return ReasoningPlan(
            plan_id=str(uuid.uuid4()), tasks=tasks,
            strategy=ReasoningStrategy.ENSEMBLE, dependencies={},
            estimated_time=len(tasks) * 1.0, estimated_cost=len(tasks) * 100,
            confidence_threshold=task.constraints.get("confidence_threshold", 0.5),
        )

    def _topological_sort(self, tasks, dependencies):
        return _strategies_topological_sort(tasks, dependencies)

    def _merge_dependency_results(self, original_input, dep_results):
        if not dep_results:
            return original_input
        return {
            "original": original_input,
            "dependencies": [r.conclusion for r in dep_results],
            "dep_confidence": np.mean([r.confidence for r in dep_results]),
        }

    def _combine_parallel_results(self, results):
        if not results:
            return None
        conclusions = [r.conclusion for r in results if r]
        if all(isinstance(c, dict) for c in conclusions):
            merged = {}
            for c in conclusions:
                if c:
                    merged.update(c)
            return merged
        elif all(isinstance(c, (int, float)) for c in conclusions if c is not None):
            return np.mean([c for c in conclusions if c is not None])
        else:
            valid_results = [r for r in results if r]
            if not valid_results:
                return None
            max_idx = np.argmax([r.confidence for r in valid_results])
            return valid_results[max_idx].conclusion

    def _weighted_voting(self, conclusions, weights):
        if not conclusions:
            return None
        try:
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            if all(isinstance(c, bool) for c in conclusions):
                true_weight = sum(w for c, w in zip(conclusions, weights) if c)
                return true_weight > 0.5
            if all(isinstance(c, str) for c in conclusions):
                vote_weights = defaultdict(float)
                for c, w in zip(conclusions, weights):
                    vote_weights[c] += w
                return max(vote_weights.items(), key=lambda x: x[1])[0]
            if all(isinstance(c, (int, float)) for c in conclusions):
                return sum(c * w for c, w in zip(conclusions, weights))
            max_idx = np.argmax(weights)
            return conclusions[max_idx]
        except Exception:
            return conclusions[0] if conclusions else None

    def _get_reasoning_type_weight(self, reasoning_type):
        if not self.enable_learning:
            return 1.0
        try:
            tool_name = reasoning_type.value if reasoning_type else "unknown"
            shared_weight = get_weight_manager().get_weight(tool_name, default=1.0)
            if shared_weight <= 0:
                shared_weight = 1.0
            historical_weight = self._get_historical_weight(reasoning_type)
            combined = (shared_weight + historical_weight) / 2
            return max(0.1, combined)
        except Exception:
            return 1.0

    def _get_historical_weight(self, reasoning_type):
        try:
            type_history = [h for h in self.reasoning_history if h.get("reasoning_type") == reasoning_type]
            if not type_history:
                return 1.0
            success_rate = sum(1 for h in type_history if h.get("success", False)) / len(type_history)
            avg_confidence = np.mean([h.get("confidence", 0.5) for h in type_history])
            return (success_rate + avg_confidence) / 2
        except Exception:
            return 1.0

    def _get_utility_weight(self, reasoning_type, context):
        type_profiles = {
            ReasoningType.PROBABILISTIC: {"speed": 0.8, "accuracy": 0.6, "energy": 0.7},
            ReasoningType.SYMBOLIC: {"speed": 0.5, "accuracy": 0.9, "energy": 0.6},
            ReasoningType.CAUSAL: {"speed": 0.4, "accuracy": 0.8, "energy": 0.5},
            ReasoningType.ANALOGICAL: {"speed": 0.7, "accuracy": 0.5, "energy": 0.8},
        }
        profile = type_profiles.get(reasoning_type, {"speed": 0.5, "accuracy": 0.5, "energy": 0.5})
        selection_components = _load_selection_components()
        ContextMode = selection_components.get("ContextMode")
        if ContextMode and hasattr(context, "mode"):
            if context.mode == ContextMode.RUSH:
                return profile["speed"]
            elif context.mode == ContextMode.ACCURATE:
                return profile["accuracy"]
            elif context.mode == ContextMode.EFFICIENT:
                return profile["energy"]
        return np.mean([profile["speed"], profile["accuracy"], profile["energy"]])

    def _estimate_energy(self, time_ms):
        return time_ms * 0.01

    def _calculate_result_utility(self, result, context, execution_time_ms):
        if not self.utility_model:
            return result.confidence
        try:
            energy_mj = self._estimate_energy(execution_time_ms)
            raw_utility = self.utility_model.compute_utility(
                quality=result.confidence, time=execution_time_ms,
                energy=energy_mj, risk=1 - result.confidence, context=context,
            )
            if raw_utility <= 0:
                return 0.01
            return raw_utility
        except Exception:
            return result.confidence

    def _compute_cache_key(self, task):
        try:
            content_parts = [f"type:{task.task_type.value}"]
            if task.query:
                query_str = str(task.query) if not isinstance(task.query, str) else task.query
                content_parts.append(f"query:{query_str}")
            if task.input_data is not None:
                if isinstance(task.input_data, str):
                    content_parts.append(f"input:{task.input_data[:1000]}")
                elif isinstance(task.input_data, dict):
                    sorted_items = sorted(task.input_data.items(), key=lambda x: str(x[0]))
                    content_parts.append(f"input:{str(sorted_items)[:1000]}")
                else:
                    content_parts.append(f"input:{str(task.input_data)[:1000]}")
            if task.constraints:
                relevant = {k: v for k, v in task.constraints.items() if k in ('confidence_threshold', 'max_steps', 'reasoning_depth', 'tools')}
                if relevant:
                    content_parts.append(f"constraints:{str(sorted(relevant.items()))}")
            content_str = "|".join(content_parts)
            content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:CACHE_HASH_LENGTH]
            return "_".join([task.task_type.value, str(type(task.input_data).__name__), content_hash])
        except Exception:
            return f"nocache_{task.task_id}_{uuid.uuid4().hex[:8]}"

    def _postprocess_result(self, result, task):
        """Post-process reasoning result with mathematical verification"""
        try:
            if not result.explanation and result.reasoning_chain:
                result.explanation = self.explainer.explain_chain(result.reasoning_chain)
            threshold = task.constraints.get("confidence_threshold", self.confidence_threshold)
            if result.confidence < threshold:
                result.metadata["below_confidence_threshold"] = True
                result.metadata["filter_reason"] = f"Confidence {result.confidence:.2f} below threshold {threshold}"
                result.metadata["threshold"] = threshold
                if not result.explanation or result.explanation.strip() == "":
                    result.explanation = "Analysis completed with moderate confidence."
            is_mathematical = task.query.get("is_mathematical", False) if task.query else False
            require_verification = task.constraints.get("require_verification", False) if task.constraints else False
            if (is_mathematical or require_verification) and self.math_verification_engine:
                verification_result = self._verify_mathematical_result(result, task)
                if verification_result:
                    result = self._apply_verification_to_result(result, verification_result, task)
        except Exception:
            pass
        return result

    def _verify_mathematical_result(self, result, task):
        from .verification import verify_mathematical_result
        return verify_mathematical_result(
            result, task, self.math_verification_engine, self._optional_components
        )

    def _apply_verification_to_result(self, result, verification, task):
        from .verification import apply_verification_to_result
        return apply_verification_to_result(
            result, verification, task,
            self._optional_components, self._math_accuracy_integration, self.learner,
        )

    # ==========================================================================
    # LEARNING (delegation stub)
    # ==========================================================================

    def _learn_from_reasoning(self, task, result):
        _learning.learn_from_reasoning(self, task, result)

    # ==========================================================================
    # UTILITIES (delegation stubs)
    # ==========================================================================

    def _extract_query_string(self, query):
        return _utilities.extract_query_string(query)

    def _create_safety_result(self, reason):
        return _utilities.create_safety_result(reason)

    def _extract_symbolic_constraints(self, text):
        return _utilities.extract_symbolic_constraints(text)

    def _check_sat_satisfiability(self, engine, extracted):
        return _utilities.check_sat_satisfiability(engine, extracted)

    def _create_empty_result(self):
        return _utilities.create_empty_result()

    def _create_error_result(self, error):
        return _utilities.create_error_result(error)

    # ==========================================================================
    # STATISTICS / SHUTDOWN (delegation stubs)
    # ==========================================================================

    def _update_metrics(self, result, elapsed_time, strategy):
        _shutdown.update_metrics(self, result, elapsed_time, strategy)

    def _add_to_history(self, task, result, elapsed_time):
        _shutdown.add_to_history(self, task, result, elapsed_time)

    def _add_audit_entry(self, task, result, strategy, elapsed_time):
        _shutdown.add_audit_entry(self, task, result, strategy, elapsed_time)

    def get_statistics(self):
        return _shutdown.get_statistics(self)

    def clear_caches(self):
        _shutdown.clear_caches(self)

    def _shutdown_component(self, component, name):
        _shutdown.shutdown_component(component, name)

    def shutdown(self, timeout=5.0, skip_save=False):
        _shutdown.shutdown(self, timeout, skip_save)

    def __getstate__(self):
        return _shutdown.getstate(self)

    def __setstate__(self, state):
        _shutdown.setstate(self, state)

    # ==========================================================================
    # IMPORTED METHODS (from other submodules)
    # ==========================================================================

    # State persistence methods
    from .persistence import save_state as save_state
    from .persistence import load_state as load_state

    # Multimodal reasoning methods
    from .multimodal_handler import reason_multimodal as reason_multimodal
    from .multimodal_handler import reason_counterfactual as reason_counterfactual
    from .multimodal_handler import reason_by_analogy as reason_by_analogy
