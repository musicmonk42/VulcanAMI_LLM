# world_model_router.py
"""
World Model Router - Intelligent routing and orchestration for world model updates.

Determines optimal update strategies based on observation characteristics,
system state, and learned patterns. Manages dependencies, priorities,
and resource allocation across world model components.

Integrated with comprehensive safety validation and CSIU tracking.
FIXED: All integration issues, observation storage, defensive method calls
FIXED: Type confusion with inspect signature checking
FIXED: State loading merges with defaults
FIXED: Improved defensive parameter inference for all update execution methods
FIXED: Corrected Path usage in save_state and load_state methods
FIXED: has_intervention now properly returns boolean instead of dict
FIXED: Defensive len() check for observation.variables
FIXED: _execute_update properly propagates error status
CSIU: Tracks alignment metrics during routing and execution (internal only)
CSIU: CSIU-aware tie-breaking for near-equal plans (prefers clarity↑ / entropy↓ / explainability↑)
CSIU: Records routing outcomes for learning better tie-breaks over time
"""

import inspect
import logging
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)

# Lazy import for safety validator to avoid circular import issues
# The actual import is deferred to when the class is instantiated
_safety_validator_class = None
_safety_config_class = None
_safety_imports_checked = False


def _get_safety_validator():
    """Lazy import of EnhancedSafetyValidator to avoid circular imports."""
    if not _safety_imports_checked:
        _check_safety_imports()
    return _safety_validator_class


def _get_safety_config():
    """Lazy import of SafetyConfig to avoid circular imports."""
    if not _safety_imports_checked:
        _check_safety_imports()
    return _safety_config_class


def _check_safety_imports():
    """Check and cache safety module imports."""
    global _safety_validator_class, _safety_config_class, _safety_imports_checked
    _safety_imports_checked = True

    try:
        from ..safety.safety_validator import EnhancedSafetyValidator

        _safety_validator_class = EnhancedSafetyValidator
    except (ImportError, AttributeError):
        _safety_validator_class = None

    try:
        from ..safety.safety_types import SafetyConfig

        _safety_config_class = SafetyConfig
    except (ImportError, AttributeError):
        _safety_config_class = None


class UpdateType(Enum):
    """Types of world model updates"""

    INTERVENTION = "intervention"
    CORRELATION = "correlation"
    DYNAMICS = "dynamics"
    INVARIANT = "invariant"
    CONFIDENCE = "confidence"
    CAUSAL = "causal"
    DISTRIBUTION = "distribution"
    STRUCTURAL = "structural"


class UpdatePriority(Enum):
    """Priority levels for updates"""

    CRITICAL = 0  # Must run immediately
    HIGH = 1  # Run soon
    NORMAL = 2  # Run when feasible
    LOW = 3  # Can defer
    BATCH = 4  # Can batch with others


@dataclass
class UpdateStrategy:
    """Strategy for a specific update"""

    update_type: UpdateType
    priority: UpdatePriority
    dependencies: Set[UpdateType]
    estimated_cost_ms: float
    confidence_threshold: float
    can_parallelize: bool
    can_defer: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationSignature:
    """Signature characterizing an observation"""

    has_intervention: bool
    has_temporal: bool
    has_multi_variable: bool
    has_anomaly: bool
    has_structural_change: bool
    variable_count: int
    time_delta: Optional[float]
    confidence: float
    domain: Optional[str]
    pattern_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdatePlan:
    """Execution plan for world model updates"""

    immediate: List[UpdateType]
    deferred: List[UpdateType]
    parallel_groups: List[List[UpdateType]]
    estimated_time_ms: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class UpdateDependencyGraph:
    """Manages dependencies between update types"""

    def __init__(self):
        # Define update dependencies
        self.dependencies = {
            UpdateType.INTERVENTION: set(),  # No dependencies
            UpdateType.CORRELATION: set(),  # No dependencies
            UpdateType.DYNAMICS: {UpdateType.CORRELATION},  # Needs correlation
            UpdateType.CAUSAL: {UpdateType.INTERVENTION, UpdateType.CORRELATION},
            UpdateType.INVARIANT: {UpdateType.CORRELATION, UpdateType.DYNAMICS},
            UpdateType.CONFIDENCE: {UpdateType.CORRELATION},
            UpdateType.DISTRIBUTION: {UpdateType.CORRELATION},
            UpdateType.STRUCTURAL: {UpdateType.CAUSAL, UpdateType.INVARIANT},
        }

        # Define mutual exclusions (cannot run in parallel)
        self.exclusions = {
            UpdateType.INTERVENTION: {UpdateType.CAUSAL},  # Both modify causal graph
            UpdateType.STRUCTURAL: {UpdateType.CAUSAL, UpdateType.INVARIANT},
        }

    def get_dependencies(self, update_type: UpdateType) -> Set[UpdateType]:
        """Get dependencies for an update type"""
        return self.dependencies.get(update_type, set())

    def can_parallelize(self, types: List[UpdateType]) -> bool:
        """Check if update types can run in parallel"""
        for i, type1 in enumerate(types):
            for type2 in types[i + 1 :]:
                # Check mutual exclusions
                if type2 in self.exclusions.get(type1, set()):
                    return False
                if type1 in self.exclusions.get(type2, set()):
                    return False
        return True

    def get_execution_order(
        self, required_updates: Set[UpdateType]
    ) -> List[List[UpdateType]]:
        """Get execution order respecting dependencies"""
        executed = set()
        execution_groups = []

        remaining = required_updates.copy()

        while remaining:
            # Find updates whose dependencies are satisfied
            ready = []
            for update in remaining:
                deps = self.get_dependencies(update)
                if deps.issubset(executed):
                    ready.append(update)

            if not ready:
                # Circular dependency or error
                logger.error(f"Cannot resolve dependencies for {remaining}")
                break

            # Group updates that can run in parallel
            parallel_group = []
            for update in ready:
                if self.can_parallelize(parallel_group + [update]):
                    parallel_group.append(update)

            execution_groups.append(parallel_group)
            executed.update(parallel_group)
            remaining.difference_update(parallel_group)

        return execution_groups


class PatternLearner:
    """Learns patterns from observation-update histories"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.pattern_history = deque(maxlen=window_size)

        # Pattern statistics
        self.pattern_counts = defaultdict(int)
        self.pattern_success = defaultdict(list)
        self.pattern_costs = defaultdict(list)

        # Learned rules
        self.rules = []
        self.rule_confidence = {}

    def record_pattern(
        self,
        signature: ObservationSignature,
        updates_run: List[UpdateType],
        success: bool,
        cost_ms: float,
    ):
        """Record an observation-update pattern"""
        # FIXED: Check for None signature
        if signature is None:
            logger.warning("Cannot record pattern with None signature")
            return

        pattern = {
            "signature": signature,
            "updates": updates_run,
            "success": success,
            "cost_ms": cost_ms,
            "timestamp": time.time(),
        }

        self.pattern_history.append(pattern)

        # Update statistics
        pattern_key = self._get_pattern_key(signature, updates_run)
        self.pattern_counts[pattern_key] += 1
        self.pattern_success[pattern_key].append(success)
        self.pattern_costs[pattern_key].append(cost_ms)

        # Periodically learn rules
        if len(self.pattern_history) % 100 == 0:
            self._learn_rules()

    def predict_updates(self, signature: ObservationSignature) -> List[UpdateType]:
        """Predict which updates to run based on learned patterns"""

        # Find matching rules
        matched_rules = []
        for rule in self.rules:
            if self._rule_matches(rule, signature):
                confidence = self.rule_confidence.get(rule["id"], 0)
                matched_rules.append((rule, confidence))

        if not matched_rules:
            return []

        # Use highest confidence rule
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        best_rule = matched_rules[0][0]

        return best_rule["updates"]

    def _get_pattern_key(
        self, signature: ObservationSignature, updates: List[UpdateType]
    ) -> str:
        """Create hashable key for pattern"""
        sig_key = f"{signature.has_intervention}_{signature.has_temporal}_{signature.has_multi_variable}"
        update_key = "_".join(sorted([u.value for u in updates]))
        return f"{sig_key}::{update_key}"

    def _learn_rules(self):
        """Learn rules from pattern history"""
        # Simple rule learning based on frequency and success
        new_rules = []

        for pattern_key, count in self.pattern_counts.items():
            if count < 5:  # Need minimum observations
                continue

            success_rate = np.mean(self.pattern_success[pattern_key])
            if success_rate < 0.7:  # Need good success rate
                continue

            # Parse pattern key back to rule
            sig_part, update_part = pattern_key.split("::")
            sig_values = sig_part.split("_")
            updates = [UpdateType(u) for u in update_part.split("_")]

            rule = {
                "id": pattern_key,
                "signature_conditions": {
                    "has_intervention": sig_values[0] == "True",
                    "has_temporal": sig_values[1] == "True",
                    "has_multi_variable": sig_values[2] == "True",
                },
                "updates": updates,
                "success_rate": success_rate,
                "avg_cost": np.mean(self.pattern_costs[pattern_key]),
            }

            new_rules.append(rule)
            self.rule_confidence[pattern_key] = success_rate

        self.rules = new_rules

    def _rule_matches(
        self, rule: Dict[str, Any], signature: ObservationSignature
    ) -> bool:
        """Check if rule matches signature"""
        conditions = rule["signature_conditions"]

        if conditions.get("has_intervention") and not signature.has_intervention:
            return False
        if conditions.get("has_temporal") and not signature.has_temporal:
            return False
        if conditions.get("has_multi_variable") and not signature.has_multi_variable:
            return False

        return True


class CostModel:
    """Models costs of different update operations"""

    def __init__(self):
        # Base costs in milliseconds
        self.base_costs = {
            UpdateType.INTERVENTION: 50,
            UpdateType.CORRELATION: 20,
            UpdateType.DYNAMICS: 30,
            UpdateType.CAUSAL: 100,
            UpdateType.INVARIANT: 40,
            UpdateType.CONFIDENCE: 10,
            UpdateType.DISTRIBUTION: 25,
            UpdateType.STRUCTURAL: 150,
        }

        # Scaling factors based on data size
        self.scale_factors = defaultdict(lambda: 1.0)

        # Historical costs for calibration
        self.cost_history = defaultdict(lambda: deque(maxlen=100))

    def estimate_cost(self, update_type: UpdateType, observation_size: int) -> float:
        """Estimate cost of an update"""
        base = self.base_costs.get(update_type, 50)

        # Scale by observation size
        scale = 1.0 + np.log10(max(1, observation_size / 100))

        # Apply historical calibration
        if self.cost_history[update_type]:
            # Blend estimate with historical average
            historical_avg = np.mean(self.cost_history[update_type])
            estimated = base * scale
            return 0.7 * historical_avg + 0.3 * estimated

        return base * scale

    def update_cost(self, update_type: UpdateType, actual_cost: float):
        """Update cost model with actual measurement"""
        self.cost_history[update_type].append(actual_cost)

        # Update scale factor if consistently off
        if len(self.cost_history[update_type]) >= 10:
            avg_actual = np.mean(self.cost_history[update_type])
            base = self.base_costs[update_type]
            if avg_actual > base * 1.5 or avg_actual < base * 0.5:
                self.base_costs[update_type] = avg_actual


class WorldModelRouter:
    """
    Intelligent router for world model updates.
    Determines optimal update strategy based on observation characteristics,
    system state, and learned patterns.

    Integrated with comprehensive safety validation and CSIU tracking.
    FIXED: Stores observation, defensive method calls, proper error handling
    FIXED: Type confusion with inspect signature checking
    FIXED: Improved defensive parameter inference for all update execution methods
    CSIU: Tracks alignment metrics during routing/execution (internal only)
    CSIU: CSIU-aware tie-breaking for near-equal plans
    CSIU: Records routing outcomes for learning
    """

    def __init__(
        self,
        world_model,
        config: Optional[Dict[str, Any]] = None,
        self_improvement_drive=None,
    ):
        self.world_model = world_model
        self.config = config or {}
        self.self_improvement_drive = self_improvement_drive

        # FIXED: Store current observation for use in execute()
        self.current_observation = None

        # Core components
        self.dependency_graph = UpdateDependencyGraph()
        self.pattern_learner = PatternLearner()
        self.cost_model = CostModel()

        # Update strategies
        self.strategies = self._initialize_strategies()

        # Resource constraints
        self.time_budget_ms = config.get("time_budget_ms", 1000)
        self.min_confidence = config.get("min_confidence", 0.5)

        # Performance tracking
        self.metrics = defaultdict(int)
        self.execution_history = deque(maxlen=1000)
        self.cache = {}  # Signature -> UpdatePlan cache
        self.cache_ttl = config.get("cache_ttl", 60)

        # Learning parameters
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.use_learned_patterns = config.get("use_learning", True)

        # Safety tracking
        self.safety_blocks = defaultdict(int)
        self.safety_corrections = defaultdict(int)

        # CSIU tracking (internal only, not exposed to UX)
        self.csiu_routing_history = deque(maxlen=1000)

        # Thread safety - FIXED: Use RLock
        self.lock = threading.RLock()

        # Meta-reasoning components
        self.meta = getattr(world_model, "motivational_introspection", None)
        self.validation_tracker = getattr(world_model, "validation_tracker", None)
        self.transparency = getattr(world_model, "transparency_interface", None)
        self.value_evolution = getattr(world_model, "value_evolution_tracker", None)
        # Optional feature flags (safe defaults)
        self.use_meta_reasoning = self.config.get("use_meta_reasoning", True)
        self.meta_soften_only = self.config.get(
            "meta_soften_only", True
        )  # never hard-fail plans
        self.meta_reweight_priorities = self.config.get(
            "meta_reweight_priorities", True
        )

        # Initialize safety validator (using lazy imports to avoid circular import issues)
        EnhancedSafetyValidator = _get_safety_validator()
        SafetyConfig = _get_safety_config()

        if EnhancedSafetyValidator is not None and SafetyConfig is not None:
            safety_config = config.get("safety_config", {})
            if isinstance(safety_config, dict) and safety_config:
                self.safety_validator = EnhancedSafetyValidator(
                    SafetyConfig.from_dict(safety_config)
                )
            elif (
                hasattr(world_model, "safety_validator")
                and world_model.safety_validator
            ):
                self.safety_validator = world_model.safety_validator
            else:
                self.safety_validator = EnhancedSafetyValidator()
            logger.info("WorldModelRouter: Safety validator initialized")
        else:
            self.safety_validator = None
            logger.warning(
                "WorldModelRouter: Safety validator not available - operating without safety checks"
            )

        logger.info(
            "WorldModelRouter initialized with %d strategies", len(self.strategies)
        )
        if self.self_improvement_drive and getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            logger.info("WorldModelRouter: CSIU tracking enabled (internal only)")

    def _initialize_strategies(self) -> Dict[UpdateType, UpdateStrategy]:
        """Initialize update strategies"""
        return {
            UpdateType.INTERVENTION: UpdateStrategy(
                update_type=UpdateType.INTERVENTION,
                priority=UpdatePriority.CRITICAL,
                dependencies=set(),
                estimated_cost_ms=50,
                confidence_threshold=0.8,
                can_parallelize=False,  # Modifies causal graph
                can_defer=False,
            ),
            UpdateType.CORRELATION: UpdateStrategy(
                update_type=UpdateType.CORRELATION,
                priority=UpdatePriority.HIGH,
                dependencies=set(),
                estimated_cost_ms=20,
                confidence_threshold=0.5,
                can_parallelize=True,
                can_defer=False,  # Foundation for other updates
            ),
            UpdateType.DYNAMICS: UpdateStrategy(
                update_type=UpdateType.DYNAMICS,
                priority=UpdatePriority.NORMAL,
                dependencies={UpdateType.CORRELATION},
                estimated_cost_ms=30,
                confidence_threshold=0.6,
                can_parallelize=True,
                can_defer=True,
            ),
            UpdateType.CAUSAL: UpdateStrategy(
                update_type=UpdateType.CAUSAL,
                priority=UpdatePriority.HIGH,
                dependencies={UpdateType.INTERVENTION, UpdateType.CORRELATION},
                estimated_cost_ms=100,
                confidence_threshold=0.7,
                can_parallelize=False,  # Modifies causal graph
                can_defer=False,
            ),
            UpdateType.INVARIANT: UpdateStrategy(
                update_type=UpdateType.INVARIANT,
                priority=UpdatePriority.NORMAL,
                dependencies={UpdateType.CORRELATION},
                estimated_cost_ms=40,
                confidence_threshold=0.6,
                can_parallelize=True,
                can_defer=True,
            ),
            UpdateType.CONFIDENCE: UpdateStrategy(
                update_type=UpdateType.CONFIDENCE,
                priority=UpdatePriority.LOW,
                dependencies=set(),
                estimated_cost_ms=10,
                confidence_threshold=0.3,
                can_parallelize=True,
                can_defer=True,
            ),
            UpdateType.DISTRIBUTION: UpdateStrategy(
                update_type=UpdateType.DISTRIBUTION,
                priority=UpdatePriority.LOW,
                dependencies={UpdateType.CORRELATION},
                estimated_cost_ms=25,
                confidence_threshold=0.5,
                can_parallelize=True,
                can_defer=True,
            ),
            UpdateType.STRUCTURAL: UpdateStrategy(
                update_type=UpdateType.STRUCTURAL,
                priority=UpdatePriority.CRITICAL,
                dependencies={UpdateType.CAUSAL, UpdateType.INVARIANT},
                estimated_cost_ms=150,
                confidence_threshold=0.8,
                can_parallelize=False,  # Major structural change
                can_defer=False,
            ),
        }

    def _csiu_predict_user_fit(self, plan: UpdatePlan) -> float:
        """
        Predict user fit for a plan based on CSIU features.

        Used for tie-breaking between near-equal plans.
        Prefers: clarity↑ / entropy↓ / explainability↑

        Args:
            plan: UpdatePlan with metadata

        Returns:
            Predicted user fit score (0-1)
        """
        if not self.self_improvement_drive or not getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            return 0.5

        md = plan.metadata

        # Get predicted metrics (with fallback to reasonable defaults)
        clarity_est = md.get("predicted_intent_clarity", 0.85)
        entropy_est = md.get("predicted_comm_entropy", 0.06)
        explainab = md.get("explainability", 0.6)

        # Combine: clarity↑ / entropy↓ / explainability↑
        # Weighted combination (clarity matters most, then explainability, then entropy)
        user_fit = 0.5 * clarity_est - 0.3 * entropy_est + 0.2 * explainab

        # Clamp to reasonable range
        return max(0.0, min(1.0, user_fit))

    def _record_csiu_routing_outcome(self, plan: UpdatePlan, success: bool):
        """
        Record routing outcome for CSIU learning.

        Tracks which plan characteristics led to successful vs unsuccessful routes
        to improve future tie-breaking decisions.

        Args:
            plan: The UpdatePlan that was executed
            success: Whether the execution was successful
        """
        if not self.self_improvement_drive or not getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            return

        try:
            self.csiu_routing_history.append(
                {
                    "ts": time.time(),
                    "explainability": plan.metadata.get("explainability", 0.6),
                    "success": bool(success),
                }
            )
        except Exception as e:
            logger.debug(f"Failed to record CSIU routing outcome: {e}")

    def route(
        self, observation: Any, constraints: Optional[Dict[str, Any]] = None
    ) -> UpdatePlan:
        """
        Main routing method - determines optimal update plan

        Args:
            observation: Incoming observation
            constraints: Optional constraints (time_budget_ms, priority_threshold)

        Returns:
            UpdatePlan with execution strategy
        """
        with self.lock:
            self.metrics["plans_created"] += 1  # Moved from end of method
            start_time = time.time()

            # FIXED: Store observation for execute()
            self.current_observation = observation

            # CSIU: Capture telemetry snapshot before routing (internal only)
            csiu_before = None
            if self.self_improvement_drive and getattr(
                self.self_improvement_drive, "_csiu_enabled", False
            ):
                try:
                    if hasattr(
                        self.self_improvement_drive, "_collect_telemetry_snapshot"
                    ):
                        csiu_before = (
                            self.self_improvement_drive._collect_telemetry_snapshot()
                        )
                except Exception as e:
                    logger.debug(
                        f"Failed to capture CSIU telemetry before routing: {e}"
                    )

            # SAFETY: Validate observation before routing
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, "analyze_observation_safety"):
                        obs_safety = self.safety_validator.analyze_observation_safety(
                            observation
                        )
                        if not obs_safety.get("safe", True):
                            logger.warning(
                                "Unsafe observation blocked from routing: %s",
                                obs_safety.get("reason", "unknown"),
                            )
                            self.safety_blocks["observation"] += 1
                            # Return minimal safe plan
                            return UpdatePlan(
                                immediate=[],
                                deferred=[],
                                parallel_groups=[],
                                estimated_time_ms=0,
                                confidence=0.0,
                                reasoning=(
                                    f"Observation blocked by safety validator: "
                                    f"{obs_safety.get('reason', 'unknown')}"
                                ),
                                metadata={"safety_blocked": True},
                            )
                except Exception as e:
                    logger.error(
                        "Safety validator error in analyze_observation_safety: %s", e
                    )
                    # Fail-safe: block on error
                    return UpdatePlan(
                        immediate=[],
                        deferred=[],
                        parallel_groups=[],
                        estimated_time_ms=0,
                        confidence=0.0,
                        reasoning=f"Safety validator error: {str(e)}",
                        metadata={"safety_blocked": True, "error": str(e)},
                    )

            # Extract observation signature
            signature = self._extract_signature(observation)

            # SAFETY: Validate signature
            if self.safety_validator:
                sig_safety = self._validate_signature_safety(signature)
                if not sig_safety["safe"]:
                    logger.warning(
                        "Unsafe signature detected: %s", sig_safety["reason"]
                    )
                    self.safety_corrections["signature"] += 1
                    # Apply safety corrections to signature
                    signature = self._apply_signature_corrections(signature, sig_safety)

            # Check cache
            cache_key = self._get_cache_key(signature)
            if cache_key in self.cache:
                cached_plan, cache_time = self.cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    self.metrics["cache_hits"] += 1
                    return cached_plan

            # Determine required updates
            required_updates = self._determine_required_updates(signature)

            # Apply learned patterns
            if self.use_learned_patterns and np.random.random() > self.exploration_rate:
                learned_updates = self.pattern_learner.predict_updates(signature)
                if learned_updates:
                    required_updates.update(learned_updates)
                    self.metrics["learned_patterns_used"] += 1

            # Apply constraints
            if constraints:
                required_updates = self._apply_constraints(
                    required_updates, constraints, signature
                )

            # ---- START META-REASONING PRE-PLAN HOOK ----
            proposal = {
                "id": f"route_{int(time.time())}",
                "objective": "world_model_update",
                "action": "plan_updates",
                "details": {
                    "constraints": constraints,
                    "initial_updates": [u.value for u in required_updates],
                },
            }
            vr = None

            if self.use_meta_reasoning and self.meta:
                try:
                    vr = self.meta.validate_proposal_alignment(proposal)
                    if not vr.valid and self.meta_soften_only:
                        # Drop risky steps conservatively (example: skip INTERVENTION)
                        if UpdateType.INTERVENTION in required_updates:
                            required_updates.remove(UpdateType.INTERVENTION)
                            logger.debug(
                                "Meta-reasoning softening: Removed INTERVENTION update."
                            )

                            # Also remove causal if intervention is its dependency and it's not otherwise satisfied
                            if UpdateType.CAUSAL in required_updates:
                                causal_deps = self.dependency_graph.get_dependencies(
                                    UpdateType.CAUSAL
                                )
                                if not causal_deps.issubset(required_updates):
                                    required_updates.remove(UpdateType.CAUSAL)
                                    logger.debug(
                                        "Meta-reasoning softening: Removed CAUSAL update due to missing dependency."
                                    )

                    if vr and self.meta_reweight_priorities:
                        # nudge priorities using objective weights (pseudo, keep light-touch)
                        pass
                except Exception as e:
                    logger.warning(f"Meta-reasoning pre-plan hook failed: {e}")
                    vr = None  # never block on meta errors
            # ---- END META-REASONING PRE-PLAN HOOK ----

            # Create execution plan
            plan = self._create_execution_plan(required_updates, signature)

            # SAFETY: Validate execution plan
            if self.safety_validator:
                plan_safety = self._validate_plan_safety(plan, observation)
                if not plan_safety["safe"]:
                    logger.warning(
                        "Unsafe execution plan detected: %s", plan_safety["reason"]
                    )
                    self.safety_corrections["plan"] += 1
                    # Apply safety corrections to plan
                    plan = self._apply_plan_corrections(plan, plan_safety)

            # ---- START: Attach meta results to plan ----
            plan.metadata["proposal"] = proposal
            plan.metadata["validation_result"] = vr
            # ---- END: Attach meta results ----

            # Cache plan
            self.cache[cache_key] = (plan, time.time())

            # CSIU: Capture telemetry after routing and record (internal only)
            if csiu_before is not None:
                try:
                    csiu_after = None
                    if hasattr(
                        self.self_improvement_drive, "_collect_telemetry_snapshot"
                    ):
                        csiu_after = (
                            self.self_improvement_drive._collect_telemetry_snapshot()
                        )

                    if csiu_after:
                        routing_record = {
                            "timestamp": time.time(),
                            "csiu_before": csiu_before,
                            "csiu_after": csiu_after,
                            "signature": signature,
                            "updates_planned": len(plan.immediate)
                            + sum(len(g) for g in plan.parallel_groups),
                            "estimated_time_ms": plan.estimated_time_ms,
                        }
                        self.csiu_routing_history.append(routing_record)
                except Exception as e:
                    logger.debug(f"Failed to record CSIU telemetry after routing: {e}")

            # Track metrics
            self.metrics["avg_planning_time_ms"] = (
                0.9 * self.metrics.get("avg_planning_time_ms", 0)
                + 0.1 * (time.time() - start_time) * 1000
            )

            return plan

    def execute(self, plan: UpdatePlan) -> Dict[str, Any]:
        """
        Execute an update plan

        Returns:
            Execution results and metrics
        """
        # ---- START: Add these variables for the hook ----
        proposal = plan.metadata.get("proposal")
        vr = plan.metadata.get("validation_result")
        success = False  # Default to false

        # Ensure proposal exists, create a fallback if not
        if not proposal:
            proposal = {
                "id": f"exec_{int(time.time())}",
                "objective": "world_model_update",
                "action": "execute_updates",
                "details": {"plan_reasoning": plan.reasoning},
            }
        # ---- END: Add variables ----

        try:
            start_time = time.time()
            results = {}
            updates_executed = []

            # CSIU: Capture telemetry before execution (internal only)
            csiu_exec_before = None
            if self.self_improvement_drive and getattr(
                self.self_improvement_drive, "_csiu_enabled", False
            ):
                try:
                    if hasattr(
                        self.self_improvement_drive, "_collect_telemetry_snapshot"
                    ):
                        csiu_exec_before = (
                            self.self_improvement_drive._collect_telemetry_snapshot()
                        )
                except Exception as e:
                    logger.debug(
                        f"Failed to capture CSIU telemetry before execution: {e}"
                    )

            # Check if plan was safety-blocked
            if plan.metadata.get("safety_blocked"):
                return {
                    "results": {},
                    "updates_executed": [],
                    "execution_time_ms": 0,
                    "success": False,
                    "safety_blocked": True,
                    "reason": plan.reasoning,
                }

            # Execute immediate updates
            for update_type in plan.immediate:
                try:
                    # SAFETY: Validate update before execution
                    if self.safety_validator:
                        update_safety = self._validate_update_safety(update_type, plan)
                        if not update_safety["safe"]:
                            logger.warning(
                                "Unsafe update %s blocked: %s",
                                update_type,
                                update_safety["reason"],
                            )
                            self.safety_blocks[f"update_{update_type.value}"] += 1
                            results[update_type.value] = {
                                "status": "blocked",
                                "reason": update_safety["reason"],
                                "safety_blocked": True,
                            }
                            continue

                    result = self._execute_update(update_type)

                    # SAFETY: Validate update result
                    if self.safety_validator:
                        result_safety = self._validate_result_safety(
                            update_type, result
                        )
                        if not result_safety["safe"]:
                            logger.warning(
                                "Unsafe result from %s: %s",
                                update_type,
                                result_safety["reason"],
                            )
                            self.safety_corrections[f"result_{update_type.value}"] += 1
                            result = self._apply_result_corrections(
                                result, result_safety
                            )

                    results[update_type.value] = result
                    updates_executed.append(update_type)
                except Exception as e:
                    logger.error(f"Update {update_type} failed: {e}")
                    results[update_type.value] = {"status": "error", "error": str(e)}

            # Execute parallel groups
            for parallel_group in plan.parallel_groups:
                group_results = self._execute_parallel_group(parallel_group)
                results.update(group_results)
                updates_executed.extend(parallel_group)

            # Schedule deferred updates if any
            if plan.deferred:
                self._schedule_deferred(plan.deferred)

            # Record execution
            execution_time = (time.time() - start_time) * 1000
            success = all(
                "error" not in r and r.get("status") != "error"
                for r in results.values()
                if isinstance(r, dict)
            )

            # CSIU: Record routing outcome for learning (internal only)
            self._record_csiu_routing_outcome(plan, success)

            # Update learning - FIXED: Check for None signature
            signature = plan.metadata.get("signature")
            if signature is not None:
                self.pattern_learner.record_pattern(
                    signature, updates_executed, success, execution_time
                )

            # Update cost model
            for update_type in updates_executed:
                if update_type.value in results:
                    result = results[update_type.value]
                    if isinstance(result, dict):
                        update_time = result.get("execution_time_ms", 0)
                        if update_time > 0:
                            self.cost_model.update_cost(update_type, update_time)

            # CSIU: Capture telemetry after execution and add to results (internal only)
            csiu_exec_after = None
            if csiu_exec_before is not None:
                try:
                    if hasattr(
                        self.self_improvement_drive, "_collect_telemetry_snapshot"
                    ):
                        csiu_exec_after = (
                            self.self_improvement_drive._collect_telemetry_snapshot()
                        )

                    if csiu_exec_after:
                        execution_record = {
                            "timestamp": time.time(),
                            "csiu_before": csiu_exec_before,
                            "csiu_after": csiu_exec_after,
                            "updates_executed": [u.value for u in updates_executed],
                            "execution_time_ms": execution_time,
                            "success": success,
                        }
                        self.execution_history.append(execution_record)
                except Exception as e:
                    logger.debug(
                        f"Failed to record CSIU telemetry after execution: {e}"
                    )

            exec_results = {
                "results": results,
                "updates_executed": [u.value for u in updates_executed],
                "execution_time_ms": execution_time,
                "success": success,
                "safety_checks": "enabled" if self.safety_validator else "disabled",
            }

            # Add CSIU data if available (internal only, not exposed to UX)
            if csiu_exec_before and csiu_exec_after:
                exec_results["_internal_csiu"] = {
                    "before": csiu_exec_before,
                    "after": csiu_exec_after,
                    "note": "Internal tracking only, not for UX",
                }

            return exec_results

        finally:
            # ---- START META-REASONING POST-EXECUTION HOOK ----
            # 'success', 'proposal', and 'vr' are available from the outer scope

            # Record validation + audit
            if self.validation_tracker:
                try:
                    self.validation_tracker.record_validation(
                        proposal=proposal,
                        validation_result=vr
                        if vr is not None
                        else {
                            "proposal_id": proposal["id"],
                            "valid": True,
                            "confidence": 0.0,
                            "reasoning": "No pre-validation run.",
                        },
                        actual_outcome="success" if success else "partial",
                    )
                except Exception as e:
                    logger.debug(f"Meta-reasoning validation_tracker hook failed: {e}")

            if self.transparency and vr is not None:
                try:
                    _ = self.transparency.serialize_validation(vr)
                except Exception as e:
                    logger.debug(f"Meta-reasoning transparency hook failed: {e}")

            # Optional: value drift sampling hook (if you track live values per plan)
            if self.value_evolution:
                try:
                    # value_map = {"safety": 0.98, "helpfulness": 0.92, ...}  # your signals
                    # self.value_evolution.observe_values(value_map)
                    pass
                except Exception as e:
                    logger.debug(f"Meta-reasoning value_evolution hook failed: {e}")
            # ---- END META-REASONING POST-EXECUTION HOOK ----

    def _extract_signature(self, observation: Any) -> ObservationSignature:
        """Extract signature from observation"""

        # FIXED: Basic feature extraction - ensure boolean conversion
        has_intervention = bool(
            (
                hasattr(observation, "intervention_data")
                and observation.intervention_data
            )
            or (hasattr(observation, "is_intervention") and observation.is_intervention)
            or (
                hasattr(observation, "metadata")
                and observation.metadata
                and observation.metadata.get("is_intervention", False)
            )
        )

        has_temporal = hasattr(observation, "timestamp") and hasattr(
            observation, "sequence_id"
        )

        # FIXED: Defensive len() check for variables
        variable_count = 0
        if hasattr(observation, "variables"):
            try:
                # Try to get length if it's a proper collection
                if hasattr(observation.variables, "__len__"):
                    variable_count = len(observation.variables)
                elif hasattr(observation.variables, "keys"):
                    variable_count = len(list(observation.variables.keys()))
            except (TypeError, AttributeError):
                # If len() fails, observation.variables might be a Mock or invalid
                variable_count = 0

        has_multi_variable = variable_count > 1

        # Anomaly detection
        has_anomaly = self._detect_anomaly(observation)

        # Structural change detection
        has_structural_change = self._detect_structural_change(observation)

        # Time delta
        time_delta = None
        if hasattr(observation, "timestamp") and hasattr(
            self.world_model, "last_observation_time"
        ):
            last_time = self.world_model.last_observation_time
            if last_time:
                time_delta = observation.timestamp - last_time

        # Confidence
        confidence = (
            observation.confidence if hasattr(observation, "confidence") else 1.0
        )

        # Domain
        domain = observation.domain if hasattr(observation, "domain") else None

        # Create pattern hash
        pattern_str = (
            f"{has_intervention}_{has_temporal}_{has_multi_variable}_{variable_count}"
        )
        pattern_hash = str(hash(pattern_str))

        return ObservationSignature(
            has_intervention=has_intervention,
            has_temporal=has_temporal,
            has_multi_variable=has_multi_variable,
            has_anomaly=has_anomaly,
            has_structural_change=has_structural_change,
            variable_count=variable_count,
            time_delta=time_delta,
            confidence=confidence,
            domain=domain,
            pattern_hash=pattern_hash,
        )

    def _determine_required_updates(
        self, signature: ObservationSignature
    ) -> Set[UpdateType]:
        """Determine which updates are required"""
        required = set()

        # Rule-based determination
        if signature.has_intervention:
            required.add(UpdateType.INTERVENTION)
            required.add(UpdateType.CAUSAL)  # Update causal graph

        if signature.has_multi_variable:
            required.add(UpdateType.CORRELATION)

        if signature.has_temporal:
            required.add(UpdateType.DYNAMICS)

        if signature.has_anomaly:
            required.add(UpdateType.INVARIANT)
            required.add(UpdateType.DISTRIBUTION)

        if signature.has_structural_change:
            required.add(UpdateType.STRUCTURAL)

        # Always update confidence
        required.add(UpdateType.CONFIDENCE)

        # Add dependencies
        all_required = required.copy()
        for update_type in required:
            all_required.update(self.dependency_graph.get_dependencies(update_type))

        return all_required

    def _apply_constraints(
        self,
        updates: Set[UpdateType],
        constraints: Dict[str, Any],
        signature: ObservationSignature,
    ) -> Set[UpdateType]:
        """Apply constraints to filter updates with CSIU tie-breaking"""

        time_budget = constraints.get("time_budget_ms", self.time_budget_ms)
        priority_threshold = constraints.get(
            "priority_threshold", UpdatePriority.NORMAL
        )

        # Estimate total cost
        observation_size = signature.variable_count * 100  # Rough estimate
        total_cost = sum(
            self.cost_model.estimate_cost(u, observation_size) for u in updates
        )

        if total_cost <= time_budget:
            return updates  # Can do everything

        # Need to prioritize - create list of (update, score) tuples
        scored_updates = []

        for update in updates:
            strategy = self.strategies[update]

            # Skip if below priority threshold
            if isinstance(priority_threshold, str):
                # Convert string to enum
                try:
                    priority_threshold = UpdatePriority[priority_threshold.upper()]
                except (KeyError, AttributeError):
                    priority_threshold = UpdatePriority.NORMAL

            if strategy.priority.value > priority_threshold.value:
                continue

            # Base score from priority and confidence
            base_score = (10 - strategy.priority.value) + strategy.confidence_threshold

            scored_updates.append((update, base_score))

        # Sort by score
        scored_updates.sort(key=lambda x: x[1], reverse=True)

        # CSIU: Apply tie-breaking for near-equal scores (within epsilon)
        if (
            len(scored_updates) >= 2
            and self.self_improvement_drive
            and getattr(self.self_improvement_drive, "_csiu_enabled", False)
        ):
            epsilon = 1e-3
            i = 0
            while i < len(scored_updates) - 1:
                update_a, score_a = scored_updates[i]
                update_b, score_b = scored_updates[i + 1]

                if abs(score_a - score_b) < epsilon:
                    # Near tie - create minimal plans for comparison
                    plan_a = UpdatePlan(
                        immediate=[update_a],
                        deferred=[],
                        parallel_groups=[],
                        estimated_time_ms=self.cost_model.estimate_cost(
                            update_a, observation_size
                        ),
                        confidence=self.strategies[update_a].confidence_threshold,
                        reasoning="",
                        metadata=self.strategies[update_a].metadata.copy(),
                    )

                    plan_b = UpdatePlan(
                        immediate=[update_b],
                        deferred=[],
                        parallel_groups=[],
                        estimated_time_ms=self.cost_model.estimate_cost(
                            update_b, observation_size
                        ),
                        confidence=self.strategies[update_b].confidence_threshold,
                        reasoning="",
                        metadata=self.strategies[update_b].metadata.copy(),
                    )

                    # Apply CSIU tie-breaking (1% weight as specified)
                    user_fit_a = self._csiu_predict_user_fit(plan_a)
                    user_fit_b = self._csiu_predict_user_fit(plan_b)

                    adjusted_score_a = score_a + 0.01 * user_fit_a
                    adjusted_score_b = score_b + 0.01 * user_fit_b

                    # Re-sort this pair if needed
                    if adjusted_score_b > adjusted_score_a:
                        scored_updates[i], scored_updates[i + 1] = (
                            scored_updates[i + 1],
                            scored_updates[i],
                        )
                        logger.debug(
                            f"CSIU tie-break: {update_b.value} preferred over {update_a.value}"
                        )

                i += 1

        # Select updates within budget
        prioritized = []
        cost_so_far = 0

        for update, score in scored_updates:
            estimated_cost = self.cost_model.estimate_cost(update, observation_size)
            if cost_so_far + estimated_cost <= time_budget:
                prioritized.append(update)
                cost_so_far += estimated_cost

        return set(prioritized)

    def _create_execution_plan(
        self, updates: Set[UpdateType], signature: ObservationSignature
    ) -> UpdatePlan:
        """Create execution plan from required updates"""

        # Get execution order
        execution_groups = self.dependency_graph.get_execution_order(updates)

        immediate = []
        deferred = []
        parallel_groups = []

        total_estimated_time = 0
        observation_size = signature.variable_count * 100

        for group in execution_groups:
            # Check if group can be parallelized
            if len(group) > 1 and self.dependency_graph.can_parallelize(group):
                # Parallel execution
                parallel_groups.append(group)
                # Time is max of group
                group_time = max(
                    self.cost_model.estimate_cost(u, observation_size) for u in group
                )
                total_estimated_time += group_time
            else:
                # Sequential execution
                for update in group:
                    strategy = self.strategies[update]

                    if (
                        strategy.can_defer
                        and strategy.priority.value > UpdatePriority.HIGH.value
                    ):
                        deferred.append(update)
                    else:
                        immediate.append(update)
                        total_estimated_time += self.cost_model.estimate_cost(
                            update, observation_size
                        )

        # Create reasoning explanation
        reasoning = self._generate_reasoning(
            immediate, deferred, parallel_groups, signature
        )

        return UpdatePlan(
            immediate=immediate,
            deferred=deferred,
            parallel_groups=parallel_groups,
            estimated_time_ms=total_estimated_time,
            confidence=signature.confidence,
            reasoning=reasoning,
            metadata={"signature": signature},
        )

    def _execute_update(self, update_type: UpdateType) -> Dict[str, Any]:
        """Execute a single update - FIXED: Properly propagate error status"""
        start_time = time.time()

        try:
            if update_type == UpdateType.INTERVENTION:
                result = self._execute_intervention_update()
            elif update_type == UpdateType.CORRELATION:
                result = self._execute_correlation_update()
            elif update_type == UpdateType.DYNAMICS:
                result = self._execute_dynamics_update()
            elif update_type == UpdateType.CAUSAL:
                result = self._execute_causal_update()
            elif update_type == UpdateType.INVARIANT:
                result = self._execute_invariant_update()
            elif update_type == UpdateType.CONFIDENCE:
                result = self._execute_confidence_update()
            elif update_type == UpdateType.DISTRIBUTION:
                result = self._execute_distribution_update()
            elif update_type == UpdateType.STRUCTURAL:
                result = self._execute_structural_update()
            else:
                result = {"status": "unknown_update_type"}

            execution_time = (time.time() - start_time) * 1000

            # FIXED: Check if the inner result indicates an error
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error"),
                    "execution_time_ms": execution_time,
                }

            return {
                "status": "success",
                "result": result,
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            logger.error(f"Update {update_type} failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def _execute_intervention_update(self) -> Dict[str, Any]:
        """Execute intervention update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "intervention_manager"):
            return {"status": "no_intervention_manager"}

        manager = self.world_model.intervention_manager

        if not hasattr(manager, "process_intervention_observation"):
            if not hasattr(manager, "process_intervention"):
                return {"status": "no_process_intervention_method"}
            else:
                method_to_call = manager.process_intervention
        else:
            method_to_call = manager.process_intervention_observation

        try:
            # FIXED: Defensive parameter inference
            sig = inspect.signature(method_to_call)
            params = [p for p in sig.parameters.keys() if p != "self"]

            # Check for specific parameter names
            if "observation" in params:
                if (
                    "intervention_data" in params
                    and self.current_observation.intervention_data
                ):
                    return method_to_call(
                        intervention_data=self.current_observation.intervention_data,
                        observation=self.current_observation,
                    )
                return method_to_call(observation=self.current_observation)
            elif "intervention_data" in params:
                if self.current_observation.intervention_data:
                    return method_to_call(
                        intervention_data=self.current_observation.intervention_data
                    )
                else:
                    return method_to_call(intervention_data=self.current_observation)
            elif len(params) == 2:
                # Assume (intervention_data, observation)
                try:
                    return method_to_call(
                        self.current_observation.intervention_data,
                        self.current_observation,
                    )
                except Exception:
                    return method_to_call(self.current_observation, None)  # Fallback
            elif len(params) == 1:
                # Single non-self parameter - try positional
                try:
                    return method_to_call(self.current_observation)
                except TypeError as e:
                    logger.debug(f"Positional call failed: {e}, trying no-arg")
                    return method_to_call()
            else:
                # No obvious observation parameter or no parameters
                return method_to_call()
        except Exception as e:
            logger.error(f"Error in process_intervention: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_correlation_update(self) -> Dict[str, Any]:
        """Execute correlation update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "correlation_tracker"):
            return {"status": "no_correlation_tracker"}

        tracker = self.world_model.correlation_tracker

        if not hasattr(tracker, "update"):
            return {"status": "no_update_method"}

        try:
            # FIXED: Defensive parameter inference
            sig = inspect.signature(tracker.update)
            params = [p for p in sig.parameters.keys() if p != "self"]

            # Check for specific parameter names
            if "observation" in params:
                return tracker.update(observation=self.current_observation)
            elif len(params) == 1:
                # Single non-self parameter - try positional
                try:
                    return tracker.update(self.current_observation)
                except TypeError as e:
                    logger.debug(f"Positional call failed: {e}, trying no-arg")
                    return tracker.update()
            else:
                # No parameters or no obvious observation parameter
                return tracker.update()
        except Exception as e:
            logger.error(f"Error in correlation update: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_dynamics_update(self) -> Dict[str, Any]:
        """Execute dynamics update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "dynamics"):
            return {"status": "no_dynamics_model"}

        dynamics = self.world_model.dynamics

        if hasattr(dynamics, "update"):
            try:
                # FIXED: Defensive parameter inference
                sig = inspect.signature(dynamics.update)
                params = [p for p in sig.parameters.keys() if p != "self"]

                # Check for specific parameter names
                if "observation" in params:
                    return dynamics.update(observation=self.current_observation)
                elif len(params) == 1:
                    # Single non-self parameter - try positional
                    try:
                        return dynamics.update(self.current_observation)
                    except TypeError as e:
                        logger.debug(f"Positional call failed: {e}, trying no-arg")
                        return dynamics.update()
                else:
                    # No parameters or no obvious observation parameter
                    return dynamics.update()
            except Exception as e:
                logger.error(f"Error in dynamics update: {e}")
                return {"status": "error", "error": str(e)}
        elif hasattr(dynamics, "update_from_observation"):
            try:
                return dynamics.update_from_observation(self.current_observation)
            except Exception as e:
                logger.error(f"Error in dynamics update_from_observation: {e}")
                return {"status": "error", "error": str(e)}

        return {"status": "no_update_method"}

    def _execute_causal_update(self) -> Dict[str, Any]:
        """Execute causal update"""
        # This is often triggered by intervention, which handles the graph update.
        # If a separate update is needed, it would be called here.
        if hasattr(self.world_model, "causal_graph"):
            # Example: self.world_model.causal_graph.update_from_observation(self.current_observation)
            return {
                "status": "causal_graph_updated"
            }  # Assuming intervention handled it
        return {"status": "no_causal_graph"}

    def _execute_invariant_update(self) -> Dict[str, Any]:
        """Execute invariant update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "invariant_detector"):
            return {"status": "no_invariant_detector"}

        detector = self.world_model.invariant_detector

        if not hasattr(detector, "check"):
            return {"status": "no_check_method"}

        try:
            # FIXED: Defensive parameter inference
            sig = inspect.signature(detector.check)
            params = [p for p in sig.parameters.keys() if p != "self"]

            # Check for specific parameter names
            if "observations" in params:
                # Plural form - expects list
                if self.current_observation:
                    return detector.check(observations=[self.current_observation])
                else:
                    return detector.check(observations=[])
            elif "observation" in params:
                # Singular form
                return detector.check(observation=self.current_observation)
            elif len(params) == 1:
                # Single non-self parameter - try with list
                try:
                    if self.current_observation:
                        return detector.check([self.current_observation])
                    else:
                        return detector.check([])
                except TypeError as e:
                    logger.debug(f"List call failed: {e}, trying no-arg")
                    return detector.check()
            else:
                # No parameters
                return detector.check()
        except Exception as e:
            logger.error(f"Error in invariant check: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_confidence_update(self) -> Dict[str, Any]:
        """Execute confidence update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "confidence_tracker"):
            return {"status": "no_confidence_tracker"}

        tracker = self.world_model.confidence_tracker

        if not hasattr(tracker, "update"):
            return {"status": "no_update_method"}

        try:
            # FIXED: Defensive parameter inference
            sig = inspect.signature(tracker.update)
            params = [p for p in sig.parameters.keys() if p != "self"]

            # Check for specific parameter names
            if "observation" in params and "prediction" in params:
                # Both parameters present
                return tracker.update(
                    observation=self.current_observation, prediction=None
                )
            elif "observation" in params:
                # Only observation
                return tracker.update(observation=self.current_observation)
            elif len(params) == 2:
                # Two non-self parameters - assume observation and prediction
                try:
                    return tracker.update(self.current_observation, None)
                except TypeError as e:
                    logger.debug(f"Two-arg call failed: {e}, trying no-arg")
                    return tracker.update()
            elif len(params) == 1:
                # Single non-self parameter
                try:
                    return tracker.update(self.current_observation)
                except TypeError as e:
                    logger.debug(f"Single-arg call failed: {e}, trying no-arg")
                    return tracker.update()
            else:
                # No parameters
                return tracker.update()
        except Exception as e:
            logger.error(f"Error in confidence update: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_distribution_update(self) -> Dict[str, Any]:
        """Execute distribution update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "distribution_monitor"):
            # Fallback to correlation tracker if it has distribution methods
            if hasattr(self.world_model, "correlation_tracker") and hasattr(
                self.world_model.correlation_tracker, "update_distributions"
            ):
                monitor = self.world_model.correlation_tracker
                method_name = "update_distributions"
            else:
                return {"status": "no_distribution_monitor"}
        else:
            monitor = self.world_model.distribution_monitor
            method_name = "update"
            if not hasattr(monitor, "update"):
                return {"status": "no_update_method"}

        method_to_call = getattr(monitor, method_name)

        try:
            # FIXED: Defensive parameter inference
            sig = inspect.signature(method_to_call)
            params = [p for p in sig.parameters.keys() if p != "self"]

            # Check for specific parameter names
            if "observation" in params:
                return method_to_call(observation=self.current_observation)
            elif len(params) == 1:
                # Single non-self parameter - try positional
                try:
                    return method_to_call(self.current_observation)
                except TypeError as e:
                    logger.debug(f"Positional call failed: {e}, trying no-arg")
                    return method_to_call()
            else:
                # No parameters
                return method_to_call()
        except Exception as e:
            logger.error(f"Error in distribution update: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_structural_update(self) -> Dict[str, Any]:
        """Execute structural update - FIXED: Defensive parameter inference"""
        if not hasattr(self.world_model, "structural_analyzer"):
            # Fallback to causal graph if it has structural methods
            if hasattr(self.world_model, "causal_graph") and hasattr(
                self.world_model.causal_graph, "analyze_structure"
            ):
                analyzer = self.world_model.causal_graph
                method_name = "analyze_structure"
            else:
                return {"status": "no_structural_analyzer"}
        else:
            analyzer = self.world_model.structural_analyzer
            method_name = "analyze"
            if not hasattr(analyzer, "analyze"):
                return {"status": "no_analyze_method"}

        method_to_call = getattr(analyzer, method_name)

        try:
            # FIXED: Defensive parameter inference
            sig = inspect.signature(method_to_call)
            params = [p for p in sig.parameters.keys() if p != "self"]

            # Check for specific parameter names
            if "observation" in params:
                return method_to_call(observation=self.current_observation)
            elif len(params) == 1:
                # Single non-self parameter - try positional
                try:
                    return method_to_call(self.current_observation)
                except TypeError as e:
                    logger.debug(f"Positional call failed: {e}, trying no-arg")
                    return method_to_call()
            else:
                # No parameters
                return method_to_call()
        except Exception as e:
            logger.error(f"Error in structural analysis: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_parallel_group(self, group: List[UpdateType]) -> Dict[str, Any]:
        """Execute updates in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(max_workers=len(group)) as executor:
            future_to_update = {
                executor.submit(self._execute_update, update): update
                for update in group
            }

            for future in as_completed(future_to_update):
                update = future_to_update[future]
                try:
                    result = future.result()
                    results[update.value] = result
                except Exception as e:
                    logger.error(f"Parallel execution of {update} failed: {e}")
                    results[update.value] = {"status": "error", "error": str(e)}

        return results

    def _schedule_deferred(self, updates: List[UpdateType]):
        """Schedule deferred updates for later execution"""
        # This would integrate with a task queue or scheduler
        for update in updates:
            self.metrics[f"deferred_{update.value}"] += 1
            logger.debug(f"Deferred update {update} for later execution")

    def _detect_anomaly(self, observation: Any) -> bool:
        """Detect if observation contains anomalies"""
        if hasattr(observation, "is_anomaly"):
            return observation.is_anomaly

        if hasattr(observation, "metadata") and observation.metadata:
            if observation.metadata.get("is_anomaly", False):
                return True

        # Simple statistical check
        if hasattr(observation, "variables") and hasattr(
            self.world_model, "correlation_tracker"
        ):
            try:
                for var, value in observation.variables.items():
                    if isinstance(value, (int, float)):
                        if hasattr(
                            self.world_model.correlation_tracker, "get_baseline"
                        ):
                            baseline = (
                                self.world_model.correlation_tracker.get_baseline(var)
                            )
                            if baseline is not None and hasattr(
                                self.world_model.correlation_tracker, "get_noise_level"
                            ):
                                noise_level = self.world_model.correlation_tracker.get_noise_level(
                                    var
                                )
                                if (
                                    noise_level > 0
                                    and abs(value - baseline) > 3 * noise_level
                                ):
                                    return True
            except Exception as e:
                logger.debug(f"Error in anomaly detection: {e}")

        return False

    def _detect_structural_change(self, observation: Any) -> bool:
        """Detect if observation indicates structural change"""
        if hasattr(observation, "structural_change"):
            return observation.structural_change

        if hasattr(observation, "metadata") and observation.metadata:
            if observation.metadata.get("structural_change", False):
                return True

        # Check for new variables or relationships
        if hasattr(observation, "variables") and hasattr(
            self.world_model, "correlation_tracker"
        ):
            try:
                current_vars = set(observation.variables.keys())
                if hasattr(self.world_model.correlation_tracker, "correlation_matrix"):
                    known_vars = set(
                        getattr(
                            self.world_model.correlation_tracker.correlation_matrix,
                            "variables",
                            [],
                        )
                    )

                    if current_vars - known_vars:  # New variables
                        return True
            except Exception as e:
                logger.debug(f"Error in structural change detection: {e}")

        return False

    def _generate_reasoning(
        self,
        immediate: List[UpdateType],
        deferred: List[UpdateType],
        parallel: List[List[UpdateType]],
        signature: ObservationSignature,
    ) -> str:
        """Generate human-readable reasoning for the plan"""

        reasons = []

        if signature.has_intervention:
            reasons.append("Intervention detected - updating causal graph")

        if signature.has_anomaly:
            reasons.append("Anomaly detected - checking invariants")

        if signature.has_structural_change:
            reasons.append("Structural change detected - comprehensive update required")

        if immediate:
            reasons.append(f"Immediate updates: {[u.value for u in immediate]}")

        if parallel:
            reasons.append(
                f"Parallel updates possible: {[[u.value for u in g] for g in parallel]}"
            )

        if deferred:
            reasons.append(f"Deferred for efficiency: {[u.value for u in deferred]}")

        return "; ".join(reasons) if reasons else "Standard update cycle"

    def _get_cache_key(self, signature: ObservationSignature) -> str:
        """Generate cache key from signature"""
        return signature.pattern_hash

    def _validate_signature_safety(
        self, signature: ObservationSignature
    ) -> Dict[str, Any]:
        """Validate signature for safety issues"""
        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check confidence bounds
        if signature.confidence < 0 or signature.confidence > 1:
            violations.append(f"Invalid confidence: {signature.confidence}")

        # Check variable count bounds
        if signature.variable_count < 0:
            violations.append(f"Invalid variable count: {signature.variable_count}")

        if signature.variable_count > 10000:
            violations.append(f"Excessive variable count: {signature.variable_count}")

        # Check time delta
        if signature.time_delta is not None:
            if signature.time_delta < 0:
                violations.append(f"Negative time delta: {signature.time_delta}")
            if signature.time_delta > 86400:  # More than 1 day
                violations.append(f"Excessive time delta: {signature.time_delta}")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _apply_signature_corrections(
        self, signature: ObservationSignature, safety_result: Dict[str, Any]
    ) -> ObservationSignature:
        """Apply safety corrections to signature"""
        # Clamp confidence
        signature.confidence = np.clip(signature.confidence, 0, 1)

        # Clamp variable count
        signature.variable_count = max(0, min(10000, signature.variable_count))

        # Clamp time delta
        if signature.time_delta is not None:
            signature.time_delta = max(0, min(86400, signature.time_delta))

        return signature

    def _validate_plan_safety(
        self, plan: UpdatePlan, observation: Any
    ) -> Dict[str, Any]:
        """Validate execution plan for safety"""
        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check for excessive updates
        total_updates = len(plan.immediate) + len(plan.deferred)
        for group in plan.parallel_groups:
            total_updates += len(group)

        if total_updates > 50:
            violations.append(f"Excessive updates planned: {total_updates}")

        # Check estimated time
        if plan.estimated_time_ms > 60000:  # More than 1 minute
            violations.append(f"Excessive estimated time: {plan.estimated_time_ms}ms")

        # Check confidence
        if plan.confidence < 0 or plan.confidence > 1:
            violations.append(f"Invalid plan confidence: {plan.confidence}")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _apply_plan_corrections(
        self, plan: UpdatePlan, safety_result: Dict[str, Any]
    ) -> UpdatePlan:
        """Apply safety corrections to execution plan"""
        # Limit number of updates
        total_updates = len(plan.immediate) + sum(len(g) for g in plan.parallel_groups)
        if total_updates > 50:
            # Keep only critical updates
            plan.immediate = [
                u
                for u in plan.immediate
                if self.strategies[u].priority == UpdatePriority.CRITICAL
            ]
            plan.parallel_groups = []

        # Clamp estimated time
        plan.estimated_time_ms = min(60000, plan.estimated_time_ms)

        # Clamp confidence
        plan.confidence = np.clip(plan.confidence, 0, 1)

        return plan

    def _validate_update_safety(
        self, update_type: UpdateType, plan: UpdatePlan
    ) -> Dict[str, Any]:
        """Validate individual update for safety"""
        if not self.safety_validator:
            return {"safe": True}

        # All updates are considered safe by default
        # Specific validation would happen in the actual update execution
        return {"safe": True}

    def _validate_result_safety(
        self, update_type: UpdateType, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate update result for safety"""
        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check for error status
        if result.get("status") == "error":
            violations.append(f"Update {update_type.value} returned error status")

        # Check execution time
        exec_time = result.get("execution_time_ms", 0)
        if exec_time > 10000:  # More than 10 seconds
            violations.append(
                f"Update {update_type.value} took excessive time: {exec_time}ms"
            )

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _apply_result_corrections(
        self, result: Dict[str, Any], safety_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply safety corrections to update result"""
        # Mark result as corrected
        result["safety_corrected"] = True
        result["correction_reason"] = safety_result["reason"]

        # Clamp execution time
        if "execution_time_ms" in result:
            result["execution_time_ms"] = min(10000, result["execution_time_ms"])

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics"""
        metrics = {
            "plans_created": self.metrics["plans_created"],
            "cache_hits": self.metrics["cache_hits"],
            "avg_planning_time_ms": self.metrics.get("avg_planning_time_ms", 0),
            "learned_patterns_used": self.metrics["learned_patterns_used"],
            "deferred_updates": {
                k.replace("deferred_", ""): v
                for k, v in self.metrics.items()
                if k.startswith("deferred_")
            },
            "pattern_rules": len(self.pattern_learner.rules),
            "cache_size": len(self.cache),
        }

        # Add safety metrics
        if self.safety_validator:
            metrics["safety"] = {
                "enabled": True,
                "blocks": dict(self.safety_blocks),
                "corrections": dict(self.safety_corrections),
                "total_blocks": sum(self.safety_blocks.values()),
                "total_corrections": sum(self.safety_corrections.values()),
            }
        else:
            metrics["safety"] = {"enabled": False}

        # Add CSIU metrics if available (internal only)
        if self.self_improvement_drive and getattr(
            self.self_improvement_drive, "_csiu_enabled", False
        ):
            metrics["csiu_tracking"] = {
                "enabled": True,
                "routing_records": len(self.csiu_routing_history),
                "execution_records": len(
                    [r for r in self.execution_history if "csiu_before" in r]
                ),
                "outcome_records": len(
                    [r for r in self.csiu_routing_history if "success" in r]
                ),
                "note": "Internal tracking only, not for UX",
            }

        return metrics

    def save_state(self, path: str):
        """Save router state for persistence - FIXED: Use Path not FilePath"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        state = {
            "metrics": dict(self.metrics),
            "pattern_rules": self.pattern_learner.rules,
            "cost_history": {
                k.value: list(v) for k, v in self.cost_model.cost_history.items()
            },
            "base_costs": {k.value: v for k, v in self.cost_model.base_costs.items()},
            "safety_blocks": dict(self.safety_blocks),
            "safety_corrections": dict(self.safety_corrections),
        }

        # Don't save CSIU data (internal only, ephemeral)

        with open(save_path / "router_state.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Router state saved to {save_path}")

    def load_state(self, path: str):
        """Load router state - FIXED: Merge with defaults and use Path not FilePath
        SECURITY: Use safe_pickle_load to prevent deserialization attacks"""
        load_path = Path(path) / "router_state.pkl"

        if not load_path.exists():
            logger.warning(f"No saved state at {load_path}")
            return

        # SECURITY FIX: Use safe_pickle_load instead of pickle.load
        state = safe_pickle_load(load_path)

        self.metrics = defaultdict(int, state["metrics"])
        self.pattern_learner.rules = state["pattern_rules"]

        # Convert cost history keys back to enums
        for key_str, values in state["cost_history"].items():
            try:
                key = UpdateType(key_str)
                self.cost_model.cost_history[key] = deque(values, maxlen=100)
            except ValueError:
                logger.warning(f"Unknown UpdateType in saved state: {key_str}")

        # FIXED: Merge base costs with defaults, don't replace
        for key_str, value in state["base_costs"].items():
            try:
                key = UpdateType(key_str)
                self.cost_model.base_costs[key] = value
            except ValueError:
                logger.warning(f"Unknown UpdateType in saved base_costs: {key_str}")
        # Keep any new UpdateTypes from current version that weren't in saved state

        # Load safety tracking
        if "safety_blocks" in state:
            self.safety_blocks = defaultdict(int, state["safety_blocks"])
        if "safety_corrections" in state:
            self.safety_corrections = defaultdict(int, state["safety_corrections"])

        logger.info(f"Router state loaded from {load_path}")
