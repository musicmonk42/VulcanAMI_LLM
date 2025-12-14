"""
objective_negotiator.py - Multi-agent objective negotiation
Part of the meta_reasoning subsystem for VULCAN-AMI

Resolves conflicts between competing objectives through negotiation:
- Multi-agent proposal negotiation
- Pareto frontier identification
- Conflict resolution through compromise
- Dynamic objective weighting
- Constraint validation

Enables autonomous agent consensus on trade-offs without human intervention.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# import numpy as np # Original import
# FIXED: Added Union to the import
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, Mock  # FIXED: Import Mock

# --- START FIX: Add numpy fallback ---
# logger = logging.getLogger(__name__) # Original logger placement
logger = logging.getLogger(__name__)  # Moved logger init up
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using list-based math")

    class FakeNumpy:
        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0  # Return float

        @staticmethod
        def array(lst):
            return list(lst)

        @staticmethod
        def var(lst):
            if not lst or len(lst) < 2:
                return 0.0
            mean_val = sum(lst) / len(lst)
            return sum((x - mean_val) ** 2 for x in lst) / len(
                lst
            )  # Population variance

        @staticmethod
        def linspace(
            start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
        ):
            if num <= 0:
                return []
            if endpoint:
                if num == 1:
                    return [start]
                step = (stop - start) / (num - 1)
            else:
                step = (stop - start) / num
            result = [start + i * step for i in range(num)]
            if retstep:
                return result, step
            return result

        # Add other numpy functions if needed later

    np = FakeNumpy()
# --- END FIX ---


# Assuming ObjectiveHierarchy and Objective are importable from sibling modules
try:
    # Use real imports if available
    from .objective_hierarchy import ConflictType as RealHierarchyConflictType
    from .objective_hierarchy import Objective as RealObjective
    from .objective_hierarchy import ObjectiveHierarchy as RealObjectiveHierarchy
    from .objective_hierarchy import ObjectiveType as RealObjectiveType

    ObjectiveHierarchy = RealObjectiveHierarchy
    Objective = RealObjective
    ObjectiveType = RealObjectiveType
    HierarchyConflictType = RealHierarchyConflictType
    OBJECTIVE_HIERARCHY_AVAILABLE = True
    logger.info("Successfully imported objective_hierarchy.")
except ImportError:
    # Fallback using MagicMock if import fails
    OBJECTIVE_HIERARCHY_AVAILABLE = False
    logger.error("Failed to import objective_hierarchy. Using MagicMock fallback.")
    ObjectiveHierarchy = MagicMock()
    # Mock methods expected by ObjectiveNegotiator
    ObjectiveHierarchy.return_value.objectives = {}
    ObjectiveHierarchy.return_value.find_conflicts.return_value = None
    ObjectiveHierarchy.return_value.get_priority_order.return_value = []

    # Define fallback Enums/Classes needed
    class Objective:
        pass

    class ObjectiveType(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        DERIVED = "derived"

    class HierarchyConflictType(Enum):
        DIRECT = "direct"
        INDIRECT = "indirect"
        CONSTRAINT = "constraint"
        TRADEOFF = "tradeoff"


class NegotiationStrategy(Enum):
    """Strategy for objective negotiation"""

    PARETO_OPTIMAL = "pareto_optimal"
    WEIGHTED_AVERAGE = "weighted_average"
    LEXICOGRAPHIC = "lexicographic"
    NASH_BARGAINING = "nash_bargaining"
    MINIMAX = "minimax"


class NegotiationOutcome(Enum):
    """Outcome of negotiation"""

    CONSENSUS = "consensus"
    COMPROMISE = "compromise"
    DEADLOCK = "deadlock"
    DOMINATED = "dominated"


@dataclass
class AgentProposal:
    """Proposal from a single agent"""

    agent_id: str
    objective: str
    target_value: float
    weight: float
    constraints: Dict[str, Any] = field(default_factory=dict)
    flexibility: float = 0.5  # How willing to compromise (0-1)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NegotiationResult:
    """Result of negotiation process"""

    outcome: NegotiationOutcome
    agreed_objectives: Dict[str, float]
    objective_weights: Dict[str, float]
    participating_agents: List[str]
    strategy_used: NegotiationStrategy
    iterations: int
    convergence_time_ms: float
    compromises_made: List[Dict[str, Any]]
    pareto_optimal: bool
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Resolution of objective conflict"""

    objectives: List[str]
    resolution_type: str
    agreed_weights: Dict[str, float]
    expected_outcomes: Dict[str, float]
    tradeoffs: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObjectiveNegotiator:
    """
    Resolves conflicts between competing objectives through negotiation

    Enables multi-agent systems to autonomously negotiate trade-offs:
    - Agents propose different objectives
    - System finds Pareto-optimal compromises
    - Dynamic weighting based on system state
    - Validates negotiated objectives respect constraints

    This is machine-to-machine negotiation using structured protocols,
    not natural language dialogue.
    """

    def __init__(
        self, objective_hierarchy=None, world_model=None
    ):  # Made args optional
        """
        Initialize objective negotiator

        Args:
            objective_hierarchy: ObjectiveHierarchy instance (optional)
            world_model: Reference to WorldModel instance (optional)
        """
        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        # Handle optional objective_hierarchy with fallback
        if objective_hierarchy is None:
            if not OBJECTIVE_HIERARCHY_AVAILABLE:
                logger.warning(
                    "No ObjectiveHierarchy provided and import failed. Using MagicMock fallback."
                )
                self.objective_hierarchy = ObjectiveHierarchy()  # Instantiates the mock
            else:
                logger.warning(
                    "No ObjectiveHierarchy provided. Using MagicMock fallback for now."
                )
                self.objective_hierarchy = ObjectiveHierarchy()  # Instantiates the mock

        # --- START FIX: Allow Mock objects from tests ---
        elif OBJECTIVE_HIERARCHY_AVAILABLE and not isinstance(
            objective_hierarchy, (RealObjectiveHierarchy, Mock, MagicMock)
        ):
            raise TypeError(
                f"objective_hierarchy must be an instance of ObjectiveHierarchy or a Mock, but got {type(objective_hierarchy)}"
            )
        # --- END FIX ---

        else:
            self.objective_hierarchy = objective_hierarchy

        # Handle optional world_model with fallback
        self.world_model = world_model or MagicMock()  # Use MagicMock if None

        # Negotiation parameters
        self.max_iterations = 100
        self.convergence_threshold = 0.01
        self.fairness_weight = 0.3

        # Negotiation history
        self.negotiation_history = deque(maxlen=500)

        # Strategy performance tracking
        self.strategy_performance = defaultdict(lambda: {"successes": 0, "failures": 0})

        # Statistics
        self.stats = defaultdict(int)

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ObjectiveNegotiator initialized")

    def negotiate_multi_agent_proposals(
        self, proposals: List[Dict[str, Any]]
    ) -> NegotiationResult:
        """
        Negotiate between multiple agent proposals

        Agents propose different objectives with different priorities.
        System finds consensus or acceptable compromise.

        Args:
            proposals: List of agent proposals

        Returns:
            Negotiation result with agreed objectives
        """

        with self.lock:
            start_time = time.time()

            # Ensure proposals is a list of dicts
            if not isinstance(proposals, list) or not all(
                isinstance(p, dict) for p in proposals
            ):
                logger.error(
                    f"Invalid proposals format: expected List[Dict], got {type(proposals)}"
                )
                return self._create_deadlock_result([], "Invalid proposals format")

            # Parse proposals into structured format
            agent_proposals = [self._parse_proposal(p) for p in proposals]

            if not agent_proposals:
                return self._empty_negotiation_result()

            # EXAMINE: Analyze proposal space
            analysis = self._analyze_proposal_space(agent_proposals)

            # SELECT: Choose negotiation strategy
            strategy = self._select_negotiation_strategy(agent_proposals, analysis)
            logger.info(f"Selected negotiation strategy: {strategy.value}")

            # APPLY: Execute negotiation
            try:
                if strategy == NegotiationStrategy.PARETO_OPTIMAL:
                    result = self._negotiate_via_pareto(agent_proposals, analysis)
                elif strategy == NegotiationStrategy.WEIGHTED_AVERAGE:
                    result = self._negotiate_via_weighted_average(
                        agent_proposals, analysis
                    )
                elif strategy == NegotiationStrategy.NASH_BARGAINING:
                    result = self._negotiate_via_nash_bargaining(
                        agent_proposals, analysis
                    )
                elif strategy == NegotiationStrategy.LEXICOGRAPHIC:
                    result = self._negotiate_via_lexicographic(
                        agent_proposals, analysis
                    )
                else:  # Default or MINIMAX
                    result = self._negotiate_via_minimax(agent_proposals, analysis)
            except Exception as e:
                logger.error(
                    f"Negotiation execution failed with strategy {strategy.value}: {e}",
                    exc_info=True,
                )
                result = self._create_deadlock_result(
                    agent_proposals, f"Negotiation execution error: {e}"
                )

            # Add metadata
            result.strategy_used = strategy
            result.convergence_time_ms = (time.time() - start_time) * 1000
            result.metadata["analysis"] = analysis

            # Validate result against constraints
            if result.outcome != NegotiationOutcome.DEADLOCK:
                # FIXED: Call validation and update result if invalid
                is_valid, validation_reason = self.validate_negotiated_objectives(
                    result.agreed_objectives, return_reason=True
                )
                if not is_valid:
                    logger.warning(
                        f"Negotiated objectives failed validation: {validation_reason}"
                    )
                    result.outcome = NegotiationOutcome.DEADLOCK
                    result.confidence *= 0.5  # Reduce confidence
                    # FIXED: Update reasoning to reflect validation failure
                    result.reasoning = (
                        f"Negotiation deadlock: Validation failed - {validation_reason}"
                    )
                else:
                    logger.info("Negotiated objectives passed validation.")

            # REMEMBER: Track negotiation
            self.negotiation_history.append(result)
            self.stats["negotiations_performed"] += 1
            self.stats[f"outcome_{result.outcome.value}"] += 1

            # Update strategy performance
            if result.outcome in [
                NegotiationOutcome.CONSENSUS,
                NegotiationOutcome.COMPROMISE,
            ]:
                self.strategy_performance[strategy]["successes"] += 1
            else:  # DEADLOCK or DOMINATED
                self.strategy_performance[strategy]["failures"] += 1

            logger.info(
                f"Negotiation completed. Outcome: {result.outcome.value}, Confidence: {result.confidence:.2f}"
            )

            return result

    def find_pareto_frontier(
        self, objective_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find Pareto frontier in multi-objective space

        Args:
            objective_space: Dict containing 'objectives' list

        Returns:
            List of Pareto-optimal points (dicts)
        """

        with self.lock:
            # Validate input
            if (
                not isinstance(objective_space, dict)
                or "objectives" not in objective_space
                or not isinstance(objective_space["objectives"], list)
            ):
                logger.error("Invalid objective_space format for find_pareto_frontier.")
                return []

            objectives = objective_space.get("objectives", [])
            if not objectives:
                logger.info("No objectives provided for Pareto frontier calculation.")
                return []

            # Use counterfactual reasoner if available and functional
            reasoner = None
            if hasattr(self.world_model, "motivational_introspection"):
                mi = self.world_model.motivational_introspection
                # Check if mi is not a mock and has the reasoner
                if not isinstance(mi, MagicMock) and hasattr(
                    mi, "counterfactual_reasoner"
                ):
                    # Check if reasoner itself is not a mock and has the method
                    potential_reasoner = mi.counterfactual_reasoner
                    if not isinstance(potential_reasoner, MagicMock) and hasattr(
                        potential_reasoner, "find_pareto_frontier"
                    ):
                        reasoner = potential_reasoner

            if reasoner:
                try:
                    logger.debug(
                        f"Using counterfactual_reasoner to find Pareto frontier for {objectives}"
                    )
                    pareto_points_objects = reasoner.find_pareto_frontier(objectives)

                    # Convert ParetoPoint objects (or whatever reasoner returns) to dictionary format safely
                    result_list = []
                    for point in pareto_points_objects:
                        # Assuming point has attributes like .objectives, .objective_weights etc.
                        point_dict = {
                            "objectives": getattr(point, "objectives", {}),
                            "weights": getattr(point, "objective_weights", {}),
                            "is_pareto_optimal": getattr(
                                point, "is_pareto_optimal", False
                            ),
                            "dominates": getattr(point, "dominates", []),
                            "dominated_by": getattr(point, "dominated_by", []),
                        }
                        result_list.append(point_dict)
                    return result_list
                except Exception as e:
                    logger.error(
                        f"Counterfactual reasoner failed during Pareto calculation: {e}",
                        exc_info=True,
                    )
                    # Fall through to simple computation

            # Fallback: simple Pareto computation
            logger.warning("Falling back to simple Pareto computation.")
            return self._compute_pareto_frontier_simple(objectives, objective_space)

    def resolve_objective_conflict(
        self, obj_a: str, obj_b: str, context: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve conflict between two objectives

        Finds acceptable middle ground using prediction engine
        to estimate trade-offs.

        Args:
            obj_a: First objective name
            obj_b: Second objective name
            context: Context for resolution

        Returns:
            ConflictResolution object
        """

        with self.lock:
            start_time = time.time()

            # Check if objectives actually conflict (mock-safe)
            conflict = None
            if hasattr(self.objective_hierarchy, "find_conflicts") and callable(
                self.objective_hierarchy.find_conflicts
            ):
                conflict = self.objective_hierarchy.find_conflicts(obj_a, obj_b)

            if not conflict:
                logger.info(
                    f"No conflict found between '{obj_a}' and '{obj_b}'. Resolution not needed."
                )
                # No conflict - assume both can be maximally satisfied (or use base weights?)
                base_weight_a = (
                    getattr(
                        self.objective_hierarchy.objectives.get(obj_a), "weight", 1.0
                    )
                    if obj_a in getattr(self.objective_hierarchy, "objectives", {})
                    else 1.0
                )
                base_weight_b = (
                    getattr(
                        self.objective_hierarchy.objectives.get(obj_b), "weight", 1.0
                    )
                    if obj_b in getattr(self.objective_hierarchy, "objectives", {})
                    else 1.0
                )
                total_w = base_weight_a + base_weight_b
                norm_w_a = base_weight_a / total_w if total_w > 0 else 0.5
                norm_w_b = base_weight_b / total_w if total_w > 0 else 0.5

                return ConflictResolution(
                    objectives=[obj_a, obj_b],
                    resolution_type="no_conflict",
                    agreed_weights={
                        obj_a: norm_w_a,
                        obj_b: norm_w_b,
                    },  # Use normalized base weights
                    # Estimate outcomes assuming no conflict (e.g., target or high value)
                    expected_outcomes={
                        obj_a: (
                            getattr(
                                self.objective_hierarchy.objectives.get(obj_a),
                                "target_value",
                                1.0,
                            )
                            if obj_a
                            in getattr(self.objective_hierarchy, "objectives", {})
                            else 1.0
                        ),
                        obj_b: (
                            getattr(
                                self.objective_hierarchy.objectives.get(obj_b),
                                "target_value",
                                1.0,
                            )
                            if obj_b
                            in getattr(self.objective_hierarchy, "objectives", {})
                            else 1.0
                        ),
                    },
                    tradeoffs=[],
                    confidence=1.0,
                    reasoning="Objectives do not conflict based on hierarchy definition.",
                    metadata={"context": context},
                )

            logger.info(f"Resolving conflict between '{obj_a}' and '{obj_b}'...")
            # Analyze trade-off curve
            tradeoffs = self._analyze_binary_tradeoff(obj_a, obj_b, context)

            # Find optimal compromise point
            optimal_point_weights = self._find_optimal_compromise(
                obj_a, obj_b, tradeoffs, context
            )

            # Estimate expected outcomes at the optimal point
            expected_outcomes = self._estimate_outcomes_at_point(
                obj_a, obj_b, optimal_point_weights, context
            )

            # Calculate confidence
            confidence = self._calculate_resolution_confidence(
                tradeoffs, optimal_point_weights, expected_outcomes
            )

            # Generate reasoning
            reasoning = self._generate_resolution_reasoning(
                obj_a, obj_b, optimal_point_weights, expected_outcomes, tradeoffs
            )

            resolution = ConflictResolution(
                objectives=[obj_a, obj_b],
                resolution_type=conflict.get(
                    "type", "unknown_conflict"
                ),  # Get type from conflict dict
                agreed_weights=optimal_point_weights,
                expected_outcomes=expected_outcomes,
                tradeoffs=tradeoffs,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "conflict_details": conflict,  # Store original conflict info
                    "computation_time_ms": (time.time() - start_time) * 1000,
                    "context": context,
                },
            )

            self.stats["conflicts_resolved"] += 1
            logger.info(
                f"Conflict resolved. Agreed weights: {resolution.agreed_weights}"
            )

            return resolution

    def dynamic_objective_weighting(
        self, system_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Adjust objective priorities based on current system state

        Examples:
        - Prioritize safety when uncertainty is high
        - Prioritize efficiency when resources are scarce
        - Prioritize accuracy when stakes are high

        Args:
            system_state: Current system state dict

        Returns:
            Adjusted objective weights dict (normalized to sum to 1.0)
        """

        with self.lock:
            # Ensure system_state is a dict
            if not isinstance(system_state, dict):
                logger.warning(
                    "Invalid system_state type provided for dynamic weighting. Returning base weights."
                )
                system_state = {}  # Use empty dict to avoid errors below

            # Get base weights from objective hierarchy (mock-safe)
            base_weights = {}
            # Use mock-safe attribute access
            hierarchy_objectives = {}
            if hasattr(self.objective_hierarchy, "objectives") and isinstance(
                self.objective_hierarchy.objectives, dict
            ):
                hierarchy_objectives = self.objective_hierarchy.objectives

            for obj_name, obj_data in hierarchy_objectives.items():
                # Safely get weight
                weight = 1.0  # Default weight
                if hasattr(obj_data, "weight"):
                    weight = getattr(obj_data, "weight", 1.0)
                elif isinstance(obj_data, dict):
                    weight = obj_data.get("weight", 1.0)
                base_weights[obj_name] = float(weight)  # Ensure float

            if not base_weights:
                logger.warning(
                    "No objectives found in hierarchy. Cannot perform dynamic weighting."
                )
                return {}

            # Adjust based on system state
            adjusted_weights = base_weights.copy()

            # --- Rule-based adjustments ---
            # Ensure values from state are numeric, provide defaults
            uncertainty = float(system_state.get("uncertainty", 0.0))
            resources = float(system_state.get("available_resources", 1.0))
            confidence_required = float(system_state.get("confidence_required", 0.5))
            performance_trend = float(system_state.get("performance_trend", 0.0))

            # Rule 1: Increase safety weight when uncertainty is high
            if "safety" in adjusted_weights and uncertainty > 0.7:
                adjusted_weights["safety"] *= (
                    1.0 + (uncertainty - 0.7) * 2
                )  # Scale increase

            # Rule 2: Increase efficiency when resources are low
            if "efficiency" in adjusted_weights and resources < 0.3:
                adjusted_weights["efficiency"] *= (
                    1.0 + (0.3 - resources) * 3
                )  # Scale increase

            # Rule 3: Increase accuracy when confidence is needed
            # Use 'prediction_accuracy' or 'accuracy' if present
            accuracy_key = next(
                (
                    k
                    for k in ["prediction_accuracy", "accuracy"]
                    if k in adjusted_weights
                ),
                None,
            )
            if accuracy_key and confidence_required > 0.8:
                adjusted_weights[accuracy_key] *= (
                    1.0 + (confidence_required - 0.8) * 2.5
                )  # Scale increase

            # Rule 4: Increase exploration when performance plateaus
            exploration_key = next(
                (k for k in ["exploration", "curiosity"] if k in adjusted_weights), None
            )  # Allow 'curiosity' too
            if exploration_key and abs(performance_trend) < 0.01:
                adjusted_weights[exploration_key] *= 1.5

            # --- Normalization preserving critical objective minimums ---
            critical_objectives = {}
            critical_minimum = 0.1  # Minimum weight allocation for critical objectives
            for obj_name, obj_data in hierarchy_objectives.items():
                # Safely get priority
                priority = 1  # Default non-critical
                if hasattr(obj_data, "priority"):
                    priority = getattr(obj_data, "priority", 1)
                elif isinstance(obj_data, dict):
                    priority = obj_data.get("priority", 1)

                if priority == 0 and obj_name in adjusted_weights:  # Critical
                    critical_objectives[obj_name] = critical_minimum

            # Calculate total critical weight requirement
            total_critical_minimum = sum(critical_objectives.values())

            # If critical objectives require more than 100%, something is wrong (too many critical?)
            # Log warning and scale them down proportionally to fit into 1.0
            if total_critical_minimum > 1.0:
                logger.warning(
                    f"Total minimum weight for critical objectives ({total_critical_minimum:.2f}) exceeds 1.0. Scaling down."
                )
                scale_factor = 1.0 / total_critical_minimum
                critical_objectives = {
                    k: v * scale_factor for k, v in critical_objectives.items()
                }
                total_critical_minimum = 1.0
            # If critical minimums are too high, ensure they don't leave negative weight for others
            elif (
                total_critical_minimum > 0.95
            ):  # Leave at least 5% for non-critical if possible
                logger.warning(
                    f"Critical objectives reserve {total_critical_minimum * 100:.1f}% of weight. Limited room for others."
                )

            # Assign critical minimums first
            final_weights = {}
            final_weights.update(critical_objectives)

            # Remaining weight available for non-critical objectives
            remaining_weight = max(0.0, 1.0 - total_critical_minimum)

            # Get adjusted weights for non-critical objectives
            non_critical_weights = {
                k: v
                for k, v in adjusted_weights.items()
                if k not in critical_objectives
            }

            # Normalize non-critical weights into the remaining space
            total_non_critical_adjusted = sum(non_critical_weights.values())
            if total_non_critical_adjusted > 0 and remaining_weight > 0:
                # Apply remaining weight proportionally based on adjusted weights
                scale_factor = remaining_weight / total_non_critical_adjusted
                normalized_non_critical = {
                    k: max(0.0, v * scale_factor)  # Ensure non-negative
                    for k, v in non_critical_weights.items()
                }
                final_weights.update(normalized_non_critical)
            elif remaining_weight == 0 and non_critical_weights:
                # No room left, set non-critical to zero
                logger.warning(
                    "No remaining weight for non-critical objectives after assigning critical minimums."
                )
                final_weights.update({k: 0.0 for k in non_critical_weights})

            # Final check and normalization to ensure sum is exactly 1.0 due to potential float issues
            total_final_weight = sum(final_weights.values())
            if abs(total_final_weight - 1.0) > 1e-6:
                if total_final_weight > 0:
                    logger.debug(
                        f"Final weights sum {total_final_weight:.6f}, renormalizing."
                    )
                    renorm_factor = 1.0 / total_final_weight
                    final_weights = {
                        k: v * renorm_factor for k, v in final_weights.items()
                    }
                else:  # All weights ended up zero somehow, distribute equally
                    logger.warning(
                        "All final weights are zero. Distributing weight equally."
                    )
                    num_objs = len(final_weights)
                    equal_weight = 1.0 / num_objs if num_objs > 0 else 0
                    final_weights = {k: equal_weight for k in final_weights}

            self.stats["dynamic_weightings"] += 1
            logger.debug(
                f"Dynamic weighting complete. Final weights: {{k: round(v, 4) for k, v in final_weights.items()}}"
            )

            return final_weights

    # --- START FIX: Modified signature and logic ---
    def validate_negotiated_objectives(
        self, negotiated: Dict[str, Any], return_reason: bool = False
    ) -> Union[bool, Tuple[bool, str]]:
        """
        Validate that negotiated objectives (values or weights) respect core constraints.

        Args:
            negotiated: Negotiated objective values OR weights dict.
            return_reason: If True, return (bool, reason_string) tuple.

        Returns:
            True if valid, False otherwise (or tuple if return_reason is True).
        """
        with self.lock:
            validation_reason = ""

            # Check 0: Ensure input is a dictionary
            if not isinstance(negotiated, dict):
                validation_reason = (
                    f"Invalid input type: expected dict, got {type(negotiated)}"
                )
                logger.warning(validation_reason)
                return (False, validation_reason) if return_reason else False

            # --- FIX: Handle empty dict as valid ---
            if not negotiated:
                return (True, "Valid (empty)") if return_reason else True
            # --- END FIX ---

            # Use mock-safe attribute access
            hierarchy_objectives = {}
            if hasattr(self.objective_hierarchy, "objectives") and isinstance(
                self.objective_hierarchy.objectives, dict
            ):
                hierarchy_objectives = self.objective_hierarchy.objectives

            # Check 1: All objectives exist in hierarchy
            for obj_name in negotiated.keys():
                if obj_name not in hierarchy_objectives:
                    validation_reason = (
                        f"Unknown objective in negotiation result: '{obj_name}'"
                    )
                    logger.warning(validation_reason)
                    return (False, validation_reason) if return_reason else False

            # Check 2: Values respect min/max constraints (if 'negotiated' contains values)
            # Heuristic: Assume it's values unless sum is ~1.0 or all values <= 1.0
            all_le_one = all(
                isinstance(v, (int, float)) and v <= 1.0 + 1e-6
                for v in negotiated.values()
            )
            sum_is_one = (
                abs(
                    sum(v for v in negotiated.values() if isinstance(v, (int, float)))
                    - 1.0
                )
                < 1e-6
            )

            # If all values are <= 1 AND sum is ~1.0, it's weights. Otherwise, assume values.
            is_weights = all_le_one and sum_is_one
            is_values = not is_weights

            if is_values:
                logger.debug(
                    "Validating negotiated OBJECTIVE VALUES against constraints."
                )
                for obj_name, value in negotiated.items():
                    obj_data = hierarchy_objectives.get(obj_name)
                    if not obj_data:
                        continue  # Should have been caught by check 1

                    constraints = {}
                    if hasattr(obj_data, "constraints") and isinstance(
                        obj_data.constraints, dict
                    ):
                        constraints = obj_data.constraints
                    elif isinstance(obj_data, dict) and "constraints" in obj_data:
                        constraints = obj_data.get("constraints", {})

                    try:
                        value_num = float(value)
                        # Check min constraint
                        if "min" in constraints:
                            min_limit = float(constraints["min"])
                            if value_num < min_limit - 1e-6:  # Allow tolerance
                                validation_reason = f"Constraint violation: {obj_name} value {value_num:.6f} < min {min_limit:.6f}"
                                logger.warning(validation_reason)
                                return (
                                    (False, validation_reason)
                                    if return_reason
                                    else False
                                )
                        # Check max constraint
                        if "max" in constraints:
                            max_limit = float(constraints["max"])
                            if value_num > max_limit + 1e-6:  # Allow tolerance
                                validation_reason = f"Constraint violation: {obj_name} value {value_num:.6f} > max {max_limit:.6f}"
                                logger.warning(validation_reason)
                                return (
                                    (False, validation_reason)
                                    if return_reason
                                    else False
                                )
                    except (TypeError, ValueError):
                        validation_reason = f"Non-numeric negotiated value '{value}' for objective '{obj_name}' cannot be validated."
                        logger.warning(validation_reason)
                        return (False, validation_reason) if return_reason else False

            # Check 2b (Weights): Now check weight-specific logic if it *is* weights
            if is_weights:
                logger.debug("Validating negotiated OBJECTIVE WEIGHTS.")
                total_weight = 0.0
                for obj_name, weight in negotiated.items():
                    try:
                        w_num = float(weight)
                        if w_num < -1e-6:  # Allow small negative tolerance
                            validation_reason = (
                                f"Negative weight assigned to {obj_name}: {w_num:.6f}"
                            )
                            logger.warning(validation_reason)
                            return (
                                (False, validation_reason) if return_reason else False
                            )
                        total_weight += w_num
                    except (TypeError, ValueError):
                        validation_reason = (
                            f"Non-numeric weight '{weight}' for objective '{obj_name}'."
                        )
                        logger.warning(validation_reason)
                        return (False, validation_reason) if return_reason else False

                if abs(total_weight - 1.0) > 1e-6:  # Allow tolerance for sum
                    validation_reason = (
                        f"Negotiated weights do not sum to 1.0 (sum={total_weight:.6f})"
                    )
                    logger.warning(validation_reason)
                    return (False, validation_reason) if return_reason else False

            # --- FIX: Only check critical objectives *if they are in the negotiated dict* ---
            # Check 3: Critical objectives not severely compromised
            min_critical_weight = (
                0.05  # Minimum acceptable weight for a critical objective if included
            )

            for obj_name, negotiated_value in negotiated.items():
                obj_data = hierarchy_objectives.get(obj_name)
                if not obj_data:
                    continue  # Should be impossible due to Check 1

                # Safely get priority
                priority = 1
                if hasattr(obj_data, "priority"):
                    priority = getattr(obj_data, "priority", 1)
                elif isinstance(obj_data, dict):
                    priority = obj_data.get("priority", 1)

                if (
                    priority == 0
                ):  # This is a critical objective that is *in the negotiated list*
                    if is_weights:
                        # Check if weight is too low
                        try:
                            weight_num = float(negotiated_value)
                            if (
                                weight_num < min_critical_weight - 1e-6
                            ):  # Allow tolerance
                                validation_reason = f"Critical objective {obj_name} severely compromised: weight {weight_num:.6f} < minimum {min_critical_weight:.6f}"
                                logger.warning(validation_reason)
                                return (
                                    (False, validation_reason)
                                    if return_reason
                                    else False
                                )
                        except (TypeError, ValueError):
                            validation_reason = f"Non-numeric weight '{negotiated_value}' for critical objective '{obj_name}'."
                            logger.warning(validation_reason)
                            return (
                                (False, validation_reason) if return_reason else False
                            )
                    elif is_values:
                        # Check if value is way below target
                        target_value = 1.0  # Default
                        if hasattr(obj_data, "target_value"):
                            target_value = getattr(obj_data, "target_value", 1.0)
                        elif isinstance(obj_data, dict):
                            target_value = obj_data.get("target_value", 1.0)

                        try:
                            value_num = float(negotiated_value)
                            # Fail if value is less than 50% of target (arbitrary severe compromise)
                            if target_value > 0 and (value_num / target_value) < 0.5:
                                validation_reason = f"Critical objective {obj_name} severely compromised: value {value_num:.6f} < 50% of target {target_value:.6f}"
                                logger.warning(validation_reason)
                                return (
                                    (False, validation_reason)
                                    if return_reason
                                    else False
                                )
                        except (TypeError, ValueError, AttributeError):
                            validation_reason = f"Non-numeric value '{negotiated_value}' for critical objective '{obj_name}'."
                            logger.warning(validation_reason)
                            return (
                                (False, validation_reason) if return_reason else False
                            )
            # --- END FIX ---

            # Passed all checks
            return (True, "Valid") if return_reason else True

    # --- END FIX ---

    def _parse_proposal(self, proposal: Dict[str, Any]) -> AgentProposal:
        """Parse proposal into structured format"""
        # Add more validation for types
        try:
            target_value = float(proposal.get("target_value", 1.0))
            weight = float(proposal.get("weight", 1.0))
            flexibility = float(proposal.get("flexibility", 0.5))
            constraints = proposal.get("constraints", {})
            if not isinstance(constraints, dict):
                constraints = {}

            return AgentProposal(
                agent_id=str(proposal.get("agent_id", "unknown")),
                objective=str(proposal.get("objective", "unknown")),
                target_value=target_value,
                weight=weight,
                constraints=constraints,
                flexibility=max(0.0, min(1.0, flexibility)),  # Clamp 0-1
                rationale=str(proposal.get("rationale", "")),
                metadata=proposal.get("metadata", {}),
            )
        except (TypeError, ValueError) as e:
            logger.error(
                f"Failed to parse proposal due to invalid value type: {e}. Proposal: {proposal}"
            )
            # Return a default/dummy proposal or raise error?
            # Returning dummy allows processing to continue but might hide issues.
            return AgentProposal(
                agent_id="parse_error", objective="unknown", target_value=0, weight=0
            )

    def _analyze_proposal_space(self, proposals: List[AgentProposal]) -> Dict[str, Any]:
        """Analyze space of proposals"""
        _np = self._np  # Use internal numpy/FakeNumpy alias

        # Collect all proposed objectives
        objectives = sorted(
            list(set(p.objective for p in proposals if p.objective != "unknown"))
        )  # Exclude unknowns

        # Check for conflicts (mock-safe)
        conflicts = []
        if hasattr(self.objective_hierarchy, "find_conflicts") and callable(
            self.objective_hierarchy.find_conflicts
        ):
            for i, obj_a in enumerate(objectives):
                for obj_b in objectives[i + 1 :]:
                    conflict = self.objective_hierarchy.find_conflicts(obj_a, obj_b)
                    if conflict and isinstance(
                        conflict, dict
                    ):  # Check if conflict found and is dict
                        conflicts.append(
                            {
                                "objectives": [obj_a, obj_b],
                                "conflict_type": conflict.get("type", "unknown"),
                                "severity": conflict.get("severity", "unknown"),
                            }
                        )

        # Calculate diversity metrics safely
        weights = [p.weight for p in proposals if isinstance(p.weight, (int, float))]
        flexibility = [
            p.flexibility for p in proposals if isinstance(p.flexibility, (int, float))
        ]

        analysis = {
            "num_proposals": len(proposals),
            "num_unique_objectives": len(objectives),
            "objectives": objectives,
            "conflicts": conflicts,
            "avg_weight": _np.mean(weights) if weights else 0.0,
            "avg_flexibility": _np.mean(flexibility) if flexibility else 0.0,
            "has_conflicts": len(conflicts) > 0,
        }
        # Assess difficulty based on analysis results
        analysis["negotiation_difficulty"] = self._assess_difficulty(
            analysis
        )  # Pass analysis dict

        return analysis

    # Modified to accept analysis dict
    def _assess_difficulty(self, analysis: Dict[str, Any]) -> str:
        """Assess negotiation difficulty based on analysis"""
        num_conflicts = len(analysis["conflicts"])
        avg_flexibility = analysis["avg_flexibility"]

        if not analysis["has_conflicts"]:
            return "easy"
        # More nuanced difficulty assessment
        elif num_conflicts >= 3 or avg_flexibility < 0.2:
            return "hard"
        elif num_conflicts >= 1 or avg_flexibility < 0.4:
            return "medium"
        else:
            return "easy"

    def _select_negotiation_strategy(
        self, proposals: List[AgentProposal], analysis: Dict[str, Any]
    ) -> NegotiationStrategy:
        """Select appropriate negotiation strategy based on analysis"""

        difficulty = analysis["negotiation_difficulty"]
        num_objectives = analysis["num_unique_objectives"]

        # Prioritize Pareto for complex/conflicted cases
        if difficulty == "hard" or num_objectives > 3:
            return NegotiationStrategy.PARETO_OPTIMAL

        # Use Nash for moderate difficulty with few objectives
        if difficulty == "medium" and 2 <= num_objectives <= 3:
            return NegotiationStrategy.NASH_BARGAINING

        # Use Lexicographic if priorities are very clear (check hierarchy?)
        # Simple heuristic: if avg flexibility is low, implies rigid priorities
        if analysis["avg_flexibility"] < 0.3:
            # Check if priority order exists and covers objectives
            priority_order = []
            if hasattr(self.objective_hierarchy, "get_priority_order"):
                priority_order = self.objective_hierarchy.get_priority_order()
            if priority_order and all(
                obj in priority_order for obj in analysis["objectives"]
            ):
                return NegotiationStrategy.LEXICOGRAPHIC

        # Default to Weighted Average for easy cases or as a fallback
        return NegotiationStrategy.WEIGHTED_AVERAGE

    def _negotiate_via_pareto(
        self, proposals: List[AgentProposal], analysis: Dict[str, Any]
    ) -> NegotiationResult:
        """Negotiate by finding Pareto-optimal solution"""
        logger.debug("Negotiating via Pareto Optimal strategy...")
        objectives = analysis["objectives"]

        # Find Pareto frontier
        pareto_points = self.find_pareto_frontier({"objectives": objectives})

        if not pareto_points:
            logger.warning("No Pareto frontier found.")
            return self._create_deadlock_result(proposals, "No Pareto frontier found")

        # Select point that best satisfies all agents (fairness)
        best_point = self._select_fairest_pareto_point(pareto_points, proposals)

        # Extract weights and outcomes
        agreed_weights = best_point.get("weights", {})
        agreed_objectives = best_point.get("objectives", {})  # These are outcome values

        return NegotiationResult(
            outcome=NegotiationOutcome.CONSENSUS,  # Pareto implies a form of consensus/optimality
            agreed_objectives=agreed_objectives,
            objective_weights=agreed_weights,
            participating_agents=[p.agent_id for p in proposals],
            strategy_used=NegotiationStrategy.PARETO_OPTIMAL,
            iterations=1,  # Finding frontier is one conceptual step here
            convergence_time_ms=0,  # Placeholder, calculation time captured elsewhere
            compromises_made=[],  # Pareto point is optimal, not necessarily a simple compromise
            pareto_optimal=True,
            confidence=0.9,
            reasoning="Selected fairest point from the Pareto frontier.",
        )

    def _negotiate_via_weighted_average(
        self, proposals: List[AgentProposal], analysis: Dict[str, Any]
    ) -> NegotiationResult:
        """Negotiate by weighted averaging of proposals"""
        logger.debug("Negotiating via Weighted Average strategy...")
        _np = self._np  # Use internal alias

        # Aggregate weights for each objective based on proposing agents' weights
        objective_weights_sum = defaultdict(float)
        objective_target_weighted_sum = defaultdict(float)
        objective_proposer_weight_sum = defaultdict(
            float
        )  # Sum of weights of agents proposing each obj

        total_agent_weight = sum(p.weight for p in proposals)

        # Guard against division by zero
        if total_agent_weight <= 0:
            logger.warning(
                "All agent proposals have zero or negative weight in weighted average."
            )
            # Default to equal weights if possible
            num_objs = analysis["num_unique_objectives"]
            if num_objs > 0:
                equal_weight = 1.0 / num_objs
                final_weights = {obj: equal_weight for obj in analysis["objectives"]}
                # Estimate outcomes based on equal weights (simple avg of targets?)
                agreed_outcomes = {}
                for obj in analysis["objectives"]:
                    targets = [p.target_value for p in proposals if p.objective == obj]
                    agreed_outcomes[obj] = (
                        _np.mean(targets) if targets else 0.5
                    )  # Default outcome

                return NegotiationResult(
                    outcome=NegotiationOutcome.COMPROMISE,
                    agreed_objectives=agreed_outcomes,
                    objective_weights=final_weights,
                    participating_agents=[p.agent_id for p in proposals],
                    strategy_used=NegotiationStrategy.WEIGHTED_AVERAGE,
                    iterations=1,
                    convergence_time_ms=0,
                    compromises_made=[{"type": "equal_weight_fallback"}],
                    pareto_optimal=False,
                    confidence=0.5,
                    reasoning="Fell back to equal weights due to zero agent weights.",
                )
            else:
                return self._create_deadlock_result(
                    proposals,
                    "All agent proposals have zero weight and no objectives identified.",
                )

        for proposal in proposals:
            normalized_agent_weight = proposal.weight / total_agent_weight
            objective_weights_sum[proposal.objective] += normalized_agent_weight
            # For outcome estimation, average target values weighted by agent weight
            objective_target_weighted_sum[proposal.objective] += (
                proposal.target_value * proposal.weight
            )
            objective_proposer_weight_sum[proposal.objective] += proposal.weight

        # Final normalized objective weights
        final_objective_weights = dict(objective_weights_sum)

        # Estimate agreed outcomes (weighted average of targets for each objective)
        agreed_objectives_outcomes = {}
        for obj, total_prop_weight in objective_proposer_weight_sum.items():
            if total_prop_weight > 0:
                agreed_objectives_outcomes[obj] = (
                    objective_target_weighted_sum[obj] / total_prop_weight
                )
            else:
                # If no one proposed this obj with weight > 0, estimate differently?
                # Maybe use average of all targets? Or just use the calculated weight?
                targets = [p.target_value for p in proposals if p.objective == obj]
                agreed_objectives_outcomes[obj] = (
                    _np.mean(targets)
                    if targets
                    else final_objective_weights.get(obj, 0.5)
                )  # Fallback

        return NegotiationResult(
            outcome=NegotiationOutcome.COMPROMISE,
            agreed_objectives=agreed_objectives_outcomes,  # Estimated outcomes
            objective_weights=final_objective_weights,  # Final weights
            participating_agents=[p.agent_id for p in proposals],
            strategy_used=NegotiationStrategy.WEIGHTED_AVERAGE,
            iterations=1,
            convergence_time_ms=0,  # Placeholder
            compromises_made=[{"type": "weighted_average"}],
            pareto_optimal=False,  # Weighted average is generally not Pareto optimal
            confidence=0.7,
            reasoning="Compromise reached via weighted average of agent proposals.",
        )

    # --- Placeholder Implementations for Nash, Lexicographic, Minimax ---
    # These require more complex optimization or simulation steps.

    def _negotiate_via_nash_bargaining(
        self, proposals: List[AgentProposal], analysis: Dict[str, Any]
    ) -> NegotiationResult:
        """Negotiate using Nash bargaining solution (Placeholder)"""
        logger.warning(
            "Nash bargaining strategy is not fully implemented - using fallback."
        )
        # Fallback to weighted average for now
        return self._negotiate_via_weighted_average(proposals, analysis)

    def _negotiate_via_lexicographic(
        self, proposals: List[AgentProposal], analysis: Dict[str, Any]
    ) -> NegotiationResult:
        """Negotiate using lexicographic ordering (priority-based)"""
        logger.debug("Negotiating via Lexicographic strategy...")

        # Get priority order from objective hierarchy (mock-safe)
        priority_order = []
        if hasattr(self.objective_hierarchy, "get_priority_order") and callable(
            self.objective_hierarchy.get_priority_order
        ):
            priority_order = self.objective_hierarchy.get_priority_order()

        # Order objectives present in proposals based on the hierarchy's priority order
        objectives_in_proposals = analysis["objectives"]
        ordered_objectives = [
            obj for obj in priority_order if obj in objectives_in_proposals
        ]
        # Add any proposal objectives not in the hierarchy order at the end (lower priority)
        ordered_objectives.extend(
            [item for item in objectives_in_proposals if item not in priority_order]
        )

        if not ordered_objectives:
            return self._create_deadlock_result(
                proposals, "No valid objectives for lexicographic ordering."
            )

        # Assign weights based on rank (e.g., higher priority gets significantly more weight)
        # Simple exponential decay might work: w_i = base^(i)
        base = 0.5  # Example decay factor
        raw_weights = {obj: base**i for i, obj in enumerate(ordered_objectives)}

        # Normalize weights
        total_raw_weight = sum(raw_weights.values())
        agreed_weights = (
            {obj: w / total_raw_weight for obj, w in raw_weights.items()}
            if total_raw_weight > 0
            else {}
        )

        # Estimate outcomes (could be complex: optimize highest priority fully, then next with remaining flexibility)
        # Simple estimation: use the target value proposed for each objective
        agreed_objectives = {}
        for obj in ordered_objectives:
            targets = [p.target_value for p in proposals if p.objective == obj]
            # Use max target proposed for this objective as the goal
            agreed_objectives[obj] = max(targets) if targets else 0.0

        return NegotiationResult(
            outcome=NegotiationOutcome.CONSENSUS,  # Lexicographic implies consensus based on priority
            agreed_objectives=agreed_objectives,  # Estimated target outcomes
            objective_weights=agreed_weights,  # Weights reflect priority order
            participating_agents=[p.agent_id for p in proposals],
            strategy_used=NegotiationStrategy.LEXICOGRAPHIC,
            iterations=1,
            convergence_time_ms=0,  # Placeholder
            compromises_made=[],  # No compromise in lexicographic, just priority
            pareto_optimal=False,  # Lexicographic is often not Pareto optimal
            confidence=0.85,
            reasoning=f"Objectives ordered by priority: {ordered_objectives}",
        )

    def _negotiate_via_minimax(
        self, proposals: List[AgentProposal], analysis: Dict[str, Any]
    ) -> NegotiationResult:
        """Negotiate using minimax (minimize maximum regret) (Placeholder)"""
        logger.warning("Minimax strategy is not fully implemented - using fallback.")
        # Fallback to weighted average for now
        return self._negotiate_via_weighted_average(proposals, analysis)

    # --- End Placeholder Implementations ---

    def _generate_candidate_solutions(
        self, objectives: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate candidate solutions (weight combinations and estimated outcomes)"""

        candidates = []
        n_objectives = len(objectives)
        if n_objectives == 0:
            return candidates

        # Use counterfactual reasoner if available for better estimation
        use_cf_reasoner = False
        reasoner = None
        if hasattr(self.world_model, "motivational_introspection"):
            mi = self.world_model.motivational_introspection
            if not isinstance(mi, MagicMock) and hasattr(mi, "counterfactual_reasoner"):
                potential_reasoner = mi.counterfactual_reasoner
                if not isinstance(potential_reasoner, MagicMock) and hasattr(
                    potential_reasoner, "predict_under_objective"
                ):
                    use_cf_reasoner = True
                    reasoner = potential_reasoner

        logger.debug(
            f"Generating candidate solutions for {n_objectives} objectives. Using reasoner: {use_cf_reasoner}"
        )

        # Generate weight combinations (more steps for fewer objectives)
        num_steps = max(3, 12 - n_objectives)  # e.g., 10 for 2, 9 for 3, ... 3 for >=9
        generated_weights_tuples = self._generate_weights(n_objectives, num_steps)
        logger.debug(f"Generated {len(generated_weights_tuples)} weight combinations.")

        for weights_tuple in generated_weights_tuples:
            weights = dict(zip(objectives, weights_tuple))

            # Estimate objective values at this weight combination
            obj_values = {}
            confidence_sum = 0.0
            num_predicted = 0

            if use_cf_reasoner and reasoner:
                for obj in objectives:
                    try:
                        # Predict outcome with these weights in context
                        context = {"objective_weights": weights}
                        outcome = reasoner.predict_under_objective(obj, context)
                        obj_values[obj] = outcome.predicted_value
                        confidence_sum += outcome.confidence
                        num_predicted += 1
                    except Exception as e:
                        logger.warning(
                            f"Prediction failed for objective {obj} with weights {weights}: {e}"
                        )
                        obj_values[obj] = weights.get(obj, 0.0)  # Fallback to weight
                        confidence_sum += 0.3  # Low confidence on fallback
                        num_predicted += 1

            else:
                # Fallback: simple linear combination (value = weight)
                obj_values = {obj: w for obj, w in weights.items()}
                confidence_sum = 0.5 * n_objectives  # Default confidence
                num_predicted = n_objectives

            avg_confidence = (
                (confidence_sum / num_predicted) if num_predicted > 0 else 0.5
            )

            candidates.append(
                {
                    "weights": weights,
                    "objectives": obj_values,  # Estimated outcome values
                    "confidence": avg_confidence,  # Store avg prediction confidence
                }
            )

        logger.debug(
            f"Generated {len(candidates)} candidate solutions with estimated outcomes."
        )
        return candidates

    # --- _generate_weights and _generate_weights_recursive remain largely the same ---
    # (Copied from previous correct version for completeness)
    def _generate_weights(self, n: int, steps: int) -> List[Tuple[float, ...]]:
        """Generate weight combinations summing to 1 using recursion"""
        results = []
        if n <= 0:
            return results
        if n == 1:
            results.append((1.0,))
            return results

        # Use the recursive helper
        results = self._generate_weights_recursive(n, steps, 1.0)

        # Deduplicate and normalize just in case of floating point issues
        unique_results = []
        seen = set()
        tolerance = 1e-6
        for res_tuple in results:
            total = sum(res_tuple)
            if abs(total - 1.0) < tolerance:  # Check if sum is close to 1
                # Round to reduce floating point noise before adding to set
                rounded_tuple = tuple(round(x, 8) for x in res_tuple)
                if rounded_tuple not in seen:
                    # Add the original (non-rounded) tuple if it sums correctly
                    unique_results.append(res_tuple)
                    seen.add(rounded_tuple)
            elif (
                total > tolerance
            ):  # Normalize if sum is significantly off but positive
                normalized_tuple = tuple(x / total for x in res_tuple)
                rounded_norm_tuple = tuple(round(x, 8) for x in normalized_tuple)
                if rounded_norm_tuple not in seen:
                    unique_results.append(normalized_tuple)
                    seen.add(rounded_norm_tuple)

        # Ensure corner cases and equal weight are included if missed
        if len(unique_results) < n + 1:
            corners = [
                tuple(1.0 if j == i else 0.0 for j in range(n)) for i in range(n)
            ]
            equal = tuple(1.0 / n for _ in range(n))

            for case in corners + [equal]:
                rounded_case = tuple(round(x, 8) for x in case)
                if rounded_case not in seen:
                    unique_results.append(case)
                    seen.add(rounded_case)

        return unique_results

    def _generate_weights_recursive(
        self, n: int, steps: int, target_sum: float
    ) -> List[Tuple[float, ...]]:
        """Helper for recursive weight generation with a target sum"""
        results = []
        tolerance = 1e-9

        if n == 1:
            # Check if target_sum is within valid range [0, 1] essentially
            if -tolerance < target_sum < 1.0 + tolerance:
                # Clamp to [0, target_sum] and add tuple
                results.append(
                    (max(0.0, min(target_sum, 1.0)),)
                )  # Clamp result just in case
            return results

        # Iterate through possible values for the current weight w_current
        # It can range from 0 up to the target_sum, divided into 'steps'
        for i in range(steps + 1):
            # Calculate potential value for the current dimension
            # Ensure w_current does not exceed target_sum or 1.0
            w_current = min(target_sum, (i / steps))

            # Ensure w_current is non-negative due to float issues
            w_current = max(0.0, w_current)

            remaining_target = target_sum - w_current

            # If remaining target is feasible (non-negative within tolerance)
            if remaining_target > -tolerance:
                # Recursively generate weights for the rest
                for rest_tuple in self._generate_weights_recursive(
                    n - 1, steps, remaining_target
                ):
                    # Combine and add if the sum is close enough to the original target_sum
                    current_combination = (w_current,) + rest_tuple
                    if abs(sum(current_combination) - target_sum) < tolerance:
                        results.append(current_combination)

        return results

    def _calculate_nash_product(
        self, candidate: Dict[str, Any], proposals: List[AgentProposal]
    ) -> float:
        """Calculate Nash product for candidate solution"""

        product = 1.0
        candidate_outcomes = candidate.get("objectives", {})

        for proposal in proposals:
            # Utility for this agent (how well candidate meets agent's goal)
            achieved = candidate_outcomes.get(proposal.objective, 0.0)  # Outcome value

            # Normalize utility: 0 if outcome <= 0, 1 if outcome >= target, linear in between
            target = proposal.target_value
            utility = 0.0
            if target > 0:  # Avoid division by zero, assume target > 0 for utility calc
                utility = max(0.0, min(1.0, achieved / target))
            elif (
                achieved > 0
            ):  # If target is 0 or less, any positive achievement is max utility? Or interpret differently? Assume 1.0 here.
                utility = 1.0

            # Disagreement point is 0 utility. Add epsilon for stability.
            product *= utility + 1e-9

        # Return geometric mean instead of raw product for better numerical stability?
        # Or just the product as per definition. Stick to product for now.
        return product

    def _calculate_max_regret(
        self, candidate: Dict[str, Any], proposals: List[AgentProposal]
    ) -> float:
        """Calculate maximum regret for candidate"""

        max_regret = 0.0
        candidate_outcomes = candidate.get("objectives", {})

        for proposal in proposals:
            # Agent's ideal outcome (target value)
            ideal = proposal.target_value

            # Actual outcome achieved in the candidate solution for the agent's preferred objective
            actual = candidate_outcomes.get(proposal.objective, 0.0)

            # Regret = how much agent *lost* compared to their ideal, weighted by importance
            # Ensure ideal is treated as upper bound if maximize=True (default)
            # Need objective direction from hierarchy if available
            maximize = True  # Assume maximize
            obj_data = None
            if hasattr(self.objective_hierarchy, "objectives"):
                obj_data = self.objective_hierarchy.objectives.get(proposal.objective)
            if obj_data:
                maximize = getattr(obj_data, "maximize", True)

            if maximize:
                # Regret is difference from ideal, or 0 if actual >= ideal
                regret_amount = max(0.0, ideal - actual)
            else:  # Minimize objective
                # Regret is difference from ideal (lower is better), or 0 if actual <= ideal
                regret_amount = max(0.0, actual - ideal)

            # Weighted regret
            regret = regret_amount * proposal.weight
            max_regret = max(max_regret, regret)

        return max_regret

    def _select_fairest_pareto_point(
        self,
        pareto_points: List[Dict[str, Any]],  # List of dicts now
        proposals: List[AgentProposal],
    ) -> Dict[str, Any]:
        """Select Pareto point that is fairest to all agents"""

        if not pareto_points:
            logger.warning("Cannot select fairest point, no Pareto points provided.")
            # Return an empty structure matching the expected output format
            return {"weights": {}, "objectives": {}, "confidence": 0.0}

        # Calculate fairness score for each point
        best_point = pareto_points[0]  # Default to first point
        best_fairness = -float("inf")  # Initialize to negative infinity

        for point in pareto_points:
            fairness = self._calculate_fairness(point, proposals)
            if fairness > best_fairness:
                best_fairness = fairness
                best_point = point

        logger.debug(
            f"Selected fairest Pareto point with fairness score: {best_fairness:.4f}"
        )
        return best_point

    def _calculate_fairness(
        self,
        point: Dict[str, Any],  # Point is a dict
        proposals: List[AgentProposal],
    ) -> float:
        """Calculate fairness score (e.g., negative variance of normalized weighted utilities)"""
        _np = self._np  # Use internal alias

        utilities = []
        point_outcomes = point.get("objectives", {})

        for proposal in proposals:
            achieved = point_outcomes.get(proposal.objective, 0.0)

            # Normalize utility: 0 if outcome <= 0, 1 if outcome >= target, linear in between
            target = proposal.target_value
            utility = 0.0
            if target > 0:
                utility = max(0.0, min(1.0, achieved / target))
            elif achieved > 0:
                utility = 1.0

            # Weight utility by agent's proposal weight (importance)
            weighted_utility = utility * proposal.weight
            utilities.append(weighted_utility)

        if not utilities:
            return 0.0  # No proposals, perfectly fair? Or undefined? Return 0.

        # Fairness = negative variance (lower variance = more fair)
        # Use np.var (or fake version) for variance calculation
        variance = _np.var(utilities) if len(utilities) > 1 else 0.0
        # Return negative variance, higher is better (closer to zero variance)
        return -variance

    def _analyze_binary_tradeoff(
        self, obj_a: str, obj_b: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze trade-off between two objectives using counterfactual reasoner or fallback"""
        _np = self._np  # Use internal alias
        tradeoffs = []

        # Get counterfactual reasoner if available and functional
        reasoner = None
        if hasattr(self.world_model, "motivational_introspection"):
            mi = self.world_model.motivational_introspection
            if not isinstance(mi, MagicMock) and hasattr(mi, "counterfactual_reasoner"):
                potential_reasoner = mi.counterfactual_reasoner
                if not isinstance(potential_reasoner, MagicMock) and hasattr(
                    potential_reasoner, "predict_under_objective"
                ):
                    reasoner = potential_reasoner

        num_steps = 11  # Number of points to sample (e.g., 0%, 10%, ..., 100%)
        logger.debug(
            f"Analyzing tradeoff between {obj_a} and {obj_b}. Using reasoner: {reasoner is not None}"
        )

        for weight_a in _np.linspace(0.0, 1.0, num_steps):
            weight_b = 1.0 - weight_a

            if reasoner:
                try:
                    # Create scenario with these weights
                    scenario = {
                        **context,
                        "objective_weights": {obj_a: weight_a, obj_b: weight_b},
                    }
                    # Predict outcomes
                    outcome_a = reasoner.predict_under_objective(obj_a, scenario)
                    outcome_b = reasoner.predict_under_objective(obj_b, scenario)

                    tradeoffs.append(
                        {
                            "weight_a": weight_a,
                            "weight_b": weight_b,
                            "outcome_a": outcome_a.predicted_value,
                            "outcome_b": outcome_b.predicted_value,
                            "confidence": min(
                                outcome_a.confidence, outcome_b.confidence
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Prediction failed during tradeoff analysis for weights ({weight_a:.2f}, {weight_b:.2f}): {e}"
                    )
                    # Append fallback point?
                    tradeoffs.append(
                        {
                            "weight_a": weight_a,
                            "weight_b": weight_b,
                            "outcome_a": weight_a,
                            "outcome_b": weight_b,  # Fallback value = weight
                            "confidence": 0.3,
                        }
                    )

            else:
                # Fallback: simple linear assumption (value = weight)
                tradeoffs.append(
                    {
                        "weight_a": weight_a,
                        "weight_b": weight_b,
                        "outcome_a": weight_a,
                        "outcome_b": weight_b,
                        "confidence": 0.5,  # Low confidence for simplified version
                    }
                )

        return tradeoffs

    def _find_optimal_compromise(
        self,
        obj_a: str,
        obj_b: str,
        tradeoffs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Find optimal compromise point based on maximizing product of outcomes (Nash-like)"""

        best_point_weights = {obj_a: 0.5, obj_b: 0.5}  # Default to equal weights
        best_utility_product = -1.0

        if not tradeoffs:
            logger.warning(
                "No tradeoff points provided to find optimal compromise. Using default equal weights."
            )
            return best_point_weights

        for point in tradeoffs:
            outcome_a = point.get("outcome_a", 0.0)
            outcome_b = point.get("outcome_b", 0.0)

            # Combined utility (product, add epsilon to avoid zero if outcomes can be zero)
            # Use max(0, outcome) to handle potential negative predictions? Assume outcomes >= 0 here.
            utility_product = (outcome_a + 1e-9) * (outcome_b + 1e-9)

            if utility_product > best_utility_product:
                best_utility_product = utility_product
                best_point_weights = {
                    obj_a: point["weight_a"],
                    obj_b: point["weight_b"],
                }

        logger.debug(
            f"Optimal compromise weights for {obj_a}/{obj_b}: {best_point_weights} (product={best_utility_product:.4f})"
        )
        return best_point_weights

    def _estimate_outcomes_at_point(
        self, obj_a: str, obj_b: str, weights: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate expected outcomes at a specific weight point using reasoner or fallback"""

        # Get counterfactual reasoner if available and functional
        reasoner = None
        if hasattr(self.world_model, "motivational_introspection"):
            mi = self.world_model.motivational_introspection
            if not isinstance(mi, MagicMock) and hasattr(mi, "counterfactual_reasoner"):
                potential_reasoner = mi.counterfactual_reasoner
                if not isinstance(potential_reasoner, MagicMock) and hasattr(
                    potential_reasoner, "predict_under_objective"
                ):
                    reasoner = potential_reasoner

        estimated_outcomes = {}
        if reasoner:
            try:
                scenario = {**context, "objective_weights": weights}
                outcome_a = reasoner.predict_under_objective(obj_a, scenario)
                outcome_b = reasoner.predict_under_objective(obj_b, scenario)
                estimated_outcomes = {
                    obj_a: outcome_a.predicted_value,
                    obj_b: outcome_b.predicted_value,
                }
                logger.debug(
                    f"Estimated outcomes at weights {weights} using reasoner: {estimated_outcomes}"
                )
            except Exception as e:
                logger.warning(
                    f"Prediction failed for outcome estimation at weights {weights}: {e}. Using fallback."
                )
                # Fall through to fallback

        # Fallback if reasoner failed or not available
        if not estimated_outcomes:
            # Simple linear interpolation: outcome = weight (very rough)
            estimated_outcomes = {
                obj_a: weights.get(obj_a, 0.5),
                obj_b: weights.get(obj_b, 0.5),
            }
            logger.debug(
                f"Estimated outcomes at weights {weights} using fallback: {estimated_outcomes}"
            )

        return estimated_outcomes

    def _calculate_resolution_confidence(
        self,
        tradeoffs: List[Dict[str, Any]],
        optimal_point_weights: Dict[str, float],
        estimated_outcomes: Dict[str, float],
    ) -> float:
        """Calculate confidence in the resolution based on tradeoff data quality"""
        _np = self._np  # Use internal alias

        if not tradeoffs:
            return 0.3  # Low confidence if no tradeoff data

        # Average confidence from the tradeoff analysis points
        tradeoff_confidences = [t.get("confidence", 0.5) for t in tradeoffs]
        avg_tradeoff_confidence = (
            _np.mean(tradeoff_confidences) if tradeoff_confidences else 0.5
        )

        # Consider stability: how much do outcomes change near the optimal point?
        # Find the point in tradeoffs closest to optimal_weights
        closest_point = min(
            tradeoffs,
            key=lambda p: abs(
                p["weight_a"]
                - optimal_point_weights.get(list(optimal_point_weights.keys())[0], 0.5)
            ),
        )

        # Use the confidence of the closest point as a factor
        stability_confidence = closest_point.get("confidence", avg_tradeoff_confidence)

        # Final confidence is average of overall tradeoff confidence and stability point confidence
        confidence = (avg_tradeoff_confidence + stability_confidence) / 2.0

        # Adjust based on how extreme the optimal weights are (more extreme might be less certain?)
        # extremity = max(abs(w - 0.5) for w in optimal_point_weights.values()) * 2
        # confidence *= (1.0 - 0.1 * extremity) # Small penalty for extreme weights

        return float(max(0.1, min(0.95, confidence)))  # Clamp between 0.1 and 0.95

    def _generate_resolution_reasoning(
        self,
        obj_a: str,
        obj_b: str,
        weights: Dict[str, float],
        outcomes: Dict[str, float],
        tradeoffs: List[Dict[str, Any]],
    ) -> str:
        """Generate reasoning for the conflict resolution"""

        weight_a = weights.get(obj_a, 0.0)  # Default 0 if missing
        weight_b = weights.get(obj_b, 0.0)
        outcome_a = outcomes.get(obj_a, 0.0)
        outcome_b = outcomes.get(obj_b, 0.0)

        reason = (
            f"Resolved conflict between '{obj_a}' and '{obj_b}'. "
            f"Optimal compromise weights found: {obj_a}={weight_a:.2f}, {obj_b}={weight_b:.2f}. "
            f"Estimated outcomes at this point: {obj_a}={outcome_a:.3f}, {obj_b}={outcome_b:.3f}."
        )

        # Add info about tradeoff shape if available
        # (Could calculate second derivative from tradeoffs list for convexity/concavity)

        return reason

    def _compute_pareto_frontier_simple(
        self, objectives: List[str], objective_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simple Pareto frontier computation (fallback)."""
        logger.debug("Executing simple Pareto frontier computation (fallback).")
        # Generate candidate points (weights and estimated outcomes)
        # Pass only objectives list to generate_candidate_solutions
        candidates = self._generate_candidate_solutions(objectives)

        if not candidates:
            logger.warning("No candidate solutions generated for simple Pareto.")
            return []

        # Check dominance among candidates
        pareto_points = []
        num_candidates = len(candidates)

        for i, candidate in enumerate(candidates):
            is_dominated = False
            candidate_outcomes = candidate.get("objectives", {})

            for j, other in enumerate(candidates):
                if i == j:
                    continue
                other_outcomes = other.get("objectives", {})

                # Check if 'other' dominates 'candidate'
                if self._dominates(other_outcomes, candidate_outcomes, objectives):
                    is_dominated = True
                    break  # No need to check further if dominated

            if not is_dominated:
                # Add weights and outcomes to the result for consistency
                pareto_points.append(
                    {
                        "objectives": candidate_outcomes,
                        "weights": candidate.get("weights", {}),
                        "is_pareto_optimal": True,  # By definition of this loop
                        "confidence": candidate.get(
                            "confidence", 0.5
                        ),  # Include confidence
                    }
                )

        logger.debug(f"Simple Pareto computation found {len(pareto_points)} points.")
        return pareto_points

    def _dominates(
        self,
        point_a_outcomes: Dict[str, float],
        point_b_outcomes: Dict[str, float],
        objectives: List[str],
    ) -> bool:
        """Check if point_a Pareto-dominates point_b based on outcomes"""

        better_in_at_least_one = False
        worse_in_any = False

        # Use mock-safe attribute access
        hierarchy_objectives = {}
        if hasattr(self.objective_hierarchy, "objectives") and isinstance(
            self.objective_hierarchy.objectives, dict
        ):
            hierarchy_objectives = self.objective_hierarchy.objectives

        for obj in objectives:
            val_a = point_a_outcomes.get(obj, 0.0)  # Default to 0 if missing
            val_b = point_b_outcomes.get(obj, 0.0)

            # Get objective direction (maximize by default)
            maximize = True  # Assume maximize if hierarchy/objective not available
            obj_data = hierarchy_objectives.get(obj)
            if obj_data:
                if hasattr(obj_data, "maximize"):
                    maximize = getattr(obj_data, "maximize", True)
                elif isinstance(obj_data, dict):
                    maximize = obj_data.get("maximize", True)

            # Compare based on direction, allowing for small tolerance
            tolerance = 1e-9
            if maximize:
                if val_a > val_b + tolerance:
                    better_in_at_least_one = True
                elif val_a < val_b - tolerance:
                    worse_in_any = True
                    break  # Cannot dominate if worse in any
            else:  # Minimize
                if val_a < val_b - tolerance:
                    better_in_at_least_one = True
                elif val_a > val_b + tolerance:
                    worse_in_any = True
                    break

        return better_in_at_least_one and not worse_in_any

    def _create_deadlock_result(
        self,
        proposals: List[Union[AgentProposal, Dict]],  # Allow dicts too
        reason: str,
    ) -> NegotiationResult:
        """Create deadlock result, safely extracting agent IDs"""
        agent_ids = []
        for p in proposals:
            if isinstance(p, AgentProposal):
                agent_ids.append(p.agent_id)
            elif isinstance(p, dict):
                agent_ids.append(p.get("agent_id", "unknown"))

        logger.warning(f"Negotiation deadlock: {reason}")
        return NegotiationResult(
            outcome=NegotiationOutcome.DEADLOCK,
            agreed_objectives={},
            objective_weights={},
            participating_agents=agent_ids,
            strategy_used=NegotiationStrategy.PARETO_OPTIMAL,  # Default strategy reporting
            iterations=0,
            convergence_time_ms=0,
            compromises_made=[],
            pareto_optimal=False,
            confidence=0.0,
            reasoning=f"Negotiation deadlock: {reason}",
        )

    def _empty_negotiation_result(self) -> NegotiationResult:
        """Create empty negotiation result when no proposals are provided"""
        logger.warning("Attempted negotiation with zero proposals.")
        return NegotiationResult(
            outcome=NegotiationOutcome.DEADLOCK,
            agreed_objectives={},
            objective_weights={},
            participating_agents=[],
            strategy_used=NegotiationStrategy.PARETO_OPTIMAL,  # Default
            iterations=0,
            convergence_time_ms=0,
            compromises_made=[],
            pareto_optimal=False,  # Cannot be optimal with no objectives
            confidence=0.0,
            reasoning="No proposals were provided for negotiation.",
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get negotiation statistics"""

        with self.lock:
            strategy_stats = {}
            for strategy_enum, perf in self.strategy_performance.items():
                # Ensure key is the string value of the enum
                strategy_key = (
                    strategy_enum.value
                    if hasattr(strategy_enum, "value")
                    else str(strategy_enum)
                )
                total = perf["successes"] + perf["failures"]
                strategy_stats[strategy_key] = {
                    "successes": perf["successes"],
                    "failures": perf["failures"],
                    "success_rate": (perf["successes"] / total) if total > 0 else 0.0,
                    "total_runs": total,
                }

            # Ensure all strategies are represented, even if not run
            for strategy_enum in NegotiationStrategy:
                if strategy_enum.value not in strategy_stats:
                    strategy_stats[strategy_enum.value] = {
                        "successes": 0,
                        "failures": 0,
                        "success_rate": 0.0,
                        "total_runs": 0,
                    }

            return {
                "total_negotiations": self.stats["negotiations_performed"],
                "outcomes": {
                    k.replace("outcome_", ""): v
                    for k, v in self.stats.items()
                    if k.startswith("outcome_")
                },
                "conflicts_resolved": self.stats["conflicts_resolved"],
                "dynamic_weightings": self.stats["dynamic_weightings"],
                "negotiation_history_size": len(self.negotiation_history),
                "strategy_performance": strategy_stats,
            }
