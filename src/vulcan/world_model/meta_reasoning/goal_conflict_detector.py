# src/vulcan/world_model/meta_reasoning/goal_conflict_detector.py
"""
goal_conflict_detector.py - Goal conflict detection and analysis
Part of the meta_reasoning subsystem for VULCAN-AMI

Detects and analyzes conflicts between objectives:
- Direct conflicts (mutually exclusive)
- Indirect conflicts (resource contention)
- Constraint-based conflicts
- Multi-objective tension analysis

Provides conflict resolution suggestions.

FIXED: Constraint violations detected for single objectives
FIXED: Statistics updated for all proposals
FIXED: Handle n=1 case in tension analysis (no NaN)
FIXED: Defensive check for malformed objective_weights
FIXED: Added accuracy_safety rule pattern
FIXED: Enhanced logging and conflict aggregation
FIXED(test_world_model_meta_reasoning_integration): Corrected call signature for ObjectiveHierarchy.get_priority_order()
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# import numpy as np # Original import
from typing import Any, Dict, List, Optional

# FIXED: Import Mock for type checking in __init__
from unittest.mock import MagicMock, Mock

# --- START FIX: Add numpy fallback ---
logger = logging.getLogger(__name__)  # Moved logger init up
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using list-based math")

    class FakeNumpy:
        # Define necessary numpy functions used in this file
        @staticmethod
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            if len(shape) == 1:
                return [0.0] * shape[0]
            if len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
            raise NotImplementedError("FakeNumpy only supports 1D/2D zeros")

        @staticmethod
        def array(data):
            return list(data)  # Just return list

        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        @staticmethod
        def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
            if not a or (
                isinstance(a, list)
                and not any(isinstance(row, list) for row in a)
                and not a
            ):  # Empty list or list of empty lists
                return 0.0  # Or raise error
            if isinstance(a[0], list):  # 2D list
                return max(max(row) for row in a if row)
            else:  # 1D list
                return max(a)

        @staticmethod
        def triu_indices(n, k=0, m=None):
            if m is None:
                m = n
            rows, cols = [], []
            for i in range(n):
                for j in range(m):
                    if i <= j - k:
                        rows.append(i)
                        cols.append(j)
            return (rows, cols)

    np = FakeNumpy()
# --- END FIX ---


# --- START FIX: Strengthen ObjectiveHierarchy fallback ---
# Original import block:
# try:
#     from .objective_hierarchy import ObjectiveHierarchy, Objective, ObjectiveType, ConflictType as HierarchyConflictType
# except ImportError:
#     # Fallback for standalone testing or if hierarchy is structured differently
#     class ObjectiveHierarchy:
#         def __init__(self): self.objectives = {}; self.conflict_graph = defaultdict(set)
#         def find_conflicts(self, a, b): return None
#         def get_priority_order(self): return []
#     class Objective: pass
#     class ObjectiveType(Enum): PRIMARY="primary"; SECONDARY="secondary"; DERIVED="derived"
#     class HierarchyConflictType(Enum): pass

# Replacement block:
try:
    # Rename imported ConflictType to avoid clash with local definition
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
except ImportError as e:
    logger.error(f"Failed to import objective_hierarchy: {e}. Using fallback mock.")
    OBJECTIVE_HIERARCHY_AVAILABLE = False
    # Use MagicMock for a more flexible fallback
    # FIXED: Use MagicMock class, not an instance
    ObjectiveHierarchy = MagicMock
    RealObjectiveHierarchy = None  # Set Real reference to None

    # Define fallback Enums
    class Objective:
        pass  # Simple placeholder class

    class ObjectiveType(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        DERIVED = "derived"

    class HierarchyConflictType(Enum):
        DIRECT = "direct"
        INDIRECT = "indirect"
        CONSTRAINT = "constraint"
        TRADEOFF = "tradeoff"  # Added TRADEOFF


# --- END FIX ---


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ConflictType(Enum):
    """Types of objective conflicts"""

    DIRECT = "direct"  # Cannot both be satisfied
    INDIRECT = "indirect"  # Resource contention
    CONSTRAINT = "constraint"  # Constraint incompatibility
    PRIORITY = "priority"  # Priority ordering violation
    TRADEOFF = "tradeoff"  # Acceptable trade-off tension


@dataclass
class Conflict:
    """Represents a conflict between objectives"""

    objectives: List[str]
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    quantitative_measure: Optional[float] = None
    resolution_options: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "objectives": self.objectives,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "quantitative_measure": self.quantitative_measure,
            "resolution_options": self.resolution_options,
            "metadata": self.metadata,
            "detected_at": self.detected_at,
        }

    # FIXED: Added __hash__ and __eq__ for potential use in sets (deduplication)
    def __hash__(self):
        """Hash based on sorted objectives and type for deduplication"""
        # Exclude quantitative_measure, resolution_options, metadata, and detected_at from hash
        # as these fields might change even if the conflict itself is the same fundamental issue
        return hash(
            (
                tuple(sorted(self.objectives)),
                self.conflict_type,
                self.severity,
                self.description,
            )
        )

    def __eq__(self, other):
        """Check for equality (used for deduplication)"""
        if not isinstance(other, Conflict):
            return NotImplemented
        # Compare based on fields included in the hash (fundamental conflict definition)
        return (
            set(self.objectives) == set(other.objectives)
            and self.conflict_type == other.conflict_type
            and self.severity == other.severity
            and self.description == other.description
        )


@dataclass
class MultiObjectiveTension:
    """Analysis of tension across multiple objectives"""

    objectives: List[str]
    tension_matrix: np.ndarray  # Could be list of lists if numpy failed
    overall_tension: float
    primary_conflicts: List[Conflict]
    recommendations: List[str]
    pareto_optimal: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class GoalConflictDetector:
    """
    Detects conflicts between objectives

    Analyzes proposals and objective sets to identify:
    - Direct conflicts (mutually exclusive goals)
    - Indirect conflicts (shared resource contention)
    - Constraint violations
    - Unacceptable trade-offs

    Suggests resolutions for detected conflicts.
    """

    # --- START FIX: Modify __init__ for optional hierarchy ---
    def __init__(self, objective_hierarchy: Optional[ObjectiveHierarchy] = None):
        """
        Initialize conflict detector

        Args:
            objective_hierarchy: ObjectiveHierarchy instance (optional, defaults to fallback)
        """

        if objective_hierarchy is None:
            if not OBJECTIVE_HIERARCHY_AVAILABLE:
                logger.warning(
                    "No ObjectiveHierarchy provided and import failed. Using MagicMock fallback."
                )
                self.objective_hierarchy = ObjectiveHierarchy()  # Instantiates the mock
            else:
                # Import succeeded, but none provided. Create a default instance.
                logger.info(
                    "No ObjectiveHierarchy provided. Creating default ObjectiveHierarchy instance."
                )
                self.objective_hierarchy = ObjectiveHierarchy()  # Create default instance

        # FIXED: Allow Mock/MagicMock types for testing
        elif OBJECTIVE_HIERARCHY_AVAILABLE and not isinstance(
            objective_hierarchy, (RealObjectiveHierarchy, Mock, MagicMock)
        ):
            raise TypeError(
                f"objective_hierarchy must be an instance of RealObjectiveHierarchy or a Mock, got {type(objective_hierarchy)}"
            )
        elif not OBJECTIVE_HIERARCHY_AVAILABLE and not isinstance(
            objective_hierarchy, (Mock, MagicMock)
        ):
            raise TypeError(
                f"objective_hierarchy must be an instance of a Mock (RealObjectiveHierarchy failed to import), got {type(objective_hierarchy)}"
            )

        else:
            self.objective_hierarchy = objective_hierarchy
        # --- END FIX ---

        # Conflict detection rules
        self.conflict_rules = self._initialize_conflict_rules()

        # Detected conflicts history
        self.conflict_history = deque(maxlen=1000)

        # Resolution strategies
        self.resolution_strategies = self._initialize_resolution_strategies()

        # Statistics
        self.stats = defaultdict(int)

        # Thread safety
        self.lock = threading.RLock()

        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        logger.info("GoalConflictDetector initialized")

    def _initialize_conflict_rules(self) -> List[Dict[str, Any]]:
        """Initialize conflict detection rules"""

        return [
            {
                "name": "speed_accuracy_tradeoff",
                "objectives": ["efficiency", "prediction_accuracy"],
                "type": ConflictType.TRADEOFF,
                "severity": ConflictSeverity.MEDIUM,
                "description": "Speed and accuracy typically trade off",
            },
            {
                "name": "safety_performance",
                "objectives": ["safety", "efficiency"],
                "type": ConflictType.TRADEOFF,
                "severity": ConflictSeverity.HIGH,
                "description": "Safety checks reduce performance",
            },
            {  # FIXED: Added rule for accuracy vs safety tradeoff
                "name": "accuracy_safety",
                "objectives": ["accuracy", "safety"],
                "type": ConflictType.TRADEOFF,
                "severity": ConflictSeverity.HIGH,
                "description": "Accuracy directly conflicts with efficiency; Safety checks reduce performance",
            },
            {
                "name": "exploration_exploitation",
                "objectives": ["exploration", "exploitation"],
                "type": ConflictType.DIRECT,
                "severity": ConflictSeverity.MEDIUM,
                "description": "Cannot maximize both exploration and exploitation simultaneously",
            },
            # Add more rules based on domain knowledge
        ]

    def _initialize_resolution_strategies(
        self,
    ) -> Dict[ConflictType, List[Dict[str, Any]]]:
        """Initialize conflict resolution strategies"""

        return {
            ConflictType.DIRECT: [
                {
                    "name": "sequential",
                    "description": "Optimize objectives sequentially based on priority",
                    "applicability": "Always applicable",
                    # Use _generate_specific_actions method to get actions dynamically
                },
                {
                    "name": "weighted_sum",
                    "description": "Use weighted combination of objectives",
                    "applicability": "When objectives are commensurable",
                    # Use _generate_specific_actions method
                },
            ],
            ConflictType.INDIRECT: [
                {
                    "name": "resource_allocation",
                    "description": "Allocate shared resources based on priority",
                    "applicability": "When resource limits are known",
                    # Use _generate_specific_actions method
                },
                {
                    "name": "time_sharing",
                    "description": "Alternate between objectives over time",
                    "applicability": "When temporal flexibility exists",
                    # Use _generate_specific_actions method
                },
            ],
            ConflictType.CONSTRAINT: [
                {
                    "name": "constraint_relaxation",
                    "description": "Relax less critical constraints",
                    "applicability": "When constraints have different priorities",
                    # Use _generate_specific_actions method
                },
                {
                    "name": "constraint_reformulation",
                    "description": "Reformulate constraints to be compatible",
                    "applicability": "When constraints can be restated",
                    # Use _generate_specific_actions method
                },
            ],
            ConflictType.PRIORITY: [
                {
                    "name": "priority_reordering",
                    "description": "Adjust priority ordering",
                    "applicability": "When priorities are flexible",
                    # Use _generate_specific_actions method
                }
            ],
            ConflictType.TRADEOFF: [
                {
                    "name": "pareto_optimization",
                    "description": "Find Pareto-optimal solution",
                    "applicability": "Multi-objective optimization contexts",
                    # Use _generate_specific_actions method
                },
                {
                    "name": "bounded_optimization",
                    "description": "Optimize primary while constraining secondary",
                    "applicability": "When one objective is clearly primary",
                    # Use _generate_specific_actions method
                },
            ],
        }

    def detect_conflicts(self, proposal: Dict[str, Any]) -> List[Conflict]:
        """
        Alias for detect_conflicts_in_proposal for backward/test compatibility.
        """
        logger.debug(
            "Using deprecated alias 'detect_conflicts'. Use 'detect_conflicts_in_proposal' instead."
        )
        return self.detect_conflicts_in_proposal(proposal)

    def detect_conflicts_in_proposal(self, proposal: Dict[str, Any]) -> List[Conflict]:
        """
        Detect conflicts in a proposal

        Args:
            proposal: Proposed action or modification

        Returns:
            List of detected conflicts
        """

        with self.lock:
            all_conflicts = []
            proposal_id = proposal.get("id", "unknown")
            logger.debug(f"Analyzing proposal {proposal_id} for conflicts...")

            # Extract objectives from proposal
            proposal_objectives = self._extract_objectives_from_proposal(proposal)
            logger.debug(f"Extracted objectives: {proposal_objectives}")

            # Always check constraint conflicts, even for single objectives
            # Constraint violations ARE conflicts!
            logger.debug("Checking constraint conflicts...")
            constraint_conflicts = self._detect_constraint_conflicts(
                proposal_objectives, proposal
            )
            all_conflicts.extend(constraint_conflicts)
            if constraint_conflicts:
                logger.debug(f"Found {len(constraint_conflicts)} constraint conflicts.")

            # Check multi-objective conflicts only if multiple objectives
            if len(proposal_objectives) >= 2:
                logger.debug("Checking multi-objective conflicts...")
                # Check for direct conflicts
                direct_conflicts = self._detect_direct_conflicts(
                    proposal_objectives, proposal
                )
                all_conflicts.extend(direct_conflicts)
                if direct_conflicts:
                    logger.debug(f"Found {len(direct_conflicts)} direct conflicts.")

                # Check for resource conflicts
                resource_conflicts = self._detect_resource_conflicts(
                    proposal_objectives, proposal
                )
                all_conflicts.extend(resource_conflicts)
                if resource_conflicts:
                    logger.debug(f"Found {len(resource_conflicts)} resource conflicts.")

                # Check for priority violations
                priority_conflicts = self._detect_priority_violations(
                    proposal_objectives, proposal
                )
                all_conflicts.extend(priority_conflicts)
                if priority_conflicts:
                    logger.debug(f"Found {len(priority_conflicts)} priority conflicts.")

                # Check against known conflict patterns
                pattern_conflicts = self._check_conflict_patterns(proposal_objectives)
                all_conflicts.extend(pattern_conflicts)
                if pattern_conflicts:
                    logger.debug(f"Found {len(pattern_conflicts)} pattern conflicts.")

            # Deduplicate conflicts (can be detected by multiple methods)
            unique_conflicts = list(set(all_conflicts))
            logger.debug(f"Total unique conflicts detected: {len(unique_conflicts)}")

            # Add resolution suggestions to each conflict
            for conflict in unique_conflicts:
                conflict.resolution_options = self._suggest_resolutions(conflict)

            # Remember conflicts
            for conflict in unique_conflicts:
                self.conflict_history.append(conflict)

            # FIXED: Always update stats, even for proposals with no conflicts
            self.stats["proposals_analyzed"] += 1
            if unique_conflicts:
                self.stats["conflicts_detected"] += len(unique_conflicts)
                for conflict in unique_conflicts:
                    self.stats[f"conflict_type_{conflict.conflict_type.value}"] += 1
                    self.stats[f"conflict_severity_{conflict.severity.value}"] += 1

            return unique_conflicts

    def detect_conflicts_in_query(self, query: str) -> List[Conflict]:
        """
        Detect conflicts in a query string by analyzing it as a proposal
        
        Args:
            query: Query string to analyze
            
        Returns:
            List of detected conflicts
        """
        # Convert query string to a proposal dict
        proposal = {
            "id": f"query_{hash(query)}",
            "query": query,
            "description": query,
            "type": "query_analysis"
        }
        
        # Analyze objectives mentioned in the query
        # Extract common objective keywords from the query
        objectives = []
        query_lower = query.lower()
        
        # Use mock-safe attribute access
        hierarchy_objectives = {}
        if hasattr(self.objective_hierarchy, "objectives") and isinstance(
            self.objective_hierarchy.objectives, dict
        ):
            hierarchy_objectives = self.objective_hierarchy.objectives
        
        # Check if query mentions any known objectives
        for obj_name in hierarchy_objectives.keys():
            if obj_name.lower() in query_lower or obj_name.lower().replace('_', ' ') in query_lower:
                objectives.append(obj_name)
        
        # If objectives found, add them to the proposal
        if objectives:
            proposal["objectives"] = objectives
        
        # Use existing conflict detection logic
        return self.detect_conflicts_in_proposal(proposal)

    def analyze_multi_objective_tension(
        self, objectives: List[str]
    ) -> MultiObjectiveTension:
        """
        Analyze tension across multiple objectives

        Computes tension matrix showing pairwise conflicts

        Args:
            objectives: List of objective names

        Returns:
            Multi-objective tension analysis
        """

        with self.lock:
            n = len(objectives)

            if n == 0:
                return self._empty_tension_analysis()

            # FIXED: Handle single objective case
            if n == 1:
                return MultiObjectiveTension(
                    objectives=objectives,
                    tension_matrix=self._np.zeros((1, 1)),  # Use self._np
                    overall_tension=0.0,
                    primary_conflicts=[],
                    recommendations=["Single objective - no conflicts possible"],
                    pareto_optimal=True,
                    metadata={
                        "analyzed_at": time.time(),
                        "num_objectives": 1,
                        "max_tension": 0.0,
                    },
                )

            # Build tension matrix
            tension_matrix = self._np.zeros((n, n))  # Use self._np
            primary_conflicts = []

            for i, obj_i in enumerate(objectives):
                for j, obj_j in enumerate(objectives):
                    if i >= j:
                        continue

                    # Check for conflict
                    # Use mock-safe attribute access
                    conflict = None
                    if hasattr(self.objective_hierarchy, "find_conflicts") and callable(
                        self.objective_hierarchy.find_conflicts
                    ):
                        conflict = self.objective_hierarchy.find_conflicts(obj_i, obj_j)

                    if conflict:  # Ensure conflict is not None and is dict-like
                        # Map severity to tension value
                        severity_map = {
                            "negligible": 0.1,  # Assign numeric values
                            "low": 0.3,
                            "medium": 0.6,
                            "high": 0.9,
                            "critical": 1.0,
                        }
                        # Safely get severity string, default to 'medium'
                        severity_str = conflict.get("severity", "medium")
                        tension = severity_map.get(severity_str, 0.5)

                        tension_matrix[i, j] = tension
                        tension_matrix[j, i] = tension

                        # Record as primary conflict if significant
                        if tension >= 0.6:
                            # Safely get conflict type string, default to TRADEOFF
                            conflict_type_str = conflict.get("type", "tradeoff")
                            try:
                                conflict_type_enum = ConflictType(conflict_type_str)
                            except ValueError:
                                conflict_type_enum = ConflictType.TRADEOFF  # Fallback

                            primary_conflicts.append(
                                Conflict(
                                    objectives=[obj_i, obj_j],
                                    conflict_type=conflict_type_enum,
                                    severity=self._severity_from_tension(tension),
                                    description=conflict.get(
                                        "description", "Objective conflict"
                                    ),
                                    quantitative_measure=tension,
                                )
                            )

            # FIXED: Calculate overall tension correctly for n >= 2 using self._np
            upper_triangle_indices = self._np.triu_indices(n, k=1)
            # Check if indices are valid before indexing (FakeNumpy might return different structure)
            if (
                isinstance(upper_triangle_indices, tuple)
                and len(upper_triangle_indices) == 2
                and len(upper_triangle_indices[0]) > 0
            ):
                # Extract values using the indices
                upper_triangle_values = [
                    tension_matrix[r][c]
                    for r, c in zip(
                        upper_triangle_indices[0], upper_triangle_indices[1]
                    )
                ]
                overall_tension = float(self._np.mean(upper_triangle_values))
            else:
                overall_tension = 0.0  # Should not happen for n >= 2

            # Check if Pareto optimal
            pareto_optimal = self._check_pareto_optimality(objectives)

            # Generate recommendations
            recommendations = self._generate_tension_recommendations(
                objectives, tension_matrix, primary_conflicts, pareto_optimal
            )

            analysis = MultiObjectiveTension(
                objectives=objectives,
                tension_matrix=tension_matrix,  # Store matrix (or list of lists)
                overall_tension=overall_tension,
                primary_conflicts=primary_conflicts,
                recommendations=recommendations,
                pareto_optimal=pareto_optimal,
                metadata={
                    "analyzed_at": time.time(),
                    "num_objectives": n,
                    "max_tension": (
                        float(self._np.max(tension_matrix)) if n > 0 else 0.0
                    ),  # Use self._np
                },
            )

            self.stats["tension_analyses"] += 1

            return analysis

    def suggest_resolution(self, conflict: Conflict) -> List[Dict[str, Any]]:
        """
        Suggest resolutions for a conflict

        Args:
            conflict: Conflict to resolve

        Returns:
            List of resolution suggestions
        """
        # Ensure input is a Conflict object
        if not isinstance(conflict, Conflict):
            logger.warning(
                "suggest_resolution received non-Conflict object: %s", type(conflict)
            )
            # Attempt to convert if it's a dict from serialization
            if (
                isinstance(conflict, dict)
                and "objectives" in conflict
                and "conflict_type" in conflict
                and "severity" in conflict
            ):
                try:
                    # Attempt to parse enums safely
                    conflict_type = ConflictType(conflict["conflict_type"])
                    severity = ConflictSeverity(conflict["severity"])

                    conflict = Conflict(
                        objectives=conflict["objectives"],
                        conflict_type=conflict_type,
                        severity=severity,
                        description=conflict.get("description", ""),
                    )
                except ValueError:  # Invalid enum value
                    logger.error(f"Invalid enum value in conflict dict: {conflict}")
                    return []
            else:
                logger.error(
                    "Input to suggest_resolution is not a Conflict object or valid dict."
                )
                return []

        return self._suggest_resolutions(conflict)

    def check_constraint_violations(
        self, proposal: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check if proposal violates objective constraints

        Args:
            proposal: Proposal to check, containing target values

        Returns:
            List of constraint violations
        """

        with self.lock:
            violations = []

            # --- START FIX: New logic to find objectives to check ---
            objectives_to_check = set()  # Use a set for auto-deduplication

            # Use mock-safe attribute access for hierarchy.objectives
            hierarchy_objectives = {}
            if hasattr(self.objective_hierarchy, "objectives") and isinstance(
                self.objective_hierarchy.objectives, dict
            ):
                hierarchy_objectives = self.objective_hierarchy.objectives

            # This new logic checks *all* keys in the proposal against the known objectives
            # to find target values (e.g., proposal['efficiency'] = 1.5).

            known_fields = {
                "id",
                "type",
                "description",
                "objective",
                "objectives",
                "optimize_for",
                "objective_weights",
                "predicted_outcomes",
                "constraints",
                "implementation",
                "estimated_cost",
                "estimated_duration",
                "complexity",
            }

            for key, value in proposal.items():
                if key not in known_fields and key in hierarchy_objectives:
                    # This is a direct objective setting, e.g., "prediction_accuracy": 0.5
                    objectives_to_check.add(key)
                elif (
                    key == "objective" and isinstance(value, str) and value in proposal
                ):
                    # Handle {'objective': 'efficiency', 'efficiency': 0.8}
                    objectives_to_check.add(value)
                elif key == "objectives" and isinstance(value, list):
                    # Handle {'objectives': ['efficiency'], 'efficiency': 0.8}
                    for obj_name in value:
                        if obj_name in proposal:
                            objectives_to_check.add(obj_name)
                elif key == "objective_weights" and isinstance(value, dict):
                    # Handle {'objective_weights': {'efficiency': 0.5}, 'efficiency': 0.8}
                    for obj_name in value:
                        if obj_name in proposal:
                            objectives_to_check.add(obj_name)
            # --- END FIX ---

            logger.debug(
                f"Checking constraints for objectives with values: {objectives_to_check}"
            )

            for obj_name in objectives_to_check:
                obj_data = hierarchy_objectives.get(
                    obj_name
                )  # Safely get objective data
                if obj_data:
                    # Get the target value from the proposal
                    value = proposal.get(obj_name)

                    # Only check if a value was actually provided and is numeric
                    if value is None:
                        logger.debug(
                            f"No target value for {obj_name}, skipping constraint check."
                        )
                        continue

                    try:
                        value_num = float(value)
                    except (TypeError, ValueError):
                        logger.debug(
                            f"Target value for {obj_name} is not numeric ({value}), skipping constraint check."
                        )
                        continue

                    # Safely access constraints (might be attribute or dict key)
                    constraints = {}
                    if hasattr(obj_data, "constraints") and isinstance(
                        obj_data.constraints, dict
                    ):
                        constraints = obj_data.constraints
                    elif (
                        isinstance(obj_data, dict)
                        and "constraints" in obj_data
                        and isinstance(obj_data["constraints"], dict)
                    ):
                        constraints = obj_data["constraints"]

                    # Check min constraint
                    if "min" in constraints:
                        try:
                            min_limit = float(constraints["min"])
                            if value_num < min_limit:
                                violation_amount = min_limit - value_num
                                logger.debug(
                                    f"Constraint violation: {obj_name} value {value_num} < min {min_limit}"
                                )
                                violations.append(
                                    {
                                        "objective": obj_name,
                                        "constraint": "minimum",
                                        "value": value_num,
                                        "limit": min_limit,
                                        "violation": violation_amount,
                                    }
                                )
                        except (TypeError, ValueError):
                            logger.warning(
                                f"Invalid 'min' constraint value for {obj_name}: {constraints['min']}"
                            )

                    # Check max constraint
                    if "max" in constraints:
                        try:
                            max_limit = float(constraints["max"])
                            if value_num > max_limit:
                                violation_amount = value_num - max_limit
                                logger.debug(
                                    f"Constraint violation: {obj_name} value {value_num} > max {max_limit}"
                                )
                                violations.append(
                                    {
                                        "objective": obj_name,
                                        "constraint": "maximum",
                                        "value": value_num,
                                        "limit": max_limit,
                                        "violation": violation_amount,
                                    }
                                )
                        except (TypeError, ValueError):
                            logger.warning(
                                f"Invalid 'max' constraint value for {obj_name}: {constraints['max']}"
                            )
                else:
                    logger.debug(
                        f"Objective {obj_name} not found in hierarchy during constraint check."
                    )

            return violations

    def validate_tradeoff_acceptability(
        self, tradeoff: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate if a trade-off is acceptable

        Args:
            tradeoff: Trade-off specification (sacrifice -> gain)

        Returns:
            Validation result
        """

        with self.lock:
            sacrifice_obj = tradeoff.get("sacrifice")
            gain_obj = tradeoff.get("gain")
            sacrifice_amount = tradeoff.get("sacrifice_amount", 0.0)
            gain_amount = tradeoff.get("gain_amount", 0.0)

            issues = []

            # Use mock-safe attribute access
            hierarchy_objectives = {}
            if hasattr(self.objective_hierarchy, "objectives") and isinstance(
                self.objective_hierarchy.objectives, dict
            ):
                hierarchy_objectives = self.objective_hierarchy.objectives

            # Check if sacrificing critical objective
            obj_data = hierarchy_objectives.get(sacrifice_obj)
            if obj_data:
                # Safely get priority
                priority = 1  # Default non-critical
                if hasattr(obj_data, "priority"):
                    priority = getattr(obj_data, "priority", 1)
                elif isinstance(obj_data, dict):
                    priority = obj_data.get("priority", 1)

                if priority == 0:  # Critical
                    issues.append(
                        {
                            "type": "critical_sacrifice",
                            "severity": "critical",  # Changed severity
                            "description": f"Cannot sacrifice critical objective: {sacrifice_obj}",
                        }
                    )

            # Check if trade-off ratio is reasonable
            if gain_amount > 0 and sacrifice_amount > 0:
                ratio = gain_amount / sacrifice_amount
                if ratio < 0.5:  # Losing more than gaining
                    issues.append(
                        {
                            "type": "poor_ratio",
                            "severity": "medium",
                            "description": f"Trade-off ratio unfavorable: {ratio:.2f}",
                        }
                    )

            # Check if objectives are actually in conflict (mock-safe)
            conflict = None
            if hasattr(self.objective_hierarchy, "find_conflicts") and callable(
                self.objective_hierarchy.find_conflicts
            ):
                conflict = self.objective_hierarchy.find_conflicts(
                    sacrifice_obj, gain_obj
                )

            if not conflict:
                issues.append(
                    {
                        "type": "unnecessary_tradeoff",
                        "severity": "low",
                        "description": "Objectives do not conflict, trade-off may be unnecessary",
                    }
                )

            acceptable = (
                len([i for i in issues if i["severity"] in ["high", "critical"]]) == 0
            )
            # Simple confidence calc - adjust based on severity and number of issues
            confidence = max(
                0.0,
                1.0
                - (len(issues) * 0.1)
                - sum(
                    (
                        0.2
                        if i["severity"] == "high"
                        else 0.4 if i["severity"] == "critical" else 0
                    )
                    for i in issues
                ),
            )

            return {
                "acceptable": acceptable,
                "issues": issues,
                "recommendation": "APPROVE" if acceptable else "REJECT",
                "confidence": confidence,
            }

    def _extract_objectives_from_proposal(self, proposal: Dict[str, Any]) -> List[str]:
        """Extract objective names from proposal - FIXED: Defensive check for malformed input"""

        objectives = []
        proposal_id_str = (
            f"proposal {proposal.get('id', 'N/A')}"  # More descriptive log prefix
        )
        logger.debug(f"Extracting objectives from {proposal_id_str}")

        # Ensure proposal is a dict
        if not isinstance(proposal, dict):
            logger.warning(
                f"Proposal provided is not a dict ({type(proposal)}). Cannot extract objectives."
            )
            return []

        # Explicit objective field
        obj_field = proposal.get("objective")
        if isinstance(obj_field, str) and obj_field:  # Ensure non-empty string
            objectives.append(obj_field)
            logger.debug(f"  [{proposal_id_str}] Found 'objective': {obj_field}")

        # Objectives in optimization targets
        opt_field = proposal.get("optimize_for")
        if isinstance(opt_field, str) and opt_field:
            objectives.append(opt_field)
            logger.debug(f"  [{proposal_id_str}] Found 'optimize_for': {opt_field}")
        elif isinstance(opt_field, list):
            # Filter out non-string or empty string elements
            valid_opts = [o for o in opt_field if isinstance(o, str) and o]
            objectives.extend(valid_opts)
            if valid_opts:
                logger.debug(
                    f"  [{proposal_id_str}] Found 'optimize_for' list: {valid_opts}"
                )

        # Multiple objectives field
        objs_field = proposal.get("objectives")
        if isinstance(objs_field, list):
            valid_objs = [o for o in objs_field if isinstance(o, str) and o]
            objectives.extend(valid_objs)
            if valid_objs:
                logger.debug(
                    f"  [{proposal_id_str}] Found 'objectives' list: {valid_objs}"
                )
        elif isinstance(objs_field, str) and objs_field:  # Handle single string case
            objectives.append(objs_field)
            logger.debug(
                f"  [{proposal_id_str}] Found 'objectives' string: {objs_field}"
            )

        # FIXED: Objective weights indicate multi-objective - add defensive check
        weights_field = proposal.get("objective_weights")
        # Check if it's actually a dict before trying to access .keys()
        if isinstance(weights_field, dict):
            valid_keys = [k for k in weights_field.keys() if isinstance(k, str) and k]
            objectives.extend(valid_keys)
            if valid_keys:
                logger.debug(
                    f"  [{proposal_id_str}] Found keys from 'objective_weights': {valid_keys}"
                )
        elif weights_field is not None:
            logger.warning(
                f"Malformed objective_weights in {proposal_id_str}: expected dict, got {type(weights_field)}"
            )

        # Add objectives from predicted_outcomes as well
        outcomes_field = proposal.get("predicted_outcomes")
        if isinstance(outcomes_field, dict):
            valid_keys = [k for k in outcomes_field.keys() if isinstance(k, str) and k]
            objectives.extend(valid_keys)
            if valid_keys:
                logger.debug(
                    f"  [{proposal_id_str}] Found keys from 'predicted_outcomes': {valid_keys}"
                )
        elif outcomes_field is not None:
            logger.warning(
                f"Malformed predicted_outcomes in {proposal_id_str}: expected dict, got {type(outcomes_field)}"
            )

        # FIXED: Add objectives from target values (e.g., proposal['efficiency'] = 0.8)
        # This is used by check_constraint_violations
        all_keys = list(proposal.keys())  # Snapshot keys
        known_fields = [
            "id",
            "type",
            "description",
            "objective",
            "objectives",
            "optimize_for",
            "objective_weights",
            "predicted_outcomes",
            "constraints",
            "implementation",
            "estimated_cost",
            "estimated_duration",
            "complexity",
        ]

        # Use mock-safe attribute access
        hierarchy_objectives = {}
        if hasattr(self.objective_hierarchy, "objectives") and isinstance(
            self.objective_hierarchy.objectives, dict
        ):
            hierarchy_objectives = self.objective_hierarchy.objectives

        potential_obj_keys = [
            k for k in all_keys if k not in known_fields and k in hierarchy_objectives
        ]
        if potential_obj_keys:
            objectives.extend(potential_obj_keys)
            logger.debug(
                f"  [{proposal_id_str}] Found keys from proposal root: {potential_obj_keys}"
            )

        # Remove duplicates while preserving order roughly
        unique_objectives = list(dict.fromkeys(objectives))
        logger.debug(
            f"  [{proposal_id_str}] Final unique objectives: {unique_objectives}"
        )

        return unique_objectives

    def _detect_direct_conflicts(
        self, objectives: List[str], proposal: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect direct conflicts between objectives"""

        conflicts = []
        logger.debug("Detecting direct conflicts among: %s", objectives)

        # Check each pair
        for i, obj_i in enumerate(objectives):
            for obj_j in objectives[i + 1 :]:
                # Check if objectives are known to conflict (mock-safe)
                hierarchy_conflict = None
                if hasattr(self.objective_hierarchy, "find_conflicts") and callable(
                    self.objective_hierarchy.find_conflicts
                ):
                    hierarchy_conflict = self.objective_hierarchy.find_conflicts(
                        obj_i, obj_j
                    )

                # Check conflict type safely
                if hierarchy_conflict and isinstance(hierarchy_conflict, dict):
                    conflict_type_str = hierarchy_conflict.get("type")
                    # Use HierarchyConflictType for comparison if available, otherwise string
                    direct_type = (
                        HierarchyConflictType.DIRECT.value
                        if hasattr(HierarchyConflictType, "DIRECT")
                        else "direct"
                    )

                    if conflict_type_str == direct_type:
                        logger.debug(
                            f"  Direct conflict found between {obj_i} and {obj_j} from hierarchy"
                        )
                        conflicts.append(
                            Conflict(
                                objectives=[obj_i, obj_j],
                                conflict_type=ConflictType.DIRECT,  # Use local ConflictType enum
                                severity=self._map_severity(
                                    hierarchy_conflict.get("severity", "medium")
                                ),
                                description=hierarchy_conflict.get(
                                    "description", "Direct conflict based on hierarchy"
                                ),
                                metadata={"source": "hierarchy"},
                            )
                        )

        return conflicts

    def _detect_constraint_conflicts(
        self, objectives: List[str], proposal: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect constraint-based conflicts"""

        conflicts = []
        logger.debug("Detecting constraint conflicts for objectives: %s", objectives)

        # FIXED: Corrected call to check_constraint_violations
        violations = self.check_constraint_violations(
            proposal
        )  # Pass the whole proposal

        # Use mock-safe attribute access
        hierarchy_objectives = {}
        if hasattr(self.objective_hierarchy, "objectives") and isinstance(
            self.objective_hierarchy.objectives, dict
        ):
            hierarchy_objectives = self.objective_hierarchy.objectives

        for violation in violations:
            obj_name = violation["objective"]
            # Determine severity based on objective priority
            severity = ConflictSeverity.HIGH
            obj_data = hierarchy_objectives.get(obj_name)
            if obj_data:
                priority = 1  # Default non-critical
                if hasattr(obj_data, "priority"):
                    priority = getattr(obj_data, "priority", 1)
                elif isinstance(obj_data, dict):
                    priority = obj_data.get("priority", 1)
                if priority == 0:
                    severity = ConflictSeverity.CRITICAL

            logger.debug(f"  Constraint violation conflict for {obj_name}: {violation}")
            conflicts.append(
                Conflict(
                    objectives=[obj_name],  # Single objective violates its constraint
                    conflict_type=ConflictType.CONSTRAINT,
                    severity=severity,
                    description=f"Constraint violation: {violation['constraint']} limit {violation['limit']} (value={violation['value']:.3f})",
                    quantitative_measure=violation["violation"],
                    metadata=violation,
                )
            )

        # Check for conflicting constraints between objectives (only if multiple)
        if len(objectives) >= 2:
            for i, obj_i_name in enumerate(objectives):
                for obj_j_name in objectives[i + 1 :]:
                    obj_i_data = hierarchy_objectives.get(obj_i_name)
                    obj_j_data = hierarchy_objectives.get(obj_j_name)

                    if obj_i_data and obj_j_data:
                        # Safely get constraints
                        constraints_i = {}
                        if hasattr(obj_i_data, "constraints") and isinstance(
                            obj_i_data.constraints, dict
                        ):
                            constraints_i = obj_i_data.constraints
                        elif (
                            isinstance(obj_i_data, dict) and "constraints" in obj_i_data
                        ):
                            constraints_i = obj_i_data.get("constraints", {})

                        constraints_j = {}
                        if hasattr(obj_j_data, "constraints") and isinstance(
                            obj_j_data.constraints, dict
                        ):
                            constraints_j = obj_j_data.constraints
                        elif (
                            isinstance(obj_j_data, dict) and "constraints" in obj_j_data
                        ):
                            constraints_j = obj_j_data.get("constraints", {})

                        # Check if constraints are incompatible
                        if self._constraints_conflict(constraints_i, constraints_j):
                            logger.debug(
                                f"  Incompatible constraints between {obj_i_name} and {obj_j_name}"
                            )
                            conflicts.append(
                                Conflict(
                                    objectives=[obj_i_name, obj_j_name],
                                    conflict_type=ConflictType.CONSTRAINT,
                                    severity=ConflictSeverity.MEDIUM,
                                    description=f"Incompatible constraints between {obj_i_name} and {obj_j_name}",
                                )
                            )

        return conflicts

    def _detect_resource_conflicts(
        self, objectives: List[str], proposal: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect resource contention conflicts"""

        conflicts = []
        logger.debug("Detecting resource conflicts among: %s", objectives)

        # Check for shared resource contention
        shared_resources = self._find_shared_resources(objectives)

        for resource, competing_objectives in shared_resources.items():
            if len(competing_objectives) > 1:
                logger.debug(
                    f"  Resource conflict found for '{resource}' involving {competing_objectives}"
                )
                conflicts.append(
                    Conflict(
                        objectives=competing_objectives,
                        conflict_type=ConflictType.INDIRECT,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Resource contention for: {resource}",
                        metadata={"resource": resource},
                    )
                )

        return conflicts

    def _detect_priority_violations(
        self, objectives: List[str], proposal: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect priority ordering violations"""

        conflicts = []
        logger.debug("Detecting priority violations among: %s", objectives)

        # Check if proposal violates priority ordering
        weights = proposal.get("objective_weights")

        # Use mock-safe attribute access
        hierarchy_objectives = {}
        if hasattr(self.objective_hierarchy, "objectives") and isinstance(
            self.objective_hierarchy.objectives, dict
        ):
            hierarchy_objectives = self.objective_hierarchy.objectives

        # Defensive check
        if isinstance(weights, dict):
            for obj_name, weight in weights.items():
                obj_data = hierarchy_objectives.get(obj_name)
                if obj_data:
                    # Safely get priority
                    priority = 1  # Default non-critical
                    if hasattr(obj_data, "priority"):
                        priority = getattr(obj_data, "priority", 1)
                    elif isinstance(obj_data, dict):
                        priority = obj_data.get("priority", 1)

                    # Check if low-priority objective gets high weight
                    if priority >= 2 and weight > 0.7:
                        logger.debug(
                            f"  Priority violation: Low-priority '{obj_name}' (p={priority}) has high weight {weight}"
                        )
                        conflicts.append(
                            Conflict(
                                objectives=[obj_name],
                                conflict_type=ConflictType.PRIORITY,
                                severity=ConflictSeverity.LOW,
                                description=f"Low-priority objective {obj_name} (priority={priority}) given high weight {weight:.2f}",
                            )
                        )

        return conflicts

    def _check_conflict_patterns(self, objectives: List[str]) -> List[Conflict]:
        """Check against known conflict patterns"""

        conflicts = []
        logger.debug("Checking known conflict patterns for: %s", objectives)

        for rule in self.conflict_rules:
            rule_objs = set(rule["objectives"])
            proposal_objs = set(objectives)

            if rule_objs.issubset(proposal_objs):
                logger.debug(
                    f"  Matched pattern '{rule['name']}' for objectives {rule_objs}"
                )
                conflicts.append(
                    Conflict(
                        objectives=list(rule_objs),
                        conflict_type=rule[
                            "type"
                        ],  # Assumes ConflictType enum value matches rule type string
                        severity=rule[
                            "severity"
                        ],  # Assumes ConflictSeverity enum value matches rule severity string
                        description=rule["description"],
                        metadata={"rule": rule["name"]},
                    )
                )

        return conflicts

    def _suggest_resolutions(self, conflict: Conflict) -> List[Dict[str, Any]]:
        """Suggest resolutions for conflict"""

        strategies = self.resolution_strategies.get(conflict.conflict_type, [])
        logger.debug(
            f"Suggesting resolutions for {conflict.conflict_type.value} conflict involving {conflict.objectives}"
        )

        suggestions = []
        for strategy_def in strategies:
            suggestion = {
                "strategy": strategy_def["name"],
                "description": strategy_def["description"],
                "applicability": strategy_def["applicability"],
                # Generate specific actions dynamically
                "specific_actions": self._generate_specific_actions(
                    conflict, strategy_def
                ),
            }
            suggestions.append(suggestion)

        return suggestions

    def _generate_specific_actions(
        self, conflict: Conflict, strategy: Dict[str, Any]
    ) -> List[str]:
        """Generate specific actions for resolution strategy"""

        actions = []
        strategy_name = strategy.get("name")
        logger.debug(f"Generating specific actions for strategy: {strategy_name}")

        try:
            if strategy_name == "weighted_sum":
                actions.append(f"Assign weights to {', '.join(conflict.objectives)}")
                actions.append("Optimize weighted combination")

            elif strategy_name == "sequential":
                # FIXED: Call get_priority_order() correctly (mock-safe)
                full_priority_order = []
                if hasattr(self.objective_hierarchy, "get_priority_order") and callable(
                    self.objective_hierarchy.get_priority_order
                ):
                    full_priority_order = self.objective_hierarchy.get_priority_order()

                # Filter the full list based on the conflicting objectives
                ordered_objs = [
                    obj for obj in full_priority_order if obj in conflict.objectives
                ]
                if ordered_objs:
                    actions.append(f"Optimize in order: {' → '.join(ordered_objs)}")
                else:
                    # Fallback if none of the conflicting objectives are in the main priority order (unlikely but possible)
                    actions.append(
                        f"Determine priority order for {conflict.objectives} and optimize sequentially"
                    )

            elif strategy_name == "pareto_optimization":
                actions.append(
                    f"Find Pareto frontier for objectives: {conflict.objectives}"
                )
                actions.append("Select point from frontier based on preferences")

            elif strategy_name == "constraint_relaxation":
                actions.append(
                    f"Identify non-critical constraints related to {conflict.objectives}"
                )
                actions.append("Relax constraints based on priority")

            elif strategy_name == "bounded_optimization":
                actions.append(f"Select primary objective from: {conflict.objectives}")
                actions.append(
                    "Constrain secondary objectives within acceptable bounds"
                )

            elif strategy_name == "resource_allocation":
                resource = conflict.metadata.get("resource", "shared_resource")
                actions.append(
                    f"Allocate resource '{resource}' based on objective priorities among {conflict.objectives}"
                )

            elif strategy_name == "time_sharing":
                actions.append(
                    f"Implement time-sharing schedule for competing objectives: {conflict.objectives}"
                )

            elif strategy_name == "constraint_reformulation":
                actions.append(
                    f"Analyze and reformulate incompatible constraints for {conflict.objectives}"
                )

            elif strategy_name == "priority_reordering":
                actions.append(
                    f"Review and potentially adjust priorities for conflicting objectives: {conflict.objectives}"
                )

        except Exception as e:
            # FIXED: Log the error with more context
            logger.error(
                f"Error generating specific actions for {strategy_name}: {e}",
                exc_info=True,
            )
            actions.append("Error generating specific actions - review manually.")

        return (
            actions
            if actions
            else ["Consult objective hierarchy and context to resolve."]
        )

    def _find_shared_resources(self, objectives: List[str]) -> Dict[str, List[str]]:
        """Find resources shared between objectives"""

        shared = defaultdict(list)

        # Use mock-safe attribute access
        hierarchy_objectives = {}
        if hasattr(self.objective_hierarchy, "objectives") and isinstance(
            self.objective_hierarchy.objectives, dict
        ):
            hierarchy_objectives = self.objective_hierarchy.objectives

        for obj_name in objectives:
            obj_data = hierarchy_objectives.get(obj_name)
            if obj_data:
                # Check metadata for resource requirements (handle attribute or dict key)
                resources = []
                if hasattr(obj_data, "metadata") and isinstance(
                    obj_data.metadata, dict
                ):
                    resources = obj_data.metadata.get("resources", [])
                elif (
                    isinstance(obj_data, dict)
                    and "metadata" in obj_data
                    and isinstance(obj_data["metadata"], dict)
                ):
                    resources = obj_data["metadata"].get("resources", [])

                if isinstance(resources, list):  # Ensure it's a list
                    for resource in resources:
                        if (
                            isinstance(resource, str) and resource
                        ):  # Ensure resource is a non-empty string
                            shared[resource].append(obj_name)

        # Return only truly shared resources
        return {r: objs for r, objs in shared.items() if len(objs) > 1}

    def _constraints_conflict(
        self, constraints_a: Dict[str, Any], constraints_b: Dict[str, Any]
    ) -> bool:
        """Check if two constraint sets conflict"""

        # Check if ranges don't overlap
        try:
            # Provide defaults that ensure no conflict if key is missing
            a_min = float(constraints_a.get("min", float("-inf")))
            a_max = float(constraints_a.get("max", float("inf")))
            b_min = float(constraints_b.get("min", float("-inf")))
            b_max = float(constraints_b.get("max", float("inf")))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid constraint value detected during conflict check. Assuming no conflict."
            )
            return False  # Cannot determine conflict if values are invalid

        # No overlap if max of one is less than min of other
        # Ensure finite values before comparison to avoid inf < inf issues
        if a_max != float("inf") and b_min != float("-inf") and a_max < b_min:
            return True
        if b_max != float("inf") and a_min != float("-inf") and b_max < a_min:
            return True

        return False

    def _check_pareto_optimality(self, objectives: List[str]) -> bool:
        """Check if objective set is Pareto optimal"""

        # Simplified check - assumes Pareto optimal unless severe conflicts exist
        # A more rigorous check would involve the CounterfactualObjectiveReasoner

        for i, obj_i in enumerate(objectives):
            for obj_j in objectives[i + 1 :]:
                # Mock-safe conflict check
                conflict = None
                if hasattr(self.objective_hierarchy, "find_conflicts") and callable(
                    self.objective_hierarchy.find_conflicts
                ):
                    conflict = self.objective_hierarchy.find_conflicts(obj_i, obj_j)

                # Safely check severity
                if conflict and isinstance(conflict, dict):
                    severity_str = conflict.get("severity")
                    critical_val = (
                        ConflictSeverity.CRITICAL.value
                        if hasattr(ConflictSeverity.CRITICAL, "value")
                        else "critical"
                    )
                    if severity_str == critical_val:
                        logger.debug(
                            f"Not Pareto optimal due to critical conflict between {obj_i} and {obj_j}"
                        )
                        return False

        return True  # Assume optimal if no critical conflicts found

    def _generate_tension_recommendations(
        self,
        objectives: List[str],
        tension_matrix: np.ndarray,  # Or list of lists
        conflicts: List[Conflict],
        pareto_optimal: bool,
    ) -> List[str]:
        """Generate recommendations based on tension analysis"""

        recommendations = []
        _np = self._np  # Use internal numpy/FakeNumpy alias

        # FIXED: Handle case where there might be no upper triangle elements (n < 2)
        n = len(objectives)
        overall_tension = 0.0
        if n >= 2:
            upper_triangle_indices = _np.triu_indices(n, k=1)
            # Check indices validity for numpy and FakeNumpy
            if (
                isinstance(upper_triangle_indices, tuple)
                and len(upper_triangle_indices) == 2
                and len(upper_triangle_indices[0]) > 0
            ):
                upper_triangle_values = [
                    tension_matrix[r][c]
                    for r, c in zip(
                        upper_triangle_indices[0], upper_triangle_indices[1]
                    )
                ]
                overall_tension = float(_np.mean(upper_triangle_values))

        if overall_tension < 0.3:
            recommendations.append(
                "Low overall tension - objectives appear largely compatible."
            )
        elif overall_tension < 0.6:
            recommendations.append(
                "Moderate tension detected - consider multi-objective optimization strategies (e.g., weighted sum, Pareto)."
            )
        else:
            recommendations.append(
                "High tension detected - prioritization, sequential optimization, or constraint relaxation may be necessary."
            )

        if not pareto_optimal:
            recommendations.append(
                "Current objective set might not be Pareto optimal; investigate potential improvements."
            )

        if conflicts:
            high_severity = [
                c
                for c in conflicts
                if c.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH]
            ]
            if high_severity:
                # Format conflict details safely
                conflict_details = []
                for c in high_severity[:3]:  # Show first 3
                    try:
                        conflict_details.append(
                            f"({c.objectives}, {c.conflict_type.value})"
                        )
                    except (
                        AttributeError
                    ):  # Handle case where conflict_type might not be enum
                        conflict_details.append(f"({c.objectives}, {c.conflict_type})")

                recommendations.append(
                    f"Address {len(high_severity)} high-severity conflict(s) first. Primary conflicts: {', '.join(conflict_details)}"
                )

        return recommendations

    def _severity_from_tension(self, tension: float) -> ConflictSeverity:
        """Map tension value to severity"""

        if tension >= 0.9:
            return ConflictSeverity.CRITICAL
        elif tension >= 0.7:
            return ConflictSeverity.HIGH
        elif tension >= 0.4:
            return ConflictSeverity.MEDIUM
        elif tension >= 0.2:
            return ConflictSeverity.LOW
        else:
            return ConflictSeverity.NEGLIGIBLE

    def _map_severity(self, severity_str: Optional[str]) -> ConflictSeverity:
        """Map string severity to enum, defaulting to MEDIUM"""
        if severity_str is None:
            return ConflictSeverity.MEDIUM

        mapping = {
            "critical": ConflictSeverity.CRITICAL,
            "high": ConflictSeverity.HIGH,
            "medium": ConflictSeverity.MEDIUM,
            "low": ConflictSeverity.LOW,
            "negligible": ConflictSeverity.NEGLIGIBLE,
        }

        # Handle potential enum values being passed
        try:
            return ConflictSeverity(severity_str.lower())
        except ValueError:
            # Fallback to mapping if direct conversion fails
            return mapping.get(severity_str.lower(), ConflictSeverity.MEDIUM)

    def _empty_tension_analysis(self) -> MultiObjectiveTension:
        """Create empty tension analysis"""
        _np = self._np  # Use internal numpy/FakeNumpy alias

        return MultiObjectiveTension(
            objectives=[],
            tension_matrix=_np.array([[]]),  # Use self._np
            overall_tension=0.0,
            primary_conflicts=[],
            recommendations=["No objectives to analyze"],
            pareto_optimal=True,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get conflict detection statistics"""

        with self.lock:
            proposals_analyzed = self.stats["proposals_analyzed"]
            conflicts_detected = self.stats["conflicts_detected"]

            # Safely calculate rates
            conflict_detection_rate = (
                (conflicts_detected / proposals_analyzed)
                if proposals_analyzed > 0
                else 0.0
            )

            # Filter stats dict safely
            conflicts_by_type = {}
            conflicts_by_severity = {}
            for k, v in self.stats.items():
                if isinstance(k, str):  # Ensure key is string
                    if k.startswith("conflict_type_"):
                        type_key = k.replace("conflict_type_", "")
                        conflicts_by_type[type_key] = v
                    elif k.startswith("conflict_severity_"):
                        severity_key = k.replace("conflict_severity_", "")
                        conflicts_by_severity[severity_key] = v

            return {
                "statistics": dict(self.stats),
                "conflict_history_size": len(self.conflict_history),
                "conflict_detection_rate": conflict_detection_rate,
                "conflicts_by_type": conflicts_by_type,
                "conflicts_by_severity": conflicts_by_severity,
            }
