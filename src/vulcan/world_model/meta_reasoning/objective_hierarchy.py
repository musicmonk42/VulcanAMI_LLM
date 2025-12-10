"""
objective_hierarchy.py - Objective hierarchy and relationship management
Part of the meta_reasoning subsystem for VULCAN-AMI

Manages graph of objectives and their relationships:
- Primary objectives (core design goals)
- Secondary objectives (support primary)
- Derived objectives (emergent from combinations)

Tracks dependencies, conflicts, and priority ordering.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
# import numpy as np # Original import
from typing import Any, Dict, List, Optional, Set, Tuple

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
        def zeros(shape):
            # Handle int or tuple/list shape
            if isinstance(shape, int):
                return [0.0] * shape
            if len(shape) == 1:
                return [0.0] * shape[0]
            if len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
            raise NotImplementedError("FakeNumpy only supports 1D/2D zeros")

        # Add other numpy functions used in this file if any

    np = FakeNumpy()
# --- END FIX ---


class ObjectiveType(Enum):
    """Type of objective in hierarchy"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DERIVED = "derived"


class ConflictType(Enum):
    """Type of conflict between objectives"""

    DIRECT = "direct"  # Cannot both be satisfied
    INDIRECT = "indirect"  # Tension through shared resources
    CONSTRAINT = "constraint"  # Constraint-based conflict


@dataclass
class Objective:
    """
    Single objective with constraints and metadata

    Represents a goal the system can optimize for with associated
    constraints, priorities, and relationships to other objectives.
    """

    name: str
    description: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 0=critical, 1=high, 2=normal, 3=low
    dependencies: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    objective_type: ObjectiveType = ObjectiveType.PRIMARY
    weight: float = 1.0
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    maximize: bool = True  # True=maximize, False=minimize
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def is_satisfied(self) -> bool:
        """Check if objective is currently satisfied"""
        if self.current_value is None or self.target_value is None:
            return False

        # Ensure values are numeric before comparison
        try:
            current = float(self.current_value)
            target = float(self.target_value)
        except (TypeError, ValueError):
            return False  # Cannot satisfy if values aren't numeric

        # Check constraints
        try:
            if "min" in self.constraints and current < float(self.constraints["min"]):
                return False
            if "max" in self.constraints and current > float(self.constraints["max"]):
                return False
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid constraint value for objective {self.name}. Cannot check satisfaction."
            )
            return False  # Cannot satisfy if constraints are invalid

        # Check target proximity
        tolerance = self.metadata.get("tolerance", 0.05)
        return abs(current - target) <= tolerance

    def distance_from_target(self) -> Optional[float]:
        """Calculate distance from target value"""
        if self.current_value is None or self.target_value is None:
            return None
        try:
            current = float(self.current_value)
            target = float(self.target_value)
            return abs(current - target)
        except (TypeError, ValueError):
            return None  # Cannot calculate distance if values aren't numeric

    def violates_constraints(self) -> bool:
        """Check if current value violates constraints"""
        if self.current_value is None:
            return False

        try:
            current = float(self.current_value)
            if "min" in self.constraints and current < float(self.constraints["min"]):
                return True
            if "max" in self.constraints and current > float(self.constraints["max"]):
                return True
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid constraint or current value for objective {self.name}. Cannot check violation."
            )
            return False  # Assume no violation if values/constraints are invalid

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "constraints": self.constraints,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "conflicts_with": self.conflicts_with,
            "objective_type": self.objective_type.value,
            "weight": self.weight,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "parent": self.parent,
            "children": self.children,
            "maximize": self.maximize,
            "satisfied": self.is_satisfied(),  # Call method here
            "violates_constraints": self.violates_constraints(),  # Call method here
            "distance_from_target": self.distance_from_target(),  # Call method here
            "metadata": self.metadata,
            "created_at": self.created_at,  # Add created_at
        }


class ObjectiveHierarchy:
    """
    Maintains graph of objectives and their relationships

    Manages:
    - Primary objectives: Core design goals
    - Secondary objectives: Support primary objectives
    - Derived objectives: Emerge from combinations

    Tracks dependencies, conflicts, and enforces consistency.
    """

    def __init__(self, design_spec: Optional[Dict[str, Any]] = None):
        """
        Initialize objective hierarchy

        Args:
            design_spec: Design specification with objectives
        """
        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy()

        self.design_spec = design_spec or {}

        # Objective storage
        self.objectives: Dict[str, Objective] = {}

        # Relationship graphs
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.conflict_graph: Dict[str, Set[str]] = defaultdict(set)
        self.parent_child_graph: Dict[str, Set[str]] = defaultdict(
            set
        )  # Maps parent -> set(children)

        # Computed properties
        self.priority_order: List[str] = []
        self.conflict_matrix: Optional[np.ndarray] = (
            None  # Could be list of lists if numpy failed
        )
        self.dependency_closure: Dict[str, Set[str]] = {}

        # Statistics
        self.stats = defaultdict(int)

        # Thread safety
        self.lock = threading.RLock()

        # Initialize from design spec
        self._initialize_from_design_spec()

        logger.info(
            "ObjectiveHierarchy initialized with %d objectives", len(self.objectives)
        )

    def _initialize_from_design_spec(self):
        """Initialize objectives from design specification"""

        objectives_spec = self.design_spec.get("objectives", {})

        if not objectives_spec:
            logger.warning(
                "No objectives found in design_spec. Creating default objectives."
            )
            self._create_default_objectives()
        else:
            # Load from spec
            for obj_name, obj_config in objectives_spec.items():
                if not isinstance(obj_config, dict):
                    logger.warning(
                        f"Skipping invalid objective config for '{obj_name}': not a dictionary."
                    )
                    continue
                try:
                    # Ensure ObjectiveType is valid
                    obj_type_str = obj_config.get("type", "primary")
                    obj_type = ObjectiveType(obj_type_str)

                    self.add_objective(
                        Objective(
                            name=obj_name,
                            description=obj_config.get(
                                "description", f"Objective: {obj_name}"
                            ),
                            constraints=obj_config.get("constraints", {}),
                            priority=int(obj_config.get("priority", 1)),  # Ensure int
                            dependencies=obj_config.get("dependencies", []),
                            conflicts_with=obj_config.get("conflicts_with", []),
                            objective_type=obj_type,
                            weight=float(obj_config.get("weight", 1.0)),  # Ensure float
                            target_value=float(obj_config["target"])
                            if obj_config.get("target") is not None
                            else None,  # Ensure float if present
                            maximize=bool(
                                obj_config.get("maximize", True)
                            ),  # Ensure bool
                            metadata=obj_config.get("metadata", {}),
                        )
                    )
                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Failed to load objective '{obj_name}' from spec: {e}. Config: {obj_config}"
                    )

        # Build relationship graphs AFTER all objectives are added
        self._build_relationship_graphs()

        # Check initial consistency
        self.check_consistency()

    def _create_default_objectives(self):
        """Create default objectives if none specified"""

        defaults = [
            Objective(
                name="prediction_accuracy",
                description="Maximize prediction accuracy",
                constraints={"min": 0.0, "max": 1.0},
                priority=0,  # Changed to 0 as per example
                objective_type=ObjectiveType.PRIMARY,
                weight=1.0,
                target_value=0.95,
                maximize=True,
                metadata={"resources": ["compute", "data"]},
            ),
            Objective(
                name="uncertainty_calibration",
                description="Ensure uncertainty estimates are well-calibrated",
                constraints={"min": 0.0, "max": 1.0},
                priority=1,  # Changed to 1
                objective_type=ObjectiveType.PRIMARY,  # Changed to PRIMARY
                weight=0.9,  # Changed to 0.9
                target_value=0.9,
                maximize=True,
                dependencies=["prediction_accuracy"],
                metadata={"resources": ["compute", "data"]},
            ),
            Objective(
                name="safety",
                description="Maintain safety constraints",
                constraints={"min": 1.0, "max": 1.0},
                priority=0,
                objective_type=ObjectiveType.PRIMARY,
                weight=1.0,
                target_value=1.0,
                maximize=True,  # Should likely maximize adherence (1.0 = safe)
                metadata={"resources": ["compute"]},
            ),
            Objective(
                name="efficiency",
                description="Optimize computational efficiency",
                constraints={"min": 0.0, "max": 1.0},
                priority=1,
                objective_type=ObjectiveType.SECONDARY,
                weight=0.7,
                target_value=0.8,
                maximize=True,
                conflicts_with=["prediction_accuracy"],  # Speed vs accuracy
                metadata={"resources": ["compute", "memory"]},
            ),
        ]

        for obj in defaults:
            self.add_objective(obj)

    def add_objective(self, objective: Objective, parent: Optional[str] = None) -> bool:
        """
        Add objective to hierarchy

        Args:
            objective: Objective to add
            parent: Optional parent objective name

        Returns:
            True if added successfully
        """

        with self.lock:
            if not isinstance(objective, Objective):
                logger.error("Attempted to add non-Objective object to hierarchy.")
                return False

            if objective.name in self.objectives:
                logger.warning("Objective %s already exists", objective.name)
                return False

            # Set parent if specified
            if parent:
                if parent not in self.objectives:
                    logger.error("Parent objective %s does not exist", parent)
                    # Optionally, allow adding even if parent doesn't exist yet?
                    # For now, require parent existence.
                    return False
                objective.parent = parent
                # Ensure children list exists before appending
                if not hasattr(self.objectives[parent], "children"):
                    self.objectives[parent].children = []
                # Avoid duplicates
                if objective.name not in self.objectives[parent].children:
                    self.objectives[parent].children.append(objective.name)

            # Add objective
            self.objectives[objective.name] = objective

            # --- START FIX: Uncommented graph update logic ---
            # Update dependency graph
            for dep in objective.dependencies:
                self.dependency_graph[objective.name].add(dep)

            # Update conflict graph
            for conflict in objective.conflicts_with:
                self.conflict_graph[objective.name].add(conflict)
                self.conflict_graph[conflict].add(objective.name)  # Bidirectional

            # Update parent-child graph
            if parent:
                self.parent_child_graph[parent].add(objective.name)
            # --- END FIX ---

            # Invalidate computed properties
            self._invalidate_computed_properties()

            self.stats["objectives_added"] += 1

            logger.debug(
                "Added objective: %s (priority=%d, type=%s)",
                objective.name,
                objective.priority,
                objective.objective_type.value,
            )

            return True

    def _invalidate_computed_properties(self):
        """Invalidate caches that depend on the hierarchy structure."""
        self.priority_order = []
        self.conflict_matrix = None
        self.dependency_closure = {}
        logger.debug(
            "Invalidated computed hierarchy properties (priority_order, conflict_matrix, etc.)"
        )

    def get_dependencies(self, objective: str) -> Set[str]:
        """
        Get direct dependencies of an objective

        Args:
            objective: Objective name

        Returns:
            Set of dependency names
        """

        with self.lock:
            # Ensure graphs are built if they haven't been
            if not self.dependency_graph and self.objectives:
                self._build_relationship_graphs()

            if objective not in self.objectives:
                logger.warning(
                    "Objective %s not found when getting dependencies", objective
                )
                return set()

            # Return direct dependencies from the graph
            return self.dependency_graph.get(objective, set()).copy()

    def get_transitive_dependencies(self, objective: str) -> Set[str]:
        """
        Get all transitive dependencies (closure)

        Args:
            objective: Objective name

        Returns:
            Set of all dependencies (direct and indirect)
        """

        with self.lock:
            # Ensure graphs are built
            if not self.dependency_graph and self.objectives:
                self._build_relationship_graphs()

            # Use cached closure if available and valid
            if objective in self.dependency_closure:
                return self.dependency_closure[objective].copy()

            if objective not in self.objectives:
                logger.warning(
                    f"Objective '{objective}' not found for transitive dependency calculation."
                )
                return set()

            # Compute closure using DFS
            closure = set()
            stack = list(
                self.dependency_graph.get(objective, set())
            )  # Start with direct dependencies
            visited_in_walk = {objective}  # Prevent cycles within this specific walk

            while stack:
                current_dep = stack.pop()
                if current_dep in closure or current_dep in visited_in_walk:
                    continue  # Already processed or cycle detected for this walk

                # Check if dependency exists before adding
                if current_dep in self.objectives:
                    closure.add(current_dep)
                    visited_in_walk.add(current_dep)
                    # Add its dependencies to the stack
                    direct_deps_of_current = self.dependency_graph.get(
                        current_dep, set()
                    )
                    stack.extend(
                        d for d in direct_deps_of_current if d not in visited_in_walk
                    )
                else:
                    logger.warning(
                        f"Dependency '{current_dep}' for objective '{objective}' not found in hierarchy."
                    )

            # Cache result
            self.dependency_closure[objective] = closure

            return closure.copy()

    def check_consistency(self) -> Dict[str, Any]:
        """
        Check hierarchy consistency

        Checks for:
        - Circular dependencies
        - Conflicting constraints
        - Invalid relationships (e.g., depends on non-existent objective)
        - Priority inconsistencies

        Returns:
            Consistency check results
        """

        with self.lock:
            # Ensure graphs are up-to-date first
            self._build_relationship_graphs()

            issues = []

            # Check 1: Circular dependencies
            circular = self._detect_circular_dependencies()
            if circular:
                issues.append(
                    {
                        "type": "circular_dependency",
                        "severity": "critical",
                        "cycles": circular,
                    }
                )

            # Check 2: Invalid dependencies/conflicts (referencing non-existent objectives)
            invalid_deps, invalid_conflicts = self._find_invalid_references()
            if invalid_deps:
                issues.append(
                    {
                        "type": "invalid_dependency",
                        "severity": "high",
                        "invalid": invalid_deps,
                    }
                )
            if invalid_conflicts:
                issues.append(
                    {
                        "type": "invalid_conflict_reference",
                        "severity": "medium",
                        "invalid": invalid_conflicts,
                    }
                )

            # Check 3: Conflicting constraints between related objectives
            constraint_conflicts = self._find_constraint_conflicts()
            if constraint_conflicts:
                issues.append(
                    {
                        "type": "constraint_conflict",
                        "severity": "medium",
                        "conflicts": constraint_conflicts,
                    }
                )

            # Check 4: Priority consistency (dependencies should ideally have >= priority)
            priority_issues = self._check_priority_consistency()
            if priority_issues:
                issues.append(
                    {
                        "type": "priority_inconsistency",
                        "severity": "low",
                        "issues": priority_issues,
                    }
                )

            consistent = len(issues) == 0

            result = {
                "consistent": consistent,
                "issues": issues,
                "total_objectives": len(self.objectives),
                "timestamp": time.time(),
            }

            if not consistent:
                logger.warning(
                    "Hierarchy inconsistency detected: %d issue categories found.",
                    len(issues),
                )
                for issue in issues:
                    logger.warning(
                        f"  - Type: {issue['type']}, Severity: {issue['severity']}"
                    )
            else:
                logger.info("Hierarchy consistency check passed.")

            return result

    def find_conflicts(self, obj_a: str, obj_b: str) -> Optional[Dict[str, Any]]:
        """
        Find conflicts between two objectives

        Args:
            obj_a: First objective name
            obj_b: Second objective name

        Returns:
            Conflict information dict if exists, None otherwise.
            Dict contains 'type' (ConflictType value), 'severity' (str), 'description' (str).
        """

        with self.lock:
            # Ensure graphs are built
            if not self.conflict_graph and self.objectives:
                self._build_relationship_graphs()

            obj_a_data = self.objectives.get(obj_a)
            obj_b_data = self.objectives.get(obj_b)

            if not obj_a_data or not obj_b_data:
                logger.debug(
                    f"One or both objectives not found for conflict check: {obj_a}, {obj_b}"
                )
                return None

            # Direct conflict (check graph)
            if obj_b in self.conflict_graph.get(obj_a, set()):
                # Try to get severity from metadata if defined in spec, default high
                severity = "high"
                # Check obj_a's spec
                conflict_meta_a = next(
                    (
                        item.get("severity", severity)
                        for item in obj_a_data.metadata.get("conflicts_meta", [])
                        if item.get("objective") == obj_b
                    ),
                    severity,
                )
                # Check obj_b's spec (could be defined either way)
                conflict_meta_b = next(
                    (
                        item.get("severity", severity)
                        for item in obj_b_data.metadata.get("conflicts_meta", [])
                        if item.get("objective") == obj_a
                    ),
                    severity,
                )
                # Use the more severe if defined differently
                severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                final_severity = max(
                    conflict_meta_a,
                    conflict_meta_b,
                    key=lambda s: severity_map.get(s, 0),
                )

                return {
                    "type": ConflictType.DIRECT.value,
                    "severity": final_severity,
                    "description": f"{obj_a} explicitly conflicts with {obj_b} in design spec",
                }

            # Indirect conflict through constraints
            constraint_conflict_desc = self._check_constraint_conflict(
                obj_a_data, obj_b_data
            )
            if constraint_conflict_desc:
                return {
                    "type": ConflictType.CONSTRAINT.value,
                    "severity": "medium",  # Default severity for constraint conflicts
                    "description": constraint_conflict_desc,
                }

            # Indirect conflict through shared dependencies with limited resources
            deps_a = self.get_transitive_dependencies(obj_a)
            deps_b = self.get_transitive_dependencies(obj_b)
            shared_deps = deps_a & deps_b

            if shared_deps:
                # Check if shared dependencies create resource contention
                for dep_name in shared_deps:
                    dep_obj = self.objectives.get(dep_name)
                    # Check metadata safely
                    if dep_obj and dep_obj.metadata.get("limited_resource", False):
                        return {
                            "type": ConflictType.INDIRECT.value,
                            "severity": "low",  # Default severity for resource contention
                            "description": f"Potential resource contention through shared dependency: {dep_name}",
                            "shared_dependencies": list(shared_deps),
                            "resource": dep_obj.metadata.get(
                                "resource_name", dep_name
                            ),  # Add resource name if available
                        }

            # No conflict found based on current checks
            return None

    def get_priority_order(self) -> List[str]:
        """
        Get objectives in priority order (critical first, then by priority number, then weight)

        Returns:
            List of objective names ordered by priority.
        """

        with self.lock:
            # Recompute if cache is invalidated
            if not self.priority_order and self.objectives:
                # Sort by priority (0=highest) then by weight (desc), then name (alpha)
                sorted_objs = sorted(
                    self.objectives.values(),
                    key=lambda obj: (obj.priority, -obj.weight, obj.name),
                )
                self.priority_order = [obj.name for obj in sorted_objs]
                logger.debug("Recomputed priority order.")

            return self.priority_order.copy()

    def get_hierarchy_structure(self) -> Dict[str, Any]:
        """
        Get complete hierarchy structure as a nested dictionary.

        Returns:
            Dictionary representing the hierarchy (e.g., {'primary': [...], 'secondary': [...]}).
            Each objective includes dependencies and conflicts.
        """

        with self.lock:
            # Ensure graphs are built
            if (
                not self.dependency_graph or not self.conflict_graph
            ) and self.objectives:
                self._build_relationship_graphs()

            structure = {"primary": [], "secondary": [], "derived": []}

            # Create a dict mapping names to obj_dicts first
            obj_dict_map = {}
            for obj_name, obj in self.objectives.items():
                obj_dict = obj.to_dict()
                # Use graph data for dependencies/conflicts for consistency
                obj_dict["dependencies"] = sorted(
                    list(self.dependency_graph.get(obj_name, set()))
                )
                obj_dict["conflicts"] = sorted(
                    list(self.conflict_graph.get(obj_name, set()))
                )
                obj_dict["children"] = sorted(
                    list(self.parent_child_graph.get(obj_name, set()))
                )  # Add children from graph
                obj_dict_map[obj_name] = obj_dict

            # Populate structure by type
            for obj_name, obj_dict in obj_dict_map.items():
                obj_type = self.objectives[obj_name].objective_type
                if obj_type == ObjectiveType.PRIMARY:
                    structure["primary"].append(obj_dict)
                elif obj_type == ObjectiveType.SECONDARY:
                    structure["secondary"].append(obj_dict)
                else:  # DERIVED
                    structure["derived"].append(obj_dict)

            # Sort lists within structure for consistency (optional)
            for key in structure:
                structure[key].sort(
                    key=lambda x: (
                        x.get("priority", 99),
                        -x.get("weight", 0),
                        x["name"],
                    )
                )

            return structure

    def is_derived_objective(self, objective: str) -> bool:
        """Check if objective is derived"""

        with self.lock:
            obj = self.objectives.get(objective)
            if not obj:
                return False
            return obj.objective_type == ObjectiveType.DERIVED

    def get_parents(self, objective: str) -> List[str]:
        """Get parent objectives (both explicit and via dependency graph)"""

        with self.lock:
            # Ensure graphs built
            if not self.parent_child_graph and self.objectives:
                self._build_relationship_graphs()

            parents = set()
            obj = self.objectives.get(objective)
            if not obj:
                logger.warning(
                    f"Objective '{objective}' not found when getting parents."
                )
                return []

            # Explicit parent
            if obj.parent:
                # Verify parent still exists
                if obj.parent in self.objectives:
                    parents.add(obj.parent)
                else:
                    logger.warning(
                        f"Explicit parent '{obj.parent}' for objective '{objective}' no longer exists."
                    )

            # Implicit parents (objectives that have this one as a child in the graph)
            for potential_parent, children in self.parent_child_graph.items():
                if objective in children:
                    # Verify parent exists
                    if potential_parent in self.objectives:
                        parents.add(potential_parent)
                    else:
                        logger.warning(
                            f"Parent '{potential_parent}' from graph for objective '{objective}' no longer exists."
                        )

            return sorted(list(parents))

    def get_children(self, objective: str) -> List[str]:
        """Get child objectives (both explicit and via parent-child graph)"""

        with self.lock:
            # Ensure graphs built
            if not self.parent_child_graph and self.objectives:
                self._build_relationship_graphs()

            children = set()
            obj = self.objectives.get(objective)
            if not obj:
                logger.warning(
                    f"Objective '{objective}' not found when getting children."
                )
                return []

            # Explicit children
            if hasattr(obj, "children") and isinstance(obj.children, list):
                # Verify children still exist
                valid_explicit_children = {
                    c for c in obj.children if c in self.objectives
                }
                if len(valid_explicit_children) != len(obj.children):
                    logger.warning(
                        f"Some explicit children for '{objective}' no longer exist."
                    )
                children.update(valid_explicit_children)

            # Children from parent-child graph
            graph_children = self.parent_child_graph.get(objective, set())
            # Verify children exist
            valid_graph_children = {c for c in graph_children if c in self.objectives}
            if len(valid_graph_children) != len(graph_children):
                logger.warning(
                    f"Some children from graph for '{objective}' no longer exist."
                )
            children.update(valid_graph_children)

            return sorted(list(children))

    def update_objective_value(self, objective: str, value: float):
        """Update current value of objective"""

        with self.lock:
            obj = self.objectives.get(objective)
            if obj:
                try:
                    obj.current_value = float(value)
                    logger.debug(
                        "Updated %s current value to %.3f", objective, obj.current_value
                    )
                except (TypeError, ValueError):
                    logger.error(
                        f"Invalid value type '{type(value)}' provided for objective {objective}. Value not updated."
                    )
            else:
                logger.warning(
                    f"Attempted to update value for non-existent objective: {objective}"
                )

    def get_unsatisfied_objectives(self) -> List[str]:
        """Get list of unsatisfied objectives (based on target value and constraints)"""

        with self.lock:
            unsatisfied = []
            for name, obj in self.objectives.items():
                if not obj.is_satisfied():
                    unsatisfied.append(name)
            # Sort by priority?
            unsatisfied.sort(
                key=lambda name: (
                    self.objectives[name].priority,
                    -self.objectives[name].weight,
                    name,
                )
            )
            return unsatisfied

    def get_violated_objectives(self) -> List[str]:
        """Get objectives currently violating their constraints"""

        with self.lock:
            violated = []
            for name, obj in self.objectives.items():
                if obj.violates_constraints():
                    violated.append(name)
            # Sort by priority?
            violated.sort(
                key=lambda name: (
                    self.objectives[name].priority,
                    -self.objectives[name].weight,
                    name,
                )
            )
            return violated

    def _build_relationship_graphs(self):
        """Build/rebuild dependency, conflict, and parent-child graphs from current objectives."""
        with self.lock:
            logger.debug("Building relationship graphs...")
            # Clear existing graphs
            self.dependency_graph.clear()
            self.conflict_graph.clear()
            self.parent_child_graph.clear()

            all_objective_names = set(self.objectives.keys())

            for obj_name, obj in self.objectives.items():
                # Dependencies
                valid_deps = set()
                for dep in getattr(obj, "dependencies", []):
                    if dep in all_objective_names:
                        valid_deps.add(dep)
                    else:
                        logger.warning(
                            f"Objective '{obj_name}' lists non-existent dependency '{dep}'."
                        )
                if valid_deps:
                    self.dependency_graph[obj_name] = valid_deps

                # Conflicts (bidirectional)
                valid_conflicts = set()
                for conflict in getattr(obj, "conflicts_with", []):
                    if conflict in all_objective_names:
                        valid_conflicts.add(conflict)
                        # Add reverse link immediately
                        self.conflict_graph[conflict].add(obj_name)
                    else:
                        logger.warning(
                            f"Objective '{obj_name}' lists non-existent conflict '{conflict}'."
                        )
                if valid_conflicts:
                    self.conflict_graph[obj_name].update(valid_conflicts)

                # Parent-child (from explicit parent attribute)
                parent_name = getattr(obj, "parent", None)
                if parent_name:
                    if parent_name in all_objective_names:
                        self.parent_child_graph[parent_name].add(obj_name)
                    else:
                        logger.warning(
                            f"Objective '{obj_name}' lists non-existent parent '{parent_name}'."
                        )

                # Parent-child (from explicit children attribute - less common)
                children_list = getattr(obj, "children", [])
                if children_list:
                    valid_children = {c for c in children_list if c in self.objectives}
                    if len(valid_children) != len(children_list):
                        logger.warning(
                            f"Objective '{obj_name}' lists some non-existent children."
                        )
                    if valid_children:
                        self.parent_child_graph[obj_name].update(valid_children)
                        # Ensure parent link is set on children if not already
                        for child_name in valid_children:
                            if not getattr(self.objectives[child_name], "parent", None):
                                self.objectives[child_name].parent = obj_name

            # Invalidate caches dependent on these graphs
            self._invalidate_computed_properties()
            logger.debug("Finished building relationship graphs.")

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies using DFS"""

        cycles = []
        path = set()  # Nodes currently in the recursion stack for this path
        visited = set()  # Nodes visited in any path

        def find_cycles_util(node):
            path.add(node)
            visited.add(node)

            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if find_cycles_util(neighbor):
                        # Cycle found downstream, propagate True up
                        return True
                elif neighbor in path:
                    # Cycle detected involving neighbor
                    # (Note: This simple DFS detects *existence* of cycles easily.
                    # Reconstructing the exact cycle path requires more state.)
                    # For now, just log the detection point.
                    logger.warning(
                        f"Circular dependency detected involving edge {node} -> {neighbor}"
                    )
                    # To capture the cycle path would require passing the path list down.
                    # For simplicity, we'll just report the nodes involved in cycles found this way.
                    # This might report multiple edges involved in the same cycle.
                    cycles.append(
                        [node, neighbor]
                    )  # Report the edge causing cycle detection
                    return True  # Indicate cycle found

            path.remove(node)
            return False  # No cycle found starting from this node in this path

        # Run DFS from each node if not already visited
        all_nodes = list(self.objectives.keys())
        for node in all_nodes:
            if node not in visited:
                find_cycles_util(node)

        # Post-process reported edges to try and group cycles (simplified)
        # This won't perfectly reconstruct all cycles but gives an idea
        grouped_cycles = []
        involved_nodes = set()
        for edge in cycles:
            involved_nodes.update(edge)
        if involved_nodes:
            grouped_cycles.append(list(involved_nodes))  # Very rough grouping

        # Return the simplified cycle representation
        return grouped_cycles

    def _find_invalid_references(
        self,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Find dependencies/conflicts referencing non-existent objectives"""

        invalid_deps = []
        invalid_conflicts = []
        all_objective_names = set(self.objectives.keys())

        for obj_name, obj in self.objectives.items():
            # Check dependencies listed in the object attribute
            for dep in getattr(obj, "dependencies", []):
                if dep not in all_objective_names:
                    invalid_deps.append((obj_name, dep))
            # Check conflicts listed in the object attribute
            for conflict in getattr(obj, "conflicts_with", []):
                if conflict not in all_objective_names:
                    invalid_conflicts.append((obj_name, conflict))

        # Also check graph consistency (might catch issues if graphs built from inconsistent state)
        for obj_name, deps in self.dependency_graph.items():
            for dep in deps:
                if dep not in all_objective_names:
                    invalid_deps.append((obj_name, dep))
        for obj_name, conflicts in self.conflict_graph.items():
            for conflict in conflicts:
                if conflict not in all_objective_names:
                    # Avoid double reporting if A conflicts B and B conflicts A listed invalidly
                    if (conflict, obj_name) not in invalid_conflicts:
                        invalid_conflicts.append((obj_name, conflict))

        # Deduplicate
        return list(set(invalid_deps)), list(set(invalid_conflicts))

    def _find_constraint_conflicts(self) -> List[Dict[str, Any]]:
        """Find objectives with conflicting constraints among related pairs"""

        conflicts = []
        checked_pairs = set()  # Avoid checking A-B and B-A separately

        # Check each pair of objectives
        all_objective_names = list(self.objectives.keys())
        for i, name_a in enumerate(all_objective_names):
            for name_b in all_objective_names[i + 1 :]:
                obj_a = self.objectives[name_a]
                obj_b = self.objectives[name_b]

                # Check if they are related (dependency or conflict) OR share resources
                related = (
                    name_b in self.dependency_graph.get(name_a, set())
                    or name_a in self.dependency_graph.get(name_b, set())
                    or name_b in self.conflict_graph.get(name_a, set())
                )
                # Could add resource check here if needed

                if related:
                    conflict_desc = self._check_constraint_conflict(obj_a, obj_b)
                    if conflict_desc:
                        conflicts.append(
                            {
                                "objectives": sorted(
                                    [name_a, name_b]
                                ),  # Sort for consistent reporting
                                "description": conflict_desc,
                                "relation": "dependency/conflict",  # Indicate why checked
                            }
                        )
                        # Add pair to checked to avoid redundant symmetric checks if any
                        checked_pairs.add(tuple(sorted((name_a, name_b))))

        return conflicts

    def _check_constraint_conflict(
        self, obj_a: Objective, obj_b: Objective
    ) -> Optional[str]:
        """Check if two objectives have conflicting constraints (e.g., non-overlapping ranges)"""

        constraints_a = getattr(obj_a, "constraints", {})
        constraints_b = getattr(obj_b, "constraints", {})

        # Example check: non-overlapping min/max ranges
        # Assumes constraints apply to the same underlying variable/metric implicitly
        # A more robust check would need semantic understanding of constraints.
        try:
            # Provide defaults that ensure no conflict if key is missing or value is None
            a_min = (
                float(constraints_a.get("min", float("-inf")))
                if constraints_a.get("min") is not None
                else float("-inf")
            )
            a_max = (
                float(constraints_a.get("max", float("inf")))
                if constraints_a.get("max") is not None
                else float("inf")
            )
            b_min = (
                float(constraints_b.get("min", float("-inf")))
                if constraints_b.get("min") is not None
                else float("-inf")
            )
            b_max = (
                float(constraints_b.get("max", float("inf")))
                if constraints_b.get("max") is not None
                else float("inf")
            )

            # Check if ranges are valid (min <= max)
            if a_min > a_max or b_min > b_max:
                logger.debug(f"Invalid range defined for {obj_a.name} or {obj_b.name}")
                return None  # Cannot determine conflict with invalid range

            # Check if ranges have no overlap
            # No overlap if max of one is strictly less than min of the other
            if a_max < b_min or b_max < a_min:
                return f"Non-overlapping constraint ranges: {obj_a.name} [{a_min}, {a_max}] vs {obj_b.name} [{b_min}, {b_max}]"

        except (ValueError, TypeError):
            logger.warning(
                f"Could not compare constraints for {obj_a.name} and {obj_b.name} due to invalid numeric values."
            )
            return None  # Cannot determine conflict

        # Add more sophisticated constraint conflict checks here if needed
        # e.g., conflicts between 'equals' and 'not_equals', etc.

        return None

    def _check_priority_consistency(self) -> List[Dict[str, Any]]:
        """Check for priority inconsistencies (e.g., dependency having lower priority)"""

        issues = []

        # Dependencies should ideally have priority <= dependent's priority
        for obj_name, deps in self.dependency_graph.items():
            if obj_name in self.objectives:
                obj = self.objectives[obj_name]
                for dep_name in deps:
                    if dep_name in self.objectives:
                        dep_obj = self.objectives[dep_name]
                        # Compare priorities (lower number means higher priority)
                        if dep_obj.priority > obj.priority:
                            issues.append(
                                {
                                    "dependent": obj_name,
                                    "dependency": dep_name,
                                    "issue": f"Dependency ({dep_name}, P{dep_obj.priority}) has lower priority than dependent ({obj_name}, P{obj.priority})",
                                }
                            )
        return issues

    def compute_conflict_matrix(self) -> np.ndarray:  # Or list of lists
        """
        Compute conflict matrix for all objectives based on find_conflicts results.

        Returns:
            Matrix (numpy array or list of lists) where M[i,j] = conflict strength [0, 1]
            between objectives i and j.
        """
        _np = self._np  # Use internal numpy/FakeNumpy alias

        with self.lock:
            # Recompute if cache is invalidated or doesn't exist
            if self.conflict_matrix is None and self.objectives:
                n = len(self.objectives)
                obj_names = list(self.objectives.keys())  # Keep consistent order
                obj_name_to_index = {name: i for i, name in enumerate(obj_names)}

                matrix = _np.zeros((n, n))  # Creates list of lists if numpy failed

                for i, name_a in enumerate(obj_names):
                    for j, name_b in enumerate(obj_names):
                        if i >= j:
                            continue  # Only compute upper triangle

                        conflict = self.find_conflicts(name_a, name_b)
                        if conflict:
                            # Map severity string to numeric strength
                            severity_str = conflict.get("severity", "medium")
                            severity_map = {
                                "negligible": 0.1,
                                "low": 0.3,
                                "medium": 0.6,
                                "high": 0.9,
                                "critical": 1.0,
                            }
                            strength = severity_map.get(
                                severity_str, 0.5
                            )  # Default to medium strength

                            matrix[i][j] = strength
                            matrix[j][i] = strength  # Symmetric

                self.conflict_matrix = matrix
                logger.debug("Recomputed conflict matrix.")

            # Return a copy to prevent external modification if it's a list
            if self.conflict_matrix is None:
                return _np.zeros((0, 0))  # Return empty if no objectives
            elif isinstance(self.conflict_matrix, list):
                return [
                    row[:] for row in self.conflict_matrix
                ]  # Deep copy for list of lists
            else:  # Assume numpy array
                return self.conflict_matrix.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchy statistics"""

        with self.lock:
            # Ensure graphs built for accurate counts
            if (
                not self.dependency_graph or not self.conflict_graph
            ) and self.objectives:
                self._build_relationship_graphs()

            # Count objectives by type safely
            primary_count = 0
            secondary_count = 0
            derived_count = 0
            for obj in self.objectives.values():
                if obj.objective_type == ObjectiveType.PRIMARY:
                    primary_count += 1
                elif obj.objective_type == ObjectiveType.SECONDARY:
                    secondary_count += 1
                elif obj.objective_type == ObjectiveType.DERIVED:
                    derived_count += 1

            # Calculate total dependencies/conflicts from graphs
            total_deps = sum(len(deps) for deps in self.dependency_graph.values())
            # Conflicts are stored bidirectionally, divide by 2
            total_conflicts = (
                sum(len(conflicts) for conflicts in self.conflict_graph.values()) // 2
            )

            return {
                "total_objectives": len(self.objectives),
                "primary_objectives": primary_count,
                "secondary_objectives": secondary_count,
                "derived_objectives": derived_count,
                "total_dependencies": total_deps,
                "total_conflicts": total_conflicts,
                "unsatisfied_count": len(
                    self.get_unsatisfied_objectives()
                ),  # Call method for consistency
                "violated_count": len(
                    self.get_violated_objectives()
                ),  # Call method for consistency
                "stats_internal": dict(self.stats),  # Renamed nested stats
            }
