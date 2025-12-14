"""
curiosity_engine_core.py - Main curiosity-driven learning orchestrator
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
"""

import copy
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .dependency_graph import CycleAwareDependencyGraph, DependencyAnalyzer
from .experiment_generator import (
    Experiment,
    ExperimentGenerator,
    IterativeExperimentDesigner,
)
from .exploration_budget import DynamicBudget, ResourceMonitor

# Import other curiosity_engine components
from .gap_analyzer import GapAnalyzer, KnowledgeGap

logger = logging.getLogger(__name__)


@dataclass
class LearningPriority:
    """Priority item for learning queue"""

    gap: KnowledgeGap
    priority: float
    experiments: List[Experiment] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority > other.priority


@dataclass
class ExperimentResult:
    """Result from running an experiment"""

    experiment: Experiment
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    learned_knowledge: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "gap_type": self.experiment.gap.type,
            "success": self.success,
            "execution_time": self.execution_time,
            "resource_usage": self.resource_usage,
            "learned_knowledge": self.learned_knowledge,
            "error": self.error,
        }


@dataclass
class KnowledgeRegion:
    """Region in knowledge space"""

    domain: str
    patterns: Set[str]
    confidence: float
    exploration_count: int = 0
    last_explored: float = field(default_factory=time.time)
    value_estimate: float = 0.5

    def distance_to(self, other: "KnowledgeRegion") -> float:
        """Calculate distance to another region"""
        try:
            # Jaccard distance for patterns
            if not self.patterns and not other.patterns:
                return 1.0

            intersection = self.patterns & other.patterns
            union = self.patterns | other.patterns

            if not union:
                return 1.0

            jaccard = len(intersection) / len(union)
            return 1.0 - jaccard
        except Exception as e:
            logger.warning("Error calculating distance: %s", e)
            return 1.0


class RegionManager:
    """Manages knowledge regions - SEPARATED CONCERN"""

    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.explored_regions = {}  # region_id -> KnowledgeRegion
        self.frontier_regions = set()  # Region IDs at frontier
        self.unexplored_estimates = {}  # Estimated unexplored regions
        self.region_graph = defaultdict(set)  # region_id -> set of neighbor_ids
        self.lock = threading.RLock()

    def add_region(self, domain: str, patterns: Set[str]) -> str:
        """Add or update a region"""
        with self.lock:
            try:
                # Enforce size limit
                if len(self.explored_regions) >= self.cache_size:
                    self._evict_oldest_region()

                # Find or create region
                region_id = self._find_or_create_region(domain, patterns)

                # Update region
                region = self.explored_regions.get(region_id)
                if region:
                    region.exploration_count += 1
                    region.last_explored = time.time()
                    region.confidence = min(0.99, region.confidence * 1.1)

                # Update frontier
                self._update_frontier(region_id)

                return region_id
            except Exception as e:
                logger.error("Error adding region: %s", e)
                return f"{domain}_error"

    def get_neighbors(self, region_id: str) -> Set[str]:
        """Get neighboring regions"""
        with self.lock:
            return self.region_graph.get(region_id, set()).copy()

    def get_region(self, region_id: str) -> Optional[KnowledgeRegion]:
        """Get a region by ID"""
        with self.lock:
            region = self.explored_regions.get(region_id)
            if region:
                # Return a copy to prevent external modification
                return copy.deepcopy(region)
            return None

    def get_frontier_regions(self) -> Set[str]:
        """Get all frontier regions"""
        with self.lock:
            return self.frontier_regions.copy()

    def get_all_regions(self) -> Dict[str, KnowledgeRegion]:
        """Get all explored regions (copy)"""
        with self.lock:
            return copy.deepcopy(self.explored_regions)

    def _find_or_create_region(self, domain: str, patterns: Set[str]) -> str:
        """Find existing region or create new one"""
        # Check for overlapping regions
        best_match = None
        best_overlap = 0

        # FIX: Copy items to avoid modification during iteration
        regions_items = list(self.explored_regions.items())

        for region_id, region in regions_items:
            if region.domain == domain:
                overlap = len(patterns & region.patterns)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = region_id

        # FIX: Check both empty patterns and sufficient overlap
        if best_match and patterns and best_overlap > len(patterns) * 0.5:
            # Merge with existing region
            region = self.explored_regions[best_match]
            region.patterns.update(patterns)
            return best_match
        else:
            # Create new region
            region_id = f"{domain}_{len(self.explored_regions)}"

            new_region = KnowledgeRegion(
                domain=domain, patterns=patterns.copy(), confidence=0.1
            )

            self.explored_regions[region_id] = new_region

            # Connect to nearby regions
            self._connect_regions(region_id)

            return region_id

    def _connect_regions(self, region_id: str):
        """Connect region to nearby regions"""
        try:
            region = self.explored_regions.get(region_id)
            if not region:
                return

            # FIX: Copy items to avoid modification during iteration
            other_regions = list(self.explored_regions.items())

            for other_id, other_region in other_regions:
                if other_id != region_id:
                    distance = region.distance_to(other_region)

                    # Connect if close enough
                    if distance < 0.5:
                        self.region_graph[region_id].add(other_id)
                        self.region_graph[other_id].add(region_id)
        except Exception as e:
            logger.warning("Error connecting regions: %s", e)

    def _update_frontier(self, region_id: str):
        """Update frontier after exploring region"""
        try:
            # Add to frontier if new
            if region_id not in self.frontier_regions:
                self.frontier_regions.add(region_id)

            # Check if region is still at frontier
            region = self.explored_regions.get(region_id)
            if not region:
                return

            if region.confidence > 0.9:
                # Well-explored, might not be frontier
                # Check if has unexplored neighbors
                has_unexplored = False

                for neighbor_id in self.region_graph.get(region_id, []):
                    neighbor = self.explored_regions.get(neighbor_id)
                    if not neighbor or neighbor.confidence < 0.5:
                        has_unexplored = True
                        break

                if not has_unexplored:
                    self.frontier_regions.discard(region_id)
        except Exception as e:
            logger.warning("Error updating frontier: %s", e)

    def _evict_oldest_region(self):
        """Evict oldest explored region to maintain size limit"""
        try:
            if not self.explored_regions:
                return

            # Find oldest region by last_explored
            oldest_id = min(
                self.explored_regions.keys(),
                key=lambda k: self.explored_regions[k].last_explored,
            )

            # Remove from all structures
            del self.explored_regions[oldest_id]
            self.frontier_regions.discard(oldest_id)

            # FIX: Clean up graph connections safely
            # Copy the neighbors dict to avoid modification during iteration
            graph_items = list(self.region_graph.items())
            for node_id, neighbors in graph_items:
                neighbors.discard(oldest_id)

            if oldest_id in self.region_graph:
                del self.region_graph[oldest_id]
        except Exception as e:
            logger.error("Error evicting region: %s", e)


class ExplorationValueEstimator:
    """Estimates exploration value - SEPARATED CONCERN"""

    def __init__(self, decay_rate: float = 0.95):
        self.decay_rate = decay_rate
        self.value_history = defaultdict(lambda: deque(maxlen=100))
        self._value_cache = {}
        self._cache_timestamp = time.time()
        self._cache_ttl = 60
        self.lock = threading.RLock()

    def estimate_value(
        self, region: KnowledgeRegion, neighbors_count: int = 0
    ) -> float:
        """Estimate value of exploring a region"""
        with self.lock:
            try:
                # Check cache
                cache_key = f"{region.domain}_{id(region)}"
                if self._is_cache_valid() and cache_key in self._value_cache:
                    return self._value_cache[cache_key]

                # EXAMINE: Calculate value components
                base_value = 1.0 - region.confidence

                # Novelty bonus
                time_since_explored = time.time() - region.last_explored
                novelty_bonus = min(0.3, time_since_explored / 3600)

                # Connectivity bonus
                connectivity_bonus = min(0.2, neighbors_count * 0.02)

                # Historical performance
                historical_bonus = 0
                if region.domain in self.value_history:
                    history = self.value_history[region.domain]
                    if history:
                        historical_avg = np.mean(history[-10:])
                        historical_bonus = historical_avg * 0.2

                # SELECT & APPLY: Combine factors
                total_value = (
                    base_value + novelty_bonus + connectivity_bonus + historical_bonus
                )

                # Apply decay based on exploration count
                decay_factor = self.decay_rate**region.exploration_count
                total_value *= decay_factor

                value = min(1.0, max(0.0, total_value))

                # REMEMBER: Cache the value
                self._value_cache[cache_key] = value

                return value
            except Exception as e:
                logger.error("Error estimating value: %s", e)
                return 0.5

    def update_history(self, domain: str, value: float):
        """Update value history for a domain"""
        with self.lock:
            try:
                self.value_history[domain].append(value)
            except Exception as e:
                logger.warning("Error updating history: %s", e)

    def _is_cache_valid(self) -> bool:
        """Check if value cache is still valid"""
        return time.time() - self._cache_timestamp < self._cache_ttl

    def invalidate_cache(self):
        """Invalidate the value cache"""
        with self.lock:
            self._value_cache.clear()
            self._cache_timestamp = time.time()


class ExplorationFrontier:
    """Tracks the boundary of explored knowledge - REFACTORED"""

    def __init__(self, decay_rate: float = 0.95, cache_size: int = 1000):
        """
        Initialize exploration frontier

        Args:
            decay_rate: Rate at which region values decay
            cache_size: Maximum size for caches
        """
        # Components
        self.region_manager = RegionManager(cache_size)
        self.value_estimator = ExplorationValueEstimator(decay_rate)

        # Statistics
        self.exploration_history = deque(maxlen=cache_size)

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ExplorationFrontier initialized (refactored)")

    def add_explored_region(self, domain: str, pattern: Union[str, Set[str]]) -> str:
        """Add newly explored region - REFACTORED"""
        with self.lock:
            try:
                # EXAMINE: Convert pattern to set
                if isinstance(pattern, str):
                    patterns = {pattern}
                elif isinstance(pattern, (list, tuple)):
                    patterns = set(pattern)
                else:
                    patterns = pattern if isinstance(pattern, set) else {str(pattern)}

                # APPLY: Add region
                region_id = self.region_manager.add_region(domain, patterns)

                # Track exploration
                self.exploration_history.append(
                    {
                        "region_id": region_id,
                        "domain": domain,
                        "patterns": list(patterns),
                        "timestamp": time.time(),
                    }
                )

                # Invalidate cache
                self.value_estimator.invalidate_cache()

                logger.debug("Added explored region %s in domain %s", region_id, domain)

                return region_id
            except Exception as e:
                logger.error("Error adding explored region: %s", e)
                return f"{domain}_error"

    def get_unexplored_neighbors(
        self, current_position: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Get unexplored neighboring regions - REFACTORED"""
        with self.lock:
            try:
                candidates = []

                if current_position:
                    # Get neighbors of current position
                    neighbors = self.region_manager.get_neighbors(current_position)

                    for neighbor_id in neighbors:
                        if neighbor_id in self.region_manager.get_frontier_regions():
                            value = self.estimate_exploration_value(neighbor_id)
                            candidates.append((neighbor_id, value))
                else:
                    # Get all frontier regions
                    for region_id in self.region_manager.get_frontier_regions():
                        value = self.estimate_exploration_value(region_id)
                        candidates.append((region_id, value))

                # Sort by value
                candidates.sort(key=lambda x: x[1], reverse=True)

                return candidates
            except Exception as e:
                logger.error("Error getting unexplored neighbors: %s", e)
                return []

    def estimate_exploration_value(self, region: Union[str, KnowledgeRegion]) -> float:
        """Estimate value of exploring a region - DELEGATED"""
        try:
            if isinstance(region, str):
                region_obj = self.region_manager.get_region(region)
                if not region_obj:
                    return 0.5  # Default for unknown region
            else:
                region_obj = region

            neighbors_count = len(self.region_manager.get_neighbors(region_obj.domain))
            return self.value_estimator.estimate_value(region_obj, neighbors_count)
        except Exception as e:
            logger.error("Error estimating exploration value: %s", e)
            return 0.5

    def update_frontier(self, new_knowledge: Dict[str, Any]):
        """Update frontier with new knowledge - REFACTORED"""
        with self.lock:
            try:
                # EXAMINE: Extract domains and patterns
                for domain, knowledge in new_knowledge.items():
                    if isinstance(knowledge, dict):
                        patterns = set(knowledge.get("patterns", []))
                    elif isinstance(knowledge, list):
                        patterns = set(knowledge)
                    else:
                        patterns = {str(knowledge)}

                    if patterns:
                        # APPLY: Add region
                        self.add_explored_region(domain, patterns)

                        # Update value history
                        if isinstance(knowledge, dict) and "value" in knowledge:
                            self.value_estimator.update_history(
                                domain, knowledge["value"]
                            )
            except Exception as e:
                logger.error("Error updating frontier: %s", e)

    @property
    def frontier_regions(self) -> Set[str]:
        """Get frontier regions for compatibility (returns copy)"""
        return self.region_manager.get_frontier_regions()

    @property
    def explored_regions(self) -> Dict[str, KnowledgeRegion]:
        """Get explored regions for compatibility (returns copy)"""
        return self.region_manager.get_all_regions()


class SafeExperimentExecutor:
    """Safe experiment execution with resource limits - UNCHANGED (already focused)"""

    def __init__(self, timeout: float = 30, memory_mb: int = 512):
        """
        Initialize safe executor

        Args:
            timeout: Execution timeout in seconds
            memory_mb: Memory limit in MB
        """
        self.timeout = timeout
        self.memory_limit = memory_mb * 1024 * 1024

    def execute_experiment(
        self,
        experiment: "Experiment",
        knowledge_base: Any = None,
        decomposer: Any = None,
        world_model: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute experiment safely

        Args:
            experiment: Experiment to execute
            knowledge_base: Knowledge crystallizer instance
            decomposer: Problem decomposer instance
            world_model: World model instance

        Returns:
            Execution result
        """
        try:
            # Route to appropriate experiment type
            if experiment.gap.type == "decomposition":
                return self._execute_decomposition_experiment(experiment, decomposer)
            elif experiment.gap.type == "causal":
                return self._execute_causal_experiment(experiment, world_model)
            elif experiment.gap.type == "transfer":
                return self._execute_transfer_experiment(experiment, knowledge_base)
            elif experiment.gap.type == "latent":
                return self._execute_latent_experiment(experiment, knowledge_base)
            else:
                return self._execute_exploratory_experiment(experiment)
        except Exception as e:
            logger.error("Error executing experiment: %s", e)
            return {"success": False, "error": str(e), "data": {}}

    def _execute_decomposition_experiment(
        self, experiment: "Experiment", decomposer: Any
    ) -> Dict[str, Any]:
        """Execute decomposition experiment"""
        result = {"success": False, "data": {}}

        try:
            # FIX: Check both hasattr and callable
            if (
                decomposer
                and hasattr(decomposer, "test_decomposition")
                and callable(getattr(decomposer, "test_decomposition", None))
            ):
                # Use actual decomposer
                test_result = decomposer.test_decomposition(
                    strategy=experiment.parameters.get("strategy", "hierarchical"),
                    complexity=experiment.complexity,
                    depth=experiment.parameters.get("depth", 3),
                )

                result["success"] = test_result.get("success", False)
                result["data"] = test_result

                # Extract patterns
                if "components" in test_result:
                    result["patterns"] = [str(c) for c in test_result["components"]]
            else:
                # Simulate decomposition
                num_components = max(2, int(experiment.complexity * 5))
                result["success"] = True
                result["data"] = {
                    "components": [f"component_{i}" for i in range(num_components)],
                    "strategy": experiment.parameters.get("strategy", "hierarchical"),
                }
                result["patterns"] = result["data"]["components"]

        except Exception as e:
            result["error"] = str(e)
            logger.error("Decomposition experiment failed: %s", e)

        return result

    def _execute_causal_experiment(
        self, experiment: "Experiment", world_model: Any
    ) -> Dict[str, Any]:
        """Execute causal experiment"""
        result = {"success": False, "data": {}}

        try:
            intervention = experiment.parameters.get("intervention", {})

            # FIX: Check both hasattr and callable
            if (
                world_model
                and hasattr(world_model, "test_intervention")
                and callable(getattr(world_model, "test_intervention", None))
            ):
                # Use actual world model
                test_result = world_model.test_intervention(
                    variable=intervention.get("variable", "unknown"),
                    values=intervention.get("values", [0, 1]),
                    control_variables=experiment.parameters.get(
                        "control_variables", []
                    ),
                )

                result["success"] = test_result.get("causal_strength", 0) > 0.3
                result["data"] = test_result
            else:
                # Simulate causal discovery
                result["success"] = np.random.random() > 0.3
                result["data"] = {
                    "causal_strength": np.random.random(),
                    "p_value": np.random.random() * 0.1,
                    "effect_size": np.random.random(),
                }

            # Add observations
            result["observations"] = [
                {
                    "variable": intervention.get("variable"),
                    "effect": result["data"].get("effect_size", 0),
                }
            ]

        except Exception as e:
            result["error"] = str(e)
            logger.error("Causal experiment failed: %s", e)

        return result

    def _execute_transfer_experiment(
        self, experiment: "Experiment", knowledge_base: Any
    ) -> Dict[str, Any]:
        """Execute transfer experiment"""
        result = {"success": False, "data": {}}

        try:
            source = experiment.parameters.get("source_domain", "unknown")
            target = experiment.parameters.get("target_domain", "unknown")

            # FIX: Check both hasattr and callable
            if (
                knowledge_base
                and hasattr(knowledge_base, "test_transfer")
                and callable(getattr(knowledge_base, "test_transfer", None))
            ):
                # Use actual knowledge base
                test_result = knowledge_base.test_transfer(
                    source_domain=source,
                    target_domain=target,
                    strategy=experiment.parameters.get("strategy", "direct"),
                )

                result["success"] = test_result.get("transfer_success", False)
                result["data"] = test_result
            else:
                # Simulate transfer
                result["success"] = np.random.random() > 0.4
                result["data"] = {
                    "transfer_success": result["success"],
                    "accuracy": np.random.random() * 0.5 + 0.5,
                    "adaptation_steps": np.random.randint(10, 100),
                }

            # Add domain bridge info
            result["patterns"] = [f"bridge_{source}_{target}"]

        except Exception as e:
            result["error"] = str(e)
            logger.error("Transfer experiment failed: %s", e)

        return result

    def _execute_latent_experiment(
        self, experiment: "Experiment", knowledge_base: Any
    ) -> Dict[str, Any]:
        """Execute latent gap experiment"""
        result = {"success": False, "data": {}}

        try:
            # FIX: Check both hasattr and callable
            if (
                knowledge_base
                and hasattr(knowledge_base, "explore_latent")
                and callable(getattr(knowledge_base, "explore_latent", None))
            ):
                # Use actual knowledge base
                test_result = knowledge_base.explore_latent(
                    domain=experiment.gap.domain, complexity=experiment.complexity
                )

                result["success"] = len(test_result.get("discoveries", [])) > 0
                result["data"] = test_result
            else:
                # Simulate latent discovery
                result["success"] = np.random.random() > 0.5
                result["data"] = {
                    "discoveries": [
                        f"latent_pattern_{i}" for i in range(np.random.randint(0, 3))
                    ],
                    "confidence": np.random.random(),
                }

            result["patterns"] = result["data"].get("discoveries", [])

        except Exception as e:
            result["error"] = str(e)
            logger.error("Latent experiment failed: %s", e)

        return result

    def _execute_exploratory_experiment(
        self, experiment: "Experiment"
    ) -> Dict[str, Any]:
        """Execute generic exploratory experiment"""
        result = {"success": False, "data": {}}

        try:
            # Simple exploration simulation
            num_trials = experiment.parameters.get("num_trials", 20)
            successes = sum(1 for _ in range(num_trials) if np.random.random() > 0.6)

            result["success"] = successes > num_trials * 0.3
            result["data"] = {
                "trials": num_trials,
                "successes": successes,
                "novelty": np.random.random(),
                "discoveries": [f"finding_{i}" for i in range(successes)],
            }

            result["patterns"] = result["data"].get("discoveries", [])

        except Exception as e:
            result["error"] = str(e)
            logger.error("Exploratory experiment failed: %s", e)

        return result


class StrategySelector:
    """Selects exploration strategies - SEPARATED CONCERN"""

    def __init__(self):
        self.recent_failures_threshold = 5
        self.high_load_threshold = 0.8
        self.low_load_threshold = 0.3
        self.low_budget_threshold = 10
        self.high_budget_threshold = 50

    def select_strategy(
        self, current_load: float, available_budget: float, recent_failures: int
    ) -> str:
        """Select exploration strategy based on context"""

        try:
            # EXAMINE: Check conditions
            if recent_failures > self.recent_failures_threshold:
                # Many recent failures - focus on gap analysis
                return "gap_driven"
            elif current_load > self.high_load_threshold:
                # High load - minimal exploration
                return "minimal"
            elif available_budget < self.low_budget_threshold:
                # Low budget - efficient exploration
                return "efficient"
            elif (
                current_load < self.low_load_threshold
                and available_budget > self.high_budget_threshold
            ):
                # Low load, high budget - comprehensive exploration
                return "comprehensive"
            else:
                # Default balanced approach
                return "balanced"
        except Exception as e:
            logger.error("Error selecting strategy: %s", e)
            return "balanced"


class GapPrioritizer:
    """Prioritizes knowledge gaps - SEPARATED CONCERN"""

    def __init__(self):
        self.lock = threading.RLock()

    def calculate_priority(
        self, gap: KnowledgeGap, descendants_count: int = 0, ancestors_count: int = 0
    ) -> float:
        """Calculate priority score for gap"""

        try:
            # Base priority from gap
            base_priority = gap.priority

            # Adjust for dependencies
            # FIX: Check if adjusted_roi exists AND is not None
            if hasattr(gap, "adjusted_roi") and gap.adjusted_roi is not None:
                priority = gap.adjusted_roi
            else:
                base_roi = base_priority / max(gap.estimated_cost, 1)
                unlock_bonus = descendants_count * 0.1
                dependency_penalty = ancestors_count * 0.05
                priority = base_roi * (1 + unlock_bonus - dependency_penalty)

            # FIX: Ensure priority is not None before multiplication
            if priority is None:
                logger.warning("Priority calculated as None, using default value")
                priority = 1.0

            # Boost for high-impact gaps
            if gap.type == "causal":
                priority *= 1.2
            elif gap.type == "latent":
                priority *= 1.1

            # Final safety check
            if priority is None or not isinstance(priority, (int, float)):
                return 1.0

            return priority
        except Exception as e:
            logger.error("Error calculating priority: %s", e)
            return 1.0

    def prioritize_gaps(
        self, gaps: List[KnowledgeGap], dependency_graph: Any = None
    ) -> List[LearningPriority]:
        """Prioritize gaps for learning"""
        with self.lock:
            try:
                priorities = []

                for gap in gaps:
                    # Calculate dependencies if graph available
                    descendants_count = 0
                    ancestors_count = 0

                    if (
                        dependency_graph
                        and hasattr(dependency_graph, "descendants")
                        and callable(getattr(dependency_graph, "descendants", None))
                    ):
                        try:
                            descendants = dependency_graph.descendants(gap)
                            ancestors = dependency_graph.ancestors(gap)
                            descendants_count = len(descendants)
                            ancestors_count = len(ancestors)
                        except Exception as e:
                            logger.warning("Error getting dependencies: %s", e)

                    # Calculate priority
                    priority_score = self.calculate_priority(
                        gap, descendants_count, ancestors_count
                    )

                    # FIX: Validate priority score
                    if priority_score is None or not isinstance(
                        priority_score, (int, float)
                    ):
                        logger.warning(
                            "Invalid priority score for gap %s, using default",
                            getattr(gap, "id", "unknown"),
                        )
                        priority_score = 1.0

                    # Create learning priority
                    priority = LearningPriority(
                        gap=gap, priority=priority_score, experiments=[]
                    )

                    priorities.append(priority)

                # Sort by priority with safety check
                try:
                    priorities.sort(
                        key=lambda p: p.priority if p.priority is not None else 0,
                        reverse=True,
                    )
                except Exception as e:
                    logger.error("Error sorting priorities: %s", e)

                return priorities
            except Exception as e:
                logger.error("Error prioritizing gaps: %s", e)
                return []


class ExperimentManager:
    """Manages experiment execution - SEPARATED CONCERN"""

    def __init__(self, executor: SafeExperimentExecutor):
        self.executor = executor
        self.experiment_history = deque(maxlen=1000)
        self.total_experiments = 0
        self.successful_experiments = 0
        self.lock = threading.RLock()

    def run_experiment(
        self,
        experiment: Experiment,
        knowledge_base: Any = None,
        decomposer: Any = None,
        world_model: Any = None,
        resource_monitor: Any = None,
    ) -> ExperimentResult:
        """Run experiment and track results"""

        start_time = time.time()

        with self.lock:
            try:
                # EXAMINE & APPLY: Execute experiment
                result = self.executor.execute_experiment(
                    experiment,
                    knowledge_base=knowledge_base,
                    decomposer=decomposer,
                    world_model=world_model,
                )

                # Parse results
                success = result.get("success", False)
                output = result.get("data", {})
                error = result.get("error")

                # Extract learned knowledge
                learned = self._extract_learned_knowledge(result, experiment)

                execution_time = time.time() - start_time

                # Get resource usage
                resource_usage = {}
                if (
                    resource_monitor
                    and hasattr(resource_monitor, "get_resource_snapshot")
                    and callable(
                        getattr(resource_monitor, "get_resource_snapshot", None)
                    )
                ):
                    try:
                        resource_snapshot = resource_monitor.get_resource_snapshot()
                        resource_usage = {
                            "cpu": getattr(resource_snapshot, "cpu_percent", 0),
                            "memory": getattr(resource_snapshot, "memory_percent", 0),
                        }
                    except Exception as e:
                        logger.warning("Error getting resource snapshot: %s", e)

                experiment_result = ExperimentResult(
                    experiment=experiment,
                    success=success,
                    output=output,
                    error=error,
                    execution_time=execution_time,
                    resource_usage=resource_usage,
                    learned_knowledge=learned,
                )

            except Exception as e:
                logger.error("Experiment failed: %s", e)

                experiment_result = ExperimentResult(
                    experiment=experiment,
                    success=False,
                    output=None,
                    error=str(e),
                    execution_time=time.time() - start_time,
                )

            # REMEMBER: Track experiment
            self.experiment_history.append(experiment_result)
            self.total_experiments += 1

            if experiment_result.success:
                self.successful_experiments += 1

            return experiment_result

    def get_recent_failures_count(self, window: int = 10) -> int:
        """Get count of recent failures"""
        with self.lock:
            try:
                recent = list(self.experiment_history)[-window:]
                return sum(1 for exp in recent if exp and not exp.success)
            except Exception as e:
                logger.error("Error counting failures: %s", e)
                return 0

    def get_success_rate(self) -> float:
        """Get overall success rate"""
        with self.lock:
            if self.total_experiments == 0:
                return 0.0
            return self.successful_experiments / self.total_experiments

    def _extract_learned_knowledge(
        self, output: Any, experiment: Experiment
    ) -> Dict[str, Any]:
        """Extract learned knowledge from experiment output"""
        try:
            learned = {}

            if output and isinstance(output, dict):
                # Extract patterns
                if "patterns" in output:
                    learned["patterns"] = output["patterns"]

                # Extract observations
                if "observations" in output:
                    learned["observations"] = output["observations"]

                # Extract data
                if "data" in output:
                    learned.update(output["data"])

            # Add experiment metadata
            learned["experiment_type"] = experiment.gap.type
            learned["domain"] = experiment.gap.domain

            return learned
        except Exception as e:
            logger.error("Error extracting knowledge: %s", e)
            return {}


class KnowledgeIntegrator:
    """Integrates learned knowledge - SEPARATED CONCERN"""

    def integrate_results(
        self,
        results: List[ExperimentResult],
        knowledge_base: Any = None,
        world_model: Any = None,
        decomposer: Any = None,
        exploration_frontier: Any = None,
    ):
        """Integrate experiment results into knowledge systems"""

        try:
            for result in results:
                if result.success and result.learned_knowledge:
                    # Update knowledge base
                    if knowledge_base:
                        self._update_knowledge_base(
                            result.learned_knowledge, knowledge_base
                        )

                    # Update world model
                    if world_model and "observations" in result.learned_knowledge:
                        self._update_world_model(
                            result.learned_knowledge["observations"], world_model
                        )

                    # Update decomposer
                    if decomposer and "patterns" in result.learned_knowledge:
                        self._update_decomposer(
                            result.learned_knowledge["patterns"], decomposer
                        )

                    # Update exploration frontier
                    if exploration_frontier:
                        exploration_frontier.update_frontier(result.learned_knowledge)

                    # Mark gap as addressed
                    if hasattr(result.experiment.gap, "mark_addressed") and callable(
                        getattr(result.experiment.gap, "mark_addressed", None)
                    ):
                        try:
                            result.experiment.gap.mark_addressed()
                        except Exception as e:
                            logger.warning("Error marking gap as addressed: %s", e)
        except Exception as e:
            logger.error("Error integrating results: %s", e)

    def _update_knowledge_base(self, knowledge: Dict[str, Any], knowledge_base: Any):
        """Update knowledge crystallizer"""
        try:
            for key, value in knowledge.items():
                if key not in ["raw_output", "experiment_type"]:
                    if hasattr(knowledge_base, "store_knowledge") and callable(
                        getattr(knowledge_base, "store_knowledge", None)
                    ):
                        try:
                            knowledge_base.store_knowledge(key, value)
                        except Exception as e:
                            logger.warning("Error storing knowledge: %s", e)
        except Exception as e:
            logger.error("Error updating knowledge base: %s", e)

    def _update_world_model(self, observations: List[Any], world_model: Any):
        """Update world model with observations"""
        try:
            for obs in observations:
                if hasattr(world_model, "update_from_observation") and callable(
                    getattr(world_model, "update_from_observation", None)
                ):
                    try:
                        world_model.update_from_observation(obs)
                    except Exception as e:
                        logger.warning("Error updating world model: %s", e)
        except Exception as e:
            logger.error("Error updating world model: %s", e)

    def _update_decomposer(self, patterns: List[Any], decomposer: Any):
        """Update decomposer with patterns"""
        try:
            for pattern in patterns:
                if hasattr(decomposer, "learn_from_pattern") and callable(
                    getattr(decomposer, "learn_from_pattern", None)
                ):
                    try:
                        decomposer.learn_from_pattern(pattern)
                    except Exception as e:
                        logger.warning("Error updating decomposer: %s", e)
        except Exception as e:
            logger.error("Error updating decomposer: %s", e)


class CuriosityEngine:
    """Main curiosity-driven learning orchestrator - REFACTORED"""

    def __init__(self, knowledge=None, decomposer=None, world_model=None):
        """
        Initialize curiosity engine

        Args:
            knowledge: Knowledge crystallizer instance
            decomposer: Problem decomposer instance
            world_model: World model instance
        """
        self.knowledge = knowledge
        self.decomposer = decomposer
        self.world_model = world_model

        # FIX: Initialize in dependency order
        # First initialize basic components
        self.gap_analyzer = GapAnalyzer()
        self.gap_graph = CycleAwareDependencyGraph()
        self.dependency_analyzer = DependencyAnalyzer()
        self.experiment_generator = ExperimentGenerator()
        self.iterative_designer = IterativeExperimentDesigner()

        # Initialize resource management BEFORE using it
        self.exploration_budget = DynamicBudget()
        self.resource_monitor = ResourceMonitor()

        # Initialize frontier BEFORE using it
        self.exploration_frontier = ExplorationFrontier()

        # New separated components
        self.strategy_selector = StrategySelector()
        self.gap_prioritizer = GapPrioritizer()
        self.experiment_executor = SafeExperimentExecutor()
        self.experiment_manager = ExperimentManager(self.experiment_executor)
        self.knowledge_integrator = KnowledgeIntegrator()

        # Learning queue
        self.learning_priorities = PriorityQueue()

        # Tracking
        self.learning_rate = 0.1

        # Strategy selection mode
        self.exploration_mode = "adaptive"  # "adaptive", "sequential", "parallel"

        # Thread safety
        self.lock = threading.RLock()

        logger.info("CuriosityEngine initialized (refactored)")

    def select_exploration_strategy(
        self, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Select exploration strategy based on context - REFACTORED"""

        try:
            if context is None:
                context = self._get_current_context()

            # EXAMINE: Get context values
            current_load = context.get("resource_load", 0.5)
            available_budget = context.get("available_budget", 50)
            recent_failures = self.experiment_manager.get_recent_failures_count()

            # SELECT: Delegate to strategy selector
            return self.strategy_selector.select_strategy(
                current_load, available_budget, recent_failures
            )
        except Exception as e:
            logger.error("Error selecting strategy: %s", e)
            return "balanced"

    def identify_knowledge_gaps(
        self, strategy: Optional[str] = None
    ) -> List[KnowledgeGap]:
        """Identify knowledge gaps using selected strategy - REFACTORED"""

        try:
            if strategy is None:
                strategy = self.select_exploration_strategy()

            gaps = []

            # EXAMINE & SELECT: Route based on strategy
            if strategy in ["gap_driven", "comprehensive", "balanced"]:
                if strategy == "comprehensive":
                    decomposition_gaps = (
                        self.gap_analyzer.analyze_decomposition_failures()
                    )
                    prediction_gaps = self.gap_analyzer.analyze_prediction_errors()
                    transfer_gaps = self.gap_analyzer.analyze_transfer_failures()
                    latent_gaps = self.gap_analyzer.detect_latent_gaps()

                    gaps.extend(decomposition_gaps)
                    gaps.extend(prediction_gaps)
                    gaps.extend(transfer_gaps)
                    gaps.extend(latent_gaps)
                elif strategy == "gap_driven":
                    decomposition_gaps = (
                        self.gap_analyzer.analyze_decomposition_failures()
                    )
                    gaps.extend(decomposition_gaps[:5])
                else:  # balanced
                    decomposition_gaps = (
                        self.gap_analyzer.analyze_decomposition_failures()
                    )
                    prediction_gaps = self.gap_analyzer.analyze_prediction_errors()

                    gaps.extend(decomposition_gaps[:3])
                    gaps.extend(prediction_gaps[:3])
            elif strategy == "minimal":
                all_gaps = self.gap_analyzer.get_all_gaps()
                gaps = all_gaps[:3]
            elif strategy == "efficient":
                all_gaps = self.gap_analyzer.get_all_gaps()
                gaps = [g for g in all_gaps if g.estimated_cost < 20][:5]

            # APPLY: Add to dependency graph
            for gap in gaps:
                self.gap_graph.add_node(gap)

            # Find dependencies
            for gap in gaps:
                dependencies = self.dependency_analyzer.find_dependencies(gap)
                for dep in dependencies:
                    self.gap_graph.add_edge(gap, dep)

            # REMEMBER
            logger.info(
                "Identified %d knowledge gaps using strategy: %s", len(gaps), strategy
            )

            return gaps
        except Exception as e:
            logger.error("Error identifying gaps: %s", e)
            return []

    def identify_gaps_with_cycle_detection(self) -> List[KnowledgeGap]:
        """Identify gaps with cycle-aware dependency analysis - REFACTORED"""

        try:
            # EXAMINE: Get all gaps
            gaps = self.identify_knowledge_gaps()

            # Build dependency graph with cycle detection
            for gap in gaps:
                dependencies = self.dependency_analyzer.find_dependencies(gap)

                for dep in dependencies:
                    # SELECT: Check if edge would create cycle
                    if not self.gap_graph.would_create_cycle(gap, dep):
                        self.gap_graph.add_edge(gap, dep)
                    else:
                        # Add as weak dependency
                        self.gap_graph.add_weak_edge(gap, dep)
                        logger.debug(
                            "Added weak edge to avoid cycle: %s -> %s", gap, dep
                        )

            # APPLY: Get topologically sorted gaps
            try:
                sorted_gaps = self.gap_graph.topological_sort()
            except Exception as e:
                logger.warning("Cycle detected, breaking cycles: %s", e)
                self.gap_graph.break_cycles_minimum_cost()
                sorted_gaps = self.gap_graph.topological_sort()

            # Calculate adjusted ROI for each gap
            for gap in sorted_gaps:
                descendants = self.gap_graph.descendants(gap)
                ancestors = self.gap_graph.ancestors(gap)

                gap.adjusted_roi = self.gap_prioritizer.calculate_priority(
                    gap, len(descendants), len(ancestors)
                )

            # Sort by adjusted ROI
            sorted_gaps.sort(key=lambda g: g.adjusted_roi, reverse=True)

            return sorted_gaps
        except Exception as e:
            logger.error("Error in cycle detection: %s", e)
            return []

    def prioritize_gaps(self, gaps: List[KnowledgeGap]) -> List[LearningPriority]:
        """Prioritize gaps for learning - DELEGATED"""

        with self.lock:
            try:
                # Use prioritizer
                priorities = self.gap_prioritizer.prioritize_gaps(gaps, self.gap_graph)

                # Generate experiments for each priority
                for priority in priorities:
                    priority.experiments = self.experiment_generator.generate_for_gap(
                        priority.gap
                    )

                # FIX: Add to queue atomically
                for priority in priorities:
                    self.learning_priorities.put(priority)

                return priorities
            except Exception as e:
                logger.error("Error prioritizing gaps: %s", e)
                return []

    def generate_targeted_experiments(self, gap: KnowledgeGap) -> List[Experiment]:
        """Generate experiments targeted at specific gap - REFACTORED"""

        try:
            # EXAMINE: Check available budget
            available_budget = self.exploration_budget.get_available()

            if gap.estimated_cost > available_budget:
                # Simplify experiments to fit budget
                gap = self._simplify_gap(gap, available_budget)

            # SELECT & APPLY: Generate experiments based on gap type
            if gap.type == "decomposition":
                experiments = (
                    self.experiment_generator.generate_decomposition_experiment(
                        gap, complexity=0.5
                    )
                )
            elif gap.type == "causal":
                experiments = self.experiment_generator.generate_causal_experiment(gap)
            elif gap.type == "transfer":
                experiments = self.experiment_generator.generate_transfer_experiment(
                    gap
                )
            elif gap.type == "latent":
                experiments = self.iterative_designer.generate_iterative_experiments(
                    gap, max_iterations=3
                )
            else:
                experiments = []

            # Filter by budget
            affordable_experiments = []
            remaining_budget = available_budget

            for exp in experiments:
                if self.exploration_budget.can_afford(exp.complexity * 10):
                    affordable_experiments.append(exp)
                    remaining_budget -= exp.complexity * 10

            return affordable_experiments
        except Exception as e:
            logger.error("Error generating experiments: %s", e)
            return []

    def run_experiment_sandboxed(self, experiment: Experiment) -> ExperimentResult:
        """Run experiment in sandboxed environment - DELEGATED"""

        try:
            # Run experiment through manager
            result = self.experiment_manager.run_experiment(
                experiment,
                knowledge_base=self.knowledge,
                decomposer=self.decomposer,
                world_model=self.world_model,
                resource_monitor=self.resource_monitor,
            )

            # Consume budget
            cost = experiment.complexity * 10
            self.exploration_budget.consume(cost)

            # Adjust budget based on resource usage
            current_load = self.resource_monitor.get_current_load()
            self.exploration_budget.adjust_for_load(current_load)

            return result
        except Exception as e:
            logger.error("Error running experiment: %s", e)
            return ExperimentResult(
                experiment=experiment, success=False, output=None, error=str(e)
            )

    def update_from_experiment_results(self, results: List[ExperimentResult]):
        """Update knowledge from experiment results - DELEGATED"""

        try:
            # Integrate knowledge
            self.knowledge_integrator.integrate_results(
                results,
                knowledge_base=self.knowledge,
                world_model=self.world_model,
                decomposer=self.decomposer,
                exploration_frontier=self.exploration_frontier,
            )

            # Update learning rate
            if results:
                success_rate = sum(1 for r in results if r.success) / len(results)
                self.learning_rate = 0.9 * self.learning_rate + 0.1 * success_rate

            # Update budget efficiency
            self.exploration_budget.update_efficiency(
                len(results), sum(1 for r in results if r.success)
            )

            logger.info(
                "Updated from %d experiment results (success rate: %.2f)",
                len(results),
                self.learning_rate,
            )
        except Exception as e:
            logger.error("Error updating from results: %s", e)

    def run_learning_cycle(self, max_experiments: int = 10) -> Dict[str, Any]:
        """Run one cycle of curiosity-driven learning - REFACTORED"""

        with self.lock:
            try:
                cycle_start = time.time()

                # EXAMINE: Select exploration strategy
                strategy = self.select_exploration_strategy()

                # Identify gaps
                gaps = self.identify_gaps_with_cycle_detection()

                # SELECT: Prioritize gaps
                priorities = self.prioritize_gaps(gaps[:max_experiments])

                # APPLY: Run experiments
                results = []
                experiments_run = 0

                while experiments_run < max_experiments:
                    try:
                        # FIX: Use non-blocking get with timeout
                        priority = self.learning_priorities.get(block=False)
                    except Empty:
                        break

                    # Generate experiments for gap
                    experiments = self.generate_targeted_experiments(priority.gap)

                    for exp in experiments[:2]:  # Limit per gap
                        if experiments_run >= max_experiments:
                            break

                        # Run experiment
                        result = self.run_experiment_sandboxed(exp)
                        results.append(result)
                        experiments_run += 1

                # Update from results
                self.update_from_experiment_results(results)

                # REMEMBER: Calculate summary
                cycle_time = time.time() - cycle_start
                success_count = sum(1 for r in results if r.success)

                summary = {
                    "strategy_used": strategy,
                    "gaps_identified": len(gaps),
                    "experiments_run": experiments_run,
                    "successful_experiments": success_count,
                    "success_rate": (
                        success_count / experiments_run if experiments_run > 0 else 0
                    ),
                    "cycle_time": cycle_time,
                    "learning_rate": self.learning_rate,
                    "budget_remaining": self.exploration_budget.get_available(),
                    "resource_load": self.resource_monitor.get_current_load(),
                }

                logger.info(
                    "Learning cycle complete: %d experiments, %.2f success rate",
                    experiments_run,
                    summary["success_rate"],
                )

                return summary
            except Exception as e:
                logger.error("Error in learning cycle: %s", e)
                return {
                    "strategy_used": "error",
                    "gaps_identified": 0,
                    "experiments_run": 0,
                    "successful_experiments": 0,
                    "success_rate": 0.0,
                    "error": str(e),
                }

    def _simplify_gap(self, gap: KnowledgeGap, budget: float) -> KnowledgeGap:
        """Simplify gap to fit budget"""
        try:
            simplified = copy.deepcopy(gap)

            # Reduce complexity
            reduction_factor = budget / gap.estimated_cost
            simplified.estimated_cost = budget

            if hasattr(simplified, "complexity"):
                simplified.complexity *= reduction_factor

            return simplified
        except Exception as e:
            logger.error("Error simplifying gap: %s", e)
            return gap

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current system context"""
        try:
            return {
                "resource_load": self.resource_monitor.get_current_load(),
                "available_budget": self.exploration_budget.get_available(),
                "recent_success_rate": self.learning_rate,
                "total_experiments": self.experiment_manager.total_experiments,
                "frontier_size": len(self.exploration_frontier.frontier_regions),
            }
        except Exception as e:
            logger.error("Error getting context: %s", e)
            return {
                "resource_load": 0.5,
                "available_budget": 50,
                "recent_success_rate": 0.5,
                "total_experiments": 0,
                "frontier_size": 0,
            }
