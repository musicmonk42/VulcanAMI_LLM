"""
curiosity_engine_core.py - Main curiosity-driven learning orchestrator
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
"""

import copy
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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

# Import ExecutionTrace type for optional crystallization feature
try:
    from ..knowledge_crystallizer.principle_extractor import (
        ExecutionTrace as PrincipleExtractorTrace,
    )

    CRYSTALLIZATION_AVAILABLE = True
except ImportError:
    PrincipleExtractorTrace = None
    CRYSTALLIZATION_AVAILABLE = False

# Note: Import resolution_bridge for cross-process state persistence
# This fixes Bug #1 (Phantom Resolution Loop) and Bug #2 (Cold Start Always Triggered)
from .resolution_bridge import (
    is_gap_resolved as _persistent_is_gap_resolved,
    mark_gap_resolved as _persistent_mark_gap_resolved,
    mark_gap_resolved_batch as _persistent_mark_gap_resolved_batch,
    get_gap_attempts as _persistent_get_gap_attempts,
    increment_gap_attempts as _persistent_increment_gap_attempts,
    reset_gap_attempts as _persistent_reset_gap_attempts,
    record_resolution_history as _persistent_record_resolution_history,
    get_recent_resolutions_count as _persistent_get_recent_resolutions_count,
    is_phantom_resolution as _persistent_is_phantom_resolution,
    get_experiment_count as _persistent_get_experiment_count,
    increment_experiment_count as _persistent_increment_experiment_count,
)

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

        # Note: Copy items to avoid modification during iteration
        regions_items = list(self.explored_regions.items())

        for region_id, region in regions_items:
            if region.domain == domain:
                overlap = len(patterns & region.patterns)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = region_id

        # Note: Check both empty patterns and sufficient overlap
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

            # Note: Copy items to avoid modification during iteration
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

            # Note: Clean up graph connections safely
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
            # Note: Check both hasattr and callable
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

            # Note: Check both hasattr and callable
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
                # Simulate causal discovery with deterministic values
                # Use intervention properties to calculate consistent results
                import zlib

                intervention_str = str(intervention.get("variable", "")) + str(
                    intervention.get("value", "")
                )
                intervention_hash = zlib.crc32(intervention_str.encode()) & 0xffffffff

                # Deterministic causal strength based on intervention
                causal_strength = (
                    0.3 + (intervention_hash % 700) / 1000.0
                )  # Range: 0.3 to 1.0
                p_value = (intervention_hash % 100) / 1000.0  # Range: 0.0 to 0.1
                effect_size = (intervention_hash % 800) / 1000.0  # Range: 0.0 to 0.8

                result["success"] = causal_strength > 0.3
                result["data"] = {
                    "causal_strength": causal_strength,
                    "p_value": p_value,
                    "effect_size": effect_size,
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

            # Note: Check both hasattr and callable
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
            # Note: Check both hasattr and callable
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
            # Note: Check if adjusted_roi exists AND is not None
            if hasattr(gap, "adjusted_roi") and gap.adjusted_roi is not None:
                priority = gap.adjusted_roi
            else:
                base_roi = base_priority / max(gap.estimated_cost, 1)
                unlock_bonus = descendants_count * 0.1
                dependency_penalty = ancestors_count * 0.05
                priority = base_roi * (1 + unlock_bonus - dependency_penalty)

            # Note: Ensure priority is not None before multiplication
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

                    # Note: Validate priority score
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
        """
        Update knowledge crystallizer with enhanced integration
        
        Attempts to crystallize experiment knowledge using full principle extraction.
        Falls back to simple storage if crystallization is not available or fails.
        
        Args:
            knowledge: Learned knowledge dictionary from experiment
            knowledge_base: Knowledge crystallizer instance
        """
        try:
            # Attempt crystallization for richer principle extraction
            if hasattr(knowledge_base, "crystallize") and callable(
                getattr(knowledge_base, "crystallize", None)
            ):
                try:
                    execution_trace = self._convert_to_execution_trace(knowledge)
                    if execution_trace:
                        result = knowledge_base.crystallize(execution_trace)
                        logger.info(
                            "Crystallized experiment knowledge: %d principles extracted with confidence %.2f",
                            len(result.principles),
                            result.confidence,
                        )
                        return  # Success - crystallization completed
                except Exception as e:
                    logger.warning(
                        "Crystallization failed, falling back to store_knowledge: %s", e
                    )
            
            # Fallback: Use simple storage for individual knowledge items
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

    def _convert_to_execution_trace(
        self, knowledge: Dict[str, Any]
    ) -> Optional["PrincipleExtractorTrace"]:
        """
        Convert experiment learned knowledge to ExecutionTrace for crystallization
        
        This method creates a properly structured ExecutionTrace that can be
        processed by the KnowledgeCrystallizer for full principle extraction,
        including pattern detection, confidence calculation, and domain assignment.
        
        Args:
            knowledge: Learned knowledge dictionary from experiment
            
        Returns:
            ExecutionTrace object if conversion successful, None otherwise
        """
        # Check if crystallization is available at module level
        if not CRYSTALLIZATION_AVAILABLE or PrincipleExtractorTrace is None:
            logger.debug("Crystallization not available, skipping trace conversion")
            return None
            
        try:
            # Extract experiment metadata
            experiment_type = knowledge.get("experiment_type", "unknown")
            domain = knowledge.get("domain", "general")
            
            # Build actions list from knowledge patterns
            actions = []
            if "patterns" in knowledge:
                patterns = knowledge["patterns"]
                if isinstance(patterns, list):
                    for i, pattern in enumerate(patterns):
                        actions.append(
                            {
                                "type": "pattern_application",
                                "params": {"pattern": pattern, "step": i},
                            }
                        )
                elif isinstance(patterns, dict):
                    for key, value in patterns.items():
                        actions.append(
                            {"type": "pattern_application", "params": {key: value}}
                        )
            
            # If no patterns, create actions from other knowledge keys
            if not actions:
                for key, value in knowledge.items():
                    if key not in [
                        "raw_output",
                        "experiment_type",
                        "domain",
                        "observations",
                    ]:
                        actions.append(
                            {"type": "knowledge_acquisition", "params": {key: value}}
                        )
            
            # Ensure we have at least one action
            if not actions:
                actions.append(
                    {
                        "type": "experiment",
                        "params": {"type": experiment_type, "data": knowledge},
                    }
                )
            
            # Extract outcomes from knowledge
            outcomes = {
                "success": True,  # Assume success if we got learned knowledge
                "experiment_type": experiment_type,
            }
            
            # Add any metrics or results
            for key in ["accuracy", "precision", "recall", "value", "score"]:
                if key in knowledge:
                    outcomes[key] = knowledge[key]
            
            # Build context from remaining knowledge
            context = {
                "domain": domain,
                "source": "curiosity_engine_experiment",
            }
            
            # Add observations to context if present
            if "observations" in knowledge:
                context["observations"] = knowledge["observations"]
            
            # Generate unique trace_id using UUID for guaranteed uniqueness
            trace_id = f"experiment_{experiment_type}_{uuid.uuid4().hex[:16]}"
            
            # Create ExecutionTrace with proper structure for principle_extractor
            trace = PrincipleExtractorTrace(
                trace_id=trace_id,
                actions=actions,
                outcomes=outcomes,
                context=context,
                success=True,
                domain=domain,
                metadata={
                    "source": "curiosity_engine",
                    "experiment_type": experiment_type,
                },
            )
            
            logger.debug(
                "Converted experiment knowledge to ExecutionTrace: %d actions, domain=%s",
                len(actions),
                domain,
            )
            
            return trace
            
        except Exception as e:
            logger.warning("Failed to convert knowledge to ExecutionTrace: %s", e)
            return None

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
    
    # Singleton pattern to prevent dual instances (fixes resource waste from duplicate PIDs)
    _instance = None
    _instance_lock = threading.Lock()
    
    # Note: Configuration constants for query ingestion
    MAX_QUERY_TRUNCATE_LENGTH = 200  # Maximum length for query truncation in failure storage

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _reset_singleton(cls):
        """Reset singleton instance for testing purposes only.
        
        WARNING: This method is intended for unit testing only.
        Do not use in production code as it defeats the purpose of the singleton.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                # Clear initialized flag so __init__ runs again
                if hasattr(cls._instance, '_initialized'):
                    cls._instance._initialized = False
            cls._instance = None

    @staticmethod
    def _safe_truncate(text: str, max_length: int) -> str:
        """
        Safely truncate text to max_length, respecting unicode character boundaries.
        
        Uses encode/decode to ensure we don't cut in the middle of multi-byte characters.
        
        Args:
            text: The text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text that doesn't cut in the middle of a unicode character
        """
        if len(text) <= max_length:
            return text
        
        # Simple approach: truncate and then encode/decode to fix any broken characters
        # This handles surrogate pairs and multi-byte sequences correctly
        truncated = text[:max_length]
        
        # Encode to UTF-8 bytes and decode with error handling to fix any broken sequences
        try:
            # This will fail if we cut in the middle of a surrogate pair
            truncated.encode('utf-8')
            return truncated
        except UnicodeEncodeError:
            # Back up character by character until we get valid UTF-8
            while truncated:
                truncated = truncated[:-1]
                try:
                    truncated.encode('utf-8')
                    return truncated
                except UnicodeEncodeError:
                    continue
            return ""

    def __init__(self, knowledge=None, decomposer=None, world_model=None):
        """
        Initialize curiosity engine

        Args:
            knowledge: Knowledge crystallizer instance
            decomposer: Problem decomposer instance
            world_model: World model instance
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized') and self._initialized:
            # Allow updating dependencies if provided
            if knowledge is not None:
                self.knowledge = knowledge
            if decomposer is not None:
                self.decomposer = decomposer
            if world_model is not None:
                self.world_model = world_model
            return
        
        self._initialized = True
        
        self.knowledge = knowledge
        self.decomposer = decomposer
        self.world_model = world_model

        # Note: Initialize in dependency order
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
        
        # Note: Track if any data has been ingested
        self._queries_ingested = 0
        self._failures_ingested = 0
        
        # ==============================================================================
        # Note: Gap resolution tracking
        # ==============================================================================
        # Gaps were growing unbounded because:
        # 1. False "errors" from bad consensus check added high_error_rate gaps
        # 2. Experiments ran but gaps were never marked resolved
        # 3. No deduplication - same gap type added repeatedly
        # Note: Track resolved gap keys with TTL (key -> resolution_timestamp)
        self._resolved_gaps: Dict[str, float] = {}  
        self._gap_last_seen: Dict[str, float] = {}  # Track when gap was last added
        self._gap_attempts: Dict[str, int] = {}  # Track experiment attempts per gap
        self._last_resolution_cleanup: float = 0.0  # Last time we cleaned up expired resolutions
        # FIX Issue #13: Track resolution counts per gap type to detect phantom resolutions
        # key -> list of (timestamp, was_success) tuples
        self._gap_resolution_history: Dict[str, List[Tuple[float, bool]]] = {}
        # Note: Increased MAX_GAPS_PER_TYPE from 2 to 5 to allow more
        # experiments per learning cycle. The previous limit of 2 was too restrictive
        # and caused experiments=0 in most cycles.
        self.MAX_GAPS_PER_TYPE = 5  # Maximum gaps of same type (was 2)
        # Note: Reduced GAP_COOLDOWN_SECONDS from 300 to 120 to allow
        # faster re-processing of gaps. 5 minute cooldown was too long.
        self.GAP_COOLDOWN_SECONDS = 120  # Don't re-add same gap for 2 min (was 5 min)
        self.GAP_RESOLUTION_TTL_SECONDS = 1800  # Note: Allow re-detection after 30 min if issue persists
        self.RESOLUTION_CLEANUP_INTERVAL = 300  # Only cleanup expired resolutions every 5 minutes
        # FIX Issue #13: Threshold for detecting phantom resolutions
        self.PHANTOM_RESOLUTION_THRESHOLD = 3  # If resolved 3+ times in an hour, it's not really resolved
        self.PHANTOM_RESOLUTION_WINDOW = 3600  # 1 hour window for counting phantom resolutions
        # Note: Phantom Resolution Loop - extended cooldown for gaps that keep returning
        self.PHANTOM_GAP_COOLDOWN_SECONDS = 3600  # 1 hour cooldown for phantom gaps
        # BUG FIX #3: Circuit breaker for phantom resolutions
        # Track suppressed gaps to prevent infinite loop of same gap being re-injected
        # key -> timestamp when suppression expires
        self._suppressed_gaps: Dict[str, float] = {}
        # Note: Learning System Give-Up Threshold
        # Increased from 3 to 10 - don't mark gaps as "resolved" after only 3 failures
        # Use environment variable to configure: VULCAN_GAP_GIVEUP_THRESHOLD
        import os
        self.GAP_GIVEUP_THRESHOLD = int(os.environ.get("VULCAN_GAP_GIVEUP_THRESHOLD", "10"))
        
        # Note: Bootstrap experiment constants
        self.BOOTSTRAP_EXPERIMENT_TIMEOUT_SECONDS = 30.0  # Short timeout for bootstrap experiments
        self.BOOTSTRAP_MEMORY_LIMIT_BYTES = 128 * 1024 * 1024  # 128MB memory limit for bootstrap
        
        # External gap injection (for OutcomeBridge connection)
        # This allows gaps detected by external systems (e.g., OutcomeBridge)
        # to be included in the learning cycle
        self._external_gaps: List[KnowledgeGap] = []
        self._external_gaps_lock = threading.RLock()
        
        # Callback to SelfImprovementDrive for gap-driven priority boosting
        # When set, detected gaps will be sent to SelfImprovementDrive to
        # boost relevant improvement objectives (e.g., slow_routing -> optimize_performance)
        self._on_gaps_detected_callback: Optional[Callable[[List[Dict[str, Any]]], Any]] = None

        # ==============================================================================
        # INTELLIGENT CYCLE MANAGEMENT (Issue: Learning cycle 340-365: 0 experiments)
        # ==============================================================================
        # The engine was blindly running every cycle regardless of work availability.
        # These attributes enable should_run_cycle() to implement progressive backoff.
        self._empty_cycles = 0  # Count of consecutive cycles with 0 experiments
        self._last_successful_cycle = time.time()  # Timestamp of last cycle with experiments
        self._last_cycle_time = time.time()  # Timestamp of last cycle attempt
        self._backoff_multiplier = 1  # Current backoff multiplier
        self._base_cycle_interval = 10  # Base interval between cycles in seconds
        self._max_backoff_exponent = 5  # Max backoff: 2^5 * base = 320 seconds
        self._last_outcome_count = 0  # Track outcome count for wake triggers
        self._cycles_without_experiments = 0  # For exploration forcing
        self._exploration_temperature = 1.0  # Multiplier for exploration when stagnant
        
        # Thresholds for intelligent cycle management
        self.EMPTY_CYCLE_SKIP_THRESHOLD = 3  # Start backoff after this many empty cycles
        self.EXPLORATION_FORCE_THRESHOLD = 20  # Force exploration after this many cycles without experiments

        # Thread safety
        self.lock = threading.RLock()

        logger.info("CuriosityEngine initialized (refactored)")

    # ========== Note: Query data ingestion methods ==========
    
    def ingest_query_result(
        self, 
        query: str, 
        result: Dict[str, Any], 
        success: bool,
        domain: str = "unknown",
        query_type: str = "general"
    ) -> None:
        """
        Feed query outcomes to the learning system.
        
        Note: This method connects the curiosity engine to actual query data.
        Call this from the query processing pipeline to feed outcomes to the engine.
        
        Args:
            query: The original query text
            result: The result dictionary from query processing
            success: Whether the query was successful
            domain: The domain/topic of the query
            query_type: The type of query (reasoning, perception, etc.)
        """
        with self.lock:
            self._queries_ingested += 1
            
            if not success:
                self._failures_ingested += 1
                
                # Record as a failure for gap analysis
                # Use safe truncation to handle unicode characters properly
                failure_data = {
                    "query": self._safe_truncate(query, self.MAX_QUERY_TRUNCATE_LENGTH),
                    "domain": domain,
                    "query_type": query_type,
                    "error": result.get("error", "Unknown error"),
                    "complexity": result.get("complexity", 0.5),
                    "pattern": query_type,
                }
                
                # Determine failure type based on query_type
                if query_type in ("decomposition", "planning"):
                    self.gap_analyzer.record_failure("decomposition", failure_data)
                elif query_type in ("reasoning", "causal"):
                    self.gap_analyzer.record_failure("prediction", failure_data)
                elif query_type in ("transfer", "learning"):
                    self.gap_analyzer.record_failure("transfer", failure_data)
                else:
                    # Default to decomposition for general failures
                    self.gap_analyzer.record_failure("decomposition", failure_data)
                
                logger.debug(
                    f"[CuriosityEngine] Ingested failure: domain={domain}, type={query_type}"
                )
            else:
                # Record successful patterns for exploration
                patterns = set()
                if query_type:
                    patterns.add(query_type)
                if domain:
                    patterns.add(domain)
                
                if patterns:
                    self.exploration_frontier.add_explored_region(domain, patterns)
                    
            # Log ingestion stats periodically
            if self._queries_ingested % 100 == 0:
                logger.info(
                    f"[CuriosityEngine] Ingested {self._queries_ingested} queries "
                    f"({self._failures_ingested} failures)"
                )
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about data ingestion for monitoring.
        
        Note: Returns stats to help diagnose why no gaps are found.
        """
        with self.lock:
            return {
                "queries_ingested": self._queries_ingested,
                "failures_ingested": self._failures_ingested,
                "gap_analyzer_stats": self.gap_analyzer.get_statistics(),
                "failure_rate": (
                    self._failures_ingested / self._queries_ingested 
                    if self._queries_ingested > 0 else 0.0
                ),
            }
    
    # ========== External Gap Injection (OutcomeBridge Connection) ==========
    
    def add_external_gap(self, gap: KnowledgeGap) -> None:
        """
        Add a gap from an external source (e.g., OutcomeBridge).
        
        This method allows external systems to inject gaps directly into the
        learning cycle. Gaps added this way will be included in the next call
        to identify_knowledge_gaps().
        
        FIX Issue 5: Now triggers CuriosityDriver wake to process gaps immediately
        instead of waiting for the dormant mode check interval.
        
        Args:
            gap: KnowledgeGap to add to the external gaps queue
        """
        with self._external_gaps_lock:
            self._external_gaps.append(gap)
            logger.info(
                f"[CuriosityEngine] Added external gap: type={gap.type}, "
                f"domain={gap.domain}, priority={gap.priority:.2f}"
            )
        
        # FIX Issue 5: Wake CuriosityDriver if dormant
        # This ensures externally injected gaps trigger immediate processing
        self._wake_curiosity_driver()
    
    def add_external_gaps(self, gaps: List[KnowledgeGap]) -> None:
        """
        Add multiple gaps from an external source.
        
        FIX Issue 5: Now triggers CuriosityDriver wake to process gaps immediately
        instead of waiting for the dormant mode check interval.
        
        Args:
            gaps: List of KnowledgeGaps to add
        """
        with self._external_gaps_lock:
            for gap in gaps:
                self._external_gaps.append(gap)
            if gaps:
                logger.info(
                    f"[CuriosityEngine] Added {len(gaps)} external gaps"
                )
                # FIX Issue 5: Wake CuriosityDriver if dormant
                self._wake_curiosity_driver()
    
    def _wake_curiosity_driver(self) -> None:
        """
        Wake the CuriosityDriver from dormant mode.
        
        FIX Issue 5: Helper method to trigger driver wake when new work arrives.
        Uses multiple fallback locations to find the driver instance.
        """
        try:
            curiosity_driver = None
            
            # Attempt 1: Import vulcan.main and check its app.state
            try:
                from src.vulcan.main import app as vulcan_app
                curiosity_driver = getattr(vulcan_app.state, 'curiosity_driver', None)
            except (ImportError, AttributeError):
                pass
            
            # Attempt 2: Try alternative import path
            if curiosity_driver is None:
                try:
                    import vulcan.main as vulcan_main
                    if hasattr(vulcan_main, 'app'):
                        curiosity_driver = getattr(vulcan_main.app.state, 'curiosity_driver', None)
                except (ImportError, AttributeError):
                    pass
            
            # Wake driver if found and dormant
            if curiosity_driver and hasattr(curiosity_driver, 'wake_from_dormant'):
                if getattr(curiosity_driver, 'is_dormant', False):
                    curiosity_driver.wake_from_dormant()
                    logger.debug(
                        "[CuriosityEngine] Woke CuriosityDriver from dormant mode "
                        "(external gap injected)"
                    )
        except Exception as wake_err:
            # Non-critical error - don't fail gap injection
            logger.debug(f"[CuriosityEngine] Could not wake CuriosityDriver: {wake_err}")
    
    def get_external_gaps_count(self) -> int:
        """Get the number of pending external gaps."""
        with self._external_gaps_lock:
            return len(self._external_gaps)
    
    # ========== SelfImprovementDrive Connection ==========
    
    def set_on_gaps_detected_callback(
        self, callback: Callable[[List[Dict[str, Any]]], Any]
    ) -> None:
        """
        Set callback to be called when gaps are detected.
        
        This connects CuriosityEngine's gap detection to SelfImprovementDrive's
        objective prioritization. When gaps are detected, the callback is called
        with the list of gaps, allowing SelfImprovementDrive to boost the weights
        of relevant improvement objectives.
        
        Usage:
            # In deployment initialization:
            curiosity_engine = CuriosityEngine()
            self_improvement = SelfImprovementDrive()
            
            # Connect gap detection to priority boosting
            curiosity_engine.set_on_gaps_detected_callback(
                self_improvement.process_gaps_from_curiosity_engine
            )
        
        Args:
            callback: Function that accepts List[Dict[str, Any]] of gaps
                     Typically: self_improvement.process_gaps_from_curiosity_engine
        """
        self._on_gaps_detected_callback = callback
        logger.info(
            "[CuriosityEngine] Gap detection callback set - "
            "detected gaps will now boost SelfImprovementDrive objectives"
        )
    
    def _notify_gaps_detected(self, gaps: List[KnowledgeGap]) -> None:
        """
        Notify callback when gaps are detected.
        
        Converts KnowledgeGap objects to dictionaries and calls the callback.
        This enables SelfImprovementDrive to boost relevant objectives based
        on the types of gaps detected.
        
        Args:
            gaps: List of detected knowledge gaps
        """
        if not self._on_gaps_detected_callback or not gaps:
            return
        
        try:
            # Convert gaps to dictionary format for callback
            gap_dicts = []
            for gap in gaps:
                gap_dict = {
                    "type": gap.type,
                    "priority": gap.priority,
                    "domain": gap.domain,
                    "estimated_cost": gap.estimated_cost,
                    "metadata": gap.metadata if hasattr(gap, "metadata") else {},
                }
                gap_dicts.append(gap_dict)
            
            # Call the callback (typically SelfImprovementDrive.process_gaps_from_curiosity_engine)
            result = self._on_gaps_detected_callback(gap_dicts)
            
            if result:
                logger.info(
                    f"[CuriosityEngine] Notified SelfImprovementDrive of {len(gaps)} gaps, "
                    f"boosted objectives: {result}"
                )
        except Exception as e:
            logger.warning(
                f"[CuriosityEngine] Gap notification callback failed: {e}"
            )
    
    def _consume_external_gaps(self) -> List[KnowledgeGap]:
        """
        Consume and return all pending external gaps.
        
        This clears the external gaps queue after returning the gaps.
        
        Returns:
            List of external gaps that were pending
        """
        with self._external_gaps_lock:
            gaps = self._external_gaps.copy()
            self._external_gaps = []
            if gaps:
                logger.info(
                    f"[CuriosityEngine] Consumed {len(gaps)} external gaps"
                )
            return gaps

    # ==============================================================================
    # Note: Gap resolution tracking methods
    # ==============================================================================
    
    def _gap_key(self, gap: Union[KnowledgeGap, Dict]) -> str:
        """Generate unique key for gap deduplication.
        
        Args:
            gap: KnowledgeGap object or dictionary with type/domain
            
        Returns:
            Unique key string for the gap
        """
        if isinstance(gap, dict):
            gap_type = gap.get('type', 'unknown')
            domain = gap.get('domain', 'general')
        else:
            gap_type = getattr(gap, 'type', 'unknown')
            domain = getattr(gap, 'domain', 'general')
        return f"{gap_type}:{domain}"
    
    def mark_gap_resolved(self, gap: Union[KnowledgeGap, Dict, str], domain: Optional[str] = None, success: bool = True, cycle_id: Optional[int] = None) -> None:
        """Mark a gap as resolved after successful experiment.
        
        Note: Ensures gaps don't accumulate forever by tracking
        which gaps have been addressed.
        
        Note: Also marks gaps as resolved when giving up (success=False)
        to prevent the attempt counter from growing indefinitely (3 → 9 → 11 → 13 → 19).
        Gaps marked as resolved (regardless of success) will be filtered out
        for the cooldown period, and attempts will be reset.
        
        Note (Phantom Resolution Loop): Now persists resolution state to SQLite
        via resolution_bridge, so resolutions survive subprocess restarts.
        
        Args:
            gap: The gap to mark as resolved (KnowledgeGap object, dict, or gap_type string)
            domain: Domain string (required if gap is a string, otherwise extracted from gap object)
            success: Whether resolution was successful (default True)
            cycle_id: Optional learning cycle ID for tracking (default None)
        """
        # Handle different calling patterns
        if isinstance(gap, str):
            # Called with (gap_type, domain, success) pattern
            if domain is None:
                raise ValueError("domain is required when gap is a string")
            key = f"{gap}:{domain}"
        else:
            # Called with gap object or dict
            key = self._gap_key(gap)
        
        current_time = time.time()
        
        # FIX Bug #1: Persist to SQLite for cross-process visibility using batch operation
        # This ensures subprocesses know about resolutions from previous cycles
        # Using batch operation for atomicity and performance
        try:
            _persistent_mark_gap_resolved_batch(
                key,
                success=success,
                cycle_id=cycle_id,
                reset_attempts=not success,  # Reset attempts when giving up
            )
        except Exception as e:
            logger.debug(f"[CuriosityEngine] Could not persist resolution to SQLite: {e}")
        
        # FIX Issue #13: Track resolution history for phantom detection (in-memory cache)
        if key not in self._gap_resolution_history:
            self._gap_resolution_history[key] = []
        self._gap_resolution_history[key].append((current_time, success))
        
        # Clean up old resolution history entries (older than window)
        self._gap_resolution_history[key] = [
            (ts, s) for ts, s in self._gap_resolution_history[key]
            if current_time - ts < self.PHANTOM_RESOLUTION_WINDOW
        ]
        
        # Note: Always add to resolved set with timestamp (not just on success)
        # This prevents the same gap from being re-added immediately
        self._resolved_gaps[key] = current_time
        
        # Note: Reset attempts counter when gap is resolved (success or give-up)
        # This prevents the log showing "after 19 attempts" when it should be "after 3 attempts"
        if key in self._gap_attempts:
            del self._gap_attempts[key]
        
        # FIX Issue #13: Check for phantom resolution pattern (from persistent storage)
        try:
            recent_resolutions = _persistent_get_recent_resolutions_count(key)
        except Exception:
            recent_resolutions = len(self._gap_resolution_history[key])
        
        # BUG FIX #3: Phantom resolution circuit breaker
        if recent_resolutions >= self.PHANTOM_RESOLUTION_THRESHOLD:
            # Calculate exponential backoff: 2^(resolutions - threshold) hours
            # 3 resolutions -> 1 hour, 4 -> 2 hours, 5 -> 4 hours, etc.
            backoff_hours = 2 ** (recent_resolutions - self.PHANTOM_RESOLUTION_THRESHOLD)
            backoff_seconds = min(backoff_hours * 3600, 24 * 3600)  # Cap at 24 hours
            
            logger.error(
                f"[CuriosityEngine] PHANTOM RESOLUTION CIRCUIT BREAKER: "
                f"Gap {key} 'resolved' {recent_resolutions}x in last hour. "
                f"SUPPRESSING this gap type for {backoff_hours} hours to prevent infinite loop."
            )
            
            # Suppress the gap for the calculated backoff period
            self._suppressed_gaps[key] = time.time() + backoff_seconds
        elif recent_resolutions > 1:
            # Still log warning for tracking, but don't suppress yet
            logger.warning(
                f"[CuriosityEngine] PHANTOM RESOLUTION: Gap {key} 'resolved' {recent_resolutions}x "
                f"in last hour - underlying issue likely NOT fixed"
            )
        
        if success:
            logger.info(f"[CuriosityEngine] Gap {key} resolved by successful experiment")
        else:
            logger.info(f"[CuriosityEngine] Gap {key} marked resolved after giving up")
    
    def _count_recent_resolutions(self, gap_type: str, domain: str = "query_processing", minutes: int = 60) -> int:
        """
        Count how many times a gap type has been 'resolved' recently.
        
        FIX Issue #13: Helps detect phantom resolutions where gaps keep
        getting marked as resolved but immediately reappear.
        
        Args:
            gap_type: Type of gap (e.g., 'high_error_rate')
            domain: Domain of gap (default 'query_processing')
            minutes: Time window in minutes (default 60)
            
        Returns:
            Number of resolutions in the time window
        """
        key = f"{gap_type}:{domain}"
        if key not in self._gap_resolution_history:
            return 0
        
        cutoff = time.time() - (minutes * 60)
        return sum(1 for ts, _ in self._gap_resolution_history[key] if ts > cutoff)
    
    def _is_gap_truly_resolved(self, gap_type: str, domain: str = "query_processing") -> bool:
        """
        Check if gap is actually resolved based on actual success rate, not just marked.
        
        FIX Issue #13: A gap should only be considered truly resolved if:
        1. Recent success rate is > 80%
        2. It hasn't been "resolved" 3+ times in the last hour (phantom resolution)
        
        Args:
            gap_type: Type of gap (e.g., 'high_error_rate')
            domain: Domain of gap (default 'query_processing')
            
        Returns:
            True if gap is truly resolved, False if it's a phantom resolution
        """
        key = f"{gap_type}:{domain}"
        
        # Check for phantom resolution pattern
        recent_resolution_count = self._count_recent_resolutions(gap_type, domain, minutes=60)
        if recent_resolution_count >= self.PHANTOM_RESOLUTION_THRESHOLD:
            logger.debug(
                f"[CuriosityEngine] Gap {key} has {recent_resolution_count} phantom resolutions - not truly resolved"
            )
            return False
        
        # Check actual success rate from outcome bridge
        # BUG #10 FIX: Use answer_quality instead of status to measure TRUE success
        # The previous code used status=='success' which only measures "didn't crash"
        # not "produced correct answer". This is the root cause of 0 experiments
        # being generated despite obvious failures.
        try:
            from .outcome_bridge import get_recent_outcomes
            outcomes = get_recent_outcomes(minutes=10)
            
            if not outcomes:
                # No data to verify - assume not resolved
                return False
            
            # BUG #10 FIX: Count quality-based success, not execution-based
            # "good" quality = answer was actually useful
            # status='success' just means "didn't crash" which is not meaningful
            quality_success_count = sum(
                1 for o in outcomes 
                if o.get('answer_quality') == 'good'
            )
            total_with_quality = sum(
                1 for o in outcomes 
                if o.get('answer_quality') is not None
            )
            
            if total_with_quality == 0:
                # No quality data - fall back to status-based check
                success_count = sum(1 for o in outcomes if o.get('status') == 'success')
                success_rate = success_count / len(outcomes)
            else:
                # Use quality-based success rate (correct metric)
                success_rate = quality_success_count / total_with_quality
            
            # Only consider resolved if success rate > 80%
            if success_rate < 0.8:
                logger.debug(
                    f"[CuriosityEngine] Gap {key} success_rate={success_rate:.1%} < 80% - not truly resolved"
                )
                return False
            
            return True
        except Exception as e:
            logger.debug(f"[CuriosityEngine] Could not verify gap resolution: {e}")
            return False
    
    def _filter_gaps_with_resolution(self, raw_gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Filter gaps based on resolution status, cooldown, and type limits.
        
        Note: Prevents gap accumulation by:
        1. Removing already resolved gaps
        2. Enforcing cooldown period per gap
        3. Limiting gaps per type
        
        Note: Resolved gaps now have a TTL - they can be re-detected
        after GAP_RESOLUTION_TTL_SECONDS (30 min) if the underlying issue persists.
        
        FIX Bug #1 (Phantom Resolution Loop): Now checks SQLite for resolved gaps
        via resolution_bridge, so subprocess instances know about resolutions from
        previous cycles. This prevents gaps being "resolved" 40-90 times per hour.
        
        Performance: Expired gap cleanup is rate-limited to every RESOLUTION_CLEANUP_INTERVAL
        seconds to avoid overhead on every call.
        
        Args:
            raw_gaps: List of raw gaps to filter
            
        Returns:
            Filtered list of gaps
        """
        current_time = time.time()
        filtered = []
        type_counts = defaultdict(int)
        
        # Note: Clean up expired resolved gaps (TTL-based)
        # Rate-limit cleanup to avoid overhead on every call
        if current_time - self._last_resolution_cleanup > self.RESOLUTION_CLEANUP_INTERVAL:
            self._last_resolution_cleanup = current_time
            expired_keys = [
                key for key, resolved_time in self._resolved_gaps.items()
                if current_time - resolved_time > self.GAP_RESOLUTION_TTL_SECONDS
            ]
            for key in expired_keys:
                del self._resolved_gaps[key]
                logger.debug(f"[CuriosityEngine] Resolved gap {key} TTL expired, can be re-detected")
        
        for gap in raw_gaps:
            key = self._gap_key(gap)
            gap_type = getattr(gap, 'type', 'unknown')
            domain = getattr(gap, 'domain', 'general')
            
            # FIX Bug #1: Check persistent storage FIRST for cross-process resolution state
            # This ensures subprocesses know about resolutions from previous cycles
            try:
                if _persistent_is_gap_resolved(key, ttl_seconds=self.GAP_RESOLUTION_TTL_SECONDS):
                    logger.info(
                        f"[CuriosityEngine] ISSUE #5 FIX: Gap {key} is resolved in persistent storage "
                        f"(TTL={self.GAP_RESOLUTION_TTL_SECONDS}s), skipping to prevent phantom resolution"
                    )
                    continue
            except Exception as e:
                logger.warning(
                    f"[CuriosityEngine] ISSUE #5: Failed to check persistent storage for gap {key}: {e}. "
                    f"Falling through to in-memory check."
                )
                # Fall through to in-memory check
            
            # FIX Bug #1: Check for phantom resolution pattern from persistent storage
            try:
                recent_resolution_count = _persistent_get_recent_resolutions_count(key)
                if recent_resolution_count > 0:
                    logger.debug(
                        f"[CuriosityEngine] Gap {key} has {recent_resolution_count} recent resolutions "
                        f"in persistent storage (threshold={self.PHANTOM_RESOLUTION_THRESHOLD})"
                    )
            except Exception as e:
                logger.debug(
                    f"[CuriosityEngine] ISSUE #5: Failed to check persistent resolution count for gap {key}: {e}"
                )
                recent_resolution_count = self._count_recent_resolutions(gap_type, domain, minutes=60)
            
            if recent_resolution_count >= self.PHANTOM_RESOLUTION_THRESHOLD:
                # Apply extended cooldown for phantom resolutions
                last_seen = self._gap_last_seen.get(key, 0)
                if current_time - last_seen < self.PHANTOM_GAP_COOLDOWN_SECONDS:
                    logger.debug(
                        f"[CuriosityEngine] Gap {key} is a phantom resolution ({recent_resolution_count}x in 1h) - "
                        f"applying extended {self.PHANTOM_GAP_COOLDOWN_SECONDS}s cooldown"
                    )
                    continue  # Skip this gap - it needs a longer break
                # If the extended cooldown has passed, log but allow through
                logger.info(
                    f"[CuriosityEngine] Gap {key} phantom resolution cooldown expired, "
                    f"allowing through after {recent_resolution_count}x resolutions"
                )
            
            # Skip if resolved in in-memory cache (and not expired)
            if key in self._resolved_gaps:
                continue
            
            # Skip if in cooldown
            last_seen = self._gap_last_seen.get(key, 0)
            if current_time - last_seen < self.GAP_COOLDOWN_SECONDS:
                continue
            
            # Skip if too many of this type
            if type_counts[gap_type] >= self.MAX_GAPS_PER_TYPE:
                continue
            
            filtered.append(gap)
            type_counts[gap_type] += 1
            self._gap_last_seen[key] = current_time
        
        logger.info(
            f"[GapAnalyzer] {len(raw_gaps)} raw → {len(filtered)} after filtering "
            f"(resolved={len(self._resolved_gaps)}, type_limits={dict(type_counts)})"
        )
        return filtered

    # ==============================================================================
    # INTELLIGENT CYCLE MANAGEMENT
    # ==============================================================================
    # Issue: Learning cycle 340-365: 0 experiments, 0.00 success rate
    # The engine was blindly running every cycle regardless of work availability.
    # These methods implement progressive backoff and wake triggers.
    
    def should_run_cycle(self) -> bool:
        """
        Determine if a learning cycle should run based on available work and backoff.
        
        This prevents the engine from blindly running every cycle when there's no
        work available, reducing resource waste.
        
        Returns:
            True if cycle should run, False if should skip
        """
        with self.lock:
            current_time = time.time()
            
            # Check for explicit wake triggers
            if self.has_wake_triggers():
                self.reset_backoff()
                logger.debug("[CuriosityEngine] Wake trigger detected, running cycle")
                return True
            
            # Progressive backoff after empty cycles
            if self._empty_cycles > 0:
                backoff_time = self._base_cycle_interval * (
                    2 ** min(self._empty_cycles, self._max_backoff_exponent)
                )
                time_since_last = current_time - self._last_cycle_time
                
                if time_since_last < backoff_time:
                    logger.debug(
                        f"[CuriosityEngine] Backoff active: {time_since_last:.1f}s < "
                        f"{backoff_time:.1f}s (empty_cycles={self._empty_cycles})"
                    )
                    return False
            
            # Check for potential gaps before running
            if not self.gap_analyzer.has_potential_gaps():
                self._empty_cycles += 1
                logger.debug(
                    f"[CuriosityEngine] No potential gaps, incrementing empty_cycles to "
                    f"{self._empty_cycles}"
                )
                return False
            
            return True
    
    def has_wake_triggers(self) -> bool:
        """
        Check if there are triggers that should wake the engine from backoff.
        
        Wake triggers include:
        - New outcome data from the outcome bridge
        - Error rate spikes
        - Tool weight drift
        - External gaps injected
        
        Returns:
            True if there's a reason to wake up and run a cycle
        """
        try:
            # Wake on external gaps injected
            with self._external_gaps_lock:
                if len(self._external_gaps) > 0:
                    logger.debug(
                        f"[CuriosityEngine] Wake trigger: {len(self._external_gaps)} external gaps"
                    )
                    return True
            
            # Wake on new outcome data
            try:
                from .outcome_bridge import get_recent_count
                recent_outcomes = get_recent_count(minutes=5)
                if recent_outcomes > self._last_outcome_count:
                    logger.debug(
                        f"[CuriosityEngine] Wake trigger: new outcomes "
                        f"({recent_outcomes} > {self._last_outcome_count})"
                    )
                    self._last_outcome_count = recent_outcomes
                    return True
            except ImportError:
                pass  # outcome_bridge not available
            except Exception as e:
                logger.debug(f"[CuriosityEngine] Could not check outcome bridge: {e}")
            
            # Wake on error rate spike
            if self._detect_error_spike():
                logger.debug("[CuriosityEngine] Wake trigger: error spike detected")
                return True
            
            # Wake on stagnation (force exploration)
            if self._cycles_without_experiments >= self.EXPLORATION_FORCE_THRESHOLD:
                logger.debug(
                    f"[CuriosityEngine] Wake trigger: stagnation "
                    f"({self._cycles_without_experiments} cycles without experiments)"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"[CuriosityEngine] Error checking wake triggers: {e}")
            return True  # Default to running if we can't check
    
    def _detect_error_spike(self) -> bool:
        """
        Detect if there's been a spike in error rates that warrants investigation.
        
        Returns:
            True if error rate has spiked significantly
        """
        try:
            stats = self.gap_analyzer.get_statistics()
            
            # Check if we have enough data
            total_failures = (
                stats.get("decomposition_failures", 0) +
                stats.get("prediction_errors", 0) +
                stats.get("transfer_failures", 0)
            )
            
            # Consider it a spike if we have 10+ recent failures
            if total_failures >= 10:
                return True
            
            return False
        except Exception as e:
            logger.debug(f"[CuriosityEngine] Error detecting error spike: {e}")
            return False
    
    def reset_backoff(self) -> None:
        """
        Reset the backoff state after a successful wake trigger or cycle.
        """
        with self.lock:
            self._empty_cycles = 0
            self._backoff_multiplier = 1
            self._exploration_temperature = 1.0
            logger.debug("[CuriosityEngine] Backoff reset")
    
    def record_cycle_result(self, experiments_run: int, success_count: int) -> None:
        """
        Record the result of a learning cycle for backoff management.
        
        Args:
            experiments_run: Number of experiments that were run
            success_count: Number of successful experiments
        """
        with self.lock:
            current_time = time.time()
            self._last_cycle_time = current_time
            
            if experiments_run > 0:
                # Successful cycle with experiments
                self._last_successful_cycle = current_time
                self._empty_cycles = 0
                self._cycles_without_experiments = 0
                self._exploration_temperature = 1.0
                logger.debug(
                    f"[CuriosityEngine] Cycle complete: {experiments_run} experiments, "
                    f"{success_count} successes - backoff reset"
                )
            else:
                # Empty cycle - increment counters
                self._empty_cycles += 1
                self._cycles_without_experiments += 1
                
                # Increase exploration temperature for stagnation
                if self._cycles_without_experiments > 10:
                    self._exploration_temperature = min(
                        2.0, 
                        self._exploration_temperature * 1.1
                    )
                
                logger.info(
                    f"[CuriosityEngine] Empty cycle - empty_cycles={self._empty_cycles}, "
                    f"total_without_experiments={self._cycles_without_experiments}, "
                    f"exploration_temp={self._exploration_temperature:.2f}"
                )

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
        """Identify knowledge gaps using selected strategy - REFACTORED
        
        Note: Now filters gaps to prevent unbounded accumulation by:
        1. Removing already resolved gaps
        2. Enforcing cooldown period per gap type
        3. Limiting gaps per type (MAX_GAPS_PER_TYPE)
        4. Clearing the gap graph at the start to prevent accumulation across cycles
        """

        try:
            if strategy is None:
                strategy = self.select_exploration_strategy()

            # FIX: Clear gap graph at start of each gap identification cycle
            # This prevents gaps from accumulating across learning cycles,
            # which was causing "Found 112 gaps: ['exploration', ...]" logs
            self.gap_graph.clear()

            raw_gaps = []
            
            # FIRST: Consume any external gaps (from OutcomeBridge, etc.)
            # This ensures gaps from external systems are always included
            external_gaps = self._consume_external_gaps()
            if external_gaps:
                raw_gaps.extend(external_gaps)
                logger.info(
                    f"[CuriosityEngine] Including {len(external_gaps)} external gaps"
                )

            # EXAMINE & SELECT: Route based on strategy
            if strategy in ["gap_driven", "comprehensive", "balanced"]:
                if strategy == "comprehensive":
                    decomposition_gaps = (
                        self.gap_analyzer.analyze_decomposition_failures()
                    )
                    prediction_gaps = self.gap_analyzer.analyze_prediction_errors()
                    transfer_gaps = self.gap_analyzer.analyze_transfer_failures()
                    latent_gaps = self.gap_analyzer.detect_latent_gaps()
                    # Note: Include outcome bridge gaps for cross-process data
                    outcome_gaps = self.gap_analyzer.analyze_from_outcome_bridge(minutes=60)

                    raw_gaps.extend(decomposition_gaps)
                    raw_gaps.extend(prediction_gaps)
                    raw_gaps.extend(transfer_gaps)
                    raw_gaps.extend(latent_gaps)
                    raw_gaps.extend(outcome_gaps)
                elif strategy == "gap_driven":
                    decomposition_gaps = (
                        self.gap_analyzer.analyze_decomposition_failures()
                    )
                    # Note: Include outcome bridge gaps for cross-process data
                    outcome_gaps = self.gap_analyzer.analyze_from_outcome_bridge(minutes=60)
                    raw_gaps.extend(decomposition_gaps[:5])
                    raw_gaps.extend(outcome_gaps[:3])
                else:  # balanced
                    decomposition_gaps = (
                        self.gap_analyzer.analyze_decomposition_failures()
                    )
                    prediction_gaps = self.gap_analyzer.analyze_prediction_errors()
                    # Note: Include outcome bridge gaps for cross-process data
                    # This enables the subprocess to read query outcomes from the main process
                    outcome_gaps = self.gap_analyzer.analyze_from_outcome_bridge(minutes=60)

                    raw_gaps.extend(decomposition_gaps[:3])
                    raw_gaps.extend(prediction_gaps[:3])
                    raw_gaps.extend(outcome_gaps[:3])
            elif strategy == "minimal":
                all_gaps = self.gap_analyzer.get_all_gaps()
                raw_gaps.extend(all_gaps[:3])
            elif strategy == "efficient":
                all_gaps = self.gap_analyzer.get_all_gaps()
                raw_gaps.extend([g for g in all_gaps if g.estimated_cost < 20][:5])

            # Note: Filter gaps to prevent accumulation
            gaps = self._filter_gaps_with_resolution(raw_gaps)
            
            # FIX OPERATIONAL: Inject synthetic gaps if no real gaps found
            # This prevents the curiosity engine from going dormant
            if len(gaps) == 0:
                logger.info(
                    "[CuriosityEngine] FIX OPERATIONAL: No real gaps found, "
                    "injecting synthetic gaps to maintain learning"
                )
                gaps = self.inject_synthetic_gaps()

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
            
            # NOTIFY: Send gaps to SelfImprovementDrive for priority boosting
            # This connects gap detection to improvement objective prioritization
            if gaps:
                self._notify_gaps_detected(gaps)

            return gaps
        except Exception as e:
            logger.error("Error identifying gaps: %s", e)
            return []

    def inject_synthetic_gaps(self) -> List[KnowledgeGap]:
        """
        FIX OPERATIONAL: Inject synthetic knowledge gaps to prevent prolonged dormancy.
        
        When the system can't find real gaps and enters dormant mode, this generates
        synthetic gaps to maintain continuous learning and exploration.
        
        BUG FIX #3: Check suppressed gaps to prevent phantom resolution loop.
        
        Returns:
            List of synthetic knowledge gaps for exploration
        """
        synthetic_gaps = []
        
        try:
            # Generate synthetic gaps for different domains
            synthetic_domains = [
                ("reasoning_efficiency", "Optimize reasoning pathway selection"),
                ("error_pattern_analysis", "Analyze error patterns in recent queries"),
                ("knowledge_consolidation", "Consolidate fragmented knowledge"),
                ("unexplored_domains", "Explore underutilized reasoning capabilities"),
            ]
            
            current_time = time.time()
            
            for domain, description in synthetic_domains:
                # BUG FIX #3: Check if this gap type is suppressed (phantom resolution)
                gap_type = "exploration"
                key = f"{gap_type}:{domain}"
                
                if key in self._suppressed_gaps:
                    if current_time < self._suppressed_gaps[key]:
                        # Still suppressed, skip this gap
                        suppression_remaining = int((self._suppressed_gaps[key] - current_time) / 60)
                        logger.debug(
                            f"[CuriosityEngine] Skipping suppressed gap {key} "
                            f"({suppression_remaining} minutes remaining)"
                        )
                        continue
                    else:
                        # Suppression expired, remove from suppressed list
                        del self._suppressed_gaps[key]
                
                gap = KnowledgeGap(
                    type=gap_type,
                    domain=domain,
                    priority=0.4,
                    estimated_cost=10.0,
                    metadata={"description": f"Synthetic gap: {description}", "expected_reward": 5.0},
                )
                synthetic_gaps.append(gap)
            
            if synthetic_gaps:
                logger.info(
                    f"[CuriosityEngine] FIX OPERATIONAL: Injected {len(synthetic_gaps)} "
                    f"synthetic gaps to prevent dormancy"
                )
            else:
                logger.info(
                    "[CuriosityEngine] All synthetic gaps are suppressed due to phantom resolutions"
                )
            
            return synthetic_gaps
        except Exception as e:
            logger.error(f"Error injecting synthetic gaps: {e}")
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

                # Note: Add to queue atomically
                for priority in priorities:
                    self.learning_priorities.put(priority)

                return priorities
            except Exception as e:
                logger.error("Error prioritizing gaps: %s", e)
                return []

    def generate_targeted_experiments(self, gap: KnowledgeGap) -> List[Experiment]:
        """Generate experiments targeted at specific gap - REFACTORED
        
        Note: Added support for outcome bridge gap types (slow_routing,
        complex_query_handling, high_error_rate, routing_variance) which
        were previously falling through to empty experiment list.
        
        FIX Defect #3: Phantom Gap Prevention
        Checks persistent attempt counter before generating experiments.
        Skips gaps that have been attempted 3+ times to prevent infinite loops.
        """

        try:
            # FIX Defect #3: Check persistent attempt counter to prevent phantom gaps
            # This prevents the same gap from being repeatedly attempted when the
            # underlying issue hasn't been truly resolved
            gap_attempts = _persistent_get_gap_attempts(gap.id)
            MAX_ATTEMPTS = 3
            
            if gap_attempts >= MAX_ATTEMPTS:
                logger.info(
                    f"[CuriosityEngine] Skipping gap '{gap.id}' (type={gap.type}) - "
                    f"already attempted {gap_attempts} times. "
                    f"This prevents phantom gap infinite loops. "
                    f"Gap will be re-evaluated after cooldown period."
                )
                return []
            
            # Log when approaching the limit
            if gap_attempts > 0:
                logger.debug(
                    f"[CuriosityEngine] Gap '{gap.id}' attempt #{gap_attempts + 1}/{MAX_ATTEMPTS}"
                )
            
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
            # Note: Handle outcome bridge gap types that were returning empty lists
            elif gap.type in ("slow_routing", "routing_variance"):
                # Routing issues are treated as decomposition problems
                # (optimizing the query routing pipeline)
                experiments = (
                    self.experiment_generator.generate_decomposition_experiment(
                        gap, complexity=0.4
                    )
                )
            elif gap.type == "complex_query_handling":
                # Complex query handling is treated as a decomposition problem
                # (breaking down complex queries into manageable parts)
                experiments = (
                    self.experiment_generator.generate_decomposition_experiment(
                        gap, complexity=0.6
                    )
                )
            elif gap.type == "high_error_rate":
                # High error rates suggest causal analysis is needed
                # (understanding why queries are failing)
                experiments = self.experiment_generator.generate_causal_experiment(gap)
            elif gap.type == "exploration":
                # Exploration gaps are synthetic gaps generated for continuous learning
                # Use lower complexity decomposition experiments to explore efficiently
                experiments = (
                    self.experiment_generator.generate_decomposition_experiment(
                        gap, complexity=0.3
                    )
                )
            else:
                # Note: Instead of returning empty list, generate generic experiments
                # This ensures gaps are never silently ignored
                logger.warning(
                    f"[CuriosityEngine] Unhandled gap type '{gap.type}', "
                    f"generating generic experiment"
                )
                experiments = self.experiment_generator.generate_for_gap(gap)

            # Filter by budget
            affordable_experiments = []
            remaining_budget = available_budget

            for exp in experiments:
                if self.exploration_budget.can_afford(exp.complexity * 10):
                    affordable_experiments.append(exp)
                    remaining_budget -= exp.complexity * 10
            
            # FIX Defect #3: Increment persistent attempt counter when experiments are created
            # This ensures the counter survives subprocess restarts
            if affordable_experiments:
                try:
                    new_attempts = _persistent_increment_gap_attempts(gap.id)
                    logger.debug(
                        f"[CuriosityEngine] Generated {len(affordable_experiments)} experiments "
                        f"for gap '{gap.id}' (attempt #{new_attempts})"
                    )
                except Exception as e:
                    logger.warning(
                        f"[CuriosityEngine] Failed to increment gap attempts in persistent store: {e}"
                    )

            return affordable_experiments
        except Exception as e:
            logger.error("Error generating experiments: %s", e)
            return []

    def run_experiment_sandboxed(self, experiment: Experiment) -> ExperimentResult:
        """Run experiment in sandboxed environment - DELEGATED
        
        Note: Now tracks experiment attempts per gap and marks gaps
        as resolved when experiments succeed or after multiple attempts.
        """

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
            
            # Note: Track experiment attempts and mark gaps resolved
            if hasattr(experiment, 'gap') and experiment.gap:
                gap = experiment.gap
                gap_key = self._gap_key(gap)
                
                # Track attempts
                self._gap_attempts[gap_key] = self._gap_attempts.get(gap_key, 0) + 1
                
                # Mark resolved if experiment succeeded with reasonable confidence
                if result.success:
                    self.mark_gap_resolved(gap, success=True)
                
                # Note: Use configurable threshold - don't give up too quickly
                # Default increased from 3 to 10 attempts before giving up
                elif self._gap_attempts[gap_key] >= self.GAP_GIVEUP_THRESHOLD:
                    logger.warning(
                        f"[CuriosityEngine] Gap {gap_key} giving up after {self._gap_attempts[gap_key]} attempts - "
                        f"marking as deferred, not resolved"
                    )
                    # Mark as "deferred" not "resolved" - it wasn't actually fixed
                    self.mark_gap_resolved(gap, success=False)

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
                "Updated from %d experiment results (learning rate: %.2f)",
                len(results),
                self.learning_rate,
            )
        except Exception as e:
            logger.error("Error updating from results: %s", e)

    def run_learning_cycle(self, max_experiments: int = 10) -> Dict[str, Any]:
        """Run one cycle of curiosity-driven learning - REFACTORED
        
        Note: Added diagnostic logging to explain why no gaps are found
        and why experiments might not be generated.
        """

        with self.lock:
            try:
                cycle_start = time.time()

                # EXAMINE: Select exploration strategy
                strategy = self.select_exploration_strategy()

                # Identify gaps
                gaps = self.identify_gaps_with_cycle_detection()
                
                # Note: Log diagnostic info if no gaps found
                if len(gaps) == 0:
                    ingestion_stats = self.get_ingestion_stats()
                    logger.debug(
                        f"[CuriosityEngine] No gaps found. "
                        f"Ingestion stats: queries={ingestion_stats['queries_ingested']}, "
                        f"failures={ingestion_stats['failures_ingested']}. "
                        f"Strategy: {strategy}. "
                        f"Hint: Call ingest_query_result() from query pipeline to feed data."
                    )
                else:
                    # Note: Log gap types found for debugging
                    gap_types = [g.type for g in gaps]
                    logger.info(
                        f"[CuriosityEngine] Found {len(gaps)} gaps: {gap_types}"
                    )

                # SELECT: Prioritize gaps
                priorities = self.prioritize_gaps(gaps[:max_experiments])
                
                # Note: Log if priorities were created
                logger.debug(
                    f"[CuriosityEngine] Created {len(priorities)} priorities "
                    f"for learning queue"
                )

                # APPLY: Run experiments
                results = []
                experiments_run = 0
                gaps_with_no_experiments = 0
                
                # Note: Track if this is a "cold start" cycle with no priorities
                # When no gaps/priorities exist, generate bootstrap experiments to 
                # kickstart the learning system
                had_initial_priorities = not self.learning_priorities.empty()

                while experiments_run < max_experiments:
                    try:
                        # Note: Use non-blocking get with timeout
                        priority = self.learning_priorities.get(block=False)
                    except Empty:
                        break

                    # Generate experiments for gap
                    experiments = self.generate_targeted_experiments(priority.gap)
                    
                    # Note: Log when no experiments are generated for a gap
                    if not experiments:
                        gaps_with_no_experiments += 1
                        logger.warning(
                            f"[CuriosityEngine] No experiments generated for gap "
                            f"type='{priority.gap.type}', id='{priority.gap.id}'"
                        )
                        continue

                    for exp in experiments[:2]:  # Limit per gap
                        if experiments_run >= max_experiments:
                            break

                        # Run experiment
                        result = self.run_experiment_sandboxed(exp)
                        results.append(result)
                        experiments_run += 1
                        
                        # FIX Bug #2: Persist experiment count to SQLite
                        # This prevents false cold-start detection in subprocesses
                        try:
                            _persistent_increment_experiment_count("total_experiments", 1)
                        except Exception:
                            pass
                
                # FIX Bug #2: Check persistent experiment count for cold start detection
                # Subprocesses should check SQLite to see if any experiments were ever run
                try:
                    persistent_experiment_count = _persistent_get_experiment_count("total_experiments")
                except Exception:
                    persistent_experiment_count = 0
                
                # Note: Generate bootstrap experiments when:
                # 1. No experiments ran at all in THIS cycle AND no persistent history, OR
                # 2. Very few experiments ran and we had no initial priorities
                # FIX Bug #2: Also consider persistent experiment count to avoid false cold start
                is_true_cold_start = (
                    experiments_run == 0 and 
                    persistent_experiment_count == 0
                )
                needs_bootstrap = (
                    is_true_cold_start or 
                    (experiments_run < 2 and not had_initial_priorities and persistent_experiment_count < 5)
                )
                if needs_bootstrap:
                    remaining_slots = max_experiments - experiments_run
                    if remaining_slots > 0:
                        logger.info(
                            f"[CuriosityEngine] Cold start detected (ran {experiments_run}/{max_experiments}, "
                            f"persistent={persistent_experiment_count}) - generating {remaining_slots} bootstrap experiments"
                        )
                        bootstrap_experiments = self._generate_bootstrap_experiments(remaining_slots)
                        for exp in bootstrap_experiments[:remaining_slots]:
                            result = self.run_experiment_sandboxed(exp)
                            results.append(result)
                            experiments_run += 1
                            # Persist bootstrap experiments too
                            try:
                                _persistent_increment_experiment_count("total_experiments", 1)
                            except Exception:
                                pass

                # Note: Log summary of gaps that generated no experiments
                if gaps_with_no_experiments > 0:
                    logger.warning(
                        f"[CuriosityEngine] {gaps_with_no_experiments} gaps "
                        f"generated no experiments"
                    )

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
    
    def _generate_bootstrap_experiments(self, max_experiments: int = 3) -> List[Experiment]:
        """
        Generate bootstrap experiments for cold-start learning cycles.
        
        Note: When no gaps are found (no query data ingested, all gaps
        in cooldown, or at type limits), generate synthetic exploratory experiments
        to ensure the learning system makes progress.
        
        These experiments focus on:
        1. Self-diagnostic experiments - test the system's own capabilities
        2. Baseline performance experiments - establish performance baselines
        3. Exploratory experiments - discover what capabilities are available
        
        Args:
            max_experiments: Maximum number of bootstrap experiments to generate
            
        Returns:
            List of bootstrap Experiment objects
        """
        from .experiment_generator import Constraint, Experiment, ExperimentType
        # Note: KnowledgeGap is already imported at module level from .gap_analyzer
        
        experiments = []
        
        try:
            # Generate synthetic gaps for bootstrap experiments
            bootstrap_gap_configs = [
                {
                    "type": "self_diagnostic",
                    "domain": "system",
                    "description": "Bootstrap: System self-diagnostic",
                    "complexity": 0.3,
                    "experiment_type": ExperimentType.VALIDATION,
                    "parameters": {
                        "test_type": "self_diagnostic",
                        "components": ["memory", "reasoning", "learning"],
                        "depth": "shallow",
                    },
                },
                {
                    "type": "baseline",
                    "domain": "performance", 
                    "description": "Bootstrap: Establish performance baseline",
                    "complexity": 0.4,
                    "experiment_type": ExperimentType.EXPLORATORY,
                    "parameters": {
                        "test_type": "baseline",
                        "metrics": ["latency", "throughput", "accuracy"],
                        "sample_size": 10,
                    },
                },
                {
                    "type": "capability_probe",
                    "domain": "reasoning",
                    "description": "Bootstrap: Probe available capabilities",
                    "complexity": 0.3,
                    "experiment_type": ExperimentType.EXPLORATORY,
                    "parameters": {
                        "test_type": "capability_probe",
                        "probe_depth": "surface",
                        "capabilities": ["decomposition", "causal", "transfer"],
                    },
                },
            ]
            
            for i, config in enumerate(bootstrap_gap_configs[:max_experiments]):
                # Create synthetic gap
                # Note: Use correct KnowledgeGap parameters:
                # - Use 'priority' instead of 'severity' (severity is not a valid parameter)
                # - Store 'description' and 'related_patterns' in metadata (not top-level params)
                gap = KnowledgeGap(
                    type=config["type"],
                    domain=config["domain"],
                    priority=0.3,  # Low priority for bootstrap experiments
                    estimated_cost=config["complexity"] * 10,
                    gap_id=f"bootstrap_{config['type']}_{i}",
                    complexity=config["complexity"],
                    metadata={
                        "bootstrap": True,
                        "auto_generated": True,
                        "description": config["description"],
                        "related_patterns": [],
                    },
                )
                
                # Create experiment
                experiment = Experiment(
                    gap=gap,
                    complexity=config["complexity"],
                    timeout=self.BOOTSTRAP_EXPERIMENT_TIMEOUT_SECONDS,
                    success_criteria={
                        "completion": True,
                        "no_errors": True,
                        "min_data_points": 1,
                    },
                    safety_constraints=[
                        Constraint(
                            "bootstrap_timeout",
                            "time",
                            self.BOOTSTRAP_EXPERIMENT_TIMEOUT_SECONDS,
                            action="abort"
                        ),
                        Constraint(
                            "bootstrap_memory",
                            "memory",
                            self.BOOTSTRAP_MEMORY_LIMIT_BYTES,
                            action="abort"
                        ),
                    ],
                    experiment_type=config["experiment_type"],
                    parameters=config["parameters"],
                    metadata={
                        "bootstrap": True,
                        "auto_generated": True,
                        "purpose": "cold_start_initialization",
                    },
                )
                experiments.append(experiment)
            
            logger.info(
                f"[CuriosityEngine] Generated {len(experiments)} bootstrap experiments "
                f"for cold-start learning cycle"
            )
            
        except Exception as e:
            logger.error(f"[CuriosityEngine] Error generating bootstrap experiments: {e}")
        
        return experiments

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


# =============================================================================
# Factory function for CuriosityEngine singleton access
# =============================================================================

_curiosity_engine_instance = None
_curiosity_engine_lock = threading.Lock()


def get_curiosity_engine(
    knowledge=None, decomposer=None, world_model=None
) -> CuriosityEngine:
    """
    Factory function to get the singleton CuriosityEngine instance.
    
    This ensures only one CuriosityEngine exists application-wide,
    preventing the dual instance issue (PIDs 247 and 215 in logs).
    
    Args:
        knowledge: Optional knowledge crystallizer instance
        decomposer: Optional problem decomposer instance  
        world_model: Optional world model instance
        
    Returns:
        The singleton CuriosityEngine instance
    """
    global _curiosity_engine_instance
    
    if _curiosity_engine_instance is None:
        with _curiosity_engine_lock:
            # Double-check locking pattern
            if _curiosity_engine_instance is None:
                _curiosity_engine_instance = CuriosityEngine(
                    knowledge=knowledge,
                    decomposer=decomposer, 
                    world_model=world_model
                )
    elif knowledge is not None or decomposer is not None or world_model is not None:
        # Update dependencies on existing instance
        if knowledge is not None:
            _curiosity_engine_instance.knowledge = knowledge
        if decomposer is not None:
            _curiosity_engine_instance.decomposer = decomposer
        if world_model is not None:
            _curiosity_engine_instance.world_model = world_model
            
    return _curiosity_engine_instance

