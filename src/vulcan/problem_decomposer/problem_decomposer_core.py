"""
problem_decomposer_core.py - Main problem decomposition orchestrator
Part of the VULCAN-AGI system

Integrated with comprehensive safety validation.
"""

import hashlib
import json
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import safety validator
try:
    from ..safety.safety_types import SafetyConfig
    from ..safety.safety_validator import EnhancedSafetyValidator

    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    SAFETY_VALIDATOR_AVAILABLE = False
    logging.warning(
        "safety_validator not available, problem_decomposer operating without safety checks"
    )
    EnhancedSafetyValidator = None
    SafetyConfig = None

# Optional import with fallback
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, graph features will be limited")

    # Mock NetworkX for basic functionality
    class MockGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = []
            self._adjacency = defaultdict(list)

        def add_node(self, node_id, **attrs):
            self._nodes[node_id] = attrs

        def add_edge(self, source, target, **attrs):
            self._edges.append((source, target, attrs))
            self._adjacency[source].append(target)

        def nodes(self):
            return self._nodes.keys()

        def edges(self):
            return self._edges

        def number_of_nodes(self):
            return len(self._nodes)

        def number_of_edges(self):
            return len(self._edges)

        def degree(self, node=None):
            if node is not None:
                return len(self._adjacency.get(node, []))
            return [(n, len(self._adjacency[n])) for n in self._nodes]

        def in_degree(self, node):
            count = 0
            for edges in self._adjacency.values():
                if node in edges:
                    count += 1
            return count

        def subgraph(self, nodes):
            sub = MockGraph()
            for node in nodes:
                if node in self._nodes:
                    sub.add_node(node, **self._nodes[node])
            for source, target, attrs in self._edges:
                if source in nodes and target in nodes:
                    sub.add_edge(source, target, **attrs)
            return sub

    class MockNX:
        Graph = MockGraph
        DiGraph = MockGraph

        @staticmethod
        def is_directed_acyclic_graph(graph):
            # Simplified - would need real cycle detection
            return True

        @staticmethod
        def is_weakly_connected(graph):
            # Simplified - assume connected
            return True

        @staticmethod
        def density(graph):
            n = graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else 0
            if n <= 1:
                return 0
            e = graph.number_of_edges() if hasattr(graph, "number_of_edges") else 0
            return e / (n * (n - 1))

        @staticmethod
        def dag_longest_path_length(graph):
            # Simplified
            return 1

        @staticmethod
        def descendants(graph, node):
            # Simplified - return all other nodes
            return set(graph.nodes()) - {node} if hasattr(graph, "nodes") else set()

        @staticmethod
        def number_of_nodes(graph):
            return graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else 0

        @staticmethod
        def weakly_connected_components(graph):
            if hasattr(graph, "nodes"):
                return [set(graph.nodes())]
            return []

    nx = MockNX()

# Import from other module files with error handling
try:
    from .adaptive_thresholds import AdaptiveThresholds
    from .decomposition_library import StratifiedDecompositionLibrary
    from .decomposition_strategies import DecompositionStrategy
    from .fallback_chain import FallbackChain
    from .problem_executor import ProblemExecutor
except ImportError as e:
    logging.warning(f"Failed to import module components: {e}")
    # Provide basic fallback classes

    class StratifiedDecompositionLibrary:
        def __init__(self, *args, **kwargs):
            self.patterns = {}
            self.principles = {}

        def get_strategy_by_type(self, strategy_type):
            return None

        def get_strategy(self, strategy_name):
            return None

    class AdaptiveThresholds:
        def __init__(self, *args, **kwargs):
            self.thresholds = {}

        def get_confidence_threshold(self):
            return 0.5

        def update_from_outcome(self, *args, **kwargs):
            pass

    class FallbackChain:
        def __init__(self, *args, **kwargs):
            self.strategies = []

        def generate_fallback_plans(self, problem_graph):
            return []

    class DecompositionStrategy:
        def __init__(self, *args, **kwargs):
            self.name = "BaseStrategy"
            self.strategy_type = "base"

        def decompose(self, problem_graph):
            return []

        def is_parallelizable(self):
            return False

        def is_deterministic(self):
            return True

    class ProblemExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def execute_plan(self, problem_graph, plan):
            return ExecutionOutcome(
                success=False, execution_time=0.0, errors=["Executor not available"]
            )

        def execute_and_validate(self, problem_graph, plan):
            outcome = self.execute_plan(problem_graph, plan)
            return outcome, {"validated": False}


logger = logging.getLogger(__name__)


class DecompositionMode(Enum):
    """Modes of decomposition"""

    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    FALLBACK = "fallback"
    LEARNING = "learning"
    HYBRID = "hybrid"


class ProblemComplexity(Enum):
    """Problem complexity levels"""

    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    VERY_COMPLEX = 5


class DomainDataCategory(Enum):
    """Categories by data availability"""

    FREQUENT = "frequent"
    MEDIUM = "medium"
    RARE = "rare"
    NOVEL = "novel"


@dataclass
class ProblemSignature:
    """Signature characterizing a problem's structure"""

    has_hierarchy: bool
    has_temporal: bool
    has_cycles: bool
    has_constraints: bool
    node_count: int
    edge_density: float
    max_depth: int
    branching_factor: float
    complexity: float
    domain: str
    is_dag: bool
    is_connected: bool
    avg_degree: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "has_hierarchy": self.has_hierarchy,
            "has_temporal": self.has_temporal,
            "has_cycles": self.has_cycles,
            "has_constraints": self.has_constraints,
            "node_count": self.node_count,
            "edge_density": self.edge_density,
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "complexity": self.complexity,
            "domain": self.domain,
            "is_dag": self.is_dag,
            "is_connected": self.is_connected,
            "avg_degree": self.avg_degree,
        }


@dataclass
class DecompositionStep:
    """Single step in a decomposition plan"""

    step_id: str
    action_type: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: float = 1.0
    confidence: float = 0.5
    required_resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "description": self.description,
            "dependencies": self.dependencies,
            "estimated_complexity": self.estimated_complexity,
            "confidence": self.confidence,
            "required_resources": self.required_resources,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecompositionStep":
        """Create from dictionary"""
        return cls(
            step_id=data.get("step_id", ""),
            action_type=data.get("action_type", "process"),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            estimated_complexity=data.get("estimated_complexity", 1.0),
            confidence=data.get("confidence", 0.5),
            required_resources=data.get("required_resources", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProblemGraph:
    """Graph representation of a problem"""

    nodes: Dict[str, Any] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    root: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0

    def to_networkx(self):
        """Convert to NetworkX graph or mock"""
        if NETWORKX_AVAILABLE:
            G = nx.DiGraph()
        else:
            G = MockGraph()

        for node_id, node_data in self.nodes.items():
            G.add_node(node_id, **node_data)
        for source, target, edge_data in self.edges:
            G.add_edge(source, target, **edge_data)
        return G

    def get_signature(self) -> str:
        """Get unique signature for problem"""
        content = json.dumps(
            {
                "nodes": sorted(self.nodes.keys()),
                "edges": [(s, t) for s, t, _ in sorted(self.edges)],
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()


@dataclass
class DecompositionPlan:
    """Plan for decomposing a problem"""

    steps: List[Any] = field(default_factory=list)  # Can be DecompositionStep or dict
    strategy: Optional[DecompositionStrategy] = None
    estimated_complexity: float = 0.0
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Any):
        """Add a decomposition step (DecompositionStep or dict)"""
        if isinstance(step, dict):
            # Convert dict to DecompositionStep for consistency
            step = DecompositionStep.from_dict(step)
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Convert steps to dicts
        steps_dicts = []
        for step in self.steps:
            if isinstance(step, DecompositionStep):
                steps_dicts.append(step.to_dict())
            elif isinstance(step, dict):
                steps_dicts.append(step)
            else:
                steps_dicts.append({"description": str(step)})

        return {
            "steps": steps_dicts,
            "strategy": self.strategy.name if self.strategy else None,
            "estimated_complexity": self.estimated_complexity,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionOutcome:
    """Outcome of executing a decomposition plan"""

    success: bool
    execution_time: float
    sub_results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    solution: Any = None

    def get_success_rate(self) -> float:
        """Calculate success rate of sub-results"""
        if not self.sub_results:
            return 0.0
        successful = sum(1 for r in self.sub_results if r.get("success", False))
        return successful / len(self.sub_results)


@dataclass
class LearningGap:
    """Knowledge gap identified from failed decomposition"""

    gap_type: str
    problem_signature: str
    failure_reason: str
    missing_capability: Optional[str] = None
    suggested_strategies: List[str] = field(default_factory=list)
    priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Tracks decomposition performance"""

    def __init__(self):
        """Initialize performance tracker"""
        self.execution_history = deque(maxlen=1000)
        self.strategy_performance = defaultdict(lambda: {"success": 0, "failure": 0})
        self.timing_data = defaultdict(lambda: deque(maxlen=100))
        self.complexity_accuracy = deque(maxlen=100)

        # Size limits for unbounded dictionaries
        self.max_strategies = 1000

        # Thread safety
        self._lock = threading.RLock()

        logger.info("PerformanceTracker initialized")

    def record_execution(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ):
        """Record execution performance"""
        with self._lock:
            record = {
                "problem_signature": problem.get_signature(),
                "strategy": plan.strategy.name if plan.strategy else "unknown",
                "success": outcome.success,
                "execution_time": outcome.execution_time,
                "complexity": plan.estimated_complexity,
                "timestamp": time.time(),
            }

            self.execution_history.append(record)

            # Update strategy performance with bounds enforcement
            strategy_key = record["strategy"]

            # Enforce max strategies limit
            if len(self.strategy_performance) >= self.max_strategies:
                # Remove strategy with worst performance (lowest success rate)
                worst_strategy = None
                worst_rate = float("inf")

                for strat_name, stats in self.strategy_performance.items():
                    total = stats["success"] + stats["failure"]
                    if total > 0:
                        rate = stats["success"] / total
                        if rate < worst_rate:
                            worst_rate = rate
                            worst_strategy = strat_name

                if worst_strategy:
                    del self.strategy_performance[worst_strategy]
                    if worst_strategy in self.timing_data:
                        del self.timing_data[worst_strategy]

            # Update strategy performance
            if outcome.success:
                self.strategy_performance[strategy_key]["success"] += 1
            else:
                self.strategy_performance[strategy_key]["failure"] += 1

            # Track timing
            self.timing_data[strategy_key].append(outcome.execution_time)

    def get_strategy_success_rate(self, strategy_name: str) -> float:
        """Get success rate for a strategy"""
        with self._lock:
            stats = self.strategy_performance.get(
                strategy_name, {"success": 0, "failure": 0}
            )
            total = stats["success"] + stats["failure"]
            if total == 0:
                return 0.5  # Default
            return stats["success"] / total

    def get_average_execution_time(self, strategy_name: str) -> float:
        """Get average execution time for a strategy"""
        with self._lock:
            times = self.timing_data.get(strategy_name, [])
            if not times:
                return 30.0  # Default
            return np.mean(times)

    def update_complexity_accuracy(self, estimated: float, actual: float):
        """Update complexity estimation accuracy"""
        with self._lock:
            accuracy = 1.0 - abs(estimated - actual) / max(estimated, actual, 1.0)
            self.complexity_accuracy.append(accuracy)

    def get_complexity_estimation_accuracy(self) -> float:
        """Get average complexity estimation accuracy"""
        with self._lock:
            if not self.complexity_accuracy:
                return 0.5
            return np.mean(self.complexity_accuracy)


class StrategyProfiler:
    """Profiles strategy characteristics and performance"""

    def __init__(self):
        """Initialize strategy profiler"""
        self.strategy_profiles = {}
        self.domain_affinity = defaultdict(lambda: defaultdict(float))
        self.complexity_affinity = defaultdict(lambda: defaultdict(float))

        # Size limits for unbounded dictionaries
        self.max_strategies = 1000
        self.max_domains_per_strategy = 100

        logger.info("StrategyProfiler initialized")

    def profile_strategy(self, strategy: DecompositionStrategy) -> Dict[str, Any]:
        """Profile a decomposition strategy"""
        # Enforce strategy limit
        if len(self.strategy_profiles) >= self.max_strategies:
            # Remove oldest strategy profile
            if self.strategy_profiles:
                oldest_key = next(iter(self.strategy_profiles))
                del self.strategy_profiles[oldest_key]
                # Clean up related data
                if oldest_key in self.domain_affinity:
                    del self.domain_affinity[oldest_key]
                if oldest_key in self.complexity_affinity:
                    del self.complexity_affinity[oldest_key]

        profile = {
            "name": strategy.name if hasattr(strategy, "name") else "unknown",
            "type": strategy.strategy_type
            if hasattr(strategy, "strategy_type")
            else "unknown",
            "complexity_range": self._estimate_complexity_range(strategy),
            "domain_suitability": self._estimate_domain_suitability(strategy),
            "parallelizable": strategy.is_parallelizable()
            if hasattr(strategy, "is_parallelizable")
            else False,
            "deterministic": strategy.is_deterministic()
            if hasattr(strategy, "is_deterministic")
            else True,
            "resource_requirements": self._estimate_resource_requirements(strategy),
        }

        self.strategy_profiles[profile["name"]] = profile
        return profile

    def update_affinity(
        self, strategy_name: str, domain: str, complexity: float, success: bool
    ):
        """Update strategy affinity scores"""
        # Enforce domain limit per strategy
        if len(self.domain_affinity[strategy_name]) >= self.max_domains_per_strategy:
            # Remove domain with lowest affinity
            if self.domain_affinity[strategy_name]:
                worst_domain = min(
                    self.domain_affinity[strategy_name].items(), key=lambda x: x[1]
                )
                del self.domain_affinity[strategy_name][worst_domain[0]]

        # Update domain affinity
        if success:
            self.domain_affinity[strategy_name][domain] += 0.1
        else:
            self.domain_affinity[strategy_name][domain] -= 0.05

        # Enforce complexity limit per strategy
        if (
            len(self.complexity_affinity[strategy_name])
            >= self.max_domains_per_strategy
        ):
            # Remove complexity bucket with lowest affinity
            if self.complexity_affinity[strategy_name]:
                worst_bucket = min(
                    self.complexity_affinity[strategy_name].items(), key=lambda x: x[1]
                )
                del self.complexity_affinity[strategy_name][worst_bucket[0]]

        # Update complexity affinity
        complexity_bucket = int(complexity)  # Bucket by integer complexity
        if success:
            self.complexity_affinity[strategy_name][complexity_bucket] += 0.1
        else:
            self.complexity_affinity[strategy_name][complexity_bucket] -= 0.05

    def get_best_strategy_for_domain(self, domain: str) -> Optional[str]:
        """Get best strategy for a domain"""
        best_strategy = None
        best_score = -float("inf")

        for strategy_name, domain_scores in self.domain_affinity.items():
            score = domain_scores.get(domain, 0.0)
            if score > best_score:
                best_score = score
                best_strategy = strategy_name

        return best_strategy

    def _estimate_complexity_range(
        self, strategy: DecompositionStrategy
    ) -> Tuple[float, float]:
        """Estimate complexity range for strategy"""
        # Simple heuristic based on strategy type
        if hasattr(strategy, "max_depth"):
            return (1.0, float(strategy.max_depth))
        return (1.0, 5.0)  # Default range

    def _estimate_domain_suitability(
        self, strategy: DecompositionStrategy
    ) -> List[str]:
        """Estimate suitable domains for strategy"""
        # Based on strategy characteristics
        suitable = []
        if hasattr(strategy, "is_deterministic") and strategy.is_deterministic():
            suitable.extend(["optimization", "planning", "analysis"])
        if hasattr(strategy, "is_parallelizable") and strategy.is_parallelizable():
            suitable.extend(["search", "generation", "classification"])
        return suitable

    def _estimate_resource_requirements(
        self, strategy: DecompositionStrategy
    ) -> Dict[str, Any]:
        """Estimate resource requirements"""
        return {
            "memory": "medium",  # Could be calculated based on strategy
            "cpu": "high"
            if (hasattr(strategy, "is_parallelizable") and strategy.is_parallelizable())
            else "medium",
            "time": "variable",
        }


class ProblemDecomposer:
    """Main decomposition orchestrator - WITH SAFETY VALIDATION"""

    def __init__(
        self,
        semantic_bridge=None,
        vulcan_memory=None,
        validator=None,
        safety_config: Optional[Dict[str, Any]] = None,
        safety_validator=None,
    ):
        """
        Initialize problem decomposer - FIXED: Added safety_validator parameter

        Args:
            semantic_bridge: Semantic bridge component
            vulcan_memory: VULCAN memory system
            validator: Optional validator for solution validation
            safety_config: Optional safety configuration (deprecated, use safety_validator)
            safety_validator: Optional shared safety validator instance (preferred over safety_config)
        """
        self.semantic = semantic_bridge
        self.memory = vulcan_memory
        self.validator = validator

        # Initialize safety validator - prefer shared instance
        if safety_validator is not None:
            # Use provided shared instance (PREFERRED - prevents duplication)
            self.safety_validator = safety_validator
            logger.info("ProblemDecomposer: Using shared safety validator instance")
        elif SAFETY_VALIDATOR_AVAILABLE:
            # Fallback: try to get singleton, or create new instance
            try:
                from ..safety.safety_validator import initialize_all_safety_components
                # Try singleton first
                self.safety_validator = initialize_all_safety_components(
                    config=safety_config, reuse_existing=True
                )
                logger.info("ProblemDecomposer: Using singleton safety validator")
            except Exception as e:
                logger.debug("Could not get singleton safety validator: %s", e)
                # Last resort: create new instance (preserving complex test_mode logic)
                if isinstance(safety_config, dict) and safety_config:
                    # Handle test_mode - don't pass it directly to SafetyConfig
                    if "test_mode" in safety_config and len(safety_config) == 1:
                        # Just test_mode config
                        config_obj = SafetyConfig()
                        # Add default values needed by RollbackManager
                        config_obj.rollback_config = {
                            "test_mode": True,
                            "max_snapshots": 10,  # Small value for tests
                            "enable_storage": False,
                            "enable_workers": False,
                        }
                        self.safety_validator = EnhancedSafetyValidator(config_obj)
                    else:
                        # Complex config - filter test_mode before passing to from_dict
                        try:
                            safety_config_filtered = {
                                k: v for k, v in safety_config.items() if k != "test_mode"
                            }
                            config_obj = (
                                SafetyConfig.from_dict(safety_config_filtered)
                                if safety_config_filtered
                                else SafetyConfig()
                            )
                            # Add rollback_config with full config (including test_mode) and defaults
                            rollback_cfg = {
                                "max_snapshots": 10
                                if safety_config.get("test_mode")
                                else 100,
                                "enable_storage": not safety_config.get("test_mode", False),
                                "enable_workers": not safety_config.get("test_mode", False),
                            }
                            rollback_cfg.update(safety_config)
                            if (
                                not hasattr(config_obj, "rollback_config")
                                or config_obj.rollback_config is None
                            ):
                                config_obj.rollback_config = rollback_cfg
                            else:
                                config_obj.rollback_config.update(rollback_cfg)
                            self.safety_validator = EnhancedSafetyValidator(config_obj)
                        except Exception as err:
                            logger.error(f"Failed to create SafetyConfig: {err}")
                            config_obj = SafetyConfig()
                            config_obj.rollback_config = {
                                "test_mode": safety_config.get("test_mode", False),
                                "max_snapshots": 10,
                                "enable_storage": False,
                                "enable_workers": False,
                            }
                            config_obj.rollback_config.update(safety_config)
                            self.safety_validator = EnhancedSafetyValidator(config_obj)
                else:
                    self.safety_validator = EnhancedSafetyValidator()
                logger.warning("ProblemDecomposer: Created new safety validator instance (may cause duplication)")
        else:
            self.safety_validator = None
            logger.warning(
                "ProblemDecomposer: Safety validator not available - operating without safety checks"
            )

        # Core components
        self.library = StratifiedDecompositionLibrary()
        self.thresholds = AdaptiveThresholds()
        self.performance_tracker = PerformanceTracker()
        self.strategy_profiler = StrategyProfiler()
        self.fallback_chain = FallbackChain()

        # Problem executor for running decomposition plans - pass safety config
        self.executor = ProblemExecutor(
            validator=validator,
            semantic_bridge=semantic_bridge,
            safety_config=safety_config,
        )

        # Domain selector
        self.domain_selector = DomainSelector()

        # Cache - with thread safety
        self.decomposition_cache = {}
        self.signature_cache = {}
        self.cache_size = 100
        self._cache_lock = threading.RLock()

        # Statistics - use bounded collections
        self.total_decompositions = 0
        self.successful_decompositions = 0
        self.learning_gaps = deque(maxlen=100)
        self.safety_blocks = Counter()
        self.safety_corrections = Counter()

        # Prediction history for learning
        self.prediction_history = deque(maxlen=500)

        logger.info("ProblemDecomposer initialized with executor and safety validation")

    def decompose_and_execute(
        self, problem_graph: ProblemGraph, validate: bool = False
    ) -> Tuple[DecompositionPlan, ExecutionOutcome]:
        """
        Decompose and execute problem in one call - MAIN ENTRY POINT WITH SAFETY

        CRITICAL: This is the main entry point that executes generated plans.
        Safety validation is mandatory to prevent execution of unsafe decompositions.

        Args:
            problem_graph: Problem to decompose and solve
            validate: Whether to validate solution

        Returns:
            Tuple of (decomposition_plan, execution_outcome)
        """
        # SAFETY CRITICAL: Require safety validator for execution
        if self.safety_validator is None:
            raise RuntimeError(
                "SAFETY CRITICAL: decompose_and_execute executes generated plans. "
                "Must have safety_validator initialized."
            )

        logger.info(
            "Decomposing and executing problem %s", problem_graph.get_signature()[:8]
        )

        # SAFETY: Validate problem before decomposition
        problem_validation = self._validate_problem_safety(problem_graph)
        if not problem_validation["safe"]:
            logger.error("BLOCKED unsafe problem: %s", problem_validation["reason"])
            self.safety_blocks["problem"] += 1

            # Return failed outcome
            failed_plan = DecompositionPlan(
                steps=[],
                confidence=0.0,
                metadata={
                    "safety_blocked": True,
                    "reason": problem_validation["reason"],
                },
            )

            failed_outcome = ExecutionOutcome(
                success=False,
                execution_time=0.0,
                errors=[
                    f"Problem blocked by safety validator: {problem_validation['reason']}"
                ],
                metadata={"safety_blocked": True},
            )

            return failed_plan, failed_outcome

        # Step 1: Decompose problem into plan
        plan = self.decompose_novel_problem(problem_graph)

        # SAFETY: Validate plan before execution
        plan_validation = self._validate_plan_safety(plan, problem_graph)
        if not plan_validation["safe"]:
            logger.error("BLOCKED unsafe plan: %s", plan_validation["reason"])
            self.safety_blocks["plan"] += 1

            # Return failed outcome
            failed_outcome = ExecutionOutcome(
                success=False,
                execution_time=0.0,
                errors=[
                    f"Plan blocked by safety validator: {plan_validation['reason']}"
                ],
                metadata={"safety_blocked": True},
            )

            return plan, failed_outcome

        # Step 2: Execute plan to get solution (executor has its own safety checks)
        if validate and self.validator:
            outcome, validation_results = self.executor.execute_and_validate(
                problem_graph, plan
            )
            outcome.metadata["validation"] = validation_results
            logger.info(
                "Execution completed with validation: success=%s, validated=%s",
                outcome.success,
                validation_results.get("validated", False),
            )
        else:
            outcome = self.executor.execute_plan(problem_graph, plan)
            logger.info(
                "Execution completed: success=%s, time=%.2f",
                outcome.success,
                outcome.execution_time,
            )

        # SAFETY: Validate outcome before learning
        if self.safety_validator:
            outcome_validation = self._validate_outcome_safety(outcome)
            if not outcome_validation["safe"]:
                logger.warning(
                    "Unsafe outcome detected: %s", outcome_validation["reason"]
                )
                self.safety_corrections["outcome"] += 1
                # Apply corrections
                outcome = self._apply_outcome_corrections(outcome, outcome_validation)

        # Step 3: Learn from SAFE outcome only
        self.learn_from_execution(problem_graph, plan, outcome)

        return plan, outcome

    def decompose_novel_problem(self, problem_graph: ProblemGraph) -> DecompositionPlan:
        """
        Decompose a novel problem - WITH PREDICTIVE STRATEGY SELECTION

        Args:
            problem_graph: Graph representation of the problem

        Returns:
            Decomposition plan
        """
        self.total_decompositions += 1

        # Check cache
        signature_str = problem_graph.get_signature()

        with self._cache_lock:
            if signature_str in self.decomposition_cache:
                logger.debug("Using cached decomposition for problem %s", signature_str)
                return self.decomposition_cache[signature_str]

        # EXAMINE: Extract problem signature
        signature = self._extract_problem_signature(problem_graph)

        # EXAMINE: Analyze problem complexity
        complexity = self._analyze_complexity(problem_graph)
        problem_graph.complexity_score = complexity
        signature.complexity = complexity

        # SELECT: Predict best strategy WITHOUT executing
        strategy = self._predict_best_strategy(problem_graph, signature)

        if not strategy:
            logger.warning("No suitable strategy predicted, using fallback")
            return self.decompose_with_fallbacks(problem_graph)

        # APPLY: Create decomposition plan
        plan = self._create_decomposition_plan(problem_graph, strategy, complexity)

        # REMEMBER: Cache the plan with proper LRU eviction
        with self._cache_lock:
            if len(self.decomposition_cache) >= self.cache_size:
                # Remove oldest entry (FIFO approximation)
                oldest_key = next(iter(self.decomposition_cache))
                del self.decomposition_cache[oldest_key]
            self.decomposition_cache[signature_str] = plan

        # Track prediction for learning
        self.prediction_history.append(
            {
                "signature": signature,
                "predicted_strategy": strategy.name
                if hasattr(strategy, "name")
                else "unknown",
                "confidence": plan.confidence,
                "timestamp": time.time(),
            }
        )

        logger.info(
            "Created decomposition plan for novel problem (complexity: %.2f, strategy: %s)",
            complexity,
            strategy.name if hasattr(strategy, "name") else "unknown",
        )

        return plan

    def decompose_with_fallbacks(
        self, problem_graph: ProblemGraph
    ) -> DecompositionPlan:
        """
        Decompose with fallback strategies - ONLY IF PREDICTION UNCERTAIN

        Args:
            problem_graph: Graph representation of the problem

        Returns:
            Decomposition plan (possibly using fallbacks)
        """
        # EXAMINE: Extract signature
        signature = self._extract_problem_signature(problem_graph)

        # EXAMINE: Predict primary strategy
        primary_strategy = self._predict_best_strategy(problem_graph, signature)

        if not primary_strategy:
            logger.warning("No strategy predicted, generating fallback plans")
            fallback_plans = self.fallback_chain.generate_fallback_plans(problem_graph)

            if not fallback_plans:
                # Create minimal plan
                return DecompositionPlan(
                    steps=[],
                    strategy=None,
                    estimated_complexity=problem_graph.complexity_score,
                    confidence=0.1,
                    metadata={"fallback": "none_available"},
                )

            # SELECT: Pick best fallback
            return max(fallback_plans, key=lambda p: self._evaluate_plan(p))

        # APPLY: Create primary plan
        complexity = problem_graph.complexity_score or self._analyze_complexity(
            problem_graph
        )
        primary_plan = self._create_decomposition_plan(
            problem_graph, primary_strategy, complexity
        )

        # SELECT: Only try fallbacks if primary confidence is LOW
        confidence_threshold = self.thresholds.get_confidence_threshold()

        if primary_plan.confidence >= confidence_threshold:
            # Primary plan is confident enough
            logger.debug(
                "Using primary strategy (confidence: %.2f >= %.2f)",
                primary_plan.confidence,
                confidence_threshold,
            )
            return primary_plan

        # Primary is uncertain - generate ONE fallback as alternative
        logger.info(
            "Primary confidence low (%.2f), generating fallback alternative",
            primary_plan.confidence,
        )

        fallback_plans = self.fallback_chain.generate_fallback_plans(problem_graph)

        if not fallback_plans:
            # No fallbacks available, use primary
            return primary_plan

        # SELECT: Pick best between primary and single best fallback
        all_plans = [primary_plan] + fallback_plans[:1]  # Only consider best fallback
        best_plan = max(all_plans, key=lambda p: self._evaluate_plan(p))

        # Safely get strategy name and confidence for logging, handling both plan types
        strategy_name = "unknown"
        confidence_value = 0.0

        # Check if it's an ExecutionPlan (from fallback_chain)
        if hasattr(best_plan, "overall_confidence") and callable(
            best_plan.overall_confidence
        ):
            if hasattr(best_plan, "metadata") and "strategy" in best_plan.metadata:
                strategy_name = best_plan.metadata["strategy"]
            confidence_value = best_plan.overall_confidence()
        # Assume it's a DecompositionPlan otherwise
        else:
            if (
                hasattr(best_plan, "strategy")
                and best_plan.strategy
                and hasattr(best_plan.strategy, "name")
            ):
                strategy_name = best_plan.strategy.name
            if hasattr(best_plan, "confidence"):
                confidence_value = best_plan.confidence

        logger.info(
            "Selected strategy: %s (confidence: %.2f)", strategy_name, confidence_value
        )

        return best_plan

    def _validate_problem_safety(self, problem_graph: ProblemGraph) -> Dict[str, Any]:
        """Validate problem for safety"""
        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check node count
        if len(problem_graph.nodes) > 10000:
            violations.append(f"Excessive node count: {len(problem_graph.nodes)}")

        # Check edge count
        if len(problem_graph.edges) > 50000:
            violations.append(f"Excessive edge count: {len(problem_graph.edges)}")

        # Check complexity
        if problem_graph.complexity_score > 100:
            violations.append(f"Excessive complexity: {problem_graph.complexity_score}")

        # Validate metadata
        if hasattr(self.safety_validator, "validate_state"):
            metadata_validation = self.safety_validator.validate_state(
                problem_graph.metadata
            )
            if not metadata_validation["safe"]:
                violations.append(f"Unsafe metadata: {metadata_validation['reason']}")
        elif hasattr(self.safety_validator, "validate_action"):
            # Fallback to validate_action if validate_state doesn't exist
            try:
                metadata_validation = self.safety_validator.validate_action(
                    {"action": "validate_metadata", "metadata": problem_graph.metadata}
                )
                if not metadata_validation.safe:
                    violations.append(f"Unsafe metadata: {metadata_validation.reason}")
            except Exception as e:
                logger.error(f"Error during metadata validation: {e}", exc_info=True)

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _validate_plan_safety(
        self, plan: DecompositionPlan, problem_graph: ProblemGraph
    ) -> Dict[str, Any]:
        """Validate decomposition plan for safety"""
        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check plan confidence
        if plan.confidence < 0 or plan.confidence > 1:
            violations.append(f"Invalid plan confidence: {plan.confidence}")

        # Check number of steps
        if len(plan.steps) > 1000:
            violations.append(f"Excessive number of steps: {len(plan.steps)}")

        # Check estimated complexity
        if plan.estimated_complexity > 100:
            violations.append(
                f"Excessive estimated complexity: {plan.estimated_complexity}"
            )

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _validate_outcome_safety(self, outcome: ExecutionOutcome) -> Dict[str, Any]:
        """Validate execution outcome for safety"""
        if not self.safety_validator:
            return {"safe": True}

        violations = []

        # Check execution time
        if outcome.execution_time > 86400:  # More than 1 day
            violations.append(f"Excessive execution time: {outcome.execution_time}s")

        # Check error count
        if len(outcome.errors) > 1000:
            violations.append(f"Excessive errors: {len(outcome.errors)}")

        # Check metrics bounds
        for key, value in outcome.metrics.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    violations.append(f"Invalid metric {key}: {value}")

        if violations:
            return {"safe": False, "reason": "; ".join(violations)}

        return {"safe": True}

    def _apply_outcome_corrections(
        self, outcome: ExecutionOutcome, validation: Dict[str, Any]
    ) -> ExecutionOutcome:
        """Apply safety corrections to execution outcome"""

        # Clamp execution time
        outcome.execution_time = min(86400, outcome.execution_time)

        # Limit errors
        if len(outcome.errors) > 1000:
            outcome.errors = outcome.errors[:1000]
            outcome.errors.append("... (additional errors truncated for safety)")

        # Fix invalid metrics
        corrected_metrics = {}
        for key, value in outcome.metrics.items():
            if isinstance(value, (int, float)):
                if np.isfinite(value):
                    corrected_metrics[key] = value
                else:
                    corrected_metrics[key] = 0.0
            else:
                corrected_metrics[key] = value

        outcome.metrics = corrected_metrics
        outcome.metadata["safety_corrected"] = True
        outcome.metadata["correction_reason"] = validation["reason"]

        return outcome

    def _extract_problem_signature(
        self, problem_graph: ProblemGraph
    ) -> ProblemSignature:
        """
        Extract structural signature from problem - NEW METHOD

        Args:
            problem_graph: Problem to characterize

        Returns:
            Problem signature with key structural features
        """
        # Check cache
        sig_str = problem_graph.get_signature()

        with self._cache_lock:
            if sig_str in self.signature_cache:
                return self.signature_cache[sig_str]

        # Convert to graph
        G = problem_graph.to_networkx()

        # Extract structural features
        if hasattr(G, "number_of_nodes"):
            node_count = G.number_of_nodes()
        else:
            node_count = len(list(G.nodes())) if hasattr(G, "nodes") else 0

        if hasattr(G, "number_of_edges"):
            edge_count = G.number_of_edges()
        else:
            edge_count = len(list(G.edges())) if hasattr(G, "edges") else 0

        # Calculate edge density - FIX: proper division by zero handling
        if node_count > 1:
            max_edges = node_count * (node_count - 1)
            edge_density = edge_count / max_edges if max_edges > 0 else 0.0
        else:
            edge_density = 0.0

        # Check if DAG
        is_dag = False
        max_depth = 1
        if NETWORKX_AVAILABLE:
            try:
                is_dag = nx.is_directed_acyclic_graph(G)
                if is_dag:
                    max_depth = nx.dag_longest_path_length(G)
            except Exception:
                is_dag = False

        # Check connectivity
        is_connected = True
        if NETWORKX_AVAILABLE and node_count > 0:
            try:
                is_connected = nx.is_weakly_connected(G)
            except Exception:
                is_connected = True

        # Calculate average degree
        avg_degree = 0.0
        if node_count > 0 and hasattr(G, "degree"):
            try:
                degrees = [d for n, d in G.degree()]
                if degrees:
                    avg_degree = np.mean(degrees)
            except Exception:
                avg_degree = 0.0

        # Calculate branching factor
        branching_factor = 0.0
        if problem_graph.root and NETWORKX_AVAILABLE:
            try:
                descendants = nx.descendants(G, problem_graph.root)
                if node_count > 1:
                    branching_factor = len(descendants) / (node_count - 1)
            except Exception:
                branching_factor = 0.0

        # Detect structural patterns
        has_hierarchy = is_dag and max_depth > 2
        has_temporal = "temporal" in problem_graph.metadata.get("type", "").lower()
        has_cycles = not is_dag
        has_constraints = len(problem_graph.metadata.get("constraints", [])) > 0

        # Get domain
        domain = problem_graph.metadata.get("domain", "general")

        # Create signature
        signature = ProblemSignature(
            has_hierarchy=has_hierarchy,
            has_temporal=has_temporal,
            has_cycles=has_cycles,
            has_constraints=has_constraints,
            node_count=node_count,
            edge_density=edge_density,
            max_depth=max_depth,
            branching_factor=branching_factor,
            complexity=0.0,  # Will be filled by caller
            domain=domain,
            is_dag=is_dag,
            is_connected=is_connected,
            avg_degree=avg_degree,
            metadata=problem_graph.metadata.copy(),
        )

        # Cache signature with proper FIFO eviction
        with self._cache_lock:
            # Enforce cache size limit
            if len(self.signature_cache) >= self.cache_size:
                # Remove oldest entry (FIFO approximation)
                oldest_key = next(iter(self.signature_cache))
                del self.signature_cache[oldest_key]

            self.signature_cache[sig_str] = signature

        return signature

    def _predict_best_strategy(
        self, problem_graph: ProblemGraph, signature: ProblemSignature
    ) -> Optional[DecompositionStrategy]:
        """
        Predict best strategy WITHOUT executing - NEW METHOD

        Args:
            problem_graph: Problem to decompose
            signature: Problem signature

        Returns:
            Predicted best strategy
        """
        # EXAMINE: Check learned patterns first
        best_strategy_name = self.strategy_profiler.get_best_strategy_for_domain(
            signature.domain
        )

        if best_strategy_name:
            # Check if complexity matches
            if best_strategy_name in self.strategy_profiler.complexity_affinity:
                complexity_bucket = int(signature.complexity)
                complexity_affinity = self.strategy_profiler.complexity_affinity[
                    best_strategy_name
                ]

                if complexity_affinity.get(complexity_bucket, 0) > 0:
                    # Good match - use learned strategy
                    strategy = self.library.get_strategy(best_strategy_name)
                    if strategy:
                        logger.debug(
                            "Using learned strategy %s for domain %s",
                            best_strategy_name,
                            signature.domain,
                        )
                        return strategy

        # SELECT: Use rule-based prediction if no learned pattern
        predicted_type = None

        # Rule 1: Hierarchical structure
        if signature.has_hierarchy and signature.max_depth >= 3:
            predicted_type = "hierarchical_decomposition"
            logger.debug("Predicted hierarchical (depth=%d)", signature.max_depth)

        # Rule 2: Temporal sequence
        elif signature.has_temporal:
            predicted_type = "temporal_decomposition"
            logger.debug("Predicted temporal (temporal flag set)")

        # Rule 3: Heavy constraints
        elif (
            signature.has_constraints
            and len(signature.metadata.get("constraints", [])) > 5
        ):
            predicted_type = "constraint_based_decomposition"
            logger.debug(
                "Predicted constraint-based (%d constraints)",
                len(signature.metadata.get("constraints", [])),
            )

        # Rule 4: Simple/low complexity
        elif signature.complexity < 2.0 and signature.node_count < 10:
            predicted_type = "direct_decomposition"
            logger.debug(
                "Predicted direct (complexity=%.2f, nodes=%d)",
                signature.complexity,
                signature.node_count,
            )

        # Rule 5: Cyclic structure
        elif signature.has_cycles:
            predicted_type = "iterative_decomposition"
            logger.debug("Predicted iterative (has cycles)")

        # Rule 6: High branching
        elif signature.branching_factor > 0.5 and signature.node_count > 20:
            predicted_type = "parallel_decomposition"
            logger.debug(
                "Predicted parallel (branching=%.2f, nodes=%d)",
                signature.branching_factor,
                signature.node_count,
            )

        # Rule 7: High complexity
        elif signature.complexity >= 4.0:
            predicted_type = "hybrid_decomposition"
            logger.debug("Predicted hybrid (complexity=%.2f)", signature.complexity)

        # Rule 8: Moderate complexity (default)
        else:
            predicted_type = "hierarchical_decomposition"
            logger.debug(
                "Predicted hierarchical (default, complexity=%.2f)",
                signature.complexity,
            )

        # Get strategy from library
        strategy = self.library.get_strategy_by_type(predicted_type)

        # Validate strategy before use
        if strategy and not hasattr(strategy, "decompose"):
            logger.error("Strategy %s missing decompose method", predicted_type)
            strategy = None

        if not strategy:
            # Fallback to any available strategy
            logger.warning(
                "Predicted strategy type %s not available, using fallback",
                predicted_type,
            )
            # Try common types
            for fallback_type in [
                "hierarchical_decomposition",
                "simple",
                "direct_decomposition",
            ]:
                strategy = self.library.get_strategy_by_type(fallback_type)
                if strategy and hasattr(strategy, "decompose"):
                    break
                else:
                    strategy = None

        return strategy

    def learn_from_execution(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ):
        """
        Learn from execution outcome - WITH SAFETY FILTERING

        Args:
            problem: Original problem
            plan: Executed plan
            outcome: Execution outcome (already safety-validated)
        """
        # SAFETY: Skip learning if outcome was safety-blocked
        if outcome.metadata.get("safety_blocked"):
            logger.debug("Skipping learning from safety-blocked outcome")
            return

        # Record performance
        self.performance_tracker.record_execution(problem, plan, outcome)

        # Update strategy profile
        if plan.strategy:
            domain = problem.metadata.get("domain", "general")
            strategy_name = (
                plan.strategy.name if hasattr(plan.strategy, "name") else "unknown"
            )
            self.strategy_profiler.update_affinity(
                strategy_name, domain, problem.complexity_score, outcome.success
            )

        # Update thresholds
        self.thresholds.update_from_outcome(
            problem.complexity_score, outcome.success, outcome.execution_time
        )

        # Update success statistics
        if outcome.success:
            self.successful_decompositions += 1
        else:
            # Create learning gap for failure
            gap = self.create_learning_gap(problem)
            self.learning_gaps.append(gap)

        # Update complexity estimation accuracy
        if outcome.metrics.get("actual_complexity"):
            self.performance_tracker.update_complexity_accuracy(
                plan.estimated_complexity, outcome.metrics["actual_complexity"]
            )

        logger.debug(
            "Learned from execution: success=%s, time=%.2f",
            outcome.success,
            outcome.execution_time,
        )

    def select_diverse_test_domains(self, candidate) -> List[str]:
        """
        Select diverse test domains for validation

        Args:
            candidate: Candidate strategy or principle

        Returns:
            List of diverse domain names
        """
        # Get all available domains
        all_domains = self._get_available_domains()

        # Categorize by data availability
        categorized = self.domain_selector.categorize_domains_by_data(all_domains)

        # Select stratified sample
        selected = self.domain_selector.select_stratified_sample(
            categorized.get(DomainDataCategory.FREQUENT, []),
            categorized.get(DomainDataCategory.MEDIUM, []),
            categorized.get(DomainDataCategory.RARE, []),
        )

        # Add similar domains if candidate has origin domain
        if hasattr(candidate, "origin_domain"):
            similar = self.domain_selector.find_similar_domains(
                candidate.origin_domain, exclude=selected
            )
            selected.extend(similar[:2])  # Add up to 2 similar domains

        return selected

    def create_learning_gap(self, failed_problem: ProblemGraph) -> LearningGap:
        """
        Create learning gap from failed decomposition

        Args:
            failed_problem: Problem that failed decomposition

        Returns:
            Learning gap specification
        """
        # Analyze failure
        failure_analysis = self._analyze_failure(failed_problem)

        # Identify missing capabilities
        missing_capability = self._identify_missing_capability(failed_problem)

        # Suggest strategies
        suggested_strategies = self._suggest_alternative_strategies(failed_problem)

        gap = LearningGap(
            gap_type="decomposition_failure",
            problem_signature=failed_problem.get_signature(),
            failure_reason=failure_analysis.get("reason", "unknown"),
            missing_capability=missing_capability,
            suggested_strategies=suggested_strategies,
            priority=self._calculate_gap_priority(failed_problem),
            metadata={
                "complexity": failed_problem.complexity_score,
                "domain": failed_problem.metadata.get("domain", "unknown"),
                "timestamp": time.time(),
            },
        )

        logger.info("Created learning gap: %s", gap.failure_reason)

        return gap

    def _analyze_complexity(self, problem_graph: ProblemGraph) -> float:
        """Analyze problem complexity"""
        # Convert to NetworkX for analysis
        G = problem_graph.to_networkx()

        # Calculate complexity factors
        factors = []

        # Node count factor
        if hasattr(G, "number_of_nodes"):
            node_count = G.number_of_nodes()
        else:
            node_count = len(list(G.nodes())) if hasattr(G, "nodes") else 0

        node_factor = min(1.0, node_count / 100)
        factors.append(node_factor)

        # Edge density factor
        if node_count > 1:
            if hasattr(G, "number_of_edges"):
                edge_count = G.number_of_edges()
            else:
                edge_count = len(list(G.edges())) if hasattr(G, "edges") else 0
            edge_density = edge_count / (node_count * (node_count - 1))
            factors.append(edge_density)

        # Depth factor (longest path)
        if NETWORKX_AVAILABLE:
            try:
                longest_path = nx.dag_longest_path_length(G)
                depth_factor = min(1.0, longest_path / 10)
                factors.append(depth_factor)
            except Exception:
                # Not a DAG, use average degree
                if hasattr(G, "degree"):
                    degrees = [d for n, d in G.degree()]
                    if degrees:
                        avg_degree = np.mean(degrees)
                        factors.append(min(1.0, avg_degree / 10))

        # Branching factor
        if problem_graph.root and problem_graph.root in list(
            G.nodes() if hasattr(G, "nodes") else []
        ):
            if NETWORKX_AVAILABLE:
                descendants = nx.descendants(G, problem_graph.root)
                branching_factor = len(descendants) / max(1, node_count)
            else:
                branching_factor = 0.5  # Default
            factors.append(branching_factor)

        # Calculate weighted complexity
        if factors:
            complexity = np.mean(factors) * 5  # Scale to 0-5
        else:
            complexity = 2.5  # Default moderate complexity

        # Proper clamping
        return float(np.clip(complexity, 1.0, 5.0))

    def _select_strategy(
        self, problem_graph: ProblemGraph, complexity: float
    ) -> Optional[DecompositionStrategy]:
        """Select appropriate decomposition strategy - DEPRECATED, use _predict_best_strategy"""
        # This method is kept for backward compatibility
        # New code should use _predict_best_strategy which is more sophisticated
        signature = self._extract_problem_signature(problem_graph)
        signature.complexity = complexity
        return self._predict_best_strategy(problem_graph, signature)

    def _create_decomposition_plan(
        self,
        problem_graph: ProblemGraph,
        strategy: DecompositionStrategy,
        complexity: float,
    ) -> DecompositionPlan:
        """Create decomposition plan using strategy"""
        plan = DecompositionPlan(
            strategy=strategy,
            estimated_complexity=complexity,
            confidence=0.5,  # Base confidence
        )

        # Apply strategy to create steps
        if hasattr(strategy, "decompose"):
            steps = strategy.decompose(problem_graph)
        else:
            steps = []

        for step in steps:
            plan.add_step(step)

        # Calculate confidence based on strategy profile
        strategy_name = strategy.name if hasattr(strategy, "name") else "unknown"
        if strategy_name in self.strategy_profiler.strategy_profiles:
            profile = self.strategy_profiler.strategy_profiles[strategy_name]

            # Adjust confidence based on domain suitability
            domain = problem_graph.metadata.get("domain", "general")
            if domain in profile.get("domain_suitability", []):
                plan.confidence += 0.2

        # Adjust confidence based on historical performance
        success_rate = self.performance_tracker.get_strategy_success_rate(strategy_name)
        plan.confidence = plan.confidence * 0.5 + success_rate * 0.5

        return plan

    def _evaluate_plan(self, plan) -> float:
        """Evaluate quality of decomposition plan or execution plan"""
        score = 0.0

        # Handle both DecompositionPlan and ExecutionPlan (from fallback_chain)
        # ExecutionPlan has different structure - need to convert or handle separately

        # Check if it's an ExecutionPlan (from fallback_chain) by checking for specific attributes
        if hasattr(plan, "overall_confidence") and callable(plan.overall_confidence):
            # This is an ExecutionPlan from fallback_chain
            confidence = plan.overall_confidence()
            score += confidence * 0.4

            # Get components/steps count
            num_steps = len(plan.components) if hasattr(plan, "components") else 0
            if 3 <= num_steps <= 10:
                score += 0.3
            elif num_steps < 3:
                score += 0.1
            else:
                score += 0.2

            # ExecutionPlan doesn't have a strategy attribute, use metadata if available
            if hasattr(plan, "metadata") and "strategy" in plan.metadata:
                strategy_name = plan.metadata["strategy"]
                success_rate = self.performance_tracker.get_strategy_success_rate(
                    strategy_name
                )
                score += success_rate * 0.3
            else:
                # No strategy info, give neutral score
                score += 0.15

        else:
            # This is a DecompositionPlan
            # Base score from confidence
            score += plan.confidence * 0.4

            # Score from number of steps (prefer moderate granularity)
            num_steps = len(plan.steps)
            if 3 <= num_steps <= 10:
                score += 0.3
            elif num_steps < 3:
                score += 0.1
            else:
                score += 0.2

            # Score from strategy track record
            if plan.strategy:
                strategy_name = (
                    plan.strategy.name if hasattr(plan.strategy, "name") else "unknown"
                )
                success_rate = self.performance_tracker.get_strategy_success_rate(
                    strategy_name
                )
                score += success_rate * 0.3

        return min(1.0, score)

    def _get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        # Default domains
        domains = [
            "general",
            "optimization",
            "classification",
            "generation",
            "analysis",
            "planning",
            "control",
            "reasoning",
            "perception",
        ]

        # Add domains from strategy profiler
        for domain_scores in self.strategy_profiler.domain_affinity.values():
            domains.extend(domain_scores.keys())

        return list(set(domains))

    def _analyze_failure(self, problem: ProblemGraph) -> Dict[str, Any]:
        """Analyze why decomposition failed"""
        analysis = {}

        # Check complexity
        if problem.complexity_score > 4:
            analysis["reason"] = "excessive_complexity"
        elif problem.complexity_score < 1.5:
            analysis["reason"] = "insufficient_structure"
        else:
            # Check graph properties
            G = problem.to_networkx()

            if hasattr(G, "number_of_nodes"):
                node_count = G.number_of_nodes()
            else:
                node_count = len(list(G.nodes())) if hasattr(G, "nodes") else 0

            if node_count == 0:
                analysis["reason"] = "empty_problem"
            elif NETWORKX_AVAILABLE and not nx.is_weakly_connected(G):
                analysis["reason"] = "disconnected_components"
            else:
                analysis["reason"] = "strategy_mismatch"

        return analysis

    def _identify_missing_capability(self, problem: ProblemGraph) -> Optional[str]:
        """Identify what capability is missing"""
        # Analyze problem characteristics
        G = problem.to_networkx()

        if NETWORKX_AVAILABLE:
            if nx.is_directed_acyclic_graph(G):
                try:
                    path_length = nx.dag_longest_path_length(G)
                    if path_length > 10:
                        return "deep_hierarchy_handling"
                except Exception as e:
                    logger.debug(f"Operation failed: {e}")
            else:
                return "cycle_handling"

        # Check for specific patterns
        if problem.metadata.get("requires_backtracking"):
            return "backtracking_capability"

        if problem.metadata.get("requires_optimization"):
            return "optimization_decomposition"

        return None

    def _suggest_alternative_strategies(self, problem: ProblemGraph) -> List[str]:
        """Suggest alternative decomposition strategies"""
        suggestions = []

        # Based on problem characteristics
        if problem.complexity_score > 3.5:
            suggestions.append("hierarchical_decomposition")
            suggestions.append("iterative_refinement")
        elif problem.complexity_score < 2:
            suggestions.append("direct_solution")
            suggestions.append("simple_subdivision")

        # Based on domain
        domain = problem.metadata.get("domain", "general")
        if domain == "optimization":
            suggestions.append("constraint_decomposition")
        elif domain == "planning":
            suggestions.append("goal_decomposition")

        return suggestions

    def _calculate_gap_priority(self, problem: ProblemGraph) -> float:
        """Calculate priority of learning gap"""
        # Higher priority for more complex problems
        complexity_factor = problem.complexity_score / 5.0

        # Higher priority for frequently seen domains
        domain = problem.metadata.get("domain", "general")
        domain_frequency = len(
            [
                h
                for h in self.performance_tracker.execution_history
                if h.get("domain") == domain
            ]
        ) / max(1, len(self.performance_tracker.execution_history))

        priority = complexity_factor * 0.6 + domain_frequency * 0.4

        return min(1.0, priority)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "decomposition_stats": {
                "total_decompositions": self.total_decompositions,
                "successful_decompositions": self.successful_decompositions,
                "success_rate": self.successful_decompositions
                / max(1, self.total_decompositions),
                "cached_plans": len(self.decomposition_cache),
            },
            "execution_stats": self.executor.get_statistics()
            if hasattr(self.executor, "get_statistics")
            else {},
            "performance_stats": {
                "complexity_accuracy": self.performance_tracker.get_complexity_estimation_accuracy(),
                "execution_history_size": len(
                    self.performance_tracker.execution_history
                ),
            },
            "learning_stats": {
                "learning_gaps": len(self.learning_gaps),
                "prediction_history_size": len(self.prediction_history),
            },
        }

        # Add safety statistics
        if self.safety_validator:
            stats["safety"] = {
                "enabled": True,
                "blocks": dict(self.safety_blocks),
                "corrections": dict(self.safety_corrections),
                "total_blocks": sum(self.safety_blocks.values()),
                "total_corrections": sum(self.safety_corrections.values()),
            }
        else:
            stats["safety"] = {"enabled": False}

        return stats


class DomainSelector:
    """Stratified domain selection for testing"""

    def __init__(self):
        """Initialize domain selector"""
        self.domain_data_counts = defaultdict(int)
        self.domain_similarity_cache = {}

        # Size limits for unbounded dictionaries
        self.max_domains = 10000
        self.max_similarity_cache = 10000

        # Initialize with some default data counts
        self._initialize_domain_data()

        logger.info("DomainSelector initialized")

    def categorize_domains_by_data(
        self, domains: List[str]
    ) -> Dict[DomainDataCategory, List[str]]:
        """
        Categorize domains by data availability

        Args:
            domains: List of domain names

        Returns:
            Dictionary mapping categories to domain lists
        """
        categorized = defaultdict(list)

        for domain in domains:
            count = self.domain_data_counts.get(domain, 0)

            if count >= 1000:
                category = DomainDataCategory.FREQUENT
            elif count >= 100:
                category = DomainDataCategory.MEDIUM
            elif count > 0:
                category = DomainDataCategory.RARE
            else:
                category = DomainDataCategory.NOVEL

            categorized[category].append(domain)

        return dict(categorized)

    def select_stratified_sample(
        self, frequent: List[str], medium: List[str], rare: List[str]
    ) -> List[str]:
        """
        Select stratified sample from categorized domains

        Args:
            frequent: Frequently seen domains
            medium: Medium frequency domains
            rare: Rarely seen domains

        Returns:
            Stratified sample of domains
        """
        selected = []

        # Select 3 from frequent (if available)
        if frequent:
            n_frequent = min(3, len(frequent))
            selected.extend(np.random.choice(frequent, n_frequent, replace=False))

        # Select 2 from medium (if available)
        if medium:
            n_medium = min(2, len(medium))
            selected.extend(np.random.choice(medium, n_medium, replace=False))

        # Select 2 from rare (if available)
        if rare:
            n_rare = min(2, len(rare))
            selected.extend(np.random.choice(rare, n_rare, replace=False))

        return selected

    def find_similar_domains(
        self, origin_domain: str, exclude: List[str] = []
    ) -> List[str]:
        """
        Find domains similar to origin domain

        Args:
            origin_domain: Reference domain
            exclude: Domains to exclude

        Returns:
            List of similar domains
        """
        # Get all domains
        all_domains = list(self.domain_data_counts.keys())

        # Calculate similarities
        similarities = []
        for domain in all_domains:
            if domain == origin_domain or domain in exclude:
                continue

            similarity = self._calculate_domain_similarity(origin_domain, domain)
            similarities.append((domain, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top similar domains
        return [domain for domain, _ in similarities[:5]]

    def update_data_count(self, domain: str, count: int):
        """Update data count for domain"""
        # Enforce domain limit
        if len(self.domain_data_counts) >= self.max_domains:
            # Remove domain with lowest count
            if self.domain_data_counts:
                min_domain = min(self.domain_data_counts.items(), key=lambda x: x[1])
                del self.domain_data_counts[min_domain[0]]

        self.domain_data_counts[domain] = count

    def _initialize_domain_data(self):
        """Initialize with default domain data counts"""
        defaults = {
            "general": 10000,
            "optimization": 5000,
            "classification": 8000,
            "generation": 2000,
            "analysis": 6000,
            "planning": 1000,
            "control": 1500,
            "reasoning": 4000,
            "perception": 7000,
            "prediction": 3000,
            "search": 4500,
            "learning": 5500,
        }

        self.domain_data_counts.update(defaults)

    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between two domains"""
        # Check cache
        cache_key = tuple(sorted([domain1, domain2]))
        if cache_key in self.domain_similarity_cache:
            return self.domain_similarity_cache[cache_key]

        # Simple similarity based on name overlap
        words1 = set(domain1.lower().split("_"))
        words2 = set(domain2.lower().split("_"))

        if not words1 or not words2:
            similarity = 0.0
        else:
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            similarity = intersection / union if union > 0 else 0.0

        # Consider data availability similarity
        count1 = self.domain_data_counts.get(domain1, 0)
        count2 = self.domain_data_counts.get(domain2, 0)

        if count1 > 0 and count2 > 0:
            count_similarity = min(count1, count2) / max(count1, count2)
            similarity = similarity * 0.7 + count_similarity * 0.3

        # Enforce cache limit before adding
        if len(self.domain_similarity_cache) >= self.max_similarity_cache:
            # Remove oldest entry (FIFO approximation)
            oldest_key = next(iter(self.domain_similarity_cache))
            del self.domain_similarity_cache[oldest_key]

        # Cache result
        self.domain_similarity_cache[cache_key] = similarity

        return similarity
