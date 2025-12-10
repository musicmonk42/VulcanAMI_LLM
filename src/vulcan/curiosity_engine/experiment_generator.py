"""
experiment_generator.py - Experiment generation for Curiosity Engine
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern
"""

import copy
import hashlib
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of experiments"""

    DECOMPOSITION = "decomposition"
    CAUSAL = "causal"
    TRANSFER = "transfer"
    SYNTHETIC = "synthetic"
    EXPLORATORY = "exploratory"
    VALIDATION = "validation"
    ITERATIVE = "iterative"
    ABLATION = "ablation"


class FailureType(Enum):
    """Types of experiment failures"""

    TOO_SIMPLE = "too_simple"
    WRONG_APPROACH = "wrong_approach"
    TIMEOUT = "timeout"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSTABLE_OUTPUT = "unstable_output"
    RESOURCE_EXCEEDED = "resource_exceeded"
    CONVERGENCE_FAILURE = "convergence_failure"
    VALIDATION_FAILURE = "validation_failure"


@dataclass
class Constraint:
    """Safety constraint for experiments"""

    name: str
    constraint_type: str  # "memory", "time", "output", "input", "resource"
    limit: Any
    action: str = "abort"  # "abort", "truncate", "warn", "adapt"
    severity: float = 1.0  # 0-1 severity level
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if constraint is satisfied

        Returns:
            Tuple of (satisfied, violation_message)
        """
        try:
            if self.constraint_type == "memory":
                satisfied = value <= self.limit
                if not satisfied:
                    return False, f"Memory usage {value} exceeds limit {self.limit}"

            elif self.constraint_type == "time":
                satisfied = value <= self.limit
                if not satisfied:
                    return False, f"Time {value} exceeds limit {self.limit}"

            elif self.constraint_type == "output":
                # FIX: Proper type handling for output constraints
                if isinstance(self.limit, int):
                    satisfied = len(str(value)) <= self.limit
                    if not satisfied:
                        return (
                            False,
                            f"Output size {len(str(value))} exceeds limit {self.limit}",
                        )
                elif isinstance(self.limit, (list, tuple, set)):
                    # Must be exact match for collections
                    satisfied = value in self.limit
                    if not satisfied:
                        return (
                            False,
                            f"Output {value} not in allowed values {self.limit}",
                        )
                elif isinstance(self.limit, str):
                    # Exact equality for string limits
                    satisfied = str(value) == self.limit
                    if not satisfied:
                        return (
                            False,
                            f"Output '{value}' does not match required '{self.limit}'",
                        )
                else:
                    satisfied = value == self.limit
                    if not satisfied:
                        return (
                            False,
                            f"Output {value} does not match limit {self.limit}",
                        )

            elif self.constraint_type == "input":
                if isinstance(self.limit, int):
                    satisfied = len(str(value)) <= self.limit
                    if not satisfied:
                        return (
                            False,
                            f"Input size {len(str(value))} exceeds limit {self.limit}",
                        )
                elif isinstance(self.limit, (list, tuple, set)):
                    satisfied = value in self.limit
                    if not satisfied:
                        return False, f"Input {value} not in allowed values"
                elif isinstance(self.limit, str):
                    satisfied = str(value) == self.limit
                    if not satisfied:
                        return (
                            False,
                            f"Input '{value}' does not match required '{self.limit}'",
                        )
                else:
                    satisfied = value == self.limit
                    if not satisfied:
                        return False, f"Input {value} does not match limit"

            elif self.constraint_type == "resource":
                # Generic resource constraint
                satisfied = value <= self.limit
                if not satisfied:
                    return False, f"Resource usage {value} exceeds limit {self.limit}"
            else:
                # Unknown constraint type - default to pass
                return True, None

            return satisfied, None

        except Exception as e:
            return False, f"Error checking constraint: {e}"

    def adapt_value(self, value: Any) -> Any:
        """Adapt value to satisfy constraint if possible"""
        try:
            if self.action != "adapt":
                return value

            if self.constraint_type in ["memory", "time", "resource"]:
                return min(value, self.limit)
            elif self.constraint_type in ["output", "input"] and isinstance(
                self.limit, int
            ):
                return str(value)[: self.limit]

            return value
        except Exception as e:
            logger.warning("Error adapting value: %s", e)
            return value


@dataclass
class KnowledgeGap:
    """Knowledge gap representation"""

    type: str
    domain: str
    priority: float
    estimated_cost: float
    missing_capability: Optional[str] = None
    id: Optional[str] = None
    gap_id: Optional[str] = None
    complexity: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure gap has a unique ID"""
        if not self.id and not self.gap_id:
            # FIX: Use UUID for true uniqueness instead of hash
            unique_id = str(uuid.uuid4())[:8]
            self.id = self.gap_id = f"gap_{self.type}_{unique_id}"
        elif self.gap_id and not self.id:
            self.id = self.gap_id
        elif self.id and not self.gap_id:
            self.gap_id = self.id


@dataclass
class Experiment:
    """Single experiment specification"""

    gap: KnowledgeGap
    complexity: float
    timeout: float
    success_criteria: Dict[str, Any]
    safety_constraints: List[Constraint] = field(default_factory=list)
    experiment_type: ExperimentType = ExperimentType.EXPLORATORY
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    experiment_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Generate deterministic experiment ID if not provided"""
        if not self.experiment_id:
            # FIX: Exclude timestamp for determinism
            content = f"{self.gap.id}_{self.experiment_type.value}_{self.iteration}"
            self.experiment_id = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]

            # Store timestamp in metadata instead
            if "created_at" not in self.metadata:
                self.metadata["created_at"] = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "experiment_id": self.experiment_id,
            "gap_type": self.gap.type,
            "complexity": self.complexity,
            "timeout": self.timeout,
            "success_criteria": self.success_criteria,
            "experiment_type": self.experiment_type.value,
            "parameters": self.parameters,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }

    def meets_criteria(self, result: Any) -> Tuple[bool, List[str]]:
        """
        Check if result meets success criteria

        Returns:
            Tuple of (success, list_of_unmet_criteria)
        """
        if not self.success_criteria:
            return True, []

        # FIX: Validate result type
        if result is None:
            return False, ["Result is None"]

        if isinstance(result, str) and result.startswith("Error"):
            return False, [f"Result is error string: {result}"]

        unmet = []

        try:
            for criterion, expected in self.success_criteria.items():
                if criterion == "min_accuracy":
                    if isinstance(result, dict):
                        accuracy = result.get("accuracy", 0)
                    else:
                        accuracy = getattr(result, "accuracy", 0)

                    if accuracy < expected:
                        unmet.append(f"Accuracy {accuracy} < {expected}")

                elif criterion == "max_error":
                    if isinstance(result, dict):
                        error = result.get("error", float("inf"))
                    else:
                        error = getattr(result, "error", float("inf"))

                    if error > expected:
                        unmet.append(f"Error {error} > {expected}")

                elif criterion == "required_output":
                    if isinstance(result, dict):
                        output = result.get("output", "")
                    else:
                        output = getattr(result, "output", "")

                    if expected not in str(output):
                        unmet.append(f"Required output '{expected}' not found")

                elif criterion == "min_confidence":
                    if isinstance(result, dict):
                        confidence = result.get("confidence", 0)
                    else:
                        confidence = getattr(result, "confidence", 0)

                    if confidence < expected:
                        unmet.append(f"Confidence {confidence} < {expected}")
        except Exception as e:
            unmet.append(f"Error checking criteria: {e}")

        return len(unmet) == 0, unmet

    def validate_constraints(self) -> Tuple[bool, List[str]]:
        """Validate all constraints"""
        violations = []

        try:
            for constraint in self.safety_constraints:
                # Check basic constraint validity
                if (
                    constraint.constraint_type == "time"
                    and self.timeout > constraint.limit
                ):
                    violations.append(
                        f"Timeout {self.timeout} exceeds time constraint {constraint.limit}"
                    )
                elif constraint.constraint_type == "memory" and hasattr(
                    self, "estimated_memory"
                ):
                    if self.estimated_memory > constraint.limit:
                        violations.append(f"Estimated memory exceeds limit")
        except Exception as e:
            violations.append(f"Error validating constraints: {e}")

        return len(violations) == 0, violations


@dataclass
class FailureAnalysis:
    """Detailed failure taxonomy"""

    type: FailureType
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_adjustments: List[str] = field(default_factory=list)
    recovery_possible: bool = True
    confidence: float = 0.5
    root_cause: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    experiment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.type.value,
            "details": self.details,
            "suggested_adjustments": self.suggested_adjustments,
            "recovery_possible": self.recovery_possible,
            "confidence": self.confidence,
            "root_cause": self.root_cause,
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
        }

    def get_primary_adjustment(self) -> Optional[str]:
        """Get the most important adjustment"""
        if self.suggested_adjustments:
            return self.suggested_adjustments[0]
        return None


class ExperimentTemplates:
    """Manages experiment templates - SEPARATED CONCERN"""

    def __init__(self):
        self.templates = self._load_templates()
        self.lock = threading.RLock()

    def get_template(self, experiment_type: str) -> Dict[str, Any]:
        """Get template for experiment type"""
        with self.lock:
            return self.templates.get(experiment_type, self._get_default_template())

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load experiment templates"""
        return {
            "decomposition": {
                "timeout_multiplier": 1.5,
                "min_components": 2,
                "strategies": ["hierarchical", "functional", "modular", "recursive"],
                "version": "1.1",
            },
            "causal": {
                "timeout_multiplier": 2.0,
                "min_sample_size": 50,
                "strategies": ["rct", "instrumental", "natural", "granger"],
                "version": "1.1",
            },
            "transfer": {
                "timeout_multiplier": 2.5,
                "min_overlap": 0.3,
                "strategies": ["direct", "gradual", "meta", "progressive"],
                "version": "1.1",
            },
            "synthetic": {
                "timeout_multiplier": 0.5,
                "max_variations": 100,
                "version": "1.0",
            },
            "exploratory": {
                "timeout_multiplier": 1.0,
                "strategies": ["random", "guided", "systematic"],
                "version": "1.0",
            },
        }

    def _get_default_template(self) -> Dict[str, Any]:
        """Get default template"""
        return {"timeout_multiplier": 1.0, "strategies": ["default"], "version": "1.0"}


class ExperimentCache:
    """Manages experiment caching - SEPARATED CONCERN"""

    def __init__(self, cache_ttl: int = 300, max_size: int = 100):
        self.cache_ttl = cache_ttl
        self.max_size = max_size
        self.cache = {}
        self.cache_times = {}
        self.lock = threading.RLock()

    def get(self, gap_id: str, gap_type: str) -> Optional[List[Experiment]]:
        """Get cached experiments"""
        with self.lock:
            try:
                cache_key = f"{gap_id}_{gap_type}"

                if cache_key in self.cache:
                    cache_time = self.cache_times.get(cache_key, 0)
                    if time.time() - cache_time < self.cache_ttl:
                        logger.debug("Using cached experiments for gap %s", gap_id)
                        return self.cache[cache_key]

                return None
            except Exception as e:
                logger.warning("Error getting from cache: %s", e)
                return None

    def put(self, gap_id: str, gap_type: str, experiments: List[Experiment]):
        """Cache experiments"""
        with self.lock:
            try:
                cache_key = f"{gap_id}_{gap_type}"

                # Limit cache size
                if len(self.cache) >= self.max_size:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        self.cache_times.keys(), key=lambda k: self.cache_times[k]
                    )[:20]
                    for key in oldest_keys:
                        self.cache.pop(key, None)
                        self.cache_times.pop(key, None)

                self.cache[cache_key] = experiments
                self.cache_times[cache_key] = time.time()
            except Exception as e:
                logger.warning("Error putting to cache: %s", e)


class ExperimentTracker:
    """Tracks experiment lifecycle - SEPARATED CONCERN"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.active_experiments = {}
        self.completed_experiments = deque(maxlen=max_history)
        self.success_rates = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()

    def track_active(self, experiment: Experiment):
        """Track active experiment"""
        with self.lock:
            try:
                self.active_experiments[experiment.experiment_id] = {
                    "experiment": experiment,
                    "start_time": time.time(),
                    "status": "running",
                }
            except Exception as e:
                logger.error("Error tracking active experiment: %s", e)

    def complete(self, experiment_id: str, result: Any):
        """Mark experiment as completed"""
        with self.lock:
            try:
                if experiment_id in self.active_experiments:
                    exp_info = self.active_experiments[experiment_id]
                    exp_info["status"] = "completed"
                    exp_info["end_time"] = time.time()
                    exp_info["result"] = result

                    # Move to completed
                    self.completed_experiments.append(exp_info)
                    del self.active_experiments[experiment_id]

                    # Update success rate
                    experiment = exp_info["experiment"]
                    success = (
                        result.get("success", False)
                        if isinstance(result, dict)
                        else False
                    )
                    self.success_rates[experiment.experiment_type.value].append(
                        1.0 if success else 0.0
                    )
            except Exception as e:
                logger.error("Error completing experiment: %s", e)

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        with self.lock:
            try:
                stats = {
                    "active_experiments": len(self.active_experiments),
                    "completed_experiments": len(self.completed_experiments),
                    "success_rates": {},
                }

                # Calculate success rates
                for exp_type, rates in self.success_rates.items():
                    if rates:
                        stats["success_rates"][exp_type] = np.mean(rates)

                return stats
            except Exception as e:
                logger.error("Error getting statistics: %s", e)
                return {}


class SyntheticDataGenerator:
    """Generates synthetic data for experiments - SEPARATED CONCERN"""

    def generate(self, gap: KnowledgeGap, noise_level: float = 0.1) -> Dict[str, Any]:
        """Generate synthetic data for testing"""
        try:
            # Use gap ID for reproducible randomness
            seed = hash(gap.id or gap.type) % 2**32
            np.random.seed(seed)

            # Generate data based on gap type
            if gap.type == "decomposition":
                data = self._generate_hierarchical_data()
            elif gap.type == "causal":
                data = self._generate_causal_data(noise_level)
            else:
                data = self._generate_generic_data(noise_level)

            return {
                "data": data,
                "metadata": {
                    "gap_type": gap.type,
                    "synthetic": True,
                    "seed": seed,
                    "noise_level": noise_level,
                    "timestamp": time.time(),
                },
            }
        except Exception as e:
            logger.error("Error generating synthetic data: %s", e)
            return {"data": {}, "metadata": {"error": str(e)}}

    def _generate_hierarchical_data(self) -> Dict[str, Any]:
        """Generate hierarchical structure data"""
        return {"structure": "tree", "nodes": 20, "depth": 4, "branching_factor": 3}

    def _generate_causal_data(self, noise_level: float) -> Dict[str, Any]:
        """Generate causal relationship data"""
        n_samples = 100
        x = np.random.randn(n_samples)
        y = 0.7 * x + np.random.randn(n_samples) * noise_level
        return {"x": x.tolist(), "y": y.tolist(), "true_coefficient": 0.7}

    def _generate_generic_data(self, noise_level: float) -> Dict[str, Any]:
        """Generate generic synthetic data"""
        data = {
            "inputs": np.random.randn(100, 10).tolist(),
            "outputs": np.random.randint(0, 2, 100).tolist(),
        }

        # Add noise
        if noise_level > 0:
            inputs = np.array(data["inputs"])
            inputs += np.random.randn(*inputs.shape) * noise_level
            data["inputs"] = inputs.tolist()

        return data


class DomainSimilarityCalculator:
    """Calculates domain similarity - SEPARATED CONCERN"""

    def __init__(self):
        self.related_domains = {
            ("machine_learning", "deep_learning"): 0.8,
            ("optimization", "search"): 0.6,
            ("planning", "scheduling"): 0.7,
        }
        self.lock = threading.RLock()

    @lru_cache(maxsize=100)
    def calculate(self, source: str, target: str) -> float:
        """Calculate similarity between domains"""
        try:
            if source == target:
                return 1.0

            # Simple heuristic based on common words
            source_words = set(source.lower().split("_"))
            target_words = set(target.lower().split("_"))

            if not source_words or not target_words:
                return 0.3

            intersection = len(source_words & target_words)
            union = len(source_words | target_words)

            similarity = intersection / union if union > 0 else 0.0

            # Boost for known related domains
            key = tuple(sorted([source, target]))
            if key in self.related_domains:
                similarity = max(similarity, self.related_domains[key])

            return similarity
        except Exception as e:
            logger.warning("Error calculating domain similarity: %s", e)
            return 0.3


class ExperimentBuilder:
    """Builds individual experiments - SEPARATED CONCERN"""

    def __init__(self, default_timeout: float = 30.0, max_complexity: float = 1.0):
        self.default_timeout = default_timeout
        self.max_complexity = max_complexity
        self.templates = ExperimentTemplates()
        self.domain_calculator = DomainSimilarityCalculator()
        self.synthetic_generator = SyntheticDataGenerator()

    def build_decomposition_experiment(
        self, gap: KnowledgeGap, complexity: float, strategy: str, level_index: int
    ) -> Experiment:
        """Build a single decomposition experiment"""
        try:
            # EXAMINE: Get template and parameters
            template = self.templates.get_template("decomposition")
            complexity = min(complexity, self.max_complexity)

            # Calculate parameters based on complexity
            depth = max(2, int(3 * complexity) + 1)
            breadth = max(2, int(5 * complexity))

            # SELECT & APPLY: Create experiment
            return Experiment(
                gap=gap,
                complexity=complexity,
                timeout=self.default_timeout * template.get("timeout_multiplier", 1.5),
                success_criteria={
                    "decomposition_success": True,
                    "min_components": template.get("min_components", 2),
                    "max_error": 0.3,
                    "min_confidence": 0.6,
                },
                safety_constraints=[
                    Constraint("memory_limit", "memory", 512 * 1024 * 1024),
                    Constraint("time_limit", "time", self.default_timeout * 2),
                    Constraint("output_size", "output", 10000),
                ],
                experiment_type=ExperimentType.DECOMPOSITION,
                parameters={
                    "strategy": strategy,
                    "depth": depth,
                    "breadth": breadth,
                    "allow_synthetic": complexity > 0.7,
                    "parallel_execution": complexity < 0.5,
                    "validation_level": "strict" if complexity > 0.8 else "normal",
                },
                metadata={
                    "complexity_level": level_index,
                    "template_version": template.get("version", "1.0"),
                },
            )
        except Exception as e:
            logger.error("Error building decomposition experiment: %s", e)
            raise

    def build_causal_experiment(
        self, gap: KnowledgeGap, strategy: str, intervention: Dict[str, Any]
    ) -> Experiment:
        """Build a single causal experiment"""
        try:
            # EXAMINE: Get template
            template = self.templates.get_template("causal")

            # Adjust parameters based on strategy
            if strategy == "direct":
                sample_size = 100
                confidence_threshold = 0.7
                complexity = 0.6
            elif strategy == "instrumental":
                sample_size = 150
                confidence_threshold = 0.6
                complexity = 0.7
            else:  # natural
                sample_size = 200
                confidence_threshold = 0.5
                complexity = 0.7

            # SELECT & APPLY: Create experiment
            return Experiment(
                gap=gap,
                complexity=complexity,
                timeout=self.default_timeout * template.get("timeout_multiplier", 2.0),
                success_criteria={
                    "causal_strength_detected": True,
                    "min_confidence": confidence_threshold,
                    "p_value": 0.05,
                    "effect_size": 0.2,
                },
                safety_constraints=[
                    Constraint("intervention_limit", "input", 1000),
                    Constraint("observation_limit", "output", 10000),
                    Constraint("sample_size", "resource", sample_size * 2),
                ],
                experiment_type=ExperimentType.CAUSAL,
                parameters={
                    "intervention": intervention,
                    "strategy": strategy,
                    "sample_size": max(
                        template.get("min_sample_size", 50), sample_size
                    ),
                    "control_variables": gap.metadata.get("confounders", []),
                    "bootstrap_iterations": 100 if strategy != "direct" else 50,
                    "significance_level": 0.05,
                },
                metadata={
                    "intervention_type": intervention.get("type", "unknown"),
                    "expected_direction": gap.metadata.get(
                        "expected_direction", "positive"
                    ),
                },
            )
        except Exception as e:
            logger.error("Error building causal experiment: %s", e)
            raise

    def build_transfer_experiment(self, gap: KnowledgeGap, strategy: str) -> Experiment:
        """Build a single transfer experiment"""
        try:
            # EXAMINE: Get template and domains
            template = self.templates.get_template("transfer")
            source_domain = gap.metadata.get("source_domain", "default")
            target_domain = gap.metadata.get("target_domain", "unknown")

            # Calculate domain similarity
            domain_similarity = self.domain_calculator.calculate(
                source_domain, target_domain
            )

            # Adjust parameters based on strategy and similarity
            if strategy == "direct":
                complexity = 0.5 + (1 - domain_similarity) * 0.3
                adaptation_steps = 20
                regularization = 0.05
            elif strategy == "fine_tune":
                complexity = 0.6 + (1 - domain_similarity) * 0.3
                adaptation_steps = 50
                regularization = 0.1
            else:  # meta_learning
                complexity = 0.7 + (1 - domain_similarity) * 0.3
                adaptation_steps = 100
                regularization = 0.2

            # SELECT & APPLY: Create experiment
            return Experiment(
                gap=gap,
                complexity=min(complexity, self.max_complexity),
                timeout=self.default_timeout * template.get("timeout_multiplier", 2.5),
                success_criteria={
                    "transfer_success": True,
                    "target_accuracy": max(0.5, domain_similarity * 0.8),
                    "adaptation_speed": adaptation_steps * 2,
                    "min_overlap": template.get("min_overlap", 0.3),
                },
                safety_constraints=[
                    Constraint(
                        "domain_constraint", "input", [source_domain, target_domain]
                    ),
                    Constraint("adaptation_limit", "resource", adaptation_steps * 2),
                    Constraint("memory_limit", "memory", 1024 * 1024 * 1024),
                ],
                experiment_type=ExperimentType.TRANSFER,
                parameters={
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "strategy": strategy,
                    "adaptation_steps": adaptation_steps,
                    "regularization": regularization,
                    "domain_similarity": domain_similarity,
                    "transfer_layers": "all" if domain_similarity > 0.7 else "top",
                    "freeze_base": strategy == "fine_tune",
                },
                metadata={
                    "expected_transfer_rate": domain_similarity,
                    "domain_distance": 1 - domain_similarity,
                },
            )
        except Exception as e:
            logger.error("Error building transfer experiment: %s", e)
            raise

    def build_synthetic_experiment(
        self, gap: KnowledgeGap, noise_level: float = 0.1
    ) -> Experiment:
        """Build synthetic test case"""
        try:
            # Generate synthetic data
            synthetic_data = self.synthetic_generator.generate(gap, noise_level)

            # Create experiment
            return Experiment(
                gap=gap,
                complexity=0.3,
                timeout=self.default_timeout * 0.5,
                success_criteria={
                    "synthetic_valid": True,
                    "coverage": 0.8,
                    "min_accuracy": 0.9,
                },
                safety_constraints=[
                    Constraint("synthetic_only", "input", True),
                    Constraint("deterministic", "resource", True),
                ],
                experiment_type=ExperimentType.SYNTHETIC,
                parameters={
                    "synthetic_data": synthetic_data,
                    "noise_level": noise_level,
                    "variations": 10,
                    "seed": synthetic_data["metadata"].get("seed", 42),
                },
                metadata={"is_synthetic": True, "generator_version": "1.0"},
            )
        except Exception as e:
            logger.error("Error building synthetic experiment: %s", e)
            raise

    def design_intervention(self, gap: KnowledgeGap) -> Dict[str, Any]:
        """Design intervention for causal experiment"""
        try:
            target_var = gap.metadata.get("target_variable", "unknown")

            return {
                "type": "synthetic",
                "variable": target_var,
                "values": np.linspace(0, 1, 10).tolist(),
                "duration": 10,
                "control_group": True,
                "randomization": "stratified",
                "washout_period": 2,
            }
        except Exception as e:
            logger.error("Error designing intervention: %s", e)
            return {"type": "default", "variable": "unknown"}


class ExperimentGenerator:
    """Creates targeted experiments for gaps - REFACTORED"""

    def __init__(
        self,
        default_timeout: float = 30.0,
        max_complexity: float = 1.0,
        max_history: int = 1000,
    ):
        """
        Initialize experiment generator

        Args:
            default_timeout: Default timeout for experiments
            max_complexity: Maximum complexity allowed
            max_history: Maximum history size
        """
        self.default_timeout = default_timeout
        self.max_complexity = max_complexity

        # Components
        self.builder = ExperimentBuilder(default_timeout, max_complexity)
        self.cache = ExperimentCache()
        self.tracker = ExperimentTracker(max_history)
        self.templates = ExperimentTemplates()

        # Statistics
        self.experiments_generated = 0
        self.generation_history = deque(maxlen=max_history)

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ExperimentGenerator initialized (refactored)")

        try:
            from vulcan.safety import SafetyValidator
        except ImportError:
            SafetyValidator = MagicMock()
            logger.warning("Using mock SafetyValidator")
        self.safety_validator = SafetyValidator()
        logger.info("ExperimentGenerator initialized")

    def generate_for_gap(
        self, gap: KnowledgeGap, use_cache: bool = True
    ) -> List[Experiment]:
        """Generate experiments for a knowledge gap - REFACTORED"""

        with self.lock:
            try:
                # EXAMINE: Check cache
                if use_cache:
                    cached = self.cache.get(gap.id, gap.type)
                    if cached:
                        return cached

                # SELECT: Route to appropriate generator
                experiments = []

                if gap.type == "decomposition":
                    experiments.extend(
                        self.generate_decomposition_experiment(gap, gap.complexity)
                    )
                elif gap.type == "causal":
                    experiments.extend(self.generate_causal_experiment(gap))
                elif gap.type == "transfer":
                    experiments.extend(self.generate_transfer_experiment(gap))
                elif gap.type == "latent":
                    experiments.extend(self.generate_latent_experiment(gap))
                elif gap.type == "semantic":
                    experiments.extend(self.generate_semantic_experiment(gap))
                else:
                    experiments.append(self._generate_exploratory_experiment(gap))

                # APPLY: Validate all experiments
                validated_experiments = []
                for exp in experiments:
                    valid, violations = exp.validate_constraints()
                    if valid:
                        validated_experiments.append(exp)
                    else:
                        logger.warning("Experiment validation failed: %s", violations)

                experiments = validated_experiments

                # REMEMBER: Update statistics and cache
                self.experiments_generated += len(experiments)

                self.generation_history.append(
                    {
                        "gap_id": gap.id,
                        "gap_type": gap.type,
                        "experiments_count": len(experiments),
                        "timestamp": time.time(),
                    }
                )

                if use_cache and experiments:
                    self.cache.put(gap.id, gap.type, experiments)

                return experiments
            except Exception as e:
                logger.error("Error generating experiments: %s", e)
                return []

    def generate_decomposition_experiment(
        self, gap: KnowledgeGap, complexity: float
    ) -> List[Experiment]:
        """Generate decomposition experiments - DELEGATED"""
        try:
            experiments = []

            # Get template
            template = self.templates.get_template("decomposition")

            # Generate experiments at different complexity levels
            complexity_levels = [
                complexity * 0.5,  # Simple
                complexity,  # Normal
                min(complexity * 1.5, self.max_complexity),  # Complex
            ]

            strategies = template.get(
                "strategies", ["hierarchical", "functional", "modular"]
            )

            for i, comp_level in enumerate(complexity_levels):
                if comp_level > self.max_complexity:
                    continue

                # Select strategy
                strategy = strategies[i % len(strategies)]

                # Build experiment
                experiment = self.builder.build_decomposition_experiment(
                    gap, comp_level, strategy, i
                )
                experiments.append(experiment)

            return experiments
        except Exception as e:
            logger.error("Error generating decomposition experiments: %s", e)
            return []

    def generate_causal_experiment(
        self, gap: KnowledgeGap, intervention: Optional[Dict[str, Any]] = None
    ) -> List[Experiment]:
        """Generate causal experiments - DELEGATED"""
        try:
            experiments = []

            # Get template
            template = self.templates.get_template("causal")

            # Default intervention if not provided
            if not intervention:
                intervention = self.builder.design_intervention(gap)

            # Generate different intervention strategies
            strategies = template.get(
                "strategies", ["direct", "instrumental", "natural"]
            )

            for strategy in strategies:
                experiment = self.builder.build_causal_experiment(
                    gap, strategy, intervention
                )
                experiments.append(experiment)

            return experiments
        except Exception as e:
            logger.error("Error generating causal experiments: %s", e)
            return []

    def generate_transfer_experiment(self, gap: KnowledgeGap) -> List[Experiment]:
        """Generate transfer learning experiments - DELEGATED"""
        try:
            experiments = []

            # Get template
            template = self.templates.get_template("transfer")

            # Generate experiments with different transfer strategies
            strategies = template.get(
                "strategies", ["direct", "fine_tune", "meta_learning"]
            )

            for strategy in strategies:
                experiment = self.builder.build_transfer_experiment(gap, strategy)
                experiments.append(experiment)

            return experiments
        except Exception as e:
            logger.error("Error generating transfer experiments: %s", e)
            return []

    def generate_latent_experiment(self, gap: KnowledgeGap) -> List[Experiment]:
        """Generate experiments for latent gaps"""
        try:
            experiments = []

            # Latent gaps require exploration
            strategies = ["clustering", "anomaly_detection", "representation_learning"]

            for strategy in strategies:
                experiment = Experiment(
                    gap=gap,
                    complexity=0.7,
                    timeout=self.default_timeout * 1.5,
                    success_criteria={
                        "pattern_found": True,
                        "min_confidence": 0.5,
                        "min_support": 0.1,
                    },
                    safety_constraints=[
                        Constraint("exploration_depth", "resource", 100),
                        Constraint("memory_limit", "memory", 512 * 1024 * 1024),
                    ],
                    experiment_type=ExperimentType.EXPLORATORY,
                    parameters={
                        "strategy": strategy,
                        "exploration_depth": 50,
                        "min_cluster_size": 5 if strategy == "clustering" else None,
                        "contamination": 0.1
                        if strategy == "anomaly_detection"
                        else None,
                        "embedding_dim": 32
                        if strategy == "representation_learning"
                        else None,
                    },
                )
                experiments.append(experiment)

            return experiments
        except Exception as e:
            logger.error("Error generating latent experiments: %s", e)
            return []

    def generate_semantic_experiment(self, gap: KnowledgeGap) -> List[Experiment]:
        """Generate experiments for semantic gaps"""
        try:
            experiments = []

            concept = gap.metadata.get("concept", "unknown")

            experiment = Experiment(
                gap=gap,
                complexity=0.5,
                timeout=self.default_timeout,
                success_criteria={"concept_learned": True, "min_similarity": 0.7},
                safety_constraints=[Constraint("concept_space", "resource", 1000)],
                experiment_type=ExperimentType.EXPLORATORY,
                parameters={
                    "target_concept": concept,
                    "learning_strategy": "embedding",
                    "similarity_threshold": 0.7,
                },
            )
            experiments.append(experiment)

            return experiments
        except Exception as e:
            logger.error("Error generating semantic experiments: %s", e)
            return []

    def create_synthetic_test_case(
        self, gap: KnowledgeGap, noise_level: float = 0.1
    ) -> Experiment:
        """Create synthetic test case for gap - DELEGATED"""
        return self.builder.build_synthetic_experiment(gap, noise_level)

    def track_experiment(self, experiment: Experiment):
        """Track active experiment - DELEGATED"""
        self.tracker.track_active(experiment)

    def complete_experiment(self, experiment_id: str, result: Any):
        """Mark experiment as completed - DELEGATED"""
        self.tracker.complete(experiment_id, result)

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics - DELEGATED"""
        try:
            tracker_stats = self.tracker.get_statistics()

            return {
                "total_generated": self.experiments_generated,
                "cache_size": len(self.cache.cache),
                **tracker_stats,
            }
        except Exception as e:
            logger.error("Error getting statistics: %s", e)
            return {}

    def _generate_exploratory_experiment(self, gap: KnowledgeGap) -> Experiment:
        """Generate generic exploratory experiment"""
        return Experiment(
            gap=gap,
            complexity=0.5,
            timeout=self.default_timeout,
            success_criteria={"discovery": True, "novelty": 0.3, "min_confidence": 0.5},
            safety_constraints=[
                Constraint("exploration_limit", "time", self.default_timeout),
                Constraint("memory_limit", "memory", 256 * 1024 * 1024),
            ],
            experiment_type=ExperimentType.EXPLORATORY,
            parameters={
                "exploration_strategy": "guided",
                "num_trials": 20,
                "early_stopping": True,
                "convergence_threshold": 0.01,
            },
            metadata={"fallback": True, "reason": "Unknown gap type"},
        )


class FailureAnalyzer:
    """Analyzes experiment failures - SEPARATED CONCERN"""

    def classify_failure(self, result: Any) -> FailureType:
        """Classify failure type from result"""
        try:
            # Handle dict results
            if isinstance(result, dict):
                error = result.get("error", "").lower()

                if "timeout" in error:
                    return FailureType.TIMEOUT
                elif "memory" in error or "resource" in error:
                    return FailureType.RESOURCE_EXCEEDED
                elif "constraint" in error:
                    return FailureType.CONSTRAINT_VIOLATION
                elif "data" in error or "sample" in error:
                    return FailureType.INSUFFICIENT_DATA

                if result.get("variance", 0) > 0.5:
                    return FailureType.UNSTABLE_OUTPUT

                if result.get("accuracy", 1.0) < 0.3:
                    return FailureType.WRONG_APPROACH

                if result.get("converged", True) is False:
                    return FailureType.CONVERGENCE_FAILURE

            # Handle object results
            if hasattr(result, "error"):
                error = str(result.error).lower()

                if "timeout" in error:
                    return FailureType.TIMEOUT
                elif "memory" in error or "resource" in error:
                    return FailureType.RESOURCE_EXCEEDED
                elif "constraint" in error:
                    return FailureType.CONSTRAINT_VIOLATION
                elif "data" in error or "sample" in error:
                    return FailureType.INSUFFICIENT_DATA

            if hasattr(result, "variance") and result.variance > 0.5:
                return FailureType.UNSTABLE_OUTPUT

            if hasattr(result, "accuracy") and result.accuracy < 0.3:
                return FailureType.WRONG_APPROACH

            if hasattr(result, "converged") and not result.converged:
                return FailureType.CONVERGENCE_FAILURE

            return FailureType.TOO_SIMPLE
        except Exception as e:
            logger.error("Error classifying failure: %s", e)
            return FailureType.TOO_SIMPLE

    def extract_details(self, result: Any) -> Dict[str, Any]:
        """Extract details from failure"""
        try:
            details = {}

            if isinstance(result, dict):
                details.update(result)
            elif hasattr(result, "__dict__"):
                details.update(result.__dict__)

            # Add computed metrics
            if "output" in details and "expected" in details:
                output = details["output"]
                expected = details["expected"]

                if isinstance(output, (int, float)) and isinstance(
                    expected, (int, float)
                ):
                    details["error_magnitude"] = abs(output - expected)
                    details["relative_error"] = abs(output - expected) / max(
                        abs(expected), 1e-10
                    )

            return details
        except Exception as e:
            logger.error("Error extracting details: %s", e)
            return {}

    def generate_adjustments(
        self, failure_type: FailureType, details: Dict[str, Any]
    ) -> List[str]:
        """Generate adjustment suggestions"""
        try:
            adjustments = []

            if failure_type == FailureType.TOO_SIMPLE:
                adjustments.append("complexity:*1.5")
                adjustments.append("depth:+2")
                adjustments.append("features:expand")

            elif failure_type == FailureType.TIMEOUT:
                adjustments.append("timeout:*2.0")
                adjustments.append("early_stopping:True")
                adjustments.append("batch_processing:True")

            elif failure_type == FailureType.INSUFFICIENT_DATA:
                adjustments.append("sample_size:*2.0")
                adjustments.append("data_augmentation:True")
                adjustments.append("synthetic_ratio:0.3")

            elif failure_type == FailureType.UNSTABLE_OUTPUT:
                adjustments.append("regularization:*2.0")
                adjustments.append("dropout:0.2")
                adjustments.append("ensemble:True")

            elif failure_type == FailureType.CONVERGENCE_FAILURE:
                adjustments.append("learning_rate:*0.5")
                adjustments.append("max_iterations:*2")
                adjustments.append("optimizer:adam")

            # Add detail-specific adjustments
            if details.get("error_magnitude", 0) > 1.0:
                adjustments.append("scale_inputs:True")

            if details.get("relative_error", 0) > 0.5:
                adjustments.append("precision:double")

            return adjustments
        except Exception as e:
            logger.error("Error generating adjustments: %s", e)
            return []


class ParameterAdjuster:
    """Adjusts experiment parameters - SEPARATED CONCERN"""

    def __init__(self):
        self.lock = threading.RLock()

    def adjust_for_failure(
        self, params: Dict[str, Any], failure_type: FailureType
    ) -> Dict[str, Any]:
        """Adjust parameters based on failure type"""
        with self.lock:
            try:
                adjusted = copy.deepcopy(params)

                if failure_type == FailureType.TOO_SIMPLE:
                    adjusted["complexity"] = min(
                        1.0, adjusted.get("complexity", 0.5) * 1.5
                    )
                    if "depth" in adjusted:
                        adjusted["depth"] = adjusted["depth"] + 1
                    if "sample_size" in adjusted:
                        adjusted["sample_size"] = int(adjusted["sample_size"] * 1.2)

                elif failure_type == FailureType.WRONG_APPROACH:
                    # Change strategy
                    strategies = adjusted.get("available_strategies", ["default"])
                    current = adjusted.get("strategy", "default")

                    if current in strategies:
                        strategies = [s for s in strategies if s != current]

                    if strategies:
                        adjusted["strategy"] = strategies[0]
                        adjusted["available_strategies"] = strategies
                    else:
                        adjusted["available_strategies"] = [
                            "hierarchical",
                            "functional",
                            "modular",
                        ]
                        adjusted["strategy"] = "hierarchical"

                elif failure_type == FailureType.TIMEOUT:
                    adjusted["timeout"] = adjusted.get("timeout", 30) * 1.5
                    adjusted["complexity"] = adjusted.get("complexity", 0.5) * 0.8
                    adjusted["early_stopping"] = True

                elif failure_type == FailureType.CONSTRAINT_VIOLATION:
                    adjusted["sample_size"] = int(
                        adjusted.get("sample_size", 100) * 0.7
                    )
                    adjusted["iterations"] = int(adjusted.get("iterations", 100) * 0.7)
                    adjusted["batch_size"] = int(adjusted.get("batch_size", 32) * 0.5)

                elif failure_type == FailureType.INSUFFICIENT_DATA:
                    adjusted["sample_size"] = int(
                        adjusted.get("sample_size", 100) * 1.5
                    )
                    adjusted["data_augmentation"] = True
                    adjusted["synthetic_ratio"] = (
                        adjusted.get("synthetic_ratio", 0.0) + 0.2
                    )

                elif failure_type == FailureType.UNSTABLE_OUTPUT:
                    adjusted["regularization"] = adjusted.get("regularization", 0.1) * 2
                    adjusted["noise_reduction"] = True
                    adjusted["ensemble_size"] = adjusted.get("ensemble_size", 1) + 2

                elif failure_type == FailureType.CONVERGENCE_FAILURE:
                    adjusted["learning_rate"] = (
                        adjusted.get("learning_rate", 0.01) * 0.5
                    )
                    adjusted["max_iterations"] = int(
                        adjusted.get("max_iterations", 100) * 1.5
                    )
                    adjusted["convergence_threshold"] = (
                        adjusted.get("convergence_threshold", 0.01) * 2
                    )

                return adjusted
            except Exception as e:
                logger.error("Error adjusting parameters: %s", e)
                return params


class IterativeExperimentDesigner:
    """Designs experiments that learn from failures - REFACTORED"""

    def __init__(self, max_iterations: int = 5):
        """
        Initialize iterative designer

        Args:
            max_iterations: Maximum iterations per gap
        """
        self.max_iterations = max_iterations

        # Components
        self.failure_analyzer = FailureAnalyzer()
        self.parameter_adjuster = ParameterAdjuster()

        # Learning storage
        self.failure_patterns = defaultdict(lambda: deque(maxlen=100))
        self.successful_adjustments = defaultdict(lambda: deque(maxlen=50))
        self.adjustment_effectiveness = defaultdict(float)

        # Statistics
        self.iteration_history = deque(maxlen=1000)
        self.adaptation_success_rate = deque(maxlen=100)

        # Thread safety
        self.lock = threading.RLock()

        logger.info("IterativeExperimentDesigner initialized (refactored)")

    def generate_iterative_experiments(
        self, gap: KnowledgeGap, max_iterations: Optional[int] = None
    ) -> List[Experiment]:
        """Generate iterative experiments that adapt - REFACTORED"""

        with self.lock:
            try:
                if max_iterations is None:
                    max_iterations = self.max_iterations

                experiments = []
                current_params = self._get_initial_parameters(gap)

                for iteration in range(min(max_iterations, self.max_iterations)):
                    # EXAMINE & SELECT: Apply learned adjustments
                    current_params = self._apply_learned_adjustments(
                        current_params, gap.type
                    )

                    # APPLY: Create experiment with current parameters
                    experiment = Experiment(
                        gap=gap,
                        complexity=current_params["complexity"],
                        timeout=current_params["timeout"],
                        success_criteria=current_params["success_criteria"],
                        safety_constraints=self._get_safety_constraints(iteration),
                        experiment_type=ExperimentType.ITERATIVE,
                        parameters=current_params,
                        iteration=iteration,
                        metadata={
                            "is_iterative": True,
                            "max_iterations": max_iterations,
                        },
                    )

                    experiments.append(experiment)

                    # Prepare next iteration parameters
                    current_params = self._adjust_for_next_iteration(
                        current_params, iteration
                    )

                # REMEMBER: Track iteration
                self.iteration_history.append(
                    {
                        "gap_id": gap.id,
                        "gap_type": gap.type,
                        "iterations": len(experiments),
                        "timestamp": time.time(),
                    }
                )

                return experiments
            except Exception as e:
                logger.error("Error generating iterative experiments: %s", e)
                return []

    def analyze_failure(
        self, result: Any, experiment: Optional[Experiment] = None
    ) -> FailureAnalysis:
        """Analyze experiment failure - REFACTORED"""

        with self.lock:
            try:
                # EXAMINE: Classify and extract details
                failure_type = self.failure_analyzer.classify_failure(result)
                details = self.failure_analyzer.extract_details(result)

                # Add experiment context if available
                if experiment:
                    details["experiment_type"] = experiment.experiment_type.value
                    details["iteration"] = experiment.iteration
                    details["gap_type"] = experiment.gap.type

                # SELECT: Generate adjustments
                adjustments = self.failure_analyzer.generate_adjustments(
                    failure_type, details
                )

                # Determine recovery possibility
                recovery_possible = self._is_recovery_possible(failure_type, details)

                # Identify root cause
                root_cause = self._identify_root_cause(failure_type)

                # APPLY: Create analysis
                analysis = FailureAnalysis(
                    type=failure_type,
                    details=details,
                    suggested_adjustments=adjustments,
                    recovery_possible=recovery_possible,
                    confidence=self._calculate_confidence(details),
                    root_cause=root_cause,
                    experiment_id=experiment.experiment_id if experiment else None,
                )

                # REMEMBER: Track failure pattern
                self.failure_patterns[failure_type].append(analysis)
                self._learn_from_failure(analysis)

                return analysis
            except Exception as e:
                logger.error("Error analyzing failure: %s", e)
                return FailureAnalysis(type=FailureType.TOO_SIMPLE)

    def adjust_experiment_parameters(
        self, params: Dict[str, Any], failure_analysis: FailureAnalysis
    ) -> Dict[str, Any]:
        """Adjust parameters based on failure analysis - DELEGATED"""

        with self.lock:
            try:
                # Use parameter adjuster
                adjusted = self.parameter_adjuster.adjust_for_failure(
                    params, failure_analysis.type
                )

                # Apply suggested adjustments
                for adjustment in failure_analysis.suggested_adjustments:
                    key, value, operation = self._parse_adjustment(adjustment)
                    if key and value is not None:
                        if operation == "multiply" and key in adjusted:
                            adjusted[key] = adjusted[key] * value
                        elif operation == "add" and key in adjusted:
                            adjusted[key] = adjusted[key] + value
                        elif operation == "set":
                            adjusted[key] = value

                # Track adjustment effectiveness
                adjustment_key = f"{failure_analysis.type.value}_adjustments"
                self.successful_adjustments[adjustment_key].append(adjusted)

                return adjusted
            except Exception as e:
                logger.error("Error adjusting parameters: %s", e)
                return params

    def pivot_experiment_strategy(
        self, experiment: Experiment, failure_analysis: FailureAnalysis
    ) -> Experiment:
        """Pivot to different experimental strategy - REFACTORED"""

        with self.lock:
            try:
                # Create new experiment with different approach
                pivoted = copy.deepcopy(experiment)
                pivoted.iteration += 1

                # Change experiment type based on failure
                if failure_analysis.type == FailureType.WRONG_APPROACH:
                    if pivoted.experiment_type == ExperimentType.DECOMPOSITION:
                        pivoted.experiment_type = ExperimentType.SYNTHETIC
                        pivoted.parameters["strategy"] = "synthetic_decomposition"
                    elif pivoted.experiment_type == ExperimentType.CAUSAL:
                        pivoted.experiment_type = ExperimentType.EXPLORATORY
                        pivoted.parameters["strategy"] = "correlation_exploration"
                    elif pivoted.experiment_type == ExperimentType.TRANSFER:
                        pivoted.experiment_type = ExperimentType.ITERATIVE
                        pivoted.parameters["strategy"] = "progressive_transfer"

                # Adjust parameters
                pivoted.parameters = self.adjust_experiment_parameters(
                    pivoted.parameters, failure_analysis
                )

                # Update success criteria to be more lenient
                for criterion in list(pivoted.success_criteria.keys()):
                    value = pivoted.success_criteria[criterion]
                    if isinstance(value, (int, float)):
                        if "min" in criterion:
                            pivoted.success_criteria[criterion] = value * 0.8
                        elif "max" in criterion:
                            pivoted.success_criteria[criterion] = value * 1.2

                # Track successful pivots
                if failure_analysis.recovery_possible:
                    self.successful_adjustments[failure_analysis.type].append(
                        {
                            "original": experiment.experiment_type.value,
                            "pivoted": pivoted.experiment_type.value,
                            "adjustment": failure_analysis.suggested_adjustments,
                        }
                    )

                    self.adaptation_success_rate.append(1.0)
                else:
                    self.adaptation_success_rate.append(0.0)

                return pivoted
            except Exception as e:
                logger.error("Error pivoting experiment: %s", e)
                return experiment

    def _get_initial_parameters(self, gap: KnowledgeGap) -> Dict[str, Any]:
        """Get initial parameters for gap"""
        params = {
            "complexity": gap.complexity,
            "timeout": 30,
            "strategy": "default",
            "sample_size": 100,
            "iterations": 100,
            "learning_rate": 0.01,
            "batch_size": 32,
            "success_criteria": {
                "min_improvement": 0.1,
                "convergence": 0.01,
                "min_confidence": 0.6,
            },
        }

        # Adjust based on gap type
        if gap.type == "causal":
            params["sample_size"] = 200
            params["strategy"] = "intervention"
            params["bootstrap_iterations"] = 100
        elif gap.type == "decomposition":
            params["complexity"] *= 0.8
            params["strategy"] = "hierarchical"
            params["depth"] = 3
        elif gap.type == "transfer":
            params["strategy"] = "gradual"
            params["adaptation_steps"] = 50
        elif gap.type == "latent":
            params["strategy"] = "exploration"
            params["exploration_depth"] = 100

        return params

    def _get_safety_constraints(self, iteration: int) -> List[Constraint]:
        """Get safety constraints for iteration"""
        # More conservative constraints for later iterations
        memory_limit = 512 * 1024 * 1024 * (1 + iteration * 0.2)
        time_limit = 30 * (1 + iteration * 0.5)

        constraints = [
            Constraint("memory", "memory", memory_limit, severity=0.8),
            Constraint("time", "time", time_limit, severity=0.9),
            Constraint("iteration", "input", iteration, action="warn"),
        ]

        # Add stricter constraints for later iterations
        if iteration > 2:
            constraints.append(
                Constraint(
                    "convergence", "resource", True, action="abort", severity=1.0
                )
            )

        return constraints

    def _adjust_for_next_iteration(
        self, params: Dict[str, Any], iteration: int
    ) -> Dict[str, Any]:
        """Adjust parameters for next iteration"""
        adjusted = copy.deepcopy(params)

        # Increase complexity gradually
        adjusted["complexity"] = min(
            1.0, adjusted["complexity"] * (1.1 + iteration * 0.05)
        )

        # Increase timeout
        adjusted["timeout"] = adjusted["timeout"] * (1.1 + iteration * 0.1)

        # Adjust success criteria
        success_criteria = adjusted.get("success_criteria", {})
        if "min_improvement" in success_criteria:
            success_criteria["min_improvement"] *= 0.9
        if "convergence" in success_criteria:
            success_criteria["convergence"] *= 1.1
        adjusted["success_criteria"] = success_criteria

        # Adjust learning parameters
        if "learning_rate" in adjusted:
            adjusted["learning_rate"] *= 0.95  # Decay learning rate

        return adjusted

    def _is_recovery_possible(
        self, failure_type: FailureType, details: Dict[str, Any]
    ) -> bool:
        """Determine if recovery is possible"""
        # Some failures are harder to recover from
        if failure_type == FailureType.RESOURCE_EXCEEDED:
            return details.get("resource_usage", 0) < 0.9

        if failure_type == FailureType.WRONG_APPROACH:
            return details.get("iteration", 0) < 3

        if failure_type == FailureType.VALIDATION_FAILURE:
            return False

        # Most failures can be recovered from
        return True

    def _identify_root_cause(self, failure_type: FailureType) -> Optional[str]:
        """Identify root cause of failure"""
        root_causes = {
            FailureType.TOO_SIMPLE: "insufficient_model_capacity",
            FailureType.TIMEOUT: "computational_complexity",
            FailureType.INSUFFICIENT_DATA: "data_scarcity",
            FailureType.UNSTABLE_OUTPUT: "high_variance",
            FailureType.CONVERGENCE_FAILURE: "optimization_difficulty",
            FailureType.WRONG_APPROACH: "strategy_mismatch",
        }

        return root_causes.get(failure_type)

    def _calculate_confidence(self, details: Dict[str, Any]) -> float:
        """Calculate confidence in failure analysis"""
        # Base confidence
        confidence = 0.5

        # Increase confidence with more evidence
        if "error" in details:
            confidence += 0.2
        if "variance" in details:
            confidence += 0.1
        if "iteration" in details and details["iteration"] > 0:
            confidence += 0.1
        if "error_magnitude" in details:
            confidence += 0.1

        return min(1.0, confidence)

    def _parse_adjustment(self, adjustment: str) -> Tuple[Optional[str], Any, str]:
        """
        Parse adjustment string into key-value-operation tuple

        Returns:
            Tuple of (key, value, operation) where operation is 'set', 'multiply', or 'add'
        """
        try:
            if ":" not in adjustment:
                return None, None, "set"

            key, value_str = adjustment.split(":", 1)
            key = key.strip()
            value_str = value_str.strip()

            # FIX: Properly handle multiplication and addition
            if value_str.startswith("*"):
                # Multiplication
                factor = float(value_str[1:].strip())
                return key, factor, "multiply"
            elif value_str.startswith("+"):
                # Addition
                amount = float(value_str[1:].strip())
                return key, amount, "add"
            elif value_str == "True":
                return key, True, "set"
            elif value_str == "False":
                return key, False, "set"
            else:
                # Try to evaluate as number
                try:
                    value = float(value_str)
                    if value == int(value):
                        value = int(value)
                    return key, value, "set"
                except Exception:
                    return key, value_str, "set"

        except Exception as e:
            logger.warning("Failed to parse adjustment '%s': %s", adjustment, e)
            return None, None, "set"

    def _apply_learned_adjustments(
        self, params: Dict[str, Any], gap_type: str
    ) -> Dict[str, Any]:
        """Apply adjustments learned from past failures"""
        try:
            # Check if we have successful adjustments for this gap type
            for failure_type in FailureType:
                key = f"{failure_type.value}_adjustments"
                if (
                    key in self.successful_adjustments
                    and self.successful_adjustments[key]
                ):
                    # Get most recent successful adjustment
                    recent_adjustments = [self.successful_adjustments[key])[-5:]

                    # Apply successful patterns
                    for adj in recent_adjustments:
                        if isinstance(adj, dict) and adj.get("gap_type") == gap_type:
                            # Apply some of the successful adjustments
                            for param_key, param_value in adj.items():
                                if param_key not in ["gap_type", "timestamp"]:
                                    # Apply with some probability based on effectiveness
                                    effectiveness = self.adjustment_effectiveness.get(
                                        param_key, 0.5
                                    )
                                    if np.random.random() < effectiveness:
                                        params[param_key] = param_value

            return params
        except Exception as e:
            logger.warning("Error applying learned adjustments: %s", e)
            return params

    def _learn_from_failure(self, analysis: FailureAnalysis):
        """Learn from failure analysis"""
        try:
            # Update adjustment effectiveness based on failure patterns
            for adjustment in analysis.suggested_adjustments:
                key, _, _ = self._parse_adjustment(adjustment)
                if key:
                    # Decay effectiveness of this adjustment type
                    current = self.adjustment_effectiveness.get(key, 0.5)
                    self.adjustment_effectiveness[key] = current * 0.95
        except Exception as e:
            logger.warning("Error learning from failure: %s", e)
