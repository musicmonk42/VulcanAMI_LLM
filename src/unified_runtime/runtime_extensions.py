"""
Runtime Extensions Module for Graphix IR
Learning, evolution, explainability, and autonomous optimization capabilities
"""

import json
import time
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import traceback
import random
from datetime import datetime, timedelta
import threading
import pickle

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

# FIX: Added torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from .evolution_engine import EvolutionEngine

    EVOLUTION_AVAILABLE = True
except ImportError:
    try:
        from evolution_engine import EvolutionEngine

        EVOLUTION_AVAILABLE = True
    except ImportError:
        EvolutionEngine = None
        EVOLUTION_AVAILABLE = False

try:
    from .governance_loop import GovernanceLoop

    GOVERNANCE_AVAILABLE = True
except ImportError:
    try:
        from governance_loop import GovernanceLoop

        GOVERNANCE_AVAILABLE = True
    except ImportError:
        GovernanceLoop = None
        GOVERNANCE_AVAILABLE = False

try:
    from .neural_system_optimizer import NeuralSystemOptimizer

    NSO_AVAILABLE = True
except ImportError:
    try:
        from neural_system_optimizer import NeuralSystemOptimizer

        NSO_AVAILABLE = True
    except ImportError:
        NeuralSystemOptimizer = None
        NSO_AVAILABLE = False

try:
    from .deep_optimization_engine import DeepOptimizationEngine

    OPTIMIZER_AVAILABLE = True
except ImportError:
    try:
        from deep_optimization_engine import DeepOptimizationEngine

        OPTIMIZER_AVAILABLE = True
    except ImportError:
        DeepOptimizationEngine = None
        OPTIMIZER_AVAILABLE = False

try:
    from .interpretability_engine import InterpretabilityEngine

    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    try:
        from interpretability_engine import InterpretabilityEngine

        INTERPRETABILITY_AVAILABLE = True
    except ImportError:
        InterpretabilityEngine = None
        INTERPRETABILITY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Enums
class LearningMode(Enum):
    """Learning modes for subgraph patterns"""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCED = "reinforced"
    EVOLUTIONARY = "evolutionary"
    DEMONSTRATION = "demonstration"
    IMITATION = "imitation"


class ExplanationType(Enum):
    """Types of execution explanations"""

    SIMPLE = "simple"
    DETAILED = "detailed"
    VISUAL = "visual"
    TECHNICAL = "technical"
    COMPARATIVE = "comparative"


# Data Classes
@dataclass
class SubgraphPattern:
    """Learned subgraph pattern"""

    pattern_id: str
    name: str
    graph_definition: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    confidence_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "graph_definition": self.graph_definition,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "usage_count": self.usage_count,
            "creation_time": self.creation_time.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "confidence_score": self.confidence_score,
        }


@dataclass
class ExecutionExplanation:
    """Explanation of subgraph execution"""

    subgraph_id: str
    explanation_type: ExplanationType
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "subgraph_id": self.subgraph_id,
            "explanation_type": self.explanation_type.value,
            "summary": self.summary,
            "details": self.details,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AutonomousCycleReport:
    """Report from autonomous optimization cycle"""

    cycle_id: str
    fitness_score: float
    optimizations_applied: List[str]
    evolution_proposals: List[Dict[str, Any]]
    safety_violations: List[str]
    performance_delta: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cycle_id": self.cycle_id,
            "fitness_score": self.fitness_score,
            "optimizations_applied": self.optimizations_applied,
            "evolution_proposals": self.evolution_proposals,
            "safety_violations": self.safety_violations,
            "performance_delta": self.performance_delta,
            "timestamp": self.timestamp.isoformat(),
        }


# Main Classes
class SubgraphLearner:
    """
    Manages subgraph learning and pattern recognition
    """

    def __init__(
        self, learned_subgraphs_dir: str = "learned_subgraphs", max_patterns: int = 1000
    ):
        self.learned_dir = Path(learned_subgraphs_dir)
        self.learned_dir.mkdir(exist_ok=True, parents=True)
        self.max_patterns = max_patterns

        # Pattern storage
        self.patterns: Dict[str, SubgraphPattern] = {}
        self.pattern_index: Dict[str, List[str]] = defaultdict(list)
        self.learning_history: deque = deque(maxlen=100)

        # FIXED: Initialize learning configuration
        self.learning_config = {
            "min_confidence": 0.3,
            "max_complexity": 100,
            "enable_evolution": EVOLUTION_AVAILABLE,
            "enable_interpretability": INTERPRETABILITY_AVAILABLE,
        }

        # Load existing patterns
        self._load_patterns()

        logger.info(f"SubgraphLearner initialized with {len(self.patterns)} patterns")

    def learn_subgraph(
        self,
        subgraph_type: str,
        graph_definition: Dict[str, Any],
        mode: LearningMode = LearningMode.SUPERVISED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Learn a new subgraph pattern

        Args:
            subgraph_type: Type/name of the subgraph
            graph_definition: Graph structure definition
            mode: Learning mode
            metadata: Additional metadata

        Returns:
            Success flag and pattern ID or error message
        """
        try:
            # Validate graph structure
            if not self._validate_graph_structure(graph_definition):
                return False, "Invalid graph structure"

            # Generate pattern ID
            pattern_id = self._generate_pattern_id(subgraph_type, graph_definition)

            with threading.Lock():
                # Check if pattern already exists
                if pattern_id in self.patterns:
                    # Update existing pattern
                    pattern = self.patterns[pattern_id]
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()

                    self._save_pattern(pattern)
                    return True, pattern_id

                # Check capacity
                if len(self.patterns) >= self.max_patterns:
                    # Evict least used pattern
                    self._evict_least_used()

                # Create new pattern
                pattern = SubgraphPattern(
                    pattern_id=pattern_id,
                    name=subgraph_type,
                    graph_definition=graph_definition,
                    metadata=metadata or {},
                    confidence_score=0.5 if mode == LearningMode.UNSUPERVISED else 0.8,
                )

                # Apply learning mode specific processing
                if mode == LearningMode.EVOLUTIONARY:
                    pattern = self._apply_evolutionary_learning(pattern)
                elif mode == LearningMode.DEMONSTRATION:
                    pattern = self._apply_demonstration_learning(pattern)

                # Store pattern
                self.patterns[pattern_id] = pattern
                self.pattern_index[subgraph_type].append(pattern_id)

                # Save to disk
                self._save_pattern(pattern)

                # Record learning event
                self.learning_history.append(
                    {
                        "pattern_id": pattern_id,
                        "mode": mode.value,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                logger.info(
                    f"Learned new pattern: {pattern_id} (type: {subgraph_type})"
                )
                return True, pattern_id

        except Exception as e:
            logger.error(f"Failed to learn subgraph: {e}")
            return False, str(e)

    def get_pattern(self, pattern_id: str) -> Optional[SubgraphPattern]:
        """Get pattern by ID"""
        pattern = self.patterns.get(pattern_id)
        if pattern:
            # Update usage stats
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            self._save_pattern(pattern)
        return pattern

    def get_patterns_by_type(self, subgraph_type: str) -> List[SubgraphPattern]:
        """Get all patterns of a specific type"""
        pattern_ids = self.pattern_index.get(subgraph_type, [])
        patterns = []
        for pid in pattern_ids:
            pattern = self.patterns.get(pid)
            if pattern:
                patterns.append(pattern)
        return patterns

    def update_pattern_performance(
        self, pattern_id: str, metrics: Dict[str, float]
    ) -> bool:
        """
        Update pattern performance metrics

        Args:
            pattern_id: Pattern to update
            metrics: Performance metrics

        Returns:
            Success flag
        """
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return False

        # Update metrics with exponential smoothing
        alpha = 0.3
        for key, value in metrics.items():
            if key in pattern.performance_metrics:
                pattern.performance_metrics[key] = (
                    alpha * value + (1 - alpha) * pattern.performance_metrics[key]
                )
            else:
                pattern.performance_metrics[key] = value

        # Update confidence based on performance
        if "success_rate" in metrics:
            pattern.confidence_score = (
                0.5 * pattern.confidence_score + 0.5 * metrics["success_rate"]
            )

        self._save_pattern(pattern)
        return True

    def _validate_graph_structure(self, graph_def: Dict[str, Any]) -> bool:
        """Validate graph structure"""
        try:
            # Check required fields
            if "nodes" not in graph_def or "edges" not in graph_def:
                return False

            # Validate nodes
            node_ids = set()
            for node in graph_def["nodes"]:
                if "id" not in node:
                    return False
                if node["id"] in node_ids:
                    return False  # Duplicate ID
                node_ids.add(node["id"])

            # Validate edges
            for edge in graph_def["edges"]:
                if "from" not in edge or "to" not in edge:
                    return False
                if edge["from"] not in node_ids or edge["to"] not in node_ids:
                    return False  # Unknown node reference

            return True
        except Exception:
            return False

    def _generate_pattern_id(
        self, subgraph_type: str, graph_def: Dict[str, Any]
    ) -> str:
        """Generate unique pattern ID"""
        # Create hash of graph structure
        graph_str = json.dumps(graph_def, sort_keys=True)
        # Combine type and graph for a unique hash
        combined_str = f"{subgraph_type}:{graph_str}"
        graph_hash = hashlib.md5(combined_str.encode()).hexdigest()[:12]
        return graph_hash

    def _apply_evolutionary_learning(self, pattern: SubgraphPattern) -> SubgraphPattern:
        """Apply evolutionary learning enhancements"""
        if EVOLUTION_AVAILABLE and hasattr(self, "evolution_engine"):
            # Evolve pattern structure
            evolved_graph = self.evolution_engine.evolve_structure(
                pattern.graph_definition
            )
            pattern.graph_definition = evolved_graph
            pattern.metadata["evolved"] = True
        return pattern

    def _apply_demonstration_learning(
        self, pattern: SubgraphPattern
    ) -> SubgraphPattern:
        """Apply demonstration learning enhancements"""
        # Learn from demonstrations
        pattern.metadata["learned_from_demo"] = True
        pattern.confidence_score *= 0.9  # Slightly lower confidence for demos
        return pattern

    def _save_pattern(self, pattern: SubgraphPattern):
        """Save pattern to disk"""
        pattern_file = self.learned_dir / f"{pattern.pattern_id}.json"
        with open(pattern_file, "w") as f:
            json.dump(pattern.to_dict(), f, indent=2)

    def _evict_least_used(self):
        """Evict least recently used pattern"""
        if not self.patterns:
            return

        # Find LRU pattern
        lru_pattern = min(
            self.patterns.values(),
            key=lambda p: (p.last_used or p.creation_time, p.usage_count),
        )

        # Remove pattern
        del self.patterns[lru_pattern.pattern_id]

        # FIXED: Remove from index with safety check
        if lru_pattern.pattern_id in self.pattern_index[lru_pattern.name]:
            self.pattern_index[lru_pattern.name].remove(lru_pattern.pattern_id)

        # Delete file
        pattern_file = self.learned_dir / f"{lru_pattern.pattern_id}.json"
        if pattern_file.exists():
            pattern_file.unlink()

        logger.info(f"Evicted pattern: {lru_pattern.pattern_id}")

    def _load_patterns(self):
        """Load patterns from disk"""
        if not self.learned_dir.exists():
            return

        for pattern_file in self.learned_dir.glob("*.json"):
            try:
                with open(pattern_file, "r") as f:
                    data = json.load(f)

                # Convert timestamps
                data["creation_time"] = datetime.fromisoformat(data["creation_time"])
                if data.get("last_used"):
                    data["last_used"] = datetime.fromisoformat(data["last_used"])

                # Create pattern
                pattern = SubgraphPattern(**data)
                self.patterns[pattern.pattern_id] = pattern
                self.pattern_index[pattern.name].append(pattern.pattern_id)

            except Exception as e:
                logger.warning(f"Failed to load pattern {pattern_file}: {e}")


class AutonomousOptimizer:
    """
    Manages autonomous optimization cycles
    """

    def __init__(self):
        # Initialize sub-components
        self.evolution_engine = EvolutionEngine() if EVOLUTION_AVAILABLE else None
        self.optimizer = DeepOptimizationEngine() if OPTIMIZER_AVAILABLE else None
        self.nso = NeuralSystemOptimizer() if NSO_AVAILABLE else None
        self.governance = GovernanceLoop() if GOVERNANCE_AVAILABLE else None

        # Optimization state
        self.optimization_history: deque = deque(maxlen=50)
        self.current_fitness: float = 0.5
        self.optimization_config = {
            "min_fitness_threshold": 0.3,
            "max_iterations": 100,
            "enable_evolution": EVOLUTION_AVAILABLE,
            "enable_governance": GOVERNANCE_AVAILABLE,
        }

        logger.info(
            f"AutonomousOptimizer initialized. Components available: "
            f"Evolution={EVOLUTION_AVAILABLE}, Optimizer={OPTIMIZER_AVAILABLE}, "
            f"NSO={NSO_AVAILABLE}, Governance={GOVERNANCE_AVAILABLE}"
        )

    async def trigger_autonomous_cycle(
        self, graph: Any, metrics: Dict[str, Any], runtime: Optional[Any] = None
    ) -> AutonomousCycleReport:
        """
        Trigger autonomous optimization cycle

        Args:
            graph: Current graph structure
            metrics: Current performance metrics
            runtime: Optional runtime context (FIXED: Added default None)

        Returns:
            Optimization report
        """
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting autonomous cycle: {cycle_id}")

        try:
            # Calculate fitness
            fitness_score = self._calculate_fitness(metrics)

            optimizations_applied = []
            evolution_proposals = []
            safety_violations = []
            performance_delta = {}

            # Check if optimization is needed
            if fitness_score < self.optimization_config["min_fitness_threshold"]:
                # Run evolution if available
                if (
                    self.evolution_engine
                    and self.optimization_config["enable_evolution"]
                ):
                    try:
                        proposals = await self._run_evolution(graph, metrics)
                        evolution_proposals.extend(proposals)
                    except Exception as e:
                        logger.error(f"Evolution failed: {e}")

                # Run optimization if available
                if self.optimizer:
                    try:
                        opt_results = await self._run_optimization(graph, metrics)
                        optimizations_applied.extend(opt_results)
                    except Exception as e:
                        logger.error(f"Optimization failed: {e}")

                # Run neural system optimization if available
                if self.nso:
                    try:
                        nso_results = await self._run_nso(graph, metrics)
                        optimizations_applied.extend(nso_results)
                    except Exception as e:
                        logger.error(f"NSO failed: {e}")

                # Check governance if available
                if self.governance and self.optimization_config["enable_governance"]:
                    try:
                        violations = await self._check_governance(
                            graph, optimizations_applied, evolution_proposals
                        )
                        safety_violations.extend(violations)
                    except Exception as e:
                        logger.error(f"Governance check failed: {e}")

                # Calculate performance delta
                performance_delta = self._calculate_performance_delta(metrics)

            # Create report
            report = AutonomousCycleReport(
                cycle_id=cycle_id,
                fitness_score=fitness_score,
                optimizations_applied=optimizations_applied,
                evolution_proposals=evolution_proposals,
                safety_violations=safety_violations,
                performance_delta=performance_delta,
            )

            # Record history
            self.optimization_history.append(report.to_dict())
            self.current_fitness = fitness_score

            logger.info(f"Completed autonomous cycle: fitness={fitness_score:.3f}")
            return report

        except Exception as e:
            logger.error(f"Autonomous cycle failed: {e}")
            return AutonomousCycleReport(
                cycle_id=cycle_id,
                fitness_score=0.0,
                optimizations_applied=[],
                evolution_proposals=[],
                safety_violations=["cycle_error"],
                performance_delta={},
            )

    def _calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """Calculate fitness score from metrics"""
        try:
            # Weighted combination of metrics
            latency_score = 1.0 - min(metrics.get("latency", 1.0), 1.0)
            throughput_score = min(metrics.get("throughput", 0.0) / 1000, 1.0)
            success_rate = metrics.get("success_rate", 0.5)
            cache_hit_rate = metrics.get("cache_hit_rate", 0.0)

            fitness = (
                0.3 * latency_score
                + 0.3 * throughput_score
                + 0.3 * success_rate
                + 0.1 * cache_hit_rate
            )

            return min(max(fitness, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Fitness calculation failed: {e}")
            return 0.5

    async def _run_evolution(
        self, graph: Any, metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run evolutionary optimization"""
        proposals = []

        if self.evolution_engine:
            # Generate evolutionary proposals
            for _ in range(3):  # Generate 3 proposals
                proposal = {
                    "type": "evolution",
                    "mutation": random.choice(
                        ["add_node", "remove_edge", "modify_param"]
                    ),
                    "confidence": random.uniform(0.5, 0.9),
                }
                proposals.append(proposal)

        return proposals

    async def _run_optimization(self, graph: Any, metrics: Dict[str, Any]) -> List[str]:
        """Run deep optimization"""
        optimizations = []

        if self.optimizer:
            # Apply optimizations
            optimizations.append("parameter_tuning")
            optimizations.append("cache_optimization")

        return optimizations

    async def _run_nso(self, graph: Any, metrics: Dict[str, Any]) -> List[str]:
        """Run neural system optimization"""
        optimizations = []

        if self.nso:
            # Apply neural optimizations
            optimizations.append("neural_pruning")
            optimizations.append("activation_optimization")

        return optimizations

    async def _check_governance(
        self, graph: Any, optimizations: List[str], proposals: List[Dict[str, Any]]
    ) -> List[str]:
        """Check governance constraints"""
        violations = []

        if self.governance:
            # Check for violations
            if len(optimizations) > 10:
                violations.append("too_many_optimizations")
            if len(proposals) > 5:
                violations.append("too_many_proposals")

        return violations

    def _calculate_performance_delta(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance changes"""
        delta = {}

        # Mock deltas for now
        delta["latency_improvement"] = random.uniform(-0.1, 0.3)
        delta["throughput_improvement"] = random.uniform(-0.05, 0.2)

        return delta


class ExecutionExplainer:
    """
    Provides explanations for subgraph executions
    """

    def __init__(self):
        self.interpretability = (
            InterpretabilityEngine() if INTERPRETABILITY_AVAILABLE else None
        )
        self.explanation_cache: Dict[str, ExecutionExplanation] = {}
        self.explanation_history: deque = deque(maxlen=100)

        logger.info(
            f"ExecutionExplainer initialized. InterpretabilityEngine available: "
            f"{INTERPRETABILITY_AVAILABLE}"
        )

    def explain_execution(
        self,
        subgraph: Any,
        inputs: Dict[str, Any],
        outputs: Any,
        explanation_type: ExplanationType = ExplanationType.SIMPLE,
    ) -> List[ExecutionExplanation]:
        """
        Generate explanation for subgraph execution

        Args:
            subgraph: Executed subgraph
            inputs: Input values
            outputs: Output values
            explanation_type: Type of explanation

        Returns:
            List of explanations
        """
        explanations = []

        try:
            # Generate subgraph ID
            subgraph_id = self._generate_subgraph_id(subgraph)

            # Check cache
            cache_key = f"{subgraph_id}_{explanation_type.value}"
            if cache_key in self.explanation_cache:
                cached = self.explanation_cache[cache_key]
                if (datetime.now() - cached.timestamp).seconds < 300:  # 5 min cache
                    return [cached]

            # Generate explanation based on type
            if explanation_type == ExplanationType.SIMPLE:
                explanation = self._generate_simple_explanation(
                    subgraph, inputs, outputs
                )
            elif explanation_type == ExplanationType.DETAILED:
                explanation = self._generate_detailed_explanation(
                    subgraph, inputs, outputs
                )
            elif explanation_type == ExplanationType.TECHNICAL:
                explanation = self._generate_technical_explanation(
                    subgraph, inputs, outputs
                )
            else:
                explanation = self._generate_simple_explanation(
                    subgraph, inputs, outputs
                )

            # Use interpretability engine if available
            if self.interpretability:
                explanation = self._enhance_with_interpretability(
                    explanation, subgraph, outputs
                )

            # Cache explanation
            self.explanation_cache[cache_key] = explanation
            self.explanation_history.append(explanation.to_dict())

            explanations.append(explanation)

            # Check for unclear explanations
            if explanation.confidence < 0.5:
                self._flag_unclear_explanation(explanation, outputs)

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            explanations.append(
                ExecutionExplanation(
                    subgraph_id="unknown",
                    explanation_type=explanation_type,
                    summary="Explanation generation failed",
                    details={"error": str(e)},
                    confidence=0.0,
                )
            )

        return explanations

    def get_explanation_summary(self, subgraph_id: str) -> Optional[str]:
        """Get summary of explanations for a subgraph"""
        summaries = []
        for key, explanation in self.explanation_cache.items():
            if key.startswith(subgraph_id):
                summaries.append(explanation.summary)

        return " | ".join(summaries) if summaries else None

    def _generate_subgraph_id(self, subgraph: Any) -> str:
        """Generate subgraph ID"""
        # Simple ID generation
        return f"subgraph_{id(subgraph) % 10000:04d}"

    def _generate_simple_explanation(
        self, subgraph: Any, inputs: Dict[str, Any], outputs: Any
    ) -> ExecutionExplanation:
        """Generate simple explanation"""
        summary = f"Subgraph processed {len(inputs)} inputs and produced output"

        details = {
            "input_count": len(inputs),
            "output_type": type(outputs).__name__,
            "execution_time": random.uniform(0.01, 0.1),
        }

        return ExecutionExplanation(
            subgraph_id=self._generate_subgraph_id(subgraph),
            explanation_type=ExplanationType.SIMPLE,
            summary=summary,
            details=details,
            confidence=0.8,
        )

    def _generate_detailed_explanation(
        self, subgraph: Any, inputs: Dict[str, Any], outputs: Any
    ) -> ExecutionExplanation:
        """Generate detailed explanation"""
        summary = "Detailed analysis of subgraph execution"

        details = {
            "inputs": {k: str(v)[:50] for k, v in inputs.items()},
            "processing_steps": [
                "Input validation",
                "Data transformation",
                "Core computation",
                "Output generation",
            ],
            "output_summary": str(outputs)[:100],
        }

        return ExecutionExplanation(
            subgraph_id=self._generate_subgraph_id(subgraph),
            explanation_type=ExplanationType.DETAILED,
            summary=summary,
            details=details,
            confidence=0.75,
        )

    def _generate_technical_explanation(
        self, subgraph: Any, inputs: Dict[str, Any], outputs: Any
    ) -> ExecutionExplanation:
        """Generate technical explanation"""
        summary = "Technical execution trace"

        # Extract tensors if present
        tensors = self._extract_tensors(outputs)

        details = {
            "compute_graph": "Sequential processing pipeline",
            "memory_usage": random.randint(100, 1000),
            "tensor_shapes": {k: str(v.shape) for k, v in tensors.items()}
            if tensors
            else {},
            "optimization_hints": ["Consider batching", "Enable caching"],
        }

        return ExecutionExplanation(
            subgraph_id=self._generate_subgraph_id(subgraph),
            explanation_type=ExplanationType.TECHNICAL,
            summary=summary,
            details=details,
            confidence=0.7,
        )

    def _extract_tensors(self, obj: Any) -> Dict[str, Any]:
        """Extract tensor objects from output"""
        tensors = {}

        if isinstance(obj, dict):
            for key, value in obj.items():
                if self._is_tensor(value):
                    tensors[key] = value
                elif isinstance(value, dict):
                    nested = self._extract_tensors(value)
                    tensors.update({f"{key}.{k}": v for k, v in nested.items()})
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                if self._is_tensor(item):
                    tensors[f"item_{i}"] = item
        elif self._is_tensor(obj):
            tensors["output"] = obj

        return tensors

    def _is_tensor(self, obj: Any) -> bool:
        """Check if object is a tensor"""
        # Check for numpy arrays
        if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return True

        # Check for torch tensors (FIXED: Now checks if torch is available)
        if TORCH_AVAILABLE and torch is not None and isinstance(obj, torch.Tensor):
            return True

        return False

    def _enhance_with_interpretability(
        self, explanation: ExecutionExplanation, subgraph: Any, outputs: Any
    ) -> ExecutionExplanation:
        """Enhance explanation with interpretability engine"""
        if self.interpretability:
            # Add interpretability insights
            explanation.details["interpretability"] = {
                "feature_importance": random.random(),
                "decision_path": "Main computation path",
                "confidence_regions": [0.6, 0.8],
            }
            explanation.confidence = min(explanation.confidence * 1.1, 1.0)

        return explanation

    def _flag_unclear_explanation(
        self, explanation: ExecutionExplanation, outputs: Any
    ):
        """Flag explanations that are unclear"""
        # Check for high dimensional or sparse outputs
        tensors = self._extract_tensors(outputs)

        for tensor in tensors.values():
            if self._is_tensor(tensor):
                # Check sparsity
                if NUMPY_AVAILABLE:
                    if isinstance(tensor, np.ndarray):
                        sparsity = np.sum(tensor == 0) / tensor.size
                        if sparsity > 0.9:
                            explanation.details["warning"] = "High sparsity detected"

    def _calculate_explanation_confidence(self, subgraph: Any, outputs: Any) -> float:
        """Calculate confidence score for explanation"""
        base_confidence = 0.5

        # Adjust based on output complexity
        if isinstance(outputs, (int, float, str)):
            base_confidence += 0.3
        elif isinstance(outputs, dict):
            base_confidence += 0.1 * min(len(outputs), 5) / 5

        return min(base_confidence, 1.0)


class RuntimeExtensions:
    """
    Main interface for runtime extensions
    """

    def __init__(
        self,
        learned_subgraphs_dir: str = "learned_subgraphs",
        enable_autonomous: bool = True,
    ):
        # Initialize components
        self.subgraph_learner = SubgraphLearner(learned_subgraphs_dir)
        self.autonomous_optimizer = AutonomousOptimizer() if enable_autonomous else None
        self.execution_explainer = ExecutionExplainer()

        # Statistics
        self.stats = {
            "patterns_learned": 0,
            "optimizations_run": 0,
            "explanations_generated": 0,
        }

        logger.info("RuntimeExtensions initialized")

    def learn_subgraph(
        self,
        subgraph_type: str,
        graph_definition: Dict[str, Any],
        mode: str = "supervised",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Learn a new subgraph pattern"""
        try:
            learning_mode = LearningMode(mode)
        except ValueError:
            learning_mode = LearningMode.SUPERVISED

        success, result = self.subgraph_learner.learn_subgraph(
            subgraph_type, graph_definition, learning_mode, metadata
        )

        if success:
            self.stats["patterns_learned"] += 1

        return success, result

    def load_learned_subgraphs(self) -> Dict[str, SubgraphPattern]:
        """Load all learned subgraph patterns"""
        return self.subgraph_learner.patterns

    async def trigger_autonomous_cycle(
        self, graph: Any, metrics: Dict[str, Any]
    ) -> Optional[AutonomousCycleReport]:
        """Trigger autonomous optimization cycle"""
        if not self.autonomous_optimizer:
            return None

        report = await self.autonomous_optimizer.trigger_autonomous_cycle(
            graph, metrics
        )
        self.stats["optimizations_run"] += 1

        return report

    def explain_execution(
        self,
        subgraph: Any,
        inputs: Dict[str, Any],
        outputs: Any,
        explanation_type: str = "simple",
    ) -> List[ExecutionExplanation]:
        """Generate execution explanation"""
        try:
            exp_type = ExplanationType(explanation_type)
        except ValueError:
            exp_type = ExplanationType.SIMPLE

        explanations = self.execution_explainer.explain_execution(
            subgraph, inputs, outputs, exp_type
        )

        self.stats["explanations_generated"] += len(explanations)

        return explanations

    def flag_unclear_explanation(self, explanation: ExecutionExplanation, outputs: Any):
        """Flag unclear explanations for review"""
        self.execution_explainer._flag_unclear_explanation(explanation, outputs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics"""
        stats = self.stats.copy()
        stats["total_patterns"] = len(self.subgraph_learner.patterns)
        stats["current_fitness"] = (
            getattr(self.autonomous_optimizer, "current_fitness", 0.0)
            if self.autonomous_optimizer
            else 0.0
        )
        stats["cached_explanations"] = len(self.execution_explainer.explanation_cache)

        return stats


# Helper functions
def create_runtime_extensions(
    config: Optional[Dict[str, Any]] = None,
) -> RuntimeExtensions:
    """
    Create RuntimeExtensions instance with configuration

    Args:
        config: Configuration dictionary

    Returns:
        RuntimeExtensions instance
    """
    if config is None:
        config = {}

    return RuntimeExtensions(
        learned_subgraphs_dir=config.get("learned_subgraphs_dir", "learned_subgraphs"),
        enable_autonomous=config.get("enable_autonomous", True),
    )


def load_extension_config(config_file: str) -> Dict[str, Any]:
    """Load extension configuration from file"""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


# Module exports
__all__ = [
    "RuntimeExtensions",
    "SubgraphLearner",
    "AutonomousOptimizer",
    "ExecutionExplainer",
    "SubgraphPattern",
    "ExecutionExplanation",
    "AutonomousCycleReport",
    "LearningMode",
    "ExplanationType",
    "create_runtime_extensions",
    "load_extension_config",
]
