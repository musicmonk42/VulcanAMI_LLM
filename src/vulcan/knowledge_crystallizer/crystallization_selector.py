"""
crystallization_selector.py - Selects appropriate crystallization methods
Part of the VULCAN-AGI system

Follows EXAMINE → SELECT → APPLY → REMEMBER pattern for method selection
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import time
import threading
from enum import Enum
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class CrystallizationMethod(Enum):
    """Available crystallization methods"""

    STANDARD = "standard"
    CASCADE_AWARE = "cascade_aware"
    INCREMENTAL = "incremental"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"
    EXPLORATORY = "exploratory"
    REFINEMENT = "refinement"


class TraceComplexity(Enum):
    """Complexity levels for execution traces"""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class DomainType(Enum):
    """Domain types for crystallization"""

    GENERAL = "general"
    SPECIALIZED = "specialized"
    CROSS_DOMAIN = "cross_domain"
    NOVEL = "novel"


@dataclass
class TraceCharacteristics:
    """Characteristics of an execution trace"""

    has_failures: bool = False
    failure_rate: float = 0.0
    is_incremental: bool = False
    iteration_count: int = 0
    batch_size: int = 1
    complexity: TraceComplexity = TraceComplexity.MODERATE
    domain_type: DomainType = DomainType.GENERAL
    action_count: int = 0
    unique_patterns: int = 0
    success_rate: float = 1.0
    has_loops: bool = False
    has_conditionals: bool = False
    has_dependencies: bool = False
    resource_usage: Dict[str, float] = field(default_factory=dict)
    timing_critical: bool = False
    confidence_level: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "has_failures": self.has_failures,
            "failure_rate": self.failure_rate,
            "is_incremental": self.is_incremental,
            "iteration_count": self.iteration_count,
            "batch_size": self.batch_size,
            "complexity": self.complexity.value,
            "domain_type": self.domain_type.value,
            "action_count": self.action_count,
            "unique_patterns": self.unique_patterns,
            "success_rate": self.success_rate,
            "has_loops": self.has_loops,
            "has_conditionals": self.has_conditionals,
            "has_dependencies": self.has_dependencies,
            "resource_usage": self.resource_usage,
            "timing_critical": self.timing_critical,
            "confidence_level": self.confidence_level,
            "metadata": self.metadata,
        }


@dataclass
class MethodSelection:
    """Selected crystallization method and parameters"""

    method: CrystallizationMethod
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    fallback_methods: List[CrystallizationMethod] = field(default_factory=list)
    estimated_cost: float = 1.0
    estimated_time: float = 1.0
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "method": self.method.value,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "fallback_methods": [m.value for m in self.fallback_methods],
            "estimated_cost": self.estimated_cost,
            "estimated_time": self.estimated_time,
            "priority": self.priority,
            "metadata": self.metadata,
        }


class SelectionStrategy:
    """Base class for selection strategies"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate strategy applicability

        Returns:
            Tuple of (score, parameters)
        """
        raise NotImplementedError

    def get_method(self) -> CrystallizationMethod:
        """Get the crystallization method for this strategy"""
        raise NotImplementedError


class StandardStrategy(SelectionStrategy):
    """Standard crystallization strategy"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate standard strategy applicability"""
        score = 0.5  # Base score

        # Good for simple to moderate complexity
        if characteristics.complexity in [
            TraceComplexity.SIMPLE,
            TraceComplexity.MODERATE,
        ]:
            score += 0.3

        # Good for high success rate
        if characteristics.success_rate > 0.8:
            score += 0.2

        # Not ideal for failures
        if characteristics.has_failures:
            score -= 0.2

        # Not ideal for incremental
        if characteristics.is_incremental:
            score -= 0.1

        parameters = {
            "validation_level": "basic"
            if characteristics.complexity == TraceComplexity.SIMPLE
            else "comprehensive"
        }

        return max(0.0, min(1.0, score)), parameters

    def get_method(self) -> CrystallizationMethod:
        return CrystallizationMethod.STANDARD


class CascadeAwareStrategy(SelectionStrategy):
    """Cascade-aware crystallization strategy"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate cascade-aware strategy applicability"""
        score = 0.3  # Base score

        # Excellent for failures
        if characteristics.has_failures:
            score += 0.4
            score += characteristics.failure_rate * 0.2

        # Good for dependencies
        if characteristics.has_dependencies:
            score += 0.2

        # Good for complex traces
        if characteristics.complexity in [
            TraceComplexity.COMPLEX,
            TraceComplexity.HIGHLY_COMPLEX,
        ]:
            score += 0.2

        # Check context for cascade history
        if context.get("cascade_failures_detected", False):
            score += 0.3

        parameters = {
            "cascade_depth": 3 if characteristics.has_dependencies else 2,
            "failure_threshold": characteristics.failure_rate,
            "circuit_breaker_enabled": characteristics.failure_rate > 0.5,
        }

        return max(0.0, min(1.0, score)), parameters

    def get_method(self) -> CrystallizationMethod:
        return CrystallizationMethod.CASCADE_AWARE


class IncrementalStrategy(SelectionStrategy):
    """Incremental crystallization strategy"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate incremental strategy applicability"""
        score = 0.2  # Base score

        # Excellent for incremental traces
        if characteristics.is_incremental:
            score += 0.5
            # FIX: Handle None iteration_count
            iteration_count = characteristics.iteration_count or 0
            score += min(0.2, iteration_count / 10)

        # Good for loops
        if characteristics.has_loops:
            score += 0.2

        # Check context for previous iterations
        if context.get("previous_iterations", 0) > 0:
            score += 0.3

        # Good for refinement
        if context.get("refinement_requested", False):
            score += 0.2

        parameters = {
            # FIX: Handle None iteration_count in parameters
            "merge_strategy": "weighted"
            if (characteristics.iteration_count or 0) > 5
            else "simple",
            "iteration_weight_decay": 0.9,
            "keep_history": True,
            "max_iterations": 100,
        }

        return max(0.0, min(1.0, score)), parameters

    def get_method(self) -> CrystallizationMethod:
        return CrystallizationMethod.INCREMENTAL


class BatchStrategy(SelectionStrategy):
    """Batch crystallization strategy"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate batch strategy applicability"""
        score = 0.1  # Base score

        # Excellent for multiple traces
        if characteristics.batch_size > 1:
            score += min(0.5, characteristics.batch_size / 20)

        # Check context for batch availability
        batch_available = context.get("batch_traces_available", 0)
        if batch_available > 5:
            score += 0.4

        # Good for pattern extraction
        if characteristics.unique_patterns > 3:
            score += 0.2

        # Good for cross-domain
        if characteristics.domain_type == DomainType.CROSS_DOMAIN:
            score += 0.1

        parameters = {
            "batch_size": max(characteristics.batch_size, batch_available),
            "parallel_processing": characteristics.batch_size > 10,
            "aggregation_method": "voting"
            if characteristics.batch_size > 5
            else "averaging",
            "outlier_detection": True,
        }

        return max(0.0, min(1.0, score)), parameters

    def get_method(self) -> CrystallizationMethod:
        return CrystallizationMethod.BATCH


class AdaptiveStrategy(SelectionStrategy):
    """Adaptive crystallization strategy"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate adaptive strategy applicability"""
        score = 0.4  # Base score - decent default

        # Good for novel domains
        if characteristics.domain_type == DomainType.NOVEL:
            score += 0.3

        # Good for moderate to high complexity with mixed results
        if characteristics.complexity in [
            TraceComplexity.MODERATE,
            TraceComplexity.COMPLEX,
        ]:
            if 0.3 < characteristics.success_rate < 0.8:
                score += 0.3

        # Good when confidence is uncertain
        if 0.3 < characteristics.confidence_level < 0.7:
            score += 0.2

        # Check context for adaptation needs
        if context.get("adaptation_requested", False):
            score += 0.3

        parameters = {
            "adaptation_rate": 0.1,
            "exploration_ratio": 0.2
            if characteristics.domain_type == DomainType.NOVEL
            else 0.1,
            "feedback_integration": True,
            "dynamic_thresholds": True,
        }

        return max(0.0, min(1.0, score)), parameters

    def get_method(self) -> CrystallizationMethod:
        return CrystallizationMethod.ADAPTIVE


class HybridStrategy(SelectionStrategy):
    """Hybrid crystallization strategy combining multiple approaches"""

    def evaluate(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate hybrid strategy applicability"""
        score = 0.3  # Base score

        # Good for highly complex scenarios
        if characteristics.complexity == TraceComplexity.HIGHLY_COMPLEX:
            score += 0.4

        # Good when multiple characteristics present
        characteristic_count = sum(
            [
                characteristics.has_failures,
                characteristics.is_incremental,
                characteristics.has_loops,
                characteristics.has_conditionals,
                characteristics.has_dependencies,
            ]
        )

        if characteristic_count >= 3:
            score += 0.3

        # Good for cross-domain
        if characteristics.domain_type == DomainType.CROSS_DOMAIN:
            score += 0.2

        # Check context for hybrid needs
        if context.get("multiple_objectives", False):
            score += 0.2

        parameters = {
            "primary_method": self._select_primary_method(characteristics),
            "secondary_methods": self._select_secondary_methods(characteristics),
            "fusion_strategy": "weighted",
            "weight_distribution": "adaptive",
        }

        return max(0.0, min(1.0, score)), parameters

    def get_method(self) -> CrystallizationMethod:
        return CrystallizationMethod.HYBRID

    def _select_primary_method(self, characteristics: TraceCharacteristics) -> str:
        """Select primary method for hybrid approach"""
        if characteristics.has_failures:
            return CrystallizationMethod.CASCADE_AWARE.value
        elif characteristics.is_incremental:
            return CrystallizationMethod.INCREMENTAL.value
        else:
            return CrystallizationMethod.STANDARD.value

    def _select_secondary_methods(
        self, characteristics: TraceCharacteristics
    ) -> List[str]:
        """Select secondary methods for hybrid approach"""
        methods = []
        if characteristics.batch_size > 1:
            methods.append(CrystallizationMethod.BATCH.value)
        if characteristics.domain_type == DomainType.NOVEL:
            methods.append(CrystallizationMethod.ADAPTIVE.value)
        return methods


class CrystallizationSelector:
    """Selects appropriate crystallization method based on context"""

    def __init__(self):
        """Initialize crystallization selector"""
        # Strategy instances
        self.strategies = {
            CrystallizationMethod.STANDARD: StandardStrategy(),
            CrystallizationMethod.CASCADE_AWARE: CascadeAwareStrategy(),
            CrystallizationMethod.INCREMENTAL: IncrementalStrategy(),
            CrystallizationMethod.BATCH: BatchStrategy(),
            CrystallizationMethod.ADAPTIVE: AdaptiveStrategy(),
            CrystallizationMethod.HYBRID: HybridStrategy(),
        }

        # Selection history for learning
        self.selection_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(lambda: {"successes": 0, "failures": 0})

        # Cache for repeated selections
        self.selection_cache = {}
        self.cache_ttl = 60  # seconds

        # Thread safety
        self.lock = threading.RLock()

        # Configuration
        self.min_confidence_threshold = 0.3
        self.enable_learning = True

        logger.info(
            "CrystallizationSelector initialized with %d strategies",
            len(self.strategies),
        )

    def select_method(
        self, trace: Any, context: Optional[Dict[str, Any]] = None
    ) -> MethodSelection:
        """
        EXAMINE → SELECT crystallization method

        Args:
            trace: Execution trace to crystallize
            context: Additional context for selection

        Returns:
            Selected method and parameters
        """
        context = context or {}

        with self.lock:
            # EXAMINE: Check cache first
            cache_key = self._generate_cache_key(trace, context)
            if cache_key in self.selection_cache:
                cached_time, cached_selection = self.selection_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    logger.debug("Using cached selection for %s", cache_key)
                    return cached_selection

            # EXAMINE: Analyze trace characteristics
            characteristics = self._analyze_trace(trace, context)

            # SELECT: Evaluate all strategies
            strategy_scores = {}
            strategy_params = {}

            for method, strategy in self.strategies.items():
                score, params = strategy.evaluate(characteristics, context)

                # Apply learning adjustments if enabled
                if self.enable_learning:
                    score = self._apply_learning_adjustment(method, score)

                strategy_scores[method] = score
                strategy_params[method] = params

            # SELECT: Choose best strategy
            best_method = max(strategy_scores.items(), key=lambda x: x[1])
            method = best_method[0]
            confidence = best_method[1]

            # Check if confidence meets threshold
            if confidence < self.min_confidence_threshold:
                # Fallback to standard if confidence too low
                method = CrystallizationMethod.STANDARD
                confidence = 0.5
                logger.warning(
                    "Low confidence %.2f, falling back to standard method", confidence
                )

            # APPLY: Create selection
            selection = MethodSelection(
                method=method,
                confidence=confidence,
                parameters=strategy_params[method],
                reasoning=self._generate_reasoning(method, characteristics, confidence),
                fallback_methods=self._identify_fallbacks(strategy_scores, method),
                estimated_cost=self._estimate_cost(method, characteristics),
                estimated_time=self._estimate_time(method, characteristics),
                priority=self._calculate_priority(characteristics, context),
                metadata={
                    "characteristics": characteristics.to_dict(),
                    "all_scores": {m.value: s for m, s in strategy_scores.items()},
                    "selection_timestamp": time.time(),
                },
            )

            # REMEMBER: Cache and track selection
            self.selection_cache[cache_key] = (time.time(), selection)
            self._track_selection(selection, characteristics)

            logger.info(
                "Selected %s method with confidence %.2f", method.value, confidence
            )

            return selection

    def _analyze_trace(
        self, trace: Any, context: Dict[str, Any]
    ) -> TraceCharacteristics:
        """
        Analyze execution trace characteristics

        Args:
            trace: Execution trace
            context: Additional context

        Returns:
            Trace characteristics
        """
        characteristics = TraceCharacteristics()

        # Basic properties
        characteristics.has_failures = not getattr(trace, "success", True)

        # FIX: Set success_rate based on actual success status
        if characteristics.has_failures:
            characteristics.success_rate = 0.0
        else:
            characteristics.success_rate = 1.0

        # Check for iterations - FIX: Ensure never None
        if hasattr(trace, "iteration"):
            characteristics.is_incremental = True
            characteristics.iteration_count = getattr(trace, "iteration", 0) or 0
        else:
            # Always set to 0, never None
            characteristics.iteration_count = 0

        # FIX: Analyze actions with defensive check for non-iterable objects
        if hasattr(trace, "actions"):
            actions = getattr(trace, "actions", [])
            try:
                characteristics.action_count = len(actions)

                # Detect patterns
                characteristics.unique_patterns = self._count_unique_patterns(actions)
                characteristics.has_loops = self._detect_loops(actions)
                characteristics.has_conditionals = self._detect_conditionals(actions)
            except (TypeError, AttributeError):
                # actions exists but is not iterable/countable (e.g., Mock object)
                characteristics.action_count = 0
                logger.warning("Trace has 'actions' attribute but it's not iterable")

        # Analyze complexity
        characteristics.complexity = self._assess_complexity(trace)

        # Domain analysis - FIX: Handle None domain
        domain = None
        if hasattr(trace, "domain"):
            domain = getattr(trace, "domain", None)
        characteristics.domain_type = self._classify_domain(domain, context)

        # Resource usage
        if hasattr(trace, "metadata"):
            metadata = getattr(trace, "metadata", {})
            if "resources" in metadata:
                characteristics.resource_usage = metadata["resources"]
            characteristics.timing_critical = metadata.get("timing_critical", False)

        # Dependencies
        characteristics.has_dependencies = self._detect_dependencies(trace)

        # Confidence
        characteristics.confidence_level = getattr(trace, "confidence", 0.5)

        # Batch information from context
        characteristics.batch_size = context.get("batch_size", 1)

        # Failure analysis
        if characteristics.has_failures:
            characteristics.failure_rate = self._calculate_failure_rate(trace, context)

        return characteristics

    def _count_unique_patterns(self, actions: List[Any]) -> int:
        """Count unique patterns in actions"""
        if not actions:
            return 0

        # Simple pattern detection based on action types/names
        patterns = set()

        for i in range(len(actions)):
            # Single action patterns
            action_type = (
                str(actions[i].get("type", actions[i]))
                if isinstance(actions[i], dict)
                else str(actions[i])
            )
            patterns.add(action_type)

            # Bigram patterns
            if i < len(actions) - 1:
                next_type = (
                    str(actions[i + 1].get("type", actions[i + 1]))
                    if isinstance(actions[i + 1], dict)
                    else str(actions[i + 1])
                )
                patterns.add(f"{action_type}->{next_type}")

        return len(patterns)

    def _detect_loops(self, actions: List[Any]) -> bool:
        """Detect loop patterns in actions"""
        if len(actions) < 3:
            return False

        # Look for repeated sequences
        action_strs = [
            str(a.get("type", a)) if isinstance(a, dict) else str(a) for a in actions
        ]

        # Check for repeated subsequences
        for length in range(2, min(10, len(actions) // 2 + 1)):
            for start in range(len(actions) - length * 2 + 1):
                subsequence = action_strs[start : start + length]
                if action_strs[start + length : start + length * 2] == subsequence:
                    return True

        return False

    def _detect_conditionals(self, actions: List[Any]) -> bool:
        """Detect conditional patterns in actions"""
        for action in actions:
            if isinstance(action, dict):
                if "condition" in action or "if" in str(action).lower():
                    return True
        return False

    def _detect_dependencies(self, trace: Any) -> bool:
        """Detect dependencies in trace"""
        if hasattr(trace, "dependencies"):
            deps = getattr(trace, "dependencies", [])
            try:
                return len(deps) > 0
            except (TypeError, AttributeError):
                # Handle Mock objects or non-iterable dependencies
                return False

        if hasattr(trace, "metadata"):
            metadata = getattr(trace, "metadata", {})
            return "dependencies" in metadata

        return False

    def _assess_complexity(self, trace: Any) -> TraceComplexity:
        """Assess trace complexity"""
        complexity_score = 0

        # Action count factor - FIXED: Better thresholds for high counts
        if hasattr(trace, "actions"):
            try:
                action_count = len(getattr(trace, "actions", []))
                if action_count > 100:
                    complexity_score += 4  # Very high action count
                elif action_count > 50:
                    complexity_score += 3
                elif action_count > 20:
                    complexity_score += 2
                elif action_count > 10:
                    complexity_score += 1
            except (TypeError, AttributeError):
                # actions not iterable, skip
                pass

        # Nested structures
        if hasattr(trace, "context"):
            context = getattr(trace, "context", {})
            depth = self._calculate_nesting_depth(context)
            complexity_score += min(3, depth)

        # Outcome complexity
        if hasattr(trace, "outcomes"):
            outcomes = getattr(trace, "outcomes", {})
            if isinstance(outcomes, dict) and len(outcomes) > 10:
                complexity_score += 2

        # Map score to complexity level
        if complexity_score <= 2:
            return TraceComplexity.SIMPLE
        elif complexity_score <= 5:
            return TraceComplexity.MODERATE
        elif complexity_score <= 8:
            return TraceComplexity.COMPLEX
        else:
            return TraceComplexity.HIGHLY_COMPLEX

    def _calculate_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of object"""
        if current_depth > 10:  # Prevent infinite recursion
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(
                self._calculate_nesting_depth(v, current_depth + 1)
                for v in obj.values()
            )
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in obj)
        else:
            return current_depth

    def _classify_domain(self, domain: str, context: Dict[str, Any]) -> DomainType:
        """Classify domain type"""
        # FIX: Handle None domain first
        if domain is None:
            return DomainType.GENERAL

        known_domains = context.get(
            "known_domains", ["general", "optimization", "analysis", "control"]
        )

        if domain == "general":
            return DomainType.GENERAL
        elif domain in known_domains:
            # Check if in known_domains BEFORE checking for underscores
            return DomainType.SPECIALIZED
        elif "_" in domain or "-" in domain:
            # Likely cross-domain
            return DomainType.CROSS_DOMAIN
        else:
            return DomainType.NOVEL

    def _calculate_failure_rate(self, trace: Any, context: Dict[str, Any]) -> float:
        """Calculate failure rate from trace and context"""
        # Check trace for failure information
        if hasattr(trace, "metadata"):
            metadata = getattr(trace, "metadata", {})
            if "failure_rate" in metadata:
                return metadata["failure_rate"]

        # Check context for historical failures
        historical_failures = context.get("historical_failure_rate", 0.0)

        # Simple heuristic: if current trace failed, estimate higher rate
        if not getattr(trace, "success", True):
            return max(0.5, historical_failures * 1.2)

        return historical_failures

    def _generate_cache_key(self, trace: Any, context: Dict[str, Any]) -> str:
        """Generate cache key for selection"""
        # Create a deterministic key from trace and context
        key_parts = []

        # Trace properties
        key_parts.append(str(getattr(trace, "trace_id", "")))
        key_parts.append(str(getattr(trace, "success", True)))
        key_parts.append(str(getattr(trace, "domain", "general")))

        # Context properties
        key_parts.append(str(context.get("batch_size", 1)))
        key_parts.append(str(context.get("cascade_failures_detected", False)))

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def _apply_learning_adjustment(
        self, method: CrystallizationMethod, base_score: float
    ) -> float:
        """Apply learning-based adjustment to score"""
        metrics = self.performance_metrics[method]

        total = metrics["successes"] + metrics["failures"]
        if total == 0:
            return base_score  # No history

        success_rate = metrics["successes"] / total

        # Adjust score based on historical performance
        # Boost successful methods, penalize unsuccessful ones
        adjustment = (success_rate - 0.5) * 0.2  # ±0.1 adjustment

        adjusted_score = base_score + adjustment

        return max(0.0, min(1.0, adjusted_score))

    def _generate_reasoning(
        self,
        method: CrystallizationMethod,
        characteristics: TraceCharacteristics,
        confidence: float,
    ) -> str:
        """Generate human-readable reasoning for selection"""
        reasons = []

        if method == CrystallizationMethod.CASCADE_AWARE:
            if characteristics.has_failures:
                reasons.append(
                    f"Detected failures with {characteristics.failure_rate:.1%} failure rate"
                )
            if characteristics.has_dependencies:
                reasons.append("Dependencies require cascade analysis")

        elif method == CrystallizationMethod.INCREMENTAL:
            if characteristics.is_incremental:
                reasons.append(
                    f"Incremental pattern with {characteristics.iteration_count} iterations"
                )
            if characteristics.has_loops:
                reasons.append("Loop patterns benefit from incremental approach")

        elif method == CrystallizationMethod.BATCH:
            if characteristics.batch_size > 1:
                reasons.append(
                    f"Batch of {characteristics.batch_size} traces available"
                )
            reasons.append(f"Found {characteristics.unique_patterns} unique patterns")

        elif method == CrystallizationMethod.ADAPTIVE:
            if characteristics.domain_type == DomainType.NOVEL:
                reasons.append("Novel domain requires adaptation")
            reasons.append(
                f"Moderate confidence level ({characteristics.confidence_level:.2f}) suits adaptive approach"
            )

        elif method == CrystallizationMethod.HYBRID:
            reasons.append(
                f"High complexity ({characteristics.complexity.value}) requires hybrid approach"
            )

        else:  # STANDARD
            reasons.append(
                f"Standard crystallization suitable for {characteristics.complexity.value} complexity"
            )

        # Add confidence qualifier
        if confidence > 0.8:
            reasons.append("High confidence in method selection")
        elif confidence < 0.5:
            reasons.append("Moderate confidence - monitoring recommended")

        return (
            "; ".join(reasons)
            if reasons
            else "Default selection based on trace characteristics"
        )

    def _identify_fallbacks(
        self,
        strategy_scores: Dict[CrystallizationMethod, float],
        selected_method: CrystallizationMethod,
    ) -> List[CrystallizationMethod]:
        """Identify fallback methods in case primary fails"""
        # Sort methods by score, excluding selected
        sorted_methods = sorted(
            [(m, s) for m, s in strategy_scores.items() if m != selected_method],
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top 2 fallbacks with score > threshold
        fallbacks = []
        for method, score in sorted_methods[:2]:
            if score > self.min_confidence_threshold:
                fallbacks.append(method)

        # Always include STANDARD as last resort if not already selected/included
        if (
            CrystallizationMethod.STANDARD not in fallbacks
            and selected_method != CrystallizationMethod.STANDARD
        ):
            fallbacks.append(CrystallizationMethod.STANDARD)

        return fallbacks

    def _estimate_cost(
        self, method: CrystallizationMethod, characteristics: TraceCharacteristics
    ) -> float:
        """Estimate computational cost of crystallization"""
        base_costs = {
            CrystallizationMethod.STANDARD: 1.0,
            CrystallizationMethod.CASCADE_AWARE: 2.5,
            CrystallizationMethod.INCREMENTAL: 1.5,
            CrystallizationMethod.BATCH: 1.2,
            CrystallizationMethod.ADAPTIVE: 2.0,
            CrystallizationMethod.HYBRID: 3.0,
            CrystallizationMethod.EXPLORATORY: 1.8,
            CrystallizationMethod.REFINEMENT: 1.3,
        }

        base_cost = base_costs.get(method, 1.0)

        # Adjust for complexity
        complexity_multipliers = {
            TraceComplexity.SIMPLE: 0.5,
            TraceComplexity.MODERATE: 1.0,
            TraceComplexity.COMPLEX: 2.0,
            TraceComplexity.HIGHLY_COMPLEX: 4.0,
        }

        complexity_mult = complexity_multipliers[characteristics.complexity]

        # Adjust for action count
        action_mult = 1.0 + (characteristics.action_count / 100)

        # Adjust for batch size
        batch_mult = (
            1.0
            if method != CrystallizationMethod.BATCH
            else (0.7 + characteristics.batch_size * 0.05)
        )

        return base_cost * complexity_mult * action_mult * batch_mult

    def _estimate_time(
        self, method: CrystallizationMethod, characteristics: TraceCharacteristics
    ) -> float:
        """Estimate time required for crystallization (in relative units)"""
        # Similar to cost but with different factors
        base_times = {
            CrystallizationMethod.STANDARD: 1.0,
            CrystallizationMethod.CASCADE_AWARE: 3.0,  # More time for analysis
            CrystallizationMethod.INCREMENTAL: 1.2,
            CrystallizationMethod.BATCH: 2.0,  # Parallelizable but still takes time
            CrystallizationMethod.ADAPTIVE: 2.5,
            CrystallizationMethod.HYBRID: 4.0,
            CrystallizationMethod.EXPLORATORY: 2.2,
            CrystallizationMethod.REFINEMENT: 1.5,
        }

        base_time = base_times.get(method, 1.0)

        # Iterations add significant time
        if characteristics.is_incremental:
            base_time *= 1 + characteristics.iteration_count * 0.1

        return base_time

    def _calculate_priority(
        self, characteristics: TraceCharacteristics, context: Dict[str, Any]
    ) -> int:
        """Calculate processing priority (1-10, higher = more urgent)"""
        priority = 5  # Default priority

        # Failures increase priority
        if characteristics.has_failures:
            priority += 2

        # Timing critical increases priority
        if characteristics.timing_critical:
            priority += 3

        # User-specified priority
        if "user_priority" in context:
            priority = context["user_priority"]

        # Novel domains get slight boost
        if characteristics.domain_type == DomainType.NOVEL:
            priority += 1

        return max(1, min(10, priority))

    def _track_selection(
        self, selection: MethodSelection, characteristics: TraceCharacteristics
    ):
        """Track selection for learning and statistics"""
        self.selection_history.append(
            {
                "method": selection.method.value,
                "confidence": selection.confidence,
                "characteristics": characteristics.to_dict(),
                "timestamp": time.time(),
            }
        )

    def update_performance(self, method: CrystallizationMethod, success: bool):
        """
        Update performance metrics for learning

        Args:
            method: Crystallization method used
            success: Whether crystallization was successful
        """
        with self.lock:
            if success:
                self.performance_metrics[method]["successes"] += 1
            else:
                self.performance_metrics[method]["failures"] += 1

            logger.debug(
                "Updated performance for %s: success=%s", method.value, success
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics"""
        with self.lock:
            method_stats = {}
            for method, metrics in self.performance_metrics.items():
                total = metrics["successes"] + metrics["failures"]
                if total > 0:
                    success_rate = metrics["successes"] / total
                else:
                    success_rate = 0.0

                method_stats[method.value] = {
                    "total_uses": total,
                    "success_rate": success_rate,
                    "successes": metrics["successes"],
                    "failures": metrics["failures"],
                }

            # Recent selection distribution
            recent_methods = Counter()
            for entry in list(self.selection_history)[-100:]:
                recent_methods[entry["method"]] += 1

            return {
                "method_performance": method_stats,
                "recent_distribution": dict(recent_methods),
                "total_selections": len(self.selection_history),
                "cache_size": len(self.selection_cache),
                "learning_enabled": self.enable_learning,
            }

    def clear_cache(self):
        """Clear selection cache"""
        with self.lock:
            self.selection_cache.clear()
            logger.info("Selection cache cleared")
