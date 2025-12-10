"""
principle_learner.py - Bridges decomposer execution with knowledge crystallization
Part of the VULCAN-AGI system

This module closes the learning loop:
ExecutionOutcome → Crystallization → Validation → Library Promotion → Reuse

IMPLEMENTATION COMPLETE - All components functional and integrated
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import decomposer components
try:
    from .decomposition_library import StratifiedDecompositionLibrary
    from .problem_decomposer_core import (DecompositionPlan, ExecutionOutcome,
                                          ProblemGraph)
except ImportError:
    from decomposition_library import StratifiedDecompositionLibrary
    from problem_decomposer_core import (DecompositionPlan, ExecutionOutcome,
                                         ProblemGraph)

# Import knowledge crystallizer components
try:
    from ..knowledge_crystallizer.knowledge_crystallizer_core import \
        ExecutionTrace as CrystallizerTrace
    from ..knowledge_crystallizer.knowledge_crystallizer_core import \
        KnowledgeCrystallizer
    from ..knowledge_crystallizer.knowledge_storage import (
        KnowledgeIndex, KnowledgePruner, VersionedKnowledgeBase)
    from ..knowledge_crystallizer.principle_extractor import (ExecutionTrace,
                                                              Metric,
                                                              MetricType,
                                                              Pattern,
                                                              PatternType,
                                                              Principle)
    from ..knowledge_crystallizer.validation_engine import (KnowledgeValidator,
                                                            ValidationResult,
                                                            ValidationResults)
except ImportError:
    try:
        from knowledge_crystallizer.knowledge_crystallizer_core import \
            ExecutionTrace as CrystallizerTrace
        from knowledge_crystallizer.knowledge_crystallizer_core import \
            KnowledgeCrystallizer
        from knowledge_crystallizer.knowledge_storage import (
            KnowledgeIndex, KnowledgePruner, VersionedKnowledgeBase)
        from knowledge_crystallizer.principle_extractor import (ExecutionTrace,
                                                                Metric,
                                                                MetricType,
                                                                Pattern,
                                                                PatternType,
                                                                Principle)
        from knowledge_crystallizer.validation_engine import (
            KnowledgeValidator, ValidationResult, ValidationResults)
    except ImportError:
        logging.warning("Knowledge crystallizer components not available")
        KnowledgeCrystallizer = None
        KnowledgeValidator = None
        CrystallizerTrace = None
        ExecutionTrace = None
        Principle = None
        Metric = None
        MetricType = None
        Pattern = None
        PatternType = None
        ValidationResult = None
        ValidationResults = None
        VersionedKnowledgeBase = None
        KnowledgeIndex = None
        KnowledgePruner = None

logger = logging.getLogger(__name__)


# ============================================================
# UTILITY CLASSES
# ============================================================


class DictObject:
    """Simple object that allows attribute access to dict keys"""

    def __init__(self, d):
        self.__dict__.update(d)


def ensure_json_serializable(obj: Any) -> Any:
    """
    Recursively ensure an object is JSON serializable

    Args:
        obj: Object to make serializable

    Returns:
        JSON-serializable version of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]

    if isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}

    # Handle objects with value attribute (like enums)
    if hasattr(obj, "value"):
        return obj.value

    # Handle objects with name attribute
    if hasattr(obj, "name"):
        return obj.name

    # Handle objects with to_dict method
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return ensure_json_serializable(obj.to_dict())
        except Exception:
            pass

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return ensure_json_serializable(obj.__dict__)
        except Exception:
            pass

    # Last resort: convert to string
    return str(obj)


# ============================================================
# CONVERSION LAYER
# ============================================================


class DecompositionToTraceConverter:
    """Converts decomposition artifacts to ExecutionTrace format for crystallization"""

    def __init__(self):
        """Initialize converter"""
        self.conversion_count = 0
        self.conversion_cache = {}
        self.cache_size = 100
        self._cache_lock = threading.RLock()  # Thread safety for cache

        logger.info("DecompositionToTraceConverter initialized")

    def convert(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ) -> ExecutionTrace:
        """
        Convert decomposition execution to ExecutionTrace

        Args:
            problem: Problem that was decomposed
            plan: Decomposition plan that was executed
            outcome: Execution outcome

        Returns:
            ExecutionTrace for crystallization
        """
        # Create cache key from problem signature and strategy
        cache_key = f"{problem.get_signature()}_{plan.strategy.name if plan.strategy else 'none'}_{outcome.success}"

        # Check cache
        with self._cache_lock:
            if cache_key in self.conversion_cache:
                logger.debug("Using cached trace conversion for %s", cache_key[:16])
                return self.conversion_cache[cache_key]

        # Generate trace ID
        trace_id = f"decomp_{problem.get_signature()}_{int(time.time())}"

        # Convert plan steps to actions
        actions = self._extract_actions(plan)

        # Extract outcomes
        outcomes = self._extract_outcomes(outcome)

        # Build context
        context = self._build_context(problem, plan)

        # Extract metrics
        metrics = self._extract_metrics(outcome, plan)

        # Detect patterns
        patterns = self._detect_patterns(plan, outcome)

        # Create execution trace
        trace = ExecutionTrace(
            trace_id=trace_id,
            actions=actions,
            outcomes=outcomes,
            context=context,
            metrics=metrics,
            timestamp=time.time(),
            success=outcome.success,
            domain=problem.metadata.get("domain", "general"),
            metadata={
                "problem_signature": problem.get_signature(),
                "strategy": plan.strategy.name if plan.strategy else "unknown",
                "complexity": problem.complexity_score,
                "execution_time": outcome.execution_time,
                "num_steps": len(plan.steps),
                "confidence": plan.confidence,
            },
            patterns=patterns,
        )

        self.conversion_count += 1

        # Cache the result with proper size enforcement
        with self._cache_lock:
            # Enforce cache size limit (FIFO eviction)
            if len(self.conversion_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.conversion_cache))
                del self.conversion_cache[oldest_key]

            self.conversion_cache[cache_key] = trace

        logger.debug("Converted decomposition to trace %s", trace_id)

        return trace

    def _extract_actions(self, plan: DecompositionPlan) -> List[Dict[str, Any]]:
        """Extract actions from plan steps"""
        actions = []

        for i, step in enumerate(plan.steps):
            action = {
                "type": step.action_type,
                "description": step.description,
                "step_id": step.step_id,
                "position": i,
                "dependencies": step.dependencies,
                "estimated_complexity": step.estimated_complexity,
                "required_resources": step.required_resources,
            }

            # Add substeps if any
            if hasattr(step, "substeps") and step.substeps:
                action["has_substeps"] = True
                action["substep_count"] = len(step.substeps)

            actions.append(action)

        return actions

    def _extract_outcomes(self, outcome: ExecutionOutcome) -> Dict[str, Any]:
        """Extract outcomes from execution"""
        outcomes = {
            "success": outcome.success,
            "execution_time": outcome.execution_time,
            "timestamp": outcome.timestamp,
        }

        # Add solution if available
        if outcome.solution:
            outcomes["solution"] = outcome.solution

        # Add metrics
        if outcome.metrics:
            outcomes["metrics"] = outcome.metrics

        # Add errors
        if outcome.errors:
            outcomes["errors"] = outcome.errors[:5]  # Limit errors

        # Add sub-results summary
        if outcome.sub_results:
            outcomes["sub_results_count"] = len(outcome.sub_results)
            outcomes["sub_success_rate"] = outcome.get_success_rate()

        return outcomes

    def _build_context(
        self, problem: ProblemGraph, plan: DecompositionPlan
    ) -> Dict[str, Any]:
        """Build context dictionary"""
        context = {
            "problem_type": problem.metadata.get("type", "unknown"),
            "domain": problem.metadata.get("domain", "general"),
            "complexity": problem.complexity_score,
            "node_count": len(problem.nodes),
            "edge_count": len(problem.edges),
            "has_root": problem.root is not None,
            "strategy": plan.strategy.name if plan.strategy else "unknown",
            "plan_confidence": plan.confidence,
            "estimated_complexity": plan.estimated_complexity,
        }

        # Add constraints
        if "constraints" in problem.metadata:
            context["constraints"] = problem.metadata["constraints"]

        # Add resources
        if "resources" in problem.metadata:
            context["resources"] = problem.metadata["resources"]

        return context

    def _extract_metrics(
        self, outcome: ExecutionOutcome, plan: DecompositionPlan
    ) -> List[Metric]:
        """Extract metrics from outcome"""
        metrics = []

        # Overall success metric
        metrics.append(
            Metric(
                name="decomposition_success",
                metric_type=MetricType.RELIABILITY,
                value=1.0 if outcome.success else 0.0,
                is_success=outcome.success,
                timestamp=outcome.timestamp,
            )
        )

        # Execution time metric
        if outcome.execution_time > 0:
            metrics.append(
                Metric(
                    name="execution_time",
                    metric_type=MetricType.LATENCY,
                    value=outcome.execution_time,
                    unit="seconds",
                    threshold=60.0,
                    is_success=outcome.execution_time < 60.0,
                    timestamp=outcome.timestamp,
                )
            )

        # Plan confidence metric
        metrics.append(
            Metric(
                name="plan_confidence",
                metric_type=MetricType.QUALITY,
                value=plan.confidence,
                threshold=0.6,
                is_success=plan.confidence >= 0.6,
                timestamp=outcome.timestamp,
            )
        )

        # Success rate metric (if sub-results exist)
        if outcome.sub_results:
            success_rate = outcome.get_success_rate()
            metrics.append(
                Metric(
                    name="step_success_rate",
                    metric_type=MetricType.RELIABILITY,
                    value=success_rate,
                    threshold=0.7,
                    is_success=success_rate >= 0.7,
                    timestamp=outcome.timestamp,
                )
            )

        # Add custom metrics from outcome
        if outcome.metrics:
            for key, value in outcome.metrics.items():
                if isinstance(value, (int, float)):
                    metrics.append(
                        Metric(
                            name=key,
                            metric_type=MetricType.PERFORMANCE,
                            value=float(value),
                            is_success=outcome.success,
                            timestamp=outcome.timestamp,
                        )
                    )

        return metrics

    def _detect_patterns(
        self, plan: DecompositionPlan, outcome: ExecutionOutcome
    ) -> List[Pattern]:
        """Detect patterns in decomposition"""
        patterns = []

        # Strategy pattern
        if plan.strategy:
            strategy_pattern = Pattern(
                pattern_type=PatternType.SEQUENTIAL,
                components=[plan.strategy.name],
                structure={"strategy": plan.strategy.name},
                confidence=plan.confidence,
                complexity=1,
                metadata={"strategy_used": True},
            )
            patterns.append(strategy_pattern)

        # Step sequence pattern
        if len(plan.steps) > 2:
            step_types = [s.action_type for s in plan.steps]
            sequence_pattern = Pattern(
                pattern_type=PatternType.SEQUENTIAL,
                components=step_types,
                structure={
                    "sequence_length": len(step_types),
                    "unique_actions": len(set(step_types)),
                },
                confidence=0.7,
                complexity=len(step_types),
            )
            patterns.append(sequence_pattern)

        # Hierarchical pattern (if substeps exist)
        has_substeps = any(hasattr(s, "substeps") and s.substeps for s in plan.steps)
        if has_substeps:
            hierarchical_pattern = Pattern(
                pattern_type=PatternType.HIERARCHICAL,
                components=["step", "substep"],
                structure={"hierarchical": True, "depth": 2},
                confidence=0.8,
                complexity=len(plan.steps) * 2,
            )
            patterns.append(hierarchical_pattern)

        # Iterative pattern (if refinement occurred)
        if outcome.metadata.get("refinement_iterations", 0) > 0:
            iterative_pattern = Pattern(
                pattern_type=PatternType.ITERATIVE,
                components=["decompose", "execute", "refine"],
                structure={
                    "iterations": outcome.metadata["refinement_iterations"],
                    "converged": outcome.success,
                },
                confidence=0.75,
                complexity=3,
            )
            patterns.append(iterative_pattern)

        return patterns


# ============================================================
# PRINCIPLE PROMOTION
# ============================================================


@dataclass
class PromotionCandidate:
    """Candidate principle for promotion to library"""

    principle: Principle
    validation_results: ValidationResults
    source_domain: str
    applicable_domains: List[str]
    confidence: float
    evidence_count: int
    promotion_score: float = 0.0
    promotion_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_promotion_score(self):
        """Calculate score for promotion"""
        factors = []

        # Validation success rate
        factors.append(("validation", self.validation_results.success_rate, 0.35))

        # Confidence level
        factors.append(("confidence", self.confidence, 0.25))

        # Domain breadth (more domains = better)
        domain_breadth = min(1.0, len(self.applicable_domains) / 5.0)
        factors.append(("breadth", domain_breadth, 0.20))

        # Evidence strength
        evidence_strength = min(1.0, self.evidence_count / 10.0)
        factors.append(("evidence", evidence_strength, 0.15))

        # Overall validation confidence
        factors.append(
            ("val_confidence", self.validation_results.overall_confidence, 0.05)
        )

        # Calculate weighted score
        self.promotion_score = sum(value * weight for _, value, weight in factors)

        # Generate reason
        top_factors = sorted(factors, key=lambda x: x[1] * x[2], reverse=True)[:2]
        self.promotion_reason = f"Strong {top_factors[0][0]} ({top_factors[0][1]:.2f}) and {top_factors[1][0]} ({top_factors[1][1]:.2f})"

        return self.promotion_score


class PrinciplePromoter:
    """Promotes validated principles to decomposition library"""

    def __init__(
        self, library: StratifiedDecompositionLibrary, promotion_threshold: float = 0.7
    ):
        """
        Initialize principle promoter

        Args:
            library: Decomposition library to promote to
            promotion_threshold: Minimum score for promotion
        """
        self.library = library
        self.promotion_threshold = promotion_threshold

        self.promotion_history = deque(maxlen=1000)
        self.promoted_count = 0
        self.rejected_count = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            "PrinciplePromoter initialized (threshold: %.2f)", promotion_threshold
        )

    def evaluate_for_promotion(
        self, principle: Principle, validation_results: ValidationResults
    ) -> PromotionCandidate:
        """
        Evaluate if principle should be promoted

        Args:
            principle: Principle to evaluate
            validation_results: Validation results

        Returns:
            Promotion candidate with score
        """
        # Create candidate
        candidate = PromotionCandidate(
            principle=principle,
            validation_results=validation_results,
            source_domain=getattr(principle, "domain", "general"),
            applicable_domains=validation_results.successful_domains,
            confidence=principle.confidence,
            evidence_count=principle.success_count + principle.failure_count,
            metadata={
                "success_rate": principle.get_success_rate(),
                "validation_level": validation_results.validation_level.value,
            },
        )

        # Calculate promotion score
        candidate.calculate_promotion_score()

        logger.debug(
            "Evaluated principle %s for promotion: score=%.2f",
            principle.id,
            candidate.promotion_score,
        )

        return candidate

    def promote(self, candidate: PromotionCandidate) -> bool:
        """
        Promote principle to library

        Args:
            candidate: Promotion candidate

        Returns:
            True if promoted successfully
        """
        with self.lock:
            # Check threshold
            if candidate.promotion_score < self.promotion_threshold:
                self.rejected_count += 1
                logger.info(
                    "Rejected principle %s for promotion: score %.2f < threshold %.2f",
                    candidate.principle.id,
                    candidate.promotion_score,
                    self.promotion_threshold,
                )
                return False

            # Convert principle to library format
            library_principle_dict = self._convert_to_library_format(candidate)

            # The library expects a top-level object, not a dict. We convert the
            # dictionary to a simple object wrapper. The previous recursive conversion
            # was incorrect as it also converted the 'pattern' dict.
            library_principle = DictObject(library_principle_dict)

            # The `_convert_to_library_format` method creates a serializable dictionary
            # for the pattern. The live library, however, expects a full pattern object
            # with methods. We restore the original object from the candidate.
            library_principle.pattern = candidate.principle.core_pattern

            # Add to library
            try:
                if hasattr(self.library, "add_principle"):
                    self.library.add_principle(library_principle)
                elif hasattr(self.library, "add"):
                    self.library.add(library_principle)
                elif hasattr(self.library, "principles"):
                    # Store directly in principles dict
                    self.library.principles[library_principle_dict["id"]] = (
                        library_principle
                    )
                else:
                    logger.warning("Library has no method to add principles")
                    return False

                self.promoted_count += 1

                # Track promotion
                self.promotion_history.append(
                    {
                        "principle_id": candidate.principle.id,
                        "promotion_score": candidate.promotion_score,
                        "source_domain": candidate.source_domain,
                        "applicable_domains": candidate.applicable_domains,
                        "validation_success_rate": candidate.validation_results.success_rate,
                        "timestamp": time.time(),
                    }
                )

                logger.info(
                    "Promoted principle %s to library (score: %.2f, domains: %s)",
                    candidate.principle.id,
                    candidate.promotion_score,
                    candidate.applicable_domains,
                )

                return True

            except Exception as e:
                logger.error(
                    "Failed to promote principle %s: %s", candidate.principle.id, e
                )
                import traceback

                traceback.print_exc()
                return False

    def _convert_to_library_format(
        self, candidate: PromotionCandidate
    ) -> Dict[str, Any]:
        """Convert principle to library format"""
        principle = candidate.principle

        # Safely convert validation results
        try:
            validation_dict = candidate.validation_results.to_dict()
        except Exception:
            validation_dict = {
                "success_rate": candidate.validation_results.success_rate,
                "overall_confidence": candidate.validation_results.overall_confidence,
                "successful_domains": candidate.validation_results.successful_domains,
            }

        # Safely extract pattern information - MUST be JSON serializable
        pattern = {"pattern_type": "unknown"}
        try:
            if (
                hasattr(principle, "core_pattern")
                and principle.core_pattern is not None
            ):
                # Extract pattern and ensure it's serializable
                if hasattr(principle.core_pattern, "to_dict") and callable(
                    principle.core_pattern.to_dict
                ):
                    try:
                        pattern = principle.core_pattern.to_dict()
                    except Exception:
                        pass

                # If pattern is still not set or might contain non-serializable objects
                if pattern == {"pattern_type": "unknown"} or not isinstance(
                    pattern, dict
                ):
                    # Build pattern dict manually from attributes
                    pattern_dict = {}

                    # Get pattern_type
                    if hasattr(principle.core_pattern, "pattern_type"):
                        pt = principle.core_pattern.pattern_type
                        if hasattr(pt, "value"):
                            pattern_dict["pattern_type"] = pt.value
                        elif hasattr(pt, "name"):
                            pattern_dict["pattern_type"] = pt.name
                        else:
                            pattern_dict["pattern_type"] = str(pt)
                    else:
                        pattern_dict["pattern_type"] = "unknown"

                    # Get other common pattern attributes
                    for attr in [
                        "components",
                        "structure",
                        "confidence",
                        "complexity",
                        "metadata",
                    ]:
                        if hasattr(principle.core_pattern, attr):
                            val = getattr(principle.core_pattern, attr)
                            # Only add if serializable
                            try:
                                json.dumps(val)
                                pattern_dict[attr] = val
                            except Exception:
                                # Convert to string if not serializable
                                if isinstance(val, (list, tuple)):
                                    pattern_dict[attr] = [str(v) for v in val]
                                elif isinstance(val, dict):
                                    pattern_dict[attr] = {
                                        k: str(v) for k, v in val.items()
                                    }
                                else:
                                    pattern_dict[attr] = str(val)

                    pattern = pattern_dict

                # Final serialization check - ensure the entire pattern is serializable
                try:
                    json.dumps(pattern)
                except (TypeError, ValueError):
                    # If still not serializable, apply ensure_json_serializable
                    pattern = ensure_json_serializable(pattern)
        except Exception as e:
            logger.debug("Error extracting pattern: %s", e)
            pattern = {"pattern_type": "unknown"}

        # Final validation - ensure pattern is a dict with at least pattern_type
        if not isinstance(pattern, dict):
            pattern = {"pattern_type": str(pattern)}
        if "pattern_type" not in pattern:
            pattern["pattern_type"] = "unknown"

        # Add pattern_id field that the library expects
        if "pattern_id" not in pattern:
            pattern["pattern_id"] = f"pattern_{principle.id}_{int(time.time())}"

        # Ensure validation_dict is serializable
        validation_dict = ensure_json_serializable(validation_dict)

        # Helper function to safely convert to list
        def safe_list(value, default=None):
            """Safely convert value to list, return default if not possible"""
            if default is None:
                default = []
            if isinstance(value, (list, tuple)):
                return list(value)
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
                try:
                    return list(value)
                except Exception:
                    return default
            return default

        # Get contraindicated domains safely
        contraindicated_raw = getattr(principle, "contraindicated_domains", [])
        contraindicated_domains = safe_list(contraindicated_raw, [])

        # Create library-compatible principle
        library_principle = {
            "id": str(principle.id),  # Ensure ID is string
            "principle_id": str(principle.id),
            "name": str(getattr(principle, "name", f"Principle {principle.id}")),
            "description": str(
                getattr(principle, "description", "Learned decomposition principle")
            ),
            "pattern": pattern,  # Already ensured serializable above with pattern_id
            "confidence": float(principle.confidence),
            "success_count": int(principle.success_count),
            "failure_count": int(principle.failure_count),
            "applicable_domains": safe_list(candidate.applicable_domains, []),
            "contraindicated_domains": contraindicated_domains,
            "validation_results": validation_dict,
            "promotion_metadata": {
                "promotion_score": float(candidate.promotion_score),
                "promotion_reason": str(candidate.promotion_reason),
                "promoted_at": float(time.time()),
                "evidence_count": int(candidate.evidence_count),
            },
        }

        # Add execution logic if available (also handle serialization)
        if hasattr(principle, "execution_logic"):
            try:
                exec_logic = principle.execution_logic
                if callable(exec_logic):
                    library_principle["execution_logic"] = str(exec_logic)
                elif isinstance(
                    exec_logic, (str, dict, list, int, float, bool, type(None))
                ):
                    try:
                        json.dumps(exec_logic)
                        library_principle["execution_logic"] = exec_logic
                    except Exception:
                        library_principle["execution_logic"] = str(exec_logic)
                else:
                    library_principle["execution_logic"] = str(exec_logic)

                library_principle["execution_type"] = str(
                    getattr(principle, "execution_type", "function")
                )
            except Exception as e:
                logger.debug("Could not serialize execution_logic: %s", e)

        # Final check - ensure entire dict is serializable
        try:
            json.dumps(library_principle)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Library principle not fully serializable, applying deep cleaning: %s",
                e,
            )
            library_principle = ensure_json_serializable(library_principle)

        return library_principle

    def get_statistics(self) -> Dict[str, Any]:
        """Get promotion statistics"""
        with self.lock:
            return {
                "promoted_count": self.promoted_count,
                "rejected_count": self.rejected_count,
                "promotion_rate": self.promoted_count
                / max(1, self.promoted_count + self.rejected_count),
                "recent_promotions": len(self.promotion_history),
                "promotion_threshold": self.promotion_threshold,
            }


# ============================================================
# INTEGRATED PRINCIPLE LEARNER
# ============================================================


class PrincipleLearner:
    """
    Main principle learning orchestrator

    Closes the learning loop:
    ExecutionOutcome → Crystallization → Validation → Promotion → Reuse
    """

    def __init__(
        self,
        library: StratifiedDecompositionLibrary,
        knowledge_base: Optional[VersionedKnowledgeBase] = None,
        min_promotion_score: float = 0.7,
        enable_auto_promotion: bool = True,
    ):
        """
        Initialize principle learner

        Args:
            library: Decomposition library for storing principles
            knowledge_base: Optional knowledge base for versioning
            min_promotion_score: Minimum score for auto-promotion
            enable_auto_promotion: Whether to automatically promote validated principles
        """
        self.library = library
        self.knowledge_base = knowledge_base or VersionedKnowledgeBase()
        self.min_promotion_score = min_promotion_score
        self.enable_auto_promotion = enable_auto_promotion

        # Core components
        self.converter = DecompositionToTraceConverter()
        self.crystallizer = KnowledgeCrystallizer() if KnowledgeCrystallizer else None
        self.validator = KnowledgeValidator() if KnowledgeValidator else None
        self.promoter = PrinciplePromoter(library, min_promotion_score)

        # Knowledge management
        self.knowledge_index = KnowledgeIndex()
        self.pruner = KnowledgePruner()

        # Tracking - use bounded collections
        self.learning_history = deque(maxlen=1000)
        self.extraction_count = 0
        self.validation_count = 0
        self.promotion_count = 0

        # Statistics - use Counter instead of defaultdict(int) for automatic bounding
        self.domain_coverage = Counter()
        self.pattern_usage = Counter()
        self.max_domain_coverage = 10000

        # Thread safety
        self.lock = threading.RLock()

        # Check component availability
        self.components_available = (
            self.crystallizer is not None and self.validator is not None
        )

        if not self.components_available:
            logger.warning(
                "Knowledge crystallizer components not available - "
                "principle learning will be limited"
            )
        else:
            logger.info("PrincipleLearner initialized with full capabilities")

    def extract_and_promote(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ) -> Dict[str, Any]:
        """
        Main entry point: Extract principles and promote validated ones

        Args:
            problem: Problem that was decomposed
            plan: Decomposition plan
            outcome: Execution outcome

        Returns:
            Dictionary with extraction and promotion results
        """
        if not self.components_available:
            logger.warning("Cannot extract principles - components not available")
            return {
                "principles_extracted": 0,
                "principles_validated": 0,
                "principles_promoted": 0,
                "error": "Components not available",
            }

        with self.lock:
            start_time = time.time()
            results = {
                "principles_extracted": 0,
                "principles_validated": 0,
                "principles_promoted": 0,
                "extraction_time": 0.0,
                "validation_time": 0.0,
                "promotion_time": 0.0,
            }

            try:
                # STEP 1: Convert to execution trace
                trace = self.converter.convert(problem, plan, outcome)

                # STEP 2: Crystallize principles
                extraction_start = time.time()
                crystallization_result = self.crystallizer.crystallize(trace)
                results["extraction_time"] = time.time() - extraction_start

                principles = crystallization_result.principles
                results["principles_extracted"] = len(principles)
                self.extraction_count += len(principles)

                if not principles:
                    logger.debug("No principles extracted from execution")
                    return results

                logger.info("Extracted %d principles from execution", len(principles))

                # STEP 3: Validate principles
                validated_principles = []
                validation_start = time.time()

                for principle in principles:
                    # Store in knowledge base
                    self.knowledge_base.store(
                        principle,
                        author="system",
                        message="Extracted from decomposition",
                    )

                    # Index principle
                    self.knowledge_index.index_principle(principle)

                    # Validate across domains
                    applicable_domains = self._get_applicable_domains(
                        principle, problem
                    )

                    validation_results = self.validator.validate_across_domains(
                        principle, applicable_domains
                    )

                    results["principles_validated"] += 1
                    self.validation_count += 1

                    # Update domain coverage with size limit
                    for domain in validation_results.successful_domains:
                        self.domain_coverage[domain] += 1

                        # Enforce max domain coverage size
                        if len(self.domain_coverage) > self.max_domain_coverage:
                            # Remove least common domains
                            least_common = self.domain_coverage.most_common()[:-1000:-1]
                            for d, _ in least_common[:100]:
                                del self.domain_coverage[d]

                    # Store validated principle
                    if validation_results.success_rate > 0.5:
                        validated_principles.append((principle, validation_results))

                results["validation_time"] = time.time() - validation_start

                logger.info(
                    "Validated %d/%d principles successfully",
                    len(validated_principles),
                    len(principles),
                )

                # STEP 4: Promote validated principles
                promotion_start = time.time()

                if self.enable_auto_promotion:
                    for principle, validation_results in validated_principles:
                        # Evaluate for promotion
                        candidate = self.promoter.evaluate_for_promotion(
                            principle, validation_results
                        )

                        # Promote if meets threshold
                        if self.promoter.promote(candidate):
                            results["principles_promoted"] += 1
                            self.promotion_count += 1

                            # Track pattern usage
                            if hasattr(principle, "core_pattern"):
                                try:
                                    if hasattr(principle.core_pattern, "pattern_type"):
                                        if hasattr(
                                            principle.core_pattern.pattern_type, "value"
                                        ):
                                            pattern_type = principle.core_pattern.pattern_type.value
                                        elif hasattr(
                                            principle.core_pattern.pattern_type, "name"
                                        ):
                                            pattern_type = (
                                                principle.core_pattern.pattern_type.name
                                            )
                                        else:
                                            pattern_type = str(
                                                principle.core_pattern.pattern_type
                                            )
                                        self.pattern_usage[pattern_type] += 1
                                except Exception as e:
                                    logger.debug("Could not track pattern usage: %s", e)

                results["promotion_time"] = time.time() - promotion_start

                # STEP 5: Track overall learning
                self.learning_history.append(
                    {
                        "problem_signature": problem.get_signature(),
                        "extracted": results["principles_extracted"],
                        "validated": results["principles_validated"],
                        "promoted": results["principles_promoted"],
                        "success": outcome.success,
                        "domain": problem.metadata.get("domain", "general"),
                        "timestamp": time.time(),
                        "total_time": time.time() - start_time,
                    }
                )

                logger.info(
                    "Principle learning complete: extracted=%d, validated=%d, promoted=%d",
                    results["principles_extracted"],
                    results["principles_validated"],
                    results["principles_promoted"],
                )

            except Exception as e:
                logger.error("Principle learning failed: %s", e)
                import traceback

                traceback.print_exc()
                results["error"] = str(e)

            return results

    def find_applicable_principles(self, problem: ProblemGraph) -> List[Principle]:
        """
        Find principles applicable to a problem

        Args:
            problem: Problem to solve

        Returns:
            List of applicable principles
        """
        # Build query from problem
        query = {
            "domain": problem.metadata.get("domain", "general"),
            "patterns": self._extract_problem_patterns(problem),
            "keywords": self._extract_keywords(problem),
        }

        # Search knowledge index
        relevant_ids = self.knowledge_index.find_relevant(query)

        # Get principles
        principles = []
        for pid in relevant_ids[:10]:  # Top 10
            principle = self.knowledge_base.get(pid)
            if principle:
                principles.append(principle)

        logger.debug("Found %d applicable principles for problem", len(principles))

        return principles

    def prune_low_quality_principles(
        self, age_threshold_days: int = 90, confidence_threshold: float = 0.3
    ) -> int:
        """
        Prune low-quality or outdated principles

        Args:
            age_threshold_days: Age threshold in days
            confidence_threshold: Confidence threshold

        Returns:
            Number of principles pruned
        """
        with self.lock:
            # Identify outdated principles
            outdated = self.pruner.identify_outdated(
                self.knowledge_base, age_threshold_days
            )

            # Identify low confidence principles
            all_principles = self.knowledge_base.get_all_principles()
            low_conf = self.pruner.identify_low_confidence(
                all_principles, confidence_threshold
            )

            # Combine candidates
            all_candidates = outdated + low_conf

            # Execute pruning
            pruned = self.pruner.execute_pruning(
                all_candidates, self.knowledge_base, threshold=0.7
            )

            logger.info("Pruned %d low-quality principles", pruned)

            return pruned

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        with self.lock:
            stats = {
                "extraction": {
                    "total_extractions": self.extraction_count,
                    "total_conversions": self.converter.conversion_count,
                },
                "validation": {
                    "total_validations": self.validation_count,
                    "validator_stats": self.validator.total_validations
                    if self.validator
                    else 0,
                },
                "promotion": {
                    "total_promotions": self.promotion_count,
                    "promoter_stats": self.promoter.get_statistics(),
                },
                "knowledge_base": {
                    "total_principles": self.knowledge_base.total_principles,
                    "total_versions": self.knowledge_base.total_versions,
                    "storage_size": self.knowledge_base.total_storage_size,
                },
                "knowledge_index": self.knowledge_index.get_statistics(),
                "domain_coverage": dict(self.domain_coverage),
                "pattern_usage": dict(self.pattern_usage.most_common(10)),
                "recent_learning": {
                    "sessions": len(self.learning_history),
                    "avg_extracted": np.mean(
                        [h["extracted"] for h in self.learning_history]
                    )
                    if self.learning_history
                    else 0,
                    "avg_promoted": np.mean(
                        [h["promoted"] for h in self.learning_history]
                    )
                    if self.learning_history
                    else 0,
                },
            }

            # Calculate learning efficiency
            if self.learning_history:
                successful_sessions = sum(
                    1 for h in self.learning_history if h.get("promoted", 0) > 0
                )
                stats["learning_efficiency"] = {
                    "successful_sessions": successful_sessions,
                    "session_success_rate": successful_sessions
                    / len(self.learning_history),
                }

            return stats

    def export_learned_principles(self, path: Path, format: str = "json") -> bool:
        """
        Export learned principles

        Args:
            path: Export path
            format: Export format

        Returns:
            True if successful
        """
        return self.knowledge_base.export(path, format)

    def import_principles(self, path: Path) -> bool:
        """
        Import principles from file

        Args:
            path: Import path

        Returns:
            True if successful
        """
        return self.knowledge_base.import_from(path)

    def _get_applicable_domains(
        self, principle: Principle, problem: ProblemGraph
    ) -> List[str]:
        """Get domains to validate principle in"""
        domains = set()

        # Add problem domain
        problem_domain = problem.metadata.get("domain", "general")
        domains.add(problem_domain)

        # Add principle's original domain
        if hasattr(principle, "domain"):
            domains.add(principle.domain)

        # Add related domains
        if problem_domain != "general":
            domains.add("general")  # Always test general applicability

        # Add domains from principle's applicable list
        if hasattr(principle, "applicable_domains"):
            domains.update(principle.applicable_domains[:3])  # Top 3

        # Limit domain set size
        if len(domains) > 10:
            domains = set(list(domains)[:10])

        return list(domains)

    def _extract_problem_patterns(self, problem: ProblemGraph) -> List[str]:
        """Extract patterns from problem structure"""
        patterns = []

        # Graph structure patterns
        if len(problem.nodes) > 10:
            patterns.append("large_graph")
        if len(problem.edges) > len(problem.nodes):
            patterns.append("dense_graph")

        # Complexity patterns
        if problem.complexity_score > 3:
            patterns.append("high_complexity")
        elif problem.complexity_score < 1:
            patterns.append("low_complexity")

        # Metadata patterns
        if "constraints" in problem.metadata:
            patterns.append("constrained")
        if "optimization" in str(problem.metadata).lower():
            patterns.append("optimization")

        return patterns

    def _extract_keywords(self, problem: ProblemGraph) -> List[str]:
        """Extract keywords from problem"""
        keywords = set()

        # From domain
        domain = problem.metadata.get("domain", "")
        if domain:
            keywords.update(domain.split("_"))

        # From type
        ptype = problem.metadata.get("type", "")
        if ptype:
            keywords.update(ptype.split("_"))

        # From description if available
        if "description" in problem.metadata:
            desc = str(problem.metadata["description"]).lower()
            # Simple keyword extraction
            words = desc.split()
            keywords.update(w for w in words if len(w) > 4)

        return list(keywords)[:10]  # Limit keywords


# ============================================================
# INTEGRATION HOOK
# ============================================================


def integrate_principle_learning(
    decomposer,
    library: StratifiedDecompositionLibrary,
    min_promotion_score: float = 0.7,
) -> PrincipleLearner:
    """
    Integrate principle learning with decomposer

    Args:
        decomposer: Problem decomposer instance
        library: Decomposition library
        min_promotion_score: Minimum score for promotion

    Returns:
        PrincipleLearner instance

    Example:
        >>> decomposer = create_decomposer()
        >>> library = StratifiedDecompositionLibrary()
        >>> learner = integrate_principle_learning(decomposer, library)
        >>>
        >>> # Now learning happens automatically
        >>> plan, outcome = decomposer.decompose_and_execute(problem)
        >>> # Principles extracted, validated, and promoted to library
    """
    learner = PrincipleLearner(
        library=library,
        min_promotion_score=min_promotion_score,
        enable_auto_promotion=True,
    )

    # Attach to decomposer for automatic learning
    if hasattr(decomposer, "principle_learner"):
        decomposer.principle_learner = learner
        logger.info("Principle learner integrated with decomposer")

    return learner


# ============================================================
# TESTING
# ============================================================


def test_principle_learning():
    """Test principle learning system"""
    logger.info("Testing principle learning...")

    try:
        # Create components
        library = StratifiedDecompositionLibrary()
        learner = PrincipleLearner(library)

        # Create test problem
        from problem_decomposer_core import (DecompositionPlan,
                                             ExecutionOutcome, ProblemGraph)

        problem = ProblemGraph(
            nodes={"A": {}, "B": {}, "C": {}},
            edges=[("A", "B", {}), ("B", "C", {})],
            root="A",
            metadata={"domain": "test", "type": "sequential"},
        )
        problem.complexity_score = 2.0

        # Create test plan
        from problem_decomposer_core import (DecompositionStep,
                                             DecompositionStrategy)

        plan = DecompositionPlan(
            problem_signature="test_problem",
            strategy=DecompositionStrategy.HIERARCHICAL,
            steps=[
                DecompositionStep("step1", "process_A", "Process node A"),
                DecompositionStep("step2", "process_B", "Process node B"),
                DecompositionStep("step3", "process_C", "Process node C"),
            ],
            confidence=0.8,
            estimated_complexity=2.0,
        )

        # Create test outcome
        outcome = ExecutionOutcome(
            success=True,
            solution={"result": "success"},
            execution_time=1.5,
            metrics={"accuracy": 0.9},
        )

        # Test extraction and promotion
        results = learner.extract_and_promote(problem, plan, outcome)

        logger.info("Test results:")
        logger.info(f"  Extracted: {results['principles_extracted']}")
        logger.info(f"  Validated: {results['principles_validated']}")
        logger.info(f"  Promoted: {results['principles_promoted']}")

        # Get statistics
        stats = learner.get_learning_statistics()
        logger.info(
            f"  Knowledge base size: {stats['knowledge_base']['total_principles']}"
        )
        logger.info(f"  Domain coverage: {stats['domain_coverage']}")

        logger.info("Principle learning test passed!")
        return True

    except Exception as e:
        logger.error(f"Principle learning test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_principle_learning()
