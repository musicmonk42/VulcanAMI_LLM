"""
knowledge_crystallizer_core.py - Main crystallization orchestrator for Knowledge Crystallizer
Part of the VULCAN-AGI system
"""

import copy
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .contraindication_tracker import (
    CascadeAnalyzer,
    Contraindication,
    ContraindicationDatabase,
    ContraindicationGraph,
)
from .crystallization_selector import CrystallizationMethod, CrystallizationSelector
from .knowledge_storage import VersionedKnowledgeBase

# Import other components
from .principle_extractor import Principle, PrincipleExtractor
from .validation_engine import KnowledgeValidator, ValidationResult

logger = logging.getLogger(__name__)


class CrystallizationMode(Enum):
    """Modes of crystallization"""

    STANDARD = "standard"
    CASCADE_AWARE = "cascade_aware"
    INCREMENTAL = "incremental"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


class ApplicationMode(Enum):
    """Modes of knowledge application"""

    DIRECT = "direct"
    ADAPTED = "adapted"
    COMBINED = "combined"
    EXPERIMENTAL = "experimental"


@dataclass
class ExecutionTrace:
    """Execution trace for crystallization"""

    trace_id: str
    actions: List[Dict[str, Any]]
    outcomes: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None
    iteration: Optional[int] = None  # For incremental learning
    batch_id: Optional[str] = None  # For batch processing

    def get_signature(self) -> str:
        """Get unique signature for trace"""
        content = json.dumps(
            {"actions": self.actions, "context": self.context}, sort_keys=True
        )
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()


@dataclass
class CrystallizationResult:
    """Result of crystallization process"""

    principles: List[Principle]
    validation_results: List[ValidationResult]
    contraindications: List[Contraindication]
    confidence: float
    mode: CrystallizationMode
    metadata: Dict[str, Any] = field(default_factory=dict)
    method_used: Optional[CrystallizationMethod] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "principles": [
                p.to_dict() if hasattr(p, "to_dict") else str(p)
                for p in self.principles
            ],
            "validation_results": [
                v.to_dict() if hasattr(v, "to_dict") else str(v)
                for v in self.validation_results
            ],
            "contraindications": [c.to_dict() for c in self.contraindications],
            "confidence": self.confidence,
            "mode": self.mode.value,
            "method_used": self.method_used.value if self.method_used else None,
            "metadata": self.metadata,
        }


@dataclass
class ApplicationResult:
    """Result of knowledge application"""

    principle_used: Optional[Principle]
    solution: Any
    confidence: float
    adaptations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "principle_used": (
                self.principle_used.to_dict()
                if self.principle_used and hasattr(self.principle_used, "to_dict")
                else str(self.principle_used)
            ),
            "solution": self.solution,
            "confidence": self.confidence,
            "adaptations": self.adaptations,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
        }


class ImbalanceHandler:
    """Handles imbalances in knowledge distribution"""

    def __init__(self):
        """Initialize imbalance handler"""
        self.domain_counts = defaultdict(int)
        self.principle_types = defaultdict(int)
        self.imbalance_threshold = 0.3

        logger.info("ImbalanceHandler initialized")

    def detect_imbalance(self, knowledge_base) -> Dict[str, float]:
        """
        Detect imbalances in knowledge

        Args:
            knowledge_base: Knowledge base to analyze

        Returns:
            Dictionary of imbalance scores by category
        """
        imbalances = {}

        # Get principle distribution
        principles = knowledge_base.get_all_principles()

        # Analyze domain distribution
        domain_dist = defaultdict(int)
        for principle in principles:
            if hasattr(principle, "domain"):
                domain_dist[principle.domain] += 1

        if domain_dist:
            counts = list(domain_dist.values())
            mean_count = np.mean(counts)
            std_count = np.std(counts) if len(counts) > 1 else 0

            # Calculate imbalance score
            if mean_count > 0:
                imbalance_score = std_count / mean_count
                imbalances["domain"] = min(1.0, imbalance_score)

        # Analyze type distribution
        type_dist = defaultdict(int)
        for principle in principles:
            if hasattr(principle, "type"):
                type_dist[principle.type] += 1

        if type_dist:
            counts = list(type_dist.values())
            mean_count = np.mean(counts)
            std_count = np.std(counts) if len(counts) > 1 else 0

            if mean_count > 0:
                imbalance_score = std_count / mean_count
                imbalances["type"] = min(1.0, imbalance_score)

        return imbalances

    def suggest_focus_areas(self, imbalances: Dict[str, float]) -> List[str]:
        """
        Suggest areas to focus on for balance

        Args:
            imbalances: Imbalance scores

        Returns:
            List of suggested focus areas
        """
        suggestions = []

        for category, score in imbalances.items():
            if score > self.imbalance_threshold:
                if category == "domain":
                    suggestions.append("Explore underrepresented domains")
                elif category == "type":
                    suggestions.append("Diversify principle types")

        return suggestions


class KnowledgeCrystallizer:
    """Main crystallization orchestrator"""

    def __init__(self, vulcan_memory=None, semantic_bridge=None):
        """
        Initialize knowledge crystallizer with adaptive configuration.
        
        The crystallizer supports different operational modes via environment variables:
        
        - VULCAN_DEVELOPMENT_MODE: Enable relaxed thresholds for experimental learning
        - VULCAN_MIN_CONFIDENCE_THRESHOLD: Explicitly set minimum confidence (0.0-1.0)

        Args:
            vulcan_memory: VULCAN memory system for storing crystallized knowledge
            semantic_bridge: Semantic bridge component for knowledge integration
        """
        self.memory = vulcan_memory
        self.semantic = semantic_bridge

        # Core components
        self.extractor = PrincipleExtractor()
        self.validator = KnowledgeValidator()
        self.contraindication_db = ContraindicationDatabase()
        self.contraindication_graph = ContraindicationGraph()
        self.cascade_analyzer = CascadeAnalyzer(
            self.contraindication_db, self.contraindication_graph
        )
        self.knowledge_base = VersionedKnowledgeBase()
        self.imbalance_handler = ImbalanceHandler()
        self.method_selector = CrystallizationSelector()

        # Application component
        self.applicator = KnowledgeApplicator(self)

        # Tracking
        self.crystallization_history = deque(maxlen=1000)
        self.application_history = deque(maxlen=1000)
        self.incremental_state = {}  # For incremental crystallization
        self.batch_accumulator = defaultdict(list)  # For batch processing

        # Configuration - Adaptive thresholds based on environment
        # VULCAN_DEVELOPMENT_MODE enables more lenient crystallization for experimental learning
        self.development_mode = self._detect_development_mode()
        self.min_confidence_threshold = self._determine_confidence_threshold()
        self.cascade_detection_enabled = True

        # Thread safety
        self.lock = threading.RLock()

        # Track component availability
        self.has_memory = vulcan_memory is not None
        self.has_semantic = semantic_bridge is not None

        logger.info(
            "KnowledgeCrystallizer initialized (memory: %s, semantic: %s, "
            "development_mode: %s, min_confidence: %.2f)",
            self.has_memory,
            self.has_semantic,
            self.development_mode,
            self.min_confidence_threshold,
        )
    
    def _detect_development_mode(self) -> bool:
        """
        Detect if running in development mode via environment variable.
        
        Development mode enables more lenient crystallization thresholds to allow
        experimental learning and exploration without strict production constraints.
        
        Returns:
            True if VULCAN_DEVELOPMENT_MODE is set to a truthy value, False otherwise
        """
        env_value = os.environ.get("VULCAN_DEVELOPMENT_MODE", "").strip().lower()
        is_dev_mode = env_value in ("true", "1", "yes", "on")
        
        if is_dev_mode:
            logger.info(
                "Development mode ENABLED via VULCAN_DEVELOPMENT_MODE environment variable. "
                "Using relaxed crystallization thresholds for experimental learning."
            )
        
        return is_dev_mode
    
    def _determine_confidence_threshold(self) -> float:
        """
        Determine the minimum confidence threshold for crystallization.
        
        In development mode, uses a lower threshold (0.2) to enable more exploratory
        learning. In production mode, uses a higher threshold (0.4) for quality assurance.
        
        Can be overridden via VULCAN_MIN_CONFIDENCE_THRESHOLD environment variable.
        
        Returns:
            Minimum confidence threshold (0.0-1.0)
        """
        # Check for explicit override first
        env_threshold = os.environ.get("VULCAN_MIN_CONFIDENCE_THRESHOLD", "").strip()
        if env_threshold:
            try:
                threshold = float(env_threshold)
                if 0.0 <= threshold <= 1.0:
                    logger.info(
                        f"Using min_confidence_threshold={threshold:.2f} from "
                        f"VULCAN_MIN_CONFIDENCE_THRESHOLD environment variable"
                    )
                    return threshold
                else:
                    logger.warning(
                        f"Invalid VULCAN_MIN_CONFIDENCE_THRESHOLD value '{env_threshold}' "
                        f"(must be 0.0-1.0). Using default based on mode."
                    )
            except ValueError:
                logger.warning(
                    f"Invalid VULCAN_MIN_CONFIDENCE_THRESHOLD value '{env_threshold}' "
                    f"(must be numeric). Using default based on mode."
                )
        
        # Use mode-based defaults
        if self.development_mode:
            threshold = 0.2  # Lower for development/exploration
            logger.info(
                f"Using development mode min_confidence_threshold={threshold:.2f} "
                f"(set VULCAN_MIN_CONFIDENCE_THRESHOLD to override)"
            )
        else:
            threshold = 0.4  # Higher for production quality
            logger.info(
                f"Using production mode min_confidence_threshold={threshold:.2f} "
                f"(set VULCAN_MIN_CONFIDENCE_THRESHOLD to override)"
            )
        
        return threshold

    def crystallize(
        self, execution_trace: ExecutionTrace, context: Optional[Dict[str, Any]] = None
    ) -> CrystallizationResult:
        """
        Main crystallization entry point following EXAMINE → SELECT → APPLY → REMEMBER

        Args:
            execution_trace: Execution trace to crystallize
            context: Additional context for crystallization

        Returns:
            Crystallization result
        """
        with self.lock:
            start_time = time.time()
            context = context or {}

            # EXAMINE: Analyze trace and select method
            method_selection = self.method_selector.select_method(
                execution_trace, context
            )

            logger.info(
                "Selected %s method with confidence %.2f: %s",
                method_selection.method.value,
                method_selection.confidence,
                method_selection.reasoning,
            )

            # SELECT: Route to appropriate crystallization method
            try:
                if method_selection.method == CrystallizationMethod.CASCADE_AWARE:
                    result = self.crystallize_with_cascade_detection(execution_trace)
                elif method_selection.method == CrystallizationMethod.INCREMENTAL:
                    result = self._crystallize_incremental(
                        execution_trace, method_selection.parameters
                    )
                elif method_selection.method == CrystallizationMethod.BATCH:
                    result = self._crystallize_batch(
                        [execution_trace], method_selection.parameters
                    )
                elif method_selection.method == CrystallizationMethod.ADAPTIVE:
                    result = self._crystallize_adaptive(
                        execution_trace, method_selection.parameters
                    )
                elif method_selection.method == CrystallizationMethod.HYBRID:
                    result = self._crystallize_hybrid(
                        execution_trace, method_selection.parameters
                    )
                else:
                    result = self.crystallize_experience(execution_trace)

                # Set method used in result
                result.method_used = method_selection.method

            except Exception as e:
                logger.error(
                    "Crystallization failed with %s, trying fallback: %s",
                    method_selection.method.value,
                    e,
                )

                # Try fallback methods
                for fallback_method in method_selection.fallback_methods:
                    try:
                        if fallback_method == CrystallizationMethod.STANDARD:
                            result = self.crystallize_experience(execution_trace)
                            result.method_used = fallback_method
                            break
                    except Exception as fallback_error:
                        logger.error(
                            "Fallback %s also failed: %s",
                            fallback_method.value,
                            fallback_error,
                        )
                        continue
                else:
                    # All methods failed, return empty result
                    result = CrystallizationResult(
                        principles=[],
                        validation_results=[],
                        contraindications=[],
                        confidence=0.0,
                        mode=CrystallizationMode.STANDARD,
                        metadata={"error": str(e)},
                    )

            # APPLY: Store results
            for principle in result.principles:
                self._store_principle(principle)

            # REMEMBER: Track crystallization
            self.crystallization_history.append(
                {
                    "trace_id": execution_trace.trace_id,
                    "method": method_selection.method.value,
                    "method_confidence": method_selection.confidence,
                    "reasoning": method_selection.reasoning,
                    "result_summary": {
                        "principles_count": len(result.principles),
                        "confidence": result.confidence,
                        "success": result.confidence > self.min_confidence_threshold,
                    },
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time(),
                }
            )

            # Update selector performance
            success = result.confidence > self.min_confidence_threshold
            self.method_selector.update_performance(method_selection.method, success)

            return result

    def crystallize_experience(
        self, execution_trace: ExecutionTrace
    ) -> CrystallizationResult:
        """
        Standard crystallization from execution trace

        Args:
            execution_trace: Execution trace to crystallize

        Returns:
            Crystallization result
        """
        start_time = time.time()

        # Extract principles
        principles = self.extractor.extract_from_trace(execution_trace)

        if not principles:
            logger.warning(
                "No principles extracted from trace %s", execution_trace.trace_id
            )
            return CrystallizationResult(
                principles=[],
                validation_results=[],
                contraindications=[],
                confidence=0.0,
                mode=CrystallizationMode.STANDARD,
            )

        # Validate principles
        validation_results = []
        valid_principles = []
        
        # ISSUE #6 FIX: Track rejection reasons for debugging
        rejection_reasons = {
            "invalid": 0,
            "low_confidence": 0,
            "passed": 0,
        }

        for principle in principles:
            validation = self.validator.validate(principle)
            validation_results.append(validation)

            # ISSUE #6 FIX: Log detailed rejection reasons
            if not validation.is_valid:
                rejection_reasons["invalid"] += 1
                # INDUSTRY STANDARD: Use getattr with default for robust attribute access
                principle_id = getattr(principle, 'id', 'unknown')
                logger.info(
                    f"[KnowledgeCrystallizer] ISSUE #6 FIX: Principle rejected - "
                    f"INVALID validation (is_valid=False). "
                    f"Principle: {principle_id}, "
                    f"Validation confidence: {validation.confidence:.2f}"
                )
            elif validation.confidence < self.min_confidence_threshold:
                rejection_reasons["low_confidence"] += 1
                # INDUSTRY STANDARD: Use getattr with default for robust attribute access
                principle_id = getattr(principle, 'id', 'unknown')
                logger.info(
                    f"[KnowledgeCrystallizer] ISSUE #6 FIX: Principle rejected - "
                    f"LOW CONFIDENCE ({validation.confidence:.2f} < {self.min_confidence_threshold:.2f}). "
                    f"Principle: {principle_id}"
                )
            else:
                rejection_reasons["passed"] += 1
                valid_principles.append(principle)

        # Check for contraindications
        contraindications = []
        for principle in valid_principles:
            # Analyze potential contraindications
            contras = self._analyze_contraindications(principle, execution_trace)
            contraindications.extend(contras)

            # Register contraindications
            for contra in contras:
                self.contraindication_db.register(principle.id, contra)

        # Store valid principles
        for principle in valid_principles:
            self._store_principle(principle)

            # Add to contraindication graph
            self.contraindication_graph.add_node(principle)

        # Update tracking
        self.crystallization_history.append(
            {
                "trace_id": execution_trace.trace_id,
                "principles_extracted": len(principles),
                "principles_valid": len(valid_principles),
                "contraindications": len(contraindications),
                "timestamp": time.time(),
            }
        )

        # Calculate overall confidence
        if validation_results:
            avg_confidence = np.mean([v.confidence for v in validation_results])
        else:
            avg_confidence = 0.0

        result = CrystallizationResult(
            principles=valid_principles,
            validation_results=validation_results,
            contraindications=contraindications,
            confidence=avg_confidence,
            mode=CrystallizationMode.STANDARD,
            metadata={
                "trace_id": execution_trace.trace_id,
                "execution_time": time.time() - start_time,
            },
        )

        logger.info(
            "Crystallized %d principles from trace %s (confidence: %.2f). "
            "ISSUE #6 FIX - Rejection summary: %d extracted, %d passed validation, "
            "%d rejected (invalid=%d, low_confidence=%d)",
            len(valid_principles),
            execution_trace.trace_id,
            avg_confidence,
            len(principles),
            rejection_reasons["passed"],
            rejection_reasons["invalid"] + rejection_reasons["low_confidence"],
            rejection_reasons["invalid"],
            rejection_reasons["low_confidence"],
        )

        return result

    def crystallize_with_cascade_detection(
        self, execution_trace: ExecutionTrace
    ) -> CrystallizationResult:
        """
        Crystallize with cascade failure detection

        Args:
            execution_trace: Execution trace to crystallize

        Returns:
            Crystallization result with cascade analysis
        """
        # First do standard crystallization
        result = self.crystallize_experience(execution_trace)

        # Analyze cascade impacts
        cascade_warnings = []
        filtered_principles = []

        for principle in result.principles:
            # Analyze cascade impact
            self.cascade_analyzer.analyze_cascade_impact(principle)

            # Check cascade risk
            cascade_risk = self.contraindication_graph.calculate_cascade_risk(
                principle.id
            )

            if cascade_risk < 0.7:  # Acceptable risk
                filtered_principles.append(principle)
            else:
                cascade_warnings.append(
                    f"Principle {principle.id} has high cascade risk ({cascade_risk:.2f})"
                )

                # Add as contraindication
                contra = Contraindication(
                    condition="high_cascade_risk",
                    failure_mode="cascading",
                    severity=cascade_risk,
                    workaround="Apply with cascade prevention",
                )
                result.contraindications.append(contra)

        # Update result
        result.principles = filtered_principles
        result.mode = CrystallizationMode.CASCADE_AWARE
        result.metadata["cascade_warnings"] = cascade_warnings
        result.metadata["cascade_filtered"] = len(result.principles) - len(
            filtered_principles
        )

        logger.info(
            "Cascade-aware crystallization: %d principles passed cascade check",
            len(filtered_principles),
        )

        return result

    def _crystallize_incremental(
        self, execution_trace: ExecutionTrace, parameters: Dict[str, Any]
    ) -> CrystallizationResult:
        """
        Incremental crystallization for iterative learning

        Args:
            execution_trace: Execution trace
            parameters: Method parameters

        Returns:
            Crystallization result
        """
        trace_signature = execution_trace.get_signature()

        # Check if we have previous state for this pattern
        if trace_signature in self.incremental_state:
            prev_state = self.incremental_state[trace_signature]
            iteration = prev_state["iteration"] + 1
            accumulated_traces = prev_state["traces"] + [execution_trace]
        else:
            iteration = 1
            accumulated_traces = [execution_trace]
            prev_state = {"principles": [], "confidence": 0.0}

        # Extract principles from current trace
        current_principles = self.extractor.extract_from_trace(execution_trace)

        # Merge with previous principles
        merge_strategy = parameters.get("merge_strategy", "weighted")
        if merge_strategy == "weighted":
            # Weight recent iterations more heavily
            weight_decay = parameters.get("iteration_weight_decay", 0.9)
            merged_principles = self._merge_principles_weighted(
                prev_state["principles"], current_principles, weight_decay**iteration
            )
        else:
            merged_principles = self._merge_principles_simple(
                prev_state["principles"], current_principles
            )

        # Validate merged principles
        validation_results = []
        valid_principles = []

        for principle in merged_principles:
            validation = self.validator.validate(principle)
            validation_results.append(validation)

            if validation.is_valid:
                # Boost confidence for consistent principles
                if hasattr(principle, "confidence"):
                    principle.confidence = min(
                        0.99, principle.confidence * (1 + 0.05 * iteration)
                    )
                valid_principles.append(principle)

        # Update state
        self.incremental_state[trace_signature] = {
            "iteration": iteration,
            "traces": accumulated_traces[-parameters.get("max_iterations", 100) :],
            "principles": valid_principles,
            "confidence": (
                np.mean([p.confidence for p in valid_principles])
                if valid_principles
                else 0.0
            ),
        }

        # Calculate overall confidence
        avg_confidence = (
            np.mean([v.confidence for v in validation_results])
            if validation_results
            else 0.0
        )

        result = CrystallizationResult(
            principles=valid_principles,
            validation_results=validation_results,
            contraindications=[],
            confidence=avg_confidence,
            mode=CrystallizationMode.INCREMENTAL,
            metadata={
                "iteration": iteration,
                "accumulated_traces": len(accumulated_traces),
                "merge_strategy": merge_strategy,
            },
        )

        logger.info(
            "Incremental crystallization iteration %d: %d principles (confidence: %.2f)",
            iteration,
            len(valid_principles),
            avg_confidence,
        )

        return result

    def _crystallize_batch(
        self, traces: List[ExecutionTrace], parameters: Dict[str, Any]
    ) -> CrystallizationResult:
        """
        Batch crystallization for multiple traces

        Args:
            traces: List of execution traces
            parameters: Method parameters

        Returns:
            Crystallization result
        """
        batch_size = parameters.get("batch_size", len(traces))
        parameters.get("parallel_processing", False)
        aggregation_method = parameters.get("aggregation_method", "voting")

        # Process each trace
        all_principles = []

        for trace in traces[:batch_size]:
            trace_principles = self.extractor.extract_from_trace(trace)
            all_principles.extend(trace_principles)

        # Aggregate principles
        if aggregation_method == "voting":
            aggregated = self._aggregate_by_voting(all_principles)
        else:
            aggregated = self._aggregate_by_averaging(all_principles)

        # Validate aggregated principles
        validation_results = []
        valid_principles = []

        for principle in aggregated:
            validation = self.validator.validate(principle)
            validation_results.append(validation)

            if (
                validation.is_valid
                and validation.confidence >= self.min_confidence_threshold
            ):
                valid_principles.append(principle)

        # Outlier detection if enabled
        if parameters.get("outlier_detection", False):
            valid_principles = self._remove_outliers(valid_principles)

        # Calculate confidence
        avg_confidence = (
            np.mean([v.confidence for v in validation_results])
            if validation_results
            else 0.0
        )

        result = CrystallizationResult(
            principles=valid_principles,
            validation_results=validation_results,
            contraindications=[],
            confidence=avg_confidence,
            mode=CrystallizationMode.BATCH,
            metadata={
                "batch_size": batch_size,
                "aggregation_method": aggregation_method,
                "total_traces_processed": len(traces),
            },
        )

        logger.info(
            "Batch crystallization: %d principles from %d traces (confidence: %.2f)",
            len(valid_principles),
            batch_size,
            avg_confidence,
        )

        return result

    def _crystallize_adaptive(
        self, execution_trace: ExecutionTrace, parameters: Dict[str, Any]
    ) -> CrystallizationResult:
        """
        Adaptive crystallization that adjusts to feedback

        Args:
            execution_trace: Execution trace
            parameters: Method parameters

        Returns:
            Crystallization result
        """
        adaptation_rate = parameters.get("adaptation_rate", 0.1)
        exploration_ratio = parameters.get("exploration_ratio", 0.2)

        # Extract principles with exploration
        principles = self.extractor.extract_from_trace(execution_trace)

        # Apply adaptive thresholds
        if parameters.get("dynamic_thresholds", True):
            # Adjust thresholds based on recent performance
            recent_successes = self._get_recent_success_rate()
            if recent_successes < 0.3:
                # Lower thresholds if struggling
                self.min_confidence_threshold *= 1 - adaptation_rate
            elif recent_successes > 0.8:
                # Raise thresholds if doing well
                self.min_confidence_threshold *= 1 + adaptation_rate

        # Validate with adaptive criteria
        validation_results = []
        valid_principles = []
        exploratory_principles = []

        for principle in principles:
            validation = self.validator.validate(principle)
            validation_results.append(validation)

            if (
                validation.is_valid
                and validation.confidence >= self.min_confidence_threshold
            ):
                valid_principles.append(principle)
            elif validation.confidence >= self.min_confidence_threshold * (
                1 - exploration_ratio
            ):
                # Keep for exploration
                exploratory_principles.append(principle)

        # Include some exploratory principles
        num_exploratory = int(len(exploratory_principles) * exploration_ratio)
        if num_exploratory > 0:
            selected_exploratory = np.random.choice(
                exploratory_principles,
                size=min(num_exploratory, len(exploratory_principles)),
                replace=False,
            ).tolist()
            valid_principles.extend(selected_exploratory)

        # Calculate confidence
        avg_confidence = (
            np.mean([v.confidence for v in validation_results])
            if validation_results
            else 0.0
        )

        result = CrystallizationResult(
            principles=valid_principles,
            validation_results=validation_results,
            contraindications=[],
            confidence=avg_confidence,
            mode=CrystallizationMode(CrystallizationMethod.ADAPTIVE.value),
            metadata={
                "adaptation_rate": adaptation_rate,
                "exploration_ratio": exploration_ratio,
                "exploratory_count": num_exploratory,
                "threshold_used": self.min_confidence_threshold,
            },
        )

        logger.info(
            "Adaptive crystallization: %d principles (including %d exploratory)",
            len(valid_principles),
            num_exploratory,
        )

        return result

    def _crystallize_hybrid(
        self, execution_trace: ExecutionTrace, parameters: Dict[str, Any]
    ) -> CrystallizationResult:
        """
        Hybrid crystallization combining multiple methods

        Args:
            execution_trace: Execution trace
            parameters: Method parameters

        Returns:
            Crystallization result
        """
        primary_method = parameters.get("primary_method", "standard")
        secondary_methods = parameters.get("secondary_methods", [])
        fusion_strategy = parameters.get("fusion_strategy", "weighted")

        results = []
        weights = []

        # Apply primary method
        if primary_method == CrystallizationMethod.CASCADE_AWARE.value:
            primary_result = self.crystallize_with_cascade_detection(execution_trace)
        else:
            primary_result = self.crystallize_experience(execution_trace)

        results.append(primary_result)
        weights.append(0.6)  # Primary gets highest weight

        # Apply secondary methods
        for method_name in secondary_methods:
            if method_name == CrystallizationMethod.ADAPTIVE.value:
                result = self._crystallize_adaptive(
                    execution_trace, {"adaptation_rate": 0.1}
                )
                results.append(result)
                weights.append(0.2)

        # Fuse results
        if fusion_strategy == "weighted":
            fused_principles = self._fuse_results_weighted(results, weights)
        else:
            fused_principles = self._fuse_results_simple(results)

        # Final validation
        validation_results = []
        valid_principles = []

        for principle in fused_principles:
            validation = self.validator.validate_stratified(principle)
            validation_results.append(validation)

            if validation.is_valid:
                valid_principles.append(principle)

        # Calculate confidence
        avg_confidence = (
            np.mean([v.confidence for v in validation_results])
            if validation_results
            else 0.0
        )

        result = CrystallizationResult(
            principles=valid_principles,
            validation_results=validation_results,
            contraindications=[],
            confidence=avg_confidence,
            mode=CrystallizationMode(CrystallizationMethod.HYBRID.value),
            metadata={
                "primary_method": primary_method,
                "secondary_methods": secondary_methods,
                "fusion_strategy": fusion_strategy,
                "methods_count": len(results),
            },
        )

        logger.info(
            "Hybrid crystallization: %d principles from %d methods",
            len(valid_principles),
            len(results),
        )

        return result

    def validate_stratified(self, candidate: Principle) -> ValidationResult:
        """
        Perform stratified validation of candidate principle

        Args:
            candidate: Candidate principle to validate

        Returns:
            Validation result
        """
        # Perform multi-level validation
        validations = []

        # Level 1: Basic validation
        basic_validation = self.validator.validate(candidate)
        validations.append(("basic", basic_validation))

        # Level 2: Domain-specific validation
        if hasattr(candidate, "domain"):
            domain_compatible, blocking = (
                self.contraindication_db.check_domain_compatibility(
                    candidate.id, getattr(candidate, "domain", "general")
                )
            )

            domain_validation = ValidationResult(
                is_valid=domain_compatible,
                confidence=0.8 if domain_compatible else 0.2,
                errors=[] if domain_compatible else ["Domain contraindications found"],
                warnings=[str(b) for b in blocking],
            )
            validations.append(("domain", domain_validation))

        # Level 3: Cascade validation
        cascade_risk = self.contraindication_graph.calculate_cascade_risk(candidate.id)
        cascade_validation = ValidationResult(
            is_valid=cascade_risk < 0.7,
            confidence=1.0 - cascade_risk,
            errors=[] if cascade_risk < 0.7 else ["High cascade risk"],
            warnings=[] if cascade_risk < 0.5 else ["Moderate cascade risk"],
        )
        validations.append(("cascade", cascade_validation))

        # Level 4: Historical validation
        historical_validation = self._validate_against_history(candidate)
        validations.append(("historical", historical_validation))

        # Combine validations
        all_valid = all(v.is_valid for _, v in validations)
        avg_confidence = np.mean([v.confidence for _, v in validations])

        all_errors = []
        all_warnings = []

        for level, validation in validations:
            all_errors.extend([f"{level}: {e}" for e in validation.errors])
            all_warnings.extend([f"{level}: {w}" for w in validation.warnings])

        combined_validation = ValidationResult(
            is_valid=all_valid,
            confidence=avg_confidence,
            errors=all_errors,
            warnings=all_warnings,
            metadata={
                "validation_levels": [level for level, _ in validations],
                "level_results": {level: v.is_valid for level, v in validations},
            },
        )

        logger.debug(
            "Stratified validation for %s: valid=%s, confidence=%.2f",
            candidate.id,
            all_valid,
            avg_confidence,
        )

        return combined_validation

    def apply_knowledge(
        self, problem: Dict[str, Any], confidence_required: float = 0.7
    ) -> ApplicationResult:
        """
        Apply crystallized knowledge to solve problem

        Args:
            problem: Problem specification
            confidence_required: Minimum confidence threshold

        Returns:
            Application result
        """
        start_time = time.time()

        # Find applicable principles
        applicable = self.applicator.find_applicable_principles(problem)

        if not applicable:
            logger.info("No applicable principles found for problem")
            return ApplicationResult(
                principle_used=None,
                solution=None,
                confidence=0.0,
                warnings=["No applicable principles found"],
            )

        # Filter by confidence
        confident_principles = [
            p
            for p in applicable
            if getattr(p, "confidence", 0.5) >= confidence_required
        ]

        if not confident_principles:
            logger.info(
                "No principles meet confidence requirement %.2f", confidence_required
            )
            max_conf = max(getattr(p, "confidence", 0) for p in applicable)
            return ApplicationResult(
                principle_used=None,
                solution=None,
                confidence=max_conf,
                warnings=[
                    f"No principles meet confidence requirement {confidence_required}"
                ],
            )

        # Select best principle
        best_principle = max(
            confident_principles, key=lambda p: getattr(p, "confidence", 0)
        )

        # Check for contraindications
        if problem.get("context"):
            compatible, blocking = self.contraindication_db.check_domain_compatibility(
                best_principle.id, problem["context"].get("domain", "general")
            )

            if not compatible and blocking:
                # Try to find alternative
                for principle in confident_principles[1:]:
                    alt_compatible, _ = (
                        self.contraindication_db.check_domain_compatibility(
                            principle.id, problem["context"].get("domain", "general")
                        )
                    )
                    if alt_compatible:
                        best_principle = principle
                        break

        # Adapt principle to context
        if problem.get("context"):
            adapted = self.applicator.adapt_principle_to_context(
                best_principle, problem["context"]
            )
        else:
            adapted = best_principle

        # Apply principle
        solution = self._apply_principle(adapted, problem)

        # Monitor application
        if solution is not None:
            self.applicator.monitor_application(adapted, {"solution": solution})

        # Track application
        self.application_history.append(
            {
                "problem": problem,
                "principle_id": best_principle.id,
                "confidence": getattr(best_principle, "confidence", 0.5),
                "success": solution is not None,
                "timestamp": time.time(),
            }
        )

        result = ApplicationResult(
            principle_used=best_principle,
            solution=solution,
            confidence=getattr(best_principle, "confidence", 0.5),
            execution_time=time.time() - start_time,
        )

        logger.info(
            "Applied principle %s to problem (confidence: %.2f)",
            best_principle.id,
            result.confidence,
        )

        return result

    def update_from_feedback(self, principle_id: str, outcome: Dict[str, Any]):
        """
        Update knowledge from application feedback

        Args:
            principle_id: ID of applied principle
            outcome: Outcome of application
        """
        with self.lock:
            # Get principle
            principle = self._get_principle(principle_id)

            if not principle:
                logger.warning(
                    "Principle %s not found for feedback update", principle_id
                )
                return

            # Update confidence based on outcome
            success = outcome.get("success", False)

            if success:
                # Increase confidence
                if hasattr(principle, "confidence"):
                    principle.confidence = min(0.99, principle.confidence * 1.1)
                if hasattr(principle, "success_count"):
                    principle.success_count += 1
            else:
                # Decrease confidence
                if hasattr(principle, "confidence"):
                    principle.confidence *= 0.9
                if hasattr(principle, "failure_count"):
                    principle.failure_count += 1

                # Check if this is a new contraindication
                if "failure_mode" in outcome:
                    contra = Contraindication(
                        condition=outcome.get("condition", "unknown"),
                        failure_mode=outcome["failure_mode"],
                        frequency=1,
                        severity=outcome.get("severity", 0.5),
                        domain=outcome.get("domain"),
                    )
                    self.contraindication_db.register(principle_id, contra)

            # Update principle in knowledge base
            self._update_principle(principle)

            # Check for imbalances
            imbalances = self.imbalance_handler.detect_imbalance(self.knowledge_base)
            if any(score > 0.5 for score in imbalances.values()):
                suggestions = self.imbalance_handler.suggest_focus_areas(imbalances)
                logger.info(
                    "Knowledge imbalance detected. Suggestions: %s", suggestions
                )

            logger.debug(
                "Updated principle %s from feedback (new confidence: %.2f)",
                principle_id,
                getattr(principle, "confidence", 0.5),
            )

    # Storage adapter methods
    def _store_principle(self, principle):
        """Adapter for principle storage"""
        return self.knowledge_base.store(principle)

    def _get_principle(self, principle_id):
        """Adapter for principle retrieval"""
        return self.knowledge_base.get(principle_id)

    def _update_principle(self, principle):
        """Adapter for principle update"""
        return self.knowledge_base.store_versioned(principle)

    # VULCAN integration method
    def store_knowledge(self, key: str, value: Any) -> bool:
        """
        Store knowledge (for VULCAN compatibility)

        Args:
            key: Knowledge key
            value: Knowledge value

        Returns:
            Success status
        """
        try:
            # Convert to principle if needed
            if not isinstance(value, Principle):
                principle = Principle(
                    id=key,
                    name=f"Knowledge_{key}",
                    description=str(value),
                    core_pattern=None,  # Note: Added required core_pattern parameter
                    confidence=0.5,
                    domain="general",
                )
            else:
                principle = value

            self._store_principle(principle)
            return True
        except Exception as e:
            logger.error("Failed to store knowledge %s: %s", key, e)
            return False

    # Helper methods for crystallization
    def _merge_principles_weighted(
        self,
        old_principles: List[Principle],
        new_principles: List[Principle],
        old_weight: float,
    ) -> List[Principle]:
        """Merge principles with weighting"""
        merged = {}

        # Add old principles with weight
        for principle in old_principles:
            if hasattr(principle, "confidence"):
                principle.confidence *= old_weight
            merged[principle.id] = principle

        # Add or update with new principles
        for principle in new_principles:
            if principle.id in merged:
                # Average confidences
                old_conf = getattr(merged[principle.id], "confidence", 0.5)
                new_conf = getattr(principle, "confidence", 0.5)
                principle.confidence = (old_conf + new_conf) / 2
            merged[principle.id] = principle

        return list(merged.values())

    def _merge_principles_simple(
        self, old_principles: List[Principle], new_principles: List[Principle]
    ) -> List[Principle]:
        """Simple merge of principles"""
        merged = {p.id: p for p in old_principles}
        merged.update({p.id: p for p in new_principles})
        return list(merged.values())

    def _aggregate_by_voting(self, principles: List[Principle]) -> List[Principle]:
        """Aggregate principles by voting"""
        principle_votes = defaultdict(int)
        principle_map = {}

        for principle in principles:
            principle_votes[principle.id] += 1
            principle_map[principle.id] = principle

        # Keep principles with multiple votes
        aggregated = []
        for pid, votes in principle_votes.items():
            if votes >= 2 or len(principles) < 3:
                principle = principle_map[pid]
                # Boost confidence based on votes
                if hasattr(principle, "confidence"):
                    principle.confidence *= 1 + 0.1 * votes
                aggregated.append(principle)

        return aggregated

    def _aggregate_by_averaging(self, principles: List[Principle]) -> List[Principle]:
        """Aggregate principles by averaging"""
        principle_groups = defaultdict(list)

        for principle in principles:
            principle_groups[principle.id].append(principle)

        aggregated = []
        for pid, group in principle_groups.items():
            if len(group) == 1:
                aggregated.append(group[0])
            else:
                # Average confidences
                avg_confidence = np.mean([getattr(p, "confidence", 0.5) for p in group])
                representative = group[0]
                representative.confidence = avg_confidence
                aggregated.append(representative)

        return aggregated

    def _remove_outliers(self, principles: List[Principle]) -> List[Principle]:
        """Remove outlier principles"""
        if len(principles) < 3:
            return principles

        confidences = [getattr(p, "confidence", 0.5) for p in principles]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        # Remove principles > 2 std deviations from mean
        filtered = []
        for principle in principles:
            conf = getattr(principle, "confidence", 0.5)
            if abs(conf - mean_conf) <= 2 * std_conf:
                filtered.append(principle)

        return filtered

    def _get_recent_success_rate(self) -> float:
        """Get recent crystallization success rate"""
        if not self.crystallization_history:
            return 0.5

        recent = list(self.crystallization_history)[-20:]
        successes = sum(
            1 for h in recent if h.get("result_summary", {}).get("success", False)
        )

        return successes / len(recent)

    def _fuse_results_weighted(
        self, results: List[CrystallizationResult], weights: List[float]
    ) -> List[Principle]:
        """Fuse results with weights"""
        all_principles = {}

        for result, weight in zip(results, weights):
            for principle in result.principles:
                if principle.id not in all_principles:
                    all_principles[principle.id] = []
                # Weight the principle
                weighted_principle = copy.deepcopy(principle)
                if hasattr(weighted_principle, "confidence"):
                    weighted_principle.confidence *= weight
                all_principles[principle.id].append(weighted_principle)

        # Aggregate weighted principles
        fused = []
        for pid, weighted_list in all_principles.items():
            if len(weighted_list) == 1:
                fused.append(weighted_list[0])
            else:
                # Average weighted confidences
                avg_confidence = np.mean(
                    [getattr(p, "confidence", 0.5) for p in weighted_list]
                )
                representative = weighted_list[0]
                representative.confidence = avg_confidence
                fused.append(representative)

        return fused

    def _fuse_results_simple(
        self, results: List[CrystallizationResult]
    ) -> List[Principle]:
        """Simple fusion of results"""
        all_principles = {}

        for result in results:
            for principle in result.principles:
                # Keep highest confidence version
                if principle.id not in all_principles:
                    all_principles[principle.id] = principle
                else:
                    existing_conf = getattr(
                        all_principles[principle.id], "confidence", 0.5
                    )
                    new_conf = getattr(principle, "confidence", 0.5)
                    if new_conf > existing_conf:
                        all_principles[principle.id] = principle

        return list(all_principles.values())

    def _analyze_contraindications(
        self, principle: Principle, trace: ExecutionTrace
    ) -> List[Contraindication]:
        """Analyze potential contraindications"""
        contraindications = []

        # Check for failure patterns in trace
        if not trace.success:
            # Extract failure information
            failure_mode = trace.metadata.get("failure_mode", "unknown")

            contra = Contraindication(
                condition=trace.context.get("condition", "failure_context"),
                failure_mode=failure_mode,
                frequency=1,
                severity=0.6,
                domain=trace.context.get("domain") or getattr(trace, "domain", None),
            )
            contraindications.append(contra)

        # Check for resource constraints
        if "resources" in trace.metadata:
            resources = trace.metadata["resources"]
            if isinstance(resources, dict) and resources.get("memory_usage", 0) > 0.8:
                contra = Contraindication(
                    condition="high_memory",
                    failure_mode="resource",
                    severity=0.5,
                    workaround="Increase memory allocation",
                )
                contraindications.append(contra)

        return contraindications

    def _validate_against_history(self, candidate: Principle) -> ValidationResult:
        """Validate against historical performance"""
        # Check if we have history for similar principles
        similar_principles = self.knowledge_base.find_similar(candidate, threshold=0.7)

        if not similar_principles:
            # No history - neutral validation
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                warnings=["No historical data for validation"],
            )

        # Calculate historical success rate
        total_success = sum(getattr(p, "success_count", 0) for p in similar_principles)
        total_failure = sum(getattr(p, "failure_count", 0) for p in similar_principles)
        total_attempts = total_success + total_failure

        if total_attempts == 0:
            success_rate = 0.5
        else:
            success_rate = total_success / total_attempts

        return ValidationResult(
            is_valid=success_rate > 0.3,
            confidence=success_rate,
            errors=[] if success_rate > 0.3 else ["Low historical success rate"],
            warnings=(
                [] if success_rate > 0.5 else ["Below average historical performance"]
            ),
        )

    def _apply_principle(self, principle: Principle, problem: Dict[str, Any]) -> Any:
        """Apply principle to problem"""
        # This is a simplified implementation
        # In practice, this would involve complex problem-solving logic

        try:
            # Extract problem parameters
            problem_type = problem.get("type", "general")

            # Apply principle logic
            if hasattr(principle, "apply"):
                solution = principle.apply(problem)
            else:
                # Generic application
                solution = {
                    "principle_id": principle.id,
                    "approach": getattr(principle, "description", "Unknown approach"),
                    "confidence": getattr(principle, "confidence", 0.5),
                    "problem_type": problem_type,
                }

            return solution

        except Exception as e:
            logger.error("Failed to apply principle %s: %s", principle.id, e)
            return None


class KnowledgeApplicator:
    """Applies crystallized knowledge to new problems"""

    def __init__(self, crystallizer: KnowledgeCrystallizer):
        """
        Initialize knowledge applicator

        Args:
            crystallizer: Parent crystallizer instance
        """
        self.crystallizer = crystallizer
        self.adaptation_history = deque(maxlen=1000)
        self.combination_cache = {}

        logger.info("KnowledgeApplicator initialized")

    def find_applicable_principles(self, problem: Dict[str, Any]) -> List[Principle]:
        """
        Find principles applicable to problem

        Args:
            problem: Problem specification

        Returns:
            List of applicable principles
        """
        applicable = []

        # Get all principles from knowledge base
        all_principles = self.crystallizer.knowledge_base.get_all_principles()

        for principle in all_principles:
            # Check domain match
            if self._matches_domain(principle, problem):
                # Check problem type match
                if self._matches_problem_type(principle, problem):
                    # Check constraints
                    if self._satisfies_constraints(principle, problem):
                        applicable.append(principle)

        # Sort by confidence and relevance
        applicable.sort(
            key=lambda p: (
                getattr(p, "confidence", 0.5) * self._calculate_relevance(p, problem)
            ),
            reverse=True,
        )

        logger.debug("Found %d applicable principles for problem", len(applicable))

        return applicable

    def adapt_principle_to_context(
        self, principle: Principle, context: Dict[str, Any]
    ) -> Principle:
        """
        Adapt principle to specific context

        Args:
            principle: Principle to adapt
            context: Target context

        Returns:
            Adapted principle
        """
        # Create copy for adaptation
        adapted = copy.deepcopy(principle)
        adapted.id = f"{principle.id}_adapted_{int(time.time())}"

        # Adapt based on context
        adaptations = []

        # Scale parameters
        if "scale" in context:
            scale = context["scale"]
            if hasattr(adapted, "parameters"):
                for param in adapted.parameters:
                    if param in ["threshold", "limit", "size"]:
                        adapted.parameters[param] *= scale
                adaptations.append(f"Scaled {param} by {scale}")

        # Adjust for domain
        if (
            "domain" in context
            and hasattr(adapted, "domain")
            and context["domain"] != adapted.domain
        ):
            adapted.domain = context["domain"]
            if hasattr(adapted, "confidence"):
                adapted.confidence *= 0.9  # Reduce confidence for cross-domain
            adaptations.append(
                f"Adapted from {getattr(principle, 'domain', 'unknown')} to {context['domain']}"
            )

        # Apply constraints
        if "constraints" in context:
            for constraint, value in context["constraints"].items():
                if hasattr(adapted, constraint):
                    setattr(adapted, constraint, value)
                    adaptations.append(f"Applied constraint {constraint}={value}")

        # Track adaptation
        self.adaptation_history.append(
            {
                "original": principle.id,
                "adapted": adapted.id,
                "adaptations": adaptations,
                "context": context,
                "timestamp": time.time(),
            }
        )

        logger.debug(
            "Adapted principle %s with %d adaptations", principle.id, len(adaptations)
        )

        return adapted

    def combine_principles(self, principles: List[Principle]) -> Optional[Principle]:
        """
        Combine multiple principles

        Args:
            principles: Principles to combine

        Returns:
            Combined principle or None
        """
        if not principles:
            return None

        if len(principles) == 1:
            return principles[0]

        # Check cache
        cache_key = tuple(sorted(p.id for p in principles))
        if cache_key in self.combination_cache:
            return self.combination_cache[cache_key]

        # Create combined principle
        # Extract base pattern from first principle
        base_pattern = getattr(principles[0], "core_pattern", None)

        combined = Principle(
            id=f"combined_{'_'.join(p.id[:8] for p in principles)}",
            name=f"Combined: {', '.join(getattr(p, 'name', p.id) for p in principles[:3])}",
            description="Combined principle",
            core_pattern=base_pattern,  # Use base pattern
            confidence=np.mean([getattr(p, "confidence", 0.5) for p in principles])
            * 0.9,
            domain="combined",
        )

        # Combine logic
        combined.sub_principles = principles
        combined.combination_type = "ensemble"

        # Cache result
        self.combination_cache[cache_key] = combined

        logger.debug("Combined %d principles into %s", len(principles), combined.id)

        return combined

    def monitor_application(self, principle: Principle, execution: Dict[str, Any]):
        """
        Monitor principle application

        Args:
            principle: Applied principle
            execution: Execution details
        """
        # Check for issues
        issues = []

        if "error" in execution:
            issues.append(f"Error: {execution['error']}")

        if "performance" in execution:
            if execution["performance"].get("time", 0) > 10:
                issues.append("Slow execution")
            if execution["performance"].get("memory", 0) > 1000:
                issues.append("High memory usage")

        # Update principle tracking
        if issues:
            # Report issues as potential contraindications
            for issue in issues:
                logger.warning(
                    "Application issue for principle %s: %s", principle.id, issue
                )
        else:
            logger.debug("Successful application of principle %s", principle.id)

    def _matches_domain(self, principle: Principle, problem: Dict[str, Any]) -> bool:
        """Check if principle matches problem domain"""
        if not hasattr(principle, "domain"):
            return True  # Domain-agnostic principle

        problem_domain = problem.get("domain", "general")
        principle_domain = getattr(principle, "domain", "general")

        # Exact match
        if principle_domain == problem_domain:
            return True

        # General principles apply everywhere
        if principle_domain == "general":
            return True

        # Check domain hierarchy
        if "_" in problem_domain:
            # Check parent domain
            parent_domain = problem_domain.split("_")[0]
            if principle_domain == parent_domain:
                return True

        return False

    def _matches_problem_type(
        self, principle: Principle, problem: Dict[str, Any]
    ) -> bool:
        """Check if principle matches problem type"""
        if not hasattr(principle, "problem_types"):
            return True  # No type restriction

        problem_type = problem.get("type", "general")

        return (
            problem_type in principle.problem_types
            or "general" in principle.problem_types
        )

    def _satisfies_constraints(
        self, principle: Principle, problem: Dict[str, Any]
    ) -> bool:
        """Check if principle satisfies problem constraints"""
        if "constraints" not in problem:
            return True

        constraints = problem["constraints"]

        # Check resource constraints
        if "max_memory" in constraints:
            if hasattr(principle, "memory_usage"):
                if principle.memory_usage > constraints["max_memory"]:
                    return False

        if "max_time" in constraints:
            if hasattr(principle, "execution_time"):
                if principle.execution_time > constraints["max_time"]:
                    return False

        return True

    def _calculate_relevance(
        self, principle: Principle, problem: Dict[str, Any]
    ) -> float:
        """Calculate relevance score"""
        relevance = 1.0

        # Domain match bonus
        if hasattr(principle, "domain"):
            if getattr(principle, "domain", None) == problem.get("domain"):
                relevance *= 1.2

        # Problem type match bonus
        if hasattr(principle, "problem_types"):
            if problem.get("type") in principle.problem_types:
                relevance *= 1.1

        # Recency bonus
        if hasattr(principle, "last_updated"):
            age_days = (time.time() - principle.last_updated) / 86400
            recency_factor = np.exp(-age_days / 30)  # Decay over 30 days
            relevance *= 0.5 + 0.5 * recency_factor

        return min(2.0, relevance)
