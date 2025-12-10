"""
concept_mapper.py - Maps patterns to concepts with grounding validation
Part of the VULCAN-AGI system

FIXED: Added safety_config and world_model integration
ENHANCED: Domain-adaptive thresholds, concept decay, improved effect categorization
PRODUCTION-READY: All unbounded data structures fixed with proper limits and eviction
"""

import hashlib
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import safety validator with multiple fallback paths
SAFETY_VALIDATOR_AVAILABLE = False
EnhancedSafetyValidator = None
SafetyConfig = None

# Try relative import first (when used as part of vulcan package)
try:
    from ..safety.safety_types import SafetyConfig
    from ..safety.safety_validator import EnhancedSafetyValidator

    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    pass

# Fallback: Try absolute import (when vulcan is in sys.path)
if not SAFETY_VALIDATOR_AVAILABLE:
    try:
        from vulcan.safety.safety_types import SafetyConfig
        from vulcan.safety.safety_validator import EnhancedSafetyValidator

        SAFETY_VALIDATOR_AVAILABLE = True
    except ImportError:
        pass

# Fallback: Try src-prefixed import (when src is in sys.path)
if not SAFETY_VALIDATOR_AVAILABLE:
    try:
        from src.vulcan.safety.safety_types import SafetyConfig
        from src.vulcan.safety.safety_validator import EnhancedSafetyValidator

        SAFETY_VALIDATOR_AVAILABLE = True
    except ImportError:
        # Note: Warning moved to __init__ to avoid spurious warnings at import time
        pass

logger = logging.getLogger(__name__)


class EffectType(Enum):
    """Types of measurable effects"""

    PERFORMANCE = "performance"
    RESOURCE = "resource"
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"


class GroundingStatus(Enum):
    """Status of concept grounding"""

    UNGROUNDED = "ungrounded"
    WEAKLY_GROUNDED = "weakly_grounded"
    GROUNDED = "grounded"
    STRONGLY_GROUNDED = "strongly_grounded"


@dataclass
class MeasurableEffect:
    """Represents a measurable effect"""

    effect_id: str
    effect_type: EffectType
    measurement: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5

    def is_consistent_with(
        self, other: "MeasurableEffect", tolerance: float = 0.2
    ) -> bool:
        """Check if effect is consistent with another"""
        if self.effect_type != other.effect_type:
            return False

        if self.unit != other.unit:
            return False

        # Check measurement consistency within tolerance
        if self.measurement == 0 and other.measurement == 0:
            return True

        max_val = max(abs(self.measurement), abs(other.measurement))
        if max_val > 0:
            relative_diff = abs(self.measurement - other.measurement) / max_val
            return relative_diff <= tolerance

        return False


@dataclass
class PatternOutcome:
    """Outcome from pattern application"""

    outcome_id: str
    pattern_signature: str
    success: bool
    measurements: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields for context
    execution_time: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class Concept:
    """Single concept representation with grounding"""

    def __init__(
        self,
        pattern_signature: str,
        grounded_effects: List[MeasurableEffect],
        confidence: float = 0.5,
    ):
        """
        Initialize concept

        Args:
            pattern_signature: Unique signature of the pattern
            grounded_effects: List of measurable grounded effects
            confidence: Initial confidence in the concept
        """
        self.concept_id = (
            f"concept_{hashlib.md5(pattern_signature.encode(), usedforsecurity=False).hexdigest()[:8]}"
        )
        self.pattern_signature = pattern_signature
        self.grounded_effects = grounded_effects
        self.confidence = confidence

        # Evidence tracking
        self.evidence_count = 0
        self.positive_evidence = 0
        self.negative_evidence = 0
        self.evidence_history = deque(maxlen=100)

        # Stability tracking
        self.stability_scores = deque(maxlen=20)
        self.last_update = time.time()
        self.creation_time = time.time()

        # Performance metrics
        self.success_rate = 0.5
        self.consistency_score = 1.0

        # Domain tracking - FIXED: Added for semantic bridge compatibility
        self.domains = set()
        self.usage_count = 0

        # Features - FIXED: Added for semantic bridge compatibility
        self.features = {}

        # Metadata
        self.metadata = {}

        # Grounding status
        self.grounding_status = self._calculate_grounding_status()

        logger.debug(
            "Created concept %s with %d grounded effects",
            self.concept_id,
            len(grounded_effects),
        )

    def update_evidence(self, new_outcomes: List[PatternOutcome]):
        """
        Update concept with new evidence

        Args:
            new_outcomes: New outcomes to incorporate
        """
        if not new_outcomes:
            return

        for outcome in new_outcomes:
            self.evidence_count += 1
            self.usage_count += 1

            # Track domains
            if hasattr(outcome, "domain") and outcome.domain:
                self.domains.add(outcome.domain)

            # Check if outcome supports concept
            support_score = self._evaluate_outcome_support(outcome)

            if support_score > 0.5:
                self.positive_evidence += 1
            else:
                self.negative_evidence += 1

            # Update evidence history
            self.evidence_history.append(
                {
                    "outcome_id": outcome.outcome_id,
                    "support_score": support_score,
                    "timestamp": outcome.timestamp,
                    "success": outcome.success,
                }
            )

        # Update success rate
        if self.evidence_count > 0:
            self.success_rate = self.positive_evidence / self.evidence_count

        # Update confidence based on evidence
        self._update_confidence()

        # Recalculate stability
        current_stability = self.calculate_stability_score()
        self.stability_scores.append(current_stability)

        # Update grounding status
        self.grounding_status = self._calculate_grounding_status()

        self.last_update = time.time()

        logger.debug(
            "Updated concept %s with %d new outcomes (success_rate: %.2f)",
            self.concept_id,
            len(new_outcomes),
            self.success_rate,
        )

    def update_usage(self, success: bool):
        """
        Update usage statistics - FIXED: Added for semantic bridge compatibility

        Args:
            success: Whether the usage was successful
        """
        self.usage_count += 1
        alpha = 0.1
        self.success_rate = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        )

    def get_signature(self) -> str:
        """
        Get concept signature - FIXED: Added for semantic bridge compatibility

        Returns:
            Pattern signature
        """
        return self.pattern_signature

    def calculate_stability_score(self) -> float:
        """
        Calculate concept stability score

        Returns:
            Stability score [0, 1] where 1 is most stable
        """
        if self.evidence_count < 3:
            # Not enough evidence for stability
            return 0.0

        # Factor 1: Consistency of outcomes
        if self.evidence_history:
            recent_scores = [
                e["support_score"] for e in list(self.evidence_history)[-10:]
            ]
            if recent_scores:
                score_variance = np.var(recent_scores)
                consistency_factor = 1.0 / (1.0 + score_variance)
            else:
                consistency_factor = 0.5
        else:
            consistency_factor = 0.0

        # Factor 2: Success rate stability
        if len(self.stability_scores) >= 5:
            recent_stability = list(self.stability_scores)[-5:]
            stability_variance = np.var(recent_stability)
            stability_factor = 1.0 / (1.0 + stability_variance)
        else:
            stability_factor = 0.5

        # Factor 3: Age and usage
        age = time.time() - self.creation_time
        age_factor = min(1.0, age / (30 * 24 * 3600))  # Max at 30 days
        usage_factor = min(1.0, self.evidence_count / 50)  # Max at 50 uses

        # Weighted combination
        stability = (
            consistency_factor * 0.4
            + stability_factor * 0.3
            + age_factor * 0.15
            + usage_factor * 0.15
        )

        return min(1.0, stability)

    def get_grounding_confidence(self) -> float:
        """
        Get confidence in concept grounding

        Returns:
            Grounding confidence [0, 1]
        """
        # Base confidence from evidence
        if self.evidence_count == 0:
            return 0.0

        evidence_factor = min(1.0, self.evidence_count / 20)

        # Success rate factor
        success_factor = self.success_rate

        # Stability factor
        stability_factor = self.calculate_stability_score()

        # Effect grounding factor
        effect_confidences = [e.confidence for e in self.grounded_effects]
        effect_factor = np.mean(effect_confidences) if effect_confidences else 0.5

        # Weighted combination
        grounding_confidence = (
            evidence_factor * 0.2
            + success_factor * 0.3
            + stability_factor * 0.2
            + effect_factor * 0.3
        )

        return min(1.0, grounding_confidence)

    def _evaluate_outcome_support(self, outcome: PatternOutcome) -> float:
        """Evaluate how much an outcome supports this concept"""
        if outcome.pattern_signature != self.pattern_signature:
            return 0.0

        support_score = 0.5  # Base score

        # Success contributes to support
        if outcome.success:
            support_score += 0.2

        # Check if measurements align with grounded effects
        alignment_scores = []
        for effect in self.grounded_effects:
            if effect.effect_type.value in outcome.measurements:
                measured = outcome.measurements[effect.effect_type.value]
                expected = effect.measurement

                # Calculate alignment
                # FIXED: Use epsilon for float comparison
                epsilon = 1e-9
                if abs(expected) > epsilon:
                    alignment = 1.0 - min(1.0, abs(measured - expected) / abs(expected))
                else:
                    # Both expected and measured should be near zero
                    alignment = 1.0 if abs(measured) < epsilon else 0.0

                alignment_scores.append(alignment)

        if alignment_scores:
            support_score += np.mean(alignment_scores) * 0.3

        return min(1.0, support_score)

    def _update_confidence(self):
        """Update concept confidence based on evidence"""
        if self.evidence_count == 0:
            return

        # Base confidence on success rate
        base_confidence = self.success_rate

        # Adjust for evidence count (more evidence = more confidence)
        evidence_factor = min(1.0, self.evidence_count / 30)

        # Adjust for recency
        if self.evidence_history:
            recent_evidence = list(self.evidence_history)[-5:]
            recent_success = sum(1 for e in recent_evidence if e["success"]) / len(
                recent_evidence
            )
            recency_factor = recent_success
        else:
            recency_factor = 0.5

        # Update confidence
        self.confidence = (
            base_confidence * 0.4 + evidence_factor * 0.3 + recency_factor * 0.3
        )

    def _calculate_grounding_status(self) -> GroundingStatus:
        """Calculate current grounding status"""
        grounding_conf = self.get_grounding_confidence()

        if grounding_conf >= 0.8 and self.evidence_count >= 20:
            return GroundingStatus.STRONGLY_GROUNDED
        elif grounding_conf >= 0.6 and self.evidence_count >= 10:
            return GroundingStatus.GROUNDED
        elif grounding_conf >= 0.4 and self.evidence_count >= 5:
            return GroundingStatus.WEAKLY_GROUNDED
        else:
            return GroundingStatus.UNGROUNDED

    def to_dict(self) -> Dict[str, Any]:
        """Convert concept to dictionary"""
        return {
            "concept_id": self.concept_id,
            "pattern_signature": self.pattern_signature,
            "confidence": self.confidence,
            "grounding_status": self.grounding_status.value,
            "grounding_confidence": self.get_grounding_confidence(),
            "stability_score": self.calculate_stability_score(),
            "evidence_count": self.evidence_count,
            "success_rate": self.success_rate,
            "grounded_effects": len(self.grounded_effects),
            "last_update": self.last_update,
            "domains": list(self.domains),
            "usage_count": self.usage_count,
        }


class ConceptMapper:
    """Maps patterns to concepts with grounding validation - FIXED with safety and world_model"""

    def __init__(
        self,
        world_model=None,
        domain_registry=None,
        safety_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize concept mapper - FIXED: Added world_model, domain_registry, and safety_config

        Args:
            world_model: World model instance for accessing causal knowledge
            domain_registry: Domain registry for domain-specific thresholds
            safety_config: Optional safety configuration
        """
        self.world_model = world_model
        self.domain_registry = domain_registry

        # Initialize safety validator
        if SAFETY_VALIDATOR_AVAILABLE:
            if isinstance(safety_config, dict) and safety_config:
                self.safety_validator = EnhancedSafetyValidator(
                    SafetyConfig.from_dict(safety_config)
                )
            else:
                self.safety_validator = EnhancedSafetyValidator()
            logger.info("ConceptMapper: Safety validator initialized")
        else:
            self.safety_validator = None
            logger.warning(
                "ConceptMapper: Safety validator not available - operating without safety checks"
            )

        # FIXED: Add size limits to all unbounded dictionaries
        self.concepts = {}  # pattern_signature -> Concept
        self.max_concepts = 10000

        self.effect_library = {}  # effect_id -> MeasurableEffect
        self.max_effects = 50000

        self.pattern_outcomes = {}  # FIXED: Changed from defaultdict to regular dict
        self.max_patterns = 5000
        self.max_outcomes_per_pattern = 100

        # FIXED: Add size limit to category_overrides
        self.category_overrides = {}
        self.max_category_overrides = 1000

        # FIXED: Add size limit to archived_concepts
        self.archived_concepts = {}
        self.max_archived = 10000

        # Configuration - base values (FIXED: domain-adaptive thresholds)
        self.base_min_instances = 5
        self.base_consistency_threshold = 0.7
        self.base_grounding_confidence = 0.6

        # Domain-specific overrides
        self.domain_thresholds = {}

        # Decay tracking (FIXED: concept decay)
        self._last_decay_time = time.time()

        # Statistics
        self.total_patterns_processed = 0
        self.total_concepts_created = 0
        self.grounded_concepts = 0

        # FIXED: Replace defaultdict(int) with Counter
        self.safety_blocks = Counter()
        self.safety_corrections = Counter()

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            "ConceptMapper initialized (production-ready) with bounded data structures"
        )

    def get_thresholds_for_domain(self, domain: str) -> Dict[str, float]:
        """
        Get thresholds adjusted for domain criticality (FIXED: domain-adaptive)

        Args:
            domain: Domain name

        Returns:
            Dictionary with threshold values
        """
        if domain in self.domain_thresholds:
            return self.domain_thresholds[domain]

        # Use domain registry if available
        if hasattr(self, "domain_registry") and self.domain_registry:
            profile = self.domain_registry.domains.get(domain)
            if profile:
                criticality = profile.criticality_score
                # Higher criticality = stricter thresholds
                return {
                    "min_instances": int(self.base_min_instances * (1 + criticality)),
                    "consistency": self.base_consistency_threshold
                    + (criticality * 0.2),
                    "grounding_confidence": self.base_grounding_confidence
                    + (criticality * 0.3),
                }

        return {
            "min_instances": self.base_min_instances,
            "consistency": self.base_consistency_threshold,
            "grounding_confidence": self.base_grounding_confidence,
        }

    def decay_unused_concepts(
        self, max_age_days: int = 30, min_usage: int = 5
    ) -> List[str]:
        """
        Remove or archive concepts that haven't been used (FIXED: concept decay)

        Args:
            max_age_days: Maximum age in days for unused concepts
            min_usage: Minimum usage count to keep

        Returns:
            List of removed concept IDs
        """
        with self._lock:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            removed = []

            for concept_id, concept in list(self.concepts.items()):
                age = current_time - concept.creation_time

                # Remove if old and unused
                if age > max_age_seconds and concept.usage_count < min_usage:
                    # Archive before removing
                    self._archive_concept(concept)

                    del self.concepts[concept_id]
                    removed.append(concept_id)
                    logger.info(
                        "Removed unused concept %s (age: %.1f days, usage: %d)",
                        concept_id,
                        age / (24 * 3600),
                        concept.usage_count,
                    )

            return removed

    def _archive_concept(self, concept: Concept):
        """
        Archive concept before removal (FIXED: concept decay)

        Args:
            concept: Concept to archive
        """
        # FIXED: Enforce archive size limit
        if len(self.archived_concepts) >= self.max_archived:
            # Remove oldest archived concept
            oldest_key = min(
                self.archived_concepts.keys(),
                key=lambda k: self.archived_concepts[k]["archive_time"],
            )
            del self.archived_concepts[oldest_key]

        self.archived_concepts[concept.concept_id] = {
            "concept_data": concept.to_dict(),
            "archive_time": time.time(),
            "archive_reason": "unused",
        }

    def set_effect_category_override(
        self, measure_pattern: str, effect_type: EffectType
    ):
        """
        Allow manual override of effect categorization (FIXED: improved categorization)

        Args:
            measure_pattern: Regex pattern or exact name
            effect_type: Target effect type
        """
        # FIXED: Enforce size limit on category_overrides
        if len(self.category_overrides) >= self.max_category_overrides:
            # Remove random override (could use LRU)
            oldest_key = next(iter(self.category_overrides))
            del self.category_overrides[oldest_key]

        self.category_overrides[measure_pattern] = effect_type
        logger.info(
            "Added category override: %s -> %s", measure_pattern, effect_type.value
        )

    def map_pattern_to_concept(self, pattern, domain: str = "general") -> Concept:
        """
        Map pattern to concept - FIXED: Added for semantic bridge compatibility

        Args:
            pattern: Pattern to map
            domain: Domain for the pattern

        Returns:
            Concept for the pattern
        """
        with self._lock:
            pattern_sig = self._get_pattern_signature(pattern)

            # Check if concept already exists
            if pattern_sig in self.concepts:
                concept = self.concepts[pattern_sig]
                concept.domains.add(domain)
                return concept

            # FIXED: Enforce concept limit before creating new
            if len(self.concepts) >= self.max_concepts:
                self._evict_least_valuable_concept()

            # Create new concept with minimal grounded effects
            effects = []
            if hasattr(pattern, "expected_effects"):
                for effect_name, effect_value in pattern.expected_effects.items():
                    effect = MeasurableEffect(
                        effect_id=f"effect_{effect_name}_{hashlib.md5(effect_name.encode(), usedforsecurity=False).hexdigest()[:6]}",
                        effect_type=self._categorize_effect(effect_name),
                        measurement=effect_value,
                        unit=self._infer_unit(effect_name, effect_value),
                        confidence=0.5,
                    )
                    effects.append(effect)

            # Create concept
            concept = Concept(
                pattern_signature=pattern_sig, grounded_effects=effects, confidence=0.5
            )
            concept.domains.add(domain)

            # Store concept
            self.concepts[pattern_sig] = concept
            self.total_concepts_created += 1

            # FIXED: Link to world model if available
            if self.world_model and effects:
                try:
                    self._link_concept_to_world_model(concept, effects)
                except Exception as e:
                    logger.debug("Failed to link concept to world model: %s", e)

            logger.debug(
                "Mapped pattern to concept %s in domain %s", concept.concept_id, domain
            )

            return concept

    def _evict_least_valuable_concept(self):
        """
        Evict least valuable concept when at capacity (FIXED: concept size limit)
        """
        # Find concept with lowest value score
        worst_concept = None
        worst_score = float("inf")

        for concept_id, concept in self.concepts.items():
            # Value = usage_count * success_rate * grounding_confidence
            value_score = (
                concept.usage_count
                * concept.success_rate
                * concept.get_grounding_confidence()
            )

            if value_score < worst_score:
                worst_score = value_score
                worst_concept = concept_id

        if worst_concept:
            # Archive before removing
            self._archive_concept(self.concepts[worst_concept])
            del self.concepts[worst_concept]
            logger.debug("Evicted concept %s (value: %.2f)", worst_concept, worst_score)

    def register_concept(self, concept: Concept):
        """
        Register existing concept - FIXED: Added for semantic bridge compatibility

        Args:
            concept: Concept to register
        """
        with self._lock:
            # FIXED: Enforce limit
            if len(self.concepts) >= self.max_concepts:
                self._evict_least_valuable_concept()

            self.concepts[concept.concept_id] = concept
            logger.debug("Registered concept %s", concept.concept_id)

    def find_similar_concepts(
        self, concept: Concept, top_k: int = 5
    ) -> List[Tuple[Concept, float]]:
        """
        Find similar concepts - FIXED: Added for semantic bridge compatibility

        Args:
            concept: Concept to find similarities for
            top_k: Number of similar concepts to return

        Returns:
            List of (concept, similarity_score) tuples
        """
        with self._lock:
            similar = []

            for other_sig, other in self.concepts.items():
                if other.concept_id == concept.concept_id:
                    continue

                # Calculate similarity based on multiple factors
                similarity = 0.0

                # Domain overlap
                if hasattr(concept, "domains") and hasattr(other, "domains"):
                    if concept.domains and other.domains:
                        domain_overlap = len(concept.domains & other.domains) / max(
                            1, len(concept.domains | other.domains)
                        )
                        similarity += domain_overlap * 0.3

                # Effect type overlap
                concept_effect_types = set(
                    e.effect_type for e in concept.grounded_effects
                )
                other_effect_types = set(e.effect_type for e in other.grounded_effects)
                if concept_effect_types and other_effect_types:
                    effect_overlap = len(
                        concept_effect_types & other_effect_types
                    ) / max(1, len(concept_effect_types | other_effect_types))
                    similarity += effect_overlap * 0.4

                # Success rate similarity
                if hasattr(concept, "success_rate") and hasattr(other, "success_rate"):
                    success_similarity = 1.0 - abs(
                        concept.success_rate - other.success_rate
                    )
                    similarity += success_similarity * 0.3

                if similarity > 0:
                    similar.append((other, similarity))

            # Sort by similarity
            similar.sort(key=lambda x: x[1], reverse=True)

            return similar[:top_k]

    def extract_measurable_effects(
        self, outcomes: List[PatternOutcome]
    ) -> List[MeasurableEffect]:
        """
        Extract measurable effects from outcomes

        Args:
            outcomes: List of pattern outcomes

        Returns:
            List of measurable effects
        """
        # SAFETY: Validate outcomes
        if self.safety_validator:
            safe_outcomes = []
            for outcome in outcomes:
                try:
                    outcome_check = self.safety_validator.analyze_outcome_safety(
                        outcome
                    )
                    if outcome_check.get("safe", True):
                        safe_outcomes.append(outcome)
                    else:
                        logger.debug(
                            "Filtered unsafe outcome: %s",
                            outcome_check.get("reason", "unknown"),
                        )
                        self.safety_blocks["outcome"] += 1
                except Exception as e:
                    logger.debug("Error validating outcome: %s", e)
                    safe_outcomes.append(outcome)  # Allow on error
            outcomes = safe_outcomes

        if not outcomes:
            return []

        effects = []
        effect_groups = defaultdict(list)

        # Group measurements by type
        for outcome in outcomes:
            for measure_name, value in outcome.measurements.items():
                # SAFETY: Validate measurement value
                if not np.isfinite(value):
                    logger.debug("Skipping non-finite measurement: %s", measure_name)
                    self.safety_corrections["non_finite_measurement"] += 1
                    continue

                effect_groups[measure_name].append(
                    {
                        "value": value,
                        "success": outcome.success,
                        "domain": outcome.domain,
                        "timestamp": outcome.timestamp,
                    }
                )

        # Get domain from first outcome for threshold lookup
        domain = outcomes[0].domain if outcomes else "general"
        thresholds = self.get_thresholds_for_domain(domain)
        min_instances = thresholds["min_instances"]

        # Create effects from consistent measurements
        for measure_name, measurements in effect_groups.items():
            if len(measurements) >= min_instances:
                # Calculate statistics
                values = [m["value"] for m in measurements]
                mean_value = np.mean(values)
                std_value = np.std(values)

                # SAFETY: Validate statistics
                if not np.isfinite(mean_value) or not np.isfinite(std_value):
                    logger.debug(
                        "Skipping effect with non-finite statistics: %s", measure_name
                    )
                    self.safety_corrections["non_finite_stats"] += 1
                    continue

                # Check consistency
                if std_value < mean_value * 0.3 or std_value < 0.1:  # Consistent enough
                    effect = MeasurableEffect(
                        effect_id=f"effect_{measure_name}_{hashlib.md5(measure_name.encode(), usedforsecurity=False).hexdigest()[:6]}",
                        effect_type=self._categorize_effect(measure_name),
                        measurement=mean_value,
                        unit=self._infer_unit(measure_name, mean_value),
                        confidence=self._calculate_effect_confidence(values),
                    )
                    effects.append(effect)

                    # FIXED: Store in library with size limit
                    if len(self.effect_library) >= self.max_effects:
                        # Remove oldest effect
                        oldest_key = min(
                            self.effect_library.keys(),
                            key=lambda k: self.effect_library[k].timestamp,
                        )
                        del self.effect_library[oldest_key]

                    self.effect_library[effect.effect_id] = effect

        logger.debug(
            "Extracted %d measurable effects from %d outcomes",
            len(effects),
            len(outcomes),
        )

        return effects

    def validate_effect_consistency(
        self, pattern: Any, effects: List[MeasurableEffect], min_instances: int = None
    ) -> bool:
        """
        Validate consistency of effects for a pattern

        Args:
            pattern: Pattern to validate
            effects: Effects to check
            min_instances: Minimum instances required (uses default if None)

        Returns:
            True if effects are consistent
        """
        if min_instances is None:
            # Get domain from pattern (FIXED: domain-adaptive thresholds)
            domain = getattr(pattern, "domain", "general")
            thresholds = self.get_thresholds_for_domain(domain)
            min_instances = thresholds["min_instances"]

        pattern_sig = self._get_pattern_signature(pattern)

        # Get historical outcomes for this pattern
        historical_outcomes = self.pattern_outcomes.get(pattern_sig, [])

        if len(historical_outcomes) < min_instances:
            logger.debug(
                "Insufficient instances for validation: %d < %d",
                len(historical_outcomes),
                min_instances,
            )
            return False

        # Extract historical effects
        historical_effects = self.extract_measurable_effects(historical_outcomes)

        # Check consistency between new and historical effects
        consistency_scores = []

        for new_effect in effects:
            # Find matching historical effects
            matches = [
                h for h in historical_effects if h.effect_type == new_effect.effect_type
            ]

            if matches:
                # Check consistency with each match
                for historical_effect in matches:
                    if new_effect.is_consistent_with(historical_effect):
                        consistency_scores.append(1.0)
                    else:
                        consistency_scores.append(0.0)

        if not consistency_scores:
            return False

        # Calculate overall consistency
        overall_consistency = np.mean(consistency_scores)

        # Get domain-specific threshold
        domain = getattr(pattern, "domain", "general")
        thresholds = self.get_thresholds_for_domain(domain)
        consistency_threshold = thresholds["consistency"]

        is_consistent = overall_consistency >= consistency_threshold

        logger.debug(
            "Effect consistency for pattern %s: %.2f (threshold: %.2f)",
            pattern_sig[:8],
            overall_consistency,
            consistency_threshold,
        )

        # FIXED: Convert numpy bool to Python bool
        return bool(is_consistent)

    def calculate_grounding_confidence(self, effects: List[MeasurableEffect]) -> float:
        """
        Calculate confidence in effect grounding

        Args:
            effects: List of effects to evaluate

        Returns:
            Grounding confidence score [0, 1]
        """
        if not effects:
            return 0.0

        # Factor 1: Individual effect confidences
        effect_confidences = [e.confidence for e in effects]
        avg_confidence = np.mean(effect_confidences)

        # Factor 2: Effect diversity (multiple types of effects = better grounding)
        effect_types = set(e.effect_type for e in effects)
        diversity_factor = min(1.0, len(effect_types) / 3.0)

        # Factor 3: Measurement consistency
        consistency_scores = []
        for effect_type in effect_types:
            type_effects = [e for e in effects if e.effect_type == effect_type]
            if len(type_effects) > 1:
                # Check pairwise consistency
                for i, e1 in enumerate(type_effects):
                    for e2 in type_effects[i + 1 :]:
                        consistency_scores.append(
                            1.0 if e1.is_consistent_with(e2) else 0.0
                        )

        consistency_factor = np.mean(consistency_scores) if consistency_scores else 0.5

        # Factor 4: Critical effect coverage
        has_performance = any(e.effect_type == EffectType.PERFORMANCE for e in effects)
        has_resource = any(e.effect_type == EffectType.RESOURCE for e in effects)
        critical_factor = (1.0 if has_performance else 0.0) * 0.5 + (
            1.0 if has_resource else 0.0
        ) * 0.5

        # Weighted combination
        grounding_confidence = (
            avg_confidence * 0.3
            + diversity_factor * 0.2
            + consistency_factor * 0.3
            + critical_factor * 0.2
        )

        logger.debug(
            "Grounding confidence: %.2f (effects: %d, types: %d)",
            grounding_confidence,
            len(effects),
            len(effect_types),
        )

        return min(1.0, grounding_confidence)

    def create_concept(
        self, pattern: Any, effects: List[MeasurableEffect], evidence_count: int = 0
    ) -> Concept:
        """
        Create concept from pattern and effects

        Args:
            pattern: Source pattern
            effects: Grounded effects
            evidence_count: Initial evidence count

        Returns:
            Created concept
        """
        with self._lock:
            pattern_sig = self._get_pattern_signature(pattern)

            # Calculate initial confidence
            grounding_conf = self.calculate_grounding_confidence(effects)
            initial_confidence = grounding_conf * 0.7 + 0.3  # Minimum 0.3 confidence

            # Create concept
            concept = Concept(
                pattern_signature=pattern_sig,
                grounded_effects=effects,
                confidence=initial_confidence,
            )

            # Set initial evidence if provided
            if evidence_count > 0:
                concept.evidence_count = evidence_count
                concept.positive_evidence = int(evidence_count * initial_confidence)
                concept.negative_evidence = evidence_count - concept.positive_evidence

            # FIXED: Enforce concept limit
            if len(self.concepts) >= self.max_concepts:
                self._evict_least_valuable_concept()

            # Store concept
            self.concepts[pattern_sig] = concept

            # Update statistics
            self.total_concepts_created += 1
            if concept.grounding_status in [
                GroundingStatus.GROUNDED,
                GroundingStatus.STRONGLY_GROUNDED,
            ]:
                self.grounded_concepts += 1

            # FIXED: Link to world model if available
            if self.world_model:
                try:
                    self._link_concept_to_world_model(concept, effects)
                except Exception as e:
                    logger.debug("Failed to link concept to world model: %s", e)

            logger.info(
                "Created concept %s with %d effects (confidence: %.2f, grounding: %s)",
                concept.concept_id,
                len(effects),
                initial_confidence,
                concept.grounding_status.value,
            )

            return concept

    def _link_concept_to_world_model(
        self, concept: Concept, effects: List[MeasurableEffect]
    ):
        """
        Link concept to world model causal graph - FIXED: New integration method

        Args:
            concept: Concept to link
            effects: Effects to link
        """
        if not self.world_model or not hasattr(self.world_model, "causal_graph"):
            return

        # For each effect, try to find causal relationships
        for effect in effects:
            effect_var = f"effect_{effect.effect_type.value}"

            # Add as a node in causal graph if not exists
            try:
                if not self.world_model.causal_graph.has_node(effect_var):
                    self.world_model.causal_graph.add_node(effect_var)

                # Link pattern to effect
                pattern_var = f"pattern_{concept.concept_id[:8]}"

                if not self.world_model.causal_graph.has_edge(pattern_var, effect_var):
                    # Validate with safety
                    if self.safety_validator:
                        try:
                            if hasattr(self.safety_validator, "validate_causal_edge"):
                                edge_validation = (
                                    self.safety_validator.validate_causal_edge(
                                        pattern_var, effect_var, effect.confidence
                                    )
                                )
                                if not edge_validation.get("safe", True):
                                    continue
                        except Exception as e:
                            logger.debug("Safety validation error: %s", e)
                            continue

                    self.world_model.causal_graph.add_edge(
                        pattern_var,
                        effect_var,
                        strength=effect.confidence,
                        evidence_type="concept_mapper",
                    )
                    logger.debug(
                        "Linked concept to world model: %s -> %s",
                        pattern_var,
                        effect_var,
                    )
            except Exception as e:
                logger.debug("Error linking to world model: %s", e)

    def process_pattern_outcomes(
        self, pattern: Any, outcomes: List[PatternOutcome]
    ) -> Optional[Concept]:
        """
        Process pattern with outcomes to create or update concept

        Args:
            pattern: Pattern that generated outcomes
            outcomes: Outcomes from pattern execution

        Returns:
            Concept if created or updated, None if insufficient data
        """
        if not outcomes:
            return None

        with self._lock:
            pattern_sig = self._get_pattern_signature(pattern)

            # FIXED: Enforce pattern limit
            if pattern_sig not in self.pattern_outcomes:
                if len(self.pattern_outcomes) >= self.max_patterns:
                    # Remove pattern with fewest outcomes
                    min_pattern = min(
                        self.pattern_outcomes.keys(),
                        key=lambda k: len(self.pattern_outcomes[k]),
                    )
                    del self.pattern_outcomes[min_pattern]
                self.pattern_outcomes[pattern_sig] = []

            # FIXED: Enforce outcomes per pattern limit
            pattern_outcomes = self.pattern_outcomes[pattern_sig]
            for outcome in outcomes:
                if len(pattern_outcomes) >= self.max_outcomes_per_pattern:
                    # Remove oldest outcome
                    pattern_outcomes.pop(0)
                pattern_outcomes.append(outcome)

            self.total_patterns_processed += 1

            # Periodically decay concepts (FIXED: concept decay)
            if time.time() - self._last_decay_time > 24 * 3600:  # Daily
                self.decay_unused_concepts()
                self._last_decay_time = time.time()

            # Check if concept already exists
            if pattern_sig in self.concepts:
                concept = self.concepts[pattern_sig]
                concept.update_evidence(outcomes)
                return concept

            # Get domain-specific thresholds
            domain = getattr(
                pattern, "domain", outcomes[0].domain if outcomes else "general"
            )
            thresholds = self.get_thresholds_for_domain(domain)
            min_instances = thresholds["min_instances"]

            # Try to create new concept if enough evidence
            if len(self.pattern_outcomes[pattern_sig]) >= min_instances:
                # Extract effects
                effects = self.extract_measurable_effects(
                    self.pattern_outcomes[pattern_sig]
                )

                if effects:
                    # Validate consistency
                    if self.validate_effect_consistency(pattern, effects):
                        # Create concept
                        concept = self.create_concept(
                            pattern, effects, len(self.pattern_outcomes[pattern_sig])
                        )

                        # Update with all historical outcomes
                        concept.update_evidence(self.pattern_outcomes[pattern_sig])

                        return concept

            return None

    def _get_pattern_signature(self, pattern: Any) -> str:
        """Get unique signature for pattern"""
        if hasattr(pattern, "get_signature"):
            return pattern.get_signature()
        elif hasattr(pattern, "pattern_id"):
            return pattern.pattern_id
        elif hasattr(pattern, "pattern_signature"):
            return pattern.pattern_signature
        else:
            return hashlib.md5(str(pattern).encode(), usedforsecurity=False).hexdigest()

    def _categorize_effect(self, measure_name: str) -> EffectType:
        """
        Categorize effect using pattern matching and domain knowledge (FIXED: improved)

        Args:
            measure_name: Name of the measurement

        Returns:
            Effect type
        """
        # Check for manual overrides first
        for pattern, effect_type in self.category_overrides.items():
            if pattern in measure_name or pattern.lower() in measure_name.lower():
                return effect_type

        measure_lower = measure_name.lower()

        # Define categorization rules with priority
        categorization_rules = [
            # Temporal (highest priority for time-related)
            (
                EffectType.TEMPORAL,
                ["time", "latency", "duration", "delay", "speed", "throughput"],
            ),
            # Resource (memory, CPU, etc.)
            (
                EffectType.RESOURCE,
                ["memory", "cpu", "resource", "usage", "consumption", "allocation"],
            ),
            # Performance (accuracy, success)
            (
                EffectType.PERFORMANCE,
                [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "success",
                    "performance",
                    "score",
                    "error_rate",
                    "quality",
                ],
            ),
            # Behavioral
            (
                EffectType.BEHAVIORAL,
                [
                    "behavior",
                    "behaviour",
                    "action",
                    "response",
                    "reaction",
                    "decision",
                    "choice",
                ],
            ),
            # Structural (lowest priority - most generic)
            (
                EffectType.STRUCTURAL,
                ["structure", "topology", "architecture", "layout", "organization"],
            ),
        ]

        # Check rules in order
        for effect_type, keywords in categorization_rules:
            if any(keyword in measure_lower for keyword in keywords):
                return effect_type

        # Default to structural if no match
        return EffectType.STRUCTURAL

    def _infer_unit(self, measure_name: str, value: float) -> str:
        """Infer unit of measurement"""
        measure_lower = measure_name.lower()

        if "time" in measure_lower or "duration" in measure_lower:
            if value < 1:
                return "ms"
            elif value < 60:
                return "s"
            else:
                return "min"
        elif "memory" in measure_lower:
            if value < 1024:
                return "KB"
            elif value < 1024 * 1024:
                return "MB"
            else:
                return "GB"
        elif "percent" in measure_lower or "rate" in measure_lower:
            return "%"
        elif "count" in measure_lower:
            return "count"
        else:
            return "units"

    def _calculate_effect_confidence(self, values: List[float]) -> float:
        """Calculate confidence for an effect based on measurements"""
        if not values:
            return 0.0

        # More measurements = higher confidence
        count_factor = min(1.0, len(values) / 20)

        # Lower variance = higher confidence
        if len(values) > 1:
            cv = np.std(values) / (np.mean(values) + 1e-10)  # Coefficient of variation
            consistency_factor = 1.0 / (1.0 + cv)
        else:
            consistency_factor = 0.5

        return count_factor * 0.5 + consistency_factor * 0.5

    def get_statistics(self) -> Dict[str, Any]:
        """Get mapper statistics"""
        stats = {
            "total_patterns_processed": self.total_patterns_processed,
            "total_concepts_created": self.total_concepts_created,
            "grounded_concepts": self.grounded_concepts,
            "grounding_rate": self.grounded_concepts
            / max(1, self.total_concepts_created),
            "active_concepts": len(self.concepts),
            "archived_concepts": len(self.archived_concepts),
            "effect_library_size": len(self.effect_library),
            "pattern_outcomes_tracked": len(self.pattern_outcomes),
            "world_model_connected": self.world_model is not None,
            "domain_registry_connected": self.domain_registry is not None,
            "max_concepts": self.max_concepts,
            "max_effects": self.max_effects,
            "max_patterns": self.max_patterns,
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
