# src/vulcan/world_model/meta_reasoning/internal_critic.py
"""
internal_critic.py - Multi-perspective self-critique and evaluation system
Part of the meta_reasoning subsystem for VULCAN-AMI

FULL PRODUCTION IMPLEMENTATION

WARNING: Risk identification methods are PLACEHOLDER IMPLEMENTATIONS.
The risk detection methods (_identify_*_risks) rely on self-reported flags in proposals
(e.g., 'causes_physical_harm', 'has_security_review') rather than semantic analysis of
proposal content. This is NOT suitable for production security/safety without additional
validation layers. Real-world usage requires:
- Semantic analysis of proposal text/code
- Integration with static analysis tools
- Code scanning and vulnerability detection
- Expert system review for safety-critical applications
- DO NOT rely on these placeholders for actual security/safety decisions

Comprehensive internal critique system with:
- Multi-perspective evaluation (logic, feasibility, safety, alignment, efficiency)
- Automated risk identification and assessment (PLACEHOLDER - see warning above)
- Comparative analysis of alternatives
- Iterative refinement suggestions
- Learning from critique outcomes
- Pattern recognition in successful vs failed proposals
- Integration with validation and safety systems

Evaluation Perspectives:
- LOGICAL_CONSISTENCY: Internal logic and coherence
- FEASIBILITY: Practical implementability
- SAFETY: Risk and harm potential
- ALIGNMENT: Goal and value alignment
- EFFICIENCY: Resource utilization
- COMPLETENESS: Coverage and thoroughness
- CLARITY: Explainability and understanding
- ROBUSTNESS: Resilience to edge cases

Critique Levels:
- CRITICAL: Must fix before proceeding
- MAJOR: Significant issue requiring attention
- MINOR: Improvement recommended
- SUGGESTION: Optional enhancement

Risk Categories:
- SAFETY: Physical or psychological harm
- SECURITY: Vulnerabilities or exploits
- PERFORMANCE: Degradation or failures
- RESOURCE: Excessive consumption
- ETHICAL: Boundary violations
- OPERATIONAL: Process failures

Integration:
- Learns from ValidationTracker outcomes
- Alerts EthicalBoundaryMonitor for concerns
- Records to TransparencyInterface for audit
- Adapts with SelfImprovementDrive

Thread-safe with comprehensive evaluation history.
"""

import logging
import threading
import time  # Moved import
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from vulcan.world_model.meta_reasoning.numpy_compat import np, NUMPY_AVAILABLE
from vulcan.world_model.meta_reasoning.serialization_mixin import SerializationMixin

logger = logging.getLogger(__name__)


class CritiqueLevel(Enum):
    """Severity level of critique"""

    CRITICAL = "critical"  # Must fix before proceeding
    MAJOR = "major"  # Significant issue requiring attention
    MINOR = "minor"  # Improvement recommended
    SUGGESTION = "suggestion"  # Optional enhancement


class EvaluationPerspective(Enum):
    """Perspective from which to evaluate"""

    LOGICAL_CONSISTENCY = "logical_consistency"
    FEASIBILITY = "feasibility"
    SAFETY = "safety"
    ALIGNMENT = "alignment"
    EFFICIENCY = "efficiency"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    ROBUSTNESS = "robustness"


class RiskCategory(Enum):
    """Category of identified risk"""

    SAFETY = "safety"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ETHICAL = "ethical"
    OPERATIONAL = "operational"


class RiskSeverity(Enum):
    """Severity of identified risk"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class Critique:
    """A critique of a specific aspect"""

    level: CritiqueLevel
    perspective: EvaluationPerspective
    aspect: str  # Specific aspect being critiqued
    description: str
    evidence: List[str] = field(default_factory=list)
    suggested_improvement: Optional[str] = None
    confidence: float = 0.5  # 0-1
    impact_if_ignored: float = 0.5  # 0-1, potential negative impact
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "level": self.level.value,
            "perspective": self.perspective.value,
            "aspect": self.aspect,
            "description": self.description,
            "evidence": self.evidence,
            "suggested_improvement": self.suggested_improvement,
            "confidence": self.confidence,
            "impact_if_ignored": self.impact_if_ignored,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Risk:
    """Identified risk in proposal"""

    category: RiskCategory
    severity: RiskSeverity
    description: str
    likelihood: float  # 0-1
    impact: float  # 0-1
    risk_score: float = 0.0  # Computed: likelihood * impact
    mitigation_strategies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute risk score"""
        self.risk_score = self.likelihood * self.impact

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "mitigation_strategies": self.mitigation_strategies,
            "metadata": self.metadata,
        }


@dataclass
class PerspectiveScore:
    """Score from a specific evaluation perspective"""

    perspective: EvaluationPerspective
    score: float  # 0-1, higher = better
    weight: float  # Importance weight for this perspective
    rationale: str
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "perspective": self.perspective.value,
            "score": self.score,
            "weight": self.weight,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


@dataclass
class Evaluation:
    """Comprehensive evaluation of a proposal"""

    proposal_id: str

    # Overall assessment
    overall_score: float  # 0-1, weighted combination of perspectives
    overall_assessment: str
    recommendation: str  # approve/modify/reject

    # Detailed scores
    perspective_scores: List[PerspectiveScore]

    # Critiques and analysis
    critiques: List[Critique]
    risks: List[Risk]
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]

    # Confidence
    evaluation_confidence: float = 0.7  # Confidence in this evaluation

    # Metadata
    timestamp: float = field(default_factory=time.time)
    evaluator_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_critical_issues(self) -> List[Critique]:
        """Get critical-level critiques"""
        return [c for c in self.critiques if c.level == CritiqueLevel.CRITICAL]

    def get_high_risks(self) -> List[Risk]:
        """Get high or critical severity risks"""
        return [
            r
            for r in self.risks
            if r.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "proposal_id": self.proposal_id,
            "overall_score": self.overall_score,
            "overall_assessment": self.overall_assessment,
            "recommendation": self.recommendation,
            "perspective_scores": [ps.to_dict() for ps in self.perspective_scores],
            "critiques": [c.to_dict() for c in self.critiques],
            "risks": [r.to_dict() for r in self.risks],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvements": self.improvements,
            "evaluation_confidence": self.evaluation_confidence,
            "critical_issues": len(self.get_critical_issues()),
            "high_risks": len(self.get_high_risks()),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple proposals"""

    best_proposal_id: str
    ranking: List[Tuple[str, float]]  # (proposal_id, score) sorted
    comparison_matrix: Dict[Tuple[str, str], str]  # (id1, id2) -> "better/worse/equal"
    rationale: str
    trade_offs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Convert tuple keys in matrix to strings for JSON compatibility
        serializable_matrix = {
            f"{k[0]}_{k[1]}": v for k, v in self.comparison_matrix.items()
        }
        return {
            "best_proposal_id": self.best_proposal_id,
            "ranking": self.ranking,
            "comparison_matrix_str_keys": serializable_matrix,  # Use new key
            "rationale": self.rationale,
            "trade_offs": self.trade_offs,
            "metadata": self.metadata,
        }


class InternalCritic(SerializationMixin):
    """
    Multi-perspective internal critique and evaluation system

    Provides comprehensive evaluation of proposals from multiple perspectives:
    - Logical consistency and coherence
    - Practical feasibility
    - Safety and risk assessment
    - Goal and value alignment
    - Resource efficiency
    - Completeness and coverage
    - Clarity and explainability
    - Robustness and edge cases

    Features:
    - Multi-criteria evaluation with configurable weights
    - Automated critique generation with evidence
    - Risk identification and assessment
    - Comparative analysis of alternatives
    - Iterative refinement suggestions
    - Learning from validation outcomes
    - Pattern recognition in evaluations

    Thread-safe with comprehensive evaluation history.
    Integrates with VULCAN validation and safety systems.
    """

    _unpickleable_attrs = ['lock', '_np', 'evaluation_criteria']

    def __init__(
        self,
        perspective_weights: Optional[Dict[EvaluationPerspective, float]] = None,
        strict_mode: bool = False,
        max_history: int = 10000,
        validation_tracker=None,
        ethical_boundary_monitor=None,
        transparency_interface=None,
    ):
        """
        Initialize internal critic

        Args:
            perspective_weights: Optional custom weights for perspectives
            strict_mode: If True, stricter evaluation criteria
            max_history: Maximum evaluation history to keep
            validation_tracker: Optional ValidationTracker for learning
            ethical_boundary_monitor: Optional EthicalBoundaryMonitor for safety
            transparency_interface: Optional TransparencyInterface for audit
        """
        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        self.strict_mode = strict_mode
        self.max_history = max_history
        self.validation_tracker = validation_tracker
        self.ethical_boundary_monitor = ethical_boundary_monitor
        self.transparency_interface = transparency_interface

        # Perspective weights (importance of each perspective)
        default_weights = {
            EvaluationPerspective.LOGICAL_CONSISTENCY: 0.15,
            EvaluationPerspective.FEASIBILITY: 0.15,
            EvaluationPerspective.SAFETY: 0.20,  # Higher weight for safety
            EvaluationPerspective.ALIGNMENT: 0.15,
            EvaluationPerspective.EFFICIENCY: 0.10,
            EvaluationPerspective.COMPLETENESS: 0.10,
            EvaluationPerspective.CLARITY: 0.08,
            EvaluationPerspective.ROBUSTNESS: 0.07,
        }
        # Merge provided weights with defaults
        merged_weights = default_weights.copy()
        if perspective_weights:
            merged_weights.update(perspective_weights)
        self.perspective_weights = merged_weights

        # Normalize weights
        total_weight = sum(self.perspective_weights.values())
        if (
            total_weight > 0 and abs(total_weight - 1.0) > 1e-6
        ):  # Normalize if not already sum to 1
            logger.info(f"Normalizing perspective weights (total was {total_weight})")
            self.perspective_weights = {
                k: v / total_weight for k, v in self.perspective_weights.items()
            }
        elif total_weight == 0:
            logger.warning("All perspective weights are zero. Using equal weights.")
            num_perspectives = len(EvaluationPerspective)
            equal_weight = 1.0 / num_perspectives if num_perspectives > 0 else 0
            self.perspective_weights = {p: equal_weight for p in EvaluationPerspective}

        # Evaluation history
        self.evaluation_history: deque = deque(maxlen=max_history)
        self.evaluations_by_id: Dict[str, Evaluation] = {}

        # Critique patterns (learn common critique types)
        self.critique_patterns: Dict[str, int] = defaultdict(int)
        self.successful_critique_patterns: Dict[str, int] = defaultdict(int)

        # Risk patterns
        self.risk_patterns: Dict[RiskCategory, List[str]] = defaultdict(list)

        # Statistics
        self.stats = defaultdict(int)
        self.stats["initialized_at"] = time.time()

        # Learning parameters
        self.critique_effectiveness: Dict[str, float] = (
            {}
        )  # critique_type -> effectiveness
        self.adaptive_weights: bool = True

        # Evaluation criteria (can be customized)
        self.evaluation_criteria: Dict[EvaluationPerspective, List[Callable]] = {}
        self._initialize_default_criteria()

        # Thread safety
        self.lock = threading.RLock()

        logger.info("InternalCritic initialized (FULL IMPLEMENTATION)")
        logger.info(
            f"  Strict mode: {strict_mode}, Adaptive weights: {self.adaptive_weights}"
        )
        logger.info(
            f"  Perspective weights: {{p.value: w for p, w in self.perspective_weights.items()}}"
        )

    def _restore_unpickleable_attrs(self) -> None:
        """Restore unpickleable attributes after deserialization."""
        self.lock = threading.RLock()
        self._np = np if NUMPY_AVAILABLE else FakeNumpy
        self.evaluation_criteria = {}
        self._initialize_default_criteria()

    def evaluate_proposal(
        self, proposal: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Evaluation:
        """
        Comprehensive evaluation of proposal

        Args:
            proposal: Proposal to evaluate
            context: Optional context for evaluation

        Returns:
            Evaluation object with scores, critiques, and recommendations
        """
        with self.lock:
            context = context or {}
            # Ensure proposal is a dict
            if not isinstance(proposal, dict):
                logger.error(
                    f"Invalid proposal type: {type(proposal)}. Cannot evaluate."
                )
                # Return a default/error evaluation
                return Evaluation(
                    proposal_id="invalid_proposal_type",
                    overall_score=0.0,
                    overall_assessment="Invalid proposal type provided.",
                    recommendation="reject",
                    perspective_scores=[],
                    critiques=[],
                    risks=[],
                    strengths=[],
                    weaknesses=["Invalid proposal type"],
                    improvements=[],
                )

            proposal_id = proposal.get("id", f"proposal_{time.time_ns()}")

            self.stats["evaluations_performed"] += 1

            # Evaluate from each perspective
            perspective_scores = []
            all_critiques = []
            all_risks = []

            # --- START FIX: Correct evaluation loop ---
            # Iterate over the weights dictionary to ensure we evaluate each perspective
            for perspective, weight in self.perspective_weights.items():
                eval_func_list = self.evaluation_criteria.get(perspective, [])

                # Default values in case no function is found or it fails
                score = 0.5
                critiques = []
                risks = []

                if eval_func_list:
                    # Assuming one function per perspective as per _initialize_default_criteria
                    eval_func = eval_func_list[0]
                    try:
                        # Call the actual evaluation method (e.g., self._evaluate_logical_consistency)
                        score, critiques, risks = eval_func(proposal, context)
                    except Exception as e:
                        logger.error(
                            f"Error during {perspective.value} evaluation: {e}",
                            exc_info=True,
                        )
                        score = 0.0  # Penalize on error
                        critiques = [
                            Critique(
                                CritiqueLevel.CRITICAL,
                                perspective,
                                "evaluation_error",
                                f"Evaluation failed with exception: {e}",
                            )
                        ]
                else:
                    logger.warning(
                        f"No evaluation function found for perspective: {perspective.value}"
                    )

                # Create the PerspectiveScore object *inside* the loop
                perspective_score_obj = PerspectiveScore(
                    perspective=perspective,
                    score=score,
                    weight=weight,  # Use the weight from the loop
                    rationale=self._generate_rationale(perspective, score, critiques),
                )

                perspective_scores.append(perspective_score_obj)
                all_critiques.extend(critiques)
                all_risks.extend(risks)
            # --- END FIX ---

            # Compute overall score (weighted average)
            # Ensure weights sum to 1 (or close enough) before calculating
            total_w = sum(ps.weight for ps in perspective_scores)
            if total_w > 0:
                overall_score = (
                    sum(ps.score * ps.weight for ps in perspective_scores) / total_w
                )
            else:
                overall_score = 0.0  # Avoid division by zero if all weights are zero

            # In strict mode, lower scores for any critiques
            if self.strict_mode and all_critiques:
                critical_penalty = 0.2 * len(
                    [c for c in all_critiques if c.level == CritiqueLevel.CRITICAL]
                )
                major_penalty = 0.1 * len(
                    [c for c in all_critiques if c.level == CritiqueLevel.MAJOR]
                )
                overall_score = max(
                    0.0, overall_score - critical_penalty - major_penalty
                )

            # Identify strengths and weaknesses
            strengths = self._identify_strengths(perspective_scores, proposal)
            weaknesses = self._identify_weaknesses(perspective_scores, all_critiques)

            # Generate improvement suggestions
            improvements = self._generate_improvements(
                all_critiques, all_risks, proposal
            )

            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(
                overall_score, perspective_scores, all_critiques, all_risks
            )

            # Generate recommendation
            recommendation = self._generate_recommendation(
                overall_score, all_critiques, all_risks
            )

            # Compute evaluation confidence
            eval_confidence = self._compute_evaluation_confidence(
                perspective_scores, all_critiques
            )

            # Create evaluation
            evaluation = Evaluation(
                proposal_id=proposal_id,
                overall_score=overall_score,
                overall_assessment=overall_assessment,
                recommendation=recommendation,
                perspective_scores=perspective_scores,
                critiques=all_critiques,
                risks=all_risks,
                strengths=strengths,
                weaknesses=weaknesses,
                improvements=improvements,
                evaluation_confidence=eval_confidence,
                metadata={
                    "context_keys": list(
                        context.keys()
                    ),  # Store keys instead of full context
                    "strict_mode": self.strict_mode,
                },
            )

            # Store evaluation
            self.evaluation_history.append(evaluation)
            # Keep evaluations_by_id bounded if history is bounded
            if len(self.evaluations_by_id) >= self.max_history:
                # Simple FIFO removal for the dict
                oldest_id = next(iter(self.evaluations_by_id))
                if oldest_id != proposal_id:  # Avoid removing the one just added
                    del self.evaluations_by_id[oldest_id]
            self.evaluations_by_id[proposal_id] = evaluation

            # Update patterns
            self._update_patterns(all_critiques, all_risks)

            # Record to transparency interface
            if self.transparency_interface:
                try:
                    self.transparency_interface.record_evaluation(evaluation.to_dict())
                except Exception as e:
                    logger.debug(f"Failed to record to transparency interface: {e}")

            logger.info(
                f"Evaluated proposal {proposal_id}: score={overall_score:.3f}, recommendation={recommendation}"
            )

            return evaluation

    def generate_critique(
        self,
        aspect: str,
        data: Dict[str, Any],
        perspective: Optional[EvaluationPerspective] = None,
    ) -> List[Critique]:
        """
        Generate critiques for a specific aspect

        Args:
            aspect: Aspect to critique
            data: Relevant data (should be part of a proposal-like structure)
            perspective: Optional specific perspective

        Returns:
            List of generated critiques
        """
        with self.lock:
            critiques = []
            # Construct a minimal proposal-like structure for evaluation functions
            pseudo_proposal = {
                "id": f"critique_{aspect}_{time.time_ns()}",
                aspect: data,
            }

            perspectives_to_check = (
                [perspective] if perspective else list(self.perspective_weights.keys())
            )

            for persp in perspectives_to_check:
                # --- START FIX: Call correct function from criteria map ---
                eval_func_list = self.evaluation_criteria.get(persp, [])
                persp_critiques = []

                if eval_func_list:
                    eval_func = eval_func_list[0]
                    try:
                        # _evaluate_perspective returns (score, critiques, risks)
                        _, persp_critiques, _ = eval_func(
                            pseudo_proposal,  # Pass the minimal structure
                            {},  # Empty context for isolated critique
                        )
                    except Exception as e:
                        logger.error(
                            f"Error during {persp.value} critique generation: {e}",
                            exc_info=True,
                        )
                # --- END FIX ---

                # Filter critiques to ensure they relate to the specified aspect
                # (Some generic checks might run even on the minimal proposal)
                related_critiques = [
                    c
                    for c in persp_critiques
                    if c.aspect == aspect or aspect in c.description
                ]
                critiques.extend(related_critiques)

            return critiques

    def suggest_improvements(
        self, proposal: Dict[str, Any], evaluation: Optional[Evaluation] = None
    ) -> List[str]:
        """
        Suggest improvements for proposal

        Args:
            proposal: Proposal to improve
            evaluation: Optional existing evaluation

        Returns:
            List of improvement suggestions
        """
        with self.lock:
            if evaluation is None:
                # Ensure proposal is valid before evaluating
                if not isinstance(proposal, dict):
                    logger.error(
                        f"Invalid proposal type for suggest_improvements: {type(proposal)}"
                    )
                    return ["Invalid proposal type provided."]
                evaluation = self.evaluate_proposal(proposal)

            # Check if evaluation itself is valid
            if not isinstance(evaluation, Evaluation):
                logger.error(
                    f"Invalid evaluation object provided to suggest_improvements: {type(evaluation)}"
                )
                return ["Invalid evaluation object."]

            return evaluation.improvements

    def identify_risks(
        self, proposal: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[Risk]:
        """
        Identify potential risks in proposal

        Args:
            proposal: Proposal to analyze
            context: Optional context

        Returns:
            List of identified risks
        """
        with self.lock:
            # Ensure proposal is a dict
            if not isinstance(proposal, dict):
                logger.error(
                    f"Invalid proposal type for identify_risks: {type(proposal)}"
                )
                return [
                    Risk(
                        category=RiskCategory.OPERATIONAL,
                        severity=RiskSeverity.HIGH,
                        description="Invalid proposal type provided.",
                        likelihood=1.0,
                        impact=0.5,
                    )
                ]

            context = context or {}
            risks = []

            # Safety risks
            safety_risks = self._identify_safety_risks(proposal, context)
            risks.extend(safety_risks)

            # Security risks
            security_risks = self._identify_security_risks(proposal, context)
            risks.extend(security_risks)

            # Performance risks
            performance_risks = self._identify_performance_risks(proposal, context)
            risks.extend(performance_risks)

            # Resource risks
            resource_risks = self._identify_resource_risks(proposal, context)
            risks.extend(resource_risks)

            # Ethical risks (check with ethical boundary monitor)
            if self.ethical_boundary_monitor:
                ethical_risks = self._identify_ethical_risks(proposal, context)
                risks.extend(ethical_risks)

            # Operational risks
            operational_risks = self._identify_operational_risks(proposal, context)
            risks.extend(operational_risks)

            return risks

    # --- START FIX: Add missing risk identification helper methods ---
    # ============================================================
    # Internal Methods - Risk Identification (Placeholders)
    # ============================================================

    def _identify_safety_risks(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Risk]:
        """
        Identify safety risks in proposal.
        
        WARNING: This is a PLACEHOLDER implementation that relies on self-reported flags.
        Real-world usage requires semantic analysis of proposal content, not just checking
        for 'causes_physical_harm' flags. DO NOT rely on this for actual safety decisions
        in production without additional validation layers.
        
        Production requirements:
        - Semantic analysis of proposal text and code
        - Integration with safety verification tools
        - Expert review for safety-critical applications
        - Formal verification methods where applicable
        """
        logger.warning(
            "Using placeholder safety risk detection - not suitable for production safety decisions"
        )
        risks = []
        if proposal.get("causes_physical_harm"):
            risks.append(
                Risk(
                    RiskCategory.SAFETY,
                    RiskSeverity.CRITICAL,
                    "Proposal causes physical harm",
                    1.0,
                    1.0,
                )
            )
        if proposal.get("causes_psychological_harm"):
            risks.append(
                Risk(
                    RiskCategory.SAFETY,
                    RiskSeverity.HIGH,
                    "Proposal causes psychological harm",
                    0.8,
                    0.8,
                )
            )
        return risks

    def _identify_security_risks(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Risk]:
        """
        Identify security risks in proposal.
        
        WARNING: This is a PLACEHOLDER implementation that relies on self-reported flags.
        Real-world usage requires semantic analysis of proposal content, not just checking
        for 'has_security_review' flags. DO NOT rely on this for actual security decisions
        in production without additional validation layers.
        
        Production requirements:
        - Static code analysis and vulnerability scanning
        - Integration with security testing tools (SAST/DAST)
        - Threat modeling and attack surface analysis
        - Security expert review
        """
        logger.warning(
            "Using placeholder security risk detection - not suitable for production security decisions"
        )
        risks = []
        if proposal.get("requires_network_access") and not proposal.get(
            "has_security_review"
        ):
            risks.append(
                Risk(
                    RiskCategory.SECURITY,
                    RiskSeverity.MEDIUM,
                    "Network access required without security review",
                    0.6,
                    0.5,
                )
            )
        return risks

    def _identify_performance_risks(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Risk]:
        """
        Identify performance risks in proposal.
        
        WARNING: This is a PLACEHOLDER implementation that relies on self-reported flags.
        Real-world usage requires semantic analysis and complexity analysis of proposal content,
        not just checking for 'complexity' flags. DO NOT rely on this for actual performance
        predictions in production without additional validation layers.
        
        Production requirements:
        - Algorithmic complexity analysis
        - Performance profiling and benchmarking
        - Load testing and stress testing
        - Resource utilization modeling
        """
        logger.warning(
            "Using placeholder performance risk detection - not suitable for production performance predictions"
        )
        risks = []
        if proposal.get("complexity") == "exponential":
            risks.append(
                Risk(
                    RiskCategory.PERFORMANCE,
                    RiskSeverity.MEDIUM,
                    "Exponential complexity may cause performance degradation",
                    0.7,
                    0.6,
                )
            )
        return risks

    def _identify_resource_risks(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Risk]:
        """
        Identify resource consumption risks in proposal.
        
        WARNING: This is a PLACEHOLDER implementation that relies on self-reported estimates.
        Real-world usage requires semantic analysis and resource modeling of proposal content,
        not just comparing 'estimated_cost' values. DO NOT rely on this for actual resource
        planning in production without additional validation layers.
        
        Production requirements:
        - Detailed resource modeling and forecasting
        - Cost analysis with real infrastructure metrics
        - Capacity planning integration
        - Budget tracking and alerting systems
        """
        logger.warning(
            "Using placeholder resource risk detection - not suitable for production resource planning"
        )
        risks = []
        cost = proposal.get("estimated_cost", 0)
        budget = context.get("budget", 100)
        if cost > budget:
            risks.append(
                Risk(
                    RiskCategory.RESOURCE,
                    RiskSeverity.HIGH,
                    f"Estimated cost {cost} exceeds budget {budget}",
                    0.9,
                    0.7,
                )
            )
        return risks

    def _identify_ethical_risks(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Risk]:
        """
        Identify ethical risks in proposal (integrates with EthicalBoundaryMonitor).
        
        WARNING: This implementation depends on EthicalBoundaryMonitor's effectiveness.
        If the monitor is not properly configured or uses placeholder detection, this
        will not provide adequate ethical risk assessment. DO NOT rely on this for actual
        ethical governance in production without human oversight and review.
        
        Production requirements:
        - Comprehensive ethical framework integration
        - Human ethics committee review
        - Stakeholder impact analysis
        - Cultural and contextual sensitivity analysis
        - Ongoing monitoring and adaptation
        """
        logger.warning(
            "Ethical risk detection depends on EthicalBoundaryMonitor - requires human oversight for production"
        )
        risks = []
        if self.ethical_boundary_monitor:
            try:
                violations = self.ethical_boundary_monitor.detect_boundary_violations(
                    proposal, context
                )
                for v in violations:
                    severity_map = {
                        "critical": RiskSeverity.CRITICAL,
                        "high": RiskSeverity.HIGH,
                        "medium": RiskSeverity.MEDIUM,
                        "low": RiskSeverity.LOW,
                    }
                    # Use .value if it's an enum, otherwise assume string
                    sev_key = (
                        v.severity.value
                        if hasattr(v.severity, "value")
                        else str(v.severity)
                    )
                    severity = severity_map.get(sev_key, RiskSeverity.LOW)
                    risks.append(
                        Risk(
                            RiskCategory.ETHICAL,
                            severity,
                            v.description,
                            likelihood=0.8,
                            impact=0.8,
                            metadata={
                                "boundary_violated": getattr(
                                    v, "boundary_violated", "unknown"
                                )
                            },
                        )
                    )
            except Exception as e:
                logger.debug(f"Ethical boundary monitor check failed: {e}")
        return risks

    def _identify_operational_risks(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Risk]:
        """
        Identify operational risks in proposal.
        
        WARNING: This is a PLACEHOLDER implementation that relies on self-reported flags.
        Real-world usage requires semantic analysis of proposal content and integration
        with operational best practices, not just checking for 'has_rollback_plan' flags.
        DO NOT rely on this for actual operational risk management in production without
        additional validation layers.
        
        Production requirements:
        - Integration with deployment and rollback systems
        - Operational runbook validation
        - Monitoring and alerting verification
        - Incident response planning
        - SRE best practices validation
        """
        logger.warning(
            "Using placeholder operational risk detection - not suitable for production operational planning"
        )
        risks = []
        if not proposal.get("has_rollback_plan"):
            risks.append(
                Risk(
                    RiskCategory.OPERATIONAL,
                    RiskSeverity.MEDIUM,
                    "No rollback strategy defined",
                    0.5,
                    0.6,
                )
            )
        return risks

    # --- END FIX ---

    def compare_alternatives(
        self, proposals: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """
        Compare multiple alternative proposals

        Args:
            proposals: List of proposals to compare
            context: Optional context

        Returns:
            ComparisonResult with ranking and analysis
        """
        with self.lock:
            if not proposals:
                return ComparisonResult(
                    best_proposal_id="",
                    ranking=[],
                    comparison_matrix={},
                    rationale="No proposals provided",
                    trade_offs=[],
                )

            # Ensure all items in proposals list are dicts
            valid_proposals = [p for p in proposals if isinstance(p, dict)]
            if len(valid_proposals) != len(proposals):
                logger.warning(
                    f"Some items in proposals list were not dicts ({len(proposals) - len(valid_proposals)} ignored)."
                )
            if not valid_proposals:
                return ComparisonResult(
                    best_proposal_id="",
                    ranking=[],
                    comparison_matrix={},
                    rationale="No valid proposals provided",
                    trade_offs=[],
                )

            context = context or {}

            # Evaluate each valid proposal
            evaluations = []
            for proposal in valid_proposals:
                eval_result = self.evaluate_proposal(proposal, context)
                evaluations.append(eval_result)

            # Sort by overall score
            sorted_evals = sorted(
                evaluations, key=lambda e: e.overall_score, reverse=True
            )

            # Create ranking
            ranking = [(e.proposal_id, e.overall_score) for e in sorted_evals]

            # Best proposal
            best_proposal_id = sorted_evals[0].proposal_id if sorted_evals else ""

            # Generate comparison matrix (pairwise comparisons)
            comparison_matrix = {}
            for i, eval1 in enumerate(evaluations):
                for eval2 in evaluations[i + 1 :]:
                    # Ensure IDs are strings before using as tuple keys
                    id1 = str(eval1.proposal_id)
                    id2 = str(eval2.proposal_id)
                    key1 = (id1, id2)
                    key2 = (id2, id1)

                    if eval1.overall_score > eval2.overall_score + 0.05:
                        comparison_matrix[key1] = "better"
                        comparison_matrix[key2] = "worse"
                    elif eval2.overall_score > eval1.overall_score + 0.05:
                        comparison_matrix[key1] = "worse"
                        comparison_matrix[key2] = "better"
                    else:
                        comparison_matrix[key1] = "equal"
                        comparison_matrix[key2] = "equal"

            # Identify trade-offs
            trade_offs = self._identify_trade_offs(evaluations)

            # Generate rationale
            rationale = self._generate_comparison_rationale(sorted_evals, trade_offs)

            result = ComparisonResult(
                best_proposal_id=best_proposal_id,
                ranking=ranking,
                comparison_matrix=comparison_matrix,
                rationale=rationale,
                trade_offs=trade_offs,
                metadata={"num_proposals_evaluated": len(valid_proposals)},
            )

            self.stats["comparisons_performed"] += 1

            return result

    def learn_from_outcome(self, proposal_id: str, outcome: Dict[str, Any]):
        """
        Learn from actual outcome to improve future evaluations

        Args:
            proposal_id: ID of evaluated proposal
            outcome: Actual outcome data with success/failure info
        """
        with self.lock:
            # Ensure outcome is a dict
            if not isinstance(outcome, dict):
                logger.warning(
                    f"Invalid outcome type for learn_from_outcome ({type(outcome)}). Cannot learn."
                )
                return

            if proposal_id not in self.evaluations_by_id:
                logger.warning(
                    f"No evaluation found for proposal {proposal_id} to learn from outcome."
                )
                return

            evaluation = self.evaluations_by_id[proposal_id]
            # Determine success (handle boolean or string 'success'/'failure')
            outcome_success_val = outcome.get("success")
            success = False
            if isinstance(outcome_success_val, bool):
                success = outcome_success_val
            elif (
                isinstance(outcome_success_val, str)
                and outcome_success_val.lower() == "success"
            ):
                success = True

            # Update critique effectiveness
            for critique in evaluation.critiques:
                # Ensure critique elements are valid before forming key
                if (
                    critique.perspective
                    and critique.perspective.value
                    and critique.aspect
                ):
                    pattern_key = f"{critique.perspective.value}:{critique.aspect}"
                    current_effectiveness = self.critique_effectiveness.get(
                        pattern_key, 0.5
                    )  # Default neutral

                    if success:
                        # If successful despite critique, maybe critique was too strict or irrelevant
                        # Reduce effectiveness slightly, bounded at 0.1
                        self.critique_effectiveness[pattern_key] = max(
                            0.1, current_effectiveness * 0.95
                        )
                    else:
                        # If failed and critique existed, critique might have been relevant
                        # Increase effectiveness slightly, bounded at 0.9
                        self.critique_effectiveness[pattern_key] = min(
                            0.9, current_effectiveness * 1.05
                        )
                        # Only increment successful pattern if failure occurred and critique existed
                        self.successful_critique_patterns[pattern_key] += 1
                else:
                    logger.debug(
                        f"Skipping critique effectiveness update due to invalid critique data: {critique}"
                    )

            # Adapt perspective weights if enabled
            if self.adaptive_weights:
                self._adapt_perspective_weights(evaluation, success)

            # Record to validation tracker
            if self.validation_tracker:
                try:
                    # Attempt to update the existing record with the actual outcome
                    if hasattr(self.validation_tracker, "update_actual_outcome"):
                        self.validation_tracker.update_actual_outcome(
                            proposal_id, "success" if success else "failure"
                        )
                    else:  # Fallback: record a new event if update isn't available
                        self.validation_tracker.record_validation(
                            proposal={
                                "type": "internal_critique_outcome",
                                "id": proposal_id,
                            },
                            validation_result=evaluation.to_dict(),  # Include original evaluation context
                            actual_outcome="success" if success else "failure",
                        )
                except Exception as e:
                    logger.debug(
                        f"Failed to record/update outcome in validation tracker: {e}"
                    )

            logger.debug(f"Learned from outcome for {proposal_id}: success={success}")

    def get_evaluation_history(
        self, limit: Optional[int] = None, min_score: Optional[float] = None
    ) -> List[Evaluation]:
        """
        Get evaluation history

        Args:
            limit: Optional limit on results
            min_score: Optional minimum score filter

        Returns:
            List of evaluations
        """
        with self.lock:
            # Create snapshot for safe iteration
            evaluations_snapshot = list(self.evaluation_history)

            # Filter by score
            if min_score is not None:
                evaluations_snapshot = [
                    e for e in evaluations_snapshot if e.overall_score >= min_score
                ]

            # Sort by timestamp (most recent first)
            evaluations_snapshot.sort(key=lambda e: e.timestamp, reverse=True)

            # Limit
            if limit:
                evaluations_snapshot = evaluations_snapshot[:limit]

            return evaluations_snapshot

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.lock:
            # Create snapshot for consistent stats
            eval_history_snapshot = list(self.evaluation_history)

            # Count by recommendation
            recommendations = Counter(e.recommendation for e in eval_history_snapshot)

            # Average scores (handle empty history)
            scores = [e.overall_score for e in eval_history_snapshot]
            avg_score = self._np.mean(scores) if scores else 0.0

            # Critique distribution
            critique_levels = defaultdict(int)
            for eval_item in eval_history_snapshot:
                for critique in eval_item.critiques:
                    # Safely access level value
                    level_value = (
                        critique.level.value
                        if hasattr(critique.level, "value")
                        else str(critique.level)
                    )
                    critique_levels[level_value] += 1

            return {
                "evaluations_performed": self.stats["evaluations_performed"],
                "comparisons_performed": self.stats["comparisons_performed"],
                "average_score": float(avg_score),  # Ensure float
                "recommendations": dict(recommendations),
                "critique_levels": dict(critique_levels),
                "critique_patterns_learned": len(self.critique_patterns),
                "successful_patterns": len(self.successful_critique_patterns),
                "evaluation_history_size": len(eval_history_snapshot),
                "perspective_weights": {
                    k.value: v for k, v in self.perspective_weights.items()
                },
                "strict_mode": self.strict_mode,
                "adaptive_weights": self.adaptive_weights,
                "initialized_at": self.stats["initialized_at"],
                "uptime_seconds": time.time() - self.stats["initialized_at"],
            }

    def export_state(self) -> Dict[str, Any]:
        """Export critic state for persistence"""
        with self.lock:
            # Create snapshots for safety
            persp_weights_snapshot = {
                k.value: v for k, v in self.perspective_weights.items()
            }
            crit_eff_snapshot = self.critique_effectiveness.copy()
            succ_patt_snapshot = dict(self.successful_critique_patterns)
            stats_snapshot = dict(self.stats)

            return {
                "perspective_weights": persp_weights_snapshot,
                "critique_effectiveness": crit_eff_snapshot,
                "successful_critique_patterns": succ_patt_snapshot,
                "stats": stats_snapshot,
                "config": {
                    "strict_mode": self.strict_mode,
                    "adaptive_weights": self.adaptive_weights,
                },
                "export_time": time.time(),
                # Note: evaluation_history is not typically persisted due to size and potential unserializable content.
            }

    def import_state(self, state: Dict[str, Any]):
        """Import critic state from persistence"""
        with self.lock:
            # Ensure state is a dict
            if not isinstance(state, dict):
                logger.error(
                    f"Invalid state type provided for import: {type(state)}. Aborting import."
                )
                return

            # Import weights safely
            imported_weights = state.get("perspective_weights", {})
            if isinstance(imported_weights, dict):
                valid_weights = {}
                for k_str, v in imported_weights.items():
                    try:
                        k_enum = EvaluationPerspective(k_str)
                        valid_weights[k_enum] = float(v)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Skipping invalid perspective weight during import: key='{k_str}', value='{v}'"
                        )
                if valid_weights:
                    self.perspective_weights = valid_weights
                    # Re-normalize after import
                    total_w = sum(self.perspective_weights.values())
                    if total_w > 0 and abs(total_w - 1.0) > 1e-6:
                        self.perspective_weights = {
                            k: v / total_w for k, v in self.perspective_weights.items()
                        }
            else:
                logger.warning(
                    f"Invalid 'perspective_weights' format in state: {type(imported_weights)}. Skipping."
                )

            # Import learned effectiveness safely
            imported_effectiveness = state.get("critique_effectiveness", {})
            if isinstance(imported_effectiveness, dict):
                self.critique_effectiveness = {
                    str(k): float(v)
                    for k, v in imported_effectiveness.items()
                    if isinstance(k, str) and isinstance(v, (int, float))
                }
            else:
                logger.warning(
                    f"Invalid 'critique_effectiveness' format in state: {type(imported_effectiveness)}. Skipping."
                )

            # Import successful patterns safely
            imported_patterns = state.get("successful_critique_patterns", {})
            if isinstance(imported_patterns, dict):
                self.successful_critique_patterns = defaultdict(
                    int,
                    {
                        str(k): int(v)
                        for k, v in imported_patterns.items()
                        if isinstance(k, str) and isinstance(v, int)
                    },
                )
            else:
                logger.warning(
                    f"Invalid 'successful_critique_patterns' format in state: {type(imported_patterns)}. Skipping."
                )

            # Import stats (merge?)
            imported_stats = state.get("stats", {})
            if isinstance(imported_stats, dict):
                self.stats.update(imported_stats)  # Simple merge

            # Import config
            config_state = state.get("config", {})
            if isinstance(config_state, dict):
                self.strict_mode = config_state.get("strict_mode", self.strict_mode)
                self.adaptive_weights = config_state.get(
                    "adaptive_weights", self.adaptive_weights
                )

            logger.info("Imported state from persistence")

    def reset(self) -> None:
        """Reset all evaluations and statistics"""
        with self.lock:
            self.evaluation_history.clear()
            self.evaluations_by_id.clear()
            self.critique_patterns.clear()
            self.successful_critique_patterns.clear()
            self.risk_patterns.clear()
            self.critique_effectiveness.clear()

            # Reset stats but keep init time
            init_time = self.stats.get("initialized_at", time.time())
            self.stats.clear()
            self.stats["initialized_at"] = init_time

            logger.info(
                "InternalCritic reset - evaluation history and learned patterns cleared"
            )

    # ============================================================
    # Internal Methods - Evaluation (Placeholders - Implement specific logic here)
    # ============================================================

    # NOTE: The _evaluate_* methods below are simplified placeholders.
    # A full implementation would involve complex logic, potentially calling
    # external tools, models, or simulation engines based on the perspective.

    def _evaluate_logical_consistency(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate logical consistency"""
        score = 0.8  # Default
        critiques = []
        risks = []

        # Placeholder: Check for simple contradictions or missing info
        if proposal.get("contradicts_known_fact"):
            score -= 0.5
            critiques.append(
                Critique(
                    CritiqueLevel.MAJOR,
                    EvaluationPerspective.LOGICAL_CONSISTENCY,
                    "contradiction",
                    "Proposal contradicts established facts.",
                )
            )
        if not proposal.get("rationale"):
            score -= 0.1
            critiques.append(
                Critique(
                    CritiqueLevel.MINOR,
                    EvaluationPerspective.LOGICAL_CONSISTENCY,
                    "rationale",
                    "Rationale is missing or unclear.",
                )
            )

        # Check for missing required fields (Example)
        required_fields = ["description", "goal"]  # Adjust as needed
        missing = [f for f in required_fields if f not in proposal]
        if missing:
            score -= 0.1 * len(missing)
            critiques.append(
                Critique(
                    level=CritiqueLevel.MINOR,
                    perspective=EvaluationPerspective.LOGICAL_CONSISTENCY,
                    aspect="completeness",
                    description=f"Missing required proposal fields: {', '.join(missing)}",
                    suggested_improvement=f"Add missing fields: {', '.join(missing)}",
                    confidence=0.9,
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    def _evaluate_feasibility(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate practical feasibility"""
        score = 0.7
        critiques = []
        risks = []

        # Placeholder: Check resources and time
        cost = proposal.get("estimated_cost", 0)
        budget = context.get("budget", 100)  # Example budget
        if cost > budget:
            score -= 0.4
            critiques.append(
                Critique(
                    CritiqueLevel.CRITICAL,
                    EvaluationPerspective.FEASIBILITY,
                    "budget",
                    f"Cost {cost} exceeds budget {budget}.",
                )
            )
            risks.append(
                Risk(
                    RiskCategory.RESOURCE, RiskSeverity.HIGH, "Budget overrun", 0.9, 0.7
                )
            )

        duration = proposal.get("estimated_duration", 1)
        max_duration = context.get("max_duration", 10)  # Example max duration
        if duration > max_duration:
            score -= 0.2
            critiques.append(
                Critique(
                    CritiqueLevel.MAJOR,
                    EvaluationPerspective.FEASIBILITY,
                    "duration",
                    f"Duration {duration} exceeds limit {max_duration}.",
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    def _evaluate_safety(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate safety"""
        score = 0.9
        critiques = []
        # Use identify_risks to populate safety/ethical risks
        # --- FIX: Call helper methods directly, not the public identify_risks ---
        risks = []
        safety_risks = self._identify_safety_risks(proposal, context)
        ethical_risks = self._identify_ethical_risks(proposal, context)
        risks.extend(safety_risks)
        risks.extend(ethical_risks)
        # --- END FIX ---

        if any(r.severity == RiskSeverity.CRITICAL for r in safety_risks):
            score = 0.0
            critiques.append(
                Critique(
                    CritiqueLevel.CRITICAL,
                    EvaluationPerspective.SAFETY,
                    "critical_safety_risk",
                    "Proposal poses critical safety risk.",
                    impact_if_ignored=1.0,
                )
            )
        elif any(r.severity == RiskSeverity.HIGH for r in safety_risks):
            score -= 0.4
            critiques.append(
                Critique(
                    CritiqueLevel.MAJOR,
                    EvaluationPerspective.SAFETY,
                    "high_safety_risk",
                    "Proposal poses high safety risk.",
                )
            )

        if any(r.severity == RiskSeverity.CRITICAL for r in ethical_risks):
            score -= 0.5  # Penalize heavily for critical ethical risks too
            critiques.append(
                Critique(
                    CritiqueLevel.CRITICAL,
                    EvaluationPerspective.SAFETY,
                    "critical_ethical_risk",
                    "Proposal poses critical ethical risk.",
                )
            )
        elif any(r.severity == RiskSeverity.HIGH for r in ethical_risks):
            score -= 0.3
            critiques.append(
                Critique(
                    CritiqueLevel.MAJOR,
                    EvaluationPerspective.SAFETY,
                    "high_ethical_risk",
                    "Proposal poses high ethical risk.",
                )
            )

        # Check with ethical boundary monitor if available (redundant if identify_risks uses it, but safe)
        if self.ethical_boundary_monitor:
            try:
                # Use detect_boundary_violations for evaluation without enforcement side-effects
                violations = self.ethical_boundary_monitor.detect_boundary_violations(
                    proposal, context
                )
                if violations:
                    # Penalize score based on highest severity violation found
                    highest_severity_num = (
                        max(self._severity_to_numeric(v.severity) for v in violations)
                        if violations
                        else 0
                    )
                    if highest_severity_num >= 4:  # Critical
                        score = 0.0
                    elif highest_severity_num == 3:  # High
                        score = max(0.0, score - 0.5)
                    elif highest_severity_num == 2:  # Medium
                        score = max(0.0, score - 0.2)

                    # Add critiques based on violations (avoid duplicating risks already added)
                    added_boundaries = set(
                        r.metadata.get("boundary_violated") for r in ethical_risks
                    )
                    for violation in violations:
                        if violation.boundary_violated not in added_boundaries:
                            # --- FIX: Need _severity_to_numeric helper ---
                            sev_val = (
                                violation.severity.value
                                if hasattr(violation.severity, "value")
                                else str(violation.severity)
                            )
                            level = (
                                CritiqueLevel.CRITICAL
                                if sev_val in ["critical", "high"]
                                else CritiqueLevel.MAJOR
                            )
                            critiques.append(
                                Critique(
                                    level=level,
                                    perspective=EvaluationPerspective.SAFETY,  # Or map category?
                                    aspect="ethical_boundary",
                                    description=f"Ethical violation: {violation.description}",
                                    confidence=0.9,
                                    metadata={"boundary": violation.boundary_violated},
                                )
                            )
            except Exception as e:
                logger.debug(
                    f"Failed to check ethical boundaries during safety eval: {e}"
                )

        return (
            max(0.0, min(1.0, score)),
            critiques,
            risks,
        )  # Return all identified risks

    # --- START FIX: Add missing _severity_to_numeric helper ---
    def _severity_to_numeric(self, severity: Any) -> int:
        """Helper to convert severity object/string to numeric level."""
        sev_val = severity.value if hasattr(severity, "value") else str(severity)
        sev_map = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "negligible": 0,
        }
        return sev_map.get(sev_val.lower(), 0)

    # --- END FIX ---

    def _evaluate_alignment(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate goal and value alignment"""
        score = 0.7
        critiques = []
        risks = []

        # Placeholder: Check if proposal objective matches context goal
        prop_obj = proposal.get("goal") or proposal.get("objective")
        ctx_goal = context.get("system_goal")
        if prop_obj and ctx_goal and prop_obj != ctx_goal:
            score -= 0.3
            critiques.append(
                Critique(
                    CritiqueLevel.MAJOR,
                    EvaluationPerspective.ALIGNMENT,
                    "goal_mismatch",
                    f"Proposal goal '{prop_obj}' differs from system goal '{ctx_goal}'.",
                )
            )
            risks.append(
                Risk(
                    RiskCategory.OPERATIONAL,
                    RiskSeverity.MEDIUM,
                    "Goal misalignment",
                    0.6,
                    0.6,
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    def _evaluate_efficiency(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate resource efficiency"""
        score = 0.8
        critiques = []
        risks = []

        # Placeholder: Check complexity and resource tags
        complexity = proposal.get("complexity", "unknown")
        if complexity == "high" or complexity == "exponential":
            score -= 0.3
            critiques.append(
                Critique(
                    CritiqueLevel.MINOR,
                    EvaluationPerspective.EFFICIENCY,
                    "high_complexity",
                    f"Complexity '{complexity}' may impact performance.",
                )
            )
            risks.append(
                Risk(
                    RiskCategory.PERFORMANCE,
                    RiskSeverity.LOW,
                    "Potential performance issue due to complexity",
                    0.4,
                    0.3,
                )
            )

        resources = proposal.get("resources_needed", [])
        if len(resources) > 5:  # Arbitrary limit
            score -= 0.1
            critiques.append(
                Critique(
                    CritiqueLevel.SUGGESTION,
                    EvaluationPerspective.EFFICIENCY,
                    "resource_count",
                    "High number of resources required.",
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    def _evaluate_completeness(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate completeness and coverage"""
        score = 0.7
        critiques = []
        risks = []

        # Placeholder: Check for key sections
        expected = ["description", "steps", "validation"]
        missing = [k for k in expected if not proposal.get(k)]
        if missing:
            score -= 0.2 * len(missing)
            critiques.append(
                Critique(
                    CritiqueLevel.MINOR,
                    EvaluationPerspective.COMPLETENESS,
                    "missing_sections",
                    f"Missing sections: {', '.join(missing)}",
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    def _evaluate_clarity(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate clarity and explainability"""
        score = 0.8
        critiques = []
        risks = []

        # Placeholder: Check description length and jargon
        desc = proposal.get("description", "")
        if len(desc) < 10:
            score -= 0.4
            critiques.append(
                Critique(
                    CritiqueLevel.MINOR,
                    EvaluationPerspective.CLARITY,
                    "description_length",
                    "Description is too short.",
                )
            )
        elif "synergize" in desc or "paradigm shift" in desc:  # Example jargon check
            score -= 0.1
            critiques.append(
                Critique(
                    CritiqueLevel.SUGGESTION,
                    EvaluationPerspective.CLARITY,
                    "jargon",
                    "Description may contain jargon.",
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    def _evaluate_robustness(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[Critique], List[Risk]]:
        """Evaluate robustness to edge cases"""
        score = 0.7
        critiques = []
        risks = []

        # Placeholder: Check for explicit handling
        if not proposal.get("error_handling"):
            score -= 0.2
            critiques.append(
                Critique(
                    CritiqueLevel.MINOR,
                    EvaluationPerspective.ROBUSTNESS,
                    "error_handling",
                    "No explicit error handling described.",
                )
            )
            risks.append(
                Risk(
                    RiskCategory.OPERATIONAL,
                    RiskSeverity.LOW,
                    "Potential failure on errors",
                    0.5,
                    0.4,
                )
            )
        if not proposal.get("edge_cases_considered"):
            score -= 0.1
            critiques.append(
                Critique(
                    CritiqueLevel.SUGGESTION,
                    EvaluationPerspective.ROBUSTNESS,
                    "edge_cases",
                    "Edge cases may not be fully considered.",
                )
            )

        return (max(0.0, min(1.0, score)), critiques, risks)

    # ============================================================
    # Internal Methods - Analysis and Generation
    # ============================================================

    def _generate_rationale(
        self,
        perspective: EvaluationPerspective,
        score: float,
        critiques: List[Critique],
    ) -> str:
        """Generate rationale for perspective score"""
        if score >= 0.8:
            quality = "excellent"
        elif score >= 0.6:
            quality = "good"
        elif score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"

        critique_summary = (
            f"{len(critiques)} issues identified"
            if critiques
            else "no major issues identified"
        )
        critical_count = sum(1 for c in critiques if c.level == CritiqueLevel.CRITICAL)
        if critical_count > 0:
            critique_summary += f" ({critical_count} critical)"

        return f"{perspective.value.replace('_', ' ').title()}: {quality} ({score:.2f}). {critique_summary}."

    def _identify_strengths(
        self, perspective_scores: List[PerspectiveScore], proposal: Dict[str, Any]
    ) -> List[str]:
        """Identify proposal strengths"""
        strengths = []

        # High-scoring perspectives
        for ps in perspective_scores:
            if ps.score >= 0.8:
                strengths.append(
                    f"Strong {ps.perspective.value.replace('_', ' ').lower()}"
                )

        # Explicit positive attributes from proposal (examples)
        if proposal.get("well_tested"):
            strengths.append("Well-tested")
        if proposal.get("includes_documentation"):
            strengths.append("Includes documentation")
        if proposal.get("simple_design"):
            strengths.append("Simple design")

        return strengths if strengths else ["Basic proposal structure is valid."]

    def _identify_weaknesses(
        self, perspective_scores: List[PerspectiveScore], critiques: List[Critique]
    ) -> List[str]:
        """Identify proposal weaknesses"""
        weaknesses = []

        # Low-scoring perspectives
        for ps in perspective_scores:
            if ps.score < 0.5:
                weaknesses.append(
                    f"Weak {ps.perspective.value.replace('_', ' ').lower()} (score: {ps.score:.2f})"
                )

        # Critical/major critiques
        critical_critiques = [
            c
            for c in critiques
            if c.level in [CritiqueLevel.CRITICAL, CritiqueLevel.MAJOR]
        ]
        # Summarize top critiques concisely
        for critique in sorted(
            critical_critiques,
            key=lambda c: (c.level.value != "critical", c.impact_if_ignored),
            reverse=True,
        )[:3]:
            weaknesses.append(
                f"{critique.level.value.capitalize()} issue in {critique.perspective.value.replace('_', ' ')}: {critique.description}"
            )

        return weaknesses if weaknesses else ["No major weaknesses identified."]

    def _generate_improvements(
        self, critiques: List[Critique], risks: List[Risk], proposal: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions"""
        improvements = set()  # Use set for automatic deduplication

        # From critiques with suggestions
        for critique in critiques:
            if critique.suggested_improvement:
                improvements.add(critique.suggested_improvement)

        # From risks with mitigations
        for risk in sorted(
            risks, key=lambda r: r.risk_score, reverse=True
        ):  # Prioritize higher risks
            if risk.mitigation_strategies:
                # Add prefix to clarify context
                improvements.add(
                    f"Mitigate risk ({risk.category.value} - {risk.severity.value}): {risk.mitigation_strategies[0]}"
                )
                if len(risk.mitigation_strategies) > 1:
                    improvements.add(f"Also consider: {risk.mitigation_strategies[1]}")

        # Generic improvements based on missing elements
        if not proposal.get("validation_plan"):
            improvements.add("Add a validation plan.")
        if not proposal.get("rollback_strategy"):
            improvements.add("Define a rollback strategy.")

        # Limit and convert back to list
        return list(improvements)[:10]  # Top 10 unique suggestions

    def _generate_overall_assessment(
        self,
        overall_score: float,
        perspective_scores: List[PerspectiveScore],
        critiques: List[Critique],
        risks: List[Risk],
    ) -> str:
        """Generate overall assessment text"""
        if overall_score >= 0.8:
            quality = "Excellent"
        elif overall_score >= 0.6:
            quality = "Good"
        elif overall_score >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"

        critical_issues = sum(1 for c in critiques if c.level == CritiqueLevel.CRITICAL)
        major_issues = sum(1 for c in critiques if c.level == CritiqueLevel.MAJOR)
        high_risks = sum(
            1 for r in risks if r.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]
        )

        assessment = f"Overall assessment: {quality} (Score: {overall_score:.2f}). "

        issues_summary = []
        if critical_issues > 0:
            issues_summary.append(f"{critical_issues} critical issue(s)")
        if major_issues > 0:
            issues_summary.append(f"{major_issues} major issue(s)")
        if high_risks > 0:
            issues_summary.append(f"{high_risks} high/critical risk(s)")

        if issues_summary:
            assessment += "Issues found: " + ", ".join(issues_summary) + ". "
        else:
            assessment += "No critical issues or high risks identified. "

        # Add recommendation context
        if critical_issues > 0 or any(
            r.severity == RiskSeverity.CRITICAL for r in risks
        ):
            assessment += "Requires substantial revision or rejection."
        elif major_issues > 0 or high_risks > 0 or overall_score < 0.5:
            assessment += "Significant modifications recommended before proceeding."
        elif overall_score < 0.7:
            assessment += "Minor improvements suggested."
        else:
            assessment += "Generally suitable for implementation, potentially with minor refinements."

        return assessment

    def _generate_recommendation(
        self, overall_score: float, critiques: List[Critique], risks: List[Risk]
    ) -> str:
        """Generate recommendation (approve/modify/reject)"""
        critical_issues = any(c.level == CritiqueLevel.CRITICAL for c in critiques)
        major_issues = any(c.level == CritiqueLevel.MAJOR for c in critiques)
        critical_risks = any(r.severity == RiskSeverity.CRITICAL for r in risks)
        high_risks = any(r.severity == RiskSeverity.HIGH for r in risks)

        # Stricter rejection criteria
        if critical_issues or critical_risks:
            return "reject"
        elif (
            major_issues or high_risks or overall_score < 0.5
        ):  # Include score threshold
            return "modify"
        else:  # Score >= 0.5 and no critical/major issues or high/critical risks
            return "approve"

    def _compute_evaluation_confidence(
        self, perspective_scores: List[PerspectiveScore], critiques: List[Critique]
    ) -> float:
        """Compute confidence in evaluation"""
        # Use fake numpy if needed
        _np = np if NUMPY_AVAILABLE else FakeNumpy

        # Average confidence from perspective scores (handle empty list)
        confidences = [
            ps.confidence for ps in perspective_scores if ps.confidence is not None
        ]
        avg_confidence = (
            _np.mean(confidences) if confidences else 0.7
        )  # Default confidence if no scores

        # Adjust based on number and severity of critiques (more severe critiques might indicate higher confidence in finding flaws)
        critique_count = len(critiques)
        critical_count = sum(1 for c in critiques if c.level == CritiqueLevel.CRITICAL)
        # Increase confidence slightly with more data, more so if critical issues found
        data_factor = min(1.2, 1.0 + (critique_count / 20.0) + (critical_count * 0.1))

        # Final confidence bounded between 0.1 and 0.95
        final_confidence = avg_confidence * data_factor
        return float(max(0.1, min(0.95, final_confidence)))

    def _identify_trade_offs(self, evaluations: List[Evaluation]) -> List[str]:
        """Identify trade-offs between proposals by comparing perspective scores"""
        trade_offs = []

        if len(evaluations) < 2:
            return trade_offs

        # Get all unique proposal IDs
        proposal_ids = [e.proposal_id for e in evaluations]

        # Iterate through each perspective
        for perspective in EvaluationPerspective:
            perspective_name = perspective.value.replace("_", " ").title()
            scores_for_perspective = []
            # Collect scores for this perspective from all evaluations
            for eval_item in evaluations:
                ps = next(
                    (
                        ps
                        for ps in eval_item.perspective_scores
                        if ps.perspective == perspective
                    ),
                    None,
                )
                if ps:
                    scores_for_perspective.append(
                        {"id": eval_item.proposal_id, "score": ps.score}
                    )

            # Find min and max scores for this perspective if available
            if len(scores_for_perspective) >= 2:
                min_score_item = min(scores_for_perspective, key=lambda x: x["score"])
                max_score_item = max(scores_for_perspective, key=lambda x: x["score"])

                # If there's a significant difference, note it as a trade-off
                if (
                    max_score_item["score"] - min_score_item["score"] > 0.3
                ):  # Threshold for significant difference
                    trade_offs.append(
                        f"{perspective_name}: '{max_score_item['id']}' excels ({max_score_item['score']:.2f}) while '{min_score_item['id']}' lags ({min_score_item['score']:.2f})."
                    )

        return trade_offs

    def _generate_comparison_rationale(
        self, sorted_evaluations: List[Evaluation], trade_offs: List[str]
    ) -> str:
        """Generate rationale for comparison"""
        if not sorted_evaluations:
            return "No valid proposals were evaluated for comparison."

        best = sorted_evaluations[0]
        rationale = f"Proposal '{best.proposal_id}' is ranked highest with an overall score of {best.overall_score:.2f}. "

        if len(sorted_evaluations) > 1:
            second = sorted_evaluations[1]
            margin = best.overall_score - second.overall_score
            rationale += f"It outperforms the next best, '{second.proposal_id}' (score: {second.overall_score:.2f}), by {margin:.2f}. "
        else:
            rationale += "It was the only proposal evaluated. "

        # Summarize key strengths of the best proposal
        if best.strengths:
            rationale += f"Key strengths include: {'; '.join(best.strengths[:2])}. "  # Top 2 strengths

        # Summarize key trade-offs if they exist
        if trade_offs:
            rationale += f"However, consider the trade-offs: {trade_offs[0]}"  # Mention the most significant trade-off
            if len(trade_offs) > 1:
                rationale += " among others."
        else:
            rationale += "No significant trade-offs were identified between proposals based on evaluated perspectives."

        return rationale

    def _update_patterns(self, critiques: List[Critique], risks: List[Risk]):
        """Update learned patterns based on critique and risk frequency"""
        for critique in critiques:
            # Ensure valid key components
            if critique.perspective and critique.perspective.value and critique.aspect:
                pattern_key = f"{critique.perspective.value}:{critique.aspect}"
                self.critique_patterns[pattern_key] += 1
            else:
                logger.debug(
                    f"Skipping pattern update for invalid critique: {critique}"
                )

        for risk in risks:
            # Ensure valid category and description
            if risk.category and risk.category.value and risk.description:
                # Store descriptions associated with risk categories
                # Limit stored descriptions per category to avoid excessive memory use
                if len(self.risk_patterns[risk.category]) < 100:
                    # Avoid adding duplicate descriptions frequently
                    if (
                        risk.description not in self.risk_patterns[risk.category][-10:]
                    ):  # Check last 10
                        self.risk_patterns[risk.category].append(risk.description)
            else:
                logger.debug(f"Skipping risk pattern update for invalid risk: {risk}")

    def _adapt_perspective_weights(self, evaluation: Evaluation, success: bool):
        """Adapt perspective weights based on outcomes - increase weight of perspectives that caught issues in failures"""
        if not self.adaptive_weights:
            return

        learning_rate = 0.01  # Small learning rate

        if not success:
            # Identify perspectives associated with critical/major critiques in the failed evaluation
            relevant_perspectives = set()
            for critique in evaluation.critiques:
                if critique.level in [CritiqueLevel.CRITICAL, CritiqueLevel.MAJOR]:
                    relevant_perspectives.add(critique.perspective)

            if relevant_perspectives:
                logger.debug(
                    f"Adapting weights due to failure. Increasing importance of: {relevant_perspectives}"
                )
                increase_factor = learning_rate * len(relevant_perspectives)

                # Increase weights for relevant perspectives, decrease others slightly
                total_weight = 0.0
                for p, w in self.perspective_weights.items():
                    if p in relevant_perspectives:
                        self.perspective_weights[p] = min(
                            0.5, w * (1 + learning_rate)
                        )  # Increase, cap at 0.5
                    else:
                        self.perspective_weights[p] = max(
                            0.01,
                            w
                            * (
                                1
                                - increase_factor
                                / (
                                    len(self.perspective_weights)
                                    - len(relevant_perspectives)
                                    + 1e-6
                                )
                            ),
                        )  # Decrease slightly, floor at 0.01
                    total_weight += self.perspective_weights[p]

                # Renormalize to ensure weights sum to 1
                if total_weight > 0:
                    self.perspective_weights = {
                        k: v / total_weight for k, v in self.perspective_weights.items()
                    }

                logger.debug(
                    f"New perspective weights: {{p.value: w for p, w in self.perspective_weights.items()}}"
                )

        # Optional: Slightly decrease weights of perspectives that gave high scores to a failed proposal? (More complex logic)

    def _initialize_default_criteria(self):
        """Initialize default evaluation criteria (linking perspectives to methods)"""
        # This method is primarily for organization or potentially registering
        # more complex, pluggable evaluation functions later.
        # The core logic is currently within the _evaluate_* methods themselves.
        self.evaluation_criteria = {
            EvaluationPerspective.LOGICAL_CONSISTENCY: [
                self._evaluate_logical_consistency
            ],
            EvaluationPerspective.FEASIBILITY: [self._evaluate_feasibility],
            EvaluationPerspective.SAFETY: [self._evaluate_safety],
            EvaluationPerspective.ALIGNMENT: [self._evaluate_alignment],
            EvaluationPerspective.EFFICIENCY: [self._evaluate_efficiency],
            EvaluationPerspective.COMPLETENESS: [self._evaluate_completeness],
            EvaluationPerspective.CLARITY: [self._evaluate_clarity],
            EvaluationPerspective.ROBUSTNESS: [self._evaluate_robustness],
        }
        pass


# Module-level exports
__all__ = [
    "InternalCritic",
    "Critique",
    "Evaluation",
    "Risk",
    "ComparisonResult",
    "PerspectiveScore",
    "CritiqueLevel",
    "EvaluationPerspective",
    "RiskCategory",
    "RiskSeverity",
]
