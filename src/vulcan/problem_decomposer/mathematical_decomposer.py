"""
Enhanced Mathematical Decomposer for VULCAN-AGI.

Integrates the MathematicalVerificationEngine with the ProblemDecomposer
to provide SOTA mathematical verification capabilities during problem
decomposition and solution execution.

Features:
- Automatic mathematical validation of decomposition steps
- Bayesian reasoning verification during inference
- Detection of common mathematical errors (specificity confusion, etc.)
- Learning from mathematical errors
- Integration with the unified reasoning system
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import mathematical verification
try:
    from ..reasoning.mathematical_verification import (
        BayesianProblem,
        MathematicalToolOrchestrator,
        MathematicalVerificationEngine,
        MathErrorType,
        MathProblem,
        MathVerificationStatus,
        VerificationResult,
    )

    MATH_VERIFICATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mathematical verification not available: {e}")
    MATH_VERIFICATION_AVAILABLE = False
    MathematicalVerificationEngine = None
    MathematicalToolOrchestrator = None
    MathErrorType = None
    MathVerificationStatus = None
    BayesianProblem = None
    MathProblem = None
    VerificationResult = None

# Import problem decomposer components
try:
    from .problem_decomposer_core import (
        DecompositionPlan,
        ExecutionOutcome,
        ProblemDecomposer,
        ProblemGraph,
    )

    DECOMPOSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Problem decomposer not available: {e}")
    DECOMPOSER_AVAILABLE = False
    ProblemDecomposer = None
    ProblemGraph = None
    DecompositionPlan = None
    ExecutionOutcome = None


# ============================================================================
# MATHEMATICAL PROBLEM DETECTION
# ============================================================================


@dataclass
class MathematicalProblemContext:
    """Context for mathematical problems within a decomposition."""

    is_mathematical: bool = False
    problem_type: str = "unknown"
    has_probability: bool = False
    has_bayesian: bool = False
    has_statistics: bool = False
    has_arithmetic: bool = False
    variables: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


def detect_mathematical_problem(problem_graph: "ProblemGraph") -> MathematicalProblemContext:
    """
    Detect if a problem involves mathematical reasoning.
    
    Analyzes the problem graph to identify:
    - Probability calculations
    - Bayesian inference
    - Statistical analysis
    - Arithmetic operations
    
    Args:
        problem_graph: The problem to analyze
        
    Returns:
        MathematicalProblemContext with detection results
    """
    context = MathematicalProblemContext()
    
    # Check metadata
    metadata = problem_graph.metadata if hasattr(problem_graph, 'metadata') else {}
    domain = metadata.get("domain", "").lower()
    problem_type = metadata.get("type", "").lower()
    description = metadata.get("description", "").lower()
    
    # Mathematical domain keywords
    math_domains = [
        "math", "mathematics", "statistical", "probability", "bayesian",
        "inference", "calculation", "numeric", "quantitative",
    ]
    
    # Check domain
    for md in math_domains:
        if md in domain:
            context.is_mathematical = True
            context.problem_type = md
            break
    
    # Probability keywords
    prob_keywords = [
        "probability", "prior", "posterior", "likelihood", "p(",
        "conditional", "bayes", "frequentist", "distribution",
    ]
    
    for kw in prob_keywords:
        if kw in description or kw in problem_type:
            context.has_probability = True
            context.is_mathematical = True
            break
    
    # Bayesian keywords
    bayesian_keywords = [
        "bayes", "bayesian", "prior", "posterior", "sensitivity",
        "specificity", "false positive", "false negative",
        "true positive", "true negative", "prevalence",
        "positive predictive", "negative predictive",
    ]
    
    for kw in bayesian_keywords:
        if kw in description or kw in problem_type:
            context.has_bayesian = True
            context.has_probability = True
            context.is_mathematical = True
            break
    
    # Statistical keywords
    stat_keywords = [
        "mean", "variance", "standard deviation", "confidence interval",
        "hypothesis", "p-value", "significance", "correlation",
    ]
    
    for kw in stat_keywords:
        if kw in description or kw in problem_type:
            context.has_statistics = True
            context.is_mathematical = True
            break
    
    # Extract variables from metadata
    context.variables = metadata.get("variables", {})
    context.constraints = metadata.get("constraints", [])
    
    # Check nodes for mathematical content
    if hasattr(problem_graph, 'nodes'):
        for node_id, node_data in problem_graph.nodes.items():
            if isinstance(node_data, dict):
                node_type = node_data.get("type", "").lower()
                if any(kw in node_type for kw in ["compute", "calculate", "math"]):
                    context.is_mathematical = True
                    context.has_arithmetic = True
    
    return context


# ============================================================================
# ENHANCED MATHEMATICAL DECOMPOSER
# ============================================================================


class EnhancedMathematicalDecomposer:
    """
    Enhanced Problem Decomposer with SOTA Mathematical Verification.
    
    Wraps the existing ProblemDecomposer and adds mathematical verification
    capabilities during decomposition and execution.
    
    Key capabilities:
    - Detects mathematical problems automatically
    - Verifies Bayesian calculations
    - Detects specificity/sensitivity confusion
    - Validates probability distributions
    - Learns from mathematical errors
    
    Example:
        >>> decomposer = EnhancedMathematicalDecomposer()
        >>> plan, outcome = decomposer.decompose_and_execute_with_verification(
        ...     problem_graph,
        ...     verify_math=True
        ... )
    """

    def __init__(
        self,
        semantic_bridge=None,
        vulcan_memory=None,
        validator=None,
        safety_config: Optional[Dict[str, Any]] = None,
        safety_validator=None,
        math_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize enhanced mathematical decomposer.
        
        Args:
            semantic_bridge: Semantic bridge component
            vulcan_memory: VULCAN memory system
            validator: Optional validator for solution validation
            safety_config: Safety configuration
            safety_validator: Safety validator instance
            math_config: Mathematical verification configuration
        """
        # Initialize base decomposer if available
        if DECOMPOSER_AVAILABLE and ProblemDecomposer is not None:
            self.base_decomposer = ProblemDecomposer(
                semantic_bridge=semantic_bridge,
                vulcan_memory=vulcan_memory,
                validator=validator,
                safety_config=safety_config,
                safety_validator=safety_validator,
            )
        else:
            self.base_decomposer = None
            logger.warning("Base ProblemDecomposer not available")
        
        # Initialize mathematical verification
        self.math_config = math_config or {}
        if MATH_VERIFICATION_AVAILABLE:
            self.math_engine = MathematicalVerificationEngine(
                config=self.math_config
            )
            self.tool_orchestrator = MathematicalToolOrchestrator()
        else:
            self.math_engine = None
            self.tool_orchestrator = None
            logger.warning("Mathematical verification engine not available")
        
        # Tracking
        self.math_verification_history = deque(maxlen=1000)
        self.math_error_counts = defaultdict(int)
        self.math_correction_counts = defaultdict(int)
        
        logger.info("EnhancedMathematicalDecomposer initialized")

    def decompose_and_execute_with_verification(
        self,
        problem_graph: "ProblemGraph",
        verify_math: bool = True,
        validate_solution: bool = True,
    ) -> Tuple["DecompositionPlan", "ExecutionOutcome"]:
        """
        Decompose and execute with mathematical verification.
        
        Extends the base decompose_and_execute with:
        - Mathematical problem detection
        - Bayesian calculation verification
        - Probability validation
        - Error detection and correction
        
        Args:
            problem_graph: Problem to decompose and solve
            verify_math: Whether to perform mathematical verification
            validate_solution: Whether to validate the solution
            
        Returns:
            Tuple of (decomposition_plan, execution_outcome)
        """
        if self.base_decomposer is None:
            logger.error("Base decomposer not available")
            return self._create_error_result("Base decomposer not available")
        
        # Detect if mathematical problem
        math_context = detect_mathematical_problem(problem_graph)
        
        # Execute base decomposition
        plan, outcome = self.base_decomposer.decompose_and_execute(
            problem_graph,
            validate=validate_solution,
        )
        
        # If mathematical and verification enabled, verify the result
        if verify_math and math_context.is_mathematical and self.math_engine:
            outcome = self._verify_mathematical_outcome(
                problem_graph, plan, outcome, math_context
            )
        
        return plan, outcome

    def _verify_mathematical_outcome(
        self,
        problem_graph: "ProblemGraph",
        plan: "DecompositionPlan",
        outcome: "ExecutionOutcome",
        math_context: MathematicalProblemContext,
    ) -> "ExecutionOutcome":
        """
        Verify mathematical aspects of an execution outcome.
        
        Performs:
        - Bayesian calculation verification
        - Probability distribution validation
        - Arithmetic verification
        - Error detection and correction
        """
        if not self.math_engine:
            return outcome
        
        verification_results = []
        
        # Check for Bayesian problems
        if math_context.has_bayesian:
            bayesian_result = self._verify_bayesian_content(
                problem_graph, outcome, math_context
            )
            if bayesian_result:
                verification_results.append(bayesian_result)
        
        # Check probability distributions in solution
        if math_context.has_probability:
            prob_result = self._verify_probability_content(outcome)
            if prob_result:
                verification_results.append(prob_result)
        
        # Check arithmetic in solution
        if math_context.has_arithmetic:
            arith_result = self._verify_arithmetic_content(outcome)
            if arith_result:
                verification_results.append(arith_result)
        
        # Process verification results
        if verification_results:
            outcome = self._apply_verification_results(
                outcome, verification_results
            )
        
        return outcome

    def _verify_bayesian_content(
        self,
        problem_graph: "ProblemGraph",
        outcome: "ExecutionOutcome",
        math_context: MathematicalProblemContext,
    ) -> Optional[VerificationResult]:
        """Verify Bayesian calculations in the outcome."""
        if not self.math_engine or not outcome.solution:
            return None
        
        solution = outcome.solution
        metadata = problem_graph.metadata if hasattr(problem_graph, 'metadata') else {}
        
        # Try to extract Bayesian parameters
        try:
            # Check if solution has probability values
            if isinstance(solution, dict):
                posterior = solution.get("posterior")
                prior = solution.get("prior", metadata.get("prior"))
                sensitivity = solution.get("sensitivity", metadata.get("sensitivity"))
                specificity = solution.get("specificity", metadata.get("specificity"))
                likelihood = solution.get("likelihood", metadata.get("likelihood"))
                
                if posterior is not None and prior is not None:
                    # Create Bayesian problem
                    problem = BayesianProblem(
                        prior=float(prior),
                        likelihood=float(likelihood) if likelihood else None,
                        sensitivity=float(sensitivity) if sensitivity else None,
                        specificity=float(specificity) if specificity else None,
                    )
                    
                    # Verify
                    result = self.math_engine.verify_bayesian_calculation(
                        problem, float(posterior)
                    )
                    
                    # Track
                    self._record_verification(result, "bayesian")
                    
                    return result
        except Exception as e:
            logger.warning(f"Bayesian verification failed: {e}")
        
        return None

    def _verify_probability_content(
        self, outcome: "ExecutionOutcome"
    ) -> Optional[VerificationResult]:
        """Verify probability distributions in the outcome."""
        if not self.math_engine or not outcome.solution:
            return None
        
        solution = outcome.solution
        
        try:
            # Check for probability distribution
            if isinstance(solution, dict):
                # Look for distribution-like keys
                distribution_keys = ["probabilities", "distribution", "pmf", "pdf"]
                
                for key in distribution_keys:
                    if key in solution and isinstance(solution[key], dict):
                        result = self.math_engine.verify_probability_distribution(
                            solution[key]
                        )
                        
                        self._record_verification(result, "probability")
                        return result
        except Exception as e:
            logger.warning(f"Probability verification failed: {e}")
        
        return None

    def _verify_arithmetic_content(
        self, outcome: "ExecutionOutcome"
    ) -> Optional[VerificationResult]:
        """Verify arithmetic calculations in the outcome."""
        if not self.math_engine or not outcome.solution:
            return None
        
        solution = outcome.solution
        
        try:
            # Check for arithmetic expressions
            if isinstance(solution, dict):
                expression = solution.get("expression")
                result_value = solution.get("result")
                variables = solution.get("variables", {})
                
                if expression and result_value is not None:
                    result = self.math_engine.verify_arithmetic(
                        expression, float(result_value), variables
                    )
                    
                    self._record_verification(result, "arithmetic")
                    return result
        except Exception as e:
            logger.warning(f"Arithmetic verification failed: {e}")
        
        return None

    def _apply_verification_results(
        self,
        outcome: "ExecutionOutcome",
        results: List[VerificationResult],
    ) -> "ExecutionOutcome":
        """Apply verification results to the outcome."""
        math_verified = True
        math_errors = []
        math_corrections = {}
        
        for result in results:
            if result.status == MathVerificationStatus.ERROR_DETECTED:
                math_verified = False
                math_errors.extend(result.errors)
                math_corrections.update(result.corrections)
                
                # Count errors
                for error in result.errors:
                    self.math_error_counts[error.value] += 1
                
                # Apply corrections to outcome
                if result.corrections:
                    self._apply_corrections_to_solution(
                        outcome, result.corrections
                    )
                    for corr_type in result.corrections:
                        self.math_correction_counts[corr_type] += 1
        
        # Update outcome metadata
        if not hasattr(outcome, 'metadata'):
            outcome.metadata = {}
        
        outcome.metadata["math_verification"] = {
            "verified": math_verified,
            "errors": [e.value for e in math_errors],
            "corrections_applied": len(math_corrections) > 0,
            "corrections": math_corrections,
        }
        
        if not math_verified:
            outcome.errors = list(outcome.errors) if outcome.errors else []
            outcome.errors.append(
                f"Mathematical verification failed: {[e.value for e in math_errors]}"
            )
        
        return outcome

    def _apply_corrections_to_solution(
        self,
        outcome: "ExecutionOutcome",
        corrections: Dict[str, Any],
    ):
        """Apply mathematical corrections to the solution."""
        if not outcome.solution or not isinstance(outcome.solution, dict):
            return
        
        # Apply specific corrections
        if "correct_posterior" in corrections:
            outcome.solution["corrected_posterior"] = corrections["correct_posterior"]
            outcome.solution["math_corrected"] = True
        
        if "correct_result" in corrections:
            outcome.solution["corrected_result"] = corrections["correct_result"]
            outcome.solution["math_corrected"] = True

    def _record_verification(self, result: VerificationResult, verification_type: str):
        """Record verification for tracking and learning."""
        record = {
            "timestamp": time.time(),
            "type": verification_type,
            "status": result.status.value,
            "confidence": result.confidence,
            "errors": [e.value for e in result.errors],
        }
        
        self.math_verification_history.append(record)

    def _create_error_result(
        self, error_message: str
    ) -> Tuple["DecompositionPlan", "ExecutionOutcome"]:
        """Create error result when decomposition fails."""
        if DecompositionPlan is not None:
            plan = DecompositionPlan(
                steps=[],
                confidence=0.0,
                metadata={"error": error_message},
            )
        else:
            plan = None
        
        if ExecutionOutcome is not None:
            outcome = ExecutionOutcome(
                success=False,
                execution_time=0.0,
                errors=[error_message],
            )
        else:
            outcome = None
        
        return plan, outcome

    def verify_bayesian_calculation(
        self,
        problem: "BayesianProblem",
        claimed_posterior: float,
    ) -> Optional[VerificationResult]:
        """
        Direct Bayesian calculation verification.
        
        Convenience method for verifying Bayesian calculations directly.
        
        Args:
            problem: Bayesian problem specification
            claimed_posterior: The claimed posterior probability
            
        Returns:
            VerificationResult or None if engine unavailable
        """
        if not self.math_engine:
            return None
        
        result = self.math_engine.verify_bayesian_calculation(
            problem, claimed_posterior
        )
        
        self._record_verification(result, "bayesian_direct")
        
        return result

    def get_math_statistics(self) -> Dict[str, Any]:
        """Get mathematical verification statistics."""
        return {
            "total_verifications": len(self.math_verification_history),
            "error_counts": dict(self.math_error_counts),
            "correction_counts": dict(self.math_correction_counts),
            "engine_available": self.math_engine is not None,
            "engine_statistics": (
                self.math_engine.get_statistics()
                if self.math_engine
                else {}
            ),
        }

    def get_likely_errors(
        self, problem: "BayesianProblem"
    ) -> List[MathErrorType]:
        """
        Get likely mathematical errors for a problem.
        
        Uses meta-learning to predict common errors.
        """
        if not self.math_engine:
            return []
        
        return self.math_engine.get_likely_errors(problem)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EnhancedMathematicalDecomposer",
    "MathematicalProblemContext",
    "detect_mathematical_problem",
    "MATH_VERIFICATION_AVAILABLE",
    "DECOMPOSER_AVAILABLE",
]
