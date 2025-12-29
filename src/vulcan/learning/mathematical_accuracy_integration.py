"""
Mathematical Accuracy Learning Integration for VULCAN-AGI.

Connects the MathematicalVerificationEngine with Vulcan's Learning System
to provide feedback loops for mathematical reasoning accuracy.

This module enables:
- Automatic penalization of tools that produce mathematical errors
- Rewarding tools that produce verified mathematical results
- Learning from mathematical error patterns
- Cross-domain mathematical pattern transfer

Critical focus: Connecting mathematical verification to tool selection weights.
"""

import logging
import threading
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
        MathematicalVerificationEngine,
        MathErrorType,
        MathVerificationStatus,
        VerificationResult,
    )

    MATH_VERIFICATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mathematical verification not available: {e}")
    MATH_VERIFICATION_AVAILABLE = False
    MathematicalVerificationEngine = None
    MathErrorType = None
    MathVerificationStatus = None
    BayesianProblem = None
    VerificationResult = None

# Import learning system
try:
    from ..learning import (
        UnifiedLearningSystem,
        LearningConfig,
        FeedbackData,
        WEIGHT_ADJUSTMENT_SUCCESS,
        WEIGHT_ADJUSTMENT_FAILURE,
    )

    LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Learning system not available: {e}")
    LEARNING_AVAILABLE = False
    UnifiedLearningSystem = None
    LearningConfig = None
    FeedbackData = None
    WEIGHT_ADJUSTMENT_SUCCESS = 0.01
    WEIGHT_ADJUSTMENT_FAILURE = -0.005


# ============================================================================
# WEIGHT ADJUSTMENT CONSTANTS FOR MATHEMATICAL ERRORS
# ============================================================================

# Different error types have different severity levels
MATH_ERROR_PENALTIES = {
    # Critical errors - severe penalty
    MathErrorType.SPECIFICITY_CONFUSION: -0.02,
    MathErrorType.BASE_RATE_NEGLECT: -0.015,
    MathErrorType.BAYES_THEOREM_ERROR: -0.02,
    # Standard errors - moderate penalty
    MathErrorType.COMPLEMENT_ERROR: -0.01,
    MathErrorType.CONDITIONAL_PROBABILITY_ERROR: -0.01,
    MathErrorType.PROBABILITY_AXIOM_VIOLATION: -0.01,
    # Recoverable errors - lighter penalty  
    MathErrorType.ARITHMETIC_ERROR: -0.008,
    MathErrorType.NUMERICAL_OVERFLOW: -0.005,
    MathErrorType.DIVISION_BY_ZERO: -0.005,
    # Conceptual errors - moderate penalty
    MathErrorType.CONCEPTUAL_MISUSE: -0.01,
    MathErrorType.UNIT_MISMATCH: -0.008,
    MathErrorType.SENSITIVITY_CONFUSION: -0.015,
} if MATH_VERIFICATION_AVAILABLE else {}

# Reward for verified correct calculations
MATH_VERIFICATION_REWARD = 0.015

# Extra reward for correctly handling difficult cases
MATH_DIFFICULT_CASE_BONUS = 0.01


# ============================================================================
# MATHEMATICAL FEEDBACK DATA
# ============================================================================


@dataclass
class MathematicalFeedback:
    """Feedback data from mathematical verification."""

    tool_name: str
    verified: bool
    error_type: Optional[str] = None
    error_severity: float = 0.0
    correction_applied: bool = False
    problem_type: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# MATHEMATICAL ACCURACY INTEGRATION
# ============================================================================


class MathematicalAccuracyIntegration:
    """
    Integrates mathematical verification with Vulcan's learning system.
    
    Provides feedback loops to penalize tools that produce mathematical errors
    and reward tools that produce verified correct results.
    
    Key Functionality:
    - Receives verification results from MathematicalVerificationEngine
    - Maps errors to appropriate tool penalties
    - Communicates with UnifiedLearningSystem for weight adjustments
    - Tracks mathematical accuracy patterns over time
    
    Example usage:
        >>> math_engine = MathematicalVerificationEngine()
        >>> integration = MathematicalAccuracyIntegration(math_engine)
        >>> 
        >>> # After a tool produces a result
        >>> verification_result = math_engine.verify_bayesian_calculation(problem, answer)
        >>> await integration.process_verification_result(
        ...     verification_result,
        ...     tool_name="probabilistic",
        ...     learning_system=learning_system
        ... )
    """

    def __init__(
        self,
        math_engine: Optional["MathematicalVerificationEngine"] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize mathematical accuracy integration.
        
        Args:
            math_engine: MathematicalVerificationEngine instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.lock = threading.RLock()
        
        # Initialize or use provided math engine
        if math_engine:
            self.math_engine = math_engine
        elif MATH_VERIFICATION_AVAILABLE:
            self.math_engine = MathematicalVerificationEngine()
        else:
            self.math_engine = None
            logger.warning("Mathematical verification engine not available")
        
        # Tracking
        self.feedback_history = deque(maxlen=10000)
        self.tool_error_counts = defaultdict(lambda: defaultdict(int))
        self.tool_success_counts = defaultdict(int)
        self.total_verifications = 0
        self.total_errors = 0
        
        # Error pattern learning
        self.error_patterns = defaultdict(list)
        self.tool_error_patterns = defaultdict(lambda: defaultdict(int))
        
        logger.info("MathematicalAccuracyIntegration initialized")

    async def process_verification_result(
        self,
        verification_result: "VerificationResult",
        tool_name: str,
        learning_system: Optional["UnifiedLearningSystem"] = None,
        problem_type: str = "unknown",
    ) -> MathematicalFeedback:
        """
        Process a verification result and provide feedback to learning system.
        
        This is the main integration point connecting mathematical verification
        to tool weight adjustments.
        
        Args:
            verification_result: Result from MathematicalVerificationEngine
            tool_name: Name of the tool that produced the result
            learning_system: Optional UnifiedLearningSystem for weight updates
            problem_type: Type of mathematical problem
            
        Returns:
            MathematicalFeedback with processing results
        """
        with self.lock:
            self.total_verifications += 1
            
            if not verification_result.is_valid():
                # Mathematical error detected - PENALIZE TOOL
                return await self._handle_error(
                    verification_result, tool_name, learning_system, problem_type
                )
            else:
                # Correct result - REWARD TOOL
                return await self._handle_success(
                    verification_result, tool_name, learning_system, problem_type
                )

    async def _handle_error(
        self,
        result: "VerificationResult",
        tool_name: str,
        learning_system: Optional["UnifiedLearningSystem"],
        problem_type: str,
    ) -> MathematicalFeedback:
        """Handle mathematical error - penalize tool."""
        self.total_errors += 1
        
        # Get primary error type
        primary_error = result.errors[0] if result.errors else None
        error_type_str = primary_error.value if primary_error else "unknown"
        
        # Calculate penalty
        penalty = self._calculate_error_penalty(result.errors)
        
        # Update tracking
        self.tool_error_counts[tool_name][error_type_str] += 1
        self.tool_error_patterns[tool_name][error_type_str] += 1
        
        # Log error for pattern learning
        self.error_patterns[error_type_str].append({
            "tool": tool_name,
            "problem_type": problem_type,
            "timestamp": time.time(),
            "confidence": result.confidence,
        })
        
        # Apply penalty to learning system
        if learning_system:
            await self._apply_tool_penalty(
                learning_system,
                tool_name,
                penalty,
                error_type_str,
            )
            logger.info(
                f"[MathLearning] PENALIZED tool '{tool_name}' by {penalty:.4f} "
                f"for {error_type_str} error"
            )
        else:
            logger.warning(
                f"[MathLearning] No learning system - would penalize '{tool_name}' "
                f"by {penalty:.4f} for {error_type_str}"
            )
        
        # Create feedback record
        feedback = MathematicalFeedback(
            tool_name=tool_name,
            verified=False,
            error_type=error_type_str,
            error_severity=abs(penalty),
            correction_applied=bool(result.corrections),
            problem_type=problem_type,
            metadata={
                "errors": [e.value for e in result.errors],
                "corrections": result.corrections,
                "explanation": result.explanation,
            },
        )
        
        self.feedback_history.append(feedback)
        return feedback

    async def _handle_success(
        self,
        result: "VerificationResult",
        tool_name: str,
        learning_system: Optional["UnifiedLearningSystem"],
        problem_type: str,
    ) -> MathematicalFeedback:
        """Handle correct mathematical result - reward tool."""
        self.tool_success_counts[tool_name] += 1
        
        # Calculate reward
        reward = MATH_VERIFICATION_REWARD
        
        # Bonus for high confidence verification
        if result.confidence > 0.95:
            reward += MATH_DIFFICULT_CASE_BONUS * (result.confidence - 0.95) / 0.05
        
        # Apply reward to learning system
        if learning_system:
            await self._apply_tool_reward(learning_system, tool_name, reward)
            logger.info(
                f"[MathLearning] REWARDED tool '{tool_name}' by {reward:.4f} "
                f"for verified {problem_type} calculation"
            )
        else:
            logger.debug(
                f"[MathLearning] No learning system - would reward '{tool_name}' "
                f"by {reward:.4f}"
            )
        
        # Create feedback record
        feedback = MathematicalFeedback(
            tool_name=tool_name,
            verified=True,
            error_type=None,
            error_severity=0.0,
            correction_applied=False,
            problem_type=problem_type,
            metadata={
                "confidence": result.confidence,
            },
        )
        
        self.feedback_history.append(feedback)
        return feedback

    def _calculate_error_penalty(
        self, errors: List["MathErrorType"]
    ) -> float:
        """Calculate total penalty for errors."""
        if not errors or not MATH_VERIFICATION_AVAILABLE:
            return WEIGHT_ADJUSTMENT_FAILURE
        
        total_penalty = 0.0
        for error in errors:
            if error in MATH_ERROR_PENALTIES:
                total_penalty += MATH_ERROR_PENALTIES[error]
            else:
                total_penalty += WEIGHT_ADJUSTMENT_FAILURE
        
        # Cap penalty to prevent excessive single-event impact
        return max(total_penalty, -0.05)

    async def _apply_tool_penalty(
        self,
        learning_system: "UnifiedLearningSystem",
        tool_name: str,
        penalty: float,
        error_type: str,
    ):
        """Apply penalty to tool weight in learning system."""
        # Use learning system's weight adjustment mechanism
        if hasattr(learning_system, '_weight_lock'):
            with learning_system._weight_lock:
                if tool_name not in learning_system.tool_weight_adjustments:
                    learning_system.tool_weight_adjustments[tool_name] = 0.0
                
                old_weight = learning_system.tool_weight_adjustments[tool_name]
                new_weight = old_weight + penalty
                
                # Apply bounds from learning system
                from ..learning import MIN_TOOL_WEIGHT, MAX_TOOL_WEIGHT
                learning_system.tool_weight_adjustments[tool_name] = max(
                    MIN_TOOL_WEIGHT, min(MAX_TOOL_WEIGHT, new_weight)
                )
                
                logger.info(
                    f"[MathLearning] Tool '{tool_name}' weight: {old_weight:.4f} -> "
                    f"{learning_system.tool_weight_adjustments[tool_name]:.4f} "
                    f"(penalty: {penalty:.4f}, error: {error_type})"
                )
        else:
            # Fallback: create outcome for process_outcome
            outcome = {
                "query_id": f"math_penalty_{tool_name}_{time.time()}",
                "status": "error",
                "tools": [tool_name],
                "query_type": "mathematical_verification",
                "routing_ms": 0,
                "total_ms": 0,
                "math_error": error_type,
            }
            await learning_system.process_outcome(outcome)

    async def _apply_tool_reward(
        self,
        learning_system: "UnifiedLearningSystem",
        tool_name: str,
        reward: float,
    ):
        """Apply reward to tool weight in learning system."""
        if hasattr(learning_system, '_weight_lock'):
            with learning_system._weight_lock:
                if tool_name not in learning_system.tool_weight_adjustments:
                    learning_system.tool_weight_adjustments[tool_name] = 0.0
                
                old_weight = learning_system.tool_weight_adjustments[tool_name]
                new_weight = old_weight + reward
                
                # Apply bounds
                from ..learning import MIN_TOOL_WEIGHT, MAX_TOOL_WEIGHT
                learning_system.tool_weight_adjustments[tool_name] = max(
                    MIN_TOOL_WEIGHT, min(MAX_TOOL_WEIGHT, new_weight)
                )
                
                logger.info(
                    f"[MathLearning] Tool '{tool_name}' weight: {old_weight:.4f} -> "
                    f"{learning_system.tool_weight_adjustments[tool_name]:.4f} "
                    f"(reward: {reward:.4f})"
                )
        else:
            # Fallback: create successful outcome
            outcome = {
                "query_id": f"math_reward_{tool_name}_{time.time()}",
                "status": "success",
                "tools": [tool_name],
                "query_type": "mathematical_verification",
                "routing_ms": 0,
                "total_ms": 0,
            }
            await learning_system.process_outcome(outcome)

    # ========================================================================
    # CONVENIENCE METHODS FOR COMMON OPERATIONS
    # ========================================================================

    async def verify_and_learn(
        self,
        problem: "BayesianProblem",
        claimed_answer: float,
        tool_name: str,
        learning_system: Optional["UnifiedLearningSystem"] = None,
    ) -> Tuple["VerificationResult", MathematicalFeedback]:
        """
        Verify a Bayesian calculation and apply learning feedback.
        
        Convenience method that combines verification and learning.
        
        Args:
            problem: BayesianProblem specification
            claimed_answer: The claimed posterior probability
            tool_name: Tool that produced the answer
            learning_system: Learning system for weight updates
            
        Returns:
            Tuple of (VerificationResult, MathematicalFeedback)
        """
        if not self.math_engine:
            raise RuntimeError("Mathematical verification engine not available")
        
        # Verify the calculation
        verification_result = self.math_engine.verify_bayesian_calculation(
            problem, claimed_answer
        )
        
        # Process result through learning integration
        feedback = await self.process_verification_result(
            verification_result,
            tool_name,
            learning_system,
            problem_type="bayesian",
        )
        
        return verification_result, feedback

    def penalize_tool(
        self,
        tool_name: str,
        error_type: "MathErrorType",
        learning_system: "UnifiedLearningSystem",
    ):
        """
        Synchronous method to penalize a tool for mathematical error.
        
        This is a convenience method for direct penalization without
        async context.
        
        Args:
            tool_name: Name of the tool to penalize
            error_type: Type of mathematical error
            learning_system: Learning system for weight update
        """
        import asyncio
        
        if not MATH_VERIFICATION_AVAILABLE:
            logger.warning("Mathematical verification not available")
            return
        
        penalty = MATH_ERROR_PENALTIES.get(error_type, WEIGHT_ADJUSTMENT_FAILURE)
        
        # Apply directly to learning system
        if hasattr(learning_system, '_weight_lock'):
            with learning_system._weight_lock:
                if tool_name not in learning_system.tool_weight_adjustments:
                    learning_system.tool_weight_adjustments[tool_name] = 0.0
                
                from ..learning import MIN_TOOL_WEIGHT, MAX_TOOL_WEIGHT
                old = learning_system.tool_weight_adjustments[tool_name]
                learning_system.tool_weight_adjustments[tool_name] = max(
                    MIN_TOOL_WEIGHT,
                    min(MAX_TOOL_WEIGHT, old + penalty)
                )
                
                logger.info(
                    f"[MathLearning] Penalized '{tool_name}': {old:.4f} -> "
                    f"{learning_system.tool_weight_adjustments[tool_name]:.4f}"
                )
        
        # Track error
        self.tool_error_counts[tool_name][error_type.value] += 1
        self.total_errors += 1

    def reward_tool(
        self,
        tool_name: str,
        learning_system: "UnifiedLearningSystem",
    ):
        """
        Synchronous method to reward a tool for correct mathematical result.
        
        Args:
            tool_name: Name of the tool to reward
            learning_system: Learning system for weight update
        """
        if hasattr(learning_system, '_weight_lock'):
            with learning_system._weight_lock:
                if tool_name not in learning_system.tool_weight_adjustments:
                    learning_system.tool_weight_adjustments[tool_name] = 0.0
                
                from ..learning import MIN_TOOL_WEIGHT, MAX_TOOL_WEIGHT
                old = learning_system.tool_weight_adjustments[tool_name]
                learning_system.tool_weight_adjustments[tool_name] = max(
                    MIN_TOOL_WEIGHT,
                    min(MAX_TOOL_WEIGHT, old + MATH_VERIFICATION_REWARD)
                )
                
                logger.info(
                    f"[MathLearning] Rewarded '{tool_name}': {old:.4f} -> "
                    f"{learning_system.tool_weight_adjustments[tool_name]:.4f}"
                )
        
        # Track success
        self.tool_success_counts[tool_name] += 1

    # ========================================================================
    # STATISTICS AND REPORTING
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get mathematical accuracy statistics."""
        with self.lock:
            tool_accuracy = {}
            for tool in set(self.tool_success_counts.keys()) | set(self.tool_error_counts.keys()):
                successes = self.tool_success_counts.get(tool, 0)
                total_errors = sum(self.tool_error_counts.get(tool, {}).values())
                total = successes + total_errors
                tool_accuracy[tool] = successes / total if total > 0 else 0.0
            
            return {
                "total_verifications": self.total_verifications,
                "total_errors": self.total_errors,
                "overall_accuracy": (
                    (self.total_verifications - self.total_errors) / self.total_verifications
                    if self.total_verifications > 0 else 0.0
                ),
                "tool_accuracy": tool_accuracy,
                "tool_error_counts": {
                    tool: dict(errors) for tool, errors in self.tool_error_counts.items()
                },
                "tool_success_counts": dict(self.tool_success_counts),
                "most_common_errors": self._get_most_common_errors(),
            }

    def _get_most_common_errors(self) -> List[Tuple[str, int]]:
        """Get most common error types across all tools."""
        error_totals = defaultdict(int)
        for tool_errors in self.tool_error_counts.values():
            for error_type, count in tool_errors.items():
                error_totals[error_type] += count
        
        return sorted(error_totals.items(), key=lambda x: x[1], reverse=True)[:10]

    def get_tool_error_report(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed error report for a specific tool."""
        with self.lock:
            errors = dict(self.tool_error_counts.get(tool_name, {}))
            successes = self.tool_success_counts.get(tool_name, 0)
            total_errors = sum(errors.values())
            total = successes + total_errors
            
            return {
                "tool_name": tool_name,
                "total_verifications": total,
                "successes": successes,
                "errors": errors,
                "accuracy": successes / total if total > 0 else 0.0,
                "most_common_error": (
                    max(errors.items(), key=lambda x: x[1])[0]
                    if errors else None
                ),
            }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_math_learning_integration(
    math_engine: Optional["MathematicalVerificationEngine"] = None,
) -> MathematicalAccuracyIntegration:
    """Create a MathematicalAccuracyIntegration instance."""
    return MathematicalAccuracyIntegration(math_engine=math_engine)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MathematicalAccuracyIntegration",
    "MathematicalFeedback",
    "create_math_learning_integration",
    "MATH_ERROR_PENALTIES",
    "MATH_VERIFICATION_REWARD",
    "MATH_DIFFICULT_CASE_BONUS",
]
