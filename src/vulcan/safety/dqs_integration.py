"""
DQS Integration for VULCAN Safety Module

Bridges gvulcan's Data Quality Score system with vulcan.safety validation.

This module provides industry-standard integration between the DQS (Data Quality Score)
system and the safety validation framework, enabling data quality-based access control
and validation workflows.

Industry standard features:
- Comprehensive input validation
- Type safety with dataclasses
- Detailed error handling and logging
- Graceful degradation when gvulcan is unavailable
- Thread-safe operations
- Comprehensive documentation
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import gvulcan components with graceful fallback
try:
    from gvulcan.dqs import DQSScorer, DQSComponents, DQSResult
    from gvulcan.opa import OPAClient, WriteBarrierInput
    DQS_AVAILABLE = True
except ImportError as e:
    DQS_AVAILABLE = False
    logger.warning(f"gvulcan.dqs not available - DQS integration disabled: {e}")
    
    # Create placeholder types for graceful degradation
    DQSScorer = None
    DQSComponents = None
    DQSResult = None
    OPAClient = None
    WriteBarrierInput = None


class DQSValidator:
    """
    Validates data quality using gvulcan's DQS system.
    
    This class integrates DQS-based validation with OPA policy enforcement
    to provide comprehensive data quality gates.
    
    Industry standard implementation with:
    - Configurable quality thresholds
    - Integration with OPA policy engine
    - Comprehensive error handling
    - Detailed logging for audit trails
    - Thread-safe operations
    
    Attributes:
        scorer: DQS scoring engine instance
        opa_client: OPA client for policy evaluation
        reject_threshold: DQS threshold below which data is rejected
        quarantine_threshold: DQS threshold below which data is quarantined
        model: DQS model version to use
    """
    
    def __init__(
        self,
        reject_threshold: float = 0.3,
        quarantine_threshold: float = 0.4,
        model: str = "v2",
        opa_url: Optional[str] = None,
    ):
        """
        Initialize DQS validator.
        
        Args:
            reject_threshold: Minimum DQS score below which data is rejected (0.0-1.0)
            quarantine_threshold: Minimum DQS score below which data is quarantined (0.0-1.0)
            model: DQS model version to use ("v1" or "v2")
            opa_url: Optional OPA server URL for remote policy evaluation
            
        Raises:
            ImportError: If gvulcan.dqs is not available
            ValueError: If thresholds are invalid
        """
        if not DQS_AVAILABLE:
            raise ImportError(
                "gvulcan.dqs required for DQS validation. "
                "Install with: pip install -e .[gvulcan]"
            )
        
        # Validate thresholds
        if not (0.0 <= reject_threshold <= 1.0):
            raise ValueError("reject_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= quarantine_threshold <= 1.0):
            raise ValueError("quarantine_threshold must be between 0.0 and 1.0")
        
        if reject_threshold > quarantine_threshold:
            raise ValueError("reject_threshold must be <= quarantine_threshold")
        
        if model not in ("v1", "v2"):
            raise ValueError("model must be 'v1' or 'v2'")
        
        self.reject_threshold = reject_threshold
        self.quarantine_threshold = quarantine_threshold
        self.model = model
        
        # Initialize DQS scorer
        self.scorer = DQSScorer(
            reject_below=reject_threshold,
            quarantine_below=quarantine_threshold,
            model=model
        )
        
        # Initialize OPA client with TTL support
        self.opa_client = OPAClient(
            bundle_version="1.0.0",
            opa_url=opa_url,
            enable_cache=True,
            cache_ttl_seconds=300,  # 5 minute TTL
        )
        
        logger.info(
            f"Initialized DQS validator: "
            f"reject_threshold={reject_threshold}, "
            f"quarantine_threshold={quarantine_threshold}, "
            f"model={model}, "
            f"opa_url={'configured' if opa_url else 'local'}"
        )
    
    def validate(
        self,
        pii_confidence: float,
        graph_completeness: float,
        syntactic_completeness: float
    ) -> "DQSResult":
        """
        Validate data quality and return decision.
        
        Computes DQS score from component metrics and determines whether
        data should be accepted, quarantined, or rejected.
        
        Industry standard implementation with:
        - Input validation for all parameters
        - Comprehensive error handling
        - Detailed audit logging
        - Type-safe return values
        
        Args:
            pii_confidence: PII confidence score (0.0-1.0)
            graph_completeness: Graph completeness score (0.0-1.0)
            syntactic_completeness: Syntactic completeness score (0.0-1.0)
            
        Returns:
            DQSResult with computed score and decision
            
        Raises:
            ValueError: If input parameters are invalid
        """
        # Validate inputs
        if not (0.0 <= pii_confidence <= 1.0):
            raise ValueError("pii_confidence must be between 0.0 and 1.0")
        
        if not (0.0 <= graph_completeness <= 1.0):
            raise ValueError("graph_completeness must be between 0.0 and 1.0")
        
        if not (0.0 <= syntactic_completeness <= 1.0):
            raise ValueError("syntactic_completeness must be between 0.0 and 1.0")
        
        try:
            # Create DQS components
            components = DQSComponents(
                pii_confidence=pii_confidence,
                graph_completeness=graph_completeness,
                syntactic_completeness=syntactic_completeness
            )
            
            # Compute DQS score
            result = self.scorer.score(components)
            
            # Log validation result for audit trail
            logger.info(
                f"DQS validation: score={result.score:.3f}, "
                f"decision={result.decision}, "
                f"pii={pii_confidence:.3f}, "
                f"graph={graph_completeness:.3f}, "
                f"syntax={syntactic_completeness:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during DQS validation: {e}", exc_info=True)
            raise
    
    def check_write_barrier(
        self,
        dqs_score: float,
        pii_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if data passes write barrier using OPA policy.
        
        Evaluates write barrier policy with DQS score and PII information
        to determine if data should be allowed to be written.
        
        Industry standard implementation with:
        - Input validation
        - Policy-based access control
        - Comprehensive error handling
        - Audit logging
        
        Args:
            dqs_score: Data Quality Score (0.0-1.0)
            pii_info: Optional PII detection information
            
        Returns:
            True if data passes write barrier, False otherwise
            
        Raises:
            ValueError: If dqs_score is invalid
        """
        # Validate DQS score
        if not (0.0 <= dqs_score <= 1.0):
            raise ValueError("dqs_score must be between 0.0 and 1.0")
        
        # Default PII info if not provided
        if pii_info is None:
            pii_info = {"detected": False}
        
        try:
            # Create write barrier input
            barrier_input = WriteBarrierInput(
                dqs=dqs_score,
                pii=pii_info
            )
            
            # Evaluate policy
            result = self.opa_client.evaluate_write_barrier(barrier_input)
            
            # Log barrier check for audit trail
            logger.info(
                f"Write barrier check: dqs={dqs_score:.3f}, "
                f"allow={result.allow}, quarantine={result.quarantine}, "
                f"reason={result.deny_reason}"
            )
            
            return result.allow
            
        except Exception as e:
            logger.error(f"Error during write barrier check: {e}", exc_info=True)
            raise
    
    def validate_and_gate(
        self,
        pii_confidence: float,
        graph_completeness: float,
        syntactic_completeness: float,
        pii_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete validation and gating workflow.
        
        Combines DQS validation with write barrier checks to provide
        a complete data quality gate decision.
        
        This is the recommended high-level API for most use cases.
        
        Args:
            pii_confidence: PII confidence score (0.0-1.0)
            graph_completeness: Graph completeness score (0.0-1.0)
            syntactic_completeness: Syntactic completeness score (0.0-1.0)
            pii_info: Optional PII detection information
            
        Returns:
            Dictionary with validation results including:
            - dqs_score: Computed DQS score
            - dqs_decision: DQS decision (accept/quarantine/reject)
            - write_barrier_passed: Whether data passes write barrier
            - final_decision: Final combined decision
            - metadata: Additional validation metadata
            
        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Perform DQS validation
            dqs_result = self.validate(
                pii_confidence=pii_confidence,
                graph_completeness=graph_completeness,
                syntactic_completeness=syntactic_completeness
            )
            
            # Check write barrier
            barrier_passed = self.check_write_barrier(
                dqs_score=dqs_result.score,
                pii_info=pii_info
            )
            
            # Determine final decision
            # Data must pass both DQS validation and write barrier
            if dqs_result.decision == "accept" and barrier_passed:
                final_decision = "accept"
            elif dqs_result.decision == "quarantine" or not barrier_passed:
                final_decision = "quarantine"
            else:
                final_decision = "reject"
            
            result = {
                "dqs_score": dqs_result.score,
                "dqs_decision": dqs_result.decision,
                "write_barrier_passed": barrier_passed,
                "final_decision": final_decision,
                "metadata": {
                    "pii_confidence": pii_confidence,
                    "graph_completeness": graph_completeness,
                    "syntactic_completeness": syntactic_completeness,
                    "reject_threshold": self.reject_threshold,
                    "quarantine_threshold": self.quarantine_threshold,
                    "model": self.model,
                }
            }
            
            logger.info(
                f"Complete validation: final_decision={final_decision}, "
                f"dqs_score={dqs_result.score:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during validation and gating: {e}", exc_info=True)
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about validator operations.
        
        Returns:
            Dictionary with validator statistics including cache stats
        """
        stats = {
            "reject_threshold": self.reject_threshold,
            "quarantine_threshold": self.quarantine_threshold,
            "model": self.model,
        }
        
        # Add OPA client statistics
        if self.opa_client:
            stats["opa"] = self.opa_client.get_statistics()
        
        return stats


def create_validator(
    reject_threshold: float = 0.3,
    quarantine_threshold: float = 0.4,
    model: str = "v2",
    opa_url: Optional[str] = None,
) -> Optional[DQSValidator]:
    """
    Factory function to create DQS validator with graceful fallback.
    
    This is the recommended way to create a DQSValidator as it handles
    the case where gvulcan is not available gracefully.
    
    Args:
        reject_threshold: Minimum DQS score below which data is rejected
        quarantine_threshold: Minimum DQS score below which data is quarantined
        model: DQS model version to use
        opa_url: Optional OPA server URL
        
    Returns:
        DQSValidator instance if gvulcan is available, None otherwise
    """
    if not DQS_AVAILABLE:
        logger.warning("Cannot create DQS validator: gvulcan not available")
        return None
    
    try:
        return DQSValidator(
            reject_threshold=reject_threshold,
            quarantine_threshold=quarantine_threshold,
            model=model,
            opa_url=opa_url,
        )
    except Exception as e:
        logger.error(f"Failed to create DQS validator: {e}", exc_info=True)
        return None


# Export public API
__all__ = [
    "DQSValidator",
    "create_validator",
    "DQS_AVAILABLE",
]
