"""
GVulcan Bridge - Integration layer for gvulcan utilities.

This module provides a clean interface to gvulcan's unique data quality and policy
enforcement capabilities. It serves as the integration point between gvulcan's
specialized components and the broader VULCAN system.

Components Integrated:
    - **DQS (Data Quality Score)**: Multi-dimensional quality scoring for queries
      and data with configurable gating thresholds
    - **OPA (Open Policy Agent)**: Policy-as-code enforcement with write barriers
      and compliance validation

Design Principles:
    - Separation of Concerns: DQS and OPA are independent, orthogonal concerns
    - Fail-Safe: Operations default to permissive when components unavailable
    - Type Safety: Full type hints for static analysis
    - Security: Policy enforcement with audit logging
    - Observability: Comprehensive logging and metrics

Note:
    For storage components (Merkle trees, Bloom filters, GraphRAG), use
    MemoryBridge instead, which integrates persistant_memory_v46.

Example:
    Basic usage for data quality validation:
    
    >>> bridge = create_gvulcan_bridge({
    ...     "dqs_reject_threshold": 0.3,
    ...     "opa_cache_enabled": True
    ... })
    >>> 
    >>> # Validate data quality
    >>> result = bridge.validate_data_quality(
    ...     pii_confidence=0.1,
    ...     graph_completeness=0.95,
    ...     syntactic_completeness=0.98
    ... )
    >>> if result["gate_decision"] == "accept":
    ...     # Proceed with operation
    ...     pass
    >>> 
    >>> # Check policy compliance
    >>> allowed = bridge.check_write_barrier(dqs_score=0.85, context={})

Thread Safety:
    All public methods are thread-safe.

Author: VULCAN-AGI Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDENCY IMPORTS WITH GRACEFUL FALLBACK
# ============================================================================

GVULCAN_AVAILABLE = False
DQS_AVAILABLE = False
OPA_AVAILABLE = False

try:
    from src.gvulcan.dqs import DQSScorer, DQSComponents, DQSResult, compute_dqs
    DQS_AVAILABLE = True
    logger.info("gvulcan.dqs available")
except ImportError as e:
    logger.debug(f"gvulcan.dqs not available: {e}")

try:
    from src.gvulcan.opa import OPAClient, WriteBarrierInput, WriteBarrierResult
    OPA_AVAILABLE = True
    logger.info("gvulcan.opa available")
except ImportError as e:
    logger.debug(f"gvulcan.opa not available: {e}")

GVULCAN_AVAILABLE = DQS_AVAILABLE or OPA_AVAILABLE


# ============================================================================
# GVULCAN BRIDGE IMPLEMENTATION
# ============================================================================


class GVulcanBridge:
    """
    Bridge to gvulcan utilities for the VULCAN system.
    
    Provides a unified interface to gvulcan's data quality scoring and policy
    enforcement capabilities. Designed to be lightweight and focused on the
    unique features that gvulcan provides.
    
    Components:
        - **DQSScorer**: Data quality scoring with multi-component analysis
        - **OPAClient**: Policy enforcement with write barriers
    
    For storage operations (Merkle trees, Bloom filters, LSM, GraphRAG),
    use MemoryBridge instead, which integrates persistant_memory_v46.
    
    Thread Safety:
        All public methods are thread-safe. Internal state is read-only
        after initialization.
    
    Example:
        >>> config = {
        ...     "dqs_model": "v2",
        ...     "dqs_reject_threshold": 0.3,
        ...     "dqs_quarantine_threshold": 0.4,
        ...     "opa_bundle_version": "1.0.0"
        ... }
        >>> bridge = GVulcanBridge(config)
        >>> 
        >>> # Validate query quality
        >>> quality = bridge.validate_data_quality(
        ...     pii_confidence=0.05,
        ...     graph_completeness=0.9,
        ...     syntactic_completeness=0.95
        ... )
        >>> print(f"Quality score: {quality['score']}")
        >>> print(f"Gate decision: {quality['gate_decision']}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the GVulcan bridge.
        
        Args:
            config: Configuration dictionary with optional keys:
                - dqs_model: DQS model version (default: "v2")
                - dqs_reject_threshold: Reject below this score (default: 0.3)
                - dqs_quarantine_threshold: Quarantine below this (default: 0.4)
                - opa_bundle_version: OPA policy bundle version (default: "1.0.0")
                - opa_cache_enabled: Enable OPA result caching (default: True)
        """
        config = config or {}
        
        # Initialize DQS scorer
        self._dqs_scorer: Optional[Any] = None
        if DQS_AVAILABLE:
            self._init_dqs_scorer(config)
        
        # Initialize OPA client
        self._opa_client: Optional[Any] = None
        if OPA_AVAILABLE:
            self._init_opa_client(config)
    
    def _init_dqs_scorer(self, config: Dict[str, Any]) -> None:
        """
        Initialize DQS scorer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        try:
            self._dqs_scorer = DQSScorer(
                model=config.get("dqs_model", "v2"),
                reject_below=config.get("dqs_reject_threshold", 0.3),
                quarantine_below=config.get("dqs_quarantine_threshold", 0.4),
            )
            logger.info(
                f"DQSScorer initialized (model={config.get('dqs_model', 'v2')}, "
                f"reject_threshold={config.get('dqs_reject_threshold', 0.3)})"
            )
        except Exception as e:
            logger.warning(f"DQSScorer initialization failed: {e}", exc_info=True)
            self._dqs_scorer = None
    
    def _init_opa_client(self, config: Dict[str, Any]) -> None:
        """
        Initialize OPA client with configuration.
        
        Args:
            config: Configuration dictionary
        """
        try:
            self._opa_client = OPAClient(
                bundle_version=config.get("opa_bundle_version", "1.0.0"),
                enable_cache=config.get("opa_cache_enabled", True),
            )
            logger.info(
                f"OPAClient initialized (bundle_version="
                f"{config.get('opa_bundle_version', '1.0.0')}, "
                f"cache_enabled={config.get('opa_cache_enabled', True)})"
            )
        except Exception as e:
            logger.warning(f"OPAClient initialization failed: {e}", exc_info=True)
            self._opa_client = None
    
    # ========================================================================
    # DATA QUALITY OPERATIONS
    # ========================================================================
    
    def validate_data_quality(
        self,
        pii_confidence: float = 0.0,
        graph_completeness: float = 1.0,
        syntactic_completeness: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate data quality using DQS scoring.
        
        Computes a multi-component data quality score based on:
        - **PII confidence**: Likelihood of containing personally identifiable
          information (0.0 = no PII, 1.0 = definite PII)
        - **Graph completeness**: Completeness of graph structure (0.0-1.0)
        - **Syntactic completeness**: Syntactic validity (0.0-1.0)
        
        The combined score determines the gate decision:
        - **accept**: Quality exceeds quarantine threshold
        - **quarantine**: Quality between reject and quarantine thresholds
        - **reject**: Quality below reject threshold
        
        Args:
            pii_confidence: PII likelihood [0.0-1.0], lower is better
            graph_completeness: Graph structure completeness [0.0-1.0]
            syntactic_completeness: Syntactic validity [0.0-1.0]
        
        Returns:
            Quality result dictionary with keys:
                - score: Overall quality score [0.0-1.0]
                - gate_decision: Decision (accept, quarantine, reject)
                - components: Individual component scores
            Returns None if DQS not available
        
        Raises:
            ValueError: If any component score is outside [0.0-1.0]
        
        Thread Safety:
            This method is thread-safe.
        
        Example:
            >>> result = bridge.validate_data_quality(
            ...     pii_confidence=0.05,
            ...     graph_completeness=0.95,
            ...     syntactic_completeness=0.98
            ... )
            >>> if result["gate_decision"] == "accept":
            ...     print(f"Quality check passed: {result['score']:.2f}")
        """
        if not self._dqs_scorer:
            logger.warning("DQS scorer not available")
            return None
        
        # Validate inputs
        for name, value in [
            ("pii_confidence", pii_confidence),
            ("graph_completeness", graph_completeness),
            ("syntactic_completeness", syntactic_completeness)
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )
        
        try:
            components = DQSComponents(
                pii_confidence=pii_confidence,
                graph_completeness=graph_completeness,
                syntactic_completeness=syntactic_completeness,
            )
            result = self._dqs_scorer.score(components)
            
            logger.debug(
                f"DQS validation: score={result.score:.3f}, "
                f"decision={result.gate_decision}"
            )
            
            return {
                "score": result.score,
                "gate_decision": result.gate_decision,
                "components": {
                    "pii_confidence": pii_confidence,
                    "graph_completeness": graph_completeness,
                    "syntactic_completeness": syntactic_completeness,
                },
            }
        except Exception as e:
            logger.error(f"DQS validation failed: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # POLICY ENFORCEMENT OPERATIONS
    # ========================================================================
    
    def check_write_barrier(
        self,
        dqs_score: float,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if operation passes write barrier policy.
        
        Evaluates OPA policies to determine if a write operation should be
        permitted based on data quality score and contextual information.
        
        The write barrier implements defense-in-depth:
        1. Quality threshold enforcement (DQS score)
        2. Context-based policy evaluation (PII, sensitivity, etc.)
        3. Rate limiting and quota checks
        
        Args:
            dqs_score: Data quality score [0.0-1.0] from DQS validation
            context: Context dictionary with optional keys:
                - pii_detected: Boolean indicating PII presence
                - sensitivity_level: Data sensitivity (low, medium, high)
                - source: Data source identifier
                - requester_id: ID of requesting entity
        
        Returns:
            True if operation is permitted, False if denied
            If OPA not available, defaults to True (fail-open)
        
        Raises:
            ValueError: If dqs_score is outside [0.0-1.0]
        
        Security Note:
            This method fails open (returns True) when OPA is unavailable
            to prevent denial of service. For production deployments requiring
            strict policy enforcement, ensure OPA is always available.
        
        Thread Safety:
            This method is thread-safe.
        
        Example:
            >>> allowed = bridge.check_write_barrier(
            ...     dqs_score=0.85,
            ...     context={
            ...         "pii_detected": False,
            ...         "sensitivity_level": "medium",
            ...         "source": "user_input"
            ...     }
            ... )
            >>> if allowed:
            ...     perform_write()
        """
        if not 0.0 <= dqs_score <= 1.0:
            raise ValueError(
                f"dqs_score must be in [0.0, 1.0], got {dqs_score}"
            )
        
        if not self._opa_client:
            logger.warning(
                "OPA client not available, defaulting to allow (fail-open)"
            )
            return True  # Fail open when OPA unavailable
        
        try:
            barrier_input = WriteBarrierInput(dqs=dqs_score, pii=context)
            result = self._opa_client.evaluate_write_barrier(barrier_input)
            
            logger.debug(
                f"Write barrier check: dqs={dqs_score:.3f}, "
                f"allowed={result.allow}"
            )
            
            return result.allow
        except Exception as e:
            logger.error(f"Write barrier check failed: {e}", exc_info=True)
            # Fail open on error to avoid blocking legitimate operations
            return True
    
    # ========================================================================
    # STATUS AND METRICS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of gvulcan components.
        
        Returns:
            Status dictionary with component availability and initialization state
        
        Thread Safety:
            This method is thread-safe.
        """
        return {
            "gvulcan_available": GVULCAN_AVAILABLE,
            "dqs_available": DQS_AVAILABLE,
            "opa_available": OPA_AVAILABLE,
            "dqs_initialized": self._dqs_scorer is not None,
            "opa_initialized": self._opa_client is not None,
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_gvulcan_bridge(
    config: Optional[Dict[str, Any]] = None
) -> Optional[GVulcanBridge]:
    """
    Factory function to create GVulcanBridge instance.
    
    Creates a bridge only if gvulcan components are available. This provides
    graceful degradation when gvulcan dependencies are not installed.
    
    Args:
        config: Configuration dictionary (see GVulcanBridge.__init__ for options)
    
    Returns:
        Initialized GVulcanBridge instance, or None if gvulcan unavailable
    
    Example:
        >>> bridge = create_gvulcan_bridge({
        ...     "dqs_reject_threshold": 0.3,
        ...     "opa_cache_enabled": True
        ... })
        >>> if bridge:
        ...     result = bridge.validate_data_quality(...)
    """
    if not GVULCAN_AVAILABLE:
        logger.info("gvulcan not available, bridge disabled")
        return None
    
    try:
        return GVulcanBridge(config)
    except Exception as e:
        logger.warning(f"Failed to create GVulcanBridge: {e}", exc_info=True)
        return None


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "GVulcanBridge",
    "create_gvulcan_bridge",
    "GVULCAN_AVAILABLE",
    "DQS_AVAILABLE",
    "OPA_AVAILABLE",
]
