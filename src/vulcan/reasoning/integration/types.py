"""
Type definitions for the reasoning integration system.

This module contains all enums, dataclasses, and constants used throughout
the reasoning integration subsystem. It follows the single responsibility
principle by isolating type definitions from business logic.

Module: vulcan.reasoning.integration.types
Author: Vulcan AI Team
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Logging prefix for consistent output
LOG_PREFIX = "[ReasoningIntegration]"

# Default execution budgets
DEFAULT_MAX_WORKERS = 4  # Maximum parallel workers for PortfolioExecutor
DEFAULT_TIME_BUDGET_MS = 5000  # Default time budget in milliseconds
DEFAULT_ENERGY_BUDGET_MJ = 1000  # Default energy budget in millijoules
DEFAULT_MIN_CONFIDENCE = 0.5  # Minimum confidence threshold for results

# Note: Maximum fallback attempts to prevent infinite retry loops
MAX_FALLBACK_ATTEMPTS = 2  # Don't retry more than 2 times per query

# Note: Minimum confidence floor when all tools fail
MIN_CONFIDENCE_FLOOR = 0.15  # Prevent total query refusal

# Note: Confidence category thresholds for result quality assessment
CONFIDENCE_HIGH_THRESHOLD = 0.7     # High quality result
CONFIDENCE_GOOD_THRESHOLD = 0.6     # Good result
CONFIDENCE_MEDIUM_THRESHOLD = 0.5   # Medium result
CONFIDENCE_LOW_THRESHOLD = 0.15     # Low result floor

# Arena delegation configuration
ARENA_REASONING_URL = os.environ.get(
    "ARENA_REASONING_URL",
    "http://127.0.0.1:8080/arena/api/run/reasoner"
)
ARENA_DELEGATION_TIMEOUT = 60.0  # Timeout for Arena delegation requests

# Complexity thresholds for strategy selection
FAST_PATH_COMPLEXITY_THRESHOLD = 0.3  # Below this, use fast path
LOW_COMPLEXITY_THRESHOLD = 0.4  # Below this, use FAST mode
HIGH_COMPLEXITY_THRESHOLD = 0.7  # Above this, use ACCURATE mode

# CONFIGURABLE: Set VULCAN_DECOMPOSITION_THRESHOLD environment variable to override
try:
    DECOMPOSITION_COMPLEXITY_THRESHOLD = float(
        os.environ.get("VULCAN_DECOMPOSITION_THRESHOLD", "0.50")
    )
except (ValueError, TypeError):
    logger.warning("Invalid VULCAN_DECOMPOSITION_THRESHOLD, using default 0.50")
    DECOMPOSITION_COMPLEXITY_THRESHOLD = 0.50

# Strategy selection thresholds
CAUSAL_REASONING_THRESHOLD = 0.6  # Complexity threshold for causal reasoning
PROBABILISTIC_REASONING_THRESHOLD = 0.5  # Complexity threshold for probabilistic

# Maximum timing samples to keep for statistics
MAX_TIMING_SAMPLES = 100


# =============================================================================
# Query Analysis Constants
# =============================================================================

# Analysis indicators that mean query needs specialized tools
ANALYSIS_INDICATORS: frozenset = frozenset({
    # Causal analysis requests
    'intervene', 'intervention', 'causal', 'causation',
    'variable', 'counterfactual', 'do-calculus',
    'which causal', 'causal link', 'causal graph',
    # Weakness/error analysis requests
    'weakest', 'weakness', 'wrong', 'error', 'mistake',
    'flaw', 'incorrect', 'identify', 'find the',
    'could be wrong', 'step that', 'one step',
    # Proof/logical analysis
    'proof', 'prove', 'provably', 'theorem', 'lemma',
    'logical', 'derive', 'deduce', 'sketch',
    # Probability/statistical analysis
    'prior', 'posterior', 'likelihood', 'probability',
    'bayesian', 'update', 'misspecified', 'distribution',
    # Value/ethical analysis
    'conflict', 'dilemma', 'choice', 'decide', 'trolley',
    # Mathematical analysis
    'calculate', 'compute', 'solve', 'equation', 'formula',
    'integrate', 'differentiate', 'sum', 'product',
    # Data analysis
    'data', 'dataset', 'analyze', 'analysis', 'pattern',
})

# Action verbs that indicate VULCAN should analyze something
ACTION_VERBS: frozenset = frozenset({
    'analyze', 'evaluate', 'examine', 'check', 'review',
    'explain', 'compare', 'contrast', 'assess', 'critique',
    'reason', 'think', 'consider', 'determine', 'figure',
})

# Analysis indicators for ethical queries that need domain analysis
ETHICAL_ANALYSIS_INDICATORS: frozenset = frozenset({
    # Analytical requests
    'analyze', 'analysis', 'examine', 'investigate',
    'explain', 'describe', 'evaluate', 'assess',
    # Conflict/problem solving
    'what breaks', 'how to resolve', 'solve', 'fix',
    'identify', 'find the', 'which', 'weakest',
    # Domain-specific
    'data', 'algorithm', 'system', 'code', 'model',
    'calculation', 'computation', 'proof',
    # Causal/probabilistic
    'cause', 'effect', 'probability', 'likelihood',
    'intervene', 'variable', 'outcome',
})

# Pure ethical phrases that indicate deontic/ethical framework questions
PURE_ETHICAL_PHRASES: frozenset = frozenset({
    # Deontic language
    "is it permissible", "is it impermissible", "is it forbidden",
    "morally permissible", "morally wrong", "morally right",
    "ethically permissible", "ethically wrong", "ethically right",
    # Ethical framework questions
    "what would a utilitarian", "what would a deontologist",
    "from a virtue ethics", "consequentialist view",
    # Classic ethical dilemmas
    "trolley problem", "should i pull the lever",
    "runaway trolley", "fat man on bridge",
    # Obligation language
    "do i have an obligation", "is there a duty",
    "moral obligation", "ethical obligation",
})


# =============================================================================
# Enums
# =============================================================================

class ReasoningStrategyType(Enum):
    """
    Enumeration of available reasoning strategies.

    Each strategy represents a different approach to solving reasoning problems,
    with varying trade-offs between speed, accuracy, and resource usage.
    """

    DIRECT = "direct"
    CAUSAL_REASONING = "causal_reasoning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    PLANNING = "planning"
    DELIBERATIVE = "deliberative"
    META_REASONING = "meta_reasoning"
    PHILOSOPHICAL_REASONING = "philosophical_reasoning"
    DEFAULT = "default"


# =============================================================================
# Strategy and Route Mappings
# =============================================================================

# Maps query types to appropriate reasoning strategies
QUERY_TYPE_STRATEGY_MAP: Dict[str, str] = {
    "reasoning": "causal_reasoning",
    "execution": "planning",
    "perception": "analogical_reasoning",
    "planning": "deliberative",
    "learning": "meta_reasoning",
    "general": "direct",
    "philosophical": "meta_reasoning",  # Route to World Model meta-reasoning
    "ethical": "meta_reasoning",        # Route to World Model meta-reasoning
}

# Maps query routes to reasoning types
ROUTE_TO_REASONING_TYPE: Dict[str, str] = {
    # Fast-path routes from query_router.py
    "PHILOSOPHICAL-FAST-PATH": "world_model",
    "MATH-FAST-PATH": "mathematical",
    "CAUSAL-PATH": "causal",
    "IDENTITY-FAST-PATH": "symbolic",
    "CONVERSATIONAL-FAST-PATH": "hybrid",
    "FACTUAL-FAST-PATH": "probabilistic",
    "ANALOGICAL-PATH": "analogical",
    # QueryType enum values from query_router.py
    "philosophical": "world_model",
    "mathematical": "mathematical",
    "causal": "causal",
    "identity": "symbolic",
    "conversational": "hybrid",
    "factual": "probabilistic",
    "general": "hybrid",
    "reasoning": "causal",
    "execution": "symbolic",
    "analogical": "analogical",
    "perception": "analogical",
    "ethical": "world_model",
    # Legacy/fallback mappings
    "HYBRID": "hybrid",
    "UNKNOWN": "hybrid",
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class RoutingDecision:
    """
    Audit trail for tool selection decisions.
    
    Tracks the full history of tool selection decisions across all layers,
    providing transparency and debugging capability for routing issues.
    
    Attributes:
        original_query: The original query string
        router_tools: Tools selected by QueryRouter (Layer 1)
        integration_tools: Tools after ReasoningIntegration (Layer 2)
        final_tools: Final tools after all overrides (Layer 3)
        override_applied: Whether any layer overrode a previous decision
        override_reasons: List of reasons for each override
        decision_history: Full history of decisions at each layer
        timestamp: When the decision was made
    """
    
    original_query: str
    router_tools: List[str]
    integration_tools: List[str]
    final_tools: List[str]
    override_applied: bool = False
    override_reasons: List[str] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize and detect overrides."""
        # Detect if overrides occurred
        if self.router_tools != self.integration_tools:
            self.override_applied = True
            self.decision_history.append({
                "layer": "reasoning_integration",
                "from": self.router_tools,
                "to": self.integration_tools,
                "timestamp": time.time()
            })
        
        if self.integration_tools != self.final_tools:
            self.override_applied = True
            self.decision_history.append({
                "layer": "execution",
                "from": self.integration_tools,
                "to": self.final_tools,
                "timestamp": time.time()
            })
    
    def add_override(
        self, layer: str, from_tools: List[str], to_tools: List[str], reason: str
    ):
        """Record an override decision."""
        self.override_applied = True
        self.override_reasons.append(f"[{layer}] {reason}")
        self.decision_history.append({
            "layer": layer,
            "from": from_tools,
            "to": to_tools,
            "reason": reason,
            "timestamp": time.time()
        })
        
        # Update appropriate tools field
        if layer == "reasoning_integration":
            self.integration_tools = to_tools
        elif layer in ["execution", "agent_pool"]:
            self.final_tools = to_tools
    
    def get_routing_integrity(self) -> float:
        """
        Calculate routing integrity score.
        
        Returns:
            1.0 if no overrides, decreasing with each override
        """
        if not self.override_applied:
            return 1.0
        
        # Penalize each override
        num_overrides = len(self.decision_history)
        return max(0.0, 1.0 - (num_overrides * 0.25))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "original_query": (
                self.original_query[:100] + "..."
                if len(self.original_query) > 100
                else self.original_query
            ),
            "router_tools": self.router_tools,
            "integration_tools": self.integration_tools,
            "final_tools": self.final_tools,
            "override_applied": self.override_applied,
            "override_reasons": self.override_reasons,
            "decision_history": self.decision_history,
            "routing_integrity": self.get_routing_integrity(),
            "timestamp": self.timestamp
        }


@dataclass
class ReasoningResult:
    """
    Result from reasoning module containing tool selection and strategy information.

    This dataclass encapsulates all information about the reasoning decision,
    including which tools were selected, what strategy was used, and metadata
    about the selection process.

    Attributes:
        selected_tools: List of tool names selected for the query
        reasoning_strategy: Name of the reasoning strategy applied
        confidence: Confidence score in the selection (0.0 to 1.0)
        rationale: Human-readable explanation of the selection decision
        metadata: Additional context information about the selection
        routing_decision: Optional audit trail for tool selection
    """

    selected_tools: List[str]
    reasoning_strategy: str
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    routing_decision: Optional[RoutingDecision] = None

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        # Ensure confidence is within valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Ensure we have at least one tool selected
        if not self.selected_tools:
            self.selected_tools = ["general"]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the reasoning result.
        """
        result = {
            "selected_tools": self.selected_tools,
            "reasoning_strategy": self.reasoning_strategy,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
        # Include routing decision if present
        if self.routing_decision:
            result["routing_decision"] = self.routing_decision.to_dict()
        return result


@dataclass
class IntegrationStatistics:
    """
    Statistics for monitoring reasoning integration performance.
    
    Tracks metrics about query processing, tool selection, and execution
    for observability and performance analysis.
    
    Attributes:
        total_queries: Total number of queries processed
        successful_selections: Number of successful tool selections
        failed_selections: Number of failed selections
        tool_selector_hits: Number of times ToolSelector was used
        portfolio_executor_hits: Number of times PortfolioExecutor was used
        avg_selection_time_ms: Average time for tool selection in milliseconds
        strategy_counts: Count of each strategy used
        tool_counts: Count of each tool selected
        error_counts: Count of each error type encountered
        fallback_counts: Count of fallback attempts by reason
    """
    
    total_queries: int = 0
    successful_selections: int = 0
    failed_selections: int = 0
    tool_selector_hits: int = 0
    portfolio_executor_hits: int = 0
    avg_selection_time_ms: float = 0.0
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    tool_counts: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    fallback_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of tool selections."""
        if self.total_queries == 0:
            return 0.0
        return self.successful_selections / self.total_queries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            "total_queries": self.total_queries,
            "successful_selections": self.successful_selections,
            "failed_selections": self.failed_selections,
            "success_rate": self.success_rate,
            "tool_selector_hits": self.tool_selector_hits,
            "portfolio_executor_hits": self.portfolio_executor_hits,
            "avg_selection_time_ms": self.avg_selection_time_ms,
            "strategy_counts": self.strategy_counts,
            "tool_counts": self.tool_counts,
            "error_counts": self.error_counts,
            "fallback_counts": self.fallback_counts,
        }


__all__ = [
    # Constants
    "LOG_PREFIX",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_TIME_BUDGET_MS",
    "DEFAULT_ENERGY_BUDGET_MJ",
    "DEFAULT_MIN_CONFIDENCE",
    "MAX_FALLBACK_ATTEMPTS",
    "MIN_CONFIDENCE_FLOOR",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_GOOD_THRESHOLD",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "CONFIDENCE_LOW_THRESHOLD",
    "ARENA_REASONING_URL",
    "ARENA_DELEGATION_TIMEOUT",
    "FAST_PATH_COMPLEXITY_THRESHOLD",
    "LOW_COMPLEXITY_THRESHOLD",
    "HIGH_COMPLEXITY_THRESHOLD",
    "DECOMPOSITION_COMPLEXITY_THRESHOLD",
    "CAUSAL_REASONING_THRESHOLD",
    "PROBABILISTIC_REASONING_THRESHOLD",
    "MAX_TIMING_SAMPLES",
    # Query Analysis Constants
    "ANALYSIS_INDICATORS",
    "ACTION_VERBS",
    "ETHICAL_ANALYSIS_INDICATORS",
    "PURE_ETHICAL_PHRASES",
    # Enums
    "ReasoningStrategyType",
    # Mappings
    "QUERY_TYPE_STRATEGY_MAP",
    "ROUTE_TO_REASONING_TYPE",
    # Dataclasses
    "RoutingDecision",
    "ReasoningResult",
    "IntegrationStatistics",
]
