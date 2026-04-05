"""
VULCAN Tool Selection Submodule

This package provides a comprehensive, production-grade system for intelligent
tool selection. It integrates multiple components to make decisions that balance
performance, cost, safety, and quality.

Components:
- ToolSelector: The main orchestrator that runs the entire selection pipeline.
- PortfolioExecutor: Manages the concurrent or sequential execution of tools.
- UtilityModel: Calculates the "value" of a potential outcome based on context.
- StochasticCostModel: An ML model that predicts the time and energy cost of tools.
- BayesianMemoryPrior: Uses historical data to form a Bayesian prior over which
  tool is most likely to succeed.
- SafetyGovernor: Enforces safety constraints and tool contracts.
- AdmissionControl: Acts as a gatekeeper to prevent system overload.
- SelectionCache: A multi-level cache to store results and avoid redundant work.
- WarmStartPool: Manages a pool of pre-warmed tool instances to reduce latency.
"""

from .admission_control import AdmissionControlIntegration, RequestPriority
from .cost_model import CostComponent, CostEstimate, StochasticCostModel
from .memory_prior import BayesianMemoryPrior, PriorType
from .portfolio_executor import (
    ExecutionMonitor,
    ExecutionStrategy,
    PortfolioExecutor,
    PortfolioResult,
)
from .safety_governor import SafetyGovernor, SafetyLevel, ToolContract, VetoReason
from .selection_cache import SelectionCache
from .selection_types import (
    SelectionMode,
    SelectionRequest,
    SelectionResult,
)
from .tool_selector import (
    ToolSelector,
    create_tool_selector,
)
from .utility_model import ContextMode, UtilityContext, UtilityModel
from .warm_pool import WarmStartPool

# Optional components that might not be available
try:
    from ..contextual_bandit import AdaptiveBanditOrchestrator, BanditContext

    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

try:
    from .semantic_tool_matcher import SemanticToolMatcher, TOOL_DESCRIPTIONS, TOOL_KEYWORDS
    SEMANTIC_MATCHER_AVAILABLE = True
except ImportError:
    SEMANTIC_MATCHER_AVAILABLE = False

try:
    from .embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
        get_embedding_circuit_breaker,
        get_circuit_breaker_stats,
        reset_embedding_circuit_breaker,
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

# Mathematical verification availability (from tool_selector)
try:
    from .tool_selector import MATH_VERIFICATION_AVAILABLE
except ImportError:
    MATH_VERIFICATION_AVAILABLE = False

# LLM Router for tool selection
try:
    from vulcan.routing.llm_router import (
        get_llm_router,
        route_query,
        RoutingDecision,
    )
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    get_llm_router = None
    route_query = None
    RoutingDecision = None


__all__ = [
    # Main Orchestrator
    "ToolSelector",
    "SelectionRequest",
    "SelectionResult",
    "SelectionMode",
    "create_tool_selector",
    # Core Components
    "PortfolioExecutor",
    "ExecutionStrategy",
    "PortfolioResult",
    "ExecutionMonitor",
    "UtilityModel",
    "ContextMode",
    "UtilityContext",
    "StochasticCostModel",
    "CostComponent",
    "CostEstimate",
    "BayesianMemoryPrior",
    "PriorType",
    "SafetyGovernor",
    "SafetyLevel",
    "VetoReason",
    "ToolContract",
    "AdmissionControlIntegration",
    "RequestPriority",
    "SelectionCache",
    "WarmStartPool",
    # Availability Flags
    "BANDIT_AVAILABLE",
    "SEMANTIC_MATCHER_AVAILABLE",
    "CIRCUIT_BREAKER_AVAILABLE",
    "MATH_VERIFICATION_AVAILABLE",
    "LLM_ROUTER_AVAILABLE",
]

# Add optional components to __all__ if they were imported successfully
if BANDIT_AVAILABLE:
    __all__.extend(["AdaptiveBanditOrchestrator", "BanditContext"])

if SEMANTIC_MATCHER_AVAILABLE:
    __all__.extend(["SemanticToolMatcher", "TOOL_DESCRIPTIONS", "TOOL_KEYWORDS"])

if CIRCUIT_BREAKER_AVAILABLE:
    __all__.extend([
        "EmbeddingCircuitBreaker",
        "get_embedding_circuit_breaker",
        "get_circuit_breaker_stats",
        "reset_embedding_circuit_breaker",
    ])

if LLM_ROUTER_AVAILABLE:
    __all__.extend(["get_llm_router", "route_query", "RoutingDecision"])
