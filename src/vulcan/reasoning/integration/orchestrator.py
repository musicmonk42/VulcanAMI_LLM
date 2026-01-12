"""
Main orchestrator for reasoning integration.

Provides the ReasoningIntegration class that coordinates all reasoning
components and implements the primary apply_reasoning() interface.
"""

import atexit
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from .types import (
    ReasoningResult,
    ReasoningStrategyType,
    RoutingDecision,
    IntegrationStatistics,
    DECOMPOSITION_COMPLEXITY_THRESHOLD,
    LOG_PREFIX,
    MAX_FALLBACK_ATTEMPTS,
    DEFAULT_MAX_WORKERS,
    HIGH_COMPLEXITY_THRESHOLD,
    LOW_COMPLEXITY_THRESHOLD,
    CAUSAL_REASONING_THRESHOLD,
    PROBABILISTIC_REASONING_THRESHOLD,
    QUERY_TYPE_STRATEGY_MAP,
)
from .safety_checker import (
    is_false_positive_safety_block,
    is_result_safety_filtered,
    get_safety_filtered_fallback_tools,
)
from .query_router import get_reasoning_type_from_route
from .query_analysis import (
    is_self_referential,
    is_ethical_query,
    consult_world_model_introspection,
)

from .decomposition import process_with_decomposition
from .cross_domain import apply_cross_domain_transfer
from .learning import learn_from_outcome, learn_from_reasoning_outcome

logger = logging.getLogger(__name__)

# Import optional components
try:
    from vulcan.reasoning.selection.tool_selector import ToolSelector
    TOOL_SELECTOR_AVAILABLE = True
except ImportError:
    TOOL_SELECTOR_AVAILABLE = False
    ToolSelector = None

try:
    from vulcan.reasoning.selection.portfolio_executor import PortfolioExecutor
    PORTFOLIO_EXECUTOR_AVAILABLE = True
except ImportError:
    PORTFOLIO_EXECUTOR_AVAILABLE = False
    PortfolioExecutor = None

try:
    from vulcan.reasoning.decomposition.problem_decomposer import ProblemDecomposer
    PROBLEM_DECOMPOSER_AVAILABLE = True
except ImportError:
    PROBLEM_DECOMPOSER_AVAILABLE = False
    ProblemDecomposer = None

try:
    from vulcan.reasoning.unified.query_bridge import QueryBridge
    QUERY_BRIDGE_AVAILABLE = True
except ImportError:
    QUERY_BRIDGE_AVAILABLE = False
    QueryBridge = None

try:
    from vulcan.reasoning.unified.semantic_bridge import SemanticBridge
    SEMANTIC_BRIDGE_AVAILABLE = True
except ImportError:
    SEMANTIC_BRIDGE_AVAILABLE = False
    SemanticBridge = None

try:
    from vulcan.reasoning.unified.domain_bridge import DomainBridge
    DOMAIN_BRIDGE_AVAILABLE = True
except ImportError:
    DOMAIN_BRIDGE_AVAILABLE = False
    DomainBridge = None

class ReasoningIntegration:
    """
    Integrates reasoning module into query processing pipeline.

    This class provides a unified interface for applying reasoning-based tool
    selection and strategy determination. It handles lazy initialization of
    heavy components and provides graceful degradation when components are
    unavailable.

    Thread Safety:
        All methods are thread-safe. Internal components are initialized
        lazily with proper double-checked locking to minimize contention.

    Attributes:
        _tool_selector: Lazy-loaded ToolSelector instance
        _portfolio_executor: Lazy-loaded PortfolioExecutor instance
        _initialized: Whether components have been initialized
        _stats: Statistics tracking object

    Example:
        >>> integration = ReasoningIntegration()
        >>> result = integration.apply_reasoning(
        ...     query="What causes X?",
        ...     query_type="reasoning",
        ...     complexity=0.75
        ... )
        >>> print(f"Strategy: {result.reasoning_strategy}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize reasoning integration with lazy component loading.

        Args:
            config: Optional configuration dictionary with keys:
                - max_workers: Maximum parallel workers (default: 4)
                - time_budget_ms: Time budget in ms (default: 5000)
                - energy_budget_mj: Energy budget in mJ (default: 1000)
                - min_confidence: Minimum confidence (default: 0.5)
                - tool_selector_config: Config passed to ToolSelector
                - enable_decomposition: Enable problem decomposition (default: True)
                - enable_cross_domain_transfer: Enable cross-domain knowledge transfer (default: True)
        """
        self._config = config or {}

        # Lazy-loaded components
        self._tool_selector: Optional[Any] = None
        self._portfolio_executor: Optional[Any] = None
        self._problem_decomposer: Optional[Any] = None
        self._query_bridge: Optional[Any] = None
        self._semantic_bridge: Optional[Any] = None
        self._domain_bridge: Optional[Any] = None

        # Initialization state with thread safety
        self._initialized = False
        self._init_lock = threading.Lock()

        # Statistics tracking with thread safety
        self._stats = IntegrationStatistics()
        self._stats_lock = threading.RLock()

        # Selection timing for performance monitoring
        self._selection_times: List[float] = []

        # Shutdown state
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        
        # Note: Track fallback attempts per query to prevent infinite loops
        # Maps query hash to number of fallback attempts
        self._fallback_attempts: Dict[str, int] = {}
        self._fallback_attempts_lock = threading.Lock()
        
        # Feature configuration
        self._decomposition_enabled = self._config.get('enable_decomposition', True)
        self._cross_domain_enabled = self._config.get('enable_cross_domain_transfer', True)

        logger.info(
            f"{LOG_PREFIX} Initialized (lazy loading enabled, "
            f"decomposition={self._decomposition_enabled}, "
            f"cross_domain={self._cross_domain_enabled})"
        )

    def _should_use_decomposition(self, complexity: float) -> bool:
        """
        Determine if problem decomposition should be used for a query.
        
        Decomposition is used when:
        - Decomposition is enabled in config
        - Query complexity is at or above the threshold (0.40)
        - ProblemDecomposer and QueryBridge are available
        
        Args:
            complexity: Query complexity score (0.0 to 1.0)
            
        Returns:
            True if decomposition should be used, False otherwise
        """
        should_decompose = (
            self._decomposition_enabled
            and complexity >= DECOMPOSITION_COMPLEXITY_THRESHOLD
            and self._problem_decomposer is not None
            and self._query_bridge is not None
        )
        
        # Diagnostic logging for ProblemDecomposer utilization tracking
        if not should_decompose and complexity >= 0.3:
            reasons = []
            if not self._decomposition_enabled:
                reasons.append("decomposition_disabled")
            if complexity < DECOMPOSITION_COMPLEXITY_THRESHOLD:
                reasons.append(f"complexity {complexity:.2f} < threshold {DECOMPOSITION_COMPLEXITY_THRESHOLD}")
            if self._problem_decomposer is None:
                reasons.append("problem_decomposer_unavailable")
            if self._query_bridge is None:
                reasons.append("query_bridge_unavailable")
            logger.debug(
                f"{LOG_PREFIX} Decomposition skipped: {', '.join(reasons)} "
                f"(complexity={complexity:.2f})"
            )
        elif should_decompose:
            logger.info(
                f"{LOG_PREFIX} Decomposition ENABLED: complexity={complexity:.2f} >= "
                f"threshold={DECOMPOSITION_COMPLEXITY_THRESHOLD}"
            )
        
        return should_decompose

    def _is_self_referential(self, query: str) -> bool:
        """
        Check if query is self-referential (asks about the system itself).
        
        Wrapper method that delegates to the imported is_self_referential function.
        
        Args:
            query: The user query to analyze
            
        Returns:
            bool: True if query is self-referential, False otherwise
        """
        return is_self_referential(query)
    
    def _is_ethical_query(self, query: str) -> bool:
        """
        Check if query requires ethical analysis or moral reasoning.
        
        Wrapper method that delegates to the imported is_ethical_query function.
        
        Args:
            query: The user query to analyze
            
        Returns:
            bool: True if query is ethical, False otherwise
        """
        return is_ethical_query(query)
    
    def _consult_world_model_introspection(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Consult world model for introspective queries about system capabilities.
        
        Wrapper method that delegates to the imported consult_world_model_introspection function.
        
        Args:
            query: The user query
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with introspection result if applicable,
                                     None otherwise
        """
        return consult_world_model_introspection(query)

    def _init_components(self) -> None:
        """
        Lazy initialization of reasoning components.

        Uses double-checked locking pattern to ensure thread-safe initialization
        while minimizing lock contention. Components that fail to initialize
        are logged but don't prevent basic operation.

        This method is idempotent and safe to call multiple times.
        """
        # Fast path - already initialized
        if self._initialized:
            return

        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            init_start = time.perf_counter()

            # Try to initialize ToolSelector
            self._tool_selector = self._init_tool_selector()

            # Try to initialize PortfolioExecutor
            self._portfolio_executor = self._init_portfolio_executor()
            
            # Try to initialize ProblemDecomposer and QueryBridge (if decomposition enabled)
            if self._decomposition_enabled:
                self._problem_decomposer = self._init_problem_decomposer()
                self._query_bridge = self._init_query_bridge()
            
            # Try to initialize SemanticBridge and DomainBridge (if cross-domain enabled)
            if self._cross_domain_enabled:
                self._semantic_bridge = self._init_semantic_bridge()
                self._domain_bridge = self._init_domain_bridge()

            init_time = (time.perf_counter() - init_start) * 1000

            self._initialized = True

            logger.info(
                f"{LOG_PREFIX} Components initialized in {init_time:.1f}ms "
                f"(ToolSelector: {self._tool_selector is not None}, "
                f"PortfolioExecutor: {self._portfolio_executor is not None}, "
                f"ProblemDecomposer: {self._problem_decomposer is not None}, "
                f"SemanticBridge: {self._semantic_bridge is not None})"
            )

    def _init_tool_selector(self) -> Optional[Any]:
        """
        Initialize ToolSelector component with error handling.

        PERFORMANCE FIX: Uses singleton from singletons.py to ensure ToolSelector
        is created exactly ONCE per process. This prevents progressive query routing
        degradation where each query creates new instances of:
        - WarmStartPool ("Warm pool initialized with 5 tool pools")
        - StochasticCostModel ("StochasticCostModel initialized")
        - BayesianMemoryPrior with SemanticToolMatcher

        Returns:
            ToolSelector instance if successful, None otherwise.
        """
        try:
            # PERFORMANCE FIX: Use singleton instead of creating new instance
            # This prevents "Tool Selector initialized with 5 tools" appearing
            # multiple times and causing progressive routing time degradation
            from vulcan.reasoning.singletons import get_tool_selector

            selector = get_tool_selector()
            if selector is not None:
                logger.info(f"{LOG_PREFIX} ToolSelector obtained from singleton")
                return selector

            # Fallback: If singleton fails, try direct creation (should be rare)
            logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating ToolSelector directly")
            from vulcan.reasoning.selection.tool_selector import ToolSelector
            selector = ToolSelector(self._config.get("tool_selector_config", {}))
            logger.info(f"{LOG_PREFIX} ToolSelector initialized successfully (fallback)")
            return selector

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} ToolSelector not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} ToolSelector initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_portfolio_executor(self) -> Optional[Any]:
        """
        Initialize PortfolioExecutor component with error handling.

        Returns:
            PortfolioExecutor instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.selection.portfolio_executor import PortfolioExecutor

            max_workers = self._config.get("max_workers", DEFAULT_MAX_WORKERS)
            executor = PortfolioExecutor(tools={}, max_workers=max_workers)
            logger.info(f"{LOG_PREFIX} PortfolioExecutor initialized successfully")
            return executor

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} PortfolioExecutor not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} PortfolioExecutor initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_problem_decomposer(self) -> Optional[Any]:
        """
        Initialize ProblemDecomposer component with error handling.
        
        PERFORMANCE FIX: Uses singleton from singletons.py to ensure ProblemDecomposer
        is created exactly ONCE per process.

        Returns:
            ProblemDecomposer instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.singletons import get_problem_decomposer

            decomposer = get_problem_decomposer()
            if decomposer is not None:
                logger.info(f"{LOG_PREFIX} ProblemDecomposer obtained from singleton")
                return decomposer

            # Fallback: If singleton fails, try direct creation
            logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating ProblemDecomposer directly")
            from vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            decomposer = create_decomposer()
            logger.info(f"{LOG_PREFIX} ProblemDecomposer initialized successfully (fallback)")
            return decomposer

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} ProblemDecomposer not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} ProblemDecomposer initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_query_bridge(self) -> Optional[Any]:
        """
        Initialize QueryToProblemBridge component with error handling.
        
        NOTE: QueryToProblemBridge has been REMOVED as part of architecture simplification.
        The bridge was patching router decomposition issues that are now fixed at root cause.

        Returns:
            None - bridge is no longer used.
        """
        # QueryToProblemBridge removed - return None
        logger.debug(f"{LOG_PREFIX} QueryToProblemBridge removed (architectural simplification)")
        return None

    def _init_semantic_bridge(self) -> Optional[Any]:
        """
        Initialize SemanticBridge component with error handling.
        
        PERFORMANCE FIX: Uses singleton from singletons.py to ensure SemanticBridge
        is created exactly ONCE per process.

        Returns:
            SemanticBridge instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.singletons import get_semantic_bridge

            bridge = get_semantic_bridge()
            if bridge is not None:
                logger.info(f"{LOG_PREFIX} SemanticBridge obtained from singleton")
                return bridge

            # Fallback: If singleton fails, try direct creation
            logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating SemanticBridge directly")
            from vulcan.semantic_bridge import create_semantic_bridge
            bridge = create_semantic_bridge()
            logger.info(f"{LOG_PREFIX} SemanticBridge initialized successfully (fallback)")
            return bridge

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} SemanticBridge not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} SemanticBridge initialization failed: {e}",
                exc_info=True
            )

        return None

    def _init_domain_bridge(self) -> Optional[Any]:
        """
        Initialize ToolDomainBridge component with error handling.

        Returns:
            ToolDomainBridge instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.tool_domain_bridge import get_tool_domain_bridge

            bridge = get_tool_domain_bridge()
            logger.info(f"{LOG_PREFIX} ToolDomainBridge initialized successfully")
            return bridge

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} ToolDomainBridge not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} ToolDomainBridge initialization failed: {e}",
                exc_info=True
            )

        return None

    def _should_use_cross_domain_transfer(
        self,
        selected_tools: List[str],
    ) -> bool:
        """
        Determine if cross-domain knowledge transfer should be used.
        
        Cross-domain transfer is used when:
        - Cross-domain transfer is enabled in config
        - SemanticBridge and DomainBridge are available
        - Query uses tools from multiple domains
        
        Args:
            selected_tools: List of selected tool names
            
        Returns:
            True if cross-domain transfer should be used, False otherwise
        """
        if not self._cross_domain_enabled:
            return False
        
        if self._semantic_bridge is None or self._domain_bridge is None:
            return False
        
        if len(selected_tools) < 2:
            return False
        
        return self._domain_bridge.is_cross_domain_query(selected_tools)

    # apply_reasoning method extracted to apply_reasoning_impl.py
    from .apply_reasoning_impl import apply_reasoning

    def _determine_selection_mode(self, complexity: float, selection_mode_enum: Any) -> Any:
        """
        Determine the selection mode based on query complexity.

        Args:
            complexity: Query complexity score (0.0 to 1.0)
            selection_mode_enum: SelectionMode enum class

        Returns:
            SelectionMode enum value
        """
        if complexity > HIGH_COMPLEXITY_THRESHOLD:
            return selection_mode_enum.ACCURATE
        elif complexity < LOW_COMPLEXITY_THRESHOLD:
            return selection_mode_enum.FAST
        else:
            return selection_mode_enum.BALANCED

    def _determine_strategy_from_query(
        self,
        query_type: str,
        complexity: float
    ) -> str:
        """
        Determine reasoning strategy based on query characteristics.

        This method implements the fallback strategy selection logic when
        the ToolSelector is unavailable or doesn't provide a strategy.

        Args:
            query_type: Type of query (reasoning, perception, planning, etc.)
            complexity: Query complexity (0.0 to 1.0)

        Returns:
            Strategy name string
        """
        # High complexity reasoning queries use causal reasoning
        if query_type == "reasoning" and complexity > CAUSAL_REASONING_THRESHOLD:
            return ReasoningStrategyType.CAUSAL_REASONING.value

        # Execution tasks use planning
        if query_type == "execution":
            return ReasoningStrategyType.PLANNING.value

        # Medium-high complexity uses probabilistic reasoning
        if complexity > PROBABILISTIC_REASONING_THRESHOLD:
            return ReasoningStrategyType.PROBABILISTIC_REASONING.value

        # Query type specific strategies
        type_strategy = QUERY_TYPE_STRATEGY_MAP.get(query_type)
        if type_strategy:
            return type_strategy

        # Default to direct for simple queries
        return ReasoningStrategyType.DIRECT.value

    def _create_default_result(
        self,
        query_type: str,
        complexity: float
    ) -> ReasoningResult:
        """
        Create a default ReasoningResult for fallback scenarios.

        Args:
            query_type: Type of query
            complexity: Query complexity

        Returns:
            Default ReasoningResult
        """
        strategy = self._determine_strategy_from_query(query_type, complexity)

        # Note: Adjust confidence based on complexity
        # High-complexity queries that fall back should have lower confidence
        # because we couldn't handle them properly
        if complexity >= 0.7:
            fallback_confidence = 0.3  # Low confidence for high-complexity fallbacks
        elif complexity >= 0.5:
            fallback_confidence = 0.4  # Medium-low confidence
        else:
            fallback_confidence = 0.5  # Default confidence for low-complexity

        return ReasoningResult(
            selected_tools=["general"],
            reasoning_strategy=strategy,
            confidence=fallback_confidence,
            rationale="Fallback to default strategy",
            metadata={
                "query_type": query_type,
                "complexity": complexity,
                "fallback": True,
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current integration statistics.

        Returns:
            Dictionary containing:
                - initialized: Initialization status
                - tool_selector_available: ToolSelector availability
                - portfolio_executor_available: PortfolioExecutor availability
                - invocations: Total invocation count
                - tool_selections: Tool selection count
                - portfolio_executions: Portfolio execution count
                - errors: Error count
                - success_rate: Success rate (0.0 to 1.0)
                - fast_path_count: Fast path usage count
                - avg_selection_time_ms: Average selection time
                - last_error: Last error message if any

        Example:
            >>> stats = integration.get_statistics()
            >>> print(f"Success rate: {stats['success_rate']:.1%}")
            >>> print(f"Avg selection time: {stats['avg_selection_time_ms']:.1f}ms")
        """
        with self._stats_lock:
            return {
                "initialized": self._initialized,
                "tool_selector_available": self._tool_selector is not None,
                "portfolio_executor_available": self._portfolio_executor is not None,
                "invocations": self._stats.invocations,
                "tool_selections": self._stats.tool_selections,
                "portfolio_executions": self._stats.portfolio_executions,
                "errors": self._stats.errors,
                "success_rate": self._stats.success_rate,
                "fast_path_count": self._stats.fast_path_count,
                "avg_selection_time_ms": self._stats.avg_selection_time_ms,
                "last_error": self._stats.last_error,
            }

    def reset_statistics(self) -> None:
        """
        Reset all statistics to initial values.

        Useful for testing or when starting a new monitoring period.
        """
        with self._stats_lock:
            self._stats = IntegrationStatistics()
            self._selection_times.clear()

        logger.info(f"{LOG_PREFIX} Statistics reset")

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Shutdown reasoning components gracefully.

        Releases resources held by the ToolSelector and PortfolioExecutor.
        After shutdown, the integration can no longer be used.

        Args:
            timeout: Maximum time to wait for shutdown in seconds

        Example:
            >>> integration.shutdown(timeout=10.0)
        """
        with self._shutdown_lock:
            if self._shutdown:
                logger.warning(f"{LOG_PREFIX} Already shutdown")
                return
            self._shutdown = True

        logger.info(f"{LOG_PREFIX} Starting shutdown (timeout={timeout}s)")

        shutdown_start = time.perf_counter()

        # Shutdown ToolSelector
        if self._tool_selector is not None:
            try:
                if hasattr(self._tool_selector, "shutdown"):
                    remaining = timeout - (time.perf_counter() - shutdown_start)
                    self._tool_selector.shutdown(timeout=max(0.1, remaining))
                    logger.debug(f"{LOG_PREFIX} ToolSelector shutdown complete")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} ToolSelector shutdown error: {e}")

        # Shutdown PortfolioExecutor
        if self._portfolio_executor is not None:
            try:
                if hasattr(self._portfolio_executor, "shutdown"):
                    remaining = timeout - (time.perf_counter() - shutdown_start)
                    self._portfolio_executor.shutdown(timeout=max(0.1, remaining))
                    logger.debug(f"{LOG_PREFIX} PortfolioExecutor shutdown complete")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} PortfolioExecutor shutdown error: {e}")

        shutdown_time = (time.perf_counter() - shutdown_start) * 1000
        logger.info(f"{LOG_PREFIX} Shutdown complete in {shutdown_time:.1f}ms")


