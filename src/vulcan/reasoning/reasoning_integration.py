"""
Reasoning Integration Layer for VULCAN-AGI System.

Part of the VULCAN-AGI system.

This module provides the integration layer between the query processing pipeline
and the reasoning subsystem. It wires the ToolSelector, PortfolioExecutor, and
reasoning strategies into a unified interface for intelligent tool selection
and query processing.

Key Features:
    - Lazy initialization of reasoning components for fast startup
    - Thread-safe singleton pattern with double-checked locking
    - Graceful degradation when components are unavailable
    - Intelligent strategy selection based on query characteristics
    - Portfolio execution for complex multi-tool queries
    - Comprehensive statistics tracking for observability
    - Configurable budgets for time, energy, and confidence thresholds

Performance Characteristics:
    - First invocation triggers lazy initialization (~100-500ms)
    - Subsequent invocations are fast (<10ms for simple queries)
    - Fast-path optimization for low-complexity queries
    - Thread-safe with minimal lock contention

Components Integrated:
    - ToolSelector: Multi-armed bandit-based intelligent tool selection
    - PortfolioExecutor: Parallel and sequential multi-tool execution
    - SelectionCache: LRU caching for repeated queries
    - SafetyGovernor: Safety validation for tool outputs

Usage:
    # Simple usage via convenience functions
    from vulcan.reasoning.reasoning_integration import apply_reasoning

    result = apply_reasoning(
        query="Explain the causal relationship between X and Y",
        query_type="reasoning",
        complexity=0.75,
    )

    print(f"Selected tools: {result.selected_tools}")
    print(f"Strategy: {result.reasoning_strategy}")
    print(f"Confidence: {result.confidence:.2f}")

    # Portfolio execution for complex queries
    from vulcan.reasoning.reasoning_integration import run_portfolio_reasoning

    portfolio_result = run_portfolio_reasoning(
        query="Complex multi-step problem",
        tools=["symbolic", "causal", "probabilistic"],
        strategy="causal_reasoning",
    )

    # Get statistics for monitoring
    from vulcan.reasoning.reasoning_integration import get_reasoning_statistics

    stats = get_reasoning_statistics()
    print(f"Success rate: {stats['success_rate']:.1%}")

Thread Safety:
    All public functions and methods are thread-safe. The module uses a
    singleton pattern with proper locking to ensure safe concurrent access.

Error Handling:
    The module follows a graceful degradation pattern. If the ToolSelector
    or PortfolioExecutor are unavailable, the module falls back to default
    strategies without raising exceptions.
"""

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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

# Complexity thresholds for strategy selection
FAST_PATH_COMPLEXITY_THRESHOLD = 0.3  # Below this, use fast path
LOW_COMPLEXITY_THRESHOLD = 0.4  # Below this, use FAST mode
HIGH_COMPLEXITY_THRESHOLD = 0.7  # Above this, use ACCURATE mode
DECOMPOSITION_COMPLEXITY_THRESHOLD = 0.40  # At or above this, use decomposition

# Strategy selection thresholds
CAUSAL_REASONING_THRESHOLD = 0.6  # Complexity threshold for causal reasoning
PROBABILISTIC_REASONING_THRESHOLD = 0.5  # Complexity threshold for probabilistic

# Maximum timing samples to keep for statistics
MAX_TIMING_SAMPLES = 100


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
    DEFAULT = "default"


# Maps query types to appropriate reasoning strategies
QUERY_TYPE_STRATEGY_MAP: Dict[str, str] = {
    "reasoning": "causal_reasoning",
    "execution": "planning",
    "perception": "analogical_reasoning",
    "planning": "deliberative",
    "learning": "meta_reasoning",
    "general": "direct",
}


@dataclass
class ReasoningResult:
    """
    Result from reasoning module containing tool selection and strategy information.

    This dataclass encapsulates all information about the reasoning decision,
    including which tools were selected, what strategy was used, and metadata
    about the selection process.

    Attributes:
        selected_tools: List of tool names selected for the query.
            Example: ["symbolic", "causal"]
        reasoning_strategy: Name of the reasoning strategy applied.
            Example: "causal_reasoning"
        confidence: Confidence score in the selection (0.0 to 1.0).
            Higher values indicate more reliable selections.
        rationale: Human-readable explanation of the selection decision.
            Useful for debugging and transparency.
        metadata: Additional context information about the selection.
            Contains timing, complexity, and component availability info.

    Example:
        >>> result = ReasoningResult(
        ...     selected_tools=["causal"],
        ...     reasoning_strategy="causal_reasoning",
        ...     confidence=0.85,
        ...     rationale="High complexity reasoning query",
        ...     metadata={"complexity": 0.75, "query_type": "reasoning"}
        ... )
    """

    selected_tools: List[str]
    reasoning_strategy: str
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        return {
            "selected_tools": self.selected_tools,
            "reasoning_strategy": self.reasoning_strategy,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass
class IntegrationStatistics:
    """
    Statistics for monitoring reasoning integration performance.

    Thread-safe dataclass for tracking performance metrics and health
    indicators of the reasoning integration layer.

    Attributes:
        invocations: Total number of reasoning invocations
        tool_selections: Number of successful tool selections via ToolSelector
        portfolio_executions: Number of portfolio executions completed
        errors: Number of errors encountered during processing
        fast_path_count: Number of queries using the fast path optimization
        avg_selection_time_ms: Rolling average time for tool selection
        last_error: Description of the most recent error (for debugging)
    """

    invocations: int = 0
    tool_selections: int = 0
    portfolio_executions: int = 0
    errors: int = 0
    fast_path_count: int = 0
    avg_selection_time_ms: float = 0.0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate as ratio of successful operations.

        Returns:
            Success rate between 0.0 and 1.0
        """
        if self.invocations == 0:
            return 0.0
        return (self.invocations - self.errors) / self.invocations


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
        """
        self._config = config or {}

        # Lazy-loaded components
        self._tool_selector: Optional[Any] = None
        self._portfolio_executor: Optional[Any] = None
        self._problem_decomposer: Optional[Any] = None
        self._query_bridge: Optional[Any] = None

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
        
        # Decomposition configuration
        self._decomposition_enabled = self._config.get('enable_decomposition', True)

        logger.info(f"{LOG_PREFIX} Initialized (lazy loading enabled, decomposition={self._decomposition_enabled})")

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

            init_time = (time.perf_counter() - init_start) * 1000

            self._initialized = True

            logger.info(
                f"{LOG_PREFIX} Components initialized in {init_time:.1f}ms "
                f"(ToolSelector: {self._tool_selector is not None}, "
                f"PortfolioExecutor: {self._portfolio_executor is not None}, "
                f"ProblemDecomposer: {self._problem_decomposer is not None})"
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

        Returns:
            QueryToProblemBridge instance if successful, None otherwise.
        """
        try:
            from vulcan.reasoning.query_to_problem_bridge import get_query_to_problem_bridge

            bridge = get_query_to_problem_bridge()
            logger.info(f"{LOG_PREFIX} QueryToProblemBridge initialized successfully")
            return bridge

        except ImportError as e:
            logger.warning(
                f"{LOG_PREFIX} QueryToProblemBridge not available (missing dependency): {e}"
            )
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} QueryToProblemBridge initialization failed: {e}",
                exc_info=True
            )

        return None

    def apply_reasoning(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Apply reasoning to select tools and determine strategy for a query.

        This is the main entry point for applying reasoning-based tool selection.
        It analyzes the query characteristics and uses the ToolSelector (if
        available) to determine the best tools and reasoning strategy.

        Args:
            query: The user query text to process.
            query_type: Type of query from the router. Valid types include:
                - "general": General knowledge queries
                - "reasoning": Logical reasoning queries
                - "execution": Action/task execution queries
                - "perception": Pattern recognition queries
                - "planning": Multi-step planning queries
                - "learning": Knowledge acquisition queries
            complexity: Query complexity score (0.0 to 1.0).
                - 0.0-0.3: Simple queries (fast path)
                - 0.3-0.7: Medium complexity
                - 0.7-1.0: High complexity (full analysis)
            context: Optional context dictionary containing:
                - conversation_id: ID of the conversation
                - history: Previous messages in conversation
                - user_preferences: User-specific settings

        Returns:
            ReasoningResult with selected tools, strategy, and metadata.

        Raises:
            No exceptions are raised. Errors result in fallback to default strategy.

        Example:
            >>> integration = ReasoningIntegration()
            >>> result = integration.apply_reasoning(
            ...     query="Explain quantum entanglement",
            ...     query_type="reasoning",
            ...     complexity=0.8,
            ...     context={"conversation_id": "conv_123"}
            ... )
            >>> print(result.selected_tools)
            ['causal', 'probabilistic']
        """
        # Check shutdown state
        with self._shutdown_lock:
            if self._shutdown:
                logger.warning(f"{LOG_PREFIX} Called after shutdown, returning default")
                return self._create_default_result(query_type, complexity)

        # Initialize components if needed
        self._init_components()

        # Track invocation
        selection_start = time.perf_counter()
        with self._stats_lock:
            self._stats.invocations += 1

        try:
            # Fast path for simple queries - skip heavy tool selection
            if complexity < FAST_PATH_COMPLEXITY_THRESHOLD:
                with self._stats_lock:
                    self._stats.fast_path_count += 1

                return ReasoningResult(
                    selected_tools=["general"],
                    reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                    confidence=0.9,
                    rationale="Simple query - using fast path direct response",
                    metadata={
                        "fast_path": True,
                        "complexity": complexity,
                        "query_type": query_type,
                        "selection_time_ms": 0.0,
                    },
                )

            # Check if we should use decomposition for complex queries
            use_decomposition = (
                self._decomposition_enabled
                and complexity >= DECOMPOSITION_COMPLEXITY_THRESHOLD
                and self._problem_decomposer is not None
                and self._query_bridge is not None
            )
            
            if use_decomposition:
                # Use decomposition path for complex queries
                logger.info(
                    f"{LOG_PREFIX} Using decomposition path (complexity={complexity:.2f} >= "
                    f"{DECOMPOSITION_COMPLEXITY_THRESHOLD})"
                )
                result = self._process_with_decomposition(
                    query, query_type, complexity, context
                )
            else:
                # Use direct tool selection for simpler queries
                result = self._select_with_tool_selector(
                    query, query_type, complexity, context
                )

            # Record timing
            selection_time = (time.perf_counter() - selection_start) * 1000
            self._record_selection_time(selection_time)

            # Add timing to metadata
            result.metadata["selection_time_ms"] = selection_time

            return result

        except Exception as e:
            # Record error and return fallback
            with self._stats_lock:
                self._stats.errors += 1
                self._stats.last_error = str(e)

            logger.error(
                f"{LOG_PREFIX} Reasoning application failed: {e}",
                exc_info=True
            )

            return self._create_default_result(query_type, complexity)

    def _select_with_tool_selector(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]],
    ) -> ReasoningResult:
        """
        Perform tool selection using the ToolSelector component.

        Args:
            query: The user query
            query_type: Type of query
            complexity: Complexity score
            context: Optional context

        Returns:
            ReasoningResult from tool selection or fallback
        """
        # Default values for fallback
        selected_tools = ["general"]
        reasoning_strategy = ReasoningStrategyType.DEFAULT.value
        confidence = 0.7
        rationale = "Default reasoning strategy"

        # Try to use ToolSelector if available
        if self._tool_selector is not None:
            try:
                # Import selection components
                from vulcan.reasoning.selection.tool_selector import (
                    SelectionRequest,
                    SelectionMode,
                )

                # Determine selection mode based on complexity
                mode = self._determine_selection_mode(complexity, SelectionMode)

                # Build constraints
                constraints = {
                    "time_budget_ms": self._config.get(
                        "time_budget_ms", DEFAULT_TIME_BUDGET_MS
                    ),
                    "energy_budget_mj": self._config.get(
                        "energy_budget_mj", DEFAULT_ENERGY_BUDGET_MJ
                    ),
                    "min_confidence": self._config.get(
                        "min_confidence", DEFAULT_MIN_CONFIDENCE
                    ),
                }

                # Create selection request
                request = SelectionRequest(
                    problem=query,
                    constraints=constraints,
                    mode=mode,
                    context=context or {},
                )

                # Execute selection
                result = self._tool_selector.select_and_execute(request)

                # Extract tools from result
                # NOTE: SelectionResult has 'selected_tool' (singular) and 'all_results'
                # We extract the list of tools from all_results keys or use selected_tool
                selected_tools = self._extract_tools_from_result(result)

                # Safely extract strategy and confidence with fallbacks
                if hasattr(result, "strategy_used") and result.strategy_used is not None:
                    reasoning_strategy = result.strategy_used.value
                    rationale = f"ToolSelector selected via {result.strategy_used.value} strategy"
                else:
                    reasoning_strategy = ReasoningStrategyType.DEFAULT.value
                    rationale = "ToolSelector selection (strategy unknown)"

                if hasattr(result, "calibrated_confidence"):
                    confidence = result.calibrated_confidence
                elif hasattr(result, "confidence"):
                    confidence = result.confidence
                else:
                    confidence = 0.7  # Default confidence

                # Track successful selection
                with self._stats_lock:
                    self._stats.tool_selections += 1

                logger.info(
                    f"{LOG_PREFIX} Tool selection complete: "
                    f"tools={selected_tools}, strategy={reasoning_strategy}, "
                    f"confidence={confidence:.2f}"
                )

            except ImportError as e:
                logger.warning(f"{LOG_PREFIX} ToolSelector imports unavailable: {e}")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} ToolSelector execution failed: {e}")
                with self._stats_lock:
                    self._stats.errors += 1

        # If no strategy was selected, determine based on query type
        if reasoning_strategy == ReasoningStrategyType.DEFAULT.value:
            reasoning_strategy = self._determine_strategy_from_query(
                query_type, complexity
            )

        return ReasoningResult(
            selected_tools=selected_tools,
            reasoning_strategy=reasoning_strategy,
            confidence=confidence,
            rationale=rationale,
            metadata={
                "query_type": query_type,
                "complexity": complexity,
                "tool_selector_available": self._tool_selector is not None,
                "portfolio_executor_available": self._portfolio_executor is not None,
            },
        )

    def _process_with_decomposition(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]],
    ) -> ReasoningResult:
        """
        Process a complex query using hierarchical problem decomposition.

        This method is called for queries with complexity >= DECOMPOSITION_COMPLEXITY_THRESHOLD.
        It breaks down the query into subproblems, applies tool selection to each,
        and aggregates the results.

        Processing Flow:
            1. Convert query to ProblemGraph via QueryToProblemBridge
            2. Decompose using ProblemDecomposer (strategies: exact, semantic, structural, etc.)
            3. For each subproblem step, apply ToolSelector
            4. Aggregate results and determine overall strategy

        Args:
            query: The user query text to process
            query_type: Type of query (reasoning, execution, etc.)
            complexity: Query complexity score (0.4 to 1.0)
            context: Optional context dictionary

        Returns:
            ReasoningResult with selected tools, strategy, and decomposition metadata

        Note:
            Falls back to direct tool selection if decomposition fails.
        """
        decomposition_start = time.perf_counter()

        try:
            # Step 1: Convert query to ProblemGraph
            query_analysis = {
                'type': query_type,
                'complexity': complexity,
                'uncertainty': context.get('uncertainty', 0.0) if context else 0.0,
                'requires_reasoning': query_type in ('reasoning', 'causal', 'planning'),
            }

            problem_graph = self._query_bridge.convert_to_problem_graph(
                query=query,
                query_analysis=query_analysis,
                tool_selection=None,  # Will be determined per subproblem
            )

            if problem_graph is None:
                logger.warning(
                    f"{LOG_PREFIX} Query bridge returned None, falling back to direct selection"
                )
                return self._select_with_tool_selector(query, query_type, complexity, context)

            # Step 2: Decompose the problem
            decomposition_plan = self._problem_decomposer.decompose_novel_problem(problem_graph)

            if decomposition_plan is None or len(decomposition_plan.steps) == 0:
                logger.warning(
                    f"{LOG_PREFIX} Decomposition returned empty plan, falling back to direct selection"
                )
                return self._select_with_tool_selector(query, query_type, complexity, context)

            logger.info(
                f"{LOG_PREFIX} Decomposed into {len(decomposition_plan.steps)} steps, "
                f"confidence={decomposition_plan.confidence:.2f}"
            )

            # Step 3: Apply tool selection to each subproblem step
            all_tools: set = set()
            step_results: List[Dict[str, Any]] = []
            total_step_confidence = 0.0

            for step in decomposition_plan.steps:
                # Extract step description
                if hasattr(step, 'description'):
                    step_query = step.description
                elif hasattr(step, 'to_dict'):
                    step_dict = step.to_dict()
                    step_query = step_dict.get('description', str(step))
                else:
                    step_query = str(step)

                # Extract step complexity
                if hasattr(step, 'estimated_complexity'):
                    step_complexity = step.estimated_complexity
                elif hasattr(step, 'complexity'):
                    step_complexity = step.complexity
                else:
                    step_complexity = complexity * 0.5  # Default to half of parent

                # Ensure step_complexity is within bounds
                step_complexity = max(0.1, min(1.0, step_complexity))

                # Apply tool selection to this step
                step_result = self._select_with_tool_selector(
                    query=step_query,
                    query_type=query_type,
                    complexity=step_complexity,
                    context=context,
                )

                # Collect tools from this step
                all_tools.update(step_result.selected_tools)
                total_step_confidence += step_result.confidence

                step_results.append({
                    'step_id': getattr(step, 'step_id', f'step_{len(step_results)}'),
                    'description': step_query[:100],  # Truncate for metadata
                    'tools': step_result.selected_tools,
                    'strategy': step_result.reasoning_strategy,
                    'confidence': step_result.confidence,
                })

            # Step 4: Determine overall strategy based on decomposition
            if decomposition_plan.strategy:
                strategy_name = getattr(decomposition_plan.strategy, 'name', 'hierarchical')
            else:
                strategy_name = 'hierarchical_decomposition'

            # Calculate overall confidence
            num_steps = len(step_results)
            avg_step_confidence = total_step_confidence / num_steps if num_steps > 0 else 0.5
            overall_confidence = (decomposition_plan.confidence * 0.4) + (avg_step_confidence * 0.6)

            decomposition_time_ms = (time.perf_counter() - decomposition_start) * 1000

            logger.info(
                f"{LOG_PREFIX} Decomposition complete: "
                f"tools={list(all_tools)}, strategy={strategy_name}, "
                f"confidence={overall_confidence:.2f}, time={decomposition_time_ms:.1f}ms"
            )

            return ReasoningResult(
                selected_tools=list(all_tools) if all_tools else ["general"],
                reasoning_strategy=strategy_name,
                confidence=overall_confidence,
                rationale=f"Hierarchical decomposition into {num_steps} subproblems",
                metadata={
                    "query_type": query_type,
                    "complexity": complexity,
                    "decomposition_path": True,
                    "decomposition_steps": num_steps,
                    "step_results": step_results,
                    "decomposition_confidence": decomposition_plan.confidence,
                    "decomposition_time_ms": decomposition_time_ms,
                    "tool_selector_available": self._tool_selector is not None,
                    "problem_decomposer_available": self._problem_decomposer is not None,
                },
            )

        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} Decomposition processing failed: {e}, "
                f"falling back to direct selection",
                exc_info=True
            )
            # Graceful degradation: fall back to direct tool selection
            return self._select_with_tool_selector(query, query_type, complexity, context)

    def _extract_tools_from_result(self, result: Any) -> List[str]:
        """
        Extract list of tools from SelectionResult.

        SelectionResult has 'selected_tool' (singular str) and 'all_results' (dict).
        We extract tools from all_results if available, otherwise use selected_tool.

        Args:
            result: SelectionResult from ToolSelector

        Returns:
            List of tool names used
        """
        # Try to get tools from all_results dictionary keys
        if hasattr(result, "all_results") and result.all_results:
            tools = list(result.all_results.keys())
            if tools:
                return tools

        # Fall back to selected_tool (singular)
        if hasattr(result, "selected_tool"):
            tool = result.selected_tool
            if tool and tool != "none":
                return [tool]

        # Default fallback
        return ["general"]

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

        return ReasoningResult(
            selected_tools=["general"],
            reasoning_strategy=strategy,
            confidence=0.5,
            rationale="Fallback to default strategy",
            metadata={
                "query_type": query_type,
                "complexity": complexity,
                "fallback": True,
            },
        )

    def _record_selection_time(self, time_ms: float) -> None:
        """
        Record selection timing for performance monitoring.

        Maintains a rolling window of timing samples for average calculation.

        Args:
            time_ms: Selection time in milliseconds
        """
        with self._stats_lock:
            self._selection_times.append(time_ms)

            # Maintain rolling window
            if len(self._selection_times) > MAX_TIMING_SAMPLES:
                self._selection_times.pop(0)

            # Update average
            if self._selection_times:
                self._stats.avg_selection_time_ms = (
                    sum(self._selection_times) / len(self._selection_times)
                )

    def run_portfolio(
        self,
        query: str,
        tools: List[str],
        strategy: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run portfolio execution for complex queries requiring multiple tools.

        Executes multiple tools using the specified strategy for queries that
        benefit from diverse reasoning approaches. The portfolio executor
        coordinates parallel or sequential execution based on the strategy.

        Args:
            query: The user query to process
            tools: List of tool names to use. Available tools include:
                - "symbolic": Symbolic/logical reasoning
                - "probabilistic": Statistical inference
                - "causal": Causal reasoning and inference
                - "analogical": Pattern matching and analogy
                - "multimodal": Multi-modal processing
            strategy: Execution strategy name. Options include:
                - "causal_reasoning": Sequential refinement
                - "probabilistic_reasoning": Speculative parallel
                - "analogical_reasoning": Cascade
                - "planning": Sequential refinement
                - "deliberative": Committee consensus
                - "direct": Single tool execution
            constraints: Optional execution constraints:
                - time_budget_ms: Maximum execution time
                - energy_budget_mj: Maximum energy budget
                - min_confidence: Minimum result confidence

        Returns:
            Dictionary with execution results:
                - status: "success", "skipped", or "error"
                - strategy_used: Actual strategy applied
                - tools_used: List of tools that executed
                - execution_time_ms: Total execution time
                - confidence: Result confidence score
                - error: Error message if status is "error"

        Example:
            >>> result = integration.run_portfolio(
            ...     query="Complex multi-step problem",
            ...     tools=["symbolic", "causal"],
            ...     strategy="causal_reasoning"
            ... )
            >>> print(f"Status: {result['status']}")
        """
        # Check shutdown state
        with self._shutdown_lock:
            if self._shutdown:
                return {"status": "skipped", "reason": "integration_shutdown"}

        # Initialize components if needed
        self._init_components()

        # Check if portfolio executor is available
        if self._portfolio_executor is None:
            logger.warning(f"{LOG_PREFIX} PortfolioExecutor not available")
            return {"status": "skipped", "reason": "executor_unavailable"}

        try:
            from vulcan.reasoning.selection.portfolio_executor import (
                ExecutionStrategy,
                ExecutionMonitor,
            )

            # Map strategy string to enum
            exec_strategy = self._map_strategy_to_execution(strategy, ExecutionStrategy)

            # Build constraints
            merged_constraints = {
                "time_budget_ms": DEFAULT_TIME_BUDGET_MS,
                "energy_budget_mj": DEFAULT_ENERGY_BUDGET_MJ,
                "min_confidence": DEFAULT_MIN_CONFIDENCE,
            }
            if constraints:
                merged_constraints.update(constraints)

            # Create execution monitor
            monitor = ExecutionMonitor(
                time_budget_ms=merged_constraints["time_budget_ms"],
                energy_budget_mj=merged_constraints["energy_budget_mj"],
                min_confidence=merged_constraints["min_confidence"],
            )

            # Execute portfolio
            exec_start = time.perf_counter()
            result = self._portfolio_executor.execute(
                strategy=exec_strategy,
                tool_names=tools,
                problem=query,
                constraints=merged_constraints,
                monitor=monitor,
            )
            exec_time = (time.perf_counter() - exec_start) * 1000

            # Track execution
            with self._stats_lock:
                self._stats.portfolio_executions += 1

            logger.info(
                f"{LOG_PREFIX} Portfolio execution complete: "
                f"strategy={result.strategy.value}, "
                f"tools={result.tools_used}, "
                f"time={exec_time:.1f}ms"
            )

            return {
                "status": "success",
                "strategy_used": result.strategy.value,
                "tools_used": result.tools_used,
                "execution_time_ms": exec_time,
                "confidence": (
                    result.consensus_confidence
                    if result.consensus_confidence is not None
                    else 0.5
                ),
                "primary_result": result.primary_result,
                "all_results": result.all_results,
            }

        except ImportError as e:
            logger.warning(f"{LOG_PREFIX} Portfolio imports unavailable: {e}")
            return {"status": "error", "error": f"Import error: {e}"}

        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} Portfolio execution failed: {e}",
                exc_info=True
            )
            with self._stats_lock:
                self._stats.errors += 1
                self._stats.last_error = str(e)

            return {"status": "error", "error": str(e)}

    def _map_strategy_to_execution(
        self,
        strategy: str,
        execution_strategy_enum: Any
    ) -> Any:
        """
        Map strategy string to ExecutionStrategy enum value.

        Args:
            strategy: Strategy name string
            execution_strategy_enum: ExecutionStrategy enum class

        Returns:
            ExecutionStrategy enum value
        """
        strategy_map = {
            "causal_reasoning": execution_strategy_enum.SEQUENTIAL_REFINEMENT,
            "probabilistic_reasoning": execution_strategy_enum.SPECULATIVE_PARALLEL,
            "analogical_reasoning": execution_strategy_enum.CASCADE,
            "planning": execution_strategy_enum.SEQUENTIAL_REFINEMENT,
            "deliberative": execution_strategy_enum.COMMITTEE_CONSENSUS,
            "direct": execution_strategy_enum.SINGLE,
        }

        return strategy_map.get(strategy, execution_strategy_enum.ADAPTIVE_MIX)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics for monitoring and observability.

        Returns a comprehensive dictionary of statistics about the reasoning
        integration's performance and health.

        Returns:
            Dictionary containing:
                - initialized: Whether components are initialized
                - tool_selector_available: ToolSelector availability
                - portfolio_executor_available: PortfolioExecutor availability
                - invocations: Total reasoning invocations
                - tool_selections: Successful tool selections
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


# =============================================================================
# Global Singleton Management
# =============================================================================

# Global singleton instance
_reasoning_integration: Optional[ReasoningIntegration] = None
_integration_lock = threading.Lock()


def get_reasoning_integration(
    config: Optional[Dict[str, Any]] = None
) -> ReasoningIntegration:
    """
    Get or create the global reasoning integration singleton.

    Uses double-checked locking pattern for thread-safe lazy initialization.

    Args:
        config: Optional configuration dictionary (only used on first call)

    Returns:
        ReasoningIntegration singleton instance

    Example:
        >>> integration = get_reasoning_integration()
        >>> result = integration.apply_reasoning(...)
    """
    global _reasoning_integration

    if _reasoning_integration is None:
        with _integration_lock:
            if _reasoning_integration is None:
                _reasoning_integration = ReasoningIntegration(config)

    return _reasoning_integration


def _shutdown_on_exit() -> None:
    """Atexit handler to shutdown integration gracefully."""
    global _reasoning_integration

    if _reasoning_integration is not None:
        try:
            _reasoning_integration.shutdown(timeout=2.0)
        except Exception:
            pass  # Ignore errors during exit shutdown


# Register atexit handler for graceful shutdown
atexit.register(_shutdown_on_exit)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_reasoning(
    query: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Convenience function to apply reasoning using the global singleton.

    This is the primary entry point for most use cases. It handles singleton
    management automatically.

    Args:
        query: The user query to process
        query_type: Type from router (general, reasoning, execution, etc.)
        complexity: Complexity score (0.0 to 1.0)
        context: Optional context dict with conversation_id, history, etc.

    Returns:
        ReasoningResult with selected tools and strategy

    Example:
        >>> from vulcan.reasoning.reasoning_integration import apply_reasoning
        >>> result = apply_reasoning(
        ...     query="What causes climate change?",
        ...     query_type="reasoning",
        ...     complexity=0.7
        ... )
        >>> print(f"Tools: {result.selected_tools}")
    """
    return get_reasoning_integration().apply_reasoning(
        query, query_type, complexity, context
    )


def run_portfolio_reasoning(
    query: str,
    tools: List[str],
    strategy: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run portfolio execution using the global singleton.

    Args:
        query: The user query
        tools: List of tools to use
        strategy: Execution strategy
        constraints: Optional execution constraints

    Returns:
        Portfolio execution result dictionary

    Example:
        >>> result = run_portfolio_reasoning(
        ...     query="Complex problem",
        ...     tools=["symbolic", "causal"],
        ...     strategy="causal_reasoning"
        ... )
    """
    return get_reasoning_integration().run_portfolio(
        query, tools, strategy, constraints
    )


def get_reasoning_statistics() -> Dict[str, Any]:
    """
    Convenience function to get reasoning integration statistics.

    Returns:
        Statistics dictionary with performance metrics

    Example:
        >>> stats = get_reasoning_statistics()
        >>> print(f"Success rate: {stats['success_rate']:.1%}")
    """
    return get_reasoning_integration().get_statistics()


def shutdown_reasoning(timeout: float = 5.0) -> None:
    """
    Shutdown the global reasoning integration.

    After calling this function, the singleton will be cleared and a new
    instance will be created on the next call to get_reasoning_integration().

    Args:
        timeout: Maximum time to wait for shutdown in seconds

    Example:
        >>> shutdown_reasoning(timeout=10.0)
    """
    global _reasoning_integration

    if _reasoning_integration is not None:
        _reasoning_integration.shutdown(timeout=timeout)

        with _integration_lock:
            _reasoning_integration = None

        logger.info(f"{LOG_PREFIX} Global singleton cleared")
