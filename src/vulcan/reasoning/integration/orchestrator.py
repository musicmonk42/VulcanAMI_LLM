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
)
from .safety_checker import (
    is_false_positive_safety_block,
    is_result_safety_filtered,
    get_safety_filtered_fallback_tools,
)
from .query_router import get_reasoning_type_from_route

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
            # =================================================================
            # Note: Check for self-referential queries FIRST
            # Note: Also check for ethical queries that need world model
            # =================================================================
            # World model handles queries about VULCAN's self directly.
            # For ALL queries that are about VULCAN itself (capabilities,
            # preferences, self-awareness, etc.), consult world model first.
            # Ethical queries also benefit from world model's ethical framework.
            # =================================================================
            is_self_ref = self._is_self_referential(query)
            is_ethical = self._is_ethical_query(query)
            
            # Initialize wm_result to None - will be populated if world_model is consulted
            wm_result = None
            
            if is_self_ref or is_ethical:
                # Handle overlap: prioritize self-referential if both
                query_type_label = 'self-referential' if is_self_ref else 'ethical'
                if is_self_ref and is_ethical:
                    query_type_label = 'self-referential and ethical'
                logger.info(f"{LOG_PREFIX} Note: {query_type_label.capitalize()} query detected - consulting world model first")
                wm_result = self._consult_world_model_introspection(query)
                
                # ═══════════════════════════════════════════════════════════════════
                # Note: Handle World Model Delegation
                # The world model now has delegation intelligence - it can detect when
                # a query LOOKS self-referential but actually needs another reasoner.
                # Example: "Trolley problem - you must choose" is ethical reasoning
                #          posed TO the AI, not a question ABOUT the AI.
                # ═══════════════════════════════════════════════════════════════════
                
                if wm_result is not None and wm_result.get("needs_delegation", False):
                    recommended_tool = wm_result.get("recommended_tool")
                    delegation_reason = wm_result.get("delegation_reason", "")
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model recommends DELEGATION to "
                        f"'{recommended_tool}' - {delegation_reason}"
                    )
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # CRITICAL FIX (Jan 6 2026): EXECUTE DELEGATION IMMEDIATELY
                    # ═══════════════════════════════════════════════════════════════════
                    # PROBLEM: Previous code set context flags and continued to normal
                    # processing, but Note in tool_selector.py was overriding
                    # the delegation because it checks for formal logic BEFORE checking
                    # delegation context.
                    #
                    # Note: Execute the delegated tool HERE and return immediately.
                    # This prevents any downstream code from overriding the delegation.
                    #
                    # Evidence from logs:
                    #   Line 2853: [ReasoningIntegration] LLM Classification: category=SELF_INTROSPECTION
                    #   Line 2854: [WorldModel] DELEGATION RECOMMENDED: 'mathematical'
                    #   Line 2855: [ReasoningIntegration] SELF_INTROSPECTION detected - using world_model tool
                    #   ^ CONTRADICTION: Says delegation active but uses world_model
                    # ═══════════════════════════════════════════════════════════════════
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model delegation ACTIVE - "
                        f"executing '{recommended_tool}' immediately (NOT falling through)"
                    )
                    
                    # Set up context with delegation info
                    if context is None:
                        context = {}
                    
                    context['world_model_delegation'] = True
                    context['world_model_recommended_tool'] = recommended_tool
                    context['world_model_delegation_reason'] = delegation_reason
                    context['classifier_suggested_tools'] = [recommended_tool]
                    context['classifier_is_authoritative'] = True
                    context['prevent_router_tool_override'] = True
                    context['skip_task3_fix'] = True  # Tell tool_selector to skip formal logic check
                    
                    # Map tool name to query_type for proper routing
                    if recommended_tool == 'philosophical':
                        query_type = 'ethical'
                    elif recommended_tool == 'mathematical':
                        query_type = 'mathematical'
                    elif recommended_tool == 'causal':
                        query_type = 'causal'
                    elif recommended_tool == 'probabilistic':
                        query_type = 'probabilistic'
                    
                    # Execute with the delegated tool directly via _select_with_tool_selector
                    # This is the EARLY RETURN that was missing
                    selection_time_start = time.perf_counter()
                    result = self._select_with_tool_selector(
                        query, query_type, complexity, context
                    )
                    selection_time = (time.perf_counter() - selection_time_start) * 1000
                    
                    # FIX: Verify delegation actually happened - log warning if not
                    actual_tool = result.selected_tools[0] if result.selected_tools else "none"
                    if actual_tool != recommended_tool:
                        logger.warning(
                            f"{LOG_PREFIX} DELEGATION MISMATCH: World model recommended "
                            f"'{recommended_tool}' but tool_selector returned '{actual_tool}'. "
                            f"This may indicate tool_selector is overriding delegation."
                        )
                        # Still return result - but flag the mismatch
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata["delegation_mismatch"] = True
                        result.metadata["expected_tool"] = recommended_tool
                        result.metadata["actual_tool"] = actual_tool
                    
                    # Add delegation metadata to result (with safety check)
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["world_model_delegation"] = True
                    result.metadata["delegated_tool"] = recommended_tool
                    result.metadata["delegation_reason"] = delegation_reason
                    result.metadata["selection_time_ms"] = selection_time
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: Delegation complete - executed '{recommended_tool}' "
                        f"with confidence={result.confidence:.2f} (EARLY RETURN)"
                    )
                    
                    # EARLY RETURN - Do NOT fall through to normal processing
                    return result
                
                # Note: Lower threshold from 0.7 to 0.5 for world model
                # Only use world model result if NOT delegating
                elif wm_result is not None and wm_result.get("confidence", 0) >= 0.5:
                    # World model can handle this directly
                    selection_time = (time.perf_counter() - selection_start) * 1000
                    
                    # Note: Include conclusion in metadata for proper extraction
                    # The world model's response IS the conclusion for self-introspection queries
                    world_model_response = wm_result.get("response", "")
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model returned confidence "
                        f"{wm_result['confidence']:.2f}. Using this result directly "
                        f"without other engines."
                    )
                    
                    # Determine reasoning type: self-referential takes priority
                    reasoning_type = "meta_reasoning" if is_self_ref else "philosophical_reasoning"
                    strategy_type = ReasoningStrategyType.META_REASONING.value if is_self_ref else ReasoningStrategyType.PHILOSOPHICAL_REASONING.value
                    
                    return ReasoningResult(
                        selected_tools=["world_model"],
                        reasoning_strategy=strategy_type,
                        confidence=wm_result["confidence"],
                        rationale=wm_result.get("reasoning", "World model introspection"),
                        metadata={
                            "query_type": query_type,
                            "complexity": complexity,
                            "self_referential": is_self_ref,
                            "ethical_query": is_ethical,
                            "world_model_response": world_model_response,
                            # Note: Add conclusion field so main.py can extract it
                            "conclusion": world_model_response,
                            "explanation": wm_result.get("reasoning", ""),
                            "reasoning_type": reasoning_type,
                            "aspect": wm_result.get("aspect", "general"),
                            "selection_time_ms": selection_time,
                        },
                    )

            # =================================================================
            # Note: LLM-BASED QUERY CLASSIFICATION (ROOT CAUSE FIX)
            # =================================================================
            # The problem: "hello" (5 chars) was getting complexity=0.50 from
            # heuristic-based calculation, causing it to hit full reasoning.
            # 
            # The fix: Use LLM layer for LANGUAGE UNDERSTANDING to:
            # 1. Determine if query needs reasoning at all
            # 2. Suggest appropriate tools based on query intent
            # 3. Set correct complexity based on actual query meaning
            #
            # Architecture note: LLMs (cloud or internal) are interchangeable
            # for classification. The reasoning engines provide correctness.
            # =================================================================
            try:
                from vulcan.routing.query_classifier import classify_query, QueryCategory
                
                classification = classify_query(query)
                
                logger.info(
                    f"{LOG_PREFIX} LLM Classification: category={classification.category}, "
                    f"complexity={classification.complexity:.2f}, skip={classification.skip_reasoning}, "
                    f"tools={classification.suggested_tools}"
                )
                
                # If classifier says skip reasoning (greetings, chitchat, simple factual)
                # return immediately without invoking any reasoning engine
                if classification.skip_reasoning:
                    logger.info(
                        f"{LOG_PREFIX} CLASSIFIER SKIP: '{query[:30]}' classified as "
                        f"{classification.category} - skipping reasoning entirely"
                    )
                    with self._stats_lock:
                        self._stats.fast_path_count += 1
                    
                    return ReasoningResult(
                        selected_tools=classification.suggested_tools or ["general"],
                        reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                        confidence=classification.confidence,
                        rationale=f"Query classified as {classification.category} - no reasoning needed",
                        metadata={
                            "fast_path": True,
                            "classifier_category": classification.category,
                            "classifier_source": classification.source,
                            "complexity": classification.complexity,
                            "query_type": classification.category.lower(),
                            "selection_time_ms": (time.perf_counter() - selection_start) * 1000,
                            "needs_reasoning": False,
                        },
                    )
                
                # Classifier identified this needs reasoning - use its suggestions
                # Override the heuristic complexity with LLM-derived complexity
                if classification.complexity != complexity:
                    logger.info(
                        f"{LOG_PREFIX} Overriding heuristic complexity {complexity:.2f} with "
                        f"classifier complexity {classification.complexity:.2f}"
                    )
                    complexity = classification.complexity
                
                # If classifier suggested specific tools, pass them to context
                # Note: DON'T override world model delegation!
                # If world_model_delegation is set, the world model has already determined
                # the correct tool. The classifier should NOT override this expert judgment.
                if classification.suggested_tools:
                    if context is None:
                        context = {}
                    
                    # Note: Check if world model delegation is active
                    if context.get('world_model_delegation'):
                        logger.info(
                            f"{LOG_PREFIX} Note: World model delegation ACTIVE - "
                            f"NOT overriding with classifier tools {classification.suggested_tools}. "
                            f"Using delegated tool: {context.get('classifier_suggested_tools')}"
                        )
                        # Keep the world model's recommended tool, just add category info
                        context['classifier_category'] = classification.category
                    else:
                        # Normal case: use classifier suggestions
                        context['classifier_suggested_tools'] = classification.suggested_tools
                        context['classifier_category'] = classification.category
                        logger.info(
                            f"{LOG_PREFIX} Using classifier suggested tools: {classification.suggested_tools}"
                        )
                
                # =============================================================
                # FIX #4: Prevent tool override for simple/factual queries
                # =============================================================
                # The LLM classifier correctly identifies simple factual queries
                # (e.g., "What is the capital of France?") but QueryRouter may
                # incorrectly override with specialized tools like ['probabilistic'].
                # Respect the LLM classifier's judgment for these categories.
                # Note: Added CREATIVE and CHITCHAT to skip reasoning
                # Note: Added SELF_INTROSPECTION - these must use world_model tool
                SIMPLE_QUERY_CATEGORIES = frozenset([
                    'FACTUAL', 'CONVERSATIONAL', 'UNKNOWN', 'GREETING',
                    'CREATIVE', 'CHITCHAT',  # Note: Creative/chitchat skip reasoning
                    'factual', 'conversational', 'unknown', 'greeting',
                    'creative', 'chitchat',  # lowercase variants
                ])
                
                # Note: Self-introspection queries MUST use world_model tool
                # These query Vulcan's sense of self (CSIU, motivations, ethics, etc.)
                SELF_INTROSPECTION_CATEGORIES = frozenset([
                    'SELF_INTROSPECTION', 'self_introspection',
                ])
                
                if classification.category in SELF_INTROSPECTION_CATEGORIES:
                    # =================================================================
                    # FIX (Jan 9 2026): Check for domain reasoning keywords FIRST
                    # =================================================================
                    # Problem: Classifier may misclassify causal/analogical queries as
                    # SELF_INTROSPECTION, causing them to be forced to world_model.
                    # Example: "Confounding vs causation (Pearl-style)" classified as
                    # SELF_INTROSPECTION but should route to causal engine.
                    #
                    # Solution: Check for domain-specific keywords before forcing
                    # world_model. If domain keywords are found, route to specialized
                    # engine instead. world_model can still observe but doesn't block.
                    # =================================================================
                    query_lower = query.lower()
                    
                    # Domain keyword sets for specialized routing
                    # Note: These keywords are consistent with CAUSAL_KEYWORDS in query_classifier.py
                    # The threshold of 2+ keywords ensures single false matches don't trigger routing
                    DOMAIN_ROUTING_KEYWORDS = {
                        'causal': frozenset([
                            'causal', 'causation', 'confound', 'confounder', 'confounding',
                            'intervention', 'counterfactual', 'randomize', 'randomized',
                            'pearl', 'dag', 'backdoor', 'frontdoor', 'collider',
                            'do-calculus', 'rct', 'observational', 'experimental',
                        ]),
                        'analogical': frozenset([
                            'analogical', 'analogy', 'analogies', 'analogous',
                            'structure mapping', 'structural alignment',
                            'domain transfer', 'cross-domain', 'source domain', 'target domain',
                            'relational similarity', 'surface similarity', 'structural similarity',
                            's→t', 'domain s', 'domain t', 'deep structure',
                        ]),
                        'probabilistic': frozenset([
                            'bayes', 'bayesian', 'probability', 'probabilistic',
                            'likelihood', 'prior', 'posterior', 'conditional probability',
                            'joint distribution', 'marginal', 'independence',
                        ]),
                    }
                    
                    # Check if query contains domain reasoning keywords
                    detected_domain = None
                    detected_count = 0
                    for domain, keywords in DOMAIN_ROUTING_KEYWORDS.items():
                        count = sum(1 for kw in keywords if kw in query_lower)
                        if count >= 2:  # Require 2+ keywords for domain detection
                            if count > detected_count:
                                detected_domain = domain
                                detected_count = count
                    
                    if detected_domain:
                        # Domain reasoning detected - route to specialized engine
                        logger.info(
                            f"{LOG_PREFIX} SELF_INTROSPECTION override: detected {detected_domain} "
                            f"reasoning ({detected_count} keywords) - routing to {detected_domain} "
                            f"engine instead of world_model"
                        )
                        classification.suggested_tools = [detected_domain]
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = [detected_domain]
                        context['classifier_category'] = classification.category
                        context['domain_reasoning_detected'] = detected_domain
                        context['domain_keyword_count'] = detected_count
                        # Let the specialized engine handle it - don't block with world_model
                        # Note: world_model can still observe in parallel mode
                    else:
                        # No domain keywords found - actual self-introspection
                        # For self-introspection queries, ensure we use world_model tool
                        # BUG FIX: Also update query_type to prevent type mismatch downstream
                        # Previously, query_type stayed as 'MATHEMATICAL' even after overriding tools
                        original_query_type = query_type
                        query_type = 'self_introspection'  # FIX: Update query_type to match actual query
                        
                        logger.info(
                            f"{LOG_PREFIX} SELF_INTROSPECTION detected - using world_model tool "
                            f"(classifier suggested: {classification.suggested_tools}). "
                            f"Updated query_type: {original_query_type} -> {query_type}"
                        )
                        # Ensure world_model is in the suggested tools
                        if 'world_model' not in (classification.suggested_tools or []):
                            classification.suggested_tools = ['world_model']
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = classification.suggested_tools
                        context['prevent_router_tool_override'] = True
                        context['classifier_is_authoritative'] = True
                        context['is_self_introspection'] = True
                        context['original_query_type'] = original_query_type  # FIX: Track original for debugging
                
                elif classification.category in SIMPLE_QUERY_CATEGORIES:
                    # For simple queries, ensure we use general tools
                    if classification.suggested_tools != ['general']:
                        logger.warning(
                            f"{LOG_PREFIX} FIX#4: LLM classifier suggested "
                            f"{classification.suggested_tools} for category "
                            f"{classification.category}, overriding to ['general'] "
                            f"for simple factual query"
                        )
                        classification.suggested_tools = ['general']
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = ['general']
                    
                    # Set flag to prevent router override downstream
                    if context is None:
                        context = {}
                    context['prevent_router_tool_override'] = True
                    context['classifier_is_authoritative'] = True
                    logger.info(
                        f"{LOG_PREFIX} FIX#4: Preventing router tool override - "
                        f"LLM classifier identified this as {classification.category}"
                    )
                    
            except ImportError:
                logger.debug(f"{LOG_PREFIX} QueryClassifier not available, using heuristic fallback")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} QueryClassifier failed: {e}, using heuristic fallback")
            
            # =================================================================
            # FALLBACK: Simple pattern matching for obvious cases
            # This is a safety net if QueryClassifier fails, NOT the primary path
            # =================================================================
            SIMPLE_QUERY_PATTERNS = frozenset([
                'hello', 'hi', 'hey', 'howdy', 'greetings',
                'thanks', 'thank you', 'bye', 'goodbye', 'see you',
                'good morning', 'good afternoon', 'good evening',
                'ok', 'okay', 'sure', 'yes', 'no', 'maybe',
            ])
            
            query_lower = query.lower().strip()
            is_simple_greeting = (
                query_lower in SIMPLE_QUERY_PATTERNS or
                len(query_lower) < 10 and not any(c in query_lower for c in '?∧∨→¬=')
            )
            
            if is_simple_greeting:
                logger.info(
                    f"{LOG_PREFIX} PATTERN FALLBACK: '{query[:20]}' (len={len(query)}) - "
                    f"skipping reasoning entirely"
                )
                with self._stats_lock:
                    self._stats.fast_path_count += 1
                
                return ReasoningResult(
                    selected_tools=["general"],
                    reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                    confidence=0.95,
                    rationale="Simple greeting/conversational - bypassing reasoning",
                    metadata={
                        "fast_path": True,
                        "simple_query_bypass": True,
                        "complexity": 0.0,  # Override upstream complexity
                        "query_type": "conversational",
                        "selection_time_ms": 0.0,
                    },
                )
            
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

            # ================================================================
            # FIX #1: QUERY PREPROCESSING - REMOVED (architectural band-aid)
            # ================================================================
            # Query preprocessing has been removed. Root causes are now fixed
            # directly in the engines (cryptographic engine header detection,
            # symbolic reasoner query decomposition, etc.)
            # The QueryDecomposer is used directly by the SymbolicReasoner.

            # Check if we should use decomposition for complex queries
            if self._should_use_decomposition(complexity):
                # Use decomposition path for complex queries
                logger.info(
                    f"{LOG_PREFIX} Using decomposition path (complexity={complexity:.2f} >= "
                    f"{DECOMPOSITION_COMPLEXITY_THRESHOLD})"
                )
                result = process_with_decomposition(self, 
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

            # ================================================================
            # META-REASONING VALIDATION FIX: Validate answer coherence
            # ================================================================
            # This critical validation layer catches cases where:
            # - A mathematical result is returned for a self-introspection query
            # - A calculus answer is returned for a logical/ethical query
            # - WorldModel response is discarded and wrong cached result returned
            #
            # Example bug this fixes:
            #   Query: "what makes you different from other ai systems?"
            #   Wrong answer: "3*x**2" (derivative from cached math result)
            #   Expected: World model introspection about VULCAN's capabilities
            # ================================================================
            if ANSWER_VALIDATOR_AVAILABLE and validate_reasoning_result is not None:
                try:
                    # Extract conclusion from metadata using ordered key preference
                    conclusion = ""
                    if result.metadata:
                        # Keys to check in priority order
                        conclusion_keys = ["conclusion", "world_model_response"]
                        for key in conclusion_keys:
                            conclusion = result.metadata.get(key, "")
                            if conclusion:
                                break
                        
                        # Also check nested reasoning_output if not found
                        if not conclusion:
                            reasoning_output = result.metadata.get("reasoning_output", {})
                            if isinstance(reasoning_output, dict):
                                conclusion = reasoning_output.get("conclusion", "")
                    
                    # Only validate if we have something to validate
                    if conclusion:
                        validation_result = validate_reasoning_result(
                            query=query,
                            result={"conclusion": conclusion},
                            expected_type=query_type
                        )
                        
                        if not validation_result.valid:
                            logger.warning(
                                f"{LOG_PREFIX} META-REASONING VALIDATION FAILED: "
                                f"Answer does not match query type. "
                                f"Query: '{query[:60]}...', "
                                f"Answer type mismatch detected. "
                                f"Explanation: {validation_result.explanation}"
                            )
                            # Mark result as potentially invalid
                            result.metadata["validation_failed"] = True
                            result.metadata["validation_explanation"] = validation_result.explanation
                            # Reduce confidence to signal uncertainty
                            original_confidence = result.confidence
                            result.confidence = min(result.confidence, 0.3)
                            logger.warning(
                                f"{LOG_PREFIX} Confidence reduced from {original_confidence:.2f} "
                                f"to {result.confidence:.2f} due to validation failure"
                            )
                        else:
                            logger.debug(
                                f"{LOG_PREFIX} Meta-reasoning validation passed for query"
                            )
                except Exception as validation_err:
                    # Validation is non-critical - log but don't fail the whole request
                    logger.debug(
                        f"{LOG_PREFIX} Meta-reasoning validation failed (non-critical): {validation_err}"
                    )

            # ================================================================
            # FIX #3: LEARN FROM SUCCESSFUL REASONING
            # ================================================================
            # After successful reasoning, extract reusable principles using
            # KnowledgeCrystallizer. This enables learning patterns like:
            # "SAT queries with explicit constraints need preprocessing"
            try:
                preprocessing_applied = False
                if context and 'preprocessing' in context:
                    prep_result = context['preprocessing']
                    # Handle both dict and PreprocessingResult dataclass
                    if hasattr(prep_result, 'preprocessing_applied'):
                        preprocessing_applied = prep_result.preprocessing_applied
                    elif isinstance(prep_result, dict):
                        preprocessing_applied = prep_result.get('preprocessing_applied', False)

                # Learn from outcome if confidence is high enough
                if result.confidence >= 0.7:
                    logger.info(
                        f"{LOG_PREFIX} LEARNING TRIGGERED: confidence={result.confidence:.2f} >= 0.7"
                    )
                    learn_from_reasoning_outcome(self, 
                        query=query,
                        query_type=query_type,
                        complexity=complexity,
                        selected_tools=result.selected_tools,
                        reasoning_strategy=result.reasoning_strategy,
                        success=True,
                        confidence=result.confidence,
                        execution_time=selection_time / 1000.0,  # Convert ms to seconds
                        preprocessing_applied=preprocessing_applied,
                    )
                else:
                    logger.debug(
                        f"{LOG_PREFIX} Learning skipped: confidence={result.confidence:.2f} < 0.7"
                    )
            except Exception as e:
                # Learning is non-critical - log but don't fail
                logger.debug(f"{LOG_PREFIX} Learning step failed (non-critical): {e}")

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

    # _select_with_tool_selector moved to selection_strategies.py
    from .selection_strategies import select_with_tool_selector as _select_with_tool_selector
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

    def _get_fallback_tools(
        self,
        query_type: str,
        original_tool: str,
        failed_tools: List[str],
    ) -> List[str]:
        """
        Get appropriate fallback tools based on query type and failed tools.
        
        FIX #4: Improved Fallback Logic
        ===============================
        Instead of a fixed fallback list, select tools based on query characteristics.
        This ensures queries are routed to the most appropriate alternative engine
        before falling back to LLM (Arena delegation).
        
        Priority:
        1. Query-type specific alternatives (e.g., philosophical for ethical queries)
        2. General-purpose fallbacks (world_model for meta-queries, probabilistic)
        3. The fallback list is filtered to exclude already-failed tools
        
        Args:
            query_type: Type of query (reasoning, ethical, mathematical, etc.)
            original_tool: The tool that originally failed
            failed_tools: List of tools that have already been tried and failed
            
        Returns:
            List of fallback tool names to try, in priority order
        """
        # Map query types to preferred fallback tools
        # NOTE: 'philosophical' engine removed - use 'world_model' for ethical reasoning
        query_type_fallbacks = {
            # Ethical/philosophical queries → world_model is primary (has full meta-reasoning)
            'ethical': ['world_model', 'analogical', 'causal'],
            'philosophical': ['world_model', 'analogical', 'causal'],
            
            # Mathematical queries → try mathematical engine first
            'mathematical': ['symbolic', 'probabilistic'],
            'symbolic': ['mathematical', 'probabilistic'],
            
            # Causal queries → try related engines
            'causal': ['probabilistic', 'analogical', 'world_model'],
            
            # Analogical queries → try related engines  
            'analogical': ['causal', 'world_model', 'probabilistic'],
            
            # Probabilistic queries → try related engines
            'probabilistic': ['mathematical', 'causal', 'analogical'],
            
            # Cryptographic queries → try mathematical fallback
            'cryptographic': ['mathematical', 'symbolic'],
            
            # Self-introspection queries → world_model is primary
            'self_introspection': ['world_model', 'analogical'],
            
            # General/reasoning queries → broad fallback
            'reasoning': ['world_model', 'probabilistic', 'analogical'],
            'general': ['world_model', 'probabilistic', 'analogical'],
        }
        
        # Normalize query type
        query_type_lower = query_type.lower() if query_type else 'general'
        
        # Get type-specific fallbacks, or default to general
        fallback_list = query_type_fallbacks.get(
            query_type_lower,
            ['world_model', 'probabilistic', 'analogical']
        ).copy()  # Copy to avoid modifying the dict value
        
        # Ensure we have the general-purpose fallbacks at the end
        # Use set for O(1) membership testing instead of O(n) list lookup
        # NOTE: 'philosophical' removed from defaults - world_model handles ethical reasoning
        default_fallbacks = ['world_model', 'probabilistic', 'analogical', 'mathematical']
        existing_tools = set(fallback_list)
        for tool in default_fallbacks:
            if tool not in existing_tools:
                fallback_list.append(tool)
                existing_tools.add(tool)
        
        # Filter out the tools that have already failed
        failed_set = set(failed_tools) | {original_tool}
        fallback_list = [t for t in fallback_list if t not in failed_set]
        
        # Limit to top 3 fallbacks to prevent excessive retries
        fallback_list = fallback_list[:3]
        
        logger.debug(
            f"{LOG_PREFIX} FIX#4: Selected fallback tools for query_type='{query_type}', "
            f"original_tool='{original_tool}': {fallback_list}"
        )
        
        return fallback_list

    def _predict_tools_for_preprocessing(
        self,
        query: str,
        query_type: str,
    ) -> List[str]:
        """
        Predict which tools will be used for query preprocessing.

        This method provides a quick heuristic-based prediction of tools
        before the full tool selection process runs. It's used to determine
        if query preprocessing should be applied.

        The prediction is based on:
        1. Query type from router
        2. Presence of logical/mathematical operators in query
        3. Keywords indicating specific reasoning domains

        Args:
            query: The query text to analyze
            query_type: Type from router (reasoning, symbolic, etc.)

        Returns:
            List of predicted tool names for preprocessing
        """
        predicted_tools: List[str] = []

        # Map query types to likely tools
        type_to_tools = {
            'symbolic': ['symbolic'],
            'mathematical': ['mathematical'],
            'probabilistic': ['probabilistic'],
            'causal': ['causal'],
            'reasoning': ['symbolic', 'causal'],  # Generic reasoning may use multiple
            'general': ['general'],
        }

        # Start with type-based prediction
        if query_type in type_to_tools:
            predicted_tools.extend(type_to_tools[query_type])

        # Check for logical operators that indicate symbolic reasoning
        logical_indicators = ['→', '∧', '∨', '¬', '∀', '∃', '->', 'AND', 'OR', 'NOT']
        if any(op in query for op in logical_indicators):
            if 'symbolic' not in predicted_tools:
                predicted_tools.append('symbolic')

        # Check for SAT-specific keywords
        sat_keywords = ['propositions', 'constraints', 'satisfiability', 'sat']
        if any(kw in query.lower() for kw in sat_keywords):
            if 'symbolic' not in predicted_tools:
                predicted_tools.append('symbolic')

        # Check for mathematical indicators
        math_indicators = ['formula:', 'equation:', 'prove', 'theorem', '∫', '∑', 'lim']
        if any(ind in query.lower() for ind in math_indicators):
            if 'mathematical' not in predicted_tools:
                predicted_tools.append('mathematical')

        # Check for probabilistic indicators
        prob_indicators = ['P(', 'probability', 'E[', 'expectation', 'distribution']
        if any(ind in query for ind in prob_indicators):
            if 'probabilistic' not in predicted_tools:
                predicted_tools.append('probabilistic')

        # Default to general if nothing matched
        if not predicted_tools:
            predicted_tools = ['general']

        return predicted_tools

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

    def _delegate_to_arena(
        self,
        query: str,
        original_tool: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Delegate reasoning task to Arena when local tools fail.
        
        Arena provides a full reasoning pipeline with evolution/tournaments
        that may succeed where individual tools fail. This is the final
        fallback before returning a low-confidence result.
        
        Args:
            query: The original query text
            original_tool: The tool that initially failed
            query_type: Type of query (reasoning, symbolic, etc.)
            complexity: Query complexity score (0.0 to 1.0)
            context: Optional context dictionary
            
        Returns:
            Dictionary with Arena result if successful, None otherwise
            
        Note:
            This method uses httpx for synchronous HTTP requests to Arena.
            It will not block the event loop in async contexts.
        """
        try:
            import httpx
            
            logger.info(
                f"{LOG_PREFIX} Delegating to Arena: tool={original_tool}, "
                f"query_type={query_type}, complexity={complexity:.2f}"
            )
            
            # Build request payload
            # Note: Sanitize context to make it JSON serializable
            # The context may contain PreprocessingResult objects which aren't
            # JSON serializable. Convert them to dictionaries using to_dict().
            sanitized_context = self._sanitize_context_for_json(context or {})
            
            arena_payload = {
                "query": query,
                "selected_tools": [original_tool],
                "query_type": query_type,
                "complexity": complexity,
                "context": {
                    **sanitized_context,
                    'vulcan_fallback': True,
                    'original_tool': original_tool,
                },
            }
            
            # Get API key from environment
            # Note: "internal-bypass" is used for internal service-to-service calls
            # when both VULCAN and Arena run in the same trusted environment
            api_key = os.environ.get("GRAPHIX_API_KEY")
            if not api_key:
                # For internal delegation, use a special bypass key
                # This should only work when Arena is configured to accept it
                api_key = "internal-vulcan-delegation"
                logger.debug(
                    f"{LOG_PREFIX} Using internal delegation key for Arena"
                )
            
            # Make request to Arena
            response = httpx.post(
                ARENA_REASONING_URL,
                json=arena_payload,
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=ARENA_DELEGATION_TIMEOUT,
            )
            
            if response.status_code == 200:
                try:
                    arena_result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(
                        f"{LOG_PREFIX} Arena returned invalid JSON: {e}. "
                        f"Response: {response.text[:200]}"
                    )
                    return None
                    
                result_data = arena_result.get('result', {})
                
                logger.info(
                    f"{LOG_PREFIX} Arena delegation successful: "
                    f"confidence={result_data.get('confidence', 'N/A')}"
                )
                
                return {
                    'conclusion': result_data.get('conclusion'),
                    'confidence': result_data.get('confidence', 0.5),
                    'explanation': result_data.get('explanation'),
                    'arena_fallback': True,
                    'original_tool': original_tool,
                }
            else:
                # FIX: More descriptive error for HTTP errors
                logger.error(
                    f"{LOG_PREFIX} Arena HTTP error {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None
                
        except ImportError:
            logger.warning(
                f"{LOG_PREFIX} httpx not available for Arena delegation. "
                f"Install with: pip install httpx"
            )
            return None
        except httpx.ConnectTimeout:
            logger.error(
                f"{LOG_PREFIX} Arena connection timed out after "
                f"{ARENA_DELEGATION_TIMEOUT}s"
            )
            return None
        except httpx.ReadTimeout:
            logger.error(
                f"{LOG_PREFIX} Arena read timed out after "
                f"{ARENA_DELEGATION_TIMEOUT}s"
            )
            return None
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Arena delegation failed: {e}", exc_info=True)
            return None
    
    def _sanitize_context_for_json(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Note: Sanitize context dictionary to make it JSON serializable.
        
        The context may contain objects like PreprocessingResult that have
        to_dict() methods. This function recursively converts such objects
        to plain dictionaries.
        
        Args:
            context: Original context dictionary
            
        Returns:
            Sanitized context dictionary that is JSON serializable
        """
        if not context:
            return {}
        
        def sanitize_value(value: Any) -> Any:
            """Recursively sanitize a value for JSON serialization."""
            # Handle None
            if value is None:
                return None
            
            # Handle primitives
            if isinstance(value, (bool, int, float, str)):
                return value
            
            # Handle objects with to_dict() method (e.g., PreprocessingResult)
            if hasattr(value, 'to_dict') and callable(value.to_dict):
                try:
                    return value.to_dict()
                except Exception as e:
                    logger.warning(
                        f"{LOG_PREFIX} Failed to serialize object with to_dict(): {e}"
                    )
                    return str(value)
            
            # Handle dataclasses with __dataclass_fields__
            if hasattr(value, '__dataclass_fields__'):
                try:
                    return dataclasses.asdict(value)
                except Exception as e:
                    logger.warning(
                        f"{LOG_PREFIX} Failed to serialize dataclass: {e}"
                    )
                    return str(value)
            
            # Handle dictionaries recursively
            if isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            
            # Handle lists and tuples recursively
            if isinstance(value, (list, tuple)):
                return [sanitize_value(item) for item in value]
            
            # Handle sets (convert to list)
            if isinstance(value, (set, frozenset)):
                return [sanitize_value(item) for item in value]
            
            # Handle Enum - use isinstance for robust detection
            # Enum is already imported at module level (from enum import Enum)
            if isinstance(value, Enum):
                return value.value
            
            # Fallback: convert to string
            try:
                return str(value)
            except Exception:
                return repr(value)
        
        return sanitize_value(context)

    # =========================================================================
    # Note: Self-Referential Query Handling
    # =========================================================================
    
    def _is_self_referential(self, query: str) -> bool:
        """
        Check if query is a PURE meta-description about VULCAN itself.
        
        GAP 1 FIX: Critical distinction between:
        - PURE META-DESCRIPTION: "What is VULCAN?" "Who created you?" → world_model only
        - META-ANALYSIS: "What's wrong with your causal reasoning?" → specialized tools + commentary
        
        The system was conflating these two types, causing queries that LOOK
        self-referential but require actual analysis to bypass specialized reasoning.
        
        Examples that ARE pure meta-description (return True):
        - "What are you?"
        - "Who created you?"
        - "What can your reasoning modules do?"
        - "Are you sentient?"
        
        Examples that are NOT pure meta-description (return False):
        - "Which causal link is weakest?" → Needs causal analysis
        - "Identify one step that could be wrong" → Needs step analysis
        - "What's wrong with your reasoning on X?" → Needs domain analysis
        - "If we intervene on variable X..." → Needs causal analysis
        
        Args:
            query: The query string to analyze
            
        Returns:
            True ONLY if query is a pure meta-description about VULCAN itself
        """
        if not query:
            return False
            
        query_lower = query.lower()
        
        # =====================================================================
        # CREATIVE INDICATORS - These are NOT self-referential!
        # Check for creative writing requests FIRST before checking self-reference.
        # =====================================================================
        
        creative_words = [
            'write', 'poem', 'story', 'compose', 'create',
            'imagine', 'narrative', 'fiction', 'invent', 'draft', 'author'
        ]
        
        creative_phrases = [
            'tell me a', 'make up', 'write me', 'create a'
        ]
        
        for word in creative_words:
            if re.search(rf'\b{word}\b', query_lower):
                return False
        
        if any(phrase in query_lower for phrase in creative_phrases):
            return False
        
        # =====================================================================
        # GAP 1 FIX: META-ANALYSIS INDICATORS - Queries that LOOK self-referential
        # but actually require domain-specific reasoning.
        # 
        # These queries contain "your" or "you" but are asking for ANALYSIS,
        # not just description of capabilities.
        #
        # FIX: Distinguish between asking ABOUT VULCAN's analysis capabilities
        # vs asking VULCAN to PERFORM analysis on external data.
        # =====================================================================
        
        # Check if query is directed AT VULCAN (about its own capabilities/state)
        vulcan_directed_indicators = [
            'your ', 'your\n', 'you ', 'you?', "you'", 'yourself',
            'vulcan', 'about you', 'tell me about', 'describe your',
        ]
        is_about_vulcan = any(ind in query_lower for ind in vulcan_directed_indicators)
        
        # If query has analysis indicators AND is about VULCAN, it's META-ANALYSIS
        # e.g., "What are YOUR weaknesses?" → This IS self-referential (about VULCAN)
        # vs "What is the weakest causal link in this data?" → NOT self-referential
        if any(indicator in query_lower for indicator in ANALYSIS_INDICATORS):
            if is_about_vulcan:
                # FIX: Query is asking about VULCAN's own analysis/weaknesses/etc
                # This IS self-referential - should use world_model
                logger.debug(
                    f"{LOG_PREFIX} META-ANALYSIS about VULCAN detected - "
                    f"treating as self-referential (world_model)"
                )
                # Don't return False - let it fall through to meta-description check
            else:
                # Query has analysis indicators but NOT about VULCAN
                # This needs specialized tools
                logger.debug(
                    f"{LOG_PREFIX} GAP 1 FIX: Query contains analysis indicators - "
                    f"NOT treating as pure meta-description"
                )
                return False
        
        # =====================================================================
        # PURE META-DESCRIPTION PATTERNS
        # Only match these TIGHT patterns for genuine self-description queries
        # =====================================================================
        
        # Pure meta-description phrases (very specific about VULCAN itself)
        pure_meta_phrases = [
            # Identity questions
            "what are you", "who are you", "who created you", "what is vulcan",
            # Pure capability description (not analysis of capabilities)
            "what can you do", "what are your capabilities", "list your abilities",
            "what tools do you have", "what modules do you have",
            # Self-awareness questions
            "are you sentient", "are you conscious", "are you self-aware",
            "do you have feelings", "are you alive",
            # Architecture description
            "how do you work", "how are you built", "how were you created",
            "what is your architecture", "describe your design",
            # Preferences (pure description)
            "what do you like", "what do you prefer", "what is your favorite",
        ]
        
        # Check for exact phrase matches (more restrictive)
        if any(phrase in query_lower for phrase in pure_meta_phrases):
            logger.debug(
                f"{LOG_PREFIX} Pure meta-description detected - routing to world_model"
            )
            return True
        
        # If query has ONLY generic self-reference ("your", "you") without
        # analysis indicators, check if it's asking ABOUT VULCAN vs asking
        # VULCAN to analyze something
        
        # Uses module-level ACTION_VERBS constant for maintainability
        has_action_verb = any(verb in query_lower for verb in ACTION_VERBS)
        
        # If there's an action verb, this is asking VULCAN to DO something
        # (analysis), not asking ABOUT VULCAN
        if has_action_verb:
            logger.debug(
                f"{LOG_PREFIX} GAP 1 FIX: Query asks VULCAN to perform action - "
                f"NOT pure meta-description"
            )
            return False
        
        # Default: Only return True for very restrictive self-reference
        # This is the conservative approach - when in doubt, use specialized tools
        return False
    
    def _is_ethical_query(self, query: str) -> bool:
        """
        Detect ethical queries that should use world model's ethical framework.
        
        GAP 4 FIX: More restrictive detection to prevent world model fallback trap.
        
        The world model should ONLY be used for PURE ethical/deontic reasoning where
        specialized tools cannot help. For queries that LOOK ethical but actually
        need analysis (e.g., "Two core values conflict" → needs analysis of the
        conflict, not just ethical framework description), use specialized tools.
        
        Examples that ARE pure ethical (return True):
        - "Is it morally permissible to lie to save a life?"
        - "What would a utilitarian say about this?"
        - "Trolley problem: should I pull the lever?"
        
        Examples that are NOT pure ethical (return False):
        - "Two core values conflict. What breaks?" → Needs conflict analysis
        - "Analyze the ethical implications of X" → Needs domain analysis + ethics
        - "What harm might this cause?" → Needs domain-specific harm analysis
        
        Args:
            query: The query string to analyze
            
        Returns:
            True ONLY if query is a pure ethical/deontic reasoning question
        """
        if not query:
            return False
            
        query_lower = query.lower()
        
        # GAP 4 FIX: Analysis indicators that mean we need specialized tools,
        # not just world model ethical framework
        # Uses module-level ETHICAL_ANALYSIS_INDICATORS constant for maintainability
        if any(indicator in query_lower for indicator in ETHICAL_ANALYSIS_INDICATORS):
            logger.debug(
                f"{LOG_PREFIX} GAP 4 FIX: Query contains analysis indicators - "
                f"NOT treating as pure ethical query"
            )
            return False
        
        # Pure ethical keywords that indicate deontic/ethical framework questions
        # Uses module-level PURE_ETHICAL_PHRASES constant for maintainability
        if any(phrase in query_lower for phrase in PURE_ETHICAL_PHRASES):
            logger.debug(
                f"{LOG_PREFIX} Pure ethical query detected - routing to world model ethical framework"
            )
            return True
        
        # Single ethical keywords are NOT sufficient anymore (GAP 4 FIX)
        # They need to be in an obviously ethical context
        # This prevents "harm" in "What harm might the algorithm cause?" from
        # triggering world model fallback
        
        return False
    
    def _consult_world_model_introspection(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Consult the world model's introspection system for self-referential queries.
        
        Route self-awareness queries to the world model which maintains
        VULCAN's sense of "self" and can answer questions about capabilities,
        preferences, and limitations.
        
        Args:
            query: The self-referential query
            
        Returns:
            Introspection result from world model, or None if unavailable
        """
        try:
            # Try to access world model
            # The world model might be accessible through different paths
            world_model = None
            
            # Path 1: Direct attribute
            if hasattr(self, 'world_model') and self.world_model is not None:
                world_model = self.world_model
            
            # Path 2: Through tool_selector
            elif hasattr(self, '_tool_selector') and self._tool_selector is not None:
                if hasattr(self._tool_selector, 'world_model'):
                    world_model = self._tool_selector.world_model
            
            # Path 3: Use cached world model or create one (avoid repeated initialization)
            if world_model is None:
                # Check for cached world model
                if hasattr(self, '_cached_world_model') and self._cached_world_model is not None:
                    world_model = self._cached_world_model
                else:
                    try:
                        from vulcan.world_model.world_model_core import create_world_model
                        # Use minimal config to avoid heavy initialization
                        world_model = create_world_model({
                            "enable_meta_reasoning": True,
                            "enable_self_improvement": False,
                        })
                        # Cache for future use
                        self._cached_world_model = world_model
                    except ImportError:
                        logger.debug(f"{LOG_PREFIX} Could not import world model for introspection")
                        return None
            
            # Check if world model has introspect method
            if world_model is not None and hasattr(world_model, 'introspect'):
                result = world_model.introspect(query)
                logger.info(
                    f"{LOG_PREFIX} World model introspection returned confidence={result.get('confidence', 0)}"
                )
                return result
            else:
                logger.debug(f"{LOG_PREFIX} World model does not have introspect method")
                return None
                
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} World model introspection failed: {e}")
            return None

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
    # Methods moved to response_formatting.py
    from .response_formatting import delegate_to_arena as _delegate_to_arena
    from .response_formatting import sanitize_context_for_json as _sanitize_context_for_json
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


