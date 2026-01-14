"""
Selection strategies for reasoning integration.

Provides tool selection and strategy determination logic for the
ReasoningIntegration orchestrator.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    ReasoningResult,
    ReasoningStrategyType,
    RoutingDecision,
    IntegrationStatistics,
    DECOMPOSITION_COMPLEXITY_THRESHOLD,
    LOG_PREFIX,
    MAX_FALLBACK_ATTEMPTS,
    CAUSAL_REASONING_THRESHOLD,
    PROBABILISTIC_REASONING_THRESHOLD,
    QUERY_TYPE_STRATEGY_MAP,
)

# Import SelectionRequest for ToolSelector calls
try:
    from vulcan.reasoning.selection.tool_selector import SelectionRequest, SelectionMode
    SELECTION_REQUEST_AVAILABLE = True
except ImportError:
    SELECTION_REQUEST_AVAILABLE = False
    SelectionRequest = None
    SelectionMode = None

logger = logging.getLogger(__name__)


class SelectionStrategies:
    """Selection strategies for reasoning integration."""
    
    def __init__(self, integration):
        """Initialize with reference to parent integration."""
        self._integration = integration
    
    @staticmethod
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

            # Step 3: Select tools ONCE based on ORIGINAL query
            # Note: Previously, step descriptions (~28 chars like "Step 1: Parse constraints")
            # were passed to ToolSelector instead of the original query (e.g., 507 chars).
            # This caused semantic matching to fail because it was matching against
            # short step descriptions instead of the actual user query.
            # 
            # The fix: Select tools once based on the original query, then apply those
            # tools to each decomposed step.
            logger.info(
                f"{LOG_PREFIX} Selecting tools based on original query "
                f"(length={len(query)} chars)"
            )
            
            primary_result = self._select_with_tool_selector(
                query=query,  # Use ORIGINAL query, not step descriptions
                query_type=query_type,
                complexity=complexity,
                context=context,
            )
            
            # The tools selected for the original query apply to all steps
            all_tools: set = set(primary_result.selected_tools)
            step_results: List[Dict[str, Any]] = []

            # Record step metadata (without re-running tool selection per step)
            for step in decomposition_plan.steps:
                # Extract step description for metadata only
                if hasattr(step, 'description'):
                    step_description = step.description
                elif hasattr(step, 'to_dict'):
                    step_dict = step.to_dict()
                    step_description = step_dict.get('description', str(step))
                else:
                    step_description = str(step)

                # Extract step complexity for metadata
                if hasattr(step, 'estimated_complexity'):
                    step_complexity = step.estimated_complexity
                elif hasattr(step, 'complexity'):
                    step_complexity = step.complexity
                else:
                    step_complexity = complexity * 0.5  # Default to half of parent

                # Ensure step_complexity is within bounds
                step_complexity = max(0.1, min(1.0, step_complexity))

                # Record step metadata - tools are inherited from primary selection
                step_results.append({
                    'step_id': getattr(step, 'step_id', f'step_{len(step_results)}'),
                    'description': step_description[:100],  # Truncate for metadata
                    'tools': primary_result.selected_tools,  # Inherited from primary
                    'strategy': primary_result.reasoning_strategy,
                    'confidence': primary_result.confidence,
                    'step_complexity': step_complexity,
                })

            # Step 4: Determine overall strategy based on decomposition
            if decomposition_plan.strategy:
                strategy_name = getattr(decomposition_plan.strategy, 'name', 'hierarchical')
            else:
                strategy_name = 'hierarchical_decomposition'

            # Calculate overall confidence
            # Use the primary tool selection confidence combined with decomposition confidence
            num_steps = len(step_results)
            overall_confidence = (decomposition_plan.confidence * 0.4) + (primary_result.confidence * 0.6)

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

    @staticmethod
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

    @staticmethod
    def _determine_strategy_from_query(
        self,
        query_type: str,
        complexity: float
    ) -> str:
        """
        Determine reasoning strategy based on query characteristics.

        This method delegates to the standalone determine_strategy_from_query function
        to avoid code duplication.

        Args:
            query_type: Type of query (reasoning, perception, planning, etc.)
            complexity: Query complexity (0.0 to 1.0)

        Returns:
            Strategy name string
        """
        return determine_strategy_from_query(query_type, complexity)


def select_with_tool_selector(
    orchestrator: Any,
    query: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Select tools using the ToolSelector component.

    This function contains the IMPLEMENTATION for tool selection. It is called by:
    1. orchestrator._select_with_tool_selector() as a thin wrapper (Adapter Pattern)
    2. decomposition.py as a fallback when decomposition fails
    
    IMPORTANT: This function must NOT delegate back to orchestrator._select_with_tool_selector
    to avoid infinite recursion. The orchestrator method delegates TO this function,
    not the other way around.

    Args:
        orchestrator: ReasoningIntegration instance with initialized components
        query: The user query text to process
        query_type: Type of query (reasoning, execution, etc.)
        complexity: Query complexity score (0.0 to 1.0)
        context: Optional context dictionary

    Returns:
        ReasoningResult with selected tools, strategy, and metadata

    Note:
        Falls back to default strategy if ToolSelector is unavailable.
    """
    # Use ToolSelector directly if available
    if orchestrator._tool_selector is not None:
        try:
            # Get selection mode based on complexity
            from vulcan.reasoning.selection.tool_selector import SelectionMode
            
            if complexity > 0.7:
                mode = SelectionMode.ACCURATE
            elif complexity < 0.4:
                mode = SelectionMode.FAST
            else:
                mode = SelectionMode.BALANCED
            
            # Build constraints from context
            constraints = {}
            if context:
                constraints = context.get('constraints', {})
            
            # Call tool selector - FIXED: Use correct method name select_and_execute
            # Build SelectionRequest
            if not SELECTION_REQUEST_AVAILABLE:
                logger.error("SelectionRequest not available, cannot call select_and_execute")
                # Fall back to direct tool selection
                tools = get_fallback_tools(query_type, 'general', [])
                return ReasoningResult(
                    selected_tools=tools if tools else ['general'],
                    reasoning_strategy='direct',
                    confidence=0.4,
                    rationale="SelectionRequest unavailable",
                    metadata={"query_type": query_type, "complexity": complexity}
                )
                
            selection_request = SelectionRequest(
                problem=query,
                mode=mode,
                constraints=constraints,
                context={'query_type': query_type},
            )
            selection = orchestrator._tool_selector.select_and_execute(selection_request)
            
            # Convert SelectionResult to ReasoningResult
            # FIX: SelectionResult has 'selected_tool' (singular), not 'selected_tools' (plural)
            # FIX: Extract actual answer from execution_result instead of ignoring it
            # FIX: Use strategy_used instead of strategy
            selected_tool = getattr(selection, 'selected_tool', 'general')
            tools = [selected_tool] if selected_tool else ['general']
            strategy = getattr(selection, 'strategy_used', 'direct')
            # Handle ExecutionStrategy enum robustly - use str() for any object
            if hasattr(strategy, 'value'):
                strategy = strategy.value
            elif hasattr(strategy, 'name'):
                strategy = strategy.name
            else:
                strategy = str(strategy)
            confidence = getattr(selection, 'confidence', 0.5)
            
            # FIX: Extract the actual reasoning result from execution_result
            # This is the KEY FIX - execution_result contains the actual answer!
            execution_result = getattr(selection, 'execution_result', None)
            conclusion = None
            explanation = None
            reasoning_type = None
            
            if execution_result is not None:
                # Extract conclusion from the execution result
                if isinstance(execution_result, dict):
                    # Use explicit None checks to allow empty strings as valid values
                    conclusion = execution_result.get('result')
                    if conclusion is None:
                        conclusion = execution_result.get('conclusion')
                    explanation = execution_result.get('explanation')
                    if explanation is None:
                        explanation = execution_result.get('rationale')
                    reasoning_type = execution_result.get('reasoning_type')
                    if reasoning_type is None:
                        reasoning_type = execution_result.get('tool')
                elif hasattr(execution_result, 'conclusion'):
                    conclusion = getattr(execution_result, 'conclusion', None)
                    explanation = getattr(execution_result, 'explanation', '')
                    reasoning_type = getattr(execution_result, 'reasoning_type', None)
                else:
                    # execution_result is the conclusion itself
                    # Validate it's a reasonable type for a conclusion
                    if isinstance(execution_result, (str, int, float, bool, list)):
                        conclusion = execution_result
                    elif hasattr(execution_result, '__str__'):
                        # Convert to string if it has str representation
                        conclusion = str(execution_result)
                    else:
                        # Unknown type - store as-is but log warning
                        logger.warning(f"{LOG_PREFIX} Unexpected execution_result type: {type(execution_result)}")
                        conclusion = execution_result
            
            # Build rationale from explanation or provide meaningful fallback
            if explanation:
                rationale = explanation
            elif conclusion:
                rationale = f"Reasoning completed with {selected_tool} engine"
            else:
                rationale = f"Tool selection completed - {selected_tool} selected"
            
            # Include metadata from selection result
            metadata = {
                "query_type": query_type,
                "complexity": complexity,
                "tool_selector_used": True,
                "selected_tool": selected_tool,
                "execution_time_ms": getattr(selection, 'execution_time_ms', 0.0),
            }
            
            # Merge any metadata from selection
            if hasattr(selection, 'metadata') and isinstance(selection.metadata, dict):
                metadata.update(selection.metadata)
            
            # FIX: Include conclusion and reasoning_type in metadata for downstream extraction
            # Note: ReasoningResult doesn't have conclusion/reasoning_type fields,
            # so we store them in metadata where they can be extracted by unified_chat.py
            if conclusion is not None:
                metadata['conclusion'] = conclusion
                metadata['has_execution_result'] = True
            if reasoning_type is not None:
                metadata['reasoning_type'] = reasoning_type
            if explanation:
                metadata['explanation'] = explanation
            
            return ReasoningResult(
                selected_tools=tools,
                reasoning_strategy=str(strategy),
                confidence=confidence,
                rationale=rationale,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ToolSelector failed: {e}, using fallback")
    
    # Ultimate fallback: determine strategy from query characteristics
    strategy = determine_strategy_from_query(query_type, complexity)
    tools = get_fallback_tools(query_type, 'general', [])
    
    return ReasoningResult(
        selected_tools=tools if tools else ['general'],
        reasoning_strategy=strategy,
        confidence=0.4,
        rationale="Fallback selection - ToolSelector unavailable",
        metadata={
            "query_type": query_type,
            "complexity": complexity,
            "fallback": True,
        }
    )


def determine_strategy_from_query(query_type: str, complexity: float) -> str:
    """
    Determine reasoning strategy based on query characteristics.

    This function implements the fallback strategy selection logic when
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


def get_fallback_tools(
    query_type: str,
    original_tool: str,
    failed_tools: List[str],
) -> List[str]:
    """
    Get appropriate fallback tools based on query type and failed tools.
    
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
    query_type_fallbacks: Dict[str, List[str]] = {
        # Ethical/philosophical queries → world_model is primary
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
        f"{LOG_PREFIX} Selected fallback tools for query_type='{query_type}', "
        f"original_tool='{original_tool}': {fallback_list}"
    )
    
    return fallback_list
