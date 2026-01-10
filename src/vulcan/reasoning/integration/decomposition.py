"""
Decomposition processing for reasoning integration.

Handles hierarchical problem decomposition using ProblemDecomposer.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .types import ReasoningResult, LOG_PREFIX

logger = logging.getLogger(__name__)


def process_with_decomposition(
    orchestrator,
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
        orchestrator: ReasoningIntegration instance
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

        problem_graph = orchestrator._query_bridge.convert_to_problem_graph(
            query=query,
            query_analysis=query_analysis,
            tool_selection=None,  # Will be determined per subproblem
        )

        if problem_graph is None:
            logger.warning(
                f"{LOG_PREFIX} Query bridge returned None, falling back to direct selection"
            )
            from .selection_strategies import select_with_tool_selector
            return select_with_tool_selector(orchestrator, query, query_type, complexity, context)

        # Step 2: Decompose the problem
        decomposition_plan = orchestrator._problem_decomposer.decompose_novel_problem(problem_graph)

        if decomposition_plan is None or len(decomposition_plan.steps) == 0:
            logger.warning(
                f"{LOG_PREFIX} Decomposition returned empty plan, falling back to direct selection"
            )
            from .selection_strategies import select_with_tool_selector
            return select_with_tool_selector(orchestrator, query, query_type, complexity, context)

        logger.info(
            f"{LOG_PREFIX} Decomposed into {len(decomposition_plan.steps)} steps, "
            f"confidence={decomposition_plan.confidence:.2f}"
        )

        # Step 3: Select tools ONCE based on ORIGINAL query
        logger.info(
            f"{LOG_PREFIX} Selecting tools based on original query "
            f"(length={len(query)} chars)"
        )
        
        from .selection_strategies import select_with_tool_selector
        primary_result = select_with_tool_selector(
            orchestrator,
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
                "tool_selector_available": orchestrator._tool_selector is not None,
                "problem_decomposer_available": orchestrator._problem_decomposer is not None,
            },
        )

    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} Decomposition processing failed: {e}, "
            f"falling back to direct selection",
            exc_info=True
        )
        # Graceful degradation: fall back to direct tool selection
        from .selection_strategies import select_with_tool_selector
        return select_with_tool_selector(orchestrator, query, query_type, complexity, context)
