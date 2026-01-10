"""
Learning from reasoning outcomes.

Handles learning from successful reasoning outcomes using SemanticBridge
and KnowledgeCrystallizer.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List

from .types import LOG_PREFIX

logger = logging.getLogger(__name__)


def learn_from_outcome(
    orchestrator,
    query: str,
    query_analysis: Dict[str, Any],
    selected_tools: List[str],
    success: bool,
    execution_time: float,
) -> None:
    """
    Learn from reasoning outcome using SemanticBridge.
    
    After successful query execution, this method creates a pattern
    from the outcome and adds it to the SemanticBridge for future
    cross-domain transfer.
    
    Args:
        orchestrator: ReasoningIntegration instance
        query: Original query string
        query_analysis: Query analysis results
        selected_tools: Tools that were used
        success: Whether execution succeeded
        execution_time: Total execution time in seconds
        
    Example:
        >>> learn_from_outcome(
        ...     orchestrator,
        ...     query="What causes X?",
        ...     query_analysis={'type': 'reasoning', 'complexity': 0.6},
        ...     selected_tools=['causal'],
        ...     success=True,
        ...     execution_time=1.5
        ... )
    """
    # Only learn from successful outcomes
    if not success:
        return
    
    # Ensure components are initialized
    orchestrator._init_components()
    
    if orchestrator._semantic_bridge is None or orchestrator._domain_bridge is None:
        return
    
    try:
        # Get domain information
        domains = orchestrator._domain_bridge.get_domains_for_tools(selected_tools)
        primary_domain = orchestrator._domain_bridge.identify_primary_domain(
            selected_tools,
            query_analysis.get('type', 'general'),
        )
        
        # Create pattern outcome for learning
        from vulcan.semantic_bridge import PatternOutcome
        
        # Use deterministic SHA-256 hash for pattern ID (hash() is not deterministic across runs)
        pattern_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        
        outcome = PatternOutcome(
            pattern_id=f"query_{pattern_hash}",
            success=success,
            domain=primary_domain,
            execution_time=execution_time,
            tools=selected_tools,
            complexity=query_analysis.get('complexity', 0.5),
        )
        
        # Create pattern from query characteristics
        pattern = {
            'query_type': query_analysis.get('type', 'general'),
            'complexity': query_analysis.get('complexity', 0.0),
            'tools': selected_tools,
            'domains': list(domains),
        }
        
        # Learn concept from pattern
        concept = orchestrator._semantic_bridge.learn_concept_from_pattern(
            pattern=pattern,
            outcomes=[outcome],
        )
        
        if concept:
            logger.debug(
                f"{LOG_PREFIX} Learned concept in domain {primary_domain}"
            )
            
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Failed to learn from outcome: {e}")


def learn_from_reasoning_outcome(
    orchestrator,
    query: str,
    query_type: str,
    complexity: float,
    selected_tools: List[str],
    reasoning_strategy: str,
    success: bool,
    confidence: float,
    execution_time: float,
    preprocessing_applied: bool = False,
) -> None:
    """
    Learn from successful reasoning outcomes using KnowledgeCrystallizer.

    This method is called after successful reasoning to extract reusable
    principles that can improve future query processing. It integrates
    with the KnowledgeCrystallizer to store patterns like:
    - "SAT queries with propositions + constraints need preprocessing"
    - "High-complexity ethical queries need philosophical reasoning"
    - "Mathematical proofs require step-by-step validation"

    Args:
        orchestrator: ReasoningIntegration instance
        query: Original query text
        query_type: Type of query (symbolic, reasoning, etc.)
        complexity: Query complexity score (0.0 to 1.0)
        selected_tools: Tools that were used for this query
        reasoning_strategy: Strategy that was applied
        success: Whether reasoning succeeded
        confidence: Confidence in the result (0.0 to 1.0)
        execution_time: Time taken in seconds
        preprocessing_applied: Whether query preprocessing was needed

    Note:
        This method is designed to be non-blocking and non-critical.
        Failures are logged but do not affect the main reasoning pipeline.
    """
    # Only learn from successful outcomes with sufficient confidence
    if not success or confidence < 0.7:
        logger.debug(
            f"{LOG_PREFIX} Skipping crystallizer learning: "
            f"success={success}, confidence={confidence:.2f}"
        )
        return

    try:
        # Try to import KnowledgeCrystallizer
        from vulcan.knowledge_crystallizer import (
            KnowledgeCrystallizer,
            ExecutionTrace,
            KNOWLEDGE_CRYSTALLIZER_AVAILABLE,
        )

        if not KNOWLEDGE_CRYSTALLIZER_AVAILABLE or KnowledgeCrystallizer is None:
            logger.debug(f"{LOG_PREFIX} KnowledgeCrystallizer not available")
            return

        # Create execution trace for crystallization
        trace_id = hashlib.sha256(
            f"{query}:{time.time()}".encode()
        ).hexdigest()[:12]

        trace = ExecutionTrace(
            trace_id=trace_id,
            actions=[
                {
                    'type': 'tool_selection',
                    'tools': selected_tools,
                    'strategy': reasoning_strategy,
                },
                {
                    'type': 'preprocessing',
                    'applied': preprocessing_applied,
                },
            ],
            outcomes={
                'success': success,
                'confidence': confidence,
                'execution_time': execution_time,
            },
            context={
                'query_type': query_type,
                'complexity': complexity,
                'query_length': len(query),
            },
            success=success,
            domain=query_type,
            metadata={
                'preprocessing_required': preprocessing_applied,
                'tools_used': selected_tools,
                'strategy': reasoning_strategy,
            },
        )

        # Get or create crystallizer instance (lazy initialization)
        if not hasattr(orchestrator, '_knowledge_crystallizer') or orchestrator._knowledge_crystallizer is None:
            orchestrator._knowledge_crystallizer = KnowledgeCrystallizer()
            logger.info(f"{LOG_PREFIX} KnowledgeCrystallizer initialized for learning")

        # Crystallize knowledge from the trace
        # Use incremental mode for single-trace learning
        from vulcan.knowledge_crystallizer import CrystallizationMode

        crystallization_result = orchestrator._knowledge_crystallizer.crystallize(
            traces=[trace],
            mode=CrystallizationMode.INCREMENTAL,
        )

        if crystallization_result and crystallization_result.principles:
            logger.info(
                f"{LOG_PREFIX} Extracted {len(crystallization_result.principles)} "
                f"principles from successful reasoning"
            )
        else:
            logger.debug(f"{LOG_PREFIX} No new principles extracted from trace")

    except ImportError:
        logger.debug(f"{LOG_PREFIX} KnowledgeCrystallizer module not available")
    except Exception as e:
        # Log but don't fail - learning is non-critical
        logger.debug(f"{LOG_PREFIX} Crystallizer learning failed: {e}")
