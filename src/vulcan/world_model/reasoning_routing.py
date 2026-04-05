"""
reasoning_routing.py - Routing queries to reasoning engines and delegation.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def delegate_to_reasoning_system(wm, query: str, **kwargs) -> Dict[str, Any]:
    """
    Delegate technical reasoning to UnifiedReasoner (which uses ToolSelector).

    INDUSTRY STANDARD - DELEGATION PATTERN:
    World Model is the "self" and "awareness" of the platform. When technical
    reasoning is needed (causal, symbolic, mathematical), World Model DELEGATES
    to the reasoning system rather than making tool selection decisions itself.

    ARCHITECTURAL HIERARCHY:
    1. World Model: Orchestrator, self-awareness, decides IF reasoning needed
    2. UnifiedReasoner: Accepts query, delegates to ToolSelector
    3. ToolSelector: THE AUTHORITY for which tool to use
    4. Reasoning Engine: Executes the selected tool

    Args:
        wm: WorldModel instance
        query: The query requiring technical reasoning
        **kwargs: Additional arguments for reasoning system

    Returns:
        Dict[str, Any]: Standardized reasoning result

    Raises:
        ImportError: If UnifiedReasoner is not available
        Exception: If reasoning execution fails
    """
    try:
        # Import UnifiedReasoner (lazy import for performance)
        from ..reasoning.unified import UnifiedReasoner
        from ..reasoning.singletons import get_unified_reasoner

        # Get or create unified reasoner instance
        reasoner = get_unified_reasoner()
        if reasoner is None:
            logger.info("[WorldModel] Creating new UnifiedReasoner instance")
            reasoner = UnifiedReasoner()

        # Prepare reasoning request
        # UnifiedReasoner will call ToolSelector to determine which engine to use
        reasoning_request = {
            'query': query,
            'context': {
                'from_world_model': True,  # Indicate request comes from platform's "self"
                'world_model_state': wm.get_system_status(),  # Provide self-awareness context
                **kwargs
            }
        }

        logger.info(
            f"[WorldModel] Delegating to UnifiedReasoner, "
            f"ToolSelector will determine appropriate engine"
        )

        # Call unified reasoner - it will use ToolSelector for tool selection
        result = reasoner.reason(reasoning_request)

        # Normalize result to World Model's standard format
        return wm._normalize_reasoning_result(result)

    except ImportError as e:
        logger.error(f"[WorldModel] Failed to import UnifiedReasoner: {e}")
        logger.error("[WorldModel] Cannot delegate - falling back to direct engine routing")
        # Fallback to old method if UnifiedReasoner unavailable
        return wm._route_to_appropriate_engine(query, **kwargs)
    except Exception as e:
        logger.error(f"[WorldModel] Reasoning delegation failed: {e}")
        raise


def route_to_appropriate_engine(wm, query: str, **kwargs) -> Dict[str, Any]:
    """
    [LEGACY/FALLBACK] Route query to specialized reasoning engine.

    DEPRECATED: This method bypasses ToolSelector and should NOT be used
    in normal operation. It exists ONLY as a fallback when UnifiedReasoner
    is unavailable.

    Args:
        wm: WorldModel instance
        query: The query to process
        **kwargs: Additional arguments for reasoning engines

    Returns:
        Dict[str, Any]: Standard WorldModel reasoning result
    """
    logger.warning(
        "[WorldModel] Using legacy _route_to_appropriate_engine() - "
        "this bypasses ToolSelector. Use _delegate_to_reasoning_system() instead."
    )
    query_lower = query.lower()
    engine_used = None
    result = None

    try:
        # =================================================================
        # CAUSAL REASONING ENGINE
        # =================================================================
        if any(indicator in query_lower for indicator in [
            'confound', 'causation', 'causal effect', 'intervention', 'do(',
            'pearl', 'backdoor', 'counterfactual'
        ]):
            logger.info("[WorldModel] Routing to CausalReasoner")
            engine_used = 'causal'

            try:
                from vulcan.reasoning.causal_reasoning import CausalReasoner
                reasoner = CausalReasoner()
                result = reasoner.analyze(query)
                logger.info("[WorldModel] CausalReasoner completed successfully")
            except ImportError as e:
                logger.error(f"[WorldModel] Failed to import CausalReasoner: {e}")
                raise
            except Exception as e:
                logger.error(f"[WorldModel] CausalReasoner execution failed: {e}")
                raise

        # =================================================================
        # ANALOGICAL REASONING ENGINE
        # =================================================================
        elif any(indicator in query_lower for indicator in [
            'structure mapping', 'analogy', 'domain s', 'domain t',
            'corresponds to', 'mapping between'
        ]):
            logger.info("[WorldModel] Routing to AnalogicalReasoner")
            engine_used = 'analogical'

            try:
                from vulcan.reasoning.analogical_reasoning import AnalogicalReasoner
                reasoner = AnalogicalReasoner()
                result = reasoner.reason(query)
                logger.info("[WorldModel] AnalogicalReasoner completed successfully")
            except ImportError as e:
                logger.error(f"[WorldModel] Failed to import AnalogicalReasoner: {e}")
                raise
            except Exception as e:
                logger.error(f"[WorldModel] AnalogicalReasoner execution failed: {e}")
                raise

        # =================================================================
        # MATHEMATICAL REASONING ENGINE
        # =================================================================
        elif any(indicator in query_lower for indicator in [
            'compute', 'calculate', 'sum', 'induction', 'prove',
            'integral', 'derivative', 'equation', 'optimization'
        ]):
            logger.info("[WorldModel] Routing to MathematicalVerificationEngine")
            engine_used = 'mathematical'

            try:
                from vulcan.reasoning.mathematical_verification import MathematicalVerificationEngine
                reasoner = MathematicalVerificationEngine()
                result = reasoner.verify(query)
                logger.info("[WorldModel] MathematicalVerificationEngine completed successfully")
            except ImportError as e:
                logger.error(f"[WorldModel] Failed to import MathematicalVerificationEngine: {e}")
                raise
            except Exception as e:
                logger.error(f"[WorldModel] MathematicalVerificationEngine execution failed: {e}")
                raise

        # =================================================================
        # SYMBOLIC/SAT REASONING ENGINE
        # =================================================================
        elif any(indicator in query_lower for indicator in [
            'satisfiable', 'sat', '\u2192', '\u2227', '\u2228', '\u00ac',
            'fol', 'first-order', 'predicate logic', 'propositional logic'
        ]):
            logger.info("[WorldModel] Routing to SymbolicReasoner")
            engine_used = 'symbolic'

            try:
                from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
                reasoner = SymbolicReasoner()
                result = reasoner.query(query, timeout=kwargs.get('timeout', 10))
                logger.info("[WorldModel] SymbolicReasoner completed successfully")
            except ImportError as e:
                logger.error(f"[WorldModel] Failed to import SymbolicReasoner: {e}")
                raise
            except Exception as e:
                logger.error(f"[WorldModel] SymbolicReasoner execution failed: {e}")
                raise

        else:
            # Should not reach here if _should_route_to_reasoning_engine is correct
            logger.warning("[WorldModel] No matching engine found despite routing detection")
            raise ValueError("No appropriate reasoning engine found for query")

        # =================================================================
        # RESULT NORMALIZATION
        # =================================================================
        return wm._normalize_engine_result(result, engine_used, query)

    except Exception as e:
        logger.error(f"[WorldModel] Engine routing failed: {e}")
        # Re-raise for upstream handling (will fall back to _general_reasoning)
        raise
