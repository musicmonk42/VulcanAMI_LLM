"""
reasoning_dispatch.py - Main reasoning entry point and routing detection.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def reason(wm, query: str, mode: str = None, **kwargs) -> Dict[str, Any]:
    """
    Main reasoning method with mode support for creative and philosophical reasoning.

    This allows WorldModel to be invoked as a reasoning tool, enabling:
    - Creative composition (poems, stories) with VULCAN-generated structure
    - Philosophical/ethical analysis using multiple frameworks
    - General introspection queries about the AI system

    Args:
        wm: WorldModel instance
        query: The query or problem to reason about
        mode: Reasoning mode - 'philosophical', 'creative', or None for general
        **kwargs: Additional arguments passed to specific reasoning methods

    Returns:
        Dict with 'response', 'confidence', 'reasoning_trace', and 'mode'
    """
    logger.info(f"[WorldModel] reason() called with mode={mode}")

    # Extract mode from query if it's a dict with a 'mode' key
    if isinstance(query, dict):
        mode = query.get('mode', mode)
        actual_query = query.get('query', query.get('text', str(query)))
    else:
        actual_query = str(query)

    # =========================================================================
    # INDUSTRY STANDARD - SINGLE AUTHORITY PATTERN
    # =========================================================================
    # World Model is the "self" and "awareness" of the platform, but it
    # DELEGATES tool selection to ToolSelector (THE authority).
    #
    # OLD APPROACH (competing decision):
    #   World Model -> detect patterns -> directly call engines
    #   This bypassed ToolSelector, creating competing decisions
    #
    # NEW APPROACH (single authority):
    #   World Model -> detect IF reasoning needed -> delegate to UnifiedReasoner
    #   UnifiedReasoner -> asks ToolSelector -> selects tool -> executes
    #
    # World Model's role:
    #   - Orchestrate overall platform awareness
    #   - Handle introspection/meta-reasoning (self-referential queries)
    #   - Provide context about platform state
    #   - Delegate technical reasoning to proper authority (ToolSelector)
    # =========================================================================

    # Check if query requires technical reasoning (not introspection/meta-reasoning)
    if mode is None and should_route_to_reasoning_engine(wm, actual_query):
        logger.info("[WorldModel] Technical reasoning detected, delegating to UnifiedReasoner")
        try:
            return wm._delegate_to_reasoning_system(actual_query, **kwargs)
        except Exception as e:
            logger.warning(f"[WorldModel] Reasoning system delegation failed: {e}, falling back")
            # Continue to mode-based routing on failure

    # =========================================================================
    # WORLD MODEL's CORE RESPONSIBILITY: SELF-AWARENESS & INTROSPECTION
    # =========================================================================
    # These methods handle the platform's sense of "self":
    # - Philosophical reasoning about consciousness, ethics, values
    # - Creative reasoning requiring internal state
    # - General introspection and self-referential queries
    #
    # World Model does NOT delegate these - they ARE the platform's awareness
    # =========================================================================

    # Route to appropriate reasoning method based on mode
    if mode == 'philosophical':
        return wm._philosophical_reasoning(actual_query, **kwargs)
    elif mode == 'creative':
        return wm._creative_reasoning(actual_query, **kwargs)
    else:
        # Default: use introspection for self-referential queries,
        # or return a general analysis
        return wm._general_reasoning(actual_query, **kwargs)


def should_route_to_reasoning_engine(wm, query: str) -> bool:
    """
    Detect queries needing specialized technical reasoning engines.

    This method determines whether a query should be routed to specialized
    reasoning engines (Symbolic, Causal, Analogical, Mathematical) instead
    of the default philosophical reasoning or introspection.

    INDUSTRY STANDARD IMPLEMENTATION:
    - Thread-safe operation (no shared state modification)
    - Comprehensive pattern detection with multiple indicators per domain
    - Defensive programming with input validation
    - Performance optimized with early returns

    Args:
        wm: WorldModel instance
        query: The query string to analyze

    Returns:
        bool: True if query should route to specialized engine, False otherwise
    """
    # Input validation
    if not query or not isinstance(query, str):
        logger.warning("[WorldModel] Invalid query for routing detection")
        return False

    # Security: Validate query length to prevent resource exhaustion
    MAX_QUERY_LENGTH = 10000
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"[WorldModel] Query exceeds max length ({len(query)} > {MAX_QUERY_LENGTH})")
        return False

    query_lower = query.lower()

    # CAUSAL REASONING INDICATORS
    causal_indicators = [
        'confound', 'confounding', 'confounder',
        'causation', 'causality', 'causal effect', 'causal inference',
        'intervention', 'do(', 'do-calculus',
        'pearl', 'structural causal model', 'scm',
        'backdoor', 'frontdoor', 'instrumental variable',
        'counterfactual', 'potential outcome',
        'treatment effect', 'ate', 'cate'
    ]

    causal_count = sum(1 for indicator in causal_indicators if indicator in query_lower)
    if causal_count >= 1:
        logger.info(f"[WorldModel] Detected causal reasoning query (indicators: {causal_count})")
        return True

    # ANALOGICAL REASONING INDICATORS
    analogical_indicators = [
        'structure mapping', 'structural mapping',
        'analogy', 'analogical', 'analogous',
        'domain s', 'source domain',
        'domain t', 'target domain',
        'corresponds to', 'correspondence',
        'mapping between', 'relation mapping',
        'base domain', 'target concept'
    ]

    analogical_count = sum(1 for indicator in analogical_indicators if indicator in query_lower)
    if analogical_count >= 1:
        logger.info(f"[WorldModel] Detected analogical reasoning query (indicators: {analogical_count})")
        return True

    # MATHEMATICAL REASONING INDICATORS
    mathematical_indicators = [
        'compute', 'calculate', 'calculation',
        'sum', 'summation', 'total',
        'induction', 'mathematical induction', 'proof by induction',
        'prove', 'proof', 'theorem',
        'integral', 'derivative', 'differential',
        'equation', 'solve for',
        'optimization', 'minimize', 'maximize',
        'convergence', 'limit'
    ]

    mathematical_count = sum(1 for indicator in mathematical_indicators if indicator in query_lower)
    if mathematical_count >= 1:
        logger.info(f"[WorldModel] Detected mathematical reasoning query (indicators: {mathematical_count})")
        return True

    # SAT/SYMBOLIC LOGIC INDICATORS
    symbolic_indicators = [
        'satisfiable', 'satisfiability',
        'sat', 'unsat',
        # Unicode logical operators
        '\u2192', '\u2227', '\u2228', '\u00ac', '\u2295', '\u2194',
        # Text logical operators (prefixed with 'logical' to avoid false positives)
        'logical implies', 'logical and', 'logical or', 'logical not',
        # Logic domains
        'fol', 'first-order', 'first order logic',
        'predicate logic', 'propositional logic',
        'cnf', 'dnf', 'conjunctive normal form',
        'formula', 'clause',
        'truth table', 'model checking'
    ]

    symbolic_count = sum(1 for indicator in symbolic_indicators if indicator in query_lower)
    if symbolic_count >= 1:
        logger.info(f"[WorldModel] Detected symbolic/SAT reasoning query (indicators: {symbolic_count})")
        return True

    # No specialized reasoning detected
    return False
