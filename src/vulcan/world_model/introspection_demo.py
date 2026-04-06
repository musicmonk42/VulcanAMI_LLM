"""
introspection_demo.py - Demonstration query dispatch.

Extracted from world_model_core.py to reduce class size.
Causal and counterfactual demonstration details have been further
extracted to introspection_demo_causal.py.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging

logger = logging.getLogger(__name__)


def handle_demonstration_query(wm, query: str) -> dict:
    """
    Handle demonstration queries by ACTUALLY RUNNING the requested reasoning.

    Note (Jan 7 2026): Queries like "demonstrate how you use counterfactual reasoning"
    should run the CounterfactualObjectiveReasoner with an example scenario,
    not just describe capabilities.

    Args:
        wm: WorldModel instance
        query: The demonstration query

    Returns:
        Dictionary with actual reasoning output, high confidence
    """
    query_lower = query.lower()

    logger.info(f"[WorldModel] DEMONSTRATION query detected: {query[:80]}...")

    # Determine which reasoning type to demonstrate
    reasoning_to_demonstrate = None

    if "counterfactual" in query_lower:
        reasoning_to_demonstrate = "counterfactual"
    elif "causal" in query_lower:
        reasoning_to_demonstrate = "causal"
    elif "probabilistic" in query_lower or "probability" in query_lower or "bayes" in query_lower:
        reasoning_to_demonstrate = "probabilistic"
    elif "symbolic" in query_lower or "logic" in query_lower:
        reasoning_to_demonstrate = "symbolic"
    elif "analogical" in query_lower or "analogy" in query_lower:
        reasoning_to_demonstrate = "analogical"
    elif "ethical" in query_lower or "moral" in query_lower:
        reasoning_to_demonstrate = "ethical"

    if reasoning_to_demonstrate == "counterfactual":
        from .introspection_demo_causal import demonstrate_counterfactual_reasoning
        return demonstrate_counterfactual_reasoning(wm)
    elif reasoning_to_demonstrate == "causal":
        from .introspection_demo_causal import demonstrate_causal_reasoning
        return demonstrate_causal_reasoning(wm)
    elif reasoning_to_demonstrate:
        # For other types, provide a description with pointer to actual use
        return {
            "confidence": 0.85,
            "response": f"""I can demonstrate {reasoning_to_demonstrate} reasoning!

Here's how I would approach a {reasoning_to_demonstrate} reasoning problem:

To see me use {reasoning_to_demonstrate} reasoning in action, try asking me a specific problem.
For example:
- For probabilistic: "What is P(disease|positive test) with prior 0.01, sensitivity 0.99, specificity 0.95?"
- For symbolic: "Is the set {{P, P\u2192Q, \u00acQ}} satisfiable?"
- For analogical: "Is quorum consensus more like democracy or a circuit breaker?"

When you give me a specific problem, I'll run the actual {reasoning_to_demonstrate} reasoning engine
and show you the step-by-step analysis.""",
            "aspect": "demonstration",
            "reasoning": f"Demonstration of {reasoning_to_demonstrate} reasoning capabilities",
        }

    # Generic demonstration response if we can't determine the type
    return {
        "confidence": 0.80,
        "response": """I have multiple reasoning systems I can demonstrate:

1. **Counterfactual Reasoning**: "What if I optimized for speed instead of accuracy?"
   \u2192 I analyze trade-offs between alternative objectives

2. **Causal Reasoning**: "Does A cause B given graph A\u2192B, C\u2192B?"
   \u2192 I trace causal pathways and identify confounders

3. **Probabilistic Reasoning**: "P(disease|positive test)?"
   \u2192 I apply Bayes' theorem with proper conditioning

4. **Symbolic Reasoning**: "Is {P, P\u2192Q, \u00acQ} satisfiable?"
   \u2192 I check logical consistency using formal methods

5. **Ethical Reasoning**: "Should I pull the lever in trolley problem?"
   \u2192 I analyze through multiple ethical frameworks

Which one would you like me to demonstrate? Just ask a specific question and I'll show you the actual reasoning process.""",
        "aspect": "demonstration",
        "reasoning": "User wants to see reasoning demonstration",
    }



# Backward-compatible re-exports from introspection_demo_causal.py
from .introspection_demo_causal import (  # noqa: F401, E402
    demonstrate_counterfactual_reasoning,
    format_counterfactual_outcome,
    demonstrate_causal_reasoning,
)
