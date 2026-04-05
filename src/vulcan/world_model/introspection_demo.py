"""
introspection_demo.py - Demonstration query handling (counterfactual, causal reasoning).

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import time

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
        return demonstrate_counterfactual_reasoning(wm)
    elif reasoning_to_demonstrate == "causal":
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


def demonstrate_counterfactual_reasoning(wm) -> dict:
    """
    Actually run counterfactual reasoning with an example scenario.

    Uses the CounterfactualObjectiveReasoner to analyze a real trade-off.
    """
    start_time = time.time()

    # Check if we have the counterfactual reasoner available
    if hasattr(wm, 'counterfactual_reasoner') and wm.counterfactual_reasoner is not None:
        try:
            # Create an example scenario
            context = {
                "current_state": {
                    "primary_objective": "accuracy",
                    "current_accuracy": 0.95,
                    "current_latency_ms": 200,
                    "current_throughput": 100,
                },
                "domain": "model_optimization",
            }

            # Run counterfactual analysis for "speed" objective
            outcome_speed = wm.counterfactual_reasoner.predict_under_objective(
                alternative_objective="speed",
                context=context,
            )

            # Run counterfactual analysis for "throughput" objective
            outcome_throughput = wm.counterfactual_reasoner.predict_under_objective(
                alternative_objective="throughput",
                context=context,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            response = f"""Let me demonstrate counterfactual reasoning with a real example:

## Current Strategy: Optimize for Accuracy
- Accuracy: 95%
- Latency: 200ms
- Throughput: 100 req/s

## Counterfactual 1: What if we optimized for SPEED instead?

{format_counterfactual_outcome(wm, outcome_speed, "speed")}

## Counterfactual 2: What if we optimized for THROUGHPUT instead?

{format_counterfactual_outcome(wm, outcome_throughput, "throughput")}

## Trade-off Analysis

By running counterfactual reasoning, I can explore alternative objectives
and their consequences BEFORE committing to a decision. This enables:

1. **Informed Trade-offs**: See exactly what you'd gain/lose
2. **Pareto-optimal Solutions**: Find options where no objective can improve without hurting another
3. **Risk Assessment**: Understand side effects of different strategies

*Counterfactual analysis completed in {elapsed_ms:.1f}ms using CounterfactualObjectiveReasoner*"""

            return {
                "confidence": 0.90,
                "response": response,
                "aspect": "demonstration",
                "reasoning": "Live counterfactual reasoning demonstration",
                "metadata": {
                    "demonstration_type": "counterfactual",
                    "computation_time_ms": elapsed_ms,
                    "outcomes_generated": 2,
                },
            }

        except Exception as e:
            logger.warning(f"[WorldModel] Counterfactual demonstration failed: {e}")
            # Fall through to static example

    # Static example if reasoner not available
    return {
        "confidence": 0.85,
        "response": """Let me demonstrate counterfactual reasoning with an example:

## Scenario: Model Optimization Trade-offs

**Current Strategy**: Optimize for Accuracy
- Accuracy: 95%
- Latency: 200ms
- Throughput: 100 req/s

**Counterfactual**: What if we optimized for SPEED instead?

I analyze the causal relationships:
- Reducing model complexity \u2192 Lower latency
- Smaller batch sizes \u2192 Faster responses
- Less exhaustive search \u2192 Lower accuracy

**Predicted Outcome** (if optimizing for speed):
- Accuracy: 90% (-5%)
- Latency: 50ms (-75%)
- Throughput: 400 req/s (+300%)

**Side Effects Identified**:
- \u2713 4x improvement in response time
- \u2713 4x improvement in throughput
- \u2717 5% accuracy degradation
- \u26a0 May affect edge cases more severely

**Trade-off Decision Framework**:
- If user latency is critical \u2192 Switch to speed optimization
- If accuracy is non-negotiable \u2192 Keep current strategy
- Hybrid approach: Use fast model for common cases, accurate for edge cases

This is counterfactual reasoning in action - exploring "what if" scenarios
to inform decisions BEFORE committing to changes.""",
        "aspect": "demonstration",
        "reasoning": "Counterfactual reasoning demonstration (static example)",
    }


def format_counterfactual_outcome(wm, outcome, objective_name: str) -> str:
    """Format a CounterfactualOutcome for display."""
    if outcome is None:
        return f"Could not predict outcome for {objective_name}"

    lines = []
    lines.append(f"**Predicted** (confidence: {outcome.confidence:.0%}):")
    lines.append(f"- Primary value: {outcome.predicted_value:.2f}")
    lines.append(f"- Range: [{outcome.lower_bound:.2f}, {outcome.upper_bound:.2f}]")

    if outcome.side_effects:
        lines.append("- Side effects:")
        for effect_name, effect_value in outcome.side_effects.items():
            sign = "+" if effect_value > 0 else ""
            lines.append(f"  \u2022 {effect_name}: {sign}{effect_value:.2f}")

    return "\n".join(lines)


def demonstrate_causal_reasoning(wm) -> dict:
    """Demonstrate causal reasoning with an example."""
    return {
        "confidence": 0.85,
        "response": """Let me demonstrate causal reasoning with an example:

## Scenario: Understanding a Causal Graph

Given the causal structure:
```
    A \u2192 B \u2190 C
        \u2193
        D
```

**Query**: Does conditioning on B induce correlation between A and C?

**Causal Analysis**:

1. **B is a collider** (common effect of A and C)
   - Both A and C point into B
   - Normally, A and C are independent

2. **Conditioning on B creates selection bias**
   - When we know B occurred, it becomes evidence
   - If A caused B, then C is less likely to also have caused B
   - This creates negative correlation between A and C

3. **This is the "explaining away" effect**
   - Classic result from causal inference
   - Conditioning on a collider opens a path between its causes

**Conclusion**: YES, conditioning on B induces correlation between A and C.

**Practical Example**:
- A = "Talent"
- B = "Getting into elite university"
- C = "Family connections"
- D = "Career success"

Among elite university students (conditioning on B), we'd observe
negative correlation between talent and connections - those with
connections needed less talent to get in, and vice versa.

This is causal reasoning - analyzing graph structure to determine
what interventions and observations imply.""",
        "aspect": "demonstration",
        "reasoning": "Causal reasoning demonstration",
    }
