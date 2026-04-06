"""
introspection_demo_causal.py - Causal and counterfactual demonstration details.

Extracted from introspection_demo.py to reduce module size.
Contains the actual demonstration implementations for counterfactual
and causal reasoning examples.

Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import time

logger = logging.getLogger(__name__)


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
