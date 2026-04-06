"""
introspection_domain.py - Domain awareness and general introspection.

Extracted from world_model_core.py to reduce class size.
Comparison, future speculation, and preference responses have been
further extracted to introspection_responses.py.
Functions take `wm` (WorldModel instance) as first parameter.
"""


def explain_domain_awareness(wm, domain: str, query: str) -> str:
    """
    Explain awareness of specific reasoning domains.

    Critical: World model should be aware of ALL domains, even technical ones
    like mathematics and logic. It's the integrating "self" of the system.
    """
    domain_explanations = {
        'mathematical': """
I am aware of mathematical reasoning as one of my core capabilities.
My mathematical engine can:
- Extract and parse expressions (including Unicode: \u2211, \u222b, \u221a)
- Perform symbolic computation
- Verify proofs by induction
- Solve equations and optimize functions

This awareness allows me to know when to route queries to mathematical
reasoning vs other modes.
""",
        'logical': """
I am aware of logical reasoning through my symbolic engine.
Capabilities include:
- SAT solving (satisfiability checking)
- Formal proof construction
- Consistency verification
- Logical inference

I maintain awareness of logical structure across other domains
(e.g., recognizing logical implications in causal reasoning).
""",
        'probabilistic': """
I am aware of probabilistic reasoning as a fundamental mode of thought.
My probabilistic engine handles:
- Bayesian inference
- Conditional probability calculations
- Uncertainty quantification
- Prior/posterior updates

This awareness extends to recognizing uncertainty in my own outputs.
""",
        'causal': """
I am aware of causal reasoning through Pearl-style inference.
This includes:
- Distinguishing correlation from causation
- Modeling interventions
- Identifying confounders
- Constructing causal graphs

Causal awareness is central to my world model's predictions.
""",
        'ethical': """
I am aware of ethical reasoning as multi-framework analysis.
My philosophical engine considers:
- Deontological constraints (Kant)
- Consequentialist calculations (utilitarianism)
- Virtue ethics
- Care ethics
- Rights-based frameworks

This awareness allows me to recognize moral dilemmas and reason
through them systematically.
""",
    }

    explanation = domain_explanations.get(domain, f"I am aware of {domain} as a reasoning domain.")

    query_preview = query[:50] + "..." if len(query) > 50 else query
    return f"{explanation}\n\nYour query about '{query_preview}' engages this awareness directly."


def general_introspection(wm, query: str) -> str:
    """
    Handle general introspective queries.

    FIX (Jan 7 2026): Instead of returning a generic template, try to
    extract the key concept from the query and provide a relevant response.
    """
    from .introspection_self import explain_boundaries
    from .introspection_meta import (
        assess_own_confidence,
        identify_own_assumptions,
        suggest_self_improvements,
        analyze_own_biases,
    )

    query_lower = query.lower()
    query_preview = query[:100] + "..." if len(query) > 100 else query

    # Check for hidden patterns not caught earlier
    if any(kw in query_lower for kw in ['limitation', 'limit', 'cannot', "can't", "unable"]):
        return explain_boundaries(wm)

    if any(kw in query_lower for kw in ['confident', 'certainty', 'sure', 'accurate']):
        return assess_own_confidence(wm, query)

    if any(kw in query_lower for kw in ['assumption', 'assume', 'presume', 'presuppose']):
        return identify_own_assumptions(wm, query)

    if any(kw in query_lower for kw in ['improve', 'better', 'redesign', 'change', 'upgrade']):
        return suggest_self_improvements(wm, query)

    if any(kw in query_lower for kw in ['bias', 'biased', 'prejudice', 'unfair']):
        return analyze_own_biases(wm, query)

    # Extract key concepts from query for a more relevant response
    key_concepts = []
    concept_keywords = {
        'reasoning': ['reason', 'think', 'logic', 'deduce'],
        'capabilities': ['can', 'able', 'capable', 'do'],
        'identity': ['who', 'what', 'are you', 'identity'],
        'purpose': ['why', 'purpose', 'goal', 'objective'],
        'knowledge': ['know', 'learn', 'understand', 'information'],
    }

    for concept, keywords in concept_keywords.items():
        if any(kw in query_lower for kw in keywords):
            key_concepts.append(concept)

    if key_concepts:
        concept_str = ', '.join(key_concepts)
        return f"""
Your question touches on: **{concept_str}**

Query: "{query_preview}"

Let me address this specifically:

**About my {key_concepts[0] if key_concepts else 'nature'}:**

I am VULCAN, an integrated reasoning system composed of multiple specialized
engines (symbolic, probabilistic, mathematical, causal, philosophical)
coordinated by this world model.

{'My reasoning capabilities include formal logic, mathematical computation, causal inference, and ethical analysis.' if 'reasoning' in key_concepts else ''}
{'I can process queries across multiple domains and maintain awareness of my own processes.' if 'capabilities' in key_concepts else ''}
{'I am an AI system designed for sophisticated reasoning tasks, with self-reflective capabilities.' if 'identity' in key_concepts else ''}
{'My purpose is to provide accurate, reasoned responses while maintaining epistemic humility.' if 'purpose' in key_concepts else ''}
{'I maintain knowledge through my training and can reason about new information within context.' if 'knowledge' in key_concepts else ''}

What specific aspect would you like me to elaborate on?
"""

    # True fallback - couldn't classify at all
    return f"""
Your question: "{query_preview}"

I recognize this as an introspective query, but I'm not certain which aspect
of my self-model you're interested in.

I can discuss:
\u2022 **Limitations** - What I cannot do
\u2022 **Confidence** - How certain I am in my responses
\u2022 **Assumptions** - What I'm taking for granted
\u2022 **Improvements** - How I could be better designed
\u2022 **Biases** - Potential systematic errors in my reasoning
\u2022 **Capabilities** - What I can actually do
\u2022 **Architecture** - How my reasoning engines work together

Could you rephrase your question to focus on one of these aspects?
Or ask a more specific question about my nature or functioning.
"""



# Backward-compatible re-exports from introspection_responses.py
from .introspection_responses import (  # noqa: F401, E402
    generate_comparison_response,
    generate_future_speculation_response,
    generate_preference_response,
)
