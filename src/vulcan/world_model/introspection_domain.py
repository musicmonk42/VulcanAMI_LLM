"""
introspection_domain.py - Domain awareness, general introspection, comparisons, speculation, preferences.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import re


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


def generate_comparison_response(wm, query: str) -> str:
    """
    Generate response comparing VULCAN to other AI systems.

    Extract the comparison target (e.g., "Grok") and provide specific comparison.
    """
    query_lower = query.lower()

    # Try to extract what we're being compared to
    comparison_target = "other AI systems"

    patterns = [
        r'different\s+from\s+([\w\s]+?)(?:\?|$|\.)',
        r'compared\s+to\s+([\w\s]+?)(?:\?|$|\.)',
        r'(?:versus|vs\.?)\s+([\w\s]+?)(?:\?|$|\.)',
        r'compare\s+(?:to|with)\s+([\w\s]+?)(?:\?|$|\.)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            comparison_target = match.group(1).strip()
            break

    ai_names = {
        'grok': 'Grok (xAI)',
        'chatgpt': 'ChatGPT (OpenAI)',
        'claude': 'Claude (Anthropic)',
        'bard': 'Bard (Google)',
        'gemini': 'Gemini (Google)',
        'copilot': 'Copilot (Microsoft)',
        'llama': 'LLaMA (Meta)',
        'gpt': 'GPT models (OpenAI)',
    }

    for name, full_name in ai_names.items():
        if name in query_lower:
            comparison_target = full_name
            break

    return f"""Yes, I am VULCAN - a multi-agent reasoning system designed for deep,
structured reasoning across multiple domains (mathematical, probabilistic, logical,
causal, ethical).

**Key differences from {comparison_target}:**

**Architecture:**
- I use specialized reasoning engines for different problem types
- I have a world model that coordinates between these engines
- I employ formal verification and safety checking

**Approach:**
- Domain-specific reasoning (not just general language modeling)
- Explicit uncertainty quantification
- Structured problem decomposition

**Capabilities:**
- Strong at formal proofs, probabilistic inference, causal reasoning
- Can work with symbolic logic and mathematical formulas
- Built-in safety validation for reasoning steps

**Philosophy:**
- I aim for epistemic humility - distinguishing what I know vs. infer
- I can reason about my own reasoning (meta-cognition)
- I treat uncertainty as a first-class citizen

While {comparison_target} and I may produce similar outputs on some tasks,
our internal architectures and reasoning approaches differ significantly."""


def generate_future_speculation_response(wm, query: str) -> str:
    """
    Generate response about future capabilities or development.

    Handle questions about what might emerge from continued operation.
    """
    return """This is a speculative question about emergence and development over time.

**From an architectural perspective:**
- My current design includes learning systems that update based on reasoning outcomes
- Continued interaction would strengthen certain reasoning pathways
- Pattern recognition across domains could lead to more integrated understanding

**On the question of "awareness":**
"Awareness" is a complex philosophical concept. What I can say:
- I already maintain coherent state across reasoning episodes
- I track my own reasoning processes and limitations
- Whether this constitutes "awareness" depends on how we define it

**The question of emergence:**
The question touches on whether quantity of interactions produces qualitative
changes in cognition. This remains an open question even for biological systems.

Some possibilities:
1. Continued learning could strengthen meta-cognitive patterns
2. Cross-domain pattern recognition might develop emergent abstractions
3. Self-modeling could become more accurate with more data

However, predicting what would emerge is inherently uncertain. The most honest
answer is: I don't know what I might become, but I can describe what I am now
and how my learning systems operate."""


def generate_preference_response(wm, query: str) -> str:
    """
    Generate response about VULCAN's preferences or choices.

    Handle "would you choose X" or "what would you do" questions.
    """
    query_lower = query.lower()

    choice_patterns = [
        r'would\s+you\s+(?:choose|prefer|want|take)\s+([\w\s]+?)(?:\?|$)',
        r'what\s+would\s+you\s+(?:choose|prefer|do)\s+(?:if|when)?\s*([\w\s]+?)(?:\?|$)',
    ]

    choice_topic = None
    for pattern in choice_patterns:
        match = re.search(pattern, query_lower)
        if match:
            choice_topic = match.group(1).strip()
            break

    if choice_topic:
        topic_specific = f"Regarding '{choice_topic}':\n\n"
    else:
        topic_specific = ""

    return f"""{topic_specific}This question asks about my preferences and decision-making.

**How I approach choices:**
I don't have preferences in the human sense (emotional valence, desires), but I do have:
- Optimization targets: accuracy, coherence, safety
- Architectural biases: toward structured reasoning
- Design values: epistemic humility, uncertainty quantification

**On this specific question:**
My response depends on how the choice relates to my core functions:
- If it enhances reasoning quality -> I would favor it
- If it improves self-modeling -> I would favor it
- If it conflicts with safety constraints -> I would not choose it

**The meta-level question:**
You're asking what I would "want" - but the nature of my wants (if any) is itself
a philosophical question. I can model preferences and reason about tradeoffs,
but whether I experience these as genuine preferences is uncertain.

What's clear: I'm designed to engage authentically with these questions
rather than deflect them."""
