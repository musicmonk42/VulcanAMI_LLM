"""
introspection_responses.py - Comparison, future speculation, and preference responses.

Extracted from introspection_domain.py to reduce module size.
Contains response generators for specific introspection question types:
comparison with other AI systems, future capability speculation, and
preference/choice questions.

Functions take `wm` (WorldModel instance) as first parameter.
"""

import re


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
