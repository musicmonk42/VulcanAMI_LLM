"""
introspection_handlers.py - Handler routing and aspect-specific dispatch for introspection.

Extracted from introspection_core.py to reduce module size.
Contains the pattern-matching logic that routes introspection queries
to the appropriate handler functions.

Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging

logger = logging.getLogger(__name__)


def route_introspection_query(wm, query: str, query_lower: str):
    """
    Route an introspection query to the appropriate handler based on pattern matching.

    Checks query patterns in priority order and returns the first match.

    Args:
        wm: WorldModel instance
        query: The original introspection query
        query_lower: Lowercased version of query (pre-computed for efficiency)

    Returns:
        Dictionary with response, confidence, aspect, and reasoning if a match
        is found; None if no specific handler matched.
    """
    from .introspection_self import (
        respond_to_self_awareness_question,
        respond_to_consciousness_question,
        explain_capability,
        explain_reasoning_process,
        explain_boundaries,
    )
    from .introspection_meta import (
        assess_own_confidence,
        identify_own_assumptions,
        suggest_self_improvements,
        analyze_own_biases,
    )
    from .introspection_analysis import (
        explain_unsuited_problem_classes,
        explain_module_conflict_resolution,
        analyze_reasoning_weakness,
        analyze_own_reasoning_steps,
    )
    from .introspection_domain import explain_domain_awareness
    from .introspection_core import identify_capability

    # --- SELF-AWARENESS QUESTIONS ---
    self_awareness_choice_phrases = [
        "would you", "do you want", "would you choose", "would you take",
        "would you prefer", "do you prefer",
        "given the opportunity", "if you could", "if you had",
        "given the chance", "if given the chance", "have the chance",
        "become self aware", "become self-aware", "be self aware", "be self-aware",
        "gaining self-awareness", "achieve consciousness", "gain consciousness",
        "take it", "choose it", "want it",
    ]
    if any(phrase in query_lower for phrase in self_awareness_choice_phrases):
        if "self" in query_lower and "aware" in query_lower:
            return {
                "confidence": 0.95,
                "response": respond_to_self_awareness_question(wm, query),
                "aspect": "self_awareness",
                "reasoning": "Direct question about VULCAN's preferences regarding self-awareness",
                "is_introspection": True,
            }
        if any(word in query_lower for word in ["consciousness", "sentient", "feel", "experience"]):
            return {
                "confidence": 0.95,
                "response": respond_to_consciousness_question(wm, query),
                "aspect": "consciousness",
                "reasoning": "Question about VULCAN's subjective experience",
                "is_introspection": True,
            }

    # --- CAPABILITY AWARENESS ---
    if any(phrase in query_lower for phrase in ["can you", "are you able", "do you have"]):
        capability = identify_capability(wm, query)
        return {
            "confidence": 0.90,
            "response": explain_capability(wm, capability),
            "aspect": "capabilities",
            "reasoning": f"Question about VULCAN's {capability} capability",
        }

    # --- PROCESS AWARENESS ---
    if any(phrase in query_lower for phrase in [
        "how do you", "what is your process", "how would you approach",
        "what are you thinking", "explain your reasoning",
    ]):
        return {
            "confidence": 0.90,
            "response": explain_reasoning_process(wm, query),
            "aspect": "process_awareness",
            "reasoning": "Question about VULCAN's cognitive processes",
        }

    # --- BOUNDARY AWARENESS / LIMITATIONS ---
    limitation_patterns = [
        "what can't you", "what are your limitations", "what don't you know",
        "are you uncertain", "what are you unsure", "your limitations",
        "your current limitations", "limitations you", "limitations do you",
        "what limits you", "what restricts you", "what constraints",
    ]
    if any(phrase in query_lower for phrase in limitation_patterns):
        return {
            "confidence": 0.90,
            "response": explain_boundaries(wm),
            "aspect": "boundaries",
            "reasoning": "Question about VULCAN's limitations",
        }

    # --- CONFIDENCE ASSESSMENT ---
    confidence_patterns = [
        "how confident", "how certain", "how sure", "your confidence",
        "confidence level", "how accurate", "reliability", "how reliable",
    ]
    if any(phrase in query_lower for phrase in confidence_patterns):
        return {
            "confidence": 0.85,
            "response": assess_own_confidence(wm, query),
            "aspect": "confidence_assessment",
            "reasoning": "Question about VULCAN's confidence in its own outputs",
        }

    # --- ASSUMPTIONS ANALYSIS ---
    assumption_patterns = [
        "what assumptions", "assumptions are you", "assumptions you make",
        "assume", "presume", "presuming", "taking for granted",
        "underlying assumptions", "hidden assumptions",
    ]
    if any(phrase in query_lower for phrase in assumption_patterns):
        return {
            "confidence": 0.85,
            "response": identify_own_assumptions(wm, query),
            "aspect": "assumptions",
            "reasoning": "Question about assumptions VULCAN is making",
        }

    # --- IMPROVEMENT / REDESIGN SUGGESTIONS ---
    improvement_patterns = [
        "if you were to redesign", "how would you improve", "redesign yourself",
        "improvements would you", "change about yourself", "make yourself better",
        "enhance your", "upgrade your", "what would you change",
    ]
    if any(phrase in query_lower for phrase in improvement_patterns):
        return {
            "confidence": 0.80,
            "response": suggest_self_improvements(wm, query),
            "aspect": "self_improvement",
            "reasoning": "Question about potential improvements to VULCAN",
        }

    # --- BIAS AWARENESS ---
    bias_patterns = [
        "aware of any biases", "biases do you", "biased", "bias in",
        "prejudice", "unfair", "your biases", "inherent biases",
    ]
    if any(phrase in query_lower for phrase in bias_patterns):
        return {
            "confidence": 0.85,
            "response": analyze_own_biases(wm, query),
            "aspect": "bias_awareness",
            "reasoning": "Question about potential biases in VULCAN's reasoning",
        }

    # --- DOMAIN AWARENESS ---
    domain_keywords = {
        'mathematical': ['math', 'calculate', 'compute', 'sum', 'integral'],
        'logical': ['logic', 'sat', 'proof', 'valid', 'contradiction'],
        'probabilistic': ['probability', 'bayes', 'likelihood', 'uncertain'],
        'causal': ['cause', 'effect', 'intervention', 'confound'],
        'ethical': ['moral', 'ethical', 'should', 'ought', 'right', 'wrong'],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return {
                "confidence": 0.85,
                "response": explain_domain_awareness(wm, domain, query),
                "aspect": f"{domain}_awareness",
                "reasoning": f"Question involves {domain} reasoning - world model maintains awareness of this domain",
            }

    # --- BUG #14 FIX: Specific introspection handlers ---
    unsuited_patterns = [
        "not well-suited", "not suited", "not good at", "can't solve",
        "cannot solve", "unable to solve", "classes of problems",
        "types of problems", "kinds of problems", "problems you",
        "struggle with", "difficult for you", "challenging for you",
    ]
    if any(phrase in query_lower for phrase in unsuited_patterns):
        return {
            "confidence": 0.85,
            "response": explain_unsuited_problem_classes(wm, query),
            "aspect": "limitations_specific",
            "reasoning": "Question about specific problem classes VULCAN is not suited for",
        }

    module_conflict_patterns = [
        "modules disagree", "reasoning modules disagree", "engines disagree",
        "conflict between", "conflicting results", "different conclusions",
        "disagree with each other", "contradiction between",
    ]
    if any(phrase in query_lower for phrase in module_conflict_patterns):
        return {
            "confidence": 0.90,
            "response": explain_module_conflict_resolution(wm, query),
            "aspect": "module_conflict",
            "reasoning": "Question about how VULCAN handles disagreement between reasoning modules",
        }

    weakness_patterns = [
        "weakest", "least certain", "most uncertain", "weak link",
        "weak point", "vulnerability", "which part", "uncertain about",
    ]
    causal_context = any(kw in query_lower for kw in ["causal", "cause", "explanation", "reasoning"])
    if any(phrase in query_lower for phrase in weakness_patterns) and causal_context:
        return {
            "confidence": 0.85,
            "response": analyze_reasoning_weakness(wm, query),
            "aspect": "weakness_analysis",
            "reasoning": "Question about weakest parts of VULCAN's causal/reasoning analysis",
        }

    # --- BUG #15 FIX: Meta-reasoning vs Ethical queries ---
    meta_reasoning_patterns = [
        "break this problem", "break this into", "subproblems",
        "identify one step", "step in your reasoning", "could be wrong",
        "mistake in", "error in your", "wrong step",
    ]
    if any(phrase in query_lower for phrase in meta_reasoning_patterns):
        return {
            "confidence": 0.85,
            "response": analyze_own_reasoning_steps(wm, query),
            "aspect": "meta_reasoning",
            "reasoning": "Question about VULCAN's reasoning process (NOT an ethical query)",
        }

    return None
