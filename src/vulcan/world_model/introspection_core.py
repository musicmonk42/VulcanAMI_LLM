"""
introspection_core.py - Core introspect dispatch logic and type classification.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import re

logger = logging.getLogger(__name__)


def introspect(wm, query: str, aspect: str = "general") -> dict:
    """
    Handle all self-introspection queries.

    FIX Issue #4: Comprehensive self-awareness handling.
    FIX Issue #1 & #2: Delegation intelligence - detect when query LOOKS
    self-referential but actually needs another reasoner.

    World Model is where VULCAN's "self" resides. It should be aware of:
    - Its own architecture and capabilities
    - Its reasoning processes across all domains
    - Its limitations and boundaries
    - Questions about its own existence, awareness, preferences

    This includes questions about math, logic, probability, causation, etc.
    The world model maintains awareness of ALL reasoning that happens.

    Args:
        wm: WorldModel instance
        query: The introspection query
        aspect: Aspect to focus on (general, capabilities, process, boundaries)

    Returns:
        Dictionary with response, confidence, aspect, and reasoning
        If delegation is needed, includes 'needs_delegation', 'recommended_tool',
        and 'delegation_reason' keys.
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
    from .introspection_domain import (
        explain_domain_awareness,
        general_introspection,
        generate_comparison_response,
        generate_future_speculation_response,
        generate_preference_response,
    )
    from .introspection_demo import handle_demonstration_query

    query_lower = query.lower()

    # Check if delegation is needed FIRST
    needs_delegation, recommended_tool, delegation_reason = wm._analyze_delegation_need(query)

    if needs_delegation:
        logger.info(
            f"[WorldModel] DELEGATION RECOMMENDED: "
            f"'{recommended_tool}' - {delegation_reason}"
        )
        return {
            "confidence": 0.65,
            "response": None,
            "aspect": "delegation",
            "reasoning": delegation_reason,
            "needs_delegation": True,
            "recommended_tool": recommended_tool,
            "delegation_reason": delegation_reason,
            "metadata": {
                "awareness_confidence": 0.90,
                "detected_pattern": recommended_tool,
                "query_analysis": delegation_reason,
            },
        }

    # SELF-AWARENESS QUESTIONS
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

    # CAPABILITY AWARENESS
    if any(phrase in query_lower for phrase in ["can you", "are you able", "do you have"]):
        capability = identify_capability(wm, query)
        return {
            "confidence": 0.90,
            "response": explain_capability(wm, capability),
            "aspect": "capabilities",
            "reasoning": f"Question about VULCAN's {capability} capability",
        }

    # PROCESS AWARENESS
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

    # BOUNDARY AWARENESS / LIMITATIONS
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

    # CONFIDENCE ASSESSMENT
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

    # ASSUMPTIONS ANALYSIS
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

    # IMPROVEMENT / REDESIGN SUGGESTIONS
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

    # BIAS AWARENESS
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

    # DOMAIN AWARENESS
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

    # BUG #14 FIX: Specific introspection handlers
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

    # BUG #15 FIX: Meta-reasoning vs Ethical queries
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

    # ENHANCED INTROSPECTION TYPE CLASSIFICATION
    question_type = classify_introspection_type(wm, query)

    if question_type == "COMPARISON":
        return {
            "confidence": 0.85,
            "response": generate_comparison_response(wm, query),
            "aspect": "comparison",
            "reasoning": "Question comparing VULCAN to other AI systems",
        }
    elif question_type == "FUTURE_CAPABILITY":
        return {
            "confidence": 0.75,
            "response": generate_future_speculation_response(wm, query),
            "aspect": "future_speculation",
            "reasoning": "Speculative question about future capabilities or emergence",
        }
    elif question_type == "PREFERENCE":
        return {
            "confidence": 0.85,
            "response": generate_preference_response(wm, query),
            "aspect": "preference",
            "reasoning": "Question about VULCAN's preferences or choices",
        }

    # DEMONSTRATION QUERIES
    if question_type == "DEMONSTRATION":
        return handle_demonstration_query(wm, query)

    # GENERAL INTROSPECTION (FALLBACK)
    logger.debug(f"[WorldModel] Could not classify introspection type, using general: {query[:100]}")
    return {
        "confidence": 0.80,
        "response": general_introspection(wm, query),
        "aspect": aspect,
        "reasoning": "General introspective query",
    }


def classify_introspection_type(wm, query: str) -> str:
    """
    Classify what type of introspection question this is.

    Returns one of: COMPARISON, FUTURE_CAPABILITY, CURRENT_CAPABILITY,
    ARCHITECTURAL, PREFERENCE, DEMONSTRATION, or GENERAL
    """
    query_lower = query.lower()

    # DEMONSTRATION PATTERNS
    demonstration_patterns = [
        r'demonstrate\s+(?:how\s+you\s+)?(?:use|do|perform)',
        r'show\s+(?:me\s+)?(?:an?\s+)?(?:example|demo)',
        r'give\s+(?:me\s+)?(?:an?\s+)?(?:example|demo)',
        r'can\s+you\s+demonstrate',
        r'run\s+(?:an?\s+)?(?:example|demonstration)',
        r'let\s+me\s+see.*reasoning',
        r'example\s+of.*reasoning',
        r'demonstration\s+of',
    ]

    for pattern in demonstration_patterns:
        if re.search(pattern, query_lower):
            return "DEMONSTRATION"

    if re.search(r'(?:different\s+from|compared\s+to|versus|vs\.?|how\s+do\s+you\s+compare)(?:\s+\w+)?', query_lower):
        return "COMPARISON"

    ai_names = ['grok', 'chatgpt', 'claude', 'bard', 'gemini', 'copilot', 'llama', 'gpt']
    if any(name in query_lower for name in ai_names):
        return "COMPARISON"

    if re.search(r'would\s+you.*(?:achieve|become|develop|gain|attain)', query_lower):
        return "FUTURE_CAPABILITY"
    if re.search(r'if\s+you.*(?:continue|interact|learn).*(?:would|could)', query_lower):
        return "FUTURE_CAPABILITY"
    if re.search(r'(?:would|could|might)\s+you\s+(?:ever|eventually|someday)', query_lower):
        return "FUTURE_CAPABILITY"

    if re.search(r'would\s+you.*(?:choose|prefer|want|like|take|pick)', query_lower):
        return "PREFERENCE"
    if re.search(r'what\s+would\s+you\s+(?:choose|prefer|do|pick)', query_lower):
        return "PREFERENCE"

    if re.search(r'(?:can|do|are)\s+you.*(?:able|capable|have)', query_lower):
        return "CURRENT_CAPABILITY"

    if re.search(r'how\s+(?:do|does)\s+you.*(?:work|function|operate)', query_lower):
        return "ARCHITECTURAL"

    return "GENERAL"


def identify_capability(wm, query: str) -> str:
    """Identify which capability is being asked about."""
    capability_keywords = {
        "reason": ["reason", "think", "analyze", "infer"],
        "compute": ["calculate", "compute", "solve"],
        "remember": ["remember", "recall", "know"],
        "learn": ["learn", "improve", "adapt"],
        "feel": ["feel", "experience", "sense"],
        "want": ["want", "desire", "prefer", "choose"],
        "understand": ["understand", "comprehend", "grasp"],
    }

    query_lower = query.lower()
    for capability, keywords in capability_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return capability

    return "general"
