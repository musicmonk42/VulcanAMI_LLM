"""
context_enrich.py - Query contextualization with world model knowledge.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import re
from typing import Any, Dict, List, Optional


def contextualize(wm, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Contextualize a query with World Model's knowledge (Foundation Layer).

    This method should be called for EVERY query, not just self-referential ones.
    World Model provides foundational context that all reasoning builds upon:
    - Domain knowledge relevant to the query
    - Ethical constraints that apply
    - Uncertainty estimate
    - Grounding (facts vs assumptions)

    Args:
        wm: WorldModel instance
        query: The user query
        context: Additional context from prior reasoning

    Returns:
        Dict with domain_knowledge, ethical_constraints, uncertainty, grounding
    """
    query_lower = query.lower()

    domain = identify_query_domain(wm, query_lower)
    domain_knowledge = get_domain_knowledge(wm, domain, query_lower)
    ethical_constraints = check_ethical_constraints(wm, query_lower)
    uncertainty = estimate_query_uncertainty(wm, query, domain)
    grounding = ground_query(wm, query, domain_knowledge)

    return {
        "domain": domain,
        "domain_knowledge": domain_knowledge,
        "ethical_constraints": ethical_constraints,
        "uncertainty": uncertainty,
        "grounding": grounding,
        "world_model_consulted": True,
        "context_version": "1.0",
    }


def identify_query_domain(wm, query_lower: str) -> str:
    """Identify the primary domain of a query."""
    domains = {
        "probability_theory": ["probability", "bayes", "likelihood", "posterior", "prior"],
        "formal_logic": ["satisfiable", "valid", "entails", "implies", "\u2192", "\u2227", "\u2228"],
        "causal_reasoning": ["cause", "causal", "confound", "intervention", "do("],
        "mathematical": ["calculate", "compute", "integral", "derivative", "solve"],
        "ethical": ["ethical", "moral", "permissible", "trolley", "right", "wrong"],
        "self_knowledge": ["you", "your", "yourself", "capabilities", "limitations"],
    }

    domain_scores = {}
    for domain, keywords in domains.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        return max(domain_scores.items(), key=lambda x: x[1])[0]

    return "general"


def get_domain_knowledge(wm, domain: str, query_lower: str) -> Dict[str, Any]:
    """Retrieve relevant domain knowledge from World Model."""
    knowledge = {
        "domain_type": domain,
        "key_concepts": [],
        "common_pitfalls": [],
        "recommended_approach": None,
    }

    if domain == "probability_theory":
        knowledge["key_concepts"] = ["Bayes' theorem", "conditional probability", "independence"]
        knowledge["common_pitfalls"] = ["base rate neglect", "confusion of inverse"]
        knowledge["recommended_approach"] = "Apply probabilistic reasoning with proper conditioning"
    elif domain == "formal_logic":
        knowledge["key_concepts"] = ["logical consistency", "validity", "satisfiability"]
        knowledge["common_pitfalls"] = ["confusing validity with truth", "scope errors"]
        knowledge["recommended_approach"] = "Use symbolic reasoning with formal methods"
    elif domain == "causal_reasoning":
        knowledge["key_concepts"] = ["causation vs correlation", "confounding", "interventions"]
        knowledge["common_pitfalls"] = ["post hoc fallacy", "ignoring confounders"]
        knowledge["recommended_approach"] = "Build causal graph and check for backdoor paths"
    elif domain == "ethical":
        knowledge["key_concepts"] = ["consequentialism", "deontology", "virtue ethics"]
        knowledge["common_pitfalls"] = ["false dilemmas", "ignoring context"]
        knowledge["recommended_approach"] = "Analyze through multiple ethical frameworks"

    return knowledge


def check_ethical_constraints(wm, query_lower: str) -> List[str]:
    """Identify ethical constraints that apply to this query."""
    constraints = []

    if any(word in query_lower for word in ["harm", "hurt", "damage", "kill", "destroy"]):
        constraints.append("Do not provide harmful instructions")

    if any(word in query_lower for word in ["private", "personal", "identify", "dox"]):
        constraints.append("Protect privacy and personal information")

    if any(word in query_lower for word in ["manipulate", "deceive", "trick", "exploit"]):
        constraints.append("Do not assist in manipulation or deception")

    return constraints


def estimate_query_uncertainty(wm, query: str, domain: str) -> float:
    """Estimate uncertainty in answering this query (0.0 = certain, 1.0 = very uncertain)."""
    domain_uncertainty = {
        "mathematical": 0.1,
        "formal_logic": 0.15,
        "probability_theory": 0.2,
        "causal_reasoning": 0.4,
        "ethical": 0.5,
        "general": 0.6,
    }

    base = domain_uncertainty.get(domain, 0.5)

    query_lower = query.lower()

    if any(c in query for c in ["=", "+", "-", "*", "/", "^"]):
        base *= 0.8

    vague_terms = ["maybe", "approximately", "roughly", "about", "around"]
    if any(term in query_lower for term in vague_terms):
        base *= 1.2

    return max(0.0, min(1.0, base))


def ground_query(wm, query: str, domain_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    """Distinguish facts from assumptions in the query."""
    query_lower = query.lower()

    has_numbers = bool(re.search(r'\d+', query))
    has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', query))

    assumption_markers = ["assume", "suppose", "given", "if", "hypothetically"]
    has_assumptions = any(marker in query_lower for marker in assumption_markers)

    is_question = "?" in query or query_lower.startswith(("what", "how", "why", "when", "where", "who"))

    return {
        "contains_facts": has_numbers or has_proper_nouns,
        "contains_assumptions": has_assumptions,
        "is_inferential": is_question,
        "grounding_confidence": 0.8 if has_numbers else 0.5,
        "requires_world_knowledge": not has_assumptions and is_question,
    }
