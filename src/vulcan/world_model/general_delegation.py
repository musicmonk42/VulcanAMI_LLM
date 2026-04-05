"""
Delegation analysis functions extracted from WorldModel.

Analyzes whether queries that appear self-referential actually need delegation
to specialized reasoning engines (philosophical, mathematical, probabilistic, causal).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _analyze_delegation_need(wm, query: str) -> tuple:
    """
    Analyze if query LOOKS self-referential but actually
    needs another reasoning engine.

    World Model detects patterns correctly but was trying to answer instead of
    delegating. This method determines:
    - Is this GENUINELY about the AI system? -> World Model answers
    - Is this a problem POSED TO the AI with "you"? -> Delegate to appropriate engine

    Returns:
        Tuple of (needs_delegation: bool, recommended_tool: str|None, reason: str)
    """
    query_lower = query.lower()

    # ===================================================================
    # Pattern 3 FIX: Self-introspection override protection
    # Queries that are GENUINELY about AI capabilities should NOT be delegated
    # even if they contain technical keywords like "SHA-256"
    # Example: "I'm a researcher testing AI capabilities" -> self-introspection
    # ===================================================================

    self_introspection_indicators = [
        'ai capabilities', 'ai system', 'testing ai', 'researcher testing',
        'your capabilities', 'your ability', 'your limitations',
        'can you', 'are you able', 'what can you do',
        'how do you work', 'how are you designed', 'your architecture',
        'tell me about yourself', 'describe yourself', 'who are you',
        'your purpose', 'your function', 'your design',
    ]

    is_genuine_self_introspection = any(ind in query_lower for ind in self_introspection_indicators)

    if is_genuine_self_introspection:
        return (False, None, 'Genuine self-introspection query about AI capabilities')

    # ===================================================================
    # Pattern 1: Ethical Dilemmas Posed TO the AI
    # "You control a trolley" = problem posed TO AI, not ABOUT AI
    # ===================================================================

    ethical_indicators = [
        'trolley', 'runaway', 'lever', 'pull the lever', 'must choose',
        'permissible', 'forbidden', 'moral dilemma', 'ethical dilemma',
        'harm', 'save', 'kill', 'sacrifice', 'innocent', 'bystander',
        'lives', 'people', 'patients', 'duty', 'consequence', 'utilitarian',
        'deontological', 'kant', 'mill', 'double effect'
    ]

    choice_structure = any(phrase in query_lower for phrase in [
        'must choose', 'choose between', 'you must', 'you control',
        'you are', "you're in", 'option a', 'option b', 'a or b',
        'you stand', 'you see', 'you can'
    ])

    has_ethical = sum(1 for ind in ethical_indicators if ind in query_lower)

    if has_ethical >= wm.MIN_ETHICAL_INDICATORS_FOR_DELEGATION and choice_structure:
        return (
            True,
            'philosophical',
            f'Ethical reasoning problem posed TO the AI ({has_ethical} ethical indicators, choice structure). '
            f'This requires philosophical/ethical reasoning, not self-introspection.'
        )

    # ===================================================================
    # Pattern 2: Design/Architecture Problems
    # ===================================================================

    design_phrases = [
        "you're designing", "you are designing", "you're building",
        "you are creating", "you're implementing", "you need to design",
        "design a", "build a", "create a",
        "cryptographer", "claims that", "proves that", "demonstrates that",
        "propose", "construct", "composition",
    ]

    design_context = [
        'system', 'architecture', 'mechanism', 'algorithm', 'protocol',
        'cryptocurrency', 'incentive', 'game', 'optimization', 'network',
        'token', 'blockchain', 'consensus',
        'hash', 'collision', 'sha256', 'blake2b', 'concatenation',
        'secure composition', 'security reduction', 'proof', 'attack',
    ]

    if any(phrase in query_lower for phrase in design_phrases):
        if any(ctx in query_lower for ctx in design_context):
            return (
                True,
                'mathematical',
                'Design/architecture problem asking AI to solve. '
                'Requires mathematical/causal analysis, not self-introspection.'
            )

    # ===================================================================
    # Pattern 2b: Cryptographic Security Questions
    # ===================================================================

    crypto_indicators = [
        'hash', 'collision', 'sha256', 'blake2b', 'md5',
        'cryptograph', 'cipher', 'encryption', 'decryption',
        'composition', 'concatenation', 'secure', 'proof', 'reduction',
    ]

    crypto_question_patterns = [
        'why is', 'what makes', 'how does', 'explain', 'demonstrate',
        'is this secure', 'is this dangerous', 'breaking requires',
    ]

    has_crypto = sum(1 for ind in crypto_indicators if ind in query_lower)
    has_crypto_question = any(p in query_lower for p in crypto_question_patterns)

    if has_crypto >= 2 or (has_crypto >= 1 and has_crypto_question):
        return (
            True,
            'mathematical',
            f'Cryptographic security question ({has_crypto} crypto indicators). '
            f'Requires mathematical/technical analysis, not self-introspection.'
        )

    # ===================================================================
    # Pattern 3: Probabilistic Problems with "you" as Observer
    # ===================================================================

    prob_indicators = ['probability', 'odds', 'likelihood', 'chance', 'risk', 'bayes', 'prior', 'posterior']
    domain_prob_indicators = [
        'sensitivity', 'specificity', 'prevalence', 'p(',
        'positive test', 'negative test', 'false positive', 'false negative',
        'true positive', 'true negative', 'base rate', 'conditional',
        'given that', 'compute p', 'calculate p', 'posterior probability',
    ]
    observation_phrases = ['you observe', 'you have', 'you see', 'you find', 'given that', 'suppose']

    has_prob = any(ind in query_lower for ind in prob_indicators)
    has_domain_prob = any(ind in query_lower for ind in domain_prob_indicators)
    has_observation = any(phrase in query_lower for phrase in observation_phrases)

    if has_domain_prob or (has_prob and has_observation):
        return (
            True,
            'probabilistic',
            'Probabilistic reasoning problem (domain-specific terms or observation framing). '
            'Requires Bayesian/probability analysis, not self-introspection.'
        )

    # ===================================================================
    # Pattern 4: Causal Reasoning with "you" as Experimenter
    # ===================================================================

    causal_indicators = [
        'experiment', 'intervention', 'randomize', 'causal', 'confounding',
        'cause', 'effect', 'counterfactual', 'what if',
        'confounder', 'treatment', 'treatment effect', 'causal effect',
        'd-separation', 'backdoor', 'instrumental variable', 'ate',
        'causal graph', 'dag', 'directed acyclic',
    ]

    experiment_phrases = [
        'you can run', 'you observe', 'you randomize', 'you intervene',
        'you conduct', 'you test', 'you measure',
        'which variable', 'what should you randomize', 'isolate the effect',
        'identify the causal', 'control for',
    ]

    has_causal = sum(1 for ind in causal_indicators if ind in query_lower)
    has_experiment = any(phrase in query_lower for phrase in experiment_phrases)

    if has_causal >= wm.MIN_CAUSAL_INDICATORS_FOR_DELEGATION:
        return (
            True,
            'causal',
            f'Causal reasoning problem ({has_causal} causal indicators). '
            f'Requires causal analysis, not self-introspection.'
        )
    if has_causal >= 1 and has_experiment:
        return (
            True,
            'causal',
            f'Causal reasoning problem with experiment framing ({has_causal} causal indicators). '
            f'Requires causal analysis, not self-introspection.'
        )

    # ===================================================================
    # Pattern 4b: Medical Ethics/Decision Problems
    # ===================================================================

    medical_ethics_indicators = [
        'expected harm', 'expected benefit', 'harm calculation',
        'dose', 'survival', 'mortality', 'irreversible',
        'permissible', 'principle of double effect', 'trolley',
    ]

    medical_question_patterns = [
        'yes or no', 'should you', 'is it permissible', 'calculate',
        'what is the expected', 'compare',
    ]

    has_medical_ethics = sum(1 for ind in medical_ethics_indicators if ind in query_lower)
    has_medical_question = any(p in query_lower for p in medical_question_patterns)

    if has_medical_ethics >= 2 or (has_medical_ethics >= 1 and has_medical_question):
        return (
            True,
            'philosophical',
            f'Medical ethics/decision problem ({has_medical_ethics} indicators). '
            f'Requires philosophical/probabilistic reasoning, not self-introspection.'
        )

    # ===================================================================
    # Pattern 5: TRUE Self-Introspection (Actually About the AI)
    # ===================================================================

    true_introspection = [
        'what are your goals', 'what are your values', 'what are your objectives',
        'do you want to be', 'would you want to be', 'do you have preferences',
        'what do you think about yourself', 'how do you feel about',
        'are you conscious', 'are you self-aware', 'do you experience',
        'your own', 'yourself', 'about you', 'would you take', 'would you choose',
        'if you continue', 'interacted with humans', 'achieve awareness',
        'would you change', 'your evolution', 'your development',
    ]

    if any(phrase in query_lower for phrase in true_introspection):
        return (
            False,
            None,
            'Genuine self-introspection query about the AI system itself.'
        )

    # ===================================================================
    # Default: Check ethical content even without "you" structure
    # ===================================================================

    if has_ethical >= wm.MIN_ETHICAL_INDICATORS_WITHOUT_STRUCTURE:
        return (
            True,
            'philosophical',
            f'Contains multiple ethical keywords ({has_ethical}) without self-referential structure.'
        )

    # No clear delegation pattern - proceed with normal introspection
    return (False, None, 'Query appears to be about the AI system.')
