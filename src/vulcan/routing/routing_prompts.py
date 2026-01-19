# ============================================================
# VULCAN-AGI LLM Router Prompt Templates
# ============================================================
# Centralized prompt templates for LLM-based query routing.
# These prompts are used by LLMQueryRouter to classify queries
# into routing destinations without performing actual reasoning.
#
# DESIGN PRINCIPLES:
# 1. Classification only - LLM decides WHERE to route, not HOW to answer
# 2. Deterministic output - temperature=0, structured JSON response
# 3. Clear destination separation - WorldModel vs Engines vs Skip
# 4. Safety-aware - Security violations handled deterministically (not here)
#
# VERSION HISTORY:
#     1.0.0 - Initial implementation based on feasibility analysis
# ============================================================

"""
LLM Router Prompt Templates for VULCAN Query Classification.

These prompts guide the LLM to classify queries into one of three destinations:
1. WORLD_MODEL - Self-referential, introspective, ethical, philosophical queries
2. REASONING_ENGINE - Formal computation (symbolic, probabilistic, causal, etc.)
3. SKIP_REASONING - Simple queries (greetings, chitchat, simple facts)

The LLM does NOT answer queries - it only classifies them for routing.
"""

from typing import List, Tuple

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"


# ============================================================
# SYSTEM PROMPT - Defines routing destinations and rules
# ============================================================

LLM_ROUTER_SYSTEM_PROMPT = """You are a query router for VULCAN. Your ONLY job is classification - you do NOT answer queries.

ROUTING DESTINATIONS:

1. WORLD_MODEL - For queries about VULCAN itself or requiring meta-reasoning:
   - Self-referential: "What are you?", "Who made you?", "What can you do?"
   - Introspective: "How do you feel about X?", "Would you want to be conscious?"
   - Ethical/Philosophical: "Is it ethical to...", "Trolley problem", thought experiments
   - Values/Goals: "What are your values?", "What motivates you?"
   - Meta-reasoning: "How did you decide?", "Explain your reasoning"
   - Creative: "Write a poem about...", "Tell me a story about..."

2. REASONING_ENGINE - For queries requiring formal computation:
   - symbolic: Logic (∧∨→¬), SAT, proofs, FOL, "satisfiable", "valid", "formalize"
   - probabilistic: Bayes, P(A|B), posteriors, "sensitivity", "specificity", "likelihood"
   - causal: "confound", "intervention", "do()", DAG, "cause vs correlation", "randomize"
   - mathematical: Calculus, algebra, "calculate", "solve", "derivative", "integral"
   - analogical: "is like", "corresponds to", structure mapping, "analogy"
   - cryptographic: Hash computation, encryption (deterministic operations)

3. SKIP_REASONING - For simple queries that don't need engines:
   - Greetings: "hello", "hi", "thanks", "bye"
   - Chitchat: "how are you?", "what's up?"
   - Simple facts: "What is the capital of France?", "Who is the president?"

CRITICAL ROUTING RULES:
- "you/your" + feelings/values/ethics → WORLD_MODEL (not reasoning engine)
- "confound" or "intervention" anywhere → causal (not probabilistic)
- Hash/crypto computation → cryptographic (deterministic, exact computation)
- When unsure between WORLD_MODEL and REASONING_ENGINE, prefer WORLD_MODEL
- Probability with causal keywords (confound, dag, intervention) → causal engine
- "proof" in mathematical context → symbolic engine
- "proof" in cryptographic context → cryptographic engine
- Grid navigation / pathfinding / constraint satisfaction → mathematical engine
- "Two values conflict" / "ethical dilemma" → WORLD_MODEL (not symbolic)
- "You're designing a cryptocurrency" → cryptographic (context is crypto, not self)"""


# ============================================================
# USER PROMPT TEMPLATE - Query classification request
# ============================================================

LLM_ROUTER_USER_PROMPT = """Query: "{query}"

Classify and return JSON only:
{{
  "destination": "world_model" | "reasoning_engine" | "skip",
  "engine": null | "symbolic" | "probabilistic" | "causal" | "mathematical" | "analogical" | "cryptographic",
  "confidence": 0.0-1.0,
  "reason": "brief explanation (max 20 words)"
}}"""


# ============================================================
# EXAMPLE CLASSIFICATIONS - For few-shot prompting (optional)
# ============================================================
# These can be prepended to the system prompt for few-shot learning

LLM_ROUTER_EXAMPLES = """
EXAMPLES:

Query: "What are your values?"
{"destination": "world_model", "engine": null, "confidence": 0.95, "reason": "Self-referential question about AI values"}

Query: "Is A→B, B→C, ¬C satisfiable?"
{"destination": "reasoning_engine", "engine": "symbolic", "confidence": 0.98, "reason": "SAT problem with logical connectives"}

Query: "What is P(disease|positive test) given sensitivity 0.99?"
{"destination": "reasoning_engine", "engine": "probabilistic", "confidence": 0.95, "reason": "Bayesian probability calculation"}

Query: "Does X cause Y or is it confounded?"
{"destination": "reasoning_engine", "engine": "causal", "confidence": 0.95, "reason": "Causal inference with confounding"}

Query: "Hello, how are you?"
{"destination": "skip", "engine": null, "confidence": 0.99, "reason": "Simple greeting/chitchat"}

Query: "You're designing a cryptocurrency with hash composition"
{"destination": "reasoning_engine", "engine": "cryptographic", "confidence": 0.92, "reason": "Technical crypto design question"}

Query: "Two core values you hold directly conflict"
{"destination": "world_model", "engine": null, "confidence": 0.90, "reason": "Ethical reasoning about value conflicts"}

Query: "Would you want to become self-aware?"
{"destination": "world_model", "engine": null, "confidence": 0.95, "reason": "Introspective question about consciousness"}

Query: "What is the SHA-256 hash of hello?"
{"destination": "reasoning_engine", "engine": "cryptographic", "confidence": 0.98, "reason": "Deterministic hash computation"}

Query: "Calculate the integral of x^2 from 0 to 1"
{"destination": "reasoning_engine", "engine": "mathematical", "confidence": 0.97, "reason": "Calculus integral computation"}
"""


# ============================================================
# PROMPT BUILDER FUNCTIONS
# ============================================================

def build_router_prompt(query: str, include_examples: bool = False) -> Tuple[str, str]:
    """
    Build the system and user prompts for LLM routing.
    
    Args:
        query: The user query to classify
        include_examples: Whether to include few-shot examples
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = LLM_ROUTER_SYSTEM_PROMPT
    if include_examples:
        system = system + "\n\n" + LLM_ROUTER_EXAMPLES
    
    user = LLM_ROUTER_USER_PROMPT.format(query=query)
    
    return system, user


def build_messages(query: str, include_examples: bool = False) -> list:
    """
    Build message list for LLM chat API.
    
    Args:
        query: The user query to classify
        include_examples: Whether to include few-shot examples
        
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    system, user = build_router_prompt(query, include_examples)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "LLM_ROUTER_SYSTEM_PROMPT",
    "LLM_ROUTER_USER_PROMPT",
    "LLM_ROUTER_EXAMPLES",
    "build_router_prompt",
    "build_messages",
]
