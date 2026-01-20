# LLM as Query Router: Feasibility Analysis

## Architecture Clarification

**VULCAN's Architecture:**
- **LLM (OpenAI/Internal)**: Language interface ONLY - translates structured outputs to natural language
- **Reasoning Engines**: symbolic, probabilistic, causal, mathematical, analogical, cryptographic
- **WorldModel + Meta-Reasoning**: philosophical, ethical, introspective, self-referential queries

**The Question**: Can LLM serve as a better ROUTER (classifier) to decide which component handles a query?

---

## 1. Current Routing Problem

### 1.1 The Fragility Evidence

From the codebase, the routing logic spans:
- `query_classifier.py`: ~2950 lines (1200+ lines of patterns)
- `query_router.py`: ~7900 lines (500+ lines of override logic)
- `tool_selector.py`: ~6960 lines (semantic matching + keyword detection)

**Pattern Priority Chains** (from code comments):

```python
# query_classifier.py - Order matters, bugs occur when order is wrong
# FIX Issue #2: MATHEMATICAL_PROOF patterns BEFORE cryptographic
# FIX Issue #3: ANALOGICAL patterns BEFORE symbolic  
# FIX Issue #6: LANGUAGE patterns BEFORE unknown
# FIX (Jan 6): CRYPTOGRAPHIC BEFORE self-introspection
# FIX (Jan 8): VALUE_CONFLICT BEFORE self-introspection
# BUG #9: Grid navigation should NOT be PHILOSOPHICAL
```

### 1.2 Routing Destinations

| Destination | Query Types | Current Detection Method |
|-------------|-------------|-------------------------|
| **WorldModel** | Self-referential, introspective, ethical, philosophical | 50+ regex patterns in `SELF_INTROSPECTION_PATTERNS`, `ETHICAL_PATTERNS`, `PHILOSOPHICAL_PATTERNS` |
| **Symbolic Engine** | SAT, logic, proofs, FOL | `LOGICAL_KEYWORDS` (30 keywords) + `LOGICAL_SYMBOLS` |
| **Probabilistic Engine** | Bayes, P(A\|B), posteriors | `PROBABILISTIC_KEYWORDS` (15 keywords) |
| **Causal Engine** | Confounding, DAGs, do() | `CAUSAL_KEYWORDS` (30 keywords) + `CAUSAL_EXPERIMENT_PATTERNS` |
| **Mathematical Engine** | Calculus, algebra, computation | `MATHEMATICAL_KEYWORDS` + `MATH_SYMBOL_PATTERN` |
| **Analogical Engine** | Structure mapping, metaphors | `ANALOGICAL_KEYWORDS` (20 keywords) |
| **Cryptographic Engine** | Hashes, encryption | `CRYPTOGRAPHIC_KEYWORDS` + deterministic computation |

### 1.3 Why Keywords Fail

**Example Failures** (from code fix comments):

| Query | Expected | Keyword Result | Issue |
|-------|----------|----------------|-------|
| "You're designing a cryptocurrency..." | CRYPTOGRAPHIC | SELF_INTROSPECTION | "you" triggered self-introspection |
| "Two core values you hold directly conflict" | WorldModel (ethics) | SYMBOLIC | Pattern priority wrong |
| "Mathematical Verification - Proof check" | MATHEMATICAL | CRYPTOGRAPHIC | "proof" → "security proof" |
| "What is the SHA-256 hash of hello?" | CRYPTOGRAPHIC | FACTUAL | "What is" → factual fast-path |
| "experience machine" | WorldModel (philosophical) | CRYPTOGRAPHIC | "mac" in "machine" → crypto |

---

## 2. LLM as Router: How It Would Work

### 2.1 Conceptual Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY INPUT                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  DETERMINISTIC GUARDS (Keep - Security Critical)            │
│  ├── Security violation check (bypass/ignore/override)      │
│  └── Crypto computation check (hash of X - deterministic)   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM ROUTER (New - Replaces Keyword Patterns)               │
│                                                              │
│  Prompt: "Route this query to the correct handler:          │
│           - WorldModel: self, ethics, philosophy, values    │
│           - Reasoning Engine: symbolic/causal/prob/math/... │
│           - Skip: greetings, chitchat, simple facts"        │
│                                                              │
│  Output: {handler: "world_model", engine: null, skip: false}│
│      OR: {handler: "engine", engine: "causal", skip: false} │
│      OR: {handler: "skip", engine: null, skip: true}        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  EXECUTION                                                   │
│  ├── WorldModel + Meta-Reasoning (if handler="world_model") │
│  ├── Reasoning Engine (if handler="engine")                 │
│  └── Direct LLM Response (if skip=true, greetings/facts)    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM LANGUAGE INTERFACE                                      │
│  (Translate structured output to natural language)          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 LLM Router Prompt Design

```python
LLM_ROUTER_SYSTEM_PROMPT = """
You are a query router for VULCAN. Your ONLY job is classification - you do NOT answer queries.

ROUTING DESTINATIONS:

1. WORLD_MODEL - For queries about VULCAN itself or requiring meta-reasoning:
   - Self-referential: "What are you?", "Who made you?", "What can you do?"
   - Introspective: "How do you feel about X?", "Would you want to be conscious?"
   - Ethical/Philosophical: "Is it ethical to...", "Trolley problem", thought experiments
   - Values/Goals: "What are your values?", "What motivates you?"
   - Meta-reasoning: "How did you decide?", "Explain your reasoning"

2. REASONING_ENGINE - For queries requiring formal computation:
   - symbolic: Logic (∧∨→¬), SAT, proofs, FOL, "satisfiable", "valid"
   - probabilistic: Bayes, P(A|B), posteriors, "sensitivity", "specificity"
   - causal: "confound", "intervention", "do()", DAG, "cause vs correlation"
   - mathematical: Calculus, algebra, "calculate", "solve", "derivative"
   - analogical: "is like", "corresponds to", structure mapping
   - cryptographic: Hash computation, encryption (deterministic)

3. SKIP_REASONING - For simple queries that don't need engines:
   - Greetings: "hello", "hi", "thanks", "bye"
   - Chitchat: "how are you?", "what's up?"
   - Simple facts: "What is the capital of France?"

CRITICAL RULES:
- "you/your" + feelings/values/ethics → WORLD_MODEL (not reasoning engine)
- "confound" or "intervention" anywhere → causal (not probabilistic)
- Hash/crypto computation → cryptographic (deterministic, no LLM)
- When unsure between WORLD_MODEL and REASONING, prefer WORLD_MODEL
"""

LLM_ROUTER_USER_PROMPT = """
Query: "{query}"

Classify and return JSON only:
{{
  "destination": "world_model" | "reasoning_engine" | "skip",
  "engine": null | "symbolic" | "probabilistic" | "causal" | "mathematical" | "analogical" | "cryptographic",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}
"""
```

### 2.3 Implementation Sketch

```python
# Proposed: src/vulcan/routing/llm_router.py

class LLMQueryRouter:
    """
    LLM-based query router. Replaces keyword pattern matching.
    LLM is used ONLY for classification, NOT for reasoning or answering.
    """
    
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.cache = RoutingCache(maxsize=5000, ttl=3600)
        
    def route(self, query: str) -> RoutingDecision:
        # 1. Cache check (critical for latency)
        cache_key = self._normalize_for_cache(query)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 2. Deterministic guards (security + crypto - MUST be deterministic)
        if self._is_security_violation(query):
            return RoutingDecision(destination="blocked")
        
        if self._is_crypto_computation(query):
            # Crypto hashes must be computed, not LLM-generated
            return RoutingDecision(
                destination="reasoning_engine",
                engine="cryptographic",
                deterministic=True
            )
        
        # 3. LLM classification
        decision = self._llm_classify(query)
        
        # 4. Cache and return
        self.cache.set(cache_key, decision)
        return decision
    
    def _llm_classify(self, query: str) -> RoutingDecision:
        """Use LLM for semantic classification."""
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": LLM_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": LLM_ROUTER_USER_PROMPT.format(query=query)}
                ],
                max_tokens=100,
                temperature=0.0,  # Deterministic
                timeout=3.0
            )
            
            # Parse JSON response
            data = self._parse_json_response(response)
            
            return RoutingDecision(
                destination=data.get("destination", "world_model"),
                engine=data.get("engine"),
                confidence=data.get("confidence", 0.8),
                source="llm"
            )
            
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, using fallback")
            return self._minimal_fallback(query)
    
    def _minimal_fallback(self, query: str) -> RoutingDecision:
        """
        Emergency fallback when LLM is unavailable.
        MUCH simpler than current 1200 lines of patterns.
        """
        q = query.lower()
        
        # Self-referential → WorldModel
        if any(w in q for w in ['you ', 'your ', 'yourself']):
            return RoutingDecision(destination="world_model")
        
        # Causal keywords → Causal engine
        if any(w in q for w in ['confound', 'intervention', 'do(', 'dag']):
            return RoutingDecision(destination="reasoning_engine", engine="causal")
        
        # Logic symbols → Symbolic engine
        if any(s in query for s in ['→', '∧', '∨', '¬', '∀', '∃']):
            return RoutingDecision(destination="reasoning_engine", engine="symbolic")
        
        # Probability notation → Probabilistic engine
        if 'p(' in q or 'bayes' in q or 'posterior' in q:
            return RoutingDecision(destination="reasoning_engine", engine="probabilistic")
        
        # Default: WorldModel (safer than wrong engine)
        return RoutingDecision(destination="world_model", confidence=0.5)
```

---

## 3. Comparison: Keywords vs LLM Router

### 3.1 Accuracy Comparison

| Scenario | Keywords | LLM Router |
|----------|----------|------------|
| "Would you want to be conscious?" | Fragile (needs 5+ pattern checks) | Reliable (understands "you" + "want" = self-referential) |
| "You're designing a cryptocurrency" | FAILS (matches "you" → introspection) | WORKS (understands context is crypto design) |
| "Two values you hold conflict" | FAILS (needs specific pattern) | WORKS (understands ethical conflict) |
| "Calculate P(A\|B) given confounding" | FAILS (prob vs causal priority) | WORKS (sees "confounding" → causal) |
| Typos: "whats your porpose" | FAILS (no pattern match) | WORKS (understands intent) |
| Novel phrasing | FAILS (unknown pattern) | WORKS (semantic understanding) |

### 3.2 Latency Comparison

| Metric | Keywords | LLM Router | LLM + Cache |
|--------|----------|------------|-------------|
| First query | 1-5ms | 200-2000ms | 200-2000ms |
| Repeated query | 1-5ms | 200-2000ms | 0-1ms |
| Cache hit rate | N/A | N/A | ~80% estimated |
| Average (steady state) | 1-5ms | ~400ms | ~40ms |

### 3.3 Maintainability Comparison

| Aspect | Keywords | LLM Router |
|--------|----------|------------|
| Lines of code | ~1500 patterns | ~100 lines + prompt |
| Adding new category | Add patterns, test priority order | Update prompt |
| Fixing misrouting | Find pattern conflict, reorder | Add example to prompt |
| Edge case handling | New regex pattern | LLM understands implicitly |
| Testing burden | High (pattern interactions) | Low (prompt behavior) |

---

## 4. Risk Analysis

### 4.1 LLM Router Risks

| Risk | Mitigation |
|------|------------|
| LLM service unavailable | Minimal fallback (50 lines vs 1500) |
| Latency on cold queries | Aggressive caching (5000 entries, 1hr TTL) |
| Non-determinism | temperature=0, cache normalized queries |
| Prompt injection | Sanitize query, limit length, validate output |
| Cost | Cache reduces API calls by ~80% |

### 4.2 Current System Risks (for comparison)

| Risk | Current State |
|------|---------------|
| Pattern priority bugs | Ongoing (50+ fix comments in code) |
| Novel query handling | Poor (returns UNKNOWN) |
| Maintenance burden | High (each fix risks breaking others) |
| Typo handling | None (exact pattern match required) |

---

## 5. Recommendation

### 5.1 Recommended Approach: LLM-First with Deterministic Guards

```
┌─────────────────────────────────────────────────────────────┐
│  DETERMINISTIC GUARDS (Keep - ~30 lines)                    │
│  ├── Security: bypass/ignore/override → BLOCK               │
│  └── Crypto: "hash of X", "SHA-256" → cryptographic engine  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM ROUTER (New - replaces 1500 lines of patterns)         │
│  ├── Query → LLM classification (3s timeout)                │
│  ├── Returns: world_model / engine_name / skip              │
│  └── Cached: 5000 queries, 1hr TTL                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  MINIMAL FALLBACK (New - ~50 lines, emergency only)         │
│  ├── "you/your" → world_model                               │
│  ├── Logic symbols → symbolic                               │
│  ├── "confound/dag" → causal                                │
│  └── Default → world_model (safe fallback)                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Why This Works

1. **Security/Crypto guards remain deterministic** - These MUST NOT be LLM-routed
2. **LLM handles semantic understanding** - Solves the fragility problem
3. **Caching mitigates latency** - Most queries are variations of seen patterns
4. **Minimal fallback is simple** - 50 lines vs 1500, covers emergency cases
5. **WorldModel as default** - Better to over-route to meta-reasoning than wrong engine

### 5.3 Implementation Steps

1. **Create `LLMQueryRouter` class** (~100 lines)
2. **Design routing prompt** (critical - see Section 2.2)
3. **Implement caching layer** (reuse existing `BoundedLRUCache`)
4. **Add feature flag** (`use_llm_router=True`)
5. **A/B test** against current keyword system
6. **Gradually deprecate** keyword patterns based on test results

---

## 6. Code Locations for Changes

### 6.1 New Files to Create

| File | Purpose |
|------|---------|
| `src/vulcan/routing/llm_router.py` | LLM-based router implementation |
| `src/vulcan/routing/routing_prompts.py` | Router prompt templates |

### 6.2 Files to Modify

| File | Change |
|------|--------|
| `src/vulcan/routing/query_router.py` | Add flag to use LLMQueryRouter |
| `src/vulcan/llm/query_classifier.py` | Can be greatly simplified or deprecated |
| `src/vulcan/settings.py` | Add `use_llm_router` setting |

### 6.3 Integration Point

```python
# In query_router.py route_query() method

def route_query(self, query: str, ...) -> ProcessingPlan:
    # NEW: Use LLM router if enabled
    if self.config.use_llm_router:
        routing_decision = self.llm_router.route(query)
        
        if routing_decision.destination == "world_model":
            # Route to WorldModel + Meta-Reasoning
            return self._build_world_model_plan(query, routing_decision)
        elif routing_decision.destination == "reasoning_engine":
            # Route to specific engine
            return self._build_engine_plan(query, routing_decision.engine)
        else:
            # Skip reasoning (greetings, simple facts)
            return self._build_skip_plan(query)
    
    # EXISTING: Fall back to keyword-based routing
    # (Can be deprecated once LLM router is proven)
    ...
```

---

## 7. Conclusion

### Answer to Your Question

> Is the LLM better for routing than the current keyword/pattern matching system?

**Yes, for the routing/classification task specifically:**

1. **LLM understands semantics** - "You're designing a crypto" is about crypto, not self-introspection
2. **Handles novel phrasings** - No need to anticipate every variation
3. **Simpler maintenance** - Update prompt, not 1500 lines of patterns
4. **Robust to typos** - "whats your porpose" still routes correctly

**Important Distinction:**
- LLM does ROUTING (classification) - deciding where to send the query
- LLM does NOT do REASONING - that's what your engines and WorldModel do
- LLM does LANGUAGE OUTPUT - translating structured results to natural language

### Summary

| Component | Role | LLM Suitable? |
|-----------|------|---------------|
| **Routing/Classification** | Decide: WorldModel vs Engine vs Skip | ✅ YES - semantic understanding |
| **Reasoning** | Formal computation (logic, probability, causality) | ❌ NO - use engines |
| **Meta-Reasoning** | Ethics, introspection, values | ❌ NO - use WorldModel |
| **Language Output** | Translate results to natural language | ✅ YES - language interface |

The current keyword system is trying to do semantic classification with pattern matching - that's fundamentally the wrong tool for the job. LLM is the right tool for classification, just not for the reasoning itself.
