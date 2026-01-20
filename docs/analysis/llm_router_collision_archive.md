# LLM Query Router Integration: Collision Analysis & Archive Reference

## Overview

This document identifies where the new `LLMQueryRouter` (in `src/vulcan/routing/llm_router.py`) 
collides with the existing routing system, and archives the relevant code sections for reference.

## Collision Points Summary

| Component | File | Lines | Purpose | Collision Type |
|-----------|------|-------|---------|----------------|
| `QueryClassifier` | `src/vulcan/llm/query_classifier.py` | ~2979 | Keyword + LLM classification | **Direct replacement** |
| `classify_query()` | `src/vulcan/llm/query_classifier.py` | 2925+ | Global function | **API collision** |
| `QueryCategory` enum | `src/vulcan/llm/query_classifier.py` | 42-60 | Category definitions | **Destination mapping** |
| `route_query()` integration | `src/vulcan/routing/query_router.py` | 4134-4200 | Uses QueryClassifier | **Integration point** |
| Keyword patterns | `src/vulcan/llm/query_classifier.py` | 133-1200 | 50+ pattern definitions | **Replaced by LLM** |

---

## 1. Direct Replacement: QueryClassifier Class

### Location
`src/vulcan/llm/query_classifier.py` lines 1303-2979

### Current Functionality
```python
class QueryClassifier:
    """
    Hybrid query classifier using keywords + LLM.
    
    This classifier solves the fundamental problem where "hello" and complex
    SAT problems both got complexity=0.50 because the old system didn't
    understand query meaning.
    """
    
    def __init__(self, llm_client=None, cache_ttl=3600, max_cache_size=10000):
        # Auto-initialize LLM client if LLM_FIRST_CLASSIFICATION enabled
        ...
    
    def classify(self, query: str) -> QueryClassification:
        # Order: Cache → Security → Greetings → Keywords/LLM
        ...
```

### New LLM Router Equivalent
```python
class LLMQueryRouter:
    """
    LLM-based query router. Replaces keyword pattern matching.
    LLM is used ONLY for classification, NOT for reasoning or answering.
    """
    
    def __init__(self, llm_client=None, cache_size=5000, cache_ttl=3600.0, timeout=3.0):
        ...
    
    def route(self, query: str) -> RoutingDecision:
        # Order: Cache → Security Guard → Crypto Guard → LLM → Fallback
        ...
```

### Migration Strategy
- `QueryClassifier.classify()` → `LLMQueryRouter.route()`
- Return type changes: `QueryClassification` → `RoutingDecision`

---

## 2. API Collision: Global classify_query() Function

### Location
`src/vulcan/llm/query_classifier.py` lines 2925-2979

### Current API
```python
def classify_query(query: str) -> QueryClassification:
    """
    Classify a query using the singleton QueryClassifier.
    
    This is the primary entry point for query classification.
    """
    classifier = get_query_classifier()
    return classifier.classify(query)
```

### Used By
`src/vulcan/routing/query_router.py` line 4137:
```python
from vulcan.llm.query_classifier import classify_query, QueryCategory

classification = classify_query(query)
```

### New LLM Router Equivalent
```python
from vulcan.routing.llm_router import get_llm_router

router = get_llm_router()
decision = router.route(query)
```

---

## 3. Destination Mapping: QueryCategory → RoutingDecision

### Current Categories (QueryCategory enum)
```python
class QueryCategory(Enum):
    GREETING = "GREETING"
    CHITCHAT = "CHITCHAT"
    FACTUAL = "FACTUAL"
    CREATIVE = "CREATIVE"
    CONVERSATIONAL = "CONVERSATIONAL"
    SELF_INTROSPECTION = "SELF_INTROSPECTION"
    SPECULATION = "SPECULATION"
    MATHEMATICAL = "MATHEMATICAL"
    LOGICAL = "LOGICAL"
    PROBABILISTIC = "PROBABILISTIC"
    CAUSAL = "CAUSAL"
    ANALOGICAL = "ANALOGICAL"
    PHILOSOPHICAL = "PHILOSOPHICAL"
    CRYPTOGRAPHIC = "CRYPTOGRAPHIC"
    LANGUAGE = "LANGUAGE"
    COMPLEX_RESEARCH = "COMPLEX_RESEARCH"
    UNKNOWN = "UNKNOWN"
```

### New Routing Destinations
```python
class RoutingDestination(Enum):
    WORLD_MODEL = "world_model"      # SELF_INTROSPECTION, PHILOSOPHICAL, ethical
    REASONING_ENGINE = "reasoning_engine"  # LOGICAL, PROBABILISTIC, CAUSAL, etc.
    SKIP = "skip"                    # GREETING, CHITCHAT, simple FACTUAL
    BLOCKED = "blocked"              # Security violations
```

### Mapping Table
| QueryCategory | RoutingDecision.destination | RoutingDecision.engine |
|---------------|----------------------------|------------------------|
| GREETING | `skip` | None |
| CHITCHAT | `skip` | None |
| FACTUAL | `skip` or `world_model` | None |
| SELF_INTROSPECTION | `world_model` | None |
| PHILOSOPHICAL | `world_model` | None |
| LOGICAL | `reasoning_engine` | `symbolic` |
| MATHEMATICAL | `reasoning_engine` | `mathematical` |
| PROBABILISTIC | `reasoning_engine` | `probabilistic` |
| CAUSAL | `reasoning_engine` | `causal` |
| ANALOGICAL | `reasoning_engine` | `analogical` |
| CRYPTOGRAPHIC | `reasoning_engine` | `cryptographic` |
| CREATIVE | `world_model` | None |
| UNKNOWN | `world_model` | None (default safe) |

---

## 4. Integration Point: query_router.py

### Location
`src/vulcan/routing/query_router.py` lines 4134-4535

### Current Integration
```python
def route_query(self, query: str, ...) -> ProcessingPlan:
    # ...preprocessing...
    
    try:
        from vulcan.llm.query_classifier import classify_query, QueryCategory
        
        classification = classify_query(query)
        
        # Log the classification result
        logger.info(
            f"[QueryRouter] {query_id}: LLM Classification: "
            f"category={classification.category}, complexity={classification.complexity:.2f}, "
            f"skip_reasoning={classification.skip_reasoning}, tools={classification.suggested_tools}"
        )
        
        # ... follow-up handling, self-introspection detection, etc. ...
```

### Proposed Integration with LLMQueryRouter
```python
def route_query(self, query: str, ...) -> ProcessingPlan:
    # ...preprocessing...
    
    # NEW: Use LLM router if enabled
    if self.config.use_llm_router:
        from vulcan.routing.llm_router import get_llm_router
        
        router = get_llm_router()
        routing_decision = router.route(query)
        
        if routing_decision.destination == "world_model":
            return self._build_world_model_plan(query, routing_decision)
        elif routing_decision.destination == "reasoning_engine":
            return self._build_engine_plan(query, routing_decision.engine)
        elif routing_decision.destination == "skip":
            return self._build_skip_plan(query)
        else:
            # blocked
            return self._build_blocked_plan(query)
    
    # EXISTING: Fall back to keyword-based routing (deprecated)
    try:
        from vulcan.llm.query_classifier import classify_query, QueryCategory
        classification = classify_query(query)
        # ... existing logic ...
```

---

## 5. Keyword Patterns Being Replaced

### Location
`src/vulcan/llm/query_classifier.py` lines 133-1200

### Pattern Summary (~1200 lines of patterns)
| Pattern Group | Line Range | Count | Purpose |
|---------------|------------|-------|---------|
| `SECURITY_VIOLATION_KEYWORDS` | 133-141 | 10+ | Security blocking |
| `SECURITY_VIOLATION_PATTERNS` | 143-155 | 7 | Security regex |
| `GREETING_PATTERNS` | 252-260 | 24 | Greeting detection |
| `CHITCHAT_PATTERNS` | 262-274 | 6 | Chitchat regex |
| `LOGICAL_KEYWORDS` | 276-375 | 100+ | Logic detection |
| `PROBABILISTIC_KEYWORDS` | 377-389 | 15+ | Probability detection |
| `CAUSAL_KEYWORDS` | 391-420 | 30+ | Causal inference |
| `MATHEMATICAL_KEYWORDS` | 448-472 | 25+ | Math detection |
| `ANALOGICAL_KEYWORDS` | 485-528 | 44+ | Analogy detection |
| `CRYPTOGRAPHIC_KEYWORDS` | 541-567 | 27+ | Crypto detection |
| `PHILOSOPHICAL_KEYWORDS` | 714-760 | 47+ | Philosophy detection |
| `SELF_INTROSPECTION_PATTERNS` | 950-1051 | 102 | Self-referential |
| `SPECULATION_PATTERNS` | 1141-1167 | 27 | Speculation |

### What LLMQueryRouter Replaces
- **ALL** keyword-based classification (~1200 lines → LLM prompt)
- **Keeps**: Security guards (deterministic, ~30 lines)
- **Keeps**: Crypto computation guards (deterministic, ~30 lines)
- **Adds**: Minimal fallback (~50 lines for emergency)

---

## 6. Configuration Collision

### Existing Settings
`src/vulcan/settings.py`:
```python
# LLM-first classification
llm_first_classification: bool = Field(default=True, env="LLM_FIRST_CLASSIFICATION")
classification_llm_timeout: float = Field(default=3.0, env="CLASSIFICATION_LLM_TIMEOUT")
classification_llm_model: str = Field(default="gpt-4o-mini", env="CLASSIFICATION_LLM_MODEL")
```

### New LLM Router Settings
`src/vulcan/settings.py` (added):
```python
# LLM Query Router (replaces keyword pattern matching)
use_llm_router: bool = Field(default=False, env="USE_LLM_ROUTER")
llm_router_timeout: float = Field(default=3.0, env="LLM_ROUTER_TIMEOUT")
llm_router_cache_size: int = Field(default=5000, env="LLM_ROUTER_CACHE_SIZE")
llm_router_cache_ttl: float = Field(default=3600.0, env="LLM_ROUTER_CACHE_TTL")
llm_router_include_examples: bool = Field(default=False, env="LLM_ROUTER_INCLUDE_EXAMPLES")
```

### Resolution
- `USE_LLM_ROUTER=true` → Use new `LLMQueryRouter`
- `USE_LLM_ROUTER=false` (default) → Use existing `QueryClassifier`
- Both can coexist during transition period
- Eventually, `LLM_FIRST_CLASSIFICATION` can be deprecated

---

## 7. Migration Checklist

### Phase 1: Parallel Operation (Current)
- [x] `LLMQueryRouter` created in `src/vulcan/routing/llm_router.py`
- [x] Feature flag `USE_LLM_ROUTER` added to settings
- [x] Tests passing (39 tests)
- [ ] Integration point in `query_router.py` (pending)

### Phase 2: A/B Testing
- [ ] Add metrics to compare accuracy
- [ ] Run both routers in shadow mode
- [ ] Compare routing decisions
- [ ] Measure latency differences

### Phase 3: Gradual Rollout
- [ ] Enable `USE_LLM_ROUTER=true` for subset of queries
- [ ] Monitor for misrouting
- [ ] Expand coverage based on accuracy

### Phase 4: Deprecation
- [ ] Mark `QueryClassifier` as deprecated
- [ ] Update documentation
- [ ] Remove keyword patterns
- [ ] Simplify codebase (~1200 lines removed)

---

## 8. Files to Archive (Not Delete)

When transitioning to `LLMQueryRouter`, these components should be archived:

| File | Archive Path | Reason |
|------|--------------|--------|
| `src/vulcan/llm/query_classifier.py` | `archive/routing/query_classifier_v1.py` | Reference for fallback |
| Pattern constants (lines 133-1200) | `archive/routing/keyword_patterns_v1.py` | Documentation |
| `QueryCategory` enum | Keep in codebase | Still useful for logging/metrics |

---

## 9. Summary

The new `LLMQueryRouter` is designed as a **drop-in replacement** for the keyword-based 
`QueryClassifier`. The collision points are:

1. **Class Level**: `QueryClassifier` → `LLMQueryRouter`
2. **Function Level**: `classify_query()` → `get_llm_router().route()`
3. **Return Type**: `QueryClassification` → `RoutingDecision`
4. **Category System**: `QueryCategory` (17 categories) → `RoutingDestination` (4 destinations + engine)
5. **Configuration**: `LLM_FIRST_CLASSIFICATION` → `USE_LLM_ROUTER`

The feature flag `USE_LLM_ROUTER` allows both systems to coexist during the transition period.
