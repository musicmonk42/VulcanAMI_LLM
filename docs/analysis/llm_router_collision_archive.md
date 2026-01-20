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
| **`ToolSelector`** | `src/vulcan/reasoning/selection/tool_selector.py` | ~6959 | Tool selection orchestrator | **Downstream consumer** |
| **`_get_llm_classification()`** | `src/vulcan/reasoning/selection/tool_selector.py` | 5723-5782 | Calls `classify_query()` | **Integration point** |
| **`_generate_candidates()`** | `src/vulcan/reasoning/selection/tool_selector.py` | 5784-5876 | Uses LLM classification | **Fallback chain** |
| **`SemanticToolMatcher`** | `src/vulcan/reasoning/selection/semantic_tool_matcher.py` | N/A | Embedding-based matching | **Alternative path** |

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

## 6. Reasoning/Selection System Collision

### Overview
The `ToolSelector` in `src/vulcan/reasoning/selection/tool_selector.py` (~6959 lines) is a **downstream consumer** of `QueryClassifier`. It uses LLM-based classification to select which reasoning engine to invoke.

### Location
`src/vulcan/reasoning/selection/tool_selector.py`

### Current Integration with QueryClassifier
```python
# Lines 147-155: Import QueryClassifier
try:
    from vulcan.llm.query_classifier import classify_query, QueryClassification
    QUERY_CLASSIFIER_AVAILABLE = True
    logger.info("QueryClassifier imported for LLM-based tool selection")
except ImportError as e:
    logger.warning(f"QueryClassifier not available: {e}")
    QUERY_CLASSIFIER_AVAILABLE = False
    classify_query = None

# Lines 5723-5782: _get_llm_classification() method
def _get_llm_classification(self, query_text: str, safe_tools: List[str]) -> Optional[List[str]]:
    """Get tool candidates from LLM-based QueryClassifier."""
    if not LLM_CLASSIFICATION_ENABLED or not QUERY_CLASSIFIER_AVAILABLE:
        return None
    
    classification = classify_query(query_text)  # <-- Calls QueryClassifier
    
    if classification.confidence < LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return None  # Falls back to SemanticToolMatcher
    
    # Filter to safe tools and return candidates
    candidates = [tool for tool in classification.suggested_tools if tool in safe_tools]
    return candidates

# Lines 5784-5876: _generate_candidates() method
def _generate_candidates(self, request, features, safe_tools, prior_dist):
    """Generate tool candidates - LLM first, then fallback."""
    
    # PHASE 1: Try LLM classification first (PRIMARY PATH)
    llm_candidates = self._get_llm_classification(query_text, safe_tools)
    
    if llm_candidates:
        # Use LLM-suggested tools as primary candidates
        for tool_name in llm_candidates:
            candidates.append({"tool_name": tool_name, "source": "llm_classification"})
    
    # PHASE 2: Fallback to SemanticToolMatcher + BayesianMemoryPrior
    # ... semantic matching logic ...
```

### Collision Analysis
| Aspect | Current State | Impact of LLMQueryRouter |
|--------|--------------|--------------------------|
| Import | `from vulcan.llm.query_classifier import classify_query` | Must update to `from vulcan.routing.llm_router import get_llm_router` |
| API | `classification = classify_query(query)` | `decision = get_llm_router().route(query)` |
| Return Type | `QueryClassification.suggested_tools` | `RoutingDecision.engine` (single engine, not list) |
| Confidence | `classification.confidence` | `decision.confidence` |
| Category | `classification.category` (17 categories) | `decision.destination` (4 destinations) |

### Migration Options

#### Option A: Update ToolSelector to use LLMQueryRouter
```python
# Replace lines 5738-5782 with:
def _get_llm_classification(self, query_text: str, safe_tools: List[str]) -> Optional[List[str]]:
    """Get tool candidates from LLMQueryRouter."""
    if not settings.use_llm_router:
        return None
    
    from vulcan.routing.llm_router import get_llm_router
    router = get_llm_router()
    decision = router.route(query_text)
    
    if decision.confidence < LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return None
    
    # Map engine to tool name
    if decision.engine and decision.engine in safe_tools:
        return [decision.engine]
    
    return None
```

#### Option B: Keep Both Systems (Recommended for Transition)
- `QueryClassifier` continues working in `ToolSelector`
- `LLMQueryRouter` works at higher level in `query_router.py`
- Both use same underlying LLM but at different levels:
  - `LLMQueryRouter`: Decides **destination** (world_model vs reasoning_engine vs skip)
  - `ToolSelector`: Decides **which tool** within reasoning_engine

### Related Components in Selection Package
| File | Lines | Purpose | Collision Risk |
|------|-------|---------|----------------|
| `semantic_tool_matcher.py` | N/A | Embedding-based tool matching | Low - alternative path |
| `selection_cache.py` | N/A | Caches selection decisions | Low - reusable |
| `memory_prior.py` | N/A | Bayesian priors for tools | Low - complementary |
| `safety_governor.py` | N/A | Safety checks for tools | None - independent |

### Recommended Migration Path
1. **Phase 1 (Current)**: `LLMQueryRouter` operates at routing level only
2. **Phase 2**: Update `ToolSelector._get_llm_classification()` to optionally use `LLMQueryRouter`
3. **Phase 3**: Deprecate direct `classify_query()` usage in `ToolSelector`
4. **Phase 4**: Simplify `ToolSelector` to receive engine decision from upstream

---

## 7. Configuration Collision

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

## 8. Migration Checklist

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

## 9. Files to Archive (Not Delete)

When transitioning to `LLMQueryRouter`, these components should be archived:

| File | Archive Path | Reason |
|------|--------------|--------|
| `src/vulcan/llm/query_classifier.py` | `archive/routing/query_classifier_v1.py` | Reference for fallback |
| Pattern constants (lines 133-1200) | `archive/routing/keyword_patterns_v1.py` | Documentation |
| `QueryCategory` enum | Keep in codebase | Still useful for logging/metrics |

---

## 10. Summary

The new `LLMQueryRouter` is designed as a **drop-in replacement** for the keyword-based 
`QueryClassifier`. The collision points are:

1. **Class Level**: `QueryClassifier` → `LLMQueryRouter`
2. **Function Level**: `classify_query()` → `get_llm_router().route()`
3. **Return Type**: `QueryClassification` → `RoutingDecision`
4. **Category System**: `QueryCategory` (17 categories) → `RoutingDestination` (4 destinations + engine)
5. **Configuration**: `LLM_FIRST_CLASSIFICATION` → `USE_LLM_ROUTER`
6. **Reasoning/Selection**: `ToolSelector._get_llm_classification()` depends on `classify_query()`

The feature flag `USE_LLM_ROUTER` allows both systems to coexist during the transition period.

### Architecture Levels
```
┌──────────────────────────────────────────────────────────────┐
│  Level 1: ROUTING (LLMQueryRouter - NEW)                     │
│  Decides: world_model vs reasoning_engine vs skip            │
│  Location: src/vulcan/routing/llm_router.py                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Level 2: TOOL SELECTION (ToolSelector - EXISTING)           │
│  Decides: which specific tool within reasoning_engine        │
│  Location: src/vulcan/reasoning/selection/tool_selector.py   │
│  Currently uses: classify_query() for LLM-based selection    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Level 3: EXECUTION (Reasoning Engines)                      │
│  Engines: symbolic, probabilistic, causal, mathematical, etc │
│  Location: src/vulcan/reasoning/                             │
└──────────────────────────────────────────────────────────────┘
```

### Recommended Transition Strategy
1. `LLMQueryRouter` handles Level 1 routing decisions
2. `ToolSelector` can receive engine hint from `LLMQueryRouter` 
3. Gradually reduce `ToolSelector`'s dependency on `classify_query()`
4. Eventually, `ToolSelector` only handles tool-specific selection within the chosen engine
