# src/vulcan/main.py Refactoring Migration Plan

## Executive Summary

The `src/vulcan/main.py` file contains **10,048 lines** of code encompassing the entire VULCAN-AGI FastAPI application, including API endpoints, LLM integration, knowledge distillation, configuration, metrics, rate limiting, testing, and CLI functionality.

This document provides a comprehensive analysis for refactoring this monolithic file into smaller, modular components while maintaining backward compatibility.

---

## 1. External Import Analysis

### 1.1 Modules That Import FROM main.py

| File | Import Pattern | Symbol(s) Imported |
|------|---------------|-------------------|
| `src/full_platform.py` | `importlib.import_module("src.vulcan.main")` | `app` (FastAPI instance) |

**Key Finding:** The only external dependency is on the `app` FastAPI instance, which is imported dynamically via `importlib` to mount the VULCAN API as a sub-application at `/vulcan`.

### 1.2 Import Method Analysis

The import in `src/full_platform.py` (line 1251):
```python
vulcan_module = importlib.import_module("src.vulcan.main")
if not hasattr(vulcan_module, "app") or not isinstance(vulcan_module.app, FastAPI):
    raise RuntimeError("src.vulcan.main does not expose a FastAPI 'app'")
app.mount("/vulcan", vulcan_module.app)
```

**Critical Export:** Only `app` needs to be re-exported from the refactored module structure.

---

## 2. Global Variables, Classes, and Functions in main.py

### 2.1 Classes (24 total)

| Line | Class Name | Purpose | Suggested Module |
|------|-----------|---------|------------------|
| 142 | `ProcessLock` | File-based process lock for split-brain prevention | `utils/process_lock.py` |
| 401 | `MockGraphixVulcanLLM` | Mock LLM implementation for safe execution | `llm/mock_llm.py` |
| 441 | `HybridLLMExecutor` | Hybrid OpenAI + local LLM execution | `llm/hybrid_executor.py` |
| 894 | `DistillationExample` | Training example dataclass | `distillation/models.py` |
| 925 | `PIIRedactor` | PII and secrets redaction | `distillation/pii_redactor.py` |
| 1027 | `GovernanceSensitivityChecker` | Content sensitivity checking | `distillation/governance_checker.py` |
| 1127 | `ExampleQualityValidator` | Training example validation | `distillation/quality_validator.py` |
| 1283 | `DistillationStorageBackend` | JSONL storage with encryption | `distillation/storage.py` |
| 1547 | `PromotionGate` | Explicit promotion gate for trained weights | `distillation/promotion_gate.py` |
| 1726 | `ShadowModelEvaluator` | Model evaluation before promotion | `distillation/evaluator.py` |
| 1857 | `OpenAIKnowledgeDistiller` | Main knowledge distillation orchestrator | `distillation/distiller.py` |
| 2361 | `Settings` | Pydantic settings model | `settings.py` |
| 3963 | `StepRequest` | API request model | `api/models.py` |
| 3981 | `PlanRequest` | API request model | `api/models.py` |
| 3987 | `MemorySearchRequest` | API request model | `api/models.py` |
| 3993 | `ErrorReportRequest` | API request model | `api/models.py` |
| 4000 | `ApprovalRequest` | API request model | `api/models.py` |
| 4006 | `ChatRequest` | API request model | `api/models.py` |
| 4011 | `ReasonRequest` | API request model | `api/models.py` |
| 4016 | `ExplainRequest` | API request model | `api/models.py` |
| 5914 | `UnifiedChatRequest` | API request model | `api/models.py` |
| 7192 | `LLMConfigUpdate` | API request model | `api/models.py` |
| 7345 | `DistillationConfigUpdate` | API request model | `api/models.py` |
| 8860 | `IntegrationTestSuite` | Integration testing | `testing/integration.py` |
| 9155 | `PerformanceBenchmark` | Performance benchmarking | `testing/benchmark.py` |

### 2.2 Key Functions

| Line | Function | Purpose | Suggested Module |
|------|----------|---------|------------------|
| 339 | `timed_async` | Async timing decorator | `utils/timing.py` |
| 359 | `timed_sync` | Sync timing decorator | `utils/timing.py` |
| 379 | `run_tasks_in_parallel` | Parallel coroutine execution | `utils/timing.py` |
| 2337 | `get_knowledge_distiller` | Get global distiller instance | `distillation/__init__.py` |
| 2342 | `initialize_knowledge_distiller` | Initialize global distiller | `distillation/__init__.py` |
| 2553 | `initialize_component` | Component initialization tracking | `utils/components.py` |
| 2573 | `_sanitize_payload` | JSON payload sanitization | `utils/sanitize.py` |
| 2620 | `_deep_sanitize_for_json` | Deep JSON sanitization | `utils/sanitize.py` |
| 2687 | `_select_arena_agent` | Arena agent selection | `arena/client.py` |
| 2742 | `_build_arena_payload` | Arena payload construction | `arena/client.py` |
| 2789 | `_execute_via_arena` | Arena API execution | `arena/client.py` |
| 2945 | `_submit_arena_feedback` | Arena feedback submission | `arena/client.py` |
| 3051 | `get_http_session` | HTTP session management | `utils/http.py` |
| 3090 | `close_http_session` | HTTP session cleanup | `utils/http.py` |
| 3105 | `lifespan` | FastAPI lifespan manager | `api/lifespan.py` |
| 3740 | `_get_or_create_metric` | Prometheus metric factory | `metrics/prometheus.py` |
| 3802 | `cleanup_rate_limits` | Rate limit cleanup | `api/middleware.py` |
| 3829 | `validate_api_key` | API key validation middleware | `api/middleware.py` |
| 3878 | `rate_limiting` | Rate limiting middleware | `api/middleware.py` |
| 3930 | `security_headers` | Security headers middleware | `api/middleware.py` |
| 5943 | `_truncate_history` | Chat history truncation | `chat/utils.py` |
| 9700 | `find_available_port` | Port availability check | `utils/network.py` |
| 9732 | `run_production_server` | Production server runner | `cli/server.py` |
| 9755 | `main` | CLI entry point | `cli/__init__.py` |

### 2.3 API Endpoints (Route Handlers)

| Line | Endpoint | Method | Path |
|------|----------|--------|------|
| 3710 | `root` | GET | `/` |
| 4027 | `execute_step` | POST | `/step` |
| 4074 | `stream_execution` | GET | `/stream` |
| 4145 | `create_plan` | POST | `/plan` |
| 4191 | `search_memory` | POST | `/memory/search` |
| 4245 | `start_self_improvement` | POST | `/improve/start` |
| 4294 | `stop_self_improvement` | POST | `/improve/stop` |
| 4335 | `get_improvement_status` | GET | `/improve/status` |
| 4367 | `report_error` | POST | `/error` |
| 4405 | `approve_improvement` | POST | `/improve/approve` |
| 4460 | `get_pending_approvals` | GET | `/improve/pending` |
| 4494 | `update_performance_metric` | POST | `/metrics/{metric}` |
| 4530 | `chat` | POST | `/chat` |
| 5870 | `reason` | POST | `/reason` |
| 5890 | `explain` | POST | `/explain` |
| 6022 | `unified_chat` | POST | `/v2/chat` |
| 7096 | `metrics` | GET | `/metrics` |
| 7105 | `health_check` | GET | `/health` |
| 7158 | `get_llm_config` | GET | `/config/llm` |
| 7207 | `update_llm_config` | POST | `/config/llm` |
| 7257 | `get_distillation_status` | GET | `/distillation/status` |
| 7290 | `trigger_distillation_flush` | POST | `/distillation/flush` |
| 7323 | `clear_distillation_buffer` | POST | `/distillation/clear` |
| 7363 | `update_distillation_config` | POST | `/distillation/config` |
| 7409 | `system_status` | GET | `/system/status` |
| 7464 | `cognitive_status` | GET | `/status/cognitive` |
| 7591 | `llm_status` | GET | `/status/llm` |
| 7631 | `routing_status` | GET | `/status/routing` |
| 7760 | `save_checkpoint` | POST | `/checkpoint/save` |
| 7793 | `get_agents_status` | GET | `/agents/status` |
| 7834 | `spawn_agent` | POST | `/agents/spawn` |
| 7872 | `submit_agent_job` | POST | `/agents/job` |
| 8013 | `flush_memory` | POST | `/memory/flush` |
| 8066 | `get_world_model_status` | GET | `/world-model/status` |
| 8103 | `world_model_intervene` | POST | `/world-model/intervene` |
| 8138 | `world_model_predict` | POST | `/world-model/predict` |
| 8176 | `get_safety_status` | GET | `/safety/status` |
| 8211 | `validate_safety_action` | POST | `/safety/validate` |
| 8244 | `get_recent_audit_logs` | GET | `/audit/logs` |
| 8287 | `get_improvement_objectives` | GET | `/improve/objectives` |
| 8346 | `transparency_query` | POST | `/transparency/query` |
| 8390 | `get_memory_status` | GET | `/memory/status` |
| 8424 | `search_memory_unversioned` | POST | `/memory/search` |
| 8458 | `store_memory` | POST | `/memory/store` |
| 8496 | `get_hardware_status` | GET | `/hardware/status` |

### 2.4 Global Variables

| Line | Variable | Type | Purpose |
|------|----------|------|---------|
| 290 | `_process_lock` | `Optional[ProcessLock]` | Global process lock instance |
| 297 | `_openai_client` | OpenAI client | Lazy-loaded OpenAI client |
| 2334 | `_knowledge_distiller` | `Optional[OpenAIKnowledgeDistiller]` | Global distiller instance |
| 2495 | `settings` | `Settings` | Global settings instance |
| 2500 | `logger` | `Logger` | Module logger |
| 2507 | `redis_client` | `Optional[Redis]` | Redis client instance |
| 2550 | `_initialized_components` | `Dict` | Component tracking dictionary |
| 3038-3043 | HTTP pool config | `int/float` | HTTP connection pool settings |
| 3698 | `app` | `FastAPI` | **CRITICAL: Main FastAPI application** |
| 3754-3792 | Prometheus metrics | Various | Metric instances |
| 3797-3799 | Rate limit state | Dict/Lock/Thread | Rate limiting state |
| 5930-5940 | Chat constants | `int` | Chat history limits |

---

## 3. Proposed File Structure

```
src/vulcan/
├── __init__.py                  # Existing (no changes needed)
├── main.py                      # REFACTORED: Slim entry point with re-exports
│
├── api/                         # NEW: API layer
│   ├── __init__.py
│   ├── app.py                   # FastAPI app creation & configuration
│   ├── lifespan.py              # Lifespan manager
│   ├── middleware.py            # Rate limiting, API key validation, security headers
│   ├── models.py                # Pydantic request/response models
│   └── routers/                 # API routers by domain
│       ├── __init__.py
│       ├── core.py              # /, /health, /metrics, /system
│       ├── chat.py              # /chat, /v2/chat, /reason, /explain
│       ├── memory.py            # /memory/*
│       ├── improvement.py       # /improve/*
│       ├── agents.py            # /agents/*
│       ├── world_model.py       # /world-model/*
│       ├── safety_endpoints.py  # /safety/* (separate from existing safety module)
│       ├── distillation.py      # /distillation/*
│       ├── config_endpoints.py  # /config/*
│       └── transparency.py      # /transparency/*, /audit/*
│
├── llm/                         # NEW: LLM integration
│   ├── __init__.py
│   ├── mock_llm.py              # MockGraphixVulcanLLM
│   ├── hybrid_executor.py       # HybridLLMExecutor
│   └── openai_client.py         # OpenAI client management
│
├── distillation/                # NEW: Knowledge distillation
│   ├── __init__.py              # get_knowledge_distiller, initialize_knowledge_distiller
│   ├── models.py                # DistillationExample
│   ├── pii_redactor.py          # PIIRedactor
│   ├── governance_checker.py    # GovernanceSensitivityChecker
│   ├── quality_validator.py     # ExampleQualityValidator
│   ├── storage.py               # DistillationStorageBackend
│   ├── promotion_gate.py        # PromotionGate
│   ├── evaluator.py             # ShadowModelEvaluator
│   └── distiller.py             # OpenAIKnowledgeDistiller
│
├── arena/                       # NEW: Arena API integration
│   ├── __init__.py
│   └── client.py                # Arena client functions
│
├── metrics/                     # NEW: Observability
│   ├── __init__.py
│   └── prometheus.py            # Prometheus metric definitions
│
├── testing/                     # NEW: Test utilities
│   ├── __init__.py
│   ├── integration.py           # IntegrationTestSuite
│   └── benchmark.py             # PerformanceBenchmark
│
├── cli/                         # NEW: CLI layer
│   ├── __init__.py              # main() function
│   └── server.py                # run_production_server (imports find_available_port from utils_main)
│
├── utils_main/                  # NEW: Utilities (named to avoid collision)
│   ├── __init__.py
│   ├── process_lock.py          # ProcessLock
│   ├── timing.py                # timed_async, timed_sync, run_tasks_in_parallel
│   ├── sanitize.py              # _sanitize_payload, _deep_sanitize_for_json
│   ├── http.py                  # HTTP session management
│   ├── network.py               # find_available_port
│   └── components.py            # Component initialization tracking
│
└── settings.py                  # Settings class
```

---

## 4. Backward-Compatible Re-Export Strategy

### 4.1 Refactored main.py (Entry Point)

```python
"""
VULCAN-AGI Main Entry Point
Backward-compatible re-export module for refactored components.
"""

# Re-export the FastAPI application (CRITICAL for external imports)
from vulcan.api.app import app

# Re-export settings for any direct imports
from vulcan.settings import settings, Settings

# Re-export CLI entry point
from vulcan.cli import main

# Re-export utility functions that may be used externally
from vulcan.utils_main.timing import timed_async, timed_sync, run_tasks_in_parallel
from vulcan.utils_main.process_lock import ProcessLock

# Re-export distillation for any direct imports
from vulcan.distillation import (
    get_knowledge_distiller,
    initialize_knowledge_distiller,
    OpenAIKnowledgeDistiller,
)

# Re-export LLM components
from vulcan.llm import HybridLLMExecutor, MockGraphixVulcanLLM

# CLI entry point
if __name__ == "__main__":
    main()
```

### 4.2 Impact on External Imports

| Current Import | Status | Action Required |
|---------------|--------|-----------------|
| `importlib.import_module("src.vulcan.main")` | ✅ Works | None - `app` re-exported |
| `vulcan_module.app` | ✅ Works | None - `app` available at module level |

---

## 5. Blast Radius Assessment

### 5.1 Zero-Change Required (External)

| File | Import Pattern | Risk |
|------|---------------|------|
| `src/full_platform.py` | Dynamic import of `app` | **NONE** - `app` re-exported |

### 5.2 Internal Changes (src/vulcan/)

| Category | Count | Risk Level |
|----------|-------|------------|
| New files to create | ~25 | Low |
| Existing files to modify | 1 (main.py) | Medium |
| Import path changes within main.py components | Internal only | Low |
| Public API changes | **0** | None |

### 5.3 Migration Risk Matrix

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| External import breakage | **Very Low** | Re-exports maintain backward compatibility |
| Circular imports | Medium | Careful dependency ordering, lazy imports |
| Runtime errors | Low | Comprehensive test coverage before migration |
| Performance impact | Very Low | No algorithmic changes |
| Deployment issues | Low | Gradual rollout with feature flags |

---

## 6. Minimal Import Path Changes

### 6.1 Changes Required in Platform Codebase

**Zero changes required.** The re-export strategy ensures:

1. `src.vulcan.main.app` continues to work
2. `from vulcan.main import app` continues to work
3. All other current imports remain functional

### 6.2 Recommended (Optional) Import Updates

While not required, these cleaner imports can be adopted gradually:

```python
# Before (still works)
vulcan_module = importlib.import_module("src.vulcan.main")
app = vulcan_module.app

# After (recommended, cleaner)
from vulcan.api.app import app
# or
from vulcan.api import app
```

---

## 7. Implementation Phases

### Phase 1: Create Module Infrastructure (Low Risk)
1. Create new directory structure
2. Create empty `__init__.py` files
3. No changes to main.py yet

### Phase 2: Extract Utility Modules (Low Risk)
1. Extract `ProcessLock` to `utils_main/process_lock.py`
2. Extract timing decorators to `utils_main/timing.py`
3. Extract sanitization to `utils_main/sanitize.py`
4. Add imports to main.py (preserves backward compatibility)

### Phase 3: Extract Distillation System (Medium Risk)
1. Extract all distillation classes to `distillation/` module
2. Maintain global state management
3. Add re-exports to main.py

### Phase 4: Extract LLM Integration (Medium Risk)
1. Extract `MockGraphixVulcanLLM` and `HybridLLMExecutor`
2. Extract OpenAI client management
3. Add re-exports to main.py

### Phase 5: Extract API Layer (Higher Risk)
1. Create `api/app.py` with FastAPI instance
2. Extract middleware to `api/middleware.py`
3. Extract request models to `api/models.py`
4. Extract endpoints to routers
5. Wire up routers in `api/app.py`
6. Update main.py to import and re-export `app`

### Phase 6: Extract Testing & CLI (Low Risk)
1. Extract `IntegrationTestSuite` and `PerformanceBenchmark`
2. Extract CLI components
3. Update `main()` entry point

### Phase 7: Cleanup (Low Risk)
1. Remove extracted code from main.py
2. Verify all re-exports work
3. Run full test suite
4. Document new module structure

---

## 8. Testing Strategy

### 8.1 Pre-Migration Tests
- Run existing test suite to establish baseline
- Document any existing failures

### 8.2 Per-Phase Tests
- After each phase, run affected tests
- Verify imports work from external code
- Check for import cycles

### 8.3 Integration Tests
- Test `src/full_platform.py` mounting
- Test CLI modes (test, benchmark, interactive, production)
- Test API endpoints end-to-end

### 8.4 Regression Tests
- Ensure all current functionality preserved
- Performance benchmarks before/after

---

## 9. Rollback Plan

Each phase should be independently revertable:

1. **Git Strategy**: Create feature branch per phase
2. **Feature Flags**: Use `VULCAN_REFACTORED_IMPORTS=1` to toggle
3. **Dual Imports**: Keep original code in main.py commented until final cleanup
4. **Quick Rollback**: Revert to previous commit if issues detected

---

## 10. Conclusion

The refactoring of `src/vulcan/main.py` is **feasible with minimal blast radius** due to:

1. **Single external dependency**: Only `app` is imported from outside
2. **Re-export strategy**: Backward compatibility maintained through module-level re-exports
3. **Phased approach**: Incremental extraction reduces risk
4. **Clear module boundaries**: Natural separation into API, LLM, distillation, etc.

### Recommended Priority Order:
1. Distillation module (self-contained, 1000+ lines)
2. LLM module (self-contained, ~500 lines)
3. API models and middleware
4. API routers (largest, most interconnected)
5. Testing and CLI

### Estimated Effort:
- **Phase 1-2**: 2-4 hours
- **Phase 3-4**: 4-8 hours
- **Phase 5**: 8-16 hours (most complex)
- **Phase 6-7**: 2-4 hours
- **Total**: 16-32 hours

---

*Document generated: 2025-12-24*
*Source file: src/vulcan/main.py (10,048 lines)*
