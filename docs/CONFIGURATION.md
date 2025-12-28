# Configuration (Environment & Profiles)

## 1. Resolution Order
1. Explicit environment variables
2. Profile manifests (dev/testing)
3. Governance-approved overrides (future signed proposals)
4. CLI/session overrides (non-sensitive)
5. Hardcoded dev defaults

Security-critical ceilings not overridable (e.g., raising risk threshold beyond governance max).

## 2. Key Environment Variables
| Var | Required | Purpose | Dev Default |
|-----|----------|---------|-------------|
| JWT_SECRET_KEY | Yes (prod) | JWT signing | "dev-change-me" |
| BOOTSTRAP_KEY | Yes | Initial registry bootstrap | "dev-bootstrap" |
| AUDIT_DB_PATH | Yes | Audit DB file path | "./audit.db" |
| ARENA_API_KEY | Optional | Arena exec auth | "" |
| METRICS_ENABLED | No | Enable metrics endpoint | "true" |
| LOG_LEVEL | No | Logging verbosity | "INFO" |
| MAX_GRAPH_NODES | No | Node cap | 10000 |
| MAX_GRAPH_EDGES | No | Edge cap | 100000 |
| NODE_TIMEOUT_DEFAULT_MS | No | Default node timeout | 5000 |
| LAYER_TIMEOUT_FACTOR | No | Layer timeout multiplier | 2.0 |
| RISK_SCORE_THRESHOLD | No | Auto-apply risk ceiling | 0.30 |
| REPLAY_WINDOW_SECONDS | No | Replay protection window | 60 |
| ENABLE_INTRINSIC_DRIVES | No | Self-improvement toggle | "false" |
| OPENAI_API_KEY | Optional | AI provider | "" |

### 2.1 Reasoning System Configuration (Singleton Pattern)

These environment variables control the reasoning singleton system which prevents progressive
query routing degradation (469ms → 152,048ms) by caching ML model instances.

| Var | Required | Purpose | Dev Default |
|-----|----------|---------|-------------|
| REASONING_PREWARM_SINGLETONS | No | Prewarm all singletons at startup | "true" |
| MEMORY_GUARD_ENABLED | No | Enable memory pressure monitoring | "true" |
| MEMORY_GUARD_THRESHOLD_PERCENT | No | Memory % to trigger GC | "85.0" |
| MEMORY_GUARD_CHECK_INTERVAL | No | Seconds between memory checks | "5.0" |
| GC_REQUEST_INTERVAL | No | GC every N requests (rate limiting) | "10" |
| PROBLEM_DECOMPOSER_ENABLED | No | Enable hierarchical problem decomposition | "true" |
| DECOMPOSITION_COMPLEXITY_THRESHOLD | No | Complexity threshold for decomposition | "0.40" |
| SEMANTIC_BRIDGE_ENABLED | No | Enable cross-domain knowledge transfer | "true" |
| CROSS_DOMAIN_TRANSFER_ENABLED | No | Enable concept transfer between domains | "true" |
| PATTERN_LEARNING_ENABLED | No | Learn patterns from successful outcomes | "true" |

### 2.2 CuriosityDriver Configuration (Active Learning)

| Var | Required | Purpose | Dev Default |
|-----|----------|---------|-------------|
| CURIOSITY_HEARTBEAT_INTERVAL | No | Seconds between heartbeat cycles | "60.0" |
| CURIOSITY_MIN_BUDGET | No | Min budget to run learning cycle | "10.0" |
| CURIOSITY_MAX_EXPERIMENTS | No | Max experiments per cycle | "5" |
| CURIOSITY_LOW_BUDGET_SLEEP | No | Sleep when budget low (seconds) | "120.0" |
| CURIOSITY_CYCLE_TIMEOUT | No | Timeout per cycle (seconds) | "300.0" |

## 3. Profiles
development:
- tracing: verbose
- self_improvement: enabled (bounded sessions)
- metrics: full

testing:
- reduced concurrency
- shorter timeouts
- intrinsic drives disabled

## 4. Feature Toggles
| Toggle | Impact | Recommended Scope |
|--------|--------|-------------------|
| enable_self_modification | Allows graph mutation nodes | Staging/controlled |
| enable_multi_agent | Crew orchestration | Staging |
| enable_explainability | Detailed execution explanations | Dev/Partial Prod |
| enable_adversarial_testing | Inject synthetic risks | Dev only |
| enforce_strict_provenance | Hard fail missing lineage | Production |
| reasoning_prewarm_singletons | Prewarm ML models at startup | Production |
| memory_guard_enabled | Auto GC on high memory | Production |
| problem_decomposer_enabled | Hierarchical query decomposition | Production |
| semantic_bridge_enabled | Cross-domain knowledge transfer | Production |

## 5. Intrinsic Drives Config Elements
- triggers: startup, degradation, periodic, low_activity, error surge
- constraints: max_sessions/day, protected paths
- rollback_on_failure flag
- dry_run preview with risk scoring prior to apply

## 6. Dynamic Reconfiguration (Future)
Signed config proposal → validation pipeline → hot reload for non-critical; critical (JWT secret) scheduled restart window.

## 7. Secret Handling Guidance
- External secret manager (Vault/KMS)
- No logging of secret values
- Rotational cadence (JWT monthly)
- Environment parity segregation (distinct keys per environment)

## 8. Composite Example
```json
{
  "runtime": {
    "profile": "development",
    "max_graph_nodes": 6000,
    "node_timeout_ms": 7000,
    "layer_timeout_factor": 2.1
  },
  "governance": {
    "risk_threshold": 0.28,
    "replay_window_seconds": 60
  },
  "observability": {
    "metrics_enabled": true,
    "tracing": "verbose"
  },
  "intrinsic_drives": {
    "enabled": true,
    "max_sessions_per_day": 3
  }
}
```

## 9. Performance Optimization Configuration

### FAISS Vector Search
The system uses FAISS for high-performance vector similarity search with automatic CPU instruction set detection:

**Instruction Sets** (best to worst performance):
- AVX-512: Optimal performance (requires CPU support)
- AVX2: Good performance (most modern CPUs)
- AVX: Standard performance
- Scalar: Fallback mode

**Configuration**: Automatic detection via `src/utils/faiss_config.py`
- No manual configuration needed
- System logs detected instruction set on startup
- Gracefully falls back to NumPy if FAISS unavailable

**Installation** (if not present):
```bash
pip install faiss-cpu  # For CPU-only systems
pip install faiss-gpu  # For GPU acceleration
```

### LLVM Compiler Backend
The LLVM backend compiles graph IR to optimized native code with configurable optimization levels:

**Optimization Levels**:
- 0: No optimization (fastest compilation)
- 1: Basic optimization
- 2: Standard optimization (default)
- 3: Aggressive optimization

**Configuration**: Set via `LLVMBackend(optimization_level=N)`
- Execution engine creation may fail in some environments (containers, ARM)
- Compiler still functions for IR generation/analysis without execution engine
- System logs detailed diagnostics on initialization

**Note**: Execution engine unavailability does not impact:
- IR generation and optimization
- Graph compilation and analysis
- Most compiler functionality

Only JIT (Just-In-Time) execution requires the execution engine.

## 10. Troubleshooting
| Symptom | Cause | Resolution |
|---------|-------|-----------|
| Validation timeout | recursion depth or large params | Increase timeout / optimize params |
| High memory estimate | large nested params | Break graph into subgraphs |
| Missing metrics | METRICS_ENABLED false | Set true + restart |
| Replay rejection | identical hash inside window | Modify proposal or wait window expiry |
| Self-improvement not triggering | ENABLE_INTRINSIC_DRIVES false | Enable & restart |
| FAISS AVX512 warnings | CPU doesn't support AVX512 | Expected; system uses best available (AVX2) |
| LLVM execution engine fails | Platform limitations | Expected in some environments; IR generation still works |
| Slow vector search | FAISS not installed | Install faiss-cpu for 10-100x speedup |
