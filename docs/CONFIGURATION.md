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

### 2.3 Performance Tuning Configuration (PR Fixes)

These environment variables were added as part of critical performance fixes to address
query routing delays, cascade timeouts, and learning system issues.

| Var | Required | Purpose | Dev Default |
|-----|----------|---------|-------------|
| VULCAN_EMBEDDING_TIMEOUT | No | Embedding timeout for tool selector (seconds) | "5.0" |
| VULCAN_DECOMPOSITION_THRESHOLD | No | Complexity threshold for decomposition | "0.70" |
| ARENA_COMPLEXITY_THRESHOLD | No | Complexity threshold for Arena fast-path skip | "0.1" |
| VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT | No | Auto-commit self-improvements to Git | "false" |
| VULCAN_GAP_GIVEUP_THRESHOLD | No | Attempts before marking gap as deferred | "10" |
| VULCAN_LLM_HARD_TIMEOUT | No | Hard timeout for VULCAN LLM operations (seconds) | "120.0" |
| VULCAN_LLM_PER_TOKEN_TIMEOUT | No | Per-token timeout for CPU execution (seconds) | "30.0" |
| OPENAI_LANGUAGE_ONLY | No | Restrict OpenAI to language-only operations | "true" |
| OPENAI_LANGUAGE_FORMATTING | No | Route output formatting to OpenAI (fast ~2-5s) | "true" |
| OPENAI_LANGUAGE_POLISH | No | Enable OpenAI for output polishing (legacy) | "false" |

**Performance Tuning Notes:**

- **VULCAN_EMBEDDING_TIMEOUT**: Reduced from 30s to 5s to prevent cascade delays when
  decomposition calls multiple embeddings. Increase for slower CPU environments.

- **VULCAN_DECOMPOSITION_THRESHOLD**: Raised from 0.40 to 0.70 so fewer queries trigger
  the slow hierarchical decomposition path.

- **ARENA_COMPLEXITY_THRESHOLD**: Lowered from 0.30 to 0.10 so more queries go through
  Arena. Set to 0.0 to disable fast-path skip entirely.

- **VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT**: Disabled by default to prevent
  "Cannot commit: /app is not a Git repository" errors in container environments.

- **VULCAN_GAP_GIVEUP_THRESHOLD**: Increased from 3 to 10 to prevent premature give-up
  on complex learning gaps.

- **VULCAN_LLM_HARD_TIMEOUT**: Hard timeout (120s) for VULCAN LLM operations.
  This prevents indefinite hangs during CPU-intensive language generation. The internal LLM
  can take 3+ seconds per token on CPU. Note: The internal LLM is for language
  generation, not reasoning. Reasoning is done by VULCAN's reasoning systems.

- **VULCAN_LLM_PER_TOKEN_TIMEOUT**: Per-token timeout (30s) for CPU execution.
  Allows for slower token generation on CPU-bound systems.

**OpenAI Language-Only Architecture:**

VULCAN handles ALL reasoning internally using its specialized reasoning systems
(symbolic, causal, probabilistic, mathematical). OpenAI is ONLY used for language
generation - converting VULCAN's reasoning results into natural language prose.

- **OPENAI_LANGUAGE_ONLY**: When "true" (default), restricts OpenAI to language-only
  operations. Operations like embeddings, image generation (DALL-E), and audio
  transcription (Whisper) are blocked and must use local models or alternatives.
  OpenAI is NEVER permitted to perform reasoning - only language generation.

- **OPENAI_LANGUAGE_FORMATTING**: When "true" (default), routes ALL natural language
  output formatting to OpenAI (gpt-4o-mini). This provides fast response times
  (~2-5 seconds vs 60+ seconds with internal LLM on CPU). VULCAN's reasoning
  systems still do ALL thinking - OpenAI only formats the output as prose.
  Every (input, output) pair is captured for distillation training.

- **OPENAI_LANGUAGE_POLISH**: Legacy option. When "true", enables OpenAI to polish
  the internal LLM's language output. Both serve the same conceptual role -
  language generation from VULCAN's reasoning results. OPENAI_LANGUAGE_FORMATTING
  is the preferred option (replaces this).

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
