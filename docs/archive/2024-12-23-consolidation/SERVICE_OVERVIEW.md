# ⚠️ DEPRECATED

**This document has been consolidated.**  
**Archived:** December 23, 2024

## Migration Path

For SERVICE_OVERVIEW.md content → See [COMPLETE_SERVICE_CATALOG.md](../../COMPLETE_SERVICE_CATALOG.md)

---

# Service Overview and Architecture

**Last Updated:** December 15, 2024  
**Platform Version:** v2.1.0

This document provides a high-level overview of all services in the VulcanAMI LLM platform, organized by functional area.

> For complete function-level documentation, see [COMPLETE_SERVICE_CATALOG.md](../../COMPLETE_SERVICE_CATALOG.md)

---

## Platform Statistics

- **Total Services:** 71 distinct service modules
- **Total Python Files:** 557 files (312 production, 245 test)
- **Total Functions:** 21,523 functions (11,811 test functions = 54.9% coverage)
- **Total Classes:** 4,353 classes (2,361 test classes = 54.2% coverage)
- **Lines of Code:** 506,329 total lines (373,289 code, 35,835 comments, 97,205 blank)

---

## Core Services (Root Level)

### 1. **app.py** - Registry API Service (Flask)
- **Purpose:** Main Flask application providing Registry API
- **Functions:** JWT authentication, agent onboarding, proposal submission, audit logs
- **Key Endpoints:**
  - `/registry/bootstrap` - Initialize first admin
  - `/auth/login` - JWT authentication
  - `/registry/onboard` - Register agents
  - `/ir/propose` - Submit graph proposals
  - `/audit/logs` - Query audit trail
  - `/health`, `/metrics` - Health and monitoring

### 2. **graphix_arena.py** - Arena API Service (FastAPI)
- **Purpose:** High-performance graph execution API
- **Functions:** Graph execution, status tracking, cancellation, metrics
- **Key Endpoints:**
  - `/execute/graph` - Execute validated graphs
  - `/execution/{id}/status` - Check execution status
  - `/execution/{id}/cancel` - Cancel execution
  - `/health`, `/ready`, `/metrics` - Health and monitoring

### 3. **api_server.py** - Unified API Gateway
- **Functions:** 87 functions, 15 classes
- **Purpose:** Centralized API routing and request handling
- **Features:** Load balancing, auth aggregation, rate limiting

### 4. **agent_interface.py** - Agent Communication Interface
- **Functions:** 69 functions, 11 classes
- **Purpose:** Agent-to-agent and agent-to-system communication
- **Features:** Message routing, protocol handling, serialization

### 5. **agent_registry.py** - Agent Registry and Management
- **Functions:** 58 functions, 13 classes
- **Purpose:** Agent lifecycle management, trust scoring, capabilities
- **Features:** Agent registration, trust updates, capability tracking

---

## VULCAN-AGI Core (src/vulcan/) ⭐

**The crown jewel of the platform - 285,000+ LOC of cognitive architecture**

### Statistics
- **Files:** 256 Python files
- **Functions:** 13,304 functions
- **Classes:** 2,545 classes
- **Coverage:** 48% test coverage (excellent for AGI system)

### Key Subsystems

#### 1. **World Model** (src/vulcan/world_model/)
- **Purpose:** Causal reasoning and state prediction
- **Key Components:**
  - `world_model_core.py` - Main orchestrator (EXAMINE→SELECT→APPLY→REMEMBER)
  - `causal_graph.py` - Causal DAG for intervention planning
  - `prediction_engine.py` - Temporal forecasting
  - `intervention_manager.py` - Safe action execution
  - `confidence_calibrator.py` - Uncertainty quantification
  - `dynamics_model.py` - State transition modeling
  - `invariant_detector.py` - Cross-domain knowledge transfer

#### 2. **Meta-Reasoning** (src/vulcan/world_model/meta_reasoning/)
- **Purpose:** Self-awareness and self-improvement
- **Key Components:**
  - `motivational_introspection.py` - Goal awareness
  - `self_improvement_drive.py` - Autonomous enhancement
  - `preference_learner.py` - RLHF integration
  - `internal_critic.py` - Self-evaluation
  - `ethical_boundary_monitor.py` - Ethics enforcement
  - `csiu_enforcement.py` - CSIU framework (Curiosity, Safety, Impact, Uncertainty)

#### 3. **Reasoning Systems** (src/vulcan/reasoning/)
- **Purpose:** Multi-modal reasoning capabilities
- **Key Components:**
  - `symbolic.py` - Logic and formal verification
  - `causal.py` - Causal inference (69,502 LOC)
  - `analogical.py` - Transfer learning (90,428 LOC)
  - `multimodal.py` - Cross-modal inference (107,700 LOC)
  - `probabilistic.py` - Bayesian inference (61,099 LOC)
  - `contextual_bandits.py` - Exploration-exploitation (54,127 LOC)
  - `unified.py` - Reasoning orchestration (119,871 LOC)

#### 4. **Memory Systems** (src/vulcan/memory/)
- **Purpose:** Hierarchical memory architecture
- **Key Components:**
  - `hierarchical.py` - Multi-level storage
  - `specialized.py` - Domain-specific memory (97,779 LOC)
  - `distributed.py` - Sharding and replication (42,622 LOC)
  - `consolidation.py` - Knowledge integration (47,694 LOC)
  - `retrieval.py` - Associative recall (48,274 LOC)

#### 5. **Planning Engine** (src/vulcan/planning/)
- **Purpose:** Strategic planning and goal decomposition
- **Key Components:**
  - `planning.py` - Task decomposition
  - `main.py` - Orchestration logic
  - `processing.py` - Data processing pipelines

#### 6. **Safety Systems** (src/vulcan/safety/)
- **Purpose:** Safety validation and constraint enforcement
- **Key Components:**
  - Safety validators
  - Adversarial testing
  - Formal verification interfaces
  - Ethical boundary monitoring

---

## Graph Execution Layer

### 1. **Compiler** (src/compiler/)
- **Files:** 5 files
- **Functions:** 75 functions, 17 classes
- **Purpose:** GraphixIR compilation to optimized native code

**Key Components:**
- `graph_compiler.py` - IR to LLVM compilation (719 LOC)
- `llvm_backend.py` - LLVM integration
- `hybrid_executor.py` - Multi-backend execution
- **Optimizations:** Operation fusion, dead code elimination, CSE
- **Performance:** 10-100x speedup vs interpreted execution

### 2. **Unified Runtime** (src/unified_runtime/)
- **Files:** 12 files
- **Functions:** 307 functions, 67 classes
- **Purpose:** Orchestrate graph execution across backends

**Key Components:**
- `unified_runtime_core.py` - Main runtime orchestrator
- `execution_engine.py` - DAG scheduler
- `node_handlers.py` - Per-node-type execution
- `neural_system_optimizer.py` - Dynamic optimization
- `hardware_dispatcher_integration.py` - Hardware selection
- `vulcan_integration.py` - VULCAN bridge

### 3. **Execution** (src/execution/)
- **Files:** 4 files
- **Functions:** 202 functions, 51 classes
- **Purpose:** Low-level execution primitives

**Key Components:**
- Node execution handlers
- Error taxonomy and handling
- Timeout management
- Result aggregation

---

## LLM and Generation Layer

### 1. **LLM Core** (src/llm_core/)
- **Files:** 7 files
- **Functions:** 125 functions, 31 classes
- **Purpose:** Custom transformer with graph execution

**Key Components:**
- `graphix_transformer.py` - Custom transformer (913 LOC)
- `graphix_executor.py` - Execution engine (1,166 LOC)
- `ir_attention.py` - Attention layers
- `ir_feedforward.py` - Feed-forward layers
- `ir_embeddings.py` - Embedding layers
- `persistant_context.py` - Context management (857 LOC)

**Features:**
- LoRA fine-tuning
- Gradient checkpointing
- Top-P sampling
- IR caching

### 2. **Generation** (src/generation/)
- **Files:** 6 files
- **Functions:** 273 functions, 62 classes
- **Purpose:** Safe AI content generation

**Key Components:**
- Governed generation with safety checks
- Content filtering
- Style control
- Multi-modal generation

### 3. **Local LLM** (src/local_llm/)
- **Files:** 3 files
- **Functions:** 33 functions, 5 classes
- **Purpose:** Local LLM model integration

---

## Storage and Memory Layer

### 1. **Persistent Memory v46** (src/persistant_memory_v46/) ⭐
- **Files:** 11 files
- **Functions:** 353 functions, 44 classes
- **Purpose:** Advanced storage with unlearning

**Key Components:**
- `graph_rag.py` - Graph-based RAG (736 LOC)
- `lsm.py` - Log-Structured Merge tree (928 LOC)
- `store.py` - S3/CloudFront storage (851 LOC)
- `unlearning.py` - Machine unlearning (743 LOC) ⭐⭐⭐
- `zk.py` - Zero-knowledge proofs (1,075 LOC)

**Features:**
- GDPR-compliant unlearning
- Cryptographic verification
- S3-backed with CDN
- Petabyte-scale support

### 2. **Memory** (src/memory/)
- **Files:** 4 files
- **Functions:** 98 functions, 35 classes
- **Purpose:** Memory management utilities

---

## Governance and Security Layer

### 1. **Governance** (src/governance/)
- **Files:** 3 files
- **Functions:** 132 functions, 41 classes
- **Purpose:** Trust-weighted consensus engine

**Key Components:**
- `consensus_engine.py` - Voting logic (26 functions)
- `consensus_manager.py` - Proposal lifecycle (25 functions)
- `governance_loop.py` - Governance orchestration (31 functions)

**Features:**
- Trust-weighted voting
- Proposal lifecycle management
- Quorum thresholds
- Policy enforcement

### 2. **Security Services**
**Files across multiple locations:**
- `security_audit_engine.py` - Security scanning (21 functions)
- `security_nodes.py` - Security nodes (14 functions, 5 classes)
- `audit_log.py` - Audit trail (28 functions, 5 classes)

**Features:**
- SQLite-backed audit trail
- Integrity checks
- Threat detection
- Compliance reporting

---

## Integration and Orchestration

### 1. **Integration** (src/integration/)
- **Files:** 11 files
- **Functions:** 193 functions, 71 classes
- **Purpose:** Component integration and orchestration

**Key Components:**
- `graphix_vulcan_bridge.py` - VULCAN integration (673 LOC)
- Cognitive cycle orchestration (EXAMINE→SELECT→APPLY→REMEMBER)
- Async execution with retry logic
- Observability integration

### 2. **G-Vulcan** (src/gvulcan/)
- **Files:** 34 files
- **Functions:** 516 functions, 163 classes
- **Purpose:** Graphix-VULCAN coordination layer

---

## Training and Optimization

### 1. **Training** (src/training/)
- **Files:** 11 files
- **Functions:** 220 functions, 38 classes
- **Purpose:** Model training and optimization

**Key Components:**
- Governed training loops
- Curriculum learning
- Distributed training
- Checkpoint management

### 2. **Evolution Engine** (src/evolve/)
- **Files:** 3 files
- **Functions:** 36 functions, 3 classes
- **Purpose:** Evolutionary optimization

**Components:**
- `evolution_engine.py` - Genetic algorithms (43 functions)
- Mutation strategies
- Fitness evaluation
- Population management

### 3. **NSO Aligner** 
- **File:** `nso_aligner.py`
- **Functions:** 65 functions, 8 classes
- **Purpose:** Neural Structure Optimization and alignment

---

## Observability and Monitoring

### 1. **Observability Manager**
- **File:** `observability_manager.py`
- **Functions:** 18 functions, 1 class
- **Purpose:** Unified observability management

**Features:**
- Prometheus metrics
- Grafana dashboards
- Custom alerting
- Performance tracking

### 2. **Hardware Dispatcher**
- **File:** `hardware_dispatcher.py`
- **Functions:** 28 functions, 6 classes
- **Purpose:** Hardware selection and optimization

**Features:**
- Cost model evaluation
- Backend selection
- Resource allocation
- Performance profiling

---

## Specialized Services

### 1. **Adversarial Tester**
- **File:** `adversarial_tester.py`
- **Functions:** 54 functions, 10 classes
- **Purpose:** Adversarial testing and robustness validation

### 2. **Data Augmentor**
- **File:** `data_augmentor.py`
- **Functions:** 21 functions, 4 classes
- **Purpose:** Data augmentation for training

### 3. **Drift Detector**
- **File:** `drift_detector.py`
- **Functions:** 24 functions, 4 classes
- **Purpose:** Concept drift detection and monitoring

### 4. **Interpretability Engine**
- **File:** `interpretability_engine.py`
- **Functions:** 12 functions, 2 classes
- **Purpose:** Model interpretability and explainability

### 5. **Explainability Node**
- **File:** `explainability_node.py`
- **Functions:** 10 functions, 4 classes
- **Purpose:** Graph node explainability

### 6. **Conformal Prediction** (src/conformal/)
- **Files:** 2 files
- **Functions:** 34 functions, 8 classes
- **Purpose:** Uncertainty quantification with conformal prediction

### 7. **Pattern Matcher**
- **File:** `pattern_matcher.py`
- **Functions:** 11 functions, 8 classes
- **Purpose:** Pattern recognition and matching

### 8. **Superoptimizer**
- **File:** `superoptimizer.py`
- **Functions:** 24 functions, 4 classes
- **Purpose:** Code optimization beyond compiler

### 9. **Tournament Manager**
- **File:** `tournament_manager.py`
- **Functions:** 16 functions, 3 classes
- **Purpose:** Model tournament and selection

---

## Hardware and Emulation

### 1. **Analog Photonic Emulator**
- **File:** `analog_photonic_emulator.py`
- **Functions:** 56 functions, 11 classes
- **Purpose:** Photonic computing emulation

**Features:**
- Optical matrix-vector multiplication
- Photonic interference simulation
- Performance modeling
- Future hardware support

### 2. **Hardware Emulator**
- **File:** `hardware_emulator.py`
- **Functions:** 15 functions, 1 class
- **Purpose:** Generic hardware emulation framework

---

## Utilities and Support

### 1. **Utils** (src/utils/)
- **Files:** 4 files
- **Functions:** 30 functions, 5 classes
- **Purpose:** Utility functions and helpers

### 2. **Tools** (src/tools/)
- **Files:** 2 files
- **Functions:** 17 functions, 4 classes
- **Purpose:** Development and operational tools

### 3. **AI Providers**
- **File:** `ai_providers.py`
- **Functions:** 65 functions, 17 classes
- **Purpose:** External AI provider integrations

**Supported Providers:**
- OpenAI (GPT-4, o1)
- Anthropic (Claude)
- Google (Gemini)
- Custom providers

### 4. **LLM Client**
- **File:** `llm_client.py`
- **Functions:** 8 functions, 1 class
- **Purpose:** Unified LLM client interface

---

## Testing and Validation

### 1. **Tests** (tests/)
- **Files:** 90 files
- **Functions:** 3,577 functions
- **Classes:** 712 classes
- **Purpose:** Comprehensive test suite

**Test Categories:**
- Unit tests (60+ files)
- Integration tests (20+ files)
- Security tests
- CI/CD tests (reproducibility, Docker, K8s)
- Performance tests
- System tests

### 2. **Stress Tests** (stress_tests/)
- **Purpose:** Load and stress testing
- **Components:**
  - `load_test.py` - Load testing (31 functions, 10 classes)
  - Scalability validation
  - Resource limit testing

### 3. **Validation**
- `run_validation_test.py` - System validation (23 functions)
- End-to-end validation
- Integration verification

---

## Configuration and Specs

### 1. **Configs** (configs/)
- **Files:** 6 files
- **Functions:** 91 functions, 10 classes
- **Purpose:** Configuration management

**Components:**
- Nginx configuration
- Redis configuration
- CloudFront configuration
- OPA policies
- Vector logging
- ZK proof configs

### 2. **Specs** (specs/)
- **Files:** 3 files
- **Functions:** 95 functions, 22 classes
- **Purpose:** Specification and validation

---

## Demo and Examples

### 1. **Demo** (demo/)
- **Files:** 2 files
- **Functions:** 30 functions, 8 classes
- **Purpose:** Demonstration scripts and examples

### 2. **Examples** (examples/)
- **Purpose:** Usage examples and tutorials

---

## Client SDKs

### 1. **Client SDK** (client_sdk/)
- **Purpose:** Client libraries for platform integration
- **Languages:** Python, JavaScript (future)

---

## Root-Level Utilities

### 1. **graphix_vulcan_llm.py**
- **Functions:** Large legacy integration file
- **Purpose:** Original integration layer (being deprecated)

### 2. **Evaluation Scripts**
- `eval_state_dict_gpt.py` - Model evaluation
- `simple_eval.py` - Simple evaluation
- `simple_eval_pkl.py` - Pickle-based evaluation

### 3. **Inspection Tools**
- `inspect_pt.py` - PyTorch model inspection
- `inspect_system_state.py` - System state inspection

### 4. **Generation Tools**
- `generate_transparency_report.py` - Transparency reporting (44 functions)
- `large_graph_generator.py` - Test graph generation

### 5. **Minimal Executor**
- `minimal_executor.py` - Minimal execution demo (16 functions)

### 6. **Setup and Bootstrap**
- `setup_agent.py` - Agent setup (4 functions)
- `setup.py` - Package setup

---

## Service Communication Patterns

### Request Flow
```
User → API Gateway → Authentication → Service Router → Core Service
                                                      ↓
                                         VULCAN-AGI (reasoning)
                                                      ↓
                                         Graph Compiler (optimization)
                                                      ↓
                                         Unified Runtime (execution)
                                                      ↓
                                         Persistent Memory (storage)
                                                      ↓
                                         Response → User
```

### Cognitive Cycle (VULCAN Integration)
```
EXAMINE → SELECT → APPLY → REMEMBER
   ↓         ↓       ↓         ↓
  Memory  Reasoning Execution Storage
```

---

## Service Dependencies

### Core Dependencies
- Python 3.10.11
- PyTorch (deep learning)
- Flask (Registry API)
- FastAPI (Arena API)
- Redis (caching, rate limiting)
- SQLite/PostgreSQL (storage)
- Prometheus (metrics)
- Grafana (dashboards)

### Optional Dependencies
- CUDA (GPU acceleration)
- ChromaDB (vector storage)
- Qdrant (vector search)
- MinIO (S3-compatible storage)
- CloudFront (CDN)

---

## Service Startup Order

1. **Infrastructure Services**
   - Redis
   - Database (SQLite/PostgreSQL)
   - MinIO (if using S3 storage)

2. **Core Services**
   - Audit Log Service
   - Agent Registry
   - Observability Manager

3. **API Services**
   - Registry API (Flask) - Port 5000
   - Arena API (FastAPI) - Port 8000
   - API Gateway (if used)

4. **Execution Services**
   - VULCAN-AGI Core
   - Graph Compiler
   - Unified Runtime
   - LLM Core

5. **Supporting Services**
   - Governance Loop
   - Security Audit Engine
   - Hardware Dispatcher

---

## Health Checks

All services expose health check endpoints:

- **Registry API:** `GET /health`
- **Arena API:** `GET /health`, `GET /ready`
- **VULCAN Core:** Internal health checks
- **Persistent Memory:** Connection validation
- **Graph Compiler:** Compilation readiness

---

## Monitoring and Metrics

### Key Metrics by Service

**Registry API:**
- `graphix_registry_agents_total` - Total agents
- `graphix_registry_proposals_total` - Total proposals
- `graphix_registry_auth_requests` - Auth requests

**Arena API:**
- `arena_executions_total` - Total executions
- `arena_execution_duration_seconds` - Execution duration
- `arena_active_executions` - Active executions

**VULCAN-AGI:**
- `vulcan_reasoning_cycles` - Reasoning cycles
- `vulcan_world_model_updates` - State updates
- `vulcan_confidence_scores` - Confidence distribution

**Persistent Memory:**
- `memory_operations_total` - Total operations
- `memory_cache_hit_rate` - Cache hit rate
- `memory_unlearning_requests` - Unlearning operations

---

## API Endpoints Summary

### Registry API (Flask - Port 5000)
- `POST /registry/bootstrap` - Bootstrap first agent
- `POST /auth/login` - Authentication
- `POST /registry/onboard` - Onboard agent
- `POST /ir/propose` - Submit proposal
- `GET /audit/logs` - Query audits
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Arena API (FastAPI - Port 8000)
- `POST /execute/graph` - Execute graph
- `GET /execution/{id}/status` - Check status
- `POST /execution/{id}/cancel` - Cancel execution
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

---

## Security Services

### Authentication
- JWT (Registry API)
- API Keys (Arena API)
- Bootstrap Keys (Initial setup)

### Authorization
- Role-based access control (RBAC)
- Trust-level based permissions
- Scope-based access

### Audit Trail
- SQLite-backed persistence
- WAL mode for integrity
- Selective alerting to Slack
- Compliance reporting

### Security Scanning
- Bandit static analysis
- Dependency scanning (4,007 hashes)
- CodeQL vulnerability detection
- Container security (non-root users)

---

## Deployment Configurations

### Development
- Single machine
- SQLite storage
- In-memory caching
- No external dependencies

### Staging
- 2-3 machines
- S3 storage
- Redis caching
- Basic monitoring

### Production
- 20-50 machines
- S3 + CloudFront
- Redis cluster
- Full observability
- Auto-scaling (Kubernetes)

---

## Component Initialization Status

### Overview

This section documents the initialization status of all documented platform components during startup. Components are categorized as either **eager-loaded** (initialized at startup) or **lazy-loaded** (initialized on first use).

### Eager-Loaded Components

These components are initialized during platform startup in `src/full_platform.py`:

#### 1. **Graph Compiler** (`src/compiler/graph_compiler.py`)
- **Status:** ✅ Initialized at startup
- **Initialization:** Creates `GraphCompiler(optimization_level=2)`
- **Verification:** Checks LLVM backend availability
- **Log Message:** `✓ GraphCompiler initialized (optimization_level=2, LLVM=available/unavailable)`
- **Dependencies:** llvmlite, networkx
- **Purpose:** Compiles Graphix IR to optimized native code

#### 2. **Persistent Memory v46** (`src/persistant_memory_v46/`)
- **Status:** ✅ Verified at startup
- **Initialization:** Imports and verifies all subsystems
- **Subsystems:**
  - LSM tree with Merkle DAG
  - Graph RAG with hybrid retrieval
  - Unlearning module (gradient surgery, SISA, influence)
  - ZK proofs (Groth16, PLONK)
  - S3/CloudFront storage backend
- **Log Message:** `✓ Persistent Memory v46.0.0 initialized`
- **Purpose:** Advanced persistent storage with privacy-preserving unlearning

#### 3. **Conformal Prediction** (`src/conformal/confidence_calibration.py`)
- **Status:** ✅ Initialized at startup
- **Initialization:** Creates `ConformalPredictor(alpha=0.1)`
- **Log Message:** `✓ ConformalPredictor initialized (alpha=0.1, coverage_guarantee=90.0%)`
- **Purpose:** Uncertainty quantification with coverage guarantees

#### 4. **Drift Detector** (`src/drift_detector.py`)
- **Status:** ✅ Initialized at startup
- **Initialization:** Creates `DriftDetector(window_size=1000, threshold=0.05, embedding_dim=768)`
- **Log Message:** `✓ DriftDetector initialized (window_size=1000, threshold=0.05)`
- **Dependencies:** faiss, numpy
- **Purpose:** Concept drift detection and automatic realignment

#### 5. **Pattern Matcher** (`src/pattern_matcher.py`)
- **Status:** ✅ Initialized at startup
- **Initialization:** Creates `PatternMatcher()`
- **Log Message:** `✓ PatternMatcher initialized`
- **Dependencies:** networkx
- **Purpose:** Subgraph pattern matching with ethical validation

#### 6. **Superoptimizer** (`src/superoptimizer.py`)
- **Status:** ✅ Initialized at startup
- **Initialization:** Creates `Superoptimizer()`
- **Log Message:** `✓ Superoptimizer initialized (cache_size=0)`
- **Dependencies:** LLM client (optional)
- **Purpose:** Hardware-specific kernel optimization

#### 7. **Tournament Manager** (`src/tournament_manager.py`)
- **Status:** ✅ Initialized at startup (in Arena)
- **Initialization:** Creates `TournamentManager(diversity_penalty=0.3, winner_percentage=0.2)`
- **Log Message:** `✓ TournamentManager initialized (diversity_penalty=0.3, winner_percentage=0.2)`
- **Connected to:** Evolution Engine (available on demand)
- **Purpose:** Adaptive tournament selection with diversity penalties

#### 8. **Evolution Engine** (`src/evolution_engine.py`)
- **Status:** ✅ Verified at startup
- **Initialization:** Lazy-loaded, connected to Tournament Manager
- **Log Message:** `✓ EvolutionEngine available (will be connected to TournamentManager on demand)`
- **Purpose:** Genetic algorithm-based graph optimization

### Lazy-Loaded Components

These components are verified for availability but not initialized until first use:

#### 1. **Interpretability Engine** (`src/interpretability_engine.py`)
- **Status:** ✅ Available (lazy-load ready)
- **Initialization:** On first request for model interpretation
- **Log Message:** `✓ InterpretabilityEngine available (lazy-load ready)`
- **Dependencies:** torch, captum (optional), matplotlib (optional)
- **Purpose:** SHAP-like attributions, attention visualization, counterfactual tracing
- **Used in:** Arena (when interpretability is requested)

### Component Initialization in Sub-Apps

When running as mounted sub-apps in `full_platform.py`, the following apps have their own initialization:

#### 1. **VULCAN-AGI** (`src/vulcan/main.py`)
- **Lifespan:** Yes (FastAPI lifespan)
- **Components Initialized:**
  - World Model (causal reasoning, prediction)
  - Reasoning Systems (symbolic, probabilistic, causal, analogical)
  - Memory Systems (LTM, associative memory)
  - Learning Systems (continual, meta-learning)
  - Safety Validator
  - Curiosity Engine
  - Knowledge Crystallizer
  - Problem Decomposer
  - Semantic Bridge
- **Self-Improvement:** Optional, enabled via config
- **Log Messages:** `✓ [Component] activated` for each subsystem

#### 2. **Arena** (`src/graphix_arena.py`)
- **Lifespan:** No (standalone FastAPI app)
- **Components Initialized:**
  - DriftDetector
  - TournamentManager
  - InterpretabilityEngine (lazy)
  - DataAugmentor
  - NSOAligner
  - ObservabilityManager
- **Log Messages:** `✓ [Component] initialized in Arena`

#### 3. **Registry** (`app.py`)
- **Framework:** Flask (no async lifespan)
- **Components:** JWT authentication, audit logging, agent registry
- **Initialization:** On first import/request

### Port Assignments

| Service | Port | Mount Path | Type |
|---------|------|------------|------|
| Unified Platform | 8080 | / | FastAPI (main) |
| VULCAN | 8080 | /vulcan | FastAPI (mounted) |
| Arena | 8080 | /arena | FastAPI (mounted) |
| Registry | 8080 | /registry | Flask (mounted via WSGI) |
| API Gateway | 8080 | /api-gateway | FastAPI (mounted) |
| DQS Service | 8080 | /dqs | FastAPI (mounted) |
| PII Service | 8080 | /pii | FastAPI (mounted) |
| API Server | 8001 | - | Standalone HTTP |
| Registry gRPC | 50051 | - | Standalone gRPC |
| Listener | 8084 | - | Standalone HTTP |

### Health Check Endpoints

- **Platform:** `GET /health` - Overall platform health
- **Components:** `GET /health/components` - Detailed component status
- **Status API:** `GET /api/status` - Service and configuration status
- **Individual Services:** Each mounted service has `[mount_path]/health`

### Startup Summary

During startup, the platform logs a comprehensive summary:

```
PLATFORM STARTUP SUMMARY
======================================================================
Services:
  ✅ vulcan: MOUNTED (path /vulcan)
  ✅ arena: MOUNTED (path /arena)
  ✅ registry: MOUNTED (path /registry)
  ✅ api_server: RUNNING (PID: 12345)
  ✅ registry_grpc: RUNNING (PID: 12346)
  ✅ listener: RUNNING (PID: 12347)

Core Components:
  ✅ VULCAN World Model
  ✅ Reasoning (5/5)
  ✅ Semantic Bridge
  ✅ Agent Pool
  ✅ Unified Runtime
  ✅ Hardware Dispatcher
  ✅ Governance Loop
  ✅ Consensus Engine
  ✅ Security Audit Engine
  ✅ Graph Compiler
  ✅ Persistent Memory v46
  ✅ Conformal Prediction
  ✅ Drift Detector
  ✅ Pattern Matcher
  ✅ Superoptimizer
  ✅ Interpretability Engine
  ✅ Tournament Manager
  ✅ Evolution Engine

======================================================================
Services: 9/9 running
Components: 18/18 initialized
======================================================================
```

---

## For More Information

- **Complete Function Catalog:** [COMPLETE_SERVICE_CATALOG.md](COMPLETE_SERVICE_CATALOG.md)
- **API Reference:** [api_reference.md](api_reference.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Repository Overview:** [../COMPREHENSIVE_REPO_OVERVIEW.md](../COMPREHENSIVE_REPO_OVERVIEW.md)
- **Platform Architecture:** [../COMPLETE_PLATFORM_ARCHITECTURE.md](../COMPLETE_PLATFORM_ARCHITECTURE.md)

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Platform Version:** v1.0
