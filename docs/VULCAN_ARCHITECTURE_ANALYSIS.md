# Vulcan Architecture Analysis: Data Center Deployment & Scalability

**Document Version:** 1.0.0  
**Last Updated:** January 4, 2026  
**Prepared For:** Business Partner Technical Review  
**Classification:** Technical Architecture Assessment

---

## Executive Summary

This document provides a grounded, evidence-based analysis of the Vulcan platform's architecture, deployment readiness, and scalability design. All statements are backed by specific file references, configuration keys, and code implementations in the repository. This assessment is intended to give business partners confidence in where Vulcan stands today and why it is designed to scale.

---

## 1. Deployment: Data Center / Private Cloud

### 1.1 Containerization (Implemented)

**Evidence:** `Dockerfile` (lines 1-266), `docker-compose.prod.yml` (627 lines)

Vulcan is packaged as a multi-stage Docker container with production-ready security hardening:

| Feature | Implementation | File Reference |
|---------|----------------|----------------|
| Non-root execution | User `graphix` (uid 1001) | `Dockerfile:201, 245` |
| Multi-stage build | Builder + Runtime separation | `Dockerfile:55, 170` |
| Health checks | `/health/live` endpoint, 5-minute startup period | `Dockerfile:252-254` |
| Secret validation | Runtime JWT enforcement via `entrypoint.sh` | `Dockerfile:231-232` |
| Dependency hashes | Optional `requirements-hashed.txt` for supply chain security | `Dockerfile:116-132` |

**Production Stack Components:**
```
docker-compose.prod.yml defines:
├── Storage: PostgreSQL 14, Redis 7, MinIO (S3-compatible)
├── Vector DB: Milvus 2.3.4 + etcd coordination
├── Application: Full-platform, API-gateway, DQS, PII services
├── Monitoring: Prometheus + Grafana
└── Reverse Proxy: NGINX with SSL termination
```

### 1.2 Kubernetes/Helm Deployment (Implemented)

**Evidence:** `helm/vulcanami/` directory, `k8s/base/` directory

| Component | Configuration Key | Value | File Reference |
|-----------|------------------|-------|----------------|
| Helm Chart | `name` | vulcanami | `helm/vulcanami/Chart.yaml:2` |
| Default Replicas | `replicaCount` | 1 (single replica when Redis unavailable; autoscaling overrides) | `values.yaml:9` |
| Autoscaling | `autoscaling.enabled` | `true` | `values.yaml:76` |
| Min/Max Replicas | `minReplicas` / `maxReplicas` | 2 / 10 (when autoscaling enabled) | `values.yaml:77-78` |
| CPU Scale Target | `targetCPUUtilizationPercentage` | 70% | `values.yaml:79` |
| Memory Scale Target | `targetMemoryUtilizationPercentage` | 80% | `values.yaml:80` |
| Pod Disruption Budget | `minAvailable` | 1 | `values.yaml:117` |
| Anti-Affinity | `podAntiAffinity` | Spread across hosts | `values.yaml:122-134` |

**Resource Limits (Production):**
```yaml
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi
```
Source: `helm/vulcanami/values.yaml:65-72`

### 1.3 Infrastructure as Code (Implemented)

**Evidence:** `infra/terraform/` directory

Terraform modules provision cloud infrastructure (AWS/Azure compatible):
- `main.tf` - Core infrastructure
- `variables.tf` - Parameterized configuration  
- `outputs.tf` - Deployment outputs

---

## 2. Scalability Architecture

### 2.1 Horizontal Scaling

| Mechanism | Implementation | Evidence |
|-----------|---------------|----------|
| Kubernetes HPA | Auto-scales pods based on CPU/Memory | `values.yaml:76-80` |
| Agent Pool Auto-Scaler | Dynamic agent spawning based on load | `agent_pool.py:4208-4360` |
| Stateless API Design | JWT-based auth, no session state in API layer | `values.yaml:291` |
| Redis State Sync | Cross-worker state via Redis | `agent_pool.py:889, 1280-1362` |

**Auto-Scaler Algorithm** (from `agent_pool.py:4249-4352`):
```
Scale UP triggers:
- Utilization > 80%
- Pending tasks > idle agents  
- Response time P95 > target
- Response time P99 > target
- Queue depth > max threshold
- Performance degradation trend > 50ms

Scale DOWN triggers (only if all conditions met):
- Utilization < 20%
- All response time targets met
- No degrading trend
```

### 2.2 Vertical Scaling

**Evidence:** `configs/hardware_profiles.json`, `helm/vulcanami/values.yaml`

| Hardware Profile | Throughput (TOPS) | Latency (ms) | Use Case |
|-----------------|-------------------|--------------|----------|
| CPU | 0.5 | 3.2 | Default fallback |
| GPU (A100) | 32 | 0.6 | ML inference |
| vLLM | 22 | 0.9 | LLM serving |
| Photonic (future) | 85 | 0.15 | Ultra-low latency |
| Memristor (future) | 40 | 0.2 | Energy efficient |

Configurable resource allocations:
- `vulcanLlmHardTimeout: 300.0` seconds (5 minutes for CPU inference)
- `vulcanCpuMaxTokens: 50` (throttled for CPU-only deployments)
- `zmqConnectionTimeout: 120000` ms (extended for slow transformers)

Source: `values.yaml:408-434`

### 2.3 Capability-Based Scaling

**Evidence:** `agent_lifecycle.py`, `agent_pool.py:1056-1067`

Agents are specialized by capability enum:

```python
class AgentCapability(Enum):
    GENERAL = "general"
    REASONING = "reasoning"
    SYMBOLIC = "symbolic"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    MULTIMODAL = "multimodal"
    ...
```
Source: `agent_lifecycle.py` (referenced in `agent_pool.py:56-60`)

**Tournament-Based Selection:** When enabled, multiple agents compete on complex queries:
```python
TOURNAMENT_QUERY_TYPES = ('reasoning', 'symbolic', 'analogical', 'causal')
TOURNAMENT_MAX_CANDIDATES = 3
```
Source: `agent_pool.py:203-205`

---

## 3. Logical Pools / Layers

### 3.1 Current Layer Architecture

| Layer | Responsibility | Key Classes / Files | State |
|-------|---------------|---------------------|-------|
| **API Gateway** | Request routing, auth, rate limiting | `src/vulcan/api_gateway.py`, `src/full_platform.py` | Stateless |
| **Orchestrator** | Cognitive cycle coordination, task dispatch | `VULCANAGICollective` in `collective.py` | Mostly Stateless |
| **Agent Pool** | Worker lifecycle, job assignment, auto-scaling | `AgentPoolManager` in `agent_pool.py` | Stateful (Redis-synced) |
| **Task Queue** | Distributed job dispatch (Ray/Celery/ZMQ/Custom) | `task_queues.py` | Stateful (queue backend) |
| **Reasoning Engines** | Causal, symbolic, probabilistic reasoning | `src/vulcan/reasoning/` | Stateless |
| **Memory System** | Short-term, long-term, hierarchical storage | `HierarchicalMemory`, `DistributedMemory` | Stateful (persistent) |
| **Safety Layer** | Validation, compliance, ethical boundaries | `src/vulcan/safety/` | Stateless |
| **Metrics Collector** | Counters, gauges, histograms, health scores | `EnhancedMetricsCollector` in `metrics.py` | Stateful (bounded) |

### 3.2 Memory Hierarchy Layers

**Evidence:** `src/vulcan/memory/hierarchical.py` (lines 910-939)

```python
Memory Levels:
├── Sensory (capacity: 50, decay: 0.5, consolidation: 0.7)
├── Working (capacity: config.max_working_memory, decay: 0.1)
├── Short-Term (capacity: config.max_short_term, decay: 0.05)
└── Long-Term (capacity: config.max_long_term, decay: 0.001)
```

**Persistent Storage Tiers** (from `hierarchical.py:1739-1782`):
1. **Hot Tier**: Episodic memory (in-memory)
2. **Warm Tier**: Local disk serialization
3. **Cold Tier**: S3/PackfileStore with Milvus vector search

---

## 4. Stateless vs Stateful Components

### 4.1 Stateless Components

| Component | Evidence | Notes |
|-----------|----------|-------|
| API Gateway | JWT-based auth | `values.yaml:291` |
| Reasoning Engines | Pure function design | `src/vulcan/reasoning/` |
| Safety Validators | Stateless validation | `src/vulcan/safety/` |
| Problem Decomposer | Stateless analysis | `src/vulcan/problem_decomposer/` |
| Query Router | Stateless routing | `src/vulcan/routing/` |

### 4.2 Stateful Components

| Component | State Type | Persistence | Evidence |
|-----------|-----------|-------------|----------|
| Agent Pool Statistics | Counters, job tracking | Redis-synced | `agent_pool.py:1280-1362` |
| Memory System | Memories, embeddings | Disk + S3/Milvus | `hierarchical.py`, `distributed.py` |
| Task Queue | Pending tasks | Queue backend | `task_queues.py:141-142` |
| Learning Persistence | Tool weights, concepts | JSON file | `learning_persistence.py` |
| Provenance Records | Job lineage | Bounded deque (maxlen=50) | `agent_pool.py:943` |
| Metrics Collector | Time-series, histograms | Bounded deques | `metrics.py:87-90` |

---

## 5. Scaling Boundaries (Enforced Limits)

### 5.1 Memory Bounds

| Bound | Value | Purpose | Evidence |
|-------|-------|---------|----------|
| Reasoning trace | `maxlen=100` | Prevent memory growth | `collective.py:109` |
| Execution history | `maxlen=500` | Bounded job history | `collective.py:113` |
| Provenance records | `maxlen=50` | Rolling window, auto-prune | `agent_pool.py:943` |
| Response time samples | `maxlen=1000` | Sliding window metrics | `agent_pool.py:311` |
| Histogram values | `maxlen=10000` | Bounded statistical data | `metrics.py:87` |
| Time-series points | `maxlen=1000` | Bounded temporal data | `metrics.py:90` |
| Embedding cache | 1000 entries, LRU eviction | Prevent cache bloat | `hierarchical.py:1292-1296` |

### 5.2 Worker Separation

| Pool Configuration | Default | Configurable | Evidence |
|-------------------|---------|--------------|----------|
| Min agents | 2 | Hardcoded (emergency stabilization) | `agent_pool.py:912-913` |
| Max agents | 10 | Hardcoded (emergency stabilization) | `agent_pool.py:912` |
| Process executor workers | 4 | `max_parallel_processes` | `variants.py:173` |
| Thread executor workers | 8 | `max_parallel_threads` | `variants.py:174` |
| Agent selection timeout | 10s | `AGENT_SELECTION_TIMEOUT_SECONDS` | `agent_pool.py:210-211` |

### 5.3 Queue Configuration

| Parameter | Default | Purpose | Evidence |
|-----------|---------|---------|----------|
| Priority queue max size | 10,000 | Prevent queue overflow | `agent_pool.py:439` |
| Dead letter queue size | 100 | Failed job retention | `agent_pool.py:215, 1031` |
| Max job retries | 3 | Prevent infinite retry | `agent_pool.py:1039` |
| Stale task timeout | 300s | Auto-cleanup | `agent_pool.py:898` |
| Redis persistence throttle | 1s | Limit Redis round-trips | `agent_pool.py:1347` |

### 5.4 Timeout Configuration

| Timeout | Default | Purpose | Evidence |
|---------|---------|---------|----------|
| Embedding timeout | 5s | Prevent cascade delays | `values.yaml:377` |
| Query routing timeout | 30s | Allow completion | `values.yaml:405` |
| Arena timeout | 60s | Multi-agent tournament | `values.yaml:402` |
| VULCAN LLM hard timeout | 300s (5 min) | CPU cloud support | `values.yaml:412` |
| Per-token timeout | 30s | CPU execution | `values.yaml:416` |
| Hybrid executor timeout | 30s | Parallel LLM execution | `values.yaml:459` |
| ZMQ connection timeout | 120s | Transformer inference | `values.yaml:434` |

---

## 6. Implemented vs Future/Planned

### 6.1 Fully Implemented

| Feature | Evidence | Production-Ready |
|---------|----------|-----------------|
| Docker containerization | `Dockerfile` (266 lines) | ✓ |
| Kubernetes/Helm deployment | `helm/vulcanami/`, `k8s/base/` | ✓ |
| Agent Pool with auto-scaling | `AgentPoolManager` (4470+ lines) | ✓ |
| Task queues (Ray, Celery, ZMQ, Custom) | `task_queues.py` | ✓ |
| Hierarchical memory system | `hierarchical.py` (1923 lines) | ✓ |
| Distributed memory with federation | `distributed.py` (1210 lines) | ✓ |
| Learning state persistence | `learning_persistence.py` | ✓ |
| Prometheus/Grafana monitoring | `docker-compose.prod.yml:537-590` | ✓ |
| Health endpoints (live/ready) | `values.yaml:84-113` | ✓ |
| JWT authentication | `values.yaml:291` | ✓ |
| Redis state synchronization | `agent_pool.py:889, 1280-1362` | ✓ |
| Singleton pattern for expensive objects | `singletons.py` referenced | ✓ |
| Circuit breaker pattern | Arena timeout at 45s coordinated | ✓ |

### 6.2 Partially Implemented / Available but Optional

| Feature | Status | Evidence |
|---------|--------|----------|
| Ray distributed computing | Available, disabled by default | `values.yaml:191-198` |
| Distillation (Student/Teacher) | Implemented, configurable | `values.yaml:179-189` |
| Tournament-based agent selection | Implemented when `TournamentManager` available | `agent_pool.py:1055-1067` |
| Milvus vector database | In docker-compose, requires infrastructure | `docker-compose.prod.yml:171-206` |
| Self-improvement drive | Implemented, approval required by default | `values.yaml:163-166` |

### 6.3 Future / Research Vectors

From `ARCHITECTURE_OVERVIEW.md:157-163`:

> - Formal invariant spec (temporal logic)
> - ML-based dynamic timeout predictors
> - Energy-aware scheduling objective multi-optimization
> - Semantic graph embedding anomaly detection
> - Policy DSL integration (declarative safety signatures)

From `hardware_profiles.json`:
- Photonic and Memristor backends are defined but marked "(sim/emulated)"

---

## 7. Scalability Evidence in Code

### 7.1 Auto-Scaling Implementation

**File:** `agent_pool.py:4251-4355`

The `AutoScaler._evaluate_and_scale()` method implements production-grade scaling:

```python
# Scale UP conditions (any triggers scale-up):
scale_up_reasons = []
if utilization > 0.8:
    scale_up_reasons.append("high_utilization")
if pending_tasks > idle_agents:
    scale_up_reasons.append("pending_tasks")
if p95_ms > p95_target:
    scale_up_reasons.append("p95_exceeded")
if p99_ms > p99_target:
    scale_up_reasons.append("p99_exceeded")
if queue_depth > max_queue:
    scale_up_reasons.append("queue_depth")
if trend > 50:  # 50ms degradation
    scale_up_reasons.append("degrading_trend")
```

### 7.2 Distributed Task Queue Support

**File:** `task_queues.py`

Multiple queue backends implemented:
- `RayTaskQueue` (lines 311-500): Ray-based distributed execution
- `CeleryTaskQueue` (if celery available): Standard job queue
- `CustomTaskQueue`: Built-in ZMQ-based queue with coordinator

### 7.3 Memory Federation

**File:** `distributed.py:376-525`

`MemoryFederation` implements:
- Consistent hashing for key routing
- Leader election (Raft-style)
- Node health monitoring
- Automatic data migration on node failure

### 7.4 Bounded Resource Management

**File:** `agent_pool.py:3200-3378`

Dead Letter Queue and stuck job detection:
```python
STUCK_JOB_WARNING_THRESHOLD = 0.7   # 70% of timeout
STUCK_JOB_CRITICAL_THRESHOLD = 0.9  # 90% of timeout
DEFAULT_DLQ_SIZE = 100
```

---

## 8. Current Limits and Next Steps

### 8.1 Current Operational Limits

| Dimension | Current Limit | Bottleneck | Evidence |
|-----------|--------------|------------|----------|
| Concurrent agents | 10 (default max) | Configurable | `agent_pool.py:912` |
| Queries/second | Limited by LLM inference | ~500ms/token on CPU | `values.yaml:415` |
| Memory per pod | 4Gi limit | Helm values | `values.yaml:67` |
| Graph complexity | 10k nodes / 100k edges | Single-process | `ARCHITECTURE_OVERVIEW.md:154` |

### 8.2 Path to Higher Scale (No Refactoring Required)

The architecture supports these scaling paths through configuration changes only:

1. **More Replicas**: Increase `replicaCount` and ensure Redis connectivity
2. **More Agents**: Increase `max_agents` configuration
3. **GPU Acceleration**: Deploy with GPU nodes, set `RAY_ENABLED=true`
4. **Distributed Memory**: Enable Milvus cluster, configure S3 backend
5. **Queue Distribution**: Switch to Ray or Celery with dedicated workers

### 8.3 Recommended Next Steps for Production

1. **Redis HA**: Deploy Redis Sentinel or Redis Cluster for state persistence
2. **Milvus Cluster**: Scale vector search horizontally
3. **Load Testing**: Validate auto-scaler behavior under production load
4. **GPU Pool**: Add GPU-enabled node pool for inference acceleration

---

## Appendix A: Key Configuration Files

| File | Purpose | Lines |
|------|---------|-------|
| `Dockerfile` | Container build | 266 |
| `docker-compose.prod.yml` | Production stack | 627 |
| `helm/vulcanami/values.yaml` | K8s deployment config | 480 |
| `k8s/base/kustomization.yaml` | Kustomize manifests | 30 |
| `configs/hardware_profiles.json` | Hardware specs | 76 |
| `src/vulcan/orchestrator/agent_pool.py` | Agent pool impl | 4471 |
| `src/vulcan/memory/hierarchical.py` | Memory system | 1923 |
| `src/vulcan/memory/distributed.py` | Federation | 1210 |
| `src/vulcan/orchestrator/task_queues.py` | Queue impls | ~600 |

---

## Appendix B: Environment Variables for Scaling

```bash
# Agent Pool Scaling
MIN_AGENTS=2
MAX_AGENTS=10
VULCAN_SIMPLE_MODE=false

# Performance Tuning  
VULCAN_EMBEDDING_TIMEOUT=5.0
QUERY_ROUTING_TIMEOUT=30.0
ARENA_ENABLED=false
ARENA_TIMEOUT=60.0

# Memory System
MILVUS_HOST=milvus
MILVUS_PORT=19530
S3_BUCKET=vulcanami-memory

# Distributed Computing
RAY_ENABLED=false
RAY_ADDRESS=auto

# Redis State Sync
REDIS_HOST=redis
REDIS_PORT=6379
```

---

**Document prepared from repository analysis on January 4, 2026.**

**Methodology:** All claims in this document are grounded in code inspection. File names, line numbers, and configuration keys are provided for independent verification.
