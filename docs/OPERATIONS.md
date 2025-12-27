# VulcanAMI Platform Operations Guide

**Version:** 2.2.0  
**Last Updated:** December 23, 2024

This comprehensive operations guide covers day-to-day operations, monitoring, observability, and troubleshooting for the VulcanAMI/GraphixVulcan platform.

> See also: [AI_OPS.md](AI_OPS.md) for AI-specific operations, [OBSERVABILITY.md](OBSERVABILITY.md) for detailed metrics.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Day-to-Day Operations](#2-day-to-day-operations)
3. [Observability & Monitoring](#3-observability--monitoring)
4. [Troubleshooting](#4-troubleshooting)
5. [Backup and Recovery](#5-backup-and-recovery)
6. [Performance Tuning](#6-performance-tuning)
7. [Disaster Recovery](#7-disaster-recovery)
8. [Operational Metrics](#8-operational-metrics)
9. [References](#9-references)

---

## 1. Overview

Graphix is an AI-native execution & evolution fabric for JSON-based directed graphs ("Graphix IR"). It powers VULCAN-AI: a cognitive architecture spanning perception, reasoning, optimization, alignment, safety, and hardware-aware dispatch.

### Capability Snapshot

| Capability | Summary | Operational Notes |
|------------|---------|-------------------|
| Graph Execution | Multi-mode (sequential, parallel, streaming, batch) with deterministic caching | Tune `max_parallel_tasks` & timeout configs per environment |
| Self-Evolution | Proposal lifecycle: submit → validate → consensus → apply → observe | Enforce replay window & similarity dampening for proposal hygiene |
| Hardware Acceleration | Photonic & memristor emulation + backend strategy selection | Fallback chain: Real → Emulated → CPU; monitor energy_nj and latency |
| Ethical Governance | Multi-model audit (LLM consensus), risky pattern removal | NSOAligner flags: eval/exec, path traversal, bias taxonomy |
| Observability | Prometheus/Grafana integration, structured audit chain | Enable metrics export only behind internal gateway |
| Testing & QA | Pytest suites (validation, hardware emulation, stress, E2E) | Parallelize with `pytest -n auto`; tag slow vs fast tests |

---

## 2. Day-to-Day Operations

### 2.1 Environment Setup

```bash
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
export PYTHONPATH=.
```

### 2.2 Service Management

**Start Services:**

```bash
# Option A - Registry API (Flask)
python app.py

# Option B - Arena API (FastAPI)
uvicorn src.graphix_arena:app --reload

# Option C - Unified Platform
python src/full_platform.py
```

**Docker Compose:**

```bash
# Development
docker compose -f docker-compose.dev.yml up -d

# Production
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.dev.yml ps

# View logs
docker compose -f docker-compose.dev.yml logs -f
```

### 2.3 Health Checks

| Action | Command | Expected |
|--------|---------|----------|
| Registry API | `curl http://localhost:5000/health` | `{"status": "healthy"}` |
| Arena API | `curl http://localhost:8000/health` | `{"status": "healthy"}` |
| Unified Platform | `curl http://localhost:8080/health` | Component status |
| **Fast Liveness** | `curl http://localhost:8000/health/live` | `{"status": "alive"}` |
| **Fast Readiness** | `curl http://localhost:8000/health/ready` | `{"status": "ready"}` |
| Metrics | `curl http://localhost:8000/metrics` | Prometheus payload |

> **Note:** Use `/health/live` and `/health/ready` for Kubernetes liveness/readiness probes.
> These endpoints are faster (<100ms) than the comprehensive `/health` endpoint.

### 2.4 Smoke Tests

```bash
# Validation Test
pytest -q src/run_validation_test.py

# Evolution Mini-Tournament
python scripts/run_sentiment_tournament.py --mode offline --generations 3

# Photonic Fallback
python -m src.hardware_dispatcher --test_mvm --provider Lightmatter
```

---

## 3. Observability & Monitoring

### 3.1 Metrics Taxonomy

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `graphix_nodes_executed` | Counter | graph | Node execution count |
| `graphix_nodes_failed` | Counter | graph | Failure count |
| `graphix_total_latency_ms` | Gauge | graph | Wall-clock latency |
| `graphix_cache_hit_rate` | Gauge | graph | Deterministic node cache efficiency |
| `graphix_success_rate` | Gauge | graph | Per-graph success fraction |
| `graphix_throughput_nodes_per_sec` | Gauge | graph | Nodes/sec |
| `graphix_rss_mb` | Gauge | graph | Resident memory |
| `graphix_cpu_percent` | Gauge | graph | CPU usage |
| `graphix_safety_blocks_total` | Counter | graph | Safety validation rejections |
| `graphix_energy_nj_total` | Counter | backend | Accumulated energy estimate |

### 3.2 Key Metrics and Thresholds

| Metric | Description | Threshold / Alert |
|--------|-------------|------------------|
| `graphix_success_rate` | Graph-level success fraction | < 0.90 triggers investigation |
| `graphix_cache_hit_rate` | Deterministic node cache efficiency | Deterioration slope > 20% |
| `graphix_vulcan_safety_blocks_total` | Count of safety rejections | Sudden spike: security review |
| `graphix_energy_nj_total` | Cumulative estimated energy | Anomaly > x3 baseline → dispatch tuning |
| `graphix_nodes_failed` | Failed node executions | Persistent growth → handler audit |

### 3.3 Prometheus Access

```bash
# Local
http://localhost:9090

# Kubernetes (port-forward)
kubectl port-forward -n vulcanami svc/prometheus 9090:9090
```

### 3.4 Grafana Dashboards

```bash
# Local
http://localhost:3000
# Default: admin / [GRAFANA_PASSWORD]

# Kubernetes (port-forward)
kubectl port-forward -n vulcanami svc/grafana 3000:3000
```

**Suggested Dashboard Panels:**
- Node latency p95/p99 trend
- Success vs failure counts
- Cache hit rate over time
- Safety block spikes
- Consensus approval latency
- Energy usage trends (hardware dispatch)

### 3.5 Cardinality Control

- Hash long IDs (proposal_id) if long
- Avoid dynamic free-form labels (user input)
- Limit node_type expansion; use experimental gating

### 3.6 Tracing

Optional spans: node execution (attributes: node_type, latency_ms, cache_hit); parent graph span linking causal chain.

### 3.7 Audit & Provenance

Hash-linked audit events; provenance graph indexing output consumption edges for lineage queries.

---

## 4. Troubleshooting

For detailed troubleshooting, see [troubleshooting.md](troubleshooting.md).

### Quick Reference

| Symptom | Likely Cause | Remediation |
|---------|--------------|------------|
| Endless pending nodes | Missing upstream outputs / cycle | Run validator; inspect audit log chain |
| High latency spike | Hardware fallback thrash | Inspect dispatch strategy, reduce concurrency |
| Repeated safety blocks | Proposal spam / alignment drift | Increase similarity threshold; audit NSOAligner weights |
| Cache hit collapse | Node determinism misflagged | Add `is_deterministic` param or adjust handler purity |
| Arena 500 errors | Missing env / provider creds | Verify `.env`, run health endpoint checks |

---

## 5. Backup and Recovery

### 5.1 Artifacts to Backup

| Artifact | Location | Strategy |
|----------|----------|----------|
| Audit DB | `audit.db` (dev) | Daily snapshot → integrity hash chain |
| Champions | `evolution_champions/` | Retain top N per day; prune by age |
| Governance Artifacts | `governance_artifacts/` | Keep all signed proposals (immutable store) |
| Learned Patterns | `learned_subgraphs/` | Versioned, LRU eviction for stale low-confidence |

### 5.2 Database Backup

```bash
# PostgreSQL backup (development)
kubectl exec -n vulcanami-development postgres-0 -- pg_dump -U vulcanami vulcanami > backup.sql

# PostgreSQL backup (production)
kubectl exec -n vulcanami-production postgres-0 -- pg_dump -U vulcanami vulcanami > backup.sql

# Restore
kubectl exec -i -n vulcanami postgres-0 -- psql -U vulcanami vulcanami < backup.sql
```

### 5.3 Application State

```bash
# Backup MinIO data
mc mirror minio/vulcanami-hot ./backup/

# Restore
mc mirror ./backup/ minio/vulcanami-hot
```

### 5.4 Recovery Process

1. Restore DB snapshot
2. Re-verify audit hashes
3. Replay last applied proposals if necessary
4. Validate system health

---

## 6. Performance Tuning

### 6.1 Using Telemetry for Optimization

- Identify hotspots (node_type latency tail)
- Adjust concurrency
- Apply batching for micro-transform nodes

### 6.2 Anomaly Heuristics

- Latency EWMA drift > factor threshold
- Error burst ratio > baseline × multiplier
- Alignment conflict frequency > rolling mean × multiplier
- Cache hit rate deterioration slope negative > threshold

### 6.3 Resource Tuning

```bash
# Enable Simple Mode for faster responses
export VULCAN_SIMPLE_MODE=true
export SKIP_BERT_EMBEDDINGS=true
export OPENAI_ONLY_MODE=true

# Reduce memory footprint
export MAX_PROVENANCE_RECORDS=50
export PROVENANCE_TTL_SECONDS=1800

# Reduce agent pool
export MIN_AGENTS=1
export MAX_AGENTS=5
```

### 6.4 Data Integrity

- Validate non-negative durations
- Enforce audit chain hash verification
- Deduplicate events

---

## 7. Disaster Recovery

### 7.1 RTO/RPO Targets

| Tier | RTO | RPO | Description |
|------|-----|-----|-------------|
| Critical | 1 hour | 15 minutes | Production services |
| Standard | 4 hours | 1 hour | Development/staging |
| Archive | 24 hours | 24 hours | Historical data |

### 7.2 Backup Procedures

1. **Continuous**: Audit log streaming to immutable store
2. **Hourly**: Database snapshots
3. **Daily**: Full system state backup
4. **Weekly**: Archive rotation

### 7.3 Recovery Procedures

1. **Service Failure**: 
   - Check health endpoints
   - Restart affected service
   - Validate audit chain integrity

2. **Data Corruption**:
   - Stop affected services
   - Restore from last known good snapshot
   - Replay audit log from checkpoint
   - Validate data integrity

3. **Complete Failure**:
   - Provision new infrastructure
   - Restore database from backup
   - Restore application state from S3
   - Replay audit log
   - Validate full system

### 7.4 Failover Procedures

```bash
# Kubernetes rollback
kubectl rollout undo deployment/vulcanami-api -n vulcanami

# Check rollout status
kubectl rollout status deployment/vulcanami-api -n vulcanami

# Helm rollback
helm rollback vulcanami -n vulcanami
```

---

## 8. Operational Metrics

### 8.1 Governance / Evolution Flow

1. Proposal submission (`ProposalNode` or API)  
2. Validator pipeline (structure → ontology → semantics → security → alignment → safety)  
3. Consensus threshold check (trust-weighted)  
4. Apply & hash chain update  
5. Post-run observation & pattern mining  
6. Autonomous optimizer may propose refinement if fitness < threshold

Ensure all proposals log fingerprint + risk vector; assert uniqueness inside replay window (configurable TTL).

### 8.2 Security Hygiene

- Never commit real API keys
- Use mock placeholders for demos
- Prevent unreviewed ontology or grammar edits by gating branch protections
- Audit chain mismatch → immediate containment

---

## 9. References

| Domain | Documentation |
|--------|---------------|
| AI Operations | [AI_OPS.md](AI_OPS.md) |
| Observability | [OBSERVABILITY.md](OBSERVABILITY.md) |
| Troubleshooting | [troubleshooting.md](troubleshooting.md) |
| Deployment | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Security | [SECURITY.md](SECURITY.md) |
| Evolution | [EVOLUTION_ROADMAP.MD](EVOLUTION_ROADMAP.MD) |
| Training | [AI_TRAINING_GUIDE.md](AI_TRAINING_GUIDE.md) |
| Visualization | [visualization_guide.md](visualization_guide.md) |

---

**Document Version:** 2.2.0  
**Last Updated:** December 23, 2024
