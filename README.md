# Graphix Vulcan (Proprietary)



Proprietary software owned and created by Novatrax Labs LTD. All rights reserved.

Copyright © 2024 Novatrax Labs LTD. All rights reserved.
Patents pending

Important legal notice
- This software and documentation are the confidential and proprietary information of Novatrax Labs LTD (“Novatrax”).
- Any use is subject to your written agreement with Novatrax. No license is granted by implication, estoppel, or otherwise.
- Do not distribute, disclose, copy, or host this code or documentation outside of approved environments and recipients identified in your agreement.
- If you received this in error, notify Novatrax and delete all copies.

Last updated: 2024-12-05

---

## Overview

Graphix Vulcan is Novatrax’s AI-native graph execution and governance platform. It enables you to:

- Represent complex AI and data workflows as a typed, JSON-based intermediate representation (IR).
- Validate and execute the graph as a safe, observable, and scalable DAG.
- Govern changes to graphs via a trust-weighted consensus engine with policy hooks.
- Collect deep observability (latency, errors, counters, disk/cleanup) and export to Prometheus/Grafana.
- Audit sensitive actions to a secure store with optional alerting (e.g., Slack).

Typical use cases
- Safety-governed agentic systems
- Provenance-aware ML operations
- Orchestrating LLM and tool pipelines with control and visibility
- Policy-driven evolution of workflows across teams

---

## Key capabilities

- Graph IR and validation
  - Typed nodes/edges with size limits, unique IDs, and cycle detection.
  - Policy hooks to extend validation for domain and safety rules.

- Executor
  - Concurrent, layerized execution for DAGs with per-node error handling and timeouts.
  - Extensible node types (input, transform, filter, generative, combine, output).
  - Observability and audit integrated at node and graph levels.

- Governance and consensus
  - Trust-weighted voting (approve/reject/abstain) with quorum thresholds.
  - Proposal lifecycle (draft/open/approved/rejected/expired/applied/failed).
  - Optional VULCAN world-model assessment hooks for additional risk/context checks.
  - Thread-safe operations with periodic cleanup of expired proposals.

- Observability and dashboards
  - Prometheus metrics: latency histograms (p50/p95), errors, explainability gauges, cleanup stats, disk usage, and more.
  - Auto-generated Grafana dashboard JSON export with example alert thresholds.

- Security and audit
  - SQLite-backed audit trail with WAL, integrity checks, and recovery routines.
  - Selective alerting to channels (e.g., Slack) with severity filtering and stats.
  - Rate limiting and JWT/API key layers for service endpoints.

---

## System requirements

- Operating system: Linux x86_64 (recommended), macOS (development)
- Python: 3.10.11
- Optional services:
  - Redis (rate limiting and caching)
  - Prometheus (metrics scrape) and Grafana (dashboards)
  - Slack Incoming Webhook (security alerts)
- Storage:
  - SQLite by default (embedded); external DB may be supported under enterprise deployment

Note: Platform components and integrations are configurable; enterprise deployment patterns may differ from development defaults.

---

## Quick start (development)

Important: The steps below are for internal or licensed development environments only. Do not expose development services to the public internet.

1) Clone and environment
```bash
git clone <your-internal-repo-url> graphix_vulcan
cd graphix_vulcan
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2) Install
```bash
pip install --upgrade pip
pip install -r requirements.txt

# For development (includes testing, linting, and code quality tools)
pip install -r requirements-dev.txt
```

3) Configure environment
Set required secrets via environment variables or a secure secret manager. At minimum:
```bash
# Use .env.example as a template
cp .env.example .env
# Edit .env and set the following variables:
# - JWT_SECRET_KEY=<strong-unique-secret>
# - BOOTSTRAP_KEY=<one-time-bootstrap-secret>  # only needed to create the initial admin/agent
# - REDIS_URL=redis://<host>:<port>            # optional; falls back to in-memory rate limiting
# - AUDIT_DB_PATH=./audit.db                   # default shown; secure paths recommended in production
# - SLACK_WEBHOOK_URL=<optional-for-alerts>

Example (development only):
```bash
export JWT_SECRET_KEY="dev-only-change-me"
export BOOTSTRAP_KEY="dev-bootstrap"
export REDIS_URL="redis://localhost:6379"
export AUDIT_DB_PATH="./audit.db"
# export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

4) Start a service

Option A — Registry API (Flask)
```bash
python app.py
# Health check: curl http://127.0.0.1:5000/health
# Metrics (if enabled): curl http://127.0.0.1:5000/metrics
```

Option B — Arena API (FastAPI)
```bash
uvicorn src.graphix_arena:app --host 0.0.0.0 --port 8000 --reload
# Health check: curl http://127.0.0.1:8000/health
# Metrics (if enabled): curl http://127.0.0.1:8000/metrics
```

5) Minimal executor demo (local)
```bash
python src/minimal_executor.py
```

Notes
- The services above are alternative entry points commonly used during development. Your licensed deployment may provide a consolidated or managed runtime with additional controls.
- Do not use development defaults in production.

### Validation & CI/CD

This repository includes comprehensive validation and testing tooling to ensure reproducible builds and correct CI/CD configuration:

```bash
# ⭐ NEW: Simulate all possible reproducibility build scenarios (RECOMMENDED)
./simulate_all_builds.sh --skip-docker  # Full validation (29 scenarios)
./simulate_all_builds.sh --quick        # Quick validation before commits

# Quick validation (recommended before commits)
./quick_test.sh quick

# Quick validation of specific components
./quick_test.sh docker      # Docker tests only
./quick_test.sh security    # Security tests only
./quick_test.sh k8s         # Kubernetes tests only

# Full comprehensive test suite (42+ checks)
./test_full_cicd.sh

# Run pytest test suite
pytest tests/test_cicd_reproducibility.py -v

# Run existing validation script
./validate_cicd_docker.sh
```

**What is validated:**
- ✅ All possible reproducibility build scenarios (29 scenarios tested)
- ✅ Docker and Docker Compose v2 configurations
- ✅ Hash-verified dependencies (requirements-hashed.txt with 4,007 SHA256 hashes)
- ✅ Docker security features (non-root user, health checks, JWT validation)
- ✅ GitHub Actions workflows (YAML validation)
- ✅ Kubernetes manifests (multi-document YAML support)
- ✅ Helm charts (lint validation)
- ✅ Security configuration (no committed secrets)
- ✅ Reproducibility settings (pinned versions)
- ✅ Python dependencies (440 pinned packages, no vulnerabilities)

**Expected output:**
```
Total Scenarios Tested: 29
Passed: 25 (✓)
Failed: 0 (✗)
Skipped: 4 (⊘)
Pass Rate: 100%

Status: ✅ 100% READY FOR DEVELOPMENT ✓
```

**Docker Compose v2 Note**: This repository uses modern Docker Compose v2 syntax (`docker compose` not `docker-compose`). Docker Compose v2 is bundled with Docker Engine 20.10.13+.

For comprehensive testing documentation, see:
- **⭐ [BUILD_SIMULATION_REPORT.md](BUILD_SIMULATION_REPORT.md)** - Complete build simulation report (NEW)
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing guide
- [CI_CD.md](CI_CD.md) - CI/CD pipeline documentation
- [REPRODUCIBLE_BUILDS.md](REPRODUCIBLE_BUILDS.md) - Reproducible build guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions

---

## Configuration reference (selected)

Authentication and authorization
- JWT_SECRET_KEY: required for JWT signing/verification (Registry API)
- API keys: X-API-Key header (Arena API), if enabled
- BOOTSTRAP_KEY: one-time onboarding key to create the first admin/agent

Rate limiting and caching
- REDIS_URL: if present, used as the rate-limit backend; otherwise in-memory fallback is used (development only)

Observability
- Exposes Prometheus metrics when enabled; ensure your service exports a /metrics endpoint using the provided registry
- Grafana dashboard JSON export available from the observability manager

Security auditing
- AUDIT_DB_PATH: path to the SQLite audit database
- SLACK_WEBHOOK_URL: optional, to receive high-severity alerts
- Retention and cleanup settings are configurable in code or via env in managed deployments

Governance and consensus
- Quorum, approval thresholds, and expiry settings are tunable; defaults provided in code are development-friendly

---

## Architecture (high level)

- Graph IR
  - JSON schema capturing nodes, edges, metadata, and policy constraints

- Validation
  - Structural validation (node/edge shapes, references, cycles, size limits)
  - Policy and safety hooks (custom domain validators)

- Execution
  - DAG scheduler with layerized concurrency
  - Per-node handlers (transform/filter/generative/combine/etc.)
  - Per-node timeouts, error taxonomy, and outcomes

- Governance
  - Proposals (draft → open → approved/rejected → applied/failed)
  - Trust-weighted votes; quorum and thresholds
  - Application layer applies IR mutations safely; audit and observability integrated

- Observability
  - Prometheus metrics via a per-process registry
  - Dashboard JSON (Grafana) and alert examples

- Security and audit
  - Persistent audit DB; integrity checks and recovery
  - Selective alerting to channels with severity/type filters

---

## APIs (indicative)

Registry API (Flask)
- POST /registry/bootstrap: guarded bootstrap to create the first agent/admin
- POST /auth/login: returns JWT for authenticated calls
- POST /registry/onboard: onboard/register agents
- POST /ir/propose: submit IR proposals for governance
- GET  /audit/logs: query audit events
- GET  /health: liveness check
- GET  /metrics: Prometheus exposition (if enabled)

Arena API (FastAPI)
- API key middleware via X-API-Key (if enabled)
- Endpoints for controlled graph execution and orchestration
- Health/metrics endpoints are recommended in managed deployments

Note: Exact routes may vary by version and deployment profile. Refer to your internal API reference for authoritative schemas and authentication flows.

---

## Security, privacy, and compliance

- Secrets: Never commit secrets to version control. Use a secret manager or encrypted env injection.
- Data handling: Audit logs and metrics may include metadata about workflows and agents. Configure retention and access controls per your policy.
- Network: Place services behind authenticated gateways. Use TLS/HTTPS in all environments outside local development.
- Least privilege: Restrict DB and Redis access to the minimal required scope. Rotate credentials periodically.
- Hardening: Enable rate limiting, input validation, and governance checks before executing externally controlled graphs.
- Responsible disclosure: Report suspected vulnerabilities to your Novatrax account team or the designated security contact in your agreement. Do not post vulnerabilities publicly.

---

## Deployment notes

- Production deployments are typically containerized and run behind an API gateway with centralized auth, logging, and metrics scrape.
- A single primary entry point is recommended per service image (avoid running multiple servers in the same container unless directed by Novatrax).
- Externalize configuration and secrets; disable debug/reload modes; enforce TLS and strict CORS as applicable.
- For horizontal scaling, use a durable store for governance and audit, and coordinate idempotent apply operations.

---

## Support and updates

- Enterprise customers should contact their Novatrax Labs account team for support, onboarding, and SLAs.
- New versions, patches, and security advisories are communicated via your agreed channel.
- Compatibility and migration guidance are provided with each release.

---

## Licensing

This software is proprietary and confidential to Novatrax Labs LTD. All rights reserved.

- No part of this software or documentation may be reproduced, distributed, or transmitted in any form or by any means without prior written permission from Novatrax.
- Use of this software is governed solely by your written agreement with Novatrax. In case of conflict between this README and your agreement, your agreement controls.
- Trademarks and service marks are the property of their respective owners.

---

## Acknowledgments

This product may integrate with or interoperate alongside third-party components and services under their respective terms. Such third-party terms do not grant any rights in Novatrax proprietary software.

---

## Feedback

For feature requests or questions, contact your Novatrax Labs representative. Please do not open public issues or discussions unless explicitly permitted in your agreement.
