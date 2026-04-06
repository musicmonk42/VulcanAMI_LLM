<div align="center">

# Vulcan AMI

**Adaptive Machine Intelligence — Multi-Modal Reasoning with Safety Governance**

[![CI](https://github.com/musicmonk42/VulcanAMI_LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/musicmonk42/VulcanAMI_LLM/actions/workflows/ci.yml)
[![Security](https://github.com/musicmonk42/VulcanAMI_LLM/actions/workflows/security.yml/badge.svg)](https://github.com/musicmonk42/VulcanAMI_LLM/actions/workflows/security.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/)

*Created by Brian D. Anderson — Novatrax Labs LTD*

</div>

---

## Table of Contents

- [Mission](#mission)
- [What Vulcan Does](#what-vulcan-does)
- [Architecture](#architecture)
- [Reasoning Pipeline](#reasoning-pipeline)
- [Safety & Governance](#safety--governance)
- [Key Subsystems](#key-subsystems)
- [Maturity Status](#maturity-status)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Testing](#testing)
- [Documentation Index](#documentation-index)
- [Security](#security)
- [License](#license)

---

## Mission

Vulcan bridges the gap between symbolic reasoning and neural AI by providing a safe, auditable execution fabric for autonomous intelligence.

It is not an LLM wrapper. It is a **cognitive architecture** that coordinates multiple reasoning engines — causal, probabilistic, symbolic, analogical, philosophical — under a unified safety governance layer, with cryptographic accountability at every step.

---

## What Vulcan Does

**For AI researchers**: A platform for composing heterogeneous reasoning systems with formal safety constraints and trust-weighted governance.

**For engineers**: A modular Python framework with FastAPI endpoints, Prometheus observability, Docker/K8s deployment, and a graph execution engine with content-addressable storage.

**For organizations**: An auditable AI system where every decision is traceable to its reasoning path, every change requires consensus, and every data removal is cryptographically verifiable.

### Core Capabilities

| Capability | Description | Status |
|-----------|-------------|--------|
| Multi-modal reasoning | Causal, probabilistic, symbolic, analogical, philosophical engines | Production |
| Safety governance | Multi-layer validation, trust-weighted consensus, ethical boundaries | Production |
| Graph execution | Typed DAG with validation pipeline, concurrent execution, policy hooks | Production |
| Tool selection | ML-based selection (LightGBM, Bayesian priors, isotonic calibration) | Production |
| Cryptographic audit | Tamper-evident hash chains, ZK-SNARK proofs (Groth16/bn128) | Production |
| Memory hierarchy | Runtime + persistent (S3-backed LSM tree) + governed unlearning | Implemented |
| Self-improvement | LLM-driven code generation with AST validation and governed pipeline | Experimental |
| Hardware abstraction | CPU/GPU dispatch with photonic/memristor/quantum emulation layer | Scaffolding |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│          Unified Platform · Health · Auth · Docs             │
├─────────────────────────────────────────────────────────────┤
│                  Safety & Governance                        │
│   Consensus Protocol · Safety Validator · Audit Logger      │
│   Trust-Weighted Voting · Ethical Boundaries · Alignment    │
├─────────────────────────────────────────────────────────────┤
│               Reasoning & Orchestration                     │
│  ┌────────────┬────────────┬─────────────┬───────────────┐  │
│  │   World    │  Unified   │    Tool     │    Agent      │  │
│  │   Model    │  Reasoner  │  Selector   │    Pool       │  │
│  │ (request   │ (strategy  │ (ML-based   │ (lifecycle,   │  │
│  │  dispatch) │  planning) │  selection) │  execution)   │  │
│  └────────────┴────────────┴─────────────┴───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Reasoning Engines                            │
│  Causal · Probabilistic · Symbolic · Analogical             │
│  Mathematical · Philosophical · Creative · Multi-Modal      │
├─────────────────────────────────────────────────────────────┤
│              Graph Execution (gvulcan)                       │
│  Content-Addressable Storage · Merkle Trees · Bloom Filters │
│  Packfiles · S3 Backend · Milvus Vector Store · ZK Proofs   │
├─────────────────────────────────────────────────────────────┤
│             Memory & Persistence                            │
│  Hierarchical Memory · LSM Tree · Graph RAG                 │
│  Governed Unlearning · Cost Optimization                    │
├─────────────────────────────────────────────────────────────┤
│            Observability & Infrastructure                   │
│  Prometheus · Grafana · Docker/K8s · Helm · Redis · SQLite  │
└─────────────────────────────────────────────────────────────┘
```

---

## Reasoning Pipeline

When a query arrives, Vulcan processes it through a structured pipeline:

```
Query → Classification → Strategy Selection → Tool Selection → Engine Execution → Safety Validation → Response
```

1. **Classification** — The World Model categorizes the query (reasoning, knowledge, creative, ethical, conversational)
2. **Strategy** — The Unified Reasoner selects a strategy (portfolio, parallel, ensemble, adaptive, hierarchical)
3. **Tool Selection** — ML-based scoring with LightGBM cost prediction, Bayesian memory priors, and confidence calibration
4. **Execution** — The selected reasoning engine processes the query (causal inference, symbolic logic, probabilistic reasoning, etc.)
5. **Safety** — Results pass through pattern-based validation, ethical boundary checks, and governance alignment
6. **Formatting** — LLM guidance formats verified content for the user

---

## Safety & Governance

Safety is not bolted on — it's load-bearing at every layer.

| Layer | What It Does |
|-------|-------------|
| **Pattern Validation** | Detects eval/exec/subprocess injection, path traversal, and code execution attempts |
| **Consensus Protocol** | Trust-weighted voting with 51% quorum, 66% approval threshold. Agents have configurable trust weights |
| **Governance Alignment** | Six levels (AUTONOMOUS → EMERGENCY) with stakeholder types (USER, OPERATOR, SAFETY_OFFICER) |
| **Ethical Boundaries** | Configurable constraints for ethical dilemma detection, principle extraction, and conflict resolution |
| **Audit Trail** | Tamper-evident hash chains with JSONL storage and optional SQLite indexing |
| **Secret Scanning** | Pre-commit hook (detect-secrets) blocks credential commits |

**Canonical protocols** (`src/protocols/`):
- [`ConsensusProtocol`](src/protocols/consensus.py) — Trust-weighted voting, leader election, proposal lifecycle
- [`AuditProtocol`](src/protocols/audit.py) — Tamper-evident logging, hash chain verification, field redaction
- [`SafetyProtocol`](src/protocols/safety.py) — Pattern matching, blacklist/whitelist, severity-tiered validation

---

## Key Subsystems

### Graph Execution Engine (`src/gvulcan/`)
Git-like content-addressable storage with packfiles, Merkle trees, Bloom filters, CRC32C integrity. S3-backed with local cache, compaction policies, and Milvus vector store integration. Includes a genuine **Groth16 ZK-SNARK** implementation using `py_ecc.bn128` elliptic curve pairings for cryptographic verification of data unlearning.

### Curiosity Engine (`src/vulcan/curiosity_engine/`)
Autonomous knowledge-gap detection with experiment generation, iterative design, dynamic budgeting, resource monitoring, and cycle-aware dependency graphs. Drives self-directed exploration.

### Semantic Bridge (`src/vulcan/semantic_bridge/`)
Cross-domain concept mapping with transfer engine, conflict resolution, domain registry, and caching. Enables reasoning transfer across knowledge domains.

### Persistent Memory (`src/persistant_memory_v46/`)
S3-backed durable storage with LSM-tree indexing, Graph-RAG retrieval, governed unlearning (GDPR-compliant data removal), and zero-knowledge proof verification.

---

## Maturity Status

> Transparency about what works, what's unproven, and what's scaffolding.

| Component | Lines | Tests | Maturity |
|-----------|------:|------:|----------|
| Reasoning pipeline (World Model, Reasoner, Tool Selector) | ~15K | 30+ | **Production** — working end-to-end |
| Safety & governance (protocols, validators, consensus) | ~5K | 40+ | **Production** — multi-layer, tested |
| Graph execution engine (gvulcan) | ~8K | 20+ | **Production** — content-addressable storage |
| Platform & API (FastAPI, auth, routes) | ~6K | 15+ | **Production** — deployed on Railway |
| Memory hierarchy (3 subsystems) | ~4K | 10+ | **Implemented** — consistency model TBD |
| Self-improvement pipeline | ~1K | 5+ | **Experimental** — approval gate incomplete |
| Hardware dispatch (CPU/GPU/photonic/quantum) | ~2K | 5+ | **Scaffolding** — real dispatch, emulated backends |
| Curiosity engine | ~3K | 5+ | **Experimental** — needs production validation |

---

## Quick Start

> **New to the project?** See [docs/NEW_ENGINEER_SETUP.md](docs/NEW_ENGINEER_SETUP.md) for detailed setup.
> **Quick reference:** See [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) for common commands.

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Redis (optional, for rate limiting)

### Setup

```bash
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   JWT_SECRET_KEY=<strong-unique-secret>
#   VULCAN_ENV=development  (or 'production' with full auth configured)
```

> **Security**: In production, auth is fail-closed. If no JWT_SECRET or API_KEY is configured and VULCAN_ENV is not `development` or `test`, the platform refuses to start.

### Run

```bash
# Unified Platform (production server)
python -m src.full_platform

# Interactive CLI
python -m vulcan.cli

# Minimal executor demo
python src/minimal_executor.py
```

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `JWT_SECRET_KEY` | Yes (prod) | JWT signing secret |
| `API_KEY` | Yes (prod) | API key for endpoint auth |
| `VULCAN_ENV` | No | `production` (default), `development`, `test` |
| `REDIS_URL` | No | Redis for rate limiting (fallback: in-memory) |
| `AUDIT_DB_PATH` | No | SQLite audit database path |
| `SLACK_WEBHOOK_URL` | No | Security alert webhook |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference.

---

## Deployment

| Method | Guide |
|--------|-------|
| Docker Compose | `docker compose up -d` |
| Kubernetes (Kustomize) | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Helm Charts | [docs/HELM_DEPLOYMENT.md](docs/HELM_DEPLOYMENT.md) |
| Azure AKS | [docs/AZURE_SETUP_GUIDE.md](docs/AZURE_SETUP_GUIDE.md) |
| Railway | [docs/RAILWAY_ENV_VARS.md](docs/RAILWAY_ENV_VARS.md) |

CI/CD workflows: [`ci.yml`](.github/workflows/ci.yml), [`deploy.yml`](.github/workflows/deploy.yml), [`security.yml`](.github/workflows/security.yml), [`docker.yml`](.github/workflows/docker.yml)

---

## Testing

```bash
# Full test suite
pytest tests/ src/vulcan/tests/ -x --tb=short

# Safety-specific tests
pytest tests/test_safety_edge_cases.py tests/test_auth_edge_cases.py tests/test_consensus_edge_cases.py -v

# Protocol tests
pytest tests/test_consensus_protocol.py tests/test_audit_protocol.py tests/test_safety_protocol.py -v

# Static analysis
pytest tests/test_no_hardcoded_secrets.py tests/test_no_nested_ternaries.py -v
```

See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for the complete testing documentation.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md) | Full system architecture deep dive |
| [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | API reference |
| [SECURITY.md](docs/SECURITY.md) | Security model and practices |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |
| [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) | Testing documentation |
| [CI_CD.md](docs/CI_CD.md) | CI/CD pipeline reference |
| [CLI_USAGE.md](docs/CLI_USAGE.md) | Interactive CLI guide |
| [NEW_ENGINEER_SETUP.md](docs/NEW_ENGINEER_SETUP.md) | Onboarding guide |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Common commands |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Full configuration reference |
| [INDEX.md](docs/INDEX.md) | Complete documentation index |

---

## Security

- **Fail-closed authentication**: Production requires configured JWT or API key credentials
- **Secret scanning**: Pre-commit hook via [detect-secrets](https://github.com/Yelp/detect-secrets) blocks credential commits
- **Tamper-evident audit**: Hash-chained logging with integrity verification
- **Rate limiting**: Redis-backed throttling with DDoS protection
- **Input validation**: Pattern-based detection of injection and code execution attempts

**Reporting vulnerabilities**: Contact your Novatrax Labs representative or the designated security contact. See [docs/SECURITY.md](docs/SECURITY.md).

---

## License

Copyright (C) 2026 Brian D. Anderson and Novatrax Labs LTD

Licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See [LICENSE](LICENSE) for the full text.

---

<div align="center">

*Vulcan AMI — Trust. Transparency. Adaptability.*

</div>
