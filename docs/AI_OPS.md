# Graphix & VULCAN-AI: AI-Native Operations Guide

**Version:** 2.2.0 
**Last Updated:** December 23, 2024 
**Authors:** Core Engineering Collective (musicmonk42, contributors)

> See also: [OPERATIONS.md](OPERATIONS.md) for comprehensive operations guide

---

## 1. Overview

Graphix is an AI-native execution & evolution fabric for JSON-based directed graphs (“Graphix IR”). It powers VULCAN-AI: a cognitive architecture spanning perception, reasoning, optimization, alignment, safety, and hardware-aware dispatch. This guide covers day‑2 operational concerns: provisioning, runtime management, safety/governance hooks, continuous integration, and observability.

---

## 2. Capability Snapshot

| Capability | Summary | Operational Notes |
|------------|---------|-------------------|
| Graph Execution | Multi-mode (sequential, parallel, streaming, batch) with deterministic caching | Tune `max_parallel_tasks` & timeout configs per environment |
| Self-Evolution | Proposal lifecycle: submit → validate → consensus → apply → observe | Enforce replay window & similarity dampening for proposal hygiene |
| Hardware Acceleration | Photonic & memristor emulation + backend strategy selection | Fallback chain: Real → Emulated → CPU; monitor energy_nj and latency |
| Ethical Governance | Multi-model audit (LLM consensus), risky pattern removal | NSOAligner flags: eval/exec, path traversal, bias taxonomy |
| Observability | Prometheus/Grafana integration, structured audit chain | Enable metrics export only behind internal gateway |
| Testing & QA | Pytest suites (validation, hardware emulation, stress, E2E) | Parallelize with `pytest -n auto`; tag slow vs fast tests |
| Dependency Surface | Core: Python 3.11+, numpy, networkx; Optional: torch, optuna, ray, vllm | Pin high-risk libs; generate SBOM for release artifacts |

---

## 3. Environment & Security Baselines

| Concern | Baseline | Hardening Recommendation |
|---------|----------|--------------------------|
| Secrets | `.env` local dev only | Migrate to Vault/KMS; rotate monthly |
| API Keys | Mock defaults permitted | Require key existence in production startup hook |
| I/O Governance | Stdout/stderr audited | Enforce `stdio_policy` checks pre-release |
| Replay Protection | Hash + TTL window | Expand rejection reason metrics to dashboard |
| Dependency Integrity | Requirements pinned | Add signature or checksum validation pipeline |

---

## 4. Getting Started (Dev Workstation / Windows + Git Bash)

```bash
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
export PYTHONPATH=.
```

Create `.env` (sample — replace with real secrets):

```bash
cat > .env <<'ENV'
GRAPHIX_API_KEY=dev-local-key
DB_URI=sqlite:///graphix_registry.db
GROK_API_KEY=sk-mock-grok-key
LIGHTMATTER_API_KEY=mock-lightmatter-key
SLACK_BOT_TOKEN=xoxb-mock-slack-token
SLACK_ALERT_CHANNEL=#graphix-alerts
ENV
```

Bootstrap agent keys:

```bash
python src/setup_agent.py validation_agent executor validator
```

Run services:

```bash
# Registry API
python app.py

# Arena (separate terminal)
uvicorn src.graphix_arena:app --reload
```

---

## 5. Smoke / Confidence Checks

| Action | Command | Expected |
|--------|---------|----------|
| Validation Test | `pytest -q src/run_validation_test.py` | PASS set of schema + ethics + execution |
| Evolution Mini-Tournament | `python scripts/run_sentiment_tournament.py --mode offline --generations 3` | Progressive fitness improvements |
| Photonic Fallback | `python -m src.hardware_dispatcher --test_mvm --provider Lightmatter` | Emulation success or graceful fallback |
| Metrics Health | Curl `/api/metrics` (Arena) | Non-empty Prometheus payload |

---

## 6. Operational Metrics (Key Selection)

| Metric | Description | Threshold / Alert |
|--------|-------------|------------------|
| `graphix_success_rate` | Graph-level success fraction | < 0.90 triggers investigation |
| `graphix_cache_hit_rate` | Deterministic node cache efficiency | Deterioration slope > 20% |
| `graphix_vulcan_safety_blocks_total` | Count of safety rejections | Sudden spike: security review |
| `graphix_energy_nj_total` | Cumulative estimated energy | Anomaly > x3 baseline → dispatch tuning |
| `graphix_nodes_failed` | Failed node executions | Persistent growth → handler audit |

---

## 7. Governance / Evolution Flow

1. Proposal submission (`ProposalNode` or API) 
2. Validator pipeline (structure → ontology → semantics → security → alignment → safety) 
3. Consensus threshold check (trust-weighted) 
4. Apply & hash chain update 
5. Post-run observation & pattern mining 
6. Autonomous optimizer may propose refinement if fitness < threshold

Ensure all proposals log fingerprint + risk vector; assert uniqueness inside replay window (configurable TTL).

---

## 8. Backups & Recovery

| Artifact | Location | Strategy |
|----------|----------|----------|
| Audit DB | `audit.db` (dev) | Daily snapshot → integrity hash chain |
| Champions | `evolution_champions/` | Retain top N per day; prune by age |
| Governance Artifacts | `governance_artifacts/` | Keep all signed proposals (immutable store) |
| Learned Patterns | `learned_subgraphs/` | Versioned, LRU eviction for stale low-confidence |

Recovery: restore DB snapshot → re-verify audit hashes → replay last applied proposals if necessary.

---

## 9. Hardening Roadmap

| Phase | Focus | Outcome |
|-------|-------|--------|
| A | Secret isolation | Vault-managed runtime injection |
| B | Policy DSL | Declarative gating (unsafe node constraints) |
| C | Distributed Execution | Sharding with provenance correlation |
| D | Formal Invariants | TLA+/Alloy spec for essential safety properties |
| E | Real Hardware | Live photonic/memristor performance integration |

---

## 10. Troubleshooting Quick Table

| Symptom | Likely Cause | Remediation |
|---------|--------------|------------|
| Endless pending nodes | Missing upstream outputs / cycle | Run validator; inspect audit log chain |
| High latency spike | Hardware fallback thrash | Inspect dispatch strategy, reduce concurrency |
| Repeated safety blocks | Proposal spam / alignment drift | Increase similarity threshold; audit NSOAligner weights |
| Cache hit collapse | Node determinism misflagged | Add `is_deterministic` param or adjust handler purity |
| Arena 500 errors | Missing env / provider creds | Verify `.env`, run health endpoint checks |

---

## 11. Security Hygiene

- Never commit real API keys. 
- Use mock placeholders for demos. 
- Prevent unreviewed ontology or grammar edits by gating branch protections. 
- Audit chain mismatch → immediate containment.

---

## 12. Reference Cross-Links

| Domain | File |
|--------|------|
| Evolution | `docs/evolution_roadmap.md` |
| Training | `docs/AI_TRAINING_GUIDE.md` |
| Transparency | `docs/transparency_report.md` |
| Visualization | `docs/visualization_guide.md` |

---

## 13. Appendix: Safe .env Template

```dotenv
GRAPHIX_API_KEY=replace-me
DB_URI=sqlite:///graphix_registry.db
GROK_API_KEY=replace-me
LIGHTMATTER_API_KEY=replace-me
SLACK_BOT_TOKEN=replace-me
SLACK_ALERT_CHANNEL=#graphix-alerts
```

---

> Operate defensively: treat all evolution events as potentially adversarial until validated.
