# Graphix IR Quick Start (Windows + Git Bash)

**Audience:** Developers & evaluators needing a fast setup path  
**Python Targets:** 3.13 (core operations) + optional 3.11 (Ray compatibility)

---

## 0) Prerequisites

- Windows + Git Bash  
- Python 3.13 installed (primary)  
- Optional Python 3.11 for Ray features  
- `py` launcher available

---

## 1) Clone & Minimal `.env`

```bash
cd /d
git clone https://your-repo-url/Graphix.git D:/Graphix || true
cd D:/Graphix

grep -q GRAPHIX_API_KEY .env 2>/dev/null || cat > .env <<'ENV'
GRAPHIX_API_KEY=dev-local-key
DB_URI=sqlite:///graphix.db
ENV
```

---

## 2) Core Setup (Python 3.13 Recommended)

```bash
py -3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt optuna llm-compressor==0.7.0
export PYTHONPATH=.
```

> Ray will not install on 3.13—only use 3.11 virtualenv for Ray tests.

---

## 3) Smoke Test (≈10 min)

**A) Start Registry API**  
```bash
python app.py
```

**B) Interact (new terminal)**  
```bash
source .venv/Scripts/activate
export API_KEY=$(grep -E '^GRAPHIX_API_KEY=' .env | cut -d= -f2-)
curl http://localhost:5000/health
curl -X POST http://localhost:5000/registry/onboard \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"id":"test_agent","roles":["executor"]}'
```

**C) Evolution & Governance**  
```bash
python scripts/run_sentiment_tournament.py --mode offline --generations 3 --population 6
python governance_loop.py --equiv-check
pytest -q tests/graphix_conformance/test_conformance.py
```

**D) Photonic Dispatch (Emulated)**  
```bash
python -m src.hardware_dispatcher --test_mvm --provider Lightmatter --model tensor_core
```

---

## 4) Optional Ray Setup (Python 3.11)

```bash
py -3.11 -m venv .venv311
source .venv311/Scripts/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements-all.txt  # includes ray[default]
export PYTHONPATH=.

pytest -q -k "consensus or distributed_fabric"
```

---

## Troubleshooting

| Issue | Cause | Resolution |
|-------|-------|-----------|
| `ModuleNotFoundError: ray` | Ray absent in 3.13 env | Use `.venv311` or skip Ray tests |
| Unauthorized curl | Missing or wrong API key | Re-export `API_KEY`; check `.env` |
| Import failures | PYTHONPATH not set | `export PYTHONPATH=.` |
| Overgrowth of artifacts | Excess saved champions | Rotate daily; keep top K |
| Long photonic test | Large tensor shapes | Reduce input dims in test invocation |

---

## Key Paths & Files

| File | Role |
|------|------|
| `app.py` | Registry API (`http://localhost:5000`) |
| `graphix_arena.py` | Orchestration server |
| `governance_loop.py` | Proposal & motif mining |
| `scripts/run_sentiment_tournament.py` | Evolution tournaments |
| `tests/graphix_conformance/` | Gold graph conformance |
| `src/security_nodes.py` | Safety & compliance nodes |
| `src/scheduler_node.py` | Reactive scheduling |
| `src/explainability_node.py` | Execution explanations |
| `src/feedback_protocol.py` | RLHF feedback endpoints |

---

## One-Liner Demo (Core)

```bash
source .venv/Scripts/activate && python app.py &
sleep 2
export API_KEY=$(grep -E '^GRAPHIX_API_KEY=' .env | cut -d= -f2-)
curl http://localhost:5000/health && \
python scripts/run_sentiment_tournament.py --mode offline --generations 3 --population 6 && \
python governance_loop.py --equiv-check && \
pytest -q tests/graphix_conformance/test_conformance.py
```

---

> For production evaluation: DO NOT reuse dev `.env`; adopt secret manager & hardened auth policies.
