# Graphix IR Quick Start (Windows + Git Bash)

**Audience:** Developers & evaluators needing a fast setup path  
**Python Targets:** 3.11 or 3.12 (recommended)

---

## 0) Prerequisites

- Windows + Git Bash  
- Python 3.11 or 3.12 installed  
- `py` launcher available (comes with Python for Windows)

---

## 1) Clone & Minimal `.env`

```bash
# Clone the repository (adjust path as needed for your setup)
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

# Create minimal .env file if it doesn't exist
grep -q GRAPHIX_API_KEY .env 2>/dev/null || cat > .env <<'ENV'
GRAPHIX_API_KEY=dev-local-key
DB_URI=sqlite:///graphix.db
ENV
```

---

## 2) Core Setup

```bash
# Create virtual environment (use 3.11 or 3.12)
py -3.12 -m venv .venv
source .venv/Scripts/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# For development/testing tools
pip install -r requirements-dev.txt

# Set Python path
export PYTHONPATH=.
```

> **Note:** The project requires Python >=3.11. Use Python 3.11 or 3.12 for best compatibility.

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
python src/governance_loop.py --equiv-check
pytest -q tests/test_graph_validation.py
```

**D) Photonic Dispatch (Emulated)**  
```bash
python -m src.hardware_dispatcher --test_mvm --provider Lightmatter --model tensor_core
```

---

## 4) Running Tests

```bash
# Ensure virtual environment is activated
source .venv/Scripts/activate
export PYTHONPATH=.

# Run specific test suites
pytest -q -k "consensus" tests/
pytest -q tests/test_graphix_arena.py
pytest -q tests/test_governance_loop.py
```

---

## Troubleshooting

| Issue | Cause | Resolution |
|-------|-------|-----------|
| `ModuleNotFoundError` | Missing dependencies | Run `pip install -r requirements.txt` |
| Unauthorized curl | Missing or wrong API key | Re-export `API_KEY`; check `.env` |
| Import failures | PYTHONPATH not set | `export PYTHONPATH=.` |
| File not found errors | Wrong directory paths | Ensure files are referenced with correct `src/` prefix |
| Test failures | Outdated test references | Use existing test files like `test_graph_validation.py` |

---

## Key Paths & Files

| File | Role |
|------|------|
| `app.py` | Registry API (`http://localhost:5000`) |
| `src/graphix_arena.py` | Orchestration server |
| `src/governance_loop.py` | Proposal & motif mining |
| `scripts/run_sentiment_tournament.py` | Evolution tournaments |
| `tests/test_graph_validation.py` | Graph validation tests |
| `src/security_nodes.py` | Safety & compliance nodes |
| `src/scheduler_node.py` | Reactive scheduling |
| `src/explainability_node.py` | Execution explanations |
| `src/feedback_protocol.py` | RLHF feedback endpoints |

---

## One-Liner Demo (Core)

```bash
source .venv/Scripts/activate && export PYTHONPATH=. && python app.py &
sleep 2
export API_KEY=$(grep -E '^GRAPHIX_API_KEY=' .env | cut -d= -f2-)
curl http://localhost:5000/health && \
python scripts/run_sentiment_tournament.py --mode offline --generations 3 --population 6 && \
python src/governance_loop.py --equiv-check && \
pytest -q tests/test_graph_validation.py
```

---

> For production evaluation: DO NOT reuse dev `.env`; adopt secret manager & hardened auth policies.
