# Easy Demo Guide (Graphix IR & VULCAN-AI) – No Coding Experience Needed

**Goal:** Run a simple demo showing AI graph creation, improvement, safety checks, and visualization.

**Time:** ~30–35 minutes (setup + run)

---

## 1. What You’ll See

- AI builds and runs a “graph” (flowchart-like)
- Improves it over generations (evolution)
- Shows efficiency (simulated photonic hardware energy savings)
- Generates a visual map of how it “connects ideas”
- Confirms safety & correctness through validation tests

---

## 2. Requirements

- Windows computer
- ~2 GB free space on `D:` drive
- Internet access
- Git Bash & Python 3.10.11

---

## 3. Install Git Bash (If Missing)

Visit: https://git-scm.com/downloads  
Install with default options. Open “Git Bash” from Start Menu (black terminal window).

---

## 4. Download Project Files

In Git Bash:

```bash
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM
```

---

## 5. Create Secrets File (`.env`)

```bash
cat > .env <<'ENV'
GRAPHIX_API_KEY=demo-local-key
DB_URI=sqlite:///graphix_registry.db
GROK_API_KEY=sk-mock-grok-key
LIGHTMATTER_API_KEY=mock-lightmatter-key
SLACK_BOT_TOKEN=xoxb-mock-slack-token
SLACK_ALERT_CHANNEL=#demo-alerts
ENV
```

---

## 6. Install Python 3.10.11

Download from: https://www.python.org/downloads/release/python-31011/  
Check “Add Python to PATH” during installation.

---

## 7. Install Demo Tools

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -r requirements-dev.txt
export PYTHONPATH=.
```

> This takes several minutes.

---

## 8. Register an AI “Agent”

```bash
python src/setup_agent.py validation_agent executor validator
```

---

## 9. Start Services (Two New Git Bash Windows)

**Window 1: Registry**

```bash
cd VulcanAMI_LLM
source .venv/Scripts/activate
python app.py
```

**Window 2: Arena**

```bash
cd VulcanAMI_LLM
source .venv/Scripts/activate
uvicorn src.graphix_arena:app --reload
```

---

## 10. Run Demo (Third Window)

Activate environment:

```bash
cd VulcanAMI_LLM
source .venv/Scripts/activate
export API_KEY=$(grep -E '^GRAPHIX_API_KEY=' .env | cut -d= -f2-)
```

### A. Generate & Run a Graph

```bash
curl -X POST http://127.0.0.1:8000/api/run/generator \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"spec_id":"sentiment_3d_spec","parameters":{}}'
```

Expected: JSON with sentiment score + “energy” (simulated).

### B. Give Feedback

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"proposal_id":"sentiment_3d_v1","score":0.9,"rationale":"Good job!"}'
```

### C. Watch Evolution

```bash
python scripts/run_sentiment_tournament.py --mode offline --generations 3 --population 6
python src/governance_loop.py --equiv-check --grammar 1.3.1
```

Expected: Generation scores improving.

### D. Verify Safety & Functionality

```bash
pytest src/run_validation_test.py -v
```

Expected: Tests show “ok”.

### E. Visualize “Thinking”

Install extra tool & generate map:

```bash
pip install graphviz
cat > demo_plot.py <<'EOF'
from src.observability_manager import ObservabilityManager
import numpy as np
obs = ObservabilityManager()
attention = np.random.rand(5,5)
img = obs.plot_semantic_map(attention, labels=['Feeling','Words','AI Think','Rules','Output'])
print(f"Saved semantic map: {img}")
EOF
python demo_plot.py
```

Open generated PNG in `observability_logs/`.

---

## 11. Why This Demo Matters

| Concept | Value |
|---------|-------|
| Self-Improvement | Shows evolution loop |
| Sustainability | Simulated low energy ops |
| Transparency | Visual map + safety tests |
| Alignment | Feedback influences future graphs |

---

## 12. Troubleshooting

| Problem | Fix |
|---------|-----|
| No output from curl | Ensure services running; test `curl http://localhost:5000/health` |
| Python error | Confirm Python 3.10.11 installed correctly |
| Missing dependency | Re-run `pip install -r requirements.txt` |
| Graph not evolving | Increase generations or population size |
| Visual script missing file | Ensure `observability_logs/` folder exists or created on first run |

---

> This demo uses mock keys and simulated hardware metrics—safe to share locally. For production previews, replace secrets and enable real dispatch.
