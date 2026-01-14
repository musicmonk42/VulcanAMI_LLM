# 🌐 Graphix Arena & VULCAN-AI – Quick Start & Interaction Guide

**Version:** 2.2.0

---

## 1. Purpose

Graphix Arena is the FastAPI-based coordination surface for the VULCAN-AI cognitive architecture:
- Serves agent endpoints
- Receives proposals & feedback
- Exposes metrics & evolution hooks
- Bridges governance + execution

---

## 2. Installation

```bash
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
export PYTHONPATH=.

cat > .env <<'ENV'
GRAPHIX_API_KEY=dev-local-key
DB_URI=sqlite:///graphix_registry.db
GROK_API_KEY=sk-your-grok-key
LIGHTMATTER_API_KEY=your-lightmatter-key
SLACK_BOT_TOKEN=xoxb-your-slack-token
SLACK_ALERT_CHANNEL=#graphix-alerts
ENV

python src/setup_agent.py validation_agent executor validator
```

Start services:

```bash
python app.py # Registry (http://localhost:5000)
uvicorn src.graphix_arena:app --reload # Arena (http://127.0.0.1:8000)
```

---

## 3. Authentication

All Arena endpoints require `X-API-Key` header with `GRAPHIX_API_KEY`.

---

## 4. Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/run/{agent_id}` | Execute task via specified agent (e.g. generator, evolver) |
| POST | `/api/feedback` | Submit performance & alignment feedback (RLHF trigger) |
| POST | `/api/tournament` | Run tournament selection among proposals |
| GET | `/api/metrics` | Prometheus-formatted metrics |
| (Future) | `/api/proposals` | List or query proposals |
| (Future) | `/api/evolution/status` | Evolution cycle progress snapshot |

---

### Example: Generate a Graph

```bash
curl -X POST http://127.0.0.1:8000/api/run/generator \
 -H "Content-Type: application/json" \
 -H "X-API-Key: YOUR_KEY" \
 -d '{"spec_id":"sentiment_3d_spec","parameters":{"goal":"Photonic sentiment analysis"}}'
```

### Example: Submit Feedback

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
 -H "Content-Type: application/json" \
 -H "X-API-Key: YOUR_KEY" \
 -d '{"proposal_id":"initial_graph_v1","score":0.92,"rationale":"Accurate, low latency"}'
```

### Example: Tournament

```bash
curl -X POST http://127.0.0.1:8000/api/tournament \
 -H "Content-Type: application/json" \
 -H "X-API-Key: YOUR_KEY" \
 -d '{"proposals":[{"id":"p1"},{"id":"p2"}],"fitness":[0.88,0.92]}'
```

---

## 5. 🚀 One-Button Demo Orchestrator

For a complete end-to-end demonstration, use the demo orchestrator script that executes the Four Acts workflow in ~90 seconds.

### Against Unified Platform

```bash
# Start unified platform (default configuration)
python src/full_platform.py

# Or with explicit settings
# python -m uvicorn src.full_platform:app --host 0.0.0.0 --port 8000

# Run orchestrator (in separate terminal)
python scripts/demo_orchestrator.py
```

### Against Standalone Arena + VULCAN

```bash
# Terminal 1: Start Arena
uvicorn src.graphix_arena:app --host 127.0.0.1 --port 8000

# Terminal 2: Start VULCAN (if separate)
python -m src.vulcan.main --port 8001

# Terminal 3: Run orchestrator with environment variables
ARENA_BASE=http://127.0.0.1:8000 \
VULCAN_BASE=http://127.0.0.1:8001 \
API_KEY=your-api-key \
python scripts/demo_orchestrator.py
```

### What It Demonstrates

1. **Act 1**: Submit graph generation request with simulated high latency
2. **Act 2**: Provide negative feedback (score 0.1) to trigger optimization
3. **Act 3**: Monitor VULCAN mind stream via SSE for ~10 seconds
4. **Act 4**: Re-run generator (faster) and query Prometheus metrics

The orchestrator includes:
- Automatic retry logic with exponential backoff
- Graceful error handling for missing services
- Deterministic behavior via `DEMO_SEED` environment variable
- Metrics summarization showing before/after comparison

---

## 6. 📡 Monitoring with SSE Mind Stream

VULCAN exposes real-time cognitive activity via Server-Sent Events at `/vulcan/v1/stream` (or `/v1/stream` for standalone VULCAN).

### Browser-Based Viewer

Open `demos/sse_mind.html` in a browser to visualize the stream:

```bash
# Configure base URL via query parameter
open "demos/sse_mind.html?base=http://0.0.0.0:8000/vulcan"

# Or edit the base URL directly in the UI
```

The viewer shows:
- Real-time event stream with timestamps
- Event counter and connection status
- Auto-scroll capability
- Last 100 events retained

### Command-Line Monitoring

```bash
# Using curl (basic)
curl -N http://0.0.0.0:8000/vulcan/v1/stream

# Or use the orchestrator's Act 3 for parsed output
python scripts/demo_orchestrator.py
```

---

## 7. Workflow Walkthrough

1. **Generation:** `/api/run/generator` emits baseline graph.
2. **Execution:** Results processed; metrics/logs emitted.
3. **Feedback:** `/api/feedback` influences internal optimization weighting.
4. **Evolution:** Tournament selects champions; governance loop consolidates motifs.
5. **Validation:** Safety & ethics enforced; failed variants quarantined.

---

## 8. Agent Catalog (Typical)

| Agent | Role |
|-------|------|
| generator | Creates initial or variant graphs |
| evolver | Applies mutation & fitness evaluation |
| visualizer | Generates semantic maps & explanations |
| photonic_optimizer | Tweaks analog parameters (noise/bandwidth/compression) |
| automl_optimizer | Hyperparameter or structural search |

---

## 9. Metrics Overview

Scrape via `/api/metrics`. Key exposure:
- Success/failure counts
- Cache hits/misses
- Safety blocks
- Consensus latency
- Energy estimates

Integrate into Prometheus: see `visualization_guide.md`.

---

## 10. Security & Governance Notes

| Concern | Handling |
|---------|----------|
| Risky Code Patterns | NSOAligner pre-exec scan |
| Proposal Replay | Hash + embedding similarity gating |
| Unauthorized Access | API key header; add rate limit in production |
| Audit Integrity | Hash chaining with periodic verification |

---

## 11. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Missing/invalid X-API-Key | Verify `.env` key & header inclusion |
| Empty metrics | Service not scraped | Confirm `/api/metrics` accessible & Prometheus target UP |
| Stalled evolution | Insufficient fitness delta | Adjust mutation diversity or scoring function |
| Recurring safety blocks | Risk pattern proliferation | Tighten NSO rules; inspect mutation operators |

---

## 12. Future Endpoint Ideas

| Endpoint | Purpose |
|----------|---------|
| `/api/alignment/score` | Real-time alignment confidence |
| `/api/proposals/{id}` | Detailed proposal lineage & risk vector |
| `/api/hardware/state` | Current backend health summary |
| `/api/explain/{graph_id}` | On-demand multi-level explanation generation |

---

> Keep evolution loop binding: safety & alignment must always precede application, never inverted for performance expediency.
