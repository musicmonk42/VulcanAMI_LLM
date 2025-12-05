# 🌈 Definitive Demo Guide – Graphix IR & VULCAN-AI

**Version:** 2.2.0  
**Date:** 2025-11-11

---

## 1. Demo Purpose

Showcase:
- Graph generation & execution
- Evolutionary improvement and governance
- Hardware-aware (emulated or real) performance
- Safety & alignment checks
- Explainability + metrics visualization

Two paths:
1. **Emulator-Based Demo** – No real hardware required.
2. **Real Hardware Demo** – Requires valid photonic API key (if available).

---

## 2. Common Setup

```bash
# Clone the repository
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

# Create virtual environment (Python 3.11 or 3.12)
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
export PYTHONPATH=.
```

Sample `.env`:

```bash
cat > .env <<'ENV'
GRAPHIX_API_KEY=demo-key
DB_URI=sqlite:///graphix_registry.db
SLACK_BOT_TOKEN=xoxb-your-slack-token
SLACK_ALERT_CHANNEL=#demo-alerts
GROK_API_KEY=sk-demo-grok
LIGHTMATTER_API_KEY=your-lightmatter-key
ENV
```

Register validation agent:

```bash
python src/setup_agent.py validation_agent executor validator
```

Run services:

```bash
# Registry API
python app.py

# Arena
uvicorn src.graphix_arena:app --reload
```

---

## 3. Demo Path A: Emulator-Based

```bash
source .venv/Scripts/activate
python - <<'PY'
from src.hardware_dispatcher import HardwareDispatcher
from src.unified_runtime import UnifiedRuntime
import numpy as np, asyncio, json

async def run_demo():
    print('--- Emulator Demo ---')
    graph = {
        'id':'photonic_demo_emulated',
        'type':'Graph',
        'nodes':[
            {'id':'mat','type':'CONST','params':{'value':np.random.rand(4,4).tolist()}},
            {'id':'vec','type':'CONST','params':{'value':np.random.rand(4,1).tolist()}},
            {'id':'mvm','type':'PhotonicMVMNode','params':{'photonic_params':{
                'noise_std':0.01,'compression':'ITU-F.748-quantized','bandwidth_ghz':100,'latency_ps':50
            }}}
        ],
        'edges':[
            {'from':{'node':'mat','port':'value'},'to':{'node':'mvm','port':'matrix'}},
            {'from':{'node':'vec','port':'value'},'to':{'node':'mvm','port':'vector'}}
        ]
    }
    runtime = UnifiedRuntime()
    runtime.hardware_dispatcher = HardwareDispatcher(use_mock=True)
    result = await runtime.execute_graph(graph)
    print(json.dumps(result, indent=2)[:400], '...')
asyncio.run(run_demo())
PY
```

**Talking Points:**  
- Emulated photonic operation with noise injection  
- Deterministic fallback chain if hardware inaccessible  
- Output tensor shows simulated analog variability

---

## 4. Demo Path B: Real Hardware (If Available)

Requirements: Valid `LIGHTMATTER_API_KEY`.

```bash
source .venv/Scripts/activate
python - <<'PY'
from src.hardware_dispatcher import HardwareDispatcher
from src.unified_runtime import UnifiedRuntime
import numpy as np, asyncio, json

async def run_demo():
    print('--- Real Hardware Demo ---')
    graph = {
        'id':'photonic_demo_real',
        'type':'Graph',
        'nodes':[
            {'id':'mat','type':'CONST','params':{'value':np.random.rand(4,4).tolist()}},
            {'id':'vec','type':'CONST','params':{'value':np.random.rand(4,1).tolist()}},
            {'id':'mvm','type':'PhotonicMVMNode','params':{'photonic_params':{
                'noise_std':0.01,'compression':'ITU-F.748-quantized','bandwidth_ghz':100,'latency_ps':50
            }}}
        ],
        'edges':[
            {'from':{'node':'mat','port':'value'},'to':{'node':'mvm','port':'matrix'}},
            {'from':{'node':'vec','port':'value'},'to':{'node':'mvm','port':'vector'}}
        ]
    }
    runtime = UnifiedRuntime()
    runtime.hardware_dispatcher = HardwareDispatcher(use_mock=False)
    result = await runtime.execute_graph(graph)
    print(json.dumps(result, indent=2)[:400], '...')
asyncio.run(run_demo())
PY
```

**Talking Points:**
- Real energy & latency metrics
- Potential API fallback message if key invalid
- Show dashboard panel for energy trend

---

## 5. 🚀 One-Button Demo Orchestrator (Four Acts)

The demo orchestrator provides a complete end-to-end demonstration of the platform in ~90 seconds, showcasing the feedback loop and evolution cycle.

### Prerequisites

Install httpx if not already installed:

```bash
pip install httpx
```

### Running Against Unified Platform (Default)

Start the unified platform:

```bash
# Start unified platform on default port 8080
python -m uvicorn src.full_platform:app --host 127.0.0.1 --port 8080
```

In a separate terminal, run the orchestrator:

```bash
# Run with defaults (unified platform at http://127.0.0.1:8080)
python scripts/demo_orchestrator.py
```

### Running Against Standalone Arena/VULCAN

Start standalone services:

```bash
# Terminal 1: Start Arena
uvicorn src.graphix_arena:app --host 127.0.0.1 --port 8000

# Terminal 2: Start VULCAN (if separate)
python -m src.vulcan.main --port 8001
```

Run orchestrator with environment variables:

```bash
# Point to standalone services
ARENA_BASE=http://127.0.0.1:8000 \
VULCAN_BASE=http://127.0.0.1:8001 \
API_KEY=your-api-key \
DEMO_SEED=42 \
python scripts/demo_orchestrator.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PLATFORM_BASE` | `http://127.0.0.1:8080` | Unified platform base URL |
| `ARENA_BASE` | `http://127.0.0.1:8000` | Standalone Arena URL |
| `VULCAN_BASE` | `http://127.0.0.1:8080/vulcan` | VULCAN URL (or standalone) |
| `API_KEY` | `demo-key` | Arena X-API-Key header value |
| `DEMO_SEED` | `42` | Random seed for reproducibility |

### The Four Acts

The orchestrator demonstrates:

1. **Act 1: Submit Inefficient Run** - Posts a graph generation request to Arena with simulated high latency (8000ms)
2. **Act 2: Submit Negative Feedback** - Sends feedback with low score (0.1) and clear rationale about performance issues
3. **Act 3: Stream VULCAN Mind** - Connects to SSE stream at `/vulcan/v1/stream` for ~10 seconds to monitor cognitive activity
4. **Act 4: Re-run & Metrics** - Re-runs the generator (should be faster) and queries Prometheus metrics to show before/after comparison

### Expected Output

```
======================================================================
  🌈 GRAPHIX/VULCAN DEMO ORCHESTRATOR 🌈
======================================================================

Configuration:
  Platform Base: http://127.0.0.1:8080
  Arena Base: http://127.0.0.1:8000
  VULCAN Base: http://127.0.0.1:8080/vulcan
  Demo Seed: 42
  API Key: demo-key...

======================================================================
  ACT 1: Submit Inefficient Run to Arena
======================================================================

📤 Submitting run to: http://127.0.0.1:8080/api/arena/run/generator
   Payload: {...}
✅ Run submitted successfully!
   Response status: 200
   Run ID: abc123
   Proposal ID: sentiment_3d_spec_v1

...

======================================================================
  🎉 DEMO COMPLETE 🎉
======================================================================
Total duration: 87.3 seconds

The Four Acts demonstrate:
  1. ✅ Graph generation with Arena
  2. ✅ Feedback submission for RLHF
  3. ✅ Real-time mind stream monitoring
  4. ✅ Metrics collection and optimization loop

🌟 Platform successfully demonstrated end-to-end workflow!
```

### Tips for Best Results

- **Deterministic behavior**: Set `DEMO_SEED` to a fixed value (e.g., 42) for reproducible results
- **Performance tuning**: Reduce photonic emulator tensor sizes in configs for faster execution
- **Error handling**: The orchestrator includes retry logic with exponential backoff - errors are gracefully handled
- **Metrics**: Ensure Prometheus client is installed and metrics are enabled in platform settings

---

## 6. 📡 SSE Mind Stream Viewer

A browser-based viewer for real-time VULCAN cognitive activity streaming via Server-Sent Events (SSE).

### Usage

1. **Start the platform or VULCAN service**:

```bash
# Unified platform
python -m uvicorn src.full_platform:app --host 127.0.0.1 --port 8080

# Or standalone VULCAN
python -m src.vulcan.main --port 8001
```

2. **Open the HTML viewer**:

```bash
# Open in browser (adjust path as needed)
open demos/sse_mind.html
# Or on Windows:
start demos/sse_mind.html
# Or on Linux:
xdg-open demos/sse_mind.html
```

3. **Configure connection**:
   - For unified platform: `http://127.0.0.1:8080/vulcan` (default)
   - For standalone VULCAN: `http://127.0.0.1:8001`
   - Via query parameter: `demos/sse_mind.html?base=http://127.0.0.1:8080/vulcan`

4. **Click "Connect"** to start streaming events

### Features

- **Real-time event display**: Shows last 100 events with timestamps
- **Auto-scroll**: Automatically scrolls to latest events (toggleable)
- **Event counter**: Tracks total events received
- **Connection status**: Visual indicator (Connected/Disconnected/Connecting)
- **Clean interface**: Modern, responsive design with color-coded events

### Interpreting the Stream

The SSE stream shows VULCAN's internal cognitive processes:
- Reasoning steps and decision-making
- Goal decomposition and planning
- Memory operations and knowledge updates
- Safety checks and constraint validation
- Intrinsic drive activations (curiosity, improvement, etc.)

---

## 7. Evolution Round

```bash
python scripts/run_sentiment_tournament.py --mode offline --generations 3 --population 6
python src/governance_loop.py --equiv-check
```

Explain:
- Each generation selects a “champion”
- Governance integrates feedback (adaptive improvement)

---

## 8. Safety & Validation

```bash
pytest src/run_validation_test.py -v
```

Interpret results:
- Schema ok
- Ethical audit ok (risky patterns removed)
- Execution stable
- Photonic parameters within safe threshold

---

## 9. Explainability Snapshot

Semantic Map:

```bash
pip install graphviz
python - <<'PY'
from src.observability_manager import ObservabilityManager
import numpy as np
obs = ObservabilityManager()
att = np.random.rand(5,5)
img = obs.plot_semantic_map(att, labels=['A','B','C','D','E'])
print('Map saved:', img)
PY
```

Discuss:
- Node connections represent influence strength
- Chart aids transparency & trust

---

## 10. Troubleshooting Matrix

| Issue | Cause | Resolution |
|-------|-------|-----------|
| API dispatch failure | Invalid key / unavailable backend | Confirm env key; emulator fallback expected |
| No graph output | Edge misconfiguration | Validate nodes & edges; run validator |
| High latency | Hardware fallback loops | Adjust concurrency or dispatch strategy |
| Evolution stagnates | Low mutation diversity | Increase population or mutation ops |
| Validation failure | Unsafe or malformed proposal | Inspect error field; refine mutation heuristics |

---

## 11. Demo Narrative (Investor-Friendly)

1. System generates graph (foundational capability).
2. Receives feedback → evolves into improved champion (autonomy).
3. Hardware-aware execution (efficiency & future scaling).
4. Safety validation & interpretability visualization (trust).
5. Continuous metrics → sustainability & reliability trajectory.

---

## 12. Optional Enhancements

- Include Grafana screenshot (latency & energy panels)
- Add Slack alert simulation (bias detection)
- Show rollback event from adversarial test
- Import `dashboards/grafana/graphix_mission_control.json` into Grafana for Mission Control dashboard
- Use SSE Mind Stream viewer (`demos/sse_mind.html`) for live cognitive monitoring

---

> End with emphasis: “Architecture unifies evolution, safety, interpretability, and sustainable compute—a forward-looking platform.”
