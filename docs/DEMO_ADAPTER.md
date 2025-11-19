# Adapter Demo (Technical Audience)

Purpose
- Show how Vulcan/Graphix governance drops into an existing stack via a thin adapter.
- Keep the executive demo separate. This one is API-first and integration-focused.
- All defaults are filled in and documented so your CTO can change paths/headers later.

## TL;DR (Run)
```bash
# Start your platform or standalone services
export PLATFORM_BASE=http://127.0.0.1:8080
export ARENA_BASE=http://127.0.0.1:8000
export VULCAN_BASE=http://127.0.0.1:8080/vulcan
export API_KEY=demo-key
export DEMO_SEED=42

# Optional: use JSON mapping instead of YAML
# export PLATFORM_MAPPING=configs/platform_mapping.json

python scripts/run_adapter_demo.py
```

## What it does (Acts)
1) Submit baseline/inefficient run (Act 1)  
2) Submit negative feedback (Act 2)  
3) Optional governed improve with objective weights (Act 2.5)  
4) Stream mind (SSE) for ~10s (Act 3)  
5) Query metrics/health (Act 4)  
6) Bonus: rollback/undo if endpoint configured  

## Where to change things
- Endpoint paths/verbs: `configs/platform_mapping.yaml` (or `.json`)
- Auth header: `scripts/platform_adapter.py` → `PlatformAdapter._headers()`
- Request body shapes: method implementations in `scripts/platform_adapter.py`

## Visuals
- Transparency artifacts: open `demos/artifact_card.html` (Base = `$VULCAN_BASE`)
- Live cognition: open existing `demos/sse_mind.html`
- Preference slider (optional): open `demos/preference_slider.html` (posts `objective_weights` to `/improve`)

## Defaults explained
| Action        | Method | Path                           | Base      |
|---------------|--------|--------------------------------|-----------|
| submit_run    | POST   | /api/arena/run/generator       | platform  |
| submit_feedback | POST | /api/arena/feedback_dispatch   | platform  |
| improve       | POST   | /improve                       | vulcan    |
| sse_stream    | GET    | /v1/stream                     | vulcan    |
| metrics       | GET    | /api/arena/metrics             | platform  |
| rollback      | POST   | /rollback                      | vulcan    |

## Troubleshooting
| Problem            | Fix |
|--------------------|-----|
| 401/403 auth error | Change auth header in `_headers()` (e.g., `Authorization: Bearer <token>`) |
| SSE not available  | Ignore Act 3; rely on metrics polling |
| No `/improve`      | Step 2.5 prints NOOP; safe to ignore |
| No `/rollback`     | Bonus step skipped automatically |
| YAML missing       | Use `configs/platform_mapping.json` or install PyYAML |

## Why separate demo?
- Executive demo: narrative-first (governance, safety, ROI).
- Adapter demo: integration-first (API mapping, artifacts, SSE).
- Clear audience targeting reduces cognitive load and maintenance risk.

## Next steps
1. Adjust endpoints or auth if needed.
2. Merge PR.
3. (Optional) Add automated test for adapter with mocked httpx responses.
