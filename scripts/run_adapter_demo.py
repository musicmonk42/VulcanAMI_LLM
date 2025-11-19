#!/usr/bin/env python3
"""
Adapter Demo Runner (separate from the executive Four Acts demo)

What this does:
1) Submit a baseline/inefficient run (Act 1)
2) Submit negative feedback (Act 2)
3) Optional governed improve with preference weights (Act 2.5)
4) Stream mind (SSE) ~10s (Act 3)
5) Query metrics/health (Act 4)
6) Bonus: rollback/undo if configured

CTO knobs:
- Endpoint mapping file: PLATFORM_MAPPING env -> configs/platform_mapping.yaml|.json
- Auth: scripts/platform_adapter.py (PlatformAdapter._headers)
- Request body tweaks: scripts/platform_adapter.py methods
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # pyright: ignore

from platform_adapter import AdapterConfig, PlatformAdapter  # type: ignore


def banner(title: str, char: str = "="):
    line = char * 70
    print(f"\n{line}\n  {title}\n{line}\n")


def load_mapping(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            j = p.with_suffix(".json")
            if j.exists():
                return json.loads(j.read_text())
            raise RuntimeError("PyYAML not installed and no JSON mapping available")
        return yaml.safe_load(p.read_text())
    elif p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    else:
        y = p.with_suffix(".yaml")
        if y.exists():
            return load_mapping(str(y))
        j = p.with_suffix(".json")
        if j.exists():
            return load_mapping(str(j))
        raise FileNotFoundError(f"No mapping found at {path} (.yaml/.json)")


async def main() -> int:
    cfg = AdapterConfig.from_env()
    mapping_path = os.getenv("PLATFORM_MAPPING", "configs/platform_mapping.yaml")
    mapping = load_mapping(mapping_path)

    print("Configuration:")
    print(f"  Platform Base: {cfg.platform_base}")
    print(f"  Arena Base:    {cfg.arena_base}")
    print(f"  VULCAN Base:   {cfg.vulcan_base}")
    print(f"  Demo Seed:     {cfg.demo_seed}")
    print(f"  API Key:       {cfg.api_key[:8]}..." if len(cfg.api_key) > 8 else f"  API Key: {cfg.api_key}")

    adapter = PlatformAdapter(cfg, mapping)
    start = time.time()
    try:
        banner("ACT 1: Submit Inefficient/Baseline Run")
        run_resp = await adapter.submit_inefficient_run()
        print("📤 Submit response:", json.dumps(run_resp, indent=2))
        run_id = run_resp.get("run_id") or run_resp.get("id") or run_resp.get("proposal_id") or "unknown"

        banner("ACT 2: Submit Negative Feedback")
        fb = await adapter.submit_feedback(run_id, score=0.1, rationale="High latency; please optimize")
        print("📝 Feedback response:", json.dumps(fb, indent=2))

        banner("ACT 2.5: Trigger Improve (preference = latency)")
        imp = await adapter.improve(trigger="performance_drop", objective_weights={"speed": 0.7, "accuracy": 0.3})
        print("🚀 Improve response:", json.dumps(imp, indent=2))
        change_id = imp.get("change_id", "")

        banner("ACT 3: Stream Mind (SSE ~10s)")
        t0 = time.time()
        async for msg in adapter.stream_mind():
            print("🧠", msg)
            if time.time() - t0 > 10:
                break

        banner("ACT 4: Query Metrics/Health")
        met = await adapter.query_metrics()
        print("📈 Metrics/Health:", json.dumps(met, indent=2) if isinstance(met, dict) else met)

        if change_id:
            banner("BONUS: Undo last safe change")
            rb = await adapter.rollback(change_id)
            print("↩ Rollback:", json.dumps(rb, indent=2))

        elapsed = time.time() - start
        banner(f"✅ ADAPTER DEMO COMPLETE in {elapsed:.1f}s")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    finally:
        await adapter.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
