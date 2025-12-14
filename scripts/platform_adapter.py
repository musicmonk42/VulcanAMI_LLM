#!/usr/bin/env python3
"""
PlatformAdapter: map the Four Acts demo to your platform without touching the main demo.

WHAT TO CHANGE (CTO):
- Auth header: see _headers()
- Endpoint paths: change configs/platform_mapping.yaml or .json
- Request bodies: edit the specific methods (submit_inefficient_run, submit_feedback, improve, rollback)

Defaults here work against the Graphix/Vulcan unified platform.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urljoin

import httpx


@dataclass
class AdapterConfig:
    platform_base: str
    arena_base: str
    vulcan_base: str
    api_key: str
    demo_seed: int

    @classmethod
    def from_env(cls) -> "AdapterConfig":
        platform_base = os.getenv("PLATFORM_BASE", "http://0.0.0.0:8000")
        arena_base = os.getenv("ARENA_BASE", "http://127.0.0.1:8000")
        vulcan_base = os.getenv("VULCAN_BASE", f"{platform_base}/vulcan")
        api_key = os.getenv("API_KEY", "demo-key")
        demo_seed = int(os.getenv("DEMO_SEED", "42"))
        return cls(platform_base, arena_base, vulcan_base, api_key, demo_seed)


class PlatformAdapter:
    """
    Thin client that wraps your endpoints for the Four Acts:
      - submit_inefficient_run
      - submit_feedback
      - stream_mind
      - query_metrics
      - improve (optional)
      - rollback (optional)
    """

    def __init__(self, cfg: AdapterConfig, mapping: Dict[str, Any]):
        self.cfg = cfg
        self.mapping = mapping
        self.client = httpx.AsyncClient(timeout=30.0, headers=self._headers())

    def _headers(self) -> Dict[str, str]:
        """
        CHANGE AUTH HERE if needed.

        Default: X-API-Key: <API_KEY>
        Example for bearer:
          h["Authorization"] = f"Bearer {self.cfg.api_key}"
        """
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["X-API-Key"] = self.cfg.api_key
        return h

    async def close(self) -> None:
        await self.client.aclose()

    async def _retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Generic HTTP caller with retries and backoff.
        Adjust exclusions (e.g., don't retry 401/403) if needed.
        """
        max_retries = kwargs.pop("max_retries", 3)
        backoff = kwargs.pop("backoff_factor", 1.5)
        for attempt in range(max_retries):
            try:
                resp = await self.client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == max_retries - 1:
                    raise
                wait = backoff**attempt
                print(f"⚠️  {method} {url} failed ({e}); retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
        raise RuntimeError("Unreachable")

    def _url(self, base_key: str, path: str) -> str:
        base = {
            "platform": self.cfg.platform_base,
            "arena": self.cfg.arena_base,
            "vulcan": self.cfg.vulcan_base,
        }[base_key]
        return urljoin(base, path)

    # Act 1
    async def submit_inefficient_run(
        self, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        m = self.mapping["submit_run"]
        url = self._url(m["base"], m["path"])
        body = payload or {
            "spec_id": "sentiment_3d_spec",
            "parameters": {
                "goal": "demo baseline",
                "demo_latency_ms": 8000,
                "complexity": "high",
            },
        }
        resp = await self._retry(m["method"], url, json=body)
        return resp.json()

    # Act 2
    async def submit_feedback(
        self, run_id: str, score: float = 0.1, rationale: str = "Too slow"
    ) -> Dict[str, Any]:
        m = self.mapping["submit_feedback"]
        url = self._url(m["base"], m["path"])
        body = {
            "run_id": run_id,
            "proposal_id": run_id,
            "score": score,
            "rationale": rationale,
            "metrics": {"latency_ms": 8500, "accuracy": 0.75, "energy_efficiency": 0.4},
        }
        resp = await self._retry(m["method"], url, json=body)
        return resp.json()

    # Act 2.5 (optional)
    async def improve(
        self,
        trigger: str = "performance_drop",
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        m = self.mapping.get("improve")
        if not m:
            return {"status": "noop", "reason": "improve not configured"}
        url = self._url(m["base"], m["path"])
        body = {"trigger": trigger}
        if objective_weights:
            body["objective_weights"] = objective_weights
        resp = await self._retry(m["method"], url, json=body)
        return resp.json()

    # Act 3
    async def stream_mind(self) -> AsyncIterator[str]:
        m = self.mapping["sse_stream"]
        url = self._url(m["base"], m["path"])
        async with self.client.stream("GET", url) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    yield line[5:].strip()

    # Act 4
    async def query_metrics(self) -> Dict[str, Any]:
        m = self.mapping["metrics"]
        url = self._url(m["base"], m["path"])
        resp = await self._retry(m["method"], url)
        ct = resp.headers.get("content-type", "")
        return resp.json() if "application/json" in ct else {"raw": resp.text}

    # Bonus
    async def rollback(self, change_id: str) -> Dict[str, Any]:
        m = self.mapping.get("rollback")
        if not m:
            return {"status": "noop", "reason": "rollback not configured"}
        url = self._url(m["base"], m["path"])
        resp = await self._retry(m["method"], url, json={"change_id": change_id})
        return resp.json()
