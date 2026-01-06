#!/usr/bin/env python3
"""
Simple load tool for /vulcan/v1/chat that captures latency and success metrics.

This is intentionally lightweight so it can run in CI or locally without extra setup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from statistics import median
from typing import Any, Dict, List

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    AIOHTTP_AVAILABLE = False


DEFAULT_PAYLOAD = {"messages": [{"role": "user", "content": "ping"}]}


@dataclass
class RequestResult:
    ok: bool
    status: int
    latency_ms: float
    error: str | None = None


def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    idx = max(0, min(len(data) - 1, int(round((pct / 100) * (len(data) - 1)))))
    return sorted(data)[idx]


async def run_request(session: aiohttp.ClientSession, url: str, timeout: float, payload: Dict[str, Any]) -> RequestResult:
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            await resp.text()
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(ok=200 <= resp.status < 300, status=resp.status, latency_ms=latency_ms)
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(ok=False, status=0, latency_ms=latency_ms, error=str(e))


async def run_load(
    url: str,
    concurrency: int,
    request_count: int,
    duration_seconds: float,
    timeout: float,
    payload: Dict[str, Any],
) -> List[RequestResult]:
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not installed; install it to run load generation")

    results: List[RequestResult] = []
    deadline = time.monotonic() + duration_seconds if duration_seconds > 0 else None

    connector = aiohttp.TCPConnector(limit=None)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def worker(idx: int) -> None:
            nonlocal results
            async with sem:
                res = await run_request(session, url, timeout, payload)
                results.append(res)

        tasks = []
        sent = 0
        while (deadline is None or time.monotonic() < deadline) and (request_count <= 0 or sent < request_count):
            tasks.append(asyncio.create_task(worker(sent)))
            sent += 1
            # small yield to avoid overwhelming event loop for tiny runs
            await asyncio.sleep(0)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    return results


def write_artifacts(results: List[RequestResult], json_path: str, summary_path: str) -> Dict[str, Any]:
    latencies = [r.latency_ms for r in results if r.ok]
    success_count = sum(1 for r in results if r.ok)
    total = len(results)
    success_rate = (success_count / total) * 100 if total else 0
    p50 = median(latencies) if latencies else 0.0
    p95 = percentile(latencies, 95) if latencies else 0.0
    errors = [r for r in results if not r.ok]

    summary: Dict[str, Any] = {
        "total_requests": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "error_count": len(errors),
        "errors": [asdict(e) for e in errors[:10]],
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": [asdict(r) for r in results], "summary": summary}, f, indent=2)

    lines = [
        f"Total requests: {total}",
        f"Success rate:   {success_rate:.2f}%",
        f"Latency p50:    {p50:.2f} ms",
        f"Latency p95:    {p95:.2f} ms",
        f"Errors:         {len(errors)}",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal load generator for /vulcan/v1/chat")
    parser.add_argument("--base-url", default=os.getenv("VULCAN_BASE_URL", "http://localhost:8000"), help="Base URL for the API")
    parser.add_argument("--path", default="/vulcan/v1/chat", help="Endpoint path")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent requests")
    parser.add_argument("--requests", type=int, default=10, help="Total request count (0 = run for duration)")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (ignored if requests>0)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout seconds")
    parser.add_argument("--payload", default=None, help="Optional JSON payload file")
    parser.add_argument("--json-output", default="artifacts/load_results.json", help="Path to write JSON results")
    parser.add_argument("--summary-output", default="artifacts/load_summary.txt", help="Path to write text summary")
    parser.add_argument("--strict", action="store_true", help="Non-zero exit if success rate <100%")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload = DEFAULT_PAYLOAD
    if args.payload:
        with open(args.payload, "r", encoding="utf-8") as f:
            payload = json.load(f)

    url = f"{args.base_url.rstrip('/')}{args.path}"

    if not AIOHTTP_AVAILABLE:
        print("aiohttp is required for load generation. Install it or run `pip install aiohttp`.")
        return 2

    try:
        results = asyncio.run(
            run_load(
                url=url,
                concurrency=max(1, args.concurrency),
                request_count=max(0, args.requests),
                duration_seconds=max(0.0, args.duration),
                timeout=args.timeout,
                payload=payload,
            )
        )
    except Exception as e:
        print(f"Load generation failed: {e}")
        return 1

    summary = write_artifacts(results, args.json_output, args.summary_output)
    if args.strict and summary["success_rate"] < 100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
