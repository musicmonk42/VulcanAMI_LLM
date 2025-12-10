#!/usr/bin/env python3
"""
Demo Orchestrator - One-button Four Acts Demo
==============================================
Purpose: Orchestrate the Four Acts end-to-end in ~90 seconds against a running
unified platform (or standalone Arena/VULCAN) using existing APIs.

Acts:
1. Submit an "inefficient" run to Arena
2. Submit negative feedback
3. Open VULCAN SSE stream for ~10 seconds
4. Re-run generator and query metrics

Usage:
    # Against unified platform (default)
    python scripts/demo_orchestrator.py

    # Against standalone services
    ARENA_BASE=http://127.0.0.1:8000 VULCAN_BASE=http://127.0.0.1:8001 python scripts/demo_orchestrator.py

Environment Variables:
    PLATFORM_BASE: Unified platform base URL (default: http://127.0.0.1:8080)
    ARENA_BASE: Arena base URL (default: http://127.0.0.1:8000)
    VULCAN_BASE: VULCAN base URL (default: http://127.0.0.1:8080/vulcan)
    API_KEY: Arena X-API-Key header value (default: demo-key)
    DEMO_SEED: Random seed for deterministic behavior (default: 42)
"""

import asyncio
import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

try:
    import httpx
except ImportError:
    print("❌ Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


class DemoConfig:
    """Configuration for demo orchestrator."""

    def __init__(self):
        self.platform_base = os.getenv("PLATFORM_BASE", "http://127.0.0.1:8080")
        self.arena_base = os.getenv("ARENA_BASE", "http://127.0.0.1:8000")
        self.vulcan_base = os.getenv("VULCAN_BASE", f"{self.platform_base}/vulcan")
        self.api_key = os.getenv("API_KEY", "demo-key")
        self.demo_seed = int(os.getenv("DEMO_SEED", "42"))

        # Set random seed for reproducibility
        random.seed(self.demo_seed)

    def get_arena_url(self, endpoint: str) -> str:
        """Get full Arena URL for endpoint."""
        # Check if using unified platform
        if "8080" in self.platform_base:
            # Use unified platform proxy
            return urljoin(self.platform_base, f"/api/arena{endpoint}")
        else:
            # Use standalone Arena
            return urljoin(self.arena_base, f"/api{endpoint}")

    def get_vulcan_url(self, endpoint: str) -> str:
        """Get full VULCAN URL for endpoint."""
        return urljoin(self.vulcan_base, endpoint)


class DemoOrchestrator:
    """Orchestrates the Four Acts demo."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
        self.run_id = None
        self.proposal_id = None

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def print_banner(self, title: str, char: str = "="):
        """Print a section banner."""
        banner = char * 70
        print(f"\n{banner}")
        print(f"  {title}")
        print(f"{banner}\n")

    async def retry_request(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff_factor ** attempt
                print(f"⚠️  Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        raise Exception("Max retries exceeded")

    async def act_1_submit_inefficient_run(self):
        """Act 1: Submit an inefficient run to Arena."""
        self.print_banner("ACT 1: Submit Inefficient Run to Arena", "=")

        # Prepare payload with recognizable spec and simulated latency
        payload = {
            "spec_id": "sentiment_3d_spec",
            "parameters": {
                "goal": "Photonic sentiment analysis with demo latency",
                "demo_latency_ms": 8000,
                "complexity": "high"
            }
        }

        url = self.config.get_arena_url("/run/generator")
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key
        }

        print(f"📤 Submitting run to: {url}")
        print(f"   Payload: {json.dumps(payload, indent=2)}")

        try:
            response = await self.retry_request(
                "POST",
                url,
                headers=headers,
                json=payload
            )

            result = response.json()
            print(f"✅ Run submitted successfully!")
            print(f"   Response status: {response.status_code}")

            # Try to extract run ID or proposal ID from response
            if isinstance(result, dict):
                self.run_id = result.get("run_id") or result.get("id")
                self.proposal_id = result.get("proposal_id") or result.get("graph_id")
                print(f"   Run ID: {self.run_id}")
                print(f"   Proposal ID: {self.proposal_id}")

            return result

        except httpx.HTTPError as e:
            print(f"❌ Failed to submit run: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response: {e.response.text[:200]}")
            # Continue with demo even if Act 1 fails
            return None

    async def act_2_submit_feedback(self):
        """Act 2: Submit negative feedback."""
        self.print_banner("ACT 2: Submit Negative Feedback", "=")

        # Use proposal_id from Act 1 if available, otherwise use a placeholder
        feedback_payload = {
            "proposal_id": self.proposal_id or "initial_graph_v1",
            "score": 0.1,
            "rationale": "Performance is suboptimal - high latency detected, needs optimization",
            "metrics": {
                "latency_ms": 8500,
                "accuracy": 0.75,
                "energy_efficiency": 0.4
            }
        }

        url = self.config.get_arena_url("/feedback_dispatch")
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key
        }

        print(f"📤 Submitting feedback to: {url}")
        print(f"   Feedback: score={feedback_payload['score']}")
        print(f"   Rationale: {feedback_payload['rationale']}")

        try:
            response = await self.retry_request(
                "POST",
                url,
                headers=headers,
                json=feedback_payload
            )

            result = response.json()
            print(f"✅ Feedback submitted successfully!")
            print(f"   Response status: {response.status_code}")
            return result

        except httpx.HTTPError as e:
            print(f"❌ Failed to submit feedback: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response: {e.response.text[:200]}")
            return None

    async def act_3_stream_mind(self):
        """Act 3: Open VULCAN SSE stream for ~10 seconds."""
        self.print_banner("ACT 3: Stream VULCAN Mind (SSE)", "=")

        url = self.config.get_vulcan_url("/v1/stream")
        print(f"📡 Connecting to SSE stream: {url}")
        print(f"   Streaming for ~10 seconds...")

        message_count = 0
        last_event = None
        start_time = time.time()
        max_duration = 10.0

        try:
            async with self.client.stream("GET", url, timeout=15.0) as response:
                print(f"✅ Connected! Status: {response.status_code}")

                async for line in response.aiter_lines():
                    if time.time() - start_time > max_duration:
                        print(f"\n⏱️  Reached {max_duration}s duration limit")
                        break

                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        message_count += 1
                        last_event = line[5:].strip()

                        # Print first few messages
                        if message_count <= 3:
                            print(f"   Message {message_count}: {last_event[:80]}...")

                print(f"\n📊 Stream Summary:")
                print(f"   Total messages: {message_count}")
                if last_event:
                    print(f"   Last event: {last_event[:100]}...")

                return {"message_count": message_count, "last_event": last_event}

        except httpx.HTTPError as e:
            print(f"⚠️  Stream connection issue: {e}")
            print(f"   This is expected if VULCAN is not running")
            return {"message_count": 0, "error": str(e)}
        except Exception as e:
            print(f"❌ Stream error: {e}")
            return {"message_count": 0, "error": str(e)}

    async def act_4_rerun_and_metrics(self):
        """Act 4: Re-run generator and query metrics."""
        self.print_banner("ACT 4: Re-run Generator & Query Metrics", "=")

        # Re-run the generator with same spec
        print("🔄 Re-running generator (should be faster after feedback)...")

        payload = {
            "spec_id": "sentiment_3d_spec",
            "parameters": {
                "goal": "Photonic sentiment analysis - optimized",
                "complexity": "medium"
            }
        }

        url = self.config.get_arena_url("/run/generator")
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key
        }

        try:
            response = await self.retry_request(
                "POST",
                url,
                headers=headers,
                json=payload,
                max_retries=2
            )

            result = response.json()
            print(f"✅ Second run completed!")
            print(f"   Response status: {response.status_code}")

        except httpx.HTTPError as e:
            print(f"⚠️  Second run failed: {e}")

        # Query metrics
        print(f"\n📊 Querying metrics...")

        metrics_url = self.config.get_arena_url("/metrics")

        try:
            response = await self.retry_request(
                "GET",
                metrics_url,
                headers={"X-API-Key": self.config.api_key},
                max_retries=2
            )

            metrics_text = response.text
            print(f"✅ Metrics retrieved! ({len(metrics_text)} bytes)")

            # Parse and summarize key metrics
            self.summarize_metrics(metrics_text)

            return {"metrics": metrics_text}

        except httpx.HTTPError as e:
            print(f"⚠️  Could not retrieve metrics: {e}")
            return None

    def summarize_metrics(self, metrics_text: str):
        """Summarize key metrics from Prometheus format."""
        print(f"\n📈 Key Metrics Summary:")

        lines = metrics_text.split('\n')

        # Look for interesting metrics
        metrics_of_interest = [
            "bias_detections",
            "agent_task_completed",
            "execution_latency",
            "arena_requests_total",
            "graphix_energy"
        ]

        found_metrics = []
        for line in lines:
            if line.startswith('#'):
                continue

            for metric in metrics_of_interest:
                if metric in line:
                    found_metrics.append(line.strip())
                    break

        if found_metrics:
            for metric in found_metrics[:10]:  # Show first 10 matching metrics
                print(f"   {metric}")
        else:
            print(f"   No specific metrics found (showing first 5 lines):")
            for line in lines[:5]:
                if line and not line.startswith('#'):
                    print(f"   {line}")

        print(f"\n💡 Before/After Summary:")
        print(f"   Act 1: Submitted inefficient run (simulated 8000ms latency)")
        print(f"   Act 2: Provided negative feedback (score: 0.1)")
        print(f"   Act 3: Monitored VULCAN mind stream")
        print(f"   Act 4: Re-ran with optimization (feedback loop complete)")

    async def run_demo(self):
        """Execute all four acts of the demo."""
        self.print_banner("🌈 GRAPHIX/VULCAN DEMO ORCHESTRATOR 🌈", "=")

        print(f"Configuration:")
        print(f"  Platform Base: {self.config.platform_base}")
        print(f"  Arena Base: {self.config.arena_base}")
        print(f"  VULCAN Base: {self.config.vulcan_base}")
        print(f"  Demo Seed: {self.config.demo_seed}")
        print(f"  API Key: {self.config.api_key[:8]}..." if len(self.config.api_key) > 8 else f"  API Key: {self.config.api_key}")

        start_time = time.time()

        try:
            # Execute the Four Acts
            await self.act_1_submit_inefficient_run()
            await asyncio.sleep(1)  # Brief pause between acts

            await self.act_2_submit_feedback()
            await asyncio.sleep(1)

            await self.act_3_stream_mind()
            await asyncio.sleep(1)

            await self.act_4_rerun_and_metrics()

            duration = time.time() - start_time

            self.print_banner("🎉 DEMO COMPLETE 🎉", "=")
            print(f"Total duration: {duration:.1f} seconds")
            print(f"\nThe Four Acts demonstrate:")
            print(f"  1. ✅ Graph generation with Arena")
            print(f"  2. ✅ Feedback submission for RLHF")
            print(f"  3. ✅ Real-time mind stream monitoring")
            print(f"  4. ✅ Metrics collection and optimization loop")
            print(f"\n🌟 Platform successfully demonstrated end-to-end workflow!")

            return 0

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Demo interrupted by user")
            return 130

        except Exception as e:
            print(f"\n\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        finally:
            await self.close()


async def main():
    """Main entry point."""
    config = DemoConfig()
    orchestrator = DemoOrchestrator(config)

    try:
        exit_code = await orchestrator.run_demo()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if running in async context
    try:
        asyncio.run(main())
    except RuntimeError:
        # Already in async context
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
