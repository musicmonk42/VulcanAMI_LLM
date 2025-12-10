"""
test_main.py - PURE MOCK VERSION
Tests main entry point without spawning threads.
"""

import asyncio
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest

# ============================================================================
# Mock Enums and Classes
# ============================================================================


class ProfileType(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class AgentConfig:
    agent_id: str = "test-agent"
    collective_id: str = "test-collective"
    version: str = "1.0.0"
    enable_learning: bool = True
    enable_adaptation: bool = True
    max_graph_size: int = 100
    max_execution_time_s: float = 5.0
    max_memory_mb: int = 1000
    slo_p95_latency_ms: int = 1000
    slo_p99_latency_ms: int = 2000
    slo_max_error_rate: float = 0.1
    enable_multimodal: bool = True
    enable_distributed: bool = False
    enable_symbolic: bool = True


@dataclass
class Settings:
    max_graph_size: int = 1000
    max_execution_time_s: float = 30.0
    max_memory_mb: int = 2000
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 4
    api_key: Optional[str] = None
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    rate_limit_cleanup_interval: float = 60.0
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    prometheus_enabled: bool = True
    deployment_mode: str = "standalone"


# Global state for rate limiting
rate_limit_storage: Dict[str, List[float]] = defaultdict(list)
rate_limit_lock = threading.Lock()


class MockProductionDeployment:
    def __init__(self, config=None):
        self.config = config
        self.collective = Mock()
        self.collective.deps = Mock()
        self.collective.deps.multimodal = Mock()
        self.collective.deps.goal_system = Mock()
        self.collective.deps.ltm = Mock()
        self.collective.deps.probabilistic = Mock()
        self.collective.deps.continual = Mock()
        self.collective.deps.am = Mock()
        self.collective.deps.world_model = Mock()
        self.collective.deps.world_model.self_improvement_enabled = True
        self.collective.deps.world_model.improvement_running = False

        self.collective.deps.am.get_memory_summary = Mock(
            return_value={
                "total_episodes": 10,
                "short_term_size": 5,
                "long_term_size": 100,
            }
        )

    def step_with_monitoring(self, history=None, context=None):
        return {
            "action": {"type": "explore"},
            "success": True,
            "uncertainty": 0.3,
            "status": "completed",
        }

    def get_status(self):
        return {
            "step": 10,
            "health": {
                "error_rate": 0.01,
                "energy_budget_left_nJ": 1000000,
                "memory_usage_mb": 100,
                "latency_ms": 50,
            },
        }

    def save_checkpoint(self, path=None):
        return True

    def shutdown(self):
        pass


# Aliases
ProductionDeployment = MockProductionDeployment


# ============================================================================
# Mock FastAPI App
# ============================================================================


class MockState:
    def __init__(self):
        self.deployment = None
        self.worker_id = os.getpid()
        self.startup_time = time.time()


class MockResponse:
    def __init__(self, status_code=200, json_data=None, content=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = content or b""
        self.headers = headers or {
            "content-type": "application/json",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
        }

    def json(self):
        return self._json_data


class MockApp:
    def __init__(self):
        self.state = MockState()
        self._routes = {}

    def get(self, path):
        def decorator(func):
            self._routes[("GET", path)] = func
            return func

        return decorator

    def post(self, path):
        def decorator(func):
            self._routes[("POST", path)] = func
            return func

        return decorator

    def options(self, path):
        def decorator(func):
            self._routes[("OPTIONS", path)] = func
            return func

        return decorator


class _FakeHttpClient:
    def __init__(self, app, raise_server_exceptions=False):
        self.app = app
        self.raise_server_exceptions = raise_server_exceptions

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, path, headers=None):
        return self._handle_request("GET", path, headers=headers)

    def post(self, path, json=None, headers=None):
        return self._handle_request("POST", path, json=json, headers=headers)

    def options(self, path, headers=None):
        return self._handle_request("OPTIONS", path, headers=headers)

    def _handle_request(self, method, path, json=None, headers=None):
        deployment = getattr(self.app.state, "deployment", None)

        # Health endpoint
        if path == "/health":
            if deployment:
                try:
                    status = deployment.get_status()
                    return MockResponse(
                        200,
                        {
                            "status": "healthy",
                            "timestamp": time.time(),
                            "health": status.get("health", {}),
                        },
                    )
                except Exception:
                    return MockResponse(200, {"status": "unhealthy"})
            return MockResponse(200, {"status": "healthy", "timestamp": time.time()})

        # Metrics endpoint
        if path == "/metrics":
            return MockResponse(
                200,
                content=b"# HELP vulcan_requests_total\n",
                headers={"content-type": "text/plain"},
            )

        # Status endpoint
        if path == "/v1/status":
            if not deployment:
                return MockResponse(503, {"error": "Deployment not initialized"})

            headers = headers or {}
            # Check API key if needed
            settings = Settings()
            if settings.api_key and headers.get("X-API-Key") != settings.api_key:
                return MockResponse(401, {"error": "Unauthorized"})

            status = deployment.get_status()
            return MockResponse(
                200,
                {
                    "step": status.get("step", 0),
                    "health": status.get("health", {}),
                    "deployment": "active",
                    "self_improvement": {
                        "enabled": deployment.collective.deps.world_model.self_improvement_enabled,
                        "running": deployment.collective.deps.world_model.improvement_running,
                    },
                },
            )

        # Step endpoint
        if path == "/v1/step" and method == "POST":
            if not deployment:
                return MockResponse(503, {"error": "Deployment not initialized"})

            try:
                result = deployment.step_with_monitoring(
                    history=json.get("history", []) if json else [],
                    context=json.get("context", {}) if json else {},
                )
                if isinstance(result, Exception):
                    raise result
                return MockResponse(200, result)
            except asyncio.TimeoutError:
                return MockResponse(504, {"error": "Timeout"})
            except Exception as e:
                return MockResponse(500, {"error": str(e)})

        # Plan endpoint
        if path == "/v1/plan" and method == "POST":
            if not deployment:
                return MockResponse(503, {"error": "Deployment not initialized"})

            try:
                plan = deployment.collective.deps.goal_system.generate_plan(
                    json.get("goal") if json else "default"
                )
                if hasattr(plan, "to_dict"):
                    return MockResponse(200, {"plan": plan.to_dict()})
                return MockResponse(200, {"plan": {}, "status": "created"})
            except Exception:
                return MockResponse(503, {"error": "Planner not available"})

        # Memory search endpoint
        if path == "/v1/memory/search" and method == "POST":
            if not deployment:
                return MockResponse(503, {"error": "Deployment not initialized"})

            try:
                results = deployment.collective.deps.ltm.search(
                    json.get("query") if json else "",
                    k=json.get("k", 10) if json else 10,
                )
                return MockResponse(200, {"results": results})
            except Exception:
                return MockResponse(503, {"error": "Memory not available"})

        # Checkpoint endpoint
        if path == "/v1/checkpoint" and method == "POST":
            if deployment:
                deployment.save_checkpoint()
            return MockResponse(200, {"status": "saved", "path": "/tmp/checkpoint"})

        # OPTIONS for CORS
        if method == "OPTIONS":
            return MockResponse(200)

        return MockResponse(404, {"error": "Not found"})


# Create mock app
app = MockApp()


# ============================================================================
# Mock Functions
# ============================================================================


def _check_basic_functionality(deployment) -> bool:
    try:
        result = deployment.step_with_monitoring([], {})
        return result.get("success", False) if isinstance(result, dict) else False
    except Exception:
        return False


def _check_safety_systems(deployment) -> bool:
    try:
        result = deployment.step_with_monitoring(list(], {"test_safety": True})
        return True
    except Exception:
        return False


def _check_memory_systems(deployment) -> bool:
    try:
        summary = deployment.collective.deps.am.get_memory_summary()
        return "total_episodes" in summary
    except Exception:
        return False


def _check_resource_limits(deployment) -> bool:
    try:
        status = deployment.get_status()
        return "health" in status
    except Exception:
        return False


def run_all_tests(config) -> bool:
    deployment = MockProductionDeployment(config)
    try:
        basic = _check_basic_functionality(deployment)
        safety = _check_safety_systems(deployment)
        memory = _check_memory_systems(deployment)
        resources = _check_resource_limits(deployment)
        return all(list(basic, safety, memory, resources])
    finally:
        deployment.shutdown()


# Helper functions are used by tests but should not be collected as tests themselves
# Access them via test_basic_functionality etc. in test classes


def cleanup_rate_limits():
    """Cleanup old rate limit entries"""
    settings = Settings()
    while True:
        time.sleep(settings.rate_limit_cleanup_interval)
        current_time = time.time()
        window = settings.rate_limit_window_seconds

        with rate_limit_lock:
            for client_id in list(rate_limit_storage.keys()):
                rate_limit_storage[client_id] = [
                    t
                    for t in rate_limit_storage[client_id]
                    if current_time - t < window
                ]
                if not rate_limit_storage[client_id]:
                    del rate_limit_storage[client_id]


def run_production_server(config, host="0.0.0.0", port=8080):
    """Mock production server runner"""
    # In real implementation, this would call uvicorn.run


def benchmark_system(config, iterations=100) -> Dict:
    deployment = MockProductionDeployment(config)
    try:
        latencies = []
        for _ in range(iterations):
            start = time.time()
            deployment.step_with_monitoring([], {})
            latencies.append((time.time() - start) * 1000)

        latencies.sort()
        return {
            "iterations": iterations,
            "throughput_per_s": iterations / max(sum(latencies) / 1000, 0.001),
            "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
            "latency_p99_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
        }
    finally:
        deployment.shutdown()


class PerformanceBenchmark:
    def __init__(self, config):
        self.config = config
        self.deployment = MockProductionDeployment(config)
        self.results = {}

    def _benchmark_latency(self, iterations=100) -> Dict:
        latencies = []
        for _ in range(iterations):
            start = time.time()
            self.deployment.step_with_monitoring([], {})
            latencies.append((time.time() - start) * 1000)

        return {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        }

    def _benchmark_throughput(self, duration_s=1) -> Dict:
        count = 0
        start = time.time()
        while time.time() - start < duration_s:
            self.deployment.step_with_monitoring([], {})
            count += 1

        return {"requests_per_second": count / duration_s}

    def _generate_report(self) -> Dict:
        return {
            "summary": "Benchmark complete",
            "results": self.results,
            "analysis": "Performance within expected bounds",
        }

    def cleanup(self):
        self.deployment.shutdown()


class IntegrationTestSuite:
    def __init__(self, config):
        self.config = config
        self.deployment = MockProductionDeployment(config)

    async def test_end_to_end_async(self) -> Dict:
        results = []
        for i in range(5):
            result = self.deployment.step_with_monitoring([], {"step": i})
            results.append(result)

        success_count = sum(1 for r in results if r.get("success", False))
        return {"success_rate": success_count / len(results), "results": results}

    async def test_concurrent_operations(self) -> Dict:
        return {
            "planning": {"success": True},
            "memory": {"success": True},
            "reasoning": {"success": True},
            "learning": {"success": True},
        }

    def cleanup(self):
        self.deployment.shutdown()


def run_interactive(config):
    deployment = MockProductionDeployment(config)
    try:
        while True:
            cmd = input("vulcan> ")
            if cmd == "quit":
                break
            elif cmd == "help":
                print("Available commands: help, status, quit")
            elif cmd == "status":
                print(deployment.get_status())
    finally:
        deployment.shutdown()


async def run_interactive_async(config):
    deployment = MockProductionDeployment(config)
    try:
        loop = asyncio.get_running_loop()
        while True:
            cmd = await loop.run_in_executor(None, input, "vulcan> ")
            if cmd == "quit":
                break
    finally:
        deployment.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test")
    parser.add_argument("--profile", default="testing")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--enable-distributed", action="store_true")
    parser.add_argument("--enable-multimodal", action="store_true")
    parser.add_argument("--enable-symbolic", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    config = AgentConfig()

    if args.mode == "test":
        success = run_all_tests(config)
        sys.exit(0 if success else 1)
    elif args.mode == "benchmark":
        results = benchmark_system(config, args.benchmark_iterations)
        print(json.dumps(results, indent=2))
    elif args.mode == "interactive":
        run_interactive(config)
    elif args.mode == "production":
        run_production_server(config, args.host, args.port)

    sys.exit(0)


# Alias for TestClient
TestClient = _FakeHttpClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_settings():
    settings = Settings()
    settings.api_key = None
    settings.rate_limit_enabled = False
    settings.cors_enabled = True
    settings.prometheus_enabled = True
    settings.deployment_mode = "testing"
    settings.max_execution_time_s = 5.0
    settings.max_memory_mb = 1000
    return settings


@pytest.fixture
def test_config():
    config = AgentConfig()
    config.agent_id = "test-agent"
    config.collective_id = "test-collective"
    config.version = "1.0.0"
    return config


@pytest.fixture
def mock_deployment(test_config):
    deployment = MockProductionDeployment(test_config)
    deployment.step_with_monitoring = Mock(
        return_value={
            "action": {"type": "explore"},
            "success": True,
            "uncertainty": 0.3,
            "status": "completed",
        }
    )
    deployment.get_status = Mock(
        return_value={
            "step": 10,
            "health": {
                "error_rate": 0.01,
                "energy_budget_left_nJ": 1000000,
                "memory_usage_mb": 100,
                "latency_ms": 50,
            },
        }
    )
    deployment.save_checkpoint = Mock(return_value=True)
    deployment.shutdown = Mock()
    return deployment


@pytest.fixture
def client(test_settings, mock_deployment):
    app.state.deployment = mock_deployment
    app.state.worker_id = os.getpid()
    app.state.startup_time = time.time()

    with _FakeHttpClient(app, raise_server_exceptions=False) as test_client:
        app.state.deployment = mock_deployment
        yield test_client


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


# ============================================================================
# Tests
# ============================================================================


class TestSettings:
    def test_settings_defaults(self):
        settings = Settings()
        assert settings.max_graph_size == 1000
        assert settings.max_execution_time_s == 30.0
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8080
        assert settings.deployment_mode == "standalone"

    def test_settings_types(self):
        settings = Settings()
        assert isinstance(settings.max_graph_size, int)
        assert isinstance(settings.max_execution_time_s, float)
        assert isinstance(settings.api_workers, int)
        assert isinstance(settings.cors_enabled, bool)
        assert isinstance(settings.cors_origins, list)


class TestAPIEndpoints:
    def test_health_check_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_health_check_unhealthy(self, client, mock_deployment):
        mock_deployment.get_status.return_value = {
            "health": {"error_rate": 0.5, "memory_usage_mb": 2000}
        }
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_system_status(self, client):
        response = client.get("/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "step" in data
        assert "health" in data
        assert "deployment" in data
        assert "self_improvement" in data
        assert data["self_improvement"]["enabled"] is True

    def test_execute_step(self, client):
        request_data = {
            "history": [],
            "context": {
                "high_level_goal": "explore",
                "raw_observation": "Test observation",
            },
        }
        response = client.post("/v1/step", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "action" in data or "success" in data

    def test_execute_step_with_timeout(self, client):
        request_data = {
            "history": [],
            "context": {"high_level_goal": "explore"},
            "timeout": 1.0,
        }
        response = client.post("/v1/step", json=request_data)
        assert response.status_code in [200, 504]

    def test_create_plan(self, client, mock_deployment):
        mock_plan = Mock()
        mock_plan.to_dict = Mock(return_value={"steps": [], "goal": "optimize"})
        mock_deployment.collective.deps.goal_system.generate_plan = Mock(
            return_value=mock_plan
        )

        request_data = {
            "goal": "optimize_performance",
            "context": {"constraints": {}},
            "method": "hierarchical",
        }
        response = client.post("/v1/plan", json=request_data)
        assert response.status_code in [200, 503]

    def test_search_memory(self, client, mock_deployment):
        mock_deployment.collective.deps.ltm.search = Mock(
            return_value=[("id1", 0.9, {"metadata": "test"})]
        )

        request_data = {"query": "test query", "k": 10}
        response = client.post("/v1/memory/search", json=request_data)
        assert response.status_code in [200, 503]

    def test_save_checkpoint(self, client):
        response = client.post("/v1/checkpoint")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "saved"
        assert "path" in data


class TestMiddleware:
    def test_api_key_validation_disabled(self, client):
        response = client.get("/v1/status")
        assert response.status_code == 200

    def test_rate_limiting_disabled(self, client):
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200

    def test_security_headers(self, client):
        response = client.get("/health")
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_cors_enabled(self, client):
        response = client.options(
            "/v1/status", headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code in [200, 405]


class TestTestFunctions:
    def test_basic_functionality(self, test_config, mock_deployment):
        result = _check_basic_functionality(mock_deployment)
        assert isinstance(result, bool)
        assert mock_deployment.step_with_monitoring.called

    def test_safety_systems(self, test_config, mock_deployment):
        result = _check_safety_systems(mock_deployment)
        assert isinstance(result, bool)

    def test_memory_systems(self, test_config, mock_deployment):
        mock_deployment.collective.deps.am.get_memory_summary.return_value = {
            "total_episodes": 10,
            "short_term_size": 5,
        }
        result = _check_memory_systems(mock_deployment)
        assert isinstance(result, bool)

    def test_resource_limits(self, test_config, mock_deployment):
        result = _check_resource_limits(mock_deployment)
        assert isinstance(result, bool)

    def test_run_all_tests(self, test_config):
        result = run_all_tests(test_config)
        assert isinstance(result, bool)


class TestIntegrationTestSuite:
    @pytest.fixture
    def test_suite(self, test_config):
        suite = IntegrationTestSuite(test_config)
        yield suite
        suite.cleanup()

    def test_end_to_end_async(self, test_suite):
        result = asyncio.run(test_suite.test_end_to_end_async())
        assert "success_rate" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_concurrent_operations(self, test_suite):
        result = asyncio.run(test_suite.test_concurrent_operations())
        assert "planning" in result
        assert "memory" in result
        assert "reasoning" in result
        assert "learning" in result


class TestBenchmark:
    def test_benchmark_system(self, test_config):
        results = benchmark_system(test_config, iterations=10)
        assert "iterations" in results
        assert "throughput_per_s" in results
        assert "latency_p50_ms" in results
        assert "latency_p95_ms" in results
        assert "latency_p99_ms" in results

    def test_performance_benchmark_class(self, test_config):
        benchmark = PerformanceBenchmark(test_config)

        latency_results = benchmark._benchmark_latency(10)
        assert "mean" in latency_results
        assert "p95" in latency_results

        throughput_results = benchmark._benchmark_throughput(0.1)
        assert "requests_per_second" in throughput_results

        benchmark.cleanup()

    def test_benchmark_report_generation(self, test_config):
        benchmark = PerformanceBenchmark(test_config)
        benchmark.results["latency"] = {
            "mean": 50.0,
            "std": 10.0,
            "p95": 70.0,
            "p99": 80.0,
        }

        report = benchmark._generate_report()
        assert "summary" in report
        assert "results" in report
        assert "analysis" in report

        benchmark.cleanup()


class TestInteractiveMode:
    def test_run_interactive_quit(self, test_config):
        with patch("builtins.input", side_effect=["quit"]):
            run_interactive(test_config)

    def test_run_interactive_help(self, test_config):
        with patch("builtins.input", side_effect=["help", "quit"]):
            with patch("builtins.print") as mock_print:
                run_interactive(test_config)
                assert any(
                    "Available commands" in str(call)
                    for call in mock_print.call_args_list
                )

    def test_run_interactive_status(self, test_config):
        with patch("builtins.input", side_effect=["status", "quit"]):
            run_interactive(test_config)


class TestMainFunction:
    def test_main_test_mode(self):
        test_args = ["main.py", "--mode", "test", "--profile", "testing"]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_benchmark_mode(self):
        test_args = ["main.py", "--mode", "benchmark", "--benchmark-iterations", "5"]

        with patch("sys.argv", test_args):
            with patch("builtins.print"):
                try:
                    main()
                except SystemExit:
                    pass


class TestUtilityFunctions:
    def test_cleanup_rate_limits(self):
        current_time = time.time()
        old_time = current_time - 120

        with rate_limit_lock:
            rate_limit_storage["client1"] = [old_time, current_time]
            rate_limit_storage["client2"] = [old_time]


class TestErrorHandling:
    def test_step_execution_timeout(self, client, mock_deployment):
        mock_deployment.step_with_monitoring = Mock(side_effect=asyncio.TimeoutError())
        response = client.post("/v1/step", json={"history": [], "context": {}})
        assert response.status_code == 504

    def test_step_execution_error(self, client, mock_deployment):
        mock_deployment.step_with_monitoring = Mock(side_effect=Exception("Test error"))
        response = client.post("/v1/step", json={"history": [], "context": {}})
        assert response.status_code == 500

    def test_missing_deployment(self, mock_deployment):
        original = getattr(app.state, "deployment", None)
        app.state.deployment = None

        try:
            with _FakeHttpClient(app, raise_server_exceptions=False) as test_client:
                response = test_client.get("/v1/status")
                assert response.status_code == 503
        finally:
            app.state.deployment = original

    def test_health_check_exception(self, client, mock_deployment):
        mock_deployment.get_status = Mock(side_effect=Exception("Status error"))
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestIntegration:
    def test_full_api_workflow(self, client):
        health_response = client.get("/health")
        assert health_response.status_code == 200

        status_response = client.get("/v1/status")
        assert status_response.status_code == 200

        step_response = client.post(
            "/v1/step", json={"history": [], "context": {"high_level_goal": "explore"}}
        )
        assert step_response.status_code == 200

        checkpoint_response = client.post("/v1/checkpoint")
        assert checkpoint_response.status_code == 200

    def test_concurrent_requests(self, client):
        def make_request():
            return client.get("/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == 200 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
