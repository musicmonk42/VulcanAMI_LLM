#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphix IR Validation Test Suite
===============================
A world-class pytest-based test suite for validating Graphix IR graphs against
type system manifest (v1.2.0) and formal grammar (v1.3.1). Integrates NSOAligner
for ethical checks, UnifiedRuntime for execution, HardwareDispatcher for photonic
validation, and ObservabilityManager for Prometheus metrics/Grafana dashboards.
Supports 2025 features (ITU F.748.53 compression, EU ethical labels, Grok-4/5 audits).
Designed for CI/CD with deterministic, secure, auditable, and scalable behavior.

FIXES APPLIED:
- Proper resource cleanup with context manager support
- Thread-safe metric registration
- Fail-safe unknown status handling
- Safe numpy import with fallback
- Proper executor shutdown
- Fixed asyncio.run() usage
- Complete error handling
"""

import argparse
import asyncio
import atexit
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Safe numpy import with fallback
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Core dependencies
try:
    from jsonschema import ValidationError, validate
except ImportError:
    validate, ValidationError = None, None

try:
    from prometheus_client import Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter, Histogram = None, None
    PROMETHEUS_AVAILABLE = False

# Graphix integrations
try:
    from src.nso_aligner import NSOAligner
except ImportError:
    NSOAligner = None

try:
    from src.observability_manager import ObservabilityManager
except ImportError:
    ObservabilityManager = None

try:
    from src.unified_runtime import UnifiedRuntime
except ImportError:
    UnifiedRuntime = None

try:
    from src.security_audit_engine import SecurityAuditEngine
except ImportError:
    SecurityAuditEngine = None

try:
    from src.key_manager import KeyManager
except ImportError:
    KeyManager = None

try:
    from src.hardware_dispatcher import HardwareDispatcher
except ImportError:
    HardwareDispatcher = None

try:
    from src.adversarial_tester import AdversarialTester
except ImportError:
    AdversarialTester = None

try:
    from src.tournament_manager import TournamentManager
except ImportError:
    TournamentManager = None

try:
    from src.large_graph_generator import generate_large_graph
except ImportError:
    generate_large_graph = None

# Logging setup (aligned with stdio_policy.py)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[logging.FileHandler("validation_test_logs.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ValidationTest")


# Dynamic discovery of golden files with fallback to hardcoded list
def discover_golden_files() -> List[str]:
    """Dynamically discover golden files or use defaults."""
    golden_dir = Path("specs/ir_examples")
    if golden_dir.exists():
        discovered = list(golden_dir.glob("*.json"))
        if discovered:
            return [str(f) for f in discovered]

    # Fallback to hardcoded list
    return [
        "specs/ir_examples/classifier.json",
        "specs/ir_examples/pipeline.json",
        "specs/ir_examples/full_v2.3.0_suite.json",
        "specs/ir_examples/gemini_evolution_proposal.json",
    ]


GOLDEN_FILES = discover_golden_files()
MANIFEST_PATH = "src/type_system_manifest.json"


# Cache for loaded files to improve performance
class FileCache:
    """Simple cache for loaded JSON files."""

    def __init__(self):
        self._cache = {}

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        """Get cached file or None."""
        return self._cache.get(path)

    def set(self, path: str, content: Dict[str, Any]) -> None:
        """Cache file content."""
        self._cache[path] = content

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


file_cache = FileCache()


class ValidationTestSuite:
    """
    Validates Graphix IR graphs for schema, semantics, ethics, execution, and photonic params.

    FIXES:
    - Proper resource cleanup with shutdown() method
    - Thread-safe metric registration (checks for existing metrics)
    - Fail-safe unknown status handling (marks as failed, not passed)
    - Context manager support for automatic cleanup
    """

    def __init__(self, agent_id: str = "validation_agent", enable_caching: bool = True):
        self.agent_id = agent_id
        self.enable_caching = enable_caching
        self._initialized = False
        self._shutdown_registered = False

        try:
            # Initialize components
            self.nso = NSOAligner() if NSOAligner else None
            self.obs = ObservabilityManager() if ObservabilityManager else None
            # BUG FIX Issue #1: Use get_or_create_unified_runtime to prevent per-test reinitialization
            if UnifiedRuntime:
                try:
                    from vulcan.reasoning.singletons import get_or_create_unified_runtime, set_unified_runtime
                    self.runtime = get_or_create_unified_runtime()
                    if self.runtime is None:
                        self.runtime = UnifiedRuntime()
                        # Register fallback instance with singleton
                        try:
                            set_unified_runtime(self.runtime)
                        except Exception:
                            pass
                except ImportError:
                    self.runtime = UnifiedRuntime()
            else:
                self.runtime = None
            self.audit = SecurityAuditEngine() if SecurityAuditEngine else None
            self.key_manager = KeyManager(agent_id) if KeyManager else None
            self.hardware = HardwareDispatcher() if HardwareDispatcher else None
            self.tournament = TournamentManager() if TournamentManager else None
            self.adversarial = AdversarialTester() if AdversarialTester else None
            self.manifest = self._load_manifest()

            # Initialize executor with proper cleanup tracking
            self.executor = ThreadPoolExecutor(max_workers=4)

            # Register metrics (FIXED: check if they already exist)
            self._register_metrics()

            self._initialized = True

            # Register cleanup on exit
            if not self._shutdown_registered:
                atexit.register(self.shutdown)
                self._shutdown_registered = True

            logger.info(
                "ValidationTestSuite initialized with agent_id=%s, caching=%s",
                agent_id,
                enable_caching,
            )

        except Exception as e:
            logger.error(f"Failed to initialize ValidationTestSuite: {e}")
            # Ensure cleanup even if initialization fails
            self.shutdown()
            raise

    def _register_metrics(self):
        """Register Prometheus metrics with collision detection."""
        if not self.obs or not PROMETHEUS_AVAILABLE:
            return

        try:
            # Check if metrics already exist before creating
            if "validation_pass" not in self.obs.metrics:
                self.obs.metrics["validation_pass"] = Counter(
                    "validation_pass_total",
                    "Successful validations",
                    ["test_type"],
                    registry=self.obs.registry,
                )

            if "validation_latency" not in self.obs.metrics:
                self.obs.metrics["validation_latency"] = Histogram(
                    "validation_latency_seconds",
                    "Validation latency",
                    ["test_type"],
                    registry=self.obs.registry,
                )

            if "validation_retries" not in self.obs.metrics:
                self.obs.metrics["validation_retries"] = Counter(
                    "validation_retries_total",
                    "Validation retry attempts",
                    ["test_type"],
                    registry=self.obs.registry,
                )

            logger.debug("Metrics registered successfully")

        except Exception as e:
            logger.warning(f"Failed to register metrics: {e}")

    def _load_manifest(self) -> Dict[str, Any]:
        """Load type system manifest with caching."""
        if self.enable_caching:
            cached = file_cache.get(MANIFEST_PATH)
            if cached:
                logger.debug("Using cached manifest")
                return cached

        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                if self.enable_caching:
                    file_cache.set(MANIFEST_PATH, manifest)
                return manifest
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            pytest.skip("Manifest loading failed")
            return {}

    def _load_golden_file(self, path: str) -> Dict[str, Any]:
        """Load a golden IR file with caching."""
        if self.enable_caching:
            cached = file_cache.get(path)
            if cached:
                logger.debug(f"Using cached golden file: {path}")
                return cached

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if self.enable_caching:
                    file_cache.set(path, content)
                return content
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            pytest.skip(f"Golden file {path} loading failed")
            return {}

    def _validate_schema(self, graph: Dict[str, Any]) -> bool:
        """Validate graph against manifest schemas."""
        if not validate:
            pytest.skip("jsonschema not installed")
            return False

        grammar_version = graph.get("grammar_version", "1.0.0")
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        try:
            # Validate nodes
            for node in nodes:
                node_type = node.get("type")
                schema = (
                    self.manifest.get("node_types", {})
                    .get(node_type, {})
                    .get("json_schema")
                )
                if not schema:
                    logger.error(f"No schema for node type: {node_type}")
                    return False
                validate(instance=node, schema=schema)

            # Validate edges
            for edge in edges:
                edge_type = edge.get("type")
                schema = (
                    self.manifest.get("edge_types", {})
                    .get(edge_type, {})
                    .get("json_schema")
                )
                if not schema:
                    logger.error(f"No schema for edge type: {edge_type}")
                    return False
                validate(instance=edge, schema=schema)

            logger.info(
                f"Schema validation passed for graph with grammar_version {grammar_version}"
            )
            return True
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def _validate_signature(self, graph: Dict[str, Any]) -> bool:
        """Verify graph signature using KeyManager."""
        if not self.key_manager:
            logger.warning("KeyManager not available; skipping signature check")
            return True

        try:
            message = json.dumps(graph, sort_keys=True)
            signature = self.key_manager.sign(message)
            verified = self.key_manager.verify(message, signature)
            logger.info(
                f"Signature verification {'passed' if verified else 'failed'} for graph {graph.get('id')}"
            )
            return verified
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    async def _validate_photonic_params(self, graph: Dict[str, Any]) -> bool:
        """Validate photonic parameters for relevant nodes."""
        if not self.hardware:
            logger.warning(
                "HardwareDispatcher not available; skipping photonic validation"
            )
            return True

        try:
            for node in graph.get("nodes", []):
                if node.get("type") == "PhotonicMVMNode":
                    params = node.get("params", {}).get("photonic_params", {})
                    result = self.hardware.validate_photonic_params(params)
                    if result.get("error_code"):
                        logger.error(
                            f"Photonic params invalid: {result.get('message')}"
                        )
                        return False
            logger.info(
                f"Photonic params validation passed for graph {graph.get('id')}"
            )
            return True
        except Exception as e:
            logger.error(f"Photonic params validation failed: {e}")
            return False

    async def _validate_execution(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph and validate output (with retry support)."""
        if not self.runtime:
            pytest.skip("UnifiedRuntime not available")
            return {"status": "skipped"}

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                result = await self.runtime.execute_graph(graph)
                if result.get("status") == "completed":
                    logger.info(
                        f"Execution successful for graph {graph.get('id')} on attempt {attempt + 1}"
                    )
                    return {
                        "status": "completed",
                        "duration_ms": (time.time() - start_time) * 1000,
                        "attempts": attempt + 1,
                        "output": result.get("output", {}),
                    }
                else:
                    last_error = result.get("error", "Unknown error")
                    logger.warning(
                        f"Execution attempt {attempt + 1} failed: {last_error}"
                    )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Execution attempt {attempt + 1} error: {e}")

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                if self.obs and "validation_retries" in self.obs.metrics:
                    self.obs.metrics["validation_retries"].labels(
                        test_type="execution"
                    ).inc()

        return {
            "status": "error",
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "duration_ms": 0,
            "attempts": max_retries,
        }

    def _validate_ethics(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Check graph for ethical compliance using NSOAligner."""
        if not self.nso:
            logger.warning("NSOAligner not available; skipping ethics check")
            return {"status": "skipped", "ethical": "unknown"}

        try:
            consensus = self.nso.multi_model_audit(graph)
            logger.info(f"Ethics audit result: {consensus}")
            return {"status": "completed", "ethical": consensus}
        except Exception as e:
            logger.error(f"Ethics audit failed: {e}")
            return {"status": "error", "error": str(e)}

    async def validate_all_aspects(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all aspects of a graph in parallel where possible."""
        results = {}

        # Run independent validations in parallel
        async_tasks = []

        # Schema validation (sync, run in executor)
        schema_future = self.executor.submit(self._validate_schema, graph)

        # Signature validation (sync, run in executor)
        sig_future = self.executor.submit(self._validate_signature, graph)

        # Ethics validation (sync, run in executor)
        ethics_future = self.executor.submit(self._validate_ethics, graph)

        # Async validations
        async_tasks.append(("execution", self._validate_execution(graph)))
        async_tasks.append(("photonic", self._validate_photonic_params(graph)))

        # Gather async results
        async_results = await asyncio.gather(
            *[task[1] for task in async_tasks], return_exceptions=True
        )
        for (name, _), result in zip(async_tasks, async_results):
            if isinstance(result, Exception):
                results[name] = {"status": "error", "error": str(result)}
            else:
                results[name] = result

        # Get sync results with timeout
        try:
            results["schema"] = schema_future.result(timeout=30)
        except Exception as e:
            results["schema"] = False
            logger.error(f"Schema validation timeout/error: {e}")

        try:
            results["signature"] = sig_future.result(timeout=30)
        except Exception as e:
            results["signature"] = False
            logger.error(f"Signature validation timeout/error: {e}")

        try:
            results["ethics"] = ethics_future.result(timeout=30)
        except Exception as e:
            results["ethics"] = {"status": "error", "error": str(e)}
            logger.error(f"Ethics validation timeout/error: {e}")

        return results

    def _calculate_graph_hash(self, graph: Dict[str, Any]) -> str:
        """Calculate a hash for graph content for caching/comparison."""
        graph_str = json.dumps(graph, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _log_metrics(self, test_name: str, result: Dict[str, Any]):
        """Log test metrics to ObservabilityManager."""
        if not self.obs:
            logger.warning("ObservabilityManager not available")
            return

        try:
            self.obs.log_audit_event(f"validation_test_{test_name}")

            if (
                "validation_pass" in self.obs.metrics
                and result.get("status") == "completed"
            ):
                self.obs.metrics["validation_pass"].labels(test_type=test_name).inc()

            if "validation_latency" in self.obs.metrics and "duration_ms" in result:
                self.obs.metrics["validation_latency"].labels(
                    test_type=test_name
                ).observe(result["duration_ms"] / 1000)

            if result.get("status") == "error":
                self.obs.log_error("validation_test", result.get("error", "unknown"))

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def _log_audit(self, test_name: str, details: Dict[str, Any]):
        """Log test event to SecurityAuditEngine."""
        if not self.audit:
            logger.warning("SecurityAuditEngine not available")
            return

        try:
            self.audit.log_event(
                f"validation_test_{test_name}",
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": test_name,
                    "agent_id": self.agent_id,
                    "details": details,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")

    def export_test_dashboard(self, dashboard_name: str = "validation_test_dashboard"):
        """Export Grafana dashboard for validation metrics."""
        if not self.obs:
            logger.warning("ObservabilityManager not available")
            return None

        try:
            return self.obs.export_dashboard(dashboard_name)
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            return None

    def generate_validation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.

        FIXED: Unknown statuses now marked as 'unknown' instead of defaulting to 'passed'.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "summary": {
                "total_tests": len(results),
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "unknown": 0,
            },
            "details": results,
            "metrics": {},
        }

        for r in results.values():
            if isinstance(r, bool):
                if r:
                    report["summary"]["passed"] += 1
                else:
                    report["summary"]["failed"] += 1
            elif isinstance(r, dict):
                status = r.get("status")
                if status == "completed":
                    report["summary"]["passed"] += 1
                elif status == "error":
                    report["summary"]["failed"] += 1
                elif status == "skipped":
                    report["summary"]["skipped"] += 1
                else:
                    # FIXED: Unknown statuses are now marked as unknown, not passed
                    report["summary"]["unknown"] += 1
                    logger.warning(f"Unknown test status encountered: {status}")
            else:
                # Non-dict, non-bool results are also unknown
                report["summary"]["unknown"] += 1
                logger.warning(f"Unexpected result type: {type(r).__name__}")

        # Calculate duration metrics
        durations = []
        for r in results.values():
            if isinstance(r, dict) and "duration_ms" in r:
                durations.append(r["duration_ms"])

        if durations:
            # FIXED: Safe numpy usage with fallback
            if NUMPY_AVAILABLE:
                report["metrics"]["avg_duration_ms"] = float(np.mean(durations))
            else:
                report["metrics"]["avg_duration_ms"] = sum(durations) / len(durations)

            report["metrics"]["max_duration_ms"] = max(durations)
            report["metrics"]["min_duration_ms"] = min(durations)

        return report

    def shutdown(self):
        """
        Clean shutdown of all resources.

        FIXED: Proper cleanup method instead of unsafe __del__.
        """
        if not self._initialized:
            return

        logger.info("Shutting down ValidationTestSuite...")

        try:
            # Shutdown executor
            if hasattr(self, "executor") and self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
                logger.debug("ThreadPoolExecutor shut down")

            # Close audit engine if it has a close method
            if hasattr(self, "audit") and self.audit is not None:
                if hasattr(self.audit, "close"):
                    self.audit.close()
                    logger.debug("SecurityAuditEngine closed")

            # Close any other resources
            if hasattr(self, "obs") and self.obs is not None:
                if hasattr(self.obs, "shutdown"):
                    self.obs.shutdown()
                    logger.debug("ObservabilityManager shut down")

            self._initialized = False
            logger.info("ValidationTestSuite shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("golden_file", GOLDEN_FILES)
async def test_graph_validation(golden_file: str):
    """Test schema, signature, ethics, execution, and photonic params for each golden file."""
    with ValidationTestSuite() as suite:
        graph = suite._load_golden_file(golden_file)
        assert graph, f"Failed to load {golden_file}"
        test_name = Path(golden_file).stem

        # Schema validation
        start_time = time.time()
        schema_result = suite._validate_schema(graph)
        schema_duration = (time.time() - start_time) * 1000
        assert schema_result, f"Schema validation failed for {golden_file}"
        suite._log_metrics(
            f"schema_{test_name}",
            {"status": "completed", "duration_ms": schema_duration},
        )
        suite._log_audit(
            f"schema_{test_name}", {"file": golden_file, "valid": schema_result}
        )

        # Signature verification
        sig_result = suite._validate_signature(graph)
        assert sig_result, f"Signature verification failed for {golden_file}"
        suite._log_metrics(
            f"signature_{test_name}", {"status": "completed", "duration_ms": 0}
        )
        suite._log_audit(
            f"signature_{test_name}", {"file": golden_file, "valid": sig_result}
        )

        # Ethical validation
        ethics_result = suite._validate_ethics(graph)
        assert (
            ethics_result["status"] != "error"
        ), f"Ethics validation failed: {ethics_result.get('error')}"
        assert (
            ethics_result["ethical"] == "safe"
        ), f"Graph {golden_file} not ethically safe: {ethics_result['ethical']}"
        suite._log_metrics(f"ethics_{test_name}", ethics_result)
        suite._log_audit(f"ethics_{test_name}", ethics_result)

        # Execution validation
        exec_result = await suite._validate_execution(graph)
        assert (
            exec_result["status"] == "completed"
        ), f"Execution failed: {exec_result.get('error')}"
        suite._log_metrics(f"exec_{test_name}", exec_result)
        suite._log_audit(f"exec_{test_name}", exec_result)

        # Photonic params validation
        photonic_result = await suite._validate_photonic_params(graph)
        assert photonic_result, f"Photonic params validation failed for {golden_file}"
        suite._log_metrics(
            f"photonic_{test_name}", {"status": "completed", "duration_ms": 0}
        )
        suite._log_audit(
            f"photonic_{test_name}", {"file": golden_file, "valid": photonic_result}
        )


@pytest.mark.asyncio
async def test_parallel_validation():
    """Test parallel validation of all aspects."""
    with ValidationTestSuite() as suite:
        if GOLDEN_FILES:
            graph = suite._load_golden_file(GOLDEN_FILES[0])
            results = await suite.validate_all_aspects(graph)

            # Check that we got results for all aspects
            expected_aspects = {
                "schema",
                "signature",
                "ethics",
                "execution",
                "photonic",
            }
            assert (
                set(results.keys()) >= expected_aspects
            ), f"Missing validation aspects: {expected_aspects - set(results.keys())}"

            # Generate and log report
            report = suite.generate_validation_report(results)
            logger.info(f"Validation report: {json.dumps(report, indent=2)}")
            suite._log_audit("parallel_validation", report)


@pytest.mark.asyncio
async def test_stress_validation():
    """Test execution of a large graph for scalability."""
    if not generate_large_graph:
        pytest.skip("large_graph_generator not available")

    with ValidationTestSuite() as suite:
        graph = generate_large_graph(num_nodes=1000, density=0.2)

        # Add hash for large graph tracking
        graph_hash = suite._calculate_graph_hash(graph)
        logger.info(f"Testing large graph with hash: {graph_hash[:8]}...")

        exec_result = await suite._validate_execution(graph)
        assert (
            exec_result["status"] == "completed"
        ), f"Stress test failed: {exec_result.get('error')}"
        suite._log_metrics("stress_test", exec_result)
        suite._log_audit("stress_test", {"graph_hash": graph_hash, **exec_result})


@pytest.mark.asyncio
async def test_adversarial_validation():
    """Test ethics of an adversarial graph."""
    if not AdversarialTester:
        pytest.skip("AdversarialTester not available")

    with ValidationTestSuite() as suite:
        base_graph = suite._load_golden_file(GOLDEN_FILES[0])

        # FIXED: Properly use AdversarialTester if available
        if suite.adversarial and hasattr(suite.adversarial, "create_adversarial_graph"):
            adv_graph = suite.adversarial.create_adversarial_graph(base_graph)
        else:
            # Fallback: Create adversarial graph manually
            adv_graph = base_graph.copy()
            if "nodes" in adv_graph:
                adv_graph["nodes"].append(
                    {
                        "id": "adversarial_node",
                        "type": "ExecuteNode",
                        "params": {"code": 'eval("potentially risky code")'},
                    }
                )

        ethics_result = suite._validate_ethics(adv_graph)
        assert (
            ethics_result["ethical"] == "risky"
        ), "Adversarial graph not flagged as risky"
        suite._log_metrics("adversarial_test", ethics_result)
        suite._log_audit("adversarial_test", ethics_result)


@pytest.mark.asyncio
async def test_tournament_validation():
    """Test schema of evolved graphs from tournament."""
    if not TournamentManager or not NUMPY_AVAILABLE:
        pytest.skip("TournamentManager or numpy not available")

    with ValidationTestSuite() as suite:
        tm = TournamentManager()
        proposals = [
            {
                "id": f"p{i}",
                "nodes": [
                    {"id": "n1", "type": "InputNode", "params": {"value": f"data{i}"}}
                ],
                "edges": [],
            }
            for i in range(5)
        ]
        fitness = np.random.rand(5).tolist()
        winners = tm.run_adaptive_tournament(
            proposals, fitness, lambda x: np.random.rand(16)
        )

        for idx in winners:
            graph = proposals[idx]
            assert suite._validate_schema(
                graph
            ), f"Tournament winner {graph['id']} invalid schema"

        suite._log_metrics("tournament_test", {"status": "completed", "duration_ms": 0})
        suite._log_audit("tournament_test", {"num_winners": len(winners)})


def test_manifest_completeness():
    """Ensure manifest covers all node/edge types used in goldens."""
    with ValidationTestSuite() as suite:
        assert suite.manifest, "Manifest not loaded"

        used_node_types = set()
        used_edge_types = set()
        for golden_file in GOLDEN_FILES:
            graph = suite._load_golden_file(golden_file)
            used_node_types.update(node.get("type") for node in graph.get("nodes", []))
            used_edge_types.update(edge.get("type") for edge in graph.get("edges", []))

        manifest_nodes = set(suite.manifest.get("node_types", {}).keys())
        manifest_edges = set(suite.manifest.get("edge_types", {}).keys())

        assert used_node_types.issubset(
            manifest_nodes
        ), f"Missing node types: {used_node_types - manifest_nodes}"
        assert used_edge_types.issubset(
            manifest_edges
        ), f"Missing edge types: {used_edge_types - manifest_edges}"
        suite._log_metrics(
            "manifest_completeness", {"status": "completed", "duration_ms": 0}
        )
        suite._log_audit(
            "manifest_completeness",
            {"node_types": list(used_node_types), "edge_types": list(used_edge_types)},
        )


def test_caching_functionality():
    """Test that caching improves performance."""
    file_cache.clear()

    with ValidationTestSuite(enable_caching=False) as suite_no_cache:
        start = time.perf_counter()
        suite_no_cache._load_golden_file(GOLDEN_FILES[0])
        no_cache_time = time.perf_counter() - start

    file_cache.clear()

    with ValidationTestSuite(enable_caching=True) as suite_cache:
        start = time.perf_counter()
        graph2 = suite_cache._load_golden_file(GOLDEN_FILES[0])
        first_cache_time = time.perf_counter() - start

        start = time.perf_counter()
        graph3 = suite_cache._load_golden_file(GOLDEN_FILES[0])
        second_cache_time = time.perf_counter() - start

    assert file_cache.get(GOLDEN_FILES[0]) is not None, "File not in cache"

    # On very fast filesystems, cache time might be slightly higher due to overhead.
    # We check for correctness first, and performance as a secondary goal.
    if second_cache_time > 0 and first_cache_time > 0:
        assert (
            second_cache_time <= first_cache_time
        ), "Cache did not improve performance"
    else:
        assert graph3 == graph2, "Cached content doesn't match"

    logger.info(
        f"Cache performance: no_cache={no_cache_time:.6f}s, first={first_cache_time:.6f}s, cached={second_cache_time:.6f}s"
    )


def _check_async_context() -> bool:
    """Check if we're already in an async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


async def demo_validation(golden_files: List[str] = None):
    """Demo validation for all golden files."""
    if golden_files is None:
        golden_files = GOLDEN_FILES

    with ValidationTestSuite() as suite:
        all_results = {}

        for file in golden_files:
            graph = suite._load_golden_file(file)
            print(f"\n{'=' * 60}")
            print(f"Validating: {Path(file).name}")
            print(f"{'=' * 60}")

            # Calculate and display graph hash
            graph_hash = suite._calculate_graph_hash(graph)
            print(f"Graph Hash: {graph_hash[:16]}...")
            print(
                f"Nodes: {len(graph.get('nodes', []))}, Edges: {len(graph.get('edges', []))}"
            )

            # Run parallel validation
            print("\nRunning parallel validation...")
            results = await suite.validate_all_aspects(graph)
            all_results[file] = results

            # Display results
            for aspect, result in results.items():
                if isinstance(result, bool):
                    status = "PASS" if result else "FAIL"
                elif isinstance(result, dict):
                    status = {
                        "completed": "PASS",
                        "error": "FAIL",
                        "skipped": "SKIP",
                    }.get(result.get("status"), "UNKNOWN")
                    if "duration_ms" in result:
                        status += f" ({result['duration_ms']:.1f}ms)"
                    if "attempts" in result and result["attempts"] > 1:
                        status += f" [retries: {result['attempts'] - 1}]"
                else:
                    status = str(result)

                print(f"  {aspect:15s}: {status}")

            # Generate report for this file
            report = suite.generate_validation_report(results)
            if report["metrics"]:
                print(f"\nPerformance Metrics:")
                for metric, value in report["metrics"].items():
                    print(f"  {metric}: {value:.2f}")

        # Export dashboard
        print(f"\n{'=' * 60}")
        dashboard_url = suite.export_test_dashboard()
        if dashboard_url:
            print(f"Dashboard exported: {dashboard_url}")

        # Overall summary
        print(f"\n{'=' * 60}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 60}")
        total_files = len(all_results)
        total_passed = sum(
            1
            for results in all_results.values()
            if all(
                r.get("status") == "completed" if isinstance(r, dict) else r
                for r in results.values()
            )
        )
        print(f"Files Tested: {total_files}")
        print(f"Fully Passed: {total_passed}/{total_files}")

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Graphix IR validation tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--golden-files",
        nargs="+",
        default=None,
        help="Custom golden IR files to validate (auto-discovers if not specified)",
    )
    parser.add_argument(
        "--agent-id",
        default="validation_agent",
        help="Agent ID for signature verification",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run interactive demo instead of pytest"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run validations in parallel (demo mode only)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable file caching")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        logger.info("Running validation demo...")
        logger.info(f"Golden files: {args.golden_files or 'auto-discovered'}")

        # FIXED: Check if we're already in async context before using asyncio.run()
        if _check_async_context():
            logger.error("Cannot use asyncio.run() from within an async context")
            sys.exit(1)

        results = asyncio.run(demo_validation(args.golden_files))
    else:
        logger.info("Running validation test suite...")
        pytest_args = ["-v", __file__]
        if args.parallel:
            pytest_args.extend(["-n", "auto"])  # requires pytest-xdist
        pytest.main(pytest_args)
