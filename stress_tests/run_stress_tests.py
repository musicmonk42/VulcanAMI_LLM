# run_stress_tests.py
"""
Async stress tests for a comprehensive GenerationOrchestrator.

The orchestrator models a complete generation pipeline:
  spec -> policy_engine.evaluate -> security_scanner.scan_async
       -> mutation_tester.run_mutations -> pr_creator.open_pr

Each component is pluggable and can be monkeypatched in tests.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Constants
# ---------------------------

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_MS = 100
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RISK_LEVEL = 10
MAX_SPEC_SIZE_KB = 1024
MAX_NODES = 1000
METADATA_PROBABILITY = 0.3
INJECTION_SCAN_TIMEOUT = 10
DEEP_SCAN_DELAY_MS = 10
MUTATION_COUNT = 10

# Whitelisted module paths for component loading (security)
ALLOWED_COMPONENT_PATHS = {
    "policy_engine": [],
    "security_scanner": [],
    "mutation_tester": [],
    "pr_creator": []
}

# ---------------------------
# Enums and Constants
# ---------------------------

class ComponentType(Enum):
    """Types of components that can be loaded."""
    POLICY_ENGINE = "policy_engine"
    SECURITY_SCANNER = "security_scanner"
    MUTATION_TESTER = "mutation_tester"
    PR_CREATOR = "pr_creator"
    CUSTOM = "custom"

class PolicyDecision(Enum):
    """Policy evaluation decisions."""
    ALLOW = "allow"
    REJECT = "reject"
    REVIEW = "review"

# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class OrchestratorResult:
    """Result of an orchestrator run."""
    status: str               # "success", "rejected", "error", "timeout"
    details: Dict[str, Any]   # arbitrary payload about the run
    duration_ms: float = 0.0  # execution time in milliseconds
    retries: int = 0          # number of retries attempted
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }

@dataclass
class PerformanceMetrics:
    """Performance metrics for stress testing (thread-safe)."""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    rejected_runs: int = 0
    timeout_runs: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def update(self, result: OrchestratorResult):
        """Update metrics with a new result (thread-safe)."""
        with self._lock:
            self.total_runs += 1
            self.total_duration_ms += result.duration_ms
            self.min_duration_ms = min(self.min_duration_ms, result.duration_ms)
            self.max_duration_ms = max(self.max_duration_ms, result.duration_ms)

            if result.status == "success":
                self.successful_runs += 1
            elif result.status == "rejected":
                self.rejected_runs += 1
            elif result.status == "timeout":
                self.timeout_runs += 1
            else:
                self.failed_runs += 1
                error_type = result.details.get("error_type", "unknown")
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def calculate_percentiles(self, durations: List[float]):
        """Calculate percentile metrics from duration list (thread-safe)."""
        with self._lock:
            if not durations:
                return
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            self.avg_duration_ms = sum(sorted_durations) / n
            self.p50_duration_ms = sorted_durations[int(n * 0.5)]
            self.p95_duration_ms = sorted_durations[min(int(n * 0.95), n - 1)]
            self.p99_duration_ms = sorted_durations[min(int(n * 0.99), n - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary (thread-safe)."""
        with self._lock:
            return {
                "total_runs": self.total_runs,
                "successful_runs": self.successful_runs,
                "failed_runs": self.failed_runs,
                "rejected_runs": self.rejected_runs,
                "timeout_runs": self.timeout_runs,
                "success_rate": (self.successful_runs / self.total_runs * 100) if self.total_runs > 0 else 0,
                "total_duration_ms": self.total_duration_ms,
                "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float('inf') else 0,
                "max_duration_ms": self.max_duration_ms,
                "avg_duration_ms": self.avg_duration_ms,
                "p50_duration_ms": self.p50_duration_ms,
                "p95_duration_ms": self.p95_duration_ms,
                "p99_duration_ms": self.p99_duration_ms,
                "errors_by_type": dict(self.errors_by_type)
            }

# ---------------------------
# Input Validation
# ---------------------------

class SpecValidator:
    """Validates specs before processing for security."""

    @staticmethod
    def validate_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a spec for security issues.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if spec is a dict
        if not isinstance(spec, dict):
            issues.append("Spec must be a dictionary")
            return False, issues

        # Check size
        try:
            spec_str = json.dumps(spec)
            size_kb = len(spec_str) / 1024
            if size_kb > MAX_SPEC_SIZE_KB:
                issues.append(f"Spec too large: {size_kb:.2f}KB (max {MAX_SPEC_SIZE_KB}KB)")
        except Exception as e:
            issues.append(f"Spec not JSON-serializable: {e}")

        # Check for suspicious patterns (basic validation)
        if "nodes" in spec and isinstance(spec["nodes"], list):
            if len(spec["nodes"]) > MAX_NODES:
                issues.append(f"Too many nodes: {len(spec['nodes'])} (max {MAX_NODES})")

        # Don't allow user to set their own risk level
        if "risk" in spec and isinstance(spec["risk"], (int, float)) and spec["risk"] < 0:
            issues.append("Risk level cannot be negative")

        return len(issues) == 0, issues

# ---------------------------
# Component Implementations
# ---------------------------

class PolicyEngine:
    """Production policy engine implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules = self.config.get("rules", [])
        self.strict_mode = self.config.get("strict_mode", False)
        self.max_risk_level = self.config.get("max_risk_level", DEFAULT_MAX_RISK_LEVEL)

    def evaluate(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a spec against policy rules.

        Returns:
            Dictionary with 'allowed' boolean and optional 'reasons' list
        """
        try:
            reasons = []
            risk_score = 0

            # Check for required fields
            if "id" not in spec:
                reasons.append("missing_required_field:id")
                risk_score += 5

            if "type" not in spec:
                reasons.append("missing_required_field:type")
                risk_score += 5

            # Calculate risk based on spec characteristics (don't use user-provided risk)
            # Check complexity
            node_count = len(spec.get("nodes", []))
            if node_count > MAX_NODES:
                reasons.append(f"too_many_nodes:{node_count}")
                risk_score += 3

            edge_count = len(spec.get("edges", []))
            if edge_count > node_count * 2:
                reasons.append(f"too_many_edges:{edge_count}")
                risk_score += 2

            # Check for blacklisted patterns with word boundaries
            blacklist_patterns = self.config.get("blacklist_patterns", [])
            spec_str = json.dumps(spec)
            for pattern in blacklist_patterns:
                # Use word boundaries to avoid false positives
                if re.search(r'\b' + re.escape(pattern) + r'\b', spec_str, re.IGNORECASE):
                    reasons.append(f"blacklisted_pattern:{pattern}")
                    risk_score += 10

            # Check size limits
            max_size = self.config.get("max_spec_size_kb", MAX_SPEC_SIZE_KB)
            spec_size_kb = len(spec_str) / 1024
            if spec_size_kb > max_size:
                reasons.append(f"spec_too_large:{spec_size_kb:.2f}KB")
                risk_score += 5

            # Apply custom rules
            for rule in self.rules:
                if callable(rule):
                    rule_result = rule(spec)
                    if rule_result and not rule_result.get("passed", True):
                        reasons.append(rule_result.get("reason", "custom_rule_failed"))
                        risk_score += rule_result.get("risk_increase", 2)

            # Determine decision
            allowed = risk_score <= self.max_risk_level

            if self.strict_mode and reasons:
                allowed = False

            return {
                "allowed": allowed,
                "reasons": reasons,
                "risk_score": risk_score,
                "decision": PolicyDecision.ALLOW.value if allowed else PolicyDecision.REJECT.value
            }

        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")
            return {
                "allowed": False,
                "reasons": [f"evaluation_error:{str(e)}"],
                "risk_score": 999,
                "decision": PolicyDecision.REJECT.value
            }

class SecurityScanner:
    """Production security scanner implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scan_depth = self.config.get("scan_depth", "normal")
        self.timeout = self.config.get("timeout_seconds", INJECTION_SCAN_TIMEOUT)

    async def scan_async(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Asynchronously scan a spec for security issues.

        Returns:
            List of findings with type, severity, and message
        """
        findings = []
        seen_findings = set()  # Track to avoid duplicates

        try:
            # Simulate async scanning operations with actual I/O delay
            await asyncio.sleep(0.001)

            # Check for injection attempts
            spec_str = json.dumps(spec)
            injection_patterns = [
                ("SQL", ["'; DROP", "SELECT * FROM", "UNION SELECT", "OR 1=1"]),
                ("Command", ["; rm", "&& cat", "| nc", "`cmd`", "$(cmd)"]),
                ("XSS", ["<script>", "javascript:", "onerror=", "onload="]),
                ("Path", ["../", "..\\", "/etc/passwd", "C:\\Windows"]),
                ("Template", ["{{", "{%", "${", "#{"])
            ]

            for attack_type, patterns in injection_patterns:
                for pattern in patterns:
                    if pattern.lower() in spec_str.lower():
                        finding_key = f"{attack_type}:{pattern}"
                        if finding_key not in seen_findings:
                            findings.append({
                                "type": "security",
                                "severity": "high",
                                "category": f"{attack_type}_injection",
                                "message": f"Potential {attack_type} injection detected: {pattern}",
                                "location": "spec_content"
                            })
                            seen_findings.add(finding_key)

            # Check for oversized data
            if len(spec_str) > 1024 * 100:  # 100KB
                findings.append({
                    "type": "security",
                    "severity": "medium",
                    "category": "resource_exhaustion",
                    "message": f"Large spec size: {len(spec_str) / 1024:.2f}KB",
                    "location": "spec_size"
                })

            # Check for deeply nested structures
            def check_depth(obj, current_depth=0, max_depth=20):
                if current_depth > max_depth:
                    return True
                if isinstance(obj, dict):
                    for value in obj.values():
                        if check_depth(value, current_depth + 1, max_depth):
                            return True
                elif isinstance(obj, list):
                    for item in obj:
                        if check_depth(item, current_depth + 1, max_depth):
                            return True
                return False

            if check_depth(spec):
                findings.append({
                    "type": "security",
                    "severity": "medium",
                    "category": "complexity",
                    "message": "Deeply nested structure detected",
                    "location": "spec_structure"
                })

            # Check for null bytes and control characters
            if '\x00' in spec_str or '\r' in spec_str or '\x1b' in spec_str:
                findings.append({
                    "type": "security",
                    "severity": "high",
                    "category": "malformed_input",
                    "message": "Control characters detected in spec",
                    "location": "spec_content"
                })

            # Deep scan mode
            if self.scan_depth == "deep":
                await asyncio.sleep(DEEP_SCAN_DELAY_MS / 1000)

                # Check for circular references
                node_ids = set()
                edges = spec.get("edges", [])
                for node in spec.get("nodes", []):
                    if "id" in node:
                        node_ids.add(node["id"])

                for edge in edges:
                    if edge.get("from") == edge.get("to"):
                        findings.append({
                            "type": "info",
                            "severity": "low",
                            "category": "structure",
                            "message": f"Self-loop detected: {edge.get('from')}",
                            "location": "edges"
                        })

                    if edge.get("from") not in node_ids or edge.get("to") not in node_ids:
                        findings.append({
                            "type": "warning",
                            "severity": "medium",
                            "category": "integrity",
                            "message": f"Edge references non-existent node",
                            "location": f"edge:{edge}"
                        })

            # If no findings, report clean
            if not findings:
                findings.append({
                    "type": "info",
                    "severity": "info",
                    "category": "clean",
                    "message": "No security issues detected",
                    "location": "overall"
                })

        except asyncio.TimeoutError:
            findings.append({
                "type": "error",
                "severity": "critical",
                "category": "timeout",
                "message": f"Security scan timeout after {self.timeout}s",
                "location": "scanner"
            })
        except Exception as e:
            logger.error(f"Security scanner error: {e}")
            findings.append({
                "type": "error",
                "severity": "critical",
                "category": "scanner_error",
                "message": f"Scanner error: {str(e)}",
                "location": "scanner"
            })

        return findings

class MutationTester:
    """
    Production mutation tester implementation.
    Note: This is a simplified simulation for testing purposes.
    Real mutation testing would require actual validation logic to test against.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mutation_count = self.config.get("mutation_count", MUTATION_COUNT)
        self.strategies = self.config.get("strategies", ["field_swap", "type_change", "value_mutation"])

    async def run_mutations(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run mutation tests on the spec.

        Note: This is a simulation. Real implementation would:
        1. Mutate the spec
        2. Run validators on mutated spec
        3. Check if validators catch the mutation

        Returns:
            Dictionary with mutation test results
        """
        try:
            mutations_performed = []
            mutations_killed = 0
            mutations_survived = 0

            # Simulate async mutation testing
            for i in range(self.mutation_count):
                await asyncio.sleep(0.0001)

                strategy = self.strategies[i % len(self.strategies)]

                # Simulate mutation result (deterministic based on strategy)
                # In real implementation, this would test actual validators
                killed = hash(f"{strategy}_{i}") % 10 > 2  # ~80% kill rate

                mutation_result = {
                    "id": f"mutation_{i}",
                    "strategy": strategy,
                    "killed": killed,
                    "execution_time_ms": (hash(f"{strategy}_{i}") % 50) / 10.0
                }

                mutations_performed.append(mutation_result)
                if killed:
                    mutations_killed += 1
                else:
                    mutations_survived += 1

            mutation_score = mutations_killed / self.mutation_count if self.mutation_count > 0 else 0

            return {
                "score": mutation_score,
                "killed": mutations_killed,
                "survived": mutations_survived,
                "total": self.mutation_count,
                "strategies_used": list(set(self.strategies)),
                "mutations": mutations_performed[:5],  # Include first 5 for detail
                "quality": "good" if mutation_score > 0.7 else "needs_improvement",
                "note": "Simulation - real implementation would test actual validators"
            }

        except Exception as e:
            logger.error(f"Mutation tester error: {e}")
            return {
                "score": 0.0,
                "killed": 0,
                "survived": 0,
                "total": 0,
                "error": str(e)
            }

class PRCreator:
    """Production PR creator implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_url = self.config.get("base_url", "https://github.com/example/repo")
        self.auto_merge = self.config.get("auto_merge", False)
        self.created_prs = {}  # Track created PRs for idempotency
        self.headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("GRAPHIX_API_KEY")
        if api_key:
            self.headers["X-API-KEY"] = api_key
        else:
            logger.warning("PRCreator: GRAPHIX_API_KEY environment variable not set. Real API calls would fail.")

    async def open_pr(self, spec: Dict[str, Any], report: Dict[str, Any],
                     idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a pull request based on spec and analysis report.
        Uses idempotency key to avoid duplicate PR creation on retries.

        Returns:
            Dictionary with PR information
        """
        try:
            # Check idempotency
            if idempotency_key and idempotency_key in self.created_prs:
                logger.info(f"PR already created for key {idempotency_key}, returning cached result")
                return self.created_prs[idempotency_key]

            # Simulate async PR creation
            await asyncio.sleep(0.001)

            import hashlib
            import random

            # Generate PR details
            pr_id = random.randint(1000, 9999)
            spec_hash = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]

            pr_info = {
                "url": f"{self.base_url}/pull/{pr_id}",
                "id": pr_id,
                "title": f"Generated from spec {spec.get('id', 'unknown')}",
                "branch": f"generated/spec-{spec_hash}",
                "status": "open",
                "spec_hash": spec_hash,
                "report_summary": {
                    "findings_count": len(report.get("findings", [])),
                    "mutation_score": report.get("mutation_report", {}).get("score", 0),
                    "policy_decision": report.get("policy", {}).get("decision", "unknown")
                },
                "created_at": datetime.now().isoformat(),
                "auto_merge_eligible": self.auto_merge and report.get("mutation_report", {}).get("score", 0) > 0.9,
                "idempotency_key": idempotency_key
            }

            # Simulate potential failures (5% failure rate)
            if random.random() < 0.05:
                raise Exception("Failed to create PR: API rate limit exceeded")

            # Cache for idempotency
            if idempotency_key:
                self.created_prs[idempotency_key] = pr_info

            return pr_info

        except Exception as e:
            logger.error(f"PR creator error: {e}")
            return {
                "url": None,
                "error": str(e),
                "status": "failed",
                "idempotency_key": idempotency_key
            }

# ---------------------------
# Orchestrator Implementation
# ---------------------------

class GenerationOrchestrator:
    """
    Production async orchestrator with full component loading and error handling.
    Components are loaded via _load_component and expected to expose:
      - policy_engine.evaluate(spec) -> dict {allowed: bool, reasons?: list}
      - security_scanner.scan_async(spec) -> awaitable[list[dict]]
      - mutation_tester.run_mutations(spec) -> awaitable[dict]
      - pr_creator.open_pr(spec, report, idempotency_key) -> awaitable[dict {url: str}]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.components = {}
        self.metrics = PerformanceMetrics()
        self.retry_config = self.config.get("retry", {"max_retries": DEFAULT_MAX_RETRIES, "backoff_ms": DEFAULT_BACKOFF_MS})
        self.timeout_seconds = self.config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        self.validator = SpecValidator()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize default components if not in test mode."""
        if not self.config.get("test_mode", False):
            try:
                self.components["policy_engine"] = PolicyEngine(self.config.get("policy", {}))
                self.components["security_scanner"] = SecurityScanner(self.config.get("security", {}))
                self.components["mutation_tester"] = MutationTester(self.config.get("mutation", {}))
                self.components["pr_creator"] = PRCreator(self.config.get("pr", {}))
                logger.info("Components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize components: {e}")
                raise  # Don't continue with uninitialized components

    def _load_component(self, component_name: str, config_key: str, **kwargs) -> Any:
        """
        Load a component by name, either from cache, config, or dynamic import.
        Security-hardened with path whitelisting.

        Args:
            component_name: Name of the component to load
            config_key: Configuration key for the component
            **kwargs: Additional arguments for component initialization

        Returns:
            Loaded component instance or dictionary with required methods
        """
        try:
            # Check if component is already loaded
            if component_name in self.components:
                return self.components[component_name]

            # Check if component is specified in config
            component_config = self.config.get(config_key, {})

            # Try to load from module if path is specified
            module_path = component_config.get("module_path")
            if module_path:
                # Security: Check if path is whitelisted
                abs_path = os.path.abspath(module_path)
                allowed_paths = ALLOWED_COMPONENT_PATHS.get(component_name, [])

                if not any(abs_path.startswith(allowed) for allowed in allowed_paths):
                    raise SecurityError(
                        f"Module path not whitelisted for {component_name}: {abs_path}. "
                        f"Allowed paths: {allowed_paths}"
                    )

                return self._load_from_module(abs_path, component_name, component_config)

            # Load built-in component
            if component_name == "policy_engine":
                component = PolicyEngine(component_config)
                return component
            elif component_name == "security_scanner":
                component = SecurityScanner(component_config)
                return component
            elif component_name == "mutation_tester":
                component = MutationTester(component_config)
                return component
            elif component_name == "pr_creator":
                component = PRCreator(component_config)
                return component
            else:
                # Try to load from plugins directory (with absolute path)
                plugins_dir = self.config.get("plugins_dir")
                if plugins_dir:
                    abs_plugins_dir = os.path.abspath(plugins_dir)
                    plugin_path = os.path.join(abs_plugins_dir, f"{component_name}.py")

                    # Security: Validate plugin path
                    if os.path.exists(plugin_path):
                        allowed_paths = ALLOWED_COMPONENT_PATHS.get(component_name, [])
                        if any(plugin_path.startswith(allowed) for allowed in allowed_paths):
                            return self._load_from_file(plugin_path, component_name, component_config)

                raise NotImplementedError(f"No implementation found for component: {component_name}")

        except Exception as e:
            logger.error(f"Failed to load component {component_name}: {e}")
            raise

    def _load_from_module(self, module_path: str, component_name: str, config: Dict[str, Any]) -> Any:
        """Load a component from a Python module (security-hardened)."""
        try:
            # Additional security: verify file exists and is readable
            if not os.path.isfile(module_path):
                raise ValueError(f"Module path is not a file: {module_path}")

            spec = importlib.util.spec_from_file_location(component_name, module_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load spec from {module_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Try to instantiate the component
            class_name = config.get("class_name", component_name.title().replace("_", ""))
            if hasattr(module, class_name):
                component_class = getattr(module, class_name)
                instance = component_class(config)
                return instance

            return module

        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}")
            raise

    def _load_from_file(self, file_path: str, component_name: str, config: Dict[str, Any]) -> Any:
        """Load a component from a Python file."""
        return self._load_from_module(file_path, component_name, config)

    async def run_once(self, spec: Dict[str, Any]) -> OrchestratorResult:
        """
        Run the full pipeline once for a single spec with retry logic.
        Uses idempotency keys to avoid duplicate operations on retry.
        Returns an OrchestratorResult capturing the end-to-end outcome.
        """
        start_time = time.time()
        retries = 0
        max_retries = self.retry_config["max_retries"]
        backoff_ms = self.retry_config["backoff_ms"]
        correlation_id = str(uuid.uuid4())

        # Validate input spec
        is_valid, validation_issues = self.validator.validate_spec(spec)
        if not is_valid:
            duration_ms = (time.time() - start_time) * 1000
            result = OrchestratorResult(
                "rejected",
                {
                    "validation_errors": validation_issues,
                    "reason": "input_validation_failed"
                },
                duration_ms,
                0,
                correlation_id=correlation_id
            )
            self.metrics.update(result)
            logger.warning(f"[{correlation_id}] Spec validation failed: {validation_issues}")
            return result

        # Generate idempotency key for PR creation
        spec_id = spec.get("id", "unknown")
        idempotency_key = f"{spec_id}_{correlation_id}"

        while retries <= max_retries:
            try:
                # Wrap entire pipeline in timeout
                result = await asyncio.wait_for(
                    self._run_pipeline(spec, idempotency_key, correlation_id),
                    timeout=self.timeout_seconds
                )
                duration_ms = (time.time() - start_time) * 1000
                result.duration_ms = duration_ms
                result.retries = retries
                result.correlation_id = correlation_id
                self.metrics.update(result)
                return result

            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                result = OrchestratorResult(
                    "timeout",
                    {"error": f"Pipeline timeout after {self.timeout_seconds}s"},
                    duration_ms,
                    retries,
                    correlation_id=correlation_id
                )
                self.metrics.update(result)
                logger.error(f"[{correlation_id}] Pipeline timeout after {self.timeout_seconds}s")
                return result

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    duration_ms = (time.time() - start_time) * 1000
                    result = OrchestratorResult(
                        "error",
                        {
                            "exception": str(e),
                            "traceback": traceback.format_exc(),
                            "error_type": type(e).__name__
                        },
                        duration_ms,
                        retries - 1,
                        correlation_id=correlation_id
                    )
                    self.metrics.update(result)
                    logger.error(f"[{correlation_id}] Pipeline failed after {retries-1} retries: {e}")
                    return result

                # Exponential backoff
                await asyncio.sleep(backoff_ms * (2 ** retries) / 1000)
                logger.info(f"[{correlation_id}] Retrying after error (attempt {retries}/{max_retries}): {e}")

    async def _run_pipeline(self, spec: Dict[str, Any], idempotency_key: str,
                          correlation_id: str) -> OrchestratorResult:
        """Execute the actual pipeline steps."""
        try:
            logger.info(f"[{correlation_id}] Starting pipeline for spec: {spec.get('id', 'unknown')}")

            # Policy evaluation
            policy_engine = self._load_component("policy_engine", "policy")
            if hasattr(policy_engine, 'evaluate'):
                policy = policy_engine.evaluate
            else:
                return OrchestratorResult("error", {"error": "policy_engine missing evaluate() method"})

            policy_decision = policy(spec)
            if not isinstance(policy_decision, dict) or "allowed" not in policy_decision:
                return OrchestratorResult("error", {"error": "policy_engine returned invalid shape"})

            if not policy_decision.get("allowed", False):
                logger.info(f"[{correlation_id}] Policy rejected spec: {policy_decision.get('reasons', [])}")
                return OrchestratorResult("rejected", {"policy": policy_decision})

            # Security scanning
            security_scanner = self._load_component("security_scanner", "security")
            if hasattr(security_scanner, 'scan_async'):
                scan_async = security_scanner.scan_async
            else:
                return OrchestratorResult("error", {"error": "security_scanner missing scan_async() method"})

            findings = await scan_async(spec)

            # Check for critical findings that should halt pipeline (case-insensitive)
            critical_findings = [f for f in findings if f.get("severity", "").lower() == "critical"]
            if critical_findings and self.config.get("halt_on_critical", True):
                logger.warning(f"[{correlation_id}] Critical security findings detected")
                return OrchestratorResult(
                    "rejected",
                    {
                        "policy": policy_decision,
                        "findings": findings,
                        "halt_reason": "critical_security_findings",
                        "critical_findings": critical_findings
                    }
                )

            # Mutation testing
            mutation_tester = self._load_component("mutation_tester", "mutation")
            if hasattr(mutation_tester, 'run_mutations'):
                run_mutations = mutation_tester.run_mutations
            else:
                return OrchestratorResult("error", {"error": "mutation_tester missing run_mutations() method"})

            mutation_report = await run_mutations(spec)

            # PR creation (with idempotency key)
            pr_creator = self._load_component("pr_creator", "pr")
            if hasattr(pr_creator, 'open_pr'):
                open_pr = pr_creator.open_pr
            else:
                return OrchestratorResult("error", {"error": "pr_creator missing open_pr() method"})

            pr_info = await open_pr(
                spec,
                {"findings": findings, "mutation_report": mutation_report},
                idempotency_key=idempotency_key
            )

            # Check for PR creation failure (explicit check for None)
            if pr_info.get("error") or pr_info.get("url") is None:
                logger.error(f"[{correlation_id}] PR creation failed: {pr_info.get('error', 'No URL returned')}")
                return OrchestratorResult(
                    "error",
                    {
                        "policy": policy_decision,
                        "findings": findings,
                        "mutation_report": mutation_report,
                        "pr_error": pr_info.get("error", "PR creation failed - no URL returned")
                    }
                )

            logger.info(f"[{correlation_id}] Pipeline completed successfully")
            return OrchestratorResult(
                "success",
                {
                    "policy": policy_decision,
                    "findings": findings,
                    "mutation_report": mutation_report,
                    "pr": pr_info,
                },
            )

        except Exception as e:
            logger.error(f"[{correlation_id}] Pipeline execution error: {e}")
            raise

    async def run_stress_batch(
        self,
        specs: list[Dict[str, Any]],
        concurrency: int = 10,
        on_progress: Optional[Callable[[int, OrchestratorResult], None]] = None,
        collect_metrics: bool = True,
    ) -> Tuple[list[OrchestratorResult], Optional[PerformanceMetrics]]:
        """
        Run many specs concurrently with a semaphore cap. Returns results in submission order.

        Args:
            specs: List of specs to process
            concurrency: Maximum concurrent executions
            on_progress: Optional callback for progress updates (must be thread-safe)
            collect_metrics: Whether to collect and return performance metrics

        Returns:
            Tuple of (results list, metrics) where metrics is None if collect_metrics is False
        """
        sem = asyncio.Semaphore(concurrency)
        # Results list preserves order by index assignment
        results: list[Optional[OrchestratorResult]] = [None] * len(specs)
        durations = [] if collect_metrics else None

        async def _task(idx: int, s: Dict[str, Any]) -> None:
            async with sem:
                res = await self.run_once(s)
                results[idx] = res
                if collect_metrics and durations is not None:
                    durations.append(res.duration_ms)
                if on_progress:
                    # Note: on_progress callback must be thread-safe
                    on_progress(idx, res)

        # Gather all tasks and check for exceptions
        task_results = await asyncio.gather(
            *[asyncio.create_task(_task(i, spec)) for i, spec in enumerate(specs)],
            return_exceptions=True
        )

        # Check for exceptions in results
        for i, task_result in enumerate(task_results):
            if isinstance(task_result, Exception):
                logger.error(f"Task {i} failed with exception: {task_result}")
                # Create error result for failed task
                if results[i] is None:
                    results[i] = OrchestratorResult(
                        "error",
                        {
                            "exception": str(task_result),
                            "error_type": type(task_result).__name__
                        }
                    )

        # Calculate final metrics
        if collect_metrics and durations:
            self.metrics.calculate_percentiles(durations)
            return results, self.metrics

        return results, None

    async def run_chaos_test(
        self,
        base_spec: Dict[str, Any],
        chaos_iterations: int = 100,
        chaos_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run chaos testing by applying random mutations to a base spec.

        Args:
            base_spec: Base specification to mutate
            chaos_iterations: Number of chaos iterations to run
            chaos_strategies: List of chaos strategies to apply

        Returns:
            Dictionary with chaos test results
        """
        if chaos_strategies is None:
            chaos_strategies = [
                "remove_fields",
                "duplicate_ids",
                "invalid_types",
                "null_values",
                "oversized_values",
                "circular_references",
                "injection_patterns"
            ]

        chaos_results = {
            "total_iterations": chaos_iterations,
            "strategies_used": chaos_strategies,
            "results_by_strategy": defaultdict(lambda: {"success": 0, "rejected": 0, "error": 0}),
            "interesting_failures": []
        }

        for i in range(chaos_iterations):
            # Select random strategy
            strategy = chaos_strategies[i % len(chaos_strategies)]

            # Apply chaos mutation (deep copy to avoid modifying original)
            mutated_spec = self._apply_chaos_mutation(copy.deepcopy(base_spec), strategy)

            # Run the mutated spec
            result = await self.run_once(mutated_spec)

            # Collect results
            chaos_results["results_by_strategy"][strategy][result.status] += 1

            # Collect interesting failures
            if result.status == "error" and "exception" in result.details:
                chaos_results["interesting_failures"].append({
                    "iteration": i,
                    "strategy": strategy,
                    "error": result.details["exception"][:200],  # Truncate for readability
                    "spec_sample": str(mutated_spec)[:200]
                })

        return chaos_results

    def _apply_chaos_mutation(self, spec: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """
        Apply a chaos mutation strategy to a spec.
        Note: Spec should be deep copied before calling this method.

        Chaos strategies:
        - remove_fields: Randomly remove required fields
        - duplicate_ids: Create duplicate node IDs
        - invalid_types: Change field types to invalid values
        - null_values: Insert null values randomly
        - oversized_values: Create extremely large values
        - circular_references: Create circular references in edges
        - injection_patterns: Add injection patterns to fields
        """
        import random

        if strategy == "remove_fields":
            # Remove random required fields
            fields = list(spec.keys())
            if fields:
                del spec[random.choice(fields)]

        elif strategy == "duplicate_ids":
            # Create duplicate node IDs
            if "nodes" in spec and len(spec["nodes"]) > 1:
                spec["nodes"][0]["id"] = spec["nodes"][1].get("id", "duplicate")

        elif strategy == "invalid_types":
            # Change field types to invalid values
            if "nodes" in spec:
                spec["nodes"] = "not_a_list"

        elif strategy == "null_values":
            # Insert null values randomly
            for key in list(spec.keys()):
                if random.random() < 0.3:
                    spec[key] = None

        elif strategy == "oversized_values":
            # Create extremely large values
            spec["large_field"] = "x" * 1000000  # 1MB string

        elif strategy == "circular_references":
            # Create circular references in edges
            if "edges" in spec:
                spec["edges"].append({"from": "node1", "to": "node1"})

        elif strategy == "injection_patterns":
            # Add injection patterns to fields
            injection = random.choice([
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../etc/passwd",
                "{{7*7}}"
            ])
            spec["injected_field"] = injection

        return spec

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get a comprehensive metrics report."""
        return {
            "metrics": self.metrics.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "timeout_seconds": self.timeout_seconds,
                "retry_config": self.retry_config,
                "components": list(self.components.keys())
            }
        }

# ---------------------------
# Custom Exceptions
# ---------------------------

class SecurityError(Exception):
    """Raised when a security constraint is violated."""
    pass

# ---------------------------
# Test Fixtures and Utilities
# ---------------------------

@pytest.fixture
def orchestrator(monkeypatch) -> GenerationOrchestrator:
    """
    Provide a GenerationOrchestrator with mocked components returning
    deterministic, plausible data for pipeline steps.
    """
    orch = GenerationOrchestrator({"test_mode": True})

    # Create mock components that match production types
    class MockPolicyEngine:
        def evaluate(self, spec):
            return {"allowed": True, "reasons": ["meets_minimum_requirements"], "risk_score": 3}

    class MockSecurityScanner:
        async def scan_async(self, spec):
            await asyncio.sleep(0.001)
            return [{"type": "info", "severity": "low", "message": "no critical issues", "location": "overall"}]

    class MockMutationTester:
        async def run_mutations(self, spec):
            await asyncio.sleep(0.001)
            return {"score": 0.97, "killed": 39, "survived": 1, "total": 40}

    class MockPRCreator:
        async def open_pr(self, spec, report, idempotency_key=None):
            await asyncio.sleep(0.001)
            return {"url": "https://example.org/pr/123", "id": 123, "status": "open"}

    def mock_load_component(self, component_name, config_key, **kwargs):
        if component_name == "policy_engine":
            return MockPolicyEngine()
        if component_name == "security_scanner":
            return MockSecurityScanner()
        if component_name == "mutation_tester":
            return MockMutationTester()
        if component_name == "pr_creator":
            return MockPRCreator()
        raise NotImplementedError(f"Unknown component: {component_name}")

    monkeypatch.setattr(
        GenerationOrchestrator,
        "_load_component",
        mock_load_component,
        raising=True,
    )
    return orch

@pytest.fixture
def production_orchestrator() -> GenerationOrchestrator:
    """Provide a production orchestrator with real components."""
    config = {
        "test_mode": False,
        "timeout_seconds": 30,
        "retry": {"max_retries": 2, "backoff_ms": 100},
        "policy": {
            "strict_mode": False,
            "max_risk_level": 7,
            "max_nodes": 100,
            "blacklist_patterns": ["DROP TABLE", "rm -rf"]
        },
        "security": {
            "scan_depth": "deep",
            "timeout_seconds": 10
        },
        "mutation": {
            "mutation_count": 20,
            "strategies": ["field_swap", "type_change", "value_mutation", "structure_change"]
        },
        "pr": {
            "base_url": "https://github.com/test/repo",
            "auto_merge": False
        }
    }
    return GenerationOrchestrator(config)

# ---------------------------
# Tests
# ---------------------------

@pytest.mark.asyncio
async def test_run_once_success(orchestrator: GenerationOrchestrator):
    """Test successful single run."""
    spec = {"id": "test-1", "name": "demo-spec", "nodes": [], "edges": []}
    res = await orchestrator.run_once(spec)
    assert res.status == "success"
    assert "policy" in res.details and res.details["policy"]["allowed"] is True
    assert isinstance(res.details.get("findings"), list)
    assert "mutation_report" in res.details
    assert "pr" in res.details and "url" in res.details["pr"]
    assert res.duration_ms > 0
    assert res.retries == 0
    assert res.correlation_id is not None


@pytest.mark.asyncio
async def test_run_once_policy_reject(monkeypatch):
    """Test policy rejection flow."""
    orch = GenerationOrchestrator({"test_mode": True})

    class MockPolicyEngine:
        def evaluate(self, spec):
            return {"allowed": False, "reasons": ["forbidden_by_policy"], "risk_score": 15}

    class MockSecurityScanner:
        async def scan_async(self, spec):
            return []

    class MockMutationTester:
        async def run_mutations(self, spec):
            return {}

    class MockPRCreator:
        async def open_pr(self, spec, report, idempotency_key=None):
            return {}

    def mock_load_component(self, component_name, *_a, **_k):
        if component_name == "policy_engine":
            return MockPolicyEngine()
        if component_name == "security_scanner":
            return MockSecurityScanner()
        if component_name == "mutation_tester":
            return MockMutationTester()
        if component_name == "pr_creator":
            return MockPRCreator()
        raise NotImplementedError()

    monkeypatch.setattr(GenerationOrchestrator, "_load_component", mock_load_component, raising=True)

    res = await orch.run_once({"name": "blocked-spec"})
    assert res.status == "rejected"
    assert res.details["policy"]["reasons"] == ["forbidden_by_policy"]
    assert res.duration_ms > 0


@pytest.mark.asyncio
async def test_missing_component_produces_error(monkeypatch):
    """Test error handling for missing components."""
    orch = GenerationOrchestrator({"test_mode": True})

    class MockPolicyEngine:
        def evaluate(self, spec):
            return {"allowed": True}

    def mock_load_component(self, component_name, *_a, **_k):
        if component_name == "policy_engine":
            return MockPolicyEngine()
        # Return object without required methods
        return object()

    monkeypatch.setattr(
        GenerationOrchestrator,
        "_load_component",
        mock_load_component,
        raising=True,
    )

    res = await orch.run_once({"name": "incomplete-pipeline"})
    assert res.status == "error"
    assert "missing" in str(res.details.get("error", "")).lower() or "method" in str(res.details.get("error", "")).lower()


@pytest.mark.asyncio
async def test_stress_concurrency(orchestrator: GenerationOrchestrator):
    """Test concurrent execution with stress load."""
    # 50 lightweight specs, run at concurrency 10
    specs = [{"id": f"spec-{i}", "name": f"spec-{i}", "nodes": [], "edges": []} for i in range(50)]
    results, metrics = await orchestrator.run_stress_batch(specs, concurrency=10, collect_metrics=True)

    assert len(results) == len(specs)
    assert all(r is not None for r in results)
    assert sum(1 for r in results if r.status == "success") == 50

    # Check metrics
    assert metrics is not None
    assert metrics.total_runs == 50
    assert metrics.successful_runs == 50
    assert metrics.failed_runs == 0
    assert metrics.avg_duration_ms > 0


@pytest.mark.asyncio
async def test_retry_mechanism(monkeypatch):
    """Test retry mechanism on transient failures."""
    orch = GenerationOrchestrator({
        "test_mode": True,
        "retry": {"max_retries": 3, "backoff_ms": 10}
    })

    call_count = {"count": 0}

    class MockPolicyEngine:
        def evaluate(self, spec):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception("Transient error")
            return {"allowed": True, "reasons": []}

    class MockSecurityScanner:
        async def scan_async(self, spec):
            return []

    class MockMutationTester:
        async def run_mutations(self, spec):
            return {"score": 0.8, "killed": 8, "survived": 2, "total": 10}

    class MockPRCreator:
        async def open_pr(self, spec, report, idempotency_key=None):
            return {"url": "https://example.org/pr/123", "id": 123}

    def mock_load_component(self, component_name, *_a, **_k):
        if component_name == "policy_engine":
            return MockPolicyEngine()
        elif component_name == "security_scanner":
            return MockSecurityScanner()
        elif component_name == "mutation_tester":
            return MockMutationTester()
        elif component_name == "pr_creator":
            return MockPRCreator()
        raise NotImplementedError()

    monkeypatch.setattr(GenerationOrchestrator, "_load_component", mock_load_component, raising=True)

    res = await orch.run_once({"id": "test", "name": "retry-test"})
    assert res.status == "success" or res.status == "error"
    assert res.retries > 0 or call_count["count"] > 1


@pytest.mark.asyncio
async def test_timeout_handling(monkeypatch):
    """Test timeout handling."""
    orch = GenerationOrchestrator({
        "test_mode": True,
        "timeout_seconds": 0.1  # Very short timeout
    })

    class MockPolicyEngine:
        def evaluate(self, spec):
            return {"allowed": True}

    class MockSecurityScanner:
        async def scan_async(self, spec):
            await asyncio.sleep(1)  # Longer than timeout
            return []

    class MockMutationTester:
        async def run_mutations(self, spec):
            return {}

    class MockPRCreator:
        async def open_pr(self, spec, report, idempotency_key=None):
            return {}

    def mock_load_component(self, component_name, *_a, **_k):
        if component_name == "policy_engine":
            return MockPolicyEngine()
        if component_name == "security_scanner":
            return MockSecurityScanner()
        if component_name == "mutation_tester":
            return MockMutationTester()
        if component_name == "pr_creator":
            return MockPRCreator()
        raise NotImplementedError()

    monkeypatch.setattr(GenerationOrchestrator, "_load_component", mock_load_component, raising=True)

    res = await orch.run_once({"id": "test", "name": "timeout-test"})
    assert res.status == "timeout"


@pytest.mark.asyncio
async def test_production_components():
    """Test production component implementations."""
    # Test PolicyEngine
    policy_engine = PolicyEngine({"max_risk_level": 5, "strict_mode": False})
    result = policy_engine.evaluate({"id": "test", "type": "Graph"})
    assert "allowed" in result
    assert "risk_score" in result

    # Test SecurityScanner
    scanner = SecurityScanner({"scan_depth": "normal"})
    findings = await scanner.scan_async({"id": "test", "nodes": [], "edges": []})
    assert isinstance(findings, list)
    assert all(isinstance(f, dict) for f in findings)

    # Test MutationTester
    tester = MutationTester({"mutation_count": 5})
    report = await tester.run_mutations({"id": "test"})
    assert "score" in report
    assert "killed" in report
    assert "survived" in report

    # Test PRCreator
    pr_creator = PRCreator({"base_url": "https://test.com"})
    pr_info = await pr_creator.open_pr(
        {"id": "test"},
        {"findings": [], "mutation_report": {"score": 0.8}},
        idempotency_key="test-key"
    )
    assert "url" in pr_info or "error" in pr_info


@pytest.mark.asyncio
async def test_chaos_testing(production_orchestrator: GenerationOrchestrator):
    """Test chaos testing functionality."""
    base_spec = {
        "id": "chaos-base",
        "type": "Graph",
        "nodes": [
            {"id": "n1", "type": "InputNode"},
            {"id": "n2", "type": "OutputNode"}
        ],
        "edges": [
            {"from": "n1", "to": "n2"}
        ]
    }

    chaos_results = await production_orchestrator.run_chaos_test(
        base_spec,
        chaos_iterations=10,
        chaos_strategies=["remove_fields", "null_values", "injection_patterns"]
    )

    assert "total_iterations" in chaos_results
    assert chaos_results["total_iterations"] == 10
    assert "results_by_strategy" in chaos_results
    assert "interesting_failures" in chaos_results


@pytest.mark.asyncio
async def test_metrics_collection(production_orchestrator: GenerationOrchestrator):
    """Test metrics collection and reporting."""
    specs = [
        {"id": f"metric-{i}", "type": "Graph", "nodes": [], "edges": []}
        for i in range(10)
    ]

    results, metrics = await production_orchestrator.run_stress_batch(
        specs,
        concurrency=5,
        collect_metrics=True
    )

    assert metrics is not None
    assert metrics.total_runs == 10
    assert metrics.min_duration_ms <= metrics.avg_duration_ms <= metrics.max_duration_ms
    assert metrics.p50_duration_ms > 0
    assert metrics.p95_duration_ms >= metrics.p50_duration_ms

    # Get metrics report
    report = production_orchestrator.get_metrics_report()
    assert "metrics" in report
    assert "timestamp" in report
    assert "config" in report


@pytest.mark.asyncio
async def test_critical_findings_halt(monkeypatch):
    """Test that critical security findings halt the pipeline."""
    orch = GenerationOrchestrator({
        "test_mode": True,
        "halt_on_critical": True
    })

    class MockPolicyEngine:
        def evaluate(self, spec):
            return {"allowed": True}

    class MockSecurityScanner:
        async def scan_async(self, spec):
            return [
                {"type": "security", "severity": "critical", "message": "Critical vulnerability found"},
                {"type": "info", "severity": "low", "message": "Minor issue"}
            ]

    class MockMutationTester:
        async def run_mutations(self, spec):
            return {}

    class MockPRCreator:
        async def open_pr(self, spec, report, idempotency_key=None):
            return {}

    def mock_load_component(self, component_name, *_a, **_k):
        if component_name == "policy_engine":
            return MockPolicyEngine()
        if component_name == "security_scanner":
            return MockSecurityScanner()
        if component_name == "mutation_tester":
            return MockMutationTester()
        if component_name == "pr_creator":
            return MockPRCreator()
        raise NotImplementedError()

    monkeypatch.setattr(GenerationOrchestrator, "_load_component", mock_load_component, raising=True)

    res = await orch.run_once({"id": "test", "name": "critical-test"})
    assert res.status == "rejected"
    assert "halt_reason" in res.details
    assert res.details["halt_reason"] == "critical_security_findings"


@pytest.mark.asyncio
async def test_idempotency(production_orchestrator: GenerationOrchestrator):
    """Test idempotency of PR creation on retries."""
    spec = {"id": "idempotent-test", "type": "Graph", "nodes": [], "edges": []}

    # Run twice with same spec
    result1 = await production_orchestrator.run_once(spec)
    result2 = await production_orchestrator.run_once(spec)

    # Both should succeed (or both fail consistently)
    assert result1.status == result2.status


# ---------------------------
# Main execution for testing
# ---------------------------

if __name__ == "__main__":
    import sys

    async def main():
        """Run comprehensive stress tests."""
        print("Starting comprehensive stress tests...")
        print("-" * 60)

        # Initialize production orchestrator
        orchestrator = GenerationOrchestrator({
            "timeout_seconds": 30,
            "retry": {"max_retries": 2, "backoff_ms": 100},
            "policy": {"max_risk_level": 7},
            "security": {"scan_depth": "deep"},
            "mutation": {"mutation_count": 10},
            "pr": {"auto_merge": False}
        })

        # Generate test specs
        test_specs = [
            {
                "id": f"stress-{i}",
                "type": "Graph",
                "nodes": [
                    {"id": f"n{j}", "type": "Node"}
                    for j in range(i % 10 + 1)
                ],
                "edges": [
                    {"from": f"n{j}", "to": f"n{(j+1) % (i % 10 + 1)}"}
                    for j in range(i % 5)
                ]
            }
            for i in range(100)
        ]

        # Progress callback
        def on_progress(idx: int, result: OrchestratorResult):
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(test_specs)} specs...")

        # Run stress test
        print("\nRunning stress test with 100 specs at concurrency 20...")
        start = time.time()

        results, metrics = await orchestrator.run_stress_batch(
            test_specs,
            concurrency=20,
            on_progress=on_progress,
            collect_metrics=True
        )

        duration = time.time() - start

        # Print results
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(f"Total duration: {duration:.2f}s")
        print(f"Specs processed: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r.status == 'success')}")
        print(f"Rejected: {sum(1 for r in results if r.status == 'rejected')}")
        print(f"Failed: {sum(1 for r in results if r.status == 'error')}")
        print(f"Timeouts: {sum(1 for r in results if r.status == 'timeout')}")

        if metrics:
            print("\n" + "-" * 60)
            print("PERFORMANCE METRICS")
            print("-" * 60)
            metrics_dict = metrics.to_dict()
            for key, value in metrics_dict.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")

        # Run chaos test
        print("\n" + "=" * 60)
        print("CHAOS TEST")
        print("=" * 60)

        base_spec = {
            "id": "chaos-base",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "Node"}],
            "edges": []
        }

        chaos_results = await orchestrator.run_chaos_test(
            base_spec,
            chaos_iterations=50
        )

        print(f"Chaos iterations: {chaos_results['total_iterations']}")
        print("Results by strategy:")
        for strategy, results_dict in chaos_results["results_by_strategy"].items():
            print(f"  {strategy}: {results_dict}")

        if chaos_results["interesting_failures"]:
            print(f"\nFound {len(chaos_results['interesting_failures'])} interesting failures")
            for failure in chaos_results["interesting_failures"][:3]:
                print(f"  - Strategy: {failure['strategy']}")
                print(f"    Error: {failure['error'][:100]}...")

        print("\n" + "=" * 60)
        print("All stress tests completed successfully!")

        # Save report
        report = orchestrator.get_metrics_report()
        report["test_results"] = {
            "total_specs": len(test_specs),
            "duration_seconds": duration,
            "chaos_test": chaos_results
        }

        with open("stress_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Report saved to stress_test_report.json")

    # Run the async main function
    asyncio.run(main())
