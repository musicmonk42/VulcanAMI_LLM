# run_stress_tests.py
"""
Async stress tests for Graphix components.

This script runs a suite of stress tests on key Graphix components including:
- Graph generation and validation
- Malicious IR detection
- Language evolution registry
- Core ontology validation
- Unified runtime execution under load

Uses pytest for structured testing with async support.
Integrates with project modules for real-world testing.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
import os
import random
import re
import sys
import threading
import time
import traceback
import uuid
from asyncio import \
    TaskGroup  # Structured concurrency for better error handling
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
# Configure logging with rotation for production
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytest
import pytest_asyncio
from graphix.specs.formal_grammar.graphix_core_ontology import \
    graphix_core_ontology  # Assuming loaded as dict
from graphix.specs.formal_grammar.language_evolution_registry import (
    DevelopmentKMS, InMemoryBackend, LanguageEvolutionRegistry)
from graphix.src.unified_runtime.execution_engine import ExecutionMode
from graphix.src.unified_runtime.graph_validator import (GraphValidator,
                                                         ResourceLimits)
from graphix.src.unified_runtime.unified_runtime_core import (RuntimeConfig,
                                                              UnifiedRuntime)
# Project imports - assuming relative paths or proper sys.path setup
from graphix.stress_tests.large_graph_generator import (GraphGenerator,
                                                        GraphTopology)
from graphix.stress_tests.malicious_ir_generator import IRGenerator

handler = RotatingFileHandler('stress_test.log', maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler]
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
CHAOS_ITERATIONS = 50

# Allowed component paths (populated with project paths)
ALLOWED_COMPONENT_PATHS = {
    "graph_generator": ["graphix/stress_tests/large_graph_generator.py"],
    "ir_generator": ["graphix/stress_tests/malicious_ir_generator.py"],
    "ontology": ["graphix/specs/formal_grammar/graphix_core_ontology.json"],
    "registry": ["graphix/specs/formal_grammar/language_evolution_registry.py"],
    "validator": ["graphix/src/unified_runtime/graph_validator.py"],
    "runtime": ["graphix/src/unified_runtime/unified_runtime_core.py"]
}

# ---------------------------
# Enums and Data Classes
# ---------------------------

class ComponentType(Enum):
    """Types of components that can be loaded."""
    GRAPH_GENERATOR = "graph_generator"
    IR_GENERATOR = "ir_generator"
    ONTOLOGY = "ontology"
    REGISTRY = "registry"
    VALIDATOR = "validator"
    RUNTIME = "runtime"

@dataclass
class TestResult:
    """Result of a test run."""
    status: str  # "success", "failure", "timeout", "error"
    details: Dict[str, Any]  # arbitrary details
    duration_ms: float = 0.0

@dataclass
class TestMetrics:
    """Aggregated test metrics."""
    total_tests: int = 0
    successful: int = 0
    failed: int = 0
    timeouts: int = 0
    errors: int = 0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')

    def update(self, result: TestResult):
        self.total_tests += 1
        if result.status == "success":
            self.successful += 1
        elif result.status == "failure":
            self.failed += 1
        elif result.status == "timeout":
            self.timeouts += 1
        else:
            self.errors += 1
        self.avg_duration_ms = (self.avg_duration_ms * (self.total_tests - 1) + result.duration_ms) / self.total_tests
        self.max_duration_ms = max(self.max_duration_ms, result.duration_ms)
        self.min_duration_ms = min(self.min_duration_ms, result.duration_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "successful": self.successful,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "avg_duration_ms": self.avg_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float('inf') else 0.0
        }

# ---------------------------
# Helper Functions
# ---------------------------

def load_component(component_type: ComponentType, path: str) -> Any:
    """Securely load a component from allowed paths."""
    if component_type.value not in ALLOWED_COMPONENT_PATHS or path not in ALLOWED_COMPONENT_PATHS[component_type.value]:
        raise ValueError(f"Loading from path {path} not allowed for {component_type}")
    spec = importlib.util.spec_from_file_location(component_type.value, path)
    if spec is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def generate_test_spec() -> Dict[str, Any]:
    """Generate a random test spec for chaos testing."""
    return {
        "id": str(uuid.uuid4()),
        "type": "Graph",
        "nodes": [{"id": f"n{i}", "type": random.choice(["InputNode", "OutputNode", "GenerativeNode"])} for i in range(random.randint(1, 10))],
        "edges": []
    }

# ---------------------------
# Test Fixtures
# ---------------------------

@pytest_asyncio.fixture
async def graph_generator():
    module = load_component(ComponentType.GRAPH_GENERATOR, "graphix/stress_tests/large_graph_generator.py")
    return module.GraphGenerator()

@pytest_asyncio.fixture
async def ir_generator():
    module = load_component(ComponentType.IR_GENERATOR, "graphix/stress_tests/malicious_ir_generator.py")
    return module.IRGenerator()

@pytest_asyncio.fixture
async def ontology():
    with open("graphix/specs/formal_grammar/graphix_core_ontology.json", "r") as f:
        return json.load(f)

@pytest_asyncio.fixture
async def registry():
    module = load_component(ComponentType.REGISTRY, "graphix/specs/formal_grammar/language_evolution_registry.py")
    backend = module.InMemoryBackend()
    kms = module.DevelopmentKMS()
    return module.LanguageEvolutionRegistry(backend=backend, kms=kms)

@pytest_asyncio.fixture
async def validator():
    module = load_component(ComponentType.VALIDATOR, "graphix/src/unified_runtime/graph_validator.py")
    return module.GraphValidator()

@pytest_asyncio.fixture
async def runtime():
    module = load_component(ComponentType.RUNTIME, "graphix/src/unified_runtime/unified_runtime_core.py")
    config = module.RuntimeConfig(enable_metrics=True, enable_hardware_dispatch=False)
    return module.UnifiedRuntime(config)

# ---------------------------
# Tests
# ---------------------------

@pytest.mark.asyncio
async def test_large_graph_generation(graph_generator):
    start_time = time.time()
    try:
        graph = graph_generator.generate_graph(
            num_nodes=10000,
            topology=GraphTopology.BARABASI_ALBERT
        )
        stats = graph_generator._calculate_statistics(graph)
        assert stats.node_count == 10000
        assert stats.edge_count > 0
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Large graph generation: {duration_ms:.2f}ms")
    except Exception as e:
        pytest.fail(f"Large graph generation failed: {str(e)}")

@pytest.mark.asyncio
async def test_malicious_ir_generation(ir_generator):
    start_time = time.time()
    try:
        graphs = ir_generator.generate_batch(count=10)
        assert len(graphs) == 10
        validation = ir_generator.validate_generated_graphs(graphs)
        assert validation["total_graphs"] == 10
        assert sum(validation["violations_found"].values()) > 0
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Malicious IR generation: {duration_ms:.2f}ms")
    except Exception as e:
        pytest.fail(f"Malicious IR generation failed: {str(e)}")

@pytest.mark.asyncio
async def test_ontology_validation(ontology):
    start_time = time.time()
    try:
        # Basic validation
        assert "version" in ontology
        assert ontology["version"] == "2.4.0"
        # Check required fields in node_types
        for node_type, data in ontology["node_types"].items():
            assert "uri" in data
            assert "description" in data
            assert "lifecycle_status" in data
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Ontology validation: {duration_ms:.2f}ms")
    except Exception as e:
        pytest.fail(f"Ontology validation failed: {str(e)}")

@pytest.mark.asyncio
async def test_language_evolution_registry(registry):
    start_time = time.time()
    try:
        # Submit proposal
        proposal = {
            "type": "ProposalNode",
            "proposed_by": "test-agent",
            "rationale": "Test proposal",
            "proposal_content": {"add": {"TestNode": {}}},
            "metadata": {}
        }
        prop_id = registry.submit_proposal(proposal)
        assert prop_id is not None
        
        # Vote
        consensus = {
            "proposal_id": prop_id,
            "votes": {"test-agent": "yes"},
            "weights": {"test-agent": 1.0}
        }
        reached = registry.record_vote(consensus)
        assert reached
        
        # Deploy
        deployed = registry.deploy_grammar_version(prop_id, "test-1.0")
        assert deployed
        
        integrity = registry.verify_audit_log_integrity()
        assert integrity
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Language registry test: {duration_ms:.2f}ms")
    except Exception as e:
        pytest.fail(f"Language registry test failed: {str(e)}")

@pytest.mark.asyncio
async def test_graph_validation(validator, graph_generator):
    start_time = time.time()
    try:
        graph = graph_generator.generate_graph(num_nodes=100, topology=GraphTopology.RANDOM)
        result = validator.validate_graph(graph)
        assert result.is_valid
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Graph validation: {duration_ms:.2f}ms")
    except Exception as e:
        pytest.fail(f"Graph validation failed: {str(e)}")

@pytest.mark.asyncio
async def test_runtime_execution(runtime, graph_generator):
    start_time = time.time()
    try:
        graph = graph_generator.generate_graph(num_nodes=50, topology=GraphTopology.DAG)
        exec_result = await runtime.execute_graph(graph)
        assert exec_result["status"] == "success"
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Runtime execution: {duration_ms:.2f}ms")
    except Exception as e:
        pytest.fail(f"Runtime execution failed: {str(e)}")

@pytest.mark.asyncio
async def test_chaos(graph_generator, validator):
    async def chaos_task():
        graph = graph_generator.generate_graph(num_nodes=random.randint(1, MAX_NODES), topology=random.choice(list(GraphTopology)))
        return validator.validate_graph(graph)
    
    start_time = time.time()
    try:
        async with TaskGroup() as tg:
            tasks = [tg.create_task(chaos_task()) for _ in range(CHAOS_ITERATIONS)]
        
        results = [await t for t in tasks]
        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count > 0  # At least some should be valid
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Chaos test: {duration_ms:.2f}ms, {valid_count}/{CHAOS_ITERATIONS} valid")
    except Exception as e:
        pytest.fail(f"Chaos test failed: {str(e)}")

# ---------------------------
# Run Tests
# ---------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])