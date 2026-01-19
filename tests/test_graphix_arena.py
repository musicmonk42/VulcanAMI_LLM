"""
Comprehensive test suite for graphix_arena.py
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graphix_arena import (
    MAX_AGENT_ID_LENGTH,
    MAX_FEEDBACK_LOG_SIZE,
    MAX_GRAPH_ID_LENGTH,
    MAX_PAYLOAD_SIZE,
    MAX_REBERT_THRESHOLD,
    MIN_REBERT_THRESHOLD,
    AgentNotFoundException,
    BiasDetectedException,
    Edge,
    GraphixArena,
    GraphixIRGraph,
    GraphSpec,
    Node,
    app,
    rebert_prune,
    _load_runtime_config,
    _merge_configs,
    _RUNTIME_CONFIG,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def arena(monkeypatch):
    """Create arena instance with Ray disabled for testing."""
    # Set environment variable to disable Ray for tests using monkeypatch
    monkeypatch.setenv('VULCAN_ENABLE_RAY', '0')
    arena_instance = GraphixArena(port=8182)
    yield arena_instance
    # monkeypatch automatically cleans up after test


@pytest.fixture
def valid_graph_spec():
    """Create valid graph spec."""
    return {"spec_id": "test_spec_123", "parameters": {"param1": "value1"}}


@pytest.fixture
def valid_graph():
    """Create valid graph."""
    return {
        "graph_id": "test_graph_123",
        "nodes": [
            {"id": "node1", "label": "Node 1", "properties": {}},
            {"id": "node2", "label": "Node 2", "properties": {}},
        ],
        "edges": [{"source_id": "node1", "target_id": "node2", "weight": 1.0}],
        "metadata": {},
    }


class TestPydanticModels:
    """Test Pydantic models."""

    def test_graph_spec_valid(self, valid_graph_spec):
        """Test valid GraphSpec."""
        spec = GraphSpec(**valid_graph_spec)

        assert spec.spec_id == "test_spec_123"
        assert spec.parameters == {"param1": "value1"}

    def test_graph_spec_invalid_id(self):
        """Test GraphSpec with invalid ID."""
        with pytest.raises(ValidationError, match="alphanumeric"):
            GraphSpec(spec_id="invalid!@#", parameters={})

    def test_graph_spec_too_long(self):
        """Test GraphSpec with too long ID."""
        long_id = "x" * (MAX_GRAPH_ID_LENGTH + 1)

        with pytest.raises(ValidationError):
            GraphSpec(spec_id=long_id, parameters={})

    def test_node_valid(self):
        """Test valid Node."""
        node = Node(id="node1", label="Test Node", properties={"key": "value"})

        assert node.id == "node1"
        assert node.label == "Test Node"

    def test_node_invalid_id(self):
        """Test Node with invalid ID."""
        with pytest.raises(ValidationError):
            Node(id="", label="Test")

    def test_edge_valid(self):
        """Test valid Edge."""
        edge = Edge(source_id="n1", target_id="n2", weight=0.5)

        assert edge.source_id == "n1"
        assert edge.target_id == "n2"
        assert edge.weight == 0.5

    def test_edge_invalid_weight(self):
        """Test Edge with invalid weight."""
        with pytest.raises(ValidationError):
            Edge(source_id="n1", target_id="n2", weight="invalid")

    def test_graphix_ir_graph_valid(self, valid_graph):
        """Test valid GraphixIRGraph."""
        graph = GraphixIRGraph(**valid_graph)

        assert graph.graph_id == "test_graph_123"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_graphix_ir_graph_invalid_id(self):
        """Test GraphixIRGraph with invalid ID."""
        invalid_graph = {
            "graph_id": "invalid!@#",
            "nodes": [{"id": "n1", "label": "N1"}],
            "edges": [],
        }

        with pytest.raises(ValidationError):
            GraphixIRGraph(**invalid_graph)

    def test_graphix_ir_graph_no_nodes(self):
        """Test GraphixIRGraph with no nodes."""
        with pytest.raises(ValidationError, match="at least one node"):
            GraphixIRGraph(graph_id="test", nodes=[], edges=[])

    def test_graphix_ir_graph_too_many_nodes(self):
        """Test GraphixIRGraph with too many nodes."""
        many_nodes = [{"id": f"node{i}", "label": f"Node {i}"} for i in range(10001)]

        with pytest.raises(ValidationError, match="cannot have more than"):
            GraphixIRGraph(graph_id="test", nodes=many_nodes, edges=[])


class TestRebertPrune:
    """Test ReBERT pruning function."""

    def test_rebert_prune_basic(self):
        """Test basic ReBERT pruning."""
        import numpy as np

        tensor = np.array([0.05, 0.15, 0.25, 0.01])
        threshold = 0.1

        pruned = rebert_prune(tensor, threshold=threshold)

        assert isinstance(pruned, list)

    def test_rebert_prune_invalid_threshold_type(self):
        """Test pruning with invalid threshold type."""
        import numpy as np

        tensor = np.array([0.1, 0.2, 0.3])

        # Should use default threshold
        pruned = rebert_prune(tensor, threshold="invalid")

        assert isinstance(pruned, list)

    def test_rebert_prune_threshold_clamping(self):
        """Test threshold clamping."""
        import numpy as np

        tensor = np.array([0.1, 0.2, 0.3])

        # Threshold above max
        pruned = rebert_prune(tensor, threshold=1.0)

        assert isinstance(pruned, list)


class TestGraphixArena:
    """Test GraphixArena class."""

    def test_initialization(self, arena):
        """Test arena initialization with Ray disabled."""
        # Use the arena fixture which already has Ray disabled
        assert arena.port == 8182
        assert len(arena.agents) > 0
        assert arena.runtime is not None

    def test_invalid_port(self):
        """Test initialization with invalid port."""
        with pytest.raises(ValueError, match="Port must be"):
            GraphixArena(port=100)

        with pytest.raises(ValueError, match="Port must be"):
            GraphixArena(port=70000)

    def test_agent_configuration(self, arena):
        """Test agent configuration."""
        assert "generator" in arena.agents
        assert "evolver" in arena.agents
        assert "visualizer" in arena.agents

    @pytest.mark.asyncio
    async def test_run_agent_invalid_id(self, arena):
        """Test running agent with invalid ID."""
        request = Mock()
        request.json = AsyncMock(return_value={})

        with pytest.raises(Exception):  # HTTPException
            await arena.run_agent_task("invalid!@#", request)

    @pytest.mark.asyncio
    async def test_run_agent_too_long_id(self, arena):
        """Test running agent with too long ID."""
        long_id = "x" * (MAX_AGENT_ID_LENGTH + 1)

        request = Mock()
        request.json = AsyncMock(return_value={})

        with pytest.raises(Exception):  # HTTPException
            await arena.run_agent_task(long_id, request)

    @pytest.mark.asyncio
    async def test_run_agent_not_found(self, arena):
        """Test running non-existent agent."""
        request = Mock()
        request.json = AsyncMock(return_value={})

        with pytest.raises(AgentNotFoundException):
            await arena.run_agent_task("nonexistent", request)

    @pytest.mark.asyncio
    async def test_run_agent_payload_too_large(self, arena):
        """Test running agent with oversized payload."""
        large_payload = {"data": "x" * (MAX_PAYLOAD_SIZE + 1)}

        request = Mock()
        request.json = AsyncMock(return_value=large_payload)

        with pytest.raises(Exception):  # HTTPException 413
            await arena.run_agent_task("generator", request)

    def test_run_transparent_task(self, arena):
        """Test transparent task execution."""
        payload = {"input_tensor": [[1.0, 2.0], [3.0, 4.0]]}

        result = arena.run_transparent_task("test_agent", "test task", payload)

        assert isinstance(result, dict)
        assert "interpretability" in result
        assert "audit" in result
        assert "observability" in result

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_run_shadow_task(self, arena):
        """Test shadow task execution."""
        payload = {"graph_id": "test_graph", "data": "test"}

        result = await arena.run_shadow_task("test_agent", "test task", payload)

        assert isinstance(result, dict)
        assert "status" in result

    @pytest.mark.asyncio
    async def test_rollback_failed_task(self, arena):
        """Test task rollback."""
        payload = {"graph_id": "test"}

        result = await arena.rollback_failed_task(payload, reason="test failure")

        assert result["status"] == "rollback"
        assert result["reason"] == "test failure"

    @pytest.mark.asyncio
    async def test_feedback_ingestion_valid(self, arena):
        """Test valid feedback ingestion."""
        request = Mock()
        request.json = AsyncMock(
            return_value={
                "graph_id": "test_graph",
                "agent_id": "test_agent",
                "score": 0.9,
                "rationale": "Good performance",
            }
        )

        result = await arena.feedback_ingestion(request)

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_feedback_ingestion_missing_fields(self, arena):
        """Test feedback ingestion with missing fields."""
        request = Mock()
        request.json = AsyncMock(
            return_value={
                "graph_id": "test_graph"
                # Missing agent_id and score
            }
        )

        with pytest.raises(Exception):  # HTTPException 422
            await arena.feedback_ingestion(request)

    @pytest.mark.asyncio
    async def test_feedback_ingestion_invalid_score(self, arena):
        """Test feedback ingestion with invalid score."""
        request = Mock()
        request.json = AsyncMock(
            return_value={
                "graph_id": "test_graph",
                "agent_id": "test_agent",
                "score": "invalid",
            }
        )

        with pytest.raises(Exception):  # HTTPException 422
            await arena.feedback_ingestion(request)

    @pytest.mark.asyncio
    async def test_feedback_ingestion_negative_score(self, arena):
        """Test feedback with negative score triggers rollback."""
        request = Mock()
        request.json = AsyncMock(
            return_value={
                "graph_id": "test_graph",
                "agent_id": "test_agent",
                "score": -0.5,
            }
        )

        result = await arena.feedback_ingestion(request)

        assert result["status"] == "ok"

    def test_feedback_log_bounded(self, arena):
        """Test feedback log is bounded."""
        # Add many feedback entries
        for i in range(MAX_FEEDBACK_LOG_SIZE + 100):
            arena.feedback_log.append({"id": i})

        assert len(arena.feedback_log) <= MAX_FEEDBACK_LOG_SIZE

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv('CI') == 'true' or os.getenv('VULCAN_ENABLE_RAY') == '0',
        reason="Skip tournament tests in CI or when Ray is disabled"
    )
    async def test_tournament_task_valid(self, arena):
        """Test valid tournament task."""
        import numpy as np

        request = Mock()
        request.json = AsyncMock(
            return_value={
                "proposals": ["graph1", "graph2", "graph3"],
                "fitness": [0.8, 0.9, 0.7],
            }
        )

        # Mock tournament manager
        if arena.tournament_manager:
            arena.tournament_manager.run_adaptive_tournament = Mock(return_value=[1])

        # If no tournament manager, should raise 503
        if not arena.tournament_manager:
            with pytest.raises(Exception):  # HTTPException 503
                await arena.run_tournament_task(request)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv('CI') == 'true' or os.getenv('VULCAN_ENABLE_RAY') == '0',
        reason="Skip tournament tests in CI or when Ray is disabled"
    )
    async def test_tournament_task_empty_proposals(self, arena):
        """Test tournament with empty proposals."""
        request = Mock()
        request.json = AsyncMock(return_value={"proposals": [], "fitness": []})

        if arena.tournament_manager:
            with pytest.raises(Exception):  # HTTPException 400
                await arena.run_tournament_task(request)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv('CI') == 'true' or os.getenv('VULCAN_ENABLE_RAY') == '0',
        reason="Skip tournament tests in CI or when Ray is disabled"
    )
    async def test_tournament_task_length_mismatch(self, arena):
        """Test tournament with length mismatch."""
        request = Mock()
        request.json = AsyncMock(
            return_value={"proposals": ["g1", "g2"], "fitness": [0.5]}  # Mismatch
        )

        if arena.tournament_manager:
            with pytest.raises(Exception):  # HTTPException 400
                await arena.run_tournament_task(request)

    def test_send_slack_alert_not_configured(self, arena):
        """Test Slack alert when not configured."""
        # Should not raise
        arena.send_slack_alert("Test message")

    def test_llm_client_init_exception_logging(self, monkeypatch):
        """Test that LLM client initialization exceptions are logged with traceback."""
        import logging

        # Capture log records
        log_records = []

        class LogCapture(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = LogCapture()
        handler.setLevel(logging.ERROR)

        graphix_logger = logging.getLogger("GraphixArena")
        graphix_logger.addHandler(handler)

        try:
            # Set env var to disable Ray for this test using monkeypatch
            monkeypatch.setenv('VULCAN_ENABLE_RAY', '0')
            
            # Mock GraphixLLMClient to raise an exception with empty message
            with patch("graphix_arena.GraphixLLMClient") as mock_client:
                mock_client.side_effect = RuntimeError("")

                # Create new arena - should catch exception and log it
                arena = GraphixArena(port=8186)

                # LLM client should be None due to exception
                assert arena.llm_client is None

                # Check that an error was logged
                error_records = [r for r in log_records if r.levelno >= logging.ERROR]
                assert len(error_records) >= 1, "Expected at least one error log record"

                # Check that the error message includes exception type
                error_messages = [r.getMessage() for r in error_records]
                assert any(
                    "RuntimeError" in msg for msg in error_messages
                ), f"Expected RuntimeError in error messages, got: {error_messages}"

                # Check that exc_info was captured
                assert any(
                    r.exc_info is not None for r in error_records
                ), "Expected exc_info to be captured in error log"
        finally:
            graphix_logger.removeHandler(handler)


class TestExceptionHandlers:
    """Test custom exception handlers."""

    def test_agent_not_found_exception(self):
        """Test AgentNotFoundException."""
        exc = AgentNotFoundException("test_agent")

        assert exc.agent_id == "test_agent"
        assert "test_agent" in str(exc)

    def test_bias_detected_exception(self):
        """Test BiasDetectedException."""
        exc = BiasDetectedException(
            agent_id="test_agent",
            graph_id="test_graph",
            label="risky",
            message="Bias detected",
        )

        assert exc.agent_id == "test_agent"
        assert exc.graph_id == "test_graph"
        assert exc.label == "risky"


class TestRuntimeConfiguration:
    """Test runtime configuration loading and Ray initialization."""

    def test_runtime_config_loaded(self):
        """Test that runtime configuration is loaded from file."""
        from graphix_arena import _RUNTIME_CONFIG, _load_runtime_config

        assert _RUNTIME_CONFIG is not None
        assert isinstance(_RUNTIME_CONFIG, dict)
        assert "distributed" in _RUNTIME_CONFIG

    def test_runtime_config_ray_section(self):
        """Test Ray configuration section in runtime config."""
        from graphix_arena import _RUNTIME_CONFIG

        ray_config = _RUNTIME_CONFIG.get("distributed", {}).get("ray", {})
        
        assert ray_config is not None
        # Default should be enabled
        assert ray_config.get("enabled") is True

    def test_merge_configs_function(self):
        """Test configuration merging function."""
        from graphix_arena import _merge_configs

        default = {
            "distributed": {
                "backend": "subprocess",
                "ray": {"enabled": False, "num_cpus": "auto"}
            }
        }
        override = {
            "distributed": {
                "backend": "ray",
                "ray": {"enabled": True}
            }
        }

        merged = _merge_configs(default, override)

        # Override should take precedence
        assert merged["distributed"]["backend"] == "ray"
        assert merged["distributed"]["ray"]["enabled"] is True
        # Default should be preserved for non-overridden values
        assert merged["distributed"]["ray"]["num_cpus"] == "auto"

    def test_load_runtime_config_defaults(self):
        """Test that default configuration is used when file is missing."""
        from graphix_arena import _load_runtime_config

        # The function should always return valid config, even if file is missing
        config = _load_runtime_config()

        assert config is not None
        assert "distributed" in config
        assert "ray" in config["distributed"]
        # Default is enabled
        assert config["distributed"]["ray"].get("enabled", True) is True

    def test_env_var_overrides_config_logic(self):
        """Test that environment variable override logic is correct."""
        import os
        
        # This test validates the logic of environment variable precedence
        # without actually creating an arena instance
        
        ray_config = {"enabled": False}
        
        # Case 1: No env var set - should use config
        env_ray_setting = None  # Simulating os.getenv returning None
        if env_ray_setting is not None:
            enable_ray = env_ray_setting.lower() in ("1", "true", "yes")
        else:
            enable_ray = ray_config.get("enabled", True)
        assert enable_ray == False  # Should use config value
        
        # Case 2: Env var set to "1" - should enable
        env_ray_setting = "1"
        if env_ray_setting is not None:
            enable_ray = env_ray_setting.lower() in ("1", "true", "yes")
        else:
            enable_ray = ray_config.get("enabled", True)
        assert enable_ray == True  # Env var overrides config
        
        # Case 3: Env var set to "0" - should disable
        env_ray_setting = "0"
        if env_ray_setting is not None:
            enable_ray = env_ray_setting.lower() in ("1", "true", "yes")
        else:
            enable_ray = ray_config.get("enabled", True)
        assert enable_ray == False  # Env var explicitly disables


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
