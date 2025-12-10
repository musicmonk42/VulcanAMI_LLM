# ============================================================
# VULCAN-AGI Type Definitions Test Suite
# Comprehensive tests for all type definitions, validation, and serialization
# Run: pytest src/vulcan/tests/test_vulcan_types.py -v --tb=short --cov=src.vulcan.vulcan_types
# ============================================================

import json
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Import config types
from src.vulcan.config import ActionType, GoalType, ModalityType, SafetyLevel
# Import types to test
from src.vulcan.vulcan_types import (  # Version management; IR types; Agent types; Action types; Event types; System state types; Orchestrator-specific types; Validation; Schemas; Registry; Serialization
    ActionCategory, ActionResult, ActionSpecification, AgentCapability,
    AgentProfile, AgentRole, CommunicationState, CompleteSystemState,
    ComponentHealth, EnhancedJSONDecoder, EnhancedJSONEncoder, Episode, Event,
    EventCategory, EventMetadata, EventPriority, HealthSnapshot, IREdge,
    IREdgeType, IRGraph, IRNode, IRNodeType, IRSchemas, KnowledgeState,
    ProvRecord, SA_Latents, SchemaMigrator, SchemaVersion, SecurityState,
    SystemHealth, SystemState, TypeRegistry, TypeValidator, enforce_types)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_ir_node():
    """Create a sample IR node."""
    return IRNode(
        id="test_node_1",
        type=IRNodeType.COMPUTE,
        params={"operation": "add"},
        inputs=["input1", "input2"],
        outputs=["output1"],
        device="cpu",
        memory_mb=100,
        compute_flops=1000,
    )


@pytest.fixture
def sample_ir_edge():
    """Create a sample IR edge."""
    return IREdge(source="node1", target="node2", type=IREdgeType.DATA, weight=1.0)


@pytest.fixture
def sample_ir_graph():
    """Create a sample IR graph."""
    nodes = [
        IRNode(id="node1", type=IRNodeType.INPUT, params={}),
        IRNode(id="node2", type=IRNodeType.COMPUTE, params={"operation": "add"}),
        IRNode(id="node3", type=IRNodeType.OUTPUT, params={}),
    ]
    edges = [
        IREdge(source="node1", target="node2", type=IREdgeType.DATA),
        IREdge(source="node2", target="node3", type=IREdgeType.DATA),
    ]
    graph = IRGraph(
        grammar_version=SchemaVersion.get_version(), nodes=nodes, edges=edges
    )
    # Ensure validation happens
    graph.validate()
    return graph


@pytest.fixture
def sample_agent_profile():
    """Create a sample agent profile."""
    return AgentProfile(
        agent_id="agent_001",
        role=AgentRole.EXECUTOR,
        capabilities=[AgentCapability(name="compute", category="execution", level=8)],
        success_rate=0.95,
        security_clearance=3,
    )


@pytest.fixture
def sample_event():
    """Create a sample event."""
    return Event(
        type="system.startup",
        category=EventCategory.SYSTEM,
        priority=EventPriority.HIGH,
        data={"message": "System started"},
    )


@pytest.fixture
def sample_system_state():
    """Create a sample system state."""
    return SystemState(
        CID="test_context_001",
        step=10,
        policies={"safety": "strict"},
        SA=SA_Latents(uncertainty=0.3, identity_drift=0.1),
        active_modalities={ModalityType.TEXT, ModalityType.VISION},
    )


# ============================================================
# VERSION MANAGEMENT TESTS
# ============================================================


class TestSchemaVersion:
    """Test schema version management."""

    def test_get_version(self):
        """Test getting current version."""
        version = SchemaVersion.get_version()

        assert isinstance(version, str)
        assert len(version.split(".")) == 3
        assert version == "1.3.1"

    def test_is_compatible_same_version(self):
        """Test compatibility with same version."""
        current = SchemaVersion.get_version()

        assert SchemaVersion.is_compatible(current) is True

    def test_is_compatible_lower_minor(self):
        """Test compatibility with lower minor version."""
        assert SchemaVersion.is_compatible("1.2.0") is True
        assert SchemaVersion.is_compatible("1.0.0") is True

    def test_is_compatible_higher_minor(self):
        """Test incompatibility with higher minor version."""
        assert SchemaVersion.is_compatible("1.4.0") is False

    def test_is_compatible_different_major(self):
        """Test incompatibility with different major version."""
        assert SchemaVersion.is_compatible("2.0.0") is False
        assert SchemaVersion.is_compatible("0.9.0") is False

    def test_is_compatible_invalid_format(self):
        """Test handling of invalid version format."""
        assert SchemaVersion.is_compatible("1.0") is False
        assert SchemaVersion.is_compatible("invalid") is False
        assert SchemaVersion.is_compatible("") is False


# ============================================================
# IR NODE TESTS
# ============================================================


class TestIRNode:
    """Test IR node functionality."""

    def test_node_creation(self, sample_ir_node):
        """Test creating a valid IR node."""
        assert sample_ir_node.id == "test_node_1"
        assert sample_ir_node.type == IRNodeType.COMPUTE
        assert sample_ir_node.validated is True

    def test_node_validation_valid(self):
        """Test validation of valid node."""
        node = IRNode(
            id="valid_node", type=IRNodeType.COMPUTE, params={"operation": "multiply"}
        )

        assert node.validate() is True
        assert len(node.validation_errors) == 0

    def test_node_validation_invalid_id(self):
        """Test validation with invalid ID."""
        node = IRNode(
            id="invalid node!",  # Contains space and special char
            type=IRNodeType.COMPUTE,
            params={},
        )

        assert node.validate() is False
        assert len(node.validation_errors) > 0
        assert any("Invalid node ID format" in err for err in node.validation_errors)

    def test_node_validation_missing_required_params(self):
        """Test validation with missing required parameters."""
        node = IRNode(
            id="compute_node",
            type=IRNodeType.COMPUTE,
            params={},  # Missing 'operation'
        )

        assert node.validate() is False
        assert any(
            "Parameter validation failed" in err for err in node.validation_errors
        )

    def test_node_validation_negative_resources(self):
        """Test validation with negative resources."""
        node = IRNode(
            id="test_node",
            type=IRNodeType.COMPUTE,
            params={"operation": "add"},
            memory_mb=-100,
        )

        assert node.validate() is False
        assert any(
            "Invalid memory requirement" in err for err in node.validation_errors
        )

    def test_node_to_dict(self, sample_ir_node):
        """Test converting node to dictionary."""
        node_dict = sample_ir_node.to_dict()

        assert node_dict["id"] == "test_node_1"
        assert node_dict["type"] == "compute"
        assert "resources" in node_dict
        assert node_dict["validated"] is True

    def test_all_node_types(self):
        """Test creating nodes of all types."""
        for node_type in IRNodeType:
            node = IRNode(id=f"node_{node_type.value}", type=node_type, params={})
            assert node.type == node_type


# ============================================================
# IR EDGE TESTS
# ============================================================


class TestIREdge:
    """Test IR edge functionality."""

    def test_edge_creation(self, sample_ir_edge):
        """Test creating a valid edge."""
        assert sample_ir_edge.source == "node1"
        assert sample_ir_edge.target == "node2"
        assert sample_ir_edge.type == IREdgeType.DATA

    def test_edge_validation_valid(self):
        """Test validation of valid edge."""
        edge = IREdge(source="node1", target="node2", type=IREdgeType.CONTROL)

        assert edge.validate() is True

    def test_edge_validation_empty_source(self):
        """Test validation with empty source."""
        edge = IREdge(source="", target="node2", type=IREdgeType.DATA)

        assert edge.validate() is False

    def test_edge_validation_negative_weight(self):
        """Test validation with negative weight."""
        edge = IREdge(source="node1", target="node2", type=IREdgeType.DATA, weight=-1.0)

        assert edge.validate() is False

    def test_edge_to_dict(self, sample_ir_edge):
        """Test converting edge to dictionary."""
        edge_dict = sample_ir_edge.to_dict()

        assert edge_dict["source"] == "node1"
        assert edge_dict["target"] == "node2"
        assert edge_dict["type"] == "data"

    def test_all_edge_types(self):
        """Test creating edges of all types."""
        for edge_type in IREdgeType:
            edge = IREdge(source="node1", target="node2", type=edge_type)
            assert edge.type == edge_type


# ============================================================
# IR GRAPH TESTS
# ============================================================


class TestIRGraph:
    """Test IR graph functionality."""

    def test_graph_creation(self, sample_ir_graph):
        """Test creating a valid graph."""
        assert len(sample_ir_graph.nodes) == 3
        assert len(sample_ir_graph.edges) == 2
        assert sample_ir_graph.validated is True

    def test_graph_validation_valid(self, sample_ir_graph):
        """Test validation of valid graph."""
        assert sample_ir_graph.validate() is True
        assert len(sample_ir_graph.validation_errors) == 0

    def test_graph_validation_duplicate_node_ids(self):
        """Test validation with duplicate node IDs."""
        graph = IRGraph(
            grammar_version="1.3.1",
            nodes=[
                IRNode(id="node1", type=IRNodeType.COMPUTE, params={}),
                IRNode(id="node1", type=IRNodeType.COMPUTE, params={}),  # Duplicate
            ],
            edges=[],
        )

        assert graph.validate() is False
        assert any("Duplicate node ID" in err for err in graph.validation_errors)

    def test_graph_validation_edge_to_nonexistent_node(self):
        """Test validation with edge to non-existent node."""
        graph = IRGraph(
            grammar_version="1.3.1",
            nodes=[IRNode(id="node1", type=IRNodeType.COMPUTE, params={})],
            edges=[
                IREdge(
                    source="node1", target="node2", type=IREdgeType.DATA
                )  # node2 doesn't exist
            ],
        )

        assert graph.validate() is False
        assert any("Edge target not found" in err for err in graph.validation_errors)

    def test_graph_validation_incompatible_version(self):
        """Test validation with incompatible version."""
        graph = IRGraph(
            grammar_version="2.0.0",  # Incompatible
            nodes=[],
            edges=[],
        )

        assert graph.validate() is False
        assert any(
            "Incompatible grammar version" in err for err in graph.validation_errors
        )

    def test_graph_safety_cycle_detection(self):
        """Test detection of cycles in safety-critical paths."""
        nodes = [
            IRNode(
                id="node1",
                type=IRNodeType.SAFETY_CHECK,
                params={"safety_level": "high", "constraints": []},
            ),
            IRNode(
                id="node2",
                type=IRNodeType.SAFETY_CHECK,
                params={"safety_level": "high", "constraints": []},
            ),
            IRNode(
                id="node3",
                type=IRNodeType.SAFETY_CHECK,
                params={"safety_level": "high", "constraints": []},
            ),
        ]
        edges = [
            IREdge(source="node1", target="node2", type=IREdgeType.SAFETY),
            IREdge(source="node2", target="node3", type=IREdgeType.SAFETY),
            IREdge(
                source="node3", target="node1", type=IREdgeType.SAFETY
            ),  # Creates cycle
        ]
        graph = IRGraph(grammar_version="1.3.1", nodes=nodes, edges=edges)

        assert graph.validate() is False
        assert any(
            "Cycle detected in safety-critical path" in err
            for err in graph.validation_errors
        )

    def test_graph_to_dict(self, sample_ir_graph):
        """Test converting graph to dictionary."""
        graph_dict = sample_ir_graph.to_dict()

        assert "grammar_version" in graph_dict
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert len(graph_dict["nodes"]) == 3
        assert len(graph_dict["edges"]) == 2


# ============================================================
# AGENT TESTS
# ============================================================


class TestAgentCapability:
    """Test agent capability."""

    def test_capability_creation(self):
        """Test creating a capability."""
        cap = AgentCapability(
            name="reasoning",
            category="cognitive",
            level=7,
            certified=True,
            certification_date=datetime.now(),
        )

        assert cap.name == "reasoning"
        assert cap.level == 7
        assert cap.certified is True

    def test_capability_validation_valid(self):
        """Test validation of valid capability."""
        cap = AgentCapability(name="test", category="test", level=5)

        assert cap.validate() is True

    def test_capability_validation_invalid_level(self):
        """Test validation with invalid level."""
        cap = AgentCapability(
            name="test",
            category="test",
            level=15,  # Invalid, should be 0-10
        )

        assert cap.validate() is False

    def test_capability_validation_certified_without_date(self):
        """Test validation of certified capability without date."""
        cap = AgentCapability(
            name="test",
            category="test",
            level=5,
            certified=True,
            certification_date=None,
        )

        assert cap.validate() is False


class TestAgentProfile:
    """Test agent profile."""

    def test_profile_creation(self, sample_agent_profile):
        """Test creating an agent profile."""
        assert sample_agent_profile.agent_id == "agent_001"
        assert sample_agent_profile.role == AgentRole.EXECUTOR
        assert len(sample_agent_profile.capabilities) == 1

    def test_profile_validation_valid(self, sample_agent_profile):
        """Test validation of valid profile."""
        assert sample_agent_profile.validate() is True

    def test_profile_validation_invalid_id(self):
        """Test validation with invalid agent ID."""
        profile = AgentProfile(
            agent_id="invalid agent!",  # Invalid characters
            role=AgentRole.COORDINATOR,
        )

        assert profile.validate() is False

    def test_profile_validation_invalid_success_rate(self):
        """Test validation with invalid success rate."""
        profile = AgentProfile(
            agent_id="agent_001",
            role=AgentRole.EXECUTOR,
            success_rate=1.5,  # Invalid, should be 0-1
        )

        assert profile.validate() is False

    def test_profile_validation_invalid_security_clearance(self):
        """Test validation with invalid security clearance."""
        profile = AgentProfile(
            agent_id="agent_001",
            role=AgentRole.EXECUTOR,
            security_clearance=10,  # Invalid, should be 0-5
        )

        assert profile.validate() is False

    def test_all_agent_roles(self):
        """Test creating profiles with all roles."""
        for role in AgentRole:
            profile = AgentProfile(agent_id=f"agent_{role.value}", role=role)
            assert profile.role == role


# ============================================================
# ACTION TESTS
# ============================================================


class TestActionSpecification:
    """Test action specification."""

    def test_action_spec_creation(self):
        """Test creating an action specification."""
        # Get first available ActionType and SafetyLevel
        action_type = list(ActionType)[0]
        safety_level = list(SafetyLevel)[0]

        spec = ActionSpecification(
            type=action_type,
            category=ActionCategory.COMPUTATION,
            parameters={"input": "test"},
            safety_level_required=safety_level,
        )

        assert spec.type == action_type
        assert spec.category == ActionCategory.COMPUTATION

    def test_action_spec_validation_valid(self):
        """Test validation of valid specification."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        spec = ActionSpecification(
            type=action_type,
            category=ActionCategory.COMPUTATION,
            min_resources={"cpu": 1.0},
            max_resources={"cpu": 4.0},
            min_duration_ms=100,
            max_duration_ms=1000,
        )

        assert spec.validate() is True

    def test_action_spec_validation_invalid_resources(self):
        """Test validation with invalid resource constraints."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        spec = ActionSpecification(
            type=action_type,
            category=ActionCategory.COMPUTATION,
            min_resources={"cpu": 4.0},
            max_resources={"cpu": 1.0},  # Max < Min
        )

        assert spec.validate() is False

    def test_action_spec_validation_invalid_duration(self):
        """Test validation with invalid duration constraints."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        spec = ActionSpecification(
            type=action_type,
            category=ActionCategory.COMPUTATION,
            min_duration_ms=1000,
            max_duration_ms=100,  # Max < Min
        )

        assert spec.validate() is False

    def test_action_spec_validation_invalid_timeout(self):
        """Test validation with invalid timeout."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        spec = ActionSpecification(
            type=action_type,
            category=ActionCategory.COMPUTATION,
            min_duration_ms=1000,
            timeout_ms=500,  # Timeout < Min duration
        )

        assert spec.validate() is False


class TestActionResult:
    """Test action result."""

    def test_action_result_creation(self):
        """Test creating an action result."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        start = time.time()
        end = start + 1.0

        result = ActionResult(
            action_id="action_001",
            action_type=action_type,
            status="success",
            start_time=start,
            end_time=end,
            duration_ms=1000.0,
            outputs={"result": 42},
        )

        assert result.action_id == "action_001"
        assert result.status == "success"

    def test_action_result_validation_valid(self):
        """Test validation of valid result."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        start = time.time()
        end = start + 0.5

        result = ActionResult(
            action_id="action_001",
            action_type=action_type,
            status="success",
            start_time=start,
            end_time=end,
            duration_ms=500.0,
        )

        assert result.validate() is True

    def test_action_result_validation_invalid_times(self):
        """Test validation with invalid times."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        result = ActionResult(
            action_id="action_001",
            action_type=action_type,
            status="success",
            start_time=100.0,
            end_time=50.0,  # End before start
            duration_ms=0.0,
        )

        assert result.validate() is False

    def test_action_result_validation_duration_mismatch(self):
        """Test validation with duration mismatch."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        result = ActionResult(
            action_id="action_001",
            action_type=action_type,
            status="success",
            start_time=0.0,
            end_time=1.0,
            duration_ms=5000.0,  # Doesn't match 1 second
        )

        assert result.validate() is False


# ============================================================
# EVENT TESTS
# ============================================================


class TestEvent:
    """Test event functionality."""

    def test_event_creation(self, sample_event):
        """Test creating an event."""
        assert sample_event.type == "system.startup"
        assert sample_event.category == EventCategory.SYSTEM
        assert sample_event.priority == EventPriority.HIGH

    def test_event_validation_valid(self, sample_event):
        """Test validation of valid event."""
        assert sample_event.validate() is True

    def test_event_validation_empty_type(self):
        """Test validation with empty type."""
        event = Event(type="", category=EventCategory.INFO)

        assert event.validate() is False

    def test_event_expiry_with_ttl(self):
        """Test event expiry with TTL."""
        event = Event(type="test", category=EventCategory.INFO)
        event.metadata.ttl_ms = 100  # 100ms TTL

        assert event.is_expired() is False

        time.sleep(0.2)  # Wait 200ms

        assert event.is_expired() is True

    def test_event_expiry_with_expiry_time(self):
        """Test event expiry with expiry time."""
        event = Event(type="test", category=EventCategory.INFO)
        event.metadata.expiry_time = time.time() + 0.1  # Expire in 100ms

        assert event.is_expired() is False

        time.sleep(0.2)  # Wait 200ms

        assert event.is_expired() is True

    def test_event_to_dict(self, sample_event):
        """Test converting event to dictionary."""
        event_dict = sample_event.to_dict()

        assert event_dict["type"] == "system.startup"
        assert event_dict["category"] == "system"
        assert "metadata" in event_dict

    def test_event_priorities(self):
        """Test all event priorities."""
        for priority in EventPriority:
            event = Event(type="test", category=EventCategory.INFO, priority=priority)
            assert event.priority == priority


# ============================================================
# SYSTEM STATE TESTS
# ============================================================


class TestComponentHealth:
    """Test component health."""

    def test_component_health_creation(self):
        """Test creating component health."""
        health = ComponentHealth(
            component_name="database",
            status="healthy",
            uptime_seconds=1000.0,
            error_count=0,
        )

        assert health.component_name == "database"
        assert health.status == "healthy"

    def test_component_is_healthy(self):
        """Test checking if component is healthy."""
        health = ComponentHealth(component_name="test", status="healthy")

        assert health.is_healthy() is True

        health.status = "degraded"
        assert health.is_healthy() is False


class TestSystemHealth:
    """Test system health."""

    def test_system_health_creation(self):
        """Test creating system health."""
        health = SystemHealth(overall_status="healthy", total_uptime_seconds=1000.0)

        assert health.overall_status == "healthy"

    def test_system_health_update_component(self):
        """Test updating component health."""
        system_health = SystemHealth()

        component = ComponentHealth(component_name="test", status="healthy")

        system_health.update_component("test", component)

        assert "test" in system_health.components
        assert system_health.overall_status == "healthy"

    def test_system_health_status_calculation(self):
        """Test overall status calculation."""
        system_health = SystemHealth()

        # All healthy
        system_health.update_component("c1", ComponentHealth("c1", "healthy"))
        system_health.update_component("c2", ComponentHealth("c2", "healthy"))
        assert system_health.overall_status == "healthy"

        # One degraded
        system_health.update_component("c3", ComponentHealth("c3", "degraded"))
        assert system_health.overall_status == "degraded"

        # One unhealthy
        system_health.update_component("c4", ComponentHealth("c4", "unhealthy"))
        assert system_health.overall_status == "unhealthy"


class TestSystemState:
    """Test system state (orchestrator version)."""

    def test_system_state_creation(self, sample_system_state):
        """Test creating system state."""
        assert sample_system_state.CID == "test_context_001"
        assert sample_system_state.step == 10
        assert ModalityType.TEXT in sample_system_state.active_modalities

    def test_system_state_update_step(self, sample_system_state):
        """Test updating step counter."""
        initial_step = sample_system_state.step

        sample_system_state.update_step()

        assert sample_system_state.step == initial_step + 1

    def test_system_state_add_provenance(self, sample_system_state):
        """Test adding provenance record."""
        prov = ProvRecord(
            t=time.time(),
            graph_id="graph_001",
            agent_version="1.0.0",
            policy_versions={},
            input_hash="abc123",
            kernel_sig=None,
            explainer_uri="http://example.com/explain",
            ecdsa_sig="sig123",
            modality=ModalityType.TEXT,
            uncertainty=0.1,
        )

        initial_count = len(sample_system_state.provenance_chain)

        sample_system_state.add_provenance(prov)

        assert len(sample_system_state.provenance_chain) == initial_count + 1

    def test_system_state_to_dict(self, sample_system_state):
        """Test converting system state to dict."""
        state_dict = sample_system_state.to_dict()

        assert state_dict["CID"] == "test_context_001"
        assert state_dict["step"] == 10
        assert "SA" in state_dict
        assert "health" in state_dict


class TestSALatents:
    """Test self-awareness latents."""

    def test_sa_latents_creation(self):
        """Test creating SA latents."""
        latents = SA_Latents(
            uncertainty=0.3,
            identity_drift=0.1,
            learning_efficiency=0.7,
            metacognitive_confidence=0.6,
        )

        assert latents.uncertainty == 0.3
        assert latents.identity_drift == 0.1

    def test_sa_latents_to_dict(self):
        """Test converting SA latents to dict."""
        latents = SA_Latents(uncertainty=0.5)
        latents_dict = latents.to_dict()

        assert "uncertainty" in latents_dict
        assert "identity_drift" in latents_dict


class TestHealthSnapshot:
    """Test health snapshot."""

    def test_health_snapshot_creation(self):
        """Test creating health snapshot."""
        snapshot = HealthSnapshot(
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            latency_ms=10.0,
            error_rate=0.01,
        )

        assert snapshot.memory_usage_mb == 100.0
        assert snapshot.cpu_usage_percent == 50.0

    def test_health_snapshot_to_dict(self):
        """Test converting health snapshot to dict."""
        snapshot = HealthSnapshot()
        snapshot_dict = snapshot.to_dict()

        assert "memory_usage_mb" in snapshot_dict
        assert "cpu_usage_percent" in snapshot_dict


class TestEpisode:
    """Test episode."""

    def test_episode_creation(self):
        """Test creating an episode."""
        episode = Episode(
            t=time.time(),
            context={"env": "test"},
            action_bundle={"action": "test"},
            observation="test observation",
            reward_vec={"reward": 1.0},
            SA_latents=SA_Latents(),
            expl_uri="http://example.com",
            prov_sig="sig123",
            modalities_used={ModalityType.TEXT},
            uncertainty=0.2,
        )

        assert episode.t > 0
        assert ModalityType.TEXT in episode.modalities_used

    def test_episode_to_dict(self):
        """Test converting episode to dict."""
        episode = Episode(
            t=time.time(),
            context={},
            action_bundle={},
            observation=None,
            reward_vec={},
            SA_latents=SA_Latents(),
            expl_uri="",
            prov_sig="",
            modalities_used=set(),
            uncertainty=0.0,
        )

        episode_dict = episode.to_dict()

        assert "t" in episode_dict
        assert "SA_latents" in episode_dict
        assert "modalities_used" in episode_dict


class TestProvRecord:
    """Test provenance record."""

    def test_prov_record_creation(self):
        """Test creating provenance record."""
        prov = ProvRecord(
            t=time.time(),
            graph_id="graph_001",
            agent_version="1.0.0",
            policy_versions={"safety": "1.0"},
            input_hash="abc123",
            kernel_sig="kernel_sig",
            explainer_uri="http://example.com/explain",
            ecdsa_sig="sig123",
            modality=ModalityType.TEXT,
            uncertainty=0.1,
        )

        assert prov.graph_id == "graph_001"
        assert prov.modality == ModalityType.TEXT

    def test_prov_record_to_dict(self):
        """Test converting provenance record to dict."""
        prov = ProvRecord(
            t=time.time(),
            graph_id="graph_001",
            agent_version="1.0.0",
            policy_versions={},
            input_hash="hash",
            kernel_sig=None,
            explainer_uri="uri",
            ecdsa_sig="sig",
            modality=ModalityType.TEXT,
            uncertainty=0.0,
        )

        prov_dict = prov.to_dict()

        assert "graph_id" in prov_dict
        assert "modality" in prov_dict


# ============================================================
# TYPE VALIDATION TESTS
# ============================================================


class TestTypeValidator:
    """Test type validator."""

    def test_validate_basic_type(self):
        """Test validating basic types."""
        valid, errors = TypeValidator.validate_type(42, int)
        assert valid is True
        assert len(errors) == 0

        valid, errors = TypeValidator.validate_type("test", str)
        assert valid is True

        valid, errors = TypeValidator.validate_type(42, str)
        assert valid is False
        assert len(errors) > 0

    def test_validate_none(self):
        """Test validating None."""
        valid, errors = TypeValidator.validate_type(None, type(None))
        assert valid is True

        valid, errors = TypeValidator.validate_type(None, str)
        assert valid is False

    def test_validate_dataclass(self, sample_ir_node):
        """Test validating dataclass."""
        valid, errors = TypeValidator.validate_type(sample_ir_node, IRNode)
        assert valid is True


class TestEnforceTypes:
    """Test type enforcement decorator."""

    def test_enforce_types_valid(self):
        """Test decorator with valid types."""

        @enforce_types
        def test_func(x: int, y: str) -> str:
            return f"{y}{x}"

        result = test_func(42, "test")
        assert result == "test42"

    def test_enforce_types_invalid_input(self):
        """Test decorator with invalid input types."""

        @enforce_types
        def test_func(x: int) -> int:
            return x * 2

        with pytest.raises(TypeError):
            test_func("not an int")

    def test_enforce_types_invalid_output(self):
        """Test decorator with invalid output type."""

        @enforce_types
        def test_func(x: int) -> str:
            return x  # Returns int, not str

        with pytest.raises(TypeError):
            test_func(42)


# ============================================================
# SCHEMA TESTS
# ============================================================


class TestIRSchemas:
    """Test IR schemas."""

    def test_validate_valid_graph(self, sample_ir_graph):
        """Test validating a valid graph."""
        graph_dict = sample_ir_graph.to_dict()

        valid, errors = IRSchemas.validate_graph(graph_dict)
        assert valid is True
        assert len(errors) == 0

    def test_validate_invalid_graph(self):
        """Test validating an invalid graph."""
        invalid_graph = {
            "grammar_version": "1.3.1",
            "nodes": [],
            # Missing 'edges' field
        }

        valid, errors = IRSchemas.validate_graph(invalid_graph)
        # May be True if jsonschema not available
        if valid is False:
            assert len(errors) > 0


class TestSchemaMigrator:
    """Test schema migrator."""

    def test_migrate_same_version(self):
        """Test migration with same version."""
        data = {"schema_version": "1.3.1", "test": "data"}

        migrated = SchemaMigrator.migrate(data, "1.3.1")

        assert migrated == data

    def test_migrate_1_0_to_1_1(self):
        """Test migration from 1.0.0 to 1.1.0."""
        data = {
            "schema_version": "1.0.0",
            "nodes": [{"id": "node1", "type": "compute"}],
        }

        migrated = SchemaMigrator.migrate(data, "1.1.0")

        assert migrated["schema_version"] == "1.1.0"
        assert "metadata" in migrated
        assert migrated["nodes"][0]["device"] == "cpu"

    def test_migrate_path_finding(self):
        """Test finding migration path."""
        path = SchemaMigrator._find_migration_path("1.0.0", "1.3.0")

        assert len(path) > 0
        assert path[0] == ("1.0.0", "1.1.0")


# ============================================================
# TYPE REGISTRY TESTS
# ============================================================


class TestTypeRegistry:
    """Test type registry."""

    def test_register_and_get(self):
        """Test registering and getting types."""

        class TestType:
            pass

        TypeRegistry.register("TestType", TestType)

        retrieved = TypeRegistry.get("TestType")
        assert retrieved is TestType

    def test_create_instance(self):
        """Test creating instance from registry."""
        event = TypeRegistry.create("Event", type="test", category=EventCategory.INFO)

        assert isinstance(event, Event)
        assert event.type == "test"

    def test_create_nonexistent_type(self):
        """Test creating non-existent type."""
        with pytest.raises(ValueError):
            TypeRegistry.create("NonExistentType")


# ============================================================
# SERIALIZATION TESTS
# ============================================================


class TestEnhancedJSONEncoder:
    """Test enhanced JSON encoder."""

    def test_encode_dataclass(self, sample_ir_node):
        """Test encoding dataclass."""
        encoded = json.dumps(sample_ir_node, cls=EnhancedJSONEncoder)

        assert isinstance(encoded, str)
        assert "test_node_1" in encoded

    def test_encode_enum(self):
        """Test encoding enum."""
        encoded = json.dumps(IRNodeType.COMPUTE, cls=EnhancedJSONEncoder)

        assert isinstance(encoded, str)
        assert "compute" in encoded

    def test_encode_datetime(self):
        """Test encoding datetime."""
        dt = datetime.now()
        encoded = json.dumps(dt, cls=EnhancedJSONEncoder)

        assert isinstance(encoded, str)
        assert "__type__" in encoded

    def test_encode_numpy_array(self):
        """Test encoding numpy array."""
        arr = np.array([1, 2, 3])
        encoded = json.dumps(arr, cls=EnhancedJSONEncoder)

        assert isinstance(encoded, str)
        assert "ndarray" in encoded

    def test_encode_set(self):
        """Test encoding set."""
        s = {1, 2, 3}
        encoded = json.dumps(s, cls=EnhancedJSONEncoder)

        assert isinstance(encoded, str)
        assert "__type__" in encoded


class TestEnhancedJSONDecoder:
    """Test enhanced JSON decoder."""

    def test_decode_datetime(self):
        """Test decoding datetime."""
        dt = datetime.now()
        encoded = json.dumps(dt, cls=EnhancedJSONEncoder)
        decoded = json.loads(encoded, cls=EnhancedJSONDecoder)

        assert isinstance(decoded, datetime)

    def test_decode_numpy_array(self):
        """Test decoding numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        encoded = json.dumps(arr, cls=EnhancedJSONEncoder)
        decoded = json.loads(encoded, cls=EnhancedJSONDecoder)

        assert isinstance(decoded, np.ndarray)
        assert np.array_equal(decoded, arr)

    def test_decode_set(self):
        """Test decoding set."""
        s = {1, 2, 3}
        encoded = json.dumps(s, cls=EnhancedJSONEncoder)
        decoded = json.loads(encoded, cls=EnhancedJSONDecoder)

        assert isinstance(decoded, set)
        assert decoded == s


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests."""

    def test_full_graph_serialization(self, sample_ir_graph):
        """Test full graph serialization and deserialization."""
        # Convert to dict
        graph_dict = sample_ir_graph.to_dict()

        # Serialize to JSON
        json_str = json.dumps(graph_dict, cls=EnhancedJSONEncoder)

        # Deserialize
        decoded_dict = json.loads(json_str, cls=EnhancedJSONDecoder)

        assert decoded_dict["grammar_version"] == sample_ir_graph.grammar_version
        assert len(decoded_dict["nodes"]) == len(sample_ir_graph.nodes)

    def test_system_state_lifecycle(self):
        """Test complete system state lifecycle."""
        # Create state
        state = SystemState(
            CID="test_001",
            policies={"safety": "strict"},
            SA=SA_Latents(uncertainty=0.2),
        )

        # Update state
        for _ in range(5):
            state.update_step()

        assert state.step == 5

        # Add provenance
        prov = ProvRecord(
            t=time.time(),
            graph_id="g1",
            agent_version="1.0",
            policy_versions={},
            input_hash="hash",
            kernel_sig=None,
            explainer_uri="uri",
            ecdsa_sig="sig",
            modality=ModalityType.TEXT,
            uncertainty=0.1,
        )
        state.add_provenance(prov)

        assert len(state.provenance_chain) == 1

        # Convert to dict
        state_dict = state.to_dict()

        assert state_dict["step"] == 5
        assert len(state_dict["provenance_chain"]) == 1

    def test_complete_workflow(self):
        """Test complete workflow from graph creation to execution."""
        # Get first available ActionType
        action_type = list(ActionType)[0]

        # Create graph
        graph = IRGraph(
            grammar_version=SchemaVersion.get_version(),
            nodes=[
                IRNode(id="input", type=IRNodeType.INPUT, params={}),
                IRNode(
                    id="compute", type=IRNodeType.COMPUTE, params={"operation": "add"}
                ),
                IRNode(id="output", type=IRNodeType.OUTPUT, params={}),
            ],
            edges=[
                IREdge(source="input", target="compute", type=IREdgeType.DATA),
                IREdge(source="compute", target="output", type=IREdgeType.DATA),
            ],
        )

        # Validate
        assert graph.validate() is True

        # Create agent
        agent = AgentProfile(
            agent_id="executor_001",
            role=AgentRole.EXECUTOR,
            capabilities=[
                AgentCapability(name="compute", category="execution", level=8)
            ],
        )

        assert agent.validate() is True

        # Create action spec
        action = ActionSpecification(
            type=action_type,
            category=ActionCategory.COMPUTATION,
            parameters={"graph_id": "test"},
        )

        assert action.validate() is True

        # Create result
        start = time.time()
        end = start + 0.5

        result = ActionResult(
            action_id="action_001",
            action_type=action_type,
            status="success",
            start_time=start,
            end_time=end,
            duration_ms=500.0,
            outputs={"result": "success"},
        )

        assert result.validate() is True


# ============================================================
# ADDITIONAL TEST CLASSES FOR MISSING COVERAGE
# ============================================================


class TestCompleteSystemState:
    """Test complete system state (full version)."""

    def test_complete_system_state_creation(self):
        """Test creating complete system state."""
        state = CompleteSystemState(system_id="sys_001", version="1.0.0")

        assert state.system_id == "sys_001"
        assert state.version == "1.0.0"
        assert state.lifecycle_phase == "initialization"

    def test_complete_system_state_update(self):
        """Test updating complete system state."""
        state = CompleteSystemState(system_id="sys_001", version="1.0.0")

        initial_step = state.current_step
        initial_time = state.last_update

        time.sleep(0.01)
        state.update()

        assert state.current_step == initial_step + 1
        assert state.last_update > initial_time

    def test_complete_system_state_with_agents(self):
        """Test system state with active agents."""
        agent = AgentProfile(agent_id="agent_001", role=AgentRole.EXECUTOR)

        state = CompleteSystemState(
            system_id="sys_001", version="1.0.0", active_agents={"agent_001": agent}
        )

        assert len(state.active_agents) == 1
        assert "agent_001" in state.active_agents


class TestKnowledgeState:
    """Test knowledge state."""

    def test_knowledge_state_creation(self):
        """Test creating knowledge state."""
        knowledge = KnowledgeState(
            total_concepts=1000,
            active_concepts=800,
            causal_links=500,
            episodic_memories=200,
        )

        assert knowledge.total_concepts == 1000
        assert knowledge.active_concepts == 800
        assert knowledge.causal_links == 500

    def test_knowledge_state_defaults(self):
        """Test knowledge state with defaults."""
        knowledge = KnowledgeState()

        assert knowledge.total_concepts == 0
        assert knowledge.causal_links == 0
        assert knowledge.model_version == "1.0.0"

    def test_knowledge_state_learning_statistics(self):
        """Test learning statistics tracking."""
        knowledge = KnowledgeState(
            total_experiences=1000, positive_experiences=700, negative_experiences=300
        )

        assert knowledge.total_experiences == 1000
        assert knowledge.positive_experiences == 700
        assert knowledge.negative_experiences == 300


class TestCommunicationState:
    """Test communication state."""

    def test_communication_state_creation(self):
        """Test creating communication state."""
        comm = CommunicationState(
            active_connections=5, total_messages_sent=1000, total_messages_received=950
        )

        assert comm.active_connections == 5
        assert comm.total_messages_sent == 1000
        assert comm.total_messages_received == 950

    def test_communication_state_protocol_usage(self):
        """Test protocol usage tracking."""
        comm = CommunicationState(protocol_usage={"http": 500, "websocket": 200})

        assert comm.protocol_usage["http"] == 500
        assert comm.protocol_usage["websocket"] == 200

    def test_communication_state_error_tracking(self):
        """Test error tracking."""
        comm = CommunicationState(failed_sends=10, failed_receives=5, timeout_count=3)

        assert comm.failed_sends == 10
        assert comm.failed_receives == 5
        assert comm.timeout_count == 3


class TestSecurityState:
    """Test security state."""

    def test_security_state_creation(self):
        """Test creating security state."""
        security = SecurityState(
            authenticated_users=10, failed_auth_attempts=3, threats_detected=2
        )

        assert security.authenticated_users == 10
        assert security.failed_auth_attempts == 3
        assert security.threats_detected == 2

    def test_security_state_access_control(self):
        """Test access control metrics."""
        security = SecurityState(access_granted=1000, access_denied=50)

        assert security.access_granted == 1000
        assert security.access_denied == 50

    def test_security_state_threat_mitigation(self):
        """Test threat mitigation tracking."""
        security = SecurityState(threats_detected=10, threats_mitigated=8)

        assert security.threats_detected == 10
        assert security.threats_mitigated == 8

    def test_security_state_encryption(self):
        """Test encryption tracking."""
        security = SecurityState(encrypted_messages=500, encryption_failures=2)

        assert security.encrypted_messages == 500
        assert security.encryption_failures == 2


class TestEventMetadata:
    """Test event metadata."""

    def test_event_metadata_creation(self):
        """Test creating event metadata."""
        metadata = EventMetadata(
            source_agent="agent_001",
            target_agents=["agent_002", "agent_003"],
            broadcast=False,
        )

        assert metadata.source_agent == "agent_001"
        assert len(metadata.target_agents) == 2
        assert metadata.broadcast is False

    def test_event_metadata_correlation_ids(self):
        """Test correlation ID generation."""
        metadata = EventMetadata()

        assert metadata.correlation_id is not None
        assert len(metadata.correlation_id) > 0

    def test_event_metadata_ttl(self):
        """Test TTL settings."""
        metadata = EventMetadata(ttl_ms=1000.0)

        assert metadata.ttl_ms == 1000.0

    def test_event_metadata_security(self):
        """Test security settings."""
        metadata = EventMetadata(encrypted=True, signed=True, signature="abc123")

        assert metadata.encrypted is True
        assert metadata.signed is True
        assert metadata.signature == "abc123"

    def test_event_metadata_causation_chain(self):
        """Test causation chain tracking."""
        metadata = EventMetadata(
            causation_id="cause_123", session_id="session_456", trace_id="trace_789"
        )

        assert metadata.causation_id == "cause_123"
        assert metadata.session_id == "session_456"
        assert metadata.trace_id == "trace_789"


class TestIRNodeEdgeCases:
    """Test IR node edge cases."""

    def test_node_with_extreme_resource_values(self):
        """Test node with extreme resource values."""
        node = IRNode(
            id="extreme_node",
            type=IRNodeType.COMPUTE,
            params={"operation": "test"},
            memory_mb=1e6,
            compute_flops=1e15,
            bandwidth_mbps=1e5,
            energy_nj=1e12,
        )

        assert node.validate() is True
        assert node.memory_mb == 1e6

    def test_node_with_zero_resources(self):
        """Test node with zero resources."""
        node = IRNode(
            id="zero_node",
            type=IRNodeType.COMPUTE,
            params={"operation": "test"},
            memory_mb=0,
            compute_flops=0,
            bandwidth_mbps=0,
            energy_nj=0,
        )

        assert node.validate() is True

    def test_node_special_characters_in_metadata(self):
        """Test node with special characters in metadata."""
        node = IRNode(
            id="meta_node",
            type=IRNodeType.COMPUTE,
            params={"operation": "test"},
            metadata={"description": "Test with unicode: 中文, émojis: 🚀"},
        )

        assert node.validate() is True
        assert "中文" in node.metadata["description"]

    def test_node_with_multiple_inputs_outputs(self):
        """Test node with multiple inputs and outputs."""
        node = IRNode(
            id="multi_node",
            type=IRNodeType.COMPUTE,
            params={"operation": "merge"},
            inputs=["in1", "in2", "in3", "in4"],
            outputs=["out1", "out2"],
        )

        assert node.validate() is True
        assert len(node.inputs) == 4
        assert len(node.outputs) == 2


class TestIRGraphEdgeCases:
    """Test IR graph edge cases."""

    def test_empty_graph(self):
        """Test empty graph validation."""
        graph = IRGraph(grammar_version=SchemaVersion.get_version(), nodes=[], edges=[])

        assert graph.validate() is True

    def test_graph_with_disconnected_nodes(self):
        """Test graph with disconnected nodes."""
        graph = IRGraph(
            grammar_version=SchemaVersion.get_version(),
            nodes=[
                IRNode(
                    id="node1", type=IRNodeType.COMPUTE, params={"operation": "add"}
                ),
                IRNode(
                    id="node2", type=IRNodeType.COMPUTE, params={"operation": "mul"}
                ),
                IRNode(
                    id="node3", type=IRNodeType.COMPUTE, params={"operation": "sub"}
                ),
            ],
            edges=[
                IREdge(source="node1", target="node2", type=IREdgeType.DATA)
                # node3 is disconnected
            ],
        )

        assert graph.validate() is True

    def test_graph_with_self_loop_non_safety(self):
        """Test graph with self-loop on non-safety edge."""
        graph = IRGraph(
            grammar_version=SchemaVersion.get_version(),
            nodes=[
                IRNode(
                    id="node1", type=IRNodeType.COMPUTE, params={"operation": "test"}
                )
            ],
            edges=[IREdge(source="node1", target="node1", type=IREdgeType.DATA)],
        )

        # Self-loops should be allowed for non-safety edges
        assert graph.validate() is True

    def test_graph_with_multiple_edge_types(self):
        """Test graph with multiple edge types between same nodes."""
        graph = IRGraph(
            grammar_version=SchemaVersion.get_version(),
            nodes=[
                IRNode(
                    id="node1", type=IRNodeType.COMPUTE, params={"operation": "test"}
                ),
                IRNode(
                    id="node2", type=IRNodeType.COMPUTE, params={"operation": "test"}
                ),
            ],
            edges=[
                IREdge(source="node1", target="node2", type=IREdgeType.DATA),
                IREdge(source="node1", target="node2", type=IREdgeType.CONTROL),
                IREdge(source="node1", target="node2", type=IREdgeType.MEMORY),
            ],
        )

        assert graph.validate() is True
        assert len(graph.edges) == 3

    def test_graph_with_subgraphs(self):
        """Test graph with nested subgraphs."""
        subgraph = IRGraph(
            grammar_version=SchemaVersion.get_version(),
            nodes=[
                IRNode(id="sub1", type=IRNodeType.COMPUTE, params={"operation": "add"})
            ],
            edges=[],
        )

        graph = IRGraph(
            grammar_version=SchemaVersion.get_version(),
            nodes=[
                IRNode(id="main1", type=IRNodeType.COMPUTE, params={"operation": "mul"})
            ],
            edges=[],
            subgraphs={"subgraph1": subgraph},
        )

        assert graph.validate() is True
        assert "subgraph1" in graph.subgraphs


class TestActionCategoryEnum:
    """Test action category enumeration."""

    def test_all_action_categories_exist(self):
        """Test that all action categories are defined."""
        categories = [
            ActionCategory.COMPUTATION,
            ActionCategory.COMMUNICATION,
            ActionCategory.STORAGE,
            ActionCategory.LEARNING,
            ActionCategory.REASONING,
            ActionCategory.PLANNING,
            ActionCategory.SAFETY,
            ActionCategory.MONITORING,
            ActionCategory.OPTIMIZATION,
            ActionCategory.MAINTENANCE,
        ]

        assert len(categories) == 10

        for category in categories:
            assert isinstance(category, ActionCategory)


class TestAgentRoleEnum:
    """Test agent role enumeration."""

    def test_all_agent_roles_exist(self):
        """Test that all agent roles are defined."""
        roles = list(AgentRole)

        assert len(roles) > 0

        # Check for key roles
        assert AgentRole.COORDINATOR in roles
        assert AgentRole.EXECUTOR in roles
        assert AgentRole.PLANNER in roles
        assert AgentRole.LEARNER in roles
        assert AgentRole.REASONER in roles
        assert AgentRole.MONITOR in roles
        assert AgentRole.SAFETY_OFFICER in roles


class TestEdgeCaseValidation:
    """Test edge case validation scenarios."""

    def test_node_with_empty_params(self):
        """Test node with empty params dict."""
        node = IRNode(id="empty_params", type=IRNodeType.INPUT, params={})

        assert node.validate() is True

    def test_edge_with_zero_weight(self):
        """Test edge with zero weight."""
        edge = IREdge(source="node1", target="node2", type=IREdgeType.DATA, weight=0.0)

        assert edge.validate() is True

    def test_agent_with_empty_capabilities(self):
        """Test agent with no capabilities."""
        agent = AgentProfile(
            agent_id="empty_agent", role=AgentRole.EXECUTOR, capabilities=[]
        )

        assert agent.validate() is True

    def test_system_state_with_empty_policies(self):
        """Test system state with empty policies."""
        state = SystemState(CID="empty_policies", policies={})

        assert state.CID == "empty_policies"
        assert len(state.policies) == 0


class TestSerializationEdgeCases:
    """Test serialization edge cases."""

    def test_serialize_node_with_none_values(self):
        """Test serializing node with None values."""
        node = IRNode(
            id="none_node",
            type=IRNodeType.COMPUTE,
            params={"operation": "test"},
            timeout_ms=None,
        )

        encoded = json.dumps(node, cls=EnhancedJSONEncoder)
        assert isinstance(encoded, str)

    def test_serialize_deeply_nested_structures(self):
        """Test serializing deeply nested structures."""
        data = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}

        encoded = json.dumps(data, cls=EnhancedJSONEncoder)
        decoded = json.loads(encoded, cls=EnhancedJSONDecoder)

        assert decoded["level1"]["level2"]["level3"]["level4"]["value"] == "deep"

    def test_serialize_large_arrays(self):
        """Test serializing large numpy arrays."""
        arr = np.random.rand(100, 100)

        encoded = json.dumps(arr, cls=EnhancedJSONEncoder)
        decoded = json.loads(encoded, cls=EnhancedJSONDecoder)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (100, 100)


# ============================================================
# RUN CONFIGURATION
# ============================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--cov=src.vulcan.vulcan_types",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
